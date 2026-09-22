//! Compute pipeline and shader management.
//!
//! Embeds GLSL compute shader sources as string constants, compiles them to
//! SPIR-V at runtime via `shaderc`, and manages Vulkan compute pipelines.

use ash::vk;

// ═══════════════════════════════════════════════════════════════════════════════
// GLSL Shader Sources (compiled to SPIR-V at runtime)
// ═══════════════════════════════════════════════════════════════════════════════

/// Fill buffer with constant value.
/// Bindings: 0 = data (storage buffer, write)
/// Push constants: { uint count; float value; }
pub const FILL_GLSL: &str = r#"
#version 450
layout(local_size_x = 256) in;

layout(set = 0, binding = 0) buffer Data { float data[]; };

layout(push_constant) uniform PC {
    uint count;
    float value;
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx < count) {
        data[idx] = value;
    }
}
"#;

/// Verify buffer contents against expected value.
/// Outputs number of mismatched elements via atomicAdd.
/// Bindings: 0 = data (read), 1 = result (read/write, contains uint error_count)
/// Push constants: { uint count; float expected; }
pub const VERIFY_GLSL: &str = r#"
#version 450
layout(local_size_x = 256) in;

layout(set = 0, binding = 0) readonly buffer Data { float data[]; };
layout(set = 0, binding = 1) buffer Result { uint error_count; };

layout(push_constant) uniform PC {
    uint count;
    float expected;
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx < count && data[idx] != expected) {
        atomicAdd(error_count, 1u);
    }
}
"#;

/// Tiled matrix multiply: C = A × B (NxN, row-major).
/// Uses 16×16 shared memory tiles for reasonable throughput.
/// Bindings: 0 = A (read), 1 = B (read), 2 = C (write)
/// Push constants: { uint N; }
pub const MATMUL_GLSL: &str = r#"
#version 450
layout(local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0) readonly buffer MatA { float a[]; };
layout(set = 0, binding = 1) readonly buffer MatB { float b[]; };
layout(set = 0, binding = 2) writeonly buffer MatC { float c[]; };

layout(push_constant) uniform PC {
    uint N;
};

shared float tileA[16][16];
shared float tileB[16][16];

void main() {
    uint row = gl_GlobalInvocationID.y;
    uint col = gl_GlobalInvocationID.x;
    uint lr  = gl_LocalInvocationID.y;
    uint lc  = gl_LocalInvocationID.x;

    float sum = 0.0;
    uint tiles = (N + 15u) / 16u;

    for (uint t = 0u; t < tiles; t++) {
        uint aCol = t * 16u + lc;
        uint bRow = t * 16u + lr;

        tileA[lr][lc] = (row < N && aCol < N) ? a[row * N + aCol] : 0.0;
        tileB[lr][lc] = (bRow < N && col  < N) ? b[bRow * N + col] : 0.0;

        barrier();

        for (uint k = 0u; k < 16u; k++) {
            sum += tileA[lr][k] * tileB[k][lc];
        }

        barrier();
    }

    if (row < N && col < N) {
        c[row * N + col] = sum;
    }
}
"#;

/// Element-wise: C[i] = A[i] + B[i] * b_scale.
/// b_scale = 1.0 → add, b_scale = -1.0 → subtract.
/// Bindings: 0 = A (read), 1 = B (read), 2 = C (write)
/// Push constants: { uint count; float b_scale; }
pub const VECTOR_ADD_GLSL: &str = r#"
#version 450
layout(local_size_x = 256) in;

layout(set = 0, binding = 0) readonly buffer BufA { float a[]; };
layout(set = 0, binding = 1) readonly buffer BufB { float b[]; };
layout(set = 0, binding = 2) writeonly buffer BufC { float c[]; };

layout(push_constant) uniform PC {
    uint count;
    float b_scale;
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx < count) {
        c[idx] = a[idx] + b[idx] * b_scale;
    }
}
"#;

/// Element-wise: C[i] = A[i] * scalar.
/// Bindings: 0 = A (read), 1 = C (write)
/// Push constants: { uint count; float scalar; }
pub const SCALE_GLSL: &str = r#"
#version 450
layout(local_size_x = 256) in;

layout(set = 0, binding = 0) readonly buffer BufA { float a[]; };
layout(set = 0, binding = 1) writeonly buffer BufC { float c[]; };

layout(push_constant) uniform PC {
    uint count;
    float scalar;
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx < count) {
        c[idx] = a[idx] * scalar;
    }
}
"#;

// ═══════════════════════════════════════════════════════════════════════════════
// Shader Compilation
// ═══════════════════════════════════════════════════════════════════════════════

/// Compile GLSL compute shader source to SPIR-V word array.
fn compile_glsl(source: &str, name: &str) -> anyhow::Result<Vec<u32>> {
    let compiler = shaderc::Compiler::new()
        .ok_or_else(|| anyhow::anyhow!("Failed to create shaderc compiler"))?;

    let artifact = compiler
        .compile_into_spirv(source, shaderc::ShaderKind::Compute, name, "main", None)
        .map_err(|e| anyhow::anyhow!("Shader '{}' compilation failed: {}", name, e))?;

    if artifact.get_num_warnings() > 0 {
        eprintln!(
            "Shader '{}' warnings: {}",
            name,
            artifact.get_warning_messages()
        );
    }

    Ok(artifact.as_binary().to_vec())
}

// ═══════════════════════════════════════════════════════════════════════════════
// ComputePipeline
// ═══════════════════════════════════════════════════════════════════════════════

/// A compiled compute pipeline with its layout and descriptor set layout.
pub struct ComputePipeline {
    pub pipeline: vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    shader_module: vk::ShaderModule,
}

impl ComputePipeline {
    /// Create a compute pipeline for a shader with `num_storage_buffers` bindings
    /// and `push_constant_size` bytes of push constants.
    pub fn new(
        device: &ash::Device,
        spirv: &[u32],
        num_storage_buffers: u32,
        push_constant_size: u32,
    ) -> anyhow::Result<Self> {
        // Shader module
        let module_info = vk::ShaderModuleCreateInfo::default().code(spirv);
        let shader_module = unsafe { device.create_shader_module(&module_info, None)? };

        // Descriptor set layout: N storage buffers at binding 0..N-1
        let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..num_storage_buffers)
            .map(|i| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();

        let layout_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let descriptor_set_layout =
            unsafe { device.create_descriptor_set_layout(&layout_info, None)? };

        // Pipeline layout
        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(push_constant_size);

        let mut pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&descriptor_set_layout));

        if push_constant_size > 0 {
            pipeline_layout_info =
                pipeline_layout_info.push_constant_ranges(std::slice::from_ref(&push_constant_range));
        }

        let pipeline_layout =
            unsafe { device.create_pipeline_layout(&pipeline_layout_info, None)? };

        // Compute pipeline
        let stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(c"main");

        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage_info)
            .layout(pipeline_layout);

        let pipeline = unsafe {
            device
                .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|(_pipelines, e)| anyhow::anyhow!("Pipeline creation failed: {:?}", e))?[0]
        };

        Ok(Self {
            pipeline,
            pipeline_layout,
            descriptor_set_layout,
            shader_module,
        })
    }

    pub fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            device.destroy_shader_module(self.shader_module, None);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Shaders — all precompiled pipelines
// ═══════════════════════════════════════════════════════════════════════════════

/// All compute pipelines used by the benchmark suite.
pub struct Shaders {
    pub fill: ComputePipeline,       // 1 buffer, 8B push
    pub verify: ComputePipeline,     // 2 buffers, 8B push
    pub matmul: ComputePipeline,     // 3 buffers, 4B push
    pub vector_add: ComputePipeline, // 3 buffers, 8B push
    pub scale: ComputePipeline,      // 2 buffers, 8B push
}

impl Shaders {
    /// Compile all shaders and create pipelines.
    pub fn new(device: &ash::Device) -> anyhow::Result<Self> {
        let fill_spv = compile_glsl(FILL_GLSL, "fill.comp")?;
        let verify_spv = compile_glsl(VERIFY_GLSL, "verify.comp")?;
        let matmul_spv = compile_glsl(MATMUL_GLSL, "matmul.comp")?;
        let vector_add_spv = compile_glsl(VECTOR_ADD_GLSL, "vector_add.comp")?;
        let scale_spv = compile_glsl(SCALE_GLSL, "scale.comp")?;

        Ok(Self {
            fill: ComputePipeline::new(device, &fill_spv, 1, 8)?,
            verify: ComputePipeline::new(device, &verify_spv, 2, 8)?,
            matmul: ComputePipeline::new(device, &matmul_spv, 3, 4)?,
            vector_add: ComputePipeline::new(device, &vector_add_spv, 3, 8)?,
            scale: ComputePipeline::new(device, &scale_spv, 2, 8)?,
        })
    }

    pub fn destroy(&self, device: &ash::Device) {
        self.fill.destroy(device);
        self.verify.destroy(device);
        self.matmul.destroy(device);
        self.vector_add.destroy(device);
        self.scale.destroy(device);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Descriptor Set Helpers
// ═══════════════════════════════════════════════════════════════════════════════

/// Create a descriptor pool large enough for `max_sets` sets,
/// each with up to `max_storage_buffers` storage buffer bindings.
pub fn create_descriptor_pool(
    device: &ash::Device,
    max_sets: u32,
    max_storage_buffers: u32,
) -> anyhow::Result<vk::DescriptorPool> {
    let pool_size = vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(max_storage_buffers);

    let pool_info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(max_sets)
        .pool_sizes(std::slice::from_ref(&pool_size));

    Ok(unsafe { device.create_descriptor_pool(&pool_info, None)? })
}

/// Allocate a single descriptor set from a pool.
pub fn allocate_descriptor_set(
    device: &ash::Device,
    pool: vk::DescriptorPool,
    layout: vk::DescriptorSetLayout,
) -> anyhow::Result<vk::DescriptorSet> {
    let alloc_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool)
        .set_layouts(std::slice::from_ref(&layout));

    Ok(unsafe { device.allocate_descriptor_sets(&alloc_info)? }[0])
}

/// Bind storage buffers to a descriptor set.
/// `bindings` is a list of (binding_index, buffer, size).
pub fn bind_storage_buffers(
    device: &ash::Device,
    descriptor_set: vk::DescriptorSet,
    bindings: &[(u32, vk::Buffer, vk::DeviceSize)],
) {
    let buffer_infos: Vec<vk::DescriptorBufferInfo> = bindings
        .iter()
        .map(|&(_, buf, size)| {
            vk::DescriptorBufferInfo::default()
                .buffer(buf)
                .offset(0)
                .range(size)
        })
        .collect();

    let writes: Vec<vk::WriteDescriptorSet> = bindings
        .iter()
        .enumerate()
        .map(|(i, &(binding, _, _))| {
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(binding)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&buffer_infos[i]))
        })
        .collect();

    unsafe {
        device.update_descriptor_sets(&writes, &[]);
    }
}
