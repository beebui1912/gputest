//! GPU benchmarks — 5 tests port of `sample.rs` to raw Vulkan.
//!
//! Each benchmark exposes a `run(...)` function that returns a [`BenchmarkResult`].
//! Progress reporting is done through the [`ProgressCallback`] trait so the UI
//! layer (main.rs) can drive `indicatif` progress bars without the core crate
//! depending on it.

pub mod bandwidth;
pub mod matmul;
pub mod precision;
pub mod stability;
pub mod vram;

use ash::vk;

use crate::compute::ComputePipeline;
use crate::vulkan::VulkanContext;

// ═══════════════════════════════════════════════════════════════════════════════
// Result Types
// ═══════════════════════════════════════════════════════════════════════════════

/// Result of a single benchmark.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub name: String,
    pub status: TestStatus,
    /// 0.0 – 100.0
    pub score: f64,
    pub details: String,
    pub metric_value: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TestStatus {
    Passed,
    Warning,
    Failed,
    Error(String),
}

// ═══════════════════════════════════════════════════════════════════════════════
// Progress Reporting
// ═══════════════════════════════════════════════════════════════════════════════

/// UI progress reporting hook. Every method has a default no-op impl so the
/// UI layer only overrides what it cares about.
pub trait ProgressCallback {
    /// Begin a new progress-tracked stage with an optional total.
    /// `total = None` signals an indeterminate/spinner stage.
    fn begin(&mut self, _total: Option<u64>, _message: &str) {}

    /// Advance current position (only meaningful when `total.is_some()`).
    fn set_position(&mut self, _pos: u64) {}

    /// Update the trailing message on the current stage.
    fn set_message(&mut self, _message: &str) {}

    /// Persist an informational line above the current progress display.
    fn log(&mut self, _message: &str) {}

    /// Persist a warning line above the current progress display.
    fn warn(&mut self, _message: &str) {}

    /// End the current stage.
    fn end(&mut self) {}
}

/// No-op progress callback — useful for tests / non-interactive use.
pub struct NullProgress;
impl ProgressCallback for NullProgress {}

// ═══════════════════════════════════════════════════════════════════════════════
// Shared Constants
// ═══════════════════════════════════════════════════════════════════════════════

/// f32 patterns that are exactly representable in IEEE-754. Rotating through
/// them across chunks ensures neighbouring memory carries different bit
/// patterns, giving the VRAM memtest a chance to detect stuck bits.
pub const VRAM_PATTERNS: [f32; 8] = [1.0, -2.0, 0.5, 4.0, -0.75, 3.0, 0.25, -1.5];

// ═══════════════════════════════════════════════════════════════════════════════
// Shared Data Generation
// ═══════════════════════════════════════════════════════════════════════════════

/// Deterministic LCG-based pseudo-random f32 vector in `[-1, 1)`.
/// Used to seed benchmark inputs so results can be cross-validated against
/// a CPU f64 reference.
pub fn lcg_vec(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((s >> 33) as f64 / 1073741824.0) - 1.0) as f32
        })
        .collect()
}

/// Scaled LCG vector: values uniformly in `[-scale, scale)`.
pub fn lcg_vec_scaled(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    lcg_vec(n, seed).into_iter().map(|v| v * scale).collect()
}

// ═══════════════════════════════════════════════════════════════════════════════
// Push Constant Layouts
// ═══════════════════════════════════════════════════════════════════════════════

/// Push constants for `fill`, `verify`, `vector_add`, `scale` shaders.
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct PushCount1f {
    pub count: u32,
    pub value: f32,
}

/// Push constants for `matmul` shader.
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct PushMatmul {
    pub n: u32,
}

// ═══════════════════════════════════════════════════════════════════════════════
// Dispatch Helpers
// ═══════════════════════════════════════════════════════════════════════════════

/// Record a 1D compute dispatch (bind pipeline+descriptor set, push constants, dispatch).
pub fn record_dispatch_1d(
    ctx: &VulkanContext,
    cb: vk::CommandBuffer,
    pipeline: &ComputePipeline,
    descriptor_set: vk::DescriptorSet,
    push_bytes: &[u8],
    element_count: u32,
    local_size_x: u32,
) {
    let groups = (element_count + local_size_x - 1) / local_size_x;
    unsafe {
        ctx.device
            .cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipeline.pipeline);
        ctx.device.cmd_bind_descriptor_sets(
            cb,
            vk::PipelineBindPoint::COMPUTE,
            pipeline.pipeline_layout,
            0,
            &[descriptor_set],
            &[],
        );
        if !push_bytes.is_empty() {
            ctx.device.cmd_push_constants(
                cb,
                pipeline.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                push_bytes,
            );
        }
        ctx.device.cmd_dispatch(cb, groups, 1, 1);
    }
}

/// Record a 2D compute dispatch for square shaders (matmul).
pub fn record_dispatch_2d(
    ctx: &VulkanContext,
    cb: vk::CommandBuffer,
    pipeline: &ComputePipeline,
    descriptor_set: vk::DescriptorSet,
    push_bytes: &[u8],
    n: u32,
    local_size: u32,
) {
    let groups = (n + local_size - 1) / local_size;
    unsafe {
        ctx.device
            .cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipeline.pipeline);
        ctx.device.cmd_bind_descriptor_sets(
            cb,
            vk::PipelineBindPoint::COMPUTE,
            pipeline.pipeline_layout,
            0,
            &[descriptor_set],
            &[],
        );
        if !push_bytes.is_empty() {
            ctx.device.cmd_push_constants(
                cb,
                pipeline.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                push_bytes,
            );
        }
        ctx.device.cmd_dispatch(cb, groups, groups, 1);
    }
}

/// Insert a global memory barrier between two compute passes.
/// Ensures all previous shader writes are visible to subsequent shader reads.
pub fn cmd_global_compute_barrier(ctx: &VulkanContext, cb: vk::CommandBuffer) {
    let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);

    unsafe {
        ctx.device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[barrier],
            &[],
            &[],
        );
    }
}

/// Byte size of a slice of `f32`s as `vk::DeviceSize`.
#[inline]
pub fn f32_bytes(count: usize) -> vk::DeviceSize {
    (count * std::mem::size_of::<f32>()) as vk::DeviceSize
}
