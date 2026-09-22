//! Test 5: Float32 Precision.
//!
//! Runs three identity-preserving operations and measures the max absolute
//! error between input and output:
//!   1. A × I ≟ A            (256×256 matmul)
//!   2. (A + B) − B ≟ A      (512×512 vector add/sub)
//!   3. A × 2 ÷ 2 ≟ A        (512×512 scalar mul/div)

use ash::vk;

use crate::compute::{allocate_descriptor_set, bind_storage_buffers, create_descriptor_pool};
use crate::memory::{download_f32, upload_f32, GpuBuffer};
use crate::vulkan::VulkanContext;

use super::{
    f32_bytes, lcg_vec_scaled, record_dispatch_1d, record_dispatch_2d, BenchmarkResult,
    ProgressCallback, PushCount1f, PushMatmul, TestStatus,
};

pub fn run(ctx: &VulkanContext, progress: &mut dyn ProgressCallback) -> BenchmarkResult {
    progress.begin(None, "Running precision checks...");

    let mut max_error: f64 = 0.0;
    let mut details_lines: Vec<String> = Vec::new();

    match check_identity_matmul(ctx) {
        Ok(err) => {
            max_error = max_error.max(err);
            details_lines.push(format!("A×I=A: max_err={:.2e}", err));
        }
        Err(e) => {
            progress.end();
            return crash_result(format!("A×I check failed: {}", e));
        }
    }

    match check_add_sub(ctx) {
        Ok(err) => {
            max_error = max_error.max(err);
            details_lines.push(format!("(A+B)-B≈A: max_err={:.2e}", err));
        }
        Err(e) => {
            progress.end();
            return crash_result(format!("Add/sub check failed: {}", e));
        }
    }

    match check_scale_mul_div(ctx) {
        Ok(err) => {
            max_error = max_error.max(err);
            details_lines.push(format!("A*2/2≈A: max_err={:.2e}", err));
        }
        Err(e) => {
            progress.end();
            return crash_result(format!("Scale check failed: {}", e));
        }
    }

    progress.end();

    for line in &details_lines {
        progress.log(line);
    }

    let score: f64 = if max_error < 1e-6 {
        100.0
    } else if max_error < 1e-5 {
        90.0
    } else if max_error < 1e-4 {
        75.0
    } else if max_error < 1e-3 {
        50.0
    } else if max_error < 1e-1 {
        25.0
    } else {
        5.0
    };

    let status = if score >= 75.0 {
        TestStatus::Passed
    } else if score >= 50.0 {
        TestStatus::Warning
    } else {
        TestStatus::Failed
    };

    let details = format!("Max precision error: {:.2e}", max_error);

    BenchmarkResult {
        name: "Float32 Precision".to_string(),
        status,
        score,
        details,
        metric_value: format!("{:.2e}", max_error),
    }
}

fn crash_result(msg: String) -> BenchmarkResult {
    BenchmarkResult {
        name: "Float32 Precision".to_string(),
        status: TestStatus::Error(msg.clone()),
        score: 0.0,
        details: format!("Precision test gặp sự cố: {}", msg),
        metric_value: "N/A".to_string(),
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 1: A × I = A
// ═══════════════════════════════════════════════════════════════════════════════

fn check_identity_matmul(ctx: &VulkanContext) -> anyhow::Result<f64> {
    const N: usize = 256;
    let av = lcg_vec_scaled(N * N, 0x1234_5678, 10.0); // uniform [-10, 10)
    let mut iv = vec![0.0f32; N * N];
    for i in 0..N {
        iv[i * N + i] = 1.0;
    }

    let bytes = f32_bytes(N * N);
    let a_buf = GpuBuffer::device_local(ctx, bytes, vk::BufferUsageFlags::STORAGE_BUFFER)?;
    let i_buf = GpuBuffer::device_local(ctx, bytes, vk::BufferUsageFlags::STORAGE_BUFFER)?;
    let c_buf = GpuBuffer::device_local(ctx, bytes, vk::BufferUsageFlags::STORAGE_BUFFER)?;

    upload_f32(ctx, &a_buf, &av)?;
    upload_f32(ctx, &i_buf, &iv)?;

    let pool = create_descriptor_pool(&ctx.device, 1, 3)?;
    let set = allocate_descriptor_set(&ctx.device, pool, ctx.shaders.matmul.descriptor_set_layout)?;
    bind_storage_buffers(
        &ctx.device,
        set,
        &[
            (0, a_buf.buffer, bytes),
            (1, i_buf.buffer, bytes),
            (2, c_buf.buffer, bytes),
        ],
    );

    let push = PushMatmul { n: N as u32 };
    let push_bytes = bytemuck::bytes_of(&push).to_vec();
    ctx.execute_commands(|cb| {
        record_dispatch_2d(ctx, cb, &ctx.shaders.matmul, set, &push_bytes, N as u32, 16);
    })?;

    let cs = download_f32(ctx, &c_buf, N * N)?;

    unsafe { ctx.device.destroy_descriptor_pool(pool, None) };
    a_buf.destroy(ctx);
    i_buf.destroy(ctx);
    c_buf.destroy(ctx);

    let mut max_err: f64 = 0.0;
    for i in 0..av.len() {
        let err = (av[i] as f64 - cs[i] as f64).abs();
        if err > max_err {
            max_err = err;
        }
    }
    Ok(max_err)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 2: (A + B) − B = A
// ═══════════════════════════════════════════════════════════════════════════════

fn check_add_sub(ctx: &VulkanContext) -> anyhow::Result<f64> {
    const N: usize = 512;
    let count = N * N;
    let av = lcg_vec_scaled(count, 0xAAAA_BBBB, 100.0); // uniform [-100, 100)
    let bv = lcg_vec_scaled(count, 0xBBBB_AAAA, 100.0);

    let bytes = f32_bytes(count);
    let a_buf = GpuBuffer::device_local(ctx, bytes, vk::BufferUsageFlags::STORAGE_BUFFER)?;
    let b_buf = GpuBuffer::device_local(ctx, bytes, vk::BufferUsageFlags::STORAGE_BUFFER)?;
    let c_buf = GpuBuffer::device_local(ctx, bytes, vk::BufferUsageFlags::STORAGE_BUFFER)?;
    let d_buf = GpuBuffer::device_local(ctx, bytes, vk::BufferUsageFlags::STORAGE_BUFFER)?;

    upload_f32(ctx, &a_buf, &av)?;
    upload_f32(ctx, &b_buf, &bv)?;

    let pool = create_descriptor_pool(&ctx.device, 2, 6)?;
    let add_set = allocate_descriptor_set(
        &ctx.device,
        pool,
        ctx.shaders.vector_add.descriptor_set_layout,
    )?;
    let sub_set = allocate_descriptor_set(
        &ctx.device,
        pool,
        ctx.shaders.vector_add.descriptor_set_layout,
    )?;

    // add_set: (A, B, C) ⇒ C = A + B * 1.0
    bind_storage_buffers(
        &ctx.device,
        add_set,
        &[
            (0, a_buf.buffer, bytes),
            (1, b_buf.buffer, bytes),
            (2, c_buf.buffer, bytes),
        ],
    );
    // sub_set: (C, B, D) ⇒ D = C + B * (-1.0)
    bind_storage_buffers(
        &ctx.device,
        sub_set,
        &[
            (0, c_buf.buffer, bytes),
            (1, b_buf.buffer, bytes),
            (2, d_buf.buffer, bytes),
        ],
    );

    let push_add = PushCount1f {
        count: count as u32,
        value: 1.0,
    };
    let push_sub = PushCount1f {
        count: count as u32,
        value: -1.0,
    };
    let push_add_bytes = bytemuck::bytes_of(&push_add).to_vec();
    let push_sub_bytes = bytemuck::bytes_of(&push_sub).to_vec();

    ctx.execute_commands(|cb| {
        record_dispatch_1d(
            ctx,
            cb,
            &ctx.shaders.vector_add,
            add_set,
            &push_add_bytes,
            count as u32,
            256,
        );
        super::cmd_global_compute_barrier(ctx, cb);
        record_dispatch_1d(
            ctx,
            cb,
            &ctx.shaders.vector_add,
            sub_set,
            &push_sub_bytes,
            count as u32,
            256,
        );
    })?;

    let ds = download_f32(ctx, &d_buf, count)?;

    unsafe { ctx.device.destroy_descriptor_pool(pool, None) };
    a_buf.destroy(ctx);
    b_buf.destroy(ctx);
    c_buf.destroy(ctx);
    d_buf.destroy(ctx);

    let mut max_err: f64 = 0.0;
    for i in 0..count {
        let err = (av[i] as f64 - ds[i] as f64).abs();
        if err > max_err {
            max_err = err;
        }
    }
    Ok(max_err)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 3: A × 2 ÷ 2 = A
// ═══════════════════════════════════════════════════════════════════════════════

fn check_scale_mul_div(ctx: &VulkanContext) -> anyhow::Result<f64> {
    const N: usize = 512;
    let count = N * N;
    let av = lcg_vec_scaled(count, 0xCCCC_DDDD, 50.0); // uniform [-50, 50)

    let bytes = f32_bytes(count);
    let a_buf = GpuBuffer::device_local(ctx, bytes, vk::BufferUsageFlags::STORAGE_BUFFER)?;
    let c_buf = GpuBuffer::device_local(ctx, bytes, vk::BufferUsageFlags::STORAGE_BUFFER)?;
    let d_buf = GpuBuffer::device_local(ctx, bytes, vk::BufferUsageFlags::STORAGE_BUFFER)?;

    upload_f32(ctx, &a_buf, &av)?;

    let pool = create_descriptor_pool(&ctx.device, 2, 4)?;
    let mul_set =
        allocate_descriptor_set(&ctx.device, pool, ctx.shaders.scale.descriptor_set_layout)?;
    let div_set =
        allocate_descriptor_set(&ctx.device, pool, ctx.shaders.scale.descriptor_set_layout)?;

    // mul_set: (A, C) ⇒ C = A * 2.0
    bind_storage_buffers(
        &ctx.device,
        mul_set,
        &[(0, a_buf.buffer, bytes), (1, c_buf.buffer, bytes)],
    );
    // div_set: (C, D) ⇒ D = C * 0.5
    bind_storage_buffers(
        &ctx.device,
        div_set,
        &[(0, c_buf.buffer, bytes), (1, d_buf.buffer, bytes)],
    );

    let push_mul = PushCount1f {
        count: count as u32,
        value: 2.0,
    };
    let push_div = PushCount1f {
        count: count as u32,
        value: 0.5,
    };
    let push_mul_bytes = bytemuck::bytes_of(&push_mul).to_vec();
    let push_div_bytes = bytemuck::bytes_of(&push_div).to_vec();

    ctx.execute_commands(|cb| {
        record_dispatch_1d(
            ctx,
            cb,
            &ctx.shaders.scale,
            mul_set,
            &push_mul_bytes,
            count as u32,
            256,
        );
        super::cmd_global_compute_barrier(ctx, cb);
        record_dispatch_1d(
            ctx,
            cb,
            &ctx.shaders.scale,
            div_set,
            &push_div_bytes,
            count as u32,
            256,
        );
    })?;

    let ds = download_f32(ctx, &d_buf, count)?;

    unsafe { ctx.device.destroy_descriptor_pool(pool, None) };
    a_buf.destroy(ctx);
    c_buf.destroy(ctx);
    d_buf.destroy(ctx);

    let mut max_err: f64 = 0.0;
    for i in 0..count {
        let err = (av[i] as f64 - ds[i] as f64).abs();
        if err > max_err {
            max_err = err;
        }
    }
    Ok(max_err)
}
