//! Test 3: Memory Bandwidth.
//!
//! Measures VRAM read/write bandwidth via a chained `C = A + B` operation
//! (each op moves 3 × buffer_size bytes: read A, read B, write C). Chains 8
//! ops with ping-pong buffers to keep the memory bus saturated, verifies the
//! result equals the mathematically-expected constant, and reports GB/s.

use std::time::Instant;

use ash::vk;

use crate::compute::{allocate_descriptor_set, bind_storage_buffers, create_descriptor_pool};
use crate::memory::{read_f32_at, read_u32_at, GpuBuffer};
use crate::vulkan::VulkanContext;

use super::{
    cmd_global_compute_barrier, f32_bytes, record_dispatch_1d, BenchmarkResult, ProgressCallback,
    PushCount1f, TestStatus,
};

const OPS: usize = 8;
const RUNS: usize = 4;

pub fn run(
    ctx: &VulkanContext,
    dedicated_vram_mb: Option<u64>,
    progress: &mut dyn ProgressCallback,
) -> BenchmarkResult {
    // Card ≥ 4 GB: 256 MB buffer to saturate the bus. Nhỏ hơn: 64 MB.
    let size: usize = if dedicated_vram_mb.unwrap_or(2048) >= 4096 {
        8192
    } else {
        4096
    };
    let count = size * size;
    let bytes = f32_bytes(count);

    progress.begin(
        None,
        &format!(
            "Buffer {}×{} f32 = {:.0} MB, chain {} ops × {} runs...",
            size,
            size,
            bytes as f64 / 1_048_576.0,
            OPS,
            RUNS
        ),
    );

    let outcome = run_inner(ctx, size, count, bytes);
    progress.end();

    match outcome {
        Ok(BandwidthResult { best_bw, worst_bw }) => {
            let spread = if best_bw > 0.0 {
                (best_bw - worst_bw) / best_bw * 100.0
            } else {
                0.0
            };
            let unstable = spread > 20.0;

            let mut score: f64 = if best_bw >= 400.0 {
                100.0
            } else if best_bw >= 200.0 {
                85.0
            } else if best_bw >= 100.0 {
                70.0
            } else if best_bw >= 50.0 {
                55.0
            } else if best_bw >= 10.0 {
                35.0
            } else {
                15.0
            };
            if unstable {
                score = score.min(65.0);
            }

            let status = if score >= 70.0 && !unstable {
                TestStatus::Passed
            } else if score >= 40.0 {
                TestStatus::Warning
            } else {
                TestStatus::Failed
            };

            let details = format!(
                "Bandwidth: {:.0} GB/s (best of {} runs, dao động {:.0}%){}",
                best_bw,
                RUNS,
                spread,
                if unstable {
                    " — bất ổn giữa các lần chạy"
                } else {
                    ""
                }
            );

            BenchmarkResult {
                name: "Memory Bandwidth".to_string(),
                status,
                score,
                details,
                metric_value: format!("{:.1} GB/s", best_bw),
            }
        }
        Err(BandwidthError::Correctness(msg)) => BenchmarkResult {
            name: "Memory Bandwidth".to_string(),
            status: TestStatus::Failed,
            score: 10.0,
            details: msg,
            metric_value: "INVALID".to_string(),
        },
        Err(BandwidthError::Vulkan(msg)) => BenchmarkResult {
            name: "Memory Bandwidth".to_string(),
            status: TestStatus::Error(msg.clone()),
            score: 0.0,
            details: format!("Không thể đo bandwidth: {}", msg),
            metric_value: "N/A".to_string(),
        },
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Internal
// ═══════════════════════════════════════════════════════════════════════════════

struct BandwidthResult {
    best_bw: f64,
    worst_bw: f64,
}

enum BandwidthError {
    Correctness(String),
    Vulkan(String),
}

fn run_inner(
    ctx: &VulkanContext,
    _size: usize,
    count: usize,
    bytes: vk::DeviceSize,
) -> Result<BandwidthResult, BandwidthError> {
    // Allocate 3 buffers + result buffer.
    let alloc = |usage: vk::BufferUsageFlags, sz: vk::DeviceSize| {
        GpuBuffer::device_local(ctx, sz, usage).map_err(|e| BandwidthError::Vulkan(e.to_string()))
    };

    let a_buf = alloc(vk::BufferUsageFlags::STORAGE_BUFFER, bytes)?;
    let b_buf = alloc(vk::BufferUsageFlags::STORAGE_BUFFER, bytes)?;
    let c_buf = alloc(vk::BufferUsageFlags::STORAGE_BUFFER, bytes)?;
    let result_buf = alloc(vk::BufferUsageFlags::STORAGE_BUFFER, 4)?;

    // Descriptor pool: fill_set + verify_set + vadd_set_ac + vadd_set_ca.
    let pool = create_descriptor_pool(&ctx.device, 4, 3 + 3 + 2 + 1)
        .map_err(|e| BandwidthError::Vulkan(e.to_string()))?;

    let fill_set = allocate_descriptor_set(
        &ctx.device,
        pool,
        ctx.shaders.fill.descriptor_set_layout,
    )
    .map_err(|e| BandwidthError::Vulkan(e.to_string()))?;
    let verify_set = allocate_descriptor_set(
        &ctx.device,
        pool,
        ctx.shaders.verify.descriptor_set_layout,
    )
    .map_err(|e| BandwidthError::Vulkan(e.to_string()))?;

    let set_ac = allocate_descriptor_set(
        &ctx.device,
        pool,
        ctx.shaders.vector_add.descriptor_set_layout,
    )
    .map_err(|e| BandwidthError::Vulkan(e.to_string()))?; // reads A, writes C
    let set_ca = allocate_descriptor_set(
        &ctx.device,
        pool,
        ctx.shaders.vector_add.descriptor_set_layout,
    )
    .map_err(|e| BandwidthError::Vulkan(e.to_string()))?; // reads C, writes A

    bind_storage_buffers(
        &ctx.device,
        set_ac,
        &[
            (0, a_buf.buffer, bytes),
            (1, b_buf.buffer, bytes),
            (2, c_buf.buffer, bytes),
        ],
    );
    bind_storage_buffers(
        &ctx.device,
        set_ca,
        &[
            (0, c_buf.buffer, bytes),
            (1, b_buf.buffer, bytes),
            (2, a_buf.buffer, bytes),
        ],
    );

    let cleanup = |ctx: &VulkanContext| {
        unsafe { ctx.device.destroy_descriptor_pool(pool, None) };
        a_buf.destroy(ctx);
        b_buf.destroy(ctx);
        c_buf.destroy(ctx);
        result_buf.destroy(ctx);
    };

    // Fill A = 1.0, B = 0.25.
    if let Err(e) = fill_buffer(ctx, &a_buf, count, 1.0, fill_set) {
        cleanup(ctx);
        return Err(BandwidthError::Vulkan(e.to_string()));
    }
    if let Err(e) = fill_buffer(ctx, &b_buf, count, 0.25, fill_set) {
        cleanup(ctx);
        return Err(BandwidthError::Vulkan(e.to_string()));
    }

    // After OPS chained additions of B (=0.25) onto A (=1.0):
    // final value = 1.0 + 0.25 × OPS.
    let expected: f32 = 1.0 + 0.25 * OPS as f32;

    // Push constants for vector_add: count + b_scale (=1.0, plain add).
    let push_add = PushCount1f {
        count: count as u32,
        value: 1.0,
    };
    let push_add_bytes = bytemuck::bytes_of(&push_add).to_vec();

    // Warmup pass: single vector_add chain to trigger any lazy compilation.
    let warm = ctx.execute_commands(|cb| {
        for i in 0..OPS {
            let set = if i % 2 == 0 { set_ac } else { set_ca };
            record_dispatch_1d(
                ctx,
                cb,
                &ctx.shaders.vector_add,
                set,
                &push_add_bytes,
                count as u32,
                256,
            );
            cmd_global_compute_barrier(ctx, cb);
        }
    });
    if let Err(e) = warm {
        cleanup(ctx);
        return Err(BandwidthError::Vulkan(e.to_string()));
    }

    let mut best_bw = 0.0f64;
    let mut worst_bw = f64::INFINITY;

    for run in 0..RUNS {
        // Reset A ← 1.0 for a fresh chain each run.
        if let Err(e) = fill_buffer(ctx, &a_buf, count, 1.0, fill_set) {
            cleanup(ctx);
            return Err(BandwidthError::Vulkan(e.to_string()));
        }

        let start = Instant::now();
        let dispatch_res = ctx.execute_commands(|cb| {
            for i in 0..OPS {
                let set = if i % 2 == 0 { set_ac } else { set_ca };
                record_dispatch_1d(
                    ctx,
                    cb,
                    &ctx.shaders.vector_add,
                    set,
                    &push_add_bytes,
                    count as u32,
                    256,
                );
                cmd_global_compute_barrier(ctx, cb);
            }
        });
        let elapsed = start.elapsed();
        if let Err(e) = dispatch_res {
            cleanup(ctx);
            return Err(BandwidthError::Vulkan(e.to_string()));
        }

        // After even OPS, final result lives in A; after odd OPS, in C.
        let final_buf = if OPS % 2 == 0 { &a_buf } else { &c_buf };

        // Probe: read one element and compare.
        let probe = read_f32_at(ctx, final_buf, 0)
            .map_err(|e| BandwidthError::Vulkan(e.to_string()));
        let probe = match probe {
            Ok(v) => v,
            Err(e) => {
                cleanup(ctx);
                return Err(e);
            }
        };
        if probe != expected {
            cleanup(ctx);
            return Err(BandwidthError::Correctness(format!(
                "Phép cộng cho kết quả sai: run {}, got {}, expected {}",
                run, probe, expected
            )));
        }

        // Full verify: run verify shader on the final buffer.
        bind_storage_buffers(
            &ctx.device,
            verify_set,
            &[(0, final_buf.buffer, bytes), (1, result_buf.buffer, 4)],
        );
        let verify_push = PushCount1f {
            count: count as u32,
            value: expected,
        };
        let verify_push_bytes = bytemuck::bytes_of(&verify_push).to_vec();

        let verify_res = ctx.execute_commands(|cb| unsafe {
            ctx.device.cmd_fill_buffer(cb, result_buf.buffer, 0, 4, 0);
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
            ctx.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
            record_dispatch_1d(
                ctx,
                cb,
                &ctx.shaders.verify,
                verify_set,
                &verify_push_bytes,
                count as u32,
                256,
            );
        });
        if let Err(e) = verify_res {
            cleanup(ctx);
            return Err(BandwidthError::Vulkan(e.to_string()));
        }
        let mismatches = read_u32_at(ctx, &result_buf, 0)
            .map_err(|e| BandwidthError::Vulkan(e.to_string()))?;
        if mismatches > 0 {
            cleanup(ctx);
            return Err(BandwidthError::Correctness(format!(
                "Đọc lại VRAM sai lệch dữ liệu (mismatches={}) — nghi ngờ VRAM hỏng",
                mismatches
            )));
        }

        // Bandwidth: each add moves 3 × buffer_size bytes (read A, read B, write C).
        let total_bytes = bytes as f64 * OPS as f64 * 3.0;
        let bw = total_bytes / elapsed.as_secs_f64() / 1e9;
        best_bw = best_bw.max(bw);
        worst_bw = worst_bw.min(bw);
    }

    cleanup(ctx);
    Ok(BandwidthResult { best_bw, worst_bw })
}

/// Fill a buffer with `value` via the fill compute shader.
fn fill_buffer(
    ctx: &VulkanContext,
    buf: &GpuBuffer,
    count: usize,
    value: f32,
    fill_set: vk::DescriptorSet,
) -> anyhow::Result<()> {
    bind_storage_buffers(&ctx.device, fill_set, &[(0, buf.buffer, buf.size)]);
    let push = PushCount1f {
        count: count as u32,
        value,
    };
    let push_bytes = bytemuck::bytes_of(&push).to_vec();
    ctx.execute_commands(|cb| {
        record_dispatch_1d(
            ctx,
            cb,
            &ctx.shaders.fill,
            fill_set,
            &push_bytes,
            count as u32,
            256,
        );
    })?;
    Ok(())
}
