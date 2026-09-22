//! Test 4: Stability (30 s stress test).
//!
//! Runs a matmul in a tight loop for 30 seconds, comparing every output
//! against a reference computed once. Also measures per-iteration timing
//! drift (25% head vs 25% tail) to detect thermal throttling.

use std::time::{Duration, Instant};

use ash::vk;

use crate::compute::{allocate_descriptor_set, bind_storage_buffers, create_descriptor_pool};
use crate::memory::{download_f32, upload_f32, GpuBuffer};
use crate::vulkan::VulkanContext;

use super::{
    f32_bytes, lcg_vec, record_dispatch_2d, BenchmarkResult, ProgressCallback, PushMatmul,
    TestStatus,
};

const SIZE: usize = 1024;
const DURATION_SECS: u64 = 30;

pub fn run(ctx: &VulkanContext, progress: &mut dyn ProgressCallback) -> BenchmarkResult {
    progress.begin(Some(DURATION_SECS), "0 vòng | 0 lỗi");

    let outcome = run_inner(ctx, progress);
    progress.set_position(DURATION_SECS);
    progress.end();

    match outcome {
        Ok(Stats {
            total_iterations,
            bad_elements,
            drop_pct,
        }) => {
            if total_iterations == 0 {
                return BenchmarkResult {
                    name: "Stability (30s)".to_string(),
                    status: TestStatus::Error("No iterations".to_string()),
                    score: 0.0,
                    details: "Không chạy được vòng lặp nào trong 30s".to_string(),
                    metric_value: "N/A".to_string(),
                };
            }

            let total_elements = total_iterations
                .saturating_mul((SIZE * SIZE) as u64)
                .max(1);
            let elem_err_rate = bad_elements as f64 / total_elements as f64;

            let mut score: f64 = if bad_elements == 0 {
                100.0
            } else if elem_err_rate < 1e-6 {
                90.0
            } else if elem_err_rate < 1e-4 {
                70.0
            } else if elem_err_rate < 1e-2 {
                40.0
            } else {
                10.0
            };

            let throttled = drop_pct > 15.0;
            if drop_pct > 30.0 {
                score *= 0.6;
            } else if drop_pct > 15.0 {
                score *= 0.85;
            }
            score = score.max(5.0);

            let status = if score >= 80.0 && !throttled {
                TestStatus::Passed
            } else if score >= 50.0 {
                TestStatus::Warning
            } else {
                TestStatus::Failed
            };

            let details = format!(
                "{} vòng | {} phần tử sai ({:.2e}) | hiệu năng giữ {:.0}%{}",
                total_iterations,
                bad_elements,
                elem_err_rate,
                100.0 - drop_pct,
                if throttled {
                    " — nghi ngờ thermal throttling"
                } else {
                    ""
                }
            );

            BenchmarkResult {
                name: "Stability (30s)".to_string(),
                status,
                score,
                details,
                metric_value: format!("{:.4}%", elem_err_rate * 100.0),
            }
        }
        Err(msg) => BenchmarkResult {
            name: "Stability (30s)".to_string(),
            status: TestStatus::Error(msg.clone()),
            score: 0.0,
            details: format!("GPU gặp sự cố trong stress test: {}", msg),
            metric_value: "CRASH".to_string(),
        },
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Internal
// ═══════════════════════════════════════════════════════════════════════════════

struct Stats {
    total_iterations: u64,
    bad_elements: u64,
    drop_pct: f64,
}

fn run_inner(ctx: &VulkanContext, progress: &mut dyn ProgressCallback) -> Result<Stats, String> {
    let av = lcg_vec(SIZE * SIZE, 0xA5A5_5A5A);
    let bv = lcg_vec(SIZE * SIZE, 0x5A5A_A5A5);

    let bytes = f32_bytes(SIZE * SIZE);
    let a_buf = GpuBuffer::device_local(ctx, bytes, vk::BufferUsageFlags::STORAGE_BUFFER)
        .map_err(|e| e.to_string())?;
    let b_buf = GpuBuffer::device_local(ctx, bytes, vk::BufferUsageFlags::STORAGE_BUFFER)
        .map_err(|e| e.to_string())?;
    let c_buf = GpuBuffer::device_local(ctx, bytes, vk::BufferUsageFlags::STORAGE_BUFFER)
        .map_err(|e| e.to_string())?;

    upload_f32(ctx, &a_buf, &av).map_err(|e| e.to_string())?;
    upload_f32(ctx, &b_buf, &bv).map_err(|e| e.to_string())?;

    let pool = create_descriptor_pool(&ctx.device, 1, 3).map_err(|e| e.to_string())?;
    let set = allocate_descriptor_set(&ctx.device, pool, ctx.shaders.matmul.descriptor_set_layout)
        .map_err(|e| e.to_string())?;
    bind_storage_buffers(
        &ctx.device,
        set,
        &[
            (0, a_buf.buffer, bytes),
            (1, b_buf.buffer, bytes),
            (2, c_buf.buffer, bytes),
        ],
    );

    let push = PushMatmul { n: SIZE as u32 };
    let push_bytes = bytemuck::bytes_of(&push).to_vec();

    // Reference: compute once, keep on CPU.
    ctx.execute_commands(|cb| {
        record_dispatch_2d(ctx, cb, &ctx.shaders.matmul, set, &push_bytes, SIZE as u32, 16);
    })
    .map_err(|e| e.to_string())?;
    let reference = download_f32(ctx, &c_buf, SIZE * SIZE).map_err(|e| e.to_string())?;

    // 30s stress loop.
    let duration = Duration::from_secs(DURATION_SECS);
    let start = Instant::now();
    let mut last_tick = Instant::now();
    let mut iter_count: u64 = 0;
    let mut bad_elements: u64 = 0;
    let mut iter_times: Vec<f64> = Vec::new();

    let mut err: Option<String> = None;
    while start.elapsed() < duration {
        let t0 = Instant::now();
        if let Err(e) = ctx.execute_commands(|cb| {
            record_dispatch_2d(ctx, cb, &ctx.shaders.matmul, set, &push_bytes, SIZE as u32, 16);
        }) {
            err = Some(e.to_string());
            break;
        }
        let got = match download_f32(ctx, &c_buf, SIZE * SIZE) {
            Ok(v) => v,
            Err(e) => {
                err = Some(e.to_string());
                break;
            }
        };
        let dt = t0.elapsed().as_secs_f64();

        // Compare against reference (element-wise, tolerance-based).
        let mut iter_bad: u64 = 0;
        for (r, g) in reference.iter().zip(got.iter()) {
            let diff = (*r as f64 - *g as f64).abs();
            if diff > 1e-4 * (1.0 + (*r as f64).abs()) {
                iter_bad += 1;
            }
        }
        bad_elements += iter_bad;
        iter_times.push(dt);
        iter_count += 1;

        if last_tick.elapsed() >= Duration::from_secs(1) {
            progress.set_position(start.elapsed().as_secs().min(DURATION_SECS));
            progress.set_message(&format!("{} vòng | {} lỗi", iter_count, bad_elements));
            last_tick = Instant::now();
        }
    }

    // Cleanup
    unsafe { ctx.device.destroy_descriptor_pool(pool, None) };
    a_buf.destroy(ctx);
    b_buf.destroy(ctx);
    c_buf.destroy(ctx);

    if let Some(msg) = err {
        if iter_count == 0 {
            return Err(msg);
        }
        // Partial failure — still return what we have with the count so far.
        // Fall through to compute stats.
    }

    // 25% head vs 25% tail → thermal throttling signal.
    let q = (iter_times.len() / 4).max(1);
    let head_avg: f64 = iter_times.iter().take(q).sum::<f64>() / q as f64;
    let tail_avg: f64 = iter_times.iter().rev().take(q).sum::<f64>() / q as f64;
    let drop_pct = if head_avg > 1e-9 {
        ((1.0 - head_avg / tail_avg) * 100.0).max(0.0)
    } else {
        0.0
    };

    Ok(Stats {
        total_iterations: iter_count,
        bad_elements,
        drop_pct,
    })
}
