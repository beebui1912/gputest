//! Test 2: MatMul Throughput.
//!
//! Measures GFLOPS of a tiled matmul compute shader while cross-validating
//! results against a CPU f64 reference. A GPU that runs fast but computes
//! wrong values fails the test — correctness is a hard gate.

use std::time::Instant;

use ash::vk;

use crate::compute::{allocate_descriptor_set, bind_storage_buffers, create_descriptor_pool};
use crate::memory::{download_f32, upload_f32, GpuBuffer};
use crate::vulkan::VulkanContext;

use super::{
    f32_bytes, lcg_vec, record_dispatch_2d, BenchmarkResult, ProgressCallback, PushMatmul,
    TestStatus,
};

/// Sizes to benchmark, iterations per size, repeats.
const SIZES: &[usize] = &[512, 1024, 2048];
const REPEATS: usize = 3;

pub fn run(ctx: &VulkanContext, progress: &mut dyn ProgressCallback) -> BenchmarkResult {
    // ─── Correctness gate: 256×256 LCG matmul vs CPU f64 reference ───────
    progress.begin(None, "Correctness gate 256×256...");

    let gate = match run_correctness_gate(ctx) {
        Ok(max_rel) => max_rel,
        Err(e) => {
            progress.end();
            return BenchmarkResult {
                name: "MatMul Throughput".to_string(),
                status: TestStatus::Error(format!("Verify crashed: {}", e)),
                score: 0.0,
                details: format!("Không chạy được phép kiểm tra matmul: {}", e),
                metric_value: "N/A".to_string(),
            };
        }
    };

    if gate >= 1e-3 {
        progress.end();
        return BenchmarkResult {
            name: "MatMul Throughput".to_string(),
            status: TestStatus::Failed,
            score: 5.0,
            details: format!(
                "Kết quả không khớp chuẩn CPU f64 (sai số {:.2e}) — nghi ngờ lỗi phần cứng/driver.",
                gate
            ),
            metric_value: "INVALID".to_string(),
        };
    }
    progress.log(&format!(
        "Kiểm tra đúng đắn: PASS (max rel err = {:.2e})",
        gate
    ));
    progress.end();

    // ─── Throughput benchmark ────────────────────────────────────────────
    let mut best_gflops: f64 = 0.0;
    let mut worst_spread: f64 = 0.0;
    let mut any_invalid = false;
    let mut any_failed = false;

    for &size in SIZES {
        progress.begin(None, &format!("MatMul {}×{}...", size, size));

        let result = run_single_size(ctx, size);

        match result {
            Ok(SizeStats {
                gflops,
                best_ms,
                spread,
                non_finite,
                max_rel,
            }) => {
                progress.end();
                let valid = non_finite == 0 && max_rel < 5e-3;
                if !valid {
                    any_invalid = true;
                }
                if gflops > best_gflops {
                    best_gflops = gflops;
                }
                if spread > worst_spread {
                    worst_spread = spread;
                }
                let verdict = if valid {
                    format!("(verify OK, err {:.1e})", max_rel)
                } else {
                    format!("(verify FAIL: {} NaN/Inf, err {:.1e})", non_finite, max_rel)
                };
                progress.log(&format!(
                    "{}×{}: {:>7.2} ms → {:>7.1} GFLOPS | spread {:>4.0}% {}",
                    size, size, best_ms, gflops, spread, verdict
                ));
            }
            Err(e) => {
                progress.end();
                any_failed = true;
                progress.warn(&format!("{}×{}: FAILED — {}", size, size, e));
            }
        }
    }

    // ─── Score ───────────────────────────────────────────────────────────
    let (score, status, details, metric) = if any_invalid {
        (
            15.0,
            TestStatus::Failed,
            "Kết quả KHÔNG hợp lệ (NaN/Inf hoặc sai số lớn so với chuẩn CPU) — GPU đang tính sai."
                .to_string(),
            "INVALID".to_string(),
        )
    } else if any_failed && best_gflops == 0.0 {
        (
            0.0,
            TestStatus::Error("Tất cả kích thước đều lỗi".to_string()),
            "Không hoàn thành phép matmul nào (hết VRAM?)".to_string(),
            "N/A".to_string(),
        )
    } else {
        let mut score: f64 = if best_gflops >= 5000.0 {
            100.0
        } else if best_gflops >= 2000.0 {
            90.0
        } else if best_gflops >= 1000.0 {
            80.0
        } else if best_gflops >= 500.0 {
            70.0
        } else if best_gflops >= 100.0 {
            55.0
        } else if best_gflops >= 10.0 {
            35.0
        } else {
            15.0
        };
        let unstable = worst_spread > 30.0;
        if unstable {
            score = score.min(65.0);
        }
        let status = if score >= 70.0 {
            TestStatus::Passed
        } else if score >= 40.0 {
            TestStatus::Warning
        } else {
            TestStatus::Failed
        };
        (
            score,
            status,
            format!(
                "Peak {:.1} GFLOPS (đã verify đúng đắn) | dao động {:.0}%{}",
                best_gflops,
                worst_spread,
                if unstable {
                    " — hiệu năng bất ổn giữa các lần chạy"
                } else {
                    ""
                }
            ),
            format!("{:.1} GFLOPS", best_gflops),
        )
    };

    BenchmarkResult {
        name: "MatMul Throughput".to_string(),
        status,
        score,
        details,
        metric_value: metric,
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Internal helpers
// ═══════════════════════════════════════════════════════════════════════════════

struct SizeStats {
    gflops: f64,
    best_ms: f64,
    spread: f64,
    non_finite: usize,
    max_rel: f64,
}

/// Run 256×256 matmul with LCG-seeded inputs, verify ALL 65 536 outputs vs
/// CPU f64 reference. Returns max relative error.
fn run_correctness_gate(ctx: &VulkanContext) -> anyhow::Result<f64> {
    const N: usize = 256;
    let av = lcg_vec(N * N, 0xDEAD_BEEF);
    let bv = lcg_vec(N * N, 0x0BAD_C0DE);

    let bytes = f32_bytes(N * N);
    let a_buf = GpuBuffer::device_local(ctx, bytes, vk::BufferUsageFlags::STORAGE_BUFFER)?;
    let b_buf = GpuBuffer::device_local(ctx, bytes, vk::BufferUsageFlags::STORAGE_BUFFER)?;
    let c_buf = GpuBuffer::device_local(ctx, bytes, vk::BufferUsageFlags::STORAGE_BUFFER)?;

    upload_f32(ctx, &a_buf, &av)?;
    upload_f32(ctx, &b_buf, &bv)?;

    let pool = create_descriptor_pool(&ctx.device, 1, 3)?;
    let set = allocate_descriptor_set(&ctx.device, pool, ctx.shaders.matmul.descriptor_set_layout)?;
    bind_storage_buffers(
        &ctx.device,
        set,
        &[
            (0, a_buf.buffer, bytes),
            (1, b_buf.buffer, bytes),
            (2, c_buf.buffer, bytes),
        ],
    );

    let push = PushMatmul { n: N as u32 };
    ctx.execute_commands(|cb| {
        record_dispatch_2d(
            ctx,
            cb,
            &ctx.shaders.matmul,
            set,
            bytemuck::bytes_of(&push),
            N as u32,
            16,
        );
    })?;

    let cs = download_f32(ctx, &c_buf, N * N)?;

    // Cleanup
    unsafe { ctx.device.destroy_descriptor_pool(pool, None) };
    a_buf.destroy(ctx);
    b_buf.destroy(ctx);
    c_buf.destroy(ctx);

    // Scale-aware relative error comparison across all positions.
    let mut max_rel: f64 = 0.0;
    for i in 0..N {
        for j in 0..N {
            let mut acc = 0.0f64;
            let mut scale = 1e-9f64;
            for k in 0..N {
                let t = av[i * N + k] as f64 * bv[k * N + j] as f64;
                acc += t;
                scale += t.abs();
            }
            let got = cs[i * N + j] as f64;
            let rel = (got - acc).abs() / scale;
            if rel > max_rel {
                max_rel = rel;
            }
        }
    }
    Ok(max_rel)
}

/// Run a single benchmark size: time matmul iterations, then verify.
fn run_single_size(ctx: &VulkanContext, size: usize) -> anyhow::Result<SizeStats> {
    let av = lcg_vec(size * size, 0xCAFE_BABE ^ size as u64);
    let bv = lcg_vec(size * size, 0xFEED_FACE ^ size as u64);

    let bytes = f32_bytes(size * size);
    let a_buf = GpuBuffer::device_local(ctx, bytes, vk::BufferUsageFlags::STORAGE_BUFFER)?;
    let b_buf = GpuBuffer::device_local(ctx, bytes, vk::BufferUsageFlags::STORAGE_BUFFER)?;
    let c_buf = GpuBuffer::device_local(ctx, bytes, vk::BufferUsageFlags::STORAGE_BUFFER)?;

    upload_f32(ctx, &a_buf, &av)?;
    upload_f32(ctx, &b_buf, &bv)?;

    let pool = create_descriptor_pool(&ctx.device, 1, 3)?;
    let set = allocate_descriptor_set(&ctx.device, pool, ctx.shaders.matmul.descriptor_set_layout)?;
    bind_storage_buffers(
        &ctx.device,
        set,
        &[
            (0, a_buf.buffer, bytes),
            (1, b_buf.buffer, bytes),
            (2, c_buf.buffer, bytes),
        ],
    );

    let push = PushMatmul { n: size as u32 };
    let push_bytes = bytemuck::bytes_of(&push).to_vec(); // owned so closure can capture

    // Warmup pass
    {
        let pb = push_bytes.clone();
        ctx.execute_commands(|cb| {
            record_dispatch_2d(ctx, cb, &ctx.shaders.matmul, set, &pb, size as u32, 16);
        })?;
    }

    let iterations = if size <= 512 {
        10
    } else if size <= 1024 {
        5
    } else {
        3
    };

    let mut times: Vec<f64> = Vec::with_capacity(REPEATS);
    for _ in 0..REPEATS {
        let start = Instant::now();
        for _ in 0..iterations {
            let pb = push_bytes.clone();
            ctx.execute_commands(|cb| {
                record_dispatch_2d(ctx, cb, &ctx.shaders.matmul, set, &pb, size as u32, 16);
            })?;
        }
        times.push(start.elapsed().as_secs_f64() / iterations as f64);
    }

    // Final result to verify — one more matmul then download.
    {
        let pb = push_bytes.clone();
        ctx.execute_commands(|cb| {
            record_dispatch_2d(ctx, cb, &ctx.shaders.matmul, set, &pb, size as u32, 16);
        })?;
    }
    let cs = download_f32(ctx, &c_buf, size * size)?;

    // Cleanup
    unsafe { ctx.device.destroy_descriptor_pool(pool, None) };
    a_buf.destroy(ctx);
    b_buf.destroy(ctx);
    c_buf.destroy(ctx);

    let mut non_finite = 0usize;
    for &v in &cs {
        if !v.is_finite() {
            non_finite += 1;
        }
    }

    // Sample 64 positions vs CPU f64 reference (scale-aware relative error).
    let mut max_rel: f64 = 0.0;
    for s in 0..64u64 {
        let idx = (s.wrapping_mul(2654435761).wrapping_add(97) as usize) % (size * size);
        let i = idx / size;
        let j = idx % size;
        let mut acc = 0.0f64;
        let mut scale = 1e-9f64;
        for k in 0..size {
            let t = av[i * size + k] as f64 * bv[k * size + j] as f64;
            acc += t;
            scale += t.abs();
        }
        let got = cs[idx] as f64;
        let rel = (got - acc).abs() / scale;
        if rel > max_rel {
            max_rel = rel;
        }
    }

    let best_time = times.iter().cloned().fold(f64::INFINITY, f64::min);
    let worst_time = times.iter().cloned().fold(0.0f64, f64::max);
    let gflops = 2.0 * (size as f64).powi(3) / best_time / 1e9;
    let spread = if worst_time > 0.0 {
        (worst_time - best_time) / worst_time * 100.0
    } else {
        0.0
    };

    Ok(SizeStats {
        gflops,
        best_ms: best_time * 1000.0,
        spread,
        non_finite,
        max_rel,
    })
}
