//! Test 1: VRAM Capacity & Integrity.
//!
//! Allocates VRAM in 128 MB chunks, filling each with a rotating pattern and
//! verifying the readback bit-exactly on the GPU. Keeps every chunk alive
//! (like a real ML workload) until allocation fails — that failure point is
//! the true available VRAM.

use ash::vk;

use crate::compute::{allocate_descriptor_set, bind_storage_buffers, create_descriptor_pool};
use crate::memory::{read_f32_at, read_u32_at, GpuBuffer};
use crate::vulkan::VulkanContext;

use super::{
    cmd_global_compute_barrier, f32_bytes, record_dispatch_1d, BenchmarkResult, ProgressCallback,
    PushCount1f, TestStatus, VRAM_PATTERNS,
};

/// Chunk size: 128 MB = 8192 × 4096 × 4 bytes (f32).
const CHUNK_MB: usize = 128;
const CHUNK_F32_COUNT: usize = 8192 * 4096; // 33_554_432

pub fn run(
    ctx: &VulkanContext,
    dedicated_vram_mb: Option<u64>,
    progress: &mut dyn ProgressCallback,
) -> BenchmarkResult {
    // 100% danh nghĩa + 1GB dự phòng (nếu số liệu hệ thống sai / WDDM tràn shared memory).
    // Không biết danh nghĩa → quét đến 16 GB.
    let target_mb = dedicated_vram_mb
        .map(|n| n as usize + 1024)
        .unwrap_or(16 * 1024);
    let total_chunks = (target_mb / CHUNK_MB).max(1);

    progress.begin(Some(total_chunks as u64), "0 MB đã kiểm tra");

    // Result buffer holds a single u32 mismatch counter.
    let result_buf = match GpuBuffer::device_local(
        ctx,
        4,
        vk::BufferUsageFlags::STORAGE_BUFFER,
    ) {
        Ok(b) => b,
        Err(e) => {
            progress.end();
            return err_result(format!("Cấp phát result buffer thất bại: {}", e));
        }
    };

    // Descriptor pool: 2 sets (fill + verify), 3 storage bindings total.
    let pool = match create_descriptor_pool(&ctx.device, 2, 3) {
        Ok(p) => p,
        Err(e) => {
            result_buf.destroy(ctx);
            progress.end();
            return err_result(format!("Tạo descriptor pool thất bại: {}", e));
        }
    };

    let fill_set = match allocate_descriptor_set(
        &ctx.device,
        pool,
        ctx.shaders.fill.descriptor_set_layout,
    ) {
        Ok(s) => s,
        Err(e) => {
            unsafe { ctx.device.destroy_descriptor_pool(pool, None) };
            result_buf.destroy(ctx);
            progress.end();
            return err_result(format!("Alloc fill descriptor: {}", e));
        }
    };
    let verify_set = match allocate_descriptor_set(
        &ctx.device,
        pool,
        ctx.shaders.verify.descriptor_set_layout,
    ) {
        Ok(s) => s,
        Err(e) => {
            unsafe { ctx.device.destroy_descriptor_pool(pool, None) };
            result_buf.destroy(ctx);
            progress.end();
            return err_result(format!("Alloc verify descriptor: {}", e));
        }
    };

    let chunk_bytes = f32_bytes(CHUNK_F32_COUNT);
    let mut chunks: Vec<GpuBuffer> = Vec::with_capacity(total_chunks);
    let mut verified_mb: usize = 0;
    let mut integrity_errors: usize = 0;

    for i in 0..total_chunks {
        progress.set_position(i as u64);
        progress.set_message(&format!("{} MB đã kiểm tra", i * CHUNK_MB));

        let pattern = VRAM_PATTERNS[i % VRAM_PATTERNS.len()];

        // Try allocation — Err ⇒ measured capacity reached.
        let chunk = match GpuBuffer::device_local(
            ctx,
            chunk_bytes,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        ) {
            Ok(b) => b,
            Err(_) => break,
        };

        // Rebind descriptor sets to point at this chunk.
        bind_storage_buffers(&ctx.device, fill_set, &[(0, chunk.buffer, chunk_bytes)]);
        bind_storage_buffers(
            &ctx.device,
            verify_set,
            &[(0, chunk.buffer, chunk_bytes), (1, result_buf.buffer, 4)],
        );

        let dispatch_res = ctx.execute_commands(|cb| unsafe {
            // Reset error_count = 0 via transfer op.
            ctx.device.cmd_fill_buffer(cb, result_buf.buffer, 0, 4, 0);

            // TRANSFER → COMPUTE barrier so verify sees the zero counter.
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

            // Fill kernel: write pattern to every element.
            let push_fill = PushCount1f {
                count: CHUNK_F32_COUNT as u32,
                value: pattern,
            };
            record_dispatch_1d(
                ctx,
                cb,
                &ctx.shaders.fill,
                fill_set,
                bytemuck::bytes_of(&push_fill),
                CHUNK_F32_COUNT as u32,
                256,
            );

            cmd_global_compute_barrier(ctx, cb);

            // Verify kernel: atomicAdd error_count for each mismatch.
            let push_verify = PushCount1f {
                count: CHUNK_F32_COUNT as u32,
                value: pattern,
            };
            record_dispatch_1d(
                ctx,
                cb,
                &ctx.shaders.verify,
                verify_set,
                bytemuck::bytes_of(&push_verify),
                CHUNK_F32_COUNT as u32,
                256,
            );
        });

        match dispatch_res {
            Ok(_) => {
                let error_count = match read_u32_at(ctx, &result_buf, 0) {
                    Ok(v) => v,
                    Err(_) => {
                        chunk.destroy(ctx);
                        break;
                    }
                };
                let probe = match read_f32_at(ctx, &chunk, 0) {
                    Ok(v) => v,
                    Err(_) => {
                        chunk.destroy(ctx);
                        break;
                    }
                };

                if error_count > 0 || probe != pattern {
                    integrity_errors += 1;
                    progress.warn(&format!(
                        "Chunk #{} (~{} MB): DỮ LIỆU SAI (mismatches={}) — nghi ngờ VRAM hỏng vùng này!",
                        i,
                        i * CHUNK_MB,
                        error_count
                    ));
                }

                chunks.push(chunk); // giữ sống như model ML thật
                verified_mb = (i + 1) * CHUNK_MB;
            }
            Err(_) => {
                chunk.destroy(ctx);
                break;
            }
        }
    }

    progress.set_position(chunks.len() as u64);
    progress.end();

    // Cleanup
    for chunk in &chunks {
        chunk.destroy(ctx);
    }
    let n_chunks = chunks.len();
    drop(chunks);
    unsafe {
        ctx.device.destroy_descriptor_pool(pool, None);
    }
    result_buf.destroy(ctx);

    if n_chunks == 0 {
        return BenchmarkResult {
            name: "VRAM Capacity".to_string(),
            status: TestStatus::Error("Allocation failed".to_string()),
            score: 0.0,
            details: "Không thể cấp phát VRAM — GPU/driver có vấn đề".to_string(),
            metric_value: "N/A".to_string(),
        };
    }

    // ─── Scoring ─────────────────────────────────────────────────────────
    let measured_mb = verified_mb as f64;
    let nominal_mb = dedicated_vram_mb.map(|v| v as usize);
    let ratio = match nominal_mb {
        Some(n) if n > 0 => Some(measured_mb.min(n as f64) / n as f64),
        _ => None,
    };

    let mut score: f64 = match ratio {
        Some(r) if r >= 0.90 => 100.0,
        Some(r) if r >= 0.75 => 88.0,
        Some(r) if r >= 0.60 => 72.0,
        Some(r) if r >= 0.40 => 50.0,
        Some(_) => 25.0,
        None => {
            if measured_mb >= 8000.0 {
                100.0
            } else if measured_mb >= 4000.0 {
                85.0
            } else if measured_mb >= 2000.0 {
                70.0
            } else if measured_mb >= 1000.0 {
                55.0
            } else {
                40.0
            }
        }
    };

    if integrity_errors > 0 {
        score = score.min(35.0) / (1.0 + integrity_errors as f64);
    }

    let status = if integrity_errors > 0 {
        TestStatus::Failed
    } else if score >= 70.0 {
        TestStatus::Passed
    } else if score >= 40.0 {
        TestStatus::Warning
    } else {
        TestStatus::Failed
    };

    let mut notes: Vec<String> = Vec::new();
    if integrity_errors > 0 {
        notes.push(format!(
            "{} chunk sai dữ liệu — VRAM có vùng hỏng",
            integrity_errors
        ));
    }
    if let Some(n) = nominal_mb {
        if measured_mb > n as f64 * 1.02 {
            notes.push(
                "cấp phát vượt mức danh nghĩa (phần vượt có thể là shared memory của Windows)"
                    .to_string(),
            );
        }
    }
    if ratio.map(|r| r < 0.9).unwrap_or(false) {
        notes.push("khả dụng thấp — có thể app khác đang chiếm VRAM hoặc VRAM yếu".to_string());
    }

    let nominal_str = match nominal_mb {
        Some(n) => format!(
            " / danh nghĩa {} MB ({:.0}%)",
            n,
            measured_mb.min(n as f64) / n as f64 * 100.0
        ),
        None => String::new(),
    };
    let notes_str = if notes.is_empty() {
        String::new()
    } else {
        format!(" | {}", notes.join(" | "))
    };

    let details = format!(
        "Kiểm tra {} chunk × {} MB: khả dụng {:.0} MB{}{}",
        n_chunks, CHUNK_MB, measured_mb, nominal_str, notes_str
    );

    BenchmarkResult {
        name: "VRAM Capacity".to_string(),
        status,
        score,
        details,
        metric_value: match nominal_mb {
            Some(n) => format!("{:.0}/{:.0} MB", measured_mb, n),
            None => format!("{:.0} MB", measured_mb),
        },
    }
}

fn err_result(msg: String) -> BenchmarkResult {
    BenchmarkResult {
        name: "VRAM Capacity".to_string(),
        status: TestStatus::Error(msg.clone()),
        score: 0.0,
        details: msg,
        metric_value: "N/A".to_string(),
    }
}
