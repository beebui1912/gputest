//! GPU Health Evaluation Tool — thin UI layer over `gpubench-core`.
//!
//! Ported from `sample.rs` lines 1351–1713 (UI/reporting logic), adapted to
//! call the raw-Vulkan benchmark library.

use std::time::Duration;

use colored::*;
use dialoguer::{theme::ColorfulTheme, Select};
use indicatif::{ProgressBar, ProgressStyle};

use gpubench_core::benchmarks::{bandwidth, matmul, precision, stability, vram};
use gpubench_core::benchmarks::{BenchmarkResult, ProgressCallback, TestStatus};
use gpubench_core::gpu_info::{enumerate_gpus, GpuInfo};
use gpubench_core::vulkan::VulkanInstance;

// ═══════════════════════════════════════════════════════════════════════════════
// Data structures — UI-only (report rating)
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug)]
struct HealthReport {
    gpu_name: String,
    results: Vec<BenchmarkResult>,
    overall_score: f64,
    rating: HealthRating,
}

#[derive(Debug)]
enum HealthRating {
    Excellent,
    Good,
    Fair,
    Poor,
}

impl HealthRating {
    fn label(&self) -> ColoredString {
        match self {
            HealthRating::Excellent => "★★★★★ EXCELLENT".bright_green().bold(),
            HealthRating::Good => "★★★★☆ GOOD".green().bold(),
            HealthRating::Fair => "★★★☆☆ FAIR".yellow().bold(),
            HealthRating::Poor => "★★☆☆☆ POOR".red().bold(),
        }
    }

    fn description(&self) -> &str {
        match self {
            HealthRating::Excellent => {
                "GPU hoạt động xuất sắc! Sẵn sàng cho mọi tác vụ ML/AI và lập trình GPU."
            }
            HealthRating::Good => {
                "GPU hoạt động ổn định. Phù hợp cho phần lớn tác vụ lập trình và training."
            }
            HealthRating::Fair => {
                "GPU có dấu hiệu giảm hiệu năng. Vẫn dùng được nhưng cần theo dõi."
            }
            HealthRating::Poor => {
                "GPU có vấn đề nghiêm trọng! Không nên dùng cho tác vụ nặng hoặc production."
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Progress adapter: gpubench-core ProgressCallback → indicatif ProgressBar
// ═══════════════════════════════════════════════════════════════════════════════

struct IndicatifProgress {
    bar: Option<ProgressBar>,
    total: Option<u64>,
}

impl IndicatifProgress {
    fn new() -> Self {
        Self {
            bar: None,
            total: None,
        }
    }
}

impl ProgressCallback for IndicatifProgress {
    fn begin(&mut self, total: Option<u64>, message: &str) {
        // Clean up any prior stage that wasn't ended explicitly.
        if let Some(bar) = self.bar.take() {
            bar.finish_and_clear();
        }

        let bar = match total {
            Some(n) => {
                let pb = ProgressBar::new(n);
                pb.set_style(
                    ProgressStyle::default_bar()
                        .template("    [{bar:40.cyan/blue}] {pos}/{len} | {msg}")
                        .unwrap()
                        .progress_chars("█▓░"),
                );
                pb
            }
            None => {
                let pb = ProgressBar::new_spinner();
                pb.set_style(
                    ProgressStyle::default_spinner()
                        .template("    {spinner:.cyan} {msg}")
                        .unwrap(),
                );
                pb.enable_steady_tick(Duration::from_millis(120));
                pb
            }
        };
        bar.set_message(message.to_string());
        self.bar = Some(bar);
        self.total = total;
    }

    fn set_position(&mut self, pos: u64) {
        if let Some(bar) = &self.bar {
            bar.set_position(pos);
        }
    }

    fn set_message(&mut self, message: &str) {
        if let Some(bar) = &self.bar {
            bar.set_message(message.to_string());
        }
    }

    fn log(&mut self, message: &str) {
        let indented = format!("    {}", message);
        match &self.bar {
            Some(bar) => bar.println(indented),
            None => println!("{}", indented),
        }
    }

    fn warn(&mut self, message: &str) {
        let indented = format!("    {} {}", "⚠".yellow(), message);
        match &self.bar {
            Some(bar) => bar.println(indented),
            None => println!("{}", indented),
        }
    }

    fn end(&mut self) {
        if let Some(bar) = self.bar.take() {
            bar.finish_and_clear();
        }
        self.total = None;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// UI helpers
// ═══════════════════════════════════════════════════════════════════════════════

fn status_icon(status: &TestStatus) -> ColoredString {
    match status {
        TestStatus::Passed => "✓".bright_green(),
        TestStatus::Warning => "⚠".yellow(),
        TestStatus::Failed => "✗".red(),
        TestStatus::Error(_) => "✗".bright_red(),
    }
}

fn score_color(score: f64) -> ColoredString {
    let text = format!("{:.0}/100", score);
    if score >= 80.0 {
        text.bright_green()
    } else if score >= 60.0 {
        text.green()
    } else if score >= 40.0 {
        text.yellow()
    } else {
        text.red()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test wrappers — print header + description, run, print status line
// ═══════════════════════════════════════════════════════════════════════════════

fn run_vram(ctx: &gpubench_core::vulkan::VulkanContext, gpu: &GpuInfo) -> BenchmarkResult {
    println!(
        "\n  {} {}",
        "▶".bright_cyan(),
        "VRAM Capacity & Integrity Test".bold()
    );
    println!("    Cấp phát từng chunk 128 MB: ghi pattern → đọc lại verify, giữ sống liên tục");
    println!("    cho tới khi hết VRAM. Mục tiêu: quét 100% VRAM danh nghĩa (+1 GB dự phòng).");

    let mut progress = IndicatifProgress::new();
    let result = vram::run(ctx, gpu.dedicated_vram_mb, &mut progress);
    println!("    {} {}", status_icon(&result.status), result.details.dimmed());
    result
}

fn run_matmul(ctx: &gpubench_core::vulkan::VulkanContext) -> BenchmarkResult {
    println!(
        "\n  {} {}",
        "▶".bright_cyan(),
        "MatMul Throughput Test".bold()
    );
    println!("    Đo GFLOPS (2×N³/giây) kèm KIỂM TRA KẾT QUẢ so với chuẩn CPU f64.");
    println!("    Tốc độ cao mà tính sai thì vô nghĩa — điều bản cũ hoàn toàn bỏ qua.");

    let mut progress = IndicatifProgress::new();
    let result = matmul::run(ctx, &mut progress);
    println!("    {} {}", status_icon(&result.status), result.details.dimmed());
    result
}

fn run_bandwidth(ctx: &gpubench_core::vulkan::VulkanContext, gpu: &GpuInfo) -> BenchmarkResult {
    println!(
        "\n  {} {}",
        "▶".bright_cyan(),
        "Memory Bandwidth Test".bold()
    );
    println!("    Bandwidth = tốc độ đọc/ghi VRAM (GB/s) — cổ chai cho inference LLM,");
    println!("    normalize/activation, copy model weights... (tác vụ không cần nhiều FLOPS).");
    let mut progress = IndicatifProgress::new();
    let result = bandwidth::run(ctx, gpu.dedicated_vram_mb, &mut progress);
    println!("    {} {}", status_icon(&result.status), result.details.dimmed());
    result
}

fn run_stability(ctx: &gpubench_core::vulkan::VulkanContext) -> BenchmarkResult {
    println!(
        "\n  {} {}",
        "▶".bright_cyan(),
        "Stability Test (30s)".bold()
    );
    println!("    Stress-test 30 giây: matmul lặp liên tục, so sánh TOÀN BỘ phần tử");
    println!("    với kết quả tham chiếu + đo độ trôi hiệu năng (phát hiện thermal throttling).");

    let mut progress = IndicatifProgress::new();
    let result = stability::run(ctx, &mut progress);
    println!("    {} {}", status_icon(&result.status), result.details.dimmed());
    result
}

fn run_precision(ctx: &gpubench_core::vulkan::VulkanContext) -> BenchmarkResult {
    println!("\n  {} {}", "▶".bright_cyan(), "Precision Test".bold());
    println!("    Đang kiểm tra độ chính xác tính toán float32...");

    let mut progress = IndicatifProgress::new();
    let result = precision::run(ctx, &mut progress);
    println!("    {} {}", status_icon(&result.status), result.details.dimmed());
    result
}

// ═══════════════════════════════════════════════════════════════════════════════
// Reporting
// ═══════════════════════════════════════════════════════════════════════════════

fn generate_report(gpu_name: &str, results: Vec<BenchmarkResult>) -> HealthReport {
    let overall_score = if results.is_empty() {
        0.0
    } else {
        results.iter().map(|r| r.score).sum::<f64>() / results.len() as f64
    };

    let rating = if overall_score >= 80.0 {
        HealthRating::Excellent
    } else if overall_score >= 60.0 {
        HealthRating::Good
    } else if overall_score >= 40.0 {
        HealthRating::Fair
    } else {
        HealthRating::Poor
    };

    HealthReport {
        gpu_name: gpu_name.to_string(),
        results,
        overall_score,
        rating,
    }
}

fn print_report(report: &HealthReport) {
    let separator = "═".repeat(70);
    let thin_sep = "─".repeat(70);

    println!("\n\n{}", separator.bright_cyan());
    println!(
        "{}",
        "  ██████╗ ██████╗ ██╗   ██╗    ██████╗ ███████╗██████╗  ██████╗ ██████╗ ████████╗"
            .bright_cyan()
    );
    println!(
        "{}",
        "  ██╔════╝ ██╔══██╗██║   ██║    ██╔══██╗██╔════╝██╔══██╗██╔═══██╗██╔══██╗╚══██╔══╝"
            .bright_cyan()
    );
    println!(
        "{}",
        "  ██║  ███╗██████╔╝██║   ██║    ██████╔╝█████╗  ██████╔╝██║   ██╗██████╔╝   ██║"
            .bright_cyan()
    );
    println!(
        "{}",
        "  ██║   ██║██╔═══╝ ██║   ██║    ██╔══██╗██╔══╝  ██╔═══╝ ██║   ██║██╔══██╗   ██║"
            .bright_cyan()
    );
    println!(
        "{}",
        "  ╚██████╔╝██║     ╚██████╔╝    ██║  ██║███████╗██║     ╚██████╔╝██║  ██║   ██║"
            .bright_cyan()
    );
    println!(
        "{}",
        "   ╚═════╝ ╚═╝      ╚═════╝     ╚═╝  ╚═╝╚══════╝╚═╝      ╚═════╝ ╚═╝  ╚═╝   ╚═╝"
            .bright_cyan()
    );
    println!("{}", separator.bright_cyan());

    println!(
        "\n  {} {}",
        "GPU:".bright_white().bold(),
        report.gpu_name.bright_yellow()
    );
    println!(
        "  {} {}",
        "Framework:".bright_white().bold(),
        "ash 0.38 (Raw Vulkan / SPIR-V)".dimmed()
    );
    println!();

    // Results table
    println!("{}", thin_sep.dimmed());
    println!(
        "  {:<25} {:<15} {:<18} {}",
        "Test".bold().bright_white(),
        "Status".bold().bright_white(),
        "Metric".bold().bright_white(),
        "Score".bold().bright_white(),
    );
    println!("{}", thin_sep.dimmed());

    for result in &report.results {
        let status_str = match &result.status {
            TestStatus::Passed => "PASSED".bright_green(),
            TestStatus::Warning => "WARNING".yellow(),
            TestStatus::Failed => "FAILED".red(),
            TestStatus::Error(msg) => format!("ERROR: {}", msg).bright_red().into(),
        };

        println!(
            "  {:<25} {:<15} {:<18} {}",
            result.name.bright_white(),
            status_str,
            result.metric_value.dimmed(),
            score_color(result.score),
        );
    }

    println!("{}", thin_sep.dimmed());

    // Per-test details
    println!("\n{}", thin_sep.dimmed());
    println!("  {}", "Chi tiết benchmark:".bold().bright_white());
    println!("{}", thin_sep.dimmed());
    for result in &report.results {
        println!(
            "  {} {}",
            format!("• {}:", result.name).bright_white(),
            result.details.dimmed()
        );
    }

    // Overall
    println!();
    println!(
        "  {} {}",
        "Overall Score:".bold().bright_white(),
        score_color(report.overall_score)
    );
    println!(
        "  {} {}",
        "Health Rating:".bold().bright_white(),
        report.rating.label()
    );
    println!();
    println!(
        "  {} {}",
        "📋".to_string(),
        report.rating.description().italic().dimmed()
    );

    // Task suitability
    println!("\n{}", thin_sep.dimmed());
    println!(
        "  {}",
        "Đánh giá cho các tác vụ lập trình:".bold().bright_white()
    );
    println!("{}", thin_sep.dimmed());

    let tasks = [
        ("ML/AI Training", 75.0),
        ("Deep Learning Inference", 60.0),
        ("Shader Compilation", 50.0),
        ("GPU Compute (General)", 45.0),
        ("3D Rendering/Graphics", 55.0),
    ];

    for (task, min_score) in &tasks {
        let suitable = report.overall_score >= *min_score;
        let icon = if suitable {
            "✓".bright_green()
        } else {
            "✗".red()
        };
        let label = if suitable {
            "Phù hợp".bright_green()
        } else {
            "Không khuyến khích".red()
        };
        println!("  {} {:<30} {}", icon, task, label);
    }

    println!("\n{}", separator.bright_cyan());
    println!(
        "  {}",
        "Powered by ash 0.38 | Raw Vulkan Backend"
            .dimmed()
            .italic()
    );
    println!("{}\n", separator.bright_cyan());
}

// ═══════════════════════════════════════════════════════════════════════════════
// GPU list & selection
// ═══════════════════════════════════════════════════════════════════════════════

fn print_gpu_list(gpus: &[GpuInfo]) {
    for gpu in gpus {
        println!(
            "    {}  {}",
            format!("[{}]", gpu.device_index).bright_cyan(),
            gpu.name.bright_white().bold()
        );
        println!(
            "        Type: {}  |  Backend: {}  |  Vendor: {} (0x{:04X})",
            gpu.device_type.dimmed(),
            "Vulkan".dimmed(),
            gpu.vendor.dimmed(),
            gpu.vendor_id
        );
        if let Some(vram_mb) = gpu.dedicated_vram_mb {
            println!(
                "        VRAM (Vulkan heap): {}",
                format!("{} MB ({:.1} GB)", vram_mb, vram_mb as f64 / 1024.0).dimmed()
            );
        }
        println!(
            "        Driver: {} | API: {}",
            gpu.driver_version.dimmed(),
            gpu.api_version.dimmed()
        );
        println!();
    }
}

fn resolve_selection(gpus: &[GpuInfo], args: &[String]) -> Option<usize> {
    for i in 0..args.len() {
        if (args[i] == "--device" || args[i] == "-d") && i + 1 < args.len() {
            if let Ok(idx) = args[i + 1].parse::<usize>() {
                if idx < gpus.len() {
                    return Some(idx);
                } else {
                    println!(
                        "  {} Chỉ số card '{}' không hợp lệ! Có {} GPU (0..{})",
                        "⚠".yellow(),
                        idx,
                        gpus.len(),
                        gpus.len() - 1
                    );
                }
            }
        }
    }

    if gpus.len() == 1 {
        println!("  {} Chỉ phát hiện 1 GPU, tự động chọn.", "ℹ".bright_blue());
        return Some(0);
    }

    let selections: Vec<String> = gpus
        .iter()
        .map(|g| format!("{} ({})", g.name, g.device_type))
        .collect();

    match Select::with_theme(&ColorfulTheme::default())
        .with_prompt("  Chọn GPU để kiểm tra")
        .items(&selections)
        .default(0)
        .interact_opt()
    {
        Ok(Some(s)) => Some(s),
        _ => {
            println!(
                "  {} Môi trường không tương tác hoặc đã hủy, mặc định chọn GPU 0.",
                "ℹ".bright_blue()
            );
            Some(0)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════════

fn main() {
    println!();
    println!(
        "{}",
        "╔══════════════════════════════════════════════════════════════════╗"
            .bright_cyan()
    );
    println!(
        "{}",
        "║       GPU HEALTH EVALUATION TOOL - ash 0.38 (Raw Vulkan)      ║"
            .bright_cyan()
            .bold()
    );
    println!(
        "{}",
        "╚══════════════════════════════════════════════════════════════════╝"
            .bright_cyan()
    );
    println!();

    // ─── Step 1: Create Vulkan instance ──────────────────────────────────
    println!(
        "  {} {}",
        "🔍".to_string(),
        "Đang khởi tạo Vulkan và quét tìm GPU...".bright_white()
    );

    let instance = match VulkanInstance::new() {
        Ok(inst) => inst,
        Err(e) => {
            println!(
                "\n  {} {}",
                "✗".red(),
                format!("Không khởi tạo được Vulkan: {}", e).red().bold()
            );
            println!("    Vui lòng kiểm tra:");
            println!("    • Driver GPU đã được cài đặt");
            println!("    • Vulkan runtime được hỗ trợ trên hệ thống");
            println!("    • GPU không bị disable trong Device Manager");
            std::process::exit(1);
        }
    };

    // ─── Step 2: Enumerate GPUs ──────────────────────────────────────────
    let gpus = match enumerate_gpus(&instance) {
        Ok(v) => v,
        Err(e) => {
            println!(
                "\n  {} {}",
                "✗".red(),
                format!("Không liệt kê được GPU: {}", e).red().bold()
            );
            std::process::exit(1);
        }
    };

    if gpus.is_empty() {
        println!(
            "\n  {} {}",
            "✗".red(),
            "Không tìm thấy GPU nào hỗ trợ Vulkan!".red().bold()
        );
        std::process::exit(1);
    }

    println!(
        "  {} Tìm thấy {} GPU:\n",
        "✓".bright_green(),
        gpus.len().to_string().bright_yellow()
    );
    print_gpu_list(&gpus);

    // ─── Step 3: CLI argument handling ──────────────────────────────────
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--list" || a == "-l") {
        println!("  Quét GPU hoàn tất (--list).");
        return;
    }

    // ─── Step 4: Select GPU ─────────────────────────────────────────────
    let selection = match resolve_selection(&gpus, &args) {
        Some(idx) => idx,
        None => {
            println!("  {} Không có GPU được chọn — thoát.", "✗".red());
            std::process::exit(1);
        }
    };

    let selected_gpu = gpus[selection].clone();
    let physical_device = selected_gpu.physical_device;

    println!(
        "\n  {} Đã chọn: {}\n",
        "►".bright_cyan(),
        selected_gpu.name.bright_yellow().bold()
    );

    // ─── Step 5: Create context (compile shaders, get compute queue) ────
    let separator_thin = "─".repeat(70);
    println!("{}", separator_thin.dimmed());
    println!("  {}", "BẮT ĐẦU ĐÁNH GIÁ GPU".bold().bright_white());
    println!(
        "  Backend: {} | Device: {}",
        "Vulkan (SPIR-V)".bright_cyan(),
        selected_gpu.name.dimmed()
    );
    println!("{}", separator_thin.dimmed());

    println!(
        "\n  {} Đang biên dịch compute shaders và khởi tạo device...",
        "⚙".bright_cyan()
    );
    let ctx = match instance.create_context(physical_device) {
        Ok(c) => c,
        Err(e) => {
            println!(
                "\n  {} {}",
                "✗".red(),
                format!("Không tạo được Vulkan context: {}", e)
                    .red()
                    .bold()
            );
            std::process::exit(1);
        }
    };
    println!("  {} Device sẵn sàng.", "✓".bright_green());

    // ─── Step 6: Run benchmarks ─────────────────────────────────────────
    let mut results: Vec<BenchmarkResult> = Vec::new();
    results.push(run_vram(&ctx, &selected_gpu));
    results.push(run_matmul(&ctx));
    results.push(run_bandwidth(&ctx, &selected_gpu));
    results.push(run_stability(&ctx));
    results.push(run_precision(&ctx));

    // ─── Step 7: Print report ───────────────────────────────────────────
    let report = generate_report(&selected_gpu.name, results);
    print_report(&report);
}
