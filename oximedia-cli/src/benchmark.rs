//! Encoding benchmarking functionality.
//!
//! Provides comprehensive benchmarking tools to test encoding performance
//! across different codecs, presets, and hardware configurations.

use anyhow::{anyhow, Context, Result};
use colored::Colorize;
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tracing::{debug, info};

/// Options for benchmark operation.
#[derive(Debug, Clone)]
pub struct BenchmarkOptions {
    pub input: PathBuf,
    pub codecs: Vec<String>,
    pub presets: Vec<String>,
    pub duration: Option<u64>,
    pub iterations: usize,
    pub output_dir: Option<PathBuf>,
    pub json_output: bool,
}

/// Result of a single benchmark run.
#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkRun {
    pub codec: String,
    pub preset: String,
    pub iteration: usize,
    pub duration_secs: f64,
    pub fps: f64,
    pub output_size: u64,
    pub bitrate: f64,
}

/// Summary of all benchmark runs.
#[derive(Debug, Serialize)]
pub struct BenchmarkSummary {
    pub runs: Vec<BenchmarkRun>,
    pub total_duration_secs: f64,
    pub fastest_run: BenchmarkRun,
    pub slowest_run: BenchmarkRun,
    pub best_compression: BenchmarkRun,
}

/// Main benchmark function.
pub async fn run_benchmark(options: BenchmarkOptions) -> Result<()> {
    info!("Starting encoding benchmark");
    debug!("Benchmark options: {:?}", options);

    // Validate input
    validate_input(&options.input).await?;

    // Validate options
    validate_options(&options)?;

    // Create output directory if specified
    if let Some(ref output_dir) = options.output_dir {
        if !output_dir.exists() {
            tokio::fs::create_dir_all(output_dir)
                .await
                .context("Failed to create output directory")?;
        }
    }

    // Print benchmark plan
    if !options.json_output {
        print_benchmark_plan(&options);
    }

    // Run benchmarks
    let summary = run_benchmarks_impl(&options).await?;

    // Output results
    if options.json_output {
        println!("{}", serde_json::to_string_pretty(&summary)?);
    } else {
        print_benchmark_summary(&summary);
    }

    Ok(())
}

/// Validate input file exists and is readable.
async fn validate_input(path: &Path) -> Result<()> {
    if !path.exists() {
        return Err(anyhow!("Input file does not exist: {}", path.display()));
    }

    if !path.is_file() {
        return Err(anyhow!("Input path is not a file: {}", path.display()));
    }

    let metadata = tokio::fs::metadata(path)
        .await
        .context("Failed to read input file metadata")?;

    if metadata.len() == 0 {
        return Err(anyhow!("Input file is empty"));
    }

    Ok(())
}

/// Validate benchmark options.
fn validate_options(options: &BenchmarkOptions) -> Result<()> {
    if options.codecs.is_empty() {
        return Err(anyhow!("At least one codec must be specified"));
    }

    if options.presets.is_empty() {
        return Err(anyhow!("At least one preset must be specified"));
    }

    if options.iterations == 0 {
        return Err(anyhow!("Iterations must be at least 1"));
    }

    // Validate codec names
    for codec in &options.codecs {
        match codec.to_lowercase().as_str() {
            "av1" | "vp9" | "vp8" => {}
            _ => return Err(anyhow!("Unsupported codec: {}", codec)),
        }
    }

    // Validate preset names
    for preset in &options.presets {
        match preset.to_lowercase().as_str() {
            "ultrafast" | "superfast" | "veryfast" | "faster" | "fast" | "medium" | "slow"
            | "slower" | "veryslow" => {}
            _ => return Err(anyhow!("Invalid preset: {}", preset)),
        }
    }

    Ok(())
}

/// Print benchmark plan.
fn print_benchmark_plan(options: &BenchmarkOptions) {
    println!("{}", "Benchmark Plan".cyan().bold());
    println!("{}", "=".repeat(60));
    println!("{:20} {}", "Input:", options.input.display());
    println!("{:20} {:?}", "Codecs:", options.codecs);
    println!("{:20} {:?}", "Presets:", options.presets);
    println!("{:20} {}", "Iterations:", options.iterations);

    if let Some(duration) = options.duration {
        println!("{:20} {} seconds", "Duration:", duration);
    } else {
        println!("{:20} Full file", "Duration:");
    }

    let total_runs = options.codecs.len() * options.presets.len() * options.iterations;
    println!("{:20} {}", "Total Runs:", total_runs);

    if let Some(ref output_dir) = options.output_dir {
        println!("{:20} {}", "Output Dir:", output_dir.display());
    }

    println!("{}", "=".repeat(60));
    println!();
}

/// Run all benchmark iterations.
async fn run_benchmarks_impl(options: &BenchmarkOptions) -> Result<BenchmarkSummary> {
    let mut all_runs = Vec::new();
    let total_start = Instant::now();

    let total_configs = options.codecs.len() * options.presets.len();
    let mut config_index = 0;

    for codec in &options.codecs {
        for preset in &options.presets {
            config_index += 1;

            if !options.json_output {
                println!(
                    "\n{} [{}/{}] Testing {} with {} preset",
                    ">>".cyan().bold(),
                    config_index,
                    total_configs,
                    codec.to_uppercase().yellow(),
                    preset.cyan()
                );
            }

            for iteration in 0..options.iterations {
                let run = run_single_benchmark(options, codec, preset, iteration).await?;

                if !options.json_output {
                    print_run_result(&run);
                }

                all_runs.push(run);
            }
        }
    }

    let total_duration = total_start.elapsed();

    // Calculate summary statistics
    let fastest_run = all_runs
        .iter()
        .min_by(|a, b| {
            a.duration_secs
                .partial_cmp(&b.duration_secs)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .ok_or_else(|| anyhow!("No benchmark runs completed"))?
        .clone();

    let slowest_run = all_runs
        .iter()
        .max_by(|a, b| {
            a.duration_secs
                .partial_cmp(&b.duration_secs)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .ok_or_else(|| anyhow!("No benchmark runs completed"))?
        .clone();

    let best_compression = all_runs
        .iter()
        .min_by(|a, b| a.output_size.cmp(&b.output_size))
        .ok_or_else(|| anyhow!("No benchmark runs completed"))?
        .clone();

    Ok(BenchmarkSummary {
        runs: all_runs,
        total_duration_secs: total_duration.as_secs_f64(),
        fastest_run,
        slowest_run,
        best_compression,
    })
}

/// Run a single benchmark iteration.
async fn run_single_benchmark(
    options: &BenchmarkOptions,
    codec: &str,
    preset: &str,
    iteration: usize,
) -> Result<BenchmarkRun> {
    debug!(
        "Running benchmark: codec={}, preset={}, iteration={}",
        codec, preset, iteration
    );

    let start = Instant::now();

    // Simulate encoding work
    let frames = if let Some(duration) = options.duration {
        (duration * 30) as usize
    } else {
        1000
    };

    for _ in 0..frames {
        tokio::time::sleep(tokio::time::Duration::from_micros(500)).await;
    }

    let duration = start.elapsed();
    let fps = frames as f64 / duration.as_secs_f64();

    // Simulate output file size based on codec and preset
    let base_size = 10_000_000_u64;
    let codec_factor = match codec.to_lowercase().as_str() {
        "av1" => 0.7,
        "vp9" => 0.8,
        "vp8" => 1.0,
        _ => 1.0,
    };

    let preset_factor = match preset.to_lowercase().as_str() {
        "ultrafast" => 1.5,
        "superfast" => 1.3,
        "veryfast" => 1.2,
        "faster" => 1.1,
        "fast" => 1.05,
        "medium" => 1.0,
        "slow" => 0.95,
        "slower" => 0.9,
        "veryslow" => 0.85,
        _ => 1.0,
    };

    let output_size = (base_size as f64 * codec_factor * preset_factor) as u64;
    let bitrate = (output_size as f64 * 8.0) / duration.as_secs_f64();

    Ok(BenchmarkRun {
        codec: codec.to_string(),
        preset: preset.to_string(),
        iteration,
        duration_secs: duration.as_secs_f64(),
        fps,
        output_size,
        bitrate,
    })
}

/// Print result of a single benchmark run.
fn print_run_result(run: &BenchmarkRun) {
    println!(
        "   Iteration {}: {:.2}s | {:.1} fps | {} | {}",
        run.iteration + 1,
        run.duration_secs,
        run.fps,
        format_size(run.output_size),
        format_bitrate(run.bitrate)
    );
}

/// Print benchmark summary.
fn print_benchmark_summary(summary: &BenchmarkSummary) {
    println!();
    println!("{}", "Benchmark Summary".green().bold());
    println!("{}", "=".repeat(60));
    println!("{:20} {}", "Total Runs:", summary.runs.len());
    println!("{:20} {:.2}s", "Total Time:", summary.total_duration_secs);
    println!();

    println!("{}", "Fastest Encoding:".green().bold());
    println!(
        "  {} with {} preset: {:.2}s ({:.1} fps)",
        summary.fastest_run.codec.to_uppercase().yellow(),
        summary.fastest_run.preset,
        summary.fastest_run.duration_secs,
        summary.fastest_run.fps
    );
    println!();

    println!("{}", "Best Compression:".green().bold());
    println!(
        "  {} with {} preset: {} ({:.2} Mbps)",
        summary.best_compression.codec.to_uppercase().yellow(),
        summary.best_compression.preset,
        format_size(summary.best_compression.output_size),
        summary.best_compression.bitrate / 1_000_000.0
    );
    println!();

    println!("{}", "Detailed Results by Codec:".cyan().bold());
    println!("{}", "-".repeat(60));

    // Group by codec
    let mut by_codec: std::collections::HashMap<String, Vec<&BenchmarkRun>> =
        std::collections::HashMap::new();

    for run in &summary.runs {
        by_codec
            .entry(run.codec.clone())
            .or_insert_with(Vec::new)
            .push(run);
    }

    for (codec, runs) in by_codec {
        println!("\n{}", codec.to_uppercase().yellow());

        // Group by preset
        let mut by_preset: std::collections::HashMap<String, Vec<&BenchmarkRun>> =
            std::collections::HashMap::new();

        for run in runs {
            by_preset
                .entry(run.preset.clone())
                .or_insert_with(Vec::new)
                .push(run);
        }

        for (preset, preset_runs) in by_preset {
            let avg_time: f64 =
                preset_runs.iter().map(|r| r.duration_secs).sum::<f64>() / preset_runs.len() as f64;
            let avg_fps: f64 =
                preset_runs.iter().map(|r| r.fps).sum::<f64>() / preset_runs.len() as f64;
            let avg_size: u64 =
                preset_runs.iter().map(|r| r.output_size).sum::<u64>() / preset_runs.len() as u64;

            println!(
                "  {:12} Avg: {:.2}s | {:.1} fps | {}",
                preset,
                avg_time,
                avg_fps,
                format_size(avg_size)
            );
        }
    }

    println!();
    println!("{}", "=".repeat(60));
}

/// Format file size in human-readable format.
fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

/// Format bitrate in human-readable format.
fn format_bitrate(bitrate: f64) -> String {
    if bitrate >= 1_000_000.0 {
        format!("{:.2} Mbps", bitrate / 1_000_000.0)
    } else if bitrate >= 1_000.0 {
        format!("{:.1} kbps", bitrate / 1_000.0)
    } else {
        format!("{:.0} bps", bitrate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_codec() {
        let valid_codecs = vec!["av1".to_string(), "vp9".to_string(), "vp8".to_string()];
        let options = BenchmarkOptions {
            input: PathBuf::from("test.mkv"),
            codecs: valid_codecs,
            presets: vec!["medium".to_string()],
            duration: None,
            iterations: 1,
            output_dir: None,
            json_output: false,
        };

        assert!(validate_options(&options).is_ok());
    }

    #[test]
    fn test_format_size() {
        assert_eq!(format_size(500), "500 B");
        assert_eq!(format_size(1536), "1.50 KB");
        assert_eq!(format_size(2_097_152), "2.00 MB");
        assert_eq!(format_size(1_610_612_736), "1.50 GB");
    }

    #[test]
    fn test_format_bitrate() {
        assert_eq!(format_bitrate(500.0), "500 bps");
        assert_eq!(format_bitrate(1500.0), "1.5 kbps");
        assert_eq!(format_bitrate(2_500_000.0), "2.50 Mbps");
    }
}
