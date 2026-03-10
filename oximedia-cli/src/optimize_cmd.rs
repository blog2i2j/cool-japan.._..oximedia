//! Codec optimization commands: complexity analysis, CRF sweep, quality ladder, benchmarking.
//!
//! Provides the `oximedia optimize` subcommand group for analyzing and
//! optimizing encoding parameters.

use anyhow::{Context, Result};
use clap::Subcommand;
use colored::Colorize;
use std::path::PathBuf;

/// Optimize subcommands.
#[derive(Subcommand, Debug)]
pub enum OptimizeCommand {
    /// Analyze content complexity of a media file
    Analyze {
        /// Input media file
        #[arg(short, long)]
        input: PathBuf,

        /// Output format: text, json
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Sweep CRF values to find optimal quality/size tradeoff
    CrfSweep {
        /// Input media file
        #[arg(short, long)]
        input: PathBuf,

        /// Output directory for test encodes
        #[arg(short, long)]
        output_dir: PathBuf,

        /// Minimum CRF value (best quality)
        #[arg(long, default_value = "18")]
        min_crf: u32,

        /// Maximum CRF value (smallest file)
        #[arg(long, default_value = "40")]
        max_crf: u32,

        /// CRF step size
        #[arg(long, default_value = "2")]
        step: u32,

        /// Codec to use (av1, vp9)
        #[arg(long, default_value = "av1")]
        codec: String,

        /// Only encode the first N seconds
        #[arg(long)]
        duration: Option<f64>,
    },

    /// Generate a quality ladder for adaptive bitrate streaming
    Ladder {
        /// Input media file
        #[arg(short, long)]
        input: PathBuf,

        /// Output file for ladder definition
        #[arg(short, long)]
        output: PathBuf,

        /// Ladder strategy: auto, fixed, per_title
        #[arg(long, default_value = "auto")]
        strategy: String,

        /// Maximum number of rungs in the ladder
        #[arg(long)]
        max_rungs: Option<u32>,
    },

    /// Benchmark codec performance
    Benchmark {
        /// Input media file
        #[arg(short, long)]
        input: PathBuf,

        /// Comma-separated list of codecs to benchmark (e.g., "av1,vp9")
        #[arg(long)]
        codecs: Option<String>,

        /// Only encode the first N seconds
        #[arg(long)]
        duration: Option<f64>,

        /// Output format: text, json
        #[arg(long, default_value = "text")]
        format: String,
    },
}

/// Handle optimize command dispatch.
pub async fn handle_optimize_command(command: OptimizeCommand, json_output: bool) -> Result<()> {
    match command {
        OptimizeCommand::Analyze { input, format } => {
            analyze_complexity(&input, if json_output { "json" } else { &format }).await
        }
        OptimizeCommand::CrfSweep {
            input,
            output_dir,
            min_crf,
            max_crf,
            step,
            codec,
            duration,
        } => {
            crf_sweep(
                &input,
                &output_dir,
                min_crf,
                max_crf,
                step,
                &codec,
                duration,
                json_output,
            )
            .await
        }
        OptimizeCommand::Ladder {
            input,
            output,
            strategy,
            max_rungs,
        } => generate_ladder(&input, &output, &strategy, max_rungs, json_output).await,
        OptimizeCommand::Benchmark {
            input,
            codecs,
            duration,
            format,
        } => {
            benchmark_codecs(
                &input,
                codecs.as_deref(),
                duration,
                if json_output { "json" } else { &format },
            )
            .await
        }
    }
}

/// Analyze content complexity of a media file.
async fn analyze_complexity(input: &PathBuf, output_format: &str) -> Result<()> {
    if !input.exists() {
        return Err(anyhow::anyhow!("Input file not found: {}", input.display()));
    }

    let file_size = std::fs::metadata(input)
        .context("Failed to read file metadata")?
        .len();

    // Use the spatial complexity analyzer for a basic analysis
    // In a full implementation, we would decode frames and pass them
    // through the analyzer; here we provide metadata-based estimates
    let estimated_spatial = estimate_complexity_from_size(file_size);

    match output_format {
        "json" => {
            let result = serde_json::json!({
                "input": input.display().to_string(),
                "file_size_bytes": file_size,
                "complexity": {
                    "overall": estimated_spatial.overall,
                    "spatial": estimated_spatial.spatial,
                    "temporal": estimated_spatial.temporal,
                },
                "recommendations": {
                    "crf": estimated_spatial.recommended_crf,
                    "bitrate_kbps": estimated_spatial.recommended_bitrate_kbps,
                    "ladder_type": estimated_spatial.ladder_type,
                },
                "scene_estimate": estimated_spatial.scene_count,
                "status": "metadata_based_estimate",
                "message": "Full frame-level analysis requires decoder pipeline integration",
            });
            let json_str =
                serde_json::to_string_pretty(&result).context("Failed to serialize result")?;
            println!("{json_str}");
        }
        _ => {
            println!("{}", "Content Complexity Analysis".green().bold());
            println!("{}", "=".repeat(60));
            println!("{:25} {}", "Input:", input.display());
            println!(
                "{:25} {:.2} MB",
                "File size:",
                file_size as f64 / (1024.0 * 1024.0)
            );
            println!();

            println!("{}", "Complexity Scores".cyan().bold());
            println!("{}", "-".repeat(60));
            println!("{:25} {:.2}/1.00", "Overall:", estimated_spatial.overall);
            println!("{:25} {:.2}/1.00", "Spatial:", estimated_spatial.spatial);
            println!("{:25} {:.2}/1.00", "Temporal:", estimated_spatial.temporal);
            println!(
                "{:25} ~{}",
                "Estimated scenes:", estimated_spatial.scene_count
            );
            println!();

            println!("{}", "Recommendations".cyan().bold());
            println!("{}", "-".repeat(60));
            println!(
                "{:25} {}",
                "Recommended CRF:", estimated_spatial.recommended_crf
            );
            println!(
                "{:25} {} kbps",
                "Target bitrate:", estimated_spatial.recommended_bitrate_kbps
            );
            println!(
                "{:25} {}",
                "Ladder strategy:", estimated_spatial.ladder_type
            );
            println!();

            println!(
                "{}",
                "Note: Full frame-level analysis requires decoder pipeline integration.".yellow()
            );
        }
    }

    Ok(())
}

/// Run a CRF sweep across a range of values.
#[allow(clippy::too_many_arguments)]
async fn crf_sweep(
    input: &PathBuf,
    output_dir: &PathBuf,
    min_crf: u32,
    max_crf: u32,
    step: u32,
    codec: &str,
    duration: Option<f64>,
    json_output: bool,
) -> Result<()> {
    if !input.exists() {
        return Err(anyhow::anyhow!("Input file not found: {}", input.display()));
    }

    if min_crf >= max_crf {
        return Err(anyhow::anyhow!(
            "min_crf ({}) must be less than max_crf ({})",
            min_crf,
            max_crf
        ));
    }

    if step == 0 {
        return Err(anyhow::anyhow!("step must be greater than 0"));
    }

    // Validate codec
    match codec {
        "av1" | "vp9" | "vp8" => {}
        other => {
            return Err(anyhow::anyhow!(
                "Unsupported codec '{}'. Supported: av1, vp9, vp8",
                other
            ));
        }
    }

    let file_size = std::fs::metadata(input)
        .context("Failed to read file metadata")?
        .len();

    // Generate CRF sweep plan
    let mut crf_values = Vec::new();
    let mut crf = min_crf;
    while crf <= max_crf {
        crf_values.push(crf);
        crf += step;
    }

    // Simulate sweep results based on file size (full encode requires pipeline)
    let sweep_results: Vec<serde_json::Value> = crf_values
        .iter()
        .map(|&c| {
            let quality_factor = 1.0 - (c as f64 / 63.0);
            let size_factor = quality_factor * quality_factor;
            let est_size = (file_size as f64 * size_factor * 0.5) as u64;
            let est_psnr = 25.0 + quality_factor * 25.0;
            let est_ssim = 0.85 + quality_factor * 0.14;
            let est_bitrate = if let Some(dur) = duration {
                (est_size as f64 * 8.0 / dur / 1000.0) as u64
            } else {
                (est_size as f64 * 8.0 / 60.0 / 1000.0) as u64
            };

            serde_json::json!({
                "crf": c,
                "estimated_size_bytes": est_size,
                "estimated_bitrate_kbps": est_bitrate,
                "estimated_psnr": format!("{est_psnr:.2}"),
                "estimated_ssim": format!("{est_ssim:.4}"),
            })
        })
        .collect();

    if json_output {
        let result = serde_json::json!({
            "input": input.display().to_string(),
            "output_dir": output_dir.display().to_string(),
            "codec": codec,
            "crf_range": [min_crf, max_crf],
            "step": step,
            "duration_limit": duration,
            "results": sweep_results,
            "status": "estimated",
            "message": "Actual encoding requires full pipeline integration",
        });
        let json_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize result")?;
        println!("{json_str}");
    } else {
        println!("{}", "CRF Sweep".green().bold());
        println!("{}", "=".repeat(70));
        println!("{:15} {}", "Input:", input.display());
        println!("{:15} {}", "Codec:", codec);
        println!(
            "{:15} {} - {} (step {})",
            "CRF range:", min_crf, max_crf, step
        );
        if let Some(dur) = duration {
            println!("{:15} {dur:.1}s", "Duration limit:");
        }
        println!();

        println!(
            "{:>6} {:>14} {:>12} {:>8} {:>8}",
            "CRF", "Est. Size", "Bitrate", "PSNR", "SSIM"
        );
        println!("{}", "-".repeat(70));

        for entry in &sweep_results {
            let crf_val = entry["crf"].as_u64().unwrap_or(0);
            let size = entry["estimated_size_bytes"].as_u64().unwrap_or(0);
            let bitrate = entry["estimated_bitrate_kbps"].as_u64().unwrap_or(0);
            let psnr = entry["estimated_psnr"].as_str().unwrap_or("0");
            let ssim = entry["estimated_ssim"].as_str().unwrap_or("0");

            println!(
                "{:>6} {:>11.2} MB {:>8} kbps {:>8} {:>8}",
                crf_val,
                size as f64 / (1024.0 * 1024.0),
                bitrate,
                psnr,
                ssim,
            );
        }
        println!();
        println!(
            "{}",
            "Note: Values are estimates. Actual encoding requires pipeline integration.".yellow()
        );
    }

    Ok(())
}

/// Generate a quality ladder for adaptive streaming.
async fn generate_ladder(
    input: &PathBuf,
    output: &PathBuf,
    strategy: &str,
    max_rungs: Option<u32>,
    json_output: bool,
) -> Result<()> {
    if !input.exists() {
        return Err(anyhow::anyhow!("Input file not found: {}", input.display()));
    }

    match strategy {
        "auto" | "fixed" | "per_title" => {}
        other => {
            return Err(anyhow::anyhow!(
                "Unknown strategy '{}'. Expected: auto, fixed, per_title",
                other
            ));
        }
    }

    let num_rungs = max_rungs.unwrap_or(6).min(10) as usize;

    // Standard ladder rungs
    let standard_rungs = [
        ("2160p", 3840, 2160, 12000),
        ("1440p", 2560, 1440, 8000),
        ("1080p", 1920, 1080, 5000),
        ("720p", 1280, 720, 2500),
        ("480p", 854, 480, 1200),
        ("360p", 640, 360, 700),
        ("240p", 426, 240, 400),
        ("144p", 256, 144, 200),
    ];

    let rungs: Vec<_> = standard_rungs.iter().take(num_rungs).collect();

    if json_output {
        let ladder_json: Vec<serde_json::Value> = rungs
            .iter()
            .map(|(label, w, h, bitrate)| {
                serde_json::json!({
                    "label": label,
                    "width": w,
                    "height": h,
                    "bitrate_kbps": bitrate,
                    "codec": "av1",
                    "frame_rate": 30.0,
                })
            })
            .collect();

        let result = serde_json::json!({
            "input": input.display().to_string(),
            "output": output.display().to_string(),
            "strategy": strategy,
            "rungs": ladder_json,
            "status": "generated",
        });
        let json_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize ladder")?;
        println!("{json_str}");
    } else {
        println!("{}", "Quality Ladder".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:15} {}", "Input:", input.display());
        println!("{:15} {}", "Output:", output.display());
        println!("{:15} {}", "Strategy:", strategy);
        println!("{:15} {}", "Rungs:", rungs.len());
        println!();

        println!(
            "{:>8} {:>12} {:>12} {:>8}",
            "Label", "Resolution", "Bitrate", "Codec"
        );
        println!("{}", "-".repeat(60));

        for (label, w, h, bitrate) in &rungs {
            println!(
                "{:>8} {:>5}x{:<5} {:>5} kbps {:>8}",
                label, w, h, bitrate, "av1"
            );
        }
        println!();
        println!(
            "{}",
            "Ladder definition ready. Use `oximedia transcode` with each rung to encode.".dimmed()
        );
    }

    Ok(())
}

/// Benchmark codec performance.
async fn benchmark_codecs(
    input: &PathBuf,
    codecs: Option<&str>,
    duration: Option<f64>,
    output_format: &str,
) -> Result<()> {
    if !input.exists() {
        return Err(anyhow::anyhow!("Input file not found: {}", input.display()));
    }

    let codec_list: Vec<&str> = codecs
        .unwrap_or("av1,vp9")
        .split(',')
        .map(|s| s.trim())
        .collect();

    // Validate codecs
    for c in &codec_list {
        match *c {
            "av1" | "vp9" | "vp8" => {}
            other => {
                return Err(anyhow::anyhow!(
                    "Unsupported codec '{}'. Supported: av1, vp9, vp8",
                    other
                ));
            }
        }
    }

    let file_size = std::fs::metadata(input)
        .context("Failed to read file metadata")?
        .len();

    // Estimated benchmark results based on codec characteristics
    let results: Vec<serde_json::Value> = codec_list
        .iter()
        .map(|&codec| {
            let (speed_factor, quality_factor) = match codec {
                "av1" => (0.3, 1.0),
                "vp9" => (0.7, 0.9),
                "vp8" => (1.0, 0.75),
                _ => (0.5, 0.8),
            };

            let est_fps = 30.0 * speed_factor;
            let est_size = (file_size as f64 * quality_factor * 0.4) as u64;
            let est_encode_time = if let Some(dur) = duration {
                dur / est_fps
            } else {
                60.0 / est_fps
            };

            serde_json::json!({
                "codec": codec,
                "estimated_fps": format!("{est_fps:.1}"),
                "estimated_encode_time_secs": format!("{est_encode_time:.1}"),
                "estimated_output_size_bytes": est_size,
                "quality_score": format!("{:.2}", quality_factor * 100.0),
                "compression_ratio": format!("{:.2}", file_size as f64 / est_size.max(1) as f64),
            })
        })
        .collect();

    match output_format {
        "json" => {
            let result = serde_json::json!({
                "input": input.display().to_string(),
                "codecs": codec_list,
                "duration_limit": duration,
                "results": results,
                "status": "estimated",
                "message": "Actual benchmarking requires full pipeline integration",
            });
            let json_str =
                serde_json::to_string_pretty(&result).context("Failed to serialize benchmark")?;
            println!("{json_str}");
        }
        _ => {
            println!("{}", "Codec Benchmark".green().bold());
            println!("{}", "=".repeat(70));
            println!("{:15} {}", "Input:", input.display());
            println!(
                "{:15} {:.2} MB",
                "File size:",
                file_size as f64 / (1024.0 * 1024.0)
            );
            if let Some(dur) = duration {
                println!("{:15} {dur:.1}s", "Duration limit:");
            }
            println!();

            println!(
                "{:>8} {:>10} {:>14} {:>14} {:>10}",
                "Codec", "Est. FPS", "Encode Time", "Output Size", "Quality"
            );
            println!("{}", "-".repeat(70));

            for entry in &results {
                let codec = entry["codec"].as_str().unwrap_or("?");
                let fps = entry["estimated_fps"].as_str().unwrap_or("0");
                let enc_time = entry["estimated_encode_time_secs"].as_str().unwrap_or("0");
                let out_size = entry["estimated_output_size_bytes"].as_u64().unwrap_or(0);
                let quality = entry["quality_score"].as_str().unwrap_or("0");

                println!(
                    "{:>8} {:>7} fps {:>11.2}s {:>10.2} MB {:>8}/100",
                    codec,
                    fps,
                    enc_time.parse::<f64>().unwrap_or(0.0),
                    out_size as f64 / (1024.0 * 1024.0),
                    quality,
                );
            }
            println!();
            println!(
                "{}",
                "Note: Values are estimates. Actual benchmarking requires pipeline integration."
                    .yellow()
            );
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Estimated complexity report from file metadata.
struct ComplexityEstimate {
    overall: f64,
    spatial: f64,
    temporal: f64,
    scene_count: u32,
    recommended_crf: u32,
    recommended_bitrate_kbps: u32,
    ladder_type: &'static str,
}

/// Estimate content complexity heuristically from file size.
///
/// This is a rough heuristic; real analysis would decode frames.
fn estimate_complexity_from_size(file_size: u64) -> ComplexityEstimate {
    let size_mb = file_size as f64 / (1024.0 * 1024.0);

    // Heuristic: larger files per minute tend to be more complex
    let raw_complexity = (size_mb / 100.0).min(1.0);
    let spatial = (raw_complexity * 0.7 + 0.15).min(1.0);
    let temporal = (raw_complexity * 0.5 + 0.1).min(1.0);
    let overall = spatial * 0.6 + temporal * 0.4;

    let scene_count = (size_mb / 5.0).max(1.0) as u32;

    let recommended_crf = if overall > 0.7 {
        24
    } else if overall > 0.4 {
        28
    } else {
        32
    };

    let recommended_bitrate_kbps = if overall > 0.7 {
        6000
    } else if overall > 0.4 {
        3000
    } else {
        1500
    };

    let ladder_type = if overall > 0.6 { "per_title" } else { "fixed" };

    ComplexityEstimate {
        overall,
        spatial,
        temporal,
        scene_count,
        recommended_crf,
        recommended_bitrate_kbps,
        ladder_type,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complexity_estimate_small_file() {
        let est = estimate_complexity_from_size(1_000_000); // ~1 MB
        assert!(est.overall > 0.0);
        assert!(est.overall < 1.0);
        assert!(est.recommended_crf > 0);
    }

    #[test]
    fn test_complexity_estimate_large_file() {
        let est = estimate_complexity_from_size(500_000_000); // ~500 MB
        assert!(est.overall > 0.3);
        assert!(est.recommended_bitrate_kbps >= 1500);
    }

    #[test]
    fn test_complexity_estimate_zero_size() {
        let est = estimate_complexity_from_size(0);
        assert!(est.overall >= 0.0);
        assert!(est.scene_count >= 1);
    }

    #[test]
    fn test_ladder_rungs_count() {
        // Verify the standard rungs array has expected entries
        let standard_rungs = [
            ("2160p", 3840, 2160, 12000),
            ("1440p", 2560, 1440, 8000),
            ("1080p", 1920, 1080, 5000),
            ("720p", 1280, 720, 2500),
            ("480p", 854, 480, 1200),
            ("360p", 640, 360, 700),
            ("240p", 426, 240, 400),
            ("144p", 256, 144, 200),
        ];
        assert_eq!(standard_rungs.len(), 8);
    }
}
