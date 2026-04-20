//! Top-level `oximedia quality` subcommand.
//!
//! Provides dedicated video/image quality assessment using full-reference metrics
//! (PSNR, SSIM, MS-SSIM, VIF, FSIM, VMAF) and no-reference metrics
//! (NIQE, BRISQUE, blockiness, blur, noise).
//!
//! Uses `oximedia-quality` for all metric computation.

use anyhow::{Context, Result};
use clap::Subcommand;
use colored::Colorize;
use oximedia_core::PixelFormat;
use oximedia_quality::{Frame, MetricType, QualityAssessor};
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Subcommand enum
// ---------------------------------------------------------------------------

/// Subcommands for `oximedia quality`.
#[derive(Subcommand, Debug)]
pub enum QualityCommand {
    /// Compare reference and distorted video/image files with full-reference metrics
    Compare {
        /// Reference (original) file
        #[arg(long)]
        reference: PathBuf,

        /// Distorted (encoded/processed) file
        #[arg(long)]
        distorted: PathBuf,

        /// Metrics to compute (comma-separated list)
        ///
        /// Full-reference: psnr, ssim, ms-ssim, vif, fsim, vmaf
        /// No-reference: niqe, brisque, blockiness, blur, noise
        #[arg(long, default_value = "psnr,ssim")]
        metrics: String,

        /// Output format: text or json
        #[arg(long, default_value = "text")]
        output_format: String,

        /// Frame dimensions width (pixels) — used for synthetic frame creation
        #[arg(long, default_value = "1920")]
        width: usize,

        /// Frame dimensions height (pixels)
        #[arg(long, default_value = "1080")]
        height: usize,
    },

    /// Analyze a single file with no-reference quality metrics
    Analyze {
        /// Input media file or image
        #[arg(short, long)]
        input: PathBuf,

        /// No-reference metrics to compute (comma-separated)
        ///
        /// Supported: niqe, brisque, blockiness, blur, noise
        #[arg(long, default_value = "brisque,blockiness,blur,noise")]
        metrics: String,

        /// Output format: text or json
        #[arg(long, default_value = "text")]
        output_format: String,
    },

    /// List all available quality metrics and their descriptions
    List,

    /// Explain a specific quality metric in detail
    Explain {
        /// Metric name (psnr, ssim, vmaf, brisque, etc.)
        #[arg(value_name = "METRIC")]
        metric: String,
    },
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Entry point called from `main.rs`.
pub async fn run_quality(command: QualityCommand, json_output: bool) -> Result<()> {
    match command {
        QualityCommand::Compare {
            reference,
            distorted,
            metrics,
            output_format,
            width,
            height,
        } => {
            let fmt = if json_output { "json" } else { &output_format };
            cmd_compare(&reference, &distorted, &metrics, fmt, width, height).await
        }

        QualityCommand::Analyze {
            input,
            metrics,
            output_format,
        } => {
            let fmt = if json_output { "json" } else { &output_format };
            cmd_analyze(&input, &metrics, fmt).await
        }

        QualityCommand::List => cmd_list(json_output),

        QualityCommand::Explain { metric } => cmd_explain(&metric, json_output),
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Parse a comma-separated metric list into `Vec<MetricType>`.
fn parse_metrics(metrics_str: &str) -> Result<Vec<MetricType>> {
    let mut result = Vec::new();
    for raw in metrics_str.split(',') {
        let name = raw.trim().to_lowercase();
        let metric = match name.as_str() {
            "psnr" => MetricType::Psnr,
            "ssim" => MetricType::Ssim,
            "ms-ssim" | "msssim" | "ms_ssim" => MetricType::MsSsim,
            "vmaf" => MetricType::Vmaf,
            "vif" => MetricType::Vif,
            "fsim" => MetricType::Fsim,
            "niqe" => MetricType::Niqe,
            "brisque" => MetricType::Brisque,
            "blockiness" | "block" => MetricType::Blockiness,
            "blur" => MetricType::Blur,
            "noise" => MetricType::Noise,
            other => {
                return Err(anyhow::anyhow!(
                    "Unknown metric '{other}'. Use `oximedia quality list` to see all available metrics."
                ))
            }
        };
        result.push(metric);
    }
    if result.is_empty() {
        return Err(anyhow::anyhow!("No metrics specified."));
    }
    Ok(result)
}

/// Create a synthetic grey frame of the given size (for demonstration when real
/// frame decoding is pending integration).
fn make_grey_frame(width: usize, height: usize) -> Result<Frame> {
    let mut frame = Frame::new(width, height, PixelFormat::Gray8)
        .map_err(|e| anyhow::anyhow!("Failed to allocate frame: {e}"))?;
    // Fill luma plane with mid-grey (128)
    frame.luma_mut().fill(128);
    Ok(frame)
}

/// Create a synthetic noisy frame (luma alternates 120/136) for distorted demo.
fn make_noisy_frame(width: usize, height: usize) -> Result<Frame> {
    let mut frame = Frame::new(width, height, PixelFormat::Gray8)
        .map_err(|e| anyhow::anyhow!("Failed to allocate frame: {e}"))?;
    let luma = frame.luma_mut();
    for (i, px) in luma.iter_mut().enumerate() {
        *px = if i % 2 == 0 { 120 } else { 136 };
    }
    Ok(frame)
}

/// Metric display name and scale description.
fn metric_display_info(metric: MetricType) -> (&'static str, &'static str) {
    match metric {
        MetricType::Psnr => ("PSNR", "dB (higher = better; ≥40 dB: excellent)"),
        MetricType::Ssim => ("SSIM", "0–1 (higher = better; ≥0.95: excellent)"),
        MetricType::MsSsim => ("MS-SSIM", "0–1 (higher = better)"),
        MetricType::Vmaf => ("VMAF", "0–100 (higher = better; ≥90: excellent)"),
        MetricType::Vif => ("VIF", "0–1 (higher = better)"),
        MetricType::Fsim => ("FSIM", "0–1 (higher = better)"),
        MetricType::Niqe => ("NIQE", "lower = better (natural images ~3–5)"),
        MetricType::Brisque => ("BRISQUE", "0–100 (lower = better)"),
        MetricType::Blockiness => ("Blockiness", "0–1 (lower = better)"),
        MetricType::Blur => ("Blur", "0–1 (lower = better)"),
        MetricType::Noise => ("Noise", "0–1 (lower = better)"),
        // Forward-compatible: handle any future variants added to the non-exhaustive enum
        _ => ("Unknown", "see oximedia quality list"),
    }
}

// ---------------------------------------------------------------------------
// Compare
// ---------------------------------------------------------------------------

async fn cmd_compare(
    reference: &PathBuf,
    distorted: &PathBuf,
    metrics_str: &str,
    output_format: &str,
    width: usize,
    height: usize,
) -> Result<()> {
    if !reference.exists() {
        return Err(anyhow::anyhow!(
            "Reference file not found: {}",
            reference.display()
        ));
    }
    if !distorted.exists() {
        return Err(anyhow::anyhow!(
            "Distorted file not found: {}",
            distorted.display()
        ));
    }

    let metrics = parse_metrics(metrics_str)?;

    // Validate all metrics are full-reference (or allowed for compare)
    for m in &metrics {
        if m.is_no_reference() {
            return Err(anyhow::anyhow!(
                "Metric '{m:?}' is a no-reference metric and cannot be used with `compare`. \
                 Use `oximedia quality analyze` instead."
            ));
        }
    }

    // Build synthetic reference and distorted frames for metric demonstration.
    // (Real pixel-accurate assessment requires frame decoding pipeline integration.)
    let ref_frame = make_grey_frame(width, height).context("Failed to create reference frame")?;
    let dist_frame = make_noisy_frame(width, height).context("Failed to create distorted frame")?;

    let assessor = QualityAssessor::new();
    let mut scores: Vec<(MetricType, f64, std::collections::HashMap<String, f64>)> = Vec::new();

    for &metric in &metrics {
        let score = assessor
            .assess(&ref_frame, &dist_frame, metric)
            .map_err(|e| anyhow::anyhow!("Metric {metric:?} calculation failed: {e}"))?;
        scores.push((metric, score.score, score.components.clone()));
    }

    if output_format == "json" {
        let results: Vec<serde_json::Value> = scores
            .iter()
            .map(|(metric, score, components)| {
                let (name, scale) = metric_display_info(*metric);
                serde_json::json!({
                    "metric": name,
                    "score": score,
                    "scale": scale,
                    "components": components,
                })
            })
            .collect();
        let output = serde_json::json!({
            "command": "quality compare",
            "reference": reference.display().to_string(),
            "distorted": distorted.display().to_string(),
            "frame_dimensions": { "width": width, "height": height },
            "metrics": results,
            "note": "Scores computed on synthetic frames; wire video decoder for pixel-accurate results."
        });
        let s = serde_json::to_string_pretty(&output).context("JSON serialization failed")?;
        println!("{s}");
        return Ok(());
    }

    // Human-readable
    println!("{}", "Quality Comparison".green().bold());
    println!("{}", "=".repeat(60));
    println!("{:20} {}", "Reference:", reference.display());
    println!("{:20} {}", "Distorted:", distorted.display());
    println!("{:20} {}×{}", "Frame size:", width, height);
    println!();
    println!("{}", "Results".cyan().bold());
    println!("{}", "-".repeat(60));
    println!("{:<12} {:>12}  Scale", "Metric", "Score");
    println!("{}", "-".repeat(60));

    for (metric, score, _components) in &scores {
        let (name, scale) = metric_display_info(*metric);
        let score_str = format!("{score:.4}").yellow().to_string();
        println!("{:<12} {:>12}  {}", name, score_str, scale.dimmed());
    }

    println!();
    println!(
        "{}",
        "Note: Scores are computed on synthetic frames for demonstration.".dimmed()
    );
    println!(
        "{}",
        "Integrate the video decoder pipeline for pixel-accurate results.".dimmed()
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// Analyze (no-reference)
// ---------------------------------------------------------------------------

async fn cmd_analyze(input: &PathBuf, metrics_str: &str, output_format: &str) -> Result<()> {
    if !input.exists() {
        return Err(anyhow::anyhow!("Input file not found: {}", input.display()));
    }

    let metrics = parse_metrics(metrics_str)?;

    // Warn if any full-reference metric was requested
    for m in &metrics {
        if m.requires_reference() {
            return Err(anyhow::anyhow!(
                "Metric '{m:?}' requires a reference file. Use `oximedia quality compare` instead."
            ));
        }
    }

    // Synthetic frame (grey) for no-reference metric demonstration
    let frame = make_grey_frame(1920, 1080).context("Failed to create analysis frame")?;
    let assessor = QualityAssessor::new();

    let mut scores: Vec<(MetricType, f64)> = Vec::new();
    for &metric in &metrics {
        let score = assessor
            .assess_no_reference(&frame, metric)
            .map_err(|e| anyhow::anyhow!("Metric {metric:?} failed: {e}"))?;
        scores.push((metric, score.score));
    }

    let file_size = std::fs::metadata(input)
        .with_context(|| format!("Cannot stat: {}", input.display()))
        .map(|m| m.len())
        .unwrap_or(0);

    if output_format == "json" {
        let results: Vec<serde_json::Value> = scores
            .iter()
            .map(|(metric, score)| {
                let (name, scale) = metric_display_info(*metric);
                serde_json::json!({
                    "metric": name,
                    "score": score,
                    "scale": scale,
                })
            })
            .collect();
        let output = serde_json::json!({
            "command": "quality analyze",
            "input": input.display().to_string(),
            "file_size_bytes": file_size,
            "metrics": results,
            "note": "Scores computed on synthetic frames; wire video decoder for pixel-accurate results."
        });
        let s = serde_json::to_string_pretty(&output).context("JSON serialization failed")?;
        println!("{s}");
        return Ok(());
    }

    println!("{}", "Quality Analysis (No-Reference)".green().bold());
    println!("{}", "=".repeat(60));
    println!("{:20} {}", "Input:", input.display());
    println!("{:20} {} bytes", "File size:", file_size);
    println!();
    println!("{}", "Results".cyan().bold());
    println!("{}", "-".repeat(60));
    println!("{:<14} {:>10}  Scale", "Metric", "Score");
    println!("{}", "-".repeat(60));

    for (metric, score) in &scores {
        let (name, scale) = metric_display_info(*metric);
        println!(
            "{:<14} {:>10.4}  {}",
            name,
            score.to_string().yellow(),
            scale.dimmed()
        );
    }

    println!();
    println!(
        "{}",
        "Note: Scores are computed on synthetic frames for demonstration.".dimmed()
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// List
// ---------------------------------------------------------------------------

fn cmd_list(json_output: bool) -> Result<()> {
    let metrics = [
        (
            MetricType::Psnr,
            "Full-reference",
            "Peak Signal-to-Noise Ratio",
        ),
        (
            MetricType::Ssim,
            "Full-reference",
            "Structural Similarity Index",
        ),
        (MetricType::MsSsim, "Full-reference", "Multi-Scale SSIM"),
        (
            MetricType::Vmaf,
            "Full-reference",
            "Video Multi-Method Assessment Fusion",
        ),
        (
            MetricType::Vif,
            "Full-reference",
            "Visual Information Fidelity",
        ),
        (
            MetricType::Fsim,
            "Full-reference",
            "Feature Similarity Index",
        ),
        (
            MetricType::Niqe,
            "No-reference",
            "Natural Image Quality Evaluator",
        ),
        (
            MetricType::Brisque,
            "No-reference",
            "Blind/Referenceless Image Spatial Quality Evaluator",
        ),
        (
            MetricType::Blockiness,
            "No-reference",
            "DCT-based blockiness detection",
        ),
        (
            MetricType::Blur,
            "No-reference",
            "Laplacian variance blur detection",
        ),
        (
            MetricType::Noise,
            "No-reference",
            "Spatial/temporal noise estimation",
        ),
    ];

    if json_output {
        let list: Vec<serde_json::Value> = metrics
            .iter()
            .map(|(m, kind, desc)| {
                let (name, scale) = metric_display_info(*m);
                serde_json::json!({
                    "name": name,
                    "kind": kind,
                    "description": desc,
                    "scale": scale,
                })
            })
            .collect();
        let result = serde_json::json!({ "metrics": list });
        let s = serde_json::to_string_pretty(&result).context("JSON serialization failed")?;
        println!("{s}");
        return Ok(());
    }

    println!("{}", "Available Quality Metrics".green().bold());
    println!("{}", "=".repeat(72));
    println!("{:<14} {:<18} Description", "Name", "Type");
    println!("{}", "-".repeat(72));

    for (m, kind, desc) in &metrics {
        let (name, _) = metric_display_info(*m);
        println!("{:<14} {:<18} {}", name, kind, desc);
    }

    println!();
    println!(
        "{}",
        "Full-reference metrics require both --reference and --distorted.".dimmed()
    );
    println!(
        "{}",
        "No-reference metrics work on a single --input file.".dimmed()
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// Explain
// ---------------------------------------------------------------------------

fn cmd_explain(metric_str: &str, json_output: bool) -> Result<()> {
    let metric_name = metric_str.trim().to_lowercase();
    let (metric, long_desc) = match metric_name.as_str() {
        "psnr" => (
            MetricType::Psnr,
            "PSNR (Peak Signal-to-Noise Ratio) measures the ratio between the \
             maximum possible power of a signal and the power of corrupting noise \
             that affects the fidelity of its representation. Expressed in decibels \
             (dB). Higher values indicate better quality. Typical values: ≥40 dB = \
             excellent, 30–40 dB = good, 20–30 dB = fair, <20 dB = poor. PSNR \
             correlates well with quality for compression artifacts but less so for \
             blurring or structural distortions.",
        ),
        "ssim" => (
            MetricType::Ssim,
            "SSIM (Structural Similarity Index) measures image quality by comparing \
             luminance, contrast, and structure between a reference and distorted \
             image. Ranges from -1 (inverse) to 1 (identical). Values ≥0.95 are \
             generally considered excellent. SSIM is more perceptually accurate than \
             PSNR for many types of distortion.",
        ),
        "ms-ssim" | "msssim" | "ms_ssim" => (
            MetricType::MsSsim,
            "MS-SSIM (Multi-Scale SSIM) extends SSIM by evaluating structural \
             similarity at multiple spatial scales. This better accounts for \
             the viewing distance and display resolution effects on perceptual quality.",
        ),
        "vmaf" => (
            MetricType::Vmaf,
            "VMAF (Video Multi-Method Assessment Fusion) is a full-reference \
             perceptual video quality metric developed by Netflix. It uses a \
             machine-learning model trained on human opinion scores, combining \
             VIF (Visual Information Fidelity), DLM (Detail Loss Metric), and \
             motion features. Score range: 0–100; ≥90 = excellent, 70–90 = good, \
             <70 = noticeable quality loss.",
        ),
        "vif" => (
            MetricType::Vif,
            "VIF (Visual Information Fidelity) quantifies the amount of visual \
             information preserved in the distorted image relative to the reference, \
             based on a natural scene statistics model in the wavelet domain. Values \
             range from 0 (no information) to 1 (perfect fidelity).",
        ),
        "fsim" => (
            MetricType::Fsim,
            "FSIM (Feature Similarity Index) measures quality by comparing salient \
             features (phase congruency and gradient magnitude) between reference and \
             distorted images. Ranges 0–1; higher is better.",
        ),
        "niqe" => (
            MetricType::Niqe,
            "NIQE (Natural Image Quality Evaluator) is a no-reference metric that \
             measures deviation from the statistical regularities of natural images \
             using a multivariate Gaussian model. Lower scores indicate more natural \
             (higher quality) images. Pristine images typically score 3–5.",
        ),
        "brisque" => (
            MetricType::Brisque,
            "BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator) is a \
             no-reference metric that uses a natural scene statistics model on \
             spatial domain features. Score range 0–100; lower is better quality. \
             Typically: <20 = excellent, 20–40 = good, 40–60 = fair, >60 = poor.",
        ),
        "blockiness" | "block" => (
            MetricType::Blockiness,
            "Blockiness detection quantifies DCT-based compression blocking artifacts \
             that appear as visible 8×8 or 16×16 block boundaries in H.264/AV1 \
             encoded video. Score 0–1; lower is better.",
        ),
        "blur" => (
            MetricType::Blur,
            "Blur detection uses Laplacian variance to measure the sharpness of an \
             image. Lower variance indicates more blur. Score 0–1; lower indicates \
             more blurring.",
        ),
        "noise" => (
            MetricType::Noise,
            "Noise estimation quantifies spatial and temporal noise artifacts. \
             Combines measurements of grain-like high-frequency components across \
             frames. Score 0–1; lower indicates less noise.",
        ),
        other => {
            return Err(anyhow::anyhow!(
            "Unknown metric '{other}'. Use `oximedia quality list` to see all available metrics."
        ))
        }
    };

    let (name, scale) = metric_display_info(metric);
    let kind = if metric.requires_reference() {
        "Full-reference"
    } else {
        "No-reference"
    };

    if json_output {
        let result = serde_json::json!({
            "metric": name,
            "kind": kind,
            "scale": scale,
            "description": long_desc,
        });
        let s = serde_json::to_string_pretty(&result).context("JSON serialization failed")?;
        println!("{s}");
        return Ok(());
    }

    println!("{} {}", "Metric:".green().bold(), name.yellow().bold());
    println!("{}", "=".repeat(60));
    println!("{:10} {}", "Type:", kind);
    println!("{:10} {}", "Scale:", scale);
    println!();
    println!("{}", "Description".cyan().bold());
    println!("{}", "-".repeat(60));

    // Word-wrap at 58 chars
    let mut line_buf = String::new();
    for word in long_desc.split_whitespace() {
        if line_buf.len() + word.len() + 1 > 58 {
            println!("  {line_buf}");
            line_buf = word.to_string();
        } else {
            if !line_buf.is_empty() {
                line_buf.push(' ');
            }
            line_buf.push_str(word);
        }
    }
    if !line_buf.is_empty() {
        println!("  {line_buf}");
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_metrics_valid() {
        let metrics = parse_metrics("psnr,ssim").expect("should parse");
        assert_eq!(metrics.len(), 2);
        assert!(metrics.contains(&MetricType::Psnr));
        assert!(metrics.contains(&MetricType::Ssim));
    }

    #[test]
    fn test_parse_metrics_no_reference() {
        let metrics = parse_metrics("brisque,blur,noise").expect("should parse");
        assert_eq!(metrics.len(), 3);
        for m in &metrics {
            assert!(m.is_no_reference());
        }
    }

    #[test]
    fn test_parse_metrics_unknown() {
        let result = parse_metrics("psnr,unknown_metric");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_metrics_empty() {
        let result = parse_metrics("  ");
        // space-only splits to one empty token → unknown
        assert!(result.is_err());
    }

    #[test]
    fn test_make_grey_frame() {
        let frame = make_grey_frame(320, 240).expect("should succeed");
        assert_eq!(frame.width, 320);
        assert_eq!(frame.height, 240);
        assert!(frame.luma().iter().all(|&p| p == 128));
    }

    #[test]
    fn test_make_noisy_frame() {
        let frame = make_noisy_frame(320, 240).expect("should succeed");
        assert_eq!(frame.width, 320);
        assert_eq!(frame.height, 240);
    }

    #[test]
    fn test_cmd_list_no_panic() {
        assert!(cmd_list(false).is_ok());
        assert!(cmd_list(true).is_ok());
    }

    #[test]
    fn test_cmd_explain_psnr() {
        assert!(cmd_explain("psnr", true).is_ok());
    }

    #[test]
    fn test_cmd_explain_vmaf() {
        assert!(cmd_explain("vmaf", false).is_ok());
    }

    #[test]
    fn test_cmd_explain_unknown() {
        assert!(cmd_explain("xyz_unknown", false).is_err());
    }

    #[tokio::test]
    async fn test_cmd_compare_missing_reference() {
        let result = cmd_compare(
            &std::env::temp_dir().join("nonexistent_ref_12345.mkv"),
            &std::env::temp_dir().join("nonexistent_dist_12345.mkv"),
            "psnr",
            "text",
            1920,
            1080,
        )
        .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_cmd_compare_no_reference_metric_rejected() {
        let dir = std::env::temp_dir();
        let ref_path = dir.join("oximedia_quality_ref_test.mkv");
        let dist_path = dir.join("oximedia_quality_dist_test.mkv");
        std::fs::write(&ref_path, b"ref").expect("write ok");
        std::fs::write(&dist_path, b"dist").expect("write ok");
        // brisque is no-reference — should be rejected in compare mode
        let result = cmd_compare(&ref_path, &dist_path, "brisque", "text", 1920, 1080).await;
        assert!(result.is_err());
        std::fs::remove_file(&ref_path).ok();
        std::fs::remove_file(&dist_path).ok();
    }

    #[tokio::test]
    async fn test_cmd_compare_psnr_ssim_json() {
        let dir = std::env::temp_dir();
        let ref_path = dir.join("oximedia_quality_ref2.mkv");
        let dist_path = dir.join("oximedia_quality_dist2.mkv");
        std::fs::write(&ref_path, b"ref").expect("write ok");
        std::fs::write(&dist_path, b"dist").expect("write ok");
        let result = cmd_compare(&ref_path, &dist_path, "psnr,ssim", "json", 64, 64).await;
        assert!(result.is_ok(), "unexpected error: {result:?}");
        std::fs::remove_file(&ref_path).ok();
        std::fs::remove_file(&dist_path).ok();
    }

    #[tokio::test]
    async fn test_cmd_analyze_missing_file() {
        let result = cmd_analyze(
            &std::env::temp_dir().join("nonexistent_analyze_12345.mkv"),
            "blur",
            "text",
        )
        .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_cmd_analyze_no_reference_json() {
        let dir = std::env::temp_dir();
        let path = dir.join("oximedia_quality_analyze_test.mkv");
        std::fs::write(&path, b"stub").expect("write ok");
        let result = cmd_analyze(&path, "blur,noise", "json").await;
        assert!(result.is_ok(), "unexpected error: {result:?}");
        std::fs::remove_file(&path).ok();
    }

    #[tokio::test]
    async fn test_cmd_analyze_full_ref_metric_rejected() {
        let dir = std::env::temp_dir();
        let path = dir.join("oximedia_quality_full_ref_reject.mkv");
        std::fs::write(&path, b"stub").expect("write ok");
        let result = cmd_analyze(&path, "psnr", "text").await;
        assert!(result.is_err());
        std::fs::remove_file(&path).ok();
    }
}
