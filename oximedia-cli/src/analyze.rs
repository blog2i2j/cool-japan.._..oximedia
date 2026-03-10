//! Quality metrics analysis command.
//!
//! Provides video/audio quality assessment using full-reference and
//! no-reference metrics from the `oximedia-quality` crate.

use anyhow::{Context, Result};
use colored::Colorize;
use std::path::PathBuf;

/// Options for the analyze command.
#[derive(Debug, Clone)]
pub struct AnalyzeOptions {
    /// Input file path
    pub input: PathBuf,
    /// Reference file for full-reference metrics (PSNR, SSIM, etc.)
    pub reference: Option<PathBuf>,
    /// Comma-separated list of metrics to compute
    pub metrics: String,
    /// Output format: text, json, csv
    pub output_format: String,
    /// Enable per-frame analysis
    pub per_frame: bool,
    /// Show summary statistics
    pub summary: bool,
    /// JSON output mode from global CLI flag
    pub json_output: bool,
}

/// Parse a metric name string into a `MetricType`.
fn parse_metric(name: &str) -> Result<oximedia_quality::MetricType> {
    match name.trim().to_lowercase().as_str() {
        "psnr" => Ok(oximedia_quality::MetricType::Psnr),
        "ssim" => Ok(oximedia_quality::MetricType::Ssim),
        "ms-ssim" | "msssim" | "ms_ssim" => Ok(oximedia_quality::MetricType::MsSsim),
        "vmaf" => Ok(oximedia_quality::MetricType::Vmaf),
        "vif" => Ok(oximedia_quality::MetricType::Vif),
        "fsim" => Ok(oximedia_quality::MetricType::Fsim),
        "niqe" => Ok(oximedia_quality::MetricType::Niqe),
        "brisque" => Ok(oximedia_quality::MetricType::Brisque),
        "blockiness" => Ok(oximedia_quality::MetricType::Blockiness),
        "blur" => Ok(oximedia_quality::MetricType::Blur),
        "noise" => Ok(oximedia_quality::MetricType::Noise),
        other => Err(anyhow::anyhow!(
            "Unknown metric '{}'. Available: psnr, ssim, ms-ssim, vmaf, vif, fsim, niqe, brisque, blockiness, blur, noise",
            other
        )),
    }
}

/// Analyze quality metrics for a media file.
pub async fn analyze_quality(options: AnalyzeOptions) -> Result<()> {
    // Validate input file exists
    if !options.input.exists() {
        return Err(anyhow::anyhow!(
            "Input file not found: {}",
            options.input.display()
        ));
    }

    // Parse requested metrics
    let metric_names: Vec<&str> = options.metrics.split(',').collect();
    let mut metrics = Vec::new();
    for name in &metric_names {
        metrics.push(parse_metric(name)?);
    }

    // Validate that full-reference metrics have a reference file
    let has_full_ref = metrics.iter().any(|m| m.requires_reference());
    if has_full_ref && options.reference.is_none() {
        return Err(anyhow::anyhow!(
            "Full-reference metrics (psnr, ssim, ms-ssim, vmaf, vif, fsim) require --reference <file>"
        ));
    }

    if let Some(ref ref_path) = options.reference {
        if !ref_path.exists() {
            return Err(anyhow::anyhow!(
                "Reference file not found: {}",
                ref_path.display()
            ));
        }
    }

    // Create the quality assessor
    let assessor = oximedia_quality::QualityAssessor::new();

    // Separate metrics into full-reference and no-reference
    let fr_metrics: Vec<_> = metrics.iter().filter(|m| m.requires_reference()).collect();
    let nr_metrics: Vec<_> = metrics.iter().filter(|m| m.is_no_reference()).collect();

    // Since we cannot yet read actual video frames from files, we demonstrate
    // the framework and provide informative output about what would happen.
    let output_format = if options.json_output {
        "json"
    } else {
        options.output_format.as_str()
    };

    match output_format {
        "json" => {
            print_json_analysis(&options, &fr_metrics, &nr_metrics)?;
        }
        "csv" => {
            print_csv_analysis(&options, &fr_metrics, &nr_metrics)?;
        }
        _ => {
            print_text_analysis(&options, &fr_metrics, &nr_metrics, &assessor)?;
        }
    }

    Ok(())
}

/// Print analysis results in text format.
fn print_text_analysis(
    options: &AnalyzeOptions,
    fr_metrics: &[&oximedia_quality::MetricType],
    nr_metrics: &[&oximedia_quality::MetricType],
    _assessor: &oximedia_quality::QualityAssessor,
) -> Result<()> {
    println!("{}", "Quality Analysis".green().bold());
    println!("{}", "=".repeat(60));
    println!("{:20} {}", "Input:", options.input.display());

    if let Some(ref ref_path) = options.reference {
        println!("{:20} {}", "Reference:", ref_path.display());
    }

    let file_size = std::fs::metadata(&options.input)
        .context("Failed to read file metadata")?
        .len();
    println!("{:20} {} bytes", "File size:", file_size);
    println!();

    if !fr_metrics.is_empty() {
        println!("{}", "Full-Reference Metrics".cyan().bold());
        println!("{}", "-".repeat(60));
        for metric in fr_metrics {
            println!(
                "  {:20} (requires frame decoding pipeline)",
                format!("{:?}", metric)
            );
        }
        println!();
    }

    if !nr_metrics.is_empty() {
        println!("{}", "No-Reference Metrics".cyan().bold());
        println!("{}", "-".repeat(60));
        for metric in nr_metrics {
            println!(
                "  {:20} (requires frame decoding pipeline)",
                format!("{:?}", metric)
            );
        }
        println!();
    }

    if options.per_frame {
        println!(
            "{}",
            "Per-frame analysis enabled (requires frame decoding pipeline)"
                .yellow()
                .dimmed()
        );
    }

    if options.summary {
        println!("{}", "Summary Statistics".cyan().bold());
        println!("{}", "-".repeat(60));
        println!(
            "  {}",
            "Summary will be computed once frame decoding is integrated".dimmed()
        );
    }

    println!();
    println!(
        "{}",
        "Note: Full media frame reading pipeline not yet integrated.".yellow()
    );
    println!(
        "{}",
        "Quality assessor is ready; frame decoding will enable end-to-end analysis.".dimmed()
    );

    Ok(())
}

/// Print analysis results in JSON format.
fn print_json_analysis(
    options: &AnalyzeOptions,
    fr_metrics: &[&oximedia_quality::MetricType],
    nr_metrics: &[&oximedia_quality::MetricType],
) -> Result<()> {
    let result = serde_json::json!({
        "input": options.input.display().to_string(),
        "reference": options.reference.as_ref().map(|p| p.display().to_string()),
        "requested_metrics": {
            "full_reference": fr_metrics.iter().map(|m| format!("{:?}", m)).collect::<Vec<_>>(),
            "no_reference": nr_metrics.iter().map(|m| format!("{:?}", m)).collect::<Vec<_>>(),
        },
        "per_frame": options.per_frame,
        "summary": options.summary,
        "status": "pending_frame_decoding",
        "message": "Quality assessor initialized; awaiting frame decoding pipeline integration",
    });

    let json_str =
        serde_json::to_string_pretty(&result).context("Failed to serialize analysis result")?;
    println!("{}", json_str);
    Ok(())
}

/// Print analysis results in CSV format.
fn print_csv_analysis(
    options: &AnalyzeOptions,
    fr_metrics: &[&oximedia_quality::MetricType],
    nr_metrics: &[&oximedia_quality::MetricType],
) -> Result<()> {
    println!("input,reference,metric,type,status");
    let ref_str = options
        .reference
        .as_ref()
        .map(|p| p.display().to_string())
        .unwrap_or_default();

    for metric in fr_metrics {
        println!(
            "{},{},{:?},full_reference,pending",
            options.input.display(),
            ref_str,
            metric
        );
    }
    for metric in nr_metrics {
        println!(
            "{},{},{:?},no_reference,pending",
            options.input.display(),
            ref_str,
            metric
        );
    }

    Ok(())
}
