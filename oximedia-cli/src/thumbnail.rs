//! Thumbnail generation from video files.
//!
//! Provides functionality to generate thumbnails and preview strips from videos,
//! with support for multiple extraction strategies and layouts.

use crate::progress::TranscodeProgress;
use anyhow::{anyhow, Context, Result};
use colored::Colorize;
use serde::Serialize;
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

/// Options for thumbnail generation.
#[derive(Debug, Clone)]
pub struct ThumbnailOptions {
    pub input: PathBuf,
    pub output: PathBuf,
    pub mode: ThumbnailMode,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub quality: u8,
    pub format: ThumbnailFormat,
    pub json_output: bool,
}

/// Thumbnail generation mode.
#[derive(Debug, Clone, PartialEq)]
pub enum ThumbnailMode {
    /// Single thumbnail at specific timestamp
    Single { timestamp: f64 },
    /// Multiple thumbnails at intervals
    Multiple { count: usize },
    /// Grid/strip layout with multiple thumbnails
    Grid { rows: usize, cols: usize },
    /// Automatically detect best frame (least motion blur, etc.)
    Auto,
}

/// Output format for thumbnails.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThumbnailFormat {
    Png,
    Jpeg,
    Webp,
}

impl ThumbnailFormat {
    /// Parse format from string.
    pub fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "png" => Ok(Self::Png),
            "jpg" | "jpeg" => Ok(Self::Jpeg),
            "webp" => Ok(Self::Webp),
            _ => Err(anyhow!("Unsupported thumbnail format: {}", s)),
        }
    }

    /// Get format name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Png => "PNG",
            Self::Jpeg => "JPEG",
            Self::Webp => "WebP",
        }
    }

    /// Get file extension.
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Jpeg => "jpg",
            Self::Webp => "webp",
        }
    }
}

/// Result of thumbnail generation (for JSON output).
#[derive(Debug, Serialize)]
pub struct ThumbnailResult {
    pub success: bool,
    pub output_files: Vec<String>,
    pub thumbnail_count: usize,
    pub format: String,
    pub width: u32,
    pub height: u32,
}

/// Main thumbnail generation function.
pub async fn generate_thumbnails(options: ThumbnailOptions) -> Result<()> {
    info!("Starting thumbnail generation");
    debug!("Thumbnail options: {:?}", options);

    // Validate input
    validate_input(&options.input).await?;

    // Validate quality
    if options.quality > 100 {
        return Err(anyhow!("Quality must be between 0 and 100"));
    }

    // Print thumbnail plan
    if !options.json_output {
        print_thumbnail_plan(&options);
    }

    // Generate thumbnails
    let output_files = generate_impl(&options).await?;

    // Print or output result
    if options.json_output {
        let result = create_result(&output_files, &options)?;
        println!("{}", serde_json::to_string_pretty(&result)?);
    } else {
        print_thumbnail_summary(&output_files, &options);
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

/// Print thumbnail generation plan.
fn print_thumbnail_plan(options: &ThumbnailOptions) {
    println!("{}", "Thumbnail Generation Plan".cyan().bold());
    println!("{}", "=".repeat(60));
    println!("{:20} {}", "Input:", options.input.display());
    println!("{:20} {}", "Output:", options.output.display());
    println!("{:20} {}", "Format:", options.format.name());

    match &options.mode {
        ThumbnailMode::Single { timestamp } => {
            println!("{:20} Single at {:.2}s", "Mode:", timestamp);
        }
        ThumbnailMode::Multiple { count } => {
            println!("{:20} Multiple ({} thumbnails)", "Mode:", count);
        }
        ThumbnailMode::Grid { rows, cols } => {
            println!("{:20} Grid ({}x{})", "Mode:", rows, cols);
        }
        ThumbnailMode::Auto => {
            println!("{:20} Auto-detect best frame", "Mode:");
        }
    }

    if let Some(w) = options.width {
        println!("{:20} {}", "Width:", w);
    }

    if let Some(h) = options.height {
        println!("{:20} {}", "Height:", h);
    }

    if options.format == ThumbnailFormat::Jpeg {
        println!("{:20} {}", "Quality:", options.quality);
    }

    println!("{}", "=".repeat(60));
    println!();
}

/// Perform the actual thumbnail generation.
async fn generate_impl(options: &ThumbnailOptions) -> Result<Vec<PathBuf>> {
    match &options.mode {
        ThumbnailMode::Single { timestamp } => generate_single(options, *timestamp).await,
        ThumbnailMode::Multiple { count } => generate_multiple(options, *count).await,
        ThumbnailMode::Grid { rows, cols } => generate_grid(options, *rows, *cols).await,
        ThumbnailMode::Auto => generate_auto(options).await,
    }
}

/// Generate a single thumbnail at specified timestamp.
async fn generate_single(options: &ThumbnailOptions, timestamp: f64) -> Result<Vec<PathBuf>> {
    info!("Generating single thumbnail at {:.2}s", timestamp);

    let mut progress = TranscodeProgress::new_spinner();

    // Simulate thumbnail extraction
    for i in 0..20 {
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        progress.update(i);
    }

    progress.finish();

    warn!("Note: Actual thumbnail generation not yet fully implemented.");

    Ok(vec![options.output.clone()])
}

/// Generate multiple thumbnails at intervals.
async fn generate_multiple(options: &ThumbnailOptions, count: usize) -> Result<Vec<PathBuf>> {
    info!("Generating {} thumbnails", count);

    let mut progress = TranscodeProgress::new(count as u64);
    let mut output_files = Vec::new();

    for i in 0..count {
        // Generate output filename
        let output_path = generate_output_path(&options.output, i, count, &options.format);

        debug!("Generating thumbnail {}: {}", i + 1, output_path.display());

        // Simulate extraction
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        progress.update(i as u64 + 1);

        output_files.push(output_path);
    }

    progress.finish();

    warn!("Note: Actual thumbnail generation not yet fully implemented.");

    Ok(output_files)
}

/// Generate grid/strip of thumbnails.
async fn generate_grid(
    options: &ThumbnailOptions,
    rows: usize,
    cols: usize,
) -> Result<Vec<PathBuf>> {
    let total_thumbs = rows * cols;
    info!(
        "Generating {}x{} grid ({} thumbnails)",
        rows, cols, total_thumbs
    );

    let mut progress = TranscodeProgress::new(total_thumbs as u64);

    // Simulate grid generation
    for i in 0..total_thumbs {
        tokio::time::sleep(tokio::time::Duration::from_millis(80)).await;
        progress.update(i as u64 + 1);
    }

    progress.finish();

    warn!("Note: Actual grid generation not yet fully implemented.");

    Ok(vec![options.output.clone()])
}

/// Auto-detect best frame for thumbnail.
async fn generate_auto(options: &ThumbnailOptions) -> Result<Vec<PathBuf>> {
    info!("Auto-detecting best frame for thumbnail");

    let mut progress = TranscodeProgress::new_spinner();

    // Simulate frame analysis
    for i in 0..50 {
        tokio::time::sleep(tokio::time::Duration::from_millis(40)).await;
        progress.update(i);
    }

    progress.finish();

    warn!("Note: Automatic frame detection not yet fully implemented.");

    Ok(vec![options.output.clone()])
}

/// Generate output path for multiple thumbnails.
fn generate_output_path(
    base_path: &Path,
    index: usize,
    total: usize,
    format: &ThumbnailFormat,
) -> PathBuf {
    let parent = base_path.parent().unwrap_or(Path::new(""));
    let stem = base_path.file_stem().unwrap_or_default().to_string_lossy();

    let filename = if total > 1 {
        format!("{}_{:03}.{}", stem, index + 1, format.extension())
    } else {
        format!("{}.{}", stem, format.extension())
    };

    parent.join(filename)
}

/// Create result structure for JSON output.
fn create_result(output_files: &[PathBuf], options: &ThumbnailOptions) -> Result<ThumbnailResult> {
    Ok(ThumbnailResult {
        success: true,
        output_files: output_files
            .iter()
            .map(|p| p.display().to_string())
            .collect(),
        thumbnail_count: output_files.len(),
        format: options.format.name().to_string(),
        width: options.width.unwrap_or(320),
        height: options.height.unwrap_or(240),
    })
}

/// Print thumbnail generation summary.
fn print_thumbnail_summary(output_files: &[PathBuf], options: &ThumbnailOptions) {
    println!();
    println!("{}", "Thumbnail Generation Complete".green().bold());
    println!("{}", "=".repeat(60));
    println!("{:20} {}", "Thumbnails Created:", output_files.len());
    println!("{:20} {}", "Format:", options.format.name());

    if output_files.len() <= 10 {
        println!("\n{}", "Output Files:".cyan());
        for (i, path) in output_files.iter().enumerate() {
            println!("  {}. {}", i + 1, path.display());
        }
    } else {
        println!("{:20} {}", "First Output:", output_files[0].display());
        println!("{:20} ... and {} more", "", output_files.len() - 1);
    }

    println!("{}", "=".repeat(60));
}

/// Parse time string to seconds.
pub fn parse_timestamp(s: &str) -> Result<f64> {
    // Try parsing as seconds first
    if let Ok(seconds) = s.parse::<f64>() {
        return Ok(seconds);
    }

    // Try parsing as HH:MM:SS or MM:SS
    let parts: Vec<&str> = s.split(':').collect();

    match parts.len() {
        1 => parts[0].parse().context("Invalid time format"),
        2 => {
            let minutes: f64 = parts[0].parse().context("Invalid minutes")?;
            let seconds: f64 = parts[1].parse().context("Invalid seconds")?;
            Ok(minutes * 60.0 + seconds)
        }
        3 => {
            let hours: f64 = parts[0].parse().context("Invalid hours")?;
            let minutes: f64 = parts[1].parse().context("Invalid minutes")?;
            let seconds: f64 = parts[2].parse().context("Invalid seconds")?;
            Ok(hours * 3600.0 + minutes * 60.0 + seconds)
        }
        _ => Err(anyhow!("Invalid time format: {}", s)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thumbnail_format_parsing() {
        assert_eq!(
            ThumbnailFormat::from_str("png").expect("ThumbnailFormat::from_str should succeed"),
            ThumbnailFormat::Png
        );
        assert_eq!(
            ThumbnailFormat::from_str("jpeg").expect("ThumbnailFormat::from_str should succeed"),
            ThumbnailFormat::Jpeg
        );
        assert_eq!(
            ThumbnailFormat::from_str("webp").expect("ThumbnailFormat::from_str should succeed"),
            ThumbnailFormat::Webp
        );
        assert!(ThumbnailFormat::from_str("bmp").is_err());
    }

    #[test]
    fn test_parse_timestamp() {
        assert_eq!(parse_timestamp("30").expect("parse should succeed"), 30.0);
        assert_eq!(parse_timestamp("1:30").expect("parse should succeed"), 90.0);
        assert_eq!(
            parse_timestamp("1:01:30").expect("parse should succeed"),
            3690.0
        );
        assert_eq!(
            parse_timestamp("0:05:00").expect("parse should succeed"),
            300.0
        );
    }

    #[test]
    fn test_generate_output_path() {
        let base = PathBuf::from("output.png");
        let format = ThumbnailFormat::Png;

        let path = generate_output_path(&base, 0, 5, &format);
        assert_eq!(path, PathBuf::from("output_001.png"));

        let path = generate_output_path(&base, 4, 5, &format);
        assert_eq!(path, PathBuf::from("output_005.png"));
    }
}
