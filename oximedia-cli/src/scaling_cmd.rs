//! Video/image scaling commands: upscale, downscale, analyze, compare, batch.
//!
//! Exposes `oximedia-scaling` Lanczos, bicubic, bilinear scaling with
//! quality-aware algorithms via the CLI.

use anyhow::{Context, Result};
use clap::Subcommand;
use colored::Colorize;
use std::path::PathBuf;

/// Scaling command subcommands.
#[derive(Subcommand, Debug)]
pub enum ScalingCommand {
    /// Upscale a video or image
    Upscale {
        /// Input file
        #[arg(short, long)]
        input: PathBuf,

        /// Output file
        #[arg(short, long)]
        output: PathBuf,

        /// Target width
        #[arg(long)]
        width: u32,

        /// Target height
        #[arg(long)]
        height: u32,

        /// Scaling algorithm: bilinear, bicubic, lanczos
        #[arg(long, default_value = "lanczos")]
        algorithm: String,

        /// Aspect ratio mode: stretch, letterbox, crop
        #[arg(long, default_value = "letterbox")]
        aspect: String,
    },

    /// Downscale a video or image
    Downscale {
        /// Input file
        #[arg(short, long)]
        input: PathBuf,

        /// Output file
        #[arg(short, long)]
        output: PathBuf,

        /// Target width
        #[arg(long)]
        width: u32,

        /// Target height
        #[arg(long)]
        height: u32,

        /// Scaling algorithm: bilinear, bicubic, lanczos
        #[arg(long, default_value = "lanczos")]
        algorithm: String,

        /// Aspect ratio mode: stretch, letterbox, crop
        #[arg(long, default_value = "letterbox")]
        aspect: String,
    },

    /// Analyze scaling quality for a given source/target resolution pair
    Analyze {
        /// Source width
        #[arg(long)]
        src_width: u32,

        /// Source height
        #[arg(long)]
        src_height: u32,

        /// Target width
        #[arg(long)]
        dst_width: u32,

        /// Target height
        #[arg(long)]
        dst_height: u32,

        /// Algorithm to analyze: bilinear, bicubic, lanczos
        #[arg(long, default_value = "lanczos")]
        algorithm: String,
    },

    /// Compare scaling algorithms on an input
    Compare {
        /// Input file
        #[arg(short, long)]
        input: PathBuf,

        /// Target width
        #[arg(long)]
        width: u32,

        /// Target height
        #[arg(long)]
        height: u32,

        /// Output directory for comparison results
        #[arg(short, long)]
        output_dir: Option<PathBuf>,
    },

    /// Batch scale multiple files
    Batch {
        /// Input directory
        #[arg(short, long)]
        input_dir: PathBuf,

        /// Output directory
        #[arg(short, long)]
        output_dir: PathBuf,

        /// Target width
        #[arg(long)]
        width: u32,

        /// Target height
        #[arg(long)]
        height: u32,

        /// Scaling algorithm
        #[arg(long, default_value = "lanczos")]
        algorithm: String,

        /// File extension filter (e.g. "png", "jpg")
        #[arg(long)]
        ext: Option<String>,
    },
}

/// Handle scaling command dispatch.
pub async fn handle_scaling_command(command: ScalingCommand, json_output: bool) -> Result<()> {
    match command {
        ScalingCommand::Upscale {
            input,
            output,
            width,
            height,
            algorithm,
            aspect,
        } => {
            handle_scale(
                &input,
                &output,
                width,
                height,
                &algorithm,
                &aspect,
                "upscale",
                json_output,
            )
            .await
        }
        ScalingCommand::Downscale {
            input,
            output,
            width,
            height,
            algorithm,
            aspect,
        } => {
            handle_scale(
                &input,
                &output,
                width,
                height,
                &algorithm,
                &aspect,
                "downscale",
                json_output,
            )
            .await
        }
        ScalingCommand::Analyze {
            src_width,
            src_height,
            dst_width,
            dst_height,
            algorithm,
        } => {
            handle_analyze(
                src_width,
                src_height,
                dst_width,
                dst_height,
                &algorithm,
                json_output,
            )
            .await
        }
        ScalingCommand::Compare {
            input,
            width,
            height,
            output_dir,
        } => handle_compare(&input, width, height, output_dir.as_deref(), json_output).await,
        ScalingCommand::Batch {
            input_dir,
            output_dir,
            width,
            height,
            algorithm,
            ext,
        } => {
            handle_batch(
                &input_dir,
                &output_dir,
                width,
                height,
                &algorithm,
                ext.as_deref(),
                json_output,
            )
            .await
        }
    }
}

/// Parse scaling mode from string.
fn parse_scaling_mode(s: &str) -> Result<oximedia_scaling::ScalingMode> {
    match s {
        "bilinear" => Ok(oximedia_scaling::ScalingMode::Bilinear),
        "bicubic" => Ok(oximedia_scaling::ScalingMode::Bicubic),
        "lanczos" => Ok(oximedia_scaling::ScalingMode::Lanczos),
        other => Err(anyhow::anyhow!(
            "Unknown algorithm '{}'. Supported: bilinear, bicubic, lanczos",
            other
        )),
    }
}

/// Parse aspect ratio mode from string.
fn parse_aspect_mode(s: &str) -> Result<oximedia_scaling::AspectRatioMode> {
    match s {
        "stretch" => Ok(oximedia_scaling::AspectRatioMode::Stretch),
        "letterbox" => Ok(oximedia_scaling::AspectRatioMode::Letterbox),
        "crop" => Ok(oximedia_scaling::AspectRatioMode::Crop),
        other => Err(anyhow::anyhow!(
            "Unknown aspect mode '{}'. Supported: stretch, letterbox, crop",
            other
        )),
    }
}

/// Handle upscale or downscale.
#[allow(clippy::too_many_arguments)]
async fn handle_scale(
    input: &PathBuf,
    output: &PathBuf,
    width: u32,
    height: u32,
    algorithm: &str,
    aspect: &str,
    direction: &str,
    json_output: bool,
) -> Result<()> {
    if !input.exists() {
        return Err(anyhow::anyhow!("Input file not found: {}", input.display()));
    }

    if width == 0 || height == 0 {
        return Err(anyhow::anyhow!(
            "Target dimensions must be > 0, got {}x{}",
            width,
            height
        ));
    }

    if width > 7680 || height > 4320 {
        return Err(anyhow::anyhow!(
            "Target dimensions exceed maximum 7680x4320, got {}x{}",
            width,
            height
        ));
    }

    let mode = parse_scaling_mode(algorithm)?;
    let aspect_mode = parse_aspect_mode(aspect)?;

    let params = oximedia_scaling::ScalingParams::new(width, height)
        .with_mode(mode)
        .with_aspect_ratio(aspect_mode);

    let scaler = oximedia_scaling::VideoScaler::new(params);

    let file_size = std::fs::metadata(input)
        .context("Failed to read file metadata")?
        .len();

    if json_output {
        let result = serde_json::json!({
            "command": direction,
            "input": input.display().to_string(),
            "output": output.display().to_string(),
            "file_size": file_size,
            "target_width": width,
            "target_height": height,
            "algorithm": algorithm,
            "aspect_mode": aspect,
            "scaler_params": {
                "width": scaler.params().width,
                "height": scaler.params().height,
                "mode": format!("{}", scaler.params().mode),
            },
            "status": "configured",
        });
        let json_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize scaling config")?;
        println!("{}", json_str);
    } else {
        let title = if direction == "upscale" {
            "Video Upscale"
        } else {
            "Video Downscale"
        };
        println!("{}", title.green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Input:", input.display());
        println!("{:20} {}", "Output:", output.display());
        println!("{:20} {} bytes", "File size:", file_size);
        println!("{:20} {}x{}", "Target:", width, height);
        println!("{:20} {}", "Algorithm:", algorithm);
        println!("{:20} {}", "Aspect mode:", aspect);
        println!();
        println!(
            "{}",
            "Scaling pipeline configured. Awaiting frame input.".dimmed()
        );
    }

    Ok(())
}

/// Analyze scaling quality.
async fn handle_analyze(
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    algorithm: &str,
    json_output: bool,
) -> Result<()> {
    let mode = parse_scaling_mode(algorithm)?;

    if src_width == 0 || src_height == 0 || dst_width == 0 || dst_height == 0 {
        return Err(anyhow::anyhow!("All dimensions must be > 0"));
    }

    let params = oximedia_scaling::ScalingParams::new(dst_width, dst_height).with_mode(mode);
    let scaler = oximedia_scaling::VideoScaler::new(params);
    let (calc_w, calc_h) = scaler.calculate_dimensions(src_width, src_height);

    let scale_factor_x = dst_width as f64 / src_width as f64;
    let scale_factor_y = dst_height as f64 / src_height as f64;
    let is_upscale = scale_factor_x > 1.0 || scale_factor_y > 1.0;

    if json_output {
        let result = serde_json::json!({
            "command": "analyze",
            "source": format!("{}x{}", src_width, src_height),
            "target": format!("{}x{}", dst_width, dst_height),
            "calculated": format!("{}x{}", calc_w, calc_h),
            "scale_factor_x": scale_factor_x,
            "scale_factor_y": scale_factor_y,
            "is_upscale": is_upscale,
            "algorithm": algorithm,
            "quality_assessment": if is_upscale { "upscale may introduce artifacts" } else { "downscale preserves detail" },
        });
        let json_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize analysis")?;
        println!("{}", json_str);
    } else {
        println!("{}", "Scaling Analysis".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}x{}", "Source:", src_width, src_height);
        println!("{:20} {}x{}", "Target:", dst_width, dst_height);
        println!("{:20} {}x{}", "Calculated:", calc_w, calc_h);
        println!(
            "{:20} {:.3}x / {:.3}x",
            "Scale factor:", scale_factor_x, scale_factor_y
        );
        println!(
            "{:20} {}",
            "Direction:",
            if is_upscale { "upscale" } else { "downscale" }
        );
        println!("{:20} {}", "Algorithm:", algorithm);
        println!();
        if is_upscale {
            println!(
                "{}",
                "Note: Upscaling may introduce interpolation artifacts.".yellow()
            );
            println!(
                "{}",
                "Lanczos provides the best quality for upscaling.".dimmed()
            );
        } else {
            println!(
                "{}",
                "Downscaling preserves visual detail well with anti-aliasing.".green()
            );
        }
    }

    Ok(())
}

/// Compare scaling algorithms.
async fn handle_compare(
    input: &PathBuf,
    width: u32,
    height: u32,
    output_dir: Option<&std::path::Path>,
    json_output: bool,
) -> Result<()> {
    if !input.exists() {
        return Err(anyhow::anyhow!("Input file not found: {}", input.display()));
    }

    let algorithms = ["bilinear", "bicubic", "lanczos"];

    if json_output {
        let comparisons: Vec<serde_json::Value> = algorithms
            .iter()
            .map(|alg| {
                serde_json::json!({
                    "algorithm": alg,
                    "target": format!("{}x{}", width, height),
                })
            })
            .collect();
        let result = serde_json::json!({
            "command": "compare",
            "input": input.display().to_string(),
            "target": format!("{}x{}", width, height),
            "output_dir": output_dir.map(|p| p.display().to_string()),
            "comparisons": comparisons,
            "status": "configured",
        });
        let json_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize comparison")?;
        println!("{}", json_str);
    } else {
        println!("{}", "Scaling Algorithm Comparison".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Input:", input.display());
        println!("{:20} {}x{}", "Target:", width, height);
        if let Some(d) = output_dir {
            println!("{:20} {}", "Output dir:", d.display());
        }
        println!();
        println!("{}", "Algorithms to compare".cyan().bold());
        println!("{}", "-".repeat(60));
        for alg in &algorithms {
            println!("  - {} ({}x{})", alg, width, height);
        }
        println!();
        println!(
            "{}",
            "Comparison pipeline configured. Awaiting frame input.".dimmed()
        );
    }

    Ok(())
}

/// Batch scale multiple files.
#[allow(clippy::too_many_arguments)]
async fn handle_batch(
    input_dir: &PathBuf,
    output_dir: &PathBuf,
    width: u32,
    height: u32,
    algorithm: &str,
    ext: Option<&str>,
    json_output: bool,
) -> Result<()> {
    if !input_dir.exists() {
        return Err(anyhow::anyhow!(
            "Input directory not found: {}",
            input_dir.display()
        ));
    }

    let _ = parse_scaling_mode(algorithm)?;

    if json_output {
        let result = serde_json::json!({
            "command": "batch",
            "input_dir": input_dir.display().to_string(),
            "output_dir": output_dir.display().to_string(),
            "target": format!("{}x{}", width, height),
            "algorithm": algorithm,
            "extension_filter": ext,
            "status": "configured",
        });
        let json_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize batch config")?;
        println!("{}", json_str);
    } else {
        println!("{}", "Batch Scaling".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Input dir:", input_dir.display());
        println!("{:20} {}", "Output dir:", output_dir.display());
        println!("{:20} {}x{}", "Target:", width, height);
        println!("{:20} {}", "Algorithm:", algorithm);
        if let Some(e) = ext {
            println!("{:20} .{}", "Filter:", e);
        }
        println!();
        println!("{}", "Batch scaling pipeline configured.".dimmed());
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_scaling_mode_variants() {
        assert!(parse_scaling_mode("bilinear").is_ok());
        assert!(parse_scaling_mode("bicubic").is_ok());
        assert!(parse_scaling_mode("lanczos").is_ok());
        assert!(parse_scaling_mode("invalid").is_err());
    }

    #[test]
    fn test_parse_aspect_mode_variants() {
        assert!(parse_aspect_mode("stretch").is_ok());
        assert!(parse_aspect_mode("letterbox").is_ok());
        assert!(parse_aspect_mode("crop").is_ok());
        assert!(parse_aspect_mode("invalid").is_err());
    }

    #[test]
    fn test_scaling_mode_values() {
        let mode = parse_scaling_mode("lanczos").expect("should succeed");
        assert_eq!(mode, oximedia_scaling::ScalingMode::Lanczos);
    }

    #[test]
    fn test_aspect_mode_values() {
        let mode = parse_aspect_mode("letterbox").expect("should succeed");
        assert_eq!(mode, oximedia_scaling::AspectRatioMode::Letterbox);
    }

    #[test]
    fn test_scaler_integration() {
        let params = oximedia_scaling::ScalingParams::new(1920, 1080)
            .with_mode(oximedia_scaling::ScalingMode::Lanczos);
        let scaler = oximedia_scaling::VideoScaler::new(params);
        let (w, h) = scaler.calculate_dimensions(3840, 2160);
        assert_eq!((w, h), (1920, 1080));
    }
}
