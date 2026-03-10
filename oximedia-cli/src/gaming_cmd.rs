//! Gaming capture, clipping, and overlay commands.
//!
//! Provides game capture configuration, highlight clip creation,
//! and webcam/stats overlay compositing via `oximedia-gaming`.

use anyhow::{Context, Result};
use clap::Subcommand;
use colored::Colorize;
use std::path::PathBuf;

/// Gaming command subcommands.
#[derive(Subcommand, Debug)]
pub enum GamingCommand {
    /// Configure game capture settings
    Capture {
        /// Output file path
        #[arg(short, long)]
        output: PathBuf,

        /// Capture width in pixels
        #[arg(long, default_value = "1920")]
        width: u32,

        /// Capture height in pixels
        #[arg(long, default_value = "1080")]
        height: u32,

        /// Target framerate
        #[arg(long, default_value = "60")]
        fps: u32,

        /// Video codec (av1, vp9, vp8)
        #[arg(long, default_value = "av1")]
        codec: String,

        /// Bitrate in kbps
        #[arg(long)]
        bitrate: Option<u32>,

        /// Recording duration in seconds (omit for unlimited)
        #[arg(long)]
        duration: Option<f64>,

        /// Container format (webm, mkv)
        #[arg(long, default_value = "webm")]
        format: String,
    },

    /// Create a highlight clip from a recording
    Clip {
        /// Input recording file
        #[arg(short, long)]
        input: PathBuf,

        /// Output clip file
        #[arg(short, long)]
        output: PathBuf,

        /// Clip start time in seconds
        #[arg(long)]
        start: f64,

        /// Clip end time in seconds
        #[arg(long)]
        end: f64,

        /// Boost highlight intensity (auto color grade)
        #[arg(long)]
        highlight_boost: bool,

        /// Apply slow-motion factor (e.g. 0.5 for half speed)
        #[arg(long)]
        slow_motion: Option<f64>,
    },

    /// Apply gaming overlay (webcam, stats, chat)
    Overlay {
        /// Input video file
        #[arg(short, long)]
        input: PathBuf,

        /// Output file with overlay applied
        #[arg(short, long)]
        output: PathBuf,

        /// Webcam footage file for picture-in-picture
        #[arg(long)]
        webcam: Option<PathBuf>,

        /// Webcam PiP position: top-left, top-right, bottom-left, bottom-right
        #[arg(long)]
        webcam_position: Option<String>,

        /// Webcam size as percentage of main video (1-100)
        #[arg(long)]
        webcam_size: Option<u32>,

        /// Show performance stats bar overlay
        #[arg(long)]
        stats_bar: bool,

        /// Corner radius for webcam PiP (pixels)
        #[arg(long)]
        border_radius: Option<u32>,
    },
}

/// Handle gaming command dispatch.
pub async fn handle_gaming_command(command: GamingCommand, json_output: bool) -> Result<()> {
    match command {
        GamingCommand::Capture {
            output,
            width,
            height,
            fps,
            codec,
            bitrate,
            duration,
            format,
        } => {
            handle_capture(
                &output,
                width,
                height,
                fps,
                &codec,
                bitrate,
                duration,
                &format,
                json_output,
            )
            .await
        }
        GamingCommand::Clip {
            input,
            output,
            start,
            end,
            highlight_boost,
            slow_motion,
        } => {
            handle_clip(
                &input,
                &output,
                start,
                end,
                highlight_boost,
                slow_motion,
                json_output,
            )
            .await
        }
        GamingCommand::Overlay {
            input,
            output,
            webcam,
            webcam_position,
            webcam_size,
            stats_bar,
            border_radius,
        } => {
            handle_overlay(
                &input,
                &output,
                webcam.as_deref(),
                webcam_position.as_deref(),
                webcam_size,
                stats_bar,
                border_radius,
                json_output,
            )
            .await
        }
    }
}

/// Configure and validate game capture settings.
#[allow(clippy::too_many_arguments)]
async fn handle_capture(
    output: &PathBuf,
    width: u32,
    height: u32,
    fps: u32,
    codec: &str,
    bitrate: Option<u32>,
    duration: Option<f64>,
    format: &str,
    json_output: bool,
) -> Result<()> {
    // Validate codec is patent-free
    let valid_codecs = ["av1", "vp9", "vp8"];
    if !valid_codecs.contains(&codec) {
        return Err(anyhow::anyhow!(
            "Unsupported codec '{}'. OxiMedia only supports patent-free codecs: {}",
            codec,
            valid_codecs.join(", ")
        ));
    }

    // Validate format
    let valid_formats = ["webm", "mkv"];
    if !valid_formats.contains(&format) {
        return Err(anyhow::anyhow!(
            "Unsupported format '{}'. Supported: {}",
            format,
            valid_formats.join(", ")
        ));
    }

    // Validate resolution
    if width == 0 || height == 0 || width > 7680 || height > 4320 {
        return Err(anyhow::anyhow!(
            "Invalid resolution {}x{}. Must be between 1x1 and 7680x4320",
            width,
            height
        ));
    }

    // Validate fps
    if fps == 0 || fps > 240 {
        return Err(anyhow::anyhow!(
            "Invalid framerate {}. Must be between 1 and 240",
            fps
        ));
    }

    // Build StreamConfig using the gaming crate
    let mut builder = oximedia_gaming::StreamConfig::builder()
        .source(oximedia_gaming::CaptureSource::PrimaryMonitor)
        .resolution(width, height)
        .framerate(fps);

    if let Some(br) = bitrate {
        builder = builder.bitrate(br);
    }

    let _config = builder
        .build()
        .map_err(|e| anyhow::anyhow!("Invalid capture configuration: {}", e))?;

    // Compute recommended bitrate if not specified
    let effective_bitrate = bitrate.unwrap_or_else(|| recommend_bitrate(width, height, fps));

    if json_output {
        let result = serde_json::json!({
            "command": "capture",
            "output": output.display().to_string(),
            "resolution": format!("{}x{}", width, height),
            "fps": fps,
            "codec": codec,
            "bitrate_kbps": effective_bitrate,
            "duration": duration,
            "format": format,
            "status": "configured",
            "message": "Capture pipeline configured; awaiting system capture integration",
        });
        let json_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize capture config")?;
        println!("{}", json_str);
    } else {
        println!("{}", "Game Capture Configuration".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Output:", output.display());
        println!("{:20} {}x{}", "Resolution:", width, height);
        println!("{:20} {} fps", "Framerate:", fps);
        println!("{:20} {}", "Codec:", codec);
        println!("{:20} {} kbps", "Bitrate:", effective_bitrate);
        println!("{:20} {}", "Format:", format);
        if let Some(dur) = duration {
            println!("{:20} {:.1}s", "Duration:", dur);
        } else {
            println!("{:20} unlimited", "Duration:");
        }
        println!();

        println!("{}", "Encoder Recommendations".cyan().bold());
        println!("{}", "-".repeat(60));
        let preset = recommend_preset(fps);
        println!("  Encoder preset:   {}", preset);
        println!("  Keyframe interval: {} frames ({:.1}s)", fps * 2, 2.0);
        println!("  Pixel format:     YUV420P (BT.709)");
        println!();

        println!(
            "{}",
            "Note: System screen capture integration pending.".yellow()
        );
        println!(
            "{}",
            "Capture pipeline configured and ready for frame input.".dimmed()
        );
    }

    Ok(())
}

/// Create a highlight clip from a recording.
#[allow(clippy::too_many_arguments)]
async fn handle_clip(
    input: &PathBuf,
    output: &PathBuf,
    start: f64,
    end: f64,
    highlight_boost: bool,
    slow_motion: Option<f64>,
    json_output: bool,
) -> Result<()> {
    if !input.exists() {
        return Err(anyhow::anyhow!("Input file not found: {}", input.display()));
    }

    if start >= end {
        return Err(anyhow::anyhow!(
            "Start time ({:.3}s) must be before end time ({:.3}s)",
            start,
            end
        ));
    }

    if start < 0.0 {
        return Err(anyhow::anyhow!(
            "Start time must be non-negative, got {:.3}s",
            start
        ));
    }

    if let Some(sm) = slow_motion {
        if sm <= 0.0 || sm > 1.0 {
            return Err(anyhow::anyhow!(
                "Slow motion factor must be between 0.0 (exclusive) and 1.0, got {:.3}",
                sm
            ));
        }
    }

    let clip_duration = end - start;
    let effective_duration = if let Some(sm) = slow_motion {
        clip_duration / sm
    } else {
        clip_duration
    };

    let file_size = std::fs::metadata(input)
        .context("Failed to read file metadata")?
        .len();

    if json_output {
        let result = serde_json::json!({
            "command": "clip",
            "input": input.display().to_string(),
            "output": output.display().to_string(),
            "start_time": start,
            "end_time": end,
            "clip_duration": clip_duration,
            "effective_duration": effective_duration,
            "highlight_boost": highlight_boost,
            "slow_motion": slow_motion,
            "input_file_size": file_size,
            "status": "configured",
            "message": "Clip extraction configured; awaiting frame decoding pipeline",
        });
        let json_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize clip config")?;
        println!("{}", json_str);
    } else {
        println!("{}", "Highlight Clip Creation".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Input:", input.display());
        println!("{:20} {}", "Output:", output.display());
        println!("{:20} {} bytes", "Input size:", file_size);
        println!();

        println!("{}", "Clip Settings".cyan().bold());
        println!("{}", "-".repeat(60));
        println!("{:20} {:.3}s", "Start time:", start);
        println!("{:20} {:.3}s", "End time:", end);
        println!("{:20} {:.3}s", "Clip duration:", clip_duration);
        if let Some(sm) = slow_motion {
            println!("{:20} {:.2}x", "Slow motion:", sm);
            println!("{:20} {:.3}s", "Output duration:", effective_duration);
        }
        println!("{:20} {}", "Highlight boost:", highlight_boost);
        println!();

        println!(
            "{}",
            "Note: Clip extraction requires frame decoding pipeline.".yellow()
        );
        println!(
            "{}",
            "Clip configuration validated and ready for processing.".dimmed()
        );
    }

    Ok(())
}

/// Apply gaming overlay to a video.
#[allow(clippy::too_many_arguments)]
async fn handle_overlay(
    input: &PathBuf,
    output: &PathBuf,
    webcam: Option<&std::path::Path>,
    webcam_position: Option<&str>,
    webcam_size: Option<u32>,
    stats_bar: bool,
    border_radius: Option<u32>,
    json_output: bool,
) -> Result<()> {
    if !input.exists() {
        return Err(anyhow::anyhow!("Input file not found: {}", input.display()));
    }

    if let Some(wc) = webcam {
        if !wc.exists() {
            return Err(anyhow::anyhow!("Webcam file not found: {}", wc.display()));
        }
    }

    // Validate webcam position
    let valid_positions = ["top-left", "top-right", "bottom-left", "bottom-right"];
    if let Some(pos) = webcam_position {
        if !valid_positions.contains(&pos) {
            return Err(anyhow::anyhow!(
                "Invalid webcam position '{}'. Supported: {}",
                pos,
                valid_positions.join(", ")
            ));
        }
    }

    // Validate webcam size
    if let Some(size) = webcam_size {
        if size == 0 || size > 100 {
            return Err(anyhow::anyhow!(
                "Webcam size must be between 1 and 100 (percent), got {}",
                size
            ));
        }
    }

    let file_size = std::fs::metadata(input)
        .context("Failed to read file metadata")?
        .len();

    let pos = webcam_position.unwrap_or("bottom-right");
    let size = webcam_size.unwrap_or(25);

    if json_output {
        let result = serde_json::json!({
            "command": "overlay",
            "input": input.display().to_string(),
            "output": output.display().to_string(),
            "input_file_size": file_size,
            "webcam": webcam.map(|p| p.display().to_string()),
            "webcam_position": pos,
            "webcam_size_percent": size,
            "stats_bar": stats_bar,
            "border_radius": border_radius,
            "status": "configured",
            "message": "Overlay pipeline configured; awaiting frame decoding integration",
        });
        let json_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize overlay config")?;
        println!("{}", json_str);
    } else {
        println!("{}", "Gaming Overlay".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Input:", input.display());
        println!("{:20} {}", "Output:", output.display());
        println!("{:20} {} bytes", "File size:", file_size);
        println!();

        println!("{}", "Overlay Layers".cyan().bold());
        println!("{}", "-".repeat(60));

        if let Some(wc) = webcam {
            println!("  [+] Webcam PiP:     {}", wc.display());
            println!("      Position:       {}", pos);
            println!("      Size:           {}% of main video", size);
            if let Some(br) = border_radius {
                println!("      Border radius:  {}px", br);
            }
        } else {
            println!("  [-] Webcam PiP:     not configured");
        }

        if stats_bar {
            println!("  [+] Stats bar:      enabled (FPS, bitrate, encoding latency)");
        } else {
            println!("  [-] Stats bar:      disabled");
        }

        println!();
        println!(
            "{}",
            "Note: Overlay compositing requires frame decoding pipeline.".yellow()
        );
        println!(
            "{}",
            "Overlay configuration validated and ready for compositing.".dimmed()
        );
    }

    Ok(())
}

/// Recommend a bitrate based on resolution and framerate.
fn recommend_bitrate(width: u32, height: u32, fps: u32) -> u32 {
    let pixels = (width as u64) * (height as u64);
    let base = match pixels {
        0..=921_600 => 2500,            // up to 720p
        921_601..=2_073_600 => 6000,    // up to 1080p
        2_073_601..=3_686_400 => 12000, // up to 1440p
        _ => 20000,                     // 4K+
    };
    // Scale for high framerate
    if fps > 60 {
        base * 3 / 2
    } else {
        base
    }
}

/// Recommend an encoder preset based on framerate.
fn recommend_preset(fps: u32) -> &'static str {
    if fps >= 120 {
        "ultra-low-latency"
    } else if fps >= 60 {
        "low-latency"
    } else {
        "balanced"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recommend_bitrate_720p() {
        let br = recommend_bitrate(1280, 720, 60);
        assert_eq!(br, 2500);
    }

    #[test]
    fn test_recommend_bitrate_1080p() {
        let br = recommend_bitrate(1920, 1080, 60);
        assert_eq!(br, 6000);
    }

    #[test]
    fn test_recommend_bitrate_4k_high_fps() {
        let br = recommend_bitrate(3840, 2160, 120);
        assert_eq!(br, 30000);
    }

    #[test]
    fn test_recommend_preset_values() {
        assert_eq!(recommend_preset(30), "balanced");
        assert_eq!(recommend_preset(60), "low-latency");
        assert_eq!(recommend_preset(144), "ultra-low-latency");
    }

    #[test]
    fn test_recommend_bitrate_1440p() {
        let br = recommend_bitrate(2560, 1440, 60);
        assert_eq!(br, 12000);
    }
}
