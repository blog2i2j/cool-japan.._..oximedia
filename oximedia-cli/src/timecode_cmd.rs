//! Timecode command for `oximedia timecode`.
//!
//! Provides convert, calculate, validate, burn, to-frames, and from-frames
//! subcommands via `oximedia-timecode`.

use anyhow::{Context, Result};
use clap::Subcommand;
use colored::Colorize;
use oximedia_timecode::{FrameRate, Timecode, TimecodeError};
use std::path::PathBuf;

/// Subcommands for `oximedia timecode`.
#[derive(Subcommand)]
pub enum TimecodeCommand {
    /// Convert a timecode between two frame rates
    Convert {
        /// Timecode string (HH:MM:SS:FF or HH:MM:SS;FF)
        #[arg(value_name = "TIMECODE")]
        timecode: String,

        /// Source frame rate (23.976, 24, 25, 29.97df, 29.97ndf, 30, 50, 59.94, 60)
        #[arg(long)]
        from_fps: String,

        /// Target frame rate
        #[arg(long)]
        to_fps: String,
    },

    /// Perform arithmetic on a timecode (add or subtract frames/seconds)
    Calculate {
        /// Timecode string (HH:MM:SS:FF)
        #[arg(value_name = "TIMECODE")]
        timecode: String,

        /// Frame rate
        #[arg(long)]
        fps: String,

        /// Operation: add-frames, sub-frames, add-seconds, sub-seconds
        #[arg(long)]
        operation: String,

        /// Value to apply (integer)
        #[arg(value_name = "VALUE")]
        value: i64,
    },

    /// Validate a timecode string
    Validate {
        /// Timecode string (HH:MM:SS:FF or HH:MM:SS;FF)
        #[arg(value_name = "TIMECODE")]
        timecode: String,

        /// Frame rate
        #[arg(long)]
        fps: String,
    },

    /// Convert timecode to total frame count since midnight
    ToFrames {
        /// Timecode string (HH:MM:SS:FF)
        #[arg(value_name = "TIMECODE")]
        timecode: String,

        /// Frame rate
        #[arg(long)]
        fps: String,
    },

    /// Convert a total frame count to timecode
    FromFrames {
        /// Frame number since midnight
        #[arg(value_name = "FRAMES")]
        frames: u64,

        /// Frame rate
        #[arg(long)]
        fps: String,
    },

    /// Burn timecode overlay into a video file (configuration preview)
    Burn {
        /// Input video file
        #[arg(short, long)]
        input: PathBuf,

        /// Output video file
        #[arg(short, long)]
        output: PathBuf,

        /// Starting timecode (HH:MM:SS:FF); defaults to 00:00:00:00
        #[arg(long, default_value = "00:00:00:00")]
        start: String,

        /// Frame rate
        #[arg(long, default_value = "25")]
        fps: String,

        /// Position: top-left, top-right, bottom-left, bottom-right, center
        #[arg(long, default_value = "bottom-right")]
        position: String,

        /// Font size in points
        #[arg(long, default_value = "36")]
        font_size: u32,
    },
}

/// Entry point called from `main.rs`.
pub async fn run_timecode(command: TimecodeCommand, json_output: bool) -> Result<()> {
    match command {
        TimecodeCommand::Convert {
            timecode,
            from_fps,
            to_fps,
        } => cmd_convert(&timecode, &from_fps, &to_fps, json_output),

        TimecodeCommand::Calculate {
            timecode,
            fps,
            operation,
            value,
        } => cmd_calculate(&timecode, &fps, &operation, value, json_output),

        TimecodeCommand::Validate { timecode, fps } => cmd_validate(&timecode, &fps, json_output),

        TimecodeCommand::ToFrames { timecode, fps } => cmd_to_frames(&timecode, &fps, json_output),

        TimecodeCommand::FromFrames { frames, fps } => cmd_from_frames(frames, &fps, json_output),

        TimecodeCommand::Burn {
            input,
            output,
            start,
            fps,
            position,
            font_size,
        } => cmd_burn(
            &input,
            &output,
            &start,
            &fps,
            &position,
            font_size,
            json_output,
        ),
    }
}

// ---------------------------------------------------------------------------
// Individual subcommand implementations
// ---------------------------------------------------------------------------

fn cmd_convert(tc_str: &str, from_fps: &str, to_fps: &str, json_output: bool) -> Result<()> {
    let src_rate = parse_frame_rate(from_fps)?;
    let dst_rate = parse_frame_rate(to_fps)?;
    let tc = parse_timecode(tc_str, src_rate)?;

    // Convert via total frame count, adjusting for frame rate ratio
    let src_frames = tc.to_frames();
    let src_fps_float = src_rate.as_float();
    let dst_fps_float = dst_rate.as_float();

    // Scale frame count to target fps
    let dst_frame_count = (src_frames as f64 * dst_fps_float / src_fps_float).round() as u64;
    let converted = Timecode::from_frames(dst_frame_count, dst_rate)
        .map_err(|e| anyhow::anyhow!("Timecode conversion failed: {}", e))?;

    if json_output {
        let obj = serde_json::json!({
            "input": tc_str,
            "from_fps": from_fps,
            "to_fps": to_fps,
            "source_frames": src_frames,
            "output_frames": dst_frame_count,
            "output": converted.to_string(),
        });
        println!("{}", serde_json::to_string_pretty(&obj)?);
        return Ok(());
    }

    println!("{}", "Timecode Conversion".green().bold());
    println!(
        "  {} {} @ {}",
        "Input:".cyan(),
        tc.to_string().yellow(),
        from_fps
    );
    println!(
        "  {} {} @ {}",
        "Output:".cyan(),
        converted.to_string().yellow(),
        to_fps
    );
    println!(
        "  {} {} → {} frames",
        "Frames:".cyan(),
        src_frames,
        dst_frame_count
    );

    Ok(())
}

fn cmd_calculate(
    tc_str: &str,
    fps_str: &str,
    operation: &str,
    value: i64,
    json_output: bool,
) -> Result<()> {
    let fps = parse_frame_rate(fps_str)?;
    let tc = parse_timecode(tc_str, fps)?;
    let initial_frames = tc.to_frames() as i64;

    let delta_frames: i64 = match operation {
        "add-frames" => value,
        "sub-frames" => -value,
        "add-seconds" => {
            let fps_val = fps.frames_per_second() as i64;
            value * fps_val
        }
        "sub-seconds" => {
            let fps_val = fps.frames_per_second() as i64;
            -value * fps_val
        }
        other => anyhow::bail!(
            "Unknown operation '{}'. Supported: add-frames, sub-frames, add-seconds, sub-seconds",
            other
        ),
    };

    let result_frames = (initial_frames + delta_frames).max(0) as u64;
    let result_tc = Timecode::from_frames(result_frames, fps)
        .map_err(|e| anyhow::anyhow!("Timecode calculation failed: {}", e))?;

    if json_output {
        let obj = serde_json::json!({
            "input": tc_str,
            "fps": fps_str,
            "operation": operation,
            "value": value,
            "initial_frames": initial_frames,
            "delta_frames": delta_frames,
            "result_frames": result_frames,
            "result": result_tc.to_string(),
        });
        println!("{}", serde_json::to_string_pretty(&obj)?);
        return Ok(());
    }

    println!("{}", "Timecode Calculate".green().bold());
    println!(
        "  {} {} @ {}",
        "Input:".cyan(),
        tc.to_string().yellow(),
        fps_str
    );
    println!(
        "  {} {} ({:+})",
        "Operation:".cyan(),
        operation,
        delta_frames
    );
    println!(
        "  {} {}",
        "Result:".cyan(),
        result_tc.to_string().yellow().bold()
    );

    Ok(())
}

fn cmd_validate(tc_str: &str, fps_str: &str, json_output: bool) -> Result<()> {
    let fps = parse_frame_rate(fps_str)?;
    let parse_result = parse_timecode(tc_str, fps);

    let (valid, reason) = match &parse_result {
        Ok(_) => (true, "Valid SMPTE timecode".to_string()),
        Err(e) => (false, e.to_string()),
    };

    if json_output {
        let obj = serde_json::json!({
            "input": tc_str,
            "fps": fps_str,
            "valid": valid,
            "reason": reason,
            "parsed": parse_result.as_ref().map(|tc| tc.to_string()).ok(),
        });
        println!("{}", serde_json::to_string_pretty(&obj)?);
        return Ok(());
    }

    println!("{}", "Timecode Validation".green().bold());
    println!("  {} {}", "Input:".cyan(), tc_str);
    println!("  {} {}", "FPS:".cyan(), fps_str);
    if valid {
        println!("  {} {}", "Status:".cyan(), "VALID".green().bold());
        if let Ok(tc) = parse_result {
            println!("  {} {}", "Parsed:".cyan(), tc.to_string().yellow());
        }
    } else {
        println!("  {} {}", "Status:".cyan(), "INVALID".red().bold());
        println!("  {} {}", "Reason:".cyan(), reason.red());
    }

    Ok(())
}

fn cmd_to_frames(tc_str: &str, fps_str: &str, json_output: bool) -> Result<()> {
    let fps = parse_frame_rate(fps_str)?;
    let tc = parse_timecode(tc_str, fps)?;
    let total_frames = tc.to_frames();

    // Also compute wall-clock time
    let fps_f = fps.as_float();
    let seconds_total = total_frames as f64 / fps_f;
    let hours = (seconds_total / 3600.0) as u64;
    let minutes = ((seconds_total % 3600.0) / 60.0) as u64;
    let seconds = (seconds_total % 60.0) as u64;

    if json_output {
        let obj = serde_json::json!({
            "input": tc_str,
            "fps": fps_str,
            "total_frames": total_frames,
            "wall_clock_seconds": seconds_total,
            "wall_clock": format!("{:02}:{:02}:{:02}", hours, minutes, seconds),
        });
        println!("{}", serde_json::to_string_pretty(&obj)?);
        return Ok(());
    }

    println!("{}", "Timecode → Frames".green().bold());
    println!(
        "  {} {} @ {}",
        "Input:".cyan(),
        tc.to_string().yellow(),
        fps_str
    );
    println!(
        "  {} {}",
        "Total frames:".cyan(),
        total_frames.to_string().yellow().bold()
    );
    println!(
        "  {} {:.3}s  ({:02}:{:02}:{:02})",
        "Wall clock:".cyan(),
        seconds_total,
        hours,
        minutes,
        seconds
    );

    Ok(())
}

fn cmd_from_frames(frames: u64, fps_str: &str, json_output: bool) -> Result<()> {
    let fps = parse_frame_rate(fps_str)?;
    let tc = Timecode::from_frames(frames, fps)
        .map_err(|e| anyhow::anyhow!("Failed to build timecode: {}", e))?;

    let fps_f = fps.as_float();
    let seconds_total = frames as f64 / fps_f;

    if json_output {
        let obj = serde_json::json!({
            "input_frames": frames,
            "fps": fps_str,
            "timecode": tc.to_string(),
            "drop_frame": tc.frame_rate.drop_frame,
            "wall_clock_seconds": seconds_total,
        });
        println!("{}", serde_json::to_string_pretty(&obj)?);
        return Ok(());
    }

    println!("{}", "Frames → Timecode".green().bold());
    println!(
        "  {} {} @ {}",
        "Input:".cyan(),
        frames.to_string().yellow(),
        fps_str
    );
    println!(
        "  {} {}",
        "Timecode:".cyan(),
        tc.to_string().yellow().bold()
    );
    println!("  {} {:.3}s", "Wall clock:".cyan(), seconds_total);

    Ok(())
}

fn cmd_burn(
    input: &PathBuf,
    output: &PathBuf,
    start: &str,
    fps_str: &str,
    position: &str,
    font_size: u32,
    json_output: bool,
) -> Result<()> {
    // Validate input exists
    if !input.exists() {
        anyhow::bail!("Input file not found: {}", input.display());
    }

    let fps = parse_frame_rate(fps_str)?;
    let start_tc =
        parse_timecode(start, fps).with_context(|| format!("Invalid start timecode: {}", start))?;

    // Validate position
    let valid_positions = [
        "top-left",
        "top-right",
        "bottom-left",
        "bottom-right",
        "center",
    ];
    if !valid_positions.contains(&position) {
        anyhow::bail!(
            "Invalid position '{}'. Supported: {}",
            position,
            valid_positions.join(", ")
        );
    }

    if json_output {
        let obj = serde_json::json!({
            "command": "timecode-burn",
            "input": input.to_string_lossy(),
            "output": output.to_string_lossy(),
            "start_timecode": start_tc.to_string(),
            "fps": fps_str,
            "position": position,
            "font_size": font_size,
            "status": "configured",
            "note": "Burn-in requires a video render backend. Parameters validated."
        });
        println!("{}", serde_json::to_string_pretty(&obj)?);
        return Ok(());
    }

    println!("{}", "Timecode Burn-in Configuration".green().bold());
    println!("  {} {}", "Input:".cyan(), input.display());
    println!("  {} {}", "Output:".cyan(), output.display());
    println!("  {} {}", "Start TC:".cyan(), start_tc.to_string().yellow());
    println!("  {} {}", "FPS:".cyan(), fps_str);
    println!("  {} {}", "Position:".cyan(), position);
    println!("  {} {}pt", "Font size:".cyan(), font_size);
    println!();

    let input_meta = std::fs::metadata(input)
        .with_context(|| format!("Cannot access input: {}", input.display()))?;
    println!("  {} Input file: {} bytes", "✓".green(), input_meta.len());
    println!();
    println!(
        "  {} Burn-in requires video frame access. Configuration is valid.",
        "!".yellow()
    );
    println!(
        "  {} Use `oximedia transcode` with filter burn_timecode for full rendering.",
        "→".blue()
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Parse a frame-rate string into a `FrameRate` enum variant.
fn parse_frame_rate(s: &str) -> Result<FrameRate> {
    match s.to_lowercase().replace(' ', "").as_str() {
        "23.976" | "23976" | "fps23976" | "23.98" => Ok(FrameRate::Fps23976),
        "24" | "fps24" => Ok(FrameRate::Fps24),
        "25" | "fps25" => Ok(FrameRate::Fps25),
        "29.97df" | "2997df" | "29.97" | "29.97dropframe" | "ntsc" => Ok(FrameRate::Fps2997DF),
        "29.97ndf" | "2997ndf" | "29.97nondropframe" => Ok(FrameRate::Fps2997NDF),
        "30" | "fps30" => Ok(FrameRate::Fps30),
        "50" | "fps50" => Ok(FrameRate::Fps50),
        "59.94" | "5994" | "fps5994" => Ok(FrameRate::Fps5994),
        "60" | "fps60" => Ok(FrameRate::Fps60),
        other => anyhow::bail!(
            "Unknown frame rate '{}'. Supported: 23.976, 24, 25, 29.97df, 29.97ndf, 30, 50, 59.94, 60",
            other
        ),
    }
}

/// Parse a timecode string `HH:MM:SS:FF` or `HH:MM:SS;FF` into a `Timecode`.
fn parse_timecode(s: &str, fps: FrameRate) -> Result<Timecode> {
    // Accept both `:` and `;` as frame separator
    let s_norm = s.replace(';', ":");
    let parts: Vec<&str> = s_norm.splitn(4, ':').collect();
    if parts.len() != 4 {
        anyhow::bail!(
            "Invalid timecode format '{}'. Expected HH:MM:SS:FF or HH:MM:SS;FF",
            s
        );
    }

    let hours: u8 = parts[0]
        .parse()
        .with_context(|| format!("Invalid hours in timecode '{}'", s))?;
    let minutes: u8 = parts[1]
        .parse()
        .with_context(|| format!("Invalid minutes in timecode '{}'", s))?;
    let seconds: u8 = parts[2]
        .parse()
        .with_context(|| format!("Invalid seconds in timecode '{}'", s))?;
    let frames: u8 = parts[3]
        .parse()
        .with_context(|| format!("Invalid frames in timecode '{}'", s))?;

    Timecode::new(hours, minutes, seconds, frames, fps)
        .map_err(|e: TimecodeError| anyhow::anyhow!("Invalid timecode '{}': {}", s, e))
}
