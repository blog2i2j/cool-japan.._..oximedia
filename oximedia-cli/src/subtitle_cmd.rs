//! Subtitle conversion, extraction, burn-in, and synchronization.
//!
//! Provides subtitle-related commands using the `oximedia-subtitle` crate.

use anyhow::{Context, Result};
use clap::Subcommand;
use colored::Colorize;
use std::path::PathBuf;

/// Subtitle command subcommands.
#[derive(Subcommand, Debug)]
pub enum SubtitleCommand {
    /// Convert subtitles between formats (SRT, WebVTT, ASS)
    Convert {
        /// Input subtitle file
        #[arg(short, long)]
        input: PathBuf,

        /// Output subtitle file
        #[arg(short, long)]
        output: PathBuf,

        /// Input format: srt, vtt, ass (auto-detected if omitted)
        #[arg(long)]
        from: Option<String>,

        /// Output format: srt, vtt, ass
        #[arg(long)]
        to: Option<String>,

        /// Apply timing offset in milliseconds
        #[arg(long, default_value = "0")]
        offset: i64,
    },

    /// Extract subtitles from a container (MKV, WebM)
    Extract {
        /// Input video file
        #[arg(short, long)]
        input: PathBuf,

        /// Output subtitle file
        #[arg(short, long)]
        output: PathBuf,

        /// Subtitle track index
        #[arg(long, default_value = "0")]
        track: usize,

        /// Output format: srt, vtt, ass
        #[arg(long, default_value = "srt")]
        format: String,
    },

    /// Burn subtitles into video
    Burn {
        /// Input video file
        #[arg(short, long)]
        input: PathBuf,

        /// Subtitle file to burn
        #[arg(long)]
        subtitle: PathBuf,

        /// Output video file
        #[arg(short, long)]
        output: PathBuf,

        /// Font size
        #[arg(long, default_value = "24")]
        font_size: u32,

        /// Subtitle format: srt, vtt, ass (auto-detected if omitted)
        #[arg(long)]
        format: Option<String>,
    },

    /// Adjust subtitle timing (sync offset)
    Sync {
        /// Input subtitle file
        #[arg(short, long)]
        input: PathBuf,

        /// Output subtitle file (defaults to overwriting input)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Timing offset in milliseconds (positive = later, negative = earlier)
        #[arg(long)]
        offset: i64,

        /// Scale timing by factor (e.g., 1.001 for 23.976->24fps correction)
        #[arg(long)]
        scale: Option<f64>,
    },
}

/// Handle subtitle command dispatch.
pub async fn handle_subtitle_command(command: SubtitleCommand, _json_output: bool) -> Result<()> {
    match command {
        SubtitleCommand::Convert {
            input,
            output,
            from,
            to,
            offset,
        } => convert_subtitles(&input, &output, from.as_deref(), to.as_deref(), offset).await,
        SubtitleCommand::Extract {
            input,
            output,
            track,
            format,
        } => extract_subtitles(&input, &output, track, &format).await,
        SubtitleCommand::Burn {
            input,
            subtitle,
            output,
            font_size,
            format,
        } => burn_subtitles(&input, &subtitle, &output, font_size, format.as_deref()).await,
        SubtitleCommand::Sync {
            input,
            output,
            offset,
            scale,
        } => sync_subtitles(&input, output.as_ref(), offset, scale).await,
    }
}

/// Detect subtitle format from file extension.
fn detect_format(path: &PathBuf) -> Option<&str> {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| match ext.to_lowercase().as_str() {
            "srt" => "srt",
            "vtt" | "webvtt" => "vtt",
            "ass" | "ssa" => "ass",
            _ => "unknown",
        })
}

/// Parse subtitles from text using the specified format.
fn parse_subtitles(text: &str, format: &str) -> Result<Vec<oximedia_subtitle::Subtitle>> {
    match format {
        "srt" => oximedia_subtitle::SrtParser::parse(text)
            .map_err(|e| anyhow::anyhow!("Failed to parse SRT: {}", e)),
        "vtt" | "webvtt" => oximedia_subtitle::WebVttParser::parse(text)
            .map_err(|e| anyhow::anyhow!("Failed to parse WebVTT: {}", e)),
        "ass" | "ssa" => oximedia_subtitle::AssParser::parse(text)
            .map_err(|e| anyhow::anyhow!("Failed to parse ASS: {}", e)),
        other => Err(anyhow::anyhow!(
            "Unknown subtitle format '{}'. Supported: srt, vtt, ass",
            other
        )),
    }
}

/// Format a timestamp in milliseconds to SRT format (HH:MM:SS,mmm).
fn format_srt_timestamp(ms: i64) -> String {
    let total_seconds = ms / 1000;
    let millis = ms % 1000;
    let hours = total_seconds / 3600;
    let minutes = (total_seconds % 3600) / 60;
    let seconds = total_seconds % 60;
    format!("{:02}:{:02}:{:02},{:03}", hours, minutes, seconds, millis)
}

/// Format a timestamp in milliseconds to WebVTT format (HH:MM:SS.mmm).
fn format_vtt_timestamp(ms: i64) -> String {
    let total_seconds = ms / 1000;
    let millis = ms % 1000;
    let hours = total_seconds / 3600;
    let minutes = (total_seconds % 3600) / 60;
    let seconds = total_seconds % 60;
    format!("{:02}:{:02}:{:02}.{:03}", hours, minutes, seconds, millis)
}

/// Serialize subtitles to a string in the specified format.
fn serialize_subtitles(subs: &[oximedia_subtitle::Subtitle], format: &str) -> Result<String> {
    let mut output = String::new();

    match format {
        "srt" => {
            for (i, sub) in subs.iter().enumerate() {
                output.push_str(&format!("{}\n", i + 1));
                output.push_str(&format!(
                    "{} --> {}\n",
                    format_srt_timestamp(sub.start_time),
                    format_srt_timestamp(sub.end_time)
                ));
                output.push_str(&sub.text);
                output.push_str("\n\n");
            }
        }
        "vtt" | "webvtt" => {
            output.push_str("WEBVTT\n\n");
            for sub in subs {
                if let Some(ref id) = sub.id {
                    output.push_str(id);
                    output.push('\n');
                }
                output.push_str(&format!(
                    "{} --> {}\n",
                    format_vtt_timestamp(sub.start_time),
                    format_vtt_timestamp(sub.end_time)
                ));
                output.push_str(&sub.text);
                output.push_str("\n\n");
            }
        }
        "ass" | "ssa" => {
            output.push_str("[Script Info]\nScriptType: v4.00+\n\n");
            output.push_str("[V4+ Styles]\n");
            output.push_str("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n");
            output.push_str("Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1\n\n");
            output.push_str("[Events]\n");
            output.push_str(
                "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n",
            );
            for sub in subs {
                let start = format_ass_timestamp(sub.start_time);
                let end = format_ass_timestamp(sub.end_time);
                output.push_str(&format!(
                    "Dialogue: 0,{},{},Default,,0,0,0,,{}\n",
                    start, end, sub.text
                ));
            }
        }
        other => {
            return Err(anyhow::anyhow!("Unsupported output format: {}", other));
        }
    }

    Ok(output)
}

/// Format a timestamp in milliseconds to ASS format (H:MM:SS.cc).
fn format_ass_timestamp(ms: i64) -> String {
    let total_seconds = ms / 1000;
    let centiseconds = (ms % 1000) / 10;
    let hours = total_seconds / 3600;
    let minutes = (total_seconds % 3600) / 60;
    let seconds = total_seconds % 60;
    format!(
        "{}:{:02}:{:02}.{:02}",
        hours, minutes, seconds, centiseconds
    )
}

/// Convert subtitles between formats.
async fn convert_subtitles(
    input: &PathBuf,
    output: &PathBuf,
    from: Option<&str>,
    to: Option<&str>,
    offset: i64,
) -> Result<()> {
    if !input.exists() {
        return Err(anyhow::anyhow!("Input file not found: {}", input.display()));
    }

    // Detect input format
    let input_format = from.unwrap_or_else(|| detect_format(input).unwrap_or("srt"));

    // Detect output format
    let output_format = to.unwrap_or_else(|| detect_format(output).unwrap_or("srt"));

    // Read input file
    let text = tokio::fs::read_to_string(input)
        .await
        .context("Failed to read input subtitle file")?;

    // Parse subtitles
    let mut subs = parse_subtitles(&text, input_format)?;

    // Apply offset if non-zero
    if offset != 0 {
        for sub in &mut subs {
            sub.start_time += offset;
            sub.end_time += offset;
        }
    }

    // Serialize to output format
    let output_text = serialize_subtitles(&subs, output_format)?;

    // Write output
    tokio::fs::write(output, &output_text)
        .await
        .context("Failed to write output subtitle file")?;

    println!("{}", "Subtitle Conversion".green().bold());
    println!("{}", "=".repeat(60));
    println!("{:20} {}", "Input:", input.display());
    println!("{:20} {}", "Output:", output.display());
    println!("{:20} {}", "From:", input_format);
    println!("{:20} {}", "To:", output_format);
    println!("{:20} {} subtitle(s)", "Converted:", subs.len());
    if offset != 0 {
        println!("{:20} {}ms", "Offset applied:", offset);
    }
    println!();
    println!("{}", "Conversion complete.".green());

    Ok(())
}

/// Extract subtitles from a container.
async fn extract_subtitles(
    input: &PathBuf,
    output: &PathBuf,
    track: usize,
    format: &str,
) -> Result<()> {
    if !input.exists() {
        return Err(anyhow::anyhow!("Input file not found: {}", input.display()));
    }

    println!("{}", "Subtitle Extraction".green().bold());
    println!("{}", "=".repeat(60));
    println!("{:20} {}", "Input:", input.display());
    println!("{:20} {}", "Output:", output.display());
    println!("{:20} {}", "Track:", track);
    println!("{:20} {}", "Format:", format);
    println!();

    println!(
        "{}",
        "Note: Container demuxing pipeline not yet integrated.".yellow()
    );
    println!(
        "{}",
        "Subtitle parsers are ready; demuxing will enable extraction from containers.".dimmed()
    );

    Ok(())
}

/// Burn subtitles into video.
async fn burn_subtitles(
    input: &PathBuf,
    subtitle: &PathBuf,
    output: &PathBuf,
    font_size: u32,
    format: Option<&str>,
) -> Result<()> {
    if !input.exists() {
        return Err(anyhow::anyhow!(
            "Input video not found: {}",
            input.display()
        ));
    }
    if !subtitle.exists() {
        return Err(anyhow::anyhow!(
            "Subtitle file not found: {}",
            subtitle.display()
        ));
    }

    let sub_format = format.unwrap_or_else(|| detect_format(subtitle).unwrap_or("srt"));

    // Verify subtitle file is parseable
    let text = tokio::fs::read_to_string(subtitle)
        .await
        .context("Failed to read subtitle file")?;
    let subs = parse_subtitles(&text, sub_format)?;

    println!("{}", "Subtitle Burn-in".green().bold());
    println!("{}", "=".repeat(60));
    println!("{:20} {}", "Input video:", input.display());
    println!("{:20} {}", "Subtitle:", subtitle.display());
    println!("{:20} {}", "Output:", output.display());
    println!("{:20} {}", "Format:", sub_format);
    println!("{:20} {}px", "Font size:", font_size);
    println!("{:20} {}", "Subtitle count:", subs.len());
    println!();

    if !subs.is_empty() {
        println!("{}", "Preview (first 3 subtitles)".cyan().bold());
        println!("{}", "-".repeat(60));
        for sub in subs.iter().take(3) {
            println!(
                "  [{} --> {}] {}",
                format_srt_timestamp(sub.start_time),
                format_srt_timestamp(sub.end_time),
                sub.text
            );
        }
        if subs.len() > 3 {
            println!("  ... and {} more", subs.len() - 3);
        }
        println!();
    }

    println!(
        "{}",
        "Note: Video encoding pipeline not yet integrated.".yellow()
    );
    println!(
        "{}",
        "Subtitle renderer is ready; video pipeline will enable burn-in.".dimmed()
    );

    Ok(())
}

/// Adjust subtitle timing.
async fn sync_subtitles(
    input: &PathBuf,
    output: Option<&PathBuf>,
    offset: i64,
    scale: Option<f64>,
) -> Result<()> {
    if !input.exists() {
        return Err(anyhow::anyhow!("Input file not found: {}", input.display()));
    }

    let format = detect_format(input).unwrap_or("srt");

    // Read and parse
    let text = tokio::fs::read_to_string(input)
        .await
        .context("Failed to read subtitle file")?;
    let mut subs = parse_subtitles(&text, format)?;

    // Apply scale factor
    if let Some(factor) = scale {
        for sub in &mut subs {
            sub.start_time = (sub.start_time as f64 * factor) as i64;
            sub.end_time = (sub.end_time as f64 * factor) as i64;
        }
    }

    // Apply offset
    if offset != 0 {
        for sub in &mut subs {
            sub.start_time += offset;
            sub.end_time += offset;
        }
    }

    // Serialize
    let output_text = serialize_subtitles(&subs, format)?;

    // Write to output or input
    let out_path = output.unwrap_or(input);
    tokio::fs::write(out_path, &output_text)
        .await
        .context("Failed to write subtitle file")?;

    println!("{}", "Subtitle Sync".green().bold());
    println!("{}", "=".repeat(60));
    println!("{:20} {}", "Input:", input.display());
    println!("{:20} {}", "Output:", out_path.display());
    println!("{:20} {}ms", "Offset:", offset);
    if let Some(factor) = scale {
        println!("{:20} {}", "Scale factor:", factor);
    }
    println!("{:20} {} subtitle(s)", "Processed:", subs.len());
    println!();
    println!("{}", "Sync complete.".green());

    Ok(())
}
