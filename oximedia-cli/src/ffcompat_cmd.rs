//! FFmpeg-compatible command handler.
//!
//! Accepts raw FFmpeg-style arguments, passes them through the
//! `oximedia-compat-ffmpeg` translation layer, prints any diagnostics,
//! and then executes each resulting transcode job by delegating to the
//! native `transcode` module.

use anyhow::{Context, Result};
use colored::Colorize;
use oximedia_compat_ffmpeg::{parse_and_translate, DiagnosticKind, ParsedFilter, TranscodeJob};
use std::path::PathBuf;
use tracing::warn;

use crate::transcode::{self, TranscodeOptions};

// ─────────────────────────────────────────────────────────────────────────────
// Public entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Run the FFmpeg-compatible command with the provided argument list.
///
/// The `dry_run` flag suppresses actual execution and only prints the plan.
pub async fn run(args: Vec<String>) -> Result<()> {
    if args.is_empty() {
        print_ffcompat_help();
        return Ok(());
    }

    // Check for --dry-run / --plan anywhere in args (OxiMedia extension).
    // We remove it before passing to the compat parser so it doesn't confuse
    // the FFmpeg arg parser.
    let dry_run = args
        .iter()
        .any(|a| a == "--dry-run" || a == "--plan" || a == "-dry-run");

    let filtered_args: Vec<String> = args
        .into_iter()
        .filter(|a| a != "--dry-run" && a != "--plan" && a != "-dry-run")
        .collect();

    let result = parse_and_translate(&filtered_args);

    // Print diagnostics to stderr using FFmpeg-style formatting.
    for diag in &result.diagnostics {
        match &diag.kind {
            DiagnosticKind::PatentCodecSubstituted { from, to } => {
                eprintln!(
                    "{} Codec '{}' is a patent codec. Using '{}' instead.",
                    "warning:".yellow().bold(),
                    from,
                    to
                );
            }
            DiagnosticKind::UnknownOptionIgnored { option } => {
                eprintln!(
                    "{} Option '{}' not supported. Ignoring.",
                    "warning:".yellow().bold(),
                    option
                );
            }
            DiagnosticKind::FilterNotSupported { filter } => {
                eprintln!(
                    "{} Filter '{}' not supported. Skipping.",
                    "warning:".yellow().bold(),
                    filter
                );
            }
            DiagnosticKind::UnsupportedFeature { description } => {
                eprintln!("{} {}.", "warning:".yellow().bold(), description);
            }
            DiagnosticKind::Info { message } => {
                println!("{} {}", "info:".cyan(), message);
            }
            DiagnosticKind::Error { message } => {
                eprintln!("{} {}", "error:".red().bold(), message);
                if let Some(hint) = &diag.suggestion {
                    eprintln!("  {} {}", "hint:".yellow(), hint);
                }
            }
            DiagnosticKind::Warning { message } => {
                eprintln!("{} {}", "warning:".yellow().bold(), message);
                if let Some(hint) = &diag.suggestion {
                    eprintln!("  {} {}", "hint:".cyan(), hint);
                }
            }
        }
    }

    if result.has_errors() {
        anyhow::bail!("translation failed with errors; see diagnostics above");
    }

    // Print the translated jobs summary.
    println!(
        "\n{} {} transcode job(s) translated from FFmpeg arguments:",
        "✓".green(),
        result.jobs.len()
    );

    for (idx, job) in result.jobs.iter().enumerate() {
        println!("\n{} Job {}:", "─".repeat(4).dimmed(), idx + 1);
        println!("  input:  {}", job.input_path.cyan());
        println!("  output: {}", job.output_path.cyan());

        if let Some(vc) = &job.video_codec {
            println!("  video codec: {}", vc.green());
        }
        if let Some(ac) = &job.audio_codec {
            println!("  audio codec: {}", ac.green());
        }
        if let Some(vb) = &job.video_bitrate {
            println!("  video bitrate: {}", vb);
        }
        if let Some(ab) = &job.audio_bitrate {
            println!("  audio bitrate: {}", ab);
        }
        if let Some(crf) = job.crf {
            println!("  crf: {:.1}", crf);
        }
        if !job.video_filters.is_empty() {
            println!("  video filters: {} filter(s)", job.video_filters.len());
        }
        if !job.audio_filters.is_empty() {
            println!("  audio filters: {} filter(s)", job.audio_filters.len());
        }
        if let Some(seek) = &job.seek {
            println!("  seek: {}", seek);
        }
        if let Some(dur) = &job.duration {
            println!("  duration: {}", dur);
        }
        if !job.metadata.is_empty() {
            for (k, v) in &job.metadata {
                println!("  metadata: {}={}", k, v);
            }
        }
        if job.no_video {
            println!("  {}", "no video".dimmed());
        }
        if job.no_audio {
            println!("  {}", "no audio".dimmed());
        }
        if job.overwrite {
            println!("  overwrite: yes");
        }
        if !job.map.is_empty() {
            println!("  map: {} stream selector(s)", job.map.len());
        }

        if dry_run {
            println!("  {}", "[dry-run: skipping execution]".yellow().italic());
        }
    }

    if dry_run {
        println!("\n{} Dry-run mode — no files were written.", "note:".cyan());
        return Ok(());
    }

    // Execute each job.
    for (idx, job) in result.jobs.iter().enumerate() {
        eprintln!(
            "\n{} Executing job {}/{}: {} → {}",
            "oximedia-ff:".green().bold(),
            idx + 1,
            result.jobs.len(),
            job.input_path.cyan(),
            job.output_path.cyan()
        );

        execute_job(job).await.with_context(|| {
            format!(
                "job {} failed: {} → {}",
                idx + 1,
                job.input_path,
                job.output_path
            )
        })?;
    }

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Job execution
// ─────────────────────────────────────────────────────────────────────────────

/// Execute a single [`TranscodeJob`] by mapping its fields onto [`TranscodeOptions`]
/// and delegating to the native `transcode::transcode` function.
async fn execute_job(job: &TranscodeJob) -> Result<()> {
    // Overwrite guard — honour the job flag.
    if !job.overwrite && std::path::Path::new(&job.output_path).exists() {
        anyhow::bail!(
            "Output file '{}' already exists. Pass -y to overwrite.",
            job.output_path
        );
    }

    // Build filter strings for both video and audio filter chains.
    let vf_string = build_filter_string(&job.video_filters);
    let af_string = build_filter_string(&job.audio_filters);

    // Extract a scale dimension from video_filters if a Scale filter is present.
    // This populates the `scale` field of TranscodeOptions.
    let scale_from_filters = extract_scale_filter(&job.video_filters);

    // Resolve the effective video codec string.
    // A value of "copy" means stream copy — pass it through as-is; the
    // transcode module will not encode those streams.
    let video_codec = match job.video_codec.as_deref() {
        Some("copy") | None if job.no_video => None,
        Some(vc) => Some(vc.to_string()),
        None => None,
    };

    let audio_codec = match job.audio_codec.as_deref() {
        Some("copy") | None if job.no_audio => None,
        Some(ac) => Some(ac.to_string()),
        None => None,
    };

    // Convert CRF: TranscodeJob uses f64; TranscodeOptions uses u32.
    let crf = job.crf.map(|c| c.round() as u32);

    let options = TranscodeOptions {
        input: PathBuf::from(&job.input_path),
        output: PathBuf::from(&job.output_path),
        preset_name: None,
        video_codec,
        audio_codec,
        video_bitrate: job.video_bitrate.clone(),
        audio_bitrate: job.audio_bitrate.clone(),
        // Use scale extracted from -vf, falling back to None.
        scale: scale_from_filters,
        video_filter: vf_string,
        audio_filter: af_string,
        start_time: job.seek.clone(),
        duration: job.duration.clone(),
        framerate: None,
        preset: "medium".to_string(),
        two_pass: false,
        crf,
        threads: num_cpus(),
        overwrite: job.overwrite,
        resume: false,
    };

    transcode::transcode(options).await
}

// ─────────────────────────────────────────────────────────────────────────────
// Filter helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Reconstruct an FFmpeg-style filter string from a slice of [`ParsedFilter`] values.
///
/// Passthrough and Unknown filters are dropped (with a warning for Unknown).
fn build_filter_string(filters: &[ParsedFilter]) -> Option<String> {
    let parts: Vec<String> = filters
        .iter()
        .filter_map(|f| match f {
            ParsedFilter::Scale { w, h } => Some(format!("scale={}:{}", w, h)),
            ParsedFilter::Fps { rate } => Some(format!("fps={}", rate)),
            ParsedFilter::HFlip => Some("hflip".to_string()),
            ParsedFilter::VFlip => Some("vflip".to_string()),
            ParsedFilter::Deinterlace => Some("yadif".to_string()),
            ParsedFilter::Rotate { angle } => Some(format!("rotate={}", angle)),
            ParsedFilter::Crop { w, h, x, y } => Some(format!("crop={}:{}:{}:{}", w, h, x, y)),
            ParsedFilter::ColorCorrect {
                brightness,
                contrast,
                saturation,
            } => Some(format!(
                "eq=brightness={}:contrast={}:saturation={}",
                brightness, contrast, saturation
            )),
            ParsedFilter::Lut3d { file } => Some(format!("lut3d=file={}", file)),
            ParsedFilter::SubtitleBurnIn { file } => Some(format!("subtitles=filename={}", file)),
            ParsedFilter::LoudNorm {
                integrated,
                true_peak,
                lra,
            } => Some(format!(
                "loudnorm=I={}:TP={}:LRA={}",
                integrated, true_peak, lra
            )),
            ParsedFilter::Volume { factor } => Some(format!("volume={}", factor)),
            ParsedFilter::Resample { sample_rate } => Some(format!("aresample={}", sample_rate)),
            ParsedFilter::Compressor { threshold, ratio } => Some(format!(
                "acompressor=threshold={}:ratio={}",
                threshold, ratio
            )),
            ParsedFilter::Passthrough => None,
            ParsedFilter::Unknown { name, args } => {
                warn!(
                    "Skipping unsupported filter '{}' (args: '{}') during execution.",
                    name, args
                );
                eprintln!(
                    "{} Skipping unsupported filter '{}' during execution.",
                    "warning:".yellow().bold(),
                    name
                );
                None
            }
        })
        .collect();

    if parts.is_empty() {
        None
    } else {
        Some(parts.join(","))
    }
}

/// Extract the first `Scale` filter from a filter list and return it as a
/// `"W:H"` string suitable for `TranscodeOptions::scale`.
///
/// `-1` dimension values are rendered as `"-1"` (keep-aspect).
fn extract_scale_filter(filters: &[ParsedFilter]) -> Option<String> {
    filters.iter().find_map(|f| {
        if let ParsedFilter::Scale { w, h } = f {
            Some(format!("{}:{}", w, h))
        } else {
            None
        }
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Platform helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Return a reasonable default thread count for the current machine.
fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

// ─────────────────────────────────────────────────────────────────────────────
// Help text
// ─────────────────────────────────────────────────────────────────────────────

/// Print brief usage information for the ffcompat command.
fn print_ffcompat_help() {
    println!("{}", "OxiMedia FFmpeg-compatible interface".cyan().bold());
    println!();
    println!("Usage:");
    println!("  oximedia ffcompat [FFmpeg arguments...]");
    println!("  oximedia ff [FFmpeg arguments...]");
    println!("  oximedia-ff [FFmpeg arguments...]");
    println!();
    println!("Options (OxiMedia extensions):");
    println!("  --dry-run / --plan    Print what would be done without executing.");
    println!();
    println!("Examples:");
    println!("  oximedia ff -i input.mkv -c:v libaom-av1 -crf 28 -c:a libopus output.webm");
    println!("  oximedia ff -y -i input.mkv -vf scale=1280:720 -b:v 2M output.webm");
    println!(
        "  oximedia ff -i input.mp4 -c:v libx264 -c:a aac output.webm  # patent codecs auto-substituted"
    );
    println!();
    println!(
        "{}",
        "Note: Only patent-free codecs are supported (AV1, VP9, VP8, Opus, Vorbis, FLAC).".yellow()
    );
    println!(
        "{}",
        "      Patent-encumbered codecs (H.264, AAC, MP3, etc.) are automatically substituted."
            .yellow()
    );
}
