//! `oximedia-ff` — FFmpeg drop-in entry point for OxiMedia.
//!
//! This binary accepts raw FFmpeg-style arguments and passes them through the
//! OxiMedia FFmpeg-compat translation layer, then **executes** each resulting
//! transcode job using the native OxiMedia transcode subsystem. It can be
//! symlinked or aliased as `ffmpeg` to act as a transparent drop-in replacement
//! for patent-free workflows.
//!
//! ## Usage
//!
//! ```sh
//! oximedia-ff -i input.mkv -c:v libaom-av1 -crf 28 -c:a libopus output.webm
//! oximedia-ff -i input.mp4 -c:v libx264 output.webm   # libx264 auto-substituted with av1
//! oximedia-ff -y -i src.mkv -vf scale=1280:720 -b:v 2M out.webm
//! oximedia-ff --dry-run -i src.mkv -c:v av1 out.webm  # print plan only
//! ```

use colored::Colorize;
use oximedia_cli::transcode::{self, TranscodeOptions};
use oximedia_compat_ffmpeg::{parse_and_translate, ParsedFilter, TranscodeJob};
use std::path::PathBuf;
use tracing::warn;

#[tokio::main]
async fn main() {
    // Initialise a minimal tracing subscriber so warn!/info! calls are visible.
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::WARN)
        .with_writer(std::io::stderr)
        .try_init();

    // Skip argv[0] (the binary name itself).
    let args: Vec<String> = std::env::args().skip(1).collect();

    if args.is_empty() {
        print_help();
        std::process::exit(1);
    }
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        std::process::exit(0);
    }

    if args.iter().any(|a| a == "--version" || a == "-version") {
        println!(
            "oximedia-ff {} (OxiMedia FFmpeg-compat layer)",
            env!("CARGO_PKG_VERSION")
        );
        std::process::exit(0);
    }

    match run(&args).await {
        Ok(()) => std::process::exit(0),
        Err(e) => {
            eprintln!("{} {}", "oximedia-ff: fatal:".red().bold(), e);
            std::process::exit(1);
        }
    }
}

async fn run(args: &[String]) -> anyhow::Result<()> {
    // Strip --dry-run / --plan before passing to the FFmpeg compat parser.
    let dry_run = args
        .iter()
        .any(|a| a == "--dry-run" || a == "--plan" || a == "-dry-run");

    let filtered: Vec<String> = args
        .iter()
        .filter(|a| *a != "--dry-run" && *a != "--plan" && *a != "-dry-run")
        .cloned()
        .collect();

    let result = parse_and_translate(&filtered);

    // Print diagnostics to stderr in FFmpeg style.
    for diag in &result.diagnostics {
        let formatted = diag.format_ffmpeg_style("oximedia-ff");
        eprintln!("{}", formatted);
    }

    if result.has_errors() {
        anyhow::bail!("aborting due to errors — see diagnostics above");
    }

    if result.jobs.is_empty() {
        print_help();
        return Ok(());
    }

    // Print brief job plan.
    for (idx, job) in result.jobs.iter().enumerate() {
        eprintln!(
            "{} Job {}: {} {} {}",
            "oximedia-ff:".green().bold(),
            idx + 1,
            job.input_path.cyan(),
            "→".bold(),
            job.output_path.cyan()
        );

        if let Some(vc) = &job.video_codec {
            eprintln!("  video: {}", vc.green());
        }
        if let Some(ac) = &job.audio_codec {
            eprintln!("  audio: {}", ac.green());
        }
        if let Some(crf) = job.crf {
            eprintln!("  crf: {:.1}", crf);
        }
        if let Some(vb) = &job.video_bitrate {
            eprintln!("  video bitrate: {}", vb);
        }
        if let Some(ab) = &job.audio_bitrate {
            eprintln!("  audio bitrate: {}", ab);
        }
        if !job.video_filters.is_empty() {
            eprintln!("  video filters: {} applied", job.video_filters.len());
        }
        if !job.audio_filters.is_empty() {
            eprintln!("  audio filters: {} applied", job.audio_filters.len());
        }
        if let Some(seek) = &job.seek {
            eprintln!("  seek: {}", seek);
        }
        if let Some(dur) = &job.duration {
            eprintln!("  duration: {}", dur);
        }
        if job.overwrite {
            eprintln!("  {}", "overwrite: yes".dimmed());
        }
        for (k, v) in &job.metadata {
            eprintln!("  metadata: {}={}", k, v);
        }

        if dry_run {
            eprintln!("  {}", "[dry-run: skipping execution]".yellow().italic());
        }
    }

    if dry_run {
        eprintln!(
            "\n{} Dry-run mode — no files were written.",
            "oximedia-ff: note:".cyan()
        );
        return Ok(());
    }

    // Execute each job.
    for (idx, job) in result.jobs.iter().enumerate() {
        eprintln!(
            "\n{} Transcoding ({}/{}) …",
            "oximedia-ff:".green().bold(),
            idx + 1,
            result.jobs.len()
        );

        execute_job(job)
            .await
            .map_err(|e| anyhow::anyhow!("job {} failed: {}", idx + 1, e))?;
    }

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Job execution
// ─────────────────────────────────────────────────────────────────────────────

async fn execute_job(job: &TranscodeJob) -> anyhow::Result<()> {
    if !job.overwrite && std::path::Path::new(&job.output_path).exists() {
        anyhow::bail!(
            "Output file '{}' already exists. Pass -y to overwrite.",
            job.output_path
        );
    }

    let vf_string = build_filter_string(&job.video_filters);
    let af_string = build_filter_string(&job.audio_filters);

    let scale_from_filters = extract_scale_filter(&job.video_filters);

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

    let crf = job.crf.map(|c| c.round() as u32);

    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    let options = TranscodeOptions {
        input: PathBuf::from(&job.input_path),
        output: PathBuf::from(&job.output_path),
        preset_name: None,
        video_codec,
        audio_codec,
        video_bitrate: job.video_bitrate.clone(),
        audio_bitrate: job.audio_bitrate.clone(),
        scale: scale_from_filters,
        video_filter: vf_string,
        audio_filter: af_string,
        start_time: job.seek.clone(),
        duration: job.duration.clone(),
        framerate: None,
        preset: "medium".to_string(),
        two_pass: false,
        crf,
        threads,
        overwrite: job.overwrite,
        resume: false,
    };

    transcode::transcode(options).await
}

// ─────────────────────────────────────────────────────────────────────────────
// Filter helpers
// ─────────────────────────────────────────────────────────────────────────────

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
                    "oximedia-ff: warning:".yellow().bold(),
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
// Help text
// ─────────────────────────────────────────────────────────────────────────────

fn print_help() {
    println!(
        "{}",
        "oximedia-ff  —  FFmpeg-compatible OxiMedia front-end".bold()
    );
    println!();
    println!("Usage: oximedia-ff [options] -i <input> [options] <output>");
    println!();
    println!("Supported codecs (patent-free only — patent codecs are auto-substituted):");
    println!("  Video  (direct): av1 / libaom-av1, vp9 / libvpx-vp9, vp8 / libvpx");
    println!("  Video  (subst.): libx264/h264 → av1, libx265/hevc → av1, mpeg4 → av1");
    println!("  Audio  (direct): opus / libopus, vorbis / libvorbis, flac, pcm_*");
    println!("  Audio  (subst.): aac → opus, mp3/libmp3lame → opus, ac3 → flac");
    println!();
    println!("Options:");
    println!("  -y                  Overwrite output without asking");
    println!("  -n                  Never overwrite output");
    println!("  --dry-run / --plan  Print plan without executing");
    println!("  -i <path>           Input file");
    println!("  -c:v / -vcodec      Video codec");
    println!("  -c:a / -acodec      Audio codec");
    println!("  -c:s / -scodec      Subtitle codec");
    println!("  -b:v <rate>         Video bitrate (e.g. 2M, 500k)");
    println!("  -b:a <rate>         Audio bitrate (e.g. 128k)");
    println!("  -crf <n>            Quality (CRF) value");
    println!("  -vf <filter>        Video filter graph");
    println!("  -af <filter>        Audio filter graph");
    println!("  -filter_complex <g> Complex filter graph");
    println!("  -r <fps>            Frame rate");
    println!("  -ar <hz>            Audio sample rate");
    println!("  -ac <n>             Audio channel count");
    println!("  -s <WxH>            Video resolution");
    println!("  -ss <time>          Seek position");
    println!("  -t <duration>       Duration");
    println!("  -map <spec>         Stream mapping");
    println!("  -vn / -an / -sn     Disable video / audio / subtitle");
    println!("  -shortest           Stop when shortest stream ends");
    println!("  -metadata key=val   Set output metadata");
    println!("  -f <format>         Force container format");
    println!("  -threads <n>        Thread count");
    println!("  -hwaccel <method>   Hardware acceleration (parsed, not executed)");
    println!("  -loglevel <level>   Log level");
    println!();
    println!("Supported video filters (-vf / -filter_complex):");
    println!("  scale=W:H, crop=w:h:x:y, fps=N, hflip, vflip, rotate=angle");
    println!("  yadif/bwdif (deinterlace), eq=brightness:contrast:saturation");
    println!("  lut3d=file=x.cube, subtitles=file=x.srt");
    println!();
    println!("Supported audio filters (-af):");
    println!("  loudnorm=I=L:TP=T:LRA=R, volume=N/NdB");
    println!("  aresample=N, acompressor=threshold=T:ratio=R");
}
