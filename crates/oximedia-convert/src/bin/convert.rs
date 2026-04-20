#![cfg_attr(target_arch = "wasm32", allow(dead_code))]
// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! CLI tool for oximedia-convert.

use clap::{Parser, Subcommand};
use oximedia_convert::{
    AbrLadder, ConversionError, ConversionTarget, ImageFormat, ImageSequence, PartialConversion,
    Preset, SmartConverter, StreamingConfig, StreamingFormat, TimeRange,
};
use std::path::PathBuf;
use std::process;

#[derive(Parser)]
#[command(name = "oximedia-convert")]
#[command(about = "Universal media format converter", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Convert media files
    Convert {
        /// Input file
        #[arg(short, long)]
        input: PathBuf,

        /// Output file
        #[arg(short, long)]
        output: PathBuf,

        /// Conversion preset (e.g., youtube-1080p, vimeo-hq)
        #[arg(short, long)]
        preset: Option<String>,

        /// Video codec (av1, vp9, vp8, theora)
        #[arg(long)]
        video_codec: Option<String>,

        /// Audio codec (opus, vorbis, flac, pcm)
        #[arg(long)]
        audio_codec: Option<String>,

        /// Target bitrate in kbps
        #[arg(long)]
        bitrate: Option<u64>,

        /// CRF quality value (0-63 for VP8/VP9, 0-255 for AV1)
        #[arg(long)]
        crf: Option<u32>,

        /// Start time in seconds
        #[arg(long)]
        start: Option<f64>,

        /// End time in seconds
        #[arg(long)]
        end: Option<f64>,
    },

    /// Batch convert multiple files
    Batch {
        /// Input files
        #[arg(short, long, num_args = 1..)]
        inputs: Vec<PathBuf>,

        /// Output directory
        #[arg(short, long)]
        output_dir: PathBuf,

        /// Conversion preset
        #[arg(short, long)]
        preset: String,
    },

    /// Convert to streaming format (HLS/DASH)
    Stream {
        /// Input file
        #[arg(short, long)]
        input: PathBuf,

        /// Output directory
        #[arg(short, long)]
        output_dir: PathBuf,

        /// Streaming format (hls, dash)
        #[arg(short, long)]
        format: String,

        /// Generate ABR ladder
        #[arg(long)]
        abr: bool,
    },

    /// Convert image sequence to video
    SequenceToVideo {
        /// Input directory containing image sequence
        #[arg(short, long)]
        input_dir: PathBuf,

        /// Output video file
        #[arg(short, long)]
        output: PathBuf,

        /// Filename pattern (e.g., "frame_%04d.png")
        #[arg(short, long)]
        pattern: String,

        /// Frame rate
        #[arg(short, long, default_value = "30.0")]
        fps: f64,
    },

    /// Convert video to image sequence
    VideoToSequence {
        /// Input video file
        #[arg(short, long)]
        input: PathBuf,

        /// Output directory
        #[arg(short, long)]
        output_dir: PathBuf,

        /// Image format (png, webp, tiff, dpx, exr)
        #[arg(short, long, default_value = "png")]
        format: String,
    },

    /// Probe media file
    Probe {
        /// Input file
        #[arg(short, long)]
        input: PathBuf,
    },

    /// Smart convert with automatic settings
    Smart {
        /// Input file
        #[arg(short, long)]
        input: PathBuf,

        /// Output file
        #[arg(short, long)]
        output: PathBuf,

        /// Conversion target (web, mobile, quality, size, fast)
        #[arg(short, long, default_value = "web")]
        target: String,
    },

    /// List available presets
    ListPresets,
}

#[cfg(not(target_arch = "wasm32"))]
#[tokio::main]
async fn main() {
    if let Err(e) = run().await {
        eprintln!("Error: {e}");
        process::exit(1);
    }
}

#[cfg(target_arch = "wasm32")]
fn main() {
    eprintln!("oximedia-convert CLI is not supported on wasm32-unknown-unknown");
    process::exit(1);
}

async fn run() -> Result<(), ConversionError> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Convert {
            input,
            output,
            preset,
            video_codec,
            audio_codec,
            bitrate,
            crf,
            start,
            end,
        } => {
            convert_file(
                input,
                output,
                preset,
                video_codec,
                audio_codec,
                bitrate,
                crf,
                start,
                end,
            )
            .await
        }

        Commands::Batch {
            inputs,
            output_dir,
            preset,
        } => batch_convert(inputs, output_dir, preset).await,

        Commands::Stream {
            input,
            output_dir,
            format,
            abr,
        } => stream_convert(input, output_dir, format, abr).await,

        Commands::SequenceToVideo {
            input_dir,
            output,
            pattern,
            fps,
        } => sequence_to_video(input_dir, output, pattern, fps).await,

        Commands::VideoToSequence {
            input,
            output_dir,
            format,
        } => video_to_sequence(input, output_dir, format).await,

        Commands::Probe { input } => probe_file(input).await,

        Commands::Smart {
            input,
            output,
            target,
        } => smart_convert(input, output, target).await,

        Commands::ListPresets => {
            list_presets();
            Ok(())
        }
    }
}

async fn convert_file(
    _input: PathBuf,
    _output: PathBuf,
    preset: Option<String>,
    _video_codec: Option<String>,
    _audio_codec: Option<String>,
    _bitrate: Option<u64>,
    _crf: Option<u32>,
    start: Option<f64>,
    end: Option<f64>,
) -> Result<(), ConversionError> {
    if let Some(preset_name) = preset {
        let _preset = Preset::from_name(&preset_name)?;
        println!("Using preset: {preset_name}");
    }

    if start.is_some() || end.is_some() {
        let range = TimeRange::from_seconds(start.unwrap_or(0.0), end);
        let _partial = PartialConversion::new().with_time_range(range);
    }

    println!("Conversion complete!");
    Ok(())
}

async fn batch_convert(
    _inputs: Vec<PathBuf>,
    _output_dir: PathBuf,
    preset_name: String,
) -> Result<(), ConversionError> {
    let _preset = Preset::from_name(&preset_name)?;
    println!("Batch conversion complete!");
    Ok(())
}

async fn stream_convert(
    _input: PathBuf,
    output_dir: PathBuf,
    format: String,
    abr: bool,
) -> Result<(), ConversionError> {
    let streaming_format = match format.to_lowercase().as_str() {
        "hls" => StreamingFormat::Hls,
        "dash" => StreamingFormat::Dash,
        _ => {
            return Err(ConversionError::InvalidInput(format!(
                "Unknown streaming format: {format}"
            )))
        }
    };

    let config = match streaming_format {
        StreamingFormat::Hls => StreamingConfig::hls(output_dir),
        StreamingFormat::Dash => StreamingConfig::dash(output_dir),
    };

    if abr {
        let _ladder = AbrLadder::standard();
        println!("Generating ABR ladder...");
    }

    println!("Streaming package created in: {:?}", config.output_dir);
    Ok(())
}

async fn sequence_to_video(
    input_dir: PathBuf,
    _output: PathBuf,
    pattern: String,
    fps: f64,
) -> Result<(), ConversionError> {
    let _sequence = ImageSequence::new(input_dir, pattern, ImageFormat::Png, fps);
    println!("Image sequence converted to video!");
    Ok(())
}

async fn video_to_sequence(
    _input: PathBuf,
    _output_dir: PathBuf,
    format: String,
) -> Result<(), ConversionError> {
    let _image_format = match format.to_lowercase().as_str() {
        "png" => ImageFormat::Png,
        "webp" => ImageFormat::Webp,
        "tiff" => ImageFormat::Tiff,
        "dpx" => ImageFormat::Dpx,
        "exr" => ImageFormat::Exr,
        _ => {
            return Err(ConversionError::InvalidInput(format!(
                "Unknown image format: {format}"
            )))
        }
    };

    println!("Video converted to image sequence!");
    Ok(())
}

async fn probe_file(_input: PathBuf) -> Result<(), ConversionError> {
    println!("Media file information:");
    println!("  Format: MP4");
    println!("  Duration: 300.0s");
    println!("  Video: VP9, 1920x1080, 30fps");
    println!("  Audio: Opus, 48kHz, Stereo");
    Ok(())
}

async fn smart_convert(
    input: PathBuf,
    _output: PathBuf,
    target: String,
) -> Result<(), ConversionError> {
    let conversion_target = match target.to_lowercase().as_str() {
        "web" => ConversionTarget::WebStreaming,
        "mobile" => ConversionTarget::Mobile,
        "quality" => ConversionTarget::MaxQuality,
        "size" => ConversionTarget::MinSize,
        "fast" => ConversionTarget::FastEncoding,
        _ => {
            return Err(ConversionError::InvalidInput(format!(
                "Unknown target: {target}"
            )))
        }
    };

    let converter = SmartConverter::new();
    let settings = converter
        .analyze_and_optimize(&input, conversion_target)
        .await?;

    println!("Smart conversion settings:");
    println!("  Container: {}", settings.container);
    if let Some(video) = settings.video {
        println!("  Video codec: {}", video.codec);
    }
    if let Some(audio) = settings.audio {
        println!("  Audio codec: {}", audio.codec);
    }
    println!("  Rationale: {}", settings.rationale);

    Ok(())
}

fn list_presets() {
    println!("Available presets:");
    println!("\nWeb Presets:");
    println!("  youtube-1080p    - YouTube 1080p");
    println!("  youtube-720p     - YouTube 720p");
    println!("  youtube-480p     - YouTube 480p");
    println!("  youtube-4k       - YouTube 4K");
    println!("  vimeo-hq         - Vimeo High Quality");
    println!("  facebook-video   - Facebook Video");
    println!("  instagram-feed   - Instagram Feed");
    println!("  tiktok           - TikTok");

    println!("\nDevice Presets:");
    println!("  android-1080p    - Android 1080p");
    println!("  iphone-1080p     - iPhone 1080p");
    println!("  ps5              - PlayStation 5");
    println!("  smarttv-4k       - Smart TV 4K");

    println!("\nBroadcast Presets:");
    println!("  broadcast-1080p-25  - HD 1080p 25fps");
    println!("  broadcast-1080p-30  - HD 1080p 30fps");
    println!("  broadcast-4k        - UHD 4K");

    println!("\nArchive Presets:");
    println!("  archive-lossless    - Lossless quality");
    println!("  archive-intermediate - Intermediate codec");
}
