//! CLI command definitions and sub-command enums.

use clap::Subcommand;
use std::path::PathBuf;

use crate::aaf_cmd;
use crate::access_cmd;
use crate::align_cmd;
use crate::archive_cmd;
use crate::archivepro_cmd;
use crate::audio_cmd;
use crate::audiopost_cmd;
use crate::auto_cmd;
use crate::calibrate_cmd;
use crate::clips_cmd;
use crate::cloud_cmd;
use crate::collab_cmd;
use crate::color_cmd;
use crate::conform_cmd;
use crate::dedup_cmd;
use crate::distributed_cmd;
use crate::dolbyvision_cmd;
use crate::drm_cmd;
use crate::edl_cmd;
use crate::farm_cmd;
use crate::filter_cmd;
use crate::gaming_cmd;
use crate::graphics_cmd;
use crate::image_cmd;
use crate::imf_cmd;
use crate::lut_cmd;
use crate::mam_cmd;
use crate::mir_cmd;
use crate::mixer_cmd;
use crate::multicam_cmd;
use crate::ndi_cmd;
use crate::optimize_cmd;
use crate::playlist_cmd;
use crate::playout_cmd;
use crate::plugin_cmd;
use crate::profiler_cmd;
use crate::proxy_cmd;
use crate::qc_cmd;
use crate::recommend_cmd;
use crate::renderfarm_cmd;
use crate::repair_cmd;
use crate::review_cmd;
use crate::rights_cmd;
use crate::routing_cmd;
use crate::scaling_cmd;
use crate::scene;
use crate::scopes_cmd;
use crate::search_cmd;
use crate::stream_cmd;
use crate::subtitle_cmd;
use crate::switcher_cmd;
use crate::timecode_cmd;
use crate::timeline_cmd;
use crate::timesync_cmd;
use crate::vfx_cmd;
use crate::videoip_cmd;
use crate::virtual_cmd;
use crate::watermark_cmd;
use crate::workflow_cmd;

/// Available CLI commands.
#[derive(Subcommand)]
pub(crate) enum Commands {
    /// Probe media file and show information
    Probe {
        /// Input file path
        #[arg(short, long)]
        input: PathBuf,

        /// Show detailed information
        #[arg(short = 'V', long)]
        verbose: bool,

        /// Show stream information
        #[arg(short, long)]
        streams: bool,

        /// Compute content hash/fingerprint
        #[arg(long)]
        hash: bool,

        /// Quick quality snapshot (no-reference metrics)
        #[arg(long)]
        quality_snapshot: bool,
    },

    /// Show supported formats and codecs
    Info,

    /// Transcode media file
    #[command(alias = "convert")]
    Transcode {
        /// Input file path (FFmpeg-compatible: -i)
        #[arg(short, long)]
        input: PathBuf,

        /// Output file path
        #[arg(short, long)]
        output: PathBuf,

        /// Use a preset (e.g., youtube-1080p, tv-4k)
        #[arg(long = "preset-name", conflicts_with_all = &["video_codec", "audio_codec", "video_bitrate", "audio_bitrate"])]
        preset_name: Option<String>,

        /// Video codec: av1, vp9, vp8 (FFmpeg-compatible: -c:v)
        #[arg(long = "codec", alias = "c:v")]
        video_codec: Option<String>,

        /// Audio codec: opus, vorbis, flac (FFmpeg-compatible: -c:a)
        #[arg(long, alias = "c:a")]
        audio_codec: Option<String>,

        /// Video bitrate (e.g., "2M", "500k") (FFmpeg-compatible: -b:v)
        #[arg(long = "bitrate", alias = "b:v")]
        video_bitrate: Option<String>,

        /// Audio bitrate (e.g., "128k") (FFmpeg-compatible: -b:a)
        #[arg(long, alias = "b:a")]
        audio_bitrate: Option<String>,

        /// Scale video (e.g., "1280:720", "1920:-1") (FFmpeg-compatible: -vf scale=)
        #[arg(long)]
        scale: Option<String>,

        /// Video filter chain (FFmpeg-compatible: -vf)
        #[arg(long, alias = "vf")]
        video_filter: Option<String>,

        /// Start time (seek) (FFmpeg-compatible: -ss)
        #[arg(long, alias = "ss")]
        start_time: Option<String>,

        /// Duration limit (FFmpeg-compatible: -t)
        #[arg(short = 't', long)]
        duration: Option<String>,

        /// Frame rate (e.g., "30", "23.976") (FFmpeg-compatible: -r)
        #[arg(short = 'r', long)]
        framerate: Option<String>,

        /// Encoder preset: ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow
        #[arg(long, default_value = "medium")]
        preset: String,

        /// Enable two-pass encoding
        #[arg(long)]
        two_pass: bool,

        /// CRF quality (0-63 for VP9/VP8, 0-255 for AV1, lower is better)
        #[arg(long)]
        crf: Option<u32>,

        /// Number of threads (0 = auto)
        #[arg(long, default_value = "0")]
        threads: usize,

        /// Overwrite output file without asking
        #[arg(short = 'y', long)]
        overwrite: bool,

        /// Resume from previous incomplete encode
        #[arg(long)]
        resume: bool,

        /// Audio filter chain (FFmpeg-compatible: -af)
        #[arg(long, alias = "af")]
        audio_filter: Option<String>,

        /// Stream mapping (e.g., "0:0", "0:1")
        #[arg(long)]
        map: Vec<String>,

        /// Normalize audio loudness
        #[arg(long)]
        normalize_audio: bool,
    },

    /// Extract frames from video
    Extract {
        /// Input video file
        #[arg(value_name = "INPUT")]
        input: PathBuf,

        /// Output pattern (e.g., "frame_%04d.png")
        #[arg(value_name = "OUTPUT_PATTERN")]
        output_pattern: String,

        /// Output format: png, jpg, ppm
        #[arg(short, long)]
        format: Option<String>,

        /// Start time (seek)
        #[arg(long, alias = "ss")]
        start_time: Option<String>,

        /// Number of frames to extract
        #[arg(short = 'n', long)]
        frames: Option<usize>,

        /// Extract every Nth frame
        #[arg(long, default_value = "1")]
        every: usize,

        /// Quality for JPEG output (0-100)
        #[arg(long, default_value = "90")]
        quality: u8,
    },

    /// Batch process multiple files
    Batch {
        /// Input directory
        #[arg(value_name = "INPUT_DIR")]
        input_dir: PathBuf,

        /// Output directory
        #[arg(value_name = "OUTPUT_DIR")]
        output_dir: PathBuf,

        /// Configuration file (TOML)
        #[arg(value_name = "CONFIG")]
        config: PathBuf,

        /// Number of parallel jobs (0 = auto)
        #[arg(short, long, default_value = "0")]
        jobs: usize,

        /// Continue on errors
        #[arg(long)]
        continue_on_error: bool,

        /// Dry run (show what would be done)
        #[arg(long)]
        dry_run: bool,
    },

    /// Concatenate multiple videos
    Concat {
        /// Input files to concatenate
        #[arg(value_name = "INPUTS", required = true, num_args = 2..)]
        inputs: Vec<PathBuf>,

        /// Output file path
        #[arg(short, long)]
        output: PathBuf,

        /// Concatenation method: simple, reencode, remux
        #[arg(long, default_value = "remux")]
        method: String,

        /// Validate input compatibility
        #[arg(long)]
        validate: bool,

        /// Overwrite output file without asking
        #[arg(short = 'y', long)]
        overwrite: bool,
    },

    /// Generate video thumbnails
    Thumbnail {
        /// Input video file
        #[arg(short, long)]
        input: PathBuf,

        /// Output file path
        #[arg(short, long)]
        output: PathBuf,

        /// Thumbnail mode: single, multiple, grid, auto
        #[arg(long, default_value = "auto")]
        mode: String,

        /// Timestamp for single mode (e.g., "30", "1:30", "1:01:30")
        #[arg(long)]
        timestamp: Option<String>,

        /// Number of thumbnails for multiple mode
        #[arg(long, default_value = "9")]
        count: usize,

        /// Grid rows for grid mode
        #[arg(long, default_value = "3")]
        rows: usize,

        /// Grid columns for grid mode
        #[arg(long, default_value = "3")]
        cols: usize,

        /// Thumbnail width (pixels)
        #[arg(long)]
        width: Option<u32>,

        /// Thumbnail height (pixels)
        #[arg(long)]
        height: Option<u32>,

        /// Output format: png, jpeg, webp
        #[arg(short, long, default_value = "png")]
        format: String,

        /// Quality for JPEG/WebP (0-100)
        #[arg(long, default_value = "90")]
        quality: u8,
    },

    /// Generate video thumbnail sprite sheet
    Sprite {
        /// Input video file
        #[arg(short, long)]
        input: PathBuf,

        /// Output sprite sheet file path
        #[arg(short, long)]
        output: PathBuf,

        /// Time interval between thumbnails in seconds (e.g., "10", "30")
        #[arg(long, conflicts_with = "count")]
        interval: Option<String>,

        /// Total number of thumbnails to generate
        #[arg(long, conflicts_with = "interval")]
        count: Option<usize>,

        /// Number of columns in grid
        #[arg(long, default_value = "5")]
        cols: usize,

        /// Number of rows in grid
        #[arg(long, default_value = "5")]
        rows: usize,

        /// Thumbnail width in pixels
        #[arg(long, default_value = "160")]
        width: u32,

        /// Thumbnail height in pixels
        #[arg(long, default_value = "90")]
        height: u32,

        /// Output format: png, jpeg, webp
        #[arg(short, long, default_value = "png")]
        format: String,

        /// Quality for JPEG/WebP (0-100)
        #[arg(long, default_value = "90")]
        quality: u8,

        /// Compression level (0-9)
        #[arg(long, default_value = "6")]
        compression: u8,

        /// Sampling strategy: uniform, scene-based, keyframe-only, smart
        #[arg(long, default_value = "uniform")]
        strategy: String,

        /// Layout mode: grid, vertical, horizontal, auto
        #[arg(long, default_value = "grid")]
        layout: String,

        /// Spacing between thumbnails in pixels
        #[arg(long, default_value = "2")]
        spacing: u32,

        /// Margin around sprite sheet in pixels
        #[arg(long, default_value = "0")]
        margin: u32,

        /// Generate WebVTT file for seeking
        #[arg(long)]
        vtt: bool,

        /// WebVTT output file path (default: `<output>`.vtt)
        #[arg(long, requires = "vtt")]
        vtt_output: Option<PathBuf>,

        /// Generate JSON manifest
        #[arg(long)]
        manifest: bool,

        /// JSON manifest output path (default: `<output>`.json)
        #[arg(long, requires = "manifest")]
        manifest_output: Option<PathBuf>,

        /// Show timestamps on thumbnails
        #[arg(long)]
        timestamps: bool,

        /// Maintain aspect ratio when scaling
        #[arg(long, default_value = "true")]
        aspect: bool,
    },

    /// Edit media metadata/tags
    Metadata {
        /// Input file
        #[arg(short, long)]
        input: PathBuf,

        /// Output file (defaults to input if not specified)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Show current metadata
        #[arg(long)]
        show: bool,

        /// Set metadata field (can be used multiple times: --set title="My Title")
        #[arg(long, value_parser = parse_key_val)]
        set: Vec<(String, String)>,

        /// Remove metadata field (can be used multiple times)
        #[arg(long)]
        remove: Vec<String>,

        /// Clear all metadata
        #[arg(long)]
        clear: bool,

        /// Copy metadata from another file
        #[arg(long)]
        copy_from: Option<PathBuf>,
    },

    /// Run encoding benchmarks
    Benchmark {
        /// Input file for benchmarking
        #[arg(short, long)]
        input: PathBuf,

        /// Codecs to test (e.g., av1, vp9, vp8)
        #[arg(long, default_values = &["av1", "vp9"])]
        codecs: Vec<String>,

        /// Presets to test (e.g., fast, medium, slow)
        #[arg(long, default_values = &["fast", "medium", "slow"])]
        presets: Vec<String>,

        /// Duration to encode in seconds (0 = full file)
        #[arg(long)]
        duration: Option<u64>,

        /// Number of iterations per configuration
        #[arg(long, default_value = "1")]
        iterations: usize,

        /// Output directory for benchmark files
        #[arg(long)]
        output_dir: Option<PathBuf>,
    },

    /// Validate file integrity
    Validate {
        /// Input files to validate
        #[arg(value_name = "INPUTS", required = true)]
        inputs: Vec<PathBuf>,

        /// Validation checks: format, codec, stream, corruption, metadata, all
        #[arg(long, default_values = &["all"])]
        checks: Vec<String>,

        /// Strict mode (fail on warnings)
        #[arg(long)]
        strict: bool,

        /// Attempt to fix issues
        #[arg(long)]
        fix: bool,

        /// Check loudness compliance
        #[arg(long)]
        loudness_check: bool,

        /// Check color gamut compliance
        #[arg(long)]
        gamut_check: bool,
    },

    /// Analyze video/audio quality metrics
    Analyze {
        /// Input file path
        #[arg(short, long)]
        input: PathBuf,

        /// Reference file for full-reference metrics
        #[arg(long)]
        reference: Option<PathBuf>,

        /// Metrics to compute (comma-separated: psnr,ssim,brisque,niqe,blockiness,blur,noise)
        #[arg(long, default_value = "brisque,niqe,blockiness,blur,noise")]
        metrics: String,

        /// Output format: text, json, csv
        #[arg(long, default_value = "text")]
        output_format: String,

        /// Per-frame analysis
        #[arg(long)]
        per_frame: bool,

        /// Show summary statistics
        #[arg(long)]
        summary: bool,
    },

    /// Scene detection, shot classification, and storyboard generation
    Scene {
        #[command(subcommand)]
        command: scene::SceneCommand,
    },

    /// Video scopes: waveform, vectorscope, histogram, parade, false color
    Scopes {
        #[command(subcommand)]
        command: scopes_cmd::ScopesCommand,
    },

    /// Audio loudness metering, normalization, spectrum, and beat detection
    Audio {
        #[command(subcommand)]
        command: audio_cmd::AudioCommand,
    },

    /// Subtitle conversion, extraction, burn-in, and synchronization
    Subtitle {
        #[command(subcommand)]
        command: subtitle_cmd::SubtitleCommand,
    },

    /// Standalone filter graph processing
    Filter {
        #[command(subcommand)]
        command: filter_cmd::FilterCommand,
    },

    /// Apply, inspect, convert, or generate LUT files
    Lut {
        #[command(subcommand)]
        command: lut_cmd::LutCommand,
    },

    /// Video denoising / noise reduction
    Denoise {
        /// Input file path
        #[arg(short, long)]
        input: PathBuf,

        /// Output file path
        #[arg(short, long)]
        output: PathBuf,

        /// Denoise mode: fast, balanced, quality, grain-aware
        #[arg(long, default_value = "balanced")]
        mode: String,

        /// Noise reduction strength (0.0-1.0)
        #[arg(long, default_value = "0.5")]
        strength: f32,

        /// Prefer spatial-only denoising
        #[arg(long)]
        spatial: bool,

        /// Prefer temporal-only denoising
        #[arg(long)]
        temporal: bool,

        /// Preserve film grain / noise texture
        #[arg(long)]
        preserve_grain: bool,
    },

    /// Video stabilisation (remove camera shake)
    Stabilize {
        /// Input file path
        #[arg(short, long)]
        input: PathBuf,

        /// Output file path
        #[arg(short, long)]
        output: PathBuf,

        /// Motion model: translation, affine, perspective, 3d
        #[arg(long, default_value = "affine")]
        mode: String,

        /// Quality preset: fast, balanced, maximum
        #[arg(long, default_value = "balanced")]
        quality: String,

        /// Smoothing window in frames
        #[arg(long, default_value = "30")]
        smoothing: u32,

        /// Auto-zoom to hide stabilisation borders
        #[arg(long)]
        zoom: bool,
    },

    /// EDL parse, validate, export, and conform
    Edl {
        #[command(subcommand)]
        command: edl_cmd::EdlCommand,
    },

    /// HLS / DASH adaptive-bitrate packaging
    Package {
        /// Input file path
        #[arg(short, long)]
        input: PathBuf,

        /// Output directory
        #[arg(short, long)]
        output: PathBuf,

        /// Packaging format: hls, hls-fmp4, dash
        #[arg(long, default_value = "hls")]
        format: String,

        /// Segment duration in seconds
        #[arg(long, default_value = "6")]
        segments: u32,

        /// Bitrate ladder: auto, or comma-separated like 1080p,720p,480p
        #[arg(long, default_value = "auto")]
        ladders: String,

        /// Encryption: none, aes128, sample-aes, cenc
        #[arg(long, default_value = "none")]
        encrypt: String,

        /// Enable low-latency mode
        #[arg(long)]
        low_latency: bool,
    },

    /// Media forensics: tamper detection, integrity analysis, provenance
    Forensics {
        /// Input media file
        #[arg(short, long)]
        input: PathBuf,

        /// Run all available forensic tests
        #[arg(long)]
        all: bool,

        /// Comma-separated list of tests: ela,noise,compression,splicing,metadata,tampering
        #[arg(long, default_value = "")]
        tests: String,

        /// Output format: text, json
        #[arg(long, default_value = "text")]
        output_format: String,

        /// Save detailed report to file
        #[arg(long)]
        report: Option<PathBuf>,
    },

    /// System monitoring: start, status, alerts, config, dashboard
    Monitor {
        #[command(subcommand)]
        command: MonitorCommand,
    },

    /// Audio/video restoration: restore degraded media
    Restore {
        #[command(subcommand)]
        command: RestoreCommand,
    },

    /// Caption processing: generate, sync, convert, burn, extract, validate
    Captions {
        #[command(subcommand)]
        command: CaptionsCommand,
    },

    /// HLS/DASH streaming: serve, ingest, record
    Stream {
        #[command(subcommand)]
        command: stream_cmd::StreamCommand,
    },

    /// Content search: text, visual similarity, fingerprint, index
    Search {
        #[command(subcommand)]
        command: search_cmd::SearchCommand,
    },

    /// Timecode convert, calculate, validate, burn-in
    Timecode {
        #[command(subcommand)]
        command: timecode_cmd::TimecodeCommand,
    },

    /// Media file repair and recovery
    Repair {
        #[command(subcommand)]
        command: repair_cmd::RepairCommand,
    },

    /// Color management: convert, info, matrix, Delta E
    Color {
        #[command(subcommand)]
        command: color_cmd::ColorCommand,
    },

    /// Playlist generation, validation, and playout automation
    Playlist {
        #[command(subcommand)]
        command: playlist_cmd::PlaylistSubcommand,
    },

    /// QC/conformance checking and fixing
    Conform {
        #[command(subcommand)]
        command: conform_cmd::ConformSubcommand,
    },

    /// IMF/archive packaging and extraction
    Archive {
        #[command(subcommand)]
        command: archive_cmd::ArchiveSubcommand,
    },

    /// Digital audio watermarking
    Watermark {
        #[command(subcommand)]
        command: watermark_cmd::WatermarkSubcommand,
    },

    /// Professional image operations (DPX, EXR, TIFF, sequences)
    Image {
        #[command(subcommand)]
        command: image_cmd::ImageCommand,
    },

    /// Broadcast graphics: lower-thirds, tickers, overlays, templates
    Graphics {
        #[command(subcommand)]
        command: graphics_cmd::GraphicsCommand,
    },

    /// Multi-camera: sync, switch, composite, color-match, export
    Multicam {
        #[command(subcommand)]
        command: multicam_cmd::MulticamCommand,
    },

    /// Timeline editing operations
    Timeline {
        #[command(subcommand)]
        command: timeline_cmd::TimelineCommand,
    },

    /// Visual effects and compositing
    Vfx {
        #[command(subcommand)]
        command: vfx_cmd::VfxCommand,
    },

    /// FFmpeg-compatible command interface (pass raw FFmpeg arguments)
    #[command(alias = "ff")]
    Ffcompat {
        /// FFmpeg-style arguments (passed through to compat layer)
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },

    /// Interactive terminal UI
    Tui,

    /// Codec optimization: complexity analysis, CRF sweep, quality ladder, benchmark
    Optimize {
        #[command(subcommand)]
        command: optimize_cmd::OptimizeCommand,
    },

    /// Manage transcoding presets
    Preset {
        #[command(subcommand)]
        command: PresetCommand,
    },

    /// Audio mixer: create, add-channel, route, render, info
    Mixer {
        #[command(subcommand)]
        command: mixer_cmd::MixerCommand,
    },

    /// Audio post-production: ADR, mix, stems, delivery, restore
    Audiopost {
        #[command(subcommand)]
        command: audiopost_cmd::AudiopostCommand,
    },

    /// Distributed encoding: coordinator, workers, job submission
    Distributed {
        #[command(subcommand)]
        command: distributed_cmd::DistributedCommand,
    },

    /// Render farm: start, submit, status, cancel, nodes
    Farm {
        #[command(subcommand)]
        command: farm_cmd::FarmCommand,
    },

    /// NDI source discovery, streaming, and monitoring
    Ndi {
        #[command(subcommand)]
        command: ndi_cmd::NdiCommand,
    },

    /// Video-over-IP: send, receive, discover, monitor (RTP/SRT/RIST)
    Videoip {
        #[command(subcommand)]
        command: videoip_cmd::VideoIpCommand,
    },

    /// Gaming: capture, clip, overlay
    Gaming {
        #[command(subcommand)]
        command: gaming_cmd::GamingCommand,
    },

    /// Media Asset Management: ingest, search, catalog, export, tag
    Mam {
        #[command(subcommand)]
        command: mam_cmd::MamCommand,
    },

    /// Cloud storage: upload, download, transcode, status, cost
    Cloud {
        #[command(subcommand)]
        command: cloud_cmd::CloudCommand,
    },

    /// Plugin management: list, info, codecs, validate, paths
    Plugin {
        #[command(subcommand)]
        command: plugin_cmd::PluginCommand,
    },

    /// Music Information Retrieval: tempo, key, chords, segmentation, analysis
    Mir {
        #[command(subcommand)]
        command: mir_cmd::MirCommand,
    },

    /// Quality Control: check, validate, report, rules, fix
    Qc {
        #[command(subcommand)]
        command: qc_cmd::QcCommand,
    },

    /// IMF: validate, package, info, extract, create
    Imf {
        #[command(subcommand)]
        command: imf_cmd::ImfCommand,
    },

    /// AAF: info, extract, convert, validate, merge
    Aaf {
        #[command(subcommand)]
        command: aaf_cmd::AafCommand,
    },

    /// Broadcast playout: schedule, start, stop, status, load, next, list
    Playout {
        #[command(subcommand)]
        command: playout_cmd::PlayoutCommand,
    },

    /// Live production switcher: create, add-source, switch, preview, record, macro
    Switcher {
        #[command(subcommand)]
        command: switcher_cmd::SwitcherCommand,
    },

    /// Workflow orchestration: create, run, status, list, cancel, template
    Workflow {
        #[command(subcommand)]
        command: workflow_cmd::WorkflowCommand,
    },

    /// Collaboration: create, join, share, comment, export, status
    Collab {
        #[command(subcommand)]
        command: collab_cmd::CollabCommand,
    },

    /// Proxy media: generate, list, link, info, clean
    Proxy {
        #[command(subcommand)]
        command: proxy_cmd::ProxyCommand,
    },

    /// Clips: create, list, export, trim, merge, tag
    Clips {
        #[command(subcommand)]
        command: clips_cmd::ClipsCommand,
    },

    /// Review: create, annotate, approve, reject, export, status
    Review {
        #[command(subcommand)]
        command: review_cmd::ReviewCommand,
    },

    /// DRM: encrypt, decrypt, keys, info, validate
    Drm {
        #[command(subcommand)]
        command: drm_cmd::DrmCommand,
    },

    /// Dedup: scan, report, clean, hash, compare
    Dedup {
        #[command(subcommand)]
        command: dedup_cmd::DedupCommand,
    },

    /// Archive Pro: ingest, verify, migrate, report, policy
    #[command(name = "archive-pro")]
    ArchivePro {
        #[command(subcommand)]
        command: archivepro_cmd::ArchiveProCommand,
    },

    /// Dolby Vision: analyze, convert, metadata, validate, info
    #[command(name = "dolby-vision")]
    DolbyVision {
        #[command(subcommand)]
        command: dolbyvision_cmd::DolbyVisionCommand,
    },

    /// Time sync: analyze, align, offset, drift, report
    #[command(name = "timesync")]
    TimeSync {
        #[command(subcommand)]
        command: timesync_cmd::TimeSyncCommand,
    },

    /// Align: audio, video, sync, offset, detect
    Align {
        #[command(subcommand)]
        command: align_cmd::AlignCommand,
    },

    /// Routing: create, add-node, connect, info, validate, list
    Routing {
        #[command(subcommand)]
        command: routing_cmd::RoutingCommand,
    },

    /// Calibrate: display, audio, color, generate-pattern, report
    Calibrate {
        #[command(subcommand)]
        command: calibrate_cmd::CalibrateCommand,
    },

    /// Virtual production: create, list, start, stop, configure
    Virtual {
        #[command(subcommand)]
        command: virtual_cmd::VirtualCommand,
    },

    /// Profiler: run, report, compare, export, bottleneck
    Profiler {
        #[command(subcommand)]
        command: profiler_cmd::ProfilerCommand,
    },

    /// Recommend: codec, settings, workflow, analyze
    Recommend {
        #[command(subcommand)]
        command: recommend_cmd::RecommendCommand,
    },

    /// Scaling: upscale, downscale, analyze, compare, batch
    Scaling {
        #[command(subcommand)]
        command: scaling_cmd::ScalingCommand,
    },

    /// Render farm cluster: init, add-node, remove-node, submit, status, dashboard
    Renderfarm {
        #[command(subcommand)]
        command: renderfarm_cmd::RenderfarmCommand,
    },

    /// Access control: grant, revoke, list, policy, audit, check
    Access {
        #[command(subcommand)]
        command: access_cmd::AccessCommand,
    },

    /// Digital rights: register, check, transfer, license, report, search
    Rights {
        #[command(subcommand)]
        command: rights_cmd::RightsCommand,
    },

    /// Auto editing: run, schedule, list, create, delete, log
    Auto {
        #[command(subcommand)]
        command: auto_cmd::AutoCommand,
    },
}

/// Monitor subcommands.
#[derive(Subcommand)]
pub(crate) enum MonitorCommand {
    /// Start monitoring a stream or file
    Start {
        /// Target to monitor (file path or stream URL)
        #[arg(value_name = "TARGET")]
        target: String,

        /// Database path for metric storage
        #[arg(long)]
        db_path: Option<PathBuf>,

        /// Metrics collection interval in milliseconds
        #[arg(long, default_value = "1000")]
        interval_ms: u64,

        /// Enable system metrics (CPU, memory, disk)
        #[arg(long)]
        system_metrics: bool,

        /// Enable quality metrics (PSNR, SSIM, bitrate)
        #[arg(long)]
        quality_metrics: bool,
    },

    /// Show current monitoring status
    Status {
        /// Database path
        #[arg(long)]
        db_path: Option<PathBuf>,

        /// Show detailed per-component status
        #[arg(long)]
        detailed: bool,
    },

    /// Show recent alerts
    Alerts {
        /// Database path
        #[arg(long)]
        db_path: Option<PathBuf>,

        /// Number of recent alerts to show
        #[arg(long, default_value = "20")]
        count: usize,

        /// Filter by severity: info, warning, error, critical
        #[arg(long)]
        severity: Option<String>,
    },

    /// Configure monitoring thresholds
    Config {
        /// Database path
        #[arg(long)]
        db_path: Option<PathBuf>,

        /// CPU alert threshold (percentage)
        #[arg(long)]
        cpu_threshold: Option<f64>,

        /// Memory alert threshold (percentage)
        #[arg(long)]
        memory_threshold: Option<f64>,

        /// Quality score alert threshold (0-100)
        #[arg(long)]
        quality_threshold: Option<f64>,

        /// Show current configuration only
        #[arg(long)]
        show: bool,
    },

    /// Display monitoring dashboard
    Dashboard {
        /// Database path
        #[arg(long)]
        db_path: Option<PathBuf>,

        /// Refresh interval in seconds
        #[arg(long, default_value = "5")]
        refresh_secs: u64,

        /// Number of history points to display
        #[arg(long, default_value = "60")]
        history_points: usize,
    },
}

/// Restore subcommands.
#[derive(Subcommand)]
pub(crate) enum RestoreCommand {
    /// Restore degraded audio
    Audio {
        /// Input audio file
        #[arg(short, long)]
        input: PathBuf,

        /// Output file
        #[arg(short, long)]
        output: PathBuf,

        /// Restoration mode: vinyl, tape, broadcast, archival, custom
        #[arg(long, default_value = "vinyl")]
        mode: String,

        /// Sample rate override (Hz)
        #[arg(long)]
        sample_rate: Option<u32>,

        /// Enable declipping
        #[arg(long)]
        declip: bool,

        /// Enable decrackle
        #[arg(long)]
        decrackle: bool,

        /// Enable hum removal
        #[arg(long)]
        dehum: bool,

        /// Enable noise reduction
        #[arg(long)]
        denoise: bool,
    },

    /// Restore degraded video
    Video {
        /// Input video file
        #[arg(short, long)]
        input: PathBuf,

        /// Output file
        #[arg(short, long)]
        output: PathBuf,

        /// Restoration mode: deinterlace, upscale, stabilize, color-correct, full
        #[arg(long, default_value = "full")]
        mode: String,

        /// Target width
        #[arg(long)]
        width: Option<u32>,

        /// Target height
        #[arg(long)]
        height: Option<u32>,
    },

    /// Analyze degradation type and severity
    Analyze {
        /// Input file
        #[arg(short, long)]
        input: PathBuf,

        /// Analysis type: audio, video, auto
        #[arg(long, default_value = "auto")]
        analysis_type: String,
    },

    /// Batch restore multiple files
    Batch {
        /// Input directory
        #[arg(short, long)]
        input_dir: PathBuf,

        /// Output directory
        #[arg(short, long)]
        output_dir: PathBuf,

        /// Restoration mode
        #[arg(long, default_value = "vinyl")]
        mode: String,

        /// File extension filter
        #[arg(long)]
        extension: Option<String>,
    },

    /// Compare before/after quality
    Compare {
        /// Original (degraded) file
        #[arg(long)]
        original: PathBuf,

        /// Restored file
        #[arg(long)]
        restored: PathBuf,
    },
}

/// Captions subcommands.
#[derive(Subcommand)]
pub(crate) enum CaptionsCommand {
    /// Generate captions from audio
    Generate {
        /// Input audio/video file
        #[arg(short, long)]
        input: PathBuf,

        /// Output caption file
        #[arg(short, long)]
        output: PathBuf,

        /// Output format: srt, vtt, ass, ttml, scc
        #[arg(long, default_value = "srt")]
        format: String,

        /// Language code (e.g. en, ja)
        #[arg(long, default_value = "en")]
        language: String,
    },

    /// Synchronize captions to audio
    Sync {
        /// Input caption file
        #[arg(short, long)]
        input: PathBuf,

        /// Reference audio/video file
        #[arg(long)]
        reference: PathBuf,

        /// Output synced caption file
        #[arg(short, long)]
        output: PathBuf,

        /// Maximum time shift in milliseconds
        #[arg(long, default_value = "5000")]
        max_shift_ms: i64,
    },

    /// Convert between caption formats
    Convert {
        /// Input caption file
        #[arg(short, long)]
        input: PathBuf,

        /// Output file
        #[arg(short, long)]
        output: PathBuf,

        /// Source format (auto-detected if not specified)
        #[arg(long)]
        from_format: Option<String>,

        /// Target format: srt, vtt, ass, ttml, scc
        #[arg(long)]
        to_format: String,
    },

    /// Burn captions into video
    Burn {
        /// Input video file
        #[arg(long)]
        video: PathBuf,

        /// Input caption file
        #[arg(long)]
        captions: PathBuf,

        /// Output video file
        #[arg(short, long)]
        output: PathBuf,

        /// Font size
        #[arg(long, default_value = "24")]
        font_size: u32,

        /// Font color (hex)
        #[arg(long, default_value = "FFFFFF")]
        font_color: String,
    },

    /// Extract captions from media
    Extract {
        /// Input media file
        #[arg(short, long)]
        input: PathBuf,

        /// Output caption file
        #[arg(short, long)]
        output: PathBuf,

        /// Output format
        #[arg(long, default_value = "srt")]
        format: String,

        /// Track index to extract
        #[arg(long, default_value = "0")]
        track: usize,
    },

    /// Validate caption file against standards
    Validate {
        /// Input caption file
        #[arg(short, long)]
        input: PathBuf,

        /// Standard: fcc, wcag, cea608, cea708, ebu
        #[arg(long, default_value = "fcc")]
        standard: String,

        /// Save report to file
        #[arg(long)]
        report: Option<PathBuf>,
    },
}

/// Preset management subcommands.
#[derive(Subcommand)]
pub(crate) enum PresetCommand {
    /// List all available presets
    List {
        /// Filter by category (web, device, quality, archival, streaming, custom)
        #[arg(short, long)]
        category: Option<String>,

        /// Show detailed information
        #[arg(short, long)]
        verbose: bool,
    },

    /// Show detailed information about a preset
    Show {
        /// Preset name
        #[arg(value_name = "NAME")]
        name: String,

        /// Output as TOML
        #[arg(long)]
        toml: bool,
    },

    /// Create a new custom preset interactively
    Create {
        /// Output directory for custom presets
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Generate a preset template file
    Template {
        /// Output file path
        #[arg(value_name = "OUTPUT")]
        output: PathBuf,
    },

    /// Import a preset from a TOML file
    Import {
        /// Input TOML file
        #[arg(value_name = "FILE")]
        file: PathBuf,
    },

    /// Export a preset to a TOML file
    Export {
        /// Preset name
        #[arg(value_name = "NAME")]
        name: String,

        /// Output file path
        #[arg(value_name = "OUTPUT")]
        output: PathBuf,
    },

    /// Remove a custom preset
    Remove {
        /// Preset name
        #[arg(value_name = "NAME")]
        name: String,
    },
}

/// Parse key=value pairs for metadata setting.
pub(crate) fn parse_key_val(s: &str) -> Result<(String, String), String> {
    let pos = s
        .find('=')
        .ok_or_else(|| format!("Invalid KEY=value: no `=` found in `{}`", s))?;
    Ok((s[..pos].to_string(), s[pos + 1..].to_string()))
}
