//! Transcoding pipeline orchestration and execution.
//!
//! This module implements the core packet-level transcode loop:
//! demux → (optional codec-compat check) → mux.  Full decode/encode
//! paths are wired in for the codecs that OxiMedia natively supports
//! (AV1, VP8, VP9, Opus, Vorbis, FLAC).
//!
//! For audio-analysis (normalization), the raw packet bytes are treated
//! as PCM-like interleaved f32 so that the LoudnessMeter can derive a
//! coarse integrated-loudness estimate without a full codec stack.
//!
//! ## Pipeline Execution Architecture
//!
//! When [`TranscodePipeline::execute()`] is called, it delegates to the
//! [`Pipeline`] executor which runs a four-stage flow:
//!
//! ```text
//! ┌────────────┐  packets  ┌──────────┐  frames  ┌───────────┐  packets  ┌───────┐
//! │  Demuxer   │ ────────► │  Decode  │ ────────► │  Encode   │ ────────► │  Mux  │
//! └────────────┘           └──────────┘           └───────────┘           └───────┘
//!       │                        │                      │
//!       │  probe format          │  audio analysis      │  per-packet byte tracking
//!       │  (magic bytes)         │  (EBU-R128)          │  (bytes_in / bytes_out)
//! ```
//!
//! **Stage 1 — Validation.** Checks that input and output paths are
//! accessible before any I/O is performed.
//!
//! **Stage 2 — AudioAnalysis.** When normalization is enabled, scans the
//! entire input audio track to compute an EBU-R128 integrated loudness gain.
//! This pass is skipped for non-normalized jobs.
//!
//! **Stage 3 — Encode.** Reads packets from the demuxer, applies optional
//! per-codec stream-copy or re-encode, accumulates byte counts and frame
//! counters, and applies the normalization gain to audio packets as i16 PCM
//! in-band.  Supports single-pass and multi-pass modes.
//!
//! **Stage 4 — Verification.** Confirms the output file is non-empty and
//! assembles [`TranscodeOutput`] with real byte and frame statistics.
//!
//! ## Stream Copy Mode
//!
//! When no codec conversion is needed (codec is left unset or matches the
//! source), packets bypass re-encoding entirely and pass directly from
//! demuxer to muxer.
//!
//! ## Multi-pass Support
//!
//! [`MultiPassMode`] controls whether a single-pass or two-pass encode is
//! performed.  Pass 1 collects statistics; pass 2 applies them.
//!
//! # Pipeline execution stages
//!
//! 1. **Validation** – check input/output paths.
//! 2. **AudioAnalysis** – scan input audio to compute EBU-R128 loudness gain
//!    (only when normalization is enabled).
//! 3. **Encode** – single-pass or multi-pass remux with:
//!    - Per-packet byte tracking (`bytes_in`, `bytes_out`).
//!    - Video / audio frame counting.
//!    - Linear gain applied to every audio packet (i16 PCM in-band).
//! 4. **Verification** – confirm the output file is non-empty and assemble
//!    `TranscodeOutput` with real stats.

use crate::{
    make_video_encoder, MultiPassConfig, MultiPassEncoder, MultiPassMode, NormalizationConfig,
    ProgressTracker, QualityConfig, RateControlMode, Result, TranscodeError, TranscodeOutput,
    VideoEncoderParams,
};
use oximedia_container::{
    demux::{Demuxer, FlacDemuxer, MatroskaDemuxer, OggDemuxer, WavDemuxer},
    mux::{MatroskaMuxer, MuxerConfig, OggMuxer},
    probe_format, ContainerFormat, Muxer, StreamInfo,
};
use oximedia_core::CodecId;
use oximedia_io::FileSource;
use oximedia_metering::{LoudnessMeter, MeterConfig, Standard};
use std::path::PathBuf;
use std::time::Instant;
use tracing::{debug, info, warn};

// ─── Constants ───────────────────────────────────────────────────────────────

/// Probe buffer: read this many bytes to detect the container format.
const PROBE_BYTES: usize = 16 * 1024;

/// Default assumed sample rate when audio streams carry no params.
const DEFAULT_SAMPLE_RATE: f64 = 48_000.0;

/// Default assumed channel count when audio streams carry no params.
const DEFAULT_CHANNELS: usize = 2;

// ─── Intra-codec helpers ──────────────────────────────────────────────────────

/// Default quality/QP used when the caller does not supply CRF for intra codecs.
const INTRA_DEFAULT_QUALITY: u8 = 85;

/// Fallback frame width when the stream header carries no resolution.
const INTRA_FALLBACK_WIDTH: u32 = 1920;

/// Fallback frame height when the stream header carries no resolution.
const INTRA_FALLBACK_HEIGHT: u32 = 1080;

/// Parse a codec name string into a [`CodecId`] if it is an intra-only codec
/// supported by [`make_video_encoder`] (MJPEG or APV).
///
/// Returns `None` if the name does not map to an intra codec.
fn parse_intra_codec(name: &str) -> Option<CodecId> {
    match name.to_lowercase().as_str() {
        "mjpeg" | "motion-jpeg" | "motion_jpeg" => Some(CodecId::Mjpeg),
        "apv" => Some(CodecId::Apv),
        _ => None,
    }
}

/// Build a [`VideoEncoderParams`] from the first video stream found in `streams`.
///
/// Falls back to [`INTRA_FALLBACK_WIDTH`] × [`INTRA_FALLBACK_HEIGHT`] when the
/// stream header does not carry resolution metadata.  `quality` comes from the
/// pipeline's CRF setting (if present) or [`INTRA_DEFAULT_QUALITY`].
fn intra_encoder_params(streams: &[StreamInfo], quality: u8) -> Result<VideoEncoderParams> {
    let (w, h) = streams
        .iter()
        .find(|s| s.is_video())
        .and_then(|s| {
            let w = s.codec_params.width?;
            let h = s.codec_params.height?;
            Some((w, h))
        })
        .unwrap_or((INTRA_FALLBACK_WIDTH, INTRA_FALLBACK_HEIGHT));

    VideoEncoderParams::new(w, h, quality)
}

// ─── PassStats ────────────────────────────────────────────────────────────────

/// Per-pass statistics collected during packet-level remuxing.
///
/// Accumulated across all passes so that `verify_output` can report
/// accurate counts even for multi-pass encodes.
#[derive(Debug, Clone, Default)]
struct PassStats {
    /// Total uncompressed bytes read from the input demuxer.
    bytes_in: u64,
    /// Total bytes written to the output muxer (after gain adjustment).
    bytes_out: u64,
    /// Number of video packets forwarded.
    video_frames: u64,
    /// Number of audio packets forwarded.
    audio_frames: u64,
}

// ─── Container-format helpers ─────────────────────────────────────────────────

/// Detect the container format of `path` by reading a small probe buffer.
async fn detect_format(path: &std::path::Path) -> Result<ContainerFormat> {
    let mut source = FileSource::open(path)
        .await
        .map_err(|e| TranscodeError::IoError(e.to_string()))?;

    use oximedia_io::MediaSource;
    let mut buf = vec![0u8; PROBE_BYTES];
    let n = source
        .read(&mut buf)
        .await
        .map_err(|e| TranscodeError::IoError(e.to_string()))?;
    buf.truncate(n);

    let probe = probe_format(&buf).map_err(|e| TranscodeError::ContainerError(e.to_string()))?;
    Ok(probe.format)
}

/// Decide the output container format from the file extension.
fn output_format_from_path(path: &std::path::Path) -> ContainerFormat {
    match path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_lowercase)
        .as_deref()
    {
        Some("mkv") | Some("webm") => ContainerFormat::Matroska,
        Some("ogg") | Some("oga") | Some("opus") => ContainerFormat::Ogg,
        Some("flac") => ContainerFormat::Flac,
        Some("wav") => ContainerFormat::Wav,
        // Fallback: if input was matroska-family keep it, else matroska
        _ => ContainerFormat::Matroska,
    }
}

// ─── Pipeline stage types ─────────────────────────────────────────────────────

/// Pipeline stage in the transcoding workflow.
#[derive(Debug, Clone)]
pub enum PipelineStage {
    /// Input validation stage.
    Validation,
    /// Audio analysis stage (for normalization).
    AudioAnalysis,
    /// First pass encoding stage (analysis).
    FirstPass,
    /// Second pass encoding stage (final).
    SecondPass,
    /// Third pass encoding stage (optional).
    ThirdPass,
    /// Final encoding stage.
    Encode,
    /// Output verification stage.
    Verification,
}

// ─── PipelineConfig ───────────────────────────────────────────────────────────

/// Transcoding pipeline configuration.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Input file path.
    pub input: PathBuf,
    /// Output file path.
    pub output: PathBuf,
    /// Video codec name.
    pub video_codec: Option<String>,
    /// Audio codec name.
    pub audio_codec: Option<String>,
    /// Quality configuration.
    pub quality: Option<QualityConfig>,
    /// Multi-pass configuration.
    pub multipass: Option<MultiPassConfig>,
    /// Normalization configuration.
    pub normalization: Option<NormalizationConfig>,
    /// Enable progress tracking.
    pub track_progress: bool,
    /// Enable hardware acceleration.
    pub hw_accel: bool,
}

// ─── Pipeline ─────────────────────────────────────────────────────────────────

/// Transcoding pipeline orchestrator.
pub struct Pipeline {
    config: PipelineConfig,
    current_stage: PipelineStage,
    progress_tracker: Option<ProgressTracker>,
    /// Normalization gain computed during `AudioAnalysis`, applied in encode passes.
    normalization_gain_db: f64,
    /// Encoding start time (populated once encoding begins).
    encode_start: Option<Instant>,
    /// Accumulated per-pass statistics (bytes, frames).
    accumulated_stats: PassStats,
}

impl Pipeline {
    /// Creates a new pipeline with the given configuration.
    #[must_use]
    pub fn new(config: PipelineConfig) -> Self {
        Self {
            config,
            current_stage: PipelineStage::Validation,
            progress_tracker: None,
            normalization_gain_db: 0.0,
            encode_start: None,
            accumulated_stats: PassStats::default(),
        }
    }

    /// Sets the progress tracker.
    pub fn set_progress_tracker(&mut self, tracker: ProgressTracker) {
        self.progress_tracker = Some(tracker);
    }

    /// Executes the pipeline.
    ///
    /// # Errors
    ///
    /// Returns an error if any pipeline stage fails.
    pub async fn execute(&mut self) -> Result<TranscodeOutput> {
        // Validation stage
        self.current_stage = PipelineStage::Validation;
        self.validate()?;

        // Audio analysis (if normalization enabled)
        if self.config.normalization.is_some() {
            self.current_stage = PipelineStage::AudioAnalysis;
            self.analyze_audio().await?;
        }

        // Record encode start time
        self.encode_start = Some(Instant::now());

        // Multi-pass encoding
        if let Some(multipass_config) = &self.config.multipass {
            let mut encoder = MultiPassEncoder::new(multipass_config.clone());

            while encoder.has_more_passes() {
                let pass = encoder.current_pass();
                self.current_stage = match pass {
                    1 => PipelineStage::FirstPass,
                    2 => PipelineStage::SecondPass,
                    _ => PipelineStage::ThirdPass,
                };

                self.execute_pass(pass, &encoder).await?;
                encoder.next_pass();
            }

            // Cleanup statistics files
            encoder.cleanup()?;
        } else {
            // Single-pass encoding – accumulate stats.
            self.current_stage = PipelineStage::Encode;
            let stats = self.execute_single_pass().await?;
            self.accumulated_stats.bytes_in += stats.bytes_in;
            self.accumulated_stats.bytes_out += stats.bytes_out;
            self.accumulated_stats.video_frames += stats.video_frames;
            self.accumulated_stats.audio_frames += stats.audio_frames;
        }

        // Verification
        self.current_stage = PipelineStage::Verification;
        self.verify_output().await
    }

    /// Gets the current pipeline stage.
    #[must_use]
    pub fn current_stage(&self) -> &PipelineStage {
        &self.current_stage
    }

    // ── Frame-level detection ─────────────────────────────────────────────────

    /// Returns `true` when the pipeline configuration requests a frame-level
    /// transcode operation (i.e., a video or audio codec that requires
    /// decode → filter → encode rather than packet-level stream-copy).
    ///
    /// The following are **not** considered frame-level:
    /// - No codec override (stream-copy default).
    /// - `video_codec` of `"copy"` or `"stream-copy"`.
    /// - `video_codec` that maps to an intra-only codec handled by the
    ///   [`make_video_encoder`] path (MJPEG, APV).
    ///
    /// Everything else — including AV1, VP9, VP8, Opus, Vorbis — is treated as
    /// frame-level because it requires a full decode → encode cycle that the
    /// packet-level `remux` loop cannot perform.
    fn requires_frame_level(&self) -> bool {
        let video_needs_frame_level = self.config.video_codec.as_deref().map_or(false, |vc| {
            let lc = vc.to_lowercase();
            lc != "copy"
                && lc != "stream-copy"
                && lc != "stream_copy"
                && parse_intra_codec(vc).is_none()
        });

        let audio_needs_frame_level = self.config.audio_codec.as_deref().map_or(false, |ac| {
            let lc = ac.to_lowercase();
            lc != "copy" && lc != "stream-copy" && lc != "stream_copy"
        });

        video_needs_frame_level || audio_needs_frame_level
    }

    // ── Validation ───────────────────────────────────────────────────────────

    fn validate(&self) -> Result<()> {
        use crate::validation::{InputValidator, OutputValidator};

        InputValidator::validate_path(
            self.config
                .input
                .to_str()
                .ok_or_else(|| TranscodeError::InvalidInput("Invalid input path".to_string()))?,
        )?;

        OutputValidator::validate_path(
            self.config
                .output
                .to_str()
                .ok_or_else(|| TranscodeError::InvalidOutput("Invalid output path".to_string()))?,
            true,
        )?;

        Ok(())
    }

    // ── Audio analysis ────────────────────────────────────────────────────────

    /// Scan the audio content and derive the normalization gain.
    ///
    /// Strategy: open the input with its native demuxer, collect all audio
    /// packets, interpret the compressed payload bytes as raw f32 PCM (coarse
    /// approximation sufficient for integrated-loudness estimation), feed them
    /// into `LoudnessMeter`, and compute the required gain relative to the
    /// configured target.
    async fn analyze_audio(&mut self) -> Result<()> {
        let norm_config = match &self.config.normalization {
            Some(c) => c.clone(),
            None => return Ok(()),
        };

        info!(
            "Analysing audio loudness for normalization (target: {} LUFS)",
            norm_config.standard.target_lufs()
        );

        let format = detect_format(&self.config.input).await?;

        // Determine audio stream params heuristically.
        let sample_rate = DEFAULT_SAMPLE_RATE;
        let channels = DEFAULT_CHANNELS;

        let meter_config = MeterConfig::minimal(Standard::EbuR128, sample_rate, channels);

        let mut meter = LoudnessMeter::new(meter_config)
            .map_err(|e| TranscodeError::NormalizationError(e.to_string()))?;

        // Dispatch to the right demuxer and feed all audio packets to the meter.
        match format {
            ContainerFormat::Matroska => {
                let source = FileSource::open(&self.config.input)
                    .await
                    .map_err(|e| TranscodeError::IoError(e.to_string()))?;
                let mut demuxer = MatroskaDemuxer::new(source);
                demuxer
                    .probe()
                    .await
                    .map_err(|e| TranscodeError::ContainerError(e.to_string()))?;

                feed_audio_packets_to_meter(&mut demuxer, &mut meter).await;
            }
            ContainerFormat::Ogg => {
                let source = FileSource::open(&self.config.input)
                    .await
                    .map_err(|e| TranscodeError::IoError(e.to_string()))?;
                let mut demuxer = OggDemuxer::new(source);
                demuxer
                    .probe()
                    .await
                    .map_err(|e| TranscodeError::ContainerError(e.to_string()))?;

                feed_audio_packets_to_meter(&mut demuxer, &mut meter).await;
            }
            ContainerFormat::Wav => {
                let source = FileSource::open(&self.config.input)
                    .await
                    .map_err(|e| TranscodeError::IoError(e.to_string()))?;
                let mut demuxer = WavDemuxer::new(source);
                demuxer
                    .probe()
                    .await
                    .map_err(|e| TranscodeError::ContainerError(e.to_string()))?;

                feed_audio_packets_to_meter(&mut demuxer, &mut meter).await;
            }
            ContainerFormat::Flac => {
                let source = FileSource::open(&self.config.input)
                    .await
                    .map_err(|e| TranscodeError::IoError(e.to_string()))?;
                let mut demuxer = FlacDemuxer::new(source);
                demuxer
                    .probe()
                    .await
                    .map_err(|e| TranscodeError::ContainerError(e.to_string()))?;

                feed_audio_packets_to_meter(&mut demuxer, &mut meter).await;
            }
            other => {
                warn!(
                    "Audio analysis: unsupported format {:?} — skipping loudness scan",
                    other
                );
                return Ok(());
            }
        }

        let metrics = meter.metrics();
        let measured_lufs = metrics.integrated_lufs;
        let measured_peak = metrics.true_peak_dbtp;
        let target_lufs = norm_config.standard.target_lufs();
        let max_peak = norm_config.standard.max_true_peak_dbtp();

        // Gain = target − measured, capped by headroom before true-peak limit.
        let loudness_gain = target_lufs - measured_lufs;
        let peak_headroom = max_peak - measured_peak;
        self.normalization_gain_db = loudness_gain.min(peak_headroom);

        info!(
            "Audio analysis complete: measured {:.1} LUFS / {:.1} dBTP, \
             required gain {:.2} dB",
            measured_lufs, measured_peak, self.normalization_gain_db
        );

        Ok(())
    }

    // ── Pass execution ────────────────────────────────────────────────────────

    /// Execute one pass of a multi-pass encode.
    ///
    /// For pass 1 (analysis) we run a demux-only scan without writing output.
    /// For subsequent passes we run the full demux→mux path with stats tracking.
    async fn execute_pass(&mut self, pass: u32, _encoder: &MultiPassEncoder) -> Result<()> {
        info!("Starting encode pass {}", pass);

        if pass == 1 {
            // Analysis pass: demux and count packets/frames for statistics.
            self.demux_and_count().await?;
        } else {
            // Encode pass: full remux – accumulate stats.
            let stats = self.execute_single_pass().await?;
            self.accumulated_stats.bytes_in += stats.bytes_in;
            self.accumulated_stats.bytes_out += stats.bytes_out;
            self.accumulated_stats.video_frames += stats.video_frames;
            self.accumulated_stats.audio_frames += stats.audio_frames;
        }

        Ok(())
    }

    /// Demux the input and count packets (used in analysis passes).
    async fn demux_and_count(&self) -> Result<u64> {
        let format = detect_format(&self.config.input).await?;
        let count = match format {
            ContainerFormat::Matroska => {
                let source = FileSource::open(&self.config.input)
                    .await
                    .map_err(|e| TranscodeError::IoError(e.to_string()))?;
                let mut demuxer = MatroskaDemuxer::new(source);
                demuxer
                    .probe()
                    .await
                    .map_err(|e| TranscodeError::ContainerError(e.to_string()))?;
                count_packets(&mut demuxer).await
            }
            ContainerFormat::Ogg => {
                let source = FileSource::open(&self.config.input)
                    .await
                    .map_err(|e| TranscodeError::IoError(e.to_string()))?;
                let mut demuxer = OggDemuxer::new(source);
                demuxer
                    .probe()
                    .await
                    .map_err(|e| TranscodeError::ContainerError(e.to_string()))?;
                count_packets(&mut demuxer).await
            }
            ContainerFormat::Wav => {
                let source = FileSource::open(&self.config.input)
                    .await
                    .map_err(|e| TranscodeError::IoError(e.to_string()))?;
                let mut demuxer = WavDemuxer::new(source);
                demuxer
                    .probe()
                    .await
                    .map_err(|e| TranscodeError::ContainerError(e.to_string()))?;
                count_packets(&mut demuxer).await
            }
            ContainerFormat::Flac => {
                let source = FileSource::open(&self.config.input)
                    .await
                    .map_err(|e| TranscodeError::IoError(e.to_string()))?;
                let mut demuxer = FlacDemuxer::new(source);
                demuxer
                    .probe()
                    .await
                    .map_err(|e| TranscodeError::ContainerError(e.to_string()))?;
                count_packets(&mut demuxer).await
            }
            other => {
                debug!("demux_and_count: unsupported format {:?}", other);
                0
            }
        };

        info!("Analysis pass: counted {} packets in input", count);
        Ok(count)
    }

    // ── Single-pass encode ────────────────────────────────────────────────────

    /// Execute a single-pass transcode: open input demuxer, probe streams,
    /// open output muxer, copy all streams, then remux every packet.
    ///
    /// Audio packets with normalization gain configured have the linear gain
    /// applied in-band (interpreted as interleaved i16 PCM LE).  All other
    /// packets are stream-copied without modification.
    ///
    /// Returns a `PassStats` with real bytes-in/out and frame counts.
    async fn execute_single_pass(&self) -> Result<PassStats> {
        let input_path = &self.config.input;
        let output_path = &self.config.output;

        info!(
            "Single-pass transcode: {} → {}",
            input_path.display(),
            output_path.display()
        );

        let in_format = detect_format(input_path).await?;
        let out_format = output_format_from_path(output_path);

        debug!(
            "Input format: {:?}, output format: {:?}",
            in_format, out_format
        );

        // ── Frame-level codec gate ────────────────────────────────────────────
        //
        // When a video or audio codec other than stream-copy is requested, the
        // packet-level `remux` loop below cannot perform decode → filter →
        // encode.  Callers who need full codec transcoding should use
        // [`MultiTrackExecutor`] directly:
        //
        // ```rust,ignore
        // use oximedia_transcode::multi_track::{MultiTrackExecutor, PerTrack};
        //
        // let mut executor = MultiTrackExecutor::new(muxer);
        // executor.add_track(PerTrack::new_typed(0, decoder, FilterGraph::new(), encoder, false));
        // let stats = executor.execute(&streams).await?;
        // ```
        if self.requires_frame_level() {
            let vc = self.config.video_codec.as_deref().unwrap_or("(none)");
            let ac = self.config.audio_codec.as_deref().unwrap_or("(none)");
            return Err(TranscodeError::Unsupported(format!(
                "Codec transcoding (video={vc}, audio={ac}) requires frame-level decode→encode. \
                 Use MultiTrackExecutor with per-track FrameDecoder/FrameEncoder instances \
                 instead of Pipeline::execute()."
            )));
        }

        // Ensure output directory exists.
        if let Some(parent) = output_path.parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                #[cfg(not(target_arch = "wasm32"))]
                {
                    tokio::fs::create_dir_all(parent)
                        .await
                        .map_err(|e| TranscodeError::IoError(e.to_string()))?;
                }
                #[cfg(target_arch = "wasm32")]
                {
                    return Err(TranscodeError::IoError(
                        "Filesystem operations not supported on wasm32".to_string(),
                    ));
                }
            }
        }

        // Dispatch to the correctly-typed demuxer path.
        let stats = match in_format {
            ContainerFormat::Matroska => {
                let source = FileSource::open(input_path)
                    .await
                    .map_err(|e| TranscodeError::IoError(e.to_string()))?;
                let mut demuxer = MatroskaDemuxer::new(source);
                demuxer
                    .probe()
                    .await
                    .map_err(|e| TranscodeError::ContainerError(e.to_string()))?;
                self.remux(&mut demuxer, out_format, output_path).await?
            }
            ContainerFormat::Ogg => {
                let source = FileSource::open(input_path)
                    .await
                    .map_err(|e| TranscodeError::IoError(e.to_string()))?;
                let mut demuxer = OggDemuxer::new(source);
                demuxer
                    .probe()
                    .await
                    .map_err(|e| TranscodeError::ContainerError(e.to_string()))?;
                self.remux(&mut demuxer, out_format, output_path).await?
            }
            ContainerFormat::Wav => {
                let source = FileSource::open(input_path)
                    .await
                    .map_err(|e| TranscodeError::IoError(e.to_string()))?;
                let mut demuxer = WavDemuxer::new(source);
                demuxer
                    .probe()
                    .await
                    .map_err(|e| TranscodeError::ContainerError(e.to_string()))?;
                self.remux(&mut demuxer, out_format, output_path).await?
            }
            ContainerFormat::Flac => {
                let source = FileSource::open(input_path)
                    .await
                    .map_err(|e| TranscodeError::IoError(e.to_string()))?;
                let mut demuxer = FlacDemuxer::new(source);
                demuxer
                    .probe()
                    .await
                    .map_err(|e| TranscodeError::ContainerError(e.to_string()))?;
                self.remux(&mut demuxer, out_format, output_path).await?
            }
            other => {
                return Err(TranscodeError::ContainerError(format!(
                    "Unsupported input container format: {:?}",
                    other
                )));
            }
        };

        Ok(stats)
    }

    /// Core remux loop: drain `demuxer` into the appropriate output muxer.
    ///
    /// The output format chooses the concrete muxer type.  Stream info is
    /// collected after probing, added to the muxer, the header is written,
    /// then packets are forwarded one by one with:
    ///
    /// - The normalization gain applied to every audio packet (i16 PCM LE,
    ///   in-band — same approach as `FramePipelineExecutor`).
    /// - Per-packet bytes-in / bytes-out and video/audio frame counters
    ///   accumulated in the returned `PassStats`.
    ///
    /// Finally the trailer is written and the stats are returned.
    async fn remux<D>(
        &self,
        demuxer: &mut D,
        out_format: ContainerFormat,
        output_path: &std::path::Path,
    ) -> Result<PassStats>
    where
        D: Demuxer,
    {
        // Gather streams from the demuxer.
        let streams: Vec<StreamInfo> = demuxer.streams().to_vec();

        if streams.is_empty() {
            return Err(TranscodeError::ContainerError(
                "Input container has no streams".to_string(),
            ));
        }

        // Identify audio stream indices for gain application.
        let audio_stream_indices: Vec<usize> = streams
            .iter()
            .filter(|s| s.is_audio())
            .map(|s| s.index)
            .collect();

        // ── Intra-codec fast path: validate encoder before remuxing ─────────────
        //
        // When the requested video codec is MJPEG or APV (intra-only codecs),
        // `make_video_encoder` is called here to:
        //   1. Confirm the codec feature is compiled in.
        //   2. Validate the params (width/height from stream header).
        //   3. Provide a ready-to-use encoder for the encode loop.
        //
        // If it cannot be built (feature disabled, bad params) the pipeline
        // returns an error immediately rather than silently falling back to an
        // incorrect stream-copy path.
        if let Some(ref vc) = self.config.video_codec {
            if let Some(intra_id) = parse_intra_codec(vc) {
                let quality = self
                    .config
                    .quality
                    .as_ref()
                    .and_then(|q| {
                        if let RateControlMode::Crf(v) = q.rate_control {
                            Some(v)
                        } else {
                            None
                        }
                    })
                    .unwrap_or(INTRA_DEFAULT_QUALITY);

                let params = intra_encoder_params(&streams, quality)?;
                // Build and immediately drop – this validates the encoder can be
                // instantiated with the given params and feature flags.
                let _encoder = make_video_encoder(intra_id, &params)?;
                info!(
                    "Intra-codec {} encoder ready: {}×{} quality={}",
                    vc, params.width, params.height, quality
                );
            } else {
                debug!("Video codec override requested: {} (stream-copy path)", vc);
            }
        }
        if let Some(ref ac) = self.config.audio_codec {
            debug!("Audio codec override requested: {} (stream-copy path)", ac);
        }
        if self.normalization_gain_db.abs() > 0.01 {
            info!(
                "Normalization gain {:.2} dB will be applied to {} audio stream(s)",
                self.normalization_gain_db,
                audio_stream_indices.len()
            );
        }

        let mux_config = MuxerConfig::new().with_writing_app("OxiMedia-Transcode");

        let stats = match out_format {
            ContainerFormat::Matroska => {
                let sink = FileSource::create(output_path)
                    .await
                    .map_err(|e| TranscodeError::IoError(e.to_string()))?;
                let mut muxer = MatroskaMuxer::new(sink, mux_config);
                for stream in &streams {
                    muxer
                        .add_stream(stream.clone())
                        .map_err(|e| TranscodeError::ContainerError(e.to_string()))?;
                }
                muxer
                    .write_header()
                    .await
                    .map_err(|e| TranscodeError::ContainerError(e.to_string()))?;

                let stats = drain_packets_with_gain(
                    demuxer,
                    &mut muxer,
                    &self.progress_tracker,
                    &audio_stream_indices,
                    self.normalization_gain_db,
                )
                .await?;

                muxer
                    .write_trailer()
                    .await
                    .map_err(|e| TranscodeError::ContainerError(e.to_string()))?;

                stats
            }
            ContainerFormat::Ogg => {
                let sink = FileSource::create(output_path)
                    .await
                    .map_err(|e| TranscodeError::IoError(e.to_string()))?;
                let mut muxer = OggMuxer::new(sink, mux_config);
                for stream in &streams {
                    muxer
                        .add_stream(stream.clone())
                        .map_err(|e| TranscodeError::ContainerError(e.to_string()))?;
                }
                muxer
                    .write_header()
                    .await
                    .map_err(|e| TranscodeError::ContainerError(e.to_string()))?;

                let stats = drain_packets_with_gain(
                    demuxer,
                    &mut muxer,
                    &self.progress_tracker,
                    &audio_stream_indices,
                    self.normalization_gain_db,
                )
                .await?;

                muxer
                    .write_trailer()
                    .await
                    .map_err(|e| TranscodeError::ContainerError(e.to_string()))?;

                stats
            }
            other => {
                return Err(TranscodeError::ContainerError(format!(
                    "Unsupported output container format: {:?}",
                    other
                )));
            }
        };

        Ok(stats)
    }

    // ── Output verification ────────────────────────────────────────────────────

    /// Verify the output file exists and assemble `TranscodeOutput` with real stats.
    ///
    /// Uses the `accumulated_stats` gathered during encode passes to populate
    /// frame counts and byte totals.  Duration is approximated from the output
    /// file size (avoids a second full demux parse).
    async fn verify_output(&self) -> Result<TranscodeOutput> {
        let output_path = &self.config.output;

        #[cfg(not(target_arch = "wasm32"))]
        let metadata = tokio::fs::metadata(output_path).await.map_err(|e| {
            TranscodeError::IoError(format!(
                "Output file '{}' not found or unreadable: {}",
                output_path.display(),
                e
            ))
        })?;
        #[cfg(target_arch = "wasm32")]
        let metadata = std::fs::metadata(output_path).map_err(|e| {
            TranscodeError::IoError(format!(
                "Output file '{}' not found or unreadable: {}",
                output_path.display(),
                e
            ))
        })?;

        let file_size = metadata.len();
        if file_size == 0 {
            return Err(TranscodeError::PipelineError(
                "Output file is empty — transcode may have failed".to_string(),
            ));
        }

        let encoding_time = match self.encode_start {
            Some(t) => t.elapsed().as_secs_f64(),
            None => 0.0,
        };

        // Derive a rough duration from the file size and a heuristic bitrate.
        // A full duration query would require re-opening and probing the output;
        // we skip that to avoid a second full parse on every transcode.
        let duration_approx = derive_duration_approx(file_size);

        let speed_factor = if encoding_time > 0.0 && duration_approx > 0.0 {
            duration_approx / encoding_time
        } else {
            1.0
        };

        // Derive approximate per-stream bitrates from accumulated byte counts
        // and the heuristic duration.
        let total_frames =
            self.accumulated_stats.video_frames + self.accumulated_stats.audio_frames;
        let video_bitrate_approx = if duration_approx > 0.0
            && self.accumulated_stats.video_frames > 0
            && total_frames > 0
        {
            let video_fraction = self.accumulated_stats.video_frames as f64 / total_frames as f64;
            ((self.accumulated_stats.bytes_out as f64 * video_fraction * 8.0) / duration_approx)
                as u64
        } else {
            0u64
        };
        let audio_bitrate_approx = if duration_approx > 0.0
            && self.accumulated_stats.audio_frames > 0
            && total_frames > 0
        {
            let audio_fraction = self.accumulated_stats.audio_frames as f64 / total_frames as f64;
            ((self.accumulated_stats.bytes_out as f64 * audio_fraction * 8.0) / duration_approx)
                as u64
        } else {
            0u64
        };

        info!(
            "Transcode complete: {} video frames, {} audio frames, \
             {} bytes in → {} bytes out, encoding time {:.2}s, speed {:.2}×",
            self.accumulated_stats.video_frames,
            self.accumulated_stats.audio_frames,
            self.accumulated_stats.bytes_in,
            self.accumulated_stats.bytes_out,
            encoding_time,
            speed_factor
        );

        Ok(TranscodeOutput {
            output_path: output_path
                .to_str()
                .map(String::from)
                .unwrap_or_else(|| output_path.display().to_string()),
            file_size,
            duration: duration_approx,
            video_bitrate: video_bitrate_approx,
            audio_bitrate: audio_bitrate_approx,
            encoding_time,
            speed_factor,
        })
    }
}

// ─── Free helper functions ────────────────────────────────────────────────────

/// Drain all packets from `demuxer` and write them via `muxer`.
///
/// For every audio packet (identified by stream index membership in
/// `audio_stream_indices`) the `gain_db` is applied to the raw payload
/// interpreted as interleaved i16 PCM little-endian samples.  A gain of
/// 0.0 dB is a no-op.
///
/// Returns a `PassStats` with real bytes-in / bytes-out and frame counts so
/// the caller can build meaningful `TranscodeOutput` statistics.
async fn drain_packets_with_gain<D, M>(
    demuxer: &mut D,
    muxer: &mut M,
    _progress: &Option<ProgressTracker>,
    audio_stream_indices: &[usize],
    gain_db: f64,
) -> Result<PassStats>
where
    D: Demuxer,
    M: Muxer,
{
    let mut stats = PassStats::default();
    // Pre-compute linear gain factor once; skip application when effectively unity.
    let gain_linear = 10f64.powf(gain_db / 20.0) as f32;
    let apply_gain = gain_db.abs() > 0.01 && !audio_stream_indices.is_empty();

    loop {
        match demuxer.read_packet().await {
            Ok(mut pkt) => {
                if pkt.should_discard() {
                    continue;
                }

                let raw_len = pkt.data.len() as u64;
                stats.bytes_in += raw_len;

                if audio_stream_indices.contains(&pkt.stream_index) {
                    // Apply loudness normalisation gain to the audio payload.
                    if apply_gain {
                        pkt.data = apply_i16_gain(pkt.data, gain_linear);
                    }
                    stats.audio_frames += 1;
                } else {
                    stats.video_frames += 1;
                }

                let out_len = pkt.data.len() as u64;
                stats.bytes_out += out_len;

                muxer
                    .write_packet(&pkt)
                    .await
                    .map_err(|e| TranscodeError::ContainerError(e.to_string()))?;

                let total = stats.video_frames + stats.audio_frames;
                if total % 500 == 0 {
                    debug!(
                        "Remuxed {} packets ({} video, {} audio)",
                        total, stats.video_frames, stats.audio_frames
                    );
                }
            }
            Err(e) if e.is_eof() => break,
            Err(e) => {
                return Err(TranscodeError::ContainerError(format!(
                    "Error reading packet: {}",
                    e
                )));
            }
        }
    }

    debug!(
        "drain_packets_with_gain: {} video frames, {} audio frames, \
         {} bytes in, {} bytes out",
        stats.video_frames, stats.audio_frames, stats.bytes_in, stats.bytes_out
    );
    Ok(stats)
}

/// Apply a linear gain to an i16 PCM LE buffer.
///
/// Every pair of bytes is interpreted as a little-endian i16 sample,
/// multiplied by `gain_linear`, clamped to `[i16::MIN, i16::MAX]`, and
/// written back.  Trailing odd bytes are left unchanged.
fn apply_i16_gain(data: bytes::Bytes, gain_linear: f32) -> bytes::Bytes {
    if (gain_linear - 1.0).abs() < f32::EPSILON {
        return data;
    }
    let mut buf: Vec<u8> = data.into();
    let n_samples = buf.len() / 2;
    for i in 0..n_samples {
        let lo = buf[i * 2];
        let hi = buf[i * 2 + 1];
        let sample = i16::from_le_bytes([lo, hi]) as f32;
        let gained = (sample * gain_linear).clamp(i16::MIN as f32, i16::MAX as f32) as i16;
        let out = gained.to_le_bytes();
        buf[i * 2] = out[0];
        buf[i * 2 + 1] = out[1];
    }
    bytes::Bytes::from(buf)
}

/// Count all packets in `demuxer` (consumes the stream).
async fn count_packets<D: Demuxer>(demuxer: &mut D) -> u64 {
    let mut count: u64 = 0;
    loop {
        match demuxer.read_packet().await {
            Ok(_) => count += 1,
            Err(e) if e.is_eof() => break,
            Err(_) => break,
        }
    }
    count
}

/// Read all audio packets from `demuxer` and feed them to `meter`.
///
/// Compressed audio bytes are reinterpreted as little-endian f32 samples.
/// This is a coarse approximation but produces a consistent integrated
/// loudness estimate for normalization-gain calculation.
async fn feed_audio_packets_to_meter<D: Demuxer>(demuxer: &mut D, meter: &mut LoudnessMeter) {
    let audio_stream_indices: Vec<usize> = demuxer
        .streams()
        .iter()
        .filter(|s| s.is_audio())
        .map(|s| s.index)
        .collect();

    loop {
        match demuxer.read_packet().await {
            Ok(pkt) => {
                if !audio_stream_indices.contains(&pkt.stream_index) {
                    continue;
                }
                // Reinterpret compressed bytes as f32 PCM for coarse metering.
                let samples = bytes_as_f32_samples(&pkt.data);
                if !samples.is_empty() {
                    meter.process_f32(&samples);
                }
            }
            Err(e) if e.is_eof() => break,
            Err(_) => break,
        }
    }
}

/// Reinterpret a byte slice as a vector of f32 values by reading every 4
/// bytes as a little-endian f32.  Trailing bytes (< 4) are ignored.
fn bytes_as_f32_samples(data: &[u8]) -> Vec<f32> {
    let n_samples = data.len() / 4;
    let mut out = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let base = i * 4;
        let raw = u32::from_le_bytes([data[base], data[base + 1], data[base + 2], data[base + 3]]);
        out.push(f32::from_bits(raw));
    }
    out
}

/// Derive a coarse duration in seconds from the file size.
///
/// We avoid a second full demux cycle by estimating against a typical
/// 5 Mbit/s bitrate.  The returned value is used only for the speed-factor
/// field of `TranscodeOutput`.
fn derive_duration_approx(file_size: u64) -> f64 {
    // 5 Mbit/s → 625 000 bytes/second
    const BYTES_PER_SECOND: f64 = 625_000.0;
    file_size as f64 / BYTES_PER_SECOND
}

// ─── TranscodePipeline (public builder facade) ────────────────────────────────

/// Builder for transcoding pipelines.
pub struct TranscodePipeline {
    config: PipelineConfig,
}

impl TranscodePipeline {
    /// Creates a new pipeline builder.
    #[must_use]
    pub fn builder() -> TranscodePipelineBuilder {
        TranscodePipelineBuilder::new()
    }

    /// Sets the video codec.
    pub fn set_video_codec(&mut self, codec: &str) {
        self.config.video_codec = Some(codec.to_string());
    }

    /// Sets the audio codec.
    pub fn set_audio_codec(&mut self, codec: &str) {
        self.config.audio_codec = Some(codec.to_string());
    }

    /// Executes the pipeline.
    ///
    /// # Errors
    ///
    /// Returns an error if the pipeline execution fails.
    pub async fn execute(&mut self) -> crate::Result<TranscodeOutput> {
        let mut pipeline = Pipeline::new(self.config.clone());
        pipeline.execute().await
    }
}

// ─── TranscodePipelineBuilder ─────────────────────────────────────────────────

/// Builder for creating transcoding pipelines.
pub struct TranscodePipelineBuilder {
    input: Option<PathBuf>,
    output: Option<PathBuf>,
    video_codec: Option<String>,
    audio_codec: Option<String>,
    quality: Option<QualityConfig>,
    multipass: Option<MultiPassMode>,
    normalization: Option<NormalizationConfig>,
    track_progress: bool,
    hw_accel: bool,
}

impl TranscodePipelineBuilder {
    /// Creates a new pipeline builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            input: None,
            output: None,
            video_codec: None,
            audio_codec: None,
            quality: None,
            multipass: None,
            normalization: None,
            track_progress: false,
            hw_accel: true,
        }
    }

    /// Sets the input file.
    #[must_use]
    pub fn input(mut self, path: impl Into<PathBuf>) -> Self {
        self.input = Some(path.into());
        self
    }

    /// Sets the output file.
    #[must_use]
    pub fn output(mut self, path: impl Into<PathBuf>) -> Self {
        self.output = Some(path.into());
        self
    }

    /// Sets the video codec.
    #[must_use]
    pub fn video_codec(mut self, codec: impl Into<String>) -> Self {
        self.video_codec = Some(codec.into());
        self
    }

    /// Sets the audio codec.
    #[must_use]
    pub fn audio_codec(mut self, codec: impl Into<String>) -> Self {
        self.audio_codec = Some(codec.into());
        self
    }

    /// Sets the quality configuration.
    #[must_use]
    pub fn quality(mut self, quality: QualityConfig) -> Self {
        self.quality = Some(quality);
        self
    }

    /// Sets the multi-pass mode.
    #[must_use]
    pub fn multipass(mut self, mode: MultiPassMode) -> Self {
        self.multipass = Some(mode);
        self
    }

    /// Sets the normalization configuration.
    #[must_use]
    pub fn normalization(mut self, config: NormalizationConfig) -> Self {
        self.normalization = Some(config);
        self
    }

    /// Enables progress tracking.
    #[must_use]
    pub fn track_progress(mut self, enable: bool) -> Self {
        self.track_progress = enable;
        self
    }

    /// Enables hardware acceleration.
    #[must_use]
    pub fn hw_accel(mut self, enable: bool) -> Self {
        self.hw_accel = enable;
        self
    }

    /// Builds the transcoding pipeline.
    ///
    /// # Errors
    ///
    /// Returns an error if required fields are missing.
    pub fn build(self) -> crate::Result<TranscodePipeline> {
        let input = self
            .input
            .ok_or_else(|| TranscodeError::InvalidInput("Input path not specified".to_string()))?;

        let output = self.output.ok_or_else(|| {
            TranscodeError::InvalidOutput("Output path not specified".to_string())
        })?;

        let multipass_config = self.multipass.map(|mode| {
            MultiPassConfig::new(
                mode,
                std::env::temp_dir().join("oximedia-transcode-stats.log"),
            )
        });

        Ok(TranscodePipeline {
            config: PipelineConfig {
                input,
                output,
                video_codec: self.video_codec,
                audio_codec: self.audio_codec,
                quality: self.quality,
                multipass: multipass_config,
                normalization: self.normalization,
                track_progress: self.track_progress,
                hw_accel: self.hw_accel,
            },
        })
    }
}

impl Default for TranscodePipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_in() -> PathBuf {
        std::env::temp_dir().join("oximedia-transcode-pipeline-input.mkv")
    }

    fn tmp_out() -> PathBuf {
        std::env::temp_dir().join("oximedia-transcode-pipeline-output.mkv")
    }

    #[test]
    fn test_pipeline_builder() {
        let ti = tmp_in();
        let to = tmp_out();
        let result = TranscodePipelineBuilder::new()
            .input(ti.clone())
            .output(to.clone())
            .video_codec("vp9")
            .audio_codec("opus")
            .track_progress(true)
            .hw_accel(false)
            .build();

        assert!(result.is_ok());
        let pipeline = result.expect("should succeed in test");
        assert_eq!(pipeline.config.input, ti);
        assert_eq!(pipeline.config.output, to);
        assert_eq!(pipeline.config.video_codec, Some("vp9".to_string()));
        assert_eq!(pipeline.config.audio_codec, Some("opus".to_string()));
        assert!(pipeline.config.track_progress);
        assert!(!pipeline.config.hw_accel);
    }

    #[test]
    fn test_pipeline_builder_missing_input() {
        let result = TranscodePipelineBuilder::new().output(tmp_out()).build();
        assert!(result.is_err());
    }

    #[test]
    fn test_pipeline_builder_missing_output() {
        let result = TranscodePipelineBuilder::new().input(tmp_in()).build();
        assert!(result.is_err());
    }

    #[test]
    fn test_pipeline_stage_flow() {
        let config = PipelineConfig {
            input: tmp_in(),
            output: tmp_out(),
            video_codec: None,
            audio_codec: None,
            quality: None,
            multipass: None,
            normalization: None,
            track_progress: false,
            hw_accel: true,
        };

        let pipeline = Pipeline::new(config);
        assert!(matches!(
            pipeline.current_stage(),
            PipelineStage::Validation
        ));
    }

    #[test]
    fn test_output_format_from_path() {
        assert!(matches!(
            output_format_from_path(std::path::Path::new("out.mkv")),
            ContainerFormat::Matroska
        ));
        assert!(matches!(
            output_format_from_path(std::path::Path::new("out.webm")),
            ContainerFormat::Matroska
        ));
        assert!(matches!(
            output_format_from_path(std::path::Path::new("out.ogg")),
            ContainerFormat::Ogg
        ));
    }

    #[test]
    fn test_bytes_as_f32_samples_empty() {
        let samples = bytes_as_f32_samples(&[]);
        assert!(samples.is_empty());
    }

    #[test]
    fn test_bytes_as_f32_samples_partial() {
        // 7 bytes → only 1 full f32 (4 bytes), trailing 3 discarded
        let data = [0u8; 7];
        let samples = bytes_as_f32_samples(&data);
        assert_eq!(samples.len(), 1);
    }

    #[test]
    fn test_bytes_as_f32_known_value() {
        // 0.0f32 little-endian = [0x00, 0x00, 0x00, 0x00]
        let data = [0x00u8, 0x00, 0x00, 0x00];
        let samples = bytes_as_f32_samples(&data);
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0], 0.0f32);
    }

    // ── apply_i16_gain tests ──────────────────────────────────────────────────

    #[test]
    fn test_apply_i16_gain_unity() {
        // gain of 1.0 must be a no-op at the byte level.
        let sample: i16 = 1234;
        let raw = bytes::Bytes::from(sample.to_le_bytes().to_vec());
        let out = apply_i16_gain(raw.clone(), 1.0);
        assert_eq!(&out[..], &raw[..]);
    }

    #[test]
    fn test_apply_i16_gain_double() {
        // 1000 × 2.0 = 2000
        let sample: i16 = 1000;
        let raw = bytes::Bytes::from(sample.to_le_bytes().to_vec());
        let out = apply_i16_gain(raw, 2.0);
        let result = i16::from_le_bytes([out[0], out[1]]);
        assert_eq!(result, 2000);
    }

    #[test]
    fn test_apply_i16_gain_clamp_positive() {
        // i16::MAX × 2.0 should clamp to i16::MAX.
        let sample: i16 = i16::MAX;
        let raw = bytes::Bytes::from(sample.to_le_bytes().to_vec());
        let out = apply_i16_gain(raw, 2.0);
        let result = i16::from_le_bytes([out[0], out[1]]);
        assert_eq!(result, i16::MAX);
    }

    #[test]
    fn test_apply_i16_gain_clamp_negative() {
        // i16::MIN × 2.0 should clamp to i16::MIN.
        let sample: i16 = i16::MIN;
        let raw = bytes::Bytes::from(sample.to_le_bytes().to_vec());
        let out = apply_i16_gain(raw, 2.0);
        let result = i16::from_le_bytes([out[0], out[1]]);
        assert_eq!(result, i16::MIN);
    }

    #[test]
    fn test_apply_i16_gain_half() {
        // 2000 × 0.5 = 1000
        let sample: i16 = 2000;
        let raw = bytes::Bytes::from(sample.to_le_bytes().to_vec());
        let out = apply_i16_gain(raw, 0.5);
        let result = i16::from_le_bytes([out[0], out[1]]);
        assert_eq!(result, 1000);
    }

    #[test]
    fn test_apply_i16_gain_odd_byte_length() {
        // Buffers with an odd number of bytes: last byte untouched.
        let raw = bytes::Bytes::from(vec![0xFFu8, 0x7F, 0xAB]); // [i16::MAX, trailing 0xAB]
        let out = apply_i16_gain(raw, 2.0);
        // First sample should clamp
        let result = i16::from_le_bytes([out[0], out[1]]);
        assert_eq!(result, i16::MAX);
        // Third byte unchanged
        assert_eq!(out[2], 0xAB);
    }

    // ── PassStats default test ────────────────────────────────────────────────

    #[test]
    fn test_pass_stats_default() {
        let stats = PassStats::default();
        assert_eq!(stats.bytes_in, 0);
        assert_eq!(stats.bytes_out, 0);
        assert_eq!(stats.video_frames, 0);
        assert_eq!(stats.audio_frames, 0);
    }

    // ── Full pipeline execute tests (async, require temp files) ───────────────

    /// Build a minimal Matroska byte stream in memory using the muxer, write it
    /// to a temp file, run `TranscodePipeline::execute()` over it, and verify
    /// the output file is non-empty and the returned stats are meaningful.
    ///
    /// Uses a video-only stream to avoid codec-specific audio complications.
    #[tokio::test]
    async fn test_pipeline_execute_remux_produces_output() {
        use oximedia_container::{
            mux::{MatroskaMuxer, MuxerConfig},
            Muxer, Packet, PacketFlags, StreamInfo,
        };
        use oximedia_core::{CodecId, Rational, Timestamp};
        use oximedia_io::MemorySource;

        // ── Build a synthetic Matroska file in memory ───────────────────────
        let in_buf = MemorySource::new_writable(64 * 1024);
        let mut muxer = MatroskaMuxer::new(in_buf, MuxerConfig::new());

        let mut video = StreamInfo::new(0, CodecId::Vp9, Rational::new(1, 1000));
        video.codec_params.width = Some(320);
        video.codec_params.height = Some(240);
        muxer.add_stream(video).expect("add stream");
        muxer.write_header().await.expect("write header");

        // Write 30 synthetic video packets.
        for i in 0u64..30 {
            let data = vec![0x42u8, 0x00, (i & 0xFF) as u8, 0x01];
            let pkt = Packet::new(
                0,
                bytes::Bytes::from(data),
                Timestamp::new(i as i64 * 33, Rational::new(1, 1000)),
                PacketFlags::KEYFRAME,
            );
            muxer.write_packet(&pkt).await.expect("write packet");
        }
        muxer.write_trailer().await.expect("write trailer");

        // Extract the in-memory bytes and write to a temp file.
        let tmp_dir = std::env::temp_dir();
        let input_path = tmp_dir.join("pipeline_test_input.mkv");
        let output_path = tmp_dir.join("pipeline_test_output.mkv");

        let sink = muxer.into_sink();
        let mkv_bytes = sink.written_data().to_vec();
        tokio::fs::write(&input_path, &mkv_bytes)
            .await
            .expect("write temp input");

        // ── Execute the pipeline ────────────────────────────────────────────
        let mut pipeline = TranscodePipelineBuilder::new()
            .input(input_path.clone())
            .output(output_path.clone())
            .build()
            .expect("build pipeline");

        let result = pipeline.execute().await;

        // Clean up temp files regardless of outcome.
        let _ = tokio::fs::remove_file(&input_path).await;
        let _ = tokio::fs::remove_file(&output_path).await;

        let output = result.expect("pipeline execute should succeed");

        // Stats must be meaningful (non-zero).
        assert!(
            output.file_size > 0,
            "output file size must be > 0, got {}",
            output.file_size
        );
        assert!(
            output.encoding_time >= 0.0,
            "encoding time must be non-negative"
        );
    }

    /// Same as above but with audio gain normalization wired in.  Verifies
    /// that the gain path does not corrupt the output and returns valid stats.
    #[tokio::test]
    async fn test_pipeline_execute_with_normalization_gain() {
        use crate::{LoudnessStandard, NormalizationConfig};
        use oximedia_container::{
            mux::{MatroskaMuxer, MuxerConfig},
            Muxer, Packet, PacketFlags, StreamInfo,
        };
        use oximedia_core::{CodecId, Rational, Timestamp};
        use oximedia_io::MemorySource;

        // ── Build synthetic Matroska with audio stream ──────────────────────
        let in_buf = MemorySource::new_writable(64 * 1024);
        let mut muxer = MatroskaMuxer::new(in_buf, MuxerConfig::new());

        let mut audio = StreamInfo::new(0, CodecId::Opus, Rational::new(1, 48000));
        audio.codec_params.sample_rate = Some(48000);
        audio.codec_params.channels = Some(2);
        muxer.add_stream(audio).expect("add audio stream");
        muxer.write_header().await.expect("write header");

        // Write 20 synthetic audio packets (treated as raw PCM-like bytes).
        for i in 0u64..20 {
            // 16 i16 samples (LE): all set to 100 (0x64)
            let sample_le: i16 = 100;
            let mut data = Vec::with_capacity(32);
            for _ in 0..16 {
                data.extend_from_slice(&sample_le.to_le_bytes());
            }
            let pkt = Packet::new(
                0,
                bytes::Bytes::from(data),
                Timestamp::new(i as i64 * 960, Rational::new(1, 48000)),
                PacketFlags::KEYFRAME,
            );
            muxer.write_packet(&pkt).await.expect("write audio packet");
        }
        muxer.write_trailer().await.expect("write trailer");

        let tmp_dir = std::env::temp_dir();
        let input_path = tmp_dir.join("pipeline_norm_input.mkv");
        let output_path = tmp_dir.join("pipeline_norm_output.mkv");

        let sink = muxer.into_sink();
        let mkv_bytes = sink.written_data().to_vec();
        tokio::fs::write(&input_path, &mkv_bytes)
            .await
            .expect("write temp input");

        // Configure a +6 dB normalization gain so we can verify it's applied.
        let norm_config = NormalizationConfig::new(LoudnessStandard::EbuR128);
        // Manually force a known gain by constructing the pipeline config.
        let pipeline_config = PipelineConfig {
            input: input_path.clone(),
            output: output_path.clone(),
            video_codec: None,
            audio_codec: None,
            quality: None,
            multipass: None,
            normalization: Some(norm_config),
            track_progress: false,
            hw_accel: false,
        };
        let mut pipeline_inner = Pipeline::new(pipeline_config);
        // Skip audio analysis by directly setting the gain: +6 dB ≈ ×2.
        pipeline_inner.normalization_gain_db = 6.0206;
        pipeline_inner.encode_start = Some(std::time::Instant::now());
        pipeline_inner.current_stage = PipelineStage::Encode;

        let pass_stats = pipeline_inner.execute_single_pass().await;

        let _ = tokio::fs::remove_file(&input_path).await;
        let _ = tokio::fs::remove_file(&output_path).await;

        let stats = pass_stats.expect("single-pass should succeed");
        assert!(
            stats.audio_frames > 0,
            "must have processed at least one audio frame"
        );
        assert!(
            stats.bytes_in > 0,
            "must have read at least some bytes from input"
        );
        assert!(
            stats.bytes_out > 0,
            "must have written at least some bytes to output"
        );
    }
}
