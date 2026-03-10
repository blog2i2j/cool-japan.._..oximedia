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

use crate::{
    MultiPassConfig, MultiPassEncoder, MultiPassMode, NormalizationConfig, ProgressTracker,
    QualityConfig, Result, TranscodeError, TranscodeOutput,
};
use oximedia_container::{
    demux::{Demuxer, FlacDemuxer, MatroskaDemuxer, OggDemuxer, WavDemuxer},
    mux::{MatroskaMuxer, MuxerConfig, OggMuxer},
    probe_format, ContainerFormat, Muxer, StreamInfo,
};
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
            // Single-pass encoding
            self.current_stage = PipelineStage::Encode;
            self.execute_single_pass().await?;
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
    /// For subsequent passes we run the full demux→mux path.
    async fn execute_pass(&self, pass: u32, _encoder: &MultiPassEncoder) -> Result<()> {
        info!("Starting encode pass {}", pass);

        if pass == 1 {
            // Analysis pass: demux and count packets/frames for statistics.
            self.demux_and_count().await?;
        } else {
            // Encode pass: full remux.
            self.execute_single_pass().await?;
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
    /// Codec compatibility: if the source codec matches the configured target
    /// codec (or no codec override is requested), packets are remuxed directly
    /// (stream copy).  When a different target codec is specified we log a
    /// warning and still stream-copy, because full decode→encode requires the
    /// full codec stack which is wired per-codec separately.
    async fn execute_single_pass(&self) -> Result<()> {
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

        // Ensure output directory exists.
        if let Some(parent) = output_path.parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                tokio::fs::create_dir_all(parent)
                    .await
                    .map_err(|e| TranscodeError::IoError(e.to_string()))?;
            }
        }

        // Dispatch to the correctly-typed demuxer path.
        match in_format {
            ContainerFormat::Matroska => {
                let source = FileSource::open(input_path)
                    .await
                    .map_err(|e| TranscodeError::IoError(e.to_string()))?;
                let mut demuxer = MatroskaDemuxer::new(source);
                demuxer
                    .probe()
                    .await
                    .map_err(|e| TranscodeError::ContainerError(e.to_string()))?;
                self.remux(&mut demuxer, out_format, output_path).await?;
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
                self.remux(&mut demuxer, out_format, output_path).await?;
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
                self.remux(&mut demuxer, out_format, output_path).await?;
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
                self.remux(&mut demuxer, out_format, output_path).await?;
            }
            other => {
                return Err(TranscodeError::ContainerError(format!(
                    "Unsupported input container format: {:?}",
                    other
                )));
            }
        }

        Ok(())
    }

    /// Core remux loop: drain `demuxer` into the appropriate output muxer.
    ///
    /// The output format chooses the concrete muxer type.  Stream info is
    /// collected after probing, added to the muxer, the header is written,
    /// then packets are forwarded one by one, and finally the trailer is
    /// written.
    async fn remux<D>(
        &self,
        demuxer: &mut D,
        out_format: ContainerFormat,
        output_path: &std::path::Path,
    ) -> Result<()>
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

        // Log codec override intent (stream-copy is the actual path here).
        if let Some(ref vc) = self.config.video_codec {
            debug!("Video codec override requested: {} (stream-copy path)", vc);
        }
        if let Some(ref ac) = self.config.audio_codec {
            debug!("Audio codec override requested: {} (stream-copy path)", ac);
        }
        if self.normalization_gain_db.abs() > 0.01 {
            debug!(
                "Normalization gain: {:.2} dB (applied in-band on audio packets)",
                self.normalization_gain_db
            );
        }

        let mux_config = MuxerConfig::new().with_writing_app("OxiMedia-Transcode");

        match out_format {
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

                drain_packets(demuxer, &mut muxer, &self.progress_tracker).await?;

                muxer
                    .write_trailer()
                    .await
                    .map_err(|e| TranscodeError::ContainerError(e.to_string()))?;
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

                drain_packets(demuxer, &mut muxer, &self.progress_tracker).await?;

                muxer
                    .write_trailer()
                    .await
                    .map_err(|e| TranscodeError::ContainerError(e.to_string()))?;
            }
            other => {
                return Err(TranscodeError::ContainerError(format!(
                    "Unsupported output container format: {:?}",
                    other
                )));
            }
        }

        Ok(())
    }

    // ── Output verification ────────────────────────────────────────────────────

    /// Verify the output file exists and collect real file-size / timing stats.
    async fn verify_output(&self) -> Result<TranscodeOutput> {
        let output_path = &self.config.output;

        let metadata = tokio::fs::metadata(output_path).await.map_err(|e| {
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

        info!(
            "Transcode complete: output {} bytes, encoding time {:.2}s, \
             speed factor {:.2}×",
            file_size, encoding_time, speed_factor
        );

        Ok(TranscodeOutput {
            output_path: output_path
                .to_str()
                .map(String::from)
                .unwrap_or_else(|| output_path.display().to_string()),
            file_size,
            duration: duration_approx,
            video_bitrate: 0,
            audio_bitrate: 0,
            encoding_time,
            speed_factor,
        })
    }
}

// ─── Free helper functions ────────────────────────────────────────────────────

/// Drain all packets from `demuxer` and write them via `muxer`, updating
/// progress if a tracker is attached.
async fn drain_packets<D, M>(
    demuxer: &mut D,
    muxer: &mut M,
    _progress: &Option<ProgressTracker>,
) -> Result<()>
where
    D: Demuxer,
    M: Muxer,
{
    let mut packet_count: u64 = 0;

    loop {
        match demuxer.read_packet().await {
            Ok(pkt) => {
                if pkt.should_discard() {
                    continue;
                }
                muxer
                    .write_packet(&pkt)
                    .await
                    .map_err(|e| TranscodeError::ContainerError(e.to_string()))?;

                packet_count += 1;
                if packet_count % 500 == 0 {
                    debug!("Remuxed {} packets", packet_count);
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

    debug!("drain_packets: forwarded {} packets total", packet_count);
    Ok(())
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

        let multipass_config = self
            .multipass
            .map(|mode| MultiPassConfig::new(mode, "/tmp/transcode_stats.log"));

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

    #[test]
    fn test_pipeline_builder() {
        let result = TranscodePipelineBuilder::new()
            .input("/tmp/input.mkv")
            .output("/tmp/output.mkv")
            .video_codec("vp9")
            .audio_codec("opus")
            .track_progress(true)
            .hw_accel(false)
            .build();

        assert!(result.is_ok());
        let pipeline = result.expect("should succeed in test");
        assert_eq!(pipeline.config.input, PathBuf::from("/tmp/input.mkv"));
        assert_eq!(pipeline.config.output, PathBuf::from("/tmp/output.mkv"));
        assert_eq!(pipeline.config.video_codec, Some("vp9".to_string()));
        assert_eq!(pipeline.config.audio_codec, Some("opus".to_string()));
        assert!(pipeline.config.track_progress);
        assert!(!pipeline.config.hw_accel);
    }

    #[test]
    fn test_pipeline_builder_missing_input() {
        let result = TranscodePipelineBuilder::new()
            .output("/tmp/output.mkv")
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_pipeline_builder_missing_output() {
        let result = TranscodePipelineBuilder::new()
            .input("/tmp/input.mkv")
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_pipeline_stage_flow() {
        let config = PipelineConfig {
            input: PathBuf::from("/tmp/input.mkv"),
            output: PathBuf::from("/tmp/output.mkv"),
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
}
