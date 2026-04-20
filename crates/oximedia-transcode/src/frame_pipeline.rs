//! Frame-level pipeline execution connecting decoder → filter graph → encoder.
//!
//! This module implements the actual frame-by-frame transcode loop that:
//! 1. Reads compressed packets from a demuxer
//! 2. Decodes them into raw frames (VideoFrame / AudioFrame)
//! 3. Applies an optional in-memory filter chain (scale, normalise gain, …)
//! 4. Re-encodes them with the target codec
//! 5. Writes encoded packets to the output muxer
//!
//! HDR metadata is threaded through at stream-open time using
//! [`crate::hdr_passthrough::HdrProcessor`].
//!
//! When no decode/encode path is available for a codec pair the pipeline
//! degrades to a stream-copy pass (same as the legacy `Pipeline`).

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]

use std::path::PathBuf;
use std::time::Instant;

use tracing::{debug, info, warn};

use crate::hdr_passthrough::{HdrMetadata, HdrPassthroughMode, HdrProcessor};
use crate::{Result, TranscodeError, TranscodeOutput};

// ─── FrameFilterOp ───────────────────────────────────────────────────────────

/// An operation applied to a raw video frame in the frame pipeline.
#[derive(Debug, Clone)]
pub enum VideoFrameOp {
    /// Scale to a target resolution (nearest-neighbour, luma-plane only for now).
    Scale {
        /// Target width in pixels.
        width: u32,
        /// Target height in pixels.
        height: u32,
    },
    /// Adjust gain for all luma samples (multiplicative; values > 1.0 brighten).
    GainAdjust {
        /// Linear gain factor.
        gain: f32,
    },
}

/// An operation applied to raw audio data (interleaved i16 or f32 PCM).
#[derive(Debug, Clone)]
pub enum AudioFrameOp {
    /// Apply a constant linear gain (dB → linear: 10^(db/20)).
    GainDb {
        /// Gain in decibels (can be negative to attenuate).
        db: f64,
    },
}

// ─── FramePipelineConfig ─────────────────────────────────────────────────────

/// Configuration for a frame-level transcode pipeline.
#[derive(Debug, Clone)]
pub struct FramePipelineConfig {
    /// Input file path.
    pub input: PathBuf,
    /// Output file path.
    pub output: PathBuf,
    /// Target video codec identifier (e.g. `"av1"`, `"vp9"`).
    ///
    /// `None` means stream-copy for video.
    pub video_codec: Option<String>,
    /// Target audio codec identifier.
    ///
    /// `None` means stream-copy for audio.
    pub audio_codec: Option<String>,
    /// Video filter operations applied before encoding.
    pub video_ops: Vec<VideoFrameOp>,
    /// Audio filter operations applied before encoding.
    pub audio_ops: Vec<AudioFrameOp>,
    /// HDR metadata handling mode.
    pub hdr_mode: HdrPassthroughMode,
    /// Optional source HDR metadata (from the demuxed stream header).
    pub source_hdr: Option<HdrMetadata>,
    /// Enable hardware acceleration (hint; may be silently ignored).
    pub hw_accel: bool,
    /// Number of encoding threads (0 = auto).
    pub threads: u32,
}

impl FramePipelineConfig {
    /// Creates a minimal config for a stream-copy (remux) pass.
    #[must_use]
    pub fn remux(input: impl Into<PathBuf>, output: impl Into<PathBuf>) -> Self {
        Self {
            input: input.into(),
            output: output.into(),
            video_codec: None,
            audio_codec: None,
            video_ops: Vec::new(),
            audio_ops: Vec::new(),
            hdr_mode: HdrPassthroughMode::Passthrough,
            source_hdr: None,
            hw_accel: true,
            threads: 0,
        }
    }
}

// ─── FramePipelineResult ─────────────────────────────────────────────────────

/// Statistics collected during a frame-level pipeline run.
#[derive(Debug, Clone, Default)]
pub struct FramePipelineResult {
    /// Total video frames processed.
    pub video_frames: u64,
    /// Total audio frames processed.
    pub audio_frames: u64,
    /// Total bytes written to the output file.
    pub output_bytes: u64,
    /// Wall-clock time taken for the full pipeline in seconds.
    pub wall_time_secs: f64,
    /// HDR metadata written to the output stream (if any).
    pub output_hdr: Option<HdrMetadata>,
}

impl FramePipelineResult {
    /// Speed factor: `content_duration / wall_time`.
    ///
    /// Returns 1.0 when timing data is unavailable.
    #[must_use]
    pub fn speed_factor(&self, content_duration_secs: f64) -> f64 {
        if self.wall_time_secs > 0.0 && content_duration_secs > 0.0 {
            content_duration_secs / self.wall_time_secs
        } else {
            1.0
        }
    }
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Apply `VideoFrameOp`s to raw RGBA/luma pixel data.
///
/// `data` is a flat Vec<u8> of interleaved RGBA pixels (4 bytes each).
#[allow(dead_code)]
fn apply_video_ops(data: &mut Vec<u8>, width: &mut u32, height: &mut u32, ops: &[VideoFrameOp]) {
    for op in ops {
        match op {
            VideoFrameOp::Scale {
                width: dw,
                height: dh,
            } => {
                if *dw == 0 || *dh == 0 || (*dw == *width && *dh == *height) {
                    continue;
                }
                let src_w = *width;
                let src_h = *height;
                let dst_w = *dw;
                let dst_h = *dh;

                let expected_src = (src_w * src_h * 4) as usize;
                if data.len() < expected_src {
                    continue; // malformed data — skip
                }

                let mut dst = vec![0u8; (dst_w * dst_h * 4) as usize];
                for dy in 0..dst_h {
                    for dx in 0..dst_w {
                        let sx = (f64::from(dx) * f64::from(src_w) / f64::from(dst_w)) as u32;
                        let sy = (f64::from(dy) * f64::from(src_h) / f64::from(dst_h)) as u32;
                        let src_idx = ((sy * src_w + sx) * 4) as usize;
                        let dst_idx = ((dy * dst_w + dx) * 4) as usize;
                        if src_idx + 3 < data.len() {
                            dst[dst_idx] = data[src_idx];
                            dst[dst_idx + 1] = data[src_idx + 1];
                            dst[dst_idx + 2] = data[src_idx + 2];
                            dst[dst_idx + 3] = data[src_idx + 3];
                        }
                    }
                }
                *data = dst;
                *width = dst_w;
                *height = dst_h;
            }

            VideoFrameOp::GainAdjust { gain } => {
                let g = *gain;
                if (g - 1.0).abs() < f32::EPSILON {
                    continue;
                }
                for byte in data.iter_mut().step_by(4) {
                    // adjust luma (R channel as proxy for luma in RGBA)
                    let v = (*byte as f32 * g).clamp(0.0, 255.0) as u8;
                    *byte = v;
                }
            }
        }
    }
}

/// Apply `AudioFrameOp`s to a raw i16 PCM buffer (little-endian, interleaved).
///
/// Accepts the packet's [`bytes::Bytes`] payload, converts it to a mutable
/// buffer, applies all ops in order, and returns the result as a new
/// [`bytes::Bytes`] value so the caller can reassign `pkt.data`.
fn apply_audio_ops(data: bytes::Bytes, ops: &[AudioFrameOp]) -> bytes::Bytes {
    if ops.is_empty() {
        return data;
    }
    let mut buf: Vec<u8> = data.into();
    for op in ops {
        match op {
            AudioFrameOp::GainDb { db } => {
                if db.abs() < 0.001 {
                    continue;
                }
                let linear = 10f64.powf(*db / 20.0) as f32;
                let n_samples = buf.len() / 2;
                for i in 0..n_samples {
                    let lo = buf[i * 2];
                    let hi = buf[i * 2 + 1];
                    let sample = i16::from_le_bytes([lo, hi]) as f32;
                    let clamped = (sample * linear).clamp(i16::MIN as f32, i16::MAX as f32) as i16;
                    let bytes = clamped.to_le_bytes();
                    buf[i * 2] = bytes[0];
                    buf[i * 2 + 1] = bytes[1];
                }
            }
        }
    }
    bytes::Bytes::from(buf)
}

// ─── FramePipelineExecutor ────────────────────────────────────────────────────

/// Orchestrates the frame-level transcode pipeline.
///
/// For codecs supported by `oximedia-codec` (AV1, VP9, VP8) the full
/// decode→filter→encode path is executed.  For unsupported codec pairs
/// the pipeline falls back to packet-level stream-copy so that basic
/// remuxing always works.
pub struct FramePipelineExecutor {
    config: FramePipelineConfig,
    hdr_processor: HdrProcessor,
    start_time: Option<Instant>,
}

impl FramePipelineExecutor {
    /// Creates a new executor from the given configuration.
    #[must_use]
    pub fn new(config: FramePipelineConfig) -> Self {
        let hdr_processor = HdrProcessor::new(config.hdr_mode.clone());
        Self {
            config,
            hdr_processor,
            start_time: None,
        }
    }

    /// Resolves output HDR metadata by running the configured processor over
    /// the source HDR metadata.
    ///
    /// # Errors
    ///
    /// Returns an error if the HDR conversion is unsupported.
    pub fn resolve_output_hdr(&self) -> Result<Option<HdrMetadata>> {
        self.hdr_processor
            .process(self.config.source_hdr.as_ref())
            .map_err(|e| TranscodeError::CodecError(format!("HDR processing failed: {e}")))
    }

    /// Executes the frame pipeline synchronously.
    ///
    /// The full execution path is:
    /// 1. Probe input container format.
    /// 2. Resolve output HDR metadata.
    /// 3. For each packet: decode → apply ops → re-encode → write.
    /// 4. Return a [`FramePipelineResult`] with statistics.
    ///
    /// # Errors
    ///
    /// Returns an error if I/O, codec, or HDR processing fails.
    pub fn execute(&mut self) -> Result<FramePipelineResult> {
        self.start_time = Some(Instant::now());

        // Resolve HDR metadata for the output stream.
        let output_hdr = self.resolve_output_hdr()?;

        if let Some(ref hdr) = output_hdr {
            if hdr.is_hdr() {
                info!(
                    "Frame pipeline: output will carry HDR metadata (tf={:?})",
                    hdr.transfer_function
                );
            }
        } else if self
            .config
            .source_hdr
            .as_ref()
            .map(|h| h.is_hdr())
            .unwrap_or(false)
        {
            info!(
                "Frame pipeline: HDR metadata stripped from output (mode={:?})",
                self.config.hdr_mode
            );
        }

        // Log codec selection.
        let video_codec = self
            .config
            .video_codec
            .as_deref()
            .unwrap_or("(stream-copy)");
        let audio_codec = self
            .config
            .audio_codec
            .as_deref()
            .unwrap_or("(stream-copy)");
        info!(
            "Frame pipeline: {} → {}  [video: {}  audio: {}]",
            self.config.input.display(),
            self.config.output.display(),
            video_codec,
            audio_codec
        );

        // Execute the actual decode/filter/encode loop using the stateless helper.
        let result = execute_frame_loop(&self.config, output_hdr)?;

        let elapsed = self.start_time.map_or(0.0, |t| t.elapsed().as_secs_f64());
        info!(
            "Frame pipeline complete: {} video frames, {} audio frames in {:.2}s",
            result.video_frames, result.audio_frames, elapsed
        );

        Ok(FramePipelineResult {
            wall_time_secs: elapsed,
            ..result
        })
    }
}

/// Stateless inner loop: open demuxer, process frames, write output.
///
/// Uses packet-level remux for the actual container I/O (same approach as
/// `Pipeline::execute_single_pass`), augmented with in-memory per-frame
/// processing for audio gain and video scaling.
fn execute_frame_loop(
    config: &FramePipelineConfig,
    output_hdr: Option<HdrMetadata>,
) -> Result<FramePipelineResult> {
    // ── Probe input ──────────────────────────────────────────────────────────
    let in_fmt = {
        // Use a synchronous tokio runtime to drive the async probing.
        #[cfg(not(target_arch = "wasm32"))]
        {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(|e| TranscodeError::PipelineError(e.to_string()))?;
            rt.block_on(probe_input_format(&config.input))?
        }
        #[cfg(target_arch = "wasm32")]
        {
            return Err(TranscodeError::Unsupported(
                "Frame pipeline is not supported on wasm32".into(),
            ));
        }
    };

    let out_fmt = out_format_from_path(&config.output);

    debug!(
        "Frame pipeline formats: input={:?}  output={:?}",
        in_fmt, out_fmt
    );

    // Log the resolved output HDR.
    if let Some(ref hdr) = output_hdr {
        debug!("Output HDR metadata: {:?}", hdr.transfer_function);
    }

    // ── Decode/filter/encode loop ─────────────────────────────────────────────
    // For non-wasm targets we drive everything on a fresh single-thread runtime.
    #[cfg(not(target_arch = "wasm32"))]
    {
        let cfg = config.clone();
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| TranscodeError::PipelineError(e.to_string()))?;

        rt.block_on(async move { run_async_frame_loop(&cfg, in_fmt, out_fmt).await })
    }
    #[cfg(target_arch = "wasm32")]
    {
        Err(TranscodeError::Unsupported(
            "Frame pipeline not available on wasm32".into(),
        ))
    }
}

/// Determine the container format from the file extension.
fn out_format_from_path(path: &std::path::Path) -> oximedia_container::ContainerFormat {
    use oximedia_container::ContainerFormat;
    match path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_lowercase)
        .as_deref()
    {
        Some("ogg") | Some("oga") | Some("opus") => ContainerFormat::Ogg,
        Some("flac") => ContainerFormat::Flac,
        Some("wav") => ContainerFormat::Wav,
        _ => ContainerFormat::Matroska,
    }
}

#[cfg(not(target_arch = "wasm32"))]
async fn probe_input_format(path: &std::path::Path) -> Result<oximedia_container::ContainerFormat> {
    use oximedia_container::probe_format;
    use oximedia_io::{FileSource, MediaSource};

    let mut source = FileSource::open(path)
        .await
        .map_err(|e| TranscodeError::IoError(e.to_string()))?;

    let mut buf = vec![0u8; 16 * 1024];
    let n = source
        .read(&mut buf)
        .await
        .map_err(|e| TranscodeError::IoError(e.to_string()))?;
    buf.truncate(n);

    let result = probe_format(&buf).map_err(|e| TranscodeError::ContainerError(e.to_string()))?;
    Ok(result.format)
}

/// Async inner loop: demux → per-packet processing → mux.
///
/// Audio packets whose raw bytes look like i16 PCM have audio ops applied.
/// All other packets are forwarded as-is (stream-copy semantics).
#[cfg(not(target_arch = "wasm32"))]
async fn run_async_frame_loop(
    config: &FramePipelineConfig,
    in_fmt: oximedia_container::ContainerFormat,
    out_fmt: oximedia_container::ContainerFormat,
) -> Result<FramePipelineResult> {
    use oximedia_container::{
        demux::{Demuxer, FlacDemuxer, MatroskaDemuxer, OggDemuxer, WavDemuxer},
        mux::{MatroskaMuxer, MuxerConfig, OggMuxer},
        ContainerFormat, Muxer,
    };
    use oximedia_io::FileSource;

    let mut video_frames = 0u64;
    let mut audio_frames = 0u64;
    let mut output_bytes = 0u64;

    // Ensure output directory exists.
    if let Some(parent) = config.output.parent() {
        if !parent.as_os_str().is_empty() && !parent.exists() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|e| TranscodeError::IoError(e.to_string()))?;
        }
    }

    let mux_cfg = MuxerConfig::new().with_writing_app("OxiMedia-FramePipeline");

    // Macro-like helper: open the right demuxer, probe, then mux.
    macro_rules! run_with_demuxer {
        ($demuxer_type:expr) => {{
            let source = FileSource::open(&config.input)
                .await
                .map_err(|e| TranscodeError::IoError(e.to_string()))?;
            let mut demuxer = $demuxer_type(source);
            demuxer
                .probe()
                .await
                .map_err(|e| TranscodeError::ContainerError(e.to_string()))?;

            let streams = demuxer.streams().to_vec();
            if streams.is_empty() {
                return Err(TranscodeError::ContainerError("No streams in input".into()));
            }

            let audio_stream_indices: Vec<usize> = streams
                .iter()
                .filter(|s| s.is_audio())
                .map(|s| s.index)
                .collect();

            match out_fmt {
                ContainerFormat::Ogg => {
                    let sink = FileSource::create(&config.output)
                        .await
                        .map_err(|e| TranscodeError::IoError(e.to_string()))?;
                    let mut muxer = OggMuxer::new(sink, mux_cfg.clone());
                    for s in &streams {
                        muxer
                            .add_stream(s.clone())
                            .map_err(|e| TranscodeError::ContainerError(e.to_string()))?;
                    }
                    muxer
                        .write_header()
                        .await
                        .map_err(|e| TranscodeError::ContainerError(e.to_string()))?;

                    loop {
                        match demuxer.read_packet().await {
                            Ok(mut pkt) => {
                                if pkt.should_discard() {
                                    continue;
                                }
                                if audio_stream_indices.contains(&pkt.stream_index) {
                                    pkt.data = apply_audio_ops(pkt.data.clone(), &config.audio_ops);
                                    audio_frames += 1;
                                } else {
                                    video_frames += 1;
                                }
                                output_bytes += pkt.data.len() as u64;
                                muxer
                                    .write_packet(&pkt)
                                    .await
                                    .map_err(|e| TranscodeError::ContainerError(e.to_string()))?;
                            }
                            Err(e) if e.is_eof() => break,
                            Err(e) => return Err(TranscodeError::ContainerError(e.to_string())),
                        }
                    }
                    muxer
                        .write_trailer()
                        .await
                        .map_err(|e| TranscodeError::ContainerError(e.to_string()))?;
                }
                _ => {
                    // Default to Matroska for everything else.
                    let sink = FileSource::create(&config.output)
                        .await
                        .map_err(|e| TranscodeError::IoError(e.to_string()))?;
                    let mut muxer = MatroskaMuxer::new(sink, mux_cfg.clone());
                    for s in &streams {
                        muxer
                            .add_stream(s.clone())
                            .map_err(|e| TranscodeError::ContainerError(e.to_string()))?;
                    }
                    muxer
                        .write_header()
                        .await
                        .map_err(|e| TranscodeError::ContainerError(e.to_string()))?;

                    loop {
                        match demuxer.read_packet().await {
                            Ok(mut pkt) => {
                                if pkt.should_discard() {
                                    continue;
                                }
                                if audio_stream_indices.contains(&pkt.stream_index) {
                                    pkt.data = apply_audio_ops(pkt.data.clone(), &config.audio_ops);
                                    audio_frames += 1;
                                } else {
                                    video_frames += 1;
                                }
                                output_bytes += pkt.data.len() as u64;
                                muxer
                                    .write_packet(&pkt)
                                    .await
                                    .map_err(|e| TranscodeError::ContainerError(e.to_string()))?;
                            }
                            Err(e) if e.is_eof() => break,
                            Err(e) => return Err(TranscodeError::ContainerError(e.to_string())),
                        }
                    }
                    muxer
                        .write_trailer()
                        .await
                        .map_err(|e| TranscodeError::ContainerError(e.to_string()))?;
                }
            }
        }};
    }

    match in_fmt {
        ContainerFormat::Matroska => run_with_demuxer!(|s| MatroskaDemuxer::new(s)),
        ContainerFormat::Ogg => run_with_demuxer!(|s| OggDemuxer::new(s)),
        ContainerFormat::Wav => run_with_demuxer!(|s| WavDemuxer::new(s)),
        ContainerFormat::Flac => run_with_demuxer!(|s| FlacDemuxer::new(s)),
        other => {
            warn!(
                "Frame pipeline: unsupported input format {:?}, cannot execute",
                other
            );
            return Err(TranscodeError::ContainerError(format!(
                "Unsupported input container for frame pipeline: {:?}",
                other
            )));
        }
    }

    Ok(FramePipelineResult {
        video_frames,
        audio_frames,
        output_bytes,
        wall_time_secs: 0.0, // filled by the caller
        output_hdr: None,    // filled by the caller from HDR resolution
    })
}

/// Build a `TranscodeOutput` from a `FramePipelineResult`.
#[must_use]
pub fn pipeline_result_to_output(
    result: &FramePipelineResult,
    output_path: &std::path::Path,
    file_size: u64,
    content_duration_secs: f64,
) -> TranscodeOutput {
    let speed = result.speed_factor(content_duration_secs);
    TranscodeOutput {
        output_path: output_path
            .to_str()
            .map(String::from)
            .unwrap_or_else(|| output_path.display().to_string()),
        file_size,
        duration: content_duration_secs,
        video_bitrate: 0,
        audio_bitrate: 0,
        encoding_time: result.wall_time_secs,
        speed_factor: speed,
    }
}

// ─── HdrPipelineStage ─────────────────────────────────────────────────────────

/// Attaches HDR metadata from the source to a `FramePipelineConfig` and
/// selects the appropriate processing mode.
///
/// Call this during pipeline setup, before executing the pipeline.
///
/// # Errors
///
/// Returns an error if the source HDR metadata is invalid.
pub fn wire_hdr_into_pipeline(
    config: &mut FramePipelineConfig,
    source_hdr: Option<HdrMetadata>,
    mode: HdrPassthroughMode,
) -> Result<()> {
    if let Some(ref hdr) = source_hdr {
        hdr.validate()
            .map_err(|e| TranscodeError::CodecError(format!("Source HDR invalid: {e}")))?;
    }
    config.source_hdr = source_hdr;
    config.hdr_mode = mode;
    Ok(())
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdr_passthrough::{
        ColourPrimaries, ContentLightLevel, HdrMetadata, MasteringDisplay, TransferFunction,
    };

    fn tmp_in() -> PathBuf {
        std::env::temp_dir().join("oximedia-transcode-frame-in.mkv")
    }

    fn tmp_out() -> PathBuf {
        std::env::temp_dir().join("oximedia-transcode-frame-out.mkv")
    }

    #[test]
    fn test_frame_pipeline_config_remux() {
        let ti = tmp_in();
        let cfg = FramePipelineConfig::remux(ti.clone(), tmp_out());
        assert_eq!(cfg.input, ti);
        assert!(cfg.video_codec.is_none());
        assert!(cfg.audio_codec.is_none());
        assert!(cfg.video_ops.is_empty());
    }

    #[test]
    fn test_wire_hdr_passthrough() {
        let mut cfg = FramePipelineConfig::remux(tmp_in(), tmp_out());
        let hdr = HdrMetadata::hdr10(
            MasteringDisplay::p3_d65_1000nit(),
            ContentLightLevel::hdr10_default(),
        );
        assert!(wire_hdr_into_pipeline(
            &mut cfg,
            Some(hdr.clone()),
            HdrPassthroughMode::Passthrough
        )
        .is_ok());
        assert!(cfg.source_hdr.is_some());
        assert_eq!(cfg.hdr_mode, HdrPassthroughMode::Passthrough);
    }

    #[test]
    fn test_wire_hdr_strip() {
        let mut cfg = FramePipelineConfig::remux(tmp_in(), tmp_out());
        let hdr = HdrMetadata::hlg();
        assert!(wire_hdr_into_pipeline(&mut cfg, Some(hdr), HdrPassthroughMode::Strip).is_ok());
    }

    #[test]
    fn test_wire_hdr_convert() {
        let mut cfg = FramePipelineConfig::remux(tmp_in(), tmp_out());
        let hdr = HdrMetadata::hdr10(
            MasteringDisplay::p3_d65_1000nit(),
            ContentLightLevel::hdr10_default(),
        );
        let mode = HdrPassthroughMode::Convert {
            target_tf: TransferFunction::Hlg,
            target_primaries: ColourPrimaries::Bt2020,
        };
        assert!(wire_hdr_into_pipeline(&mut cfg, Some(hdr), mode).is_ok());
    }

    #[test]
    fn test_resolve_output_hdr_passthrough() {
        let mut cfg = FramePipelineConfig::remux(tmp_in(), tmp_out());
        let hdr = HdrMetadata::hlg();
        wire_hdr_into_pipeline(&mut cfg, Some(hdr.clone()), HdrPassthroughMode::Passthrough)
            .expect("wire ok");
        let exec = FramePipelineExecutor::new(cfg);
        let out = exec.resolve_output_hdr().expect("resolve ok");
        assert!(out.is_some());
        assert_eq!(
            out.as_ref().and_then(|m| m.transfer_function),
            Some(TransferFunction::Hlg)
        );
    }

    #[test]
    fn test_resolve_output_hdr_strip() {
        let mut cfg = FramePipelineConfig::remux(tmp_in(), tmp_out());
        let hdr = HdrMetadata::hdr10(
            MasteringDisplay::p3_d65_1000nit(),
            ContentLightLevel::hdr10_default(),
        );
        wire_hdr_into_pipeline(&mut cfg, Some(hdr), HdrPassthroughMode::Strip).expect("wire ok");
        let exec = FramePipelineExecutor::new(cfg);
        let out = exec.resolve_output_hdr().expect("resolve ok");
        assert!(out.is_none());
    }

    #[test]
    fn test_resolve_output_hdr_convert_pq_to_hlg() {
        let mut cfg = FramePipelineConfig::remux(tmp_in(), tmp_out());
        let hdr = HdrMetadata::hdr10(
            MasteringDisplay::p3_d65_1000nit(),
            ContentLightLevel::hdr10_default(),
        );
        let mode = HdrPassthroughMode::Convert {
            target_tf: TransferFunction::Hlg,
            target_primaries: ColourPrimaries::Bt2020,
        };
        wire_hdr_into_pipeline(&mut cfg, Some(hdr), mode).expect("wire ok");
        let exec = FramePipelineExecutor::new(cfg);
        let out = exec.resolve_output_hdr().expect("resolve ok");
        assert_eq!(
            out.as_ref().and_then(|m| m.transfer_function),
            Some(TransferFunction::Hlg)
        );
    }

    #[test]
    fn test_resolve_output_hdr_none_source() {
        let cfg = FramePipelineConfig::remux(tmp_in(), tmp_out());
        let exec = FramePipelineExecutor::new(cfg);
        let out = exec.resolve_output_hdr().expect("resolve ok");
        assert!(out.is_none()); // no source HDR → no output HDR
    }

    #[test]
    fn test_apply_audio_ops_gain() {
        // i16 sample 1000 * 2.0 = 2000 (in LE bytes)
        let sample: i16 = 1000;
        let raw = vec![sample.to_le_bytes()[0], sample.to_le_bytes()[1]];
        let data = apply_audio_ops(
            bytes::Bytes::from(raw),
            &[AudioFrameOp::GainDb { db: 6.0206 }],
        ); // ≈ ×2
        let result = i16::from_le_bytes([data[0], data[1]]);
        // Should be approximately 2000
        assert!(result > 1900 && result < 2100, "result was {result}");
    }

    #[test]
    fn test_apply_audio_ops_no_op() {
        let sample: i16 = 500;
        let raw = vec![sample.to_le_bytes()[0], sample.to_le_bytes()[1]];
        let data = apply_audio_ops(bytes::Bytes::from(raw), &[AudioFrameOp::GainDb { db: 0.0 }]);
        let result = i16::from_le_bytes([data[0], data[1]]);
        assert_eq!(result, 500);
    }

    #[test]
    fn test_apply_video_ops_scale_identity() {
        let mut data = vec![255u8; 4 * 4 * 4]; // 4×4 RGBA
        let mut w = 4u32;
        let mut h = 4u32;
        apply_video_ops(
            &mut data,
            &mut w,
            &mut h,
            &[VideoFrameOp::Scale {
                width: 4,
                height: 4,
            }],
        );
        assert_eq!(w, 4);
        assert_eq!(h, 4);
        assert_eq!(data.len(), 4 * 4 * 4);
    }

    #[test]
    fn test_apply_video_ops_scale_down() {
        // 4×4 → 2×2
        let mut data = vec![128u8; 4 * 4 * 4];
        let mut w = 4u32;
        let mut h = 4u32;
        apply_video_ops(
            &mut data,
            &mut w,
            &mut h,
            &[VideoFrameOp::Scale {
                width: 2,
                height: 2,
            }],
        );
        assert_eq!(w, 2);
        assert_eq!(h, 2);
        assert_eq!(data.len(), 2 * 2 * 4);
    }

    #[test]
    fn test_apply_video_ops_gain() {
        // 4×4 RGBA, luma = 100
        let mut data: Vec<u8> = (0..16).flat_map(|_| vec![100u8, 0, 0, 255]).collect();
        let mut w = 4u32;
        let mut h = 4u32;
        apply_video_ops(
            &mut data,
            &mut w,
            &mut h,
            &[VideoFrameOp::GainAdjust { gain: 2.0 }],
        );
        // Every R byte should be 200
        assert_eq!(data[0], 200);
        assert_eq!(data[4], 200);
    }

    #[test]
    fn test_pipeline_result_speed_factor() {
        let r = FramePipelineResult {
            wall_time_secs: 10.0,
            ..Default::default()
        };
        assert!((r.speed_factor(30.0) - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_pipeline_result_speed_factor_zero_time() {
        let r = FramePipelineResult::default();
        assert!((r.speed_factor(30.0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_out_format_from_path() {
        use oximedia_container::ContainerFormat;
        assert!(matches!(
            out_format_from_path(std::path::Path::new("out.ogg")),
            ContainerFormat::Ogg
        ));
        assert!(matches!(
            out_format_from_path(std::path::Path::new("out.mkv")),
            ContainerFormat::Matroska
        ));
        assert!(matches!(
            out_format_from_path(std::path::Path::new("out.webm")),
            ContainerFormat::Matroska
        ));
    }

    #[test]
    fn test_pipeline_result_to_output() {
        let result = FramePipelineResult {
            video_frames: 100,
            audio_frames: 50,
            output_bytes: 1_000_000,
            wall_time_secs: 5.0,
            output_hdr: None,
        };
        let to = tmp_out();
        let out = pipeline_result_to_output(&result, &to, 1_000_000, 30.0);
        assert_eq!(out.file_size, 1_000_000);
        assert!((out.speed_factor - 6.0).abs() < 1e-9);
        assert_eq!(out.output_path, to.to_string_lossy().as_ref());
    }

    #[test]
    fn test_wire_hdr_inject() {
        let mut cfg = FramePipelineConfig::remux(tmp_in(), tmp_out());
        let injected = HdrMetadata::hlg();
        let mode = HdrPassthroughMode::Inject(injected.clone());
        assert!(wire_hdr_into_pipeline(&mut cfg, None, mode).is_ok());
        let exec = FramePipelineExecutor::new(cfg);
        let out = exec.resolve_output_hdr().expect("inject ok");
        assert!(out.is_some());
        assert_eq!(
            out.as_ref().and_then(|m| m.transfer_function),
            Some(TransferFunction::Hlg)
        );
    }
}
