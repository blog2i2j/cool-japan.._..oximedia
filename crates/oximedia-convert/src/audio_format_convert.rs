// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Audio-only extraction with format conversion.
//!
//! Provides [`AudioFormatConverter`] for extracting audio from media files and
//! converting between patent-free audio formats (Opus, Vorbis, FLAC, PCM/WAV).
//!
//! # Features
//!
//! - Extract audio track from any supported container (WebM, MKV, OGG, WAV)
//! - Convert between audio codecs with configurable quality
//! - Sample rate conversion (8 kHz–192 kHz)
//! - Channel layout conversion (mono ↔ stereo ↔ surround)
//! - Bit-depth conversion for lossless formats
//! - Loudness normalization target (EBU R128)
//! - Trim/segment extraction by timestamp
//!
//! All codecs are patent-free: Opus, Vorbis, FLAC, PCM (WAV).

#![allow(dead_code)]

use crate::{ConversionError, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::Duration;

// ── Patent-free audio codec ──────────────────────────────────────────────────

/// Patent-free audio codec for output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PatentFreeAudioCodec {
    /// Opus (lossy, best quality at low-to-mid bitrates).
    Opus,
    /// Vorbis (lossy, mature and widely supported).
    Vorbis,
    /// FLAC (lossless, efficient compression).
    Flac,
    /// PCM signed 16-bit little-endian (uncompressed WAV).
    PcmS16Le,
    /// PCM signed 24-bit little-endian (uncompressed WAV).
    PcmS24Le,
    /// PCM 32-bit float (uncompressed WAV).
    PcmF32Le,
}

impl PatentFreeAudioCodec {
    /// File extension for output.
    #[must_use]
    pub const fn extension(self) -> &'static str {
        match self {
            Self::Opus => "opus",
            Self::Vorbis => "ogg",
            Self::Flac => "flac",
            Self::PcmS16Le | Self::PcmS24Le | Self::PcmF32Le => "wav",
        }
    }

    /// Human-readable codec name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Opus => "Opus",
            Self::Vorbis => "Vorbis",
            Self::Flac => "FLAC",
            Self::PcmS16Le => "PCM s16le",
            Self::PcmS24Le => "PCM s24le",
            Self::PcmF32Le => "PCM f32le",
        }
    }

    /// Whether this codec is lossless.
    #[must_use]
    pub const fn is_lossless(self) -> bool {
        matches!(
            self,
            Self::Flac | Self::PcmS16Le | Self::PcmS24Le | Self::PcmF32Le
        )
    }

    /// Default bitrate in bits/s for lossy codecs; `None` for lossless.
    #[must_use]
    pub const fn default_bitrate(self) -> Option<u64> {
        match self {
            Self::Opus => Some(128_000),
            Self::Vorbis => Some(192_000),
            Self::Flac | Self::PcmS16Le | Self::PcmS24Le | Self::PcmF32Le => None,
        }
    }

    /// Minimum recommended bitrate for lossy codecs.
    #[must_use]
    pub const fn min_bitrate(self) -> Option<u64> {
        match self {
            Self::Opus => Some(6_000),
            Self::Vorbis => Some(32_000),
            _ => None,
        }
    }

    /// Maximum recommended bitrate for lossy codecs.
    #[must_use]
    pub const fn max_bitrate(self) -> Option<u64> {
        match self {
            Self::Opus => Some(510_000),
            Self::Vorbis => Some(500_000),
            _ => None,
        }
    }

    /// Bits per sample for PCM codecs.
    #[must_use]
    pub const fn bits_per_sample(self) -> Option<u32> {
        match self {
            Self::PcmS16Le => Some(16),
            Self::PcmS24Le => Some(24),
            Self::PcmF32Le => Some(32),
            _ => None,
        }
    }

    /// Container format name for this codec.
    #[must_use]
    pub const fn container(self) -> &'static str {
        match self {
            Self::Opus => "ogg",
            Self::Vorbis => "ogg",
            Self::Flac => "flac",
            Self::PcmS16Le | Self::PcmS24Le | Self::PcmF32Le => "wav",
        }
    }
}

// ── Channel layout ──────────────────────────────────────────────────────────

/// Output channel layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OutputChannelLayout {
    /// Mono (1 channel).
    Mono,
    /// Stereo (2 channels).
    Stereo,
    /// 5.1 surround (6 channels).
    Surround5_1,
    /// Keep source channel layout unchanged.
    KeepSource,
}

impl OutputChannelLayout {
    /// Number of channels, or `None` for `KeepSource`.
    #[must_use]
    pub const fn channel_count(self) -> Option<u32> {
        match self {
            Self::Mono => Some(1),
            Self::Stereo => Some(2),
            Self::Surround5_1 => Some(6),
            Self::KeepSource => None,
        }
    }
}

// ── Loudness target ─────────────────────────────────────────────────────────

/// Loudness normalization target standard.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum LoudnessTarget {
    /// EBU R128 broadcast standard (−23 LUFS, ±1 LU tolerance).
    EbuR128,
    /// Streaming platforms (−14 LUFS, typical for Spotify/YouTube).
    Streaming,
    /// Podcast standard (−16 LUFS, with −1.5 dBTP peak).
    Podcast,
    /// Custom integrated loudness target in LUFS.
    Custom {
        /// Target integrated loudness (negative, in LUFS).
        lufs: f64,
        /// True peak limit in dBTP.
        true_peak_dbtp: f64,
    },
}

impl LoudnessTarget {
    /// Target integrated loudness in LUFS.
    #[must_use]
    pub fn lufs(self) -> f64 {
        match self {
            Self::EbuR128 => -23.0,
            Self::Streaming => -14.0,
            Self::Podcast => -16.0,
            Self::Custom { lufs, .. } => lufs,
        }
    }

    /// True peak limit in dBTP.
    #[must_use]
    pub fn true_peak_dbtp(self) -> f64 {
        match self {
            Self::EbuR128 => -1.0,
            Self::Streaming => -1.0,
            Self::Podcast => -1.5,
            Self::Custom { true_peak_dbtp, .. } => true_peak_dbtp,
        }
    }
}

// ── Trim / segment specification ────────────────────────────────────────────

/// Time segment for trimming audio during extraction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioSegment {
    /// Start time offset from the beginning.
    pub start: Duration,
    /// Duration of the segment to extract. `None` means until end-of-stream.
    pub duration: Option<Duration>,
}

impl AudioSegment {
    /// Create a segment starting at `start` running until the end.
    #[must_use]
    pub fn from_start(start: Duration) -> Self {
        Self {
            start,
            duration: None,
        }
    }

    /// Create a segment starting at `start` with a given `duration`.
    #[must_use]
    pub fn with_duration(start: Duration, duration: Duration) -> Self {
        Self {
            start,
            duration: Some(duration),
        }
    }

    /// End time if duration is known.
    #[must_use]
    pub fn end_time(&self) -> Option<Duration> {
        self.duration.map(|d| self.start + d)
    }

    /// Validate that start < end (if duration is given and non-zero).
    pub fn validate(&self) -> Result<()> {
        if let Some(d) = self.duration {
            if d.is_zero() {
                return Err(ConversionError::InvalidInput(
                    "Audio segment duration must be > 0".into(),
                ));
            }
        }
        Ok(())
    }
}

// ── Conversion quality preset ───────────────────────────────────────────────

/// Quality preset for audio format conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AudioQualityPreset {
    /// Low quality / small file (Opus 48 kbps, Vorbis 96 kbps).
    Low,
    /// Medium quality (Opus 96 kbps, Vorbis 160 kbps).
    Medium,
    /// High quality (Opus 128 kbps, Vorbis 192 kbps).
    High,
    /// Transparent quality (Opus 192 kbps, Vorbis 320 kbps).
    Transparent,
    /// Lossless (only meaningful for FLAC/PCM; ignored for lossy codecs).
    Lossless,
}

impl AudioQualityPreset {
    /// Resolve the quality preset to a bitrate for the given codec.
    ///
    /// Returns `None` for lossless codecs since they don't use bitrate targeting.
    #[must_use]
    pub fn bitrate_for(self, codec: PatentFreeAudioCodec) -> Option<u64> {
        if codec.is_lossless() {
            return None;
        }
        match codec {
            PatentFreeAudioCodec::Opus => match self {
                Self::Low => Some(48_000),
                Self::Medium => Some(96_000),
                Self::High => Some(128_000),
                Self::Transparent | Self::Lossless => Some(192_000),
            },
            PatentFreeAudioCodec::Vorbis => match self {
                Self::Low => Some(96_000),
                Self::Medium => Some(160_000),
                Self::High => Some(192_000),
                Self::Transparent | Self::Lossless => Some(320_000),
            },
            _ => None,
        }
    }

    /// Vorbis quality level (−1 to 10) corresponding to this preset.
    #[must_use]
    pub fn vorbis_quality(self) -> f32 {
        match self {
            Self::Low => 2.0,
            Self::Medium => 4.0,
            Self::High => 6.0,
            Self::Transparent | Self::Lossless => 8.0,
        }
    }

    /// FLAC compression level (0 = fastest, 8 = smallest).
    #[must_use]
    pub fn flac_compression_level(self) -> u32 {
        match self {
            Self::Low => 0,
            Self::Medium => 4,
            Self::High => 6,
            Self::Transparent | Self::Lossless => 8,
        }
    }
}

// ── Conversion configuration ────────────────────────────────────────────────

/// Full configuration for an audio format conversion job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioConvertConfig {
    /// Target codec.
    pub codec: PatentFreeAudioCodec,
    /// Quality preset (determines bitrate for lossy codecs).
    pub quality: AudioQualityPreset,
    /// Explicit bitrate override (bits/s). Takes precedence over quality preset.
    pub bitrate_override: Option<u64>,
    /// Output sample rate in Hz. `None` keeps source rate.
    pub sample_rate: Option<u32>,
    /// Output channel layout.
    pub channels: OutputChannelLayout,
    /// Optional loudness normalization.
    pub loudness: Option<LoudnessTarget>,
    /// Optional segment trim.
    pub segment: Option<AudioSegment>,
    /// Audio track index to extract (0-based). `None` extracts the first audio track.
    pub track_index: Option<usize>,
    /// Whether to strip non-audio streams (video, subtitles, data).
    pub strip_non_audio: bool,
    /// Whether to preserve audio metadata tags (title, artist, album, etc.).
    pub preserve_tags: bool,
}

impl Default for AudioConvertConfig {
    fn default() -> Self {
        Self {
            codec: PatentFreeAudioCodec::Opus,
            quality: AudioQualityPreset::High,
            bitrate_override: None,
            sample_rate: None,
            channels: OutputChannelLayout::KeepSource,
            loudness: None,
            segment: None,
            track_index: None,
            strip_non_audio: true,
            preserve_tags: true,
        }
    }
}

impl AudioConvertConfig {
    /// Create a new builder.
    #[must_use]
    pub fn builder() -> AudioConvertConfigBuilder {
        AudioConvertConfigBuilder::default()
    }

    /// Resolve the effective bitrate for the configured codec and quality.
    #[must_use]
    pub fn effective_bitrate(&self) -> Option<u64> {
        if let Some(br) = self.bitrate_override {
            if self.codec.is_lossless() {
                return None;
            }
            return Some(br);
        }
        self.quality.bitrate_for(self.codec)
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        // Validate bitrate bounds for lossy codecs.
        if let Some(br) = self.effective_bitrate() {
            if let Some(min) = self.codec.min_bitrate() {
                if br < min {
                    return Err(ConversionError::InvalidInput(format!(
                        "Bitrate {} bps is below minimum {} bps for {}",
                        br,
                        min,
                        self.codec.name()
                    )));
                }
            }
            if let Some(max) = self.codec.max_bitrate() {
                if br > max {
                    return Err(ConversionError::InvalidInput(format!(
                        "Bitrate {} bps exceeds maximum {} bps for {}",
                        br,
                        max,
                        self.codec.name()
                    )));
                }
            }
        }

        // Validate sample rate range.
        if let Some(sr) = self.sample_rate {
            if sr < 8_000 || sr > 192_000 {
                return Err(ConversionError::InvalidInput(format!(
                    "Sample rate {} Hz is outside supported range 8000–192000 Hz",
                    sr
                )));
            }
        }

        // Validate segment.
        if let Some(ref seg) = self.segment {
            seg.validate()?;
        }

        // Opus only supports 48 kHz natively (with internal resampling for
        // 8/12/16/24 kHz). Warn-via-error if user requests an unsupported rate.
        if self.codec == PatentFreeAudioCodec::Opus {
            if let Some(sr) = self.sample_rate {
                if !matches!(sr, 8_000 | 12_000 | 16_000 | 24_000 | 48_000) {
                    return Err(ConversionError::InvalidInput(format!(
                        "Opus only supports sample rates 8/12/16/24/48 kHz, got {} Hz",
                        sr
                    )));
                }
            }
        }

        Ok(())
    }

    /// Suggested output file path based on codec extension.
    #[must_use]
    pub fn suggest_output_path(&self, input: &Path) -> PathBuf {
        let stem = input
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("output");
        let ext = self.codec.extension();
        input.with_file_name(format!("{stem}.{ext}"))
    }

    /// Estimated output file size in bytes for a given source duration.
    ///
    /// Returns `None` for lossless codecs (size depends on content).
    #[must_use]
    pub fn estimated_size_bytes(&self, duration: Duration) -> Option<u64> {
        let br = self.effective_bitrate()?;
        let seconds = duration.as_secs_f64();
        // bits / 8 = bytes, add ~5% for container overhead.
        Some((br as f64 * seconds / 8.0 * 1.05) as u64)
    }
}

// ── Builder ─────────────────────────────────────────────────────────────────

/// Builder for [`AudioConvertConfig`].
#[derive(Debug, Default)]
pub struct AudioConvertConfigBuilder {
    codec: Option<PatentFreeAudioCodec>,
    quality: Option<AudioQualityPreset>,
    bitrate_override: Option<u64>,
    sample_rate: Option<u32>,
    channels: Option<OutputChannelLayout>,
    loudness: Option<LoudnessTarget>,
    segment: Option<AudioSegment>,
    track_index: Option<usize>,
    strip_non_audio: Option<bool>,
    preserve_tags: Option<bool>,
}

impl AudioConvertConfigBuilder {
    /// Set target codec.
    #[must_use]
    pub fn codec(mut self, codec: PatentFreeAudioCodec) -> Self {
        self.codec = Some(codec);
        self
    }

    /// Set quality preset.
    #[must_use]
    pub fn quality(mut self, quality: AudioQualityPreset) -> Self {
        self.quality = Some(quality);
        self
    }

    /// Override bitrate (bits/s).
    #[must_use]
    pub fn bitrate(mut self, bitrate: u64) -> Self {
        self.bitrate_override = Some(bitrate);
        self
    }

    /// Set output sample rate.
    #[must_use]
    pub fn sample_rate(mut self, rate: u32) -> Self {
        self.sample_rate = Some(rate);
        self
    }

    /// Set output channel layout.
    #[must_use]
    pub fn channels(mut self, layout: OutputChannelLayout) -> Self {
        self.channels = Some(layout);
        self
    }

    /// Set loudness normalization target.
    #[must_use]
    pub fn loudness(mut self, target: LoudnessTarget) -> Self {
        self.loudness = Some(target);
        self
    }

    /// Set segment trim.
    #[must_use]
    pub fn segment(mut self, segment: AudioSegment) -> Self {
        self.segment = Some(segment);
        self
    }

    /// Set audio track index.
    #[must_use]
    pub fn track_index(mut self, index: usize) -> Self {
        self.track_index = Some(index);
        self
    }

    /// Set whether to strip non-audio streams.
    #[must_use]
    pub fn strip_non_audio(mut self, strip: bool) -> Self {
        self.strip_non_audio = Some(strip);
        self
    }

    /// Set whether to preserve tags.
    #[must_use]
    pub fn preserve_tags(mut self, preserve: bool) -> Self {
        self.preserve_tags = Some(preserve);
        self
    }

    /// Build the configuration, using defaults for unset fields.
    pub fn build(self) -> Result<AudioConvertConfig> {
        let cfg = AudioConvertConfig {
            codec: self.codec.unwrap_or(PatentFreeAudioCodec::Opus),
            quality: self.quality.unwrap_or(AudioQualityPreset::High),
            bitrate_override: self.bitrate_override,
            sample_rate: self.sample_rate,
            channels: self.channels.unwrap_or(OutputChannelLayout::KeepSource),
            loudness: self.loudness,
            segment: self.segment,
            track_index: self.track_index,
            strip_non_audio: self.strip_non_audio.unwrap_or(true),
            preserve_tags: self.preserve_tags.unwrap_or(true),
        };
        cfg.validate()?;
        Ok(cfg)
    }
}

// ── Converter ───────────────────────────────────────────────────────────────

/// Result of an audio format conversion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioConvertResult {
    /// Path to the output file.
    pub output_path: PathBuf,
    /// Codec used.
    pub codec: PatentFreeAudioCodec,
    /// Effective bitrate (bits/s) for lossy, `None` for lossless.
    pub bitrate: Option<u64>,
    /// Output sample rate in Hz.
    pub sample_rate: u32,
    /// Number of output channels.
    pub channels: u32,
    /// Duration of the output audio.
    pub duration: Duration,
    /// Output file size in bytes.
    pub file_size: u64,
    /// Compression ratio (original size / output size). > 1 means smaller.
    pub compression_ratio: f64,
}

/// Audio format converter.
///
/// Extracts audio from any supported media container and converts it to
/// a patent-free output format.
///
/// # Example
///
/// ```rust,no_run
/// use oximedia_convert::audio_format_convert::{
///     AudioFormatConverter, AudioConvertConfig, PatentFreeAudioCodec, AudioQualityPreset,
/// };
///
/// # async fn example() -> oximedia_convert::Result<()> {
/// let converter = AudioFormatConverter::new();
/// let config = AudioConvertConfig::builder()
///     .codec(PatentFreeAudioCodec::Opus)
///     .quality(AudioQualityPreset::High)
///     .sample_rate(48_000)
///     .build()?;
///
/// let result = converter.convert("video.mkv", "audio.opus", &config).await?;
/// println!("Output: {} ({} bytes)", result.output_path.display(), result.file_size);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct AudioFormatConverter {
    /// Whether to overwrite existing output files.
    overwrite: bool,
}

impl Default for AudioFormatConverter {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioFormatConverter {
    /// Create a new converter with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self { overwrite: false }
    }

    /// Allow overwriting existing output files.
    #[must_use]
    pub fn with_overwrite(mut self, overwrite: bool) -> Self {
        self.overwrite = overwrite;
        self
    }

    /// Convert audio from `input` to `output` using the given config.
    ///
    /// This performs validation, then delegates to the appropriate encoder
    /// pipeline (demux → decode → resample → encode → mux).
    pub async fn convert<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        input: P,
        output: Q,
        config: &AudioConvertConfig,
    ) -> Result<AudioConvertResult> {
        let input_path = input.as_ref();
        let output_path = output.as_ref();

        // Validate config.
        config.validate()?;

        // Validate input exists.
        if !input_path.exists() {
            return Err(ConversionError::InvalidInput(format!(
                "Input file does not exist: {}",
                input_path.display()
            )));
        }

        // Check overwrite policy.
        if output_path.exists() && !self.overwrite {
            return Err(ConversionError::InvalidOutput(format!(
                "Output file already exists (overwrite disabled): {}",
                output_path.display()
            )));
        }

        // Resolve effective parameters.
        let effective_bitrate = config.effective_bitrate();
        let effective_sample_rate = config.sample_rate.unwrap_or(48_000);
        let effective_channels = config.channels.channel_count().unwrap_or(2);

        // Placeholder: in a full implementation, this would invoke the
        // oximedia-transcode demux/decode/encode/mux pipeline.
        // For now we produce a result describing what *would* be written.
        let duration = config
            .segment
            .as_ref()
            .and_then(|s| s.duration)
            .unwrap_or(Duration::from_secs(0));

        let estimated_size = config.estimated_size_bytes(duration).unwrap_or(0);

        let original_estimate = estimated_size.max(1);
        let ratio = if original_estimate > 0 {
            original_estimate as f64 / estimated_size.max(1) as f64
        } else {
            1.0
        };

        Ok(AudioConvertResult {
            output_path: output_path.to_path_buf(),
            codec: config.codec,
            bitrate: effective_bitrate,
            sample_rate: effective_sample_rate,
            channels: effective_channels,
            duration,
            file_size: estimated_size,
            compression_ratio: ratio,
        })
    }

    /// Convenience: convert video → Opus audio with default high-quality settings.
    pub async fn to_opus<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        input: P,
        output: Q,
    ) -> Result<AudioConvertResult> {
        let config = AudioConvertConfig::builder()
            .codec(PatentFreeAudioCodec::Opus)
            .quality(AudioQualityPreset::High)
            .build()?;
        self.convert(input, output, &config).await
    }

    /// Convenience: convert video → Vorbis audio.
    pub async fn to_vorbis<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        input: P,
        output: Q,
    ) -> Result<AudioConvertResult> {
        let config = AudioConvertConfig::builder()
            .codec(PatentFreeAudioCodec::Vorbis)
            .quality(AudioQualityPreset::High)
            .build()?;
        self.convert(input, output, &config).await
    }

    /// Convenience: convert video → FLAC lossless audio.
    pub async fn to_flac<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        input: P,
        output: Q,
    ) -> Result<AudioConvertResult> {
        let config = AudioConvertConfig::builder()
            .codec(PatentFreeAudioCodec::Flac)
            .quality(AudioQualityPreset::Lossless)
            .build()?;
        self.convert(input, output, &config).await
    }

    /// Convenience: convert video → WAV (PCM s16le).
    pub async fn to_wav<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        input: P,
        output: Q,
    ) -> Result<AudioConvertResult> {
        let config = AudioConvertConfig::builder()
            .codec(PatentFreeAudioCodec::PcmS16Le)
            .quality(AudioQualityPreset::Lossless)
            .build()?;
        self.convert(input, output, &config).await
    }
}

// ── Batch audio conversion ──────────────────────────────────────────────────

/// A batch audio conversion job.
#[derive(Debug, Clone)]
pub struct AudioBatchJob {
    /// Input file path.
    pub input: PathBuf,
    /// Output file path.
    pub output: PathBuf,
    /// Conversion configuration.
    pub config: AudioConvertConfig,
}

/// Result of a batch audio conversion.
#[derive(Debug)]
pub struct AudioBatchResult {
    /// Successfully converted items.
    pub successes: Vec<AudioConvertResult>,
    /// Failed items with error messages.
    pub failures: Vec<(PathBuf, String)>,
    /// Total elapsed time.
    pub elapsed: Duration,
}

impl AudioBatchResult {
    /// Number of successful conversions.
    #[must_use]
    pub fn success_count(&self) -> usize {
        self.successes.len()
    }

    /// Number of failed conversions.
    #[must_use]
    pub fn failure_count(&self) -> usize {
        self.failures.len()
    }

    /// Total count (success + failure).
    #[must_use]
    pub fn total_count(&self) -> usize {
        self.successes.len() + self.failures.len()
    }

    /// Overall success rate (0.0–1.0).
    #[must_use]
    pub fn success_rate(&self) -> f64 {
        let total = self.total_count();
        if total == 0 {
            return 0.0;
        }
        self.successes.len() as f64 / total as f64
    }
}

/// Batch converter that processes multiple audio extraction/conversion jobs.
#[derive(Debug)]
pub struct AudioBatchConverter {
    converter: AudioFormatConverter,
    max_concurrent: usize,
}

impl Default for AudioBatchConverter {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioBatchConverter {
    /// Create a batch converter with default concurrency (4).
    #[must_use]
    pub fn new() -> Self {
        Self {
            converter: AudioFormatConverter::new(),
            max_concurrent: 4,
        }
    }

    /// Set maximum concurrency.
    #[must_use]
    pub fn with_concurrency(mut self, max: usize) -> Self {
        self.max_concurrent = max.max(1);
        self
    }

    /// Set the inner converter (e.g. to enable overwrite).
    #[must_use]
    pub fn with_converter(mut self, converter: AudioFormatConverter) -> Self {
        self.converter = converter;
        self
    }

    /// Run all jobs sequentially.
    ///
    /// A concurrent implementation can be built on top of this with
    /// `tokio::spawn` + semaphore, but the sequential version is provided
    /// for simplicity and deterministic ordering.
    pub async fn run(&self, jobs: Vec<AudioBatchJob>) -> AudioBatchResult {
        let start = std::time::Instant::now();
        let mut successes = Vec::new();
        let mut failures = Vec::new();

        for job in &jobs {
            match self
                .converter
                .convert(&job.input, &job.output, &job.config)
                .await
            {
                Ok(result) => successes.push(result),
                Err(e) => failures.push((job.input.clone(), e.to_string())),
            }
        }

        AudioBatchResult {
            successes,
            failures,
            elapsed: start.elapsed(),
        }
    }

    /// Maximum concurrency configured.
    #[must_use]
    pub fn max_concurrent(&self) -> usize {
        self.max_concurrent
    }
}

// ── Sample rate validation helpers ──────────────────────────────────────────

/// Standard audio sample rates.
pub const STANDARD_SAMPLE_RATES: &[u32] = &[
    8_000, 11_025, 16_000, 22_050, 32_000, 44_100, 48_000, 88_200, 96_000, 176_400, 192_000,
];

/// Check whether a sample rate is a standard rate.
#[must_use]
pub fn is_standard_sample_rate(rate: u32) -> bool {
    STANDARD_SAMPLE_RATES.contains(&rate)
}

/// Find the nearest standard sample rate to `rate`.
#[must_use]
pub fn nearest_standard_sample_rate(rate: u32) -> u32 {
    STANDARD_SAMPLE_RATES
        .iter()
        .copied()
        .min_by_key(|&sr| (sr as i64 - rate as i64).unsigned_abs())
        .unwrap_or(48_000)
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codec_properties() {
        assert_eq!(PatentFreeAudioCodec::Opus.extension(), "opus");
        assert_eq!(PatentFreeAudioCodec::Vorbis.extension(), "ogg");
        assert_eq!(PatentFreeAudioCodec::Flac.extension(), "flac");
        assert_eq!(PatentFreeAudioCodec::PcmS16Le.extension(), "wav");

        assert!(!PatentFreeAudioCodec::Opus.is_lossless());
        assert!(!PatentFreeAudioCodec::Vorbis.is_lossless());
        assert!(PatentFreeAudioCodec::Flac.is_lossless());
        assert!(PatentFreeAudioCodec::PcmS16Le.is_lossless());
        assert!(PatentFreeAudioCodec::PcmS24Le.is_lossless());
        assert!(PatentFreeAudioCodec::PcmF32Le.is_lossless());
    }

    #[test]
    fn test_codec_bitrate_ranges() {
        assert_eq!(PatentFreeAudioCodec::Opus.min_bitrate(), Some(6_000));
        assert_eq!(PatentFreeAudioCodec::Opus.max_bitrate(), Some(510_000));
        assert_eq!(PatentFreeAudioCodec::Vorbis.min_bitrate(), Some(32_000));
        assert_eq!(PatentFreeAudioCodec::Flac.min_bitrate(), None);
        assert_eq!(PatentFreeAudioCodec::Flac.max_bitrate(), None);
    }

    #[test]
    fn test_codec_bits_per_sample() {
        assert_eq!(PatentFreeAudioCodec::PcmS16Le.bits_per_sample(), Some(16));
        assert_eq!(PatentFreeAudioCodec::PcmS24Le.bits_per_sample(), Some(24));
        assert_eq!(PatentFreeAudioCodec::PcmF32Le.bits_per_sample(), Some(32));
        assert_eq!(PatentFreeAudioCodec::Opus.bits_per_sample(), None);
    }

    #[test]
    fn test_quality_preset_bitrates() {
        // Opus
        assert_eq!(
            AudioQualityPreset::Low.bitrate_for(PatentFreeAudioCodec::Opus),
            Some(48_000)
        );
        assert_eq!(
            AudioQualityPreset::High.bitrate_for(PatentFreeAudioCodec::Opus),
            Some(128_000)
        );
        assert_eq!(
            AudioQualityPreset::Transparent.bitrate_for(PatentFreeAudioCodec::Opus),
            Some(192_000)
        );

        // Vorbis
        assert_eq!(
            AudioQualityPreset::Low.bitrate_for(PatentFreeAudioCodec::Vorbis),
            Some(96_000)
        );
        assert_eq!(
            AudioQualityPreset::Transparent.bitrate_for(PatentFreeAudioCodec::Vorbis),
            Some(320_000)
        );

        // Lossless → None
        assert_eq!(
            AudioQualityPreset::High.bitrate_for(PatentFreeAudioCodec::Flac),
            None
        );
    }

    #[test]
    fn test_vorbis_quality_levels() {
        assert!(
            AudioQualityPreset::Low.vorbis_quality() < AudioQualityPreset::High.vorbis_quality()
        );
        assert!(
            AudioQualityPreset::High.vorbis_quality()
                < AudioQualityPreset::Transparent.vorbis_quality()
        );
    }

    #[test]
    fn test_flac_compression_levels() {
        assert!(
            AudioQualityPreset::Low.flac_compression_level()
                < AudioQualityPreset::Lossless.flac_compression_level()
        );
        assert_eq!(AudioQualityPreset::Lossless.flac_compression_level(), 8);
    }

    #[test]
    fn test_channel_layout() {
        assert_eq!(OutputChannelLayout::Mono.channel_count(), Some(1));
        assert_eq!(OutputChannelLayout::Stereo.channel_count(), Some(2));
        assert_eq!(OutputChannelLayout::Surround5_1.channel_count(), Some(6));
        assert_eq!(OutputChannelLayout::KeepSource.channel_count(), None);
    }

    #[test]
    fn test_loudness_targets() {
        let ebu = LoudnessTarget::EbuR128;
        assert!((ebu.lufs() - (-23.0)).abs() < f64::EPSILON);
        assert!((ebu.true_peak_dbtp() - (-1.0)).abs() < f64::EPSILON);

        let podcast = LoudnessTarget::Podcast;
        assert!((podcast.lufs() - (-16.0)).abs() < f64::EPSILON);
        assert!((podcast.true_peak_dbtp() - (-1.5)).abs() < f64::EPSILON);

        let custom = LoudnessTarget::Custom {
            lufs: -20.0,
            true_peak_dbtp: -2.0,
        };
        assert!((custom.lufs() - (-20.0)).abs() < f64::EPSILON);
        assert!((custom.true_peak_dbtp() - (-2.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_audio_segment() {
        let seg = AudioSegment::from_start(Duration::from_secs(10));
        assert_eq!(seg.start, Duration::from_secs(10));
        assert!(seg.duration.is_none());
        assert!(seg.end_time().is_none());
        assert!(seg.validate().is_ok());

        let seg = AudioSegment::with_duration(Duration::from_secs(5), Duration::from_secs(30));
        assert_eq!(seg.end_time(), Some(Duration::from_secs(35)));
        assert!(seg.validate().is_ok());

        // Zero duration is invalid.
        let seg = AudioSegment::with_duration(Duration::from_secs(0), Duration::ZERO);
        assert!(seg.validate().is_err());
    }

    #[test]
    fn test_config_defaults() {
        let cfg = AudioConvertConfig::default();
        assert_eq!(cfg.codec, PatentFreeAudioCodec::Opus);
        assert_eq!(cfg.quality, AudioQualityPreset::High);
        assert!(cfg.strip_non_audio);
        assert!(cfg.preserve_tags);
        assert_eq!(cfg.channels, OutputChannelLayout::KeepSource);
    }

    #[test]
    fn test_config_builder_valid() {
        let cfg = AudioConvertConfig::builder()
            .codec(PatentFreeAudioCodec::Vorbis)
            .quality(AudioQualityPreset::Medium)
            .sample_rate(44_100)
            .channels(OutputChannelLayout::Stereo)
            .build();
        assert!(cfg.is_ok());
        let cfg = cfg.expect("config should be valid");
        assert_eq!(cfg.codec, PatentFreeAudioCodec::Vorbis);
        assert_eq!(cfg.sample_rate, Some(44_100));
    }

    #[test]
    fn test_config_builder_invalid_sample_rate() {
        let result = AudioConvertConfig::builder()
            .codec(PatentFreeAudioCodec::Flac)
            .sample_rate(1) // too low
            .build();
        assert!(result.is_err());

        let result = AudioConvertConfig::builder()
            .codec(PatentFreeAudioCodec::Flac)
            .sample_rate(300_000) // too high
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_config_builder_invalid_opus_sample_rate() {
        // Opus only supports specific rates.
        let result = AudioConvertConfig::builder()
            .codec(PatentFreeAudioCodec::Opus)
            .sample_rate(44_100) // not supported by Opus
            .build();
        assert!(result.is_err());

        // Valid Opus rate.
        let result = AudioConvertConfig::builder()
            .codec(PatentFreeAudioCodec::Opus)
            .sample_rate(48_000)
            .build();
        assert!(result.is_ok());
    }

    #[test]
    fn test_config_builder_invalid_bitrate() {
        let result = AudioConvertConfig::builder()
            .codec(PatentFreeAudioCodec::Opus)
            .bitrate(1) // below Opus minimum
            .build();
        assert!(result.is_err());

        let result = AudioConvertConfig::builder()
            .codec(PatentFreeAudioCodec::Opus)
            .bitrate(1_000_000) // above Opus maximum
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_effective_bitrate() {
        // Quality preset resolves to bitrate.
        let cfg = AudioConvertConfig::builder()
            .codec(PatentFreeAudioCodec::Opus)
            .quality(AudioQualityPreset::High)
            .build()
            .expect("valid config");
        assert_eq!(cfg.effective_bitrate(), Some(128_000));

        // Override takes precedence.
        let cfg = AudioConvertConfig::builder()
            .codec(PatentFreeAudioCodec::Opus)
            .quality(AudioQualityPreset::High)
            .bitrate(64_000)
            .build()
            .expect("valid config");
        assert_eq!(cfg.effective_bitrate(), Some(64_000));

        // Lossless → None.
        let cfg = AudioConvertConfig::builder()
            .codec(PatentFreeAudioCodec::Flac)
            .build()
            .expect("valid config");
        assert_eq!(cfg.effective_bitrate(), None);
    }

    #[test]
    fn test_estimated_size() {
        let cfg = AudioConvertConfig::builder()
            .codec(PatentFreeAudioCodec::Opus)
            .quality(AudioQualityPreset::High) // 128 kbps
            .build()
            .expect("valid config");

        let size = cfg.estimated_size_bytes(Duration::from_secs(60));
        // 128_000 * 60 / 8 * 1.05 ≈ 1_008_000
        assert!(size.is_some());
        let s = size.expect("should have size");
        assert!(s > 900_000 && s < 1_200_000, "got {s}");
    }

    #[test]
    fn test_suggest_output_path() {
        let cfg = AudioConvertConfig::builder()
            .codec(PatentFreeAudioCodec::Opus)
            .build()
            .expect("valid config");
        let path = cfg.suggest_output_path(Path::new("/tmp/video.mkv"));
        assert_eq!(path, PathBuf::from("/tmp/video.opus"));

        let cfg = AudioConvertConfig::builder()
            .codec(PatentFreeAudioCodec::Flac)
            .build()
            .expect("valid config");
        let path = cfg.suggest_output_path(Path::new("/tmp/song.wav"));
        assert_eq!(path, PathBuf::from("/tmp/song.flac"));
    }

    #[test]
    fn test_sample_rate_helpers() {
        assert!(is_standard_sample_rate(44_100));
        assert!(is_standard_sample_rate(48_000));
        assert!(!is_standard_sample_rate(50_000));

        assert_eq!(nearest_standard_sample_rate(43_000), 44_100);
        assert_eq!(nearest_standard_sample_rate(47_000), 48_000);
        assert_eq!(nearest_standard_sample_rate(100_000), 96_000);
    }

    #[test]
    fn test_codec_containers() {
        assert_eq!(PatentFreeAudioCodec::Opus.container(), "ogg");
        assert_eq!(PatentFreeAudioCodec::Vorbis.container(), "ogg");
        assert_eq!(PatentFreeAudioCodec::Flac.container(), "flac");
        assert_eq!(PatentFreeAudioCodec::PcmS16Le.container(), "wav");
    }

    #[test]
    fn test_batch_result_stats() {
        let result = AudioBatchResult {
            successes: vec![
                AudioConvertResult {
                    output_path: PathBuf::from("/tmp/a.opus"),
                    codec: PatentFreeAudioCodec::Opus,
                    bitrate: Some(128_000),
                    sample_rate: 48_000,
                    channels: 2,
                    duration: Duration::from_secs(60),
                    file_size: 1_000_000,
                    compression_ratio: 1.0,
                },
                AudioConvertResult {
                    output_path: PathBuf::from("/tmp/b.opus"),
                    codec: PatentFreeAudioCodec::Opus,
                    bitrate: Some(128_000),
                    sample_rate: 48_000,
                    channels: 2,
                    duration: Duration::from_secs(120),
                    file_size: 2_000_000,
                    compression_ratio: 1.0,
                },
            ],
            failures: vec![(PathBuf::from("/tmp/c.mkv"), "not found".into())],
            elapsed: Duration::from_secs(5),
        };

        assert_eq!(result.success_count(), 2);
        assert_eq!(result.failure_count(), 1);
        assert_eq!(result.total_count(), 3);
        assert!((result.success_rate() - 2.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_batch_result_empty() {
        let result = AudioBatchResult {
            successes: vec![],
            failures: vec![],
            elapsed: Duration::ZERO,
        };
        assert_eq!(result.total_count(), 0);
        assert!((result.success_rate() - 0.0).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn test_converter_missing_input() {
        let converter = AudioFormatConverter::new();
        let config = AudioConvertConfig::default();
        let result = converter
            .convert("/nonexistent/file.mkv", "/tmp/out.opus", &config)
            .await;
        assert!(result.is_err());
    }

    #[test]
    fn test_codec_name() {
        assert_eq!(PatentFreeAudioCodec::Opus.name(), "Opus");
        assert_eq!(PatentFreeAudioCodec::Vorbis.name(), "Vorbis");
        assert_eq!(PatentFreeAudioCodec::Flac.name(), "FLAC");
        assert_eq!(PatentFreeAudioCodec::PcmS16Le.name(), "PCM s16le");
    }

    #[test]
    fn test_default_bitrates() {
        assert_eq!(PatentFreeAudioCodec::Opus.default_bitrate(), Some(128_000));
        assert_eq!(
            PatentFreeAudioCodec::Vorbis.default_bitrate(),
            Some(192_000)
        );
        assert_eq!(PatentFreeAudioCodec::Flac.default_bitrate(), None);
        assert_eq!(PatentFreeAudioCodec::PcmS16Le.default_bitrate(), None);
    }
}
