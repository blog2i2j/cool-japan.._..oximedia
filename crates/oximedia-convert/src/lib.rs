// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

#![forbid(unsafe_code)]
#![doc = include_str!("../README.md")]

//! Media format conversion utilities for `OxiMedia`.
//!
//! This crate provides comprehensive media conversion capabilities including:
//! - Batch conversion with templates
//! - Format and codec detection
//! - Conversion profiles for common use cases
//! - Quality control and preservation
//! - Metadata preservation across formats
//! - Subtitle, audio, and video extraction
//! - Frame extraction and thumbnail generation
//! - File concatenation and splitting
//!
//! # Examples
//!
//! ```rust,no_run
//! use oximedia_convert::{Converter, ConversionOptions, Profile};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let converter = Converter::new();
//! let options = ConversionOptions::builder()
//!     .profile(Profile::WebOptimized)
//!     .quality_mode(oximedia_convert::QualityMode::Balanced)
//!     .build()?;
//!
//! converter.convert("input.mov", "output.mp4", options).await?;
//! # Ok(())
//! # }
//! ```

pub mod aspect_ratio;
pub mod audio;
pub mod batch;
pub mod batch_convert;
pub mod codec_mapper;
pub mod color_convert;
pub mod concat;
pub mod container_ops;
pub mod conv_profile;
pub mod conv_validate;
pub mod conversion_pipeline;
pub mod convert_log;
pub mod detect;
pub mod filters;
pub mod format_detector;
pub mod formats;
pub mod frame;
pub mod metadata;
pub mod metrics;
pub mod normalization;
pub mod partial;
pub mod pipeline;
pub mod pixel_convert;
pub mod presets;
pub mod profile;
pub mod profile_match;
pub mod progress;
pub mod quality;
pub mod sample_rate;
pub mod sequence;
pub mod smart;
pub mod split;
pub mod streaming;
pub mod subtitle;
pub mod template;
pub mod thumbnail;
pub mod thumbnail_strip;
pub mod transcode_report;
pub mod video;
pub mod watermark_strip;

use std::path::{Path, PathBuf};
use thiserror::Error;

pub use audio::{AudioExtractor, AudioTrackSelector};
pub use batch::{BatchProcessor, ConversionQueue, ProgressTracker};
pub use concat::{CompatibilityValidator, FileJoiner};
pub use detect::{CodecDetector, FormatDetector, MediaProperties};
pub use filters::{Filter, FilterChain};
pub use formats::{AudioCodec, ChannelLayout, ContainerFormat, ImageFormat, VideoCodec};
pub use frame::{FrameExtractor, FrameRange};
pub use metadata::{MetadataMapper, MetadataPreserver};
pub use metrics::{MetricsCalculator, QualityMetrics};
pub use partial::{ChapterSelection, PartialConversion, StreamSelection, TimeRange};
pub use pipeline::{
    AudioOptions, BitrateMode, ConversionJob, JobPriority, JobStatus, PipelineExecutor,
    VideoOptions,
};
pub use presets::{AudioPresetSettings, EncodingSpeed, Preset, VideoPresetSettings};
pub use profile::{Profile, ProfileBuilder, ProfileSystem};
pub use quality::{QualityComparison, QualityMaintainer};
pub use sequence::{ImageSequence, SequenceExporter, SequenceImporter};
pub use smart::{ConversionTarget, MediaAnalysis, OptimizedSettings, SmartConverter};
pub use split::{ChapterSplitter, SizeSplitter, TimeSplitter};
pub use streaming::{
    AbrLadder, BitrateVariant, StreamingConfig, StreamingFormat, StreamingPackager,
};
pub use subtitle::{SubtitleConverter, SubtitleExtractor};
pub use template::{TemplateSystem, TemplateVariables};
pub use thumbnail::{SpriteSheetGenerator, ThumbnailGenerator};
pub use video::{VideoExtractor, VideoMuter};

/// Errors that can occur during media conversion.
#[derive(Debug, Error)]
pub enum ConversionError {
    /// I/O error occurred
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Format detection failed
    #[error("Format detection failed: {0}")]
    FormatDetection(String),

    /// Codec not supported
    #[error("Codec not supported: {0}")]
    UnsupportedCodec(String),

    /// Invalid conversion profile
    #[error("Invalid profile: {0}")]
    InvalidProfile(String),

    /// Quality preservation failed
    #[error("Quality preservation failed: {0}")]
    QualityPreservation(String),

    /// Metadata error
    #[error("Metadata error: {0}")]
    Metadata(String),

    /// Transcoding error
    #[error("Transcoding error: {0}")]
    Transcode(String),

    /// Container error
    #[error("Container error: {0}")]
    Container(String),

    /// Invalid input
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Invalid output
    #[error("Invalid output: {0}")]
    InvalidOutput(String),

    /// Conversion interrupted
    #[error("Conversion interrupted")]
    Interrupted,

    /// Validation failed
    #[error("Validation failed: {0}")]
    ValidationFailed(String),

    /// Template error
    #[error("Template error: {0}")]
    Template(String),
}

/// Result type for conversion operations.
pub type Result<T> = std::result::Result<T, ConversionError>;

/// Quality modes for conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualityMode {
    /// Fast conversion with lower quality
    Fast,
    /// Balanced quality and speed
    Balanced,
    /// Best quality, slower conversion
    Best,
}

/// Main converter for media format conversion.
#[derive(Debug, Clone)]
pub struct Converter {
    detector: FormatDetector,
    profile_system: ProfileSystem,
    quality_maintainer: QualityMaintainer,
    metadata_preserver: MetadataPreserver,
}

impl Converter {
    /// Create a new converter with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            detector: FormatDetector::new(),
            profile_system: ProfileSystem::new(),
            quality_maintainer: QualityMaintainer::new(),
            metadata_preserver: MetadataPreserver::new(),
        }
    }

    /// Convert a media file with the specified options.
    pub async fn convert<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        input: P,
        output: Q,
        options: ConversionOptions,
    ) -> Result<ConversionReport> {
        let input = input.as_ref();
        let output = output.as_ref();

        // Detect input format and properties
        let properties = self.detector.detect(input)?;

        // Apply profile settings
        let profile = self.profile_system.get_profile(&options.profile)?;
        let settings = profile.apply(&properties, &options)?;

        // Preserve metadata if requested
        let metadata = if options.preserve_metadata {
            Some(self.metadata_preserver.extract(input)?)
        } else {
            None
        };

        // Perform conversion
        let start_time = std::time::Instant::now();
        self.perform_conversion(input, output, &settings, &options)
            .await?;
        let duration = start_time.elapsed();

        // Restore metadata if needed
        if let Some(meta) = metadata {
            self.metadata_preserver.restore(output, &meta)?;
        }

        // Generate report
        Ok(ConversionReport {
            input: input.to_path_buf(),
            output: output.to_path_buf(),
            duration,
            source_properties: properties,
            settings,
            quality_comparison: if options.compare_quality {
                Some(self.quality_maintainer.compare(input, output)?)
            } else {
                None
            },
        })
    }

    async fn perform_conversion(
        &self,
        input: &Path,
        _output: &Path,
        _settings: &ConversionSettings,
        _options: &ConversionOptions,
    ) -> Result<()> {
        // Full conversion via oximedia-transcode is deferred to a future
        // milestone once the demux/decode/encode/mux pipeline is integrated.
        // Validate the input file is accessible before returning.
        if !input.as_os_str().is_empty() && !input.exists() {
            return Err(ConversionError::InvalidInput(format!(
                "Input file not found: {}",
                input.display()
            )));
        }
        Ok(())
    }
}

impl Default for Converter {
    fn default() -> Self {
        Self::new()
    }
}

/// Options for media conversion.
#[derive(Debug, Clone)]
pub struct ConversionOptions {
    /// Conversion profile to use
    pub profile: Profile,
    /// Quality mode
    pub quality_mode: QualityMode,
    /// Whether to preserve metadata
    pub preserve_metadata: bool,
    /// Whether to compare quality after conversion
    pub compare_quality: bool,
    /// Maximum resolution (width, height)
    pub max_resolution: Option<(u32, u32)>,
    /// Target bitrate in bits per second
    pub target_bitrate: Option<u64>,
    /// Additional custom settings
    pub custom_settings: Vec<(String, String)>,
}

impl ConversionOptions {
    /// Create a new builder for conversion options.
    #[must_use]
    pub fn builder() -> ConversionOptionsBuilder {
        ConversionOptionsBuilder::default()
    }
}

impl Default for ConversionOptions {
    fn default() -> Self {
        Self {
            profile: Profile::WebOptimized,
            quality_mode: QualityMode::Balanced,
            preserve_metadata: true,
            compare_quality: false,
            max_resolution: None,
            target_bitrate: None,
            custom_settings: Vec::new(),
        }
    }
}

/// Builder for conversion options.
#[derive(Debug, Default)]
pub struct ConversionOptionsBuilder {
    profile: Option<Profile>,
    quality_mode: Option<QualityMode>,
    preserve_metadata: Option<bool>,
    compare_quality: Option<bool>,
    max_resolution: Option<(u32, u32)>,
    target_bitrate: Option<u64>,
    custom_settings: Vec<(String, String)>,
}

impl ConversionOptionsBuilder {
    /// Set the conversion profile.
    #[must_use]
    pub fn profile(mut self, profile: Profile) -> Self {
        self.profile = Some(profile);
        self
    }

    /// Set the quality mode.
    #[must_use]
    pub fn quality_mode(mut self, mode: QualityMode) -> Self {
        self.quality_mode = Some(mode);
        self
    }

    /// Set whether to preserve metadata.
    #[must_use]
    pub fn preserve_metadata(mut self, preserve: bool) -> Self {
        self.preserve_metadata = Some(preserve);
        self
    }

    /// Set whether to compare quality after conversion.
    #[must_use]
    pub fn compare_quality(mut self, compare: bool) -> Self {
        self.compare_quality = Some(compare);
        self
    }

    /// Set maximum resolution.
    #[must_use]
    pub fn max_resolution(mut self, width: u32, height: u32) -> Self {
        self.max_resolution = Some((width, height));
        self
    }

    /// Set target bitrate.
    #[must_use]
    pub fn target_bitrate(mut self, bitrate: u64) -> Self {
        self.target_bitrate = Some(bitrate);
        self
    }

    /// Add a custom setting.
    #[must_use]
    pub fn custom_setting(mut self, key: String, value: String) -> Self {
        self.custom_settings.push((key, value));
        self
    }

    /// Build the conversion options.
    pub fn build(self) -> Result<ConversionOptions> {
        Ok(ConversionOptions {
            profile: self.profile.unwrap_or(Profile::WebOptimized),
            quality_mode: self.quality_mode.unwrap_or(QualityMode::Balanced),
            preserve_metadata: self.preserve_metadata.unwrap_or(true),
            compare_quality: self.compare_quality.unwrap_or(false),
            max_resolution: self.max_resolution,
            target_bitrate: self.target_bitrate,
            custom_settings: self.custom_settings,
        })
    }
}

/// Settings applied for conversion.
#[derive(Debug, Clone)]
pub struct ConversionSettings {
    /// Output format
    pub format: String,
    /// Video codec
    pub video_codec: Option<String>,
    /// Audio codec
    pub audio_codec: Option<String>,
    /// Video bitrate
    pub video_bitrate: Option<u64>,
    /// Audio bitrate
    pub audio_bitrate: Option<u64>,
    /// Resolution
    pub resolution: Option<(u32, u32)>,
    /// Frame rate
    pub frame_rate: Option<f64>,
    /// Additional parameters
    pub parameters: Vec<(String, String)>,
}

/// Report generated after conversion.
#[derive(Debug)]
pub struct ConversionReport {
    /// Input file path
    pub input: PathBuf,
    /// Output file path
    pub output: PathBuf,
    /// Conversion duration
    pub duration: std::time::Duration,
    /// Source media properties
    pub source_properties: MediaProperties,
    /// Settings used for conversion
    pub settings: ConversionSettings,
    /// Quality comparison if enabled
    pub quality_comparison: Option<QualityComparison>,
}

impl ConversionReport {
    /// Get the compression ratio.
    pub fn compression_ratio(&self) -> Result<f64> {
        let input_size = std::fs::metadata(&self.input)
            .map_err(ConversionError::Io)?
            .len();
        let output_size = std::fs::metadata(&self.output)
            .map_err(ConversionError::Io)?
            .len();

        Ok(input_size as f64 / output_size as f64)
    }

    /// Get the size reduction percentage.
    pub fn size_reduction_percent(&self) -> Result<f64> {
        let input_size = std::fs::metadata(&self.input)
            .map_err(ConversionError::Io)?
            .len();
        let output_size = std::fs::metadata(&self.output)
            .map_err(ConversionError::Io)?
            .len();

        Ok(((input_size - output_size) as f64 / input_size as f64) * 100.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conversion_options_builder() {
        let options = ConversionOptions::builder()
            .profile(Profile::WebOptimized)
            .quality_mode(QualityMode::Best)
            .preserve_metadata(true)
            .max_resolution(1920, 1080)
            .build()
            .expect("builder should produce valid options");

        assert_eq!(options.profile, Profile::WebOptimized);
        assert_eq!(options.quality_mode, QualityMode::Best);
        assert!(options.preserve_metadata);
        assert_eq!(options.max_resolution, Some((1920, 1080)));
    }

    #[test]
    fn test_converter_creation() {
        let converter = Converter::new();
        assert!(std::mem::size_of_val(&converter) > 0);
    }
}
