//! Professional broadcast loudness normalization for `OxiMedia`.
//!
//! This crate provides comprehensive loudness normalization compliant with broadcast
//! and streaming standards, including EBU R128, ATSC A/85, and all major streaming platforms.
//!
//! # Supported Standards
//!
//! - **EBU R128** - European Broadcasting Union (-23 LUFS ±1 LU, -1 dBTP max)
//! - **ATSC A/85** - US broadcast standard (-24 LKFS ±2 dB, -2 dBTP max)
//! - **Streaming Platforms** - Spotify (-14 LUFS), YouTube (-14 LUFS), Apple Music (-16 LUFS), etc.
//! - **Netflix** - Streaming drama (-27 LUFS) and loud content (-24 LUFS)
//! - **ReplayGain** - Album and track gain (reference 89 dB SPL)
//!
//! # Features
//!
//! ## Processing Modes
//!
//! - **Two-pass Normalization** - Analyze first, then apply precise gain
//! - **One-pass Normalization** - Real-time with lookahead buffer
//! - **Linear Gain** - Simple gain adjustment to target loudness
//! - **Dynamic Normalization** - Dynamic range compression for consistent loudness
//! - **True Peak Limiting** - Brick-wall limiter preventing clipping
//!
//! ## Analysis
//!
//! - **Integrated Loudness** - ITU-R BS.1770-4 gated measurement
//! - **Loudness Range (LRA)** - Dynamic range quantification
//! - **True Peak Detection** - Inter-sample peak detection via 4x oversampling
//! - **Compliance Checking** - Verify against all broadcast standards
//!
//! ## Metadata
//!
//! - **ReplayGain Tags** - Track and album gain/peak
//! - **R128 Tags** - EBU R128 loudness metadata
//! - **iTunes Tags** - Sound Check metadata
//! - **Loudness Metadata** - Standard loudness descriptors
//!
//! # Example Usage
//!
//! ## Two-pass Normalization
//!
//! ```rust,no_run
//! use oximedia_normalize::{Normalizer, NormalizerConfig};
//! use oximedia_metering::Standard;
//!
//! // Configure for EBU R128 normalization
//! let config = NormalizerConfig::new(Standard::EbuR128, 48000.0, 2);
//!
//! let mut normalizer = Normalizer::new(config).expect("Failed to create normalizer");
//!
//! // Pass 1: Analyze
//! # let audio_samples: &[f32] = &[];
//! normalizer.analyze_f32(audio_samples);
//! let mut analysis = normalizer.get_analysis();
//! println!("Current loudness: {:.1} LUFS", analysis.integrated_lufs);
//! println!("Gain needed: {:.1} dB", analysis.recommended_gain_db);
//!
//! // Pass 2: Normalize
//! let mut output = vec![0.0f32; audio_samples.len()];
//! normalizer.process_f32(audio_samples, &mut output).expect("Processing failed");
//! ```
//!
//! ## Real-time Normalization
//!
//! ```rust,no_run
//! use oximedia_normalize::{RealtimeNormalizer, RealtimeConfig};
//! use oximedia_metering::Standard;
//!
//! let config = RealtimeConfig::new(Standard::Spotify, 48000.0, 2);
//!
//! let mut normalizer = RealtimeNormalizer::new(config).expect("Failed to create");
//!
//! // Process audio chunks with lookahead buffering
//! # let chunk: &[f32] = &[];
//! let mut output = vec![0.0f32; chunk.len()];
//! normalizer.process_chunk(chunk, &mut output).expect("Processing failed");
//! ```
//!
//! ## Batch Processing
//!
//! ```rust,no_run
//! use oximedia_normalize::{BatchProcessor, BatchConfig};
//! use oximedia_metering::Standard;
//! use std::path::Path;
//!
//! let config = BatchConfig::new(Standard::Spotify);
//!
//! let processor = BatchProcessor::new(config);
//!
//! // Process multiple files
//! # let input_dir = Path::new(".");
//! # let output_dir = Path::new(".");
//! processor.process_directory(input_dir, output_dir).expect("Processing failed");
//! ```
//!
//! # Implementation Details
//!
//! ## Loudness Normalization Algorithm
//!
//! 1. **Analysis**: Measure integrated loudness using ITU-R BS.1770-4 with gating
//! 2. **Gain Calculation**: Compute gain needed to reach target loudness
//! 3. **Limiting Check**: Verify that gain won't cause true peak clipping
//! 4. **Gain Application**: Apply calculated gain to all samples
//! 5. **True Peak Limiting**: Apply brick-wall limiter if enabled
//! 6. **DRC**: Apply dynamic range compression if enabled
//!
//! ## True Peak Limiting
//!
//! - Uses lookahead buffer to prevent clipping
//! - 4x oversampling for accurate peak detection
//! - Attack/release envelope shaping
//! - Zero latency option for non-critical applications
//!
//! ## Dynamic Range Control
//!
//! - Broadcast-style DRC with configurable threshold
//! - Attack/release times matched to loudness gates
//! - Preserves transients while controlling peaks
//! - Optional makeup gain

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::similar_names)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::unused_self)]
#![allow(clippy::match_same_arms)]
#![allow(clippy::no_effect_underscore_binding)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::unused_unit)]
#![allow(clippy::format_in_format_args)]
#![allow(clippy::manual_let_else)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::float_cmp)]
#![allow(dead_code)]

pub mod ab_comparison;
pub mod agc;
pub mod analyzer;
pub mod auto_gain;
pub mod batch;
pub mod broadcast_standard;
pub mod compliance_checker;
pub mod crossfade_norm;
pub mod dc_offset;
pub mod dialogue_norm;
pub mod drc;
pub mod dynamic_range;
pub mod ebu_r128;
pub mod fade_normalization;
pub mod format_detect;
pub mod format_loudness;
pub mod gain_schedule;
pub mod limiter;
pub mod limiter_chain;
pub mod loudness_gate;
pub mod loudness_history;
pub mod loudness_target;
pub mod metadata;
pub mod metering_bridge;
pub mod multi_channel_loud;
pub mod multipass;
pub mod noise_profile;
pub mod normalize_report;
pub mod parallel_channels;
pub mod peak_limit;
pub mod phase_correction;
pub mod processor;
pub mod realtime;
pub mod replaygain;
pub mod sidechain;
pub mod spectral_balance;
pub mod stem_loudness;
pub mod stereo_width;
pub mod surround_norm;
pub mod target_loudness;
pub mod targets;
pub mod true_peak_limiter;
pub mod voice_activity;

use oximedia_metering::{LoudnessMetrics, MeteringError, Standard};
use thiserror::Error;

pub use analyzer::{AnalysisResult, LoudnessAnalyzer};
pub use batch::{BatchConfig, BatchProcessor, BatchResult};
pub use drc::{DrcConfig, DynamicRangeCompressor};
pub use limiter::{LimiterConfig, TruePeakLimiter};
pub use metadata::{LoudnessMetadata, MetadataWriter};
pub use multipass::{MultiPassConfig, MultiPassProcessor};
pub use processor::{NormalizationProcessor, ProcessorConfig};
pub use realtime::{RealtimeConfig, RealtimeNormalizer};
pub use replaygain::{ReplayGainCalculator, ReplayGainValues};
pub use targets::{NormalizationTarget, TargetPreset};

/// Normalization error types.
#[derive(Error, Debug)]
pub enum NormalizeError {
    /// Metering error.
    #[error("Metering error: {0}")]
    MeteringError(#[from] MeteringError),

    /// Invalid configuration.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Insufficient data for normalization.
    #[error("Insufficient data: {0}")]
    InsufficientData(String),

    /// Processing error.
    #[error("Processing error: {0}")]
    ProcessingError(String),

    /// I/O error.
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// Metadata error.
    #[error("Metadata error: {0}")]
    MetadataError(String),

    /// Analysis not complete.
    #[error("Analysis not complete: {0}")]
    AnalysisNotComplete(String),

    /// Gain would cause clipping.
    #[error("Gain would cause clipping: peak would be {0:.2} dBTP (max: {1:.2} dBTP)")]
    WouldClip(f64, f64),
}

/// Normalization result type.
pub type NormalizeResult<T> = std::result::Result<T, NormalizeError>;

/// Processing mode for normalization.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ProcessingMode {
    /// Analyze only - no modification.
    AnalyzeOnly,

    /// Two-pass: analyze first, then normalize.
    TwoPass,

    /// One-pass: real-time with lookahead buffer.
    OnePass,

    /// Normalize with linear gain only.
    LinearGain,

    /// Normalize with dynamic range compression.
    NormalizeDrc,

    /// Normalize with true peak limiting.
    NormalizeLimiter,

    /// Full processing: normalize + DRC + limiter.
    Full,
}

/// Normalizer configuration.
#[derive(Clone, Debug)]
#[allow(clippy::struct_excessive_bools)]
pub struct NormalizerConfig {
    /// Target loudness standard.
    pub standard: Standard,

    /// Sample rate in Hz.
    pub sample_rate: f64,

    /// Number of channels.
    pub channels: usize,

    /// Processing mode.
    pub processing_mode: ProcessingMode,

    /// Enable true peak limiter.
    pub enable_limiter: bool,

    /// Enable dynamic range compression.
    pub enable_drc: bool,

    /// Lookahead time in milliseconds.
    pub lookahead_ms: f64,

    /// Maximum allowed gain in dB (safety limit).
    pub max_gain_db: f64,

    /// Preserve loudness range when possible.
    pub preserve_lra: bool,

    /// Write metadata tags.
    pub write_metadata: bool,
}

impl NormalizerConfig {
    /// Create a new normalizer configuration.
    pub fn new(standard: Standard, sample_rate: f64, channels: usize) -> Self {
        Self {
            standard,
            sample_rate,
            channels,
            processing_mode: ProcessingMode::TwoPass,
            enable_limiter: true,
            enable_drc: false,
            lookahead_ms: 5.0,
            max_gain_db: 20.0,
            preserve_lra: true,
            write_metadata: false,
        }
    }

    /// Create a minimal configuration (linear gain only).
    pub fn minimal(standard: Standard, sample_rate: f64, channels: usize) -> Self {
        Self {
            standard,
            sample_rate,
            channels,
            processing_mode: ProcessingMode::LinearGain,
            enable_limiter: false,
            enable_drc: false,
            lookahead_ms: 0.0,
            max_gain_db: 20.0,
            preserve_lra: false,
            write_metadata: false,
        }
    }

    /// Create a broadcast configuration (full processing).
    pub fn broadcast(standard: Standard, sample_rate: f64, channels: usize) -> Self {
        Self {
            standard,
            sample_rate,
            channels,
            processing_mode: ProcessingMode::Full,
            enable_limiter: true,
            enable_drc: true,
            lookahead_ms: 10.0,
            max_gain_db: 15.0,
            preserve_lra: true,
            write_metadata: true,
        }
    }

    /// Validate the configuration.
    pub fn validate(&self) -> NormalizeResult<()> {
        if self.sample_rate < 8000.0 || self.sample_rate > 192_000.0 {
            return Err(NormalizeError::InvalidConfig(format!(
                "Sample rate {} Hz is out of valid range (8000-192000 Hz)",
                self.sample_rate
            )));
        }

        if self.channels == 0 || self.channels > 16 {
            return Err(NormalizeError::InvalidConfig(format!(
                "Channel count {} is out of valid range (1-16)",
                self.channels
            )));
        }

        if self.lookahead_ms < 0.0 || self.lookahead_ms > 100.0 {
            return Err(NormalizeError::InvalidConfig(format!(
                "Lookahead {} ms is out of valid range (0-100 ms)",
                self.lookahead_ms
            )));
        }

        if self.max_gain_db <= 0.0 || self.max_gain_db > 60.0 {
            return Err(NormalizeError::InvalidConfig(format!(
                "Max gain {} dB is out of valid range (0-60 dB)",
                self.max_gain_db
            )));
        }

        Ok(())
    }
}

/// Main loudness normalizer.
///
/// This is the primary interface for loudness normalization. It combines analysis
/// and processing into a single high-level API.
pub struct Normalizer {
    config: NormalizerConfig,
    analyzer: LoudnessAnalyzer,
    processor: NormalizationProcessor,
    analysis_complete: bool,
}

impl Normalizer {
    /// Create a new normalizer.
    pub fn new(config: NormalizerConfig) -> NormalizeResult<Self> {
        config.validate()?;

        let analyzer = LoudnessAnalyzer::new(config.standard, config.sample_rate, config.channels)?;

        let processor_config = ProcessorConfig {
            sample_rate: config.sample_rate,
            channels: config.channels,
            enable_limiter: config.enable_limiter,
            enable_drc: config.enable_drc,
            lookahead_ms: config.lookahead_ms,
        };

        let processor = NormalizationProcessor::new(processor_config)?;

        Ok(Self {
            config,
            analyzer,
            processor,
            analysis_complete: false,
        })
    }

    /// Analyze audio without modifying it (pass 1 of two-pass).
    pub fn analyze_f32(&mut self, samples: &[f32]) {
        self.analyzer.process_f32(samples);
        self.analysis_complete = true;
    }

    /// Analyze audio without modifying it (pass 1 of two-pass).
    pub fn analyze_f64(&mut self, samples: &[f64]) {
        self.analyzer.process_f64(samples);
        self.analysis_complete = true;
    }

    /// Get the analysis result.
    pub fn get_analysis(&mut self) -> AnalysisResult {
        self.analyzer.result()
    }

    /// Process and normalize audio (pass 2 of two-pass or one-pass).
    pub fn process_f32(&mut self, input: &[f32], output: &mut [f32]) -> NormalizeResult<()> {
        if output.len() != input.len() {
            return Err(NormalizeError::ProcessingError(
                "Output buffer must be same size as input".to_string(),
            ));
        }

        let gain_db = if self.analysis_complete {
            self.analyzer.result().recommended_gain_db
        } else if matches!(self.config.processing_mode, ProcessingMode::OnePass) {
            // For one-pass, analyze on the fly
            self.analyzer.process_f32(input);
            self.analyzer.result().recommended_gain_db
        } else {
            return Err(NormalizeError::AnalysisNotComplete(
                "Must call analyze_f32() before process_f32() in two-pass mode".to_string(),
            ));
        };

        // Clamp gain to max allowed
        let gain_db = gain_db.clamp(-60.0, self.config.max_gain_db);

        self.processor.process_f32(input, output, gain_db)?;

        Ok(())
    }

    /// Process and normalize audio (pass 2 of two-pass or one-pass).
    pub fn process_f64(&mut self, input: &[f64], output: &mut [f64]) -> NormalizeResult<()> {
        if output.len() != input.len() {
            return Err(NormalizeError::ProcessingError(
                "Output buffer must be same size as input".to_string(),
            ));
        }

        let gain_db = if self.analysis_complete {
            self.analyzer.result().recommended_gain_db
        } else if matches!(self.config.processing_mode, ProcessingMode::OnePass) {
            // For one-pass, analyze on the fly
            self.analyzer.process_f64(input);
            self.analyzer.result().recommended_gain_db
        } else {
            return Err(NormalizeError::AnalysisNotComplete(
                "Must call analyze_f64() before process_f64() in two-pass mode".to_string(),
            ));
        };

        // Clamp gain to max allowed
        let gain_db = gain_db.clamp(-60.0, self.config.max_gain_db);

        self.processor.process_f64(input, output, gain_db)?;

        Ok(())
    }

    /// Get current loudness metrics.
    pub fn metrics(&mut self) -> LoudnessMetrics {
        self.analyzer.metrics()
    }

    /// Check if analysis is complete.
    pub fn is_analysis_complete(&self) -> bool {
        self.analysis_complete
    }

    /// Reset the normalizer to initial state.
    pub fn reset(&mut self) {
        self.analyzer.reset();
        self.processor.reset();
        self.analysis_complete = false;
    }

    /// Get the normalizer configuration.
    pub fn config(&self) -> &NormalizerConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_validation() {
        let config = NormalizerConfig::new(Standard::EbuR128, 48000.0, 2);
        assert!(config.validate().is_ok());

        let bad_config = NormalizerConfig {
            sample_rate: 1000.0, // Too low
            ..config
        };
        assert!(bad_config.validate().is_err());
    }

    #[test]
    fn test_normalizer_creation() {
        let config = NormalizerConfig::new(Standard::EbuR128, 48000.0, 2);
        let normalizer = Normalizer::new(config);
        assert!(normalizer.is_ok());
    }

    #[test]
    fn test_processing_modes() {
        assert_eq!(ProcessingMode::TwoPass, ProcessingMode::TwoPass);
        assert_ne!(ProcessingMode::TwoPass, ProcessingMode::OnePass);
    }
}
