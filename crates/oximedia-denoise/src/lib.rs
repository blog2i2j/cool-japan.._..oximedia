//! Professional video denoising for `OxiMedia`.
//!
//! This crate provides comprehensive video denoising capabilities including:
//!
//! - **Spatial Denoising**: Remove noise within individual frames
//!   - Bilateral filtering (edge-preserving)
//!   - Non-Local Means (patch-based)
//!   - Wiener filtering (frequency-domain)
//!   - Wavelet denoising (multi-resolution)
//!
//! - **Temporal Denoising**: Remove noise across frame sequences
//!   - Temporal averaging (weighted)
//!   - Temporal median filtering
//!   - Motion-compensated filtering
//!   - Kalman filtering (prediction/correction)
//!
//! - **Hybrid Denoising**: Combined spatial and temporal
//!   - Spatio-temporal filtering
//!   - Adaptive content-aware denoising
//!
//! - **Advanced Features**:
//!   - Motion estimation and compensation
//!   - Film grain analysis and preservation
//!   - Multi-scale pyramid and wavelet processing
//!   - Automatic noise level estimation
//!
//! # Example
//!
//! ```
//! use oximedia_denoise::{DenoiseConfig, DenoiseMode, Denoiser};
//! use oximedia_codec::VideoFrame;
//! use oximedia_core::PixelFormat;
//!
//! let config = DenoiseConfig {
//!     mode: DenoiseMode::Balanced,
//!     strength: 0.7,
//!     temporal_window: 5,
//!     preserve_edges: true,
//!     preserve_grain: false,
//! };
//!
//! let mut denoiser = Denoiser::new(config);
//! let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 1920, 1080);
//! frame.allocate();
//!
//! // Process frame
//! let denoised = denoiser.process(&frame)?;
//! ```

#![forbid(unsafe_code)]
// Algorithmic casts for bounds checking in image processing are necessary
#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::module_name_repetitions
)]
#![warn(missing_docs)]

pub mod adaptive_denoise;
pub mod audio;
pub mod audio_denoise;
pub mod bilateral;
pub mod chroma_denoise;
pub mod deblock;
pub mod denoise_config;
pub mod denoise_metrics;
pub mod estimator;
pub mod grain;
pub mod hybrid;
pub mod motion;
pub mod multiscale;
pub mod noise_estimate;
pub mod noise_model;
pub mod profile;
pub mod region_denoise;
pub mod spatial;
pub mod spectral_gate;
pub mod temporal;
pub mod video;
pub mod video_denoise;

use oximedia_codec::VideoFrame;
use oximedia_core::PixelFormat;
use thiserror::Error;

/// Denoising error types.
#[derive(Error, Debug)]
pub enum DenoiseError {
    /// Unsupported pixel format.
    #[error("Unsupported pixel format: {0:?}")]
    UnsupportedFormat(PixelFormat),

    /// Invalid configuration.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Processing error.
    #[error("Processing error: {0}")]
    ProcessingError(String),

    /// Motion estimation failed.
    #[error("Motion estimation failed: {0}")]
    MotionEstimationError(String),

    /// Insufficient frames for temporal processing.
    #[error("Insufficient frames: need at least {0}")]
    InsufficientFrames(usize),
}

/// Result type for denoising operations.
pub type DenoiseResult<T> = Result<T, DenoiseError>;

/// Denoising mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DenoiseMode {
    /// Fast bilateral filter (real-time capable).
    Fast,
    /// Balanced motion-compensated temporal + spatial.
    Balanced,
    /// High quality `NLMeans` or wavelet (slow but best quality).
    Quality,
    /// Grain-aware mode that preserves film grain.
    GrainAware,
    /// Custom mode with manual algorithm selection.
    Custom,
}

/// Denoising configuration.
#[derive(Clone, Debug)]
pub struct DenoiseConfig {
    /// Processing mode.
    pub mode: DenoiseMode,
    /// Denoising strength (0.0 = none, 1.0 = maximum).
    pub strength: f32,
    /// Temporal window size (number of frames to consider).
    pub temporal_window: usize,
    /// Preserve edges while denoising.
    pub preserve_edges: bool,
    /// Preserve film grain.
    pub preserve_grain: bool,
}

impl Default for DenoiseConfig {
    fn default() -> Self {
        Self {
            mode: DenoiseMode::Balanced,
            strength: 0.5,
            temporal_window: 5,
            preserve_edges: true,
            preserve_grain: false,
        }
    }
}

impl DenoiseConfig {
    /// Create configuration for light denoising.
    #[must_use]
    pub fn light() -> Self {
        Self {
            strength: 0.3,
            ..Default::default()
        }
    }

    /// Create configuration for medium denoising.
    #[must_use]
    pub fn medium() -> Self {
        Self {
            strength: 0.5,
            ..Default::default()
        }
    }

    /// Create configuration for strong denoising.
    #[must_use]
    pub fn strong() -> Self {
        Self {
            strength: 0.8,
            temporal_window: 7,
            ..Default::default()
        }
    }

    /// Validate configuration.
    pub fn validate(&self) -> DenoiseResult<()> {
        if !(0.0..=1.0).contains(&self.strength) {
            return Err(DenoiseError::InvalidConfig(
                "Strength must be between 0.0 and 1.0".to_string(),
            ));
        }

        if self.temporal_window < 3 || self.temporal_window > 15 {
            return Err(DenoiseError::InvalidConfig(
                "Temporal window must be between 3 and 15".to_string(),
            ));
        }

        if self.temporal_window % 2 == 0 {
            return Err(DenoiseError::InvalidConfig(
                "Temporal window must be odd".to_string(),
            ));
        }

        Ok(())
    }
}

/// Main denoiser interface.
pub struct Denoiser {
    config: DenoiseConfig,
    frame_buffer: Vec<VideoFrame>,
    motion_estimator: Option<motion::estimation::MotionEstimator>,
    noise_estimator: estimator::noise::NoiseEstimator,
}

impl Denoiser {
    /// Create a new denoiser with the given configuration.
    pub fn new(config: DenoiseConfig) -> Self {
        config.validate().expect("Invalid configuration");

        Self {
            config,
            frame_buffer: Vec::new(),
            motion_estimator: None,
            noise_estimator: estimator::noise::NoiseEstimator::new(),
        }
    }

    /// Process a single frame.
    pub fn process(&mut self, frame: &VideoFrame) -> DenoiseResult<VideoFrame> {
        // Estimate noise level if not already done
        if self.noise_estimator.noise_level().is_none() {
            self.noise_estimator.estimate(frame)?;
        }

        // Add frame to buffer
        self.frame_buffer.push(frame.clone());

        // Keep buffer size limited
        let max_buffer = self.config.temporal_window;
        if self.frame_buffer.len() > max_buffer {
            self.frame_buffer.remove(0);
        }

        // Apply denoising based on mode
        match self.config.mode {
            DenoiseMode::Fast => self.process_fast(frame),
            DenoiseMode::Balanced => self.process_balanced(frame),
            DenoiseMode::Quality => self.process_quality(frame),
            DenoiseMode::GrainAware => self.process_grain_aware(frame),
            DenoiseMode::Custom => self.process_custom(frame),
        }
    }

    /// Reset the denoiser state.
    pub fn reset(&mut self) {
        self.frame_buffer.clear();
        self.motion_estimator = None;
        self.noise_estimator = estimator::noise::NoiseEstimator::new();
    }

    /// Get the current configuration.
    #[must_use]
    pub fn config(&self) -> &DenoiseConfig {
        &self.config
    }

    /// Get estimated noise level.
    #[must_use]
    pub fn noise_level(&self) -> Option<f32> {
        self.noise_estimator.noise_level()
    }

    fn process_fast(&self, frame: &VideoFrame) -> DenoiseResult<VideoFrame> {
        spatial::bilateral::bilateral_filter(frame, self.config.strength)
    }

    fn process_balanced(&mut self, frame: &VideoFrame) -> DenoiseResult<VideoFrame> {
        if self.frame_buffer.len() < 3 {
            // Not enough frames for temporal processing, use spatial only
            return self.process_fast(frame);
        }

        // Use motion-compensated temporal + spatial
        hybrid::spatiotemporal::spatio_temporal_denoise(
            frame,
            &self.frame_buffer,
            self.config.strength,
            self.config.preserve_edges,
        )
    }

    fn process_quality(&self, frame: &VideoFrame) -> DenoiseResult<VideoFrame> {
        // Use Non-Local Means for highest quality
        spatial::nlmeans::nlmeans_filter(frame, self.config.strength)
    }

    fn process_grain_aware(&mut self, frame: &VideoFrame) -> DenoiseResult<VideoFrame> {
        // Analyze grain pattern
        let grain_map = grain::analysis::analyze_grain(frame)?;

        // Apply denoising with grain preservation
        grain::preserve::preserve_grain_denoise(frame, &grain_map, self.config.strength)
    }

    fn process_custom(&mut self, frame: &VideoFrame) -> DenoiseResult<VideoFrame> {
        // Custom processing can be extended by users
        self.process_balanced(frame)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = DenoiseConfig::default();
        assert_eq!(config.mode, DenoiseMode::Balanced);
        assert!((config.strength - 0.5).abs() < f32::EPSILON);
        assert_eq!(config.temporal_window, 5);
    }

    #[test]
    fn test_config_presets() {
        let light = DenoiseConfig::light();
        assert!((light.strength - 0.3).abs() < f32::EPSILON);

        let medium = DenoiseConfig::medium();
        assert!((medium.strength - 0.5).abs() < f32::EPSILON);

        let strong = DenoiseConfig::strong();
        assert!((strong.strength - 0.8).abs() < f32::EPSILON);
        assert_eq!(strong.temporal_window, 7);
    }

    #[test]
    fn test_config_validation() {
        let mut config = DenoiseConfig::default();
        assert!(config.validate().is_ok());

        config.strength = 1.5;
        assert!(config.validate().is_err());

        config.strength = 0.5;
        config.temporal_window = 2;
        assert!(config.validate().is_err());

        config.temporal_window = 4; // even
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_denoiser_creation() {
        let config = DenoiseConfig::default();
        let denoiser = Denoiser::new(config);
        assert_eq!(denoiser.config().mode, DenoiseMode::Balanced);
    }

    #[test]
    fn test_denoiser_process() {
        let config = DenoiseConfig::default();
        let mut denoiser = Denoiser::new(config);

        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 64, 64);
        frame.allocate();

        let result = denoiser.process(&frame);
        assert!(result.is_ok());
    }

    #[test]
    fn test_denoiser_reset() {
        let config = DenoiseConfig::default();
        let mut denoiser = Denoiser::new(config);

        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 64, 64);
        frame.allocate();

        let _ = denoiser.process(&frame);
        assert!(!denoiser.frame_buffer.is_empty());

        denoiser.reset();
        assert!(denoiser.frame_buffer.is_empty());
    }
}
