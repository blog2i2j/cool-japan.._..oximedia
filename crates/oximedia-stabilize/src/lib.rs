//! Professional video stabilization for `OxiMedia`.
//!
//! `oximedia-stabilize` provides comprehensive video stabilization algorithms with
//! advanced features including motion estimation, trajectory smoothing, rolling shutter
//! correction, 3D stabilization, and horizon leveling.
//!
//! # Features
//!
//! - **Motion Estimation**: Track camera motion across frames using various models
//! - **Motion Smoothing**: Multiple smoothing algorithms (Gaussian, Kalman, adaptive)
//! - **Transform Calculation**: Compute optimal stabilization transforms
//! - **Frame Warping**: Apply stabilization with various interpolation methods
//! - **Rolling Shutter Correction**: Fix rolling shutter artifacts
//! - **3D Stabilization**: Full 3D camera motion estimation and correction
//! - **Horizon Leveling**: Automatic horizon detection and correction
//! - **Zoom Optimization**: Minimize black borders while maximizing output resolution
//! - **Motion Blur**: Optional synthetic motion blur for smooth results
//! - **Multi-pass Analysis**: Analyze entire video before stabilizing for optimal results
//!
//! # Stabilization Modes
//!
//! - **Translation Only**: Simple pan/tilt stabilization
//! - **Affine**: Translation + rotation + scale
//! - **Perspective**: Full perspective correction
//! - **3D**: Full 3D camera motion estimation
//!
//! # Quality Presets
//!
//! - **Fast**: Basic stabilization with minimal quality loss
//! - **Balanced**: Good stabilization with reasonable cropping
//! - **Maximum**: Best stabilization, may crop more
//!
//! # Example
//!
//! ```rust,no_run
//! use oximedia_stabilize::{Stabilizer, StabilizeConfig, StabilizationMode, QualityPreset};
//!
//! // Create a stabilizer with balanced quality
//! let config = StabilizeConfig::new()
//!     .with_mode(StabilizationMode::Affine)
//!     .with_quality(QualityPreset::Balanced)
//!     .with_smoothing_strength(0.8);
//!
//! let mut stabilizer = Stabilizer::new(config);
//!
//! // Process frames (example - actual implementation would use real frames)
//! // let stabilized = stabilizer.stabilize(&frames)?;
//! ```
//!
//! # Architecture
//!
//! The stabilization pipeline consists of:
//!
//! 1. **Motion Tracking**: Detect and track features across frames
//! 2. **Motion Estimation**: Estimate camera motion (translation, rotation, scale)
//! 3. **Trajectory Building**: Build motion trajectories over time
//! 4. **Smoothing**: Apply smoothing filters to trajectories
//! 5. **Transform Calculation**: Compute stabilization transforms
//! 6. **Optimization**: Optimize transforms to minimize cropping
//! 7. **Warping**: Apply transforms to frames with interpolation
//! 8. **Post-processing**: Optional motion blur, horizon leveling, etc.

#![deny(unsafe_code)]
#![warn(missing_docs)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::similar_names)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::too_many_lines)]
#![allow(dead_code)]

pub mod adaptive_crop;
pub mod blur;
pub mod crop;
pub mod crop_region;
pub mod error;
pub mod feature_track;
pub mod gyro;
pub mod horizon;
pub mod jitter_detect;
pub mod keyframe_filter;
pub mod mesh_warp;
pub mod motion;
pub mod motion_model;
pub mod multipass;
pub mod parallax_compensate;
pub mod path_planner;
pub mod rolling;
pub mod rolling_shutter;
pub mod smooth;
pub mod smoothing;
pub mod stabilize_config;
pub mod stabilize_report;
pub mod three_d;
pub mod trajectory;
pub mod transform;
pub mod vibration_isolate;
pub mod warp;
pub mod warp_field;
pub mod zoom;

// Re-export commonly used items
pub use error::{StabilizeError, StabilizeResult};

use scirs2_core::ndarray::Array2;
use serde::{Deserialize, Serialize};

/// Stabilization mode determines the type of camera motion to correct.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StabilizationMode {
    /// Translation only (pan/tilt)
    Translation,
    /// Affine transformation (translation + rotation + scale)
    Affine,
    /// Perspective transformation (full homography)
    Perspective,
    /// Full 3D camera motion
    ThreeD,
}

impl Default for StabilizationMode {
    fn default() -> Self {
        Self::Affine
    }
}

/// Quality preset for stabilization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityPreset {
    /// Fast processing with basic stabilization
    Fast,
    /// Balanced quality and performance
    Balanced,
    /// Maximum quality, slower processing
    Maximum,
}

impl Default for QualityPreset {
    fn default() -> Self {
        Self::Balanced
    }
}

impl QualityPreset {
    /// Get smoothing window size for this preset.
    #[must_use]
    pub const fn smoothing_window(self) -> usize {
        match self {
            Self::Fast => 15,
            Self::Balanced => 30,
            Self::Maximum => 60,
        }
    }

    /// Get feature count for this preset.
    #[must_use]
    pub const fn feature_count(self) -> usize {
        match self {
            Self::Fast => 200,
            Self::Balanced => 500,
            Self::Maximum => 1000,
        }
    }

    /// Get whether to enable multi-pass analysis.
    #[must_use]
    pub const fn enable_multipass(self) -> bool {
        match self {
            Self::Fast => false,
            Self::Balanced => true,
            Self::Maximum => true,
        }
    }
}

/// Stabilization configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilizeConfig {
    /// Stabilization mode
    pub mode: StabilizationMode,
    /// Quality preset
    pub quality: QualityPreset,
    /// Smoothing strength (0.0 = no smoothing, 1.0 = maximum)
    pub smoothing_strength: f64,
    /// Crop ratio to avoid black borders (0.0-1.0)
    pub crop_ratio: f64,
    /// Enable rolling shutter correction
    pub enable_rolling_shutter: bool,
    /// Enable horizon leveling
    pub enable_horizon_leveling: bool,
    /// Enable motion blur synthesis
    pub enable_motion_blur: bool,
    /// Enable zoom optimization
    pub enable_zoom_optimization: bool,
    /// Maximum motion magnitude for adaptive smoothing
    pub max_motion_magnitude: f64,
    /// Number of features to track
    pub feature_count: usize,
    /// Enable multi-pass analysis
    pub enable_multipass: bool,
    /// Enable GPU acceleration (if available)
    pub enable_gpu: bool,
    /// Enable preview mode (faster, lower quality)
    pub preview_mode: bool,
}

impl Default for StabilizeConfig {
    fn default() -> Self {
        Self {
            mode: StabilizationMode::default(),
            quality: QualityPreset::default(),
            smoothing_strength: 0.8,
            crop_ratio: 0.95,
            enable_rolling_shutter: false,
            enable_horizon_leveling: false,
            enable_motion_blur: false,
            enable_zoom_optimization: true,
            max_motion_magnitude: 100.0,
            feature_count: 500,
            enable_multipass: true,
            enable_gpu: false,
            preview_mode: false,
        }
    }
}

impl StabilizeConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set stabilization mode.
    #[must_use]
    pub const fn with_mode(mut self, mode: StabilizationMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set quality preset.
    #[must_use]
    pub fn with_quality(mut self, quality: QualityPreset) -> Self {
        self.quality = quality;
        self.feature_count = quality.feature_count();
        self.enable_multipass = quality.enable_multipass();
        self
    }

    /// Set smoothing strength.
    #[must_use]
    pub fn with_smoothing_strength(mut self, strength: f64) -> Self {
        self.smoothing_strength = strength.clamp(0.0, 1.0);
        self
    }

    /// Set crop ratio.
    #[must_use]
    pub fn with_crop_ratio(mut self, ratio: f64) -> Self {
        self.crop_ratio = ratio.clamp(0.0, 1.0);
        self
    }

    /// Enable or disable rolling shutter correction.
    #[must_use]
    pub const fn with_rolling_shutter(mut self, enable: bool) -> Self {
        self.enable_rolling_shutter = enable;
        self
    }

    /// Enable or disable horizon leveling.
    #[must_use]
    pub const fn with_horizon_leveling(mut self, enable: bool) -> Self {
        self.enable_horizon_leveling = enable;
        self
    }

    /// Enable or disable motion blur.
    #[must_use]
    pub const fn with_motion_blur(mut self, enable: bool) -> Self {
        self.enable_motion_blur = enable;
        self
    }

    /// Enable or disable zoom optimization.
    #[must_use]
    pub const fn with_zoom_optimization(mut self, enable: bool) -> Self {
        self.enable_zoom_optimization = enable;
        self
    }

    /// Enable or disable preview mode.
    #[must_use]
    pub const fn with_preview_mode(mut self, enable: bool) -> Self {
        self.preview_mode = enable;
        self
    }

    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if any configuration parameter is invalid.
    pub fn validate(&self) -> StabilizeResult<()> {
        if !(0.0..=1.0).contains(&self.smoothing_strength) {
            return Err(StabilizeError::InvalidParameter {
                name: "smoothing_strength".to_string(),
                value: self.smoothing_strength.to_string(),
            });
        }
        if !(0.0..=1.0).contains(&self.crop_ratio) {
            return Err(StabilizeError::InvalidParameter {
                name: "crop_ratio".to_string(),
                value: self.crop_ratio.to_string(),
            });
        }
        if self.max_motion_magnitude <= 0.0 {
            return Err(StabilizeError::InvalidParameter {
                name: "max_motion_magnitude".to_string(),
                value: self.max_motion_magnitude.to_string(),
            });
        }
        if self.feature_count == 0 {
            return Err(StabilizeError::InvalidParameter {
                name: "feature_count".to_string(),
                value: self.feature_count.to_string(),
            });
        }
        Ok(())
    }
}

/// Main video stabilizer.
#[derive(Debug)]
pub struct Stabilizer {
    config: StabilizeConfig,
    motion_tracker: motion::tracker::MotionTracker,
    motion_estimator: motion::estimate::MotionEstimator,
    trajectory_smoother: smooth::filter::TrajectorySmoother,
    transform_calculator: transform::calculate::TransformCalculator,
    frame_warper: warp::apply::FrameWarper,
    rolling_shutter_corrector: Option<rolling::correct::RollingShutterCorrector>,
    three_d_stabilizer: Option<three_d::stabilize::ThreeDStabilizer>,
    horizon_leveler: Option<horizon::level::HorizonLeveler>,
    zoom_optimizer: Option<zoom::calculate::ZoomOptimizer>,
    motion_blur: Option<blur::motion::MotionBlur>,
    multipass_analyzer: Option<multipass::analyze::MultipassAnalyzer>,
}

impl Stabilizer {
    /// Create a new stabilizer with the given configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid.
    pub fn new(config: StabilizeConfig) -> StabilizeResult<Self> {
        config.validate()?;

        let motion_tracker = motion::tracker::MotionTracker::new(config.feature_count);
        let motion_estimator = motion::estimate::MotionEstimator::new(config.mode);
        let trajectory_smoother = smooth::filter::TrajectorySmoother::new(
            config.quality.smoothing_window(),
            config.smoothing_strength,
        );
        let transform_calculator = transform::calculate::TransformCalculator::new();
        let frame_warper = warp::apply::FrameWarper::new();

        let rolling_shutter_corrector = if config.enable_rolling_shutter {
            Some(rolling::correct::RollingShutterCorrector::new())
        } else {
            None
        };

        let three_d_stabilizer = if config.mode == StabilizationMode::ThreeD {
            Some(three_d::stabilize::ThreeDStabilizer::new())
        } else {
            None
        };

        let horizon_leveler = if config.enable_horizon_leveling {
            Some(horizon::level::HorizonLeveler::new())
        } else {
            None
        };

        let zoom_optimizer = if config.enable_zoom_optimization {
            Some(zoom::calculate::ZoomOptimizer::new(config.crop_ratio))
        } else {
            None
        };

        let motion_blur = if config.enable_motion_blur {
            Some(blur::motion::MotionBlur::new())
        } else {
            None
        };

        let multipass_analyzer = if config.enable_multipass {
            Some(multipass::analyze::MultipassAnalyzer::new())
        } else {
            None
        };

        Ok(Self {
            config,
            motion_tracker,
            motion_estimator,
            trajectory_smoother,
            transform_calculator,
            frame_warper,
            rolling_shutter_corrector,
            three_d_stabilizer,
            horizon_leveler,
            zoom_optimizer,
            motion_blur,
            multipass_analyzer,
        })
    }

    /// Create a stabilizer with default configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the default configuration is invalid (should never happen).
    pub fn default_config() -> StabilizeResult<Self> {
        Self::new(StabilizeConfig::default())
    }

    /// Get a reference to the configuration.
    #[must_use]
    pub const fn config(&self) -> &StabilizeConfig {
        &self.config
    }

    /// Stabilize a sequence of frames.
    ///
    /// This is the main entry point for video stabilization. It processes the entire
    /// video sequence and returns stabilized frames.
    ///
    /// # Arguments
    ///
    /// * `frames` - Input video frames (as raw pixel data)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Frame sequence is empty
    /// - Motion tracking fails
    /// - Transform calculation fails
    /// - Frame warping fails
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use oximedia_stabilize::{Stabilizer, StabilizeConfig};
    ///
    /// let stabilizer = Stabilizer::new(StabilizeConfig::default()).expect("should succeed in test");
    /// // let stabilized = stabilizer.stabilize(&frames).expect("should succeed in test");
    /// ```
    pub fn stabilize(&mut self, frames: &[Frame]) -> StabilizeResult<Vec<Frame>> {
        if frames.is_empty() {
            return Err(StabilizeError::EmptyFrameSequence);
        }

        // Multi-pass analysis if enabled
        if let Some(analyzer) = &mut self.multipass_analyzer {
            let analysis = analyzer.analyze(frames)?;
            self.apply_multipass_optimization(&analysis)?;
        }

        // Step 1: Track features across frames
        let feature_tracks = match self.motion_tracker.track(frames) {
            Ok(tracks) => tracks,
            Err(StabilizeError::InsufficientFeatures { .. }) => {
                // Fall back to identity transforms when features cannot be detected
                let transforms: Vec<_> = (0..frames.len())
                    .map(transform::calculate::StabilizationTransform::identity)
                    .collect();
                let stabilized_frames = self.frame_warper.warp(frames, &transforms)?;
                return Ok(stabilized_frames);
            }
            Err(e) => return Err(e),
        };

        // Step 2: Estimate motion between frames
        let motion_models = self
            .motion_estimator
            .estimate(&feature_tracks, frames.len())?;

        // Step 3: Build trajectories from motion models
        let trajectories = motion::trajectory::Trajectory::from_models(&motion_models)?;

        // Step 4: Smooth trajectories
        let smoothed_trajectories = self.trajectory_smoother.smooth(&trajectories)?;

        // Step 5: Calculate stabilization transforms
        let mut transforms = self
            .transform_calculator
            .calculate(&trajectories, &smoothed_trajectories)?;

        // Step 6: Apply rolling shutter correction if enabled
        if let Some(corrector) = &mut self.rolling_shutter_corrector {
            transforms = corrector.correct_transforms(&transforms, frames)?;
        }

        // Step 7: Apply 3D stabilization if enabled
        if let Some(stabilizer) = &mut self.three_d_stabilizer {
            transforms = stabilizer.stabilize_3d(&transforms, &feature_tracks)?;
        }

        // Step 8: Optimize transforms (minimize cropping)
        if let Some(optimizer) = &mut self.zoom_optimizer {
            transforms = optimizer.optimize(&transforms, frames[0].width, frames[0].height)?;
        }

        // Step 9: Apply horizon leveling if enabled
        if let Some(leveler) = &mut self.horizon_leveler {
            transforms = leveler.level(&transforms, frames)?;
        }

        // Step 10: Warp frames with transforms
        let mut stabilized_frames = self.frame_warper.warp(frames, &transforms)?;

        // Step 11: Apply motion blur if enabled
        if let Some(blur) = &mut self.motion_blur {
            stabilized_frames = blur.apply(&stabilized_frames, &transforms)?;
        }

        Ok(stabilized_frames)
    }

    /// Apply optimizations based on multi-pass analysis.
    fn apply_multipass_optimization(
        &mut self,
        analysis: &multipass::analyze::MultipassAnalysis,
    ) -> StabilizeResult<()> {
        // Adjust smoothing strength based on detected motion patterns
        if analysis.max_motion_magnitude > self.config.max_motion_magnitude * 2.0 {
            // High motion - reduce smoothing to avoid over-smoothing
            self.trajectory_smoother
                .set_strength(self.config.smoothing_strength * 0.7);
        } else if analysis.max_motion_magnitude < self.config.max_motion_magnitude * 0.5 {
            // Low motion - can increase smoothing
            self.trajectory_smoother
                .set_strength(self.config.smoothing_strength * 1.2);
        }

        Ok(())
    }
}

/// A video frame representation for stabilization.
///
/// This is a simplified frame structure for the stabilization pipeline.
/// In practice, this would integrate with the full `OxiMedia` frame types.
#[derive(Debug, Clone)]
pub struct Frame {
    /// Frame width in pixels
    pub width: usize,
    /// Frame height in pixels
    pub height: usize,
    /// Frame timestamp in seconds
    pub timestamp: f64,
    /// Raw pixel data (grayscale for motion tracking)
    pub data: Array2<u8>,
}

impl Frame {
    /// Create a new frame.
    #[must_use]
    pub fn new(width: usize, height: usize, timestamp: f64, data: Array2<u8>) -> Self {
        Self {
            width,
            height,
            timestamp,
            data,
        }
    }

    /// Get frame dimensions.
    #[must_use]
    pub const fn dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }
}

/// Stabilization statistics and metrics.
#[derive(Debug, Clone, Default)]
pub struct StabilizationStats {
    /// Number of frames processed
    pub frame_count: usize,
    /// Average motion magnitude before stabilization
    pub avg_motion_before: f64,
    /// Average motion magnitude after stabilization
    pub avg_motion_after: f64,
    /// Maximum motion magnitude before stabilization
    pub max_motion_before: f64,
    /// Maximum motion magnitude after stabilization
    pub max_motion_after: f64,
    /// Improvement ratio (0.0-1.0, higher is better)
    pub improvement_ratio: f64,
    /// Average cropping percentage
    pub avg_crop_percentage: f64,
    /// Processing time in seconds
    pub processing_time: f64,
}

impl StabilizationStats {
    /// Create new statistics.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            frame_count: 0,
            avg_motion_before: 0.0,
            avg_motion_after: 0.0,
            max_motion_before: 0.0,
            max_motion_after: 0.0,
            improvement_ratio: 0.0,
            avg_crop_percentage: 0.0,
            processing_time: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = StabilizeConfig::default();
        assert_eq!(config.mode, StabilizationMode::Affine);
        assert_eq!(config.quality, QualityPreset::Balanced);
        assert!((config.smoothing_strength - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn test_config_builder() {
        let config = StabilizeConfig::new()
            .with_mode(StabilizationMode::ThreeD)
            .with_quality(QualityPreset::Maximum)
            .with_smoothing_strength(0.9)
            .with_crop_ratio(0.85);

        assert_eq!(config.mode, StabilizationMode::ThreeD);
        assert_eq!(config.quality, QualityPreset::Maximum);
        assert!((config.smoothing_strength - 0.9).abs() < f64::EPSILON);
        assert!((config.crop_ratio - 0.85).abs() < f64::EPSILON);
    }

    #[test]
    fn test_config_validation() {
        let config = StabilizeConfig::default();
        assert!(config.validate().is_ok());

        let mut bad_config = StabilizeConfig::default();
        bad_config.smoothing_strength = 1.5;
        assert!(bad_config.validate().is_err());

        let mut bad_config2 = StabilizeConfig::default();
        bad_config2.feature_count = 0;
        assert!(bad_config2.validate().is_err());
    }

    #[test]
    fn test_quality_presets() {
        assert_eq!(QualityPreset::Fast.smoothing_window(), 15);
        assert_eq!(QualityPreset::Balanced.smoothing_window(), 30);
        assert_eq!(QualityPreset::Maximum.smoothing_window(), 60);

        assert_eq!(QualityPreset::Fast.feature_count(), 200);
        assert_eq!(QualityPreset::Balanced.feature_count(), 500);
        assert_eq!(QualityPreset::Maximum.feature_count(), 1000);
    }

    #[test]
    fn test_stabilizer_creation() {
        let config = StabilizeConfig::default();
        let stabilizer = Stabilizer::new(config);
        assert!(stabilizer.is_ok());
    }

    #[test]
    fn test_frame_creation() {
        let data = Array2::zeros((100, 100));
        let frame = Frame::new(100, 100, 0.0, data);
        assert_eq!(frame.dimensions(), (100, 100));
        assert!((frame.timestamp - 0.0).abs() < f64::EPSILON);
    }
}

/// Example usage and integration patterns.
pub mod examples {
    use super::*;

    /// Basic stabilization example.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use oximedia_stabilize::examples::basic_stabilization_example;
    ///
    /// // basic_stabilization_example();
    /// ```
    #[allow(dead_code)]
    pub fn basic_stabilization_example() -> StabilizeResult<()> {
        // Create a simple configuration
        let config = StabilizeConfig::new()
            .with_mode(StabilizationMode::Affine)
            .with_smoothing_strength(0.7)
            .with_crop_ratio(0.95);

        let mut stabilizer = Stabilizer::new(config)?;

        // Create dummy frames for demonstration
        let frames = create_test_frames(5, 64, 64);

        // Stabilize the video
        let _stabilized = stabilizer.stabilize(&frames)?;

        Ok(())
    }

    /// Advanced stabilization with all features enabled.
    #[allow(dead_code)]
    pub fn advanced_stabilization_example() -> StabilizeResult<()> {
        let config = StabilizeConfig::new()
            .with_mode(StabilizationMode::ThreeD)
            .with_quality(QualityPreset::Maximum)
            .with_smoothing_strength(0.9)
            .with_rolling_shutter(true)
            .with_horizon_leveling(true)
            .with_motion_blur(true)
            .with_zoom_optimization(true);

        let mut stabilizer = Stabilizer::new(config)?;

        let frames = create_test_frames(5, 64, 64);
        let _stabilized = stabilizer.stabilize(&frames)?;

        Ok(())
    }

    /// Fast preview mode example.
    #[allow(dead_code)]
    pub fn preview_mode_example() -> StabilizeResult<()> {
        let config = StabilizeConfig::new()
            .with_quality(QualityPreset::Fast)
            .with_preview_mode(true);

        let mut stabilizer = Stabilizer::new(config)?;

        let frames = create_test_frames(5, 64, 64);
        let _stabilized = stabilizer.stabilize(&frames)?;

        Ok(())
    }

    /// Custom parameter tuning example.
    #[allow(dead_code)]
    pub fn custom_tuning_example() -> StabilizeResult<()> {
        // Start with a preset
        let mut config = StabilizeConfig::new().with_quality(QualityPreset::Balanced);

        // Customize specific parameters
        config.smoothing_strength = 0.85;
        config.crop_ratio = 0.92;
        config.max_motion_magnitude = 120.0;
        config.feature_count = 750;

        let mut stabilizer = Stabilizer::new(config)?;

        let frames = create_test_frames(5, 64, 64);
        let _stabilized = stabilizer.stabilize(&frames)?;

        Ok(())
    }

    /// Helper to create test frames.
    fn create_test_frames(count: usize, width: usize, height: usize) -> Vec<Frame> {
        (0..count)
            .map(|i| {
                Frame::new(
                    width,
                    height,
                    i as f64 / 30.0,
                    Array2::zeros((height, width)),
                )
            })
            .collect()
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_basic_example() {
            let result = basic_stabilization_example();
            assert!(result.is_ok());
        }

        #[test]
        fn test_preview_mode() {
            let result = preview_mode_example();
            assert!(result.is_ok());
        }
    }
}

/// Utilities for benchmarking and performance analysis.
pub mod benchmarks {
    use super::*;
    use std::time::Instant;

    /// Benchmark results.
    #[derive(Debug, Clone)]
    pub struct BenchmarkResult {
        /// Total processing time in seconds
        pub total_time: f64,
        /// Time per frame in milliseconds
        pub time_per_frame: f64,
        /// Frames per second
        pub fps: f64,
        /// Memory usage (bytes)
        pub memory_usage: usize,
    }

    /// Benchmark stabilization performance.
    #[allow(dead_code)]
    pub fn benchmark_stabilization(
        config: StabilizeConfig,
        frame_count: usize,
        width: usize,
        height: usize,
    ) -> StabilizeResult<BenchmarkResult> {
        let frames = (0..frame_count)
            .map(|i| {
                Frame::new(
                    width,
                    height,
                    i as f64 / 30.0,
                    Array2::zeros((height, width)),
                )
            })
            .collect::<Vec<_>>();

        let mut stabilizer = Stabilizer::new(config)?;

        let start = Instant::now();
        let _stabilized = stabilizer.stabilize(&frames)?;
        let elapsed = start.elapsed().as_secs_f64();

        let time_per_frame = (elapsed / frame_count as f64) * 1000.0;
        let fps = frame_count as f64 / elapsed;

        let memory_usage = std::mem::size_of::<Frame>() * frame_count * 2; // Input + output

        Ok(BenchmarkResult {
            total_time: elapsed,
            time_per_frame,
            fps,
            memory_usage,
        })
    }

    /// Compare different quality presets.
    #[allow(dead_code)]
    pub fn compare_presets() -> StabilizeResult<Vec<(QualityPreset, BenchmarkResult)>> {
        let presets = [
            QualityPreset::Fast,
            QualityPreset::Balanced,
            QualityPreset::Maximum,
        ];

        let mut results = Vec::new();

        for &preset in &presets {
            let config = StabilizeConfig::new().with_quality(preset);
            let result = benchmark_stabilization(config, 5, 64, 64)?;
            results.push((preset, result));
        }

        Ok(results)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_benchmark() {
            let config = StabilizeConfig::default();
            let result = benchmark_stabilization(config, 3, 16, 16);
            assert!(result.is_ok());
        }

        #[test]
        fn test_preset_comparison() {
            let result = compare_presets();
            assert!(result.is_ok());
            let results = result.expect("should succeed in test");
            assert_eq!(results.len(), 3);
        }
    }
}

/// Validation and quality checks.
pub mod validation {
    use super::*;

    /// Validation result.
    #[derive(Debug, Clone)]
    pub struct ValidationResult {
        /// Whether validation passed
        pub passed: bool,
        /// Issues found
        pub issues: Vec<String>,
        /// Warnings
        pub warnings: Vec<String>,
    }

    /// Validate stabilization configuration.
    #[must_use]
    pub fn validate_config(config: &StabilizeConfig) -> ValidationResult {
        let mut issues = Vec::new();
        let mut warnings = Vec::new();

        if config.smoothing_strength < 0.0 || config.smoothing_strength > 1.0 {
            issues.push(format!(
                "Invalid smoothing_strength: {}. Must be between 0.0 and 1.0",
                config.smoothing_strength
            ));
        }

        if config.crop_ratio < 0.5 {
            warnings.push(format!(
                "Low crop_ratio: {}. May result in excessive cropping",
                config.crop_ratio
            ));
        }

        if config.feature_count < 100 {
            warnings.push(format!(
                "Low feature_count: {}. May result in poor tracking",
                config.feature_count
            ));
        }

        if config.enable_motion_blur && config.preview_mode {
            warnings
                .push("Motion blur enabled in preview mode. May slow down processing".to_string());
        }

        ValidationResult {
            passed: issues.is_empty(),
            issues,
            warnings,
        }
    }

    /// Validate input frames.
    #[must_use]
    pub fn validate_frames(frames: &[Frame]) -> ValidationResult {
        let mut issues = Vec::new();
        let mut warnings = Vec::new();

        if frames.is_empty() {
            issues.push("Frame sequence is empty".to_string());
            return ValidationResult {
                passed: false,
                issues,
                warnings,
            };
        }

        if frames.len() < 10 {
            warnings.push(format!(
                "Very short sequence: {} frames. Stabilization may not be effective",
                frames.len()
            ));
        }

        // Check for dimension consistency
        let first_dim = frames[0].dimensions();
        for (i, frame) in frames.iter().enumerate().skip(1) {
            if frame.dimensions() != first_dim {
                issues.push(format!(
                    "Dimension mismatch at frame {}: {:?} vs {:?}",
                    i,
                    frame.dimensions(),
                    first_dim
                ));
            }
        }

        ValidationResult {
            passed: issues.is_empty(),
            issues,
            warnings,
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_config_validation() {
            let config = StabilizeConfig::default();
            let result = validate_config(&config);
            assert!(result.passed);
        }

        #[test]
        fn test_invalid_config() {
            let mut config = StabilizeConfig::default();
            config.smoothing_strength = 1.5;
            let result = validate_config(&config);
            assert!(!result.passed);
            assert!(!result.issues.is_empty());
        }

        #[test]
        fn test_frame_validation() {
            let frames = vec![Frame::new(640, 480, 0.0, Array2::zeros((480, 640)))];
            let result = validate_frames(&frames);
            assert!(result.passed);
        }
    }
}

/// Integration tests and end-to-end examples.
#[cfg(test)]
mod integration_tests {
    use super::*;

    /// Test complete stabilization pipeline.
    #[test]
    fn test_end_to_end_stabilization() {
        let config = StabilizeConfig::default();
        let mut stabilizer = Stabilizer::new(config).expect("should succeed in test");

        let frames = create_test_sequence(5, 64, 64);
        let result = stabilizer.stabilize(&frames);

        assert!(result.is_ok());
        let stabilized = result.expect("should succeed in test");
        assert_eq!(stabilized.len(), frames.len());
    }

    /// Test different stabilization modes.
    #[test]
    fn test_all_stabilization_modes() {
        let modes = vec![
            StabilizationMode::Translation,
            StabilizationMode::Affine,
            StabilizationMode::Perspective,
        ];

        for mode in modes {
            let config = StabilizeConfig::new().with_mode(mode);
            let mut stabilizer = Stabilizer::new(config).expect("should succeed in test");

            let frames = create_test_sequence(5, 64, 64);
            let result = stabilizer.stabilize(&frames);
            assert!(result.is_ok());
        }
    }

    /// Test quality presets.
    #[test]
    fn test_quality_presets() {
        let presets = vec![
            QualityPreset::Fast,
            QualityPreset::Balanced,
            QualityPreset::Maximum,
        ];

        for preset in presets {
            let config = StabilizeConfig::new().with_quality(preset);
            let mut stabilizer = Stabilizer::new(config).expect("should succeed in test");

            let frames = create_test_sequence(5, 64, 64);
            let result = stabilizer.stabilize(&frames);
            assert!(result.is_ok());
        }
    }

    /// Test error handling.
    #[test]
    fn test_error_cases() {
        let config = StabilizeConfig::default();
        let mut stabilizer = Stabilizer::new(config).expect("should succeed in test");

        // Empty frame sequence
        let result = stabilizer.stabilize(&[]);
        assert!(result.is_err());

        // Invalid configuration
        let mut bad_config = StabilizeConfig::default();
        bad_config.smoothing_strength = 2.0;
        let result = bad_config.validate();
        assert!(result.is_err());
    }

    /// Test trajectory smoothing.
    #[test]
    fn test_trajectory_smoothing() {
        use motion::trajectory::Trajectory;

        let traj = Trajectory::new(50);
        let mut smoother = smooth::filter::TrajectorySmoother::new(15, 0.7);

        let result = smoother.smooth(&traj);
        assert!(result.is_ok());
    }

    /// Test transform calculation.
    #[test]
    fn test_transform_calculation() {
        use motion::trajectory::Trajectory;
        use transform::calculate::TransformCalculator;

        let original = Trajectory::new(30);
        let smoothed = Trajectory::new(30);

        let calculator = TransformCalculator::new();
        let result = calculator.calculate(&original, &smoothed);

        assert!(result.is_ok());
        let transforms = result.expect("should succeed in test");
        assert_eq!(transforms.len(), 30);
    }

    /// Test frame warping.
    #[test]
    fn test_frame_warping() {
        use transform::calculate::StabilizationTransform;
        use warp::apply::FrameWarper;

        let frames = create_test_sequence(3, 16, 16);
        let transforms: Vec<_> = (0..3)
            .map(|i| StabilizationTransform::identity(i))
            .collect();

        let warper = FrameWarper::new();
        let result = warper.warp(&frames, &transforms);

        assert!(result.is_ok());
    }

    /// Test motion tracking returns InsufficientFeatures for blank frames.
    #[test]
    fn test_motion_tracking() {
        use motion::tracker::MotionTracker;

        let mut tracker = MotionTracker::new(200);
        let frames = create_test_sequence(5, 64, 64);

        let result = tracker.track(&frames);
        // Blank (zero) frames have no corners, so tracker correctly reports insufficient features
        assert!(matches!(
            result,
            Err(StabilizeError::InsufficientFeatures { .. })
        ));
    }

    /// Test adaptive smoothing.
    #[test]
    fn test_adaptive_smoothing() {
        use motion::trajectory::Trajectory;
        use smooth::adaptive::AdaptiveSmoother;

        let mut smoother = AdaptiveSmoother::new(30, 100.0);
        let traj = Trajectory::new(40);

        let result = smoother.smooth(&traj);
        assert!(result.is_ok());
    }

    /// Test temporal coherence.
    #[test]
    fn test_temporal_coherence() {
        use motion::trajectory::Trajectory;
        use smooth::temporal::TemporalCoherence;

        let tc = TemporalCoherence::new(5.0, 2.0);
        let traj = Trajectory::new(50);

        let result = tc.optimize(&traj);
        assert!(result.is_ok());
    }

    /// Helper to create test frame sequence.
    fn create_test_sequence(count: usize, width: usize, height: usize) -> Vec<Frame> {
        (0..count)
            .map(|i| {
                Frame::new(
                    width,
                    height,
                    i as f64 / 30.0,
                    Array2::zeros((height, width)),
                )
            })
            .collect()
    }
}

/// Performance and stress tests.
#[cfg(test)]
mod performance_tests {
    use super::*;

    /// Test performance with high resolution.
    #[test]
    #[ignore] // Ignore by default due to time
    fn test_high_resolution_performance() {
        let config = StabilizeConfig::default();
        let mut stabilizer = Stabilizer::new(config).expect("should succeed in test");

        let frames = create_hd_sequence(10);
        let result = stabilizer.stabilize(&frames);

        assert!(result.is_ok());
    }

    /// Test performance with long sequence.
    #[test]
    #[ignore]
    fn test_long_sequence_performance() {
        let config = StabilizeConfig::default();
        let mut stabilizer = Stabilizer::new(config).expect("should succeed in test");

        let frames = create_test_sequence(300);
        let result = stabilizer.stabilize(&frames);

        assert!(result.is_ok());
    }

    /// Test memory efficiency.
    #[test]
    fn test_memory_efficiency() {
        let config = StabilizeConfig::default().with_preview_mode(true);
        let mut stabilizer = Stabilizer::new(config).expect("should succeed in test");

        let frames = create_test_sequence(5);
        let result = stabilizer.stabilize(&frames);

        assert!(result.is_ok());
    }

    fn create_test_sequence(count: usize) -> Vec<Frame> {
        (0..count)
            .map(|i| Frame::new(64, 64, i as f64 / 30.0, Array2::zeros((64, 64))))
            .collect()
    }

    fn create_hd_sequence(count: usize) -> Vec<Frame> {
        (0..count)
            .map(|i| Frame::new(1920, 1080, i as f64 / 30.0, Array2::zeros((1080, 1920))))
            .collect()
    }
}

/// Documentation examples that are tested.
#[cfg(test)]
mod doc_examples {
    use super::*;

    /// Example: Basic usage.
    #[test]
    fn example_basic_usage() {
        let config = StabilizeConfig::default();
        let mut stabilizer = Stabilizer::new(config).expect("should succeed in test");

        let frames = vec![
            Frame::new(16, 16, 0.0, Array2::zeros((16, 16))),
            Frame::new(16, 16, 1.0 / 30.0, Array2::zeros((16, 16))),
        ];

        let result = stabilizer.stabilize(&frames);
        assert!(result.is_ok());
    }

    /// Example: Custom configuration.
    #[test]
    fn example_custom_config() {
        let config = StabilizeConfig::new()
            .with_mode(StabilizationMode::Affine)
            .with_quality(QualityPreset::Maximum)
            .with_smoothing_strength(0.9);

        assert!(config.validate().is_ok());
    }

    /// Example: Quality presets.
    #[test]
    fn example_quality_presets() {
        let fast = StabilizeConfig::new().with_quality(QualityPreset::Fast);
        let balanced = StabilizeConfig::new().with_quality(QualityPreset::Balanced);
        let maximum = StabilizeConfig::new().with_quality(QualityPreset::Maximum);

        assert!(fast.feature_count < balanced.feature_count);
        assert!(balanced.feature_count < maximum.feature_count);
    }
}
