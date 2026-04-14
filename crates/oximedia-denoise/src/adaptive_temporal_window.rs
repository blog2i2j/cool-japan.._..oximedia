//! Adaptive temporal window sizing based on motion level.
//!
//! This module dynamically adjusts the temporal window size used for
//! temporal denoising based on the detected motion level in the scene.
//!
//! **Key insight**: Static scenes benefit from large temporal windows
//! (more frames averaged = less noise), while high-motion scenes need
//! small windows (to avoid ghosting / motion blur artifacts).
//!
//! The adaptive strategy:
//! - Low motion: expand window to maximum (e.g., 11-15 frames)
//! - Medium motion: use moderate window (e.g., 5-7 frames)
//! - High motion: shrink window to minimum (e.g., 3 frames)
//! - Scene cut detected: reset to minimum and rebuild
//!
//! Motion level is estimated from block-based SAD (Sum of Absolute
//! Differences) between consecutive frames.

use crate::{DenoiseError, DenoiseResult};
use oximedia_codec::VideoFrame;
use std::collections::VecDeque;

/// Configuration for adaptive temporal window.
#[derive(Clone, Debug)]
pub struct AdaptiveWindowConfig {
    /// Minimum window size (must be odd, >= 3).
    pub min_window: usize,
    /// Maximum window size (must be odd, <= 15).
    pub max_window: usize,
    /// Motion threshold below which the scene is considered static.
    /// Measured in average SAD per pixel (0-255 scale).
    pub static_threshold: f32,
    /// Motion threshold above which the scene is considered high-motion.
    pub high_motion_threshold: f32,
    /// Scene cut detection threshold. SAD above this triggers a reset.
    pub scene_cut_threshold: f32,
    /// Smoothing factor for window size transitions (0.0 = instant, 1.0 = never change).
    pub smoothing: f32,
    /// Block size for motion estimation.
    pub block_size: usize,
}

impl Default for AdaptiveWindowConfig {
    fn default() -> Self {
        Self {
            min_window: 3,
            max_window: 11,
            static_threshold: 2.0,
            high_motion_threshold: 15.0,
            scene_cut_threshold: 40.0,
            smoothing: 0.3,
            block_size: 16,
        }
    }
}

impl AdaptiveWindowConfig {
    /// Validate the configuration.
    pub fn validate(&self) -> DenoiseResult<()> {
        if self.min_window < 3 {
            return Err(DenoiseError::InvalidConfig(
                "min_window must be >= 3".to_string(),
            ));
        }
        if self.min_window % 2 == 0 {
            return Err(DenoiseError::InvalidConfig(
                "min_window must be odd".to_string(),
            ));
        }
        if self.max_window < self.min_window {
            return Err(DenoiseError::InvalidConfig(
                "max_window must be >= min_window".to_string(),
            ));
        }
        if self.max_window % 2 == 0 {
            return Err(DenoiseError::InvalidConfig(
                "max_window must be odd".to_string(),
            ));
        }
        if self.max_window > 15 {
            return Err(DenoiseError::InvalidConfig(
                "max_window must be <= 15".to_string(),
            ));
        }
        if self.static_threshold < 0.0 || self.high_motion_threshold < 0.0 {
            return Err(DenoiseError::InvalidConfig(
                "Motion thresholds must be non-negative".to_string(),
            ));
        }
        if self.static_threshold >= self.high_motion_threshold {
            return Err(DenoiseError::InvalidConfig(
                "static_threshold must be < high_motion_threshold".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&self.smoothing) {
            return Err(DenoiseError::InvalidConfig(
                "smoothing must be between 0.0 and 1.0".to_string(),
            ));
        }
        if self.block_size == 0 || self.block_size > 64 {
            return Err(DenoiseError::InvalidConfig(
                "block_size must be between 1 and 64".to_string(),
            ));
        }
        Ok(())
    }
}

/// Motion level classification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MotionLevel {
    /// Very little motion (static scene).
    Static,
    /// Low motion (slow pans, gentle movement).
    Low,
    /// Medium motion (normal action).
    Medium,
    /// High motion (fast action, sports).
    High,
    /// Scene cut detected.
    SceneCut,
}

impl MotionLevel {
    /// Get a human-readable description.
    #[must_use]
    pub fn description(&self) -> &str {
        match self {
            Self::Static => "static (no motion)",
            Self::Low => "low motion",
            Self::Medium => "medium motion",
            Self::High => "high motion",
            Self::SceneCut => "scene cut",
        }
    }
}

/// Result of motion analysis for a frame pair.
#[derive(Clone, Debug)]
pub struct MotionAnalysis {
    /// Average SAD (Sum of Absolute Differences) per pixel.
    pub avg_sad: f32,
    /// Maximum block SAD (worst-case motion).
    pub max_block_sad: f32,
    /// Motion level classification.
    pub level: MotionLevel,
    /// Recommended temporal window size.
    pub recommended_window: usize,
    /// Fraction of blocks with significant motion (> static threshold).
    pub motion_coverage: f32,
}

/// Adaptive temporal window controller.
///
/// Tracks motion history and smoothly adjusts the temporal window size
/// based on detected motion levels.
pub struct AdaptiveTemporalWindow {
    config: AdaptiveWindowConfig,
    /// Current (smoothed) window size as a float for smooth transitions.
    current_window_f: f32,
    /// Previous frame's luma plane for motion detection.
    prev_luma: Option<Vec<u8>>,
    /// Previous frame dimensions.
    prev_dimensions: Option<(usize, usize)>,
    /// History of motion levels.
    motion_history: VecDeque<f32>,
    /// Maximum history length.
    history_len: usize,
}

impl AdaptiveTemporalWindow {
    /// Create a new adaptive temporal window controller.
    pub fn new(config: AdaptiveWindowConfig) -> DenoiseResult<Self> {
        config.validate()?;
        let initial_window = config.min_window as f32;
        Ok(Self {
            config,
            current_window_f: initial_window,
            prev_luma: None,
            prev_dimensions: None,
            motion_history: VecDeque::with_capacity(16),
            history_len: 8,
        })
    }

    /// Analyze motion between the current frame and the previous frame,
    /// and return the recommended temporal window size.
    ///
    /// Must be called for each frame in sequence.
    pub fn analyze_frame(&mut self, frame: &VideoFrame) -> DenoiseResult<MotionAnalysis> {
        if frame.planes.is_empty() {
            return Err(DenoiseError::ProcessingError(
                "Frame has no planes".to_string(),
            ));
        }

        let plane = &frame.planes[0];
        let (width, height) = frame.plane_dimensions(0);
        let w = width as usize;
        let h = height as usize;
        let stride = plane.stride;

        // Extract luma data (packed, no stride gaps)
        let mut current_luma = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                current_luma[y * w + x] = plane.data[y * stride + x];
            }
        }

        let analysis = if let (Some(ref prev), Some((pw, ph))) =
            (&self.prev_luma, self.prev_dimensions)
        {
            if pw == w && ph == h {
                self.compute_motion(&current_luma, prev, w, h)
            } else {
                // Dimension change = scene cut
                MotionAnalysis {
                    avg_sad: self.config.scene_cut_threshold + 1.0,
                    max_block_sad: self.config.scene_cut_threshold + 1.0,
                    level: MotionLevel::SceneCut,
                    recommended_window: self.config.min_window,
                    motion_coverage: 1.0,
                }
            }
        } else {
            // First frame: assume static
            MotionAnalysis {
                avg_sad: 0.0,
                max_block_sad: 0.0,
                level: MotionLevel::Static,
                recommended_window: self.config.max_window,
                motion_coverage: 0.0,
            }
        };

        // Update motion history
        self.motion_history.push_back(analysis.avg_sad);
        if self.motion_history.len() > self.history_len {
            self.motion_history.pop_front();
        }

        // Smooth the window transition
        let target = analysis.recommended_window as f32;
        if analysis.level == MotionLevel::SceneCut {
            // Instant reset on scene cut
            self.current_window_f = self.config.min_window as f32;
        } else {
            self.current_window_f = self.current_window_f * self.config.smoothing
                + target * (1.0 - self.config.smoothing);
        }

        // Store current frame for next comparison
        self.prev_luma = Some(current_luma);
        self.prev_dimensions = Some((w, h));

        Ok(analysis)
    }

    /// Get the current recommended window size (odd integer).
    #[must_use]
    pub fn current_window(&self) -> usize {
        let raw = self.current_window_f.round() as usize;
        // Ensure odd
        let window = if raw % 2 == 0 { raw + 1 } else { raw };
        window.clamp(self.config.min_window, self.config.max_window)
    }

    /// Get the smoothed motion level (exponential moving average of SAD).
    #[must_use]
    pub fn smoothed_motion(&self) -> f32 {
        if self.motion_history.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.motion_history.iter().sum();
        sum / self.motion_history.len() as f32
    }

    /// Reset the controller state (e.g., after a seek).
    pub fn reset(&mut self) {
        self.prev_luma = None;
        self.prev_dimensions = None;
        self.current_window_f = self.config.min_window as f32;
        self.motion_history.clear();
    }

    /// Get the configuration.
    #[must_use]
    pub fn config(&self) -> &AdaptiveWindowConfig {
        &self.config
    }

    /// Compute motion analysis between two luma planes.
    fn compute_motion(
        &self,
        current: &[u8],
        previous: &[u8],
        width: usize,
        height: usize,
    ) -> MotionAnalysis {
        let bs = self.config.block_size;
        let num_bx = width.div_ceil(bs);
        let num_by = height.div_ceil(bs);
        let total_blocks = num_bx * num_by;

        if total_blocks == 0 {
            return MotionAnalysis {
                avg_sad: 0.0,
                max_block_sad: 0.0,
                level: MotionLevel::Static,
                recommended_window: self.config.max_window,
                motion_coverage: 0.0,
            };
        }

        let mut total_sad = 0.0f64;
        let mut max_block_sad = 0.0f32;
        let mut motion_blocks = 0u32;

        for by in 0..num_by {
            for bx in 0..num_bx {
                let x0 = bx * bs;
                let y0 = by * bs;
                let x1 = (x0 + bs).min(width);
                let y1 = (y0 + bs).min(height);
                let block_pixels = (x1 - x0) * (y1 - y0);

                if block_pixels == 0 {
                    continue;
                }

                let mut block_sad = 0u64;
                for y in y0..y1 {
                    for x in x0..x1 {
                        let idx = y * width + x;
                        let diff =
                            (i32::from(current[idx]) - i32::from(previous[idx])).unsigned_abs();
                        block_sad += u64::from(diff);
                    }
                }

                let avg_block_sad = block_sad as f32 / block_pixels as f32;
                total_sad += f64::from(avg_block_sad);
                if avg_block_sad > max_block_sad {
                    max_block_sad = avg_block_sad;
                }
                if avg_block_sad > self.config.static_threshold {
                    motion_blocks += 1;
                }
            }
        }

        let avg_sad = (total_sad / total_blocks as f64) as f32;
        let motion_coverage = motion_blocks as f32 / total_blocks as f32;

        // Classify motion level
        let level = if avg_sad > self.config.scene_cut_threshold {
            MotionLevel::SceneCut
        } else if avg_sad > self.config.high_motion_threshold {
            MotionLevel::High
        } else if avg_sad > self.config.static_threshold * 3.0 {
            MotionLevel::Medium
        } else if avg_sad > self.config.static_threshold {
            MotionLevel::Low
        } else {
            MotionLevel::Static
        };

        // Map motion level to recommended window size
        let recommended_window = self.motion_to_window(avg_sad);

        MotionAnalysis {
            avg_sad,
            max_block_sad,
            level,
            recommended_window,
            motion_coverage,
        }
    }

    /// Map average SAD to a recommended window size.
    ///
    /// Uses a smooth interpolation between min and max window based on
    /// the motion level relative to the thresholds.
    fn motion_to_window(&self, avg_sad: f32) -> usize {
        let min_w = self.config.min_window as f32;
        let max_w = self.config.max_window as f32;

        let t = if avg_sad <= self.config.static_threshold {
            0.0 // Use maximum window
        } else if avg_sad >= self.config.high_motion_threshold {
            1.0 // Use minimum window
        } else {
            // Linear interpolation between thresholds
            let range = self.config.high_motion_threshold - self.config.static_threshold;
            (avg_sad - self.config.static_threshold) / range
        };

        // Interpolate: t=0 -> max_window, t=1 -> min_window
        let window_f = max_w * (1.0 - t) + min_w * t;
        let raw = window_f.round() as usize;

        // Ensure odd
        let window = if raw % 2 == 0 { raw + 1 } else { raw };
        window.clamp(self.config.min_window, self.config.max_window)
    }
}

/// Convenience function: analyze a frame sequence and return per-frame
/// window recommendations.
pub fn analyze_sequence_motion(
    frames: &[VideoFrame],
    config: &AdaptiveWindowConfig,
) -> DenoiseResult<Vec<MotionAnalysis>> {
    let mut controller = AdaptiveTemporalWindow::new(config.clone())?;
    let mut results = Vec::with_capacity(frames.len());

    for frame in frames {
        let analysis = controller.analyze_frame(frame)?;
        results.push(analysis);
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_core::PixelFormat;

    fn make_test_frame(width: u32, height: u32, fill: u8) -> VideoFrame {
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, width, height);
        frame.allocate();
        let stride = frame.planes[0].stride;
        for y in 0..height as usize {
            for x in 0..width as usize {
                frame.planes[0].data[y * stride + x] = fill;
            }
        }
        frame
    }

    fn make_shifted_frame(width: u32, height: u32, shift_x: usize) -> VideoFrame {
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, width, height);
        frame.allocate();
        let stride = frame.planes[0].stride;
        for y in 0..height as usize {
            for x in 0..width as usize {
                // Gradient pattern shifted horizontally
                let val = (((x + shift_x) * 4) % 256) as u8;
                frame.planes[0].data[y * stride + x] = val;
            }
        }
        frame
    }

    #[test]
    fn test_config_default_valid() {
        let config = AdaptiveWindowConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_invalid_min_window_even() {
        let config = AdaptiveWindowConfig {
            min_window: 4,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_invalid_max_less_than_min() {
        let config = AdaptiveWindowConfig {
            min_window: 7,
            max_window: 5,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_invalid_thresholds() {
        let config = AdaptiveWindowConfig {
            static_threshold: 20.0,
            high_motion_threshold: 10.0,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_controller_creation() {
        let config = AdaptiveWindowConfig::default();
        let controller = AdaptiveTemporalWindow::new(config);
        assert!(controller.is_ok());
    }

    #[test]
    fn test_first_frame_is_static() {
        let config = AdaptiveWindowConfig::default();
        let mut ctrl = AdaptiveTemporalWindow::new(config)
            .expect("controller should be valid");

        let frame = make_test_frame(64, 64, 128);
        let analysis = ctrl.analyze_frame(&frame).expect("analysis should succeed");

        assert_eq!(analysis.level, MotionLevel::Static);
        assert!((analysis.avg_sad - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_identical_frames_are_static() {
        let config = AdaptiveWindowConfig::default();
        let mut ctrl = AdaptiveTemporalWindow::new(config)
            .expect("controller should be valid");

        let frame = make_test_frame(64, 64, 128);
        let _ = ctrl.analyze_frame(&frame).expect("frame 1");
        let analysis = ctrl.analyze_frame(&frame).expect("frame 2");

        assert_eq!(analysis.level, MotionLevel::Static);
        assert!((analysis.avg_sad - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_static_scene_gets_large_window() {
        let config = AdaptiveWindowConfig::default();
        let mut ctrl = AdaptiveTemporalWindow::new(config.clone())
            .expect("controller should be valid");

        let frame = make_test_frame(64, 64, 128);
        // Feed multiple identical frames
        for _ in 0..5 {
            let _ = ctrl.analyze_frame(&frame).expect("analysis ok");
        }

        let window = ctrl.current_window();
        assert!(
            window >= config.max_window - 2,
            "Static scene should converge toward max window, got {window}"
        );
    }

    #[test]
    fn test_high_motion_gets_small_window() {
        let config = AdaptiveWindowConfig::default();
        let mut ctrl = AdaptiveTemporalWindow::new(config.clone())
            .expect("controller should be valid");

        // Alternate between very different frames
        for i in 0..8 {
            let fill = if i % 2 == 0 { 50 } else { 200 };
            let frame = make_test_frame(64, 64, fill);
            let _ = ctrl.analyze_frame(&frame).expect("analysis ok");
        }

        let window = ctrl.current_window();
        assert!(
            window <= config.min_window + 2,
            "High motion should converge toward min window, got {window}"
        );
    }

    #[test]
    fn test_scene_cut_resets_window() {
        let config = AdaptiveWindowConfig::default();
        let mut ctrl = AdaptiveTemporalWindow::new(config.clone())
            .expect("controller should be valid");

        // Build up to large window with static scene
        let frame_a = make_test_frame(64, 64, 128);
        for _ in 0..6 {
            let _ = ctrl.analyze_frame(&frame_a).expect("analysis ok");
        }

        // Now a scene cut (dramatically different frame)
        let frame_b = make_test_frame(64, 64, 10);
        let analysis = ctrl.analyze_frame(&frame_b).expect("scene cut analysis");

        // Should detect scene cut or high motion and reset
        assert!(
            analysis.level == MotionLevel::SceneCut || analysis.level == MotionLevel::High,
            "Expected scene cut or high motion, got {:?}",
            analysis.level
        );
        assert_eq!(
            ctrl.current_window(),
            config.min_window,
            "Window should reset to minimum after scene cut"
        );
    }

    #[test]
    fn test_moderate_motion_intermediate_window() {
        let config = AdaptiveWindowConfig::default();
        let mut ctrl = AdaptiveTemporalWindow::new(config.clone())
            .expect("controller should be valid");

        // Create frames with moderate motion (small gradient shift)
        for i in 0..8 {
            let frame = make_shifted_frame(64, 64, i * 2);
            let _ = ctrl.analyze_frame(&frame).expect("analysis ok");
        }

        let window = ctrl.current_window();
        assert!(
            window >= config.min_window && window <= config.max_window,
            "Moderate motion window should be in range, got {window}"
        );
    }

    #[test]
    fn test_motion_to_window_interpolation() {
        let config = AdaptiveWindowConfig::default();
        let ctrl = AdaptiveTemporalWindow::new(config.clone())
            .expect("controller should be valid");

        // Below static threshold -> max window
        let w_static = ctrl.motion_to_window(0.5);
        assert_eq!(w_static, config.max_window);

        // Above high motion threshold -> min window
        let w_high = ctrl.motion_to_window(config.high_motion_threshold + 1.0);
        assert_eq!(w_high, config.min_window);

        // In between should be in range
        let mid = (config.static_threshold + config.high_motion_threshold) / 2.0;
        let w_mid = ctrl.motion_to_window(mid);
        assert!(w_mid >= config.min_window && w_mid <= config.max_window);
    }

    #[test]
    fn test_current_window_always_odd() {
        let config = AdaptiveWindowConfig::default();
        let mut ctrl = AdaptiveTemporalWindow::new(config)
            .expect("controller should be valid");

        for i in 0..10 {
            let fill = (i * 30 + 20) as u8;
            let frame = make_test_frame(64, 64, fill);
            let _ = ctrl.analyze_frame(&frame).expect("analysis ok");
            let window = ctrl.current_window();
            assert!(window % 2 == 1, "Window must be odd, got {window}");
        }
    }

    #[test]
    fn test_reset_controller() {
        let config = AdaptiveWindowConfig::default();
        let mut ctrl = AdaptiveTemporalWindow::new(config.clone())
            .expect("controller should be valid");

        let frame = make_test_frame(64, 64, 128);
        for _ in 0..5 {
            let _ = ctrl.analyze_frame(&frame).expect("analysis ok");
        }

        ctrl.reset();
        assert_eq!(ctrl.current_window(), config.min_window);
        assert!((ctrl.smoothed_motion() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_smoothed_motion_empty() {
        let config = AdaptiveWindowConfig::default();
        let ctrl = AdaptiveTemporalWindow::new(config)
            .expect("controller should be valid");
        assert!((ctrl.smoothed_motion() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_analyze_sequence_motion() {
        let config = AdaptiveWindowConfig::default();
        let frames: Vec<VideoFrame> = (0..5)
            .map(|i| make_test_frame(64, 64, (i * 10 + 100) as u8))
            .collect();

        let results = analyze_sequence_motion(&frames, &config);
        assert!(results.is_ok());
        let analyses = results.expect("analyses should be valid");
        assert_eq!(analyses.len(), 5);
    }

    #[test]
    fn test_empty_planes_error() {
        let config = AdaptiveWindowConfig::default();
        let mut ctrl = AdaptiveTemporalWindow::new(config)
            .expect("controller should be valid");

        let frame = VideoFrame::new(PixelFormat::Yuv420p, 64, 64);
        // Not allocated
        let result = ctrl.analyze_frame(&frame);
        assert!(result.is_err());
    }

    #[test]
    fn test_motion_coverage() {
        let config = AdaptiveWindowConfig::default();
        let mut ctrl = AdaptiveTemporalWindow::new(config)
            .expect("controller should be valid");

        // Two identical frames -> zero coverage
        let frame = make_test_frame(64, 64, 128);
        let _ = ctrl.analyze_frame(&frame).expect("frame 1");
        let analysis = ctrl.analyze_frame(&frame).expect("frame 2");
        assert!(
            (analysis.motion_coverage - 0.0).abs() < f32::EPSILON,
            "Identical frames should have zero motion coverage"
        );
    }

    #[test]
    fn test_motion_level_descriptions() {
        assert!(!MotionLevel::Static.description().is_empty());
        assert!(!MotionLevel::Low.description().is_empty());
        assert!(!MotionLevel::Medium.description().is_empty());
        assert!(!MotionLevel::High.description().is_empty());
        assert!(!MotionLevel::SceneCut.description().is_empty());
    }
}
