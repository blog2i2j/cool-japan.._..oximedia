//! Fade in/out detection (fade to/from black or white).

use crate::error::{ShotError, ShotResult};
use crate::frame_buffer::FrameBuffer;

/// Fade direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FadeDirection {
    /// Fade to black/white.
    Out,
    /// Fade from black/white.
    In,
}

/// Fade color.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FadeColor {
    /// Fade to/from black.
    Black,
    /// Fade to/from white.
    White,
}

/// Fade detector.
pub struct FadeDetector {
    /// Threshold for fade detection (0.0 to 1.0).
    threshold: f32,
    /// Minimum fade duration in frames.
    min_duration: usize,
}

impl FadeDetector {
    /// Create a new fade detector.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            threshold: 0.2,
            min_duration: 8,
        }
    }

    /// Detect fade in a sequence of frames.
    ///
    /// # Errors
    ///
    /// Returns error if frames are invalid.
    pub fn detect_fade(
        &self,
        frames: &[FrameBuffer],
    ) -> ShotResult<Option<(FadeDirection, FadeColor, usize)>> {
        if frames.len() < self.min_duration {
            return Ok(None);
        }

        // Calculate brightness for each frame
        let brightness: Vec<f32> = frames
            .iter()
            .map(|f| self.calculate_brightness(f))
            .collect::<Result<Vec<_>, _>>()?;

        // Check for fade to black
        if let Some(pos) = self.detect_fade_pattern(&brightness, true, false) {
            return Ok(Some((FadeDirection::Out, FadeColor::Black, pos)));
        }

        // Check for fade from black
        if let Some(pos) = self.detect_fade_pattern(&brightness, false, false) {
            return Ok(Some((FadeDirection::In, FadeColor::Black, pos)));
        }

        // Check for fade to white
        if let Some(pos) = self.detect_fade_pattern(&brightness, true, true) {
            return Ok(Some((FadeDirection::Out, FadeColor::White, pos)));
        }

        // Check for fade from white
        if let Some(pos) = self.detect_fade_pattern(&brightness, false, true) {
            return Ok(Some((FadeDirection::In, FadeColor::White, pos)));
        }

        Ok(None)
    }

    /// Calculate average brightness of a frame.
    fn calculate_brightness(&self, frame: &FrameBuffer) -> ShotResult<f32> {
        let shape = frame.dim();
        if shape.2 < 3 {
            return Err(ShotError::InvalidFrame(
                "Frame must have at least 3 channels".to_string(),
            ));
        }

        let total_pixels = (shape.0 * shape.1) as f32;
        let mut sum = 0.0;

        for y in 0..shape.0 {
            for x in 0..shape.1 {
                let r = f32::from(frame.get(y, x, 0));
                let g = f32::from(frame.get(y, x, 1));
                let b = f32::from(frame.get(y, x, 2));
                sum += (r + g + b) / 3.0;
            }
        }

        Ok(sum / total_pixels / 255.0) // Normalize to 0-1
    }

    /// Detect fade pattern in brightness values.
    fn detect_fade_pattern(
        &self,
        brightness: &[f32],
        fade_out: bool,
        to_white: bool,
    ) -> Option<usize> {
        let target = if to_white { 1.0 } else { 0.0 };

        for i in self.min_duration..brightness.len() {
            let window = &brightness[i.saturating_sub(self.min_duration)..=i];

            // Check if we have a monotonic trend toward target
            let is_fade = if fade_out {
                self.is_monotonic_toward_target(window, target)
            } else {
                self.is_monotonic_away_from_target(window, target)
            };

            if is_fade {
                return Some(i);
            }
        }

        None
    }

    /// Check if brightness values are monotonically moving toward target.
    fn is_monotonic_toward_target(&self, values: &[f32], target: f32) -> bool {
        if values.len() < 2 {
            return false;
        }

        let start = values[0];
        let end = values[values.len() - 1];

        // Check if we're moving toward target
        let moving_toward = (end - target).abs() < (start - target).abs();
        if !moving_toward {
            return false;
        }

        // Check if change is significant
        let change = (end - start).abs();
        if change < self.threshold {
            return false;
        }

        // Check monotonicity
        let direction = if start < target { 1.0 } else { -1.0 };
        for i in 1..values.len() {
            let diff = values[i] - values[i - 1];
            if diff * direction < -0.05 {
                // Allow small deviations
                return false;
            }
        }

        true
    }

    /// Check if brightness values are monotonically moving away from target.
    fn is_monotonic_away_from_target(&self, values: &[f32], target: f32) -> bool {
        if values.len() < 2 {
            return false;
        }

        let start = values[0];
        let end = values[values.len() - 1];

        // Check if we're moving away from target
        let moving_away = (end - target).abs() > (start - target).abs();
        if !moving_away {
            return false;
        }

        // Check if change is significant
        let change = (end - start).abs();
        if change < self.threshold {
            return false;
        }

        // Check monotonicity
        let direction = if start < target { -1.0 } else { 1.0 };
        for i in 1..values.len() {
            let diff = values[i] - values[i - 1];
            if diff * direction < -0.05 {
                return false;
            }
        }

        true
    }
}

impl Default for FadeDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fade_detector_creation() {
        let detector = FadeDetector::new();
        assert!((detector.threshold - 0.2).abs() < f32::EPSILON);
    }

    #[test]
    fn test_brightness_calculation() {
        let detector = FadeDetector::new();
        let black_frame = FrameBuffer::zeros(100, 100, 3);
        let white_frame = FrameBuffer::from_elem(100, 100, 3, 255);

        let black_brightness = detector.calculate_brightness(&black_frame);
        assert!(black_brightness.is_ok());
        if let Ok(b) = black_brightness {
            assert!(b < 0.01);
        }

        let white_brightness = detector.calculate_brightness(&white_frame);
        assert!(white_brightness.is_ok());
        if let Ok(b) = white_brightness {
            assert!((b - 1.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_no_fade_in_uniform_sequence() {
        let detector = FadeDetector::new();
        let frames = vec![FrameBuffer::from_elem(100, 100, 3, 128); 20];
        let result = detector.detect_fade(&frames);
        assert!(result.is_ok());
        if let Ok(fade) = result {
            assert!(fade.is_none());
        }
    }
}
