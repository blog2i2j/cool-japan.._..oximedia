//! Dissolve transition detection.

use crate::error::{ShotError, ShotResult};
use crate::frame_buffer::FrameBuffer;

/// Dissolve transition detector.
pub struct DissolveDetector {
    /// Threshold for dissolve detection.
    threshold: f32,
    /// Minimum dissolve duration in frames.
    min_duration: usize,
}

impl DissolveDetector {
    /// Create a new dissolve detector.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            threshold: 0.15,
            min_duration: 10,
        }
    }

    /// Detect dissolve transition in a sequence of frames.
    ///
    /// # Errors
    ///
    /// Returns error if frames are invalid or have mismatched dimensions.
    pub fn detect_dissolve(&self, frames: &[FrameBuffer]) -> ShotResult<(bool, f32, usize)> {
        if frames.len() < self.min_duration {
            return Ok((false, 0.0, 0));
        }

        let mut dissolve_scores = Vec::new();

        // Calculate variance for each frame
        for frame in frames {
            let variance = self.calculate_variance(frame)?;
            dissolve_scores.push(variance);
        }

        // Look for characteristic dissolve pattern (increasing then decreasing variance)
        let mut max_dissolve_score = 0.0;
        let mut max_dissolve_pos = 0;

        for i in self.min_duration..frames.len() {
            let window = &dissolve_scores[i.saturating_sub(self.min_duration)..=i];
            let score = self.analyze_dissolve_pattern(window);

            if score > max_dissolve_score {
                max_dissolve_score = score;
                max_dissolve_pos = i;
            }
        }

        let is_dissolve = max_dissolve_score > self.threshold;
        Ok((is_dissolve, max_dissolve_score, max_dissolve_pos))
    }

    /// Calculate variance in a frame (indicates blending).
    fn calculate_variance(&self, frame: &FrameBuffer) -> ShotResult<f32> {
        let shape = frame.dim();
        if shape.2 < 3 {
            return Err(ShotError::InvalidFrame(
                "Frame must have at least 3 channels".to_string(),
            ));
        }

        let total_pixels = (shape.0 * shape.1 * 3) as f32;
        let mut sum = 0.0;
        let mut sum_sq = 0.0;

        for y in 0..shape.0 {
            for x in 0..shape.1 {
                for c in 0..3 {
                    let val = f32::from(frame.get(y, x, c));
                    sum += val;
                    sum_sq += val * val;
                }
            }
        }

        let mean = sum / total_pixels;
        let variance = (sum_sq / total_pixels) - (mean * mean);

        Ok(variance / 255.0) // Normalize
    }

    /// Analyze variance pattern to detect dissolve.
    fn analyze_dissolve_pattern(&self, variances: &[f32]) -> f32 {
        if variances.len() < 3 {
            return 0.0;
        }

        // Calculate smoothness of transition
        let mut smoothness = 0.0;
        for i in 1..variances.len() {
            let diff = (variances[i] - variances[i - 1]).abs();
            smoothness += diff;
        }
        smoothness /= variances.len() as f32;

        // Dissolves should have smooth transitions
        let dissolve_score = if smoothness < 0.1 {
            1.0 - smoothness * 5.0
        } else {
            0.0
        };

        dissolve_score.max(0.0).min(1.0)
    }
}

impl Default for DissolveDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dissolve_detector_creation() {
        let detector = DissolveDetector::new();
        assert!((detector.threshold - 0.15).abs() < f32::EPSILON);
    }

    #[test]
    fn test_no_dissolve_in_single_frames() {
        let detector = DissolveDetector::new();
        let frames = vec![FrameBuffer::zeros(100, 100, 3)];
        let result = detector.detect_dissolve(&frames);
        assert!(result.is_ok());
        if let Ok((is_dissolve, _, _)) = result {
            assert!(!is_dissolve);
        }
    }

    #[test]
    fn test_variance_calculation() {
        let detector = DissolveDetector::new();
        let frame = FrameBuffer::from_elem(100, 100, 3, 128);
        let variance = detector.calculate_variance(&frame);
        assert!(variance.is_ok());
        if let Ok(v) = variance {
            assert!(v.abs() < 0.01); // Low variance for uniform frame
        }
    }
}
