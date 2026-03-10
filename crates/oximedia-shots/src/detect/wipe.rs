//! Wipe transition detection.

use crate::error::{ShotError, ShotResult};
use crate::frame_buffer::FrameBuffer;

/// Wipe direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WipeDirection {
    /// Left to right.
    Left,
    /// Right to left.
    Right,
    /// Top to bottom.
    Down,
    /// Bottom to top.
    Up,
}

/// Wipe transition detector.
pub struct WipeDetector {
    /// Threshold for wipe detection.
    threshold: f32,
    /// Number of sample points to check.
    sample_points: usize,
}

impl WipeDetector {
    /// Create a new wipe detector.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            threshold: 0.3,
            sample_points: 10,
        }
    }

    /// Detect wipe transition between frames.
    ///
    /// # Errors
    ///
    /// Returns error if frames are invalid or dimensions don't match.
    pub fn detect_wipe(
        &self,
        frames: &[FrameBuffer],
    ) -> ShotResult<Option<(WipeDirection, usize)>> {
        if frames.len() < 2 {
            return Ok(None);
        }

        // Check each direction
        if let Some(pos) = self.detect_horizontal_wipe(frames, true)? {
            return Ok(Some((WipeDirection::Left, pos)));
        }

        if let Some(pos) = self.detect_horizontal_wipe(frames, false)? {
            return Ok(Some((WipeDirection::Right, pos)));
        }

        if let Some(pos) = self.detect_vertical_wipe(frames, true)? {
            return Ok(Some((WipeDirection::Down, pos)));
        }

        if let Some(pos) = self.detect_vertical_wipe(frames, false)? {
            return Ok(Some((WipeDirection::Up, pos)));
        }

        Ok(None)
    }

    /// Detect horizontal wipe (left or right).
    fn detect_horizontal_wipe(
        &self,
        frames: &[FrameBuffer],
        left_to_right: bool,
    ) -> ShotResult<Option<usize>> {
        for i in 1..frames.len() {
            let frame1 = &frames[i - 1];
            let frame2 = &frames[i];

            if frame1.dim() != frame2.dim() {
                return Err(ShotError::InvalidFrame(
                    "Frame dimensions do not match".to_string(),
                ));
            }

            let shape = frame1.dim();
            let width = shape.1;
            let _height = shape.0;

            // Sample vertical strips across the frame
            let mut wipe_score = 0.0;
            for strip_idx in 0..self.sample_points {
                let x = (strip_idx * width) / self.sample_points;
                let diff = self.calculate_strip_difference_vertical(frame1, frame2, x)?;

                // For left-to-right wipe, strips on the left should change more
                let expected_diff = if left_to_right {
                    1.0 - (strip_idx as f32 / self.sample_points as f32)
                } else {
                    strip_idx as f32 / self.sample_points as f32
                };

                // Check if the difference pattern matches wipe
                if (diff - expected_diff).abs() < 0.3 {
                    wipe_score += 1.0;
                }
            }

            wipe_score /= self.sample_points as f32;

            if wipe_score > self.threshold {
                return Ok(Some(i));
            }
        }

        Ok(None)
    }

    /// Detect vertical wipe (up or down).
    fn detect_vertical_wipe(
        &self,
        frames: &[FrameBuffer],
        top_to_bottom: bool,
    ) -> ShotResult<Option<usize>> {
        for i in 1..frames.len() {
            let frame1 = &frames[i - 1];
            let frame2 = &frames[i];

            if frame1.dim() != frame2.dim() {
                return Err(ShotError::InvalidFrame(
                    "Frame dimensions do not match".to_string(),
                ));
            }

            let shape = frame1.dim();
            let height = shape.0;

            // Sample horizontal strips across the frame
            let mut wipe_score = 0.0;
            for strip_idx in 0..self.sample_points {
                let y = (strip_idx * height) / self.sample_points;
                let diff = self.calculate_strip_difference_horizontal(frame1, frame2, y)?;

                // For top-to-bottom wipe, strips on top should change more
                let expected_diff = if top_to_bottom {
                    1.0 - (strip_idx as f32 / self.sample_points as f32)
                } else {
                    strip_idx as f32 / self.sample_points as f32
                };

                if (diff - expected_diff).abs() < 0.3 {
                    wipe_score += 1.0;
                }
            }

            wipe_score /= self.sample_points as f32;

            if wipe_score > self.threshold {
                return Ok(Some(i));
            }
        }

        Ok(None)
    }

    /// Calculate difference in a vertical strip.
    fn calculate_strip_difference_vertical(
        &self,
        frame1: &FrameBuffer,
        frame2: &FrameBuffer,
        x: usize,
    ) -> ShotResult<f32> {
        let shape = frame1.dim();
        let height = shape.0;

        let mut diff_sum = 0.0;
        let mut count = 0;

        for y in 0..height {
            for c in 0..3 {
                let val1 = f32::from(frame1.get(y, x, c));
                let val2 = f32::from(frame2.get(y, x, c));
                diff_sum += (val1 - val2).abs();
                count += 1;
            }
        }

        Ok((diff_sum / count as f32) / 255.0)
    }

    /// Calculate difference in a horizontal strip.
    fn calculate_strip_difference_horizontal(
        &self,
        frame1: &FrameBuffer,
        frame2: &FrameBuffer,
        y: usize,
    ) -> ShotResult<f32> {
        let shape = frame1.dim();
        let width = shape.1;

        let mut diff_sum = 0.0;
        let mut count = 0;

        for x in 0..width {
            for c in 0..3 {
                let val1 = f32::from(frame1.get(y, x, c));
                let val2 = f32::from(frame2.get(y, x, c));
                diff_sum += (val1 - val2).abs();
                count += 1;
            }
        }

        Ok((diff_sum / count as f32) / 255.0)
    }
}

impl Default for WipeDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wipe_detector_creation() {
        let detector = WipeDetector::new();
        assert!((detector.threshold - 0.3).abs() < f32::EPSILON);
    }

    #[test]
    fn test_no_wipe_in_single_frame() {
        let detector = WipeDetector::new();
        let frames = vec![FrameBuffer::zeros(100, 100, 3)];
        let result = detector.detect_wipe(&frames);
        assert!(result.is_ok());
        if let Ok(wipe) = result {
            assert!(wipe.is_none());
        }
    }

    #[test]
    fn test_no_wipe_in_identical_frames() {
        let detector = WipeDetector::new();
        let frames = vec![FrameBuffer::zeros(100, 100, 3); 10];
        let result = detector.detect_wipe(&frames);
        assert!(result.is_ok());
        if let Ok(wipe) = result {
            assert!(wipe.is_none());
        }
    }
}
