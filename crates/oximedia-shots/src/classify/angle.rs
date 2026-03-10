//! Camera angle classification.

use crate::error::{ShotError, ShotResult};
use crate::frame_buffer::{FrameBuffer, GrayImage};
use crate::types::CameraAngle;

/// Camera angle classifier.
pub struct AngleClassifier {
    /// Confidence threshold.
    confidence_threshold: f32,
}

impl AngleClassifier {
    /// Create a new angle classifier.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            confidence_threshold: 0.5,
        }
    }

    /// Classify camera angle.
    ///
    /// # Errors
    ///
    /// Returns error if frame is invalid.
    pub fn classify(&self, frame: &FrameBuffer) -> ShotResult<(CameraAngle, f32)> {
        let shape = frame.dim();
        if shape.2 < 3 {
            return Err(ShotError::InvalidFrame(
                "Frame must have at least 3 channels".to_string(),
            ));
        }

        // Analyze horizon line and vanishing points
        let horizon_position = self.detect_horizon_position(frame)?;

        // Analyze perspective distortion
        let perspective_score = self.analyze_perspective(frame)?;

        // Classify based on horizon and perspective
        let (angle, confidence) = if (horizon_position - 0.5).abs() < 0.1 && perspective_score < 0.2
        {
            (CameraAngle::EyeLevel, 0.8)
        } else if horizon_position < 0.3 {
            (CameraAngle::Low, 0.7 + perspective_score)
        } else if horizon_position > 0.7 {
            (CameraAngle::High, 0.7 + perspective_score)
        } else if horizon_position < 0.1 {
            (CameraAngle::BirdsEye, 0.75)
        } else if perspective_score > 0.6 {
            (CameraAngle::Dutch, 0.65)
        } else {
            (CameraAngle::Unknown, 0.4)
        };

        Ok((angle, confidence))
    }

    /// Detect horizon position (0.0 = top, 1.0 = bottom).
    fn detect_horizon_position(&self, frame: &FrameBuffer) -> ShotResult<f32> {
        let shape = frame.dim();
        let height = shape.0;

        // Convert to grayscale
        let gray = self.to_grayscale(frame);

        // Find strongest horizontal edges (likely horizon)
        let mut max_edge_strength = 0.0;
        let mut horizon_y = height / 2;

        for y in (height / 4)..(3 * height / 4) {
            let edge_strength = self.calculate_horizontal_edge_strength(&gray, y);
            if edge_strength > max_edge_strength {
                max_edge_strength = edge_strength;
                horizon_y = y;
            }
        }

        Ok(horizon_y as f32 / height as f32)
    }

    /// Calculate horizontal edge strength at a given row.
    fn calculate_horizontal_edge_strength(&self, gray: &GrayImage, y: usize) -> f32 {
        let shape = gray.dim();
        let width = shape.1;

        let mut edge_sum = 0.0;

        for x in 0..width {
            if y > 0 && y < shape.0 - 1 {
                let above = f32::from(gray.get(y - 1, x));
                let current = f32::from(gray.get(y, x));
                let below = f32::from(gray.get(y + 1, x));

                let gradient = ((above - current).abs() + (current - below).abs()) / 2.0;
                edge_sum += gradient;
            }
        }

        edge_sum / width as f32
    }

    /// Analyze perspective distortion (for Dutch angle detection).
    fn analyze_perspective(&self, frame: &FrameBuffer) -> ShotResult<f32> {
        let gray = self.to_grayscale(frame);
        let shape = gray.dim();

        // Detect lines using simplified Hough transform
        let mut horizontal_lines = 0;
        let mut vertical_lines = 0;
        let mut diagonal_lines = 0;

        // Sample edges at regular intervals
        for y in (10..shape.0).step_by(10) {
            for x in (10..shape.1).step_by(10) {
                if let Some(direction) = self.detect_edge_direction(&gray, x, y) {
                    match direction {
                        0 => horizontal_lines += 1,
                        1 => vertical_lines += 1,
                        _ => diagonal_lines += 1,
                    }
                }
            }
        }

        let total_lines = horizontal_lines + vertical_lines + diagonal_lines;
        if total_lines == 0 {
            return Ok(0.0);
        }

        // High diagonal ratio indicates Dutch angle
        Ok(diagonal_lines as f32 / total_lines as f32)
    }

    /// Detect edge direction at a point (0 = horizontal, 1 = vertical, 2 = diagonal).
    fn detect_edge_direction(&self, gray: &GrayImage, x: usize, y: usize) -> Option<u8> {
        let shape = gray.dim();

        if y < 2 || y >= shape.0 - 2 || x < 2 || x >= shape.1 - 2 {
            return None;
        }

        // Calculate gradients
        let gx = (f32::from(gray.get(y, x + 1)) - f32::from(gray.get(y, x - 1))).abs();
        let gy = (f32::from(gray.get(y + 1, x)) - f32::from(gray.get(y - 1, x))).abs();

        let threshold = 20.0;

        if gx < threshold && gy < threshold {
            return None;
        }

        if gx > gy * 1.5 {
            Some(1) // Vertical edge
        } else if gy > gx * 1.5 {
            Some(0) // Horizontal edge
        } else {
            Some(2) // Diagonal edge
        }
    }

    /// Convert RGB to grayscale.
    fn to_grayscale(&self, frame: &FrameBuffer) -> GrayImage {
        let shape = frame.dim();
        let mut gray = GrayImage::zeros(shape.0, shape.1);

        for y in 0..shape.0 {
            for x in 0..shape.1 {
                let r = f32::from(frame.get(y, x, 0));
                let g = f32::from(frame.get(y, x, 1));
                let b = f32::from(frame.get(y, x, 2));
                gray.set(y, x, ((r * 0.299) + (g * 0.587) + (b * 0.114)) as u8);
            }
        }

        gray
    }
}

impl Default for AngleClassifier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_angle_classifier_creation() {
        let classifier = AngleClassifier::new();
        assert!((classifier.confidence_threshold - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_classify_frame() {
        let classifier = AngleClassifier::new();
        let frame = FrameBuffer::from_elem(100, 100, 3, 128);
        let result = classifier.classify(&frame);
        assert!(result.is_ok());
    }

    #[test]
    fn test_horizon_detection() {
        let classifier = AngleClassifier::new();
        let mut frame = FrameBuffer::zeros(100, 100, 3);

        // Create artificial horizon in middle
        for x in 0..100 {
            for c in 0..3 {
                frame.set(50, x, c, 255);
            }
        }

        let horizon = classifier.detect_horizon_position(&frame);
        assert!(horizon.is_ok());
        if let Ok(pos) = horizon {
            assert!((pos - 0.5).abs() < 0.1);
        }
    }
}
