//! Shot type classification (ECU, CU, MCU, MS, MLS, LS, ELS).

use crate::error::{ShotError, ShotResult};
use crate::frame_buffer::{FrameBuffer, GrayImage};
use crate::types::ShotType;

/// Shot type classifier using face/person detection and framing analysis.
pub struct ShotTypeClassifier {
    /// Confidence threshold for classification.
    confidence_threshold: f32,
}

impl ShotTypeClassifier {
    /// Create a new shot type classifier.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            confidence_threshold: 0.5,
        }
    }

    /// Classify shot type based on frame content.
    ///
    /// # Errors
    ///
    /// Returns error if frame is invalid.
    pub fn classify(&self, frame: &FrameBuffer) -> ShotResult<(ShotType, f32)> {
        let shape = frame.dim();
        if shape.2 < 3 {
            return Err(ShotError::InvalidFrame(
                "Frame must have at least 3 channels".to_string(),
            ));
        }

        // Detect faces/people in frame
        let face_ratio = self.detect_face_size_ratio(frame)?;

        // Classify based on face/person size ratio
        let (shot_type, confidence) = if face_ratio > 0.6 {
            (ShotType::ExtremeCloseUp, 0.9)
        } else if face_ratio > 0.4 {
            (ShotType::CloseUp, 0.85)
        } else if face_ratio > 0.25 {
            (ShotType::MediumCloseUp, 0.8)
        } else if face_ratio > 0.15 {
            (ShotType::MediumShot, 0.75)
        } else if face_ratio > 0.08 {
            (ShotType::MediumLongShot, 0.7)
        } else if face_ratio > 0.03 {
            (ShotType::LongShot, 0.65)
        } else if face_ratio > 0.0 {
            (ShotType::ExtremeLongShot, 0.6)
        } else {
            // No face detected, analyze overall composition
            let composition_score = self.analyze_composition(frame)?;
            if composition_score > 0.7 {
                (ShotType::ExtremeLongShot, 0.5)
            } else {
                (ShotType::Unknown, 0.3)
            }
        };

        Ok((shot_type, confidence))
    }

    /// Detect face size ratio in frame (simplified Haar-like features).
    fn detect_face_size_ratio(&self, frame: &FrameBuffer) -> ShotResult<f32> {
        let shape = frame.dim();
        let height = shape.0;
        let width = shape.1;

        // Convert to grayscale
        let gray = self.to_grayscale(frame);

        // Simple face detection using skin tone and symmetry
        let mut max_face_ratio: f32 = 0.0;

        // Sample different regions
        let regions = [
            (width / 4, height / 4, width / 2, height / 2), // Center
            (width / 3, height / 5, width / 3, height / 2), // Upper center
        ];

        for (x, y, w, h) in regions {
            let skin_ratio = self.detect_skin_tone_ratio(&gray, x, y, w, h);
            let symmetry = self.calculate_symmetry(&gray, x, y, w, h);

            // Face likelihood based on skin tone and symmetry
            let face_likelihood = (skin_ratio * 0.7) + (symmetry * 0.3);

            if face_likelihood > 0.5 {
                let region_ratio = (w * h) as f32 / (width * height) as f32;
                max_face_ratio = max_face_ratio.max(region_ratio * face_likelihood);
            }
        }

        Ok(max_face_ratio)
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

    /// Detect skin tone ratio in region (simplified).
    fn detect_skin_tone_ratio(
        &self,
        _gray: &GrayImage,
        _x: usize,
        _y: usize,
        _w: usize,
        _h: usize,
    ) -> f32 {
        // Simplified implementation - would normally check RGB values for skin tone
        0.6
    }

    /// Calculate symmetry in region.
    fn calculate_symmetry(&self, gray: &GrayImage, x: usize, y: usize, w: usize, h: usize) -> f32 {
        let shape = gray.dim();
        let x_end = (x + w).min(shape.1);
        let y_end = (y + h).min(shape.0);

        let mut symmetry_score = 0.0;
        let mut count = 0;

        for dy in y..y_end {
            let mid_x = x + w / 2;
            for dx in 0..(w / 2) {
                let left_x = x + dx;
                let right_x = mid_x + (mid_x - left_x);

                if right_x < x_end {
                    let left_val = f32::from(gray.get(dy, left_x));
                    let right_val = f32::from(gray.get(dy, right_x));
                    let diff = (left_val - right_val).abs();
                    symmetry_score += 1.0 - (diff / 255.0);
                    count += 1;
                }
            }
        }

        if count > 0 {
            symmetry_score / count as f32
        } else {
            0.0
        }
    }

    /// Analyze composition for scene classification.
    fn analyze_composition(&self, frame: &FrameBuffer) -> ShotResult<f32> {
        let shape = frame.dim();

        // Calculate edge density (landscape shots have more edges)
        let gray = self.to_grayscale(frame);
        let mut edge_count = 0;
        let total_pixels = (shape.0 * shape.1) as f32;

        for y in 1..(shape.0.saturating_sub(1)) {
            for x in 1..(shape.1.saturating_sub(1)) {
                let center = i32::from(gray.get(y, x));
                let mut grad = 0;

                for dy in -1..=1 {
                    for dx in -1..=1 {
                        if dx != 0 || dy != 0 {
                            let ny = (y as i32 + dy) as usize;
                            let nx = (x as i32 + dx) as usize;
                            grad += (center - i32::from(gray.get(ny, nx))).abs();
                        }
                    }
                }

                if grad > 100 {
                    edge_count += 1;
                }
            }
        }

        Ok(edge_count as f32 / total_pixels)
    }
}

impl Default for ShotTypeClassifier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classifier_creation() {
        let classifier = ShotTypeClassifier::new();
        assert!((classifier.confidence_threshold - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_classify_black_frame() {
        let classifier = ShotTypeClassifier::new();
        let frame = FrameBuffer::zeros(100, 100, 3);
        let result = classifier.classify(&frame);
        assert!(result.is_ok());
        if let Ok((shot_type, _)) = result {
            assert_ne!(shot_type, ShotType::ExtremeCloseUp);
        }
    }

    #[test]
    fn test_classify_uniform_frame() {
        let classifier = ShotTypeClassifier::new();
        let frame = FrameBuffer::from_elem(100, 100, 3, 128);
        let result = classifier.classify(&frame);
        assert!(result.is_ok());
    }
}
