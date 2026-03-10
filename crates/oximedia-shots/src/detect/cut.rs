//! Hard cut detection using histogram difference and edge change.

use crate::error::{ShotError, ShotResult};
use crate::frame_buffer::{FrameBuffer, GrayImage};

/// Cut detection using multiple algorithms.
pub struct CutDetector {
    /// Histogram difference threshold.
    histogram_threshold: f32,
    /// Edge change threshold.
    edge_threshold: f32,
    /// Minimum frames between cuts.
    min_frames_between: usize,
}

impl CutDetector {
    /// Create a new cut detector with default parameters.
    #[must_use]
    pub fn new() -> Self {
        Self {
            histogram_threshold: 0.3,
            edge_threshold: 0.4,
            min_frames_between: 5,
        }
    }

    /// Create a new cut detector with custom parameters.
    #[must_use]
    pub const fn with_params(
        histogram_threshold: f32,
        edge_threshold: f32,
        min_frames_between: usize,
    ) -> Self {
        Self {
            histogram_threshold,
            edge_threshold,
            min_frames_between,
        }
    }

    /// Detect cuts between two frames.
    ///
    /// # Errors
    ///
    /// Returns error if frame dimensions don't match or frames are invalid.
    pub fn detect_cut(
        &self,
        frame1: &FrameBuffer,
        frame2: &FrameBuffer,
    ) -> ShotResult<(bool, f32)> {
        if frame1.dim() != frame2.dim() {
            return Err(ShotError::InvalidFrame(
                "Frame dimensions do not match".to_string(),
            ));
        }

        // Calculate histogram difference
        let hist_diff = self.histogram_difference(frame1, frame2)?;

        // Calculate edge change
        let edge_diff = self.edge_change_ratio(frame1, frame2)?;

        // Combine metrics
        let combined_score = (hist_diff * 0.6) + (edge_diff * 0.4);

        // Determine if it's a cut
        let is_cut = hist_diff > self.histogram_threshold || edge_diff > self.edge_threshold;

        Ok((is_cut, combined_score))
    }

    /// Calculate histogram difference between two frames.
    fn histogram_difference(&self, frame1: &FrameBuffer, frame2: &FrameBuffer) -> ShotResult<f32> {
        let shape = frame1.dim();
        if shape.2 < 3 {
            return Err(ShotError::InvalidFrame(
                "Frame must have at least 3 channels".to_string(),
            ));
        }

        let mut total_diff = 0.0;
        let num_bins = 16;
        let bin_size = 256.0 / num_bins as f32;

        // Calculate histograms for each channel
        for channel in 0..3 {
            let mut hist1 = vec![0u32; num_bins as usize];
            let mut hist2 = vec![0u32; num_bins as usize];

            // Build histograms
            for y in 0..shape.0 {
                for x in 0..shape.1 {
                    let val1 = frame1.get(y, x, channel);
                    let val2 = frame2.get(y, x, channel);

                    let bin1 = (f32::from(val1) / bin_size).min((num_bins - 1) as f32) as usize;
                    let bin2 = (f32::from(val2) / bin_size).min((num_bins - 1) as f32) as usize;

                    hist1[bin1] += 1;
                    hist2[bin2] += 1;
                }
            }

            // Normalize histograms
            let total_pixels = (shape.0 * shape.1) as f32;
            let hist1_norm: Vec<f32> = hist1.iter().map(|&v| v as f32 / total_pixels).collect();
            let hist2_norm: Vec<f32> = hist2.iter().map(|&v| v as f32 / total_pixels).collect();

            // Calculate chi-square distance
            for i in 0..num_bins as usize {
                let sum = hist1_norm[i] + hist2_norm[i];
                if sum > 0.0 {
                    let diff = hist1_norm[i] - hist2_norm[i];
                    total_diff += (diff * diff) / sum;
                }
            }
        }

        Ok((total_diff / 3.0).sqrt())
    }

    /// Calculate edge change ratio between two frames.
    fn edge_change_ratio(&self, frame1: &FrameBuffer, frame2: &FrameBuffer) -> ShotResult<f32> {
        let shape = frame1.dim();

        // Convert to grayscale
        let gray1 = self.to_grayscale(frame1);
        let gray2 = self.to_grayscale(frame2);

        // Detect edges using simple gradient
        let edges1 = self.detect_edges(&gray1);
        let edges2 = self.detect_edges(&gray2);

        // Count edge pixels
        let mut edge_count1 = 0;
        let mut edge_count2 = 0;
        let mut edge_diff = 0;

        for y in 0..shape.0 {
            for x in 0..shape.1 {
                if edges1.get(y, x) > 128 {
                    edge_count1 += 1;
                }
                if edges2.get(y, x) > 128 {
                    edge_count2 += 1;
                }
                if (edges1.get(y, x) > 128) != (edges2.get(y, x) > 128) {
                    edge_diff += 1;
                }
            }
        }

        let max_edges = edge_count1.max(edge_count2);
        if max_edges == 0 {
            return Ok(0.0);
        }

        Ok(edge_diff as f32 / max_edges as f32)
    }

    /// Convert RGB frame to grayscale.
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

    /// Detect edges using Sobel operator.
    fn detect_edges(&self, gray: &GrayImage) -> GrayImage {
        let shape = gray.dim();
        let mut edges = GrayImage::zeros(shape.0, shape.1);

        // Sobel kernels
        let sobel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]];
        let sobel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]];

        for y in 1..(shape.0.saturating_sub(1)) {
            for x in 1..(shape.1.saturating_sub(1)) {
                let mut gx = 0i32;
                let mut gy = 0i32;

                for dy in 0..3 {
                    for dx in 0..3 {
                        let pixel = i32::from(gray.get(y + dy - 1, x + dx - 1));
                        gx += pixel * sobel_x[dy][dx];
                        gy += pixel * sobel_y[dy][dx];
                    }
                }

                let magnitude = ((gx * gx + gy * gy) as f32).sqrt();
                edges.set(y, x, magnitude.min(255.0) as u8);
            }
        }

        edges
    }
}

impl Default for CutDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cut_detector_creation() {
        let detector = CutDetector::new();
        assert!((detector.histogram_threshold - 0.3).abs() < f32::EPSILON);
    }

    #[test]
    fn test_identical_frames() {
        let detector = CutDetector::new();
        let frame = FrameBuffer::zeros(100, 100, 3);
        let result = detector.detect_cut(&frame, &frame);
        assert!(result.is_ok());
        if let Ok((is_cut, score)) = result {
            assert!(!is_cut);
            assert!(score < 0.1);
        }
    }

    #[test]
    fn test_different_frames() {
        let detector = CutDetector::new();
        let frame1 = FrameBuffer::zeros(100, 100, 3);
        let mut frame2 = FrameBuffer::zeros(100, 100, 3);
        // Make frame2 completely white
        frame2.fill(255);
        let result = detector.detect_cut(&frame1, &frame2);
        assert!(result.is_ok());
        if let Ok((is_cut, score)) = result {
            assert!(is_cut);
            assert!(score > 0.3);
        }
    }

    #[test]
    fn test_dimension_mismatch() {
        let detector = CutDetector::new();
        let frame1 = FrameBuffer::zeros(100, 100, 3);
        let frame2 = FrameBuffer::zeros(50, 50, 3);
        let result = detector.detect_cut(&frame1, &frame2);
        assert!(result.is_err());
    }
}
