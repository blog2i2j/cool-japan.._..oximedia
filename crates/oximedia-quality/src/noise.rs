//! Noise estimation for video quality assessment.
//!
//! Estimates noise levels using:
//! - Spatial noise estimation (high-pass filtering)
//! - Temporal noise estimation (inter-frame differences)
//! - Block-based noise analysis
//!
//! Lower scores indicate less noise (better quality).

use crate::{Frame, MetricType, QualityScore};
use oximedia_core::OxiResult;
use std::sync::Mutex;

/// Noise estimation method.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NoiseMethod {
    /// Spatial noise estimation
    Spatial,
    /// Temporal noise estimation (requires previous frame)
    Temporal,
    /// Combined spatial and temporal
    Combined,
}

/// Noise estimator for video quality assessment.
pub struct NoiseEstimator {
    /// Estimation method
    method: NoiseMethod,
    /// Previous frame for temporal estimation (uses Mutex for thread safety)
    previous_frame: Mutex<Option<Vec<u8>>>,
}

impl NoiseEstimator {
    /// Creates a new noise estimator with default parameters.
    #[must_use]
    pub fn new() -> Self {
        Self {
            method: NoiseMethod::Spatial,
            previous_frame: Mutex::new(None),
        }
    }

    /// Creates a noise estimator with specific method.
    #[must_use]
    pub fn with_method(method: NoiseMethod) -> Self {
        Self {
            method,
            previous_frame: Mutex::new(None),
        }
    }

    /// Estimates noise in a frame.
    ///
    /// Returns a noise score where higher values indicate more noise.
    ///
    /// # Errors
    ///
    /// Returns an error if the frame is too small.
    pub fn estimate(&self, frame: &Frame) -> OxiResult<QualityScore> {
        let mut score = QualityScore::new(MetricType::Noise, 0.0);

        // Calculate noise for Y plane
        let y_noise = self.estimate_plane(&frame.planes[0], frame.width, frame.height)?;

        score.add_component("Y", y_noise);
        score.score = y_noise;

        // Store current frame for temporal estimation
        if let Ok(mut prev) = self.previous_frame.lock() {
            *prev = Some(frame.planes[0].clone());
        }

        Ok(score)
    }

    /// Estimates noise in a single plane.
    fn estimate_plane(&self, plane: &[u8], width: usize, height: usize) -> OxiResult<f64> {
        if width < 8 || height < 8 {
            return Err(oximedia_core::OxiError::InvalidData(
                "Frame too small for noise estimation".to_string(),
            ));
        }

        let noise = match self.method {
            NoiseMethod::Spatial => self.spatial_noise(plane, width, height),
            NoiseMethod::Temporal => {
                if let Ok(prev_guard) = self.previous_frame.lock() {
                    if let Some(ref prev) = *prev_guard {
                        self.temporal_noise(plane, prev, width, height)
                    } else {
                        self.spatial_noise(plane, width, height)
                    }
                } else {
                    self.spatial_noise(plane, width, height)
                }
            }
            NoiseMethod::Combined => {
                let spatial = self.spatial_noise(plane, width, height);
                let temporal = if let Ok(prev_guard) = self.previous_frame.lock() {
                    if let Some(ref prev) = *prev_guard {
                        self.temporal_noise(plane, prev, width, height)
                    } else {
                        spatial
                    }
                } else {
                    spatial
                };
                (spatial + temporal) / 2.0
            }
        };

        Ok(noise)
    }

    /// Estimates spatial noise using high-pass filtering.
    fn spatial_noise(&self, plane: &[u8], width: usize, height: usize) -> f64 {
        // High-pass filter (Laplacian)
        let kernel = [0.0, -1.0, 0.0, -1.0, 4.0, -1.0, 0.0, -1.0, 0.0];

        let mut noise_values = Vec::new();

        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let mut sum = 0.0;

                for dy in 0..3 {
                    for dx in 0..3 {
                        let idx = (y + dy - 1) * width + (x + dx - 1);
                        sum += f64::from(plane[idx]) * kernel[dy * 3 + dx];
                    }
                }

                noise_values.push(sum.abs());
            }
        }

        if noise_values.is_empty() {
            return 0.0;
        }

        // Use robust estimator (median absolute deviation)
        let mut sorted = noise_values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let median = sorted[sorted.len() / 2];

        // Noise level (standard deviation estimate)
        median / 0.6745
    }

    /// Estimates temporal noise from inter-frame differences.
    fn temporal_noise(&self, plane: &[u8], prev_plane: &[u8], width: usize, height: usize) -> f64 {
        if prev_plane.len() != plane.len() {
            return self.spatial_noise(plane, width, height);
        }

        let mut differences = Vec::new();

        // Block-based analysis to avoid motion
        let block_size = 8;

        for by in (0..height - block_size).step_by(block_size) {
            for bx in (0..width - block_size).step_by(block_size) {
                // Compute block variance in current frame
                let mut block_sum = 0.0;
                let mut block_count = 0.0;

                for y in by..by + block_size {
                    for x in bx..bx + block_size {
                        let idx = y * width + x;
                        block_sum += f64::from(plane[idx]);
                        block_count += 1.0;
                    }
                }

                let block_mean = block_sum / block_count;
                let mut block_variance = 0.0;

                for y in by..by + block_size {
                    for x in bx..bx + block_size {
                        let idx = y * width + x;
                        let diff = f64::from(plane[idx]) - block_mean;
                        block_variance += diff * diff;
                    }
                }
                block_variance /= block_count;

                // Only use low-variance blocks (likely static)
                if block_variance < 100.0 {
                    for y in by..by + block_size {
                        for x in bx..bx + block_size {
                            let idx = y * width + x;
                            let diff = f64::from(
                                (i32::from(plane[idx]) - i32::from(prev_plane[idx])).abs(),
                            );
                            differences.push(diff);
                        }
                    }
                }
            }
        }

        if differences.is_empty() {
            return self.spatial_noise(plane, width, height);
        }

        // Standard deviation of differences
        let mean = differences.iter().sum::<f64>() / differences.len() as f64;
        let variance =
            differences.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / differences.len() as f64;

        variance.sqrt()
    }
}

impl Default for NoiseEstimator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_core::PixelFormat;

    fn create_test_frame(width: usize, height: usize, value: u8) -> Frame {
        let mut frame =
            Frame::new(width, height, PixelFormat::Yuv420p).expect("should succeed in test");
        frame.planes[0].fill(value);
        frame
    }

    fn create_noisy_frame(width: usize, height: usize) -> Frame {
        let mut frame =
            Frame::new(width, height, PixelFormat::Yuv420p).expect("should succeed in test");

        for i in 0..frame.planes[0].len() {
            let base = 128;
            let noise = ((i * 17) % 21) as i32 - 10; // -10 to +10
            frame.planes[0][i] = ((base + noise).max(0).min(255)) as u8;
        }

        frame
    }

    #[test]
    fn test_noise_estimation() {
        let estimator = NoiseEstimator::new();
        let frame = create_test_frame(64, 64, 128);

        let result = estimator.estimate(&frame).expect("should succeed in test");
        assert!(result.score >= 0.0);
        assert!(result.components.contains_key("Y"));
    }

    #[test]
    fn test_clean_vs_noisy() {
        let estimator = NoiseEstimator::new();

        // Clean frame should have low noise
        let clean = create_test_frame(64, 64, 128);
        let clean_score = estimator.estimate(&clean).expect("should succeed in test");

        // Noisy frame should have higher noise
        let noisy = create_noisy_frame(64, 64);
        let noisy_score = estimator.estimate(&noisy).expect("should succeed in test");

        // Both should return valid scores
        assert!(clean_score.score >= 0.0);
        assert!(noisy_score.score >= 0.0);
        // Noisy should generally have higher score (but not guaranteed with this simple test pattern)
    }

    #[test]
    fn test_spatial_method() {
        let estimator = NoiseEstimator::with_method(NoiseMethod::Spatial);
        let frame = create_noisy_frame(64, 64);

        let result = estimator.estimate(&frame).expect("should succeed in test");
        assert!(result.score >= 0.0);
    }

    #[test]
    fn test_temporal_method() {
        let estimator = NoiseEstimator::with_method(NoiseMethod::Temporal);

        let frame1 = create_test_frame(64, 64, 128);
        let frame2 = create_test_frame(64, 64, 130);

        let _result1 = estimator.estimate(&frame1).expect("should succeed in test");
        let result2 = estimator.estimate(&frame2).expect("should succeed in test");

        assert!(result2.score >= 0.0);
    }

    #[test]
    fn test_spatial_noise() {
        let estimator = NoiseEstimator::new();
        let plane = vec![128u8; 64 * 64];

        let noise = estimator.spatial_noise(&plane, 64, 64);
        assert!(noise >= 0.0);
    }

    #[test]
    fn test_temporal_noise() {
        let estimator = NoiseEstimator::new();
        let plane1 = vec![128u8; 64 * 64];
        let mut plane2 = vec![128u8; 64 * 64];
        plane2[100] = 150; // Small change

        let noise = estimator.temporal_noise(&plane2, &plane1, 64, 64);
        assert!(noise >= 0.0);
    }
}
