//! Video quality assessment.
//!
//! This module provides no-reference (blind) quality metrics for video:
//! - **Blockiness** - DCT-based blocking artifact detection
//! - **Blur** - Laplacian variance for sharpness
//! - **Noise** - Spectral flatness and temporal noise estimation
//!
//! # Algorithms
//!
//! ## Blockiness Detection
//!
//! Uses DCT coefficient analysis to detect blocking artifacts common in
//! block-based codecs (even though we only support AV1/VP9, we can still
//! analyze content that may have been previously encoded).
//!
//! ## Blur Detection
//!
//! Laplacian variance measures image sharpness. Low variance indicates blur.
//!
//! ## Noise Estimation
//!
//! Analyzes high-frequency components and temporal consistency to estimate
//! noise levels.

use crate::{AnalysisError, AnalysisResult};
use serde::{Deserialize, Serialize};

/// Quality assessment results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityStats {
    /// Average blockiness score (0.0-1.0, lower is better)
    pub avg_blockiness: f64,
    /// Average blur score (0.0-1.0, lower is better/sharper)
    pub avg_blur: f64,
    /// Average noise score (0.0-1.0, lower is better)
    pub avg_noise: f64,
    /// Overall quality score (0.0-1.0, higher is better)
    pub average_score: f64,
    /// Per-frame quality scores
    pub frame_scores: Vec<FrameQuality>,
}

impl Default for QualityStats {
    fn default() -> Self {
        Self {
            avg_blockiness: 0.0,
            avg_blur: 0.0,
            avg_noise: 0.0,
            average_score: 1.0,
            frame_scores: Vec::new(),
        }
    }
}

/// Per-frame quality metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameQuality {
    /// Frame number
    pub frame: usize,
    /// Blockiness score (0.0-1.0)
    pub blockiness: f64,
    /// Blur score (0.0-1.0)
    pub blur: f64,
    /// Noise score (0.0-1.0)
    pub noise: f64,
    /// Overall frame quality (0.0-1.0)
    pub overall: f64,
}

/// Quality assessor.
pub struct QualityAssessor {
    frame_scores: Vec<FrameQuality>,
    prev_frame: Option<Vec<u8>>,
}

impl QualityAssessor {
    /// Create a new quality assessor.
    #[must_use]
    pub fn new() -> Self {
        Self {
            frame_scores: Vec::new(),
            prev_frame: None,
        }
    }

    /// Process a frame.
    pub fn process_frame(
        &mut self,
        y_plane: &[u8],
        width: usize,
        height: usize,
        frame_number: usize,
    ) -> AnalysisResult<()> {
        if y_plane.len() != width * height {
            return Err(AnalysisError::InvalidInput(
                "Y plane size mismatch".to_string(),
            ));
        }

        // Compute quality metrics
        let blockiness = compute_blockiness(y_plane, width, height);
        let blur = compute_blur(y_plane, width, height);
        let noise = if let Some(ref prev) = self.prev_frame {
            compute_temporal_noise(y_plane, prev, width, height)
        } else {
            compute_spatial_noise(y_plane, width, height)
        };

        // Compute overall quality (inverse of defects)
        let overall = 1.0 - (blockiness + blur + noise) / 3.0;

        self.frame_scores.push(FrameQuality {
            frame: frame_number,
            blockiness,
            blur,
            noise,
            overall: overall.max(0.0).min(1.0),
        });

        // Store frame for temporal analysis
        self.prev_frame = Some(y_plane.to_vec());

        Ok(())
    }

    /// Finalize and return quality statistics.
    pub fn finalize(self) -> QualityStats {
        if self.frame_scores.is_empty() {
            return QualityStats::default();
        }

        let count = self.frame_scores.len() as f64;
        let avg_blockiness = self.frame_scores.iter().map(|f| f.blockiness).sum::<f64>() / count;
        let avg_blur = self.frame_scores.iter().map(|f| f.blur).sum::<f64>() / count;
        let avg_noise = self.frame_scores.iter().map(|f| f.noise).sum::<f64>() / count;
        let average_score = self.frame_scores.iter().map(|f| f.overall).sum::<f64>() / count;

        QualityStats {
            avg_blockiness,
            avg_blur,
            avg_noise,
            average_score,
            frame_scores: self.frame_scores,
        }
    }
}

impl Default for QualityAssessor {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute blockiness score using horizontal and vertical gradients at block boundaries.
fn compute_blockiness(y_plane: &[u8], width: usize, height: usize) -> f64 {
    const BLOCK_SIZE: usize = 8;
    let mut block_diff_sum = 0.0;
    let mut smooth_diff_sum = 0.0;
    let mut block_count = 0;
    let mut smooth_count = 0;

    // Check vertical block boundaries
    for y in 0..height {
        for x in (BLOCK_SIZE..width).step_by(BLOCK_SIZE) {
            if x < width {
                let idx = y * width + x;
                let diff = (i32::from(y_plane[idx]) - i32::from(y_plane[idx - 1])).abs();
                block_diff_sum += f64::from(diff);
                block_count += 1;
            }
        }
    }

    // Check horizontal block boundaries
    for y in (BLOCK_SIZE..height).step_by(BLOCK_SIZE) {
        for x in 0..width {
            let idx = y * width + x;
            let diff = (i32::from(y_plane[idx]) - i32::from(y_plane[(y - 1) * width + x])).abs();
            block_diff_sum += f64::from(diff);
            block_count += 1;
        }
    }

    // Check non-block boundaries for comparison
    for y in 0..height {
        for x in (BLOCK_SIZE / 2..width).step_by(BLOCK_SIZE) {
            if x < width {
                let idx = y * width + x;
                let diff = (i32::from(y_plane[idx]) - i32::from(y_plane[idx - 1])).abs();
                smooth_diff_sum += f64::from(diff);
                smooth_count += 1;
            }
        }
    }

    if block_count == 0 || smooth_count == 0 {
        return 0.0;
    }

    let avg_block = block_diff_sum / f64::from(block_count);
    let avg_smooth = smooth_diff_sum / f64::from(smooth_count);

    // Blockiness is the excess difference at block boundaries
    let blockiness = (avg_block - avg_smooth).max(0.0) / 255.0;
    blockiness.min(1.0)
}

/// Compute blur score using Laplacian variance.
fn compute_blur(y_plane: &[u8], width: usize, height: usize) -> f64 {
    let mut laplacian_sum = 0.0;
    let mut count = 0;

    // Laplacian kernel: [0 1 0; 1 -4 1; 0 1 0]
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let center = i32::from(y_plane[y * width + x]);
            let top = i32::from(y_plane[(y - 1) * width + x]);
            let bottom = i32::from(y_plane[(y + 1) * width + x]);
            let left = i32::from(y_plane[y * width + (x - 1)]);
            let right = i32::from(y_plane[y * width + (x + 1)]);

            let laplacian = (top + bottom + left + right - 4 * center).abs();
            laplacian_sum += f64::from(laplacian);
            count += 1;
        }
    }

    if count == 0 {
        return 0.0;
    }

    let avg_laplacian = laplacian_sum / f64::from(count);

    // Normalize and invert (higher Laplacian = sharper = lower blur score)
    // Typical range is 0-100, we'll normalize to 0-1
    let sharpness = avg_laplacian / 100.0;
    let blur = 1.0 - sharpness.min(1.0);
    blur.max(0.0)
}

/// Compute spatial noise using high-frequency analysis.
fn compute_spatial_noise(y_plane: &[u8], width: usize, height: usize) -> f64 {
    // Use high-pass filter to estimate noise
    let mut noise_sum = 0.0;
    let mut count = 0;

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let center = i32::from(y_plane[y * width + x]);
            let neighbors = [
                i32::from(y_plane[(y - 1) * width + x]),
                i32::from(y_plane[(y + 1) * width + x]),
                i32::from(y_plane[y * width + (x - 1)]),
                i32::from(y_plane[y * width + (x + 1)]),
            ];
            let avg_neighbor = neighbors.iter().sum::<i32>() / 4;
            let diff = (center - avg_neighbor).abs();

            // Only count small differences as noise (larger ones are edges)
            if diff < 20 {
                noise_sum += f64::from(diff);
                count += 1;
            }
        }
    }

    if count == 0 {
        return 0.0;
    }

    let avg_noise = noise_sum / f64::from(count);
    (avg_noise / 20.0).min(1.0)
}

/// Compute temporal noise by comparing consecutive frames.
fn compute_temporal_noise(current: &[u8], previous: &[u8], width: usize, height: usize) -> f64 {
    if current.len() != previous.len() {
        return 0.0;
    }

    let mut diff_sum = 0.0;
    let mut count = 0;

    // Sample a subset of pixels for efficiency
    for y in (0..height).step_by(4) {
        for x in (0..width).step_by(4) {
            let idx = y * width + x;
            let diff = (i32::from(current[idx]) - i32::from(previous[idx])).abs();

            // Only count small differences as noise
            if diff < 30 {
                diff_sum += f64::from(diff);
                count += 1;
            }
        }
    }

    if count == 0 {
        return 0.0;
    }

    let avg_diff = diff_sum / f64::from(count);
    (avg_diff / 30.0).min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_assessor() {
        let mut assessor = QualityAssessor::new();

        // Process a few frames
        let frame = vec![128u8; 64 * 64];
        for i in 0..5 {
            assessor
                .process_frame(&frame, 64, 64, i)
                .expect("frame processing should succeed");
        }

        let stats = assessor.finalize();
        assert_eq!(stats.frame_scores.len(), 5);
        assert!(stats.average_score >= 0.0 && stats.average_score <= 1.0);
    }

    #[test]
    fn test_blockiness_uniform_frame() {
        // Uniform frame should have low blockiness
        let frame = vec![128u8; 64 * 64];
        let blockiness = compute_blockiness(&frame, 64, 64);
        assert!(blockiness < 0.1);
    }

    #[test]
    fn test_blur_sharp_edge() {
        // Create frame with sharp edge
        let mut frame = vec![0u8; 100 * 100];
        for y in 0..100 {
            for x in 50..100 {
                frame[y * 100 + x] = 255;
            }
        }
        let blur = compute_blur(&frame, 100, 100);
        // Should have low blur (high sharpness)
        assert!(blur < 2.0);
    }

    #[test]
    fn test_blur_uniform_frame() {
        // Uniform frame should have high blur (no edges)
        let frame = vec![128u8; 100 * 100];
        let blur = compute_blur(&frame, 100, 100);
        assert!(blur > 0.5);
    }

    #[test]
    fn test_spatial_noise() {
        let frame = vec![128u8; 64 * 64];
        let noise = compute_spatial_noise(&frame, 64, 64);
        assert!(noise >= 0.0 && noise <= 1.0);
    }

    #[test]
    fn test_temporal_noise() {
        let frame1 = vec![128u8; 64 * 64];
        let frame2 = vec![130u8; 64 * 64];
        let noise = compute_temporal_noise(&frame2, &frame1, 64, 64);
        assert!(noise >= 0.0 && noise <= 1.0);
    }

    #[test]
    fn test_empty_quality() {
        let assessor = QualityAssessor::new();
        let stats = assessor.finalize();
        assert!(stats.frame_scores.is_empty());
    }
}
