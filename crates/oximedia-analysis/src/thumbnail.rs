//! Intelligent thumbnail generation and representative frame selection.
//!
//! This module selects the most representative frames from video content for
//! thumbnail generation. Selection criteria include:
//! - Visual quality (sharpness, contrast)
//! - Content diversity
//! - Temporal distribution
//! - Avoiding black frames, transitions, and blur
//!
//! # Algorithm
//!
//! 1. Score all frames based on quality metrics
//! 2. Cluster frames temporally
//! 3. Select best frame from each cluster
//! 4. Ensure even temporal distribution

use crate::{AnalysisError, AnalysisResult};
use serde::{Deserialize, Serialize};

/// Thumbnail information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThumbnailInfo {
    /// Frame number
    pub frame: usize,
    /// Quality score (0.0-1.0)
    pub score: f64,
    /// Average luminance
    pub avg_luminance: f64,
    /// Contrast level
    pub contrast: f64,
    /// Sharpness level
    pub sharpness: f64,
}

/// Thumbnail selector.
pub struct ThumbnailSelector {
    target_count: usize,
    candidates: Vec<ThumbnailCandidate>,
}

#[derive(Debug, Clone)]
struct ThumbnailCandidate {
    frame: usize,
    score: f64,
    avg_luminance: f64,
    contrast: f64,
    sharpness: f64,
}

impl ThumbnailSelector {
    /// Create a new thumbnail selector.
    ///
    /// # Parameters
    ///
    /// - `target_count`: Number of thumbnails to generate
    #[must_use]
    pub fn new(target_count: usize) -> Self {
        Self {
            target_count,
            candidates: Vec::new(),
        }
    }

    /// Process a frame as a potential thumbnail candidate.
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

        // Compute frame metrics
        let avg_luminance = compute_average_luminance(y_plane);
        let contrast = compute_contrast(y_plane);
        let sharpness = compute_sharpness(y_plane, width, height);

        // Skip frames that are too dark, too bright, or have low quality
        if !(20.0..=235.0).contains(&avg_luminance) || contrast < 10.0 || sharpness < 5.0 {
            return Ok(());
        }

        // Compute overall quality score
        let score = compute_thumbnail_score(avg_luminance, contrast, sharpness);

        self.candidates.push(ThumbnailCandidate {
            frame: frame_number,
            score,
            avg_luminance,
            contrast,
            sharpness,
        });

        Ok(())
    }

    /// Finalize and return selected thumbnails.
    pub fn finalize(mut self) -> Vec<ThumbnailInfo> {
        if self.candidates.is_empty() {
            return Vec::new();
        }

        // Sort candidates by score
        self.candidates.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Select top frames with temporal diversity
        let mut selected = Vec::new();
        let min_frame_distance = if self.candidates.len() > self.target_count {
            self.candidates.len() / self.target_count
        } else {
            1
        };

        for candidate in &self.candidates {
            // Check if this frame is far enough from already selected frames
            let too_close = selected.iter().any(|sel: &ThumbnailCandidate| {
                candidate.frame.abs_diff(sel.frame) < min_frame_distance
            });

            if !too_close {
                selected.push(candidate.clone());
                if selected.len() >= self.target_count {
                    break;
                }
            }
        }

        // Convert to ThumbnailInfo and sort by frame number
        let mut thumbnails: Vec<_> = selected
            .into_iter()
            .map(|c| ThumbnailInfo {
                frame: c.frame,
                score: c.score,
                avg_luminance: c.avg_luminance,
                contrast: c.contrast,
                sharpness: c.sharpness,
            })
            .collect();

        thumbnails.sort_by_key(|t| t.frame);
        thumbnails
    }
}

/// Compute average luminance.
fn compute_average_luminance(y_plane: &[u8]) -> f64 {
    if y_plane.is_empty() {
        return 0.0;
    }
    let sum: usize = y_plane.iter().map(|&p| p as usize).sum();
    sum as f64 / y_plane.len() as f64
}

/// Compute contrast (standard deviation of luminance).
fn compute_contrast(y_plane: &[u8]) -> f64 {
    if y_plane.is_empty() {
        return 0.0;
    }

    let avg = compute_average_luminance(y_plane);
    let variance: f64 = y_plane
        .iter()
        .map(|&p| {
            let diff = f64::from(p) - avg;
            diff * diff
        })
        .sum::<f64>()
        / y_plane.len() as f64;

    variance.sqrt()
}

/// Compute sharpness using Laplacian variance.
fn compute_sharpness(y_plane: &[u8], width: usize, height: usize) -> f64 {
    let mut laplacian_sum = 0.0;
    let mut count = 0;

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

    laplacian_sum / f64::from(count)
}

/// Compute overall thumbnail quality score.
fn compute_thumbnail_score(avg_luminance: f64, contrast: f64, sharpness: f64) -> f64 {
    // Prefer mid-range luminance (around 128)
    let luminance_score = 1.0 - ((avg_luminance - 128.0).abs() / 128.0);

    // Normalize contrast and sharpness
    let contrast_score = (contrast / 80.0).min(1.0);
    let sharpness_score = (sharpness / 50.0).min(1.0);

    // Weighted combination
    (luminance_score * 0.3 + contrast_score * 0.35 + sharpness_score * 0.35)
        .max(0.0)
        .min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_average_luminance() {
        let frame = vec![128u8; 64 * 64];
        let avg = compute_average_luminance(&frame);
        assert!((avg - 128.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_contrast_uniform_frame() {
        let frame = vec![128u8; 64 * 64];
        let contrast = compute_contrast(&frame);
        assert!(contrast < f64::EPSILON); // Uniform frame has zero contrast
    }

    #[test]
    fn test_contrast_varied_frame() {
        let mut frame = vec![0u8; 64 * 64];
        for i in 0..frame.len() / 2 {
            frame[i] = 255;
        }
        let contrast = compute_contrast(&frame);
        assert!(contrast > 100.0); // High contrast
    }

    #[test]
    fn test_sharpness() {
        // Create frame with edges
        let mut frame = vec![0u8; 100 * 100];
        for y in 0..100 {
            for x in 50..100 {
                frame[y * 100 + x] = 255;
            }
        }
        let sharpness = compute_sharpness(&frame, 100, 100);
        assert!(sharpness > 1.0);
    }

    #[test]
    fn test_thumbnail_selector() {
        let mut selector = ThumbnailSelector::new(5);

        // Add some good quality frames
        let good_frame = vec![128u8; 64 * 64];
        for i in (0..20).step_by(2) {
            selector
                .process_frame(&good_frame, 64, 64, i)
                .expect("frame processing should succeed");
        }

        // Add some bad quality frames (too dark)
        let bad_frame = vec![10u8; 64 * 64];
        selector
            .process_frame(&bad_frame, 64, 64, 10)
            .expect("frame processing should succeed");

        let thumbnails = selector.finalize();
        assert!(thumbnails.len() <= 5);

        // Thumbnails should be sorted by frame number
        for i in 1..thumbnails.len() {
            assert!(thumbnails[i].frame > thumbnails[i - 1].frame);
        }
    }

    #[test]
    fn test_empty_selector() {
        let selector = ThumbnailSelector::new(5);
        let thumbnails = selector.finalize();
        assert!(thumbnails.is_empty());
    }

    #[test]
    fn test_thumbnail_score() {
        // Perfect score conditions
        let score1 = compute_thumbnail_score(128.0, 60.0, 40.0);
        assert!(score1 > 0.8);

        // Poor score conditions
        let score2 = compute_thumbnail_score(10.0, 5.0, 2.0);
        assert!(score2 < 0.3);
    }

    #[test]
    fn test_temporal_diversity() {
        let mut selector = ThumbnailSelector::new(3);

        // Add many frames
        let frame = vec![128u8; 64 * 64];
        for i in 0..20 {
            selector
                .process_frame(&frame, 64, 64, i)
                .expect("frame processing should succeed");
        }

        let thumbnails = selector.finalize();
        assert!(thumbnails.len() <= 3);

        // Check that selected frames are spread out
        if thumbnails.len() >= 2 {
            let min_distance = thumbnails
                .windows(2)
                .map(|w| w[1].frame - w[0].frame)
                .min()
                .unwrap_or(0);
            assert!(min_distance > 2);
        }
    }
}
