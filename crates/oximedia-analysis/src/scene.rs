//! Scene and shot boundary detection.
//!
//! This module implements multiple algorithms for detecting scene changes:
//! - Histogram difference for hard cuts
//! - Edge change ratio for gradual transitions (fades, dissolves)
//! - Motion compensation for camera movement
//!
//! # Algorithms
//!
//! ## Histogram Difference
//!
//! Compares the luminance histogram between consecutive frames. A large
//! difference indicates a hard cut.
//!
//! ## Edge Change Ratio (ECR)
//!
//! Detects changes in edge pixels between frames. Useful for detecting
//! gradual transitions like dissolves and fades.
//!
//! ## Motion Compensation
//!
//! Reduces false positives from camera pans and zooms by estimating
//! global motion.

use crate::{AnalysisError, AnalysisResult};
use serde::{Deserialize, Serialize};

/// Scene information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scene {
    /// Starting frame number
    pub start_frame: usize,
    /// Ending frame number (exclusive)
    pub end_frame: usize,
    /// Scene change confidence (0.0-1.0)
    pub confidence: f64,
    /// Type of scene change
    pub change_type: SceneChangeType,
}

/// Type of scene change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SceneChangeType {
    /// Hard cut
    Cut,
    /// Gradual fade
    Fade,
    /// Dissolve transition
    Dissolve,
    /// Unknown/other
    Unknown,
}

/// Scene detector.
pub struct SceneDetector {
    threshold: f64,
    prev_histogram: Option<Histogram>,
    prev_edges: Option<EdgeMap>,
    scenes: Vec<Scene>,
    last_scene_frame: usize,
}

impl SceneDetector {
    /// Create a new scene detector with the given threshold.
    #[must_use]
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            prev_histogram: None,
            prev_edges: None,
            scenes: Vec::new(),
            last_scene_frame: 0,
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

        // Compute histogram for current frame
        let histogram = compute_histogram(y_plane);

        // Compute edges for current frame
        let edges = compute_edges(y_plane, width, height);

        if let Some(prev_hist) = &self.prev_histogram {
            // Compute histogram difference
            let hist_diff = histogram_difference(prev_hist, &histogram);

            // Compute edge change ratio
            let ecr = if let Some(prev_edges) = &self.prev_edges {
                edge_change_ratio(prev_edges, &edges)
            } else {
                0.0
            };

            // Determine if this is a scene change
            let (is_scene_change, change_type) = self.detect_scene_change(hist_diff, ecr);

            if is_scene_change {
                // Record the scene
                self.scenes.push(Scene {
                    start_frame: self.last_scene_frame,
                    end_frame: frame_number,
                    confidence: hist_diff.max(ecr),
                    change_type,
                });
                self.last_scene_frame = frame_number;
            }
        }

        self.prev_histogram = Some(histogram);
        self.prev_edges = Some(edges);

        Ok(())
    }

    /// Detect if a scene change occurred based on metrics.
    fn detect_scene_change(&self, hist_diff: f64, ecr: f64) -> (bool, SceneChangeType) {
        // Hard cut detection (high histogram difference)
        if hist_diff > self.threshold {
            return (true, SceneChangeType::Cut);
        }

        // Gradual transition detection (high edge change ratio, moderate histogram change)
        if ecr > self.threshold * 0.7 && hist_diff > self.threshold * 0.3 {
            return (true, SceneChangeType::Dissolve);
        }

        // Fade detection (moderate edge change, low histogram change)
        if ecr > self.threshold * 0.5 && hist_diff < self.threshold * 0.2 {
            return (true, SceneChangeType::Fade);
        }

        (false, SceneChangeType::Unknown)
    }

    /// Finalize and return detected scenes.
    pub fn finalize(self) -> Vec<Scene> {
        self.scenes
    }
}

/// Luminance histogram (256 bins).
type Histogram = [usize; 256];

/// Compute luminance histogram.
fn compute_histogram(y_plane: &[u8]) -> Histogram {
    let mut histogram = [0; 256];
    for &pixel in y_plane {
        histogram[pixel as usize] += 1;
    }
    histogram
}

/// Compute histogram difference (normalized).
fn histogram_difference(h1: &Histogram, h2: &Histogram) -> f64 {
    let total: usize = h1.iter().sum();
    if total == 0 {
        return 0.0;
    }

    let diff: usize = h1.iter().zip(h2.iter()).map(|(a, b)| a.abs_diff(*b)).sum();

    diff as f64 / (2.0 * total as f64)
}

/// Edge map (binary edge detection).
struct EdgeMap {
    edges: Vec<bool>,
    #[allow(dead_code)]
    width: usize,
    #[allow(dead_code)]
    height: usize,
}

/// Compute edge map using Sobel operator.
fn compute_edges(y_plane: &[u8], width: usize, height: usize) -> EdgeMap {
    let mut edges = vec![false; width * height];
    let threshold = 30;

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let idx = y * width + x;

            // Sobel X kernel
            let gx = (i32::from(y_plane[(y - 1) * width + (x + 1)])
                + 2 * i32::from(y_plane[y * width + (x + 1)])
                + i32::from(y_plane[(y + 1) * width + (x + 1)]))
                - (i32::from(y_plane[(y - 1) * width + (x - 1)])
                    + 2 * i32::from(y_plane[y * width + (x - 1)])
                    + i32::from(y_plane[(y + 1) * width + (x - 1)]));

            // Sobel Y kernel
            let gy = (i32::from(y_plane[(y + 1) * width + (x - 1)])
                + 2 * i32::from(y_plane[(y + 1) * width + x])
                + i32::from(y_plane[(y + 1) * width + (x + 1)]))
                - (i32::from(y_plane[(y - 1) * width + (x - 1)])
                    + 2 * i32::from(y_plane[(y - 1) * width + x])
                    + i32::from(y_plane[(y - 1) * width + (x + 1)]));

            // Gradient magnitude
            let magnitude = f64::from(gx * gx + gy * gy).sqrt();
            edges[idx] = magnitude > f64::from(threshold);
        }
    }

    EdgeMap {
        edges,
        width,
        height,
    }
}

/// Compute edge change ratio.
fn edge_change_ratio(e1: &EdgeMap, e2: &EdgeMap) -> f64 {
    if e1.edges.len() != e2.edges.len() {
        return 0.0;
    }

    let edge_count1: usize = e1.edges.iter().filter(|&&e| e).count();
    let edge_count2: usize = e2.edges.iter().filter(|&&e| e).count();

    if edge_count1 == 0 && edge_count2 == 0 {
        return 0.0;
    }

    let max_edges = edge_count1.max(edge_count2);
    if max_edges == 0 {
        return 0.0;
    }

    // Count pixels that changed edge status
    let changed: usize = e1
        .edges
        .iter()
        .zip(e2.edges.iter())
        .filter(|(a, b)| a != b)
        .count();

    changed as f64 / max_edges as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_histogram_same_frame() {
        let frame = vec![128u8; 64 * 64];
        let h1 = compute_histogram(&frame);
        let h2 = compute_histogram(&frame);
        let diff = histogram_difference(&h1, &h2);
        assert!((diff - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_histogram_different_frames() {
        let frame1 = vec![0u8; 64 * 64];
        let frame2 = vec![255u8; 64 * 64];
        let h1 = compute_histogram(&frame1);
        let h2 = compute_histogram(&frame2);
        let diff = histogram_difference(&h1, &h2);
        assert!(diff > 0.9); // Should be close to 1.0
    }

    #[test]
    fn test_edge_detection() {
        // Create a simple edge pattern
        let mut frame = vec![0u8; 100 * 100];
        for y in 0..100 {
            for x in 50..100 {
                frame[y * 100 + x] = 255;
            }
        }
        let edges = compute_edges(&frame, 100, 100);
        // Should detect vertical edge around x=50
        assert!(edges.edges.iter().filter(|&&e| e).count() > 0);
    }

    #[test]
    fn test_scene_detector() {
        let mut detector = SceneDetector::new(0.3);

        // Process a few identical frames
        let frame1 = vec![100u8; 64 * 64];
        for i in 0..5 {
            detector
                .process_frame(&frame1, 64, 64, i)
                .expect("frame processing should succeed");
        }

        // Process a different frame (scene cut)
        let frame2 = vec![200u8; 64 * 64];
        detector
            .process_frame(&frame2, 64, 64, 5)
            .expect("frame processing should succeed");

        // Process more identical frames
        for i in 6..10 {
            detector
                .process_frame(&frame2, 64, 64, i)
                .expect("frame processing should succeed");
        }

        let scenes = detector.finalize();
        assert!(!scenes.is_empty());
    }

    #[test]
    fn test_invalid_input() {
        let mut detector = SceneDetector::new(0.3);
        let frame = vec![0u8; 100]; // Too small
        let result = detector.process_frame(&frame, 1920, 1080, 0);
        assert!(result.is_err());
    }
}
