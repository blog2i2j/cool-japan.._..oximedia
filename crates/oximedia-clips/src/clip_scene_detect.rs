//! Automatic scene detection and subclip creation at scene boundaries.
//!
//! This module provides scene-boundary detection algorithms that analyse
//! per-frame histogram data or inter-frame pixel differences to locate
//! *cuts* and *gradual transitions* in a clip. Detected boundaries can
//! then be used to automatically split a clip into subclips.
//!
//! Two detectors are provided:
//!
//! - **`ThresholdSceneDetector`** — compares successive frames by
//!   mean absolute difference.  Fast and suitable for most content.
//! - **`HistogramSceneDetector`** — compares normalised RGB histograms
//!   using histogram intersection.  More robust to small motion/noise.
//!
//! # Example
//!
//! ```rust
//! use oximedia_clips::clip_scene_detect::{ThresholdSceneDetector, SceneBoundary};
//!
//! let mut detector = ThresholdSceneDetector::new(0.3);
//! let frames: Vec<Vec<f32>> = vec![
//!     vec![0.0; 16], // frame 0 — black
//!     vec![0.0; 16], // frame 1 — black (same scene)
//!     vec![1.0; 16], // frame 2 — white (cut!)
//!     vec![0.9; 16], // frame 3 — still white
//! ];
//! let boundaries = detector.detect(&frames);
//! assert_eq!(boundaries.len(), 1);
//! assert_eq!(boundaries[0].frame_number, 2);
//! ```

#![allow(dead_code)]

use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────────────────────────
// SceneBoundary
// ─────────────────────────────────────────────────────────────────────────────

/// The type of transition detected at a scene boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransitionType {
    /// A hard cut (instantaneous transition).
    Cut,
    /// A gradual transition (dissolve, fade, wipe).
    Gradual,
}

/// A detected scene boundary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneBoundary {
    /// The *first* frame of the new scene.
    pub frame_number: u64,
    /// Difference score that triggered the boundary (scale varies by detector).
    pub score: f32,
    /// Type of transition detected.
    pub transition: TransitionType,
}

// ─────────────────────────────────────────────────────────────────────────────
// SubclipSpec — output of scene splitting
// ─────────────────────────────────────────────────────────────────────────────

/// A subclip defined by its in-point, out-point, and optional label.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubclipSpec {
    /// First frame of the subclip (inclusive).
    pub start_frame: u64,
    /// Last frame of the subclip (inclusive).
    pub end_frame: u64,
    /// Auto-generated label (e.g. `"Scene 1"`).
    pub label: String,
    /// Approximate duration in frames.
    pub duration_frames: u64,
}

impl SubclipSpec {
    fn new(start: u64, end: u64, index: usize) -> Self {
        Self {
            start_frame: start,
            end_frame: end,
            label: format!("Scene {}", index + 1),
            duration_frames: end.saturating_sub(start) + 1,
        }
    }
}

/// Split a clip into subclip specs based on detected scene boundaries.
///
/// `total_frames` is the number of frames in the clip (0-based, so the last
/// frame index is `total_frames - 1`).
///
/// Returns one [`SubclipSpec`] per detected scene, ordered chronologically.
#[must_use]
pub fn boundaries_to_subclips(
    boundaries: &[SceneBoundary],
    total_frames: u64,
) -> Vec<SubclipSpec> {
    if total_frames == 0 {
        return Vec::new();
    }
    let mut starts: Vec<u64> = std::iter::once(0)
        .chain(boundaries.iter().map(|b| b.frame_number))
        .collect();
    starts.sort_unstable();
    starts.dedup();

    starts
        .windows(2)
        .enumerate()
        .map(|(i, w)| SubclipSpec::new(w[0], w[1].saturating_sub(1), i))
        .chain(std::iter::once_with(|| {
            let last_start = *starts.last().unwrap_or(&0);
            SubclipSpec::new(last_start, total_frames - 1, starts.len() - 1)
        }))
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// ThresholdSceneDetector
// ─────────────────────────────────────────────────────────────────────────────

/// Frame-difference-based scene detector.
///
/// Each frame is represented as a flat slice of pixel values in `[0.0, 1.0]`.
/// The mean absolute difference between consecutive frames is compared with
/// `threshold`.  When the difference exceeds `threshold` a hard cut is
/// recorded.  When the difference exceeds `gradual_threshold` (if set) a
/// gradual transition is recorded instead.
pub struct ThresholdSceneDetector {
    /// Difference threshold for a hard cut (mean absolute difference).
    pub threshold: f32,
    /// Optional lower threshold for gradual-transition detection.
    pub gradual_threshold: Option<f32>,
    /// Minimum scene length in frames. Prevents spurious splits on short bursts.
    pub min_scene_length: u64,
}

impl ThresholdSceneDetector {
    /// Create a detector with a hard-cut threshold.
    #[must_use]
    pub fn new(threshold: f32) -> Self {
        Self {
            threshold,
            gradual_threshold: None,
            min_scene_length: 1,
        }
    }

    /// Set the gradual-transition threshold (must be < `threshold`).
    #[must_use]
    pub fn with_gradual_threshold(mut self, t: f32) -> Self {
        self.gradual_threshold = Some(t);
        self
    }

    /// Set the minimum scene length in frames.
    #[must_use]
    pub fn with_min_scene_length(mut self, frames: u64) -> Self {
        self.min_scene_length = frames.max(1);
        self
    }

    /// Run detection over a sequence of frames.
    ///
    /// Each element of `frames` is a flat slice of pixel values (e.g., a
    /// row-major RGB image scaled to `[0.0, 1.0]`).  The slices must all be
    /// the same length; mismatched lengths are skipped.
    #[must_use]
    pub fn detect(&self, frames: &[Vec<f32>]) -> Vec<SceneBoundary> {
        if frames.len() < 2 {
            return Vec::new();
        }
        let mut boundaries = Vec::new();
        // `None` means no boundary has been found yet; the min_scene_length
        // filter only applies *after* the first boundary is recorded.
        let mut last_boundary: Option<u64> = None;

        for i in 1..frames.len() {
            let a = &frames[i - 1];
            let b = &frames[i];
            if a.len() != b.len() || a.is_empty() {
                continue;
            }
            let n = a.len() as f32;
            let mad: f32 = a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).abs())
                .sum::<f32>()
                / n;

            let frame = i as u64;
            if let Some(prev) = last_boundary {
                if (frame - prev) < self.min_scene_length {
                    continue;
                }
            }

            let transition = if mad >= self.threshold {
                TransitionType::Cut
            } else if let Some(gt) = self.gradual_threshold {
                if mad >= gt {
                    TransitionType::Gradual
                } else {
                    continue;
                }
            } else {
                continue;
            };

            boundaries.push(SceneBoundary {
                frame_number: frame,
                score: mad,
                transition,
            });
            last_boundary = Some(frame);
        }
        boundaries
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// HistogramSceneDetector
// ─────────────────────────────────────────────────────────────────────────────

/// Histogram-intersection-based scene detector.
///
/// Each frame is represented as a pre-computed histogram (a flat slice of
/// bin counts or frequencies summing to approximately 1.0).  Histogram
/// intersection is used to measure similarity between successive frames:
///
/// ```text
///   intersection(H1, H2) = Σ min(H1[i], H2[i])
/// ```
///
/// Values close to 1.0 mean nearly identical histograms; values close to 0.0
/// mean very different histograms. A scene boundary is detected when
/// `intersection < (1 - threshold)`.
pub struct HistogramSceneDetector {
    /// A value in `[0.0, 1.0]`. Boundary detected when
    /// `intersection < (1 - threshold)`.
    pub threshold: f32,
    /// Minimum scene length in frames.
    pub min_scene_length: u64,
}

impl HistogramSceneDetector {
    /// Create a detector with the given threshold.
    #[must_use]
    pub fn new(threshold: f32) -> Self {
        Self {
            threshold: threshold.clamp(0.0, 1.0),
            min_scene_length: 1,
        }
    }

    /// Set the minimum scene length in frames.
    #[must_use]
    pub fn with_min_scene_length(mut self, frames: u64) -> Self {
        self.min_scene_length = frames.max(1);
        self
    }

    /// Run detection over a sequence of normalised histograms.
    #[must_use]
    pub fn detect(&self, histograms: &[Vec<f32>]) -> Vec<SceneBoundary> {
        if histograms.len() < 2 {
            return Vec::new();
        }
        let mut boundaries = Vec::new();
        let mut last_boundary: u64 = 0;
        let min_intersection = 1.0 - self.threshold;

        for i in 1..histograms.len() {
            let h1 = &histograms[i - 1];
            let h2 = &histograms[i];
            if h1.len() != h2.len() || h1.is_empty() {
                continue;
            }
            let intersection: f32 = h1
                .iter()
                .zip(h2.iter())
                .map(|(a, b)| a.min(*b))
                .sum();

            let frame = i as u64;
            if (frame - last_boundary) < self.min_scene_length {
                continue;
            }

            if intersection < min_intersection {
                boundaries.push(SceneBoundary {
                    frame_number: frame,
                    score: 1.0 - intersection,
                    transition: TransitionType::Cut,
                });
                last_boundary = frame;
            }
        }
        boundaries
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ─── ThresholdSceneDetector ──────────────────────────────────────────────

    #[test]
    fn test_threshold_no_boundaries_same_frames() {
        let frames = vec![vec![0.5_f32; 8]; 10];
        let det = ThresholdSceneDetector::new(0.3);
        assert!(det.detect(&frames).is_empty());
    }

    #[test]
    fn test_threshold_hard_cut_detected() {
        let mut frames: Vec<Vec<f32>> = vec![vec![0.0_f32; 16]; 3];
        frames.push(vec![1.0_f32; 16]); // abrupt change at frame 3
        frames.push(vec![1.0_f32; 16]);
        let det = ThresholdSceneDetector::new(0.3);
        let bd = det.detect(&frames);
        assert_eq!(bd.len(), 1);
        assert_eq!(bd[0].frame_number, 3);
        assert_eq!(bd[0].transition, TransitionType::Cut);
    }

    #[test]
    fn test_threshold_gradual_detected() {
        let frames = vec![
            vec![0.0_f32; 8],
            vec![0.1_f32; 8], // gradual
            vec![1.0_f32; 8], // hard cut
        ];
        let det = ThresholdSceneDetector::new(0.5).with_gradual_threshold(0.08);
        let bd = det.detect(&frames);
        // gradual at 1, cut at 2
        assert_eq!(bd.len(), 2);
        assert_eq!(bd[0].transition, TransitionType::Gradual);
        assert_eq!(bd[1].transition, TransitionType::Cut);
    }

    #[test]
    fn test_threshold_min_scene_length_suppresses_spurious() {
        let mut frames: Vec<Vec<f32>> = vec![vec![0.0_f32; 8]; 2];
        frames.push(vec![1.0_f32; 8]); // cut at 2
        frames.push(vec![0.0_f32; 8]); // would be cut at 3 — within min_scene_length=3
        frames.push(vec![1.0_f32; 8]);
        let det = ThresholdSceneDetector::new(0.3).with_min_scene_length(3);
        let bd = det.detect(&frames);
        // Only the first cut (frame 2) should be detected; frame 3 is within
        // the 3-frame minimum.
        assert_eq!(bd[0].frame_number, 2);
    }

    #[test]
    fn test_threshold_empty_input() {
        let det = ThresholdSceneDetector::new(0.3);
        assert!(det.detect(&[]).is_empty());
        assert!(det.detect(&[vec![0.0_f32]]).is_empty());
    }

    // ─── HistogramSceneDetector ──────────────────────────────────────────────

    #[test]
    fn test_histogram_no_boundaries() {
        // Identical histograms → no boundaries.
        let hist = vec![vec![0.25_f32; 4]; 5];
        let det = HistogramSceneDetector::new(0.3);
        assert!(det.detect(&hist).is_empty());
    }

    #[test]
    fn test_histogram_cut_detected() {
        // Frame 0: all-red histogram; frame 1: all-blue histogram.
        let hist = vec![
            vec![1.0_f32, 0.0, 0.0, 0.0],
            vec![0.0_f32, 0.0, 0.0, 1.0],
        ];
        let det = HistogramSceneDetector::new(0.3);
        let bd = det.detect(&hist);
        assert_eq!(bd.len(), 1);
        assert_eq!(bd[0].frame_number, 1);
    }

    // ─── boundaries_to_subclips ──────────────────────────────────────────────

    #[test]
    fn test_no_boundaries_single_subclip() {
        let subs = boundaries_to_subclips(&[], 100);
        assert_eq!(subs.len(), 1);
        assert_eq!(subs[0].start_frame, 0);
        assert_eq!(subs[0].end_frame, 99);
    }

    #[test]
    fn test_one_boundary_two_subclips() {
        let bd = vec![SceneBoundary {
            frame_number: 50,
            score: 0.9,
            transition: TransitionType::Cut,
        }];
        let subs = boundaries_to_subclips(&bd, 100);
        assert_eq!(subs.len(), 2);
        assert_eq!(subs[0].start_frame, 0);
        assert_eq!(subs[0].end_frame, 49);
        assert_eq!(subs[1].start_frame, 50);
        assert_eq!(subs[1].end_frame, 99);
    }

    #[test]
    fn test_zero_frames_empty_output() {
        let subs = boundaries_to_subclips(&[], 0);
        assert!(subs.is_empty());
    }

    #[test]
    fn test_subclip_labels() {
        let bd = vec![
            SceneBoundary { frame_number: 30, score: 0.5, transition: TransitionType::Cut },
            SceneBoundary { frame_number: 60, score: 0.5, transition: TransitionType::Cut },
        ];
        let subs = boundaries_to_subclips(&bd, 90);
        assert_eq!(subs[0].label, "Scene 1");
        assert_eq!(subs[1].label, "Scene 2");
        assert_eq!(subs[2].label, "Scene 3");
    }
}
