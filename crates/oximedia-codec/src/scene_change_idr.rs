//! Scene-change detection and adaptive I-frame (IDR) insertion.
//!
//! This module analyses consecutive frames and inserts an IDR/key-frame
//! whenever a scene change is detected.  The detector computes a histogram-
//! based difference metric; if the difference exceeds a configurable threshold
//! an IDR frame is forced into the bitstream.
//!
//! # Design
//!
//! - **Histogram diff** — build 256-bin luma histograms for consecutive frames
//!   and compute the L1 distance normalised to [0, 1].
//! - **Threshold** — if the normalised difference exceeds `threshold` (default
//!   0.45) the frame is flagged as a scene change.
//! - **Minimum IDR interval** — a configurable minimum number of frames between
//!   forced IDRs prevents excessive refresh after a hard cut.
//! - **Force-IDR callback** — the `SceneChangeIdrController` emits
//!   `FrameDecision::ForceIdr` when a scene change is detected so the encoder
//!   can act immediately.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the scene-change IDR controller.
#[derive(Debug, Clone)]
pub struct SceneChangeIdrConfig {
    /// Normalised histogram-diff threshold in [0.0, 1.0].
    ///
    /// Values above this trigger an IDR insertion.  Default: `0.45`.
    pub threshold: f32,
    /// Minimum number of inter frames between two forced IDRs.
    ///
    /// Prevents rapid IDR insertion after a hard cut sequence.  Default: `12`.
    pub min_idr_interval: u32,
    /// Maximum GOP length between regular (non-scene-change) IDRs.
    ///
    /// An IDR is also inserted when this limit is reached.  Default: `250`.
    pub max_gop_length: u32,
    /// If `true` the controller also inserts IDRs at the very first frame.
    /// Default: `true`.
    pub force_first_idr: bool,
}

impl Default for SceneChangeIdrConfig {
    fn default() -> Self {
        Self {
            threshold: 0.45,
            min_idr_interval: 12,
            max_gop_length: 250,
            force_first_idr: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Frame decision
// ---------------------------------------------------------------------------

/// Decision returned for each frame by the IDR controller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameDecision {
    /// Encode frame as a regular inter (P or B) frame.
    Inter,
    /// Force this frame to be an IDR / key-frame.
    ForceIdr,
}

impl FrameDecision {
    /// Returns `true` if the decision requires an IDR frame.
    #[must_use]
    pub fn is_idr(self) -> bool {
        matches!(self, Self::ForceIdr)
    }
}

// ---------------------------------------------------------------------------
// Luma histogram helper
// ---------------------------------------------------------------------------

/// 256-bin normalised luma histogram.
#[derive(Debug, Clone)]
struct LumaHistogram {
    bins: [f32; 256],
}

impl LumaHistogram {
    /// Build a histogram from a raw luma slice.
    fn from_luma(luma: &[u8]) -> Self {
        let mut counts = [0u64; 256];
        for &p in luma {
            counts[p as usize] += 1;
        }
        let n = luma.len().max(1) as f32;
        let mut bins = [0.0f32; 256];
        for (b, &c) in bins.iter_mut().zip(counts.iter()) {
            *b = c as f32 / n;
        }
        Self { bins }
    }

    /// L1 distance between two histograms, normalised to [0, 1].
    fn l1_distance(&self, other: &LumaHistogram) -> f32 {
        self.bins
            .iter()
            .zip(other.bins.iter())
            .map(|(&a, &b)| (a - b).abs())
            .sum::<f32>()
            / 2.0 // divide by 2: max L1 for normalised histograms is 2
    }
}

// ---------------------------------------------------------------------------
// Controller
// ---------------------------------------------------------------------------

/// Scene-change-aware IDR frame insertion controller.
///
/// Call [`SceneChangeIdrController::push_frame`] with the luma plane of each
/// input frame to receive a [`FrameDecision`].
#[derive(Debug)]
pub struct SceneChangeIdrController {
    cfg: SceneChangeIdrConfig,
    /// Previous frame's histogram.
    prev_hist: Option<LumaHistogram>,
    /// Number of frames since the last IDR.
    frames_since_idr: u32,
    /// Total frames processed.
    frame_count: u64,
    /// Number of scene-change IDRs inserted.
    scene_change_count: u32,
    /// Number of regular (GOP-boundary) IDRs inserted.
    regular_idr_count: u32,
    /// Log of frame indices at which IDRs were inserted.
    idr_positions: Vec<u64>,
}

impl SceneChangeIdrController {
    /// Create a new controller with the given configuration.
    #[must_use]
    pub fn new(cfg: SceneChangeIdrConfig) -> Self {
        Self {
            cfg,
            prev_hist: None,
            frames_since_idr: 0,
            frame_count: 0,
            scene_change_count: 0,
            regular_idr_count: 0,
            idr_positions: Vec::new(),
        }
    }

    /// Create a controller with default configuration.
    #[must_use]
    pub fn default_controller() -> Self {
        Self::new(SceneChangeIdrConfig::default())
    }

    /// Analyse a frame and return the encoding decision.
    ///
    /// `luma` is the raw luma plane (8-bit, row-major).  Width and height are
    /// only needed for informational purposes; the analysis works on the flat
    /// slice.
    pub fn push_frame(&mut self, luma: &[u8]) -> FrameDecision {
        let current_hist = LumaHistogram::from_luma(luma);
        let current_index = self.frame_count;
        self.frame_count += 1;

        // Always IDR the first frame if configured
        if self.cfg.force_first_idr && current_index == 0 {
            self.prev_hist = Some(current_hist);
            self.frames_since_idr = 0;
            self.regular_idr_count += 1;
            self.idr_positions.push(current_index);
            return FrameDecision::ForceIdr;
        }

        // Regular GOP-boundary IDR
        if self.frames_since_idr >= self.cfg.max_gop_length {
            self.prev_hist = Some(current_hist);
            self.frames_since_idr = 0;
            self.regular_idr_count += 1;
            self.idr_positions.push(current_index);
            return FrameDecision::ForceIdr;
        }

        // Scene-change detection
        let decision = if let Some(ref prev) = self.prev_hist {
            let diff = prev.l1_distance(&current_hist);
            if diff >= self.cfg.threshold && self.frames_since_idr >= self.cfg.min_idr_interval {
                self.scene_change_count += 1;
                self.idr_positions.push(current_index);
                FrameDecision::ForceIdr
            } else {
                FrameDecision::Inter
            }
        } else {
            // No previous histogram: first frame without force_first_idr
            self.idr_positions.push(current_index);
            FrameDecision::ForceIdr
        };

        self.prev_hist = Some(current_hist);

        if decision == FrameDecision::ForceIdr {
            self.frames_since_idr = 0;
        } else {
            self.frames_since_idr += 1;
        }

        decision
    }

    /// Total number of frames processed.
    #[must_use]
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Number of scene-change-triggered IDR insertions.
    #[must_use]
    pub fn scene_change_count(&self) -> u32 {
        self.scene_change_count
    }

    /// Number of regular (GOP-boundary) IDR insertions.
    #[must_use]
    pub fn regular_idr_count(&self) -> u32 {
        self.regular_idr_count
    }

    /// All frame indices at which IDR frames were inserted.
    #[must_use]
    pub fn idr_positions(&self) -> &[u64] {
        &self.idr_positions
    }

    /// Reset the controller state (useful when starting a new encode session).
    pub fn reset(&mut self) {
        self.prev_hist = None;
        self.frames_since_idr = 0;
        self.frame_count = 0;
        self.scene_change_count = 0;
        self.regular_idr_count = 0;
        self.idr_positions.clear();
    }

    /// Access the current configuration.
    #[must_use]
    pub fn config(&self) -> &SceneChangeIdrConfig {
        &self.cfg
    }
}

// ---------------------------------------------------------------------------
// SceneChangeDetector — simple stateless scene-change test
// ---------------------------------------------------------------------------

/// A stateless scene-change detector that classifies a single histogram
/// difference value against a configurable threshold.
///
/// This is a lightweight helper for callers that compute histogram differences
/// externally (e.g. from a motion-estimation pass) and just need to decide
/// whether to force an I-frame.
///
/// For a stateful, full-featured detector with GOP-boundary management see
/// [`SceneChangeIdrController`].
#[derive(Debug, Clone, Default)]
pub struct SceneChangeDetector;

impl SceneChangeDetector {
    /// Create a new `SceneChangeDetector`.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Classify a pre-computed histogram difference as a scene change.
    ///
    /// # Parameters
    /// - `hist_diff`  – normalised histogram distance in [0.0, 1.0].
    ///   A value of 0.0 means identical frames; 1.0 means maximally different.
    /// - `threshold`  – decision boundary; values **strictly above** the
    ///   threshold are considered scene changes.  Typical value: `0.45`.
    ///
    /// # Returns
    /// `true` if `hist_diff > threshold` (scene change detected).
    #[must_use]
    pub fn is_scene_change(hist_diff: f32, threshold: f32) -> bool {
        hist_diff > threshold
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform_luma(value: u8, size: usize) -> Vec<u8> {
        vec![value; size]
    }

    fn ramp_luma(size: usize) -> Vec<u8> {
        (0..size).map(|i| (i % 256) as u8).collect()
    }

    #[test]
    fn test_first_frame_is_idr() {
        let mut ctrl = SceneChangeIdrController::default_controller();
        let luma = uniform_luma(128, 1920 * 1080);
        let dec = ctrl.push_frame(&luma);
        assert_eq!(dec, FrameDecision::ForceIdr);
    }

    #[test]
    fn test_identical_frames_are_inter() {
        let mut ctrl = SceneChangeIdrController::default_controller();
        let luma = uniform_luma(100, 1024);
        // First frame → IDR
        ctrl.push_frame(&luma);
        // Push min_idr_interval + 1 more identical frames
        let min = ctrl.cfg.min_idr_interval as usize + 1;
        for _ in 0..min {
            let dec = ctrl.push_frame(&luma);
            assert_eq!(dec, FrameDecision::Inter);
        }
    }

    #[test]
    fn test_hard_cut_triggers_idr() {
        let cfg = SceneChangeIdrConfig {
            threshold: 0.30,
            min_idr_interval: 2,
            max_gop_length: 500,
            force_first_idr: true,
        };
        let mut ctrl = SceneChangeIdrController::new(cfg);

        // Feed several identical frames
        let dark = uniform_luma(10, 1024);
        ctrl.push_frame(&dark); // IDR
        ctrl.push_frame(&dark); // inter
        ctrl.push_frame(&dark); // inter (now frames_since_idr >= min_idr_interval)

        // Now push a completely different (bright) frame
        let bright = uniform_luma(250, 1024);
        let dec = ctrl.push_frame(&bright);
        assert_eq!(dec, FrameDecision::ForceIdr, "hard cut should force IDR");
    }

    #[test]
    fn test_gop_boundary_triggers_idr() {
        let cfg = SceneChangeIdrConfig {
            threshold: 0.99, // effectively no scene-change detection
            min_idr_interval: 0,
            max_gop_length: 5,
            force_first_idr: false,
        };
        let mut ctrl = SceneChangeIdrController::new(cfg);
        let luma = ramp_luma(1024);

        let decisions: Vec<FrameDecision> = (0..10).map(|_| ctrl.push_frame(&luma)).collect();

        // With max_gop_length = 5, IDR should appear at frame 0 (no hist) and
        // after 5 inter frames (frames_since_idr reaches 5) → frame 6.
        // Frame 0: IDR (no prev), frames_since_idr=0
        // Frames 1-5: Inter (frames_since_idr increments from 0 to 5)
        // Frame 6: frames_since_idr=5 >= max_gop_length=5 → IDR
        let idr_indices: Vec<usize> = decisions
            .iter()
            .enumerate()
            .filter(|(_, &d)| d == FrameDecision::ForceIdr)
            .map(|(i, _)| i)
            .collect();

        assert!(
            idr_indices.contains(&0),
            "frame 0 should be IDR (no prev hist)"
        );
        assert!(
            idr_indices.contains(&6),
            "frame 6 should be IDR (GOP boundary after 5 inter frames)"
        );
    }

    #[test]
    fn test_reset_clears_state() {
        let mut ctrl = SceneChangeIdrController::default_controller();
        let luma = uniform_luma(100, 64);
        ctrl.push_frame(&luma);
        ctrl.push_frame(&luma);
        ctrl.reset();
        assert_eq!(ctrl.frame_count(), 0);
        assert!(ctrl.idr_positions().is_empty());
    }

    #[test]
    fn test_idr_positions_logged() {
        let mut ctrl = SceneChangeIdrController::default_controller();
        let luma = uniform_luma(100, 64);
        ctrl.push_frame(&luma); // first IDR → pos 0
        assert_eq!(ctrl.idr_positions(), &[0]);
    }

    #[test]
    fn test_frame_decision_is_idr_helper() {
        assert!(FrameDecision::ForceIdr.is_idr());
        assert!(!FrameDecision::Inter.is_idr());
    }

    #[test]
    fn scene_change_detector_above_threshold() {
        assert!(SceneChangeDetector::is_scene_change(0.6, 0.45));
    }

    #[test]
    fn scene_change_detector_at_threshold_is_not_change() {
        // Strictly greater than threshold → at threshold is NOT a scene change
        assert!(!SceneChangeDetector::is_scene_change(0.45, 0.45));
    }

    #[test]
    fn scene_change_detector_below_threshold() {
        assert!(!SceneChangeDetector::is_scene_change(0.2, 0.45));
    }

    #[test]
    fn scene_change_detector_zero_diff() {
        assert!(!SceneChangeDetector::is_scene_change(0.0, 0.45));
    }

    #[test]
    fn scene_change_detector_max_diff() {
        assert!(SceneChangeDetector::is_scene_change(1.0, 0.0));
    }

    #[test]
    fn test_min_idr_interval_respected() {
        let cfg = SceneChangeIdrConfig {
            threshold: 0.01, // very sensitive
            min_idr_interval: 10,
            max_gop_length: 500,
            force_first_idr: true,
        };
        let mut ctrl = SceneChangeIdrController::new(cfg);

        let dark = uniform_luma(0, 512);
        let bright = uniform_luma(255, 512);

        ctrl.push_frame(&dark); // IDR
                                // Immediately switch — should be suppressed by min_idr_interval
        for _ in 0..5 {
            let dec = ctrl.push_frame(&bright);
            assert_eq!(
                dec,
                FrameDecision::Inter,
                "min_idr_interval should block early IDR"
            );
        }
    }
}
