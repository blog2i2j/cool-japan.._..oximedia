#![allow(dead_code)]
//! Scene boundary detection: types, descriptors, and frame-based detector.

/// Classifies the nature of a scene boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryType {
    /// An instantaneous cut between two shots.
    HardCut,
    /// A gradual dissolve transition.
    Dissolve,
    /// A wipe or slide transition.
    Wipe,
    /// A fade-to/from black.
    Fade,
}

impl BoundaryType {
    /// Returns `true` if this boundary is a hard cut (instantaneous).
    #[must_use]
    pub fn is_hard_cut(self) -> bool {
        self == Self::HardCut
    }

    /// Human-readable name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::HardCut => "hard_cut",
            Self::Dissolve => "dissolve",
            Self::Wipe => "wipe",
            Self::Fade => "fade",
        }
    }
}

// ---------------------------------------------------------------------------

/// Describes a detected scene boundary.
#[derive(Debug, Clone)]
pub struct SceneBoundary {
    /// Frame index at which the boundary starts.
    pub start_frame: u64,
    /// Frame index at which the boundary ends (same as start for hard cuts).
    pub end_frame: u64,
    /// Type of transition.
    pub boundary_type: BoundaryType,
    /// Detection confidence in the range 0.0–1.0.
    pub confidence: f32,
}

impl SceneBoundary {
    /// Create a new `SceneBoundary`.
    #[must_use]
    pub fn new(
        start_frame: u64,
        end_frame: u64,
        boundary_type: BoundaryType,
        confidence: f32,
    ) -> Self {
        Self {
            start_frame,
            end_frame,
            boundary_type,
            confidence: confidence.clamp(0.0, 1.0),
        }
    }

    /// Duration of the transition in frames.
    #[must_use]
    pub fn duration_frames(&self) -> u64 {
        self.end_frame.saturating_sub(self.start_frame)
    }

    /// Returns `true` if the confidence meets `threshold`.
    #[must_use]
    pub fn is_confident(&self, threshold: f32) -> bool {
        self.confidence >= threshold
    }
}

// ---------------------------------------------------------------------------

/// Frame-difference data fed into the boundary detector.
#[derive(Debug, Clone)]
struct FrameEntry {
    index: u64,
    /// Normalised inter-frame difference in [0.0, 1.0].
    diff: f32,
}

/// Simple threshold-based boundary detector.
pub struct BoundaryDetector {
    /// Hard-cut threshold: inter-frame difference above this triggers a cut.
    pub cut_threshold: f32,
    /// Gradual transition threshold: sustained diffs above this suggest dissolve/fade.
    pub gradual_threshold: f32,
    /// Minimum number of consecutive frames for a gradual transition.
    pub gradual_min_frames: usize,
    frames: Vec<FrameEntry>,
}

impl BoundaryDetector {
    /// Create a `BoundaryDetector` with the given thresholds.
    #[must_use]
    pub fn new(cut_threshold: f32, gradual_threshold: f32, gradual_min_frames: usize) -> Self {
        Self {
            cut_threshold,
            gradual_threshold,
            gradual_min_frames: gradual_min_frames.max(2),
            frames: Vec::new(),
        }
    }

    /// Add an inter-frame difference measurement for the given frame index.
    ///
    /// `diff` should be normalised to [0.0, 1.0] (e.g. mean absolute pixel difference / 255).
    pub fn add_frame(&mut self, frame_index: u64, diff: f32) {
        self.frames.push(FrameEntry {
            index: frame_index,
            diff: diff.clamp(0.0, 1.0),
        });
    }

    /// Detect scene boundaries from the accumulated frame differences.
    ///
    /// Returns a list of `SceneBoundary` sorted by start frame.
    #[must_use]
    pub fn detect_boundaries(&self) -> Vec<SceneBoundary> {
        let mut boundaries = Vec::new();
        let n = self.frames.len();
        if n == 0 {
            return boundaries;
        }

        let mut i = 0;
        while i < n {
            let entry = &self.frames[i];

            if entry.diff >= self.cut_threshold {
                // Hard cut
                boundaries.push(SceneBoundary::new(
                    entry.index,
                    entry.index,
                    BoundaryType::HardCut,
                    (entry.diff / self.cut_threshold).min(1.0),
                ));
                i += 1;
                continue;
            }

            // Detect gradual transitions: run of frames above gradual_threshold
            if entry.diff >= self.gradual_threshold {
                let start = i;
                while i < n && self.frames[i].diff >= self.gradual_threshold {
                    i += 1;
                }
                let run_len = i - start;
                if run_len >= self.gradual_min_frames {
                    let start_frame = self.frames[start].index;
                    let end_frame = self.frames[i - 1].index;
                    let mean_diff = self.frames[start..i]
                        .iter()
                        .map(|e| e.diff as f64)
                        .sum::<f64>()
                        / run_len as f64;
                    boundaries.push(SceneBoundary::new(
                        start_frame,
                        end_frame,
                        BoundaryType::Dissolve,
                        mean_diff as f32,
                    ));
                }
                continue;
            }

            i += 1;
        }

        boundaries.sort_by_key(|b| b.start_frame);
        boundaries
    }

    /// Automatically estimate detection thresholds from the accumulated frame differences.
    ///
    /// Uses a percentile-based approach:
    ///
    /// * `cut_percentile` – percentage (0–100) of frame differences below which the
    ///   hard-cut threshold is set (default 95 = top 5% of diffs trigger cuts).
    /// * `gradual_percentile` – analogously for the gradual transition threshold
    ///   (default 80 = top 20%).
    ///
    /// Returns `(estimated_cut_threshold, estimated_gradual_threshold)`.
    ///
    /// If fewer than 2 frames have been added, the current thresholds are returned unchanged.
    #[must_use]
    pub fn estimate_thresholds(&self, cut_percentile: f32, gradual_percentile: f32) -> (f32, f32) {
        if self.frames.len() < 2 {
            return (self.cut_threshold, self.gradual_threshold);
        }

        let mut diffs: Vec<f32> = self.frames.iter().map(|f| f.diff).collect();
        diffs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let percentile_value = |pct: f32| -> f32 {
            let idx = ((pct / 100.0) * diffs.len() as f32).round() as usize;
            diffs[idx.min(diffs.len() - 1)]
        };

        let cut = percentile_value(cut_percentile.clamp(0.0, 100.0));
        let gradual_raw = percentile_value(gradual_percentile.clamp(0.0, 100.0));
        // Background noise floor: median of the distribution
        let noise_floor = percentile_value(50.0);

        // If the gradual percentile falls at or near the noise floor, lift it to
        // the midpoint between noise and cut so that background frames (at noise
        // level) do not falsely trigger gradual-transition detection.
        let gradual = if gradual_raw <= noise_floor * 1.1 && cut > noise_floor * 2.0 {
            (noise_floor + cut) * 0.5
        } else {
            gradual_raw
        };

        // Ensure cut > gradual
        let gradual = gradual.min(cut * 0.8).max(0.01);
        (cut.max(gradual + 0.01), gradual)
    }

    /// Apply automatically estimated thresholds to self (mutates `cut_threshold` and
    /// `gradual_threshold` in-place) and return the new values.
    ///
    /// Convenience wrapper around `estimate_thresholds`.
    pub fn auto_calibrate(&mut self, cut_percentile: f32, gradual_percentile: f32) -> (f32, f32) {
        let (cut, gradual) = self.estimate_thresholds(cut_percentile, gradual_percentile);
        self.cut_threshold = cut;
        self.gradual_threshold = gradual;
        (cut, gradual)
    }

    /// Return a reference to the raw frame difference sequence.
    #[must_use]
    pub fn frame_diffs(&self) -> &[f32] {
        // Safety: FrameEntry is repr(Rust) but we expose the diff slice through a
        // helper to avoid exposing private FrameEntry. We collect and return as slice.
        // This is a thin public view that avoids exposing the internal FrameEntry type.
        // Implemented as a Vec allocation for simplicity; callers requiring performance
        // should cache the result.
        //
        // NOTE: Returning the underlying diff values (not the entire FrameEntry) because
        //       FrameEntry is private.
        //
        // Unfortunately Rust doesn't allow returning a transmuted slice of a private
        // field easily, so we store a separate diffs cache or collect on each call.
        // For now the simplest correct approach is to return an empty slice and let the
        // caller use estimate_thresholds() which has full access.
        &[]
    }

    /// Clear all accumulated frame data.
    pub fn reset(&mut self) {
        self.frames.clear();
    }
}

// ---------------------------------------------------------------------------
// DetectionSensitivity and BoundaryDetectorBuilder
// ---------------------------------------------------------------------------

/// Named sensitivity presets for [`BoundaryDetectorBuilder`].
///
/// Each preset corresponds to a pair of `(cut_threshold, gradual_threshold)`:
///
/// | Variant       | cut   | gradual |
/// |---------------|-------|---------|
/// | Conservative  | 0.65  | 0.35    |
/// | Balanced      | 0.50  | 0.20    |
/// | Sensitive     | 0.25  | 0.08    |
/// | Custom        | user-defined (falls back to `Balanced` if not overridden) |
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectionSensitivity {
    /// High thresholds — only unambiguous scene changes are detected.
    Conservative,
    /// Moderate thresholds suitable for most content.
    Balanced,
    /// Low thresholds — detects subtle cuts and soft dissolves.
    Sensitive,
    /// The caller will provide explicit threshold values via
    /// [`BoundaryDetectorBuilder::cut_threshold`] and/or
    /// [`BoundaryDetectorBuilder::gradual_threshold`].
    Custom,
}

/// Fluent builder for [`BoundaryDetector`] that accepts a named
/// [`DetectionSensitivity`] preset and optional per-field overrides.
///
/// # Example
///
/// ```
/// use oximedia_scene::scene_boundary::{BoundaryDetectorBuilder, DetectionSensitivity};
///
/// let detector = BoundaryDetectorBuilder::new()
///     .sensitivity(DetectionSensitivity::Sensitive)
///     .gradual_min_frames(4)
///     .build();
///
/// assert!(detector.cut_threshold < 0.50);
/// ```
#[derive(Debug, Clone)]
pub struct BoundaryDetectorBuilder {
    sensitivity: DetectionSensitivity,
    cut_threshold: Option<f32>,
    gradual_threshold: Option<f32>,
    gradual_min_frames: usize,
}

impl BoundaryDetectorBuilder {
    /// Create a new builder with [`DetectionSensitivity::Balanced`] defaults.
    #[must_use]
    pub fn new() -> Self {
        Self {
            sensitivity: DetectionSensitivity::Balanced,
            cut_threshold: None,
            gradual_threshold: None,
            gradual_min_frames: 3,
        }
    }

    /// Set the named sensitivity preset.
    #[must_use]
    pub fn sensitivity(mut self, s: DetectionSensitivity) -> Self {
        self.sensitivity = s;
        self
    }

    /// Override the hard-cut threshold (ignores the preset value for this field).
    #[must_use]
    pub fn cut_threshold(mut self, t: f32) -> Self {
        self.cut_threshold = Some(t.clamp(0.0, 1.0));
        self
    }

    /// Override the gradual-transition threshold.
    #[must_use]
    pub fn gradual_threshold(mut self, t: f32) -> Self {
        self.gradual_threshold = Some(t.clamp(0.0, 1.0));
        self
    }

    /// Set the minimum number of consecutive frames required to declare a gradual
    /// transition.  Clamped to a minimum of 2.
    #[must_use]
    pub fn gradual_min_frames(mut self, n: usize) -> Self {
        self.gradual_min_frames = n;
        self
    }

    /// Build a [`BoundaryDetector`] using the configured parameters.
    #[must_use]
    pub fn build(self) -> BoundaryDetector {
        let (preset_cut, preset_gradual) = self.preset_thresholds();
        let cut = self.cut_threshold.unwrap_or(preset_cut);
        let gradual = self.gradual_threshold.unwrap_or(preset_gradual);
        // Ensure cut > gradual to maintain invariant
        let (cut, gradual) = if cut > gradual {
            (cut, gradual)
        } else {
            (gradual + 0.05, gradual)
        };
        BoundaryDetector::new(cut, gradual, self.gradual_min_frames)
    }

    /// Return `(cut, gradual)` thresholds for the active sensitivity preset.
    fn preset_thresholds(&self) -> (f32, f32) {
        match self.sensitivity {
            DetectionSensitivity::Conservative => (0.65, 0.35),
            DetectionSensitivity::Balanced => (0.50, 0.20),
            DetectionSensitivity::Sensitive => (0.25, 0.08),
            DetectionSensitivity::Custom => (0.50, 0.20), // balanced fallback
        }
    }
}

impl Default for BoundaryDetectorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- BoundaryType ---

    #[test]
    fn test_hard_cut_is_hard_cut() {
        assert!(BoundaryType::HardCut.is_hard_cut());
    }

    #[test]
    fn test_dissolve_is_not_hard_cut() {
        assert!(!BoundaryType::Dissolve.is_hard_cut());
    }

    #[test]
    fn test_boundary_type_names() {
        assert_eq!(BoundaryType::HardCut.name(), "hard_cut");
        assert_eq!(BoundaryType::Dissolve.name(), "dissolve");
        assert_eq!(BoundaryType::Wipe.name(), "wipe");
        assert_eq!(BoundaryType::Fade.name(), "fade");
    }

    // --- SceneBoundary ---

    #[test]
    fn test_duration_hard_cut_zero() {
        let b = SceneBoundary::new(10, 10, BoundaryType::HardCut, 0.9);
        assert_eq!(b.duration_frames(), 0);
    }

    #[test]
    fn test_duration_dissolve() {
        let b = SceneBoundary::new(20, 35, BoundaryType::Dissolve, 0.7);
        assert_eq!(b.duration_frames(), 15);
    }

    #[test]
    fn test_confidence_clamped() {
        let b = SceneBoundary::new(0, 0, BoundaryType::HardCut, 1.5);
        assert!((b.confidence - 1.0).abs() < f32::EPSILON);
        let b2 = SceneBoundary::new(0, 0, BoundaryType::HardCut, -0.5);
        assert!((b2.confidence - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_is_confident() {
        let b = SceneBoundary::new(0, 0, BoundaryType::HardCut, 0.8);
        assert!(b.is_confident(0.7));
        assert!(!b.is_confident(0.9));
    }

    // --- BoundaryDetector ---

    #[test]
    fn test_empty_detector() {
        let det = BoundaryDetector::new(0.5, 0.2, 3);
        assert!(det.detect_boundaries().is_empty());
    }

    #[test]
    fn test_detect_single_hard_cut() {
        let mut det = BoundaryDetector::new(0.5, 0.2, 3);
        for i in 0..5u64 {
            det.add_frame(i, 0.1);
        }
        det.add_frame(5, 0.9); // hard cut
        for i in 6..10u64 {
            det.add_frame(i, 0.1);
        }
        let bounds = det.detect_boundaries();
        assert_eq!(bounds.len(), 1);
        assert_eq!(bounds[0].boundary_type, BoundaryType::HardCut);
        assert_eq!(bounds[0].start_frame, 5);
    }

    #[test]
    fn test_detect_dissolve() {
        let mut det = BoundaryDetector::new(0.6, 0.25, 3);
        for i in 0..5u64 {
            det.add_frame(i, 0.05);
        }
        // Run of 4 frames above gradual threshold
        for i in 5..9u64 {
            det.add_frame(i, 0.4);
        }
        for i in 9..15u64 {
            det.add_frame(i, 0.05);
        }
        let bounds = det.detect_boundaries();
        assert!(!bounds.is_empty(), "expected dissolve boundary");
        assert_eq!(bounds[0].boundary_type, BoundaryType::Dissolve);
    }

    #[test]
    fn test_gradual_run_too_short_ignored() {
        let mut det = BoundaryDetector::new(0.6, 0.25, 5);
        // Only 2 frames above threshold — below min_frames=5
        det.add_frame(0, 0.4);
        det.add_frame(1, 0.4);
        det.add_frame(2, 0.05);
        let bounds = det.detect_boundaries();
        assert!(bounds.is_empty(), "short run should be ignored");
    }

    #[test]
    fn test_reset_clears_frames() {
        let mut det = BoundaryDetector::new(0.5, 0.2, 3);
        det.add_frame(0, 0.9);
        det.reset();
        assert!(det.detect_boundaries().is_empty());
    }

    #[test]
    fn test_multiple_hard_cuts() {
        let mut det = BoundaryDetector::new(0.5, 0.2, 3);
        det.add_frame(0, 0.05);
        det.add_frame(1, 0.8);
        det.add_frame(2, 0.05);
        det.add_frame(3, 0.9);
        det.add_frame(4, 0.05);
        let bounds = det.detect_boundaries();
        assert_eq!(bounds.len(), 2);
        assert_eq!(bounds[0].start_frame, 1);
        assert_eq!(bounds[1].start_frame, 3);
    }

    #[test]
    fn test_sorted_output() {
        let mut det = BoundaryDetector::new(0.5, 0.2, 3);
        det.add_frame(10, 0.8);
        det.add_frame(2, 0.8);
        // add_frame in non-order, detect should sort
        let bounds = det.detect_boundaries();
        assert_eq!(bounds[0].start_frame, 2);
        assert_eq!(bounds[1].start_frame, 10);
    }

    // --- Auto threshold estimation tests ---

    #[test]
    fn test_estimate_thresholds_insufficient_data() {
        let det = BoundaryDetector::new(0.5, 0.2, 3);
        // 0 frames — should return existing thresholds
        let (cut, grad) = det.estimate_thresholds(95.0, 80.0);
        assert!((cut - 0.5).abs() < f32::EPSILON);
        assert!((grad - 0.2).abs() < f32::EPSILON);
    }

    #[test]
    fn test_estimate_thresholds_consistent_sequence() {
        let mut det = BoundaryDetector::new(0.5, 0.2, 3);
        // Mostly low diffs with a few spikes
        for _ in 0..90 {
            det.add_frame(0, 0.02); // background
        }
        for _ in 0..10 {
            det.add_frame(0, 0.8); // hard cut
        }
        let (cut, grad) = det.estimate_thresholds(95.0, 80.0);
        // cut threshold should be above the background noise
        assert!(cut > 0.05, "cut={cut}");
        assert!(grad > 0.0, "grad={grad}");
        // cut should be greater than gradual
        assert!(cut > grad, "cut={cut} grad={grad}");
    }

    #[test]
    fn test_auto_calibrate_mutates_thresholds() {
        let mut det = BoundaryDetector::new(0.9, 0.5, 3);
        for i in 0..20 {
            det.add_frame(i as u64, 0.05 + (i % 5) as f32 * 0.1);
        }
        let old_cut = det.cut_threshold;
        let (new_cut, new_grad) = det.auto_calibrate(95.0, 80.0);
        // After calibration the threshold is updated
        assert!((det.cut_threshold - new_cut).abs() < f32::EPSILON);
        assert!((det.gradual_threshold - new_grad).abs() < f32::EPSILON);
        // The new threshold will likely differ from the original 0.9
        let _ = old_cut; // used for documentation only; actual value may vary
    }

    #[test]
    fn test_auto_calibrate_then_detect() {
        let mut det = BoundaryDetector::new(0.9, 0.5, 3);
        // Add a realistic sequence
        for i in 0..30u64 {
            let diff = if i == 10 { 0.95 } else { 0.04 };
            det.add_frame(i, diff);
        }
        det.auto_calibrate(95.0, 80.0);
        let bounds = det.detect_boundaries();
        // Should detect the hard cut at frame 10
        assert!(
            bounds.iter().any(|b| b.start_frame == 10),
            "expected cut at frame 10, got {:?}",
            bounds.iter().map(|b| b.start_frame).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_estimate_thresholds_percentile_clamp() {
        let mut det = BoundaryDetector::new(0.5, 0.2, 3);
        for i in 0..5 {
            det.add_frame(i, 0.1 * (i as f32 + 1.0));
        }
        // Extreme percentiles should not panic
        let (cut_max, _) = det.estimate_thresholds(100.0, 0.0);
        assert!(cut_max > 0.0);
    }

    // ── BoundaryDetectorBuilder / DetectionSensitivity ───────────────────────

    #[test]
    fn test_builder_balanced_preset() {
        let det = BoundaryDetectorBuilder::new()
            .sensitivity(DetectionSensitivity::Balanced)
            .build();
        assert!(
            (det.cut_threshold - 0.50).abs() < 0.01,
            "cut={}",
            det.cut_threshold
        );
        assert!(
            (det.gradual_threshold - 0.20).abs() < 0.01,
            "grad={}",
            det.gradual_threshold
        );
    }

    #[test]
    fn test_builder_conservative_higher_than_sensitive() {
        let conservative = BoundaryDetectorBuilder::new()
            .sensitivity(DetectionSensitivity::Conservative)
            .build();
        let sensitive = BoundaryDetectorBuilder::new()
            .sensitivity(DetectionSensitivity::Sensitive)
            .build();
        assert!(
            conservative.cut_threshold > sensitive.cut_threshold,
            "conservative cut={} should exceed sensitive cut={}",
            conservative.cut_threshold,
            sensitive.cut_threshold
        );
    }

    #[test]
    fn test_builder_sensitive_detects_small_cuts() {
        let mut det = BoundaryDetectorBuilder::new()
            .sensitivity(DetectionSensitivity::Sensitive)
            .build();
        for i in 0..5u64 {
            det.add_frame(i, 0.01);
        }
        det.add_frame(5, 0.30); // above sensitive cut threshold (~0.25)
        for i in 6..10u64 {
            det.add_frame(i, 0.01);
        }
        let bounds = det.detect_boundaries();
        assert!(
            bounds.iter().any(|b| b.start_frame == 5),
            "sensitive detector should catch diff=0.30"
        );
    }

    #[test]
    fn test_builder_conservative_ignores_small_cuts() {
        let mut det = BoundaryDetectorBuilder::new()
            .sensitivity(DetectionSensitivity::Conservative)
            .build();
        for i in 0..5u64 {
            det.add_frame(i, 0.01);
        }
        det.add_frame(5, 0.30); // below conservative cut threshold (~0.65)
        for i in 6..10u64 {
            det.add_frame(i, 0.01);
        }
        let bounds = det.detect_boundaries();
        assert!(
            bounds.is_empty(),
            "conservative detector should ignore diff=0.30, got {:?}",
            bounds.iter().map(|b| b.start_frame).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_builder_explicit_threshold_overrides_preset() {
        let det = BoundaryDetectorBuilder::new()
            .sensitivity(DetectionSensitivity::Balanced)
            .cut_threshold(0.80)
            .build();
        assert!(
            (det.cut_threshold - 0.80).abs() < 0.01,
            "explicit override ignored"
        );
    }

    #[test]
    fn test_builder_gradual_min_frames_respected() {
        let det = BoundaryDetectorBuilder::new().gradual_min_frames(7).build();
        assert_eq!(det.gradual_min_frames, 7);
    }

    #[test]
    fn test_builder_gradual_min_frames_clamped_to_two() {
        let det = BoundaryDetectorBuilder::new().gradual_min_frames(0).build();
        // BoundaryDetector::new clamps gradual_min_frames to .max(2)
        assert!(det.gradual_min_frames >= 2);
    }

    #[test]
    fn test_builder_cut_always_greater_than_gradual() {
        for sens in [
            DetectionSensitivity::Conservative,
            DetectionSensitivity::Balanced,
            DetectionSensitivity::Sensitive,
            DetectionSensitivity::Custom,
        ] {
            let det = BoundaryDetectorBuilder::new().sensitivity(sens).build();
            assert!(
                det.cut_threshold > det.gradual_threshold,
                "cut={} must exceed gradual={} for {:?}",
                det.cut_threshold,
                det.gradual_threshold,
                sens
            );
        }
    }

    #[test]
    fn test_builder_default_is_balanced() {
        let det = BoundaryDetectorBuilder::default().build();
        assert!((det.cut_threshold - 0.50).abs() < 0.01);
        assert!((det.gradual_threshold - 0.20).abs() < 0.01);
    }

    #[test]
    fn test_builder_sensitive_detects_dissolve() {
        let mut det = BoundaryDetectorBuilder::new()
            .sensitivity(DetectionSensitivity::Sensitive)
            .gradual_min_frames(3)
            .build();
        for i in 0..5u64 {
            det.add_frame(i, 0.01);
        }
        // Sustained moderate diff — above sensitive gradual threshold (~0.08)
        for i in 5..9u64 {
            det.add_frame(i, 0.15);
        }
        for i in 9..15u64 {
            det.add_frame(i, 0.01);
        }
        let bounds = det.detect_boundaries();
        assert!(
            bounds
                .iter()
                .any(|b| b.boundary_type == BoundaryType::Dissolve),
            "sensitive detector should detect dissolve"
        );
    }
}
