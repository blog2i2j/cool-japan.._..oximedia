//! Scene-level quality assessment over time.
//!
//! This module provides data structures for tracking quality metrics across
//! temporal windows of a video, enabling detection of quality degradation
//! at scene or segment granularity.

#![allow(dead_code)]

/// Quality statistics for a contiguous window of frames.
#[derive(Clone, Debug)]
pub struct SceneQualityWindow {
    /// Index of the first frame in the window (inclusive).
    pub start_frame: u64,
    /// Index of the last frame in the window (inclusive).
    pub end_frame: u64,
    /// Mean quality score across the window (lower is typically better for
    /// no-reference metrics; interpretation depends on the metric used).
    pub mean_score: f32,
    /// Minimum (worst) quality score in the window.
    pub min_score: f32,
    /// Maximum (best) quality score in the window.
    pub max_score: f32,
}

impl SceneQualityWindow {
    /// Returns the number of frames in the window.
    #[must_use]
    pub fn duration_frames(&self) -> u64 {
        self.end_frame.saturating_sub(self.start_frame) + 1
    }

    /// Returns `true` if the mean score exceeds `threshold`, indicating
    /// potential quality degradation.
    #[must_use]
    pub fn is_degraded(&self, threshold: f32) -> bool {
        self.mean_score > threshold
    }

    /// Returns the range between the best and worst scores in the window.
    #[must_use]
    pub fn score_range(&self) -> f32 {
        self.max_score - self.min_score
    }
}

/// A timeline of quality windows covering an entire video.
#[derive(Clone, Debug, Default)]
pub struct QualityTimeline {
    /// Ordered list of quality windows.
    pub windows: Vec<SceneQualityWindow>,
}

impl QualityTimeline {
    /// Creates an empty timeline.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a quality window to the timeline.
    pub fn add(&mut self, window: SceneQualityWindow) {
        self.windows.push(window);
    }

    /// Returns references to all windows whose mean score exceeds `threshold`.
    #[must_use]
    pub fn degraded_windows(&self, threshold: f32) -> Vec<&SceneQualityWindow> {
        self.windows
            .iter()
            .filter(|w| w.is_degraded(threshold))
            .collect()
    }

    /// Returns the overall mean score across all windows, or `0.0` if empty.
    #[must_use]
    pub fn overall_mean(&self) -> f32 {
        if self.windows.is_empty() {
            return 0.0;
        }
        let total: f32 = self.windows.iter().map(|w| w.mean_score).sum();
        total / self.windows.len() as f32
    }

    /// Returns a reference to the window with the highest mean score (worst
    /// quality), or `None` if the timeline is empty.
    #[must_use]
    pub fn find_worst_window(&self) -> Option<&SceneQualityWindow> {
        self.windows.iter().max_by(|a, b| {
            a.mean_score
                .partial_cmp(&b.mean_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_window(start: u64, end: u64, mean: f32, min: f32, max: f32) -> SceneQualityWindow {
        SceneQualityWindow {
            start_frame: start,
            end_frame: end,
            mean_score: mean,
            min_score: min,
            max_score: max,
        }
    }

    // ── SceneQualityWindow ─────────────────────────────────────────────────

    #[test]
    fn test_duration_single_frame() {
        let w = make_window(5, 5, 0.0, 0.0, 0.0);
        assert_eq!(w.duration_frames(), 1);
    }

    #[test]
    fn test_duration_multi_frame() {
        let w = make_window(0, 99, 0.0, 0.0, 0.0);
        assert_eq!(w.duration_frames(), 100);
    }

    #[test]
    fn test_is_degraded_above_threshold() {
        let w = make_window(0, 10, 50.0, 40.0, 60.0);
        assert!(w.is_degraded(40.0));
    }

    #[test]
    fn test_is_not_degraded_below_threshold() {
        let w = make_window(0, 10, 20.0, 15.0, 25.0);
        assert!(!w.is_degraded(40.0));
    }

    #[test]
    fn test_is_not_degraded_at_threshold() {
        let w = make_window(0, 10, 40.0, 35.0, 45.0);
        assert!(!w.is_degraded(40.0));
    }

    #[test]
    fn test_score_range_zero_when_uniform() {
        let w = make_window(0, 10, 30.0, 30.0, 30.0);
        assert!((w.score_range()).abs() < f32::EPSILON);
    }

    #[test]
    fn test_score_range_positive() {
        let w = make_window(0, 10, 30.0, 20.0, 50.0);
        assert!((w.score_range() - 30.0).abs() < f32::EPSILON);
    }

    // ── QualityTimeline ────────────────────────────────────────────────────

    #[test]
    fn test_empty_timeline_overall_mean() {
        let tl = QualityTimeline::new();
        assert!((tl.overall_mean()).abs() < f32::EPSILON);
    }

    #[test]
    fn test_empty_timeline_find_worst_is_none() {
        let tl = QualityTimeline::new();
        assert!(tl.find_worst_window().is_none());
    }

    #[test]
    fn test_add_increases_window_count() {
        let mut tl = QualityTimeline::new();
        tl.add(make_window(0, 24, 10.0, 5.0, 15.0));
        assert_eq!(tl.windows.len(), 1);
        tl.add(make_window(25, 49, 20.0, 15.0, 25.0));
        assert_eq!(tl.windows.len(), 2);
    }

    #[test]
    fn test_overall_mean_single_window() {
        let mut tl = QualityTimeline::new();
        tl.add(make_window(0, 24, 42.0, 30.0, 55.0));
        assert!((tl.overall_mean() - 42.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_overall_mean_multiple_windows() {
        let mut tl = QualityTimeline::new();
        tl.add(make_window(0, 24, 20.0, 10.0, 30.0));
        tl.add(make_window(25, 49, 40.0, 30.0, 50.0));
        assert!((tl.overall_mean() - 30.0).abs() < 1e-5);
    }

    #[test]
    fn test_degraded_windows_none_below_threshold() {
        let mut tl = QualityTimeline::new();
        tl.add(make_window(0, 24, 10.0, 5.0, 15.0));
        tl.add(make_window(25, 49, 15.0, 10.0, 20.0));
        assert!(tl.degraded_windows(50.0).is_empty());
    }

    #[test]
    fn test_degraded_windows_all_above_threshold() {
        let mut tl = QualityTimeline::new();
        tl.add(make_window(0, 24, 60.0, 50.0, 70.0));
        tl.add(make_window(25, 49, 80.0, 70.0, 90.0));
        assert_eq!(tl.degraded_windows(50.0).len(), 2);
    }

    #[test]
    fn test_degraded_windows_partial() {
        let mut tl = QualityTimeline::new();
        tl.add(make_window(0, 24, 20.0, 10.0, 30.0)); // not degraded
        tl.add(make_window(25, 49, 70.0, 60.0, 80.0)); // degraded
        let degraded = tl.degraded_windows(50.0);
        assert_eq!(degraded.len(), 1);
        assert_eq!(degraded[0].start_frame, 25);
    }

    #[test]
    fn test_find_worst_window_single() {
        let mut tl = QualityTimeline::new();
        tl.add(make_window(0, 24, 42.0, 30.0, 55.0));
        let worst = tl.find_worst_window().expect("should succeed in test");
        assert!((worst.mean_score - 42.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_find_worst_window_multiple() {
        let mut tl = QualityTimeline::new();
        tl.add(make_window(0, 24, 10.0, 5.0, 15.0));
        tl.add(make_window(25, 49, 90.0, 80.0, 100.0));
        tl.add(make_window(50, 74, 30.0, 20.0, 40.0));
        let worst = tl.find_worst_window().expect("should succeed in test");
        assert_eq!(worst.start_frame, 25);
    }
}
