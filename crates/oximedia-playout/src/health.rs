//! Playout health monitoring.
//!
//! [`PlayoutHealth`] tracks cumulative dropped-frame and late-segment counts
//! and exposes an `is_healthy` predicate that returns `false` once either
//! counter exceeds a configurable threshold.
//!
//! # Example
//!
//! ```
//! use oximedia_playout::health::PlayoutHealth;
//!
//! let mut health = PlayoutHealth::new();
//! health.update(0, 0);
//! assert!(health.is_healthy(5));
//!
//! health.update(3, 2);
//! assert!(health.is_healthy(5));  // 3 ≤ 5 and 2 ≤ 5
//!
//! health.update(10, 0);
//! assert!(!health.is_healthy(5)); // 13 > 5
//! ```

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// PlayoutHealth
// ---------------------------------------------------------------------------

/// Accumulates playout quality metrics and reports overall health status.
#[derive(Debug, Clone, Default)]
pub struct PlayoutHealth {
    /// Total dropped frames since the last `reset()` call.
    dropped_frames: u64,
    /// Total late segments since the last `reset()` call.
    late_segments: u64,
    /// Number of `update()` calls recorded.
    update_count: u64,
}

impl PlayoutHealth {
    /// Create a new `PlayoutHealth` with all counters at zero.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a new observation of dropped-frame and late-segment counts.
    ///
    /// Values are *accumulated* — they represent the counts since the
    /// previous observation, not absolute totals.  Call `reset()` to clear
    /// the accumulators.
    pub fn update(&mut self, dropped_frames: u32, late_segments: u32) {
        self.dropped_frames = self.dropped_frames.saturating_add(u64::from(dropped_frames));
        self.late_segments = self.late_segments.saturating_add(u64::from(late_segments));
        self.update_count = self.update_count.saturating_add(1);
    }

    /// Returns `true` when both the total dropped-frame count **and** the
    /// total late-segment count are each ≤ `threshold`.
    ///
    /// A threshold of `0` means any single dropped frame or late segment
    /// makes the server unhealthy.
    #[must_use]
    pub fn is_healthy(&self, threshold: u32) -> bool {
        self.dropped_frames <= u64::from(threshold)
            && self.late_segments <= u64::from(threshold)
    }

    /// Total accumulated dropped frames.
    #[must_use]
    pub fn dropped_frames(&self) -> u64 {
        self.dropped_frames
    }

    /// Total accumulated late segments.
    #[must_use]
    pub fn late_segments(&self) -> u64 {
        self.late_segments
    }

    /// Number of `update()` calls since creation or the last `reset()`.
    #[must_use]
    pub fn update_count(&self) -> u64 {
        self.update_count
    }

    /// Reset all counters to zero.
    pub fn reset(&mut self) {
        self.dropped_frames = 0;
        self.late_segments = 0;
        self.update_count = 0;
    }

    /// Returns a human-readable health summary string.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "dropped_frames={} late_segments={} updates={}",
            self.dropped_frames, self.late_segments, self.update_count
        )
    }
}

// ---------------------------------------------------------------------------
// HealthReport — snapshot for logging / telemetry
// ---------------------------------------------------------------------------

/// A point-in-time snapshot of playout health metrics.
#[derive(Debug, Clone)]
pub struct HealthReport {
    /// Accumulated dropped frames.
    pub dropped_frames: u64,
    /// Accumulated late segments.
    pub late_segments: u64,
    /// Number of updates recorded.
    pub update_count: u64,
    /// Whether the server is considered healthy under the given threshold.
    pub healthy: bool,
}

impl PlayoutHealth {
    /// Build a [`HealthReport`] snapshot.
    #[must_use]
    pub fn report(&self, threshold: u32) -> HealthReport {
        HealthReport {
            dropped_frames: self.dropped_frames,
            late_segments: self.late_segments,
            update_count: self.update_count,
            healthy: self.is_healthy(threshold),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── new / default ─────────────────────────────────────────────────────────

    #[test]
    fn test_new_starts_at_zero() {
        let h = PlayoutHealth::new();
        assert_eq!(h.dropped_frames(), 0);
        assert_eq!(h.late_segments(), 0);
        assert_eq!(h.update_count(), 0);
    }

    // ── is_healthy ────────────────────────────────────────────────────────────

    #[test]
    fn test_healthy_with_zero_threshold_and_no_issues() {
        let h = PlayoutHealth::new();
        assert!(h.is_healthy(0), "No drops → healthy even at threshold 0");
    }

    #[test]
    fn test_unhealthy_when_dropped_exceeds_threshold() {
        let mut h = PlayoutHealth::new();
        h.update(6, 0);
        assert!(!h.is_healthy(5));
    }

    #[test]
    fn test_unhealthy_when_late_exceeds_threshold() {
        let mut h = PlayoutHealth::new();
        h.update(0, 6);
        assert!(!h.is_healthy(5));
    }

    #[test]
    fn test_healthy_at_exact_threshold() {
        let mut h = PlayoutHealth::new();
        h.update(5, 5);
        assert!(h.is_healthy(5), "Exactly at threshold → healthy");
    }

    #[test]
    fn test_unhealthy_one_above_threshold() {
        let mut h = PlayoutHealth::new();
        h.update(6, 0);
        assert!(!h.is_healthy(5));
    }

    // ── update accumulates ────────────────────────────────────────────────────

    #[test]
    fn test_update_accumulates_dropped_frames() {
        let mut h = PlayoutHealth::new();
        h.update(2, 0);
        h.update(3, 0);
        assert_eq!(h.dropped_frames(), 5);
    }

    #[test]
    fn test_update_accumulates_late_segments() {
        let mut h = PlayoutHealth::new();
        h.update(0, 1);
        h.update(0, 2);
        assert_eq!(h.late_segments(), 3);
    }

    #[test]
    fn test_update_increments_count() {
        let mut h = PlayoutHealth::new();
        h.update(0, 0);
        h.update(0, 0);
        assert_eq!(h.update_count(), 2);
    }

    // ── reset ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_reset_clears_all_counters() {
        let mut h = PlayoutHealth::new();
        h.update(10, 5);
        h.reset();
        assert_eq!(h.dropped_frames(), 0);
        assert_eq!(h.late_segments(), 0);
        assert_eq!(h.update_count(), 0);
    }

    #[test]
    fn test_reset_then_healthy_again() {
        let mut h = PlayoutHealth::new();
        h.update(100, 100);
        assert!(!h.is_healthy(5));
        h.reset();
        assert!(h.is_healthy(5));
    }

    // ── report ────────────────────────────────────────────────────────────────

    #[test]
    fn test_report_healthy_true() {
        let mut h = PlayoutHealth::new();
        h.update(1, 1);
        let r = h.report(5);
        assert!(r.healthy);
        assert_eq!(r.dropped_frames, 1);
        assert_eq!(r.late_segments, 1);
        assert_eq!(r.update_count, 1);
    }

    #[test]
    fn test_report_healthy_false() {
        let mut h = PlayoutHealth::new();
        h.update(10, 0);
        let r = h.report(5);
        assert!(!r.healthy);
    }

    // ── summary ───────────────────────────────────────────────────────────────

    #[test]
    fn test_summary_contains_counters() {
        let mut h = PlayoutHealth::new();
        h.update(3, 2);
        let s = h.summary();
        assert!(s.contains("dropped_frames=3"));
        assert!(s.contains("late_segments=2"));
        assert!(s.contains("updates=1"));
    }
}
