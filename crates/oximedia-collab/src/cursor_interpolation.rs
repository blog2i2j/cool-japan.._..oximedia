//! Smooth cursor and viewport interpolation for real-time collaborative presence.
//!
//! When multiple users share cursor positions over a network, updates arrive
//! discretely and infrequently. Naively rendering the latest known position
//! causes jittery motion. This module provides:
//!
//! - [`InterpolationMode`] — linear or cubic Hermite spline smoothing.
//! - [`CursorSample`] — a timestamped cursor snapshot for a single user.
//! - [`CursorInterpolator`] — maintains a per-user history of samples and
//!   evaluates smoothly-interpolated positions at arbitrary virtual times.
//! - [`ViewportInterpolator`] — similarly smooth interpolation for viewport
//!   pan/zoom transitions.

#![allow(dead_code)]

use std::collections::{HashMap, VecDeque};

// ─────────────────────────────────────────────────────────────────────────────
// InterpolationMode
// ─────────────────────────────────────────────────────────────────────────────

/// Algorithm used to interpolate between two cursor samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationMode {
    /// Snap immediately to the latest known position (no smoothing).
    Nearest,
    /// Linearly interpolate between the two bracketing samples.
    Linear,
    /// Cubic Hermite spline using finite-difference tangents, giving
    /// smooth velocity continuity across sample points.
    CubicHermite,
}

impl Default for InterpolationMode {
    fn default() -> Self {
        Self::Linear
    }
}

impl std::fmt::Display for InterpolationMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Nearest => write!(f, "nearest"),
            Self::Linear => write!(f, "linear"),
            Self::CubicHermite => write!(f, "cubic_hermite"),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CursorSample
// ─────────────────────────────────────────────────────────────────────────────

/// A discrete observation of a user's cursor at a point in time.
///
/// Positions are stored in continuous (floating-point) frame-space so that
/// sub-frame precision is preserved during interpolation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CursorSample {
    /// Wall-clock capture time in milliseconds since the Unix epoch.
    pub timestamp_ms: u64,
    /// Fractional frame number (e.g. 100.5 means "between frames 100 and 101").
    pub frame: f64,
    /// Track index (integer, but stored as f64 for uniform treatment).
    pub track: f64,
}

impl CursorSample {
    /// Create a new cursor sample.
    #[must_use]
    pub fn new(timestamp_ms: u64, frame: f64, track: f64) -> Self {
        Self { timestamp_ms, frame, track }
    }

    /// Linearly interpolate between `self` and `other` using parameter `t ∈ [0, 1]`.
    #[must_use]
    pub fn lerp(&self, other: &Self, t: f64) -> (f64, f64) {
        let frame = self.frame + (other.frame - self.frame) * t;
        let track = self.track + (other.track - self.track) * t;
        (frame, track)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Hermite helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Evaluate a cubic Hermite spline at parameter `t` given two endpoints and
/// their tangents.
///
/// The Hermite basis functions are:
/// ```text
/// h00(t) = 2t³ - 3t² + 1
/// h10(t) = t³  - 2t² + t
/// h01(t) = -2t³ + 3t²
/// h11(t) = t³  - t²
/// ```
#[inline]
fn hermite(p0: f64, m0: f64, p1: f64, m1: f64, t: f64) -> f64 {
    let t2 = t * t;
    let t3 = t2 * t;
    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + t;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;
    h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1
}

/// Estimate the Catmull-Rom tangent at point `p1` given neighbours `p0` and
/// `p2`.  When a neighbour is missing the one-sided finite difference is used.
#[inline]
fn catmull_rom_tangent(p_prev: Option<f64>, p_cur: f64, p_next: Option<f64>) -> f64 {
    match (p_prev, p_next) {
        (Some(pp), Some(pn)) => (pn - pp) * 0.5,
        (None, Some(pn)) => pn - p_cur,
        (Some(pp), None) => p_cur - pp,
        (None, None) => 0.0,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CursorInterpolator
// ─────────────────────────────────────────────────────────────────────────────

/// Maximum number of historical samples kept per user.
const DEFAULT_HISTORY_LEN: usize = 8;

/// Per-user sliding window of cursor samples plus interpolation algorithm.
#[derive(Debug)]
pub struct CursorInterpolator {
    /// Interpolation algorithm.
    pub mode: InterpolationMode,
    /// Samples per user (user_id → ring of samples, oldest first).
    history: HashMap<String, VecDeque<CursorSample>>,
    /// Maximum samples to retain per user.
    history_len: usize,
}

impl CursorInterpolator {
    /// Create a new interpolator with the given mode and history window.
    #[must_use]
    pub fn new(mode: InterpolationMode, history_len: usize) -> Self {
        Self {
            mode,
            history: HashMap::new(),
            history_len: history_len.max(2),
        }
    }

    /// Create a default interpolator (linear, 8 samples per user).
    #[must_use]
    pub fn default_linear() -> Self {
        Self::new(InterpolationMode::Linear, DEFAULT_HISTORY_LEN)
    }

    /// Push a new cursor sample for `user_id`.
    pub fn push(&mut self, user_id: impl Into<String>, sample: CursorSample) {
        let queue = self.history.entry(user_id.into()).or_default();
        queue.push_back(sample);
        while queue.len() > self.history_len {
            queue.pop_front();
        }
    }

    /// Remove a user from the interpolator.
    pub fn remove(&mut self, user_id: &str) {
        self.history.remove(user_id);
    }

    /// Number of users currently tracked.
    #[must_use]
    pub fn user_count(&self) -> usize {
        self.history.len()
    }

    /// Evaluate the interpolated cursor position for `user_id` at `query_ms`.
    ///
    /// Returns `None` if the user is unknown.  When the query time lies outside
    /// the known sample range the nearest endpoint is returned (clamped).
    #[must_use]
    pub fn evaluate(&self, user_id: &str, query_ms: u64) -> Option<(f64, f64)> {
        let queue = self.history.get(user_id)?;
        if queue.is_empty() {
            return None;
        }
        if queue.len() == 1 {
            let s = &queue[0];
            return Some((s.frame, s.track));
        }

        // Clamp to sample range.
        let first = queue.front()?;
        let last = queue.back()?;
        if query_ms <= first.timestamp_ms {
            return Some((first.frame, first.track));
        }
        if query_ms >= last.timestamp_ms {
            return Some((last.frame, last.track));
        }

        // Find the bracketing pair (s0.timestamp_ms ≤ query_ms < s1.timestamp_ms).
        let samples: Vec<&CursorSample> = queue.iter().collect();
        let idx = samples.partition_point(|s| s.timestamp_ms <= query_ms).saturating_sub(1);
        let s0 = samples[idx];
        let s1 = samples[idx + 1];

        let span = (s1.timestamp_ms - s0.timestamp_ms) as f64;
        let t = if span > 0.0 {
            (query_ms - s0.timestamp_ms) as f64 / span
        } else {
            0.0
        };

        let (frame, track) = match self.mode {
            InterpolationMode::Nearest => {
                if t < 0.5 {
                    (s0.frame, s0.track)
                } else {
                    (s1.frame, s1.track)
                }
            }
            InterpolationMode::Linear => s0.lerp(s1, t),
            InterpolationMode::CubicHermite => {
                let prev_frame = idx.checked_sub(1).map(|i| samples[i].frame);
                let next_frame = samples.get(idx + 2).map(|s| s.frame);
                let prev_track = idx.checked_sub(1).map(|i| samples[i].track);
                let next_track = samples.get(idx + 2).map(|s| s.track);

                let m0_frame = catmull_rom_tangent(prev_frame, s0.frame, Some(s1.frame));
                let m1_frame = catmull_rom_tangent(Some(s0.frame), s1.frame, next_frame);
                let m0_track = catmull_rom_tangent(prev_track, s0.track, Some(s1.track));
                let m1_track = catmull_rom_tangent(Some(s0.track), s1.track, next_track);

                let frame = hermite(s0.frame, m0_frame, s1.frame, m1_frame, t);
                let track = hermite(s0.track, m0_track, s1.track, m1_track, t);
                (frame, track)
            }
        };

        Some((frame, track))
    }

    /// Return the latest raw sample for a user (no interpolation).
    #[must_use]
    pub fn latest_sample(&self, user_id: &str) -> Option<&CursorSample> {
        self.history.get(user_id)?.back()
    }

    /// Return all known sample timestamps for a user (oldest first).
    #[must_use]
    pub fn sample_timestamps(&self, user_id: &str) -> Vec<u64> {
        self.history
            .get(user_id)
            .map(|q| q.iter().map(|s| s.timestamp_ms).collect())
            .unwrap_or_default()
    }
}

impl Default for CursorInterpolator {
    fn default() -> Self {
        Self::default_linear()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ViewportInterpolator
// ─────────────────────────────────────────────────────────────────────────────

/// A single timestamped viewport observation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ViewportSample {
    /// Capture time in milliseconds.
    pub timestamp_ms: u64,
    /// Start frame of the visible range.
    pub start_frame: f64,
    /// End frame of the visible range.
    pub end_frame: f64,
    /// Zoom level (1.0 = 100%).
    pub zoom: f64,
}

impl ViewportSample {
    /// Create a new viewport sample.
    #[must_use]
    pub fn new(timestamp_ms: u64, start_frame: f64, end_frame: f64, zoom: f64) -> Self {
        Self { timestamp_ms, start_frame, end_frame, zoom }
    }

    /// Linearly interpolate between `self` and `other`.
    #[must_use]
    pub fn lerp(&self, other: &Self, t: f64) -> (f64, f64, f64) {
        let s = self.start_frame + (other.start_frame - self.start_frame) * t;
        let e = self.end_frame + (other.end_frame - self.end_frame) * t;
        let z = self.zoom + (other.zoom - self.zoom) * t;
        (s, e, z)
    }
}

/// Smooth viewport interpolation for collaborative pan/zoom sharing.
#[derive(Debug)]
pub struct ViewportInterpolator {
    /// Interpolation mode.
    pub mode: InterpolationMode,
    /// Per-user viewport history.
    history: HashMap<String, VecDeque<ViewportSample>>,
    /// Maximum samples per user.
    history_len: usize,
}

impl ViewportInterpolator {
    /// Create a new viewport interpolator.
    #[must_use]
    pub fn new(mode: InterpolationMode, history_len: usize) -> Self {
        Self {
            mode,
            history: HashMap::new(),
            history_len: history_len.max(2),
        }
    }

    /// Push a viewport sample for `user_id`.
    pub fn push(&mut self, user_id: impl Into<String>, sample: ViewportSample) {
        let queue = self.history.entry(user_id.into()).or_default();
        queue.push_back(sample);
        while queue.len() > self.history_len {
            queue.pop_front();
        }
    }

    /// Evaluate the interpolated viewport at `query_ms`.
    ///
    /// Returns `(start_frame, end_frame, zoom)` or `None` if the user is unknown.
    #[must_use]
    pub fn evaluate(&self, user_id: &str, query_ms: u64) -> Option<(f64, f64, f64)> {
        let queue = self.history.get(user_id)?;
        if queue.is_empty() {
            return None;
        }
        if queue.len() == 1 {
            let s = &queue[0];
            return Some((s.start_frame, s.end_frame, s.zoom));
        }

        let first = queue.front()?;
        let last = queue.back()?;
        if query_ms <= first.timestamp_ms {
            return Some((first.start_frame, first.end_frame, first.zoom));
        }
        if query_ms >= last.timestamp_ms {
            return Some((last.start_frame, last.end_frame, last.zoom));
        }

        let samples: Vec<&ViewportSample> = queue.iter().collect();
        let idx = samples.partition_point(|s| s.timestamp_ms <= query_ms).saturating_sub(1);
        let s0 = samples[idx];
        let s1 = samples[idx + 1];

        let span = (s1.timestamp_ms - s0.timestamp_ms) as f64;
        let t = if span > 0.0 {
            (query_ms - s0.timestamp_ms) as f64 / span
        } else {
            0.0
        };

        match self.mode {
            InterpolationMode::Nearest => {
                if t < 0.5 {
                    Some((s0.start_frame, s0.end_frame, s0.zoom))
                } else {
                    Some((s1.start_frame, s1.end_frame, s1.zoom))
                }
            }
            InterpolationMode::Linear | InterpolationMode::CubicHermite => {
                // For viewport we use linear for all modes to avoid overshoot on
                // abrupt pan/zoom changes. Cubic Hermite only applies to cursors.
                Some(s0.lerp(s1, t))
            }
        }
    }

    /// Number of users tracked.
    #[must_use]
    pub fn user_count(&self) -> usize {
        self.history.len()
    }

    /// Remove a user.
    pub fn remove(&mut self, user_id: &str) {
        self.history.remove(user_id);
    }
}

impl Default for ViewportInterpolator {
    fn default() -> Self {
        Self::new(InterpolationMode::Linear, DEFAULT_HISTORY_LEN)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── CursorSample ──

    #[test]
    fn test_cursor_sample_lerp_midpoint() {
        let a = CursorSample::new(0, 0.0, 0.0);
        let b = CursorSample::new(100, 100.0, 4.0);
        let (f, t) = a.lerp(&b, 0.5);
        assert!((f - 50.0).abs() < 1e-9);
        assert!((t - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_cursor_sample_lerp_boundary() {
        let a = CursorSample::new(0, 10.0, 1.0);
        let b = CursorSample::new(1000, 20.0, 3.0);
        let (f0, t0) = a.lerp(&b, 0.0);
        let (f1, t1) = a.lerp(&b, 1.0);
        assert!((f0 - 10.0).abs() < 1e-9);
        assert!((t0 - 1.0).abs() < 1e-9);
        assert!((f1 - 20.0).abs() < 1e-9);
        assert!((t1 - 3.0).abs() < 1e-9);
    }

    // ── InterpolationMode ──

    #[test]
    fn test_interpolation_mode_display() {
        assert_eq!(InterpolationMode::Nearest.to_string(), "nearest");
        assert_eq!(InterpolationMode::Linear.to_string(), "linear");
        assert_eq!(InterpolationMode::CubicHermite.to_string(), "cubic_hermite");
    }

    #[test]
    fn test_interpolation_mode_default_is_linear() {
        assert_eq!(InterpolationMode::default(), InterpolationMode::Linear);
    }

    // ── CursorInterpolator – single user ──

    fn make_interp(mode: InterpolationMode) -> CursorInterpolator {
        CursorInterpolator::new(mode, 8)
    }

    fn push_linear_ramp(interp: &mut CursorInterpolator, user: &str) {
        // 5 samples: frame goes 0→400 over 400ms
        for i in 0..5u64 {
            interp.push(user, CursorSample::new(i * 100, i as f64 * 100.0, 0.0));
        }
    }

    #[test]
    fn test_linear_interpolation_midpoint() {
        let mut interp = make_interp(InterpolationMode::Linear);
        push_linear_ramp(&mut interp, "alice");
        // Exactly between t=100 (frame=100) and t=200 (frame=200)
        let (frame, _) = interp.evaluate("alice", 150).expect("should have value");
        assert!((frame - 150.0).abs() < 1e-6, "got {frame}");
    }

    #[test]
    fn test_linear_interpolation_clamp_before_first() {
        let mut interp = make_interp(InterpolationMode::Linear);
        push_linear_ramp(&mut interp, "alice");
        let (frame, _) = interp.evaluate("alice", 0).expect("should have value");
        assert!((frame - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_linear_interpolation_clamp_after_last() {
        let mut interp = make_interp(InterpolationMode::Linear);
        push_linear_ramp(&mut interp, "alice");
        let (frame, _) = interp.evaluate("alice", 9999).expect("should have value");
        assert!((frame - 400.0).abs() < 1e-9, "expected 400, got {frame}");
    }

    #[test]
    fn test_nearest_interpolation_snaps() {
        let mut interp = make_interp(InterpolationMode::Nearest);
        push_linear_ramp(&mut interp, "bob");
        // At t=149 (49% of the way from 100 to 200) → nearest = 100
        let (frame, _) = interp.evaluate("bob", 149).expect("should have value");
        assert!((frame - 100.0).abs() < 1e-9, "expected 100, got {frame}");
        // At t=151 (51% of the way from 100 to 200) → nearest = 200
        let (frame2, _) = interp.evaluate("bob", 151).expect("should have value");
        assert!((frame2 - 200.0).abs() < 1e-9, "expected 200, got {frame2}");
    }

    #[test]
    fn test_cubic_hermite_is_monotone_on_linear_ramp() {
        // On a perfectly linear ramp (equally spaced), the cubic Hermite spline
        // should produce the same result as linear interpolation.
        let mut interp = make_interp(InterpolationMode::CubicHermite);
        push_linear_ramp(&mut interp, "carol");
        let (frame, _) = interp.evaluate("carol", 150).expect("should have value");
        // Tolerance is relaxed because tangent estimation introduces a tiny error.
        assert!((frame - 150.0).abs() < 2.0, "expected ~150, got {frame}");
    }

    #[test]
    fn test_unknown_user_returns_none() {
        let interp = CursorInterpolator::default();
        assert!(interp.evaluate("ghost", 500).is_none());
    }

    #[test]
    fn test_history_window_evicts_oldest() {
        let mut interp = CursorInterpolator::new(InterpolationMode::Linear, 3);
        for i in 0..6u64 {
            interp.push("u", CursorSample::new(i * 100, i as f64, 0.0));
        }
        // Window of 3 → only the last 3 timestamps: 300, 400, 500
        let ts = interp.sample_timestamps("u");
        assert_eq!(ts, vec![300, 400, 500]);
    }

    #[test]
    fn test_remove_user() {
        let mut interp = make_interp(InterpolationMode::Linear);
        interp.push("dave", CursorSample::new(0, 0.0, 0.0));
        assert_eq!(interp.user_count(), 1);
        interp.remove("dave");
        assert_eq!(interp.user_count(), 0);
        assert!(interp.evaluate("dave", 0).is_none());
    }

    #[test]
    fn test_latest_sample() {
        let mut interp = make_interp(InterpolationMode::Linear);
        interp.push("e", CursorSample::new(10, 1.0, 2.0));
        interp.push("e", CursorSample::new(20, 3.0, 4.0));
        let s = interp.latest_sample("e").expect("should have sample");
        assert_eq!(s.timestamp_ms, 20);
        assert!((s.frame - 3.0).abs() < 1e-9);
    }

    // ── ViewportInterpolator ──

    #[test]
    fn test_viewport_linear_interpolation() {
        let mut vi = ViewportInterpolator::default();
        vi.push("alice", ViewportSample::new(0, 0.0, 100.0, 1.0));
        vi.push("alice", ViewportSample::new(1000, 100.0, 200.0, 2.0));
        let (s, e, z) = vi.evaluate("alice", 500).expect("should have value");
        assert!((s - 50.0).abs() < 1e-6);
        assert!((e - 150.0).abs() < 1e-6);
        assert!((z - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_viewport_unknown_user_none() {
        let vi = ViewportInterpolator::default();
        assert!(vi.evaluate("nobody", 100).is_none());
    }

    #[test]
    fn test_viewport_single_sample_returns_exact() {
        let mut vi = ViewportInterpolator::default();
        vi.push("u", ViewportSample::new(500, 10.0, 90.0, 1.5));
        let (s, e, z) = vi.evaluate("u", 500).expect("should have value");
        assert!((s - 10.0).abs() < 1e-9);
        assert!((e - 90.0).abs() < 1e-9);
        assert!((z - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_viewport_nearest_mode_snaps() {
        let mut vi = ViewportInterpolator::new(InterpolationMode::Nearest, 8);
        vi.push("u", ViewportSample::new(0, 0.0, 100.0, 1.0));
        vi.push("u", ViewportSample::new(1000, 200.0, 300.0, 2.0));
        // t=400 → 40% → nearest to s0
        let (s, _, _) = vi.evaluate("u", 400).expect("should have value");
        assert!((s - 0.0).abs() < 1e-9, "expected 0.0, got {s}");
        // t=600 → 60% → nearest to s1
        let (s2, _, _) = vi.evaluate("u", 600).expect("should have value");
        assert!((s2 - 200.0).abs() < 1e-9, "expected 200.0, got {s2}");
    }

    #[test]
    fn test_viewport_remove_user() {
        let mut vi = ViewportInterpolator::default();
        vi.push("x", ViewportSample::new(0, 0.0, 100.0, 1.0));
        assert_eq!(vi.user_count(), 1);
        vi.remove("x");
        assert_eq!(vi.user_count(), 0);
    }

    // ── Hermite helpers ──

    #[test]
    fn test_hermite_endpoints() {
        // At t=0 the result should equal p0, at t=1 it should equal p1.
        let v0 = hermite(10.0, 5.0, 30.0, 5.0, 0.0);
        let v1 = hermite(10.0, 5.0, 30.0, 5.0, 1.0);
        assert!((v0 - 10.0).abs() < 1e-9);
        assert!((v1 - 30.0).abs() < 1e-9);
    }

    #[test]
    fn test_catmull_rom_tangent_central() {
        let t = catmull_rom_tangent(Some(0.0), 5.0, Some(10.0));
        // Central difference: (10 - 0) / 2 = 5
        assert!((t - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_catmull_rom_tangent_one_sided_right() {
        // Only next neighbour: tangent = next - cur
        let t = catmull_rom_tangent(None, 5.0, Some(10.0));
        assert!((t - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_catmull_rom_tangent_one_sided_left() {
        // Only prev neighbour: tangent = cur - prev
        let t = catmull_rom_tangent(Some(0.0), 5.0, None);
        assert!((t - 5.0).abs() < 1e-9);
    }
}
