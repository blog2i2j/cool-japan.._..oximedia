//! Automation playback engine — time-based parameter automation.
//!
//! This module provides a standalone, time-based automation system that is
//! independent of the sample-based [`crate::automation_player`] module.
//!
//! The primary types are:
//! - [`AutomationPoint`]: a `(time_secs, value, curve)` triplet.
//! - [`AutomationLane`]: a named sequence of points for a single parameter.
//! - [`AutomationPlayer`]: advances a clock and emits parameter updates.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// AutomationCurveType
// ---------------------------------------------------------------------------

/// Interpolation curve between two automation points.
#[derive(Debug, Clone, PartialEq)]
pub enum AutomationCurveType {
    /// Straight-line interpolation between points.
    Linear,
    /// Step: hold the value at this point until the next point.
    Hold,
    /// Smooth S-curve: `value = 3t² - 2t³` where `t` is normalized position.
    SCurve,
}

// ---------------------------------------------------------------------------
// AutomationPoint
// ---------------------------------------------------------------------------

/// A single keyframe: maps a point in time to a parameter value with a curve.
#[derive(Debug, Clone)]
pub struct AutomationPoint {
    /// Position in seconds.
    pub time_secs: f64,
    /// Parameter value at this position.
    pub value: f32,
    /// Curve type used when interpolating *from* this point to the next.
    pub curve: AutomationCurveType,
}

impl AutomationPoint {
    /// Create a new automation point with [`AutomationCurveType::Linear`] curve.
    #[must_use]
    pub fn new(time_secs: f64, value: f32) -> Self {
        Self {
            time_secs,
            value,
            curve: AutomationCurveType::Linear,
        }
    }

    /// Create a new automation point with an explicit curve type.
    #[must_use]
    pub fn with_curve(time_secs: f64, value: f32, curve: AutomationCurveType) -> Self {
        Self {
            time_secs,
            value,
            curve,
        }
    }
}

// ---------------------------------------------------------------------------
// AutomationLane
// ---------------------------------------------------------------------------

/// A named automation lane for a single parameter.
///
/// Points are kept in ascending time order internally.
#[derive(Debug, Clone)]
pub struct AutomationLane {
    /// Identifier for the parameter this lane controls.
    pub parameter_id: String,
    /// Minimum allowed value (clamped on output).
    pub min_value: f32,
    /// Maximum allowed value (clamped on output).
    pub max_value: f32,
    /// Sorted list of keyframe points.
    pub points: Vec<AutomationPoint>,
}

impl AutomationLane {
    /// Create a new, empty automation lane.
    #[must_use]
    pub fn new(parameter_id: impl Into<String>, min: f32, max: f32) -> Self {
        Self {
            parameter_id: parameter_id.into(),
            min_value: min,
            max_value: max,
            points: Vec::new(),
        }
    }

    /// Append a point with the default [`AutomationCurveType::Linear`] curve.
    ///
    /// Points are inserted in sorted order by `time_secs`.
    #[must_use]
    pub fn add_point(mut self, time_secs: f64, value: f32) -> Self {
        self.insert_sorted(AutomationPoint::new(time_secs, value));
        self
    }

    /// Append a point with an explicit curve type.
    #[must_use]
    pub fn add_point_with_curve(
        mut self,
        time_secs: f64,
        value: f32,
        curve: AutomationCurveType,
    ) -> Self {
        self.insert_sorted(AutomationPoint::with_curve(time_secs, value, curve));
        self
    }

    fn insert_sorted(&mut self, point: AutomationPoint) {
        let pos = self
            .points
            .partition_point(|p| p.time_secs <= point.time_secs);
        self.points.insert(pos, point);
    }

    /// Evaluate the interpolated (and clamped) value at `time_secs`.
    ///
    /// - Before the first point: returns the first point's value.
    /// - After the last point: returns the last point's value.
    /// - Between two points: interpolates according to the earlier point's curve.
    #[must_use]
    pub fn evaluate(&self, time_secs: f64) -> f32 {
        if self.points.is_empty() {
            return (self.min_value + self.max_value) * 0.5;
        }

        let first = &self.points[0];
        let last = &self.points[self.points.len() - 1];

        if time_secs <= first.time_secs {
            return first.value.clamp(self.min_value, self.max_value);
        }
        if time_secs >= last.time_secs {
            return last.value.clamp(self.min_value, self.max_value);
        }

        // Binary search for the surrounding segment.
        let idx = self
            .points
            .partition_point(|p| p.time_secs <= time_secs)
            .saturating_sub(1);

        let p0 = &self.points[idx];
        let p1 = &self.points[idx + 1];

        let span = p1.time_secs - p0.time_secs;
        let raw = if span < f64::EPSILON {
            p1.value
        } else {
            let t = ((time_secs - p0.time_secs) / span) as f32;
            match p0.curve {
                AutomationCurveType::Hold => p0.value,
                AutomationCurveType::Linear => p0.value + t * (p1.value - p0.value),
                AutomationCurveType::SCurve => {
                    let s = 3.0 * t * t - 2.0 * t * t * t;
                    p0.value + s * (p1.value - p0.value)
                }
            }
        };

        raw.clamp(self.min_value, self.max_value)
    }

    /// Total duration of the lane (time of the last point, or 0 if empty).
    #[must_use]
    pub fn duration_secs(&self) -> f64 {
        self.points.last().map_or(0.0, |p| p.time_secs)
    }

    /// Number of keyframe points in this lane.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.points.len()
    }
}

// ---------------------------------------------------------------------------
// AutomationPlayer
// ---------------------------------------------------------------------------

/// Automation player that advances a clock and fires parameter updates.
pub struct AutomationPlayer {
    lanes: Vec<AutomationLane>,
    current_time_secs: f64,
    playing: bool,
}

impl AutomationPlayer {
    /// Create a new, stopped automation player at time zero.
    #[must_use]
    pub fn new() -> Self {
        Self {
            lanes: Vec::new(),
            current_time_secs: 0.0,
            playing: false,
        }
    }

    /// Add an automation lane and return `self` (builder pattern).
    #[must_use]
    pub fn add_lane(mut self, lane: AutomationLane) -> Self {
        self.lanes.push(lane);
        self
    }

    /// Advance the player by `dt_secs` and return a map of `parameter_id → value`
    /// for all registered lanes.
    ///
    /// If the player is stopped this still returns the current snapshot values
    /// (useful for scrubbing) but does *not* advance the clock.
    pub fn advance(&mut self, dt_secs: f64) -> HashMap<String, f32> {
        if self.playing {
            self.current_time_secs += dt_secs;
        }
        self.snapshot()
    }

    fn snapshot(&self) -> HashMap<String, f32> {
        let mut map = HashMap::with_capacity(self.lanes.len());
        for lane in &self.lanes {
            let value = lane.evaluate(self.current_time_secs);
            map.insert(lane.parameter_id.clone(), value);
        }
        map
    }

    /// Seek (jump) to an absolute time without advancing.
    pub fn seek(&mut self, time_secs: f64) {
        self.current_time_secs = time_secs.max(0.0);
    }

    /// Start playback.
    pub fn play(&mut self) {
        self.playing = true;
    }

    /// Stop playback (clock freezes; `advance` still returns current snapshot).
    pub fn stop(&mut self) {
        self.playing = false;
    }

    /// Current playhead position in seconds.
    #[must_use]
    pub fn current_time(&self) -> f64 {
        self.current_time_secs
    }

    /// Whether the player is currently playing.
    #[must_use]
    pub fn is_playing(&self) -> bool {
        self.playing
    }

    /// Get the current interpolated value for a parameter without advancing time.
    ///
    /// Returns `None` if no lane is registered for `parameter_id`.
    #[must_use]
    pub fn current_value(&self, parameter_id: &str) -> Option<f32> {
        self.lanes
            .iter()
            .find(|l| l.parameter_id == parameter_id)
            .map(|l| l.evaluate(self.current_time_secs))
    }
}

impl Default for AutomationPlayer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // AutomationLane
    // -----------------------------------------------------------------------

    #[test]
    fn test_linear_interpolation() {
        let lane = AutomationLane::new("gain", 0.0, 1.0)
            .add_point(0.0, 0.0)
            .add_point(1.0, 1.0);
        let mid = lane.evaluate(0.5);
        assert!((mid - 0.5).abs() < 1e-5, "linear midpoint: {mid}");
    }

    #[test]
    fn test_hold_curve_stays_constant() {
        let lane = AutomationLane::new("vol", 0.0, 1.0)
            .add_point_with_curve(0.0, 0.25, AutomationCurveType::Hold)
            .add_point(1.0, 1.0);
        // Anywhere between 0 and 1 the value should be held at 0.25.
        for &t in &[0.1_f64, 0.4, 0.9] {
            let v = lane.evaluate(t);
            assert!(
                (v - 0.25).abs() < 1e-5,
                "Hold curve should stay at 0.25 at t={t}, got {v}"
            );
        }
    }

    #[test]
    fn test_scurve_midpoint_near_half() {
        let lane = AutomationLane::new("pan", 0.0, 1.0)
            .add_point_with_curve(0.0, 0.0, AutomationCurveType::SCurve)
            .add_point(1.0, 1.0);
        // S-curve: f(0.5) = 3*(0.5)² - 2*(0.5)³ = 0.75 - 0.25 = 0.5 exactly.
        let mid = lane.evaluate(0.5);
        assert!(
            (mid - 0.5).abs() < 1e-4,
            "S-curve midpoint should be ~0.5, got {mid}"
        );
    }

    #[test]
    fn test_before_first_point_returns_first_value() {
        let lane = AutomationLane::new("x", 0.0, 1.0)
            .add_point(2.0, 0.7)
            .add_point(4.0, 1.0);
        assert!(
            (lane.evaluate(0.0) - 0.7).abs() < 1e-5,
            "Should return first point value before start"
        );
        assert!(
            (lane.evaluate(1.9) - 0.7).abs() < 1e-5,
            "Should return first point value just before first key"
        );
    }

    #[test]
    fn test_after_last_point_returns_last_value() {
        let lane = AutomationLane::new("x", 0.0, 1.0)
            .add_point(0.0, 0.0)
            .add_point(3.0, 0.6);
        assert!(
            (lane.evaluate(10.0) - 0.6).abs() < 1e-5,
            "Should return last point value after end"
        );
    }

    #[test]
    fn test_duration_secs() {
        let lane = AutomationLane::new("t", 0.0, 1.0)
            .add_point(0.0, 0.0)
            .add_point(5.0, 1.0);
        assert!((lane.duration_secs() - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_point_count() {
        let lane = AutomationLane::new("t", 0.0, 1.0)
            .add_point(0.0, 0.0)
            .add_point(1.0, 0.5)
            .add_point(2.0, 1.0);
        assert_eq!(lane.point_count(), 3);
    }

    #[test]
    fn test_value_clamped_to_range() {
        // Add a point whose value is outside the lane's declared range.
        let lane = AutomationLane {
            parameter_id: "vol".to_string(),
            min_value: 0.0,
            max_value: 0.5,
            points: vec![AutomationPoint::new(0.0, 1.0)],
        };
        assert!(
            (lane.evaluate(0.0) - 0.5).abs() < 1e-5,
            "Value should be clamped to max"
        );
    }

    // -----------------------------------------------------------------------
    // AutomationPlayer
    // -----------------------------------------------------------------------

    #[test]
    fn test_player_advance_emits_correct_values() {
        let lane = AutomationLane::new("gain", 0.0, 1.0)
            .add_point(0.0, 0.0)
            .add_point(2.0, 1.0);

        let mut player = AutomationPlayer::new().add_lane(lane);
        player.play();

        // After 1 second the linear lane should be at 0.5.
        let updates = player.advance(1.0);
        let gain = updates.get("gain").copied().unwrap_or(-1.0);
        assert!(
            (gain - 0.5).abs() < 1e-4,
            "Expected 0.5 at t=1s, got {gain}"
        );
    }

    #[test]
    fn test_player_seek_resets_time() {
        let mut player = AutomationPlayer::new();
        player.play();
        let _ = player.advance(5.0);
        player.seek(0.0);
        assert!((player.current_time() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_player_play_stop_control() {
        let mut player = AutomationPlayer::new();
        assert!(!player.is_playing());
        player.play();
        assert!(player.is_playing());
        player.stop();
        assert!(!player.is_playing());
    }

    #[test]
    fn test_stopped_player_does_not_advance_time() {
        let mut player = AutomationPlayer::new();
        // player is stopped by default
        let _ = player.advance(3.0);
        assert!(
            (player.current_time() - 0.0).abs() < 1e-9,
            "Stopped player should not advance time"
        );
    }

    #[test]
    fn test_current_value_without_advance() {
        let lane = AutomationLane::new("pan", -1.0, 1.0)
            .add_point(0.0, -0.5)
            .add_point(1.0, 0.5);

        let player = AutomationPlayer::new().add_lane(lane);
        // At time 0 the lane should return -0.5.
        let v = player.current_value("pan").unwrap_or(0.0);
        assert!(
            (v - (-0.5)).abs() < 1e-5,
            "current_value at t=0 should be -0.5, got {v}"
        );
    }

    #[test]
    fn test_current_value_missing_parameter() {
        let player = AutomationPlayer::new();
        assert!(player.current_value("nonexistent").is_none());
    }

    #[test]
    fn test_scurve_symmetry() {
        // S-curve should be symmetric around the midpoint of the range.
        let lane = AutomationLane::new("x", 0.0, 1.0)
            .add_point_with_curve(0.0, 0.0, AutomationCurveType::SCurve)
            .add_point(1.0, 1.0);
        let at_quarter = lane.evaluate(0.25);
        let at_three_quarters = lane.evaluate(0.75);
        // f(0.25) + f(0.75) should equal 1.0 due to symmetry.
        let sum = at_quarter + at_three_quarters;
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "S-curve symmetry: f(0.25)+f(0.75)={sum} should be 1.0"
        );
    }
}
