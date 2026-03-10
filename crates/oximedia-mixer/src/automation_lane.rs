//! Per-parameter automation lane for the mixer.
//!
//! An automation lane stores a time-ordered sequence of breakpoints and
//! evaluates the parameter value at any time position via linear interpolation.
//! This is a standalone module; it purposely avoids re-exporting or conflicting
//! with the types defined in the existing `automation` module.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// A single automation breakpoint in a lane.
#[derive(Debug, Clone, PartialEq)]
pub struct LanePoint {
    /// Time position in seconds.
    pub time: f64,
    /// Parameter value at this point (normalised 0.0 – 1.0 or raw dB, etc.).
    pub value: f32,
}

impl LanePoint {
    /// Create a new `LanePoint`.
    #[must_use]
    pub fn new(time: f64, value: f32) -> Self {
        Self { time, value }
    }

    /// Linearly interpolate from `self` to `other` at `query_time`.
    ///
    /// `query_time` must be between `self.time` and `other.time`.
    /// Values outside that range are clamped to the nearest endpoint.
    #[must_use]
    pub fn interpolate_to(&self, other: &Self, query_time: f64) -> f32 {
        if other.time <= self.time {
            return self.value;
        }
        let t = ((query_time - self.time) / (other.time - self.time)).clamp(0.0, 1.0) as f32;
        self.value + (other.value - self.value) * t
    }
}

/// What the automation engine is allowed to do with a lane's data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaneMode {
    /// Play back recorded data; ignore live input.
    Read,
    /// Record live input, overwriting existing data.
    Write,
    /// Record only while the user actively moves the control.
    Touch,
    /// Like `Touch` but latch to the last written value on release.
    Latch,
    /// Automation suspended; pass live input through unchanged.
    Bypass,
}

impl LaneMode {
    /// Returns `true` if the mode allows writing (recording) automation data.
    #[must_use]
    pub fn allows_write(self) -> bool {
        matches!(self, LaneMode::Write | LaneMode::Touch | LaneMode::Latch)
    }

    /// Returns `true` if the mode actively plays back stored automation.
    #[must_use]
    pub fn is_reading(self) -> bool {
        matches!(self, LaneMode::Read | LaneMode::Touch | LaneMode::Latch)
    }
}

/// An automation lane: a sorted collection of `LanePoint` breakpoints.
#[derive(Debug, Clone)]
pub struct AutoLane {
    /// Breakpoints, always kept sorted by `time`.
    points: Vec<LanePoint>,
    /// Operating mode.
    pub mode: LaneMode,
    /// Name of the parameter this lane controls.
    pub parameter_name: String,
    /// Default value returned when the lane is empty.
    pub default_value: f32,
}

impl AutoLane {
    /// Create an empty `AutoLane` for the given parameter.
    #[must_use]
    pub fn new(parameter_name: impl Into<String>, default_value: f32) -> Self {
        Self {
            points: Vec::new(),
            mode: LaneMode::Read,
            parameter_name: parameter_name.into(),
            default_value,
        }
    }

    /// Add a breakpoint, keeping the lane sorted by time.
    ///
    /// If a point already exists at exactly the same time, it is replaced.
    pub fn add_point(&mut self, point: LanePoint) {
        // Remove any existing point at the same time
        self.points.retain(|p| (p.time - point.time).abs() > 1e-9);
        // Insert in sorted position
        let pos = self.points.partition_point(|p| p.time < point.time);
        self.points.insert(pos, point);
    }

    /// Remove the breakpoint nearest to `time` (within `tolerance` seconds).
    ///
    /// Returns `true` if a point was removed.
    pub fn remove_near(&mut self, time: f64, tolerance: f64) -> bool {
        if let Some(idx) = self
            .points
            .iter()
            .position(|p| (p.time - time).abs() <= tolerance)
        {
            self.points.remove(idx);
            true
        } else {
            false
        }
    }

    /// Evaluate the parameter value at `time` seconds.
    ///
    /// - With 0 points: returns `default_value`.
    /// - With 1 point: returns that point's value.
    /// - Before the first point: returns the first point's value.
    /// - After the last point: returns the last point's value.
    /// - Otherwise: linearly interpolates between adjacent breakpoints.
    #[must_use]
    pub fn value_at(&self, time: f64) -> f32 {
        match self.points.len() {
            0 => self.default_value,
            1 => self.points[0].value,
            _ => {
                if time <= self.points[0].time {
                    return self.points[0].value;
                }
                // Safety: match arm guarantees len() >= 2, so last() is always Some
                let last = match self.points.last() {
                    Some(p) => p,
                    None => return self.default_value,
                };
                if time >= last.time {
                    return last.value;
                }
                // Binary search for the segment
                let idx = self.points.partition_point(|p| p.time <= time);
                let lo = &self.points[idx - 1];
                let hi = &self.points[idx];
                lo.interpolate_to(hi, time)
            }
        }
    }

    /// Return the number of breakpoints.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.points.len()
    }

    /// Clear all breakpoints.
    pub fn clear(&mut self) {
        self.points.clear();
    }

    /// Return a slice of all breakpoints (sorted by time).
    #[must_use]
    pub fn points(&self) -> &[LanePoint] {
        &self.points
    }

    /// Return the time span covered by this lane (`first_time`, `last_time`).
    /// Returns `None` if the lane is empty.
    #[must_use]
    pub fn time_span(&self) -> Option<(f64, f64)> {
        if self.points.is_empty() {
            None
        } else {
            let last_time = self
                .points
                .last()
                .map(|p| p.time)
                .unwrap_or(self.points[0].time);
            Some((self.points[0].time, last_time))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lane_point_interpolate_midpoint() {
        let a = LanePoint::new(0.0, 0.0);
        let b = LanePoint::new(1.0, 2.0);
        let v = a.interpolate_to(&b, 0.5);
        assert!((v - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_lane_point_interpolate_at_start() {
        let a = LanePoint::new(0.0, 0.5);
        let b = LanePoint::new(1.0, 1.5);
        assert!((a.interpolate_to(&b, 0.0) - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_lane_point_interpolate_at_end() {
        let a = LanePoint::new(0.0, 0.0);
        let b = LanePoint::new(1.0, 1.0);
        assert!((a.interpolate_to(&b, 1.0) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_lane_point_interpolate_clamped_before() {
        let a = LanePoint::new(1.0, 0.25);
        let b = LanePoint::new(2.0, 0.75);
        // query before start → clamped to a.value
        assert!((a.interpolate_to(&b, 0.0) - 0.25).abs() < 1e-5);
    }

    #[test]
    fn test_lane_mode_allows_write() {
        assert!(!LaneMode::Read.allows_write());
        assert!(LaneMode::Write.allows_write());
        assert!(LaneMode::Touch.allows_write());
        assert!(LaneMode::Latch.allows_write());
        assert!(!LaneMode::Bypass.allows_write());
    }

    #[test]
    fn test_lane_mode_is_reading() {
        assert!(LaneMode::Read.is_reading());
        assert!(!LaneMode::Write.is_reading());
        assert!(LaneMode::Touch.is_reading());
        assert!(LaneMode::Latch.is_reading());
        assert!(!LaneMode::Bypass.is_reading());
    }

    #[test]
    fn test_auto_lane_empty_returns_default() {
        let lane = AutoLane::new("volume", 0.8);
        assert!((lane.value_at(0.0) - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_auto_lane_single_point() {
        let mut lane = AutoLane::new("volume", 0.0);
        lane.add_point(LanePoint::new(1.0, 0.5));
        assert!((lane.value_at(0.0) - 0.5).abs() < 1e-6);
        assert!((lane.value_at(2.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_auto_lane_interpolation() {
        let mut lane = AutoLane::new("pan", 0.0);
        lane.add_point(LanePoint::new(0.0, 0.0));
        lane.add_point(LanePoint::new(4.0, 1.0));
        assert!((lane.value_at(2.0) - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_auto_lane_before_first_point() {
        let mut lane = AutoLane::new("gain", 0.0);
        lane.add_point(LanePoint::new(2.0, 0.3));
        lane.add_point(LanePoint::new(4.0, 0.7));
        assert!((lane.value_at(0.0) - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_auto_lane_after_last_point() {
        let mut lane = AutoLane::new("gain", 0.0);
        lane.add_point(LanePoint::new(0.0, 0.2));
        lane.add_point(LanePoint::new(2.0, 0.9));
        assert!((lane.value_at(99.0) - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_auto_lane_replace_duplicate_time() {
        let mut lane = AutoLane::new("vol", 0.0);
        lane.add_point(LanePoint::new(1.0, 0.3));
        lane.add_point(LanePoint::new(1.0, 0.7)); // replaces
        assert_eq!(lane.point_count(), 1);
        assert!((lane.value_at(1.0) - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_auto_lane_sorted_after_adds() {
        let mut lane = AutoLane::new("send", 0.0);
        lane.add_point(LanePoint::new(3.0, 0.9));
        lane.add_point(LanePoint::new(1.0, 0.1));
        lane.add_point(LanePoint::new(2.0, 0.5));
        assert_eq!(lane.point_count(), 3);
        assert!(lane.points()[0].time < lane.points()[1].time);
        assert!(lane.points()[1].time < lane.points()[2].time);
    }

    #[test]
    fn test_auto_lane_remove_near() {
        let mut lane = AutoLane::new("eq", 0.0);
        lane.add_point(LanePoint::new(1.0, 0.5));
        let removed = lane.remove_near(1.0, 0.01);
        assert!(removed);
        assert_eq!(lane.point_count(), 0);
    }

    #[test]
    fn test_auto_lane_clear() {
        let mut lane = AutoLane::new("vol", 0.0);
        lane.add_point(LanePoint::new(0.0, 0.5));
        lane.add_point(LanePoint::new(1.0, 1.0));
        lane.clear();
        assert_eq!(lane.point_count(), 0);
    }

    #[test]
    fn test_auto_lane_time_span() {
        let mut lane = AutoLane::new("vol", 0.0);
        assert!(lane.time_span().is_none());
        lane.add_point(LanePoint::new(1.0, 0.0));
        lane.add_point(LanePoint::new(5.0, 1.0));
        let (start, end) = lane.time_span().expect("time_span should succeed");
        assert!((start - 1.0).abs() < 1e-9);
        assert!((end - 5.0).abs() < 1e-9);
    }
}
