//! Time-varying gain scheduling for dynamic loudness control.
//!
//! Provides `GainPoint`, `GainSchedule`, and `GainScheduleApplier` for
//! defining and applying time-varying gain curves to audio buffers.

#![allow(dead_code)]

/// A single point on a gain curve: a time position and a linear gain multiplier.
#[derive(Debug, Clone, PartialEq)]
pub struct GainPoint {
    /// Time offset from the start of the audio stream, in seconds.
    pub time_s: f64,
    /// Gain value in dB at this point.
    pub gain_db: f64,
}

impl GainPoint {
    /// Create a new gain point.
    pub fn new(time_s: f64, gain_db: f64) -> Self {
        Self { time_s, gain_db }
    }

    /// Linearly interpolate the gain_db toward `other` at the given fraction `t` in [0, 1].
    pub fn interpolate_to(&self, other: &Self, t: f64) -> f64 {
        let t = t.clamp(0.0, 1.0);
        self.gain_db + (other.gain_db - self.gain_db) * t
    }

    /// Convert this point's gain_db to a linear amplitude multiplier.
    #[allow(clippy::cast_precision_loss)]
    pub fn linear_gain(&self) -> f64 {
        10.0_f64.powf(self.gain_db / 20.0)
    }
}

/// A sequence of `GainPoint`s defining a time-varying gain curve.
#[derive(Debug, Clone, Default)]
pub struct GainSchedule {
    points: Vec<GainPoint>,
}

impl GainSchedule {
    /// Create an empty gain schedule.
    pub fn new() -> Self {
        Self { points: Vec::new() }
    }

    /// Add a point to the schedule.  Points are kept sorted by `time_s`.
    pub fn add_point(&mut self, point: GainPoint) {
        let pos = self.points.partition_point(|p| p.time_s <= point.time_s);
        self.points.insert(pos, point);
    }

    /// Return the number of points in the schedule.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Return `true` when there are no points.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Return a reference to the sorted points.
    pub fn points(&self) -> &[GainPoint] {
        &self.points
    }

    /// Interpolate the gain_db at the given time `t_s` in seconds.
    ///
    /// - Before the first point: returns first point's gain.
    /// - After the last point: returns last point's gain.
    /// - Between two points: linearly interpolates.
    pub fn gain_at(&self, t_s: f64) -> f64 {
        match self.points.len() {
            0 => 0.0,
            1 => self.points[0].gain_db,
            _ => {
                if t_s <= self.points[0].time_s {
                    return self.points[0].gain_db;
                }
                // SAFETY: match arm `_` is only reached when len >= 2 (arms 0 and 1 handle smaller
                // cases), so the slice is always non-empty here
                let last = &self.points[self.points.len() - 1];
                if t_s >= last.time_s {
                    return last.gain_db;
                }
                // Find bracketing segment
                let idx = self.points.partition_point(|p| p.time_s <= t_s);
                let prev = &self.points[idx - 1];
                let next = &self.points[idx];
                let span = next.time_s - prev.time_s;
                let frac = if span > 0.0 {
                    (t_s - prev.time_s) / span
                } else {
                    0.0
                };
                prev.interpolate_to(next, frac)
            }
        }
    }

    /// Return the linear amplitude gain at time `t_s`.
    #[allow(clippy::cast_precision_loss)]
    pub fn linear_gain_at(&self, t_s: f64) -> f64 {
        10.0_f64.powf(self.gain_at(t_s) / 20.0)
    }

    /// Return `true` if the gain values are monotonically non-decreasing over time.
    pub fn is_monotonic_increasing(&self) -> bool {
        self.points.windows(2).all(|w| w[1].gain_db >= w[0].gain_db)
    }

    /// Return `true` if the gain values are monotonically non-decreasing or
    /// non-increasing (i.e. always moving in one direction).
    pub fn is_monotonic(&self) -> bool {
        if self.points.len() < 2 {
            return true;
        }
        let increasing = self.points.windows(2).all(|w| w[1].gain_db >= w[0].gain_db);
        let decreasing = self.points.windows(2).all(|w| w[1].gain_db <= w[0].gain_db);
        increasing || decreasing
    }

    /// Total duration covered by this schedule in seconds.
    pub fn duration_s(&self) -> f64 {
        if let (Some(first), Some(last)) = (self.points.first(), self.points.last()) {
            last.time_s - first.time_s
        } else {
            0.0
        }
    }
}

/// Applies a `GainSchedule` to a flat interleaved audio buffer.
pub struct GainScheduleApplier {
    schedule: GainSchedule,
    sample_rate: f64,
    channels: usize,
}

impl GainScheduleApplier {
    /// Create a new applier.
    pub fn new(schedule: GainSchedule, sample_rate: f64, channels: usize) -> Self {
        Self {
            schedule,
            sample_rate,
            channels,
        }
    }

    /// Apply the gain schedule to `buffer` (interleaved, `channels` channels),
    /// starting at `start_time_s` for the beginning of the buffer.
    /// Returns the number of frames processed.
    #[allow(clippy::cast_precision_loss)]
    pub fn apply_to_buffer(&self, buffer: &mut [f32], start_time_s: f64) -> usize {
        if self.channels == 0 || self.sample_rate <= 0.0 {
            return 0;
        }
        let frames = buffer.len() / self.channels;
        for frame in 0..frames {
            let t = start_time_s + (frame as f64) / self.sample_rate;
            let gain = self.schedule.linear_gain_at(t) as f32;
            for ch in 0..self.channels {
                buffer[frame * self.channels + ch] *= gain;
            }
        }
        frames
    }

    /// Return a reference to the underlying schedule.
    pub fn schedule(&self) -> &GainSchedule {
        &self.schedule
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gain_point_new() {
        let p = GainPoint::new(1.0, -6.0);
        assert!((p.time_s - 1.0).abs() < 1e-9);
        assert!((p.gain_db - -6.0).abs() < 1e-9);
    }

    #[test]
    fn test_interpolate_to_midpoint() {
        let a = GainPoint::new(0.0, 0.0);
        let b = GainPoint::new(1.0, 10.0);
        let mid = a.interpolate_to(&b, 0.5);
        assert!((mid - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_interpolate_to_clamps() {
        let a = GainPoint::new(0.0, 0.0);
        let b = GainPoint::new(1.0, 10.0);
        assert!((a.interpolate_to(&b, -0.5) - 0.0).abs() < 1e-9);
        assert!((a.interpolate_to(&b, 1.5) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_linear_gain_0db() {
        let p = GainPoint::new(0.0, 0.0);
        assert!((p.linear_gain() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_gain_schedule_empty() {
        let sched = GainSchedule::new();
        assert!(sched.is_empty());
        assert_eq!(sched.gain_at(1.0), 0.0);
    }

    #[test]
    fn test_gain_schedule_add_point_sorted() {
        let mut sched = GainSchedule::new();
        sched.add_point(GainPoint::new(2.0, -3.0));
        sched.add_point(GainPoint::new(0.0, 0.0));
        sched.add_point(GainPoint::new(1.0, -1.5));
        assert_eq!(sched.points()[0].time_s, 0.0);
        assert_eq!(sched.points()[1].time_s, 1.0);
        assert_eq!(sched.points()[2].time_s, 2.0);
    }

    #[test]
    fn test_gain_at_before_first() {
        let mut sched = GainSchedule::new();
        sched.add_point(GainPoint::new(1.0, -6.0));
        assert!((sched.gain_at(0.0) - -6.0).abs() < 1e-9);
    }

    #[test]
    fn test_gain_at_after_last() {
        let mut sched = GainSchedule::new();
        sched.add_point(GainPoint::new(0.0, 0.0));
        sched.add_point(GainPoint::new(1.0, -6.0));
        assert!((sched.gain_at(5.0) - -6.0).abs() < 1e-9);
    }

    #[test]
    fn test_gain_at_interpolation() {
        let mut sched = GainSchedule::new();
        sched.add_point(GainPoint::new(0.0, 0.0));
        sched.add_point(GainPoint::new(2.0, -4.0));
        // At t=1.0 (midpoint), should be -2.0
        assert!((sched.gain_at(1.0) - -2.0).abs() < 1e-9);
    }

    #[test]
    fn test_is_monotonic_true() {
        let mut sched = GainSchedule::new();
        sched.add_point(GainPoint::new(0.0, -6.0));
        sched.add_point(GainPoint::new(1.0, -3.0));
        sched.add_point(GainPoint::new(2.0, 0.0));
        assert!(sched.is_monotonic());
    }

    #[test]
    fn test_is_monotonic_false() {
        let mut sched = GainSchedule::new();
        sched.add_point(GainPoint::new(0.0, 0.0));
        sched.add_point(GainPoint::new(1.0, -6.0));
        sched.add_point(GainPoint::new(2.0, 3.0));
        assert!(!sched.is_monotonic());
    }

    #[test]
    fn test_apply_to_buffer_unity() {
        let sched = GainSchedule::new(); // 0 dB everywhere
        let applier = GainScheduleApplier::new(sched, 48000.0, 1);
        let mut buf = vec![1.0_f32; 48]; // 48 samples mono
        let frames = applier.apply_to_buffer(&mut buf, 0.0);
        assert_eq!(frames, 48);
        // With 0 dB gain schedule (empty => 0.0 dB => linear 1.0), all samples stay 1.0
        assert!(buf.iter().all(|&s| (s - 1.0).abs() < 1e-5));
    }

    #[test]
    fn test_duration_s() {
        let mut sched = GainSchedule::new();
        sched.add_point(GainPoint::new(1.0, 0.0));
        sched.add_point(GainPoint::new(4.0, -6.0));
        assert!((sched.duration_s() - 3.0).abs() < 1e-9);
    }
}
