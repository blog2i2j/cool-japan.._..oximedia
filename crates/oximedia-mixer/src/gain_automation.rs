//! Sample-accurate gain and pan automation recording (Write / Touch / Latch).
//!
//! This module provides a write-side automation engine that records parameter
//! changes as the mixer processes audio in real time.  It complements the
//! existing [`crate::automation_player`] read-back engine.
//!
//! # Automation Modes
//!
//! | Mode    | Behaviour |
//! |---------|-----------|
//! | `Off`   | No recording; existing lane data is untouched. |
//! | `Read`  | No recording; the read-back engine drives parameters. |
//! | `Write` | All incoming values are stamped and appended to the lane. |
//! | `Touch` | Values are recorded only while `touching` is `true`.  When the touch ends the lane reverts to the pre-existing curve. |
//! | `Latch` | Like `Touch` but the last value written is held after the touch ends, overriding the lane until the next playback reset. |
//! | `Trim`  | Incoming values are treated as *offsets* added to the current lane value rather than absolute replacements. |
//!
//! # Example
//!
//! ```rust
//! use oximedia_mixer::gain_automation::{AutomationRecorder, GainLane, RecordMode};
//!
//! let mut lane = GainLane::new();
//! let mut recorder = AutomationRecorder::new(48_000, RecordMode::Write);
//!
//! // Stamp gain = 0.8 at sample 0 and pan = -0.2 at sample 512.
//! recorder.record_gain(&mut lane, 0, 0.8);
//! recorder.record_gain(&mut lane, 512, 0.5);
//!
//! assert_eq!(lane.point_count(), 2);
//! let (_, value) = lane.get_point(0).unwrap();
//! assert!((value - 0.8).abs() < 1e-6);
//! ```

use std::collections::BTreeMap;

// ---------------------------------------------------------------------------
// RecordMode
// ---------------------------------------------------------------------------

/// Automation record mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordMode {
    /// No recording; leave lane data untouched.
    Off,
    /// Playback only; no new data is written.
    Read,
    /// Record all incoming values unconditionally.
    Write,
    /// Record only while `touching` flag is set on the [`AutomationRecorder`].
    Touch,
    /// Like `Touch` but hold the last written value after touch ends.
    Latch,
    /// Record relative (additive) changes on top of the existing lane value.
    Trim,
}

// ---------------------------------------------------------------------------
// GainLane
// ---------------------------------------------------------------------------

/// A single-parameter automation lane storing `(sample_position → value)` pairs.
///
/// Points are stored in a [`BTreeMap`] for efficient ordered access.  The lane
/// supports basic read-back via linear interpolation between adjacent points.
#[derive(Debug, Clone, Default)]
pub struct GainLane {
    /// Breakpoints: sample position → parameter value.
    points: BTreeMap<u64, f32>,
    /// If set, this value overrides interpolation (Latch mode hold value).
    latch_value: Option<f32>,
}

impl GainLane {
    /// Create an empty lane.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of breakpoints stored in the lane.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.points.len()
    }

    /// Insert or overwrite a breakpoint at `sample_pos` with `value`.
    pub fn set_point(&mut self, sample_pos: u64, value: f32) {
        self.points.insert(sample_pos, value);
    }

    /// Remove the breakpoint at `sample_pos`.  Returns `true` if removed.
    pub fn remove_point(&mut self, sample_pos: u64) -> bool {
        self.points.remove(&sample_pos).is_some()
    }

    /// Get a breakpoint by index (insertion order = sorted by sample position).
    ///
    /// Returns `None` if `index >= point_count()`.
    #[must_use]
    pub fn get_point(&self, index: usize) -> Option<(u64, f32)> {
        self.points
            .iter()
            .nth(index)
            .map(|(&pos, &val)| (pos, val))
    }

    /// Evaluate the lane at `sample_pos` using linear interpolation.
    ///
    /// - Before the first point: returns the first point's value.
    /// - After the last point: returns the last point's value (or the latch
    ///   value if set).
    /// - Between two points: linearly interpolates.
    ///
    /// Returns `None` if the lane is empty.
    #[must_use]
    pub fn evaluate(&self, sample_pos: u64) -> Option<f32> {
        if let Some(&latch) = self.latch_value.as_ref() {
            return Some(latch);
        }
        if self.points.is_empty() {
            return None;
        }
        // Find the bracket around sample_pos.
        let mut before: Option<(u64, f32)> = None;
        let mut after: Option<(u64, f32)> = None;
        for (&pos, &val) in &self.points {
            if pos <= sample_pos {
                before = Some((pos, val));
            } else {
                after = Some((pos, val));
                break;
            }
        }
        match (before, after) {
            (None, Some((_, v))) => Some(v),
            (Some((_, v)), None) => Some(v),
            (Some((p0, v0)), Some((p1, v1))) => {
                if p1 == p0 {
                    Some(v0)
                } else {
                    let t = (sample_pos - p0) as f32 / (p1 - p0) as f32;
                    Some(v0 + t * (v1 - v0))
                }
            }
            (None, None) => None,
        }
    }

    /// Clear all breakpoints and reset the latch value.
    pub fn clear(&mut self) {
        self.points.clear();
        self.latch_value = None;
    }

    /// Set the latch hold value, overriding lane interpolation.
    pub fn set_latch(&mut self, value: f32) {
        self.latch_value = Some(value);
    }

    /// Clear the latch hold value (resume normal interpolation).
    pub fn clear_latch(&mut self) {
        self.latch_value = None;
    }

    /// Remove all breakpoints after `sample_pos` (inclusive).
    ///
    /// This is used in Write mode to erase future data when the transport
    /// is recording.
    pub fn erase_from(&mut self, sample_pos: u64) {
        self.points.retain(|&pos, _| pos < sample_pos);
    }

    /// Returns the sample position of the first breakpoint, if any.
    #[must_use]
    pub fn first_pos(&self) -> Option<u64> {
        self.points.keys().next().copied()
    }

    /// Returns the sample position of the last breakpoint, if any.
    #[must_use]
    pub fn last_pos(&self) -> Option<u64> {
        self.points.keys().next_back().copied()
    }
}

// ---------------------------------------------------------------------------
// AutomationRecorder
// ---------------------------------------------------------------------------

/// Records automation data into [`GainLane`]s during realtime processing.
///
/// The recorder tracks touching state (for Touch/Latch modes) and the
/// sample-accurate playhead position.
#[derive(Debug, Clone)]
pub struct AutomationRecorder {
    /// Current sample position (advanced each buffer).
    sample_pos: u64,
    /// Audio sample rate.
    sample_rate: u32,
    /// Current record mode.
    mode: RecordMode,
    /// Whether the user is currently "touching" a control (Touch/Latch modes).
    touching: bool,
    /// Last written value per lane key, used by Latch mode.
    last_values: std::collections::HashMap<String, f32>,
    /// Minimum samples between recorded breakpoints (reduces data density).
    /// 0 = record every value.
    min_interval_samples: u64,
    /// Sample position of the last recorded point per lane key.
    last_record_pos: std::collections::HashMap<String, u64>,
}

impl AutomationRecorder {
    /// Create a new recorder at sample position 0.
    #[must_use]
    pub fn new(sample_rate: u32, mode: RecordMode) -> Self {
        Self {
            sample_pos: 0,
            sample_rate,
            mode,
            touching: false,
            last_values: std::collections::HashMap::new(),
            min_interval_samples: 0,
            last_record_pos: std::collections::HashMap::new(),
        }
    }

    /// Advance the playhead by `samples`.
    pub fn advance(&mut self, samples: u64) {
        self.sample_pos += samples;
    }

    /// Reset the playhead to `pos`.
    pub fn seek(&mut self, pos: u64) {
        self.sample_pos = pos;
    }

    /// Current sample position.
    #[must_use]
    pub fn sample_pos(&self) -> u64 {
        self.sample_pos
    }

    /// Set the record mode.
    pub fn set_mode(&mut self, mode: RecordMode) {
        self.mode = mode;
    }

    /// Get the current record mode.
    #[must_use]
    pub fn mode(&self) -> RecordMode {
        self.mode
    }

    /// Notify the recorder that the user is touching (or has released) a
    /// control.  Only meaningful in `Touch` and `Latch` modes.
    pub fn set_touching(&mut self, touching: bool) {
        self.touching = touching;
    }

    /// Set the minimum interval between breakpoints in milliseconds.
    ///
    /// Lower density reduces lane size at the cost of temporal resolution.
    /// Set to 0 for maximum resolution (one point per `record_gain` call).
    pub fn set_min_interval_ms(&mut self, ms: f32) {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        {
            self.min_interval_samples =
                (ms * self.sample_rate as f32 / 1000.0).max(0.0) as u64;
        }
    }

    /// Returns `true` if the recorder should write a breakpoint for lane `key`
    /// at the current position, given the active mode.
    fn should_record(&self, _key: &str) -> bool {
        match self.mode {
            RecordMode::Off | RecordMode::Read => false,
            RecordMode::Write => true,
            RecordMode::Touch | RecordMode::Trim => self.touching,
            RecordMode::Latch => self.touching,
        }
    }

    /// Check whether the interval constraint is satisfied for lane `key`.
    fn interval_ok(&self, key: &str) -> bool {
        if self.min_interval_samples == 0 {
            return true;
        }
        match self.last_record_pos.get(key) {
            None => true,
            Some(&last) => self.sample_pos.saturating_sub(last) >= self.min_interval_samples,
        }
    }

    /// Record a gain value into `lane`.
    ///
    /// A breakpoint is added only when the current mode and interval constraints
    /// allow it.  In `Trim` mode the incoming `value` is added to whatever the
    /// lane currently evaluates to at the current position.
    pub fn record_gain(&mut self, lane: &mut GainLane, sample_pos: u64, value: f32) {
        let key = "gain".to_string();
        if !self.should_record(&key) {
            return;
        }
        self.sample_pos = sample_pos;
        if !self.interval_ok(&key) {
            return;
        }
        let record_value = if self.mode == RecordMode::Trim {
            let base = lane.evaluate(sample_pos).unwrap_or(0.0);
            (base + value).clamp(0.0, 4.0)
        } else {
            value
        };
        lane.set_point(sample_pos, record_value);
        self.last_values.insert(key.clone(), record_value);
        self.last_record_pos.insert(key, sample_pos);

        // In Write mode erase any existing data ahead of this point.
        if self.mode == RecordMode::Write {
            lane.erase_from(sample_pos + 1);
        }
        // In Latch mode, update the hold value.
        if self.mode == RecordMode::Latch && self.touching {
            lane.set_latch(record_value);
        }
    }

    /// Record a pan value (same logic as `record_gain` but clamped to [-1, 1]).
    pub fn record_pan(&mut self, lane: &mut GainLane, sample_pos: u64, value: f32) {
        let key = "pan".to_string();
        if !self.should_record(&key) {
            return;
        }
        self.sample_pos = sample_pos;
        if !self.interval_ok(&key) {
            return;
        }
        let record_value = if self.mode == RecordMode::Trim {
            let base = lane.evaluate(sample_pos).unwrap_or(0.0);
            (base + value).clamp(-1.0, 1.0)
        } else {
            value.clamp(-1.0, 1.0)
        };
        lane.set_point(sample_pos, record_value);
        self.last_values.insert(key.clone(), record_value);
        self.last_record_pos.insert(key, sample_pos);

        if self.mode == RecordMode::Write {
            lane.erase_from(sample_pos + 1);
        }
        if self.mode == RecordMode::Latch && self.touching {
            lane.set_latch(record_value);
        }
    }

    /// Release touch: in Latch mode, stamp the last value so the held value is
    /// preserved in the lane even after [`clear_latch`](GainLane::clear_latch)
    /// is called.
    pub fn release_touch(&mut self, lane: &mut GainLane) {
        if self.mode == RecordMode::Latch {
            if let Some(&latch) = self.last_values.get("gain") {
                lane.set_point(self.sample_pos, latch);
                lane.clear_latch();
            }
        }
        self.touching = false;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: create a basic Write-mode recorder at 48 kHz.
    fn write_recorder() -> AutomationRecorder {
        AutomationRecorder::new(48_000, RecordMode::Write)
    }

    #[test]
    fn test_write_mode_records_point() {
        let mut lane = GainLane::new();
        let mut rec = write_recorder();
        rec.record_gain(&mut lane, 0, 0.8);
        assert_eq!(lane.point_count(), 1);
        let (pos, val) = lane.get_point(0).unwrap();
        assert_eq!(pos, 0);
        assert!((val - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_off_mode_does_not_record() {
        let mut lane = GainLane::new();
        let mut rec = AutomationRecorder::new(48_000, RecordMode::Off);
        rec.record_gain(&mut lane, 0, 0.8);
        assert_eq!(lane.point_count(), 0);
    }

    #[test]
    fn test_read_mode_does_not_record() {
        let mut lane = GainLane::new();
        let mut rec = AutomationRecorder::new(48_000, RecordMode::Read);
        rec.record_gain(&mut lane, 0, 0.8);
        assert_eq!(lane.point_count(), 0);
    }

    #[test]
    fn test_touch_mode_records_only_while_touching() {
        let mut lane = GainLane::new();
        let mut rec = AutomationRecorder::new(48_000, RecordMode::Touch);
        // Not touching → should not record.
        rec.record_gain(&mut lane, 0, 0.5);
        assert_eq!(lane.point_count(), 0);
        // Start touching → should record.
        rec.set_touching(true);
        rec.record_gain(&mut lane, 512, 0.7);
        assert_eq!(lane.point_count(), 1);
        // Stop touching → no more recording.
        rec.set_touching(false);
        rec.record_gain(&mut lane, 1024, 0.3);
        assert_eq!(lane.point_count(), 1);
    }

    #[test]
    fn test_linear_interpolation() {
        let mut lane = GainLane::new();
        lane.set_point(0, 0.0);
        lane.set_point(100, 1.0);
        let mid = lane.evaluate(50).unwrap();
        assert!((mid - 0.5).abs() < 1e-5, "mid-point should be 0.5, got {mid}");
    }

    #[test]
    fn test_evaluate_before_first_point() {
        let mut lane = GainLane::new();
        lane.set_point(100, 0.9);
        // Before first breakpoint → clamp to first value.
        let val = lane.evaluate(0).unwrap();
        assert!((val - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_evaluate_after_last_point() {
        let mut lane = GainLane::new();
        lane.set_point(0, 0.4);
        lane.set_point(100, 0.8);
        let val = lane.evaluate(9999).unwrap();
        assert!((val - 0.8).abs() < 1e-6, "past last point should hold last value");
    }

    #[test]
    fn test_evaluate_empty_lane_returns_none() {
        let lane = GainLane::new();
        assert!(lane.evaluate(0).is_none());
    }

    #[test]
    fn test_write_mode_erases_future_data() {
        let mut lane = GainLane::new();
        // Pre-populate future data.
        lane.set_point(0, 0.5);
        lane.set_point(1000, 0.9);
        lane.set_point(2000, 1.0);

        let mut rec = write_recorder();
        // Recording at 500 should erase everything at 501+.
        rec.record_gain(&mut lane, 500, 0.3);

        assert!(
            lane.evaluate(1000).map_or(false, |v| (v - 0.3).abs() < 1e-6),
            "Future points should be erased; lane should hold 0.3 after sample 500"
        );
    }

    #[test]
    fn test_trim_mode_adds_offset() {
        let mut lane = GainLane::new();
        lane.set_point(0, 0.5);
        lane.set_point(1000, 0.5);

        let mut rec = AutomationRecorder::new(48_000, RecordMode::Trim);
        rec.set_touching(true);
        // Trim +0.1 at position 500 — base value is 0.5, result should be 0.6.
        rec.record_gain(&mut lane, 500, 0.1);
        let val = lane.evaluate(500).unwrap();
        assert!((val - 0.6).abs() < 1e-5, "trim should add to base: got {val}");
    }

    #[test]
    fn test_latch_mode_holds_value() {
        let mut lane = GainLane::new();
        let mut rec = AutomationRecorder::new(48_000, RecordMode::Latch);
        rec.set_touching(true);
        rec.record_gain(&mut lane, 0, 0.7);
        // Latch value should now be 0.7.
        let val_during_latch = lane.evaluate(99999).unwrap();
        assert!(
            (val_during_latch - 0.7).abs() < 1e-6,
            "latch should hold 0.7, got {val_during_latch}"
        );
        // Release touch → latch cleared, lane holds last stamped value.
        rec.release_touch(&mut lane);
        lane.clear_latch();
        let val_after = lane.evaluate(9999).unwrap();
        assert!(
            (val_after - 0.7).abs() < 1e-6,
            "after latch release, lane should hold 0.7, got {val_after}"
        );
    }

    #[test]
    fn test_min_interval_throttles_recording() {
        let mut lane = GainLane::new();
        let mut rec = write_recorder();
        // Set 10 ms minimum interval at 48 kHz = 480 samples.
        rec.set_min_interval_ms(10.0);
        rec.record_gain(&mut lane, 0, 0.5);
        rec.record_gain(&mut lane, 100, 0.6); // too soon → should be skipped
        rec.record_gain(&mut lane, 480, 0.7); // exactly at interval → should record
        assert_eq!(lane.point_count(), 2, "only 2 points should be written due to throttling");
    }

    #[test]
    fn test_erase_from() {
        let mut lane = GainLane::new();
        lane.set_point(0, 0.1);
        lane.set_point(100, 0.2);
        lane.set_point(200, 0.3);
        lane.erase_from(100);
        assert_eq!(lane.point_count(), 1);
        let (pos, _) = lane.get_point(0).unwrap();
        assert_eq!(pos, 0);
    }

    #[test]
    fn test_first_last_pos() {
        let mut lane = GainLane::new();
        assert!(lane.first_pos().is_none());
        assert!(lane.last_pos().is_none());
        lane.set_point(50, 0.5);
        lane.set_point(200, 0.9);
        assert_eq!(lane.first_pos(), Some(50));
        assert_eq!(lane.last_pos(), Some(200));
    }

    #[test]
    fn test_clear_resets_lane() {
        let mut lane = GainLane::new();
        lane.set_point(0, 1.0);
        lane.set_latch(0.5);
        lane.clear();
        assert_eq!(lane.point_count(), 0);
        assert!(lane.evaluate(0).is_none());
    }

    #[test]
    fn test_pan_recording_clamped() {
        let mut lane = GainLane::new();
        let mut rec = write_recorder();
        rec.set_mode(RecordMode::Write);
        rec.record_pan(&mut lane, 0, 2.5); // > 1.0, should clamp
        let (_, val) = lane.get_point(0).unwrap();
        assert!((val - 1.0).abs() < 1e-6, "pan should be clamped to 1.0, got {val}");
    }

    #[test]
    fn test_advance_seek() {
        let mut rec = write_recorder();
        rec.advance(1000);
        assert_eq!(rec.sample_pos(), 1000);
        rec.seek(0);
        assert_eq!(rec.sample_pos(), 0);
    }
}
