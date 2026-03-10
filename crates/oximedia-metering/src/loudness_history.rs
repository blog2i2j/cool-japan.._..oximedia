//! Loudness history tracking and trend analysis.
#![allow(dead_code)]

/// A single loudness reading in LUFS.
#[derive(Clone, Debug)]
pub struct LoudnessReading {
    /// Loudness value in LUFS (negative; e.g. -23.0).
    pub lufs: f64,
    /// Monotonic sample index at which this reading was taken.
    pub sample_index: u64,
}

impl LoudnessReading {
    /// Create a new reading.
    pub fn new(lufs: f64, sample_index: u64) -> Self {
        Self { lufs, sample_index }
    }

    /// Return `true` when the loudness is above the given threshold (e.g. threshold = -14.0).
    pub fn is_loud(&self, threshold_lufs: f64) -> bool {
        self.lufs > threshold_lufs
    }
}

/// Rolling history of loudness readings.
#[derive(Debug)]
pub struct LoudnessHistory {
    readings: Vec<LoudnessReading>,
    capacity: usize,
}

impl LoudnessHistory {
    /// Create a new history with the given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            readings: Vec::with_capacity(capacity),
            capacity: capacity.max(1),
        }
    }

    /// Append a new reading, evicting the oldest if at capacity.
    pub fn push(&mut self, reading: LoudnessReading) {
        if self.readings.len() >= self.capacity {
            self.readings.remove(0);
        }
        self.readings.push(reading);
    }

    /// Push a LUFS value with its sample index.
    pub fn push_value(&mut self, lufs: f64, sample_index: u64) {
        self.push(LoudnessReading::new(lufs, sample_index));
    }

    /// Compute the arithmetic average of all readings within the window.
    /// Returns `None` if the history is empty.
    pub fn window_average(&self) -> Option<f64> {
        if self.readings.is_empty() {
            return None;
        }
        let sum: f64 = self.readings.iter().map(|r| r.lufs).sum();
        Some(sum / self.readings.len() as f64)
    }

    /// Return the loudest (maximum) LUFS reading, or `None` if empty.
    pub fn peak(&self) -> Option<f64> {
        self.readings.iter().map(|r| r.lufs).reduce(f64::max)
    }

    /// Return the quietest (minimum) LUFS reading, or `None` if empty.
    pub fn trough(&self) -> Option<f64> {
        self.readings.iter().map(|r| r.lufs).reduce(f64::min)
    }

    /// Number of stored readings.
    pub fn count(&self) -> usize {
        self.readings.len()
    }

    /// `true` when no readings have been pushed yet.
    pub fn is_empty(&self) -> bool {
        self.readings.is_empty()
    }

    /// Immutable slice of all stored readings.
    pub fn readings(&self) -> &[LoudnessReading] {
        &self.readings
    }

    /// Clear all readings.
    pub fn clear(&mut self) {
        self.readings.clear();
    }
}

/// Direction of a loudness trend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrendDirection {
    /// Loudness is rising over the window.
    Rising,
    /// Loudness is falling over the window.
    Falling,
    /// Loudness is roughly stable (change < threshold).
    Stable,
}

/// Trend detector for a `LoudnessHistory`.
#[derive(Debug)]
pub struct LoudnessTrend {
    /// Minimum LU change to classify as rising or falling (default 1.0).
    pub sensitivity_lu: f64,
}

impl LoudnessTrend {
    /// Create a trend detector with the given sensitivity in LU.
    pub fn new(sensitivity_lu: f64) -> Self {
        Self { sensitivity_lu }
    }

    /// Detect the trend from a history window.
    /// Compares the average of the first half with the average of the second half.
    /// Returns `None` if there are fewer than 2 readings.
    pub fn detect_trend(&self, history: &LoudnessHistory) -> Option<TrendDirection> {
        let readings = history.readings();
        if readings.len() < 2 {
            return None;
        }
        let mid = readings.len() / 2;
        let first_half: f64 = readings[..mid].iter().map(|r| r.lufs).sum::<f64>() / mid as f64;
        let second_half: f64 =
            readings[mid..].iter().map(|r| r.lufs).sum::<f64>() / (readings.len() - mid) as f64;
        let delta = second_half - first_half;
        if delta > self.sensitivity_lu {
            Some(TrendDirection::Rising)
        } else if delta < -self.sensitivity_lu {
            Some(TrendDirection::Falling)
        } else {
            Some(TrendDirection::Stable)
        }
    }

    /// Detect trend using default sensitivity (1.0 LU).
    pub fn detect(&self, history: &LoudnessHistory) -> Option<TrendDirection> {
        self.detect_trend(history)
    }
}

impl Default for LoudnessTrend {
    fn default() -> Self {
        Self::new(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── LoudnessReading ──────────────────────────────────────────────────────

    #[test]
    fn reading_is_loud_above_threshold() {
        let r = LoudnessReading::new(-12.0, 0);
        assert!(r.is_loud(-14.0));
    }

    #[test]
    fn reading_not_loud_below_threshold() {
        let r = LoudnessReading::new(-20.0, 0);
        assert!(!r.is_loud(-14.0));
    }

    #[test]
    fn reading_not_loud_exactly_at_threshold() {
        let r = LoudnessReading::new(-14.0, 0);
        // is_loud uses strict `>`, so exactly at threshold is NOT loud
        assert!(!r.is_loud(-14.0));
    }

    // ── LoudnessHistory ──────────────────────────────────────────────────────

    #[test]
    fn history_starts_empty() {
        let h = LoudnessHistory::with_capacity(5);
        assert!(h.is_empty());
        assert_eq!(h.count(), 0);
    }

    #[test]
    fn history_push_stores_reading() {
        let mut h = LoudnessHistory::with_capacity(5);
        h.push_value(-23.0, 0);
        assert_eq!(h.count(), 1);
        assert!((h.readings()[0].lufs - (-23.0)).abs() < 1e-9);
    }

    #[test]
    fn history_window_average_correct() {
        let mut h = LoudnessHistory::with_capacity(3);
        h.push_value(-20.0, 0);
        h.push_value(-22.0, 1);
        h.push_value(-24.0, 2);
        let avg = h.window_average().expect("avg should be valid");
        assert!((avg - (-22.0)).abs() < 1e-9);
    }

    #[test]
    fn history_window_average_none_when_empty() {
        let h = LoudnessHistory::with_capacity(5);
        assert!(h.window_average().is_none());
    }

    #[test]
    fn history_peak_returns_loudest_value() {
        let mut h = LoudnessHistory::with_capacity(5);
        h.push_value(-23.0, 0);
        h.push_value(-10.0, 1);
        h.push_value(-30.0, 2);
        assert!((h.peak().expect("peak should succeed") - (-10.0)).abs() < 1e-9);
    }

    #[test]
    fn history_trough_returns_quietest_value() {
        let mut h = LoudnessHistory::with_capacity(5);
        h.push_value(-23.0, 0);
        h.push_value(-10.0, 1);
        h.push_value(-40.0, 2);
        assert!((h.trough().expect("trough should succeed") - (-40.0)).abs() < 1e-9);
    }

    #[test]
    fn history_evicts_oldest_when_full() {
        let mut h = LoudnessHistory::with_capacity(2);
        h.push_value(-10.0, 0);
        h.push_value(-20.0, 1);
        h.push_value(-30.0, 2); // evicts -10.0
        assert_eq!(h.count(), 2);
        assert!((h.readings()[0].lufs - (-20.0)).abs() < 1e-9);
    }

    #[test]
    fn history_clear_empties_storage() {
        let mut h = LoudnessHistory::with_capacity(4);
        h.push_value(-23.0, 0);
        h.clear();
        assert!(h.is_empty());
    }

    // ── LoudnessTrend ────────────────────────────────────────────────────────

    #[test]
    fn trend_none_for_single_reading() {
        let mut h = LoudnessHistory::with_capacity(5);
        h.push_value(-23.0, 0);
        let t = LoudnessTrend::default();
        assert!(t.detect_trend(&h).is_none());
    }

    #[test]
    fn trend_rising_detected() {
        let mut h = LoudnessHistory::with_capacity(6);
        for i in 0..6u64 {
            h.push_value(-30.0 + i as f64 * 3.0, i);
        }
        let t = LoudnessTrend::new(1.0);
        assert_eq!(t.detect_trend(&h), Some(TrendDirection::Rising));
    }

    #[test]
    fn trend_falling_detected() {
        let mut h = LoudnessHistory::with_capacity(6);
        for i in 0..6u64 {
            h.push_value(-10.0 - i as f64 * 3.0, i);
        }
        let t = LoudnessTrend::new(1.0);
        assert_eq!(t.detect_trend(&h), Some(TrendDirection::Falling));
    }

    #[test]
    fn trend_stable_when_flat() {
        let mut h = LoudnessHistory::with_capacity(4);
        h.push_value(-23.0, 0);
        h.push_value(-23.1, 1);
        h.push_value(-22.9, 2);
        h.push_value(-23.0, 3);
        let t = LoudnessTrend::new(1.0);
        assert_eq!(t.detect_trend(&h), Some(TrendDirection::Stable));
    }

    #[test]
    fn trend_detect_alias_works() {
        let h = LoudnessHistory::with_capacity(2);
        let t = LoudnessTrend::default();
        assert!(t.detect(&h).is_none());
    }
}
