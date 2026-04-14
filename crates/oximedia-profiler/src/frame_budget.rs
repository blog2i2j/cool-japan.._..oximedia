//! Real-time frame budget visualization module.
//!
//! Produces per-frame waterfall chart data showing how the frame budget is
//! consumed across processing phases, with over/under-budget annotations.
//!
//! # Example
//!
//! ```
//! use oximedia_profiler::frame_budget::{FrameBudgetTracker, FrameBudgetConfig};
//!
//! let config = FrameBudgetConfig::for_fps(60.0);
//! let mut tracker = FrameBudgetTracker::new(config);
//!
//! tracker.begin_frame(0);
//! tracker.record_phase("decode", 3.2);
//! tracker.record_phase("process", 4.1);
//! tracker.record_phase("encode", 5.5);
//! let frame = tracker.end_frame().expect("frame was open");
//!
//! println!("Over budget: {}", frame.over_budget);
//! ```

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// FrameBudgetConfig
// ---------------------------------------------------------------------------

/// Configuration parameters for frame-budget analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameBudgetConfig {
    /// Total frame budget in milliseconds (e.g. 16.67 ms for 60 fps).
    pub budget_ms: f64,
    /// Target frames per second (informational, derived from `budget_ms`).
    pub target_fps: f64,
    /// Maximum number of recent frames to retain in the rolling window.
    pub window_size: usize,
    /// Fraction of the budget (0.0–1.0) at which a "warning" is issued.
    /// Default: 0.85 (85%).
    pub warning_threshold: f64,
}

impl FrameBudgetConfig {
    /// Creates a configuration targeting a given frames-per-second rate.
    #[must_use]
    pub fn for_fps(fps: f64) -> Self {
        let budget_ms = if fps > 0.0 { 1_000.0 / fps } else { 16.667 };
        Self {
            budget_ms,
            target_fps: fps,
            window_size: 120,
            warning_threshold: 0.85,
        }
    }

    /// Returns the warning threshold in milliseconds.
    #[must_use]
    pub fn warning_ms(&self) -> f64 {
        self.budget_ms * self.warning_threshold
    }
}

impl Default for FrameBudgetConfig {
    fn default() -> Self {
        Self::for_fps(60.0)
    }
}

// ---------------------------------------------------------------------------
// PhaseEntry
// ---------------------------------------------------------------------------

/// A single named phase within a frame's processing pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseEntry {
    /// Phase name (e.g. "decode", "process", "encode").
    pub name: String,
    /// Duration consumed by this phase (milliseconds).
    pub duration_ms: f64,
    /// Cumulative offset from frame start (milliseconds).
    /// This is the start time of this phase in the waterfall chart.
    pub offset_ms: f64,
}

impl PhaseEntry {
    /// Creates a new phase entry.
    #[must_use]
    pub fn new(name: impl Into<String>, duration_ms: f64, offset_ms: f64) -> Self {
        Self {
            name: name.into(),
            duration_ms,
            offset_ms,
        }
    }

    /// Returns the end time (offset + duration) for waterfall rendering.
    #[must_use]
    pub fn end_ms(&self) -> f64 {
        self.offset_ms + self.duration_ms
    }

    /// Returns the fraction of the given budget consumed by this phase.
    #[must_use]
    pub fn budget_fraction(&self, budget_ms: f64) -> f64 {
        if budget_ms > 0.0 {
            self.duration_ms / budget_ms
        } else {
            0.0
        }
    }
}

// ---------------------------------------------------------------------------
// FrameRecord
// ---------------------------------------------------------------------------

/// Complete timing record for a single frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameRecord {
    /// Sequential frame identifier.
    pub frame_id: u64,
    /// Ordered list of phase entries forming the waterfall.
    pub phases: Vec<PhaseEntry>,
    /// Total frame duration (sum of all phases, milliseconds).
    pub total_ms: f64,
    /// Frame budget (milliseconds) at the time this frame was recorded.
    pub budget_ms: f64,
    /// True if total duration exceeded the budget.
    pub over_budget: bool,
    /// True if total duration exceeded the warning threshold.
    pub near_budget: bool,
    /// Remaining budget after all phases (negative if over budget).
    pub slack_ms: f64,
    /// Fraction of budget consumed (> 1.0 means over budget).
    pub budget_utilisation: f64,
}

impl FrameRecord {
    /// Returns the phase that consumed the most time.
    #[must_use]
    pub fn dominant_phase(&self) -> Option<&PhaseEntry> {
        self.phases
            .iter()
            .max_by(|a, b| a.duration_ms.partial_cmp(&b.duration_ms).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Returns true if this frame's dominant phase consumed more than `threshold`
    /// fraction of the budget.
    #[must_use]
    pub fn has_dominant_phase_above(&self, threshold: f64) -> bool {
        self.dominant_phase()
            .map(|p| p.budget_fraction(self.budget_ms) > threshold)
            .unwrap_or(false)
    }
}

// ---------------------------------------------------------------------------
// BudgetStats
// ---------------------------------------------------------------------------

/// Aggregated statistics across a window of recorded frames.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetStats {
    /// Number of frames in the window.
    pub frame_count: usize,
    /// Number of frames that exceeded the budget.
    pub over_budget_count: usize,
    /// Number of frames that exceeded the warning threshold.
    pub near_budget_count: usize,
    /// Mean frame duration (milliseconds).
    pub mean_ms: f64,
    /// Maximum frame duration in the window (milliseconds).
    pub max_ms: f64,
    /// Minimum frame duration in the window (milliseconds).
    pub min_ms: f64,
    /// 95th-percentile frame duration (milliseconds).
    pub p95_ms: f64,
    /// 99th-percentile frame duration (milliseconds).
    pub p99_ms: f64,
    /// Mean per-phase durations (milliseconds).
    pub phase_means_ms: std::collections::HashMap<String, f64>,
}

impl BudgetStats {
    /// Returns the percentage of frames over budget (0.0–100.0).
    #[must_use]
    pub fn over_budget_pct(&self) -> f64 {
        if self.frame_count == 0 {
            0.0
        } else {
            self.over_budget_count as f64 / self.frame_count as f64 * 100.0
        }
    }
}

// ---------------------------------------------------------------------------
// FrameBudgetTracker
// ---------------------------------------------------------------------------

/// Tracks per-frame phase timing and generates waterfall visualization data.
///
/// Maintains a rolling window of [`FrameRecord`]s and provides aggregated
/// statistics across that window.
pub struct FrameBudgetTracker {
    config: FrameBudgetConfig,
    /// Rolling window of completed frames.
    frames: VecDeque<FrameRecord>,
    /// Frame currently being built (frame_id, phases so far, running offset).
    in_flight: Option<(u64, Vec<PhaseEntry>, f64)>,
}

impl FrameBudgetTracker {
    /// Creates a new tracker with the given configuration.
    #[must_use]
    pub fn new(config: FrameBudgetConfig) -> Self {
        Self {
            config,
            frames: VecDeque::new(),
            in_flight: None,
        }
    }

    /// Creates a tracker configured for 60 fps.
    #[must_use]
    pub fn at_60fps() -> Self {
        Self::new(FrameBudgetConfig::for_fps(60.0))
    }

    /// Creates a tracker configured for 30 fps.
    #[must_use]
    pub fn at_30fps() -> Self {
        Self::new(FrameBudgetConfig::for_fps(30.0))
    }

    /// Begins recording a new frame.
    ///
    /// Any previously open frame (not ended with [`end_frame`](Self::end_frame))
    /// is discarded.
    pub fn begin_frame(&mut self, frame_id: u64) {
        self.in_flight = Some((frame_id, Vec::new(), 0.0));
    }

    /// Records a named phase duration within the current open frame.
    ///
    /// Phases are accumulated in insertion order; the cumulative offset is
    /// computed automatically.
    ///
    /// If no frame is currently open (i.e. [`begin_frame`](Self::begin_frame)
    /// has not been called), this is a no-op.
    pub fn record_phase(&mut self, name: impl Into<String>, duration_ms: f64) {
        if let Some((_, phases, offset)) = self.in_flight.as_mut() {
            let entry = PhaseEntry::new(name, duration_ms.max(0.0), *offset);
            *offset += duration_ms.max(0.0);
            phases.push(entry);
        }
    }

    /// Finalises the current open frame and returns the completed [`FrameRecord`].
    ///
    /// Returns `None` if no frame is open.
    pub fn end_frame(&mut self) -> Option<FrameRecord> {
        let (frame_id, phases, _) = self.in_flight.take()?;

        let total_ms: f64 = phases.iter().map(|p| p.duration_ms).sum();
        let budget_ms = self.config.budget_ms;
        let over_budget = total_ms > budget_ms;
        let near_budget = total_ms > self.config.warning_ms();
        let slack_ms = budget_ms - total_ms;
        let budget_utilisation = if budget_ms > 0.0 {
            total_ms / budget_ms
        } else {
            0.0
        };

        let record = FrameRecord {
            frame_id,
            phases,
            total_ms,
            budget_ms,
            over_budget,
            near_budget,
            slack_ms,
            budget_utilisation,
        };

        // Evict oldest frames if the window is full.
        if self.frames.len() >= self.config.window_size {
            self.frames.pop_front();
        }
        self.frames.push_back(record.clone());

        Some(record)
    }

    /// Returns the rolling window of completed frame records.
    #[must_use]
    pub fn frames(&self) -> &VecDeque<FrameRecord> {
        &self.frames
    }

    /// Returns the most recently completed frame record.
    #[must_use]
    pub fn last_frame(&self) -> Option<&FrameRecord> {
        self.frames.back()
    }

    /// Returns whether a frame is currently open.
    #[must_use]
    pub fn is_frame_open(&self) -> bool {
        self.in_flight.is_some()
    }

    /// Clears the rolling window of completed frames.
    pub fn clear_history(&mut self) {
        self.frames.clear();
    }

    /// Computes aggregated [`BudgetStats`] across the current rolling window.
    ///
    /// Returns `None` if no frames have been completed.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn stats(&self) -> Option<BudgetStats> {
        if self.frames.is_empty() {
            return None;
        }

        let frame_count = self.frames.len();
        let mut over_budget_count = 0usize;
        let mut near_budget_count = 0usize;
        let mut sum_ms = 0.0_f64;
        let mut durations: Vec<f64> = Vec::with_capacity(frame_count);
        let mut phase_sums: std::collections::HashMap<String, (f64, usize)> =
            std::collections::HashMap::new();

        for frame in &self.frames {
            if frame.over_budget {
                over_budget_count += 1;
            }
            if frame.near_budget {
                near_budget_count += 1;
            }
            sum_ms += frame.total_ms;
            durations.push(frame.total_ms);

            for phase in &frame.phases {
                let entry = phase_sums
                    .entry(phase.name.clone())
                    .or_insert((0.0, 0));
                entry.0 += phase.duration_ms;
                entry.1 += 1;
            }
        }

        durations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mean_ms = sum_ms / frame_count as f64;
        let max_ms = durations.last().copied().unwrap_or(0.0);
        let min_ms = durations.first().copied().unwrap_or(0.0);

        let percentile = |p: f64| -> f64 {
            if durations.is_empty() {
                return 0.0;
            }
            let idx = ((p / 100.0) * (durations.len() - 1) as f64) as usize;
            durations[idx.min(durations.len() - 1)]
        };

        let p95_ms = percentile(95.0);
        let p99_ms = percentile(99.0);

        let phase_means_ms = phase_sums
            .into_iter()
            .map(|(k, (sum, cnt))| (k, sum / cnt as f64))
            .collect();

        Some(BudgetStats {
            frame_count,
            over_budget_count,
            near_budget_count,
            mean_ms,
            max_ms,
            min_ms,
            p95_ms,
            p99_ms,
            phase_means_ms,
        })
    }

    /// Returns the tracker configuration.
    #[must_use]
    pub fn config(&self) -> &FrameBudgetConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// WaterfallRow  (visualization data)
// ---------------------------------------------------------------------------

/// One row in a waterfall chart, suitable for rendering or serializing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WaterfallRow {
    /// Frame identifier.
    pub frame_id: u64,
    /// Total frame duration.
    pub total_ms: f64,
    /// Whether the frame was over budget.
    pub over_budget: bool,
    /// Serializable list of `(name, start_ms, end_ms)` spans.
    pub spans: Vec<WaterfallSpan>,
}

/// A single span (phase) in a waterfall row.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WaterfallSpan {
    /// Phase name.
    pub name: String,
    /// Start time relative to frame start (milliseconds).
    pub start_ms: f64,
    /// End time relative to frame start (milliseconds).
    pub end_ms: f64,
    /// Duration (milliseconds).
    pub duration_ms: f64,
}

impl WaterfallRow {
    /// Builds a `WaterfallRow` from a completed [`FrameRecord`].
    #[must_use]
    pub fn from_record(record: &FrameRecord) -> Self {
        let spans = record
            .phases
            .iter()
            .map(|p| WaterfallSpan {
                name: p.name.clone(),
                start_ms: p.offset_ms,
                end_ms: p.end_ms(),
                duration_ms: p.duration_ms,
            })
            .collect();

        Self {
            frame_id: record.frame_id,
            total_ms: record.total_ms,
            over_budget: record.over_budget,
            spans,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tracker() -> FrameBudgetTracker {
        FrameBudgetTracker::new(FrameBudgetConfig::for_fps(60.0))
    }

    fn record_frame(tracker: &mut FrameBudgetTracker, id: u64, phases: &[(&str, f64)]) -> FrameRecord {
        tracker.begin_frame(id);
        for (name, dur) in phases {
            tracker.record_phase(*name, *dur);
        }
        tracker.end_frame().expect("frame was open")
    }

    #[test]
    fn test_budget_for_60fps() {
        let cfg = FrameBudgetConfig::for_fps(60.0);
        assert!((cfg.budget_ms - 16.667).abs() < 0.01);
    }

    #[test]
    fn test_budget_for_30fps() {
        let cfg = FrameBudgetConfig::for_fps(30.0);
        assert!((cfg.budget_ms - 33.333).abs() < 0.01);
    }

    #[test]
    fn test_under_budget_frame() {
        let mut t = make_tracker();
        let f = record_frame(&mut t, 0, &[("decode", 3.0), ("process", 4.0), ("encode", 5.0)]);
        assert!(!f.over_budget);
        assert!((f.total_ms - 12.0).abs() < 0.001);
        assert!(f.slack_ms > 0.0);
    }

    #[test]
    fn test_over_budget_frame() {
        let mut t = make_tracker();
        let f = record_frame(&mut t, 1, &[("decode", 8.0), ("process", 5.0), ("encode", 6.0)]);
        // 8+5+6 = 19 ms > 16.667 ms
        assert!(f.over_budget);
        assert!(f.slack_ms < 0.0);
    }

    #[test]
    fn test_phase_offsets_are_cumulative() {
        let mut t = make_tracker();
        t.begin_frame(0);
        t.record_phase("a", 2.0);
        t.record_phase("b", 3.0);
        t.record_phase("c", 4.0);
        let f = t.end_frame().expect("open");
        assert!((f.phases[0].offset_ms - 0.0).abs() < 1e-9);
        assert!((f.phases[1].offset_ms - 2.0).abs() < 1e-9);
        assert!((f.phases[2].offset_ms - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_phase_end_ms() {
        let entry = PhaseEntry::new("test", 5.0, 3.0);
        assert!((entry.end_ms() - 8.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_over_budget_count() {
        let mut t = make_tracker();
        record_frame(&mut t, 0, &[("decode", 20.0)]); // over
        record_frame(&mut t, 1, &[("decode", 5.0)]);  // under
        record_frame(&mut t, 2, &[("decode", 18.0)]); // over
        let stats = t.stats().expect("non-empty");
        assert_eq!(stats.over_budget_count, 2);
        assert_eq!(stats.frame_count, 3);
    }

    #[test]
    fn test_stats_mean_ms() {
        let mut t = make_tracker();
        record_frame(&mut t, 0, &[("phase", 10.0)]);
        record_frame(&mut t, 1, &[("phase", 20.0)]);
        let stats = t.stats().expect("non-empty");
        assert!((stats.mean_ms - 15.0).abs() < 1e-9);
    }

    #[test]
    fn test_waterfall_row_from_record() {
        let mut t = make_tracker();
        let f = record_frame(&mut t, 42, &[("decode", 3.0), ("encode", 5.0)]);
        let row = WaterfallRow::from_record(&f);
        assert_eq!(row.frame_id, 42);
        assert_eq!(row.spans.len(), 2);
        assert_eq!(row.spans[0].name, "decode");
        assert!((row.spans[0].start_ms).abs() < 1e-9);
        assert!((row.spans[1].start_ms - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_window_size_eviction() {
        let config = FrameBudgetConfig {
            budget_ms: 16.667,
            target_fps: 60.0,
            window_size: 3,
            warning_threshold: 0.85,
        };
        let mut t = FrameBudgetTracker::new(config);
        for i in 0..5 {
            record_frame(&mut t, i, &[("phase", 5.0)]);
        }
        assert_eq!(t.frames().len(), 3);
    }

    #[test]
    fn test_end_frame_without_begin_returns_none() {
        let mut t = make_tracker();
        assert!(t.end_frame().is_none());
    }

    #[test]
    fn test_dominant_phase_identified() {
        let mut t = make_tracker();
        let f = record_frame(&mut t, 0, &[("fast", 1.0), ("slow", 10.0), ("medium", 3.0)]);
        let dom = f.dominant_phase().expect("phases exist");
        assert_eq!(dom.name, "slow");
    }

    #[test]
    fn test_stats_phase_means() {
        let mut t = make_tracker();
        record_frame(&mut t, 0, &[("decode", 2.0), ("encode", 4.0)]);
        record_frame(&mut t, 1, &[("decode", 4.0), ("encode", 6.0)]);
        let stats = t.stats().expect("non-empty");
        let decode_mean = stats.phase_means_ms.get("decode").copied().unwrap_or(0.0);
        let encode_mean = stats.phase_means_ms.get("encode").copied().unwrap_or(0.0);
        assert!((decode_mean - 3.0).abs() < 1e-9);
        assert!((encode_mean - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_clear_history() {
        let mut t = make_tracker();
        record_frame(&mut t, 0, &[("phase", 5.0)]);
        record_frame(&mut t, 1, &[("phase", 5.0)]);
        t.clear_history();
        assert_eq!(t.frames().len(), 0);
        assert!(t.stats().is_none());
    }
}
