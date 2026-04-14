#![allow(dead_code)]
//! Stream health monitoring for live media streams.
//!
//! Provides comprehensive per-stream health tracking including:
//! - Bitrate stability via coefficient of variation
//! - Packet-loss rate (short-window + long-window)
//! - Jitter (RFC 3550 inter-arrival jitter estimate)
//! - Rebuffering ratio (stall duration / total playback duration)
//! - Composite health score (0–100) and [`StreamHealthStatus`] classification
//!
//! The monitor is **entirely sync** and allocation-light (fixed-capacity ring
//! buffers).  It is designed to be called from a tokio task or any thread.

use std::collections::VecDeque;
use std::fmt;

// ─── Metric sample ────────────────────────────────────────────────────────────

/// A single observation of stream health metrics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HealthSample {
    /// Wall-clock timestamp in milliseconds (monotonic, arbitrary epoch).
    pub timestamp_ms: u64,
    /// Observed bitrate in kbps for this interval.
    pub bitrate_kbps: f64,
    /// Fraction of packets lost in this interval (0.0 .. 1.0).
    pub packet_loss: f64,
    /// Inter-arrival jitter in microseconds (RFC 3550 estimate).
    pub jitter_us: u64,
    /// Duration of any rebuffering (stall) event in this interval, in ms.
    pub stall_ms: u64,
    /// Duration of normal playback in this interval, in ms.
    pub playback_ms: u64,
}

impl HealthSample {
    /// Creates a new [`HealthSample`].
    ///
    /// `packet_loss` is clamped to `[0.0, 1.0]`.
    #[must_use]
    pub fn new(
        timestamp_ms: u64,
        bitrate_kbps: f64,
        packet_loss: f64,
        jitter_us: u64,
        stall_ms: u64,
        playback_ms: u64,
    ) -> Self {
        Self {
            timestamp_ms,
            bitrate_kbps: bitrate_kbps.max(0.0),
            packet_loss: packet_loss.clamp(0.0, 1.0),
            jitter_us,
            stall_ms,
            playback_ms,
        }
    }

    /// Rebuffering ratio for this sample (stall / (stall + playback)).
    ///
    /// Returns `0.0` when both durations are zero (no data yet).
    #[must_use]
    pub fn rebuffer_ratio(&self) -> f64 {
        let total = self.stall_ms + self.playback_ms;
        if total == 0 {
            return 0.0;
        }
        self.stall_ms as f64 / total as f64
    }
}

// ─── Status classification ────────────────────────────────────────────────────

/// High-level stream health status derived from the composite score.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StreamHealthStatus {
    /// Score ≥ 90 — no measurable impairment.
    Healthy,
    /// Score 70 – 89 — minor degradation; viewer unlikely to notice.
    Degraded,
    /// Score 50 – 69 — noticeable impairment; possible viewer impact.
    Impaired,
    /// Score 20 – 49 — severe degradation; likely viewer complaints.
    Critical,
    /// Score < 20 — stream is effectively unusable.
    Failed,
}

impl fmt::Display for StreamHealthStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Healthy => "Healthy",
            Self::Degraded => "Degraded",
            Self::Impaired => "Impaired",
            Self::Critical => "Critical",
            Self::Failed => "Failed",
        };
        f.write_str(s)
    }
}

impl StreamHealthStatus {
    /// Returns `true` when the stream needs immediate operator attention.
    #[must_use]
    pub fn is_alarm(&self) -> bool {
        matches!(self, Self::Critical | Self::Failed)
    }

    /// Maps a numeric score to a [`StreamHealthStatus`].
    #[must_use]
    pub fn from_score(score: f64) -> Self {
        if score >= 90.0 {
            Self::Healthy
        } else if score >= 70.0 {
            Self::Degraded
        } else if score >= 50.0 {
            Self::Impaired
        } else if score >= 20.0 {
            Self::Critical
        } else {
            Self::Failed
        }
    }
}

// ─── Summary ──────────────────────────────────────────────────────────────────

/// Aggregate health summary produced by [`StreamHealthMonitor::summary`].
#[derive(Debug, Clone, PartialEq)]
pub struct HealthSummary {
    /// Number of samples in the current window.
    pub sample_count: usize,
    /// Mean bitrate in kbps.
    pub mean_bitrate_kbps: f64,
    /// Coefficient of variation of bitrate (std_dev / mean).  `0.0` when mean
    /// is zero or only one sample present.
    pub bitrate_cv: f64,
    /// Mean packet-loss fraction.
    pub mean_packet_loss: f64,
    /// Peak packet-loss fraction in the window.
    pub peak_packet_loss: f64,
    /// Mean jitter in microseconds.
    pub mean_jitter_us: f64,
    /// Peak jitter in microseconds.
    pub peak_jitter_us: u64,
    /// Aggregate rebuffering ratio across the window.
    pub rebuffer_ratio: f64,
    /// Composite health score (0–100; higher is better).
    pub score: f64,
    /// Derived status classification.
    pub status: StreamHealthStatus,
}

// ─── Monitor ──────────────────────────────────────────────────────────────────

/// Configuration for [`StreamHealthMonitor`].
#[derive(Debug, Clone)]
pub struct HealthMonitorConfig {
    /// Maximum number of samples to retain (sliding window).
    pub window_size: usize,
    /// Penalty weight applied per unit of bitrate coefficient of variation.
    /// Default: 15.0  (CV of 1.0 → −15 pts, capped at 25).
    pub cv_penalty_weight: f64,
    /// Penalty per 1% of packet loss.  Default: 20.0  (5 % loss → −100 pts).
    pub loss_penalty_per_pct: f64,
    /// Penalty per 1 ms of mean jitter.  Default: 5.0  (8 ms jitter → −40 pts).
    pub jitter_penalty_per_ms: f64,
    /// Penalty per 1% of rebuffering ratio.  Default: 30.0.
    pub rebuffer_penalty_per_pct: f64,
}

impl Default for HealthMonitorConfig {
    fn default() -> Self {
        Self {
            window_size: 60,
            cv_penalty_weight: 15.0,
            loss_penalty_per_pct: 20.0,
            jitter_penalty_per_ms: 5.0,
            rebuffer_penalty_per_pct: 30.0,
        }
    }
}

/// Sliding-window stream health monitor.
///
/// # Example
/// ```rust
/// use oximedia_net::stream_health_monitor::{
///     HealthSample, StreamHealthMonitor, HealthMonitorConfig,
/// };
/// let mut monitor = StreamHealthMonitor::new(HealthMonitorConfig::default());
/// monitor.push(HealthSample::new(0, 4_000.0, 0.001, 200, 0, 1_000));
/// let summary = monitor.summary();
/// assert!(summary.score > 80.0);
/// ```
#[derive(Debug)]
pub struct StreamHealthMonitor {
    config: HealthMonitorConfig,
    samples: VecDeque<HealthSample>,
}

impl StreamHealthMonitor {
    /// Creates a new monitor with the given configuration.
    ///
    /// # Panics
    /// Panics if `config.window_size == 0`.
    #[must_use]
    pub fn new(config: HealthMonitorConfig) -> Self {
        assert!(config.window_size > 0, "window_size must be > 0");
        let capacity = config.window_size;
        Self {
            config,
            samples: VecDeque::with_capacity(capacity),
        }
    }

    /// Adds a new sample, evicting the oldest when the window is full.
    pub fn push(&mut self, sample: HealthSample) {
        if self.samples.len() == self.config.window_size {
            self.samples.pop_front();
        }
        self.samples.push_back(sample);
    }

    /// Number of samples currently in the window.
    #[must_use]
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Returns `true` when no samples are present.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Clears all samples without changing the configuration.
    pub fn reset(&mut self) {
        self.samples.clear();
    }

    /// Returns the most recent sample, if any.
    #[must_use]
    pub fn latest(&self) -> Option<&HealthSample> {
        self.samples.back()
    }

    /// Computes an aggregate [`HealthSummary`] from the current window.
    ///
    /// When the window is empty, returns a perfect-score summary with all
    /// metrics set to zero.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn summary(&self) -> HealthSummary {
        if self.samples.is_empty() {
            return HealthSummary {
                sample_count: 0,
                mean_bitrate_kbps: 0.0,
                bitrate_cv: 0.0,
                mean_packet_loss: 0.0,
                peak_packet_loss: 0.0,
                mean_jitter_us: 0.0,
                peak_jitter_us: 0,
                rebuffer_ratio: 0.0,
                score: 100.0,
                status: StreamHealthStatus::Healthy,
            };
        }

        let n = self.samples.len() as f64;

        // ── Bitrate ──────────────────────────────────────────────────────────
        let mean_bitrate_kbps: f64 = self.samples.iter().map(|s| s.bitrate_kbps).sum::<f64>() / n;

        let bitrate_cv = if mean_bitrate_kbps > 0.0 && self.samples.len() > 1 {
            let variance: f64 = self
                .samples
                .iter()
                .map(|s| {
                    let diff = s.bitrate_kbps - mean_bitrate_kbps;
                    diff * diff
                })
                .sum::<f64>()
                / n;
            variance.sqrt() / mean_bitrate_kbps
        } else {
            0.0
        };

        // ── Packet loss ───────────────────────────────────────────────────────
        let mean_packet_loss: f64 = self.samples.iter().map(|s| s.packet_loss).sum::<f64>() / n;
        let peak_packet_loss: f64 = self
            .samples
            .iter()
            .map(|s| s.packet_loss)
            .fold(0.0_f64, f64::max);

        // ── Jitter ────────────────────────────────────────────────────────────
        let mean_jitter_us: f64 = self.samples.iter().map(|s| s.jitter_us as f64).sum::<f64>() / n;
        let peak_jitter_us: u64 = self.samples.iter().map(|s| s.jitter_us).max().unwrap_or(0);

        // ── Rebuffering ───────────────────────────────────────────────────────
        let total_stall_ms: u64 = self.samples.iter().map(|s| s.stall_ms).sum();
        let total_playback_ms: u64 = self.samples.iter().map(|s| s.playback_ms).sum();
        let combined_ms = total_stall_ms + total_playback_ms;
        let rebuffer_ratio = if combined_ms == 0 {
            0.0
        } else {
            total_stall_ms as f64 / combined_ms as f64
        };

        // ── Score ─────────────────────────────────────────────────────────────
        let score =
            self.compute_score(bitrate_cv, mean_packet_loss, mean_jitter_us, rebuffer_ratio);

        HealthSummary {
            sample_count: self.samples.len(),
            mean_bitrate_kbps,
            bitrate_cv,
            mean_packet_loss,
            peak_packet_loss,
            mean_jitter_us,
            peak_jitter_us,
            rebuffer_ratio,
            score,
            status: StreamHealthStatus::from_score(score),
        }
    }

    /// Computes the composite health score.
    ///
    /// All penalties are capped individually to prevent one dimension from
    /// dominating the entire score.
    fn compute_score(
        &self,
        bitrate_cv: f64,
        mean_loss: f64,
        mean_jitter_us: f64,
        rebuffer_ratio: f64,
    ) -> f64 {
        // Bitrate instability penalty: CV × weight, cap 25 pts
        let cv_penalty = (bitrate_cv * self.config.cv_penalty_weight).min(25.0);

        // Loss penalty: loss% × weight, cap 50 pts
        let loss_penalty = (mean_loss * 100.0 * self.config.loss_penalty_per_pct).min(50.0);

        // Jitter penalty: jitter_ms × weight, cap 30 pts
        let jitter_ms = mean_jitter_us / 1_000.0;
        let jitter_penalty = (jitter_ms * self.config.jitter_penalty_per_ms).min(30.0);

        // Rebuffer penalty: rebuffer% × weight, cap 50 pts
        let rebuffer_penalty =
            (rebuffer_ratio * 100.0 * self.config.rebuffer_penalty_per_pct).min(50.0);

        (100.0 - cv_penalty - loss_penalty - jitter_penalty - rebuffer_penalty).max(0.0)
    }
}

// ─── Rebuffer event tracker ───────────────────────────────────────────────────

/// Tracks discrete rebuffering (stall) events over time.
///
/// Use this alongside [`StreamHealthMonitor`] when you need fine-grained
/// rebuffering event history in addition to the rolling-window aggregate.
#[derive(Debug, Default)]
pub struct RebufferTracker {
    events: Vec<RebufferEvent>,
    total_stall_ms: u64,
    total_playback_ms: u64,
}

/// A single rebuffering (stall) event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RebufferEvent {
    /// Wall-clock start of the stall (ms).
    pub start_ms: u64,
    /// Duration of the stall (ms).
    pub duration_ms: u64,
}

impl RebufferEvent {
    /// Creates a new [`RebufferEvent`].
    #[must_use]
    pub const fn new(start_ms: u64, duration_ms: u64) -> Self {
        Self {
            start_ms,
            duration_ms,
        }
    }

    /// Wall-clock time when playback resumed (ms).
    #[must_use]
    pub const fn end_ms(&self) -> u64 {
        self.start_ms + self.duration_ms
    }
}

impl RebufferTracker {
    /// Creates a new, empty tracker.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a rebuffering event.
    pub fn record_stall(&mut self, event: RebufferEvent) {
        self.total_stall_ms += event.duration_ms;
        self.events.push(event);
    }

    /// Advances the playback clock by `duration_ms` milliseconds.
    pub fn advance_playback(&mut self, duration_ms: u64) {
        self.total_playback_ms += duration_ms;
    }

    /// Aggregate rebuffering ratio (total stall / (total stall + playback)).
    ///
    /// Returns `0.0` when no data has been recorded.
    #[must_use]
    pub fn rebuffer_ratio(&self) -> f64 {
        let total = self.total_stall_ms + self.total_playback_ms;
        if total == 0 {
            return 0.0;
        }
        self.total_stall_ms as f64 / total as f64
    }

    /// Number of discrete stall events recorded.
    #[must_use]
    pub fn stall_count(&self) -> usize {
        self.events.len()
    }

    /// Total accumulated stall time in milliseconds.
    #[must_use]
    pub const fn total_stall_ms(&self) -> u64 {
        self.total_stall_ms
    }

    /// All recorded stall events (oldest first).
    #[must_use]
    pub fn events(&self) -> &[RebufferEvent] {
        &self.events
    }

    /// Mean stall duration in milliseconds, or `0.0` if no events.
    #[must_use]
    pub fn mean_stall_ms(&self) -> f64 {
        if self.events.is_empty() {
            return 0.0;
        }
        self.total_stall_ms as f64 / self.events.len() as f64
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    fn perfect_sample(ts: u64) -> HealthSample {
        HealthSample::new(ts, 4_000.0, 0.0, 100, 0, 1_000)
    }

    fn make_monitor() -> StreamHealthMonitor {
        StreamHealthMonitor::new(HealthMonitorConfig::default())
    }

    // 1. empty monitor returns perfect score
    #[test]
    fn test_empty_monitor_perfect_score() {
        let m = make_monitor();
        let s = m.summary();
        assert_eq!(s.sample_count, 0);
        assert!((s.score - 100.0).abs() < f64::EPSILON);
        assert_eq!(s.status, StreamHealthStatus::Healthy);
    }

    // 2. single perfect sample
    #[test]
    fn test_single_perfect_sample() {
        let mut m = make_monitor();
        m.push(perfect_sample(0));
        let s = m.summary();
        assert_eq!(s.sample_count, 1);
        assert!(s.score > 90.0, "score={}", s.score);
        assert_eq!(s.status, StreamHealthStatus::Healthy);
    }

    // 3. window eviction at capacity
    #[test]
    fn test_window_eviction() {
        let cfg = HealthMonitorConfig {
            window_size: 3,
            ..Default::default()
        };
        let mut m = StreamHealthMonitor::new(cfg);
        for ts in 0..5u64 {
            m.push(perfect_sample(ts * 1_000));
        }
        assert_eq!(m.len(), 3);
    }

    // 4. high packet loss → Critical or Failed
    #[test]
    fn test_high_packet_loss_status() {
        let mut m = make_monitor();
        for ts in 0..5u64 {
            m.push(HealthSample::new(ts * 1_000, 4_000.0, 0.15, 100, 0, 1_000));
        }
        let s = m.summary();
        assert!(
            s.status == StreamHealthStatus::Critical || s.status == StreamHealthStatus::Failed,
            "status={:?}",
            s.status
        );
    }

    // 5. rebuffering ratio computed correctly
    #[test]
    fn test_rebuffer_ratio() {
        let mut m = make_monitor();
        // 100 ms stall, 900 ms playback → ratio = 0.10
        m.push(HealthSample::new(0, 4_000.0, 0.0, 100, 100, 900));
        let s = m.summary();
        assert!(
            (s.rebuffer_ratio - 0.10).abs() < 1e-9,
            "ratio={}",
            s.rebuffer_ratio
        );
    }

    // 6. bitrate CV zero when all samples identical
    #[test]
    fn test_cv_zero_identical_bitrate() {
        let mut m = make_monitor();
        for ts in 0..4u64 {
            m.push(HealthSample::new(ts * 1_000, 5_000.0, 0.0, 50, 0, 1_000));
        }
        let s = m.summary();
        assert!(s.bitrate_cv < 1e-9, "cv={}", s.bitrate_cv);
    }

    // 7. high jitter drives score down
    #[test]
    fn test_high_jitter_lower_score() {
        let mut low_jitter = make_monitor();
        low_jitter.push(HealthSample::new(0, 4_000.0, 0.0, 100, 0, 1_000));
        let low_score = low_jitter.summary().score;

        let mut high_jitter = make_monitor();
        high_jitter.push(HealthSample::new(0, 4_000.0, 0.0, 10_000_000, 0, 1_000));
        let high_score = high_jitter.summary().score;

        assert!(
            high_score < low_score,
            "high_score={} low_score={}",
            high_score,
            low_score
        );
    }

    // 8. reset clears window
    #[test]
    fn test_reset() {
        let mut m = make_monitor();
        m.push(perfect_sample(0));
        m.reset();
        assert!(m.is_empty());
        assert_eq!(m.len(), 0);
    }

    // 9. latest returns the most recent sample
    #[test]
    fn test_latest() {
        let mut m = make_monitor();
        m.push(perfect_sample(0));
        m.push(perfect_sample(1_000));
        let latest = m.latest().expect("should be Some");
        assert_eq!(latest.timestamp_ms, 1_000);
    }

    // 10. StreamHealthStatus::is_alarm
    #[test]
    fn test_is_alarm() {
        assert!(!StreamHealthStatus::Healthy.is_alarm());
        assert!(!StreamHealthStatus::Degraded.is_alarm());
        assert!(!StreamHealthStatus::Impaired.is_alarm());
        assert!(StreamHealthStatus::Critical.is_alarm());
        assert!(StreamHealthStatus::Failed.is_alarm());
    }

    // 11. StreamHealthStatus Display
    #[test]
    fn test_status_display() {
        assert_eq!(format!("{}", StreamHealthStatus::Healthy), "Healthy");
        assert_eq!(format!("{}", StreamHealthStatus::Failed), "Failed");
    }

    // 12. RebufferTracker records events correctly
    #[test]
    fn test_rebuffer_tracker() {
        let mut tracker = RebufferTracker::new();
        tracker.advance_playback(5_000);
        tracker.record_stall(RebufferEvent::new(5_000, 500));
        tracker.advance_playback(4_500);

        assert_eq!(tracker.stall_count(), 1);
        assert_eq!(tracker.total_stall_ms(), 500);
        assert!((tracker.rebuffer_ratio() - 0.05).abs() < 1e-9);
        assert!((tracker.mean_stall_ms() - 500.0).abs() < f64::EPSILON);
    }

    // 13. RebufferEvent::end_ms
    #[test]
    fn test_rebuffer_event_end_ms() {
        let ev = RebufferEvent::new(1_000, 300);
        assert_eq!(ev.end_ms(), 1_300);
    }

    // 14. HealthSample::rebuffer_ratio zero guard
    #[test]
    fn test_sample_rebuffer_ratio_zero() {
        let s = HealthSample::new(0, 1_000.0, 0.0, 0, 0, 0);
        assert_eq!(s.rebuffer_ratio(), 0.0);
    }

    // 15. score capped at 0 when all dimensions maxed
    #[test]
    fn test_score_floor() {
        let mut m = make_monitor();
        // maximum loss, jitter and rebuffering simultaneously
        m.push(HealthSample::new(0, 100.0, 1.0, 100_000_000, 9_000, 1_000));
        let s = m.summary();
        assert!(s.score >= 0.0);
        assert!(s.score <= 1.0); // should be at floor
    }
}
