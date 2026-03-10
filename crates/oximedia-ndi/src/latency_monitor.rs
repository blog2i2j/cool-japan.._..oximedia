#![allow(dead_code)]
//! NDI stream latency monitoring and alerting for `oximedia-ndi`.
//!
//! Tracks end-to-end latency of NDI streams, maintains a rolling window of
//! measurements, detects latency spikes, and supports configurable threshold
//! alerts.  Useful for live-production environments where low-latency delivery
//! is critical.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::module_name_repetitions)]

use std::collections::VecDeque;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// LatencyThresholds
// ---------------------------------------------------------------------------

/// Configurable threshold levels that drive alerting behaviour.
#[derive(Debug, Clone)]
pub struct LatencyThresholds {
    /// Maximum acceptable latency before a warning is raised.
    pub warning_ms: f64,
    /// Maximum acceptable latency before a critical alert is raised.
    pub critical_ms: f64,
    /// Number of consecutive breaches required before an alert fires.
    pub consecutive_breaches: u32,
}

impl Default for LatencyThresholds {
    fn default() -> Self {
        Self {
            warning_ms: 33.3,     // ~1 frame at 30 fps
            critical_ms: 66.6,    // ~2 frames at 30 fps
            consecutive_breaches: 3,
        }
    }
}

impl LatencyThresholds {
    /// Create thresholds based on a target frame rate.
    pub fn from_fps(fps: f64) -> Self {
        let frame_ms = if fps > 0.0 { 1000.0 / fps } else { 33.3 };
        Self {
            warning_ms: frame_ms,
            critical_ms: frame_ms * 2.0,
            consecutive_breaches: 3,
        }
    }
}

// ---------------------------------------------------------------------------
// AlertLevel
// ---------------------------------------------------------------------------

/// Severity of a latency alert.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AlertLevel {
    /// Latency is within acceptable bounds.
    Normal,
    /// Latency exceeded the warning threshold.
    Warning,
    /// Latency exceeded the critical threshold.
    Critical,
}

// ---------------------------------------------------------------------------
// LatencyAlert
// ---------------------------------------------------------------------------

/// A latency alert generated when thresholds are breached.
#[derive(Debug, Clone)]
pub struct LatencyAlert {
    /// The alert severity.
    pub level: AlertLevel,
    /// The latency value that triggered the alert (milliseconds).
    pub latency_ms: f64,
    /// The configured threshold that was breached.
    pub threshold_ms: f64,
    /// Number of consecutive breaches at the time of the alert.
    pub consecutive: u32,
}

// ---------------------------------------------------------------------------
// LatencySample
// ---------------------------------------------------------------------------

/// A single latency measurement.
#[derive(Debug, Clone, Copy)]
pub struct LatencySample {
    /// Timestamp when the sample was recorded.
    pub timestamp: Instant,
    /// Measured latency in milliseconds.
    pub latency_ms: f64,
}

// ---------------------------------------------------------------------------
// LatencyStats
// ---------------------------------------------------------------------------

/// Aggregated latency statistics over a measurement window.
#[derive(Debug, Clone)]
pub struct LatencyStats {
    /// Number of samples in the window.
    pub count: u64,
    /// Minimum latency observed (ms).
    pub min_ms: f64,
    /// Maximum latency observed (ms).
    pub max_ms: f64,
    /// Mean latency (ms).
    pub mean_ms: f64,
    /// Standard deviation of latency (ms).
    pub std_dev_ms: f64,
    /// 95th-percentile latency (ms).
    pub p95_ms: f64,
    /// 99th-percentile latency (ms).
    pub p99_ms: f64,
    /// Current jitter (absolute difference of last two samples, ms).
    pub jitter_ms: f64,
}

// ---------------------------------------------------------------------------
// LatencyMonitor
// ---------------------------------------------------------------------------

/// Rolling-window latency monitor for a single NDI stream.
///
/// Records latency samples, computes statistics, and generates alerts when
/// configured thresholds are exceeded.
#[derive(Debug)]
pub struct LatencyMonitor {
    /// Human-readable label for the stream being monitored.
    label: String,
    /// Fixed-capacity ring of recent samples.
    samples: VecDeque<LatencySample>,
    /// Maximum number of samples to retain.
    max_samples: usize,
    /// Alert thresholds.
    thresholds: LatencyThresholds,
    /// Consecutive breaches at the warning level.
    warning_streak: u32,
    /// Consecutive breaches at the critical level.
    critical_streak: u32,
    /// Total samples ever recorded.
    total_count: u64,
    /// Accumulated alerts.
    alerts: Vec<LatencyAlert>,
}

impl LatencyMonitor {
    /// Create a new monitor with the given label and default settings.
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            samples: VecDeque::with_capacity(1024),
            max_samples: 1024,
            thresholds: LatencyThresholds::default(),
            warning_streak: 0,
            critical_streak: 0,
            total_count: 0,
            alerts: Vec::new(),
        }
    }

    /// Create a monitor with custom capacity and thresholds.
    pub fn with_config(
        label: impl Into<String>,
        max_samples: usize,
        thresholds: LatencyThresholds,
    ) -> Self {
        let cap = max_samples.max(16);
        Self {
            label: label.into(),
            samples: VecDeque::with_capacity(cap),
            max_samples: cap,
            thresholds,
            warning_streak: 0,
            critical_streak: 0,
            total_count: 0,
            alerts: Vec::new(),
        }
    }

    /// Return the stream label.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Record a latency measurement in milliseconds.
    pub fn record(&mut self, latency_ms: f64) {
        let sample = LatencySample {
            timestamp: Instant::now(),
            latency_ms,
        };
        if self.samples.len() >= self.max_samples {
            self.samples.pop_front();
        }
        self.samples.push_back(sample);
        self.total_count += 1;
        self.evaluate_thresholds(latency_ms);
    }

    /// Record a latency derived from a send and receive timestamp pair.
    pub fn record_timestamps(&mut self, sent: Instant, received: Instant) {
        let elapsed = received.duration_since(sent);
        let ms = elapsed.as_secs_f64() * 1000.0;
        self.record(ms);
    }

    /// Evaluate a sample against the configured thresholds and update streaks.
    fn evaluate_thresholds(&mut self, latency_ms: f64) {
        if latency_ms >= self.thresholds.critical_ms {
            self.critical_streak += 1;
            self.warning_streak += 1;
            if self.critical_streak >= self.thresholds.consecutive_breaches {
                self.alerts.push(LatencyAlert {
                    level: AlertLevel::Critical,
                    latency_ms,
                    threshold_ms: self.thresholds.critical_ms,
                    consecutive: self.critical_streak,
                });
            }
        } else if latency_ms >= self.thresholds.warning_ms {
            self.warning_streak += 1;
            self.critical_streak = 0;
            if self.warning_streak >= self.thresholds.consecutive_breaches {
                self.alerts.push(LatencyAlert {
                    level: AlertLevel::Warning,
                    latency_ms,
                    threshold_ms: self.thresholds.warning_ms,
                    consecutive: self.warning_streak,
                });
            }
        } else {
            self.warning_streak = 0;
            self.critical_streak = 0;
        }
    }

    /// Return the current alert level based on the most recent sample.
    pub fn current_level(&self) -> AlertLevel {
        if let Some(last) = self.samples.back() {
            if last.latency_ms >= self.thresholds.critical_ms {
                AlertLevel::Critical
            } else if last.latency_ms >= self.thresholds.warning_ms {
                AlertLevel::Warning
            } else {
                AlertLevel::Normal
            }
        } else {
            AlertLevel::Normal
        }
    }

    /// Return all accumulated alerts and clear the internal list.
    pub fn drain_alerts(&mut self) -> Vec<LatencyAlert> {
        std::mem::take(&mut self.alerts)
    }

    /// Number of pending alerts.
    pub fn pending_alert_count(&self) -> usize {
        self.alerts.len()
    }

    /// Number of samples currently in the window.
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Total number of samples ever recorded.
    pub fn total_samples(&self) -> u64 {
        self.total_count
    }

    /// Compute jitter from the last two samples.
    pub fn current_jitter_ms(&self) -> f64 {
        if self.samples.len() < 2 {
            return 0.0;
        }
        let n = self.samples.len();
        (self.samples[n - 1].latency_ms - self.samples[n - 2].latency_ms).abs()
    }

    /// Compute aggregate statistics over the current window.
    pub fn stats(&self) -> LatencyStats {
        if self.samples.is_empty() {
            return LatencyStats {
                count: 0,
                min_ms: 0.0,
                max_ms: 0.0,
                mean_ms: 0.0,
                std_dev_ms: 0.0,
                p95_ms: 0.0,
                p99_ms: 0.0,
                jitter_ms: 0.0,
            };
        }

        let n = self.samples.len() as f64;
        let mut sorted: Vec<f64> = self.samples.iter().map(|s| s.latency_ms).collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let min_ms = sorted[0];
        let max_ms = sorted[sorted.len() - 1];
        let sum: f64 = sorted.iter().sum();
        let mean_ms = sum / n;

        let var: f64 = sorted.iter().map(|v| (v - mean_ms).powi(2)).sum::<f64>() / n;
        let std_dev_ms = var.sqrt();

        let p95_ms = percentile(&sorted, 95.0);
        let p99_ms = percentile(&sorted, 99.0);
        let jitter_ms = self.current_jitter_ms();

        LatencyStats {
            count: self.samples.len() as u64,
            min_ms,
            max_ms,
            mean_ms,
            std_dev_ms,
            p95_ms,
            p99_ms,
            jitter_ms,
        }
    }

    /// Reset all samples, streaks, and alerts.
    pub fn reset(&mut self) {
        self.samples.clear();
        self.warning_streak = 0;
        self.critical_streak = 0;
        self.total_count = 0;
        self.alerts.clear();
    }

    /// Return the most recent sample, if any.
    pub fn last_sample(&self) -> Option<&LatencySample> {
        self.samples.back()
    }

    /// Check whether the stream has been stable over the last `n` samples.
    ///
    /// "Stable" means all recent samples are below the warning threshold.
    pub fn is_stable(&self, n: usize) -> bool {
        let check = n.min(self.samples.len());
        if check == 0 {
            return true;
        }
        let start = self.samples.len() - check;
        self.samples
            .range(start..)
            .all(|s| s.latency_ms < self.thresholds.warning_ms)
    }

    /// Return the configured thresholds.
    pub fn thresholds(&self) -> &LatencyThresholds {
        &self.thresholds
    }

    /// Update the thresholds and reset streak counters.
    pub fn set_thresholds(&mut self, thresholds: LatencyThresholds) {
        self.thresholds = thresholds;
        self.warning_streak = 0;
        self.critical_streak = 0;
    }
}

/// Compute the `p`-th percentile of a **sorted** slice.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let idx = (p / 100.0) * (sorted.len() - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil().min((sorted.len() - 1) as f64) as usize;
    let frac = idx - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_thresholds() {
        let t = LatencyThresholds::default();
        assert!(t.warning_ms > 0.0);
        assert!(t.critical_ms > t.warning_ms);
        assert_eq!(t.consecutive_breaches, 3);
    }

    #[test]
    fn test_thresholds_from_fps() {
        let t = LatencyThresholds::from_fps(60.0);
        assert!((t.warning_ms - 1000.0 / 60.0).abs() < 0.01);
        assert!((t.critical_ms - 2.0 * 1000.0 / 60.0).abs() < 0.01);
    }

    #[test]
    fn test_new_monitor() {
        let m = LatencyMonitor::new("test-stream");
        assert_eq!(m.label(), "test-stream");
        assert_eq!(m.sample_count(), 0);
        assert_eq!(m.total_samples(), 0);
    }

    #[test]
    fn test_record_single() {
        let mut m = LatencyMonitor::new("s");
        m.record(5.0);
        assert_eq!(m.sample_count(), 1);
        assert_eq!(m.total_samples(), 1);
    }

    #[test]
    fn test_stats_basic() {
        let mut m = LatencyMonitor::new("s");
        for v in [10.0, 20.0, 30.0, 40.0, 50.0] {
            m.record(v);
        }
        let s = m.stats();
        assert_eq!(s.count, 5);
        assert!((s.mean_ms - 30.0).abs() < 0.001);
        assert!((s.min_ms - 10.0).abs() < 0.001);
        assert!((s.max_ms - 50.0).abs() < 0.001);
    }

    #[test]
    fn test_stats_empty() {
        let m = LatencyMonitor::new("empty");
        let s = m.stats();
        assert_eq!(s.count, 0);
        assert!((s.mean_ms).abs() < 0.001);
    }

    #[test]
    fn test_jitter() {
        let mut m = LatencyMonitor::new("j");
        m.record(10.0);
        m.record(15.0);
        assert!((m.current_jitter_ms() - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_jitter_single_sample() {
        let mut m = LatencyMonitor::new("j");
        m.record(10.0);
        assert!((m.current_jitter_ms()).abs() < 0.001);
    }

    #[test]
    fn test_window_eviction() {
        let mut m = LatencyMonitor::with_config(
            "evict",
            4,
            LatencyThresholds::default(),
        );
        for i in 0..10 {
            m.record(i as f64);
        }
        assert_eq!(m.sample_count(), 4);
        assert_eq!(m.total_samples(), 10);
    }

    #[test]
    fn test_alert_warning() {
        let thresholds = LatencyThresholds {
            warning_ms: 10.0,
            critical_ms: 50.0,
            consecutive_breaches: 2,
        };
        let mut m = LatencyMonitor::with_config("w", 128, thresholds);
        m.record(15.0);
        assert_eq!(m.pending_alert_count(), 0);
        m.record(15.0);
        assert!(m.pending_alert_count() > 0);
        let alerts = m.drain_alerts();
        assert_eq!(alerts[0].level, AlertLevel::Warning);
    }

    #[test]
    fn test_alert_critical() {
        let thresholds = LatencyThresholds {
            warning_ms: 10.0,
            critical_ms: 20.0,
            consecutive_breaches: 1,
        };
        let mut m = LatencyMonitor::with_config("c", 128, thresholds);
        m.record(25.0);
        let alerts = m.drain_alerts();
        assert!(!alerts.is_empty());
        assert_eq!(alerts[0].level, AlertLevel::Critical);
    }

    #[test]
    fn test_current_level() {
        let thresholds = LatencyThresholds {
            warning_ms: 10.0,
            critical_ms: 20.0,
            consecutive_breaches: 1,
        };
        let mut m = LatencyMonitor::with_config("lvl", 128, thresholds);
        assert_eq!(m.current_level(), AlertLevel::Normal);
        m.record(5.0);
        assert_eq!(m.current_level(), AlertLevel::Normal);
        m.record(15.0);
        assert_eq!(m.current_level(), AlertLevel::Warning);
        m.record(25.0);
        assert_eq!(m.current_level(), AlertLevel::Critical);
    }

    #[test]
    fn test_is_stable() {
        let thresholds = LatencyThresholds {
            warning_ms: 20.0,
            critical_ms: 40.0,
            consecutive_breaches: 3,
        };
        let mut m = LatencyMonitor::with_config("st", 128, thresholds);
        for _ in 0..5 {
            m.record(5.0);
        }
        assert!(m.is_stable(5));
        m.record(25.0);
        assert!(!m.is_stable(1));
    }

    #[test]
    fn test_reset() {
        let mut m = LatencyMonitor::new("r");
        m.record(10.0);
        m.record(20.0);
        m.reset();
        assert_eq!(m.sample_count(), 0);
        assert_eq!(m.total_samples(), 0);
        assert_eq!(m.pending_alert_count(), 0);
    }

    #[test]
    fn test_percentile_helper() {
        let sorted = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let p50 = percentile(&sorted, 50.0);
        assert!((p50 - 3.0).abs() < 0.001);
        let p0 = percentile(&sorted, 0.0);
        assert!((p0 - 1.0).abs() < 0.001);
        let p100 = percentile(&sorted, 100.0);
        assert!((p100 - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_record_timestamps() {
        let mut m = LatencyMonitor::new("ts");
        let t0 = Instant::now();
        let t1 = t0 + Duration::from_millis(12);
        m.record_timestamps(t0, t1);
        assert_eq!(m.sample_count(), 1);
        let s = m.last_sample().expect("expected last sample to exist");
        assert!((s.latency_ms - 12.0).abs() < 1.0);
    }

    #[test]
    fn test_set_thresholds_resets_streaks() {
        let t1 = LatencyThresholds {
            warning_ms: 5.0,
            critical_ms: 10.0,
            consecutive_breaches: 1,
        };
        let mut m = LatencyMonitor::with_config("x", 128, t1);
        m.record(8.0); // warning streak = 1
        let t2 = LatencyThresholds {
            warning_ms: 20.0,
            critical_ms: 40.0,
            consecutive_breaches: 3,
        };
        m.set_thresholds(t2);
        // Streaks should be reset - recording a normal value should not alert
        m.record(8.0);
        assert_eq!(m.current_level(), AlertLevel::Normal);
    }
}
