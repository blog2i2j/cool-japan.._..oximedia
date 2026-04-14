//! Real-time quality monitoring with rolling windows and configurable alerting.
//!
//! [`RealtimeQualityMonitor`] ingests per-frame SSIM and VMAF values, maintains
//! bounded sliding windows, detects quality trends (improving / stable /
//! degrading) and emits periodic [`QualitySnapshot`]s with threshold-based
//! [`QualityAlert`]s.
//!
//! # Example
//!
//! ```
//! use oximedia_quality::realtime_quality::{
//!     RealtimeQualityConfig, RealtimeQualityMonitor,
//! };
//!
//! let mut monitor = RealtimeQualityMonitor::new(RealtimeQualityConfig::default());
//! for i in 0..30 {
//!     monitor.push_frame(Some(0.95), Some(82.0));
//! }
//! let snap = monitor.push_frame(Some(0.95), Some(82.0));
//! // snap is Some(_) every update_interval_frames frames
//! ```

use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for [`RealtimeQualityMonitor`].
#[derive(Debug, Clone)]
pub struct RealtimeQualityConfig {
    /// Number of frames kept in the rolling window (default 150 ≈ 5 s @ 30 fps)
    pub window_frames: usize,
    /// Trigger an alert when the window mean SSIM drops below this value (default 0.85)
    pub alert_ssim_drop: f32,
    /// Trigger an alert when the window mean VMAF drops below this value (default 60.0)
    pub alert_vmaf_drop: f32,
    /// Emit a [`QualitySnapshot`] every `update_interval_frames` pushes (default 30)
    pub update_interval_frames: usize,
}

impl RealtimeQualityConfig {
    /// Creates a default configuration (5-second window, moderate thresholds).
    #[must_use]
    pub fn default_config() -> Self {
        Self {
            window_frames: 150,
            alert_ssim_drop: 0.85,
            alert_vmaf_drop: 60.0,
            update_interval_frames: 30,
        }
    }

    /// Strict configuration: smaller window and higher alert thresholds.
    #[must_use]
    pub fn strict() -> Self {
        Self {
            window_frames: 60,
            alert_ssim_drop: 0.92,
            alert_vmaf_drop: 75.0,
            update_interval_frames: 15,
        }
    }

    /// Lenient configuration: larger window and lower alert thresholds.
    #[must_use]
    pub fn lenient() -> Self {
        Self {
            window_frames: 300,
            alert_ssim_drop: 0.70,
            alert_vmaf_drop: 40.0,
            update_interval_frames: 60,
        }
    }
}

impl Default for RealtimeQualityConfig {
    fn default() -> Self {
        Self::default_config()
    }
}

// ---------------------------------------------------------------------------
// Supporting types
// ---------------------------------------------------------------------------

/// Indicates whether quality is improving, holding steady, or declining.
#[derive(Debug, Clone, PartialEq)]
pub enum QualityTrend {
    /// Recent values are higher on average than earlier values in the window
    Improving,
    /// No statistically meaningful change detected
    Stable,
    /// Recent values are lower on average than earlier values in the window
    Degrading,
}

/// A threshold-violation alert emitted by the monitor.
#[derive(Debug, Clone)]
pub struct QualityAlert {
    /// Index of the frame that triggered this alert
    pub frame_index: usize,
    /// Short description of the alert type (e.g. `"ssim_drop"`)
    pub alert_type: String,
    /// The metric value that violated the threshold
    pub value: f32,
    /// The threshold that was violated
    pub threshold: f32,
}

/// Periodic quality summary produced every `update_interval_frames` frames.
#[derive(Debug, Clone)]
pub struct QualitySnapshot {
    /// Frame index at which this snapshot was produced
    pub frame_index: usize,
    /// Mean SSIM across the current window — `None` when no SSIM data available
    pub window_mean_ssim: Option<f32>,
    /// Mean VMAF across the current window — `None` when no VMAF data available
    pub window_mean_vmaf: Option<f32>,
    /// Minimum VMAF value in the current window — `None` when no VMAF data available
    pub window_min_vmaf: Option<f32>,
    /// Trend derived from comparing first vs second half of the SSIM window
    pub trend_ssim: QualityTrend,
    /// Alerts accumulated since the last snapshot (or since monitor creation)
    pub alerts: Vec<QualityAlert>,
}

// ---------------------------------------------------------------------------
// Monitor
// ---------------------------------------------------------------------------

/// Real-time quality monitor that processes a stream of frames one by one.
///
/// Call [`push_frame`](RealtimeQualityMonitor::push_frame) for every encoded
/// frame.  The monitor returns a [`QualitySnapshot`] every
/// `config.update_interval_frames` frames (and `None` otherwise).
pub struct RealtimeQualityMonitor {
    config: RealtimeQualityConfig,
    ssim_window: VecDeque<f32>,
    vmaf_window: VecDeque<f32>,
    frame_count: usize,
    /// Alerts accumulated since the last snapshot flush
    pending_alerts: Vec<QualityAlert>,
    /// All alerts ever generated (preserved after flush for inspection)
    all_alerts: Vec<QualityAlert>,
}

impl RealtimeQualityMonitor {
    /// Creates a new monitor with the supplied configuration.
    #[must_use]
    pub fn new(config: RealtimeQualityConfig) -> Self {
        Self {
            config,
            ssim_window: VecDeque::new(),
            vmaf_window: VecDeque::new(),
            frame_count: 0,
            pending_alerts: Vec::new(),
            all_alerts: Vec::new(),
        }
    }

    /// Pushes a frame's quality metrics into the monitor.
    ///
    /// Returns `Some(QualitySnapshot)` every `update_interval_frames` frames;
    /// returns `None` for intermediate frames.
    ///
    /// `ssim` and `vmaf` may be `None` when the corresponding metric is not
    /// available for a particular frame.
    pub fn push_frame(&mut self, ssim: Option<f32>, vmaf: Option<f32>) -> Option<QualitySnapshot> {
        self.frame_count += 1;

        // Maintain SSIM window
        if let Some(s) = ssim {
            self.ssim_window.push_back(s);
            while self.ssim_window.len() > self.config.window_frames {
                self.ssim_window.pop_front();
            }
        }

        // Maintain VMAF window
        if let Some(v) = vmaf {
            self.vmaf_window.push_back(v);
            while self.vmaf_window.len() > self.config.window_frames {
                self.vmaf_window.pop_front();
            }
        }

        // Check alert conditions against current window means
        let mean_ssim = Self::window_mean(&self.ssim_window);
        let mean_vmaf = Self::window_mean(&self.vmaf_window);

        if let Some(ms) = mean_ssim {
            if ms < self.config.alert_ssim_drop {
                let alert = QualityAlert {
                    frame_index: self.frame_count,
                    alert_type: "ssim_drop".to_string(),
                    value: ms,
                    threshold: self.config.alert_ssim_drop,
                };
                self.all_alerts.push(alert.clone());
                self.pending_alerts.push(alert);
            }
        }

        if let Some(mv) = mean_vmaf {
            if mv < self.config.alert_vmaf_drop {
                let alert = QualityAlert {
                    frame_index: self.frame_count,
                    alert_type: "vmaf_drop".to_string(),
                    value: mv,
                    threshold: self.config.alert_vmaf_drop,
                };
                self.all_alerts.push(alert.clone());
                self.pending_alerts.push(alert);
            }
        }

        // Emit snapshot only on interval
        let interval = self.config.update_interval_frames.max(1);
        if self.frame_count % interval != 0 {
            return None;
        }

        let min_vmaf = self
            .vmaf_window
            .iter()
            .copied()
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let trend_ssim = Self::compute_trend(&self.ssim_window);

        let alerts = std::mem::take(&mut self.pending_alerts);

        Some(QualitySnapshot {
            frame_index: self.frame_count,
            window_mean_ssim: mean_ssim,
            window_mean_vmaf: mean_vmaf,
            window_min_vmaf: min_vmaf,
            trend_ssim,
            alerts,
        })
    }

    /// Returns all alerts ever generated (including flushed ones).
    #[must_use]
    pub fn all_alerts(&self) -> &[QualityAlert] {
        &self.all_alerts
    }

    /// Returns the total number of frames pushed so far.
    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.frame_count
    }

    /// Returns the current window mean SSIM, or `None` when the window is empty.
    #[must_use]
    pub fn current_mean_ssim(&self) -> Option<f32> {
        Self::window_mean(&self.ssim_window)
    }

    /// Returns the current window mean VMAF, or `None` when the window is empty.
    #[must_use]
    pub fn current_mean_vmaf(&self) -> Option<f32> {
        Self::window_mean(&self.vmaf_window)
    }

    /// Resets the monitor to its initial state (clears all windows, counters, and alerts).
    pub fn reset(&mut self) {
        self.ssim_window.clear();
        self.vmaf_window.clear();
        self.frame_count = 0;
        self.pending_alerts.clear();
        self.all_alerts.clear();
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    fn window_mean(window: &VecDeque<f32>) -> Option<f32> {
        if window.is_empty() {
            return None;
        }
        Some(window.iter().sum::<f32>() / window.len() as f32)
    }

    /// Computes trend by comparing the mean of the first half vs the second half.
    ///
    /// * delta > 0.01 → Improving
    /// * delta < −0.01 → Degrading
    /// * otherwise → Stable
    fn compute_trend(window: &VecDeque<f32>) -> QualityTrend {
        let n = window.len();
        if n < 2 {
            return QualityTrend::Stable;
        }

        let half = n / 2;
        let first_half: Vec<f32> = window.iter().copied().take(half).collect();
        let second_half: Vec<f32> = window.iter().copied().skip(n - half).collect();

        let mean_first = if first_half.is_empty() {
            return QualityTrend::Stable;
        } else {
            first_half.iter().sum::<f32>() / first_half.len() as f32
        };

        let mean_second = if second_half.is_empty() {
            return QualityTrend::Stable;
        } else {
            second_half.iter().sum::<f32>() / second_half.len() as f32
        };

        let delta = mean_second - mean_first;
        if delta > 0.01 {
            QualityTrend::Improving
        } else if delta < -0.01 {
            QualityTrend::Degrading
        } else {
            QualityTrend::Stable
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_monitor(window: usize, interval: usize) -> RealtimeQualityMonitor {
        RealtimeQualityMonitor::new(RealtimeQualityConfig {
            window_frames: window,
            update_interval_frames: interval,
            alert_ssim_drop: 0.85,
            alert_vmaf_drop: 60.0,
        })
    }

    #[test]
    fn test_push_frame_returns_snapshot_at_interval() {
        let mut monitor = make_monitor(150, 5);
        for _ in 0..4 {
            assert!(monitor.push_frame(Some(0.95), Some(80.0)).is_none());
        }
        // 5th push should return a snapshot
        let snap = monitor.push_frame(Some(0.95), Some(80.0));
        assert!(snap.is_some(), "expected snapshot at frame 5");
    }

    #[test]
    fn test_no_snapshot_between_intervals() {
        let mut monitor = make_monitor(150, 10);
        for i in 0..9 {
            let result = monitor.push_frame(Some(0.9), Some(75.0));
            assert!(result.is_none(), "frame {i} should not yield a snapshot");
        }
        let snap = monitor.push_frame(Some(0.9), Some(75.0));
        assert!(snap.is_some(), "frame 10 should yield a snapshot");
    }

    #[test]
    fn test_window_bounded() {
        let mut monitor = make_monitor(5, 1000);
        for i in 0..20 {
            monitor.push_frame(Some(i as f32 * 0.01 + 0.80), Some(70.0 + i as f32));
        }
        // Window is bounded to 5; we pushed SSIM from 0.80..0.99
        // After 20 pushes the window holds the last 5 (frames 15..19 → ssim 0.95..0.99)
        let mean = monitor.current_mean_ssim().unwrap_or(0.0);
        assert!(
            mean >= 0.94,
            "window mean should reflect recent frames, got {mean}"
        );
        assert!(mean <= 1.0);
    }

    #[test]
    fn test_alert_generated_on_vmaf_drop() {
        let mut monitor = RealtimeQualityMonitor::new(RealtimeQualityConfig {
            window_frames: 10,
            alert_ssim_drop: 0.85,
            alert_vmaf_drop: 60.0,
            update_interval_frames: 1000, // prevent auto snapshot
        });
        // Push frames with VMAF well below threshold
        for _ in 0..5 {
            monitor.push_frame(Some(0.9), Some(40.0));
        }
        let alerts = monitor.all_alerts();
        assert!(!alerts.is_empty(), "expected vmaf_drop alerts");
        assert!(alerts.iter().any(|a| a.alert_type == "vmaf_drop"));
    }

    #[test]
    fn test_alert_generated_on_ssim_drop() {
        let mut monitor = RealtimeQualityMonitor::new(RealtimeQualityConfig {
            window_frames: 10,
            alert_ssim_drop: 0.85,
            alert_vmaf_drop: 60.0,
            update_interval_frames: 1000,
        });
        for _ in 0..5 {
            monitor.push_frame(Some(0.70), Some(75.0));
        }
        let alerts = monitor.all_alerts();
        assert!(alerts.iter().any(|a| a.alert_type == "ssim_drop"));
    }

    #[test]
    fn test_trend_degrading() {
        // Build a window that's clearly degrading: high SSIM first, low SSIM later
        let config = RealtimeQualityConfig {
            window_frames: 20,
            alert_ssim_drop: 0.0, // suppress alerts
            alert_vmaf_drop: 0.0,
            update_interval_frames: 20,
        };
        let mut monitor = RealtimeQualityMonitor::new(config);
        for _ in 0..10 {
            monitor.push_frame(Some(0.98), None);
        }
        for _ in 0..9 {
            monitor.push_frame(Some(0.70), None);
        }
        let snap = monitor
            .push_frame(Some(0.70), None)
            .expect("snapshot at frame 20");
        assert_eq!(snap.trend_ssim, QualityTrend::Degrading);
    }

    #[test]
    fn test_trend_improving() {
        let config = RealtimeQualityConfig {
            window_frames: 20,
            alert_ssim_drop: 0.0,
            alert_vmaf_drop: 0.0,
            update_interval_frames: 20,
        };
        let mut monitor = RealtimeQualityMonitor::new(config);
        for _ in 0..10 {
            monitor.push_frame(Some(0.70), None);
        }
        for _ in 0..9 {
            monitor.push_frame(Some(0.98), None);
        }
        let snap = monitor
            .push_frame(Some(0.98), None)
            .expect("snapshot at frame 20");
        assert_eq!(snap.trend_ssim, QualityTrend::Improving);
    }

    #[test]
    fn test_reset_clears_state() {
        let mut monitor = make_monitor(150, 5);
        for _ in 0..10 {
            monitor.push_frame(Some(0.95), Some(80.0));
        }
        assert_eq!(monitor.frame_count(), 10);
        monitor.reset();
        assert_eq!(monitor.frame_count(), 0);
        assert!(monitor.current_mean_ssim().is_none());
        assert!(monitor.current_mean_vmaf().is_none());
        assert!(monitor.all_alerts().is_empty());
    }

    #[test]
    fn test_mean_computed_correctly() {
        let mut monitor = make_monitor(150, 1000);
        monitor.push_frame(Some(0.90), Some(70.0));
        monitor.push_frame(Some(0.80), Some(90.0));
        let ssim = monitor.current_mean_ssim().unwrap_or(0.0);
        let vmaf = monitor.current_mean_vmaf().unwrap_or(0.0);
        assert!(
            (ssim - 0.85).abs() < 0.001,
            "mean ssim expected 0.85, got {ssim}"
        );
        assert!(
            (vmaf - 80.0).abs() < 0.001,
            "mean vmaf expected 80, got {vmaf}"
        );
    }

    #[test]
    fn test_no_alerts_without_drops() {
        let mut monitor = make_monitor(150, 1000);
        for _ in 0..50 {
            monitor.push_frame(Some(0.95), Some(85.0));
        }
        assert!(
            monitor.all_alerts().is_empty(),
            "no alerts expected for healthy frames"
        );
    }

    #[test]
    fn test_snapshot_contains_window_min_vmaf() {
        let config = RealtimeQualityConfig {
            window_frames: 10,
            alert_ssim_drop: 0.0,
            alert_vmaf_drop: 0.0,
            update_interval_frames: 3,
        };
        let mut monitor = RealtimeQualityMonitor::new(config);
        monitor.push_frame(None, Some(90.0));
        monitor.push_frame(None, Some(60.0));
        let snap = monitor
            .push_frame(None, Some(80.0))
            .expect("snapshot at frame 3");
        let min_vmaf = snap.window_min_vmaf.unwrap_or(0.0);
        assert!(
            (min_vmaf - 60.0).abs() < 0.001,
            "min vmaf expected 60, got {min_vmaf}"
        );
    }

    #[test]
    fn test_strict_config_has_higher_thresholds() {
        let strict = RealtimeQualityConfig::strict();
        let lenient = RealtimeQualityConfig::lenient();
        assert!(strict.alert_ssim_drop > lenient.alert_ssim_drop);
        assert!(strict.alert_vmaf_drop > lenient.alert_vmaf_drop);
        assert!(strict.window_frames < lenient.window_frames);
    }

    #[test]
    fn test_default_config_via_trait() {
        let config = RealtimeQualityConfig::default();
        assert_eq!(config.window_frames, 150);
        assert_eq!(config.update_interval_frames, 30);
    }
}

// ---------------------------------------------------------------------------
// RealtimeQualityGate
// ---------------------------------------------------------------------------

/// Stateless quality gate that accepts or rejects a single-frame quality score
/// against a configured minimum threshold.
///
/// # Example
///
/// ```
/// use oximedia_quality::realtime_quality::RealtimeQualityGate;
///
/// let gate = RealtimeQualityGate::new(0.80);
/// assert!(gate.check(0.85));
/// assert!(!gate.check(0.75));
/// ```
pub struct RealtimeQualityGate {
    min_score: f32,
}

impl RealtimeQualityGate {
    /// Creates a gate that passes scores ≥ `min_score`.
    #[must_use]
    pub fn new(min_score: f32) -> Self {
        Self { min_score }
    }

    /// Returns `true` when `score >= min_score`.
    #[must_use]
    pub fn check(&self, score: f32) -> bool {
        score >= self.min_score
    }

    /// Returns the configured minimum score threshold.
    #[must_use]
    pub fn threshold(&self) -> f32 {
        self.min_score
    }
}

#[cfg(test)]
mod gate_tests {
    use super::RealtimeQualityGate;

    #[test]
    fn test_gate_passes_above_threshold() {
        let gate = RealtimeQualityGate::new(0.80);
        assert!(gate.check(0.85));
        assert!(gate.check(0.80));
    }

    #[test]
    fn test_gate_rejects_below_threshold() {
        let gate = RealtimeQualityGate::new(0.80);
        assert!(!gate.check(0.79));
        assert!(!gate.check(0.0));
    }

    #[test]
    fn test_gate_at_boundary() {
        let gate = RealtimeQualityGate::new(0.75);
        assert!(gate.check(0.75));
    }

    #[test]
    fn test_gate_threshold_accessor() {
        let gate = RealtimeQualityGate::new(0.65);
        assert!((gate.threshold() - 0.65).abs() < f32::EPSILON);
    }
}
