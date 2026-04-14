//! Real-time game stream quality monitoring.
//!
//! Tracks dropped frames, encoding latency variance, bitrate stability,
//! and produces an aggregate QoS (Quality of Service) score suitable for
//! triggering adaptive bitrate changes or alerting the user.
//!
//! # Example
//!
//! ```rust
//! use oximedia_gaming::stream_quality_monitor::{StreamQualityMonitor, QosConfig};
//! use std::time::Duration;
//!
//! let cfg = QosConfig::default();
//! let mut monitor = StreamQualityMonitor::new(cfg);
//!
//! // Feed metrics as frames arrive
//! monitor.record_frame(Duration::from_millis(16), 18_000, false);
//! monitor.record_frame(Duration::from_millis(18), 17_500, false);
//! monitor.record_frame(Duration::from_millis(200), 0, true);  // dropped
//!
//! let report = monitor.report();
//! println!("QoS score: {:.1}", report.qos_score);
//! ```

use std::collections::VecDeque;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the `StreamQualityMonitor`.
#[derive(Debug, Clone)]
pub struct QosConfig {
    /// Target encoding latency. Frames above this threshold count as "late".
    pub target_latency: Duration,
    /// Latency above this is treated as a severe spike (multiplied penalty).
    pub spike_threshold: Duration,
    /// Target bitrate in bits-per-second (used to compute bitrate variance).
    pub target_bitrate_bps: u64,
    /// Rolling window size (number of recent frames to consider).
    pub window_size: usize,
    /// Maximum tolerable drop rate (0.0–1.0) before QoS is severely penalised.
    pub max_drop_rate: f64,
}

impl Default for QosConfig {
    fn default() -> Self {
        Self {
            target_latency: Duration::from_millis(33),
            spike_threshold: Duration::from_millis(100),
            target_bitrate_bps: 6_000_000,
            window_size: 300,
            max_drop_rate: 0.02,
        }
    }
}

// ---------------------------------------------------------------------------
// Per-frame record
// ---------------------------------------------------------------------------

/// Internal record for one captured/encoded frame.
#[derive(Debug, Clone)]
struct FrameRecord {
    /// Time the encoder spent on this frame.
    encode_latency: Duration,
    /// Bytes output by the encoder for this frame (0 if dropped).
    bytes_out: u64,
    /// Whether this frame was dropped before encoding.
    dropped: bool,
}

// ---------------------------------------------------------------------------
// QoS report
// ---------------------------------------------------------------------------

/// A quality-of-service snapshot produced by [`StreamQualityMonitor::report`].
#[derive(Debug, Clone)]
pub struct QosReport {
    /// Aggregate QoS score in the range [0.0, 100.0].
    /// 100 = perfect, 0 = catastrophic.
    pub qos_score: f64,
    /// Frames sampled in the current window.
    pub window_frames: usize,
    /// Number of dropped frames in the window.
    pub dropped_frames: usize,
    /// Drop rate in the window (0.0–1.0).
    pub drop_rate: f64,
    /// Mean encoding latency across non-dropped frames.
    pub mean_latency: Duration,
    /// 95th-percentile encoding latency across non-dropped frames.
    pub p95_latency: Duration,
    /// Number of latency spikes (frames exceeding `QosConfig::spike_threshold`).
    pub latency_spikes: usize,
    /// Mean output bitrate in the window (bits-per-second).
    pub mean_bitrate_bps: f64,
    /// Coefficient of variation of bitrate (std-dev / mean). 0 = perfectly
    /// stable; higher = more variable.
    pub bitrate_cv: f64,
    /// Overall quality classification.
    pub quality_level: QualityLevel,
}

/// Coarse quality classification derived from the QoS score.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualityLevel {
    /// QoS ≥ 90
    Excellent,
    /// QoS ≥ 75
    Good,
    /// QoS ≥ 55
    Fair,
    /// QoS ≥ 35
    Poor,
    /// QoS < 35
    Critical,
}

impl QualityLevel {
    /// Derive the level from a QoS score.
    #[must_use]
    pub fn from_score(score: f64) -> Self {
        if score >= 90.0 {
            Self::Excellent
        } else if score >= 75.0 {
            Self::Good
        } else if score >= 55.0 {
            Self::Fair
        } else if score >= 35.0 {
            Self::Poor
        } else {
            Self::Critical
        }
    }
}

// ---------------------------------------------------------------------------
// Monitor
// ---------------------------------------------------------------------------

/// Rolling-window stream quality monitor.
///
/// Feed per-frame measurements via [`record_frame`][`Self::record_frame`] and
/// retrieve a [`QosReport`] at any time via [`report`][`Self::report`].
pub struct StreamQualityMonitor {
    config: QosConfig,
    window: VecDeque<FrameRecord>,
    /// Cumulative counts since the monitor was created (never reset).
    total_frames: u64,
    total_dropped: u64,
}

impl StreamQualityMonitor {
    /// Create a new monitor with the supplied configuration.
    #[must_use]
    pub fn new(config: QosConfig) -> Self {
        let cap = config.window_size;
        Self {
            config,
            window: VecDeque::with_capacity(cap),
            total_frames: 0,
            total_dropped: 0,
        }
    }

    /// Record a single frame's metrics.
    ///
    /// * `encode_latency` – time the encoder spent on this frame (0 if dropped).
    /// * `bytes_out`      – compressed bytes produced (0 if dropped).
    /// * `dropped`        – `true` if the frame was discarded before encoding.
    pub fn record_frame(&mut self, encode_latency: Duration, bytes_out: u64, dropped: bool) {
        self.total_frames += 1;
        if dropped {
            self.total_dropped += 1;
        }

        if self.window.len() == self.config.window_size {
            self.window.pop_front();
        }
        self.window.push_back(FrameRecord {
            encode_latency,
            bytes_out,
            dropped,
        });
    }

    /// Total frames recorded since the monitor was created.
    #[must_use]
    pub fn total_frames(&self) -> u64 {
        self.total_frames
    }

    /// Total frames dropped since the monitor was created.
    #[must_use]
    pub fn total_dropped(&self) -> u64 {
        self.total_dropped
    }

    /// Produce a [`QosReport`] from the current rolling window.
    #[must_use]
    pub fn report(&self) -> QosReport {
        let window_frames = self.window.len();
        if window_frames == 0 {
            return QosReport {
                qos_score: 100.0,
                window_frames: 0,
                dropped_frames: 0,
                drop_rate: 0.0,
                mean_latency: Duration::ZERO,
                p95_latency: Duration::ZERO,
                latency_spikes: 0,
                mean_bitrate_bps: 0.0,
                bitrate_cv: 0.0,
                quality_level: QualityLevel::Excellent,
            };
        }

        let dropped_frames = self.window.iter().filter(|r| r.dropped).count();
        let drop_rate = dropped_frames as f64 / window_frames as f64;

        // Collect latencies for encoded frames only
        let mut latencies_us: Vec<u64> = self
            .window
            .iter()
            .filter(|r| !r.dropped)
            .map(|r| r.encode_latency.as_micros() as u64)
            .collect();

        let (mean_latency, p95_latency, latency_spikes) = if latencies_us.is_empty() {
            (Duration::ZERO, Duration::ZERO, 0_usize)
        } else {
            let mean_us = latencies_us.iter().sum::<u64>() / latencies_us.len() as u64;
            latencies_us.sort_unstable();
            let p95_idx = ((latencies_us.len() as f64 * 0.95) as usize)
                .min(latencies_us.len().saturating_sub(1));
            let p95_us = latencies_us[p95_idx];
            let spikes = self
                .window
                .iter()
                .filter(|r| !r.dropped && r.encode_latency >= self.config.spike_threshold)
                .count();
            (
                Duration::from_micros(mean_us),
                Duration::from_micros(p95_us),
                spikes,
            )
        };

        // Bitrate stats: assume 1 frame = 1/fps; we don't know fps here so
        // we treat each frame as an independent sample in bits.
        let byte_samples: Vec<f64> = self
            .window
            .iter()
            .filter(|r| !r.dropped)
            .map(|r| (r.bytes_out * 8) as f64)
            .collect();

        let (mean_bitrate_bps, bitrate_cv) = if byte_samples.is_empty() {
            (0.0, 0.0)
        } else {
            let mean = byte_samples.iter().sum::<f64>() / byte_samples.len() as f64;
            let variance = byte_samples
                .iter()
                .map(|b| {
                    let diff = b - mean;
                    diff * diff
                })
                .sum::<f64>()
                / byte_samples.len() as f64;
            let std_dev = variance.sqrt();
            let cv = if mean > 0.0 { std_dev / mean } else { 0.0 };
            (mean, cv)
        };

        let qos_score = self.compute_qos(drop_rate, mean_latency, p95_latency, bitrate_cv);
        let quality_level = QualityLevel::from_score(qos_score);

        QosReport {
            qos_score,
            window_frames,
            dropped_frames,
            drop_rate,
            mean_latency,
            p95_latency,
            latency_spikes,
            mean_bitrate_bps,
            bitrate_cv,
            quality_level,
        }
    }

    /// Compute aggregate QoS score in [0, 100].
    fn compute_qos(
        &self,
        drop_rate: f64,
        mean_latency: Duration,
        p95_latency: Duration,
        bitrate_cv: f64,
    ) -> f64 {
        // --- drop penalty (0–40 points) ---
        let drop_penalty = if drop_rate == 0.0 {
            0.0
        } else {
            let ratio = (drop_rate / self.config.max_drop_rate).min(1.0);
            40.0 * ratio
        };

        // --- latency penalty (0–40 points) ---
        let target_us = self.config.target_latency.as_micros() as f64;
        let spike_us = self.config.spike_threshold.as_micros() as f64;
        let mean_us = mean_latency.as_micros() as f64;
        let p95_us = p95_latency.as_micros() as f64;

        let mean_ratio = if target_us > 0.0 {
            ((mean_us - target_us) / spike_us).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let p95_ratio = if spike_us > 0.0 {
            ((p95_us - target_us) / spike_us).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let latency_penalty = 30.0 * mean_ratio + 10.0 * p95_ratio;

        // --- bitrate instability penalty (0–20 points) ---
        // cv > 0.5 = heavily penalised
        let bitrate_penalty = (20.0 * (bitrate_cv / 0.5)).min(20.0);

        let score = 100.0 - drop_penalty - latency_penalty - bitrate_penalty;
        score.clamp(0.0, 100.0)
    }

    /// Reset the rolling window (but not the cumulative totals).
    pub fn reset_window(&mut self) {
        self.window.clear();
    }

    /// Access the current configuration.
    #[must_use]
    pub fn config(&self) -> &QosConfig {
        &self.config
    }

    /// Update the configuration. The window is cleared on config change.
    pub fn set_config(&mut self, config: QosConfig) {
        self.config = config;
        self.window.clear();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_monitor() -> StreamQualityMonitor {
        StreamQualityMonitor::new(QosConfig {
            target_latency: Duration::from_millis(33),
            spike_threshold: Duration::from_millis(100),
            target_bitrate_bps: 6_000_000,
            window_size: 60,
            max_drop_rate: 0.02,
        })
    }

    #[test]
    fn test_empty_report_is_perfect() {
        let monitor = make_monitor();
        let report = monitor.report();
        assert!((report.qos_score - 100.0).abs() < f64::EPSILON);
        assert_eq!(report.quality_level, QualityLevel::Excellent);
        assert_eq!(report.window_frames, 0);
    }

    #[test]
    fn test_perfect_stream_scores_high() {
        let mut monitor = make_monitor();
        for _ in 0..60 {
            monitor.record_frame(Duration::from_millis(10), 12_000, false);
        }
        let report = monitor.report();
        assert!(report.qos_score >= 90.0, "score={}", report.qos_score);
        assert_eq!(report.quality_level, QualityLevel::Excellent);
        assert_eq!(report.dropped_frames, 0);
        assert!((report.drop_rate - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_high_drop_rate_lowers_score() {
        let mut monitor = make_monitor();
        for i in 0..60 {
            let dropped = i % 3 == 0; // ~33 % drop rate
            monitor.record_frame(
                Duration::from_millis(10),
                if dropped { 0 } else { 12_000 },
                dropped,
            );
        }
        let report = monitor.report();
        assert!(
            report.qos_score < 75.0,
            "expected low score, got {}",
            report.qos_score
        );
        assert!(report.dropped_frames > 0);
    }

    #[test]
    fn test_latency_spikes_lower_score() {
        let mut monitor = make_monitor();
        for i in 0..60 {
            let lat = if i % 5 == 0 {
                Duration::from_millis(200) // spike
            } else {
                Duration::from_millis(15)
            };
            monitor.record_frame(lat, 12_000, false);
        }
        let report = monitor.report();
        assert!(report.latency_spikes > 0);
        assert!(
            report.qos_score < 90.0,
            "expected penalty, got {}",
            report.qos_score
        );
    }

    #[test]
    fn test_window_eviction() {
        let mut monitor = StreamQualityMonitor::new(QosConfig {
            window_size: 10,
            ..QosConfig::default()
        });
        for _ in 0..20 {
            monitor.record_frame(Duration::from_millis(5), 5_000, false);
        }
        assert_eq!(monitor.report().window_frames, 10);
    }

    #[test]
    fn test_total_counters_accumulate() {
        let mut monitor = make_monitor();
        monitor.record_frame(Duration::from_millis(5), 5_000, false);
        monitor.record_frame(Duration::from_millis(5), 0, true);
        monitor.record_frame(Duration::from_millis(5), 5_000, false);
        assert_eq!(monitor.total_frames(), 3);
        assert_eq!(monitor.total_dropped(), 1);
    }

    #[test]
    fn test_reset_window_clears_data() {
        let mut monitor = make_monitor();
        for _ in 0..10 {
            monitor.record_frame(Duration::from_millis(5), 5_000, false);
        }
        monitor.reset_window();
        assert_eq!(monitor.report().window_frames, 0);
        // But cumulative totals survive
        assert_eq!(monitor.total_frames(), 10);
    }

    #[test]
    fn test_quality_level_thresholds() {
        assert_eq!(QualityLevel::from_score(95.0), QualityLevel::Excellent);
        assert_eq!(QualityLevel::from_score(80.0), QualityLevel::Good);
        assert_eq!(QualityLevel::from_score(60.0), QualityLevel::Fair);
        assert_eq!(QualityLevel::from_score(40.0), QualityLevel::Poor);
        assert_eq!(QualityLevel::from_score(10.0), QualityLevel::Critical);
    }

    #[test]
    fn test_set_config_clears_window() {
        let mut monitor = make_monitor();
        for _ in 0..10 {
            monitor.record_frame(Duration::from_millis(5), 5_000, false);
        }
        monitor.set_config(QosConfig::default());
        assert_eq!(monitor.report().window_frames, 0);
    }

    #[test]
    fn test_bitrate_cv_detected() {
        let mut monitor = make_monitor();
        // Alternate between very high and very low bytes → high CV
        for i in 0..60 {
            let bytes = if i % 2 == 0 { 100_000_u64 } else { 100 };
            monitor.record_frame(Duration::from_millis(5), bytes, false);
        }
        let report = monitor.report();
        // CV should be large (> 0.5 triggers max bitrate penalty)
        assert!(
            report.bitrate_cv > 0.5,
            "expected high CV, got {}",
            report.bitrate_cv
        );
    }

    #[test]
    fn test_p95_latency_above_mean() {
        let mut monitor = make_monitor();
        // 95 % of frames are fast, 5 % are slow
        for i in 0..100 {
            let lat = if i < 95 {
                Duration::from_millis(10)
            } else {
                Duration::from_millis(90)
            };
            monitor.record_frame(lat, 8_000, false);
        }
        let report = monitor.report();
        assert!(
            report.p95_latency >= report.mean_latency,
            "p95 must be >= mean"
        );
    }
}
