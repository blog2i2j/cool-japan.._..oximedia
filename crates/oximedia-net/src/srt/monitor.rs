//! SRT connection monitoring and quality metrics.
//!
//! Provides utilities for monitoring connection quality and bandwidth.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Bandwidth estimator using sliding window.
#[derive(Debug)]
pub struct BandwidthEstimator {
    /// Sliding window of samples.
    samples: VecDeque<BandwidthSample>,
    /// Window duration.
    window_duration: Duration,
    /// Total bytes in window.
    total_bytes: u64,
}

/// A single bandwidth sample.
#[derive(Debug, Clone)]
struct BandwidthSample {
    /// Timestamp of the sample.
    timestamp: Instant,
    /// Bytes transferred.
    bytes: u64,
}

impl BandwidthEstimator {
    /// Creates a new bandwidth estimator.
    #[must_use]
    pub fn new(window_duration: Duration) -> Self {
        Self {
            samples: VecDeque::new(),
            window_duration,
            total_bytes: 0,
        }
    }

    /// Records a new sample.
    pub fn record(&mut self, bytes: u64) {
        let now = Instant::now();

        // Remove old samples outside the window
        while let Some(sample) = self.samples.front() {
            if now.duration_since(sample.timestamp) > self.window_duration {
                // Front was confirmed by `while let Some(sample)`.
                if let Some(old) = self.samples.pop_front() {
                    self.total_bytes = self.total_bytes.saturating_sub(old.bytes);
                }
            } else {
                break;
            }
        }

        // Add new sample
        self.samples.push_back(BandwidthSample {
            timestamp: now,
            bytes,
        });
        self.total_bytes += bytes;
    }

    /// Returns current bandwidth estimate in bytes per second.
    #[must_use]
    pub fn estimate(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }

        let oldest = match self.samples.front() {
            Some(s) => s.timestamp,
            None => return 0.0,
        };
        let duration = Instant::now().duration_since(oldest).as_secs_f64();

        if duration > 0.0 {
            self.total_bytes as f64 / duration
        } else {
            0.0
        }
    }

    /// Returns bandwidth in megabits per second.
    #[must_use]
    pub fn mbps(&self) -> f64 {
        (self.estimate() * 8.0) / 1_000_000.0
    }

    /// Clears all samples.
    pub fn reset(&mut self) {
        self.samples.clear();
        self.total_bytes = 0;
    }
}

/// Connection quality metrics.
#[derive(Debug, Clone, Default)]
pub struct QualityMetrics {
    /// Current RTT in microseconds.
    pub rtt: u32,
    /// RTT variance in microseconds.
    pub rtt_var: u32,
    /// Packet loss rate (0.0 to 1.0).
    pub loss_rate: f64,
    /// Available bandwidth estimate (bytes/sec).
    pub bandwidth: f64,
    /// Send buffer utilization (0.0 to 1.0).
    pub send_buffer_util: f64,
    /// Receive buffer utilization (0.0 to 1.0).
    pub recv_buffer_util: f64,
    /// Number of retransmissions.
    pub retransmit_count: u64,
    /// Jitter in microseconds.
    pub jitter: u32,
}

impl QualityMetrics {
    /// Creates new quality metrics.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            rtt: 0,
            rtt_var: 0,
            loss_rate: 0.0,
            bandwidth: 0.0,
            send_buffer_util: 0.0,
            recv_buffer_util: 0.0,
            retransmit_count: 0,
            jitter: 0,
        }
    }

    /// Returns true if quality is good (low loss, low RTT).
    #[must_use]
    pub const fn is_good(&self) -> bool {
        self.loss_rate < 0.01 && self.rtt < 100_000
    }

    /// Returns true if quality is degraded.
    #[must_use]
    pub const fn is_degraded(&self) -> bool {
        self.loss_rate > 0.05 || self.rtt > 500_000
    }

    /// Returns a quality score (0.0 to 100.0).
    #[must_use]
    pub fn quality_score(&self) -> f64 {
        let mut score = 100.0;

        // Penalize for packet loss
        score -= self.loss_rate * 1000.0;

        // Penalize for high RTT
        let rtt_ms = self.rtt as f64 / 1000.0;
        if rtt_ms > 50.0 {
            score -= (rtt_ms - 50.0) * 0.5;
        }

        // Penalize for high jitter
        let jitter_ms = self.jitter as f64 / 1000.0;
        if jitter_ms > 10.0 {
            score -= (jitter_ms - 10.0) * 0.3;
        }

        score.clamp(0.0, 100.0)
    }
}

/// Jitter calculator using RFC 3550 algorithm.
#[derive(Debug)]
pub struct JitterCalculator {
    /// Last arrival time.
    last_arrival: Option<Instant>,
    /// Last RTP timestamp.
    last_timestamp: u32,
    /// Current jitter estimate (microseconds).
    jitter: f64,
}

impl JitterCalculator {
    /// Creates a new jitter calculator.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            last_arrival: None,
            last_timestamp: 0,
            jitter: 0.0,
        }
    }

    /// Updates jitter estimate with a new packet.
    pub fn update(&mut self, arrival: Instant, timestamp: u32) {
        if let Some(last_arrival) = self.last_arrival {
            let arrival_delta = arrival.duration_since(last_arrival).as_micros() as i64;
            let timestamp_delta = timestamp.wrapping_sub(self.last_timestamp) as i64;

            let delta = arrival_delta - timestamp_delta;
            let abs_delta = delta.unsigned_abs() as f64;

            // J = J + (|D| - J) / 16
            self.jitter += (abs_delta - self.jitter) / 16.0;
        }

        self.last_arrival = Some(arrival);
        self.last_timestamp = timestamp;
    }

    /// Returns current jitter estimate in microseconds.
    #[must_use]
    pub const fn jitter(&self) -> u32 {
        self.jitter as u32
    }

    /// Resets the jitter calculator.
    pub fn reset(&mut self) {
        self.last_arrival = None;
        self.last_timestamp = 0;
        self.jitter = 0.0;
    }
}

impl Default for JitterCalculator {
    fn default() -> Self {
        Self::new()
    }
}

/// Loss rate estimator.
#[derive(Debug)]
pub struct LossRateEstimator {
    /// Total packets expected.
    total_expected: u64,
    /// Total packets lost.
    total_lost: u64,
    /// Recent window of samples.
    recent_samples: VecDeque<(u64, u64)>, // (expected, lost)
    /// Window size.
    window_size: usize,
}

impl LossRateEstimator {
    /// Creates a new loss rate estimator.
    #[must_use]
    pub fn new(window_size: usize) -> Self {
        Self {
            total_expected: 0,
            total_lost: 0,
            recent_samples: VecDeque::with_capacity(window_size),
            window_size,
        }
    }

    /// Records packet statistics.
    pub fn record(&mut self, expected: u64, lost: u64) {
        self.total_expected += expected;
        self.total_lost += lost;

        self.recent_samples.push_back((expected, lost));

        if self.recent_samples.len() > self.window_size {
            self.recent_samples.pop_front();
        }
    }

    /// Returns overall loss rate.
    #[must_use]
    pub fn overall_loss_rate(&self) -> f64 {
        if self.total_expected == 0 {
            0.0
        } else {
            self.total_lost as f64 / self.total_expected as f64
        }
    }

    /// Returns recent loss rate (within window).
    #[must_use]
    pub fn recent_loss_rate(&self) -> f64 {
        let (total_exp, total_lost): (u64, u64) = self
            .recent_samples
            .iter()
            .fold((0, 0), |(exp, lost), &(e, l)| (exp + e, lost + l));

        if total_exp == 0 {
            0.0
        } else {
            total_lost as f64 / total_exp as f64
        }
    }

    /// Resets statistics.
    pub fn reset(&mut self) {
        self.total_expected = 0;
        self.total_lost = 0;
        self.recent_samples.clear();
    }
}

/// Connection monitor that aggregates all metrics.
#[derive(Debug)]
pub struct ConnectionMonitor {
    /// Bandwidth estimator.
    bandwidth: BandwidthEstimator,
    /// Jitter calculator.
    jitter: JitterCalculator,
    /// Loss rate estimator.
    loss_rate: LossRateEstimator,
    /// Last update time.
    last_update: Instant,
}

impl ConnectionMonitor {
    /// Creates a new connection monitor.
    #[must_use]
    pub fn new() -> Self {
        Self {
            bandwidth: BandwidthEstimator::new(Duration::from_secs(5)),
            jitter: JitterCalculator::new(),
            loss_rate: LossRateEstimator::new(100),
            last_update: Instant::now(),
        }
    }

    /// Records sent data.
    pub fn record_send(&mut self, bytes: u64) {
        self.bandwidth.record(bytes);
        self.last_update = Instant::now();
    }

    /// Records received packet.
    pub fn record_receive(&mut self, bytes: u64, timestamp: u32) {
        self.bandwidth.record(bytes);
        self.jitter.update(Instant::now(), timestamp);
        self.last_update = Instant::now();
    }

    /// Records packet loss.
    pub fn record_loss(&mut self, expected: u64, lost: u64) {
        self.loss_rate.record(expected, lost);
    }

    /// Returns current quality metrics.
    #[must_use]
    pub fn metrics(&self, rtt: u32, rtt_var: u32, retransmit_count: u64) -> QualityMetrics {
        QualityMetrics {
            rtt,
            rtt_var,
            loss_rate: self.loss_rate.recent_loss_rate(),
            bandwidth: self.bandwidth.estimate(),
            send_buffer_util: 0.0, // Would be set from external stats
            recv_buffer_util: 0.0,
            retransmit_count,
            jitter: self.jitter.jitter(),
        }
    }

    /// Returns bandwidth in megabits per second.
    #[must_use]
    pub fn bandwidth_mbps(&self) -> f64 {
        self.bandwidth.mbps()
    }

    /// Resets all statistics.
    pub fn reset(&mut self) {
        self.bandwidth.reset();
        self.jitter.reset();
        self.loss_rate.reset();
        self.last_update = Instant::now();
    }

    /// Returns time since last update.
    #[must_use]
    pub fn time_since_update(&self) -> Duration {
        self.last_update.elapsed()
    }
}

impl Default for ConnectionMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bandwidth_estimator() {
        let mut est = BandwidthEstimator::new(Duration::from_secs(1));
        est.record(1000);
        assert!(est.estimate() > 0.0);
    }

    #[test]
    fn test_quality_metrics_score() {
        let metrics = QualityMetrics {
            rtt: 50_000,
            loss_rate: 0.0,
            jitter: 5_000,
            ..Default::default()
        };

        let score = metrics.quality_score();
        assert!(score > 90.0);
        assert!(metrics.is_good());
    }

    #[test]
    fn test_quality_metrics_degraded() {
        let metrics = QualityMetrics {
            rtt: 600_000,
            loss_rate: 0.1,
            ..Default::default()
        };

        assert!(metrics.is_degraded());
        assert!(!metrics.is_good());
    }

    #[test]
    fn test_jitter_calculator() {
        let mut calc = JitterCalculator::new();
        let now = Instant::now();

        calc.update(now, 0);
        calc.update(now + Duration::from_millis(20), 20_000);

        assert_eq!(calc.jitter(), 0);
    }

    #[test]
    fn test_loss_rate_estimator() {
        let mut est = LossRateEstimator::new(10);
        est.record(100, 5);
        est.record(100, 3);

        let rate = est.overall_loss_rate();
        assert!((rate - 0.04).abs() < 0.01);
    }

    #[test]
    fn test_connection_monitor() {
        let mut monitor = ConnectionMonitor::new();
        monitor.record_send(1000);
        monitor.record_receive(500, 1000);

        let metrics = monitor.metrics(50_000, 10_000, 0);
        assert_eq!(metrics.rtt, 50_000);
    }
}
