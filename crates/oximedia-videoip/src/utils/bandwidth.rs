//! Bandwidth adaptation and estimation utilities.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Bandwidth estimator using sliding window.
pub struct BandwidthEstimator {
    /// Samples of (timestamp, bytes).
    samples: VecDeque<(Instant, usize)>,
    /// Window size for estimation.
    window_size: Duration,
    /// Maximum samples to keep.
    max_samples: usize,
}

impl BandwidthEstimator {
    /// Creates a new bandwidth estimator.
    #[must_use]
    pub fn new(window_size: Duration) -> Self {
        Self {
            samples: VecDeque::new(),
            window_size,
            max_samples: 1000,
        }
    }

    /// Records a data transmission.
    pub fn record(&mut self, bytes: usize) {
        let now = Instant::now();
        self.samples.push_back((now, bytes));

        // Remove old samples outside the window
        let cutoff = now.checked_sub(self.window_size).unwrap_or(now);
        while let Some(&(timestamp, _)) = self.samples.front() {
            if timestamp < cutoff {
                self.samples.pop_front();
            } else {
                break;
            }
        }

        // Limit max samples
        while self.samples.len() > self.max_samples {
            self.samples.pop_front();
        }
    }

    /// Returns the current bandwidth estimate in bits per second.
    #[must_use]
    pub fn estimate(&self) -> u64 {
        if self.samples.len() < 2 {
            return 0;
        }

        let total_bytes: usize = self.samples.iter().map(|(_, bytes)| bytes).sum();
        let Some(first_entry) = self.samples.front() else {
            return 0;
        };
        let Some(last_entry) = self.samples.back() else {
            return 0;
        };
        let first_time = first_entry.0;
        let last_time = last_entry.0;

        let duration = last_time.duration_since(first_time).as_secs_f64();
        if duration < 0.001 {
            return 0;
        }

        ((total_bytes as f64 * 8.0) / duration) as u64
    }

    /// Returns the average bandwidth over the window.
    #[must_use]
    pub fn average(&self) -> u64 {
        if self.samples.is_empty() {
            return 0;
        }

        let total_bytes: usize = self.samples.iter().map(|(_, bytes)| bytes).sum();
        let bits = total_bytes * 8;
        let samples = self.samples.len() as u64;

        (bits as u64) / samples.max(1)
    }

    /// Returns the peak bandwidth in the window.
    #[must_use]
    pub fn peak(&self) -> u64 {
        self.samples
            .iter()
            .map(|(_, bytes)| (*bytes * 8) as u64)
            .max()
            .unwrap_or(0)
    }

    /// Clears all samples.
    pub fn clear(&mut self) {
        self.samples.clear();
    }
}

/// Adaptive bitrate controller.
pub struct AdaptiveBitrateController {
    /// Current target bitrate.
    target_bitrate: u64,
    /// Minimum allowed bitrate.
    min_bitrate: u64,
    /// Maximum allowed bitrate.
    max_bitrate: u64,
    /// Bandwidth estimator.
    bandwidth_estimator: BandwidthEstimator,
    /// Buffer occupancy threshold for increasing bitrate (0.0-1.0).
    increase_threshold: f64,
    /// Buffer occupancy threshold for decreasing bitrate (0.0-1.0).
    decrease_threshold: f64,
    /// Bitrate adjustment step (percentage).
    adjustment_step: f64,
}

impl AdaptiveBitrateController {
    /// Creates a new adaptive bitrate controller.
    #[must_use]
    pub fn new(min_bitrate: u64, max_bitrate: u64, initial_bitrate: u64) -> Self {
        Self {
            target_bitrate: initial_bitrate.clamp(min_bitrate, max_bitrate),
            min_bitrate,
            max_bitrate,
            bandwidth_estimator: BandwidthEstimator::new(Duration::from_secs(5)),
            increase_threshold: 0.3,
            decrease_threshold: 0.7,
            adjustment_step: 0.1,
        }
    }

    /// Records a transmission for bandwidth estimation.
    pub fn record_transmission(&mut self, bytes: usize) {
        self.bandwidth_estimator.record(bytes);
    }

    /// Adjusts the target bitrate based on current conditions.
    pub fn adjust(&mut self, buffer_occupancy: f64, packet_loss_rate: f64) -> BitrateAdjustment {
        let available_bandwidth = self.bandwidth_estimator.estimate();

        if packet_loss_rate > 0.05 {
            // High packet loss - decrease bitrate aggressively
            self.decrease_bitrate(0.2);
            BitrateAdjustment::Decrease
        } else if buffer_occupancy > self.decrease_threshold {
            // Buffer filling up - decrease bitrate
            self.decrease_bitrate(self.adjustment_step);
            BitrateAdjustment::Decrease
        } else if buffer_occupancy < self.increase_threshold
            && available_bandwidth > self.target_bitrate
        {
            // Buffer draining and bandwidth available - increase bitrate
            self.increase_bitrate(self.adjustment_step);
            BitrateAdjustment::Increase
        } else {
            BitrateAdjustment::NoChange
        }
    }

    /// Increases the target bitrate by a percentage.
    fn increase_bitrate(&mut self, percentage: f64) {
        let increase = (self.target_bitrate as f64 * percentage) as u64;
        self.target_bitrate = (self.target_bitrate + increase).min(self.max_bitrate);
    }

    /// Decreases the target bitrate by a percentage.
    fn decrease_bitrate(&mut self, percentage: f64) {
        let decrease = (self.target_bitrate as f64 * percentage) as u64;
        self.target_bitrate = self
            .target_bitrate
            .saturating_sub(decrease)
            .max(self.min_bitrate);
    }

    /// Returns the current target bitrate.
    #[must_use]
    pub const fn target_bitrate(&self) -> u64 {
        self.target_bitrate
    }

    /// Returns the estimated available bandwidth.
    #[must_use]
    pub fn available_bandwidth(&self) -> u64 {
        self.bandwidth_estimator.estimate()
    }
}

/// Bitrate adjustment result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitrateAdjustment {
    /// Bitrate increased.
    Increase,
    /// Bitrate decreased.
    Decrease,
    /// No change needed.
    NoChange,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_bandwidth_estimator() {
        let mut estimator = BandwidthEstimator::new(Duration::from_secs(1));

        estimator.record(1000);
        thread::sleep(Duration::from_millis(100));
        estimator.record(1000);

        let estimate = estimator.estimate();
        assert!(estimate > 0);
    }

    #[test]
    fn test_bandwidth_estimator_clear() {
        let mut estimator = BandwidthEstimator::new(Duration::from_secs(1));
        estimator.record(1000);
        estimator.clear();

        assert_eq!(estimator.estimate(), 0);
    }

    #[test]
    fn test_adaptive_bitrate_controller() {
        let mut controller = AdaptiveBitrateController::new(
            1_000_000,  // 1 Mbps min
            10_000_000, // 10 Mbps max
            5_000_000,  // 5 Mbps initial
        );

        assert_eq!(controller.target_bitrate(), 5_000_000);

        // High buffer occupancy should decrease bitrate
        let adjustment = controller.adjust(0.8, 0.0);
        assert_eq!(adjustment, BitrateAdjustment::Decrease);
        assert!(controller.target_bitrate() < 5_000_000);
    }

    #[test]
    fn test_bitrate_bounds() {
        let mut controller = AdaptiveBitrateController::new(1_000_000, 10_000_000, 1_000_000);

        // Try to decrease below minimum
        controller.adjust(1.0, 0.5);
        assert_eq!(controller.target_bitrate(), 1_000_000);

        // Set to maximum
        controller.target_bitrate = 10_000_000;

        // Try to increase above maximum
        controller.record_transmission(10_000_000);
        controller.adjust(0.1, 0.0);
        assert_eq!(controller.target_bitrate(), 10_000_000);
    }

    #[test]
    fn test_packet_loss_response() {
        let mut controller = AdaptiveBitrateController::new(1_000_000, 10_000_000, 5_000_000);

        let initial = controller.target_bitrate();

        // High packet loss should decrease bitrate significantly
        controller.adjust(0.5, 0.1);

        assert!(controller.target_bitrate() < initial);
    }
}
