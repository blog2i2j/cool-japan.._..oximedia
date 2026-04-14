//! Bandwidth-aware segment prefetch scheduler.
//!
//! Determines how many segments ahead the player should prefetch based on the
//! current downstream bandwidth, segment bitrate, target buffer depth, and
//! configurable safety margins.
//!
//! The scheduler tracks a short history of bandwidth measurements (EWMA-smoothed)
//! and recomputes the recommended prefetch depth on every update.  It also
//! maintains a queue of segment keys scheduled for prefetch, advancing the
//! read head as segments complete.
//!
//! # Design rationale
//!
//! A larger prefetch depth improves resilience to bandwidth drops at the cost
//! of increased memory and start-up latency.  The scheduler balances these
//! forces by:
//!
//! 1. Estimating the time to download one segment (`segment_bits / bandwidth`).
//! 2. Dividing the target buffer goal (seconds) by the segment duration to get
//!    the ideal number of buffered segments.
//! 3. Capping the result to `[min_depth, max_depth]`.
//! 4. Applying a safety multiplier when bandwidth is volatile (high variance
//!    between samples).
//!
//! # Example
//!
//! ```
//! use oximedia_stream::prefetch_scheduler::{PrefetchScheduler, PrefetchConfig};
//!
//! let config = PrefetchConfig {
//!     segment_duration_secs: 2.0,
//!     segment_bitrate_kbps: 3_000,
//!     target_buffer_secs: 12.0,
//!     min_depth: 1,
//!     max_depth: 8,
//!     safety_multiplier: 1.2,
//!     ewma_alpha: 0.25,
//! };
//! let mut scheduler = PrefetchScheduler::new(config);
//! scheduler.record_bandwidth(10_000); // 10 Mbps
//! let depth = scheduler.recommended_depth();
//! assert!(depth >= 1 && depth <= 8);
//! ```

use std::collections::VecDeque;

// ─── PrefetchConfig ───────────────────────────────────────────────────────────

/// Tuning parameters for the prefetch scheduler.
#[derive(Debug, Clone)]
pub struct PrefetchConfig {
    /// Nominal duration of one segment in seconds.
    pub segment_duration_secs: f64,
    /// Nominal bitrate of the active rendition in kbps.
    pub segment_bitrate_kbps: u32,
    /// Desired total buffer depth in seconds.
    pub target_buffer_secs: f64,
    /// Minimum prefetch depth (always prefetch at least this many segments).
    pub min_depth: usize,
    /// Maximum prefetch depth (never schedule more than this).
    pub max_depth: usize,
    /// Multiplier applied to the raw depth estimate to add a safety margin.
    /// A value of `1.0` means no safety margin.
    pub safety_multiplier: f64,
    /// EWMA smoothing factor for bandwidth measurements (0.0–1.0).
    /// Smaller values give slower, more stable estimates.
    pub ewma_alpha: f64,
}

impl Default for PrefetchConfig {
    fn default() -> Self {
        Self {
            segment_duration_secs: 2.0,
            segment_bitrate_kbps: 3_000,
            target_buffer_secs: 10.0,
            min_depth: 1,
            max_depth: 6,
            safety_multiplier: 1.1,
            ewma_alpha: 0.25,
        }
    }
}

// ─── BandwidthSample ─────────────────────────────────────────────────────────

/// One bandwidth observation.
#[derive(Debug, Clone, Copy)]
pub struct BandwidthSample {
    /// Measured downstream bandwidth in kbps.
    pub bandwidth_kbps: u32,
    /// Monotonic sequence number (assigned by the scheduler).
    pub seq: u64,
}

// ─── PrefetchScheduler ────────────────────────────────────────────────────────

/// Computes segment prefetch depth from live bandwidth measurements.
#[derive(Debug)]
pub struct PrefetchScheduler {
    /// Configuration parameters.
    config: PrefetchConfig,
    /// EWMA bandwidth estimate in kbps (stored as f64 for precision).
    ewma_bw_kbps: f64,
    /// Whether the EWMA has been seeded with at least one sample.
    initialized: bool,
    /// Sliding window of raw bandwidth samples (up to 32 kept).
    samples: VecDeque<BandwidthSample>,
    /// Monotonic sequence counter for samples.
    seq: u64,
    /// Queue of segment keys scheduled for prefetch.
    prefetch_queue: VecDeque<String>,
    /// Total segments successfully prefetched.
    completed_count: u64,
    /// Last computed prefetch depth.
    last_depth: usize,
}

impl PrefetchScheduler {
    /// Maximum number of raw samples retained in the sliding window.
    const MAX_SAMPLES: usize = 32;

    /// Create a new scheduler with the provided configuration.
    pub fn new(config: PrefetchConfig) -> Self {
        // Seed the EWMA with a conservative estimate (segment bitrate × 1.5).
        let seed = config.segment_bitrate_kbps as f64 * 1.5;
        let min_depth = config.min_depth.max(1);
        Self {
            config: PrefetchConfig {
                min_depth,
                max_depth: config.max_depth.max(min_depth),
                ewma_alpha: config.ewma_alpha.clamp(0.01, 1.0),
                safety_multiplier: config.safety_multiplier.max(1.0),
                ..config
            },
            ewma_bw_kbps: seed,
            initialized: false,
            samples: VecDeque::with_capacity(Self::MAX_SAMPLES),
            seq: 0,
            prefetch_queue: VecDeque::new(),
            completed_count: 0,
            last_depth: min_depth,
        }
    }

    /// Record a new bandwidth measurement and recompute the prefetch depth.
    ///
    /// `bandwidth_kbps` must be greater than zero; values of zero are ignored.
    pub fn record_bandwidth(&mut self, bandwidth_kbps: u32) {
        if bandwidth_kbps == 0 {
            return;
        }
        let alpha = self.config.ewma_alpha;
        self.ewma_bw_kbps = if self.initialized {
            alpha * bandwidth_kbps as f64 + (1.0 - alpha) * self.ewma_bw_kbps
        } else {
            bandwidth_kbps as f64
        };
        self.initialized = true;

        self.seq += 1;
        if self.samples.len() >= Self::MAX_SAMPLES {
            self.samples.pop_front();
        }
        self.samples.push_back(BandwidthSample {
            bandwidth_kbps,
            seq: self.seq,
        });

        self.last_depth = self.compute_depth();
    }

    /// Return the currently recommended prefetch depth in number of segments.
    ///
    /// Returns `config.min_depth` until at least one bandwidth sample has been
    /// recorded.
    pub fn recommended_depth(&self) -> usize {
        self.last_depth
    }

    /// Schedule a segment key for prefetch.
    ///
    /// Returns `false` if the key is already in the queue.
    pub fn schedule(&mut self, key: String) -> bool {
        if self.prefetch_queue.contains(&key) {
            return false;
        }
        self.prefetch_queue.push_back(key);
        true
    }

    /// Mark the head-of-queue segment as completed and remove it.
    ///
    /// Returns the segment key if one was pending, or `None` if the queue was
    /// empty.
    pub fn complete_head(&mut self) -> Option<String> {
        if let Some(key) = self.prefetch_queue.pop_front() {
            self.completed_count += 1;
            Some(key)
        } else {
            None
        }
    }

    /// Number of segments currently waiting in the prefetch queue.
    pub fn queue_depth(&self) -> usize {
        self.prefetch_queue.len()
    }

    /// Whether the prefetch queue has capacity for more segments.
    ///
    /// Returns `true` when `queue_depth() < recommended_depth()`.
    pub fn has_capacity(&self) -> bool {
        self.prefetch_queue.len() < self.last_depth
    }

    /// Total number of segments that have been completed (popped via
    /// [`Self::complete_head`]).
    pub fn completed_count(&self) -> u64 {
        self.completed_count
    }

    /// Current EWMA bandwidth estimate in kbps.
    pub fn ewma_bandwidth_kbps(&self) -> f64 {
        self.ewma_bw_kbps
    }

    /// Raw bandwidth sample history (up to 32 entries, oldest first).
    pub fn samples(&self) -> &VecDeque<BandwidthSample> {
        &self.samples
    }

    /// Return the standard deviation of the raw bandwidth samples (kbps).
    ///
    /// Returns `0.0` if fewer than 2 samples are available.
    pub fn bandwidth_stddev_kbps(&self) -> f64 {
        if self.samples.len() < 2 {
            return 0.0;
        }
        let n = self.samples.len() as f64;
        let mean = self
            .samples
            .iter()
            .map(|s| s.bandwidth_kbps as f64)
            .sum::<f64>()
            / n;
        let variance = self
            .samples
            .iter()
            .map(|s| {
                let d = s.bandwidth_kbps as f64 - mean;
                d * d
            })
            .sum::<f64>()
            / n;
        variance.sqrt()
    }

    /// Update only the segment bitrate (e.g. after an ABR quality switch).
    pub fn update_segment_bitrate(&mut self, bitrate_kbps: u32) {
        self.config.segment_bitrate_kbps = bitrate_kbps;
        self.last_depth = self.compute_depth();
    }

    /// Update the target buffer depth in seconds.
    pub fn update_target_buffer(&mut self, target_secs: f64) {
        self.config.target_buffer_secs = target_secs.max(self.config.segment_duration_secs);
        self.last_depth = self.compute_depth();
    }

    /// Return a copy of the current configuration.
    pub fn config(&self) -> &PrefetchConfig {
        &self.config
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Compute prefetch depth from the current EWMA bandwidth and config.
    fn compute_depth(&self) -> usize {
        let seg_bits = self.config.segment_bitrate_kbps as f64 * self.config.segment_duration_secs;
        let bw = self.ewma_bw_kbps.max(1.0);

        // Time to download one segment (seconds).
        let download_time_secs = seg_bits / bw;

        // How many segments can we prefetch in the target buffer time?
        let raw_depth = if download_time_secs <= 0.0 {
            self.config.max_depth as f64
        } else {
            self.config.target_buffer_secs / download_time_secs
        };

        // Apply safety multiplier (add margin for bandwidth volatility).
        let adjusted = raw_depth * self.config.safety_multiplier;

        // Convert to integer and clamp.
        let depth = (adjusted.ceil() as usize)
            .max(self.config.min_depth)
            .min(self.config.max_depth);

        depth
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_scheduler() -> PrefetchScheduler {
        PrefetchScheduler::new(PrefetchConfig::default())
    }

    #[test]
    fn test_initial_depth_is_min_depth() {
        let sched = default_scheduler();
        // Before any sample, last_depth is set to min_depth during construction.
        assert_eq!(sched.recommended_depth(), sched.config().min_depth);
    }

    #[test]
    fn test_depth_increases_with_high_bandwidth() {
        let mut sched = default_scheduler();
        // Very high bandwidth → can buffer many segments ahead.
        sched.record_bandwidth(100_000); // 100 Mbps
        let depth = sched.recommended_depth();
        assert!(
            depth > sched.config().min_depth,
            "high bandwidth should increase prefetch depth; got {depth}"
        );
    }

    #[test]
    fn test_depth_at_min_on_low_bandwidth() {
        // Bandwidth barely above segment bitrate: download one segment takes
        // almost the full segment duration, so buffering 2+ segments ahead is
        // not feasible.  Use a short target buffer so the scheduler picks low
        // depths even with the safety multiplier.
        let config = PrefetchConfig {
            segment_duration_secs: 2.0,
            segment_bitrate_kbps: 5_000,
            target_buffer_secs: 4.0, // just 2 segment-lengths of buffer target
            min_depth: 1,
            max_depth: 8,
            safety_multiplier: 1.0,
            ewma_alpha: 1.0, // instant convergence for determinism
        };
        let mut sched = PrefetchScheduler::new(config);
        // bandwidth ≈ segment_bitrate → download_time ≈ segment_duration
        // raw_depth = target_buffer / download_time = 4.0 / 2.0 = 2
        sched.record_bandwidth(5_000);
        let depth = sched.recommended_depth();
        assert!(
            depth <= 3,
            "low bandwidth short-buffer config should keep depth ≤ 3; got {depth}"
        );
    }

    #[test]
    fn test_depth_clamped_to_max() {
        let config = PrefetchConfig {
            segment_bitrate_kbps: 100,
            target_buffer_secs: 120.0,
            min_depth: 1,
            max_depth: 5,
            safety_multiplier: 1.0,
            ewma_alpha: 1.0,
            ..PrefetchConfig::default()
        };
        let mut sched = PrefetchScheduler::new(config);
        sched.record_bandwidth(1_000_000); // Extreme bandwidth
        assert_eq!(
            sched.recommended_depth(),
            5,
            "depth must never exceed max_depth"
        );
    }

    #[test]
    fn test_depth_at_least_min_depth() {
        let mut sched = default_scheduler();
        sched.record_bandwidth(1); // Extremely low
        assert!(
            sched.recommended_depth() >= sched.config().min_depth,
            "depth must never go below min_depth"
        );
    }

    #[test]
    fn test_ewma_converges_to_sample() {
        let mut sched = PrefetchScheduler::new(PrefetchConfig {
            ewma_alpha: 1.0, // instant convergence
            ..PrefetchConfig::default()
        });
        sched.record_bandwidth(8_000);
        assert!(
            (sched.ewma_bandwidth_kbps() - 8_000.0).abs() < 1.0,
            "with alpha=1.0, EWMA should equal the latest sample"
        );
    }

    #[test]
    fn test_ewma_smoothing() {
        let mut sched = PrefetchScheduler::new(PrefetchConfig {
            ewma_alpha: 0.5,
            ..PrefetchConfig::default()
        });
        sched.record_bandwidth(10_000);
        let first = sched.ewma_bandwidth_kbps();
        sched.record_bandwidth(0); // zero is ignored
        assert!(
            (sched.ewma_bandwidth_kbps() - first).abs() < 1.0,
            "zero-bandwidth sample must be ignored"
        );
    }

    #[test]
    fn test_schedule_and_complete_head() {
        let mut sched = default_scheduler();
        assert!(sched.schedule("seg-001".to_string()));
        assert!(sched.schedule("seg-002".to_string()));
        assert_eq!(sched.queue_depth(), 2);
        let completed = sched.complete_head();
        assert_eq!(completed.as_deref(), Some("seg-001"));
        assert_eq!(sched.queue_depth(), 1);
        assert_eq!(sched.completed_count(), 1);
    }

    #[test]
    fn test_schedule_duplicate_rejected() {
        let mut sched = default_scheduler();
        assert!(sched.schedule("seg-001".to_string()));
        assert!(
            !sched.schedule("seg-001".to_string()),
            "duplicate key must be rejected"
        );
        assert_eq!(sched.queue_depth(), 1);
    }

    #[test]
    fn test_complete_head_on_empty_queue() {
        let mut sched = default_scheduler();
        assert!(sched.complete_head().is_none());
        assert_eq!(sched.completed_count(), 0);
    }

    #[test]
    fn test_has_capacity_respects_depth() {
        let mut sched = PrefetchScheduler::new(PrefetchConfig {
            min_depth: 2,
            max_depth: 2,
            ..PrefetchConfig::default()
        });
        sched.record_bandwidth(10_000);
        assert!(sched.has_capacity());
        sched.schedule("seg-001".to_string());
        sched.schedule("seg-002".to_string());
        assert!(!sched.has_capacity(), "queue full at depth=2");
    }

    #[test]
    fn test_bandwidth_stddev_needs_two_samples() {
        let mut sched = default_scheduler();
        assert_eq!(sched.bandwidth_stddev_kbps(), 0.0);
        sched.record_bandwidth(1000);
        assert_eq!(sched.bandwidth_stddev_kbps(), 0.0);
        sched.record_bandwidth(3000);
        assert!(
            sched.bandwidth_stddev_kbps() > 0.0,
            "std dev should be nonzero with two different samples"
        );
    }

    #[test]
    fn test_update_segment_bitrate_recomputes_depth() {
        let mut sched = default_scheduler();
        sched.record_bandwidth(10_000);
        let depth_before = sched.recommended_depth();
        // Halve the bitrate → more segments should fit in the buffer.
        sched.update_segment_bitrate(sched.config().segment_bitrate_kbps / 2);
        let depth_after = sched.recommended_depth();
        assert!(
            depth_after >= depth_before,
            "lower bitrate should maintain or increase depth"
        );
    }

    #[test]
    fn test_update_target_buffer_affects_depth() {
        let mut sched = default_scheduler();
        sched.record_bandwidth(10_000);
        let depth_small = {
            sched.update_target_buffer(4.0);
            sched.recommended_depth()
        };
        let depth_large = {
            sched.update_target_buffer(30.0);
            sched.recommended_depth()
        };
        assert!(
            depth_large >= depth_small,
            "larger target buffer should yield equal or greater depth"
        );
    }

    #[test]
    fn test_samples_window_capped_at_32() {
        let mut sched = default_scheduler();
        for i in 0..40u32 {
            sched.record_bandwidth(1000 + i);
        }
        assert_eq!(
            sched.samples().len(),
            32,
            "sample window must not exceed 32"
        );
    }
}
