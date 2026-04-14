#![allow(dead_code)]
//! Frame cache warming via predictive pre-rendering.
//!
//! During playback, the editor's playhead moves through the timeline at a
//! measurable velocity.  [`CacheWarmer`] predicts which frames will be needed
//! in the near future and queues them for background pre-rendering, so that
//! by the time the playhead reaches them they are already in the frame cache.
//!
//! The warming strategy adapts to playhead behaviour:
//!
//! - **Forward playback** — warms `lookahead` frames ahead of the playhead.
//! - **Reverse playback** — warms frames behind the playhead.
//! - **Scrubbing** — uses velocity-based prediction with a momentum model to
//!   warm frames in the direction of the scrub.
//! - **Stopped** — warms a small neighbourhood around the current position.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use crate::types::Position;

/// Playback direction detected from velocity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlayDirection {
    /// Playhead is not moving.
    Stopped,
    /// Normal forward playback.
    Forward,
    /// Reverse playback.
    Reverse,
    /// Scrubbing (irregular, potentially fast movement).
    Scrubbing,
}

/// Configuration for the cache warmer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheWarmerConfig {
    /// Number of frames to pre-render ahead of the playhead.
    pub lookahead: usize,
    /// Number of frames to pre-render behind the playhead (for reverse).
    pub lookbehind: usize,
    /// Number of frames to warm around the playhead when stopped.
    pub stopped_radius: usize,
    /// Maximum number of frames in the warming queue at any time.
    pub max_queue_size: usize,
    /// Velocity threshold (frames/sample) below which the playhead is
    /// considered stopped.
    pub stopped_velocity_threshold: f64,
    /// Velocity threshold above which movement is classified as scrubbing.
    pub scrub_velocity_threshold: f64,
    /// Number of velocity samples used for smoothing.
    pub velocity_window: usize,
}

impl Default for CacheWarmerConfig {
    fn default() -> Self {
        Self {
            lookahead: 30,
            lookbehind: 10,
            stopped_radius: 5,
            max_queue_size: 60,
            stopped_velocity_threshold: 0.1,
            scrub_velocity_threshold: 3.0,
            velocity_window: 5,
        }
    }
}

/// A request to pre-render a specific frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WarmRequest {
    /// Frame number to pre-render.
    pub frame: i64,
    /// Priority (lower = higher priority).
    pub priority: u32,
}

impl WarmRequest {
    /// Creates a new warm request.
    #[must_use]
    pub fn new(frame: i64, priority: u32) -> Self {
        Self { frame, priority }
    }
}

/// Velocity estimator using a sliding window of playhead positions.
#[derive(Debug, Clone)]
struct VelocityEstimator {
    /// Recent playhead positions.
    samples: VecDeque<i64>,
    /// Maximum number of samples to keep.
    window_size: usize,
}

impl VelocityEstimator {
    fn new(window_size: usize) -> Self {
        Self {
            samples: VecDeque::with_capacity(window_size + 1),
            window_size: window_size.max(2),
        }
    }

    fn push(&mut self, position: i64) {
        self.samples.push_back(position);
        while self.samples.len() > self.window_size {
            self.samples.pop_front();
        }
    }

    /// Returns the average velocity (frames per sample interval).
    fn velocity(&self) -> f64 {
        if self.samples.len() < 2 {
            return 0.0;
        }
        let n = self.samples.len();
        let first = self.samples[0];
        let last = self.samples[n - 1];
        (last - first) as f64 / (n - 1) as f64
    }

    /// Returns the absolute velocity magnitude.
    fn speed(&self) -> f64 {
        self.velocity().abs()
    }

    fn clear(&mut self) {
        self.samples.clear();
    }
}

/// Predicts and queues frames for pre-rendering based on playhead movement.
#[derive(Debug, Clone)]
pub struct CacheWarmer {
    config: CacheWarmerConfig,
    velocity_estimator: VelocityEstimator,
    /// Current detected direction.
    direction: PlayDirection,
    /// Current warming queue.
    queue: Vec<WarmRequest>,
    /// Total frames that have been requested for warming.
    total_requested: u64,
    /// Frames that were in cache when accessed (hits).
    cache_hits: u64,
}

impl CacheWarmer {
    /// Creates a new cache warmer with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(CacheWarmerConfig::default())
    }

    /// Creates a new cache warmer with custom configuration.
    #[must_use]
    pub fn with_config(config: CacheWarmerConfig) -> Self {
        let window = config.velocity_window;
        Self {
            config,
            velocity_estimator: VelocityEstimator::new(window),
            direction: PlayDirection::Stopped,
            queue: Vec::new(),
            total_requested: 0,
            cache_hits: 0,
        }
    }

    /// Returns the current configuration.
    #[must_use]
    pub fn config(&self) -> &CacheWarmerConfig {
        &self.config
    }

    /// Returns the current detected play direction.
    #[must_use]
    pub fn direction(&self) -> PlayDirection {
        self.direction
    }

    /// Returns the current estimated velocity (frames per tick).
    #[must_use]
    pub fn velocity(&self) -> f64 {
        self.velocity_estimator.velocity()
    }

    /// Returns the current warming queue.
    #[must_use]
    pub fn queue(&self) -> &[WarmRequest] {
        &self.queue
    }

    /// Returns the total frames requested for warming.
    #[must_use]
    pub fn total_requested(&self) -> u64 {
        self.total_requested
    }

    /// Record a cache hit (for statistics).
    pub fn record_hit(&mut self) {
        self.cache_hits += 1;
    }

    /// Returns the cache hit ratio (0.0–1.0).
    #[must_use]
    pub fn hit_ratio(&self) -> f64 {
        if self.total_requested == 0 {
            return 0.0;
        }
        self.cache_hits as f64 / self.total_requested as f64
    }

    /// Update the warmer with a new playhead position.
    ///
    /// This updates velocity estimation, detects the play direction,
    /// and regenerates the warming queue.
    pub fn update(&mut self, playhead: Position) {
        self.velocity_estimator.push(playhead.value());

        let speed = self.velocity_estimator.speed();
        let velocity = self.velocity_estimator.velocity();

        self.direction = if speed < self.config.stopped_velocity_threshold {
            PlayDirection::Stopped
        } else if speed > self.config.scrub_velocity_threshold {
            PlayDirection::Scrubbing
        } else if velocity > 0.0 {
            PlayDirection::Forward
        } else {
            PlayDirection::Reverse
        };

        self.generate_queue(playhead.value());
    }

    /// Reset the warmer (e.g., after a seek).
    pub fn reset(&mut self) {
        self.velocity_estimator.clear();
        self.direction = PlayDirection::Stopped;
        self.queue.clear();
    }

    /// Generate the warming queue based on current direction and position.
    fn generate_queue(&mut self, current_frame: i64) {
        self.queue.clear();

        match self.direction {
            PlayDirection::Forward => {
                for i in 1..=self.config.lookahead {
                    let frame = current_frame + i as i64;
                    #[allow(clippy::cast_possible_truncation)]
                    self.queue.push(WarmRequest::new(frame, i as u32));
                }
            }
            PlayDirection::Reverse => {
                for i in 1..=self.config.lookbehind {
                    let frame = current_frame - i as i64;
                    if frame >= 0 {
                        #[allow(clippy::cast_possible_truncation)]
                        self.queue.push(WarmRequest::new(frame, i as u32));
                    }
                }
            }
            PlayDirection::Scrubbing => {
                // Predict based on velocity direction with reduced range
                let velocity = self.velocity_estimator.velocity();
                let range = self.config.lookahead / 2;
                if velocity > 0.0 {
                    for i in 1..=range {
                        let frame = current_frame + i as i64;
                        #[allow(clippy::cast_possible_truncation)]
                        self.queue.push(WarmRequest::new(frame, i as u32));
                    }
                } else {
                    for i in 1..=range {
                        let frame = current_frame - i as i64;
                        if frame >= 0 {
                            #[allow(clippy::cast_possible_truncation)]
                            self.queue.push(WarmRequest::new(frame, i as u32));
                        }
                    }
                }
            }
            PlayDirection::Stopped => {
                let r = self.config.stopped_radius as i64;
                let mut priority = 0u32;
                for offset in -r..=r {
                    let frame = current_frame + offset;
                    if frame >= 0 && offset != 0 {
                        self.queue.push(WarmRequest::new(frame, priority));
                        priority += 1;
                    }
                }
            }
        }

        // Truncate to max queue size
        self.queue.truncate(self.config.max_queue_size);
        self.total_requested += self.queue.len() as u64;
    }

    /// Drain the next `n` warm requests from the queue (highest priority first).
    pub fn drain(&mut self, n: usize) -> Vec<WarmRequest> {
        self.queue.sort_by_key(|r| r.priority);
        let taken: Vec<_> = self.queue.drain(..n.min(self.queue.len())).collect();
        taken
    }

    /// Peek at the next `n` frames that would be warmed.
    #[must_use]
    pub fn peek(&self, n: usize) -> Vec<i64> {
        let mut sorted = self.queue.clone();
        sorted.sort_by_key(|r| r.priority);
        sorted.iter().take(n).map(|r| r.frame).collect()
    }
}

impl Default for CacheWarmer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_warmer_new() {
        let warmer = CacheWarmer::new();
        assert_eq!(warmer.direction(), PlayDirection::Stopped);
        assert!(warmer.queue().is_empty());
        assert_eq!(warmer.total_requested(), 0);
    }

    #[test]
    fn test_velocity_estimator_empty() {
        let est = VelocityEstimator::new(5);
        assert!((est.velocity() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_velocity_estimator_forward() {
        let mut est = VelocityEstimator::new(5);
        for i in 0..5 {
            est.push(i * 10);
        }
        // velocity = (40 - 0) / 4 = 10.0
        assert!((est.velocity() - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_velocity_estimator_reverse() {
        let mut est = VelocityEstimator::new(5);
        for i in (0..5).rev() {
            est.push(i * 10);
        }
        // velocity = (0 - 40) / 4 = -10.0
        assert!((est.velocity() - (-10.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_forward_detection() {
        let mut warmer = CacheWarmer::with_config(CacheWarmerConfig {
            lookahead: 10,
            velocity_window: 3,
            stopped_velocity_threshold: 0.1,
            scrub_velocity_threshold: 5.0,
            ..Default::default()
        });
        // Simulate forward playback
        for i in 0..5 {
            warmer.update(Position::new(i));
        }
        assert_eq!(warmer.direction(), PlayDirection::Forward);
        assert!(!warmer.queue().is_empty());
    }

    #[test]
    fn test_reverse_detection() {
        let mut warmer = CacheWarmer::with_config(CacheWarmerConfig {
            lookbehind: 10,
            velocity_window: 3,
            stopped_velocity_threshold: 0.1,
            scrub_velocity_threshold: 5.0,
            ..Default::default()
        });
        // Simulate reverse playback
        for i in (0..5).rev() {
            warmer.update(Position::new(100 - i));
        }
        // This creates positions 100, 97, 98, 99, 100 — actually let me fix
        // Actually (0..5).rev() = 4,3,2,1,0 → 100-4=96, 100-3=97, 100-2=98, 100-1=99, 100-0=100
        // That's forward! Let me fix:
        let mut warmer2 = CacheWarmer::with_config(CacheWarmerConfig {
            lookbehind: 10,
            velocity_window: 3,
            stopped_velocity_threshold: 0.1,
            scrub_velocity_threshold: 5.0,
            ..Default::default()
        });
        for i in 0..5 {
            warmer2.update(Position::new(100 - i));
        }
        assert_eq!(warmer2.direction(), PlayDirection::Reverse);
    }

    #[test]
    fn test_stopped_detection() {
        let mut warmer = CacheWarmer::with_config(CacheWarmerConfig {
            velocity_window: 5,
            stopped_velocity_threshold: 0.5,
            ..Default::default()
        });
        // Same position repeated
        for _ in 0..5 {
            warmer.update(Position::new(50));
        }
        assert_eq!(warmer.direction(), PlayDirection::Stopped);
    }

    #[test]
    fn test_scrubbing_detection() {
        let mut warmer = CacheWarmer::with_config(CacheWarmerConfig {
            velocity_window: 3,
            stopped_velocity_threshold: 0.1,
            scrub_velocity_threshold: 3.0,
            ..Default::default()
        });
        // Fast movement (10 frames per tick)
        for i in 0..5 {
            warmer.update(Position::new(i * 10));
        }
        assert_eq!(warmer.direction(), PlayDirection::Scrubbing);
    }

    #[test]
    fn test_forward_queue_content() {
        let mut warmer = CacheWarmer::with_config(CacheWarmerConfig {
            lookahead: 5,
            velocity_window: 3,
            stopped_velocity_threshold: 0.1,
            scrub_velocity_threshold: 5.0,
            ..Default::default()
        });
        for i in 0..4 {
            warmer.update(Position::new(i));
        }
        assert_eq!(warmer.direction(), PlayDirection::Forward);
        // Current frame is 3; queue should contain frames 4..=8
        let frames: Vec<i64> = warmer.queue().iter().map(|r| r.frame).collect();
        assert_eq!(frames, vec![4, 5, 6, 7, 8]);
    }

    #[test]
    fn test_stopped_queue_neighbourhood() {
        let mut warmer = CacheWarmer::with_config(CacheWarmerConfig {
            stopped_radius: 2,
            velocity_window: 3,
            stopped_velocity_threshold: 0.5,
            ..Default::default()
        });
        for _ in 0..3 {
            warmer.update(Position::new(50));
        }
        assert_eq!(warmer.direction(), PlayDirection::Stopped);
        let frames: Vec<i64> = warmer.queue().iter().map(|r| r.frame).collect();
        // Should contain frames around 50 (excluding 50 itself)
        assert!(frames.contains(&48));
        assert!(frames.contains(&49));
        assert!(frames.contains(&51));
        assert!(frames.contains(&52));
        assert!(!frames.contains(&50)); // current frame excluded
    }

    #[test]
    fn test_drain() {
        let mut warmer = CacheWarmer::with_config(CacheWarmerConfig {
            lookahead: 10,
            velocity_window: 3,
            stopped_velocity_threshold: 0.1,
            scrub_velocity_threshold: 5.0,
            ..Default::default()
        });
        for i in 0..4 {
            warmer.update(Position::new(i));
        }
        let drained = warmer.drain(3);
        assert_eq!(drained.len(), 3);
        // Queue should have remaining items
        assert_eq!(warmer.queue().len(), 7); // 10 - 3
    }

    #[test]
    fn test_peek() {
        let mut warmer = CacheWarmer::with_config(CacheWarmerConfig {
            lookahead: 5,
            velocity_window: 3,
            stopped_velocity_threshold: 0.1,
            scrub_velocity_threshold: 5.0,
            ..Default::default()
        });
        for i in 0..4 {
            warmer.update(Position::new(i));
        }
        let peeked = warmer.peek(3);
        assert_eq!(peeked.len(), 3);
        // Queue should be unchanged
        assert_eq!(warmer.queue().len(), 5);
    }

    #[test]
    fn test_reset() {
        let mut warmer = CacheWarmer::new();
        for i in 0..10 {
            warmer.update(Position::new(i));
        }
        warmer.reset();
        assert_eq!(warmer.direction(), PlayDirection::Stopped);
        assert!(warmer.queue().is_empty());
    }

    #[test]
    fn test_hit_ratio() {
        let mut warmer = CacheWarmer::new();
        // Manually set stats
        warmer.total_requested = 100;
        warmer.cache_hits = 75;
        assert!((warmer.hit_ratio() - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn test_hit_ratio_zero() {
        let warmer = CacheWarmer::new();
        assert!((warmer.hit_ratio() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_record_hit() {
        let mut warmer = CacheWarmer::new();
        warmer.record_hit();
        warmer.record_hit();
        assert_eq!(warmer.cache_hits, 2);
    }

    #[test]
    fn test_max_queue_size() {
        let mut warmer = CacheWarmer::with_config(CacheWarmerConfig {
            lookahead: 100,
            max_queue_size: 10,
            velocity_window: 3,
            stopped_velocity_threshold: 0.1,
            scrub_velocity_threshold: 5.0,
            ..Default::default()
        });
        for i in 0..4 {
            warmer.update(Position::new(i));
        }
        assert!(warmer.queue().len() <= 10);
    }

    #[test]
    fn test_reverse_no_negative_frames() {
        let mut warmer = CacheWarmer::with_config(CacheWarmerConfig {
            lookbehind: 20,
            velocity_window: 3,
            stopped_velocity_threshold: 0.1,
            scrub_velocity_threshold: 5.0,
            ..Default::default()
        });
        // Move backwards near frame 0
        for i in (0..5).rev() {
            warmer.update(Position::new(5 - i));
        }
        // Check: no negative frames should be generated
        // Actually this goes 5,4,3,2,1 → wait that's wrong
        // (0..5).rev() = 4,3,2,1,0 → 5-4=1, 5-3=2, 5-2=3, 5-1=4, 5-0=5 → forward
        // Let me fix:
        let mut warmer2 = CacheWarmer::with_config(CacheWarmerConfig {
            lookbehind: 20,
            velocity_window: 3,
            stopped_velocity_threshold: 0.1,
            scrub_velocity_threshold: 5.0,
            ..Default::default()
        });
        for i in 0..5 {
            warmer2.update(Position::new(5 - i));
        }
        // 5,4,3,2,1 → reverse
        for req in warmer2.queue() {
            assert!(req.frame >= 0, "frame should not be negative");
        }
    }

    #[test]
    fn test_warm_request_new() {
        let req = WarmRequest::new(42, 1);
        assert_eq!(req.frame, 42);
        assert_eq!(req.priority, 1);
    }

    #[test]
    fn test_config_default() {
        let config = CacheWarmerConfig::default();
        assert_eq!(config.lookahead, 30);
        assert_eq!(config.lookbehind, 10);
        assert_eq!(config.stopped_radius, 5);
        assert_eq!(config.max_queue_size, 60);
    }
}
