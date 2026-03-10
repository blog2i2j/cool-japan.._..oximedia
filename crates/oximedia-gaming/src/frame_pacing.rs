#![allow(dead_code)]
//! Frame pacing and timing controller for smooth game streaming.
//!
//! Ensures frames are presented at a stable cadence, tracks jitter,
//! and provides adaptive pacing when the source frame-rate fluctuates.

use std::collections::VecDeque;
use std::time::Duration;

/// Target frame cadence descriptor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameCadence {
    /// Target frames per second.
    pub fps: f64,
    /// Computed ideal interval between frames.
    pub interval: Duration,
}

impl FrameCadence {
    /// Build a cadence from a target FPS value.
    ///
    /// # Panics
    ///
    /// Panics if `fps` is zero or negative.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    pub fn from_fps(fps: f64) -> Self {
        assert!(fps > 0.0, "FPS must be positive");
        let nanos = (1_000_000_000.0 / fps) as u64;
        Self {
            fps,
            interval: Duration::from_nanos(nanos),
        }
    }

    /// Return the ideal interval in seconds.
    #[must_use]
    pub fn interval_secs(&self) -> f64 {
        self.interval.as_secs_f64()
    }
}

/// Statistics produced by the pacing controller.
#[derive(Debug, Clone)]
pub struct PacingStats {
    /// Average frame interval over the measurement window.
    pub avg_interval: Duration,
    /// Maximum frame interval observed.
    pub max_interval: Duration,
    /// Minimum frame interval observed.
    pub min_interval: Duration,
    /// Standard deviation of frame intervals (in seconds).
    pub jitter_std: f64,
    /// Number of frames that arrived late (exceeded target + tolerance).
    pub late_frames: u64,
    /// Number of frames considered on-time.
    pub on_time_frames: u64,
}

/// A frame timing record.
#[derive(Debug, Clone, Copy)]
struct FrameTick {
    /// Presentation timestamp in nanoseconds since stream start.
    pts_ns: u64,
}

/// Frame pacing controller.
///
/// Accumulates frame presentation timestamps and computes pacing quality
/// metrics. The controller does **not** block or sleep -- it is purely
/// analytical, so the caller can decide what to do with late / early frames.
#[derive(Debug)]
pub struct FramePacer {
    cadence: FrameCadence,
    /// Tolerance before a frame is counted as late.
    tolerance: Duration,
    /// Sliding window of recent frame ticks.
    history: VecDeque<FrameTick>,
    /// Maximum number of ticks to retain.
    window_size: usize,
    late_frames: u64,
    on_time_frames: u64,
}

impl FramePacer {
    /// Create a new pacer targeting the given cadence.
    #[must_use]
    pub fn new(cadence: FrameCadence, tolerance: Duration, window_size: usize) -> Self {
        Self {
            cadence,
            tolerance,
            history: VecDeque::with_capacity(window_size),
            window_size,
            late_frames: 0,
            on_time_frames: 0,
        }
    }

    /// Record a frame presentation at the given nanosecond timestamp.
    pub fn record_frame(&mut self, pts_ns: u64) {
        if let Some(last) = self.history.back() {
            let delta_ns = pts_ns.saturating_sub(last.pts_ns);
            let target_ns = self.cadence.interval.as_nanos() as u64;
            let tolerance_ns = self.tolerance.as_nanos() as u64;
            if delta_ns > target_ns + tolerance_ns {
                self.late_frames += 1;
            } else {
                self.on_time_frames += 1;
            }
        }
        if self.history.len() >= self.window_size {
            self.history.pop_front();
        }
        self.history.push_back(FrameTick { pts_ns });
    }

    /// Compute pacing statistics over the current window.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn stats(&self) -> PacingStats {
        let intervals = self.intervals_ns();
        if intervals.is_empty() {
            return PacingStats {
                avg_interval: Duration::ZERO,
                max_interval: Duration::ZERO,
                min_interval: Duration::ZERO,
                jitter_std: 0.0,
                late_frames: self.late_frames,
                on_time_frames: self.on_time_frames,
            };
        }
        let sum: u64 = intervals.iter().sum();
        let avg = sum / intervals.len() as u64;
        let max = *intervals.iter().max().unwrap_or(&0);
        let min = *intervals.iter().min().unwrap_or(&0);

        let avg_f = avg as f64;
        let variance = intervals
            .iter()
            .map(|&v| {
                let d = v as f64 - avg_f;
                d * d
            })
            .sum::<f64>()
            / intervals.len() as f64;
        let jitter_std = variance.sqrt() / 1_000_000_000.0; // convert ns -> s

        PacingStats {
            avg_interval: Duration::from_nanos(avg),
            max_interval: Duration::from_nanos(max),
            min_interval: Duration::from_nanos(min),
            jitter_std,
            late_frames: self.late_frames,
            on_time_frames: self.on_time_frames,
        }
    }

    /// Return the ideal next presentation timestamp, or `None` if no frames
    /// have been recorded yet.
    #[must_use]
    pub fn next_ideal_pts_ns(&self) -> Option<u64> {
        self.history
            .back()
            .map(|t| t.pts_ns + self.cadence.interval.as_nanos() as u64)
    }

    /// Reset all internal state.
    pub fn reset(&mut self) {
        self.history.clear();
        self.late_frames = 0;
        self.on_time_frames = 0;
    }

    /// Return the number of recorded frames.
    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.history.len()
    }

    // -- internal helpers --

    fn intervals_ns(&self) -> Vec<u64> {
        self.history
            .iter()
            .zip(self.history.iter().skip(1))
            .map(|(a, b)| b.pts_ns.saturating_sub(a.pts_ns))
            .collect()
    }
}

/// Determine if a frame should be dropped to maintain target cadence.
///
/// Returns `true` if the frame arrives too early relative to the ideal
/// cadence interval, suggesting it would cause stutter if presented.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn should_drop_early_frame(
    last_pts_ns: u64,
    current_pts_ns: u64,
    cadence: &FrameCadence,
    min_fraction: f64,
) -> bool {
    let delta = current_pts_ns.saturating_sub(last_pts_ns);
    let threshold = (cadence.interval.as_nanos() as f64 * min_fraction) as u64;
    delta < threshold
}

/// Convert a frame index at a given FPS to a nanosecond timestamp.
#[must_use]
#[allow(clippy::cast_precision_loss)]
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
pub fn frame_index_to_ns(index: u64, fps: f64) -> u64 {
    assert!(fps > 0.0, "FPS must be positive");
    (index as f64 / fps * 1_000_000_000.0) as u64
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cadence_from_fps_60() {
        let c = FrameCadence::from_fps(60.0);
        assert!((c.fps - 60.0).abs() < f64::EPSILON);
        // ~16.666 ms
        let ms = c.interval.as_nanos() as f64 / 1_000_000.0;
        assert!((ms - 16.666).abs() < 0.1);
    }

    #[test]
    fn test_cadence_from_fps_30() {
        let c = FrameCadence::from_fps(30.0);
        let ms = c.interval.as_nanos() as f64 / 1_000_000.0;
        assert!((ms - 33.333).abs() < 0.1);
    }

    #[test]
    fn test_cadence_interval_secs() {
        let c = FrameCadence::from_fps(60.0);
        assert!((c.interval_secs() - 1.0 / 60.0).abs() < 1e-6);
    }

    #[test]
    fn test_pacer_empty_stats() {
        let c = FrameCadence::from_fps(60.0);
        let pacer = FramePacer::new(c, Duration::from_millis(2), 120);
        let stats = pacer.stats();
        assert_eq!(stats.avg_interval, Duration::ZERO);
        assert_eq!(stats.late_frames, 0);
    }

    #[test]
    fn test_pacer_uniform_frames() {
        let c = FrameCadence::from_fps(60.0);
        let mut pacer = FramePacer::new(c, Duration::from_millis(2), 120);
        for i in 0..60 {
            pacer.record_frame(frame_index_to_ns(i, 60.0));
        }
        let stats = pacer.stats();
        let avg_ms = stats.avg_interval.as_nanos() as f64 / 1_000_000.0;
        assert!((avg_ms - 16.666).abs() < 0.1);
        assert_eq!(stats.late_frames, 0);
    }

    #[test]
    fn test_pacer_late_frame_detection() {
        let c = FrameCadence::from_fps(60.0);
        let mut pacer = FramePacer::new(c, Duration::from_millis(2), 120);
        pacer.record_frame(0);
        // Normal frame at ~16.6ms
        pacer.record_frame(16_666_666);
        // Late frame at ~40ms (should be ~33.3ms)
        pacer.record_frame(56_666_666);
        let stats = pacer.stats();
        assert!(stats.late_frames >= 1);
    }

    #[test]
    fn test_pacer_next_ideal_pts() {
        let c = FrameCadence::from_fps(60.0);
        let mut pacer = FramePacer::new(c, Duration::from_millis(2), 120);
        assert!(pacer.next_ideal_pts_ns().is_none());
        pacer.record_frame(0);
        let next = pacer.next_ideal_pts_ns().expect("pts should succeed");
        assert!(next > 0);
    }

    #[test]
    fn test_pacer_reset() {
        let c = FrameCadence::from_fps(60.0);
        let mut pacer = FramePacer::new(c, Duration::from_millis(2), 120);
        pacer.record_frame(0);
        pacer.record_frame(16_666_666);
        pacer.reset();
        assert_eq!(pacer.frame_count(), 0);
        assert_eq!(pacer.stats().late_frames, 0);
    }

    #[test]
    fn test_should_drop_early_frame() {
        let c = FrameCadence::from_fps(60.0);
        // Frame arrives after only 5ms -- should be dropped (min_fraction 0.5 = 8.3ms)
        assert!(should_drop_early_frame(0, 5_000_000, &c, 0.5));
        // Frame arrives after 10ms -- should NOT be dropped
        assert!(!should_drop_early_frame(0, 10_000_000, &c, 0.5));
    }

    #[test]
    fn test_frame_index_to_ns() {
        let ns = frame_index_to_ns(60, 60.0);
        // 60 frames at 60fps = 1 second = 1_000_000_000 ns
        assert!((ns as f64 - 1_000_000_000.0).abs() < 1.0);
    }

    #[test]
    fn test_pacer_window_eviction() {
        let c = FrameCadence::from_fps(60.0);
        let mut pacer = FramePacer::new(c, Duration::from_millis(2), 5);
        for i in 0..10 {
            pacer.record_frame(frame_index_to_ns(i, 60.0));
        }
        assert_eq!(pacer.frame_count(), 5);
    }

    #[test]
    fn test_pacing_stats_jitter_zero_for_uniform() {
        let c = FrameCadence::from_fps(60.0);
        let mut pacer = FramePacer::new(c, Duration::from_millis(2), 120);
        for i in 0..30 {
            pacer.record_frame(frame_index_to_ns(i, 60.0));
        }
        let stats = pacer.stats();
        // For perfectly uniform frames, jitter should be near zero
        assert!(stats.jitter_std < 0.001);
    }

    #[test]
    fn test_cadence_120fps() {
        let c = FrameCadence::from_fps(120.0);
        let ms = c.interval.as_nanos() as f64 / 1_000_000.0;
        assert!((ms - 8.333).abs() < 0.1);
    }
}
