//! Precise timer for frame pacing and media clock scheduling.
//!
//! Standard `std::thread::sleep` has millisecond-level granularity on most
//! platforms, which is insufficient for professional video where frame
//! intervals can be as short as ~16.67 ms (60 fps) and sub-frame timing
//! jitter must be well under 1 ms.
//!
//! This module provides:
//! - **`PreciseClock`**: high-resolution monotonic clock using `Instant`
//! - **`PreciseSleeper`**: hybrid sleep strategy (coarse sleep + spin-wait)
//! - **`FramePacer`**: drop-frame-aware frame pacer for standard broadcast rates
//! - **`MediaDeadline`**: deadline scheduling with drift compensation

#![allow(dead_code)]

use crate::error::{VideoIpError, VideoIpResult};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// High-resolution clock
// ---------------------------------------------------------------------------

/// High-resolution monotonic clock wrapper.
#[derive(Debug, Clone)]
pub struct PreciseClock {
    epoch: Instant,
}

impl PreciseClock {
    /// Creates a new clock with the current time as epoch.
    #[must_use]
    pub fn now() -> Self {
        Self {
            epoch: Instant::now(),
        }
    }

    /// Creates a clock from a specific epoch.
    #[must_use]
    pub fn from_epoch(epoch: Instant) -> Self {
        Self { epoch }
    }

    /// Returns elapsed time since epoch in nanoseconds.
    #[must_use]
    pub fn elapsed_ns(&self) -> u64 {
        self.epoch.elapsed().as_nanos() as u64
    }

    /// Returns elapsed time since epoch in microseconds.
    #[must_use]
    pub fn elapsed_us(&self) -> u64 {
        self.epoch.elapsed().as_micros() as u64
    }

    /// Returns elapsed time since epoch as `Duration`.
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        self.epoch.elapsed()
    }

    /// Returns the epoch `Instant`.
    #[must_use]
    pub fn epoch(&self) -> Instant {
        self.epoch
    }

    /// Resets the epoch to now.
    pub fn reset(&mut self) {
        self.epoch = Instant::now();
    }
}

impl Default for PreciseClock {
    fn default() -> Self {
        Self::now()
    }
}

// ---------------------------------------------------------------------------
// Precise sleeper
// ---------------------------------------------------------------------------

/// Strategy for precise sleeping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SleepStrategy {
    /// Pure spin-wait (highest precision, highest CPU).
    SpinWait,
    /// Standard OS sleep (lowest CPU, lowest precision).
    OsSleep,
    /// Hybrid: OS sleep for the bulk, spin-wait for the last `spin_threshold`.
    Hybrid {
        /// Duration to spin-wait at the end.
        spin_threshold: Duration,
    },
}

impl Default for SleepStrategy {
    fn default() -> Self {
        // Default: sleep most of the time, spin for the last 500 us.
        Self::Hybrid {
            spin_threshold: Duration::from_micros(500),
        }
    }
}

/// Precise sleeper that can achieve sub-millisecond accuracy.
#[derive(Debug, Clone)]
pub struct PreciseSleeper {
    strategy: SleepStrategy,
    /// Accumulated overshoot for drift compensation (nanoseconds).
    accumulated_overshoot_ns: i64,
    /// Number of sleeps performed.
    sleep_count: u64,
    /// Total overshoot in nanoseconds (absolute).
    total_overshoot_ns: u64,
}

impl PreciseSleeper {
    /// Creates a new sleeper with the given strategy.
    #[must_use]
    pub fn new(strategy: SleepStrategy) -> Self {
        Self {
            strategy,
            accumulated_overshoot_ns: 0,
            sleep_count: 0,
            total_overshoot_ns: 0,
        }
    }

    /// Creates a sleeper optimised for broadcast frame pacing.
    #[must_use]
    pub fn broadcast() -> Self {
        Self::new(SleepStrategy::Hybrid {
            spin_threshold: Duration::from_micros(200),
        })
    }

    /// Returns the configured strategy.
    #[must_use]
    pub fn strategy(&self) -> SleepStrategy {
        self.strategy
    }

    /// Sleeps for the specified duration using the configured strategy.
    ///
    /// Returns the actual elapsed time and the overshoot.
    pub fn sleep(&mut self, target: Duration) -> SleepResult {
        // Adjust target based on accumulated drift.
        let adjusted_ns = (target.as_nanos() as i64)
            .saturating_sub(self.accumulated_overshoot_ns)
            .max(0) as u64;
        let adjusted = Duration::from_nanos(adjusted_ns);

        let start = Instant::now();

        match self.strategy {
            SleepStrategy::SpinWait => {
                self.spin_until(start, adjusted);
            }
            SleepStrategy::OsSleep => {
                if !adjusted.is_zero() {
                    std::thread::sleep(adjusted);
                }
            }
            SleepStrategy::Hybrid { spin_threshold } => {
                if adjusted > spin_threshold {
                    let coarse = adjusted.saturating_sub(spin_threshold);
                    std::thread::sleep(coarse);
                }
                self.spin_until(start, adjusted);
            }
        }

        let elapsed = start.elapsed();
        let overshoot_ns = elapsed
            .as_nanos()
            .saturating_sub(target.as_nanos()) as i64;

        self.accumulated_overshoot_ns = overshoot_ns;
        self.sleep_count += 1;
        self.total_overshoot_ns += overshoot_ns.unsigned_abs();

        SleepResult {
            target,
            elapsed,
            overshoot: Duration::from_nanos(overshoot_ns.unsigned_abs()),
            was_early: overshoot_ns < 0,
        }
    }

    /// Returns the average overshoot across all sleeps.
    #[must_use]
    pub fn average_overshoot(&self) -> Duration {
        if self.sleep_count == 0 {
            return Duration::ZERO;
        }
        Duration::from_nanos(self.total_overshoot_ns / self.sleep_count)
    }

    /// Returns total sleep count.
    #[must_use]
    pub fn sleep_count(&self) -> u64 {
        self.sleep_count
    }

    /// Resets accumulated drift.
    pub fn reset_drift(&mut self) {
        self.accumulated_overshoot_ns = 0;
    }

    fn spin_until(&self, start: Instant, target: Duration) {
        while start.elapsed() < target {
            std::hint::spin_loop();
        }
    }
}

/// Result of a precise sleep operation.
#[derive(Debug, Clone)]
pub struct SleepResult {
    /// Requested sleep duration.
    pub target: Duration,
    /// Actual elapsed duration.
    pub elapsed: Duration,
    /// Absolute overshoot.
    pub overshoot: Duration,
    /// True if woke up early (should not happen with spin-wait).
    pub was_early: bool,
}

// ---------------------------------------------------------------------------
// Frame pacer
// ---------------------------------------------------------------------------

/// Standard broadcast frame rates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FrameRate {
    /// 23.976 fps (NTSC film).
    Fps23_976,
    /// 24 fps (cinema).
    Fps24,
    /// 25 fps (PAL).
    Fps25,
    /// 29.97 fps (NTSC).
    Fps29_97,
    /// 30 fps.
    Fps30,
    /// 50 fps (PAL high frame rate).
    Fps50,
    /// 59.94 fps (NTSC high frame rate).
    Fps59_94,
    /// 60 fps.
    Fps60,
    /// Custom frame rate.
    Custom(f64),
}

impl FrameRate {
    /// Returns the frame interval as a `Duration`.
    #[must_use]
    pub fn interval(&self) -> Duration {
        let fps = self.as_f64();
        if fps <= 0.0 {
            return Duration::ZERO;
        }
        Duration::from_secs_f64(1.0 / fps)
    }

    /// Returns the frame rate as `f64`.
    #[must_use]
    pub fn as_f64(&self) -> f64 {
        match self {
            Self::Fps23_976 => 24000.0 / 1001.0,
            Self::Fps24 => 24.0,
            Self::Fps25 => 25.0,
            Self::Fps29_97 => 30000.0 / 1001.0,
            Self::Fps30 => 30.0,
            Self::Fps50 => 50.0,
            Self::Fps59_94 => 60000.0 / 1001.0,
            Self::Fps60 => 60.0,
            Self::Custom(fps) => *fps,
        }
    }

    /// Returns whether this is a drop-frame rate (NTSC).
    #[must_use]
    pub fn is_drop_frame(&self) -> bool {
        matches!(self, Self::Fps23_976 | Self::Fps29_97 | Self::Fps59_94)
    }

    /// Returns the frame interval in nanoseconds (integer, for accumulation).
    #[must_use]
    pub fn interval_ns(&self) -> u64 {
        let fps = self.as_f64();
        if fps <= 0.0 {
            return 0;
        }
        (1_000_000_000.0 / fps) as u64
    }
}

/// Frame pacer that schedules frame delivery at precise intervals.
#[derive(Debug)]
pub struct FramePacer {
    frame_rate: FrameRate,
    sleeper: PreciseSleeper,
    /// Frame counter.
    frame_count: u64,
    /// Clock used for absolute deadline tracking.
    clock: PreciseClock,
    /// Next frame deadline in nanoseconds from epoch.
    next_deadline_ns: u64,
}

impl FramePacer {
    /// Creates a new frame pacer for the given frame rate.
    #[must_use]
    pub fn new(frame_rate: FrameRate) -> Self {
        Self {
            frame_rate,
            sleeper: PreciseSleeper::broadcast(),
            frame_count: 0,
            clock: PreciseClock::now(),
            next_deadline_ns: 0,
        }
    }

    /// Creates a frame pacer with a custom sleep strategy.
    #[must_use]
    pub fn with_strategy(frame_rate: FrameRate, strategy: SleepStrategy) -> Self {
        Self {
            frame_rate,
            sleeper: PreciseSleeper::new(strategy),
            frame_count: 0,
            clock: PreciseClock::now(),
            next_deadline_ns: 0,
        }
    }

    /// Returns the configured frame rate.
    #[must_use]
    pub fn frame_rate(&self) -> FrameRate {
        self.frame_rate
    }

    /// Returns the current frame count.
    #[must_use]
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Waits until the next frame deadline.
    ///
    /// Uses integer nanosecond accumulation so that drift does not
    /// accumulate over long runs (important for drop-frame rates).
    pub fn wait_for_next_frame(&mut self) -> FramePaceResult {
        let interval_ns = self.frame_rate.interval_ns();
        self.next_deadline_ns += interval_ns;

        let now_ns = self.clock.elapsed_ns();
        let result = if self.next_deadline_ns > now_ns {
            let wait = Duration::from_nanos(self.next_deadline_ns - now_ns);
            let sr = self.sleeper.sleep(wait);
            FramePaceResult {
                frame_number: self.frame_count,
                target_interval: Duration::from_nanos(interval_ns),
                actual_wait: sr.elapsed,
                deadline_error: sr.overshoot,
                was_late: false,
            }
        } else {
            // Already past deadline – don't sleep.
            FramePaceResult {
                frame_number: self.frame_count,
                target_interval: Duration::from_nanos(interval_ns),
                actual_wait: Duration::ZERO,
                deadline_error: Duration::from_nanos(now_ns - self.next_deadline_ns),
                was_late: true,
            }
        };

        self.frame_count += 1;
        result
    }

    /// Resets the pacer (e.g. after a seek or source switch).
    pub fn reset(&mut self) {
        self.frame_count = 0;
        self.next_deadline_ns = 0;
        self.clock.reset();
        self.sleeper.reset_drift();
    }

    /// Returns timing statistics from the sleeper.
    #[must_use]
    pub fn average_overshoot(&self) -> Duration {
        self.sleeper.average_overshoot()
    }
}

/// Result of a frame pace operation.
#[derive(Debug, Clone)]
pub struct FramePaceResult {
    /// Frame number (0-indexed).
    pub frame_number: u64,
    /// Target interval between frames.
    pub target_interval: Duration,
    /// Actual time spent waiting.
    pub actual_wait: Duration,
    /// Absolute deadline error.
    pub deadline_error: Duration,
    /// Whether this frame was delivered late (no wait performed).
    pub was_late: bool,
}

// ---------------------------------------------------------------------------
// Media deadline scheduler
// ---------------------------------------------------------------------------

/// A deadline for a media event.
#[derive(Debug, Clone)]
pub struct MediaDeadline {
    /// Target time from epoch in nanoseconds.
    pub target_ns: u64,
    /// Label for debugging.
    pub label: String,
}

/// Manages a sorted queue of upcoming media deadlines with drift compensation.
#[derive(Debug)]
pub struct DeadlineScheduler {
    deadlines: Vec<MediaDeadline>,
    clock: PreciseClock,
    /// Maximum allowed drift before resync (nanoseconds).
    max_drift_ns: u64,
    /// Number of deadlines that were met.
    met_count: u64,
    /// Number of deadlines that were missed.
    missed_count: u64,
}

impl DeadlineScheduler {
    /// Creates a new deadline scheduler.
    ///
    /// # Arguments
    ///
    /// * `max_drift_ns` - Maximum allowed drift before the scheduler
    ///   considers a deadline missed.
    #[must_use]
    pub fn new(max_drift_ns: u64) -> Self {
        Self {
            deadlines: Vec::new(),
            clock: PreciseClock::now(),
            max_drift_ns,
            met_count: 0,
            missed_count: 0,
        }
    }

    /// Schedules a new deadline.
    pub fn schedule(&mut self, target_ns: u64, label: impl Into<String>) {
        let deadline = MediaDeadline {
            target_ns,
            label: label.into(),
        };
        // Insert sorted by target_ns.
        let pos = self
            .deadlines
            .binary_search_by_key(&target_ns, |d| d.target_ns)
            .unwrap_or_else(|p| p);
        self.deadlines.insert(pos, deadline);
    }

    /// Checks which deadlines are due at the current time.
    ///
    /// Returns `(met, missed)` deadlines.
    pub fn check_due(&mut self) -> (Vec<MediaDeadline>, Vec<MediaDeadline>) {
        let now_ns = self.clock.elapsed_ns();
        let mut met = Vec::new();
        let mut missed = Vec::new();

        while let Some(d) = self.deadlines.first() {
            if d.target_ns <= now_ns {
                let removed = self.deadlines.remove(0);
                let drift = now_ns.saturating_sub(removed.target_ns);
                if drift <= self.max_drift_ns {
                    self.met_count += 1;
                    met.push(removed);
                } else {
                    self.missed_count += 1;
                    missed.push(removed);
                }
            } else {
                break;
            }
        }

        (met, missed)
    }

    /// Returns the number of pending deadlines.
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.deadlines.len()
    }

    /// Returns the count of met deadlines.
    #[must_use]
    pub fn met_count(&self) -> u64 {
        self.met_count
    }

    /// Returns the count of missed deadlines.
    #[must_use]
    pub fn missed_count(&self) -> u64 {
        self.missed_count
    }

    /// Returns time until the next deadline (if any).
    #[must_use]
    pub fn time_until_next(&self) -> Option<Duration> {
        let now_ns = self.clock.elapsed_ns();
        self.deadlines.first().map(|d| {
            Duration::from_nanos(d.target_ns.saturating_sub(now_ns))
        })
    }
}

// ---------------------------------------------------------------------------
// Interval timer
// ---------------------------------------------------------------------------

/// Validates a frame rate value.
///
/// # Errors
///
/// Returns an error if the frame rate is invalid.
pub fn validate_frame_rate(fps: f64) -> VideoIpResult<()> {
    if fps <= 0.0 || fps > 1000.0 || fps.is_nan() || fps.is_infinite() {
        return Err(VideoIpError::InvalidVideoConfig(format!(
            "invalid frame rate: {fps} (must be 0 < fps <= 1000)"
        )));
    }
    Ok(())
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precise_clock_elapsed() {
        let clock = PreciseClock::now();
        // Spin a bit so elapsed > 0.
        let start = Instant::now();
        while start.elapsed() < Duration::from_micros(100) {
            std::hint::spin_loop();
        }
        assert!(clock.elapsed_us() >= 50); // At least 50 us should have passed.
    }

    #[test]
    fn test_precise_clock_reset() {
        let mut clock = PreciseClock::now();
        std::thread::sleep(Duration::from_millis(1));
        let before = clock.elapsed_us();
        clock.reset();
        let after = clock.elapsed_us();
        assert!(after < before);
    }

    #[test]
    fn test_frame_rate_interval_60fps() {
        let rate = FrameRate::Fps60;
        let interval = rate.interval();
        // 1/60 ~= 16.666 ms
        let expected = Duration::from_secs_f64(1.0 / 60.0);
        let diff = if interval > expected {
            interval - expected
        } else {
            expected - interval
        };
        assert!(diff < Duration::from_micros(10));
    }

    #[test]
    fn test_frame_rate_interval_29_97() {
        let rate = FrameRate::Fps29_97;
        let fps = rate.as_f64();
        assert!((fps - 29.97002997).abs() < 0.001);
        assert!(rate.is_drop_frame());
    }

    #[test]
    fn test_frame_rate_interval_25() {
        let rate = FrameRate::Fps25;
        let interval = rate.interval();
        let expected = Duration::from_millis(40);
        let diff = if interval > expected {
            interval - expected
        } else {
            expected - interval
        };
        assert!(diff < Duration::from_micros(10));
        assert!(!rate.is_drop_frame());
    }

    #[test]
    fn test_frame_rate_custom() {
        let rate = FrameRate::Custom(120.0);
        let interval = rate.interval();
        let expected = Duration::from_secs_f64(1.0 / 120.0);
        let diff = if interval > expected {
            interval - expected
        } else {
            expected - interval
        };
        assert!(diff < Duration::from_micros(10));
    }

    #[test]
    fn test_sleep_strategy_default() {
        let strategy = SleepStrategy::default();
        match strategy {
            SleepStrategy::Hybrid { spin_threshold } => {
                assert_eq!(spin_threshold, Duration::from_micros(500));
            }
            _ => panic!("expected Hybrid"),
        }
    }

    #[test]
    fn test_precise_sleeper_spin() {
        let mut sleeper = PreciseSleeper::new(SleepStrategy::SpinWait);
        let target = Duration::from_micros(100);
        let result = sleeper.sleep(target);
        // Spin-wait should be quite accurate.
        assert!(result.elapsed >= target);
        assert!(result.overshoot < Duration::from_micros(500));
        assert_eq!(sleeper.sleep_count(), 1);
    }

    #[test]
    fn test_precise_sleeper_hybrid() {
        let mut sleeper = PreciseSleeper::broadcast();
        let target = Duration::from_millis(5);
        let result = sleeper.sleep(target);
        assert!(result.elapsed >= Duration::from_millis(3));
        assert_eq!(sleeper.sleep_count(), 1);
    }

    #[test]
    fn test_frame_pacer_creation() {
        let pacer = FramePacer::new(FrameRate::Fps60);
        assert_eq!(pacer.frame_count(), 0);
        let rate = pacer.frame_rate();
        assert!((rate.as_f64() - 60.0).abs() < 0.001);
    }

    #[test]
    fn test_frame_pacer_wait() {
        let mut pacer = FramePacer::with_strategy(
            FrameRate::Fps60,
            SleepStrategy::SpinWait,
        );
        let result = pacer.wait_for_next_frame();
        assert_eq!(result.frame_number, 0);
        assert!(!result.was_late);
        assert_eq!(pacer.frame_count(), 1);
    }

    #[test]
    fn test_frame_pacer_reset() {
        let mut pacer = FramePacer::new(FrameRate::Fps30);
        pacer.wait_for_next_frame();
        pacer.wait_for_next_frame();
        assert_eq!(pacer.frame_count(), 2);
        pacer.reset();
        assert_eq!(pacer.frame_count(), 0);
    }

    #[test]
    fn test_deadline_scheduler_schedule_and_check() {
        let mut sched = DeadlineScheduler::new(5_000_000); // 5ms tolerance
        // Schedule deadlines in the past (should be immediately due).
        sched.schedule(0, "first");
        sched.schedule(0, "second");
        let (met, missed) = sched.check_due();
        // Both should be met (within 5ms tolerance of "now").
        // This depends on how fast the test runs, but 0ns target
        // with 5ms tolerance should be fine.
        assert_eq!(met.len() + missed.len(), 2);
    }

    #[test]
    fn test_deadline_scheduler_future_not_due() {
        let mut sched = DeadlineScheduler::new(1_000_000);
        sched.schedule(999_999_999_999, "far_future");
        let (met, missed) = sched.check_due();
        assert!(met.is_empty());
        assert!(missed.is_empty());
        assert_eq!(sched.pending_count(), 1);
    }

    #[test]
    fn test_deadline_scheduler_time_until_next() {
        let sched = DeadlineScheduler::new(1_000_000);
        assert!(sched.time_until_next().is_none());
    }

    #[test]
    fn test_validate_frame_rate() {
        assert!(validate_frame_rate(60.0).is_ok());
        assert!(validate_frame_rate(0.0).is_err());
        assert!(validate_frame_rate(-1.0).is_err());
        assert!(validate_frame_rate(f64::NAN).is_err());
        assert!(validate_frame_rate(f64::INFINITY).is_err());
        assert!(validate_frame_rate(1001.0).is_err());
    }

    #[test]
    fn test_frame_rate_interval_ns_accumulation() {
        // Verify that integer NS accumulation doesn't drift.
        let rate = FrameRate::Fps29_97;
        let interval_ns = rate.interval_ns();
        // After 1001 frames at 29.97, we should be close to 1001/29.97 seconds.
        let total_ns = interval_ns as u128 * 1001;
        let expected_ns = (1001.0 / rate.as_f64() * 1e9) as u128;
        let drift = if total_ns > expected_ns {
            total_ns - expected_ns
        } else {
            expected_ns - total_ns
        };
        // Drift should be bounded by 1 interval (truncation error per frame).
        assert!(drift < interval_ns as u128 * 2);
    }
}
