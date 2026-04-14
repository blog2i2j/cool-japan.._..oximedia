//! Production countdown and elapsed timers for live broadcast workflows.
//!
//! Provides segment timers, show timers, and segment-remaining timers that
//! integrate with the switcher's frame clock.  Each timer can be configured to
//! auto-reset when a cut occurs on a designated M/E row.
//!
//! # Timer Types
//!
//! | Type | Counts | Use case |
//! |------|--------|----------|
//! | `Countdown` | Down to zero | Segment / interview time-limit |
//! | `Elapsed` | Up from zero | Show duration / segment elapsed |
//! | `SegmentRemaining` | Down from segment budget | Remaining time in the current segment |
//!
//! # Example
//!
//! ```rust
//! use oximedia_switcher::production_timer::{ProductionTimer, TimerConfig, TimerKind, TimerState};
//!
//! let cfg = TimerConfig::countdown("Segment A", 30_000);
//! let mut timer = ProductionTimer::new(0, cfg);
//!
//! timer.start().expect("timer should start");
//! assert_eq!(timer.state(), TimerState::Running);
//!
//! // Advance 1 000 ms worth of frames (25 fps → 25 frames).
//! for _ in 0..25 {
//!     timer.tick_frame(40);   // 40 ms per frame at 25 fps
//! }
//! assert!(timer.elapsed_ms() >= 1_000);
//! ```

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ────────────────────────────────────────────────────────────────────────────
// Errors
// ────────────────────────────────────────────────────────────────────────────

/// Errors from the production timer subsystem.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum ProductionTimerError {
    /// Attempted to start a timer that is already running.
    #[error("Timer {0} is already running")]
    AlreadyRunning(usize),

    /// Attempted to pause a timer that is not running.
    #[error("Timer {0} is not running")]
    NotRunning(usize),

    /// Attempted to operate on an unknown timer ID.
    #[error("Timer ID {0} not found")]
    NotFound(usize),

    /// The configured duration is zero, which is invalid for a countdown timer.
    #[error("Countdown duration must be greater than zero")]
    ZeroDuration,
}

// ────────────────────────────────────────────────────────────────────────────
// Timer kind & state
// ────────────────────────────────────────────────────────────────────────────

/// What the timer counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TimerKind {
    /// Counts down from a preset duration to zero; fires an `Expired` event.
    Countdown,
    /// Counts up from zero; has no natural end.
    Elapsed,
    /// Counts down the remaining budget within the current segment; resets
    /// automatically when a cut is detected on the linked M/E row.
    SegmentRemaining,
}

/// Current operational state of a timer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TimerState {
    /// Timer has been created but never started.
    Stopped,
    /// Timer is actively counting.
    Running,
    /// Timer has been paused mid-count.
    Paused,
    /// Timer has reached zero (countdown/segment-remaining only).
    Expired,
}

// ────────────────────────────────────────────────────────────────────────────
// Configuration
// ────────────────────────────────────────────────────────────────────────────

/// Configuration for a production timer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimerConfig {
    /// Human-readable label shown in the multiviewer UMD or production display.
    pub label: String,
    /// What the timer counts.
    pub kind: TimerKind,
    /// Preset duration in milliseconds (ignored for `Elapsed` timers).
    pub duration_ms: u64,
    /// When `true` the timer resets to its initial state after expiry.
    pub auto_reset: bool,
    /// Optional M/E row that triggers an auto-reset on cut (0-based index).
    pub auto_reset_on_cut_me: Option<usize>,
    /// When `true` the timer automatically starts after an auto-reset.
    pub auto_start_after_reset: bool,
}

impl TimerConfig {
    /// Create a countdown timer configuration.
    pub fn countdown(label: impl Into<String>, duration_ms: u64) -> Self {
        Self {
            label: label.into(),
            kind: TimerKind::Countdown,
            duration_ms,
            auto_reset: false,
            auto_reset_on_cut_me: None,
            auto_start_after_reset: false,
        }
    }

    /// Create an elapsed (count-up) timer configuration.
    pub fn elapsed(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            kind: TimerKind::Elapsed,
            duration_ms: 0,
            auto_reset: false,
            auto_reset_on_cut_me: None,
            auto_start_after_reset: false,
        }
    }

    /// Create a segment-remaining timer that resets automatically when a cut
    /// occurs on the specified M/E row.
    pub fn segment_remaining(
        label: impl Into<String>,
        segment_duration_ms: u64,
        me_row: usize,
    ) -> Self {
        Self {
            label: label.into(),
            kind: TimerKind::SegmentRemaining,
            duration_ms: segment_duration_ms,
            auto_reset: true,
            auto_reset_on_cut_me: Some(me_row),
            auto_start_after_reset: true,
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// ProductionTimer
// ────────────────────────────────────────────────────────────────────────────

/// A single production timer instance.
pub struct ProductionTimer {
    /// Unique timer ID within the manager.
    pub id: usize,
    config: TimerConfig,
    state: TimerState,
    /// Accumulated elapsed milliseconds since the last reset/start.
    elapsed_ms: u64,
}

impl ProductionTimer {
    /// Create a new timer with the given ID and configuration.
    pub fn new(id: usize, config: TimerConfig) -> Self {
        Self {
            id,
            config,
            state: TimerState::Stopped,
            elapsed_ms: 0,
        }
    }

    // ── Control ──────────────────────────────────────────────────────────────

    /// Start or resume the timer.
    pub fn start(&mut self) -> Result<(), ProductionTimerError> {
        match self.state {
            TimerState::Running => Err(ProductionTimerError::AlreadyRunning(self.id)),
            TimerState::Expired => {
                // Allow restart from expired state after manual start.
                self.elapsed_ms = 0;
                self.state = TimerState::Running;
                Ok(())
            }
            _ => {
                self.state = TimerState::Running;
                Ok(())
            }
        }
    }

    /// Pause the timer.
    pub fn pause(&mut self) -> Result<(), ProductionTimerError> {
        if self.state != TimerState::Running {
            return Err(ProductionTimerError::NotRunning(self.id));
        }
        self.state = TimerState::Paused;
        Ok(())
    }

    /// Stop and reset the timer to its initial state.
    pub fn stop(&mut self) {
        self.state = TimerState::Stopped;
        self.elapsed_ms = 0;
    }

    /// Reset the timer to zero without changing the running state.
    pub fn reset(&mut self) {
        self.elapsed_ms = 0;
        if self.state == TimerState::Expired {
            self.state = TimerState::Stopped;
        }
    }

    // ── Frame tick ───────────────────────────────────────────────────────────

    /// Advance the timer by `frame_duration_ms` milliseconds (one video frame).
    ///
    /// Returns `true` if the timer just expired this tick.
    pub fn tick_frame(&mut self, frame_duration_ms: u64) -> bool {
        if self.state != TimerState::Running {
            return false;
        }

        self.elapsed_ms = self.elapsed_ms.saturating_add(frame_duration_ms);

        let expired = match self.config.kind {
            TimerKind::Countdown | TimerKind::SegmentRemaining => {
                self.elapsed_ms >= self.config.duration_ms
            }
            TimerKind::Elapsed => false,
        };

        if expired {
            if self.config.auto_reset {
                self.elapsed_ms = 0;
                if !self.config.auto_start_after_reset {
                    self.state = TimerState::Stopped;
                }
                // If auto_start_after_reset is true we stay Running.
            } else {
                self.elapsed_ms = self.config.duration_ms;
                self.state = TimerState::Expired;
            }
            return true;
        }

        false
    }

    /// Notify the timer that a cut occurred on the given M/E row.
    ///
    /// If this timer is configured for `auto_reset_on_cut_me` with a matching
    /// row, the timer is reset and (optionally) restarted.
    pub fn notify_cut(&mut self, me_row: usize) {
        if let Some(linked_me) = self.config.auto_reset_on_cut_me {
            if linked_me == me_row {
                self.elapsed_ms = 0;
                self.state = if self.config.auto_start_after_reset {
                    TimerState::Running
                } else {
                    TimerState::Stopped
                };
            }
        }
    }

    // ── Accessors ────────────────────────────────────────────────────────────

    /// Current timer state.
    pub fn state(&self) -> TimerState {
        self.state
    }

    /// Milliseconds elapsed since the last reset.
    pub fn elapsed_ms(&self) -> u64 {
        self.elapsed_ms
    }

    /// For countdown/segment-remaining timers: milliseconds remaining.
    /// Returns `None` for elapsed timers.
    pub fn remaining_ms(&self) -> Option<u64> {
        match self.config.kind {
            TimerKind::Elapsed => None,
            _ => Some(self.config.duration_ms.saturating_sub(self.elapsed_ms)),
        }
    }

    /// The timer's configuration.
    pub fn config(&self) -> &TimerConfig {
        &self.config
    }

    /// Whether the timer is currently running.
    pub fn is_running(&self) -> bool {
        self.state == TimerState::Running
    }

    /// Whether the timer has expired.
    pub fn is_expired(&self) -> bool {
        self.state == TimerState::Expired
    }

    /// Formatted remaining time as `MM:SS` (countdown timers only).
    pub fn remaining_display(&self) -> Option<String> {
        let remaining = self.remaining_ms()?;
        let total_secs = remaining / 1_000;
        let mins = total_secs / 60;
        let secs = total_secs % 60;
        Some(format!("{mins:02}:{secs:02}"))
    }

    /// Formatted elapsed time as `MM:SS`.
    pub fn elapsed_display(&self) -> String {
        let total_secs = self.elapsed_ms / 1_000;
        let mins = total_secs / 60;
        let secs = total_secs % 60;
        format!("{mins:02}:{secs:02}")
    }
}

// ────────────────────────────────────────────────────────────────────────────
// ProductionTimerManager
// ────────────────────────────────────────────────────────────────────────────

/// Manages a set of production timers, distributing frame ticks and cut
/// notifications to all registered timers.
pub struct ProductionTimerManager {
    timers: Vec<ProductionTimer>,
    next_id: usize,
}

impl ProductionTimerManager {
    /// Create a new, empty timer manager.
    pub fn new() -> Self {
        Self {
            timers: Vec::new(),
            next_id: 0,
        }
    }

    /// Register a new timer and return its assigned ID.
    pub fn add_timer(&mut self, config: TimerConfig) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        self.timers.push(ProductionTimer::new(id, config));
        id
    }

    /// Remove a timer by ID.  Returns an error if not found.
    pub fn remove_timer(&mut self, id: usize) -> Result<(), ProductionTimerError> {
        let pos = self
            .timers
            .iter()
            .position(|t| t.id == id)
            .ok_or(ProductionTimerError::NotFound(id))?;
        self.timers.remove(pos);
        Ok(())
    }

    /// Get an immutable reference to a timer.
    pub fn get_timer(&self, id: usize) -> Option<&ProductionTimer> {
        self.timers.iter().find(|t| t.id == id)
    }

    /// Get a mutable reference to a timer.
    pub fn get_timer_mut(&mut self, id: usize) -> Option<&mut ProductionTimer> {
        self.timers.iter_mut().find(|t| t.id == id)
    }

    /// Advance all timers by one video frame of `frame_duration_ms`.
    ///
    /// Returns the IDs of any timers that expired during this tick.
    pub fn tick_frame(&mut self, frame_duration_ms: u64) -> Vec<usize> {
        let mut expired_ids = Vec::new();
        for timer in &mut self.timers {
            if timer.tick_frame(frame_duration_ms) {
                expired_ids.push(timer.id);
            }
        }
        expired_ids
    }

    /// Notify all timers that a cut occurred on `me_row`.
    pub fn notify_cut(&mut self, me_row: usize) {
        for timer in &mut self.timers {
            timer.notify_cut(me_row);
        }
    }

    /// Start a timer by ID.
    pub fn start(&mut self, id: usize) -> Result<(), ProductionTimerError> {
        self.get_timer_mut(id)
            .ok_or(ProductionTimerError::NotFound(id))
            .and_then(|t| t.start())
    }

    /// Pause a timer by ID.
    pub fn pause(&mut self, id: usize) -> Result<(), ProductionTimerError> {
        self.get_timer_mut(id)
            .ok_or(ProductionTimerError::NotFound(id))
            .and_then(|t| t.pause())
    }

    /// Stop and reset a timer by ID.
    pub fn stop(&mut self, id: usize) -> Result<(), ProductionTimerError> {
        self.get_timer_mut(id)
            .ok_or(ProductionTimerError::NotFound(id))
            .map(|t| t.stop())
    }

    /// Total number of registered timers.
    pub fn timer_count(&self) -> usize {
        self.timers.len()
    }

    /// IDs of all currently running timers.
    pub fn running_timer_ids(&self) -> Vec<usize> {
        self.timers
            .iter()
            .filter(|t| t.is_running())
            .map(|t| t.id)
            .collect()
    }
}

impl Default for ProductionTimerManager {
    fn default() -> Self {
        Self::new()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// 40 ms per frame → 25 fps
    const FRAME_MS: u64 = 40;

    #[test]
    fn test_countdown_timer_creation() {
        let cfg = TimerConfig::countdown("Test", 10_000);
        let timer = ProductionTimer::new(0, cfg);
        assert_eq!(timer.state(), TimerState::Stopped);
        assert_eq!(timer.elapsed_ms(), 0);
        assert_eq!(timer.remaining_ms(), Some(10_000));
    }

    #[test]
    fn test_start_and_tick() {
        let cfg = TimerConfig::countdown("T", 5_000);
        let mut timer = ProductionTimer::new(0, cfg);
        timer.start().expect("start ok");
        assert_eq!(timer.state(), TimerState::Running);

        for _ in 0..25 {
            timer.tick_frame(FRAME_MS);
        }
        // 25 × 40 ms = 1 000 ms elapsed
        assert_eq!(timer.elapsed_ms(), 1_000);
        assert_eq!(timer.remaining_ms(), Some(4_000));
    }

    #[test]
    fn test_countdown_expiry() {
        let cfg = TimerConfig::countdown("T", 400); // 10 frames
        let mut timer = ProductionTimer::new(0, cfg);
        timer.start().expect("start ok");

        let mut fired = false;
        for _ in 0..15 {
            if timer.tick_frame(FRAME_MS) {
                fired = true;
            }
        }
        assert!(fired, "expiry event should have fired");
        assert_eq!(timer.state(), TimerState::Expired);
        assert!(timer.is_expired());
    }

    #[test]
    fn test_elapsed_timer_no_expiry() {
        let cfg = TimerConfig::elapsed("Show Timer");
        let mut timer = ProductionTimer::new(0, cfg);
        timer.start().expect("start ok");

        for _ in 0..100 {
            let expired = timer.tick_frame(FRAME_MS);
            assert!(!expired, "elapsed timers never expire");
        }
        assert_eq!(timer.elapsed_ms(), 100 * FRAME_MS);
        assert_eq!(timer.remaining_ms(), None);
    }

    #[test]
    fn test_pause_stops_counting() {
        let cfg = TimerConfig::countdown("T", 10_000);
        let mut timer = ProductionTimer::new(0, cfg);
        timer.start().expect("start ok");

        for _ in 0..10 {
            timer.tick_frame(FRAME_MS);
        }
        let elapsed_before = timer.elapsed_ms();

        timer.pause().expect("pause ok");
        assert_eq!(timer.state(), TimerState::Paused);

        for _ in 0..10 {
            timer.tick_frame(FRAME_MS);
        }
        // Should not have advanced while paused.
        assert_eq!(timer.elapsed_ms(), elapsed_before);
    }

    #[test]
    fn test_stop_resets_timer() {
        let cfg = TimerConfig::elapsed("T");
        let mut timer = ProductionTimer::new(0, cfg);
        timer.start().expect("start ok");
        for _ in 0..50 {
            timer.tick_frame(FRAME_MS);
        }
        assert!(timer.elapsed_ms() > 0);

        timer.stop();
        assert_eq!(timer.state(), TimerState::Stopped);
        assert_eq!(timer.elapsed_ms(), 0);
    }

    #[test]
    fn test_double_start_error() {
        let cfg = TimerConfig::countdown("T", 5_000);
        let mut timer = ProductionTimer::new(0, cfg);
        timer.start().expect("first start ok");
        let err = timer.start().expect_err("double start should fail");
        assert!(matches!(err, ProductionTimerError::AlreadyRunning(0)));
    }

    #[test]
    fn test_pause_when_stopped_error() {
        let cfg = TimerConfig::countdown("T", 5_000);
        let mut timer = ProductionTimer::new(0, cfg);
        let err = timer.pause().expect_err("pause when stopped should fail");
        assert!(matches!(err, ProductionTimerError::NotRunning(0)));
    }

    #[test]
    fn test_segment_remaining_auto_reset_on_cut() {
        let cfg = TimerConfig::segment_remaining("Seg", 2_000, 0);
        let mut timer = ProductionTimer::new(0, cfg);
        timer.start().expect("start ok");

        for _ in 0..25 {
            timer.tick_frame(FRAME_MS); // 1 000 ms
        }
        assert_eq!(timer.elapsed_ms(), 1_000);

        // Cut on M/E row 0 should reset.
        timer.notify_cut(0);
        assert_eq!(timer.elapsed_ms(), 0);
        // auto_start_after_reset is true for segment_remaining.
        assert_eq!(timer.state(), TimerState::Running);
    }

    #[test]
    fn test_segment_remaining_ignores_other_me_cut() {
        let cfg = TimerConfig::segment_remaining("Seg", 2_000, 0);
        let mut timer = ProductionTimer::new(0, cfg);
        timer.start().expect("start ok");

        for _ in 0..10 {
            timer.tick_frame(FRAME_MS);
        }
        let elapsed_before = timer.elapsed_ms();

        // Cut on a different M/E row should be ignored.
        timer.notify_cut(1);
        assert_eq!(timer.elapsed_ms(), elapsed_before);
    }

    #[test]
    fn test_timer_manager_add_and_tick() {
        let mut mgr = ProductionTimerManager::new();
        let id = mgr.add_timer(TimerConfig::elapsed("Show"));
        assert_eq!(mgr.timer_count(), 1);

        mgr.start(id).expect("start ok");
        let expired = mgr.tick_frame(FRAME_MS);
        assert!(expired.is_empty()); // elapsed timers never expire

        let timer = mgr.get_timer(id).expect("timer found");
        assert_eq!(timer.elapsed_ms(), FRAME_MS);
    }

    #[test]
    fn test_timer_manager_expiry_notification() {
        let mut mgr = ProductionTimerManager::new();
        let id = mgr.add_timer(TimerConfig::countdown("T", 80)); // 2 frames

        mgr.start(id).expect("start ok");
        mgr.tick_frame(FRAME_MS); // 40 ms
        let expired = mgr.tick_frame(FRAME_MS); // 80 ms → expire

        assert!(expired.contains(&id));
        assert!(mgr.get_timer(id).expect("found").is_expired());
    }

    #[test]
    fn test_timer_manager_notify_cut_propagates() {
        let mut mgr = ProductionTimerManager::new();
        let id = mgr.add_timer(TimerConfig::segment_remaining("Seg", 3_000, 0));

        mgr.start(id).expect("start ok");
        for _ in 0..25 {
            mgr.tick_frame(FRAME_MS);
        }
        assert_eq!(mgr.get_timer(id).expect("found").elapsed_ms(), 1_000);

        mgr.notify_cut(0);
        assert_eq!(mgr.get_timer(id).expect("found").elapsed_ms(), 0);
    }

    #[test]
    fn test_timer_manager_remove() {
        let mut mgr = ProductionTimerManager::new();
        let id = mgr.add_timer(TimerConfig::elapsed("T"));
        assert_eq!(mgr.timer_count(), 1);

        mgr.remove_timer(id).expect("remove ok");
        assert_eq!(mgr.timer_count(), 0);

        let err = mgr.remove_timer(id).expect_err("double remove");
        assert!(matches!(err, ProductionTimerError::NotFound(_)));
    }

    #[test]
    fn test_remaining_display() {
        let cfg = TimerConfig::countdown("T", 90_000); // 1:30
        let timer = ProductionTimer::new(0, cfg);
        assert_eq!(timer.remaining_display(), Some("01:30".to_string()));
    }

    #[test]
    fn test_elapsed_display() {
        let cfg = TimerConfig::elapsed("T");
        let mut timer = ProductionTimer::new(0, cfg);
        timer.start().expect("start ok");
        for _ in 0..75 {
            // 75 × 40 ms = 3 000 ms = 0:03
            timer.tick_frame(FRAME_MS);
        }
        assert_eq!(timer.elapsed_display(), "00:03");
    }

    #[test]
    fn test_running_timer_ids() {
        let mut mgr = ProductionTimerManager::new();
        let a = mgr.add_timer(TimerConfig::elapsed("A"));
        let b = mgr.add_timer(TimerConfig::elapsed("B"));
        let _c = mgr.add_timer(TimerConfig::elapsed("C"));

        mgr.start(a).expect("start a");
        mgr.start(b).expect("start b");

        let running = mgr.running_timer_ids();
        assert_eq!(running.len(), 2);
        assert!(running.contains(&a));
        assert!(running.contains(&b));
    }
}
