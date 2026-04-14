//! T-bar manual transition control for video switchers.
//!
//! A T-bar (also known as a "fader bar") is a physical or virtual slider that
//! allows an operator to manually drive a transition between the program and
//! preview sources.  Unlike an auto-transition, the T-bar gives the operator
//! full creative control over the timing of the blend.
//!
//! The T-bar position is a value in the range `[0.0, 1.0]`:
//!
//! * `0.0` — the transition has not started; program source is fully visible.
//! * `1.0` — the transition is complete; preview source is now fully visible.
//!
//! When the T-bar reaches 1.0 the transition is considered "committed" and the
//! program/preview buses should be swapped.  If the operator returns the T-bar
//! to 0.0 the transition is cancelled.
//!
//! # Auto-complete
//!
//! If the operator releases the T-bar past a configurable threshold (default
//! 0.95) the transition can automatically snap to completion.  Similarly, if
//! released below a threshold (default 0.05) it snaps back to 0.0.
//!
//! # Example
//!
//! ```rust
//! use oximedia_switcher::tbar_control::{TbarController, TbarConfig};
//!
//! let mut tbar = TbarController::new(TbarConfig::default());
//! tbar.begin_transition(0, 1, 2).expect("begin ok");
//!
//! tbar.set_position(0.5).expect("set ok");
//! assert!((tbar.position() - 0.5).abs() < f32::EPSILON);
//!
//! tbar.set_position(1.0).expect("complete");
//! assert!(tbar.is_committed());
//! ```

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ────────────────────────────────────────────────────────────────────────────
// Errors
// ────────────────────────────────────────────────────────────────────────────

/// Errors from the T-bar controller.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum TbarError {
    /// Position value is outside the valid `[0.0, 1.0]` range.
    #[error("T-bar position {0} is outside the valid range [0.0, 1.0]")]
    PositionOutOfRange(String),

    /// A transition is already in progress.
    #[error("T-bar transition already in progress on M/E row {0}")]
    AlreadyInProgress(usize),

    /// No transition is currently active.
    #[error("No T-bar transition is active")]
    NotActive,

    /// M/E row index is out of range.
    #[error("M/E row {0} is out of range")]
    MeRowOutOfRange(usize),

    /// Program and preview sources are the same.
    #[error("Program and preview sources must differ (both are {0})")]
    SameSources(usize),

    /// Auto-complete threshold is invalid.
    #[error("Auto-complete threshold {0} must be in (0.5, 1.0)")]
    InvalidThreshold(String),
}

// ────────────────────────────────────────────────────────────────────────────
// Configuration
// ────────────────────────────────────────────────────────────────────────────

/// Configuration for the T-bar controller.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TbarConfig {
    /// If the T-bar is released above this position the transition
    /// auto-completes to 1.0.
    pub auto_complete_threshold: f32,
    /// If the T-bar is released below this position the transition
    /// auto-cancels back to 0.0.
    pub auto_cancel_threshold: f32,
    /// Whether auto-complete / auto-cancel is enabled.
    pub auto_snap_enabled: bool,
    /// Smoothing factor applied to position updates (0.0 = no smoothing,
    /// 1.0 = maximum smoothing).  Higher values yield a more "damped" feel.
    pub smoothing: f32,
    /// Deadzone around the current position: movements smaller than this
    /// value are ignored.
    pub deadzone: f32,
}

impl Default for TbarConfig {
    fn default() -> Self {
        Self {
            auto_complete_threshold: 0.95,
            auto_cancel_threshold: 0.05,
            auto_snap_enabled: true,
            smoothing: 0.0,
            deadzone: 0.001,
        }
    }
}

impl TbarConfig {
    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), TbarError> {
        if self.auto_complete_threshold <= 0.5 || self.auto_complete_threshold > 1.0 {
            return Err(TbarError::InvalidThreshold(format!(
                "{}",
                self.auto_complete_threshold
            )));
        }
        if self.auto_cancel_threshold < 0.0 || self.auto_cancel_threshold >= 0.5 {
            return Err(TbarError::InvalidThreshold(format!(
                "{}",
                self.auto_cancel_threshold
            )));
        }
        Ok(())
    }
}

// ────────────────────────────────────────────────────────────────────────────
// State
// ────────────────────────────────────────────────────────────────────────────

/// Current state of the T-bar.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TbarState {
    /// No transition is being driven by the T-bar.
    Idle,
    /// The operator is actively moving the T-bar.
    Active,
    /// The T-bar has reached 1.0 and the transition has been committed
    /// (program/preview swap pending).
    Committed,
    /// The T-bar was returned to 0.0 and the transition was cancelled.
    Cancelled,
}

// ────────────────────────────────────────────────────────────────────────────
// Direction tracking
// ────────────────────────────────────────────────────────────────────────────

/// Direction the T-bar is being moved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TbarDirection {
    /// Moving towards completion (increasing position).
    Forward,
    /// Moving back towards start (decreasing position).
    Reverse,
    /// Not moving.
    Stationary,
}

// ────────────────────────────────────────────────────────────────────────────
// Event
// ────────────────────────────────────────────────────────────────────────────

/// Events emitted by the T-bar controller to inform the switcher of required
/// actions.
#[derive(Debug, Clone, PartialEq)]
pub enum TbarEvent {
    /// The T-bar position changed; the transition engine should update its mix
    /// ratio accordingly.
    PositionChanged { me_row: usize, position: f32 },
    /// The transition is committed (position reached 1.0); the switcher should
    /// swap program and preview on this M/E row.
    TransitionCommitted {
        me_row: usize,
        program_source: usize,
        preview_source: usize,
    },
    /// The transition was cancelled (position returned to 0.0).
    TransitionCancelled { me_row: usize },
}

// ────────────────────────────────────────────────────────────────────────────
// Controller
// ────────────────────────────────────────────────────────────────────────────

/// Controls a single T-bar transition on one M/E row.
#[derive(Debug, Clone)]
pub struct TbarController {
    config: TbarConfig,
    state: TbarState,
    /// Current logical position (0.0 .. 1.0).
    position: f32,
    /// Previous position (for direction detection).
    prev_position: f32,
    /// M/E row this transition acts on.
    me_row: usize,
    /// The source currently on program.
    program_source: usize,
    /// The source currently on preview (destination).
    preview_source: usize,
    /// Accumulated events since last drain.
    events: Vec<TbarEvent>,
}

impl TbarController {
    /// Create a new T-bar controller with the given configuration.
    pub fn new(config: TbarConfig) -> Self {
        Self {
            config,
            state: TbarState::Idle,
            position: 0.0,
            prev_position: 0.0,
            me_row: 0,
            program_source: 0,
            preview_source: 0,
            events: Vec::new(),
        }
    }

    // ── Accessors ───────────────────────────────────────────────────────────

    /// Current position `[0.0, 1.0]`.
    pub fn position(&self) -> f32 {
        self.position
    }

    /// Current state.
    pub fn state(&self) -> TbarState {
        self.state
    }

    /// Whether the transition has been committed (reached 1.0).
    pub fn is_committed(&self) -> bool {
        self.state == TbarState::Committed
    }

    /// Whether the transition was cancelled (returned to 0.0).
    pub fn is_cancelled(&self) -> bool {
        self.state == TbarState::Cancelled
    }

    /// Whether a transition is currently being driven.
    pub fn is_active(&self) -> bool {
        self.state == TbarState::Active
    }

    /// Get the M/E row this controller acts on.
    pub fn me_row(&self) -> usize {
        self.me_row
    }

    /// Get the current direction of travel.
    pub fn direction(&self) -> TbarDirection {
        let delta = self.position - self.prev_position;
        if delta.abs() < self.config.deadzone {
            TbarDirection::Stationary
        } else if delta > 0.0 {
            TbarDirection::Forward
        } else {
            TbarDirection::Reverse
        }
    }

    /// Get the configuration.
    pub fn config(&self) -> &TbarConfig {
        &self.config
    }

    /// Update the configuration.
    pub fn set_config(&mut self, config: TbarConfig) -> Result<(), TbarError> {
        config.validate()?;
        self.config = config;
        Ok(())
    }

    // ── Lifecycle ───────────────────────────────────────────────────────────

    /// Begin a new T-bar transition on the given M/E row.
    pub fn begin_transition(
        &mut self,
        me_row: usize,
        program_source: usize,
        preview_source: usize,
    ) -> Result<(), TbarError> {
        if self.state == TbarState::Active {
            return Err(TbarError::AlreadyInProgress(self.me_row));
        }
        if program_source == preview_source {
            return Err(TbarError::SameSources(program_source));
        }
        self.me_row = me_row;
        self.program_source = program_source;
        self.preview_source = preview_source;
        self.position = 0.0;
        self.prev_position = 0.0;
        self.state = TbarState::Active;
        self.events.clear();
        Ok(())
    }

    /// Set the T-bar position.
    ///
    /// Returns `Ok(())` on success.  The controller may emit events (position
    /// change, commit, cancel) that can be drained via [`Self::drain_events`].
    pub fn set_position(&mut self, raw_position: f32) -> Result<(), TbarError> {
        if self.state != TbarState::Active {
            return Err(TbarError::NotActive);
        }
        if !(0.0..=1.0).contains(&raw_position) {
            return Err(TbarError::PositionOutOfRange(format!("{raw_position}")));
        }

        // Deadzone filter
        let delta = (raw_position - self.position).abs();
        if delta < self.config.deadzone && raw_position != 0.0 && raw_position != 1.0 {
            return Ok(());
        }

        // Apply smoothing
        let new_pos = if self.config.smoothing > f32::EPSILON {
            let alpha = 1.0 - self.config.smoothing.clamp(0.0, 0.99);
            self.position + alpha * (raw_position - self.position)
        } else {
            raw_position
        };

        self.prev_position = self.position;
        self.position = new_pos.clamp(0.0, 1.0);

        // Emit position change event
        self.events.push(TbarEvent::PositionChanged {
            me_row: self.me_row,
            position: self.position,
        });

        // Check for commit
        if self.position >= 1.0 - f32::EPSILON {
            self.position = 1.0;
            self.state = TbarState::Committed;
            self.events.push(TbarEvent::TransitionCommitted {
                me_row: self.me_row,
                program_source: self.program_source,
                preview_source: self.preview_source,
            });
        }
        // Check for cancel
        else if self.position <= f32::EPSILON {
            self.position = 0.0;
            self.state = TbarState::Cancelled;
            self.events.push(TbarEvent::TransitionCancelled {
                me_row: self.me_row,
            });
        }

        Ok(())
    }

    /// Simulate releasing the T-bar at its current position.
    ///
    /// If auto-snap is enabled and the position is above the auto-complete
    /// threshold, the transition is committed.  If below the auto-cancel
    /// threshold, it is cancelled.  Otherwise the position remains where it is.
    pub fn release(&mut self) -> Result<(), TbarError> {
        if self.state != TbarState::Active {
            return Err(TbarError::NotActive);
        }

        if !self.config.auto_snap_enabled {
            return Ok(());
        }

        if self.position >= self.config.auto_complete_threshold {
            self.prev_position = self.position;
            self.position = 1.0;
            self.state = TbarState::Committed;
            self.events.push(TbarEvent::PositionChanged {
                me_row: self.me_row,
                position: 1.0,
            });
            self.events.push(TbarEvent::TransitionCommitted {
                me_row: self.me_row,
                program_source: self.program_source,
                preview_source: self.preview_source,
            });
        } else if self.position <= self.config.auto_cancel_threshold {
            self.prev_position = self.position;
            self.position = 0.0;
            self.state = TbarState::Cancelled;
            self.events.push(TbarEvent::PositionChanged {
                me_row: self.me_row,
                position: 0.0,
            });
            self.events.push(TbarEvent::TransitionCancelled {
                me_row: self.me_row,
            });
        }

        Ok(())
    }

    /// Reset the controller to idle, discarding any in-progress transition.
    pub fn reset(&mut self) {
        self.state = TbarState::Idle;
        self.position = 0.0;
        self.prev_position = 0.0;
        self.events.clear();
    }

    // ── Event draining ──────────────────────────────────────────────────────

    /// Drain all pending events.
    pub fn drain_events(&mut self) -> Vec<TbarEvent> {
        std::mem::take(&mut self.events)
    }

    /// Returns `true` if there are pending events.
    pub fn has_events(&self) -> bool {
        !self.events.is_empty()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Multi-row T-bar manager
// ────────────────────────────────────────────────────────────────────────────

/// Manages T-bar controllers for multiple M/E rows.
pub struct TbarManager {
    controllers: Vec<TbarController>,
}

impl TbarManager {
    /// Create a new manager with one controller per M/E row.
    pub fn new(me_count: usize, config: TbarConfig) -> Self {
        let controllers = (0..me_count)
            .map(|_| TbarController::new(config.clone()))
            .collect();
        Self { controllers }
    }

    /// Get the controller for a specific M/E row.
    pub fn controller(&self, me_row: usize) -> Option<&TbarController> {
        self.controllers.get(me_row)
    }

    /// Get a mutable controller for a specific M/E row.
    pub fn controller_mut(&mut self, me_row: usize) -> Option<&mut TbarController> {
        self.controllers.get_mut(me_row)
    }

    /// Begin a transition on a specific M/E row.
    pub fn begin_transition(
        &mut self,
        me_row: usize,
        program_source: usize,
        preview_source: usize,
    ) -> Result<(), TbarError> {
        let ctrl = self
            .controllers
            .get_mut(me_row)
            .ok_or(TbarError::MeRowOutOfRange(me_row))?;
        ctrl.begin_transition(me_row, program_source, preview_source)
    }

    /// Set the position on a specific M/E row.
    pub fn set_position(&mut self, me_row: usize, position: f32) -> Result<(), TbarError> {
        let ctrl = self
            .controllers
            .get_mut(me_row)
            .ok_or(TbarError::MeRowOutOfRange(me_row))?;
        ctrl.set_position(position)
    }

    /// Release the T-bar on a specific M/E row.
    pub fn release(&mut self, me_row: usize) -> Result<(), TbarError> {
        let ctrl = self
            .controllers
            .get_mut(me_row)
            .ok_or(TbarError::MeRowOutOfRange(me_row))?;
        ctrl.release()
    }

    /// Drain events from all controllers.
    pub fn drain_all_events(&mut self) -> Vec<TbarEvent> {
        self.controllers
            .iter_mut()
            .flat_map(|c| c.drain_events())
            .collect()
    }

    /// Number of M/E rows.
    pub fn me_count(&self) -> usize {
        self.controllers.len()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_tbar() -> TbarController {
        TbarController::new(TbarConfig::default())
    }

    #[test]
    fn test_begin_transition() {
        let mut tbar = default_tbar();
        tbar.begin_transition(0, 1, 2).expect("begin ok");
        assert!(tbar.is_active());
        assert_eq!(tbar.me_row(), 0);
        assert!((tbar.position() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_same_sources_error() {
        let mut tbar = default_tbar();
        let err = tbar.begin_transition(0, 3, 3).expect_err("same source");
        assert!(matches!(err, TbarError::SameSources(3)));
    }

    #[test]
    fn test_already_in_progress_error() {
        let mut tbar = default_tbar();
        tbar.begin_transition(0, 1, 2).expect("ok");
        let err = tbar.begin_transition(0, 3, 4).expect_err("already active");
        assert!(matches!(err, TbarError::AlreadyInProgress(0)));
    }

    #[test]
    fn test_set_position_midway() {
        let mut tbar = default_tbar();
        tbar.begin_transition(0, 1, 2).expect("ok");
        tbar.set_position(0.5).expect("ok");
        assert!((tbar.position() - 0.5).abs() < f32::EPSILON);
        assert!(tbar.is_active());
    }

    #[test]
    fn test_set_position_to_one_commits() {
        let mut tbar = default_tbar();
        tbar.begin_transition(0, 1, 2).expect("ok");
        tbar.set_position(1.0).expect("ok");
        assert!(tbar.is_committed());
        assert!((tbar.position() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_set_position_to_zero_cancels() {
        let mut tbar = default_tbar();
        tbar.begin_transition(0, 1, 2).expect("ok");
        tbar.set_position(0.5).expect("ok");
        tbar.set_position(0.0).expect("ok");
        assert!(tbar.is_cancelled());
    }

    #[test]
    fn test_position_out_of_range_error() {
        let mut tbar = default_tbar();
        tbar.begin_transition(0, 1, 2).expect("ok");
        let err = tbar.set_position(1.5).expect_err("out of range");
        assert!(matches!(err, TbarError::PositionOutOfRange(_)));
        let err2 = tbar.set_position(-0.1).expect_err("negative");
        assert!(matches!(err2, TbarError::PositionOutOfRange(_)));
    }

    #[test]
    fn test_not_active_set_position_error() {
        let mut tbar = default_tbar();
        let err = tbar.set_position(0.5).expect_err("not active");
        assert!(matches!(err, TbarError::NotActive));
    }

    #[test]
    fn test_release_auto_complete() {
        let mut tbar = default_tbar();
        tbar.begin_transition(0, 1, 2).expect("ok");
        tbar.set_position(0.96).expect("ok");
        tbar.release().expect("ok");
        assert!(tbar.is_committed());
        assert!((tbar.position() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_release_auto_cancel() {
        let mut tbar = default_tbar();
        tbar.begin_transition(0, 1, 2).expect("ok");
        tbar.set_position(0.03).expect("ok");
        tbar.release().expect("ok");
        assert!(tbar.is_cancelled());
        assert!((tbar.position() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_release_no_snap_in_middle() {
        let mut tbar = default_tbar();
        tbar.begin_transition(0, 1, 2).expect("ok");
        tbar.set_position(0.5).expect("ok");
        tbar.release().expect("ok");
        // Position stays at 0.5 — no auto-snap.
        assert!(tbar.is_active());
        assert!((tbar.position() - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_release_auto_snap_disabled() {
        let mut config = TbarConfig::default();
        config.auto_snap_enabled = false;
        let mut tbar = TbarController::new(config);
        tbar.begin_transition(0, 1, 2).expect("ok");
        tbar.set_position(0.98).expect("ok");
        tbar.release().expect("ok");
        // Should NOT auto-complete.
        assert!(tbar.is_active());
    }

    #[test]
    fn test_direction_detection() {
        let mut tbar = default_tbar();
        tbar.begin_transition(0, 1, 2).expect("ok");
        assert_eq!(tbar.direction(), TbarDirection::Stationary);
        tbar.set_position(0.5).expect("ok");
        assert_eq!(tbar.direction(), TbarDirection::Forward);
        tbar.set_position(0.3).expect("ok");
        assert_eq!(tbar.direction(), TbarDirection::Reverse);
    }

    #[test]
    fn test_events_emitted() {
        let mut tbar = default_tbar();
        tbar.begin_transition(0, 1, 2).expect("ok");
        tbar.set_position(0.5).expect("ok");
        let events = tbar.drain_events();
        assert!(!events.is_empty());
        assert!(events
            .iter()
            .any(|e| matches!(e, TbarEvent::PositionChanged { me_row: 0, .. })));
    }

    #[test]
    fn test_commit_event_emitted() {
        let mut tbar = default_tbar();
        tbar.begin_transition(0, 1, 2).expect("ok");
        tbar.set_position(1.0).expect("ok");
        let events = tbar.drain_events();
        assert!(events
            .iter()
            .any(|e| matches!(e, TbarEvent::TransitionCommitted { me_row: 0, .. })));
    }

    #[test]
    fn test_cancel_event_emitted() {
        let mut tbar = default_tbar();
        tbar.begin_transition(0, 1, 2).expect("ok");
        tbar.set_position(0.5).expect("ok");
        tbar.drain_events();
        tbar.set_position(0.0).expect("ok");
        let events = tbar.drain_events();
        assert!(events
            .iter()
            .any(|e| matches!(e, TbarEvent::TransitionCancelled { me_row: 0 })));
    }

    #[test]
    fn test_reset() {
        let mut tbar = default_tbar();
        tbar.begin_transition(0, 1, 2).expect("ok");
        tbar.set_position(0.5).expect("ok");
        tbar.reset();
        assert_eq!(tbar.state(), TbarState::Idle);
        assert!((tbar.position() - 0.0).abs() < f32::EPSILON);
        assert!(!tbar.has_events());
    }

    #[test]
    fn test_tbar_config_validate() {
        let mut cfg = TbarConfig::default();
        cfg.validate().expect("default is valid");

        cfg.auto_complete_threshold = 0.3;
        assert!(cfg.validate().is_err());

        cfg.auto_complete_threshold = 0.95;
        cfg.auto_cancel_threshold = 0.6;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_tbar_manager_basic() {
        let mut mgr = TbarManager::new(2, TbarConfig::default());
        assert_eq!(mgr.me_count(), 2);
        mgr.begin_transition(0, 1, 2).expect("ok");
        mgr.set_position(0, 0.5).expect("ok");
        let ctrl = mgr.controller(0).expect("exists");
        assert!((ctrl.position() - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_tbar_manager_me_out_of_range() {
        let mut mgr = TbarManager::new(1, TbarConfig::default());
        let err = mgr.begin_transition(5, 1, 2).expect_err("out of range");
        assert!(matches!(err, TbarError::MeRowOutOfRange(5)));
    }

    #[test]
    fn test_tbar_manager_drain_all_events() {
        let mut mgr = TbarManager::new(2, TbarConfig::default());
        mgr.begin_transition(0, 1, 2).expect("ok");
        mgr.begin_transition(1, 3, 4).expect("ok");
        mgr.set_position(0, 0.5).expect("ok");
        mgr.set_position(1, 0.7).expect("ok");
        let events = mgr.drain_all_events();
        assert!(events.len() >= 2);
    }

    #[test]
    fn test_tbar_manager_release() {
        let mut mgr = TbarManager::new(1, TbarConfig::default());
        mgr.begin_transition(0, 1, 2).expect("ok");
        mgr.set_position(0, 0.96).expect("ok");
        mgr.release(0).expect("ok");
        let ctrl = mgr.controller(0).expect("exists");
        assert!(ctrl.is_committed());
    }

    #[test]
    fn test_smoothing_applied() {
        let mut config = TbarConfig::default();
        config.smoothing = 0.5;
        let mut tbar = TbarController::new(config);
        tbar.begin_transition(0, 1, 2).expect("ok");
        tbar.set_position(1.0).expect("ok");
        // With smoothing, position should not jump directly to 1.0
        // (unless the smoothed value rounds to >= 1.0 - EPSILON).
        // With alpha = 0.5, new_pos = 0.0 + 0.5 * 1.0 = 0.5
        assert!(tbar.position() < 0.9);
    }
}
