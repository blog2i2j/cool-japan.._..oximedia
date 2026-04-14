//! Real-time cursor and viewport sharing for collaborative sessions.
//!
//! Tracks every participant's cursor position and viewport in the timeline,
//! propagates updates as [`CursorEvent`]s, and smoothly interpolates between
//! samples so UI renderers can produce fluid motion without waiting for the
//! next network packet.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Core value types
// ─────────────────────────────────────────────────────────────────────────────

/// A floating-point timeline cursor position.
///
/// Using `f64` rather than an integer frame index lets us represent
/// sub-frame positions during smooth interpolation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CursorPosition {
    /// Frame position (may be fractional during interpolation).
    pub frame: f64,
    /// Track index (may be fractional during interpolation).
    pub track: f64,
}

impl CursorPosition {
    /// Create a new cursor position.
    pub fn new(frame: f64, track: f64) -> Self {
        Self { frame, track }
    }

    /// Create from integer coordinates.
    pub fn from_ints(frame: u64, track: u32) -> Self {
        Self {
            frame: frame as f64,
            track: track as f64,
        }
    }

    /// Linear interpolation between `self` and `other` by factor `t` ∈ [0, 1].
    #[must_use]
    pub fn lerp(self, other: Self, t: f64) -> Self {
        let t = t.clamp(0.0, 1.0);
        Self {
            frame: self.frame + (other.frame - self.frame) * t,
            track: self.track + (other.track - self.track) * t,
        }
    }

    /// Euclidean distance to another position (frame-space units).
    #[must_use]
    pub fn distance(self, other: Self) -> f64 {
        let df = self.frame - other.frame;
        let dt = self.track - other.track;
        (df * df + dt * dt).sqrt()
    }
}

impl std::fmt::Display for CursorPosition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({:.2}f, {:.2}t)", self.frame, self.track)
    }
}

/// A viewport window on the timeline.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Viewport {
    /// Left edge in frames (inclusive).
    pub start_frame: f64,
    /// Right edge in frames (exclusive).
    pub end_frame: f64,
    /// Zoom level (1.0 = default, >1.0 = zoomed in).
    pub zoom: f64,
}

impl Viewport {
    /// Create a new viewport.
    pub fn new(start_frame: f64, end_frame: f64, zoom: f64) -> Self {
        Self {
            start_frame,
            end_frame,
            zoom,
        }
    }

    /// Width of the viewport in frames.
    #[must_use]
    pub fn width(&self) -> f64 {
        (self.end_frame - self.start_frame).max(0.0)
    }

    /// Check whether a frame is visible in this viewport.
    #[must_use]
    pub fn contains(&self, frame: f64) -> bool {
        frame >= self.start_frame && frame < self.end_frame
    }

    /// Linear interpolation to another viewport.
    #[must_use]
    pub fn lerp(self, other: Self, t: f64) -> Self {
        let t = t.clamp(0.0, 1.0);
        Self {
            start_frame: self.start_frame + (other.start_frame - self.start_frame) * t,
            end_frame: self.end_frame + (other.end_frame - self.end_frame) * t,
            zoom: self.zoom + (other.zoom - self.zoom) * t,
        }
    }

    /// Whether this viewport overlaps another.
    #[must_use]
    pub fn overlaps(&self, other: &Self) -> bool {
        self.start_frame < other.end_frame && other.start_frame < self.end_frame
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CursorState
// ─────────────────────────────────────────────────────────────────────────────

/// Full cursor + viewport state for a single participant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CursorState {
    /// User identifier.
    pub user_id: String,
    /// Display name shown in the UI.
    pub display_name: String,
    /// Hex colour string (e.g. `"#FF6B6B"`).
    pub color: String,
    /// Current cursor position.
    pub position: CursorPosition,
    /// Current viewport.
    pub viewport: Viewport,
    /// Whether the user is actively scrubbing / dragging.
    pub is_dragging: bool,
    /// Wall-clock timestamp of the last update (milliseconds since Unix epoch).
    pub last_updated_ms: u64,
}

impl CursorState {
    /// Create a new cursor state at the origin.
    pub fn new(
        user_id: impl Into<String>,
        display_name: impl Into<String>,
        color: impl Into<String>,
        now_ms: u64,
    ) -> Self {
        Self {
            user_id: user_id.into(),
            display_name: display_name.into(),
            color: color.into(),
            position: CursorPosition::new(0.0, 0.0),
            viewport: Viewport::new(0.0, 1000.0, 1.0),
            is_dragging: false,
            last_updated_ms: now_ms,
        }
    }

    /// Milliseconds elapsed since the last update (relative to `now_ms`).
    #[must_use]
    pub fn age_ms(&self, now_ms: u64) -> u64 {
        now_ms.saturating_sub(self.last_updated_ms)
    }

    /// Whether this state is considered stale (older than `threshold_ms`).
    #[must_use]
    pub fn is_stale(&self, now_ms: u64, threshold_ms: u64) -> bool {
        self.age_ms(now_ms) >= threshold_ms
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CursorEvent
// ─────────────────────────────────────────────────────────────────────────────

/// An event that describes a change in a user's cursor or viewport state.
/// These events are designed to be sent over the network (WebSocket / CRDT
/// broadcast) and applied by all peers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CursorEvent {
    /// The user who generated the event.
    pub user_id: String,
    /// Payload.
    pub kind: CursorEventKind,
    /// Wall-clock timestamp at event creation (milliseconds since Unix epoch).
    pub timestamp_ms: u64,
}

/// Discriminated payload for [`CursorEvent`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CursorEventKind {
    /// Cursor moved to a new position.
    Moved {
        /// New cursor position.
        position: CursorPosition,
    },
    /// Viewport scrolled or zoomed.
    ViewportChanged {
        /// New viewport.
        viewport: Viewport,
    },
    /// Drag started.
    DragStarted {
        /// Position where the drag began.
        position: CursorPosition,
    },
    /// Drag ended.
    DragEnded {
        /// Final position of the drag.
        position: CursorPosition,
    },
    /// User went idle (no cursor events for a while).
    Idle,
    /// User disconnected / left the session.
    Disconnected,
}

impl CursorEvent {
    /// Create a "cursor moved" event.
    pub fn moved(user_id: impl Into<String>, position: CursorPosition, timestamp_ms: u64) -> Self {
        Self {
            user_id: user_id.into(),
            kind: CursorEventKind::Moved { position },
            timestamp_ms,
        }
    }

    /// Create a "viewport changed" event.
    pub fn viewport_changed(
        user_id: impl Into<String>,
        viewport: Viewport,
        timestamp_ms: u64,
    ) -> Self {
        Self {
            user_id: user_id.into(),
            kind: CursorEventKind::ViewportChanged { viewport },
            timestamp_ms,
        }
    }

    /// Create a "disconnected" event.
    pub fn disconnected(user_id: impl Into<String>, timestamp_ms: u64) -> Self {
        Self {
            user_id: user_id.into(),
            kind: CursorEventKind::Disconnected,
            timestamp_ms,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// InterpolatedCursor – smooth rendering helper
// ─────────────────────────────────────────────────────────────────────────────

/// A pair of (from, to) snapshots used to smoothly interpolate a cursor's
/// rendered position without blocking on the next network update.
#[derive(Debug, Clone)]
pub struct InterpolatedCursor {
    /// Previous confirmed state (interpolation origin).
    pub from: CursorState,
    /// Most recent confirmed state (interpolation target).
    pub to: CursorState,
    /// Lerp factor in [0, 1].  0 = at `from`, 1 = at `to`.
    pub t: f64,
}

impl InterpolatedCursor {
    /// Create a new interpolated cursor from an initial state.
    pub fn new(initial: CursorState) -> Self {
        Self {
            from: initial.clone(),
            to: initial,
            t: 1.0, // already at target
        }
    }

    /// Update the target state and reset the interpolation factor.
    pub fn update_target(&mut self, new_state: CursorState) {
        self.from = CursorState {
            position: self.current_position(),
            viewport: self.current_viewport(),
            ..self.to.clone()
        };
        self.to = new_state;
        self.t = 0.0;
    }

    /// Advance interpolation by `delta_t` (should be in [0, 1] per-frame).
    pub fn advance(&mut self, delta_t: f64) {
        self.t = (self.t + delta_t).min(1.0);
    }

    /// Compute the currently interpolated cursor position.
    #[must_use]
    pub fn current_position(&self) -> CursorPosition {
        self.from.position.lerp(self.to.position, self.t)
    }

    /// Compute the currently interpolated viewport.
    #[must_use]
    pub fn current_viewport(&self) -> Viewport {
        self.from.viewport.lerp(self.to.viewport, self.t)
    }

    /// Whether this cursor has fully reached its target.
    #[must_use]
    pub fn is_settled(&self) -> bool {
        (self.t - 1.0).abs() < f64::EPSILON
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CursorMap – the central registry
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the cursor map.
#[derive(Debug, Clone)]
pub struct CursorMapConfig {
    /// Time (ms) after which a user is considered idle if no update is received.
    pub idle_threshold_ms: u64,
    /// Time (ms) after which a stale cursor is removed from the map.
    pub expiry_threshold_ms: u64,
    /// Interpolation step size per rendering frame.
    pub lerp_step: f64,
}

impl Default for CursorMapConfig {
    fn default() -> Self {
        Self {
            idle_threshold_ms: 5_000,
            expiry_threshold_ms: 30_000,
            lerp_step: 0.15,
        }
    }
}

/// Tracks all participants' cursor states and handles smooth interpolation.
///
/// This is the single source of truth for cursor data within a session.
/// It accepts [`CursorEvent`]s (from the network or local user) and exposes
/// interpolated snapshots for rendering.
#[derive(Debug)]
pub struct CursorMap {
    config: CursorMapConfig,
    /// Raw (authoritative) cursor states keyed by `user_id`.
    states: HashMap<String, CursorState>,
    /// Interpolated rendering state keyed by `user_id`.
    interpolated: HashMap<String, InterpolatedCursor>,
    /// Monotonic clock reference for idle/expiry checks.
    /// We keep track of the last `now_ms` provided from the outside so we
    /// avoid calling `Instant::now()` internally (deterministic in tests).
    last_now_ms: u64,
}

impl CursorMap {
    /// Create a new cursor map.
    pub fn new(config: CursorMapConfig) -> Self {
        Self {
            config,
            states: HashMap::new(),
            interpolated: HashMap::new(),
            last_now_ms: 0,
        }
    }

    /// Apply a [`CursorEvent`] received from the network or generated locally.
    pub fn apply_event(&mut self, event: CursorEvent) {
        let now_ms = event.timestamp_ms;
        self.last_now_ms = self.last_now_ms.max(now_ms);

        match &event.kind {
            CursorEventKind::Disconnected => {
                self.remove(&event.user_id);
                return;
            }
            CursorEventKind::Idle => {
                // Just update timestamp so expiry is deferred.
                if let Some(state) = self.states.get_mut(&event.user_id) {
                    state.last_updated_ms = now_ms;
                }
                return;
            }
            _ => {}
        }

        let state = self
            .states
            .entry(event.user_id.clone())
            .or_insert_with(|| CursorState::new(&event.user_id, &event.user_id, "#AAAAAA", now_ms));

        // Apply the event payload.
        match event.kind {
            CursorEventKind::Moved { position } => {
                state.position = position;
            }
            CursorEventKind::ViewportChanged { viewport } => {
                state.viewport = viewport;
            }
            CursorEventKind::DragStarted { position } => {
                state.position = position;
                state.is_dragging = true;
            }
            CursorEventKind::DragEnded { position } => {
                state.position = position;
                state.is_dragging = false;
            }
            CursorEventKind::Idle | CursorEventKind::Disconnected => {
                // Already handled above.
            }
        }
        state.last_updated_ms = now_ms;

        let updated_state = state.clone();

        // Update interpolation target.
        let interp = self
            .interpolated
            .entry(event.user_id.clone())
            .or_insert_with(|| InterpolatedCursor::new(updated_state.clone()));
        interp.update_target(updated_state);
    }

    /// Register a user with an explicit display name and colour.
    ///
    /// Call this when a user joins so their name/colour are available before
    /// the first move event arrives.
    pub fn register(
        &mut self,
        user_id: impl Into<String>,
        display_name: impl Into<String>,
        color: impl Into<String>,
        now_ms: u64,
    ) {
        let uid = user_id.into();
        let state = CursorState::new(&uid, display_name, color, now_ms);
        self.interpolated
            .entry(uid.clone())
            .or_insert_with(|| InterpolatedCursor::new(state.clone()));
        self.states.entry(uid).or_insert(state);
    }

    /// Remove a user from the map.
    pub fn remove(&mut self, user_id: &str) {
        self.states.remove(user_id);
        self.interpolated.remove(user_id);
    }

    /// Advance interpolation for all cursors by one rendering step.
    ///
    /// Call this once per animation frame.
    pub fn tick(&mut self) {
        let step = self.config.lerp_step;
        for interp in self.interpolated.values_mut() {
            interp.advance(step);
        }
    }

    /// Remove cursors that have not been updated within the expiry window.
    ///
    /// Returns the `user_id`s that were removed.
    pub fn expire_stale(&mut self, now_ms: u64) -> Vec<String> {
        let threshold = self.config.expiry_threshold_ms;
        let expired: Vec<String> = self
            .states
            .iter()
            .filter(|(_, s)| s.is_stale(now_ms, threshold))
            .map(|(id, _)| id.clone())
            .collect();
        for id in &expired {
            self.remove(id);
        }
        expired
    }

    /// Get the raw (authoritative) state for a user.
    #[must_use]
    pub fn get_state(&self, user_id: &str) -> Option<&CursorState> {
        self.states.get(user_id)
    }

    /// Get the interpolated rendering state for a user.
    #[must_use]
    pub fn get_interpolated(&self, user_id: &str) -> Option<&InterpolatedCursor> {
        self.interpolated.get(user_id)
    }

    /// Number of tracked users.
    #[must_use]
    pub fn count(&self) -> usize {
        self.states.len()
    }

    /// All tracked user IDs.
    #[must_use]
    pub fn user_ids(&self) -> Vec<String> {
        self.states.keys().cloned().collect()
    }

    /// All users whose cursor is within `radius` frames of `frame`.
    #[must_use]
    pub fn users_near_frame(&self, frame: f64, radius: f64) -> Vec<&CursorState> {
        self.states
            .values()
            .filter(|s| (s.position.frame - frame).abs() <= radius)
            .collect()
    }

    /// All users whose viewport overlaps the given range.
    #[must_use]
    pub fn users_viewing(&self, start_frame: f64, end_frame: f64) -> Vec<&CursorState> {
        let query = Viewport::new(start_frame, end_frame, 1.0);
        self.states
            .values()
            .filter(|s| s.viewport.overlaps(&query))
            .collect()
    }

    /// All users currently dragging.
    #[must_use]
    pub fn dragging_users(&self) -> Vec<&CursorState> {
        self.states.values().filter(|s| s.is_dragging).collect()
    }

    /// All users considered idle (no update for `idle_threshold_ms`).
    #[must_use]
    pub fn idle_users(&self, now_ms: u64) -> Vec<&CursorState> {
        let threshold = self.config.idle_threshold_ms;
        self.states
            .values()
            .filter(|s| s.is_stale(now_ms, threshold))
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(v: u64) -> u64 {
        v * 1_000
    }

    // ── CursorPosition ───────────────────────────────────────────────────────

    #[test]
    fn test_cursor_position_lerp_at_extremes() {
        let a = CursorPosition::new(0.0, 0.0);
        let b = CursorPosition::new(100.0, 4.0);

        let at0 = a.lerp(b, 0.0);
        assert_eq!(at0.frame, 0.0);
        assert_eq!(at0.track, 0.0);

        let at1 = a.lerp(b, 1.0);
        assert_eq!(at1.frame, 100.0);
        assert_eq!(at1.track, 4.0);
    }

    #[test]
    fn test_cursor_position_lerp_midpoint() {
        let a = CursorPosition::new(0.0, 0.0);
        let b = CursorPosition::new(200.0, 8.0);
        let mid = a.lerp(b, 0.5);
        assert!((mid.frame - 100.0).abs() < 1e-9);
        assert!((mid.track - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_cursor_position_lerp_clamps_t() {
        let a = CursorPosition::new(0.0, 0.0);
        let b = CursorPosition::new(100.0, 0.0);
        // t > 1 should clamp to 1
        let over = a.lerp(b, 2.0);
        assert_eq!(over.frame, 100.0);
        // t < 0 should clamp to 0
        let under = a.lerp(b, -1.0);
        assert_eq!(under.frame, 0.0);
    }

    #[test]
    fn test_cursor_position_distance() {
        let a = CursorPosition::new(0.0, 0.0);
        let b = CursorPosition::new(3.0, 4.0);
        assert!((a.distance(b) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_cursor_position_display() {
        let p = CursorPosition::new(1.5, 2.25);
        assert_eq!(p.to_string(), "(1.50f, 2.25t)");
    }

    // ── Viewport ─────────────────────────────────────────────────────────────

    #[test]
    fn test_viewport_width() {
        let vp = Viewport::new(100.0, 600.0, 1.0);
        assert!((vp.width() - 500.0).abs() < 1e-9);
    }

    #[test]
    fn test_viewport_contains() {
        let vp = Viewport::new(100.0, 500.0, 1.0);
        assert!(vp.contains(100.0));
        assert!(vp.contains(499.9));
        assert!(!vp.contains(500.0));
        assert!(!vp.contains(50.0));
    }

    #[test]
    fn test_viewport_overlaps() {
        let a = Viewport::new(0.0, 100.0, 1.0);
        let b = Viewport::new(50.0, 150.0, 1.0);
        let c = Viewport::new(100.0, 200.0, 1.0);
        assert!(a.overlaps(&b));
        assert!(!a.overlaps(&c));
    }

    #[test]
    fn test_viewport_lerp() {
        let a = Viewport::new(0.0, 1000.0, 1.0);
        let b = Viewport::new(500.0, 1500.0, 2.0);
        let mid = a.lerp(b, 0.5);
        assert!((mid.start_frame - 250.0).abs() < 1e-9);
        assert!((mid.end_frame - 1250.0).abs() < 1e-9);
        assert!((mid.zoom - 1.5).abs() < 1e-9);
    }

    // ── CursorEvent application ───────────────────────────────────────────────

    #[test]
    fn test_apply_moved_event() {
        let mut map = CursorMap::new(CursorMapConfig::default());
        map.register("alice", "Alice", "#FF0000", ms(0));

        let ev = CursorEvent::moved("alice", CursorPosition::new(300.0, 2.0), ms(1));
        map.apply_event(ev);

        let state = map.get_state("alice").expect("state should exist");
        assert!((state.position.frame - 300.0).abs() < 1e-9);
        assert!((state.position.track - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_apply_viewport_changed_event() {
        let mut map = CursorMap::new(CursorMapConfig::default());
        map.register("bob", "Bob", "#00FF00", ms(0));

        let new_vp = Viewport::new(1000.0, 2000.0, 2.0);
        let ev = CursorEvent::viewport_changed("bob", new_vp, ms(1));
        map.apply_event(ev);

        let state = map.get_state("bob").expect("state should exist");
        assert!((state.viewport.start_frame - 1000.0).abs() < 1e-9);
        assert!((state.viewport.zoom - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_apply_disconnect_removes_user() {
        let mut map = CursorMap::new(CursorMapConfig::default());
        map.register("charlie", "Charlie", "#0000FF", ms(0));
        assert_eq!(map.count(), 1);

        let ev = CursorEvent::disconnected("charlie", ms(1));
        map.apply_event(ev);
        assert_eq!(map.count(), 0);
    }

    // ── Drag state ───────────────────────────────────────────────────────────

    #[test]
    fn test_drag_started_sets_dragging_flag() {
        let mut map = CursorMap::new(CursorMapConfig::default());
        map.register("u1", "U1", "#111111", ms(0));

        let ev = CursorEvent {
            user_id: "u1".to_string(),
            kind: CursorEventKind::DragStarted {
                position: CursorPosition::new(50.0, 0.0),
            },
            timestamp_ms: ms(1),
        };
        map.apply_event(ev);
        assert!(map.get_state("u1").map(|s| s.is_dragging).unwrap_or(false));
    }

    #[test]
    fn test_drag_ended_clears_dragging_flag() {
        let mut map = CursorMap::new(CursorMapConfig::default());
        map.register("u2", "U2", "#222222", ms(0));

        map.apply_event(CursorEvent {
            user_id: "u2".to_string(),
            kind: CursorEventKind::DragStarted {
                position: CursorPosition::new(10.0, 1.0),
            },
            timestamp_ms: ms(1),
        });
        map.apply_event(CursorEvent {
            user_id: "u2".to_string(),
            kind: CursorEventKind::DragEnded {
                position: CursorPosition::new(200.0, 1.0),
            },
            timestamp_ms: ms(2),
        });

        let state = map.get_state("u2").expect("state should exist");
        assert!(!state.is_dragging);
        assert!((state.position.frame - 200.0).abs() < 1e-9);
    }

    // ── Interpolation ─────────────────────────────────────────────────────────

    #[test]
    fn test_interpolation_starts_at_zero_after_update() {
        let mut map = CursorMap::new(CursorMapConfig::default());
        map.register("ix", "IX", "#CCCCCC", ms(0));

        let ev = CursorEvent::moved("ix", CursorPosition::new(500.0, 0.0), ms(1));
        map.apply_event(ev);

        let interp = map.get_interpolated("ix").expect("interp should exist");
        // t should be 0 right after a new target is set
        assert_eq!(interp.t, 0.0);
        // current position should still reflect from state (near 0)
        assert!(interp.current_position().frame < 500.0);
    }

    #[test]
    fn test_tick_advances_interpolation() {
        let mut map = CursorMap::new(CursorMapConfig {
            lerp_step: 0.5,
            ..Default::default()
        });
        map.register("iy", "IY", "#BBBBBB", ms(0));
        map.apply_event(CursorEvent::moved(
            "iy",
            CursorPosition::new(100.0, 0.0),
            ms(1),
        ));

        map.tick();
        let interp = map.get_interpolated("iy").expect("interp should exist");
        assert!((interp.t - 0.5).abs() < 1e-9);
        // position should be halfway
        assert!((interp.current_position().frame - 50.0).abs() < 1e-9);
    }

    // ── Expiry ───────────────────────────────────────────────────────────────

    #[test]
    fn test_expire_stale_removes_old_cursors() {
        let mut map = CursorMap::new(CursorMapConfig {
            expiry_threshold_ms: 5_000,
            ..Default::default()
        });
        map.register("old", "Old", "#000000", 0);
        map.register("new", "New", "#FFFFFF", ms(10));

        // Expire at t=10s; "old" was last updated at 0 (age=10s ≥ 5s threshold)
        let removed = map.expire_stale(ms(10));
        assert_eq!(removed.len(), 1);
        assert_eq!(removed[0], "old");
        assert_eq!(map.count(), 1);
    }

    // ── Spatial queries ───────────────────────────────────────────────────────

    #[test]
    fn test_users_near_frame() {
        let mut map = CursorMap::new(CursorMapConfig::default());
        map.register("a", "A", "#1", ms(0));
        map.register("b", "B", "#2", ms(0));
        map.register("c", "C", "#3", ms(0));

        map.apply_event(CursorEvent::moved("a", CursorPosition::new(100.0, 0.0), ms(1)));
        map.apply_event(CursorEvent::moved("b", CursorPosition::new(110.0, 0.0), ms(1)));
        map.apply_event(CursorEvent::moved("c", CursorPosition::new(500.0, 0.0), ms(1)));

        let near = map.users_near_frame(105.0, 20.0);
        assert_eq!(near.len(), 2);
    }

    #[test]
    fn test_users_viewing_viewport_range() {
        let mut map = CursorMap::new(CursorMapConfig::default());
        map.register("p", "P", "#P", ms(0));
        map.register("q", "Q", "#Q", ms(0));

        // p viewport: 0..500; q viewport: 1000..2000
        map.apply_event(CursorEvent {
            user_id: "p".to_string(),
            kind: CursorEventKind::ViewportChanged {
                viewport: Viewport::new(0.0, 500.0, 1.0),
            },
            timestamp_ms: ms(1),
        });
        map.apply_event(CursorEvent {
            user_id: "q".to_string(),
            kind: CursorEventKind::ViewportChanged {
                viewport: Viewport::new(1000.0, 2000.0, 1.0),
            },
            timestamp_ms: ms(1),
        });

        // Query 400..1200 overlaps both
        let viewing = map.users_viewing(400.0, 1200.0);
        assert_eq!(viewing.len(), 2);

        // Query 600..900 overlaps neither
        let empty = map.users_viewing(600.0, 900.0);
        assert_eq!(empty.len(), 0);
    }
}
