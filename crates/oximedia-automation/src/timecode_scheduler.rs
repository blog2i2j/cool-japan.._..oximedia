//! Frame-accurate timecode-based scheduler for broadcast automation.
//!
//! This module provides a broadcast timecode scheduler that fires automation
//! actions at precise timecodes. Timecodes are represented as absolute frame
//! counts, enabling sub-frame-accurate scheduling at any frame rate.
//!
//! # Design
//!
//! Actions are registered with a target timecode (frame number). The scheduler
//! maintains a sorted timeline of pending actions. Each call to [`TimecodeScheduler::advance`]
//! provides the current frame position, and the scheduler fires all actions
//! whose target timecode has been reached or passed.
//!
//! Pre-roll offsets allow actions to fire a configurable number of frames before
//! their nominal cue point, enabling device pre-cuing and network latency compensation.
//!
//! # Example
//!
//! ```rust
//! use oximedia_automation::timecode_scheduler::{
//!     TimecodeScheduler, ScheduledAction, TimecodeActionKind,
//! };
//!
//! let mut scheduler = TimecodeScheduler::new(30);
//! scheduler.schedule(ScheduledAction::new(
//!     "cue-001",
//!     900, // fire at frame 900 (30 seconds at 30fps)
//!     TimecodeActionKind::PlayClip { clip_id: "news_intro".to_string() },
//! ));
//!
//! // Advance to frame 900 — fires the clip play action.
//! let fired = scheduler.advance(900);
//! assert_eq!(fired.len(), 1);
//! ```

use std::collections::BinaryHeap;
use std::cmp::Ordering;

// ---------------------------------------------------------------------------
// TimecodeActionKind
// ---------------------------------------------------------------------------

/// The kind of action to be performed at a scheduled timecode.
#[derive(Debug, Clone, PartialEq)]
pub enum TimecodeActionKind {
    /// Start playing a clip with the given identifier.
    PlayClip {
        /// Clip identifier to play.
        clip_id: String,
    },
    /// Stop playout on the current item.
    StopPlayout,
    /// Cut to a different source.
    CutToSource {
        /// Target source identifier.
        source_id: String,
    },
    /// Insert a graphic overlay.
    InsertGraphic {
        /// Graphic template identifier.
        template_id: String,
        /// Optional duration in frames (indefinite if `None`).
        duration_frames: Option<u64>,
    },
    /// Remove a previously inserted graphic.
    RemoveGraphic {
        /// Graphic template identifier to remove.
        template_id: String,
    },
    /// Start an ad break of a given duration.
    StartAdBreak {
        /// Total ad break duration in frames.
        duration_frames: u64,
    },
    /// End the current ad break.
    EndAdBreak,
    /// Activate an EAS (Emergency Alert System) alert.
    EasAlert {
        /// Alert identifier.
        alert_id: String,
    },
    /// Execute a Lua macro by name.
    ExecuteMacro {
        /// Macro name.
        macro_name: String,
    },
    /// Custom action identified by an arbitrary string key and payload.
    Custom {
        /// Action key.
        key: String,
        /// Payload data.
        payload: String,
    },
}

impl TimecodeActionKind {
    /// Returns a short descriptive label for this action kind.
    pub fn label(&self) -> &str {
        match self {
            Self::PlayClip { .. } => "PlayClip",
            Self::StopPlayout => "StopPlayout",
            Self::CutToSource { .. } => "CutToSource",
            Self::InsertGraphic { .. } => "InsertGraphic",
            Self::RemoveGraphic { .. } => "RemoveGraphic",
            Self::StartAdBreak { .. } => "StartAdBreak",
            Self::EndAdBreak => "EndAdBreak",
            Self::EasAlert { .. } => "EasAlert",
            Self::ExecuteMacro { .. } => "ExecuteMacro",
            Self::Custom { .. } => "Custom",
        }
    }
}

// ---------------------------------------------------------------------------
// ScheduledAction
// ---------------------------------------------------------------------------

/// An automation action registered with a specific target timecode.
#[derive(Debug, Clone)]
pub struct ScheduledAction {
    /// Unique action identifier.
    pub id: String,
    /// Target timecode expressed as an absolute frame number.
    pub target_frame: u64,
    /// Number of frames before the nominal cue to fire (pre-roll offset).
    ///
    /// If `pre_roll_frames` is 5 and `target_frame` is 100, the action fires
    /// when the scheduler reaches frame 95.
    pub pre_roll_frames: u64,
    /// The kind of action to perform.
    pub kind: TimecodeActionKind,
    /// Whether this action has already been fired during this playback pass.
    fired: bool,
}

impl ScheduledAction {
    /// Create a new scheduled action with no pre-roll.
    pub fn new(id: impl Into<String>, target_frame: u64, kind: TimecodeActionKind) -> Self {
        Self {
            id: id.into(),
            target_frame,
            pre_roll_frames: 0,
            kind,
            fired: false,
        }
    }

    /// Builder: set a pre-roll offset in frames.
    pub fn with_pre_roll(mut self, frames: u64) -> Self {
        self.pre_roll_frames = frames;
        self
    }

    /// Effective frame at which this action will be triggered.
    ///
    /// This is `target_frame.saturating_sub(pre_roll_frames)`.
    pub fn trigger_frame(&self) -> u64 {
        self.target_frame.saturating_sub(self.pre_roll_frames)
    }

    /// Returns `true` if this action has already fired.
    pub fn has_fired(&self) -> bool {
        self.fired
    }
}

// ---------------------------------------------------------------------------
// FiredAction (result of advance())
// ---------------------------------------------------------------------------

/// A record of an action that was fired by the scheduler.
#[derive(Debug, Clone)]
pub struct FiredAction {
    /// Original action identifier.
    pub id: String,
    /// Nominal target timecode frame.
    pub target_frame: u64,
    /// Frame number at which the action actually fired.
    pub fired_at_frame: u64,
    /// Kind of action that fired.
    pub kind: TimecodeActionKind,
}

// ---------------------------------------------------------------------------
// Heap entry (min-heap by trigger_frame)
// ---------------------------------------------------------------------------

/// Internal heap entry used to maintain the sorted action timeline.
#[derive(Debug)]
struct HeapEntry {
    /// Effective trigger frame (negated for min-heap ordering via BinaryHeap).
    trigger_frame: u64,
    /// Index into the actions Vec.
    index: usize,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.trigger_frame == other.trigger_frame
    }
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// Min-heap: reverse ordering so smallest trigger_frame is at the top.
impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other.trigger_frame.cmp(&self.trigger_frame)
    }
}

// ---------------------------------------------------------------------------
// TimecodeScheduler
// ---------------------------------------------------------------------------

/// Frame-accurate broadcast timecode scheduler.
///
/// Maintains a priority queue of [`ScheduledAction`]s sorted by their
/// effective trigger frame. Call [`advance`][TimecodeScheduler::advance] with
/// the current frame position to fire all pending actions that have been
/// reached.
#[derive(Debug)]
pub struct TimecodeScheduler {
    /// Nominal frame rate (frames per second) used for display / validation.
    fps: u32,
    /// All registered actions (both pending and fired).
    actions: Vec<ScheduledAction>,
    /// Min-heap ordered by trigger_frame.
    heap: BinaryHeap<HeapEntry>,
    /// Last frame position seen by `advance`.
    last_frame: u64,
    /// Total number of actions fired since creation or last reset.
    total_fired: u64,
}

impl TimecodeScheduler {
    /// Create a new scheduler for the given frame rate.
    ///
    /// The `fps` parameter is informational; all timing is expressed in
    /// absolute frame numbers so any integer frame rate is supported.
    pub fn new(fps: u32) -> Self {
        Self {
            fps,
            actions: Vec::new(),
            heap: BinaryHeap::new(),
            last_frame: 0,
            total_fired: 0,
        }
    }

    /// Return the configured frame rate.
    pub fn fps(&self) -> u32 {
        self.fps
    }

    /// Return the last frame position seen by [`advance`][Self::advance].
    pub fn last_frame(&self) -> u64 {
        self.last_frame
    }

    /// Total number of actions fired since creation or last [`reset`][Self::reset].
    pub fn total_fired(&self) -> u64 {
        self.total_fired
    }

    /// Number of actions currently registered (pending and fired).
    pub fn action_count(&self) -> usize {
        self.actions.len()
    }

    /// Number of actions still pending (not yet fired).
    pub fn pending_count(&self) -> usize {
        self.actions.iter().filter(|a| !a.fired).count()
    }

    /// Schedule an action to fire at its specified timecode.
    ///
    /// If the action's trigger frame is before or equal to the current
    /// `last_frame`, it is still registered but will not fire retroactively
    /// (it will only fire if the scheduler is reset and the position revisited).
    pub fn schedule(&mut self, action: ScheduledAction) {
        let trigger = action.trigger_frame();
        let index = self.actions.len();
        self.actions.push(action);
        self.heap.push(HeapEntry { trigger_frame: trigger, index });
    }

    /// Remove a scheduled action by ID before it fires.
    ///
    /// Returns `true` if the action was found and cancelled.
    /// Note: the heap entry will remain but the action will be marked as fired
    /// to skip it during `advance`.
    pub fn cancel(&mut self, id: &str) -> bool {
        for action in &mut self.actions {
            if action.id == id && !action.fired {
                action.fired = true;
                return true;
            }
        }
        false
    }

    /// Advance the scheduler to `current_frame`, firing all pending actions
    /// whose trigger frame has been reached.
    ///
    /// Returns a list of [`FiredAction`]s in the order they fired.
    /// Actions are fired in ascending trigger-frame order; ties are broken by
    /// insertion order.
    pub fn advance(&mut self, current_frame: u64) -> Vec<FiredAction> {
        self.last_frame = current_frame;
        let mut fired = Vec::new();

        loop {
            let Some(top) = self.heap.peek() else { break };
            if top.trigger_frame > current_frame {
                break;
            }
            let entry = self.heap.pop().expect("peek succeeded so pop must succeed");
            let action = &mut self.actions[entry.index];
            if action.fired {
                // Already cancelled or fired — skip silently.
                continue;
            }
            action.fired = true;
            self.total_fired += 1;
            fired.push(FiredAction {
                id: action.id.clone(),
                target_frame: action.target_frame,
                fired_at_frame: current_frame,
                kind: action.kind.clone(),
            });
        }

        fired
    }

    /// Reset the scheduler: clear all actions and reset the frame position to 0.
    ///
    /// Useful when restarting playout from the beginning.
    pub fn reset(&mut self) {
        self.actions.clear();
        self.heap.clear();
        self.last_frame = 0;
        self.total_fired = 0;
    }

    /// Seek to a new frame position without firing intermediate actions.
    ///
    /// Any action at or before `frame` that has not yet fired is marked as
    /// fired and **not** returned (they are considered missed). Use this for
    /// non-linear seeks.
    ///
    /// Returns the number of actions that were skipped (not fired).
    pub fn seek(&mut self, frame: u64) -> usize {
        self.last_frame = frame;
        let mut skipped = 0usize;

        loop {
            let Some(top) = self.heap.peek() else { break };
            if top.trigger_frame > frame {
                break;
            }
            let entry = self.heap.pop().expect("peek succeeded so pop must succeed");
            let action = &mut self.actions[entry.index];
            if !action.fired {
                action.fired = true;
                skipped += 1;
            }
        }

        skipped
    }

    /// Return a reference to all registered actions (pending and fired).
    pub fn actions(&self) -> &[ScheduledAction] {
        &self.actions
    }

    /// Convert a frame count to a HH:MM:SS:FF timecode string at the
    /// scheduler's configured frame rate.
    ///
    /// Returns `"??:??:??:??"` if `fps` is zero (degenerate case).
    pub fn frame_to_timecode(&self, frame: u64) -> String {
        if self.fps == 0 {
            return "??:??:??:??".to_string();
        }
        let fps = self.fps as u64;
        let ff = frame % fps;
        let total_seconds = frame / fps;
        let ss = total_seconds % 60;
        let total_minutes = total_seconds / 60;
        let mm = total_minutes % 60;
        let hh = total_minutes / 60;
        format!("{hh:02}:{mm:02}:{ss:02}:{ff:02}")
    }

    /// Parse a `HH:MM:SS:FF` timecode string into an absolute frame count.
    ///
    /// Returns `None` if the string is malformed or the fps is zero.
    pub fn timecode_to_frame(&self, tc: &str) -> Option<u64> {
        if self.fps == 0 {
            return None;
        }
        let parts: Vec<&str> = tc.split(':').collect();
        if parts.len() != 4 {
            return None;
        }
        let hh: u64 = parts[0].parse().ok()?;
        let mm: u64 = parts[1].parse().ok()?;
        let ss: u64 = parts[2].parse().ok()?;
        let ff: u64 = parts[3].parse().ok()?;
        let fps = self.fps as u64;
        if ff >= fps || mm >= 60 || ss >= 60 {
            return None;
        }
        Some(((hh * 3600 + mm * 60 + ss) * fps) + ff)
    }
}

impl Default for TimecodeScheduler {
    fn default() -> Self {
        Self::new(25)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_play(id: &str, frame: u64) -> ScheduledAction {
        ScheduledAction::new(id, frame, TimecodeActionKind::PlayClip {
            clip_id: format!("clip_{id}"),
        })
    }

    #[test]
    fn test_scheduler_new() {
        let s = TimecodeScheduler::new(30);
        assert_eq!(s.fps(), 30);
        assert_eq!(s.last_frame(), 0);
        assert_eq!(s.total_fired(), 0);
        assert_eq!(s.action_count(), 0);
    }

    #[test]
    fn test_schedule_and_advance_fires_at_frame() {
        let mut s = TimecodeScheduler::new(25);
        s.schedule(make_play("a", 50));
        let fired = s.advance(50);
        assert_eq!(fired.len(), 1);
        assert_eq!(fired[0].id, "a");
        assert_eq!(fired[0].fired_at_frame, 50);
    }

    #[test]
    fn test_advance_before_target_does_not_fire() {
        let mut s = TimecodeScheduler::new(25);
        s.schedule(make_play("b", 100));
        let fired = s.advance(99);
        assert!(fired.is_empty());
    }

    #[test]
    fn test_advance_past_target_fires() {
        let mut s = TimecodeScheduler::new(25);
        s.schedule(make_play("c", 100));
        let fired = s.advance(150);
        assert_eq!(fired.len(), 1);
        assert_eq!(fired[0].id, "c");
    }

    #[test]
    fn test_multiple_actions_fire_in_order() {
        let mut s = TimecodeScheduler::new(25);
        s.schedule(make_play("first", 10));
        s.schedule(make_play("second", 20));
        s.schedule(make_play("third", 30));
        let fired = s.advance(30);
        assert_eq!(fired.len(), 3);
        // Should be ordered by trigger frame ascending
        assert_eq!(fired[0].id, "first");
        assert_eq!(fired[1].id, "second");
        assert_eq!(fired[2].id, "third");
    }

    #[test]
    fn test_action_fires_only_once() {
        let mut s = TimecodeScheduler::new(25);
        s.schedule(make_play("once", 10));
        s.advance(10);
        let fired_again = s.advance(10);
        assert!(fired_again.is_empty());
    }

    #[test]
    fn test_pre_roll_fires_early() {
        let mut s = TimecodeScheduler::new(25);
        let action = ScheduledAction::new("early", 100, TimecodeActionKind::StopPlayout)
            .with_pre_roll(5);
        assert_eq!(action.trigger_frame(), 95);
        s.schedule(action);
        let fired = s.advance(95);
        assert_eq!(fired.len(), 1);
        assert_eq!(fired[0].target_frame, 100);
        assert_eq!(fired[0].fired_at_frame, 95);
    }

    #[test]
    fn test_cancel_prevents_firing() {
        let mut s = TimecodeScheduler::new(25);
        s.schedule(make_play("cancel_me", 50));
        let cancelled = s.cancel("cancel_me");
        assert!(cancelled);
        let fired = s.advance(100);
        assert!(fired.is_empty());
    }

    #[test]
    fn test_cancel_nonexistent_returns_false() {
        let mut s = TimecodeScheduler::new(25);
        assert!(!s.cancel("ghost"));
    }

    #[test]
    fn test_pending_count_decreases_after_fire() {
        let mut s = TimecodeScheduler::new(25);
        s.schedule(make_play("p1", 10));
        s.schedule(make_play("p2", 20));
        assert_eq!(s.pending_count(), 2);
        s.advance(10);
        assert_eq!(s.pending_count(), 1);
    }

    #[test]
    fn test_reset_clears_everything() {
        let mut s = TimecodeScheduler::new(25);
        s.schedule(make_play("x", 10));
        s.advance(10);
        s.reset();
        assert_eq!(s.action_count(), 0);
        assert_eq!(s.total_fired(), 0);
        assert_eq!(s.last_frame(), 0);
    }

    #[test]
    fn test_seek_skips_actions_silently() {
        let mut s = TimecodeScheduler::new(25);
        s.schedule(make_play("s1", 10));
        s.schedule(make_play("s2", 20));
        let skipped = s.seek(25);
        assert_eq!(skipped, 2);
        assert_eq!(s.total_fired(), 0); // seek does not count as fired
        let fired = s.advance(25); // no new actions
        assert!(fired.is_empty());
    }

    #[test]
    fn test_frame_to_timecode_30fps() {
        let s = TimecodeScheduler::new(30);
        // 1 hour = 3600 * 30 = 108000 frames
        assert_eq!(s.frame_to_timecode(108000), "01:00:00:00");
        assert_eq!(s.frame_to_timecode(0), "00:00:00:00");
        assert_eq!(s.frame_to_timecode(30), "00:00:01:00");
        assert_eq!(s.frame_to_timecode(31), "00:00:01:01");
    }

    #[test]
    fn test_timecode_to_frame_round_trip() {
        let s = TimecodeScheduler::new(25);
        let frame = 25 * 3600 + 25 * 60 + 25 * 30 + 12; // 01:30:30:12 at 25fps
        let tc = s.frame_to_timecode(frame);
        let parsed = s.timecode_to_frame(&tc);
        assert_eq!(parsed, Some(frame));
    }

    #[test]
    fn test_timecode_to_frame_invalid_format() {
        let s = TimecodeScheduler::new(25);
        assert!(s.timecode_to_frame("not:a:timecode").is_none());
        assert!(s.timecode_to_frame("00:00:00").is_none());
        assert!(s.timecode_to_frame("00:00:00:99").is_none()); // ff >= fps
    }

    #[test]
    fn test_action_kind_labels() {
        assert_eq!(TimecodeActionKind::StopPlayout.label(), "StopPlayout");
        assert_eq!(TimecodeActionKind::EndAdBreak.label(), "EndAdBreak");
        assert_eq!(
            TimecodeActionKind::PlayClip { clip_id: "x".to_string() }.label(),
            "PlayClip"
        );
    }

    #[test]
    fn test_total_fired_counter() {
        let mut s = TimecodeScheduler::new(25);
        s.schedule(make_play("a", 1));
        s.schedule(make_play("b", 2));
        s.advance(5);
        assert_eq!(s.total_fired(), 2);
    }

    #[test]
    fn test_default_fps_is_25() {
        let s = TimecodeScheduler::default();
        assert_eq!(s.fps(), 25);
    }

    #[test]
    fn test_eas_alert_action_kind() {
        let mut s = TimecodeScheduler::new(30);
        s.schedule(ScheduledAction::new(
            "eas",
            300,
            TimecodeActionKind::EasAlert { alert_id: "tornado-001".to_string() },
        ));
        let fired = s.advance(300);
        assert_eq!(fired.len(), 1);
        assert!(matches!(fired[0].kind, TimecodeActionKind::EasAlert { .. }));
    }

    #[test]
    fn test_custom_action_kind() {
        let mut s = TimecodeScheduler::new(30);
        s.schedule(ScheduledAction::new(
            "custom",
            60,
            TimecodeActionKind::Custom {
                key: "gpi_trigger".to_string(),
                payload: "relay=1".to_string(),
            },
        ));
        let fired = s.advance(60);
        assert_eq!(fired.len(), 1);
        if let TimecodeActionKind::Custom { key, payload } = &fired[0].kind {
            assert_eq!(key, "gpi_trigger");
            assert_eq!(payload, "relay=1");
        } else {
            panic!("Expected Custom action kind");
        }
    }
}
