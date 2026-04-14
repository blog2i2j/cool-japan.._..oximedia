//! Frame-accurate cue point triggering for broadcast playout.
//!
//! A `CueTrigger` holds a list of `CuePoint` entries, each anchored to a
//! specific absolute frame number.  On every call to `check_frame` the
//! trigger compares the current frame against the cue list and fires any
//! pending cues whose frame has been reached.
//!
//! # Actions
//!
//! Each cue carries a `CueAction` describing what should happen:
//!
//! | Variant | Description |
//! |---------|-------------|
//! | `PlayClip(String)` | Load and play a named clip. |
//! | `ShowGraphic(String)` | Display a named graphics asset. |
//! | `MuteAudio` | Mute all audio outputs. |
//! | `FadeToBlack` | Fade the video to black. |
//!
//! # Example
//!
//! ```
//! use oximedia_playout::cue_trigger::{CuePoint, CueAction, CueTrigger};
//!
//! let mut trigger = CueTrigger::new(25.0);
//! trigger.add_cue(CuePoint {
//!     id: "intro-slate".to_string(),
//!     timecode_frames: 0,
//!     action: CueAction::ShowGraphic("slate.png".to_string()),
//!     triggered: false,
//! });
//!
//! let actions = trigger.check_frame(0);
//! assert_eq!(actions.len(), 1);
//! ```

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// CueAction
// ---------------------------------------------------------------------------

/// The action to perform when a cue point fires.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CueAction {
    /// Load and play the named clip (clip name / path).
    PlayClip(String),
    /// Display the named graphics asset (asset name / path).
    ShowGraphic(String),
    /// Mute all audio outputs.
    MuteAudio,
    /// Fade the video output to black.
    FadeToBlack,
}

impl CueAction {
    /// Human-readable description of the action for logging.
    pub fn description(&self) -> String {
        match self {
            Self::PlayClip(name) => format!("play-clip:{}", name),
            Self::ShowGraphic(name) => format!("show-graphic:{}", name),
            Self::MuteAudio => "mute-audio".to_string(),
            Self::FadeToBlack => "fade-to-black".to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// CuePoint
// ---------------------------------------------------------------------------

/// A single frame-accurate cue point.
#[derive(Debug, Clone)]
pub struct CuePoint {
    /// Unique identifier used for logging and de-duplication.
    pub id: String,
    /// Absolute frame number at which this cue should fire.
    pub timecode_frames: u64,
    /// The action to execute when the cue fires.
    pub action: CueAction,
    /// Whether this cue has already been triggered in the current pass.
    pub triggered: bool,
}

impl CuePoint {
    /// Create a new, un-triggered cue point.
    pub fn new(id: impl Into<String>, timecode_frames: u64, action: CueAction) -> Self {
        Self {
            id: id.into(),
            timecode_frames,
            action,
            triggered: false,
        }
    }
}

// ---------------------------------------------------------------------------
// CueTrigger
// ---------------------------------------------------------------------------

/// Frame-accurate cue point registry.
///
/// Maintains an ordered list of `CuePoint` entries and fires them as
/// `check_frame` is called with advancing frame numbers.
#[derive(Debug)]
pub struct CueTrigger {
    /// All registered cue points (not necessarily sorted).
    pub cue_points: Vec<CuePoint>,
    /// Frame rate in frames per second (informational; used for conversions).
    pub fps: f64,
}

impl CueTrigger {
    /// Create a new `CueTrigger` for content at the given frame rate.
    pub fn new(fps: f64) -> Self {
        Self {
            cue_points: Vec::new(),
            fps,
        }
    }

    /// Register a new cue point.
    pub fn add_cue(&mut self, cue: CuePoint) {
        self.cue_points.push(cue);
    }

    /// Check whether any cue points should fire at `current_frame`.
    ///
    /// All pending cue points whose `timecode_frames <= current_frame` are
    /// marked as triggered and their `CueAction` values are returned.  A cue
    /// that has already been triggered (i.e. `triggered == true`) is skipped
    /// so it will not fire a second time.
    ///
    /// The returned `Vec` is in the order the cue points appear in
    /// `self.cue_points` (insertion order), not necessarily in frame order.
    pub fn check_frame(&mut self, current_frame: u64) -> Vec<CueAction> {
        let mut actions = Vec::new();
        for cue in &mut self.cue_points {
            if !cue.triggered && cue.timecode_frames <= current_frame {
                cue.triggered = true;
                actions.push(cue.action.clone());
            }
        }
        actions
    }

    /// Reset all cue points to their un-triggered state.
    ///
    /// Call this when the timeline is rewound or looped so that cues can fire
    /// again from the beginning.
    pub fn reset_all(&mut self) {
        for cue in &mut self.cue_points {
            cue.triggered = false;
        }
    }

    /// Remove all cue points from the registry.
    pub fn clear(&mut self) {
        self.cue_points.clear();
    }

    /// Number of registered cue points.
    pub fn len(&self) -> usize {
        self.cue_points.len()
    }

    /// Whether the registry has no cue points.
    pub fn is_empty(&self) -> bool {
        self.cue_points.is_empty()
    }

    /// Number of cue points that have not yet been triggered.
    pub fn pending_count(&self) -> usize {
        self.cue_points.iter().filter(|c| !c.triggered).count()
    }

    /// Millisecond timestamp of a frame number given the configured fps.
    ///
    /// Returns 0 if `fps` is zero or negative.
    pub fn frame_to_ms(&self, frame: u64) -> u64 {
        if self.fps <= 0.0 {
            return 0;
        }
        ((frame as f64 / self.fps) * 1000.0) as u64
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_trigger() -> CueTrigger {
        CueTrigger::new(25.0)
    }

    // ── CueAction ────────────────────────────────────────────────────────────

    #[test]
    fn test_cue_action_description_play_clip() {
        let a = CueAction::PlayClip("myclip.mxf".to_string());
        assert_eq!(a.description(), "play-clip:myclip.mxf");
    }

    #[test]
    fn test_cue_action_description_show_graphic() {
        let a = CueAction::ShowGraphic("logo.png".to_string());
        assert_eq!(a.description(), "show-graphic:logo.png");
    }

    #[test]
    fn test_cue_action_description_mute_audio() {
        assert_eq!(CueAction::MuteAudio.description(), "mute-audio");
    }

    #[test]
    fn test_cue_action_description_fade_to_black() {
        assert_eq!(CueAction::FadeToBlack.description(), "fade-to-black");
    }

    // ── CueTrigger::check_frame ───────────────────────────────────────────────

    #[test]
    fn test_check_frame_fires_at_exact_frame() {
        let mut t = make_trigger();
        t.add_cue(CuePoint::new("c1", 100, CueAction::FadeToBlack));
        // Not yet at frame 100
        assert!(t.check_frame(99).is_empty());
        // Exactly at frame 100
        let actions = t.check_frame(100);
        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0], CueAction::FadeToBlack);
    }

    #[test]
    fn test_check_frame_does_not_double_fire() {
        let mut t = make_trigger();
        t.add_cue(CuePoint::new("c1", 50, CueAction::MuteAudio));
        let first = t.check_frame(50);
        assert_eq!(first.len(), 1);
        let second = t.check_frame(100);
        assert!(second.is_empty(), "cue must not fire a second time");
    }

    #[test]
    fn test_check_frame_fires_multiple_cues_at_same_frame() {
        let mut t = make_trigger();
        t.add_cue(CuePoint::new("c1", 0, CueAction::MuteAudio));
        t.add_cue(CuePoint::new("c2", 0, CueAction::FadeToBlack));
        let actions = t.check_frame(0);
        assert_eq!(actions.len(), 2);
    }

    #[test]
    fn test_check_frame_fires_cues_passed_in_jump() {
        // Simulate a seek: jump from frame 0 directly to frame 500.
        let mut t = make_trigger();
        t.add_cue(CuePoint::new("early", 100, CueAction::MuteAudio));
        t.add_cue(CuePoint::new("mid", 300, CueAction::FadeToBlack));
        t.add_cue(CuePoint::new("late", 600, CueAction::MuteAudio));
        let actions = t.check_frame(500);
        // "early" and "mid" should fire; "late" (frame 600) should not.
        assert_eq!(actions.len(), 2);
    }

    // ── CueTrigger::reset_all ─────────────────────────────────────────────────

    #[test]
    fn test_reset_all_allows_re_fire() {
        let mut t = make_trigger();
        t.add_cue(CuePoint::new("c1", 10, CueAction::PlayClip("x".to_string())));
        let first = t.check_frame(10);
        assert_eq!(first.len(), 1);

        t.reset_all();
        let second = t.check_frame(10);
        assert_eq!(second.len(), 1, "after reset the cue should fire again");
    }

    // ── CueTrigger utility methods ────────────────────────────────────────────

    #[test]
    fn test_add_and_len() {
        let mut t = make_trigger();
        assert!(t.is_empty());
        t.add_cue(CuePoint::new("c1", 0, CueAction::MuteAudio));
        t.add_cue(CuePoint::new("c2", 100, CueAction::FadeToBlack));
        assert_eq!(t.len(), 2);
    }

    #[test]
    fn test_pending_count_decreases_after_firing() {
        let mut t = make_trigger();
        t.add_cue(CuePoint::new("c1", 0, CueAction::MuteAudio));
        t.add_cue(CuePoint::new("c2", 200, CueAction::FadeToBlack));
        assert_eq!(t.pending_count(), 2);
        t.check_frame(0);
        assert_eq!(t.pending_count(), 1);
    }

    #[test]
    fn test_frame_to_ms() {
        let t = CueTrigger::new(25.0);
        // 25 frames at 25 fps = 1000 ms
        assert_eq!(t.frame_to_ms(25), 1000);
    }

    #[test]
    fn test_frame_to_ms_zero_fps() {
        let t = CueTrigger::new(0.0);
        assert_eq!(t.frame_to_ms(100), 0);
    }

    #[test]
    fn test_clear() {
        let mut t = make_trigger();
        t.add_cue(CuePoint::new("c1", 0, CueAction::FadeToBlack));
        t.clear();
        assert!(t.is_empty());
    }

    // ── CuePoint ─────────────────────────────────────────────────────────────

    #[test]
    fn test_cue_point_new_not_triggered() {
        let cue = CuePoint::new("test", 100, CueAction::MuteAudio);
        assert!(!cue.triggered);
        assert_eq!(cue.timecode_frames, 100);
    }

    #[test]
    fn test_cue_action_equality() {
        assert_eq!(
            CueAction::PlayClip("a.mxf".to_string()),
            CueAction::PlayClip("a.mxf".to_string())
        );
        assert_ne!(
            CueAction::PlayClip("a.mxf".to_string()),
            CueAction::PlayClip("b.mxf".to_string())
        );
    }

    #[test]
    fn test_check_frame_play_clip_action() {
        let mut t = make_trigger();
        t.add_cue(CuePoint::new(
            "clip-cue",
            75,
            CueAction::PlayClip("commercial.mxf".to_string()),
        ));
        let actions = t.check_frame(75);
        assert_eq!(actions.len(), 1);
        assert_eq!(
            actions[0],
            CueAction::PlayClip("commercial.mxf".to_string())
        );
    }

    #[test]
    fn test_check_frame_show_graphic_action() {
        let mut t = make_trigger();
        t.add_cue(CuePoint::new(
            "g1",
            0,
            CueAction::ShowGraphic("lower_third.png".to_string()),
        ));
        let actions = t.check_frame(0);
        assert_eq!(
            actions[0],
            CueAction::ShowGraphic("lower_third.png".to_string())
        );
    }
}
