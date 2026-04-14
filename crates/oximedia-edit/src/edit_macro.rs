//! Edit macro recording and playback.
//!
//! Records a sequence of atomic edit actions (move, trim, split, delete, etc.)
//! as a [`Macro`] that can be replayed on any compatible timeline.  Macros can
//! be named, serialized to a compact text representation, and composed.
//!
//! # Example
//! ```rust
//! use oximedia_edit::edit_macro::{Macro, MacroAction, MacroRecorder};
//!
//! let mut rec = MacroRecorder::new("My Macro");
//! rec.record(MacroAction::MoveClip { clip_id: 1, delta: 100 });
//! rec.record(MacroAction::TrimIn { clip_id: 1, delta: -50 });
//! let m = rec.finish();
//! assert_eq!(m.actions().len(), 2);
//! assert_eq!(m.name(), "My Macro");
//! ```

#![allow(dead_code)]

use std::fmt;

// ─────────────────────────────────────────────────────────────────────────────
// MacroAction
// ─────────────────────────────────────────────────────────────────────────────

/// An atomic edit operation that can be recorded and replayed.
#[derive(Clone, Debug, PartialEq)]
pub enum MacroAction {
    /// Move a clip by `delta` timebase units on its current track.
    MoveClip {
        /// Target clip ID.
        clip_id: u64,
        /// Signed displacement (positive = rightward).
        delta: i64,
    },
    /// Trim the in-point of a clip.
    TrimIn {
        /// Target clip ID.
        clip_id: u64,
        /// Signed trim amount.
        delta: i64,
    },
    /// Trim the out-point of a clip.
    TrimOut {
        /// Target clip ID.
        clip_id: u64,
        /// Signed trim amount.
        delta: i64,
    },
    /// Split a clip at a relative offset from its start.
    SplitAt {
        /// Target clip ID.
        clip_id: u64,
        /// Offset from clip start (must be positive and less than duration).
        offset: i64,
    },
    /// Delete a clip.
    DeleteClip {
        /// Target clip ID.
        clip_id: u64,
    },
    /// Set clip opacity.
    SetOpacity {
        /// Target clip ID.
        clip_id: u64,
        /// New opacity value (0.0 -- 1.0).
        opacity: f32,
    },
    /// Set clip speed multiplier.
    SetSpeed {
        /// Target clip ID.
        clip_id: u64,
        /// New speed value.
        speed: f64,
    },
    /// Toggle clip mute.
    ToggleMute {
        /// Target clip ID.
        clip_id: u64,
    },
    /// Insert a gap (ripple) at a timeline position.
    InsertGap {
        /// Timeline position.
        position: i64,
        /// Gap duration (timebase units).
        duration: i64,
    },
    /// A no-operation marker (can be used as a separator in sequences).
    Noop,
}

impl MacroAction {
    /// Human-readable label for the action type.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::MoveClip { .. } => "Move Clip",
            Self::TrimIn { .. } => "Trim In",
            Self::TrimOut { .. } => "Trim Out",
            Self::SplitAt { .. } => "Split",
            Self::DeleteClip { .. } => "Delete",
            Self::SetOpacity { .. } => "Set Opacity",
            Self::SetSpeed { .. } => "Set Speed",
            Self::ToggleMute { .. } => "Toggle Mute",
            Self::InsertGap { .. } => "Insert Gap",
            Self::Noop => "Noop",
        }
    }

    /// Returns the clip ID this action targets, if any.
    #[must_use]
    pub fn targets_clip(&self) -> Option<u64> {
        match self {
            Self::MoveClip { clip_id, .. }
            | Self::TrimIn { clip_id, .. }
            | Self::TrimOut { clip_id, .. }
            | Self::SplitAt { clip_id, .. }
            | Self::DeleteClip { clip_id }
            | Self::SetOpacity { clip_id, .. }
            | Self::SetSpeed { clip_id, .. }
            | Self::ToggleMute { clip_id } => Some(*clip_id),
            Self::InsertGap { .. } | Self::Noop => None,
        }
    }

    /// Remap clip IDs using a provided mapping function.
    ///
    /// Returns a new action with the remapped ID.  If `f` returns `None` for
    /// a clip ID the action targets, the original ID is kept.
    #[must_use]
    pub fn remap_clip_id<F>(&self, f: F) -> Self
    where
        F: Fn(u64) -> Option<u64>,
    {
        match self {
            Self::MoveClip { clip_id, delta } => {
                let id = f(*clip_id).unwrap_or(*clip_id);
                Self::MoveClip {
                    clip_id: id,
                    delta: *delta,
                }
            }
            Self::TrimIn { clip_id, delta } => {
                let id = f(*clip_id).unwrap_or(*clip_id);
                Self::TrimIn {
                    clip_id: id,
                    delta: *delta,
                }
            }
            Self::TrimOut { clip_id, delta } => {
                let id = f(*clip_id).unwrap_or(*clip_id);
                Self::TrimOut {
                    clip_id: id,
                    delta: *delta,
                }
            }
            Self::SplitAt { clip_id, offset } => {
                let id = f(*clip_id).unwrap_or(*clip_id);
                Self::SplitAt {
                    clip_id: id,
                    offset: *offset,
                }
            }
            Self::DeleteClip { clip_id } => {
                let id = f(*clip_id).unwrap_or(*clip_id);
                Self::DeleteClip { clip_id: id }
            }
            Self::SetOpacity { clip_id, opacity } => {
                let id = f(*clip_id).unwrap_or(*clip_id);
                Self::SetOpacity {
                    clip_id: id,
                    opacity: *opacity,
                }
            }
            Self::SetSpeed { clip_id, speed } => {
                let id = f(*clip_id).unwrap_or(*clip_id);
                Self::SetSpeed {
                    clip_id: id,
                    speed: *speed,
                }
            }
            Self::ToggleMute { clip_id } => {
                let id = f(*clip_id).unwrap_or(*clip_id);
                Self::ToggleMute { clip_id: id }
            }
            Self::InsertGap { position, duration } => Self::InsertGap {
                position: *position,
                duration: *duration,
            },
            Self::Noop => Self::Noop,
        }
    }
}

impl fmt::Display for MacroAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MoveClip { clip_id, delta } => {
                write!(f, "move({clip_id},{delta})")
            }
            Self::TrimIn { clip_id, delta } => {
                write!(f, "trim_in({clip_id},{delta})")
            }
            Self::TrimOut { clip_id, delta } => {
                write!(f, "trim_out({clip_id},{delta})")
            }
            Self::SplitAt { clip_id, offset } => {
                write!(f, "split({clip_id},{offset})")
            }
            Self::DeleteClip { clip_id } => {
                write!(f, "delete({clip_id})")
            }
            Self::SetOpacity { clip_id, opacity } => {
                write!(f, "opacity({clip_id},{opacity})")
            }
            Self::SetSpeed { clip_id, speed } => {
                write!(f, "speed({clip_id},{speed})")
            }
            Self::ToggleMute { clip_id } => {
                write!(f, "mute({clip_id})")
            }
            Self::InsertGap { position, duration } => {
                write!(f, "gap({position},{duration})")
            }
            Self::Noop => write!(f, "noop"),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Macro
// ─────────────────────────────────────────────────────────────────────────────

/// A named, replayable sequence of edit actions.
#[derive(Clone, Debug)]
pub struct Macro {
    name: String,
    actions: Vec<MacroAction>,
}

impl Macro {
    /// Create a macro with the given name and action list.
    #[must_use]
    pub fn new(name: impl Into<String>, actions: Vec<MacroAction>) -> Self {
        Self {
            name: name.into(),
            actions,
        }
    }

    /// Macro name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Recorded actions.
    #[must_use]
    pub fn actions(&self) -> &[MacroAction] {
        &self.actions
    }

    /// Number of actions.
    #[must_use]
    pub fn len(&self) -> usize {
        self.actions.len()
    }

    /// Whether the macro is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.actions.is_empty()
    }

    /// Create a new macro by appending another macro's actions to this one.
    #[must_use]
    pub fn chain(&self, other: &Macro) -> Macro {
        let mut combined = self.actions.clone();
        combined.extend_from_slice(&other.actions);
        Macro {
            name: format!("{} + {}", self.name, other.name),
            actions: combined,
        }
    }

    /// Return a new macro with all clip IDs remapped through `f`.
    #[must_use]
    pub fn remap_ids<F>(&self, f: F) -> Macro
    where
        F: Fn(u64) -> Option<u64>,
    {
        let actions = self.actions.iter().map(|a| a.remap_clip_id(&f)).collect();
        Macro {
            name: self.name.clone(),
            actions,
        }
    }

    /// Serialize to a compact multi-line text representation.
    #[must_use]
    pub fn to_text(&self) -> String {
        let mut out = format!("# {}\n", self.name);
        for action in &self.actions {
            out.push_str(&format!("{action}\n"));
        }
        out
    }

    /// Return unique clip IDs referenced by this macro.
    #[must_use]
    pub fn referenced_clip_ids(&self) -> Vec<u64> {
        let mut ids: Vec<u64> = self
            .actions
            .iter()
            .filter_map(|a| a.targets_clip())
            .collect();
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    /// Filter out `Noop` actions and return a cleaned copy.
    #[must_use]
    pub fn compact(&self) -> Macro {
        let actions = self
            .actions
            .iter()
            .filter(|a| !matches!(a, MacroAction::Noop))
            .cloned()
            .collect();
        Macro {
            name: self.name.clone(),
            actions,
        }
    }

    /// Repeat the macro `n` times (concatenate actions n times).
    #[must_use]
    pub fn repeat(&self, n: usize) -> Macro {
        let mut actions = Vec::with_capacity(self.actions.len() * n);
        for _ in 0..n {
            actions.extend_from_slice(&self.actions);
        }
        Macro {
            name: format!("{} x{n}", self.name),
            actions,
        }
    }
}

impl fmt::Display for Macro {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Macro '{}' ({} actions)", self.name, self.actions.len())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MacroRecorder
// ─────────────────────────────────────────────────────────────────────────────

/// Records edit actions into a [`Macro`].
#[derive(Debug)]
pub struct MacroRecorder {
    name: String,
    actions: Vec<MacroAction>,
    recording: bool,
}

impl MacroRecorder {
    /// Start recording a new macro with the given name.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            actions: Vec::new(),
            recording: true,
        }
    }

    /// Record an action (appended to the end).  Does nothing if the recorder
    /// has been finished.
    pub fn record(&mut self, action: MacroAction) {
        if self.recording {
            self.actions.push(action);
        }
    }

    /// How many actions have been recorded so far.
    #[must_use]
    pub fn action_count(&self) -> usize {
        self.actions.len()
    }

    /// Whether the recorder is still accepting actions.
    #[must_use]
    pub fn is_recording(&self) -> bool {
        self.recording
    }

    /// Undo the last recorded action (pop).
    ///
    /// Returns the removed action, or `None` if empty or not recording.
    pub fn undo_last(&mut self) -> Option<MacroAction> {
        if self.recording {
            self.actions.pop()
        } else {
            None
        }
    }

    /// Finish recording and produce the [`Macro`].
    ///
    /// After this call the recorder is consumed.
    #[must_use]
    pub fn finish(mut self) -> Macro {
        self.recording = false;
        Macro {
            name: self.name,
            actions: self.actions,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MacroLibrary
// ─────────────────────────────────────────────────────────────────────────────

/// A named collection of macros for quick retrieval.
#[derive(Debug, Default)]
pub struct MacroLibrary {
    macros: Vec<Macro>,
}

impl MacroLibrary {
    /// Create an empty library.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a macro to the library.
    pub fn add(&mut self, m: Macro) {
        self.macros.push(m);
    }

    /// Find a macro by name (first match).
    #[must_use]
    pub fn find(&self, name: &str) -> Option<&Macro> {
        self.macros.iter().find(|m| m.name() == name)
    }

    /// Remove a macro by name (first match).  Returns the removed macro.
    pub fn remove(&mut self, name: &str) -> Option<Macro> {
        if let Some(pos) = self.macros.iter().position(|m| m.name() == name) {
            Some(self.macros.remove(pos))
        } else {
            None
        }
    }

    /// Number of macros in the library.
    #[must_use]
    pub fn len(&self) -> usize {
        self.macros.len()
    }

    /// Whether the library is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.macros.is_empty()
    }

    /// List all macro names.
    #[must_use]
    pub fn names(&self) -> Vec<&str> {
        self.macros.iter().map(|m| m.name()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recorder_basic_flow() {
        let mut rec = MacroRecorder::new("Test");
        assert!(rec.is_recording());
        rec.record(MacroAction::MoveClip {
            clip_id: 1,
            delta: 100,
        });
        rec.record(MacroAction::TrimIn {
            clip_id: 2,
            delta: -50,
        });
        assert_eq!(rec.action_count(), 2);
        let m = rec.finish();
        assert_eq!(m.name(), "Test");
        assert_eq!(m.len(), 2);
    }

    #[test]
    fn test_recorder_undo_last() {
        let mut rec = MacroRecorder::new("Undo Test");
        rec.record(MacroAction::Noop);
        rec.record(MacroAction::DeleteClip { clip_id: 5 });
        let undone = rec.undo_last();
        assert_eq!(undone, Some(MacroAction::DeleteClip { clip_id: 5 }));
        assert_eq!(rec.action_count(), 1);
    }

    #[test]
    fn test_recorder_finish_stops_recording() {
        let mut rec = MacroRecorder::new("Done");
        rec.record(MacroAction::Noop);
        let m = rec.finish();
        assert_eq!(m.len(), 1);
    }

    #[test]
    fn test_macro_chain() {
        let a = Macro::new("A", vec![MacroAction::Noop]);
        let b = Macro::new("B", vec![MacroAction::DeleteClip { clip_id: 1 }]);
        let c = a.chain(&b);
        assert_eq!(c.len(), 2);
        assert_eq!(c.name(), "A + B");
    }

    #[test]
    fn test_macro_repeat() {
        let m = Macro::new("R", vec![MacroAction::Noop]);
        let repeated = m.repeat(3);
        assert_eq!(repeated.len(), 3);
        assert!(repeated.name().contains("x3"));
    }

    #[test]
    fn test_macro_compact_removes_noops() {
        let m = Macro::new(
            "Compact",
            vec![
                MacroAction::Noop,
                MacroAction::MoveClip {
                    clip_id: 1,
                    delta: 10,
                },
                MacroAction::Noop,
            ],
        );
        let c = m.compact();
        assert_eq!(c.len(), 1);
        assert_eq!(
            c.actions()[0],
            MacroAction::MoveClip {
                clip_id: 1,
                delta: 10
            }
        );
    }

    #[test]
    fn test_macro_referenced_clip_ids() {
        let m = Macro::new(
            "IDs",
            vec![
                MacroAction::MoveClip {
                    clip_id: 3,
                    delta: 0,
                },
                MacroAction::DeleteClip { clip_id: 1 },
                MacroAction::TrimIn {
                    clip_id: 3,
                    delta: 10,
                },
                MacroAction::InsertGap {
                    position: 0,
                    duration: 50,
                },
            ],
        );
        let ids = m.referenced_clip_ids();
        assert_eq!(ids, vec![1, 3]);
    }

    #[test]
    fn test_macro_remap_ids() {
        let m = Macro::new(
            "Remap",
            vec![
                MacroAction::MoveClip {
                    clip_id: 1,
                    delta: 10,
                },
                MacroAction::DeleteClip { clip_id: 2 },
            ],
        );
        let remapped = m.remap_ids(|id| Some(id + 100));
        assert_eq!(
            remapped.actions()[0],
            MacroAction::MoveClip {
                clip_id: 101,
                delta: 10
            }
        );
        assert_eq!(
            remapped.actions()[1],
            MacroAction::DeleteClip { clip_id: 102 }
        );
    }

    #[test]
    fn test_macro_to_text() {
        let m = Macro::new(
            "TextTest",
            vec![
                MacroAction::MoveClip {
                    clip_id: 1,
                    delta: 100,
                },
                MacroAction::Noop,
            ],
        );
        let text = m.to_text();
        assert!(text.starts_with("# TextTest\n"));
        assert!(text.contains("move(1,100)"));
        assert!(text.contains("noop"));
    }

    #[test]
    fn test_macro_display() {
        let m = Macro::new("Show", vec![MacroAction::Noop]);
        let s = format!("{m}");
        assert!(s.contains("Show"));
        assert!(s.contains("1 actions"));
    }

    #[test]
    fn test_action_label() {
        assert_eq!(
            MacroAction::MoveClip {
                clip_id: 0,
                delta: 0
            }
            .label(),
            "Move Clip"
        );
        assert_eq!(MacroAction::Noop.label(), "Noop");
        assert_eq!(
            MacroAction::SetOpacity {
                clip_id: 0,
                opacity: 0.0
            }
            .label(),
            "Set Opacity"
        );
    }

    #[test]
    fn test_action_targets_clip() {
        assert_eq!(
            MacroAction::MoveClip {
                clip_id: 42,
                delta: 0
            }
            .targets_clip(),
            Some(42)
        );
        assert_eq!(MacroAction::Noop.targets_clip(), None);
        assert_eq!(
            MacroAction::InsertGap {
                position: 0,
                duration: 50
            }
            .targets_clip(),
            None
        );
    }

    #[test]
    fn test_library_crud() {
        let mut lib = MacroLibrary::new();
        assert!(lib.is_empty());
        lib.add(Macro::new("Alpha", vec![]));
        lib.add(Macro::new("Beta", vec![MacroAction::Noop]));
        assert_eq!(lib.len(), 2);
        assert_eq!(lib.names(), vec!["Alpha", "Beta"]);

        let found = lib.find("Beta");
        assert!(found.is_some());
        assert_eq!(found.map(|m| m.len()), Some(1));

        let removed = lib.remove("Alpha");
        assert!(removed.is_some());
        assert_eq!(lib.len(), 1);

        assert!(lib.find("Alpha").is_none());
    }

    #[test]
    fn test_library_find_missing_returns_none() {
        let lib = MacroLibrary::new();
        assert!(lib.find("nonexistent").is_none());
    }

    #[test]
    fn test_action_display_all_variants() {
        let actions = vec![
            MacroAction::MoveClip {
                clip_id: 1,
                delta: 10,
            },
            MacroAction::TrimIn {
                clip_id: 2,
                delta: -5,
            },
            MacroAction::TrimOut {
                clip_id: 3,
                delta: 5,
            },
            MacroAction::SplitAt {
                clip_id: 4,
                offset: 50,
            },
            MacroAction::DeleteClip { clip_id: 5 },
            MacroAction::SetOpacity {
                clip_id: 6,
                opacity: 0.5,
            },
            MacroAction::SetSpeed {
                clip_id: 7,
                speed: 2.0,
            },
            MacroAction::ToggleMute { clip_id: 8 },
            MacroAction::InsertGap {
                position: 100,
                duration: 50,
            },
            MacroAction::Noop,
        ];
        for action in &actions {
            let s = format!("{action}");
            assert!(!s.is_empty());
        }
    }

    #[test]
    fn test_remap_noop_and_gap_unchanged() {
        let noop = MacroAction::Noop;
        assert_eq!(noop.remap_clip_id(|id| Some(id + 1)), MacroAction::Noop);

        let gap = MacroAction::InsertGap {
            position: 10,
            duration: 20,
        };
        assert_eq!(
            gap.remap_clip_id(|id| Some(id + 1)),
            MacroAction::InsertGap {
                position: 10,
                duration: 20,
            }
        );
    }
}
