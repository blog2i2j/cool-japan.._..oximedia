#![allow(dead_code)]
//! Timeline diff and comparison for visual conflict resolution.
//!
//! `timeline_compare` compares two timeline snapshots (or a timeline against
//! a reference) and produces a structured diff describing what changed.
//!
//! # Use Cases
//!
//! - Collaborative editing conflict resolution
//! - Version history diffing
//! - Round-trip verification after EDL/XML export-import

use serde::{Deserialize, Serialize};

use crate::clip::ClipId;
use crate::track::TrackId;
use crate::types::{Duration, Position};

// ─────────────────────────────────────────────────────────────────
// Lightweight snapshot types used by the comparator
// ─────────────────────────────────────────────────────────────────

/// Minimal description of a clip as seen by the comparator.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClipSnapshot {
    /// Clip identifier.
    pub id: ClipId,
    /// Human-readable name.
    pub name: String,
    /// Position on the timeline.
    pub position: Position,
    /// Duration on the timeline.
    pub duration: Duration,
    /// Source media path or identifier.
    pub source: String,
    /// In-point in source media.
    pub source_in: Position,
}

/// Minimal description of a track as seen by the comparator.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrackSnapshot {
    /// Track identifier.
    pub id: TrackId,
    /// Track name.
    pub name: String,
    /// Whether the track is muted.
    pub muted: bool,
    /// Whether the track is locked.
    pub locked: bool,
    /// Clips on this track.
    pub clips: Vec<ClipSnapshot>,
}

/// A lightweight snapshot of an entire timeline for diffing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineSnapshot {
    /// Human-readable label (e.g. version tag or timestamp).
    pub label: String,
    /// Tracks in order.
    pub tracks: Vec<TrackSnapshot>,
    /// Timeline frame rate numerator.
    pub fps_num: u32,
    /// Timeline frame rate denominator.
    pub fps_den: u32,
}

impl TimelineSnapshot {
    /// Create an empty snapshot.
    #[must_use]
    pub fn new(label: impl Into<String>, fps_num: u32, fps_den: u32) -> Self {
        Self {
            label: label.into(),
            tracks: Vec::new(),
            fps_num,
            fps_den,
        }
    }

    /// Add a track snapshot.
    pub fn add_track(&mut self, track: TrackSnapshot) {
        self.tracks.push(track);
    }

    /// Find a track by ID.
    #[must_use]
    pub fn find_track(&self, id: TrackId) -> Option<&TrackSnapshot> {
        self.tracks.iter().find(|t| t.id == id)
    }
}

// ─────────────────────────────────────────────────────────────────
// Diff types
// ─────────────────────────────────────────────────────────────────

/// A change to a clip's position on the timeline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClipMove {
    /// Track containing the clip.
    pub track_id: TrackId,
    /// Clip that moved.
    pub clip_id: ClipId,
    /// Previous timeline position.
    pub from: Position,
    /// New timeline position.
    pub to: Position,
}

/// A change to a clip's duration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClipDurationChange {
    /// Track containing the clip.
    pub track_id: TrackId,
    /// Clip whose duration changed.
    pub clip_id: ClipId,
    /// Old duration.
    pub from: Duration,
    /// New duration.
    pub to: Duration,
}

/// A change to a clip's source media or in-point.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClipSourceChange {
    /// Track containing the clip.
    pub track_id: TrackId,
    /// Clip that changed source.
    pub clip_id: ClipId,
    /// Old source identifier.
    pub from_source: String,
    /// New source identifier.
    pub to_source: String,
}

/// A single item in the timeline diff.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DiffItem {
    /// A track was added.
    TrackAdded {
        /// ID of the added track.
        track_id: TrackId,
        /// Name of the added track.
        name: String,
    },
    /// A track was removed.
    TrackRemoved {
        /// ID of the removed track.
        track_id: TrackId,
        /// Name of the removed track.
        name: String,
    },
    /// A track was renamed.
    TrackRenamed {
        /// Track ID.
        track_id: TrackId,
        /// Old name.
        from: String,
        /// New name.
        to: String,
    },
    /// A track's mute state changed.
    TrackMuteChanged {
        /// Track ID.
        track_id: TrackId,
        /// New mute value.
        muted: bool,
    },
    /// A clip was added to a track.
    ClipAdded {
        /// Track containing the new clip.
        track_id: TrackId,
        /// The added clip snapshot.
        clip: ClipSnapshot,
    },
    /// A clip was removed from a track.
    ClipRemoved {
        /// Track that contained the removed clip.
        track_id: TrackId,
        /// ID of the removed clip.
        clip_id: ClipId,
        /// Name of the removed clip.
        name: String,
    },
    /// A clip moved to a different position.
    ClipMoved(ClipMove),
    /// A clip's duration changed (trim).
    ClipTrimmed(ClipDurationChange),
    /// A clip's source media or in-point changed.
    ClipSourceChanged(ClipSourceChange),
}

impl DiffItem {
    /// Human-readable description.
    #[must_use]
    pub fn describe(&self) -> String {
        match self {
            Self::TrackAdded { name, .. } => format!("Track added: '{name}'"),
            Self::TrackRemoved { name, .. } => format!("Track removed: '{name}'"),
            Self::TrackRenamed { from, to, .. } => format!("Track renamed: '{from}' → '{to}'"),
            Self::TrackMuteChanged { track_id, muted } => {
                format!("Track {track_id} mute = {muted}")
            }
            Self::ClipAdded { clip, .. } => {
                format!("Clip added: '{}' at frame {}", clip.name, clip.position.0)
            }
            Self::ClipRemoved { name, .. } => format!("Clip removed: '{name}'"),
            Self::ClipMoved(m) => {
                format!(
                    "Clip moved: frame {} → {}",
                    m.from.0, m.to.0
                )
            }
            Self::ClipTrimmed(t) => {
                format!("Clip trimmed: {} → {} frames", t.from.0, t.to.0)
            }
            Self::ClipSourceChanged(c) => {
                format!("Clip re-sourced: '{}' → '{}'", c.from_source, c.to_source)
            }
        }
    }

    /// Returns `true` if this is a potentially conflicting structural change
    /// (clip moved, added, or removed).
    #[must_use]
    pub fn is_structural(&self) -> bool {
        matches!(
            self,
            Self::ClipAdded { .. }
                | Self::ClipRemoved { .. }
                | Self::ClipMoved(_)
                | Self::TrackAdded { .. }
                | Self::TrackRemoved { .. }
        )
    }
}

/// The full diff between two timeline snapshots.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct TimelineDiff {
    /// Changes found in the diff.
    pub items: Vec<DiffItem>,
}

impl TimelineDiff {
    /// Number of diff items.
    #[must_use]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Returns `true` if no differences were found.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Count of structurally significant changes.
    #[must_use]
    pub fn structural_change_count(&self) -> usize {
        self.items.iter().filter(|i| i.is_structural()).count()
    }

    /// Return only structural changes.
    #[must_use]
    pub fn structural_changes(&self) -> Vec<&DiffItem> {
        self.items.iter().filter(|i| i.is_structural()).collect()
    }
}

// ─────────────────────────────────────────────────────────────────
// Comparator
// ─────────────────────────────────────────────────────────────────

/// Compares two [`TimelineSnapshot`]s and produces a [`TimelineDiff`].
#[derive(Debug, Default)]
pub struct TimelineComparator;

impl TimelineComparator {
    /// Create a new comparator.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Compare `base` against `revised` and return the diff.
    ///
    /// Items appear in `revised` but not `base` → Added.
    /// Items in `base` but not in `revised` → Removed.
    /// Items present in both but with changed fields → specific change type.
    #[must_use]
    pub fn compare(&self, base: &TimelineSnapshot, revised: &TimelineSnapshot) -> TimelineDiff {
        let mut items = Vec::new();

        // ── Track-level diff ──────────────────────────────────────────
        // Tracks removed in revised
        for bt in &base.tracks {
            if revised.find_track(bt.id).is_none() {
                items.push(DiffItem::TrackRemoved {
                    track_id: bt.id,
                    name: bt.name.clone(),
                });
            }
        }
        // Tracks added in revised
        for rt in &revised.tracks {
            if base.find_track(rt.id).is_none() {
                items.push(DiffItem::TrackAdded {
                    track_id: rt.id,
                    name: rt.name.clone(),
                });
            }
        }
        // Tracks in both — check for mutations
        for bt in &base.tracks {
            if let Some(rt) = revised.find_track(bt.id) {
                if bt.name != rt.name {
                    items.push(DiffItem::TrackRenamed {
                        track_id: bt.id,
                        from: bt.name.clone(),
                        to: rt.name.clone(),
                    });
                }
                if bt.muted != rt.muted {
                    items.push(DiffItem::TrackMuteChanged {
                        track_id: bt.id,
                        muted: rt.muted,
                    });
                }
                // ── Clip-level diff ───────────────────────────────────
                self.diff_clips(bt, rt, &mut items);
            }
        }

        TimelineDiff { items }
    }

    /// Diff the clips on a pair of tracks that share the same [`TrackId`].
    fn diff_clips(
        &self,
        base_track: &TrackSnapshot,
        revised_track: &TrackSnapshot,
        items: &mut Vec<DiffItem>,
    ) {
        let track_id = base_track.id;

        // Index base clips by ID.
        let base_map: std::collections::HashMap<ClipId, &ClipSnapshot> =
            base_track.clips.iter().map(|c| (c.id, c)).collect();
        let revised_map: std::collections::HashMap<ClipId, &ClipSnapshot> =
            revised_track.clips.iter().map(|c| (c.id, c)).collect();

        // Removed clips
        for (id, bc) in &base_map {
            if !revised_map.contains_key(id) {
                items.push(DiffItem::ClipRemoved {
                    track_id,
                    clip_id: *id,
                    name: bc.name.clone(),
                });
            }
        }

        // Added clips
        for (id, rc) in &revised_map {
            if !base_map.contains_key(id) {
                items.push(DiffItem::ClipAdded {
                    track_id,
                    clip: (*rc).clone(),
                });
            }
        }

        // Clips in both — check mutations
        for (id, bc) in &base_map {
            if let Some(rc) = revised_map.get(id) {
                if bc.position != rc.position {
                    items.push(DiffItem::ClipMoved(ClipMove {
                        track_id,
                        clip_id: *id,
                        from: bc.position,
                        to: rc.position,
                    }));
                }
                if bc.duration != rc.duration {
                    items.push(DiffItem::ClipTrimmed(ClipDurationChange {
                        track_id,
                        clip_id: *id,
                        from: bc.duration,
                        to: rc.duration,
                    }));
                }
                if bc.source != rc.source || bc.source_in != rc.source_in {
                    items.push(DiffItem::ClipSourceChanged(ClipSourceChange {
                        track_id,
                        clip_id: *id,
                        from_source: bc.source.clone(),
                        to_source: rc.source.clone(),
                    }));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_track(name: &str) -> TrackSnapshot {
        TrackSnapshot {
            id: TrackId::new(),
            name: name.to_string(),
            muted: false,
            locked: false,
            clips: Vec::new(),
        }
    }

    fn make_clip(name: &str, pos: i64, dur: i64) -> ClipSnapshot {
        ClipSnapshot {
            id: ClipId::new(),
            name: name.to_string(),
            position: Position::new(pos),
            duration: Duration(dur),
            source: "file.mov".to_string(),
            source_in: Position::new(0),
        }
    }

    #[test]
    fn test_no_changes() {
        let base = TimelineSnapshot::new("base", 24000, 1001);
        let revised = base.clone();
        let diff = TimelineComparator::new().compare(&base, &revised);
        assert!(diff.is_empty());
    }

    #[test]
    fn test_track_added() {
        let base = TimelineSnapshot::new("base", 24000, 1001);
        let mut revised = base.clone();
        revised.add_track(make_track("V2"));
        let diff = TimelineComparator::new().compare(&base, &revised);
        assert_eq!(diff.items.len(), 1);
        assert!(matches!(diff.items[0], DiffItem::TrackAdded { .. }));
    }

    #[test]
    fn test_track_removed() {
        let mut base = TimelineSnapshot::new("base", 24000, 1001);
        base.add_track(make_track("V1"));
        let revised = TimelineSnapshot::new("revised", 24000, 1001);
        let diff = TimelineComparator::new().compare(&base, &revised);
        assert!(matches!(diff.items[0], DiffItem::TrackRemoved { .. }));
    }

    #[test]
    fn test_clip_added() {
        let mut t = make_track("V1");
        let track_id = t.id;
        let base_snap = {
            let mut s = TimelineSnapshot::new("base", 24, 1);
            s.add_track(TrackSnapshot {
                id: track_id,
                name: "V1".to_string(),
                muted: false,
                locked: false,
                clips: Vec::new(),
            });
            s
        };
        t.clips.push(make_clip("clip1", 0, 50));
        let mut revised = TimelineSnapshot::new("revised", 24, 1);
        revised.add_track(t);

        let diff = TimelineComparator::new().compare(&base_snap, &revised);
        assert_eq!(diff.items.len(), 1);
        assert!(matches!(diff.items[0], DiffItem::ClipAdded { .. }));
    }

    #[test]
    fn test_clip_moved() {
        let clip = make_clip("clip1", 0, 100);
        let clip_id = clip.id;

        let t_id = TrackId::new();
        let make_snap_with_clip = |pos: i64| -> TimelineSnapshot {
            let mut s = TimelineSnapshot::new("s", 24, 1);
            s.add_track(TrackSnapshot {
                id: t_id,
                name: "V1".to_string(),
                muted: false,
                locked: false,
                clips: vec![ClipSnapshot {
                    id: clip_id,
                    name: "clip1".to_string(),
                    position: Position::new(pos),
                    duration: Duration(100),
                    source: "f.mov".to_string(),
                    source_in: Position::new(0),
                }],
            });
            s
        };

        let base = make_snap_with_clip(0);
        let revised = make_snap_with_clip(50);

        let diff = TimelineComparator::new().compare(&base, &revised);
        assert_eq!(diff.items.len(), 1);
        if let DiffItem::ClipMoved(m) = &diff.items[0] {
            assert_eq!(m.from.0, 0);
            assert_eq!(m.to.0, 50);
        } else {
            panic!("expected ClipMoved");
        }
    }

    #[test]
    fn test_structural_change_count() {
        let mut base = TimelineSnapshot::new("base", 24, 1);
        base.add_track(make_track("V1"));
        let revised = TimelineSnapshot::new("revised", 24, 1);
        let diff = TimelineComparator::new().compare(&base, &revised);
        assert_eq!(diff.structural_change_count(), 1);
    }
}
