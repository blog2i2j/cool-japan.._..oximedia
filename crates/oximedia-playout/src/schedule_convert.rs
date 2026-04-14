//! Playlist-to-schedule conversion for broadcast playout.
//!
//! Converts a flat `PlaylistItem` sequence into a timeline of `ScheduledItem`
//! entries, each anchored to an absolute Unix timestamp (milliseconds).
//!
//! # Example
//!
//! ```
//! use oximedia_playout::schedule_convert::{PlaylistToSchedule, ScheduledItem};
//! use oximedia_playout::schedule_convert::SchedulePlaylistItem;
//!
//! let items = vec![
//!     SchedulePlaylistItem { name: "Clip A".into(), duration_frames: Some(250), fps: 25.0 },
//!     SchedulePlaylistItem { name: "Clip B".into(), duration_frames: Some(500), fps: 25.0 },
//! ];
//! let scheduled = PlaylistToSchedule::convert(&items, 1_700_000_000_000);
//! assert_eq!(scheduled.len(), 2);
//! assert_eq!(scheduled[0].start_ts_ms, 1_700_000_000_000);
//! ```

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// Lightweight playlist item for conversion (avoids circular deps)
// ---------------------------------------------------------------------------

/// Minimal playlist item representation used during schedule conversion.
///
/// Full `PlaylistItem` structs cannot be used here because of `parking_lot`
/// and other heavy dependencies; callers can cheaply convert from the real
/// type.
#[derive(Debug, Clone)]
pub struct SchedulePlaylistItem {
    /// Display name / title of the clip.
    pub name: String,
    /// Duration in frames, or `None` for live / unknown-duration sources.
    pub duration_frames: Option<u64>,
    /// Frame rate in frames per second used to convert frame counts to wall
    /// clock time.  Defaults to 25.0 if zero or negative.
    pub fps: f64,
}

impl SchedulePlaylistItem {
    /// Duration of this item in whole milliseconds.
    ///
    /// Returns 0 for items with no known duration.
    #[must_use]
    pub fn duration_ms(&self) -> u64 {
        let fps = if self.fps > 0.0 { self.fps } else { 25.0 };
        match self.duration_frames {
            Some(frames) => ((frames as f64 / fps) * 1_000.0) as u64,
            None => 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Output type
// ---------------------------------------------------------------------------

/// A single entry in the converted broadcast schedule.
#[derive(Debug, Clone, PartialEq)]
pub struct ScheduledItem {
    /// The index of the source playlist item (0-based).
    pub index: usize,
    /// Human-readable title.
    pub title: String,
    /// Absolute start timestamp in Unix milliseconds.
    pub start_ts_ms: u64,
    /// Duration in milliseconds (0 for live/unknown).
    pub duration_ms: u64,
    /// Derived end timestamp in Unix milliseconds (0 for live/unknown).
    pub end_ts_ms: u64,
}

impl ScheduledItem {
    /// Returns `true` when the item has a known finite duration.
    #[must_use]
    pub fn has_duration(&self) -> bool {
        self.duration_ms > 0
    }
}

// ---------------------------------------------------------------------------
// Converter
// ---------------------------------------------------------------------------

/// Converts a `PlaylistItem` sequence into an absolute-time schedule.
pub struct PlaylistToSchedule;

impl PlaylistToSchedule {
    /// Convert `playlist` to a `Vec<ScheduledItem>` anchored at `start_ts`.
    ///
    /// Items are laid out sequentially: each item starts immediately after
    /// the previous one ends.  Items with unknown duration (duration_ms == 0)
    /// are given a zero-length slot — they do not advance the clock.
    ///
    /// # Arguments
    ///
    /// * `playlist` — Ordered slice of playlist items.
    /// * `start_ts`  — Absolute Unix timestamp in milliseconds for the first
    ///                  item's air time.
    ///
    /// # Returns
    ///
    /// A `Vec<ScheduledItem>` with one entry per input item.  If `playlist` is
    /// empty, an empty `Vec` is returned.
    #[must_use]
    pub fn convert(playlist: &[SchedulePlaylistItem], start_ts: u64) -> Vec<ScheduledItem> {
        let mut result = Vec::with_capacity(playlist.len());
        let mut cursor_ms = start_ts;

        for (index, item) in playlist.iter().enumerate() {
            let dur = item.duration_ms();
            let end = if dur > 0 {
                cursor_ms.saturating_add(dur)
            } else {
                cursor_ms
            };

            result.push(ScheduledItem {
                index,
                title: item.name.clone(),
                start_ts_ms: cursor_ms,
                duration_ms: dur,
                end_ts_ms: end,
            });

            if dur > 0 {
                cursor_ms = cursor_ms.saturating_add(dur);
            }
        }

        result
    }

    /// Convert but skip disabled items (items whose name starts with `!`).
    ///
    /// This is a convenience wrapper demonstrating filter-before-convert.
    #[must_use]
    pub fn convert_enabled(playlist: &[SchedulePlaylistItem], start_ts: u64) -> Vec<ScheduledItem> {
        let enabled: Vec<SchedulePlaylistItem> = playlist
            .iter()
            .filter(|item| !item.name.starts_with('!'))
            .cloned()
            .collect();
        Self::convert(&enabled, start_ts)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn item(name: &str, frames: u64, fps: f64) -> SchedulePlaylistItem {
        SchedulePlaylistItem {
            name: name.to_string(),
            duration_frames: Some(frames),
            fps,
        }
    }

    fn live_item(name: &str) -> SchedulePlaylistItem {
        SchedulePlaylistItem {
            name: name.to_string(),
            duration_frames: None,
            fps: 25.0,
        }
    }

    // ── convert ──────────────────────────────────────────────────────────────

    #[test]
    fn test_convert_empty_playlist() {
        let result = PlaylistToSchedule::convert(&[], 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_convert_single_item() {
        let playlist = vec![item("Clip A", 250, 25.0)]; // 10 s
        let result = PlaylistToSchedule::convert(&playlist, 1_000_000);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].start_ts_ms, 1_000_000);
        assert_eq!(result[0].duration_ms, 10_000);
        assert_eq!(result[0].end_ts_ms, 1_010_000);
    }

    #[test]
    fn test_convert_sequential_items() {
        // 2 clips, each 25 frames @ 25 fps = 1 s each
        let playlist = vec![item("A", 25, 25.0), item("B", 25, 25.0)];
        let result = PlaylistToSchedule::convert(&playlist, 0);
        assert_eq!(result[0].start_ts_ms, 0);
        assert_eq!(result[0].end_ts_ms, 1_000);
        assert_eq!(result[1].start_ts_ms, 1_000);
        assert_eq!(result[1].end_ts_ms, 2_000);
    }

    #[test]
    fn test_convert_preserves_index() {
        let playlist = vec![item("X", 50, 25.0), item("Y", 50, 25.0), item("Z", 50, 25.0)];
        let result = PlaylistToSchedule::convert(&playlist, 0);
        for (i, s) in result.iter().enumerate() {
            assert_eq!(s.index, i);
        }
    }

    #[test]
    fn test_convert_preserves_title() {
        let playlist = vec![item("My Clip", 100, 25.0)];
        let result = PlaylistToSchedule::convert(&playlist, 0);
        assert_eq!(result[0].title, "My Clip");
    }

    #[test]
    fn test_convert_live_item_does_not_advance_clock() {
        let playlist = vec![item("Pre", 25, 25.0), live_item("Live"), item("Post", 25, 25.0)];
        let result = PlaylistToSchedule::convert(&playlist, 0);
        // Pre ends at 1000, Live starts at 1000 with 0 duration, Post also starts at 1000
        assert_eq!(result[1].start_ts_ms, 1_000);
        assert_eq!(result[1].duration_ms, 0);
        assert_eq!(result[2].start_ts_ms, 1_000);
    }

    #[test]
    fn test_convert_has_duration_false_for_live() {
        let playlist = vec![live_item("Live")];
        let result = PlaylistToSchedule::convert(&playlist, 0);
        assert!(!result[0].has_duration());
    }

    #[test]
    fn test_convert_has_duration_true_for_clip() {
        let playlist = vec![item("Clip", 25, 25.0)];
        let result = PlaylistToSchedule::convert(&playlist, 0);
        assert!(result[0].has_duration());
    }

    #[test]
    fn test_convert_zero_fps_defaults_to_25() {
        let it = SchedulePlaylistItem {
            name: "Test".to_string(),
            duration_frames: Some(25),
            fps: 0.0,
        };
        assert_eq!(it.duration_ms(), 1_000);
    }

    #[test]
    fn test_convert_negative_fps_defaults_to_25() {
        let it = SchedulePlaylistItem {
            name: "Test".to_string(),
            duration_frames: Some(25),
            fps: -10.0,
        };
        assert_eq!(it.duration_ms(), 1_000);
    }

    // ── convert_enabled ───────────────────────────────────────────────────────

    #[test]
    fn test_convert_enabled_filters_disabled() {
        let playlist = vec![item("A", 25, 25.0), item("!Disabled", 25, 25.0), item("B", 25, 25.0)];
        let result = PlaylistToSchedule::convert_enabled(&playlist, 0);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].title, "A");
        assert_eq!(result[1].title, "B");
    }

    #[test]
    fn test_convert_enabled_all_enabled() {
        let playlist = vec![item("A", 25, 25.0), item("B", 25, 25.0)];
        let result = PlaylistToSchedule::convert_enabled(&playlist, 0);
        assert_eq!(result.len(), 2);
    }

    // ── SchedulePlaylistItem::duration_ms ────────────────────────────────────

    #[test]
    fn test_duration_ms_30fps() {
        let it = SchedulePlaylistItem {
            name: "T".to_string(),
            duration_frames: Some(30),
            fps: 30.0,
        };
        assert_eq!(it.duration_ms(), 1_000);
    }

    #[test]
    fn test_duration_ms_none() {
        let it = live_item("L");
        assert_eq!(it.duration_ms(), 0);
    }
}
