//! Review playlists: organize multiple clips into sequential review sessions.
//!
//! A [`ReviewPlaylist`] groups a set of content items (identified by their
//! `content_id` strings) into an ordered sequence for batch reviewing.
//! Reviewers step through the playlist item by item; each item can carry its
//! own set of annotations and approval state without affecting the others.

use serde::{Deserialize, Serialize};

// ─── PlaylistItem ──────────────────────────────────────────────────────────

/// A single item in a review playlist.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlaylistItem {
    /// Position in the playlist (0-indexed).
    pub index: usize,
    /// Content identifier (e.g. an asset ID or file path).
    pub content_id: String,
    /// Human-readable label shown in the playlist UI.
    pub label: String,
    /// Optional duration in milliseconds (for time-based media).
    pub duration_ms: Option<u64>,
    /// Whether this item has been reviewed.
    pub reviewed: bool,
}

impl PlaylistItem {
    /// Create a new playlist item.
    #[must_use]
    pub fn new(index: usize, content_id: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            index,
            content_id: content_id.into(),
            label: label.into(),
            duration_ms: None,
            reviewed: false,
        }
    }

    /// Set the media duration (builder pattern).
    #[must_use]
    pub fn with_duration_ms(mut self, ms: u64) -> Self {
        self.duration_ms = Some(ms);
        self
    }
}

// ─── ReviewPlaylist ───────────────────────────────────────────────────────────

/// An ordered list of content items for sequential review.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewPlaylist {
    /// Unique playlist identifier.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Ordered items in the playlist.
    pub items: Vec<PlaylistItem>,
    /// Index of the currently active item.
    pub current_index: usize,
}

impl ReviewPlaylist {
    /// Create a new empty playlist.
    #[must_use]
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            items: Vec::new(),
            current_index: 0,
        }
    }

    /// Append an item to the playlist.
    pub fn push(&mut self, content_id: impl Into<String>, label: impl Into<String>) {
        let index = self.items.len();
        self.items.push(PlaylistItem::new(index, content_id, label));
    }

    /// Number of items in the playlist.
    #[must_use]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Returns `true` if the playlist contains no items.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Get the currently active item, if any.
    #[must_use]
    pub fn current(&self) -> Option<&PlaylistItem> {
        self.items.get(self.current_index)
    }

    /// Advance to the next item.  Returns `Some(&item)` or `None` if already at the end.
    pub fn advance(&mut self) -> Option<&PlaylistItem> {
        if self.current_index + 1 < self.items.len() {
            self.current_index += 1;
            self.items.get(self.current_index)
        } else {
            None
        }
    }

    /// Move to the previous item.  Returns `Some(&item)` or `None` if already at the start.
    pub fn rewind(&mut self) -> Option<&PlaylistItem> {
        if self.current_index > 0 {
            self.current_index -= 1;
            self.items.get(self.current_index)
        } else {
            None
        }
    }

    /// Mark the current item as reviewed.
    pub fn mark_current_reviewed(&mut self) {
        if let Some(item) = self.items.get_mut(self.current_index) {
            item.reviewed = true;
        }
    }

    /// Count how many items have been reviewed.
    #[must_use]
    pub fn reviewed_count(&self) -> usize {
        self.items.iter().filter(|i| i.reviewed).count()
    }

    /// Returns `true` if all items have been reviewed.
    #[must_use]
    pub fn all_reviewed(&self) -> bool {
        !self.items.is_empty() && self.items.iter().all(|i| i.reviewed)
    }

    /// Total duration of all items in milliseconds.
    #[must_use]
    pub fn total_duration_ms(&self) -> u64 {
        self.items.iter().filter_map(|i| i.duration_ms).sum()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_playlist() -> ReviewPlaylist {
        let mut p = ReviewPlaylist::new("pl-1", "Daily Rushes");
        p.push("clip-01", "Scene 01");
        p.push("clip-02", "Scene 02");
        p.push("clip-03", "Scene 03");
        p
    }

    #[test]
    fn playlist_len() {
        let p = make_playlist();
        assert_eq!(p.len(), 3);
        assert!(!p.is_empty());
    }

    #[test]
    fn playlist_empty() {
        let p = ReviewPlaylist::new("x", "y");
        assert!(p.is_empty());
        assert_eq!(p.len(), 0);
    }

    #[test]
    fn playlist_current_starts_at_zero() {
        let p = make_playlist();
        assert_eq!(p.current().map(|i| i.index), Some(0));
    }

    #[test]
    fn playlist_advance() {
        let mut p = make_playlist();
        let item = p.advance().expect("advance to 1");
        assert_eq!(item.index, 1);
    }

    #[test]
    fn playlist_advance_past_end_returns_none() {
        let mut p = make_playlist();
        p.advance();
        p.advance();
        assert!(p.advance().is_none());
    }

    #[test]
    fn playlist_rewind() {
        let mut p = make_playlist();
        p.advance();
        let item = p.rewind().expect("rewind");
        assert_eq!(item.index, 0);
    }

    #[test]
    fn playlist_rewind_at_start_returns_none() {
        let mut p = make_playlist();
        assert!(p.rewind().is_none());
    }

    #[test]
    fn mark_current_reviewed() {
        let mut p = make_playlist();
        p.mark_current_reviewed();
        assert!(p.items[0].reviewed);
        assert_eq!(p.reviewed_count(), 1);
    }

    #[test]
    fn all_reviewed() {
        let mut p = make_playlist();
        assert!(!p.all_reviewed());
        for _ in 0..p.len() {
            p.mark_current_reviewed();
            p.advance();
        }
        assert!(p.all_reviewed());
    }

    #[test]
    fn total_duration_ms() {
        let mut p = ReviewPlaylist::new("x", "y");
        p.push("a", "A");
        p.items[0].duration_ms = Some(1000);
        p.push("b", "B");
        p.items[1].duration_ms = Some(2000);
        assert_eq!(p.total_duration_ms(), 3000);
    }
}
