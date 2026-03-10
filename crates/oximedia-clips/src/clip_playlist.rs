#![allow(dead_code)]
//! Playlist management for organized clip playback.
//!
//! This module provides playlist creation, ordering, shuffling, and
//! repeat-mode management for video clips. It supports named playlists
//! with metadata, duration tracking, and insertion/removal operations.

/// Unique identifier for a playlist.
pub type PlaylistId = u64;

/// Repeat mode for playlist playback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepeatMode {
    /// Play once and stop.
    None,
    /// Repeat the entire playlist.
    RepeatAll,
    /// Repeat the current clip.
    RepeatOne,
}

/// A single item in a playlist.
#[derive(Debug, Clone, PartialEq)]
pub struct PlaylistItem {
    /// Clip identifier or name.
    pub clip_id: String,
    /// Display title.
    pub title: String,
    /// Duration in milliseconds.
    pub duration_ms: u64,
    /// Start offset within the clip in milliseconds.
    pub start_offset_ms: u64,
    /// End offset within the clip in milliseconds (0 = full clip).
    pub end_offset_ms: u64,
}

impl PlaylistItem {
    /// Creates a new playlist item.
    pub fn new(clip_id: &str, title: &str, duration_ms: u64) -> Self {
        Self {
            clip_id: clip_id.to_string(),
            title: title.to_string(),
            duration_ms,
            start_offset_ms: 0,
            end_offset_ms: 0,
        }
    }

    /// Creates an item with custom in/out points.
    pub fn with_range(clip_id: &str, title: &str, duration_ms: u64, start: u64, end: u64) -> Self {
        Self {
            clip_id: clip_id.to_string(),
            title: title.to_string(),
            duration_ms,
            start_offset_ms: start,
            end_offset_ms: end,
        }
    }

    /// Returns the effective playback duration.
    pub fn effective_duration(&self) -> u64 {
        if self.end_offset_ms > self.start_offset_ms {
            self.end_offset_ms - self.start_offset_ms
        } else {
            self.duration_ms.saturating_sub(self.start_offset_ms)
        }
    }
}

/// A playlist containing ordered clip items.
#[derive(Debug, Clone)]
pub struct Playlist {
    /// Unique playlist ID.
    pub id: PlaylistId,
    /// Playlist name.
    pub name: String,
    /// Description.
    pub description: String,
    /// Items in the playlist.
    items: Vec<PlaylistItem>,
    /// Current playback index.
    current_index: usize,
    /// Repeat mode.
    repeat_mode: RepeatMode,
}

impl Playlist {
    /// Creates a new empty playlist.
    pub fn new(id: PlaylistId, name: &str) -> Self {
        Self {
            id,
            name: name.to_string(),
            description: String::new(),
            items: Vec::new(),
            current_index: 0,
            repeat_mode: RepeatMode::None,
        }
    }

    /// Sets the description.
    pub fn set_description(&mut self, desc: &str) {
        self.description = desc.to_string();
    }

    /// Sets the repeat mode.
    pub fn set_repeat_mode(&mut self, mode: RepeatMode) {
        self.repeat_mode = mode;
    }

    /// Returns the current repeat mode.
    pub fn repeat_mode(&self) -> RepeatMode {
        self.repeat_mode
    }

    /// Appends an item to the end of the playlist.
    pub fn push(&mut self, item: PlaylistItem) {
        self.items.push(item);
    }

    /// Inserts an item at a specific position.
    pub fn insert(&mut self, index: usize, item: PlaylistItem) {
        let pos = index.min(self.items.len());
        self.items.insert(pos, item);
        // Adjust current index if needed
        if pos <= self.current_index && !self.items.is_empty() {
            self.current_index += 1;
        }
    }

    /// Removes the item at the given index.
    pub fn remove(&mut self, index: usize) -> Option<PlaylistItem> {
        if index < self.items.len() {
            let item = self.items.remove(index);
            if self.current_index >= self.items.len() && !self.items.is_empty() {
                self.current_index = self.items.len() - 1;
            }
            Some(item)
        } else {
            None
        }
    }

    /// Returns the number of items.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Returns true if the playlist is empty.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Returns the total duration of all items in milliseconds.
    pub fn total_duration_ms(&self) -> u64 {
        self.items.iter().map(|i| i.effective_duration()).sum()
    }

    /// Returns a reference to the current item.
    pub fn current(&self) -> Option<&PlaylistItem> {
        self.items.get(self.current_index)
    }

    /// Returns the current playback index.
    pub fn current_index(&self) -> usize {
        self.current_index
    }

    /// Advances to the next item. Returns `true` if playback continues.
    pub fn next(&mut self) -> bool {
        if self.items.is_empty() {
            return false;
        }
        match self.repeat_mode {
            RepeatMode::RepeatOne => true,
            RepeatMode::RepeatAll => {
                self.current_index = (self.current_index + 1) % self.items.len();
                true
            }
            RepeatMode::None => {
                if self.current_index + 1 < self.items.len() {
                    self.current_index += 1;
                    true
                } else {
                    false
                }
            }
        }
    }

    /// Goes to the previous item. Returns `true` if successful.
    pub fn previous(&mut self) -> bool {
        if self.items.is_empty() {
            return false;
        }
        match self.repeat_mode {
            RepeatMode::RepeatOne => true,
            RepeatMode::RepeatAll => {
                if self.current_index == 0 {
                    self.current_index = self.items.len() - 1;
                } else {
                    self.current_index -= 1;
                }
                true
            }
            RepeatMode::None => {
                if self.current_index > 0 {
                    self.current_index -= 1;
                    true
                } else {
                    false
                }
            }
        }
    }

    /// Jumps to a specific index. Returns `true` if valid.
    pub fn jump_to(&mut self, index: usize) -> bool {
        if index < self.items.len() {
            self.current_index = index;
            true
        } else {
            false
        }
    }

    /// Resets playback to the beginning.
    pub fn reset(&mut self) {
        self.current_index = 0;
    }

    /// Moves an item from one position to another.
    pub fn move_item(&mut self, from: usize, to: usize) -> bool {
        if from >= self.items.len() || to >= self.items.len() {
            return false;
        }
        let item = self.items.remove(from);
        self.items.insert(to, item);
        true
    }

    /// Reverses the order of items.
    pub fn reverse(&mut self) {
        self.items.reverse();
        if !self.items.is_empty() {
            self.current_index = 0;
        }
    }

    /// Returns a reference to the items slice.
    pub fn items(&self) -> &[PlaylistItem] {
        &self.items
    }

    /// Finds items matching a title substring.
    pub fn find_by_title(&self, query: &str) -> Vec<usize> {
        let query_lower = query.to_lowercase();
        self.items
            .iter()
            .enumerate()
            .filter(|(_, item)| item.title.to_lowercase().contains(&query_lower))
            .map(|(i, _)| i)
            .collect()
    }

    /// Clears all items and resets playback.
    pub fn clear(&mut self) {
        self.items.clear();
        self.current_index = 0;
    }
}

/// A collection of playlists.
#[derive(Debug)]
pub struct PlaylistCollection {
    /// All playlists in the collection.
    playlists: Vec<Playlist>,
    /// Next ID to assign.
    next_id: PlaylistId,
}

impl PlaylistCollection {
    /// Creates a new empty collection.
    pub fn new() -> Self {
        Self {
            playlists: Vec::new(),
            next_id: 1,
        }
    }

    /// Creates a new playlist and returns its ID.
    pub fn create_playlist(&mut self, name: &str) -> PlaylistId {
        let id = self.next_id;
        self.next_id += 1;
        self.playlists.push(Playlist::new(id, name));
        id
    }

    /// Returns a reference to a playlist by ID.
    pub fn get(&self, id: PlaylistId) -> Option<&Playlist> {
        self.playlists.iter().find(|p| p.id == id)
    }

    /// Returns a mutable reference to a playlist by ID.
    pub fn get_mut(&mut self, id: PlaylistId) -> Option<&mut Playlist> {
        self.playlists.iter_mut().find(|p| p.id == id)
    }

    /// Returns the number of playlists.
    pub fn count(&self) -> usize {
        self.playlists.len()
    }
}

impl Default for PlaylistCollection {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_playlist_item_new() {
        let item = PlaylistItem::new("c1", "Intro", 5000);
        assert_eq!(item.clip_id, "c1");
        assert_eq!(item.title, "Intro");
        assert_eq!(item.duration_ms, 5000);
    }

    #[test]
    fn test_playlist_item_effective_duration() {
        let item = PlaylistItem::new("c1", "Full", 10000);
        assert_eq!(item.effective_duration(), 10000);

        let ranged = PlaylistItem::with_range("c2", "Partial", 10000, 2000, 7000);
        assert_eq!(ranged.effective_duration(), 5000);
    }

    #[test]
    fn test_playlist_item_effective_duration_with_start_only() {
        let mut item = PlaylistItem::new("c1", "Trimmed", 10000);
        item.start_offset_ms = 3000;
        assert_eq!(item.effective_duration(), 7000);
    }

    #[test]
    fn test_playlist_new() {
        let pl = Playlist::new(1, "My Playlist");
        assert_eq!(pl.id, 1);
        assert_eq!(pl.name, "My Playlist");
        assert!(pl.is_empty());
        assert_eq!(pl.len(), 0);
    }

    #[test]
    fn test_playlist_push_and_len() {
        let mut pl = Playlist::new(1, "Test");
        pl.push(PlaylistItem::new("c1", "A", 1000));
        pl.push(PlaylistItem::new("c2", "B", 2000));
        assert_eq!(pl.len(), 2);
        assert!(!pl.is_empty());
    }

    #[test]
    fn test_playlist_total_duration() {
        let mut pl = Playlist::new(1, "Test");
        pl.push(PlaylistItem::new("c1", "A", 3000));
        pl.push(PlaylistItem::new("c2", "B", 4000));
        assert_eq!(pl.total_duration_ms(), 7000);
    }

    #[test]
    fn test_playlist_navigation_none() {
        let mut pl = Playlist::new(1, "Nav");
        pl.push(PlaylistItem::new("c1", "A", 1000));
        pl.push(PlaylistItem::new("c2", "B", 1000));
        pl.push(PlaylistItem::new("c3", "C", 1000));

        assert_eq!(pl.current_index(), 0);
        assert!(pl.next());
        assert_eq!(pl.current_index(), 1);
        assert!(pl.next());
        assert_eq!(pl.current_index(), 2);
        assert!(!pl.next()); // end
    }

    #[test]
    fn test_playlist_navigation_repeat_all() {
        let mut pl = Playlist::new(1, "Rep");
        pl.set_repeat_mode(RepeatMode::RepeatAll);
        pl.push(PlaylistItem::new("c1", "A", 1000));
        pl.push(PlaylistItem::new("c2", "B", 1000));

        assert!(pl.next()); // 0->1
        assert!(pl.next()); // 1->0 (wrap)
        assert_eq!(pl.current_index(), 0);
    }

    #[test]
    fn test_playlist_navigation_repeat_one() {
        let mut pl = Playlist::new(1, "One");
        pl.set_repeat_mode(RepeatMode::RepeatOne);
        pl.push(PlaylistItem::new("c1", "A", 1000));
        assert!(pl.next());
        assert_eq!(pl.current_index(), 0); // stays
    }

    #[test]
    fn test_playlist_previous() {
        let mut pl = Playlist::new(1, "Prev");
        pl.push(PlaylistItem::new("c1", "A", 1000));
        pl.push(PlaylistItem::new("c2", "B", 1000));
        pl.jump_to(1);
        assert!(pl.previous());
        assert_eq!(pl.current_index(), 0);
        assert!(!pl.previous()); // can't go before 0
    }

    #[test]
    fn test_playlist_remove() {
        let mut pl = Playlist::new(1, "Del");
        pl.push(PlaylistItem::new("c1", "A", 1000));
        pl.push(PlaylistItem::new("c2", "B", 1000));
        let removed = pl.remove(0);
        assert!(removed.is_some());
        assert_eq!(removed.expect("value should be valid").clip_id, "c1");
        assert_eq!(pl.len(), 1);
    }

    #[test]
    fn test_playlist_move_item() {
        let mut pl = Playlist::new(1, "Mv");
        pl.push(PlaylistItem::new("c1", "A", 1000));
        pl.push(PlaylistItem::new("c2", "B", 1000));
        pl.push(PlaylistItem::new("c3", "C", 1000));
        assert!(pl.move_item(0, 2));
        assert_eq!(pl.items()[0].clip_id, "c2");
        assert_eq!(pl.items()[2].clip_id, "c1");
    }

    #[test]
    fn test_playlist_find_by_title() {
        let mut pl = Playlist::new(1, "Search");
        pl.push(PlaylistItem::new("c1", "Interview Part 1", 1000));
        pl.push(PlaylistItem::new("c2", "B-Roll", 1000));
        pl.push(PlaylistItem::new("c3", "Interview Part 2", 1000));
        let matches = pl.find_by_title("interview");
        assert_eq!(matches.len(), 2);
        assert_eq!(matches[0], 0);
        assert_eq!(matches[1], 2);
    }

    #[test]
    fn test_playlist_collection() {
        let mut col = PlaylistCollection::new();
        let id1 = col.create_playlist("Favorites");
        let id2 = col.create_playlist("Recent");
        assert_eq!(col.count(), 2);
        assert!(col.get(id1).is_some());
        assert!(col.get(id2).is_some());
    }

    #[test]
    fn test_playlist_clear() {
        let mut pl = Playlist::new(1, "Clr");
        pl.push(PlaylistItem::new("c1", "A", 1000));
        pl.push(PlaylistItem::new("c2", "B", 2000));
        pl.clear();
        assert!(pl.is_empty());
        assert_eq!(pl.current_index(), 0);
    }
}
