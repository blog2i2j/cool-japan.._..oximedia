//! Quick-access favourite collections and recent clips list.
//!
//! This module provides two complementary features:
//!
//! 1. **`FavoriteCollection`** — a named, ordered set of clip IDs that the
//!    user has pinned for quick access.  Collections support manual ordering
//!    and per-clip notes.
//!
//! 2. **`RecentClipList`** — a bounded history of recently accessed clip IDs
//!    (LRU-style: adding a clip that already exists moves it to the front).
//!
//! # Example
//!
//! ```rust
//! use oximedia_clips::clip_favorites::{FavoriteCollection, RecentClipList};
//! use oximedia_clips::ClipId;
//! use uuid::Uuid;
//!
//! let id_a = ClipId::from_uuid(Uuid::nil());
//!
//! // Favourite collection
//! let mut fav = FavoriteCollection::new("My Picks".to_string());
//! fav.add(id_a);
//! assert_eq!(fav.len(), 1);
//!
//! // Recent list
//! let mut recent = RecentClipList::new(5);
//! recent.record_access(id_a);
//! assert_eq!(recent.most_recent(), Some(id_a));
//! ```

#![allow(dead_code)]

use crate::clip::ClipId;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

// ─────────────────────────────────────────────────────────────────────────────
// FavoriteEntry
// ─────────────────────────────────────────────────────────────────────────────

/// A single entry in a favourite collection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FavoriteEntry {
    /// The clip that has been favourited.
    pub clip_id: ClipId,
    /// When the clip was added to this collection.
    pub added_at: DateTime<Utc>,
    /// Optional user-supplied note attached to this favourite entry.
    pub note: Option<String>,
    /// Sort order within the collection (lower = first).  Defaults to the
    /// insertion timestamp millis; can be manually overridden.
    pub sort_order: i64,
}

impl FavoriteEntry {
    fn new(clip_id: ClipId) -> Self {
        let now = Utc::now();
        Self {
            clip_id,
            added_at: now,
            note: None,
            sort_order: now.timestamp_millis(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FavoriteCollection
// ─────────────────────────────────────────────────────────────────────────────

/// A named, ordered collection of favourite clips.
///
/// Duplicate clip IDs are rejected (a clip can only appear once in a
/// collection).  The collection can hold any number of clips; there is no
/// capacity limit (unlike `RecentClipList`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FavoriteCollection {
    /// Display name of this collection.
    pub name: String,
    /// Optional description.
    pub description: Option<String>,
    /// Entries keyed by `ClipId` for O(1) lookup.
    entries: HashMap<ClipId, FavoriteEntry>,
    /// Ordered list of clip IDs (reflects `sort_order`).
    order: Vec<ClipId>,
    /// When the collection was created.
    pub created_at: DateTime<Utc>,
    /// When the collection was last modified.
    pub modified_at: DateTime<Utc>,
}

impl FavoriteCollection {
    /// Create a new empty collection.
    #[must_use]
    pub fn new(name: String) -> Self {
        let now = Utc::now();
        Self {
            name,
            description: None,
            entries: HashMap::new(),
            order: Vec::new(),
            created_at: now,
            modified_at: now,
        }
    }

    /// Add a clip to the collection.
    ///
    /// Returns `true` if the clip was added, `false` if it was already present.
    pub fn add(&mut self, clip_id: ClipId) -> bool {
        if self.entries.contains_key(&clip_id) {
            return false;
        }
        let entry = FavoriteEntry::new(clip_id);
        self.order.push(clip_id);
        self.entries.insert(clip_id, entry);
        self.modified_at = Utc::now();
        true
    }

    /// Add a clip with an optional note.
    pub fn add_with_note(&mut self, clip_id: ClipId, note: impl Into<String>) -> bool {
        let note_str = note.into();
        if self.entries.contains_key(&clip_id) {
            return false;
        }
        let mut entry = FavoriteEntry::new(clip_id);
        entry.note = Some(note_str);
        self.order.push(clip_id);
        self.entries.insert(clip_id, entry);
        self.modified_at = Utc::now();
        true
    }

    /// Remove a clip from the collection.
    ///
    /// Returns `true` if the clip was present and removed.
    pub fn remove(&mut self, clip_id: &ClipId) -> bool {
        if self.entries.remove(clip_id).is_some() {
            self.order.retain(|id| id != clip_id);
            self.modified_at = Utc::now();
            true
        } else {
            false
        }
    }

    /// Check whether a clip is in this collection.
    #[must_use]
    pub fn contains(&self, clip_id: &ClipId) -> bool {
        self.entries.contains_key(clip_id)
    }

    /// Number of clips in this collection.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the collection is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Clip IDs in sorted order.
    #[must_use]
    pub fn clip_ids(&self) -> Vec<ClipId> {
        let mut ordered: Vec<ClipId> = self.order.clone();
        ordered.sort_by_key(|id| {
            self.entries
                .get(id)
                .map(|e| e.sort_order)
                .unwrap_or(i64::MAX)
        });
        ordered
    }

    /// Retrieve the entry for a clip, if present.
    #[must_use]
    pub fn entry(&self, clip_id: &ClipId) -> Option<&FavoriteEntry> {
        self.entries.get(clip_id)
    }

    /// Set the note for an existing favourite entry.
    ///
    /// Returns `false` if the clip is not in this collection.
    pub fn set_note(&mut self, clip_id: &ClipId, note: impl Into<String>) -> bool {
        if let Some(entry) = self.entries.get_mut(clip_id) {
            entry.note = Some(note.into());
            self.modified_at = Utc::now();
            true
        } else {
            false
        }
    }

    /// Move a clip to a specific position in the manual order (0-based).
    ///
    /// Does nothing if the clip is not present or the index is out of range.
    pub fn move_to(&mut self, clip_id: &ClipId, position: usize) {
        let current_pos = match self.order.iter().position(|id| id == clip_id) {
            Some(p) => p,
            None => return,
        };
        if position >= self.order.len() {
            return;
        }
        self.order.remove(current_pos);
        self.order.insert(position, *clip_id);
        // Update sort_order values
        for (i, id) in self.order.iter().enumerate() {
            if let Some(entry) = self.entries.get_mut(id) {
                entry.sort_order = i as i64;
            }
        }
        self.modified_at = Utc::now();
    }

    /// Clear all clips from the collection.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.order.clear();
        self.modified_at = Utc::now();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RecentClipList
// ─────────────────────────────────────────────────────────────────────────────

/// A bounded LRU-style recent-clips list.
///
/// Each time `record_access` is called for a clip the clip is moved to the
/// front of the list.  The list is bounded to `capacity` entries; older clips
/// are evicted from the tail when the list is full.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecentClipList {
    /// Maximum number of clips retained.
    capacity: usize,
    /// Ordered list of clip IDs, most-recent first.
    list: VecDeque<ClipId>,
    /// Access timestamps, keyed by clip ID.
    access_times: HashMap<ClipId, DateTime<Utc>>,
}

impl RecentClipList {
    /// Create a new recent-clips list with the given capacity.
    ///
    /// `capacity` must be at least 1; values of 0 are clamped to 1.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            list: VecDeque::new(),
            access_times: HashMap::new(),
        }
    }

    /// Record an access to a clip (moves it to the front).
    ///
    /// If the clip is already in the list it is moved to the front. If the
    /// list is full the oldest (tail) entry is evicted.
    pub fn record_access(&mut self, clip_id: ClipId) {
        // If already in list, remove from current position.
        if let Some(pos) = self.list.iter().position(|id| *id == clip_id) {
            self.list.remove(pos);
        } else {
            // Evict tail if at capacity.
            while self.list.len() >= self.capacity {
                if let Some(evicted) = self.list.pop_back() {
                    self.access_times.remove(&evicted);
                }
            }
        }
        self.list.push_front(clip_id);
        self.access_times.insert(clip_id, Utc::now());
    }

    /// The most-recently accessed clip, if any.
    #[must_use]
    pub fn most_recent(&self) -> Option<ClipId> {
        self.list.front().copied()
    }

    /// Ordered list of clip IDs, most-recent first.
    #[must_use]
    pub fn clip_ids(&self) -> Vec<ClipId> {
        self.list.iter().copied().collect()
    }

    /// When a clip was last accessed, if tracked.
    #[must_use]
    pub fn last_access(&self, clip_id: &ClipId) -> Option<DateTime<Utc>> {
        self.access_times.get(clip_id).copied()
    }

    /// Number of clips currently tracked.
    #[must_use]
    pub fn len(&self) -> usize {
        self.list.len()
    }

    /// Whether the list is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.list.is_empty()
    }

    /// Remove a clip from the list.
    pub fn remove(&mut self, clip_id: &ClipId) {
        if let Some(pos) = self.list.iter().position(|id| id == clip_id) {
            self.list.remove(pos);
            self.access_times.remove(clip_id);
        }
    }

    /// Clear the entire list.
    pub fn clear(&mut self) {
        self.list.clear();
        self.access_times.clear();
    }

    /// The configured capacity.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FavoritesManager
// ─────────────────────────────────────────────────────────────────────────────

/// A top-level manager holding multiple named favourite collections and a
/// single recent-clips list.
///
/// This is the main entry point for production use; individual collection and
/// recent-list types are also useful standalone.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FavoritesManager {
    /// Named favourite collections.
    pub collections: HashMap<String, FavoriteCollection>,
    /// Recent clips list.
    pub recent: RecentClipList,
}

impl FavoritesManager {
    /// Create a new manager with a recent-list capacity.
    #[must_use]
    pub fn new(recent_capacity: usize) -> Self {
        Self {
            collections: HashMap::new(),
            recent: RecentClipList::new(recent_capacity),
        }
    }

    /// Create a new named favourite collection.
    ///
    /// Returns `false` if a collection with that name already exists.
    pub fn create_collection(&mut self, name: String) -> bool {
        if self.collections.contains_key(&name) {
            return false;
        }
        self.collections
            .insert(name.clone(), FavoriteCollection::new(name));
        true
    }

    /// Remove a named collection. Returns `false` if not found.
    pub fn remove_collection(&mut self, name: &str) -> bool {
        self.collections.remove(name).is_some()
    }

    /// Add a clip to a named collection (creating the collection if needed).
    pub fn add_to_collection(&mut self, collection: &str, clip_id: ClipId) {
        self.collections
            .entry(collection.to_string())
            .or_insert_with(|| FavoriteCollection::new(collection.to_string()))
            .add(clip_id);
    }

    /// Record a clip access (updates the recent list).
    pub fn record_access(&mut self, clip_id: ClipId) {
        self.recent.record_access(clip_id);
    }

    /// All collection names.
    #[must_use]
    pub fn collection_names(&self) -> Vec<&str> {
        self.collections.keys().map(String::as_str).collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn id(n: u8) -> ClipId {
        let mut bytes = [0u8; 16];
        bytes[15] = n;
        ClipId::from_uuid(Uuid::from_bytes(bytes))
    }

    // ─── FavoriteCollection ──────────────────────────────────────────────────

    #[test]
    fn test_fav_add_and_contains() {
        let mut col = FavoriteCollection::new("Test".to_string());
        assert!(col.add(id(1)));
        assert!(col.contains(&id(1)));
        assert!(!col.contains(&id(2)));
    }

    #[test]
    fn test_fav_no_duplicates() {
        let mut col = FavoriteCollection::new("Test".to_string());
        assert!(col.add(id(1)));
        assert!(!col.add(id(1))); // duplicate
        assert_eq!(col.len(), 1);
    }

    #[test]
    fn test_fav_remove() {
        let mut col = FavoriteCollection::new("Test".to_string());
        col.add(id(1));
        assert!(col.remove(&id(1)));
        assert!(!col.contains(&id(1)));
        assert_eq!(col.len(), 0);
    }

    #[test]
    fn test_fav_set_note() {
        let mut col = FavoriteCollection::new("Test".to_string());
        col.add(id(1));
        assert!(col.set_note(&id(1), "Great shot"));
        assert_eq!(col.entry(&id(1)).and_then(|e| e.note.as_deref()), Some("Great shot"));
    }

    #[test]
    fn test_fav_move_to() {
        let mut col = FavoriteCollection::new("Test".to_string());
        col.add(id(1));
        col.add(id(2));
        col.add(id(3));
        col.move_to(&id(3), 0);
        let ids = col.clip_ids();
        assert_eq!(ids[0], id(3));
    }

    #[test]
    fn test_fav_clip_ids_sorted() {
        let mut col = FavoriteCollection::new("Test".to_string());
        col.add(id(1));
        col.add(id(2));
        col.add(id(3));
        let ids = col.clip_ids();
        assert_eq!(ids.len(), 3);
    }

    #[test]
    fn test_fav_with_note() {
        let mut col = FavoriteCollection::new("T".to_string());
        col.add_with_note(id(5), "B-roll pick");
        assert_eq!(
            col.entry(&id(5)).and_then(|e| e.note.as_deref()),
            Some("B-roll pick")
        );
    }

    // ─── RecentClipList ──────────────────────────────────────────────────────

    #[test]
    fn test_recent_lru_most_recent() {
        let mut r = RecentClipList::new(5);
        r.record_access(id(1));
        r.record_access(id(2));
        assert_eq!(r.most_recent(), Some(id(2)));
    }

    #[test]
    fn test_recent_move_to_front_on_re_access() {
        let mut r = RecentClipList::new(5);
        r.record_access(id(1));
        r.record_access(id(2));
        r.record_access(id(1)); // id(1) should now be at front
        assert_eq!(r.most_recent(), Some(id(1)));
        assert_eq!(r.len(), 2);
    }

    #[test]
    fn test_recent_eviction_at_capacity() {
        let mut r = RecentClipList::new(3);
        r.record_access(id(1));
        r.record_access(id(2));
        r.record_access(id(3));
        r.record_access(id(4)); // evicts id(1)
        assert_eq!(r.len(), 3);
        assert!(!r.clip_ids().contains(&id(1)));
    }

    #[test]
    fn test_recent_remove() {
        let mut r = RecentClipList::new(5);
        r.record_access(id(1));
        r.remove(&id(1));
        assert!(r.is_empty());
    }

    #[test]
    fn test_recent_capacity_one() {
        let mut r = RecentClipList::new(1);
        r.record_access(id(1));
        r.record_access(id(2));
        assert_eq!(r.len(), 1);
        assert_eq!(r.most_recent(), Some(id(2)));
    }

    // ─── FavoritesManager ────────────────────────────────────────────────────

    #[test]
    fn test_manager_create_and_remove_collection() {
        let mut mgr = FavoritesManager::new(10);
        assert!(mgr.create_collection("picks".to_string()));
        assert!(!mgr.create_collection("picks".to_string())); // duplicate
        assert!(mgr.remove_collection("picks"));
        assert!(!mgr.remove_collection("picks")); // not found
    }

    #[test]
    fn test_manager_add_to_and_record() {
        let mut mgr = FavoritesManager::new(5);
        mgr.add_to_collection("highlights", id(1));
        mgr.record_access(id(1));
        assert_eq!(mgr.recent.most_recent(), Some(id(1)));
        let col = mgr.collections.get("highlights").expect("collection exists");
        assert!(col.contains(&id(1)));
    }
}
