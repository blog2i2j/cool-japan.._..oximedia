#![allow(dead_code)]
//! VOD (Video on Demand) management for recorded game sessions.
//!
//! Manages recorded game sessions as VOD assets, including chapter markers,
//! highlights, metadata tagging, thumbnail generation, and export to various
//! platforms. Supports splitting long sessions into episodes and attaching
//! game-specific metadata.

use std::collections::HashMap;

/// Unique identifier for a VOD asset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VodId(pub u64);

/// A chapter marker within a VOD.
#[derive(Debug, Clone, PartialEq)]
pub struct Chapter {
    /// Chapter title.
    pub title: String,
    /// Start time in seconds from the beginning of the VOD.
    pub start_secs: f64,
    /// End time in seconds from the beginning of the VOD.
    pub end_secs: f64,
    /// Optional description.
    pub description: Option<String>,
}

impl Chapter {
    /// Create a new chapter.
    #[must_use]
    pub fn new(title: &str, start_secs: f64, end_secs: f64) -> Self {
        Self {
            title: title.to_string(),
            start_secs: start_secs.max(0.0),
            end_secs: end_secs.max(start_secs.max(0.0)),
            description: None,
        }
    }

    /// Set the description.
    #[must_use]
    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = Some(desc.to_string());
        self
    }

    /// Duration of this chapter in seconds.
    #[must_use]
    pub fn duration_secs(&self) -> f64 {
        self.end_secs - self.start_secs
    }

    /// Whether a given time falls within this chapter.
    #[must_use]
    pub fn contains_time(&self, time_secs: f64) -> bool {
        time_secs >= self.start_secs && time_secs < self.end_secs
    }
}

/// Status of a VOD asset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VodStatus {
    /// Recording in progress.
    Recording,
    /// Processing (encoding, thumbnail generation, etc.).
    Processing,
    /// Ready for viewing.
    Ready,
    /// Export in progress to an external platform.
    Exporting,
    /// Archived (not immediately available).
    Archived,
    /// Deleted / marked for removal.
    Deleted,
}

/// Export target platform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExportTarget {
    /// `YouTube` upload.
    YouTube,
    /// Twitch VOD / highlight.
    Twitch,
    /// Local file export.
    LocalFile,
    /// Generic cloud storage.
    CloudStorage,
}

/// Result of an export operation.
#[derive(Debug, Clone, PartialEq)]
pub struct ExportResult {
    /// Target platform.
    pub target: ExportTarget,
    /// Whether the export succeeded.
    pub success: bool,
    /// URL or path of the exported asset (if successful).
    pub location: Option<String>,
    /// Error message (if failed).
    pub error: Option<String>,
}

/// A single VOD asset.
#[derive(Debug, Clone)]
pub struct VodAsset {
    /// Unique identifier.
    pub id: VodId,
    /// Title of the VOD.
    pub title: String,
    /// Game name or category.
    pub game: String,
    /// Duration of the VOD in seconds.
    pub duration_secs: f64,
    /// Chapters within the VOD.
    pub chapters: Vec<Chapter>,
    /// Current status.
    pub status: VodStatus,
    /// Tags for search and categorization.
    pub tags: Vec<String>,
    /// Custom metadata key-value pairs.
    pub metadata: HashMap<String, String>,
    /// File size in bytes (0 if unknown).
    pub file_size_bytes: u64,
    /// Resolution as (width, height).
    pub resolution: (u32, u32),
    /// Average bitrate in kbps.
    pub bitrate_kbps: u32,
}

impl VodAsset {
    /// Create a new VOD asset.
    #[must_use]
    pub fn new(id: VodId, title: &str, game: &str, duration_secs: f64) -> Self {
        Self {
            id,
            title: title.to_string(),
            game: game.to_string(),
            duration_secs: duration_secs.max(0.0),
            chapters: Vec::new(),
            status: VodStatus::Recording,
            tags: Vec::new(),
            metadata: HashMap::new(),
            file_size_bytes: 0,
            resolution: (1920, 1080),
            bitrate_kbps: 6000,
        }
    }

    /// Add a chapter to the VOD.
    pub fn add_chapter(&mut self, chapter: Chapter) {
        self.chapters.push(chapter);
        self.chapters.sort_by(|a, b| {
            a.start_secs
                .partial_cmp(&b.start_secs)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Find the chapter at a given time.
    #[must_use]
    pub fn chapter_at(&self, time_secs: f64) -> Option<&Chapter> {
        self.chapters.iter().find(|c| c.contains_time(time_secs))
    }

    /// Add a tag (no duplicates).
    pub fn add_tag(&mut self, tag: &str) {
        let tag_str = tag.to_string();
        if !self.tags.contains(&tag_str) {
            self.tags.push(tag_str);
        }
    }

    /// Remove a tag, returning whether it was present.
    pub fn remove_tag(&mut self, tag: &str) -> bool {
        if let Some(pos) = self.tags.iter().position(|t| t == tag) {
            self.tags.remove(pos);
            true
        } else {
            false
        }
    }

    /// Set a metadata value.
    pub fn set_metadata(&mut self, key: &str, value: &str) {
        self.metadata.insert(key.to_string(), value.to_string());
    }

    /// Get a metadata value.
    #[must_use]
    pub fn get_metadata(&self, key: &str) -> Option<&String> {
        self.metadata.get(key)
    }

    /// Number of chapters.
    #[must_use]
    pub fn chapter_count(&self) -> usize {
        self.chapters.len()
    }

    /// Estimated file size based on duration and bitrate.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    pub fn estimated_size_bytes(&self) -> u64 {
        // bitrate_kbps * 1000 / 8 * duration
        let bytes_per_sec = f64::from(self.bitrate_kbps) * 1000.0 / 8.0;
        (bytes_per_sec * self.duration_secs) as u64
    }
}

/// Manages a collection of VOD assets.
#[derive(Debug)]
pub struct VodManager {
    /// All VOD assets keyed by ID.
    assets: HashMap<VodId, VodAsset>,
    /// Next ID to assign.
    next_id: u64,
    /// Maximum total storage budget in bytes (0 = unlimited).
    storage_budget_bytes: u64,
}

impl VodManager {
    /// Create a new VOD manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            assets: HashMap::new(),
            next_id: 1,
            storage_budget_bytes: 0,
        }
    }

    /// Create a new VOD manager with a storage budget.
    #[must_use]
    pub fn with_budget(budget_bytes: u64) -> Self {
        Self {
            assets: HashMap::new(),
            next_id: 1,
            storage_budget_bytes: budget_bytes,
        }
    }

    /// Create a new VOD asset and return its ID.
    pub fn create_vod(&mut self, title: &str, game: &str, duration_secs: f64) -> VodId {
        let id = VodId(self.next_id);
        self.next_id += 1;
        let asset = VodAsset::new(id, title, game, duration_secs);
        self.assets.insert(id, asset);
        id
    }

    /// Get a VOD asset by ID.
    #[must_use]
    pub fn get(&self, id: VodId) -> Option<&VodAsset> {
        self.assets.get(&id)
    }

    /// Get a mutable reference to a VOD asset by ID.
    pub fn get_mut(&mut self, id: VodId) -> Option<&mut VodAsset> {
        self.assets.get_mut(&id)
    }

    /// Total number of VOD assets.
    #[must_use]
    pub fn count(&self) -> usize {
        self.assets.len()
    }

    /// Update the status of a VOD.
    pub fn set_status(&mut self, id: VodId, status: VodStatus) -> bool {
        if let Some(asset) = self.assets.get_mut(&id) {
            asset.status = status;
            true
        } else {
            false
        }
    }

    /// Search VODs by tag.
    #[must_use]
    pub fn search_by_tag(&self, tag: &str) -> Vec<VodId> {
        self.assets
            .values()
            .filter(|a| a.tags.iter().any(|t| t == tag))
            .map(|a| a.id)
            .collect()
    }

    /// Search VODs by game name.
    #[must_use]
    pub fn search_by_game(&self, game: &str) -> Vec<VodId> {
        self.assets
            .values()
            .filter(|a| a.game == game)
            .map(|a| a.id)
            .collect()
    }

    /// Total estimated storage used by all VODs.
    #[must_use]
    pub fn total_storage_used(&self) -> u64 {
        self.assets
            .values()
            .map(|a| {
                if a.file_size_bytes > 0 {
                    a.file_size_bytes
                } else {
                    a.estimated_size_bytes()
                }
            })
            .sum()
    }

    /// Check if total storage is within budget.
    #[must_use]
    pub fn within_budget(&self) -> bool {
        if self.storage_budget_bytes == 0 {
            return true;
        }
        self.total_storage_used() <= self.storage_budget_bytes
    }

    /// Delete a VOD by marking it as deleted.
    pub fn delete_vod(&mut self, id: VodId) -> bool {
        self.set_status(id, VodStatus::Deleted)
    }

    /// Permanently remove deleted VODs from the collection.
    pub fn purge_deleted(&mut self) -> usize {
        let before = self.assets.len();
        self.assets.retain(|_, a| a.status != VodStatus::Deleted);
        before - self.assets.len()
    }

    /// List all VOD IDs with a given status.
    #[must_use]
    pub fn list_by_status(&self, status: VodStatus) -> Vec<VodId> {
        self.assets
            .values()
            .filter(|a| a.status == status)
            .map(|a| a.id)
            .collect()
    }

    /// Calculate total duration of all non-deleted VODs in seconds.
    #[must_use]
    pub fn total_duration_secs(&self) -> f64 {
        self.assets
            .values()
            .filter(|a| a.status != VodStatus::Deleted)
            .map(|a| a.duration_secs)
            .sum()
    }
}

impl Default for VodManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chapter_creation() {
        let ch = Chapter::new("Intro", 0.0, 60.0);
        assert_eq!(ch.title, "Intro");
        assert!((ch.duration_secs() - 60.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_chapter_contains_time() {
        let ch = Chapter::new("Boss Fight", 120.0, 300.0);
        assert!(ch.contains_time(150.0));
        assert!(ch.contains_time(120.0));
        assert!(!ch.contains_time(300.0)); // exclusive end
        assert!(!ch.contains_time(50.0));
    }

    #[test]
    fn test_chapter_with_description() {
        let ch = Chapter::new("Finale", 500.0, 600.0).with_description("The last battle");
        assert_eq!(ch.description.as_deref(), Some("The last battle"));
    }

    #[test]
    fn test_chapter_negative_times() {
        let ch = Chapter::new("Test", -5.0, 10.0);
        assert!((ch.start_secs).abs() < f64::EPSILON);
        assert!((ch.end_secs - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_vod_asset_creation() {
        let vod = VodAsset::new(VodId(1), "Stream Day 1", "Elden Ring", 7200.0);
        assert_eq!(vod.title, "Stream Day 1");
        assert_eq!(vod.game, "Elden Ring");
        assert_eq!(vod.status, VodStatus::Recording);
        assert_eq!(vod.chapter_count(), 0);
    }

    #[test]
    fn test_vod_asset_chapters() {
        let mut vod = VodAsset::new(VodId(1), "Test", "Game", 600.0);
        vod.add_chapter(Chapter::new("Part 2", 300.0, 600.0));
        vod.add_chapter(Chapter::new("Part 1", 0.0, 300.0));
        assert_eq!(vod.chapter_count(), 2);
        // Should be sorted by start time
        assert_eq!(vod.chapters[0].title, "Part 1");
        assert_eq!(vod.chapters[1].title, "Part 2");

        let ch = vod.chapter_at(150.0);
        assert!(ch.is_some());
        assert_eq!(ch.expect("should succeed").title, "Part 1");
    }

    #[test]
    fn test_vod_asset_tags() {
        let mut vod = VodAsset::new(VodId(1), "T", "G", 100.0);
        vod.add_tag("highlight");
        vod.add_tag("highlight"); // duplicate
        assert_eq!(vod.tags.len(), 1);
        assert!(vod.remove_tag("highlight"));
        assert!(!vod.remove_tag("highlight"));
    }

    #[test]
    fn test_vod_asset_metadata() {
        let mut vod = VodAsset::new(VodId(1), "T", "G", 100.0);
        vod.set_metadata("region", "NA");
        assert_eq!(vod.get_metadata("region"), Some(&"NA".to_string()));
        assert!(vod.get_metadata("unknown").is_none());
    }

    #[test]
    fn test_vod_estimated_size() {
        let mut vod = VodAsset::new(VodId(1), "T", "G", 3600.0);
        vod.bitrate_kbps = 8000;
        // 8000 * 1000 / 8 * 3600 = 3_600_000_000
        assert_eq!(vod.estimated_size_bytes(), 3_600_000_000);
    }

    #[test]
    fn test_manager_create_and_get() {
        let mut mgr = VodManager::new();
        let id = mgr.create_vod("Session 1", "Dark Souls", 5400.0);
        assert_eq!(mgr.count(), 1);
        let vod = mgr.get(id).expect("entry should exist");
        assert_eq!(vod.title, "Session 1");
    }

    #[test]
    fn test_manager_search_by_tag() {
        let mut mgr = VodManager::new();
        let id1 = mgr.create_vod("V1", "Game A", 100.0);
        let _id2 = mgr.create_vod("V2", "Game B", 200.0);
        mgr.get_mut(id1)
            .expect("entry should exist")
            .add_tag("epic");

        let results = mgr.search_by_tag("epic");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], id1);
    }

    #[test]
    fn test_manager_search_by_game() {
        let mut mgr = VodManager::new();
        mgr.create_vod("V1", "Zelda", 100.0);
        mgr.create_vod("V2", "Zelda", 200.0);
        mgr.create_vod("V3", "Mario", 300.0);

        let results = mgr.search_by_game("Zelda");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_manager_delete_and_purge() {
        let mut mgr = VodManager::new();
        let id1 = mgr.create_vod("V1", "G", 100.0);
        let _id2 = mgr.create_vod("V2", "G", 200.0);

        assert!(mgr.delete_vod(id1));
        assert_eq!(mgr.count(), 2);
        let purged = mgr.purge_deleted();
        assert_eq!(purged, 1);
        assert_eq!(mgr.count(), 1);
    }

    #[test]
    fn test_manager_budget() {
        let mut mgr = VodManager::with_budget(1_000_000);
        mgr.create_vod("V1", "G", 1.0); // very small
        assert!(mgr.within_budget());
    }

    #[test]
    fn test_manager_total_duration() {
        let mut mgr = VodManager::new();
        mgr.create_vod("V1", "G", 100.0);
        let id2 = mgr.create_vod("V2", "G", 200.0);
        mgr.delete_vod(id2);
        // Deleted VODs should not count
        assert!((mgr.total_duration_secs() - 100.0).abs() < f64::EPSILON);
    }
}
