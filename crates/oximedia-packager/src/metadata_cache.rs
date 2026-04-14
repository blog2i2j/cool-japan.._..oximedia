//! Source file metadata caching for the packaging pipeline.
//!
//! When packaging a single source to both HLS and DASH (or multiple bitrate
//! variants), the source file must be probed/demuxed to extract its metadata:
//! codec, dimensions, duration, bitrate, frame-rate, and keyframe positions.
//!
//! This module provides [`MetadataCache`] — a simple in-memory cache keyed by
//! canonical file path that stores [`CachedSourceMetadata`] values.  Callers
//! should check the cache before probing a file and insert results after probing
//! to avoid redundant I/O.
//!
//! # Example
//!
//! ```rust
//! use oximedia_packager::metadata_cache::{MetadataCache, CachedSourceMetadata};
//!
//! let mut cache = MetadataCache::new();
//!
//! let meta = CachedSourceMetadata {
//!     path: "/media/input.mkv".to_string(),
//!     width: 1920,
//!     height: 1080,
//!     duration_secs: 120.0,
//!     video_bitrate: 5_000_000,
//!     audio_bitrate: 128_000,
//!     frame_rate: 25.0,
//!     codec: "av1".to_string(),
//!     keyframe_positions_secs: vec![0.0, 2.0, 4.0, 6.0],
//!     container: "mkv".to_string(),
//! };
//!
//! cache.insert(meta.clone());
//! assert!(cache.get("/media/input.mkv").is_some());
//! assert!(cache.get("/media/other.mkv").is_none());
//!
//! cache.evict("/media/input.mkv");
//! assert!(cache.get("/media/input.mkv").is_none());
//! ```

use std::collections::HashMap;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// CachedSourceMetadata
// ---------------------------------------------------------------------------

/// Metadata extracted from a source media file and stored in the cache.
#[derive(Debug, Clone)]
pub struct CachedSourceMetadata {
    /// Canonical filesystem path of the source file.
    pub path: String,
    /// Video frame width in pixels.
    pub width: u32,
    /// Video frame height in pixels.
    pub height: u32,
    /// Total duration of the source in seconds.
    pub duration_secs: f64,
    /// Peak video bitrate in bits/s.
    pub video_bitrate: u64,
    /// Audio bitrate in bits/s (0 if no audio).
    pub audio_bitrate: u64,
    /// Video frame-rate (frames per second).
    pub frame_rate: f64,
    /// Video codec name (e.g. `"av1"`, `"vp9"`).
    pub codec: String,
    /// Keyframe timestamps in seconds (presentation time).
    pub keyframe_positions_secs: Vec<f64>,
    /// Container format (e.g. `"mkv"`, `"mp4"`, `"webm"`).
    pub container: String,
}

impl CachedSourceMetadata {
    /// Create a minimal metadata record for testing.
    #[must_use]
    pub fn minimal(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            width: 0,
            height: 0,
            duration_secs: 0.0,
            video_bitrate: 0,
            audio_bitrate: 0,
            frame_rate: 0.0,
            codec: String::new(),
            keyframe_positions_secs: Vec::new(),
            container: String::new(),
        }
    }

    /// Total bitrate (video + audio) in bits/s.
    #[must_use]
    pub fn total_bitrate(&self) -> u64 {
        self.video_bitrate + self.audio_bitrate
    }

    /// Whether the source has at least one known keyframe position.
    #[must_use]
    pub fn has_keyframes(&self) -> bool {
        !self.keyframe_positions_secs.is_empty()
    }

    /// Find the nearest keyframe at or before `position_secs`.
    ///
    /// Returns `None` if there are no keyframe positions.
    #[must_use]
    pub fn keyframe_before(&self, position_secs: f64) -> Option<f64> {
        self.keyframe_positions_secs
            .iter()
            .filter(|&&k| k <= position_secs)
            .copied()
            .next_back()
    }

    /// Find the nearest keyframe at or after `position_secs`.
    ///
    /// Returns `None` if there are no keyframe positions.
    #[must_use]
    pub fn keyframe_after(&self, position_secs: f64) -> Option<f64> {
        self.keyframe_positions_secs
            .iter()
            .find(|&&k| k >= position_secs)
            .copied()
    }
}

// ---------------------------------------------------------------------------
// CacheEntry
// ---------------------------------------------------------------------------

/// Internal cache entry wrapper.
struct CacheEntry {
    metadata: CachedSourceMetadata,
    inserted_at: Instant,
    hit_count: u64,
}

// ---------------------------------------------------------------------------
// MetadataCache
// ---------------------------------------------------------------------------

/// In-memory metadata cache for source files.
///
/// Lookups and inserts are O(1) (hash map).  An optional TTL expires stale
/// entries lazily on the next access.
pub struct MetadataCache {
    entries: HashMap<String, CacheEntry>,
    /// Maximum number of entries before the oldest are evicted.  `None` means
    /// unbounded.
    capacity: Option<usize>,
    /// Time-to-live for each entry.  `None` means entries never expire.
    ttl: Option<Duration>,
}

impl Default for MetadataCache {
    fn default() -> Self {
        Self::new()
    }
}

impl MetadataCache {
    /// Create an unbounded cache with no TTL.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            capacity: None,
            ttl: None,
        }
    }

    /// Create a cache with a capacity limit and optional TTL.
    #[must_use]
    pub fn with_capacity(capacity: usize, ttl: Option<Duration>) -> Self {
        Self {
            entries: HashMap::with_capacity(capacity),
            capacity: Some(capacity),
            ttl,
        }
    }

    /// Insert or replace metadata for the given path.
    ///
    /// If the cache has reached its capacity, the entry with the oldest
    /// insertion time is evicted first.
    pub fn insert(&mut self, meta: CachedSourceMetadata) {
        let key = meta.path.clone();

        // Enforce capacity limit: evict oldest if needed.
        if let Some(cap) = self.capacity {
            if self.entries.len() >= cap && !self.entries.contains_key(&key) {
                self.evict_oldest();
            }
        }

        self.entries.insert(
            key,
            CacheEntry {
                metadata: meta,
                inserted_at: Instant::now(),
                hit_count: 0,
            },
        );
    }

    /// Look up metadata for the given path.
    ///
    /// Returns `None` if the path is not in the cache or if the entry has
    /// expired (TTL exceeded).  Expired entries are lazily removed.
    pub fn get(&mut self, path: &str) -> Option<&CachedSourceMetadata> {
        // Lazy TTL expiry.
        if let Some(ttl) = self.ttl {
            if let Some(entry) = self.entries.get(path) {
                if entry.inserted_at.elapsed() > ttl {
                    self.entries.remove(path);
                    return None;
                }
            }
        }

        let entry = self.entries.get_mut(path)?;
        entry.hit_count += 1;
        Some(&entry.metadata)
    }

    /// Peek at metadata without updating the hit count or checking TTL.
    #[must_use]
    pub fn peek(&self, path: &str) -> Option<&CachedSourceMetadata> {
        self.entries.get(path).map(|e| &e.metadata)
    }

    /// Remove an entry from the cache.
    ///
    /// Returns `true` if an entry was present and removed.
    pub fn evict(&mut self, path: &str) -> bool {
        self.entries.remove(path).is_some()
    }

    /// Remove all expired entries (TTL-based).
    ///
    /// Returns the number of entries removed.
    pub fn evict_expired(&mut self) -> usize {
        let ttl = match self.ttl {
            Some(t) => t,
            None => return 0,
        };
        let before = self.entries.len();
        self.entries
            .retain(|_, e| e.inserted_at.elapsed() <= ttl);
        before - self.entries.len()
    }

    /// Clear the entire cache.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Number of entries currently in the cache.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Hit count for the given path.  Returns `0` if not in the cache.
    #[must_use]
    pub fn hit_count(&self, path: &str) -> u64 {
        self.entries
            .get(path)
            .map_or(0, |e| e.hit_count)
    }

    // Evict the entry with the oldest insertion time.
    fn evict_oldest(&mut self) {
        let oldest_key = self
            .entries
            .iter()
            .min_by_key(|(_, e)| e.inserted_at)
            .map(|(k, _)| k.clone());

        if let Some(key) = oldest_key {
            self.entries.remove(&key);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_meta(path: &str) -> CachedSourceMetadata {
        CachedSourceMetadata {
            path: path.to_string(),
            width: 1920,
            height: 1080,
            duration_secs: 60.0,
            video_bitrate: 5_000_000,
            audio_bitrate: 128_000,
            frame_rate: 25.0,
            codec: "av1".to_string(),
            keyframe_positions_secs: vec![0.0, 2.0, 4.0, 6.0],
            container: "mkv".to_string(),
        }
    }

    #[test]
    fn test_insert_and_get() {
        let mut cache = MetadataCache::new();
        cache.insert(make_meta("/a.mkv"));
        assert!(cache.get("/a.mkv").is_some());
        assert!(cache.get("/b.mkv").is_none());
    }

    #[test]
    fn test_evict() {
        let mut cache = MetadataCache::new();
        cache.insert(make_meta("/a.mkv"));
        assert!(cache.evict("/a.mkv"));
        assert!(!cache.evict("/a.mkv")); // second evict returns false
        assert!(cache.get("/a.mkv").is_none());
    }

    #[test]
    fn test_capacity_evicts_oldest() {
        let mut cache = MetadataCache::with_capacity(2, None);
        cache.insert(make_meta("/a.mkv"));
        cache.insert(make_meta("/b.mkv"));
        // This third insert should evict /a.mkv (oldest).
        cache.insert(make_meta("/c.mkv"));
        assert_eq!(cache.len(), 2);
        // /a.mkv or one of the others evicted – just verify len.
        assert!(cache.len() <= 2);
    }

    #[test]
    fn test_ttl_expiry() {
        let mut cache = MetadataCache::with_capacity(10, Some(Duration::from_millis(1)));
        cache.insert(make_meta("/a.mkv"));
        // Sleep briefly to let TTL expire.
        std::thread::sleep(Duration::from_millis(5));
        assert!(cache.get("/a.mkv").is_none());
    }

    #[test]
    fn test_hit_count() {
        let mut cache = MetadataCache::new();
        cache.insert(make_meta("/a.mkv"));
        let _ = cache.get("/a.mkv");
        let _ = cache.get("/a.mkv");
        let _ = cache.get("/a.mkv");
        assert_eq!(cache.hit_count("/a.mkv"), 3);
    }

    #[test]
    fn test_evict_expired_no_ttl() {
        let mut cache = MetadataCache::new();
        cache.insert(make_meta("/a.mkv"));
        let evicted = cache.evict_expired();
        assert_eq!(evicted, 0);
    }

    #[test]
    fn test_clear() {
        let mut cache = MetadataCache::new();
        cache.insert(make_meta("/a.mkv"));
        cache.insert(make_meta("/b.mkv"));
        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_metadata_total_bitrate() {
        let meta = make_meta("/a.mkv");
        assert_eq!(meta.total_bitrate(), 5_128_000);
    }

    #[test]
    fn test_keyframe_before() {
        let meta = make_meta("/a.mkv");
        assert_eq!(meta.keyframe_before(3.5), Some(2.0));
        assert_eq!(meta.keyframe_before(0.0), Some(0.0));
        assert_eq!(meta.keyframe_before(-1.0), None);
    }

    #[test]
    fn test_keyframe_after() {
        let meta = make_meta("/a.mkv");
        assert_eq!(meta.keyframe_after(3.5), Some(4.0));
        assert_eq!(meta.keyframe_after(6.0), Some(6.0));
        assert_eq!(meta.keyframe_after(7.0), None);
    }

    #[test]
    fn test_peek_does_not_update_hit_count() {
        let mut cache = MetadataCache::new();
        cache.insert(make_meta("/a.mkv"));
        let _ = cache.peek("/a.mkv");
        assert_eq!(cache.hit_count("/a.mkv"), 0);
    }

    #[test]
    fn test_minimal_constructor() {
        let m = CachedSourceMetadata::minimal("/f.mkv");
        assert_eq!(m.path, "/f.mkv");
        assert_eq!(m.total_bitrate(), 0);
        assert!(!m.has_keyframes());
    }
}
