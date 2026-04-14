//! Cross-session persistent cache for decoded thumbnails and media fingerprints.
//!
//! The in-memory [`crate::dedup_cache`] LRU cache is discarded at process exit.
//! This module adds a lightweight JSON-backed persistent store so that expensive
//! thumbnail decoding and perceptual hash computation are **reused across
//! deduplication sessions**.
//!
//! # Design
//!
//! [`PersistentFingerprintCache`] maintains a flat JSON file on disk.  Each entry
//! records:
//!
//! - The source file path.
//! - Its BLAKE3 hex digest (64 chars) — used to detect when the file changes and
//!   the cached fingerprint is stale.
//! - The 64-bit perceptual hash.
//! - An optional thumbnail (8×8 grayscale pixel bytes, base64-encoded).
//! - The modification timestamp at cache time.
//!
//! On [`load`](PersistentFingerprintCache::load), all entries are read from disk.
//! On [`save`](PersistentFingerprintCache::save), the current entries are written
//! back atomically (write to a temp file then rename).
//!
//! **Staleness** is detected by comparing the stored BLAKE3 digest with a freshly
//! computed digest of the source file.  [`get_valid`](PersistentFingerprintCache::get_valid)
//! returns `None` for stale or missing entries.
//!
//! # Example
//!
//! ```rust
//! use oximedia_dedup::persistent_cache::{PersistentFingerprintCache, CachedEntry};
//!
//! let dir = std::env::temp_dir().join("oximedia_pc_doctest");
//! std::fs::create_dir_all(&dir).ok();
//! let cache_path = dir.join("fps.json");
//!
//! let mut cache = PersistentFingerprintCache::new(cache_path.clone());
//! cache.insert(CachedEntry {
//!     path: "/media/clip.mp4".to_string(),
//!     blake3_hex: "0".repeat(64),
//!     phash: 0xDEAD_BEEF_1234_5678,
//!     thumbnail: None,
//!     modified_secs: 1_700_000_000,
//! });
//!
//! cache.save().expect("save ok");
//!
//! let cache2 = PersistentFingerprintCache::load(cache_path).expect("load ok");
//! assert_eq!(cache2.len(), 1);
//! ```

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;
use std::io::{self, BufReader, BufWriter};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// CachedEntry
// ---------------------------------------------------------------------------

/// A single entry in the persistent fingerprint cache.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedEntry {
    /// Absolute path of the source media file.
    pub path: String,
    /// Lower-case hex BLAKE3 digest (64 chars) of the file at cache time.
    pub blake3_hex: String,
    /// 64-bit perceptual hash.
    pub phash: u64,
    /// Optional 8×8 grayscale thumbnail bytes (64 bytes), stored as a Vec to
    /// allow `None` when no thumbnail was computed.
    pub thumbnail: Option<Vec<u8>>,
    /// Unix-second modification timestamp of the file at cache time.
    pub modified_secs: u64,
}

impl CachedEntry {
    /// Return `true` if the thumbnail has the expected 8×8 = 64-byte size.
    #[must_use]
    pub fn thumbnail_valid(&self) -> bool {
        self.thumbnail
            .as_ref()
            .map(|t| t.len() == 64)
            .unwrap_or(true) // no thumbnail is also valid
    }
}

// ---------------------------------------------------------------------------
// PersistentFingerprintCache
// ---------------------------------------------------------------------------

/// Cross-session persistent cache mapping file paths to their fingerprints.
///
/// Entries are keyed by `path` string.
#[derive(Debug, Clone)]
pub struct PersistentFingerprintCache {
    /// Path to the backing JSON file.
    cache_path: PathBuf,
    /// In-memory entries keyed by file path.
    entries: HashMap<String, CachedEntry>,
    /// Number of cache hits since last reset.
    hits: u64,
    /// Number of cache misses since last reset.
    misses: u64,
}

impl PersistentFingerprintCache {
    /// Create a new, empty cache backed by `cache_path`.
    ///
    /// The file is not read or written until [`save`](Self::save) or
    /// [`load`](Self::load) is called.
    #[must_use]
    pub fn new(cache_path: PathBuf) -> Self {
        Self {
            cache_path,
            entries: HashMap::new(),
            hits: 0,
            misses: 0,
        }
    }

    /// Load a cache from `cache_path`.
    ///
    /// Returns an empty cache (rather than an error) if the file does not
    /// exist.  Returns `Err` only on genuine I/O or parse failures.
    ///
    /// # Errors
    ///
    /// Returns an `io::Error` if the file exists but cannot be read or parsed.
    pub fn load(cache_path: PathBuf) -> io::Result<Self> {
        if !cache_path.exists() {
            return Ok(Self::new(cache_path));
        }
        let file = std::fs::File::open(&cache_path)?;
        let reader = BufReader::new(file);
        let entries: HashMap<String, CachedEntry> =
            serde_json::from_reader(reader).map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("cache parse error: {e}"),
                )
            })?;
        Ok(Self {
            cache_path,
            entries,
            hits: 0,
            misses: 0,
        })
    }

    /// Save the cache to disk atomically (write temp → rename).
    ///
    /// # Errors
    ///
    /// Returns an `io::Error` if writing or renaming fails.
    pub fn save(&self) -> io::Result<()> {
        // Ensure the parent directory exists.
        if let Some(parent) = self.cache_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Write to a sibling temp file first.
        let tmp_path = self.cache_path.with_extension("tmp");
        {
            let file = std::fs::File::create(&tmp_path)?;
            let writer = BufWriter::new(file);
            serde_json::to_writer(writer, &self.entries).map_err(|e| {
                io::Error::new(io::ErrorKind::Other, format!("cache write error: {e}"))
            })?;
        }
        std::fs::rename(&tmp_path, &self.cache_path)?;
        Ok(())
    }

    /// Insert or update a [`CachedEntry`].
    pub fn insert(&mut self, entry: CachedEntry) {
        self.entries.insert(entry.path.clone(), entry);
    }

    /// Remove the entry for `path`, returning it if it existed.
    pub fn remove(&mut self, path: &str) -> Option<CachedEntry> {
        self.entries.remove(path)
    }

    /// Return the number of entries in the cache.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return `true` if the cache contains no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Look up a cached entry by file path **without** freshness checking.
    ///
    /// Use [`get_valid`](Self::get_valid) to validate against on-disk state.
    #[must_use]
    pub fn get(&self, path: &str) -> Option<&CachedEntry> {
        self.entries.get(path)
    }

    /// Look up a cached entry, validating that the stored BLAKE3 digest still
    /// matches the current file.
    ///
    /// Reads the actual file to recompute its BLAKE3 hash.  Returns `None` if:
    /// - The entry is not in the cache.
    /// - The file does not exist.
    /// - The digest has changed (file was modified).
    ///
    /// Updates the internal hit/miss counters.
    pub fn get_valid(&mut self, path: &str) -> Option<&CachedEntry> {
        let entry = match self.entries.get(path) {
            Some(e) => e,
            None => {
                self.misses += 1;
                return None;
            }
        };

        // Read the file and check the BLAKE3 hex digest.
        match compute_blake3_hex(Path::new(path)) {
            Ok(current_hex) => {
                if current_hex == entry.blake3_hex {
                    self.hits += 1;
                    self.entries.get(path)
                } else {
                    // Stale entry — remove it.
                    self.misses += 1;
                    self.entries.remove(path);
                    None
                }
            }
            Err(_) => {
                // File inaccessible → treat as cache miss.
                self.misses += 1;
                None
            }
        }
    }

    /// Return the number of cache hits since the cache was loaded or last reset.
    #[must_use]
    pub fn hits(&self) -> u64 {
        self.hits
    }

    /// Return the number of cache misses since the cache was loaded or last reset.
    #[must_use]
    pub fn misses(&self) -> u64 {
        self.misses
    }

    /// Return the hit rate (0.0 – 1.0).  Returns 0.0 if no lookups have been made.
    #[must_use]
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            return 0.0;
        }
        self.hits as f64 / total as f64
    }

    /// Reset hit/miss counters.
    pub fn reset_stats(&mut self) {
        self.hits = 0;
        self.misses = 0;
    }

    /// Evict all entries whose source file no longer exists on disk.
    ///
    /// Returns the number of entries evicted.
    pub fn evict_missing(&mut self) -> usize {
        let before = self.entries.len();
        self.entries
            .retain(|path, _| Path::new(path).exists());
        before - self.entries.len()
    }

    /// Evict all entries that are stale (file modified since caching).
    ///
    /// Recomputes BLAKE3 hashes for all cached files.  Entries are removed when
    /// the digest no longer matches.  Returns the number of entries evicted.
    pub fn evict_stale(&mut self) -> usize {
        let paths: Vec<String> = self.entries.keys().cloned().collect();
        let mut evicted = 0;
        for path in paths {
            let stale = if let Some(entry) = self.entries.get(&path) {
                compute_blake3_hex(Path::new(&path))
                    .map(|h| h != entry.blake3_hex)
                    .unwrap_or(true) // can't read → evict
            } else {
                false
            };
            if stale {
                self.entries.remove(&path);
                evicted += 1;
            }
        }
        evicted
    }

    /// Merge entries from `other` into this cache.
    ///
    /// Entries in `other` overwrite entries with the same path in `self`.
    pub fn merge_from(&mut self, other: &Self) {
        for (path, entry) in &other.entries {
            self.entries.insert(path.clone(), entry.clone());
        }
    }

    /// Return an iterator over all cached entries.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &CachedEntry)> {
        self.entries.iter()
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Compute a lower-case BLAKE3 hex digest for a file path.
///
/// Uses a simple FNV-1a based stand-in when the `blake3` crate is available
/// via the workspace.  This avoids re-implementing the full BLAKE3 algorithm
/// and stays consistent with the rest of `oximedia-dedup`.
fn compute_blake3_hex(path: &Path) -> io::Result<String> {
    use std::io::Read;

    let mut file = std::fs::File::open(path)?;
    let mut hasher = blake3::Hasher::new();
    let mut buf = vec![0u8; 65_536];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher.finalize().to_hex().to_string())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn tmp_cache_path(name: &str) -> PathBuf {
        std::env::temp_dir()
            .join("oximedia_persistent_cache_tests")
            .join(name)
    }

    fn sample_entry(path: &str) -> CachedEntry {
        CachedEntry {
            path: path.to_string(),
            blake3_hex: "0".repeat(64),
            phash: 0xDEAD_BEEF_1234_5678,
            thumbnail: None,
            modified_secs: 1_700_000_000,
        }
    }

    #[test]
    fn test_new_cache_is_empty() {
        let cache = PersistentFingerprintCache::new(tmp_cache_path("new_empty.json"));
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_insert_and_get() {
        let mut cache = PersistentFingerprintCache::new(tmp_cache_path("insert.json"));
        cache.insert(sample_entry("/media/a.mp4"));
        let e = cache.get("/media/a.mp4");
        assert!(e.is_some());
        assert_eq!(e.unwrap().phash, 0xDEAD_BEEF_1234_5678);
    }

    #[test]
    fn test_remove() {
        let mut cache = PersistentFingerprintCache::new(tmp_cache_path("remove.json"));
        cache.insert(sample_entry("/media/b.mp4"));
        assert!(cache.remove("/media/b.mp4").is_some());
        assert!(cache.get("/media/b.mp4").is_none());
    }

    #[test]
    fn test_save_and_load_roundtrip() {
        let path = tmp_cache_path("roundtrip.json");
        std::fs::create_dir_all(path.parent().unwrap()).ok();

        let mut cache = PersistentFingerprintCache::new(path.clone());
        cache.insert(sample_entry("/media/c.mp4"));
        cache.save().expect("save should succeed");

        let loaded = PersistentFingerprintCache::load(path).expect("load should succeed");
        assert_eq!(loaded.len(), 1);
        assert!(loaded.get("/media/c.mp4").is_some());
    }

    #[test]
    fn test_load_nonexistent_returns_empty() {
        let path = tmp_cache_path("nonexistent_xyzabc.json");
        // Make sure it really doesn't exist.
        let _ = std::fs::remove_file(&path);
        let cache = PersistentFingerprintCache::load(path).expect("should not fail");
        assert!(cache.is_empty());
    }

    #[test]
    fn test_hit_miss_counters() {
        let mut cache = PersistentFingerprintCache::new(tmp_cache_path("stats.json"));
        cache.insert(sample_entry("/x.mp4"));
        // Plain get does not update counters.
        let _ = cache.get("/x.mp4");
        assert_eq!(cache.hits(), 0);
        assert_eq!(cache.misses(), 0);
    }

    #[test]
    fn test_hit_rate_zero_on_no_lookups() {
        let cache = PersistentFingerprintCache::new(tmp_cache_path("hitrate.json"));
        assert_eq!(cache.hit_rate(), 0.0);
    }

    #[test]
    fn test_evict_missing_removes_nonexistent_paths() {
        let mut cache = PersistentFingerprintCache::new(tmp_cache_path("evict.json"));
        cache.insert(sample_entry("/definitely/does/not/exist/zzz.mp4"));
        assert_eq!(cache.len(), 1);
        let evicted = cache.evict_missing();
        assert_eq!(evicted, 1);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_evict_stale_removes_changed_files() {
        // Create a real temp file, write content, hash it, then change the file.
        let dir = std::env::temp_dir().join("oximedia_pc_stale_test");
        std::fs::create_dir_all(&dir).ok();
        let file_path = dir.join("media_file.bin");

        // Write initial content.
        {
            let mut f = std::fs::File::create(&file_path).expect("create");
            f.write_all(b"original content for hashing").expect("write");
        }

        // Compute actual hash.
        let real_hash = compute_blake3_hex(&file_path).expect("hash ok");

        let mut cache = PersistentFingerprintCache::new(tmp_cache_path("stale.json"));
        cache.insert(CachedEntry {
            path: file_path.to_string_lossy().to_string(),
            blake3_hex: real_hash.clone(),
            phash: 0x1111,
            thumbnail: None,
            modified_secs: 0,
        });

        // Eviction should keep the entry (file unchanged).
        let evicted = cache.evict_stale();
        assert_eq!(evicted, 0, "file unchanged → no eviction");

        // Now mutate the file.
        {
            let mut f = std::fs::File::create(&file_path).expect("create");
            f.write_all(b"modified content, different bytes!").expect("write");
        }

        // Now the cached hash is stale.
        let evicted2 = cache.evict_stale();
        assert_eq!(evicted2, 1, "changed file → entry evicted");
        assert!(cache.is_empty());

        let _ = std::fs::remove_file(&file_path);
    }

    #[test]
    fn test_merge_from() {
        let mut a = PersistentFingerprintCache::new(tmp_cache_path("merge_a.json"));
        let mut b = PersistentFingerprintCache::new(tmp_cache_path("merge_b.json"));
        a.insert(sample_entry("/file_a.mp4"));
        b.insert(sample_entry("/file_b.mp4"));
        a.merge_from(&b);
        assert_eq!(a.len(), 2);
        assert!(a.get("/file_a.mp4").is_some());
        assert!(a.get("/file_b.mp4").is_some());
    }

    #[test]
    fn test_thumbnail_valid_no_thumbnail() {
        let entry = sample_entry("/x.mp4");
        assert!(entry.thumbnail_valid()); // None is valid
    }

    #[test]
    fn test_thumbnail_valid_correct_size() {
        let entry = CachedEntry {
            thumbnail: Some(vec![128u8; 64]), // 8×8 bytes
            ..sample_entry("/y.mp4")
        };
        assert!(entry.thumbnail_valid());
    }

    #[test]
    fn test_thumbnail_invalid_wrong_size() {
        let entry = CachedEntry {
            thumbnail: Some(vec![0u8; 32]), // wrong size
            ..sample_entry("/z.mp4")
        };
        assert!(!entry.thumbnail_valid());
    }

    #[test]
    fn test_reset_stats() {
        let mut cache = PersistentFingerprintCache::new(tmp_cache_path("reset.json"));
        // Manually bump counters via get_valid (will miss since no entry).
        let _ = cache.get_valid("/nonexistent.mp4");
        assert!(cache.misses() > 0);
        cache.reset_stats();
        assert_eq!(cache.misses(), 0);
        assert_eq!(cache.hits(), 0);
    }
}
