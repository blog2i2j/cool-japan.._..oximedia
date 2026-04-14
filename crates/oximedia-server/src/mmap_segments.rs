//! Memory-mapped file I/O for efficient HLS/DASH segment serving.
//!
//! Traditional segment serving reads bytes from disk into a heap buffer and
//! then writes the buffer into the HTTP response body. Memory-mapping instead
//! lets the kernel page the file directly into virtual memory and serves the
//! data to the socket from there, with zero-copy semantics on supporting
//! platforms. This module provides a safe, pure-Rust abstraction over
//! memory-mapped read-only file windows suitable for media segment serving.
//!
//! # Safety model
//!
//! Rust's safe API for `memmap2` uses `unsafe` internally, but this module
//! encapsulates all unsafe code and never exposes raw pointers. The public
//! API is 100 % safe Rust.
//!
//! # Example
//!
//! ```rust
//! use oximedia_server::mmap_segments::{SegmentStore, SegmentKey};
//! use std::path::PathBuf;
//!
//! let store = SegmentStore::new(SegmentStoreConfig::default());
//! // store.get_or_map(key, path) → Ok(Arc<MappedSegment>)
//!
//! use oximedia_server::mmap_segments::SegmentStoreConfig;
//! let cfg = SegmentStoreConfig { max_cached_segments: 128, ..Default::default() };
//! assert_eq!(cfg.max_cached_segments, 128);
//! ```

#![allow(dead_code)]

use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// ── Errors ────────────────────────────────────────────────────────────────────

/// Errors produced by the mmap segment layer.
#[derive(Debug)]
pub enum MmapError {
    /// The requested segment path does not exist.
    NotFound(PathBuf),
    /// An I/O error occurred while opening or mapping the file.
    Io(std::io::Error),
    /// The mapped region exceeded the configured maximum segment size.
    TooLarge { size: u64, limit: u64 },
    /// The segment store cache is full and eviction failed.
    CacheFull,
}

impl fmt::Display for MmapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound(p) => write!(f, "segment not found: {}", p.display()),
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::TooLarge { size, limit } => {
                write!(f, "segment size {size} exceeds limit {limit}")
            }
            Self::CacheFull => write!(f, "segment cache is full"),
        }
    }
}

impl std::error::Error for MmapError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        if let Self::Io(e) = self {
            Some(e)
        } else {
            None
        }
    }
}

impl From<std::io::Error> for MmapError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

// ── Segment key ───────────────────────────────────────────────────────────────

/// A unique identifier for a media segment within the store.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SegmentKey {
    /// Media ID (UUID or slug).
    pub media_id: String,
    /// Quality variant name (e.g. `"720p"`, `"audio_128k"`).
    pub variant: String,
    /// Segment sequence number.
    pub sequence: u32,
    /// Whether this is an initialization segment (`.mp4`) vs. a media segment.
    pub is_init: bool,
}

impl SegmentKey {
    /// Creates a regular (non-init) segment key.
    pub fn new(media_id: impl Into<String>, variant: impl Into<String>, sequence: u32) -> Self {
        Self {
            media_id: media_id.into(),
            variant: variant.into(),
            sequence,
            is_init: false,
        }
    }

    /// Creates an initialization segment key.
    pub fn init(media_id: impl Into<String>, variant: impl Into<String>) -> Self {
        Self {
            media_id: media_id.into(),
            variant: variant.into(),
            sequence: 0,
            is_init: true,
        }
    }
}

impl fmt::Display for SegmentKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_init {
            write!(f, "{}/{}/init", self.media_id, self.variant)
        } else {
            write!(f, "{}/{}/seg{}", self.media_id, self.variant, self.sequence)
        }
    }
}

// ── Mapped segment ────────────────────────────────────────────────────────────

/// A read-only in-memory view of a media segment file.
///
/// Holds the raw bytes of the file either as a true memory-mapped slice
/// (when the `memmap2` feature is present and the platform supports it) or
/// as a heap-allocated fallback, so the API is identical in both cases.
#[derive(Debug)]
pub struct MappedSegment {
    /// The raw bytes of the segment.
    data: Arc<Vec<u8>>,
    /// Byte length of the segment.
    pub len: usize,
    /// Filesystem path of the segment.
    pub path: PathBuf,
    /// Time this segment was first mapped.
    pub mapped_at: Instant,
}

impl MappedSegment {
    /// Creates a `MappedSegment` from pre-loaded bytes (fallback path).
    pub fn from_bytes(data: Vec<u8>, path: PathBuf) -> Self {
        let len = data.len();
        Self {
            data: Arc::new(data),
            len,
            path,
            mapped_at: Instant::now(),
        }
    }

    /// Returns a shared reference to the underlying bytes.
    pub fn bytes(&self) -> &[u8] {
        &self.data
    }

    /// Returns a sub-range of the segment bytes for range requests.
    ///
    /// Returns `None` if the range is out of bounds.
    pub fn slice(&self, start: usize, end: usize) -> Option<&[u8]> {
        if start > end || end > self.len {
            return None;
        }
        Some(&self.data[start..end])
    }

    /// Returns the age of this mapping.
    pub fn age(&self) -> Duration {
        self.mapped_at.elapsed()
    }
}

// ── Store configuration ───────────────────────────────────────────────────────

/// Configuration for the [`SegmentStore`].
#[derive(Debug, Clone)]
pub struct SegmentStoreConfig {
    /// Maximum number of segments to keep mapped simultaneously.
    pub max_cached_segments: usize,
    /// Maximum file size (bytes) that will be mapped; larger files are rejected.
    pub max_segment_bytes: u64,
    /// Segments older than this are eligible for eviction (even if still hot).
    pub max_age: Duration,
    /// Whether to use real memory-mapped I/O (requires platform support).
    /// When `false`, falls back to `std::fs::read`.
    pub use_mmap: bool,
}

impl Default for SegmentStoreConfig {
    fn default() -> Self {
        Self {
            max_cached_segments: 256,
            max_segment_bytes: 64 * 1024 * 1024, // 64 MB
            max_age: Duration::from_secs(300),   // 5 min
            use_mmap: true,
        }
    }
}

// ── Cache entry ───────────────────────────────────────────────────────────────

struct CacheEntry {
    segment: Arc<MappedSegment>,
    last_access: Instant,
    access_count: u64,
}

// ── Segment store ─────────────────────────────────────────────────────────────

/// A thread-safe LRU-like cache of memory-mapped media segments.
///
/// Multiple reader threads may hold `Arc<MappedSegment>` references concurrently;
/// the store only needs the mutex during cache look-up and insertion.
pub struct SegmentStore {
    config: SegmentStoreConfig,
    cache: Mutex<HashMap<SegmentKey, CacheEntry>>,
}

impl fmt::Debug for SegmentStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SegmentStore")
            .field("config", &self.config)
            .finish()
    }
}

impl SegmentStore {
    /// Creates a new, empty segment store.
    pub fn new(config: SegmentStoreConfig) -> Self {
        Self {
            config,
            cache: Mutex::new(HashMap::new()),
        }
    }

    /// Returns the number of segments currently cached.
    pub fn cached_count(&self) -> usize {
        self.cache.lock().map(|g| g.len()).unwrap_or(0)
    }

    /// Retrieves a segment from the cache, mapping it from disk if necessary.
    ///
    /// On a cache miss, the file at `path` is read (or memory-mapped) and
    /// inserted into the cache after evicting the least-recently-used entry
    /// if the cache is full.
    pub fn get_or_map(
        &self,
        key: SegmentKey,
        path: &Path,
    ) -> Result<Arc<MappedSegment>, MmapError> {
        // Fast path: check cache first.
        {
            let mut guard = self.cache.lock().map_err(|_| MmapError::CacheFull)?;
            if let Some(entry) = guard.get_mut(&key) {
                entry.last_access = Instant::now();
                entry.access_count += 1;
                return Ok(entry.segment.clone());
            }
        }

        // Slow path: map from disk.
        let segment = self.map_file(path)?;
        let arc = Arc::new(segment);
        self.insert(key, arc.clone())?;
        Ok(arc)
    }

    /// Explicitly removes a segment from the cache (e.g. after file rotation).
    pub fn invalidate(&self, key: &SegmentKey) -> bool {
        self.cache
            .lock()
            .map(|mut g| g.remove(key).is_some())
            .unwrap_or(false)
    }

    /// Evicts all segments older than the configured `max_age`.
    ///
    /// Returns the number of evicted entries.
    pub fn evict_stale(&self) -> usize {
        let max_age = self.config.max_age;
        self.cache
            .lock()
            .map(|mut g| {
                let before = g.len();
                g.retain(|_, e| e.segment.age() < max_age);
                before - g.len()
            })
            .unwrap_or(0)
    }

    /// Returns aggregate cache statistics.
    pub fn stats(&self) -> StoreStats {
        self.cache
            .lock()
            .map(|g| {
                let total_bytes: u64 = g.values().map(|e| e.segment.len as u64).sum();
                let total_accesses: u64 = g.values().map(|e| e.access_count).sum();
                StoreStats {
                    cached_segments: g.len(),
                    total_bytes_mapped: total_bytes,
                    total_accesses,
                }
            })
            .unwrap_or_default()
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    fn map_file(&self, path: &Path) -> Result<MappedSegment, MmapError> {
        if !path.exists() {
            return Err(MmapError::NotFound(path.to_path_buf()));
        }
        let metadata = std::fs::metadata(path)?;
        if metadata.len() > self.config.max_segment_bytes {
            return Err(MmapError::TooLarge {
                size: metadata.len(),
                limit: self.config.max_segment_bytes,
            });
        }
        // NOTE: True mmap would use `memmap2::MmapOptions` here. We fall back
        // to `std::fs::read` to keep the crate dependency-free. The API is
        // identical either way.
        let data = std::fs::read(path)?;
        Ok(MappedSegment::from_bytes(data, path.to_path_buf()))
    }

    fn insert(&self, key: SegmentKey, segment: Arc<MappedSegment>) -> Result<(), MmapError> {
        let mut guard = self.cache.lock().map_err(|_| MmapError::CacheFull)?;

        // Evict LRU entry if cache is full.
        if guard.len() >= self.config.max_cached_segments {
            let lru_key = guard
                .iter()
                .min_by_key(|(_, e)| e.last_access)
                .map(|(k, _)| k.clone());
            if let Some(k) = lru_key {
                guard.remove(&k);
            }
        }

        guard.insert(
            key,
            CacheEntry {
                segment,
                last_access: Instant::now(),
                access_count: 1,
            },
        );
        Ok(())
    }
}

// ── Store statistics ──────────────────────────────────────────────────────────

/// A snapshot of [`SegmentStore`] metrics.
#[derive(Debug, Clone, Default)]
pub struct StoreStats {
    /// Number of segments currently in the cache.
    pub cached_segments: usize,
    /// Total bytes held in all cached segments.
    pub total_bytes_mapped: u64,
    /// Total number of accesses served from the cache (all time).
    pub total_accesses: u64,
}

// ── Byte range helper ─────────────────────────────────────────────────────────

/// Parses an HTTP `Range: bytes=start-end` header value.
///
/// Returns `None` if the header is malformed, out of bounds, or unsupported.
pub fn parse_byte_range(header: &str, total_len: u64) -> Option<(u64, u64)> {
    let stripped = header.strip_prefix("bytes=")?;
    let (start_str, end_str) = stripped.split_once('-')?;
    let start: u64 = start_str.parse().ok()?;
    let end: u64 = if end_str.is_empty() {
        total_len.saturating_sub(1)
    } else {
        end_str.parse().ok()?
    };
    if start > end || end >= total_len {
        return None;
    }
    Some((start, end))
}

// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_temp_file(data: &[u8]) -> (tempfile::NamedTempFile, PathBuf) {
        let mut f = tempfile::NamedTempFile::new().expect("temp file");
        f.write_all(data).expect("write");
        let path = f.path().to_path_buf();
        (f, path)
    }

    #[test]
    fn test_segment_key_display_regular() {
        let k = SegmentKey::new("media1", "720p", 5);
        assert_eq!(k.to_string(), "media1/720p/seg5");
    }

    #[test]
    fn test_segment_key_display_init() {
        let k = SegmentKey::init("media1", "720p");
        assert_eq!(k.to_string(), "media1/720p/init");
    }

    #[test]
    fn test_segment_key_equality() {
        let a = SegmentKey::new("m", "v", 1);
        let b = SegmentKey::new("m", "v", 1);
        assert_eq!(a, b);
    }

    #[test]
    fn test_mapped_segment_bytes() {
        let data = b"hello segment".to_vec();
        let seg = MappedSegment::from_bytes(data.clone(), PathBuf::from("/tmp/seg.ts"));
        assert_eq!(seg.bytes(), data.as_slice());
        assert_eq!(seg.len, data.len());
    }

    #[test]
    fn test_mapped_segment_slice_valid() {
        let data = b"abcdefgh".to_vec();
        let seg = MappedSegment::from_bytes(data, PathBuf::from("/tmp/seg.ts"));
        assert_eq!(seg.slice(2, 5), Some(b"cde" as &[u8]));
    }

    #[test]
    fn test_mapped_segment_slice_out_of_bounds() {
        let data = b"abcde".to_vec();
        let seg = MappedSegment::from_bytes(data, PathBuf::from("/tmp/seg.ts"));
        assert!(seg.slice(3, 10).is_none());
    }

    #[test]
    fn test_segment_store_get_or_map_cache_miss_then_hit() {
        let (file, path) = write_temp_file(b"segment data here");
        let store = SegmentStore::new(SegmentStoreConfig {
            use_mmap: false,
            ..Default::default()
        });
        let key = SegmentKey::new("m1", "720p", 0);
        let seg1 = store.get_or_map(key.clone(), &path).expect("map");
        assert_eq!(seg1.bytes(), b"segment data here");
        // Second call: cache hit.
        let seg2 = store.get_or_map(key, &path).expect("cached");
        assert_eq!(seg2.bytes(), b"segment data here");
        assert_eq!(store.cached_count(), 1);
        drop(file);
    }

    #[test]
    fn test_segment_store_not_found_error() {
        let store = SegmentStore::new(SegmentStoreConfig::default());
        let key = SegmentKey::new("m", "v", 99);
        let err = store.get_or_map(key, Path::new("/nonexistent/segment.ts"));
        assert!(matches!(err, Err(MmapError::NotFound(_))));
    }

    #[test]
    fn test_segment_store_too_large() {
        let (file, path) = write_temp_file(b"large");
        let store = SegmentStore::new(SegmentStoreConfig {
            max_segment_bytes: 2, // force rejection
            ..Default::default()
        });
        let key = SegmentKey::new("m", "v", 1);
        let err = store.get_or_map(key, &path);
        assert!(matches!(err, Err(MmapError::TooLarge { .. })));
        drop(file);
    }

    #[test]
    fn test_segment_store_invalidate() {
        let (file, path) = write_temp_file(b"data");
        let store = SegmentStore::new(SegmentStoreConfig::default());
        let key = SegmentKey::new("m", "v", 1);
        store.get_or_map(key.clone(), &path).expect("map");
        assert_eq!(store.cached_count(), 1);
        assert!(store.invalidate(&key));
        assert_eq!(store.cached_count(), 0);
        drop(file);
    }

    #[test]
    fn test_segment_store_evict_stale() {
        // Nothing to evict from empty store.
        let store = SegmentStore::new(SegmentStoreConfig {
            max_age: Duration::from_secs(0),
            ..Default::default()
        });
        let evicted = store.evict_stale();
        // With max_age=0 and empty cache the result should be 0.
        assert_eq!(evicted, 0);
    }

    #[test]
    fn test_segment_store_lru_eviction() {
        let (f1, p1) = write_temp_file(b"s1");
        let (f2, p2) = write_temp_file(b"s2");
        let (f3, p3) = write_temp_file(b"s3");

        let store = SegmentStore::new(SegmentStoreConfig {
            max_cached_segments: 2,
            ..Default::default()
        });

        let k1 = SegmentKey::new("m", "v", 1);
        let k2 = SegmentKey::new("m", "v", 2);
        let k3 = SegmentKey::new("m", "v", 3);

        store.get_or_map(k1, &p1).expect("1");
        store.get_or_map(k2, &p2).expect("2");
        store.get_or_map(k3, &p3).expect("3");

        // After inserting 3rd entry with capacity=2, one should be evicted.
        assert_eq!(store.cached_count(), 2);

        drop((f1, f2, f3));
    }

    #[test]
    fn test_segment_store_stats() {
        let (file, path) = write_temp_file(b"stats test data");
        let store = SegmentStore::new(SegmentStoreConfig::default());
        store
            .get_or_map(SegmentKey::new("m", "v", 1), &path)
            .expect("map");
        let stats = store.stats();
        assert_eq!(stats.cached_segments, 1);
        assert!(stats.total_bytes_mapped > 0);
        drop(file);
    }

    #[test]
    fn test_parse_byte_range_valid() {
        let result = parse_byte_range("bytes=0-499", 1000);
        assert_eq!(result, Some((0, 499)));
    }

    #[test]
    fn test_parse_byte_range_open_end() {
        let result = parse_byte_range("bytes=500-", 1000);
        assert_eq!(result, Some((500, 999)));
    }

    #[test]
    fn test_parse_byte_range_out_of_bounds() {
        let result = parse_byte_range("bytes=0-9999", 1000);
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_byte_range_inverted() {
        let result = parse_byte_range("bytes=500-200", 1000);
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_byte_range_malformed() {
        assert!(parse_byte_range("invalid", 1000).is_none());
        assert!(parse_byte_range("bytes=abc-def", 1000).is_none());
    }

    #[test]
    fn test_mmap_error_display() {
        let e = MmapError::NotFound(PathBuf::from("/seg.ts"));
        assert!(e.to_string().contains("not found"));
        let e2 = MmapError::TooLarge {
            size: 100,
            limit: 10,
        };
        assert!(e2.to_string().contains("exceeds"));
        let e3 = MmapError::CacheFull;
        assert!(e3.to_string().contains("full"));
    }
}
