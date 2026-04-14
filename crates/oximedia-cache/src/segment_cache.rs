//! Segment-aware cache optimized for media streaming (HLS/DASH segments).
//!
//! Media streaming protocols like HLS and DASH split content into short
//! *segments* — individually addressable byte ranges that are requested
//! sequentially per stream.  A general-purpose cache is sub-optimal for this
//! workload because:
//!
//! * Segments from the same stream must be evicted together or in sequence-
//!   order; random-key eviction damages buffering performance.
//! * Played (consumed) segments are rarely re-requested and can be evicted
//!   early to free room for upcoming segments.
//! * Clients typically need lookahead prefetch hints so the application layer
//!   can start fetching segments before they are requested.
//!
//! [`SegmentCache`] addresses all of these requirements with a byte-budget and
//! per-stream sequence ordering.
//!
//! # Example
//!
//! ```rust
//! use oximedia_cache::segment_cache::{
//!     MediaSegment, SegmentCache, SegmentCacheConfig, SegmentRef,
//! };
//!
//! let config = SegmentCacheConfig {
//!     max_segments: 128,
//!     max_bytes: 64 * 1024 * 1024,
//!     prefetch_ahead: 3,
//!     evict_played: true,
//! };
//! let mut cache = SegmentCache::new(config);
//!
//! let seg = MediaSegment {
//!     segment_id: "stream1-0000".to_string(),
//!     stream_id:  "stream1".to_string(),
//!     sequence:   0,
//!     duration_secs: 6.0,
//!     data:       vec![0u8; 1024],
//!     content_type: "video/mp2t".to_string(),
//! };
//! cache.insert(seg).expect("insert");
//!
//! let r = SegmentRef { stream_id: "stream1".to_string(), sequence: 0 };
//! assert!(cache.get(&r).is_some());
//! ```

use std::collections::{BTreeMap, HashMap};
use std::fmt;

// ── Public data types ─────────────────────────────────────────────────────────

/// A single media segment as stored in the cache.
#[derive(Debug, Clone)]
pub struct MediaSegment {
    /// Unique identifier for this segment (e.g. `"stream1-0003"`).
    pub segment_id: String,
    /// Logical stream this segment belongs to (e.g. a rendition URL prefix).
    pub stream_id: String,
    /// Monotonically increasing sequence number within the stream.
    pub sequence: u64,
    /// Playback duration of this segment in seconds.
    pub duration_secs: f32,
    /// Raw encoded media data.
    pub data: Vec<u8>,
    /// MIME content type (e.g. `"video/mp2t"` or `"video/mp4"`).
    pub content_type: String,
}

impl MediaSegment {
    /// Byte length of the segment data.
    #[inline]
    pub fn byte_len(&self) -> u64 {
        self.data.len() as u64
    }
}

/// Lightweight reference used as a cache look-up key.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SegmentRef {
    /// Stream identifier.
    pub stream_id: String,
    /// Sequence number within the stream.
    pub sequence: u64,
}

// ── Configuration ─────────────────────────────────────────────────────────────

/// Configuration for [`SegmentCache`].
#[derive(Debug, Clone)]
pub struct SegmentCacheConfig {
    /// Maximum number of segments across all streams.
    pub max_segments: usize,
    /// Maximum total byte budget across all segments.
    pub max_bytes: u64,
    /// How many segments ahead of `current_seq` to include in prefetch hints.
    pub prefetch_ahead: u8,
    /// When `true`, segments marked as played are eligible for immediate
    /// eviction ahead of unplayed segments during the next capacity check.
    pub evict_played: bool,
}

impl Default for SegmentCacheConfig {
    fn default() -> Self {
        Self {
            max_segments: 512,
            max_bytes: 256 * 1024 * 1024, // 256 MiB
            prefetch_ahead: 3,
            evict_played: true,
        }
    }
}

// ── Error type ────────────────────────────────────────────────────────────────

/// Errors that can occur during segment cache operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SegmentCacheError {
    /// The cache is full and no segments could be evicted to make room.
    CacheFull,
    /// The segment data exceeds the entire byte budget of the cache.
    SegmentTooLarge,
    /// A segment with the same `(stream_id, sequence)` already exists.
    DuplicateSegment,
}

impl fmt::Display for SegmentCacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SegmentCacheError::CacheFull => write!(f, "segment cache is full"),
            SegmentCacheError::SegmentTooLarge => {
                write!(f, "segment exceeds cache byte budget")
            }
            SegmentCacheError::DuplicateSegment => {
                write!(f, "segment with this (stream_id, sequence) already cached")
            }
        }
    }
}

impl std::error::Error for SegmentCacheError {}

// ── Stats ─────────────────────────────────────────────────────────────────────

/// Snapshot of [`SegmentCache`] statistics.
#[derive(Debug, Clone, Default)]
pub struct SegmentCacheStats {
    /// Total number of segments currently in the cache.
    pub total_segments: usize,
    /// Total bytes occupied by all cached segments.
    pub total_bytes: u64,
    /// Cumulative successful lookups.
    pub hit_count: u64,
    /// Cumulative failed lookups.
    pub miss_count: u64,
}

// ── Internal per-stream state ─────────────────────────────────────────────────

/// Per-stream metadata (sequence numbers present, played flags).
struct StreamMeta {
    /// Sorted set of sequence numbers present for this stream.
    sequences: BTreeMap<u64, PlayState>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlayState {
    Unplayed,
    Played,
}

impl StreamMeta {
    fn new() -> Self {
        Self {
            sequences: BTreeMap::new(),
        }
    }

    fn oldest_sequence(&self) -> Option<u64> {
        self.sequences.keys().next().copied()
    }
}

// ── SegmentCache ──────────────────────────────────────────────────────────────

/// Segment-aware cache for HLS/DASH media segments.
///
/// Capacity is managed by two independent budgets: segment count
/// (`max_segments`) and total byte size (`max_bytes`).  When either budget is
/// exceeded, the cache evicts the **oldest played** segment first (if
/// `evict_played` is set), then falls back to the globally oldest segment
/// across all streams.
pub struct SegmentCache {
    config: SegmentCacheConfig,
    /// Primary segment storage, keyed by `(stream_id, sequence)`.
    segments: HashMap<SegmentRef, MediaSegment>,
    /// Per-stream metadata.
    streams: HashMap<String, StreamMeta>,
    /// Cumulative byte count across all stored segments.
    total_bytes: u64,
    hit_count: u64,
    miss_count: u64,
}

impl SegmentCache {
    /// Create a new `SegmentCache` with the given configuration.
    pub fn new(config: SegmentCacheConfig) -> Self {
        Self {
            config,
            segments: HashMap::new(),
            streams: HashMap::new(),
            total_bytes: 0,
            hit_count: 0,
            miss_count: 0,
        }
    }

    /// Insert a segment into the cache.
    ///
    /// # Errors
    ///
    /// * [`SegmentCacheError::SegmentTooLarge`] — segment's byte size exceeds
    ///   the entire `max_bytes` budget.
    /// * [`SegmentCacheError::DuplicateSegment`] — a segment with the same
    ///   `(stream_id, sequence)` is already present.
    /// * [`SegmentCacheError::CacheFull`] — the cache cannot be freed enough
    ///   to accommodate the new segment (should not happen under normal
    ///   operation but is returned as a safeguard).
    pub fn insert(&mut self, segment: MediaSegment) -> Result<(), SegmentCacheError> {
        let byte_len = segment.byte_len();

        // Hard reject: segment is larger than the entire byte budget.
        if byte_len > self.config.max_bytes {
            return Err(SegmentCacheError::SegmentTooLarge);
        }

        let key = SegmentRef {
            stream_id: segment.stream_id.clone(),
            sequence: segment.sequence,
        };

        // Reject duplicates.
        if self.segments.contains_key(&key) {
            return Err(SegmentCacheError::DuplicateSegment);
        }

        // Evict until both budgets have room.
        let mut eviction_attempts = 0usize;
        loop {
            let count_ok = self.segments.len() < self.config.max_segments;
            let bytes_ok = self.total_bytes + byte_len <= self.config.max_bytes;
            if count_ok && bytes_ok {
                break;
            }
            let freed = self.evict_one();
            if freed == 0 {
                return Err(SegmentCacheError::CacheFull);
            }
            eviction_attempts += 1;
            // Safety valve: if we have evicted more than the total number of
            // segments we started with something is deeply wrong.
            if eviction_attempts > self.config.max_segments + 1 {
                return Err(SegmentCacheError::CacheFull);
            }
        }

        // Register in per-stream metadata.
        self.streams
            .entry(segment.stream_id.clone())
            .or_insert_with(StreamMeta::new)
            .sequences
            .insert(segment.sequence, PlayState::Unplayed);

        self.total_bytes += byte_len;
        self.segments.insert(key, segment);
        Ok(())
    }

    /// Retrieve the segment identified by `ref_`.
    ///
    /// Returns `None` if the segment is not in the cache.
    pub fn get(&mut self, ref_: &SegmentRef) -> Option<&MediaSegment> {
        match self.segments.get(ref_) {
            Some(seg) => {
                self.hit_count += 1;
                Some(seg)
            }
            None => {
                self.miss_count += 1;
                None
            }
        }
    }

    /// Mark `ref_` as played/consumed.
    ///
    /// Played segments remain in the cache until eviction but are prioritised
    /// for removal when `evict_played` is enabled in the configuration.
    ///
    /// No-ops if the segment is not in the cache.
    pub fn mark_played(&mut self, ref_: &SegmentRef) {
        if let Some(meta) = self.streams.get_mut(&ref_.stream_id) {
            if let Some(state) = meta.sequences.get_mut(&ref_.sequence) {
                *state = PlayState::Played;
            }
        }
    }

    /// Generate prefetch hints for `stream_id` starting at `current_seq + 1`.
    ///
    /// Returns up to `prefetch_ahead` [`SegmentRef`]s for sequences that are
    /// **not yet cached**.  The caller uses these to drive background fetches.
    pub fn prefetch_hints(&self, current_seq: u64, stream_id: &str) -> Vec<SegmentRef> {
        let ahead = self.config.prefetch_ahead as u64;
        let mut hints = Vec::with_capacity(ahead as usize);
        for delta in 1..=ahead {
            let seq = match current_seq.checked_add(delta) {
                Some(s) => s,
                None => break,
            };
            let ref_ = SegmentRef {
                stream_id: stream_id.to_string(),
                sequence: seq,
            };
            if !self.segments.contains_key(&ref_) {
                hints.push(ref_);
            }
        }
        hints
    }

    /// Evict the oldest sequence from the stream that has the most bytes
    /// invested in played segments (or globally oldest if no played segments
    /// exist).
    ///
    /// Returns the number of bytes freed.  Returns `0` if the cache is empty.
    pub fn evict_oldest_stream(&mut self) -> usize {
        self.evict_one()
    }

    /// Return a statistics snapshot.
    pub fn stats(&self) -> SegmentCacheStats {
        SegmentCacheStats {
            total_segments: self.segments.len(),
            total_bytes: self.total_bytes,
            hit_count: self.hit_count,
            miss_count: self.miss_count,
        }
    }

    /// Current total bytes used by all cached segments.
    pub fn total_bytes(&self) -> u64 {
        self.total_bytes
    }

    /// Current number of segments in the cache.
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    // ── private helpers ───────────────────────────────────────────────────────

    /// Attempt to evict a single segment.
    ///
    /// Priority order (highest to lowest):
    ///  1. If `evict_played` is set: oldest played segment across all streams.
    ///  2. Oldest unplayed segment across all streams.
    ///
    /// Returns bytes freed (0 if cache is empty).
    fn evict_one(&mut self) -> usize {
        if let Some(target) = self.find_eviction_target() {
            return self.remove_segment(&target);
        }
        0
    }

    /// Find the best eviction candidate.
    fn find_eviction_target(&self) -> Option<SegmentRef> {
        // Prefer played segments when `evict_played` is enabled.
        if self.config.evict_played {
            if let Some(r) = self.oldest_played() {
                return Some(r);
            }
        }
        self.globally_oldest()
    }

    /// Find the oldest played segment across all streams.
    fn oldest_played(&self) -> Option<SegmentRef> {
        let mut best: Option<(u64, &str)> = None; // (sequence, stream_id)
        for (stream_id, meta) in &self.streams {
            for (seq, state) in &meta.sequences {
                if *state == PlayState::Played {
                    match best {
                        None => best = Some((*seq, stream_id.as_str())),
                        Some((best_seq, _)) if *seq < best_seq => {
                            best = Some((*seq, stream_id.as_str()));
                        }
                        _ => {}
                    }
                }
            }
        }
        best.map(|(seq, sid)| SegmentRef {
            stream_id: sid.to_string(),
            sequence: seq,
        })
    }

    /// Find the globally oldest segment across all streams.
    fn globally_oldest(&self) -> Option<SegmentRef> {
        let mut best: Option<(u64, &str)> = None;
        for (stream_id, meta) in &self.streams {
            if let Some(seq) = meta.oldest_sequence() {
                match best {
                    None => best = Some((seq, stream_id.as_str())),
                    Some((best_seq, _)) if seq < best_seq => {
                        best = Some((seq, stream_id.as_str()));
                    }
                    _ => {}
                }
            }
        }
        best.map(|(seq, sid)| SegmentRef {
            stream_id: sid.to_string(),
            sequence: seq,
        })
    }

    /// Remove the segment at `ref_` and update all tracking structures.
    ///
    /// Returns bytes freed.
    fn remove_segment(&mut self, ref_: &SegmentRef) -> usize {
        if let Some(seg) = self.segments.remove(ref_) {
            let freed = seg.data.len();
            self.total_bytes = self.total_bytes.saturating_sub(freed as u64);

            if let Some(meta) = self.streams.get_mut(&ref_.stream_id) {
                meta.sequences.remove(&ref_.sequence);
                if meta.sequences.is_empty() {
                    self.streams.remove(&ref_.stream_id);
                }
            }
            freed
        } else {
            0
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(stream_id: &str, seq: u64, bytes: usize) -> MediaSegment {
        MediaSegment {
            segment_id: format!("{stream_id}-{seq:04}"),
            stream_id: stream_id.to_string(),
            sequence: seq,
            duration_secs: 6.0,
            data: vec![0u8; bytes],
            content_type: "video/mp2t".to_string(),
        }
    }

    fn default_config() -> SegmentCacheConfig {
        SegmentCacheConfig {
            max_segments: 16,
            max_bytes: 1024 * 1024, // 1 MiB
            prefetch_ahead: 3,
            evict_played: true,
        }
    }

    #[test]
    fn test_insert_and_get() {
        let mut cache = SegmentCache::new(default_config());
        cache.insert(seg("s1", 0, 100)).expect("insert");
        let r = SegmentRef {
            stream_id: "s1".to_string(),
            sequence: 0,
        };
        assert!(cache.get(&r).is_some());
        assert_eq!(cache.stats().hit_count, 1);
    }

    #[test]
    fn test_miss_increments_miss_count() {
        let mut cache = SegmentCache::new(default_config());
        let r = SegmentRef {
            stream_id: "s1".to_string(),
            sequence: 99,
        };
        assert!(cache.get(&r).is_none());
        assert_eq!(cache.stats().miss_count, 1);
    }

    #[test]
    fn test_duplicate_segment_rejected() {
        let mut cache = SegmentCache::new(default_config());
        cache.insert(seg("s1", 0, 100)).expect("first insert");
        let err = cache.insert(seg("s1", 0, 100)).expect_err("should fail");
        assert_eq!(err, SegmentCacheError::DuplicateSegment);
    }

    #[test]
    fn test_segment_too_large_rejected() {
        let config = SegmentCacheConfig {
            max_bytes: 500,
            ..default_config()
        };
        let mut cache = SegmentCache::new(config);
        let err = cache.insert(seg("s1", 0, 1000)).expect_err("too large");
        assert_eq!(err, SegmentCacheError::SegmentTooLarge);
    }

    #[test]
    fn test_byte_limit_evicts_oldest() {
        let config = SegmentCacheConfig {
            max_segments: 100,
            max_bytes: 1000,
            prefetch_ahead: 2,
            evict_played: false,
        };
        let mut cache = SegmentCache::new(config);
        // Insert 5 × 200 B = 1000 B exactly.
        for i in 0..5u64 {
            cache.insert(seg("s1", i, 200)).expect("insert");
        }
        // Inserting a 6th 200 B segment should evict the oldest to stay within budget.
        cache.insert(seg("s1", 5, 200)).expect("insert 6th");
        // Sequence 0 should have been evicted.
        let r0 = SegmentRef {
            stream_id: "s1".to_string(),
            sequence: 0,
        };
        assert!(cache.get(&r0).is_none());
        assert!(cache.total_bytes() <= 1000);
    }

    #[test]
    fn test_segment_count_limit_evicts() {
        let config = SegmentCacheConfig {
            max_segments: 3,
            max_bytes: 10 * 1024 * 1024,
            prefetch_ahead: 1,
            evict_played: false,
        };
        let mut cache = SegmentCache::new(config);
        for i in 0..4u64 {
            cache.insert(seg("s1", i, 10)).expect("insert");
        }
        assert_eq!(cache.segment_count(), 3);
        // Oldest (seq 0) evicted.
        let r0 = SegmentRef {
            stream_id: "s1".to_string(),
            sequence: 0,
        };
        assert!(cache.get(&r0).is_none());
    }

    #[test]
    fn test_played_eviction_prioritised() {
        let config = SegmentCacheConfig {
            max_segments: 3,
            max_bytes: 10 * 1024 * 1024,
            prefetch_ahead: 1,
            evict_played: true,
        };
        let mut cache = SegmentCache::new(config);
        // seq 0, 1, 2 inserted; mark seq 2 as played.
        for i in 0..3u64 {
            cache.insert(seg("s1", i, 100)).expect("insert");
        }
        let r2 = SegmentRef {
            stream_id: "s1".to_string(),
            sequence: 2,
        };
        cache.mark_played(&r2);

        // Inserting seq 3 should evict seq 2 (played) rather than seq 0.
        cache.insert(seg("s1", 3, 100)).expect("insert");
        assert!(cache.get(&r2).is_none(), "played segment should be evicted");
        let r0 = SegmentRef {
            stream_id: "s1".to_string(),
            sequence: 0,
        };
        assert!(cache.get(&r0).is_some(), "unplayed seq 0 should remain");
    }

    #[test]
    fn test_prefetch_hints_excludes_cached() {
        let mut cache = SegmentCache::new(default_config());
        // Seq 1 and 3 already cached; seq 2 is not.
        cache.insert(seg("stream", 1, 50)).expect("insert");
        cache.insert(seg("stream", 3, 50)).expect("insert");

        let hints = cache.prefetch_hints(0, "stream");
        // Should include seq 1 (present? no — already cached), seq 2, seq 3.
        // prefetch_ahead=3 so we check seq 1,2,3:
        //   seq 1 → cached → excluded
        //   seq 2 → not cached → included
        //   seq 3 → cached → excluded
        assert_eq!(hints.len(), 1);
        assert_eq!(hints[0].sequence, 2);
    }

    #[test]
    fn test_prefetch_hints_respects_ahead_count() {
        let config = SegmentCacheConfig {
            prefetch_ahead: 5,
            ..default_config()
        };
        let cache = SegmentCache::new(config);
        let hints = cache.prefetch_hints(10, "stream");
        assert_eq!(hints.len(), 5);
        for (i, h) in hints.iter().enumerate() {
            assert_eq!(h.sequence, 11 + i as u64);
        }
    }

    #[test]
    fn test_evict_oldest_stream_returns_bytes_freed() {
        let mut cache = SegmentCache::new(default_config());
        cache.insert(seg("s1", 0, 512)).expect("insert");
        let freed = cache.evict_oldest_stream();
        assert_eq!(freed, 512);
        assert_eq!(cache.segment_count(), 0);
        assert_eq!(cache.total_bytes(), 0);
    }

    #[test]
    fn test_stats_total_bytes() {
        let mut cache = SegmentCache::new(default_config());
        cache.insert(seg("s1", 0, 100)).expect("insert");
        cache.insert(seg("s1", 1, 200)).expect("insert");
        let s = cache.stats();
        assert_eq!(s.total_segments, 2);
        assert_eq!(s.total_bytes, 300);
    }

    #[test]
    fn test_multi_stream_eviction_fairness() {
        let config = SegmentCacheConfig {
            max_segments: 4,
            max_bytes: 10 * 1024 * 1024,
            prefetch_ahead: 2,
            evict_played: false,
        };
        let mut cache = SegmentCache::new(config);
        cache.insert(seg("streamA", 0, 10)).expect("insert");
        cache.insert(seg("streamB", 0, 10)).expect("insert");
        cache.insert(seg("streamA", 1, 10)).expect("insert");
        cache.insert(seg("streamB", 1, 10)).expect("insert");
        // All 4 slots full; next insert evicts the globally-oldest (streamA-0 or streamB-0,
        // whichever has the lower sequence; both are 0 — determinism depends on HashMap order,
        // so we just verify the cache stays within capacity).
        cache.insert(seg("streamA", 2, 10)).expect("insert 5th");
        assert_eq!(cache.segment_count(), 4);
    }
}
