//! Streaming duplicate detection without loading entire files into memory.
//!
//! This module implements content-defined chunking over arbitrary `io::Read`
//! sources so that very large media files can be fingerprinted without mapping
//! the full file into RAM.  The approach:
//!
//! 1. A [`StreamChunker`] wraps any `io::Read` and emits content-defined
//!    chunk hashes via `Iterator<Item = ChunkDigest>`.  Data flows through a
//!    fixed-size internal buffer (`BUF_SIZE` bytes), so memory use is bounded
//!    regardless of file size.
//! 2. [`StreamFingerprint`] aggregates chunk digests into a compact file-level
//!    fingerprint that survives byte-level insertions and deletions (unlike a
//!    whole-file BLAKE3 hash).
//! 3. [`StreamDedupIndex`] stores fingerprints and answers "is this stream a
//!    near-duplicate of something already indexed?" via chunk-level Jaccard
//!    similarity.
//!
//! # Rationale
//!
//! The existing [`crate::rolling_hash`] module provides content-defined chunking
//! over in-memory byte slices.  This module extends the deduplication pipeline
//! with a streaming interface that satisfies the TODO item:
//! *"Optimize `rolling_hash.rs` for streaming duplicate detection without
//! loading entire files"*.
//!
//! # Example
//!
//! ```rust
//! use oximedia_dedup::stream_dedup::{StreamChunkerConfig, StreamDedupIndex};
//! use std::io::Cursor;
//!
//! let config = StreamChunkerConfig::default();
//! let mut index = StreamDedupIndex::new(config.clone());
//!
//! let data = vec![42u8; 32_768];
//! let fp = index.ingest("file-a", Cursor::new(data.clone())).expect("ingest ok");
//! assert!(fp.chunk_count() > 0);
//!
//! // A second identical stream should be detected as a duplicate.
//! let fp2 = index.ingest("file-b", Cursor::new(data)).expect("ingest ok");
//! let sim = index.jaccard_similarity(&fp, &fp2);
//! assert!((sim - 1.0).abs() < 1e-9);
//! ```

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]

use std::collections::{HashMap, HashSet};
use std::io::{self, Read};

/// Internal I/O buffer size for streaming reads.
const BUF_SIZE: usize = 65_536; // 64 KiB

// ---------------------------------------------------------------------------
// StreamChunkerConfig
// ---------------------------------------------------------------------------

/// Configuration for the streaming content-defined chunker.
#[derive(Debug, Clone)]
pub struct StreamChunkerConfig {
    /// Minimum chunk length in bytes.
    pub min_chunk: usize,
    /// Maximum chunk length in bytes.
    pub max_chunk: usize,
    /// Rolling hash window size.
    pub window_size: usize,
    /// Number of low-order bits of the rolling hash used for boundary detection.
    /// A boundary occurs when `(hash & boundary_mask) == 0`.
    pub mask_bits: u32,
}

impl Default for StreamChunkerConfig {
    fn default() -> Self {
        Self {
            min_chunk: 4_096,
            max_chunk: 131_072,
            window_size: 48,
            mask_bits: 12, // average chunk ≈ 4096 bytes
        }
    }
}

impl StreamChunkerConfig {
    /// Compute the boundary mask from `mask_bits`.
    #[must_use]
    pub fn boundary_mask(&self) -> u64 {
        (1u64 << self.mask_bits) - 1
    }

    /// Validate the configuration.
    ///
    /// Returns `true` when all invariants hold.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.min_chunk > 0
            && self.max_chunk >= self.min_chunk
            && self.window_size > 0
            && self.mask_bits > 0
            && self.mask_bits < 32
    }
}

// ---------------------------------------------------------------------------
// ChunkDigest
// ---------------------------------------------------------------------------

/// The hash of a single content-defined chunk.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ChunkDigest {
    /// FNV-1a 64-bit digest of the chunk bytes.
    pub hash: u64,
    /// Number of bytes in this chunk.
    pub len: usize,
}

/// Compute a FNV-1a 64-bit hash of a byte slice.
fn fnv1a_64(data: &[u8]) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0100_0000_01b3;
    let mut h = OFFSET;
    for &b in data {
        h ^= u64::from(b);
        h = h.wrapping_mul(PRIME);
    }
    h
}

// ---------------------------------------------------------------------------
// Streaming rolling hash (Buzhash-lite)
// ---------------------------------------------------------------------------

/// A minimal rolling hash used internally by [`StreamChunker`].
///
/// Uses a power-of-2 lookup table (byte → 64-bit random value) with XOR
/// rotation to provide the rolling property.
struct RollingHash {
    table: [u64; 256],
    window: Vec<u8>,
    window_size: usize,
    head: usize,
    value: u64,
    count: usize,
}

impl RollingHash {
    fn new(window_size: usize) -> Self {
        // Deterministic table derived from FNV-1a of the byte value.
        let mut table = [0u64; 256];
        for (i, slot) in table.iter_mut().enumerate() {
            *slot = fnv1a_64(&[i as u8, 0x5A, 0xA5]);
        }
        Self {
            table,
            window: vec![0u8; window_size],
            window_size,
            head: 0,
            value: 0,
            count: 0,
        }
    }

    /// Feed one byte; returns the updated rolling hash.
    fn update(&mut self, byte: u8) -> u64 {
        let outgoing = self.window[self.head];
        self.window[self.head] = byte;
        self.head = (self.head + 1) % self.window_size;
        // Rotate left by 1, XOR in new byte, XOR out old byte (rotated by window_size).
        self.value = self.value.rotate_left(1)
            ^ self.table[byte as usize]
            ^ self.table[outgoing as usize].rotate_left(self.window_size as u32 & 63);
        self.count += 1;
        self.value
    }
}

// ---------------------------------------------------------------------------
// StreamChunker
// ---------------------------------------------------------------------------

/// An iterator over content-defined [`ChunkDigest`]s read from an `io::Read`.
///
/// Internally buffers data in a fixed-size heap buffer so the caller's memory
/// usage is bounded regardless of file size.  The I/O buffer `io_buf` holds
/// the most recently read batch; `io_pos` tracks where processing should resume
/// within that batch, allowing the chunker to return mid-buffer and resume on
/// the next call without discarding unprocessed bytes.
pub struct StreamChunker<R: Read> {
    reader: R,
    config: StreamChunkerConfig,
    rolling: RollingHash,
    /// I/O read buffer.
    io_buf: Vec<u8>,
    /// How many bytes are valid in `io_buf`.
    io_len: usize,
    /// Current read position within `io_buf`.
    io_pos: usize,
    /// Accumulation buffer for the current in-flight chunk.
    chunk_buf: Vec<u8>,
    /// Set to true once the underlying reader returns EOF.
    done: bool,
}

impl<R: Read> StreamChunker<R> {
    /// Create a new `StreamChunker` wrapping `reader`.
    #[must_use]
    pub fn new(reader: R, config: StreamChunkerConfig) -> Self {
        let window_size = config.window_size;
        Self {
            reader,
            config,
            rolling: RollingHash::new(window_size),
            io_buf: vec![0u8; BUF_SIZE],
            io_len: 0,
            io_pos: 0,
            chunk_buf: Vec::with_capacity(8_192),
            done: false,
        }
    }

    /// Collect all chunk digests eagerly, consuming `self`.
    ///
    /// # Errors
    ///
    /// Propagates any `io::Error` from the underlying reader.
    pub fn collect_all(mut self) -> io::Result<Vec<ChunkDigest>> {
        let mut out = Vec::new();
        loop {
            match self.next_chunk() {
                Ok(Some(d)) => out.push(d),
                Ok(None) => break,
                Err(e) => return Err(e),
            }
        }
        Ok(out)
    }

    /// Advance to the next chunk.
    ///
    /// Returns `Ok(None)` when the stream is exhausted.
    ///
    /// # Errors
    ///
    /// Returns an `io::Error` on read failure.
    pub fn next_chunk(&mut self) -> io::Result<Option<ChunkDigest>> {
        if self.done && self.io_pos >= self.io_len {
            return Ok(None);
        }
        let mask = self.config.boundary_mask();

        loop {
            // Refill I/O buffer when the current batch is exhausted.
            if self.io_pos >= self.io_len {
                if self.done {
                    break;
                }
                let n = self.reader.read(&mut self.io_buf)?;
                if n == 0 {
                    self.done = true;
                    break;
                }
                self.io_len = n;
                self.io_pos = 0;
            }

            // Process bytes from the current buffer position.
            while self.io_pos < self.io_len {
                let byte = self.io_buf[self.io_pos];
                self.io_pos += 1;

                let h = self.rolling.update(byte);
                self.chunk_buf.push(byte);
                let chunk_len = self.chunk_buf.len();

                if chunk_len < self.config.min_chunk {
                    continue;
                }
                let is_boundary =
                    (h & mask) == 0 || chunk_len >= self.config.max_chunk;
                if is_boundary {
                    let digest = ChunkDigest {
                        hash: fnv1a_64(&self.chunk_buf),
                        len: chunk_len,
                    };
                    self.chunk_buf.clear();
                    return Ok(Some(digest));
                }
            }
            // Current batch exhausted; loop to refill.
        }

        // EOF reached — emit trailing chunk if any data remains.
        if self.chunk_buf.is_empty() {
            return Ok(None);
        }
        let digest = ChunkDigest {
            hash: fnv1a_64(&self.chunk_buf),
            len: self.chunk_buf.len(),
        };
        self.chunk_buf.clear();
        Ok(Some(digest))
    }
}

// ---------------------------------------------------------------------------
// StreamFingerprint
// ---------------------------------------------------------------------------

/// A file-level fingerprint derived from its content-defined chunk hashes.
///
/// The fingerprint is robust against small edits: only the chunks that actually
/// changed will differ, so the Jaccard similarity between two near-identical
/// files remains high.
#[derive(Debug, Clone)]
pub struct StreamFingerprint {
    /// Ordered list of chunk digests.
    pub chunks: Vec<ChunkDigest>,
    /// Total bytes processed.
    pub total_bytes: u64,
}

impl StreamFingerprint {
    /// Number of chunks in the fingerprint.
    #[must_use]
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Set of unique chunk hashes (for Jaccard computation).
    #[must_use]
    pub fn chunk_set(&self) -> HashSet<u64> {
        self.chunks.iter().map(|c| c.hash).collect()
    }

    /// Compute the Jaccard similarity between this fingerprint and another.
    ///
    /// `J(A,B) = |A ∩ B| / |A ∪ B|`
    ///
    /// Returns 1.0 when both fingerprints are identical, 0.0 when completely
    /// disjoint.
    #[must_use]
    pub fn jaccard(&self, other: &Self) -> f64 {
        let a = self.chunk_set();
        let b = other.chunk_set();
        if a.is_empty() && b.is_empty() {
            return 1.0;
        }
        let intersection = a.intersection(&b).count();
        let union = a.union(&b).count();
        if union == 0 {
            return 1.0;
        }
        intersection as f64 / union as f64
    }
}

// ---------------------------------------------------------------------------
// StreamDedupIndex
// ---------------------------------------------------------------------------

/// Index of stream fingerprints for near-duplicate detection.
///
/// Files are added by name via [`ingest`](Self::ingest); duplicates are
/// retrieved via [`find_duplicates`](Self::find_duplicates).
#[derive(Debug)]
pub struct StreamDedupIndex {
    config: StreamChunkerConfig,
    entries: HashMap<String, StreamFingerprint>,
}

impl StreamDedupIndex {
    /// Create a new, empty index with the given chunker config.
    #[must_use]
    pub fn new(config: StreamChunkerConfig) -> Self {
        Self {
            config,
            entries: HashMap::new(),
        }
    }

    /// Ingest a stream and store its fingerprint under `name`.
    ///
    /// Returns the computed [`StreamFingerprint`].
    ///
    /// # Errors
    ///
    /// Propagates `io::Error` from `reader`.
    pub fn ingest<R: Read>(
        &mut self,
        name: &str,
        reader: R,
    ) -> io::Result<StreamFingerprint> {
        let chunker = StreamChunker::new(reader, self.config.clone());
        let chunks = chunker.collect_all()?;
        let total_bytes: u64 = chunks.iter().map(|c| c.len as u64).sum();
        let fp = StreamFingerprint { chunks, total_bytes };
        self.entries.insert(name.to_string(), fp.clone());
        Ok(fp)
    }

    /// Number of indexed entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return `true` if the index is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Compute the Jaccard similarity between two fingerprints.
    #[must_use]
    pub fn jaccard_similarity(
        &self,
        a: &StreamFingerprint,
        b: &StreamFingerprint,
    ) -> f64 {
        a.jaccard(b)
    }

    /// Find all pairs of indexed entries whose Jaccard similarity exceeds
    /// `threshold`.
    ///
    /// Returns a list of `(name_a, name_b, similarity)` tuples sorted by
    /// descending similarity.
    #[must_use]
    pub fn find_duplicates(&self, threshold: f64) -> Vec<(String, String, f64)> {
        let names: Vec<&String> = self.entries.keys().collect();
        let n = names.len();
        let mut pairs = Vec::new();

        for i in 0..n {
            for j in (i + 1)..n {
                let fp_a = &self.entries[names[i]];
                let fp_b = &self.entries[names[j]];
                let sim = fp_a.jaccard(fp_b);
                if sim >= threshold {
                    pairs.push((names[i].clone(), names[j].clone(), sim));
                }
            }
        }

        pairs.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        pairs
    }

    /// Retrieve the fingerprint stored under `name`, if any.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&StreamFingerprint> {
        self.entries.get(name)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn small_config() -> StreamChunkerConfig {
        StreamChunkerConfig {
            min_chunk: 64,
            max_chunk: 512,
            window_size: 16,
            mask_bits: 6, // average ≈ 64 bytes
        }
    }

    #[test]
    fn test_config_default_is_valid() {
        assert!(StreamChunkerConfig::default().is_valid());
    }

    #[test]
    fn test_config_small_is_valid() {
        assert!(small_config().is_valid());
    }

    #[test]
    fn test_boundary_mask() {
        let cfg = StreamChunkerConfig {
            mask_bits: 8,
            ..Default::default()
        };
        assert_eq!(cfg.boundary_mask(), 0xFF);
    }

    #[test]
    fn test_empty_stream_produces_no_chunks() {
        let cfg = small_config();
        let chunker = StreamChunker::new(Cursor::new(b""), cfg);
        let chunks = chunker.collect_all().expect("io should not fail");
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_small_data_single_chunk() {
        // Data smaller than min_chunk → emitted as one trailing chunk.
        let cfg = small_config(); // min_chunk = 64
        let data = vec![0xABu8; 32]; // 32 bytes < 64
        let chunker = StreamChunker::new(Cursor::new(data), cfg);
        let chunks = chunker.collect_all().expect("ok");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len, 32);
    }

    #[test]
    fn test_large_data_multiple_chunks() {
        let cfg = small_config();
        // 8 KiB of repeating data should produce several chunks.
        let data = vec![0x5Au8; 8192];
        let chunker = StreamChunker::new(Cursor::new(data.clone()), cfg);
        let chunks = chunker.collect_all().expect("ok");
        // Total bytes must equal data length.
        let total: usize = chunks.iter().map(|c| c.len).sum();
        assert_eq!(total, 8192);
    }

    #[test]
    fn test_deterministic_chunking() {
        let cfg = small_config();
        let data: Vec<u8> = (0..4096_u16).map(|i| (i % 251) as u8).collect();
        let c1 = StreamChunker::new(Cursor::new(data.clone()), cfg.clone())
            .collect_all()
            .expect("ok");
        let c2 = StreamChunker::new(Cursor::new(data), cfg)
            .collect_all()
            .expect("ok");
        assert_eq!(c1, c2, "chunking must be deterministic");
    }

    #[test]
    fn test_identical_streams_jaccard_one() {
        let cfg = small_config();
        let data = vec![0x7Fu8; 4096];
        let mut index = StreamDedupIndex::new(cfg);
        let fp1 = index.ingest("a", Cursor::new(data.clone())).expect("ok");
        let fp2 = index.ingest("b", Cursor::new(data)).expect("ok");
        let sim = index.jaccard_similarity(&fp1, &fp2);
        assert!(
            (sim - 1.0).abs() < 1e-9,
            "identical streams must have Jaccard = 1.0, got {sim}"
        );
    }

    #[test]
    fn test_completely_different_streams_jaccard_near_zero() {
        let cfg = small_config();
        let data_a = vec![0x00u8; 4096];
        let data_b = vec![0xFFu8; 4096];
        let mut index = StreamDedupIndex::new(cfg);
        let fp_a = index.ingest("a", Cursor::new(data_a)).expect("ok");
        let fp_b = index.ingest("b", Cursor::new(data_b)).expect("ok");
        let sim = index.jaccard_similarity(&fp_a, &fp_b);
        // Chunks should be different → low similarity.
        assert!(sim < 0.5, "different data should have low Jaccard, got {sim}");
    }

    #[test]
    fn test_find_duplicates_returns_pairs_above_threshold() {
        let cfg = small_config();
        let data = vec![0xCCu8; 2048];
        let mut index = StreamDedupIndex::new(cfg);
        index.ingest("x", Cursor::new(data.clone())).expect("ok");
        index.ingest("y", Cursor::new(data)).expect("ok");

        let pairs = index.find_duplicates(0.9);
        assert!(!pairs.is_empty());
        let (ref na, ref nb, sim) = pairs[0];
        // names may be in any order
        assert!(
            (na == "x" || na == "y") && (nb == "x" || nb == "y") && na != nb,
            "pair names should be x and y"
        );
        assert!(sim >= 0.9);
    }

    #[test]
    fn test_find_duplicates_no_pairs_above_high_threshold() {
        let cfg = small_config();
        let mut index = StreamDedupIndex::new(cfg);
        index.ingest("p", Cursor::new(vec![0x11u8; 2048])).expect("ok");
        index.ingest("q", Cursor::new(vec![0x22u8; 2048])).expect("ok");
        // Very high threshold; different data won't meet it.
        let pairs = index.find_duplicates(0.99);
        assert!(
            pairs.is_empty() || pairs.iter().all(|(_, _, s)| *s >= 0.99),
            "all returned pairs must meet the threshold"
        );
    }

    #[test]
    fn test_index_len_and_is_empty() {
        let mut index = StreamDedupIndex::new(small_config());
        assert!(index.is_empty());
        index.ingest("file", Cursor::new(vec![1u8; 100])).expect("ok");
        assert_eq!(index.len(), 1);
        assert!(!index.is_empty());
    }

    #[test]
    fn test_get_fingerprint_after_ingest() {
        let mut index = StreamDedupIndex::new(small_config());
        index.ingest("myfile", Cursor::new(vec![42u8; 512])).expect("ok");
        let fp = index.get("myfile");
        assert!(fp.is_some());
        assert!(fp.unwrap().chunk_count() >= 1);
    }

    #[test]
    fn test_fnv1a_deterministic() {
        let h1 = fnv1a_64(b"hello world");
        let h2 = fnv1a_64(b"hello world");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_total_bytes_matches_data_length() {
        let cfg = small_config();
        let data = vec![0x33u8; 3333];
        let mut index = StreamDedupIndex::new(cfg);
        let fp = index.ingest("sz", Cursor::new(data)).expect("ok");
        assert_eq!(fp.total_bytes, 3333);
    }
}
