//! Deduplicating writer that skips duplicate data blocks using content hashing.
//!
//! [`DedupWriter`] wraps any `std::io::Write` target and maintains a hash set
//! of previously written blocks (using a pure-Rust FNV-1a hash).  When a block
//! whose hash has already been seen arrives, the write is elided and a counter
//! is incremented.
//!
//! # Note
//! The deduplication is best-effort: hash collisions are theoretically possible
//! but unlikely for practical media data.

#![allow(dead_code)]

use std::collections::HashSet;
use std::io::{self, Write};

// ─── FNV-1a (64-bit) ─────────────────────────────────────────────────────────

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Compute a 64-bit FNV-1a hash of `data`.
#[must_use]
pub fn fnv1a_64(data: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET;
    for &byte in data {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

// ─── DedupStats ──────────────────────────────────────────────────────────────

/// Statistics collected by [`DedupWriter`].
#[derive(Debug, Clone, Default)]
pub struct DedupStats {
    /// Total number of write calls.
    pub total_writes: u64,
    /// Number of writes that were deduplicated (elided).
    pub dedup_writes: u64,
    /// Total bytes that would have been written (including deduplicated bytes).
    pub total_bytes: u64,
    /// Bytes actually written (after deduplication).
    pub written_bytes: u64,
}

impl DedupStats {
    /// Returns the fraction of writes that were deduplicated (0.0–1.0).
    #[must_use]
    pub fn dedup_ratio(&self) -> f64 {
        if self.total_writes == 0 {
            0.0
        } else {
            self.dedup_writes as f64 / self.total_writes as f64
        }
    }

    /// Returns the byte savings due to deduplication.
    #[must_use]
    pub fn bytes_saved(&self) -> u64 {
        self.total_bytes.saturating_sub(self.written_bytes)
    }
}

// ─── DedupWriter ─────────────────────────────────────────────────────────────

/// A writer that deduplicates blocks of data by content hash.
///
/// Each call to `write` computes an FNV-1a hash of the buffer.  If the hash
/// has been seen before the write is skipped; otherwise the data is forwarded
/// to the inner writer and the hash is recorded.
///
/// # Example
///
/// ```rust
/// use oximedia_io::dedup_writer::DedupWriter;
/// use std::io::Write;
///
/// let mut buf = Vec::new();
/// let mut writer = DedupWriter::new(&mut buf);
///
/// writer.write_all(b"hello").unwrap();
/// writer.write_all(b"hello").unwrap(); // duplicate — skipped
/// writer.write_all(b"world").unwrap();
///
/// assert_eq!(writer.stats().total_writes, 3);
/// assert_eq!(writer.stats().dedup_writes, 1);
/// ```
pub struct DedupWriter<W: Write> {
    inner: W,
    seen: HashSet<u64>,
    stats: DedupStats,
}

impl<W: Write> DedupWriter<W> {
    /// Wrap `inner` in a new deduplicating writer.
    pub fn new(inner: W) -> Self {
        Self {
            inner,
            seen: HashSet::new(),
            stats: DedupStats::default(),
        }
    }

    /// Return a reference to the collected statistics.
    #[must_use]
    pub fn stats(&self) -> &DedupStats {
        &self.stats
    }

    /// Consume the writer and return the inner writer and final statistics.
    pub fn into_inner(self) -> (W, DedupStats) {
        (self.inner, self.stats)
    }

    /// Clear the seen-hash set (reset deduplication state without resetting stats).
    pub fn reset(&mut self) {
        self.seen.clear();
    }

    /// Clear the seen-hash set **and** reset all statistics counters.
    pub fn reset_all(&mut self) {
        self.seen.clear();
        self.stats = DedupStats::default();
    }
}

impl<W: Write> Write for DedupWriter<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let len = buf.len();
        self.stats.total_writes += 1;
        self.stats.total_bytes += len as u64;

        let hash = fnv1a_64(buf);
        if self.seen.contains(&hash) {
            self.stats.dedup_writes += 1;
            return Ok(len);
        }

        self.seen.insert(hash);
        let n = self.inner.write(buf)?;
        self.stats.written_bytes += n as u64;
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_fnv1a_deterministic() {
        let h1 = fnv1a_64(b"hello world");
        let h2 = fnv1a_64(b"hello world");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_fnv1a_different_data() {
        let h1 = fnv1a_64(b"hello");
        let h2 = fnv1a_64(b"world");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_dedup_skips_duplicate() {
        let mut buf = Vec::new();
        let mut writer = DedupWriter::new(&mut buf);

        writer.write_all(b"hello").expect("write should succeed");
        writer.write_all(b"hello").expect("write should succeed");
        writer.write_all(b"world").expect("write should succeed");

        assert_eq!(writer.stats().total_writes, 3);
        assert_eq!(writer.stats().dedup_writes, 1);
        assert_eq!(&buf, b"helloworld");
    }

    #[test]
    fn test_stats_dedup_ratio() {
        let mut buf = Vec::new();
        let mut writer = DedupWriter::new(&mut buf);

        writer.write_all(b"x").expect("write should succeed");
        writer.write_all(b"x").expect("write should succeed");
        writer.write_all(b"x").expect("write should succeed");

        let stats = writer.stats();
        assert_eq!(stats.total_writes, 3);
        assert_eq!(stats.dedup_writes, 2);
        assert!((stats.dedup_ratio() - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_bytes_saved() {
        let mut buf = Vec::new();
        let mut writer = DedupWriter::new(&mut buf);
        writer.write_all(b"hello").expect("write should succeed");
        writer.write_all(b"hello").expect("write should succeed");

        let stats = writer.stats().clone();
        assert_eq!(stats.bytes_saved(), 5);
    }

    #[test]
    fn test_reset_clears_seen_set() {
        let mut buf = Vec::new();
        let mut writer = DedupWriter::new(&mut buf);
        writer.write_all(b"data").expect("write should succeed");
        writer.reset();
        writer.write_all(b"data").expect("write should succeed");
        assert_eq!(writer.stats().dedup_writes, 0);
    }

    #[test]
    fn test_reset_all_clears_stats() {
        let mut buf = Vec::new();
        let mut writer = DedupWriter::new(&mut buf);
        writer.write_all(b"abc").expect("write should succeed");
        writer.write_all(b"abc").expect("write should succeed");
        assert_eq!(writer.stats().total_writes, 2);
        writer.reset_all();
        assert_eq!(writer.stats().total_writes, 0);
        assert_eq!(writer.stats().dedup_writes, 0);
        assert_eq!(writer.stats().total_bytes, 0);
        assert_eq!(writer.stats().written_bytes, 0);
        // After reset_all, writing same data again is NOT a duplicate
        writer.write_all(b"abc").expect("write should succeed");
        assert_eq!(writer.stats().dedup_writes, 0);
    }

    #[test]
    fn test_into_inner_returns_inner_and_stats() {
        let buf = Vec::new();
        let mut writer = DedupWriter::new(buf);
        writer.write_all(b"hello").expect("write should succeed");
        writer.write_all(b"world").expect("write should succeed");
        writer.write_all(b"hello").expect("write should succeed");
        let (inner, stats) = writer.into_inner();
        assert_eq!(&inner, b"helloworld");
        assert_eq!(stats.total_writes, 3);
        assert_eq!(stats.dedup_writes, 1);
    }

    #[test]
    fn test_dedup_ratio_no_writes() {
        let buf = Vec::new();
        let writer = DedupWriter::new(buf);
        assert!((writer.stats().dedup_ratio() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_flush_propagates() {
        use std::io::Write;
        let buf = Vec::new();
        let mut writer = DedupWriter::new(buf);
        // flush on a Vec should always succeed
        assert!(writer.flush().is_ok());
    }

    #[test]
    fn test_unique_blocks_all_written() {
        let mut buf = Vec::new();
        let mut writer = DedupWriter::new(&mut buf);
        for i in 0u8..8 {
            let block = vec![i; 4];
            writer.write_all(&block).expect("write should succeed");
        }
        assert_eq!(writer.stats().total_writes, 8);
        assert_eq!(writer.stats().dedup_writes, 0);
        assert_eq!(writer.stats().written_bytes, 32);
        assert_eq!(writer.stats().bytes_saved(), 0);
    }
}
