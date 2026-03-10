//! I/O verification utilities for data integrity.
//!
//! Provides byte-level verification helpers:
//! - [`VerifyWriter`]: a tee-writer that computes a running checksum while writing
//! - [`compare_streams`]: byte-by-byte comparison of two readers
//! - [`PatternFill`]: fills a writer with a repeating byte pattern (test utility)

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::io::{self, Read, Write};

// ---------------------------------------------------------------------------
// Simple FNV-1a 64-bit hash (no external deps)
// ---------------------------------------------------------------------------

/// FNV-1a 64-bit offset basis.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a 64-bit prime.
const FNV_PRIME: u64 = 0x0100_0000_01b3;

/// Compute FNV-1a 64-bit hash of `data`.
#[must_use]
pub fn fnv1a_64(data: &[u8]) -> u64 {
    let mut h = FNV_OFFSET;
    for &b in data {
        h ^= u64::from(b);
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// Incrementally compute FNV-1a by feeding one byte at a time.
#[derive(Debug, Clone)]
pub struct Fnv1aHasher {
    /// Current hash state.
    state: u64,
}

impl Fnv1aHasher {
    /// Create a new hasher with the standard offset basis.
    #[must_use]
    pub fn new() -> Self {
        Self { state: FNV_OFFSET }
    }

    /// Feed a single byte.
    pub fn update_byte(&mut self, b: u8) {
        self.state ^= u64::from(b);
        self.state = self.state.wrapping_mul(FNV_PRIME);
    }

    /// Feed a slice of bytes.
    pub fn update(&mut self, data: &[u8]) {
        for &b in data {
            self.update_byte(b);
        }
    }

    /// Return the current hash value.
    #[must_use]
    pub fn finish(&self) -> u64 {
        self.state
    }

    /// Reset to initial state.
    pub fn reset(&mut self) {
        self.state = FNV_OFFSET;
    }
}

impl Default for Fnv1aHasher {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// VerifyWriter
// ---------------------------------------------------------------------------

/// A writer that computes a running FNV-1a checksum on all bytes written.
///
/// After writing is complete, call [`checksum`](VerifyWriter::checksum) to
/// retrieve the 64-bit digest of everything written through this wrapper.
pub struct VerifyWriter<W> {
    /// Inner writer.
    inner: W,
    /// Running hasher.
    hasher: Fnv1aHasher,
    /// Total bytes written.
    bytes_written: u64,
}

impl<W: Write> VerifyWriter<W> {
    /// Wrap `inner` with integrity tracking.
    pub fn new(inner: W) -> Self {
        Self {
            inner,
            hasher: Fnv1aHasher::new(),
            bytes_written: 0,
        }
    }

    /// Return the current FNV-1a checksum.
    #[must_use]
    pub fn checksum(&self) -> u64 {
        self.hasher.finish()
    }

    /// Return total bytes written.
    #[must_use]
    pub fn bytes_written(&self) -> u64 {
        self.bytes_written
    }

    /// Consume and return the inner writer.
    #[must_use]
    pub fn into_inner(self) -> W {
        self.inner
    }
}

impl<W: Write> Write for VerifyWriter<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.hasher.update(&buf[..n]);
        self.bytes_written += n as u64;
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

// ---------------------------------------------------------------------------
// Stream comparison
// ---------------------------------------------------------------------------

/// Result of comparing two byte streams.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompareResult {
    /// Both streams are identical.
    Equal {
        /// Total bytes compared.
        bytes: u64,
    },
    /// Streams differ at the given offset.
    DifferentAt {
        /// Byte offset where the first difference was found.
        offset: u64,
        /// Byte value from the first stream.
        byte_a: u8,
        /// Byte value from the second stream.
        byte_b: u8,
    },
    /// Streams have different lengths (one ended before the other).
    DifferentLength {
        /// Byte offset where one stream ended.
        shorter_len: u64,
    },
}

impl CompareResult {
    /// Return `true` if the streams are equal.
    #[must_use]
    pub fn is_equal(&self) -> bool {
        matches!(self, Self::Equal { .. })
    }
}

/// Compare two readers byte-by-byte and return the result.
///
/// Reads both streams in 8 KiB blocks for efficiency.
///
/// # Errors
///
/// Returns an `io::Error` if either reader produces an error.
pub fn compare_streams<A: Read, B: Read>(mut a: A, mut b: B) -> io::Result<CompareResult> {
    let mut buf_a = [0u8; 8192];
    let mut buf_b = [0u8; 8192];
    let mut offset: u64 = 0;

    loop {
        let na = read_full(&mut a, &mut buf_a)?;
        let nb = read_full(&mut b, &mut buf_b)?;

        let min_len = na.min(nb);
        for i in 0..min_len {
            if buf_a[i] != buf_b[i] {
                return Ok(CompareResult::DifferentAt {
                    offset: offset + i as u64,
                    byte_a: buf_a[i],
                    byte_b: buf_b[i],
                });
            }
        }

        if na != nb {
            return Ok(CompareResult::DifferentLength {
                shorter_len: offset + min_len as u64,
            });
        }

        if na == 0 {
            return Ok(CompareResult::Equal { bytes: offset });
        }

        offset += na as u64;
    }
}

/// Read as many bytes as possible to fill `buf`, handling short reads.
fn read_full<R: Read>(reader: &mut R, buf: &mut [u8]) -> io::Result<usize> {
    let mut pos = 0;
    while pos < buf.len() {
        match reader.read(&mut buf[pos..]) {
            Ok(0) => break,
            Ok(n) => pos += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(pos)
}

// ---------------------------------------------------------------------------
// PatternFill
// ---------------------------------------------------------------------------

/// Generates a repeating byte-pattern into a writer (useful for testing).
pub struct PatternFill {
    /// The pattern to repeat.
    pattern: Vec<u8>,
}

impl PatternFill {
    /// Create from a byte slice pattern.
    #[must_use]
    pub fn new(pattern: &[u8]) -> Self {
        let pattern = if pattern.is_empty() {
            vec![0]
        } else {
            pattern.to_vec()
        };
        Self { pattern }
    }

    /// Write exactly `total` bytes of the repeating pattern into `w`.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the writer fails.
    pub fn write_to<W: Write>(&self, w: &mut W, total: usize) -> io::Result<()> {
        let mut remaining = total;
        while remaining > 0 {
            let chunk = remaining.min(self.pattern.len());
            w.write_all(&self.pattern[..chunk])?;
            remaining -= chunk;
        }
        Ok(())
    }

    /// Return the underlying pattern.
    #[must_use]
    pub fn pattern(&self) -> &[u8] {
        &self.pattern
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_fnv1a_empty() {
        assert_eq!(fnv1a_64(b""), FNV_OFFSET);
    }

    #[test]
    fn test_fnv1a_deterministic() {
        let h1 = fnv1a_64(b"hello");
        let h2 = fnv1a_64(b"hello");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_fnv1a_different() {
        assert_ne!(fnv1a_64(b"abc"), fnv1a_64(b"abd"));
    }

    #[test]
    fn test_fnv1a_hasher_incremental() {
        let mut h = Fnv1aHasher::new();
        h.update(b"hel");
        h.update(b"lo");
        assert_eq!(h.finish(), fnv1a_64(b"hello"));
    }

    #[test]
    fn test_fnv1a_hasher_reset() {
        let mut h = Fnv1aHasher::new();
        h.update(b"data");
        h.reset();
        assert_eq!(h.finish(), FNV_OFFSET);
    }

    #[test]
    fn test_verify_writer_basic() {
        let mut out = Vec::new();
        let mut vw = VerifyWriter::new(&mut out);
        vw.write_all(b"test data").expect("failed to write");
        assert_eq!(vw.bytes_written(), 9);
        assert_eq!(vw.checksum(), fnv1a_64(b"test data"));
    }

    #[test]
    fn test_verify_writer_empty() {
        let mut out = Vec::new();
        let vw = VerifyWriter::new(&mut out);
        assert_eq!(vw.bytes_written(), 0);
        assert_eq!(vw.checksum(), FNV_OFFSET);
    }

    #[test]
    fn test_compare_equal() {
        let a = Cursor::new(b"identical data".to_vec());
        let b = Cursor::new(b"identical data".to_vec());
        let result = compare_streams(a, b).expect("compare should succeed");
        assert!(result.is_equal());
        assert_eq!(result, CompareResult::Equal { bytes: 14 });
    }

    #[test]
    fn test_compare_different_byte() {
        let a = Cursor::new(b"abc".to_vec());
        let b = Cursor::new(b"axc".to_vec());
        let result = compare_streams(a, b).expect("compare should succeed");
        assert_eq!(
            result,
            CompareResult::DifferentAt {
                offset: 1,
                byte_a: b'b',
                byte_b: b'x',
            }
        );
    }

    #[test]
    fn test_compare_different_length() {
        let a = Cursor::new(b"short".to_vec());
        let b = Cursor::new(b"short and longer".to_vec());
        let result = compare_streams(a, b).expect("compare should succeed");
        assert_eq!(result, CompareResult::DifferentLength { shorter_len: 5 });
    }

    #[test]
    fn test_compare_both_empty() {
        let a = Cursor::new(Vec::<u8>::new());
        let b = Cursor::new(Vec::<u8>::new());
        let result = compare_streams(a, b).expect("compare should succeed");
        assert!(result.is_equal());
    }

    #[test]
    fn test_pattern_fill_basic() {
        let pf = PatternFill::new(b"ab");
        let mut out = Vec::new();
        pf.write_to(&mut out, 7).expect("write_to should succeed");
        assert_eq!(out, b"abababa");
    }

    #[test]
    fn test_pattern_fill_empty_pattern() {
        let pf = PatternFill::new(b"");
        assert_eq!(pf.pattern(), &[0u8]);
    }

    #[test]
    fn test_pattern_fill_zero_len() {
        let pf = PatternFill::new(b"xyz");
        let mut out = Vec::new();
        pf.write_to(&mut out, 0).expect("write_to should succeed");
        assert!(out.is_empty());
    }
}
