//! Chunked upload utilities: split data into fixed-size chunks and reassemble.
//!
//! [`ChunkedUploadSplitter`] divides a byte slice into equal-sized pieces
//! (the last piece may be shorter).  [`ChunkedUploadReassembler`] concatenates
//! an ordered slice of parts back into the original data.
//!
//! # Use-cases
//!
//! - Large-file HTTP multipart uploads where a server requires a fixed maximum
//!   part size.
//! - Resumable uploads: each chunk can be independently verified and re-sent.
//! - Parallel upload pipelines where chunks are dispatched to workers.
//!
//! # Example
//!
//! ```
//! use oximedia_io::chunked_upload::{ChunkedUploadSplitter, ChunkedUploadReassembler};
//!
//! let data: Vec<u8> = (0u8..100).collect();
//! let splitter = ChunkedUploadSplitter::new(30);
//! let chunks = splitter.split(&data);
//!
//! assert_eq!(chunks.len(), 4); // 30 + 30 + 30 + 10
//!
//! let reassembled = ChunkedUploadReassembler::reassemble(&chunks);
//! assert_eq!(reassembled, data);
//! ```

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// ChunkedUploadSplitter
// ---------------------------------------------------------------------------

/// Splits a byte slice into fixed-size chunks for chunked upload.
///
/// The last chunk may be smaller than `chunk_size` if the data length is not
/// an exact multiple of `chunk_size`.
#[derive(Debug, Clone)]
pub struct ChunkedUploadSplitter {
    /// Maximum bytes per chunk (clamped to ≥ 1).
    chunk_size: usize,
}

impl ChunkedUploadSplitter {
    /// Create a splitter that produces chunks of at most `chunk_size` bytes.
    ///
    /// `chunk_size` is clamped to a minimum of `1`.
    #[must_use]
    pub fn new(chunk_size: usize) -> Self {
        Self {
            chunk_size: chunk_size.max(1),
        }
    }

    /// Split `data` into a `Vec` of owned byte chunks.
    ///
    /// - Returns an empty `Vec` when `data` is empty.
    /// - The final element may be shorter than `chunk_size`.
    /// - All other elements are exactly `chunk_size` bytes.
    #[must_use]
    pub fn split(&self, data: &[u8]) -> Vec<Vec<u8>> {
        if data.is_empty() {
            return Vec::new();
        }
        data.chunks(self.chunk_size)
            .map(|c| c.to_vec())
            .collect()
    }

    /// The configured maximum chunk size in bytes.
    #[must_use]
    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    /// Compute the number of chunks needed for `data_len` bytes.
    #[must_use]
    pub fn chunk_count(&self, data_len: usize) -> usize {
        if data_len == 0 {
            return 0;
        }
        (data_len + self.chunk_size - 1) / self.chunk_size
    }
}

// ---------------------------------------------------------------------------
// ChunkedUploadReassembler
// ---------------------------------------------------------------------------

/// Reassembles an ordered sequence of byte chunks into the original data.
///
/// This is a stateless utility struct; all logic is in the associated
/// [`reassemble`](ChunkedUploadReassembler::reassemble) function.
pub struct ChunkedUploadReassembler;

impl ChunkedUploadReassembler {
    /// Concatenate `parts` in order and return the reassembled byte vector.
    ///
    /// - Returns an empty `Vec` when `parts` is empty.
    /// - The returned buffer is pre-allocated to the exact total size for
    ///   efficiency.
    #[must_use]
    pub fn reassemble(parts: &[Vec<u8>]) -> Vec<u8> {
        let total: usize = parts.iter().map(|p| p.len()).sum();
        let mut out = Vec::with_capacity(total);
        for part in parts {
            out.extend_from_slice(part);
        }
        out
    }

    /// Validate that `parts` match an expected total byte count.
    ///
    /// Returns `true` if the sum of part lengths equals `expected_total`.
    #[must_use]
    pub fn validate_total(parts: &[Vec<u8>], expected_total: usize) -> bool {
        parts.iter().map(|p| p.len()).sum::<usize>() == expected_total
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── ChunkedUploadSplitter ─────────────────────────────────────────────────

    #[test]
    fn test_split_empty_data() {
        let s = ChunkedUploadSplitter::new(100);
        assert!(s.split(&[]).is_empty());
    }

    #[test]
    fn test_split_exact_multiple() {
        let data: Vec<u8> = (0..9).collect(); // 9 bytes
        let s = ChunkedUploadSplitter::new(3);
        let chunks = s.split(&data);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], &[0, 1, 2]);
        assert_eq!(chunks[1], &[3, 4, 5]);
        assert_eq!(chunks[2], &[6, 7, 8]);
    }

    #[test]
    fn test_split_non_exact_multiple() {
        let data: Vec<u8> = (0..10).collect(); // 10 bytes
        let s = ChunkedUploadSplitter::new(3);
        let chunks = s.split(&data);
        assert_eq!(chunks.len(), 4); // 3 + 3 + 3 + 1
        assert_eq!(chunks[3], &[9]);
    }

    #[test]
    fn test_split_larger_than_data() {
        let data = vec![1u8, 2, 3];
        let s = ChunkedUploadSplitter::new(1000);
        let chunks = s.split(&data);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], data);
    }

    #[test]
    fn test_split_chunk_size_one() {
        let data = vec![10u8, 20, 30];
        let s = ChunkedUploadSplitter::new(1);
        let chunks = s.split(&data);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], &[10]);
        assert_eq!(chunks[1], &[20]);
        assert_eq!(chunks[2], &[30]);
    }

    #[test]
    fn test_split_chunk_size_zero_clamped_to_one() {
        let data = vec![0u8; 5];
        let s = ChunkedUploadSplitter::new(0); // clamped to 1
        assert_eq!(s.chunk_size(), 1);
        let chunks = s.split(&data);
        assert_eq!(chunks.len(), 5);
    }

    #[test]
    fn test_chunk_count() {
        let s = ChunkedUploadSplitter::new(10);
        assert_eq!(s.chunk_count(0), 0);
        assert_eq!(s.chunk_count(10), 1);
        assert_eq!(s.chunk_count(11), 2);
        assert_eq!(s.chunk_count(20), 2);
        assert_eq!(s.chunk_count(21), 3);
    }

    // ── ChunkedUploadReassembler ──────────────────────────────────────────────

    #[test]
    fn test_reassemble_empty() {
        let parts: &[Vec<u8>] = &[];
        assert!(ChunkedUploadReassembler::reassemble(parts).is_empty());
    }

    #[test]
    fn test_reassemble_single_chunk() {
        let parts = vec![vec![1u8, 2, 3]];
        assert_eq!(
            ChunkedUploadReassembler::reassemble(&parts),
            vec![1u8, 2, 3]
        );
    }

    #[test]
    fn test_reassemble_multiple_chunks() {
        let parts = vec![vec![1u8, 2], vec![3u8, 4], vec![5u8]];
        assert_eq!(
            ChunkedUploadReassembler::reassemble(&parts),
            vec![1u8, 2, 3, 4, 5]
        );
    }

    #[test]
    fn test_split_then_reassemble_roundtrip() {
        let original: Vec<u8> = (0u8..=255).collect();
        let s = ChunkedUploadSplitter::new(64);
        let chunks = s.split(&original);
        let rebuilt = ChunkedUploadReassembler::reassemble(&chunks);
        assert_eq!(rebuilt, original);
    }

    #[test]
    fn test_validate_total_correct() {
        let parts = vec![vec![0u8; 100], vec![0u8; 56]];
        assert!(ChunkedUploadReassembler::validate_total(&parts, 156));
    }

    #[test]
    fn test_validate_total_wrong() {
        let parts = vec![vec![0u8; 100]];
        assert!(!ChunkedUploadReassembler::validate_total(&parts, 200));
    }

    #[test]
    fn test_large_roundtrip() {
        let original: Vec<u8> = (0u8..=255).cycle().take(10_000).collect();
        let s = ChunkedUploadSplitter::new(1024);
        let chunks = s.split(&original);
        let rebuilt = ChunkedUploadReassembler::reassemble(&chunks);
        assert_eq!(rebuilt, original);
    }
}
