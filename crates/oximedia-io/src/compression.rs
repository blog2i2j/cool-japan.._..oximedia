//! Lightweight I/O compression utilities.
//!
//! Provides run-length encoding (RLE) for compressing repetitive byte streams,
//! compression level selection, and statistics tracking.

#![allow(dead_code)]

// ──────────────────────────────────────────────────────────────────────────────
// CompressionLevel
// ──────────────────────────────────────────────────────────────────────────────

/// Compression effort level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionLevel {
    /// Prioritise speed over compression ratio.
    Fast,
    /// Balanced speed / compression ratio.
    Default,
    /// Maximise compression ratio.
    Best,
}

impl CompressionLevel {
    /// Numeric effort value (1 = fastest, 9 = best compression).
    #[must_use]
    pub fn effort(self) -> u32 {
        match self {
            Self::Fast => 1,
            Self::Default => 5,
            Self::Best => 9,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// RLE codec
// ──────────────────────────────────────────────────────────────────────────────

/// Encode `data` using run-length encoding.
///
/// The output is a sequence of `[count, value]` byte pairs where each run is
/// at most 255 bytes long.  Empty input produces empty output.
#[must_use]
pub fn rle_encode(data: &[u8]) -> Vec<u8> {
    if data.is_empty() {
        return Vec::new();
    }

    let mut out = Vec::new();
    let mut iter = data.iter().copied();

    // Safe: data is non-empty, so the iterator has at least one element.
    let Some(mut current) = iter.next() else {
        return out;
    };
    let mut count: u8 = 1;

    for byte in iter {
        if byte == current && count < u8::MAX {
            count += 1;
        } else {
            out.push(count);
            out.push(current);
            current = byte;
            count = 1;
        }
    }
    out.push(count);
    out.push(current);
    out
}

/// Decode data produced by [`rle_encode`].
///
/// Returns `Err` if the encoded stream has an odd number of bytes (malformed).
///
/// # Errors
///
/// Returns `Err(String)` if `data` has an odd number of bytes (malformed RLE stream).
pub fn rle_decode(data: &[u8]) -> Result<Vec<u8>, String> {
    if data.len() % 2 != 0 {
        return Err(format!(
            "RLE stream length must be even, got {} bytes",
            data.len()
        ));
    }

    let mut out = Vec::with_capacity(data.len());
    let mut i = 0;
    while i + 1 < data.len() {
        let count = data[i] as usize;
        let value = data[i + 1];
        for _ in 0..count {
            out.push(value);
        }
        i += 2;
    }
    Ok(out)
}

// ──────────────────────────────────────────────────────────────────────────────
// CompressionStats
// ──────────────────────────────────────────────────────────────────────────────

/// Statistics describing the result of a compression operation.
#[derive(Debug, Clone, Copy)]
pub struct CompressionStats {
    /// Number of bytes in the original (uncompressed) data.
    pub original_bytes: u64,
    /// Number of bytes in the compressed output.
    pub compressed_bytes: u64,
}

impl CompressionStats {
    /// Create new statistics.
    #[must_use]
    pub fn new(original_bytes: u64, compressed_bytes: u64) -> Self {
        Self {
            original_bytes,
            compressed_bytes,
        }
    }

    /// Compression ratio: `compressed / original`.
    ///
    /// Returns `1.0` when `original_bytes` is zero (no data, no savings).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn ratio(&self) -> f64 {
        if self.original_bytes == 0 {
            return 1.0;
        }
        self.compressed_bytes as f64 / self.original_bytes as f64
    }

    /// Percentage of space saved: `(1 - ratio) * 100`.
    ///
    /// May be negative if the compressed output is larger than the original.
    #[must_use]
    pub fn space_saved_pct(&self) -> f64 {
        (1.0 - self.ratio()) * 100.0
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // CompressionLevel ────────────────────────────────────────────────────────

    #[test]
    fn test_level_effort_ordering() {
        assert!(CompressionLevel::Fast.effort() < CompressionLevel::Default.effort());
        assert!(CompressionLevel::Default.effort() < CompressionLevel::Best.effort());
    }

    #[test]
    fn test_level_fast_effort() {
        assert_eq!(CompressionLevel::Fast.effort(), 1);
    }

    #[test]
    fn test_level_best_effort() {
        assert_eq!(CompressionLevel::Best.effort(), 9);
    }

    // rle_encode ──────────────────────────────────────────────────────────────

    #[test]
    fn test_rle_encode_empty() {
        assert!(rle_encode(&[]).is_empty());
    }

    #[test]
    fn test_rle_encode_single_byte() {
        assert_eq!(rle_encode(&[42]), vec![1, 42]);
    }

    #[test]
    fn test_rle_encode_run() {
        // 4 × 0xFF
        assert_eq!(rle_encode(&[0xFF, 0xFF, 0xFF, 0xFF]), vec![4, 0xFF]);
    }

    #[test]
    fn test_rle_encode_multiple_runs() {
        let input = vec![1, 1, 2, 2, 2];
        assert_eq!(rle_encode(&input), vec![2, 1, 3, 2]);
    }

    #[test]
    fn test_rle_encode_no_repetition() {
        let input = vec![1, 2, 3];
        assert_eq!(rle_encode(&input), vec![1, 1, 1, 2, 1, 3]);
    }

    // rle_decode ──────────────────────────────────────────────────────────────

    #[test]
    fn test_rle_decode_empty() {
        assert_eq!(
            rle_decode(&[]).expect("operation should succeed"),
            Vec::<u8>::new()
        );
    }

    #[test]
    fn test_rle_decode_single_pair() {
        assert_eq!(
            rle_decode(&[3, 7]).expect("operation should succeed"),
            vec![7, 7, 7]
        );
    }

    #[test]
    fn test_rle_roundtrip() {
        let original = vec![0u8, 0, 0, 1, 2, 2, 2, 3];
        let encoded = rle_encode(&original);
        let decoded = rle_decode(&encoded).expect("operation should succeed");
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_rle_decode_odd_length_error() {
        assert!(rle_decode(&[1]).is_err());
    }

    // CompressionStats ────────────────────────────────────────────────────────

    #[test]
    fn test_stats_ratio_half() {
        let s = CompressionStats::new(100, 50);
        assert!((s.ratio() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_stats_space_saved_pct() {
        let s = CompressionStats::new(100, 50);
        assert!((s.space_saved_pct() - 50.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_zero_original() {
        let s = CompressionStats::new(0, 0);
        assert!((s.ratio() - 1.0).abs() < 1e-9);
        assert!((s.space_saved_pct() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_expansion() {
        // Compressed is larger than original (typical for small / random data).
        let s = CompressionStats::new(10, 20);
        assert!(s.ratio() > 1.0);
        assert!(s.space_saved_pct() < 0.0);
    }
}
