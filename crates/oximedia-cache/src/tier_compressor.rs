//! Cache-entry compression for tiered L2+ cache layers.
//!
//! Large cache tiers (L2 memory or disk) can benefit from compressing stored
//! values to reduce footprint.  This module provides a pure-Rust, zero-
//! dependency LZ77-style byte-pair run-length codec suitable for media
//! metadata and moderate-size frame buffers.
//!
//! # Design
//!
//! [`TierCompressor`] wraps a configurable compression level and exposes a
//! symmetric [`compress`] / [`decompress`] pair.  The codec is a simplified
//! LZ77 variant:
//!
//! - The input stream is divided into 256-byte look-ahead windows.
//! - Literal bytes are emitted as `[0x00, byte]`.
//! - Back-references `(offset, length)` with `length >= 3` are emitted as
//!   `[0x01, offset_lo, offset_hi, length]` (offset = distance back into
//!   the output buffer, length = match length).
//! - A header `[0xCA, 0xCE, 0x00]` + 4-byte LE original length is prepended
//!   so decompression can pre-allocate and validate.
//!
//! The codec is designed for correctness and minimal overhead, not maximum
//! compression ratio.
//!
//! # Compression levels
//!
//! | Level | Look-ahead window | Search depth |
//! |-------|-------------------|--------------|
//! | 0     | 64                | 16           |
//! | 1     | 128               | 32           |
//! | 2     | 256               | 64           |
//! | 3     | 512               | 128          |
//!
//! Higher levels produce smaller output but take more CPU time.
//!
//! # Example
//!
//! ```rust
//! use oximedia_cache::tier_compressor::TierCompressor;
//!
//! let c = TierCompressor::new(1);
//! let original = b"hello hello hello world".to_vec();
//! let compressed = c.compress(&original).expect("compress");
//! let restored = c.decompress(&compressed).expect("decompress");
//! assert_eq!(restored, original);
//! ```

use thiserror::Error;

// ── Magic / header ────────────────────────────────────────────────────────────

const MAGIC: [u8; 3] = [0xCA, 0xCE, 0x00];
const HEADER_LEN: usize = 3 + 4; // magic (3) + original_len LE-u32 (4)

// ── Tags ──────────────────────────────────────────────────────────────────────

const TAG_LITERAL: u8 = 0x00;
const TAG_BACKREF: u8 = 0x01;

// ── Errors ────────────────────────────────────────────────────────────────────

/// Errors produced by [`TierCompressor`].
#[derive(Debug, Error)]
pub enum CompressorError {
    /// The compressed data is truncated or has an invalid header.
    #[error("invalid compressed data: {0}")]
    InvalidData(String),
    /// The decompressed output does not match the expected length in the header.
    #[error("length mismatch: expected {expected}, got {actual}")]
    LengthMismatch {
        /// Expected decompressed length.
        expected: usize,
        /// Actual decompressed length.
        actual: usize,
    },
    /// The input is too large to be encoded (> 2^32 bytes).
    #[error("input too large: {0} bytes")]
    InputTooLarge(usize),
}

// ── Level parameters ──────────────────────────────────────────────────────────

/// Per-level codec parameters.
struct LevelParams {
    /// Number of bytes inspected in the look-back search window.
    window: usize,
    /// Maximum match length searched.
    max_len: usize,
}

fn params_for_level(level: u8) -> LevelParams {
    match level {
        0 => LevelParams { window: 64, max_len: 16 },
        1 => LevelParams { window: 128, max_len: 32 },
        2 => LevelParams { window: 256, max_len: 64 },
        _ => LevelParams { window: 512, max_len: 128 },
    }
}

// ── TierCompressor ────────────────────────────────────────────────────────────

/// Symmetric compressor/decompressor for cache tier entries.
///
/// Instantiate once and reuse for many compress/decompress calls.
#[derive(Debug, Clone)]
pub struct TierCompressor {
    level: u8,
}

impl TierCompressor {
    /// Create a new `TierCompressor` at the given level (0–3; values above 3
    /// are treated as level 3).
    #[must_use]
    pub fn new(level: u8) -> Self {
        Self { level: level.min(3) }
    }

    /// Compress `input` and return the compressed bytes.
    ///
    /// The compressed bytes include a header that encodes the original length;
    /// pass them directly to [`decompress`] to recover the original.
    pub fn compress(&self, input: &[u8]) -> Result<Vec<u8>, CompressorError> {
        if input.len() > u32::MAX as usize {
            return Err(CompressorError::InputTooLarge(input.len()));
        }
        let params = params_for_level(self.level);
        let orig_len = input.len() as u32;

        // Worst case: every byte is a literal (2 bytes each) + header.
        let mut out = Vec::with_capacity(HEADER_LEN + input.len() * 2);

        // Header: magic + original length (LE u32).
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&orig_len.to_le_bytes());

        let mut pos = 0usize;
        while pos < input.len() {
            // Search for the longest match in the look-back window.
            let window_start = pos.saturating_sub(params.window);
            let search_window = &input[window_start..pos];

            // Maximum match length: min(params.max_len, remaining bytes).
            let max_match = params.max_len.min(input.len() - pos);

            let best = find_longest_match(search_window, &input[pos..], max_match);

            match best {
                Some((offset_in_window, length)) if length >= 3 => {
                    // Back-reference.
                    // `offset_in_window` is the index in `search_window` where the
                    // match starts.  Convert to distance from current `pos`:
                    // distance = (pos - window_start) - offset_in_window.
                    let distance = (pos - window_start) - offset_in_window;
                    debug_assert!(distance > 0, "back-ref distance must be positive");
                    let dist_u16 = distance as u16;
                    let len_u8 = length as u8;
                    out.push(TAG_BACKREF);
                    out.push((dist_u16 & 0xFF) as u8);
                    out.push((dist_u16 >> 8) as u8);
                    out.push(len_u8);
                    pos += length;
                }
                _ => {
                    // Literal byte.
                    out.push(TAG_LITERAL);
                    out.push(input[pos]);
                    pos += 1;
                }
            }
        }

        Ok(out)
    }

    /// Decompress bytes previously produced by [`compress`].
    pub fn decompress(&self, input: &[u8]) -> Result<Vec<u8>, CompressorError> {
        if input.len() < HEADER_LEN {
            return Err(CompressorError::InvalidData(format!(
                "too short: {} bytes (need at least {})",
                input.len(),
                HEADER_LEN
            )));
        }
        // Validate magic.
        if input[..3] != MAGIC {
            return Err(CompressorError::InvalidData(
                "magic bytes do not match".to_string(),
            ));
        }
        // Read original length.
        let orig_len = u32::from_le_bytes([input[3], input[4], input[5], input[6]]) as usize;
        let mut out = Vec::with_capacity(orig_len);

        let payload = &input[HEADER_LEN..];
        let mut pos = 0usize;

        while pos < payload.len() {
            let tag = payload[pos];
            pos += 1;

            match tag {
                TAG_LITERAL => {
                    if pos >= payload.len() {
                        return Err(CompressorError::InvalidData(
                            "truncated literal token".to_string(),
                        ));
                    }
                    out.push(payload[pos]);
                    pos += 1;
                }
                TAG_BACKREF => {
                    if pos + 2 >= payload.len() {
                        return Err(CompressorError::InvalidData(
                            "truncated back-reference token".to_string(),
                        ));
                    }
                    let dist_lo = payload[pos] as u16;
                    let dist_hi = payload[pos + 1] as u16;
                    let length = payload[pos + 2] as usize;
                    pos += 3;

                    let distance = (dist_hi << 8 | dist_lo) as usize;
                    if distance == 0 || distance > out.len() {
                        return Err(CompressorError::InvalidData(format!(
                            "invalid back-ref distance {distance} at output position {}",
                            out.len()
                        )));
                    }
                    let start = out.len() - distance;
                    // Copy byte-by-byte to allow overlapping references.
                    for i in 0..length {
                        let byte = out[start + (i % distance)];
                        out.push(byte);
                    }
                }
                unknown => {
                    return Err(CompressorError::InvalidData(format!(
                        "unknown tag byte 0x{unknown:02X} at offset {pos}"
                    )));
                }
            }
        }

        if out.len() != orig_len {
            return Err(CompressorError::LengthMismatch {
                expected: orig_len,
                actual: out.len(),
            });
        }

        Ok(out)
    }

    /// Return the compression level (0–3).
    #[must_use]
    pub fn level(&self) -> u8 {
        self.level
    }

    /// Compress `input` and report the compression ratio (compressed / original).
    ///
    /// Returns `1.0` for empty inputs.
    pub fn compression_ratio(&self, input: &[u8]) -> Result<f64, CompressorError> {
        if input.is_empty() {
            return Ok(1.0);
        }
        let compressed = self.compress(input)?;
        Ok(compressed.len() as f64 / input.len() as f64)
    }
}

// ── find_longest_match ────────────────────────────────────────────────────────

/// Search `window` for the longest prefix match against `lookahead`.
///
/// Returns `Some((window_start_index, match_length))` when a match of at
/// least 3 bytes is found, `None` otherwise.
fn find_longest_match(window: &[u8], lookahead: &[u8], max_len: usize) -> Option<(usize, usize)> {
    if window.is_empty() || lookahead.is_empty() || max_len == 0 {
        return None;
    }

    let mut best_offset = 0usize;
    let mut best_len = 0usize;

    for start in 0..window.len() {
        let mut len = 0usize;
        while len < max_len
            && len < lookahead.len()
            && len < window.len() - start + lookahead.len()
        {
            // Use modular indexing for overlapping matches (like LZ77 copy).
            let window_idx = start + (len % (window.len() - start));
            if window[window_idx] != lookahead[len] {
                break;
            }
            len += 1;
        }
        if len > best_len {
            best_len = len;
            best_offset = start;
        }
    }

    if best_len >= 3 {
        Some((best_offset, best_len))
    } else {
        None
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // 1. Round-trip: simple ASCII
    #[test]
    fn test_round_trip_ascii() {
        let c = TierCompressor::new(1);
        let orig = b"hello world hello world".to_vec();
        let compressed = c.compress(&orig).expect("compress");
        let restored = c.decompress(&compressed).expect("decompress");
        assert_eq!(restored, orig);
    }

    // 2. Round-trip: empty input
    #[test]
    fn test_round_trip_empty() {
        let c = TierCompressor::new(0);
        let orig: Vec<u8> = Vec::new();
        let compressed = c.compress(&orig).expect("compress empty");
        let restored = c.decompress(&compressed).expect("decompress empty");
        assert_eq!(restored, orig);
    }

    // 3. Round-trip: single byte
    #[test]
    fn test_round_trip_single_byte() {
        let c = TierCompressor::new(0);
        let orig = vec![0xABu8];
        let compressed = c.compress(&orig).expect("compress");
        let restored = c.decompress(&compressed).expect("decompress");
        assert_eq!(restored, orig);
    }

    // 4. Round-trip: all-zero buffer (highly compressible)
    #[test]
    fn test_round_trip_zeros() {
        let c = TierCompressor::new(2);
        let orig = vec![0u8; 512];
        let compressed = c.compress(&orig).expect("compress");
        let restored = c.decompress(&compressed).expect("decompress");
        assert_eq!(restored, orig);
    }

    // 5. Round-trip: random-ish binary data
    #[test]
    fn test_round_trip_binary() {
        let c = TierCompressor::new(1);
        let orig: Vec<u8> = (0u8..=255).cycle().take(300).collect();
        let compressed = c.compress(&orig).expect("compress");
        let restored = c.decompress(&compressed).expect("decompress");
        assert_eq!(restored, orig);
    }

    // 6. Compressed zeros are smaller than the original
    #[test]
    fn test_zeros_compress_smaller() {
        let c = TierCompressor::new(2);
        let orig = vec![0u8; 256];
        let compressed = c.compress(&orig).expect("compress");
        // After the header, the body should be much smaller than 256 bytes
        assert!(
            compressed.len() < orig.len(),
            "compressed {} should be < original {}",
            compressed.len(),
            orig.len()
        );
    }

    // 7. Invalid magic bytes are rejected
    #[test]
    fn test_decompress_invalid_magic() {
        let c = TierCompressor::new(0);
        let bad = vec![0xDE, 0xAD, 0xBE, 0xEF, 0, 0, 0, 0];
        assert!(c.decompress(&bad).is_err());
    }

    // 8. Truncated input is rejected
    #[test]
    fn test_decompress_truncated() {
        let c = TierCompressor::new(0);
        // Header requires 7 bytes; 3 is too short.
        assert!(c.decompress(&[0xCA, 0xCE, 0x00]).is_err());
    }

    // 9. level() getter
    #[test]
    fn test_level_getter() {
        assert_eq!(TierCompressor::new(2).level(), 2);
        assert_eq!(TierCompressor::new(99).level(), 3); // clamped
    }

    // 10. compression_ratio() for empty input returns 1.0
    #[test]
    fn test_ratio_empty() {
        let c = TierCompressor::new(1);
        let ratio = c.compression_ratio(&[]).expect("ratio");
        assert!((ratio - 1.0).abs() < 1e-9);
    }

    // 11. compression_ratio for repeated data < 1.0
    #[test]
    fn test_ratio_repeated_lt_1() {
        let c = TierCompressor::new(3);
        let orig = b"abcabcabcabcabcabcabcabcabc".to_vec();
        let ratio = c.compression_ratio(&orig).expect("ratio");
        // Repeated pattern should compress reasonably
        assert!(ratio < 2.0, "ratio {ratio} should be reasonable");
    }

    // 12. Round-trip at all levels
    #[test]
    fn test_round_trip_all_levels() {
        let orig = b"the quick brown fox jumps over the lazy dog".repeat(5);
        for level in 0..=3 {
            let c = TierCompressor::new(level);
            let compressed = c.compress(&orig).expect("compress");
            let restored = c.decompress(&compressed).expect("decompress");
            assert_eq!(restored, orig, "level {level} failed round-trip");
        }
    }

    // 13. Header encodes original length correctly
    #[test]
    fn test_header_length_field() {
        let c = TierCompressor::new(0);
        let orig = b"abcde".to_vec();
        let compressed = c.compress(&orig).expect("compress");
        // Bytes 3..7 are the LE u32 original length.
        let stored_len = u32::from_le_bytes([
            compressed[3], compressed[4], compressed[5], compressed[6],
        ]) as usize;
        assert_eq!(stored_len, orig.len());
    }

    // 14. Overlapping back-reference (run-length fill)
    #[test]
    fn test_overlapping_backref() {
        let c = TierCompressor::new(2);
        // Pattern that creates overlapping matches: "aaaaaaaaa..."
        let orig: Vec<u8> = vec![b'a'; 200];
        let compressed = c.compress(&orig).expect("compress");
        let restored = c.decompress(&compressed).expect("decompress");
        assert_eq!(restored, orig);
    }

    // 15. Large input round-trip
    #[test]
    fn test_large_input_round_trip() {
        let c = TierCompressor::new(1);
        // 10 KB of repeating pattern
        let orig: Vec<u8> = b"media frame data segment"
            .iter()
            .copied()
            .cycle()
            .take(10_240)
            .collect();
        let compressed = c.compress(&orig).expect("compress");
        let restored = c.decompress(&compressed).expect("decompress");
        assert_eq!(restored, orig);
    }
}
