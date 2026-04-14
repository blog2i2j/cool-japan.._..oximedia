//! Hadamard–Walsh Transform (WHT) public API.
//!
//! This module exposes a clean, safe Rust API for forward and inverse
//! Hadamard–Walsh transforms (WHT) at multiple block sizes.  The WHT is a key
//! primitive in video codec rate-distortion optimisation (used by SATD, for
//! example) and in signal-processing pipelines that require a fast orthogonal
//! transform that avoids floating-point arithmetic entirely.
//!
//! # Transform properties
//!
//! - The WHT is self-inverse up to a scaling factor: `WHT(WHT(x)) = N · x`
//!   where `N` is the transform length.
//! - For 2-D blocks of size `N×N`: `WHT2D(WHT2D(x)) = N² · x`.
//! - All arithmetic is performed in `i32`, which is sufficient for 8-bit pixel
//!   residuals up to 128×128 blocks without overflow.
//!
//! # Supported block sizes
//!
//! | Variant         | Side | Pixels |
//! |-----------------|------|--------|
//! | [`WhtSize::N2`] | 2    | 4      |
//! | [`WhtSize::N4`] | 4    | 16     |
//! | [`WhtSize::N8`] | 8    | 64     |
//! | [`WhtSize::N16`]| 16   | 256    |
//! | [`WhtSize::N32`]| 32   | 1024   |
//!
//! # Examples
//!
//! ```
//! use oximedia_simd::hadamard::{wht_forward_2d, wht_inverse_2d, WhtSize};
//!
//! let original: Vec<i32> = (0..64).collect();
//! let mut buf = original.clone();
//!
//! // Forward transform
//! wht_forward_2d(&mut buf, WhtSize::N8).expect("forward WHT should not fail");
//!
//! // Inverse transform (WHT is self-inverse up to N² scaling)
//! wht_inverse_2d(&mut buf, WhtSize::N8).expect("inverse WHT should not fail");
//!
//! // Recover originals: each value should equal original * N² = original * 64
//! for (i, (&orig, &result)) in original.iter().zip(buf.iter()).enumerate() {
//!     assert_eq!(result, orig * 64, "element {i}");
//! }
//! ```

#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_sign_loss)]

use crate::{Result, SimdError};

// ── Block-size enum ────────────────────────────────────────────────────────────

/// Supported WHT block sizes.
///
/// The side length `N` of the square block.  All values are powers of two.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WhtSize {
    /// 2×2 block (4 coefficients).
    N2,
    /// 4×4 block (16 coefficients).
    N4,
    /// 8×8 block (64 coefficients).
    N8,
    /// 16×16 block (256 coefficients).
    N16,
    /// 32×32 block (1024 coefficients).
    N32,
}

impl WhtSize {
    /// Side length of the square block.
    #[must_use]
    pub fn side(self) -> usize {
        match self {
            Self::N2 => 2,
            Self::N4 => 4,
            Self::N8 => 8,
            Self::N16 => 16,
            Self::N32 => 32,
        }
    }

    /// Total number of coefficients (side²).
    #[must_use]
    pub fn len(self) -> usize {
        let s = self.side();
        s * s
    }

    /// Always returns `false`; a zero-size block is not representable.
    #[must_use]
    pub fn is_empty(self) -> bool {
        false
    }

    /// Scaling divisor for the normalised inverse: `N²`.
    ///
    /// After `wht_inverse_2d` the values are `N²` times the originals.
    /// Divide by this factor to recover the exact input.
    #[must_use]
    pub fn scale_factor(self) -> i64 {
        let s = self.side() as i64;
        s * s
    }
}

// ── Core 1-D WHT butterfly ─────────────────────────────────────────────────────

/// In-place 1-D Hadamard–Walsh transform on a power-of-two length slice.
///
/// Uses the iterative Cooley–Tukey butterfly decomposition (stride-doubling
/// form).  After this call `work[i]` holds the un-normalised WHT coefficient at
/// frequency `i`.
///
/// Panics in debug builds if `work.len()` is not a power of two.
fn wht_1d_inplace(work: &mut [i32]) {
    let n = work.len();
    debug_assert!(n.is_power_of_two(), "WHT length must be a power of two");

    let mut step = 1usize;
    while step < n {
        let mut i = 0;
        while i < n {
            for j in i..i + step {
                let a = work[j];
                let b = work[j + step];
                work[j] = a + b;
                work[j + step] = a - b;
            }
            i += 2 * step;
        }
        step *= 2;
    }
}

// ── 2-D separable WHT ─────────────────────────────────────────────────────────

/// Apply the 2-D separable WHT to `block` in-place.
///
/// First transforms every row, then every column.  The block is stored in
/// row-major order and must have exactly `n * n` elements where `n = size.side()`.
///
/// Returns [`SimdError::InvalidBufferSize`] if `block.len() < size.len()`.
///
/// # Errors
///
/// Returns [`SimdError::InvalidBufferSize`] if the buffer is too small.
pub fn wht_2d_inplace(block: &mut [i32], size: WhtSize) -> Result<()> {
    let n = size.side();
    let required = size.len();
    if block.len() < required {
        return Err(SimdError::InvalidBufferSize);
    }

    // Row transforms
    let mut row_buf = vec![0i32; n];
    for r in 0..n {
        row_buf.copy_from_slice(&block[r * n..(r + 1) * n]);
        wht_1d_inplace(&mut row_buf);
        block[r * n..(r + 1) * n].copy_from_slice(&row_buf);
    }

    // Column transforms
    let mut col_buf = vec![0i32; n];
    for c in 0..n {
        for r in 0..n {
            col_buf[r] = block[r * n + c];
        }
        wht_1d_inplace(&mut col_buf);
        for r in 0..n {
            block[r * n + c] = col_buf[r];
        }
    }

    Ok(())
}

// ── Public forward / inverse entry points ─────────────────────────────────────

/// Perform a 2-D forward Hadamard–Walsh transform.
///
/// The un-normalised WHT coefficients are stored back in `block` in row-major
/// order.  To recover the original values, apply [`wht_inverse_2d`] and then
/// divide each element by `size.scale_factor()`.
///
/// # Errors
///
/// Returns [`SimdError::InvalidBufferSize`] if `block` has fewer than
/// `size.len()` elements.
pub fn wht_forward_2d(block: &mut [i32], size: WhtSize) -> Result<()> {
    wht_2d_inplace(block, size)
}

/// Perform a 2-D inverse Hadamard–Walsh transform.
///
/// Because the WHT is self-inverse up to scaling, this function is identical
/// to [`wht_forward_2d`].  The caller is responsible for dividing the output
/// by `size.scale_factor()` to recover the original values.
///
/// # Errors
///
/// Returns [`SimdError::InvalidBufferSize`] if `block` has fewer than
/// `size.len()` elements.
pub fn wht_inverse_2d(block: &mut [i32], size: WhtSize) -> Result<()> {
    wht_2d_inplace(block, size)
}

/// Compute the Hadamard-domain Sum of Absolute Differences between two blocks.
///
/// `SATD = Σ |WHT(src - ref_)| / normalization`.
///
/// Both slices must be packed row-major (stride == `size.side()`) and have at
/// least `size.len()` elements.
///
/// The returned value is the un-normalised SATD (consistent with x264/x265
/// conventions).
///
/// # Errors
///
/// Returns [`SimdError::InvalidBufferSize`] if either slice is too small.
pub fn wht_satd(src: &[u8], ref_: &[u8], size: WhtSize) -> Result<u32> {
    let n = size.len();
    if src.len() < n || ref_.len() < n {
        return Err(SimdError::InvalidBufferSize);
    }

    let mut diff: Vec<i32> = src[..n]
        .iter()
        .zip(ref_[..n].iter())
        .map(|(&s, &r)| i32::from(s) - i32::from(r))
        .collect();

    wht_2d_inplace(&mut diff, size)?;

    let satd: u32 = diff.iter().map(|&v| v.unsigned_abs()).sum();
    Ok(satd)
}

/// Compute the WHT power spectrum of a block.
///
/// Returns a `Vec<u64>` of squared WHT coefficient magnitudes (|c|²), useful
/// for frequency-domain analysis and perceptual quality metrics.
///
/// `block` must have at least `size.len()` `i32` elements (e.g. residuals or
/// luma values cast to `i32`).
///
/// # Errors
///
/// Returns [`SimdError::InvalidBufferSize`] if `block` is too small.
pub fn wht_power_spectrum(block: &[i32], size: WhtSize) -> Result<Vec<u64>> {
    let n = size.len();
    if block.len() < n {
        return Err(SimdError::InvalidBufferSize);
    }

    let mut buf = block[..n].to_vec();
    wht_2d_inplace(&mut buf, size)?;

    let spectrum: Vec<u64> = buf
        .iter()
        .map(|&c| {
            let c64 = i64::from(c);
            (c64 * c64) as u64
        })
        .collect();

    Ok(spectrum)
}

/// Normalise a WHT output buffer by dividing every element by `scale`.
///
/// This is a convenience helper for the common post-inverse-WHT step.  Any
/// remainder from integer division is truncated toward zero (consistent with
/// standard codec practice).
///
/// Returns [`SimdError::InvalidBufferSize`] if `buf` is empty and `scale > 0`.
///
/// # Errors
///
/// Returns [`SimdError::UnsupportedOperation`] if `scale` is zero (would be
/// integer division by zero).
pub fn wht_normalize(buf: &mut [i32], scale: i64) -> Result<()> {
    if scale == 0 {
        return Err(SimdError::UnsupportedOperation);
    }
    let scale_i32 = scale as i32;
    for v in buf.iter_mut() {
        *v /= scale_i32;
    }
    Ok(())
}

/// Compute the 1-D WHT of a slice and return the result in a new `Vec<i32>`.
///
/// `input` must have a power-of-two length between 2 and 4096.
///
/// # Errors
///
/// Returns [`SimdError::InvalidBufferSize`] if `input` is empty or its length
/// is not a power of two.
pub fn wht_1d(input: &[i32]) -> Result<Vec<i32>> {
    let n = input.len();
    if n == 0 || !n.is_power_of_two() {
        return Err(SimdError::InvalidBufferSize);
    }
    let mut buf = input.to_vec();
    wht_1d_inplace(&mut buf);
    Ok(buf)
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── 1-D WHT ───────────────────────────────────────────────────────────────

    #[test]
    fn wht_1d_length4_known_values() {
        // WHT([1,2,3,4]) = [10, -2, -4, 0] via butterfly
        let result = wht_1d(&[1, 2, 3, 4]).expect("wht_1d should succeed");
        assert_eq!(result, vec![10, -2, -4, 0]);
    }

    #[test]
    fn wht_1d_empty_returns_error() {
        assert_eq!(wht_1d(&[]), Err(SimdError::InvalidBufferSize));
    }

    #[test]
    fn wht_1d_non_power_of_two_returns_error() {
        // Length 3 is not a power of two
        assert_eq!(wht_1d(&[1, 2, 3]), Err(SimdError::InvalidBufferSize));
    }

    #[test]
    fn wht_1d_double_application_scales_by_n() {
        // WHT(WHT(x)) = N * x
        let input = vec![1i32, 2, 3, 4, 5, 6, 7, 8];
        let n = input.len() as i32;
        let once = wht_1d(&input).expect("first WHT");
        let twice = wht_1d(&once).expect("second WHT");
        for (i, (&orig, &result)) in input.iter().zip(twice.iter()).enumerate() {
            assert_eq!(result, orig * n, "mismatch at index {i}");
        }
    }

    // ── 2-D WHT ───────────────────────────────────────────────────────────────

    #[test]
    fn wht_2d_zero_block_stays_zero() {
        let mut buf = vec![0i32; 64];
        wht_forward_2d(&mut buf, WhtSize::N8).expect("forward WHT");
        assert!(
            buf.iter().all(|&v| v == 0),
            "zero block WHT must be all-zero"
        );
    }

    #[test]
    fn wht_2d_roundtrip_n4() {
        let original: Vec<i32> = (0..16).map(|i| i * 5).collect();
        let mut buf = original.clone();
        wht_forward_2d(&mut buf, WhtSize::N4).expect("forward");
        wht_inverse_2d(&mut buf, WhtSize::N4).expect("inverse");
        // After two WHTs, each element is scaled by N² = 16
        let scale = WhtSize::N4.scale_factor();
        for (i, (&orig, &result)) in original.iter().zip(buf.iter()).enumerate() {
            assert_eq!(result, orig * scale as i32, "mismatch at index {i}");
        }
    }

    #[test]
    fn wht_2d_roundtrip_n8() {
        let original: Vec<i32> = (0..64).map(|i| i * 3 - 50).collect();
        let mut buf = original.clone();
        wht_forward_2d(&mut buf, WhtSize::N8).expect("forward");
        wht_inverse_2d(&mut buf, WhtSize::N8).expect("inverse");
        let scale = WhtSize::N8.scale_factor() as i32; // 64
        for (i, (&orig, &result)) in original.iter().zip(buf.iter()).enumerate() {
            assert_eq!(result, orig * scale, "N8 roundtrip mismatch at {i}");
        }
    }

    #[test]
    fn wht_2d_buffer_too_small_returns_error() {
        let mut buf = vec![0i32; 10]; // N8 needs 64
        let err = wht_forward_2d(&mut buf, WhtSize::N8);
        assert_eq!(err, Err(SimdError::InvalidBufferSize));
    }

    // ── SATD via WHT ─────────────────────────────────────────────────────────

    #[test]
    fn wht_satd_identical_blocks_is_zero() {
        let block = vec![128u8; 64];
        let satd = wht_satd(&block, &block, WhtSize::N8).expect("satd");
        assert_eq!(satd, 0, "SATD of identical blocks must be zero");
    }

    #[test]
    fn wht_satd_nonzero_for_different_blocks() {
        let a = vec![100u8; 64];
        let b = vec![200u8; 64];
        let satd = wht_satd(&a, &b, WhtSize::N8).expect("satd");
        assert!(satd > 0, "SATD of different blocks must be non-zero");
    }

    #[test]
    fn wht_satd_is_symmetric() {
        let a: Vec<u8> = (0..64).map(|i| (i * 3 % 256) as u8).collect();
        let b: Vec<u8> = (0..64).map(|i| (i * 2 + 10 % 256) as u8).collect();
        let ab = wht_satd(&a, &b, WhtSize::N8).expect("ab");
        let ba = wht_satd(&b, &a, WhtSize::N8).expect("ba");
        assert_eq!(ab, ba, "SATD must be symmetric");
    }

    #[test]
    fn wht_satd_4x4_block() {
        let a = vec![10u8; 16];
        let b = vec![20u8; 16];
        let satd = wht_satd(&a, &b, WhtSize::N4).expect("satd_4x4");
        assert!(satd > 0);
    }

    #[test]
    fn wht_satd_buffer_too_small_returns_error() {
        let a = vec![0u8; 10];
        let b = vec![0u8; 10];
        assert_eq!(
            wht_satd(&a, &b, WhtSize::N8),
            Err(SimdError::InvalidBufferSize)
        );
    }

    // ── Power spectrum ────────────────────────────────────────────────────────

    #[test]
    fn wht_power_spectrum_zero_block_is_all_zeros() {
        let buf = vec![0i32; 64];
        let spec = wht_power_spectrum(&buf, WhtSize::N8).expect("spectrum");
        assert!(spec.iter().all(|&v| v == 0));
    }

    #[test]
    fn wht_power_spectrum_length_matches_block_size() {
        let buf = vec![1i32; 16];
        let spec = wht_power_spectrum(&buf, WhtSize::N4).expect("spectrum");
        assert_eq!(spec.len(), WhtSize::N4.len());
    }

    // ── Normalise ─────────────────────────────────────────────────────────────

    #[test]
    fn wht_normalize_divides_by_scale() {
        let mut buf = vec![64i32; 4];
        wht_normalize(&mut buf, 64).expect("normalize");
        assert!(buf.iter().all(|&v| v == 1));
    }

    #[test]
    fn wht_normalize_zero_scale_returns_error() {
        let mut buf = vec![1i32; 4];
        assert_eq!(
            wht_normalize(&mut buf, 0),
            Err(SimdError::UnsupportedOperation)
        );
    }

    // ── WhtSize helpers ───────────────────────────────────────────────────────

    #[test]
    fn wht_size_properties() {
        assert_eq!(WhtSize::N2.side(), 2);
        assert_eq!(WhtSize::N4.side(), 4);
        assert_eq!(WhtSize::N8.side(), 8);
        assert_eq!(WhtSize::N16.side(), 16);
        assert_eq!(WhtSize::N32.side(), 32);

        assert_eq!(WhtSize::N4.len(), 16);
        assert_eq!(WhtSize::N8.len(), 64);

        assert_eq!(WhtSize::N4.scale_factor(), 16);
        assert_eq!(WhtSize::N8.scale_factor(), 64);

        assert!(!WhtSize::N8.is_empty());
    }

    #[test]
    fn wht_roundtrip_then_normalize_recovers_original() {
        let original: Vec<i32> = (0..16).map(|i| i - 8).collect();
        let mut buf = original.clone();
        wht_forward_2d(&mut buf, WhtSize::N4).expect("forward");
        wht_inverse_2d(&mut buf, WhtSize::N4).expect("inverse");
        wht_normalize(&mut buf, WhtSize::N4.scale_factor()).expect("normalize");
        for (i, (&orig, &result)) in original.iter().zip(buf.iter()).enumerate() {
            assert_eq!(result, orig, "full roundtrip mismatch at {i}");
        }
    }

    // ── Additional tests for Hadamard transform (scalar + platform backends) ──

    #[test]
    fn wht_1d_single_element_identity() {
        // WHT of a single element [v] = [v]
        let result = wht_1d(&[42]).expect("wht_1d single element");
        assert_eq!(result, vec![42]);
    }

    #[test]
    fn wht_1d_two_elements_butterfly() {
        // WHT([a, b]) = [a+b, a-b]
        let result = wht_1d(&[3, 7]).expect("wht_1d 2 elements");
        assert_eq!(result, vec![10, -4]);
    }

    #[test]
    fn wht_1d_all_zeros_stays_zero() {
        let result = wht_1d(&[0i32; 16]).expect("wht_1d zeros");
        assert!(
            result.iter().all(|&v| v == 0),
            "WHT of zeros must be all-zero"
        );
    }

    #[test]
    fn wht_1d_impulse_response_is_all_ones() {
        // WHT of delta function [1, 0, 0, ..., 0] = [1, 1, 1, ..., 1]
        let mut impulse = vec![0i32; 8];
        impulse[0] = 1;
        let result = wht_1d(&impulse).expect("wht_1d impulse");
        assert!(
            result.iter().all(|&v| v == 1),
            "WHT of impulse must be all-ones"
        );
    }

    #[test]
    fn wht_2d_roundtrip_n2() {
        let original = vec![1i32, 2, 3, 4];
        let mut buf = original.clone();
        wht_forward_2d(&mut buf, WhtSize::N2).expect("forward N2");
        wht_inverse_2d(&mut buf, WhtSize::N2).expect("inverse N2");
        let scale = WhtSize::N2.scale_factor() as i32; // 4
        for (i, (&orig, &result)) in original.iter().zip(buf.iter()).enumerate() {
            assert_eq!(result, orig * scale, "N2 roundtrip mismatch at {i}");
        }
    }

    #[test]
    fn wht_2d_roundtrip_n16() {
        let original: Vec<i32> = (0..256).map(|i| (i as i32) - 128).collect();
        let mut buf = original.clone();
        wht_forward_2d(&mut buf, WhtSize::N16).expect("forward N16");
        wht_inverse_2d(&mut buf, WhtSize::N16).expect("inverse N16");
        let scale = WhtSize::N16.scale_factor() as i32; // 256
        for (i, (&orig, &result)) in original.iter().zip(buf.iter()).enumerate() {
            assert_eq!(result, orig * scale, "N16 roundtrip mismatch at {i}");
        }
    }

    #[test]
    fn wht_2d_roundtrip_n32() {
        let original: Vec<i32> = (0..1024).map(|i| (i as i32) % 64 - 32).collect();
        let mut buf = original.clone();
        wht_forward_2d(&mut buf, WhtSize::N32).expect("forward N32");
        wht_inverse_2d(&mut buf, WhtSize::N32).expect("inverse N32");
        let scale = WhtSize::N32.scale_factor() as i32; // 1024
        for (i, (&orig, &result)) in original.iter().zip(buf.iter()).enumerate() {
            assert_eq!(result, orig * scale, "N32 roundtrip mismatch at {i}");
        }
    }

    #[test]
    fn wht_satd_scales_with_difference_magnitude() {
        // Larger difference → larger SATD
        let flat = vec![128u8; 64];
        let small_diff = vec![130u8; 64];
        let large_diff = vec![150u8; 64];
        let satd_small = wht_satd(&flat, &small_diff, WhtSize::N8).expect("small");
        let satd_large = wht_satd(&flat, &large_diff, WhtSize::N8).expect("large");
        assert!(
            satd_large > satd_small,
            "larger diff ({large_diff:?}) must produce larger SATD: {satd_large} vs {satd_small}"
        );
    }

    #[test]
    fn wht_satd_4x4_known_constant_blocks() {
        // Block A = all 100, Block B = all 120 → diff = -20 everywhere
        // WHT of constant [-20; 16] = [-20*16, 0, 0, ...] = only DC component
        // SATD = |WHT[-20 * 16]| = 320
        let a = vec![100u8; 16];
        let b = vec![120u8; 16];
        let satd = wht_satd(&a, &b, WhtSize::N4).expect("satd");
        // The DC coefficient of WHT of constant block is N² × value
        // For 4×4: DC = 16 × 20 = 320
        assert_eq!(
            satd, 320,
            "SATD of constant-diff 4×4 block should be 320, got {satd}"
        );
    }

    #[test]
    fn wht_power_spectrum_nonzero_for_nonconstant_block() {
        let buf: Vec<i32> = (0..64).map(|i| i as i32).collect();
        let spec = wht_power_spectrum(&buf, WhtSize::N8).expect("spectrum");
        let total_power: u64 = spec.iter().sum();
        assert!(
            total_power > 0,
            "non-constant block must have non-zero power"
        );
    }

    #[test]
    fn wht_normalize_negative_scale() {
        // Dividing by -1 should negate all elements
        let mut buf = vec![1i32, -2, 3, -4];
        wht_normalize(&mut buf, -1).expect("normalize by -1");
        assert_eq!(buf, vec![-1, 2, -3, 4]);
    }

    #[test]
    fn wht_2d_constant_block_only_dc_nonzero() {
        // A constant block's WHT should have all-zero except DC
        let mut buf = vec![5i32; 64];
        wht_2d_inplace(&mut buf, WhtSize::N8).expect("WHT");
        // DC coefficient = sum of all 64 values = 64 * 5 = 320
        assert_eq!(buf[0], 320, "DC coefficient should be 320, got {}", buf[0]);
        // All AC coefficients should be zero for a constant block
        for (i, &v) in buf[1..].iter().enumerate() {
            assert_eq!(
                v, 0,
                "AC coefficient {i} should be zero for constant block, got {v}"
            );
        }
    }
}
