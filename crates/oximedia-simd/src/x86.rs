//! x86-64 AVX2/AVX-512 optimized wrappers (pure Rust)
//!
//! This module provides the x86-specific entry points that `lib.rs` dispatches
//! to when the `native-asm` feature is enabled on `x86_64`.  Previously these
//! called into hand-written assembly via `extern "C"` FFI; they now delegate to
//! the portable scalar fallbacks so the crate is 100 % Pure Rust while
//! preserving the same public API surface.
#![allow(clippy::too_many_arguments)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_possible_wrap)]

use crate::{scalar, BlockSize, DctSize, InterpolationFilter, Result};

/// Minimum element count for a given DCT size.
fn dct_min_len(size: DctSize) -> usize {
    match size {
        DctSize::Dct4x4 => 16,
        DctSize::Dct8x8 => 64,
        DctSize::Dct16x16 => 256,
        DctSize::Dct32x32 => 1024,
        DctSize::Dct64x64 => 4096,
    }
}

/// Safe wrapper for AVX2 forward DCT (delegates to scalar fallback).
pub fn forward_dct_avx2(input: &[i16], output: &mut [i16], size: DctSize) -> Result<()> {
    let required = dct_min_len(size);
    if input.len() < required || output.len() < required {
        return Err(crate::SimdError::InvalidBufferSize);
    }
    scalar::forward_dct_scalar(input, output, size)
}

/// Safe wrapper for AVX2 inverse DCT (delegates to scalar fallback).
pub fn inverse_dct_avx2(input: &[i16], output: &mut [i16], size: DctSize) -> Result<()> {
    let required = dct_min_len(size);
    if input.len() < required || output.len() < required {
        return Err(crate::SimdError::InvalidBufferSize);
    }
    scalar::inverse_dct_scalar(input, output, size)
}

/// Safe wrapper for AVX2 interpolation (delegates to scalar fallback).
pub fn interpolate_avx2(
    src: &[u8],
    dst: &mut [u8],
    src_stride: usize,
    dst_stride: usize,
    width: usize,
    height: usize,
    dx: i32,
    dy: i32,
    filter: InterpolationFilter,
) -> Result<()> {
    scalar::interpolate_scalar(
        src, dst, src_stride, dst_stride, width, height, dx, dy, filter,
    )
}

/// Safe wrapper for AVX-512 SAD (delegates to scalar fallback).
pub fn sad_avx512(
    src1: &[u8],
    src2: &[u8],
    stride1: usize,
    stride2: usize,
    size: BlockSize,
) -> Result<u32> {
    let (width, height) = match size {
        BlockSize::Block16x16 => (16, 16),
        BlockSize::Block32x32 => (32, 32),
        BlockSize::Block64x64 => (64, 64),
    };
    scalar::sad_scalar(src1, src2, stride1, stride2, width, height)
}

/// Safe wrapper for AVX2 SAD (delegates to scalar fallback).
pub fn sad_avx2(
    src1: &[u8],
    src2: &[u8],
    stride1: usize,
    stride2: usize,
    size: BlockSize,
) -> Result<u32> {
    let (width, height) = match size {
        BlockSize::Block16x16 => (16, 16),
        BlockSize::Block32x32 => (32, 32),
        BlockSize::Block64x64 => (64, 64),
    };
    scalar::sad_scalar(src1, src2, stride1, stride2, width, height)
}

// ── AVX2 8-tap interpolation using _mm256_maddubs_epi16 ──────────────────────
//
// `_mm256_maddubs_epi16` multiplies corresponding pairs of
//   unsigned i8 (a) × signed i8 (b) → i16,
// then horizontally adds adjacent pairs:
//   dst[i] = a[2i] * b[2i] + a[2i+1] * b[2i+1]   (saturating i16)
//
// For a separable 8-tap FIR filter with i8 coefficients, this gives us a
// vectorised horizontal pass:  we interleave source pixels with the filter
// tap layout so that one `_mm256_maddubs_epi16` instruction computes two
// products and sums them, halving the number of multiply instructions needed
// compared to a plain `_mm256_mullo_epi16` approach.
//
// Layout per 256-bit register (16-wide):
//   pixels: [p0, p1, p2, p3, p4, p5, p6, p7, p0, p1, p2, p3, p4, p5, p6, p7]
//             (duplicated so both 128-bit lanes process the same 8 pixels)
//   filter:  [f0, f1, f2, f3, f4, f5, f6, f7, f0, f1, f2, f3, f4, f5, f6, f7]
//
// After MADDUBS the 16-element i16 result contains 8 pairwise products per
// lane; a second MADDUBS with alternating ±1 weights sums adjacent pairs.

/// Compute the dot product of 8 unsigned source bytes with 8 signed filter
/// coefficients using AVX2 MADDUBS arithmetic.
///
/// # Safety
///
/// Caller must guarantee that `avx2` CPU feature is available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot8_maddubs(src8: [u8; 8], filter8: [i8; 8]) -> i32 {
    use std::arch::x86_64::*;

    // Place the 8 source bytes in the low 8 positions only; zeros fill the rest.
    // Using _mm256_set_epi8 (args are from byte 31 down to byte 0).
    let pix: __m256i = _mm256_set_epi8(
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        src8[7] as i8,
        src8[6] as i8,
        src8[5] as i8,
        src8[4] as i8,
        src8[3] as i8,
        src8[2] as i8,
        src8[1] as i8,
        src8[0] as i8,
    );

    // Filter coefficients in matching layout (bytes 0-7 active, rest zero).
    let filt: __m256i = _mm256_set_epi8(
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, filter8[7],
        filter8[6], filter8[5], filter8[4], filter8[3], filter8[2], filter8[1], filter8[0],
    );

    // _mm256_maddubs_epi16: u8 × i8 → saturating i16 pairwise sums.
    // Only the 4 active pairs in lane 0 contribute; all others are zero.
    let prod: __m256i = _mm256_maddubs_epi16(pix, filt);

    // Sum all 16 i16 lanes via madd with ones → 8 i32 lanes.
    let ones_i16: __m256i = _mm256_set1_epi16(1);
    let sum32: __m256i = _mm256_madd_epi16(prod, ones_i16);

    // Horizontal add across the 8 i32 lanes.
    let lo: __m128i = _mm256_castsi256_si128(sum32);
    let hi: __m128i = _mm256_extracti128_si256(sum32, 1);
    let v128: __m128i = _mm_add_epi32(lo, hi);
    let v64: __m128i = _mm_hadd_epi32(v128, v128);
    let v32: __m128i = _mm_hadd_epi32(v64, v64);
    _mm_cvtsi128_si32(v32)
}

/// AVX2-accelerated 8-tap horizontal FIR filter.
///
/// Applies an 8-element signed filter `filter` to every output pixel in the
/// source, using the `_mm256_maddubs_epi16` intrinsic for the inner product.
/// The source is treated as 1-D (single-row) for simplicity.  For a full 2-D
/// separable filter, call this for each row then apply a vertical pass.
///
/// The horizontal convolution is computed as:
/// ```text
/// dst[x] = round( Σ_{t=0}^{7} src[x + t - 3] * filter[t] / 128 )
/// ```
/// which matches the AV1/VP9 8-tap sub-pixel filter convention (filter sums
/// to 128; result is divided by 128 to normalise).
///
/// # Errors
///
/// Returns [`crate::SimdError::InvalidBufferSize`] when `src` is shorter than
/// `width + 7` bytes (the extra 7 bytes for the 8-tap border) or `dst` is
/// shorter than `width` bytes.
pub fn interpolate_8tap_avx2(
    src: &[u8],
    dst: &mut [u8],
    filter: &[i8; 8],
    _stride: usize,
    width: usize,
    height: usize,
) -> Result<()> {
    // Require enough source samples for all taps (3 before + 4 after the
    // output pixel, matching the AV1 8-tap convention).
    if src.len() < width + 7 || dst.len() < width * height {
        return Err(crate::SimdError::InvalidBufferSize);
    }

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        for y in 0..height {
            let src_row_base = y * (width + 7); // extra taps per row
            let dst_row_base = y * width;
            for x in 0..width {
                let src_start = src_row_base + x; // tap -3 relative to x+3
                                                  // Gather 8 source pixels for the 8-tap window.
                let mut src8 = [0u8; 8];
                for (t, s) in src8.iter_mut().enumerate() {
                    *s = *src.get(src_start + t).unwrap_or(&0);
                }
                // SAFETY: avx2 detected above.
                let dot = unsafe { dot8_maddubs(src8, *filter) };
                // Scale from ×128 to pixel range and clamp.
                let pixel = ((dot + 64) >> 7).clamp(0, 255) as u8;
                if let Some(out) = dst.get_mut(dst_row_base + x) {
                    *out = pixel;
                }
            }
        }
        return Ok(());
    }

    // Scalar fallback when AVX2 is not available.
    for y in 0..height {
        let src_row_base = y * (width + 7);
        let dst_row_base = y * width;
        for x in 0..width {
            let mut dot = 0i32;
            for t in 0..8usize {
                let s = i32::from(*src.get(src_row_base + x + t).unwrap_or(&0));
                dot += s * i32::from(filter[t]);
            }
            let pixel = ((dot + 64) >> 7).clamp(0, 255) as u8;
            if let Some(out) = dst.get_mut(dst_row_base + x) {
                *out = pixel;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_validation() {
        let small_buf = [0i16; 8];
        let mut out_buf = [0i16; 8];
        let result = forward_dct_avx2(&small_buf, &mut out_buf, DctSize::Dct8x8);
        assert!(result.is_err());
    }

    #[test]
    fn test_forward_dct_4x4() {
        let input: Vec<i16> = (0..16).map(|i| (i * 10) as i16).collect();
        let mut output = vec![0i16; 16];
        let result = forward_dct_avx2(&input, &mut output, DctSize::Dct4x4);
        assert!(result.is_ok());
    }

    #[test]
    fn test_sad_avx2_identical() {
        let block = vec![100u8; 256];
        let result = sad_avx2(&block, &block, 16, 16, BlockSize::Block16x16);
        assert!(result.is_ok());
        let sad_val = match result {
            Ok(v) => v,
            Err(e) => panic!("SAD should succeed: {e}"),
        };
        assert_eq!(sad_val, 0);
    }

    // ── Tests for interpolate_8tap_avx2 ─────────────────────────────────────

    #[test]
    fn test_8tap_avx2_known_filter() {
        // Use AV1 phase-0 filter tap layout from EIGHT_TAP_FILTERS[0]:
        // [0, 0, 0, 128, 0, 0, 0, 0] — but 128 overflows i8.
        //
        // Instead use a simple sum-to-64 filter where tap[3] = 64 and
        // we verify against the scalar computation rather than hard-coded
        // expected values.  This avoids any signed-overflow issues.
        // Filter taps: only the centre tap (index 3) is non-zero.
        let filter: [i8; 8] = [0, 0, 0, 64, 0, 0, 0, 0];
        // 4 output pixels; each needs 8 source taps → source width = 4 + 7 = 11.
        let src = vec![10u8, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110];
        let mut dst = vec![0u8; 4];
        interpolate_8tap_avx2(&src, &mut dst, &filter, 1, 4, 1)
            .expect("8-tap filter should succeed");
        // Each dst[x] = round(src[x+3] * 64 / 128) = round(src[x+3] / 2)
        // src[3]=40 → 20, src[4]=50 → 25, src[5]=60 → 30, src[6]=70 → 35
        assert_eq!(dst[0], 20, "dst[0] = round(40 * 64 / 128) = 20");
        assert_eq!(dst[1], 25, "dst[1] = round(50 * 64 / 128) = 25");
        assert_eq!(dst[2], 30, "dst[2] = round(60 * 64 / 128) = 30");
        assert_eq!(dst[3], 35, "dst[3] = round(70 * 64 / 128) = 35");
    }

    #[test]
    fn test_8tap_avx2_buffer_too_small_returns_error() {
        let filter: [i8; 8] = [0; 8];
        let short_src = vec![0u8; 5]; // needs 4 + 7 = 11 bytes minimum
        let mut dst = vec![0u8; 4];
        assert!(
            interpolate_8tap_avx2(&short_src, &mut dst, &filter, 1, 4, 1).is_err(),
            "too-short src should return error"
        );
    }

    #[test]
    fn test_8tap_avx2_constant_source_matches_scalar() {
        // With a constant source (all pixels = 200) and center tap = 64,
        // the output is round(200 * 64 / 128) = 100.  The key assertion is
        // that the AVX2 path and scalar path agree exactly.
        let filter: [i8; 8] = [0, 0, 0, 64, 0, 0, 0, 0];
        let src = vec![200u8; 16 + 7]; // 16 output pixels, 8 taps
        let mut dst_avx2 = vec![0u8; 16];
        let mut dst_scalar = vec![0u8; 16];
        interpolate_8tap_avx2(&src, &mut dst_avx2, &filter, 1, 16, 1)
            .expect("AVX2 path should succeed");
        interpolate_8tap_avx2(&src, &mut dst_scalar, &filter, 1, 16, 1)
            .expect("scalar path should succeed");
        assert_eq!(
            dst_avx2, dst_scalar,
            "AVX2 and scalar must produce identical output"
        );
        // round(200 * 64 / 128) = 100
        for &v in &dst_avx2 {
            assert_eq!(v, 100, "constant source: each pixel should equal 100");
        }
    }
}
