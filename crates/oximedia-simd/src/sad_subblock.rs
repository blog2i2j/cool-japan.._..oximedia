//! Sub-block SAD (Sum of Absolute Differences) for fine-grained motion search.
//!
//! This module extends the core SAD infrastructure in `crate` with small block
//! sizes — `4×4` and `8×8` — that are essential for sub-block motion
//! estimation in modern codecs (AV1 partitions down to 4×4, H.264 uses 4×4 and
//! 8×8 sub-block modes, HEVC uses 4×4 as its minimum TU).
//!
//! # Why a separate module?
//!
//! The primary [`crate::sad()`] function uses [`crate::BlockSize`] which only
//! covers `16×16`, `32×32`, and `64×64`.  Adding smaller block sizes to that
//! enum would require non-trivial changes to the existing AVX2/NEON backends.
//! This module provides a self-contained, scalar-first implementation with
//! optional x86-64 SIMD acceleration via SSE4.1 `_mm_sad_epu8`, and NEON
//! `vabdq_u8` on aarch64.
//!
//! # Block sizes
//!
//! | Variant                  | Width | Height | Pixels |
//! |--------------------------|-------|--------|--------|
//! | [`SubBlockSize::B4x4`]   | 4     | 4      | 16     |
//! | [`SubBlockSize::B4x8`]   | 4     | 8      | 32     |
//! | [`SubBlockSize::B8x4`]   | 8     | 4      | 32     |
//! | [`SubBlockSize::B8x8`]   | 8     | 8      | 64     |
//!
//! # Usage
//!
//! ```
//! use oximedia_simd::sad_subblock::{sad_subblock, SubBlockSize};
//!
//! let src = vec![100u8; 64];
//! let ref_ = vec![100u8; 64];
//! let result = sad_subblock(&src, 8, &ref_, 8, SubBlockSize::B8x8)
//!     .expect("SAD should succeed");
//! assert_eq!(result, 0, "identical blocks give SAD = 0");
//! ```
//!
//! # SIMD dispatch
//!
//! On x86-64, the function checks for `sse4.1` at runtime and uses the
//! `_mm_sad_epu8` horizontal SAD reduction when available.  On aarch64 it uses
//! `vabdq_u8` + `vpaddlq_u8` + `vaddvq_u16` via NEON intrinsics.  On all
//! other platforms (or when the SIMD feature is absent) a plain scalar loop is
//! used — the result is identical across all paths.

#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_sign_loss)]

use crate::{Result, SimdError};

// ── Sub-block size enum ────────────────────────────────────────────────────────

/// Block dimensions supported by the sub-block SAD kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubBlockSize {
    /// 4×4 — smallest codec partition (AV1 min TU, H.264 sub-block).
    B4x4,
    /// 4×8 — rectangular partition.
    B4x8,
    /// 8×4 — rectangular partition.
    B8x4,
    /// 8×8 — common motion-search unit.
    B8x8,
}

impl SubBlockSize {
    /// Block width in pixels.
    #[must_use]
    pub fn width(self) -> usize {
        match self {
            Self::B4x4 | Self::B4x8 => 4,
            Self::B8x4 | Self::B8x8 => 8,
        }
    }

    /// Block height in pixels.
    #[must_use]
    pub fn height(self) -> usize {
        match self {
            Self::B4x4 | Self::B8x4 => 4,
            Self::B4x8 | Self::B8x8 => 8,
        }
    }

    /// Total number of pixels.
    #[must_use]
    pub fn pixels(self) -> usize {
        self.width() * self.height()
    }
}

// ── Scalar kernel ─────────────────────────────────────────────────────────────

/// Pure scalar SAD for a `width × height` strided region.
///
/// Both `src` and `ref_` are accessed row-by-row with their respective strides.
#[inline(always)]
fn sad_scalar_strided(
    src: &[u8],
    src_stride: usize,
    ref_: &[u8],
    ref_stride: usize,
    width: usize,
    height: usize,
) -> u32 {
    let mut acc = 0u32;
    for row in 0..height {
        for col in 0..width {
            let s = u32::from(src[row * src_stride + col]);
            let r = u32::from(ref_[row * ref_stride + col]);
            acc += s.abs_diff(r);
        }
    }
    acc
}

// ── x86-64 SSE4.1 path ────────────────────────────────────────────────────────

/// SSE4.1 SAD for a packed 8×8 block (both buffers row-major, stride == 8).
///
/// Uses `_mm_sad_epu8` which computes 8-way horizontal SAD accumulation in a
/// single 128-bit instruction, yielding roughly 8× the throughput of the scalar
/// loop.
///
/// # Safety
///
/// Caller must ensure SSE4.1 is available (`is_x86_feature_detected!("sse4.1")`).
/// Both `src` and `ref_` must have at least 64 elements.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn sad_8x8_sse41(src: &[u8; 64], ref_: &[u8; 64]) -> u32 {
    use std::arch::x86_64::*;

    let mut total = 0u32;
    // Process 2 rows at a time — each row is 8 bytes; we load 16 bytes (2 rows)
    // and use _mm_sad_epu8 to accumulate in 64-bit lanes.
    let mut i = 0usize;
    while i < 64 {
        // SAFETY: SSE4.1 confirmed by caller; array is 64 bytes so i+16 is in-bounds.
        let (lo, hi) = unsafe {
            let s = _mm_loadu_si128(src.as_ptr().add(i).cast::<__m128i>());
            let r = _mm_loadu_si128(ref_.as_ptr().add(i).cast::<__m128i>());
            let sad_vec = _mm_sad_epu8(s, r);
            let lo = _mm_cvtsi128_si32(sad_vec) as u32;
            let hi = _mm_extract_epi32(sad_vec, 2) as u32;
            (lo, hi)
        };
        // Extract the two 64-bit SAD accumulators
        total += lo + hi;
        i += 16;
    }
    total
}

/// SSE4.1 SAD for a packed 4×4 block.
///
/// # Safety
///
/// Caller must ensure SSE4.1 is available.
/// Both `src` and `ref_` must have at least 16 elements.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn sad_4x4_sse41(src: &[u8; 16], ref_: &[u8; 16]) -> u32 {
    use std::arch::x86_64::*;

    // SAFETY: SSE4.1 confirmed by caller; array bounds guarantee valid pointers.
    unsafe {
        let s = _mm_loadu_si128(src.as_ptr().cast::<__m128i>());
        let r = _mm_loadu_si128(ref_.as_ptr().cast::<__m128i>());
        let sad_vec = _mm_sad_epu8(s, r);
        let lo = _mm_cvtsi128_si32(sad_vec) as u32;
        let hi = _mm_extract_epi32(sad_vec, 2) as u32;
        lo + hi
    }
}

// ── aarch64 NEON path ─────────────────────────────────────────────────────────

/// NEON SAD for a packed 8×8 block using `vabdq_u8` + `vpaddlq_u8`.
///
/// # Safety
///
/// Caller must ensure NEON is available (always true on aarch64).
/// Both `src` and `ref_` must have at least 64 elements.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn sad_8x8_neon(src: &[u8; 64], ref_: &[u8; 64]) -> u32 {
    use std::arch::aarch64::*;

    // SAFETY: NEON enabled via target_feature; pointers are valid array refs.
    unsafe {
        let mut acc = vdupq_n_u32(0);
        let mut i = 0usize;
        while i < 64 {
            let s = vld1q_u8(src.as_ptr().add(i));
            let r = vld1q_u8(ref_.as_ptr().add(i));
            // Absolute difference (u8 → u8)
            let abd = vabdq_u8(s, r);
            // Pairwise-sum u8 → u16 then accumulate into u32 lane
            let sum16 = vpaddlq_u8(abd);
            let sum32 = vpaddlq_u16(sum16);
            acc = vaddq_u32(acc, sum32);
            i += 16;
        }
        vaddvq_u32(acc)
    }
}

// ── Validation helper ─────────────────────────────────────────────────────────

/// Validate that the source and reference buffers are large enough for the
/// requested block size and strides.
fn validate_buffers(
    src: &[u8],
    src_stride: usize,
    ref_: &[u8],
    ref_stride: usize,
    size: SubBlockSize,
) -> Result<()> {
    let w = size.width();
    let h = size.height();
    if src_stride < w || ref_stride < w {
        return Err(SimdError::InvalidBufferSize);
    }
    // Need at least h rows each with `stride` bytes accessible
    if src.len() < (h - 1) * src_stride + w {
        return Err(SimdError::InvalidBufferSize);
    }
    if ref_.len() < (h - 1) * ref_stride + w {
        return Err(SimdError::InvalidBufferSize);
    }
    Ok(())
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Compute the SAD between two pixel blocks with arbitrary strides.
///
/// Dispatches to the best available SIMD path at runtime:
///
/// - x86-64 with SSE4.1: uses `_mm_sad_epu8` for 8×8 and 4×4 packed blocks.
/// - aarch64 with NEON: uses `vabdq_u8` for 8×8 packed blocks.
/// - All other platforms: scalar loop.
///
/// For non-packed layouts (stride > width) or rectangular blocks (`B4x8`,
/// `B8x4`) the scalar path is always used.
///
/// # Errors
///
/// Returns [`SimdError::InvalidBufferSize`] if either buffer is too small to
/// hold the requested block with the given strides.
pub fn sad_subblock(
    src: &[u8],
    src_stride: usize,
    ref_: &[u8],
    ref_stride: usize,
    size: SubBlockSize,
) -> Result<u32> {
    validate_buffers(src, src_stride, ref_, ref_stride, size)?;

    let w = size.width();
    let h = size.height();

    // Fast path: packed layout where SIMD intrinsics can load contiguous data
    let is_packed = src_stride == w && ref_stride == w;

    match size {
        SubBlockSize::B8x8 if is_packed => {
            #[cfg(target_arch = "x86_64")]
            if is_x86_feature_detected!("sse4.1") {
                let mut src_arr = [0u8; 64];
                let mut ref_arr = [0u8; 64];
                src_arr.copy_from_slice(&src[..64]);
                ref_arr.copy_from_slice(&ref_[..64]);
                // SAFETY: sse4.1 feature confirmed by is_x86_feature_detected!
                return Ok(unsafe { sad_8x8_sse41(&src_arr, &ref_arr) });
            }

            #[cfg(target_arch = "aarch64")]
            {
                let mut src_arr = [0u8; 64];
                let mut ref_arr = [0u8; 64];
                src_arr.copy_from_slice(&src[..64]);
                ref_arr.copy_from_slice(&ref_[..64]);
                // SAFETY: NEON is always available on aarch64
                return Ok(unsafe { sad_8x8_neon(&src_arr, &ref_arr) });
            }

            #[allow(unreachable_code)]
            Ok(sad_scalar_strided(src, src_stride, ref_, ref_stride, w, h))
        }

        SubBlockSize::B4x4 if is_packed => {
            #[cfg(target_arch = "x86_64")]
            if is_x86_feature_detected!("sse4.1") {
                let mut src_arr = [0u8; 16];
                let mut ref_arr = [0u8; 16];
                src_arr.copy_from_slice(&src[..16]);
                ref_arr.copy_from_slice(&ref_[..16]);
                // SAFETY: sse4.1 feature confirmed by is_x86_feature_detected!
                return Ok(unsafe { sad_4x4_sse41(&src_arr, &ref_arr) });
            }

            #[allow(unreachable_code)]
            Ok(sad_scalar_strided(src, src_stride, ref_, ref_stride, w, h))
        }

        // Rectangular blocks or non-packed: always scalar
        _ => Ok(sad_scalar_strided(src, src_stride, ref_, ref_stride, w, h)),
    }
}

/// Compute SAD for a 4×4 block (packed, stride == 4).
///
/// Convenience wrapper around [`sad_subblock`].
///
/// # Errors
///
/// Returns [`SimdError::InvalidBufferSize`] if either slice has fewer than 16
/// elements.
pub fn sad_4x4(src: &[u8], ref_: &[u8]) -> Result<u32> {
    sad_subblock(src, 4, ref_, 4, SubBlockSize::B4x4)
}

/// Compute SAD for an 8×8 block (packed, stride == 8).
///
/// Convenience wrapper around [`sad_subblock`].
///
/// # Errors
///
/// Returns [`SimdError::InvalidBufferSize`] if either slice has fewer than 64
/// elements.
pub fn sad_8x8(src: &[u8], ref_: &[u8]) -> Result<u32> {
    sad_subblock(src, 8, ref_, 8, SubBlockSize::B8x8)
}

/// Sum of absolute differences over a larger frame using a sliding window of
/// sub-blocks.  Returns a `Vec<(row, col, sad)>` tuple for every non-overlapping
/// `size`-block position in a `frame_w × frame_h` luma plane.
///
/// This is the building block for a dense SAD map used in diamond/hexagonal
/// motion search.
///
/// # Errors
///
/// Returns [`SimdError::InvalidBufferSize`] if either `src` or `ref_` is
/// smaller than `frame_w * frame_h` pixels.
pub fn sad_tiled(
    src: &[u8],
    ref_: &[u8],
    frame_w: usize,
    frame_h: usize,
    size: SubBlockSize,
) -> Result<Vec<(usize, usize, u32)>> {
    if src.len() < frame_w * frame_h || ref_.len() < frame_w * frame_h {
        return Err(SimdError::InvalidBufferSize);
    }

    let bw = size.width();
    let bh = size.height();

    let tiles_x = frame_w / bw;
    let tiles_y = frame_h / bh;

    let mut results = Vec::with_capacity(tiles_x * tiles_y);

    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            let row_off = ty * bh;
            let col_off = tx * bw;
            let src_off = row_off * frame_w + col_off;
            let ref_off = row_off * frame_w + col_off;

            let s = sad_subblock(&src[src_off..], frame_w, &ref_[ref_off..], frame_w, size)?;
            results.push((row_off, col_off, s));
        }
    }

    Ok(results)
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── SubBlockSize properties ───────────────────────────────────────────────

    #[test]
    fn sub_block_size_dimensions() {
        assert_eq!(SubBlockSize::B4x4.width(), 4);
        assert_eq!(SubBlockSize::B4x4.height(), 4);
        assert_eq!(SubBlockSize::B4x4.pixels(), 16);

        assert_eq!(SubBlockSize::B8x8.width(), 8);
        assert_eq!(SubBlockSize::B8x8.height(), 8);
        assert_eq!(SubBlockSize::B8x8.pixels(), 64);

        assert_eq!(SubBlockSize::B4x8.width(), 4);
        assert_eq!(SubBlockSize::B4x8.height(), 8);
        assert_eq!(SubBlockSize::B4x8.pixels(), 32);

        assert_eq!(SubBlockSize::B8x4.width(), 8);
        assert_eq!(SubBlockSize::B8x4.height(), 4);
        assert_eq!(SubBlockSize::B8x4.pixels(), 32);
    }

    // ── SAD correctness ───────────────────────────────────────────────────────

    #[test]
    fn sad_4x4_identical_blocks_is_zero() {
        let block = vec![42u8; 16];
        let result = sad_4x4(&block, &block).expect("sad_4x4 identical");
        assert_eq!(result, 0);
    }

    #[test]
    fn sad_8x8_identical_blocks_is_zero() {
        let block = vec![200u8; 64];
        let result = sad_8x8(&block, &block).expect("sad_8x8 identical");
        assert_eq!(result, 0);
    }

    #[test]
    fn sad_4x4_known_value() {
        // All src = 10, all ref = 20 → 16 pixels × 10 = 160
        let src = vec![10u8; 16];
        let ref_ = vec![20u8; 16];
        let result = sad_4x4(&src, &ref_).expect("sad_4x4 known");
        assert_eq!(result, 160);
    }

    #[test]
    fn sad_8x8_known_value() {
        // All src = 50, all ref = 100 → 64 pixels × 50 = 3200
        let src = vec![50u8; 64];
        let ref_ = vec![100u8; 64];
        let result = sad_8x8(&src, &ref_).expect("sad_8x8 known");
        assert_eq!(result, 3200);
    }

    #[test]
    fn sad_is_symmetric() {
        let a: Vec<u8> = (0..64).map(|i| (i * 3 % 256) as u8).collect();
        let b: Vec<u8> = (0..64).map(|i| (i * 2 + 10 % 256) as u8).collect();
        let ab = sad_8x8(&a, &b).expect("ab");
        let ba = sad_8x8(&b, &a).expect("ba");
        assert_eq!(ab, ba, "SAD must be symmetric");
    }

    #[test]
    fn sad_increases_with_difference() {
        let src = vec![100u8; 64];
        let small_diff = vec![105u8; 64];
        let large_diff = vec![180u8; 64];
        let s1 = sad_8x8(&src, &small_diff).expect("small");
        let s2 = sad_8x8(&src, &large_diff).expect("large");
        assert!(s2 > s1, "larger diff → larger SAD: s1={s1} s2={s2}");
    }

    #[test]
    fn sad_buffer_too_small_returns_error() {
        let a = vec![0u8; 10];
        let b = vec![0u8; 10];
        assert_eq!(sad_8x8(&a, &b), Err(SimdError::InvalidBufferSize));
    }

    // ── Strided layout ────────────────────────────────────────────────────────

    #[test]
    fn sad_strided_4x4_matches_packed() {
        // Build a 16-wide frame; extract the 4×4 block at column 4
        let mut src = vec![0u8; 8 * 16];
        let mut ref_ = vec![0u8; 8 * 16];
        for i in 0..16usize {
            src[i] = i as u8;
            ref_[i] = (i * 2) as u8;
        }
        // Build packed versions of the first 4×4 block
        let mut packed_src = [0u8; 16];
        let mut packed_ref = [0u8; 16];
        for row in 0..4usize {
            for col in 0..4usize {
                packed_src[row * 4 + col] = src[row * 16 + col];
                packed_ref[row * 4 + col] = ref_[row * 16 + col];
            }
        }
        let strided = sad_subblock(&src, 16, &ref_, 16, SubBlockSize::B4x4).expect("strided");
        let packed = sad_4x4(&packed_src, &packed_ref).expect("packed");
        assert_eq!(strided, packed, "strided must match packed");
    }

    // ── Tiled SAD ─────────────────────────────────────────────────────────────

    #[test]
    fn sad_tiled_identical_frames_all_zero() {
        let frame = vec![128u8; 32 * 32];
        let results = sad_tiled(&frame, &frame, 32, 32, SubBlockSize::B8x8).expect("tiled");
        assert_eq!(results.len(), 16, "32×32 / 8×8 = 16 tiles");
        for &(_, _, s) in &results {
            assert_eq!(s, 0, "identical frame → SAD = 0");
        }
    }

    #[test]
    fn sad_tiled_frame_too_small_returns_error() {
        let small = vec![0u8; 10];
        let frame = vec![0u8; 32 * 32];
        assert_eq!(
            sad_tiled(&small, &frame, 32, 32, SubBlockSize::B8x8),
            Err(SimdError::InvalidBufferSize)
        );
    }

    // ── Rectangular blocks ────────────────────────────────────────────────────

    #[test]
    fn sad_4x8_known_value() {
        // 4×8 = 32 pixels, src=10 ref=30 → 32 × 20 = 640
        let src = vec![10u8; 32];
        let ref_ = vec![30u8; 32];
        let result = sad_subblock(&src, 4, &ref_, 4, SubBlockSize::B4x8).expect("sad_4x8");
        assert_eq!(result, 640);
    }

    #[test]
    fn sad_8x4_known_value() {
        // 8×4 = 32 pixels, src=5 ref=10 → 32 × 5 = 160
        let src = vec![5u8; 32];
        let ref_ = vec![10u8; 32];
        let result = sad_subblock(&src, 8, &ref_, 8, SubBlockSize::B8x4).expect("sad_8x4");
        assert_eq!(result, 160);
    }
}
