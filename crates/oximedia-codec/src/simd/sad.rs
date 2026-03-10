//! Sum of Absolute Differences (SAD) operations.
//!
//! SAD is fundamental to motion estimation in video codecs. It measures
//! the similarity between blocks of pixels, with lower values indicating
//! better matches.
//!
//! This module provides optimized SAD calculations for common block sizes:
//! - 4x4 (used in H.264/AV1 for small partitions)
//! - 8x8 (common block size)
//! - 16x16 (macroblock size)
//! - 32x32 (used in HEVC/AV1)
//!
//! All functions are designed to map efficiently to SIMD instructions.

#![forbid(unsafe_code)]
// Allow loop indexing for SIMD-like element-wise operations
#![allow(clippy::needless_range_loop)]

use super::scalar::ScalarFallback;
use super::traits::SimdOps;
use super::types::U8x16;

/// SAD operations using SIMD.
pub struct SadOps<S: SimdOps> {
    simd: S,
}

impl<S: SimdOps + Default> Default for SadOps<S> {
    fn default() -> Self {
        Self::new(S::default())
    }
}

impl<S: SimdOps> SadOps<S> {
    /// Create a new SAD operations instance.
    #[inline]
    #[must_use]
    pub const fn new(simd: S) -> Self {
        Self { simd }
    }

    /// Get the underlying SIMD implementation.
    #[inline]
    #[must_use]
    pub const fn simd(&self) -> &S {
        &self.simd
    }

    /// Calculate SAD for a 4x4 block.
    ///
    /// # Arguments
    /// * `src` - Source block data (row-major, stride = `src_stride`)
    /// * `src_stride` - Stride between source rows
    /// * `ref_block` - Reference block data (row-major, stride = `ref_stride`)
    /// * `ref_stride` - Stride between reference rows
    ///
    /// # Returns
    /// Sum of absolute differences for all 16 pixels.
    #[inline]
    pub fn sad_4x4(
        &self,
        src: &[u8],
        src_stride: usize,
        ref_block: &[u8],
        ref_stride: usize,
    ) -> u32 {
        let mut sum = 0u32;

        for row in 0..4 {
            let src_offset = row * src_stride;
            let ref_offset = row * ref_stride;

            if src_offset + 4 <= src.len() && ref_offset + 4 <= ref_block.len() {
                for col in 0..4 {
                    let diff =
                        i32::from(src[src_offset + col]) - i32::from(ref_block[ref_offset + col]);
                    sum += diff.unsigned_abs();
                }
            }
        }

        sum
    }

    /// Calculate SAD for an 8x8 block.
    #[inline]
    pub fn sad_8x8(
        &self,
        src: &[u8],
        src_stride: usize,
        ref_block: &[u8],
        ref_stride: usize,
    ) -> u32 {
        let mut sum = 0u32;

        for row in 0..8 {
            let src_offset = row * src_stride;
            let ref_offset = row * ref_stride;

            if src_offset + 8 <= src.len() && ref_offset + 8 <= ref_block.len() {
                sum += self.simd.sad_8(
                    &src[src_offset..src_offset + 8],
                    &ref_block[ref_offset..ref_offset + 8],
                );
            }
        }

        sum
    }

    /// Calculate SAD for a 16x16 block.
    #[inline]
    pub fn sad_16x16(
        &self,
        src: &[u8],
        src_stride: usize,
        ref_block: &[u8],
        ref_stride: usize,
    ) -> u32 {
        let mut sum = 0u32;

        for row in 0..16 {
            let src_offset = row * src_stride;
            let ref_offset = row * ref_stride;

            if src_offset + 16 <= src.len() && ref_offset + 16 <= ref_block.len() {
                let src_row = U8x16::from_array(
                    src[src_offset..src_offset + 16]
                        .try_into()
                        .unwrap_or([0; 16]),
                );
                let ref_row = U8x16::from_array(
                    ref_block[ref_offset..ref_offset + 16]
                        .try_into()
                        .unwrap_or([0; 16]),
                );
                sum += self.simd.sad_u8x16(src_row, ref_row);
            }
        }

        sum
    }

    /// Calculate SAD for a 32x32 block.
    #[inline]
    pub fn sad_32x32(
        &self,
        src: &[u8],
        src_stride: usize,
        ref_block: &[u8],
        ref_stride: usize,
    ) -> u32 {
        let mut sum = 0u32;

        for row in 0..32 {
            let src_offset = row * src_stride;
            let ref_offset = row * ref_stride;

            if src_offset + 32 <= src.len() && ref_offset + 32 <= ref_block.len() {
                // Process as two 16-byte chunks
                for chunk in 0..2 {
                    let chunk_offset = chunk * 16;
                    let src_row = U8x16::from_array(
                        src[src_offset + chunk_offset..src_offset + chunk_offset + 16]
                            .try_into()
                            .unwrap_or([0; 16]),
                    );
                    let ref_row = U8x16::from_array(
                        ref_block[ref_offset + chunk_offset..ref_offset + chunk_offset + 16]
                            .try_into()
                            .unwrap_or([0; 16]),
                    );
                    sum += self.simd.sad_u8x16(src_row, ref_row);
                }
            }
        }

        sum
    }

    /// Calculate SAD for an arbitrary block size.
    ///
    /// Less efficient than size-specific functions but more flexible.
    #[allow(dead_code)]
    pub fn sad_nxn(
        &self,
        src: &[u8],
        src_stride: usize,
        ref_block: &[u8],
        ref_stride: usize,
        width: usize,
        height: usize,
    ) -> u32 {
        let mut sum = 0u32;

        for row in 0..height {
            let src_offset = row * src_stride;
            let ref_offset = row * ref_stride;

            if src_offset + width <= src.len() && ref_offset + width <= ref_block.len() {
                // Process 16-byte chunks
                let mut col = 0;
                while col + 16 <= width {
                    let src_chunk = U8x16::from_array(
                        src[src_offset + col..src_offset + col + 16]
                            .try_into()
                            .unwrap_or([0; 16]),
                    );
                    let ref_chunk = U8x16::from_array(
                        ref_block[ref_offset + col..ref_offset + col + 16]
                            .try_into()
                            .unwrap_or([0; 16]),
                    );
                    sum += self.simd.sad_u8x16(src_chunk, ref_chunk);
                    col += 16;
                }

                // Process remaining bytes
                while col < width {
                    let diff =
                        i32::from(src[src_offset + col]) - i32::from(ref_block[ref_offset + col]);
                    sum += diff.unsigned_abs();
                    col += 1;
                }
            }
        }

        sum
    }

    /// Calculate SATD (Sum of Absolute Transformed Differences) for 4x4 block.
    ///
    /// SATD applies a Hadamard transform before summing, providing a better
    /// cost metric for rate-distortion optimization.
    #[allow(dead_code)]
    pub fn satd_4x4(
        &self,
        src: &[u8],
        src_stride: usize,
        ref_block: &[u8],
        ref_stride: usize,
    ) -> u32 {
        // Calculate differences
        let mut diff = [[0i16; 4]; 4];
        for row in 0..4 {
            let src_offset = row * src_stride;
            let ref_offset = row * ref_stride;
            for col in 0..4 {
                if src_offset + col < src.len() && ref_offset + col < ref_block.len() {
                    diff[row][col] =
                        i16::from(src[src_offset + col]) - i16::from(ref_block[ref_offset + col]);
                }
            }
        }

        // Horizontal Hadamard
        let mut tmp = [[0i16; 4]; 4];
        for row in 0..4 {
            let a = diff[row][0] + diff[row][1];
            let b = diff[row][2] + diff[row][3];
            let c = diff[row][0] - diff[row][1];
            let d = diff[row][2] - diff[row][3];

            tmp[row][0] = a + b;
            tmp[row][1] = c + d;
            tmp[row][2] = a - b;
            tmp[row][3] = c - d;
        }

        // Vertical Hadamard
        let mut result = [[0i16; 4]; 4];
        for col in 0..4 {
            let a = tmp[0][col] + tmp[1][col];
            let b = tmp[2][col] + tmp[3][col];
            let c = tmp[0][col] - tmp[1][col];
            let d = tmp[2][col] - tmp[3][col];

            result[0][col] = a + b;
            result[1][col] = c + d;
            result[2][col] = a - b;
            result[3][col] = c - d;
        }

        // Sum absolute values
        let mut sum = 0u32;
        for row in 0..4 {
            for col in 0..4 {
                sum += u32::from(result[row][col].unsigned_abs());
            }
        }

        // Normalize (divide by 2 as Hadamard doubles values)
        (sum + 1) >> 1
    }
}

/// Create a SAD operations instance with scalar fallback.
#[inline]
#[must_use]
pub fn sad_ops() -> SadOps<ScalarFallback> {
    SadOps::new(ScalarFallback::new())
}

/// Calculate SAD for multiple candidate positions (motion search).
///
/// Returns the index of the best (lowest SAD) position.
#[allow(dead_code, clippy::cast_sign_loss)]
#[must_use]
pub fn find_best_match_4x4(
    src: &[u8],
    src_stride: usize,
    ref_frame: &[u8],
    ref_stride: usize,
    candidates: &[(i32, i32)],
    ref_width: usize,
    ref_height: usize,
) -> Option<(usize, u32)> {
    let ops = sad_ops();
    let mut best_idx = None;
    let mut best_sad = u32::MAX;

    for (idx, &(dx, dy)) in candidates.iter().enumerate() {
        // Check bounds
        if dx < 0 || dy < 0 {
            continue;
        }
        let x = dx as usize;
        let y = dy as usize;

        if x + 4 > ref_width || y + 4 > ref_height {
            continue;
        }

        let ref_offset = y * ref_stride + x;
        if ref_offset + 3 * ref_stride + 4 > ref_frame.len() {
            continue;
        }

        let sad = ops.sad_4x4(src, src_stride, &ref_frame[ref_offset..], ref_stride);

        if sad < best_sad {
            best_sad = sad;
            best_idx = Some(idx);
        }
    }

    best_idx.map(|idx| (idx, best_sad))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sad_4x4_identical() {
        let ops = sad_ops();

        let block = [
            100u8, 110, 120, 130, 105, 115, 125, 135, 110, 120, 130, 140, 115, 125, 135, 145,
        ];

        let sad = ops.sad_4x4(&block, 4, &block, 4);
        assert_eq!(sad, 0);
    }

    #[test]
    fn test_sad_4x4_constant_diff() {
        let ops = sad_ops();

        let src = [100u8; 16];
        let ref_block = [110u8; 16];

        // Each pixel differs by 10, 16 pixels = 160
        let sad = ops.sad_4x4(&src, 4, &ref_block, 4);
        assert_eq!(sad, 160);
    }

    #[test]
    fn test_sad_8x8_identical() {
        let ops = sad_ops();

        let block = [128u8; 64];
        let sad = ops.sad_8x8(&block, 8, &block, 8);
        assert_eq!(sad, 0);
    }

    #[test]
    fn test_sad_8x8_constant_diff() {
        let ops = sad_ops();

        let src = [100u8; 64];
        let ref_block = [105u8; 64];

        // Each pixel differs by 5, 64 pixels = 320
        let sad = ops.sad_8x8(&src, 8, &ref_block, 8);
        assert_eq!(sad, 320);
    }

    #[test]
    fn test_sad_16x16_identical() {
        let ops = sad_ops();

        let block = [128u8; 256];
        let sad = ops.sad_16x16(&block, 16, &block, 16);
        assert_eq!(sad, 0);
    }

    #[test]
    fn test_sad_16x16_constant_diff() {
        let ops = sad_ops();

        let src = [100u8; 256];
        let ref_block = [102u8; 256];

        // Each pixel differs by 2, 256 pixels = 512
        let sad = ops.sad_16x16(&src, 16, &ref_block, 16);
        assert_eq!(sad, 512);
    }

    #[test]
    fn test_sad_32x32_identical() {
        let ops = sad_ops();

        let block = [128u8; 1024];
        let sad = ops.sad_32x32(&block, 32, &block, 32);
        assert_eq!(sad, 0);
    }

    #[test]
    fn test_sad_with_stride() {
        let ops = sad_ops();

        // Create a larger buffer with stride > block width
        let stride = 8;
        let mut src = [0u8; 32]; // 4 rows * 8 stride
        let mut ref_block = [0u8; 32];

        for row in 0..4 {
            for col in 0..4 {
                src[row * stride + col] = 100;
                ref_block[row * stride + col] = 110;
            }
        }

        let sad = ops.sad_4x4(&src, stride, &ref_block, stride);
        assert_eq!(sad, 160); // 16 pixels * 10 diff
    }

    #[test]
    fn test_satd_4x4_identical() {
        let ops = sad_ops();

        let block = [128u8; 16];
        let satd = ops.satd_4x4(&block, 4, &block, 4);
        assert_eq!(satd, 0);
    }

    #[test]
    fn test_satd_4x4_constant_diff() {
        let ops = sad_ops();

        let src = [100u8; 16];
        let ref_block = [110u8; 16];

        // SATD of constant difference is special case
        let satd = ops.satd_4x4(&src, 4, &ref_block, 4);
        // After Hadamard, DC coefficient captures all energy
        // Result should be 16 * 10 / 2 = 80 (approximately)
        assert!(satd > 0);
    }

    #[test]
    fn test_find_best_match() {
        let src = [100u8; 16];

        // Create a reference frame with the matching block at (4, 4)
        let mut ref_frame = [50u8; 256]; // 16x16
        for row in 0..4 {
            for col in 0..4 {
                ref_frame[(row + 4) * 16 + col + 4] = 100;
            }
        }

        let candidates = vec![
            (0, 0),
            (4, 0),
            (0, 4),
            (4, 4), // This should be the best match
            (8, 8),
        ];

        let result = find_best_match_4x4(&src, 4, &ref_frame, 16, &candidates, 16, 16);
        assert!(result.is_some());
        let (idx, sad) = result.expect("should succeed");
        assert_eq!(idx, 3); // (4, 4) is at index 3
        assert_eq!(sad, 0); // Perfect match
    }

    #[test]
    fn test_sad_nxn() {
        let ops = sad_ops();

        // Test 12x12 block (non-power-of-2)
        let src = [100u8; 144]; // 12x12
        let ref_block = [103u8; 144];

        let sad = ops.sad_nxn(&src, 12, &ref_block, 12, 12, 12);
        // 144 pixels * 3 diff = 432
        assert_eq!(sad, 432);
    }
}
