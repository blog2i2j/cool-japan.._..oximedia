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
}
