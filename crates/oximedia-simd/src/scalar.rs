//! Scalar fallback implementations for SIMD operations
//!
//! These implementations provide correct algorithmic behaviour used when
//! hardware SIMD is not available, and also serve as reference for testing.
#![allow(clippy::too_many_arguments)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_possible_wrap)]

use crate::{DctSize, InterpolationFilter, Result};

// ── DCT / IDCT helpers ─────────────────────────────────────────────────────

/// Pre-computed cosine table for 1-D DCT-II of a given size N.
///
/// Entry `[k][n]` = `cos(pi * (2*n + 1) * k / (2*N))`.
fn dct_cosine_table<const N: usize>() -> [[f64; N]; N] {
    let mut table = [[0.0f64; N]; N];
    let factor = std::f64::consts::PI / (2.0 * N as f64);
    for k in 0..N {
        for n in 0..N {
            table[k][n] = (factor * k as f64 * (2 * n + 1) as f64).cos();
        }
    }
    table
}

/// Normalization factor C(k) for DCT-II: `1/sqrt(N)` for k==0,
/// `sqrt(2/N)` otherwise.
fn dct_norm_factor(k: usize, n: usize) -> f64 {
    if k == 0 {
        1.0 / (n as f64).sqrt()
    } else {
        (2.0 / n as f64).sqrt()
    }
}

/// Forward 1-D DCT-II of length N (in-place via `src` -> `dst`).
fn forward_dct_1d(src: &[f64], dst: &mut [f64], cos_table: &[&[f64]], n: usize) {
    for k in 0..n {
        let norm = dct_norm_factor(k, n);
        let mut sum = 0.0f64;
        for i in 0..n {
            sum += src[i] * cos_table[k][i];
        }
        dst[k] = norm * sum;
    }
}

/// Inverse 1-D DCT-II (DCT-III) of length N.
fn inverse_dct_1d(src: &[f64], dst: &mut [f64], cos_table: &[&[f64]], n: usize) {
    for i in 0..n {
        let mut sum = 0.0f64;
        for k in 0..n {
            let norm = dct_norm_factor(k, n);
            sum += norm * src[k] * cos_table[k][i];
        }
        dst[i] = sum;
    }
}

/// Generic 2-D separable forward DCT-II for an NxN block.
fn forward_dct_2d_generic(input: &[i16], output: &mut [i16], n: usize) {
    // Build cosine table
    let factor = std::f64::consts::PI / (2.0 * n as f64);
    let mut cos_flat = vec![0.0f64; n * n];
    for k in 0..n {
        for j in 0..n {
            cos_flat[k * n + j] = (factor * k as f64 * (2 * j + 1) as f64).cos();
        }
    }
    let cos_rows: Vec<&[f64]> = (0..n).map(|k| &cos_flat[k * n..(k + 1) * n]).collect();

    let total = n * n;
    let mut temp = vec![0.0f64; total];
    let mut row_src = vec![0.0f64; n];
    let mut row_dst = vec![0.0f64; n];

    // Transform rows
    for r in 0..n {
        for c in 0..n {
            row_src[c] = f64::from(input[r * n + c]);
        }
        forward_dct_1d(&row_src, &mut row_dst, &cos_rows, n);
        for c in 0..n {
            temp[r * n + c] = row_dst[c];
        }
    }

    // Transform columns
    let mut col_src = vec![0.0f64; n];
    let mut col_dst = vec![0.0f64; n];
    for c in 0..n {
        for r in 0..n {
            col_src[r] = temp[r * n + c];
        }
        forward_dct_1d(&col_src, &mut col_dst, &cos_rows, n);
        for r in 0..n {
            output[r * n + c] = col_dst[r].round().clamp(-32768.0, 32767.0) as i16;
        }
    }
}

/// Generic 2-D separable inverse DCT (DCT-III) for an NxN block.
fn inverse_dct_2d_generic(input: &[i16], output: &mut [i16], n: usize) {
    let factor = std::f64::consts::PI / (2.0 * n as f64);
    let mut cos_flat = vec![0.0f64; n * n];
    for k in 0..n {
        for j in 0..n {
            cos_flat[k * n + j] = (factor * k as f64 * (2 * j + 1) as f64).cos();
        }
    }
    let cos_rows: Vec<&[f64]> = (0..n).map(|k| &cos_flat[k * n..(k + 1) * n]).collect();

    let total = n * n;
    let mut temp = vec![0.0f64; total];
    let mut row_src = vec![0.0f64; n];
    let mut row_dst = vec![0.0f64; n];

    // Inverse-transform rows
    for r in 0..n {
        for c in 0..n {
            row_src[c] = f64::from(input[r * n + c]);
        }
        inverse_dct_1d(&row_src, &mut row_dst, &cos_rows, n);
        for c in 0..n {
            temp[r * n + c] = row_dst[c];
        }
    }

    // Inverse-transform columns
    let mut col_src = vec![0.0f64; n];
    let mut col_dst = vec![0.0f64; n];
    for c in 0..n {
        for r in 0..n {
            col_src[r] = temp[r * n + c];
        }
        inverse_dct_1d(&col_src, &mut col_dst, &cos_rows, n);
        for r in 0..n {
            output[r * n + c] = col_dst[r].round().clamp(-32768.0, 32767.0) as i16;
        }
    }
}

// ── Public DCT entry points ────────────────────────────────────────────────

/// Scalar forward DCT implementation using the real DCT-II transform.
#[allow(clippy::unnecessary_wraps)]
pub fn forward_dct_scalar(input: &[i16], output: &mut [i16], size: DctSize) -> Result<()> {
    let n = match size {
        DctSize::Dct4x4 => 4,
        DctSize::Dct8x8 => 8,
        DctSize::Dct16x16 => 16,
        DctSize::Dct32x32 => 32,
        DctSize::Dct64x64 => 64,
    };
    forward_dct_2d_generic(input, output, n);
    Ok(())
}

/// Scalar inverse DCT implementation using the real DCT-III (IDCT) transform.
#[allow(clippy::unnecessary_wraps)]
pub fn inverse_dct_scalar(input: &[i16], output: &mut [i16], size: DctSize) -> Result<()> {
    let n = match size {
        DctSize::Dct4x4 => 4,
        DctSize::Dct8x8 => 8,
        DctSize::Dct16x16 => 16,
        DctSize::Dct32x32 => 32,
        DctSize::Dct64x64 => 64,
    };
    inverse_dct_2d_generic(input, output, n);
    Ok(())
}

// ── Hadamard transform ──────────────────────────────────────────────────────

/// 1-D Hadamard–Walsh transform of length N (must be a power of 2).
///
/// The transform is applied in-place via the fast butterfly algorithm.
/// The un-normalised WHT satisfies `WHT(WHT(x)) = N * x`, so callers that
/// need energy-preserving round-trips must divide by N.
///
/// `work` must have length exactly `n` (power-of-two).
pub fn hadamard_1d(work: &mut [i32]) {
    let n = work.len();
    debug_assert!(n.is_power_of_two(), "Hadamard length must be power of two");

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

/// 2-D separable Hadamard–Walsh transform for an `n × n` block (in-place).
///
/// Applies 1-D WHT to each row, then to each column.  The result contains
/// un-normalised WHT coefficients (`divide by n²` to recover original energy).
///
/// `block` must have exactly `n * n` elements.  `n` must be a power of two.
pub fn hadamard_2d(block: &mut [i32], n: usize) {
    debug_assert_eq!(block.len(), n * n);
    debug_assert!(n.is_power_of_two());

    // Row transforms
    let mut row_buf = vec![0i32; n];
    for r in 0..n {
        row_buf.copy_from_slice(&block[r * n..(r + 1) * n]);
        hadamard_1d(&mut row_buf);
        block[r * n..(r + 1) * n].copy_from_slice(&row_buf);
    }

    // Column transforms
    let mut col_buf = vec![0i32; n];
    for c in 0..n {
        for r in 0..n {
            col_buf[r] = block[r * n + c];
        }
        hadamard_1d(&mut col_buf);
        for r in 0..n {
            block[r * n + c] = col_buf[r];
        }
    }
}

/// Compute SATD for two `n × n` pixel blocks using the Hadamard transform.
///
/// SATD = (1 / n) × Σ |WHT(src - ref_)|  (sum over all WHT coefficients).
/// This is a perceptually better motion-estimation cost than plain SAD because
/// it is insensitive to DC bias.
///
/// Both `src` and `ref_` must have at least `n * n` elements.  `n` must be a
/// power of two (4 or 8 are the most common values in codec practice).
///
/// Returns the SATD value (un-normalised; comparable to SAD for the same N).
pub fn satd_scalar_nxn(src: &[u8], ref_: &[u8], n: usize) -> u32 {
    debug_assert!(n.is_power_of_two());
    debug_assert!(src.len() >= n * n);
    debug_assert!(ref_.len() >= n * n);

    let mut diff = vec![0i32; n * n];
    for (d, (&s, &r)) in diff
        .iter_mut()
        .zip(src[..n * n].iter().zip(ref_[..n * n].iter()))
    {
        *d = i32::from(s) - i32::from(r);
    }

    hadamard_2d(&mut diff, n);

    // Sum of absolute WHT coefficients (un-normalised)
    diff.iter().map(|&v| v.unsigned_abs()).sum()
}

// ── Interpolation ──────────────────────────────────────────────────────────

/// Scalar interpolation implementation dispatch.
pub fn interpolate_scalar(
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
    match filter {
        InterpolationFilter::Bilinear => {
            interpolate_bilinear_scalar(src, dst, src_stride, dst_stride, width, height, dx, dy)
        }
        InterpolationFilter::Bicubic => {
            interpolate_bicubic_scalar(src, dst, src_stride, dst_stride, width, height, dx, dy)
        }
        InterpolationFilter::EightTap => {
            interpolate_8tap_scalar(src, dst, src_stride, dst_stride, width, height, dx, dy)
        }
        InterpolationFilter::Lanczos => {
            interpolate_lanczos_scalar(src, dst, src_stride, dst_stride, width, height, dx, dy)
        }
    }
}

/// Bilinear interpolation (2x2 kernel).
#[allow(clippy::unnecessary_wraps)]
fn interpolate_bilinear_scalar(
    src: &[u8],
    dst: &mut [u8],
    src_stride: usize,
    dst_stride: usize,
    width: usize,
    height: usize,
    dx: i32,
    dy: i32,
) -> Result<()> {
    let fx = dx & 15;
    let fy = dy & 15;

    for y in 0..height {
        // Prefetch the next row's data into L1 cache (non-faulting — safe on any
        // pointer; the CPU simply ignores prefetches that would fault).
        #[cfg(target_arch = "x86_64")]
        {
            let next_row_start = (y + 2) * src_stride;
            if next_row_start < src.len() {
                // SAFETY: `_mm_prefetch` is a hint instruction and never faults.
                // We check bounds to avoid hinting past the buffer.
                unsafe {
                    use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
                    _mm_prefetch(src.as_ptr().add(next_row_start).cast::<i8>(), _MM_HINT_T0);
                }
            }
        }

        for x in 0..width {
            let src_idx = y * src_stride + x;
            let p00 = i32::from(src[src_idx]);
            let p01 = i32::from(src[src_idx + 1]);
            let p10 = i32::from(src[src_idx + src_stride]);
            let p11 = i32::from(src[src_idx + src_stride + 1]);

            let v0 = p00 * (16 - fx) + p01 * fx;
            let v1 = p10 * (16 - fx) + p11 * fx;
            let v = (v0 * (16 - fy) + v1 * fy + 128) >> 8;

            #[allow(clippy::cast_sign_loss)]
            {
                dst[y * dst_stride + x] = v.clamp(0, 255) as u8;
            }
        }
    }

    Ok(())
}

/// Catmull-Rom cubic interpolation kernel.
///
/// `t` is the fractional position in [0, 1).  Returns weights for the four
/// taps at positions -1, 0, 1, 2 relative to the integer sample.
fn catmull_rom_weights(t: f64) -> [f64; 4] {
    let t2 = t * t;
    let t3 = t2 * t;
    [
        -0.5 * t3 + t2 - 0.5 * t,
        1.5 * t3 - 2.5 * t2 + 1.0,
        -1.5 * t3 + 2.0 * t2 + 0.5 * t,
        0.5 * t3 - 0.5 * t2,
    ]
}

/// Fetch a source sample with clamped addressing.
fn fetch_clamped(src: &[u8], stride: usize, x: i32, y: i32, max_x: i32, max_y: i32) -> f64 {
    let cx = x.clamp(0, max_x) as usize;
    let cy = y.clamp(0, max_y) as usize;
    let idx = cy * stride + cx;
    if idx < src.len() {
        f64::from(src[idx])
    } else {
        0.0
    }
}

/// Bicubic interpolation using Catmull-Rom 4-tap kernel.
#[allow(clippy::unnecessary_wraps)]
fn interpolate_bicubic_scalar(
    src: &[u8],
    dst: &mut [u8],
    src_stride: usize,
    dst_stride: usize,
    width: usize,
    height: usize,
    dx: i32,
    dy: i32,
) -> Result<()> {
    let frac_x = (dx & 15) as f64 / 16.0;
    let frac_y = (dy & 15) as f64 / 16.0;

    let wx = catmull_rom_weights(frac_x);
    let wy = catmull_rom_weights(frac_y);

    let max_x = (width as i32).saturating_sub(1).max(0);
    // Estimate max_y from the source buffer size
    let src_height = src.len().checked_div(src_stride).unwrap_or(height);
    let max_y = (src_height as i32).saturating_sub(1).max(0);

    for row in 0..height {
        for col in 0..width {
            let base_x = col as i32;
            let base_y = row as i32;

            // 4x4 convolution
            let mut sum = 0.0f64;
            for ky in 0..4i32 {
                let sy = base_y + ky - 1;
                let mut row_sum = 0.0f64;
                for kx in 0..4i32 {
                    let sx = base_x + kx - 1;
                    row_sum +=
                        wx[kx as usize] * fetch_clamped(src, src_stride, sx, sy, max_x, max_y);
                }
                sum += wy[ky as usize] * row_sum;
            }

            #[allow(clippy::cast_sign_loss)]
            {
                dst[row * dst_stride + col] = sum.round().clamp(0.0, 255.0) as u8;
            }
        }
    }

    Ok(())
}

/// 8-tap sub-pixel interpolation filter coefficients (AV1/VP9 style).
///
/// Each set of 8 coefficients sums to 128.  The fractional position index
/// 0..15 selects one of 16 filter phases.  Index 0 is the identity (centre
/// tap only).
const EIGHT_TAP_FILTERS: [[i16; 8]; 16] = [
    [0, 0, 0, 128, 0, 0, 0, 0],
    [0, 1, -5, 126, 8, -3, 1, 0],
    [-1, 3, -10, 122, 18, -6, 2, 0],
    [-1, 4, -13, 118, 27, -9, 3, -1],
    [-1, 4, -16, 112, 37, -11, 4, -1],
    [-1, 5, -18, 105, 48, -14, 4, -1],
    [-1, 5, -19, 97, 58, -16, 5, -1],
    [-1, 6, -19, 88, 68, -18, 5, -1],
    [-1, 6, -19, 78, 78, -19, 6, -1],
    [-1, 5, -18, 68, 88, -19, 6, -1],
    [-1, 5, -16, 58, 97, -19, 5, -1],
    [-1, 4, -14, 48, 105, -18, 5, -1],
    [-1, 4, -11, 37, 112, -16, 4, -1],
    [-1, 3, -9, 27, 118, -13, 4, -1],
    [0, 2, -6, 18, 122, -10, 3, -1],
    [0, 1, -3, 8, 126, -5, 1, 0],
];

/// 8-tap sub-pixel interpolation (AV1/VP9 style filter kernel).
#[allow(clippy::unnecessary_wraps)]
fn interpolate_8tap_scalar(
    src: &[u8],
    dst: &mut [u8],
    src_stride: usize,
    dst_stride: usize,
    width: usize,
    height: usize,
    dx: i32,
    dy: i32,
) -> Result<()> {
    let phase_x = (dx & 15) as usize;
    let phase_y = (dy & 15) as usize;

    let filter_h = &EIGHT_TAP_FILTERS[phase_x];
    let filter_v = &EIGHT_TAP_FILTERS[phase_y];

    let src_height = src.len().checked_div(src_stride).unwrap_or(height);

    // Intermediate buffer: apply horizontal filter first.
    // We need 8 extra rows (3 above + 4 below) for the vertical pass.
    let v_start = 3i32;
    let rows_needed = height + 7;
    let mut intermediate = vec![0i32; rows_needed * width];

    for iy in 0..rows_needed {
        // Prefetch the row two ahead so it is warm when we process this row.
        #[cfg(target_arch = "x86_64")]
        {
            let next_src_y = (iy as i32 + 2) - v_start;
            let next_src_y_clamped = next_src_y.clamp(0, src_height as i32 - 1) as usize;
            let next_row_ptr_offset = next_src_y_clamped * src_stride;
            if next_row_ptr_offset < src.len() {
                // SAFETY: prefetch is a non-faulting hint; bounds checked above.
                unsafe {
                    use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
                    _mm_prefetch(
                        src.as_ptr().add(next_row_ptr_offset).cast::<i8>(),
                        _MM_HINT_T0,
                    );
                }
            }
        }
        let src_y = (iy as i32) - v_start;
        for x in 0..width {
            let mut sum = 0i32;
            for t in 0..8i32 {
                let sx = (x as i32 + t - 3).clamp(0, width as i32 - 1) as usize;
                let sy = src_y.clamp(0, src_height as i32 - 1) as usize;
                let idx = sy * src_stride + sx;
                let sample = if idx < src.len() {
                    i32::from(src[idx])
                } else {
                    0
                };
                sum += sample * i32::from(filter_h[t as usize]);
            }
            intermediate[iy * width + x] = sum;
        }
    }

    // Vertical pass
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0i64;
            for t in 0..8usize {
                let iy = y + t; // v_start offset is already in intermediate layout
                let val = i64::from(intermediate[iy * width + x]);
                sum += val * i64::from(filter_v[t]);
            }
            // Two rounds of rounding: first 7 bits (horizontal was *128), then 7 bits again
            let rounded = ((sum + 8192) >> 14).clamp(0, 255);

            #[allow(clippy::cast_sign_loss)]
            {
                dst[y * dst_stride + x] = rounded as u8;
            }
        }
    }

    Ok(())
}

// ── Lanczos interpolation ──────────────────────────────────────────────────

/// Compute the Lanczos kernel value `L(x)` for `a = 3` lobes.
///
/// L(x) = sinc(x) * sinc(x / a)  for |x| < a
/// L(x) = 0                       for |x| >= a
fn lanczos_kernel(x: f64, a: f64) -> f64 {
    if x.abs() < f64::EPSILON {
        return 1.0;
    }
    if x.abs() >= a {
        return 0.0;
    }
    let pi = std::f64::consts::PI;
    let sinc_x = (pi * x).sin() / (pi * x);
    let sinc_xa = (pi * x / a).sin() / (pi * x / a);
    sinc_x * sinc_xa
}

/// Lanczos resampling interpolation (a=3, 6×6 kernel).
///
/// Uses a windowed sinc filter with 3 lobes for each dimension.  The
/// fractional position is encoded in `dx` and `dy` as 1/16th pixel units
/// (0–15, consistent with the other filter phases).
///
/// The filter provides higher quality than Bicubic for downscaling at the
/// cost of a larger kernel (6 taps vs 4 taps per dimension).
#[allow(clippy::unnecessary_wraps)]
fn interpolate_lanczos_scalar(
    src: &[u8],
    dst: &mut [u8],
    src_stride: usize,
    dst_stride: usize,
    width: usize,
    height: usize,
    dx: i32,
    dy: i32,
) -> Result<()> {
    const A: f64 = 3.0; // Lanczos-3: 3 lobes → 6-tap kernel

    let frac_x = (dx & 15) as f64 / 16.0;
    let frac_y = (dy & 15) as f64 / 16.0;

    let src_height = src.len().checked_div(src_stride).unwrap_or(height + 8);
    let max_x = (width as i32).saturating_sub(1).max(0);
    let max_y = (src_height as i32).saturating_sub(1).max(0);

    // Pre-compute the 6 Lanczos weights for x and y dimensions
    let mut wx = [0.0f64; 6];
    let mut wy = [0.0f64; 6];
    for (tap, w) in wx.iter_mut().enumerate() {
        // taps at offsets -2, -1, 0, +1, +2, +3 relative to integer position
        *w = lanczos_kernel(frac_x - (tap as f64 - 2.0), A);
    }
    for (tap, w) in wy.iter_mut().enumerate() {
        *w = lanczos_kernel(frac_y - (tap as f64 - 2.0), A);
    }

    // Normalize weights to prevent DC gain error from discrete sampling
    let wx_sum: f64 = wx.iter().sum();
    let wy_sum: f64 = wy.iter().sum();
    let wx_norm: [f64; 6] = if wx_sum.abs() > f64::EPSILON {
        wx.map(|w| w / wx_sum)
    } else {
        wx
    };
    let wy_norm: [f64; 6] = if wy_sum.abs() > f64::EPSILON {
        wy.map(|w| w / wy_sum)
    } else {
        wy
    };

    for row in 0..height {
        for col in 0..width {
            let base_x = col as i32;
            let base_y = row as i32;

            let mut acc = 0.0f64;
            for (ky, &wy_k) in wy_norm.iter().enumerate() {
                let sy = (base_y + ky as i32 - 2).clamp(0, max_y) as usize;
                let mut row_acc = 0.0f64;
                for (kx, &wx_k) in wx_norm.iter().enumerate() {
                    let sx = (base_x + kx as i32 - 2).clamp(0, max_x) as usize;
                    let idx = sy * src_stride + sx;
                    let sample = if idx < src.len() {
                        f64::from(src[idx])
                    } else {
                        0.0
                    };
                    row_acc += wx_k * sample;
                }
                acc += wy_k * row_acc;
            }

            #[allow(clippy::cast_sign_loss)]
            {
                dst[row * dst_stride + col] = acc.round().clamp(0.0, 255.0) as u8;
            }
        }
    }

    Ok(())
}

// ── SAD ────────────────────────────────────────────────────────────────────

/// Scalar SAD implementation.
#[allow(clippy::unnecessary_wraps)]
pub fn sad_scalar(
    src1: &[u8],
    src2: &[u8],
    stride1: usize,
    stride2: usize,
    width: usize,
    height: usize,
) -> Result<u32> {
    let mut sad = 0u32;

    for y in 0..height {
        for x in 0..width {
            let p1 = i32::from(src1[y * stride1 + x]);
            let p2 = i32::from(src2[y * stride2 + x]);
            sad += (p1 - p2).unsigned_abs();
        }
    }

    Ok(sad)
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dct_4x4_roundtrip() {
        let input: Vec<i16> = (0..16).map(|i| (i * 10) as i16).collect();
        let mut dct_out = vec![0i16; 16];
        let mut idct_out = vec![0i16; 16];

        forward_dct_scalar(&input, &mut dct_out, DctSize::Dct4x4)
            .expect("forward DCT should succeed");
        inverse_dct_scalar(&dct_out, &mut idct_out, DctSize::Dct4x4)
            .expect("inverse DCT should succeed");

        for (i, (&orig, &recovered)) in input.iter().zip(idct_out.iter()).enumerate() {
            let diff = (i32::from(orig) - i32::from(recovered)).abs();
            assert!(
                diff <= 2,
                "sample {i}: orig={orig} recovered={recovered} diff={diff}"
            );
        }
    }

    #[test]
    fn test_dct_8x8_roundtrip() {
        let input: Vec<i16> = (0..64).map(|i| ((i * 4) % 200) as i16).collect();
        let mut dct_out = vec![0i16; 64];
        let mut idct_out = vec![0i16; 64];

        forward_dct_scalar(&input, &mut dct_out, DctSize::Dct8x8)
            .expect("forward DCT should succeed");
        inverse_dct_scalar(&dct_out, &mut idct_out, DctSize::Dct8x8)
            .expect("inverse DCT should succeed");

        for (i, (&orig, &recovered)) in input.iter().zip(idct_out.iter()).enumerate() {
            let diff = (i32::from(orig) - i32::from(recovered)).abs();
            assert!(
                diff <= 2,
                "sample {i}: orig={orig} recovered={recovered} diff={diff}"
            );
        }
    }

    #[test]
    fn test_dct_16x16_roundtrip() {
        let input: Vec<i16> = (0..256).map(|i| ((i * 3) % 150) as i16).collect();
        let mut dct_out = vec![0i16; 256];
        let mut idct_out = vec![0i16; 256];

        forward_dct_scalar(&input, &mut dct_out, DctSize::Dct16x16)
            .expect("forward DCT should succeed");
        inverse_dct_scalar(&dct_out, &mut idct_out, DctSize::Dct16x16)
            .expect("inverse DCT should succeed");

        for (i, (&orig, &recovered)) in input.iter().zip(idct_out.iter()).enumerate() {
            let diff = (i32::from(orig) - i32::from(recovered)).abs();
            assert!(
                diff <= 2,
                "sample {i}: orig={orig} recovered={recovered} diff={diff}"
            );
        }
    }

    #[test]
    fn test_dct_32x32_roundtrip() {
        let input: Vec<i16> = (0..1024).map(|i| ((i * 2) % 100) as i16).collect();
        let mut dct_out = vec![0i16; 1024];
        let mut idct_out = vec![0i16; 1024];

        forward_dct_scalar(&input, &mut dct_out, DctSize::Dct32x32)
            .expect("forward DCT should succeed");
        inverse_dct_scalar(&dct_out, &mut idct_out, DctSize::Dct32x32)
            .expect("inverse DCT should succeed");

        for (i, (&orig, &recovered)) in input.iter().zip(idct_out.iter()).enumerate() {
            let diff = (i32::from(orig) - i32::from(recovered)).abs();
            assert!(
                diff <= 2,
                "sample {i}: orig={orig} recovered={recovered} diff={diff}"
            );
        }
    }

    #[test]
    fn test_dct_64x64_roundtrip() {
        // Smaller value range to keep i16 headroom through the full 64×64 DCT.
        let input: Vec<i16> = (0..4096).map(|i| ((i % 50) as i16) - 25).collect();
        let mut dct_out = vec![0i16; 4096];
        let mut idct_out = vec![0i16; 4096];

        forward_dct_scalar(&input, &mut dct_out, DctSize::Dct64x64)
            .expect("forward DCT64 should succeed");
        inverse_dct_scalar(&dct_out, &mut idct_out, DctSize::Dct64x64)
            .expect("inverse DCT64 should succeed");

        // Round-trip tolerance is slightly larger for 64×64 due to accumulated
        // floating-point error across 64 butterfly stages.
        for (i, (&orig, &recovered)) in input.iter().zip(idct_out.iter()).enumerate() {
            let diff = (i32::from(orig) - i32::from(recovered)).abs();
            assert!(
                diff <= 4,
                "sample {i}: orig={orig} recovered={recovered} diff={diff}"
            );
        }
    }

    #[test]
    fn test_hadamard_1d_power_of_two() {
        // WHT of length 4: known values
        let mut buf = [1i32, 2, 3, 4];
        hadamard_1d(&mut buf);
        // WHT([1,2,3,4]) = [10, -2, -4, 0] (standard butterfly)
        assert_eq!(buf[0], 10, "DC coefficient");
        assert_eq!(buf[1], -2);
        assert_eq!(buf[2], -4);
        assert_eq!(buf[3], 0);
    }

    #[test]
    fn test_hadamard_2d_zero_is_zero() {
        let mut buf = vec![0i32; 16];
        hadamard_2d(&mut buf, 4);
        assert!(buf.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_satd_scalar_identical_is_zero() {
        let block = vec![100u8; 64];
        let result = satd_scalar_nxn(&block, &block, 8);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_satd_scalar_nonzero_for_different_blocks() {
        let a = vec![100u8; 64];
        let b = vec![200u8; 64];
        let result = satd_scalar_nxn(&a, &b, 8);
        assert!(result > 0, "SATD should be non-zero for different blocks");
    }

    #[test]
    fn test_bilinear_zero_fraction_is_copy() {
        // 4x4 source, stride=4.  With dx=0, dy=0 bilinear needs access to
        // src[y*stride + x + 1] and src[(y+1)*stride + x + 1], so we need
        // at least (height + 1) * stride bytes in src.
        let src = vec![
            10u8, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,
        ];
        // dst: 2 pixels wide, 2 rows, stride 2 => 4 bytes
        let mut dst = vec![0u8; 4];
        interpolate_bilinear_scalar(&src, &mut dst, 4, 2, 2, 2, 0, 0)
            .expect("interpolation should succeed");
        assert_eq!(dst[0], 10);
        assert_eq!(dst[1], 20);
        assert_eq!(dst[2], 50);
        assert_eq!(dst[3], 60);
    }

    #[test]
    fn test_catmull_rom_weights_sum_to_one() {
        for i in 0..16 {
            let t = i as f64 / 16.0;
            let w = catmull_rom_weights(t);
            let sum: f64 = w.iter().sum();
            assert!((sum - 1.0).abs() < 1e-10, "weights at t={t} sum to {sum}");
        }
    }

    #[test]
    fn test_8tap_filters_sum_to_128() {
        for (phase, filter) in EIGHT_TAP_FILTERS.iter().enumerate() {
            let sum: i16 = filter.iter().sum();
            assert_eq!(sum, 128, "phase {phase} sums to {sum}, expected 128");
        }
    }

    #[test]
    fn test_bicubic_identity_at_integer_position() {
        // With dx=0, dy=0, bicubic should closely reproduce the source
        let src = vec![100u8; 64]; // 8x8 constant
        let mut dst = vec![0u8; 16]; // 4x4 output
        interpolate_bicubic_scalar(&src, &mut dst, 8, 4, 4, 4, 0, 0)
            .expect("interpolation should succeed");
        for &v in &dst {
            assert_eq!(v, 100);
        }
    }

    #[test]
    fn test_8tap_identity_at_integer_position() {
        // With dx=0, dy=0 (phase 0), 8-tap should pass through centre tap
        let src = vec![128u8; 128]; // constant image
        let mut dst = vec![0u8; 16]; // 4x4
        interpolate_8tap_scalar(&src, &mut dst, 8, 4, 4, 4, 0, 0)
            .expect("interpolation should succeed");
        for &v in &dst {
            assert_eq!(v, 128);
        }
    }

    #[test]
    fn test_sad_identical() {
        let block = vec![100u8; 256];
        let result = sad_scalar(&block, &block, 16, 16, 16, 16).expect("SAD should succeed");
        assert_eq!(result, 0);
    }

    #[test]
    fn test_sad_known_value() {
        let a = vec![10u8; 4];
        let b = vec![20u8; 4];
        let result = sad_scalar(&a, &b, 2, 2, 2, 2).expect("SAD should succeed");
        assert_eq!(result, 40); // 4 pixels * 10 difference
    }
}
