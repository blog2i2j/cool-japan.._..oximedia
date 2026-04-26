//! SSE4.1-accelerated NV12 → RGB24 pixel format conversion.
//!
//! This module is compiled only on `x86_64` targets and **requires** the
//! `sse4.1` CPU feature to be present at runtime (guarded at the call site by
//! `is_x86_feature_detected!("sse4.1")`).
//!
//! # Safety invariant
//!
//! Every public function in this module is `unsafe` and carries
//! `#[target_feature(enable = "sse4.1")]`.  Callers must guarantee that the
//! CPU supports SSE4.1 before invoking any function herein.

#![allow(unsafe_code)]

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::simd_pixel::YuvCoeffs;

/// SSE4.1-accelerated NV12 → RGB24 conversion.
///
/// Processes the image in horizontal runs of 4 pixels at a time using 4×i32
/// SIMD lanes with `_mm_mullo_epi32` (SSE4.1).  Remaining columns (when width
/// is not a multiple of 4) and any rows when `height` is odd are handled by the
/// scalar kernel.
///
/// # Safety
///
/// Caller **must** ensure the CPU supports SSE4.1 (checked via
/// `is_x86_feature_detected!("sse4.1")` before calling).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
pub unsafe fn nv12_to_rgb24_sse41(
    y_plane: &[u8],
    uv_plane: &[u8],
    rgb_out: &mut [u8],
    width: usize,
    height: usize,
    coeffs: &YuvCoeffs,
) {
    // Splat coefficient registers once — reused every 4-pixel group.
    let y_scale_v = _mm_set1_epi32(coeffs.y_scale);
    let cr_to_r_v = _mm_set1_epi32(coeffs.cr_to_r);
    let cb_to_g_v = _mm_set1_epi32(coeffs.cb_to_g);
    let cr_to_g_v = _mm_set1_epi32(coeffs.cr_to_g);
    let cb_to_b_v = _mm_set1_epi32(coeffs.cb_to_b);
    let bias16_v = _mm_set1_epi32(16); // luma bias
    let bias128_v = _mm_set1_epi32(128); // chroma bias

    // Number of "full" 4-pixel groups per row.
    let cols4 = (width / 4) * 4; // == width when width is multiple of 4

    for row in 0..height {
        let uv_row_base = (row / 2) * width; // NV12: uv row stride == luma stride
        let y_row_base = row * width;
        let rgb_row_base = row * width * 3;

        let mut col = 0usize;
        while col < cols4 {
            // ── Load 4 Y samples ──────────────────────────────────────────────
            // Indices: y_row_base + col .. col+3
            let y0 = i32::from(y_plane[y_row_base + col]);
            let y1 = i32::from(y_plane[y_row_base + col + 1]);
            let y2 = i32::from(y_plane[y_row_base + col + 2]);
            let y3 = i32::from(y_plane[y_row_base + col + 3]);
            let y_vec = _mm_set_epi32(y3, y2, y1, y0);

            // ── Load 2 UV pairs (4 bytes: U0,V0,U1,V1) ───────────────────────
            // col is a multiple of 4 → (col & !1) == col.
            // Pair 0 serves pixels col, col+1; pair 1 serves col+2, col+3.
            let uv_idx0 = uv_row_base + col; // U0,V0
            let uv_idx1 = uv_row_base + col + 2; // U1,V1
            let u0 = i32::from(uv_plane[uv_idx0]);
            let v0 = i32::from(uv_plane[uv_idx0 + 1]);
            let u1 = i32::from(uv_plane[uv_idx1]);
            let v1 = i32::from(uv_plane[uv_idx1 + 1]);

            // Expand U/V to per-pixel vectors (lanes 0,1 share pair 0; lanes 2,3 share pair 1).
            let u_vec = _mm_set_epi32(u1, u1, u0, u0);
            let v_vec = _mm_set_epi32(v1, v1, v0, v0);

            // ── Fixed-point arithmetic matching scalar exactly ─────────────
            // y_scaled = (Y - 16) * y_scale
            let y_m16 = _mm_sub_epi32(y_vec, bias16_v);
            let y_scaled = _mm_mullo_epi32(y_m16, y_scale_v); // SSE4.1

            // cb = U - 128,  cr = V - 128
            let cb = _mm_sub_epi32(u_vec, bias128_v);
            let cr = _mm_sub_epi32(v_vec, bias128_v);

            // r = (y_scaled + cr_to_r * cr) >> 14
            let r_vec = _mm_srai_epi32(_mm_add_epi32(y_scaled, _mm_mullo_epi32(cr, cr_to_r_v)), 14);

            // g = (y_scaled + cb_to_g * cb + cr_to_g * cr) >> 14
            let g_vec = _mm_srai_epi32(
                _mm_add_epi32(
                    y_scaled,
                    _mm_add_epi32(
                        _mm_mullo_epi32(cb, cb_to_g_v),
                        _mm_mullo_epi32(cr, cr_to_g_v),
                    ),
                ),
                14,
            );

            // b = (y_scaled + cb_to_b * cb) >> 14
            let b_vec = _mm_srai_epi32(_mm_add_epi32(y_scaled, _mm_mullo_epi32(cb, cb_to_b_v)), 14);

            // ── Pack i32 → u8 with saturation to [0, 255] ─────────────────
            // _mm_packus_epi32(a, b) : i32→u16 saturate, 8 × u16
            // _mm_packus_epi16(a, b) : i16→u8  saturate, 16 × u8
            // We use the same register twice so the first pack gives 8 u16 in the low half.
            let r_u16 = _mm_packus_epi32(r_vec, r_vec); // SSE4.1
            let g_u16 = _mm_packus_epi32(g_vec, g_vec); // SSE4.1
            let b_u16 = _mm_packus_epi32(b_vec, b_vec); // SSE4.1

            let r_u8 = _mm_packus_epi16(r_u16, r_u16); // SSE2 — low 4 bytes valid
            let g_u8 = _mm_packus_epi16(g_u16, g_u16);
            let b_u8 = _mm_packus_epi16(b_u16, b_u16);

            // Extract 4 clamped bytes per channel.
            let r0 = _mm_extract_epi8(r_u8, 0) as u8;
            let r1 = _mm_extract_epi8(r_u8, 1) as u8;
            let r2 = _mm_extract_epi8(r_u8, 2) as u8;
            let r3 = _mm_extract_epi8(r_u8, 3) as u8;

            let g0 = _mm_extract_epi8(g_u8, 0) as u8;
            let g1 = _mm_extract_epi8(g_u8, 1) as u8;
            let g2 = _mm_extract_epi8(g_u8, 2) as u8;
            let g3 = _mm_extract_epi8(g_u8, 3) as u8;

            let b0 = _mm_extract_epi8(b_u8, 0) as u8;
            let b1 = _mm_extract_epi8(b_u8, 1) as u8;
            let b2 = _mm_extract_epi8(b_u8, 2) as u8;
            let b3 = _mm_extract_epi8(b_u8, 3) as u8;

            // ── Write interleaved RGB24 ────────────────────────────────────
            let out = rgb_row_base + col * 3;
            rgb_out[out] = r0;
            rgb_out[out + 1] = g0;
            rgb_out[out + 2] = b0;
            rgb_out[out + 3] = r1;
            rgb_out[out + 4] = g1;
            rgb_out[out + 5] = b1;
            rgb_out[out + 6] = r2;
            rgb_out[out + 7] = g2;
            rgb_out[out + 8] = b2;
            rgb_out[out + 9] = r3;
            rgb_out[out + 10] = g3;
            rgb_out[out + 11] = b3;

            col += 4;
        }

        // ── Scalar remainder (cols not covered by the 4-wide loop) ────────
        while col < width {
            nv12_pixel_scalar(
                y_plane,
                uv_plane,
                rgb_out,
                col,
                uv_row_base,
                y_row_base,
                rgb_row_base,
                coeffs,
            );
            col += 1;
        }
    }
}

/// Scalar conversion of a single NV12 pixel to RGB24.  Extracted so that both
/// the main SIMD loop and the per-row remainder share the same arithmetic.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn nv12_pixel_scalar(
    y_plane: &[u8],
    uv_plane: &[u8],
    rgb_out: &mut [u8],
    col: usize,
    uv_row_base: usize,
    y_row_base: usize,
    rgb_row_base: usize,
    coeffs: &YuvCoeffs,
) {
    let y_idx = y_row_base + col;
    let uv_idx = uv_row_base + (col & !1);
    let y_val = i32::from(y_plane[y_idx]);
    let u_val = i32::from(uv_plane[uv_idx]);
    let v_val = i32::from(uv_plane[uv_idx + 1]);

    let y_scaled = (y_val - 16) * coeffs.y_scale;
    let cb = u_val - 128;
    let cr = v_val - 128;

    let r = ((y_scaled + coeffs.cr_to_r * cr) >> 14).clamp(0, 255) as u8;
    let g = ((y_scaled + coeffs.cb_to_g * cb + coeffs.cr_to_g * cr) >> 14).clamp(0, 255) as u8;
    let b = ((y_scaled + coeffs.cb_to_b * cb) >> 14).clamp(0, 255) as u8;

    let out = rgb_row_base + col * 3;
    rgb_out[out] = r;
    rgb_out[out + 1] = g;
    rgb_out[out + 2] = b;
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[cfg(target_arch = "x86_64")]
mod tests {
    use super::*;
    use crate::convert::simd_pixel::{nv12_to_rgb24, SimdColorMatrix, YuvCoeffs};

    // Call the scalar path directly for ground truth in comparisons.
    fn scalar_nv12_to_rgb24(
        y_plane: &[u8],
        uv_plane: &[u8],
        width: usize,
        height: usize,
        coeffs: &YuvCoeffs,
    ) -> Vec<u8> {
        let mut rgb = vec![0u8; width * height * 3];
        for row in 0..height {
            let uv_row_base = (row / 2) * width;
            let y_row_base = row * width;
            let rgb_row_base = row * width * 3;
            for col in 0..width {
                let y_idx = y_row_base + col;
                let uv_idx = uv_row_base + (col & !1);
                let y_val = i32::from(y_plane[y_idx]);
                let u_val = i32::from(uv_plane[uv_idx]);
                let v_val = i32::from(uv_plane[uv_idx + 1]);

                let y_scaled = (y_val - 16) * coeffs.y_scale;
                let cb = u_val - 128;
                let cr = v_val - 128;

                let r = ((y_scaled + coeffs.cr_to_r * cr) >> 14).clamp(0, 255) as u8;
                let g = ((y_scaled + coeffs.cb_to_g * cb + coeffs.cr_to_g * cr) >> 14).clamp(0, 255)
                    as u8;
                let b = ((y_scaled + coeffs.cb_to_b * cb) >> 14).clamp(0, 255) as u8;

                let out = rgb_row_base + col * 3;
                rgb[out] = r;
                rgb[out + 1] = g;
                rgb[out + 2] = b;
            }
        }
        rgb
    }

    fn compare_within_1(label: &str, sse: &[u8], scalar: &[u8]) {
        assert_eq!(
            sse.len(),
            scalar.len(),
            "{label}: length mismatch {sse_len} vs {scalar_len}",
            sse_len = sse.len(),
            scalar_len = scalar.len()
        );
        for (i, (&a, &b)) in sse.iter().zip(scalar.iter()).enumerate() {
            let diff = (i32::from(a) - i32::from(b)).abs();
            assert!(
                diff <= 1,
                "{label}: byte {i} differs by {diff}: sse={a} scalar={b}"
            );
        }
    }

    fn run_sse41(
        y_plane: &[u8],
        uv_plane: &[u8],
        width: usize,
        height: usize,
        coeffs: &YuvCoeffs,
    ) -> Vec<u8> {
        let mut rgb = vec![0u8; width * height * 3];
        if is_x86_feature_detected!("sse4.1") {
            // SAFETY: sse4.1 confirmed above.
            unsafe {
                nv12_to_rgb24_sse41(y_plane, uv_plane, &mut rgb, width, height, coeffs);
            }
        } else {
            // Fall back to scalar so tests still compile/run on non-SSE4.1 hosts.
            let scalar = scalar_nv12_to_rgb24(y_plane, uv_plane, width, height, coeffs);
            rgb.copy_from_slice(&scalar);
        }
        rgb
    }

    // ── Test 1: uniform neutral gray ─────────────────────────────────────────

    #[test]
    fn test_nv12_sse41_vs_scalar_uniform() {
        let width = 8usize;
        let height = 4usize;
        let y_plane = vec![128u8; width * height];
        let uv_plane = vec![128u8; (width / 2) * (height / 2) * 2];
        let coeffs = YuvCoeffs::for_matrix(SimdColorMatrix::Bt709);
        let sse = run_sse41(&y_plane, &uv_plane, width, height, &coeffs);
        let scalar = scalar_nv12_to_rgb24(&y_plane, &uv_plane, width, height, &coeffs);
        compare_within_1("uniform_gray", &sse, &scalar);
    }

    // ── Test 2: Y gradient, neutral chroma ───────────────────────────────────

    #[test]
    fn test_nv12_sse41_vs_scalar_gradient() {
        let width = 8usize;
        let height = 4usize;
        let y_plane: Vec<u8> = (0..(width * height))
            .map(|i| (i * 7 % 220 + 16) as u8)
            .collect();
        let uv_plane = vec![128u8; (width / 2) * (height / 2) * 2];
        let coeffs = YuvCoeffs::for_matrix(SimdColorMatrix::Bt601);
        let sse = run_sse41(&y_plane, &uv_plane, width, height, &coeffs);
        let scalar = scalar_nv12_to_rgb24(&y_plane, &uv_plane, width, height, &coeffs);
        compare_within_1("gradient", &sse, &scalar);
    }

    // ── Test 3: 4×4 image ────────────────────────────────────────────────────

    #[test]
    fn test_nv12_sse41_vs_scalar_4x4() {
        let width = 4usize;
        let height = 4usize;
        let y_plane: Vec<u8> = (0..(width * height)).map(|i| (16 + i * 15) as u8).collect();
        let uv_size = width * height / 2; // NV12: uv plane is half the luma plane
        let uv_plane: Vec<u8> = (0..uv_size)
            .map(|i| 100u8.wrapping_add(i as u8 * 10))
            .collect();
        let coeffs = YuvCoeffs::for_matrix(SimdColorMatrix::Bt709);
        let sse = run_sse41(&y_plane, &uv_plane, width, height, &coeffs);
        let scalar = scalar_nv12_to_rgb24(&y_plane, &uv_plane, width, height, &coeffs);
        compare_within_1("4x4", &sse, &scalar);
    }

    // ── Test 4: 8×2 image ────────────────────────────────────────────────────

    #[test]
    fn test_nv12_sse41_vs_scalar_8x2() {
        let width = 8usize;
        let height = 2usize;
        // Varied Y and non-neutral chroma.
        let y_plane: Vec<u8> = (0..(width * height)).map(|i| (50 + i * 13) as u8).collect();
        let uv_plane: Vec<u8> = (0..((width / 2) * (height / 2) * 2))
            .map(|i| (100 + i * 17) as u8)
            .collect();
        let coeffs = YuvCoeffs::for_matrix(SimdColorMatrix::Bt2020);
        let sse = run_sse41(&y_plane, &uv_plane, width, height, &coeffs);
        let scalar = scalar_nv12_to_rgb24(&y_plane, &uv_plane, width, height, &coeffs);
        compare_within_1("8x2", &sse, &scalar);
    }

    // ── Test 5: 16×16 image ──────────────────────────────────────────────────

    #[test]
    fn test_nv12_sse41_vs_scalar_16x16() {
        let width = 16usize;
        let height = 16usize;
        let y_plane: Vec<u8> = (0..(width * height))
            .map(|i| (i % 220 + 16) as u8)
            .collect();
        let uv_plane: Vec<u8> = (0..((width / 2) * (height / 2) * 2))
            .map(|i| (64 + i * 5 % 128) as u8)
            .collect();
        let coeffs = YuvCoeffs::for_matrix(SimdColorMatrix::Bt601);
        let sse = run_sse41(&y_plane, &uv_plane, width, height, &coeffs);
        let scalar = scalar_nv12_to_rgb24(&y_plane, &uv_plane, width, height, &coeffs);
        compare_within_1("16x16", &sse, &scalar);
    }

    // ── Test 6: non-multiple-of-4 width (remainder columns exercised) ─────────

    #[test]
    fn test_nv12_sse41_vs_scalar_remainder_cols() {
        let width = 6usize; // 4+2 columns: one full SIMD group + 2 scalar remainder
        let height = 4usize;
        let y_plane: Vec<u8> = (0..(width * height)).map(|i| (30 + i * 11) as u8).collect();
        let uv_plane = vec![128u8; (width / 2) * (height / 2) * 2];
        let coeffs = YuvCoeffs::for_matrix(SimdColorMatrix::Bt709);
        let sse = run_sse41(&y_plane, &uv_plane, width, height, &coeffs);
        let scalar = scalar_nv12_to_rgb24(&y_plane, &uv_plane, width, height, &coeffs);
        compare_within_1("6-wide remainder", &sse, &scalar);
    }

    // ── Test 7: dispatch (nv12_to_rgb24 public fn uses SSE4.1 when available) ─

    #[test]
    fn test_nv12_to_rgb24_dispatch_matches_scalar() {
        let width = 8usize;
        let height = 4usize;
        let y_plane: Vec<u8> = (0..(width * height))
            .map(|i| (i % 220 + 16) as u8)
            .collect();
        let uv_plane: Vec<u8> = (0..((width / 2) * (height / 2) * 2))
            .map(|i| (80 + i * 3 % 160) as u8)
            .collect();
        let coeffs = YuvCoeffs::for_matrix(SimdColorMatrix::Bt709);
        let dispatched = nv12_to_rgb24(&y_plane, &uv_plane, width, height, SimdColorMatrix::Bt709);
        let scalar = scalar_nv12_to_rgb24(&y_plane, &uv_plane, width, height, &coeffs);
        compare_within_1("dispatch", &dispatched, &scalar);
    }
}
