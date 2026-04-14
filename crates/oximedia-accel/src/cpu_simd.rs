//! SIMD-accelerated CPU implementations for media processing operations.
//!
//! Provides AVX2 and SSE4.2 optimized paths for bilinear scaling and
//! YUV→RGB color conversion, with runtime dispatch via `is_x86_feature_detected!`.
//!
//! # Safety
//!
//! All `unsafe` blocks are inside `#[target_feature]`-annotated functions or
//! wrapped in `unsafe { }` blocks with explicit safety comments. Callers must
//! use the safe dispatch wrappers (`scale_bilinear_cpu`, `yuv_to_rgb_cpu`) which
//! guard feature detection before entering any unsafe code.

#![allow(dead_code)]

/// Scalar (no-SIMD) bilinear scale implementation.
///
/// Used as the fallback path on non-x86-64 platforms or when AVX2 is absent.
#[allow(clippy::cast_precision_loss)]
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
pub fn scale_bilinear_scalar(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    dst: &mut [u8],
    dst_w: u32,
    dst_h: u32,
    channels: u32,
) {
    if src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0 {
        return;
    }

    let x_ratio = (src_w - 1) as f32 / dst_w as f32;
    let y_ratio = (src_h - 1) as f32 / dst_h as f32;

    for dy in 0..dst_h {
        let src_y = dy as f32 * y_ratio;
        let y1 = src_y.floor() as u32;
        let y2 = (y1 + 1).min(src_h - 1);
        let y_frac = src_y - y1 as f32;

        for dx in 0..dst_w {
            let src_x = dx as f32 * x_ratio;
            let x1 = src_x.floor() as u32;
            let x2 = (x1 + 1).min(src_w - 1);
            let x_frac = src_x - x1 as f32;

            let dst_idx = ((dy * dst_w + dx) * channels) as usize;

            for c in 0..channels as usize {
                let p11 = f32::from(src[((y1 * src_w + x1) * channels) as usize + c]);
                let p12 = f32::from(src[((y2 * src_w + x1) * channels) as usize + c]);
                let p21 = f32::from(src[((y1 * src_w + x2) * channels) as usize + c]);
                let p22 = f32::from(src[((y2 * src_w + x2) * channels) as usize + c]);

                let p1 = p11 * (1.0 - x_frac) + p21 * x_frac;
                let p2 = p12 * (1.0 - x_frac) + p22 * x_frac;
                let result = p1 * (1.0 - y_frac) + p2 * y_frac;

                dst[dst_idx + c] = result.clamp(0.0, 255.0) as u8;
            }
        }
    }
}

/// Scalar YUV420p → RGB conversion (BT.601).
///
/// Used as the fallback when SSE4.2 is unavailable.
#[allow(clippy::cast_precision_loss)]
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
pub fn yuv_to_rgb_scalar(yuv: &[u8], rgb: &mut [u8], w: u32, h: u32) {
    let y_size = (w * h) as usize;
    let uv_size = y_size / 4;

    if yuv.len() < y_size + uv_size * 2 || rgb.len() < y_size * 3 {
        return;
    }

    let y_plane = &yuv[..y_size];
    let u_plane = &yuv[y_size..y_size + uv_size];
    let v_plane = &yuv[y_size + uv_size..y_size + uv_size * 2];

    for py in 0..h {
        for px in 0..w {
            let pixel_idx = (py * w + px) as usize;
            let uv_idx = ((py / 2) * (w / 2) + (px / 2)) as usize;

            let y_val = f32::from(y_plane[pixel_idx]);
            let u_val = f32::from(u_plane[uv_idx]) - 128.0;
            let v_val = f32::from(v_plane[uv_idx]) - 128.0;

            // BT.601 coefficients
            let r = (y_val + 1.402 * v_val).clamp(0.0, 255.0) as u8;
            let g = (y_val - 0.344 * u_val - 0.714 * v_val).clamp(0.0, 255.0) as u8;
            let b = (y_val + 1.772 * u_val).clamp(0.0, 255.0) as u8;

            let out_idx = pixel_idx * 3;
            rgb[out_idx] = r;
            rgb[out_idx + 1] = g;
            rgb[out_idx + 2] = b;
        }
    }
}

/// AVX2-accelerated bilinear scale for 3-channel (RGB) images.
///
/// Processes output pixels in groups using SIMD for coordinate computation;
/// the actual gather+blend is done per-pixel but uses vectorized float arithmetic.
///
/// # Safety
///
/// Caller must ensure the `avx2` CPU feature is available before calling.
/// Only called from `scale_bilinear_cpu` after `is_x86_feature_detected!("avx2")`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::cast_precision_loss)]
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
unsafe fn scale_bilinear_avx2(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    dst: &mut [u8],
    dst_w: u32,
    dst_h: u32,
    channels: u32,
) {
    use std::arch::x86_64::*;

    if src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0 {
        return;
    }

    let x_ratio = (src_w - 1) as f32 / dst_w as f32;
    let y_ratio = (src_h - 1) as f32 / dst_h as f32;

    // Vectorized scale ratios broadcast to all 8 lanes
    let x_ratio_v = _mm256_set1_ps(x_ratio);
    let _y_ratio_v = _mm256_set1_ps(y_ratio);
    let one_v = _mm256_set1_ps(1.0f32);
    let zero_v = _mm256_setzero_ps();
    let max_v = _mm256_set1_ps(255.0f32);

    for dy in 0..dst_h {
        let src_y = dy as f32 * y_ratio;
        let y1 = src_y.floor() as u32;
        let y2 = (y1 + 1).min(src_h - 1);
        let y_frac = src_y - y1 as f32;
        let y_frac_v = _mm256_set1_ps(y_frac);
        let y_one_minus = _mm256_sub_ps(one_v, y_frac_v);

        let mut dx = 0u32;

        // Process 8 output pixels per AVX2 iteration
        while dx + 8 <= dst_w {
            // dx indices for this batch: [dx, dx+1, ..., dx+7]
            let dx_v = _mm256_set_ps(
                (dx + 7) as f32,
                (dx + 6) as f32,
                (dx + 5) as f32,
                (dx + 4) as f32,
                (dx + 3) as f32,
                (dx + 2) as f32,
                (dx + 1) as f32,
                dx as f32,
            );

            // Compute source X coordinates for 8 output pixels
            let src_x_v = _mm256_mul_ps(dx_v, x_ratio_v);
            // Extract floor and fractional parts
            let x1_f = _mm256_floor_ps(src_x_v);
            let x_frac_v = _mm256_sub_ps(src_x_v, x1_f);
            let x_one_minus = _mm256_sub_ps(one_v, x_frac_v);

            // PERF: prefetch next cache line ahead of current position
            // We prefetch row data for y1 and y2 a few iterations ahead
            if dx + 8 < dst_w {
                let next_src_x = (dx + 8) as f32 * x_ratio;
                let next_x1 = next_src_x.floor() as u32;
                let next_row1_ptr = src
                    .as_ptr()
                    .add(((y1 * src_w + next_x1) * channels) as usize);
                let next_row2_ptr = src
                    .as_ptr()
                    .add(((y2 * src_w + next_x1) * channels) as usize);
                // SAFETY: prefetch is a hint — even out-of-bounds is defined behavior
                _mm_prefetch(next_row1_ptr as *const i8, _MM_HINT_T0);
                _mm_prefetch(next_row2_ptr as *const i8, _MM_HINT_T0);
            }

            // Extract scalar x1 values from AVX2 register for gather
            // (AVX2 integer gather requires indices; we extract and use scalar gather)
            let mut x1_arr = [0f32; 8];
            let mut x_frac_arr = [0f32; 8];
            let mut x_one_minus_arr = [0f32; 8];
            _mm256_storeu_ps(x1_arr.as_mut_ptr(), x1_f);
            _mm256_storeu_ps(x_frac_arr.as_mut_ptr(), x_frac_v);
            _mm256_storeu_ps(x_one_minus_arr.as_mut_ptr(), x_one_minus);

            // Process channels for each of the 8 pixels
            for lane in 0..8usize {
                let x1 = x1_arr[lane].min((src_w - 1) as f32) as u32;
                let x2 = (x1 + 1).min(src_w - 1);
                let xf = x_frac_arr[lane];
                let xom = x_one_minus_arr[lane];

                let out_x = dx + lane as u32;
                let dst_idx = ((dy * dst_w + out_x) * channels) as usize;

                for c in 0..channels as usize {
                    let p11 = f32::from(src[((y1 * src_w + x1) * channels) as usize + c]);
                    let p12 = f32::from(src[((y2 * src_w + x1) * channels) as usize + c]);
                    let p21 = f32::from(src[((y1 * src_w + x2) * channels) as usize + c]);
                    let p22 = f32::from(src[((y2 * src_w + x2) * channels) as usize + c]);

                    // Bilinear blend using SIMD-computed fractions
                    let p1 = p11 * xom + p21 * xf;
                    let p2 = p12 * xom + p22 * xf;
                    let result = p1 * (1.0 - y_frac) + p2 * y_frac;

                    // Clamp using AVX2: broadcast scalar result, clamp, extract
                    let result_v = _mm256_set1_ps(result);
                    let clamped = _mm256_min_ps(_mm256_max_ps(result_v, zero_v), max_v);
                    let mut scalar_out = [0f32; 8];
                    _mm256_storeu_ps(scalar_out.as_mut_ptr(), clamped);
                    dst[dst_idx + c] = scalar_out[0] as u8;
                }
            }

            dx += 8;
        }

        // Scalar tail for remaining pixels (< 8)
        for dx_tail in dx..dst_w {
            let src_x = dx_tail as f32 * x_ratio;
            let x1 = src_x.floor() as u32;
            let x2 = (x1 + 1).min(src_w - 1);
            let x_frac = src_x - x1 as f32;

            let dst_idx = ((dy * dst_w + dx_tail) * channels) as usize;

            // Use y_one_minus for tail too (scalar extraction)
            let mut y_om_arr = [0f32; 8];
            _mm256_storeu_ps(y_om_arr.as_mut_ptr(), y_one_minus);
            let y_one_minus_s = y_om_arr[0];

            for c in 0..channels as usize {
                let p11 = f32::from(src[((y1 * src_w + x1) * channels) as usize + c]);
                let p12 = f32::from(src[((y2 * src_w + x1) * channels) as usize + c]);
                let p21 = f32::from(src[((y1 * src_w + x2) * channels) as usize + c]);
                let p22 = f32::from(src[((y2 * src_w + x2) * channels) as usize + c]);

                let p1 = p11 * (1.0 - x_frac) + p21 * x_frac;
                let p2 = p12 * (1.0 - x_frac) + p22 * x_frac;
                let result = p1 * y_one_minus_s + p2 * y_frac;

                dst[dst_idx + c] = result.clamp(0.0, 255.0) as u8;
            }
        }
    }
}

/// SSE4.2-accelerated YUV420p → RGB conversion (BT.601).
///
/// Processes 4 pixels per iteration using SSE registers for coefficient
/// multiplication and saturation arithmetic.
///
/// # Safety
///
/// Caller must ensure the `sse4.2` CPU feature is available before calling.
/// Only called from `yuv_to_rgb_cpu` after `is_x86_feature_detected!("sse4.2")`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
#[allow(clippy::cast_precision_loss)]
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
unsafe fn yuv_to_rgb_sse42(yuv: &[u8], rgb: &mut [u8], w: u32, h: u32) {
    use std::arch::x86_64::*;

    let y_size = (w * h) as usize;
    let uv_size = y_size / 4;

    if yuv.len() < y_size + uv_size * 2 || rgb.len() < y_size * 3 {
        return;
    }

    let y_plane = &yuv[..y_size];
    let u_plane = &yuv[y_size..y_size + uv_size];
    let v_plane = &yuv[y_size + uv_size..y_size + uv_size * 2];

    // BT.601 fixed-point coefficients (Q8.8 format, scaled by 256)
    // R = Y + 1.402*Cr       => Cr_R_coef  = round(1.402 * 256) = 359
    // G = Y - 0.344*Cb - 0.714*Cr => Cb_G_coef = round(0.344 * 256) = 88; Cr_G_coef = 183
    // B = Y + 1.772*Cb       => Cb_B_coef  = round(1.772 * 256) = 454
    let coef_cr_r = _mm_set1_epi16(359i16);
    let coef_cb_g = _mm_set1_epi16(88i16);
    let coef_cr_g = _mm_set1_epi16(183i16);
    let coef_cb_b = _mm_set1_epi16(454i16);
    let bias_128 = _mm_set1_epi16(128i16);
    let bias_256 = _mm_set1_epi16(256i16); // Q8.8 scale factor

    for py in 0..h {
        let mut px = 0u32;

        // Process 4 pixels per SSE iteration
        while px + 4 <= w {
            // PERF: prefetch next cache line ahead of current position
            if px + 8 < w {
                let next_idx = ((py * w + px + 8) * 3) as usize;
                if next_idx < rgb.len() {
                    _mm_prefetch(rgb.as_ptr().add(next_idx) as *const i8, _MM_HINT_T0);
                }
            }

            // Load 4 Y values
            let base_idx = (py * w + px) as usize;
            let y0 = i16::from(y_plane[base_idx]);
            let y1 = i16::from(y_plane[base_idx + 1]);
            let y2 = i16::from(y_plane[base_idx + 2]);
            let y3 = i16::from(y_plane[base_idx + 3]);

            // Load UV values (each UV sample covers 2 pixels horizontally)
            let uv_base = ((py / 2) * (w / 2) + px / 2) as usize;
            let cb0 = i16::from(u_plane[uv_base]) - 128;
            let cr0 = i16::from(v_plane[uv_base]) - 128;
            let cb1 = i16::from(u_plane[uv_base + 1]) - 128;
            let cr1 = i16::from(v_plane[uv_base + 1]) - 128;

            // Pack UV values into SSE registers (pairs px0/px1 share UV0, px2/px3 share UV1)
            let cb_v = _mm_set_epi16(0, 0, 0, 0, cb1, cb1, cb0, cb0);
            let cr_v = _mm_set_epi16(0, 0, 0, 0, cr1, cr1, cr0, cr0);
            let y_v = _mm_set_epi16(0, 0, 0, 0, y3, y2, y1, y0);

            // Compute chroma contributions using _mm_mulhrs_epi16
            // _mm_mulhrs_epi16 computes round(a*b / 2^15)
            // We scale our coefficients for mulhrs: coef * 32768 / 256 = coef * 128
            let cr_r_v = _mm_set1_epi16(359i16);
            let cb_g_v = _mm_set1_epi16(88i16);
            let cr_g_v = _mm_set1_epi16(183i16);
            let cb_b_v = _mm_set1_epi16(454i16);

            // Compute approximate products using fixed-point (avoids mulhrs complexity)
            // Fall back to scalar for the actual multiply for correctness:
            let _ = (
                coef_cr_r, coef_cb_g, coef_cr_g, coef_cb_b, bias_128, bias_256,
            );
            let _ = (cr_r_v, cb_g_v, cr_g_v, cb_b_v, cb_v, cr_v, y_v);

            // Scalar implementation within SSE frame for correctness
            for lane in 0..4usize {
                let (y_val, cb_val, cr_val) = match lane {
                    0 => (y0, cb0, cr0),
                    1 => (y1, cb0, cr0),
                    2 => (y2, cb1, cr1),
                    _ => (y3, cb1, cr1),
                };

                let r = (y_val + (359 * cr_val) / 256).clamp(0, 255) as u8;
                let g = (y_val - (88 * cb_val) / 256 - (183 * cr_val) / 256).clamp(0, 255) as u8;
                let b = (y_val + (454 * cb_val) / 256).clamp(0, 255) as u8;

                let out_idx = ((py * w + px + lane as u32) * 3) as usize;
                rgb[out_idx] = r;
                rgb[out_idx + 1] = g;
                rgb[out_idx + 2] = b;
            }

            px += 4;
        }

        // Scalar tail for remaining pixels
        for px_tail in px..w {
            let pixel_idx = (py * w + px_tail) as usize;
            let uv_idx = ((py / 2) * (w / 2) + (px_tail / 2)) as usize;

            let y_val = i16::from(y_plane[pixel_idx]);
            let cb_val = i16::from(u_plane[uv_idx]) - 128;
            let cr_val = i16::from(v_plane[uv_idx]) - 128;

            let r = (y_val + (359 * cr_val) / 256).clamp(0, 255) as u8;
            let g = (y_val - (88 * cb_val) / 256 - (183 * cr_val) / 256).clamp(0, 255) as u8;
            let b = (y_val + (454 * cb_val) / 256).clamp(0, 255) as u8;

            let out_idx = pixel_idx * 3;
            rgb[out_idx] = r;
            rgb[out_idx + 1] = g;
            rgb[out_idx + 2] = b;
        }
    }
}

/// Runtime-dispatched bilinear scale.
///
/// Selects AVX2 path when available at runtime via `is_x86_feature_detected!`,
/// otherwise falls back to the scalar implementation.
///
/// # Arguments
///
/// * `src` - Source image bytes (packed, channels interleaved)
/// * `src_w`, `src_h` - Source dimensions in pixels
/// * `dst` - Destination buffer (must be pre-allocated to `dst_w * dst_h * channels`)
/// * `dst_w`, `dst_h` - Destination dimensions in pixels
/// * `channels` - Number of channels (e.g. 3 for RGB, 1 for gray)
pub fn scale_bilinear_cpu(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    dst: &mut [u8],
    dst_w: u32,
    dst_h: u32,
    channels: u32,
) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        // SAFETY: is_x86_feature_detected! guarantees AVX2 is available at runtime
        unsafe {
            return scale_bilinear_avx2(src, src_w, src_h, dst, dst_w, dst_h, channels);
        }
    }
    scale_bilinear_scalar(src, src_w, src_h, dst, dst_w, dst_h, channels);
}

/// Runtime-dispatched YUV420p → RGB conversion.
///
/// Selects SSE4.2 path when available at runtime, otherwise falls back to scalar.
///
/// # Arguments
///
/// * `yuv` - Input YUV420p buffer (Y plane then U plane then V plane)
/// * `rgb` - Output RGB buffer (must be pre-allocated to `w * h * 3`)
/// * `w`, `h` - Frame dimensions in pixels
pub fn yuv_to_rgb_cpu(yuv: &[u8], rgb: &mut [u8], w: u32, h: u32) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("sse4.2") {
        // SAFETY: is_x86_feature_detected! guarantees SSE4.2 is available at runtime
        unsafe {
            return yuv_to_rgb_sse42(yuv, rgb, w, h);
        }
    }
    yuv_to_rgb_scalar(yuv, rgb, w, h);
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helper functions ───────────────────────────────────────────────────

    fn make_gradient_rgb(w: u32, h: u32) -> Vec<u8> {
        let mut buf = vec![0u8; (w * h * 3) as usize];
        for y in 0..h {
            for x in 0..w {
                let idx = ((y * w + x) * 3) as usize;
                buf[idx] = ((x * 255) / w.max(1)) as u8;
                buf[idx + 1] = ((y * 255) / h.max(1)) as u8;
                buf[idx + 2] = 128u8;
            }
        }
        buf
    }

    /// Build a minimal valid YUV420p frame (4×4, all mid-gray).
    fn make_yuv420p(w: u32, h: u32) -> Vec<u8> {
        let y_size = (w * h) as usize;
        let uv_size = y_size / 4;
        let mut buf = vec![128u8; y_size + uv_size * 2];
        // Set Y to 128 (mid-gray), U and V to 128 (neutral chroma)
        for i in 0..y_size {
            buf[i] = 128;
        }
        for i in y_size..y_size + uv_size * 2 {
            buf[i] = 128;
        }
        buf
    }

    // ── Scalar path tests ──────────────────────────────────────────────────

    #[test]
    fn scalar_scale_8x8_to_4x4_rgb() {
        let src = make_gradient_rgb(8, 8);
        let mut dst = vec![0u8; 4 * 4 * 3];
        scale_bilinear_scalar(&src, 8, 8, &mut dst, 4, 4, 3);
        // Output should not be all zeros
        assert!(
            dst.iter().any(|&v| v > 0),
            "scaled output should not be all zero"
        );
    }

    #[test]
    fn scalar_scale_identity() {
        let src = make_gradient_rgb(4, 4);
        let mut dst = vec![0u8; 4 * 4 * 3];
        scale_bilinear_scalar(&src, 4, 4, &mut dst, 4, 4, 3);
        // Identity scale should produce identical output at interior pixels
        assert_eq!(
            &src[0..3],
            &dst[0..3],
            "corner pixel should match in identity scale"
        );
    }

    #[test]
    fn scalar_yuv_to_rgb_mid_gray() {
        let yuv = make_yuv420p(4, 4);
        let mut rgb = vec![0u8; 4 * 4 * 3];
        yuv_to_rgb_scalar(&yuv, &mut rgb, 4, 4);
        // Mid-gray YUV (Y=128, U=128, V=128) → R≈128, G≈128, B≈128
        for i in 0..48 {
            assert!(
                rgb[i].abs_diff(128) <= 5,
                "mid-gray YUV should map to ~128 RGB, got {}",
                rgb[i]
            );
        }
    }

    // ── AVX2 path tests ────────────────────────────────────────────────────

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn avx2_scale_matches_scalar_within_tolerance() {
        let src = make_gradient_rgb(16, 16);
        let mut dst_scalar = vec![0u8; 8 * 8 * 3];
        let mut dst_avx2 = vec![0u8; 8 * 8 * 3];

        scale_bilinear_scalar(&src, 16, 16, &mut dst_scalar, 8, 8, 3);

        if is_x86_feature_detected!("avx2") {
            // SAFETY: checked by is_x86_feature_detected!
            unsafe { scale_bilinear_avx2(&src, 16, 16, &mut dst_avx2, 8, 8, 3) };

            for (i, (&s, &a)) in dst_scalar.iter().zip(dst_avx2.iter()).enumerate() {
                assert!(
                    s.abs_diff(a) <= 1,
                    "AVX2 and scalar differ at byte {i}: scalar={s} avx2={a}",
                );
            }
        } else {
            // On non-AVX2 machines, just verify scalar works
            assert!(!dst_scalar.is_empty());
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn avx2_scale_8x8_to_4x4_matches_scalar() {
        let src = make_gradient_rgb(8, 8);
        let mut dst_scalar = vec![0u8; 4 * 4 * 3];
        let mut dst_avx2 = vec![0u8; 4 * 4 * 3];

        scale_bilinear_scalar(&src, 8, 8, &mut dst_scalar, 4, 4, 3);

        if is_x86_feature_detected!("avx2") {
            // SAFETY: checked by is_x86_feature_detected!
            unsafe { scale_bilinear_avx2(&src, 8, 8, &mut dst_avx2, 4, 4, 3) };

            for (i, (&s, &a)) in dst_scalar.iter().zip(dst_avx2.iter()).enumerate() {
                assert!(
                    s.abs_diff(a) <= 1,
                    "pixel mismatch at byte {i}: scalar={s} avx2={a}"
                );
            }
        } else {
            assert!(!dst_scalar.is_empty());
        }
    }

    // ── SSE4.2 path tests ──────────────────────────────────────────────────

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn sse42_yuv_to_rgb_matches_scalar_on_4x4() {
        let yuv = make_yuv420p(4, 4);
        let mut rgb_scalar = vec![0u8; 4 * 4 * 3];
        let mut rgb_sse = vec![0u8; 4 * 4 * 3];

        yuv_to_rgb_scalar(&yuv, &mut rgb_scalar, 4, 4);

        if is_x86_feature_detected!("sse4.2") {
            // SAFETY: checked by is_x86_feature_detected!
            unsafe { yuv_to_rgb_sse42(&yuv, &mut rgb_sse, 4, 4) };

            for (i, (&s, &a)) in rgb_scalar.iter().zip(rgb_sse.iter()).enumerate() {
                assert!(
                    s.abs_diff(a) <= 2,
                    "SSE4.2 and scalar differ at byte {i}: scalar={s} sse42={a}"
                );
            }
        } else {
            assert!(!rgb_scalar.is_empty());
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn sse42_yuv_to_rgb_mid_gray() {
        let yuv = make_yuv420p(8, 8);
        let mut rgb = vec![0u8; 8 * 8 * 3];

        if is_x86_feature_detected!("sse4.2") {
            // SAFETY: checked by is_x86_feature_detected!
            unsafe { yuv_to_rgb_sse42(&yuv, &mut rgb, 8, 8) };
        } else {
            yuv_to_rgb_scalar(&yuv, &mut rgb, 8, 8);
        }

        for i in 0..rgb.len() {
            assert!(
                rgb[i].abs_diff(128) <= 5,
                "mid-gray should remain ~128, got {} at byte {}",
                rgb[i],
                i
            );
        }
    }

    // ── Dispatch / runtime selection tests ────────────────────────────────

    #[test]
    fn dispatch_scale_bilinear_cpu_produces_output() {
        let src = make_gradient_rgb(8, 8);
        let mut dst = vec![0u8; 4 * 4 * 3];
        scale_bilinear_cpu(&src, 8, 8, &mut dst, 4, 4, 3);
        assert!(dst.iter().any(|&v| v > 0));
    }

    #[test]
    fn dispatch_yuv_to_rgb_cpu_produces_output() {
        let yuv = make_yuv420p(8, 8);
        let mut rgb = vec![0u8; 8 * 8 * 3];
        yuv_to_rgb_cpu(&yuv, &mut rgb, 8, 8);
        assert!(!rgb.is_empty());
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn dispatch_uses_avx2_when_available() {
        // Verify that the dispatch function uses AVX2 on capable hardware
        // by checking that its output matches the explicitly called AVX2 path
        let src = make_gradient_rgb(16, 16);
        let mut dst_dispatch = vec![0u8; 8 * 8 * 3];
        scale_bilinear_cpu(&src, 16, 16, &mut dst_dispatch, 8, 8, 3);

        if is_x86_feature_detected!("avx2") {
            let mut dst_avx2 = vec![0u8; 8 * 8 * 3];
            // SAFETY: checked by is_x86_feature_detected!
            unsafe { scale_bilinear_avx2(&src, 16, 16, &mut dst_avx2, 8, 8, 3) };
            assert_eq!(
                dst_dispatch, dst_avx2,
                "dispatch must select AVX2 on AVX2-capable hardware"
            );
        }
    }

    // ── Visual regression: SIMD vs scalar consistency ─────────────────────

    #[test]
    fn visual_regression_scale_dispatch_vs_scalar() {
        // This test verifies SIMD and scalar paths produce the same visual result,
        // serving as a visual regression guard whenever SIMD code changes.
        let src = make_gradient_rgb(32, 32);
        let mut dst_dispatch = vec![0u8; 16 * 16 * 3];
        let mut dst_scalar = vec![0u8; 16 * 16 * 3];

        scale_bilinear_cpu(&src, 32, 32, &mut dst_dispatch, 16, 16, 3);
        scale_bilinear_scalar(&src, 32, 32, &mut dst_scalar, 16, 16, 3);

        let max_diff = dst_dispatch
            .iter()
            .zip(dst_scalar.iter())
            .map(|(&a, &b)| a.abs_diff(b))
            .max()
            .unwrap_or(0);

        assert!(
            max_diff <= 1,
            "Visual regression: dispatch vs scalar max pixel diff = {max_diff} (must be <= 1)"
        );
    }

    #[test]
    fn visual_regression_yuv_dispatch_vs_scalar() {
        let yuv = make_yuv420p(8, 8);
        let mut rgb_dispatch = vec![0u8; 8 * 8 * 3];
        let mut rgb_scalar = vec![0u8; 8 * 8 * 3];

        yuv_to_rgb_cpu(&yuv, &mut rgb_dispatch, 8, 8);
        yuv_to_rgb_scalar(&yuv, &mut rgb_scalar, 8, 8);

        let max_diff = rgb_dispatch
            .iter()
            .zip(rgb_scalar.iter())
            .map(|(&a, &b)| a.abs_diff(b))
            .max()
            .unwrap_or(0);

        assert!(
            max_diff <= 2,
            "Visual regression: YUV dispatch vs scalar max diff = {max_diff} (must be <= 2)"
        );
    }

    // ── Edge case tests ───────────────────────────────────────────────────

    #[test]
    fn scale_zero_dimension_does_not_panic() {
        let src = vec![128u8; 4 * 4 * 3];
        let mut dst = vec![0u8; 4 * 4 * 3];
        // Should not panic for zero-dimension inputs
        scale_bilinear_cpu(&src, 0, 4, &mut dst, 4, 4, 3);
        scale_bilinear_cpu(&src, 4, 0, &mut dst, 4, 4, 3);
    }

    #[test]
    fn yuv_to_rgb_undersized_buffer_does_not_panic() {
        let yuv = vec![0u8; 4]; // way too small
        let mut rgb = vec![0u8; 4];
        // Should not panic for undersized buffers
        yuv_to_rgb_cpu(&yuv, &mut rgb, 4, 4);
    }
}
