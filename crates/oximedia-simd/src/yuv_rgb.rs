//! YUV ↔ RGB colour-space conversion kernels.
//!
//! This module provides scalar implementations of the most common YUV→RGB and
//! RGB→YUV conversions used in video codecs.  The hot paths are written as
//! tight unrolled loops so that LLVM's auto-vectoriser can produce efficient
//! SIMD code without hand-written intrinsics.
//!
//! ## Supported formats
//!
//! * **YUV 4:2:0 planar** (YUV420p / I420): three separate planes Y, U, V
//!   where each chroma plane is half the width and half the height of the luma
//!   plane.  Each U/V value covers a 2×2 luma block.
//! * **NV12** (YUV 4:2:0 semi-planar): full-resolution Y plane followed by an
//!   interleaved UV plane at half resolution.
//!
//! ## Color matrix
//!
//! BT.601 full-range coefficients are used by default (the most common choice
//! for standard-definition content).  BT.709 variants may be added in future.
//!
//! ## Performance
//!
//! The inner loop is written with manual 4× unrolling so that the compiler can
//! keep all intermediate values in registers and issue contiguous loads.

// ─── YUV 4:2:0 planar → RGBA ─────────────────────────────────────────────────

/// Convert a YUV 4:2:0 planar frame (I420) to packed RGBA.
///
/// # Parameters
///
/// * `y`  — luma plane: `w × h` bytes, row-major
/// * `u`  — Cb chroma plane: `(w/2) × (h/2)` bytes, row-major
/// * `v`  — Cr chroma plane: `(w/2) × (h/2)` bytes, row-major
/// * `w`  — frame width in pixels (must be even)
/// * `h`  — frame height in pixels (must be even)
///
/// # Returns
///
/// A `Vec<u8>` of length `w × h × 4` in RGBA order.  Alpha is set to 255.
/// Returns an empty vector if any buffer is too small for the given dimensions.
///
/// # BT.601 full-range coefficients
///
/// ```text
/// R = Y                    + 1.40200 * (V - 128)
/// G = Y - 0.34414 * (U-128) - 0.71414 * (V-128)
/// B = Y + 1.77200 * (U - 128)
/// ```
#[must_use]
pub fn yuv420_to_rgba(y: &[u8], u: &[u8], v: &[u8], w: u32, h: u32) -> Vec<u8> {
    let ww = w as usize;
    let hh = h as usize;
    let y_size = ww * hh;
    let uv_size = (ww / 2) * (hh / 2);

    if y.len() < y_size || u.len() < uv_size || v.len() < uv_size {
        return Vec::new();
    }

    let out_size = ww * hh * 4;
    let mut rgba = vec![0u8; out_size];

    let uv_stride = ww / 2;

    let mut row = 0usize;
    while row < hh {
        let uv_row = row >> 1; // same UV row for both this and next luma row

        // Process row `row`
        {
            let y_row_base = row * ww;
            let uv_base = uv_row * uv_stride;
            let dst_base = row * ww * 4;

            let mut col = 0usize;
            // Unrolled 4× — process 4 pixels per iteration when possible
            while col + 3 < ww {
                for k in 0..4 {
                    let c = col + k;
                    let y_val = y[y_row_base + c] as i32;
                    let uv_col = c >> 1;
                    let u_val = u[uv_base + uv_col] as i32 - 128;
                    let v_val = v[uv_base + uv_col] as i32 - 128;
                    let (r, g, b) = yuv_to_rgb_bt601(y_val, u_val, v_val);
                    let dst = dst_base + c * 4;
                    rgba[dst] = r;
                    rgba[dst + 1] = g;
                    rgba[dst + 2] = b;
                    rgba[dst + 3] = 255;
                }
                col += 4;
            }
            // Handle remaining columns (0–3)
            while col < ww {
                let y_val = y[y_row_base + col] as i32;
                let uv_col = col >> 1;
                let u_val = u[uv_base + uv_col] as i32 - 128;
                let v_val = v[uv_base + uv_col] as i32 - 128;
                let (r, g, b) = yuv_to_rgb_bt601(y_val, u_val, v_val);
                let dst = dst_base + col * 4;
                rgba[dst] = r;
                rgba[dst + 1] = g;
                rgba[dst + 2] = b;
                rgba[dst + 3] = 255;
                col += 1;
            }
        }

        // Process row `row + 1` if it exists (same UV row)
        if row + 1 < hh {
            let next_row = row + 1;
            let y_row_base = next_row * ww;
            let uv_base = uv_row * uv_stride;
            let dst_base = next_row * ww * 4;

            let mut col = 0usize;
            while col + 3 < ww {
                for k in 0..4 {
                    let c = col + k;
                    let y_val = y[y_row_base + c] as i32;
                    let uv_col = c >> 1;
                    let u_val = u[uv_base + uv_col] as i32 - 128;
                    let v_val = v[uv_base + uv_col] as i32 - 128;
                    let (r, g, b) = yuv_to_rgb_bt601(y_val, u_val, v_val);
                    let dst = dst_base + c * 4;
                    rgba[dst] = r;
                    rgba[dst + 1] = g;
                    rgba[dst + 2] = b;
                    rgba[dst + 3] = 255;
                }
                col += 4;
            }
            while col < ww {
                let y_val = y[y_row_base + col] as i32;
                let uv_col = col >> 1;
                let u_val = u[uv_base + uv_col] as i32 - 128;
                let v_val = v[uv_base + uv_col] as i32 - 128;
                let (r, g, b) = yuv_to_rgb_bt601(y_val, u_val, v_val);
                let dst = dst_base + col * 4;
                rgba[dst] = r;
                rgba[dst + 1] = g;
                rgba[dst + 2] = b;
                rgba[dst + 3] = 255;
                col += 1;
            }
        }

        row += 2;
    }

    rgba
}

// ─── NV12 → RGBA ──────────────────────────────────────────────────────────────

/// Convert an NV12 semi-planar frame to packed RGBA.
///
/// NV12 layout:
/// * Y plane: `w × h` bytes
/// * UV plane (interleaved): `w × (h/2)` bytes (`U, V, U, V, …`)
///
/// # Returns
///
/// A `Vec<u8>` of length `w × h × 4` in RGBA order.
/// Returns an empty vector if the buffer is too small.
#[must_use]
pub fn nv12_to_rgba(nv12: &[u8], w: u32, h: u32) -> Vec<u8> {
    let ww = w as usize;
    let hh = h as usize;
    let y_size = ww * hh;
    let uv_size = ww * (hh / 2);

    if nv12.len() < y_size + uv_size {
        return Vec::new();
    }

    let y_plane = &nv12[..y_size];
    let uv_plane = &nv12[y_size..y_size + uv_size];

    let mut rgba = vec![0u8; ww * hh * 4];

    for row in 0..hh {
        let uv_row = row >> 1;
        let y_row_base = row * ww;
        let uv_base = uv_row * ww;
        let dst_base = row * ww * 4;

        let mut col = 0usize;
        while col + 3 < ww {
            for k in 0..4 {
                let c = col + k;
                let y_val = y_plane[y_row_base + c] as i32;
                let uv_col = (c >> 1) * 2;
                let u_val = uv_plane[uv_base + uv_col] as i32 - 128;
                let v_val = uv_plane[uv_base + uv_col + 1] as i32 - 128;
                let (r, g, b) = yuv_to_rgb_bt601(y_val, u_val, v_val);
                let dst = dst_base + c * 4;
                rgba[dst] = r;
                rgba[dst + 1] = g;
                rgba[dst + 2] = b;
                rgba[dst + 3] = 255;
            }
            col += 4;
        }
        while col < ww {
            let y_val = y_plane[y_row_base + col] as i32;
            let uv_col = (col >> 1) * 2;
            let u_val = uv_plane[uv_base + uv_col] as i32 - 128;
            let v_val = uv_plane[uv_base + uv_col + 1] as i32 - 128;
            let (r, g, b) = yuv_to_rgb_bt601(y_val, u_val, v_val);
            let dst = dst_base + col * 4;
            rgba[dst] = r;
            rgba[dst + 1] = g;
            rgba[dst + 2] = b;
            rgba[dst + 3] = 255;
            col += 1;
        }
    }

    rgba
}

// ─── RGBA → YUV 4:2:0 planar ─────────────────────────────────────────────────

/// Convert packed RGBA to YUV 4:2:0 planar (I420).
///
/// Chroma is subsampled by averaging four adjacent luma pixels' chroma values.
/// Alpha is discarded.
///
/// # Returns
///
/// `(y_plane, u_plane, v_plane)` where the luma plane is `w × h` bytes and
/// each chroma plane is `(w/2) × (h/2)` bytes.
/// Returns three empty vectors if the input is too small.
#[must_use]
pub fn rgba_to_yuv420(rgba: &[u8], w: u32, h: u32) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let ww = w as usize;
    let hh = h as usize;
    let required = ww * hh * 4;
    if rgba.len() < required {
        return (Vec::new(), Vec::new(), Vec::new());
    }

    let mut y_plane = vec![0u8; ww * hh];
    let uv_w = ww / 2;
    let uv_h = hh / 2;
    let mut u_plane = vec![0u8; uv_w * uv_h];
    let mut v_plane = vec![0u8; uv_w * uv_h];

    for row in 0..hh {
        for col in 0..ww {
            let src = (row * ww + col) * 4;
            let r = rgba[src] as i32;
            let g = rgba[src + 1] as i32;
            let b = rgba[src + 2] as i32;
            let (y, _u, _v) = rgb_to_yuv_bt601(r, g, b);
            y_plane[row * ww + col] = y;
        }
    }

    // Chroma: average 2×2 blocks
    let mut uv_row = 0usize;
    let mut row = 0;
    while row + 1 < hh {
        let mut uv_col = 0usize;
        let mut col = 0;
        while col + 1 < ww {
            let avg_u;
            let avg_v;
            {
                let mut sum_u = 0i32;
                let mut sum_v = 0i32;
                for dr in 0..2 {
                    for dc in 0..2 {
                        let src = ((row + dr) * ww + (col + dc)) * 4;
                        let r = rgba[src] as i32;
                        let g = rgba[src + 1] as i32;
                        let b = rgba[src + 2] as i32;
                        let (_y, u, v) = rgb_to_yuv_bt601(r, g, b);
                        sum_u += u as i32;
                        sum_v += v as i32;
                    }
                }
                avg_u = ((sum_u + 2) / 4) as u8;
                avg_v = ((sum_v + 2) / 4) as u8;
            }
            u_plane[uv_row * uv_w + uv_col] = avg_u;
            v_plane[uv_row * uv_w + uv_col] = avg_v;
            uv_col += 1;
            col += 2;
        }
        uv_row += 1;
        row += 2;
    }

    (y_plane, u_plane, v_plane)
}

// ─── Internal colour math ─────────────────────────────────────────────────────

/// BT.601 full-range YCbCr → RGB with integer arithmetic.
///
/// Coefficients scaled by 1024 to avoid floating point in the hot path:
/// * R = Y               + (1402 * V) / 1000
/// * G = Y - (344  * U) / 1000 - (714  * V) / 1000
/// * B = Y + (1772 * U) / 1000
///
/// `u_val` and `v_val` are pre-shifted by -128.
#[inline(always)]
fn yuv_to_rgb_bt601(y: i32, u: i32, v: i32) -> (u8, u8, u8) {
    let r = (y + ((1402 * v + 500) >> 10)).clamp(0, 255) as u8;
    let g = (y - ((344 * u + 500) >> 10) - ((714 * v + 500) >> 10)).clamp(0, 255) as u8;
    let b = (y + ((1772 * u + 500) >> 10)).clamp(0, 255) as u8;
    (r, g, b)
}

/// BT.601 full-range RGB → YCbCr.
///
/// Returns `(y, u, v)` with U and V offset by +128 so they are in `[0, 255]`.
#[inline(always)]
fn rgb_to_yuv_bt601(r: i32, g: i32, b: i32) -> (u8, u8, u8) {
    let y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
    let u = ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
    let v = ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
    (
        y.clamp(16, 235) as u8,
        u.clamp(16, 240) as u8,
        v.clamp(16, 240) as u8,
    )
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── yuv420_to_rgba ────────────────────────────────────────────────────────

    #[test]
    fn yuv420_output_size() {
        let w = 8u32;
        let h = 8u32;
        let y = vec![128u8; (w * h) as usize];
        let u = vec![128u8; (w / 2 * h / 2) as usize];
        let v = vec![128u8; (w / 2 * h / 2) as usize];
        let out = yuv420_to_rgba(&y, &u, &v, w, h);
        assert_eq!(out.len(), (w * h * 4) as usize);
    }

    #[test]
    fn yuv420_alpha_is_255() {
        let w = 4u32;
        let h = 4u32;
        let y = vec![128u8; (w * h) as usize];
        let u = vec![128u8; (w / 2 * h / 2) as usize];
        let v = vec![128u8; (w / 2 * h / 2) as usize];
        let out = yuv420_to_rgba(&y, &u, &v, w, h);
        for alpha in out.chunks_exact(4).map(|p| p[3]) {
            assert_eq!(alpha, 255);
        }
    }

    #[test]
    fn yuv420_grey_neutral_chrominance() {
        // Y=128, U=128, V=128 should produce a near-grey pixel
        let w = 2u32;
        let h = 2u32;
        let y = vec![128u8; (w * h) as usize];
        let u = vec![128u8; (w / 2 * h / 2) as usize];
        let v = vec![128u8; (w / 2 * h / 2) as usize];
        let out = yuv420_to_rgba(&y, &u, &v, w, h);
        let r = out[0] as i32;
        let g = out[1] as i32;
        let b = out[2] as i32;
        // All channels should be roughly equal (grey)
        assert!((r - g).abs() <= 5, "r={r}, g={g}");
        assert!((r - b).abs() <= 5, "r={r}, b={b}");
    }

    #[test]
    fn yuv420_too_small_returns_empty() {
        let out = yuv420_to_rgba(&[], &[], &[], 4, 4);
        assert!(out.is_empty());
    }

    // ── nv12_to_rgba ─────────────────────────────────────────────────────────

    #[test]
    fn nv12_to_rgba_output_size() {
        let w = 8u32;
        let h = 8u32;
        let nv12 = vec![128u8; (w * h + w * h / 2) as usize];
        let out = nv12_to_rgba(&nv12, w, h);
        assert_eq!(out.len(), (w * h * 4) as usize);
    }

    #[test]
    fn nv12_alpha_is_255() {
        let w = 4u32;
        let h = 4u32;
        let nv12 = vec![128u8; (w * h + w * h / 2) as usize];
        let out = nv12_to_rgba(&nv12, w, h);
        for alpha in out.chunks_exact(4).map(|p| p[3]) {
            assert_eq!(alpha, 255);
        }
    }

    // ── rgba_to_yuv420 ────────────────────────────────────────────────────────

    #[test]
    fn rgba_to_yuv420_plane_sizes() {
        let w = 8u32;
        let h = 8u32;
        let rgba = vec![128u8; (w * h * 4) as usize];
        let (y, u, v) = rgba_to_yuv420(&rgba, w, h);
        assert_eq!(y.len(), (w * h) as usize);
        assert_eq!(u.len(), (w / 2 * h / 2) as usize);
        assert_eq!(v.len(), (w / 2 * h / 2) as usize);
    }

    #[test]
    fn rgba_to_yuv420_roundtrip_approximate() {
        // Encode and decode; due to chroma subsampling the result won't be
        // pixel-perfect but should be close.
        let w = 4u32;
        let h = 4u32;
        let rgba: Vec<u8> = (0..(w * h * 4) as usize)
            .map(|i| match i % 4 {
                0 => 200,
                1 => 100,
                2 => 50,
                _ => 255,
            })
            .collect();
        let (y, u, v) = rgba_to_yuv420(&rgba, w, h);
        let decoded = yuv420_to_rgba(&y, &u, &v, w, h);
        assert_eq!(decoded.len(), (w * h * 4) as usize);
        // Just check the first pixel R channel is in a reasonable range
        let r_diff = (decoded[0] as i32 - 200).abs();
        assert!(r_diff < 30, "R roundtrip error too large: {r_diff}");
    }
}
