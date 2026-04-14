//! Image padding / framing operations.
//!
//! This module provides three classic approaches for fitting a source image
//! into a different destination rectangle while preserving the source aspect
//! ratio:
//!
//! | Function | Strategy |
//! |---|---|
//! | `letterbox` | Add horizontal bars (top/bottom) to fill wider targets |
//! | `pillarbox` | Add vertical bars (left/right) to fill taller targets |
//! | `center_crop` | Crop the source to exactly fill the destination |
//!
//! All functions operate on flat row-major **RGBA** (`4 bytes per pixel`)
//! buffers.  The padding colour is solid black (`[0, 0, 0, 255]`).
//!
//! # Example
//!
//! ```
//! use oximedia_scaling::padding::{letterbox, pillarbox, center_crop};
//!
//! // 4×2 RGBA source → pad into 4×4 (letterbox)
//! let src: Vec<u8> = vec![255u8; 4 * 2 * 4];
//! let dst = letterbox(&src, 4, 2, 4, 4);
//! assert_eq!(dst.len(), 4 * 4 * 4);
//! ```

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

const CHANNELS: usize = 4;
const BLACK_PIXEL: [u8; CHANNELS] = [0, 0, 0, 255];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Scale a source region with nearest-neighbour interpolation into `dst`.
///
/// `dst` is an output slice of size `dst_w * dst_h * CHANNELS` that has
/// already been filled with a background colour.  The source pixels are
/// blitted into the rectangle defined by `(offset_x, offset_y)` with size
/// `(place_w, place_h)`.
fn blit_scaled(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    dst: &mut [u8],
    dst_w: u32,
    place_x: u32,
    place_y: u32,
    place_w: u32,
    place_h: u32,
) {
    if src_w == 0 || src_h == 0 || place_w == 0 || place_h == 0 {
        return;
    }
    let sw = src_w as usize;
    let sh = src_h as usize;
    let pw = place_w as usize;
    let ph = place_h as usize;
    let dw = dst_w as usize;

    for dy in 0..ph {
        let sy = (dy * sh / ph).min(sh - 1);
        for dx in 0..pw {
            let sx = (dx * sw / pw).min(sw - 1);

            let src_offset = (sy * sw + sx) * CHANNELS;
            let dst_offset = ((dy + place_y as usize) * dw + (dx + place_x as usize)) * CHANNELS;

            if src_offset + CHANNELS <= src.len() && dst_offset + CHANNELS <= dst.len() {
                dst[dst_offset..dst_offset + CHANNELS]
                    .copy_from_slice(&src[src_offset..src_offset + CHANNELS]);
            }
        }
    }
}

/// Fill an RGBA buffer with the black padding colour.
fn fill_black(buf: &mut [u8]) {
    for px in buf.chunks_exact_mut(CHANNELS) {
        px.copy_from_slice(&BLACK_PIXEL);
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Fit the source image into `dst_w × dst_h` by adding horizontal black bars.
///
/// The source is scaled (nearest-neighbour) so that its width equals `dst_w`.
/// If the resulting height is less than `dst_h`, black bars are added at the
/// top and bottom.  If the source is already taller than wide relative to the
/// destination, the function falls back to centring without additional
/// letterbox bars (the image fills the height and the width is pillarboxed
/// instead — identical to `pillarbox` in that case).
///
/// # Panics
///
/// Panics in debug mode if `src.len() < src_w * src_h * 4`.
///
/// # Returns
///
/// An RGBA pixel buffer of length `dst_w * dst_h * 4`.
pub fn letterbox(src: &[u8], src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> Vec<u8> {
    let total = (dst_w * dst_h) as usize * CHANNELS;
    let mut out = vec![0u8; total];
    fill_black(&mut out);

    if src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0 {
        return out;
    }

    // Scale so width fits exactly; compute the resulting height.
    let scale = dst_w as f64 / src_w as f64;
    let scaled_h = ((src_h as f64 * scale).round() as u32).max(1).min(dst_h);
    let scaled_w = dst_w;

    let offset_y = (dst_h.saturating_sub(scaled_h)) / 2;

    blit_scaled(
        src, src_w, src_h, &mut out, dst_w, 0, offset_y, scaled_w, scaled_h,
    );
    out
}

/// Fit the source image into `dst_w × dst_h` by adding vertical black bars.
///
/// The source is scaled (nearest-neighbour) so that its height equals `dst_h`.
/// If the resulting width is less than `dst_w`, black bars are added to the
/// left and right.
///
/// # Returns
///
/// An RGBA pixel buffer of length `dst_w * dst_h * 4`.
pub fn pillarbox(src: &[u8], src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> Vec<u8> {
    let total = (dst_w * dst_h) as usize * CHANNELS;
    let mut out = vec![0u8; total];
    fill_black(&mut out);

    if src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0 {
        return out;
    }

    // Scale so height fits exactly; compute the resulting width.
    let scale = dst_h as f64 / src_h as f64;
    let scaled_w = ((src_w as f64 * scale).round() as u32).max(1).min(dst_w);
    let scaled_h = dst_h;

    let offset_x = (dst_w.saturating_sub(scaled_w)) / 2;

    blit_scaled(
        src, src_w, src_h, &mut out, dst_w, offset_x, 0, scaled_w, scaled_h,
    );
    out
}

/// Crop and scale the source image to fill `dst_w × dst_h` exactly.
///
/// The scale factor is chosen so that both dimensions of the source are
/// at least as large as the destination (i.e., the *larger* of the two
/// possible scale factors is selected).  The scaled source is then
/// centre-cropped to `dst_w × dst_h`.
///
/// # Returns
///
/// An RGBA pixel buffer of length `dst_w * dst_h * 4`.
pub fn center_crop(src: &[u8], src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> Vec<u8> {
    let total = (dst_w * dst_h) as usize * CHANNELS;
    let mut out = vec![0u8; total];
    fill_black(&mut out);

    if src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0 {
        return out;
    }

    // Use the larger scale factor so the source covers the destination in both axes.
    let scale_w = dst_w as f64 / src_w as f64;
    let scale_h = dst_h as f64 / src_h as f64;
    let scale = scale_w.max(scale_h);

    let scaled_w = ((src_w as f64 * scale).round() as u32).max(1);
    let scaled_h = ((src_h as f64 * scale).round() as u32).max(1);

    // The crop offset into the scaled image (centred).
    let crop_x = scaled_w.saturating_sub(dst_w) / 2;
    let crop_y = scaled_h.saturating_sub(dst_h) / 2;

    let sw = src_w as usize;
    let sh = src_h as usize;
    let scaled_w_u = scaled_w as usize;
    let scaled_h_u = scaled_h as usize;
    let dw = dst_w as usize;
    let dh = dst_h as usize;
    let cx = crop_x as usize;
    let cy = crop_y as usize;

    for dy in 0..dh {
        let sy_scaled = dy + cy;
        if sy_scaled >= scaled_h_u {
            break;
        }
        let sy = (sy_scaled * sh / scaled_h_u).min(sh - 1);

        for dx in 0..dw {
            let sx_scaled = dx + cx;
            if sx_scaled >= scaled_w_u {
                break;
            }
            let sx = (sx_scaled * sw / scaled_w_u).min(sw - 1);

            let src_offset = (sy * sw + sx) * CHANNELS;
            let dst_offset = (dy * dw + dx) * CHANNELS;

            if src_offset + CHANNELS <= src.len() && dst_offset + CHANNELS <= out.len() {
                out[dst_offset..dst_offset + CHANNELS]
                    .copy_from_slice(&src[src_offset..src_offset + CHANNELS]);
            }
        }
    }

    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_rgba(r: u8, g: u8, b: u8, a: u8, w: u32, h: u32) -> Vec<u8> {
        let n = (w * h) as usize * CHANNELS;
        let mut buf = vec![0u8; n];
        for px in buf.chunks_exact_mut(CHANNELS) {
            px.copy_from_slice(&[r, g, b, a]);
        }
        buf
    }

    // ─── letterbox ────────────────────────────────────────────────────────────

    #[test]
    fn test_letterbox_output_size() {
        let src = solid_rgba(255, 0, 0, 255, 4, 2);
        let out = letterbox(&src, 4, 2, 4, 4);
        assert_eq!(out.len(), 4 * 4 * CHANNELS);
    }

    #[test]
    fn test_letterbox_same_size_no_bars() {
        let src = solid_rgba(200, 100, 50, 255, 8, 8);
        let out = letterbox(&src, 8, 8, 8, 8);
        // All pixels should be from the source (no bars needed)
        for i in 0..8usize {
            let off = (i * 8) * CHANNELS;
            assert_eq!(out[off], 200, "row {i}: red channel should be 200");
        }
    }

    #[test]
    fn test_letterbox_bars_are_black() {
        // Source 8×4 (wide) → dst 8×8: expect top/bottom rows to be black
        let src = solid_rgba(255, 255, 255, 255, 8, 4);
        let out = letterbox(&src, 8, 4, 8, 8);
        // The first row should be black padding
        let first_row_r = out[0];
        let first_row_g = out[1];
        let first_row_b = out[2];
        assert_eq!(first_row_r, 0, "letterbox top bar red should be 0");
        assert_eq!(first_row_g, 0, "letterbox top bar green should be 0");
        assert_eq!(first_row_b, 0, "letterbox top bar blue should be 0");
    }

    #[test]
    fn test_letterbox_zero_dims_returns_empty() {
        let out = letterbox(&[], 0, 0, 4, 4);
        assert_eq!(out.len(), 4 * 4 * CHANNELS);
        // Should be all black
        assert!(out.iter().take(3).all(|&b| b == 0));
    }

    // ─── pillarbox ────────────────────────────────────────────────────────────

    #[test]
    fn test_pillarbox_output_size() {
        let src = solid_rgba(0, 255, 0, 255, 2, 4);
        let out = pillarbox(&src, 2, 4, 4, 4);
        assert_eq!(out.len(), 4 * 4 * CHANNELS);
    }

    #[test]
    fn test_pillarbox_bars_are_black() {
        // Source 4×8 (tall) → dst 8×8: expect left/right columns to be black
        let src = solid_rgba(128, 0, 0, 255, 4, 8);
        let out = pillarbox(&src, 4, 8, 8, 8);
        // Column 0, row 0 should be black
        let px0_r = out[0];
        assert_eq!(px0_r, 0, "pillarbox left bar should be black");
    }

    #[test]
    fn test_pillarbox_same_aspect_no_bars() {
        let src = solid_rgba(64, 128, 192, 255, 4, 4);
        let out = pillarbox(&src, 4, 4, 4, 4);
        // All pixels should match source
        for chunk in out.chunks_exact(CHANNELS) {
            assert_eq!(chunk[0], 64);
        }
    }

    // ─── center_crop ──────────────────────────────────────────────────────────

    #[test]
    fn test_center_crop_output_size() {
        let src = solid_rgba(0, 0, 255, 255, 8, 8);
        let out = center_crop(&src, 8, 8, 4, 4);
        assert_eq!(out.len(), 4 * 4 * CHANNELS);
    }

    #[test]
    fn test_center_crop_fills_destination() {
        // Source 8×4 → crop to 4×4: no black pixels expected
        let src = solid_rgba(200, 0, 0, 255, 8, 4);
        let out = center_crop(&src, 8, 4, 4, 4);
        for chunk in out.chunks_exact(CHANNELS) {
            assert_eq!(chunk[0], 200, "crop output should not contain black pixels");
        }
    }

    #[test]
    fn test_center_crop_zero_dst() {
        let src = solid_rgba(1, 2, 3, 255, 4, 4);
        let out = center_crop(&src, 4, 4, 0, 4);
        assert_eq!(out.len(), 0);
    }
}
