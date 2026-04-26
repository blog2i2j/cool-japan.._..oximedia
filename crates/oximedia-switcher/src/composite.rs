//! Alpha compositing primitives used by upstream and downstream keyers.
//!
//! All functions operate at the plane level and are format-agnostic — they
//! work on any `VideoFrame` that uses tightly-packed byte planes (no padding).
//! Alpha values are in the range 0–255 (0 = fully transparent, 255 = fully
//! opaque).

use crate::keyer::KeyerError;
use oximedia_codec::{Plane, VideoFrame};

/// Porter-Duff "over" composite: place `fill` over `background` using `alpha`.
///
/// For each plane the composite formula is:
///
/// ```text
/// out[i] = (fill[i] * a + bg[i] * (255 - a)) / 255
/// ```
///
/// where `a` is sampled from `alpha` at the position corresponding to each
/// plane pixel (accounting for chroma sub-sampling via the ratio of plane
/// dimensions to frame dimensions).
///
/// # Errors
///
/// Returns [`KeyerError::ProcessingError`] if:
/// * `fill` and `background` differ in pixel format, width or height.
/// * `alpha.len() != (fill.width * fill.height) as usize`.
pub(crate) fn composite_over(
    fill: &VideoFrame,
    background: &VideoFrame,
    alpha: &[u8],
) -> Result<VideoFrame, KeyerError> {
    // Validate dimensions and format match.
    if fill.format != background.format
        || fill.width != background.width
        || fill.height != background.height
    {
        return Err(KeyerError::ProcessingError(
            "fill and background frames must have identical format and dimensions".to_string(),
        ));
    }

    let frame_w = fill.width as usize;
    let frame_h = fill.height as usize;

    if alpha.len() != frame_w * frame_h {
        return Err(KeyerError::ProcessingError(format!(
            "alpha length {} does not match frame pixels {}",
            alpha.len(),
            frame_w * frame_h
        )));
    }

    let mut out = background.clone();

    for (plane_idx, (fill_plane, bg_plane)) in
        fill.planes.iter().zip(background.planes.iter()).enumerate()
    {
        // Skip planes that have no data.
        if fill_plane.data.is_empty() || bg_plane.data.is_empty() {
            continue;
        }

        let plane_w = fill_plane.width as usize;
        let plane_h = fill_plane.height as usize;

        // Bytes per pixel for this plane.
        let pixel_count = plane_w * plane_h;
        let bpp = fill_plane
            .data
            .len()
            .checked_div(pixel_count)
            .unwrap_or(1)
            .max(1);

        // Horizontal and vertical sub-sampling ratios relative to luma.
        let h_scale = frame_w.checked_div(plane_w).unwrap_or(1).max(1);
        let v_scale = frame_h.checked_div(plane_h).unwrap_or(1).max(1);

        let out_plane = &mut out.planes[plane_idx];

        for py in 0..plane_h {
            for px in 0..plane_w {
                // Map this plane pixel back to its alpha sample (luma-grid position).
                let alpha_x = (px * h_scale).min(frame_w.saturating_sub(1));
                let alpha_y = (py * v_scale).min(frame_h.saturating_sub(1));
                let alpha_idx = alpha_y * frame_w + alpha_x;
                let a = alpha[alpha_idx] as u16;
                let ia = 255u16 - a;

                let base = py * fill_plane.stride + px * bpp;
                let out_base = py * out_plane.stride + px * bpp;

                for b in 0..bpp {
                    let fi = base + b;
                    let bi = out_base + b;
                    if fi < fill_plane.data.len() && bi < bg_plane.data.len() {
                        let blended =
                            (fill_plane.data[fi] as u16 * a + bg_plane.data[bi] as u16 * ia) / 255;
                        out_plane.data[bi] = blended.min(255) as u8;
                    }
                }
            }
        }
    }

    Ok(out)
}

/// Apply clip, gain and optional invert to an alpha matte in-place.
///
/// Steps (in order):
/// 1. **Clip**: pixels with value `< clip * 255` are set to `0`.
/// 2. **Gain**: remaining pixels are scaled by `gain`, saturating at `255`.
/// 3. **Invert**: if `invert` is `true`, each value is replaced by `255 - x`.
///
/// * `clip`  — in `[0.0, 1.0]`; pixels below `clip * 255` → 0.
/// * `gain`  — `>= 1.0` scale factor; values saturate at 255.
/// * `invert` — whether to flip the alpha after clipping/gain.
pub(crate) fn apply_clip_gain_invert(alpha: &mut [u8], clip: f32, gain: f32, invert: bool) {
    // Convert clip ∈ [0, 1] to a u16 threshold using `clip * 256.0` (not 255).
    // This gives a half-open range [0, threshold) that maps cleanly:
    //   clip=0.0 → threshold=0   → no pixels clipped  (identity)
    //   clip=0.5 → threshold=128 → values 0..127 → 0, 128..255 pass through
    //   clip=1.0 → threshold=256 → all u8 values (0..255) are clipped to zero
    let clip_threshold = (clip * 256.0).ceil() as u16;

    for px in alpha.iter_mut() {
        let v = *px as u16;

        // Clip: values strictly below the threshold become fully transparent.
        let clipped = if v < clip_threshold { 0u16 } else { v };

        // Gain: scale and saturate.
        let gained = if gain != 1.0 {
            ((clipped as f32 * gain).round() as u16).min(255)
        } else {
            clipped
        };

        // Invert if requested.
        *px = if invert {
            (255u16 - gained.min(255)) as u8
        } else {
            gained as u8
        };
    }
}

/// Clone `fill` and append an extra plane containing the given raw alpha bytes.
///
/// The new plane has:
/// * `stride = fill.width as usize`
/// * `width  = fill.width`
/// * `height = fill.height`
///
/// The alpha plane is appended as the *last* plane of the returned frame.
pub(crate) fn attach_alpha_plane(fill: &VideoFrame, alpha: Vec<u8>) -> VideoFrame {
    let mut out = fill.clone();
    out.planes.push(Plane::with_dimensions(
        alpha,
        fill.width as usize,
        fill.width,
        fill.height,
    ));
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_codec::VideoFrame;
    use oximedia_core::{PixelFormat, Rational, Timestamp};

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn make_rgb24_frame(w: u32, h: u32, fill_val: u8) -> VideoFrame {
        let mut f = VideoFrame::new(PixelFormat::Rgb24, w, h);
        let data = vec![fill_val; (w * h * 3) as usize];
        f.planes
            .push(Plane::with_dimensions(data, (w * 3) as usize, w, h));
        f.timestamp = Timestamp::new(0, Rational::new(1, 1000));
        f
    }

    fn make_alpha(w: u32, h: u32, val: u8) -> Vec<u8> {
        vec![val; (w * h) as usize]
    }

    // -----------------------------------------------------------------------
    // apply_clip_gain_invert
    // -----------------------------------------------------------------------

    #[test]
    fn test_clip_zero_is_identity() {
        let mut alpha: Vec<u8> = (0u8..=255).collect();
        let original = alpha.clone();
        apply_clip_gain_invert(&mut alpha, 0.0, 1.0, false);
        assert_eq!(
            alpha, original,
            "clip=0, gain=1, invert=false must be identity"
        );
    }

    #[test]
    fn test_clip_one_gives_all_zeros() {
        let mut alpha: Vec<u8> = (0u8..=255).collect();
        apply_clip_gain_invert(&mut alpha, 1.0, 1.0, false);
        for (i, &v) in alpha.iter().enumerate() {
            assert_eq!(v, 0, "pixel {i}: clip=1.0 must zero all values");
        }
    }

    #[test]
    fn test_gain_doubles_then_saturates() {
        // Start with values: 0, 50, 100, 127, 200
        let mut alpha = vec![0u8, 50, 100, 127, 200];
        apply_clip_gain_invert(&mut alpha, 0.0, 2.0, false);
        // 0*2=0, 50*2=100, 100*2=200, 127*2=254, 200*2=400→255
        assert_eq!(alpha[0], 0);
        assert_eq!(alpha[1], 100);
        assert_eq!(alpha[2], 200);
        assert_eq!(alpha[3], 254);
        assert_eq!(alpha[4], 255);
    }

    #[test]
    fn test_invert_flips_values() {
        let mut alpha = vec![0u8, 128, 255];
        apply_clip_gain_invert(&mut alpha, 0.0, 1.0, true);
        assert_eq!(alpha[0], 255);
        assert_eq!(alpha[1], 127);
        assert_eq!(alpha[2], 0);
    }

    #[test]
    fn test_clip_then_invert() {
        // clip=0.5 zeros anything below 128; invert=true flips remainder
        let mut alpha = vec![0u8, 64, 128, 200, 255];
        apply_clip_gain_invert(&mut alpha, 0.5, 1.0, true);
        // 0 < 127.5 → 0 → inverted → 255
        assert_eq!(alpha[0], 255);
        // 64 < 127.5 → 0 → inverted → 255
        assert_eq!(alpha[1], 255);
        // 128 >= 127.5 → 128 → inverted → 127
        assert_eq!(alpha[2], 127);
        // 200 → 200 → inverted → 55
        assert_eq!(alpha[3], 55);
        // 255 → 255 → inverted → 0
        assert_eq!(alpha[4], 0);
    }

    // -----------------------------------------------------------------------
    // composite_over
    // -----------------------------------------------------------------------

    #[test]
    fn test_composite_over_full_opaque_gives_fill() {
        let w = 4u32;
        let h = 4u32;
        let fill = make_rgb24_frame(w, h, 200);
        let bg = make_rgb24_frame(w, h, 50);
        let alpha = make_alpha(w, h, 255);

        let result = composite_over(&fill, &bg, &alpha).expect("composite_over should succeed");

        for byte in &result.planes[0].data {
            assert_eq!(*byte, 200, "full alpha must select fill pixel");
        }
    }

    #[test]
    fn test_composite_over_full_transparent_gives_background() {
        let w = 4u32;
        let h = 4u32;
        let fill = make_rgb24_frame(w, h, 200);
        let bg = make_rgb24_frame(w, h, 50);
        let alpha = make_alpha(w, h, 0);

        let result = composite_over(&fill, &bg, &alpha).expect("composite_over should succeed");

        for byte in &result.planes[0].data {
            assert_eq!(*byte, 50, "zero alpha must select background pixel");
        }
    }

    #[test]
    fn test_composite_over_midpoint_blend() {
        let w = 2u32;
        let h = 2u32;
        let fill = make_rgb24_frame(w, h, 200);
        let bg = make_rgb24_frame(w, h, 100);
        // 50% alpha (128/255 ≈ 0.502)
        let alpha = make_alpha(w, h, 128);

        let result = composite_over(&fill, &bg, &alpha).expect("composite_over should succeed");

        // Expected: (200*128 + 100*127) / 255 = (25600 + 12700) / 255 = 38300/255 ≈ 150
        let expected = (200u16 * 128 + 100u16 * 127) / 255;
        for byte in &result.planes[0].data {
            let diff = (*byte as i16 - expected as i16).unsigned_abs();
            assert!(
                diff <= 1,
                "midpoint blend: got {byte}, expected ~{expected}"
            );
        }
    }

    #[test]
    fn test_composite_over_format_mismatch_errors() {
        use oximedia_core::PixelFormat;

        let w = 2u32;
        let h = 2u32;
        let fill = make_rgb24_frame(w, h, 128);

        // Different pixel format.
        let mut bg = fill.clone();
        bg.format = PixelFormat::Yuv420p;
        let alpha = make_alpha(w, h, 128);
        assert!(
            composite_over(&fill, &bg, &alpha).is_err(),
            "mismatched format must error"
        );
    }

    #[test]
    fn test_composite_over_alpha_length_mismatch_errors() {
        let w = 4u32;
        let h = 4u32;
        let fill = make_rgb24_frame(w, h, 128);
        let bg = make_rgb24_frame(w, h, 64);
        // Wrong alpha length.
        let alpha = vec![255u8; 10];
        assert!(
            composite_over(&fill, &bg, &alpha).is_err(),
            "wrong alpha length must error"
        );
    }

    // -----------------------------------------------------------------------
    // attach_alpha_plane
    // -----------------------------------------------------------------------

    #[test]
    fn test_attach_alpha_plane_increments_plane_count() {
        let fill = make_rgb24_frame(4, 4, 128);
        let original_count = fill.planes.len();
        let alpha = vec![200u8; 4 * 4];

        let result = attach_alpha_plane(&fill, alpha.clone());

        assert_eq!(
            result.planes.len(),
            original_count + 1,
            "attach_alpha_plane must add exactly one plane"
        );
    }

    #[test]
    fn test_attach_alpha_plane_preserves_alpha_bytes() {
        let fill = make_rgb24_frame(4, 4, 128);
        let alpha: Vec<u8> = (0..16).map(|i| i * 16).collect();

        let result = attach_alpha_plane(&fill, alpha.clone());
        let added = result.planes.last().expect("last plane must exist");

        assert_eq!(added.data, alpha, "alpha bytes must be preserved exactly");
    }

    #[test]
    fn test_attach_alpha_plane_stride_equals_width() {
        let fill = make_rgb24_frame(6, 4, 64);
        let alpha = vec![255u8; 6 * 4];

        let result = attach_alpha_plane(&fill, alpha);
        let added = result.planes.last().expect("last plane must exist");

        assert_eq!(
            added.stride, fill.width as usize,
            "alpha plane stride must equal frame width"
        );
        assert_eq!(added.width, fill.width);
        assert_eq!(added.height, fill.height);
    }
}
