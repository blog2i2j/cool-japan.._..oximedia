//! Video superimpose — compositing two video streams with alpha blending.
//!
//! This module provides compositing of a *foreground* (overlay) RGBA frame on
//! top of a *background* frame using per-pixel alpha.  It supports:
//!
//! - **Position offset**: place the overlay at any (x, y) pixel offset,
//!   including partially off-screen.
//! - **Scale**: scale the overlay to a target size before compositing.
//! - **Blend modes**: Normal, Add, Multiply, Screen, and Overlay.
//!
//! All pixel data is in RGBA format — 4 bytes per pixel, channels R, G, B, A.
//! The background may be RGB (3 bytes/pixel) or RGBA (4 bytes/pixel); output
//! always matches the background channel layout.

// -----------------------------------------------------------------------
// Error type
// -----------------------------------------------------------------------

/// Errors from the superimpose compositing functions.
#[derive(Debug, thiserror::Error)]
pub enum SuperimposeError {
    /// A frame has invalid (zero) dimensions.
    #[error("invalid frame dimensions: {width}x{height}")]
    InvalidDimensions {
        /// Frame width.
        width: u32,
        /// Frame height.
        height: u32,
    },

    /// A buffer has an unexpected size given its declared dimensions.
    #[error("buffer size mismatch for '{label}': expected {expected}, got {actual}")]
    BufferSizeMismatch {
        /// Label identifying which buffer is mismatched.
        label: &'static str,
        /// Expected byte count.
        expected: usize,
        /// Actual byte count.
        actual: usize,
    },

    /// An invalid scale factor was provided.
    #[error("scale factor must be positive and finite, got {value}")]
    InvalidScale {
        /// The invalid scale value.
        value: f64,
    },
}

// -----------------------------------------------------------------------
// Blend modes
// -----------------------------------------------------------------------

/// Pixel-level blend modes for compositing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BlendMode {
    /// Standard alpha compositing (Porter-Duff SRC_OVER).
    #[default]
    Normal,
    /// Additive blend: `dst + src * alpha` (clamped).
    Add,
    /// Multiply blend: `dst * src / 255` with alpha mixing.
    Multiply,
    /// Screen blend: `255 - (255-dst)*(255-src)/255` with alpha mixing.
    Screen,
    /// Photoshop-style overlay: multiply for dark sources, screen for light.
    Overlay,
}

// -----------------------------------------------------------------------
// Configuration
// -----------------------------------------------------------------------

/// Configuration for a single superimpose operation.
#[derive(Debug, Clone)]
pub struct SuperimposeConfig {
    /// Pixel offset of the top-left corner of the overlay relative to the
    /// top-left corner of the background.  May be negative (partial off-screen).
    pub x_offset: i32,
    /// Vertical pixel offset (positive = down).
    pub y_offset: i32,
    /// Uniform scale applied to the overlay *before* compositing.
    ///
    /// `1.0` uses the overlay at its native resolution.  `0.5` halves the
    /// overlay dimensions; `2.0` doubles them.  Must be positive and finite.
    pub scale: f64,
    /// Blend mode used when combining foreground and background pixels.
    pub blend_mode: BlendMode,
    /// Global opacity multiplier applied on top of the per-pixel alpha channel.
    ///
    /// Range `[0.0, 1.0]`.  `1.0` uses the alpha channel as-is.
    pub opacity: f64,
}

impl Default for SuperimposeConfig {
    fn default() -> Self {
        Self {
            x_offset: 0,
            y_offset: 0,
            scale: 1.0,
            blend_mode: BlendMode::Normal,
            opacity: 1.0,
        }
    }
}

// -----------------------------------------------------------------------
// Public API
// -----------------------------------------------------------------------

/// Composite an RGBA overlay onto an RGB background frame.
///
/// # Arguments
///
/// * `bg` – background RGB buffer (`bg_width * bg_height * 3` bytes).
/// * `bg_width` / `bg_height` – background dimensions.
/// * `fg` – foreground RGBA buffer (`fg_width * fg_height * 4` bytes).
/// * `fg_width` / `fg_height` – foreground/overlay dimensions before scaling.
/// * `cfg` – compositing configuration.
///
/// # Returns
///
/// A new `Vec<u8>` containing the composited RGB frame.
///
/// # Errors
///
/// Returns [`SuperimposeError`] if any buffer has an unexpected size, or
/// if dimensions or the scale factor are invalid.
pub fn superimpose_rgb(
    bg: &[u8],
    bg_width: u32,
    bg_height: u32,
    fg: &[u8],
    fg_width: u32,
    fg_height: u32,
    cfg: &SuperimposeConfig,
) -> Result<Vec<u8>, SuperimposeError> {
    validate_dims(bg_width, bg_height)?;
    validate_dims(fg_width, fg_height)?;
    validate_buffer("bg", bg, bg_width, bg_height, 3)?;
    validate_buffer("fg", fg, fg_width, fg_height, 4)?;
    validate_scale(cfg.scale)?;

    let opacity = cfg.opacity.clamp(0.0, 1.0);

    // Scale the foreground to the target size.
    let (scaled_fg, sw, sh) = scale_rgba(fg, fg_width, fg_height, cfg.scale)?;

    let bw = bg_width as usize;
    let bh = bg_height as usize;
    let mut out = bg.to_vec();

    for sy in 0..sh {
        let dy = cfg.y_offset + sy as i32;
        if dy < 0 || dy >= bh as i32 {
            continue;
        }
        let dy = dy as usize;
        for sx in 0..sw {
            let dx = cfg.x_offset + sx as i32;
            if dx < 0 || dx >= bw as i32 {
                continue;
            }
            let dx = dx as usize;

            let fg_base = (sy * sw + sx) * 4;
            let fr = scaled_fg[fg_base] as f64;
            let fg_g = scaled_fg[fg_base + 1] as f64;
            let fb = scaled_fg[fg_base + 2] as f64;
            let fa = scaled_fg[fg_base + 3] as f64 / 255.0 * opacity;

            let bg_base = (dy * bw + dx) * 3;
            let br = out[bg_base] as f64;
            let bg_g = out[bg_base + 1] as f64;
            let bb = out[bg_base + 2] as f64;

            let (cr, cg, cb) = blend_pixel(br, bg_g, bb, fr, fg_g, fb, fa, cfg.blend_mode);
            out[bg_base] = cr.round().clamp(0.0, 255.0) as u8;
            out[bg_base + 1] = cg.round().clamp(0.0, 255.0) as u8;
            out[bg_base + 2] = cb.round().clamp(0.0, 255.0) as u8;
        }
    }
    Ok(out)
}

/// Composite an RGBA overlay onto an RGBA background frame.
///
/// The output alpha channel is computed using the Porter-Duff `src-over`
/// formula regardless of blend mode (blend mode only affects RGB channels).
///
/// # Errors
///
/// Returns [`SuperimposeError`] on invalid buffer sizes, dimensions, or scale.
pub fn superimpose_rgba(
    bg: &[u8],
    bg_width: u32,
    bg_height: u32,
    fg: &[u8],
    fg_width: u32,
    fg_height: u32,
    cfg: &SuperimposeConfig,
) -> Result<Vec<u8>, SuperimposeError> {
    validate_dims(bg_width, bg_height)?;
    validate_dims(fg_width, fg_height)?;
    validate_buffer("bg", bg, bg_width, bg_height, 4)?;
    validate_buffer("fg", fg, fg_width, fg_height, 4)?;
    validate_scale(cfg.scale)?;

    let opacity = cfg.opacity.clamp(0.0, 1.0);

    let (scaled_fg, sw, sh) = scale_rgba(fg, fg_width, fg_height, cfg.scale)?;

    let bw = bg_width as usize;
    let bh = bg_height as usize;
    let mut out = bg.to_vec();

    for sy in 0..sh {
        let dy = cfg.y_offset + sy as i32;
        if dy < 0 || dy >= bh as i32 {
            continue;
        }
        let dy = dy as usize;
        for sx in 0..sw {
            let dx = cfg.x_offset + sx as i32;
            if dx < 0 || dx >= bw as i32 {
                continue;
            }
            let dx = dx as usize;

            let fg_base = (sy * sw + sx) * 4;
            let fr = scaled_fg[fg_base] as f64;
            let fg_g = scaled_fg[fg_base + 1] as f64;
            let fb = scaled_fg[fg_base + 2] as f64;
            let fa = scaled_fg[fg_base + 3] as f64 / 255.0 * opacity;

            let bg_base = (dy * bw + dx) * 4;
            let br = out[bg_base] as f64;
            let bg_g = out[bg_base + 1] as f64;
            let bb = out[bg_base + 2] as f64;
            let ba = out[bg_base + 3] as f64 / 255.0;

            let (cr, cg, cb) = blend_pixel(br, bg_g, bb, fr, fg_g, fb, fa, cfg.blend_mode);

            // Porter-Duff src-over alpha composition.
            let out_a = fa + ba * (1.0 - fa);

            out[bg_base] = cr.round().clamp(0.0, 255.0) as u8;
            out[bg_base + 1] = cg.round().clamp(0.0, 255.0) as u8;
            out[bg_base + 2] = cb.round().clamp(0.0, 255.0) as u8;
            out[bg_base + 3] = (out_a * 255.0).round().clamp(0.0, 255.0) as u8;
        }
    }
    Ok(out)
}

// -----------------------------------------------------------------------
// Internal helpers
// -----------------------------------------------------------------------

/// Validate that width and height are both non-zero.
fn validate_dims(width: u32, height: u32) -> Result<(), SuperimposeError> {
    if width == 0 || height == 0 {
        Err(SuperimposeError::InvalidDimensions { width, height })
    } else {
        Ok(())
    }
}

/// Validate that a buffer has `width * height * channels` bytes.
fn validate_buffer(
    label: &'static str,
    buf: &[u8],
    width: u32,
    height: u32,
    channels: usize,
) -> Result<(), SuperimposeError> {
    let expected = width as usize * height as usize * channels;
    if buf.len() != expected {
        Err(SuperimposeError::BufferSizeMismatch {
            label,
            expected,
            actual: buf.len(),
        })
    } else {
        Ok(())
    }
}

/// Validate that a scale factor is positive and finite.
fn validate_scale(scale: f64) -> Result<(), SuperimposeError> {
    if !scale.is_finite() || scale <= 0.0 {
        Err(SuperimposeError::InvalidScale { value: scale })
    } else {
        Ok(())
    }
}

/// Nearest-neighbour scale of an RGBA buffer.
///
/// Returns `(scaled_buf, new_width, new_height)`.
fn scale_rgba(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    scale: f64,
) -> Result<(Vec<u8>, usize, usize), SuperimposeError> {
    let dst_w = ((src_w as f64 * scale).round() as usize).max(1);
    let dst_h = ((src_h as f64 * scale).round() as usize).max(1);

    if scale == 1.0 && dst_w == src_w as usize && dst_h == src_h as usize {
        return Ok((src.to_vec(), dst_w, dst_h));
    }

    let sw = src_w as usize;
    let sh = src_h as usize;
    let mut out = vec![0u8; dst_w * dst_h * 4];

    for dy in 0..dst_h {
        // Map destination pixel centre back to source via inverse scale.
        let sy = ((dy as f64 + 0.5) / scale) as usize;
        let sy = sy.min(sh - 1);
        for dx in 0..dst_w {
            let sx = ((dx as f64 + 0.5) / scale) as usize;
            let sx = sx.min(sw - 1);

            let src_idx = (sy * sw + sx) * 4;
            let dst_idx = (dy * dst_w + dx) * 4;
            out[dst_idx..dst_idx + 4].copy_from_slice(&src[src_idx..src_idx + 4]);
        }
    }
    Ok((out, dst_w, dst_h))
}

/// Apply the requested blend mode and return composited (R, G, B) as `f64`.
///
/// `fa` is already in `[0, 1]`.
#[inline]
fn blend_pixel(
    br: f64,
    bg: f64,
    bb: f64,
    fr: f64,
    fg: f64,
    fb: f64,
    fa: f64,
    mode: BlendMode,
) -> (f64, f64, f64) {
    let (mr, mg, mb) = match mode {
        BlendMode::Normal => (fr, fg, fb),
        BlendMode::Add => (
            (br + fr).min(255.0),
            (bg + fg).min(255.0),
            (bb + fb).min(255.0),
        ),
        BlendMode::Multiply => (br * fr / 255.0, bg * fg / 255.0, bb * fb / 255.0),
        BlendMode::Screen => (
            255.0 - (255.0 - br) * (255.0 - fr) / 255.0,
            255.0 - (255.0 - bg) * (255.0 - fg) / 255.0,
            255.0 - (255.0 - bb) * (255.0 - fb) / 255.0,
        ),
        BlendMode::Overlay => (
            overlay_channel(br, fr),
            overlay_channel(bg, fg),
            overlay_channel(bb, fb),
        ),
    };

    // Alpha-composite the blended result over the background.
    let cr = mr * fa + br * (1.0 - fa);
    let cg = mg * fa + bg * (1.0 - fa);
    let cb = mb * fa + bb * (1.0 - fa);
    (cr, cg, cb)
}

/// Photoshop overlay blend for a single channel.
///
/// Dark background values use multiply; bright ones use screen.
#[inline]
fn overlay_channel(b: f64, f: f64) -> f64 {
    if b < 128.0 {
        2.0 * b * f / 255.0
    } else {
        255.0 - 2.0 * (255.0 - b) * (255.0 - f) / 255.0
    }
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a solid RGBA buffer of `w*h` pixels.
    fn rgba_frame(w: u32, h: u32, r: u8, g: u8, b: u8, a: u8) -> Vec<u8> {
        let n = (w * h) as usize * 4;
        let mut buf = vec![0u8; n];
        for px in buf.chunks_exact_mut(4) {
            px[0] = r;
            px[1] = g;
            px[2] = b;
            px[3] = a;
        }
        buf
    }

    /// Create a solid RGB buffer.
    fn rgb_frame(w: u32, h: u32, r: u8, g: u8, b: u8) -> Vec<u8> {
        let n = (w * h) as usize * 3;
        let mut buf = vec![0u8; n];
        for px in buf.chunks_exact_mut(3) {
            px[0] = r;
            px[1] = g;
            px[2] = b;
        }
        buf
    }

    // 1. Fully transparent overlay leaves background unchanged.
    #[test]
    fn test_transparent_overlay_no_change() {
        let bg = rgb_frame(4, 4, 100, 100, 100);
        let fg = rgba_frame(2, 2, 255, 0, 0, 0); // alpha = 0
        let cfg = SuperimposeConfig::default();
        let out = superimpose_rgb(&bg, 4, 4, &fg, 2, 2, &cfg).unwrap();
        assert_eq!(out, bg);
    }

    // 2. Fully opaque overlay completely replaces background pixels.
    #[test]
    fn test_opaque_overlay_replaces_bg() {
        let bg = rgb_frame(4, 4, 50, 50, 50);
        let fg = rgba_frame(4, 4, 200, 0, 0, 255); // fully opaque red
        let cfg = SuperimposeConfig::default();
        let out = superimpose_rgb(&bg, 4, 4, &fg, 4, 4, &cfg).unwrap();
        for px in out.chunks_exact(3) {
            assert_eq!(px[0], 200, "R should be 200");
            assert_eq!(px[1], 0, "G should be 0");
            assert_eq!(px[2], 0, "B should be 0");
        }
    }

    // 3. Overlay at non-zero offset only affects the correct region.
    #[test]
    fn test_overlay_offset() {
        let bg = rgb_frame(8, 8, 0, 0, 0);
        let fg = rgba_frame(2, 2, 255, 255, 255, 255);
        let cfg = SuperimposeConfig {
            x_offset: 3,
            y_offset: 3,
            ..SuperimposeConfig::default()
        };
        let out = superimpose_rgb(&bg, 8, 8, &fg, 2, 2, &cfg).unwrap();

        // Pixel at (3,3) should be white.
        let idx = (3 * 8 + 3) * 3;
        assert_eq!(out[idx], 255);

        // Pixel at (0,0) should still be black.
        assert_eq!(out[0], 0);
    }

    // 4. Partially off-screen overlay does not panic.
    #[test]
    fn test_overlay_partially_offscreen() {
        let bg = rgb_frame(8, 8, 50, 50, 50);
        let fg = rgba_frame(4, 4, 255, 0, 0, 255);
        let cfg = SuperimposeConfig {
            x_offset: 6,  // extends past right edge
            y_offset: -1, // starts above frame
            ..SuperimposeConfig::default()
        };
        // Should succeed without panicking.
        let out = superimpose_rgb(&bg, 8, 8, &fg, 4, 4, &cfg).unwrap();
        assert_eq!(out.len(), 8 * 8 * 3);
    }

    // 5. Scale = 0.5 produces half-size overlay.
    #[test]
    fn test_scale_half() {
        let bg = rgb_frame(8, 8, 0, 0, 0);
        let fg = rgba_frame(4, 4, 255, 255, 255, 255);
        let cfg = SuperimposeConfig {
            scale: 0.5,
            ..SuperimposeConfig::default()
        };
        let out = superimpose_rgb(&bg, 8, 8, &fg, 4, 4, &cfg).unwrap();
        // With scale 0.5, the 4×4 overlay becomes 2×2.
        // Pixel (0,0) should be white (inside the scaled overlay).
        let idx = 0;
        assert_eq!(
            out[idx], 255,
            "top-left should be white (inside scaled 2x2 overlay)"
        );
        // Pixel (3,3) should be black (outside the 2×2 overlay).
        let idx2 = (3 * 8 + 3) * 3;
        assert_eq!(out[idx2], 0, "pixel outside scaled overlay should be black");
    }

    // 6. Add blend mode accumulates brightness.
    #[test]
    fn test_add_blend() {
        let bg = rgb_frame(2, 2, 100, 100, 100);
        let fg = rgba_frame(2, 2, 100, 100, 100, 255);
        let cfg = SuperimposeConfig {
            blend_mode: BlendMode::Add,
            ..SuperimposeConfig::default()
        };
        let out = superimpose_rgb(&bg, 2, 2, &fg, 2, 2, &cfg).unwrap();
        // 100 + 100 = 200 for each channel.
        for px in out.chunks_exact(3) {
            assert_eq!(px[0], 200);
        }
    }

    // 7. Multiply blend on white background leaves foreground unchanged.
    #[test]
    fn test_multiply_white_bg() {
        let bg = rgb_frame(2, 2, 255, 255, 255);
        let fg = rgba_frame(2, 2, 128, 64, 200, 255);
        let cfg = SuperimposeConfig {
            blend_mode: BlendMode::Multiply,
            ..SuperimposeConfig::default()
        };
        let out = superimpose_rgb(&bg, 2, 2, &fg, 2, 2, &cfg).unwrap();
        // Multiply with white (255): result = bg * fg / 255 = 255 * fg / 255 = fg.
        // With alpha=1 composite: result = fg * 1 + bg * 0 = fg.
        for px in out.chunks_exact(3) {
            assert_eq!(px[0], 128, "R multiply/white expected 128, got {}", px[0]);
        }
    }

    // 8. Screen blend on black background leaves foreground unchanged.
    #[test]
    fn test_screen_black_bg() {
        let bg = rgb_frame(2, 2, 0, 0, 0);
        let fg = rgba_frame(2, 2, 100, 100, 100, 255);
        let cfg = SuperimposeConfig {
            blend_mode: BlendMode::Screen,
            ..SuperimposeConfig::default()
        };
        let out = superimpose_rgb(&bg, 2, 2, &fg, 2, 2, &cfg).unwrap();
        // Screen with black bg: 255 - (255-0)*(255-100)/255 = 255 - 255*155/255 = 255-155=100.
        for px in out.chunks_exact(3) {
            assert_eq!(px[0], 100);
        }
    }

    // 9. RGBA-on-RGBA compositing updates the alpha channel.
    #[test]
    fn test_rgba_alpha_compositing() {
        let bg = rgba_frame(2, 2, 0, 0, 0, 128);
        let fg = rgba_frame(2, 2, 255, 255, 255, 255);
        let cfg = SuperimposeConfig::default();
        let out = superimpose_rgba(&bg, 2, 2, &fg, 2, 2, &cfg).unwrap();
        // Fully opaque fg over any bg → alpha = 255.
        for px in out.chunks_exact(4) {
            assert_eq!(
                px[3], 255,
                "output alpha should be 255 when fg is fully opaque"
            );
        }
    }

    // 10. opacity=0 means the overlay is invisible.
    #[test]
    fn test_opacity_zero_invisible() {
        let bg = rgb_frame(4, 4, 50, 100, 150);
        let fg = rgba_frame(4, 4, 255, 0, 0, 255);
        let cfg = SuperimposeConfig {
            opacity: 0.0,
            ..SuperimposeConfig::default()
        };
        let out = superimpose_rgb(&bg, 4, 4, &fg, 4, 4, &cfg).unwrap();
        assert_eq!(out, bg);
    }

    // 11. Invalid scale returns an error.
    #[test]
    fn test_invalid_scale_error() {
        let bg = rgb_frame(4, 4, 0, 0, 0);
        let fg = rgba_frame(2, 2, 0, 0, 0, 255);
        let cfg = SuperimposeConfig {
            scale: -1.0,
            ..SuperimposeConfig::default()
        };
        assert!(matches!(
            superimpose_rgb(&bg, 4, 4, &fg, 2, 2, &cfg),
            Err(SuperimposeError::InvalidScale { .. })
        ));
    }

    // 12. Buffer size mismatch is detected.
    #[test]
    fn test_buffer_mismatch() {
        let bg = vec![0u8; 10]; // wrong size
        let fg = rgba_frame(2, 2, 0, 0, 0, 255);
        let cfg = SuperimposeConfig::default();
        assert!(matches!(
            superimpose_rgb(&bg, 4, 4, &fg, 2, 2, &cfg),
            Err(SuperimposeError::BufferSizeMismatch { .. })
        ));
    }

    // 13. Zero background dimensions return error.
    #[test]
    fn test_zero_bg_dimensions() {
        let bg = vec![];
        let fg = rgba_frame(2, 2, 0, 0, 0, 255);
        let cfg = SuperimposeConfig::default();
        assert!(matches!(
            superimpose_rgb(&bg, 0, 4, &fg, 2, 2, &cfg),
            Err(SuperimposeError::InvalidDimensions { .. })
        ));
    }
}
