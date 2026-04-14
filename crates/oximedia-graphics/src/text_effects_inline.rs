//! Inline text outline and drop shadow rendering helpers.
//!
//! Provides [`OutlineConfig`] and [`InlineShadowConfig`] types plus rendering
//! functions that operate on alpha-only text masks and composite results into
//! RGBA pixel buffers.

use crate::image_filter;

/// Configuration for a text outline rendered around glyph shapes.
#[derive(Debug, Clone)]
pub struct OutlineConfig {
    /// Outline colour in RGBA order.
    pub color: [u8; 4],
    /// Outline thickness in pixels (dilation radius).
    pub width: u32,
    /// When `true` the outline is drawn behind the text fill; when `false` it
    /// overwrites fill pixels as well.
    pub behind_fill: bool,
}

/// Configuration for an inline drop shadow beneath text.
#[derive(Debug, Clone)]
pub struct InlineShadowConfig {
    /// Shadow colour in RGBA order.
    pub color: [u8; 4],
    /// Horizontal offset in pixels (positive = right).
    pub offset_x: i32,
    /// Vertical offset in pixels (positive = down).
    pub offset_y: i32,
    /// Gaussian blur sigma applied to the shadow mask. Zero means a hard shadow.
    pub blur_sigma: f32,
    /// Shadow opacity in `[0.0, 1.0]`.
    pub opacity: f32,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Pack an alpha-only buffer into an RGBA buffer (white RGB, alpha channel set).
fn alpha_to_rgba(alpha: &[u8], width: u32, height: u32) -> Vec<u8> {
    let len = (width as usize) * (height as usize);
    let mut rgba = vec![0u8; len * 4];
    for i in 0..len {
        if i < alpha.len() {
            rgba[i * 4] = 255;
            rgba[i * 4 + 1] = 255;
            rgba[i * 4 + 2] = 255;
            rgba[i * 4 + 3] = alpha[i];
        }
    }
    rgba
}

/// Extract alpha channel from RGBA buffer back to alpha-only.
fn rgba_to_alpha(rgba: &[u8], len: usize) -> Vec<u8> {
    let mut out = vec![0u8; len];
    for i in 0..len {
        if i * 4 + 3 < rgba.len() {
            out[i] = rgba[i * 4 + 3];
        }
    }
    out
}

/// Dilate an alpha-only buffer by `radius` pixels using the image_filter's
/// RGBA-based dilation on a temporary RGBA wrapper.
fn dilate_alpha_buffer(alpha: &[u8], width: u32, height: u32, radius: u32) -> Vec<u8> {
    if radius == 0 || width == 0 || height == 0 {
        return alpha.to_vec();
    }
    let len = (width as usize) * (height as usize);
    let mut rgba = alpha_to_rgba(alpha, width, height);
    image_filter::dilate_alpha(&mut rgba, width, height, radius);
    rgba_to_alpha(&rgba, len)
}

/// Blur an alpha-only buffer with Gaussian sigma using the image_filter's
/// RGBA-based gaussian blur on a temporary RGBA wrapper.
fn blur_alpha_buffer(alpha: &[u8], width: u32, height: u32, sigma: f32) -> Vec<u8> {
    if sigma <= 0.0 || width == 0 || height == 0 {
        return alpha.to_vec();
    }
    let len = (width as usize) * (height as usize);
    let mut rgba = alpha_to_rgba(alpha, width, height);
    image_filter::gaussian_blur(&mut rgba, width, height, sigma);
    rgba_to_alpha(&rgba, len)
}

/// Composite a coloured layer onto `pixels` using premultiplied-alpha blending.
/// `mask` is the per-pixel alpha, `color` is the flat colour, `extra_opacity`
/// is an additional multiplier (1.0 = full).
fn composite_color_mask(
    pixels: &mut [u8],
    mask: &[u8],
    width: u32,
    height: u32,
    color: [u8; 4],
    extra_opacity: f32,
) {
    let len = (width as usize) * (height as usize);
    let opacity = extra_opacity.clamp(0.0, 1.0);
    for i in 0..len {
        if i >= mask.len() {
            break;
        }
        let idx = i * 4;
        if idx + 3 >= pixels.len() {
            break;
        }
        let mask_a = mask[i] as f32 / 255.0;
        let src_a = (color[3] as f32 / 255.0) * mask_a * opacity;
        if src_a <= 0.0 {
            continue;
        }
        let inv_src_a = 1.0 - src_a;
        let dst_a = pixels[idx + 3] as f32 / 255.0;
        let out_a = src_a + dst_a * inv_src_a;
        if out_a > 0.0 {
            let blend = |sc: u8, dc: u8| -> u8 {
                let s = sc as f32 / 255.0;
                let d = dc as f32 / 255.0;
                let c = (s * src_a + d * dst_a * inv_src_a) / out_a;
                (c * 255.0).round().min(255.0).max(0.0) as u8
            };
            pixels[idx] = blend(color[0], pixels[idx]);
            pixels[idx + 1] = blend(color[1], pixels[idx + 1]);
            pixels[idx + 2] = blend(color[2], pixels[idx + 2]);
            pixels[idx + 3] = (out_a * 255.0).round().min(255.0).max(0.0) as u8;
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Render a text outline by dilating the alpha channel and compositing.
///
/// `text_alpha` is an alpha-only buffer (`width * height` bytes) representing
/// the glyph coverage mask. The outline is composited onto `pixels` (RGBA,
/// `width * height * 4` bytes).
///
/// When `config.behind_fill` is `true` only the *ring* around the original
/// glyphs is drawn (dilated minus original alpha). Otherwise the full dilated
/// area, including the text interior, receives the outline colour.
pub fn render_outline(
    text_alpha: &[u8],
    pixels: &mut [u8],
    width: u32,
    height: u32,
    config: &OutlineConfig,
) {
    if config.width == 0 || width == 0 || height == 0 {
        return;
    }
    let len = (width as usize) * (height as usize);

    // Dilate the alpha mask to get the outline region.
    let dilated = dilate_alpha_buffer(text_alpha, width, height, config.width);

    // Build the effective mask: if behind_fill, subtract original text alpha.
    let mask = if config.behind_fill {
        let mut m = dilated;
        for i in 0..len.min(m.len()).min(text_alpha.len()) {
            m[i] = m[i].saturating_sub(text_alpha[i]);
        }
        m
    } else {
        dilated
    };

    composite_color_mask(pixels, &mask, width, height, config.color, 1.0);
}

/// Render a drop shadow by blurring and offsetting the alpha channel.
///
/// `text_alpha` is an alpha-only buffer (`width * height` bytes). The shadow
/// is composited onto `pixels` (RGBA, `width * height * 4` bytes).
pub fn render_shadow(
    text_alpha: &[u8],
    pixels: &mut [u8],
    width: u32,
    height: u32,
    config: &InlineShadowConfig,
) {
    if width == 0 || height == 0 || config.opacity <= 0.0 {
        return;
    }
    let w = width as usize;
    let h = height as usize;
    let len = w * h;

    // Step 1: offset the alpha mask.
    let mut shifted = vec![0u8; len];
    for dst_y in 0..h {
        let src_y = dst_y as i64 - config.offset_y as i64;
        if src_y < 0 || src_y >= h as i64 {
            continue;
        }
        for dst_x in 0..w {
            let src_x = dst_x as i64 - config.offset_x as i64;
            if src_x < 0 || src_x >= w as i64 {
                continue;
            }
            let si = src_y as usize * w + src_x as usize;
            let di = dst_y * w + dst_x;
            if si < text_alpha.len() {
                shifted[di] = text_alpha[si];
            }
        }
    }

    // Step 2: blur the shifted mask.
    let blurred = blur_alpha_buffer(&shifted, width, height, config.blur_sigma);

    // Step 3: composite onto pixels.
    composite_color_mask(pixels, &blurred, width, height, config.color, config.opacity);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_outline_config_default() {
        let cfg = OutlineConfig {
            color: [255, 0, 0, 255],
            width: 2,
            behind_fill: true,
        };
        assert_eq!(cfg.color, [255, 0, 0, 255]);
        assert_eq!(cfg.width, 2);
        assert!(cfg.behind_fill);
    }

    #[test]
    fn test_shadow_config_default() {
        let cfg = InlineShadowConfig {
            color: [0, 0, 0, 200],
            offset_x: 3,
            offset_y: 4,
            blur_sigma: 2.5,
            opacity: 0.8,
        };
        assert_eq!(cfg.color, [0, 0, 0, 200]);
        assert_eq!(cfg.offset_x, 3);
        assert_eq!(cfg.offset_y, 4);
        assert!((cfg.blur_sigma - 2.5).abs() < f32::EPSILON);
        assert!((cfg.opacity - 0.8).abs() < f32::EPSILON);
    }

    /// A filled square of alpha should grow after outline rendering.
    #[test]
    fn test_render_outline_expands_area() {
        let (w, h): (u32, u32) = (20, 20);
        let len = (w * h) as usize;
        // 6x6 filled square in the centre (rows 7..13, cols 7..13)
        let mut alpha = vec![0u8; len];
        for y in 7..13u32 {
            for x in 7..13u32 {
                alpha[(y * w + x) as usize] = 255;
            }
        }
        let original_nonzero = alpha.iter().filter(|&&a| a > 0).count();

        let mut pixels = vec![0u8; len * 4];
        let cfg = OutlineConfig {
            color: [255, 0, 0, 255],
            width: 2,
            behind_fill: false,
        };
        render_outline(&alpha, &mut pixels, w, h, &cfg);

        let rendered_nonzero = (0..len).filter(|&i| pixels[i * 4 + 3] > 0).count();
        assert!(
            rendered_nonzero > original_nonzero,
            "outline should expand: rendered {} vs original {}",
            rendered_nonzero,
            original_nonzero,
        );
    }

    /// Outline pixels should carry the configured colour.
    #[test]
    fn test_render_outline_color() {
        let (w, h): (u32, u32) = (16, 16);
        let len = (w * h) as usize;
        let mut alpha = vec![0u8; len];
        // single pixel in centre
        alpha[(8 * w + 8) as usize] = 255;

        let mut pixels = vec![0u8; len * 4];
        let cfg = OutlineConfig {
            color: [0, 255, 0, 255],
            width: 1,
            behind_fill: false,
        };
        render_outline(&alpha, &mut pixels, w, h, &cfg);

        // Find a non-zero pixel that is NOT the original centre
        let mut found_green = false;
        for i in 0..len {
            if i == (8 * w + 8) as usize {
                continue;
            }
            if pixels[i * 4 + 3] > 0 {
                assert_eq!(pixels[i * 4], 0, "R should be 0");
                assert_eq!(pixels[i * 4 + 1], 255, "G should be 255");
                assert_eq!(pixels[i * 4 + 2], 0, "B should be 0");
                found_green = true;
                break;
            }
        }
        assert!(found_green, "should find a green outline pixel");
    }

    /// Shadow should appear shifted by the configured offset.
    #[test]
    fn test_render_shadow_offset() {
        let (w, h): (u32, u32) = (20, 20);
        let len = (w * h) as usize;
        let mut alpha = vec![0u8; len];
        // single pixel at (5, 5)
        alpha[(5 * w + 5) as usize] = 255;

        let mut pixels = vec![0u8; len * 4];
        let cfg = InlineShadowConfig {
            color: [0, 0, 0, 255],
            offset_x: 3,
            offset_y: 4,
            blur_sigma: 0.0,
            opacity: 1.0,
        };
        render_shadow(&alpha, &mut pixels, w, h, &cfg);

        // Shadow pixel should be at (5+3, 5+4) = (8, 9)
        let shadow_idx = (9 * w + 8) as usize;
        assert!(
            pixels[shadow_idx * 4 + 3] > 0,
            "shadow pixel at (8,9) should be non-transparent"
        );
    }

    /// With sigma=0 the shadow should have the same shape as the input (shifted).
    #[test]
    fn test_render_shadow_blur_zero_hard() {
        let (w, h): (u32, u32) = (16, 16);
        let len = (w * h) as usize;
        let mut alpha = vec![0u8; len];
        alpha[(4 * w + 4) as usize] = 255;

        let mut pixels = vec![0u8; len * 4];
        let cfg = InlineShadowConfig {
            color: [0, 0, 0, 255],
            offset_x: 2,
            offset_y: 2,
            blur_sigma: 0.0,
            opacity: 1.0,
        };
        render_shadow(&alpha, &mut pixels, w, h, &cfg);

        // Exactly one non-transparent pixel at (6, 6)
        let target = (6 * w + 6) as usize;
        let nonzero_count = (0..len).filter(|&i| pixels[i * 4 + 3] > 0).count();
        assert_eq!(nonzero_count, 1, "hard shadow should produce exactly one pixel");
        assert!(pixels[target * 4 + 3] > 0);
    }

    /// Opacity 0 should produce no visible shadow.
    #[test]
    fn test_render_shadow_opacity_zero() {
        let (w, h): (u32, u32) = (10, 10);
        let len = (w * h) as usize;
        let mut alpha = vec![0u8; len];
        alpha[(5 * w + 5) as usize] = 255;

        let mut pixels = vec![0u8; len * 4];
        let cfg = InlineShadowConfig {
            color: [0, 0, 0, 255],
            offset_x: 1,
            offset_y: 1,
            blur_sigma: 1.0,
            opacity: 0.0,
        };
        render_shadow(&alpha, &mut pixels, w, h, &cfg);

        let any_visible = (0..len).any(|i| pixels[i * 4 + 3] > 0);
        assert!(!any_visible, "opacity=0 should produce no visible pixels");
    }

    /// Outline width 0 should not change the output buffer.
    #[test]
    fn test_render_outline_width_zero() {
        let (w, h): (u32, u32) = (10, 10);
        let len = (w * h) as usize;
        let mut alpha = vec![0u8; len];
        alpha[(5 * w + 5) as usize] = 255;

        let mut pixels = vec![0u8; len * 4];
        let cfg = OutlineConfig {
            color: [255, 255, 255, 255],
            width: 0,
            behind_fill: false,
        };
        render_outline(&alpha, &mut pixels, w, h, &cfg);

        let any_visible = (0..len).any(|i| pixels[i * 4 + 3] > 0);
        assert!(!any_visible, "width=0 outline should produce no output");
    }

    /// Empty buffers should not cause panics.
    #[test]
    fn test_render_no_panic_empty() {
        let alpha: Vec<u8> = vec![];
        let mut pixels: Vec<u8> = vec![];

        let outline_cfg = OutlineConfig {
            color: [255, 0, 0, 255],
            width: 2,
            behind_fill: true,
        };
        render_outline(&alpha, &mut pixels, 0, 0, &outline_cfg);

        let shadow_cfg = InlineShadowConfig {
            color: [0, 0, 0, 128],
            offset_x: 2,
            offset_y: 2,
            blur_sigma: 1.5,
            opacity: 0.5,
        };
        render_shadow(&alpha, &mut pixels, 0, 0, &shadow_cfg);
        // No panic means pass.
    }

    /// Shadow and outline can be applied sequentially without issues.
    #[test]
    fn test_render_shadow_and_outline_combined() {
        let (w, h): (u32, u32) = (20, 20);
        let len = (w * h) as usize;
        let mut alpha = vec![0u8; len];
        for y in 8..12u32 {
            for x in 8..12u32 {
                alpha[(y * w + x) as usize] = 255;
            }
        }

        let mut pixels = vec![0u8; len * 4];

        // First: shadow (drawn behind)
        let shadow_cfg = InlineShadowConfig {
            color: [0, 0, 0, 200],
            offset_x: 2,
            offset_y: 2,
            blur_sigma: 1.0,
            opacity: 0.7,
        };
        render_shadow(&alpha, &mut pixels, w, h, &shadow_cfg);

        let after_shadow = (0..len).filter(|&i| pixels[i * 4 + 3] > 0).count();

        // Then: outline
        let outline_cfg = OutlineConfig {
            color: [255, 255, 0, 255],
            width: 1,
            behind_fill: true,
        };
        render_outline(&alpha, &mut pixels, w, h, &outline_cfg);

        let after_both = (0..len).filter(|&i| pixels[i * 4 + 3] > 0).count();
        assert!(
            after_both >= after_shadow,
            "combined should have at least as many visible pixels: both={} shadow={}",
            after_both,
            after_shadow,
        );
    }

    /// Outline with behind_fill=true should not colour the interior.
    #[test]
    fn test_render_outline_behind_fill_ring() {
        let (w, h): (u32, u32) = (20, 20);
        let len = (w * h) as usize;
        let mut alpha = vec![0u8; len];
        for y in 8..12u32 {
            for x in 8..12u32 {
                alpha[(y * w + x) as usize] = 255;
            }
        }

        let mut pixels = vec![0u8; len * 4];
        let cfg = OutlineConfig {
            color: [255, 0, 0, 255],
            width: 2,
            behind_fill: true,
        };
        render_outline(&alpha, &mut pixels, w, h, &cfg);

        // Interior pixels (where alpha was 255) should be untouched (alpha=0).
        for y in 8..12u32 {
            for x in 8..12u32 {
                let idx = (y * w + x) as usize;
                assert_eq!(
                    pixels[idx * 4 + 3], 0,
                    "behind_fill interior at ({},{}) should remain transparent",
                    x, y
                );
            }
        }
        // But surrounding ring should have content.
        let ring_count = (0..len)
            .filter(|&i| pixels[i * 4 + 3] > 0)
            .count();
        assert!(ring_count > 0, "ring should have visible pixels");
    }

    /// Negative shadow offsets work correctly (shadow to upper-left).
    #[test]
    fn test_render_shadow_negative_offset() {
        let (w, h): (u32, u32) = (20, 20);
        let len = (w * h) as usize;
        let mut alpha = vec![0u8; len];
        alpha[(10 * w + 10) as usize] = 255;

        let mut pixels = vec![0u8; len * 4];
        let cfg = InlineShadowConfig {
            color: [0, 0, 0, 255],
            offset_x: -3,
            offset_y: -3,
            blur_sigma: 0.0,
            opacity: 1.0,
        };
        render_shadow(&alpha, &mut pixels, w, h, &cfg);

        // Shadow should appear at (10-3, 10-3) = (7, 7)
        let target = (7 * w + 7) as usize;
        assert!(
            pixels[target * 4 + 3] > 0,
            "negative offset shadow at (7,7) should be visible"
        );
    }
}
