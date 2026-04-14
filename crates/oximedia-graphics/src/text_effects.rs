//! Text effect rasterization for broadcast graphics.
//!
//! Provides outline, drop shadow, and glow effects for rasterized glyph bitmaps.
//! Effects are composited in a specific order (shadow → glow → outline → fill)
//! to produce professional broadcast-quality text overlays.

/// Text outline configuration.
#[derive(Debug, Clone)]
pub struct TextOutlineConfig {
    /// Outline width in pixels.
    pub width_px: f32,
    /// RGBA outline color.
    pub color: [u8; 4],
    /// Number of offset samples around the circle (default 16).
    pub samples: u32,
}

impl Default for TextOutlineConfig {
    fn default() -> Self {
        Self {
            width_px: 2.0,
            color: [0, 0, 0, 255],
            samples: 16,
        }
    }
}

/// Drop shadow configuration.
#[derive(Debug, Clone)]
pub struct DropShadowConfig {
    /// Horizontal shadow offset in pixels.
    pub offset_x: f32,
    /// Vertical shadow offset in pixels.
    pub offset_y: f32,
    /// Blur radius (0 = hard shadow).
    pub blur_radius: f32,
    /// RGBA shadow color.
    pub color: [u8; 4],
    /// Shadow opacity (0.0–1.0).
    pub opacity: f32,
}

impl Default for DropShadowConfig {
    fn default() -> Self {
        Self {
            offset_x: 2.0,
            offset_y: 2.0,
            blur_radius: 3.0,
            color: [0, 0, 0, 255],
            opacity: 0.7,
        }
    }
}

/// Text glow configuration.
#[derive(Debug, Clone)]
pub struct TextGlowConfig {
    /// Glow spread radius in pixels.
    pub radius: f32,
    /// RGBA glow color.
    pub color: [u8; 4],
    /// Glow intensity (0.0–1.0).
    pub intensity: f32,
}

impl Default for TextGlowConfig {
    fn default() -> Self {
        Self {
            radius: 4.0,
            color: [255, 255, 128, 255],
            intensity: 0.6,
        }
    }
}

/// Combined text effect stack applied in order: shadow → glow → outline → fill.
#[derive(Debug, Clone, Default)]
pub struct TextEffectStack {
    /// Optional drop shadow.
    pub shadow: Option<DropShadowConfig>,
    /// Optional text outline.
    pub outline: Option<TextOutlineConfig>,
    /// Optional text glow.
    pub glow: Option<TextGlowConfig>,
}

/// A rasterized glyph bitmap for effect processing.
#[derive(Debug, Clone)]
pub struct GlyphBitmap {
    /// Alpha channel coverage values (one byte per pixel).
    pub data: Vec<u8>,
    /// Bitmap width in pixels.
    pub width: u32,
    /// Bitmap height in pixels.
    pub height: u32,
    /// Horizontal position in the target buffer.
    pub x: i32,
    /// Vertical position in the target buffer.
    pub y: i32,
}

impl GlyphBitmap {
    /// Create a new glyph bitmap.
    pub fn new(data: Vec<u8>, width: u32, height: u32, x: i32, y: i32) -> Self {
        Self {
            data,
            width,
            height,
            x,
            y,
        }
    }

    /// Returns true if this glyph has zero area.
    pub fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0 || self.data.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Alpha blending helper
// ---------------------------------------------------------------------------

/// Source-over alpha composite a single pixel.
///
/// `dst` must be a 4-byte RGBA slice. `coverage` is the glyph alpha value
/// that modulates `src_color`'s alpha channel.
fn blend_pixel(dst: &mut [u8], src_color: [u8; 4], coverage: u8) {
    if coverage == 0 || src_color[3] == 0 {
        return;
    }

    let src_a = (src_color[3] as u32 * coverage as u32) / 255;
    if src_a == 0 {
        return;
    }

    let dst_a = dst[3] as u32;
    let inv_src_a = 255 - src_a;

    let out_a = src_a + (dst_a * inv_src_a) / 255;
    if out_a == 0 {
        return;
    }

    for i in 0..3 {
        let src_c = src_color[i] as u32 * src_a;
        let dst_c = dst[i] as u32 * dst_a * inv_src_a / 255;
        dst[i] = ((src_c + dst_c) / out_a).min(255) as u8;
    }
    dst[3] = out_a.min(255) as u8;
}

// ---------------------------------------------------------------------------
// Stamp a glyph into a target buffer
// ---------------------------------------------------------------------------

/// Stamp a glyph bitmap into `target` at an offset, using the given color.
///
/// `target` is an RGBA buffer of dimensions `stride/4` wide.
fn stamp_glyph(
    glyph: &GlyphBitmap,
    offset_x: i32,
    offset_y: i32,
    color: [u8; 4],
    target: &mut [u8],
    stride: u32,
) {
    if glyph.is_empty() {
        return;
    }

    let target_width = stride / 4;
    let target_height = target.len() as u32 / stride;

    for gy in 0..glyph.height {
        let ty = glyph.y + offset_y + gy as i32;
        if ty < 0 || ty >= target_height as i32 {
            continue;
        }
        for gx in 0..glyph.width {
            let tx = glyph.x + offset_x + gx as i32;
            if tx < 0 || tx >= target_width as i32 {
                continue;
            }

            let glyph_idx = (gy * glyph.width + gx) as usize;
            let coverage = match glyph.data.get(glyph_idx) {
                Some(&v) => v,
                None => continue,
            };

            let pixel_offset = (ty as u32 * stride + tx as u32 * 4) as usize;
            if pixel_offset + 4 <= target.len() {
                blend_pixel(&mut target[pixel_offset..pixel_offset + 4], color, coverage);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Separable box blur
// ---------------------------------------------------------------------------

/// Apply a separable box blur to an alpha buffer.
///
/// Runs two passes (horizontal then vertical) repeated `iterations` times
/// to approximate a Gaussian blur.
fn box_blur_alpha(
    buf: &mut Vec<u8>,
    width: u32,
    height: u32,
    radius: u32,
    iterations: u32,
) {
    if radius == 0 || width == 0 || height == 0 {
        return;
    }
    let w = width as usize;
    let h = height as usize;
    let r = radius as usize;

    let mut tmp = vec![0u8; buf.len()];

    for _ in 0..iterations {
        // Horizontal pass: buf → tmp
        for y in 0..h {
            let row = y * w;
            let mut sum: u32 = 0;
            let mut count: u32 = 0;

            // Initialise window
            for kx in 0..=r.min(w.saturating_sub(1)) {
                sum += buf[row + kx] as u32;
                count += 1;
            }
            tmp[row] = (sum / count.max(1)) as u8;

            for x in 1..w {
                // Add right edge
                let add_x = x + r;
                if add_x < w {
                    sum += buf[row + add_x] as u32;
                    count += 1;
                }
                // Remove left edge
                if x > r {
                    let rem_x = x - r - 1;
                    sum -= buf[row + rem_x] as u32;
                    count -= 1;
                }
                tmp[row + x] = (sum / count.max(1)) as u8;
            }
        }

        // Vertical pass: tmp → buf
        for x in 0..w {
            let mut sum: u32 = 0;
            let mut count: u32 = 0;

            for ky in 0..=r.min(h.saturating_sub(1)) {
                sum += tmp[ky * w + x] as u32;
                count += 1;
            }
            buf[x] = (sum / count.max(1)) as u8;

            for y in 1..h {
                let add_y = y + r;
                if add_y < h {
                    sum += tmp[add_y * w + x] as u32;
                    count += 1;
                }
                if y > r {
                    let rem_y = y - r - 1;
                    sum -= tmp[rem_y * w + x] as u32;
                    count -= 1;
                }
                buf[y * w + x] = (sum / count.max(1)) as u8;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Public effect renderers
// ---------------------------------------------------------------------------

/// Render a text outline using the multi-stamp approach.
///
/// The glyph is stamped at `samples` evenly-spaced positions around a circle
/// of `width_px` radius in the outline color, then the original fill is drawn
/// on top.
pub fn render_outline(
    glyph: &GlyphBitmap,
    config: &TextOutlineConfig,
    target: &mut [u8],
    stride: u32,
) {
    if glyph.is_empty() || config.width_px <= 0.0 || config.samples == 0 {
        return;
    }

    let samples = config.samples.max(4);
    let step = std::f32::consts::TAU / samples as f32;

    for i in 0..samples {
        let angle = step * i as f32;
        let ox = (angle.cos() * config.width_px).round() as i32;
        let oy = (angle.sin() * config.width_px).round() as i32;
        stamp_glyph(glyph, ox, oy, config.color, target, stride);
    }
}

/// Render a drop shadow beneath a glyph.
///
/// When `blur_radius > 0`, a separable box blur (2 iterations) is applied to
/// approximate a Gaussian blur.
pub fn render_shadow(
    glyph: &GlyphBitmap,
    config: &DropShadowConfig,
    target: &mut [u8],
    stride: u32,
) {
    if glyph.is_empty() || config.opacity <= 0.0 {
        return;
    }

    let mut shadow_color = config.color;
    shadow_color[3] = (shadow_color[3] as f32 * config.opacity.clamp(0.0, 1.0)) as u8;

    let ox = config.offset_x.round() as i32;
    let oy = config.offset_y.round() as i32;

    if config.blur_radius <= 0.5 {
        // Hard shadow — just stamp at offset
        stamp_glyph(glyph, ox, oy, shadow_color, target, stride);
        return;
    }

    // Blurred shadow: rasterize into a temp alpha buffer, blur, then composite
    let blur_r = config.blur_radius.ceil() as u32;
    let pad = blur_r + 1;
    let buf_w = glyph.width + pad * 2;
    let buf_h = glyph.height + pad * 2;
    let mut alpha_buf = vec![0u8; (buf_w * buf_h) as usize];

    // Copy glyph alpha into centre of buffer
    for gy in 0..glyph.height {
        for gx in 0..glyph.width {
            let src_idx = (gy * glyph.width + gx) as usize;
            let dst_idx = ((gy + pad) * buf_w + (gx + pad)) as usize;
            if let Some(&v) = glyph.data.get(src_idx) {
                if let Some(dst) = alpha_buf.get_mut(dst_idx) {
                    *dst = v;
                }
            }
        }
    }

    box_blur_alpha(&mut alpha_buf, buf_w, buf_h, blur_r, 2);

    // Composite blurred shadow into target
    let target_width = stride / 4;
    let target_height = (target.len() as u32).checked_div(stride).unwrap_or(0);

    for by in 0..buf_h {
        let ty = glyph.y + oy - pad as i32 + by as i32;
        if ty < 0 || ty >= target_height as i32 {
            continue;
        }
        for bx in 0..buf_w {
            let tx = glyph.x + ox - pad as i32 + bx as i32;
            if tx < 0 || tx >= target_width as i32 {
                continue;
            }

            let alpha_idx = (by * buf_w + bx) as usize;
            let coverage = match alpha_buf.get(alpha_idx) {
                Some(&v) => v,
                None => continue,
            };

            let pixel_off = (ty as u32 * stride + tx as u32 * 4) as usize;
            if pixel_off + 4 <= target.len() {
                blend_pixel(&mut target[pixel_off..pixel_off + 4], shadow_color, coverage);
            }
        }
    }
}

/// Render a glow effect around a glyph.
///
/// A glow is functionally a drop shadow with zero offset and `intensity` as
/// opacity, spread over the glow `radius`.
pub fn render_glow(
    glyph: &GlyphBitmap,
    config: &TextGlowConfig,
    target: &mut [u8],
    stride: u32,
) {
    if glyph.is_empty() || config.intensity <= 0.0 {
        return;
    }

    let shadow_cfg = DropShadowConfig {
        offset_x: 0.0,
        offset_y: 0.0,
        blur_radius: config.radius,
        color: config.color,
        opacity: config.intensity.clamp(0.0, 1.0),
    };
    render_shadow(glyph, &shadow_cfg, target, stride);
}

/// Render glyphs with a full effect stack.
///
/// Render order: shadow → glow → outline → fill.
pub fn render_text_with_effects(
    glyphs: &[GlyphBitmap],
    fill_color: [u8; 4],
    effects: &TextEffectStack,
    target: &mut [u8],
    width: u32,
    _height: u32,
) {
    let stride = width * 4;

    // 1. Shadow pass
    if let Some(ref shadow) = effects.shadow {
        for glyph in glyphs {
            render_shadow(glyph, shadow, target, stride);
        }
    }

    // 2. Glow pass
    if let Some(ref glow) = effects.glow {
        for glyph in glyphs {
            render_glow(glyph, glow, target, stride);
        }
    }

    // 3. Outline pass
    if let Some(ref outline) = effects.outline {
        for glyph in glyphs {
            render_outline(glyph, outline, target, stride);
        }
    }

    // 4. Fill pass
    for glyph in glyphs {
        stamp_glyph(glyph, 0, 0, fill_color, target, stride);
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a small test glyph (4x4 filled square).
    fn test_glyph(x: i32, y: i32) -> GlyphBitmap {
        GlyphBitmap::new(vec![255; 16], 4, 4, x, y)
    }

    /// Helper: create a blank RGBA target buffer.
    fn make_target(w: u32, h: u32) -> (Vec<u8>, u32) {
        let stride = w * 4;
        (vec![0u8; (stride * h) as usize], stride)
    }

    /// Helper: check that at least one non-zero pixel exists.
    fn has_nonzero_alpha(target: &[u8]) -> bool {
        target.chunks_exact(4).any(|p| p[3] > 0)
    }

    // -----------------------------------------------------------------------
    // Alpha blending
    // -----------------------------------------------------------------------

    #[test]
    fn test_blend_pixel_full_coverage() {
        let mut dst = [0u8, 0, 0, 0];
        blend_pixel(&mut dst, [255, 0, 0, 255], 255);
        assert_eq!(dst[0], 255); // red
        assert_eq!(dst[3], 255); // alpha
    }

    #[test]
    fn test_blend_pixel_zero_coverage() {
        let mut dst = [128, 128, 128, 255];
        let original = dst;
        blend_pixel(&mut dst, [255, 0, 0, 255], 0);
        assert_eq!(dst, original);
    }

    #[test]
    fn test_blend_pixel_zero_src_alpha() {
        let mut dst = [128, 128, 128, 255];
        let original = dst;
        blend_pixel(&mut dst, [255, 0, 0, 0], 128);
        assert_eq!(dst, original);
    }

    #[test]
    fn test_blend_pixel_partial_coverage() {
        let mut dst = [0u8, 0, 0, 0];
        blend_pixel(&mut dst, [255, 255, 255, 255], 128);
        // Should produce roughly half-alpha white
        assert!(dst[3] > 0);
        assert!(dst[3] < 255);
    }

    #[test]
    fn test_blend_over_existing() {
        let mut dst = [0, 0, 255, 255]; // blue background
        blend_pixel(&mut dst, [255, 0, 0, 255], 128);
        // Red blended over blue
        assert!(dst[0] > 0); // some red
        assert!(dst[3] == 255); // still full alpha
    }

    // -----------------------------------------------------------------------
    // GlyphBitmap
    // -----------------------------------------------------------------------

    #[test]
    fn test_glyph_empty() {
        let g = GlyphBitmap::new(vec![], 0, 0, 0, 0);
        assert!(g.is_empty());
    }

    #[test]
    fn test_glyph_not_empty() {
        let g = test_glyph(0, 0);
        assert!(!g.is_empty());
    }

    #[test]
    fn test_glyph_zero_width() {
        let g = GlyphBitmap::new(vec![255; 4], 0, 4, 0, 0);
        assert!(g.is_empty());
    }

    // -----------------------------------------------------------------------
    // Outline rendering
    // -----------------------------------------------------------------------

    #[test]
    fn test_outline_1px() {
        let (mut target, stride) = make_target(20, 20);
        let glyph = test_glyph(8, 8);
        let config = TextOutlineConfig {
            width_px: 1.0,
            color: [255, 0, 0, 255],
            samples: 16,
        };
        render_outline(&glyph, &config, &mut target, stride);
        assert!(has_nonzero_alpha(&target));
    }

    #[test]
    fn test_outline_3px() {
        let (mut target, stride) = make_target(30, 30);
        let glyph = test_glyph(10, 10);
        let config = TextOutlineConfig {
            width_px: 3.0,
            color: [0, 255, 0, 255],
            samples: 16,
        };
        render_outline(&glyph, &config, &mut target, stride);
        // The outline should extend beyond the glyph bounds
        // Check a pixel that is 3px left of the glyph origin
        let check_x = 7u32; // glyph.x(10) - 3
        let check_y = 12u32;
        let idx = (check_y * stride + check_x * 4) as usize;
        assert!(target[idx + 3] > 0, "outline should extend 3px left");
    }

    #[test]
    fn test_outline_5px() {
        let (mut target, stride) = make_target(40, 40);
        let glyph = test_glyph(15, 15);
        let config = TextOutlineConfig {
            width_px: 5.0,
            color: [0, 0, 255, 255],
            samples: 32,
        };
        render_outline(&glyph, &config, &mut target, stride);
        assert!(has_nonzero_alpha(&target));
    }

    #[test]
    fn test_outline_zero_width() {
        let (mut target, stride) = make_target(20, 20);
        let glyph = test_glyph(5, 5);
        let config = TextOutlineConfig {
            width_px: 0.0,
            ..Default::default()
        };
        render_outline(&glyph, &config, &mut target, stride);
        // Zero width should produce no output
        assert!(!has_nonzero_alpha(&target));
    }

    #[test]
    fn test_outline_empty_glyph() {
        let (mut target, stride) = make_target(20, 20);
        let glyph = GlyphBitmap::new(vec![], 0, 0, 5, 5);
        let config = TextOutlineConfig::default();
        render_outline(&glyph, &config, &mut target, stride);
        assert!(!has_nonzero_alpha(&target));
    }

    // -----------------------------------------------------------------------
    // Shadow rendering
    // -----------------------------------------------------------------------

    #[test]
    fn test_shadow_hard() {
        let (mut target, stride) = make_target(20, 20);
        let glyph = test_glyph(5, 5);
        let config = DropShadowConfig {
            offset_x: 3.0,
            offset_y: 3.0,
            blur_radius: 0.0,
            color: [0, 0, 0, 255],
            opacity: 1.0,
        };
        render_shadow(&glyph, &config, &mut target, stride);
        // Check that shadow appears at offset position (8,8)
        let idx = (8 * stride + 8 * 4) as usize;
        assert!(target[idx + 3] > 0, "hard shadow should appear at offset");
    }

    #[test]
    fn test_shadow_blurred() {
        let (mut target, stride) = make_target(30, 30);
        let glyph = test_glyph(10, 10);
        let config = DropShadowConfig {
            offset_x: 2.0,
            offset_y: 2.0,
            blur_radius: 4.0,
            color: [0, 0, 0, 255],
            opacity: 0.8,
        };
        render_shadow(&glyph, &config, &mut target, stride);
        assert!(has_nonzero_alpha(&target));
    }

    #[test]
    fn test_shadow_zero_opacity() {
        let (mut target, stride) = make_target(20, 20);
        let glyph = test_glyph(5, 5);
        let config = DropShadowConfig {
            opacity: 0.0,
            ..Default::default()
        };
        render_shadow(&glyph, &config, &mut target, stride);
        assert!(!has_nonzero_alpha(&target));
    }

    #[test]
    fn test_shadow_empty_glyph() {
        let (mut target, stride) = make_target(20, 20);
        let glyph = GlyphBitmap::new(vec![], 0, 0, 5, 5);
        let config = DropShadowConfig::default();
        render_shadow(&glyph, &config, &mut target, stride);
        assert!(!has_nonzero_alpha(&target));
    }

    // -----------------------------------------------------------------------
    // Glow rendering
    // -----------------------------------------------------------------------

    #[test]
    fn test_glow_basic() {
        let (mut target, stride) = make_target(30, 30);
        let glyph = test_glyph(10, 10);
        let config = TextGlowConfig {
            radius: 3.0,
            color: [255, 255, 0, 255],
            intensity: 0.8,
        };
        render_glow(&glyph, &config, &mut target, stride);
        assert!(has_nonzero_alpha(&target));
    }

    #[test]
    fn test_glow_zero_intensity() {
        let (mut target, stride) = make_target(20, 20);
        let glyph = test_glyph(5, 5);
        let config = TextGlowConfig {
            radius: 3.0,
            color: [255, 255, 0, 255],
            intensity: 0.0,
        };
        render_glow(&glyph, &config, &mut target, stride);
        assert!(!has_nonzero_alpha(&target));
    }

    #[test]
    fn test_glow_empty_glyph() {
        let (mut target, stride) = make_target(20, 20);
        let glyph = GlyphBitmap::new(vec![], 0, 0, 5, 5);
        let config = TextGlowConfig::default();
        render_glow(&glyph, &config, &mut target, stride);
        assert!(!has_nonzero_alpha(&target));
    }

    // -----------------------------------------------------------------------
    // Box blur
    // -----------------------------------------------------------------------

    #[test]
    fn test_box_blur_single_pixel() {
        let mut buf = vec![0u8; 25]; // 5x5
        buf[12] = 255; // centre pixel
        box_blur_alpha(&mut buf, 5, 5, 1, 1);
        // Centre should have decreased, neighbours should have increased
        assert!(buf[12] < 255);
        assert!(buf[11] > 0); // left of centre
        assert!(buf[13] > 0); // right of centre
    }

    #[test]
    fn test_box_blur_zero_radius() {
        let mut buf = vec![128u8; 9]; // 3x3
        let original = buf.clone();
        box_blur_alpha(&mut buf, 3, 3, 0, 2);
        assert_eq!(buf, original);
    }

    #[test]
    fn test_box_blur_uniform_unchanged() {
        let mut buf = vec![100u8; 16]; // 4x4 uniform
        box_blur_alpha(&mut buf, 4, 4, 1, 2);
        // Blurring a uniform image should keep it roughly uniform
        for &v in &buf {
            assert!((v as i32 - 100).unsigned_abs() <= 1);
        }
    }

    // -----------------------------------------------------------------------
    // Combined effect stack
    // -----------------------------------------------------------------------

    #[test]
    fn test_effect_stack_all_effects() {
        let glyphs = vec![test_glyph(10, 10)];
        let effects = TextEffectStack {
            shadow: Some(DropShadowConfig::default()),
            outline: Some(TextOutlineConfig::default()),
            glow: Some(TextGlowConfig::default()),
        };
        let w = 30u32;
        let h = 30u32;
        let mut target = vec![0u8; (w * h * 4) as usize];
        render_text_with_effects(&glyphs, [255, 255, 255, 255], &effects, &mut target, w, h);
        assert!(has_nonzero_alpha(&target));
    }

    #[test]
    fn test_effect_stack_shadow_only() {
        let glyphs = vec![test_glyph(8, 8)];
        let effects = TextEffectStack {
            shadow: Some(DropShadowConfig::default()),
            outline: None,
            glow: None,
        };
        let w = 20u32;
        let h = 20u32;
        let mut target = vec![0u8; (w * h * 4) as usize];
        render_text_with_effects(&glyphs, [255, 255, 255, 255], &effects, &mut target, w, h);
        assert!(has_nonzero_alpha(&target));
    }

    #[test]
    fn test_effect_stack_no_effects() {
        let glyphs = vec![test_glyph(5, 5)];
        let effects = TextEffectStack::default();
        let w = 20u32;
        let h = 20u32;
        let mut target = vec![0u8; (w * h * 4) as usize];
        render_text_with_effects(&glyphs, [255, 255, 255, 255], &effects, &mut target, w, h);
        // Fill only — should still produce output
        assert!(has_nonzero_alpha(&target));
    }

    #[test]
    fn test_effect_stack_empty_glyphs() {
        let glyphs: Vec<GlyphBitmap> = vec![];
        let effects = TextEffectStack {
            shadow: Some(DropShadowConfig::default()),
            outline: Some(TextOutlineConfig::default()),
            glow: Some(TextGlowConfig::default()),
        };
        let w = 20u32;
        let h = 20u32;
        let mut target = vec![0u8; (w * h * 4) as usize];
        render_text_with_effects(&glyphs, [255, 255, 255, 255], &effects, &mut target, w, h);
        assert!(!has_nonzero_alpha(&target));
    }

    #[test]
    fn test_effect_stack_multiple_glyphs() {
        let glyphs = vec![test_glyph(5, 5), test_glyph(12, 5), test_glyph(19, 5)];
        let effects = TextEffectStack {
            shadow: Some(DropShadowConfig {
                offset_x: 1.0,
                offset_y: 1.0,
                blur_radius: 2.0,
                color: [0, 0, 0, 255],
                opacity: 0.5,
            }),
            outline: Some(TextOutlineConfig {
                width_px: 1.0,
                color: [0, 0, 0, 255],
                samples: 8,
            }),
            glow: None,
        };
        let w = 30u32;
        let h = 20u32;
        let mut target = vec![0u8; (w * h * 4) as usize];
        render_text_with_effects(&glyphs, [255, 0, 0, 255], &effects, &mut target, w, h);
        assert!(has_nonzero_alpha(&target));
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_glyph_off_screen_negative() {
        let (mut target, stride) = make_target(10, 10);
        let glyph = test_glyph(-20, -20);
        stamp_glyph(&glyph, 0, 0, [255, 255, 255, 255], &mut target, stride);
        assert!(!has_nonzero_alpha(&target));
    }

    #[test]
    fn test_glyph_off_screen_positive() {
        let (mut target, stride) = make_target(10, 10);
        let glyph = test_glyph(100, 100);
        stamp_glyph(&glyph, 0, 0, [255, 255, 255, 255], &mut target, stride);
        assert!(!has_nonzero_alpha(&target));
    }

    #[test]
    fn test_glyph_partial_clip() {
        let (mut target, stride) = make_target(10, 10);
        // Glyph starts at (-2, -2) so only bottom-right portion is visible
        let glyph = test_glyph(-2, -2);
        stamp_glyph(&glyph, 0, 0, [255, 255, 255, 255], &mut target, stride);
        // Pixels (0,0) and (1,1) should have content
        let idx = 0usize;
        assert!(target[idx + 3] > 0, "partial clip should render visible portion");
    }

    #[test]
    fn test_default_configs() {
        let _outline = TextOutlineConfig::default();
        let _shadow = DropShadowConfig::default();
        let _glow = TextGlowConfig::default();
        let _stack = TextEffectStack::default();
    }

    #[test]
    fn test_shadow_large_blur() {
        let (mut target, stride) = make_target(40, 40);
        let glyph = test_glyph(15, 15);
        let config = DropShadowConfig {
            offset_x: 0.0,
            offset_y: 0.0,
            blur_radius: 10.0,
            color: [0, 0, 0, 255],
            opacity: 1.0,
        };
        render_shadow(&glyph, &config, &mut target, stride);
        assert!(has_nonzero_alpha(&target));
    }
}
