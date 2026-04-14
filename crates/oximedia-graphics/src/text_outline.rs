//! Text outline and drop shadow rendering — composited directly into RGBA buffers.
//!
//! This module provides pixel-level text outline (stroke) and drop shadow effects
//! without relying on any post-processing pass. Both effects are rendered in a
//! single forward pass over a pre-rasterized glyph mask, making them suitable
//! for real-time broadcast graphics pipelines.
//!
//! # How it works
//!
//! 1. The caller supplies a **glyph mask** — a single-channel (alpha) image where
//!    `255` = solid glyph pixel and `0` = transparent background.
//! 2. [`render_outline`] erodes/dilates the mask with a square structuring element
//!    and composites the outline color beneath the glyph fill.
//! 3. [`render_drop_shadow`] blurs a copy of the mask with a box blur, offsets it
//!    by `(offset_x, offset_y)`, and composites it beneath the fill layer.
//! 4. [`render_text_with_effects`] combines both in the correct paint order:
//!    shadow → outline → glyph fill → compose onto the destination RGBA buffer.
//!
//! All operations are performed in straight (non-premultiplied) alpha so that the
//! resulting RGBA is ready for standard alpha compositing.

/// A single-channel (alpha-only) glyph mask.
///
/// Pixel order: row-major, top-left origin. Values range `0..=255`.
#[derive(Debug, Clone)]
pub struct GlyphMask {
    /// Raw alpha bytes.  Length must equal `width × height`.
    pub data: Vec<u8>,
    /// Mask width in pixels.
    pub width: u32,
    /// Mask height in pixels.
    pub height: u32,
}

impl GlyphMask {
    /// Create a new mask filled with zeros.
    pub fn blank(width: u32, height: u32) -> Self {
        Self {
            data: vec![0u8; (width * height) as usize],
            width,
            height,
        }
    }

    /// Create a mask from raw bytes.
    ///
    /// Returns `None` if `data.len() != width * height`.
    pub fn from_raw(data: Vec<u8>, width: u32, height: u32) -> Option<Self> {
        if data.len() == (width * height) as usize {
            Some(Self { data, width, height })
        } else {
            None
        }
    }

    /// Read alpha at `(x, y)`.  Returns `0` for out-of-bounds coordinates.
    #[inline]
    pub fn get(&self, x: i32, y: i32) -> u8 {
        if x < 0 || y < 0 || x >= self.width as i32 || y >= self.height as i32 {
            return 0;
        }
        self.data[(y as u32 * self.width + x as u32) as usize]
    }

    /// Write alpha at `(x, y)`.  Does nothing for out-of-bounds coordinates.
    #[inline]
    pub fn set(&mut self, x: i32, y: i32, value: u8) {
        if x < 0 || y < 0 || x >= self.width as i32 || y >= self.height as i32 {
            return;
        }
        self.data[(y as u32 * self.width + x as u32) as usize] = value;
    }
}

// ---------------------------------------------------------------------------
// Outline generation
// ---------------------------------------------------------------------------

/// Configuration for text outline rendering.
#[derive(Debug, Clone)]
pub struct OutlineConfig {
    /// Stroke half-width in pixels.  A value of `1` produces a 1-pixel border.
    pub radius: u32,
    /// Outline color as `[R, G, B, A]`.
    pub color: [u8; 4],
}

impl Default for OutlineConfig {
    fn default() -> Self {
        Self {
            radius: 2,
            color: [0, 0, 0, 255],
        }
    }
}

/// Generate an outline mask for `glyph` by dilating it with a square
/// structuring element of half-size `radius`.
///
/// The returned mask has the same dimensions as `glyph` and contains `255`
/// where the outline pixel is "on" (i.e. within `radius` of the original
/// glyph boundary) and `0` elsewhere.
///
/// **Interior pixels** (covered by the original glyph) are set to `0` in the
/// outline mask so that compositing the outline beneath the fill is trivial.
pub fn generate_outline_mask(glyph: &GlyphMask, config: &OutlineConfig) -> GlyphMask {
    let w = glyph.width as i32;
    let h = glyph.height as i32;
    let r = config.radius as i32;
    let mut outline = GlyphMask::blank(glyph.width, glyph.height);

    for y in 0..h {
        for x in 0..w {
            // Pixels that belong to the glyph itself are not part of the outline.
            if glyph.get(x, y) > 0 {
                continue;
            }
            // Check neighbourhood: any glyph pixel within radius?
            let mut hits = false;
            'outer: for dy in -r..=r {
                for dx in -r..=r {
                    if glyph.get(x + dx, y + dy) > 0 {
                        hits = true;
                        break 'outer;
                    }
                }
            }
            if hits {
                outline.set(x, y, 255);
            }
        }
    }
    outline
}

// ---------------------------------------------------------------------------
// Drop shadow generation
// ---------------------------------------------------------------------------

/// Configuration for drop shadow rendering.
#[derive(Debug, Clone)]
pub struct DropShadowConfig {
    /// Horizontal shadow offset in pixels (positive = right).
    pub offset_x: i32,
    /// Vertical shadow offset in pixels (positive = down).
    pub offset_y: i32,
    /// Box-blur radius in pixels (0 = no blur, sharp shadow).
    pub blur_radius: u32,
    /// Shadow color as `[R, G, B, A]`.  The alpha is additionally modulated
    /// by the blurred mask.
    pub color: [u8; 4],
}

impl Default for DropShadowConfig {
    fn default() -> Self {
        Self {
            offset_x: 3,
            offset_y: 3,
            blur_radius: 2,
            color: [0, 0, 0, 180],
        }
    }
}

/// Apply a simple box blur to a single-channel mask.
///
/// The blur kernel has size `(2*radius+1) × (2*radius+1)`.
/// Returns a new `GlyphMask` of the same dimensions.
pub fn box_blur_mask(mask: &GlyphMask, radius: u32) -> GlyphMask {
    if radius == 0 {
        return mask.clone();
    }

    let w = mask.width as i32;
    let h = mask.height as i32;
    let r = radius as i32;
    let kernel_size = (2 * r + 1) * (2 * r + 1);
    let mut blurred = GlyphMask::blank(mask.width, mask.height);

    for y in 0..h {
        for x in 0..w {
            let mut sum: u32 = 0;
            for ky in -r..=r {
                for kx in -r..=r {
                    sum += u32::from(mask.get(x + kx, y + ky));
                }
            }
            let avg = sum / kernel_size as u32;
            blurred.set(x, y, avg.min(255) as u8);
        }
    }
    blurred
}

/// Generate a shadow mask by blurring and offsetting the `glyph` mask.
///
/// The returned mask is the same size as `glyph`. Pixels that would fall
/// outside the canvas bounds after the offset are simply clipped.
pub fn generate_shadow_mask(glyph: &GlyphMask, config: &DropShadowConfig) -> GlyphMask {
    let blurred = box_blur_mask(glyph, config.blur_radius);
    let w = glyph.width as i32;
    let h = glyph.height as i32;
    let dx = config.offset_x;
    let dy = config.offset_y;
    let mut shadow = GlyphMask::blank(glyph.width, glyph.height);

    for y in 0..h {
        for x in 0..w {
            // The shadow at (x, y) comes from the blurred glyph at (x - dx, y - dy).
            let src_x = x - dx;
            let src_y = y - dy;
            shadow.set(x, y, blurred.get(src_x, src_y));
        }
    }
    shadow
}

// ---------------------------------------------------------------------------
// Compositing helpers
// ---------------------------------------------------------------------------

/// Alpha-composite `src` over `dst` using straight-alpha Porter-Duff "over".
///
/// Both `src` and `dst` are `[R, G, B, A]` tuples.  The result is written
/// into `dst` in-place.
#[inline]
fn blend_over(dst: &mut [u8; 4], src: [u8; 4]) {
    let sa = src[3] as u32;
    let da = dst[3] as u32;
    if sa == 0 {
        return;
    }
    if sa == 255 {
        *dst = src;
        return;
    }
    let out_a = sa + da * (255 - sa) / 255;
    if out_a == 0 {
        *dst = [0, 0, 0, 0];
        return;
    }
    for i in 0..3 {
        let sc = src[i] as u32;
        let dc = dst[i] as u32;
        dst[i] = ((sc * sa + dc * da * (255 - sa) / 255) / out_a) as u8;
    }
    dst[3] = out_a as u8;
}

// ---------------------------------------------------------------------------
// Combined render
// ---------------------------------------------------------------------------

/// Render options bundling outline and shadow configuration.
#[derive(Debug, Clone)]
pub struct TextEffectConfig {
    /// Optional outline.
    pub outline: Option<OutlineConfig>,
    /// Optional drop shadow.
    pub shadow: Option<DropShadowConfig>,
    /// Glyph fill color `[R, G, B, A]`.
    pub fill_color: [u8; 4],
}

impl Default for TextEffectConfig {
    fn default() -> Self {
        Self {
            outline: Some(OutlineConfig::default()),
            shadow: Some(DropShadowConfig::default()),
            fill_color: [255, 255, 255, 255],
        }
    }
}

/// Render a glyph with optional outline and drop shadow into an RGBA buffer.
///
/// `canvas` must be `width × height × 4` bytes (RGBA, straight alpha).
///
/// **Compositing order** (back to front):
/// 1. Drop shadow (if enabled)
/// 2. Outline stroke (if enabled)
/// 3. Glyph fill
pub fn render_text_with_effects(
    canvas: &mut [u8],
    canvas_width: u32,
    canvas_height: u32,
    glyph: &GlyphMask,
    config: &TextEffectConfig,
) {
    assert_eq!(
        canvas.len(),
        (canvas_width * canvas_height * 4) as usize,
        "canvas dimensions do not match buffer length"
    );

    // Build optional effect layers.
    let shadow_mask = config.shadow.as_ref().map(|sc| {
        let m = generate_shadow_mask(glyph, sc);
        (m, sc.color)
    });
    let outline_mask = config.outline.as_ref().map(|oc| {
        let m = generate_outline_mask(glyph, oc);
        (m, oc.color)
    });

    let cw = canvas_width as i32;
    let ch = canvas_height as i32;
    let gw = glyph.width as i32;
    let gh = glyph.height as i32;

    for gy in 0..gh {
        for gx in 0..gw {
            let cx = gx;
            let cy = gy;
            if cx < 0 || cy < 0 || cx >= cw || cy >= ch {
                continue;
            }
            let idx = (cy as u32 * canvas_width + cx as u32) as usize * 4;
            let mut pixel: [u8; 4] = [
                canvas[idx],
                canvas[idx + 1],
                canvas[idx + 2],
                canvas[idx + 3],
            ];

            // Layer 1: shadow.
            if let Some((ref sm, sc)) = shadow_mask {
                let alpha = sm.get(gx, gy);
                if alpha > 0 {
                    let mut shadow_px = sc;
                    shadow_px[3] = ((shadow_px[3] as u32 * alpha as u32) / 255) as u8;
                    blend_over(&mut pixel, shadow_px);
                }
            }

            // Layer 2: outline.
            if let Some((ref om, oc)) = outline_mask {
                let alpha = om.get(gx, gy);
                if alpha > 0 {
                    let mut out_px = oc;
                    out_px[3] = ((out_px[3] as u32 * alpha as u32) / 255) as u8;
                    blend_over(&mut pixel, out_px);
                }
            }

            // Layer 3: glyph fill.
            let fill_alpha = glyph.get(gx, gy);
            if fill_alpha > 0 {
                let mut fill_px = config.fill_color;
                fill_px[3] = ((fill_px[3] as u32 * fill_alpha as u32) / 255) as u8;
                blend_over(&mut pixel, fill_px);
            }

            canvas[idx] = pixel[0];
            canvas[idx + 1] = pixel[1];
            canvas[idx + 2] = pixel[2];
            canvas[idx + 3] = pixel[3];
        }
    }
}

/// Render only the outline of a glyph into an RGBA buffer (no fill, no shadow).
pub fn render_outline(
    canvas: &mut [u8],
    canvas_width: u32,
    canvas_height: u32,
    glyph: &GlyphMask,
    config: &OutlineConfig,
) {
    let mask = generate_outline_mask(glyph, config);
    let cw = canvas_width as i32;
    let ch = canvas_height as i32;
    let gw = glyph.width as i32;
    let gh = glyph.height as i32;

    for gy in 0..gh {
        for gx in 0..gw {
            let cx = gx;
            let cy = gy;
            if cx < 0 || cy < 0 || cx >= cw || cy >= ch {
                continue;
            }
            let alpha = mask.get(gx, gy);
            if alpha == 0 {
                continue;
            }
            let idx = (cy as u32 * canvas_width + cx as u32) as usize * 4;
            let mut pixel: [u8; 4] = [
                canvas[idx],
                canvas[idx + 1],
                canvas[idx + 2],
                canvas[idx + 3],
            ];
            let mut src = config.color;
            src[3] = ((src[3] as u32 * alpha as u32) / 255) as u8;
            blend_over(&mut pixel, src);
            canvas[idx] = pixel[0];
            canvas[idx + 1] = pixel[1];
            canvas[idx + 2] = pixel[2];
            canvas[idx + 3] = pixel[3];
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_mask(w: u32, h: u32) -> GlyphMask {
        GlyphMask {
            data: vec![255u8; (w * h) as usize],
            width: w,
            height: h,
        }
    }

    fn cross_mask() -> GlyphMask {
        // 5×5 mask: center cross is solid, corners are empty.
        let mut m = GlyphMask::blank(5, 5);
        for i in 0..5_i32 {
            m.set(2, i, 255); // vertical bar
            m.set(i, 2, 255); // horizontal bar
        }
        m
    }

    // 1. GlyphMask::blank creates zero-filled mask of correct size.
    #[test]
    fn test_glyph_mask_blank() {
        let m = GlyphMask::blank(4, 3);
        assert_eq!(m.data.len(), 12);
        assert!(m.data.iter().all(|&v| v == 0));
    }

    // 2. GlyphMask::from_raw rejects wrong data length.
    #[test]
    fn test_glyph_mask_from_raw_invalid() {
        let result = GlyphMask::from_raw(vec![0u8; 5], 4, 3);
        assert!(result.is_none());
    }

    // 3. GlyphMask::get returns 0 for out-of-bounds.
    #[test]
    fn test_glyph_mask_oob_get() {
        let m = solid_mask(3, 3);
        assert_eq!(m.get(-1, 0), 0);
        assert_eq!(m.get(3, 0), 0);
        assert_eq!(m.get(0, 3), 0);
    }

    // 4. generate_outline_mask leaves solid interior at 0 (outline is around glyph).
    #[test]
    fn test_outline_mask_interior_zero() {
        let glyph = solid_mask(5, 5);
        let cfg = OutlineConfig { radius: 1, color: [0, 0, 0, 255] };
        let outline = generate_outline_mask(&glyph, &cfg);
        // Interior pixels (inside glyph) must be 0.
        assert_eq!(outline.get(2, 2), 0, "interior must not be in outline");
    }

    // 5. Outline mask surrounds the glyph — a pixel just outside the 5×5 square
    //    (in a 9×9 canvas) is set by the outline.
    #[test]
    fn test_outline_mask_border_pixels_set() {
        // 9×9 glyph mask with a 5×5 solid block in the center.
        let mut glyph = GlyphMask::blank(9, 9);
        for y in 2..7_i32 {
            for x in 2..7_i32 {
                glyph.set(x, y, 255);
            }
        }
        let cfg = OutlineConfig { radius: 1, color: [0, 0, 0, 255] };
        let outline = generate_outline_mask(&glyph, &cfg);
        // Pixel at (1, 4) is adjacent to the block → should be in outline.
        assert_eq!(outline.get(1, 4), 255, "adjacent pixel should be in outline");
        // Pixel at (0, 0) is far from block → should not be in outline (radius=1).
        assert_eq!(outline.get(0, 0), 0, "far corner must not be in outline");
    }

    // 6. box_blur_mask with radius 0 returns an identical mask.
    #[test]
    fn test_box_blur_radius_zero_identity() {
        let glyph = cross_mask();
        let blurred = box_blur_mask(&glyph, 0);
        assert_eq!(blurred.data, glyph.data);
    }

    // 7. box_blur_mask with radius > 0 diffuses the mask.
    #[test]
    fn test_box_blur_diffuses() {
        let mut glyph = GlyphMask::blank(7, 7);
        glyph.set(3, 3, 255); // single solid center pixel
        let blurred = box_blur_mask(&glyph, 1);
        // Neighbouring pixels must have non-zero values after blurring.
        assert!(blurred.get(2, 3) > 0, "blur should spread to neighbors");
        assert!(blurred.get(3, 2) > 0);
    }

    // 8. generate_shadow_mask shifts the glyph by the offset.
    #[test]
    fn test_shadow_mask_offset() {
        let mut glyph = GlyphMask::blank(10, 10);
        glyph.set(3, 3, 255);
        let sc = DropShadowConfig {
            offset_x: 2,
            offset_y: 2,
            blur_radius: 0, // no blur so we can pinpoint exactly
            color: [0, 0, 0, 180],
        };
        let shadow = generate_shadow_mask(&glyph, &sc);
        // The shadow of the pixel at (3,3) should appear at (3+2, 3+2) = (5,5).
        assert_eq!(shadow.get(5, 5), 255, "shadow must appear at offset position");
        // Original position should be 0.
        assert_eq!(shadow.get(3, 3), 0, "original position must not have shadow");
    }

    // 9. render_text_with_effects writes to canvas alpha channel for solid glyph.
    #[test]
    fn test_render_effects_writes_alpha() {
        let mut canvas = vec![0u8; 10 * 10 * 4];
        let glyph = solid_mask(5, 5);
        let config = TextEffectConfig {
            outline: None,
            shadow: None,
            fill_color: [200, 100, 50, 255],
        };
        render_text_with_effects(&mut canvas, 10, 10, &glyph, &config);
        // Pixel (0,0) inside the glyph should have non-zero alpha.
        assert!(canvas[3] > 0, "alpha must be written for solid glyph");
    }

    // 10. render_outline writes outline pixels.
    #[test]
    fn test_render_outline_writes_outline() {
        let glyph = solid_mask(3, 3);
        // Canvas is 7×7; center the glyph at (2,2) by using a padded glyph.
        let mut big_glyph = GlyphMask::blank(7, 7);
        for y in 2..5_i32 {
            for x in 2..5_i32 {
                big_glyph.set(x, y, 255);
            }
        }
        let mut canvas = vec![0u8; 7 * 7 * 4];
        let cfg = OutlineConfig { radius: 1, color: [255, 0, 0, 255] };
        render_outline(&mut canvas, 7, 7, &big_glyph, &cfg);
        // Pixel at (1,3) should be red (outline of the 3×3 block).
        let idx = (3 * 7 + 1) * 4;
        assert_eq!(canvas[idx], 255, "red channel must be set for outline pixel");
        assert!(canvas[idx + 3] > 0, "outline pixel must have non-zero alpha");
    }

    // 11. blend_over opaque src replaces dst.
    #[test]
    fn test_blend_over_opaque() {
        let mut dst = [100u8, 100, 100, 200];
        let src = [255u8, 0, 0, 255];
        blend_over(&mut dst, src);
        assert_eq!(dst, [255, 0, 0, 255]);
    }

    // 12. blend_over transparent src leaves dst unchanged.
    #[test]
    fn test_blend_over_transparent() {
        let mut dst = [50u8, 60, 70, 255];
        let original = dst;
        let src = [255u8, 0, 0, 0];
        blend_over(&mut dst, src);
        assert_eq!(dst, original);
    }

    // 13. Shadow and outline rendered together do not panic.
    #[test]
    fn test_render_combined_no_panic() {
        let mut canvas = vec![0u8; 20 * 20 * 4];
        let mut glyph = GlyphMask::blank(10, 10);
        for y in 2..8_i32 {
            for x in 2..8_i32 {
                glyph.set(x, y, 255);
            }
        }
        let config = TextEffectConfig::default();
        render_text_with_effects(&mut canvas, 20, 20, &glyph, &config);
        // If we reach here without panic, the test passes.
    }
}
