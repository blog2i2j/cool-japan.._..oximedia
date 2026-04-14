#![allow(dead_code)]
//! Pixel-level inline text rendering effects (outline, shadow, fill compositing).
//!
//! Provides [`OutlineConfig`], [`InlineShadowConfig`], and [`TextRenderConfig`]
//! types plus the [`render_text_effects`] function that composites shadow, outline,
//! and fill layers onto an RGBA canvas in painter's order.

// ============================================================================
// OutlineConfig / InlineShadowConfig — pixel-level inline rendering
// ============================================================================

/// Configuration for inline text outline rendering (dilation-based stroke).
#[derive(Debug, Clone)]
pub struct OutlineConfig {
    /// Outline colour as `[R, G, B, A]`.
    pub color: [u8; 4],
    /// Outline width in pixels (clamped to 1–8).
    pub width: u32,
    /// When `true`, the outline is drawn behind the fill (stroke effect).
    pub behind_fill: bool,
}

impl OutlineConfig {
    /// Create a new `OutlineConfig`.
    pub fn new(color: [u8; 4], width: u32, behind_fill: bool) -> Self {
        Self { color, width: width.clamp(0, 8), behind_fill }
    }
}

/// Configuration for inline drop shadow rendering.
#[derive(Debug, Clone)]
pub struct InlineShadowConfig {
    /// Shadow colour as `[R, G, B, A]`.
    pub color: [u8; 4],
    /// Horizontal shadow offset in pixels (positive = right).
    pub offset_x: i32,
    /// Vertical shadow offset in pixels (positive = down).
    pub offset_y: i32,
    /// Gaussian blur sigma in pixels (`0.0` = hard shadow).
    pub blur_sigma: f32,
    /// Shadow opacity multiplier in `[0.0, 1.0]`.
    pub opacity: f32,
}

impl InlineShadowConfig {
    /// Create a new `InlineShadowConfig`.
    pub fn new(color: [u8; 4], offset_x: i32, offset_y: i32, blur_sigma: f32, opacity: f32) -> Self {
        Self {
            color, offset_x, offset_y,
            blur_sigma: blur_sigma.max(0.0),
            opacity: opacity.clamp(0.0, 1.0),
        }
    }
}

/// Text render configuration that carries optional inline outline and shadow.
#[derive(Debug, Clone, Default)]
pub struct TextRenderConfig {
    /// Optional inline outline.
    pub outline: Option<OutlineConfig>,
    /// Optional inline drop shadow.
    pub shadow: Option<InlineShadowConfig>,
}

impl TextRenderConfig {
    /// Create a default config with no effects.
    pub fn new() -> Self { Self::default() }

    /// Attach an outline.
    pub fn with_outline(mut self, outline: OutlineConfig) -> Self {
        self.outline = Some(outline); self
    }

    /// Attach a shadow.
    pub fn with_shadow(mut self, shadow: InlineShadowConfig) -> Self {
        self.shadow = Some(shadow); self
    }
}

// ============================================================================
// Pixel-level render helper
// ============================================================================

/// Alpha-composite `src` (RGBA) over `dst` (RGBA) using `src_alpha * opacity`.
#[inline]
fn composite_over(dst: &mut [u8; 4], src: [u8; 4], opacity: f32) {
    let sa = src[3] as f32 / 255.0 * opacity;
    let ia = 1.0 - sa;
    dst[0] = (src[0] as f32 * sa + dst[0] as f32 * ia) as u8;
    dst[1] = (src[1] as f32 * sa + dst[1] as f32 * ia) as u8;
    dst[2] = (src[2] as f32 * sa + dst[2] as f32 * ia) as u8;
    dst[3] = ((sa + dst[3] as f32 / 255.0 * ia) * 255.0).min(255.0) as u8;
}

/// Render text inline effects — shadow, outline, and fill — into an RGBA8 pixel buffer.
///
/// `text_alpha` is a single-channel (alpha-only) mask of the rendered text glyphs,
/// stored row-major with `width * height` bytes.  The function composites effects
/// in painter's order (shadow → outline → fill) onto `canvas` (RGBA8, `width * height * 4` bytes).
///
/// `fill_color` is the solid fill colour `[R, G, B, A]` used when no gradient is wanted.
pub fn render_text_effects(
    canvas: &mut [u8],
    text_alpha: &[u8],
    width: u32,
    height: u32,
    fill_color: [u8; 4],
    config: &TextRenderConfig,
) {
    use crate::image_filter::{dilate_alpha, gaussian_blur};

    let w = width as usize;
    let h = height as usize;
    let n = w * h;
    if n == 0 || canvas.len() < n * 4 || text_alpha.len() < n {
        return;
    }

    // --- 1. Shadow (behind everything) ---
    if let Some(ref s) = config.shadow {
        if s.opacity > 0.0 {
            // Build a single-channel alpha buffer, then blur it
            let mut shadow_rgba = vec![0u8; n * 4];
            for i in 0..n {
                let a = text_alpha[i];
                shadow_rgba[i * 4]     = s.color[0];
                shadow_rgba[i * 4 + 1] = s.color[1];
                shadow_rgba[i * 4 + 2] = s.color[2];
                shadow_rgba[i * 4 + 3] = a;
            }
            if s.blur_sigma > 0.0 {
                gaussian_blur(&mut shadow_rgba, width, height, s.blur_sigma);
            }
            // Shift and composite
            let ox = s.offset_x;
            let oy = s.offset_y;
            for y in 0..h {
                for x in 0..w {
                    let src_x = x as i32 - ox;
                    let src_y = y as i32 - oy;
                    if src_x < 0 || src_x >= w as i32 || src_y < 0 || src_y >= h as i32 {
                        continue;
                    }
                    let src_idx = src_y as usize * w + src_x as usize;
                    let sa = shadow_rgba[src_idx * 4 + 3];
                    if sa == 0 { continue; }
                    let dst_idx = y * w + x;
                    let mut dst = [
                        canvas[dst_idx * 4],
                        canvas[dst_idx * 4 + 1],
                        canvas[dst_idx * 4 + 2],
                        canvas[dst_idx * 4 + 3],
                    ];
                    let src_px = [s.color[0], s.color[1], s.color[2], sa];
                    composite_over(&mut dst, src_px, s.opacity);
                    canvas[dst_idx * 4]     = dst[0];
                    canvas[dst_idx * 4 + 1] = dst[1];
                    canvas[dst_idx * 4 + 2] = dst[2];
                    canvas[dst_idx * 4 + 3] = dst[3];
                }
            }
        }
    }

    // --- 2. Outline (behind fill) ---
    if let Some(ref o) = config.outline {
        if o.width > 0 {
            // Build dilated RGBA buffer: set colour+alpha from text_alpha, then dilate
            let mut outline_rgba = vec![0u8; n * 4];
            for i in 0..n {
                outline_rgba[i * 4]     = o.color[0];
                outline_rgba[i * 4 + 1] = o.color[1];
                outline_rgba[i * 4 + 2] = o.color[2];
                outline_rgba[i * 4 + 3] = text_alpha[i];
            }
            dilate_alpha(&mut outline_rgba, width, height, o.width);
            for i in 0..n {
                let oa = outline_rgba[i * 4 + 3];
                if oa == 0 { continue; }
                let mut dst = [
                    canvas[i * 4],
                    canvas[i * 4 + 1],
                    canvas[i * 4 + 2],
                    canvas[i * 4 + 3],
                ];
                let src_px = [o.color[0], o.color[1], o.color[2], oa];
                composite_over(&mut dst, src_px, 1.0);
                canvas[i * 4]     = dst[0];
                canvas[i * 4 + 1] = dst[1];
                canvas[i * 4 + 2] = dst[2];
                canvas[i * 4 + 3] = dst[3];
            }
        }
    }

    // --- 3. Text fill (on top) ---
    for i in 0..n {
        let fa = text_alpha[i];
        if fa == 0 { continue; }
        let mut dst = [
            canvas[i * 4],
            canvas[i * 4 + 1],
            canvas[i * 4 + 2],
            canvas[i * 4 + 3],
        ];
        let src_px = [fill_color[0], fill_color[1], fill_color[2], fa];
        composite_over(&mut dst, src_px, fill_color[3] as f32 / 255.0);
        canvas[i * 4]     = dst[0];
        canvas[i * 4 + 1] = dst[1];
        canvas[i * 4 + 2] = dst[2];
        canvas[i * 4 + 3] = dst[3];
    }
}

// ============================================================================
// Tests for inline outline and shadow rendering
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_alpha_rect(w: usize, h: usize, x0: usize, y0: usize, x1: usize, y1: usize) -> Vec<u8> {
        let mut buf = vec![0u8; w * h];
        for y in y0..y1.min(h) {
            for x in x0..x1.min(w) {
                buf[y * w + x] = 255;
            }
        }
        buf
    }

    fn count_nonzero_alpha(canvas: &[u8]) -> usize {
        canvas.chunks_exact(4).filter(|p| p[3] > 0).count()
    }

    #[test]
    fn test_outline_renders_larger_than_text() {
        let w = 30usize; let h = 30usize;
        let alpha = make_alpha_rect(w, h, 10, 10, 20, 20);
        // Baseline: fill only
        let mut canvas_base = vec![0u8; w * h * 4];
        let cfg_none = TextRenderConfig::new();
        render_text_effects(&mut canvas_base, &alpha, w as u32, h as u32, [255,255,255,255], &cfg_none);
        let base_count = count_nonzero_alpha(&canvas_base);
        // With outline width=3
        let mut canvas_out = vec![0u8; w * h * 4];
        let cfg_out = TextRenderConfig::new().with_outline(OutlineConfig::new([255,0,0,255], 3, true));
        render_text_effects(&mut canvas_out, &alpha, w as u32, h as u32, [255,255,255,255], &cfg_out);
        let out_count = count_nonzero_alpha(&canvas_out);
        assert!(out_count > base_count, "outline count={out_count} should exceed base={base_count}");
    }

    #[test]
    fn test_outline_color_present() {
        let w = 20usize; let h = 20usize;
        // Text occupies only the interior; outline should bleed into ring around it
        let alpha = make_alpha_rect(w, h, 5, 5, 15, 15);
        let mut canvas = vec![0u8; w * h * 4];
        let outline_color = [200u8, 50u8, 10u8, 255u8];
        let cfg = TextRenderConfig::new().with_outline(OutlineConfig::new(outline_color, 2, true));
        render_text_effects(&mut canvas, &alpha, w as u32, h as u32, [255,255,255,255], &cfg);
        // Pixel at (4,5) — just outside the text block — should be outline color
        let px_idx = (5 * w + 3) * 4;
        assert!(canvas[px_idx + 3] > 0, "outline pixel should have non-zero alpha");
    }

    #[test]
    fn test_shadow_renders_offset() {
        let w = 30usize; let h = 30usize;
        let alpha = make_alpha_rect(w, h, 5, 5, 10, 10);
        let mut canvas = vec![0u8; w * h * 4];
        // Shadow offset +8,+8, no blur
        let shadow = InlineShadowConfig::new([0,0,0,255], 8, 8, 0.0, 1.0);
        let cfg = TextRenderConfig::new().with_shadow(shadow);
        render_text_effects(&mut canvas, &alpha, w as u32, h as u32, [255,255,255,255], &cfg);
        // Shadow should appear at (5+8, 5+8) = (13,13)
        let shadow_px = (13 * w + 13) * 4;
        assert!(canvas[shadow_px + 3] > 0, "shadow should appear at offset position");
    }

    #[test]
    fn test_shadow_blur_sigma_zero_hard_edge() {
        let w = 20usize; let h = 20usize;
        let alpha = make_alpha_rect(w, h, 5, 5, 10, 10);
        let mut canvas = vec![0u8; w * h * 4];
        let shadow = InlineShadowConfig::new([0,0,0,255], 2, 2, 0.0, 1.0);
        let cfg = TextRenderConfig::new().with_shadow(shadow);
        render_text_effects(&mut canvas, &alpha, w as u32, h as u32, [255,255,255,255], &cfg);
        // With no blur, shadow pixel at offset position should be max alpha
        let px = ((5+2) * w + (5+2)) * 4;
        assert_eq!(canvas[px + 3], 255, "hard shadow should be full alpha");
        // Adjacent pixel just outside the shadow footprint should be zero
        let px2 = ((5+2) * w + (10+2)) * 4;
        assert_eq!(canvas[px2 + 3], 0, "pixel outside shadow range should be zero");
    }

    #[test]
    fn test_shadow_behind_text() {
        let w = 20usize; let h = 20usize;
        let alpha = make_alpha_rect(w, h, 5, 5, 10, 10);
        let mut canvas = vec![0u8; w * h * 4];
        let fill = [255u8, 255u8, 255u8, 255u8];
        let shadow = InlineShadowConfig::new([0,0,200,255], 0, 0, 0.0, 1.0); // same position, blue shadow
        let cfg = TextRenderConfig::new().with_shadow(shadow);
        render_text_effects(&mut canvas, &alpha, w as u32, h as u32, fill, &cfg);
        // Text fill (white) should dominate at a text pixel: red channel = 255
        let px = (7 * w + 7) * 4;
        assert_eq!(canvas[px], 255, "fill should dominate over shadow at text pixel");
    }

    #[test]
    fn test_outline_behind_fill() {
        let w = 20usize; let h = 20usize;
        let alpha = make_alpha_rect(w, h, 5, 5, 15, 15);
        let mut canvas = vec![0u8; w * h * 4];
        let fill = [255u8, 0u8, 0u8, 255u8]; // red fill
        let outline = OutlineConfig::new([0, 0, 255, 255], 2, true); // blue outline
        let cfg = TextRenderConfig::new().with_outline(outline);
        render_text_effects(&mut canvas, &alpha, w as u32, h as u32, fill, &cfg);
        // At the center of the text block, fill (red) should dominate
        let px = (10 * w + 10) * 4;
        assert!(canvas[px] > canvas[px + 2], "fill red should be greater than outline blue at text center");
    }

    #[test]
    fn test_no_outline_unchanged() {
        let w = 20usize; let h = 20usize;
        let alpha = make_alpha_rect(w, h, 5, 5, 15, 15);
        let mut canvas1 = vec![0u8; w * h * 4];
        let mut canvas2 = vec![0u8; w * h * 4];
        let cfg = TextRenderConfig::new(); // no effects
        render_text_effects(&mut canvas1, &alpha, w as u32, h as u32, [200,200,200,255], &cfg);
        render_text_effects(&mut canvas2, &alpha, w as u32, h as u32, [200,200,200,255], &cfg);
        assert_eq!(canvas1, canvas2, "two identical no-effect renders should match");
    }

    #[test]
    fn test_no_shadow_unchanged() {
        let w = 20usize; let h = 20usize;
        let alpha = make_alpha_rect(w, h, 5, 5, 15, 15);
        let mut canvas1 = vec![0u8; w * h * 4];
        let mut canvas2 = vec![0u8; w * h * 4];
        let cfg = TextRenderConfig::new(); // shadow: None
        render_text_effects(&mut canvas1, &alpha, w as u32, h as u32, [180,180,180,255], &cfg);
        render_text_effects(&mut canvas2, &alpha, w as u32, h as u32, [180,180,180,255], &cfg);
        assert_eq!(canvas1, canvas2, "two identical no-shadow renders should match");
    }

    #[test]
    fn test_outline_width_zero_no_effect() {
        let w = 20usize; let h = 20usize;
        let alpha = make_alpha_rect(w, h, 5, 5, 15, 15);
        let mut canvas_base = vec![0u8; w * h * 4];
        let mut canvas_zero = vec![0u8; w * h * 4];
        let cfg_base = TextRenderConfig::new();
        let cfg_zero = TextRenderConfig::new().with_outline(OutlineConfig::new([255,0,0,255], 0, true));
        render_text_effects(&mut canvas_base, &alpha, w as u32, h as u32, [255,255,255,255], &cfg_base);
        render_text_effects(&mut canvas_zero, &alpha, w as u32, h as u32, [255,255,255,255], &cfg_zero);
        assert_eq!(canvas_base, canvas_zero, "width=0 outline should have no visible effect");
    }

    #[test]
    fn test_shadow_opacity_zero_invisible() {
        let w = 20usize; let h = 20usize;
        let alpha = make_alpha_rect(w, h, 5, 5, 10, 10);
        let mut canvas_base = vec![0u8; w * h * 4];
        let mut canvas_shadow = vec![0u8; w * h * 4];
        let cfg_base = TextRenderConfig::new();
        let shadow = InlineShadowConfig::new([0,0,0,255], 4, 4, 0.0, 0.0); // opacity=0
        let cfg_shadow = TextRenderConfig::new().with_shadow(shadow);
        render_text_effects(&mut canvas_base, &alpha, w as u32, h as u32, [255,255,255,255], &cfg_base);
        render_text_effects(&mut canvas_shadow, &alpha, w as u32, h as u32, [255,255,255,255], &cfg_shadow);
        assert_eq!(canvas_base, canvas_shadow, "opacity=0 shadow should be invisible");
    }
}
