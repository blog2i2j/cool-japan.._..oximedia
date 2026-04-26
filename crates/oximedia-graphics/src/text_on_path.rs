//! Text-on-path renderer for broadcast graphics.
//!
//! Provides geometric placement of text characters along a [`BezierPath`],
//! computing per-glyph position and rotation from the path tangent so that
//! text flows naturally around curves, arcs, and complex spline shapes.
//!
//! # Example
//!
//! ```
//! use oximedia_graphics::bezier_path::BezierPath;
//! use oximedia_graphics::text_on_path::{TextOnPathConfig, PathSide};
//!
//! let mut path = BezierPath::new();
//! path.move_to(0.0, 50.0).line_to(200.0, 50.0);
//!
//! let config = TextOnPathConfig::new(path);
//! let glyphs = config.layout_glyphs("Hello");
//! assert!(!glyphs.is_empty());
//! ```

#![allow(dead_code)]

use crate::bezier_path::BezierPath;

// ── Epsilon for numerical tangent estimation ──────────────────────────────────

const TANGENT_EPS: f64 = 0.001;

// Number of samples used for arc-length approximation.
const ARC_LENGTH_SAMPLES: usize = 128;

// ── PathSide ──────────────────────────────────────────────────────────────────

/// Which side of the path to place text.
///
/// When `Left` (the default) text sits above/to-the-left of the travel
/// direction.  When `Right` the rotation is flipped by π so glyphs appear
/// below/to-the-right.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PathSide {
    /// Text above / left of the path direction (default).
    #[default]
    Left,
    /// Text below / right of the path direction.
    Right,
}

// ── PathGlyph ─────────────────────────────────────────────────────────────────

/// A single character positioned and oriented along a path.
#[derive(Debug, Clone)]
pub struct PathGlyph {
    /// The character to render at this position.
    pub ch: char,
    /// Centre position on the path in pixel space `(x, y)`.
    pub position: (f32, f32),
    /// Rotation angle in radians derived from the path tangent.
    ///
    /// A consumer can apply `rotate(rotation)` around `position` before
    /// drawing the glyph.
    pub rotation: f32,
    /// Parametric position `t ∈ [0, 1]` along the source path.
    pub t: f32,
}

// ── TextOnPathConfig ──────────────────────────────────────────────────────────

/// Configuration for rendering text along a [`BezierPath`].
///
/// Build one of these, call [`layout_glyphs`](TextOnPathConfig::layout_glyphs)
/// to obtain per-glyph transforms, and then either call
/// [`render`](TextOnPathConfig::render) for the built-in block rasteriser or
/// use the returned [`PathGlyph`] list with your own glyph renderer (e.g.
/// *fontdue*).
#[derive(Debug, Clone)]
pub struct TextOnPathConfig {
    /// The path the text follows.
    pub path: BezierPath,
    /// Starting offset along the path expressed as a fraction `[0, 1]`.
    ///
    /// `0.0` places the first glyph at the very beginning of the path;
    /// `0.5` starts halfway along.
    pub start_offset: f32,
    /// Which side of the path the text is placed on.
    pub side: PathSide,
    /// Approximate character advance width in pixels.
    ///
    /// This is the horizontal extent of one glyph cell and drives how far
    /// along the path to step between characters.
    pub char_width: f32,
    /// Approximate character height (ascent + descent) in pixels.
    pub char_height: f32,
    /// Additional space between characters expressed as a fraction of
    /// `char_width`.  `0.0` = tight; `0.2` = 20 % extra gap.
    pub letter_spacing: f32,
    /// Text colour as `[R, G, B, A]` bytes.
    pub color: [u8; 4],
    /// Nominal font size in pixels (informational; used by the block
    /// rasteriser to scale the glyph cell).
    pub font_size: f32,
}

impl TextOnPathConfig {
    /// Create a new config following a `path` with sensible defaults.
    ///
    /// | Field | Default |
    /// |-------|---------|
    /// | `start_offset` | `0.0` |
    /// | `side` | `PathSide::Left` |
    /// | `char_width` | `12.0` |
    /// | `char_height` | `16.0` |
    /// | `letter_spacing` | `0.1` |
    /// | `color` | white `[255, 255, 255, 255]` |
    /// | `font_size` | `16.0` |
    pub fn new(path: BezierPath) -> Self {
        Self {
            path,
            start_offset: 0.0,
            side: PathSide::Left,
            char_width: 12.0,
            char_height: 16.0,
            letter_spacing: 0.1,
            color: [255, 255, 255, 255],
            font_size: 16.0,
        }
    }

    // ── Internal helpers ─────────────────────────────────────────────────────

    /// Approximate total arc length of the path.
    fn total_arc_length(&self) -> f64 {
        self.path.approximate_length(ARC_LENGTH_SAMPLES)
    }

    /// Convert a fractional `t` position to an arc-length position by sampling
    /// a dense polyline and walking it.
    ///
    /// Returns a point `(x, y)` on the path at the given `t`.
    fn point_at_t(&self, t: f64) -> (f64, f64) {
        // Use sample_uniform with a fine grid; pick the index nearest to `t`.
        // Since sample_uniform spaces samples uniformly in arc-length, we map
        // `t` through arc-length instead:  generate enough samples and
        // interpolate.
        //
        // For better accuracy we use a two-stage approach:
        //   1. Flatten to a dense polyline (via `sample_uniform`)
        //   2. Walk the polyline to find the position at arc fraction `t`
        let t_clamped = t.clamp(0.0, 1.0);
        let n = 512usize;
        let samples = self.path.sample_uniform(n);
        if samples.is_empty() {
            return (0.0, 0.0);
        }
        if samples.len() == 1 {
            return (samples[0].x, samples[0].y);
        }

        // The samples are equally spaced in arc-length.
        let idx_f = t_clamped * (samples.len() - 1) as f64;
        let idx_lo = idx_f.floor() as usize;
        let idx_hi = (idx_lo + 1).min(samples.len() - 1);
        let frac = idx_f - idx_lo as f64;

        let lo = &samples[idx_lo];
        let hi = &samples[idx_hi];
        (lo.x + (hi.x - lo.x) * frac, lo.y + (hi.y - lo.y) * frac)
    }

    /// Numerically estimate the unit tangent vector at `t`.
    ///
    /// Uses central differences clamped to `[0, 1]`.
    fn tangent_at_t(&self, t: f64) -> (f64, f64) {
        let t_lo = (t - TANGENT_EPS).max(0.0);
        let t_hi = (t + TANGENT_EPS).min(1.0);

        let (x0, y0) = self.point_at_t(t_lo);
        let (x1, y1) = self.point_at_t(t_hi);

        let dx = x1 - x0;
        let dy = y1 - y0;
        let len = (dx * dx + dy * dy).sqrt();
        if len < 1e-12 {
            // Degenerate: return a right-pointing unit vector.
            return (1.0, 0.0);
        }
        (dx / len, dy / len)
    }

    // ── Public API ───────────────────────────────────────────────────────────

    /// Compute per-glyph positions and orientations for `text`.
    ///
    /// Characters that would be placed beyond `t = 1.0` are silently dropped
    /// so the caller never receives glyphs outside the path bounds.
    pub fn layout_glyphs(&self, text: &str) -> Vec<PathGlyph> {
        if text.is_empty() {
            return Vec::new();
        }

        let total_arc = self.total_arc_length();
        if total_arc < 1e-12 {
            // Degenerate path — no layout possible.
            return Vec::new();
        }

        // How far along the path (as a fraction of total arc) to advance per
        // character.
        let advance_fraction =
            (self.char_width as f64 * (1.0 + self.letter_spacing as f64)) / total_arc;

        let mut t = self.start_offset as f64;
        let mut glyphs = Vec::with_capacity(text.chars().count());

        for ch in text.chars() {
            if t > 1.0 {
                break;
            }

            let (px, py) = self.point_at_t(t);
            let (dx, dy) = self.tangent_at_t(t);
            let mut rotation = dy.atan2(dx) as f32;

            if self.side == PathSide::Right {
                rotation += std::f32::consts::PI;
            }

            glyphs.push(PathGlyph {
                ch,
                position: (px as f32, py as f32),
                rotation,
                t: t as f32,
            });

            t += advance_fraction;
        }

        glyphs
    }

    /// Rasterise `text` along the path into an RGBA8 pixel buffer.
    ///
    /// `pixels` must be exactly `width * height * 4` bytes.  Each glyph is
    /// drawn as a filled rectangle (a "block glyph") centred at its path
    /// position and rotated to match the path tangent.
    ///
    /// For real outline glyph rendering pass a `fontdue::Font` to
    /// [`render_with_font`](TextOnPathConfig::render_with_font) instead.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `pixels.len() != width as usize * height as usize * 4`.
    pub fn render(&self, text: &str, pixels: &mut [u8], width: u32, height: u32) {
        debug_assert_eq!(
            pixels.len(),
            width as usize * height as usize * 4,
            "pixels buffer length mismatch"
        );

        let glyphs = self.layout_glyphs(text);

        for glyph in &glyphs {
            self.render_glyph(glyph, pixels, width, height);
        }
    }

    /// Rasterise `text` along the path using real fontdue glyph outlines.
    ///
    /// This is the high-quality counterpart of [`render`](TextOnPathConfig::render).
    /// Instead of drawing block rectangles it uses *fontdue* to rasterise each
    /// character from the supplied `font`, then alpha-composites the greyscale
    /// coverage bitmap into `pixels` at the correct path position and rotation.
    ///
    /// `pixels` must be exactly `width * height * 4` RGBA bytes.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use oximedia_graphics::bezier_path::BezierPath;
    /// use oximedia_graphics::text_on_path::TextOnPathConfig;
    ///
    /// let font_bytes = std::fs::read("/path/to/font.ttf").unwrap();
    /// let font = fontdue::Font::from_bytes(font_bytes.as_slice(),
    ///     fontdue::FontSettings::default()).unwrap();
    ///
    /// let mut path = BezierPath::new();
    /// path.move_to(0.0, 50.0).line_to(200.0, 50.0);
    ///
    /// let config = TextOnPathConfig::new(path);
    /// let mut pixels = vec![0u8; 200 * 200 * 4];
    /// config.render_with_font("Hello", &font, &mut pixels, 200, 200);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `pixels.len() != width as usize * height as usize * 4`.
    pub fn render_with_font(
        &self,
        text: &str,
        font: &fontdue::Font,
        pixels: &mut [u8],
        width: u32,
        height: u32,
    ) {
        debug_assert_eq!(
            pixels.len(),
            width as usize * height as usize * 4,
            "pixels buffer length mismatch"
        );

        let glyphs = self.layout_glyphs(text);

        for glyph in &glyphs {
            self.render_glyph_fontdue(glyph, font, pixels, width, height);
        }
    }

    // ── Private rendering helpers ─────────────────────────────────────────────

    /// Draw a single glyph using fontdue outline rasterization.
    ///
    /// Rasterises `glyph.ch` at `self.font_size` pixels, then for every
    /// bitmap sample projects the bitmap coordinate through the glyph's
    /// rotation matrix and alpha-composites it into `pixels`.
    fn render_glyph_fontdue(
        &self,
        glyph: &PathGlyph,
        font: &fontdue::Font,
        pixels: &mut [u8],
        width: u32,
        height: u32,
    ) {
        let (metrics, bitmap) = font.rasterize(glyph.ch, self.font_size);

        // Space and other zero-area glyphs produce an empty bitmap — skip.
        if bitmap.is_empty() || metrics.width == 0 || metrics.height == 0 {
            return;
        }

        let cx = glyph.position.0;
        let cy = glyph.position.1;
        let rot = glyph.rotation;
        let cos_r = rot.cos();
        let sin_r = rot.sin();

        let half_bw = metrics.width as f32 * 0.5;
        let half_bh = metrics.height as f32 * 0.5;

        let src_r = self.color[0];
        let src_g = self.color[1];
        let src_b = self.color[2];
        let src_a = self.color[3];

        let img_w = width as i32;
        let img_h = height as i32;

        for by in 0..metrics.height {
            for bx in 0..metrics.width {
                let coverage_raw = bitmap[by * metrics.width + bx];
                if coverage_raw == 0 {
                    continue;
                }

                // Pixel offset from the bitmap centre.
                let local_x = bx as f32 - half_bw;
                let local_y = by as f32 - half_bh;

                // Rotate the local offset by the glyph's path-tangent angle.
                let world_x = cx + cos_r * local_x - sin_r * local_y;
                let world_y = cy + sin_r * local_x + cos_r * local_y;

                let px = world_x.round() as i32;
                let py = world_y.round() as i32;

                if px < 0 || px >= img_w || py < 0 || py >= img_h {
                    continue;
                }

                let offset = (py as usize * width as usize + px as usize) * 4;
                if offset + 3 >= pixels.len() {
                    continue;
                }

                // Scale source alpha by fontdue coverage value.
                let effective_alpha = (coverage_raw as u32 * src_a as u32 / 255).min(255) as u8;

                blend_pixel(
                    &mut pixels[offset..offset + 4],
                    src_r,
                    src_g,
                    src_b,
                    effective_alpha,
                );
            }
        }
    }

    /// Draw a single block glyph into `pixels`.
    fn render_glyph(&self, glyph: &PathGlyph, pixels: &mut [u8], width: u32, height: u32) {
        let cx = glyph.position.0;
        let cy = glyph.position.1;
        let half_w = self.char_width * 0.5;
        let half_h = self.char_height * 0.5;

        let rot = glyph.rotation;
        let cos_r = rot.cos();
        let sin_r = rot.sin();

        // Compute a conservative axis-aligned bounding box for the rotated
        // glyph cell so we only iterate over a small pixel neighbourhood.
        let diag = (half_w * half_w + half_h * half_h).sqrt();
        let bbox_min_x = (cx - diag).floor() as i32;
        let bbox_max_x = (cx + diag).ceil() as i32;
        let bbox_min_y = (cy - diag).floor() as i32;
        let bbox_max_y = (cy + diag).ceil() as i32;

        let img_w = width as i32;
        let img_h = height as i32;

        let src_r = self.color[0];
        let src_g = self.color[1];
        let src_b = self.color[2];
        let src_a = self.color[3];

        for py in bbox_min_y..=bbox_max_y {
            if py < 0 || py >= img_h {
                continue;
            }
            for px in bbox_min_x..=bbox_max_x {
                if px < 0 || px >= img_w {
                    continue;
                }

                // Translate pixel to glyph-local space.
                let dx = px as f32 - cx;
                let dy = py as f32 - cy;

                // Inverse-rotate: rotate by -rotation (transpose of rotation matrix).
                let lx = dx * cos_r + dy * sin_r;
                let ly = -dx * sin_r + dy * cos_r;

                // Check if (lx, ly) falls within the glyph cell rectangle.
                if lx < -half_w || lx > half_w || ly < -half_h || ly > half_h {
                    continue;
                }

                // Alpha-composite `color` over the existing pixel.
                let offset = (py as usize * width as usize + px as usize) * 4;

                // Guard: offset + 3 must be within slice.
                if offset + 3 >= pixels.len() {
                    continue;
                }

                blend_pixel(&mut pixels[offset..offset + 4], src_r, src_g, src_b, src_a);
            }
        }
    }
}

// ── Alpha compositing helper ──────────────────────────────────────────────────

/// Standard "source over" alpha composite.
///
/// `dst` is a 4-byte RGBA slice that is updated in-place.
#[inline]
fn blend_pixel(dst: &mut [u8], src_r: u8, src_g: u8, src_b: u8, src_a: u8) {
    let sa = src_a as f32 / 255.0;
    let da = dst[3] as f32 / 255.0;

    // Porter–Duff source-over composite.
    let out_a = sa + da * (1.0 - sa);

    if out_a < 1e-6 {
        dst[0] = 0;
        dst[1] = 0;
        dst[2] = 0;
        dst[3] = 0;
        return;
    }

    let inv_out_a = 1.0 / out_a;

    dst[0] = ((src_r as f32 * sa + dst[0] as f32 * da * (1.0 - sa)) * inv_out_a) as u8;
    dst[1] = ((src_g as f32 * sa + dst[1] as f32 * da * (1.0 - sa)) * inv_out_a) as u8;
    dst[2] = ((src_b as f32 * sa + dst[2] as f32 * da * (1.0 - sa)) * inv_out_a) as u8;
    dst[3] = (out_a * 255.0) as u8;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helper to build a simple horizontal path ──────────────────────────────

    fn horizontal_path(length: f64) -> BezierPath {
        let mut p = BezierPath::new();
        p.move_to(0.0, 50.0).line_to(length, 50.0);
        p
    }

    fn vertical_path(length: f64) -> BezierPath {
        let mut p = BezierPath::new();
        p.move_to(50.0, 0.0).line_to(50.0, length);
        p
    }

    // ── Test 1: glyph count ───────────────────────────────────────────────────

    /// A short text on a long path should produce one glyph per character.
    #[test]
    fn test_layout_glyphs_count() {
        let path = horizontal_path(1000.0);
        let config = TextOnPathConfig::new(path);
        let glyphs = config.layout_glyphs("Hello");
        assert_eq!(glyphs.len(), 5, "expected one glyph per character");
    }

    // ── Test 2: monotone x positions ─────────────────────────────────────────

    /// On a horizontal left-to-right path each successive glyph should have a
    /// greater x-coordinate than the previous.
    #[test]
    fn test_layout_glyphs_positions_monotone() {
        let path = horizontal_path(500.0);
        let mut config = TextOnPathConfig::new(path);
        config.char_width = 14.0;
        config.letter_spacing = 0.1;

        let glyphs = config.layout_glyphs("abcde");
        assert!(
            glyphs.len() >= 2,
            "need at least 2 glyphs to check monotonicity"
        );

        for window in glyphs.windows(2) {
            assert!(
                window[1].position.0 > window[0].position.0,
                "x positions should be monotonically increasing: {} <= {}",
                window[1].position.0,
                window[0].position.0
            );
        }
    }

    // ── Test 3: empty string ──────────────────────────────────────────────────

    #[test]
    fn test_layout_glyphs_empty_string() {
        let path = horizontal_path(500.0);
        let config = TextOnPathConfig::new(path);
        let glyphs = config.layout_glyphs("");
        assert!(glyphs.is_empty());
    }

    // ── Test 4: overflow truncation ───────────────────────────────────────────

    /// A very long string on a very short path should produce fewer glyphs
    /// than there are characters.
    #[test]
    fn test_layout_glyphs_overflow_truncates() {
        let path = horizontal_path(30.0); // only ~2 chars wide
        let mut config = TextOnPathConfig::new(path);
        config.char_width = 12.0;
        config.letter_spacing = 0.0;

        let text = "ABCDEFGHIJ"; // 10 characters
        let glyphs = config.layout_glyphs(text);
        assert!(
            glyphs.len() < text.len(),
            "expected truncation but got {} glyphs for {} chars",
            glyphs.len(),
            text.len()
        );
    }

    // ── Test 5: horizontal path rotation ─────────────────────────────────────

    /// On a purely horizontal path the tangent points right, so rotation
    /// should be close to 0.0 (or 2π) for PathSide::Left.
    #[test]
    fn test_layout_glyphs_rotation_horizontal() {
        let path = horizontal_path(500.0);
        let config = TextOnPathConfig::new(path);
        let glyphs = config.layout_glyphs("A");
        assert_eq!(glyphs.len(), 1);

        // atan2(0, positive) == 0
        let rot = glyphs[0].rotation;
        let normalised = rot.rem_euclid(2.0 * std::f32::consts::PI);
        let dist_from_zero = normalised.min(2.0 * std::f32::consts::PI - normalised);
        assert!(
            dist_from_zero < 0.1,
            "expected rotation ≈ 0 on horizontal path, got {rot}"
        );
    }

    // ── Test 6: vertical path rotation ───────────────────────────────────────

    /// On a vertical top-to-bottom path the tangent points down (+y), so
    /// rotation should be close to π/2.
    #[test]
    fn test_layout_glyphs_rotation_vertical() {
        let path = vertical_path(500.0);
        let config = TextOnPathConfig::new(path);
        let glyphs = config.layout_glyphs("A");
        assert_eq!(glyphs.len(), 1);

        let rot = glyphs[0].rotation;
        let expected = std::f32::consts::FRAC_PI_2;
        assert!(
            (rot - expected).abs() < 0.1,
            "expected rotation ≈ π/2 on vertical path, got {rot}"
        );
    }

    // ── Test 7: start offset shifts starting position ─────────────────────────

    /// With start_offset = 0.5 the first glyph should appear roughly halfway
    /// along the path.
    #[test]
    fn test_layout_glyphs_start_offset() {
        let path = horizontal_path(200.0);

        let mut config_start = TextOnPathConfig::new(path.clone());
        config_start.start_offset = 0.0;
        let glyphs_start = config_start.layout_glyphs("X");

        let mut config_mid = TextOnPathConfig::new(path);
        config_mid.start_offset = 0.5;
        let glyphs_mid = config_mid.layout_glyphs("X");

        assert_eq!(glyphs_start.len(), 1);
        assert_eq!(glyphs_mid.len(), 1);

        // Mid should have a larger x than start (path goes left-to-right).
        assert!(
            glyphs_mid[0].position.0 > glyphs_start[0].position.0,
            "start_offset=0.5 should place glyph further along the path"
        );
    }

    // ── Test 8: render does not panic ─────────────────────────────────────────

    #[test]
    fn test_render_no_panic() {
        let path = horizontal_path(180.0);
        let config = TextOnPathConfig::new(path);
        let mut pixels = vec![0u8; 200 * 200 * 4];
        config.render("Hi!", &mut pixels, 200, 200);
        // Reaching here without panic is the pass criterion.
    }

    // ── Test 9: render writes non-zero pixels ─────────────────────────────────

    /// Rendering a single opaque white character on a black buffer should
    /// produce at least one non-black pixel.
    #[test]
    fn test_render_colors_pixels() {
        let mut path = BezierPath::new();
        // Place the path squarely in the middle of the buffer.
        path.move_to(10.0, 100.0).line_to(190.0, 100.0);

        let mut config = TextOnPathConfig::new(path);
        config.color = [255, 0, 0, 255]; // fully opaque red
        config.char_width = 20.0;
        config.char_height = 20.0;

        let mut pixels = vec![0u8; 200 * 200 * 4];
        config.render("A", &mut pixels, 200, 200);

        let any_non_zero = pixels.chunks_exact(4).any(|px| px[3] > 0);
        assert!(
            any_non_zero,
            "expected at least one coloured pixel after rendering"
        );
    }

    // ── Test 10: render stays within buffer bounds ────────────────────────────

    /// Render to a tiny buffer and verify no out-of-bounds write occurred.
    /// (A write beyond the buffer would panic in debug mode; reaching the end
    /// of this test proves bounds checking worked.)
    #[test]
    fn test_render_bounds_respected() {
        let mut path = BezierPath::new();
        // Path deliberately goes outside a 20x20 buffer.
        path.move_to(-50.0, -50.0).line_to(200.0, 200.0);

        let mut config = TextOnPathConfig::new(path);
        config.char_width = 30.0;
        config.char_height = 30.0;
        config.color = [200, 200, 200, 255];

        let mut pixels = vec![0u8; 20 * 20 * 4];
        config.render("BOUNDS", &mut pixels, 20, 20);
        // If we reach here the bounds-guard worked.
    }

    // ── Test 11: PathSide::Right flips rotation ───────────────────────────────

    /// A glyph with PathSide::Right should have a rotation approximately π
    /// radians different from the same glyph with PathSide::Left.
    #[test]
    fn test_path_side_right_flips_rotation() {
        let path = horizontal_path(300.0);

        let mut config_left = TextOnPathConfig::new(path.clone());
        config_left.side = PathSide::Left;
        let left_glyphs = config_left.layout_glyphs("A");

        let mut config_right = TextOnPathConfig::new(path);
        config_right.side = PathSide::Right;
        let right_glyphs = config_right.layout_glyphs("A");

        assert_eq!(left_glyphs.len(), 1);
        assert_eq!(right_glyphs.len(), 1);

        let diff = (right_glyphs[0].rotation - left_glyphs[0].rotation).abs();
        // diff should be close to π (or close to π within floating-point rounding).
        let pi = std::f32::consts::PI;
        let normalised_diff = (diff - pi).abs();
        assert!(
            normalised_diff < 0.01,
            "expected rotation difference of π between Left and Right sides, got {diff}"
        );
    }

    // ── Test 12: TextOnPathConfig::new defaults ───────────────────────────────

    #[test]
    fn test_new_defaults() {
        let path = horizontal_path(100.0);
        let config = TextOnPathConfig::new(path);

        assert!(config.font_size > 0.0, "font_size must be positive");
        assert!(config.char_width > 0.0, "char_width must be positive");
        assert!(config.char_height > 0.0, "char_height must be positive");
        assert!(
            config.letter_spacing >= 0.0,
            "letter_spacing must be non-negative"
        );
        assert_eq!(config.start_offset, 0.0);
        assert_eq!(config.side, PathSide::Left);
        assert_eq!(config.color, [255, 255, 255, 255]);
    }

    // ── Fontdue candidate font paths ──────────────────────────────────────────

    /// Returns bytes for a candidate system TTF, or `None` if no font is found.
    ///
    /// The function tries several well-known macOS / Linux paths in order.
    /// Tests that need a real font early-return when this returns `None`, so
    /// CI on systems without any of these fonts is not broken.
    fn try_load_system_font() -> Option<Vec<u8>> {
        const CANDIDATES: &[&str] = &[
            // macOS Supplemental fonts
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Courier New.ttf",
            // Linux common paths
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
        ];

        for path in CANDIDATES {
            if let Ok(bytes) = std::fs::read(path) {
                return Some(bytes);
            }
        }
        None
    }

    // ── Test 13: fontdue rasterises 'A' to a non-empty bitmap ────────────────

    /// Load a system font with fontdue and verify that rasterizing 'A' at 16 px
    /// produces a non-empty coverage bitmap.
    #[test]
    fn test_fontdue_glyph_render_returns_pixels() {
        let bytes = match try_load_system_font() {
            Some(b) => b,
            None => {
                // No suitable font found on this system — skip gracefully.
                eprintln!(
                    "test_fontdue_glyph_render_returns_pixels: no system font found, skipping"
                );
                return;
            }
        };

        let font = fontdue::Font::from_bytes(bytes.as_slice(), fontdue::FontSettings::default())
            .expect("valid font bytes");

        let (metrics, bitmap) = font.rasterize('A', 16.0);

        assert!(
            !bitmap.is_empty(),
            "expected non-empty coverage bitmap for 'A'"
        );
        assert!(metrics.width > 0, "expected non-zero glyph width");
        assert!(metrics.height > 0, "expected non-zero glyph height");
        assert_eq!(
            bitmap.len(),
            metrics.width * metrics.height,
            "bitmap length must equal width * height"
        );

        // At least some pixels should have non-zero coverage.
        let any_lit = bitmap.iter().any(|&c| c > 0);
        assert!(
            any_lit,
            "expected at least one non-zero coverage pixel for 'A'"
        );
    }

    // ── Test 14: render_with_font produces visible pixels ────────────────────

    /// Render a short text string with a real fontdue font along a horizontal
    /// path and verify that at least one pixel is non-zero in the output buffer.
    #[test]
    fn test_text_on_path_with_fontdue() {
        let bytes = match try_load_system_font() {
            Some(b) => b,
            None => {
                eprintln!("test_text_on_path_with_fontdue: no system font found, skipping");
                return;
            }
        };

        let font = fontdue::Font::from_bytes(bytes.as_slice(), fontdue::FontSettings::default())
            .expect("valid font bytes");

        let mut path = BezierPath::new();
        path.move_to(10.0, 100.0).line_to(390.0, 100.0);

        let mut config = TextOnPathConfig::new(path);
        config.font_size = 24.0;
        config.char_width = 20.0;
        config.char_height = 24.0;
        config.color = [255, 255, 255, 255];

        let mut pixels = vec![0u8; 400 * 200 * 4];
        config.render_with_font("Hi!", &font, &mut pixels, 400, 200);

        let any_non_zero = pixels.chunks_exact(4).any(|px| px[3] > 0);
        assert!(
            any_non_zero,
            "expected at least one coloured pixel after fontdue render"
        );
    }
}
