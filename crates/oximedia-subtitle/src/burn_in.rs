//! Subtitle burn-in rendering configuration and job management.
//!
//! Provides configuration presets, position computation with safe-area enforcement,
//! and job descriptions for subtitle burn-in workflows.

/// Configuration for subtitle burn-in rendering.
#[derive(Debug, Clone)]
pub struct BurnInConfig {
    /// Font size in pixels.
    pub font_size: u32,
    /// Margin from frame edge in pixels.
    pub margin_px: u32,
    /// Whether to render a semi-transparent background box behind the text.
    pub background_box: bool,
    /// Background box opacity (0 = transparent, 255 = opaque).
    pub background_opacity: u8,
    /// Safe area as a percentage of the frame dimension (e.g. 0.05 = 5%).
    pub safe_area_pct: f32,
}

impl BurnInConfig {
    /// Broadcast-safe configuration: larger text, 10% safe area, background box.
    #[must_use]
    pub fn broadcast() -> Self {
        Self {
            font_size: 72,
            margin_px: 30,
            background_box: true,
            background_opacity: 180,
            safe_area_pct: 0.10,
        }
    }

    /// Web streaming configuration: medium text, 5% safe area, no background box.
    #[must_use]
    pub fn web() -> Self {
        Self {
            font_size: 48,
            margin_px: 20,
            background_box: false,
            background_opacity: 0,
            safe_area_pct: 0.05,
        }
    }
}

/// Alignment options for burn-in positioning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BurnInAlignment {
    /// Top-left corner.
    TopLeft,
    /// Top center.
    TopCenter,
    /// Top-right corner.
    TopRight,
    /// Bottom-left corner.
    BottomLeft,
    /// Bottom center (most common for subtitles).
    BottomCenter,
    /// Bottom-right corner.
    BottomRight,
}

impl BurnInAlignment {
    /// Whether this alignment places content near the top of the frame.
    #[must_use]
    pub const fn is_top(&self) -> bool {
        matches!(self, Self::TopLeft | Self::TopCenter | Self::TopRight)
    }

    /// Whether this alignment places content near the left edge.
    #[must_use]
    pub const fn is_left(&self) -> bool {
        matches!(self, Self::TopLeft | Self::BottomLeft)
    }
}

/// Renderer that computes burn-in positions and validates safe areas.
#[derive(Debug, Clone)]
pub struct BurnInRenderer {
    /// The configuration used for this renderer.
    pub config: BurnInConfig,
}

impl BurnInRenderer {
    /// Create a new renderer with the given configuration.
    #[must_use]
    pub fn new(config: BurnInConfig) -> Self {
        Self { config }
    }

    /// Compute the pixel position (x, y) for a text block within a frame.
    ///
    /// - `text_w`, `text_h`: width and height of the rendered text block in pixels.
    /// - `frame_w`, `frame_h`: width and height of the video frame in pixels.
    /// - `align`: desired alignment.
    ///
    /// The position respects `margin_px` and `safe_area_pct`.
    #[must_use]
    pub fn compute_position(
        &self,
        text_w: u32,
        text_h: u32,
        frame_w: u32,
        frame_h: u32,
        align: &BurnInAlignment,
    ) -> (u32, u32) {
        #[allow(clippy::cast_possible_truncation)]
        #[allow(clippy::cast_sign_loss)]
        let safe_x = (frame_w as f32 * self.config.safe_area_pct) as u32;
        #[allow(clippy::cast_possible_truncation)]
        #[allow(clippy::cast_sign_loss)]
        let safe_y = (frame_h as f32 * self.config.safe_area_pct) as u32;

        let margin = self.config.margin_px;

        let x = match align {
            BurnInAlignment::TopLeft | BurnInAlignment::BottomLeft => safe_x + margin,
            BurnInAlignment::TopCenter | BurnInAlignment::BottomCenter => {
                let center = frame_w / 2;
                center.saturating_sub(text_w / 2)
            }
            BurnInAlignment::TopRight | BurnInAlignment::BottomRight => {
                frame_w.saturating_sub(text_w + safe_x + margin)
            }
        };

        let y = if align.is_top() {
            safe_y + margin
        } else {
            frame_h.saturating_sub(text_h + safe_y + margin)
        };

        (x, y)
    }

    /// Check whether a text block at (x, y) with size (w, h) lies within the
    /// safe area of the frame.
    ///
    /// Returns `true` if the block is fully within the safe area.
    #[must_use]
    pub fn validate_safe_area(
        &self,
        x: u32,
        y: u32,
        w: u32,
        h: u32,
        frame_w: u32,
        frame_h: u32,
    ) -> bool {
        #[allow(clippy::cast_possible_truncation)]
        #[allow(clippy::cast_sign_loss)]
        let safe_x = (frame_w as f32 * self.config.safe_area_pct) as u32;
        #[allow(clippy::cast_possible_truncation)]
        #[allow(clippy::cast_sign_loss)]
        let safe_y = (frame_h as f32 * self.config.safe_area_pct) as u32;

        x >= safe_x
            && y >= safe_y
            && (x + w) <= frame_w.saturating_sub(safe_x)
            && (y + h) <= frame_h.saturating_sub(safe_y)
    }
}

/// Color for burn-in rendering (RGBA).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BurnInColor {
    /// Red channel (0-255).
    pub r: u8,
    /// Green channel (0-255).
    pub g: u8,
    /// Blue channel (0-255).
    pub b: u8,
    /// Alpha channel (0 = transparent, 255 = opaque).
    pub a: u8,
}

impl BurnInColor {
    /// Create a new color.
    #[must_use]
    pub const fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    /// White with full opacity.
    #[must_use]
    pub const fn white() -> Self {
        Self::new(255, 255, 255, 255)
    }

    /// Black with specified opacity.
    #[must_use]
    pub const fn black_with_alpha(a: u8) -> Self {
        Self::new(0, 0, 0, a)
    }

    /// Convert RGB to BT.709 YUV.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    pub fn to_yuv(&self) -> (u8, u8, u8) {
        let r = self.r as f32;
        let g = self.g as f32;
        let b = self.b as f32;
        let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        let u = (b - y) / 1.8556 + 128.0;
        let v = (r - y) / 1.5748 + 128.0;
        (
            y.clamp(0.0, 255.0) as u8,
            u.clamp(0.0, 255.0) as u8,
            v.clamp(0.0, 255.0) as u8,
        )
    }
}

/// A simple monochrome glyph bitmap for burn-in text rendering.
///
/// Each pixel is an alpha coverage value (0 = transparent, 255 = fully covered).
/// Glyphs are rendered using a built-in bitmap font approximation.
#[derive(Debug, Clone)]
pub struct BurnInGlyph {
    /// Glyph bitmap (alpha coverage values).
    pub bitmap: Vec<u8>,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

impl BurnInGlyph {
    /// Render a character using a simple built-in bitmap representation.
    ///
    /// This generates a basic rectangular glyph scaled to `font_size` pixels high.
    /// For production use, a full font rasterizer should be substituted.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    #[allow(clippy::cast_precision_loss)]
    pub fn render_char(ch: char, font_size: u32) -> Self {
        // Each character cell is font_size tall, approx 0.6 * font_size wide
        let cell_w = ((font_size as f32) * 0.6).ceil() as u32;
        let cell_h = font_size;
        let mut bitmap = vec![0u8; (cell_w * cell_h) as usize];

        // For whitespace, return empty glyph
        if ch == ' ' {
            return Self {
                bitmap,
                width: cell_w,
                height: cell_h,
            };
        }

        // Generate a simple stroke-based glyph
        // Use a 5x7 template scaled up to cell_w x cell_h
        let template = char_to_5x7_template(ch);

        let scale_x = cell_w as f32 / 5.0;
        let scale_y = cell_h as f32 / 7.0;

        for py in 0..cell_h {
            for px in 0..cell_w {
                // Map pixel back to template coordinates
                let tx = (px as f32 / scale_x).min(4.0) as usize;
                let ty = (py as f32 / scale_y).min(6.0) as usize;

                if tx < 5 && ty < 7 {
                    let template_idx = ty * 5 + tx;
                    if template_idx < template.len() && template[template_idx] > 0 {
                        // Apply anti-aliasing at edges using bilinear sampling
                        let alpha = template[template_idx];
                        bitmap[(py * cell_w + px) as usize] = alpha;
                    }
                }
            }
        }

        Self {
            bitmap,
            width: cell_w,
            height: cell_h,
        }
    }
}

/// Get a 5x7 pixel template for an ASCII character.
///
/// Returns a 35-element array of alpha values (0 or 255).
fn char_to_5x7_template(ch: char) -> Vec<u8> {
    // Standard 5x7 bitmap font patterns for common characters
    let pattern: [u8; 35] = match ch {
        'A' | 'a' => [
            0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0,
            1, 1, 0, 0, 0, 1,
        ],
        'B' | 'b' => [
            1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0,
            1, 1, 1, 1, 1, 0,
        ],
        'C' | 'c' => [
            0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0,
            1, 0, 1, 1, 1, 0,
        ],
        'D' | 'd' => [
            1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0,
            1, 1, 1, 1, 1, 0,
        ],
        'E' | 'e' => [
            1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0,
            0, 1, 1, 1, 1, 1,
        ],
        'H' | 'h' => [
            1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0,
            1, 1, 0, 0, 0, 1,
        ],
        'I' | 'i' => [
            0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
            0, 0, 1, 1, 1, 0,
        ],
        'L' | 'l' => [
            1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0,
            0, 1, 1, 1, 1, 1,
        ],
        'O' | 'o' | '0' => [
            0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0,
            1, 0, 1, 1, 1, 0,
        ],
        'T' | 't' => [
            1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
            0, 0, 0, 1, 0, 0,
        ],
        '1' => [
            0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
            0, 0, 1, 1, 1, 0,
        ],
        ':' => [
            0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 0,
        ],
        // Default: filled rectangle for unknown characters
        _ => [
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1,
        ],
    };

    pattern
        .iter()
        .map(|&p| if p > 0 { 255u8 } else { 0u8 })
        .collect()
}

/// Render a text string into a sequence of positioned glyphs for burn-in.
///
/// Returns a composite bitmap containing all characters laid out horizontally,
/// along with its dimensions.
#[must_use]
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
pub fn render_text_bitmap(text: &str, font_size: u32) -> BurnInGlyph {
    let glyphs: Vec<BurnInGlyph> = text
        .chars()
        .map(|ch| BurnInGlyph::render_char(ch, font_size))
        .collect();

    if glyphs.is_empty() {
        return BurnInGlyph {
            bitmap: Vec::new(),
            width: 0,
            height: 0,
        };
    }

    // Calculate total width and max height
    let total_width: u32 = glyphs.iter().map(|g| g.width).sum();
    let max_height = glyphs.iter().map(|g| g.height).max().unwrap_or(0);

    let mut composite = vec![0u8; (total_width * max_height) as usize];
    let mut cursor_x: u32 = 0;

    for glyph in &glyphs {
        for gy in 0..glyph.height.min(max_height) {
            for gx in 0..glyph.width {
                let src_idx = (gy * glyph.width + gx) as usize;
                let dst_x = cursor_x + gx;
                let dst_idx = (gy * total_width + dst_x) as usize;
                if src_idx < glyph.bitmap.len() && dst_idx < composite.len() {
                    composite[dst_idx] = glyph.bitmap[src_idx];
                }
            }
        }
        cursor_x += glyph.width;
    }

    BurnInGlyph {
        bitmap: composite,
        width: total_width,
        height: max_height,
    }
}

impl BurnInRenderer {
    /// Render subtitle text directly onto an RGBA pixel buffer.
    ///
    /// The `buffer` must be `frame_w * frame_h * 4` bytes (RGBA8).
    /// The text is rendered at the computed position based on alignment.
    ///
    /// # Errors
    ///
    /// Returns an error description if the buffer is too small.
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    pub fn render_to_rgba(
        &self,
        buffer: &mut [u8],
        frame_w: u32,
        frame_h: u32,
        text: &str,
        color: BurnInColor,
        align: &BurnInAlignment,
    ) -> Result<(), String> {
        let expected_size = (frame_w as usize) * (frame_h as usize) * 4;
        if buffer.len() < expected_size {
            return Err(format!(
                "Buffer too small: {} < {expected_size}",
                buffer.len()
            ));
        }

        // Render text to bitmap
        let text_bitmap = render_text_bitmap(text, self.config.font_size);
        if text_bitmap.width == 0 || text_bitmap.height == 0 {
            return Ok(());
        }

        let (pos_x, pos_y) = self.compute_position(
            text_bitmap.width,
            text_bitmap.height,
            frame_w,
            frame_h,
            align,
        );

        // Render background box if configured
        if self.config.background_box {
            let bg_color = BurnInColor::black_with_alpha(self.config.background_opacity);
            let padding = self.config.margin_px / 2;
            let bg_x1 = pos_x.saturating_sub(padding);
            let bg_y1 = pos_y.saturating_sub(padding);
            let bg_x2 = (pos_x + text_bitmap.width + padding).min(frame_w);
            let bg_y2 = (pos_y + text_bitmap.height + padding).min(frame_h);

            for py in bg_y1..bg_y2 {
                for px in bg_x1..bg_x2 {
                    let idx = ((py * frame_w + px) * 4) as usize;
                    if idx + 3 < buffer.len() {
                        let alpha_f = bg_color.a as f32 / 255.0;
                        let inv_alpha = 1.0 - alpha_f;
                        buffer[idx] =
                            (bg_color.r as f32 * alpha_f + buffer[idx] as f32 * inv_alpha) as u8;
                        buffer[idx + 1] = (bg_color.g as f32 * alpha_f
                            + buffer[idx + 1] as f32 * inv_alpha)
                            as u8;
                        buffer[idx + 2] = (bg_color.b as f32 * alpha_f
                            + buffer[idx + 2] as f32 * inv_alpha)
                            as u8;
                        buffer[idx + 3] = buffer[idx + 3]
                            .saturating_add(((255.0 - buffer[idx + 3] as f32) * alpha_f) as u8);
                    }
                }
            }
        }

        // Alpha-blend the text bitmap onto the frame
        for gy in 0..text_bitmap.height {
            for gx in 0..text_bitmap.width {
                let px = pos_x + gx;
                let py = pos_y + gy;

                if px >= frame_w || py >= frame_h {
                    continue;
                }

                let glyph_idx = (gy * text_bitmap.width + gx) as usize;
                let glyph_alpha = if glyph_idx < text_bitmap.bitmap.len() {
                    text_bitmap.bitmap[glyph_idx]
                } else {
                    0
                };

                if glyph_alpha == 0 {
                    continue;
                }

                let idx = ((py * frame_w + px) * 4) as usize;
                if idx + 3 < buffer.len() {
                    let alpha_f = glyph_alpha as f32 / 255.0 * color.a as f32 / 255.0;
                    let inv_alpha = 1.0 - alpha_f;
                    buffer[idx] = (color.r as f32 * alpha_f + buffer[idx] as f32 * inv_alpha) as u8;
                    buffer[idx + 1] =
                        (color.g as f32 * alpha_f + buffer[idx + 1] as f32 * inv_alpha) as u8;
                    buffer[idx + 2] =
                        (color.b as f32 * alpha_f + buffer[idx + 2] as f32 * inv_alpha) as u8;
                    buffer[idx + 3] = buffer[idx + 3]
                        .saturating_add(((255.0 - buffer[idx + 3] as f32) * alpha_f) as u8);
                }
            }
        }

        Ok(())
    }

    /// Render subtitle text onto a YUV420p frame buffer.
    ///
    /// - `y_plane`: luma plane (frame_w * frame_h bytes)
    /// - `u_plane`: chroma-U plane ((frame_w/2) * (frame_h/2) bytes)
    /// - `v_plane`: chroma-V plane ((frame_w/2) * (frame_h/2) bytes)
    ///
    /// # Errors
    ///
    /// Returns an error description if any plane is too small.
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    pub fn render_to_yuv420p(
        &self,
        y_plane: &mut [u8],
        u_plane: &mut [u8],
        v_plane: &mut [u8],
        frame_w: u32,
        frame_h: u32,
        text: &str,
        color: BurnInColor,
        align: &BurnInAlignment,
    ) -> Result<(), String> {
        let y_size = (frame_w * frame_h) as usize;
        let uv_size = ((frame_w / 2) * (frame_h / 2)) as usize;

        if y_plane.len() < y_size {
            return Err("Y plane too small".to_string());
        }
        if u_plane.len() < uv_size || v_plane.len() < uv_size {
            return Err("UV planes too small".to_string());
        }

        let text_bitmap = render_text_bitmap(text, self.config.font_size);
        if text_bitmap.width == 0 || text_bitmap.height == 0 {
            return Ok(());
        }

        let (pos_x, pos_y) = self.compute_position(
            text_bitmap.width,
            text_bitmap.height,
            frame_w,
            frame_h,
            align,
        );

        let (y_val, u_val, v_val) = color.to_yuv();
        let uv_w = frame_w / 2;

        // Render background box in YUV
        if self.config.background_box {
            let bg_yuv = BurnInColor::black_with_alpha(self.config.background_opacity).to_yuv();
            let bg_alpha = self.config.background_opacity as f32 / 255.0;
            let padding = self.config.margin_px / 2;
            let bg_x1 = pos_x.saturating_sub(padding);
            let bg_y1 = pos_y.saturating_sub(padding);
            let bg_x2 = (pos_x + text_bitmap.width + padding).min(frame_w);
            let bg_y2 = (pos_y + text_bitmap.height + padding).min(frame_h);

            for py in bg_y1..bg_y2 {
                for px in bg_x1..bg_x2 {
                    let y_idx = (py * frame_w + px) as usize;
                    if y_idx < y_plane.len() {
                        let inv = 1.0 - bg_alpha;
                        y_plane[y_idx] =
                            (bg_yuv.0 as f32 * bg_alpha + y_plane[y_idx] as f32 * inv) as u8;
                    }
                    let uv_x = px / 2;
                    let uv_y = py / 2;
                    let uv_idx = (uv_y * uv_w + uv_x) as usize;
                    if uv_idx < u_plane.len() {
                        let inv = 1.0 - bg_alpha * 0.25;
                        u_plane[uv_idx] = (bg_yuv.1 as f32 * bg_alpha * 0.25
                            + u_plane[uv_idx] as f32 * inv)
                            as u8;
                        v_plane[uv_idx] = (bg_yuv.2 as f32 * bg_alpha * 0.25
                            + v_plane[uv_idx] as f32 * inv)
                            as u8;
                    }
                }
            }
        }

        // Blend text
        for gy in 0..text_bitmap.height {
            for gx in 0..text_bitmap.width {
                let px = pos_x + gx;
                let py = pos_y + gy;
                if px >= frame_w || py >= frame_h {
                    continue;
                }

                let glyph_idx = (gy * text_bitmap.width + gx) as usize;
                let glyph_alpha = if glyph_idx < text_bitmap.bitmap.len() {
                    text_bitmap.bitmap[glyph_idx]
                } else {
                    0
                };
                if glyph_alpha == 0 {
                    continue;
                }

                let alpha_f = glyph_alpha as f32 / 255.0 * color.a as f32 / 255.0;
                let inv = 1.0 - alpha_f;

                let y_idx = (py * frame_w + px) as usize;
                if y_idx < y_plane.len() {
                    y_plane[y_idx] = (y_val as f32 * alpha_f + y_plane[y_idx] as f32 * inv) as u8;
                }

                let uv_x = px / 2;
                let uv_y = py / 2;
                let uv_idx = (uv_y * uv_w + uv_x) as usize;
                if uv_idx < u_plane.len() {
                    let uv_alpha = alpha_f * 0.25;
                    let uv_inv = 1.0 - uv_alpha;
                    u_plane[uv_idx] =
                        (u_val as f32 * uv_alpha + u_plane[uv_idx] as f32 * uv_inv) as u8;
                    v_plane[uv_idx] =
                        (v_val as f32 * uv_alpha + v_plane[uv_idx] as f32 * uv_inv) as u8;
                }
            }
        }

        Ok(())
    }
}

/// A burn-in job describing input/output paths and configuration.
#[derive(Debug, Clone)]
pub struct BurnInJob {
    /// Path to the subtitle file (SRT, VTT, etc.).
    pub subtitle_path: String,
    /// Path to the source video file.
    pub video_path: String,
    /// Path for the output video file.
    pub output_path: String,
    /// Burn-in rendering configuration.
    pub config: BurnInConfig,
}

impl BurnInJob {
    /// Create a new burn-in job.
    #[must_use]
    pub fn new(
        subtitle_path: impl Into<String>,
        video_path: impl Into<String>,
        output_path: impl Into<String>,
        config: BurnInConfig,
    ) -> Self {
        Self {
            subtitle_path: subtitle_path.into(),
            video_path: video_path.into(),
            output_path: output_path.into(),
            config,
        }
    }

    /// Estimated processing time in milliseconds for a video of the given duration.
    ///
    /// Uses a simple heuristic: 1.5× real-time for broadcast config, 1.0× for web.
    #[must_use]
    pub fn estimated_processing_ms(&self, duration_ms: u64) -> u64 {
        if self.config.background_box {
            // Broadcast-style: slightly more expensive
            duration_ms + duration_ms / 2
        } else {
            // Web-style: roughly real-time
            duration_ms
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broadcast_config_font_size() {
        let cfg = BurnInConfig::broadcast();
        assert_eq!(cfg.font_size, 72);
        assert!(cfg.background_box);
    }

    #[test]
    fn test_web_config_no_background() {
        let cfg = BurnInConfig::web();
        assert!(!cfg.background_box);
        assert_eq!(cfg.font_size, 48);
    }

    #[test]
    fn test_alignment_is_top() {
        assert!(BurnInAlignment::TopLeft.is_top());
        assert!(BurnInAlignment::TopCenter.is_top());
        assert!(!BurnInAlignment::BottomRight.is_top());
    }

    #[test]
    fn test_alignment_is_left() {
        assert!(BurnInAlignment::TopLeft.is_left());
        assert!(BurnInAlignment::BottomLeft.is_left());
        assert!(!BurnInAlignment::TopCenter.is_left());
        assert!(!BurnInAlignment::TopRight.is_left());
    }

    #[test]
    fn test_compute_position_bottom_center() {
        let renderer = BurnInRenderer::new(BurnInConfig::web());
        let (x, y) = renderer.compute_position(200, 50, 1920, 1080, &BurnInAlignment::BottomCenter);
        // x should be near center
        let expected_x = 1920 / 2 - 200 / 2;
        assert_eq!(x, expected_x);
        // y should be near the bottom
        assert!(y > 1080 / 2, "y={y} should be in the lower half");
    }

    #[test]
    fn test_compute_position_top_left() {
        let renderer = BurnInRenderer::new(BurnInConfig::web());
        let (x, y) = renderer.compute_position(100, 50, 1920, 1080, &BurnInAlignment::TopLeft);
        // Should be a small positive value
        assert!(x < 200, "x={x}");
        assert!(y < 200, "y={y}");
    }

    #[test]
    fn test_compute_position_bottom_right() {
        let renderer = BurnInRenderer::new(BurnInConfig::web());
        let (x, _y) = renderer.compute_position(200, 50, 1920, 1080, &BurnInAlignment::BottomRight);
        // Should be near the right side
        assert!(x > 1920 / 2, "x={x}");
    }

    #[test]
    fn test_validate_safe_area_inside() {
        let renderer = BurnInRenderer::new(BurnInConfig::web());
        // safe_area_pct = 0.05 → safe_x = 96, safe_y = 54 for 1920x1080
        let ok = renderer.validate_safe_area(100, 60, 200, 50, 1920, 1080);
        assert!(ok, "Should be inside safe area");
    }

    #[test]
    fn test_validate_safe_area_outside_left() {
        let renderer = BurnInRenderer::new(BurnInConfig::web());
        // x=0 is outside the 5% safe area
        let ok = renderer.validate_safe_area(0, 60, 200, 50, 1920, 1080);
        assert!(!ok, "x=0 should be outside safe area");
    }

    #[test]
    fn test_validate_safe_area_outside_right() {
        let renderer = BurnInRenderer::new(BurnInConfig::web());
        // x + w exceeds frame_w - safe_x
        let ok = renderer.validate_safe_area(1800, 60, 200, 50, 1920, 1080);
        assert!(!ok, "Right edge outside safe area");
    }

    #[test]
    fn test_burn_in_job_estimated_broadcast() {
        let job = BurnInJob::new("a.srt", "v.mp4", "out.mp4", BurnInConfig::broadcast());
        assert_eq!(job.estimated_processing_ms(10_000), 15_000);
    }

    #[test]
    fn test_burn_in_job_estimated_web() {
        let job = BurnInJob::new("a.srt", "v.mp4", "out.mp4", BurnInConfig::web());
        assert_eq!(job.estimated_processing_ms(10_000), 10_000);
    }

    #[test]
    fn test_burn_in_job_fields() {
        let job = BurnInJob::new("sub.srt", "video.mp4", "output.mp4", BurnInConfig::web());
        assert_eq!(job.subtitle_path, "sub.srt");
        assert_eq!(job.video_path, "video.mp4");
        assert_eq!(job.output_path, "output.mp4");
    }

    #[test]
    fn test_burn_in_color_white() {
        let c = BurnInColor::white();
        assert_eq!(c.r, 255);
        assert_eq!(c.g, 255);
        assert_eq!(c.b, 255);
        assert_eq!(c.a, 255);
    }

    #[test]
    fn test_burn_in_color_to_yuv() {
        let white = BurnInColor::white();
        let (y, u, v) = white.to_yuv();
        assert!(y > 200, "Y should be bright: {y}");
        assert!(
            (u as i16 - 128).unsigned_abs() < 10,
            "U should be near 128: {u}"
        );
        assert!(
            (v as i16 - 128).unsigned_abs() < 10,
            "V should be near 128: {v}"
        );
    }

    #[test]
    fn test_render_text_bitmap_not_empty() {
        let bm = render_text_bitmap("Hello", 24);
        assert!(bm.width > 0);
        assert!(bm.height > 0);
        assert!(!bm.bitmap.is_empty());
    }

    #[test]
    fn test_render_text_bitmap_space() {
        let bm = render_text_bitmap(" ", 24);
        assert!(bm.width > 0);
        // Space should be mostly transparent
        assert!(bm.bitmap.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_render_to_rgba_basic() {
        let renderer = BurnInRenderer::new(BurnInConfig::web());
        let w = 320u32;
        let h = 240u32;
        let mut buffer = vec![0u8; (w * h * 4) as usize];

        let result = renderer.render_to_rgba(
            &mut buffer,
            w,
            h,
            "Hi",
            BurnInColor::white(),
            &BurnInAlignment::BottomCenter,
        );
        assert!(result.is_ok());

        // At least some pixels should have been modified
        let modified = buffer.iter().any(|&b| b > 0);
        assert!(modified, "Some pixels should be modified");
    }

    #[test]
    fn test_render_to_rgba_buffer_too_small() {
        let renderer = BurnInRenderer::new(BurnInConfig::web());
        let mut buffer = vec![0u8; 10]; // Way too small

        let result = renderer.render_to_rgba(
            &mut buffer,
            320,
            240,
            "Hi",
            BurnInColor::white(),
            &BurnInAlignment::BottomCenter,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_render_to_yuv420p_basic() {
        let renderer = BurnInRenderer::new(BurnInConfig::web());
        let w = 320u32;
        let h = 240u32;
        let mut y_plane = vec![16u8; (w * h) as usize];
        let mut u_plane = vec![128u8; ((w / 2) * (h / 2)) as usize];
        let mut v_plane = vec![128u8; ((w / 2) * (h / 2)) as usize];

        let result = renderer.render_to_yuv420p(
            &mut y_plane,
            &mut u_plane,
            &mut v_plane,
            w,
            h,
            "TC",
            BurnInColor::white(),
            &BurnInAlignment::TopLeft,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_render_to_rgba_with_background() {
        let renderer = BurnInRenderer::new(BurnInConfig::broadcast());
        let w = 640u32;
        let h = 480u32;
        let mut buffer = vec![0u8; (w * h * 4) as usize];

        let result = renderer.render_to_rgba(
            &mut buffer,
            w,
            h,
            "TEST",
            BurnInColor::white(),
            &BurnInAlignment::BottomCenter,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_glyph_render_char_dimensions() {
        let glyph = BurnInGlyph::render_char('A', 48);
        assert_eq!(glyph.height, 48);
        assert!(glyph.width > 0);
        assert_eq!(glyph.bitmap.len(), (glyph.width * glyph.height) as usize);
    }
}
