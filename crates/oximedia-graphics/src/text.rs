//! Text rendering engine with advanced typography

use crate::color::Color;
use crate::error::{GraphicsError, Result};
use crate::primitives::{Point, Rect};
use ab_glyph::{Font, FontRef, PxScale, ScaleFont};
use fontdue::FontSettings;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Well-known system font paths searched in order on macOS.
#[cfg(target_os = "macos")]
const SYSTEM_FONT_PATHS: &[&str] = &[
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/SFNS.ttf",
];

/// Well-known system font paths searched in order on Linux.
#[cfg(target_os = "linux")]
const SYSTEM_FONT_PATHS: &[&str] = &[
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
];

/// Well-known system font paths searched in order on Windows.
#[cfg(target_os = "windows")]
const SYSTEM_FONT_PATHS: &[&str] = &[
    "C:\\Windows\\Fonts\\arial.ttf",
    "C:\\Windows\\Fonts\\segoeui.ttf",
];

/// Fallback empty list for platforms that don't match the above.
#[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
const SYSTEM_FONT_PATHS: &[&str] = &[];

/// Try to parse font bytes with `fontdue` as a validity check.
///
/// Returns `Ok(bytes)` if the bytes constitute a parseable font, or an
/// error describing why the bytes were rejected.
pub(crate) fn validate_font_bytes(bytes: Vec<u8>) -> Result<Vec<u8>> {
    fontdue::Font::from_bytes(bytes.as_slice(), FontSettings::default())
        .map(|_| bytes)
        .map_err(|e| GraphicsError::FontError(format!("Font parse error: {e}")))
}

/// Search the platform's well-known font directories and the `OXIMEDIA_SYSTEM_FONT`
/// environment variable for a usable font file.
///
/// Returns `Ok(Vec<u8>)` containing the raw font bytes of the first valid font
/// found, or a [`GraphicsError::FontError`] if no font could be loaded.
fn find_system_font_bytes() -> Result<Vec<u8>> {
    // 1. Try each platform-specific path in declaration order.
    for path in SYSTEM_FONT_PATHS {
        match std::fs::read(path) {
            Ok(bytes) => match validate_font_bytes(bytes) {
                Ok(valid) => return Ok(valid),
                Err(_) => continue, // unreadable or invalid format — try next
            },
            Err(_) => continue,
        }
    }

    // 2. Honour the OXIMEDIA_SYSTEM_FONT environment variable as a final override.
    if let Ok(env_path) = std::env::var("OXIMEDIA_SYSTEM_FONT") {
        let bytes = std::fs::read(&env_path).map_err(|e| {
            GraphicsError::FontError(format!(
                "OXIMEDIA_SYSTEM_FONT path '{env_path}' is not readable: {e}",
            ))
        })?;
        return validate_font_bytes(bytes);
    }

    Err(GraphicsError::FontError(
        "No system font found: searched platform paths and OXIMEDIA_SYSTEM_FONT env var"
            .to_string(),
    ))
}

/// Font family
#[derive(Debug, Clone)]
pub struct FontFamily {
    /// Regular font
    pub regular: Arc<Vec<u8>>,
    /// Bold font
    pub bold: Option<Arc<Vec<u8>>>,
    /// Italic font
    pub italic: Option<Arc<Vec<u8>>>,
    /// Bold italic font
    pub bold_italic: Option<Arc<Vec<u8>>>,
}

impl FontFamily {
    /// Create from regular font only
    #[must_use]
    pub fn from_regular(data: Vec<u8>) -> Self {
        Self {
            regular: Arc::new(data),
            bold: None,
            italic: None,
            bold_italic: None,
        }
    }

    /// Get font for style
    #[must_use]
    pub fn get_font(&self, style: FontStyle, weight: FontWeight) -> &[u8] {
        match (style, weight) {
            (FontStyle::Italic, FontWeight::Bold) => {
                if let Some(ref font) = self.bold_italic {
                    font
                } else if let Some(ref font) = self.bold {
                    font
                } else if let Some(ref font) = self.italic {
                    font
                } else {
                    &self.regular
                }
            }
            (FontStyle::Italic, _) => {
                if let Some(ref font) = self.italic {
                    font
                } else {
                    &self.regular
                }
            }
            (_, FontWeight::Bold) => {
                if let Some(ref font) = self.bold {
                    font
                } else {
                    &self.regular
                }
            }
            _ => &self.regular,
        }
    }
}

/// Font manager
#[derive(Debug, Clone)]
pub struct FontManager {
    families: HashMap<String, FontFamily>,
}

impl FontManager {
    /// Create a new font manager
    #[must_use]
    pub fn new() -> Self {
        Self {
            families: HashMap::new(),
        }
    }

    /// Add font family
    pub fn add_family(&mut self, name: String, family: FontFamily) {
        self.families.insert(name, family);
    }

    /// Get font family
    #[must_use]
    pub fn get_family(&self, name: &str) -> Option<&FontFamily> {
        self.families.get(name)
    }

    /// Load a system font and register it under `name`.
    ///
    /// Searches well-known platform font directories in order, then falls back
    /// to the `OXIMEDIA_SYSTEM_FONT` environment variable.  Pure-Rust only —
    /// no fontconfig FFI, no CoreText FFI.
    ///
    /// # Errors
    ///
    /// Returns [`GraphicsError::FontError`] if no valid system font can be
    /// found or parsed.
    pub fn load_system_font(&mut self, name: &str) -> Result<()> {
        let bytes = find_system_font_bytes()?;
        let family = FontFamily::from_regular(bytes);
        self.add_family(name.to_string(), family);
        Ok(())
    }
}

impl Default for FontManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Font style
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FontStyle {
    /// Normal style
    Normal,
    /// Italic style
    Italic,
}

/// Font weight
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FontWeight {
    /// Normal weight
    Normal,
    /// Bold weight
    Bold,
}

/// Text alignment
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TextAlign {
    /// Left aligned
    Left,
    /// Center aligned
    Center,
    /// Right aligned
    Right,
    /// Justified
    Justify,
}

/// Vertical text alignment
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerticalAlign {
    /// Top aligned
    Top,
    /// Middle aligned
    Middle,
    /// Bottom aligned
    Bottom,
}

/// Text style
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextStyle {
    /// Font family name
    pub font_family: String,
    /// Font size in pixels
    pub font_size: f32,
    /// Font style
    pub font_style: FontStyle,
    /// Font weight
    pub font_weight: FontWeight,
    /// Text color
    pub color: Color,
    /// Line height multiplier
    pub line_height: f32,
    /// Letter spacing
    pub letter_spacing: f32,
    /// Text alignment
    pub align: TextAlign,
    /// Vertical alignment
    pub vertical_align: VerticalAlign,
}

impl TextStyle {
    /// Create a new text style
    #[must_use]
    pub fn new(font_family: String, font_size: f32, color: Color) -> Self {
        Self {
            font_family,
            font_size,
            font_style: FontStyle::Normal,
            font_weight: FontWeight::Normal,
            color,
            line_height: 1.2,
            letter_spacing: 0.0,
            align: TextAlign::Left,
            vertical_align: VerticalAlign::Top,
        }
    }

    /// Set font style
    #[must_use]
    pub fn with_style(mut self, style: FontStyle) -> Self {
        self.font_style = style;
        self
    }

    /// Set font weight
    #[must_use]
    pub fn with_weight(mut self, weight: FontWeight) -> Self {
        self.font_weight = weight;
        self
    }

    /// Set line height
    #[must_use]
    pub fn with_line_height(mut self, line_height: f32) -> Self {
        self.line_height = line_height;
        self
    }

    /// Set alignment
    #[must_use]
    pub fn with_align(mut self, align: TextAlign) -> Self {
        self.align = align;
        self
    }
}

impl Default for TextStyle {
    fn default() -> Self {
        Self::new("Arial".to_string(), 16.0, Color::BLACK)
    }
}

/// Text shadow effect
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TextShadow {
    /// Offset X
    pub offset_x: f32,
    /// Offset Y
    pub offset_y: f32,
    /// Blur radius
    pub blur: f32,
    /// Shadow color
    pub color: Color,
}

impl TextShadow {
    /// Create a new text shadow
    #[must_use]
    pub fn new(offset_x: f32, offset_y: f32, blur: f32, color: Color) -> Self {
        Self {
            offset_x,
            offset_y,
            blur,
            color,
        }
    }
}

/// Text outline
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TextOutline {
    /// Outline width
    pub width: f32,
    /// Outline color
    pub color: Color,
}

impl TextOutline {
    /// Create a new text outline
    #[must_use]
    pub fn new(width: f32, color: Color) -> Self {
        Self { width, color }
    }
}

/// Rich text segment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextSegment {
    /// Text content
    pub text: String,
    /// Style for this segment
    pub style: TextStyle,
}

impl TextSegment {
    /// Create a new text segment
    #[must_use]
    pub fn new(text: String, style: TextStyle) -> Self {
        Self { text, style }
    }
}

/// Multi-line text layout
#[derive(Debug, Clone)]
pub struct TextLayout {
    /// Text segments
    pub segments: Vec<TextSegment>,
    /// Maximum width (for wrapping)
    pub max_width: Option<f32>,
    /// Text shadow
    pub shadow: Option<TextShadow>,
    /// Text outline
    pub outline: Option<TextOutline>,
}

impl TextLayout {
    /// Create a new text layout
    #[must_use]
    pub fn new() -> Self {
        Self {
            segments: Vec::new(),
            max_width: None,
            shadow: None,
            outline: None,
        }
    }

    /// Add a text segment
    pub fn add_segment(&mut self, segment: TextSegment) -> &mut Self {
        self.segments.push(segment);
        self
    }

    /// Set maximum width
    #[must_use]
    pub fn with_max_width(mut self, width: f32) -> Self {
        self.max_width = Some(width);
        self
    }

    /// Set shadow
    #[must_use]
    pub fn with_shadow(mut self, shadow: TextShadow) -> Self {
        self.shadow = Some(shadow);
        self
    }

    /// Set outline
    #[must_use]
    pub fn with_outline(mut self, outline: TextOutline) -> Self {
        self.outline = Some(outline);
        self
    }

    /// Measure text bounds
    pub fn measure(&self, font_manager: &FontManager) -> Result<Rect> {
        let mut max_width = 0.0_f32;
        let mut total_height = 0.0_f32;

        for segment in &self.segments {
            let family = font_manager
                .get_family(&segment.style.font_family)
                .ok_or_else(|| {
                    GraphicsError::FontError(format!(
                        "Font family not found: {}",
                        segment.style.font_family
                    ))
                })?;

            let font_data = family.get_font(segment.style.font_style, segment.style.font_weight);
            let font = FontRef::try_from_slice(font_data)
                .map_err(|e| GraphicsError::FontError(format!("Failed to parse font: {e}")))?;

            let scale = PxScale::from(segment.style.font_size);
            let scaled_font = font.as_scaled(scale);

            let mut line_width = 0.0_f32;
            let mut prev_glyph_id = None;

            for ch in segment.text.chars() {
                let glyph_id = scaled_font.glyph_id(ch);
                let advance_width = scaled_font.h_advance(glyph_id);

                if let Some(prev_id) = prev_glyph_id {
                    line_width += scaled_font.kern(prev_id, glyph_id);
                }

                line_width += advance_width + segment.style.letter_spacing;
                prev_glyph_id = Some(glyph_id);
            }

            max_width = max_width.max(line_width);
            total_height += segment.style.font_size * segment.style.line_height;
        }

        Ok(Rect::new(0.0, 0.0, max_width, total_height))
    }
}

impl Default for TextLayout {
    fn default() -> Self {
        Self::new()
    }
}

/// Glyph position
#[derive(Debug, Clone)]
pub struct GlyphPosition {
    /// Character
    pub ch: char,
    /// Position
    pub position: Point,
    /// Font size
    pub size: f32,
    /// Color
    pub color: Color,
}

/// Text renderer
pub struct TextRenderer {
    font_manager: FontManager,
}

impl TextRenderer {
    /// Create a new text renderer
    #[must_use]
    pub fn new(font_manager: FontManager) -> Self {
        Self { font_manager }
    }

    /// Layout text and get glyph positions
    pub fn layout_glyphs(
        &self,
        layout: &TextLayout,
        position: Point,
    ) -> Result<Vec<GlyphPosition>> {
        let mut glyphs = Vec::new();
        let mut y = position.y;

        for segment in &layout.segments {
            let family = self
                .font_manager
                .get_family(&segment.style.font_family)
                .ok_or_else(|| {
                    GraphicsError::FontError(format!(
                        "Font family not found: {}",
                        segment.style.font_family
                    ))
                })?;

            let font_data = family.get_font(segment.style.font_style, segment.style.font_weight);
            let font = FontRef::try_from_slice(font_data)
                .map_err(|e| GraphicsError::FontError(format!("Failed to parse font: {e}")))?;

            let scale = PxScale::from(segment.style.font_size);
            let scaled_font = font.as_scaled(scale);

            let mut x = position.x;
            let mut prev_glyph_id = None;

            for ch in segment.text.chars() {
                if ch == '\n' {
                    y += segment.style.font_size * segment.style.line_height;
                    x = position.x;
                    prev_glyph_id = None;
                    continue;
                }

                let glyph_id = scaled_font.glyph_id(ch);

                if let Some(prev_id) = prev_glyph_id {
                    x += scaled_font.kern(prev_id, glyph_id);
                }

                glyphs.push(GlyphPosition {
                    ch,
                    position: Point::new(x, y),
                    size: segment.style.font_size,
                    color: segment.style.color,
                });

                x += scaled_font.h_advance(glyph_id) + segment.style.letter_spacing;
                prev_glyph_id = Some(glyph_id);
            }

            y += segment.style.font_size * segment.style.line_height;
        }

        Ok(glyphs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_font_family() {
        let family = FontFamily::from_regular(vec![0; 100]);
        assert!(family.bold.is_none());
        assert!(family.italic.is_none());
    }

    #[test]
    fn test_font_manager() {
        let mut manager = FontManager::new();
        let family = FontFamily::from_regular(vec![0; 100]);
        manager.add_family("Test".to_string(), family);
        assert!(manager.get_family("Test").is_some());
        assert!(manager.get_family("NotFound").is_none());
    }

    #[test]
    fn test_text_style() {
        let style = TextStyle::new("Arial".to_string(), 16.0, Color::BLACK)
            .with_style(FontStyle::Italic)
            .with_weight(FontWeight::Bold)
            .with_line_height(1.5)
            .with_align(TextAlign::Center);

        assert_eq!(style.font_size, 16.0);
        assert_eq!(style.font_style, FontStyle::Italic);
        assert_eq!(style.font_weight, FontWeight::Bold);
        assert_eq!(style.line_height, 1.5);
        assert_eq!(style.align, TextAlign::Center);
    }

    #[test]
    fn test_text_shadow() {
        let shadow = TextShadow::new(2.0, 2.0, 4.0, Color::new(0, 0, 0, 128));
        assert_eq!(shadow.offset_x, 2.0);
        assert_eq!(shadow.offset_y, 2.0);
        assert_eq!(shadow.blur, 4.0);
    }

    #[test]
    fn test_text_outline() {
        let outline = TextOutline::new(2.0, Color::BLACK);
        assert_eq!(outline.width, 2.0);
        assert_eq!(outline.color, Color::BLACK);
    }

    #[test]
    fn test_text_layout() {
        let mut layout = TextLayout::new();
        let style = TextStyle::default();
        layout.add_segment(TextSegment::new("Hello".to_string(), style));
        assert_eq!(layout.segments.len(), 1);
    }

    #[test]
    fn test_text_segment() {
        let style = TextStyle::default();
        let segment = TextSegment::new("Test".to_string(), style);
        assert_eq!(segment.text, "Test");
    }

    /// `load_system_font` either succeeds (finds a platform font) or returns a
    /// `FontError`.  It must never panic.
    ///
    /// This test validates the complete execution path through `load_system_font`
    /// including the `find_system_font_bytes` search loop and the `FontFamily`
    /// registration, on whatever host fonts are available.
    #[test]
    fn test_load_system_font_no_panic() {
        let mut manager = FontManager::new();
        let result = manager.load_system_font("SysFont");
        match result {
            Ok(()) => {
                // A platform font was found and registered.
                assert!(
                    manager.get_family("SysFont").is_some(),
                    "font family should be present after Ok result"
                );
            }
            Err(GraphicsError::FontError(_)) => {
                // No font on this host — correct error variant, no panic.
            }
            Err(other) => {
                panic!("Unexpected error variant from load_system_font: {other}");
            }
        }
    }

    /// `validate_font_bytes` rejects garbage bytes and returns a descriptive `FontError`.
    ///
    /// We write known-invalid bytes to `temp_dir()`, read them back, and then
    /// call `validate_font_bytes` — exactly mirroring what `find_system_font_bytes`
    /// does when it reads a corrupt or non-font file.
    #[test]
    fn test_validate_font_bytes_rejects_garbage() {
        // Write known-garbage bytes to a temp dir file.
        let mut tmp = std::env::temp_dir();
        tmp.push("oximedia_test_garbage_font_xyz987.ttf");

        let garbage: Vec<u8> = vec![0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x00, 0x00, 0x00];
        std::fs::write(&tmp, &garbage).expect("failed to write garbage font bytes to temp_dir");

        // Read back (mirrors what find_system_font_bytes does).
        let read_back = std::fs::read(&tmp).expect("temp file should be readable");

        // validate_font_bytes must return FontError for garbage.
        let result = validate_font_bytes(read_back);

        let _ = std::fs::remove_file(&tmp);

        assert!(
            result.is_err(),
            "validate_font_bytes must reject known-garbage bytes"
        );
        assert!(
            matches!(result, Err(GraphicsError::FontError(_))),
            "error must be GraphicsError::FontError"
        );
    }

    /// When a valid font is available on the system, copying it to `temp_dir()`
    /// and passing those bytes through `validate_font_bytes` must succeed, and the
    /// resulting bytes must produce a valid `FontFamily`.
    ///
    /// Gracefully skipped (early return) on hosts with no system fonts so CI
    /// does not fail in a minimal environment.
    #[test]
    fn test_font_family_from_valid_font_bytes_via_temp_dir() {
        // Locate a real, validate_font_bytes-parseable font from the platform list.
        let mut source_bytes: Option<Vec<u8>> = None;
        for path in SYSTEM_FONT_PATHS {
            if let Ok(bytes) = std::fs::read(path) {
                if validate_font_bytes(bytes.clone()).is_ok() {
                    source_bytes = Some(bytes);
                    break;
                }
            }
        }

        let bytes = match source_bytes {
            Some(b) => b,
            None => return, // no font on this host — skip gracefully
        };

        // Round-trip: write to temp dir then read back.
        let mut tmp = std::env::temp_dir();
        tmp.push("oximedia_test_valid_roundtrip_font.ttf");
        std::fs::write(&tmp, &bytes).expect("failed to write font to temp_dir");
        let read_back = std::fs::read(&tmp).expect("failed to read font from temp_dir");
        let _ = std::fs::remove_file(&tmp);

        // validate_font_bytes must accept the round-tripped bytes.
        let validated = validate_font_bytes(read_back);
        assert!(
            validated.is_ok(),
            "validate_font_bytes must accept round-tripped valid font bytes"
        );

        // Build a FontFamily from the validated bytes and verify registration.
        let family = FontFamily::from_regular(validated.expect("already checked Ok"));
        assert!(
            !family.regular.is_empty(),
            "FontFamily::regular must hold the font bytes"
        );

        let mut manager = FontManager::new();
        manager.add_family("RoundTrip".to_string(), family);
        assert!(
            manager.get_family("RoundTrip").is_some(),
            "FontFamily must be retrievable by name after add_family"
        );
    }
}
