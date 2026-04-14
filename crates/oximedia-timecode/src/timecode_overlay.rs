//! Timecode overlay module for rendering timecode as text overlay on video frames.
//!
//! This module provides the ability to render timecode values as text overlays
//! that can be composited onto video frames. It integrates with the burn-in
//! module and supports multiple display styles, positions, and formatting.
//!
//! # Features
//!
//! - Configurable overlay position (corners, center, custom coordinates)
//! - Multiple font size presets (small, medium, large, custom)
//! - Background box rendering with configurable opacity
//! - Drop-frame and non-drop-frame visual indicators
//! - Field dominance indicators for interlaced content
//! - Timecode + metadata (reel name, scene/take) overlay
//!
//! # Example
//!
//! ```rust
//! use oximedia_timecode::{Timecode, FrameRate};
//! use oximedia_timecode::timecode_overlay::{OverlayConfig, OverlayPosition, render_overlay};
//!
//! let tc = Timecode::new(1, 30, 0, 12, FrameRate::Fps25).expect("valid tc");
//! let config = OverlayConfig::default();
//! let overlay = render_overlay(&tc, &config);
//! assert!(!overlay.text.is_empty());
//! ```

#![allow(dead_code)]
#![allow(clippy::cast_possible_truncation)]

use crate::Timecode;

/// Position of the timecode overlay on the video frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum OverlayPosition {
    /// Top-left corner.
    TopLeft,
    /// Top-center.
    TopCenter,
    /// Top-right corner.
    TopRight,
    /// Bottom-left corner.
    BottomLeft,
    /// Bottom-center.
    BottomCenter,
    /// Bottom-right corner.
    BottomRight,
    /// Center of the frame.
    Center,
    /// Custom position in pixels from top-left origin.
    Custom { x: u32, y: u32 },
}

impl Default for OverlayPosition {
    fn default() -> Self {
        Self::BottomLeft
    }
}

/// Font size preset for the overlay text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum FontSize {
    /// Small (suitable for monitoring).
    Small,
    /// Medium (general purpose).
    Medium,
    /// Large (visible on small screens).
    Large,
    /// Custom pixel height.
    Custom(u32),
}

impl FontSize {
    /// Return the pixel height for this font size at the given frame height.
    #[must_use]
    pub fn pixel_height(&self, frame_height: u32) -> u32 {
        match self {
            FontSize::Small => frame_height / 40,
            FontSize::Medium => frame_height / 25,
            FontSize::Large => frame_height / 15,
            FontSize::Custom(h) => *h,
        }
    }
}

impl Default for FontSize {
    fn default() -> Self {
        Self::Medium
    }
}

/// RGBA colour (0-255 per channel).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Rgba {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Rgba {
    /// Opaque white.
    pub const WHITE: Self = Self {
        r: 255,
        g: 255,
        b: 255,
        a: 255,
    };
    /// Opaque black.
    pub const BLACK: Self = Self {
        r: 0,
        g: 0,
        b: 0,
        a: 255,
    };
    /// Semi-transparent black (for background boxes).
    pub const SEMI_BLACK: Self = Self {
        r: 0,
        g: 0,
        b: 0,
        a: 180,
    };
    /// Opaque red (for drop-frame indicator).
    pub const RED: Self = Self {
        r: 255,
        g: 0,
        b: 0,
        a: 255,
    };

    /// Create a new colour.
    #[must_use]
    pub const fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    /// Blend this colour over a background colour using alpha compositing.
    #[must_use]
    pub fn blend_over(&self, bg: &Rgba) -> Rgba {
        let sa = self.a as u32;
        let da = bg.a as u32;
        let inv_sa = 255 - sa;

        let out_a = sa + da * inv_sa / 255;
        if out_a == 0 {
            return Rgba::new(0, 0, 0, 0);
        }

        let blend = |fg: u8, bg_ch: u8| -> u8 {
            let v = (fg as u32 * sa + bg_ch as u32 * da * inv_sa / 255) / out_a;
            v.min(255) as u8
        };

        Rgba {
            r: blend(self.r, bg.r),
            g: blend(self.g, bg.g),
            b: blend(self.b, bg.b),
            a: out_a.min(255) as u8,
        }
    }
}

/// Configuration for the timecode overlay.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OverlayConfig {
    /// Position of the overlay on screen.
    pub position: OverlayPosition,
    /// Font size.
    pub font_size: FontSize,
    /// Foreground (text) colour.
    pub fg_color: Rgba,
    /// Background box colour (set alpha to 0 for no background).
    pub bg_color: Rgba,
    /// Margin in pixels from the edge of the frame.
    pub margin: u32,
    /// Whether to show a drop-frame indicator (red dot or "DF" suffix).
    pub show_df_indicator: bool,
    /// Whether to render a background box behind the text.
    pub show_background: bool,
    /// Optional prefix text (e.g., reel name or "REC").
    pub prefix: Option<String>,
    /// Optional suffix text (e.g., scene/take number).
    pub suffix: Option<String>,
    /// Whether to show field indicator (F1/F2) for interlaced content.
    pub show_field_indicator: bool,
    /// Current field (1 or 2) when `show_field_indicator` is true.
    pub current_field: u8,
}

impl Default for OverlayConfig {
    fn default() -> Self {
        Self {
            position: OverlayPosition::default(),
            font_size: FontSize::default(),
            fg_color: Rgba::WHITE,
            bg_color: Rgba::SEMI_BLACK,
            margin: 16,
            show_df_indicator: true,
            show_background: true,
            prefix: None,
            suffix: None,
            show_field_indicator: false,
            current_field: 1,
        }
    }
}

impl OverlayConfig {
    /// Create a config for monitoring (small, bottom-left, semi-transparent BG).
    #[must_use]
    pub fn monitoring() -> Self {
        Self {
            font_size: FontSize::Small,
            position: OverlayPosition::BottomLeft,
            ..Self::default()
        }
    }

    /// Create a config for burn-in (large, top-center, opaque BG).
    #[must_use]
    pub fn burn_in() -> Self {
        Self {
            font_size: FontSize::Large,
            position: OverlayPosition::TopCenter,
            bg_color: Rgba::BLACK,
            ..Self::default()
        }
    }

    /// Create a config with no background box.
    #[must_use]
    pub fn no_background() -> Self {
        Self {
            show_background: false,
            bg_color: Rgba::new(0, 0, 0, 0),
            ..Self::default()
        }
    }

    /// Set a prefix string.
    #[must_use]
    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = Some(prefix.into());
        self
    }

    /// Set a suffix string.
    #[must_use]
    pub fn with_suffix(mut self, suffix: impl Into<String>) -> Self {
        self.suffix = Some(suffix.into());
        self
    }

    /// Enable field indicator display.
    #[must_use]
    pub fn with_field(mut self, field: u8) -> Self {
        self.show_field_indicator = true;
        self.current_field = field.clamp(1, 2);
        self
    }
}

/// Rendered overlay result containing text, position info, and styling.
#[derive(Debug, Clone)]
pub struct RenderedOverlay {
    /// The formatted timecode text to display.
    pub text: String,
    /// The overlay position.
    pub position: OverlayPosition,
    /// Foreground colour.
    pub fg_color: Rgba,
    /// Background colour.
    pub bg_color: Rgba,
    /// Font pixel height (for a 1080p frame).
    pub font_height: u32,
    /// Margin in pixels.
    pub margin: u32,
    /// Whether a background box should be rendered.
    pub show_background: bool,
    /// Approximate text width in characters.
    pub text_char_width: usize,
}

/// Compute pixel coordinates for the overlay given frame dimensions.
///
/// Returns `(x, y)` coordinates for the top-left corner of the overlay text.
#[must_use]
pub fn compute_position(
    position: &OverlayPosition,
    frame_width: u32,
    frame_height: u32,
    text_width_px: u32,
    text_height_px: u32,
    margin: u32,
) -> (u32, u32) {
    match position {
        OverlayPosition::TopLeft => (margin, margin),
        OverlayPosition::TopCenter => {
            let x = frame_width.saturating_sub(text_width_px) / 2;
            (x, margin)
        }
        OverlayPosition::TopRight => {
            let x = frame_width.saturating_sub(text_width_px + margin);
            (x, margin)
        }
        OverlayPosition::BottomLeft => {
            let y = frame_height.saturating_sub(text_height_px + margin);
            (margin, y)
        }
        OverlayPosition::BottomCenter => {
            let x = frame_width.saturating_sub(text_width_px) / 2;
            let y = frame_height.saturating_sub(text_height_px + margin);
            (x, y)
        }
        OverlayPosition::BottomRight => {
            let x = frame_width.saturating_sub(text_width_px + margin);
            let y = frame_height.saturating_sub(text_height_px + margin);
            (x, y)
        }
        OverlayPosition::Center => {
            let x = frame_width.saturating_sub(text_width_px) / 2;
            let y = frame_height.saturating_sub(text_height_px) / 2;
            (x, y)
        }
        OverlayPosition::Custom { x, y } => (*x, *y),
    }
}

/// Render a timecode overlay with the given configuration.
///
/// This produces a [`RenderedOverlay`] containing the formatted text and
/// all styling information needed to composite the overlay onto a video frame.
#[must_use]
pub fn render_overlay(tc: &Timecode, config: &OverlayConfig) -> RenderedOverlay {
    let separator = if tc.frame_rate.drop_frame { ';' } else { ':' };

    let mut text = String::new();

    // Prefix
    if let Some(ref prefix) = config.prefix {
        text.push_str(prefix);
        text.push(' ');
    }

    // Timecode
    text.push_str(&format!(
        "{:02}:{:02}:{:02}{}{:02}",
        tc.hours, tc.minutes, tc.seconds, separator, tc.frames
    ));

    // Drop-frame indicator
    if config.show_df_indicator && tc.frame_rate.drop_frame {
        text.push_str(" DF");
    }

    // Field indicator
    if config.show_field_indicator {
        text.push_str(&format!(" F{}", config.current_field));
    }

    // Suffix
    if let Some(ref suffix) = config.suffix {
        text.push(' ');
        text.push_str(suffix);
    }

    let font_height = config.font_size.pixel_height(1080);
    let text_char_width = text.len();

    RenderedOverlay {
        text,
        position: config.position,
        fg_color: config.fg_color,
        bg_color: config.bg_color,
        font_height,
        margin: config.margin,
        show_background: config.show_background,
        text_char_width,
    }
}

/// Estimate the pixel width of the overlay text for a monospaced font.
///
/// Assumes each character is approximately 0.6x the font height.
#[must_use]
pub fn estimate_text_width(text_len: usize, font_height: u32) -> u32 {
    let char_width = (font_height as f64 * 0.6).ceil() as u32;
    text_len as u32 * char_width
}

/// Render a background box behind the overlay text.
///
/// Returns `(x, y, width, height)` of the background rectangle with padding.
#[must_use]
pub fn background_rect(
    text_x: u32,
    text_y: u32,
    text_width_px: u32,
    font_height: u32,
    padding: u32,
) -> (u32, u32, u32, u32) {
    let x = text_x.saturating_sub(padding);
    let y = text_y.saturating_sub(padding);
    let w = text_width_px + padding * 2;
    let h = font_height + padding * 2;
    (x, y, w, h)
}

/// Batch render overlays for a sequence of timecodes.
///
/// Useful for pre-computing overlay data for an entire timeline segment.
#[must_use]
pub fn render_batch(timecodes: &[Timecode], config: &OverlayConfig) -> Vec<RenderedOverlay> {
    timecodes
        .iter()
        .map(|tc| render_overlay(tc, config))
        .collect()
}

/// A timecode overlay layer that can stamp successive frames.
///
/// Holds configuration and provides a stateful interface for frame-by-frame
/// overlay rendering with optional automatic timecode advancement.
#[derive(Debug, Clone)]
pub struct OverlayStamper {
    config: OverlayConfig,
    frame_width: u32,
    frame_height: u32,
}

impl OverlayStamper {
    /// Create a new overlay stamper for a given frame size.
    #[must_use]
    pub fn new(config: OverlayConfig, frame_width: u32, frame_height: u32) -> Self {
        Self {
            config,
            frame_width,
            frame_height,
        }
    }

    /// Render overlay for a single frame and return pixel coordinates + text.
    #[must_use]
    pub fn stamp(&self, tc: &Timecode) -> (u32, u32, RenderedOverlay) {
        let overlay = render_overlay(tc, &self.config);
        let text_width = estimate_text_width(overlay.text_char_width, overlay.font_height);
        let (x, y) = compute_position(
            &overlay.position,
            self.frame_width,
            self.frame_height,
            text_width,
            overlay.font_height,
            overlay.margin,
        );
        (x, y, overlay)
    }

    /// Get the frame dimensions.
    #[must_use]
    pub fn frame_size(&self) -> (u32, u32) {
        (self.frame_width, self.frame_height)
    }

    /// Get a reference to the configuration.
    #[must_use]
    pub fn config(&self) -> &OverlayConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FrameRate;

    fn tc25(h: u8, m: u8, s: u8, f: u8) -> Timecode {
        Timecode::new(h, m, s, f, FrameRate::Fps25).expect("valid tc")
    }

    fn tc_df(h: u8, m: u8, s: u8, f: u8) -> Timecode {
        Timecode::new(h, m, s, f, FrameRate::Fps2997DF).expect("valid tc")
    }

    #[test]
    fn test_render_overlay_basic() {
        let tc = tc25(1, 30, 0, 12);
        let config = OverlayConfig::default();
        let overlay = render_overlay(&tc, &config);
        assert_eq!(overlay.text, "01:30:00:12");
        assert_eq!(overlay.position, OverlayPosition::BottomLeft);
    }

    #[test]
    fn test_render_overlay_drop_frame_indicator() {
        let tc = tc_df(0, 1, 0, 2);
        let config = OverlayConfig::default();
        let overlay = render_overlay(&tc, &config);
        assert!(overlay.text.contains(';'));
        assert!(overlay.text.contains("DF"));
    }

    #[test]
    fn test_render_overlay_no_df_indicator() {
        let tc = tc_df(0, 1, 0, 2);
        let mut config = OverlayConfig::default();
        config.show_df_indicator = false;
        let overlay = render_overlay(&tc, &config);
        assert!(!overlay.text.contains("DF"));
    }

    #[test]
    fn test_render_overlay_with_prefix_and_suffix() {
        let tc = tc25(0, 0, 0, 0);
        let config = OverlayConfig::default()
            .with_prefix("REC")
            .with_suffix("SC1/TK3");
        let overlay = render_overlay(&tc, &config);
        assert!(overlay.text.starts_with("REC "));
        assert!(overlay.text.ends_with("SC1/TK3"));
    }

    #[test]
    fn test_render_overlay_with_field_indicator() {
        let tc = tc25(0, 0, 0, 0);
        let config = OverlayConfig::default().with_field(2);
        let overlay = render_overlay(&tc, &config);
        assert!(overlay.text.contains("F2"));
    }

    #[test]
    fn test_compute_position_corners() {
        let (x, y) = compute_position(&OverlayPosition::TopLeft, 1920, 1080, 200, 40, 16);
        assert_eq!(x, 16);
        assert_eq!(y, 16);

        let (x, y) = compute_position(&OverlayPosition::BottomRight, 1920, 1080, 200, 40, 16);
        assert_eq!(x, 1920 - 200 - 16);
        assert_eq!(y, 1080 - 40 - 16);
    }

    #[test]
    fn test_compute_position_center() {
        let (x, y) = compute_position(&OverlayPosition::Center, 1920, 1080, 200, 40, 16);
        assert_eq!(x, (1920 - 200) / 2);
        assert_eq!(y, (1080 - 40) / 2);
    }

    #[test]
    fn test_compute_position_custom() {
        let (x, y) = compute_position(
            &OverlayPosition::Custom { x: 100, y: 200 },
            1920,
            1080,
            200,
            40,
            16,
        );
        assert_eq!(x, 100);
        assert_eq!(y, 200);
    }

    #[test]
    fn test_font_size_pixel_height() {
        assert_eq!(FontSize::Small.pixel_height(1080), 27);
        assert_eq!(FontSize::Medium.pixel_height(1080), 43);
        assert_eq!(FontSize::Large.pixel_height(1080), 72);
        assert_eq!(FontSize::Custom(50).pixel_height(1080), 50);
    }

    #[test]
    fn test_estimate_text_width() {
        let w = estimate_text_width(11, 40); // "01:30:00:12" = 11 chars
        assert_eq!(w, 11 * 24); // 0.6 * 40 = 24
    }

    #[test]
    fn test_background_rect() {
        let (x, y, w, h) = background_rect(100, 200, 264, 40, 8);
        assert_eq!(x, 92);
        assert_eq!(y, 192);
        assert_eq!(w, 280);
        assert_eq!(h, 56);
    }

    #[test]
    fn test_render_batch() {
        let tcs = vec![tc25(0, 0, 0, 0), tc25(0, 0, 0, 1), tc25(0, 0, 0, 2)];
        let config = OverlayConfig::default();
        let overlays = render_batch(&tcs, &config);
        assert_eq!(overlays.len(), 3);
        assert!(overlays[1].text.contains("01"));
    }

    #[test]
    fn test_overlay_stamper() {
        let config = OverlayConfig::monitoring();
        let stamper = OverlayStamper::new(config, 1920, 1080);
        let tc = tc25(12, 0, 0, 0);
        let (x, y, overlay) = stamper.stamp(&tc);
        assert!(x < 1920);
        assert!(y < 1080);
        assert!(overlay.text.contains("12:00:00:00"));
        assert_eq!(stamper.frame_size(), (1920, 1080));
    }

    #[test]
    fn test_rgba_blend_over() {
        let fg = Rgba::new(255, 0, 0, 128);
        let bg = Rgba::WHITE;
        let result = fg.blend_over(&bg);
        // Partially red over white should be pinkish
        assert!(result.r > result.g);
        assert_eq!(result.a, 255);
    }

    #[test]
    fn test_monitoring_preset() {
        let config = OverlayConfig::monitoring();
        assert_eq!(config.font_size, FontSize::Small);
        assert_eq!(config.position, OverlayPosition::BottomLeft);
    }

    #[test]
    fn test_burn_in_preset() {
        let config = OverlayConfig::burn_in();
        assert_eq!(config.font_size, FontSize::Large);
        assert_eq!(config.position, OverlayPosition::TopCenter);
        assert_eq!(config.bg_color, Rgba::BLACK);
    }
}
