//! Caption style presets for popular broadcast standards.
//!
//! Provides ready-made [`CaptionStyleConfig`] values conforming to the
//! Netflix, BBC, and WCAG caption style guidelines.
//!
//! # References
//!
//! - **Netflix**: Timed Text Style Guide (2023)
//! - **BBC**: Subtitle Guidelines (2022)
//! - **WCAG 2.1**: Success Criterion 1.4.3 / 1.4.6 Contrast requirements
//!
//! # Example
//!
//! ```rust
//! use oximedia_caption_gen::style_presets::CaptionStyle;
//!
//! let netflix = CaptionStyle::netflix();
//! assert_eq!(netflix.font_size_px, 40);
//! assert!(netflix.background_opacity > 0.0);
//!
//! let bbc = CaptionStyle::bbc();
//! assert_eq!(bbc.text_color, "#FFFFFF");
//!
//! let wcag = CaptionStyle::wcag();
//! assert!(wcag.contrast_ratio >= 4.5);
//! ```

/// Horizontal alignment of caption text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextAlignment {
    /// Left-aligned text.
    Left,
    /// Centred text.
    Center,
    /// Right-aligned text.
    Right,
}

/// Font weight options.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FontWeight {
    /// Normal (400) weight.
    Normal,
    /// Bold (700) weight.
    Bold,
}

/// Caption position on the video frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptionPlacement {
    /// Bottom of the frame (most common).
    BottomCenter,
    /// Top of the frame (for sign-language windows).
    TopCenter,
    /// Custom percentage-based position (x%, y%).
    Custom { x_pct: u8, y_pct: u8 },
}

/// A complete caption visual style configuration.
///
/// Values are chosen to match the respective broadcast or accessibility
/// standard as closely as possible in a platform-neutral representation.
#[derive(Debug, Clone)]
pub struct CaptionStyleConfig {
    /// Font size in CSS pixels.
    pub font_size_px: u32,
    /// Font family (CSS font stack).
    pub font_family: &'static str,
    /// Font weight.
    pub font_weight: FontWeight,
    /// Foreground (text) colour as a CSS hex string (e.g. `"#FFFFFF"`).
    pub text_color: &'static str,
    /// Background colour as a CSS hex string.
    pub background_color: &'static str,
    /// Background opacity in `[0.0, 1.0]`.
    pub background_opacity: f32,
    /// Text shadow / outline (CSS `text-shadow` value, or empty).
    pub text_shadow: &'static str,
    /// Horizontal text alignment.
    pub alignment: TextAlignment,
    /// Caption placement on screen.
    pub placement: CaptionPlacement,
    /// Minimum contrast ratio enforced by this style (informational).
    pub contrast_ratio: f32,
    /// Maximum lines visible simultaneously.
    pub max_lines: u8,
    /// Maximum characters per line.
    pub max_chars_per_line: u8,
    /// Name of this preset for diagnostics.
    pub preset_name: &'static str,
}

/// Caption style preset factory.
pub struct CaptionStyle;

impl CaptionStyle {
    /// Netflix Timed Text Style Guide preset.
    ///
    /// White 40 px Proportional Serif, semi-transparent black background,
    /// bottom-centred, max 2 lines × 42 chars.
    #[must_use]
    pub fn netflix() -> CaptionStyleConfig {
        CaptionStyleConfig {
            font_size_px: 40,
            font_family: "Arial, Helvetica, sans-serif",
            font_weight: FontWeight::Normal,
            text_color: "#FFFFFF",
            background_color: "#000000",
            background_opacity: 0.8,
            text_shadow: "0 0 4px #000000",
            alignment: TextAlignment::Center,
            placement: CaptionPlacement::BottomCenter,
            contrast_ratio: 7.0,
            max_lines: 2,
            max_chars_per_line: 42,
            preset_name: "Netflix",
        }
    }

    /// BBC Subtitle Guidelines preset.
    ///
    /// White 38 px Reith Sans, 75% opacity black box, bottom-centred,
    /// max 2 lines × 37 chars.
    #[must_use]
    pub fn bbc() -> CaptionStyleConfig {
        CaptionStyleConfig {
            font_size_px: 38,
            font_family: "BBC Reith Sans, Arial, sans-serif",
            font_weight: FontWeight::Normal,
            text_color: "#FFFFFF",
            background_color: "#000000",
            background_opacity: 0.75,
            text_shadow: "",
            alignment: TextAlignment::Center,
            placement: CaptionPlacement::BottomCenter,
            contrast_ratio: 5.0,
            max_lines: 2,
            max_chars_per_line: 37,
            preset_name: "BBC",
        }
    }

    /// WCAG 2.1 AA/AAA compliant preset.
    ///
    /// Meets WCAG 2.1 Success Criterion 1.4.3 (AA: 4.5:1) and 1.4.6 (AAA: 7:1)
    /// with a high-contrast white-on-black style and explicit text outline.
    #[must_use]
    pub fn wcag() -> CaptionStyleConfig {
        CaptionStyleConfig {
            font_size_px: 36,
            font_family: "Arial, Helvetica, sans-serif",
            font_weight: FontWeight::Bold,
            text_color: "#FFFFFF",
            background_color: "#000000",
            background_opacity: 1.0,
            text_shadow: "1px 1px 2px #000000, -1px -1px 2px #000000",
            alignment: TextAlignment::Center,
            placement: CaptionPlacement::BottomCenter,
            contrast_ratio: 7.1,
            max_lines: 3,
            max_chars_per_line: 40,
            preset_name: "WCAG",
        }
    }

    /// Apple TV+/iTunes delivery preset.
    ///
    /// White 40 px Helvetica Neue, drop-shadow only (no box background),
    /// bottom-centred, max 2 lines × 40 chars.
    #[must_use]
    pub fn apple_tv() -> CaptionStyleConfig {
        CaptionStyleConfig {
            font_size_px: 40,
            font_family: "Helvetica Neue, Arial, sans-serif",
            font_weight: FontWeight::Normal,
            text_color: "#FFFFFF",
            background_color: "#000000",
            background_opacity: 0.0,
            text_shadow: "0 2px 4px rgba(0,0,0,0.9)",
            alignment: TextAlignment::Center,
            placement: CaptionPlacement::BottomCenter,
            contrast_ratio: 4.5,
            max_lines: 2,
            max_chars_per_line: 40,
            preset_name: "AppleTV",
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_netflix_preset_font_size() {
        let p = CaptionStyle::netflix();
        assert_eq!(p.font_size_px, 40);
    }

    #[test]
    fn test_netflix_preset_max_lines() {
        let p = CaptionStyle::netflix();
        assert_eq!(p.max_lines, 2);
    }

    #[test]
    fn test_bbc_text_color_white() {
        let p = CaptionStyle::bbc();
        assert_eq!(p.text_color, "#FFFFFF");
    }

    #[test]
    fn test_bbc_max_chars_per_line() {
        let p = CaptionStyle::bbc();
        assert_eq!(p.max_chars_per_line, 37);
    }

    #[test]
    fn test_wcag_contrast_ratio_aaaa() {
        let p = CaptionStyle::wcag();
        // WCAG AAA requires ≥ 7:1 contrast ratio.
        assert!(
            p.contrast_ratio >= 7.0,
            "WCAG preset must meet AAA contrast: {}",
            p.contrast_ratio
        );
    }

    #[test]
    fn test_wcag_fully_opaque_background() {
        let p = CaptionStyle::wcag();
        assert!((p.background_opacity - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_all_presets_have_names() {
        for preset in [
            CaptionStyle::netflix(),
            CaptionStyle::bbc(),
            CaptionStyle::wcag(),
            CaptionStyle::apple_tv(),
        ] {
            assert!(!preset.preset_name.is_empty());
        }
    }

    #[test]
    fn test_all_presets_bottom_centred() {
        for preset in [
            CaptionStyle::netflix(),
            CaptionStyle::bbc(),
            CaptionStyle::wcag(),
        ] {
            assert_eq!(preset.placement, CaptionPlacement::BottomCenter);
        }
    }

    #[test]
    fn test_contrast_ratio_positive() {
        for preset in [
            CaptionStyle::netflix(),
            CaptionStyle::bbc(),
            CaptionStyle::wcag(),
        ] {
            assert!(preset.contrast_ratio > 0.0);
        }
    }
}
