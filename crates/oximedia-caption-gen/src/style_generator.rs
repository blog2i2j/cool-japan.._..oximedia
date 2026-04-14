//! Style suggestion engine for caption rendering.
//!
//! This module analyses video frame metadata (luminance, dominant colour,
//! content complexity) and suggests [`CaptionStyleSuggestion`] values —
//! font size, position, and text/background colours — that maximise legibility
//! against the underlying video.
//!
//! The engine operates entirely in pure Rust with no external dependencies.
//! Colour contrast calculations follow the WCAG 2.1 relative luminance formula
//! (IEC 61966-2-1 sRGB linearisation).
//!
//! ## Quick start
//!
//! ```rust
//! use oximedia_caption_gen::style_generator::{StyleGenerator, FrameAnalysis, ContentZone};
//!
//! let frame = FrameAnalysis {
//!     average_luminance: 0.7,
//!     dominant_rgb: [200, 180, 160],
//!     content_complexity: 0.3,
//!     safe_zones: vec![ContentZone::BottomThird],
//!     frame_width: 1920,
//!     frame_height: 1080,
//! };
//!
//! let suggestion = StyleGenerator::suggest(&frame);
//! assert!(suggestion.contrast_ratio >= 4.5);
//! ```

// ─── Types ────────────────────────────────────────────────────────────────────

/// A region of the video frame that is safe to place captions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContentZone {
    /// Lower third of the frame (default caption area).
    BottomThird,
    /// Upper third of the frame.
    TopThird,
    /// Centre of the frame (avoid unless top/bottom are occupied).
    Centre,
    /// Custom region expressed as pixel coordinates `(x, y, width, height)`.
    Custom {
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    },
}

/// Per-frame visual metadata used to drive style suggestions.
///
/// Luminance values are normalised to `[0.0, 1.0]` (0 = black, 1 = white).
#[derive(Debug, Clone)]
pub struct FrameAnalysis {
    /// Average luminance of the caption placement zone (0.0–1.0).
    pub average_luminance: f32,
    /// Dominant colour in the placement zone as sRGB `[R, G, B]`.
    pub dominant_rgb: [u8; 3],
    /// Perceptual content complexity in `[0.0, 1.0]`.
    ///
    /// 0.0 = flat/uniform content (easy background), 1.0 = highly textured.
    pub content_complexity: f32,
    /// Ordered list of available safe zones (first = preferred).
    pub safe_zones: Vec<ContentZone>,
    /// Video frame width in pixels.
    pub frame_width: u32,
    /// Video frame height in pixels.
    pub frame_height: u32,
}

impl FrameAnalysis {
    /// Preferred safe zone, falling back to [`ContentZone::BottomThird`].
    pub fn preferred_zone(&self) -> ContentZone {
        self.safe_zones
            .first()
            .copied()
            .unwrap_or(ContentZone::BottomThird)
    }

    /// Returns `true` if the frame is considered "bright" (luminance > 0.5).
    pub fn is_bright(&self) -> bool {
        self.average_luminance > 0.5
    }

    /// Returns `true` if the frame has complex / busy content.
    pub fn is_complex(&self) -> bool {
        self.content_complexity > 0.6
    }
}

/// Suggested background treatment for caption text.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BackgroundTreatment {
    /// Solid opaque box.
    SolidBox,
    /// Semi-transparent box (drop-shadow to help on complex backgrounds).
    SemiTransparentBox { opacity: f32 },
    /// Drop shadow only (no box).
    DropShadowOnly,
    /// No background treatment (only suitable on very uniform backgrounds).
    None,
}

/// A complete set of caption style recommendations.
#[derive(Debug, Clone)]
pub struct CaptionStyleSuggestion {
    /// Recommended font size in pixels.
    pub font_size_px: u32,
    /// Text colour as sRGB `[R, G, B]`.
    pub text_rgb: [u8; 3],
    /// Background colour as sRGB `[R, G, B]`.
    pub background_rgb: [u8; 3],
    /// Background rendering strategy.
    pub background_treatment: BackgroundTreatment,
    /// Placement zone for the caption.
    pub placement_zone: ContentZone,
    /// WCAG relative luminance contrast ratio of text over background.
    pub contrast_ratio: f32,
    /// Whether to apply a text outline (useful on complex backgrounds).
    pub use_text_outline: bool,
    /// Reasoning behind the suggestion (human-readable).
    pub reason: String,
}

impl CaptionStyleSuggestion {
    /// Returns `true` if the suggestion meets WCAG 2.1 AA contrast (4.5:1).
    pub fn meets_wcag_aa(&self) -> bool {
        self.contrast_ratio >= 4.5
    }

    /// Returns `true` if the suggestion meets WCAG 2.1 AAA contrast (7.0:1).
    pub fn meets_wcag_aaa(&self) -> bool {
        self.contrast_ratio >= 7.0
    }
}

// ─── StyleGenerator ───────────────────────────────────────────────────────────

/// Stateless caption style suggestion engine.
pub struct StyleGenerator;

impl StyleGenerator {
    /// Generate a [`CaptionStyleSuggestion`] for the given frame analysis.
    ///
    /// The suggestion selects text and background colours to maximise the WCAG
    /// contrast ratio, adapts font size to the frame height, chooses a
    /// background treatment based on content complexity, and places the caption
    /// in the best available safe zone.
    #[must_use]
    pub fn suggest(frame: &FrameAnalysis) -> CaptionStyleSuggestion {
        // Choose text colour: white on dark frames, black on bright frames.
        let (text_rgb, background_rgb) = Self::choose_colours(frame);

        let contrast_ratio = compute_contrast_ratio(text_rgb, background_rgb);

        let background_treatment = Self::choose_background_treatment(frame, contrast_ratio);
        let font_size_px = Self::choose_font_size(frame);
        let placement_zone = Self::choose_placement(frame);
        let use_text_outline = frame.is_complex() || contrast_ratio < 7.0;

        let reason = Self::build_reason(frame, &text_rgb, contrast_ratio, &background_treatment);

        CaptionStyleSuggestion {
            font_size_px,
            text_rgb,
            background_rgb,
            background_treatment,
            placement_zone,
            contrast_ratio,
            use_text_outline,
            reason,
        }
    }

    /// Suggest a font size based on frame height and content complexity.
    ///
    /// Larger frames get proportionally larger font sizes, and complex frames
    /// use a slightly larger size for legibility.
    #[must_use]
    pub fn suggest_font_size(frame_height: u32, content_complexity: f32) -> u32 {
        let base = (frame_height as f32 * 0.05).round() as u32;
        let base = base.max(24).min(72);
        if content_complexity > 0.6 {
            (base as f32 * 1.1).round().min(80.0) as u32
        } else {
            base
        }
    }

    // ─── Private helpers ──────────────────────────────────────────────────────

    fn choose_colours(frame: &FrameAnalysis) -> ([u8; 3], [u8; 3]) {
        if frame.is_bright() {
            // Bright frame: dark text on light background.
            ([0, 0, 0], [255, 255, 255])
        } else {
            // Dark frame: white text on black background.
            ([255, 255, 255], [0, 0, 0])
        }
    }

    fn choose_background_treatment(
        frame: &FrameAnalysis,
        contrast_ratio: f32,
    ) -> BackgroundTreatment {
        if contrast_ratio >= 7.0 && !frame.is_complex() {
            // High contrast + simple background: drop shadow is enough.
            BackgroundTreatment::DropShadowOnly
        } else if frame.is_complex() {
            // Complex background: solid box ensures legibility.
            BackgroundTreatment::SolidBox
        } else {
            // Default: semi-transparent box with 80% opacity.
            BackgroundTreatment::SemiTransparentBox { opacity: 0.80 }
        }
    }

    fn choose_font_size(frame: &FrameAnalysis) -> u32 {
        Self::suggest_font_size(frame.frame_height, frame.content_complexity)
    }

    fn choose_placement(frame: &FrameAnalysis) -> ContentZone {
        // Prefer the caller's preferred zone if it is BottomThird or TopThird.
        // Otherwise fall back to BottomThird.
        let preferred = frame.preferred_zone();
        match preferred {
            ContentZone::BottomThird | ContentZone::TopThird | ContentZone::Custom { .. } => {
                preferred
            }
            ContentZone::Centre => {
                // Centre is undesirable; try first non-Centre zone or fall back.
                frame
                    .safe_zones
                    .iter()
                    .find(|&&z| z != ContentZone::Centre)
                    .copied()
                    .unwrap_or(ContentZone::BottomThird)
            }
        }
    }

    fn build_reason(
        frame: &FrameAnalysis,
        text_rgb: &[u8; 3],
        contrast_ratio: f32,
        treatment: &BackgroundTreatment,
    ) -> String {
        let brightness = if frame.is_bright() { "bright" } else { "dark" };
        let complexity = if frame.is_complex() {
            "complex"
        } else {
            "simple"
        };
        let text_desc = if text_rgb[0] > 128 { "white" } else { "black" };
        let treatment_desc = match treatment {
            BackgroundTreatment::SolidBox => "solid box",
            BackgroundTreatment::SemiTransparentBox { opacity } => {
                &format!("{:.0}% opacity box", opacity * 100.0)
            }
            BackgroundTreatment::DropShadowOnly => "drop shadow",
            BackgroundTreatment::None => "none",
        };
        format!(
            "Frame is {brightness}/{complexity}: {text_desc} text with {treatment_desc} \
             (contrast {contrast_ratio:.1}:1)"
        )
    }
}

// ─── WCAG luminance helpers ───────────────────────────────────────────────────

/// Compute the WCAG 2.1 relative luminance of an sRGB colour.
///
/// Uses the IEC 61966-2-1 linearisation formula.
/// <https://www.w3.org/TR/WCAG21/#dfn-relative-luminance>
#[must_use]
pub fn relative_luminance(rgb: [u8; 3]) -> f32 {
    fn linearise(c: u8) -> f32 {
        let s = c as f32 / 255.0;
        if s <= 0.04045 {
            s / 12.92
        } else {
            ((s + 0.055) / 1.055).powf(2.4)
        }
    }
    let r = linearise(rgb[0]);
    let g = linearise(rgb[1]);
    let b = linearise(rgb[2]);
    0.2126 * r + 0.7152 * g + 0.0722 * b
}

/// Compute the WCAG 2.1 contrast ratio between two sRGB colours.
///
/// Returns a value ≥ 1.0.  A ratio of 21.0 represents maximum contrast
/// (black vs white).
#[must_use]
pub fn compute_contrast_ratio(foreground: [u8; 3], background: [u8; 3]) -> f32 {
    let l1 = relative_luminance(foreground);
    let l2 = relative_luminance(background);
    let (lighter, darker) = if l1 > l2 { (l1, l2) } else { (l2, l1) };
    (lighter + 0.05) / (darker + 0.05)
}

/// Determine whether `foreground` over `background` meets WCAG AA (4.5:1).
#[must_use]
pub fn meets_wcag_aa(foreground: [u8; 3], background: [u8; 3]) -> bool {
    compute_contrast_ratio(foreground, background) >= 4.5
}

/// Determine whether `foreground` over `background` meets WCAG AAA (7.0:1).
#[must_use]
pub fn meets_wcag_aaa(foreground: [u8; 3], background: [u8; 3]) -> bool {
    compute_contrast_ratio(foreground, background) >= 7.0
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn dark_frame() -> FrameAnalysis {
        FrameAnalysis {
            average_luminance: 0.1,
            dominant_rgb: [20, 20, 20],
            content_complexity: 0.2,
            safe_zones: vec![ContentZone::BottomThird],
            frame_width: 1920,
            frame_height: 1080,
        }
    }

    fn bright_frame() -> FrameAnalysis {
        FrameAnalysis {
            average_luminance: 0.85,
            dominant_rgb: [220, 210, 200],
            content_complexity: 0.1,
            safe_zones: vec![ContentZone::BottomThird],
            frame_width: 1920,
            frame_height: 1080,
        }
    }

    fn complex_frame() -> FrameAnalysis {
        FrameAnalysis {
            average_luminance: 0.4,
            dominant_rgb: [100, 100, 100],
            content_complexity: 0.85,
            safe_zones: vec![ContentZone::BottomThird],
            frame_width: 1920,
            frame_height: 1080,
        }
    }

    // ── relative_luminance ────────────────────────────────────────────────────

    #[test]
    fn black_has_zero_luminance() {
        let l = relative_luminance([0, 0, 0]);
        assert!(l.abs() < 1e-6, "black luminance should be 0, got {l}");
    }

    #[test]
    fn white_has_unit_luminance() {
        let l = relative_luminance([255, 255, 255]);
        assert!(
            (l - 1.0).abs() < 0.001,
            "white luminance should be ~1.0, got {l}"
        );
    }

    #[test]
    fn red_has_expected_luminance() {
        // Pure red: L ≈ 0.2126
        let l = relative_luminance([255, 0, 0]);
        assert!((l - 0.2126).abs() < 0.01, "red luminance ≈ 0.2126, got {l}");
    }

    // ── compute_contrast_ratio ────────────────────────────────────────────────

    #[test]
    fn white_on_black_maximum_contrast() {
        let ratio = compute_contrast_ratio([255, 255, 255], [0, 0, 0]);
        assert!(
            (ratio - 21.0).abs() < 0.1,
            "white/black ratio ≈ 21.0, got {ratio}"
        );
    }

    #[test]
    fn identical_colours_minimum_contrast() {
        let ratio = compute_contrast_ratio([128, 128, 128], [128, 128, 128]);
        assert!(
            (ratio - 1.0).abs() < 0.01,
            "identical colours → 1.0, got {ratio}"
        );
    }

    #[test]
    fn contrast_ratio_is_symmetric() {
        let fwd = compute_contrast_ratio([255, 255, 255], [0, 0, 0]);
        let rev = compute_contrast_ratio([0, 0, 0], [255, 255, 255]);
        assert!((fwd - rev).abs() < 1e-5, "ratio should be symmetric");
    }

    // ── meets_wcag_aa / aaa ────────────────────────────────────────────────────

    #[test]
    fn white_on_black_meets_both_aa_and_aaa() {
        assert!(meets_wcag_aa([255, 255, 255], [0, 0, 0]));
        assert!(meets_wcag_aaa([255, 255, 255], [0, 0, 0]));
    }

    #[test]
    fn low_contrast_pair_fails_aa() {
        // Mid-grey on slightly lighter grey: contrast < 4.5
        assert!(!meets_wcag_aa([120, 120, 120], [160, 160, 160]));
    }

    // ── StyleGenerator::suggest ───────────────────────────────────────────────

    #[test]
    fn dark_frame_uses_white_text() {
        let suggestion = StyleGenerator::suggest(&dark_frame());
        assert_eq!(
            suggestion.text_rgb,
            [255, 255, 255],
            "dark frame should use white text"
        );
    }

    #[test]
    fn bright_frame_uses_dark_text() {
        let suggestion = StyleGenerator::suggest(&bright_frame());
        // Dark text on bright background.
        assert!(
            suggestion.text_rgb[0] < 128,
            "bright frame should use dark text, got {:?}",
            suggestion.text_rgb
        );
    }

    #[test]
    fn suggestion_always_meets_wcag_aa() {
        for frame in [dark_frame(), bright_frame(), complex_frame()] {
            let s = StyleGenerator::suggest(&frame);
            assert!(
                s.meets_wcag_aa(),
                "suggestion must meet WCAG AA (4.5:1), got {:.2}",
                s.contrast_ratio
            );
        }
    }

    #[test]
    fn complex_frame_uses_solid_box() {
        let s = StyleGenerator::suggest(&complex_frame());
        assert!(
            matches!(s.background_treatment, BackgroundTreatment::SolidBox),
            "complex frame should use solid box, got {:?}",
            s.background_treatment
        );
    }

    #[test]
    fn complex_frame_enables_text_outline() {
        let s = StyleGenerator::suggest(&complex_frame());
        assert!(
            s.use_text_outline,
            "complex frame should enable text outline"
        );
    }

    #[test]
    fn font_size_scales_with_frame_height() {
        let small = StyleGenerator::suggest_font_size(480, 0.0);
        let large = StyleGenerator::suggest_font_size(2160, 0.0);
        assert!(
            large > small,
            "larger frame should produce larger font size"
        );
    }

    #[test]
    fn complex_frame_increases_font_size() {
        let simple = StyleGenerator::suggest_font_size(1080, 0.1);
        let complex = StyleGenerator::suggest_font_size(1080, 0.9);
        assert!(
            complex >= simple,
            "complex content should not shrink font size"
        );
    }

    #[test]
    fn reason_string_is_non_empty() {
        let s = StyleGenerator::suggest(&dark_frame());
        assert!(!s.reason.is_empty());
    }

    #[test]
    fn preferred_zone_falls_back_to_bottom_third() {
        let mut frame = dark_frame();
        frame.safe_zones.clear(); // no zones specified
        let s = StyleGenerator::suggest(&frame);
        assert_eq!(s.placement_zone, ContentZone::BottomThird);
    }

    #[test]
    fn centre_zone_is_avoided_when_alternatives_exist() {
        let mut frame = dark_frame();
        frame.safe_zones = vec![ContentZone::Centre, ContentZone::BottomThird];
        let s = StyleGenerator::suggest(&frame);
        // Should skip Centre and use BottomThird.
        assert_ne!(s.placement_zone, ContentZone::Centre);
    }

    #[test]
    fn font_size_clamped_to_minimum() {
        // Very small frame height should still produce at least 24px.
        let size = StyleGenerator::suggest_font_size(100, 0.0);
        assert!(size >= 24, "font size must be at least 24px, got {size}");
    }

    #[test]
    fn suggestion_contrast_ratio_at_least_one() {
        let s = StyleGenerator::suggest(&dark_frame());
        assert!(s.contrast_ratio >= 1.0, "contrast ratio must be ≥ 1.0");
    }

    #[test]
    fn meets_wcag_aaa_for_black_and_white_pair() {
        let s = CaptionStyleSuggestion {
            font_size_px: 40,
            text_rgb: [255, 255, 255],
            background_rgb: [0, 0, 0],
            background_treatment: BackgroundTreatment::SolidBox,
            placement_zone: ContentZone::BottomThird,
            contrast_ratio: 21.0,
            use_text_outline: false,
            reason: String::new(),
        };
        assert!(s.meets_wcag_aaa());
        assert!(s.meets_wcag_aa());
    }

    #[test]
    fn frame_analysis_is_bright_threshold() {
        let mut frame = dark_frame();
        frame.average_luminance = 0.5;
        assert!(!frame.is_bright(), "0.5 should not be bright");
        frame.average_luminance = 0.51;
        assert!(frame.is_bright(), "0.51 should be bright");
    }

    #[test]
    fn frame_analysis_is_complex_threshold() {
        let mut frame = dark_frame();
        frame.content_complexity = 0.6;
        assert!(!frame.is_complex(), "0.6 should not be complex");
        frame.content_complexity = 0.61;
        assert!(frame.is_complex(), "0.61 should be complex");
    }
}
