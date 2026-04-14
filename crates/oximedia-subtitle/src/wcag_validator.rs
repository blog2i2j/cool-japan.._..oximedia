//! WCAG 2.1 AA/AAA subtitle contrast checker.
//!
//! Validates foreground/background color combinations against WCAG 2.0/2.1
//! contrast ratio thresholds. AA requires 4.5:1 for normal text, AAA requires 7:1.
//! Provides computed ratios, pass/fail results, and suggested color corrections.
//!
//! # Example
//!
//! ```
//! use oximedia_subtitle::wcag_validator::{WcagValidator, WcagLevel};
//!
//! let validator = WcagValidator::new();
//! let result = validator.check_contrast(
//!     (255, 255, 255), // white foreground
//!     (0, 0, 0),       // black background
//!     WcagLevel::Aa,
//! );
//! assert!(result.passes);
//! ```

/// WCAG conformance level.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WcagLevel {
    /// AA level — 4.5:1 for normal text, 3:1 for large text.
    Aa,
    /// AAA level — 7:1 for normal text, 4.5:1 for large text.
    Aaa,
}

impl WcagLevel {
    /// Minimum contrast ratio for normal text at this level.
    #[must_use]
    pub const fn min_ratio(self) -> f64 {
        match self {
            Self::Aa => 4.5,
            Self::Aaa => 7.0,
        }
    }

    /// Minimum contrast ratio for large text (18pt+ or 14pt bold+) at this level.
    #[must_use]
    pub const fn min_ratio_large_text(self) -> f64 {
        match self {
            Self::Aa => 3.0,
            Self::Aaa => 4.5,
        }
    }
}

/// Result of a WCAG contrast check.
#[derive(Clone, Debug)]
pub struct ContrastResult {
    /// Computed contrast ratio (always >= 1.0).
    pub ratio: f64,
    /// Whether this passes the requested WCAG level for normal text.
    pub passes: bool,
    /// Whether this passes for large text (lower threshold).
    pub passes_large_text: bool,
    /// The WCAG level checked.
    pub level: WcagLevel,
    /// Foreground color (r, g, b).
    pub foreground: (u8, u8, u8),
    /// Background color (r, g, b).
    pub background: (u8, u8, u8),
    /// Suggested corrected foreground color if the check fails.
    pub suggested_foreground: Option<(u8, u8, u8)>,
    /// Suggested corrected background color if the check fails.
    pub suggested_background: Option<(u8, u8, u8)>,
}

/// WCAG 2.1 contrast validator.
pub struct WcagValidator {
    /// Maximum iterations for color correction search.
    max_correction_iterations: u32,
}

impl Default for WcagValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl WcagValidator {
    /// Create a new validator with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            max_correction_iterations: 256,
        }
    }

    /// Set max iterations for color correction search.
    #[must_use]
    pub fn with_max_iterations(mut self, max: u32) -> Self {
        self.max_correction_iterations = max;
        self
    }

    /// Calculate relative luminance of an sRGB color per WCAG 2.0 formula.
    ///
    /// See: <https://www.w3.org/TR/WCAG20/#relativeluminancedef>
    #[must_use]
    pub fn relative_luminance(r: u8, g: u8, b: u8) -> f64 {
        let rs = Self::linearize(r);
        let gs = Self::linearize(g);
        let bs = Self::linearize(b);
        0.2126 * rs + 0.7152 * gs + 0.0722 * bs
    }

    /// Linearize a single sRGB channel value (0-255) to linear light.
    fn linearize(value: u8) -> f64 {
        let v = f64::from(value) / 255.0;
        if v <= 0.04045 {
            v / 12.92
        } else {
            ((v + 0.055) / 1.055).powf(2.4)
        }
    }

    /// Calculate contrast ratio between two colors.
    ///
    /// Returns a value >= 1.0 where 21:1 is the maximum (black vs white).
    #[must_use]
    pub fn contrast_ratio(fg: (u8, u8, u8), bg: (u8, u8, u8)) -> f64 {
        let l1 = Self::relative_luminance(fg.0, fg.1, fg.2);
        let l2 = Self::relative_luminance(bg.0, bg.1, bg.2);
        let lighter = l1.max(l2);
        let darker = l1.min(l2);
        (lighter + 0.05) / (darker + 0.05)
    }

    /// Check contrast between foreground and background at the given WCAG level.
    #[must_use]
    pub fn check_contrast(
        &self,
        fg: (u8, u8, u8),
        bg: (u8, u8, u8),
        level: WcagLevel,
    ) -> ContrastResult {
        let ratio = Self::contrast_ratio(fg, bg);
        let min_normal = level.min_ratio();
        let min_large = level.min_ratio_large_text();
        let passes = ratio >= min_normal;
        let passes_large = ratio >= min_large;

        let (suggested_fg, suggested_bg) = if passes {
            (None, None)
        } else {
            (
                self.suggest_foreground(fg, bg, level),
                self.suggest_background(fg, bg, level),
            )
        };

        ContrastResult {
            ratio,
            passes,
            passes_large_text: passes_large,
            level,
            foreground: fg,
            background: bg,
            suggested_foreground: suggested_fg,
            suggested_background: suggested_bg,
        }
    }

    /// Check multiple color pairs at once.
    #[must_use]
    pub fn check_batch(
        &self,
        pairs: &[((u8, u8, u8), (u8, u8, u8))],
        level: WcagLevel,
    ) -> Vec<ContrastResult> {
        pairs
            .iter()
            .map(|&(fg, bg)| self.check_contrast(fg, bg, level))
            .collect()
    }

    /// Suggest a corrected foreground color that passes the given level.
    ///
    /// Adjusts luminance while preserving hue/saturation direction.
    fn suggest_foreground(
        &self,
        fg: (u8, u8, u8),
        bg: (u8, u8, u8),
        level: WcagLevel,
    ) -> Option<(u8, u8, u8)> {
        let bg_lum = Self::relative_luminance(bg.0, bg.1, bg.2);
        let fg_lum = Self::relative_luminance(fg.0, fg.1, fg.2);
        let target_ratio = level.min_ratio();

        // Decide direction: darken or lighten foreground
        let go_darker = fg_lum > bg_lum;

        let mut best = fg;
        for i in 0..self.max_correction_iterations {
            let t = (i as f64) / (self.max_correction_iterations as f64);
            let candidate = if go_darker {
                Self::lerp_color(fg, (0, 0, 0), t)
            } else {
                Self::lerp_color(fg, (255, 255, 255), t)
            };
            let ratio = Self::contrast_ratio(candidate, bg);
            if ratio >= target_ratio {
                best = candidate;
                break;
            }
            best = candidate;
        }

        // Only suggest if it actually passes now
        if Self::contrast_ratio(best, bg) >= target_ratio {
            Some(best)
        } else {
            // Extreme fallback: pure black or white
            if Self::contrast_ratio((0, 0, 0), bg) >= target_ratio {
                Some((0, 0, 0))
            } else if Self::contrast_ratio((255, 255, 255), bg) >= target_ratio {
                Some((255, 255, 255))
            } else {
                None
            }
        }
    }

    /// Suggest a corrected background color.
    fn suggest_background(
        &self,
        fg: (u8, u8, u8),
        bg: (u8, u8, u8),
        level: WcagLevel,
    ) -> Option<(u8, u8, u8)> {
        let bg_lum = Self::relative_luminance(bg.0, bg.1, bg.2);
        let fg_lum = Self::relative_luminance(fg.0, fg.1, fg.2);
        let target_ratio = level.min_ratio();

        // Opposite direction of foreground adjustment
        let go_darker = bg_lum > fg_lum;

        let mut best = bg;
        for i in 0..self.max_correction_iterations {
            let t = (i as f64) / (self.max_correction_iterations as f64);
            let candidate = if go_darker {
                Self::lerp_color(bg, (0, 0, 0), t)
            } else {
                Self::lerp_color(bg, (255, 255, 255), t)
            };
            let ratio = Self::contrast_ratio(fg, candidate);
            if ratio >= target_ratio {
                best = candidate;
                break;
            }
            best = candidate;
        }

        if Self::contrast_ratio(fg, best) >= target_ratio {
            Some(best)
        } else {
            if Self::contrast_ratio(fg, (0, 0, 0)) >= target_ratio {
                Some((0, 0, 0))
            } else if Self::contrast_ratio(fg, (255, 255, 255)) >= target_ratio {
                Some((255, 255, 255))
            } else {
                None
            }
        }
    }

    /// Linear interpolation between two colors.
    fn lerp_color(a: (u8, u8, u8), b: (u8, u8, u8), t: f64) -> (u8, u8, u8) {
        let t = t.clamp(0.0, 1.0);
        let r = (f64::from(a.0) * (1.0 - t) + f64::from(b.0) * t).round() as u8;
        let g = (f64::from(a.1) * (1.0 - t) + f64::from(b.1) * t).round() as u8;
        let b_val = (f64::from(a.2) * (1.0 - t) + f64::from(b.2) * t).round() as u8;
        (r, g, b_val)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relative_luminance_black() {
        let lum = WcagValidator::relative_luminance(0, 0, 0);
        assert!((lum - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_relative_luminance_white() {
        let lum = WcagValidator::relative_luminance(255, 255, 255);
        assert!((lum - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_contrast_ratio_black_white() {
        let ratio = WcagValidator::contrast_ratio((255, 255, 255), (0, 0, 0));
        // Should be 21:1
        assert!((ratio - 21.0).abs() < 0.1);
    }

    #[test]
    fn test_contrast_ratio_same_color() {
        let ratio = WcagValidator::contrast_ratio((128, 128, 128), (128, 128, 128));
        assert!((ratio - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_aa_pass_white_on_black() {
        let validator = WcagValidator::new();
        let result = validator.check_contrast((255, 255, 255), (0, 0, 0), WcagLevel::Aa);
        assert!(result.passes);
        assert!(result.passes_large_text);
        assert!(result.ratio >= 4.5);
    }

    #[test]
    fn test_aaa_pass_white_on_black() {
        let validator = WcagValidator::new();
        let result = validator.check_contrast((255, 255, 255), (0, 0, 0), WcagLevel::Aaa);
        assert!(result.passes);
        assert!(result.ratio >= 7.0);
    }

    #[test]
    fn test_aa_fail_low_contrast() {
        let validator = WcagValidator::new();
        // Light gray on white — poor contrast
        let result = validator.check_contrast((200, 200, 200), (255, 255, 255), WcagLevel::Aa);
        assert!(!result.passes);
        assert!(result.ratio < 4.5);
    }

    #[test]
    fn test_suggested_correction_on_failure() {
        let validator = WcagValidator::new();
        // Light gray on white — should suggest darker foreground
        let result = validator.check_contrast((200, 200, 200), (255, 255, 255), WcagLevel::Aa);
        assert!(!result.passes);
        assert!(result.suggested_foreground.is_some());

        // Verify suggested color actually passes
        let suggested = result.suggested_foreground.expect("test");
        let new_ratio = WcagValidator::contrast_ratio(suggested, (255, 255, 255));
        assert!(new_ratio >= 4.5);
    }

    #[test]
    fn test_contrast_ratio_order_independent() {
        let r1 = WcagValidator::contrast_ratio((100, 50, 200), (200, 220, 30));
        let r2 = WcagValidator::contrast_ratio((200, 220, 30), (100, 50, 200));
        assert!((r1 - r2).abs() < 1e-10);
    }

    #[test]
    fn test_batch_check() {
        let validator = WcagValidator::new();
        let pairs = vec![
            ((255, 255, 255), (0, 0, 0)),
            ((200, 200, 200), (255, 255, 255)),
        ];
        let results = validator.check_batch(&pairs, WcagLevel::Aa);
        assert_eq!(results.len(), 2);
        assert!(results[0].passes);
        assert!(!results[1].passes);
    }

    #[test]
    fn test_large_text_threshold() {
        let validator = WcagValidator::new();
        // Medium contrast that might pass large text but not normal
        // Ratio should be between 3.0 and 4.5
        let result = validator.check_contrast((119, 119, 119), (255, 255, 255), WcagLevel::Aa);
        // ~3.6:1 ratio
        assert!(!result.passes); // fails normal text
        assert!(result.passes_large_text); // passes large text at AA
    }

    #[test]
    fn test_linearize_low_values() {
        // Values below threshold use linear formula
        let lin = WcagValidator::relative_luminance(10, 10, 10);
        assert!(lin > 0.0);
        assert!(lin < 0.01);
    }

    #[test]
    fn test_wcag_level_thresholds() {
        assert!((WcagLevel::Aa.min_ratio() - 4.5).abs() < f64::EPSILON);
        assert!((WcagLevel::Aaa.min_ratio() - 7.0).abs() < f64::EPSILON);
        assert!((WcagLevel::Aa.min_ratio_large_text() - 3.0).abs() < f64::EPSILON);
        assert!((WcagLevel::Aaa.min_ratio_large_text() - 4.5).abs() < f64::EPSILON);
    }
}
