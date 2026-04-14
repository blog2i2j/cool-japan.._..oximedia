//! Color grading LUT suggestion based on dominant hue and saturation.
//!
//! [`ColorGradingSuggester`] maps scene color characteristics to a named
//! color-grading Look-Up Table (LUT) preset. The suggestions are heuristic
//! and intended as a starting point for a colorist rather than a final grade.
//!
//! # Hue wheel partitioning
//!
//! Hue is expressed in degrees `[0, 360)` following the standard HSL/HSV
//! color wheel:
//!
//! | Hue range   | Color family   |
//! |-------------|----------------|
//! | 0 – 30      | Red / warm     |
//! | 30 – 75     | Orange / amber |
//! | 75 – 160    | Yellow / green |
//! | 160 – 260   | Green / cyan   |
//! | 260 – 300   | Blue           |
//! | 300 – 360   | Magenta / red  |
//!
//! Low saturation (< 0.15) triggers a desaturation / monochrome suggestion
//! regardless of hue.
//!
//! # Example
//!
//! ```rust
//! use oximedia_auto::color_suggest::ColorGradingSuggester;
//!
//! // Warm sunset scene (medium saturation so hue-based selection applies).
//! let lut = ColorGradingSuggester::suggest_lut(25.0, 0.6);
//! assert_eq!(lut, "warm_golden");
//!
//! // Desaturated / monochrome scene.
//! let lut2 = ColorGradingSuggester::suggest_lut(120.0, 0.05);
//! assert_eq!(lut2, "desaturated_mono");
//! ```

#![allow(dead_code)]

/// Threshold below which the scene is considered desaturated / monochromatic.
const LOW_SATURATION_THRESHOLD: f32 = 0.15;
/// Threshold above which the scene is considered highly saturated (vivid).
const HIGH_SATURATION_THRESHOLD: f32 = 0.75;

/// Color grading LUT suggestion service.
///
/// All methods are pure functions with no persistent state.
pub struct ColorGradingSuggester;

impl ColorGradingSuggester {
    /// Suggest a named LUT preset given the `dominant_hue` (degrees, 0–360)
    /// and the overall `saturation` level (0.0–1.0).
    ///
    /// # Returns
    ///
    /// A `'static str` naming a LUT preset.  The following presets are defined:
    ///
    /// | Preset name              | Typical use                                |
    /// |--------------------------|--------------------------------------------|
    /// | `"desaturated_mono"`     | Neutral / monochrome, low saturation       |
    /// | `"warm_golden"`          | Warm reds / oranges, sunset, skin tones    |
    /// | `"amber_filmic"`         | Amber / orange-tinted filmic look          |
    /// | `"natural_green"`        | Natural green landscapes, forests          |
    /// | `"cool_teal"`            | Teal-green, ocean, cool daytime            |
    /// | `"cinematic_blue"`       | Blue hour, night scenes                    |
    /// | `"magenta_dream"`        | Purple / pink / magenta, fantasy           |
    /// | `"vivid_boost"`          | High-saturation scene, any hue             |
    /// | `"neutral_balanced"`     | Neutral, balanced saturation               |
    #[must_use]
    pub fn suggest_lut(dominant_hue: f32, saturation: f32) -> &'static str {
        let hue = dominant_hue.rem_euclid(360.0); // normalize to [0, 360)
        let sat = saturation.clamp(0.0, 1.0);

        // Low saturation → monochrome / desaturated regardless of hue.
        if sat < LOW_SATURATION_THRESHOLD {
            return "desaturated_mono";
        }

        // High saturation → vivid boost overrides hue-based suggestion.
        if sat >= HIGH_SATURATION_THRESHOLD {
            return "vivid_boost";
        }

        // Hue-based suggestion for medium saturation.
        match hue {
            h if h < 30.0 => "warm_golden",
            h if h < 75.0 => "amber_filmic",
            h if h < 160.0 => "natural_green",
            h if h < 260.0 => "cool_teal",
            h if h < 300.0 => "cinematic_blue",
            _ => "magenta_dream",
        }
    }

    /// Suggest a LUT and return a structured [`LutSuggestion`] with reasoning.
    #[must_use]
    pub fn suggest(dominant_hue: f32, saturation: f32) -> LutSuggestion {
        let name = Self::suggest_lut(dominant_hue, saturation);
        let description = Self::describe(name);
        LutSuggestion {
            name,
            description,
            dominant_hue,
            saturation,
        }
    }

    /// Return a human-readable description of a named LUT preset.
    #[must_use]
    pub fn describe(lut_name: &str) -> &'static str {
        match lut_name {
            "desaturated_mono" => {
                "Desaturated monochrome look; removes colour cast for neutral B&W feel."
            }
            "warm_golden" => "Golden warm grade; lifts shadows into amber for a sunset feel.",
            "amber_filmic" => "Orange-amber filmic tint reminiscent of vintage celluloid stock.",
            "natural_green" => {
                "Earthy natural green; boosts mid greens for foliage and landscape scenes."
            }
            "cool_teal" => "Teal-cyan grade; pushes shadows into teal for a modern cinematic look.",
            "cinematic_blue" => "Dramatic blue-hour grade; suitable for night or twilight scenes.",
            "magenta_dream" => "Soft magenta-pink grade; adds a dreamy, fantastical quality.",
            "vivid_boost" => {
                "Vivid saturation boost; emphasises colour for travel or nature footage."
            }
            _ => "Neutral balanced grade; minimal colour shift for general use.",
        }
    }
}

// ---------------------------------------------------------------------------
// LutSuggestion
// ---------------------------------------------------------------------------

/// A structured color-grading LUT suggestion with metadata.
#[derive(Debug, Clone)]
pub struct LutSuggestion {
    /// LUT preset name (e.g. `"warm_golden"`).
    pub name: &'static str,
    /// Human-readable description of the LUT.
    pub description: &'static str,
    /// Dominant hue supplied to the suggester (degrees).
    pub dominant_hue: f32,
    /// Saturation level supplied to the suggester.
    pub saturation: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── desaturated ───────────────────────────────────────────────────────────

    #[test]
    fn test_low_saturation_returns_mono() {
        assert_eq!(
            ColorGradingSuggester::suggest_lut(180.0, 0.0),
            "desaturated_mono"
        );
        assert_eq!(
            ColorGradingSuggester::suggest_lut(45.0, 0.10),
            "desaturated_mono"
        );
    }

    // ── vivid boost ───────────────────────────────────────────────────────────

    #[test]
    fn test_high_saturation_returns_vivid() {
        assert_eq!(
            ColorGradingSuggester::suggest_lut(120.0, 0.8),
            "vivid_boost"
        );
        assert_eq!(
            ColorGradingSuggester::suggest_lut(270.0, 1.0),
            "vivid_boost"
        );
    }

    // ── hue-specific ─────────────────────────────────────────────────────────

    #[test]
    fn test_warm_golden_for_red_hue() {
        assert_eq!(ColorGradingSuggester::suggest_lut(10.0, 0.5), "warm_golden");
        assert_eq!(ColorGradingSuggester::suggest_lut(25.0, 0.5), "warm_golden");
    }

    #[test]
    fn test_amber_filmic_for_orange_hue() {
        assert_eq!(
            ColorGradingSuggester::suggest_lut(50.0, 0.5),
            "amber_filmic"
        );
        assert_eq!(
            ColorGradingSuggester::suggest_lut(70.0, 0.5),
            "amber_filmic"
        );
    }

    #[test]
    fn test_natural_green_for_green_hue() {
        assert_eq!(
            ColorGradingSuggester::suggest_lut(100.0, 0.4),
            "natural_green"
        );
        assert_eq!(
            ColorGradingSuggester::suggest_lut(155.0, 0.4),
            "natural_green"
        );
    }

    #[test]
    fn test_cool_teal_for_teal_hue() {
        assert_eq!(ColorGradingSuggester::suggest_lut(180.0, 0.5), "cool_teal");
        assert_eq!(ColorGradingSuggester::suggest_lut(250.0, 0.5), "cool_teal");
    }

    #[test]
    fn test_cinematic_blue_for_blue_hue() {
        assert_eq!(
            ColorGradingSuggester::suggest_lut(270.0, 0.5),
            "cinematic_blue"
        );
        assert_eq!(
            ColorGradingSuggester::suggest_lut(290.0, 0.5),
            "cinematic_blue"
        );
    }

    #[test]
    fn test_magenta_dream_for_magenta() {
        assert_eq!(
            ColorGradingSuggester::suggest_lut(310.0, 0.5),
            "magenta_dream"
        );
        assert_eq!(
            ColorGradingSuggester::suggest_lut(350.0, 0.5),
            "magenta_dream"
        );
    }

    #[test]
    fn test_hue_wraps_360() {
        // Hue 360 should wrap to 0 → red → "warm_golden".
        assert_eq!(
            ColorGradingSuggester::suggest_lut(360.0, 0.5),
            ColorGradingSuggester::suggest_lut(0.0, 0.5)
        );
    }

    #[test]
    fn test_suggest_returns_correct_name() {
        let s = ColorGradingSuggester::suggest(25.0, 0.8);
        // saturation 0.8 ≥ 0.75 → vivid_boost
        assert_eq!(s.name, "vivid_boost");
    }

    #[test]
    fn test_describe_all_presets() {
        for preset in &[
            "desaturated_mono",
            "warm_golden",
            "amber_filmic",
            "natural_green",
            "cool_teal",
            "cinematic_blue",
            "magenta_dream",
            "vivid_boost",
        ] {
            let desc = ColorGradingSuggester::describe(preset);
            assert!(
                !desc.is_empty(),
                "Description should not be empty for {preset}"
            );
        }
    }

    #[test]
    fn test_saturation_clamped() {
        // saturation > 1.0 should be treated as 1.0 (vivid).
        assert_eq!(
            ColorGradingSuggester::suggest_lut(180.0, 2.0),
            "vivid_boost"
        );
        // saturation < 0.0 should be treated as 0.0 (mono).
        assert_eq!(
            ColorGradingSuggester::suggest_lut(180.0, -1.0),
            "desaturated_mono"
        );
    }
}
