//! ICC rendering intent handling for color management.
//!
//! Implements the four ICC standard rendering intents:
//! - Perceptual: compresses gamut while preserving visual relationships
//! - Relative Colorimetric: clips out-of-gamut colors to nearest in-gamut
//! - Absolute Colorimetric: preserves absolute colorimetry (media white preserved)
//! - Saturation: maximises saturation at the expense of hue accuracy

#![allow(dead_code)]

/// The four ICC standard rendering intents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RenderingIntent {
    /// Perceptual rendering intent (ICC value 0).
    ///
    /// Compresses the source gamut into the destination gamut while
    /// maintaining relative colour relationships. Best for photographic images.
    Perceptual,

    /// Relative colorimetric intent (ICC value 1).
    ///
    /// Maps source white to destination white, then clips out-of-gamut colors
    /// to the nearest reproducible colour. Best for logos/spot colours.
    RelativeColorimetric,

    /// Saturation intent (ICC value 2).
    ///
    /// Maximises saturation of in-gamut colours, sacrificing hue and lightness
    /// accuracy. Best for business graphics / charts.
    Saturation,

    /// Absolute colorimetric intent (ICC value 3).
    ///
    /// Preserves absolute colorimetry including the media white point. Suitable
    /// for proofing applications where the substrate colour is important.
    AbsoluteColorimetric,
}

impl RenderingIntent {
    /// Returns the ICC numeric identifier for this rendering intent.
    #[must_use]
    pub fn icc_value(self) -> u32 {
        match self {
            Self::Perceptual => 0,
            Self::RelativeColorimetric => 1,
            Self::Saturation => 2,
            Self::AbsoluteColorimetric => 3,
        }
    }

    /// Construct a `RenderingIntent` from an ICC numeric value.
    ///
    /// Returns `None` for values outside the range 0-3.
    #[must_use]
    pub fn from_icc_value(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::Perceptual),
            1 => Some(Self::RelativeColorimetric),
            2 => Some(Self::Saturation),
            3 => Some(Self::AbsoluteColorimetric),
            _ => None,
        }
    }

    /// Human-readable name for this rendering intent.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Perceptual => "Perceptual",
            Self::RelativeColorimetric => "Relative Colorimetric",
            Self::Saturation => "Saturation",
            Self::AbsoluteColorimetric => "Absolute Colorimetric",
        }
    }

    /// Whether this intent uses the media white point of the source profile.
    #[must_use]
    pub fn uses_media_white_point(self) -> bool {
        matches!(self, Self::AbsoluteColorimetric)
    }
}

/// Result of applying a rendering intent to an out-of-gamut value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MappedColor {
    /// The resulting in-gamut colour.
    pub value: [f64; 3],
    /// Whether the colour was originally out of gamut.
    pub was_clipped: bool,
}

/// Apply relative colorimetric intent: clip to [min, max] per channel.
///
/// Performs a simple channel-wise clip with no hue preservation.
#[must_use]
pub fn relative_colorimetric_clip(color: [f64; 3], min: f64, max: f64) -> MappedColor {
    let clipped = [
        color[0].clamp(min, max),
        color[1].clamp(min, max),
        color[2].clamp(min, max),
    ];
    let was_clipped = clipped != color;
    MappedColor {
        value: clipped,
        was_clipped,
    }
}

/// Apply absolute colorimetric intent.
///
/// Adjusts for the source media white point relative to destination white before
/// clipping. `src_white` and `dst_white` are in XYZ (Y=1 normalised).
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn absolute_colorimetric(
    xyz: [f64; 3],
    src_white: [f64; 3],
    dst_white: [f64; 3],
) -> MappedColor {
    // Scale each channel by the white point ratio
    let scaled = [
        xyz[0] * (dst_white[0] / src_white[0]),
        xyz[1] * (dst_white[1] / src_white[1]),
        xyz[2] * (dst_white[2] / src_white[2]),
    ];
    let was_clipped = scaled[0] < 0.0
        || scaled[0] > 1.0
        || scaled[1] < 0.0
        || scaled[1] > 1.0
        || scaled[2] < 0.0
        || scaled[2] > 1.0;
    let clipped = [
        scaled[0].clamp(0.0, 1.0),
        scaled[1].clamp(0.0, 1.0),
        scaled[2].clamp(0.0, 1.0),
    ];
    MappedColor {
        value: clipped,
        was_clipped,
    }
}

/// Simple perceptual gamut compression using a knee function.
///
/// Values within `threshold` of the gamut boundary are softly compressed.
/// Values outside the gamut are mapped non-linearly to stay within [0, 1].
///
/// This is a simplified model; full perceptual intent requires an ICC profile's
/// perceptual intent table (B2A/A2B).
#[must_use]
pub fn perceptual_compress(value: f64, threshold: f64) -> f64 {
    debug_assert!((0.0..=1.0).contains(&threshold));
    if value <= threshold {
        value
    } else if value >= 1.0 {
        // Hard clip at max
        1.0
    } else {
        // Soft knee: compress [threshold, 1] -> [threshold, 1]
        let t = (value - threshold) / (1.0 - threshold);
        threshold + (1.0 - threshold) * t / (1.0 + t)
    }
}

/// Apply perceptual intent to an RGB colour triple.
#[must_use]
pub fn perceptual_map(color: [f64; 3], threshold: f64) -> MappedColor {
    let mapped = [
        perceptual_compress(color[0], threshold),
        perceptual_compress(color[1], threshold),
        perceptual_compress(color[2], threshold),
    ];
    let was_clipped = mapped[0] < color[0] || mapped[1] < color[1] || mapped[2] < color[2];
    MappedColor {
        value: mapped,
        was_clipped,
    }
}

/// Saturation-based mapping: maximise chroma while keeping hue angle.
///
/// Converts to a simple HSL-like representation, boosts saturation, then clips.
/// This is a coarse model of the saturation intent.
#[must_use]
pub fn saturation_map(rgb: [f64; 3], boost: f64) -> MappedColor {
    let r = rgb[0];
    let g = rgb[1];
    let b = rgb[2];
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let lightness = (max + min) / 2.0;
    let delta = max - min;

    if delta < 1e-10 {
        // Achromatic: no saturation to boost
        return MappedColor {
            value: rgb,
            was_clipped: false,
        };
    }

    // Increase saturation: move channels away from grey
    let grey = [lightness, lightness, lightness];
    let boosted = [
        grey[0] + (r - grey[0]) * boost,
        grey[1] + (g - grey[1]) * boost,
        grey[2] + (b - grey[2]) * boost,
    ];
    let clipped = [
        boosted[0].clamp(0.0, 1.0),
        boosted[1].clamp(0.0, 1.0),
        boosted[2].clamp(0.0, 1.0),
    ];
    let was_clipped = clipped != boosted;
    MappedColor {
        value: clipped,
        was_clipped,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_icc_values() {
        assert_eq!(RenderingIntent::Perceptual.icc_value(), 0);
        assert_eq!(RenderingIntent::RelativeColorimetric.icc_value(), 1);
        assert_eq!(RenderingIntent::Saturation.icc_value(), 2);
        assert_eq!(RenderingIntent::AbsoluteColorimetric.icc_value(), 3);
    }

    #[test]
    fn test_from_icc_value_round_trip() {
        for v in 0..=3 {
            let ri = RenderingIntent::from_icc_value(v).expect("ICC value parsing should succeed");
            assert_eq!(ri.icc_value(), v);
        }
    }

    #[test]
    fn test_from_icc_value_invalid() {
        assert!(RenderingIntent::from_icc_value(4).is_none());
        assert!(RenderingIntent::from_icc_value(255).is_none());
    }

    #[test]
    fn test_uses_media_white_point() {
        assert!(RenderingIntent::AbsoluteColorimetric.uses_media_white_point());
        assert!(!RenderingIntent::Perceptual.uses_media_white_point());
        assert!(!RenderingIntent::RelativeColorimetric.uses_media_white_point());
        assert!(!RenderingIntent::Saturation.uses_media_white_point());
    }

    #[test]
    fn test_rendering_intent_names_non_empty() {
        for v in 0..=3 {
            let ri = RenderingIntent::from_icc_value(v).expect("ICC value parsing should succeed");
            assert!(!ri.name().is_empty());
        }
    }

    #[test]
    fn test_relative_colorimetric_in_gamut() {
        let c = [0.5, 0.3, 0.7];
        let m = relative_colorimetric_clip(c, 0.0, 1.0);
        assert!(!m.was_clipped);
        assert!(approx_eq(m.value[0], 0.5, 1e-10));
    }

    #[test]
    fn test_relative_colorimetric_clips_over() {
        let c = [1.2, 0.5, -0.1];
        let m = relative_colorimetric_clip(c, 0.0, 1.0);
        assert!(m.was_clipped);
        assert!(approx_eq(m.value[0], 1.0, 1e-10));
        assert!(approx_eq(m.value[2], 0.0, 1e-10));
    }

    #[test]
    fn test_absolute_colorimetric_same_white() {
        let xyz = [0.5, 0.4, 0.3];
        let white = [0.95047, 1.0, 1.08883];
        let m = absolute_colorimetric(xyz, white, white);
        assert!(!m.was_clipped);
        assert!(approx_eq(m.value[0], 0.5, 1e-6));
    }

    #[test]
    fn test_perceptual_compress_below_threshold() {
        assert!(approx_eq(perceptual_compress(0.3, 0.8), 0.3, 1e-10));
    }

    #[test]
    fn test_perceptual_compress_at_one() {
        assert!(approx_eq(perceptual_compress(1.0, 0.8), 1.0, 1e-10));
    }

    #[test]
    fn test_perceptual_compress_above_threshold_less_than_input() {
        let compressed = perceptual_compress(0.95, 0.8);
        assert!(compressed < 0.95);
        assert!(compressed > 0.8);
    }

    #[test]
    fn test_perceptual_map_in_gamut() {
        let c = [0.2, 0.4, 0.6];
        let m = perceptual_map(c, 0.9);
        assert!(!m.was_clipped);
    }

    #[test]
    fn test_saturation_map_achromatic() {
        let grey = [0.5, 0.5, 0.5];
        let m = saturation_map(grey, 2.0);
        assert!(!m.was_clipped);
        assert!(approx_eq(m.value[0], 0.5, 1e-10));
    }

    #[test]
    fn test_saturation_map_boosts_chroma() {
        // A muted colour should become more saturated
        let muted = [0.5, 0.45, 0.4];
        let m = saturation_map(muted, 2.0);
        // R channel should increase, B channel should decrease
        assert!(m.value[0] >= muted[0]);
        assert!(m.value[2] <= muted[2]);
    }
}
