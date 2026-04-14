//! SDR-to-HDR inverse tone mapping (upconversion).
//!
//! Provides [`SdrToHdrConverter`] which lifts a linear SDR RGB triple into a
//! HDR linear light value by applying luminance boost, gamut saturation
//! enhancement, and optional highlight expansion.
//!
//! This is a simplified "boost + saturate" model suitable for real-time
//! upconversion where full perceptual inverse tone-mapping is not required.
//! For a reference-quality implementation see [`crate::tone_mapping_ext`].

#![allow(dead_code)]

// ── SdrToHdrConverter ─────────────────────────────────────────────────────────

/// Converts SDR linear RGB triples to HDR linear RGB.
///
/// Two parameters control the conversion:
/// - `boost`      — overall luminance scale factor (> 1.0 = brighter).
/// - `saturation` — colour saturation enhancement (1.0 = no change,
///   > 1.0 = more vivid, < 1.0 = desaturated).
///
/// # Example
///
/// ```rust
/// use oximedia_hdr::sdr_to_hdr::SdrToHdrConverter;
///
/// let converter = SdrToHdrConverter::new(4.0, 1.2);
/// let hdr = converter.convert([0.5, 0.3, 0.1]);
/// // HDR values are in linear light, typically > 1.0 for boosted input
/// assert!(hdr[0] > 0.5, "boost should increase luminance");
/// ```
#[derive(Debug, Clone)]
pub struct SdrToHdrConverter {
    /// Luminance boost factor (> 0.0).
    boost: f32,
    /// Saturation multiplier applied to the chroma components.
    saturation: f32,
}

impl SdrToHdrConverter {
    /// Create a new `SdrToHdrConverter`.
    ///
    /// * `boost`      — luminance multiplier; typical range 2.0–10.0.
    ///   Values ≤ 0 are clamped to 0.001.
    /// * `saturation` — saturation scale; 1.0 = unchanged, > 1.0 = boosted.
    ///   Values < 0 are clamped to 0.
    #[must_use]
    pub fn new(boost: f32, saturation: f32) -> Self {
        Self {
            boost: boost.max(0.001),
            saturation: saturation.max(0.0),
        }
    }

    /// Return the configured boost.
    #[must_use]
    pub fn boost(&self) -> f32 {
        self.boost
    }

    /// Return the configured saturation.
    #[must_use]
    pub fn saturation(&self) -> f32 {
        self.saturation
    }

    /// Convert a linear SDR RGB triple to HDR linear light.
    ///
    /// The conversion proceeds as:
    /// 1. Compute SDR luma Y using Rec. 709 coefficients.
    /// 2. Scale all channels by `boost`.
    /// 3. Enhance chroma distance from luma by `saturation`.
    ///
    /// The returned values are in linear light (may exceed 1.0).
    #[must_use]
    pub fn convert(&self, rgb: [f32; 3]) -> [f32; 3] {
        let [r, g, b] = rgb;

        // Step 1 — compute Rec. 709 luma
        let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;

        // Step 2 — boost all channels
        let rb = r * self.boost;
        let gb = g * self.boost;
        let bb = b * self.boost;
        let yb = y * self.boost;

        // Step 3 — saturation enhancement around the boosted luma
        if self.saturation == 1.0 {
            return [rb, gb, bb];
        }

        let r_out = yb + self.saturation * (rb - yb);
        let g_out = yb + self.saturation * (gb - yb);
        let b_out = yb + self.saturation * (bb - yb);

        [r_out.max(0.0), g_out.max(0.0), b_out.max(0.0)]
    }

    /// Convert a batch of RGB pixels (each a `[f32; 3]`) to HDR linear light.
    #[must_use]
    pub fn convert_batch(&self, pixels: &[[f32; 3]]) -> Vec<[f32; 3]> {
        pixels.iter().map(|&p| self.convert(p)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_and_accessors() {
        let c = SdrToHdrConverter::new(4.0, 1.2);
        assert!((c.boost() - 4.0).abs() < 1e-5);
        assert!((c.saturation() - 1.2).abs() < 1e-5);
    }

    #[test]
    fn test_new_clamps_boost_below_zero() {
        let c = SdrToHdrConverter::new(-1.0, 1.0);
        assert!(c.boost() > 0.0);
    }

    #[test]
    fn test_new_clamps_saturation_below_zero() {
        let c = SdrToHdrConverter::new(2.0, -0.5);
        assert_eq!(c.saturation(), 0.0);
    }

    #[test]
    fn test_convert_black_stays_black() {
        let c = SdrToHdrConverter::new(4.0, 1.5);
        let out = c.convert([0.0, 0.0, 0.0]);
        assert_eq!(out, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_convert_boost_increases_brightness() {
        let c = SdrToHdrConverter::new(4.0, 1.0); // saturation=1 = boost only
        let out = c.convert([0.5, 0.4, 0.3]);
        assert!((out[0] - 2.0).abs() < 1e-4, "R should be 0.5 * 4 = 2.0");
        assert!((out[1] - 1.6).abs() < 1e-4, "G should be 0.4 * 4 = 1.6");
        assert!((out[2] - 1.2).abs() < 1e-4, "B should be 0.3 * 4 = 1.2");
    }

    #[test]
    fn test_convert_saturation_one_equals_boost_only() {
        let c1 = SdrToHdrConverter::new(3.0, 1.0);
        let c2 = SdrToHdrConverter::new(3.0, 1.0);
        let a = c1.convert([0.3, 0.5, 0.2]);
        let b = c2.convert([0.3, 0.5, 0.2]);
        assert_eq!(a, b);
    }

    #[test]
    fn test_convert_grey_invariant_to_saturation() {
        // Pure grey (r==g==b): chroma=0, saturation has no effect
        let c1 = SdrToHdrConverter::new(2.0, 0.5);
        let c2 = SdrToHdrConverter::new(2.0, 2.0);
        let grey = [0.5, 0.5, 0.5];
        let a = c1.convert(grey);
        let b = c2.convert(grey);
        // For grey, luma == r == g == b, so (x - luma) = 0 for all channels
        // Both should give the same result
        for i in 0..3 {
            assert!(
                (a[i] - b[i]).abs() < 1e-4,
                "grey should be saturation-invariant channel {i}: a={}, b={}",
                a[i],
                b[i]
            );
        }
    }

    #[test]
    fn test_convert_output_non_negative() {
        let c = SdrToHdrConverter::new(8.0, 0.0);
        let out = c.convert([0.1, 0.2, 0.3]);
        for v in out {
            assert!(v >= 0.0, "output must not be negative");
        }
    }

    #[test]
    fn test_convert_batch_length() {
        let c = SdrToHdrConverter::new(2.0, 1.0);
        let pixels = vec![[0.1f32, 0.2, 0.3]; 100];
        let out = c.convert_batch(&pixels);
        assert_eq!(out.len(), 100);
    }

    #[test]
    fn test_convert_saturation_zero_produces_grey() {
        let c = SdrToHdrConverter::new(1.0, 0.0);
        let [r, g, b] = c.convert([0.8, 0.4, 0.2]);
        // sat=0 → all channels equal boosted luma
        let y = 0.2126 * 0.8 + 0.7152 * 0.4 + 0.0722 * 0.2;
        assert!((r - y).abs() < 1e-4, "r should equal luma: r={r}, y={y}");
        assert!((g - y).abs() < 1e-4, "g should equal luma: g={g}, y={y}");
        assert!((b - y).abs() < 1e-4, "b should equal luma: b={b}, y={y}");
    }
}
