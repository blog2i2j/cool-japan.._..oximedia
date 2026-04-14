//! Display P3 colour gamut definitions and conversions.
//!
//! Display P3 is a wide colour gamut defined by the DCI-P3 primaries combined
//! with sRGB transfer function and D65 white point.  It is the standard
//! wide-colour-gamut for Apple displays, consumer HDR TVs, and the web CSS
//! `color(display-p3 …)` colour space.
//!
//! # Primaries (CIE 1931 xy)
//!
//! | Primary | x | y |
//! |---------|--------|--------|
//! | Red | 0.680 | 0.320 |
//! | Green | 0.265 | 0.690 |
//! | Blue | 0.150 | 0.060 |
//!
//! White point: D65 (x=0.3127, y=0.3290)

// ── Primaries ─────────────────────────────────────────────────────────────────

/// CIE 1931 xy chromaticity coordinates of the Display P3 primaries.
///
/// Layout: `[[rx, ry], [gx, gy], [bx, by]]`
pub const DISPLAY_P3_PRIMARIES: [[f32; 2]; 3] = [
    [0.680, 0.320], // Red
    [0.265, 0.690], // Green
    [0.150, 0.060], // Blue
];

/// D65 white point xy chromaticity used by Display P3.
pub const DISPLAY_P3_WHITE_POINT: [f32; 2] = [0.3127, 0.3290];

// ── Conversion matrices (linear light) ───────────────────────────────────────

/// Linear sRGB → Linear Display P3, D65 white point, row-major.
///
/// Derived from the ICC sRGB and Display P3 primary matrices.
const SRGB_TO_DISPLAY_P3_MATRIX: [[f32; 3]; 3] = [
    [0.8224_6209, 0.1775_3791, 0.0],
    [0.0331_9419, 0.9668_0581, 0.0],
    [0.0170_8275, 0.0723_9744, 0.9105_5981],
];

// ── Public API ────────────────────────────────────────────────────────────────

/// A zero-size type representing the Display P3 colour space.
pub struct DisplayP3;

impl DisplayP3 {
    /// Return the CIE 1931 xy chromaticity primaries.
    ///
    /// Layout: `[[rx, ry], [gx, gy], [bx, by]]`
    #[must_use]
    pub fn primaries() -> [[f32; 2]; 3] {
        DISPLAY_P3_PRIMARIES
    }

    /// Return the D65 white point xy chromaticity coordinates.
    #[must_use]
    pub fn white_point() -> [f32; 2] {
        DISPLAY_P3_WHITE_POINT
    }

    /// Convert a **linear-light** sRGB triplet to **linear-light** Display P3.
    ///
    /// Both the input and output are scene-linear values (i.e. the sRGB
    /// gamma / OETF must be removed before calling this function, and the
    /// Display P3 OETF applied after if an encoded value is required).
    #[must_use]
    pub fn linear_srgb_to_linear_display_p3(rgb: [f32; 3]) -> [f32; 3] {
        let m = &SRGB_TO_DISPLAY_P3_MATRIX;
        let [r, g, b] = rgb;
        [
            m[0][0] * r + m[0][1] * g + m[0][2] * b,
            m[1][0] * r + m[1][1] * g + m[1][2] * b,
            m[2][0] * r + m[2][1] * g + m[2][2] * b,
        ]
    }
}

/// Convert an **encoded** sRGB triplet to an **encoded** Display P3 triplet.
///
/// This function linearises the sRGB input, applies the Display P3 matrix, and
/// then re-encodes with the sRGB transfer curve (which is also used for Display
/// P3 as specified by Apple's Color Space documentation).
///
/// Values outside `[0.0, 1.0]` are clamped after the matrix transform to keep
/// the output in the nominal Display P3 gamut.
#[must_use]
pub fn srgb_to_display_p3(rgb: [f32; 3]) -> [f32; 3] {
    let linear = [
        srgb_eotf(rgb[0]),
        srgb_eotf(rgb[1]),
        srgb_eotf(rgb[2]),
    ];
    let p3_linear = DisplayP3::linear_srgb_to_linear_display_p3(linear);
    // Clamp to Display P3 nominal range before encoding
    let clamped = [
        p3_linear[0].clamp(0.0, 1.0),
        p3_linear[1].clamp(0.0, 1.0),
        p3_linear[2].clamp(0.0, 1.0),
    ];
    [
        srgb_oetf(clamped[0]),
        srgb_oetf(clamped[1]),
        srgb_oetf(clamped[2]),
    ]
}

// ── Transfer functions ────────────────────────────────────────────────────────

/// sRGB EOTF (inverse OETF): encoded → linear.
#[inline]
fn srgb_eotf(v: f32) -> f32 {
    if v <= 0.040_448_237 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// sRGB OETF: linear → encoded.
#[inline]
fn srgb_oetf(v: f32) -> f32 {
    if v <= 0.003_130_8 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_primaries_shape() {
        let p = DisplayP3::primaries();
        assert_eq!(p.len(), 3, "three primaries");
        // Each primary has two coordinates
        for coord in &p {
            assert_eq!(coord.len(), 2);
        }
    }

    #[test]
    fn test_white_point() {
        let wp = DisplayP3::white_point();
        assert!(approx_eq(wp[0], 0.3127, 1e-4));
        assert!(approx_eq(wp[1], 0.3290, 1e-4));
    }

    #[test]
    fn test_srgb_white_stays_white() {
        // sRGB white [1, 1, 1] should map to Display P3 white [1, 1, 1].
        let out = srgb_to_display_p3([1.0, 1.0, 1.0]);
        assert!(approx_eq(out[0], 1.0, 1e-3), "R: {}", out[0]);
        assert!(approx_eq(out[1], 1.0, 1e-3), "G: {}", out[1]);
        assert!(approx_eq(out[2], 1.0, 1e-3), "B: {}", out[2]);
    }

    #[test]
    fn test_srgb_black_stays_black() {
        let out = srgb_to_display_p3([0.0, 0.0, 0.0]);
        assert!(approx_eq(out[0], 0.0, 1e-6));
        assert!(approx_eq(out[1], 0.0, 1e-6));
        assert!(approx_eq(out[2], 0.0, 1e-6));
    }

    #[test]
    fn test_output_in_range() {
        // Any encoded sRGB triplet should produce an encoded Display P3 triplet
        // in [0, 1].
        let test_values = [
            [0.5_f32, 0.3, 0.2],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.2, 0.4, 0.8],
        ];
        for rgb in &test_values {
            let out = srgb_to_display_p3(*rgb);
            for (i, &v) in out.iter().enumerate() {
                assert!(
                    (0.0..=1.0).contains(&v),
                    "channel {i} = {v} out of range for input {rgb:?}"
                );
            }
        }
    }

    #[test]
    fn test_linear_identity_primaries() {
        // Linear sRGB [1, 0, 0] should lie close to (but not equal to) [1, 0, 0]
        // in Display P3 because the primaries differ.
        let out = DisplayP3::linear_srgb_to_linear_display_p3([1.0, 0.0, 0.0]);
        // sRGB red has some green/blue spill in Display P3
        assert!(out[0] > 0.8, "red primary should dominate: {}", out[0]);
    }
}
