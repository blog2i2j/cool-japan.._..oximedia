//! ICC (International Color Consortium) profile to 3-D LUT conversion.
//!
//! This module provides utilities for converting ICC colour profiles into 3-D
//! LUTs that can be applied without an ICC-aware runtime.  The focus is on the
//! most common profile classes (display, input, output) and the most widely
//! used tag types (TRC curves, matrix, and `A2Bx`/`B2Ax` multi-process element
//! tables).
//!
//! # Overview
//!
//! An ICC profile describes a colour transform from a device colour space to
//! the ICC Profile Connection Space (PCS — either CIE XYZ D50 or CIELAB).
//! Converting a profile to a 3-D LUT proceeds as follows:
//!
//! 1. Parse the ICC header to determine profile class and colour space.
//! 2. Extract the forward transform (device → PCS).
//! 3. Compose with the PCS → display colour space transform.
//! 4. Sample the composed transform on a regular grid to produce the LUT.
//!
//! # Supported Tag Types
//!
//! | Tag | Description |
//! |-----|-------------|
//! | `rTRC`/`gTRC`/`bTRC` + `rXYZ`/`gXYZ`/`bXYZ` | Matrix/TRC profile (most display profiles) |
//! | Parametric curve `(0)`..`(4)` | Power-law and sRGB-like curves |
//!
//! # Example
//!
//! ```rust
//! use oximedia_lut::icc_profile::{IccTrc, IccMatrixProfile, IccToLutConverter};
//!
//! // Build a synthetic sRGB-like display profile
//! let profile = IccMatrixProfile {
//!     description: "sRGB IEC61966-2-1".to_string(),
//!     // D65 adapted to D50 via Bradford (standard for ICC)
//!     matrix_to_xyz_d50: [
//!         [0.436_065, 0.385_151, 0.143_081],
//!         [0.222_491, 0.716_888, 0.060_621],
//!         [0.013_920, 0.097_045, 0.714_136],
//!     ],
//!     trc_r: IccTrc::Parametric { gamma: 2.4, a: 1.0 / 1.055, b: 0.055 / 1.055, c: 1.0 / 12.92, d: 0.04045, e: 0.0, f: 0.0 },
//!     trc_g: IccTrc::Parametric { gamma: 2.4, a: 1.0 / 1.055, b: 0.055 / 1.055, c: 1.0 / 12.92, d: 0.04045, e: 0.0, f: 0.0 },
//!     trc_b: IccTrc::Parametric { gamma: 2.4, a: 1.0 / 1.055, b: 0.055 / 1.055, c: 1.0 / 12.92, d: 0.04045, e: 0.0, f: 0.0 },
//! };
//!
//! let converter = IccToLutConverter::new(profile);
//! let lut = converter.to_lut3d(17);
//! assert_eq!(lut.len(), 17 * 17 * 17);
//! ```

use crate::error::{LutError, LutResult};
use crate::Rgb;

// ============================================================================
// Tone Reproduction Curve types
// ============================================================================

/// A tone reproduction curve (TRC) as found in ICC profiles.
#[derive(Debug, Clone)]
pub enum IccTrc {
    /// Simple power-law gamma curve: `y = x ^ gamma`.
    Gamma(f64),

    /// ICC parametric curve type 3 (IEC 61966-2-1 / sRGB):
    ///
    /// ```text
    /// if x >= d { y = (a*x + b)^gamma + e }
    /// else       { y = c*x + f }
    /// ```
    Parametric {
        /// Exponent.
        gamma: f64,
        /// Linear scale in the curved segment.
        a: f64,
        /// Offset in the curved segment.
        b: f64,
        /// Slope in the linear segment.
        c: f64,
        /// Cross-over point (device value below which linear is used).
        d: f64,
        /// Offset added to the curved segment (ICC type 4).
        e: f64,
        /// Offset added to the linear segment (ICC type 4).
        f: f64,
    },

    /// Tabulated curve: piecewise-linear over `values` (normalised to `[0, 1]`).
    Table(Vec<f64>),

    /// Identity (no curve — linear pass-through).
    Identity,
}

impl IccTrc {
    /// Apply the forward TRC transform (device → linear light).
    #[must_use]
    pub fn apply(&self, x: f64) -> f64 {
        let x_c = x.clamp(0.0, 1.0);
        match self {
            Self::Identity => x_c,
            Self::Gamma(g) => x_c.powf(*g),
            Self::Parametric {
                gamma,
                a,
                b,
                c,
                d,
                e,
                f,
            } => {
                if x_c >= *d {
                    (a * x_c + b).max(0.0).powf(*gamma) + e
                } else {
                    c * x_c + f
                }
            }
            Self::Table(values) => {
                if values.is_empty() {
                    return x_c;
                }
                let n = values.len() - 1;
                let pos = x_c * n as f64;
                let i0 = pos.floor() as usize;
                let i1 = (i0 + 1).min(n);
                let frac = pos - i0 as f64;
                values[i0] * (1.0 - frac) + values[i1] * frac
            }
        }
    }

    /// Apply the inverse TRC transform (linear light → device).
    #[must_use]
    pub fn apply_inverse(&self, y: f64) -> f64 {
        let y_c = y.clamp(0.0, 1.0);
        match self {
            Self::Identity => y_c,
            Self::Gamma(g) => {
                if *g <= 0.0 {
                    y_c
                } else {
                    y_c.powf(1.0 / g)
                }
            }
            Self::Parametric {
                gamma,
                a,
                b,
                c,
                d,
                e,
                f,
            } => {
                // Determine linear threshold in linear-light space
                let y_thresh = if *d > 0.0 {
                    c * d + f
                } else {
                    0.0
                };
                if y_c < y_thresh {
                    if *c <= 0.0 {
                        0.0
                    } else {
                        (y_c - f) / c
                    }
                } else {
                    let inner = (y_c - e).max(0.0).powf(1.0 / gamma);
                    if *a <= 0.0 {
                        0.0
                    } else {
                        (inner - b) / a
                    }
                }
            }
            Self::Table(values) => {
                // Reverse lookup via binary search
                if values.is_empty() {
                    return y_c;
                }
                if values.len() == 1 {
                    return 0.0;
                }
                // Find surrounding indices
                let n = values.len() - 1;
                let mut lo = 0usize;
                let mut hi = n;
                // Handle non-monotone: just clamp
                while hi > lo + 1 {
                    let mid = (lo + hi) / 2;
                    if values[mid] <= y_c {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                let v0 = values[lo];
                let v1 = values[hi];
                let dv = v1 - v0;
                if dv.abs() < 1e-15 {
                    return lo as f64 / n as f64;
                }
                let t = (y_c - v0) / dv;
                ((lo as f64 + t) / n as f64).clamp(0.0, 1.0)
            }
        }
    }
}

// ============================================================================
// Matrix+TRC profile (most common for display profiles)
// ============================================================================

/// An ICC matrix/TRC profile (monochromatic-input profile class).
///
/// The forward transform is:
/// ```text
/// linear_rgb = TRC(device_rgb)
/// xyz_d50    = matrix_to_xyz_d50 * linear_rgb
/// ```
#[derive(Debug, Clone)]
pub struct IccMatrixProfile {
    /// Human-readable description.
    pub description: String,
    /// 3×3 matrix from linear RGB to CIE XYZ adapted to D50 (ICC PCS white point).
    pub matrix_to_xyz_d50: [[f64; 3]; 3],
    /// Red channel TRC.
    pub trc_r: IccTrc,
    /// Green channel TRC.
    pub trc_g: IccTrc,
    /// Blue channel TRC.
    pub trc_b: IccTrc,
}

impl IccMatrixProfile {
    /// Convert device RGB to CIE XYZ D50 (the ICC PCS).
    #[must_use]
    pub fn device_to_xyz(&self, rgb: &Rgb) -> [f64; 3] {
        let linear = [
            self.trc_r.apply(rgb[0]),
            self.trc_g.apply(rgb[1]),
            self.trc_b.apply(rgb[2]),
        ];
        let m = &self.matrix_to_xyz_d50;
        [
            m[0][0] * linear[0] + m[0][1] * linear[1] + m[0][2] * linear[2],
            m[1][0] * linear[0] + m[1][1] * linear[1] + m[1][2] * linear[2],
            m[2][0] * linear[0] + m[2][1] * linear[1] + m[2][2] * linear[2],
        ]
    }

    /// Convert CIE XYZ D50 to device RGB.
    ///
    /// This is the inverse of [`Self::device_to_xyz`]; it inverts the matrix
    /// and then applies the inverse TRC.
    #[must_use]
    pub fn xyz_to_device(&self, xyz: &[f64; 3]) -> Rgb {
        let m_inv = invert_3x3(&self.matrix_to_xyz_d50);
        let linear = [
            m_inv[0][0] * xyz[0] + m_inv[0][1] * xyz[1] + m_inv[0][2] * xyz[2],
            m_inv[1][0] * xyz[0] + m_inv[1][1] * xyz[1] + m_inv[1][2] * xyz[2],
            m_inv[2][0] * xyz[0] + m_inv[2][1] * xyz[1] + m_inv[2][2] * xyz[2],
        ];
        [
            self.trc_r.apply_inverse(linear[0].clamp(0.0, 1.0)),
            self.trc_g.apply_inverse(linear[1].clamp(0.0, 1.0)),
            self.trc_b.apply_inverse(linear[2].clamp(0.0, 1.0)),
        ]
    }
}

// ============================================================================
// 3×3 matrix inversion helper
// ============================================================================

/// Invert a 3×3 matrix using Cramer's rule.  Returns the identity on
/// degenerate input (determinant ≈ 0).
fn invert_3x3(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

    if det.abs() < 1e-15 {
        // Return identity on degenerate input
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }

    let inv_det = 1.0 / det;

    [
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det,
        ],
    ]
}

// ============================================================================
// Output colour space for the baked LUT
// ============================================================================

/// Target display colour space used when baking the ICC profile into a LUT.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IccOutputSpace {
    /// sRGB (IEC 61966-2-1) — most common display target.
    Srgb,
    /// Linear sRGB (no transfer function applied to output).
    LinearSrgb,
    /// Rec.709 (BT.709) — broadcast standard (same primaries as sRGB, different TRC).
    Rec709,
    /// DCI-P3 with D65 white point (Display P3).
    DisplayP3,
}

// ============================================================================
// ICC → LUT converter
// ============================================================================

/// Converts an [`IccMatrixProfile`] into a flat 3-D LUT.
#[derive(Debug, Clone)]
pub struct IccToLutConverter {
    /// Source profile to convert from.
    pub profile: IccMatrixProfile,
    /// Output colour space the LUT maps into.
    pub output_space: IccOutputSpace,
}

impl IccToLutConverter {
    /// Create a new converter targeting `sRGB` output by default.
    #[must_use]
    pub fn new(profile: IccMatrixProfile) -> Self {
        Self {
            profile,
            output_space: IccOutputSpace::Srgb,
        }
    }

    /// Set the output colour space.
    #[must_use]
    pub fn with_output_space(mut self, space: IccOutputSpace) -> Self {
        self.output_space = space;
        self
    }

    /// Bake the profile into a flat `size × size × size` 3-D LUT.
    ///
    /// The LUT is indexed `[r][g][b]` in row-major order.  Each entry is an
    /// `[R, G, B]` output value in the target colour space.
    #[must_use]
    pub fn to_lut3d(&self, size: usize) -> Vec<Rgb> {
        let size = size.max(2);
        let scale = (size - 1) as f64;
        let mut lut = Vec::with_capacity(size * size * size);

        for ri in 0..size {
            for gi in 0..size {
                for bi in 0..size {
                    let rgb_in: Rgb = [ri as f64 / scale, gi as f64 / scale, bi as f64 / scale];
                    let xyz = self.profile.device_to_xyz(&rgb_in);
                    let rgb_out = self.xyz_to_output(&xyz);
                    lut.push(rgb_out);
                }
            }
        }
        lut
    }

    /// Convert XYZ D50 to the chosen output colour space.
    fn xyz_to_output(&self, xyz: &[f64; 3]) -> Rgb {
        match self.output_space {
            IccOutputSpace::LinearSrgb => xyz_d50_to_linear_srgb(xyz),
            IccOutputSpace::Srgb => {
                let linear = xyz_d50_to_linear_srgb(xyz);
                [
                    linear_to_srgb(linear[0]),
                    linear_to_srgb(linear[1]),
                    linear_to_srgb(linear[2]),
                ]
            }
            IccOutputSpace::Rec709 => {
                let linear = xyz_d50_to_linear_srgb(xyz); // same primaries
                [
                    linear_to_rec709(linear[0]),
                    linear_to_rec709(linear[1]),
                    linear_to_rec709(linear[2]),
                ]
            }
            IccOutputSpace::DisplayP3 => {
                let linear = xyz_d50_to_linear_p3(xyz);
                [
                    linear_to_srgb(linear[0]), // P3 uses sRGB-like EOTF
                    linear_to_srgb(linear[1]),
                    linear_to_srgb(linear[2]),
                ]
            }
        }
    }
}

// ============================================================================
// Colour space conversion helpers
// ============================================================================

/// CIE XYZ D50 → linear sRGB/Rec.709 (Bradford-adapted from D50 to D65).
fn xyz_d50_to_linear_srgb(xyz: &[f64; 3]) -> Rgb {
    // D50 → D65 Bradford adaptation then XYZ D65 → linear sRGB
    // Combined matrix (from ICC/ICC4 specification):
    let x = xyz[0];
    let y = xyz[1];
    let z = xyz[2];
    [
        (3.134_187 * x - 1.617_209 * y - 0.490_694 * z).clamp(0.0, 1.0),
        (-0.978_749 * x + 1.916_130 * y + 0.033_451 * z).clamp(0.0, 1.0),
        (0.071_942 * x - 0.228_866 * y + 1.405_388 * z).clamp(0.0, 1.0),
    ]
}

/// CIE XYZ D50 → linear Display P3.
fn xyz_d50_to_linear_p3(xyz: &[f64; 3]) -> Rgb {
    let x = xyz[0];
    let y = xyz[1];
    let z = xyz[2];
    // XYZ D50 → XYZ D65 Bradford, then XYZ D65 → P3 linear
    [
        (2.493_497 * x - 0.931_384 * y - 0.402_715 * z).clamp(0.0, 1.0),
        (-0.829_489 * x + 1.762_664 * y + 0.023_625 * z).clamp(0.0, 1.0),
        (0.035_846 * x - 0.076_172 * y + 0.956_885 * z).clamp(0.0, 1.0),
    ]
}

/// Linear light → sRGB electro-optical transfer function (IEC 61966-2-1).
fn linear_to_srgb(v: f64) -> f64 {
    let v = v.clamp(0.0, 1.0);
    if v <= 0.003_130_8 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

/// Linear light → Rec.709 electro-optical transfer function.
fn linear_to_rec709(v: f64) -> f64 {
    let v = v.clamp(0.0, 1.0);
    if v < 0.018 {
        v * 4.5
    } else {
        1.099 * v.powf(0.45) - 0.099
    }
}

// ============================================================================
// ICC profile metadata (minimal header fields)
// ============================================================================

/// Minimal ICC profile header fields needed for colour transform identification.
#[derive(Debug, Clone)]
pub struct IccHeader {
    /// Profile class (e.g. `"mntr"` for display monitor).
    pub profile_class: String,
    /// Colour space of data (e.g. `"RGB "`, `"CMYK"`).
    pub color_space: String,
    /// Profile Connection Space (`"XYZ "` or `"Lab "`).
    pub pcs: String,
    /// Rendering intent (0 = perceptual, 1 = relative colourimetric, …).
    pub rendering_intent: u32,
}

/// Parse a minimal ICC profile header from a raw byte slice.
///
/// Only the 4-byte signature fields in the 128-byte ICC header are decoded.
///
/// # Errors
///
/// Returns [`LutError::InvalidData`] if the slice is shorter than 128 bytes or
/// the file signature (`acsp`) is not found at offset 36.
pub fn parse_icc_header(data: &[u8]) -> LutResult<IccHeader> {
    if data.len() < 128 {
        return Err(LutError::InvalidData(
            "ICC data too short (need at least 128 bytes)".to_string(),
        ));
    }
    // Verify ICC signature at offset 36
    let sig = &data[36..40];
    if sig != b"acsp" {
        return Err(LutError::InvalidData(
            "Not a valid ICC profile (missing 'acsp' signature)".to_string(),
        ));
    }

    let profile_class = bytes4_to_str(&data[12..16]);
    let color_space = bytes4_to_str(&data[16..20]);
    let pcs = bytes4_to_str(&data[20..24]);
    let rendering_intent = u32::from_be_bytes([data[64], data[65], data[66], data[67]]);

    Ok(IccHeader {
        profile_class,
        color_space,
        pcs,
        rendering_intent,
    })
}

/// Convert a 4-byte ICC tag to a trimmed string.
fn bytes4_to_str(b: &[u8]) -> String {
    let s: String = b
        .iter()
        .map(|&c| if c.is_ascii_graphic() || c == b' ' { c as char } else { '?' })
        .collect();
    s.trim_end().to_string()
}

// ============================================================================
// Built-in canonical profile definitions
// ============================================================================

/// Returns a parametric ICC profile equivalent to sRGB (IEC 61966-2-1).
#[must_use]
pub fn srgb_profile() -> IccMatrixProfile {
    let trc = IccTrc::Parametric {
        gamma: 2.4,
        a: 1.0 / 1.055,
        b: 0.055 / 1.055,
        c: 1.0 / 12.92,
        d: 0.040_45,
        e: 0.0,
        f: 0.0,
    };
    IccMatrixProfile {
        description: "sRGB IEC61966-2-1".to_string(),
        matrix_to_xyz_d50: [
            [0.436_065, 0.385_151, 0.143_081],
            [0.222_491, 0.716_888, 0.060_621],
            [0.013_920, 0.097_045, 0.714_136],
        ],
        trc_r: trc.clone(),
        trc_g: trc.clone(),
        trc_b: trc,
    }
}

/// Returns a parametric ICC profile equivalent to Display P3.
#[must_use]
pub fn display_p3_profile() -> IccMatrixProfile {
    let trc = IccTrc::Parametric {
        gamma: 2.4,
        a: 1.0 / 1.055,
        b: 0.055 / 1.055,
        c: 1.0 / 12.92,
        d: 0.040_45,
        e: 0.0,
        f: 0.0,
    };
    IccMatrixProfile {
        description: "Display P3".to_string(),
        matrix_to_xyz_d50: [
            [0.515_102, 0.291_965, 0.157_230],
            [0.241_196, 0.692_236, 0.066_568],
            [-0.001_054, 0.041_893, 0.784_161],
        ],
        trc_r: trc.clone(),
        trc_g: trc.clone(),
        trc_b: trc,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Convenience: build the sRGB profile converter targeting sRGB.
    fn srgb_converter() -> IccToLutConverter {
        IccToLutConverter::new(srgb_profile())
    }

    // --- TRC tests -----------------------------------------------------------

    #[test]
    fn test_trc_identity() {
        let trc = IccTrc::Identity;
        assert!((trc.apply(0.5) - 0.5).abs() < 1e-12);
        assert!((trc.apply_inverse(0.5) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_trc_gamma_roundtrip() {
        let trc = IccTrc::Gamma(2.2);
        for v in [0.0, 0.1, 0.5, 0.9, 1.0] {
            let encoded = trc.apply(v);
            let decoded = trc.apply_inverse(encoded);
            assert!((decoded - v).abs() < 1e-10, "gamma roundtrip failed at {v}: got {decoded}");
        }
    }

    #[test]
    fn test_trc_table_lookup() {
        // Identity table
        let values: Vec<f64> = (0..=100).map(|i| i as f64 / 100.0).collect();
        let trc = IccTrc::Table(values);
        assert!((trc.apply(0.0) - 0.0).abs() < 1e-10);
        assert!((trc.apply(1.0) - 1.0).abs() < 1e-10);
        assert!((trc.apply(0.5) - 0.5).abs() < 0.02);
    }

    #[test]
    fn test_trc_table_inverse_lookup() {
        let values: Vec<f64> = (0..=100).map(|i| i as f64 / 100.0).collect();
        let trc = IccTrc::Table(values);
        assert!((trc.apply_inverse(0.5) - 0.5).abs() < 0.02);
    }

    #[test]
    fn test_trc_srgb_known_values() {
        // sRGB parametric curve
        let trc = IccTrc::Parametric {
            gamma: 2.4,
            a: 1.0 / 1.055,
            b: 0.055 / 1.055,
            c: 1.0 / 12.92,
            d: 0.040_45,
            e: 0.0,
            f: 0.0,
        };
        // sRGB(0) = 0
        assert!((trc.apply(0.0) - 0.0).abs() < 1e-10);
        // sRGB(1) = 1
        assert!((trc.apply(1.0) - 1.0).abs() < 1e-6);
        // sRGB(0.5) ≈ 0.214 (well-known reference value)
        let y = trc.apply(0.5);
        assert!((y - 0.2140).abs() < 0.001, "sRGB(0.5) = {y}");
    }

    // --- Matrix profile tests ------------------------------------------------

    #[test]
    fn test_srgb_profile_white_point() {
        // White input [1,1,1] should map to approximately D50 white [0.9642, 1.0, 0.8249]
        let profile = srgb_profile();
        let xyz = profile.device_to_xyz(&[1.0, 1.0, 1.0]);
        assert!((xyz[0] - 0.9642).abs() < 0.01, "X = {}", xyz[0]);
        assert!((xyz[1] - 1.0).abs() < 0.01, "Y = {}", xyz[1]);
        assert!((xyz[2] - 0.8249).abs() < 0.01, "Z = {}", xyz[2]);
    }

    #[test]
    fn test_srgb_profile_black_point() {
        let profile = srgb_profile();
        let xyz = profile.device_to_xyz(&[0.0, 0.0, 0.0]);
        assert!(xyz[0].abs() < 1e-10);
        assert!(xyz[1].abs() < 1e-10);
        assert!(xyz[2].abs() < 1e-10);
    }

    // --- Converter / LUT baking tests ----------------------------------------

    #[test]
    fn test_lut3d_size() {
        let conv = srgb_converter();
        let lut = conv.to_lut3d(5);
        assert_eq!(lut.len(), 5 * 5 * 5);
    }

    #[test]
    fn test_lut3d_identity_srgb_to_srgb() {
        // Converting sRGB → sRGB should produce a near-identity LUT
        let conv = IccToLutConverter::new(srgb_profile()).with_output_space(IccOutputSpace::Srgb);
        let lut = conv.to_lut3d(5);
        let scale = 4.0;
        // Check a few entries
        for ri in 0..5_usize {
            for gi in 0..5_usize {
                for bi in 0..5_usize {
                    let idx = ri * 25 + gi * 5 + bi;
                    let expected_r = ri as f64 / scale;
                    let expected_g = gi as f64 / scale;
                    let expected_b = bi as f64 / scale;
                    assert!(
                        (lut[idx][0] - expected_r).abs() < 0.01,
                        "R mismatch at ({ri},{gi},{bi}): {} vs {expected_r}",
                        lut[idx][0]
                    );
                    assert!(
                        (lut[idx][1] - expected_g).abs() < 0.01,
                        "G mismatch at ({ri},{gi},{bi}): {} vs {expected_g}",
                        lut[idx][1]
                    );
                    assert!(
                        (lut[idx][2] - expected_b).abs() < 0.01,
                        "B mismatch at ({ri},{gi},{bi}): {} vs {expected_b}",
                        lut[idx][2]
                    );
                }
            }
        }
    }

    #[test]
    fn test_lut3d_output_linear_srgb() {
        let conv = IccToLutConverter::new(srgb_profile())
            .with_output_space(IccOutputSpace::LinearSrgb);
        let lut = conv.to_lut3d(3);
        assert_eq!(lut.len(), 27);
        // Black → black
        assert!(lut[0][0].abs() < 1e-10);
    }

    #[test]
    fn test_lut3d_output_rec709() {
        let conv = IccToLutConverter::new(srgb_profile())
            .with_output_space(IccOutputSpace::Rec709);
        let lut = conv.to_lut3d(3);
        assert_eq!(lut.len(), 27);
    }

    #[test]
    fn test_lut3d_output_display_p3() {
        let conv = IccToLutConverter::new(display_p3_profile())
            .with_output_space(IccOutputSpace::DisplayP3);
        let lut = conv.to_lut3d(3);
        assert_eq!(lut.len(), 27);
        // All values should be in [0, 1]
        for entry in &lut {
            assert!(entry[0] >= 0.0 && entry[0] <= 1.0);
            assert!(entry[1] >= 0.0 && entry[1] <= 1.0);
            assert!(entry[2] >= 0.0 && entry[2] <= 1.0);
        }
    }

    // --- parse_icc_header tests -----------------------------------------------

    #[test]
    fn test_parse_icc_header_too_short() {
        let data = vec![0u8; 64];
        let result = parse_icc_header(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_icc_header_bad_signature() {
        let data = vec![0u8; 128];
        let result = parse_icc_header(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_icc_header_valid() {
        let mut data = vec![0u8; 128];
        // Write 'acsp' at offset 36
        data[36..40].copy_from_slice(b"acsp");
        // Write profile class 'mntr' at offset 12
        data[12..16].copy_from_slice(b"mntr");
        // Write colour space 'RGB ' at offset 16
        data[16..20].copy_from_slice(b"RGB ");
        // Write PCS 'XYZ ' at offset 20
        data[20..24].copy_from_slice(b"XYZ ");
        // Rendering intent = 0 at offset 64 (already 0)
        let header = parse_icc_header(&data).expect("should parse");
        assert_eq!(header.profile_class, "mntr");
        assert_eq!(header.color_space, "RGB");
        assert_eq!(header.pcs, "XYZ");
        assert_eq!(header.rendering_intent, 0);
    }

    // --- Matrix inversion test -----------------------------------------------

    #[test]
    fn test_invert_3x3_identity() {
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let inv = invert_3x3(&identity);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((inv[i][j] - expected).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_invert_3x3_srgb_matrix() {
        let m = srgb_profile().matrix_to_xyz_d50;
        let inv = invert_3x3(&m);
        // M * M^-1 should be close to identity
        for i in 0..3 {
            for j in 0..3 {
                let mut acc = 0.0f64;
                for k in 0..3 {
                    acc += m[i][k] * inv[k][j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((acc - expected).abs() < 1e-8, "M*M^-1[{i}][{j}] = {acc}");
            }
        }
    }
}
