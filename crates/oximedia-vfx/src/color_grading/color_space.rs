//! OCIO-style color space transforms for the color-grading pipeline.
//!
//! Provides working-space conversions between common color spaces used in
//! post-production: Rec.709, Rec.2020, linear light, and ACEScg.
//!
//! All transforms operate on normalised linear-light RGB triplets (`[0.0, ∞)` range).
//! Non-linear (gamma-encoded) inputs must be linearised first.
//!
//! # Architecture
//!
//! A [`ColorSpaceTransform`] encodes a 3×3 linear matrix that converts
//! primaries between two color spaces.  Matrices are precomputed constants
//! derived from the Bradford chromatic adaptation and the ITU/ACES standards.
//!
//! A [`ColorGradingPipeline`] chains an input linearisation (scene/camera
//! encoding → linear), an optional working-space conversion, and an output
//! non-linearisation (linear → display encoding).

use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────────────────────────
// ColorSpaceId
// ─────────────────────────────────────────────────────────────────────────────

/// Identifies a color space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ColorSpaceId {
    /// ITU-R BT.709 primaries, linear light (no gamma applied).
    Rec709Linear,
    /// ITU-R BT.709 primaries with standard gamma 2.4 encoding.
    Rec709Gamma,
    /// ITU-R BT.2020 primaries, linear light.
    Rec2020Linear,
    /// ITU-R BT.2020 primaries with PQ (ST 2084) transfer function (not handled here).
    Rec2020Pq,
    /// CIE 1931 XYZ (D65 white point), linear.
    CieXyz,
    /// ACEScg linear color space (AP1 primaries, D60).
    AcesCg,
    /// ACES2065-1 (AP0 primaries, D60).  Wide-gamut archival space.
    Aces2065,
    /// sRGB primaries (same as Rec.709), linear light.
    SrgbLinear,
    /// sRGB with IEC 61966-2-1 piecewise gamma (~2.2).
    SrgbGamma,
}

impl ColorSpaceId {
    /// Human-readable name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Rec709Linear => "Rec.709 (linear)",
            Self::Rec709Gamma => "Rec.709 (gamma 2.4)",
            Self::Rec2020Linear => "Rec.2020 (linear)",
            Self::Rec2020Pq => "Rec.2020 (PQ)",
            Self::CieXyz => "CIE XYZ (D65)",
            Self::AcesCg => "ACEScg (AP1)",
            Self::Aces2065 => "ACES 2065-1 (AP0)",
            Self::SrgbLinear => "sRGB (linear)",
            Self::SrgbGamma => "sRGB (gamma)",
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3×3 Matrix helper
// ─────────────────────────────────────────────────────────────────────────────

/// A row-major 3×3 matrix for linear RGB transforms.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Matrix3x3 {
    /// Row-major coefficients: `[row0col0, row0col1, row0col2, row1…]`.
    pub m: [f32; 9],
}

impl Matrix3x3 {
    /// Identity matrix.
    #[must_use]
    pub const fn identity() -> Self {
        Self {
            m: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        }
    }

    /// Apply the matrix to an RGB triplet `(r, g, b)`.
    #[must_use]
    pub fn apply(&self, r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        let [m00, m01, m02, m10, m11, m12, m20, m21, m22] = self.m;
        (
            m00 * r + m01 * g + m02 * b,
            m10 * r + m11 * g + m12 * b,
            m20 * r + m21 * g + m22 * b,
        )
    }

    /// Matrix multiplication: `self * rhs`.
    #[must_use]
    pub fn mul(&self, rhs: &Self) -> Self {
        let a = &self.m;
        let b = &rhs.m;
        Self {
            m: [
                a[0] * b[0] + a[1] * b[3] + a[2] * b[6],
                a[0] * b[1] + a[1] * b[4] + a[2] * b[7],
                a[0] * b[2] + a[1] * b[5] + a[2] * b[8],
                a[3] * b[0] + a[4] * b[3] + a[5] * b[6],
                a[3] * b[1] + a[4] * b[4] + a[5] * b[7],
                a[3] * b[2] + a[4] * b[5] + a[5] * b[8],
                a[6] * b[0] + a[7] * b[3] + a[8] * b[6],
                a[6] * b[1] + a[7] * b[4] + a[8] * b[7],
                a[6] * b[2] + a[7] * b[5] + a[8] * b[8],
            ],
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Precomputed standard matrices
// ─────────────────────────────────────────────────────────────────────────────

/// Rec.709 → CIE XYZ (D65).
/// Source: ITU-R BT.709 standard.
pub const REC709_TO_XYZ: Matrix3x3 = Matrix3x3 {
    m: [
        0.412_391, 0.357_585, 0.180_481,
        0.212_639, 0.715_170, 0.072_192,
        0.019_331, 0.119_195, 0.950_532,
    ],
};

/// CIE XYZ (D65) → Rec.709.
pub const XYZ_TO_REC709: Matrix3x3 = Matrix3x3 {
    m: [
         3.240_970, -1.537_383, -0.498_611,
        -0.969_244,  1.875_968,  0.041_555,
         0.055_630, -0.203_977,  1.056_972,
    ],
};

/// Rec.2020 → CIE XYZ (D65).
/// Source: ITU-R BT.2020 Table 4.
pub const REC2020_TO_XYZ: Matrix3x3 = Matrix3x3 {
    m: [
        0.636_958, 0.144_617, 0.168_881,
        0.262_700, 0.677_998, 0.059_302,
        0.000_000, 0.028_073, 1.060_985,
    ],
};

/// CIE XYZ (D65) → Rec.2020.
pub const XYZ_TO_REC2020: Matrix3x3 = Matrix3x3 {
    m: [
         1.716_651, -0.355_671, -0.253_366,
        -0.666_684,  1.616_481,  0.015_769,
         0.017_640, -0.042_771,  0.942_103,
    ],
};

/// ACEScg (AP1, D60) → CIE XYZ (D65).
/// Via Bradford adaptation D60→D65 from AP1 primaries.
pub const ACESCG_TO_XYZ: Matrix3x3 = Matrix3x3 {
    m: [
        0.664_993, 0.134_004, 0.156_346,
        0.272_229, 0.674_082, 0.053_689,
       -0.005_575, 0.004_061, 1.010_486,
    ],
};

/// CIE XYZ (D65) → ACEScg (AP1).
pub const XYZ_TO_ACESCG: Matrix3x3 = Matrix3x3 {
    m: [
         1.641_023, -0.324_803, -0.236_477,
        -0.663_662,  1.615_332,  0.016_756,
         0.011_722, -0.008_284,  0.988_395,
    ],
};

/// ACES 2065-1 (AP0, D60) → CIE XYZ (D65).
pub const ACES2065_TO_XYZ: Matrix3x3 = Matrix3x3 {
    m: [
        0.952_552, 0.000_000, 0.000_094,
        0.343_967, 0.728_167, -0.072_134,
        0.000_000, 0.000_000, 1.008_825,
    ],
};

/// CIE XYZ (D65) → ACES 2065-1 (AP0).
pub const XYZ_TO_ACES2065: Matrix3x3 = Matrix3x3 {
    m: [
         1.049_811, 0.000_000, -0.000_098,
        -0.495_903, 1.373_314,  0.098_240,
         0.000_000, 0.000_000,  0.991_252,
    ],
};

// ─────────────────────────────────────────────────────────────────────────────
// Transfer functions
// ─────────────────────────────────────────────────────────────────────────────

/// Linearise a single channel value from sRGB gamma encoding.
#[must_use]
pub fn srgb_to_linear(v: f32) -> f32 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// Apply sRGB gamma to a linear channel value.
#[must_use]
pub fn linear_to_srgb(v: f32) -> f32 {
    let v = v.max(0.0);
    if v <= 0.003_130_8 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

/// Linearise Rec.709 / Rec.2020 OETF (gamma 2.4).
#[must_use]
pub fn rec_to_linear(v: f32) -> f32 {
    if v < 0.081 {
        v / 4.5
    } else {
        ((v + 0.099) / 1.099).powf(1.0 / 0.45)
    }
}

/// Apply Rec.709 OETF (gamma 2.4) to linear.
#[must_use]
pub fn linear_to_rec(v: f32) -> f32 {
    let v = v.max(0.0);
    if v < 0.018 {
        v * 4.5
    } else {
        1.099 * v.powf(0.45) - 0.099
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ColorSpaceTransform
// ─────────────────────────────────────────────────────────────────────────────

/// A working-space color transform: convert linear RGB from `src` to `dst`.
///
/// Both src and dst must be *linear* (no gamma).  Use the transfer function
/// helpers to linearise/encode before/after.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorSpaceTransform {
    /// Source color space (must be linear).
    pub src: ColorSpaceId,
    /// Destination color space (must be linear).
    pub dst: ColorSpaceId,
    /// Combined src→XYZ→dst transform matrix.
    matrix: Matrix3x3,
}

impl ColorSpaceTransform {
    /// Build a transform from `src` to `dst`.
    ///
    /// Returns `None` when no conversion matrix is available for the pair.
    /// Currently supported: any combination of `Rec709Linear`, `Rec2020Linear`,
    /// `SrgbLinear`, `CieXyz`, `AcesCg`, `Aces2065`.
    #[must_use]
    pub fn new(src: ColorSpaceId, dst: ColorSpaceId) -> Option<Self> {
        if src == dst {
            return Some(Self {
                src,
                dst,
                matrix: Matrix3x3::identity(),
            });
        }
        let to_xyz = Self::to_xyz_matrix(src)?;
        let from_xyz = Self::from_xyz_matrix(dst)?;
        let matrix = from_xyz.mul(&to_xyz);
        Some(Self { src, dst, matrix })
    }

    /// Identity transform (no conversion).
    #[must_use]
    pub fn identity(space: ColorSpaceId) -> Self {
        Self {
            src: space,
            dst: space,
            matrix: Matrix3x3::identity(),
        }
    }

    /// Apply the transform to a single RGB triplet.
    ///
    /// Input and output values may be negative or exceed 1.0 (wide-gamut / HDR).
    #[must_use]
    pub fn apply(&self, r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        self.matrix.apply(r, g, b)
    }

    /// Apply the transform to a flat interleaved RGB slice (3 floats per pixel).
    ///
    /// Returns the transformed slice.  Length must be divisible by 3.
    #[must_use]
    pub fn apply_frame(&self, pixels: &[f32]) -> Vec<f32> {
        let mut out = Vec::with_capacity(pixels.len());
        for chunk in pixels.chunks_exact(3) {
            let (r, g, b) = self.apply(chunk[0], chunk[1], chunk[2]);
            out.push(r);
            out.push(g);
            out.push(b);
        }
        out
    }

    /// Get the underlying 3×3 matrix.
    #[must_use]
    pub fn matrix(&self) -> &Matrix3x3 {
        &self.matrix
    }

    fn to_xyz_matrix(src: ColorSpaceId) -> Option<Matrix3x3> {
        match src {
            ColorSpaceId::Rec709Linear | ColorSpaceId::SrgbLinear => Some(REC709_TO_XYZ),
            ColorSpaceId::Rec2020Linear => Some(REC2020_TO_XYZ),
            ColorSpaceId::AcesCg => Some(ACESCG_TO_XYZ),
            ColorSpaceId::Aces2065 => Some(ACES2065_TO_XYZ),
            ColorSpaceId::CieXyz => Some(Matrix3x3::identity()),
            _ => None,
        }
    }

    fn from_xyz_matrix(dst: ColorSpaceId) -> Option<Matrix3x3> {
        match dst {
            ColorSpaceId::Rec709Linear | ColorSpaceId::SrgbLinear => Some(XYZ_TO_REC709),
            ColorSpaceId::Rec2020Linear => Some(XYZ_TO_REC2020),
            ColorSpaceId::AcesCg => Some(XYZ_TO_ACESCG),
            ColorSpaceId::Aces2065 => Some(XYZ_TO_ACES2065),
            ColorSpaceId::CieXyz => Some(Matrix3x3::identity()),
            _ => None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ColorGradingPipeline
// ─────────────────────────────────────────────────────────────────────────────

/// Stage in the color-grading pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PipelineStage {
    /// Convert from gamma-encoded input to linear light.
    Linearise(ColorSpaceId),
    /// Apply a color-space transform in linear light.
    Transform(ColorSpaceTransform),
    /// Apply a global exposure offset in linear light (stops: positive = brighter).
    Exposure(f32),
    /// Apply a per-channel gain (linear multipliers).
    ChannelGain { r: f32, g: f32, b: f32 },
    /// Encode from linear light to output gamma.
    Encode(ColorSpaceId),
}

/// A sequential list of color-grading operations for pipeline-based transforms.
///
/// Pixel values travel through each stage in order.  Use [`apply_rgb`] to
/// process a single RGB triplet through the entire pipeline.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ColorGradingPipeline {
    stages: Vec<PipelineStage>,
}

impl ColorGradingPipeline {
    /// Create an empty pipeline.
    #[must_use]
    pub fn new() -> Self {
        Self { stages: Vec::new() }
    }

    /// Add a stage to the end of the pipeline.
    pub fn push(&mut self, stage: PipelineStage) {
        self.stages.push(stage);
    }

    /// Common pipeline: sRGB input → ACEScg working space → Rec.709 output.
    #[must_use]
    pub fn srgb_to_rec709_via_aces() -> Self {
        let mut p = Self::new();
        p.push(PipelineStage::Linearise(ColorSpaceId::SrgbGamma));
        if let Some(t) =
            ColorSpaceTransform::new(ColorSpaceId::SrgbLinear, ColorSpaceId::AcesCg)
        {
            p.push(PipelineStage::Transform(t));
        }
        if let Some(t) =
            ColorSpaceTransform::new(ColorSpaceId::AcesCg, ColorSpaceId::Rec709Linear)
        {
            p.push(PipelineStage::Transform(t));
        }
        p.push(PipelineStage::Encode(ColorSpaceId::Rec709Gamma));
        p
    }

    /// Apply all pipeline stages to a single RGB triplet `[0, 1]` (sRGB/Rec.709
    /// convention for gamma stages, linear for others).
    #[must_use]
    pub fn apply_rgb(&self, r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        let mut cur = (r, g, b);
        for stage in &self.stages {
            cur = match stage {
                PipelineStage::Linearise(cs) => match cs {
                    ColorSpaceId::SrgbGamma => (
                        srgb_to_linear(cur.0),
                        srgb_to_linear(cur.1),
                        srgb_to_linear(cur.2),
                    ),
                    ColorSpaceId::Rec709Gamma => (
                        rec_to_linear(cur.0),
                        rec_to_linear(cur.1),
                        rec_to_linear(cur.2),
                    ),
                    _ => cur,
                },
                PipelineStage::Transform(t) => t.apply(cur.0, cur.1, cur.2),
                PipelineStage::Exposure(stops) => {
                    let gain = (2.0f32).powf(*stops);
                    (cur.0 * gain, cur.1 * gain, cur.2 * gain)
                }
                PipelineStage::ChannelGain { r, g, b } => (cur.0 * r, cur.1 * g, cur.2 * b),
                PipelineStage::Encode(cs) => match cs {
                    ColorSpaceId::SrgbGamma => (
                        linear_to_srgb(cur.0),
                        linear_to_srgb(cur.1),
                        linear_to_srgb(cur.2),
                    ),
                    ColorSpaceId::Rec709Gamma => (
                        linear_to_rec(cur.0),
                        linear_to_rec(cur.1),
                        linear_to_rec(cur.2),
                    ),
                    _ => cur,
                },
            };
        }
        cur
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_identity_transform_passthrough() {
        let t = ColorSpaceTransform::identity(ColorSpaceId::Rec709Linear);
        let (r, g, b) = t.apply(0.3, 0.5, 0.7);
        assert!(approx(r, 0.3, 1e-5));
        assert!(approx(g, 0.5, 1e-5));
        assert!(approx(b, 0.7, 1e-5));
    }

    #[test]
    fn test_same_space_is_identity() {
        let t = ColorSpaceTransform::new(ColorSpaceId::Rec709Linear, ColorSpaceId::Rec709Linear)
            .expect("same space");
        let (r, g, b) = t.apply(0.2, 0.4, 0.6);
        assert!(approx(r, 0.2, 1e-5));
        assert!(approx(g, 0.4, 1e-5));
        assert!(approx(b, 0.6, 1e-5));
    }

    #[test]
    fn test_rec709_to_rec2020_roundtrip() {
        let fwd = ColorSpaceTransform::new(ColorSpaceId::Rec709Linear, ColorSpaceId::Rec2020Linear)
            .expect("fwd");
        let inv =
            ColorSpaceTransform::new(ColorSpaceId::Rec2020Linear, ColorSpaceId::Rec709Linear)
                .expect("inv");

        let (r0, g0, b0) = (0.4, 0.5, 0.3);
        let (r1, g1, b1) = fwd.apply(r0, g0, b0);
        let (r2, g2, b2) = inv.apply(r1, g1, b1);

        assert!(approx(r2, r0, 1e-4), "R roundtrip: {r2} vs {r0}");
        assert!(approx(g2, g0, 1e-4), "G roundtrip: {g2} vs {g0}");
        assert!(approx(b2, b0, 1e-4), "B roundtrip: {b2} vs {b0}");
    }

    #[test]
    fn test_rec709_to_acescg_roundtrip() {
        let fwd = ColorSpaceTransform::new(ColorSpaceId::Rec709Linear, ColorSpaceId::AcesCg)
            .expect("fwd");
        let inv = ColorSpaceTransform::new(ColorSpaceId::AcesCg, ColorSpaceId::Rec709Linear)
            .expect("inv");
        let (r0, g0, b0) = (0.6, 0.3, 0.2);
        let (r1, g1, b1) = fwd.apply(r0, g0, b0);
        let (r2, g2, b2) = inv.apply(r1, g1, b1);
        assert!(approx(r2, r0, 1e-4), "R: {r2} vs {r0}");
        assert!(approx(g2, g0, 1e-4), "G: {g2} vs {g0}");
        assert!(approx(b2, b0, 1e-4), "B: {b2} vs {b0}");
    }

    #[test]
    fn test_matrix3x3_identity() {
        let m = Matrix3x3::identity();
        let (r, g, b) = m.apply(1.0, 2.0, 3.0);
        assert!(approx(r, 1.0, 1e-6));
        assert!(approx(g, 2.0, 1e-6));
        assert!(approx(b, 3.0, 1e-6));
    }

    #[test]
    fn test_srgb_transfer_roundtrip() {
        for v in [0.0, 0.01, 0.1, 0.5, 0.9, 1.0] {
            let rt = linear_to_srgb(srgb_to_linear(v));
            assert!(approx(rt, v, 1e-5), "sRGB roundtrip failed at {v}: got {rt}");
        }
    }

    #[test]
    fn test_rec_transfer_roundtrip() {
        for v in [0.0, 0.01, 0.1, 0.5, 0.9, 1.0] {
            let rt = linear_to_rec(rec_to_linear(v));
            assert!(approx(rt, v, 1e-4), "Rec roundtrip failed at {v}: got {rt}");
        }
    }

    #[test]
    fn test_color_grading_pipeline_exposure() {
        let mut p = ColorGradingPipeline::new();
        p.push(PipelineStage::Exposure(1.0)); // +1 stop = ×2
        let (r, g, b) = p.apply_rgb(0.5, 0.25, 0.125);
        assert!(approx(r, 1.0, 1e-5));
        assert!(approx(g, 0.5, 1e-5));
        assert!(approx(b, 0.25, 1e-5));
    }

    #[test]
    fn test_color_grading_pipeline_channel_gain() {
        let mut p = ColorGradingPipeline::new();
        p.push(PipelineStage::ChannelGain {
            r: 2.0,
            g: 0.5,
            b: 1.0,
        });
        let (r, g, b) = p.apply_rgb(0.5, 0.4, 0.3);
        assert!(approx(r, 1.0, 1e-5));
        assert!(approx(g, 0.2, 1e-5));
        assert!(approx(b, 0.3, 1e-5));
    }

    #[test]
    fn test_apply_frame_length() {
        let t = ColorSpaceTransform::identity(ColorSpaceId::Rec709Linear);
        let pixels = vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6];
        let out = t.apply_frame(&pixels);
        assert_eq!(out.len(), 6);
    }

    #[test]
    fn test_color_space_id_names_non_empty() {
        let ids = [
            ColorSpaceId::Rec709Linear,
            ColorSpaceId::Rec709Gamma,
            ColorSpaceId::Rec2020Linear,
            ColorSpaceId::AcesCg,
            ColorSpaceId::SrgbGamma,
        ];
        for id in ids {
            assert!(!id.name().is_empty());
        }
    }

    #[test]
    fn test_unsupported_gamma_transform_returns_none() {
        // Rec709Gamma → Rec709Gamma would need gamma linearisation, not a matrix
        let t = ColorSpaceTransform::new(ColorSpaceId::Rec709Gamma, ColorSpaceId::Rec2020Linear);
        // Rec709Gamma is not handled by to_xyz_matrix, so should be None
        assert!(t.is_none());
    }
}
