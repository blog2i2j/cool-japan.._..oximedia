//! OCIO-style color management for the VFX grading pipeline.
//!
//! Provides working-space transforms so that colour grading operations happen
//! in a well-defined, scene-referred linear colour space.  The pipeline
//! supports:
//!
//! - **Color spaces**: sRGB, Linear sRGB, ACEScg, Rec.709, Rec.2020, DCI-P3,
//!   Display P3.
//! - **Transfer functions**: sRGB OETF/EOTF, PQ (ST 2084), HLG, Gamma 2.2/2.6,
//!   Linear.
//! - **Chromatic adaptation**: Bradford transform for white-point conversion
//!   (D50, D65, ACES).
//! - **ColorPipeline**: chains input → working → output transforms and applies
//!   them to RGBA frame data.
//!
//! # Example
//!
//! ```
//! use oximedia_vfx::color_management::{ColorPipeline, ColorSpace, TransferFunction};
//!
//! let pipeline = ColorPipeline::new(
//!     ColorSpace::Srgb,
//!     ColorSpace::AcesCg,
//!     ColorSpace::Srgb,
//! );
//! let mut pixel = [0.5f32, 0.3, 0.1];
//! pipeline.input_to_working(&mut pixel);
//! // ... grading operations in ACEScg ...
//! pipeline.working_to_output(&mut pixel);
//! ```

use crate::{Frame, VfxResult};
use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────────────────────────
// White point
// ─────────────────────────────────────────────────────────────────────────────

/// CIE standard illuminant white points (XYZ tristimulus, Y=1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WhitePoint {
    /// D50 (ICC profile connection space).
    D50,
    /// D65 (sRGB, Rec.709, Rec.2020 reference).
    D65,
    /// ACES white (≈ D60, CIE 1931 xy = 0.32168, 0.33767).
    AcesWhite,
}

impl WhitePoint {
    /// XYZ coordinates (Y = 1 normalised).
    #[must_use]
    pub const fn xyz(&self) -> [f32; 3] {
        match self {
            Self::D50 => [0.964_22, 1.0, 0.825_21],
            Self::D65 => [0.950_47, 1.0, 1.088_83],
            Self::AcesWhite => [0.952_646, 1.0, 1.008_825],
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Transfer functions
// ─────────────────────────────────────────────────────────────────────────────

/// Electro-optical / opto-electronic transfer functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TransferFunction {
    /// Linear (no transfer).
    Linear,
    /// sRGB piecewise curve (IEC 61966-2-1).
    Srgb,
    /// Pure power-law gamma.
    Gamma22,
    /// BT.1886 / cinema gamma 2.6.
    Gamma26,
    /// PQ (SMPTE ST 2084).
    Pq,
    /// HLG (ARIB STD-B67).
    Hlg,
}

impl TransferFunction {
    /// Linearise a single channel value (EOTF: display-referred → linear).
    #[must_use]
    pub fn to_linear(self, v: f32) -> f32 {
        let v = v.clamp(0.0, 1.0);
        match self {
            Self::Linear => v,
            Self::Srgb => {
                if v <= 0.040_45 {
                    v / 12.92
                } else {
                    ((v + 0.055) / 1.055).powf(2.4)
                }
            }
            Self::Gamma22 => v.powf(2.2),
            Self::Gamma26 => v.powf(2.6),
            Self::Pq => pq_eotf(v),
            Self::Hlg => hlg_oetf_inv(v),
        }
    }

    /// Encode a linear value (OETF / inverse EOTF: linear → display-referred).
    #[must_use]
    pub fn from_linear(self, v: f32) -> f32 {
        let v = v.max(0.0);
        match self {
            Self::Linear => v.min(1.0),
            Self::Srgb => {
                if v <= 0.003_130_8 {
                    (12.92 * v).min(1.0)
                } else {
                    (1.055 * v.powf(1.0 / 2.4) - 0.055).clamp(0.0, 1.0)
                }
            }
            Self::Gamma22 => v.powf(1.0 / 2.2).min(1.0),
            Self::Gamma26 => v.powf(1.0 / 2.6).min(1.0),
            Self::Pq => pq_oetf(v),
            Self::Hlg => hlg_oetf(v),
        }
    }
}

/// PQ EOTF (ST 2084): signal value → normalised linear light.
fn pq_eotf(v: f32) -> f32 {
    let m1: f32 = 0.159_301_76;
    let m2: f32 = 78.843_75;
    let c1: f32 = 0.835_937_5;
    let c2: f32 = 18.851_563;
    let c3: f32 = 18.6875;

    let vp = v.max(0.0).powf(1.0 / m2);
    let num = (vp - c1).max(0.0);
    let den = c2 - c3 * vp;
    if den <= 0.0 {
        return 0.0;
    }
    (num / den).powf(1.0 / m1)
}

/// PQ OETF (inverse): normalised linear light → signal value.
fn pq_oetf(v: f32) -> f32 {
    let m1: f32 = 0.159_301_76;
    let m2: f32 = 78.843_75;
    let c1: f32 = 0.835_937_5;
    let c2: f32 = 18.851_563;
    let c3: f32 = 18.6875;

    let yp = v.max(0.0).powf(m1);
    let num = c1 + c2 * yp;
    let den = 1.0 + c3 * yp;
    (num / den).powf(m2).clamp(0.0, 1.0)
}

/// HLG OETF: linear → signal.
fn hlg_oetf(v: f32) -> f32 {
    let a: f32 = 0.178_832_77;
    let b: f32 = 0.284_668_92;
    let c: f32 = 0.559_910_7;
    if v <= 1.0 / 12.0 {
        (3.0 * v).sqrt().clamp(0.0, 1.0)
    } else {
        (a * (12.0 * v - b).ln() + c).clamp(0.0, 1.0)
    }
}

/// HLG inverse OETF: signal → linear.
fn hlg_oetf_inv(v: f32) -> f32 {
    let a: f32 = 0.178_832_77;
    let b: f32 = 0.284_668_92;
    let c: f32 = 0.559_910_7;
    if v <= 0.5 {
        (v * v / 3.0).clamp(0.0, 1.0)
    } else {
        let inner = ((v - c) / a).exp() + b;
        (inner / 12.0).clamp(0.0, 1.0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Color space
// ─────────────────────────────────────────────────────────────────────────────

/// Supported colour spaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ColorSpace {
    /// sRGB (IEC 61966-2-1), D65, sRGB transfer.
    Srgb,
    /// Linear sRGB (same primaries, linear transfer).
    LinearSrgb,
    /// ACEScg (AP1 primaries, linear, ACES white).
    AcesCg,
    /// Rec.709 (same primaries as sRGB, BT.1886 gamma).
    Rec709,
    /// Rec.2020 (wide gamut, linear).
    Rec2020,
    /// DCI-P3 (D65 variant, gamma 2.6).
    DciP3,
    /// Display P3 (DCI-P3 primaries, sRGB transfer, D65).
    DisplayP3,
}

impl ColorSpace {
    /// Transfer function associated with this colour space.
    #[must_use]
    pub const fn transfer(self) -> TransferFunction {
        match self {
            Self::Srgb | Self::DisplayP3 => TransferFunction::Srgb,
            Self::LinearSrgb | Self::AcesCg | Self::Rec2020 => TransferFunction::Linear,
            Self::Rec709 => TransferFunction::Gamma22,
            Self::DciP3 => TransferFunction::Gamma26,
        }
    }

    /// White point for this colour space.
    #[must_use]
    pub const fn white_point(self) -> WhitePoint {
        match self {
            Self::AcesCg => WhitePoint::AcesWhite,
            _ => WhitePoint::D65,
        }
    }

    /// RGB-to-XYZ 3x3 matrix (row-major, for linear-light values).
    #[must_use]
    pub fn rgb_to_xyz_matrix(&self) -> [[f32; 3]; 3] {
        match self {
            Self::Srgb | Self::LinearSrgb | Self::Rec709 => [
                [0.412_390_8, 0.357_584_3, 0.180_480_8],
                [0.212_639, 0.715_169, 0.072_192],
                [0.019_330_8, 0.119_194_8, 0.950_532_2],
            ],
            Self::AcesCg => [
                [0.662_454, 0.134_004, 0.156_188],
                [0.272_229, 0.674_082, 0.053_689],
                [-0.005_575, 0.004_060, 1.010_339],
            ],
            Self::Rec2020 => [
                [0.636_958, 0.144_617, 0.168_881],
                [0.262_700, 0.677_998, 0.059_302],
                [0.000_000, 0.028_073, 1.060_985],
            ],
            Self::DciP3 | Self::DisplayP3 => [
                [0.486_571, 0.265_668, 0.198_217],
                [0.228_975, 0.691_739, 0.079_287],
                [0.000_000, 0.045_113, 1.043_944],
            ],
        }
    }

    /// XYZ-to-RGB 3x3 matrix (inverse of `rgb_to_xyz_matrix`).
    #[must_use]
    pub fn xyz_to_rgb_matrix(&self) -> [[f32; 3]; 3] {
        invert_3x3(self.rgb_to_xyz_matrix())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Matrix helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Multiply a 3x3 matrix by a 3-element column vector.
fn mat3_mul_vec(m: &[[f32; 3]; 3], v: &[f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// Multiply two 3x3 matrices.
fn mat3_mul(a: &[[f32; 3]; 3], b: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut out = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    out
}

/// Invert a 3x3 matrix using Cramer's rule.
fn invert_3x3(m: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if det.abs() < 1e-12 {
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

// ─────────────────────────────────────────────────────────────────────────────
// Bradford chromatic adaptation
// ─────────────────────────────────────────────────────────────────────────────

/// Bradford LMS cone-response matrix.
const BRADFORD: [[f32; 3]; 3] = [
    [0.895_1, 0.266_4, -0.161_4],
    [-0.750_2, 1.713_5, 0.036_8],
    [0.038_9, -0.068_5, 1.029_6],
];

/// Compute Bradford chromatic adaptation matrix from `src` to `dst` white point.
fn bradford_adaptation(src: WhitePoint, dst: WhitePoint) -> [[f32; 3]; 3] {
    if src == dst {
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }
    let src_xyz = src.xyz();
    let dst_xyz = dst.xyz();

    let src_lms = mat3_mul_vec(&BRADFORD, &src_xyz);
    let dst_lms = mat3_mul_vec(&BRADFORD, &dst_xyz);

    // Diagonal scale matrix
    let scale: [[f32; 3]; 3] = [
        [dst_lms[0] / src_lms[0], 0.0, 0.0],
        [0.0, dst_lms[1] / src_lms[1], 0.0],
        [0.0, 0.0, dst_lms[2] / src_lms[2]],
    ];

    let inv_brad = invert_3x3(BRADFORD);
    let tmp = mat3_mul(&scale, &BRADFORD);
    mat3_mul(&inv_brad, &tmp)
}

// ─────────────────────────────────────────────────────────────────────────────
// ColorPipeline
// ─────────────────────────────────────────────────────────────────────────────

/// A colour management pipeline: input → working → output.
///
/// Converts between colour spaces via XYZ connection space with Bradford
/// chromatic adaptation for white-point mismatches.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorPipeline {
    /// Input colour space.
    pub input: ColorSpace,
    /// Working colour space (where grading happens).
    pub working: ColorSpace,
    /// Output colour space.
    pub output: ColorSpace,
}

impl ColorPipeline {
    /// Create a new pipeline.
    #[must_use]
    pub fn new(input: ColorSpace, working: ColorSpace, output: ColorSpace) -> Self {
        Self {
            input,
            working,
            output,
        }
    }

    /// Identity pipeline (no conversion).
    #[must_use]
    pub fn identity() -> Self {
        Self::new(ColorSpace::LinearSrgb, ColorSpace::LinearSrgb, ColorSpace::LinearSrgb)
    }

    /// Build the combined 3x3 matrix to go from colour space `a` (linear) to
    /// colour space `b` (linear), through XYZ with Bradford adaptation.
    fn build_matrix(src: ColorSpace, dst: ColorSpace) -> [[f32; 3]; 3] {
        if src == dst {
            return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        }
        let to_xyz = src.rgb_to_xyz_matrix();
        let adapt = bradford_adaptation(src.white_point(), dst.white_point());
        let from_xyz = dst.xyz_to_rgb_matrix();
        let adapted_xyz = mat3_mul(&adapt, &to_xyz);
        mat3_mul(&from_xyz, &adapted_xyz)
    }

    /// Convert an RGB triplet from input space to working space (in-place).
    ///
    /// The values are expected in [0, 1] range, display-referred (with the
    /// input space's transfer function already baked in — i.e. the raw pixel
    /// values). The output will be linear-light in the working space.
    pub fn input_to_working(&self, rgb: &mut [f32; 3]) {
        // Step 1: linearise using input transfer function
        let tf_in = self.input.transfer();
        for ch in rgb.iter_mut() {
            *ch = tf_in.to_linear(*ch);
        }
        // Step 2: matrix transform
        let mat = Self::build_matrix(self.input, self.working);
        let out = mat3_mul_vec(&mat, rgb);
        *rgb = out;
    }

    /// Convert an RGB triplet from working space to output space (in-place).
    ///
    /// Input is linear-light in the working space.  Output is display-referred
    /// with the output space's transfer function applied.
    pub fn working_to_output(&self, rgb: &mut [f32; 3]) {
        let mat = Self::build_matrix(self.working, self.output);
        let out = mat3_mul_vec(&mat, rgb);
        *rgb = out;
        // Apply output transfer function
        let tf_out = self.output.transfer();
        for ch in rgb.iter_mut() {
            *ch = tf_out.from_linear(*ch);
        }
    }

    /// Apply the full input→working→output pipeline to a [`Frame`], performing
    /// grading in working space via a user-supplied closure.
    ///
    /// The closure receives `&mut [f32; 3]` in working-space linear light.
    pub fn apply_to_frame(
        &self,
        frame: &mut Frame,
        grade_fn: impl Fn(&mut [f32; 3]),
    ) -> VfxResult<()> {
        let pixel_count = (frame.width as usize) * (frame.height as usize);
        let tf_in = self.input.transfer();
        let tf_out = self.output.transfer();
        let mat_in = Self::build_matrix(self.input, self.working);
        let mat_out = Self::build_matrix(self.working, self.output);

        for p in 0..pixel_count {
            let base = p * 4;
            let mut rgb = [
                tf_in.to_linear(frame.data[base] as f32 / 255.0),
                tf_in.to_linear(frame.data[base + 1] as f32 / 255.0),
                tf_in.to_linear(frame.data[base + 2] as f32 / 255.0),
            ];
            rgb = mat3_mul_vec(&mat_in, &rgb);
            grade_fn(&mut rgb);
            rgb = mat3_mul_vec(&mat_out, &rgb);
            frame.data[base] = (tf_out.from_linear(rgb[0]) * 255.0).clamp(0.0, 255.0) as u8;
            frame.data[base + 1] = (tf_out.from_linear(rgb[1]) * 255.0).clamp(0.0, 255.0) as u8;
            frame.data[base + 2] = (tf_out.from_linear(rgb[2]) * 255.0).clamp(0.0, 255.0) as u8;
            // alpha unchanged
        }
        Ok(())
    }
}

impl Default for ColorPipeline {
    fn default() -> Self {
        Self::new(ColorSpace::Srgb, ColorSpace::AcesCg, ColorSpace::Srgb)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Transfer functions ──────────────────────────────────────────────

    #[test]
    fn test_srgb_roundtrip() {
        for i in 0..=10 {
            let v = i as f32 / 10.0;
            let lin = TransferFunction::Srgb.to_linear(v);
            let back = TransferFunction::Srgb.from_linear(lin);
            assert!(
                (v - back).abs() < 0.005,
                "sRGB roundtrip: v={v}, back={back}"
            );
        }
    }

    #[test]
    fn test_gamma22_roundtrip() {
        for i in 0..=10 {
            let v = i as f32 / 10.0;
            let lin = TransferFunction::Gamma22.to_linear(v);
            let back = TransferFunction::Gamma22.from_linear(lin);
            assert!(
                (v - back).abs() < 0.005,
                "gamma 2.2 roundtrip: v={v}, back={back}"
            );
        }
    }

    #[test]
    fn test_pq_roundtrip() {
        for i in 1..=10 {
            let v = i as f32 / 10.0;
            let lin = TransferFunction::Pq.to_linear(v);
            let back = TransferFunction::Pq.from_linear(lin);
            assert!(
                (v - back).abs() < 0.01,
                "PQ roundtrip: v={v}, lin={lin}, back={back}"
            );
        }
    }

    #[test]
    fn test_hlg_roundtrip() {
        for i in 1..=10 {
            let v = i as f32 / 10.0;
            let lin = TransferFunction::Hlg.to_linear(v);
            let back = TransferFunction::Hlg.from_linear(lin);
            assert!(
                (v - back).abs() < 0.02,
                "HLG roundtrip: v={v}, lin={lin}, back={back}"
            );
        }
    }

    #[test]
    fn test_linear_passthrough() {
        let v = 0.42;
        assert_eq!(TransferFunction::Linear.to_linear(v), v);
        assert_eq!(TransferFunction::Linear.from_linear(v), v);
    }

    // ── Matrix helpers ──────────────────────────────────────────────────

    #[test]
    fn test_mat3_identity_inverse() {
        let id = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let inv = invert_3x3(id);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((inv[i][j] - expected).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn test_srgb_matrix_roundtrip() {
        let m = ColorSpace::Srgb.rgb_to_xyz_matrix();
        let inv = ColorSpace::Srgb.xyz_to_rgb_matrix();
        let product = mat3_mul(&inv, &m);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (product[i][j] - expected).abs() < 1e-3,
                    "sRGB matrix roundtrip [{i}][{j}]: {}",
                    product[i][j]
                );
            }
        }
    }

    // ── Bradford adaptation ─────────────────────────────────────────────

    #[test]
    fn test_bradford_identity_when_same_wp() {
        let m = bradford_adaptation(WhitePoint::D65, WhitePoint::D65);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((m[i][j] - expected).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn test_bradford_d65_to_d50() {
        let m = bradford_adaptation(WhitePoint::D65, WhitePoint::D50);
        // The adaptation matrix should not be identity
        let is_identity = (0..3).all(|i| (0..3).all(|j| {
            let expected = if i == j { 1.0 } else { 0.0 };
            (m[i][j] - expected).abs() < 0.01
        }));
        assert!(!is_identity, "D65->D50 should not be identity");
    }

    // ── ColorPipeline ───────────────────────────────────────────────────

    #[test]
    fn test_pipeline_identity_no_change() {
        let pipe = ColorPipeline::identity();
        let mut rgb = [0.5, 0.3, 0.1];
        let original = rgb;
        pipe.input_to_working(&mut rgb);
        pipe.working_to_output(&mut rgb);
        for i in 0..3 {
            assert!(
                (rgb[i] - original[i]).abs() < 0.01,
                "identity pipeline should not change values: ch={i} orig={} got={}",
                original[i],
                rgb[i]
            );
        }
    }

    #[test]
    fn test_pipeline_srgb_to_acescg_roundtrip() {
        let pipe_fwd = ColorPipeline::new(ColorSpace::Srgb, ColorSpace::AcesCg, ColorSpace::Srgb);
        let mut rgb = [0.5, 0.3, 0.1];
        let original = rgb;
        pipe_fwd.input_to_working(&mut rgb);
        pipe_fwd.working_to_output(&mut rgb);
        for i in 0..3 {
            assert!(
                (rgb[i] - original[i]).abs() < 0.03,
                "sRGB->ACEScg->sRGB roundtrip: ch={i} orig={} got={}",
                original[i],
                rgb[i]
            );
        }
    }

    #[test]
    fn test_pipeline_apply_to_frame() {
        let pipe = ColorPipeline::new(ColorSpace::Srgb, ColorSpace::LinearSrgb, ColorSpace::Srgb);
        let mut frame = Frame::new(4, 4).expect("frame");
        frame.clear([128, 64, 200, 255]);

        pipe.apply_to_frame(&mut frame, |_rgb| {
            // No-op grading — just test the pipeline runs
        })
        .expect("apply_to_frame");

        // Alpha should be preserved
        for y in 0..4 {
            for x in 0..4 {
                let p = frame.get_pixel(x, y).unwrap_or([0; 4]);
                assert_eq!(p[3], 255, "alpha preserved at ({x},{y})");
            }
        }
    }

    #[test]
    fn test_pipeline_apply_to_frame_with_grading() {
        let pipe = ColorPipeline::new(ColorSpace::Srgb, ColorSpace::LinearSrgb, ColorSpace::Srgb);
        let mut frame = Frame::new(4, 4).expect("frame");
        frame.clear([128, 128, 128, 255]);

        // Brightness boost in linear space
        pipe.apply_to_frame(&mut frame, |rgb| {
            for ch in rgb.iter_mut() {
                *ch *= 1.5;
            }
        })
        .expect("apply");

        let p = frame.get_pixel(2, 2).unwrap_or([0; 4]);
        assert!(
            p[0] > 128,
            "brightness boost should increase pixel value: got {}",
            p[0]
        );
    }

    #[test]
    fn test_color_space_properties() {
        assert_eq!(ColorSpace::Srgb.transfer(), TransferFunction::Srgb);
        assert_eq!(ColorSpace::AcesCg.transfer(), TransferFunction::Linear);
        assert_eq!(ColorSpace::AcesCg.white_point(), WhitePoint::AcesWhite);
        assert_eq!(ColorSpace::Srgb.white_point(), WhitePoint::D65);
    }

    #[test]
    fn test_white_point_xyz_values() {
        let d65 = WhitePoint::D65.xyz();
        // D65 Y should be 1.0
        assert!((d65[1] - 1.0).abs() < 1e-5);
        // D65 X should be ~0.9505
        assert!((d65[0] - 0.9505).abs() < 0.01);
    }

    #[test]
    fn test_default_pipeline() {
        let pipe = ColorPipeline::default();
        assert_eq!(pipe.input, ColorSpace::Srgb);
        assert_eq!(pipe.working, ColorSpace::AcesCg);
        assert_eq!(pipe.output, ColorSpace::Srgb);
    }

    #[test]
    fn test_rec2020_matrix_properties() {
        let m = ColorSpace::Rec2020.rgb_to_xyz_matrix();
        // Row 1 should sum to ~1 (Y coefficients for luminance)
        let y_sum = m[1][0] + m[1][1] + m[1][2];
        assert!(
            (y_sum - 1.0).abs() < 0.01,
            "Rec.2020 luminance row should sum to ~1: {y_sum}"
        );
    }
}
