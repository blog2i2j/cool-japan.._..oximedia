//! Ambient light compensation — adjust display profiles based on ambient light.
//!
//! In environments with varying ambient illumination, a display that was
//! calibrated under D65 lighting may look incorrect under warm tungsten or
//! cool overcast daylight. This module measures (or estimates) the ambient
//! light and derives a compensation matrix that partially adapts the display
//! profile to the viewing environment.
//!
//! # Algorithm
//!
//! 1. Measure ambient illumination: correlated colour temperature (CCT) in
//!    Kelvin and illuminance in lux.
//! 2. Compute a chromatic adaptation transform (Bradford or Von Kries) from
//!    the calibration reference white to the ambient white point.
//! 3. Blend the full adaptation matrix with the identity by the user's
//!    `adaptation_strength` setting (0 = no change, 1 = full adaptation).
//! 4. Optionally clamp the output to valid display gamut.

use crate::error::CalibrationResult;
use crate::{CalibrationError, Matrix3x3, Rgb, Xyz};
use serde::{Deserialize, Serialize};

// ─── Ambient measurement ──────────────────────────────────────────────────────

/// A measurement of the viewing environment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmbientMeasurement {
    /// Correlated colour temperature of the ambient light in Kelvin.
    pub cct_kelvin: f64,
    /// Ambient illuminance in lux.
    pub illuminance_lux: f64,
    /// XYZ tristimulus of the ambient white point (normalised so Y=1).
    pub white_point_xyz: Xyz,
}

impl AmbientMeasurement {
    /// Create a measurement from CCT and illuminance, estimating the XYZ
    /// white point from the Planckian locus.
    ///
    /// # Errors
    ///
    /// Returns `CalibrationError::InvalidMeasurement` if the CCT is outside
    /// the valid range \[1000 K, 25000 K\].
    pub fn from_cct(cct_kelvin: f64, illuminance_lux: f64) -> CalibrationResult<Self> {
        if !(1000.0..=25_000.0).contains(&cct_kelvin) {
            return Err(CalibrationError::InvalidMeasurement(format!(
                "CCT {cct_kelvin} K is outside valid range [1000, 25000]"
            )));
        }
        let white_point_xyz = planckian_white_point(cct_kelvin);
        Ok(Self {
            cct_kelvin,
            illuminance_lux,
            white_point_xyz,
        })
    }

    /// Construct from a known XYZ white point directly (e.g. from a
    /// tristimulus colorimeter).
    ///
    /// # Errors
    ///
    /// Returns `CalibrationError::InvalidMeasurement` if `white_point_xyz[1]`
    /// (Y) is zero or negative.
    pub fn from_xyz(
        white_point_xyz: Xyz,
        illuminance_lux: f64,
    ) -> CalibrationResult<Self> {
        if white_point_xyz[1] <= 0.0 {
            return Err(CalibrationError::InvalidMeasurement(
                "White point Y must be positive".to_string(),
            ));
        }
        let y = white_point_xyz[1];
        let normalised = [
            white_point_xyz[0] / y,
            1.0,
            white_point_xyz[2] / y,
        ];
        // Estimate CCT via Robertson's method (approximate).
        let cct_kelvin = xyz_to_cct_robertson(normalised);
        Ok(Self {
            cct_kelvin,
            illuminance_lux,
            white_point_xyz: normalised,
        })
    }
}

// ─── Configuration ───────────────────────────────────────────────────────────

/// Chromatic adaptation method for ambient compensation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AmbientAdaptMethod {
    /// Bradford chromatic adaptation (industry standard).
    Bradford,
    /// Simple Von Kries diagonal adaptation (faster, less accurate).
    VonKries,
    /// XYZ scaling (simple but inaccurate for large shifts).
    XyzScaling,
}

/// Configuration for the ambient compensation algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmbientCompensationConfig {
    /// Reference white point of the display calibration (usually D65).
    pub reference_white: Xyz,
    /// Adaptation method.
    pub method: AmbientAdaptMethod,
    /// Blend factor: 0.0 = no compensation, 1.0 = full chromatic adaptation.
    pub adaptation_strength: f64,
    /// Minimum illuminance (lux) below which compensation is disabled.
    pub min_illuminance_lux: f64,
    /// Whether to clamp the resulting RGB values to \[0, 1\].
    pub clamp_output: bool,
}

impl Default for AmbientCompensationConfig {
    fn default() -> Self {
        // D65 reference white, Y=1 normalised
        Self {
            reference_white: [0.950_47, 1.0, 1.088_83],
            method: AmbientAdaptMethod::Bradford,
            adaptation_strength: 0.5,
            min_illuminance_lux: 5.0,
            clamp_output: true,
        }
    }
}

// ─── Compensation result ─────────────────────────────────────────────────────

/// The result of computing an ambient compensation adjustment.
#[derive(Debug, Clone)]
pub struct AmbientCompensationResult {
    /// 3×3 XYZ adaptation matrix to apply after display ICC transform.
    pub adaptation_matrix: Matrix3x3,
    /// Effective CCT shift applied (destination − source in Kelvin).
    pub cct_shift_k: f64,
    /// Whether compensation was applied (false if illuminance was below
    /// the configured threshold).
    pub compensation_applied: bool,
}

// ─── Main engine ─────────────────────────────────────────────────────────────

/// Computes ambient light compensation matrices for display profiles.
///
/// # Example
///
/// ```rust
/// use oximedia_calibrate::ambient_compensation::{
///     AmbientCompensationEngine, AmbientCompensationConfig, AmbientMeasurement,
/// };
///
/// let engine = AmbientCompensationEngine::new(AmbientCompensationConfig::default());
/// let measurement = AmbientMeasurement::from_cct(3200.0, 150.0).unwrap();
/// let result = engine.compute(&measurement).unwrap();
/// assert!(result.compensation_applied);
/// ```
pub struct AmbientCompensationEngine {
    config: AmbientCompensationConfig,
}

impl AmbientCompensationEngine {
    /// Create a new engine with the given configuration.
    #[must_use]
    pub fn new(config: AmbientCompensationConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration (D65 reference, Bradford, 50% strength).
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(AmbientCompensationConfig::default())
    }

    /// Compute the ambient compensation for a given measurement.
    ///
    /// # Errors
    ///
    /// Returns `CalibrationError::NumericalInstability` if the chromatic
    /// adaptation matrix is singular or produces out-of-range values.
    pub fn compute(
        &self,
        measurement: &AmbientMeasurement,
    ) -> CalibrationResult<AmbientCompensationResult> {
        // Disable if illuminance below threshold
        if measurement.illuminance_lux < self.config.min_illuminance_lux {
            return Ok(AmbientCompensationResult {
                adaptation_matrix: identity_matrix(),
                cct_shift_k: 0.0,
                compensation_applied: false,
            });
        }

        let src = self.config.reference_white;
        let dst = measurement.white_point_xyz;

        let full_matrix = match self.config.method {
            AmbientAdaptMethod::Bradford => bradford_cat(src, dst)?,
            AmbientAdaptMethod::VonKries => von_kries_cat(src, dst)?,
            AmbientAdaptMethod::XyzScaling => xyz_scaling_cat(src, dst),
        };

        // Blend identity with the full matrix by adaptation_strength
        let strength = self.config.adaptation_strength.clamp(0.0, 1.0);
        let identity = identity_matrix();
        let blended = blend_matrices(identity, full_matrix, strength);

        // Use a simple approximation: map reference white CCT from xy chromaticity
        let ref_cct = xyz_to_cct_robertson(src);
        let cct_shift = measurement.cct_kelvin - ref_cct;

        Ok(AmbientCompensationResult {
            adaptation_matrix: blended,
            cct_shift_k: cct_shift,
            compensation_applied: true,
        })
    }

    /// Apply the compensation matrix to a linear-light RGB triplet.
    ///
    /// The input is assumed to be in the display's native RGB space
    /// (not XYZ). The matrix is applied as-is; for a proper pipeline
    /// you should convert to XYZ, apply, then convert back.
    #[must_use]
    pub fn apply_to_rgb(&self, rgb: Rgb, matrix: &Matrix3x3) -> Rgb {
        let r = matrix[0][0] * rgb[0] + matrix[0][1] * rgb[1] + matrix[0][2] * rgb[2];
        let g = matrix[1][0] * rgb[0] + matrix[1][1] * rgb[1] + matrix[1][2] * rgb[2];
        let b = matrix[2][0] * rgb[0] + matrix[2][1] * rgb[1] + matrix[2][2] * rgb[2];
        if self.config.clamp_output {
            [r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0)]
        } else {
            [r, g, b]
        }
    }
}

// ─── Math helpers ─────────────────────────────────────────────────────────────

fn identity_matrix() -> Matrix3x3 {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

fn blend_matrices(a: Matrix3x3, b: Matrix3x3, t: f64) -> Matrix3x3 {
    let mut out = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = a[i][j] * (1.0 - t) + b[i][j] * t;
        }
    }
    out
}

/// Bradford chromatic adaptation from `src_white` to `dst_white`.
fn bradford_cat(src: Xyz, dst: Xyz) -> CalibrationResult<Matrix3x3> {
    // Bradford matrix M and its inverse M_inv (published constants)
    #[rustfmt::skip]
    let m: Matrix3x3 = [
        [ 0.8951,  0.2664, -0.1614],
        [-0.7502,  1.7135,  0.0367],
        [ 0.0389, -0.0685,  1.0296],
    ];
    #[rustfmt::skip]
    let m_inv: Matrix3x3 = [
        [ 0.986_993, -0.147_054,  0.159_828],
        [ 0.432_305,  0.518_360,  0.049_291],
        [-0.008_529,  0.040_043,  0.968_487],
    ];

    let src_cone = mat3x3_mul_vec(m, src);
    let dst_cone = mat3x3_mul_vec(m, dst);

    if src_cone[0].abs() < 1e-12 || src_cone[1].abs() < 1e-12 || src_cone[2].abs() < 1e-12 {
        return Err(CalibrationError::NumericalInstability(
            "Bradford source cone response near zero".to_string(),
        ));
    }

    // Diagonal gain matrix
    let gain: [f64; 3] = [
        dst_cone[0] / src_cone[0],
        dst_cone[1] / src_cone[1],
        dst_cone[2] / src_cone[2],
    ];
    let gain_mat: Matrix3x3 = [
        [gain[0], 0.0, 0.0],
        [0.0, gain[1], 0.0],
        [0.0, 0.0, gain[2]],
    ];

    // M_inv * Gain * M
    let gm = mat3x3_mul(gain_mat, m);
    Ok(mat3x3_mul(m_inv, gm))
}

/// Simple Von Kries diagonal adaptation.
fn von_kries_cat(src: Xyz, dst: Xyz) -> CalibrationResult<Matrix3x3> {
    if src[0].abs() < 1e-12 || src[1].abs() < 1e-12 || src[2].abs() < 1e-12 {
        return Err(CalibrationError::NumericalInstability(
            "Von Kries source near zero".to_string(),
        ));
    }
    Ok([
        [dst[0] / src[0], 0.0, 0.0],
        [0.0, dst[1] / src[1], 0.0],
        [0.0, 0.0, dst[2] / src[2]],
    ])
}

/// Simple XYZ scaling (poorest adaptation quality, but always valid).
fn xyz_scaling_cat(src: Xyz, dst: Xyz) -> Matrix3x3 {
    let sx = if src[0].abs() > 1e-12 { dst[0] / src[0] } else { 1.0 };
    let sy = if src[1].abs() > 1e-12 { dst[1] / src[1] } else { 1.0 };
    let sz = if src[2].abs() > 1e-12 { dst[2] / src[2] } else { 1.0 };
    [[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, sz]]
}

fn mat3x3_mul(a: Matrix3x3, b: Matrix3x3) -> Matrix3x3 {
    let mut c = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    c
}

fn mat3x3_mul_vec(m: Matrix3x3, v: Xyz) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// Estimate CCT from XYZ white point using Robertson's reciprocal megakelvin
/// tables (simplified to a polynomial approximation).
fn xyz_to_cct_robertson(xyz: Xyz) -> f64 {
    // Convert to chromaticity
    let sum = xyz[0] + xyz[1] + xyz[2];
    if sum < 1e-12 {
        return 6500.0; // default
    }
    let x = xyz[0] / sum;
    let y = xyz[1] / sum;
    // Approximate CCT from xy (McCamy's formula)
    let n = (x - 0.3320) / (y - 0.1858);
    let cct = -449.0 * n * n * n + 3525.0 * n * n - 6823.3 * n + 5520.33;
    cct.max(1000.0).min(25_000.0)
}

/// Compute approximate D-illuminant XYZ white point from CCT using a
/// polynomial fit to the CIE Planckian locus.
fn planckian_white_point(cct: f64) -> Xyz {
    // CIE Planckian locus approximation (Kang et al., 2002)
    let (xc, yc) = if cct <= 4000.0 {
        let xc = -0.266_0 * (1e9 / cct.powi(3))
            + 0.234_7 * (1e6 / cct.powi(2))
            + 0.870_1 * (1e3 / cct)
            + 0.178_6;
        let yc = -3.000_0 * xc * xc + 2.870_0 * xc - 0.275_0;
        (xc, yc)
    } else {
        let xc = -3.025_0 * (1e9 / cct.powi(3))
            + 2.107_8 * (1e6 / cct.powi(2))
            + 0.222_5 * (1e3 / cct)
            + 0.240_5;
        let yc = -3.000_0 * xc * xc + 2.870_0 * xc - 0.275_0;
        (xc, yc)
    };
    let y_norm = 1.0;
    let x_norm = xc * y_norm / yc.max(1e-12);
    let z_norm = (1.0 - xc - yc) * y_norm / yc.max(1e-12);
    [x_norm, y_norm, z_norm]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ambient_measurement_from_cct_valid() {
        let m = AmbientMeasurement::from_cct(6500.0, 200.0);
        assert!(m.is_ok());
        let m = m.unwrap();
        assert!((m.cct_kelvin - 6500.0).abs() < f64::EPSILON);
        assert!((m.white_point_xyz[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_ambient_measurement_from_cct_invalid() {
        assert!(AmbientMeasurement::from_cct(100.0, 50.0).is_err());
        assert!(AmbientMeasurement::from_cct(30_000.0, 50.0).is_err());
    }

    #[test]
    fn test_ambient_measurement_from_xyz() {
        let xyz = [0.9505, 1.0, 1.0888]; // approx D65
        let m = AmbientMeasurement::from_xyz(xyz, 100.0);
        assert!(m.is_ok());
    }

    #[test]
    fn test_ambient_measurement_from_xyz_invalid_y() {
        let xyz = [0.5, 0.0, 0.5];
        assert!(AmbientMeasurement::from_xyz(xyz, 100.0).is_err());
    }

    #[test]
    fn test_compute_no_compensation_low_light() {
        let engine = AmbientCompensationEngine::with_defaults();
        let m = AmbientMeasurement::from_cct(3200.0, 1.0).unwrap(); // below 5 lux
        let result = engine.compute(&m).unwrap();
        assert!(!result.compensation_applied);
        // Matrix should be identity
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((result.adaptation_matrix[i][j] - expected).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn test_compute_bradford_d65_to_d65_identity() {
        // Adapting from D65 to D65 should give (near-)identity
        let cfg = AmbientCompensationConfig {
            adaptation_strength: 1.0,
            ..AmbientCompensationConfig::default()
        };
        let engine = AmbientCompensationEngine::new(cfg);
        // Measurement with D65 white point (same as reference)
        let m = AmbientMeasurement {
            cct_kelvin: 6500.0,
            illuminance_lux: 200.0,
            white_point_xyz: [0.950_47, 1.0, 1.088_83],
        };
        let result = engine.compute(&m).unwrap();
        assert!(result.compensation_applied);
        // Diagonal should be near 1, off-diagonal near 0
        for i in 0..3 {
            assert!((result.adaptation_matrix[i][i] - 1.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_compute_von_kries() {
        let cfg = AmbientCompensationConfig {
            method: AmbientAdaptMethod::VonKries,
            adaptation_strength: 1.0,
            ..AmbientCompensationConfig::default()
        };
        let engine = AmbientCompensationEngine::new(cfg);
        let m = AmbientMeasurement::from_cct(3200.0, 200.0).unwrap();
        let result = engine.compute(&m);
        assert!(result.is_ok());
        assert!(result.unwrap().compensation_applied);
    }

    #[test]
    fn test_compute_xyz_scaling() {
        let cfg = AmbientCompensationConfig {
            method: AmbientAdaptMethod::XyzScaling,
            adaptation_strength: 1.0,
            ..AmbientCompensationConfig::default()
        };
        let engine = AmbientCompensationEngine::new(cfg);
        let m = AmbientMeasurement::from_cct(4000.0, 100.0).unwrap();
        let result = engine.compute(&m);
        assert!(result.is_ok());
    }

    #[test]
    fn test_apply_to_rgb_no_clamping() {
        let cfg = AmbientCompensationConfig {
            clamp_output: false,
            ..AmbientCompensationConfig::default()
        };
        let engine = AmbientCompensationEngine::new(cfg);
        let identity = identity_matrix();
        let rgb = [0.5, 0.5, 0.5];
        let out = engine.apply_to_rgb(rgb, &identity);
        assert!((out[0] - 0.5).abs() < 1e-9);
        assert!((out[1] - 0.5).abs() < 1e-9);
        assert!((out[2] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_apply_to_rgb_clamped() {
        let engine = AmbientCompensationEngine::with_defaults();
        // Matrix that scales up — should clamp at 1.0
        let scale: Matrix3x3 = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]];
        let rgb = [0.8, 0.8, 0.8];
        let out = engine.apply_to_rgb(rgb, &scale);
        for v in &out {
            assert!(*v <= 1.0);
        }
    }

    #[test]
    fn test_adaptation_strength_zero_returns_identity() {
        let cfg = AmbientCompensationConfig {
            adaptation_strength: 0.0,
            ..AmbientCompensationConfig::default()
        };
        let engine = AmbientCompensationEngine::new(cfg);
        let m = AmbientMeasurement::from_cct(2700.0, 100.0).unwrap();
        let result = engine.compute(&m).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((result.adaptation_matrix[i][j] - expected).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn test_planckian_white_point_y_is_one() {
        for cct in [2000.0, 3200.0, 5000.0, 6500.0, 9000.0] {
            let wp = planckian_white_point(cct);
            assert!((wp[1] - 1.0).abs() < 1e-6, "Y should be 1 for cct={cct}");
        }
    }

    #[test]
    fn test_xyz_to_cct_robertson_d65_roundtrip() {
        let d65 = [0.950_47, 1.0, 1.088_83];
        let cct = xyz_to_cct_robertson(d65);
        // Should be approximately 6500 K
        assert!((cct - 6500.0).abs() < 300.0, "CCT should be near 6500 K, got {cct}");
    }
}
