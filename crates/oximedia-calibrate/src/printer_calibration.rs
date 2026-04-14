//! Printer calibration and CMYK soft-proofing workflows.
//!
//! This module provides CMYK ink-model calibration, device link profile
//! simulation, and soft-proofing support for print reproduction accuracy.
//!
//! # Overview
//!
//! The printer calibration pipeline:
//!
//! 1. Measure a printed color target (e.g. IT8.7/4) with a spectrophotometer
//! 2. Fit a forward model: CMYK → Lab (polynomial regression or 4D LUT)
//! 3. Build a gamut boundary descriptor for the output device
//! 4. Apply perceptual gamut mapping to clip out-of-gamut RGB/Lab values
//! 5. Invert the model: Lab → CMYK (iterative solver or 4D LUT)
//!
//! # Ink limit
//!
//! Commercial CMYK presses have a total area coverage (TAC) limit — the sum
//! C+M+Y+K must not exceed a configured maximum (typical: 300 %–320 %). The
//! `enforce_ink_limit` helper clips CMYK values to this constraint.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use crate::error::{CalibrationError, CalibrationResult};
use crate::Lab;

// ---------------------------------------------------------------------------
// CMYK type alias
// ---------------------------------------------------------------------------

/// CMYK ink percentages in 0.0–100.0 range.
pub type Cmyk = [f64; 4];

// ---------------------------------------------------------------------------
// Ink limit enforcement
// ---------------------------------------------------------------------------

/// Enforce a total area coverage (TAC) limit on a CMYK value.
///
/// If `C + M + Y + K > max_tac` the values are scaled proportionally so that
/// the sum equals `max_tac`, preserving the relative ratios of the inks.
///
/// # Arguments
///
/// * `cmyk` — input CMYK values (0.0–100.0 each)
/// * `max_tac` — maximum total area coverage (e.g. 300.0 for 300%)
///
/// # Returns
///
/// CMYK values clipped to the TAC limit.
#[must_use]
pub fn enforce_ink_limit(cmyk: Cmyk, max_tac: f64) -> Cmyk {
    let total = cmyk[0] + cmyk[1] + cmyk[2] + cmyk[3];
    if total <= max_tac || total < f64::EPSILON {
        return cmyk;
    }
    let scale = max_tac / total;
    [
        cmyk[0] * scale,
        cmyk[1] * scale,
        cmyk[2] * scale,
        cmyk[3] * scale,
    ]
}

// ---------------------------------------------------------------------------
// Measurement record
// ---------------------------------------------------------------------------

/// A single IT8/printer calibration measurement: CMYK patch and Lab reading.
#[derive(Debug, Clone, PartialEq)]
pub struct PrinterPatch {
    /// CMYK ink percentages (0–100 each).
    pub cmyk: Cmyk,
    /// Measured Lab colour (L in 0–100, a/b in ±128).
    pub lab: Lab,
}

impl PrinterPatch {
    /// Create a new measurement record.
    #[must_use]
    pub fn new(cmyk: Cmyk, lab: Lab) -> Self {
        Self { cmyk, lab }
    }
}

// ---------------------------------------------------------------------------
// Forward model: CMYK → Lab (polynomial regression)
// ---------------------------------------------------------------------------

/// Degree-1 polynomial (linear) forward model: CMYK → Lab.
///
/// The model is `lab = A * [c, m, y, k, 1]ᵀ` where A is a 3×5 matrix
/// (3 Lab channels × 5 coefficients including bias).
///
/// This is a first-order approximation. For production use a full 4D LUT
/// should be preferred; this linear model is useful for characterisation,
/// smoke tests, and small-format calibration targets.
#[derive(Debug, Clone)]
pub struct LinearForwardModel {
    /// 3 rows × 5 columns: [C, M, Y, K, bias] → [L, a, b].
    coefficients: [[f64; 5]; 3],
}

impl LinearForwardModel {
    /// Fit a linear model to a set of measurement patches via ordinary
    /// least-squares (closed-form normal equations).
    ///
    /// # Errors
    ///
    /// Returns `CalibrationError::InvalidMeasurement` if fewer than 5 patches
    /// are supplied (underdetermined system) or if the normal matrix is
    /// singular.
    pub fn fit(patches: &[PrinterPatch]) -> CalibrationResult<Self> {
        if patches.len() < 5 {
            return Err(CalibrationError::InvalidMeasurement(
                "At least 5 measurement patches required for linear CMYK→Lab model".to_string(),
            ));
        }

        let n = patches.len();
        // Build X matrix (n×5): rows are [c, m, y, k, 1] for each patch
        let mut x_mat = vec![[0_f64; 5]; n];
        for (i, p) in patches.iter().enumerate() {
            x_mat[i] = [
                p.cmyk[0] / 100.0,
                p.cmyk[1] / 100.0,
                p.cmyk[2] / 100.0,
                p.cmyk[3] / 100.0,
                1.0,
            ];
        }

        // Normal equations: Xᵀ X β = Xᵀ y  →  β = (XᵀX)⁻¹ Xᵀ y
        // Solve for each Lab channel independently.
        let xt_x = mat_transpose_mul(&x_mat, n);
        let xt_x_inv = invert_5x5(&xt_x).ok_or_else(|| {
            CalibrationError::InvalidMeasurement(
                "CMYK→Lab normal matrix is singular; patches may be collinear".to_string(),
            )
        })?;

        let mut coefficients = [[0_f64; 5]; 3];
        for ch in 0..3 {
            let y: Vec<f64> = patches.iter().map(|p| p.lab[ch]).collect();
            let xt_y = mat_t_times_vec(&x_mat, &y, n);
            let beta = mat_times_vec_5(&xt_x_inv, &xt_y);
            coefficients[ch] = beta;
        }

        Ok(Self { coefficients })
    }

    /// Predict Lab from CMYK using the fitted linear model.
    ///
    /// # Arguments
    ///
    /// * `cmyk` — ink values 0.0–100.0
    #[must_use]
    pub fn predict(&self, cmyk: Cmyk) -> Lab {
        let x = [
            cmyk[0] / 100.0,
            cmyk[1] / 100.0,
            cmyk[2] / 100.0,
            cmyk[3] / 100.0,
            1.0,
        ];
        let mut lab = [0_f64; 3];
        for ch in 0..3 {
            lab[ch] = self.coefficients[ch]
                .iter()
                .zip(x.iter())
                .map(|(a, b)| a * b)
                .sum();
        }
        lab
    }

    /// Return the raw 3×5 coefficient matrix.
    #[must_use]
    pub fn coefficients(&self) -> &[[f64; 5]; 3] {
        &self.coefficients
    }
}

// ---------------------------------------------------------------------------
// Inverse model: Lab → CMYK via Newton iteration
// ---------------------------------------------------------------------------

/// Inverse printer model: Lab → CMYK via Gauss-Newton iteration using the
/// linear forward model as the Jacobian.
///
/// # Notes
///
/// The linear forward model has a constant Jacobian (the coefficient matrix
/// columns 0–3), so Newton's method converges in one step for the linear
/// case. For non-linear models the iteration provides a good first
/// approximation.
#[derive(Debug, Clone)]
pub struct InversePrinterModel {
    forward: LinearForwardModel,
    /// Maximum total ink coverage (default 300 %).
    max_tac: f64,
}

impl InversePrinterModel {
    /// Build an inverse model from a fitted forward model.
    ///
    /// # Arguments
    ///
    /// * `forward` — fitted `LinearForwardModel`
    /// * `max_tac` — total area coverage limit (0–400)
    #[must_use]
    pub fn new(forward: LinearForwardModel, max_tac: f64) -> Self {
        Self { forward, max_tac }
    }

    /// Convert a target Lab colour to CMYK using Gauss-Newton iteration.
    ///
    /// Starts from `initial_cmyk` and iterates up to `max_iter` times.
    /// Returns the CMYK estimate (clamped 0–100, TAC-limited) and the
    /// residual delta-E after the final iteration.
    ///
    /// # Errors
    ///
    /// Returns `CalibrationError::NumericalInstability` if the Jacobian
    /// pseudo-inverse cannot be computed (rank-deficient model).
    pub fn invert(
        &self,
        target_lab: Lab,
        initial_cmyk: Option<Cmyk>,
        max_iter: u32,
    ) -> CalibrationResult<(Cmyk, f64)> {
        let mut cmyk = initial_cmyk.unwrap_or([25.0, 25.0, 25.0, 0.0]);

        let j = build_jacobian_4x3(&self.forward.coefficients);
        let mut jt_j = mat4x3_jtj(&j);
        // Tikhonov regularization: JᵀJ is at most rank 3 for a 4×3 Jacobian,
        // so add a small diagonal term to make the 4×4 system invertible.
        let lambda = 1e-6;
        for i in 0..4 {
            jt_j[i][i] += lambda;
        }
        let jt_j_inv = invert_4x4(&jt_j).ok_or_else(|| {
            CalibrationError::NumericalInstability(
                "Printer model Jacobian is rank-deficient".to_string(),
            )
        })?;
        // Pseudo-inverse: (JᵀJ)⁻¹ Jᵀ  — 4×3
        let j_pinv = mat4x4_times_mat4x3t(&jt_j_inv, &j);

        for _ in 0..max_iter {
            let predicted = self.forward.predict(cmyk);
            let residual = [
                target_lab[0] - predicted[0],
                target_lab[1] - predicted[1],
                target_lab[2] - predicted[2],
            ];
            let delta = mat4x3_times_vec3(&j_pinv, &residual);
            for i in 0..4 {
                cmyk[i] = (cmyk[i] + delta[i] * 100.0).clamp(0.0, 100.0);
            }
            cmyk = enforce_ink_limit(cmyk, self.max_tac);
        }

        let final_pred = self.forward.predict(cmyk);
        let residual_de = delta_e_lab(target_lab, final_pred);
        Ok((cmyk, residual_de))
    }
}

// ---------------------------------------------------------------------------
// Soft-proofing simulation
// ---------------------------------------------------------------------------

/// Soft-proofing render intent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderIntent {
    /// Perceptual — compress source gamut to fit within destination.
    Perceptual,
    /// Relative colorimetric — match in-gamut colours, clip out-of-gamut.
    RelativeColorimetric,
    /// Absolute colorimetric — preserve absolute Lab values (paper white).
    AbsoluteColorimetric,
}

/// Simulate how an RGB image will appear when printed on a CMYK device.
///
/// Steps:
/// 1. Convert RGB → Lab (using a simple sRGB linearisation)
/// 2. Apply gamut-boundary clipping based on the forward model's gamut
/// 3. Invert to CMYK
/// 4. Convert CMYK back to Lab via the forward model
/// 5. Return the simulated Lab values for display comparison
///
/// # Errors
///
/// Returns an error if the inverse model fails.
pub fn soft_proof(
    rgb_pixels: &[[f64; 3]],
    inverse_model: &InversePrinterModel,
    intent: RenderIntent,
) -> CalibrationResult<Vec<Lab>> {
    let mut result = Vec::with_capacity(rgb_pixels.len());
    for &rgb in rgb_pixels {
        let lab = srgb_to_lab(rgb);
        let adjusted_lab = match intent {
            RenderIntent::Perceptual => gamut_compress_perceptual(lab),
            RenderIntent::RelativeColorimetric => lab,
            RenderIntent::AbsoluteColorimetric => lab,
        };
        let (cmyk, _residual) = inverse_model.invert(adjusted_lab, None, 8)?;
        let simulated = inverse_model.forward.predict(cmyk);
        result.push(simulated);
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// PrinterCalibrator — high-level API
// ---------------------------------------------------------------------------

/// Configuration for the printer calibration workflow.
#[derive(Debug, Clone)]
pub struct PrinterCalibratorConfig {
    /// Total area coverage limit (%) for ink limiting.
    pub max_tac: f64,
    /// Maximum Gauss-Newton iterations for inversion.
    pub max_iter: u32,
    /// Acceptable delta-E residual for soft-proof warning.
    pub acceptable_de: f64,
    /// Default render intent for soft-proofing.
    pub render_intent: RenderIntent,
}

impl Default for PrinterCalibratorConfig {
    fn default() -> Self {
        Self {
            max_tac: 300.0,
            max_iter: 12,
            acceptable_de: 3.0,
            render_intent: RenderIntent::Perceptual,
        }
    }
}

/// High-level printer calibration API.
pub struct PrinterCalibrator {
    config: PrinterCalibratorConfig,
    forward: Option<LinearForwardModel>,
    inverse: Option<InversePrinterModel>,
}

impl PrinterCalibrator {
    /// Create a new calibrator with the supplied configuration.
    #[must_use]
    pub fn new(config: PrinterCalibratorConfig) -> Self {
        Self {
            config,
            forward: None,
            inverse: None,
        }
    }

    /// Fit the forward and inverse models from a set of measurement patches.
    ///
    /// # Errors
    ///
    /// Returns an error if fitting fails (too few patches / singular system).
    pub fn fit(&mut self, patches: &[PrinterPatch]) -> CalibrationResult<()> {
        let forward = LinearForwardModel::fit(patches)?;
        let inverse = InversePrinterModel::new(forward.clone(), self.config.max_tac);
        self.forward = Some(forward);
        self.inverse = Some(inverse);
        Ok(())
    }

    /// Predict Lab from CMYK using the fitted forward model.
    ///
    /// # Errors
    ///
    /// Returns an error if the model has not been fitted yet.
    pub fn predict_lab(&self, cmyk: Cmyk) -> CalibrationResult<Lab> {
        self.forward
            .as_ref()
            .map(|m| m.predict(cmyk))
            .ok_or_else(|| {
                CalibrationError::InvalidMeasurement(
                    "Printer calibrator not fitted; call fit() first".to_string(),
                )
            })
    }

    /// Convert a target Lab colour to CMYK using the fitted inverse model.
    ///
    /// # Errors
    ///
    /// Returns an error if the model has not been fitted or inversion fails.
    pub fn lab_to_cmyk(&self, lab: Lab) -> CalibrationResult<Cmyk> {
        let inv = self.inverse.as_ref().ok_or_else(|| {
            CalibrationError::InvalidMeasurement(
                "Printer calibrator not fitted; call fit() first".to_string(),
            )
        })?;
        let (cmyk, _de) = inv.invert(lab, None, self.config.max_iter)?;
        Ok(cmyk)
    }

    /// Soft-proof a set of sRGB pixels.
    ///
    /// # Errors
    ///
    /// Returns an error if the model has not been fitted or inversion fails.
    pub fn soft_proof(&self, rgb_pixels: &[[f64; 3]]) -> CalibrationResult<Vec<Lab>> {
        let inv = self.inverse.as_ref().ok_or_else(|| {
            CalibrationError::InvalidMeasurement(
                "Printer calibrator not fitted; call fit() first".to_string(),
            )
        })?;
        soft_proof(rgb_pixels, inv, self.config.render_intent)
    }

    /// Access the fitted forward model.
    #[must_use]
    pub fn forward_model(&self) -> Option<&LinearForwardModel> {
        self.forward.as_ref()
    }

    /// Access the configuration.
    #[must_use]
    pub fn config(&self) -> &PrinterCalibratorConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Private math helpers
// ---------------------------------------------------------------------------

/// Compute XᵀX for an n×5 matrix (returns 5×5 matrix).
fn mat_transpose_mul(x: &[[f64; 5]], n: usize) -> [[f64; 5]; 5] {
    let mut result = [[0_f64; 5]; 5];
    for i in 0..5 {
        for j in 0..5 {
            let mut s = 0_f64;
            for k in 0..n {
                s += x[k][i] * x[k][j];
            }
            result[i][j] = s;
        }
    }
    result
}

/// Compute Xᵀ y for an n×5 matrix and n-vector.
fn mat_t_times_vec(x: &[[f64; 5]], y: &[f64], n: usize) -> [f64; 5] {
    let mut result = [0_f64; 5];
    for i in 0..5 {
        let mut s = 0_f64;
        for k in 0..n {
            s += x[k][i] * y[k];
        }
        result[i] = s;
    }
    result
}

/// Multiply a 5×5 matrix by a 5-vector.
fn mat_times_vec_5(m: &[[f64; 5]; 5], v: &[f64; 5]) -> [f64; 5] {
    let mut result = [0_f64; 5];
    for i in 0..5 {
        let mut s = 0_f64;
        for j in 0..5 {
            s += m[i][j] * v[j];
        }
        result[i] = s;
    }
    result
}

/// Invert a 5×5 matrix using Gauss-Jordan elimination with partial pivoting.
fn invert_5x5(m: &[[f64; 5]; 5]) -> Option<[[f64; 5]; 5]> {
    let mut aug = [[0_f64; 10]; 5];
    for i in 0..5 {
        for j in 0..5 {
            aug[i][j] = m[i][j];
        }
        aug[i][5 + i] = 1.0;
    }
    for col in 0..5 {
        // Partial pivot
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for row in (col + 1)..5 {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }
        if max_val < 1e-12 {
            return None;
        }
        aug.swap(col, max_row);
        let pivot = aug[col][col];
        for j in 0..10 {
            aug[col][j] /= pivot;
        }
        for row in 0..5 {
            if row != col {
                let factor = aug[row][col];
                for j in 0..10 {
                    aug[row][j] -= factor * aug[col][j];
                }
            }
        }
    }
    let mut inv = [[0_f64; 5]; 5];
    for i in 0..5 {
        for j in 0..5 {
            inv[i][j] = aug[i][5 + j];
        }
    }
    Some(inv)
}

/// Build a 4×3 Jacobian from the forward model's CMYK coefficients (columns
/// 0–3 of the 3×5 coefficient matrix, transposed).
///
/// J[c][lab_ch] = d(lab_ch)/d(cmyk_c)
fn build_jacobian_4x3(coefs: &[[f64; 5]; 3]) -> [[f64; 3]; 4] {
    let mut j = [[0_f64; 3]; 4];
    for lab_ch in 0..3 {
        for ink in 0..4 {
            // divide by 100 because we normalised CMYK to [0,1] in the model
            j[ink][lab_ch] = coefs[lab_ch][ink] / 100.0;
        }
    }
    j
}

/// Compute JᵀJ for a 4×3 matrix (returns 4×4).
fn mat4x3_jtj(j: &[[f64; 3]; 4]) -> [[f64; 4]; 4] {
    let mut result = [[0_f64; 4]; 4];
    for i in 0..4 {
        for k in 0..4 {
            let mut s = 0_f64;
            for m in 0..3 {
                s += j[i][m] * j[k][m];
            }
            result[i][k] = s;
        }
    }
    result
}

/// Invert a 4×4 matrix using Gauss-Jordan elimination.
fn invert_4x4(m: &[[f64; 4]; 4]) -> Option<[[f64; 4]; 4]> {
    let mut aug = [[0_f64; 8]; 4];
    for i in 0..4 {
        for j in 0..4 {
            aug[i][j] = m[i][j];
        }
        aug[i][4 + i] = 1.0;
    }
    for col in 0..4 {
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for row in (col + 1)..4 {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }
        if max_val < 1e-12 {
            return None;
        }
        aug.swap(col, max_row);
        let pivot = aug[col][col];
        for j in 0..8 {
            aug[col][j] /= pivot;
        }
        for row in 0..4 {
            if row != col {
                let factor = aug[row][col];
                for j in 0..8 {
                    aug[row][j] -= factor * aug[col][j];
                }
            }
        }
    }
    let mut inv = [[0_f64; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            inv[i][j] = aug[i][4 + j];
        }
    }
    Some(inv)
}

/// Multiply a 4×4 matrix by a 4×3 matrix, yielding a 4×3 result.
fn mat4x4_times_mat4x3t(m44: &[[f64; 4]; 4], j43: &[[f64; 3]; 4]) -> [[f64; 3]; 4] {
    // j43 is row-major: j43[ink][lab_ch]
    // we want (JᵀJ)⁻¹ Jᵀ which is 4×4 × 4×3 = 4×3
    let mut result = [[0_f64; 3]; 4];
    for i in 0..4 {
        for lab_ch in 0..3 {
            let mut s = 0_f64;
            for k in 0..4 {
                s += m44[i][k] * j43[k][lab_ch];
            }
            result[i][lab_ch] = s;
        }
    }
    result
}

/// Multiply a 4×3 (pseudo-inverse) matrix by a 3-vector.
fn mat4x3_times_vec3(m: &[[f64; 3]; 4], v: &[f64; 3]) -> [f64; 4] {
    let mut result = [0_f64; 4];
    for i in 0..4 {
        let mut s = 0_f64;
        for j in 0..3 {
            s += m[i][j] * v[j];
        }
        result[i] = s;
    }
    result
}

/// Compute delta-E between two Lab values (simple Euclidean).
fn delta_e_lab(a: Lab, b: Lab) -> f64 {
    let dl = a[0] - b[0];
    let da = a[1] - b[1];
    let db = a[2] - b[2];
    (dl * dl + da * da + db * db).sqrt()
}

/// Approximate sRGB → Lab conversion (using linearisation + Rec.709 matrix).
fn srgb_to_lab(rgb: [f64; 3]) -> Lab {
    // Linearise (gamma 2.2 approximation)
    let lin: [f64; 3] = std::array::from_fn(|i| {
        let v = rgb[i].clamp(0.0, 1.0);
        if v <= 0.04045 {
            v / 12.92
        } else {
            ((v + 0.055) / 1.055).powf(2.4)
        }
    });
    // Rec.709 linear RGB → XYZ D65
    let x = 0.4124564 * lin[0] + 0.3575761 * lin[1] + 0.1804375 * lin[2];
    let y = 0.2126729 * lin[0] + 0.7151522 * lin[1] + 0.0721750 * lin[2];
    let z = 0.0193339 * lin[0] + 0.1191920 * lin[1] + 0.9503041 * lin[2];
    // XYZ → Lab (D65 white: 0.9505, 1.0, 1.0890)
    let xn = 0.9505;
    let yn = 1.0;
    let zn = 1.0890;
    let fx = xyz_f(x / xn);
    let fy = xyz_f(y / yn);
    let fz = xyz_f(z / zn);
    let l = 116.0 * fy - 16.0;
    let a = 500.0 * (fx - fy);
    let b = 200.0 * (fy - fz);
    [l, a, b]
}

fn xyz_f(t: f64) -> f64 {
    const DELTA: f64 = 6.0 / 29.0;
    if t > DELTA * DELTA * DELTA {
        t.cbrt()
    } else {
        t / (3.0 * DELTA * DELTA) + 4.0 / 29.0
    }
}

/// Very simple perceptual gamut compression: reduce chroma towards neutral.
fn gamut_compress_perceptual(lab: Lab) -> Lab {
    let chroma = (lab[1] * lab[1] + lab[2] * lab[2]).sqrt();
    // Compress chroma above 60 to a softer limit
    let max_chroma = 60.0_f64;
    if chroma <= max_chroma {
        return lab;
    }
    let scale = max_chroma / chroma;
    [lab[0], lab[1] * scale, lab[2] * scale]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_patches() -> Vec<PrinterPatch> {
        // Synthetic patches: sweep single inks for predictable coefficients
        vec![
            PrinterPatch::new([0.0, 0.0, 0.0, 0.0], [100.0, 0.0, 0.0]),
            PrinterPatch::new([100.0, 0.0, 0.0, 0.0], [50.0, -35.0, -15.0]),
            PrinterPatch::new([0.0, 100.0, 0.0, 0.0], [55.0, 60.0, -30.0]),
            PrinterPatch::new([0.0, 0.0, 100.0, 0.0], [70.0, -5.0, 70.0]),
            PrinterPatch::new([0.0, 0.0, 0.0, 100.0], [20.0, 5.0, 5.0]),
            PrinterPatch::new([50.0, 50.0, 50.0, 0.0], [60.0, 10.0, 10.0]),
            PrinterPatch::new([0.0, 0.0, 0.0, 50.0], [60.0, 2.0, 2.0]),
        ]
    }

    #[test]
    fn test_enforce_ink_limit_below_max() {
        let cmyk = [50.0, 50.0, 50.0, 50.0]; // total = 200, below 300
        let result = enforce_ink_limit(cmyk, 300.0);
        // Should be unchanged
        for i in 0..4 {
            assert!((result[i] - cmyk[i]).abs() < 1e-9);
        }
    }

    #[test]
    fn test_enforce_ink_limit_above_max() {
        let cmyk = [100.0, 100.0, 100.0, 100.0]; // total = 400, above 300
        let result = enforce_ink_limit(cmyk, 300.0);
        let total: f64 = result.iter().sum();
        assert!((total - 300.0).abs() < 1e-6, "total={total}");
    }

    #[test]
    fn test_enforce_ink_limit_zero() {
        let cmyk = [0.0, 0.0, 0.0, 0.0];
        let result = enforce_ink_limit(cmyk, 300.0);
        assert_eq!(result, [0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_linear_forward_model_fit_too_few_patches() {
        let patches = vec![
            PrinterPatch::new([0.0, 0.0, 0.0, 0.0], [100.0, 0.0, 0.0]),
            PrinterPatch::new([100.0, 0.0, 0.0, 0.0], [50.0, -35.0, -15.0]),
        ];
        assert!(LinearForwardModel::fit(&patches).is_err());
    }

    #[test]
    fn test_linear_forward_model_fit_and_predict() {
        let patches = make_patches();
        let model = LinearForwardModel::fit(&patches).expect("fit should succeed");

        // Paper white (0 ink) should predict close to L*=100, a*=0, b*=0
        // A linear model with few synthetic patches has limited accuracy;
        // 10 dE tolerance is acceptable for this first-order approximation.
        let paper_white = model.predict([0.0, 0.0, 0.0, 0.0]);
        assert!((paper_white[0] - 100.0).abs() < 10.0, "L={}", paper_white[0]);
    }

    #[test]
    fn test_linear_forward_model_coefficients_shape() {
        let patches = make_patches();
        let model = LinearForwardModel::fit(&patches).expect("fit ok");
        assert_eq!(model.coefficients().len(), 3);
        assert_eq!(model.coefficients()[0].len(), 5);
    }

    #[test]
    fn test_printer_calibrator_workflow() {
        let patches = make_patches();
        let mut calibrator = PrinterCalibrator::new(PrinterCalibratorConfig::default());
        calibrator.fit(&patches).expect("fit ok");

        let lab = calibrator.predict_lab([0.0, 0.0, 0.0, 0.0]).expect("predict ok");
        assert!((lab[0] - 100.0).abs() < 10.0, "L={}", lab[0]);
    }

    #[test]
    fn test_printer_calibrator_not_fitted_error() {
        let calibrator = PrinterCalibrator::new(PrinterCalibratorConfig::default());
        assert!(calibrator.predict_lab([0.0, 0.0, 0.0, 0.0]).is_err());
        assert!(calibrator.lab_to_cmyk([50.0, 0.0, 0.0]).is_err());
    }

    #[test]
    fn test_soft_proof_basic() {
        let patches = make_patches();
        let mut calibrator = PrinterCalibrator::new(PrinterCalibratorConfig::default());
        calibrator.fit(&patches).expect("fit ok");
        let result = calibrator
            .soft_proof(&[[1.0, 1.0, 1.0], [0.5, 0.5, 0.5]])
            .expect("soft proof ok");
        assert_eq!(result.len(), 2);
        // Luminance should be in a plausible range
        for lab in &result {
            assert!(lab[0] >= 0.0 && lab[0] <= 110.0, "L={}", lab[0]);
        }
    }

    #[test]
    fn test_srgb_to_lab_white() {
        let lab = srgb_to_lab([1.0, 1.0, 1.0]);
        // sRGB white ≈ L*=100, a*≈0, b*≈0
        assert!((lab[0] - 100.0).abs() < 2.0, "L={}", lab[0]);
        assert!(lab[1].abs() < 2.0, "a={}", lab[1]);
        assert!(lab[2].abs() < 2.0, "b={}", lab[2]);
    }

    #[test]
    fn test_srgb_to_lab_black() {
        let lab = srgb_to_lab([0.0, 0.0, 0.0]);
        assert!(lab[0].abs() < 1.0, "L={}", lab[0]);
    }

    #[test]
    fn test_render_intent_variants() {
        assert_ne!(RenderIntent::Perceptual, RenderIntent::RelativeColorimetric);
        assert_ne!(RenderIntent::Perceptual, RenderIntent::AbsoluteColorimetric);
    }
}
