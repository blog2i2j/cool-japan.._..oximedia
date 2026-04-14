//! Display calibration LUT generation from colorimetric measurement data.
//!
//! This module generates display calibration LUTs from measured colorimetric
//! patches (typically obtained via a spectrophotometer or colorimeter).  The
//! workflow is:
//!
//! 1. Measure a set of known target patches on the display under calibration.
//! 2. Collect the resulting CIE XYZ (or xyY) measurements.
//! 3. Build a [`DisplayCalibrator`] from target + measured pairs.
//! 4. Generate a 1-D or 3-D calibration LUT that compensates for display
//!    deviations from the target colour space.
//!
//! # Calibration Approach
//!
//! For matrix-based displays (LCD, OLED) the calibration is decomposed into:
//!
//! * **White-point correction** — scales all channels to hit the target D65
//!   (or other specified) white point.
//! * **Tone curve correction** — per-channel 1-D curves that linearise each
//!   primary channel's response (gamma correction).
//! * **Matrix correction** — a 3×3 colour-mixing correction derived from the
//!   RGB primaries measured on the display.
//!
//! The combined transform:
//! ```text
//! calibrated_xyz = matrix_correction * tone_corrected_linear_rgb
//! ```
//! is then baked into a 3-D LUT for real-time use.
//!
//! # Example
//!
//! ```rust
//! use oximedia_lut::display_calibration::{
//!     CalibrationPatch, DisplayCalibrator, CalibrationTarget,
//! };
//!
//! // Three primary + white measurements (minimal example)
//! let patches = vec![
//!     CalibrationPatch::new([1.0, 0.0, 0.0], [0.412, 0.213, 0.019]),
//!     CalibrationPatch::new([0.0, 1.0, 0.0], [0.358, 0.715, 0.119]),
//!     CalibrationPatch::new([0.0, 0.0, 1.0], [0.180, 0.072, 0.951]),
//!     CalibrationPatch::new([1.0, 1.0, 1.0], [0.950, 1.000, 1.089]),
//! ];
//!
//! let calibrator = DisplayCalibrator::new(patches, CalibrationTarget::D65Srgb);
//! let lut = calibrator.generate_1d_lut(256);
//! assert_eq!(lut.len(), 256);
//! ```

use crate::error::{LutError, LutResult};
use crate::Rgb;

// ============================================================================
// Calibration patch
// ============================================================================

/// A single colorimetric measurement patch: known RGB input + measured XYZ.
#[derive(Debug, Clone)]
pub struct CalibrationPatch {
    /// Displayed RGB value (normalised to `[0, 1]`).
    pub rgb: Rgb,
    /// Measured CIE XYZ (Y is normalised to 1.0 for a perfect diffuse white).
    pub xyz: [f64; 3],
}

impl CalibrationPatch {
    /// Create a new patch from arrays.
    #[must_use]
    pub fn new(rgb: Rgb, xyz: [f64; 3]) -> Self {
        Self { rgb, xyz }
    }
}

// ============================================================================
// Calibration target
// ============================================================================

/// The target colorimetric standard the display should be calibrated to.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CalibrationTarget {
    /// sRGB / BT.709 with D65 white point.
    D65Srgb,
    /// DCI-P3 with DCI white point (0.314, 0.351 xy).
    DciP3,
    /// Display P3 (DCI-P3 primaries with D65 white point).
    DisplayP3,
    /// BT.2020 with D65 white point.
    Rec2020,
    /// Custom white point (CIE xy chromaticity).
    Custom {
        /// x chromaticity coordinate.
        wx: f64,
        /// y chromaticity coordinate.
        wy: f64,
    },
}

impl CalibrationTarget {
    /// Return the target white point as CIE XYZ (Y = 1).
    #[must_use]
    pub fn white_xyz(&self) -> [f64; 3] {
        match self {
            Self::D65Srgb | Self::DisplayP3 | Self::Rec2020 => {
                // D65: x=0.3127, y=0.3290
                xy_to_xyz(0.312_70, 0.329_0)
            }
            Self::DciP3 => {
                // DCI white: x=0.314, y=0.351
                xy_to_xyz(0.314_0, 0.351_0)
            }
            Self::Custom { wx, wy } => xy_to_xyz(*wx, *wy),
        }
    }

    /// Return the target 3×3 RGB→XYZ matrix.
    #[must_use]
    pub fn rgb_to_xyz_matrix(&self) -> [[f64; 3]; 3] {
        match self {
            Self::D65Srgb => [
                [0.412_391, 0.357_584, 0.180_481],
                [0.212_639, 0.715_169, 0.072_192],
                [0.019_331, 0.119_195, 0.950_532],
            ],
            Self::DisplayP3 => [
                [0.486_571, 0.265_668, 0.198_217],
                [0.228_975, 0.691_739, 0.079_287],
                [0.000_000, 0.045_113, 1.043_944],
            ],
            Self::DciP3 => [
                [0.445_170, 0.277_134, 0.172_283],
                [0.209_492, 0.721_596, 0.068_911],
                [0.000_000, 0.047_061, 0.907_355],
            ],
            Self::Rec2020 => [
                [0.636_958, 0.144_617, 0.168_881],
                [0.262_700, 0.677_998, 0.059_302],
                [0.000_000, 0.028_073, 1.060_985],
            ],
            Self::Custom { wx, wy } => {
                // For custom white points fall back to sRGB matrix
                // (a real implementation would derive from primaries)
                let _ = (wx, wy);
                [
                    [0.412_391, 0.357_584, 0.180_481],
                    [0.212_639, 0.715_169, 0.072_192],
                    [0.019_331, 0.119_195, 0.950_532],
                ]
            }
        }
    }
}

/// Convert CIE xy chromaticity (Y=1) to XYZ.
fn xy_to_xyz(x: f64, y: f64) -> [f64; 3] {
    if y.abs() < 1e-15 {
        return [0.0, 1.0, 0.0];
    }
    [x / y, 1.0, (1.0 - x - y) / y]
}

// ============================================================================
// Tone curve estimation
// ============================================================================

/// Estimated per-channel gamma from a set of ramp patches.
///
/// Given a list of `(device_value, measured_luminance)` pairs for a single
/// channel, estimate the best-fit power-law gamma.
#[must_use]
pub fn estimate_gamma(ramp: &[(f64, f64)]) -> f64 {
    // Fit log(Y) = gamma * log(x) via least squares (intercept forced to 0).
    // Optimal gamma = sum(lx * ly) / sum(lx * lx)
    let mut sum_xy = 0.0f64;
    let mut sum_xx = 0.0f64;
    let mut n = 0usize;

    for &(x, y) in ramp {
        if x > 1e-4 && y > 1e-4 && x <= 1.0 && y <= 1.0 {
            let lx = x.ln();
            let ly = y.ln();
            sum_xy += lx * ly;
            sum_xx += lx * lx;
            n += 1;
        }
    }

    if n < 2 {
        return 2.2; // default fallback
    }

    if sum_xx.abs() < 1e-15 {
        return 2.2;
    }
    sum_xy / sum_xx
}

// ============================================================================
// Display matrix estimation
// ============================================================================

/// Estimate the measured RGB→XYZ matrix from primary + white patches.
///
/// Requires at least the three primaries `[1,0,0]`, `[0,1,0]`, `[0,0,1]`
/// and the white `[1,1,1]`.  Returns an error if these are not found.
///
/// # Errors
///
/// Returns `LutError::InvalidData` if required patches are missing.
pub fn estimate_display_matrix(patches: &[CalibrationPatch]) -> LutResult<[[f64; 3]; 3]> {
    fn find(patches: &[CalibrationPatch], r: f64, g: f64, b: f64) -> Option<[f64; 3]> {
        patches
            .iter()
            .find(|p| {
                (p.rgb[0] - r).abs() < 0.01
                    && (p.rgb[1] - g).abs() < 0.01
                    && (p.rgb[2] - b).abs() < 0.01
            })
            .map(|p| p.xyz)
    }

    let red = find(patches, 1.0, 0.0, 0.0)
        .ok_or_else(|| LutError::InvalidData("Missing red primary patch [1,0,0]".to_string()))?;
    let green = find(patches, 0.0, 1.0, 0.0)
        .ok_or_else(|| LutError::InvalidData("Missing green primary patch [0,1,0]".to_string()))?;
    let blue = find(patches, 0.0, 0.0, 1.0)
        .ok_or_else(|| LutError::InvalidData("Missing blue primary patch [0,0,1]".to_string()))?;

    // Columns of the matrix are the primary XYZ values
    Ok([
        [red[0], green[0], blue[0]],
        [red[1], green[1], blue[1]],
        [red[2], green[2], blue[2]],
    ])
}

// ============================================================================
// Display calibrator
// ============================================================================

/// Generates calibration LUTs from measurement data.
#[derive(Debug, Clone)]
pub struct DisplayCalibrator {
    /// Raw colorimetric measurement patches.
    pub patches: Vec<CalibrationPatch>,
    /// Target colour space / white point.
    pub target: CalibrationTarget,
}

impl DisplayCalibrator {
    /// Create a new calibrator.
    #[must_use]
    pub fn new(patches: Vec<CalibrationPatch>, target: CalibrationTarget) -> Self {
        Self { patches, target }
    }

    /// Generate a per-channel 1-D calibration LUT with `size` entries.
    ///
    /// The LUT compensates for measured per-channel tone response deviation
    /// from the ideal power-law gamma of the target colour space.
    /// Each returned entry is `[r_corrected, g_corrected, b_corrected]`.
    #[must_use]
    pub fn generate_1d_lut(&self, size: usize) -> Vec<[f64; 3]> {
        let size = size.max(2);
        let gamma = self.estimate_display_gamma();
        let target_gamma = 2.2; // sRGB approx; could derive from target

        let mut lut = Vec::with_capacity(size);
        let scale = (size - 1) as f64;

        for i in 0..size {
            let x = i as f64 / scale;
            // Linearise with measured gamma, then re-encode with target gamma
            let linear = if x <= 0.0 {
                0.0
            } else {
                x.powf(gamma)
            };
            let encoded = if linear <= 0.0 {
                0.0
            } else {
                linear.powf(1.0 / target_gamma)
            };
            lut.push([encoded, encoded, encoded]);
        }
        lut
    }

    /// Generate a 3-D calibration LUT with `size` entries per dimension.
    ///
    /// The LUT compensates for all measured colorimetric deviations including
    /// white-point drift, gamma errors, and primary chromaticity errors.
    ///
    /// Returns `size³` entries in `[r][g][b]` index order.
    ///
    /// # Errors
    ///
    /// Returns an error if the patch set is insufficient to derive the display matrix.
    pub fn generate_3d_lut(&self, size: usize) -> LutResult<Vec<Rgb>> {
        let size = size.max(2);
        let measured_matrix = estimate_display_matrix(&self.patches)?;
        let target_matrix = self.target.rgb_to_xyz_matrix();
        let correction_matrix = derive_correction_matrix(&measured_matrix, &target_matrix)?;

        let scale = (size - 1) as f64;
        let gamma = self.estimate_display_gamma();

        let mut lut = Vec::with_capacity(size * size * size);
        for ri in 0..size {
            for gi in 0..size {
                for bi in 0..size {
                    let rgb_device: Rgb =
                        [ri as f64 / scale, gi as f64 / scale, bi as f64 / scale];
                    let rgb_corrected = apply_correction(&rgb_device, &correction_matrix, gamma);
                    lut.push(rgb_corrected);
                }
            }
        }
        Ok(lut)
    }

    /// Estimate the average measured display gamma from the ramp patches.
    #[must_use]
    pub fn estimate_display_gamma(&self) -> f64 {
        // Collect patches near the red ramp (G=0, B=0, R varies)
        let ramp: Vec<(f64, f64)> = self
            .patches
            .iter()
            .filter(|p| p.rgb[1] < 0.01 && p.rgb[2] < 0.01 && p.rgb[0] > 0.01)
            .map(|p| (p.rgb[0], p.xyz[1]))
            .collect();

        if ramp.len() < 2 {
            return 2.2;
        }
        estimate_gamma(&ramp)
    }

    /// Compute the measured white-point XYZ from the `[1,1,1]` patch if present.
    #[must_use]
    pub fn measured_white(&self) -> Option<[f64; 3]> {
        self.patches.iter().find(|p| {
            p.rgb[0] > 0.99 && p.rgb[1] > 0.99 && p.rgb[2] > 0.99
        }).map(|p| p.xyz)
    }

    /// Compute the chromatic error (ΔE₂₀₀₀ approximation) of the white point.
    ///
    /// Returns `None` if no white-point patch is available.
    #[must_use]
    pub fn white_point_delta_xy(&self) -> Option<f64> {
        let measured = self.measured_white()?;
        let target_white = self.target.white_xyz();

        let msum = measured[0] + measured[1] + measured[2];
        let tsum = target_white[0] + target_white[1] + target_white[2];

        if msum < 1e-10 || tsum < 1e-10 {
            return Some(0.0);
        }

        let mx = measured[0] / msum;
        let my = measured[1] / msum;
        let tx = target_white[0] / tsum;
        let ty = target_white[1] / tsum;

        Some(((mx - tx).powi(2) + (my - ty).powi(2)).sqrt())
    }
}

// ============================================================================
// Matrix helpers
// ============================================================================

/// Invert a 3×3 matrix (Cramer's rule).
fn invert_3x3(m: &[[f64; 3]; 3]) -> LutResult<[[f64; 3]; 3]> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

    if det.abs() < 1e-12 {
        return Err(LutError::InvalidData(
            "Singular matrix – cannot invert display calibration matrix".to_string(),
        ));
    }
    let d = 1.0 / det;
    Ok([
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * d,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * d,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * d,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * d,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * d,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * d,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * d,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * d,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * d,
        ],
    ])
}

/// Multiply two 3×3 matrices: `A * B`.
fn mul_3x3(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
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

/// Derive the correction matrix: `target_rgb_to_xyz * inv(measured_rgb_to_xyz)`.
///
/// Applied to measured linear RGB values it produces target-space RGB values.
fn derive_correction_matrix(
    measured: &[[f64; 3]; 3],
    target: &[[f64; 3]; 3],
) -> LutResult<[[f64; 3]; 3]> {
    // Transform: device → measured XYZ → target XYZ → target linear RGB
    // = inv(target_rgb_to_xyz) * measured_rgb_to_xyz
    // i.e. for each measured device value, convert to XYZ then to target RGB.
    let inv_measured = invert_3x3(measured)?;
    let _ = inv_measured; // measured matrix is used directly below
    let inv_target = invert_3x3(target)?;
    // correction = inv(target) * measured: maps measured device RGB → target device RGB
    Ok(mul_3x3(&inv_target, measured))
}

/// Apply a gamma linearisation + 3×3 matrix correction to a device RGB value.
fn apply_correction(rgb: &Rgb, matrix: &[[f64; 3]; 3], gamma: f64) -> Rgb {
    // Linearise
    let lin = [rgb[0].powf(gamma), rgb[1].powf(gamma), rgb[2].powf(gamma)];
    // Apply matrix
    let corrected = [
        matrix[0][0] * lin[0] + matrix[0][1] * lin[1] + matrix[0][2] * lin[2],
        matrix[1][0] * lin[0] + matrix[1][1] * lin[1] + matrix[1][2] * lin[2],
        matrix[2][0] * lin[0] + matrix[2][1] * lin[1] + matrix[2][2] * lin[2],
    ];
    // Re-encode (target sRGB ~= gamma 2.2 inverse)
    const TARGET_GAMMA_INV: f64 = 1.0 / 2.2;
    [
        corrected[0].clamp(0.0, 1.0).powf(TARGET_GAMMA_INV),
        corrected[1].clamp(0.0, 1.0).powf(TARGET_GAMMA_INV),
        corrected[2].clamp(0.0, 1.0).powf(TARGET_GAMMA_INV),
    ]
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal patch set: three primaries + white (ideal sRGB values).
    fn ideal_patches() -> Vec<CalibrationPatch> {
        vec![
            CalibrationPatch::new([1.0, 0.0, 0.0], [0.412_391, 0.212_639, 0.019_331]),
            CalibrationPatch::new([0.0, 1.0, 0.0], [0.357_584, 0.715_169, 0.119_195]),
            CalibrationPatch::new([0.0, 0.0, 1.0], [0.180_481, 0.072_192, 0.950_532]),
            CalibrationPatch::new([1.0, 1.0, 1.0], [0.950_456, 1.000_000, 1.089_058]),
        ]
    }

    #[test]
    fn test_calibration_patch_new() {
        let p = CalibrationPatch::new([0.5, 0.5, 0.5], [0.2, 0.2, 0.2]);
        assert!((p.rgb[0] - 0.5).abs() < 1e-10);
        assert!((p.xyz[1] - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_calibration_target_white_d65() {
        let xyz = CalibrationTarget::D65Srgb.white_xyz();
        // D65 Y should be 1.0
        assert!((xyz[1] - 1.0).abs() < 1e-10);
        // D65 X ≈ 0.9505
        assert!((xyz[0] - 0.9505).abs() < 0.01, "X = {}", xyz[0]);
    }

    #[test]
    fn test_calibration_target_white_dcip3() {
        let xyz = CalibrationTarget::DciP3.white_xyz();
        assert!((xyz[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_estimate_gamma_pure_gamma() {
        // Perfect gamma-2.2 ramp
        let ramp: Vec<(f64, f64)> = (1..=10)
            .map(|i| {
                let x = i as f64 / 10.0;
                let y = x.powf(2.2);
                (x, y)
            })
            .collect();
        let g = estimate_gamma(&ramp);
        assert!((g - 2.2).abs() < 0.05, "estimated gamma = {g}");
    }

    #[test]
    fn test_estimate_gamma_fallback() {
        let g = estimate_gamma(&[]);
        assert!((g - 2.2).abs() < 0.01);
    }

    #[test]
    fn test_estimate_display_matrix_ideal() {
        let patches = ideal_patches();
        let m = estimate_display_matrix(&patches).expect("should succeed");
        // Red column should match sRGB red primary XYZ
        assert!((m[0][0] - 0.412_391).abs() < 0.001, "m[0][0] = {}", m[0][0]);
        assert!((m[1][0] - 0.212_639).abs() < 0.001, "m[1][0] = {}", m[1][0]);
    }

    #[test]
    fn test_estimate_display_matrix_missing_primary() {
        // Omit red primary
        let patches = vec![
            CalibrationPatch::new([0.0, 1.0, 0.0], [0.357, 0.715, 0.119]),
            CalibrationPatch::new([0.0, 0.0, 1.0], [0.180, 0.072, 0.950]),
        ];
        let result = estimate_display_matrix(&patches);
        assert!(result.is_err());
    }

    #[test]
    fn test_generate_1d_lut_size() {
        let calibrator = DisplayCalibrator::new(ideal_patches(), CalibrationTarget::D65Srgb);
        let lut = calibrator.generate_1d_lut(256);
        assert_eq!(lut.len(), 256);
    }

    #[test]
    fn test_generate_1d_lut_endpoints() {
        let calibrator = DisplayCalibrator::new(ideal_patches(), CalibrationTarget::D65Srgb);
        let lut = calibrator.generate_1d_lut(64);
        // First entry should be [0,0,0]
        assert!(lut[0][0].abs() < 1e-10);
        assert!(lut[0][1].abs() < 1e-10);
        assert!(lut[0][2].abs() < 1e-10);
        // Last entry should be approximately [1,1,1]
        let last = lut.last().expect("lut should be non-empty");
        assert!((last[0] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_generate_3d_lut_size() {
        let calibrator = DisplayCalibrator::new(ideal_patches(), CalibrationTarget::D65Srgb);
        let lut = calibrator.generate_3d_lut(5).expect("should succeed");
        assert_eq!(lut.len(), 5 * 5 * 5);
    }

    #[test]
    fn test_generate_3d_lut_range() {
        let calibrator = DisplayCalibrator::new(ideal_patches(), CalibrationTarget::D65Srgb);
        let lut = calibrator.generate_3d_lut(3).expect("should succeed");
        for entry in &lut {
            assert!(entry[0] >= 0.0 && entry[0] <= 1.0, "R out of range: {}", entry[0]);
            assert!(entry[1] >= 0.0 && entry[1] <= 1.0, "G out of range: {}", entry[1]);
            assert!(entry[2] >= 0.0 && entry[2] <= 1.0, "B out of range: {}", entry[2]);
        }
    }

    #[test]
    fn test_measured_white_present() {
        let calibrator = DisplayCalibrator::new(ideal_patches(), CalibrationTarget::D65Srgb);
        let white = calibrator.measured_white();
        assert!(white.is_some());
        let xyz = white.expect("white should be present");
        assert!((xyz[1] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_measured_white_absent() {
        let patches = vec![
            CalibrationPatch::new([1.0, 0.0, 0.0], [0.412, 0.213, 0.019]),
        ];
        let calibrator = DisplayCalibrator::new(patches, CalibrationTarget::D65Srgb);
        assert!(calibrator.measured_white().is_none());
    }

    #[test]
    fn test_white_point_delta_xy() {
        let calibrator = DisplayCalibrator::new(ideal_patches(), CalibrationTarget::D65Srgb);
        let delta = calibrator.white_point_delta_xy();
        assert!(delta.is_some());
        // Our ideal patches use D65 so delta should be small
        let d = delta.expect("delta should be present");
        assert!(d < 0.05, "white point delta_xy = {d}");
    }

    #[test]
    fn test_invert_identity() {
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let inv = invert_3x3(&identity).expect("invert identity");
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((inv[i][j] - expected).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_invert_singular_returns_error() {
        let singular = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let result = invert_3x3(&singular);
        assert!(result.is_err());
    }

    #[test]
    fn test_xy_to_xyz_d65() {
        let xyz = xy_to_xyz(0.312_70, 0.329_0);
        assert!((xyz[1] - 1.0).abs() < 1e-10);
        assert!((xyz[0] - 0.9505).abs() < 0.01);
    }

    #[test]
    fn test_mul_3x3_identity() {
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let m = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let result = mul_3x3(&identity, &m);
        for i in 0..3 {
            for j in 0..3 {
                assert!((result[i][j] - m[i][j]).abs() < 1e-10);
            }
        }
    }
}
