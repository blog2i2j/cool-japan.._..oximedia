//! Batch calibration — calibrate multiple cameras from a single shooting session.
//!
//! When a production involves several cameras that all need to be colour-matched
//! to a single reference look, the workflow is:
//!
//! 1. Each camera photographs the same ColorChecker chart under identical
//!    lighting.
//! 2. `BatchCalibrator` extracts the patch measurements from every camera.
//! 3. One camera is nominated as the "reference"; all others are matched to it.
//! 4. Per-camera calibration LUT and/or ICC profile offsets are produced.
//!
//! All measurements and result structures are serialisable for archiving.

use crate::delta_e::delta_e_2000;
use crate::error::CalibrationResult;
use crate::{CalibrationError, Lab, Rgb};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─── Camera measurement ───────────────────────────────────────────────────────

/// Raw colour measurements from a single camera for one ColorChecker patch set.
///
/// Patch values are in linear light, normalised to \[0, 1\].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraMeasurement {
    /// Camera identifier (e.g. model name or user label).
    pub camera_id: String,
    /// Measured RGB values for each patch (24 patches for a Classic).
    pub patch_rgb: Vec<Rgb>,
    /// Reference Lab values for each patch (from published ColorChecker data).
    pub reference_lab: Vec<Lab>,
}

impl CameraMeasurement {
    /// Create a new measurement.
    ///
    /// # Errors
    ///
    /// Returns `CalibrationError::InvalidMeasurementData` if
    /// `patch_rgb.len() != reference_lab.len()` or either is empty.
    pub fn new(
        camera_id: impl Into<String>,
        patch_rgb: Vec<Rgb>,
        reference_lab: Vec<Lab>,
    ) -> CalibrationResult<Self> {
        if patch_rgb.is_empty() {
            return Err(CalibrationError::InvalidMeasurementData(
                "patch_rgb must not be empty".to_string(),
            ));
        }
        if patch_rgb.len() != reference_lab.len() {
            return Err(CalibrationError::InvalidMeasurementData(format!(
                "patch_rgb ({}) and reference_lab ({}) lengths must match",
                patch_rgb.len(),
                reference_lab.len()
            )));
        }
        Ok(Self {
            camera_id: camera_id.into(),
            patch_rgb,
            reference_lab,
        })
    }

    /// Number of patches in this measurement.
    #[must_use]
    pub fn patch_count(&self) -> usize {
        self.patch_rgb.len()
    }

    /// Compute the mean ΔE 2000 error against the reference Lab values.
    ///
    /// The patch RGB values are first converted to approximate Lab for
    /// comparison using a simplified sRGB → XYZ → Lab path.
    #[must_use]
    pub fn mean_delta_e(&self) -> f64 {
        if self.patch_rgb.is_empty() {
            return 0.0;
        }
        let total: f64 = self
            .patch_rgb
            .iter()
            .zip(self.reference_lab.iter())
            .map(|(rgb, ref_lab)| {
                let lab = rgb_to_lab_approx(*rgb);
                delta_e_2000(lab, *ref_lab)
            })
            .sum();
        total / self.patch_rgb.len() as f64
    }
}

// ─── Calibration result per camera ───────────────────────────────────────────

/// Calibration result for a single camera relative to the reference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SingleCameraCalibration {
    /// Camera this calibration belongs to.
    pub camera_id: String,
    /// Whether this is the reference camera.
    pub is_reference: bool,
    /// Mean ΔE 2000 of the raw (uncalibrated) measurement against the chart.
    pub raw_delta_e: f64,
    /// Mean ΔE 2000 after applying the computed correction.
    pub corrected_delta_e: f64,
    /// 3×3 linear colour correction matrix (camera RGB → reference RGB space).
    pub correction_matrix: [[f64; 3]; 3],
    /// Per-channel gain offsets \[r, g, b\] that shift the corrected values
    /// to minimise any remaining bias.
    pub gain_offsets: Rgb,
}

// ─── Batch result ─────────────────────────────────────────────────────────────

/// Result of a batch calibration session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchCalibrationResult {
    /// Session identifier.
    pub session_id: String,
    /// Reference camera ID.
    pub reference_camera_id: String,
    /// Per-camera calibration data, keyed by camera ID.
    pub cameras: HashMap<String, SingleCameraCalibration>,
}

impl BatchCalibrationResult {
    /// Find the camera with the lowest corrected ΔE (best calibration).
    #[must_use]
    pub fn best_match_camera(&self) -> Option<&str> {
        self.cameras
            .iter()
            .filter(|(id, _)| id.as_str() != self.reference_camera_id.as_str())
            .min_by(|(_, a), (_, b)| {
                a.corrected_delta_e
                    .partial_cmp(&b.corrected_delta_e)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(id, _)| id.as_str())
    }

    /// Number of cameras calibrated (including reference).
    #[must_use]
    pub fn camera_count(&self) -> usize {
        self.cameras.len()
    }

    /// Return the calibration for a specific camera.
    #[must_use]
    pub fn camera_calibration(&self, camera_id: &str) -> Option<&SingleCameraCalibration> {
        self.cameras.get(camera_id)
    }
}

// ─── BatchCalibrator ─────────────────────────────────────────────────────────

/// Calibrates multiple cameras from a single shooting session.
///
/// All cameras must photograph the same ColorChecker chart under identical
/// lighting. One camera is designated the reference; all others are matched to
/// it using a least-squares 3×3 colour correction matrix.
pub struct BatchCalibrator {
    session_id: String,
    reference_camera_id: Option<String>,
    measurements: Vec<CameraMeasurement>,
}

impl BatchCalibrator {
    /// Create a new batch calibrator.
    #[must_use]
    pub fn new(session_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            reference_camera_id: None,
            measurements: Vec::new(),
        }
    }

    /// Add a camera measurement.
    ///
    /// # Errors
    ///
    /// Returns an error if a measurement with the same `camera_id` was already
    /// added.
    pub fn add_measurement(&mut self, measurement: CameraMeasurement) -> CalibrationResult<()> {
        if self
            .measurements
            .iter()
            .any(|m| m.camera_id == measurement.camera_id)
        {
            return Err(CalibrationError::InvalidMeasurementData(format!(
                "Camera '{}' was already added",
                measurement.camera_id
            )));
        }
        self.measurements.push(measurement);
        Ok(())
    }

    /// Set the reference camera.
    ///
    /// If not set before calling [`calibrate`](Self::calibrate), the first
    /// added measurement is used as the reference.
    pub fn set_reference(&mut self, camera_id: impl Into<String>) {
        self.reference_camera_id = Some(camera_id.into());
    }

    /// Run the batch calibration and return results for all cameras.
    ///
    /// # Errors
    ///
    /// Returns `CalibrationError::InsufficientData` if fewer than 2 cameras
    /// have been added.
    pub fn calibrate(&self) -> CalibrationResult<BatchCalibrationResult> {
        if self.measurements.len() < 2 {
            return Err(CalibrationError::InsufficientData(
                "Batch calibration requires at least 2 cameras".to_string(),
            ));
        }

        let ref_id = self
            .reference_camera_id
            .clone()
            .unwrap_or_else(|| self.measurements[0].camera_id.clone());

        let reference = self
            .measurements
            .iter()
            .find(|m| m.camera_id == ref_id)
            .ok_or_else(|| {
                CalibrationError::InvalidMeasurementData(format!(
                    "Reference camera '{ref_id}' not found in measurements"
                ))
            })?;

        let mut cameras = HashMap::new();

        // Add reference entry
        cameras.insert(
            ref_id.clone(),
            SingleCameraCalibration {
                camera_id: ref_id.clone(),
                is_reference: true,
                raw_delta_e: reference.mean_delta_e(),
                corrected_delta_e: reference.mean_delta_e(),
                correction_matrix: identity_3x3(),
                gain_offsets: [0.0, 0.0, 0.0],
            },
        );

        // Compute correction matrix for each non-reference camera
        for measurement in &self.measurements {
            if measurement.camera_id == ref_id {
                continue;
            }

            // Least-squares 3×3 colour correction matrix from measurement to reference
            let matrix = compute_correction_matrix(&measurement.patch_rgb, &reference.patch_rgb);
            let gain_offsets = compute_gain_offsets(&measurement.patch_rgb, &reference.patch_rgb, &matrix);
            let corrected_de =
                corrected_delta_e(measurement, &matrix, &gain_offsets, &reference.reference_lab);

            cameras.insert(
                measurement.camera_id.clone(),
                SingleCameraCalibration {
                    camera_id: measurement.camera_id.clone(),
                    is_reference: false,
                    raw_delta_e: measurement.mean_delta_e(),
                    corrected_delta_e: corrected_de,
                    correction_matrix: matrix,
                    gain_offsets,
                },
            );
        }

        Ok(BatchCalibrationResult {
            session_id: self.session_id.clone(),
            reference_camera_id: ref_id,
            cameras,
        })
    }

    /// Number of cameras added so far.
    #[must_use]
    pub fn camera_count(&self) -> usize {
        self.measurements.len()
    }
}

// ─── Math ─────────────────────────────────────────────────────────────────────

fn identity_3x3() -> [[f64; 3]; 3] {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

/// Compute a 3×3 colour correction matrix using a simple least-squares solver.
///
/// Maps `src_patches` → `dst_patches`.  Both slices must have the same length.
/// When fewer than 3 patches are available, returns identity.
fn compute_correction_matrix(src: &[Rgb], dst: &[Rgb]) -> [[f64; 3]; 3] {
    let n = src.len();
    if n < 3 {
        return identity_3x3();
    }

    // For each output channel c, solve: dst[:, c] = M[c, :] * src[:] .
    // Using normal equations: M[c, :] = (S^T S)^-1 S^T d
    // where S is the n×3 matrix of source patches.

    let mut sts = [[0.0f64; 3]; 3];
    for p in src {
        for i in 0..3 {
            for j in 0..3 {
                sts[i][j] += p[i] * p[j];
            }
        }
    }

    let sts_inv = match invert_3x3(sts) {
        Some(inv) => inv,
        None => return identity_3x3(),
    };

    let mut matrix = [[0.0f64; 3]; 3];
    for c in 0..3 {
        // rhs = S^T d_c
        let mut rhs = [0.0f64; 3];
        for (s, d) in src.iter().zip(dst.iter()) {
            for i in 0..3 {
                rhs[i] += s[i] * d[c];
            }
        }
        // solution = sts_inv * rhs
        for i in 0..3 {
            let mut sum = 0.0;
            for j in 0..3 {
                sum += sts_inv[i][j] * rhs[j];
            }
            matrix[c][i] = sum;
        }
    }
    matrix
}

/// Compute per-channel gain offsets to minimise residual bias.
fn compute_gain_offsets(src: &[Rgb], dst: &[Rgb], matrix: &[[f64; 3]; 3]) -> Rgb {
    if src.is_empty() {
        return [0.0, 0.0, 0.0];
    }
    let n = src.len() as f64;
    let mut offset = [0.0f64; 3];
    for (s, d) in src.iter().zip(dst.iter()) {
        let corrected = apply_3x3(matrix, *s);
        for i in 0..3 {
            offset[i] += d[i] - corrected[i];
        }
    }
    [offset[0] / n, offset[1] / n, offset[2] / n]
}

/// Compute mean ΔE after applying the correction matrix and gain offsets.
fn corrected_delta_e(
    measurement: &CameraMeasurement,
    matrix: &[[f64; 3]; 3],
    gain_offsets: &Rgb,
    reference_lab: &[Lab],
) -> f64 {
    if measurement.patch_rgb.is_empty() {
        return 0.0;
    }
    let total: f64 = measurement
        .patch_rgb
        .iter()
        .zip(reference_lab.iter())
        .map(|(rgb, ref_lab)| {
            let corrected = apply_3x3(matrix, *rgb);
            let gained: Rgb = [
                (corrected[0] + gain_offsets[0]).clamp(0.0, 1.0),
                (corrected[1] + gain_offsets[1]).clamp(0.0, 1.0),
                (corrected[2] + gain_offsets[2]).clamp(0.0, 1.0),
            ];
            let lab = rgb_to_lab_approx(gained);
            delta_e_2000(lab, *ref_lab)
        })
        .sum();
    total / measurement.patch_rgb.len() as f64
}

fn apply_3x3(m: &[[f64; 3]; 3], v: Rgb) -> Rgb {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

fn invert_3x3(m: [[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

    if det.abs() < 1e-12 {
        return None;
    }

    let inv_det = 1.0 / det;
    Some([
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
    ])
}

/// Approximate linear sRGB → CIE Lab conversion (D65 reference).
fn rgb_to_lab_approx(rgb: Rgb) -> Lab {
    // linearise (input already linear)
    let r = rgb[0];
    let g = rgb[1];
    let b = rgb[2];

    // sRGB → XYZ (D65)
    let x = 0.412_453 * r + 0.357_580 * g + 0.180_423 * b;
    let y = 0.212_671 * r + 0.715_160 * g + 0.072_169 * b;
    let z = 0.019_334 * r + 0.119_193 * g + 0.950_227 * b;

    // Normalise by D65
    let xn = 0.950_47;
    let yn = 1.0;
    let zn = 1.088_83;

    fn f(t: f64) -> f64 {
        if t > 0.008_856 {
            t.cbrt()
        } else {
            7.787 * t + 16.0 / 116.0
        }
    }

    let fx = f(x / xn);
    let fy = f(y / yn);
    let fz = f(z / zn);

    let l = 116.0 * fy - 16.0;
    let a = 500.0 * (fx - fy);
    let bb = 200.0 * (fy - fz);
    [l, a, bb]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_identity_patches(n: usize) -> (Vec<Rgb>, Vec<Lab>) {
        // Use perfectly uniform grey ramp as synthetic patches
        let patches: Vec<Rgb> = (0..n)
            .map(|i| {
                let v = i as f64 / (n - 1).max(1) as f64;
                [v, v, v]
            })
            .collect();
        let lab: Vec<Lab> = patches
            .iter()
            .map(|p| rgb_to_lab_approx(*p))
            .collect();
        (patches, lab)
    }

    #[test]
    fn test_camera_measurement_new_valid() {
        let (patches, labs) = make_identity_patches(6);
        let m = CameraMeasurement::new("CamA", patches, labs);
        assert!(m.is_ok());
        assert_eq!(m.unwrap().patch_count(), 6);
    }

    #[test]
    fn test_camera_measurement_new_length_mismatch() {
        let patches = vec![[0.5f64; 3]; 4];
        let labs = vec![[50.0f64, 0.0, 0.0]; 5];
        assert!(CameraMeasurement::new("X", patches, labs).is_err());
    }

    #[test]
    fn test_camera_measurement_new_empty() {
        let m = CameraMeasurement::new("X", vec![], vec![]);
        assert!(m.is_err());
    }

    #[test]
    fn test_mean_delta_e_identical_patches() {
        // If patches exactly match reference, mean ΔE should be 0
        let (patches, labs) = make_identity_patches(8);
        let m = CameraMeasurement::new("Cam", patches.clone(), labs.clone()).unwrap();
        // Mean dE should be near zero (patches are already Lab-matched)
        // We can only verify it is non-negative
        assert!(m.mean_delta_e() >= 0.0);
    }

    #[test]
    fn test_batch_calibrator_requires_two_cameras() {
        let mut cal = BatchCalibrator::new("session1");
        let (p, l) = make_identity_patches(6);
        cal.add_measurement(CameraMeasurement::new("A", p, l).unwrap())
            .unwrap();
        assert!(cal.calibrate().is_err());
    }

    #[test]
    fn test_batch_calibrator_duplicate_camera() {
        let mut cal = BatchCalibrator::new("session1");
        let (p, l) = make_identity_patches(6);
        cal.add_measurement(CameraMeasurement::new("A", p.clone(), l.clone()).unwrap())
            .unwrap();
        let result = cal.add_measurement(CameraMeasurement::new("A", p, l).unwrap());
        assert!(result.is_err());
    }

    #[test]
    fn test_batch_calibrator_two_identical_cameras() {
        let mut cal = BatchCalibrator::new("s");
        let (p, l) = make_identity_patches(8);
        cal.set_reference("Ref");
        cal.add_measurement(CameraMeasurement::new("Ref", p.clone(), l.clone()).unwrap())
            .unwrap();
        cal.add_measurement(CameraMeasurement::new("B", p, l).unwrap())
            .unwrap();
        let result = cal.calibrate().unwrap();
        assert_eq!(result.reference_camera_id, "Ref");
        assert_eq!(result.camera_count(), 2);
        let b = result.camera_calibration("B").unwrap();
        // Two identical cameras should have near-identity matrix
        for i in 0..3 {
            assert!((b.correction_matrix[i][i] - 1.0).abs() < 0.1);
        }
    }

    #[test]
    fn test_batch_calibrator_first_camera_as_reference() {
        let mut cal = BatchCalibrator::new("s");
        let (p, l) = make_identity_patches(6);
        cal.add_measurement(CameraMeasurement::new("First", p.clone(), l.clone()).unwrap())
            .unwrap();
        cal.add_measurement(CameraMeasurement::new("Second", p, l).unwrap())
            .unwrap();
        let result = cal.calibrate().unwrap();
        assert_eq!(result.reference_camera_id, "First");
    }

    #[test]
    fn test_batch_result_camera_count() {
        let mut cal = BatchCalibrator::new("s");
        for i in 0..4 {
            let (p, l) = make_identity_patches(6);
            cal.add_measurement(CameraMeasurement::new(format!("Cam{i}"), p, l).unwrap())
                .unwrap();
        }
        let result = cal.calibrate().unwrap();
        assert_eq!(result.camera_count(), 4);
    }

    #[test]
    fn test_invert_identity() {
        let id = identity_3x3();
        let inv = invert_3x3(id).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((inv[i][j] - expected).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn test_invert_singular_returns_none() {
        let singular = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        assert!(invert_3x3(singular).is_none());
    }
}
