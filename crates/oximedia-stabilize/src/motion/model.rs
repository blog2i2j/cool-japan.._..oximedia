//! Motion models for camera transformation.
//!
//! This module defines different motion models used to represent camera movement:
//! - Translation: Simple pan/tilt
//! - Affine: Translation + rotation + scale
//! - Perspective: Full homography
//! - 3D: Full 3D camera pose

use crate::error::{StabilizeError, StabilizeResult};
use serde::{Deserialize, Serialize};

/// A 3x3 matrix stored in row-major order.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Matrix3x3 {
    /// Row-major data: [r0c0, r0c1, r0c2, r1c0, r1c1, r1c2, r2c0, r2c1, r2c2]
    pub data: [f64; 9],
}

impl Matrix3x3 {
    /// Create a new matrix from elements in row-major order.
    #[must_use]
    pub const fn new(
        r0c0: f64,
        r0c1: f64,
        r0c2: f64,
        r1c0: f64,
        r1c1: f64,
        r1c2: f64,
        r2c0: f64,
        r2c1: f64,
        r2c2: f64,
    ) -> Self {
        Self {
            data: [r0c0, r0c1, r0c2, r1c0, r1c1, r1c2, r2c0, r2c1, r2c2],
        }
    }

    /// Create a 3x3 identity matrix.
    #[must_use]
    pub const fn identity() -> Self {
        Self::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    }

    /// Create a 3x3 zero matrix.
    #[must_use]
    pub const fn zeros() -> Self {
        Self { data: [0.0; 9] }
    }

    /// Get element at (row, col).
    #[must_use]
    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.data[row * 3 + col]
    }

    /// Set element at (row, col).
    pub fn set(&mut self, row: usize, col: usize, val: f64) {
        self.data[row * 3 + col] = val;
    }

    /// Multiply two 3x3 matrices.
    #[must_use]
    pub fn mul(&self, other: &Self) -> Self {
        let mut result = Self::zeros();
        for i in 0..3 {
            for j in 0..3 {
                let mut sum = 0.0;
                for k in 0..3 {
                    sum += self.get(i, k) * other.get(k, j);
                }
                result.set(i, j, sum);
            }
        }
        result
    }

    /// Transform a 2D point using this 3x3 matrix (homogeneous coordinates).
    #[must_use]
    pub fn transform_point(&self, x: f64, y: f64) -> (f64, f64) {
        let w = self.get(2, 0) * x + self.get(2, 1) * y + self.get(2, 2);
        if w.abs() < 1e-10 {
            return (x, y);
        }
        let px = self.get(0, 0) * x + self.get(0, 1) * y + self.get(0, 2);
        let py = self.get(1, 0) * x + self.get(1, 1) * y + self.get(1, 2);
        (px / w, py / w)
    }

    /// Compute determinant.
    #[must_use]
    pub fn determinant(&self) -> f64 {
        let a = self.data;
        a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6])
            + a[2] * (a[3] * a[7] - a[4] * a[6])
    }

    /// Compute inverse (returns None if singular).
    #[must_use]
    pub fn try_inverse(&self) -> Option<Self> {
        let det = self.determinant();
        if det.abs() < 1e-15 {
            return None;
        }
        let inv_det = 1.0 / det;
        let a = self.data;

        Some(Self::new(
            (a[4] * a[8] - a[5] * a[7]) * inv_det,
            (a[2] * a[7] - a[1] * a[8]) * inv_det,
            (a[1] * a[5] - a[2] * a[4]) * inv_det,
            (a[5] * a[6] - a[3] * a[8]) * inv_det,
            (a[0] * a[8] - a[2] * a[6]) * inv_det,
            (a[2] * a[3] - a[0] * a[5]) * inv_det,
            (a[3] * a[7] - a[4] * a[6]) * inv_det,
            (a[1] * a[6] - a[0] * a[7]) * inv_det,
            (a[0] * a[4] - a[1] * a[3]) * inv_det,
        ))
    }

    /// Frobenius norm.
    #[must_use]
    pub fn norm(&self) -> f64 {
        self.data.iter().map(|v| v * v).sum::<f64>().sqrt()
    }

    /// Subtract another matrix.
    #[must_use]
    pub fn sub(&self, other: &Self) -> Self {
        let mut result = Self::zeros();
        for i in 0..9 {
            result.data[i] = self.data[i] - other.data[i];
        }
        result
    }

    /// Scale all elements by a factor.
    #[must_use]
    pub fn scale(&self, factor: f64) -> Self {
        let mut result = *self;
        for v in &mut result.data {
            *v *= factor;
        }
        result
    }
}

/// A 3D vector.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Vector3 {
    /// Vector elements.
    pub data: [f64; 3],
}

impl Vector3 {
    /// Create a new vector.
    #[must_use]
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { data: [x, y, z] }
    }

    /// Create a zero vector.
    #[must_use]
    pub const fn zeros() -> Self {
        Self { data: [0.0; 3] }
    }

    /// Get element at index.
    #[must_use]
    pub fn get(&self, idx: usize) -> f64 {
        self.data[idx]
    }

    /// Norm of the vector.
    #[must_use]
    pub fn norm(&self) -> f64 {
        self.data.iter().map(|v| v * v).sum::<f64>().sqrt()
    }

    /// Negate the vector.
    #[must_use]
    pub fn neg(&self) -> Self {
        Self::new(-self.data[0], -self.data[1], -self.data[2])
    }
}

/// A motion model representing camera transformation between frames.
pub trait MotionModel: Send + Sync {
    /// Get the transformation matrix.
    fn matrix(&self) -> Matrix3x3;

    /// Set the transformation matrix.
    fn set_matrix(&mut self, matrix: Matrix3x3) -> StabilizeResult<()>;

    /// Transform a point.
    fn transform_point(&self, x: f64, y: f64) -> (f64, f64);

    /// Get motion parameters as a vector.
    fn parameters(&self) -> Vec<f64>;

    /// Set motion parameters from a vector.
    fn set_parameters(&mut self, params: &[f64]) -> StabilizeResult<()>;

    /// Compose with another model (self * other).
    fn compose(&self, other: &dyn MotionModel) -> StabilizeResult<Box<dyn MotionModel>>;

    /// Invert the transformation.
    fn invert(&self) -> StabilizeResult<Box<dyn MotionModel>>;

    /// Clone the motion model.
    fn clone_box(&self) -> Box<dyn MotionModel>;
}

/// Translation-only motion model (2 parameters: dx, dy).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslationModel {
    /// Translation in X
    pub dx: f64,
    /// Translation in Y
    pub dy: f64,
}

impl TranslationModel {
    /// Create a new translation model.
    #[must_use]
    pub const fn new(dx: f64, dy: f64) -> Self {
        Self { dx, dy }
    }

    /// Create an identity translation.
    #[must_use]
    pub const fn identity() -> Self {
        Self { dx: 0.0, dy: 0.0 }
    }

    /// Get magnitude of translation.
    #[must_use]
    pub fn magnitude(&self) -> f64 {
        (self.dx * self.dx + self.dy * self.dy).sqrt()
    }
}

impl MotionModel for TranslationModel {
    fn matrix(&self) -> Matrix3x3 {
        Matrix3x3::new(1.0, 0.0, self.dx, 0.0, 1.0, self.dy, 0.0, 0.0, 1.0)
    }

    fn set_matrix(&mut self, matrix: Matrix3x3) -> StabilizeResult<()> {
        self.dx = matrix.get(0, 2);
        self.dy = matrix.get(1, 2);
        Ok(())
    }

    fn transform_point(&self, x: f64, y: f64) -> (f64, f64) {
        (x + self.dx, y + self.dy)
    }

    fn parameters(&self) -> Vec<f64> {
        vec![self.dx, self.dy]
    }

    fn set_parameters(&mut self, params: &[f64]) -> StabilizeResult<()> {
        if params.len() != 2 {
            return Err(StabilizeError::invalid_parameter(
                "parameters",
                format!("expected 2, got {}", params.len()),
            ));
        }
        self.dx = params[0];
        self.dy = params[1];
        Ok(())
    }

    fn compose(&self, other: &dyn MotionModel) -> StabilizeResult<Box<dyn MotionModel>> {
        let params = other.parameters();
        if params.len() >= 2 {
            Ok(Box::new(Self::new(
                self.dx + params[0],
                self.dy + params[1],
            )))
        } else {
            Err(StabilizeError::invalid_parameter(
                "other model",
                "incompatible model type",
            ))
        }
    }

    fn invert(&self) -> StabilizeResult<Box<dyn MotionModel>> {
        Ok(Box::new(Self::new(-self.dx, -self.dy)))
    }

    fn clone_box(&self) -> Box<dyn MotionModel> {
        Box::new(self.clone())
    }
}

/// Affine motion model (6 parameters: rotation, scale, translation).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AffineModel {
    /// Translation in X
    pub dx: f64,
    /// Translation in Y
    pub dy: f64,
    /// Rotation angle in radians
    pub angle: f64,
    /// Scale factor
    pub scale: f64,
    /// Shear X
    pub shear_x: f64,
    /// Shear Y
    pub shear_y: f64,
}

impl AffineModel {
    /// Create a new affine model.
    #[must_use]
    pub const fn new(dx: f64, dy: f64, angle: f64, scale: f64, shear_x: f64, shear_y: f64) -> Self {
        Self {
            dx,
            dy,
            angle,
            scale,
            shear_x,
            shear_y,
        }
    }

    /// Create an identity affine transformation.
    #[must_use]
    pub const fn identity() -> Self {
        Self {
            dx: 0.0,
            dy: 0.0,
            angle: 0.0,
            scale: 1.0,
            shear_x: 0.0,
            shear_y: 0.0,
        }
    }

    /// Create from translation, rotation, and scale only.
    #[must_use]
    pub const fn from_trs(dx: f64, dy: f64, angle: f64, scale: f64) -> Self {
        Self {
            dx,
            dy,
            angle,
            scale,
            shear_x: 0.0,
            shear_y: 0.0,
        }
    }

    /// Get motion magnitude.
    #[must_use]
    pub fn magnitude(&self) -> f64 {
        let trans = (self.dx * self.dx + self.dy * self.dy).sqrt();
        let rot = self.angle.abs();
        let scale_dev = (self.scale - 1.0).abs();
        trans + rot * 10.0 + scale_dev * 10.0
    }
}

impl MotionModel for AffineModel {
    fn matrix(&self) -> Matrix3x3 {
        let cos_a = self.angle.cos();
        let sin_a = self.angle.sin();
        let s = self.scale;

        Matrix3x3::new(
            s * cos_a + self.shear_x,
            -s * sin_a,
            self.dx,
            s * sin_a + self.shear_y,
            s * cos_a,
            self.dy,
            0.0,
            0.0,
            1.0,
        )
    }

    fn set_matrix(&mut self, matrix: Matrix3x3) -> StabilizeResult<()> {
        self.dx = matrix.get(0, 2);
        self.dy = matrix.get(1, 2);

        let a = matrix.get(0, 0);
        let b = matrix.get(0, 1);
        let c = matrix.get(1, 0);
        let _d = matrix.get(1, 1);

        self.scale = (a * a + b * b).sqrt();
        self.angle = b.atan2(a);

        if self.scale > 0.0 {
            self.shear_x = a - self.scale * self.angle.cos();
            self.shear_y = c - self.scale * self.angle.sin();
        }

        Ok(())
    }

    fn transform_point(&self, x: f64, y: f64) -> (f64, f64) {
        let mat = self.matrix();
        let px = mat.get(0, 0) * x + mat.get(0, 1) * y + mat.get(0, 2);
        let py = mat.get(1, 0) * x + mat.get(1, 1) * y + mat.get(1, 2);
        (px, py)
    }

    fn parameters(&self) -> Vec<f64> {
        vec![
            self.dx,
            self.dy,
            self.angle,
            self.scale,
            self.shear_x,
            self.shear_y,
        ]
    }

    fn set_parameters(&mut self, params: &[f64]) -> StabilizeResult<()> {
        if params.len() != 6 {
            return Err(StabilizeError::invalid_parameter(
                "parameters",
                format!("expected 6, got {}", params.len()),
            ));
        }
        self.dx = params[0];
        self.dy = params[1];
        self.angle = params[2];
        self.scale = params[3];
        self.shear_x = params[4];
        self.shear_y = params[5];
        Ok(())
    }

    fn compose(&self, other: &dyn MotionModel) -> StabilizeResult<Box<dyn MotionModel>> {
        let m1 = self.matrix();
        let m2 = other.matrix();
        let result = m1.mul(&m2);

        let mut model = Self::identity();
        model.set_matrix(result)?;
        Ok(Box::new(model))
    }

    fn invert(&self) -> StabilizeResult<Box<dyn MotionModel>> {
        let mat = self.matrix();
        if let Some(inv) = mat.try_inverse() {
            let mut model = Self::identity();
            model.set_matrix(inv)?;
            Ok(Box::new(model))
        } else {
            Err(StabilizeError::matrix("Matrix is not invertible"))
        }
    }

    fn clone_box(&self) -> Box<dyn MotionModel> {
        Box::new(self.clone())
    }
}

/// Perspective motion model (8 parameters: full homography).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerspectiveModel {
    /// Homography matrix (3x3)
    homography: Matrix3x3,
}

impl PerspectiveModel {
    /// Create a new perspective model.
    #[must_use]
    pub fn new(homography: Matrix3x3) -> Self {
        Self { homography }
    }

    /// Create an identity perspective transformation.
    #[must_use]
    pub fn identity() -> Self {
        Self {
            homography: Matrix3x3::identity(),
        }
    }

    /// Get motion magnitude.
    #[must_use]
    pub fn magnitude(&self) -> f64 {
        // Frobenius norm of deviation from identity
        let identity = Matrix3x3::identity();
        let diff = self.homography.sub(&identity);
        diff.norm()
    }
}

impl MotionModel for PerspectiveModel {
    fn matrix(&self) -> Matrix3x3 {
        self.homography
    }

    fn set_matrix(&mut self, matrix: Matrix3x3) -> StabilizeResult<()> {
        self.homography = matrix;
        Ok(())
    }

    fn transform_point(&self, x: f64, y: f64) -> (f64, f64) {
        self.homography.transform_point(x, y)
    }

    fn parameters(&self) -> Vec<f64> {
        self.homography.data.to_vec()
    }

    fn set_parameters(&mut self, params: &[f64]) -> StabilizeResult<()> {
        if params.len() != 9 {
            return Err(StabilizeError::invalid_parameter(
                "parameters",
                format!("expected 9, got {}", params.len()),
            ));
        }

        for i in 0..3 {
            for j in 0..3 {
                self.homography.set(i, j, params[i * 3 + j]);
            }
        }
        Ok(())
    }

    fn compose(&self, other: &dyn MotionModel) -> StabilizeResult<Box<dyn MotionModel>> {
        let result = self.homography.mul(&other.matrix());
        Ok(Box::new(Self::new(result)))
    }

    fn invert(&self) -> StabilizeResult<Box<dyn MotionModel>> {
        if let Some(inv) = self.homography.try_inverse() {
            Ok(Box::new(Self::new(inv)))
        } else {
            Err(StabilizeError::matrix("Homography is not invertible"))
        }
    }

    fn clone_box(&self) -> Box<dyn MotionModel> {
        Box::new(self.clone())
    }
}

/// 3D camera pose model (12 parameters: rotation matrix + translation).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreeDModel {
    /// Rotation matrix (3x3)
    pub rotation: Matrix3x3,
    /// Translation vector
    pub translation: Vector3,
    /// Camera focal length
    pub focal_length: f64,
    /// Camera principal point (cx, cy)
    pub principal_point: (f64, f64),
}

impl ThreeDModel {
    /// Create a new 3D model.
    #[must_use]
    pub fn new(
        rotation: Matrix3x3,
        translation: Vector3,
        focal_length: f64,
        principal_point: (f64, f64),
    ) -> Self {
        Self {
            rotation,
            translation,
            focal_length,
            principal_point,
        }
    }

    /// Create an identity 3D transformation.
    #[must_use]
    pub fn identity() -> Self {
        Self {
            rotation: Matrix3x3::identity(),
            translation: Vector3::zeros(),
            focal_length: 1.0,
            principal_point: (0.0, 0.0),
        }
    }

    /// Get motion magnitude.
    #[must_use]
    pub fn magnitude(&self) -> f64 {
        let trans_mag = self.translation.norm();
        let rot_mag = self.rotation.sub(&Matrix3x3::identity()).norm();
        trans_mag + rot_mag * 10.0
    }

    /// Project 3D point to 2D.
    #[must_use]
    pub fn project_3d(&self, point: Vector3) -> (f64, f64) {
        // R * p + t
        let tx = self.rotation.get(0, 0) * point.get(0)
            + self.rotation.get(0, 1) * point.get(1)
            + self.rotation.get(0, 2) * point.get(2)
            + self.translation.get(0);
        let ty = self.rotation.get(1, 0) * point.get(0)
            + self.rotation.get(1, 1) * point.get(1)
            + self.rotation.get(1, 2) * point.get(2)
            + self.translation.get(1);
        let tz = self.rotation.get(2, 0) * point.get(0)
            + self.rotation.get(2, 1) * point.get(1)
            + self.rotation.get(2, 2) * point.get(2)
            + self.translation.get(2);

        if tz.abs() < 1e-10 {
            return self.principal_point;
        }

        let x = self.focal_length * tx / tz + self.principal_point.0;
        let y = self.focal_length * ty / tz + self.principal_point.1;

        (x, y)
    }
}

impl MotionModel for ThreeDModel {
    fn matrix(&self) -> Matrix3x3 {
        // Return homography approximation for 2D projection
        self.rotation
    }

    fn set_matrix(&mut self, matrix: Matrix3x3) -> StabilizeResult<()> {
        self.rotation = matrix;
        Ok(())
    }

    fn transform_point(&self, x: f64, y: f64) -> (f64, f64) {
        // Simple 2D approximation
        let mat = self.rotation;
        let px = mat.get(0, 0) * x + mat.get(0, 1) * y + self.translation.get(0);
        let py = mat.get(1, 0) * x + mat.get(1, 1) * y + self.translation.get(1);
        (px, py)
    }

    fn parameters(&self) -> Vec<f64> {
        let mut params = Vec::with_capacity(15);

        // Rotation matrix (9 values)
        for i in 0..3 {
            for j in 0..3 {
                params.push(self.rotation.get(i, j));
            }
        }

        // Translation (3 values)
        params.push(self.translation.get(0));
        params.push(self.translation.get(1));
        params.push(self.translation.get(2));

        // Intrinsics (3 values)
        params.push(self.focal_length);
        params.push(self.principal_point.0);
        params.push(self.principal_point.1);

        params
    }

    fn set_parameters(&mut self, params: &[f64]) -> StabilizeResult<()> {
        if params.len() != 15 {
            return Err(StabilizeError::invalid_parameter(
                "parameters",
                format!("expected 15, got {}", params.len()),
            ));
        }

        // Rotation matrix
        for i in 0..3 {
            for j in 0..3 {
                self.rotation.set(i, j, params[i * 3 + j]);
            }
        }

        // Translation
        self.translation = Vector3::new(params[9], params[10], params[11]);

        // Intrinsics
        self.focal_length = params[12];
        self.principal_point = (params[13], params[14]);

        Ok(())
    }

    fn compose(&self, other: &dyn MotionModel) -> StabilizeResult<Box<dyn MotionModel>> {
        let params = other.parameters();
        if params.len() >= 12 {
            // Compose rotations and translations
            let other_rot = other.matrix();
            let result_rot = self.rotation.mul(&other_rot);

            let other_trans = Vector3::new(params[9], params[10], params[11]);
            // R1 * t2 + t1
            let rt0 = self.rotation.get(0, 0) * other_trans.get(0)
                + self.rotation.get(0, 1) * other_trans.get(1)
                + self.rotation.get(0, 2) * other_trans.get(2)
                + self.translation.get(0);
            let rt1 = self.rotation.get(1, 0) * other_trans.get(0)
                + self.rotation.get(1, 1) * other_trans.get(1)
                + self.rotation.get(1, 2) * other_trans.get(2)
                + self.translation.get(1);
            let rt2 = self.rotation.get(2, 0) * other_trans.get(0)
                + self.rotation.get(2, 1) * other_trans.get(1)
                + self.rotation.get(2, 2) * other_trans.get(2)
                + self.translation.get(2);

            Ok(Box::new(Self::new(
                result_rot,
                Vector3::new(rt0, rt1, rt2),
                self.focal_length,
                self.principal_point,
            )))
        } else {
            Err(StabilizeError::invalid_parameter(
                "other model",
                "incompatible model type",
            ))
        }
    }

    fn invert(&self) -> StabilizeResult<Box<dyn MotionModel>> {
        if let Some(inv_rot) = self.rotation.try_inverse() {
            // -R^(-1) * t
            let t = self.translation;
            let it0 = -(inv_rot.get(0, 0) * t.get(0)
                + inv_rot.get(0, 1) * t.get(1)
                + inv_rot.get(0, 2) * t.get(2));
            let it1 = -(inv_rot.get(1, 0) * t.get(0)
                + inv_rot.get(1, 1) * t.get(1)
                + inv_rot.get(1, 2) * t.get(2));
            let it2 = -(inv_rot.get(2, 0) * t.get(0)
                + inv_rot.get(2, 1) * t.get(1)
                + inv_rot.get(2, 2) * t.get(2));
            Ok(Box::new(Self::new(
                inv_rot,
                Vector3::new(it0, it1, it2),
                self.focal_length,
                self.principal_point,
            )))
        } else {
            Err(StabilizeError::matrix("Rotation matrix is not invertible"))
        }
    }

    fn clone_box(&self) -> Box<dyn MotionModel> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_translation_model() {
        let model = TranslationModel::new(10.0, 20.0);
        let (x, y) = model.transform_point(0.0, 0.0);
        assert!((x - 10.0).abs() < f64::EPSILON);
        assert!((y - 20.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_translation_identity() {
        let model = TranslationModel::identity();
        let (x, y) = model.transform_point(5.0, 7.0);
        assert!((x - 5.0).abs() < f64::EPSILON);
        assert!((y - 7.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_affine_model() {
        let model = AffineModel::from_trs(10.0, 20.0, 0.0, 1.0);
        let (x, y) = model.transform_point(0.0, 0.0);
        assert!((x - 10.0).abs() < 1e-10);
        assert!((y - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_affine_rotation() {
        let model = AffineModel::from_trs(0.0, 0.0, std::f64::consts::PI / 2.0, 1.0);
        let (x, y) = model.transform_point(1.0, 0.0);
        assert!((x - 0.0).abs() < 1e-10);
        assert!((y - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_perspective_identity() {
        let model = PerspectiveModel::identity();
        let (x, y) = model.transform_point(5.0, 7.0);
        assert!((x - 5.0).abs() < f64::EPSILON);
        assert!((y - 7.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_model_parameters() {
        let model = TranslationModel::new(10.0, 20.0);
        let params = model.parameters();
        assert_eq!(params.len(), 2);
        assert!((params[0] - 10.0).abs() < f64::EPSILON);
        assert!((params[1] - 20.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_model_inversion() {
        let model = TranslationModel::new(10.0, 20.0);
        let inv = model.invert().expect("should succeed in test");
        let params = inv.parameters();
        assert!((params[0] + 10.0).abs() < f64::EPSILON);
        assert!((params[1] + 20.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_matrix3x3_inverse() {
        let m = Matrix3x3::new(2.0, 1.0, 0.0, 0.0, 3.0, 1.0, 1.0, 0.0, 2.0);
        let inv = m.try_inverse().expect("should be invertible");
        let product = m.mul(&inv);
        // Should be close to identity
        assert!((product.get(0, 0) - 1.0).abs() < 1e-10);
        assert!((product.get(1, 1) - 1.0).abs() < 1e-10);
        assert!((product.get(2, 2) - 1.0).abs() < 1e-10);
    }
}
