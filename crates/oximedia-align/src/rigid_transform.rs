#![allow(dead_code)]
//! Rigid body (rotation + translation) transformations for alignment.
//!
//! A rigid transform preserves distances and angles. It is characterised by a
//! 2-D rotation angle `theta` and a translation vector `(tx, ty)`. This is the
//! simplest geometric model for aligning images that differ only by camera
//! rotation and shift (no scaling or shearing).
//!
//! # Features
//!
//! - [`RigidTransform`] representation (angle + translation)
//! - Application of the transform to 2-D points
//! - Inverse transform
//! - Estimation from matched point pairs via least-squares
//! - Composition / chaining of transforms
//! - Residual error computation

use std::f64::consts::PI;

/// A rigid 2-D transform: rotation by `theta` followed by translation `(tx, ty)`.
///
/// The transformation of a point `(x, y)` is:
///
/// ```text
/// x' = cos(theta) * x - sin(theta) * y + tx
/// y' = sin(theta) * x + cos(theta) * y + ty
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RigidTransform {
    /// Rotation angle in radians
    pub theta: f64,
    /// Translation along the X axis
    pub tx: f64,
    /// Translation along the Y axis
    pub ty: f64,
}

impl RigidTransform {
    /// Create a new rigid transform.
    pub fn new(theta: f64, tx: f64, ty: f64) -> Self {
        Self { theta, tx, ty }
    }

    /// The identity transform (no rotation, no translation).
    pub fn identity() -> Self {
        Self {
            theta: 0.0,
            tx: 0.0,
            ty: 0.0,
        }
    }

    /// Create a pure translation (no rotation).
    pub fn translation(tx: f64, ty: f64) -> Self {
        Self {
            theta: 0.0,
            tx,
            ty,
        }
    }

    /// Create a pure rotation about the origin (no translation).
    pub fn rotation(theta: f64) -> Self {
        Self {
            theta,
            tx: 0.0,
            ty: 0.0,
        }
    }

    /// Apply this transform to a point.
    pub fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        let (sin_t, cos_t) = self.theta.sin_cos();
        let xp = cos_t * x - sin_t * y + self.tx;
        let yp = sin_t * x + cos_t * y + self.ty;
        (xp, yp)
    }

    /// Compute the inverse transform.
    pub fn inverse(&self) -> Self {
        let (sin_t, cos_t) = self.theta.sin_cos();
        // Inverse rotation is -theta, inverse translation rotated back
        let tx_inv = -(cos_t * self.tx + sin_t * self.ty);
        let ty_inv = -(-sin_t * self.tx + cos_t * self.ty);
        Self {
            theta: -self.theta,
            tx: tx_inv,
            ty: ty_inv,
        }
    }

    /// Compose two rigid transforms: first `self`, then `other`.
    /// Equivalent to applying `self` followed by `other`.
    pub fn compose(&self, other: &Self) -> Self {
        let theta = self.theta + other.theta;
        let (sin_o, cos_o) = other.theta.sin_cos();
        let tx = cos_o * self.tx - sin_o * self.ty + other.tx;
        let ty = sin_o * self.tx + cos_o * self.ty + other.ty;
        Self { theta, tx, ty }
    }

    /// Normalise the angle to `[-PI, PI)`.
    pub fn normalize_angle(&mut self) {
        self.theta = ((self.theta + PI) % (2.0 * PI) + 2.0 * PI) % (2.0 * PI) - PI;
    }

    /// Return the rotation angle in degrees.
    #[allow(clippy::cast_precision_loss)]
    pub fn angle_degrees(&self) -> f64 {
        self.theta.to_degrees()
    }

    /// Compute the translation magnitude.
    pub fn translation_magnitude(&self) -> f64 {
        (self.tx * self.tx + self.ty * self.ty).sqrt()
    }
}

/// A pair of corresponding 2-D points.
#[derive(Debug, Clone, Copy)]
pub struct PointPair {
    /// Source point X
    pub src_x: f64,
    /// Source point Y
    pub src_y: f64,
    /// Destination point X
    pub dst_x: f64,
    /// Destination point Y
    pub dst_y: f64,
}

impl PointPair {
    /// Create a new point pair.
    pub fn new(src_x: f64, src_y: f64, dst_x: f64, dst_y: f64) -> Self {
        Self {
            src_x,
            src_y,
            dst_x,
            dst_y,
        }
    }
}

/// Estimate a rigid transform from 2 or more point correspondences using
/// least-squares (the Procrustes solution without scaling).
///
/// Returns `None` if fewer than 2 correspondences are provided.
#[allow(clippy::cast_precision_loss)]
pub fn estimate_rigid(pairs: &[PointPair]) -> Option<RigidTransform> {
    if pairs.len() < 2 {
        return None;
    }
    let n = pairs.len() as f64;

    // Compute centroids
    let (cx_s, cy_s) = pairs
        .iter()
        .fold((0.0, 0.0), |(sx, sy), p| (sx + p.src_x, sy + p.src_y));
    let (cx_d, cy_d) = pairs
        .iter()
        .fold((0.0, 0.0), |(sx, sy), p| (sx + p.dst_x, sy + p.dst_y));
    let (cx_s, cy_s) = (cx_s / n, cy_s / n);
    let (cx_d, cy_d) = (cx_d / n, cy_d / n);

    // Compute cross-covariance sums for rotation
    let mut sum_sin = 0.0;
    let mut sum_cos = 0.0;
    for p in pairs {
        let sx = p.src_x - cx_s;
        let sy = p.src_y - cy_s;
        let dx = p.dst_x - cx_d;
        let dy = p.dst_y - cy_d;
        sum_cos += sx * dx + sy * dy;
        sum_sin += sx * dy - sy * dx;
    }

    let theta = sum_sin.atan2(sum_cos);
    let (sin_t, cos_t) = theta.sin_cos();
    let tx = cx_d - (cos_t * cx_s - sin_t * cy_s);
    let ty = cy_d - (sin_t * cx_s + cos_t * cy_s);

    Some(RigidTransform { theta, tx, ty })
}

/// Compute the root mean square error of a rigid transform given correspondences.
#[allow(clippy::cast_precision_loss)]
pub fn rmse(transform: &RigidTransform, pairs: &[PointPair]) -> f64 {
    if pairs.is_empty() {
        return 0.0;
    }
    let sum: f64 = pairs
        .iter()
        .map(|p| {
            let (xp, yp) = transform.apply(p.src_x, p.src_y);
            let dx = xp - p.dst_x;
            let dy = yp - p.dst_y;
            dx * dx + dy * dy
        })
        .sum();
    (sum / pairs.len() as f64).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-9;

    #[test]
    fn test_identity_apply() {
        let t = RigidTransform::identity();
        let (x, y) = t.apply(3.0, 4.0);
        assert!((x - 3.0).abs() < EPS);
        assert!((y - 4.0).abs() < EPS);
    }

    #[test]
    fn test_translation_apply() {
        let t = RigidTransform::translation(5.0, -3.0);
        let (x, y) = t.apply(1.0, 2.0);
        assert!((x - 6.0).abs() < EPS);
        assert!((y + 1.0).abs() < EPS);
    }

    #[test]
    fn test_rotation_90_degrees() {
        let t = RigidTransform::rotation(PI / 2.0);
        let (x, y) = t.apply(1.0, 0.0);
        assert!(x.abs() < EPS);
        assert!((y - 1.0).abs() < EPS);
    }

    #[test]
    fn test_inverse_roundtrip() {
        let t = RigidTransform::new(0.3, 5.0, -2.0);
        let inv = t.inverse();
        let (x, y) = t.apply(7.0, 3.0);
        let (xb, yb) = inv.apply(x, y);
        assert!((xb - 7.0).abs() < EPS);
        assert!((yb - 3.0).abs() < EPS);
    }

    #[test]
    fn test_compose_with_identity() {
        let t = RigidTransform::new(0.5, 1.0, 2.0);
        let id = RigidTransform::identity();
        let c = t.compose(&id);
        assert!((c.theta - t.theta).abs() < EPS);
        assert!((c.tx - t.tx).abs() < EPS);
        assert!((c.ty - t.ty).abs() < EPS);
    }

    #[test]
    fn test_compose_two_translations() {
        let t1 = RigidTransform::translation(1.0, 2.0);
        let t2 = RigidTransform::translation(3.0, 4.0);
        let c = t1.compose(&t2);
        assert!(c.theta.abs() < EPS);
        assert!((c.tx - 4.0).abs() < EPS);
        assert!((c.ty - 6.0).abs() < EPS);
    }

    #[test]
    fn test_angle_degrees() {
        let t = RigidTransform::rotation(PI / 4.0);
        assert!((t.angle_degrees() - 45.0).abs() < 1e-6);
    }

    #[test]
    fn test_translation_magnitude() {
        let t = RigidTransform::translation(3.0, 4.0);
        assert!((t.translation_magnitude() - 5.0).abs() < EPS);
    }

    #[test]
    fn test_normalize_angle() {
        let mut t = RigidTransform::rotation(3.0 * PI);
        t.normalize_angle();
        assert!((t.theta - PI).abs() < 1e-6);
    }

    #[test]
    fn test_estimate_pure_translation() {
        let pairs = vec![
            PointPair::new(0.0, 0.0, 1.0, 2.0),
            PointPair::new(1.0, 0.0, 2.0, 2.0),
            PointPair::new(0.0, 1.0, 1.0, 3.0),
        ];
        let t = estimate_rigid(&pairs).expect("t should be valid");
        assert!(t.theta.abs() < 1e-6);
        assert!((t.tx - 1.0).abs() < 1e-6);
        assert!((t.ty - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_estimate_insufficient_points() {
        let pairs = vec![PointPair::new(0.0, 0.0, 1.0, 1.0)];
        assert!(estimate_rigid(&pairs).is_none());
    }

    #[test]
    fn test_rmse_perfect() {
        let t = RigidTransform::translation(1.0, 0.0);
        let pairs = vec![
            PointPair::new(0.0, 0.0, 1.0, 0.0),
            PointPair::new(1.0, 0.0, 2.0, 0.0),
        ];
        assert!(rmse(&t, &pairs) < EPS);
    }

    #[test]
    fn test_rmse_empty() {
        let t = RigidTransform::identity();
        assert!(rmse(&t, &[]).abs() < EPS);
    }
}
