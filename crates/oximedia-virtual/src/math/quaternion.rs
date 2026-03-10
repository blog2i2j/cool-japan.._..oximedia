//! Quaternion types for 3D rotation.

use super::matrix::Matrix3;
use super::vector::Vector3;
use serde::{Deserialize, Serialize};
use std::ops::Mul;

// ---------------------------------------------------------------------------
// Quaternion
// ---------------------------------------------------------------------------

/// A quaternion (w + xi + yj + zk).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Quaternion<T> {
    pub w: T,
    pub x: T,
    pub y: T,
    pub z: T,
}

impl Quaternion<f64> {
    /// Construct.
    #[must_use]
    pub fn new(w: f64, x: f64, y: f64, z: f64) -> Self {
        Self { w, x, y, z }
    }

    /// Norm.
    #[must_use]
    pub fn norm(&self) -> f64 {
        (self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Normalise.
    #[must_use]
    pub fn normalize(&self) -> Self {
        let n = self.norm();
        if n < 1e-15 {
            return *self;
        }
        Self::new(self.w / n, self.x / n, self.y / n, self.z / n)
    }

    /// Conjugate.
    #[must_use]
    pub fn conjugate(&self) -> Self {
        Self::new(self.w, -self.x, -self.y, -self.z)
    }
}

impl Mul for Quaternion<f64> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self::new(
            self.w * rhs.w - self.x * rhs.x - self.y * rhs.y - self.z * rhs.z,
            self.w * rhs.x + self.x * rhs.w + self.y * rhs.z - self.z * rhs.y,
            self.w * rhs.y - self.x * rhs.z + self.y * rhs.w + self.z * rhs.x,
            self.w * rhs.z + self.x * rhs.y - self.y * rhs.x + self.z * rhs.w,
        )
    }
}

// ---------------------------------------------------------------------------
// Unit (for axis normalisation)
// ---------------------------------------------------------------------------

/// A unit-length value wrapper (like nalgebra::Unit).
#[derive(Debug, Clone, Copy)]
pub struct Unit<T>(pub T);

impl Unit<Vector3<f64>> {
    /// Normalise a vector into a Unit wrapper.
    #[must_use]
    pub fn new_normalize(v: Vector3<f64>) -> Self {
        Unit(v.normalize())
    }
}

// ---------------------------------------------------------------------------
// UnitQuaternion
// ---------------------------------------------------------------------------

/// A unit quaternion representing a rotation in 3D.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct UnitQuaternion<T> {
    q: Quaternion<T>,
}

impl UnitQuaternion<f64> {
    /// Identity rotation.
    #[must_use]
    pub fn identity() -> Self {
        Self {
            q: Quaternion::new(1.0, 0.0, 0.0, 0.0),
        }
    }

    /// Construct from an already-normalised Quaternion.
    #[must_use]
    pub fn from_quaternion(q: Quaternion<f64>) -> Self {
        Self { q: q.normalize() }
    }

    /// Access inner quaternion.
    #[must_use]
    pub fn quaternion(&self) -> &Quaternion<f64> {
        &self.q
    }

    /// Construct from Euler angles (roll, pitch, yaw) — ZYX convention.
    #[must_use]
    pub fn from_euler_angles(roll: f64, pitch: f64, yaw: f64) -> Self {
        let (sr, cr) = (roll / 2.0).sin_cos();
        let (sp, cp) = (pitch / 2.0).sin_cos();
        let (sy, cy) = (yaw / 2.0).sin_cos();

        let w = cr * cp * cy + sr * sp * sy;
        let x = sr * cp * cy - cr * sp * sy;
        let y = cr * sp * cy + sr * cp * sy;
        let z = cr * cp * sy - sr * sp * cy;

        Self {
            q: Quaternion::new(w, x, y, z).normalize(),
        }
    }

    /// Extract Euler angles (roll, pitch, yaw).
    #[must_use]
    pub fn euler_angles(&self) -> (f64, f64, f64) {
        let q = &self.q;
        // Roll (x-axis rotation)
        let sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z);
        let cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y);
        let roll = sinr_cosp.atan2(cosr_cosp);

        // Pitch (y-axis rotation)
        let sinp = 2.0 * (q.w * q.y - q.z * q.x);
        let pitch = if sinp.abs() >= 1.0 {
            std::f64::consts::FRAC_PI_2.copysign(sinp)
        } else {
            sinp.asin()
        };

        // Yaw (z-axis rotation)
        let siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
        let cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
        let yaw = siny_cosp.atan2(cosy_cosp);

        (roll, pitch, yaw)
    }

    /// Construct from axis-angle.
    #[must_use]
    pub fn from_axis_angle(axis: &Unit<Vector3<f64>>, angle: f64) -> Self {
        let half = angle / 2.0;
        let s = half.sin();
        let c = half.cos();
        let a = &axis.0;
        Self {
            q: Quaternion::new(c, a.x * s, a.y * s, a.z * s).normalize(),
        }
    }

    /// Construct from a rotation matrix (3x3).
    #[must_use]
    pub fn from_matrix(m: &Matrix3<f64>) -> Self {
        let d = &m.data;
        let trace = d[0][0] + d[1][1] + d[2][2];

        let (w, x, y, z) = if trace > 0.0 {
            let s = (trace + 1.0).sqrt() * 2.0; // s = 4*w
            (
                0.25 * s,
                (d[2][1] - d[1][2]) / s,
                (d[0][2] - d[2][0]) / s,
                (d[1][0] - d[0][1]) / s,
            )
        } else if d[0][0] > d[1][1] && d[0][0] > d[2][2] {
            let s = (1.0 + d[0][0] - d[1][1] - d[2][2]).sqrt() * 2.0;
            (
                (d[2][1] - d[1][2]) / s,
                0.25 * s,
                (d[0][1] + d[1][0]) / s,
                (d[0][2] + d[2][0]) / s,
            )
        } else if d[1][1] > d[2][2] {
            let s = (1.0 + d[1][1] - d[0][0] - d[2][2]).sqrt() * 2.0;
            (
                (d[0][2] - d[2][0]) / s,
                (d[0][1] + d[1][0]) / s,
                0.25 * s,
                (d[1][2] + d[2][1]) / s,
            )
        } else {
            let s = (1.0 + d[2][2] - d[0][0] - d[1][1]).sqrt() * 2.0;
            (
                (d[1][0] - d[0][1]) / s,
                (d[0][2] + d[2][0]) / s,
                (d[1][2] + d[2][1]) / s,
                0.25 * s,
            )
        };

        Self {
            q: Quaternion::new(w, x, y, z).normalize(),
        }
    }

    /// SLERP interpolation.
    #[must_use]
    pub fn slerp(&self, other: &Self, t: f64) -> Self {
        let mut dot = self.q.w * other.q.w
            + self.q.x * other.q.x
            + self.q.y * other.q.y
            + self.q.z * other.q.z;

        // If the dot product is negative, negate one to take the shorter path.
        let mut other_q = other.q;
        if dot < 0.0 {
            other_q = Quaternion::new(-other_q.w, -other_q.x, -other_q.y, -other_q.z);
            dot = -dot;
        }

        // Clamp for numerical safety.
        dot = dot.min(1.0);

        if dot > 0.9995 {
            // Very close — linear interpolation then normalise.
            let result = Quaternion::new(
                self.q.w + t * (other_q.w - self.q.w),
                self.q.x + t * (other_q.x - self.q.x),
                self.q.y + t * (other_q.y - self.q.y),
                self.q.z + t * (other_q.z - self.q.z),
            );
            return Self {
                q: result.normalize(),
            };
        }

        let theta = dot.acos();
        let sin_theta = theta.sin();
        let a = ((1.0 - t) * theta).sin() / sin_theta;
        let b = (t * theta).sin() / sin_theta;

        Self {
            q: Quaternion::new(
                a * self.q.w + b * other_q.w,
                a * self.q.x + b * other_q.x,
                a * self.q.y + b * other_q.y,
                a * self.q.z + b * other_q.z,
            )
            .normalize(),
        }
    }

    /// Convert to rotation matrix.
    #[must_use]
    pub fn to_rotation_matrix(&self) -> Matrix3<f64> {
        let q = &self.q;
        let xx = q.x * q.x;
        let yy = q.y * q.y;
        let zz = q.z * q.z;
        let xy = q.x * q.y;
        let xz = q.x * q.z;
        let yz = q.y * q.z;
        let wx = q.w * q.x;
        let wy = q.w * q.y;
        let wz = q.w * q.z;

        let mut m = Matrix3::zeros();
        m.data[0][0] = 1.0 - 2.0 * (yy + zz);
        m.data[0][1] = 2.0 * (xy - wz);
        m.data[0][2] = 2.0 * (xz + wy);
        m.data[1][0] = 2.0 * (xy + wz);
        m.data[1][1] = 1.0 - 2.0 * (xx + zz);
        m.data[1][2] = 2.0 * (yz - wx);
        m.data[2][0] = 2.0 * (xz - wy);
        m.data[2][1] = 2.0 * (yz + wx);
        m.data[2][2] = 1.0 - 2.0 * (xx + yy);
        m
    }
}

/// `UnitQuaternion * Vector3` — rotate a vector.
impl Mul<Vector3<f64>> for UnitQuaternion<f64> {
    type Output = Vector3<f64>;
    fn mul(self, v: Vector3<f64>) -> Vector3<f64> {
        let qv = Quaternion::new(0.0, v.x, v.y, v.z);
        let result = self.q * qv * self.q.conjugate();
        Vector3::new(result.x, result.y, result.z)
    }
}

/// `&UnitQuaternion * Vector3`
impl Mul<Vector3<f64>> for &UnitQuaternion<f64> {
    type Output = Vector3<f64>;
    fn mul(self, v: Vector3<f64>) -> Vector3<f64> {
        (*self) * v
    }
}

/// `UnitQuaternion * UnitQuaternion` — compose rotations.
impl Mul for UnitQuaternion<f64> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self {
            q: (self.q * rhs.q).normalize(),
        }
    }
}

/// `UnitQuaternion *= UnitQuaternion`
impl std::ops::MulAssign for UnitQuaternion<f64> {
    fn mul_assign(&mut self, rhs: Self) {
        self.q = (self.q * rhs.q).normalize();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_rotation() {
        let q = UnitQuaternion::identity();
        let v = Vector3::new(1.0, 0.0, 0.0);
        let rotated = q * v;
        assert!((rotated.x - 1.0).abs() < 1e-10);
        assert!(rotated.y.abs() < 1e-10);
    }

    #[test]
    fn test_slerp_endpoints() {
        let a = UnitQuaternion::identity();
        let b = UnitQuaternion::from_euler_angles(0.5, 0.0, 0.0);
        let at0 = a.slerp(&b, 0.0);
        let at1 = a.slerp(&b, 1.0);
        assert!((at0.q.w - a.q.w).abs() < 1e-6);
        assert!((at1.q.w - b.q.w).abs() < 1e-6);
    }

    #[test]
    fn test_euler_roundtrip() {
        let q = UnitQuaternion::from_euler_angles(0.1, 0.2, 0.3);
        let (r, p, y) = q.euler_angles();
        assert!((r - 0.1).abs() < 1e-6);
        assert!((p - 0.2).abs() < 1e-6);
        assert!((y - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_from_matrix_identity() {
        let m = Matrix3::identity();
        let q = UnitQuaternion::from_matrix(&m);
        assert!((q.q.w - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_axis_angle_90_deg() {
        let axis = Unit::new_normalize(Vector3::new(0.0, 0.0, 1.0));
        let q = UnitQuaternion::from_axis_angle(&axis, std::f64::consts::FRAC_PI_2);
        let v = q * Vector3::new(1.0, 0.0, 0.0);
        assert!(v.x.abs() < 1e-6);
        assert!((v.y - 1.0).abs() < 1e-6);
    }
}
