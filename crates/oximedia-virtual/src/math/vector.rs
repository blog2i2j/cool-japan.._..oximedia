//! Vector and point types for 3D geometry.

use serde::{Deserialize, Serialize};
use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

// ---------------------------------------------------------------------------
// Point2
// ---------------------------------------------------------------------------

/// A 2D point.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Point2<T> {
    pub x: T,
    pub y: T,
}

impl<T: Default> Point2<T> {
    /// Origin (0, 0).
    #[must_use]
    pub fn origin() -> Self {
        Self {
            x: T::default(),
            y: T::default(),
        }
    }
}

impl<T> Point2<T> {
    /// Construct from components.
    #[must_use]
    pub fn new(x: T, y: T) -> Self {
        Self { x, y }
    }
}

// ---------------------------------------------------------------------------
// Vector3
// ---------------------------------------------------------------------------

/// A 3-component vector.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Vector3<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

impl<T: Default> Vector3<T> {
    /// Zero vector.
    #[must_use]
    pub fn zeros() -> Self {
        Self {
            x: T::default(),
            y: T::default(),
            z: T::default(),
        }
    }
}

impl<T> Vector3<T> {
    /// Construct from components.
    #[must_use]
    pub fn new(x: T, y: T, z: T) -> Self {
        Self { x, y, z }
    }
}

impl Vector3<f64> {
    /// Euclidean norm.
    #[must_use]
    pub fn norm(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Normalise to unit length.  Returns zero vector if norm is near zero.
    #[must_use]
    pub fn normalize(&self) -> Self {
        let n = self.norm();
        if n < 1e-15 {
            Self::zeros()
        } else {
            Self::new(self.x / n, self.y / n, self.z / n)
        }
    }

    /// Dot product.
    #[must_use]
    pub fn dot(&self, rhs: &Self) -> f64 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    /// Cross product.
    #[must_use]
    pub fn cross(&self, rhs: &Self) -> Self {
        Self::new(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
    }

    /// Outer product (self * rhs^T), yielding a 3x3 matrix.
    #[must_use]
    pub fn outer(&self, rhs: &Self) -> super::Matrix3<f64> {
        let mut m = super::Matrix3::zeros();
        m.data[0][0] = self.x * rhs.x;
        m.data[0][1] = self.x * rhs.y;
        m.data[0][2] = self.x * rhs.z;
        m.data[1][0] = self.y * rhs.x;
        m.data[1][1] = self.y * rhs.y;
        m.data[1][2] = self.y * rhs.z;
        m.data[2][0] = self.z * rhs.x;
        m.data[2][1] = self.z * rhs.y;
        m.data[2][2] = self.z * rhs.z;
        m
    }

    /// Return the 1x3 transpose as a `TransposedVector3` for nalgebra-like
    /// `centered * world_centered.transpose()` patterns producing a Matrix3.
    #[must_use]
    pub fn transpose(&self) -> TransposedVector3 {
        TransposedVector3(*self)
    }

    /// Scale in place (matches nalgebra's `scale_mut` on vector views).
    pub fn scale_mut(&mut self, s: f64) {
        self.x *= s;
        self.y *= s;
        self.z *= s;
    }
}

/// A transposed 3-vector (row vector) used for outer-product patterns.
#[derive(Debug, Clone, Copy)]
pub struct TransposedVector3(pub Vector3<f64>);

/// `Vector3 * TransposedVector3 → Matrix3` (outer product).
impl Mul<TransposedVector3> for Vector3<f64> {
    type Output = super::Matrix3<f64>;
    fn mul(self, rhs: TransposedVector3) -> Self::Output {
        self.outer(&rhs.0)
    }
}

// -- Arithmetic impls for Vector3<f64> --

impl Add for Vector3<f64> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl Sub for Vector3<f64> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl Neg for Vector3<f64> {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.x, -self.y, -self.z)
    }
}

impl Mul<f64> for Vector3<f64> {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

impl Mul<Vector3<f64>> for f64 {
    type Output = Vector3<f64>;
    fn mul(self, rhs: Vector3<f64>) -> Vector3<f64> {
        Vector3::new(self * rhs.x, self * rhs.y, self * rhs.z)
    }
}

impl Div<f64> for Vector3<f64> {
    type Output = Self;
    fn div(self, rhs: f64) -> Self {
        Self::new(self.x / rhs, self.y / rhs, self.z / rhs)
    }
}

impl AddAssign for Vector3<f64> {
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

impl SubAssign for Vector3<f64> {
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
        self.z -= rhs.z;
    }
}

impl MulAssign<f64> for Vector3<f64> {
    fn mul_assign(&mut self, rhs: f64) {
        self.x *= rhs;
        self.y *= rhs;
        self.z *= rhs;
    }
}

impl DivAssign<f64> for Vector3<f64> {
    fn div_assign(&mut self, rhs: f64) {
        self.x /= rhs;
        self.y /= rhs;
        self.z /= rhs;
    }
}

// ---------------------------------------------------------------------------
// Point3
// ---------------------------------------------------------------------------

/// A 3D point (distinct from Vector3 to model affine semantics).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Point3<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

impl<T: Default> Point3<T> {
    /// Origin point.
    #[must_use]
    pub fn origin() -> Self {
        Self {
            x: T::default(),
            y: T::default(),
            z: T::default(),
        }
    }
}

impl<T> Point3<T> {
    /// Construct from components.
    #[must_use]
    pub fn new(x: T, y: T, z: T) -> Self {
        Self { x, y, z }
    }
}

impl Point3<f64> {
    /// Coords as a Vector3.
    #[must_use]
    pub fn coords(&self) -> Vector3<f64> {
        Vector3::new(self.x, self.y, self.z)
    }

    /// Construct from a Vector3.
    #[must_use]
    pub fn from_vec(v: Vector3<f64>) -> Self {
        Self::new(v.x, v.y, v.z)
    }

    /// Convert to homogeneous coordinates (4-element column vector).
    #[must_use]
    pub fn to_homogeneous(&self) -> [f64; 4] {
        [self.x, self.y, self.z, 1.0]
    }

    /// Create from homogeneous coordinates.  Returns `None` if w is near zero.
    #[must_use]
    pub fn from_homogeneous(h: [f64; 4]) -> Option<Self> {
        if h[3].abs() < 1e-15 {
            None
        } else {
            Some(Self::new(h[0] / h[3], h[1] / h[3], h[2] / h[3]))
        }
    }
}

/// `Point3 - Point3 → Vector3`
impl Sub for Point3<f64> {
    type Output = Vector3<f64>;
    fn sub(self, rhs: Self) -> Vector3<f64> {
        Vector3::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

/// `&Point3 - &Point3 → Vector3`
impl<'a, 'b> Sub<&'b Point3<f64>> for &'a Point3<f64> {
    type Output = Vector3<f64>;
    fn sub(self, rhs: &'b Point3<f64>) -> Vector3<f64> {
        Vector3::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

/// `Point3 + Vector3 → Point3`
impl Add<Vector3<f64>> for Point3<f64> {
    type Output = Point3<f64>;
    fn add(self, rhs: Vector3<f64>) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

/// `Point3 += Vector3`
impl AddAssign<Vector3<f64>> for Point3<f64> {
    fn add_assign(&mut self, rhs: Vector3<f64>) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

/// `Point3 - Vector3 → Point3`
impl Sub<Vector3<f64>> for Point3<f64> {
    type Output = Point3<f64>;
    fn sub(self, rhs: Vector3<f64>) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

/// `Point3 * f64` — scale coordinates (matches nalgebra pattern).
impl Mul<f64> for Point3<f64> {
    type Output = Vector3<f64>;
    fn mul(self, rhs: f64) -> Vector3<f64> {
        Vector3::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

/// Construct Point3 from Vector3 (via From trait).
impl From<Vector3<f64>> for Point3<f64> {
    fn from(v: Vector3<f64>) -> Self {
        Self::new(v.x, v.y, v.z)
    }
}

// ---------------------------------------------------------------------------
// Vector6
// ---------------------------------------------------------------------------

/// A 6-component vector (used for Kalman state: [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z]).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Vector6<T> {
    pub data: [T; 6],
}

impl<T: Default + Copy> Vector6<T> {
    /// Zero vector.
    #[must_use]
    pub fn zeros() -> Self {
        Self {
            data: [T::default(); 6],
        }
    }
}

impl<T> Index<usize> for Vector6<T> {
    type Output = T;
    fn index(&self, i: usize) -> &T {
        &self.data[i]
    }
}

impl<T> IndexMut<usize> for Vector6<T> {
    fn index_mut(&mut self, i: usize) -> &mut T {
        &mut self.data[i]
    }
}

impl Add for Vector6<f64> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let mut out = self;
        for i in 0..6 {
            out.data[i] += rhs.data[i];
        }
        out
    }
}

impl Sub for Vector6<f64> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        let mut out = self;
        for i in 0..6 {
            out.data[i] -= rhs.data[i];
        }
        out
    }
}

impl AddAssign for Vector6<f64> {
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..6 {
            self.data[i] += rhs.data[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector3_norm() {
        let v = Vector3::new(3.0, 4.0, 0.0);
        assert!((v.norm() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_vector3_normalize() {
        let v = Vector3::new(0.0, 0.0, 5.0);
        let n = v.normalize();
        assert!((n.z - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_vector3_cross() {
        let a = Vector3::new(1.0, 0.0, 0.0);
        let b = Vector3::new(0.0, 1.0, 0.0);
        let c = a.cross(&b);
        assert!((c.z - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_vector3_dot() {
        let a = Vector3::new(1.0, 2.0, 3.0);
        let b = Vector3::new(4.0, 5.0, 6.0);
        assert!((a.dot(&b) - 32.0).abs() < 1e-10);
    }

    #[test]
    fn test_point3_sub() {
        let a = Point3::new(3.0, 4.0, 5.0);
        let b = Point3::new(1.0, 1.0, 1.0);
        let v = a - b;
        assert!((v.x - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_point3_add_vector() {
        let p = Point3::new(1.0, 2.0, 3.0);
        let v = Vector3::new(10.0, 20.0, 30.0);
        let q = p + v;
        assert!((q.x - 11.0).abs() < 1e-10);
    }

    #[test]
    fn test_point3_homogeneous_roundtrip() {
        let p = Point3::new(1.0, 2.0, 3.0);
        let h = p.to_homogeneous();
        let p2 = Point3::from_homogeneous(h).expect("should succeed in test");
        assert!((p.x - p2.x).abs() < 1e-10);
    }

    #[test]
    fn test_vector6_indexing() {
        let mut v: Vector6<f64> = Vector6::zeros();
        v[3] = 42.0;
        assert!((v[3] - 42.0).abs() < 1e-10);
    }
}
