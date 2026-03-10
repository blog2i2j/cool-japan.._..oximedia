//! Matrix types for 3D geometry and Kalman filtering.

use super::vector::{Vector3, Vector6};
use serde::{Deserialize, Serialize};
use std::ops::{Add, AddAssign, Index, IndexMut, Mul, Sub};

// ---------------------------------------------------------------------------
// Matrix3  (row-major 3x3)
// ---------------------------------------------------------------------------

/// A 3x3 matrix stored in row-major order.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Matrix3<T> {
    /// `data[row][col]`
    pub data: [[T; 3]; 3],
}

impl<T: Default + Copy> Matrix3<T> {
    /// All-zero matrix.
    #[must_use]
    pub fn zeros() -> Self {
        Self {
            data: [[T::default(); 3]; 3],
        }
    }
}

impl Matrix3<f64> {
    /// Identity.
    #[must_use]
    pub fn identity() -> Self {
        let mut m = Self::zeros();
        m.data[0][0] = 1.0;
        m.data[1][1] = 1.0;
        m.data[2][2] = 1.0;
        m
    }

    /// Build from three column vectors (nalgebra compat).
    #[must_use]
    pub fn from_columns(c0: &Vector3<f64>, c1: &Vector3<f64>, c2: &Vector3<f64>) -> Self {
        let mut m = Self::zeros();
        m.data[0][0] = c0.x;
        m.data[1][0] = c0.y;
        m.data[2][0] = c0.z;
        m.data[0][1] = c1.x;
        m.data[1][1] = c1.y;
        m.data[2][1] = c1.z;
        m.data[0][2] = c2.x;
        m.data[1][2] = c2.y;
        m.data[2][2] = c2.z;
        m
    }

    /// Build from a column-major iterator (nalgebra compat: column-major flat).
    #[must_use]
    pub fn from_iterator<I: IntoIterator<Item = f64>>(iter: I) -> Self {
        let elems: Vec<f64> = iter.into_iter().collect();
        let mut m = Self::zeros();
        // nalgebra stores column-major
        for (idx, val) in elems.iter().enumerate() {
            let col = idx / 3;
            let row = idx % 3;
            if row < 3 && col < 3 {
                m.data[row][col] = *val;
            }
        }
        m
    }

    /// Transpose.
    #[must_use]
    pub fn transpose(&self) -> Self {
        let mut m = Self::zeros();
        for r in 0..3 {
            for c in 0..3 {
                m.data[c][r] = self.data[r][c];
            }
        }
        m
    }

    /// Determinant.
    #[must_use]
    pub fn determinant(&self) -> f64 {
        let d = &self.data;
        d[0][0] * (d[1][1] * d[2][2] - d[1][2] * d[2][1])
            - d[0][1] * (d[1][0] * d[2][2] - d[1][2] * d[2][0])
            + d[0][2] * (d[1][0] * d[2][1] - d[1][1] * d[2][0])
    }

    /// Try to invert.
    #[must_use]
    pub fn try_inverse(&self) -> Option<Self> {
        let det = self.determinant();
        if det.abs() < 1e-15 {
            return None;
        }
        let inv_det = 1.0 / det;
        let d = &self.data;
        let mut m = Self::zeros();
        m.data[0][0] = (d[1][1] * d[2][2] - d[1][2] * d[2][1]) * inv_det;
        m.data[0][1] = (d[0][2] * d[2][1] - d[0][1] * d[2][2]) * inv_det;
        m.data[0][2] = (d[0][1] * d[1][2] - d[0][2] * d[1][1]) * inv_det;
        m.data[1][0] = (d[1][2] * d[2][0] - d[1][0] * d[2][2]) * inv_det;
        m.data[1][1] = (d[0][0] * d[2][2] - d[0][2] * d[2][0]) * inv_det;
        m.data[1][2] = (d[0][2] * d[1][0] - d[0][0] * d[1][2]) * inv_det;
        m.data[2][0] = (d[1][0] * d[2][1] - d[1][1] * d[2][0]) * inv_det;
        m.data[2][1] = (d[0][1] * d[2][0] - d[0][0] * d[2][1]) * inv_det;
        m.data[2][2] = (d[0][0] * d[1][1] - d[0][1] * d[1][0]) * inv_det;
        Some(m)
    }

    /// SVD decomposition. Returns `(u, sigma, v_t)` where each is `Option<Matrix3>` for
    /// compatibility with nalgebra's SVD API.  Uses Jacobi one-sided SVD.
    #[must_use]
    pub fn svd(&self, compute_u: bool, compute_v: bool) -> Svd3 {
        svd_3x3(self, compute_u, compute_v)
    }

    /// Column as Vector3.
    #[must_use]
    pub fn column(&self, c: usize) -> Vector3<f64> {
        Vector3::new(self.data[0][c], self.data[1][c], self.data[2][c])
    }

    /// Multiply by Vector3.
    #[must_use]
    pub fn mul_vec(&self, v: &Vector3<f64>) -> Vector3<f64> {
        Vector3::new(
            self.data[0][0] * v.x + self.data[0][1] * v.y + self.data[0][2] * v.z,
            self.data[1][0] * v.x + self.data[1][1] * v.y + self.data[1][2] * v.z,
            self.data[2][0] * v.x + self.data[2][1] * v.y + self.data[2][2] * v.z,
        )
    }
}

impl Index<(usize, usize)> for Matrix3<f64> {
    type Output = f64;
    fn index(&self, (r, c): (usize, usize)) -> &f64 {
        &self.data[r][c]
    }
}

impl IndexMut<(usize, usize)> for Matrix3<f64> {
    fn index_mut(&mut self, (r, c): (usize, usize)) -> &mut f64 {
        &mut self.data[r][c]
    }
}

impl Mul for Matrix3<f64> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let mut m = Self::zeros();
        for r in 0..3 {
            for c in 0..3 {
                for k in 0..3 {
                    m.data[r][c] += self.data[r][k] * rhs.data[k][c];
                }
            }
        }
        m
    }
}

impl Mul<f64> for Matrix3<f64> {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        let mut m = self;
        for r in 0..3 {
            for c in 0..3 {
                m.data[r][c] *= rhs;
            }
        }
        m
    }
}

impl Mul<Matrix3<f64>> for f64 {
    type Output = Matrix3<f64>;
    fn mul(self, rhs: Matrix3<f64>) -> Matrix3<f64> {
        rhs * self
    }
}

impl Add for Matrix3<f64> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let mut m = self;
        for r in 0..3 {
            for c in 0..3 {
                m.data[r][c] += rhs.data[r][c];
            }
        }
        m
    }
}

impl AddAssign for Matrix3<f64> {
    fn add_assign(&mut self, rhs: Self) {
        for r in 0..3 {
            for c in 0..3 {
                self.data[r][c] += rhs.data[r][c];
            }
        }
    }
}

/// `Matrix3 * Vector3 → Vector3`
impl Mul<Vector3<f64>> for Matrix3<f64> {
    type Output = Vector3<f64>;
    fn mul(self, rhs: Vector3<f64>) -> Vector3<f64> {
        self.mul_vec(&rhs)
    }
}

/// `Matrix3 * Point3 → Point3` (transform point).
impl Mul<super::vector::Point3<f64>> for Matrix3<f64> {
    type Output = super::vector::Point3<f64>;
    fn mul(self, rhs: super::vector::Point3<f64>) -> super::vector::Point3<f64> {
        let v = self.mul_vec(&rhs.coords());
        super::vector::Point3::new(v.x, v.y, v.z)
    }
}

// ---------------------------------------------------------------------------
// SVD for 3x3 (Jacobi)
// ---------------------------------------------------------------------------

/// SVD result for 3x3.
pub struct Svd3 {
    pub u: Option<Matrix3<f64>>,
    pub singular_values: [f64; 3],
    pub v_t: Option<Matrix3<f64>>,
}

/// Compute SVD of a 3x3 matrix via one-sided Jacobi rotations.
fn svd_3x3(a: &Matrix3<f64>, compute_u: bool, compute_v: bool) -> Svd3 {
    // We compute A^T A, find its eigenvalues/vectors, then derive U.
    let ata = a.transpose() * *a;

    // Jacobi eigendecomposition of symmetric 3x3 matrix.
    let (eigenvalues, eigenvectors) = jacobi_eigen_3x3(&ata);

    // Singular values are sqrt of eigenvalues.
    let mut sigma = [0.0f64; 3];
    for i in 0..3 {
        sigma[i] = eigenvalues[i].max(0.0).sqrt();
    }

    // V = eigenvectors (columns of V).
    let v = eigenvectors;
    let v_t = v.transpose();

    // U = A * V * Sigma^-1
    let u_mat = if compute_u {
        let mut u = Matrix3::zeros();
        for col in 0..3 {
            if sigma[col] > 1e-14 {
                let v_col = v.column(col);
                let av = a.mul_vec(&v_col);
                let inv_s = 1.0 / sigma[col];
                u.data[0][col] = av.x * inv_s;
                u.data[1][col] = av.y * inv_s;
                u.data[2][col] = av.z * inv_s;
            }
        }
        Some(u)
    } else {
        None
    };

    Svd3 {
        u: u_mat,
        singular_values: sigma,
        v_t: if compute_v { Some(v_t) } else { None },
    }
}

/// Jacobi eigendecomposition of a symmetric 3x3 matrix.
/// Returns (eigenvalues sorted descending, eigenvector matrix with columns = eigenvectors).
fn jacobi_eigen_3x3(m: &Matrix3<f64>) -> ([f64; 3], Matrix3<f64>) {
    let mut a = *m;
    let mut v = Matrix3::identity();

    for _ in 0..100 {
        // Find largest off-diagonal.
        let mut max_val = 0.0f64;
        let mut p = 0;
        let mut q = 1;
        for i in 0..3 {
            for j in (i + 1)..3 {
                if a.data[i][j].abs() > max_val {
                    max_val = a.data[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if max_val < 1e-14 {
            break;
        }

        // Compute Jacobi rotation angle.
        let app = a.data[p][p];
        let aqq = a.data[q][q];
        let apq = a.data[p][q];
        let theta = if (app - aqq).abs() < 1e-15 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * apq / (app - aqq)).atan()
        };

        let c = theta.cos();
        let s = theta.sin();

        // Apply rotation to A: A' = G^T A G
        let mut new_a = a;
        // Update rows/cols p and q.
        for i in 0..3 {
            if i != p && i != q {
                new_a.data[i][p] = c * a.data[i][p] + s * a.data[i][q];
                new_a.data[p][i] = new_a.data[i][p];
                new_a.data[i][q] = -s * a.data[i][p] + c * a.data[i][q];
                new_a.data[q][i] = new_a.data[i][q];
            }
        }
        new_a.data[p][p] = c * c * app + 2.0 * c * s * apq + s * s * aqq;
        new_a.data[q][q] = s * s * app - 2.0 * c * s * apq + c * c * aqq;
        new_a.data[p][q] = 0.0;
        new_a.data[q][p] = 0.0;
        a = new_a;

        // Update eigenvector matrix: V' = V * G
        let mut new_v = v;
        for i in 0..3 {
            new_v.data[i][p] = c * v.data[i][p] + s * v.data[i][q];
            new_v.data[i][q] = -s * v.data[i][p] + c * v.data[i][q];
        }
        v = new_v;
    }

    // Sort eigenvalues descending.
    let mut eigenvalues = [a.data[0][0], a.data[1][1], a.data[2][2]];
    let mut indices = [0usize, 1, 2];
    // Simple sort.
    for i in 0..3 {
        for j in (i + 1)..3 {
            if eigenvalues[j] > eigenvalues[i] {
                eigenvalues.swap(i, j);
                indices.swap(i, j);
            }
        }
    }

    // Reorder columns of V.
    let mut v_sorted = Matrix3::zeros();
    for col in 0..3 {
        let src = indices[col];
        for row in 0..3 {
            v_sorted.data[row][col] = v.data[row][src];
        }
    }

    (eigenvalues, v_sorted)
}

// ---------------------------------------------------------------------------
// Matrix4  (row-major 4x4)
// ---------------------------------------------------------------------------

/// A 4x4 matrix stored in row-major order.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Matrix4<T> {
    pub data: [[T; 4]; 4],
}

impl<T: Default + Copy> Matrix4<T> {
    /// All-zero matrix.
    #[must_use]
    pub fn zeros() -> Self {
        Self {
            data: [[T::default(); 4]; 4],
        }
    }
}

impl Matrix4<f64> {
    /// 4x4 identity.
    #[must_use]
    pub fn identity() -> Self {
        let mut m = Self::zeros();
        for i in 0..4 {
            m.data[i][i] = 1.0;
        }
        m
    }

    /// Transpose.
    #[must_use]
    pub fn transpose(&self) -> Self {
        let mut m = Self::zeros();
        for r in 0..4 {
            for c in 0..4 {
                m.data[c][r] = self.data[r][c];
            }
        }
        m
    }

    /// Try to invert (Gauss-Jordan).
    #[must_use]
    pub fn try_inverse(&self) -> Option<Self> {
        let mut aug = [[0.0f64; 8]; 4];
        for r in 0..4 {
            for c in 0..4 {
                aug[r][c] = self.data[r][c];
                aug[r][c + 4] = if r == c { 1.0 } else { 0.0 };
            }
        }
        for col in 0..4 {
            // Partial pivot.
            let mut max_row = col;
            let mut max_val = aug[col][col].abs();
            for row in (col + 1)..4 {
                if aug[row][col].abs() > max_val {
                    max_row = row;
                    max_val = aug[row][col].abs();
                }
            }
            if max_val < 1e-15 {
                return None;
            }
            aug.swap(col, max_row);
            let pivot = aug[col][col];
            for k in 0..8 {
                aug[col][k] /= pivot;
            }
            for row in 0..4 {
                if row == col {
                    continue;
                }
                let factor = aug[row][col];
                for k in 0..8 {
                    aug[row][k] -= factor * aug[col][k];
                }
            }
        }
        let mut m = Self::zeros();
        for r in 0..4 {
            for c in 0..4 {
                m.data[r][c] = aug[r][c + 4];
            }
        }
        Some(m)
    }

    /// Fixed 3x3 sub-view (read-only) at (r0, c0). Returns Matrix3.
    #[must_use]
    pub fn fixed_view_3x3(&self, r0: usize, c0: usize) -> Matrix3<f64> {
        let mut m = Matrix3::zeros();
        for r in 0..3 {
            for c in 0..3 {
                m.data[r][c] = self.data[r0 + r][c0 + c];
            }
        }
        m
    }

    /// Copy a Matrix3 into the 3x3 sub-block at (r0, c0).
    pub fn set_block_3x3(&mut self, r0: usize, c0: usize, src: &Matrix3<f64>) {
        for r in 0..3 {
            for c in 0..3 {
                self.data[r0 + r][c0 + c] = src.data[r][c];
            }
        }
    }

    /// Copy a Vector3 into the 3x1 sub-block at (r0, c0).
    pub fn set_block_3x1(&mut self, r0: usize, c0: usize, src: &Vector3<f64>) {
        self.data[r0][c0] = src.x;
        self.data[r0 + 1][c0] = src.y;
        self.data[r0 + 2][c0] = src.z;
    }

    /// Get 3x1 sub-block as Vector3.
    #[must_use]
    pub fn get_block_3x1(&self, r0: usize, c0: usize) -> Vector3<f64> {
        Vector3::new(
            self.data[r0][c0],
            self.data[r0 + 1][c0],
            self.data[r0 + 2][c0],
        )
    }

    /// Multiply this matrix by a 4-element homogeneous vector.
    #[must_use]
    pub fn mul_homogeneous(&self, v: &[f64; 4]) -> [f64; 4] {
        let mut out = [0.0f64; 4];
        for r in 0..4 {
            for c in 0..4 {
                out[r] += self.data[r][c] * v[c];
            }
        }
        out
    }
}

impl Index<(usize, usize)> for Matrix4<f64> {
    type Output = f64;
    fn index(&self, (r, c): (usize, usize)) -> &f64 {
        &self.data[r][c]
    }
}

impl IndexMut<(usize, usize)> for Matrix4<f64> {
    fn index_mut(&mut self, (r, c): (usize, usize)) -> &mut f64 {
        &mut self.data[r][c]
    }
}

impl Mul for Matrix4<f64> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let mut m = Self::zeros();
        for r in 0..4 {
            for c in 0..4 {
                for k in 0..4 {
                    m.data[r][c] += self.data[r][k] * rhs.data[k][c];
                }
            }
        }
        m
    }
}

/// `Matrix4 * [f64; 4]` — homogeneous transform.
impl Mul<[f64; 4]> for Matrix4<f64> {
    type Output = [f64; 4];
    fn mul(self, rhs: [f64; 4]) -> [f64; 4] {
        self.mul_homogeneous(&rhs)
    }
}

// ---------------------------------------------------------------------------
// Matrix6  (row-major 6x6)
// ---------------------------------------------------------------------------

/// A 6x6 matrix (used for Kalman covariance).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Matrix6<T> {
    pub data: [[T; 6]; 6],
}

impl<T: Default + Copy> Matrix6<T> {
    /// All-zero matrix.
    #[must_use]
    pub fn zeros() -> Self {
        Self {
            data: [[T::default(); 6]; 6],
        }
    }
}

impl Matrix6<f64> {
    /// 6x6 identity.
    #[must_use]
    pub fn identity() -> Self {
        let mut m = Self::zeros();
        for i in 0..6 {
            m.data[i][i] = 1.0;
        }
        m
    }

    /// Transpose.
    #[must_use]
    pub fn transpose(&self) -> Self {
        let mut m = Self::zeros();
        for r in 0..6 {
            for c in 0..6 {
                m.data[c][r] = self.data[r][c];
            }
        }
        m
    }
}

impl Index<(usize, usize)> for Matrix6<f64> {
    type Output = f64;
    fn index(&self, (r, c): (usize, usize)) -> &f64 {
        &self.data[r][c]
    }
}

impl IndexMut<(usize, usize)> for Matrix6<f64> {
    fn index_mut(&mut self, (r, c): (usize, usize)) -> &mut f64 {
        &mut self.data[r][c]
    }
}

/// `Matrix6 * f64`
impl Mul<f64> for Matrix6<f64> {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        let mut m = self;
        for r in 0..6 {
            for c in 0..6 {
                m.data[r][c] *= rhs;
            }
        }
        m
    }
}

/// `Matrix6 * Matrix6`
impl Mul for Matrix6<f64> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let mut m = Self::zeros();
        for r in 0..6 {
            for c in 0..6 {
                for k in 0..6 {
                    m.data[r][c] += self.data[r][k] * rhs.data[k][c];
                }
            }
        }
        m
    }
}

/// `Matrix6 * Vector6`
impl Mul<Vector6<f64>> for Matrix6<f64> {
    type Output = Vector6<f64>;
    fn mul(self, rhs: Vector6<f64>) -> Vector6<f64> {
        let mut out = Vector6::zeros();
        for r in 0..6 {
            for c in 0..6 {
                out.data[r] += self.data[r][c] * rhs.data[c];
            }
        }
        out
    }
}

impl Add for Matrix6<f64> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let mut m = self;
        for r in 0..6 {
            for c in 0..6 {
                m.data[r][c] += rhs.data[r][c];
            }
        }
        m
    }
}

impl Sub for Matrix6<f64> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        let mut m = self;
        for r in 0..6 {
            for c in 0..6 {
                m.data[r][c] -= rhs.data[r][c];
            }
        }
        m
    }
}

// ---------------------------------------------------------------------------
// Matrix3x6  (3 rows, 6 cols) — for Kalman observation matrix
// ---------------------------------------------------------------------------

/// A 3x6 matrix (Kalman observation / measurement matrix).
#[derive(Debug, Clone, Copy)]
pub struct Matrix3x6 {
    pub data: [[f64; 6]; 3],
}

impl Matrix3x6 {
    /// Build from a column-major flat iterator (nalgebra compat).
    #[must_use]
    pub fn from_iterator<I: IntoIterator<Item = f64>>(iter: I) -> Self {
        let elems: Vec<f64> = iter.into_iter().collect();
        let mut m = Self {
            data: [[0.0; 6]; 3],
        };
        // nalgebra column-major: index = col * nrows + row
        for (idx, val) in elems.iter().enumerate() {
            let col = idx / 3;
            let row = idx % 3;
            if row < 3 && col < 6 {
                m.data[row][col] = *val;
            }
        }
        m
    }

    /// Multiply by Vector6 → Vector3.
    #[must_use]
    pub fn mul_vec6(&self, v: &Vector6<f64>) -> Vector3<f64> {
        let mut out = [0.0f64; 3];
        for r in 0..3 {
            for c in 0..6 {
                out[r] += self.data[r][c] * v.data[c];
            }
        }
        Vector3::new(out[0], out[1], out[2])
    }

    /// Transpose → 6x3 stored as `Matrix6x3`.
    #[must_use]
    pub fn transpose(&self) -> Matrix6x3 {
        let mut m = Matrix6x3 {
            data: [[0.0; 3]; 6],
        };
        for r in 0..3 {
            for c in 0..6 {
                m.data[c][r] = self.data[r][c];
            }
        }
        m
    }
}

/// `Matrix3x6 * Vector6 → Vector3`
impl Mul<Vector6<f64>> for Matrix3x6 {
    type Output = Vector3<f64>;
    fn mul(self, rhs: Vector6<f64>) -> Vector3<f64> {
        self.mul_vec6(&rhs)
    }
}

/// `Matrix3x6 * Matrix6 → Matrix3x6` (H * P)
impl Mul<Matrix6<f64>> for Matrix3x6 {
    type Output = Self;
    fn mul(self, rhs: Matrix6<f64>) -> Self {
        let mut m = Self {
            data: [[0.0; 6]; 3],
        };
        for r in 0..3 {
            for c in 0..6 {
                for k in 0..6 {
                    m.data[r][c] += self.data[r][k] * rhs.data[k][c];
                }
            }
        }
        m
    }
}

/// `Matrix3x6 * Matrix6x3 → Matrix3` (H * P * H^T)
impl Mul<Matrix6x3> for Matrix3x6 {
    type Output = Matrix3<f64>;
    fn mul(self, rhs: Matrix6x3) -> Matrix3<f64> {
        let mut m = Matrix3::zeros();
        for r in 0..3 {
            for c in 0..3 {
                for k in 0..6 {
                    m.data[r][c] += self.data[r][k] * rhs.data[k][c];
                }
            }
        }
        m
    }
}

// ---------------------------------------------------------------------------
// Matrix6x3  (6 rows, 3 cols) — transpose of Matrix3x6
// ---------------------------------------------------------------------------

/// A 6x3 matrix (transpose of Kalman observation matrix).
#[derive(Debug, Clone, Copy)]
pub struct Matrix6x3 {
    pub data: [[f64; 3]; 6],
}

/// `Matrix6x3 * Matrix3 → Matrix6x3` (P * H^T * S^{-1})
impl Mul<Matrix3<f64>> for Matrix6x3 {
    type Output = Self;
    fn mul(self, rhs: Matrix3<f64>) -> Self {
        let mut m = Self {
            data: [[0.0; 3]; 6],
        };
        for r in 0..6 {
            for c in 0..3 {
                for k in 0..3 {
                    m.data[r][c] += self.data[r][k] * rhs.data[k][c];
                }
            }
        }
        m
    }
}

/// `Matrix6<f64> * Matrix6x3 → Matrix6x3` (P * H^T)
impl Mul<Matrix6x3> for Matrix6<f64> {
    type Output = Matrix6x3;
    fn mul(self, rhs: Matrix6x3) -> Matrix6x3 {
        let mut m = Matrix6x3 {
            data: [[0.0; 3]; 6],
        };
        for r in 0..6 {
            for c in 0..3 {
                for k in 0..6 {
                    m.data[r][c] += self.data[r][k] * rhs.data[k][c];
                }
            }
        }
        m
    }
}

// ---------------------------------------------------------------------------
// KalmanGain (6x3) — helper type for Kalman gain K
// ---------------------------------------------------------------------------

/// Kalman gain is a 6x3 matrix.  We reuse `Matrix6x3` but provide extra operations.
impl Matrix6x3 {
    /// `K * innovation(3x1) → Vector6`
    #[must_use]
    pub fn mul_vec3(&self, v: &Vector3<f64>) -> Vector6<f64> {
        let mut out = Vector6::zeros();
        let arr = [v.x, v.y, v.z];
        for r in 0..6 {
            for c in 0..3 {
                out.data[r] += self.data[r][c] * arr[c];
            }
        }
        out
    }

    /// `K(6x3) * H(3x6) → Matrix6`
    #[must_use]
    pub fn mul_3x6(&self, rhs: &Matrix3x6) -> Matrix6<f64> {
        let mut m = Matrix6::zeros();
        for r in 0..6 {
            for c in 0..6 {
                for k in 0..3 {
                    m.data[r][c] += self.data[r][k] * rhs.data[k][c];
                }
            }
        }
        m
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix3_identity_determinant() {
        let m = Matrix3::identity();
        assert!((m.determinant() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrix3_inverse() {
        let m = Matrix3::identity();
        let inv = m.try_inverse().expect("should succeed in test");
        assert!((inv.data[0][0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrix4_inverse() {
        let m = Matrix4::identity();
        let inv = m.try_inverse().expect("should succeed in test");
        for i in 0..4 {
            assert!((inv.data[i][i] - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_matrix6_mul_identity() {
        let m = Matrix6::identity();
        let result = m * m;
        for i in 0..6 {
            assert!((result.data[i][i] - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_svd_identity() {
        let m = Matrix3::identity();
        let svd = m.svd(true, true);
        for i in 0..3 {
            assert!((svd.singular_values[i] - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_matrix3x6_mul_vec6() {
        // Column-major layout: col0=[1,0,0], col1=[0,1,0], col2=[0,0,1], cols 3-5 = zeros
        // → identity block [[1,0,0,0,0,0], [0,1,0,0,0,0], [0,0,1,0,0,0]]
        let h = Matrix3x6::from_iterator([
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0,
        ]);
        let v = Vector6 {
            data: [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        };
        let result = h * v;
        assert!((result.x - 10.0).abs() < 1e-10);
        assert!((result.y - 20.0).abs() < 1e-10);
        assert!((result.z - 30.0).abs() < 1e-10);
    }
}
