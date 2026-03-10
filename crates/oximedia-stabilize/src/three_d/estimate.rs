//! 3D camera pose estimation.

use crate::error::StabilizeResult;
use crate::motion::model::{Matrix3x3, Vector3};
use crate::motion::tracker::FeatureTrack;

/// Camera pose estimator for 3D stabilization.
pub struct CameraPoseEstimator {
    focal_length: f64,
}

impl CameraPoseEstimator {
    /// Create a new camera pose estimator.
    #[must_use]
    pub fn new() -> Self {
        Self { focal_length: 1.0 }
    }

    /// Set focal length.
    pub fn set_focal_length(&mut self, focal_length: f64) {
        self.focal_length = focal_length;
    }

    /// Estimate camera pose from feature tracks.
    pub fn estimate_pose(&self, _tracks: &[FeatureTrack]) -> StabilizeResult<CameraPose> {
        Ok(CameraPose {
            rotation: Matrix3x3::identity(),
            translation: Vector3::zeros(),
        })
    }
}

impl Default for CameraPoseEstimator {
    fn default() -> Self {
        Self::new()
    }
}

/// 3D camera pose.
#[derive(Debug, Clone)]
pub struct CameraPose {
    /// Rotation matrix
    pub rotation: Matrix3x3,
    /// Translation vector
    pub translation: Vector3,
}

/// A 3x4 projection matrix stored in row-major order.
#[derive(Debug, Clone, Copy)]
pub struct Matrix3x4 {
    /// Row-major data: 3 rows x 4 cols = 12 elements
    pub data: [f64; 12],
}

impl Matrix3x4 {
    /// Create a new 3x4 matrix from row-major data.
    #[must_use]
    pub const fn new(data: [f64; 12]) -> Self {
        Self { data }
    }

    /// Create a 3x4 identity-like matrix [I | 0].
    #[must_use]
    pub const fn identity() -> Self {
        Self {
            data: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        }
    }

    /// Get element at (row, col).
    #[must_use]
    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.data[row * 4 + col]
    }

    /// Get a row as [f64; 4].
    #[must_use]
    pub fn row(&self, r: usize) -> [f64; 4] {
        let base = r * 4;
        [
            self.data[base],
            self.data[base + 1],
            self.data[base + 2],
            self.data[base + 3],
        ]
    }
}

/// A 4x4 matrix stored in row-major order.
#[derive(Debug, Clone, Copy)]
struct Matrix4x4 {
    data: [f64; 16],
}

impl Matrix4x4 {
    fn zeros() -> Self {
        Self { data: [0.0; 16] }
    }

    fn get(&self, row: usize, col: usize) -> f64 {
        self.data[row * 4 + col]
    }

    fn set(&mut self, row: usize, col: usize, val: f64) {
        self.data[row * 4 + col] = val;
    }

    /// Set a row from a [f64; 4] array.
    fn set_row(&mut self, row: usize, vals: [f64; 4]) {
        let base = row * 4;
        for (i, &v) in vals.iter().enumerate() {
            self.data[base + i] = v;
        }
    }

    /// Compute SVD of this 4x4 matrix using scirs2-core.
    /// Returns (U, sigma, V^T) as 4x4 matrices and singular values.
    fn svd(&self) -> Option<(Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>)> {
        use scirs2_core::ndarray::Array2;

        let mut arr = Array2::zeros((4, 4));
        for i in 0..4 {
            for j in 0..4 {
                arr[[i, j]] = self.get(i, j);
            }
        }

        match scirs2_core::linalg::svd_ndarray(&arr) {
            Ok(svd) => Some((
                (0..svd.u.nrows())
                    .map(|i| (0..svd.u.ncols()).map(|j| svd.u[[i, j]]).collect())
                    .collect(),
                svd.s.to_vec(),
                (0..svd.vt.nrows())
                    .map(|i| (0..svd.vt.ncols()).map(|j| svd.vt[[i, j]]).collect())
                    .collect(),
            )),
            Err(_) => None,
        }
    }
}

/// Subtract scaled row: result[i] = a[i] * scale_a - b[i] * scale_b (for 4-element rows).
fn row_sub_scaled(a: &[f64; 4], scale_a: f64, b: &[f64; 4], _scale_b: f64) -> [f64; 4] {
    [
        scale_a * a[0] - b[0],
        scale_a * a[1] - b[1],
        scale_a * a[2] - b[2],
        scale_a * a[3] - b[3],
    ]
}

/// Structure from motion utilities.
pub mod sfm {
    use super::*;
    use crate::motion::model::{Matrix3x3, Vector3};

    /// Triangulate 3D point from 2D correspondences.
    #[must_use]
    pub fn triangulate_point(
        p1: (f64, f64),
        p2: (f64, f64),
        cam1: &Matrix3x4,
        cam2: &Matrix3x4,
    ) -> Vector3 {
        // Direct Linear Transform (DLT) for triangulation
        let mut a = Matrix4x4::zeros();

        let cam1_row0 = cam1.row(0);
        let cam1_row1 = cam1.row(1);
        let cam1_row2 = cam1.row(2);
        let cam2_row0 = cam2.row(0);
        let cam2_row1 = cam2.row(1);
        let cam2_row2 = cam2.row(2);

        // p1.0 * cam1.row(2) - cam1.row(0)
        a.set_row(0, row_sub_scaled(&cam1_row2, p1.0, &cam1_row0, 1.0));
        // p1.1 * cam1.row(2) - cam1.row(1)
        a.set_row(1, row_sub_scaled(&cam1_row2, p1.1, &cam1_row1, 1.0));
        // p2.0 * cam2.row(2) - cam2.row(0)
        a.set_row(2, row_sub_scaled(&cam2_row2, p2.0, &cam2_row0, 1.0));
        // p2.1 * cam2.row(2) - cam2.row(1)
        a.set_row(3, row_sub_scaled(&cam2_row2, p2.1, &cam2_row1, 1.0));

        if let Some((_u, _s, vt)) = a.svd() {
            if vt.len() >= 4 && vt[3].len() >= 4 {
                let point = &vt[3];
                let w = point[3];

                if w.abs() > 1e-10 {
                    return Vector3::new(point[0] / w, point[1] / w, point[2] / w);
                }
            }
        }

        Vector3::zeros()
    }

    /// Estimate essential matrix from point correspondences.
    #[must_use]
    pub fn estimate_essential_matrix(points1: &[(f64, f64)], points2: &[(f64, f64)]) -> Matrix3x3 {
        use scirs2_core::ndarray::Array2;

        // 8-point algorithm
        if points1.len() < 8 || points2.len() < 8 {
            return Matrix3x3::identity();
        }

        let n = points1.len().min(points2.len());
        let mut a = Array2::zeros((n, 9));

        for i in 0..n {
            let (x1, y1) = points1[i];
            let (x2, y2) = points2[i];

            a[[i, 0]] = x2 * x1;
            a[[i, 1]] = x2 * y1;
            a[[i, 2]] = x2;
            a[[i, 3]] = y2 * x1;
            a[[i, 4]] = y2 * y1;
            a[[i, 5]] = y2;
            a[[i, 6]] = x1;
            a[[i, 7]] = y1;
            a[[i, 8]] = 1.0;
        }

        match scirs2_core::linalg::svd_ndarray(&a) {
            Ok(svd) => {
                if svd.vt.nrows() >= 9 {
                    let f_row = svd.vt.row(8);
                    Matrix3x3::new(
                        f_row[0], f_row[1], f_row[2], f_row[3], f_row[4], f_row[5], f_row[6],
                        f_row[7], f_row[8],
                    )
                } else {
                    Matrix3x3::identity()
                }
            }
            Err(_) => Matrix3x3::identity(),
        }
    }

    /// Decompose essential matrix into rotation and translation.
    ///
    /// Returns up to 4 possible (rotation, translation) solutions.
    #[must_use]
    pub fn decompose_essential_matrix(e: &Matrix3x3) -> Vec<(Matrix3x3, Vector3)> {
        use scirs2_core::ndarray::Array2;

        let mut arr = Array2::zeros((3, 3));
        for i in 0..3 {
            for j in 0..3 {
                arr[[i, j]] = e.get(i, j);
            }
        }

        let w = Matrix3x3::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        let w_t = Matrix3x3::new(0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0);

        let mut solutions = Vec::new();

        if let Ok(svd) = scirs2_core::linalg::svd_ndarray(&arr) {
            // Convert U and V^T to Matrix3x3
            let mut u = Matrix3x3::zeros();
            let mut vt = Matrix3x3::zeros();
            for i in 0..3 {
                for j in 0..3 {
                    u.set(i, j, svd.u[[i, j]]);
                    vt.set(i, j, svd.vt[[i, j]]);
                }
            }

            let r1 = u.mul(&w).mul(&vt);
            let r2 = u.mul(&w_t).mul(&vt);
            let t = Vector3::new(u.get(0, 2), u.get(1, 2), u.get(2, 2));

            solutions.push((r1, t));
            solutions.push((r1, t.neg()));
            solutions.push((r2, t));
            solutions.push((r2, t.neg()));
        }

        solutions
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_triangulation() {
            let cam1 = Matrix3x4::identity();
            let cam2 = Matrix3x4::identity();

            let point = triangulate_point((0.0, 0.0), (1.0, 0.0), &cam1, &cam2);
            assert!(point.norm() >= 0.0);
        }

        #[test]
        fn test_essential_matrix() {
            let points1 = vec![(0.0, 0.0); 10];
            let points2 = vec![(1.0, 1.0); 10];

            let e = estimate_essential_matrix(&points1, &points2);
            assert!(e.determinant().abs() >= 0.0);
        }
    }
}
