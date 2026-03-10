#![allow(dead_code)]
//! Elastic (non-rigid) alignment for deformable media registration.
//!
//! This module implements non-rigid alignment techniques that can handle local deformations
//! such as lens distortion residuals, rolling shutter wobble, and object-level motion.
//!
//! # Features
//!
//! - **Thin-plate spline (TPS) warping** for smooth non-rigid transforms
//! - **Control point management** with automatic correspondence
//! - **Regularized alignment** to prevent overfitting
//! - **Deformation field** representation and analysis

use crate::{AlignError, AlignResult, Point2D};

/// A control point pair used as a landmark for elastic alignment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ControlPoint {
    /// Source position.
    pub source: Point2D,
    /// Target position (where the source should map to).
    pub target: Point2D,
    /// Weight for this control point (higher means stronger influence).
    pub weight: f64,
}

impl ControlPoint {
    /// Create a new control point pair.
    #[must_use]
    pub fn new(source: Point2D, target: Point2D, weight: f64) -> Self {
        Self {
            source,
            target,
            weight,
        }
    }

    /// Create with default weight 1.0.
    #[must_use]
    pub fn with_unit_weight(source: Point2D, target: Point2D) -> Self {
        Self {
            source,
            target,
            weight: 1.0,
        }
    }

    /// Compute the displacement vector (target - source).
    #[must_use]
    pub fn displacement(&self) -> (f64, f64) {
        (self.target.x - self.source.x, self.target.y - self.source.y)
    }

    /// Compute the displacement magnitude.
    #[must_use]
    pub fn displacement_magnitude(&self) -> f64 {
        let (dx, dy) = self.displacement();
        (dx * dx + dy * dy).sqrt()
    }
}

/// Configuration for elastic alignment.
#[derive(Debug, Clone)]
pub struct ElasticAlignConfig {
    /// Regularization parameter (lambda). Higher values produce smoother warps.
    pub regularization: f64,
    /// Minimum number of control points required.
    pub min_control_points: usize,
    /// Maximum allowed displacement in pixels.
    pub max_displacement: f64,
    /// Grid resolution for deformation field output (pixels per cell).
    pub grid_resolution: u32,
}

impl Default for ElasticAlignConfig {
    fn default() -> Self {
        Self {
            regularization: 0.01,
            min_control_points: 4,
            max_displacement: 100.0,
            grid_resolution: 16,
        }
    }
}

/// Thin-Plate Spline coefficients for one coordinate dimension.
#[derive(Debug, Clone)]
pub struct TpsCoefficients {
    /// Weights for each control point (non-linear part).
    pub weights: Vec<f64>,
    /// Affine part: a0 + a1*x + a2*y.
    pub affine: [f64; 3],
}

/// Result of elastic alignment computation.
#[derive(Debug, Clone)]
pub struct ElasticAlignResult {
    /// TPS coefficients for x-coordinate mapping.
    pub tps_x: TpsCoefficients,
    /// TPS coefficients for y-coordinate mapping.
    pub tps_y: TpsCoefficients,
    /// The control points used.
    pub control_points: Vec<ControlPoint>,
    /// Root-mean-square alignment error in pixels.
    pub rms_error: f64,
    /// Maximum alignment error in pixels.
    pub max_error: f64,
    /// Bending energy of the deformation (lower is smoother).
    pub bending_energy: f64,
}

/// A sampled deformation field on a regular grid.
#[derive(Debug, Clone)]
pub struct DeformationField {
    /// Horizontal displacement for each grid cell.
    pub dx: Vec<f64>,
    /// Vertical displacement for each grid cell.
    pub dy: Vec<f64>,
    /// Number of grid columns.
    pub cols: u32,
    /// Number of grid rows.
    pub rows: u32,
    /// Cell size in pixels.
    pub cell_size: u32,
}

impl DeformationField {
    /// Create a zero deformation field.
    #[must_use]
    pub fn new(width: u32, height: u32, cell_size: u32) -> Self {
        let cols = width.div_ceil(cell_size);
        let rows = height.div_ceil(cell_size);
        let count = (cols * rows) as usize;
        Self {
            dx: vec![0.0; count],
            dy: vec![0.0; count],
            cols,
            rows,
            cell_size,
        }
    }

    /// Get displacement at grid cell (cx, cy).
    #[must_use]
    pub fn get(&self, cx: u32, cy: u32) -> Option<(f64, f64)> {
        if cx < self.cols && cy < self.rows {
            let idx = (cy * self.cols + cx) as usize;
            Some((self.dx[idx], self.dy[idx]))
        } else {
            None
        }
    }

    /// Set displacement at grid cell (cx, cy).
    pub fn set(&mut self, cx: u32, cy: u32, dx: f64, dy: f64) {
        if cx < self.cols && cy < self.rows {
            let idx = (cy * self.cols + cx) as usize;
            self.dx[idx] = dx;
            self.dy[idx] = dy;
        }
    }

    /// Compute the average displacement magnitude across the field.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn average_displacement(&self) -> f64 {
        if self.dx.is_empty() {
            return 0.0;
        }
        let total: f64 = self
            .dx
            .iter()
            .zip(self.dy.iter())
            .map(|(x, y)| (x * x + y * y).sqrt())
            .sum();
        total / self.dx.len() as f64
    }

    /// Compute maximum displacement magnitude in the field.
    #[must_use]
    pub fn max_displacement(&self) -> f64 {
        self.dx
            .iter()
            .zip(self.dy.iter())
            .map(|(x, y)| (x * x + y * y).sqrt())
            .fold(0.0_f64, f64::max)
    }
}

/// The Thin-Plate Spline radial basis function: r^2 * ln(r).
#[allow(clippy::cast_precision_loss)]
fn tps_kernel(r: f64) -> f64 {
    if r < 1e-15 {
        0.0
    } else {
        r * r * r.ln()
    }
}

/// Elastic aligner using thin-plate spline interpolation.
#[derive(Debug, Clone)]
pub struct ElasticAligner {
    /// Configuration.
    config: ElasticAlignConfig,
}

impl ElasticAligner {
    /// Create a new elastic aligner with the given configuration.
    #[must_use]
    pub fn new(config: ElasticAlignConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self {
            config: ElasticAlignConfig::default(),
        }
    }

    /// Compute TPS alignment from control point correspondences.
    pub fn align(&self, control_points: &[ControlPoint]) -> AlignResult<ElasticAlignResult> {
        let n = control_points.len();
        if n < self.config.min_control_points {
            return Err(AlignError::InsufficientData(format!(
                "Need at least {} control points, got {}",
                self.config.min_control_points, n
            )));
        }

        // Check max displacement constraint
        for cp in control_points {
            if cp.displacement_magnitude() > self.config.max_displacement {
                return Err(AlignError::InvalidConfig(format!(
                    "Control point displacement {:.1} exceeds max {:.1}",
                    cp.displacement_magnitude(),
                    self.config.max_displacement
                )));
            }
        }

        // Solve TPS for x and y independently
        let tps_x = self.solve_tps(control_points, true)?;
        let tps_y = self.solve_tps(control_points, false)?;

        // Compute errors
        let (rms_error, max_error) = self.compute_errors(control_points, &tps_x, &tps_y);

        // Compute bending energy
        let bending_energy = self.compute_bending_energy(control_points, &tps_x, &tps_y);

        Ok(ElasticAlignResult {
            tps_x,
            tps_y,
            control_points: control_points.to_vec(),
            rms_error,
            max_error,
            bending_energy,
        })
    }

    /// Transform a point using TPS coefficients.
    #[must_use]
    pub fn transform_point(&self, point: &Point2D, result: &ElasticAlignResult) -> Point2D {
        let new_x = self.evaluate_tps(point, &result.tps_x, &result.control_points);
        let new_y = self.evaluate_tps(point, &result.tps_y, &result.control_points);
        Point2D::new(new_x, new_y)
    }

    /// Generate a sampled deformation field from an alignment result.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn generate_deformation_field(
        &self,
        result: &ElasticAlignResult,
        width: u32,
        height: u32,
    ) -> DeformationField {
        let cell_size = self.config.grid_resolution;
        let mut field = DeformationField::new(width, height, cell_size);

        for cy in 0..field.rows {
            for cx in 0..field.cols {
                let px = f64::from(cx * cell_size + cell_size / 2);
                let py = f64::from(cy * cell_size + cell_size / 2);
                let src = Point2D::new(px, py);
                let dst = self.transform_point(&src, result);
                field.set(cx, cy, dst.x - px, dst.y - py);
            }
        }

        field
    }

    /// Solve TPS for one coordinate (x if `for_x` is true, y otherwise).
    #[allow(clippy::cast_precision_loss)]
    fn solve_tps(&self, points: &[ControlPoint], for_x: bool) -> AlignResult<TpsCoefficients> {
        let n = points.len();
        // System size: n + 3 (n weights + 3 affine params)
        let size = n + 3;

        // Build the system matrix L and right-hand side v
        // L = | K  P |   v = | target_coords |
        //     | P' 0 |       |       0       |
        let mut l_matrix = vec![0.0f64; size * size];
        let mut rhs = vec![0.0f64; size];

        // Fill K (n x n kernel matrix) + regularization on diagonal
        for i in 0..n {
            for j in 0..n {
                let r = points[i].source.distance(&points[j].source);
                l_matrix[i * size + j] = tps_kernel(r);
            }
            // Regularization
            l_matrix[i * size + i] += self.config.regularization / points[i].weight;
        }

        // Fill P (n x 3) and P^T (3 x n)
        for i in 0..n {
            l_matrix[i * size + n] = 1.0;
            l_matrix[i * size + n + 1] = points[i].source.x;
            l_matrix[i * size + n + 2] = points[i].source.y;

            l_matrix[(n) * size + i] = 1.0;
            l_matrix[(n + 1) * size + i] = points[i].source.x;
            l_matrix[(n + 2) * size + i] = points[i].source.y;
        }

        // Fill rhs
        for i in 0..n {
            rhs[i] = if for_x {
                points[i].target.x
            } else {
                points[i].target.y
            };
        }

        // Solve using Gauss elimination with partial pivoting
        let solution = Self::gauss_solve(&mut l_matrix, &mut rhs, size)?;

        let weights = solution[..n].to_vec();
        let affine = [solution[n], solution[n + 1], solution[n + 2]];

        Ok(TpsCoefficients { weights, affine })
    }

    /// Evaluate TPS at a point.
    fn evaluate_tps(
        &self,
        point: &Point2D,
        tps: &TpsCoefficients,
        control_points: &[ControlPoint],
    ) -> f64 {
        let mut val = tps.affine[0] + tps.affine[1] * point.x + tps.affine[2] * point.y;

        for (i, cp) in control_points.iter().enumerate() {
            let r = point.distance(&cp.source);
            val += tps.weights[i] * tps_kernel(r);
        }

        val
    }

    /// Compute RMS and max error.
    #[allow(clippy::cast_precision_loss)]
    fn compute_errors(
        &self,
        points: &[ControlPoint],
        tps_x: &TpsCoefficients,
        tps_y: &TpsCoefficients,
    ) -> (f64, f64) {
        let mut sum_sq = 0.0;
        let mut max_e = 0.0_f64;

        for cp in points {
            let px = self.evaluate_tps(&cp.source, tps_x, points);
            let py = self.evaluate_tps(&cp.source, tps_y, points);
            let err = ((px - cp.target.x).powi(2) + (py - cp.target.y).powi(2)).sqrt();
            sum_sq += err * err;
            max_e = max_e.max(err);
        }

        let rms = (sum_sq / points.len() as f64).sqrt();
        (rms, max_e)
    }

    /// Compute bending energy.
    fn compute_bending_energy(
        &self,
        points: &[ControlPoint],
        tps_x: &TpsCoefficients,
        tps_y: &TpsCoefficients,
    ) -> f64 {
        let n = points.len();
        let mut energy = 0.0;

        for i in 0..n {
            for j in 0..n {
                let r = points[i].source.distance(&points[j].source);
                let k = tps_kernel(r);
                energy += tps_x.weights[i] * tps_x.weights[j] * k;
                energy += tps_y.weights[i] * tps_y.weights[j] * k;
            }
        }

        energy.abs()
    }

    /// Gaussian elimination with partial pivoting.
    fn gauss_solve(a: &mut [f64], b: &mut [f64], n: usize) -> AlignResult<Vec<f64>> {
        // Forward elimination
        for col in 0..n {
            // Partial pivoting
            let mut max_val = a[col * n + col].abs();
            let mut max_row = col;
            for row in (col + 1)..n {
                let val = a[row * n + col].abs();
                if val > max_val {
                    max_val = val;
                    max_row = row;
                }
            }

            if max_val < 1e-15 {
                return Err(AlignError::NumericalError(
                    "Singular matrix in TPS solve".to_string(),
                ));
            }

            // Swap rows
            if max_row != col {
                for k in 0..n {
                    a.swap(col * n + k, max_row * n + k);
                }
                b.swap(col, max_row);
            }

            // Eliminate below
            let pivot = a[col * n + col];
            for row in (col + 1)..n {
                let factor = a[row * n + col] / pivot;
                for k in col..n {
                    a[row * n + k] -= factor * a[col * n + k];
                }
                b[row] -= factor * b[col];
            }
        }

        // Back substitution
        let mut x = vec![0.0f64; n];
        for col in (0..n).rev() {
            let mut sum = b[col];
            for k in (col + 1)..n {
                sum -= a[col * n + k] * x[k];
            }
            x[col] = sum / a[col * n + col];
        }

        Ok(x)
    }

    /// Get the current configuration.
    #[must_use]
    pub fn config(&self) -> &ElasticAlignConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_control_point_creation() {
        let cp = ControlPoint::new(Point2D::new(10.0, 20.0), Point2D::new(12.0, 22.0), 1.0);
        assert!((cp.source.x - 10.0).abs() < f64::EPSILON);
        assert!((cp.target.x - 12.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_control_point_displacement() {
        let cp = ControlPoint::new(Point2D::new(0.0, 0.0), Point2D::new(3.0, 4.0), 1.0);
        let (dx, dy) = cp.displacement();
        assert!((dx - 3.0).abs() < f64::EPSILON);
        assert!((dy - 4.0).abs() < f64::EPSILON);
        assert!((cp.displacement_magnitude() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_control_point_unit_weight() {
        let cp = ControlPoint::with_unit_weight(Point2D::new(0.0, 0.0), Point2D::new(1.0, 1.0));
        assert!((cp.weight - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_config_default() {
        let config = ElasticAlignConfig::default();
        assert!((config.regularization - 0.01).abs() < f64::EPSILON);
        assert_eq!(config.min_control_points, 4);
    }

    #[test]
    fn test_deformation_field_creation() {
        let field = DeformationField::new(320, 240, 16);
        assert_eq!(field.cols, 20);
        assert_eq!(field.rows, 15);
        assert_eq!(field.dx.len(), 300);
    }

    #[test]
    fn test_deformation_field_get_set() {
        let mut field = DeformationField::new(64, 64, 16);
        field.set(1, 2, 3.5, -1.5);
        let (dx, dy) = field.get(1, 2).expect("get should succeed");
        assert!((dx - 3.5).abs() < f64::EPSILON);
        assert!((dy - (-1.5)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_deformation_field_average() {
        let mut field = DeformationField::new(32, 32, 16);
        field.set(0, 0, 3.0, 4.0); // magnitude 5
        field.set(1, 0, 0.0, 0.0);
        field.set(0, 1, 0.0, 0.0);
        field.set(1, 1, 0.0, 0.0);
        assert!((field.average_displacement() - 1.25).abs() < 1e-10);
    }

    #[test]
    fn test_deformation_field_max() {
        let mut field = DeformationField::new(32, 32, 16);
        field.set(0, 0, 3.0, 4.0);
        field.set(1, 0, 1.0, 0.0);
        assert!((field.max_displacement() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_tps_kernel() {
        assert!((tps_kernel(0.0)).abs() < f64::EPSILON);
        // tps_kernel(1.0) = 1^2 * ln(1) = 0
        assert!((tps_kernel(1.0)).abs() < f64::EPSILON);
        // tps_kernel(e) = e^2 * ln(e) = e^2
        let e = std::f64::consts::E;
        assert!((tps_kernel(e) - e * e).abs() < 1e-10);
    }

    #[test]
    fn test_elastic_align_insufficient_points() {
        let aligner = ElasticAligner::with_defaults();
        let points = vec![ControlPoint::with_unit_weight(
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 1.0),
        )];
        let result = aligner.align(&points);
        assert!(result.is_err());
    }

    #[test]
    fn test_elastic_align_identity() {
        let aligner = ElasticAligner::new(ElasticAlignConfig {
            regularization: 0.001,
            min_control_points: 4,
            max_displacement: 100.0,
            grid_resolution: 16,
        });

        // Identity mapping: source == target
        let points = vec![
            ControlPoint::with_unit_weight(Point2D::new(0.0, 0.0), Point2D::new(0.0, 0.0)),
            ControlPoint::with_unit_weight(Point2D::new(100.0, 0.0), Point2D::new(100.0, 0.0)),
            ControlPoint::with_unit_weight(Point2D::new(0.0, 100.0), Point2D::new(0.0, 100.0)),
            ControlPoint::with_unit_weight(Point2D::new(100.0, 100.0), Point2D::new(100.0, 100.0)),
        ];

        let result = aligner.align(&points).expect("result should be valid");
        // RMS error should be very small for identity
        assert!(result.rms_error < 1.0);
    }

    #[test]
    fn test_elastic_align_translation() {
        let aligner = ElasticAligner::new(ElasticAlignConfig {
            regularization: 0.001,
            min_control_points: 4,
            max_displacement: 100.0,
            grid_resolution: 16,
        });

        // Translation of (5, 3)
        let points = vec![
            ControlPoint::with_unit_weight(Point2D::new(0.0, 0.0), Point2D::new(5.0, 3.0)),
            ControlPoint::with_unit_weight(Point2D::new(100.0, 0.0), Point2D::new(105.0, 3.0)),
            ControlPoint::with_unit_weight(Point2D::new(0.0, 100.0), Point2D::new(5.0, 103.0)),
            ControlPoint::with_unit_weight(Point2D::new(100.0, 100.0), Point2D::new(105.0, 103.0)),
        ];

        let result = aligner.align(&points).expect("result should be valid");
        // Transform a test point
        let transformed = aligner.transform_point(&Point2D::new(50.0, 50.0), &result);
        // Should be approximately (55, 53)
        assert!((transformed.x - 55.0).abs() < 2.0);
        assert!((transformed.y - 53.0).abs() < 2.0);
    }

    #[test]
    fn test_elastic_align_max_displacement_exceeded() {
        let aligner = ElasticAligner::new(ElasticAlignConfig {
            max_displacement: 5.0,
            ..ElasticAlignConfig::default()
        });

        let points = vec![
            ControlPoint::with_unit_weight(Point2D::new(0.0, 0.0), Point2D::new(100.0, 100.0)),
            ControlPoint::with_unit_weight(Point2D::new(10.0, 0.0), Point2D::new(110.0, 100.0)),
            ControlPoint::with_unit_weight(Point2D::new(0.0, 10.0), Point2D::new(100.0, 110.0)),
            ControlPoint::with_unit_weight(Point2D::new(10.0, 10.0), Point2D::new(110.0, 110.0)),
        ];

        let result = aligner.align(&points);
        assert!(result.is_err());
    }
}
