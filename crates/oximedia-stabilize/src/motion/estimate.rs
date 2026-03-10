//! Motion estimation from feature tracks.
//!
//! This module estimates camera motion models from tracked features using
//! robust fitting algorithms like RANSAC.

use crate::error::{StabilizeError, StabilizeResult};
use crate::motion::model::{
    AffineModel, Matrix3x3, MotionModel, PerspectiveModel, TranslationModel,
};
use crate::motion::tracker::FeatureTrack;
use crate::StabilizationMode;
use scirs2_core::random::Random;

/// Motion estimator that fits motion models to feature tracks.
#[derive(Debug)]
pub struct MotionEstimator {
    /// Stabilization mode (determines model type)
    mode: StabilizationMode,
    /// RANSAC iterations
    ransac_iterations: usize,
    /// RANSAC inlier threshold
    ransac_threshold: f64,
    /// Minimum inlier ratio
    min_inlier_ratio: f64,
}

impl MotionEstimator {
    /// Create a new motion estimator.
    #[must_use]
    pub fn new(mode: StabilizationMode) -> Self {
        Self {
            mode,
            ransac_iterations: 1000,
            ransac_threshold: 3.0,
            min_inlier_ratio: 0.5,
        }
    }

    /// Set RANSAC parameters.
    pub fn set_ransac_params(&mut self, iterations: usize, threshold: f64, min_inlier_ratio: f64) {
        self.ransac_iterations = iterations;
        self.ransac_threshold = threshold;
        self.min_inlier_ratio = min_inlier_ratio.clamp(0.0, 1.0);
    }

    /// Estimate motion models from feature tracks.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Not enough features for estimation
    /// - Model fitting fails
    pub fn estimate(
        &self,
        tracks: &[FeatureTrack],
        frame_count: usize,
    ) -> StabilizeResult<Vec<Box<dyn MotionModel>>> {
        if tracks.is_empty() {
            return Err(StabilizeError::insufficient_features(0, 10));
        }

        let mut models = Vec::with_capacity(frame_count);

        // First frame has identity transformation
        models.push(self.create_identity_model());

        // Estimate transformation for each consecutive frame pair
        for frame_idx in 1..frame_count {
            let model = self.estimate_frame_pair(tracks, frame_idx - 1, frame_idx)?;
            models.push(model);
        }

        Ok(models)
    }

    /// Estimate motion between two specific frames.
    fn estimate_frame_pair(
        &self,
        tracks: &[FeatureTrack],
        frame1: usize,
        frame2: usize,
    ) -> StabilizeResult<Box<dyn MotionModel>> {
        // Collect correspondence points
        let mut correspondences = Vec::new();

        for track in tracks {
            if let (Some(pos1), Some(pos2)) = (track.position_at(frame1), track.position_at(frame2))
            {
                correspondences.push(Correspondence {
                    src: (pos1.0, pos1.1),
                    dst: (pos2.0, pos2.1),
                });
            }
        }

        if correspondences.is_empty() {
            return Ok(self.create_identity_model());
        }

        // Fit model using RANSAC
        let model = match self.mode {
            StabilizationMode::Translation => self.fit_translation_ransac(&correspondences)?,
            StabilizationMode::Affine => self.fit_affine_ransac(&correspondences)?,
            StabilizationMode::Perspective => self.fit_perspective_ransac(&correspondences)?,
            StabilizationMode::ThreeD => {
                // For 3D mode, start with affine approximation
                self.fit_affine_ransac(&correspondences)?
            }
        };

        Ok(model)
    }

    /// Create an identity model based on current mode.
    fn create_identity_model(&self) -> Box<dyn MotionModel> {
        match self.mode {
            StabilizationMode::Translation => Box::new(TranslationModel::identity()),
            StabilizationMode::Affine => Box::new(AffineModel::identity()),
            StabilizationMode::Perspective => Box::new(PerspectiveModel::identity()),
            StabilizationMode::ThreeD => Box::new(AffineModel::identity()),
        }
    }

    /// Fit translation model using RANSAC.
    fn fit_translation_ransac(
        &self,
        correspondences: &[Correspondence],
    ) -> StabilizeResult<Box<dyn MotionModel>> {
        if correspondences.is_empty() {
            return Ok(Box::new(TranslationModel::identity()));
        }

        let mut best_model: Option<TranslationModel> = None;
        let mut best_inliers = 0;
        let mut rng = Random::default();

        for _ in 0..self.ransac_iterations {
            // Sample one correspondence (minimum for translation)
            let idx = rng.random_range(0..correspondences.len());
            let corr = &correspondences[idx];

            // Fit model
            let model = TranslationModel::new(corr.dst.0 - corr.src.0, corr.dst.1 - corr.src.1);

            // Count inliers
            let inliers = self.count_inliers(&model, correspondences);

            if inliers > best_inliers {
                best_inliers = inliers;
                best_model = Some(model);
            }
        }

        // Check if we have enough inliers
        let inlier_ratio = best_inliers as f64 / correspondences.len() as f64;
        if inlier_ratio < self.min_inlier_ratio {
            return Ok(Box::new(TranslationModel::identity()));
        }

        // Refine using all inliers
        if let Some(model) = best_model {
            let refined = self.refine_translation(&model, correspondences);
            Ok(Box::new(refined))
        } else {
            Ok(Box::new(TranslationModel::identity()))
        }
    }

    /// Fit affine model using RANSAC.
    fn fit_affine_ransac(
        &self,
        correspondences: &[Correspondence],
    ) -> StabilizeResult<Box<dyn MotionModel>> {
        if correspondences.len() < 3 {
            return self.fit_translation_ransac(correspondences);
        }

        let mut best_model: Option<AffineModel> = None;
        let mut best_inliers = 0;
        let mut rng = Random::default();

        for _ in 0..self.ransac_iterations {
            // Sample 3 correspondences (minimum for affine)
            let mut indices = Vec::new();
            while indices.len() < 3 {
                let idx = rng.random_range(0..correspondences.len());
                if !indices.contains(&idx) {
                    indices.push(idx);
                }
            }

            let samples: Vec<_> = indices.iter().map(|&i| &correspondences[i]).collect();

            // Fit model
            if let Ok(model) = self.fit_affine_exact(&samples) {
                // Count inliers
                let inliers = self.count_inliers(&model, correspondences);

                if inliers > best_inliers {
                    best_inliers = inliers;
                    best_model = Some(model);
                }
            }
        }

        // Check if we have enough inliers
        let inlier_ratio = best_inliers as f64 / correspondences.len() as f64;
        if inlier_ratio < self.min_inlier_ratio {
            return Ok(Box::new(AffineModel::identity()));
        }

        // Refine using all inliers
        if let Some(model) = best_model {
            let refined = self.refine_affine(&model, correspondences);
            Ok(Box::new(refined))
        } else {
            Ok(Box::new(AffineModel::identity()))
        }
    }

    /// Fit perspective model using RANSAC.
    fn fit_perspective_ransac(
        &self,
        correspondences: &[Correspondence],
    ) -> StabilizeResult<Box<dyn MotionModel>> {
        if correspondences.len() < 4 {
            return self.fit_affine_ransac(correspondences);
        }

        let mut best_model: Option<PerspectiveModel> = None;
        let mut best_inliers = 0;
        let mut rng = Random::default();

        for _ in 0..self.ransac_iterations {
            // Sample 4 correspondences (minimum for homography)
            let mut indices = Vec::new();
            while indices.len() < 4 {
                let idx = rng.random_range(0..correspondences.len());
                if !indices.contains(&idx) {
                    indices.push(idx);
                }
            }

            let samples: Vec<_> = indices.iter().map(|&i| &correspondences[i]).collect();

            // Fit model
            if let Ok(model) = self.fit_homography(&samples) {
                // Count inliers
                let inliers = self.count_inliers(&model, correspondences);

                if inliers > best_inliers {
                    best_inliers = inliers;
                    best_model = Some(model);
                }
            }
        }

        // Check if we have enough inliers
        let inlier_ratio = best_inliers as f64 / correspondences.len() as f64;
        if inlier_ratio < self.min_inlier_ratio {
            return Ok(Box::new(PerspectiveModel::identity()));
        }

        // Refine using all inliers
        if let Some(model) = best_model {
            let refined = self.refine_homography(&model, correspondences);
            Ok(Box::new(refined))
        } else {
            Ok(Box::new(PerspectiveModel::identity()))
        }
    }

    /// Fit affine transformation to exactly 3 points.
    fn fit_affine_exact(
        &self,
        correspondences: &[&Correspondence],
    ) -> StabilizeResult<AffineModel> {
        if correspondences.len() != 3 {
            return Err(StabilizeError::invalid_parameter(
                "correspondences",
                format!("expected 3, got {}", correspondences.len()),
            ));
        }

        // Build 6x6 linear system for affine transformation
        // [x1 y1 1 0  0  0] [a11]   [x1']
        // [0  0  0 x1 y1 1] [a12] = [y1']
        // [x2 y2 1 0  0  0] [tx ]   [x2']
        // [0  0  0 x2 y2 1] [a21]   [y2']
        // [x3 y3 1 0  0  0] [a22]   [x3']
        // [0  0  0 x3 y3 1] [ty ]   [y3']
        let mut a_mat = [[0.0f64; 6]; 6];
        let mut b_vec = [0.0f64; 6];

        for (i, corr) in correspondences.iter().enumerate() {
            let row = i * 2;
            a_mat[row][0] = corr.src.0;
            a_mat[row][1] = corr.src.1;
            a_mat[row][2] = 1.0;
            b_vec[row] = corr.dst.0;

            a_mat[row + 1][3] = corr.src.0;
            a_mat[row + 1][4] = corr.src.1;
            a_mat[row + 1][5] = 1.0;
            b_vec[row + 1] = corr.dst.1;
        }

        // Solve using Gaussian elimination with partial pivoting
        let params = solve_6x6(&a_mat, &b_vec)
            .ok_or_else(|| StabilizeError::matrix("Matrix is singular"))?;

        // Extract affine parameters
        let a11 = params[0];
        let a12 = params[1];
        let a21 = params[3];
        let _a22 = params[4];
        let dx = params[2];
        let dy = params[5];

        // Decompose into TRS
        let scale = (a11 * a11 + a21 * a21).sqrt();
        let angle = a21.atan2(a11);
        let shear_x = a12 / scale;
        let shear_y = _a22 / scale - 1.0;

        Ok(AffineModel::new(dx, dy, angle, scale, shear_x, shear_y))
    }

    /// Fit homography to 4 or more points using DLT.
    fn fit_homography(
        &self,
        correspondences: &[&Correspondence],
    ) -> StabilizeResult<PerspectiveModel> {
        if correspondences.len() < 4 {
            return Err(StabilizeError::insufficient_features(
                correspondences.len(),
                4,
            ));
        }

        // Build DLT matrix and solve using SVD via scirs2-core
        let n = correspondences.len();
        use scirs2_core::ndarray::Array2;

        let mut a = Array2::<f64>::zeros((2 * n, 9));

        for (i, corr) in correspondences.iter().enumerate() {
            let x1 = corr.src.0;
            let y1 = corr.src.1;
            let x2 = corr.dst.0;
            let y2 = corr.dst.1;

            let row = i * 2;

            // First row
            a[[row, 0]] = -x1;
            a[[row, 1]] = -y1;
            a[[row, 2]] = -1.0;
            a[[row, 6]] = x2 * x1;
            a[[row, 7]] = x2 * y1;
            a[[row, 8]] = x2;

            // Second row
            a[[row + 1, 3]] = -x1;
            a[[row + 1, 4]] = -y1;
            a[[row + 1, 5]] = -1.0;
            a[[row + 1, 6]] = y2 * x1;
            a[[row + 1, 7]] = y2 * y1;
            a[[row + 1, 8]] = y2;
        }

        // Solve using SVD
        let svd = scirs2_core::linalg::svd_ndarray(&a)
            .map_err(|e| StabilizeError::matrix(format!("SVD failed: {e}")))?;

        // The last row of Vt gives the null space solution
        let vt = &svd.vt;
        let last_row = vt.nrows() - 1;

        let mut homography = Matrix3x3::zeros();
        for i in 0..3 {
            for j in 0..3 {
                homography.set(i, j, vt[[last_row, i * 3 + j]]);
            }
        }

        // Normalize
        let h22 = homography.get(2, 2);
        if h22.abs() > 1e-10 {
            homography = homography.scale(1.0 / h22);
        }

        Ok(PerspectiveModel::new(homography))
    }

    /// Count inliers for a given model.
    fn count_inliers(&self, model: &dyn MotionModel, correspondences: &[Correspondence]) -> usize {
        correspondences
            .iter()
            .filter(|corr| {
                let (pred_x, pred_y) = model.transform_point(corr.src.0, corr.src.1);
                let dx = pred_x - corr.dst.0;
                let dy = pred_y - corr.dst.1;
                let error = (dx * dx + dy * dy).sqrt();
                error < self.ransac_threshold
            })
            .count()
    }

    /// Refine translation model using all inliers.
    fn refine_translation(
        &self,
        model: &TranslationModel,
        correspondences: &[Correspondence],
    ) -> TranslationModel {
        let mut sum_dx = 0.0;
        let mut sum_dy = 0.0;
        let mut count = 0;

        for corr in correspondences {
            let (pred_x, pred_y) = model.transform_point(corr.src.0, corr.src.1);
            let dx = pred_x - corr.dst.0;
            let dy = pred_y - corr.dst.1;
            let error = (dx * dx + dy * dy).sqrt();

            if error < self.ransac_threshold {
                sum_dx += corr.dst.0 - corr.src.0;
                sum_dy += corr.dst.1 - corr.src.1;
                count += 1;
            }
        }

        if count > 0 {
            TranslationModel::new(sum_dx / count as f64, sum_dy / count as f64)
        } else {
            model.clone()
        }
    }

    /// Refine affine model using all inliers.
    fn refine_affine(
        &self,
        model: &AffineModel,
        correspondences: &[Correspondence],
    ) -> AffineModel {
        // Collect inliers
        let inliers: Vec<_> = correspondences
            .iter()
            .filter(|corr| {
                let (pred_x, pred_y) = model.transform_point(corr.src.0, corr.src.1);
                let dx = pred_x - corr.dst.0;
                let dy = pred_y - corr.dst.1;
                let error = (dx * dx + dy * dy).sqrt();
                error < self.ransac_threshold
            })
            .collect();

        if inliers.len() < 3 {
            return model.clone();
        }

        // Least squares refinement using all inliers
        // Compute mean displacements as refined estimate
        let mut sum_dx = 0.0;
        let mut sum_dy = 0.0;
        for corr in &inliers {
            sum_dx += corr.dst.0 - corr.src.0;
            sum_dy += corr.dst.1 - corr.src.1;
        }
        let n = inliers.len() as f64;
        let refined_dx = sum_dx / n;
        let refined_dy = sum_dy / n;

        AffineModel::new(
            refined_dx,
            refined_dy,
            model.angle,
            model.scale,
            model.shear_x,
            model.shear_y,
        )
    }

    /// Refine homography using all inliers.
    fn refine_homography(
        &self,
        model: &PerspectiveModel,
        correspondences: &[Correspondence],
    ) -> PerspectiveModel {
        // Collect inliers
        let inliers: Vec<_> = correspondences
            .iter()
            .filter(|corr| {
                let (pred_x, pred_y) = model.transform_point(corr.src.0, corr.src.1);
                let dx = pred_x - corr.dst.0;
                let dy = pred_y - corr.dst.1;
                let error = (dx * dx + dy * dy).sqrt();
                error < self.ransac_threshold
            })
            .collect();

        if inliers.len() < 4 {
            return model.clone();
        }

        // Fit to all inliers
        if let Ok(refined) = self.fit_homography(&inliers.iter().copied().collect::<Vec<_>>()) {
            refined
        } else {
            model.clone()
        }
    }
}

/// Solve a 6x6 linear system using Gaussian elimination with partial pivoting.
fn solve_6x6(a: &[[f64; 6]; 6], b: &[f64; 6]) -> Option<[f64; 6]> {
    let mut aug = [[0.0f64; 7]; 6];
    for i in 0..6 {
        for j in 0..6 {
            aug[i][j] = a[i][j];
        }
        aug[i][6] = b[i];
    }

    // Forward elimination with partial pivoting
    for col in 0..6 {
        // Find pivot
        let mut max_val = aug[col][col].abs();
        let mut max_row = col;
        for row in (col + 1)..6 {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }

        if max_val < 1e-15 {
            return None; // Singular
        }

        // Swap rows
        if max_row != col {
            aug.swap(col, max_row);
        }

        // Eliminate below
        for row in (col + 1)..6 {
            let factor = aug[row][col] / aug[col][col];
            for j in col..7 {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }

    // Back substitution
    let mut x = [0.0f64; 6];
    for i in (0..6).rev() {
        let mut sum = aug[i][6];
        for j in (i + 1)..6 {
            sum -= aug[i][j] * x[j];
        }
        x[i] = sum / aug[i][i];
    }

    Some(x)
}

/// A correspondence between points in two frames.
#[derive(Debug, Clone, Copy)]
struct Correspondence {
    /// Source point (x, y)
    src: (f64, f64),
    /// Destination point (x, y)
    dst: (f64, f64),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimator_creation() {
        let estimator = MotionEstimator::new(StabilizationMode::Affine);
        assert_eq!(estimator.ransac_iterations, 1000);
    }

    #[test]
    fn test_identity_models() {
        let estimator = MotionEstimator::new(StabilizationMode::Translation);
        let model = estimator.create_identity_model();
        let (x, y) = model.transform_point(5.0, 7.0);
        assert!((x - 5.0).abs() < 1e-10);
        assert!((y - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_correspondence() {
        let corr = Correspondence {
            src: (0.0, 0.0),
            dst: (10.0, 20.0),
        };
        assert!((corr.src.0 - 0.0).abs() < f64::EPSILON);
        assert!((corr.dst.0 - 10.0).abs() < f64::EPSILON);
    }
}
