//! Feature-based image registration.
//!
//! This module provides feature detection, description, matching, and transformation estimation
//! using keypoint-based methods similar to SIFT, SURF, and ORB.

use crate::error::{CvError, CvResult};
use crate::registration::{RegistrationQuality, TransformMatrix, TransformationType};
use std::f64::consts::PI;

/// Feature detector type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureDetectorType {
    /// SIFT-like feature detector.
    Sift,
    /// SURF-like feature detector.
    Surf,
    /// ORB-like feature detector.
    Orb,
    /// Harris corner detector.
    Harris,
}

/// Keypoint representation.
#[derive(Debug, Clone, Copy)]
pub struct Keypoint {
    /// X coordinate.
    pub x: f32,
    /// Y coordinate.
    pub y: f32,
    /// Scale/size.
    pub scale: f32,
    /// Orientation angle (radians).
    pub angle: f32,
    /// Response strength.
    pub response: f32,
    /// Octave level.
    pub octave: i32,
}

impl Keypoint {
    /// Create a new keypoint.
    #[must_use]
    pub const fn new(x: f32, y: f32, scale: f32, angle: f32, response: f32, octave: i32) -> Self {
        Self {
            x,
            y,
            scale,
            angle,
            response,
            octave,
        }
    }
}

/// Feature descriptor.
#[derive(Debug, Clone)]
pub struct Descriptor {
    /// Descriptor data (typically 128 or 256 dimensions).
    pub data: Vec<f32>,
    /// Associated keypoint index.
    pub keypoint_idx: usize,
}

impl Descriptor {
    /// Compute Euclidean distance to another descriptor.
    #[must_use]
    pub fn distance(&self, other: &Descriptor) -> f32 {
        if self.data.len() != other.data.len() {
            return f32::MAX;
        }
        let mut sum = 0.0;
        for i in 0..self.data.len() {
            let diff = self.data[i] - other.data[i];
            sum += diff * diff;
        }
        sum.sqrt()
    }

    /// Compute normalized cross-correlation.
    #[must_use]
    pub fn correlation(&self, other: &Descriptor) -> f32 {
        if self.data.len() != other.data.len() {
            return 0.0;
        }
        let mut dot = 0.0;
        let mut norm1 = 0.0;
        let mut norm2 = 0.0;
        for i in 0..self.data.len() {
            dot += self.data[i] * other.data[i];
            norm1 += self.data[i] * self.data[i];
            norm2 += other.data[i] * other.data[i];
        }
        if norm1 < f32::EPSILON || norm2 < f32::EPSILON {
            return 0.0;
        }
        dot / (norm1.sqrt() * norm2.sqrt())
    }
}

/// Feature match between two keypoints.
#[derive(Debug, Clone, Copy)]
pub struct Match {
    /// Query keypoint index.
    pub query_idx: usize,
    /// Train keypoint index.
    pub train_idx: usize,
    /// Match distance/score.
    pub distance: f32,
}

/// Feature detector and descriptor.
pub struct FeatureDetector {
    detector_type: FeatureDetectorType,
    max_features: usize,
    threshold: f32,
    num_octaves: usize,
    scales_per_octave: usize,
}

impl FeatureDetector {
    /// Create a new feature detector.
    #[must_use]
    pub const fn new(detector_type: FeatureDetectorType) -> Self {
        Self {
            detector_type,
            max_features: 500,
            threshold: 0.01,
            num_octaves: 4,
            scales_per_octave: 3,
        }
    }

    /// Set maximum number of features.
    #[must_use]
    pub const fn with_max_features(mut self, max_features: usize) -> Self {
        self.max_features = max_features;
        self
    }

    /// Set detection threshold.
    #[must_use]
    pub const fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Detect keypoints in an image.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions are invalid.
    pub fn detect(&self, image: &[u8], width: u32, height: u32) -> CvResult<Vec<Keypoint>> {
        if width == 0 || height == 0 {
            return Err(CvError::invalid_dimensions(width, height));
        }

        match self.detector_type {
            FeatureDetectorType::Sift => self.detect_sift(image, width, height),
            FeatureDetectorType::Surf => self.detect_surf(image, width, height),
            FeatureDetectorType::Orb => self.detect_orb(image, width, height),
            FeatureDetectorType::Harris => self.detect_harris(image, width, height),
        }
    }

    /// Compute descriptors for keypoints.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions are invalid.
    pub fn compute(
        &self,
        image: &[u8],
        width: u32,
        height: u32,
        keypoints: &[Keypoint],
    ) -> CvResult<Vec<Descriptor>> {
        if width == 0 || height == 0 {
            return Err(CvError::invalid_dimensions(width, height));
        }

        match self.detector_type {
            FeatureDetectorType::Sift => self.compute_sift(image, width, height, keypoints),
            FeatureDetectorType::Surf => self.compute_surf(image, width, height, keypoints),
            FeatureDetectorType::Orb => self.compute_orb(image, width, height, keypoints),
            FeatureDetectorType::Harris => self.compute_brief(image, width, height, keypoints),
        }
    }

    /// Detect and compute in one step.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions are invalid.
    pub fn detect_and_compute(
        &self,
        image: &[u8],
        width: u32,
        height: u32,
    ) -> CvResult<(Vec<Keypoint>, Vec<Descriptor>)> {
        let keypoints = self.detect(image, width, height)?;
        let descriptors = self.compute(image, width, height, &keypoints)?;
        Ok((keypoints, descriptors))
    }

    /// SIFT-like detection using DoG (Difference of Gaussians).
    fn detect_sift(&self, image: &[u8], width: u32, height: u32) -> CvResult<Vec<Keypoint>> {
        let mut keypoints = Vec::new();

        // Build Gaussian pyramid
        let pyramid = build_gaussian_pyramid(image, width, height, self.num_octaves, self.scales_per_octave);

        // Detect DoG extrema
        for octave in 0..self.num_octaves {
            let octave_start = octave * self.scales_per_octave;
            for scale in 1..self.scales_per_octave - 1 {
                let idx = octave_start + scale;
                if idx + 1 >= pyramid.len() {
                    continue;
                }

                let (prev_img, prev_w, prev_h) = &pyramid[idx - 1];
                let (curr_img, curr_w, curr_h) = &pyramid[idx];
                let (next_img, next_w, next_h) = &pyramid[idx + 1];

                // Find local extrema in DoG
                for y in 1..curr_h - 1 {
                    for x in 1..curr_w - 1 {
                        let center_idx = (y * curr_w + x) as usize;
                        let center = curr_img[center_idx] as f32;

                        // Check if local maximum or minimum
                        let mut is_extremum = true;
                        for dy in -1..=1 {
                            for dx in -1..=1 {
                                let nx = (x as i32 + dx) as u32;
                                let ny = (y as i32 + dy) as u32;
                                let nidx = (ny * curr_w + nx) as usize;

                                if curr_img[nidx] as f32 > center {
                                    is_extremum = false;
                                    break;
                                }
                            }
                            if !is_extremum {
                                break;
                            }
                        }

                        if is_extremum && center > self.threshold * 255.0 {
                            let scale_factor = 1.0 / (1 << octave) as f32;
                            let kp = Keypoint::new(
                                x as f32 / scale_factor,
                                y as f32 / scale_factor,
                                (1 << octave) as f32 * 1.6,
                                0.0,
                                center,
                                octave as i32,
                            );
                            keypoints.push(kp);
                        }
                    }
                }
            }
        }

        // Sort by response and limit
        keypoints.sort_by(|a, b| b.response.partial_cmp(&a.response).unwrap_or(std::cmp::Ordering::Equal));
        keypoints.truncate(self.max_features);

        Ok(keypoints)
    }

    /// SURF-like detection using Hessian.
    fn detect_surf(&self, image: &[u8], width: u32, height: u32) -> CvResult<Vec<Keypoint>> {
        let mut keypoints = Vec::new();
        let w = width as i32;
        let h = height as i32;

        // Compute integral image
        let integral = compute_integral_image(image, width, height);

        // Multi-scale detection
        for octave in 0..self.num_octaves {
            let scale = (1 << octave) * 9;
            let half = scale / 2;

            for y in half..h - half {
                for x in half..w - half {
                    // Compute Hessian determinant using box filters
                    let dxx = box_filter(&integral, width, x, y, scale, scale / 3) as f32;
                    let dyy = box_filter(&integral, width, x, y, scale / 3, scale) as f32;
                    let dxy = box_filter(&integral, width, x, y, scale / 2, scale / 2) as f32;

                    let det = dxx * dyy - 0.81 * dxy * dxy;

                    if det > self.threshold * 1000.0 {
                        let kp = Keypoint::new(
                            x as f32,
                            y as f32,
                            scale as f32,
                            0.0,
                            det,
                            octave as i32,
                        );
                        keypoints.push(kp);
                    }
                }
            }
        }

        // Non-maximum suppression
        keypoints = non_maximum_suppression(&keypoints, 10.0);

        // Sort and limit
        keypoints.sort_by(|a, b| b.response.partial_cmp(&a.response).unwrap_or(std::cmp::Ordering::Equal));
        keypoints.truncate(self.max_features);

        Ok(keypoints)
    }

    /// ORB-like detection using FAST corners.
    fn detect_orb(&self, image: &[u8], width: u32, height: u32) -> CvResult<Vec<Keypoint>> {
        let mut keypoints = Vec::new();

        // Build pyramid
        let pyramid = build_pyramid(image, width, height, self.num_octaves);

        for (octave, (img, w, h)) in pyramid.iter().enumerate() {
            let kps = detect_fast(img, *w, *h, self.threshold)?;

            let scale_factor = 1.0 / (1 << octave) as f32;
            for kp in kps {
                let scaled_kp = Keypoint::new(
                    kp.x / scale_factor,
                    kp.y / scale_factor,
                    kp.scale / scale_factor,
                    kp.angle,
                    kp.response,
                    octave as i32,
                );
                keypoints.push(scaled_kp);
            }
        }

        // Sort and limit
        keypoints.sort_by(|a, b| b.response.partial_cmp(&a.response).unwrap_or(std::cmp::Ordering::Equal));
        keypoints.truncate(self.max_features);

        // Compute orientations
        for kp in &mut keypoints {
            kp.angle = compute_orientation(image, width, height, kp.x as u32, kp.y as u32);
        }

        Ok(keypoints)
    }

    /// Harris corner detection.
    fn detect_harris(&self, image: &[u8], width: u32, height: u32) -> CvResult<Vec<Keypoint>> {
        let mut keypoints = Vec::new();
        let w = width as i32;
        let h = height as i32;

        // Compute gradients
        let (grad_x, grad_y) = compute_gradients(image, width, height);

        // Compute Harris response
        let window = 3;
        for y in window..h - window {
            for x in window..w - window {
                let mut m_xx = 0.0;
                let mut m_xy = 0.0;
                let mut m_yy = 0.0;

                // Sum over window
                for dy in -window..=window {
                    for dx in -window..=window {
                        let px = x + dx;
                        let py = y + dy;
                        let idx = (py * w + px) as usize;

                        let gx = grad_x[idx];
                        let gy = grad_y[idx];

                        m_xx += gx * gx;
                        m_xy += gx * gy;
                        m_yy += gy * gy;
                    }
                }

                // Harris response: det(M) - k * trace(M)^2
                let det = m_xx * m_yy - m_xy * m_xy;
                let trace = m_xx + m_yy;
                let response = det - 0.04 * trace * trace;

                if response > self.threshold * 10000.0 {
                    let kp = Keypoint::new(x as f32, y as f32, 1.0, 0.0, response as f32, 0);
                    keypoints.push(kp);
                }
            }
        }

        // Non-maximum suppression
        keypoints = non_maximum_suppression(&keypoints, 5.0);

        // Sort and limit
        keypoints.sort_by(|a, b| b.response.partial_cmp(&a.response).unwrap_or(std::cmp::Ordering::Equal));
        keypoints.truncate(self.max_features);

        Ok(keypoints)
    }

    /// Compute SIFT-like descriptors.
    #[allow(clippy::too_many_arguments)]
    fn compute_sift(
        &self,
        image: &[u8],
        width: u32,
        height: u32,
        keypoints: &[Keypoint],
    ) -> CvResult<Vec<Descriptor>> {
        let mut descriptors = Vec::new();

        for (idx, kp) in keypoints.iter().enumerate() {
            let descriptor_data = compute_sift_descriptor(image, width, height, kp);
            descriptors.push(Descriptor {
                data: descriptor_data,
                keypoint_idx: idx,
            });
        }

        Ok(descriptors)
    }

    /// Compute SURF-like descriptors.
    #[allow(clippy::too_many_arguments)]
    fn compute_surf(
        &self,
        image: &[u8],
        width: u32,
        height: u32,
        keypoints: &[Keypoint],
    ) -> CvResult<Vec<Descriptor>> {
        let integral = compute_integral_image(image, width, height);
        let mut descriptors = Vec::new();

        for (idx, kp) in keypoints.iter().enumerate() {
            let descriptor_data = compute_surf_descriptor(&integral, width, height, kp);
            descriptors.push(Descriptor {
                data: descriptor_data,
                keypoint_idx: idx,
            });
        }

        Ok(descriptors)
    }

    /// Compute ORB descriptors (rBRIEF).
    #[allow(clippy::too_many_arguments)]
    fn compute_orb(
        &self,
        image: &[u8],
        width: u32,
        height: u32,
        keypoints: &[Keypoint],
    ) -> CvResult<Vec<Descriptor>> {
        let mut descriptors = Vec::new();

        for (idx, kp) in keypoints.iter().enumerate() {
            let descriptor_data = compute_orb_descriptor(image, width, height, kp);
            descriptors.push(Descriptor {
                data: descriptor_data,
                keypoint_idx: idx,
            });
        }

        Ok(descriptors)
    }

    /// Compute BRIEF descriptors.
    #[allow(clippy::too_many_arguments)]
    fn compute_brief(
        &self,
        image: &[u8],
        width: u32,
        height: u32,
        keypoints: &[Keypoint],
    ) -> CvResult<Vec<Descriptor>> {
        self.compute_orb(image, width, height, keypoints)
    }
}

/// Feature matcher.
pub struct FeatureMatcher {
    ratio_threshold: f32,
    cross_check: bool,
}

impl FeatureMatcher {
    /// Create a new feature matcher.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            ratio_threshold: 0.75,
            cross_check: true,
        }
    }

    /// Set ratio test threshold (Lowe's ratio test).
    #[must_use]
    pub const fn with_ratio_threshold(mut self, threshold: f32) -> Self {
        self.ratio_threshold = threshold;
        self
    }

    /// Enable/disable cross-check filtering.
    #[must_use]
    pub const fn with_cross_check(mut self, enabled: bool) -> Self {
        self.cross_check = enabled;
        self
    }

    /// Match descriptors using brute-force method.
    #[must_use]
    pub fn match_descriptors(
        &self,
        query: &[Descriptor],
        train: &[Descriptor],
    ) -> Vec<Match> {
        let mut matches = Vec::new();

        for (q_idx, q_desc) in query.iter().enumerate() {
            let mut best_dist = f32::MAX;
            let mut second_best_dist = f32::MAX;
            let mut best_idx = 0;

            for (t_idx, t_desc) in train.iter().enumerate() {
                let dist = q_desc.distance(t_desc);
                if dist < best_dist {
                    second_best_dist = best_dist;
                    best_dist = dist;
                    best_idx = t_idx;
                } else if dist < second_best_dist {
                    second_best_dist = dist;
                }
            }

            // Lowe's ratio test
            if best_dist < self.ratio_threshold * second_best_dist {
                matches.push(Match {
                    query_idx: q_idx,
                    train_idx: best_idx,
                    distance: best_dist,
                });
            }
        }

        // Cross-check filtering
        if self.cross_check {
            matches = self.apply_cross_check(&matches, query, train);
        }

        matches
    }

    /// Apply cross-check filtering.
    fn apply_cross_check(
        &self,
        matches: &[Match],
        query: &[Descriptor],
        train: &[Descriptor],
    ) -> Vec<Match> {
        let mut filtered = Vec::new();

        for m in matches {
            // Check reverse match
            let t_desc = &train[m.train_idx];
            let mut best_dist = f32::MAX;
            let mut best_idx = 0;

            for (q_idx, q_desc) in query.iter().enumerate() {
                let dist = t_desc.distance(q_desc);
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = q_idx;
                }
            }

            if best_idx == m.query_idx {
                filtered.push(*m);
            }
        }

        filtered
    }
}

impl Default for FeatureMatcher {
    fn default() -> Self {
        Self::new()
    }
}

/// Estimate homography using DLT (Direct Linear Transform).
///
/// # Errors
///
/// Returns an error if there are insufficient points.
pub fn estimate_homography(
    src_points: &[(f64, f64)],
    dst_points: &[(f64, f64)],
) -> CvResult<TransformMatrix> {
    if src_points.len() < 4 || src_points.len() != dst_points.len() {
        return Err(CvError::computation("need at least 4 point correspondences"));
    }

    let n = src_points.len();
    let mut a = vec![0.0; 2 * n * 9];

    for i in 0..n {
        let (x, y) = src_points[i];
        let (x_prime, y_prime) = dst_points[i];

        let row1 = i * 2;
        let row2 = i * 2 + 1;

        a[row1 * 9 + 0] = x;
        a[row1 * 9 + 1] = y;
        a[row1 * 9 + 2] = 1.0;
        a[row1 * 9 + 6] = -x * x_prime;
        a[row1 * 9 + 7] = -y * x_prime;
        a[row1 * 9 + 8] = -x_prime;

        a[row2 * 9 + 3] = x;
        a[row2 * 9 + 4] = y;
        a[row2 * 9 + 5] = 1.0;
        a[row2 * 9 + 6] = -x * y_prime;
        a[row2 * 9 + 7] = -y * y_prime;
        a[row2 * 9 + 8] = -y_prime;
    }

    // Solve using SVD (simplified)
    let h = solve_dlt(&a, 2 * n, 9)?;

    Ok(TransformMatrix {
        data: h,
        transform_type: TransformationType::Homography,
    })
}

/// Estimate affine transformation.
///
/// # Errors
///
/// Returns an error if there are insufficient points.
pub fn estimate_affine(
    src_points: &[(f64, f64)],
    dst_points: &[(f64, f64)],
) -> CvResult<TransformMatrix> {
    if src_points.len() < 3 || src_points.len() != dst_points.len() {
        return Err(CvError::computation("need at least 3 point correspondences"));
    }

    // Solve for affine parameters using least squares
    let n = src_points.len();
    let mut at_a = [0.0; 36];
    let mut at_b = [0.0; 6];

    for i in 0..n {
        let (x, y) = src_points[i];
        let (x_prime, y_prime) = dst_points[i];

        // Build normal equations
        let row = [x, y, 1.0, 0.0, 0.0, 0.0];
        for j in 0..6 {
            for k in 0..6 {
                at_a[j * 6 + k] += row[j] * row[k];
            }
            at_b[j] += row[j] * x_prime;
        }

        let row = [0.0, 0.0, 0.0, x, y, 1.0];
        for j in 0..6 {
            for k in 0..6 {
                at_a[j * 6 + k] += row[j] * row[k];
            }
            at_b[j] += row[j] * y_prime;
        }
    }

    // Solve 6x6 system
    let params = solve_linear_system_6x6(&at_a, &at_b)?;

    Ok(TransformMatrix {
        data: [
            params[0], params[1], params[2], params[3], params[4], params[5], 0.0, 0.0, 1.0,
        ],
        transform_type: TransformationType::Affine,
    })
}

/// RANSAC-based robust homography estimation.
///
/// # Errors
///
/// Returns an error if estimation fails.
#[allow(clippy::too_many_arguments)]
pub fn ransac_homography(
    src_points: &[(f64, f64)],
    dst_points: &[(f64, f64)],
    threshold: f64,
    max_iterations: usize,
    confidence: f64,
) -> CvResult<(TransformMatrix, Vec<bool>)> {
    if src_points.len() < 4 {
        return Err(CvError::computation("insufficient points for RANSAC"));
    }

    let mut best_inliers = Vec::new();
    let mut best_model = TransformMatrix::identity();
    let mut best_count = 0;

    for _ in 0..max_iterations {
        // Randomly sample 4 points
        let indices = random_sample(src_points.len(), 4);
        let sample_src: Vec<_> = indices.iter().map(|&i| src_points[i]).collect();
        let sample_dst: Vec<_> = indices.iter().map(|&i| dst_points[i]).collect();

        // Estimate model
        if let Ok(model) = estimate_homography(&sample_src, &sample_dst) {
            // Count inliers
            let mut inliers = vec![false; src_points.len()];
            let mut count = 0;

            for i in 0..src_points.len() {
                let (x, y) = src_points[i];
                let (x_prime, y_prime) = model.transform_point(x, y);
                let (x_expected, y_expected) = dst_points[i];

                let error = ((x_prime - x_expected).powi(2) + (y_prime - y_expected).powi(2)).sqrt();
                if error < threshold {
                    inliers[i] = true;
                    count += 1;
                }
            }

            if count > best_count {
                best_count = count;
                best_inliers = inliers;
                best_model = model;

                // Early termination check
                let inlier_ratio = count as f64 / src_points.len() as f64;
                if inlier_ratio > confidence {
                    break;
                }
            }
        }
    }

    if best_count < 4 {
        return Err(CvError::computation("RANSAC failed to find sufficient inliers"));
    }

    // Refine using all inliers
    let inlier_src: Vec<_> = src_points
        .iter()
        .enumerate()
        .filter(|(i, _)| best_inliers[*i])
        .map(|(_, p)| *p)
        .collect();
    let inlier_dst: Vec<_> = dst_points
        .iter()
        .enumerate()
        .filter(|(i, _)| best_inliers[*i])
        .map(|(_, p)| *p)
        .collect();

    let refined_model = estimate_homography(&inlier_src, &inlier_dst)?;

    Ok((refined_model, best_inliers))
}

/// Register images using feature-based method.
///
/// # Errors
///
/// Returns an error if registration fails.
#[allow(clippy::too_many_arguments)]
pub fn register_feature_based(
    reference: &[u8],
    target: &[u8],
    width: u32,
    height: u32,
    transform_type: TransformationType,
) -> CvResult<(TransformMatrix, RegistrationQuality)> {
    // Detect and compute features
    let detector = FeatureDetector::new(FeatureDetectorType::Orb).with_max_features(1000);

    let (ref_kps, ref_descs) = detector.detect_and_compute(reference, width, height)?;
    let (tgt_kps, tgt_descs) = detector.detect_and_compute(target, width, height)?;

    if ref_kps.is_empty() || tgt_kps.is_empty() {
        return Err(CvError::computation("no features detected"));
    }

    // Match features
    let matcher = FeatureMatcher::new();
    let matches = matcher.match_descriptors(&ref_descs, &tgt_descs);

    if matches.len() < 4 {
        return Err(CvError::computation("insufficient matches"));
    }

    // Extract point correspondences
    let src_points: Vec<_> = matches
        .iter()
        .map(|m| {
            let kp = &ref_kps[m.query_idx];
            (kp.x as f64, kp.y as f64)
        })
        .collect();

    let dst_points: Vec<_> = matches
        .iter()
        .map(|m| {
            let kp = &tgt_kps[m.train_idx];
            (kp.x as f64, kp.y as f64)
        })
        .collect();

    // Estimate transformation with RANSAC
    let (transform, inliers) = match transform_type {
        TransformationType::Homography => {
            ransac_homography(&src_points, &dst_points, 3.0, 1000, 0.9)?
        }
        TransformationType::Affine => {
            let (model, inliers) = ransac_affine(&src_points, &dst_points, 3.0, 1000)?;
            (model, inliers)
        }
        _ => ransac_homography(&src_points, &dst_points, 3.0, 1000, 0.9)?,
    };

    // Compute quality metrics
    let inlier_count = inliers.iter().filter(|&&x| x).count();
    let mut rmse = 0.0;
    let mut count = 0;

    for i in 0..src_points.len() {
        if inliers[i] {
            let (x, y) = src_points[i];
            let (x_prime, y_prime) = transform.transform_point(x, y);
            let (x_exp, y_exp) = dst_points[i];
            rmse += (x_prime - x_exp).powi(2) + (y_prime - y_exp).powi(2);
            count += 1;
        }
    }

    rmse = if count > 0 {
        (rmse / count as f64).sqrt()
    } else {
        f64::MAX
    };

    let quality = RegistrationQuality {
        success: inlier_count >= 10,
        rmse,
        inliers: inlier_count,
        confidence: inlier_count as f64 / matches.len() as f64,
        iterations: 0,
    };

    Ok((transform, quality))
}

// Helper functions

fn ransac_affine(
    src_points: &[(f64, f64)],
    dst_points: &[(f64, f64)],
    threshold: f64,
    max_iterations: usize,
) -> CvResult<(TransformMatrix, Vec<bool>)> {
    let mut best_inliers = Vec::new();
    let mut best_model = TransformMatrix::identity();
    let mut best_count = 0;

    for _ in 0..max_iterations {
        let indices = random_sample(src_points.len(), 3);
        let sample_src: Vec<_> = indices.iter().map(|&i| src_points[i]).collect();
        let sample_dst: Vec<_> = indices.iter().map(|&i| dst_points[i]).collect();

        if let Ok(model) = estimate_affine(&sample_src, &sample_dst) {
            let mut inliers = vec![false; src_points.len()];
            let mut count = 0;

            for i in 0..src_points.len() {
                let (x, y) = src_points[i];
                let (x_prime, y_prime) = model.transform_point(x, y);
                let (x_expected, y_expected) = dst_points[i];

                let error =
                    ((x_prime - x_expected).powi(2) + (y_prime - y_expected).powi(2)).sqrt();
                if error < threshold {
                    inliers[i] = true;
                    count += 1;
                }
            }

            if count > best_count {
                best_count = count;
                best_inliers = inliers;
                best_model = model;
            }
        }
    }

    if best_count < 3 {
        return Err(CvError::computation("RANSAC affine failed"));
    }

    Ok((best_model, best_inliers))
}

fn build_gaussian_pyramid(
    image: &[u8],
    width: u32,
    height: u32,
    octaves: usize,
    scales: usize,
) -> Vec<(Vec<u8>, u32, u32)> {
    let mut pyramid = Vec::new();
    pyramid.push((image.to_vec(), width, height));

    for _ in 1..octaves * scales {
        let Some((prev_img, prev_w, prev_h)) = pyramid.last() else {
            break;
        };
        let blurred = gaussian_blur(prev_img, *prev_w, *prev_h, 1.6);
        pyramid.push((blurred, *prev_w, *prev_h));
    }

    pyramid
}

fn build_pyramid(image: &[u8], width: u32, height: u32, levels: usize) -> Vec<(Vec<u8>, u32, u32)> {
    let mut pyramid = Vec::new();
    pyramid.push((image.to_vec(), width, height));

    for _ in 1..levels {
        let Some((prev_img, prev_w, prev_h)) = pyramid.last() else {
            break;
        };
        if *prev_w < 8 || *prev_h < 8 {
            break;
        }
        let downsampled = downsample_image(prev_img, *prev_w, *prev_h);
        pyramid.push((downsampled, prev_w / 2, prev_h / 2));
    }

    pyramid
}

fn downsample_image(image: &[u8], width: u32, height: u32) -> Vec<u8> {
    let new_w = width / 2;
    let new_h = height / 2;
    let mut result = vec![0u8; (new_w * new_h) as usize];

    for y in 0..new_h {
        for x in 0..new_w {
            let sx = x * 2;
            let sy = y * 2;
            let idx1 = (sy * width + sx) as usize;
            let idx2 = (sy * width + sx + 1) as usize;
            let idx3 = ((sy + 1) * width + sx) as usize;
            let idx4 = ((sy + 1) * width + sx + 1) as usize;

            let avg = (image[idx1] as u32 + image[idx2] as u32 + image[idx3] as u32
                + image[idx4] as u32)
                / 4;
            result[(y * new_w + x) as usize] = avg as u8;
        }
    }

    result
}

fn gaussian_blur(image: &[u8], width: u32, height: u32, _sigma: f64) -> Vec<u8> {
    // Simplified Gaussian blur using box filter approximation
    let kernel = [1u32, 2, 1, 2, 4, 2, 1, 2, 1];
    let sum = 16;

    let mut result = vec![0u8; image.len()];

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let mut val = 0u32;
            for dy in 0..3 {
                for dx in 0..3 {
                    let px = x + dx - 1;
                    let py = y + dy - 1;
                    let idx = (py * width + px) as usize;
                    val += image[idx] as u32 * kernel[dy as usize * 3 + dx as usize];
                }
            }
            result[(y * width + x) as usize] = (val / sum) as u8;
        }
    }

    result
}

fn compute_integral_image(image: &[u8], width: u32, height: u32) -> Vec<u64> {
    let mut integral = vec![0u64; (width * height) as usize];

    for y in 0..height {
        let mut row_sum = 0u64;
        for x in 0..width {
            let idx = (y * width + x) as usize;
            row_sum += image[idx] as u64;
            if y > 0 {
                let prev_idx = ((y - 1) * width + x) as usize;
                integral[idx] = row_sum + integral[prev_idx];
            } else {
                integral[idx] = row_sum;
            }
        }
    }

    integral
}

fn box_filter(integral: &[u64], width: u32, cx: i32, cy: i32, w: i32, h: i32) -> i64 {
    let x1 = (cx - w / 2).max(0) as u32;
    let y1 = (cy - h / 2).max(0) as u32;
    let x2 = (cx + w / 2).min(width as i32 - 1) as u32;
    let y2 = (cy + h / 2).min(width as i32 - 1) as u32;

    let a = if x1 > 0 && y1 > 0 {
        integral[((y1 - 1) * width + x1 - 1) as usize] as i64
    } else {
        0
    };
    let b = if y1 > 0 {
        integral[((y1 - 1) * width + x2) as usize] as i64
    } else {
        0
    };
    let c = if x1 > 0 {
        integral[(y2 * width + x1 - 1) as usize] as i64
    } else {
        0
    };
    let d = integral[(y2 * width + x2) as usize] as i64;

    d - b - c + a
}

fn detect_fast(image: &[u8], width: u32, height: u32, threshold: f32) -> CvResult<Vec<Keypoint>> {
    let mut keypoints = Vec::new();
    let t = (threshold * 255.0) as i32;

    // FAST-9 circle offsets
    let offsets = [
        (0, 3),
        (1, 3),
        (2, 2),
        (3, 1),
        (3, 0),
        (3, -1),
        (2, -2),
        (1, -3),
        (0, -3),
        (-1, -3),
        (-2, -2),
        (-3, -1),
        (-3, 0),
        (-3, 1),
        (-2, 2),
        (-1, 3),
    ];

    for y in 3..height - 3 {
        for x in 3..width - 3 {
            let center = image[(y * width + x) as usize] as i32;
            let mut brighter = 0;
            let mut darker = 0;

            for (dx, dy) in &offsets {
                let px = (x as i32 + dx) as u32;
                let py = (y as i32 + dy) as u32;
                let val = image[(py * width + px) as usize] as i32;

                if val > center + t {
                    brighter += 1;
                    darker = 0;
                } else if val < center - t {
                    darker += 1;
                    brighter = 0;
                } else {
                    brighter = 0;
                    darker = 0;
                }

                if brighter >= 9 || darker >= 9 {
                    keypoints.push(Keypoint::new(
                        x as f32,
                        y as f32,
                        1.0,
                        0.0,
                        center.abs() as f32,
                        0,
                    ));
                    break;
                }
            }
        }
    }

    Ok(keypoints)
}

fn compute_orientation(image: &[u8], width: u32, height: u32, x: u32, y: u32) -> f32 {
    let mut mx = 0.0;
    let mut my = 0.0;
    let radius = 7;

    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let px = x as i32 + dx;
            let py = y as i32 + dy;

            if px >= 0 && px < width as i32 && py >= 0 && py < height as i32 {
                let val = image[(py as u32 * width + px as u32) as usize] as f32;
                mx += dx as f32 * val;
                my += dy as f32 * val;
            }
        }
    }

    my.atan2(mx)
}

fn compute_gradients(image: &[u8], width: u32, height: u32) -> (Vec<f64>, Vec<f64>) {
    let size = (width * height) as usize;
    let mut grad_x = vec![0.0; size];
    let mut grad_y = vec![0.0; size];

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let idx = (y * width + x) as usize;
            let idx_l = (y * width + x - 1) as usize;
            let idx_r = (y * width + x + 1) as usize;
            let idx_t = ((y - 1) * width + x) as usize;
            let idx_b = ((y + 1) * width + x) as usize;

            grad_x[idx] = (image[idx_r] as f64 - image[idx_l] as f64) / 2.0;
            grad_y[idx] = (image[idx_b] as f64 - image[idx_t] as f64) / 2.0;
        }
    }

    (grad_x, grad_y)
}

fn non_maximum_suppression(keypoints: &[Keypoint], radius: f32) -> Vec<Keypoint> {
    let mut result = Vec::new();

    for i in 0..keypoints.len() {
        let kp = keypoints[i];
        let mut is_maximum = true;

        for j in 0..keypoints.len() {
            if i == j {
                continue;
            }
            let other = keypoints[j];
            let dx = kp.x - other.x;
            let dy = kp.y - other.y;
            let dist = (dx * dx + dy * dy).sqrt();

            if dist < radius && other.response > kp.response {
                is_maximum = false;
                break;
            }
        }

        if is_maximum {
            result.push(kp);
        }
    }

    result
}

fn compute_sift_descriptor(image: &[u8], width: u32, height: u32, kp: &Keypoint) -> Vec<f32> {
    let mut descriptor = vec![0.0f32; 128];
    let patch_size = 16;
    let bins = 8;

    let cx = kp.x as i32;
    let cy = kp.y as i32;

    for dy in -patch_size..patch_size {
        for dx in -patch_size..patch_size {
            let px = cx + dx;
            let py = cy + dy;

            if px >= 1 && px < width as i32 - 1 && py >= 1 && py < height as i32 - 1 {
                let idx = (py * width as i32 + px) as usize;
                let idx_r = (py * width as i32 + px + 1) as usize;
                let idx_b = ((py + 1) * width as i32 + px) as usize;

                let gx = image[idx_r] as f32 - image[idx] as f32;
                let gy = image[idx_b] as f32 - image[idx] as f32;

                let mag = (gx * gx + gy * gy).sqrt();
                let mut angle = gy.atan2(gx);
                angle = (angle + 2.0 * PI as f32) % (2.0 * PI as f32);

                let bin = ((angle / (2.0 * PI as f32)) * bins as f32) as usize % bins;
                let subregion_x = ((dx + patch_size) / (patch_size / 2)) as usize;
                let subregion_y = ((dy + patch_size) / (patch_size / 2)) as usize;

                if subregion_x < 4 && subregion_y < 4 {
                    let desc_idx = (subregion_y * 4 + subregion_x) * bins + bin;
                    if desc_idx < 128 {
                        descriptor[desc_idx] += mag;
                    }
                }
            }
        }
    }

    // Normalize
    let norm: f32 = descriptor.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for val in &mut descriptor {
            *val /= norm;
        }
    }

    descriptor
}

fn compute_surf_descriptor(integral: &[u64], width: u32, height: u32, kp: &Keypoint) -> Vec<f32> {
    let mut descriptor = vec![0.0f32; 64];
    let scale = kp.scale as i32;
    let cx = kp.x as i32;
    let cy = kp.y as i32;

    for i in 0..4 {
        for j in 0..4 {
            let x = cx + (j - 2) * scale;
            let y = cy + (i - 2) * scale;

            let dx = box_filter(integral, width, x, y, scale, scale / 2) as f32;
            let dy = box_filter(integral, width, x, y, scale / 2, scale) as f32;

            let idx = (i * 4 + j) as usize * 4;
            if idx + 3 < descriptor.len() {
                descriptor[idx] = dx;
                descriptor[idx + 1] = dx.abs();
                descriptor[idx + 2] = dy;
                descriptor[idx + 3] = dy.abs();
            }
        }
    }

    descriptor
}

fn compute_orb_descriptor(image: &[u8], width: u32, height: u32, kp: &Keypoint) -> Vec<f32> {
    let mut descriptor = vec![0.0f32; 256];
    let pairs = generate_brief_pairs();

    let cos_a = kp.angle.cos();
    let sin_a = kp.angle.sin();

    for (idx, (p1, p2)) in pairs.iter().enumerate() {
        // Rotate sample points
        let x1 = kp.x + cos_a * p1.0 - sin_a * p1.1;
        let y1 = kp.y + sin_a * p1.0 + cos_a * p1.1;
        let x2 = kp.x + cos_a * p2.0 - sin_a * p2.1;
        let y2 = kp.y + sin_a * p2.0 + cos_a * p2.1;

        if x1 >= 0.0
            && x1 < width as f32
            && y1 >= 0.0
            && y1 < height as f32
            && x2 >= 0.0
            && x2 < width as f32
            && y2 >= 0.0
            && y2 < height as f32
        {
            let idx1 = (y1 as u32 * width + x1 as u32) as usize;
            let idx2 = (y2 as u32 * width + x2 as u32) as usize;

            if idx1 < image.len() && idx2 < image.len() {
                descriptor[idx] = if image[idx1] < image[idx2] { 1.0 } else { 0.0 };
            }
        }
    }

    descriptor
}

fn generate_brief_pairs() -> Vec<((f32, f32), (f32, f32))> {
    let mut pairs = Vec::new();
    // Simplified: generate fixed pattern (in practice, use predefined optimal pattern)
    for i in 0..256 {
        let angle1 = (i as f32 * 0.1) % (2.0 * PI as f32);
        let angle2 = ((i + 128) as f32 * 0.1) % (2.0 * PI as f32);
        let r = 10.0;
        pairs.push((
            (r * angle1.cos(), r * angle1.sin()),
            (r * angle2.cos(), r * angle2.sin()),
        ));
    }
    pairs
}

fn random_sample(n: usize, k: usize) -> Vec<usize> {
    // Simple pseudo-random sampling
    let mut result = Vec::new();
    let mut used = vec![false; n];
    let mut seed = 12345u64;

    while result.len() < k && result.len() < n {
        seed = (seed.wrapping_mul(1103515245).wrapping_add(12345)) & 0x7fffffff;
        let idx = (seed as usize) % n;

        if !used[idx] {
            result.push(idx);
            used[idx] = true;
        }
    }

    result
}

fn solve_dlt(a: &[f64], rows: usize, cols: usize) -> CvResult<[f64; 9]> {
    // Simplified SVD solution - in practice, use a proper linear algebra library
    // For now, use a simplified least squares approach

    if rows < cols {
        return Err(CvError::computation("underdetermined system"));
    }

    // Use normal equations A^T * A * x = 0
    let mut ata = vec![0.0; cols * cols];
    for i in 0..cols {
        for j in 0..cols {
            for k in 0..rows {
                ata[i * cols + j] += a[k * cols + i] * a[k * cols + j];
            }
        }
    }

    // Find eigenvector for smallest eigenvalue (simplified - just use last column)
    let mut h = [0.0; 9];
    for i in 0..9 {
        h[i] = if i < cols { ata[i * cols + (cols - 1)] } else { 0.0 };
    }

    // Normalize
    if h[8].abs() > f64::EPSILON {
        for val in &mut h {
            *val /= h[8];
        }
    }

    Ok(h)
}

fn solve_linear_system_6x6(a: &[f64; 36], b: &[f64; 6]) -> CvResult<[f64; 6]> {
    // Gaussian elimination for 6x6 system
    let mut aug = [[0.0; 7]; 6];

    for i in 0..6 {
        for j in 0..6 {
            aug[i][j] = a[i * 6 + j];
        }
        aug[i][6] = b[i];
    }

    // Forward elimination
    for i in 0..6 {
        // Find pivot
        let mut max_row = i;
        for k in i + 1..6 {
            if aug[k][i].abs() > aug[max_row][i].abs() {
                max_row = k;
            }
        }
        aug.swap(i, max_row);

        if aug[i][i].abs() < f64::EPSILON {
            return Err(CvError::computation("singular matrix in affine estimation"));
        }

        for k in i + 1..6 {
            let factor = aug[k][i] / aug[i][i];
            for j in i..7 {
                aug[k][j] -= factor * aug[i][j];
            }
        }
    }

    // Back substitution
    let mut x = [0.0; 6];
    for i in (0..6).rev() {
        x[i] = aug[i][6];
        for j in i + 1..6 {
            x[i] -= aug[i][j] * x[j];
        }
        x[i] /= aug[i][i];
    }

    Ok(x)
}
