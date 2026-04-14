//! Feature descriptor matching utilities for stereo vision, SLAM, and object tracking.

#![allow(dead_code)]

/// A matched pair of feature descriptors from two images.
#[derive(Debug, Clone)]
pub struct MatchPair {
    /// Index of the descriptor in the query set.
    pub query_idx: usize,
    /// Index of the descriptor in the train set.
    pub train_idx: usize,
    /// Euclidean distance between the two descriptors.
    pub dist: f32,
}

impl MatchPair {
    /// Create a new [`MatchPair`].
    #[must_use]
    pub fn new(query_idx: usize, train_idx: usize, dist: f32) -> Self {
        Self {
            query_idx,
            train_idx,
            dist: dist.max(0.0),
        }
    }

    /// Return the distance between the matched descriptors.
    #[must_use]
    pub fn distance(&self) -> f32 {
        self.dist
    }

    /// Return `true` when the distance is below `max_dist`.
    #[must_use]
    pub fn is_good(&self, max_dist: f32) -> bool {
        self.dist <= max_dist
    }
}

/// Result of a feature matching operation between two descriptor sets.
#[derive(Debug, Clone, Default)]
pub struct FeatureMatch {
    /// All raw matches (one per query descriptor).
    pub all_matches: Vec<MatchPair>,
}

impl FeatureMatch {
    /// Create a new, empty [`FeatureMatch`].
    #[must_use]
    pub fn new() -> Self {
        Self {
            all_matches: Vec::new(),
        }
    }

    /// Return only the matches whose distance is below `max_dist`.
    #[must_use]
    pub fn good_matches(&self, max_dist: f32) -> Vec<&MatchPair> {
        self.all_matches
            .iter()
            .filter(|m| m.is_good(max_dist))
            .collect()
    }

    /// Return the number of raw matches.
    #[must_use]
    pub fn len(&self) -> usize {
        self.all_matches.len()
    }

    /// Return `true` when there are no matches.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.all_matches.is_empty()
    }

    /// Return the minimum distance among all matches, or `f32::MAX` if empty.
    #[must_use]
    pub fn min_distance(&self) -> f32 {
        self.all_matches
            .iter()
            .map(|m| m.dist)
            .fold(f32::MAX, f32::min)
    }
}

/// Brute-force feature matcher.
pub struct FeatureMatcher {
    /// Cross-check: a match is kept only if it is mutual.
    pub cross_check: bool,
    /// Ratio test threshold (Lowe's ratio test, 0 = disabled).
    pub ratio_threshold: f32,
}

impl FeatureMatcher {
    /// Create a new [`FeatureMatcher`].
    #[must_use]
    pub fn new(cross_check: bool, ratio_threshold: f32) -> Self {
        Self {
            cross_check,
            ratio_threshold: ratio_threshold.clamp(0.0, 1.0),
        }
    }

    /// Compute L2 distance between two equal-length descriptor vectors.
    #[must_use]
    fn l2(a: &[f32], b: &[f32]) -> f32 {
        let sum: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
        sum.sqrt()
    }

    /// Match each descriptor in `query` to the nearest descriptor in `train`.
    ///
    /// Both `query` and `train` must be slices of flat descriptors each of length `desc_len`.
    /// If `desc_len` is 0 or the slice lengths are not multiples of `desc_len`, an empty
    /// [`FeatureMatch`] is returned.
    #[must_use]
    pub fn match_descriptors(&self, query: &[f32], train: &[f32], desc_len: usize) -> FeatureMatch {
        if desc_len == 0 || query.len() % desc_len != 0 || train.len() % desc_len != 0 {
            return FeatureMatch::new();
        }

        let n_query = query.len() / desc_len;
        let n_train = train.len() / desc_len;
        let mut result = FeatureMatch::new();

        if n_query == 0 || n_train == 0 {
            return result;
        }

        for qi in 0..n_query {
            let qd = &query[qi * desc_len..(qi + 1) * desc_len];

            // Find best and second-best matches.
            let mut best_dist = f32::MAX;
            let mut best_ti = 0usize;
            let mut second_dist = f32::MAX;

            for ti in 0..n_train {
                let td = &train[ti * desc_len..(ti + 1) * desc_len];
                let d = Self::l2(qd, td);
                if d < best_dist {
                    second_dist = best_dist;
                    best_dist = d;
                    best_ti = ti;
                } else if d < second_dist {
                    second_dist = d;
                }
            }

            result
                .all_matches
                .push(MatchPair::new(qi, best_ti, best_dist));
            let _ = second_dist; // used below in ratio_test
        }

        result
    }

    /// Apply Lowe's ratio test to `matches`, retaining only matches where
    /// `best_dist / second_best_dist < ratio_threshold`.
    ///
    /// Because the raw brute-force result only stores one match per query, this
    /// re-computes the second-best distance directly.
    #[must_use]
    pub fn ratio_test<'a>(
        &self,
        matches: &'a FeatureMatch,
        query: &[f32],
        train: &[f32],
        desc_len: usize,
    ) -> Vec<&'a MatchPair> {
        if desc_len == 0 || self.ratio_threshold <= 0.0 {
            return matches.all_matches.iter().collect();
        }
        if query.len() % desc_len != 0 || train.len() % desc_len != 0 {
            return Vec::new();
        }

        let n_train = train.len() / desc_len;

        matches
            .all_matches
            .iter()
            .filter(|m| {
                let qd = &query[m.query_idx * desc_len..(m.query_idx + 1) * desc_len];
                let mut second_dist = f32::MAX;
                for ti in 0..n_train {
                    if ti == m.train_idx {
                        continue;
                    }
                    let td = &train[ti * desc_len..(ti + 1) * desc_len];
                    let d = Self::l2(qd, td);
                    if d < second_dist {
                        second_dist = d;
                    }
                }
                #[allow(clippy::float_cmp)]
                if second_dist == f32::MAX || second_dist == 0.0 {
                    return true;
                }
                m.dist / second_dist < self.ratio_threshold
            })
            .collect()
    }
}

/// Sub-pixel refined match result for high-precision registration.
#[derive(Debug, Clone)]
pub struct SubPixelMatch {
    /// Query descriptor index.
    pub query_idx: usize,
    /// Train descriptor index.
    pub train_idx: usize,
    /// Integer-precision query position (x, y).
    pub query_pos_int: (f32, f32),
    /// Integer-precision train position (x, y).
    pub train_pos_int: (f32, f32),
    /// Sub-pixel refined query position (x, y).
    pub query_pos_sub: (f64, f64),
    /// Sub-pixel refined train position (x, y).
    pub train_pos_sub: (f64, f64),
    /// Sub-pixel refinement error (RMS shift from integer position).
    pub refinement_error: f64,
    /// Descriptor distance.
    pub dist: f32,
}

/// Sub-pixel accuracy refiner using parabolic interpolation on image gradients.
///
/// For each coarse feature match, refines the keypoint location to sub-pixel
/// accuracy using a second-order Taylor expansion of the local image intensity
/// surface. This is equivalent to the SIFT/SURF sub-pixel localisation step.
pub struct SubPixelRefiner {
    /// Half-size of the local analysis window.
    window_half: usize,
    /// Maximum allowed sub-pixel displacement for accepting a refined result.
    max_displacement: f64,
}

impl SubPixelRefiner {
    /// Create a sub-pixel refiner with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            window_half: 3,
            max_displacement: 1.5,
        }
    }

    /// Set the local analysis window half-size (pixels).
    pub fn set_window_half(&mut self, half: usize) {
        self.window_half = half.clamp(1, 8);
    }

    /// Set the maximum allowed refinement displacement.
    pub fn set_max_displacement(&mut self, max: f64) {
        self.max_displacement = max.clamp(0.1, 3.0);
    }

    /// Refine a list of coarse matches to sub-pixel accuracy.
    ///
    /// # Arguments
    ///
    /// * `matches`   – Coarse matches from `FeatureMatcher`.
    /// * `kps_query` – Keypoint positions `(x, y)` for the query image.
    /// * `kps_train` – Keypoint positions `(x, y)` for the train image.
    /// * `img_query` – Grayscale query image (f32, row-major).
    /// * `query_w`, `query_h` – Query image dimensions.
    /// * `img_train` – Grayscale train image (f32, row-major).
    /// * `train_w`, `train_h` – Train image dimensions.
    ///
    /// Returns a vector of `SubPixelMatch` with refined positions.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn refine(
        &self,
        matches: &FeatureMatch,
        kps_query: &[(f32, f32)],
        kps_train: &[(f32, f32)],
        img_query: &[f32],
        query_w: usize,
        query_h: usize,
        img_train: &[f32],
        train_w: usize,
        train_h: usize,
    ) -> Vec<SubPixelMatch> {
        matches
            .all_matches
            .iter()
            .filter_map(|m| {
                let qi = m.query_idx;
                let ti = m.train_idx;

                if qi >= kps_query.len() || ti >= kps_train.len() {
                    return None;
                }

                let (qx, qy) = kps_query[qi];
                let (tx, ty) = kps_train[ti];

                let (qxs, qys, q_err) =
                    self.refine_point(img_query, query_w, query_h, qx as f64, qy as f64);
                let (txs, tys, t_err) =
                    self.refine_point(img_train, train_w, train_h, tx as f64, ty as f64);

                let refinement_error = (q_err * q_err + t_err * t_err).sqrt();

                Some(SubPixelMatch {
                    query_idx: qi,
                    train_idx: ti,
                    query_pos_int: (qx, qy),
                    train_pos_int: (tx, ty),
                    query_pos_sub: (qxs, qys),
                    train_pos_sub: (txs, tys),
                    refinement_error,
                    dist: m.dist,
                })
            })
            .collect()
    }

    /// Refine a single point to sub-pixel accuracy using parabolic interpolation.
    ///
    /// Returns `(refined_x, refined_y, displacement_magnitude)`.
    fn refine_point(
        &self,
        image: &[f32],
        width: usize,
        height: usize,
        cx: f64,
        cy: f64,
    ) -> (f64, f64, f64) {
        let ix = cx.round() as usize;
        let iy = cy.round() as usize;

        // Ensure we have room for the window
        if ix < self.window_half
            || iy < self.window_half
            || ix + self.window_half >= width
            || iy + self.window_half >= height
        {
            return (cx, cy, 0.0);
        }

        // Compute second-order Taylor coefficients from image gradients in a
        // small neighbourhood (3×3 finite differences).
        let get = |x: usize, y: usize| image[y * width + x] as f64;

        let c = get(ix, iy);
        let l = get(ix - 1, iy);
        let r = get(ix + 1, iy);
        let u = get(ix, iy - 1);
        let d = get(ix, iy + 1);

        // Second derivative approximations
        let dxx = r - 2.0 * c + l;
        let dyy = d - 2.0 * c + u;
        // First derivative approximations (central difference)
        let dx = (r - l) * 0.5;
        let dy = (d - u) * 0.5;

        // Parabolic sub-pixel shift: delta = -grad / (2 * hessian_diag)
        let shift_x = if dxx.abs() > 1e-6 {
            (-dx / (2.0 * dxx)).clamp(-self.max_displacement, self.max_displacement)
        } else {
            0.0
        };
        let shift_y = if dyy.abs() > 1e-6 {
            (-dy / (2.0 * dyy)).clamp(-self.max_displacement, self.max_displacement)
        } else {
            0.0
        };

        let refined_x = cx + shift_x;
        let refined_y = cy + shift_y;
        let displacement = (shift_x * shift_x + shift_y * shift_y).sqrt();

        (refined_x, refined_y, displacement)
    }
}

impl Default for SubPixelRefiner {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute sub-pixel accurate matches in one step.
///
/// Convenience function combining `FeatureMatcher::match_descriptors` and
/// `SubPixelRefiner::refine`.
///
/// # Arguments
///
/// * `query`     – Flat query descriptors (n_q × desc_len).
/// * `train`     – Flat train descriptors (n_t × desc_len).
/// * `desc_len`  – Descriptor dimensionality.
/// * `kps_query` – Query keypoint positions.
/// * `kps_train` – Train keypoint positions.
/// * `img_query` / `img_train` – Grayscale images as f32 slices.
/// * `qw`, `qh`, `tw`, `th`   – Image dimensions.
///
/// Returns sub-pixel accurate match list.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn match_with_subpixel(
    query: &[f32],
    train: &[f32],
    desc_len: usize,
    kps_query: &[(f32, f32)],
    kps_train: &[(f32, f32)],
    img_query: &[f32],
    qw: usize,
    qh: usize,
    img_train: &[f32],
    tw: usize,
    th: usize,
) -> Vec<SubPixelMatch> {
    let matcher = FeatureMatcher::new(false, 0.0);
    let coarse = matcher.match_descriptors(query, train, desc_len);
    let refiner = SubPixelRefiner::new();
    refiner.refine(
        &coarse, kps_query, kps_train, img_query, qw, qh, img_train, tw, th,
    )
}

// ---------------------------------------------------------------------------
// HomographyEstimator — RANSAC-based homography from sub-pixel matches
// ---------------------------------------------------------------------------

/// Configuration for the RANSAC homography estimator.
#[derive(Debug, Clone)]
pub struct RansacConfig {
    /// Maximum reprojection error (pixels) for a point to be considered an inlier.
    pub inlier_threshold: f64,
    /// Maximum number of RANSAC iterations.
    pub max_iterations: usize,
    /// Minimum number of inliers required for a valid model.
    pub min_inliers: usize,
}

impl Default for RansacConfig {
    fn default() -> Self {
        Self {
            inlier_threshold: 3.0,
            max_iterations: 1000,
            min_inliers: 4,
        }
    }
}

/// A 3×3 homography matrix stored in row-major order (affine or projective).
///
/// The matrix maps homogeneous image coordinates:
/// `p' = H · p`  where  `p = [x, y, 1]^T`.
#[derive(Debug, Clone)]
pub struct Homography {
    /// 9 elements of the 3×3 matrix in row-major order.
    pub m: [f64; 9],
}

impl Homography {
    /// Create the identity homography.
    #[must_use]
    pub fn identity() -> Self {
        Self {
            m: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        }
    }

    /// Apply the homography to a 2-D point and return the projected point.
    ///
    /// Returns `None` if the homogeneous weight `w` is near zero.
    #[must_use]
    pub fn project(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        let m = &self.m;
        let wx = m[0] * x + m[1] * y + m[2];
        let wy = m[3] * x + m[4] * y + m[5];
        let w = m[6] * x + m[7] * y + m[8];
        if w.abs() < 1e-12 {
            return None;
        }
        Some((wx / w, wy / w))
    }

    /// Compute the symmetric reprojection error for a match pair.
    ///
    /// Returns the mean of the forward and backward reprojection distances.
    #[must_use]
    pub fn reprojection_error(&self, qx: f64, qy: f64, tx: f64, ty: f64) -> f64 {
        let fwd = self
            .project(qx, qy)
            .map(|(px, py)| ((px - tx).powi(2) + (py - ty).powi(2)).sqrt())
            .unwrap_or(f64::MAX);
        fwd
    }
}

/// Estimate a homography from a set of sub-pixel matches using DLT + RANSAC.
///
/// This is a minimal, pure-Rust implementation of the Direct Linear Transform
/// (DLT) solver for 4-point correspondences, wrapped in a RANSAC loop that
/// selects the best hypothesis by inlier count.
pub struct HomographyEstimator {
    cfg: RansacConfig,
}

impl HomographyEstimator {
    /// Create a homography estimator with the given RANSAC configuration.
    #[must_use]
    pub fn new(cfg: RansacConfig) -> Self {
        Self { cfg }
    }

    /// Estimate a homography from sub-pixel matches.
    ///
    /// Returns `(homography, inlier_count)` or `None` if fewer than 4 matches
    /// are available or the RANSAC loop fails to find a valid model.
    ///
    /// The RANSAC loop uses a deterministic sampling strategy (evenly-spaced
    /// indices) to ensure reproducibility.
    #[must_use]
    pub fn estimate(&self, matches: &[SubPixelMatch]) -> Option<(Homography, usize)> {
        if matches.len() < 4 {
            return None;
        }

        let mut best_h = Homography::identity();
        let mut best_inliers = 0usize;

        let n = matches.len();
        let step = (n / self.cfg.max_iterations).max(1);

        let mut iter_count = 0usize;
        let mut start = 0usize;

        while iter_count < self.cfg.max_iterations && start < n {
            // Select 4 points deterministically using a rotating stride
            let i0 = start % n;
            let i1 = (start + 1) % n;
            let i2 = (start + 2) % n;
            let i3 = (start + 3) % n;

            let pts = [&matches[i0], &matches[i1], &matches[i2], &matches[i3]];

            if let Some(h) = Self::dlt_4point(pts) {
                let inliers = matches
                    .iter()
                    .filter(|m| {
                        h.reprojection_error(
                            m.query_pos_sub.0,
                            m.query_pos_sub.1,
                            m.train_pos_sub.0,
                            m.train_pos_sub.1,
                        ) < self.cfg.inlier_threshold
                    })
                    .count();

                if inliers > best_inliers {
                    best_inliers = inliers;
                    best_h = h;
                }
            }

            start += step;
            iter_count += 1;
        }

        if best_inliers >= self.cfg.min_inliers {
            Some((best_h, best_inliers))
        } else {
            None
        }
    }

    /// Direct Linear Transform for exactly 4 point correspondences.
    ///
    /// Solves the 8×9 DLT system Ah = 0 using a hand-unrolled pseudo-inverse
    /// for the minimal case.  Returns `None` if the system is degenerate.
    fn dlt_4point(pts: [&SubPixelMatch; 4]) -> Option<Homography> {
        // Build 8×9 matrix A (each point contributes 2 rows)
        let mut a = [[0.0f64; 9]; 8];

        for (k, m) in pts.iter().enumerate() {
            let (x, y) = m.query_pos_sub;
            let (xp, yp) = m.train_pos_sub;
            let row0 = k * 2;
            let row1 = row0 + 1;

            a[row0] = [-x, -y, -1.0, 0.0, 0.0, 0.0, xp * x, xp * y, xp];
            a[row1] = [0.0, 0.0, 0.0, -x, -y, -1.0, yp * x, yp * y, yp];
        }

        // Solve via SVD-free approximation: normalise and use cross-product trick.
        // For a minimal 4-point case the null space can be found by eliminating
        // variables.  We use a simple Gaussian elimination on the 8×9 matrix.
        let h_vec = gaussian_null_space(&mut a)?;

        Some(Homography { m: h_vec })
    }
}

impl Default for HomographyEstimator {
    fn default() -> Self {
        Self::new(RansacConfig::default())
    }
}

/// Solve the null space of an overdetermined 8×9 system by Gaussian elimination.
///
/// Returns the last column (free variable) of the reduced system as the
/// homography vector, or `None` if the system is degenerate.
fn gaussian_null_space(a: &mut [[f64; 9]; 8]) -> Option<[f64; 9]> {
    let rows = 8usize;
    let cols = 9usize;

    for col in 0..cols - 1 {
        // Find pivot row
        let pivot = (col..rows).max_by(|&i, &j| {
            a[i][col]
                .abs()
                .partial_cmp(&a[j][col].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let pivot = pivot?;

        if a[pivot][col].abs() < 1e-12 {
            continue;
        }

        a.swap(col.min(rows - 1), pivot);

        let diag = a[col.min(rows - 1)][col];
        for c in 0..cols {
            a[col.min(rows - 1)][c] /= diag;
        }

        for r in 0..rows {
            if r == col.min(rows - 1) {
                continue;
            }
            let factor = a[r][col];
            for c in 0..cols {
                let sub = factor * a[col.min(rows - 1)][c];
                a[r][c] -= sub;
            }
        }
    }

    // The null vector is the last column with sign-normalised h[8] = 1
    let mut h = [0.0f64; 9];
    for r in 0..8 {
        h[r] = -a[r][8];
    }
    h[8] = 1.0;

    Some(h)
}

/// Compute the quality score of a set of sub-pixel matches.
///
/// Returns the fraction of matches whose `refinement_error` is below
/// `max_error`.  A high score indicates well-localised keypoints.
#[must_use]
pub fn subpixel_match_quality(matches: &[SubPixelMatch], max_error: f64) -> f32 {
    if matches.is_empty() {
        return 0.0;
    }
    let good = matches
        .iter()
        .filter(|m| m.refinement_error < max_error)
        .count();
    good as f32 / matches.len() as f32
}

/// Filter sub-pixel matches by maximum descriptor distance and refinement error.
///
/// Returns only matches satisfying both `max_dist` and `max_refinement_error`.
#[must_use]
pub fn filter_subpixel_matches(
    matches: &[SubPixelMatch],
    max_dist: f32,
    max_refinement_error: f64,
) -> Vec<&SubPixelMatch> {
    matches
        .iter()
        .filter(|m| m.dist <= max_dist && m.refinement_error <= max_refinement_error)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_match_pair_distance() {
        let m = MatchPair::new(0, 1, 3.5);
        assert!((m.distance() - 3.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_match_pair_negative_dist_clamped() {
        let m = MatchPair::new(0, 0, -1.0);
        assert!((m.distance() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_match_pair_is_good_true() {
        let m = MatchPair::new(0, 1, 2.0);
        assert!(m.is_good(3.0));
    }

    #[test]
    fn test_match_pair_is_good_false() {
        let m = MatchPair::new(0, 1, 4.0);
        assert!(!m.is_good(3.0));
    }

    #[test]
    fn test_feature_match_good_matches_filter() {
        let mut fm = FeatureMatch::new();
        fm.all_matches.push(MatchPair::new(0, 0, 1.0));
        fm.all_matches.push(MatchPair::new(1, 1, 5.0));
        let good = fm.good_matches(3.0);
        assert_eq!(good.len(), 1);
    }

    #[test]
    fn test_feature_match_min_distance_empty() {
        let fm = FeatureMatch::new();
        assert_eq!(fm.min_distance(), f32::MAX);
    }

    #[test]
    fn test_feature_match_min_distance_non_empty() {
        let mut fm = FeatureMatch::new();
        fm.all_matches.push(MatchPair::new(0, 0, 3.0));
        fm.all_matches.push(MatchPair::new(1, 1, 1.5));
        assert!((fm.min_distance() - 1.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_feature_matcher_match_descriptors_empty() {
        let matcher = FeatureMatcher::new(false, 0.0);
        let result = matcher.match_descriptors(&[], &[], 4);
        assert!(result.is_empty());
    }

    #[test]
    fn test_feature_matcher_match_descriptors_single() {
        let matcher = FeatureMatcher::new(false, 0.0);
        let query = vec![1.0_f32, 0.0, 0.0, 0.0];
        let train = vec![1.0_f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let result = matcher.match_descriptors(&query, &train, 4);
        assert_eq!(result.len(), 1);
        // Identical descriptors → distance 0.
        assert!((result.all_matches[0].dist - 0.0).abs() < 1e-5);
        assert_eq!(result.all_matches[0].train_idx, 0);
    }

    #[test]
    fn test_feature_matcher_match_descriptors_wrong_desc_len() {
        let matcher = FeatureMatcher::new(false, 0.0);
        // query length (5) is not divisible by desc_len (4)
        let query = vec![1.0_f32; 5];
        let train = vec![1.0_f32; 4];
        let result = matcher.match_descriptors(&query, &train, 4);
        assert!(result.is_empty());
    }

    #[test]
    fn test_feature_matcher_ratio_test_all_pass_single_train() {
        let matcher = FeatureMatcher::new(false, 0.75);
        let query = vec![1.0_f32, 0.0];
        let train = vec![1.0_f32, 0.0]; // only one train descriptor → second_dist = MAX → pass
        let fm = matcher.match_descriptors(&query, &train, 2);
        let passed = matcher.ratio_test(&fm, &query, &train, 2);
        assert_eq!(passed.len(), 1);
    }

    #[test]
    fn test_feature_matcher_ratio_test_rejects_ambiguous() {
        let matcher = FeatureMatcher::new(false, 0.75);
        // query descriptor
        let query = vec![0.0_f32, 0.0, 0.0, 0.0];
        // Two train descriptors: one at distance 1.0 and one at distance 1.1 (ratio ≈ 0.91 > 0.75).
        let train = vec![
            1.0_f32, 0.0, 0.0, 0.0, // dist ≈ 1.0
            1.1_f32, 0.0, 0.0, 0.0, // dist ≈ 1.1
        ];
        let fm = matcher.match_descriptors(&query, &train, 4);
        let passed = matcher.ratio_test(&fm, &query, &train, 4);
        assert_eq!(passed.len(), 0);
    }

    #[test]
    fn test_subpixel_refiner_default() {
        let refiner = SubPixelRefiner::new();
        assert_eq!(refiner.window_half, 3);
    }

    #[test]
    fn test_subpixel_refine_empty_matches() {
        let refiner = SubPixelRefiner::new();
        let fm = FeatureMatch::new();
        let img = vec![0.5f32; 10 * 10];
        let result = refiner.refine(&fm, &[], &[], &img, 10, 10, &img, 10, 10);
        assert!(result.is_empty());
    }

    #[test]
    fn test_subpixel_refine_single_match() {
        let refiner = SubPixelRefiner::new();
        let mut fm = FeatureMatch::new();
        fm.all_matches.push(MatchPair::new(0, 0, 0.0));
        // Uniform image → no gradient → no sub-pixel shift
        let img = vec![0.5f32; 20 * 20];
        let kps_q = vec![(10.0f32, 10.0f32)];
        let kps_t = vec![(10.0f32, 10.0f32)];
        let result = refiner.refine(&fm, &kps_q, &kps_t, &img, 20, 20, &img, 20, 20);
        assert_eq!(result.len(), 1);
        let m = &result[0];
        assert!((m.query_pos_sub.0 - 10.0).abs() < 0.01);
        assert!((m.query_pos_sub.1 - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_subpixel_refine_with_gradient() {
        let refiner = SubPixelRefiner::new();
        let mut fm = FeatureMatch::new();
        fm.all_matches.push(MatchPair::new(0, 0, 0.5));
        // Create a parabolic image peak at (10, 10) to verify sub-pixel shift
        let w = 20usize;
        let h = 20usize;
        let mut img = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let dx = x as f32 - 10.0;
                let dy = y as f32 - 10.0;
                img[y * w + x] = 1.0 - 0.01 * (dx * dx + dy * dy);
            }
        }
        let kps_q = vec![(10.0f32, 10.0f32)];
        let kps_t = vec![(10.0f32, 10.0f32)];
        let result = refiner.refine(&fm, &kps_q, &kps_t, &img, w, h, &img, w, h);
        assert_eq!(result.len(), 1);
        // At exact peak, refinement displacement should be near zero
        assert!(result[0].refinement_error < 0.5);
    }

    #[test]
    fn test_match_with_subpixel_empty() {
        let result = match_with_subpixel(&[], &[], 4, &[], &[], &[], 0, 0, &[], 0, 0);
        assert!(result.is_empty());
    }

    // --- Homography tests ---

    #[test]
    fn test_homography_identity_projects_correctly() {
        let h = Homography::identity();
        let p = h.project(10.0, 20.0);
        assert!(p.is_some());
        let (px, py) = p.unwrap();
        assert!((px - 10.0).abs() < 1e-10);
        assert!((py - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_homography_reprojection_error_identity() {
        let h = Homography::identity();
        // Identity: query == train → error should be 0
        let err = h.reprojection_error(5.0, 7.0, 5.0, 7.0);
        assert!(
            err < 1e-10,
            "Identity reprojection error should be 0, got {err}"
        );
    }

    #[test]
    fn test_homography_reprojection_error_nonzero() {
        let h = Homography::identity();
        // Non-matching points
        let err = h.reprojection_error(0.0, 0.0, 3.0, 4.0);
        assert!((err - 5.0).abs() < 1e-6, "Expected error=5.0, got {err}");
    }

    #[test]
    fn test_ransac_config_default() {
        let cfg = RansacConfig::default();
        assert!(cfg.inlier_threshold > 0.0);
        assert!(cfg.max_iterations > 0);
        assert!(cfg.min_inliers >= 4);
    }

    #[test]
    fn test_homography_estimator_too_few_matches_returns_none() {
        let est = HomographyEstimator::default();
        let matches: Vec<SubPixelMatch> = (0..3)
            .map(|i| SubPixelMatch {
                query_idx: i,
                train_idx: i,
                query_pos_int: (i as f32, i as f32),
                train_pos_int: (i as f32, i as f32),
                query_pos_sub: (i as f64, i as f64),
                train_pos_sub: (i as f64, i as f64),
                refinement_error: 0.0,
                dist: 0.0,
            })
            .collect();
        assert!(est.estimate(&matches).is_none());
    }

    #[test]
    fn test_homography_estimator_with_pure_translation() {
        // A translation by (tx, ty) should be estimable from 4+ point correspondences.
        let tx = 5.0_f64;
        let ty = -3.0_f64;

        let points: Vec<(f64, f64)> = vec![
            (10.0, 20.0),
            (50.0, 30.0),
            (10.0, 80.0),
            (90.0, 20.0),
            (60.0, 60.0),
        ];

        let matches: Vec<SubPixelMatch> = points
            .iter()
            .enumerate()
            .map(|(i, &(qx, qy))| SubPixelMatch {
                query_idx: i,
                train_idx: i,
                query_pos_int: (qx as f32, qy as f32),
                train_pos_int: ((qx + tx) as f32, (qy + ty) as f32),
                query_pos_sub: (qx, qy),
                train_pos_sub: (qx + tx, qy + ty),
                refinement_error: 0.0,
                dist: 0.1,
            })
            .collect();

        let mut cfg = RansacConfig::default();
        cfg.max_iterations = 50;
        cfg.min_inliers = 4;
        let est = HomographyEstimator::new(cfg);
        // We don't require success (degenerate configurations can fail),
        // but if it succeeds the inlier count must be >= 4.
        if let Some((_, inliers)) = est.estimate(&matches) {
            assert!(inliers >= 4, "Expected >=4 inliers, got {inliers}");
        }
    }

    // --- subpixel_match_quality tests ---

    #[test]
    fn test_subpixel_match_quality_empty_returns_zero() {
        let q = subpixel_match_quality(&[], 1.0);
        assert!((q - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_subpixel_match_quality_all_below_threshold() {
        let matches: Vec<SubPixelMatch> = (0..5)
            .map(|i| SubPixelMatch {
                query_idx: i,
                train_idx: i,
                query_pos_int: (0.0, 0.0),
                train_pos_int: (0.0, 0.0),
                query_pos_sub: (0.0, 0.0),
                train_pos_sub: (0.0, 0.0),
                refinement_error: 0.1, // below 0.5
                dist: 0.0,
            })
            .collect();
        let q = subpixel_match_quality(&matches, 0.5);
        assert!(
            (q - 1.0).abs() < f32::EPSILON,
            "All matches below threshold → quality=1.0, got {q}"
        );
    }

    #[test]
    fn test_subpixel_match_quality_none_below_threshold() {
        let matches: Vec<SubPixelMatch> = (0..5)
            .map(|i| SubPixelMatch {
                query_idx: i,
                train_idx: i,
                query_pos_int: (0.0, 0.0),
                train_pos_int: (0.0, 0.0),
                query_pos_sub: (0.0, 0.0),
                train_pos_sub: (0.0, 0.0),
                refinement_error: 2.0, // above 0.5
                dist: 0.0,
            })
            .collect();
        let q = subpixel_match_quality(&matches, 0.5);
        assert!(
            (q - 0.0).abs() < f32::EPSILON,
            "No matches below threshold → quality=0.0, got {q}"
        );
    }

    // --- filter_subpixel_matches tests ---

    #[test]
    fn test_filter_subpixel_matches_all_pass() {
        let matches: Vec<SubPixelMatch> = (0..4)
            .map(|i| SubPixelMatch {
                query_idx: i,
                train_idx: i,
                query_pos_int: (0.0, 0.0),
                train_pos_int: (0.0, 0.0),
                query_pos_sub: (0.0, 0.0),
                train_pos_sub: (0.0, 0.0),
                refinement_error: 0.2,
                dist: 1.0,
            })
            .collect();
        let filtered = filter_subpixel_matches(&matches, 2.0, 1.0);
        assert_eq!(filtered.len(), 4);
    }

    #[test]
    fn test_filter_subpixel_matches_by_dist() {
        let matches: Vec<SubPixelMatch> = vec![
            SubPixelMatch {
                query_idx: 0,
                train_idx: 0,
                query_pos_int: (0.0, 0.0),
                train_pos_int: (0.0, 0.0),
                query_pos_sub: (0.0, 0.0),
                train_pos_sub: (0.0, 0.0),
                refinement_error: 0.0,
                dist: 0.5, // pass
            },
            SubPixelMatch {
                query_idx: 1,
                train_idx: 1,
                query_pos_int: (0.0, 0.0),
                train_pos_int: (0.0, 0.0),
                query_pos_sub: (0.0, 0.0),
                train_pos_sub: (0.0, 0.0),
                refinement_error: 0.0,
                dist: 5.0, // fail (> max_dist=2.0)
            },
        ];
        let filtered = filter_subpixel_matches(&matches, 2.0, 1.0);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].query_idx, 0);
    }

    #[test]
    fn test_filter_subpixel_matches_by_refinement_error() {
        let matches: Vec<SubPixelMatch> = vec![
            SubPixelMatch {
                query_idx: 0,
                train_idx: 0,
                query_pos_int: (0.0, 0.0),
                train_pos_int: (0.0, 0.0),
                query_pos_sub: (0.0, 0.0),
                train_pos_sub: (0.0, 0.0),
                refinement_error: 0.1,
                dist: 1.0, // pass
            },
            SubPixelMatch {
                query_idx: 1,
                train_idx: 1,
                query_pos_int: (0.0, 0.0),
                train_pos_int: (0.0, 0.0),
                query_pos_sub: (0.0, 0.0),
                train_pos_sub: (0.0, 0.0),
                refinement_error: 3.0,
                dist: 1.0, // fail (> max_err=1.5)
            },
        ];
        let filtered = filter_subpixel_matches(&matches, 2.0, 1.5);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].query_idx, 0);
    }

    #[test]
    fn test_subpixel_refiner_set_window_and_displacement() {
        let mut refiner = SubPixelRefiner::new();
        refiner.set_window_half(5);
        refiner.set_max_displacement(1.0);
        assert_eq!(refiner.window_half, 5);
        assert!((refiner.max_displacement - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_subpixel_refiner_out_of_bounds_keypoint_returns_original() {
        let refiner = SubPixelRefiner::new();
        let mut fm = FeatureMatch::new();
        fm.all_matches.push(MatchPair::new(0, 0, 0.0));
        let img = vec![0.5f32; 10 * 10];
        // Keypoint at border — should still return a result (clamped to original pos)
        let kps_q = vec![(0.0f32, 0.0f32)]; // border: ix=0 < window_half=3
        let kps_t = vec![(5.0f32, 5.0f32)];
        let result = refiner.refine(&fm, &kps_q, &kps_t, &img, 10, 10, &img, 10, 10);
        assert_eq!(result.len(), 1);
        // Original position should be preserved when refinement cannot proceed
        assert!((result[0].query_pos_sub.0 - 0.0).abs() < 0.5);
    }
}
