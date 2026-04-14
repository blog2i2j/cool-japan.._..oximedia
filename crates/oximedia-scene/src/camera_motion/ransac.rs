//! RANSAC-based homography estimation using the Direct Linear Transform (DLT).
//!
//! A 2D homography (projective transformation) maps points from one plane to
//! another.  It is represented by a 3 × 3 matrix **H** acting on homogeneous
//! coordinates:
//!
//! ```text
//! [x2]   [h00 h01 h02] [x1]
//! [y2] ~ [h10 h11 h12] [y1]
//! [ 1]   [h20 h21  1 ] [ 1]
//! ```
//!
//! # RANSAC algorithm
//!
//! 1. Repeat `iterations` times:
//!    - Sample 4 correspondences using a deterministic LCG.
//!    - Estimate **H** via the DLT (8-DOF, normalised).
//!    - Count *inliers*: correspondences whose reprojection error < `threshold`.
//! 2. Return the **H** that had the most inliers.
//!
//! # References
//!
//! * Hartley & Zisserman, *Multiple View Geometry in Computer Vision* (2nd ed.),
//!   §4.1 (DLT) and §4.8 (RANSAC).

/// A 2D projective homography represented as a 3 × 3 row-major matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct Homography(pub [[f64; 3]; 3]);

impl Homography {
    /// Identity homography.
    #[must_use]
    pub fn identity() -> Self {
        Self([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    }

    /// Access the underlying 3 × 3 matrix.
    #[must_use]
    pub fn matrix(&self) -> &[[f64; 3]; 3] {
        &self.0
    }
}

/// Project a point `(x, y)` through homography `h`.
///
/// Returns `(x', y')` in Euclidean coordinates (i.e. after dividing by the
/// homogeneous scale `w`).  When `w` is zero or very small (degenerate
/// homography) the original coordinates are returned unchanged.
#[must_use]
pub fn apply_homography(h: &Homography, x: f32, y: f32) -> (f32, f32) {
    let m = &h.0;
    let xd = x as f64;
    let yd = y as f64;

    let wx = m[0][0] * xd + m[0][1] * yd + m[0][2];
    let wy = m[1][0] * xd + m[1][1] * yd + m[1][2];
    let ww = m[2][0] * xd + m[2][1] * yd + m[2][2];

    if ww.abs() < 1e-10 {
        return (x, y);
    }

    ((wx / ww) as f32, (wy / ww) as f32)
}

/// Estimate a homography from point correspondences using RANSAC.
///
/// # Arguments
///
/// * `matches` – slice of `(x1, y1, x2, y2)` correspondences.
/// * `iterations` – number of RANSAC iterations.
/// * `threshold` – maximum reprojection error (in pixels) for a match to be
///   considered an inlier.
///
/// # Returns
///
/// `Some(Homography)` with the model that had the most inliers, or `None`
/// when fewer than 4 matches are provided (minimum required by DLT) or when
/// no valid homography could be computed.
#[must_use]
pub fn ransac_homography(
    matches: &[(f32, f32, f32, f32)],
    iterations: u32,
    threshold: f32,
) -> Option<Homography> {
    if matches.len() < 4 {
        return None;
    }

    let n = matches.len();
    let thresh_sq = (threshold as f64) * (threshold as f64);

    let mut best_h: Option<Homography> = None;
    let mut best_inliers = 0usize;

    // Deterministic LCG seeded from the first match's coordinates.
    let seed = {
        let (x1, y1, x2, y2) = matches[0];
        let bits = (x1.to_bits() as u64)
            ^ ((y1.to_bits() as u64) << 7)
            ^ ((x2.to_bits() as u64) << 13)
            ^ ((y2.to_bits() as u64) << 19);
        bits.wrapping_add(1)
    };
    let mut lcg = seed;

    for _ in 0..iterations {
        // Sample 4 distinct indices
        let mut indices = [0usize; 4];
        let mut chosen = 0usize;
        let mut attempts = 0usize;
        while chosen < 4 && attempts < 64 {
            lcg = lcg
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let idx = (lcg >> 33) as usize % n;
            if indices[..chosen].iter().all(|&prev| prev != idx) {
                indices[chosen] = idx;
                chosen += 1;
            }
            attempts += 1;
        }
        if chosen < 4 {
            continue;
        }

        let sample: Vec<(f32, f32, f32, f32)> = indices.iter().map(|&i| matches[i]).collect();

        let h = match dlt_homography(&sample) {
            Some(h) => h,
            None => continue,
        };

        let inliers = count_inliers(matches, &h, thresh_sq);
        if inliers > best_inliers {
            best_inliers = inliers;
            best_h = Some(h);
        }
    }

    // Optionally refine on all inliers
    if let Some(ref h) = best_h {
        let inlier_matches: Vec<(f32, f32, f32, f32)> = matches
            .iter()
            .filter(|&&(x1, y1, x2, y2)| reprojection_error_sq(h, x1, y1, x2, y2) < thresh_sq)
            .copied()
            .collect();

        if inlier_matches.len() >= 4 {
            if let Some(refined) = dlt_homography(&inlier_matches) {
                return Some(refined);
            }
        }
    }

    best_h
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Count inliers: correspondences whose reprojection error² < `thresh_sq`.
fn count_inliers(matches: &[(f32, f32, f32, f32)], h: &Homography, thresh_sq: f64) -> usize {
    matches
        .iter()
        .filter(|&&(x1, y1, x2, y2)| reprojection_error_sq(h, x1, y1, x2, y2) < thresh_sq)
        .count()
}

/// Squared reprojection error: |H * p1 − p2|².
fn reprojection_error_sq(h: &Homography, x1: f32, y1: f32, x2: f32, y2: f32) -> f64 {
    let (px, py) = apply_homography(h, x1, y1);
    let dx = px as f64 - x2 as f64;
    let dy = py as f64 - y2 as f64;
    dx * dx + dy * dy
}

/// Direct Linear Transform (DLT) for homography estimation.
///
/// Given ≥ 4 point correspondences, builds the 2n × 9 system matrix **A**
/// and solves **A h = 0** via the minimal (rank-8) SVD-like decomposition
/// (Jacobi SVD implemented below).
///
/// Returns `None` when the system is degenerate.
fn dlt_homography(matches: &[(f32, f32, f32, f32)]) -> Option<Homography> {
    if matches.len() < 4 {
        return None;
    }

    // Build 2n × 9 matrix A
    let n = matches.len();
    let rows = 2 * n;
    let mut a = vec![0.0_f64; rows * 9];

    for (i, &(x1, y1, x2, y2)) in matches.iter().enumerate() {
        let x1 = x1 as f64;
        let y1 = y1 as f64;
        let x2 = x2 as f64;
        let y2 = y2 as f64;

        // Row 2i:   [-x1, -y1, -1, 0, 0, 0, x2*x1, x2*y1, x2]
        let r1 = i * 2;
        a[r1 * 9 + 0] = -x1;
        a[r1 * 9 + 1] = -y1;
        a[r1 * 9 + 2] = -1.0;
        a[r1 * 9 + 3] = 0.0;
        a[r1 * 9 + 4] = 0.0;
        a[r1 * 9 + 5] = 0.0;
        a[r1 * 9 + 6] = x2 * x1;
        a[r1 * 9 + 7] = x2 * y1;
        a[r1 * 9 + 8] = x2;

        // Row 2i+1: [0, 0, 0, -x1, -y1, -1, y2*x1, y2*y1, y2]
        let r2 = r1 + 1;
        a[r2 * 9 + 0] = 0.0;
        a[r2 * 9 + 1] = 0.0;
        a[r2 * 9 + 2] = 0.0;
        a[r2 * 9 + 3] = -x1;
        a[r2 * 9 + 4] = -y1;
        a[r2 * 9 + 5] = -1.0;
        a[r2 * 9 + 6] = y2 * x1;
        a[r2 * 9 + 7] = y2 * y1;
        a[r2 * 9 + 8] = y2;
    }

    // Solve via AtA eigenvector (null-space of A via smallest eigenvalue of AtA)
    let h_vec = smallest_ata_eigenvec(&a, rows, 9)?;

    // Reshape h_vec (9 elements) into 3×3
    let mat = [
        [h_vec[0], h_vec[1], h_vec[2]],
        [h_vec[3], h_vec[4], h_vec[5]],
        [h_vec[6], h_vec[7], h_vec[8]],
    ];

    // Normalise so that H[2][2] = 1 (if possible)
    let scale = mat[2][2];
    if scale.abs() < 1e-10 {
        return Some(Homography(mat));
    }
    let inv = 1.0 / scale;
    let normalised = [
        [mat[0][0] * inv, mat[0][1] * inv, mat[0][2] * inv],
        [mat[1][0] * inv, mat[1][1] * inv, mat[1][2] * inv],
        [mat[2][0] * inv, mat[2][1] * inv, 1.0],
    ];

    Some(Homography(normalised))
}

/// Compute AtA (9×9 symmetric matrix) and return the eigenvector corresponding
/// to the smallest eigenvalue via the power-iteration variant (inverse power
/// method with deflation).
///
/// For a 9×9 system this is fast and accurate enough for RANSAC use.
fn smallest_ata_eigenvec(a: &[f64], rows: usize, cols: usize) -> Option<Vec<f64>> {
    // Build AtA (cols × cols = 9 × 9)
    let mut ata = vec![0.0_f64; cols * cols];
    for i in 0..rows {
        for j in 0..cols {
            for k in 0..cols {
                ata[j * cols + k] += a[i * cols + j] * a[i * cols + k];
            }
        }
    }

    // Use Jacobi eigenvalue algorithm to find all eigenvalues/vectors of the
    // symmetric 9×9 matrix.
    jacobi_eigenvec_smallest(&ata, cols)
}

/// Jacobi cyclic-sweep eigenvalue algorithm for a symmetric matrix.
///
/// Diagonalises the n×n symmetric matrix `mat` iteratively and extracts the
/// eigenvector corresponding to the **smallest** eigenvalue (needed by DLT to
/// find the null vector of A^T A).
///
/// Returns `None` only when the resulting eigenvector has zero norm (fully
/// degenerate input).
fn jacobi_eigenvec_smallest(mat: &[f64], n: usize) -> Option<Vec<f64>> {
    const MAX_SWEEPS: usize = 100;
    const EPS: f64 = 1e-14;

    // Working copy of the matrix
    let mut a = mat.to_vec();
    // V accumulates eigenvectors as *columns*; start from identity
    let mut v = vec![0.0_f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    for _ in 0..MAX_SWEEPS {
        // Check convergence: sum of squared off-diagonal elements
        let off: f64 = (0..n)
            .flat_map(|i| (i + 1..n).map(move |j| (i, j)))
            .map(|(i, j)| a[i * n + j] * a[i * n + j])
            .sum();
        if off < EPS {
            break;
        }

        // Cyclic sweep over all upper-triangular pairs
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if apq.abs() < 1e-15 {
                    continue;
                }

                let app = a[p * n + p];
                let aqq = a[q * n + q];

                // Wilkinson's formula for t = tan(theta)
                let tau = (aqq - app) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    1.0 / (tau - (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                // Update diagonal
                a[p * n + p] = app - t * apq;
                a[q * n + q] = aqq + t * apq;
                a[p * n + q] = 0.0;
                a[q * n + p] = 0.0;

                // Update remaining rows / columns r ≠ p, q
                for r in 0..n {
                    if r == p || r == q {
                        continue;
                    }
                    let arp = a[r * n + p];
                    let arq = a[r * n + q];
                    let new_arp = c * arp - s * arq;
                    let new_arq = s * arp + c * arq;
                    a[r * n + p] = new_arp;
                    a[p * n + r] = new_arp;
                    a[r * n + q] = new_arq;
                    a[q * n + r] = new_arq;
                }

                // Accumulate rotation: columns p and q of V
                for r in 0..n {
                    let vrp = v[r * n + p];
                    let vrq = v[r * n + q];
                    v[r * n + p] = c * vrp - s * vrq;
                    v[r * n + q] = s * vrp + c * vrq;
                }
            }
        }
    }

    // Find column index of smallest diagonal element (smallest eigenvalue)
    let min_idx = (0..n).min_by(|&i, &j| {
        a[i * n + i]
            .partial_cmp(&a[j * n + j])
            .unwrap_or(std::cmp::Ordering::Equal)
    })?;

    // Extract column `min_idx` of V (the corresponding eigenvector)
    let evec: Vec<f64> = (0..n).map(|i| v[i * n + min_idx]).collect();

    // Normalise to unit length
    let norm: f64 = evec.iter().map(|&x| x * x).sum::<f64>().sqrt();
    if norm < 1e-10 {
        return None;
    }
    Some(evec.iter().map(|&x| x / norm).collect())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: create a set of point correspondences that exactly obey a known
    // homography (a pure translation by (tx, ty)).
    fn translation_matches(tx: f32, ty: f32, count: usize) -> Vec<(f32, f32, f32, f32)> {
        (0..count)
            .map(|i| {
                let x = (i as f32) * 10.0;
                let y = (i as f32) * 5.0;
                (x, y, x + tx, y + ty)
            })
            .collect()
    }

    #[test]
    fn test_apply_homography_identity() {
        let h = Homography::identity();
        let (x2, y2) = apply_homography(&h, 3.0, 7.0);
        assert!((x2 - 3.0).abs() < 1e-4, "x2={x2}");
        assert!((y2 - 7.0).abs() < 1e-4, "y2={y2}");
    }

    #[test]
    fn test_apply_homography_translation() {
        // A homography encoding a pure translation (5, 3)
        let mut m = [[0.0_f64; 3]; 3];
        m[0][0] = 1.0;
        m[0][2] = 5.0;
        m[1][1] = 1.0;
        m[1][2] = 3.0;
        m[2][2] = 1.0;
        let h = Homography(m);
        let (x2, y2) = apply_homography(&h, 10.0, 20.0);
        assert!((x2 - 15.0).abs() < 1e-3, "x2={x2}");
        assert!((y2 - 23.0).abs() < 1e-3, "y2={y2}");
    }

    #[test]
    fn test_ransac_homography_too_few_matches() {
        let result = ransac_homography(&[(0.0, 0.0, 1.0, 1.0)], 10, 1.0);
        assert!(result.is_none(), "fewer than 4 matches should return None");
    }

    #[test]
    fn test_ransac_homography_translation() {
        // Pure translation: (x1, y1) → (x1 + 10, y1 + 5)
        let matches = translation_matches(10.0, 5.0, 8);
        let result = ransac_homography(&matches, 50, 2.0);
        assert!(result.is_some(), "should find a homography");
        let h = result.expect("homography expected");
        // Verify that reprojection error is small for all matches
        for &(x1, y1, x2, y2) in &matches {
            let (px, py) = apply_homography(&h, x1, y1);
            assert!((px - x2).abs() < 5.0, "x error: |{px} - {x2}| >= 5.0");
            assert!((py - y2).abs() < 5.0, "y error: |{py} - {y2}| >= 5.0");
        }
    }

    #[test]
    fn test_ransac_homography_with_outliers() {
        // 6 inliers with translation (3, 2) + 2 large outliers
        let mut matches = translation_matches(3.0, 2.0, 6);
        matches.push((0.0, 0.0, 200.0, 300.0)); // outlier
        matches.push((5.0, 5.0, -100.0, -100.0)); // outlier

        let result = ransac_homography(&matches, 100, 3.0);
        assert!(
            result.is_some(),
            "should find a homography despite outliers"
        );
        let h = result.expect("homography expected");
        // Inliers should reproject well
        let inlier_count = matches
            .iter()
            .filter(|&&(x1, y1, x2, y2)| {
                let (px, py) = apply_homography(&h, x1, y1);
                ((px - x2).powi(2) + (py - y2).powi(2)).sqrt() < 5.0
            })
            .count();
        assert!(
            inlier_count >= 4,
            "at least 4 inliers expected, got {inlier_count}"
        );
    }

    #[test]
    fn test_ransac_homography_returns_some_for_minimal_4() {
        let matches = translation_matches(1.0, 1.0, 4);
        let result = ransac_homography(&matches, 20, 1.0);
        // With exactly 4 perfect correspondences we should get a result
        assert!(
            result.is_some(),
            "4 perfect correspondences should yield Some"
        );
    }

    #[test]
    fn test_homography_identity_struct() {
        let h = Homography::identity();
        assert!((h.0[0][0] - 1.0).abs() < 1e-10);
        assert!((h.0[1][1] - 1.0).abs() < 1e-10);
        assert!((h.0[2][2] - 1.0).abs() < 1e-10);
        assert!(h.0[0][1].abs() < 1e-10);
    }
}
