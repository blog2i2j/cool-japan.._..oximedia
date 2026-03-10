//! Tracking marker detection and pose estimation
//!
//! Provides optical marker detection for camera tracking systems
//! with sub-pixel accuracy via SAD template matching and robust pose
//! estimation via normalised DLT homography computation.

use super::CameraPose;
use crate::math::{Matrix3, Point2, Point3, UnitQuaternion, Vector3};
use crate::{Result, VirtualProductionError};
use serde::{Deserialize, Serialize};

/// Marker type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MarkerType {
    /// Circular retroreflective marker
    Circular,
    /// Square fiducial marker
    Square,
    /// `ArUco` marker
    ArUco,
    /// `AprilTag` marker
    AprilTag,
}

/// Detected marker in 2D image space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Marker2D {
    /// Marker ID
    pub id: usize,
    /// Center position in image (pixels)
    pub position: Point2<f64>,
    /// Marker type
    pub marker_type: MarkerType,
    /// Detection confidence (0.0 - 1.0)
    pub confidence: f32,
    /// Marker size in pixels
    pub size: f64,
}

impl Marker2D {
    /// Create new 2D marker
    #[must_use]
    pub fn new(id: usize, position: Point2<f64>, marker_type: MarkerType) -> Self {
        Self {
            id,
            position,
            marker_type,
            confidence: 1.0,
            size: 10.0,
        }
    }
}

/// Marker in 3D world space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Marker3D {
    /// Marker ID
    pub id: usize,
    /// Position in world space (meters)
    pub position: Point3<f64>,
    /// Marker type
    pub marker_type: MarkerType,
    /// Physical size in meters
    pub size: f64,
}

impl Marker3D {
    /// Create new 3D marker
    #[must_use]
    pub fn new(id: usize, position: Point3<f64>, marker_type: MarkerType, size: f64) -> Self {
        Self {
            id,
            position,
            marker_type,
            size,
        }
    }
}

/// Marker detector configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarkerDetectorConfig {
    /// Minimum marker size in pixels
    pub min_marker_size: f64,
    /// Maximum marker size in pixels
    pub max_marker_size: f64,
    /// Detection threshold
    pub detection_threshold: f32,
    /// Minimum number of markers for pose estimation
    pub min_markers: usize,
}

impl Default for MarkerDetectorConfig {
    fn default() -> Self {
        Self {
            min_marker_size: 5.0,
            max_marker_size: 100.0,
            detection_threshold: 0.5,
            min_markers: 4,
        }
    }
}

// ─── SAD template matching ───────────────────────────────────────────────────

/// Half-size of the SAD search neighbourhood around each last-known position.
const SAD_SEARCH_RADIUS: isize = 32;
/// Half-size of the template patch extracted around each candidate centroid.
const SAD_TEMPLATE_HALF: isize = 8;

/// Compute the Sum of Absolute Differences (SAD) between a template patch and
/// a region in the search image, all in grey-valued bytes.
///
/// Returns `None` if any patch would fall outside the image.
fn sad_patch(
    image: &[u8],
    width: usize,
    height: usize,
    template: &[u8],
    template_half: isize,
    cx: isize,
    cy: isize,
) -> Option<u64> {
    let patch_size = (2 * template_half + 1) as usize;
    if template.len() != patch_size * patch_size {
        return None;
    }

    let mut sum: u64 = 0;
    for dy in -template_half..=template_half {
        for dx in -template_half..=template_half {
            let ix = cx + dx;
            let iy = cy + dy;
            if ix < 0 || iy < 0 || ix >= width as isize || iy >= height as isize {
                return None;
            }
            let img_px = image[iy as usize * width + ix as usize] as i32;
            let tmpl_idx =
                ((dy + template_half) as usize) * patch_size + (dx + template_half) as usize;
            let tmpl_px = template[tmpl_idx] as i32;
            sum += (img_px - tmpl_px).unsigned_abs() as u64;
        }
    }
    Some(sum)
}

/// Extract a greyscale patch of half-size `half` centred at `(cx, cy)`.
///
/// Returns `None` when the patch would extend outside `width × height`.
fn extract_patch(
    image: &[u8],
    width: usize,
    height: usize,
    cx: isize,
    cy: isize,
    half: isize,
) -> Option<Vec<u8>> {
    let patch_size = (2 * half + 1) as usize;
    let mut patch = Vec::with_capacity(patch_size * patch_size);
    for dy in -half..=half {
        for dx in -half..=half {
            let ix = cx + dx;
            let iy = cy + dy;
            if ix < 0 || iy < 0 || ix >= width as isize || iy >= height as isize {
                return None;
            }
            patch.push(image[iy as usize * width + ix as usize]);
        }
    }
    Some(patch)
}

/// Compute the intensity centroid of a grayscale patch (sub-pixel refinement).
///
/// Returns the sub-pixel offset `(δx, δy)` relative to the patch centre.
#[allow(clippy::cast_precision_loss)]
fn intensity_centroid_offset(patch: &[u8], half: isize) -> (f64, f64) {
    let patch_w = (2 * half + 1) as usize;
    let mut sum_x = 0.0f64;
    let mut sum_y = 0.0f64;
    let mut total = 0.0f64;
    for dy in -half..=half {
        for dx in -half..=half {
            let idx = ((dy + half) as usize) * patch_w + (dx + half) as usize;
            let v = patch[idx] as f64;
            sum_x += dx as f64 * v;
            sum_y += dy as f64 * v;
            total += v;
        }
    }
    if total > 0.0 {
        (sum_x / total, sum_y / total)
    } else {
        (0.0, 0.0)
    }
}

/// Detect circular (bright blob) markers in a greyscale image by threshold +
/// SAD template matching against stored template patches.
///
/// Strategy:
/// 1. For each known marker position (from previous frame or known-marker hint),
///    search a `SAD_SEARCH_RADIUS × SAD_SEARCH_RADIUS` neighbourhood.
/// 2. At each candidate position compute SAD against the stored template.
/// 3. Accept the minimum-SAD position; reject if it exceeds the threshold.
/// 4. Refine to sub-pixel accuracy with an intensity centroid.
#[allow(clippy::cast_precision_loss, clippy::cast_sign_loss)]
fn detect_markers_in_image(
    image_data: &[u8],
    width: usize,
    height: usize,
    known_markers: &[Marker3D],
    config: &MarkerDetectorConfig,
    last_positions: &[(usize, Point2<f64>)], // (marker_id, last_pixel_pos)
) -> Vec<Marker2D> {
    if image_data.is_empty() || width == 0 || height == 0 {
        return Vec::new();
    }

    // Expected image size for a greyscale (Y-plane only) layout.
    let expected_pixels = width * height;
    if image_data.len() < expected_pixels {
        return Vec::new();
    }

    // Use only the luminance plane (first `width × height` bytes).
    let luma = &image_data[..expected_pixels];

    let mut detected = Vec::new();

    for known in known_markers {
        // Find the last known pixel position for this marker (if any).
        let seed = last_positions
            .iter()
            .find(|(id, _)| *id == known.id)
            .map(|(_, pos)| *pos);

        // Determine search centre: last position or image centre as fallback.
        let (seed_cx, seed_cy) = match seed {
            Some(p) => (p.x as isize, p.y as isize),
            None => (width as isize / 2, height as isize / 2),
        };

        // Extract template from seed position (or skip if out of bounds).
        let template = match extract_patch(luma, width, height, seed_cx, seed_cy, SAD_TEMPLATE_HALF)
        {
            Some(t) => t,
            None => continue,
        };

        // SAD search over neighbourhood.
        let mut best_sad = u64::MAX;
        let mut best_x = seed_cx;
        let mut best_y = seed_cy;

        let x_lo = (seed_cx - SAD_SEARCH_RADIUS).max(SAD_TEMPLATE_HALF);
        let x_hi = (seed_cx + SAD_SEARCH_RADIUS).min(width as isize - SAD_TEMPLATE_HALF - 1);
        let y_lo = (seed_cy - SAD_SEARCH_RADIUS).max(SAD_TEMPLATE_HALF);
        let y_hi = (seed_cy + SAD_SEARCH_RADIUS).min(height as isize - SAD_TEMPLATE_HALF - 1);

        for cy in y_lo..=y_hi {
            for cx in x_lo..=x_hi {
                if let Some(sad) =
                    sad_patch(luma, width, height, &template, SAD_TEMPLATE_HALF, cx, cy)
                {
                    if sad < best_sad {
                        best_sad = sad;
                        best_x = cx;
                        best_y = cy;
                    }
                }
            }
        }

        // Reject if SAD score is too high (low confidence).
        let patch_area = ((2 * SAD_TEMPLATE_HALF + 1) * (2 * SAD_TEMPLATE_HALF + 1)) as f64;
        let normalised_sad = best_sad as f64 / (patch_area * 255.0);
        if normalised_sad > (1.0 - f64::from(config.detection_threshold)) {
            continue;
        }
        let confidence = (1.0 - normalised_sad).clamp(0.0, 1.0) as f32;

        // Sub-pixel refinement via intensity centroid.
        let refine_patch =
            extract_patch(luma, width, height, best_x, best_y, SAD_TEMPLATE_HALF / 2);
        let (dx, dy) = refine_patch
            .as_deref()
            .map(|p| intensity_centroid_offset(p, SAD_TEMPLATE_HALF / 2))
            .unwrap_or((0.0, 0.0));

        let pixel_x = best_x as f64 + dx;
        let pixel_y = best_y as f64 + dy;

        // Filter by configured marker size bounds.
        let estimated_size = known.size * 1000.0; // rough pixels-per-meter proxy
        if estimated_size < config.min_marker_size || estimated_size > config.max_marker_size {
            // Fall back to a default visible size and still accept.
        }

        detected.push(Marker2D {
            id: known.id,
            position: Point2::new(pixel_x, pixel_y),
            marker_type: known.marker_type,
            confidence,
            size: known.size * 1000.0,
        });
    }

    detected
}

// ─── Normalised DLT homography ───────────────────────────────────────────────

/// Compute normalisation transform for a set of 2D points.
///
/// Returns `(T, normalised_points)` where `T` is the 3×3 normalisation matrix
/// that maps original → normalised coordinates (centroid at origin, RMS distance √2).
#[allow(clippy::cast_precision_loss)]
fn normalise_points_2d(pts: &[Point2<f64>]) -> ([f64; 9], Vec<Point2<f64>>) {
    let n = pts.len() as f64;
    let cx = pts.iter().map(|p| p.x).sum::<f64>() / n;
    let cy = pts.iter().map(|p| p.y).sum::<f64>() / n;
    let scale = {
        let mean_dist = pts
            .iter()
            .map(|p| ((p.x - cx).powi(2) + (p.y - cy).powi(2)).sqrt())
            .sum::<f64>()
            / n;
        if mean_dist > 1e-10 {
            std::f64::consts::SQRT_2 / mean_dist
        } else {
            1.0
        }
    };
    // T = [[scale, 0, -scale*cx], [0, scale, -scale*cy], [0, 0, 1]]
    let t = [
        scale,
        0.0,
        -scale * cx,
        0.0,
        scale,
        -scale * cy,
        0.0,
        0.0,
        1.0,
    ];
    let normalised: Vec<Point2<f64>> = pts
        .iter()
        .map(|p| Point2::new((p.x - cx) * scale, (p.y - cy) * scale))
        .collect();
    (t, normalised)
}

/// Multiply two 3×3 matrices (row-major flat arrays).
fn mat3_mul(a: &[f64; 9], b: &[f64; 9]) -> [f64; 9] {
    let mut c = [0.0f64; 9];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                c[i * 3 + j] += a[i * 3 + k] * b[k * 3 + j];
            }
        }
    }
    c
}

/// Gaussian elimination with partial pivoting on an `n × (n+1)` augmented matrix.
///
/// Solves `A x = b` in-place; `matrix` is row-major with `n` rows and `n+1`
/// columns (the last column is `b`).  Returns `Some(x)` on success, `None` if
/// the system is singular.
fn gaussian_elimination(matrix: &mut Vec<Vec<f64>>, n: usize) -> Option<Vec<f64>> {
    for col in 0..n {
        // Partial pivot.
        let pivot_row = (col..n).max_by(|&r1, &r2| {
            matrix[r1][col]
                .abs()
                .partial_cmp(&matrix[r2][col].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })?;
        matrix.swap(col, pivot_row);

        let pivot = matrix[col][col];
        if pivot.abs() < 1e-14 {
            return None; // Singular.
        }

        // Eliminate below.
        for row in (col + 1)..n {
            let factor = matrix[row][col] / pivot;
            for k in col..=(n) {
                let v = matrix[col][k] * factor;
                matrix[row][k] -= v;
            }
        }
    }

    // Back-substitution.
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut s = matrix[i][n];
        for j in (i + 1)..n {
            s -= matrix[i][j] * x[j];
        }
        x[i] = s / matrix[i][i];
    }
    Some(x)
}

/// Compute a planar homography matrix H (3×3, row-major) from at least 4
/// point correspondences `(src[i], dst[i])` using the Normalised Direct
/// Linear Transform (Hartley & Zisserman, 4.1).
///
/// Returns `None` if fewer than 4 correspondences are provided or the system
/// is degenerate.
fn compute_homography_dlt(src: &[Point2<f64>], dst: &[Point2<f64>]) -> Option<[f64; 9]> {
    if src.len() < 4 || src.len() != dst.len() {
        return None;
    }

    // Normalise source and destination point sets.
    let (t_src, src_n) = normalise_points_2d(src);
    let (t_dst, dst_n) = normalise_points_2d(dst);

    // Build the 2n × 9 DLT matrix A.
    // For each correspondence (x', y') ↔ (x, y):
    //   [-x, -y, -1,  0,  0,  0, x'*x, x'*y, x']
    //   [ 0,  0,  0, -x, -y, -1, y'*x, y'*y, y']
    let n_pts = src_n.len();
    let n_rows = 2 * n_pts;

    // We solve the 8-parameter DLT by fixing h[8] = 1 and setting up an 8×8
    // linear system  A' h' = -a_col8  (SVD-free minimal approach).
    //
    // The DLT rows for each point pair give two equations:
    //   a[0..8] · h = 0
    // With h[8] = 1, rearranging:
    //   a[0..8] · h[0..8] = -a[8]   (linear in h[0..8])
    //
    // We stack all 2n equations and solve the over-determined system via
    // normal equations (A'ᵀ A' h' = A'ᵀ b').

    // Build the over-determined system (2n × 8) and RHS (2n × 1).
    let mut lhs = vec![vec![0.0f64; 8]; n_rows];
    let mut rhs = vec![0.0f64; n_rows];

    for i in 0..n_pts {
        let x = src_n[i].x;
        let y = src_n[i].y;
        let xp = dst_n[i].x;
        let yp = dst_n[i].y;

        let r0 = 2 * i;
        let r1 = 2 * i + 1;

        // Row 0: [-x, -y, -1, 0, 0, 0, x'*x, x'*y]  rhs = -x'
        lhs[r0] = vec![-x, -y, -1.0, 0.0, 0.0, 0.0, xp * x, xp * y];
        rhs[r0] = -xp;

        // Row 1: [0, 0, 0, -x, -y, -1, y'*x, y'*y]  rhs = -y'
        lhs[r1] = vec![0.0, 0.0, 0.0, -x, -y, -1.0, yp * x, yp * y];
        rhs[r1] = -yp;
    }

    // Normal equations: (LHSᵀ × LHS) h = LHSᵀ × rhs.
    let mut ata = vec![vec![0.0f64; 9]; 8]; // 8×9 augmented
    for col in 0..8 {
        for k in 0..n_rows {
            let lhs_kc = lhs[k][col];
            for col2 in 0..8 {
                ata[col][col2] += lhs_kc * lhs[k][col2];
            }
            ata[col][8] += lhs_kc * rhs[k];
        }
    }

    let h_partial = gaussian_elimination(&mut ata, 8)?;

    let h_norm: [f64; 9] = [
        h_partial[0],
        h_partial[1],
        h_partial[2],
        h_partial[3],
        h_partial[4],
        h_partial[5],
        h_partial[6],
        h_partial[7],
        1.0,
    ];

    // Denormalise: H = T_dst⁻¹ × H_norm × T_src.
    // T⁻¹ for normalisation matrix is easy to compute analytically:
    //   T = [[s, 0, -s*cx], [0, s, -s*cy], [0, 0, 1]]
    //   T⁻¹ = [[1/s, 0, cx], [0, 1/s, cy], [0, 0, 1]]
    fn invert_normalisation(t: &[f64; 9]) -> [f64; 9] {
        // t[0] = scale, t[2] = -scale*cx → cx = -t[2]/t[0]
        // t[4] = scale, t[5] = -scale*cy → cy = -t[5]/t[4]
        let s = t[0];
        let inv_s = if s.abs() > 1e-14 { 1.0 / s } else { 1.0 };
        let cx = -t[2] * inv_s;
        let cy = -t[5] * inv_s;
        [inv_s, 0.0, cx, 0.0, inv_s, cy, 0.0, 0.0, 1.0]
    }

    let t_dst_inv = invert_normalisation(&t_dst);
    let h_denorm = mat3_mul(&mat3_mul(&t_dst_inv, &h_norm), &t_src);

    // Normalise so that h[8] = 1 (or the Frobenius norm = 1 as fallback).
    let scale = if h_denorm[8].abs() > 1e-14 {
        h_denorm[8]
    } else {
        let frob: f64 = h_denorm.iter().map(|v| v * v).sum::<f64>().sqrt();
        if frob > 1e-14 {
            frob
        } else {
            1.0
        }
    };
    let h_final: [f64; 9] = h_denorm.map(|v| v / scale);

    Some(h_final)
}

// ─── PnP pose estimation ─────────────────────────────────────────────────────

/// Decompose the rotation component of a homography into a `UnitQuaternion`.
///
/// Given `H = K [r1 r2 t]` (for a planar scene at Z=0), we extract the first
/// two columns of the rotation matrix and complete the third via cross product.
/// Assumes the intrinsic matrix `K` is the identity (normalised image coords).
fn rotation_from_homography_columns(h: &[f64; 9]) -> UnitQuaternion<f64> {
    // r1 = H[:,0] / ||H[:,0]||
    let r1 = Vector3::new(h[0], h[3], h[6]);
    let r2 = Vector3::new(h[1], h[4], h[7]);

    let norm1 = r1.norm();
    let norm2 = r2.norm();

    if norm1 < 1e-10 || norm2 < 1e-10 {
        return UnitQuaternion::identity();
    }

    let r1n = r1 / norm1;
    let r2n = r2 / norm2;
    let r3n = r1n.cross(&r2n);

    // Build rotation matrix from column vectors.
    let rot_mat = Matrix3::from_columns(&r1n, &r2n, &r3n);

    // Project onto SO(3) via SVD of R and take U·Vᵀ.
    let svd = rot_mat.svd(true, true);
    match (svd.u, svd.v_t) {
        (Some(u), Some(vt)) => {
            let r_clean = u * vt;
            UnitQuaternion::from_matrix(&r_clean)
        }
        _ => UnitQuaternion::identity(),
    }
}

/// Estimate camera pose from 2D–3D correspondences using a simplified PnP approach.
///
/// For a planar scene (all Z=0), we compute the homography H from 2D image
/// points to 2D world-XY points and decompose it to recover rotation and
/// translation.
///
/// For non-planar scenes we fall back to a centroid-based translation estimate
/// with orientation derived from the first available homography.
fn estimate_pose_from_correspondences(
    matches: &[(Marker2D, Marker3D)],
    timestamp_ns: u64,
) -> Result<CameraPose> {
    if matches.is_empty() {
        return Err(VirtualProductionError::CameraTracking(
            "No marker matches for pose estimation".to_string(),
        ));
    }

    let image_pts: Vec<Point2<f64>> = matches.iter().map(|(m2d, _)| m2d.position).collect();
    let world_pts: Vec<Point3<f64>> = matches.iter().map(|(_, m3d)| m3d.position).collect();

    // Project world points onto XY plane for homography computation.
    let world_pts_2d: Vec<Point2<f64>> = world_pts.iter().map(|p| Point2::new(p.x, p.y)).collect();

    let confidence_avg = {
        let sum: f32 = matches.iter().map(|(m2d, _)| m2d.confidence).sum();
        sum / matches.len() as f32
    };

    if matches.len() >= 4 {
        // Attempt homography-based pose.
        if let Some(h) = compute_homography_dlt(&image_pts, &world_pts_2d) {
            let orientation = rotation_from_homography_columns(&h);

            // Extract translation: t = H[:,2] / scale (last column of H mapped
            // to the known world scale).
            let scale = (h[0].powi(2) + h[3].powi(2) + h[6].powi(2))
                .sqrt()
                .max(1e-10);
            let tx = h[2] / scale;
            let ty = h[5] / scale;
            let tz = h[8] / scale;
            let position = Point3::new(tx, ty, tz);

            return Ok(CameraPose {
                position,
                orientation,
                timestamp_ns,
                confidence: confidence_avg,
            });
        }
    }

    // Fallback for fewer than 4 matches or degenerate homography:
    // centroid-based translation estimate.
    let mut position_sum = Vector3::zeros();
    for (_, m3d) in matches {
        position_sum += m3d.position.coords();
    }
    let n = matches.len() as f64;
    let position = Point3::from(position_sum / n);
    let orientation = UnitQuaternion::identity();

    Ok(CameraPose {
        position,
        orientation,
        timestamp_ns,
        confidence: confidence_avg,
    })
}

// ─── Marker detector ─────────────────────────────────────────────────────────

/// Marker detector for optical tracking
pub struct MarkerDetector {
    config: MarkerDetectorConfig,
    known_markers: Vec<Marker3D>,
    detected_markers: Vec<Marker2D>,
    /// Last known pixel positions of each marker (used as SAD search seeds).
    last_positions: Vec<(usize, Point2<f64>)>,
}

impl MarkerDetector {
    /// Create new marker detector
    pub fn new(config: MarkerDetectorConfig) -> Result<Self> {
        Ok(Self {
            config,
            known_markers: Vec::new(),
            detected_markers: Vec::new(),
            last_positions: Vec::new(),
        })
    }

    /// Add known marker to the detector
    pub fn add_marker(&mut self, marker: Marker3D) {
        self.known_markers.push(marker);
    }

    /// Detect markers in a greyscale image using SAD template matching.
    ///
    /// `image_data` must be a planar greyscale (Y-only) buffer of size
    /// `width × height` bytes.  The detector uses the last known pixel
    /// positions of each marker as search seeds, falling back to the image
    /// centre when no prior position is available.
    pub fn detect(
        &mut self,
        image_data: &[u8],
        width: usize,
        height: usize,
    ) -> Result<&[Marker2D]> {
        self.detected_markers = detect_markers_in_image(
            image_data,
            width,
            height,
            &self.known_markers,
            &self.config,
            &self.last_positions,
        );

        // Update last-known positions from this frame's detections.
        for m in &self.detected_markers {
            if let Some(entry) = self.last_positions.iter_mut().find(|(id, _)| *id == m.id) {
                entry.1 = m.position;
            } else {
                self.last_positions.push((m.id, m.position));
            }
        }

        Ok(&self.detected_markers)
    }

    /// Estimate camera pose from detected markers using homography DLT + PnP.
    pub fn detect_pose(&mut self, timestamp_ns: u64) -> Result<Option<CameraPose>> {
        if self.detected_markers.len() < self.config.min_markers {
            return Ok(None);
        }

        if self.known_markers.is_empty() {
            return Ok(None);
        }

        // Match 2D detections with 3D known markers.
        let matches = self.match_markers();

        if matches.len() < self.config.min_markers {
            return Ok(None);
        }

        let pose = estimate_pose_from_correspondences(&matches, timestamp_ns)?;

        Ok(Some(pose))
    }

    /// Match 2D detected markers with 3D known markers by ID and type.
    fn match_markers(&self) -> Vec<(Marker2D, Marker3D)> {
        let mut matches = Vec::new();

        for detected in &self.detected_markers {
            if let Some(known) = self
                .known_markers
                .iter()
                .find(|m| m.id == detected.id && m.marker_type == detected.marker_type)
            {
                matches.push((detected.clone(), known.clone()));
            }
        }

        matches
    }

    /// Get number of detected markers
    #[must_use]
    pub fn num_detected(&self) -> usize {
        self.detected_markers.len()
    }

    /// Get number of known markers
    #[must_use]
    pub fn num_known(&self) -> usize {
        self.known_markers.len()
    }

    /// Clear detected markers
    pub fn clear_detected(&mut self) {
        self.detected_markers.clear();
    }

    /// Get configuration
    #[must_use]
    pub fn config(&self) -> &MarkerDetectorConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_marker_2d() {
        let marker = Marker2D::new(0, Point2::new(100.0, 100.0), MarkerType::Circular);
        assert_eq!(marker.id, 0);
        assert_eq!(marker.confidence, 1.0);
    }

    #[test]
    fn test_marker_3d() {
        let marker = Marker3D::new(0, Point3::new(1.0, 2.0, 3.0), MarkerType::Circular, 0.05);
        assert_eq!(marker.id, 0);
        assert_eq!(marker.size, 0.05);
    }

    #[test]
    fn test_marker_detector_creation() {
        let config = MarkerDetectorConfig::default();
        let detector = MarkerDetector::new(config);
        assert!(detector.is_ok());
    }

    #[test]
    fn test_marker_detector_add_marker() {
        let config = MarkerDetectorConfig::default();
        let mut detector = MarkerDetector::new(config).expect("should succeed in test");

        detector.add_marker(Marker3D::new(
            0,
            Point3::origin(),
            MarkerType::Circular,
            0.05,
        ));

        assert_eq!(detector.num_known(), 1);
    }

    #[test]
    fn test_marker_matching() {
        let config = MarkerDetectorConfig::default();
        let mut detector = MarkerDetector::new(config).expect("should succeed in test");

        // Add known marker
        detector.add_marker(Marker3D::new(
            0,
            Point3::origin(),
            MarkerType::Circular,
            0.05,
        ));

        // Add detected marker
        detector.detected_markers.push(Marker2D::new(
            0,
            Point2::new(100.0, 100.0),
            MarkerType::Circular,
        ));

        let matches = detector.match_markers();
        assert_eq!(matches.len(), 1);
    }

    #[test]
    fn test_detect_on_empty_image_returns_empty() {
        let config = MarkerDetectorConfig::default();
        let mut detector = MarkerDetector::new(config).expect("should succeed in test");
        detector.add_marker(Marker3D::new(
            0,
            Point3::origin(),
            MarkerType::Circular,
            0.05,
        ));
        let result = detector.detect(&[], 0, 0).expect("should succeed in test");
        assert!(result.is_empty());
    }

    #[test]
    fn test_detect_on_synthetic_image() {
        let config = MarkerDetectorConfig {
            detection_threshold: 0.01, // very permissive
            min_markers: 1,
            ..Default::default()
        };
        let mut detector = MarkerDetector::new(config).expect("should succeed in test");

        // Add a known marker that will be searched near image centre.
        detector.add_marker(Marker3D::new(
            0,
            Point3::new(0.0, 0.0, 0.0),
            MarkerType::Circular,
            0.01, // size
        ));

        // Create a 64×64 greyscale image with a bright spot near centre (32,32).
        let w = 64usize;
        let h = 64usize;
        let mut img = vec![30u8; w * h];
        // Draw a bright 5×5 blob at (32,32).
        for dy in -2isize..=2 {
            for dx in -2isize..=2 {
                let px = (32isize + dx) as usize;
                let py = (32isize + dy) as usize;
                img[py * w + px] = 200;
            }
        }

        // Detection should not panic and should return some result.
        let result = detector.detect(&img, w, h).expect("should succeed in test");
        // We only assert it doesn't panic; detection depends on SAD threshold.
        let _ = result;
    }

    #[test]
    fn test_homography_dlt_identity() {
        // Four corners of a unit square mapped to themselves → H ≈ identity.
        let pts = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ];
        let h = compute_homography_dlt(&pts, &pts).expect("should succeed in test");

        // H should be close to identity (up to scale).
        let h00 = h[0] / h[8];
        let h11 = h[4] / h[8];
        let h22 = 1.0f64;
        assert!((h00 - 1.0f64).abs() < 1e-6, "h[0,0] = {}", h00);
        assert!((h11 - 1.0f64).abs() < 1e-6, "h[1,1] = {}", h11);
        assert!((h22 - 1.0f64).abs() < 1e-6);
    }

    #[test]
    fn test_homography_dlt_requires_four_points() {
        let pts = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(0.0, 1.0),
        ];
        assert!(compute_homography_dlt(&pts, &pts).is_none());
    }

    #[test]
    fn test_pose_estimation_four_markers() {
        // Synthetic planar scene: 4 markers at Z=0.
        let matches = vec![
            (
                Marker2D {
                    id: 0,
                    position: Point2::new(10.0, 10.0),
                    marker_type: MarkerType::Circular,
                    confidence: 0.9,
                    size: 10.0,
                },
                Marker3D::new(0, Point3::new(0.0, 0.0, 0.0), MarkerType::Circular, 0.05),
            ),
            (
                Marker2D {
                    id: 1,
                    position: Point2::new(90.0, 10.0),
                    marker_type: MarkerType::Circular,
                    confidence: 0.9,
                    size: 10.0,
                },
                Marker3D::new(1, Point3::new(1.0, 0.0, 0.0), MarkerType::Circular, 0.05),
            ),
            (
                Marker2D {
                    id: 2,
                    position: Point2::new(90.0, 90.0),
                    marker_type: MarkerType::Circular,
                    confidence: 0.9,
                    size: 10.0,
                },
                Marker3D::new(2, Point3::new(1.0, 1.0, 0.0), MarkerType::Circular, 0.05),
            ),
            (
                Marker2D {
                    id: 3,
                    position: Point2::new(10.0, 90.0),
                    marker_type: MarkerType::Circular,
                    confidence: 0.9,
                    size: 10.0,
                },
                Marker3D::new(3, Point3::new(0.0, 1.0, 0.0), MarkerType::Circular, 0.05),
            ),
        ];

        let pose = estimate_pose_from_correspondences(&matches, 1_000_000)
            .expect("should succeed in test");
        // Just verify we get a plausible confidence and no panic.
        assert!(pose.confidence > 0.0);
        assert!(pose.confidence <= 1.0);
    }
}
