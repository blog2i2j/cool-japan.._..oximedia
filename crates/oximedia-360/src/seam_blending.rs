//! Seam blending at cubemap face boundaries.
//!
//! When a cubemap is assembled from individual face images (e.g. after
//! per-face processing or re-encoding), the boundary pixels of adjacent faces
//! may not align perfectly, producing visible seams in the rendered 360° view.
//! This module implements two complementary techniques to reduce these artefacts:
//!
//! 1. **Border feathering** — A cosine-weighted alpha ramp is applied to each
//!    face image near its boundary pixels so that the blended overlap region
//!    fades smoothly to the neighbouring face.
//!
//! 2. **Cross-face averaging** — For each boundary pixel on a face, the colour
//!    is averaged with the corresponding mirror pixel on the adjacent face,
//!    weighted by the feather alpha.  This eliminates first-order colour
//!    discontinuities at seams.
//!
//! ## Pixel format
//!
//! All operations work on packed **RGB** images (3 bytes per pixel, row-major).
//! The face images are assumed to be square (`size × size`).
//!
//! ## Usage
//!
//! ```rust
//! use oximedia_360::seam_blending::{feather_face, SeamBlender, BlendEdge};
//!
//! // Tiny 4×4 face, all white
//! let face = vec![255u8; 4 * 4 * 3];
//! let result = feather_face(&face, 4, 2).expect("ok");
//! assert_eq!(result.len(), face.len());
//! ```

use crate::VrError;

// ─── BlendEdge ────────────────────────────────────────────────────────────────

/// Which edge(s) of a cubemap face to apply seam blending to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendEdge {
    /// The top row (row 0).
    Top,
    /// The bottom row (row size−1).
    Bottom,
    /// The left column (col 0).
    Left,
    /// The right column (col size−1).
    Right,
}

impl BlendEdge {
    /// All four edges.
    pub const ALL: [BlendEdge; 4] = [
        BlendEdge::Top,
        BlendEdge::Bottom,
        BlendEdge::Left,
        BlendEdge::Right,
    ];
}

// ─── feather_weights ──────────────────────────────────────────────────────────

/// Compute a 1-D cosine feather weight ramp of length `size`.
///
/// The ramp starts at 0.0 (at the boundary pixel, index 0) and rises to 1.0
/// at the interior (index `size − 1`).  The transition follows a cosine curve
/// over the first `blend_width` pixels; beyond that the weight is 1.0.
///
/// # Parameters
///
/// * `size`        — length of the ramp (must be ≥ 1)
/// * `blend_width` — number of pixels to blend over (clamped to `size`)
///
/// # Errors
///
/// Returns [`VrError::InvalidDimensions`] if `size` is 0.
pub fn feather_weights(size: u32, blend_width: u32) -> Result<Vec<f32>, VrError> {
    if size == 0 {
        return Err(VrError::InvalidDimensions("size must be > 0".into()));
    }
    let bw = blend_width.min(size) as usize;
    let n = size as usize;
    let mut weights = vec![1.0f32; n];
    for i in 0..bw {
        // Cosine ramp: 0 at boundary, 1 at interior
        let t = i as f32 / bw.max(1) as f32;
        weights[i] = 0.5 * (1.0 - (PI * t).cos());
    }
    Ok(weights)
}

// ─── feather_face ────────────────────────────────────────────────────────────

/// Apply a cosine feather alpha ramp along all four edges of a square face.
///
/// Each channel value is multiplied by `min(ramp_from_left, ramp_from_right,
/// ramp_from_top, ramp_from_bottom)`, where each ramp is evaluated using
/// [`feather_weights`].
///
/// # Parameters
///
/// * `pixels`      — packed RGB input (3 bytes per pixel, row-major)
/// * `size`        — side length of the square face image
/// * `blend_width` — number of border pixels to blend over
///
/// # Errors
///
/// Returns [`VrError::InvalidDimensions`] if `size` is 0, or
/// [`VrError::BufferTooSmall`] if `pixels` is too small.
pub fn feather_face(pixels: &[u8], size: u32, blend_width: u32) -> Result<Vec<u8>, VrError> {
    validate_face_buffer(pixels, size)?;

    let weights = feather_weights(size, blend_width)?;
    let n = size as usize;
    let mut out = pixels.to_vec();

    for row in 0..n {
        let w_v = weights[row].min(weights[n - 1 - row]);
        for col in 0..n {
            let w_h = weights[col].min(weights[n - 1 - col]);
            let w = w_v.min(w_h);
            let base = (row * n + col) * 3;
            out[base] = ((out[base] as f32) * w).round() as u8;
            out[base + 1] = ((out[base + 1] as f32) * w).round() as u8;
            out[base + 2] = ((out[base + 2] as f32) * w).round() as u8;
        }
    }
    Ok(out)
}

// ─── SeamBlender ─────────────────────────────────────────────────────────────

/// Cross-face seam blender for assembled cubemap faces.
///
/// Takes six pre-rendered face images and blends the boundary pixels of
/// adjacent faces together, eliminating first-order colour discontinuities
/// at cubemap seams.
///
/// ## Face ordering
///
/// Faces are passed as an array of six slices in the canonical order:
/// `[Front, Back, Left, Right, Top, Bottom]` (matching [`crate::CubeFace`]).
///
/// ## Algorithm
///
/// For each shared seam between two adjacent faces, `blend_width` rows/columns
/// of pixels on each face are blended with the mirror pixels from the
/// neighbouring face.  The mix weight is determined by a cosine feather ramp:
/// boundary pixels are 50% self / 50% neighbour; the mix ratio fades toward
/// 100% self over the `blend_width` region.
#[derive(Debug, Clone)]
pub struct SeamBlender {
    /// Side length of each face in pixels.
    pub face_size: u32,
    /// Number of pixels to blend on each side of the seam.
    pub blend_width: u32,
}

impl SeamBlender {
    /// Create a new seam blender.
    ///
    /// # Errors
    ///
    /// Returns [`VrError::InvalidDimensions`] if `face_size` is 0.
    pub fn new(face_size: u32, blend_width: u32) -> Result<Self, VrError> {
        if face_size == 0 {
            return Err(VrError::InvalidDimensions("face_size must be > 0".into()));
        }
        Ok(Self {
            face_size,
            blend_width,
        })
    }

    /// Blend a seam between `face_a` and `face_b` along the specified edges.
    ///
    /// `edge_a` is the edge on `face_a` that borders `face_b`.
    /// `edge_b` is the corresponding edge on `face_b` that borders `face_a`.
    ///
    /// Both faces are modified **in place**.
    ///
    /// # Errors
    ///
    /// Returns [`VrError::BufferTooSmall`] if either buffer is too small for
    /// the declared `face_size`.
    pub fn blend_seam(
        &self,
        face_a: &mut [u8],
        edge_a: BlendEdge,
        face_b: &mut [u8],
        edge_b: BlendEdge,
    ) -> Result<(), VrError> {
        validate_face_buffer(face_a, self.face_size)?;
        validate_face_buffer(face_b, self.face_size)?;

        let weights = feather_weights(self.face_size, self.blend_width)?;
        let bw = self.blend_width.min(self.face_size) as usize;
        let n = self.face_size as usize;

        for depth in 0..bw {
            // Weight: 0.5 at boundary, rising toward 1.0 (self) at depth bw-1
            let self_weight = 0.5 + 0.5 * weights[depth];
            let neighbour_weight = 1.0 - self_weight;

            for pos in 0..n {
                let (idx_a, idx_b) = seam_pixel_indices(edge_a, edge_b, depth, pos, n);

                let base_a = idx_a * 3;
                let base_b = idx_b * 3;

                for ch in 0..3 {
                    let va = face_a[base_a + ch] as f32;
                    let vb = face_b[base_b + ch] as f32;
                    face_a[base_a + ch] = (va * self_weight + vb * neighbour_weight).round() as u8;
                    face_b[base_b + ch] = (vb * self_weight + va * neighbour_weight).round() as u8;
                }
            }
        }
        Ok(())
    }

    /// Apply border feathering to a single face (convenience wrapper).
    ///
    /// See [`feather_face`] for details.
    pub fn feather(&self, pixels: &[u8]) -> Result<Vec<u8>, VrError> {
        feather_face(pixels, self.face_size, self.blend_width)
    }

    /// Blend all four edges of `face` against the provided adjacent faces.
    ///
    /// `neighbours` maps each [`BlendEdge`] to `(face_pixels, mirror_edge)`.
    /// Any edge without a neighbour is feathered instead.
    ///
    /// # Errors
    ///
    /// Returns [`VrError::BufferTooSmall`] if any buffer is too small.
    pub fn blend_all_edges(
        &self,
        face: &mut Vec<u8>,
        neighbours: &mut [(BlendEdge, &mut Vec<u8>, BlendEdge)],
    ) -> Result<(), VrError> {
        validate_face_buffer(face, self.face_size)?;
        for (_, nb, _) in neighbours.iter() {
            validate_face_buffer(nb, self.face_size)?;
        }

        for (edge_a, nb, edge_b) in neighbours.iter_mut() {
            let edge_a = *edge_a;
            let edge_b = *edge_b;
            self.blend_seam(face, edge_a, nb, edge_b)?;
        }
        Ok(())
    }
}

// ─── SeamQualityMetrics ───────────────────────────────────────────────────────

/// Measures the quality of a seam between two adjacent face edges.
///
/// Returns statistics about the colour difference across the seam boundary,
/// which can be used to decide whether blending is needed and to validate
/// the effectiveness of seam blending.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SeamQualityMetrics {
    /// Mean absolute difference of boundary pixel colours (0–255 scale).
    pub mean_absolute_diff: f32,
    /// Maximum absolute difference across any boundary pixel channel.
    pub max_absolute_diff: f32,
    /// Root-mean-square colour error across the seam.
    pub rms_error: f32,
}

impl SeamQualityMetrics {
    /// Compute seam quality metrics for the given face edge pair.
    ///
    /// `edge_a` and `edge_b` identify which edges of `face_a` and `face_b`
    /// face each other.  Only the outermost row/column of pixels (depth 0)
    /// is compared.
    ///
    /// # Errors
    ///
    /// Returns [`VrError::BufferTooSmall`] if either buffer is too small.
    pub fn compute(
        face_a: &[u8],
        edge_a: BlendEdge,
        face_b: &[u8],
        edge_b: BlendEdge,
        face_size: u32,
    ) -> Result<Self, VrError> {
        validate_face_buffer(face_a, face_size)?;
        validate_face_buffer(face_b, face_size)?;

        let n = face_size as usize;
        let mut sum_abs = 0.0f32;
        let mut sum_sq = 0.0f32;
        let mut max_abs = 0.0f32;
        let mut count = 0usize;

        for pos in 0..n {
            let (idx_a, idx_b) = seam_pixel_indices(edge_a, edge_b, 0, pos, n);
            let base_a = idx_a * 3;
            let base_b = idx_b * 3;

            for ch in 0..3 {
                let diff = (face_a[base_a + ch] as f32 - face_b[base_b + ch] as f32).abs();
                sum_abs += diff;
                sum_sq += diff * diff;
                if diff > max_abs {
                    max_abs = diff;
                }
                count += 1;
            }
        }

        let n_f = count.max(1) as f32;
        Ok(Self {
            mean_absolute_diff: sum_abs / n_f,
            max_absolute_diff: max_abs,
            rms_error: (sum_sq / n_f).sqrt(),
        })
    }

    /// Returns `true` if the seam quality is acceptable (RMS error ≤ threshold).
    #[must_use]
    pub fn is_acceptable(&self, rms_threshold: f32) -> bool {
        self.rms_error <= rms_threshold
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

const PI: f32 = std::f32::consts::PI;

/// Validate that `pixels` is large enough for a `size × size` RGB face.
fn validate_face_buffer(pixels: &[u8], size: u32) -> Result<(), VrError> {
    if size == 0 {
        return Err(VrError::InvalidDimensions("face size must be > 0".into()));
    }
    let expected = (size as usize) * (size as usize) * 3;
    if pixels.len() < expected {
        return Err(VrError::BufferTooSmall {
            expected,
            got: pixels.len(),
        });
    }
    Ok(())
}

/// Compute pixel linear indices for `face_a[depth, pos]` and `face_b[depth, pos]`
/// given the edge relationship.
///
/// * `depth` — how many pixels from the boundary (0 = boundary, 1 = one inward…)
/// * `pos`   — position along the edge (0 .. n)
/// * `n`     — face size
///
/// Returns `(linear_index_in_face_a, linear_index_in_face_b)`.
fn seam_pixel_indices(
    edge_a: BlendEdge,
    edge_b: BlendEdge,
    depth: usize,
    pos: usize,
    n: usize,
) -> (usize, usize) {
    let idx_a = edge_pixel_index(edge_a, depth, pos, n);
    // On face_b the "depth" grows from the *other* edge inward, so
    // depth 0 on face_a corresponds to depth 0 on face_b (both are boundary).
    let idx_b = edge_pixel_index(edge_b, depth, pos, n);
    (idx_a, idx_b)
}

/// Compute the linear pixel index within a face for a given edge/depth/pos.
///
/// * `edge`  — which edge of the face
/// * `depth` — distance from the edge (0 = outermost boundary row/col)
/// * `pos`   — position along the edge (0 = start, n−1 = end)
/// * `n`     — face size
fn edge_pixel_index(edge: BlendEdge, depth: usize, pos: usize, n: usize) -> usize {
    let row;
    let col;
    match edge {
        BlendEdge::Top => {
            row = depth;
            col = pos;
        }
        BlendEdge::Bottom => {
            row = n - 1 - depth;
            col = pos;
        }
        BlendEdge::Left => {
            row = pos;
            col = depth;
        }
        BlendEdge::Right => {
            row = pos;
            col = n - 1 - depth;
        }
    }
    row * n + col
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_face(size: u32, r: u8, g: u8, b: u8) -> Vec<u8> {
        let n = (size * size * 3) as usize;
        let mut v = Vec::with_capacity(n);
        for _ in 0..(size * size) {
            v.push(r);
            v.push(g);
            v.push(b);
        }
        v
    }

    // ── feather_weights ───────────────────────────────────────────────────────

    #[test]
    fn feather_weights_interior_is_one() {
        let w = feather_weights(16, 4).unwrap();
        // Pixels beyond the blend width should be 1.0
        for &v in &w[4..] {
            assert!((v - 1.0).abs() < 1e-5, "expected 1.0 got {v}");
        }
    }

    #[test]
    fn feather_weights_boundary_approaches_zero() {
        let w = feather_weights(16, 4).unwrap();
        // First pixel (depth 0) should have weight near 0
        assert!(w[0] < 0.2, "boundary weight too high: {}", w[0]);
    }

    #[test]
    fn feather_weights_monotonically_non_decreasing_in_blend_zone() {
        let w = feather_weights(20, 8).unwrap();
        for i in 1..8usize {
            assert!(w[i] >= w[i - 1], "weights not monotone at i={i}");
        }
    }

    #[test]
    fn feather_weights_rejects_zero_size() {
        assert!(feather_weights(0, 2).is_err());
    }

    // ── feather_face ──────────────────────────────────────────────────────────

    #[test]
    fn feather_face_output_size_unchanged() {
        let face = solid_face(8, 200, 150, 100);
        let out = feather_face(&face, 8, 2).unwrap();
        assert_eq!(out.len(), face.len());
    }

    #[test]
    fn feather_face_darkens_border_pixels() {
        let face = solid_face(8, 255, 255, 255);
        let out = feather_face(&face, 8, 3).unwrap();
        // Top-left corner pixel (row=0, col=0) should be dimmer than interior
        let corner_r = out[0] as f32;
        let centre = {
            let row = 4usize;
            let col = 4usize;
            out[(row * 8 + col) * 3] as f32
        };
        assert!(corner_r < centre, "corner={corner_r} centre={centre}");
    }

    #[test]
    fn feather_face_buffer_too_small_returns_error() {
        let face = vec![0u8; 10]; // Too small for 8×8×3
        assert!(feather_face(&face, 8, 2).is_err());
    }

    // ── SeamBlender ───────────────────────────────────────────────────────────

    #[test]
    fn seam_blender_new_rejects_zero_size() {
        assert!(SeamBlender::new(0, 2).is_err());
    }

    #[test]
    fn seam_blend_reduces_boundary_difference() {
        // face_a: all white (255), face_b: all black (0)
        let mut face_a = solid_face(8, 255, 255, 255);
        let mut face_b = solid_face(8, 0, 0, 0);

        let metrics_before =
            SeamQualityMetrics::compute(&face_a, BlendEdge::Right, &face_b, BlendEdge::Left, 8)
                .unwrap();

        let blender = SeamBlender::new(8, 3).unwrap();
        blender
            .blend_seam(&mut face_a, BlendEdge::Right, &mut face_b, BlendEdge::Left)
            .unwrap();

        let metrics_after =
            SeamQualityMetrics::compute(&face_a, BlendEdge::Right, &face_b, BlendEdge::Left, 8)
                .unwrap();

        assert!(
            metrics_after.mean_absolute_diff < metrics_before.mean_absolute_diff,
            "blending should reduce seam error: before={:.1} after={:.1}",
            metrics_before.mean_absolute_diff,
            metrics_after.mean_absolute_diff
        );
    }

    #[test]
    fn seam_quality_metrics_identical_faces_zero_error() {
        let face = solid_face(8, 128, 64, 32);
        let metrics =
            SeamQualityMetrics::compute(&face, BlendEdge::Top, &face, BlendEdge::Bottom, 8)
                .unwrap();
        assert!(
            metrics.rms_error < 1e-5,
            "identical faces should have zero error: {}",
            metrics.rms_error
        );
    }

    #[test]
    fn seam_quality_is_acceptable_threshold() {
        let m = SeamQualityMetrics {
            mean_absolute_diff: 5.0,
            max_absolute_diff: 10.0,
            rms_error: 6.0,
        };
        assert!(m.is_acceptable(7.0));
        assert!(!m.is_acceptable(5.0));
    }

    #[test]
    fn edge_pixel_index_top_row_correct() {
        // Top edge, depth=0 should be row 0
        let idx = edge_pixel_index(BlendEdge::Top, 0, 3, 8);
        assert_eq!(idx, 3, "top-edge depth=0 pos=3 should be pixel 3");
    }

    #[test]
    fn edge_pixel_index_bottom_row_correct() {
        // Bottom edge, depth=0 should be the last row
        let n = 8usize;
        let idx = edge_pixel_index(BlendEdge::Bottom, 0, 0, n);
        assert_eq!(
            idx,
            (n - 1) * n,
            "bottom-edge depth=0 pos=0 should be last row start"
        );
    }
}
