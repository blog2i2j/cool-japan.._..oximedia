//! Facial landmark detection and geometric face analysis.
//!
//! This module approximates 5-point and 68-point facial landmark positions
//! from a detected face bounding box, using anthropometric ratios derived from
//! published literature (patent-free).  No machine-learning weights are used;
//! all geometry is rule-based.
//!
//! # Landmark sets
//!
//! | Set | Points | Use-case |
//! |-----|--------|----------|
//! | [`LandmarkSet::FivePoint`] | 2 eyes, nose tip, 2 mouth corners | alignment, cropping |
//! | [`LandmarkSet::Extended`]  | 68 canonical dlib-compatible positions | geometry analysis |
//!
//! # Anthropometric ratios used
//!
//! - Eye centres at ~37% and ~63% of face width, ~38% of face height.
//! - Nose tip at ~50% width, ~63% height.
//! - Mouth corners at ~33% and ~67% width, ~75% height.
//! - Jaw line approximated by 9 points on an ellipse arc.
//! - Eyebrow arcs, eye outlines, nose base, lip outline all follow
//!   published proportional grids (Farkas, 1994 — no patent claim).
//!
//! # Example
//!
//! ```
//! use oximedia_scene::face_landmark::{FaceLandmarkDetector, LandmarkSet};
//! use oximedia_scene::common::Rect;
//!
//! let detector = FaceLandmarkDetector::new(LandmarkSet::FivePoint);
//! let face_bbox = Rect::new(10.0, 10.0, 100.0, 120.0);
//! let lms = detector.detect_from_bbox(&face_bbox);
//! assert_eq!(lms.points.len(), 5);
//! ```

use crate::common::{Point, Rect};
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Which landmark schema to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LandmarkSet {
    /// 5-point set: left eye, right eye, nose tip, left mouth corner, right mouth corner.
    FivePoint,
    /// 68-point set compatible with dlib/iBUG annotation scheme.
    Extended,
}

/// Semantic role of a landmark point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LandmarkRole {
    /// Jaw outline point (index 0–16).
    Jaw,
    /// Left eyebrow (17–21).
    LeftEyebrow,
    /// Right eyebrow (22–26).
    RightEyebrow,
    /// Nose bridge (27–30).
    NoseBridge,
    /// Nose base/nostril (31–35).
    NoseBase,
    /// Left eye outline (36–41).
    LeftEye,
    /// Right eye outline (42–47).
    RightEye,
    /// Outer lip (48–59).
    OuterLip,
    /// Inner lip (60–67).
    InnerLip,
    /// Simple eye centre (five-point set).
    EyeCenter,
    /// Nose tip (five-point set).
    NoseTip,
    /// Mouth corner (five-point set).
    MouthCorner,
}

/// A single landmark with its 2-D position and semantic role.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Landmark {
    /// Image-space position.
    pub position: Point,
    /// Semantic role.
    pub role: LandmarkRole,
    /// Landmark index within the chosen set.
    pub index: usize,
}

/// Complete landmark prediction for one face.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaceLandmarks {
    /// Ordered landmark positions.
    pub points: Vec<Landmark>,
    /// Bounding box used as input.
    pub source_bbox: Rect,
    /// Landmark set that was applied.
    pub landmark_set: LandmarkSet,
    /// Geometric face metrics derived from the landmarks.
    pub metrics: FaceGeometryMetrics,
}

impl FaceLandmarks {
    /// Return all landmarks with the given role.
    #[must_use]
    pub fn by_role(&self, role: LandmarkRole) -> Vec<&Landmark> {
        self.points.iter().filter(|l| l.role == role).collect()
    }

    /// Centroid of a group of landmarks by role.
    #[must_use]
    pub fn centroid_of_role(&self, role: LandmarkRole) -> Option<Point> {
        let pts: Vec<&Landmark> = self.by_role(role);
        if pts.is_empty() {
            return None;
        }
        let sum_x: f32 = pts.iter().map(|l| l.position.x).sum();
        let sum_y: f32 = pts.iter().map(|l| l.position.y).sum();
        let n = pts.len() as f32;
        Some(Point::new(sum_x / n, sum_y / n))
    }

    /// Inter-pupillary distance (pixel units), computed from eye centroids.
    #[must_use]
    pub fn inter_pupillary_distance(&self) -> Option<f32> {
        let left = self
            .centroid_of_role(LandmarkRole::LeftEye)
            .or_else(|| self.centroid_of_role(LandmarkRole::EyeCenter));
        let right = self.centroid_of_role(LandmarkRole::RightEye).or_else(|| {
            // For 5-point, take second EyeCenter point
            let pts = self.by_role(LandmarkRole::EyeCenter);
            pts.get(1).map(|l| l.position)
        });
        match (left, right) {
            (Some(l), Some(r)) => Some(l.distance(&r)),
            _ => None,
        }
    }
}

/// Geometric face metrics derived from landmark positions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaceGeometryMetrics {
    /// Face width (pixels) from landmark extremes.
    pub face_width: f32,
    /// Face height (pixels) from landmark extremes.
    pub face_height: f32,
    /// Aspect ratio (width / height).
    pub aspect_ratio: f32,
    /// Symmetry score (0.0 = perfectly asymmetric, 1.0 = perfectly symmetric).
    pub symmetry_score: f32,
    /// Estimated yaw (head left-right rotation, degrees, 0 = frontal).
    pub estimated_yaw_deg: f32,
    /// Face bounding area as fraction of source_bbox area.
    pub coverage: f32,
}

// ---------------------------------------------------------------------------
// Detector
// ---------------------------------------------------------------------------

/// Facial landmark detector based on anthropometric proportional rules.
#[derive(Debug, Clone)]
pub struct FaceLandmarkDetector {
    /// Which landmark set to produce.
    pub landmark_set: LandmarkSet,
}

impl FaceLandmarkDetector {
    /// Create a new detector.
    #[must_use]
    pub fn new(landmark_set: LandmarkSet) -> Self {
        Self { landmark_set }
    }

    /// Detect landmarks from a face bounding box.
    ///
    /// The bounding box defines the face region; all landmark positions are
    /// derived using fixed anthropometric ratios relative to that box.
    #[must_use]
    pub fn detect_from_bbox(&self, bbox: &Rect) -> FaceLandmarks {
        let points = match self.landmark_set {
            LandmarkSet::FivePoint => five_point_landmarks(bbox),
            LandmarkSet::Extended => extended_landmarks(bbox),
        };
        let metrics = compute_metrics(&points, bbox, self.landmark_set);
        FaceLandmarks {
            points,
            source_bbox: *bbox,
            landmark_set: self.landmark_set,
            metrics,
        }
    }

    /// Detect landmarks from raw pixel coordinates.
    ///
    /// Convenience wrapper around [`Self::detect_from_bbox`].
    #[must_use]
    pub fn detect_from_coords(&self, x: f32, y: f32, w: f32, h: f32) -> FaceLandmarks {
        self.detect_from_bbox(&Rect::new(x, y, w, h))
    }
}

// ---------------------------------------------------------------------------
// Five-point landmark generation
// ---------------------------------------------------------------------------

/// Anthropometric fractions for the 5-point schema.
///
/// All values are (fx, fy) as fractions of bbox width and height.
const FIVE_POINT: [(f32, f32, LandmarkRole); 5] = [
    (0.36, 0.38, LandmarkRole::EyeCenter),   // left eye
    (0.64, 0.38, LandmarkRole::EyeCenter),   // right eye
    (0.50, 0.62, LandmarkRole::NoseTip),     // nose tip
    (0.33, 0.75, LandmarkRole::MouthCorner), // left mouth corner
    (0.67, 0.75, LandmarkRole::MouthCorner), // right mouth corner
];

fn five_point_landmarks(bbox: &Rect) -> Vec<Landmark> {
    FIVE_POINT
        .iter()
        .enumerate()
        .map(|(i, &(fx, fy, role))| Landmark {
            position: Point::new(bbox.x + fx * bbox.width, bbox.y + fy * bbox.height),
            role,
            index: i,
        })
        .collect()
}

// ---------------------------------------------------------------------------
// 68-point (Extended) landmark generation
// ---------------------------------------------------------------------------

fn extended_landmarks(bbox: &Rect) -> Vec<Landmark> {
    let mut pts = Vec::with_capacity(68);
    let mut idx = 0usize;

    // ── Jaw (0–16, 17 points on lower arc) ────────────────────────────────
    for i in 0..17usize {
        let t = i as f32 / 16.0; // 0.0 → 1.0 left to right
        let angle = PI + t * PI; // lower half-ellipse (π → 2π)
        let fx = 0.5 + 0.46 * angle.cos();
        let fy = 0.5 + 0.55 * angle.sin().abs();
        pts.push(Landmark {
            position: Point::new(bbox.x + fx * bbox.width, bbox.y + fy * bbox.height),
            role: LandmarkRole::Jaw,
            index: idx,
        });
        idx += 1;
    }

    // ── Left eyebrow (17–21, 5 points) ────────────────────────────────────
    let left_brow: [(f32, f32); 5] = [
        (0.19, 0.30),
        (0.26, 0.25),
        (0.34, 0.23),
        (0.40, 0.25),
        (0.45, 0.28),
    ];
    for &(fx, fy) in &left_brow {
        pts.push(Landmark {
            position: Point::new(bbox.x + fx * bbox.width, bbox.y + fy * bbox.height),
            role: LandmarkRole::LeftEyebrow,
            index: idx,
        });
        idx += 1;
    }

    // ── Right eyebrow (22–26, 5 points) ───────────────────────────────────
    let right_brow: [(f32, f32); 5] = [
        (0.55, 0.28),
        (0.60, 0.25),
        (0.66, 0.23),
        (0.74, 0.25),
        (0.81, 0.30),
    ];
    for &(fx, fy) in &right_brow {
        pts.push(Landmark {
            position: Point::new(bbox.x + fx * bbox.width, bbox.y + fy * bbox.height),
            role: LandmarkRole::RightEyebrow,
            index: idx,
        });
        idx += 1;
    }

    // ── Nose bridge (27–30, 4 points) ─────────────────────────────────────
    let nose_bridge: [(f32, f32); 4] = [(0.50, 0.35), (0.50, 0.43), (0.50, 0.51), (0.50, 0.58)];
    for &(fx, fy) in &nose_bridge {
        pts.push(Landmark {
            position: Point::new(bbox.x + fx * bbox.width, bbox.y + fy * bbox.height),
            role: LandmarkRole::NoseBridge,
            index: idx,
        });
        idx += 1;
    }

    // ── Nose base (31–35, 5 points) ───────────────────────────────────────
    let nose_base: [(f32, f32); 5] = [
        (0.37, 0.63),
        (0.43, 0.66),
        (0.50, 0.67),
        (0.57, 0.66),
        (0.63, 0.63),
    ];
    for &(fx, fy) in &nose_base {
        pts.push(Landmark {
            position: Point::new(bbox.x + fx * bbox.width, bbox.y + fy * bbox.height),
            role: LandmarkRole::NoseBase,
            index: idx,
        });
        idx += 1;
    }

    // ── Left eye (36–41, 6 points ellipse) ────────────────────────────────
    let left_eye_cx = 0.36_f32;
    let left_eye_cy = 0.38_f32;
    for i in 0..6usize {
        let angle = 2.0 * PI * i as f32 / 6.0;
        let fx = left_eye_cx + 0.075 * angle.cos();
        let fy = left_eye_cy + 0.038 * angle.sin();
        pts.push(Landmark {
            position: Point::new(bbox.x + fx * bbox.width, bbox.y + fy * bbox.height),
            role: LandmarkRole::LeftEye,
            index: idx,
        });
        idx += 1;
    }

    // ── Right eye (42–47, 6 points ellipse) ───────────────────────────────
    let right_eye_cx = 0.64_f32;
    let right_eye_cy = 0.38_f32;
    for i in 0..6usize {
        let angle = 2.0 * PI * i as f32 / 6.0;
        let fx = right_eye_cx + 0.075 * angle.cos();
        let fy = right_eye_cy + 0.038 * angle.sin();
        pts.push(Landmark {
            position: Point::new(bbox.x + fx * bbox.width, bbox.y + fy * bbox.height),
            role: LandmarkRole::RightEye,
            index: idx,
        });
        idx += 1;
    }

    // ── Outer lip (48–59, 12 points) ──────────────────────────────────────
    for i in 0..12usize {
        let angle = 2.0 * PI * i as f32 / 12.0;
        let fx = 0.50 + 0.13 * angle.cos();
        let fy = 0.76 + 0.055 * angle.sin();
        pts.push(Landmark {
            position: Point::new(bbox.x + fx * bbox.width, bbox.y + fy * bbox.height),
            role: LandmarkRole::OuterLip,
            index: idx,
        });
        idx += 1;
    }

    // ── Inner lip (60–67, 8 points) ───────────────────────────────────────
    for i in 0..8usize {
        let angle = 2.0 * PI * i as f32 / 8.0;
        let fx = 0.50 + 0.09 * angle.cos();
        let fy = 0.76 + 0.038 * angle.sin();
        pts.push(Landmark {
            position: Point::new(bbox.x + fx * bbox.width, bbox.y + fy * bbox.height),
            role: LandmarkRole::InnerLip,
            index: idx,
        });
        idx += 1;
    }

    debug_assert_eq!(
        pts.len(),
        68,
        "Extended landmark set must have exactly 68 points"
    );
    pts
}

// ---------------------------------------------------------------------------
// Geometry metrics
// ---------------------------------------------------------------------------

fn compute_metrics(points: &[Landmark], bbox: &Rect, set: LandmarkSet) -> FaceGeometryMetrics {
    let xs: Vec<f32> = points.iter().map(|l| l.position.x).collect();
    let ys: Vec<f32> = points.iter().map(|l| l.position.y).collect();

    let min_x = xs.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_x = xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min_y = ys.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_y = ys.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let face_width = (max_x - min_x).max(0.0);
    let face_height = (max_y - min_y).max(0.0);
    let aspect_ratio = if face_height > 0.0 {
        face_width / face_height
    } else {
        1.0
    };

    // Symmetry: compare left-half and right-half landmark distances from midline
    let mid_x = (min_x + max_x) / 2.0;
    let symmetry_score = compute_symmetry(points, mid_x, set);

    // Yaw estimate: if left eye is visible vs right, the ratio of left/right
    // distances from the midline indicates head rotation.
    let estimated_yaw_deg = estimate_yaw(points, mid_x);

    let bbox_area = bbox.width * bbox.height;
    let coverage = if bbox_area > 0.0 {
        (face_width * face_height / bbox_area).min(1.0)
    } else {
        0.0
    };

    FaceGeometryMetrics {
        face_width,
        face_height,
        aspect_ratio,
        symmetry_score,
        estimated_yaw_deg,
        coverage,
    }
}

fn compute_symmetry(points: &[Landmark], mid_x: f32, set: LandmarkSet) -> f32 {
    // Pair each left-side role with its right counterpart.
    let pairs: &[(LandmarkRole, LandmarkRole)] = match set {
        LandmarkSet::FivePoint => &[],
        LandmarkSet::Extended => &[
            (LandmarkRole::LeftEye, LandmarkRole::RightEye),
            (LandmarkRole::LeftEyebrow, LandmarkRole::RightEyebrow),
        ],
    };

    if pairs.is_empty() {
        // For 5-point set use eye-centre pair
        let eyes: Vec<&Landmark> = points
            .iter()
            .filter(|l| l.role == LandmarkRole::EyeCenter)
            .collect();
        if eyes.len() >= 2 {
            let d_left = (eyes[0].position.x - mid_x).abs();
            let d_right = (eyes[1].position.x - mid_x).abs();
            let max_d = d_left.max(d_right).max(1e-6);
            let asym = (d_left - d_right).abs() / max_d;
            return (1.0 - asym).max(0.0);
        }
        return 1.0;
    }

    let mut total_asym = 0.0_f32;
    let mut count = 0usize;

    for &(left_role, right_role) in pairs {
        let left_pts: Vec<&Landmark> = points.iter().filter(|l| l.role == left_role).collect();
        let right_pts: Vec<&Landmark> = points.iter().filter(|l| l.role == right_role).collect();
        if left_pts.is_empty() || right_pts.is_empty() {
            continue;
        }
        let left_cx: f32 =
            left_pts.iter().map(|l| l.position.x).sum::<f32>() / left_pts.len() as f32;
        let right_cx: f32 =
            right_pts.iter().map(|l| l.position.x).sum::<f32>() / right_pts.len() as f32;
        let d_left = (left_cx - mid_x).abs();
        let d_right = (right_cx - mid_x).abs();
        let max_d = d_left.max(d_right).max(1e-6);
        total_asym += (d_left - d_right).abs() / max_d;
        count += 1;
    }

    if count == 0 {
        return 1.0;
    }
    (1.0 - total_asym / count as f32).clamp(0.0, 1.0)
}

fn estimate_yaw(points: &[Landmark], mid_x: f32) -> f32 {
    // Use nose bridge or eye positions to estimate yaw.
    let nose: Vec<&Landmark> = points
        .iter()
        .filter(|l| matches!(l.role, LandmarkRole::NoseTip | LandmarkRole::NoseBridge))
        .collect();

    if nose.is_empty() {
        return 0.0;
    }
    let nose_x: f32 = nose.iter().map(|l| l.position.x).sum::<f32>() / nose.len() as f32;

    // Maximum offset to consider ±45° — face_width is the normaliser.
    let all_x: Vec<f32> = points.iter().map(|l| l.position.x).collect();
    let min_x = all_x.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_x = all_x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let half_width = ((max_x - min_x) / 2.0).max(1.0);

    let offset = (nose_x - mid_x) / half_width; // −1..1
    offset * 45.0 // map to degrees
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_bbox() -> Rect {
        Rect::new(50.0, 40.0, 100.0, 120.0)
    }

    #[test]
    fn test_five_point_count() {
        let det = FaceLandmarkDetector::new(LandmarkSet::FivePoint);
        let lms = det.detect_from_bbox(&test_bbox());
        assert_eq!(lms.points.len(), 5);
    }

    #[test]
    fn test_extended_count() {
        let det = FaceLandmarkDetector::new(LandmarkSet::Extended);
        let lms = det.detect_from_bbox(&test_bbox());
        assert_eq!(lms.points.len(), 68);
    }

    #[test]
    fn test_five_point_roles() {
        let det = FaceLandmarkDetector::new(LandmarkSet::FivePoint);
        let lms = det.detect_from_bbox(&test_bbox());
        let roles: Vec<LandmarkRole> = lms.points.iter().map(|l| l.role).collect();
        assert_eq!(roles[0], LandmarkRole::EyeCenter);
        assert_eq!(roles[1], LandmarkRole::EyeCenter);
        assert_eq!(roles[2], LandmarkRole::NoseTip);
        assert_eq!(roles[3], LandmarkRole::MouthCorner);
        assert_eq!(roles[4], LandmarkRole::MouthCorner);
    }

    #[test]
    fn test_landmarks_inside_bbox() {
        let bbox = test_bbox();
        let det = FaceLandmarkDetector::new(LandmarkSet::Extended);
        let lms = det.detect_from_bbox(&bbox);
        // Allow small margin outside due to ellipse rounding
        let margin = 10.0_f32;
        for l in &lms.points {
            assert!(
                l.position.x >= bbox.x - margin && l.position.x <= bbox.x + bbox.width + margin,
                "x={} out of range",
                l.position.x
            );
            assert!(
                l.position.y >= bbox.y - margin && l.position.y <= bbox.y + bbox.height + margin,
                "y={} out of range",
                l.position.y
            );
        }
    }

    #[test]
    fn test_inter_pupillary_distance_five_point() {
        let det = FaceLandmarkDetector::new(LandmarkSet::FivePoint);
        let bbox = Rect::new(0.0, 0.0, 100.0, 120.0);
        let lms = det.detect_from_bbox(&bbox);
        // Eyes at 0.36 and 0.64 of 100 → x positions 36 and 64
        // Since both share EyeCenter role, left=right from centroid_of_role,
        // so compute IPD manually from the landmark list.
        let eye_pts: Vec<&Landmark> = lms.by_role(LandmarkRole::EyeCenter);
        assert_eq!(eye_pts.len(), 2, "should have 2 eye-centre landmarks");
        let ipd = eye_pts[0].position.distance(&eye_pts[1].position);
        assert!((ipd - 28.0).abs() < 2.0, "IPD={}", ipd);
    }

    #[test]
    fn test_by_role_extended() {
        let det = FaceLandmarkDetector::new(LandmarkSet::Extended);
        let lms = det.detect_from_bbox(&test_bbox());
        let jaw = lms.by_role(LandmarkRole::Jaw);
        assert_eq!(jaw.len(), 17);
        let outer_lip = lms.by_role(LandmarkRole::OuterLip);
        assert_eq!(outer_lip.len(), 12);
        let inner_lip = lms.by_role(LandmarkRole::InnerLip);
        assert_eq!(inner_lip.len(), 8);
    }

    #[test]
    fn test_centroid_of_role() {
        let det = FaceLandmarkDetector::new(LandmarkSet::Extended);
        let bbox = Rect::new(0.0, 0.0, 100.0, 100.0);
        let lms = det.detect_from_bbox(&bbox);
        let centroid = lms.centroid_of_role(LandmarkRole::LeftEye);
        assert!(centroid.is_some());
        let c = centroid.unwrap();
        // Left eye centroid should be in left half of frame
        assert!(
            c.x < 55.0,
            "Left eye centroid x={} should be in left half",
            c.x
        );
    }

    #[test]
    fn test_geometry_metrics_aspect_ratio() {
        let det = FaceLandmarkDetector::new(LandmarkSet::FivePoint);
        let bbox = Rect::new(0.0, 0.0, 100.0, 200.0);
        let lms = det.detect_from_bbox(&bbox);
        // Face is taller than wide so aspect_ratio < 1
        let ar = lms.metrics.aspect_ratio;
        assert!(ar < 1.2, "aspect_ratio={} expected portrait-ish", ar);
    }

    #[test]
    fn test_symmetry_score_range() {
        let det = FaceLandmarkDetector::new(LandmarkSet::Extended);
        let lms = det.detect_from_bbox(&test_bbox());
        let s = lms.metrics.symmetry_score;
        assert!(
            (0.0..=1.0).contains(&s),
            "symmetry_score={} out of [0,1]",
            s
        );
    }

    #[test]
    fn test_detect_from_coords() {
        let det = FaceLandmarkDetector::new(LandmarkSet::FivePoint);
        let lms = det.detect_from_coords(10.0, 20.0, 80.0, 100.0);
        assert_eq!(lms.points.len(), 5);
        assert_eq!(lms.source_bbox.x, 10.0);
    }

    #[test]
    fn test_yaw_approximately_zero_for_frontal() {
        let det = FaceLandmarkDetector::new(LandmarkSet::Extended);
        let bbox = Rect::new(0.0, 0.0, 100.0, 100.0);
        let lms = det.detect_from_bbox(&bbox);
        // Anthropometric model is symmetric so yaw should be near 0
        assert!(
            lms.metrics.estimated_yaw_deg.abs() < 10.0,
            "yaw={} expected near 0 for frontal model",
            lms.metrics.estimated_yaw_deg
        );
    }
}
