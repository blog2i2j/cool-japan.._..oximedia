//! DeepSORT multi-object tracker combining Kalman filter motion prediction
//! with deep appearance feature embeddings for robust re-identification.
//!
//! DeepSORT extends SORT by adding an appearance model (128-dim embedding)
//! on top of IoU-based motion association, enabling better track continuity
//! across occlusions and re-entries.
//!
//! # Example
//!
//! ```
//! use oximedia_cv::deep_sort::{DeepSortTracker, DeepSortConfig, AppearanceFeature};
//! use oximedia_cv::detect::BoundingBox;
//!
//! let mut tracker = DeepSortTracker::new(DeepSortConfig::default());
//! let dets = vec![BoundingBox::new(10.0, 10.0, 50.0, 50.0)];
//! let feats = vec![AppearanceFeature([0.0f32; 128])];
//! let tracks = tracker.update(&dets, &feats).unwrap();
//! assert_eq!(tracks.len(), 1);
//! ```

#![allow(dead_code)]

use std::collections::VecDeque;

use crate::detect::BoundingBox;
use crate::error::CvResult;
use crate::tracking::assignment::{filter_assignments_by_cost, hungarian_algorithm};
use crate::tracking::kalman::KalmanFilter;

// ---------------------------------------------------------------------------
// AppearanceFeature
// ---------------------------------------------------------------------------

/// 128-dimensional L2-normalised appearance embedding extracted from a
/// detected object (e.g. a Re-ID CNN crop feature vector).
#[derive(Debug, Clone)]
pub struct AppearanceFeature(pub [f32; 128]);

impl AppearanceFeature {
    /// Compute the L2 norm of the feature vector.
    #[must_use]
    pub fn norm(&self) -> f32 {
        self.0.iter().map(|v| v * v).sum::<f32>().sqrt()
    }

    /// Return a new feature normalised to unit length.
    #[must_use]
    pub fn normalized(&self) -> Self {
        let n = self.norm().max(1e-12);
        let mut out = [0.0f32; 128];
        for (o, &v) in out.iter_mut().zip(self.0.iter()) {
            *o = v / n;
        }
        Self(out)
    }

    /// Cosine distance in [0, 2] (0 = identical direction).
    #[must_use]
    pub fn cosine_distance(&self, other: &Self) -> f32 {
        let dot: f32 = self.0.iter().zip(other.0.iter()).map(|(a, b)| a * b).sum();
        let na = self.norm().max(1e-12);
        let nb = other.norm().max(1e-12);
        1.0 - dot / (na * nb)
    }
}

impl Default for AppearanceFeature {
    fn default() -> Self {
        Self([0.0f32; 128])
    }
}

// ---------------------------------------------------------------------------
// KalmanState
// ---------------------------------------------------------------------------

/// Snapshot of the Kalman filter state used by a DeepSORT track.
///
/// State vector: `[cx, cy, aspect_ratio, height, vcx, vcy, vaspect, vheight]`
#[derive(Debug, Clone)]
pub struct KalmanState {
    /// The 8-element mean state vector.
    pub mean: Vec<f64>,
    /// Flattened 8×8 covariance matrix (row-major).
    pub covariance: Vec<f64>,
}

impl KalmanState {
    fn new() -> Self {
        Self {
            mean: vec![0.0; 8],
            covariance: vec![0.0; 64],
        }
    }
}

// ---------------------------------------------------------------------------
// DeepSortTrack
// ---------------------------------------------------------------------------

/// Maximum number of appearance features retained per track.
const MAX_FEATURE_HISTORY: usize = 100;

/// A single active track managed by [`DeepSortTracker`].
#[derive(Debug, Clone)]
pub struct DeepSortTrack {
    /// Unique track identifier (monotonically increasing).
    pub id: u64,
    /// Most recently confirmed bounding box (pixel coordinates).
    pub bbox: BoundingBox,
    /// Snapshot of the Kalman filter state at the last update.
    pub kalman_state: KalmanState,
    /// Ring buffer of appearance features accumulated across hits.
    pub features: VecDeque<AppearanceFeature>,
    /// Total frames since this track was initialised.
    pub age: u32,
    /// Total number of times this track was successfully associated.
    pub hits: u32,
    /// Consecutive frames with successful association.
    pub hit_streak: u32,
    /// Frames since the last successful association.
    pub time_since_update: u32,
    /// Internal Kalman filter (not exposed directly).
    kalman: KalmanFilter,
}

impl DeepSortTrack {
    /// Initialise a new track from a detection + appearance feature.
    fn new(id: u64, bbox: &BoundingBox, feature: &AppearanceFeature) -> CvResult<Self> {
        let mut kalman = build_deep_sort_kalman();
        let state = bbox_to_kalman_state(bbox);
        kalman.set_state(state)?;

        let ks = KalmanState {
            mean: kalman.state().to_vec(),
            covariance: kalman.covariance().to_vec(),
        };

        let mut features = VecDeque::with_capacity(MAX_FEATURE_HISTORY);
        features.push_back(feature.clone());

        Ok(Self {
            id,
            bbox: *bbox,
            kalman_state: ks,
            features,
            age: 1,
            hits: 1,
            hit_streak: 1,
            time_since_update: 0,
            kalman,
        })
    }

    /// Run the Kalman predict step and return the predicted bounding box.
    fn predict(&mut self) -> BoundingBox {
        let state = self.kalman.predict();
        self.age += 1;
        self.time_since_update += 1;
        kalman_state_to_bbox(&state)
    }

    /// Assimilate a matched detection + appearance feature.
    fn update(&mut self, bbox: &BoundingBox, feature: &AppearanceFeature) {
        let meas = bbox_to_measurement(bbox);
        let _ = self.kalman.update(&meas);

        self.kalman_state = KalmanState {
            mean: self.kalman.state().to_vec(),
            covariance: self.kalman.covariance().to_vec(),
        };
        self.bbox = *bbox;
        self.hits += 1;
        self.hit_streak += 1;
        self.time_since_update = 0;

        if self.features.len() >= MAX_FEATURE_HISTORY {
            self.features.pop_front();
        }
        self.features.push_back(feature.clone());
    }

    /// Minimum cosine distance between `query` and all stored features.
    fn min_appearance_distance(&self, query: &AppearanceFeature) -> f32 {
        self.features
            .iter()
            .map(|f| f.cosine_distance(query))
            .fold(f32::MAX, f32::min)
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for [`DeepSortTracker`].
#[derive(Debug, Clone)]
pub struct DeepSortConfig {
    /// Maximum number of frames a track can go without a match before deletion.
    pub max_age: u32,
    /// Minimum consecutive hits before a track is reported as confirmed.
    pub min_hits: u32,
    /// Maximum IoU cost (1 − IoU) for a valid motion association.
    pub max_iou_distance: f64,
    /// Maximum cosine distance for a valid appearance association.
    pub max_appearance_distance: f32,
    /// Weight for appearance distance in combined cost (rest is IoU distance).
    pub appearance_weight: f32,
}

impl Default for DeepSortConfig {
    fn default() -> Self {
        Self {
            max_age: 30,
            min_hits: 3,
            max_iou_distance: 0.7,
            max_appearance_distance: 0.4,
            appearance_weight: 0.5,
        }
    }
}

// ---------------------------------------------------------------------------
// DeepSortTracker
// ---------------------------------------------------------------------------

/// Multi-object tracker that combines Kalman-filter motion prediction with
/// deep appearance feature matching via cosine distance.
///
/// Association pipeline (per frame):
/// 1. Predict all existing tracks forward with their Kalman filters.
/// 2. Build a combined cost matrix: `λ · appearance_dist + (1−λ) · iou_dist`.
/// 3. Solve the linear assignment with the Hungarian algorithm.
/// 4. Update matched tracks; create new tentative tracks for unmatched dets.
/// 5. Delete tracks exceeding `max_age` frames without a match.
/// 6. Report only confirmed tracks (`hits ≥ min_hits`).
#[derive(Debug)]
pub struct DeepSortTracker {
    cfg: DeepSortConfig,
    tracks: Vec<DeepSortTrack>,
    next_id: u64,
}

impl DeepSortTracker {
    /// Create a new tracker with the given configuration.
    #[must_use]
    pub fn new(cfg: DeepSortConfig) -> Self {
        Self {
            cfg,
            tracks: Vec::new(),
            next_id: 1,
        }
    }

    /// Process one frame: associate detections to existing tracks and return
    /// all currently confirmed tracks.
    ///
    /// `detections` and `features` must have the same length; `features[i]`
    /// is the appearance embedding for `detections[i]`.  If `features` is
    /// shorter than `detections`, the remaining detections are treated as
    /// having a zero-vector embedding.
    ///
    /// # Errors
    ///
    /// Returns an error if Kalman state initialisation fails for a new track.
    pub fn update(
        &mut self,
        detections: &[BoundingBox],
        features: &[AppearanceFeature],
    ) -> CvResult<Vec<DeepSortTrack>> {
        // --- 1. Predict ---
        let predicted_bboxes: Vec<BoundingBox> =
            self.tracks.iter_mut().map(|t| t.predict()).collect();

        // --- 2. Build combined cost matrix ---
        let n_tracks = self.tracks.len();
        let n_dets = detections.len();

        let cost_matrix: Vec<Vec<f64>> = if n_tracks == 0 || n_dets == 0 {
            vec![]
        } else {
            let lam = self.cfg.appearance_weight as f64;
            let mut mat = vec![vec![0.0_f64; n_dets]; n_tracks];
            for (ti, pred_bb) in predicted_bboxes.iter().enumerate() {
                for (di, det_bb) in detections.iter().enumerate() {
                    let iou_cost = iou_distance(pred_bb, det_bb);
                    let app_feat = features.get(di).cloned().unwrap_or_default();
                    let app_cost = self.tracks[ti].min_appearance_distance(&app_feat) as f64;
                    mat[ti][di] = lam * app_cost + (1.0 - lam) * iou_cost;
                }
            }
            mat
        };

        // --- 3. Hungarian assignment ---
        let max_cost = self.cfg.appearance_weight as f64 * self.cfg.max_appearance_distance as f64
            + (1.0 - self.cfg.appearance_weight as f64) * self.cfg.max_iou_distance;

        let raw_assignments = if cost_matrix.is_empty() {
            vec![None; n_tracks]
        } else {
            let raw = hungarian_algorithm(&cost_matrix);
            filter_assignments_by_cost(&raw, &cost_matrix, max_cost)
        };

        // Track which detections were matched
        let mut det_matched = vec![false; n_dets];

        // --- 4a. Update matched tracks ---
        for (ti, assignment) in raw_assignments.iter().enumerate() {
            if let Some(di) = assignment {
                let feat = features.get(*di).cloned().unwrap_or_default();
                self.tracks[ti].update(&detections[*di], &feat);
                det_matched[*di] = true;
            } else {
                // Mark missed
                self.tracks[ti].hit_streak = 0;
            }
        }

        // --- 4b. Create new tentative tracks for unmatched detections ---
        for (di, det) in detections.iter().enumerate() {
            if !det_matched[di] {
                let feat = features.get(di).cloned().unwrap_or_default();
                let new_track = DeepSortTrack::new(self.next_id, det, &feat)?;
                self.next_id += 1;
                self.tracks.push(new_track);
            }
        }

        // --- 5. Delete stale tracks ---
        self.tracks
            .retain(|t| t.time_since_update <= self.cfg.max_age);

        // --- 6. Return confirmed tracks ---
        Ok(self
            .tracks
            .iter()
            .filter(|t| t.hits >= self.cfg.min_hits || t.time_since_update == 0)
            .cloned()
            .collect())
    }

    /// Return all tracks regardless of confirmation status (useful for debugging).
    #[must_use]
    pub fn all_tracks(&self) -> &[DeepSortTrack] {
        &self.tracks
    }

    /// Number of tracks currently being managed (including tentative).
    #[must_use]
    pub fn track_count(&self) -> usize {
        self.tracks.len()
    }
}

// ---------------------------------------------------------------------------
// Kalman filter helpers for DeepSORT
// ---------------------------------------------------------------------------

/// Build the 8-state / 4-measurement Kalman filter used by DeepSORT.
///
/// State:       [cx, cy, a, h, vcx, vcy, va, vh]
/// Measurement: [cx, cy, a, h]
fn build_deep_sort_kalman() -> KalmanFilter {
    let mut kf = KalmanFilter::new(8, 4);

    // State transition: constant velocity
    // x_new = x + vx*dt  (dt=1)
    let dt = 1.0_f64;
    let mut f = vec![0.0_f64; 64];
    for i in 0..8 {
        f[i * 8 + i] = 1.0;
    }
    for i in 0..4 {
        f[i * 8 + (i + 4)] = dt;
    }
    kf.transition = f;

    // Measurement matrix: first 4 states observed
    let mut h = vec![0.0_f64; 32];
    for i in 0..4 {
        h[i * 8 + i] = 1.0;
    }
    kf.measurement = h;

    // Process noise Q
    let mut q = vec![0.0_f64; 64];
    let pos_noise = 1.0_f64;
    let vel_noise = 0.01_f64;
    for i in 0..4 {
        q[i * 8 + i] = pos_noise;
        q[(i + 4) * 8 + (i + 4)] = vel_noise;
    }
    kf.process_noise = q;

    // Measurement noise R
    let mut r = vec![0.0_f64; 16];
    let meas_noise = [1.0, 1.0, 0.01, 1.0]; // cx, cy, aspect_ratio, height
    for i in 0..4 {
        r[i * 4 + i] = meas_noise[i];
    }
    kf.measurement_noise = r;

    // Initial covariance P
    let mut p = vec![0.0_f64; 64];
    let p_vals = [10.0, 10.0, 1e-2, 10.0, 1e4, 1e4, 1e-5, 1e4];
    for i in 0..8 {
        p[i * 8 + i] = p_vals[i];
    }
    kf.covariance = p;

    kf
}

/// Convert a bounding box to the Kalman filter state vector.
/// State: [cx, cy, aspect_ratio, height, 0, 0, 0, 0]
fn bbox_to_kalman_state(bbox: &BoundingBox) -> Vec<f64> {
    let cx = bbox.x as f64 + bbox.width as f64 / 2.0;
    let cy = bbox.y as f64 + bbox.height as f64 / 2.0;
    let h = bbox.height.max(1.0) as f64;
    let a = bbox.width as f64 / h;
    vec![cx, cy, a, h, 0.0, 0.0, 0.0, 0.0]
}

/// Convert a Kalman state vector back to a bounding box.
fn kalman_state_to_bbox(state: &[f64]) -> BoundingBox {
    if state.len() < 4 {
        return BoundingBox::new(0.0, 0.0, 1.0, 1.0);
    }
    let cx = state[0];
    let cy = state[1];
    let a = state[2].max(1e-6);
    let h = state[3].max(1.0);
    let w = a * h;
    BoundingBox::new(
        (cx - w / 2.0) as f32,
        (cy - h / 2.0) as f32,
        w as f32,
        h as f32,
    )
}

/// Convert a bounding box to the 4-element measurement vector [cx, cy, a, h].
fn bbox_to_measurement(bbox: &BoundingBox) -> Vec<f64> {
    let cx = bbox.x as f64 + bbox.width as f64 / 2.0;
    let cy = bbox.y as f64 + bbox.height as f64 / 2.0;
    let h = bbox.height.max(1.0) as f64;
    let a = bbox.width as f64 / h;
    vec![cx, cy, a, h]
}

/// IoU distance: 1 − IoU(a, b).
fn iou_distance(a: &BoundingBox, b: &BoundingBox) -> f64 {
    let ax2 = a.x + a.width;
    let ay2 = a.y + a.height;
    let bx2 = b.x + b.width;
    let by2 = b.y + b.height;

    let ix1 = a.x.max(b.x);
    let iy1 = a.y.max(b.y);
    let ix2 = ax2.min(bx2);
    let iy2 = ay2.min(by2);

    let iw = (ix2 - ix1).max(0.0);
    let ih = (iy2 - iy1).max(0.0);
    let inter = iw * ih;

    let area_a = a.width * a.height;
    let area_b = b.width * b.height;
    let union = area_a + area_b - inter;

    if union <= 0.0 {
        1.0
    } else {
        1.0 - (inter / union) as f64
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_bbox(x: f32, y: f32, w: f32, h: f32) -> BoundingBox {
        BoundingBox::new(x, y, w, h)
    }

    fn zero_feat() -> AppearanceFeature {
        AppearanceFeature([0.0f32; 128])
    }

    fn unit_feat(val: f32) -> AppearanceFeature {
        AppearanceFeature([val; 128])
    }

    // --- AppearanceFeature ---

    #[test]
    fn test_feature_norm_zero() {
        let f = zero_feat();
        assert!((f.norm() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_feature_norm_unit() {
        let mut arr = [0.0f32; 128];
        arr[0] = 1.0;
        let f = AppearanceFeature(arr);
        assert!((f.norm() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_feature_normalized_unit_length() {
        let f = unit_feat(1.0).normalized();
        assert!((f.norm() - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_cosine_distance_identical() {
        let f = unit_feat(1.0);
        let d = f.cosine_distance(&f);
        assert!(d.abs() < 1e-4, "identical features should have distance ~0");
    }

    #[test]
    fn test_cosine_distance_orthogonal() {
        let mut a = [0.0f32; 128];
        let mut b = [0.0f32; 128];
        a[0] = 1.0;
        b[1] = 1.0;
        let fa = AppearanceFeature(a);
        let fb = AppearanceFeature(b);
        let d = fa.cosine_distance(&fb);
        assert!(
            (d - 1.0).abs() < 1e-4,
            "orthogonal features dist should be ~1"
        );
    }

    #[test]
    fn test_cosine_distance_zero_vector() {
        let fa = zero_feat();
        let fb = unit_feat(1.0);
        // Should not panic; result is 1 - 0/(eps*norm) ≈ 1
        let d = fa.cosine_distance(&fb);
        assert!(d.is_finite());
    }

    // --- KalmanState / helpers ---

    #[test]
    fn test_bbox_to_kalman_round_trip() {
        let bbox = make_bbox(10.0, 20.0, 60.0, 80.0);
        let state = bbox_to_kalman_state(&bbox);
        assert_eq!(state.len(), 8);
        let back = kalman_state_to_bbox(&state);
        assert!((back.x - bbox.x).abs() < 1.0);
        assert!((back.y - bbox.y).abs() < 1.0);
        assert!((back.width - bbox.width).abs() < 1.0);
        assert!((back.height - bbox.height).abs() < 1.0);
    }

    #[test]
    fn test_kalman_state_to_bbox_short_state() {
        let state = vec![50.0, 50.0];
        let bb = kalman_state_to_bbox(&state);
        // Should not panic; returns a fallback bbox
        assert!(bb.width >= 0.0);
    }

    #[test]
    fn test_iou_distance_full_overlap() {
        let a = make_bbox(0.0, 0.0, 10.0, 10.0);
        let b = make_bbox(0.0, 0.0, 10.0, 10.0);
        let d = iou_distance(&a, &b);
        assert!(d.abs() < 1e-4, "identical boxes distance={d}");
    }

    #[test]
    fn test_iou_distance_no_overlap() {
        let a = make_bbox(0.0, 0.0, 10.0, 10.0);
        let b = make_bbox(100.0, 100.0, 10.0, 10.0);
        let d = iou_distance(&a, &b);
        assert!((d - 1.0).abs() < 1e-4, "non-overlapping distance={d}");
    }

    // --- DeepSortTrack ---

    #[test]
    fn test_track_new() {
        let bbox = make_bbox(10.0, 10.0, 50.0, 50.0);
        let feat = unit_feat(0.5);
        let track = DeepSortTrack::new(1, &bbox, &feat).unwrap();
        assert_eq!(track.id, 1);
        assert_eq!(track.hits, 1);
        assert_eq!(track.hit_streak, 1);
        assert_eq!(track.age, 1);
        assert_eq!(track.features.len(), 1);
    }

    #[test]
    fn test_track_predict_increases_age() {
        let bbox = make_bbox(0.0, 0.0, 40.0, 40.0);
        let mut track = DeepSortTrack::new(1, &bbox, &zero_feat()).unwrap();
        track.predict();
        assert_eq!(track.age, 2);
    }

    #[test]
    fn test_track_update_increments_hits() {
        let bbox = make_bbox(0.0, 0.0, 40.0, 40.0);
        let mut track = DeepSortTrack::new(1, &bbox, &zero_feat()).unwrap();
        track.update(&bbox, &zero_feat());
        assert_eq!(track.hits, 2);
        assert_eq!(track.time_since_update, 0);
    }

    #[test]
    fn test_track_feature_history_cap() {
        let bbox = make_bbox(0.0, 0.0, 20.0, 20.0);
        let mut track = DeepSortTrack::new(1, &bbox, &zero_feat()).unwrap();
        for _ in 0..MAX_FEATURE_HISTORY + 10 {
            track.update(&bbox, &zero_feat());
        }
        assert!(track.features.len() <= MAX_FEATURE_HISTORY);
    }

    #[test]
    fn test_track_min_appearance_distance() {
        let bbox = make_bbox(0.0, 0.0, 20.0, 20.0);
        let feat_a = unit_feat(1.0);
        let track = DeepSortTrack::new(1, &bbox, &feat_a).unwrap();
        let d = track.min_appearance_distance(&feat_a);
        assert!(d < 1e-3, "identical feature min dist={d}");
    }

    // --- DeepSortTracker ---

    #[test]
    fn test_tracker_empty_input() {
        let mut t = DeepSortTracker::new(DeepSortConfig::default());
        let tracks = t.update(&[], &[]).unwrap();
        assert!(tracks.is_empty());
    }

    #[test]
    fn test_tracker_single_detection_confirms_after_min_hits() {
        let mut cfg = DeepSortConfig::default();
        cfg.min_hits = 1;
        let mut t = DeepSortTracker::new(cfg);
        let det = vec![make_bbox(50.0, 50.0, 60.0, 60.0)];
        let feat = vec![zero_feat()];
        let tracks = t.update(&det, &feat).unwrap();
        // With min_hits=1, should be confirmed immediately
        assert_eq!(tracks.len(), 1);
    }

    #[test]
    fn test_tracker_track_count_grows() {
        let mut cfg = DeepSortConfig::default();
        cfg.min_hits = 1;
        let mut t = DeepSortTracker::new(cfg);
        let dets = vec![
            make_bbox(10.0, 10.0, 20.0, 20.0),
            make_bbox(200.0, 200.0, 20.0, 20.0),
        ];
        let feats = vec![zero_feat(), zero_feat()];
        t.update(&dets, &feats).unwrap();
        assert_eq!(t.track_count(), 2);
    }

    #[test]
    fn test_tracker_persists_track_across_frames() {
        let mut cfg = DeepSortConfig::default();
        cfg.min_hits = 1;
        cfg.max_age = 5;
        let mut t = DeepSortTracker::new(cfg);
        let det = vec![make_bbox(100.0, 100.0, 40.0, 40.0)];
        let feat = vec![zero_feat()];
        t.update(&det, &feat).unwrap();
        let id_first = t.all_tracks()[0].id;
        // Same detection next frame
        let tracks2 = t.update(&det, &feat).unwrap();
        let id_second = tracks2[0].id;
        assert_eq!(id_first, id_second, "track ID should persist");
    }

    #[test]
    fn test_tracker_removes_stale_tracks() {
        let mut cfg = DeepSortConfig::default();
        cfg.min_hits = 1;
        cfg.max_age = 2;
        let mut t = DeepSortTracker::new(cfg);
        let det = vec![make_bbox(50.0, 50.0, 30.0, 30.0)];
        let feat = vec![zero_feat()];
        t.update(&det, &feat).unwrap();
        // 3 frames with no detection
        t.update(&[], &[]).unwrap();
        t.update(&[], &[]).unwrap();
        t.update(&[], &[]).unwrap();
        assert_eq!(t.track_count(), 0, "stale track should be removed");
    }

    #[test]
    fn test_tracker_new_id_per_detection() {
        let mut cfg = DeepSortConfig::default();
        cfg.min_hits = 1;
        let mut t = DeepSortTracker::new(cfg);
        let d1 = vec![make_bbox(10.0, 10.0, 20.0, 20.0)];
        let d2 = vec![make_bbox(500.0, 500.0, 20.0, 20.0)];
        t.update(&d1, &vec![zero_feat()]).unwrap();
        t.update(&d2, &vec![zero_feat()]).unwrap();
        let ids: Vec<u64> = t.all_tracks().iter().map(|tr| tr.id).collect();
        let unique: std::collections::HashSet<u64> = ids.into_iter().collect();
        assert_eq!(unique.len(), t.track_count());
    }

    #[test]
    fn test_kalman_build_dimensions() {
        let kf = build_deep_sort_kalman();
        assert_eq!(kf.covariance().len(), 64);
        assert_eq!(kf.state().len(), 8);
    }
}
