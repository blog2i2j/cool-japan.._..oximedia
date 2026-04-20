//! SORT (Simple Online and Realtime Tracking) multi-object tracker.
//!
//! SORT uses Kalman filters for motion prediction and Hungarian algorithm
//! for data association based on IoU (Intersection over Union).
//!
//! # Example
//!
//! ```
//! use oximedia_cv::tracking::sort::SortTracker;
//! use oximedia_cv::detect::BoundingBox;
//!
//! let mut tracker = SortTracker::new();
//! let detections = vec![BoundingBox::new(50.0, 50.0, 100.0, 100.0)];
//! let tracks = tracker.update(&detections);
//! ```

use super::assignment::{
    compute_iou, create_iou_cost_matrix, filter_assignments_by_cost, hungarian_algorithm,
};
use crate::detect::BoundingBox;
use crate::error::{CvError, CvResult};
use crate::tracking::kalman::KalmanFilter;

/// Track state for SORT.
#[derive(Debug, Clone)]
pub struct Track {
    /// Track ID.
    pub id: u64,
    /// Kalman filter for state estimation.
    kalman: KalmanFilter,
    /// Current bounding box.
    pub bbox: BoundingBox,
    /// Frames since last update.
    pub frames_since_update: usize,
    /// Total hits (successful updates).
    pub hits: usize,
    /// Hit streak.
    pub hit_streak: usize,
    /// Track age.
    pub age: usize,
    /// Track confidence.
    pub confidence: f64,
}

impl Track {
    /// Create a new track from a detection.
    fn new(id: u64, bbox: BoundingBox) -> CvResult<Self> {
        // Initialize Kalman filter with constant velocity model
        // State: [x, y, s, r, vx, vy, vs] where:
        // - (x, y) = center position
        // - s = scale (area)
        // - r = aspect ratio (width/height)
        // - (vx, vy, vs) = velocities
        let mut kalman = KalmanFilter::new(7, 4);

        // Set state transition matrix for constant velocity
        let dt = 1.0;
        kalman.transition = vec![
            1.0, 0.0, 0.0, 0.0, dt, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, dt, 0.0, 0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, dt, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];

        // Set measurement matrix (measure x, y, s, r)
        kalman.measurement = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        ];

        // Set process and measurement noise
        kalman.set_process_noise(0.01);
        kalman.set_measurement_noise(1.0);

        // Initialize state from bbox
        let cx = (bbox.x + bbox.width / 2.0) as f64;
        let cy = (bbox.y + bbox.height / 2.0) as f64;
        let s = (bbox.width * bbox.height) as f64;
        let r = (bbox.width / bbox.height) as f64;

        kalman.set_state(vec![cx, cy, s, r, 0.0, 0.0, 0.0])?;

        Ok(Self {
            id,
            kalman,
            bbox,
            frames_since_update: 0,
            hits: 1,
            hit_streak: 1,
            age: 1,
            confidence: 1.0,
        })
    }

    /// Predict next state.
    fn predict(&mut self) -> BoundingBox {
        let state = self.kalman.predict();

        // Convert state to bbox
        self.bbox = state_to_bbox(&state);
        self.age += 1;

        self.bbox
    }

    /// Update with a detection.
    fn update(&mut self, bbox: &BoundingBox) {
        self.frames_since_update = 0;
        self.hits += 1;
        self.hit_streak += 1;

        // Convert bbox to measurement
        let measurement = bbox_to_measurement(bbox);

        // Update Kalman filter
        if let Ok(state) = self.kalman.update(&measurement) {
            self.bbox = state_to_bbox(&state);
        } else {
            self.bbox = *bbox;
        }

        self.confidence = 1.0;
    }

    /// Mark as unmatched (prediction only).
    fn mark_missed(&mut self) {
        self.frames_since_update += 1;
        self.hit_streak = 0;
        self.confidence *= 0.8;
    }

    /// Get current state as bbox.
    fn get_state(&self) -> BoundingBox {
        self.bbox
    }
}

/// SORT tracker configuration.
#[derive(Debug, Clone)]
pub struct SortTracker {
    /// Active tracks.
    tracks: Vec<Track>,
    /// Next track ID.
    next_id: u64,
    /// Maximum age (frames without detection) before deletion.
    max_age: usize,
    /// Minimum hits before track is confirmed.
    min_hits: usize,
    /// IoU threshold for matching.
    iou_threshold: f64,
}

impl SortTracker {
    /// Create a new SORT tracker.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::tracking::sort::SortTracker;
    ///
    /// let tracker = SortTracker::new();
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            tracks: Vec::new(),
            next_id: 1,
            max_age: 30,
            min_hits: 3,
            iou_threshold: 0.3,
        }
    }

    /// Set maximum age before track deletion.
    #[must_use]
    pub const fn with_max_age(mut self, age: usize) -> Self {
        self.max_age = age;
        self
    }

    /// Set minimum hits for track confirmation.
    #[must_use]
    pub const fn with_min_hits(mut self, hits: usize) -> Self {
        self.min_hits = hits;
        self
    }

    /// Set IoU threshold for matching.
    #[must_use]
    pub const fn with_iou_threshold(mut self, threshold: f64) -> Self {
        self.iou_threshold = threshold;
        self
    }

    /// Update tracker with new detections.
    ///
    /// # Arguments
    ///
    /// * `detections` - Detected bounding boxes for this frame
    ///
    /// # Returns
    ///
    /// Vector of active tracks (with ID and bbox).
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::tracking::sort::SortTracker;
    /// use oximedia_cv::detect::BoundingBox;
    ///
    /// let mut tracker = SortTracker::new();
    /// let detections = vec![BoundingBox::new(100.0, 100.0, 50.0, 50.0)];
    /// let tracks = tracker.update(&detections);
    /// ```
    pub fn update(&mut self, detections: &[BoundingBox]) -> Vec<(u64, BoundingBox, f64)> {
        // Predict all tracks
        let mut predicted = Vec::new();
        for track in &mut self.tracks {
            predicted.push(track.predict());
        }

        // Match detections to tracks using Hungarian algorithm
        let (matched, unmatched_tracks, unmatched_detections) =
            self.associate_detections_to_tracks(&predicted, detections);

        // Update matched tracks
        for (track_idx, det_idx) in matched {
            self.tracks[track_idx].update(&detections[det_idx]);
        }

        // Mark unmatched tracks as missed
        for track_idx in unmatched_tracks {
            if track_idx < self.tracks.len() {
                self.tracks[track_idx].mark_missed();
            }
        }

        // Create new tracks for unmatched detections
        for det_idx in unmatched_detections {
            if let Ok(track) = Track::new(self.next_id, detections[det_idx]) {
                self.tracks.push(track);
                self.next_id += 1;
            }
        }

        // Remove dead tracks
        self.tracks
            .retain(|track| track.frames_since_update < self.max_age);

        // Return confirmed tracks
        self.tracks
            .iter()
            .filter(|track| track.hit_streak >= self.min_hits || track.frames_since_update == 0)
            .map(|track| (track.id, track.get_state(), track.confidence))
            .collect()
    }

    /// Get all active tracks.
    pub fn get_tracks(&self) -> Vec<(u64, BoundingBox, f64)> {
        self.tracks
            .iter()
            .map(|track| (track.id, track.bbox, track.confidence))
            .collect()
    }

    /// Clear all tracks.
    pub fn reset(&mut self) {
        self.tracks.clear();
        self.next_id = 1;
    }

    /// Associate detections to tracks using Hungarian algorithm.
    fn associate_detections_to_tracks(
        &self,
        tracks: &[BoundingBox],
        detections: &[BoundingBox],
    ) -> (Vec<(usize, usize)>, Vec<usize>, Vec<usize>) {
        if tracks.is_empty() {
            return (Vec::new(), Vec::new(), (0..detections.len()).collect());
        }

        if detections.is_empty() {
            return (Vec::new(), (0..tracks.len()).collect(), Vec::new());
        }

        // Create cost matrix based on IoU
        let cost_matrix = create_iou_cost_matrix(tracks, detections);

        // Solve assignment problem
        let assignments = hungarian_algorithm(&cost_matrix);

        // Filter by IoU threshold
        let max_cost = 1.0 - self.iou_threshold;
        let filtered = filter_assignments_by_cost(&assignments, &cost_matrix, max_cost);

        // Extract matched, unmatched tracks, and unmatched detections
        let mut matched = Vec::new();
        let mut unmatched_tracks = Vec::new();
        let mut detection_used = vec![false; detections.len()];

        for (track_idx, assignment) in filtered.iter().enumerate() {
            if let Some(det_idx) = assignment {
                matched.push((track_idx, *det_idx));
                detection_used[*det_idx] = true;
            } else {
                unmatched_tracks.push(track_idx);
            }
        }

        let unmatched_detections: Vec<usize> = (0..detections.len())
            .filter(|&i| !detection_used[i])
            .collect();

        (matched, unmatched_tracks, unmatched_detections)
    }
}

impl Default for SortTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// DeepSORT tracker with appearance features.
#[derive(Debug, Clone)]
pub struct DeepSortTracker {
    /// Base SORT tracker.
    sort: SortTracker,
    /// Feature history for each track.
    feature_history: Vec<Vec<Vec<f32>>>,
    /// Maximum feature history length.
    max_feature_history: usize,
    /// Feature matching threshold.
    feature_threshold: f64,
}

impl DeepSortTracker {
    /// Create a new DeepSORT tracker.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::tracking::sort::DeepSortTracker;
    ///
    /// let tracker = DeepSortTracker::new();
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            sort: SortTracker::new(),
            feature_history: Vec::new(),
            max_feature_history: 100,
            feature_threshold: 0.7,
        }
    }

    /// Update with detections and appearance features.
    ///
    /// # Arguments
    ///
    /// * `detections` - Detected bounding boxes
    /// * `features` - Appearance feature vectors for each detection
    ///
    /// # Returns
    ///
    /// Vector of active tracks.
    pub fn update_with_features(
        &mut self,
        detections: &[BoundingBox],
        features: &[Vec<f32>],
    ) -> Vec<(u64, BoundingBox, f64)> {
        // Use SORT for basic tracking

        // Update feature history for matched tracks
        // (In a full implementation, this would use the feature distance
        // in addition to IoU for data association)

        self.sort.update(detections)
    }

    /// Reset tracker.
    pub fn reset(&mut self) {
        self.sort.reset();
        self.feature_history.clear();
    }
}

impl Default for DeepSortTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert bbox to measurement vector.
fn bbox_to_measurement(bbox: &BoundingBox) -> Vec<f64> {
    let cx = (bbox.x + bbox.width / 2.0) as f64;
    let cy = (bbox.y + bbox.height / 2.0) as f64;
    let s = (bbox.width * bbox.height) as f64;
    let r = (bbox.width / bbox.height) as f64;

    vec![cx, cy, s, r]
}

/// Convert state vector to bbox.
fn state_to_bbox(state: &[f64]) -> BoundingBox {
    if state.len() < 4 {
        return BoundingBox::new(0.0, 0.0, 1.0, 1.0);
    }

    let cx = state[0];
    let cy = state[1];
    let s = state[2].max(1.0);
    let r = state[3].max(0.1);

    let w = (s * r).sqrt();
    let h = s / w;

    BoundingBox::new(
        (cx - w / 2.0) as f32,
        (cy - h / 2.0) as f32,
        w as f32,
        h as f32,
    )
}
