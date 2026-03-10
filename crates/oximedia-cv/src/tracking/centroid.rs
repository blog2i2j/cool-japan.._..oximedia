//! Centroid tracker for multi-object tracking.
//!
//! Simple tracker that associates objects based on centroid distance.
//! Useful for tracking well-separated objects with relatively linear motion.
//!
//! # Example
//!
//! ```
//! use oximedia_cv::tracking::centroid::CentroidTracker;
//! use oximedia_cv::detect::BoundingBox;
//!
//! let mut tracker = CentroidTracker::new();
//! let detections = vec![BoundingBox::new(50.0, 50.0, 100.0, 100.0)];
//! let tracks = tracker.update(&detections);
//! ```

use crate::detect::BoundingBox;
use std::collections::HashMap;

/// Centroid-based track.
#[derive(Debug, Clone)]
struct CentroidTrack {
    /// Track ID.
    id: u64,
    /// Current centroid.
    centroid: (f32, f32),
    /// Current bounding box.
    bbox: BoundingBox,
    /// Frames since last update.
    disappeared: usize,
}

/// Centroid tracker for multi-object tracking.
#[derive(Debug, Clone)]
pub struct CentroidTracker {
    /// Active tracks.
    tracks: HashMap<u64, CentroidTrack>,
    /// Next track ID.
    next_id: u64,
    /// Maximum disappeared frames before deletion.
    max_disappeared: usize,
    /// Maximum distance for association.
    max_distance: f64,
}

impl CentroidTracker {
    /// Create a new centroid tracker.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::tracking::centroid::CentroidTracker;
    ///
    /// let tracker = CentroidTracker::new();
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            tracks: HashMap::new(),
            next_id: 1,
            max_disappeared: 50,
            max_distance: 100.0,
        }
    }

    /// Set maximum disappeared frames.
    #[must_use]
    pub const fn with_max_disappeared(mut self, frames: usize) -> Self {
        self.max_disappeared = frames;
        self
    }

    /// Set maximum distance for association.
    #[must_use]
    pub const fn with_max_distance(mut self, distance: f64) -> Self {
        self.max_distance = distance;
        self
    }

    /// Update tracker with new detections.
    ///
    /// # Arguments
    ///
    /// * `detections` - Detected bounding boxes
    ///
    /// # Returns
    ///
    /// Vector of active tracks (ID, bbox, confidence).
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::tracking::centroid::CentroidTracker;
    /// use oximedia_cv::detect::BoundingBox;
    ///
    /// let mut tracker = CentroidTracker::new();
    /// let detections = vec![BoundingBox::new(100.0, 100.0, 50.0, 50.0)];
    /// let tracks = tracker.update(&detections);
    /// ```
    pub fn update(&mut self, detections: &[BoundingBox]) -> Vec<(u64, BoundingBox, f64)> {
        // If no detections, mark all as disappeared
        if detections.is_empty() {
            for track in self.tracks.values_mut() {
                track.disappeared += 1;
            }

            // Remove disappeared tracks
            self.tracks
                .retain(|_, track| track.disappeared < self.max_disappeared);

            return self.get_active_tracks();
        }

        // Compute centroids for detections
        let detection_centroids: Vec<(f32, f32)> = detections
            .iter()
            .map(|bbox| (bbox.x + bbox.width / 2.0, bbox.y + bbox.height / 2.0))
            .collect();

        // If no existing tracks, create new ones
        if self.tracks.is_empty() {
            for (i, bbox) in detections.iter().enumerate() {
                self.register(detection_centroids[i], *bbox);
            }

            return self.get_active_tracks();
        }

        // Get existing track centroids
        let track_ids: Vec<u64> = self.tracks.keys().copied().collect();
        let track_centroids: Vec<(f32, f32)> = track_ids
            .iter()
            .map(|id| self.tracks[id].centroid)
            .collect();

        // Compute distance matrix
        let distances = compute_distance_matrix(&track_centroids, &detection_centroids);

        // Find optimal assignment (greedy nearest neighbor)
        let (matched_tracks, matched_detections) = self.greedy_assignment(&distances, &track_ids);

        // Update matched tracks
        for (track_id, det_idx) in matched_tracks.iter().zip(matched_detections.iter()) {
            if let Some(track) = self.tracks.get_mut(track_id) {
                track.centroid = detection_centroids[*det_idx];
                track.bbox = detections[*det_idx];
                track.disappeared = 0;
            }
        }

        // Mark unmatched tracks as disappeared
        for track_id in &track_ids {
            if !matched_tracks.contains(track_id) {
                if let Some(track) = self.tracks.get_mut(track_id) {
                    track.disappeared += 1;
                }
            }
        }

        // Register new tracks for unmatched detections
        let mut detection_matched = vec![false; detections.len()];
        for &det_idx in &matched_detections {
            detection_matched[det_idx] = true;
        }

        for (i, &matched) in detection_matched.iter().enumerate() {
            if !matched {
                self.register(detection_centroids[i], detections[i]);
            }
        }

        // Remove disappeared tracks
        self.tracks
            .retain(|_, track| track.disappeared < self.max_disappeared);

        self.get_active_tracks()
    }

    /// Register a new track.
    fn register(&mut self, centroid: (f32, f32), bbox: BoundingBox) {
        let track = CentroidTrack {
            id: self.next_id,
            centroid,
            bbox,
            disappeared: 0,
        };

        self.tracks.insert(self.next_id, track);
        self.next_id += 1;
    }

    /// Greedy assignment based on distance.
    fn greedy_assignment(
        &self,
        distances: &[Vec<f64>],
        track_ids: &[u64],
    ) -> (Vec<u64>, Vec<usize>) {
        let mut matched_tracks = Vec::new();
        let mut matched_detections = Vec::new();

        if distances.is_empty() || distances[0].is_empty() {
            return (matched_tracks, matched_detections);
        }

        let n_tracks = distances.len();
        let n_detections = distances[0].len();

        let mut track_used = vec![false; n_tracks];
        let mut detection_used = vec![false; n_detections];

        // Create sorted list of (distance, track_idx, det_idx)
        let mut pairs = Vec::new();
        for (i, row) in distances.iter().enumerate() {
            for (j, &dist) in row.iter().enumerate() {
                if dist <= self.max_distance {
                    pairs.push((dist, i, j));
                }
            }
        }

        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Assign greedily
        for (_, track_idx, det_idx) in pairs {
            if !track_used[track_idx] && !detection_used[det_idx] {
                matched_tracks.push(track_ids[track_idx]);
                matched_detections.push(det_idx);
                track_used[track_idx] = true;
                detection_used[det_idx] = true;
            }
        }

        (matched_tracks, matched_detections)
    }

    /// Get active tracks.
    fn get_active_tracks(&self) -> Vec<(u64, BoundingBox, f64)> {
        self.tracks
            .values()
            .map(|track| {
                let confidence = 1.0 / (1.0 + track.disappeared as f64);
                (track.id, track.bbox, confidence)
            })
            .collect()
    }

    /// Get all tracks (including disappeared).
    pub fn get_all_tracks(&self) -> Vec<(u64, BoundingBox, usize)> {
        self.tracks
            .values()
            .map(|track| (track.id, track.bbox, track.disappeared))
            .collect()
    }

    /// Reset tracker.
    pub fn reset(&mut self) {
        self.tracks.clear();
        self.next_id = 1;
    }

    /// Get number of active tracks.
    pub fn num_tracks(&self) -> usize {
        self.tracks.len()
    }
}

impl Default for CentroidTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute Euclidean distance matrix between two sets of centroids.
fn compute_distance_matrix(centroids1: &[(f32, f32)], centroids2: &[(f32, f32)]) -> Vec<Vec<f64>> {
    let mut distances = vec![vec![0.0; centroids2.len()]; centroids1.len()];

    for (i, &(x1, y1)) in centroids1.iter().enumerate() {
        for (j, &(x2, y2)) in centroids2.iter().enumerate() {
            let dx = (x1 - x2) as f64;
            let dy = (y1 - y2) as f64;
            distances[i][j] = (dx * dx + dy * dy).sqrt();
        }
    }

    distances
}
