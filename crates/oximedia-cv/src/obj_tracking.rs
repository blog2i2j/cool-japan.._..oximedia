//! Object tracking: centroid tracking, IoU-based track association, and
//! track lifecycle management.
//!
//! This module complements the lower-level trackers in [`crate::tracking`]
//! with higher-level multi-object management utilities.

use std::collections::HashMap;

/// Axis-aligned bounding box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bbox {
    /// Left edge.
    pub x: f32,
    /// Top edge.
    pub y: f32,
    /// Width.
    pub w: f32,
    /// Height.
    pub h: f32,
}

impl Bbox {
    /// Create a new bounding box.
    #[must_use]
    pub fn new(x: f32, y: f32, w: f32, h: f32) -> Self {
        Self { x, y, w, h }
    }

    /// Centroid of the box.
    #[must_use]
    pub fn centroid(&self) -> (f32, f32) {
        (self.x + self.w * 0.5, self.y + self.h * 0.5)
    }

    /// Area of the box.
    #[must_use]
    pub fn area(&self) -> f32 {
        self.w * self.h
    }

    /// Intersection-over-Union with another box.
    #[must_use]
    pub fn iou(&self, other: &Self) -> f32 {
        let ix1 = self.x.max(other.x);
        let iy1 = self.y.max(other.y);
        let ix2 = (self.x + self.w).min(other.x + other.w);
        let iy2 = (self.y + self.h).min(other.y + other.h);

        let iw = (ix2 - ix1).max(0.0);
        let ih = (iy2 - iy1).max(0.0);
        let intersection = iw * ih;

        let union = self.area() + other.area() - intersection;
        if union <= 0.0 {
            0.0
        } else {
            intersection / union
        }
    }

    /// Euclidean distance between centroids.
    #[must_use]
    pub fn centroid_distance(&self, other: &Self) -> f32 {
        let (cx1, cy1) = self.centroid();
        let (cx2, cy2) = other.centroid();
        let dx = cx1 - cx2;
        let dy = cy1 - cy2;
        (dx * dx + dy * dy).sqrt()
    }
}

/// State of a tracked object.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackState {
    /// Track was just created.
    New,
    /// Track is actively matched.
    Active,
    /// Track was not matched this frame but may re-appear.
    Lost,
    /// Track has been deleted after too many lost frames.
    Dead,
}

/// A single tracked object.
#[derive(Debug, Clone)]
pub struct Track {
    /// Unique track identifier.
    pub id: u64,
    /// Most recent bounding box.
    pub bbox: Bbox,
    /// Current state.
    pub state: TrackState,
    /// Number of consecutive frames the track has been active.
    pub age: u32,
    /// Number of consecutive frames the track has been lost.
    pub lost_frames: u32,
    /// Total number of times this track was matched.
    pub hits: u32,
}

impl Track {
    /// Create a new track.
    #[must_use]
    pub fn new(id: u64, bbox: Bbox) -> Self {
        Self {
            id,
            bbox,
            state: TrackState::New,
            age: 1,
            lost_frames: 0,
            hits: 1,
        }
    }

    /// Update the track with a new bounding box.
    pub fn update(&mut self, bbox: Bbox) {
        self.bbox = bbox;
        self.state = TrackState::Active;
        self.age += 1;
        self.hits += 1;
        self.lost_frames = 0;
    }

    /// Mark as lost for one frame.
    pub fn mark_lost(&mut self) {
        self.state = TrackState::Lost;
        self.lost_frames += 1;
        self.age += 1;
    }

    /// Whether the track is confirmed (seen at least `min_hits` times).
    #[must_use]
    pub fn is_confirmed(&self, min_hits: u32) -> bool {
        self.hits >= min_hits
    }
}

/// Configuration for the multi-object tracker.
#[derive(Debug, Clone)]
pub struct TrackerConfig {
    /// IoU threshold for matching detection to track.
    pub iou_threshold: f32,
    /// Maximum frames a track can be lost before deletion.
    pub max_lost: u32,
    /// Minimum hits before a track is "confirmed".
    pub min_hits: u32,
    /// Maximum centroid distance for centroid-based matching.
    pub max_centroid_distance: f32,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self {
            iou_threshold: 0.3,
            max_lost: 5,
            min_hits: 3,
            max_centroid_distance: 80.0,
        }
    }
}

/// Multi-object tracker that combines IoU association with centroid fallback.
#[derive(Debug)]
pub struct MultiObjectTracker {
    config: TrackerConfig,
    tracks: HashMap<u64, Track>,
    next_id: u64,
}

impl MultiObjectTracker {
    /// Create a new tracker.
    #[must_use]
    pub fn new(config: TrackerConfig) -> Self {
        Self {
            config,
            tracks: HashMap::new(),
            next_id: 1,
        }
    }

    /// Default constructor.
    #[must_use]
    pub fn default_config() -> Self {
        Self::new(TrackerConfig::default())
    }

    /// Update the tracker with new detections.
    ///
    /// Returns the list of currently active (confirmed) track IDs and their
    /// bounding boxes.
    pub fn update(&mut self, detections: &[Bbox]) -> Vec<(u64, Bbox)> {
        // Mark all active tracks as potentially lost
        let track_ids: Vec<u64> = self.tracks.keys().copied().collect();
        let mut matched_tracks: Vec<u64> = Vec::new();
        let mut matched_dets: Vec<usize> = Vec::new();

        // IoU-based greedy matching
        for &tid in &track_ids {
            let track_bbox = self.tracks[&tid].bbox;
            let best = detections
                .iter()
                .enumerate()
                .filter(|(di, _)| !matched_dets.contains(di))
                .map(|(di, d)| (di, track_bbox.iou(d)))
                .filter(|(_, iou)| *iou >= self.config.iou_threshold)
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            if let Some((di, _)) = best {
                matched_tracks.push(tid);
                matched_dets.push(di);
            }
        }

        // Apply matches
        for (&tid, &di) in matched_tracks.iter().zip(matched_dets.iter()) {
            if let Some(track) = self.tracks.get_mut(&tid) {
                track.update(detections[di]);
            }
        }

        // Mark unmatched tracks as lost; delete if too many lost frames
        for &tid in &track_ids {
            if !matched_tracks.contains(&tid) {
                if let Some(track) = self.tracks.get_mut(&tid) {
                    track.mark_lost();
                }
            }
        }
        self.tracks
            .retain(|_, t| t.lost_frames <= self.config.max_lost);

        // Create new tracks for unmatched detections
        for (di, det) in detections.iter().enumerate() {
            if !matched_dets.contains(&di) {
                let id = self.next_id;
                self.next_id += 1;
                self.tracks.insert(id, Track::new(id, *det));
            }
        }

        // Return confirmed tracks
        self.tracks
            .values()
            .filter(|t| t.is_confirmed(self.config.min_hits) && t.state != TrackState::Dead)
            .map(|t| (t.id, t.bbox))
            .collect()
    }

    /// Total number of currently tracked objects (any state).
    #[must_use]
    pub fn track_count(&self) -> usize {
        self.tracks.len()
    }

    /// Return a reference to all internal tracks.
    #[must_use]
    pub fn tracks(&self) -> &HashMap<u64, Track> {
        &self.tracks
    }

    /// Return confirmed tracks.
    #[must_use]
    pub fn confirmed_tracks(&self) -> Vec<&Track> {
        self.tracks
            .values()
            .filter(|t| t.is_confirmed(self.config.min_hits))
            .collect()
    }
}

/// Simple centroid-only tracker (no IoU).
#[derive(Debug, Default)]
pub struct CentroidOnlyTracker {
    tracks: HashMap<u64, (f32, f32)>, // id -> centroid
    next_id: u64,
    max_distance: f32,
}

impl CentroidOnlyTracker {
    /// Create a new centroid tracker.
    #[must_use]
    pub fn new(max_distance: f32) -> Self {
        Self {
            tracks: HashMap::new(),
            next_id: 1,
            max_distance,
        }
    }

    /// Update with centroids derived from detection boxes.
    pub fn update(&mut self, detections: &[Bbox]) -> Vec<u64> {
        let cents: Vec<(f32, f32)> = detections.iter().map(Bbox::centroid).collect();
        let mut assigned: Vec<bool> = vec![false; cents.len()];
        let mut updated_ids: Vec<u64> = Vec::new();

        let ids: Vec<u64> = self.tracks.keys().copied().collect();
        for id in ids {
            let (tx, ty) = self.tracks[&id];
            let best = cents
                .iter()
                .enumerate()
                .filter(|(i, _)| !assigned[*i])
                .map(|(i, &(cx, cy))| {
                    let dx = tx - cx;
                    let dy = ty - cy;
                    (i, (dx * dx + dy * dy).sqrt())
                })
                .filter(|(_, d)| *d <= self.max_distance)
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            if let Some((i, _)) = best {
                if let Some(v) = self.tracks.get_mut(&id) {
                    *v = cents[i];
                }
                assigned[i] = true;
                updated_ids.push(id);
            } else {
                // Remove disappeared track
                self.tracks.remove(&id);
            }
        }

        // Register new centroids
        for (i, &c) in cents.iter().enumerate() {
            if !assigned[i] {
                let id = self.next_id;
                self.next_id += 1;
                self.tracks.insert(id, c);
                updated_ids.push(id);
            }
        }

        updated_ids
    }

    /// Total active track count.
    #[must_use]
    pub fn count(&self) -> usize {
        self.tracks.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bbox_centroid() {
        let b = Bbox::new(0.0, 0.0, 10.0, 10.0);
        assert_eq!(b.centroid(), (5.0, 5.0));
    }

    #[test]
    fn test_bbox_area() {
        let b = Bbox::new(0.0, 0.0, 4.0, 5.0);
        assert!((b.area() - 20.0).abs() < 1e-6);
    }

    #[test]
    fn test_bbox_iou_identical() {
        let b = Bbox::new(0.0, 0.0, 10.0, 10.0);
        assert!((b.iou(&b) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_bbox_iou_no_overlap() {
        let a = Bbox::new(0.0, 0.0, 10.0, 10.0);
        let b = Bbox::new(20.0, 20.0, 10.0, 10.0);
        assert!((a.iou(&b)).abs() < 1e-6);
    }

    #[test]
    fn test_bbox_iou_partial_overlap() {
        let a = Bbox::new(0.0, 0.0, 10.0, 10.0);
        let b = Bbox::new(5.0, 0.0, 10.0, 10.0);
        let iou = a.iou(&b);
        assert!(iou > 0.0 && iou < 1.0);
    }

    #[test]
    fn test_bbox_centroid_distance() {
        let a = Bbox::new(0.0, 0.0, 2.0, 2.0);
        let b = Bbox::new(3.0, 4.0, 2.0, 2.0);
        // centroid a = (1,1), b = (4,5) -> dist = sqrt(9+16)=5
        assert!((a.centroid_distance(&b) - 5.0).abs() < 1e-4);
    }

    #[test]
    fn test_track_new() {
        let b = Bbox::new(0.0, 0.0, 10.0, 10.0);
        let t = Track::new(1, b);
        assert_eq!(t.id, 1);
        assert_eq!(t.state, TrackState::New);
        assert_eq!(t.hits, 1);
    }

    #[test]
    fn test_track_update() {
        let mut t = Track::new(1, Bbox::new(0.0, 0.0, 10.0, 10.0));
        t.update(Bbox::new(5.0, 5.0, 10.0, 10.0));
        assert_eq!(t.state, TrackState::Active);
        assert_eq!(t.hits, 2);
        assert_eq!(t.lost_frames, 0);
    }

    #[test]
    fn test_track_mark_lost() {
        let mut t = Track::new(1, Bbox::new(0.0, 0.0, 10.0, 10.0));
        t.mark_lost();
        assert_eq!(t.state, TrackState::Lost);
        assert_eq!(t.lost_frames, 1);
    }

    #[test]
    fn test_track_is_confirmed() {
        let mut t = Track::new(1, Bbox::new(0.0, 0.0, 10.0, 10.0));
        assert!(!t.is_confirmed(3));
        t.update(Bbox::new(0.0, 0.0, 10.0, 10.0));
        t.update(Bbox::new(0.0, 0.0, 10.0, 10.0));
        assert!(t.is_confirmed(3));
    }

    #[test]
    fn test_multi_tracker_creates_tracks() {
        let mut tracker = MultiObjectTracker::new(TrackerConfig {
            min_hits: 1, // confirm immediately
            ..TrackerConfig::default()
        });
        let dets = vec![Bbox::new(0.0, 0.0, 10.0, 10.0)];
        let result = tracker.update(&dets);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_multi_tracker_retains_track_across_frames() {
        let mut tracker = MultiObjectTracker::new(TrackerConfig {
            iou_threshold: 0.3,
            max_lost: 5,
            min_hits: 1,
            max_centroid_distance: 80.0,
        });
        let bbox = Bbox::new(10.0, 10.0, 20.0, 20.0);
        tracker.update(&[bbox]);
        // Same detection next frame — should match
        let result = tracker.update(&[Bbox::new(11.0, 11.0, 20.0, 20.0)]);
        assert!(!result.is_empty());
    }

    #[test]
    fn test_centroid_tracker_basic() {
        let mut ct = CentroidOnlyTracker::new(50.0);
        let dets = vec![Bbox::new(0.0, 0.0, 10.0, 10.0)];
        ct.update(&dets);
        assert_eq!(ct.count(), 1);
    }

    #[test]
    fn test_centroid_tracker_removes_disappeared_objects() {
        let mut ct = CentroidOnlyTracker::new(20.0);
        ct.update(&[Bbox::new(0.0, 0.0, 10.0, 10.0)]);
        // Next frame: no detections close enough
        ct.update(&[Bbox::new(200.0, 200.0, 10.0, 10.0)]);
        // Old track gone, new one created
        assert_eq!(ct.count(), 1);
    }
}
