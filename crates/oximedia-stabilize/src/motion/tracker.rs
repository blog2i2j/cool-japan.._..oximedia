//! Feature tracking across video frames.
//!
//! This module implements feature detection and tracking algorithms for
//! estimating camera motion between consecutive frames.

use crate::error::{StabilizeError, StabilizeResult};
use crate::Frame;
use scirs2_core::ndarray::Array2;
use std::collections::HashMap;

/// A tracked feature point in an image.
#[derive(Debug, Clone, Copy)]
pub struct Feature {
    /// X coordinate
    pub x: f64,
    /// Y coordinate
    pub y: f64,
    /// Feature quality/strength (0.0-1.0)
    pub quality: f64,
    /// Feature ID for tracking
    pub id: usize,
}

impl Feature {
    /// Create a new feature.
    #[must_use]
    pub const fn new(x: f64, y: f64, quality: f64, id: usize) -> Self {
        Self { x, y, quality, id }
    }

    /// Calculate distance to another feature.
    #[must_use]
    pub fn distance_to(&self, other: &Self) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }

    /// Check if feature is valid (within bounds).
    #[must_use]
    pub fn is_valid(&self, width: usize, height: usize) -> bool {
        self.x >= 0.0
            && self.y >= 0.0
            && self.x < width as f64
            && self.y < height as f64
            && self.quality > 0.0
    }
}

/// A track of a single feature across multiple frames.
#[derive(Debug, Clone)]
pub struct FeatureTrack {
    /// Feature ID
    pub id: usize,
    /// Feature positions in each frame (frame_index -> position)
    pub positions: HashMap<usize, (f64, f64)>,
    /// Track start frame
    pub start_frame: usize,
    /// Track end frame
    pub end_frame: usize,
    /// Average quality across track
    pub avg_quality: f64,
}

impl FeatureTrack {
    /// Create a new feature track.
    #[must_use]
    pub fn new(id: usize, start_frame: usize) -> Self {
        Self {
            id,
            positions: HashMap::new(),
            start_frame,
            end_frame: start_frame,
            avg_quality: 0.0,
        }
    }

    /// Add a position to the track.
    pub fn add_position(&mut self, frame: usize, x: f64, y: f64, quality: f64) {
        self.positions.insert(frame, (x, y));
        self.end_frame = self.end_frame.max(frame);

        // Update average quality
        let count = self.positions.len();
        self.avg_quality = (self.avg_quality * (count - 1) as f64 + quality) / count as f64;
    }

    /// Get position at a specific frame.
    #[must_use]
    pub fn position_at(&self, frame: usize) -> Option<(f64, f64)> {
        self.positions.get(&frame).copied()
    }

    /// Get track length (number of frames).
    #[must_use]
    pub fn length(&self) -> usize {
        self.positions.len()
    }

    /// Check if track is active at a given frame.
    #[must_use]
    pub fn is_active_at(&self, frame: usize) -> bool {
        frame >= self.start_frame && frame <= self.end_frame
    }

    /// Calculate track displacement (total motion).
    #[must_use]
    pub fn total_displacement(&self) -> f64 {
        if self.positions.len() < 2 {
            return 0.0;
        }

        let start_pos = self.positions.get(&self.start_frame);
        let end_pos = self.positions.get(&self.end_frame);

        match (start_pos, end_pos) {
            (Some((x1, y1)), Some((x2, y2))) => {
                let dx = x2 - x1;
                let dy = y2 - y1;
                (dx * dx + dy * dy).sqrt()
            }
            _ => 0.0,
        }
    }
}

/// Motion tracker that detects and tracks features across frames.
#[derive(Debug)]
pub struct MotionTracker {
    /// Maximum number of features to track
    max_features: usize,
    /// Minimum feature quality threshold
    quality_threshold: f64,
    /// Feature detection grid size (for spatial distribution)
    grid_size: usize,
    /// Current feature ID counter
    next_feature_id: usize,
    /// Active feature tracks
    active_tracks: Vec<FeatureTrack>,
}

impl MotionTracker {
    /// Create a new motion tracker.
    #[must_use]
    pub fn new(max_features: usize) -> Self {
        Self {
            max_features,
            quality_threshold: 0.01,
            grid_size: 10,
            next_feature_id: 0,
            active_tracks: Vec::new(),
        }
    }

    /// Set quality threshold for feature detection.
    pub fn set_quality_threshold(&mut self, threshold: f64) {
        self.quality_threshold = threshold.clamp(0.0, 1.0);
    }

    /// Track features across a sequence of frames.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Frame sequence is empty
    /// - Feature detection fails
    /// - Insufficient features are found
    pub fn track(&mut self, frames: &[Frame]) -> StabilizeResult<Vec<FeatureTrack>> {
        if frames.is_empty() {
            return Err(StabilizeError::EmptyFrameSequence);
        }

        // Detect features in the first frame
        let mut features = self.detect_features(&frames[0])?;

        // Initialize tracks
        self.active_tracks = features
            .iter()
            .map(|f| {
                let mut track = FeatureTrack::new(f.id, 0);
                track.add_position(0, f.x, f.y, f.quality);
                track
            })
            .collect();

        // Track features through remaining frames
        for (frame_idx, frame) in frames.iter().enumerate().skip(1) {
            // Track existing features
            let prev_frame = &frames[frame_idx - 1];
            features = self.track_features(prev_frame, frame, &features)?;

            // Update tracks
            for feature in &features {
                if let Some(track) = self.active_tracks.iter_mut().find(|t| t.id == feature.id) {
                    track.add_position(frame_idx, feature.x, feature.y, feature.quality);
                }
            }

            // Detect new features if needed
            if features.len() < self.max_features / 2 {
                let new_features = self.detect_new_features(frame, &features)?;
                for feature in new_features {
                    let mut track = FeatureTrack::new(feature.id, frame_idx);
                    track.add_position(frame_idx, feature.x, feature.y, feature.quality);
                    self.active_tracks.push(track);
                    features.push(feature);
                }
            }
        }

        // Filter out short tracks (less than 3 frames)
        self.active_tracks.retain(|track| track.length() >= 3);

        if self.active_tracks.is_empty() {
            return Err(StabilizeError::insufficient_features(0, 10));
        }

        Ok(self.active_tracks.clone())
    }

    /// Detect features in a frame using Harris corner detection.
    fn detect_features(&mut self, frame: &Frame) -> StabilizeResult<Vec<Feature>> {
        let mut features = Vec::new();

        // Compute Harris corner response
        let corners = self.harris_corner_detection(&frame.data)?;

        // Extract features from corners
        let cell_width = frame.width / self.grid_size;
        let cell_height = frame.height / self.grid_size;

        // Ensure spatial distribution by dividing image into grid
        for grid_y in 0..self.grid_size {
            for grid_x in 0..self.grid_size {
                let x_start = grid_x * cell_width;
                let y_start = grid_y * cell_height;
                let x_end = ((grid_x + 1) * cell_width).min(frame.width);
                let y_end = ((grid_y + 1) * cell_height).min(frame.height);

                // Find best corner in this cell
                let mut best_corner = None;
                let mut best_quality = self.quality_threshold;

                for y in y_start..y_end {
                    for x in x_start..x_end {
                        let quality = corners[[y, x]];
                        if quality > best_quality {
                            best_quality = quality;
                            best_corner = Some((x, y));
                        }
                    }
                }

                if let Some((x, y)) = best_corner {
                    features.push(Feature::new(
                        x as f64,
                        y as f64,
                        best_quality,
                        self.next_feature_id,
                    ));
                    self.next_feature_id += 1;

                    if features.len() >= self.max_features {
                        return Ok(features);
                    }
                }
            }
        }

        if features.len() < 10 {
            return Err(StabilizeError::insufficient_features(features.len(), 10));
        }

        Ok(features)
    }

    /// Track existing features from one frame to the next using optical flow.
    fn track_features(
        &self,
        prev_frame: &Frame,
        curr_frame: &Frame,
        features: &[Feature],
    ) -> StabilizeResult<Vec<Feature>> {
        let mut tracked_features = Vec::new();

        for feature in features {
            if let Some(new_pos) = self.track_single_feature(prev_frame, curr_frame, feature) {
                if new_pos.is_valid(curr_frame.width, curr_frame.height) {
                    tracked_features.push(new_pos);
                }
            }
        }

        Ok(tracked_features)
    }

    /// Track a single feature using Lucas-Kanade optical flow.
    fn track_single_feature(
        &self,
        prev_frame: &Frame,
        curr_frame: &Frame,
        feature: &Feature,
    ) -> Option<Feature> {
        let window_size = 21;
        let half_window = window_size / 2;

        let x = feature.x as usize;
        let y = feature.y as usize;

        // Check bounds
        if x < half_window
            || y < half_window
            || x + half_window >= prev_frame.width
            || y + half_window >= prev_frame.height
        {
            return None;
        }

        // Simple template matching (in production, use Lucas-Kanade)
        let mut best_dx = 0;
        let mut best_dy = 0;
        let mut best_score = f64::MAX;

        let search_radius = 20;

        for dy in -(search_radius as i32)..=(search_radius as i32) {
            for dx in -(search_radius as i32)..=(search_radius as i32) {
                let nx_i = x as i32 + dx;
                let ny_i = y as i32 + dy;

                if nx_i < half_window as i32
                    || ny_i < half_window as i32
                    || nx_i + half_window as i32 >= curr_frame.width as i32
                    || ny_i + half_window as i32 >= curr_frame.height as i32
                {
                    continue;
                }

                let new_x = nx_i as usize;
                let new_y = ny_i as usize;

                let score = self.template_match(
                    &prev_frame.data,
                    &curr_frame.data,
                    x,
                    y,
                    new_x,
                    new_y,
                    window_size,
                );

                if score < best_score {
                    best_score = score;
                    best_dx = dx;
                    best_dy = dy;
                }
            }
        }

        // Quality based on matching score
        let quality = 1.0 / (1.0 + best_score);

        if quality < self.quality_threshold {
            return None;
        }

        Some(Feature::new(
            (x as i32 + best_dx) as f64,
            (y as i32 + best_dy) as f64,
            quality,
            feature.id,
        ))
    }

    /// Template matching using sum of squared differences.
    fn template_match(
        &self,
        prev: &Array2<u8>,
        curr: &Array2<u8>,
        x1: usize,
        y1: usize,
        x2: usize,
        y2: usize,
        window_size: usize,
    ) -> f64 {
        let half = window_size / 2;
        let mut sum = 0.0;
        let mut count = 0;

        for dy in 0..window_size {
            for dx in 0..window_size {
                let py = y1 + dy - half;
                let px = x1 + dx - half;
                let cy = y2 + dy - half;
                let cx = x2 + dx - half;

                let p1 = prev[[py, px]] as f64;
                let p2 = curr[[cy, cx]] as f64;

                let diff = p1 - p2;
                sum += diff * diff;
                count += 1;
            }
        }

        sum / count as f64
    }

    /// Detect new features avoiding existing ones.
    fn detect_new_features(
        &mut self,
        frame: &Frame,
        existing: &[Feature],
    ) -> StabilizeResult<Vec<Feature>> {
        let mut new_features = Vec::new();
        let corners = self.harris_corner_detection(&frame.data)?;

        let min_distance = 20.0; // Minimum distance between features

        for y in 10..(frame.height - 10) {
            for x in 10..(frame.width - 10) {
                let quality = corners[[y, x]];

                if quality < self.quality_threshold {
                    continue;
                }

                // Check distance to existing features
                let too_close = existing.iter().any(|f| {
                    let dx = f.x - x as f64;
                    let dy = f.y - y as f64;
                    (dx * dx + dy * dy).sqrt() < min_distance
                });

                if too_close {
                    continue;
                }

                new_features.push(Feature::new(
                    x as f64,
                    y as f64,
                    quality,
                    self.next_feature_id,
                ));
                self.next_feature_id += 1;

                if new_features.len() >= self.max_features / 2 {
                    return Ok(new_features);
                }
            }
        }

        Ok(new_features)
    }

    /// Harris corner detection.
    fn harris_corner_detection(&self, image: &Array2<u8>) -> StabilizeResult<Array2<f64>> {
        let (height, width) = image.dim();
        let mut corners = Array2::zeros((height, width));

        // Compute gradients
        let (grad_x, grad_y) = self.compute_gradients(image);

        // Compute structure tensor components
        let window_size = 5;
        let half = window_size / 2;
        let k = 0.04; // Harris parameter

        for y in half..(height - half) {
            for x in half..(width - half) {
                let mut ixx = 0.0;
                let mut iyy = 0.0;
                let mut ixy = 0.0;

                // Sum over window
                for dy in 0..window_size {
                    for dx in 0..window_size {
                        let py = y + dy - half;
                        let px = x + dx - half;

                        let gx = grad_x[[py, px]];
                        let gy = grad_y[[py, px]];

                        ixx += gx * gx;
                        iyy += gy * gy;
                        ixy += gx * gy;
                    }
                }

                // Compute corner response
                let det = ixx * iyy - ixy * ixy;
                let trace = ixx + iyy;
                let response = det - k * trace * trace;

                corners[[y, x]] = response.max(0.0);
            }
        }

        // Normalize to 0-1
        let max_response = corners.iter().fold(0.0_f64, |a, &b| a.max(b));
        if max_response > 0.0 {
            corners.mapv_inplace(|v| v / max_response);
        }

        Ok(corners)
    }

    /// Compute image gradients using Sobel operator.
    fn compute_gradients(&self, image: &Array2<u8>) -> (Array2<f64>, Array2<f64>) {
        let (height, width) = image.dim();
        let mut grad_x = Array2::zeros((height, width));
        let mut grad_y = Array2::zeros((height, width));

        // Sobel kernels
        let sobel_x = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]];
        let sobel_y = [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]];

        for y in 1..(height - 1) {
            for x in 1..(width - 1) {
                let mut gx = 0.0;
                let mut gy = 0.0;

                for ky in 0..3 {
                    for kx in 0..3 {
                        let pixel = image[[y + ky - 1, x + kx - 1]] as f64;
                        gx += pixel * sobel_x[ky][kx];
                        gy += pixel * sobel_y[ky][kx];
                    }
                }

                grad_x[[y, x]] = gx;
                grad_y[[y, x]] = gy;
            }
        }

        (grad_x, grad_y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_creation() {
        let feature = Feature::new(10.0, 20.0, 0.8, 0);
        assert!((feature.x - 10.0).abs() < f64::EPSILON);
        assert!((feature.y - 20.0).abs() < f64::EPSILON);
        assert!((feature.quality - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn test_feature_distance() {
        let f1 = Feature::new(0.0, 0.0, 1.0, 0);
        let f2 = Feature::new(3.0, 4.0, 1.0, 1);
        assert!((f1.distance_to(&f2) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_feature_validity() {
        let feature = Feature::new(50.0, 60.0, 0.5, 0);
        assert!(feature.is_valid(100, 100));

        let invalid = Feature::new(150.0, 60.0, 0.5, 0);
        assert!(!invalid.is_valid(100, 100));
    }

    #[test]
    fn test_feature_track() {
        let mut track = FeatureTrack::new(0, 0);
        track.add_position(0, 10.0, 20.0, 0.8);
        track.add_position(1, 15.0, 25.0, 0.7);

        assert_eq!(track.length(), 2);
        assert!(track.is_active_at(1));
        assert!(!track.is_active_at(5));
    }

    #[test]
    fn test_motion_tracker_creation() {
        let tracker = MotionTracker::new(500);
        assert_eq!(tracker.max_features, 500);
    }
}

/// Feature matching utilities.
pub mod matching {
    use super::Feature;

    /// Match features between two frames using descriptors.
    pub struct FeatureMatcher {
        max_distance: f64,
        ratio_threshold: f64,
    }

    impl FeatureMatcher {
        /// Create a new feature matcher.
        #[must_use]
        pub fn new() -> Self {
            Self {
                max_distance: 50.0,
                ratio_threshold: 0.8,
            }
        }

        /// Match features using nearest neighbor.
        #[must_use]
        pub fn match_features(&self, features1: &[Feature], features2: &[Feature]) -> Vec<Match> {
            let mut matches = Vec::new();

            for (i, f1) in features1.iter().enumerate() {
                let mut best_distance = f64::MAX;
                let mut best_match = None;
                let mut second_best = f64::MAX;

                for (j, f2) in features2.iter().enumerate() {
                    let dist = f1.distance_to(f2);

                    if dist < best_distance {
                        second_best = best_distance;
                        best_distance = dist;
                        best_match = Some(j);
                    } else if dist < second_best {
                        second_best = dist;
                    }
                }

                // Lowe's ratio test
                if best_distance < self.max_distance
                    && best_distance < second_best * self.ratio_threshold
                {
                    if let Some(j) = best_match {
                        matches.push(Match {
                            index1: i,
                            index2: j,
                            distance: best_distance,
                        });
                    }
                }
            }

            matches
        }

        /// Filter matches using geometric constraints.
        #[must_use]
        pub fn filter_geometric(
            &self,
            matches: &[Match],
            features1: &[Feature],
            features2: &[Feature],
        ) -> Vec<Match> {
            matches
                .iter()
                .filter(|m| {
                    let f1 = &features1[m.index1];
                    let f2 = &features2[m.index2];

                    // Check if motion is reasonable
                    let motion = f1.distance_to(f2);
                    motion < self.max_distance
                })
                .copied()
                .collect()
        }
    }

    impl Default for FeatureMatcher {
        fn default() -> Self {
            Self::new()
        }
    }

    /// A match between two features.
    #[derive(Debug, Clone, Copy)]
    pub struct Match {
        /// Index in first feature set
        pub index1: usize,
        /// Index in second feature set
        pub index2: usize,
        /// Match distance
        pub distance: f64,
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_matcher() {
            let matcher = FeatureMatcher::new();
            let features1 = vec![Feature::new(0.0, 0.0, 1.0, 0)];
            let features2 = vec![Feature::new(5.0, 5.0, 1.0, 1)];

            let matches = matcher.match_features(&features1, &features2);
            assert!(!matches.is_empty());
        }
    }
}

/// Feature descriptor computation.
pub mod descriptors {
    use scirs2_core::ndarray::Array2;

    /// Compute BRIEF descriptor for a feature.
    #[must_use]
    pub fn compute_brief_descriptor(
        image: &Array2<u8>,
        x: usize,
        y: usize,
        size: usize,
    ) -> Vec<u8> {
        let mut descriptor = Vec::new();
        let half = size / 2;

        // Sample pairs of pixels
        for dy in 0..size {
            for dx in 0..size {
                let y1 = y.saturating_add(dy).saturating_sub(half);
                let x1 = x.saturating_add(dx).saturating_sub(half);

                if y1 < image.dim().0 && x1 < image.dim().1 {
                    descriptor.push(image[[y1, x1]]);
                }
            }
        }

        descriptor
    }

    /// Compute ORB descriptor.
    #[must_use]
    pub fn compute_orb_descriptor(image: &Array2<u8>, x: usize, y: usize) -> Vec<u8> {
        // Simplified ORB descriptor
        compute_brief_descriptor(image, x, y, 31)
    }

    /// Hamming distance between binary descriptors.
    #[must_use]
    pub fn hamming_distance(desc1: &[u8], desc2: &[u8]) -> usize {
        desc1
            .iter()
            .zip(desc2.iter())
            .map(|(a, b)| (a ^ b).count_ones() as usize)
            .sum()
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_brief_descriptor() {
            let image = Array2::from_elem((100, 100), 128);
            let desc = compute_brief_descriptor(&image, 50, 50, 11);
            assert!(!desc.is_empty());
        }

        #[test]
        fn test_hamming_distance() {
            let desc1 = vec![0b11110000, 0b00001111];
            let desc2 = vec![0b11110000, 0b11110000];
            let dist = hamming_distance(&desc1, &desc2);
            assert_eq!(dist, 8);
        }
    }
}
