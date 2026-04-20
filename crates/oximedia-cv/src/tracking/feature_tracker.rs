//! Feature tracking using KLT algorithm.
//!
//! This module provides feature tracking with:
//! - KLT (Kanade-Lucas-Tomasi) feature tracker
//! - Feature lifecycle management
//! - Outlier rejection
//! - Feature descriptor matching
//!
//! # Example
//!
//! ```
//! use oximedia_cv::tracking::FeatureTracker;
//!
//! let mut tracker = FeatureTracker::new(100);
//! ```

use crate::error::{CvError, CvResult};
use crate::tracking::{OpticalFlow, Point2D};

/// Feature tracker using KLT algorithm.
///
/// Tracks features across frames with automatic feature detection,
/// lifecycle management, and outlier rejection.
///
/// # Examples
///
/// ```
/// use oximedia_cv::tracking::FeatureTracker;
///
/// let mut tracker = FeatureTracker::new(500);
/// ```
#[derive(Debug, Clone)]
pub struct FeatureTracker {
    /// Optical flow estimator.
    optical_flow: OpticalFlow,
    /// Maximum number of features to track.
    max_features: usize,
    /// Minimum distance between features.
    min_distance: f32,
    /// Quality threshold for corner detection (0.0-1.0).
    quality_level: f64,
    /// Currently tracked features.
    features: Vec<TrackedFeature>,
    /// Next feature ID to assign.
    next_id: u64,
    /// Previous frame for tracking.
    previous_frame: Option<Vec<u8>>,
    /// Frame dimensions.
    frame_width: u32,
    /// Frame height.
    frame_height: u32,
}

impl FeatureTracker {
    /// Create a new feature tracker.
    ///
    /// # Arguments
    ///
    /// * `max_features` - Maximum number of features to track simultaneously
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::tracking::FeatureTracker;
    ///
    /// let tracker = FeatureTracker::new(300);
    /// ```
    #[must_use]
    pub fn new(max_features: usize) -> Self {
        Self {
            optical_flow: OpticalFlow::default().with_window_size(21),
            max_features,
            min_distance: 10.0,
            quality_level: 0.01,
            features: Vec::new(),
            next_id: 0,
            previous_frame: None,
            frame_width: 0,
            frame_height: 0,
        }
    }

    /// Set minimum distance between features.
    #[must_use]
    pub fn with_min_distance(mut self, distance: f32) -> Self {
        self.min_distance = distance;
        self
    }

    /// Set quality level for corner detection.
    #[must_use]
    pub fn with_quality_level(mut self, level: f64) -> Self {
        self.quality_level = level;
        self
    }

    /// Set optical flow window size.
    #[must_use]
    pub fn with_window_size(mut self, size: u32) -> Self {
        self.optical_flow = self.optical_flow.with_window_size(size);
        self
    }

    /// Track features in a new frame.
    ///
    /// # Arguments
    ///
    /// * `frame` - Grayscale frame data
    /// * `w` - Frame width
    /// * `h` - Frame height
    ///
    /// # Returns
    ///
    /// Vector of currently tracked features.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions are invalid or tracking fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::tracking::FeatureTracker;
    ///
    /// fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut tracker = FeatureTracker::new(100);
    /// let frame = vec![100u8; 10000];
    /// let features = tracker.track(&frame, 100, 100)?;
    /// Ok(())
    /// }
    /// ```
    pub fn track(&mut self, frame: &[u8], w: u32, h: u32) -> CvResult<Vec<TrackedFeature>> {
        if w == 0 || h == 0 {
            return Err(CvError::invalid_dimensions(w, h));
        }

        let size = w as usize * h as usize;
        if frame.len() < size {
            return Err(CvError::insufficient_data(size, frame.len()));
        }

        self.frame_width = w;
        self.frame_height = h;

        // First frame - detect initial features
        if self.previous_frame.is_none() {
            self.detect_new_features(frame, w, h)?;
            self.previous_frame = Some(frame.to_vec());
            return Ok(self.features.clone());
        }

        // previous_frame is Some: is_none check returned early above
        let Some(prev_frame) = self.previous_frame.as_ref() else {
            return Ok(self.features.clone());
        };

        // Track existing features
        if !self.features.is_empty() {
            let old_points: Vec<Point2D> = self.features.iter().map(|f| f.position).collect();

            let new_points =
                self.optical_flow
                    .compute_sparse(prev_frame, frame, w, h, &old_points)?;

            // Update features and filter outliers
            let mut valid_features = Vec::new();
            for (i, &new_pos) in new_points.iter().enumerate() {
                if self.is_valid_position(new_pos, w, h) {
                    let old_feature = &self.features[i];

                    // Calculate velocity
                    let vx = new_pos.x - old_feature.position.x;
                    let vy = new_pos.y - old_feature.position.y;

                    // Reject if displacement is too large (likely tracking error)
                    let displacement = (vx * vx + vy * vy).sqrt();
                    if displacement < 50.0 {
                        let mut feature = old_feature.clone();
                        feature.position = new_pos;
                        feature.velocity = (vx, vy);
                        feature.age += 1;

                        // Update confidence based on tracking quality
                        feature.confidence = self.compute_confidence(&feature, displacement);

                        if feature.confidence > 0.1 {
                            valid_features.push(feature);
                        }
                    }
                }
            }

            self.features = valid_features;
        }

        // Detect new features if needed
        if self.features.len() < self.max_features {
            self.detect_new_features(frame, w, h)?;
        }

        self.previous_frame = Some(frame.to_vec());

        Ok(self.features.clone())
    }

    /// Add features at specified points.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::tracking::{FeatureTracker, Point2D};
    ///
    /// let mut tracker = FeatureTracker::new(100);
    /// let points = vec![Point2D::new(10.0, 20.0)];
    /// tracker.add_features(points);
    /// ```
    pub fn add_features(&mut self, points: Vec<Point2D>) {
        for pt in points {
            if self.features.len() >= self.max_features {
                break;
            }

            // Check minimum distance from existing features
            if self.is_far_enough_from_existing(&pt) {
                self.features.push(TrackedFeature {
                    id: self.next_id,
                    position: pt,
                    velocity: (0.0, 0.0),
                    age: 0,
                    confidence: 1.0,
                });
                self.next_id += 1;
            }
        }
    }

    /// Get current number of tracked features.
    #[must_use]
    pub fn feature_count(&self) -> usize {
        self.features.len()
    }

    /// Reset the tracker.
    pub fn reset(&mut self) {
        self.features.clear();
        self.previous_frame = None;
        self.next_id = 0;
    }

    /// Detect new features in the frame.
    fn detect_new_features(&mut self, frame: &[u8], w: u32, h: u32) -> CvResult<()> {
        let needed = self.max_features.saturating_sub(self.features.len());
        if needed == 0 {
            return Ok(());
        }

        // Use Shi-Tomasi corner detection
        let corners =
            detect_good_features(frame, w, h, needed, self.quality_level, self.min_distance)?;

        // Filter corners that are far from existing features
        for corner in corners {
            if self.features.len() >= self.max_features {
                break;
            }

            if self.is_far_enough_from_existing(&corner) {
                self.features.push(TrackedFeature {
                    id: self.next_id,
                    position: corner,
                    velocity: (0.0, 0.0),
                    age: 0,
                    confidence: 1.0,
                });
                self.next_id += 1;
            }
        }

        Ok(())
    }

    /// Check if position is valid within frame bounds.
    fn is_valid_position(&self, pos: Point2D, w: u32, h: u32) -> bool {
        pos.x >= 0.0 && pos.x < w as f32 && pos.y >= 0.0 && pos.y < h as f32
    }

    /// Check if point is far enough from existing features.
    fn is_far_enough_from_existing(&self, pt: &Point2D) -> bool {
        let min_dist_sq = self.min_distance * self.min_distance;
        for feature in &self.features {
            if feature.position.distance_squared(pt) < min_dist_sq {
                return false;
            }
        }
        true
    }

    /// Compute tracking confidence for a feature.
    fn compute_confidence(&self, feature: &TrackedFeature, displacement: f32) -> f32 {
        let mut confidence = feature.confidence;

        // Decrease confidence with large displacement
        if displacement > 20.0 {
            confidence *= 0.8;
        }

        // Increase confidence for stable features
        if feature.age > 10 && displacement < 5.0 {
            confidence = (confidence * 1.1).min(1.0);
        }

        confidence
    }
}

/// Tracked feature with metadata.
///
/// # Examples
///
/// ```
/// use oximedia_cv::tracking::{TrackedFeature, Point2D};
///
/// let feature = TrackedFeature {
///     id: 0,
///     position: Point2D::new(100.0, 150.0),
///     velocity: (1.0, 2.0),
///     age: 5,
///     confidence: 0.95,
/// };
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct TrackedFeature {
    /// Unique feature identifier.
    pub id: u64,
    /// Current position.
    pub position: Point2D,
    /// Velocity (dx, dy) per frame.
    pub velocity: (f32, f32),
    /// Number of frames tracked.
    pub age: u32,
    /// Tracking confidence (0.0-1.0).
    pub confidence: f32,
}

impl TrackedFeature {
    /// Create a new tracked feature.
    #[must_use]
    pub const fn new(id: u64, position: Point2D) -> Self {
        Self {
            id,
            position,
            velocity: (0.0, 0.0),
            age: 0,
            confidence: 1.0,
        }
    }

    /// Get predicted next position based on velocity.
    #[must_use]
    pub fn predict_next_position(&self) -> Point2D {
        Point2D::new(
            self.position.x + self.velocity.0,
            self.position.y + self.velocity.1,
        )
    }

    /// Get speed (magnitude of velocity).
    #[must_use]
    pub fn speed(&self) -> f32 {
        (self.velocity.0 * self.velocity.0 + self.velocity.1 * self.velocity.1).sqrt()
    }
}

/// Detect good features to track using Shi-Tomasi corner detection.
///
/// # Arguments
///
/// * `frame` - Grayscale image
/// * `w` - Image width
/// * `h` - Image height
/// * `max_corners` - Maximum number of corners to detect
/// * `quality_level` - Quality threshold (0.0-1.0)
/// * `min_distance` - Minimum distance between corners
///
/// # Returns
///
/// Vector of detected corner positions.
///
/// # Errors
///
/// Returns an error if dimensions are invalid.
#[allow(clippy::too_many_arguments)]
pub fn detect_good_features(
    frame: &[u8],
    w: u32,
    h: u32,
    max_corners: usize,
    quality_level: f64,
    min_distance: f32,
) -> CvResult<Vec<Point2D>> {
    if w == 0 || h == 0 {
        return Err(CvError::invalid_dimensions(w, h));
    }

    let size = w as usize * h as usize;
    if frame.len() < size {
        return Err(CvError::insufficient_data(size, frame.len()));
    }

    // Compute corner response using Shi-Tomasi (minimum eigenvalue)
    let corner_response = compute_shi_tomasi_response(frame, w, h);

    // Find local maxima
    let corners_with_response = find_local_maxima(&corner_response, w, h, quality_level);

    // Apply non-maximum suppression with minimum distance
    let mut corners = apply_min_distance_suppression(corners_with_response, min_distance);

    // Take top max_corners
    if corners.len() > max_corners {
        corners.truncate(max_corners);
    }

    Ok(corners)
}

/// Compute Shi-Tomasi corner response (minimum eigenvalue).
fn compute_shi_tomasi_response(frame: &[u8], w: u32, h: u32) -> Vec<f64> {
    let wi = w as i32;
    let hi = h as i32;
    let size = w as usize * h as usize;
    let mut response = vec![0.0; size];

    for y in 3..hi - 3 {
        for x in 3..wi - 3 {
            // Compute structure tensor in 5x5 window
            let mut gxx = 0.0;
            let mut gxy = 0.0;
            let mut gyy = 0.0;

            for dy in -2..=2 {
                for dx in -2..=2 {
                    let px = x + dx;
                    let py = y + dy;

                    // Sobel gradients
                    let idx_l = (py * wi + (px - 1)) as usize;
                    let idx_r = (py * wi + (px + 1)) as usize;
                    let idx_t = ((py - 1) * wi + px) as usize;
                    let idx_b = ((py + 1) * wi + px) as usize;

                    let ix = (frame[idx_r] as f64 - frame[idx_l] as f64) / 2.0;
                    let iy = (frame[idx_b] as f64 - frame[idx_t] as f64) / 2.0;

                    gxx += ix * ix;
                    gxy += ix * iy;
                    gyy += iy * iy;
                }
            }

            // Minimum eigenvalue (Shi-Tomasi)
            let trace = gxx + gyy;
            let det = gxx * gyy - gxy * gxy;
            let min_eigenvalue = (trace - (trace * trace - 4.0 * det).sqrt()) / 2.0;

            response[(y * wi + x) as usize] = min_eigenvalue.max(0.0);
        }
    }

    response
}

/// Find local maxima in corner response.
fn find_local_maxima(response: &[f64], w: u32, h: u32, quality_level: f64) -> Vec<(Point2D, f64)> {
    let wi = w as i32;
    let hi = h as i32;
    let mut corners = Vec::new();

    // Find maximum response
    let max_response = response.iter().fold(0.0f64, |a, &b| a.max(b));
    let threshold = max_response * quality_level;

    // Find local maxima
    for y in 3..hi - 3 {
        for x in 3..wi - 3 {
            let idx = (y * wi + x) as usize;
            let val = response[idx];

            if val < threshold {
                continue;
            }

            // Check 3x3 neighborhood
            let mut is_maximum = true;
            'outer: for dy in -1..=1 {
                for dx in -1..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }

                    let nidx = ((y + dy) * wi + (x + dx)) as usize;
                    if response[nidx] > val {
                        is_maximum = false;
                        break 'outer;
                    }
                }
            }

            if is_maximum {
                corners.push((Point2D::new(x as f32, y as f32), val));
            }
        }
    }

    // Sort by response (descending)
    corners.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    corners
}

/// Apply minimum distance suppression.
fn apply_min_distance_suppression(
    mut corners: Vec<(Point2D, f64)>,
    min_distance: f32,
) -> Vec<Point2D> {
    if min_distance <= 0.0 {
        return corners.into_iter().map(|(pt, _)| pt).collect();
    }

    let min_dist_sq = min_distance * min_distance;
    let mut result = Vec::new();

    while let Some((pt, _)) = corners.pop() {
        // Check distance from already accepted corners
        let mut too_close = false;
        for &accepted in &result {
            if pt.distance_squared(&accepted) < min_dist_sq {
                too_close = true;
                break;
            }
        }

        if !too_close {
            result.push(pt);
        }
    }

    result.reverse(); // Restore quality order
    result
}

/// Compute feature descriptor for matching.
///
/// Simple patch-based descriptor using normalized pixel values.
pub fn compute_descriptor(
    frame: &[u8],
    w: u32,
    h: u32,
    pt: Point2D,
    patch_size: u32,
) -> Option<Vec<f32>> {
    let half = (patch_size / 2) as i32;
    let x = pt.x as i32;
    let y = pt.y as i32;

    if x < half || x >= w as i32 - half || y < half || y >= h as i32 - half {
        return None;
    }

    let mut descriptor = Vec::with_capacity((patch_size * patch_size) as usize);
    let mut sum = 0.0f32;
    let mut sum_sq = 0.0f32;

    // Extract patch
    for dy in -half..=half {
        for dx in -half..=half {
            let px = (x + dx) as u32;
            let py = (y + dy) as u32;
            let idx = (py * w + px) as usize;
            let val = frame[idx] as f32;
            descriptor.push(val);
            sum += val;
            sum_sq += val * val;
        }
    }

    // Normalize (zero mean, unit variance)
    let n = descriptor.len() as f32;
    let mean = sum / n;
    let std = ((sum_sq / n) - (mean * mean)).sqrt();

    if std > 1.0 {
        for val in &mut descriptor {
            *val = (*val - mean) / std;
        }
    }

    Some(descriptor)
}

/// Match features between two frames using descriptors.
///
/// # Returns
///
/// Pairs of matched feature indices.
#[allow(dead_code)]
pub fn match_features(
    descriptors1: &[Vec<f32>],
    descriptors2: &[Vec<f32>],
    threshold: f32,
) -> Vec<(usize, usize)> {
    let mut matches = Vec::new();

    for (i, desc1) in descriptors1.iter().enumerate() {
        let mut best_match = None;
        let mut best_dist = f32::MAX;

        for (j, desc2) in descriptors2.iter().enumerate() {
            let dist = descriptor_distance(desc1, desc2);
            if dist < best_dist {
                best_dist = dist;
                best_match = Some(j);
            }
        }

        if best_dist < threshold {
            if let Some(j) = best_match {
                matches.push((i, j));
            }
        }
    }

    matches
}

/// Compute normalized cross-correlation distance between descriptors.
fn descriptor_distance(desc1: &[f32], desc2: &[f32]) -> f32 {
    if desc1.len() != desc2.len() {
        return f32::MAX;
    }

    let mut dot_product = 0.0;
    let mut norm1 = 0.0;
    let mut norm2 = 0.0;

    for i in 0..desc1.len() {
        dot_product += desc1[i] * desc2[i];
        norm1 += desc1[i] * desc1[i];
        norm2 += desc2[i] * desc2[i];
    }

    let norm_product = (norm1 * norm2).sqrt();
    if norm_product > f32::EPSILON {
        1.0 - (dot_product / norm_product)
    } else {
        f32::MAX
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_tracker_new() {
        let tracker = FeatureTracker::new(100);
        assert_eq!(tracker.max_features, 100);
        assert_eq!(tracker.feature_count(), 0);
    }

    #[test]
    fn test_feature_tracker_with_params() {
        let tracker = FeatureTracker::new(100)
            .with_min_distance(15.0)
            .with_quality_level(0.05);

        assert!((tracker.min_distance - 15.0).abs() < f32::EPSILON);
        assert!((tracker.quality_level - 0.05).abs() < f64::EPSILON);
    }

    #[test]
    fn test_tracked_feature_new() {
        let feature = TrackedFeature::new(42, Point2D::new(10.0, 20.0));
        assert_eq!(feature.id, 42);
        assert_eq!(feature.position.x, 10.0);
        assert_eq!(feature.age, 0);
    }

    #[test]
    fn test_tracked_feature_predict() {
        let mut feature = TrackedFeature::new(0, Point2D::new(10.0, 20.0));
        feature.velocity = (5.0, 3.0);

        let next = feature.predict_next_position();
        assert_eq!(next.x, 15.0);
        assert_eq!(next.y, 23.0);
    }

    #[test]
    fn test_tracked_feature_speed() {
        let mut feature = TrackedFeature::new(0, Point2D::new(0.0, 0.0));
        feature.velocity = (3.0, 4.0);

        assert!((feature.speed() - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_feature_tracker_track() {
        let mut tracker = FeatureTracker::new(50);
        let frame = vec![100u8; 10000];

        let features = tracker
            .track(&frame, 100, 100)
            .expect("track should succeed");
        assert!(features.len() <= 50);
    }

    #[test]
    fn test_feature_tracker_add_features() {
        let mut tracker = FeatureTracker::new(10);
        let points = vec![Point2D::new(10.0, 10.0), Point2D::new(50.0, 50.0)];

        tracker.add_features(points);
        assert_eq!(tracker.feature_count(), 2);
    }

    #[test]
    fn test_feature_tracker_reset() {
        let mut tracker = FeatureTracker::new(10);
        tracker.add_features(vec![Point2D::new(10.0, 10.0)]);

        tracker.reset();
        assert_eq!(tracker.feature_count(), 0);
    }

    #[test]
    fn test_detect_good_features() {
        let frame = vec![100u8; 10000];
        let corners = detect_good_features(&frame, 100, 100, 50, 0.01, 10.0)
            .expect("detect_good_features should succeed");
        assert!(corners.len() <= 50);
    }

    #[test]
    fn test_compute_descriptor() {
        let frame = vec![100u8; 10000];
        let descriptor = compute_descriptor(&frame, 100, 100, Point2D::new(50.0, 50.0), 7);
        assert!(descriptor.is_some());

        let desc = descriptor.expect("desc should be valid");
        assert_eq!(desc.len(), 49); // 7x7 patch
    }

    #[test]
    fn test_descriptor_distance() {
        let desc1 = vec![1.0, 0.0, -1.0];
        let desc2 = vec![1.0, 0.0, -1.0];

        let dist = descriptor_distance(&desc1, &desc2);
        assert!(dist < 0.001); // Should be ~0 for identical descriptors
    }
}
