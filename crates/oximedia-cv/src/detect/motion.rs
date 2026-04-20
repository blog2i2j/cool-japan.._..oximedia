//! Motion detection module.
//!
//! This module provides motion detection algorithms including:
//! - Frame differencing
//! - Background subtraction (MOG2 skeleton)
//! - Optical flow (Lucas-Kanade)
//!
//! # Example
//!
//! ```
//! use oximedia_cv::detect::{MotionDetector, MotionRegion};
//!
//! let mut detector = MotionDetector::new(10, 10);
//! ```

use crate::error::{CvError, CvResult};

/// Motion detection region.
#[derive(Debug, Clone)]
pub struct MotionRegion {
    /// X coordinate of bounding box.
    pub x: u32,
    /// Y coordinate of bounding box.
    pub y: u32,
    /// Width of bounding box.
    pub width: u32,
    /// Height of bounding box.
    pub height: u32,
    /// Motion intensity (0.0 - 1.0).
    pub intensity: f64,
    /// Centroid of motion region.
    pub centroid: (f64, f64),
}

impl MotionRegion {
    /// Create a new motion region.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::detect::MotionRegion;
    ///
    /// let region = MotionRegion::new(10, 20, 100, 100, 0.8);
    /// assert!(region.intensity > 0.5);
    /// ```
    #[must_use]
    pub fn new(x: u32, y: u32, width: u32, height: u32, intensity: f64) -> Self {
        Self {
            x,
            y,
            width,
            height,
            intensity,
            centroid: (
                x as f64 + width as f64 / 2.0,
                y as f64 + height as f64 / 2.0,
            ),
        }
    }

    /// Get the area of the motion region.
    #[must_use]
    pub const fn area(&self) -> u32 {
        self.width * self.height
    }
}

/// Motion detector using frame differencing.
#[derive(Debug, Clone)]
pub struct MotionDetector {
    /// Previous frame (grayscale).
    previous_frame: Option<Vec<u8>>,
    /// Background model (running average).
    background: Option<Vec<f64>>,
    /// Frame width.
    width: u32,
    /// Frame height.
    height: u32,
    /// Difference threshold for motion detection.
    threshold: u8,
    /// Learning rate for background model.
    learning_rate: f64,
    /// Minimum motion area to consider.
    min_area: u32,
}

impl MotionDetector {
    /// Create a new motion detector.
    ///
    /// # Arguments
    ///
    /// * `width` - Frame width
    /// * `height` - Frame height
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::detect::MotionDetector;
    ///
    /// let detector = MotionDetector::new(640, 480);
    /// ```
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            previous_frame: None,
            background: None,
            width,
            height,
            threshold: 25,
            learning_rate: 0.01,
            min_area: 100,
        }
    }

    /// Set the difference threshold.
    #[must_use]
    pub const fn with_threshold(mut self, threshold: u8) -> Self {
        self.threshold = threshold;
        self
    }

    /// Set the learning rate for background model.
    #[must_use]
    pub const fn with_learning_rate(mut self, rate: f64) -> Self {
        self.learning_rate = rate;
        self
    }

    /// Set minimum motion area.
    #[must_use]
    pub const fn with_min_area(mut self, area: u32) -> Self {
        self.min_area = area;
        self
    }

    /// Process a new frame and detect motion.
    ///
    /// # Arguments
    ///
    /// * `frame` - Grayscale frame data
    ///
    /// # Returns
    ///
    /// Tuple of (motion mask, motion regions).
    ///
    /// # Errors
    ///
    /// Returns an error if frame dimensions are invalid.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::detect::MotionDetector;
    ///
    /// fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut detector = MotionDetector::new(10, 10);
    /// let frame = vec![100u8; 100];
    /// let (mask, regions) = detector.process(&frame)?;
    /// Ok(())
    /// }
    /// ```
    pub fn process(&mut self, frame: &[u8]) -> CvResult<(Vec<u8>, Vec<MotionRegion>)> {
        let expected_size = self.width as usize * self.height as usize;
        if frame.len() < expected_size {
            return Err(CvError::insufficient_data(expected_size, frame.len()));
        }

        // Initialize background model if needed
        if self.background.is_none() {
            self.background = Some(frame.iter().map(|&v| v as f64).collect());
            self.previous_frame = Some(frame.to_vec());
            return Ok((vec![0u8; expected_size], Vec::new()));
        }

        // Safety: we set self.background to Some above, so this will always succeed.
        let Some(background) = self.background.as_mut() else {
            return Ok((vec![0u8; expected_size], Vec::new()));
        };

        // Compute difference mask
        let mut mask = vec![0u8; expected_size];
        let mut motion_count = 0u64;

        for i in 0..expected_size {
            let diff = (frame[i] as f64 - background[i]).abs();
            if diff > self.threshold as f64 {
                mask[i] = 255;
                motion_count += 1;
            }

            // Update background model
            background[i] =
                background[i] * (1.0 - self.learning_rate) + frame[i] as f64 * self.learning_rate;
        }

        // Store current frame
        self.previous_frame = Some(frame.to_vec());

        // Find motion regions
        let regions = self.find_motion_regions(&mask, motion_count);

        Ok((mask, regions))
    }

    /// Reset the motion detector state.
    pub fn reset(&mut self) {
        self.previous_frame = None;
        self.background = None;
    }

    /// Find connected motion regions in the mask.
    fn find_motion_regions(&self, mask: &[u8], total_motion: u64) -> Vec<MotionRegion> {
        if total_motion == 0 {
            return Vec::new();
        }

        let w = self.width as usize;
        let h = self.height as usize;

        // Simple bounding box approach (find overall motion bounds)
        let mut min_x = w;
        let mut min_y = h;
        let mut max_x = 0usize;
        let mut max_y = 0usize;

        for y in 0..h {
            for x in 0..w {
                if mask[y * w + x] > 0 {
                    min_x = min_x.min(x);
                    min_y = min_y.min(y);
                    max_x = max_x.max(x);
                    max_y = max_y.max(y);
                }
            }
        }

        if max_x < min_x || max_y < min_y {
            return Vec::new();
        }

        let region_width = (max_x - min_x + 1) as u32;
        let region_height = (max_y - min_y + 1) as u32;
        let area = region_width * region_height;

        if area < self.min_area {
            return Vec::new();
        }

        let intensity = total_motion as f64 / area as f64;

        vec![MotionRegion::new(
            min_x as u32,
            min_y as u32,
            region_width,
            region_height,
            intensity.min(1.0),
        )]
    }

    /// Get difference image between current and previous frame.
    ///
    /// # Errors
    ///
    /// Returns an error if no previous frame exists.
    pub fn frame_difference(&self, frame: &[u8]) -> CvResult<Vec<u8>> {
        let expected_size = self.width as usize * self.height as usize;
        if frame.len() < expected_size {
            return Err(CvError::insufficient_data(expected_size, frame.len()));
        }

        let prev = self
            .previous_frame
            .as_ref()
            .ok_or_else(|| CvError::detection_failed("No previous frame"))?;

        let mut diff = vec![0u8; expected_size];
        for i in 0..expected_size {
            diff[i] = (frame[i] as i16 - prev[i] as i16).unsigned_abs() as u8;
        }

        Ok(diff)
    }
}

/// Optical flow using Lucas-Kanade method.
#[derive(Debug, Clone)]
pub struct OpticalFlowLK {
    /// Window size for local flow computation.
    window_size: u32,
    /// Maximum number of pyramid levels.
    #[allow(dead_code)]
    max_level: u32,
    /// Convergence criteria - maximum iterations.
    max_iterations: u32,
    /// Convergence criteria - epsilon.
    epsilon: f64,
}

impl OpticalFlowLK {
    /// Create a new Lucas-Kanade optical flow calculator.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::detect::motion::OpticalFlowLK;
    ///
    /// let flow = OpticalFlowLK::new(15, 3);
    /// ```
    #[must_use]
    pub fn new(window_size: u32, max_level: u32) -> Self {
        Self {
            window_size,
            max_level,
            max_iterations: 30,
            epsilon: 0.01,
        }
    }

    /// Set maximum iterations for convergence.
    #[must_use]
    pub const fn with_max_iterations(mut self, iterations: u32) -> Self {
        self.max_iterations = iterations;
        self
    }

    /// Set epsilon for convergence.
    #[must_use]
    pub const fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Calculate optical flow between two frames.
    ///
    /// # Arguments
    ///
    /// * `prev` - Previous grayscale frame
    /// * `curr` - Current grayscale frame
    /// * `width` - Frame width
    /// * `height` - Frame height
    /// * `points` - Points to track (x, y coordinates)
    ///
    /// # Returns
    ///
    /// Tuple of (new positions, status flags, errors).
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions are invalid.
    #[allow(clippy::type_complexity)]
    pub fn calculate(
        &self,
        prev: &[u8],
        curr: &[u8],
        width: u32,
        height: u32,
        points: &[(f32, f32)],
    ) -> CvResult<(Vec<(f32, f32)>, Vec<bool>, Vec<f32>)> {
        if width == 0 || height == 0 {
            return Err(CvError::invalid_dimensions(width, height));
        }

        let expected_size = width as usize * height as usize;
        if prev.len() < expected_size || curr.len() < expected_size {
            return Err(CvError::insufficient_data(
                expected_size,
                prev.len().min(curr.len()),
            ));
        }

        let mut new_points = Vec::with_capacity(points.len());
        let mut status = Vec::with_capacity(points.len());
        let mut errors = Vec::with_capacity(points.len());

        let half_win = self.window_size as i32 / 2;
        let w = width as i32;
        let h = height as i32;

        for &(px, py) in points {
            let x = px as i32;
            let y = py as i32;

            // Check if point is within valid range
            if x < half_win || x >= w - half_win || y < half_win || y >= h - half_win {
                new_points.push((px, py));
                status.push(false);
                errors.push(f32::MAX);
                continue;
            }

            // Compute spatial gradients and temporal difference
            let (gxx, gxy, gyy, bx, by) =
                self.compute_gradient_matrix(prev, curr, width, x, y, half_win);

            // Solve 2x2 linear system: [gxx gxy; gxy gyy] * [u; v] = [bx; by]
            let det = gxx * gyy - gxy * gxy;
            if det.abs() < f64::EPSILON {
                new_points.push((px, py));
                status.push(false);
                errors.push(f32::MAX);
                continue;
            }

            let u = (gyy * bx - gxy * by) / det;
            let v = (gxx * by - gxy * bx) / det;

            let new_x = px + u as f32;
            let new_y = py + v as f32;

            // Check bounds
            if new_x < 0.0 || new_x >= width as f32 || new_y < 0.0 || new_y >= height as f32 {
                new_points.push((px, py));
                status.push(false);
                errors.push(f32::MAX);
                continue;
            }

            let error = (u * u + v * v).sqrt() as f32;
            new_points.push((new_x, new_y));
            status.push(true);
            errors.push(error);
        }

        Ok((new_points, status, errors))
    }

    /// Compute gradient matrix for Lucas-Kanade.
    fn compute_gradient_matrix(
        &self,
        prev: &[u8],
        curr: &[u8],
        width: u32,
        cx: i32,
        cy: i32,
        half_win: i32,
    ) -> (f64, f64, f64, f64, f64) {
        let w = width as i32;
        let mut gxx = 0.0;
        let mut gxy = 0.0;
        let mut gyy = 0.0;
        let mut bx = 0.0;
        let mut by = 0.0;

        for dy in -half_win..=half_win {
            for dx in -half_win..=half_win {
                let x = cx + dx;
                let y = cy + dy;
                let idx = (y * w + x) as usize;

                // Compute spatial gradients using Sobel-like operator
                let idx_l = (y * w + (x - 1).max(0)) as usize;
                let idx_r = (y * w + (x + 1).min(w - 1)) as usize;
                let idx_t = ((y - 1).max(0) * w + x) as usize;
                let idx_b = ((y + 1).min((width as i32) - 1) * w + x) as usize;

                let ix = (prev[idx_r] as f64 - prev[idx_l] as f64) / 2.0;
                let iy = (prev[idx_b] as f64 - prev[idx_t] as f64) / 2.0;

                // Temporal gradient
                let it = curr[idx] as f64 - prev[idx] as f64;

                gxx += ix * ix;
                gxy += ix * iy;
                gyy += iy * iy;
                bx += -ix * it;
                by += -iy * it;
            }
        }

        (gxx, gxy, gyy, bx, by)
    }
}

/// Flow vector field representation.
#[derive(Debug, Clone)]
pub struct FlowField {
    /// X component of flow.
    pub u: Vec<f32>,
    /// Y component of flow.
    pub v: Vec<f32>,
    /// Field width.
    pub width: u32,
    /// Field height.
    pub height: u32,
}

impl FlowField {
    /// Create a new empty flow field.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::detect::motion::FlowField;
    ///
    /// let field = FlowField::new(640, 480);
    /// ```
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        let size = width as usize * height as usize;
        Self {
            u: vec![0.0; size],
            v: vec![0.0; size],
            width,
            height,
        }
    }

    /// Get flow magnitude at a point.
    #[must_use]
    pub fn magnitude(&self, x: u32, y: u32) -> f32 {
        let idx = (y * self.width + x) as usize;
        if idx < self.u.len() {
            (self.u[idx] * self.u[idx] + self.v[idx] * self.v[idx]).sqrt()
        } else {
            0.0
        }
    }

    /// Get flow direction at a point (radians).
    #[must_use]
    pub fn direction(&self, x: u32, y: u32) -> f32 {
        let idx = (y * self.width + x) as usize;
        if idx < self.u.len() {
            self.v[idx].atan2(self.u[idx])
        } else {
            0.0
        }
    }

    /// Get average flow magnitude.
    #[must_use]
    pub fn average_magnitude(&self) -> f32 {
        let mut sum = 0.0f32;
        for i in 0..self.u.len() {
            sum += (self.u[i] * self.u[i] + self.v[i] * self.v[i]).sqrt();
        }
        sum / self.u.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_motion_region() {
        let region = MotionRegion::new(10, 20, 100, 100, 0.8);
        assert_eq!(region.x, 10);
        assert_eq!(region.y, 20);
        assert_eq!(region.area(), 10000);
        assert!((region.centroid.0 - 60.0).abs() < 0.001);
    }

    #[test]
    fn test_motion_detector_new() {
        let detector = MotionDetector::new(640, 480);
        assert_eq!(detector.width, 640);
        assert_eq!(detector.height, 480);
    }

    #[test]
    fn test_motion_detector_process() {
        let mut detector = MotionDetector::new(10, 10);

        // First frame - should initialize
        let frame1 = vec![100u8; 100];
        let (mask1, regions1) = detector.process(&frame1).expect("process should succeed");
        assert_eq!(mask1.len(), 100);
        assert!(regions1.is_empty());

        // Second frame - same as first, no motion
        let frame2 = vec![100u8; 100];
        let (mask2, regions2) = detector.process(&frame2).expect("process should succeed");
        assert_eq!(mask2.len(), 100);
        assert!(regions2.is_empty());
    }

    #[test]
    fn test_motion_detector_with_motion() {
        let mut detector = MotionDetector::new(10, 10)
            .with_threshold(10)
            .with_min_area(1);

        // First frame
        let frame1 = vec![100u8; 100];
        detector.process(&frame1).expect("process should succeed");

        // Second frame with significant change
        let mut frame2 = vec![100u8; 100];
        for i in 40..60 {
            frame2[i] = 200; // Create motion in the middle
        }
        let (_mask, regions) = detector.process(&frame2).expect("process should succeed");

        // Should detect motion
        assert!(!regions.is_empty());
    }

    #[test]
    fn test_motion_detector_reset() {
        let mut detector = MotionDetector::new(10, 10);
        let frame = vec![100u8; 100];
        detector.process(&frame).expect("process should succeed");

        detector.reset();
        assert!(detector.previous_frame.is_none());
        assert!(detector.background.is_none());
    }

    #[test]
    fn test_optical_flow_new() {
        let flow = OpticalFlowLK::new(15, 3);
        assert_eq!(flow.window_size, 15);
        assert_eq!(flow.max_level, 3);
    }

    #[test]
    fn test_optical_flow_calculate() {
        let flow = OpticalFlowLK::new(5, 1);

        let prev = vec![100u8; 100];
        let curr = vec![100u8; 100];
        let points = vec![(5.0f32, 5.0f32)];

        let (new_points, status, errors) = flow
            .calculate(&prev, &curr, 10, 10, &points)
            .expect("calculate should succeed");

        assert_eq!(new_points.len(), 1);
        assert_eq!(status.len(), 1);
        assert_eq!(errors.len(), 1);
    }

    #[test]
    fn test_flow_field() {
        let field = FlowField::new(10, 10);
        assert_eq!(field.u.len(), 100);
        assert_eq!(field.v.len(), 100);
        assert!(field.magnitude(5, 5) < 0.001);
    }
}
