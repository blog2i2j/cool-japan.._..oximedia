//! Corner detection algorithms.
//!
//! This module provides corner detection methods including:
//!
//! - Harris corner detector
//! - Shi-Tomasi (Good Features to Track)
//! - FAST (Features from Accelerated Segment Test)
//!
//! # Example
//!
//! ```
//! use oximedia_cv::detect::corner::{HarrisDetector, CornerDetector};
//!
//! let detector = HarrisDetector::new();
//! ```

use crate::error::{CvError, CvResult};

/// Corner point with response strength.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Corner {
    /// X coordinate.
    pub x: u32,
    /// Y coordinate.
    pub y: u32,
    /// Corner response strength.
    pub response: f64,
}

impl Corner {
    /// Create a new corner.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::detect::corner::Corner;
    ///
    /// let corner = Corner::new(100, 150, 0.85);
    /// ```
    #[must_use]
    pub const fn new(x: u32, y: u32, response: f64) -> Self {
        Self { x, y, response }
    }
}

/// Trait for corner detectors.
pub trait CornerDetector {
    /// Detect corners in a grayscale image.
    ///
    /// # Arguments
    ///
    /// * `image` - Grayscale image data
    /// * `width` - Image width
    /// * `height` - Image height
    ///
    /// # Returns
    ///
    /// Vector of detected corners.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions are invalid.
    fn detect(&self, image: &[u8], width: u32, height: u32) -> CvResult<Vec<Corner>>;
}

/// Harris corner detector.
///
/// Detects corners using the Harris corner response function.
///
/// # Examples
///
/// ```
/// use oximedia_cv::detect::corner::{HarrisDetector, CornerDetector};
///
/// let detector = HarrisDetector::new();
/// ```
#[derive(Debug, Clone)]
pub struct HarrisDetector {
    /// Block size for computing derivatives.
    block_size: usize,
    /// Aperture parameter for Sobel operator.
    aperture_size: usize,
    /// Harris detector free parameter k (typically 0.04-0.06).
    k: f64,
    /// Threshold for corner response.
    threshold: f64,
}

impl HarrisDetector {
    /// Create a new Harris corner detector.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::detect::corner::HarrisDetector;
    ///
    /// let detector = HarrisDetector::new();
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            block_size: 5,
            aperture_size: 3,
            k: 0.04,
            threshold: 0.01,
        }
    }

    /// Set block size.
    #[must_use]
    pub const fn with_block_size(mut self, size: usize) -> Self {
        self.block_size = size;
        self
    }

    /// Set Harris parameter k.
    #[must_use]
    pub const fn with_k(mut self, k: f64) -> Self {
        self.k = k;
        self
    }

    /// Set threshold.
    #[must_use]
    pub const fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }
}

impl Default for HarrisDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl CornerDetector for HarrisDetector {
    fn detect(&self, image: &[u8], width: u32, height: u32) -> CvResult<Vec<Corner>> {
        if width == 0 || height == 0 {
            return Err(CvError::invalid_dimensions(width, height));
        }

        let size = width as usize * height as usize;
        if image.len() < size {
            return Err(CvError::insufficient_data(size, image.len()));
        }

        // Compute gradients
        let gx = compute_sobel_x(image, width, height);
        let gy = compute_sobel_y(image, width, height);

        // Compute structure tensor components
        let mut gxx = vec![0.0; size];
        let mut gxy = vec![0.0; size];
        let mut gyy = vec![0.0; size];

        for i in 0..size {
            let ix = gx[i] as f64;
            let iy = gy[i] as f64;
            gxx[i] = ix * ix;
            gxy[i] = ix * iy;
            gyy[i] = iy * iy;
        }

        // Apply Gaussian smoothing to structure tensor
        let gxx_smooth = box_filter(&gxx, width, height, self.block_size);
        let gxy_smooth = box_filter(&gxy, width, height, self.block_size);
        let gyy_smooth = box_filter(&gyy, width, height, self.block_size);

        // Compute Harris corner response
        let mut response = vec![0.0; size];
        let mut max_response = 0.0f64;

        for i in 0..size {
            let a = gxx_smooth[i];
            let b = gxy_smooth[i];
            let c = gyy_smooth[i];

            // R = det(M) - k * trace(M)^2
            let det = a * c - b * b;
            let trace = a + c;
            let r = det - self.k * trace * trace;

            response[i] = r;
            max_response = max_response.max(r);
        }

        // Find local maxima above threshold
        // If max_response is zero (e.g. uniform image), return empty (no corners)
        if max_response <= 0.0 {
            return Ok(Vec::new());
        }
        let corners = find_corners_nms(&response, width, height, max_response * self.threshold);

        Ok(corners)
    }
}

/// Shi-Tomasi corner detector (Good Features to Track).
///
/// Uses minimum eigenvalue of the structure tensor as corner response.
///
/// # Examples
///
/// ```
/// use oximedia_cv::detect::corner::{ShiTomasiDetector, CornerDetector};
///
/// let detector = ShiTomasiDetector::new();
/// ```
#[derive(Debug, Clone)]
pub struct ShiTomasiDetector {
    /// Block size for computing derivatives.
    block_size: usize,
    /// Quality level (fraction of maximum response).
    quality_level: f64,
    /// Minimum distance between corners.
    min_distance: f64,
}

impl ShiTomasiDetector {
    /// Create a new Shi-Tomasi detector.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::detect::corner::ShiTomasiDetector;
    ///
    /// let detector = ShiTomasiDetector::new();
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            block_size: 5,
            quality_level: 0.01,
            min_distance: 10.0,
        }
    }

    /// Set block size.
    #[must_use]
    pub const fn with_block_size(mut self, size: usize) -> Self {
        self.block_size = size;
        self
    }

    /// Set quality level.
    #[must_use]
    pub const fn with_quality_level(mut self, level: f64) -> Self {
        self.quality_level = level;
        self
    }

    /// Set minimum distance.
    #[must_use]
    pub const fn with_min_distance(mut self, distance: f64) -> Self {
        self.min_distance = distance;
        self
    }
}

impl Default for ShiTomasiDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl CornerDetector for ShiTomasiDetector {
    fn detect(&self, image: &[u8], width: u32, height: u32) -> CvResult<Vec<Corner>> {
        if width == 0 || height == 0 {
            return Err(CvError::invalid_dimensions(width, height));
        }

        let size = width as usize * height as usize;
        if image.len() < size {
            return Err(CvError::insufficient_data(size, image.len()));
        }

        // Compute gradients
        let gx = compute_sobel_x(image, width, height);
        let gy = compute_sobel_y(image, width, height);

        // Compute structure tensor
        let mut gxx = vec![0.0; size];
        let mut gxy = vec![0.0; size];
        let mut gyy = vec![0.0; size];

        for i in 0..size {
            let ix = gx[i] as f64;
            let iy = gy[i] as f64;
            gxx[i] = ix * ix;
            gxy[i] = ix * iy;
            gyy[i] = iy * iy;
        }

        // Apply smoothing
        let gxx_smooth = box_filter(&gxx, width, height, self.block_size);
        let gxy_smooth = box_filter(&gxy, width, height, self.block_size);
        let gyy_smooth = box_filter(&gyy, width, height, self.block_size);

        // Compute minimum eigenvalue
        let mut response = vec![0.0; size];
        let mut max_response = 0.0f64;

        for i in 0..size {
            let a = gxx_smooth[i];
            let b = gxy_smooth[i];
            let c = gyy_smooth[i];

            // Minimum eigenvalue: (trace - sqrt(trace^2 - 4*det)) / 2
            let trace = a + c;
            let det = a * c - b * b;
            let discriminant = trace * trace - 4.0 * det;

            let min_eigenvalue = if discriminant >= 0.0 {
                (trace - discriminant.sqrt()) / 2.0
            } else {
                0.0
            };

            response[i] = min_eigenvalue.max(0.0);
            max_response = max_response.max(response[i]);
        }

        // Find corners with NMS and minimum distance
        let mut corners =
            find_corners_nms(&response, width, height, max_response * self.quality_level);

        // Apply minimum distance constraint
        corners = apply_min_distance(&corners, self.min_distance);

        Ok(corners)
    }
}

/// FAST (Features from Accelerated Segment Test) corner detector.
///
/// Fast corner detection using circle-based feature detection.
///
/// # Examples
///
/// ```
/// use oximedia_cv::detect::corner::{FastDetector, CornerDetector};
///
/// let detector = FastDetector::new();
/// ```
#[derive(Debug, Clone)]
pub struct FastDetector {
    /// Threshold for intensity difference.
    threshold: u8,
    /// Whether to use non-maximum suppression.
    non_max_suppression: bool,
}

impl FastDetector {
    /// Create a new FAST detector.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::detect::corner::FastDetector;
    ///
    /// let detector = FastDetector::new();
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            threshold: 20,
            non_max_suppression: true,
        }
    }

    /// Set threshold.
    #[must_use]
    pub const fn with_threshold(mut self, threshold: u8) -> Self {
        self.threshold = threshold;
        self
    }

    /// Set whether to use non-maximum suppression.
    #[must_use]
    pub const fn with_nms(mut self, nms: bool) -> Self {
        self.non_max_suppression = nms;
        self
    }
}

impl Default for FastDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl CornerDetector for FastDetector {
    fn detect(&self, image: &[u8], width: u32, height: u32) -> CvResult<Vec<Corner>> {
        if width == 0 || height == 0 {
            return Err(CvError::invalid_dimensions(width, height));
        }

        let size = width as usize * height as usize;
        if image.len() < size {
            return Err(CvError::insufficient_data(size, image.len()));
        }

        // FAST-9 circle offsets (Bresenham circle with radius 3)
        let circle: [(i32, i32); 16] = [
            (0, 3),
            (1, 3),
            (2, 2),
            (3, 1),
            (3, 0),
            (3, -1),
            (2, -2),
            (1, -3),
            (0, -3),
            (-1, -3),
            (-2, -2),
            (-3, -1),
            (-3, 0),
            (-3, 1),
            (-2, 2),
            (-1, 3),
        ];

        let w = width as i32;
        let h = height as i32;
        let threshold = self.threshold as i32;

        let mut corners = Vec::new();

        // Detect FAST corners
        for y in 3..(h - 3) {
            for x in 3..(w - 3) {
                let center_idx = (y * w + x) as usize;
                let center_val = image[center_idx] as i32;

                // Check if this is a corner
                let mut brighter = 0u32;
                let mut darker = 0u32;

                for &(dx, dy) in &circle {
                    let px = x + dx;
                    let py = y + dy;
                    let idx = (py * w + px) as usize;
                    let val = image[idx] as i32;

                    let diff = val - center_val;

                    if diff > threshold {
                        brighter += 1;
                        darker = 0;
                    } else if diff < -threshold {
                        darker += 1;
                        brighter = 0;
                    } else {
                        brighter = 0;
                        darker = 0;
                    }

                    // Need 9 consecutive pixels
                    if brighter >= 9 || darker >= 9 {
                        break;
                    }
                }

                if brighter >= 9 || darker >= 9 {
                    // Compute corner score (sum of absolute differences)
                    let mut score = 0.0;
                    for &(dx, dy) in &circle {
                        let px = x + dx;
                        let py = y + dy;
                        let idx = (py * w + px) as usize;
                        let diff = (image[idx] as i32 - center_val).abs();
                        score += diff as f64;
                    }

                    corners.push(Corner::new(x as u32, y as u32, score));
                }
            }
        }

        // Apply non-maximum suppression if enabled
        if self.non_max_suppression {
            corners = nms_fast_corners(&corners, width, height);
        }

        Ok(corners)
    }
}

/// Compute Sobel X gradient.
fn compute_sobel_x(image: &[u8], width: u32, height: u32) -> Vec<i16> {
    let w = width as usize;
    let h = height as usize;
    let mut result = vec![0i16; w * h];

    for y in 0..h {
        for x in 0..w {
            let x0 = x.saturating_sub(1);
            let x2 = (x + 1).min(w - 1);
            let y0 = y.saturating_sub(1);
            let y2 = (y + 1).min(h - 1);

            let p00 = image[y0 * w + x0] as i16;
            let p01 = image[y * w + x0] as i16;
            let p02 = image[y2 * w + x0] as i16;
            let p20 = image[y0 * w + x2] as i16;
            let p21 = image[y * w + x2] as i16;
            let p22 = image[y2 * w + x2] as i16;

            result[y * w + x] = -p00 - 2 * p01 - p02 + p20 + 2 * p21 + p22;
        }
    }

    result
}

/// Compute Sobel Y gradient.
fn compute_sobel_y(image: &[u8], width: u32, height: u32) -> Vec<i16> {
    let w = width as usize;
    let h = height as usize;
    let mut result = vec![0i16; w * h];

    for y in 0..h {
        for x in 0..w {
            let x0 = x.saturating_sub(1);
            let x2 = (x + 1).min(w - 1);
            let y0 = y.saturating_sub(1);
            let y2 = (y + 1).min(h - 1);

            let p00 = image[y0 * w + x0] as i16;
            let p10 = image[y0 * w + x] as i16;
            let p20 = image[y0 * w + x2] as i16;
            let p02 = image[y2 * w + x0] as i16;
            let p12 = image[y2 * w + x] as i16;
            let p22 = image[y2 * w + x2] as i16;

            result[y * w + x] = -p00 - 2 * p10 - p20 + p02 + 2 * p12 + p22;
        }
    }

    result
}

/// Box filter for smoothing.
fn box_filter(data: &[f64], width: u32, height: u32, kernel_size: usize) -> Vec<f64> {
    let w = width as usize;
    let h = height as usize;
    let half = kernel_size / 2;
    let mut result = vec![0.0; w * h];

    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0;
            let mut count = 0.0;

            for dy in 0..kernel_size {
                let py = (y + dy).saturating_sub(half);
                if py >= h {
                    continue;
                }

                for dx in 0..kernel_size {
                    let px = (x + dx).saturating_sub(half);
                    if px >= w {
                        continue;
                    }

                    sum += data[py * w + px];
                    count += 1.0;
                }
            }

            result[y * w + x] = if count > 0.0 { sum / count } else { 0.0 };
        }
    }

    result
}

/// Find corners using non-maximum suppression.
fn find_corners_nms(response: &[f64], width: u32, height: u32, threshold: f64) -> Vec<Corner> {
    let w = width as i32;
    let h = height as i32;
    let mut corners = Vec::new();

    for y in 1..(h - 1) {
        for x in 1..(w - 1) {
            let idx = (y * w + x) as usize;
            let val = response[idx];

            if val < threshold {
                continue;
            }

            // Check 3x3 neighborhood
            let mut is_max = true;
            'outer: for dy in -1..=1 {
                for dx in -1..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }

                    let nidx = ((y + dy) * w + (x + dx)) as usize;
                    if response[nidx] > val {
                        is_max = false;
                        break 'outer;
                    }
                }
            }

            if is_max {
                corners.push(Corner::new(x as u32, y as u32, val));
            }
        }
    }

    // Sort by response (descending)
    corners.sort_by(|a, b| {
        b.response
            .partial_cmp(&a.response)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    corners
}

/// Apply minimum distance constraint.
fn apply_min_distance(corners: &[Corner], min_distance: f64) -> Vec<Corner> {
    if min_distance <= 0.0 {
        return corners.to_vec();
    }

    let min_dist_sq = min_distance * min_distance;
    let mut result: Vec<Corner> = Vec::new();

    for corner in corners {
        let mut too_close = false;

        for accepted in &result {
            let dx = corner.x as f64 - accepted.x as f64;
            let dy = corner.y as f64 - accepted.y as f64;
            let dist_sq = dx * dx + dy * dy;

            if dist_sq < min_dist_sq {
                too_close = true;
                break;
            }
        }

        if !too_close {
            result.push(*corner);
        }
    }

    result
}

/// Non-maximum suppression for FAST corners.
fn nms_fast_corners(corners: &[Corner], width: u32, height: u32) -> Vec<Corner> {
    if corners.is_empty() {
        return Vec::new();
    }

    let w = width as usize;
    let h = height as usize;

    // Create score map
    let mut score_map = vec![0.0; w * h];
    for corner in corners {
        let idx = corner.y as usize * w + corner.x as usize;
        if idx < score_map.len() {
            score_map[idx] = corner.response;
        }
    }

    // Apply NMS
    let mut result: Vec<Corner> = Vec::new();

    for corner in corners {
        let x = corner.x as i32;
        let y = corner.y as i32;
        let idx = (y * width as i32 + x) as usize;
        let val = score_map[idx];

        if val == 0.0 {
            continue;
        }

        // Check 3x3 neighborhood
        let mut is_max = true;
        for dy in -1..=1 {
            for dx in -1..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }

                let nx = x + dx;
                let ny = y + dy;

                if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                    let nidx = (ny * width as i32 + nx) as usize;
                    if nidx < score_map.len() && score_map[nidx] > val {
                        is_max = false;
                        break;
                    }
                }
            }
            if !is_max {
                break;
            }
        }

        if is_max {
            result.push(*corner);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_corner_new() {
        let corner = Corner::new(100, 150, 0.85);
        assert_eq!(corner.x, 100);
        assert_eq!(corner.y, 150);
        assert!((corner.response - 0.85).abs() < f64::EPSILON);
    }

    #[test]
    fn test_harris_detector_new() {
        let detector = HarrisDetector::new();
        assert_eq!(detector.block_size, 5);
        assert!((detector.k - 0.04).abs() < f64::EPSILON);
    }

    #[test]
    fn test_harris_detector_with_params() {
        let detector = HarrisDetector::new()
            .with_block_size(7)
            .with_k(0.05)
            .with_threshold(0.02);

        assert_eq!(detector.block_size, 7);
        assert!((detector.k - 0.05).abs() < f64::EPSILON);
    }

    #[test]
    fn test_harris_detect() {
        let detector = HarrisDetector::new();
        let image = vec![100u8; 100];

        let corners = detector
            .detect(&image, 10, 10)
            .expect("detect should succeed");
        // Uniform image should have no corners
        assert!(corners.is_empty() || corners.len() < 5);
    }

    #[test]
    fn test_shi_tomasi_detector_new() {
        let detector = ShiTomasiDetector::new();
        assert_eq!(detector.block_size, 5);
        assert!((detector.quality_level - 0.01).abs() < f64::EPSILON);
    }

    #[test]
    fn test_shi_tomasi_detect() {
        let detector = ShiTomasiDetector::new();
        let image = vec![100u8; 100];

        let corners = detector
            .detect(&image, 10, 10)
            .expect("detect should succeed");
        assert!(corners.is_empty() || corners.len() < 5);
    }

    #[test]
    fn test_fast_detector_new() {
        let detector = FastDetector::new();
        assert_eq!(detector.threshold, 20);
        assert!(detector.non_max_suppression);
    }

    #[test]
    fn test_fast_detector_with_params() {
        let detector = FastDetector::new().with_threshold(30).with_nms(false);

        assert_eq!(detector.threshold, 30);
        assert!(!detector.non_max_suppression);
    }

    #[test]
    fn test_fast_detect() {
        let detector = FastDetector::new();
        let image = vec![100u8; 100];

        let corners = detector
            .detect(&image, 10, 10)
            .expect("detect should succeed");
        // Uniform image should have no FAST corners
        assert!(corners.is_empty());
    }

    #[test]
    fn test_compute_sobel() {
        let image = vec![100u8; 100];
        let gx = compute_sobel_x(&image, 10, 10);
        let gy = compute_sobel_y(&image, 10, 10);

        assert_eq!(gx.len(), 100);
        assert_eq!(gy.len(), 100);

        // Uniform image should have zero gradient
        for &val in &gx {
            assert_eq!(val, 0);
        }
    }

    #[test]
    fn test_box_filter() {
        let data = vec![1.0; 100];
        let filtered = box_filter(&data, 10, 10, 3);

        assert_eq!(filtered.len(), 100);
        // Uniform data should remain uniform
        for &val in &filtered {
            assert!((val - 1.0).abs() < 0.001);
        }
    }

    #[test]
    fn test_apply_min_distance() {
        let corners = vec![
            Corner::new(10, 10, 1.0),
            Corner::new(12, 12, 0.9),
            Corner::new(50, 50, 0.8),
        ];

        let filtered = apply_min_distance(&corners, 5.0);
        // Second corner should be filtered (too close to first)
        assert_eq!(filtered.len(), 2);
    }
}
