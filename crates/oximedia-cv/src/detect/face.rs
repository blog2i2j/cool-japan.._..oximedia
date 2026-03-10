//! Face detection module.
//!
//! This module provides face detection functionality using Haar cascade
//! classifiers with integral image computation for fast feature evaluation.
//!
//! # Example
//!
//! ```
//! use oximedia_cv::detect::{IntegralImage, DetectionResult};
//!
//! let image = vec![100u8; 100];
//! let integral = IntegralImage::compute(&image, 10, 10);
//! assert_eq!(integral.width(), 10);
//! ```

use crate::error::{CvError, CvResult};

/// Trait for face detection algorithms.
pub trait FaceDetector {
    /// Detect faces in an image.
    ///
    /// # Arguments
    ///
    /// * `image` - Grayscale image data
    /// * `width` - Image width
    /// * `height` - Image height
    ///
    /// # Returns
    ///
    /// Vector of detected face regions.
    ///
    /// # Errors
    ///
    /// Returns an error if detection fails.
    fn detect(&self, image: &[u8], width: u32, height: u32) -> CvResult<Vec<FaceRegion>>;

    /// Set minimum face size for detection.
    fn set_min_size(&mut self, width: u32, height: u32);

    /// Set maximum face size for detection.
    fn set_max_size(&mut self, width: u32, height: u32);
}

/// Detection result with bounding box and confidence.
#[derive(Debug, Clone, Copy)]
pub struct DetectionResult {
    /// X coordinate of top-left corner.
    pub x: u32,
    /// Y coordinate of top-left corner.
    pub y: u32,
    /// Width of detection box.
    pub width: u32,
    /// Height of detection box.
    pub height: u32,
    /// Detection confidence (0.0 - 1.0).
    pub confidence: f64,
}

impl DetectionResult {
    /// Create a new detection result.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::detect::DetectionResult;
    ///
    /// let result = DetectionResult::new(10, 20, 100, 100, 0.95);
    /// assert_eq!(result.x, 10);
    /// assert!(result.confidence > 0.9);
    /// ```
    #[must_use]
    pub const fn new(x: u32, y: u32, width: u32, height: u32, confidence: f64) -> Self {
        Self {
            x,
            y,
            width,
            height,
            confidence,
        }
    }

    /// Get the center point of the detection.
    #[must_use]
    pub const fn center(&self) -> (u32, u32) {
        (self.x + self.width / 2, self.y + self.height / 2)
    }

    /// Get the area of the detection box.
    #[must_use]
    pub const fn area(&self) -> u32 {
        self.width * self.height
    }
}

/// Face region with optional landmarks.
#[derive(Debug, Clone)]
pub struct FaceRegion {
    /// Bounding box of the face.
    pub bbox: DetectionResult,
    /// Facial landmarks (optional).
    pub landmarks: Option<FaceLandmarks>,
}

impl FaceRegion {
    /// Create a new face region without landmarks.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::detect::{FaceRegion, DetectionResult};
    ///
    /// let bbox = DetectionResult::new(10, 20, 100, 100, 0.95);
    /// let face = FaceRegion::new(bbox);
    /// assert!(face.landmarks.is_none());
    /// ```
    #[must_use]
    pub const fn new(bbox: DetectionResult) -> Self {
        Self {
            bbox,
            landmarks: None,
        }
    }

    /// Create a face region with landmarks.
    #[must_use]
    pub const fn with_landmarks(bbox: DetectionResult, landmarks: FaceLandmarks) -> Self {
        Self {
            bbox,
            landmarks: Some(landmarks),
        }
    }
}

/// Facial landmarks (5-point or 68-point).
#[derive(Debug, Clone)]
pub struct FaceLandmarks {
    /// Left eye center.
    pub left_eye: (f32, f32),
    /// Right eye center.
    pub right_eye: (f32, f32),
    /// Nose tip.
    pub nose: (f32, f32),
    /// Left mouth corner.
    pub mouth_left: (f32, f32),
    /// Right mouth corner.
    pub mouth_right: (f32, f32),
    /// Additional landmarks (for 68-point models).
    pub extra: Vec<(f32, f32)>,
}

impl FaceLandmarks {
    /// Create 5-point facial landmarks.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::detect::face::FaceLandmarks;
    ///
    /// let landmarks = FaceLandmarks::five_point(
    ///     (30.0, 40.0),
    ///     (70.0, 40.0),
    ///     (50.0, 60.0),
    ///     (35.0, 80.0),
    ///     (65.0, 80.0),
    /// );
    /// ```
    #[must_use]
    pub fn five_point(
        left_eye: (f32, f32),
        right_eye: (f32, f32),
        nose: (f32, f32),
        mouth_left: (f32, f32),
        mouth_right: (f32, f32),
    ) -> Self {
        Self {
            left_eye,
            right_eye,
            nose,
            mouth_left,
            mouth_right,
            extra: Vec::new(),
        }
    }

    /// Calculate eye distance (inter-pupillary distance).
    #[must_use]
    pub fn eye_distance(&self) -> f32 {
        let dx = self.right_eye.0 - self.left_eye.0;
        let dy = self.right_eye.1 - self.left_eye.1;
        (dx * dx + dy * dy).sqrt()
    }
}

/// Integral image for fast feature computation.
///
/// An integral image allows O(1) computation of the sum of any rectangular
/// region, which is essential for efficient Haar feature evaluation.
#[derive(Debug, Clone)]
pub struct IntegralImage {
    /// Integral image data (width+1 x height+1).
    data: Vec<u64>,
    /// Squared integral image for variance computation.
    squared: Vec<u64>,
    /// Original image width.
    width: u32,
    /// Original image height.
    height: u32,
}

impl IntegralImage {
    /// Compute integral image from grayscale image.
    ///
    /// # Arguments
    ///
    /// * `image` - Grayscale image data
    /// * `width` - Image width
    /// * `height` - Image height
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::detect::IntegralImage;
    ///
    /// let image = vec![100u8; 100];
    /// let integral = IntegralImage::compute(&image, 10, 10);
    /// ```
    #[must_use]
    pub fn compute(image: &[u8], width: u32, height: u32) -> Self {
        let w = width as usize;
        let h = height as usize;
        let iw = w + 1;
        let ih = h + 1;

        let mut data = vec![0u64; iw * ih];
        let mut squared = vec![0u64; iw * ih];

        for y in 0..h {
            for x in 0..w {
                let pixel = image[y * w + x] as u64;
                let idx = (y + 1) * iw + (x + 1);

                data[idx] =
                    pixel + data[y * iw + (x + 1)] + data[(y + 1) * iw + x] - data[y * iw + x];

                squared[idx] =
                    pixel * pixel + squared[y * iw + (x + 1)] + squared[(y + 1) * iw + x]
                        - squared[y * iw + x];
            }
        }

        Self {
            data,
            squared,
            width,
            height,
        }
    }

    /// Get original image width.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Get original image height.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Get sum of a rectangular region.
    ///
    /// # Arguments
    ///
    /// * `x` - Left coordinate
    /// * `y` - Top coordinate
    /// * `w` - Region width
    /// * `h` - Region height
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::detect::IntegralImage;
    ///
    /// let image = vec![1u8; 100];
    /// let integral = IntegralImage::compute(&image, 10, 10);
    /// let sum = integral.sum(0, 0, 5, 5);
    /// assert_eq!(sum, 25); // 5x5 region of 1s
    /// ```
    #[must_use]
    pub fn sum(&self, x: u32, y: u32, w: u32, h: u32) -> u64 {
        let x0 = x as usize;
        let y0 = y as usize;
        let x1 = (x + w).min(self.width) as usize;
        let y1 = (y + h).min(self.height) as usize;
        let iw = self.width as usize + 1;

        self.data[y1 * iw + x1] + self.data[y0 * iw + x0]
            - self.data[y0 * iw + x1]
            - self.data[y1 * iw + x0]
    }

    /// Get squared sum of a rectangular region (for variance).
    #[must_use]
    pub fn squared_sum(&self, x: u32, y: u32, w: u32, h: u32) -> u64 {
        let x0 = x as usize;
        let y0 = y as usize;
        let x1 = (x + w).min(self.width) as usize;
        let y1 = (y + h).min(self.height) as usize;
        let iw = self.width as usize + 1;

        self.squared[y1 * iw + x1] + self.squared[y0 * iw + x0]
            - self.squared[y0 * iw + x1]
            - self.squared[y1 * iw + x0]
    }

    /// Get mean value of a rectangular region.
    #[must_use]
    pub fn mean(&self, x: u32, y: u32, w: u32, h: u32) -> f64 {
        let area = w as f64 * h as f64;
        if area > 0.0 {
            self.sum(x, y, w, h) as f64 / area
        } else {
            0.0
        }
    }

    /// Get variance of a rectangular region.
    #[must_use]
    pub fn variance(&self, x: u32, y: u32, w: u32, h: u32) -> f64 {
        let area = w as f64 * h as f64;
        if area > 0.0 {
            let sum = self.sum(x, y, w, h) as f64;
            let sq_sum = self.squared_sum(x, y, w, h) as f64;
            let mean = sum / area;
            (sq_sum / area) - (mean * mean)
        } else {
            0.0
        }
    }
}

/// Haar cascade classifier for face detection.
///
/// This is a skeleton implementation that provides the structure
/// for Haar cascade-based face detection.
#[derive(Debug, Clone)]
pub struct HaarCascade {
    /// Cascade stages.
    stages: Vec<CascadeStage>,
    /// Window width.
    window_width: u32,
    /// Window height.
    window_height: u32,
    /// Minimum detectable face size.
    min_size: (u32, u32),
    /// Maximum detectable face size.
    max_size: (u32, u32),
    /// Scale factor for multi-scale detection.
    scale_factor: f64,
    /// Minimum number of neighbors for grouping.
    min_neighbors: u32,
}

/// Cascade stage containing weak classifiers.
#[derive(Debug, Clone)]
struct CascadeStage {
    /// Weak classifiers in this stage.
    classifiers: Vec<WeakClassifier>,
    /// Stage threshold.
    threshold: f64,
}

/// Weak classifier based on Haar-like feature.
#[derive(Debug, Clone)]
struct WeakClassifier {
    /// Haar feature.
    feature: HaarFeature,
    /// Threshold for this classifier.
    threshold: f64,
    /// Value if feature < threshold.
    left_val: f64,
    /// Value if feature >= threshold.
    right_val: f64,
}

/// Haar-like feature.
#[derive(Debug, Clone)]
struct HaarFeature {
    /// Rectangular regions with weights.
    rects: Vec<HaarRect>,
}

/// Weighted rectangle for Haar feature.
#[derive(Debug, Clone)]
struct HaarRect {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    weight: f64,
}

impl HaarCascade {
    /// Create a new empty Haar cascade.
    ///
    /// In practice, cascades are loaded from trained model files.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::detect::HaarCascade;
    ///
    /// let cascade = HaarCascade::new(24, 24);
    /// ```
    #[must_use]
    pub fn new(window_width: u32, window_height: u32) -> Self {
        Self {
            stages: Vec::new(),
            window_width,
            window_height,
            min_size: (window_width, window_height),
            max_size: (0, 0), // 0 means no limit
            scale_factor: 1.1,
            min_neighbors: 3,
        }
    }

    /// Set scale factor for multi-scale detection.
    #[must_use]
    pub const fn with_scale_factor(mut self, factor: f64) -> Self {
        self.scale_factor = factor;
        self
    }

    /// Set minimum neighbors for grouping.
    #[must_use]
    pub const fn with_min_neighbors(mut self, min: u32) -> Self {
        self.min_neighbors = min;
        self
    }

    /// Evaluate cascade at a single location.
    fn evaluate(&self, integral: &IntegralImage, x: u32, y: u32, scale: f64) -> Option<f64> {
        let scaled_w = (self.window_width as f64 * scale) as u32;
        let scaled_h = (self.window_height as f64 * scale) as u32;

        // Check bounds
        if x + scaled_w > integral.width() || y + scaled_h > integral.height() {
            return None;
        }

        // Compute window variance for normalization
        let variance = integral.variance(x, y, scaled_w, scaled_h);
        if variance < f64::EPSILON {
            return None;
        }
        let std_dev = variance.sqrt();

        let mut total_score = 0.0;

        for stage in &self.stages {
            let mut stage_sum = 0.0;

            for classifier in &stage.classifiers {
                let feature_val = self.evaluate_feature(&classifier.feature, integral, x, y, scale);
                let normalized = feature_val / std_dev;

                stage_sum += if normalized < classifier.threshold {
                    classifier.left_val
                } else {
                    classifier.right_val
                };
            }

            if stage_sum < stage.threshold {
                return None; // Rejected by this stage
            }

            total_score += stage_sum;
        }

        Some(total_score)
    }

    /// Evaluate a single Haar feature.
    fn evaluate_feature(
        &self,
        feature: &HaarFeature,
        integral: &IntegralImage,
        x: u32,
        y: u32,
        scale: f64,
    ) -> f64 {
        let mut sum = 0.0;

        for rect in &feature.rects {
            let rx = x + (rect.x as f64 * scale) as u32;
            let ry = y + (rect.y as f64 * scale) as u32;
            let rw = (rect.width as f64 * scale) as u32;
            let rh = (rect.height as f64 * scale) as u32;

            sum += integral.sum(rx, ry, rw, rh) as f64 * rect.weight;
        }

        sum
    }
}

impl FaceDetector for HaarCascade {
    fn detect(&self, image: &[u8], width: u32, height: u32) -> CvResult<Vec<FaceRegion>> {
        if width == 0 || height == 0 {
            return Err(CvError::invalid_dimensions(width, height));
        }

        let expected_size = width as usize * height as usize;
        if image.len() < expected_size {
            return Err(CvError::insufficient_data(expected_size, image.len()));
        }

        let integral = IntegralImage::compute(image, width, height);
        let mut detections = Vec::new();

        // Multi-scale detection
        let mut scale = 1.0;
        let max_scale = (width as f64 / self.window_width as f64)
            .min(height as f64 / self.window_height as f64);

        while scale < max_scale {
            let scaled_w = (self.window_width as f64 * scale) as u32;
            let scaled_h = (self.window_height as f64 * scale) as u32;

            // Check size constraints
            if self.min_size.0 > 0 && scaled_w < self.min_size.0 {
                scale *= self.scale_factor;
                continue;
            }
            if self.max_size.0 > 0 && scaled_w > self.max_size.0 {
                break;
            }

            // Sliding window
            let step = (scale * 2.0).max(1.0) as u32;

            let mut y = 0;
            while y + scaled_h <= height {
                let mut x = 0;
                while x + scaled_w <= width {
                    if let Some(confidence) = self.evaluate(&integral, x, y, scale) {
                        let bbox = DetectionResult::new(
                            x,
                            y,
                            scaled_w,
                            scaled_h,
                            (confidence / 10.0).min(1.0), // Normalize confidence
                        );
                        detections.push(FaceRegion::new(bbox));
                    }
                    x += step;
                }
                y += step;
            }

            scale *= self.scale_factor;
        }

        // Group overlapping detections
        let grouped = group_detections(&detections, self.min_neighbors);

        Ok(grouped)
    }

    fn set_min_size(&mut self, width: u32, height: u32) {
        self.min_size = (width, height);
    }

    fn set_max_size(&mut self, width: u32, height: u32) {
        self.max_size = (width, height);
    }
}

/// Group overlapping detections using non-maximum suppression.
fn group_detections(detections: &[FaceRegion], min_neighbors: u32) -> Vec<FaceRegion> {
    if detections.is_empty() {
        return Vec::new();
    }

    // Simple grouping based on overlap
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut assigned = vec![false; detections.len()];

    for i in 0..detections.len() {
        if assigned[i] {
            continue;
        }

        let mut group = vec![i];
        assigned[i] = true;

        for j in (i + 1)..detections.len() {
            if assigned[j] {
                continue;
            }

            if detection_overlap(&detections[i].bbox, &detections[j].bbox) > 0.3 {
                group.push(j);
                assigned[j] = true;
            }
        }

        groups.push(group);
    }

    // Merge groups with enough detections
    let mut result = Vec::new();

    for group in groups {
        if group.len() >= min_neighbors as usize {
            // Average the detections in the group
            let mut sum_x = 0u64;
            let mut sum_y = 0u64;
            let mut sum_w = 0u64;
            let mut sum_h = 0u64;
            let mut max_conf = 0.0f64;

            for &idx in &group {
                let d = &detections[idx].bbox;
                sum_x += d.x as u64;
                sum_y += d.y as u64;
                sum_w += d.width as u64;
                sum_h += d.height as u64;
                max_conf = max_conf.max(d.confidence);
            }

            let n = group.len() as u64;
            let merged = DetectionResult::new(
                (sum_x / n) as u32,
                (sum_y / n) as u32,
                (sum_w / n) as u32,
                (sum_h / n) as u32,
                max_conf,
            );

            result.push(FaceRegion::new(merged));
        }
    }

    result
}

/// Calculate overlap ratio between two detections (`IoU`).
fn detection_overlap(a: &DetectionResult, b: &DetectionResult) -> f64 {
    let x1 = a.x.max(b.x);
    let y1 = a.y.max(b.y);
    let x2 = (a.x + a.width).min(b.x + b.width);
    let y2 = (a.y + a.height).min(b.y + b.height);

    if x2 <= x1 || y2 <= y1 {
        return 0.0;
    }

    let intersection = (x2 - x1) as f64 * (y2 - y1) as f64;
    let union = a.area() as f64 + b.area() as f64 - intersection;

    if union > 0.0 {
        intersection / union
    } else {
        0.0
    }
}

// ============================================================================
// CNN-based Face Detection with ONNX
// ============================================================================

#[cfg(feature = "onnx")]
use ndarray::Array4;
#[cfg(feature = "onnx")]
use ort::session::builder::GraphOptimizationLevel;
#[cfg(feature = "onnx")]
use ort::session::Session;
#[cfg(feature = "onnx")]
use ort::value::Value;
#[cfg(feature = "onnx")]
use std::path::Path;

/// Point in 2D space.
#[derive(Debug, Clone, Copy)]
pub struct Point2D {
    /// X coordinate.
    pub x: f32,
    /// Y coordinate.
    pub y: f32,
}

impl Point2D {
    /// Create a new 2D point.
    #[must_use]
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// Calculate Euclidean distance to another point.
    #[must_use]
    pub fn distance(&self, other: &Self) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }
}

/// Bounding box for face detection.
#[derive(Debug, Clone, Copy)]
pub struct BoundingBox {
    /// Left x coordinate.
    pub x: f32,
    /// Top y coordinate.
    pub y: f32,
    /// Box width.
    pub width: f32,
    /// Box height.
    pub height: f32,
}

impl BoundingBox {
    /// Create a new bounding box.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::detect::face::BoundingBox;
    ///
    /// let bbox = BoundingBox::new(10.0, 20.0, 100.0, 150.0);
    /// assert_eq!(bbox.x, 10.0);
    /// assert!(bbox.area() > 0.0);
    /// ```
    #[must_use]
    pub const fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Get the area of the bounding box.
    #[must_use]
    pub const fn area(&self) -> f32 {
        self.width * self.height
    }

    /// Get the center point of the bounding box.
    #[must_use]
    pub const fn center(&self) -> Point2D {
        Point2D {
            x: self.x + self.width / 2.0,
            y: self.y + self.height / 2.0,
        }
    }

    /// Calculate intersection over union (IoU) with another box.
    #[must_use]
    pub fn iou(&self, other: &Self) -> f32 {
        let x1 = self.x.max(other.x);
        let y1 = self.y.max(other.y);
        let x2 = (self.x + self.width).min(other.x + other.width);
        let y2 = (self.y + self.height).min(other.y + other.height);

        if x2 <= x1 || y2 <= y1 {
            return 0.0;
        }

        let intersection = (x2 - x1) * (y2 - y1);
        let union = self.area() + other.area() - intersection;

        if union > 0.0 {
            intersection / union
        } else {
            0.0
        }
    }

    /// Expand the box by a margin (as fraction of width/height).
    #[must_use]
    pub fn expand(&self, margin: f32) -> Self {
        let dx = self.width * margin;
        let dy = self.height * margin;
        Self {
            x: self.x - dx,
            y: self.y - dy,
            width: self.width + 2.0 * dx,
            height: self.height + 2.0 * dy,
        }
    }

    /// Clip the box to image boundaries.
    #[must_use]
    pub fn clip(&self, img_width: f32, img_height: f32) -> Self {
        let x = self.x.max(0.0);
        let y = self.y.max(0.0);
        let x2 = (self.x + self.width).min(img_width);
        let y2 = (self.y + self.height).min(img_height);
        Self {
            x,
            y,
            width: (x2 - x).max(0.0),
            height: (y2 - y).max(0.0),
        }
    }
}

/// Face detection result with bounding box, confidence, and landmarks.
#[cfg(feature = "onnx")]
#[derive(Debug, Clone)]
pub struct FaceDetection {
    /// Bounding box of the detected face.
    pub bbox: BoundingBox,
    /// Detection confidence (0.0 - 1.0).
    pub confidence: f32,
    /// Facial landmarks (5-point or 68-point).
    pub landmarks: Vec<Point2D>,
    /// Face rotation angle in degrees.
    pub angle: f32,
}

#[cfg(feature = "onnx")]
impl FaceDetection {
    /// Create a new face detection result.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::detect::face::{FaceDetection, BoundingBox};
    ///
    /// let bbox = BoundingBox::new(10.0, 20.0, 100.0, 100.0);
    /// let detection = FaceDetection::new(bbox, 0.95);
    /// assert!(detection.confidence > 0.9);
    /// ```
    #[must_use]
    pub fn new(bbox: BoundingBox, confidence: f32) -> Self {
        Self {
            bbox,
            confidence,
            landmarks: Vec::new(),
            angle: 0.0,
        }
    }

    /// Create a face detection with landmarks.
    #[must_use]
    pub fn with_landmarks(bbox: BoundingBox, confidence: f32, landmarks: Vec<Point2D>) -> Self {
        // Calculate face angle from eye landmarks if available
        let angle = if landmarks.len() >= 2 {
            let left_eye = &landmarks[0];
            let right_eye = &landmarks[1];
            let dy = right_eye.y - left_eye.y;
            let dx = right_eye.x - left_eye.x;
            dy.atan2(dx).to_degrees()
        } else {
            0.0
        };

        Self {
            bbox,
            confidence,
            landmarks,
            angle,
        }
    }

    /// Get landmarks as a slice of 5-point landmarks (if available).
    #[must_use]
    pub fn five_point_landmarks(&self) -> Option<[Point2D; 5]> {
        if self.landmarks.len() >= 5 {
            Some([
                self.landmarks[0],
                self.landmarks[1],
                self.landmarks[2],
                self.landmarks[3],
                self.landmarks[4],
            ])
        } else {
            None
        }
    }
}

/// CNN-based face detector using ONNX Runtime.
///
/// Supports multi-scale detection, face landmark detection, and NMS.
///
/// # Model Architecture
///
/// Expected input/output format:
/// - Detection model input: RGB image, variable size or fixed size (e.g., 640x640)
/// - Detection model output: [N, 6] tensor (x, y, w, h, confidence, class)
/// - Landmark model input: Face crop, fixed size (e.g., 112x112)
/// - Landmark model output: [N, 10] tensor (5 points: left_eye, right_eye, nose, left_mouth, right_mouth)
///
/// # Examples
///
/// ```no_run
/// use oximedia_cv::detect::face::CnnFaceDetector;
///
/// # fn example() -> oximedia_cv::CvResult<()> {
/// // Create detector (model path should point to a valid ONNX model)
/// let detector = CnnFaceDetector::new("/path/to/model.onnx")?;
///
/// // Detect faces in RGB image
/// let image = vec![0u8; 640 * 480 * 3];
/// let detections = detector.detect(&image, 640, 480)?;
///
/// for detection in detections {
///     println!("Face at ({}, {}) with confidence {}",
///         detection.bbox.x, detection.bbox.y, detection.confidence);
/// }
/// # Ok(())
/// # }
/// ```
#[cfg(feature = "onnx")]
#[allow(dead_code)]
pub struct CnnFaceDetector {
    detection_model: Session,
    landmark_model: Option<Session>,
    min_face_size: u32,
    confidence_threshold: f32,
    nms_threshold: f32,
    input_size: (u32, u32),
    scales: Vec<f32>,
}

#[cfg(feature = "onnx")]
impl CnnFaceDetector {
    /// Create a new CNN face detector with ONNX model.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to the ONNX detection model file
    ///
    /// # Returns
    ///
    /// Returns a configured face detector, or an error if the model cannot be loaded.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Model file does not exist
    /// - Model cannot be loaded by ONNX Runtime
    /// - Model format is invalid
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use oximedia_cv::detect::face::CnnFaceDetector;
    ///
    /// # fn example() -> oximedia_cv::CvResult<()> {
    /// let detector = CnnFaceDetector::new("/path/to/model.onnx")?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(model_path: impl AsRef<Path>) -> CvResult<Self> {
        let model_path = model_path.as_ref();

        if !model_path.exists() {
            return Err(CvError::detection_failed(format!(
                "Model file not found: {}",
                model_path.display()
            )));
        }

        // Create session with optimizations
        let detection_model = Session::builder()
            .map_err(|e| {
                CvError::detection_failed(format!("Failed to create session builder: {e}"))
            })?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| {
                CvError::detection_failed(format!("Failed to set optimization level: {e}"))
            })?
            .with_intra_threads(4)
            .map_err(|e| CvError::detection_failed(format!("Failed to set thread count: {e}")))?
            .commit_from_file(model_path)
            .map_err(|e| CvError::detection_failed(format!("Failed to load model: {e}")))?;

        Ok(Self {
            detection_model,
            landmark_model: None,
            min_face_size: 20,
            confidence_threshold: 0.5,
            nms_threshold: 0.4,
            input_size: (640, 640),
            scales: vec![1.0],
        })
    }

    /// Set landmark detection model (optional).
    ///
    /// # Errors
    ///
    /// Returns an error if the landmark model cannot be loaded.
    pub fn with_landmark_model(mut self, model_path: impl AsRef<Path>) -> CvResult<Self> {
        let model_path = model_path.as_ref();

        if !model_path.exists() {
            return Err(CvError::detection_failed(format!(
                "Landmark model file not found: {}",
                model_path.display()
            )));
        }

        let landmark_model = Session::builder()
            .map_err(|e| {
                CvError::detection_failed(format!("Failed to create session builder: {e}"))
            })?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| {
                CvError::detection_failed(format!("Failed to set optimization level: {e}"))
            })?
            .with_intra_threads(2)
            .map_err(|e| CvError::detection_failed(format!("Failed to set thread count: {e}")))?
            .commit_from_file(model_path)
            .map_err(|e| {
                CvError::detection_failed(format!("Failed to load landmark model: {e}"))
            })?;

        self.landmark_model = Some(landmark_model);
        Ok(self)
    }

    /// Set minimum face size for detection.
    #[must_use]
    pub const fn with_min_face_size(mut self, size: u32) -> Self {
        self.min_face_size = size;
        self
    }

    /// Set confidence threshold for detection.
    #[must_use]
    pub const fn with_confidence_threshold(mut self, threshold: f32) -> Self {
        self.confidence_threshold = threshold;
        self
    }

    /// Set NMS threshold for overlapping detections.
    #[must_use]
    pub const fn with_nms_threshold(mut self, threshold: f32) -> Self {
        self.nms_threshold = threshold;
        self
    }

    /// Set input image size for the model.
    #[must_use]
    pub const fn with_input_size(mut self, width: u32, height: u32) -> Self {
        self.input_size = (width, height);
        self
    }

    /// Set scale factors for multi-scale detection.
    #[must_use]
    pub fn with_scales(mut self, scales: Vec<f32>) -> Self {
        self.scales = scales;
        self
    }

    /// Detect faces in an RGB image.
    ///
    /// # Arguments
    ///
    /// * `image` - RGB image data (interleaved format)
    /// * `width` - Image width in pixels
    /// * `height` - Image height in pixels
    ///
    /// # Returns
    ///
    /// Vector of face detections with bounding boxes and confidence scores.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Image dimensions are invalid
    /// - Image data is insufficient
    /// - Model inference fails
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use oximedia_cv::detect::face::CnnFaceDetector;
    ///
    /// # fn example() -> oximedia_cv::CvResult<()> {
    /// let detector = CnnFaceDetector::new("/path/to/model.onnx")?;
    /// let image = vec![0u8; 640 * 480 * 3];
    /// let detections = detector.detect(&image, 640, 480)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn detect(
        &mut self,
        image: &[u8],
        width: u32,
        height: u32,
    ) -> CvResult<Vec<FaceDetection>> {
        if width == 0 || height == 0 {
            return Err(CvError::invalid_dimensions(width, height));
        }

        let expected_size = (width * height * 3) as usize;
        if image.len() < expected_size {
            return Err(CvError::insufficient_data(expected_size, image.len()));
        }

        let mut all_detections = Vec::new();

        // Collect scales first to avoid borrow conflict with run_inference
        let scales: Vec<f32> = self.scales.clone();

        // Multi-scale detection
        for scale in scales {
            let scaled_w = (width as f32 * scale) as u32;
            let scaled_h = (height as f32 * scale) as u32;

            if scaled_w < self.min_face_size || scaled_h < self.min_face_size {
                continue;
            }

            // Resize image if needed
            #[allow(clippy::float_cmp)]
            let (resized_image, resize_w, resize_h) = if scale == 1.0 {
                (image.to_vec(), width, height)
            } else {
                let resized = resize_rgb_bilinear(image, width, height, scaled_w, scaled_h);
                (resized, scaled_w, scaled_h)
            };

            // Preprocess image for model input
            let input_tensor = self.preprocess_image(&resized_image, resize_w, resize_h)?;

            // Run inference
            let detections = self.run_inference(input_tensor, resize_w, resize_h, scale)?;

            all_detections.extend(detections);
        }

        // Apply NMS to remove overlapping detections
        let filtered = self.non_maximum_suppression(all_detections);

        Ok(filtered)
    }

    /// Detect faces with landmark detection.
    ///
    /// # Errors
    ///
    /// Returns an error if landmark model is not loaded or inference fails.
    pub fn detect_with_landmarks(
        &mut self,
        image: &[u8],
        width: u32,
        height: u32,
    ) -> CvResult<Vec<FaceDetection>> {
        // First detect faces
        let mut detections = self.detect(image, width, height)?;

        // If landmark model is available, detect landmarks
        if let Some(ref mut landmark_model) = self.landmark_model {
            for detection in &mut detections {
                if let Ok(landmarks) = Self::detect_landmarks_static(
                    image,
                    width,
                    height,
                    &detection.bbox,
                    landmark_model,
                ) {
                    detection.landmarks = landmarks;

                    // Recalculate angle based on landmarks
                    if detection.landmarks.len() >= 2 {
                        let left_eye = &detection.landmarks[0];
                        let right_eye = &detection.landmarks[1];
                        let dy = right_eye.y - left_eye.y;
                        let dx = right_eye.x - left_eye.x;
                        detection.angle = dy.atan2(dx).to_degrees();
                    }
                }
            }
        }

        Ok(detections)
    }

    /// Preprocess image for model input.
    fn preprocess_image(&self, image: &[u8], width: u32, height: u32) -> CvResult<Array4<f32>> {
        let (target_w, target_h) = self.input_size;

        // Resize to model input size
        let resized = resize_rgb_bilinear(image, width, height, target_w, target_h);

        // Convert to float and normalize (0-1 range)
        let mut input = Array4::<f32>::zeros((1, 3, target_h as usize, target_w as usize));

        for c in 0..3 {
            for y in 0..target_h {
                for x in 0..target_w {
                    let idx = (y * target_w + x) as usize * 3 + c;
                    let value = f32::from(resized[idx]) / 255.0;
                    input[[0, c, y as usize, x as usize]] = value;
                }
            }
        }

        Ok(input)
    }

    /// Run model inference.
    #[allow(clippy::too_many_arguments)]
    fn run_inference(
        &mut self,
        input: Array4<f32>,
        orig_width: u32,
        orig_height: u32,
        scale: f32,
    ) -> CvResult<Vec<FaceDetection>> {
        // Create input tensor
        let input_value = Value::from_array(input).map_err(|e| {
            CvError::detection_failed(format!("Failed to create input tensor: {e}"))
        })?;

        // Run inference and extract owned data to release the session borrow
        let (shape_owned, data_owned) = {
            let outputs = self
                .detection_model
                .run(ort::inputs![input_value])
                .map_err(|e| CvError::detection_failed(format!("Inference failed: {e}")))?;
            let (shape, data) = outputs[0].try_extract_tensor::<f32>().map_err(|e| {
                CvError::detection_failed(format!("Failed to extract output tensor: {e}"))
            })?;
            let shape_owned: Vec<i64> = shape.iter().copied().collect();
            let data_owned: Vec<f32> = data.to_vec();
            (shape_owned, data_owned)
        };

        // Parse detections from output
        self.parse_detections(&shape_owned, &data_owned, orig_width, orig_height, scale)
    }

    /// Parse detections from model output.
    fn parse_detections(
        &self,
        shape: &[i64],
        data: &[f32],
        width: u32,
        height: u32,
        scale: f32,
    ) -> CvResult<Vec<FaceDetection>> {
        let mut detections = Vec::new();

        // Expected output shape: [batch, num_detections, 6] or [num_detections, 6]
        let num_detections = if shape.len() == 3 {
            shape[1] as usize
        } else {
            shape[0] as usize
        };

        for i in 0..num_detections {
            // Calculate flat index for [batch, detection, feature] layout
            let base_idx = i * 6;
            let x = data[base_idx];
            let y = data[base_idx + 1];
            let w = data[base_idx + 2];
            let h = data[base_idx + 3];
            let confidence = data[base_idx + 4];
            let _ = data[base_idx + 5];

            // Filter by confidence threshold
            if confidence < self.confidence_threshold {
                continue;
            }

            // Scale coordinates back to original image size
            let bbox = BoundingBox {
                x: x * width as f32 / scale,
                y: y * height as f32 / scale,
                width: w * width as f32 / scale,
                height: h * height as f32 / scale,
            };

            // Filter by minimum face size
            if bbox.width < self.min_face_size as f32 || bbox.height < self.min_face_size as f32 {
                continue;
            }

            detections.push(FaceDetection::new(bbox, confidence));
        }

        Ok(detections)
    }

    /// Detect facial landmarks for a face region.
    fn detect_landmarks_static(
        image: &[u8],
        img_width: u32,
        img_height: u32,
        bbox: &BoundingBox,
        landmark_model: &mut Session,
    ) -> CvResult<Vec<Point2D>> {
        // Extract face region with some margin
        let expanded = bbox.expand(0.2).clip(img_width as f32, img_height as f32);

        let face_crop = extract_rgb_region(
            image,
            img_width,
            img_height,
            expanded.x as u32,
            expanded.y as u32,
            expanded.width as u32,
            expanded.height as u32,
        )?;

        // Resize to landmark model input size (typically 112x112)
        let landmark_size = 112;
        let resized = resize_rgb_bilinear(
            &face_crop,
            expanded.width as u32,
            expanded.height as u32,
            landmark_size,
            landmark_size,
        );

        // Normalize and create input tensor
        let mut input =
            Array4::<f32>::zeros((1, 3, landmark_size as usize, landmark_size as usize));
        for c in 0..3 {
            for y in 0..landmark_size {
                for x in 0..landmark_size {
                    let idx = (y * landmark_size + x) as usize * 3 + c;
                    input[[0, c, y as usize, x as usize]] = f32::from(resized[idx]) / 255.0;
                }
            }
        }

        // Run inference
        let input_value = Value::from_array(input).map_err(|e| {
            CvError::detection_failed(format!("Failed to create input tensor: {e}"))
        })?;

        let outputs = landmark_model
            .run(ort::inputs![input_value])
            .map_err(|e| CvError::detection_failed(format!("Landmark inference failed: {e}")))?;

        // Extract output tensor - get first output
        let output = &outputs[0];

        // Extract landmarks
        let (shape, data) = output.try_extract_tensor::<f32>().map_err(|e| {
            CvError::detection_failed(format!("Failed to extract landmark tensor: {e}"))
        })?;

        // Expected: [1, 10] for 5 landmarks (x, y pairs)
        let num_landmarks = (shape[1] / 2) as usize;
        let mut landmarks = Vec::new();

        for i in 0..num_landmarks {
            let x = data[i * 2] * expanded.width + expanded.x;
            let y = data[i * 2 + 1] * expanded.height + expanded.y;
            landmarks.push(Point2D { x, y });
        }

        Ok(landmarks)
    }

    /// Apply Non-Maximum Suppression to filter overlapping detections.
    fn non_maximum_suppression(&self, mut detections: Vec<FaceDetection>) -> Vec<FaceDetection> {
        if detections.is_empty() {
            return Vec::new();
        }

        // Sort by confidence (descending)
        detections.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut keep = Vec::new();
        let mut suppressed = vec![false; detections.len()];

        for i in 0..detections.len() {
            if suppressed[i] {
                continue;
            }

            keep.push(detections[i].clone());

            // Suppress overlapping detections
            for j in (i + 1)..detections.len() {
                if suppressed[j] {
                    continue;
                }

                let iou = detections[i].bbox.iou(&detections[j].bbox);
                if iou > self.nms_threshold {
                    suppressed[j] = true;
                }
            }
        }

        keep
    }
}

// ============================================================================
// Image Processing Utilities
// ============================================================================

/// Resize RGB image using bilinear interpolation.
fn resize_rgb_bilinear(
    image: &[u8],
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
) -> Vec<u8> {
    let mut output = vec![0u8; (dst_width * dst_height * 3) as usize];

    let x_ratio = src_width as f32 / dst_width as f32;
    let y_ratio = src_height as f32 / dst_height as f32;

    for y in 0..dst_height {
        for x in 0..dst_width {
            let src_x = x as f32 * x_ratio;
            let src_y = y as f32 * y_ratio;

            let x0 = src_x.floor() as u32;
            let y0 = src_y.floor() as u32;
            let x1 = (x0 + 1).min(src_width - 1);
            let y1 = (y0 + 1).min(src_height - 1);

            let fx = src_x - x0 as f32;
            let fy = src_y - y0 as f32;

            for c in 0..3 {
                let idx00 = ((y0 * src_width + x0) * 3 + c) as usize;
                let idx01 = ((y0 * src_width + x1) * 3 + c) as usize;
                let idx10 = ((y1 * src_width + x0) * 3 + c) as usize;
                let idx11 = ((y1 * src_width + x1) * 3 + c) as usize;

                let v00 = f32::from(image[idx00]);
                let v01 = f32::from(image[idx01]);
                let v10 = f32::from(image[idx10]);
                let v11 = f32::from(image[idx11]);

                let v0 = v00 * (1.0 - fx) + v01 * fx;
                let v1 = v10 * (1.0 - fx) + v11 * fx;
                let v = v0 * (1.0 - fy) + v1 * fy;

                let out_idx = ((y * dst_width + x) * 3 + c) as usize;
                output[out_idx] = v.round() as u8;
            }
        }
    }

    output
}

/// Extract a rectangular region from an RGB image.
fn extract_rgb_region(
    image: &[u8],
    img_width: u32,
    img_height: u32,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
) -> CvResult<Vec<u8>> {
    let x2 = (x + width).min(img_width);
    let y2 = (y + height).min(img_height);
    let actual_width = x2 - x;
    let actual_height = y2 - y;

    let mut output = vec![0u8; (actual_width * actual_height * 3) as usize];

    for dy in 0..actual_height {
        let src_y = y + dy;
        let src_offset = (src_y * img_width + x) as usize * 3;
        let dst_offset = (dy * actual_width) as usize * 3;
        let copy_len = (actual_width * 3) as usize;

        output[dst_offset..dst_offset + copy_len]
            .copy_from_slice(&image[src_offset..src_offset + copy_len]);
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detection_result() {
        let result = DetectionResult::new(10, 20, 100, 100, 0.95);
        assert_eq!(result.x, 10);
        assert_eq!(result.y, 20);
        assert_eq!(result.center(), (60, 70));
        assert_eq!(result.area(), 10000);
    }

    #[test]
    fn test_face_region() {
        let bbox = DetectionResult::new(10, 20, 100, 100, 0.95);
        let face = FaceRegion::new(bbox);
        assert!(face.landmarks.is_none());
    }

    #[test]
    fn test_face_landmarks() {
        let landmarks = FaceLandmarks::five_point(
            (30.0, 40.0),
            (70.0, 40.0),
            (50.0, 60.0),
            (35.0, 80.0),
            (65.0, 80.0),
        );
        assert!((landmarks.eye_distance() - 40.0).abs() < 0.001);
    }

    #[test]
    fn test_integral_image() {
        let image = vec![1u8; 100];
        let integral = IntegralImage::compute(&image, 10, 10);

        assert_eq!(integral.width(), 10);
        assert_eq!(integral.height(), 10);
        assert_eq!(integral.sum(0, 0, 5, 5), 25);
        assert_eq!(integral.sum(0, 0, 10, 10), 100);
    }

    #[test]
    fn test_integral_image_mean() {
        let image = vec![100u8; 100];
        let integral = IntegralImage::compute(&image, 10, 10);

        let mean = integral.mean(0, 0, 10, 10);
        assert!((mean - 100.0).abs() < 0.001);
    }

    #[test]
    fn test_integral_image_variance() {
        let image = vec![100u8; 100];
        let integral = IntegralImage::compute(&image, 10, 10);

        let variance = integral.variance(0, 0, 10, 10);
        assert!(variance < 0.001); // Uniform image has zero variance
    }

    #[test]
    fn test_haar_cascade_new() {
        let cascade = HaarCascade::new(24, 24);
        assert_eq!(cascade.window_width, 24);
        assert_eq!(cascade.window_height, 24);
    }

    #[test]
    fn test_haar_cascade_detect_empty() {
        let cascade = HaarCascade::new(24, 24);
        let image = vec![100u8; 1000];
        let result = cascade
            .detect(&image, 100, 10)
            .expect("detect should succeed");

        // Empty cascade should return no detections
        assert!(result.is_empty());
    }

    #[test]
    fn test_detection_overlap() {
        let a = DetectionResult::new(0, 0, 100, 100, 1.0);
        let b = DetectionResult::new(50, 50, 100, 100, 1.0);

        let overlap = detection_overlap(&a, &b);
        assert!(overlap > 0.0);
        assert!(overlap < 1.0);
    }

    #[test]
    fn test_detection_no_overlap() {
        let a = DetectionResult::new(0, 0, 50, 50, 1.0);
        let b = DetectionResult::new(100, 100, 50, 50, 1.0);

        let overlap = detection_overlap(&a, &b);
        assert!(overlap < 0.001);
    }

    // CNN detector tests
    #[test]
    fn test_point2d() {
        let p1 = Point2D::new(0.0, 0.0);
        let p2 = Point2D::new(3.0, 4.0);
        assert!((p1.distance(&p2) - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_bounding_box() {
        let bbox = BoundingBox::new(10.0, 20.0, 100.0, 150.0);
        assert_eq!(bbox.x, 10.0);
        assert_eq!(bbox.area(), 15000.0);

        let center = bbox.center();
        assert_eq!(center.x, 60.0);
        assert_eq!(center.y, 95.0);
    }

    #[test]
    fn test_bbox_iou() {
        let bbox1 = BoundingBox::new(0.0, 0.0, 100.0, 100.0);
        let bbox2 = BoundingBox::new(50.0, 50.0, 100.0, 100.0);

        let iou = bbox1.iou(&bbox2);
        assert!(iou > 0.0);
        assert!(iou < 1.0);
    }

    #[test]
    fn test_bbox_expand() {
        let bbox = BoundingBox::new(100.0, 100.0, 100.0, 100.0);
        let expanded = bbox.expand(0.1);

        assert_eq!(expanded.x, 90.0);
        assert_eq!(expanded.y, 90.0);
        assert_eq!(expanded.width, 120.0);
        assert_eq!(expanded.height, 120.0);
    }

    #[test]
    fn test_bbox_clip() {
        let bbox = BoundingBox::new(-10.0, -10.0, 100.0, 100.0);
        let clipped = bbox.clip(640.0, 480.0);

        assert_eq!(clipped.x, 0.0);
        assert_eq!(clipped.y, 0.0);
        assert!(clipped.width <= 90.0);
        assert!(clipped.height <= 90.0);
    }

    #[test]
    #[cfg(feature = "onnx")]
    fn test_face_detection_new() {
        let bbox = BoundingBox::new(10.0, 20.0, 100.0, 100.0);
        let detection = FaceDetection::new(bbox, 0.95);

        assert_eq!(detection.bbox.x, 10.0);
        assert_eq!(detection.confidence, 0.95);
        assert!(detection.landmarks.is_empty());
        assert_eq!(detection.angle, 0.0);
    }

    #[test]
    #[cfg(feature = "onnx")]
    fn test_face_detection_with_landmarks() {
        let bbox = BoundingBox::new(10.0, 20.0, 100.0, 100.0);
        let landmarks = vec![
            Point2D::new(30.0, 40.0),
            Point2D::new(70.0, 40.0),
            Point2D::new(50.0, 60.0),
            Point2D::new(35.0, 80.0),
            Point2D::new(65.0, 80.0),
        ];

        let detection = FaceDetection::with_landmarks(bbox, 0.95, landmarks.clone());

        assert_eq!(detection.landmarks.len(), 5);
        assert_eq!(detection.angle, 0.0); // Eyes at same height
    }

    #[test]
    fn test_resize_rgb_bilinear() {
        let image = vec![128u8; 100 * 100 * 3];
        let resized = resize_rgb_bilinear(&image, 100, 100, 50, 50);

        assert_eq!(resized.len(), 50 * 50 * 3);
        // Uniform image should stay uniform
        assert_eq!(resized[0], 128);
    }

    #[test]
    fn test_extract_rgb_region() {
        let image = vec![0u8; 100 * 100 * 3];
        let region = extract_rgb_region(&image, 100, 100, 10, 10, 20, 20)
            .expect("extract_rgb_region should succeed");

        assert_eq!(region.len(), 20 * 20 * 3);
    }

    #[test]
    fn test_extract_rgb_region_clipped() {
        let image = vec![0u8; 100 * 100 * 3];
        // Request region that extends beyond image bounds
        let region = extract_rgb_region(&image, 100, 100, 90, 90, 20, 20)
            .expect("extract_rgb_region should succeed");

        // Should be clipped to 10x10
        assert_eq!(region.len(), 10 * 10 * 3);
    }
}
