//! Object detection module.
//!
//! This module provides generic object detection utilities including:
//! - Bounding box operations
//! - Non-Maximum Suppression (NMS)
//! - Intersection over Union (`IoU`)
//!
//! # Example
//!
//! ```
//! use oximedia_cv::detect::{BoundingBox, Detection};
//!
//! let bbox = BoundingBox::new(10.0, 20.0, 100.0, 150.0);
//! let detection = Detection::new(bbox, 0, 0.95);
//! ```

use crate::error::CvResult;

/// Trait for object detection algorithms.
pub trait ObjectDetector {
    /// Detect objects in an image.
    ///
    /// # Arguments
    ///
    /// * `image` - Image data (format depends on implementation)
    /// * `width` - Image width
    /// * `height` - Image height
    ///
    /// # Returns
    ///
    /// Vector of detections.
    ///
    /// # Errors
    ///
    /// Returns an error if detection fails.
    fn detect(&mut self, image: &[u8], width: u32, height: u32) -> CvResult<Vec<Detection>>;

    /// Get class names supported by this detector.
    fn class_names(&self) -> &[String];
}

/// Bounding box representation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundingBox {
    /// X coordinate of top-left corner.
    pub x: f32,
    /// Y coordinate of top-left corner.
    pub y: f32,
    /// Width of the box.
    pub width: f32,
    /// Height of the box.
    pub height: f32,
}

impl BoundingBox {
    /// Create a new bounding box.
    ///
    /// # Arguments
    ///
    /// * `x` - X coordinate of top-left corner
    /// * `y` - Y coordinate of top-left corner
    /// * `width` - Box width
    /// * `height` - Box height
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::detect::BoundingBox;
    ///
    /// let bbox = BoundingBox::new(10.0, 20.0, 100.0, 150.0);
    /// assert_eq!(bbox.x, 10.0);
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

    /// Create a bounding box from center coordinates.
    ///
    /// # Arguments
    ///
    /// * `cx` - Center X coordinate
    /// * `cy` - Center Y coordinate
    /// * `width` - Box width
    /// * `height` - Box height
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::detect::BoundingBox;
    ///
    /// let bbox = BoundingBox::from_center(50.0, 50.0, 100.0, 100.0);
    /// assert_eq!(bbox.x, 0.0);
    /// ```
    #[must_use]
    pub fn from_center(cx: f32, cy: f32, width: f32, height: f32) -> Self {
        Self {
            x: cx - width / 2.0,
            y: cy - height / 2.0,
            width,
            height,
        }
    }

    /// Create a bounding box from two corner points.
    ///
    /// # Arguments
    ///
    /// * `x1` - X coordinate of first corner
    /// * `y1` - Y coordinate of first corner
    /// * `x2` - X coordinate of opposite corner
    /// * `y2` - Y coordinate of opposite corner
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::detect::BoundingBox;
    ///
    /// let bbox = BoundingBox::from_corners(10.0, 20.0, 110.0, 170.0);
    /// assert_eq!(bbox.width, 100.0);
    /// ```
    #[must_use]
    pub fn from_corners(x1: f32, y1: f32, x2: f32, y2: f32) -> Self {
        let min_x = x1.min(x2);
        let min_y = y1.min(y2);
        let max_x = x1.max(x2);
        let max_y = y1.max(y2);

        Self {
            x: min_x,
            y: min_y,
            width: max_x - min_x,
            height: max_y - min_y,
        }
    }

    /// Get the area of the bounding box.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::detect::BoundingBox;
    ///
    /// let bbox = BoundingBox::new(0.0, 0.0, 10.0, 20.0);
    /// assert_eq!(bbox.area(), 200.0);
    /// ```
    #[must_use]
    pub fn area(&self) -> f32 {
        self.width * self.height
    }

    /// Get the center point of the bounding box.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::detect::BoundingBox;
    ///
    /// let bbox = BoundingBox::new(0.0, 0.0, 100.0, 100.0);
    /// let (cx, cy) = bbox.center();
    /// assert_eq!(cx, 50.0);
    /// ```
    #[must_use]
    pub fn center(&self) -> (f32, f32) {
        (self.x + self.width / 2.0, self.y + self.height / 2.0)
    }

    /// Get the right edge coordinate.
    #[must_use]
    pub fn right(&self) -> f32 {
        self.x + self.width
    }

    /// Get the bottom edge coordinate.
    #[must_use]
    pub fn bottom(&self) -> f32 {
        self.y + self.height
    }

    /// Check if this box contains a point.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::detect::BoundingBox;
    ///
    /// let bbox = BoundingBox::new(0.0, 0.0, 100.0, 100.0);
    /// assert!(bbox.contains(50.0, 50.0));
    /// assert!(!bbox.contains(150.0, 50.0));
    /// ```
    #[must_use]
    pub fn contains(&self, px: f32, py: f32) -> bool {
        px >= self.x && px <= self.right() && py >= self.y && py <= self.bottom()
    }

    /// Check if this box intersects with another.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::detect::BoundingBox;
    ///
    /// let a = BoundingBox::new(0.0, 0.0, 100.0, 100.0);
    /// let b = BoundingBox::new(50.0, 50.0, 100.0, 100.0);
    /// assert!(a.intersects(&b));
    /// ```
    #[must_use]
    pub fn intersects(&self, other: &Self) -> bool {
        self.x < other.right()
            && self.right() > other.x
            && self.y < other.bottom()
            && self.bottom() > other.y
    }

    /// Get the intersection box with another bounding box.
    ///
    /// Returns None if the boxes don't intersect.
    #[must_use]
    pub fn intersection(&self, other: &Self) -> Option<Self> {
        let x1 = self.x.max(other.x);
        let y1 = self.y.max(other.y);
        let x2 = self.right().min(other.right());
        let y2 = self.bottom().min(other.bottom());

        if x2 > x1 && y2 > y1 {
            Some(Self::new(x1, y1, x2 - x1, y2 - y1))
        } else {
            None
        }
    }

    /// Get the union box with another bounding box.
    #[must_use]
    pub fn union(&self, other: &Self) -> Self {
        let x1 = self.x.min(other.x);
        let y1 = self.y.min(other.y);
        let x2 = self.right().max(other.right());
        let y2 = self.bottom().max(other.bottom());

        Self::new(x1, y1, x2 - x1, y2 - y1)
    }

    /// Scale the bounding box by a factor.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::detect::BoundingBox;
    ///
    /// let bbox = BoundingBox::new(0.0, 0.0, 100.0, 100.0);
    /// let scaled = bbox.scale(0.5);
    /// assert_eq!(scaled.width, 50.0);
    /// ```
    #[must_use]
    pub fn scale(&self, factor: f32) -> Self {
        Self {
            x: self.x * factor,
            y: self.y * factor,
            width: self.width * factor,
            height: self.height * factor,
        }
    }

    /// Expand the bounding box by a margin.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::detect::BoundingBox;
    ///
    /// let bbox = BoundingBox::new(10.0, 10.0, 80.0, 80.0);
    /// let expanded = bbox.expand(5.0);
    /// assert_eq!(expanded.x, 5.0);
    /// assert_eq!(expanded.width, 90.0);
    /// ```
    #[must_use]
    pub fn expand(&self, margin: f32) -> Self {
        Self {
            x: self.x - margin,
            y: self.y - margin,
            width: self.width + 2.0 * margin,
            height: self.height + 2.0 * margin,
        }
    }

    /// Clamp the bounding box to image bounds.
    #[must_use]
    pub fn clamp(&self, img_width: f32, img_height: f32) -> Self {
        let x = self.x.max(0.0);
        let y = self.y.max(0.0);
        let right = self.right().min(img_width);
        let bottom = self.bottom().min(img_height);

        Self {
            x,
            y,
            width: (right - x).max(0.0),
            height: (bottom - y).max(0.0),
        }
    }
}

impl Default for BoundingBox {
    fn default() -> Self {
        Self::new(0.0, 0.0, 0.0, 0.0)
    }
}

/// Object detection result.
#[derive(Debug, Clone)]
pub struct Detection {
    /// Bounding box of the detected object.
    pub bbox: BoundingBox,
    /// Class ID.
    pub class_id: u32,
    /// Detection confidence (0.0 - 1.0).
    pub confidence: f32,
    /// Optional class name.
    pub class_name: Option<String>,
}

impl Detection {
    /// Create a new detection.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::detect::{BoundingBox, Detection};
    ///
    /// let bbox = BoundingBox::new(10.0, 20.0, 100.0, 100.0);
    /// let detection = Detection::new(bbox, 0, 0.95);
    /// ```
    #[must_use]
    pub fn new(bbox: BoundingBox, class_id: u32, confidence: f32) -> Self {
        Self {
            bbox,
            class_id,
            confidence,
            class_name: None,
        }
    }

    /// Create a detection with a class name.
    #[must_use]
    pub fn with_class_name(mut self, name: impl Into<String>) -> Self {
        self.class_name = Some(name.into());
        self
    }
}

/// Calculate Intersection over Union (`IoU`) between two bounding boxes.
///
/// # Arguments
///
/// * `a` - First bounding box
/// * `b` - Second bounding box
///
/// # Returns
///
/// `IoU` value between 0.0 and 1.0.
///
/// # Examples
///
/// ```
/// use oximedia_cv::detect::{BoundingBox, object::iou};
///
/// let a = BoundingBox::new(0.0, 0.0, 100.0, 100.0);
/// let b = BoundingBox::new(50.0, 50.0, 100.0, 100.0);
/// let overlap = iou(&a, &b);
/// assert!(overlap > 0.0 && overlap < 1.0);
/// ```
#[must_use]
pub fn iou(a: &BoundingBox, b: &BoundingBox) -> f32 {
    if let Some(intersection) = a.intersection(b) {
        let intersection_area = intersection.area();
        let union_area = a.area() + b.area() - intersection_area;

        if union_area > 0.0 {
            intersection_area / union_area
        } else {
            0.0
        }
    } else {
        0.0
    }
}

/// Calculate Generalized `IoU` (`GIoU`) between two bounding boxes.
///
/// `GIoU` is a more robust metric that handles non-overlapping boxes better.
///
/// # Returns
///
/// `GIoU` value between -1.0 and 1.0.
#[must_use]
pub fn giou(a: &BoundingBox, b: &BoundingBox) -> f32 {
    let intersection_area = a.intersection(b).map_or(0.0, |i| i.area());
    let union_area = a.area() + b.area() - intersection_area;

    if union_area <= 0.0 {
        return -1.0;
    }

    let iou_val = intersection_area / union_area;

    // Enclosing box
    let enclosing = a.union(b);
    let enclosing_area = enclosing.area();

    if enclosing_area <= 0.0 {
        return iou_val;
    }

    // GIoU = IoU - (C - U) / C
    // where C is enclosing area and U is union area
    iou_val - (enclosing_area - union_area) / enclosing_area
}

/// Perform Non-Maximum Suppression on detections.
///
/// # Arguments
///
/// * `detections` - Input detections
/// * `iou_threshold` - `IoU` threshold for suppression (typically 0.45-0.5)
///
/// # Returns
///
/// Filtered detections after NMS.
///
/// # Examples
///
/// ```
/// use oximedia_cv::detect::{BoundingBox, Detection, object::nms};
///
/// let detections = vec![
///     Detection::new(BoundingBox::new(0.0, 0.0, 100.0, 100.0), 0, 0.9),
///     Detection::new(BoundingBox::new(10.0, 10.0, 100.0, 100.0), 0, 0.8),
/// ];
/// let filtered = nms(&detections, 0.5);
/// // The second detection should be suppressed due to high overlap
/// ```
#[must_use]
pub fn nms(detections: &[Detection], iou_threshold: f32) -> Vec<Detection> {
    if detections.is_empty() {
        return Vec::new();
    }

    // Sort by confidence (descending)
    let mut sorted: Vec<_> = detections.iter().enumerate().collect();
    sorted.sort_by(|a, b| {
        b.1.confidence
            .partial_cmp(&a.1.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut keep = vec![true; detections.len()];
    let mut result = Vec::new();

    for i in 0..sorted.len() {
        let (orig_idx, detection) = sorted[i];
        if !keep[orig_idx] {
            continue;
        }

        result.push(detection.clone());

        // Suppress overlapping detections with lower confidence
        for j in (i + 1)..sorted.len() {
            let (other_idx, other) = sorted[j];
            if !keep[other_idx] {
                continue;
            }

            // Only suppress same class
            if detection.class_id == other.class_id
                && iou(&detection.bbox, &other.bbox) > iou_threshold
            {
                keep[other_idx] = false;
            }
        }
    }

    result
}

/// Perform soft NMS on detections.
///
/// Instead of completely suppressing overlapping detections,
/// soft NMS reduces their confidence scores.
///
/// # Arguments
///
/// * `detections` - Input detections
/// * `iou_threshold` - `IoU` threshold for soft suppression
/// * `sigma` - Gaussian sigma for score decay
/// * `score_threshold` - Minimum score to keep a detection
///
/// # Returns
///
/// Detections with adjusted confidence scores.
#[must_use]
pub fn soft_nms(
    detections: &[Detection],
    iou_threshold: f32,
    sigma: f32,
    score_threshold: f32,
) -> Vec<Detection> {
    if detections.is_empty() {
        return Vec::new();
    }

    let mut detections = detections.to_vec();
    let mut result = Vec::new();

    while !detections.is_empty() {
        // Find detection with highest confidence
        // detections is non-empty (loop condition), so max_by always yields Some
        let max_idx = detections
            .iter()
            .enumerate()
            .max_by(|a, b| {
                a.1.confidence
                    .partial_cmp(&b.1.confidence)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0);

        let best = detections.remove(max_idx);
        result.push(best.clone());

        // Update scores of remaining detections
        for det in &mut detections {
            if det.class_id == best.class_id {
                let overlap = iou(&best.bbox, &det.bbox);
                if overlap > iou_threshold {
                    // Gaussian decay
                    det.confidence *= (-overlap * overlap / sigma).exp();
                }
            }
        }

        // Remove low-confidence detections
        detections.retain(|d| d.confidence >= score_threshold);
    }

    result
}

/// Filter detections by confidence threshold.
#[must_use]
pub fn filter_by_confidence(detections: &[Detection], threshold: f32) -> Vec<Detection> {
    detections
        .iter()
        .filter(|d| d.confidence >= threshold)
        .cloned()
        .collect()
}

/// Filter detections by class ID.
#[must_use]
pub fn filter_by_class(detections: &[Detection], class_id: u32) -> Vec<Detection> {
    detections
        .iter()
        .filter(|d| d.class_id == class_id)
        .cloned()
        .collect()
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_bounding_box_new() {
        let bbox = BoundingBox::new(10.0, 20.0, 100.0, 150.0);
        assert_eq!(bbox.x, 10.0);
        assert_eq!(bbox.y, 20.0);
        assert_eq!(bbox.width, 100.0);
        assert_eq!(bbox.height, 150.0);
    }

    #[test]
    fn test_bounding_box_from_center() {
        let bbox = BoundingBox::from_center(50.0, 50.0, 100.0, 100.0);
        assert_eq!(bbox.x, 0.0);
        assert_eq!(bbox.y, 0.0);
    }

    #[test]
    fn test_bounding_box_from_corners() {
        let bbox = BoundingBox::from_corners(10.0, 20.0, 110.0, 170.0);
        assert_eq!(bbox.x, 10.0);
        assert_eq!(bbox.y, 20.0);
        assert_eq!(bbox.width, 100.0);
        assert_eq!(bbox.height, 150.0);
    }

    #[test]
    fn test_bounding_box_area() {
        let bbox = BoundingBox::new(0.0, 0.0, 10.0, 20.0);
        assert_eq!(bbox.area(), 200.0);
    }

    #[test]
    fn test_bounding_box_center() {
        let bbox = BoundingBox::new(0.0, 0.0, 100.0, 100.0);
        let (cx, cy) = bbox.center();
        assert_eq!(cx, 50.0);
        assert_eq!(cy, 50.0);
    }

    #[test]
    fn test_bounding_box_contains() {
        let bbox = BoundingBox::new(0.0, 0.0, 100.0, 100.0);
        assert!(bbox.contains(50.0, 50.0));
        assert!(!bbox.contains(150.0, 50.0));
    }

    #[test]
    fn test_bounding_box_intersects() {
        let a = BoundingBox::new(0.0, 0.0, 100.0, 100.0);
        let b = BoundingBox::new(50.0, 50.0, 100.0, 100.0);
        let c = BoundingBox::new(200.0, 200.0, 50.0, 50.0);

        assert!(a.intersects(&b));
        assert!(!a.intersects(&c));
    }

    #[test]
    fn test_bounding_box_intersection() {
        let a = BoundingBox::new(0.0, 0.0, 100.0, 100.0);
        let b = BoundingBox::new(50.0, 50.0, 100.0, 100.0);

        let inter = a.intersection(&b).expect("intersection should succeed");
        assert_eq!(inter.x, 50.0);
        assert_eq!(inter.y, 50.0);
        assert_eq!(inter.width, 50.0);
        assert_eq!(inter.height, 50.0);
    }

    #[test]
    fn test_bounding_box_union() {
        let a = BoundingBox::new(0.0, 0.0, 50.0, 50.0);
        let b = BoundingBox::new(50.0, 50.0, 50.0, 50.0);

        let union = a.union(&b);
        assert_eq!(union.x, 0.0);
        assert_eq!(union.y, 0.0);
        assert_eq!(union.width, 100.0);
        assert_eq!(union.height, 100.0);
    }

    #[test]
    fn test_bounding_box_scale() {
        let bbox = BoundingBox::new(10.0, 20.0, 100.0, 100.0);
        let scaled = bbox.scale(0.5);
        assert_eq!(scaled.x, 5.0);
        assert_eq!(scaled.width, 50.0);
    }

    #[test]
    fn test_bounding_box_expand() {
        let bbox = BoundingBox::new(10.0, 10.0, 80.0, 80.0);
        let expanded = bbox.expand(5.0);
        assert_eq!(expanded.x, 5.0);
        assert_eq!(expanded.y, 5.0);
        assert_eq!(expanded.width, 90.0);
        assert_eq!(expanded.height, 90.0);
    }

    #[test]
    fn test_detection_new() {
        let bbox = BoundingBox::new(10.0, 20.0, 100.0, 100.0);
        let detection = Detection::new(bbox, 0, 0.95);
        assert_eq!(detection.class_id, 0);
        assert!(detection.confidence > 0.9);
    }

    #[test]
    fn test_iou() {
        let a = BoundingBox::new(0.0, 0.0, 100.0, 100.0);
        let b = BoundingBox::new(50.0, 50.0, 100.0, 100.0);

        let overlap = iou(&a, &b);
        // Intersection: 50x50 = 2500
        // Union: 10000 + 10000 - 2500 = 17500
        // IoU: 2500 / 17500 = 0.142857...
        assert!(overlap > 0.14 && overlap < 0.15);
    }

    #[test]
    fn test_iou_no_overlap() {
        let a = BoundingBox::new(0.0, 0.0, 50.0, 50.0);
        let b = BoundingBox::new(100.0, 100.0, 50.0, 50.0);

        assert_eq!(iou(&a, &b), 0.0);
    }

    #[test]
    fn test_iou_same_box() {
        let a = BoundingBox::new(0.0, 0.0, 100.0, 100.0);
        let b = BoundingBox::new(0.0, 0.0, 100.0, 100.0);

        assert!((iou(&a, &b) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_giou() {
        let a = BoundingBox::new(0.0, 0.0, 100.0, 100.0);
        let b = BoundingBox::new(50.0, 50.0, 100.0, 100.0);

        let g = giou(&a, &b);
        // GIoU can range from -1 to 1
        assert!((-1.0..=1.0).contains(&g));

        // Same boxes should have GIoU of 1
        let c = BoundingBox::new(0.0, 0.0, 100.0, 100.0);
        let g2 = giou(&a, &c);
        assert!((g2 - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_nms() {
        let detections = vec![
            Detection::new(BoundingBox::new(0.0, 0.0, 100.0, 100.0), 0, 0.9),
            Detection::new(BoundingBox::new(10.0, 10.0, 100.0, 100.0), 0, 0.8),
            Detection::new(BoundingBox::new(200.0, 200.0, 100.0, 100.0), 0, 0.85),
        ];

        let filtered = nms(&detections, 0.5);

        // First two overlap significantly, keep only the best
        // Third one doesn't overlap, should be kept
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_nms_different_classes() {
        let detections = vec![
            Detection::new(BoundingBox::new(0.0, 0.0, 100.0, 100.0), 0, 0.9),
            Detection::new(BoundingBox::new(0.0, 0.0, 100.0, 100.0), 1, 0.8),
        ];

        let filtered = nms(&detections, 0.5);

        // Different classes, both should be kept
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_soft_nms() {
        let detections = vec![
            Detection::new(BoundingBox::new(0.0, 0.0, 100.0, 100.0), 0, 0.9),
            Detection::new(BoundingBox::new(10.0, 10.0, 100.0, 100.0), 0, 0.8),
        ];

        let filtered = soft_nms(&detections, 0.3, 0.5, 0.1);

        // Both should be kept but with adjusted scores
        assert!(!filtered.is_empty());
    }

    #[test]
    fn test_filter_by_confidence() {
        let detections = vec![
            Detection::new(BoundingBox::new(0.0, 0.0, 100.0, 100.0), 0, 0.9),
            Detection::new(BoundingBox::new(0.0, 0.0, 100.0, 100.0), 0, 0.3),
        ];

        let filtered = filter_by_confidence(&detections, 0.5);
        assert_eq!(filtered.len(), 1);
    }

    #[test]
    fn test_filter_by_class() {
        let detections = vec![
            Detection::new(BoundingBox::new(0.0, 0.0, 100.0, 100.0), 0, 0.9),
            Detection::new(BoundingBox::new(0.0, 0.0, 100.0, 100.0), 1, 0.8),
        ];

        let filtered = filter_by_class(&detections, 0);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].class_id, 0);
    }
}
