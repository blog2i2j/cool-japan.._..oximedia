//! TLD (Tracking-Learning-Detection) tracker.
//!
//! TLD combines a tracker with a detector and a learner to handle
//! long-term tracking with recovery from failures.
//!
//! # Example
//!
//! ```
//! use oximedia_cv::tracking::tld::TldTracker;
//! use oximedia_cv::detect::BoundingBox;
//!
//! let bbox = BoundingBox::new(50.0, 50.0, 100.0, 100.0);
//! let tracker = TldTracker::new(bbox);
//! ```

use crate::detect::BoundingBox;
use crate::error::{CvError, CvResult};

/// Positive/negative patch for learning.
#[derive(Debug, Clone)]
struct Patch {
    /// Patch features.
    features: Vec<f64>,
    /// Label (true = positive, false = negative).
    label: bool,
}

/// TLD tracker configuration.
#[derive(Debug, Clone)]
pub struct TldTracker {
    /// Current bounding box.
    bbox: BoundingBox,
    /// Tracker confidence.
    confidence: f64,
    /// Positive patches (learned object model).
    positive_patches: Vec<Patch>,
    /// Negative patches (background model).
    negative_patches: Vec<Patch>,
    /// Template size.
    template_size: (usize, usize),
    /// Previous frame.
    prev_frame: Vec<u8>,
    /// Previous frame dimensions.
    prev_dims: (u32, u32),
    /// Learning enabled.
    learning_enabled: bool,
    /// Maximum patches to store.
    max_patches: usize,
    /// Similarity threshold.
    similarity_threshold: f64,
    /// Grid for scanning detector.
    detector_scales: Vec<f64>,
    /// Tracking valid.
    tracking_valid: bool,
}

impl TldTracker {
    /// Create a new TLD tracker.
    ///
    /// # Arguments
    ///
    /// * `bbox` - Initial bounding box
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::tracking::tld::TldTracker;
    /// use oximedia_cv::detect::BoundingBox;
    ///
    /// let bbox = BoundingBox::new(100.0, 100.0, 50.0, 50.0);
    /// let tracker = TldTracker::new(bbox);
    /// ```
    #[must_use]
    pub fn new(bbox: BoundingBox) -> Self {
        Self {
            bbox,
            confidence: 1.0,
            positive_patches: Vec::new(),
            negative_patches: Vec::new(),
            template_size: (32, 32),
            prev_frame: Vec::new(),
            prev_dims: (0, 0),
            learning_enabled: true,
            max_patches: 100,
            similarity_threshold: 0.7,
            detector_scales: vec![0.8, 0.9, 1.0, 1.1, 1.2],
            tracking_valid: true,
        }
    }

    /// Enable or disable learning.
    #[must_use]
    pub const fn with_learning(mut self, enabled: bool) -> Self {
        self.learning_enabled = enabled;
        self
    }

    /// Set similarity threshold.
    #[must_use]
    pub const fn with_similarity_threshold(mut self, threshold: f64) -> Self {
        self.similarity_threshold = threshold;
        self
    }

    /// Initialize the tracker with the first frame.
    ///
    /// # Errors
    ///
    /// Returns an error if frame dimensions are invalid.
    pub fn initialize(&mut self, frame: &[u8], width: u32, height: u32) -> CvResult<()> {
        if width == 0 || height == 0 {
            return Err(CvError::invalid_dimensions(width, height));
        }

        // Extract positive patch from initial bbox
        let positive_patch = self.extract_features(frame, width, height, &self.bbox)?;
        self.positive_patches.push(Patch {
            features: positive_patch,
            label: true,
        });

        // Sample negative patches from surrounding area
        let negative_bboxes = self.sample_negative_boxes(width, height);
        for neg_bbox in negative_bboxes {
            if let Ok(features) = self.extract_features(frame, width, height, &neg_bbox) {
                self.negative_patches.push(Patch {
                    features,
                    label: false,
                });
            }
        }

        // Store frame for tracking
        self.prev_frame = frame.to_vec();
        self.prev_dims = (width, height);

        Ok(())
    }

    /// Update tracker with a new frame.
    ///
    /// # Errors
    ///
    /// Returns an error if tracking fails or dimensions are invalid.
    #[allow(clippy::too_many_lines)]
    pub fn update(&mut self, frame: &[u8], width: u32, height: u32) -> CvResult<BoundingBox> {
        if self.prev_frame.is_empty() {
            return Err(CvError::tracking_error("Tracker not initialized"));
        }

        // Step 1: TRACK - Use median flow or simple tracking
        let tracked_bbox = if self.tracking_valid {
            self.track_frame(frame, width, height)?
        } else {
            self.bbox
        };

        // Step 2: DETECT - Scan for object at multiple scales
        let detected_bboxes = self.detect_object(frame, width, height)?;

        // Step 3: INTEGRATE - Combine tracker and detector
        let (final_bbox, tracking_confidence) =
            self.integrate_results(&tracked_bbox, &detected_bboxes, frame, width, height)?;

        // Update tracking validity
        self.tracking_valid = tracking_confidence > 0.5;

        // Step 4: LEARN - Update positive and negative models
        if self.learning_enabled && tracking_confidence > 0.6 {
            self.learn(frame, width, height, &final_bbox)?;
        }

        // Update state
        self.bbox = final_bbox;
        self.confidence = tracking_confidence;

        // Store frame for next iteration
        self.prev_frame = frame.to_vec();
        self.prev_dims = (width, height);

        Ok(final_bbox)
    }

    /// Get current bounding box.
    #[must_use]
    pub const fn bbox(&self) -> &BoundingBox {
        &self.bbox
    }

    /// Get current confidence.
    #[must_use]
    pub const fn confidence(&self) -> f64 {
        self.confidence
    }

    /// Reset tracker with new bounding box.
    pub fn reset(&mut self, bbox: BoundingBox) {
        self.bbox = bbox;
        self.confidence = 1.0;
        self.positive_patches.clear();
        self.negative_patches.clear();
        self.prev_frame.clear();
        self.tracking_valid = true;
    }

    /// Track object in new frame using simple optical flow.
    fn track_frame(&self, frame: &[u8], width: u32, height: u32) -> CvResult<BoundingBox> {
        // Simple template matching for tracking
        let search_radius = 30;
        let cx = (self.bbox.x + self.bbox.width / 2.0) as i32;
        let cy = (self.bbox.y + self.bbox.height / 2.0) as i32;

        let mut best_score = f64::NEG_INFINITY;
        let mut best_offset = (0, 0);

        // Extract template from previous frame
        let template = self.extract_features(
            &self.prev_frame,
            self.prev_dims.0,
            self.prev_dims.1,
            &self.bbox,
        )?;

        // Search in current frame
        for dy in -search_radius..=search_radius {
            for dx in -search_radius..=search_radius {
                let test_bbox = BoundingBox::new(
                    self.bbox.x + dx as f32,
                    self.bbox.y + dy as f32,
                    self.bbox.width,
                    self.bbox.height,
                );

                if let Ok(features) = self.extract_features(frame, width, height, &test_bbox) {
                    let similarity = compute_similarity(&template, &features);
                    if similarity > best_score {
                        best_score = similarity;
                        best_offset = (dx, dy);
                    }
                }
            }
        }

        Ok(BoundingBox::new(
            self.bbox.x + best_offset.0 as f32,
            self.bbox.y + best_offset.1 as f32,
            self.bbox.width,
            self.bbox.height,
        ))
    }

    /// Detect object using learned detector.
    fn detect_object(
        &self,
        frame: &[u8],
        width: u32,
        height: u32,
    ) -> CvResult<Vec<(BoundingBox, f64)>> {
        let mut detections = Vec::new();

        // Scan at multiple scales
        for &scale in &self.detector_scales {
            let scaled_w = (self.bbox.width * scale as f32) as u32;
            let scaled_h = (self.bbox.height * scale as f32) as u32;

            // Stride for scanning
            let stride_x = scaled_w / 4;
            let stride_y = scaled_h / 4;

            let mut y = 0;
            while y + scaled_h < height {
                let mut x = 0;
                while x + scaled_w < width {
                    let test_bbox =
                        BoundingBox::new(x as f32, y as f32, scaled_w as f32, scaled_h as f32);

                    if let Ok(features) = self.extract_features(frame, width, height, &test_bbox) {
                        let confidence = self.classify_patch(&features);
                        if confidence > self.similarity_threshold {
                            detections.push((test_bbox, confidence));
                        }
                    }

                    x += stride_x;
                }
                y += stride_y;
            }
        }

        // Non-maximum suppression
        let filtered = non_max_suppression(&detections, 0.3);

        Ok(filtered)
    }

    /// Integrate tracking and detection results.
    fn integrate_results(
        &self,
        tracked: &BoundingBox,
        detected: &[(BoundingBox, f64)],
        frame: &[u8],
        width: u32,
        height: u32,
    ) -> CvResult<(BoundingBox, f64)> {
        // Validate tracking
        let tracked_features = self.extract_features(frame, width, height, tracked)?;
        let tracking_confidence = self.classify_patch(&tracked_features);

        // If tracking is confident, use it
        if tracking_confidence > 0.8 {
            return Ok((*tracked, tracking_confidence));
        }

        // If tracking failed but we have detections, use best detection
        if !detected.is_empty() {
            if let Some(best_detection) = detected
                .iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            {
                return Ok(*best_detection);
            }
        }

        // Fall back to tracking even if confidence is low
        Ok((*tracked, tracking_confidence))
    }

    /// Learn from current frame.
    fn learn(&mut self, frame: &[u8], width: u32, height: u32, bbox: &BoundingBox) -> CvResult<()> {
        // Add positive patch
        let positive_features = self.extract_features(frame, width, height, bbox)?;
        self.positive_patches.push(Patch {
            features: positive_features,
            label: true,
        });

        // Sample new negative patches
        let negative_bboxes = self.sample_negative_boxes(width, height);
        for neg_bbox in negative_bboxes.iter().take(5) {
            if let Ok(features) = self.extract_features(frame, width, height, neg_bbox) {
                self.negative_patches.push(Patch {
                    features,
                    label: false,
                });
            }
        }

        // Limit patch history
        if self.positive_patches.len() > self.max_patches {
            self.positive_patches
                .drain(0..self.positive_patches.len() - self.max_patches);
        }

        if self.negative_patches.len() > self.max_patches {
            self.negative_patches
                .drain(0..self.negative_patches.len() - self.max_patches);
        }

        Ok(())
    }

    /// Extract features from a bounding box region.
    fn extract_features(
        &self,
        frame: &[u8],
        width: u32,
        height: u32,
        bbox: &BoundingBox,
    ) -> CvResult<Vec<f64>> {
        let (tw, th) = self.template_size;

        let x0 = bbox.x.max(0.0) as usize;
        let y0 = bbox.y.max(0.0) as usize;
        let x1 = (bbox.x + bbox.width).min(width as f32) as usize;
        let y1 = (bbox.y + bbox.height).min(height as f32) as usize;

        if x1 <= x0 || y1 <= y0 {
            return Err(CvError::tracking_error("Invalid bounding box"));
        }

        let mut features = vec![0.0; tw * th];

        // Resize to template size
        for y in 0..th {
            for x in 0..tw {
                let src_x = x0 + (x * (x1 - x0)) / tw;
                let src_y = y0 + (y * (y1 - y0)) / th;

                if src_x < width as usize && src_y < height as usize {
                    let idx = src_y * width as usize + src_x;
                    if idx < frame.len() {
                        features[y * tw + x] = frame[idx] as f64;
                    }
                }
            }
        }

        // Normalize
        normalize_features(&mut features);

        Ok(features)
    }

    /// Classify a patch using nearest neighbor.
    fn classify_patch(&self, features: &[f64]) -> f64 {
        if self.positive_patches.is_empty() {
            return 0.0;
        }

        // Find max similarity to positive patches
        let pos_similarity = self
            .positive_patches
            .iter()
            .map(|patch| compute_similarity(&patch.features, features))
            .fold(f64::NEG_INFINITY, f64::max);

        // Find max similarity to negative patches
        let neg_similarity = if self.negative_patches.is_empty() {
            0.0
        } else {
            self.negative_patches
                .iter()
                .map(|patch| compute_similarity(&patch.features, features))
                .fold(f64::NEG_INFINITY, f64::max)
        };

        // Confidence is relative similarity to positive vs negative
        if pos_similarity > neg_similarity {
            pos_similarity
        } else {
            0.0
        }
    }

    /// Sample negative bounding boxes around current bbox.
    fn sample_negative_boxes(&self, width: u32, height: u32) -> Vec<BoundingBox> {
        let mut boxes = Vec::new();

        let margin = self.bbox.width.max(self.bbox.height) * 2.0;

        // Sample boxes around the current bbox
        for _ in 0..10 {
            let offset_x = (rand_float() - 0.5) * margin;
            let offset_y = (rand_float() - 0.5) * margin;

            let x = (self.bbox.x + offset_x)
                .max(0.0)
                .min(width as f32 - self.bbox.width);
            let y = (self.bbox.y + offset_y)
                .max(0.0)
                .min(height as f32 - self.bbox.height);

            let test_bbox = BoundingBox::new(x, y, self.bbox.width, self.bbox.height);

            // Only add if not overlapping much with current bbox
            let iou = super::assignment::compute_iou(&self.bbox, &test_bbox);
            if iou < 0.3 {
                boxes.push(test_bbox);
            }
        }

        boxes
    }
}

/// Normalize features to zero mean and unit variance.
fn normalize_features(features: &mut [f64]) {
    let n = features.len() as f64;
    let mean = features.iter().sum::<f64>() / n;
    let variance = features
        .iter()
        .map(|&x| (x - mean) * (x - mean))
        .sum::<f64>()
        / n;
    let std = (variance + 1e-5).sqrt();

    for val in features {
        *val = (*val - mean) / std;
    }
}

/// Compute normalized cross-correlation between two feature vectors.
fn compute_similarity(features1: &[f64], features2: &[f64]) -> f64 {
    if features1.len() != features2.len() {
        return 0.0;
    }

    let n = features1.len() as f64;
    let dot_product: f64 = features1
        .iter()
        .zip(features2.iter())
        .map(|(a, b)| a * b)
        .sum();

    // Features are already normalized, so NCC = dot product / n
    dot_product / n
}

/// Non-maximum suppression for bounding boxes.
fn non_max_suppression(
    detections: &[(BoundingBox, f64)],
    iou_threshold: f64,
) -> Vec<(BoundingBox, f64)> {
    if detections.is_empty() {
        return Vec::new();
    }

    // Sort by confidence
    let mut sorted = detections.to_vec();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut keep = Vec::new();

    while !sorted.is_empty() {
        let current = sorted.remove(0);
        keep.push(current);

        // Remove overlapping boxes
        sorted.retain(|(bbox, _)| {
            let iou = super::assignment::compute_iou(&current.0, bbox);
            iou < iou_threshold
        });
    }

    keep
}

/// Simple random float generator (0.0 to 1.0).
fn rand_float() -> f32 {
    // Very simple PRNG (not cryptographically secure)
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();
    (nanos % 1000) as f32 / 1000.0
}
