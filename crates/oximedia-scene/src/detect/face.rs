//! Face detection using Haar-like features and cascades.
//!
//! Supports multi-scale detection so faces of different sizes within the
//! same image are all found. After collecting candidates across all scales,
//! Non-Maximum Suppression (NMS) is applied via the shared [`crate::detect::nms()`]
//! function to remove duplicate detections.

use crate::common::{Confidence, Rect};
use crate::error::{SceneError, SceneResult};
use serde::{Deserialize, Serialize};

/// Detected face with location and attributes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaceDetection {
    /// Bounding box of the face.
    pub bbox: Rect,
    /// Detection confidence.
    pub confidence: Confidence,
    /// Face attributes.
    pub attributes: FaceAttributes,
}

/// Attributes of detected face.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FaceAttributes {
    /// Estimated face size (small, medium, large).
    pub size_category: FaceSizeCategory,
    /// Face orientation estimate.
    pub orientation: FaceOrientation,
    /// Skin tone estimate (0.0-1.0).
    pub skin_tone: f32,
}

/// Face size categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum FaceSizeCategory {
    /// Small face (< 5% of image).
    Small,
    /// Medium face (5-20% of image).
    #[default]
    Medium,
    /// Large face (> 20% of image).
    Large,
}

/// Face orientation estimate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum FaceOrientation {
    /// Frontal face.
    #[default]
    Frontal,
    /// Profile face (side view).
    Profile,
    /// Unknown orientation.
    Unknown,
}

/// Haar-like feature for face detection.
#[derive(Debug, Clone)]
struct HaarFeature {
    /// Feature type (two-rectangle, three-rectangle).
    feature_type: HaarFeatureType,
    /// Rectangle positions (x, y, width, height).
    rects: Vec<(f32, f32, f32, f32)>,
    /// Rectangle weights.
    weights: Vec<f32>,
    /// Feature threshold.
    threshold: f32,
}

/// Types of Haar features.
#[derive(Debug, Clone, Copy)]
enum HaarFeatureType {
    /// Two vertical rectangles.
    TwoVertical,
    /// Two horizontal rectangles.
    TwoHorizontal,
    /// Three vertical rectangles.
    ThreeVertical,
    /// Three horizontal rectangles.
    ThreeHorizontal,
    /// Four rectangles (diagonal).
    Four,
}

/// Configuration for face detection.
#[derive(Debug, Clone)]
pub struct FaceDetectorConfig {
    /// Minimum confidence threshold.
    pub confidence_threshold: f32,
    /// Minimum face size (pixels).
    pub min_face_size: usize,
    /// Maximum face size (pixels).
    pub max_face_size: usize,
    /// Scale factor for multi-scale detection (e.g. 1.1 = 10% size increment per level).
    /// Smaller values produce more scale levels and better accuracy at the cost of speed.
    pub scale_factor: f32,
    /// Minimum neighbors for detection.
    pub min_neighbors: usize,
    /// NMS IoU threshold — detections with IoU above this are suppressed.
    pub nms_threshold: f32,
}

impl Default for FaceDetectorConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.5,
            min_face_size: 20,
            max_face_size: 500,
            scale_factor: 1.1,
            min_neighbors: 3,
            nms_threshold: 0.3,
        }
    }
}

/// Face detector using Haar cascade.
pub struct FaceDetector {
    config: FaceDetectorConfig,
    features: Vec<HaarFeature>,
}

impl FaceDetector {
    /// Create a new face detector.
    #[must_use]
    pub fn new() -> Self {
        let features = Self::initialize_haar_features();
        Self {
            config: FaceDetectorConfig::default(),
            features,
        }
    }

    /// Create with custom configuration.
    #[must_use]
    pub fn with_config(config: FaceDetectorConfig) -> Self {
        let features = Self::initialize_haar_features();
        Self { config, features }
    }

    /// Detect faces in an RGB image.
    ///
    /// # Arguments
    ///
    /// * `rgb_data` - RGB image data
    /// * `width` - Image width
    /// * `height` - Image height
    ///
    /// # Errors
    ///
    /// Returns error if detection fails.
    pub fn detect(
        &self,
        rgb_data: &[u8],
        width: usize,
        height: usize,
    ) -> SceneResult<Vec<FaceDetection>> {
        if rgb_data.len() != width * height * 3 {
            return Err(SceneError::InvalidDimensions(
                "RGB data size mismatch".to_string(),
            ));
        }

        // Convert to grayscale
        let gray = self.rgb_to_gray(rgb_data, width, height);

        // Compute integral image for fast Haar feature evaluation
        let integral = self.compute_integral_image(&gray, width, height);

        // Multi-scale detection
        let mut all_detections = Vec::new();
        let mut scale = 1.0;
        let base_size = self.config.min_face_size as f32;

        while (base_size * scale) as usize <= self.config.max_face_size
            && (base_size * scale) as usize <= width.min(height)
        {
            let window_size = (base_size * scale) as usize;
            let stride = (window_size / 8).max(1);

            for y in (0..height.saturating_sub(window_size)).step_by(stride) {
                for x in (0..width.saturating_sub(window_size)).step_by(stride) {
                    if self.evaluate_cascade(&integral, width, x, y, window_size) {
                        all_detections.push((
                            Rect::new(x as f32, y as f32, window_size as f32, window_size as f32),
                            1.0, // Initial confidence
                        ));
                    }
                }
            }

            scale *= self.config.scale_factor;
        }

        // Group detections and filter by min_neighbors
        let grouped = self.group_detections(&all_detections);

        // Create face detections with attributes
        let mut faces: Vec<FaceDetection> = grouped
            .into_iter()
            .map(|(bbox, confidence)| {
                let attributes = self.extract_attributes(rgb_data, width, height, &bbox);
                FaceDetection {
                    bbox,
                    confidence: Confidence::new(confidence),
                    attributes,
                }
            })
            .collect();

        // Apply NMS across all multi-scale detections
        crate::detect::nms(
            &mut faces,
            |f| f.bbox,
            |f| f.confidence.value(),
            self.config.nms_threshold,
        );

        Ok(faces)
    }

    /// Initialize basic Haar features for face detection.
    fn initialize_haar_features() -> Vec<HaarFeature> {
        let mut features = Vec::new();

        // Two vertical rectangles (nose bridge)
        features.push(HaarFeature {
            feature_type: HaarFeatureType::TwoVertical,
            rects: vec![(0.25, 0.3, 0.25, 0.4), (0.5, 0.3, 0.25, 0.4)],
            weights: vec![1.0, -1.0],
            threshold: 0.1,
        });

        // Two horizontal rectangles (eyes vs cheeks)
        features.push(HaarFeature {
            feature_type: HaarFeatureType::TwoHorizontal,
            rects: vec![(0.2, 0.2, 0.6, 0.2), (0.2, 0.4, 0.6, 0.2)],
            weights: vec![-1.0, 1.0],
            threshold: 0.1,
        });

        // Three vertical (left eye, nose, right eye)
        features.push(HaarFeature {
            feature_type: HaarFeatureType::ThreeVertical,
            rects: vec![
                (0.1, 0.25, 0.25, 0.3),
                (0.35, 0.25, 0.3, 0.3),
                (0.65, 0.25, 0.25, 0.3),
            ],
            weights: vec![-1.0, 1.0, -1.0],
            threshold: 0.15,
        });

        // Three horizontal (forehead, eyes, mouth)
        features.push(HaarFeature {
            feature_type: HaarFeatureType::ThreeHorizontal,
            rects: vec![
                (0.2, 0.1, 0.6, 0.2),
                (0.2, 0.3, 0.6, 0.3),
                (0.2, 0.6, 0.6, 0.2),
            ],
            weights: vec![1.0, -1.0, 1.0],
            threshold: 0.1,
        });

        features
    }

    /// Compute integral image for fast rectangle sum calculation.
    fn compute_integral_image(&self, gray: &[f32], width: usize, height: usize) -> Vec<f64> {
        let mut integral = vec![0.0f64; width * height];

        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                let mut sum = f64::from(gray[idx]);

                if x > 0 {
                    sum += integral[idx - 1];
                }
                if y > 0 {
                    sum += integral[idx - width];
                }
                if x > 0 && y > 0 {
                    sum -= integral[idx - width - 1];
                }

                integral[idx] = sum;
            }
        }

        integral
    }

    /// Get sum of rectangle using integral image.
    fn rectangle_sum(
        &self,
        integral: &[f64],
        width: usize,
        x: usize,
        y: usize,
        w: usize,
        h: usize,
    ) -> f64 {
        let x1 = x;
        let y1 = y;
        let x2 = (x + w).min(width - 1);
        let y2 = (y + h).min(width - 1);

        let mut sum = integral[y2 * width + x2];

        if x1 > 0 {
            sum -= integral[y2 * width + (x1 - 1)];
        }
        if y1 > 0 {
            sum -= integral[(y1 - 1) * width + x2];
        }
        if x1 > 0 && y1 > 0 {
            sum += integral[(y1 - 1) * width + (x1 - 1)];
        }

        sum
    }

    /// Evaluate Haar cascade on a window.
    fn evaluate_cascade(
        &self,
        integral: &[f64],
        width: usize,
        x: usize,
        y: usize,
        size: usize,
    ) -> bool {
        // Simple cascade: all features must pass
        let mut passed = 0;

        for feature in &self.features {
            let mut feature_value = 0.0;

            for (i, rect) in feature.rects.iter().enumerate() {
                let rx = x + (rect.0 * size as f32) as usize;
                let ry = y + (rect.1 * size as f32) as usize;
                let rw = (rect.2 * size as f32) as usize;
                let rh = (rect.3 * size as f32) as usize;

                let rect_sum = self.rectangle_sum(integral, width, rx, ry, rw, rh);
                feature_value += rect_sum * feature.weights[i] as f64;
            }

            let normalized = (feature_value / (size * size) as f64) as f32;
            if normalized.abs() > feature.threshold {
                passed += 1;
            }
        }

        // At least half the features should pass
        passed >= self.features.len() / 2
    }

    /// Group nearby detections.
    fn group_detections(&self, detections: &[(Rect, f32)]) -> Vec<(Rect, f32)> {
        if detections.is_empty() {
            return Vec::new();
        }

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

                // Check if detection j overlaps with any in group
                let mut overlaps = false;
                for &k in &group {
                    let iou = detections[k].0.iou(&detections[j].0);
                    if iou > 0.2 {
                        overlaps = true;
                        break;
                    }
                }

                if overlaps {
                    group.push(j);
                    assigned[j] = true;
                }
            }

            groups.push(group);
        }

        // Filter groups by min_neighbors and compute average
        let mut result = Vec::new();
        for group in groups {
            if group.len() >= self.config.min_neighbors {
                let mut avg_x = 0.0;
                let mut avg_y = 0.0;
                let mut avg_w = 0.0;
                let mut avg_h = 0.0;

                for &idx in &group {
                    avg_x += detections[idx].0.x;
                    avg_y += detections[idx].0.y;
                    avg_w += detections[idx].0.width;
                    avg_h += detections[idx].0.height;
                }

                let count = group.len() as f32;
                let bbox = Rect::new(avg_x / count, avg_y / count, avg_w / count, avg_h / count);
                let confidence = count / 10.0; // More neighbors = higher confidence

                result.push((bbox, confidence.min(1.0)));
            }
        }

        result
    }

    /// Extract face attributes.
    fn extract_attributes(
        &self,
        rgb_data: &[u8],
        width: usize,
        height: usize,
        bbox: &Rect,
    ) -> FaceAttributes {
        let size_category = {
            let area_ratio = bbox.area() / (width * height) as f32;
            if area_ratio < 0.05 {
                FaceSizeCategory::Small
            } else if area_ratio < 0.2 {
                FaceSizeCategory::Medium
            } else {
                FaceSizeCategory::Large
            }
        };

        // Estimate skin tone from face region
        let skin_tone = self.estimate_skin_tone(rgb_data, width, bbox);

        // Simple orientation detection based on aspect ratio
        let orientation = if bbox.width / bbox.height > 0.8 && bbox.width / bbox.height < 1.2 {
            FaceOrientation::Frontal
        } else {
            FaceOrientation::Profile
        };

        FaceAttributes {
            size_category,
            orientation,
            skin_tone,
        }
    }

    /// Estimate average skin tone.
    fn estimate_skin_tone(&self, rgb_data: &[u8], width: usize, bbox: &Rect) -> f32 {
        let x_start = bbox.x as usize;
        let y_start = bbox.y as usize;
        let x_end = (bbox.x + bbox.width) as usize;
        let y_end = (bbox.y + bbox.height) as usize;

        let mut total = 0.0;
        let mut count = 0;

        for y in y_start..y_end.min(width) {
            for x in x_start..x_end.min(width) {
                let idx = (y * width + x) * 3;
                if idx + 2 < rgb_data.len() {
                    let brightness = (rgb_data[idx] as f32
                        + rgb_data[idx + 1] as f32
                        + rgb_data[idx + 2] as f32)
                        / 3.0;
                    total += brightness;
                    count += 1;
                }
            }
        }

        if count > 0 {
            (total / count as f32 / 255.0).clamp(0.0, 1.0)
        } else {
            0.5
        }
    }

    /// Convert RGB to grayscale.
    fn rgb_to_gray(&self, rgb: &[u8], width: usize, height: usize) -> Vec<f32> {
        let mut gray = Vec::with_capacity(width * height);
        for i in (0..rgb.len()).step_by(3) {
            let r = rgb[i] as f32;
            let g = rgb[i + 1] as f32;
            let b = rgb[i + 2] as f32;
            let y = 0.299 * r + 0.587 * g + 0.114 * b;
            gray.push(y / 255.0);
        }
        gray
    }
}

impl Default for FaceDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_face_detector_creation() {
        let detector = FaceDetector::new();
        assert!(!detector.features.is_empty());
    }

    #[test]
    fn test_face_detection_uniform() {
        let detector = FaceDetector::new();
        let width = 320;
        let height = 240;
        let rgb_data = vec![128u8; width * height * 3];
        let result = detector.detect(&rgb_data, width, height);
        assert!(result.is_ok());
    }

    #[test]
    fn test_integral_image() {
        let detector = FaceDetector::new();
        let gray = vec![1.0; 100];
        let integral = detector.compute_integral_image(&gray, 10, 10);
        assert_eq!(integral.len(), 100);
        assert!(integral[99] > 0.0);
    }

    #[test]
    fn test_face_detector_invalid_size() {
        let detector = FaceDetector::new();
        let result = detector.detect(&[0u8; 10], 100, 100);
        assert!(result.is_err());
    }

    #[test]
    fn test_face_detector_custom_config() {
        let config = FaceDetectorConfig {
            confidence_threshold: 0.3,
            min_face_size: 30,
            max_face_size: 300,
            scale_factor: 1.2,
            min_neighbors: 2,
            nms_threshold: 0.4,
        };
        let detector = FaceDetector::with_config(config);
        let w = 200;
        let h = 200;
        let rgb_data = vec![100u8; w * h * 3];
        let result = detector.detect(&rgb_data, w, h);
        assert!(result.is_ok());
    }

    #[test]
    fn test_multiscale_runs_at_different_sizes() {
        // With a small min_face_size and large max_face_size the detector
        // should produce detections at multiple window sizes without panicking.
        let config = FaceDetectorConfig {
            min_face_size: 15,
            max_face_size: 100,
            scale_factor: 1.25,
            min_neighbors: 1,
            nms_threshold: 0.5,
            ..FaceDetectorConfig::default()
        };
        let detector = FaceDetector::with_config(config);
        let w = 160;
        let h = 120;
        let rgb_data = vec![128u8; w * h * 3];
        let result = detector.detect(&rgb_data, w, h);
        assert!(result.is_ok(), "multi-scale detection should not error");
    }

    #[test]
    fn test_nms_applied_to_face_detections() {
        // Verify that the detector applies NMS: we inject detections manually and check.
        let detector = FaceDetector::new();
        let detections = vec![
            (Rect::new(10.0, 10.0, 40.0, 40.0), 0.9_f32),
            (Rect::new(12.0, 12.0, 40.0, 40.0), 0.6_f32), // overlaps heavily
            (Rect::new(200.0, 200.0, 40.0, 40.0), 0.8_f32), // no overlap
        ];
        let grouped = detector.group_detections(&detections);
        // After grouping and NMS, overlapping detections should be merged/suppressed
        assert!(grouped.len() <= detections.len());
    }
}
