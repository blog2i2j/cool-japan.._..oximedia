//! Lightweight object detection for media content analysis.
//!
//! This module provides bounding-box primitives, a detection result type, and a
//! rule-based [`ObjectDetector`] that derives candidate detections from raw image
//! statistics without requiring a trained ML model.  The design is intentionally
//! deterministic so that unit tests can make precise assertions.
//!
//! # Non-Maximum Suppression
//!
//! [`ObjectDetector::nms`] implements the standard greedy NMS algorithm:
//! 1. Sort detections by confidence (descending).
//! 2. Greedily keep the top detection.
//! 3. Suppress any remaining detection whose IoU with the kept box exceeds
//!    [`NmsConfig::iou_threshold`].
//! 4. Repeat until the list is exhausted.

// ─────────────────────────────────────────────────────────────────────────────
// BoundingBox
// ─────────────────────────────────────────────────────────────────────────────

/// An axis-aligned bounding box with coordinates normalised to `[0, 1]`.
///
/// `x` and `y` are the coordinates of the **top-left** corner; `width` and
/// `height` give the extent.  All four values are clamped to `[0, 1]` during
/// construction via [`BoundingBox::new`].
#[derive(Debug, Clone, PartialEq)]
pub struct BoundingBox {
    /// Normalised x coordinate of the top-left corner (`[0, 1]`).
    pub x: f32,
    /// Normalised y coordinate of the top-left corner (`[0, 1]`).
    pub y: f32,
    /// Normalised width (`[0, 1]`).
    pub width: f32,
    /// Normalised height (`[0, 1]`).
    pub height: f32,
}

impl BoundingBox {
    /// Creates a new `BoundingBox`, clamping all values to `[0, 1]`.
    ///
    /// Negative extents are treated as zero.
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        let x = x.clamp(0.0, 1.0);
        let y = y.clamp(0.0, 1.0);
        let width = width.max(0.0).min(1.0 - x);
        let height = height.max(0.0).min(1.0 - y);
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Returns the area of the bounding box (in normalised units²).
    #[inline]
    pub fn area(&self) -> f32 {
        self.width * self.height
    }

    /// Computes the **Intersection over Union** (IoU / Jaccard index) with
    /// another bounding box.
    ///
    /// Returns a value in `[0, 1]`.  Returns `0.0` for degenerate (zero-area)
    /// boxes so that the function is always well-defined.
    pub fn iou(&self, other: &BoundingBox) -> f32 {
        // Compute intersection rectangle.
        let ix1 = self.x.max(other.x);
        let iy1 = self.y.max(other.y);
        let ix2 = (self.x + self.width).min(other.x + other.width);
        let iy2 = (self.y + self.height).min(other.y + other.height);

        let inter_w = (ix2 - ix1).max(0.0);
        let inter_h = (iy2 - iy1).max(0.0);
        let intersection = inter_w * inter_h;

        let union = self.area() + other.area() - intersection;
        if union <= 0.0 {
            return 0.0;
        }
        intersection / union
    }

    /// Returns `true` if the box has positive area.
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.width > 0.0 && self.height > 0.0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DetectionClass
// ─────────────────────────────────────────────────────────────────────────────

/// Semantic category of a detected region.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DetectionClass {
    /// A human body.
    Person,
    /// A human face (sub-category of Person).
    Face,
    /// Any motorised or non-motorised vehicle.
    Vehicle,
    /// Any non-human animal.
    Animal,
    /// Overlaid text, lower-thirds, or OCR-detectable text.
    Text,
    /// A watermark, station bug, or channel logo.
    Logo,
    /// An arbitrary object identified by its label string.
    Object(String),
}

impl DetectionClass {
    /// Returns a human-readable label for the class.
    pub fn label(&self) -> &str {
        match self {
            Self::Person => "person",
            Self::Face => "face",
            Self::Vehicle => "vehicle",
            Self::Animal => "animal",
            Self::Text => "text",
            Self::Logo => "logo",
            Self::Object(s) => s.as_str(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Detection
// ─────────────────────────────────────────────────────────────────────────────

/// A single object detection result.
#[derive(Debug, Clone)]
pub struct Detection {
    /// Semantic class of the detected object.
    pub class: DetectionClass,
    /// Confidence score in `[0, 1]`.
    pub confidence: f32,
    /// Predicted bounding box in normalised image coordinates.
    pub bbox: BoundingBox,
}

impl Detection {
    /// Creates a new `Detection`.
    pub fn new(class: DetectionClass, confidence: f32, bbox: BoundingBox) -> Self {
        Self {
            class,
            confidence: confidence.clamp(0.0, 1.0),
            bbox,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NmsConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for non-maximum suppression.
#[derive(Debug, Clone)]
pub struct NmsConfig {
    /// IoU overlap threshold above which two boxes are considered duplicates.
    /// Boxes with IoU **strictly greater** than this value are suppressed.
    /// Default: `0.45`.
    pub iou_threshold: f32,
    /// Minimum confidence score required to keep a detection.
    /// Detections with confidence **strictly less** than this value are removed
    /// before NMS begins.  Default: `0.3`.
    pub score_threshold: f32,
}

impl NmsConfig {
    /// Creates an `NmsConfig` with the supplied thresholds.
    pub fn new(iou_threshold: f32, score_threshold: f32) -> Self {
        Self {
            iou_threshold: iou_threshold.clamp(0.0, 1.0),
            score_threshold: score_threshold.clamp(0.0, 1.0),
        }
    }
}

impl Default for NmsConfig {
    fn default() -> Self {
        Self {
            iou_threshold: 0.45,
            score_threshold: 0.30,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ObjectDetector
// ─────────────────────────────────────────────────────────────────────────────

/// Rule-based object detector that derives candidate detections from image
/// statistics.
///
/// Because this detector does not embed a real neural-network model, its
/// primary purpose is **testing the detection pipeline** (NMS, IoU filtering,
/// coordinate normalisation) against deterministic inputs.  Real deployments
/// would replace [`ObjectDetector::detect`] with a proper forward pass.
///
/// ## Detection heuristics
///
/// The detector computes the following statistics over the raw `u8` pixel
/// buffer (assumed to be packed RGB):
///
/// | Statistic | Detected as |
/// |-----------|-------------|
/// | Mean luminance > 200 (very bright) | Logo (top-right quadrant) |
/// | Mean luminance < 50 (very dark) | Object("dark_region") |
/// | High spatial variance > 3000 | Person (centre) |
/// | Low spatial variance < 50 | Text (bottom strip) |
/// | Strong horizontal gradient energy | Vehicle (left strip) |
/// | Strong vertical gradient energy | Face (upper-centre) |
/// | Near-uniform hue (small std-dev) | Animal (full frame) |
pub struct ObjectDetector {
    nms_config: NmsConfig,
}

impl ObjectDetector {
    /// Creates an `ObjectDetector` with default NMS settings.
    pub fn new() -> Self {
        Self {
            nms_config: NmsConfig::default(),
        }
    }

    /// Creates an `ObjectDetector` with custom NMS settings.
    pub fn with_config(nms_config: NmsConfig) -> Self {
        Self { nms_config }
    }

    /// Returns a reference to the NMS configuration.
    pub fn nms_config(&self) -> &NmsConfig {
        &self.nms_config
    }

    /// Derives candidate detections from raw image bytes.
    ///
    /// `image` is expected to be packed **RGB** bytes (3 bytes per pixel).
    /// If the slice length does not match `width × height × 3`, the detector
    /// falls back to treating the image as single-channel and re-interprets
    /// the statistics accordingly.
    ///
    /// Returns detections **before** NMS; call [`ObjectDetector::nms`] to
    /// filter duplicates.
    pub fn detect(&self, image: &[u8], width: u32, height: u32) -> Vec<Detection> {
        if image.is_empty() || width == 0 || height == 0 {
            return Vec::new();
        }

        let stats = ImageStats::compute(image, width, height);
        let mut detections = Vec::new();

        // ── Heuristic 1: very bright region → logo (top-right quadrant) ──────
        if stats.mean_luma > 200.0 {
            detections.push(Detection::new(
                DetectionClass::Logo,
                (stats.mean_luma / 255.0).min(1.0),
                BoundingBox::new(0.65, 0.0, 0.35, 0.25),
            ));
        }

        // ── Heuristic 2: very dark region → generic dark object ───────────────
        if stats.mean_luma < 50.0 {
            detections.push(Detection::new(
                DetectionClass::Object("dark_region".to_string()),
                1.0 - stats.mean_luma / 255.0,
                BoundingBox::new(0.1, 0.1, 0.8, 0.8),
            ));
        }

        // ── Heuristic 3: high spatial variance → person (centre) ─────────────
        if stats.spatial_variance > 3000.0 {
            let conf = (stats.spatial_variance / 10_000.0).min(1.0);
            detections.push(Detection::new(
                DetectionClass::Person,
                conf,
                BoundingBox::new(0.25, 0.1, 0.5, 0.8),
            ));
        }

        // ── Heuristic 4: low spatial variance → text (bottom strip) ──────────
        if stats.spatial_variance < 50.0 && stats.mean_luma >= 50.0 {
            detections.push(Detection::new(
                DetectionClass::Text,
                0.75,
                BoundingBox::new(0.05, 0.85, 0.9, 0.1),
            ));
        }

        // ── Heuristic 5: horizontal gradient energy dominates → vehicle ───────
        if stats.horiz_energy > stats.vert_energy * 1.5 && stats.horiz_energy > 500.0 {
            let conf = (stats.horiz_energy / (stats.horiz_energy + stats.vert_energy)).min(1.0);
            detections.push(Detection::new(
                DetectionClass::Vehicle,
                conf,
                BoundingBox::new(0.0, 0.2, 0.45, 0.6),
            ));
        }

        // ── Heuristic 6: vertical gradient energy dominates → face ────────────
        if stats.vert_energy > stats.horiz_energy * 1.5 && stats.vert_energy > 500.0 {
            let conf = (stats.vert_energy / (stats.horiz_energy + stats.vert_energy)).min(1.0);
            detections.push(Detection::new(
                DetectionClass::Face,
                conf,
                BoundingBox::new(0.3, 0.05, 0.4, 0.45),
            ));
        }

        // ── Heuristic 7: near-uniform hue → animal (full frame) ───────────────
        if stats.channel_std_dev < 15.0 && stats.mean_luma > 50.0 && stats.mean_luma < 200.0 {
            let conf = 1.0 - stats.channel_std_dev / 30.0;
            detections.push(Detection::new(
                DetectionClass::Animal,
                conf.max(0.0),
                BoundingBox::new(0.1, 0.1, 0.8, 0.8),
            ));
        }

        detections
    }

    /// Applies **non-maximum suppression** to a list of detections.
    ///
    /// The algorithm:
    /// 1. Remove all detections with `confidence < config.score_threshold`.
    /// 2. Sort remaining detections by confidence (descending).
    /// 3. Greedily keep the top detection; suppress any detection with
    ///    IoU > `config.iou_threshold` with respect to the kept box.
    /// 4. Repeat until all detections are processed.
    ///
    /// This is the standard greedy NMS used in most detection pipelines.
    pub fn nms(detections: Vec<Detection>, config: &NmsConfig) -> Vec<Detection> {
        // Step 1: apply score threshold.
        let mut candidates: Vec<Detection> = detections
            .into_iter()
            .filter(|d| d.confidence >= config.score_threshold)
            .collect();

        // Step 2: sort by confidence descending.
        candidates.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Step 3-4: greedy suppression.
        let mut kept: Vec<Detection> = Vec::with_capacity(candidates.len());
        let mut suppressed = vec![false; candidates.len()];

        for i in 0..candidates.len() {
            if suppressed[i] {
                continue;
            }
            // Keep this detection.
            for j in (i + 1)..candidates.len() {
                if suppressed[j] {
                    continue;
                }
                let iou = candidates[i].bbox.iou(&candidates[j].bbox);
                if iou > config.iou_threshold {
                    suppressed[j] = true;
                }
            }
            // Defer the push to after the inner loop to avoid borrow issues.
            suppressed[i] = true; // mark processed
            kept.push(candidates[i].clone());
        }

        kept
    }
}

impl Default for ObjectDetector {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal image statistics helper
// ─────────────────────────────────────────────────────────────────────────────

/// Aggregate statistics extracted from a raw pixel buffer.
struct ImageStats {
    mean_luma: f32,
    spatial_variance: f32,
    horiz_energy: f32,
    vert_energy: f32,
    channel_std_dev: f32,
}

impl ImageStats {
    fn compute(image: &[u8], width: u32, height: u32) -> Self {
        let total_pixels = (width as usize) * (height as usize);
        let channels = if image.len() >= total_pixels * 3 {
            3usize
        } else {
            1
        };

        // Mean luminance (BT.601 luma coefficients for RGB; single-channel fallback).
        let mean_luma = if channels == 3 {
            let sum: f64 = image
                .chunks_exact(3)
                .take(total_pixels)
                .map(|px| 0.299 * px[0] as f64 + 0.587 * px[1] as f64 + 0.114 * px[2] as f64)
                .sum();
            (sum / total_pixels.max(1) as f64) as f32
        } else {
            let sum: f64 = image.iter().take(total_pixels).map(|&b| b as f64).sum();
            (sum / total_pixels.max(1) as f64) as f32
        };

        // Spatial variance of luma values.
        let luma_vals: Vec<f32> = if channels == 3 {
            image
                .chunks_exact(3)
                .take(total_pixels)
                .map(|px| 0.299 * px[0] as f32 + 0.587 * px[1] as f32 + 0.114 * px[2] as f32)
                .collect()
        } else {
            image.iter().take(total_pixels).map(|&b| b as f32).collect()
        };

        let var_sum: f32 = luma_vals.iter().map(|&v| (v - mean_luma).powi(2)).sum();
        let spatial_variance = var_sum / total_pixels.max(1) as f32;

        // Horizontal gradient energy: sum of |luma[i+1] - luma[i]| along rows.
        let w = width as usize;
        let h = height as usize;
        let mut horiz_energy = 0.0f32;
        let mut vert_energy = 0.0f32;

        for row in 0..h {
            for col in 0..w {
                let idx = row * w + col;
                if col + 1 < w {
                    horiz_energy += (luma_vals[idx + 1] - luma_vals[idx]).abs();
                }
                if row + 1 < h {
                    vert_energy += (luma_vals[idx + w] - luma_vals[idx]).abs();
                }
            }
        }
        // Normalise by pixel count so it's comparable across image sizes.
        horiz_energy /= total_pixels.max(1) as f32;
        vert_energy /= total_pixels.max(1) as f32;

        // Channel std-dev: std of mean channel values (R,G,B) to detect uniform hue.
        let channel_std_dev = if channels == 3 {
            let mut ch_means = [0.0f64; 3];
            for px in image.chunks_exact(3).take(total_pixels) {
                ch_means[0] += px[0] as f64;
                ch_means[1] += px[1] as f64;
                ch_means[2] += px[2] as f64;
            }
            let n = total_pixels.max(1) as f64;
            ch_means[0] /= n;
            ch_means[1] /= n;
            ch_means[2] /= n;
            let overall = (ch_means[0] + ch_means[1] + ch_means[2]) / 3.0;
            let variance = ch_means.iter().map(|&m| (m - overall).powi(2)).sum::<f64>() / 3.0;
            variance.sqrt() as f32
        } else {
            0.0
        };

        // Scale gradient energies to a useful range (multiply back by pixel count).
        let scale = total_pixels.max(1) as f32;
        ImageStats {
            mean_luma,
            spatial_variance,
            horiz_energy: horiz_energy * scale / 255.0,
            vert_energy: vert_energy * scale / 255.0,
            channel_std_dev,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── BoundingBox ──────────────────────────────────────────────────────────

    #[test]
    fn test_bbox_area_unit_square() {
        let b = BoundingBox::new(0.0, 0.0, 1.0, 1.0);
        assert!((b.area() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_bbox_area_half_square() {
        let b = BoundingBox::new(0.0, 0.0, 0.5, 0.5);
        assert!((b.area() - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_bbox_iou_identical_is_one() {
        let a = BoundingBox::new(0.1, 0.1, 0.5, 0.5);
        let b = BoundingBox::new(0.1, 0.1, 0.5, 0.5);
        assert!(
            (a.iou(&b) - 1.0).abs() < 1e-6,
            "IoU of identical boxes must be 1.0"
        );
    }

    #[test]
    fn test_bbox_iou_non_overlapping_is_zero() {
        let a = BoundingBox::new(0.0, 0.0, 0.3, 0.3);
        let b = BoundingBox::new(0.7, 0.7, 0.3, 0.3);
        assert!(
            a.iou(&b).abs() < 1e-6,
            "IoU of non-overlapping boxes must be 0.0"
        );
    }

    #[test]
    fn test_bbox_iou_partial_overlap() {
        // Two 0.5×0.5 boxes that overlap by 0.25×0.25 = 0.0625.
        let a = BoundingBox::new(0.0, 0.0, 0.5, 0.5);
        let b = BoundingBox::new(0.25, 0.25, 0.5, 0.5);
        let iou = a.iou(&b);
        // intersection = 0.25*0.25 = 0.0625; union = 0.25+0.25-0.0625 = 0.4375
        let expected = 0.0625_f32 / 0.4375;
        assert!(
            (iou - expected).abs() < 1e-5,
            "iou={iou}, expected={expected}"
        );
    }

    #[test]
    fn test_bbox_iou_zero_area_returns_zero() {
        let a = BoundingBox::new(0.0, 0.0, 0.0, 0.0); // zero area
        let b = BoundingBox::new(0.0, 0.0, 0.5, 0.5);
        assert!(a.iou(&b).abs() < 1e-6);
    }

    #[test]
    fn test_bbox_clamping() {
        let b = BoundingBox::new(-1.0, -1.0, 3.0, 3.0);
        assert!(b.x >= 0.0 && b.y >= 0.0);
        assert!(b.x + b.width <= 1.0 + 1e-6);
        assert!(b.y + b.height <= 1.0 + 1e-6);
    }

    // ── NMS ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_nms_removes_high_iou_duplicates() {
        // Two nearly-identical boxes for the same class; NMS should keep one.
        let a = Detection::new(
            DetectionClass::Person,
            0.9,
            BoundingBox::new(0.1, 0.1, 0.5, 0.5),
        );
        let b = Detection::new(
            DetectionClass::Person,
            0.85,
            BoundingBox::new(0.11, 0.11, 0.49, 0.49),
        );
        let config = NmsConfig::new(0.45, 0.3);
        let result = ObjectDetector::nms(vec![a, b], &config);
        assert_eq!(
            result.len(),
            1,
            "NMS must suppress the lower-confidence duplicate"
        );
        assert!((result[0].confidence - 0.9).abs() < 1e-5);
    }

    #[test]
    fn test_nms_keeps_non_overlapping_boxes() {
        let a = Detection::new(
            DetectionClass::Person,
            0.9,
            BoundingBox::new(0.0, 0.0, 0.3, 0.3),
        );
        let b = Detection::new(
            DetectionClass::Vehicle,
            0.85,
            BoundingBox::new(0.7, 0.7, 0.3, 0.3),
        );
        let config = NmsConfig::new(0.45, 0.3);
        let result = ObjectDetector::nms(vec![a, b], &config);
        assert_eq!(
            result.len(),
            2,
            "Non-overlapping boxes must both survive NMS"
        );
    }

    #[test]
    fn test_nms_score_threshold_filters_low_confidence() {
        let detections: Vec<Detection> = (0..5)
            .map(|i| {
                Detection::new(
                    DetectionClass::Object(format!("obj_{i}")),
                    0.1 * (i + 1) as f32,
                    BoundingBox::new(0.0, 0.0, 0.2, 0.2),
                )
            })
            .collect();
        let config = NmsConfig::new(0.45, 0.4); // threshold = 0.4
        let result = ObjectDetector::nms(detections, &config);
        // Only detections with confidence >= 0.4 should survive before NMS.
        // Confidences: 0.1, 0.2, 0.3, 0.4, 0.5 → keep 0.4 and 0.5.
        // But they share the same box so NMS keeps only the higher one (0.5).
        assert_eq!(result.len(), 1);
        assert!((result[0].confidence - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_nms_empty_input_returns_empty() {
        let config = NmsConfig::default();
        let result = ObjectDetector::nms(vec![], &config);
        assert!(result.is_empty());
    }

    #[test]
    fn test_nms_single_detection_always_kept() {
        let d = Detection::new(
            DetectionClass::Face,
            0.99,
            BoundingBox::new(0.3, 0.3, 0.4, 0.4),
        );
        let config = NmsConfig::default();
        let result = ObjectDetector::nms(vec![d], &config);
        assert_eq!(result.len(), 1);
    }

    // ── ObjectDetector ───────────────────────────────────────────────────────

    #[test]
    fn test_detector_empty_image_returns_empty() {
        let detector = ObjectDetector::new();
        let result = detector.detect(&[], 0, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_detector_bright_image_produces_logo() {
        // Very bright RGB image → Logo heuristic should fire.
        let pixels: Vec<u8> = vec![255u8; 16 * 16 * 3];
        let detector = ObjectDetector::new();
        let detections = detector.detect(&pixels, 16, 16);
        assert!(
            detections
                .iter()
                .any(|d| matches!(d.class, DetectionClass::Logo)),
            "Bright image must generate a Logo detection"
        );
    }

    #[test]
    fn test_detector_dark_image_produces_dark_region() {
        let pixels: Vec<u8> = vec![10u8; 16 * 16 * 3];
        let detector = ObjectDetector::new();
        let detections = detector.detect(&pixels, 16, 16);
        assert!(
            detections
                .iter()
                .any(|d| matches!(&d.class, DetectionClass::Object(s) if s == "dark_region")),
            "Dark image must generate a dark_region detection"
        );
    }

    #[test]
    fn test_detection_confidence_in_range() {
        // Any image should produce detections with confidence in [0,1].
        let pixels: Vec<u8> = (0u8..=255).cycle().take(32 * 32 * 3).collect();
        let detector = ObjectDetector::new();
        let detections = detector.detect(&pixels, 32, 32);
        for d in &detections {
            assert!(
                (0.0..=1.0).contains(&d.confidence),
                "confidence {} out of [0,1]",
                d.confidence
            );
        }
    }

    #[test]
    fn test_detection_class_label() {
        assert_eq!(DetectionClass::Person.label(), "person");
        assert_eq!(DetectionClass::Object("cat".to_string()).label(), "cat");
    }
}
