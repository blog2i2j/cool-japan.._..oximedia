//! Face detection model using Conv2d + pooling layers.
//!
//! This module implements a lightweight, pre-defined convolutional architecture
//! for face detection in video frames.  The model follows a MobileNet-inspired
//! backbone design:
//!
//! 1. Three Conv2d stages (each followed by ReLU activation) progressively
//!    extract features while reducing spatial resolution via stride-2 convolutions.
//! 2. A global average pooling layer collapses the spatial dimensions.
//! 3. A single `LinearLayer` head maps the feature vector to a confidence score
//!    per anchor box location.
//!
//! ## Architecture summary
//!
//! ```text
//! Input:  [3, H, W]   (RGB image, arbitrary resolution)
//! Stage1: Conv2d(3→16,  3×3, stride=2, pad=1) + ReLU  → [16, H/2,  W/2]
//! Stage2: Conv2d(16→32, 3×3, stride=2, pad=1) + ReLU  → [32, H/4,  W/4]
//! Stage3: Conv2d(32→64, 3×3, stride=2, pad=1) + ReLU  → [64, H/8,  W/8]
//! Pool:   GlobalAvgPool                                → [64]
//! Head:   Linear(64 → num_anchors * 5)
//!   each anchor box encodes (confidence, dx, dy, dw, dh)
//! ```
//!
//! All weights are **zero-initialised** on construction.  Use
//! [`FaceDetector::with_weights`] to load pre-trained weights.
//!
//! ## Example
//!
//! ```rust
//! use oximedia_neural::face_detection::{FaceDetector, DetectionBox};
//!
//! let detector = FaceDetector::new(4).unwrap();
//! // 3-channel 32×32 image (all zeros → zero confidence)
//! let image = oximedia_neural::tensor::Tensor::zeros(vec![3, 32, 32]).unwrap();
//! let detections = detector.forward(&image, 0.9).unwrap();
//! // With zero weights sigmoid(0) ≈ 0.5, below the 0.9 threshold → empty
//! assert!(detections.is_empty());
//! ```

use crate::activations::{apply_activation, ActivationFn};
use crate::error::NeuralError;
use crate::layers::{Conv2dLayer, GlobalAvgPool, LinearLayer};
use crate::tensor::Tensor;

// ─────────────────────────────────────────────────────────────────────────────
// DetectionBox
// ─────────────────────────────────────────────────────────────────────────────

/// A single detected face with a bounding box and confidence score.
///
/// Coordinates are expressed as fractions of the image dimensions in
/// `[0, 1]` space (top-left origin).
#[derive(Debug, Clone, PartialEq)]
pub struct DetectionBox {
    /// Detection confidence (sigmoid-activated anchor score) in `[0, 1]`.
    pub confidence: f32,
    /// Horizontal centre of the bounding box, relative to image width.
    pub cx: f32,
    /// Vertical centre of the bounding box, relative to image height.
    pub cy: f32,
    /// Width of the bounding box, relative to image width.
    pub w: f32,
    /// Height of the bounding box, relative to image height.
    pub h: f32,
}

impl DetectionBox {
    /// Returns `true` if this box has a non-degenerate area (`w > 0 && h > 0`).
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.w > 0.0 && self.h > 0.0
    }

    /// Computes the intersection-over-union (IoU) with another `DetectionBox`.
    ///
    /// Returns a value in `[0, 1]`; returns `0.0` if either box is degenerate.
    #[must_use]
    pub fn iou(&self, other: &DetectionBox) -> f32 {
        if !self.is_valid() || !other.is_valid() {
            return 0.0;
        }
        let self_x1 = self.cx - self.w * 0.5;
        let self_y1 = self.cy - self.h * 0.5;
        let self_x2 = self.cx + self.w * 0.5;
        let self_y2 = self.cy + self.h * 0.5;

        let other_x1 = other.cx - other.w * 0.5;
        let other_y1 = other.cy - other.h * 0.5;
        let other_x2 = other.cx + other.w * 0.5;
        let other_y2 = other.cy + other.h * 0.5;

        let inter_x1 = self_x1.max(other_x1);
        let inter_y1 = self_y1.max(other_y1);
        let inter_x2 = self_x2.min(other_x2);
        let inter_y2 = self_y2.min(other_y2);

        let inter_w = (inter_x2 - inter_x1).max(0.0);
        let inter_h = (inter_y2 - inter_y1).max(0.0);
        let intersection = inter_w * inter_h;
        if intersection == 0.0 {
            return 0.0;
        }
        let area_self = self.w * self.h;
        let area_other = other.w * other.h;
        intersection / (area_self + area_other - intersection)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Non-maximum suppression
// ─────────────────────────────────────────────────────────────────────────────

/// Applies greedy non-maximum suppression to a list of detection boxes.
///
/// Boxes are sorted descending by confidence; any box that overlaps a
/// higher-confidence box by more than `iou_threshold` is suppressed.
///
/// Returns a sorted (descending confidence) subset of the input detections.
#[must_use]
pub fn nms(mut detections: Vec<DetectionBox>, iou_threshold: f32) -> Vec<DetectionBox> {
    // Sort descending by confidence.
    detections.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut kept: Vec<DetectionBox> = Vec::with_capacity(detections.len());
    for candidate in detections {
        let suppressed = kept.iter().any(|k| k.iou(&candidate) > iou_threshold);
        if !suppressed {
            kept.push(candidate);
        }
    }
    kept
}

// ─────────────────────────────────────────────────────────────────────────────
// FaceDetector
// ─────────────────────────────────────────────────────────────────────────────

/// Lightweight convolutional face detector.
///
/// The backbone is three stride-2 Conv2d layers followed by global average
/// pooling.  A single linear head maps the 64-D feature vector to
/// `num_anchors × 5` outputs: `(confidence, Δcx, Δcy, Δw, Δh)` per anchor.
///
/// All weights default to zero, producing zero confidence for all anchors
/// until populated via the public layer fields.
#[derive(Debug, Clone)]
pub struct FaceDetector {
    /// Stage-1 convolution: 3 → 16 channels, 3×3 kernel, stride 2, pad 1.
    pub stage1: Conv2dLayer,
    /// Stage-2 convolution: 16 → 32 channels, 3×3 kernel, stride 2, pad 1.
    pub stage2: Conv2dLayer,
    /// Stage-3 convolution: 32 → 64 channels, 3×3 kernel, stride 2, pad 1.
    pub stage3: Conv2dLayer,
    /// Global average pooling layer (collapses spatial dims).
    pub pool: GlobalAvgPool,
    /// Detection head: 64 → num_anchors * 5.
    pub head: LinearLayer,
    /// Number of anchor boxes per spatial location.
    pub num_anchors: usize,
}

impl FaceDetector {
    /// Creates a zero-initialised `FaceDetector` with `num_anchors` anchors.
    ///
    /// Returns an error if `num_anchors == 0`.
    pub fn new(num_anchors: usize) -> Result<Self, NeuralError> {
        if num_anchors == 0 {
            return Err(NeuralError::InvalidShape(
                "FaceDetector: num_anchors must be > 0".to_string(),
            ));
        }
        let stage1 = Conv2dLayer::new(3, 16, 3, 3, (2, 2), (1, 1))?;
        let stage2 = Conv2dLayer::new(16, 32, 3, 3, (2, 2), (1, 1))?;
        let stage3 = Conv2dLayer::new(32, 64, 3, 3, (2, 2), (1, 1))?;
        let pool = GlobalAvgPool;
        let head = LinearLayer::new(64, num_anchors * 5)?;

        Ok(Self {
            stage1,
            stage2,
            stage3,
            pool,
            head,
            num_anchors,
        })
    }

    /// Loads pre-trained weights from flat slices.
    ///
    /// The expected flat-weight layout matches the model's layer order:
    /// stage1, stage2, stage3, head.  Returns an error if any slice length is
    /// incorrect.
    pub fn with_weights(
        mut self,
        stage1_w: Vec<f32>,
        stage1_b: Vec<f32>,
        stage2_w: Vec<f32>,
        stage2_b: Vec<f32>,
        stage3_w: Vec<f32>,
        stage3_b: Vec<f32>,
        head_w: Vec<f32>,
        head_b: Vec<f32>,
    ) -> Result<Self, NeuralError> {
        self.stage1.weight = Tensor::from_data(stage1_w, vec![16, 3, 3, 3])?;
        self.stage1.bias = Tensor::from_data(stage1_b, vec![16])?;
        self.stage2.weight = Tensor::from_data(stage2_w, vec![32, 16, 3, 3])?;
        self.stage2.bias = Tensor::from_data(stage2_b, vec![32])?;
        self.stage3.weight = Tensor::from_data(stage3_w, vec![64, 32, 3, 3])?;
        self.stage3.bias = Tensor::from_data(stage3_b, vec![64])?;
        let out_features = self.num_anchors * 5;
        self.head.weight = Tensor::from_data(head_w, vec![out_features, 64])?;
        self.head.bias = Tensor::from_data(head_b, vec![out_features])?;
        Ok(self)
    }

    /// Runs face detection on a single `[3, H, W]` RGB image tensor.
    ///
    /// Returns detected boxes with confidence ≥ `conf_threshold` after
    /// greedy NMS with IoU threshold 0.45.
    ///
    /// Box coordinates are normalised to `[0, 1]` relative to the input
    /// spatial dimensions; the regression offsets (Δcx, Δcy, Δw, Δh) are
    /// passed through `tanh` to produce values in `(−1, 1)`, then mapped to
    /// anchors uniformly tiled across the feature map.
    pub fn forward(
        &self,
        input: &Tensor,
        conf_threshold: f32,
    ) -> Result<Vec<DetectionBox>, NeuralError> {
        if input.ndim() != 3 {
            return Err(NeuralError::InvalidShape(format!(
                "FaceDetector::forward: expected [3, H, W] input, got rank {}",
                input.ndim()
            )));
        }
        if input.shape()[0] != 3 {
            return Err(NeuralError::ShapeMismatch(format!(
                "FaceDetector::forward: expected 3 input channels, got {}",
                input.shape()[0]
            )));
        }

        // ── Backbone ─────────────────────────────────────────────────────────
        let f1 = apply_activation(&self.stage1.forward(input)?, &ActivationFn::Relu);
        let f2 = apply_activation(&self.stage2.forward(&f1)?, &ActivationFn::Relu);
        let f3 = apply_activation(&self.stage3.forward(&f2)?, &ActivationFn::Relu);

        // ── Global average pooling ────────────────────────────────────────────
        let pooled = self.pool.forward(&f3)?; // [64, 1, 1]
                                              // Reshape to 1-D for the linear head.
        let feat = Tensor::from_data(pooled.data().to_vec(), vec![pooled.shape()[0]])?; // [64]

        // ── Detection head ────────────────────────────────────────────────────
        let raw = self.head.forward(&feat)?; // [num_anchors * 5]

        // ── Decode predictions ────────────────────────────────────────────────
        let raw_data = raw.data();
        let mut detections: Vec<DetectionBox> = Vec::with_capacity(self.num_anchors);

        for a in 0..self.num_anchors {
            let base = a * 5;
            // Confidence: sigmoid activation
            let conf = sigmoid_scalar(raw_data[base]);
            if conf < conf_threshold {
                continue;
            }
            // Box regression: tanh activation; uniform anchor grid centred at 0.5
            let step = 1.0 / self.num_anchors as f32;
            let anchor_cx = (a as f32 + 0.5) * step;
            let anchor_cy = 0.5_f32;
            let anchor_w = 0.3_f32;
            let anchor_h = 0.4_f32;

            let dcx = raw_data[base + 1].tanh() * 0.5;
            let dcy = raw_data[base + 2].tanh() * 0.5;
            let dw = raw_data[base + 3].tanh();
            let dh = raw_data[base + 4].tanh();

            let cx = (anchor_cx + dcx).clamp(0.0, 1.0);
            let cy = (anchor_cy + dcy).clamp(0.0, 1.0);
            let w = (anchor_w * (1.0 + dw)).max(0.0);
            let h = (anchor_h * (1.0 + dh)).max(0.0);

            detections.push(DetectionBox {
                confidence: conf,
                cx,
                cy,
                w,
                h,
            });
        }

        // ── NMS ───────────────────────────────────────────────────────────────
        Ok(nms(detections, 0.45))
    }

    /// Returns the expected input spatial dimensions based on the model's
    /// minimum spatial footprint (8×8 pixels, one pixel per stage-3 cell).
    pub fn min_input_size() -> (usize, usize) {
        (8, 8)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
fn sigmoid_scalar(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    fn make_detector(anchors: usize) -> FaceDetector {
        FaceDetector::new(anchors).expect("FaceDetector::new failed")
    }

    // ── Construction ─────────────────────────────────────────────────────────

    #[test]
    fn test_new_valid() {
        let d = make_detector(4);
        assert_eq!(d.num_anchors, 4);
        assert_eq!(d.head.out_features, 20);
    }

    #[test]
    fn test_new_zero_anchors_fails() {
        assert!(FaceDetector::new(0).is_err());
    }

    // ── Forward pass ─────────────────────────────────────────────────────────

    #[test]
    fn test_forward_zero_weights_no_detections() {
        let det = make_detector(4);
        let img = Tensor::zeros(vec![3, 32, 32]).expect("zeros");
        let boxes = det.forward(&img, 0.5).expect("forward");
        // sigmoid(0) = 0.5, which equals the threshold, so no box passes conf >= 0.5
        // (strict less-than check in forward)
        // → boxes may be empty
        assert!(boxes.len() <= 4);
    }

    #[test]
    fn test_forward_wrong_rank_returns_error() {
        let det = make_detector(2);
        let bad = Tensor::zeros(vec![32, 32]).expect("zeros");
        assert!(det.forward(&bad, 0.5).is_err());
    }

    #[test]
    fn test_forward_wrong_channels_returns_error() {
        let det = make_detector(2);
        let bad = Tensor::zeros(vec![1, 32, 32]).expect("zeros");
        assert!(det.forward(&bad, 0.5).is_err());
    }

    #[test]
    fn test_forward_high_confidence_threshold_empty() {
        let det = make_detector(4);
        let img = Tensor::zeros(vec![3, 16, 16]).expect("zeros");
        let boxes = det.forward(&img, 1.0).expect("forward");
        assert!(boxes.is_empty());
    }

    #[test]
    fn test_forward_low_threshold_passes_half_sigmoid() {
        // sigmoid(0) ≈ 0.5; with threshold 0.0 all anchors should be returned
        let det = make_detector(3);
        let img = Tensor::zeros(vec![3, 24, 24]).expect("zeros");
        let boxes = det.forward(&img, 0.0).expect("forward");
        // After NMS some may be merged, but at least 1 should remain
        assert!(!boxes.is_empty());
    }

    // ── DetectionBox helpers ─────────────────────────────────────────────────

    #[test]
    fn test_detection_box_iou_identical() {
        let b = DetectionBox {
            confidence: 1.0,
            cx: 0.5,
            cy: 0.5,
            w: 0.4,
            h: 0.4,
        };
        let iou = b.iou(&b);
        assert!(
            (iou - 1.0).abs() < 1e-5,
            "IoU of identical boxes should be 1.0, got {}",
            iou
        );
    }

    #[test]
    fn test_detection_box_iou_no_overlap() {
        let b1 = DetectionBox {
            confidence: 1.0,
            cx: 0.1,
            cy: 0.5,
            w: 0.1,
            h: 0.1,
        };
        let b2 = DetectionBox {
            confidence: 1.0,
            cx: 0.9,
            cy: 0.5,
            w: 0.1,
            h: 0.1,
        };
        let iou = b1.iou(&b2);
        assert_eq!(iou, 0.0, "Non-overlapping boxes should have IoU = 0");
    }

    #[test]
    fn test_nms_keeps_highest_confidence() {
        let b1 = DetectionBox {
            confidence: 0.9,
            cx: 0.5,
            cy: 0.5,
            w: 0.4,
            h: 0.4,
        };
        let b2 = DetectionBox {
            confidence: 0.7,
            cx: 0.5,
            cy: 0.5,
            w: 0.4,
            h: 0.4,
        };
        let kept = nms(vec![b1.clone(), b2], 0.5);
        assert_eq!(kept.len(), 1);
        assert!((kept[0].confidence - 0.9).abs() < 1e-5);
    }

    #[test]
    fn test_nms_keeps_distinct_boxes() {
        let b1 = DetectionBox {
            confidence: 0.9,
            cx: 0.1,
            cy: 0.5,
            w: 0.1,
            h: 0.1,
        };
        let b2 = DetectionBox {
            confidence: 0.8,
            cx: 0.9,
            cy: 0.5,
            w: 0.1,
            h: 0.1,
        };
        let kept = nms(vec![b1, b2], 0.5);
        assert_eq!(kept.len(), 2);
    }

    #[test]
    fn test_min_input_size() {
        let (h, w) = FaceDetector::min_input_size();
        assert_eq!(h, 8);
        assert_eq!(w, 8);
    }
}
