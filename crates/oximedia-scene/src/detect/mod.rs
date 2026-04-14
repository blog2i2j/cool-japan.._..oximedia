//! Object, face, logo, and text detection.
//!
//! This module provides various detection algorithms using patent-free methods:
//!
//! - **Object detection**: HOG-based detection for common objects
//! - **Face detection**: Haar cascade-based face detection (multi-scale)
//! - **Logo detection**: Template matching and feature-based detection
//! - **Text detection**: Text region detection using connected components
//!
//! All detectors support Non-Maximum Suppression (NMS) to suppress overlapping
//! detections. The shared [`nms()`] function can be applied to any detection type
//! that provides a bounding box and confidence score.

pub mod face;
pub mod logo;
pub mod nms;
pub mod object;
pub mod pyramid;
pub mod text;

pub use face::{FaceDetection, FaceDetector};
pub use logo::{LogoDetection, LogoDetector};
pub use nms::{non_maximum_suppression, soft_nms_boxes, Detection, DetectionBox};
pub use object::{ObjectDetection, ObjectDetector, ObjectType};
pub use text::{TextDetection, TextDetector};

use crate::common::Rect;

/// Apply Non-Maximum Suppression (NMS) to a list of detections.
///
/// Detections are sorted by descending confidence (provided by `conf_fn`),
/// and any lower-confidence detection that overlaps the current best detection
/// by more than `iou_threshold` (IoU) is suppressed.
///
/// # Arguments
///
/// * `detections` – mutable list of detections, modified in place.
/// * `bbox_fn` – closure extracting the bounding box from a detection.
/// * `conf_fn` – closure extracting the confidence value from a detection.
/// * `iou_threshold` – IoU threshold above which a detection is suppressed.
pub fn nms<T, B, C>(detections: &mut Vec<T>, bbox_fn: B, conf_fn: C, iou_threshold: f32)
where
    T: Clone,
    B: Fn(&T) -> Rect,
    C: Fn(&T) -> f32,
{
    // Sort descending by confidence
    detections.sort_by(|a, b| {
        conf_fn(b)
            .partial_cmp(&conf_fn(a))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let n = detections.len();
    let mut suppressed = vec![false; n];

    for i in 0..n {
        if suppressed[i] {
            continue;
        }
        let bbox_i = bbox_fn(&detections[i]);
        for j in (i + 1)..n {
            if suppressed[j] {
                continue;
            }
            let bbox_j = bbox_fn(&detections[j]);
            if bbox_i.iou(&bbox_j) > iou_threshold {
                suppressed[j] = true;
            }
        }
    }

    let mut out = Vec::with_capacity(n);
    for (i, det) in detections.drain(..).enumerate() {
        if !suppressed[i] {
            out.push(det);
        }
    }
    *detections = out;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct Det {
        bbox: Rect,
        conf: f32,
    }

    #[test]
    fn test_nms_removes_overlapping() {
        let mut dets = vec![
            Det {
                bbox: Rect::new(0.0, 0.0, 100.0, 100.0),
                conf: 0.9,
            },
            Det {
                bbox: Rect::new(5.0, 5.0, 100.0, 100.0),
                conf: 0.7,
            },
            Det {
                bbox: Rect::new(200.0, 200.0, 50.0, 50.0),
                conf: 0.8,
            },
        ];
        nms(&mut dets, |d| d.bbox, |d| d.conf, 0.5);
        assert_eq!(dets.len(), 2, "should keep two non-overlapping detections");
        // Highest confidence should survive
        assert!((dets[0].conf - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn test_nms_empty() {
        let mut dets: Vec<Det> = Vec::new();
        nms(&mut dets, |d| d.bbox, |d| d.conf, 0.5);
        assert!(dets.is_empty());
    }

    #[test]
    fn test_nms_no_overlap() {
        let mut dets = vec![
            Det {
                bbox: Rect::new(0.0, 0.0, 10.0, 10.0),
                conf: 0.9,
            },
            Det {
                bbox: Rect::new(100.0, 100.0, 10.0, 10.0),
                conf: 0.7,
            },
        ];
        nms(&mut dets, |d| d.bbox, |d| d.conf, 0.5);
        assert_eq!(dets.len(), 2, "non-overlapping boxes should both survive");
    }
}
