//! Utility functions for YOLO object detection.
//!
//! This module provides helper functions for:
//! - Letterbox resizing with padding
//! - YOLO output decoding (YOLOv5 and YOLOv8)
//! - Non-Maximum Suppression (NMS)
//! - Visualization (drawing bounding boxes)

#![allow(clippy::too_many_arguments)]
#![allow(clippy::similar_names)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::many_single_char_names)]

use crate::detect::object::{iou, BoundingBox, Detection};
use crate::error::{CvError, CvResult};
use ndarray::ArrayD;

/// Letterbox resize parameters.
///
/// Stores information about padding and scaling applied during letterbox resize.
#[derive(Debug, Clone, Copy)]
pub struct LetterboxParams {
    /// Scale factor applied to the image.
    pub scale: f32,
    /// Padding on the left side.
    pub pad_left: u32,
    /// Padding on the top side.
    pub pad_top: u32,
    /// New width after padding.
    pub new_width: u32,
    /// New height after padding.
    pub new_height: u32,
}

/// Resize image with letterbox (maintain aspect ratio and pad).
///
/// This function resizes the image to fit within the target size while
/// maintaining the aspect ratio, then pads the remaining space.
///
/// # Arguments
///
/// * `image` - RGB image data (row-major, channels last)
/// * `width` - Original image width
/// * `height` - Original image height
/// * `target_width` - Target width
/// * `target_height` - Target height
///
/// # Returns
///
/// Tuple of (resized and padded image, letterbox parameters).
///
/// # Errors
///
/// Returns an error if dimensions are invalid.
pub fn letterbox_resize(
    image: &[u8],
    width: u32,
    height: u32,
    target_width: u32,
    target_height: u32,
) -> CvResult<(Vec<u8>, LetterboxParams)> {
    if width == 0 || height == 0 || target_width == 0 || target_height == 0 {
        return Err(CvError::invalid_dimensions(width, height));
    }

    // Calculate scale to fit image within target size
    let scale = (target_width as f32 / width as f32).min(target_height as f32 / height as f32);

    let new_width = (width as f32 * scale) as u32;
    let new_height = (height as f32 * scale) as u32;

    // Calculate padding
    let pad_left = (target_width - new_width) / 2;
    let pad_top = (target_height - new_height) / 2;

    // Resize image using bilinear interpolation
    let resized = resize_bilinear(image, width, height, new_width, new_height)?;

    // Create padded image (filled with gray = 114)
    let mut padded = vec![114u8; (target_width * target_height * 3) as usize];

    // Copy resized image to center
    for y in 0..new_height {
        for x in 0..new_width {
            let src_idx = ((y * new_width + x) * 3) as usize;
            let dst_x = x + pad_left;
            let dst_y = y + pad_top;
            let dst_idx = ((dst_y * target_width + dst_x) * 3) as usize;

            padded[dst_idx] = resized[src_idx];
            padded[dst_idx + 1] = resized[src_idx + 1];
            padded[dst_idx + 2] = resized[src_idx + 2];
        }
    }

    let params = LetterboxParams {
        scale,
        pad_left,
        pad_top,
        new_width: target_width,
        new_height: target_height,
    };

    Ok((padded, params))
}

/// Resize image using bilinear interpolation.
fn resize_bilinear(
    image: &[u8],
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
) -> CvResult<Vec<u8>> {
    let mut result = vec![0u8; (dst_width * dst_height * 3) as usize];

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

            let dx = src_x - x0 as f32;
            let dy = src_y - y0 as f32;

            for c in 0..3 {
                let idx00 = ((y0 * src_width + x0) * 3 + c) as usize;
                let idx01 = ((y0 * src_width + x1) * 3 + c) as usize;
                let idx10 = ((y1 * src_width + x0) * 3 + c) as usize;
                let idx11 = ((y1 * src_width + x1) * 3 + c) as usize;

                let v00 = image[idx00] as f32;
                let v01 = image[idx01] as f32;
                let v10 = image[idx10] as f32;
                let v11 = image[idx11] as f32;

                let v0 = v00 * (1.0 - dx) + v01 * dx;
                let v1 = v10 * (1.0 - dx) + v11 * dx;
                let v = v0 * (1.0 - dy) + v1 * dy;

                let dst_idx = ((y * dst_width + x) * 3 + c) as usize;
                result[dst_idx] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
    }

    Ok(result)
}

/// Decode YOLOv5 output.
///
/// YOLOv5 output format: [batch, num_predictions, 5 + num_classes]
/// Each prediction: [x_center, y_center, width, height, objectness, class_scores...]
///
/// # Arguments
///
/// * `output` - Model output tensor
/// * `conf_threshold` - Confidence threshold
/// * `iou_threshold` - IoU threshold for NMS
/// * `num_classes` - Number of classes
/// * `per_class_nms` - Whether to apply NMS per class
/// * `max_detections` - Maximum number of detections to return
///
/// # Returns
///
/// Vector of detections.
pub fn decode_yolov5_output(
    output: &ArrayD<f32>,
    conf_threshold: f32,
    iou_threshold: f32,
    num_classes: usize,
    per_class_nms: bool,
    max_detections: usize,
) -> CvResult<Vec<Detection>> {
    let shape = output.shape();
    if shape.len() < 3 {
        return Err(CvError::detection_failed(
            "Invalid output shape: expected 3 dimensions [batch, predictions, values]",
        ));
    }

    let num_predictions = shape[1];
    let num_values = shape[2];

    if num_values < 5 + num_classes {
        return Err(CvError::detection_failed(format!(
            "Invalid output format: expected at least {} values, got {}",
            5 + num_classes,
            num_values
        )));
    }

    let mut detections = Vec::new();

    for i in 0..num_predictions {
        // Get prediction values directly via indexing
        let cx = output[[0, i, 0]];
        let cy = output[[0, i, 1]];
        let w = output[[0, i, 2]];
        let h = output[[0, i, 3]];
        let objectness = output[[0, i, 4]];

        // Get class scores
        let mut max_class_score = 0.0_f32;
        let mut class_id = 0;
        for c in 0..num_classes {
            let score = output[[0, i, 5 + c]];
            if score > max_class_score {
                max_class_score = score;
                class_id = c;
            }
        }

        // Combined confidence (objectness * class_score)
        let confidence = objectness * max_class_score;

        if confidence >= conf_threshold {
            let bbox = BoundingBox::from_center(cx, cy, w, h);
            detections.push(Detection::new(bbox, class_id as u32, confidence));
        }
    }

    // Apply NMS
    if per_class_nms {
        detections = apply_per_class_nms(&detections, iou_threshold);
    } else {
        detections = apply_nms(&detections, iou_threshold);
    }

    // Limit to max detections
    detections.truncate(max_detections);

    Ok(detections)
}

/// Decode YOLOv8 output.
///
/// YOLOv8 output format: [batch, 4 + num_classes, num_predictions]
/// First 4 channels: [x_center, y_center, width, height]
/// Remaining channels: class scores (no objectness score)
///
/// # Arguments
///
/// * `output` - Model output tensor
/// * `conf_threshold` - Confidence threshold
/// * `iou_threshold` - IoU threshold for NMS
/// * `num_classes` - Number of classes
/// * `per_class_nms` - Whether to apply NMS per class
/// * `max_detections` - Maximum number of detections to return
///
/// # Returns
///
/// Vector of detections.
pub fn decode_yolov8_output(
    output: &ArrayD<f32>,
    conf_threshold: f32,
    iou_threshold: f32,
    num_classes: usize,
    per_class_nms: bool,
    max_detections: usize,
) -> CvResult<Vec<Detection>> {
    let shape = output.shape();
    if shape.len() < 3 {
        return Err(CvError::detection_failed("Invalid output shape"));
    }

    // YOLOv8 format: [batch, 4 + num_classes, num_predictions]
    let num_channels = shape[1];
    let num_predictions = shape[2];

    if num_channels < 4 + num_classes {
        return Err(CvError::detection_failed(format!(
            "Invalid output format: expected {} channels, got {}",
            4 + num_classes,
            num_channels
        )));
    }

    let mut detections = Vec::new();

    for i in 0..num_predictions {
        // Get bounding box coordinates
        let cx = output[[0, 0, i]];
        let cy = output[[0, 1, i]];
        let w = output[[0, 2, i]];
        let h = output[[0, 3, i]];

        // Get class scores (no objectness in YOLOv8)
        let mut max_class_score = 0.0_f32;
        let mut class_id = 0;
        for c in 0..num_classes {
            let score = output[[0, 4 + c, i]];
            if score > max_class_score {
                max_class_score = score;
                class_id = c;
            }
        }

        let confidence = max_class_score;

        if confidence >= conf_threshold {
            let bbox = BoundingBox::from_center(cx, cy, w, h);
            detections.push(Detection::new(bbox, class_id as u32, confidence));
        }
    }

    // Apply NMS
    if per_class_nms {
        detections = apply_per_class_nms(&detections, iou_threshold);
    } else {
        detections = apply_nms(&detections, iou_threshold);
    }

    // Limit to max detections
    detections.truncate(max_detections);

    Ok(detections)
}

/// Apply Non-Maximum Suppression (class-agnostic).
fn apply_nms(detections: &[Detection], iou_threshold: f32) -> Vec<Detection> {
    if detections.is_empty() {
        return Vec::new();
    }

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

        for j in (i + 1)..sorted.len() {
            let (other_idx, other) = sorted[j];
            if !keep[other_idx] {
                continue;
            }

            if iou(&detection.bbox, &other.bbox) > iou_threshold {
                keep[other_idx] = false;
            }
        }
    }

    result
}

/// Apply Non-Maximum Suppression per class.
fn apply_per_class_nms(detections: &[Detection], iou_threshold: f32) -> Vec<Detection> {
    if detections.is_empty() {
        return Vec::new();
    }

    // Group by class
    let mut by_class: Vec<Vec<Detection>> = Vec::new();
    let max_class = detections.iter().map(|d| d.class_id).max().unwrap_or(0) as usize;

    for _ in 0..=max_class {
        by_class.push(Vec::new());
    }

    for det in detections {
        if det.class_id <= max_class as u32 {
            by_class[det.class_id as usize].push(det.clone());
        }
    }

    // Apply NMS per class
    let mut result = Vec::new();
    for class_dets in by_class {
        result.extend(apply_nms(&class_dets, iou_threshold));
    }

    // Sort by confidence
    result.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    result
}

/// Draw bounding boxes on image.
///
/// # Arguments
///
/// * `image` - RGB image data
/// * `width` - Image width
/// * `height` - Image height
/// * `detections` - Detections to draw
///
/// # Returns
///
/// Annotated image with bounding boxes.
///
/// # Errors
///
/// Returns an error if image dimensions are invalid.
pub fn draw_detections(
    image: &[u8],
    width: u32,
    height: u32,
    detections: &[Detection],
) -> CvResult<Vec<u8>> {
    if width == 0 || height == 0 {
        return Err(CvError::invalid_dimensions(width, height));
    }

    let expected_size = (width * height * 3) as usize;
    if image.len() != expected_size {
        return Err(CvError::insufficient_data(expected_size, image.len()));
    }

    let mut result = image.to_vec();

    for det in detections {
        let bbox = &det.bbox;

        // Get box coordinates
        let x1 = bbox.x.max(0.0).min(width as f32 - 1.0) as u32;
        let y1 = bbox.y.max(0.0).min(height as f32 - 1.0) as u32;
        let x2 = bbox.right().max(0.0).min(width as f32 - 1.0) as u32;
        let y2 = bbox.bottom().max(0.0).min(height as f32 - 1.0) as u32;

        // Choose color based on class ID
        let color = get_color(det.class_id);

        // Draw rectangle (2 pixels thick)
        for thickness in 0..2 {
            // Top and bottom
            for x in x1..=x2 {
                if y1 + thickness < height {
                    draw_pixel(&mut result, x, y1 + thickness, width, color);
                }
                if y2 >= thickness {
                    draw_pixel(&mut result, x, y2 - thickness, width, color);
                }
            }

            // Left and right
            for y in y1..=y2 {
                if x1 + thickness < width {
                    draw_pixel(&mut result, x1 + thickness, y, width, color);
                }
                if x2 >= thickness {
                    draw_pixel(&mut result, x2 - thickness, y, width, color);
                }
            }
        }
    }

    Ok(result)
}

/// Draw a pixel with the given color.
fn draw_pixel(image: &mut [u8], x: u32, y: u32, width: u32, color: [u8; 3]) {
    let idx = ((y * width + x) * 3) as usize;
    if idx + 2 < image.len() {
        image[idx] = color[0];
        image[idx + 1] = color[1];
        image[idx + 2] = color[2];
    }
}

/// Get a distinct color for a class ID.
fn get_color(class_id: u32) -> [u8; 3] {
    // Generate colors using golden ratio
    let hue = (class_id as f32 * 0.618_034) % 1.0;
    hsv_to_rgb(hue, 0.8, 0.95)
}

/// Convert HSV to RGB.
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [u8; 3] {
    let c = v * s;
    let x = c * (1.0 - ((h * 6.0) % 2.0 - 1.0).abs());
    let m = v - c;

    let (r, g, b) = match (h * 6.0) as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };

    [
        ((r + m) * 255.0) as u8,
        ((g + m) * 255.0) as u8,
        ((b + m) * 255.0) as u8,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_letterbox_params() {
        let params = LetterboxParams {
            scale: 0.5,
            pad_left: 10,
            pad_top: 20,
            new_width: 640,
            new_height: 640,
        };

        assert_eq!(params.scale, 0.5);
        assert_eq!(params.pad_left, 10);
        assert_eq!(params.pad_top, 20);
    }

    #[test]
    fn test_letterbox_resize() {
        let image = vec![128u8; 800 * 600 * 3];
        let result = letterbox_resize(&image, 800, 600, 640, 640);
        assert!(result.is_ok());

        let (resized, params) = result.expect("operation should succeed");
        assert_eq!(resized.len(), 640 * 640 * 3);
        assert!(params.scale > 0.0);
    }

    #[test]
    fn test_resize_bilinear() {
        let image = vec![128u8; 100 * 100 * 3];
        let result = resize_bilinear(&image, 100, 100, 50, 50);
        assert!(result.is_ok());

        let resized = result.expect("resized should be valid");
        assert_eq!(resized.len(), 50 * 50 * 3);
    }

    #[test]
    fn test_apply_nms() {
        let detections = vec![
            Detection::new(BoundingBox::new(0.0, 0.0, 100.0, 100.0), 0, 0.9),
            Detection::new(BoundingBox::new(10.0, 10.0, 100.0, 100.0), 0, 0.8),
            Detection::new(BoundingBox::new(200.0, 200.0, 100.0, 100.0), 0, 0.85),
        ];

        let filtered = apply_nms(&detections, 0.5);
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_apply_per_class_nms() {
        let detections = vec![
            Detection::new(BoundingBox::new(0.0, 0.0, 100.0, 100.0), 0, 0.9),
            Detection::new(BoundingBox::new(0.0, 0.0, 100.0, 100.0), 1, 0.8),
        ];

        let filtered = apply_per_class_nms(&detections, 0.5);
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_get_color() {
        let color1 = get_color(0);
        let color2 = get_color(1);

        assert_ne!(color1, color2);
    }

    #[test]
    fn test_hsv_to_rgb() {
        let color = hsv_to_rgb(0.0, 1.0, 1.0); // Red
        assert_eq!(color[0], 255);

        let color = hsv_to_rgb(0.333, 1.0, 1.0); // Green (approx)
        assert!(color[1] > 200);
    }

    #[test]
    fn test_draw_detections() {
        let image = vec![0u8; 640 * 640 * 3];
        let detections = vec![Detection::new(
            BoundingBox::new(100.0, 100.0, 200.0, 200.0),
            0,
            0.9,
        )];

        let result = draw_detections(&image, 640, 640, &detections);
        assert!(result.is_ok());

        let annotated = result.expect("annotated should be valid");
        assert_eq!(annotated.len(), image.len());
    }

    #[test]
    fn test_draw_detections_invalid_dimensions() {
        let image = vec![0u8; 100];
        let detections = vec![];

        let result = draw_detections(&image, 0, 0, &detections);
        assert!(result.is_err());
    }

    #[test]
    fn test_decode_yolov8_invalid_shape() {
        let output = ArrayD::<f32>::zeros(vec![1, 2]); // Invalid shape
        let result = decode_yolov8_output(&output, 0.5, 0.45, 80, false, 300);
        assert!(result.is_err());
    }

    #[test]
    fn test_decode_yolov5_invalid_shape() {
        let output = ArrayD::<f32>::zeros(vec![1, 2]); // Invalid shape (needs 3 dims)
        let result = decode_yolov5_output(&output, 0.5, 0.45, 80, false, 300);
        assert!(result.is_err());
    }

    #[test]
    fn test_decode_yolov5_insufficient_channels() {
        // Shape [1, 100, 10] with num_classes=80 requires at least 85 channels
        let output = ArrayD::<f32>::zeros(vec![1, 100, 10]);
        let result = decode_yolov5_output(&output, 0.5, 0.45, 80, false, 300);
        assert!(result.is_err());
    }

    #[test]
    fn test_decode_yolov8_insufficient_channels() {
        // Shape [1, 10, 100] with num_classes=80 requires at least 84 channels
        let output = ArrayD::<f32>::zeros(vec![1, 10, 100]);
        let result = decode_yolov8_output(&output, 0.5, 0.45, 80, false, 300);
        assert!(result.is_err());
    }

    #[test]
    fn test_decode_yolov5_no_detections() {
        // All-zero output should produce no detections above threshold
        let output = ArrayD::<f32>::zeros(vec![1, 100, 85]);
        let result = decode_yolov5_output(&output, 0.25, 0.45, 80, false, 300);
        assert!(result.is_ok());
        let detections = result.expect("detections should be valid");
        assert!(detections.is_empty());
    }

    #[test]
    fn test_decode_yolov8_no_detections() {
        // All-zero output should produce no detections above threshold
        let output = ArrayD::<f32>::zeros(vec![1, 84, 100]);
        let result = decode_yolov8_output(&output, 0.25, 0.45, 80, false, 300);
        assert!(result.is_ok());
        let detections = result.expect("detections should be valid");
        assert!(detections.is_empty());
    }

    #[test]
    fn test_decode_yolov5_with_detection() {
        // Create a tensor with one strong detection
        let num_predictions = 10;
        let num_classes = 2;
        let num_values = 5 + num_classes; // 7
        let mut data = vec![0.0f32; 1 * num_predictions * num_values];

        // Set prediction 0 to a high-confidence detection of class 0
        // cx=100, cy=100, w=50, h=50, objectness=0.9, class0_score=0.95, class1_score=0.1
        let base = 0;
        data[base] = 100.0; // cx
        data[base + 1] = 100.0; // cy
        data[base + 2] = 50.0; // w
        data[base + 3] = 50.0; // h
        data[base + 4] = 0.9; // objectness
        data[base + 5] = 0.95; // class 0 score
        data[base + 6] = 0.1; // class 1 score

        let output = ArrayD::from_shape_vec(vec![1, num_predictions, num_values], data)
            .expect("from_shape_vec should succeed");
        let result = decode_yolov5_output(&output, 0.5, 0.45, num_classes, false, 300);
        assert!(result.is_ok());
        let detections = result.expect("detections should be valid");
        assert!(!detections.is_empty(), "Should find at least one detection");
        assert_eq!(detections[0].class_id, 0);
    }

    #[test]
    fn test_apply_nms_max_detections() {
        // Create many non-overlapping detections
        let detections: Vec<_> = (0..20)
            .map(|i| {
                Detection::new(
                    BoundingBox::new(i as f32 * 200.0, 0.0, 100.0, 100.0),
                    0,
                    0.9 - i as f32 * 0.01,
                )
            })
            .collect();

        let filtered = apply_nms(&detections, 0.5);
        // All should survive NMS since they don't overlap
        assert_eq!(filtered.len(), 20);
    }

    #[test]
    fn test_nms_suppresses_overlapping() {
        // Two heavily overlapping boxes - NMS should keep only the higher confidence one
        let detections = vec![
            Detection::new(BoundingBox::new(0.0, 0.0, 100.0, 100.0), 0, 0.95),
            Detection::new(BoundingBox::new(5.0, 5.0, 100.0, 100.0), 0, 0.85),
        ];
        let filtered = apply_nms(&detections, 0.45);
        assert_eq!(filtered.len(), 1);
        assert!((filtered[0].confidence - 0.95).abs() < 1e-6);
    }

    #[test]
    fn test_letterbox_preserves_aspect_ratio() {
        // A 1600x900 image letterboxed to 640x640
        let image = vec![0u8; 1600 * 900 * 3];
        let (_, params) =
            letterbox_resize(&image, 1600, 900, 640, 640).expect("letterbox_resize should succeed");
        // Width should be the constraining dimension
        // scale = min(640/1600, 640/900) = min(0.4, 0.711) = 0.4
        assert!((params.scale - 0.4).abs() < 0.001);
        // Should have top/bottom padding
        assert!(params.pad_top > 0);
        // No left/right padding (limited by width)
        assert_eq!(params.pad_left, 0);
    }

    #[test]
    fn test_letterbox_resize_square_input() {
        // Square input to square target: scale = 1.0 (well, 640/640=1.0)
        let image = vec![100u8; 640 * 640 * 3];
        let (resized, params) =
            letterbox_resize(&image, 640, 640, 640, 640).expect("letterbox_resize should succeed");
        assert_eq!(resized.len(), 640 * 640 * 3);
        assert!((params.scale - 1.0).abs() < 0.001);
        assert_eq!(params.pad_left, 0);
        assert_eq!(params.pad_top, 0);
    }

    #[test]
    fn test_draw_detections_with_multiple_classes() {
        let image = vec![0u8; 320 * 240 * 3];
        let detections = vec![
            Detection::new(BoundingBox::new(10.0, 10.0, 50.0, 50.0), 0, 0.9),
            Detection::new(BoundingBox::new(100.0, 100.0, 60.0, 60.0), 1, 0.8),
            Detection::new(BoundingBox::new(200.0, 100.0, 40.0, 40.0), 5, 0.7),
        ];
        let result = draw_detections(&image, 320, 240, &detections);
        assert!(result.is_ok());
        let annotated = result.expect("annotated should be valid");
        assert_eq!(annotated.len(), 320 * 240 * 3);
    }
}
