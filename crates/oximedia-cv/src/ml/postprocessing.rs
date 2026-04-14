//! Post-processing utilities for ML model outputs.
//!
//! This module provides common post-processing operations including:
//! - Non-Maximum Suppression (NMS)
//! - Activation functions (softmax, sigmoid, tanh)
//! - Confidence thresholding
//! - Bounding box decoding

use crate::detect::{BoundingBox, Detection};
use crate::error::{CvError, CvResult};
use crate::ml::tensor::Tensor;

/// Apply softmax activation to a tensor.
///
/// Converts logits to probabilities that sum to 1.
///
/// # Arguments
///
/// * `tensor` - Input tensor with logits
/// * `axis` - Axis along which to apply softmax
///
/// # Errors
///
/// Returns an error if tensor operations fail.
///
/// # Example
///
/// ```
/// use oximedia_cv::ml::{Tensor, postprocessing::softmax};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut tensor = Tensor::zeros(&[1, 1000]);
/// softmax(&mut tensor, 1)?;
/// # Ok(())
/// # }
/// ```
pub fn softmax(tensor: &mut Tensor, axis: usize) -> CvResult<()> {
    let mut data = tensor.data().to_f32()?;
    let shape = data.shape().to_vec();

    if axis >= shape.len() {
        return Err(CvError::invalid_parameter(
            "axis",
            format!("{axis} >= {}", shape.len()),
        ));
    }

    // For simplicity, we'll handle 2D case (batch, classes)
    if shape.len() == 2 && axis == 1 {
        let batch_size = shape[0];
        let num_classes = shape[1];

        for b in 0..batch_size {
            // Find max for numerical stability
            let mut max_val = data[[b, 0]];
            for c in 1..num_classes {
                max_val = max_val.max(data[[b, c]]);
            }

            // Compute exp(x - max)
            let mut sum = 0.0;
            for c in 0..num_classes {
                let exp_val = (data[[b, c]] - max_val).exp();
                data[[b, c]] = exp_val;
                sum += exp_val;
            }

            // Normalize
            for c in 0..num_classes {
                data[[b, c]] /= sum;
            }
        }
    } else {
        return Err(CvError::tensor_error(
            "Softmax currently only supports 2D tensors with axis=1",
        ));
    }

    *tensor = Tensor::new_f32(data, tensor.layout());
    Ok(())
}

/// Apply sigmoid activation to a tensor.
///
/// Maps values to (0, 1) range using sigmoid function: 1 / (1 + exp(-x))
///
/// # Arguments
///
/// * `tensor` - Input tensor
///
/// # Errors
///
/// Returns an error if tensor operations fail.
///
/// # Example
///
/// ```
/// use oximedia_cv::ml::{Tensor, postprocessing::sigmoid};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut tensor = Tensor::zeros(&[1, 100]);
/// sigmoid(&mut tensor)?;
/// # Ok(())
/// # }
/// ```
pub fn sigmoid(tensor: &mut Tensor) -> CvResult<()> {
    let mut data = tensor.data().to_f32()?;
    data.mapv_inplace(|x| 1.0 / (1.0 + (-x).exp()));
    *tensor = Tensor::new_f32(data, tensor.layout());
    Ok(())
}

/// Apply tanh activation to a tensor.
///
/// Maps values to (-1, 1) range using hyperbolic tangent.
///
/// # Arguments
///
/// * `tensor` - Input tensor
///
/// # Errors
///
/// Returns an error if tensor operations fail.
pub fn tanh(tensor: &mut Tensor) -> CvResult<()> {
    let mut data = tensor.data().to_f32()?;
    data.mapv_inplace(f32::tanh);
    *tensor = Tensor::new_f32(data, tensor.layout());
    Ok(())
}

/// Apply ReLU activation to a tensor.
///
/// Sets negative values to zero: max(0, x)
///
/// # Arguments
///
/// * `tensor` - Input tensor
///
/// # Errors
///
/// Returns an error if tensor operations fail.
pub fn relu(tensor: &mut Tensor) -> CvResult<()> {
    let mut data = tensor.data().to_f32()?;
    data.mapv_inplace(|x| x.max(0.0));
    *tensor = Tensor::new_f32(data, tensor.layout());
    Ok(())
}

/// Filter detections by confidence threshold.
///
/// # Arguments
///
/// * `detections` - Input detections
/// * `threshold` - Minimum confidence (0.0 - 1.0)
///
/// # Returns
///
/// Detections with confidence >= threshold.
///
/// # Example
///
/// ```
/// use oximedia_cv::detect::{BoundingBox, Detection};
/// use oximedia_cv::ml::postprocessing::confidence_threshold;
///
/// let detections = vec![
///     Detection::new(BoundingBox::new(0.0, 0.0, 10.0, 10.0), 0, 0.9),
///     Detection::new(BoundingBox::new(0.0, 0.0, 10.0, 10.0), 0, 0.3),
/// ];
/// let filtered = confidence_threshold(&detections, 0.5);
/// assert_eq!(filtered.len(), 1);
/// ```
#[must_use]
pub fn confidence_threshold(detections: &[Detection], threshold: f32) -> Vec<Detection> {
    detections
        .iter()
        .filter(|d| d.confidence >= threshold)
        .cloned()
        .collect()
}

/// Perform Non-Maximum Suppression on detections.
///
/// This is a re-export of the NMS function from the detect module
/// for convenience in ML pipelines.
///
/// # Arguments
///
/// * `detections` - Input detections
/// * `iou_threshold` - IoU threshold for suppression (typically 0.45-0.5)
///
/// # Returns
///
/// Filtered detections after NMS.
///
/// # Example
///
/// ```
/// use oximedia_cv::detect::{BoundingBox, Detection};
/// use oximedia_cv::ml::postprocessing::nms;
///
/// let detections = vec![
///     Detection::new(BoundingBox::new(0.0, 0.0, 100.0, 100.0), 0, 0.9),
///     Detection::new(BoundingBox::new(10.0, 10.0, 100.0, 100.0), 0, 0.8),
/// ];
/// let filtered = nms(&detections, 0.5);
/// ```
#[must_use]
pub fn nms(detections: &[Detection], iou_threshold: f32) -> Vec<Detection> {
    crate::detect::object::nms(detections, iou_threshold)
}

/// Perform soft NMS on detections.
///
/// This is a re-export of the soft NMS function from the detect module
/// for convenience in ML pipelines.
///
/// # Arguments
///
/// * `detections` - Input detections
/// * `iou_threshold` - IoU threshold for soft suppression
/// * `sigma` - Gaussian sigma for score decay
/// * `score_threshold` - Minimum score to keep a detection
///
/// # Returns
///
/// Detections with adjusted confidence scores.
///
/// # Example
///
/// ```
/// use oximedia_cv::detect::{BoundingBox, Detection};
/// use oximedia_cv::ml::postprocessing::soft_nms;
///
/// let detections = vec![
///     Detection::new(BoundingBox::new(0.0, 0.0, 100.0, 100.0), 0, 0.9),
///     Detection::new(BoundingBox::new(10.0, 10.0, 100.0, 100.0), 0, 0.8),
/// ];
/// let filtered = soft_nms(&detections, 0.3, 0.5, 0.1);
/// ```
#[must_use]
pub fn soft_nms(
    detections: &[Detection],
    iou_threshold: f32,
    sigma: f32,
    score_threshold: f32,
) -> Vec<Detection> {
    crate::detect::object::soft_nms(detections, iou_threshold, sigma, score_threshold)
}

/// Decode YOLO-style bounding boxes.
///
/// Converts YOLO format (cx, cy, w, h) to corner format (x, y, width, height).
///
/// # Arguments
///
/// * `predictions` - Tensor with YOLO predictions [batch, num_boxes, 5+num_classes]
///   Format: [cx, cy, w, h, objectness, class_probs...]
/// * `img_width` - Original image width
/// * `img_height` - Original image height
/// * `confidence_threshold` - Minimum confidence threshold
///
/// # Errors
///
/// Returns an error if tensor format is invalid.
///
/// # Example
///
/// ```
/// use oximedia_cv::ml::{Tensor, postprocessing::decode_yolo_boxes};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let predictions = Tensor::zeros(&[1, 100, 85]); // 80 classes + 5
/// let detections = decode_yolo_boxes(&predictions, 640, 480, 0.5)?;
/// # Ok(())
/// # }
/// ```
pub fn decode_yolo_boxes(
    predictions: &Tensor,
    img_width: u32,
    img_height: u32,
    confidence_threshold: f32,
) -> CvResult<Vec<Detection>> {
    let data = predictions.data().to_f32()?;
    let shape = data.shape();

    if shape.len() != 3 {
        return Err(CvError::tensor_error(
            "YOLO predictions must be 3D: [batch, num_boxes, box_data]",
        ));
    }

    let batch_size = shape[0];
    let num_boxes = shape[1];
    let box_data_size = shape[2];

    if box_data_size < 5 {
        return Err(CvError::tensor_error(
            "YOLO box data must have at least 5 elements (cx, cy, w, h, objectness)",
        ));
    }

    let num_classes = box_data_size - 5;
    let mut detections = Vec::new();

    for b in 0..batch_size {
        for i in 0..num_boxes {
            let objectness = data[[b, i, 4]];

            if objectness < confidence_threshold {
                continue;
            }

            let cx = data[[b, i, 0]] * img_width as f32;
            let cy = data[[b, i, 1]] * img_height as f32;
            let w = data[[b, i, 2]] * img_width as f32;
            let h = data[[b, i, 3]] * img_height as f32;

            // Find best class
            let mut best_class = 0;
            let mut best_score = data[[b, i, 5]];

            for c in 1..num_classes {
                let score = data[[b, i, 5 + c]];
                if score > best_score {
                    best_score = score;
                    best_class = c;
                }
            }

            let confidence = objectness * best_score;

            if confidence >= confidence_threshold {
                let bbox = BoundingBox::from_center(cx, cy, w, h);
                let detection = Detection::new(bbox, best_class as u32, confidence);
                detections.push(detection);
            }
        }
    }

    Ok(detections)
}

/// Decode SSD-style bounding boxes.
///
/// Converts SSD format with anchor boxes to absolute coordinates.
///
/// # Arguments
///
/// * `predictions` - Tensor with SSD predictions
/// * `anchors` - Anchor boxes
/// * `img_width` - Original image width
/// * `img_height` - Original image height
/// * `confidence_threshold` - Minimum confidence threshold
///
/// # Errors
///
/// Returns an error if tensor format is invalid.
#[allow(dead_code)]
pub fn decode_ssd_boxes(
    predictions: &Tensor,
    anchors: &[BoundingBox],
    _img_width: u32,
    _img_height: u32,
    confidence_threshold: f32,
) -> CvResult<Vec<Detection>> {
    let data = predictions.data().to_f32()?;
    let shape = data.shape();

    if shape.len() != 3 {
        return Err(CvError::tensor_error(
            "SSD predictions must be 3D: [batch, num_boxes, box_data]",
        ));
    }

    let num_boxes = shape[1];
    if anchors.len() != num_boxes {
        return Err(CvError::shape_mismatch(
            vec![anchors.len()],
            vec![num_boxes],
        ));
    }

    let mut detections = Vec::new();

    for i in 0..num_boxes {
        let confidence = data[[0, i, 0]];

        if confidence < confidence_threshold {
            continue;
        }

        let dx = data[[0, i, 1]];
        let dy = data[[0, i, 2]];
        let dw = data[[0, i, 3]];
        let dh = data[[0, i, 4]];

        let anchor = &anchors[i];
        let (anchor_cx, anchor_cy) = anchor.center();
        let anchor_w = anchor.width;
        let anchor_h = anchor.height;

        // Decode box
        let cx = anchor_cx + dx * anchor_w;
        let cy = anchor_cy + dy * anchor_h;
        let w = anchor_w * dw.exp();
        let h = anchor_h * dh.exp();

        let bbox = BoundingBox::from_center(cx, cy, w, h);
        let class_id = data[[0, i, 5]] as u32;
        let detection = Detection::new(bbox, class_id, confidence);
        detections.push(detection);
    }

    Ok(detections)
}

/// Apply temperature scaling to logits.
///
/// Scales logits by temperature before softmax to control confidence.
/// Temperature > 1 makes predictions softer, < 1 makes them sharper.
///
/// # Arguments
///
/// * `tensor` - Input tensor with logits
/// * `temperature` - Temperature value (typically 0.1 - 10.0)
///
/// # Errors
///
/// Returns an error if temperature is invalid or tensor operations fail.
pub fn temperature_scale(tensor: &mut Tensor, temperature: f32) -> CvResult<()> {
    if temperature <= 0.0 {
        return Err(CvError::invalid_parameter(
            "temperature",
            format!("{temperature}"),
        ));
    }

    let mut data = tensor.data().to_f32()?;
    data.mapv_inplace(|x| x / temperature);
    *tensor = Tensor::new_f32(data, tensor.layout());
    Ok(())
}

/// Apply top-k filtering to keep only the top k values.
///
/// Sets all but the top k values to a minimum value.
///
/// # Arguments
///
/// * `tensor` - Input tensor (2D: batch x classes)
/// * `k` - Number of top values to keep
/// * `min_value` - Value to set for filtered elements
///
/// # Errors
///
/// Returns an error if tensor format is invalid.
pub fn top_k_filter(tensor: &mut Tensor, k: usize, min_value: f32) -> CvResult<()> {
    let mut data = tensor.data().to_f32()?;
    let shape = data.shape();

    if shape.len() != 2 {
        return Err(CvError::tensor_error("Top-k filtering requires 2D tensor"));
    }

    let batch_size = shape[0];
    let num_classes = shape[1];

    if k > num_classes {
        return Err(CvError::invalid_parameter(
            "k",
            format!("{k} > {num_classes}"),
        ));
    }

    for b in 0..batch_size {
        let mut values: Vec<(usize, f32)> = (0..num_classes).map(|c| (c, data[[b, c]])).collect();

        // Sort by value descending
        values.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Keep only top k
        for (c, _) in values.iter().skip(k) {
            data[[b, *c]] = min_value;
        }
    }

    *tensor = Tensor::new_f32(data, tensor.layout());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        let mut tensor = Tensor::zeros(&[1, 10]);
        let result = sigmoid(&mut tensor);
        assert!(result.is_ok());
    }

    #[test]
    fn test_tanh() {
        let mut tensor = Tensor::zeros(&[1, 10]);
        let result = tanh(&mut tensor);
        assert!(result.is_ok());
    }

    #[test]
    fn test_relu() {
        let mut tensor = Tensor::zeros(&[1, 10]);
        let result = relu(&mut tensor);
        assert!(result.is_ok());
    }

    #[test]
    fn test_confidence_threshold() {
        let detections = vec![
            Detection::new(BoundingBox::new(0.0, 0.0, 10.0, 10.0), 0, 0.9),
            Detection::new(BoundingBox::new(0.0, 0.0, 10.0, 10.0), 0, 0.3),
        ];
        let filtered = confidence_threshold(&detections, 0.5);
        assert_eq!(filtered.len(), 1);
        assert!(filtered[0].confidence >= 0.5);
    }

    #[test]
    fn test_softmax_2d() {
        let mut tensor = Tensor::zeros(&[2, 10]);
        let result = softmax(&mut tensor, 1);
        assert!(result.is_ok());
    }

    #[test]
    fn test_temperature_scale() {
        let mut tensor = Tensor::zeros(&[1, 10]);
        let result = temperature_scale(&mut tensor, 1.5);
        assert!(result.is_ok());
    }

    #[test]
    fn test_temperature_scale_invalid() {
        let mut tensor = Tensor::zeros(&[1, 10]);
        let result = temperature_scale(&mut tensor, -1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_top_k_filter() {
        let mut tensor = Tensor::ones(&[1, 10]);
        let result = top_k_filter(&mut tensor, 3, -1000.0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_decode_yolo_boxes() {
        let predictions = Tensor::zeros(&[1, 10, 85]);
        let result = decode_yolo_boxes(&predictions, 640, 480, 0.5);
        assert!(result.is_ok());
    }
}
