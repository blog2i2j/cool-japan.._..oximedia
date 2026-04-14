//! YOLO (You Only Look Once) object detection.
//!
//! This module provides real-time object detection using YOLO models (YOLOv5/YOLOv8)
//! via ONNX Runtime. It supports:
//!
//! - YOLOv5 and YOLOv8 model formats
//! - Anchor-free detection (YOLOv8 style)
//! - Multi-scale predictions (P3, P4, P5)
//! - Non-Maximum Suppression (NMS)
//! - Confidence thresholding
//! - Letterbox preprocessing with aspect ratio preservation
//! - COCO dataset classes (80 classes)
//!
//! # Example
//!
//! ```no_run
//! # use oximedia_cv::detect::yolo::{YoloDetector, YoloConfig};
//! # use oximedia_cv::error::CvResult;
//! # fn example() -> CvResult<()> {
//! let config = YoloConfig::default()
//!     .with_confidence_threshold(0.5)
//!     .with_iou_threshold(0.45);
//!
//! let mut detector = YoloDetector::new("model.onnx", config)?;
//! let image = vec![0u8; 640 * 640 * 3]; // RGB image
//! let detections = detector.detect(&image, 640, 640)?;
//! # Ok(())
//! # }
//! ```

#![allow(clippy::too_many_arguments)]
#![allow(clippy::similar_names)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::module_name_repetitions)]

use crate::detect::object::{BoundingBox, Detection, ObjectDetector};
use crate::detect::yolo_utils::{
    decode_yolov5_output, decode_yolov8_output, draw_detections, letterbox_resize, LetterboxParams,
};

use crate::error::{CvError, CvResult};
use ndarray::{Array, ArrayD, IxDyn};
use oxionnx::Session;
use std::collections::HashMap;
use std::path::Path;

/// YOLO model version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum YoloVersion {
    /// YOLOv5 (anchor-based or anchor-free).
    V5,
    /// YOLOv8 (anchor-free).
    V8,
    /// YOLOv9 (programmable gradient information, anchor-free).
    V9,
}

/// Dynamic input resolution strategy.
///
/// Controls how input resolution is determined for the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputResolution {
    /// Fixed resolution (traditional).
    Fixed(u32, u32),
    /// Dynamic resolution: automatically selects from standard sizes
    /// based on the input image, rounding to the nearest multiple of stride.
    Dynamic {
        /// Minimum input dimension.
        min_size: u32,
        /// Maximum input dimension.
        max_size: u32,
        /// Stride alignment (typically 32 for YOLO).
        stride: u32,
    },
}

impl Default for InputResolution {
    fn default() -> Self {
        Self::Fixed(640, 640)
    }
}

impl InputResolution {
    /// Compute the actual input size for a given image.
    ///
    /// For fixed resolution, returns the fixed size.
    /// For dynamic resolution, computes the optimal size based on image dimensions,
    /// aligned to the stride.
    #[must_use]
    pub fn compute_size(&self, img_width: u32, img_height: u32) -> (u32, u32) {
        match *self {
            Self::Fixed(w, h) => (w, h),
            Self::Dynamic {
                min_size,
                max_size,
                stride,
            } => {
                let stride = stride.max(1);
                // Scale the longer side to max_size, preserving aspect ratio
                let max_dim = img_width.max(img_height);
                let scale = if max_dim > max_size {
                    max_size as f64 / max_dim as f64
                } else if max_dim < min_size {
                    min_size as f64 / max_dim as f64
                } else {
                    1.0
                };

                let new_w = ((img_width as f64 * scale) as u32).max(stride);
                let new_h = ((img_height as f64 * scale) as u32).max(stride);

                // Align to stride
                let aligned_w = ((new_w + stride - 1) / stride) * stride;
                let aligned_h = ((new_h + stride - 1) / stride) * stride;

                (
                    aligned_w.clamp(min_size, max_size),
                    aligned_h.clamp(min_size, max_size),
                )
            }
        }
    }
}

/// YOLO detector configuration.
#[derive(Debug, Clone)]
pub struct YoloConfig {
    /// Model version (YOLOv5, YOLOv8, or YOLOv9).
    pub version: YoloVersion,
    /// Input size (width, height). Default is (640, 640).
    pub input_size: (u32, u32),
    /// Dynamic input resolution strategy. When set, overrides `input_size`.
    pub input_resolution: Option<InputResolution>,
    /// Confidence threshold for detections. Default is 0.25.
    pub confidence_threshold: f32,
    /// IoU threshold for NMS. Default is 0.45.
    pub iou_threshold: f32,
    /// Maximum number of detections to return. Default is 300.
    pub max_detections: usize,
    /// Class names. If None, uses COCO classes.
    pub class_names: Option<Vec<String>>,
    /// Whether to use per-class NMS. Default is false (class-agnostic).
    pub per_class_nms: bool,
    /// ONNX execution providers. Default is CPU.
    pub execution_providers: Vec<String>,
}

impl Default for YoloConfig {
    fn default() -> Self {
        Self {
            version: YoloVersion::V8,
            input_size: (640, 640),
            input_resolution: None,
            confidence_threshold: 0.25,
            iou_threshold: 0.45,
            max_detections: 300,
            class_names: None,
            per_class_nms: false,
            execution_providers: vec!["CPUExecutionProvider".to_string()],
        }
    }
}

impl YoloConfig {
    /// Create a new YOLO configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the YOLO version.
    #[must_use]
    pub fn with_version(mut self, version: YoloVersion) -> Self {
        self.version = version;
        self
    }

    /// Set the input size.
    #[must_use]
    pub fn with_input_size(mut self, width: u32, height: u32) -> Self {
        self.input_size = (width, height);
        self
    }

    /// Set the confidence threshold.
    #[must_use]
    pub fn with_confidence_threshold(mut self, threshold: f32) -> Self {
        self.confidence_threshold = threshold;
        self
    }

    /// Set the IoU threshold for NMS.
    #[must_use]
    pub fn with_iou_threshold(mut self, threshold: f32) -> Self {
        self.iou_threshold = threshold;
        self
    }

    /// Set the maximum number of detections.
    #[must_use]
    pub fn with_max_detections(mut self, max: usize) -> Self {
        self.max_detections = max;
        self
    }

    /// Set custom class names.
    #[must_use]
    pub fn with_class_names(mut self, names: Vec<String>) -> Self {
        self.class_names = Some(names);
        self
    }

    /// Enable per-class NMS.
    #[must_use]
    pub fn with_per_class_nms(mut self, enabled: bool) -> Self {
        self.per_class_nms = enabled;
        self
    }

    /// Set ONNX execution providers.
    #[must_use]
    pub fn with_execution_providers(mut self, providers: Vec<String>) -> Self {
        self.execution_providers = providers;
        self
    }

    /// Set dynamic input resolution strategy.
    ///
    /// When set, the detector will automatically compute optimal input dimensions
    /// based on the input image, aligned to the model stride.
    #[must_use]
    pub fn with_input_resolution(mut self, resolution: InputResolution) -> Self {
        self.input_resolution = Some(resolution);
        self
    }

    /// Get the effective input size for a given image.
    ///
    /// If dynamic resolution is configured, computes the optimal size.
    /// Otherwise returns the fixed input_size.
    #[must_use]
    pub fn effective_input_size(&self, img_width: u32, img_height: u32) -> (u32, u32) {
        match &self.input_resolution {
            Some(resolution) => resolution.compute_size(img_width, img_height),
            None => self.input_size,
        }
    }
}

/// YOLO object detector.
///
/// Performs real-time object detection using YOLO models (YOLOv5/YOLOv8).
pub struct YoloDetector {
    session: Session,
    config: YoloConfig,
    class_names: Vec<String>,
    num_classes: usize,
}

impl YoloDetector {
    /// Create a new YOLO detector from a model file.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to the ONNX model file
    /// * `config` - YOLO configuration
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The model file cannot be loaded
    /// - The ONNX session cannot be created
    /// - The model format is invalid
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use oximedia_cv::detect::yolo::{YoloDetector, YoloConfig};
    /// # use oximedia_cv::error::CvResult;
    /// # fn example() -> CvResult<()> {
    /// let config = YoloConfig::default();
    /// let detector = YoloDetector::new("yolov8n.onnx", config)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(model_path: impl AsRef<Path>, config: YoloConfig) -> CvResult<Self> {
        // Initialize oxionnx session
        let session = Session::builder()
            .with_optimization_level(oxionnx::OptLevel::All)
            .load(model_path.as_ref())
            .map_err(|e| CvError::detection_failed(format!("Failed to load model: {e}")))?;

        // Get class names
        let class_names = config.class_names.clone().unwrap_or_else(coco_class_names);
        let num_classes = class_names.len();

        Ok(Self {
            session,
            config,
            class_names,
            num_classes,
        })
    }

    /// Create a new YOLO detector from model bytes.
    ///
    /// # Arguments
    ///
    /// * `model_bytes` - ONNX model as bytes
    /// * `config` - YOLO configuration
    ///
    /// # Errors
    ///
    /// Returns an error if the ONNX session cannot be created.
    pub fn from_bytes(model_bytes: &[u8], config: YoloConfig) -> CvResult<Self> {
        let session = Session::builder()
            .with_optimization_level(oxionnx::OptLevel::All)
            .load_from_bytes(model_bytes)
            .map_err(|e| {
                CvError::detection_failed(format!("Failed to load model from memory: {e}"))
            })?;

        let class_names = config.class_names.clone().unwrap_or_else(coco_class_names);
        let num_classes = class_names.len();

        Ok(Self {
            session,
            config,
            class_names,
            num_classes,
        })
    }

    /// Detect objects in an RGB image.
    ///
    /// # Arguments
    ///
    /// * `image` - RGB image data (row-major, channels last)
    /// * `width` - Image width
    /// * `height` - Image height
    ///
    /// # Returns
    ///
    /// Vector of detections with bounding boxes, class IDs, and confidence scores.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Image dimensions are invalid
    /// - Preprocessing fails
    /// - Model inference fails
    /// - Output decoding fails
    pub fn detect(&mut self, image: &[u8], width: u32, height: u32) -> CvResult<Vec<Detection>> {
        // Validate input
        if width == 0 || height == 0 {
            return Err(CvError::invalid_dimensions(width, height));
        }

        let expected_size = (width as usize) * (height as usize) * 3;
        if image.len() != expected_size {
            return Err(CvError::insufficient_data(expected_size, image.len()));
        }

        // Compute effective input size (dynamic or fixed)
        let (input_w, input_h) = self.config.effective_input_size(width, height);

        // Preprocess image (letterbox resize and normalize)
        let (input_tensor, letterbox_params) =
            self.preprocess(image, width, height, input_w, input_h)?;

        // Run inference
        let outputs = self.run_inference(&input_tensor)?;

        // Postprocess outputs
        let detections = self.postprocess(&outputs, &letterbox_params, width, height)?;

        Ok(detections)
    }

    /// Detect objects and return image with drawn bounding boxes.
    ///
    /// # Arguments
    ///
    /// * `image` - RGB image data
    /// * `width` - Image width
    /// * `height` - Image height
    ///
    /// # Returns
    ///
    /// Tuple of (annotated image, detections).
    ///
    /// # Errors
    ///
    /// Returns an error if detection fails.
    pub fn detect_with_visualization(
        &mut self,
        image: &[u8],
        width: u32,
        height: u32,
    ) -> CvResult<(Vec<u8>, Vec<Detection>)> {
        let detections = self.detect(image, width, height)?;
        let annotated = draw_detections(image, width, height, &detections)?;
        Ok((annotated, detections))
    }

    /// Preprocess image for YOLO model.
    ///
    /// Applies letterbox resize to maintain aspect ratio and normalizes to [0, 1].
    fn preprocess(
        &self,
        image: &[u8],
        width: u32,
        height: u32,
        input_w: u32,
        input_h: u32,
    ) -> CvResult<(ArrayD<f32>, LetterboxParams)> {
        // Letterbox resize
        let (resized, params) = letterbox_resize(image, width, height, input_w, input_h)?;

        // Convert to CHW format and normalize to [0, 1]
        let mut input_tensor = Array::zeros(IxDyn(&[1, 3, input_h as usize, input_w as usize]));

        for c in 0..3 {
            for y in 0..input_h as usize {
                for x in 0..input_w as usize {
                    let idx = (y * input_w as usize + x) * 3 + c;
                    let value = resized[idx] as f32 / 255.0;
                    input_tensor[[0, c, y, x]] = value;
                }
            }
        }

        Ok((input_tensor, params))
    }

    /// Run model inference and return extracted tensor data.
    fn run_inference(&mut self, input: &ArrayD<f32>) -> CvResult<ArrayD<f32>> {
        // Convert ndarray → flat Vec<f32> + shape for oxionnx
        let flat: Vec<f32> = input.iter().copied().collect();
        let shape: Vec<usize> = input.shape().to_vec();
        let tensor = oxionnx::Tensor::new(flat, shape);

        // Determine input name (use first available, or fall back to "images")
        let input_name = self
            .session
            .input_names()
            .first()
            .cloned()
            .unwrap_or_else(|| "images".to_string());

        // Run inference
        let mut inputs = HashMap::new();
        inputs.insert(input_name.as_str(), tensor);
        let outputs = self
            .session
            .run(&inputs)
            .map_err(|e| CvError::detection_failed(format!("Inference failed: {e}")))?;

        // Extract first output tensor
        let output_name = self
            .session
            .output_names()
            .first()
            .cloned()
            .unwrap_or_default();
        let out_tensor = outputs
            .get(&output_name)
            .ok_or_else(|| CvError::detection_failed("No output tensor found".to_owned()))?;

        let out_shape: Vec<usize> = out_tensor.shape.clone();
        ArrayD::from_shape_vec(IxDyn(&out_shape), out_tensor.data.clone())
            .map_err(|e| CvError::detection_failed(format!("Failed to create output array: {e}")))
    }

    /// Postprocess model outputs to detections.
    fn postprocess(
        &self,
        output_tensor: &ArrayD<f32>,
        letterbox_params: &LetterboxParams,
        orig_width: u32,
        orig_height: u32,
    ) -> CvResult<Vec<Detection>> {
        // Decode based on YOLO version
        // YOLOv9 uses the same anchor-free output format as YOLOv8
        // (programmable gradient information affects training, not inference output)
        let detections = match self.config.version {
            YoloVersion::V5 => decode_yolov5_output(
                output_tensor,
                self.config.confidence_threshold,
                self.config.iou_threshold,
                self.num_classes,
                self.config.per_class_nms,
                self.config.max_detections,
            )?,
            YoloVersion::V8 | YoloVersion::V9 => decode_yolov8_output(
                output_tensor,
                self.config.confidence_threshold,
                self.config.iou_threshold,
                self.num_classes,
                self.config.per_class_nms,
                self.config.max_detections,
            )?,
        };

        // Transform coordinates back to original image space
        let detections =
            self.transform_detections(detections, letterbox_params, orig_width, orig_height);

        Ok(detections)
    }

    /// Transform detection coordinates from model space to original image space.
    fn transform_detections(
        &self,
        detections: Vec<Detection>,
        params: &LetterboxParams,
        orig_width: u32,
        orig_height: u32,
    ) -> Vec<Detection> {
        detections
            .into_iter()
            .map(|mut det| {
                // Scale coordinates
                let x = (det.bbox.x - params.pad_left as f32) / params.scale;
                let y = (det.bbox.y - params.pad_top as f32) / params.scale;
                let w = det.bbox.width / params.scale;
                let h = det.bbox.height / params.scale;

                // Clamp to image bounds
                det.bbox =
                    BoundingBox::new(x, y, w, h).clamp(orig_width as f32, orig_height as f32);

                // Add class name
                if det.class_id < self.class_names.len() as u32 {
                    det.class_name = Some(self.class_names[det.class_id as usize].clone());
                }

                det
            })
            .collect()
    }

    /// Get the model input size.
    #[must_use]
    pub const fn input_size(&self) -> (u32, u32) {
        self.config.input_size
    }

    /// Get the confidence threshold.
    #[must_use]
    pub const fn confidence_threshold(&self) -> f32 {
        self.config.confidence_threshold
    }

    /// Get the IoU threshold.
    #[must_use]
    pub const fn iou_threshold(&self) -> f32 {
        self.config.iou_threshold
    }

    /// Get the number of classes.
    #[must_use]
    pub const fn num_classes(&self) -> usize {
        self.num_classes
    }
}

impl ObjectDetector for YoloDetector {
    fn detect(&mut self, image: &[u8], width: u32, height: u32) -> CvResult<Vec<Detection>> {
        YoloDetector::detect(self, image, width, height)
    }

    fn class_names(&self) -> &[String] {
        &self.class_names
    }
}

/// Get COCO dataset class names (80 classes).
///
/// # Returns
///
/// Vector of 80 COCO class names.
#[must_use]
pub fn coco_class_names() -> Vec<String> {
    vec![
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coco_class_names() {
        let names = coco_class_names();
        assert_eq!(names.len(), 80);
        assert_eq!(names[0], "person");
        assert_eq!(names[79], "toothbrush");
    }

    #[test]
    fn test_yolo_config_default() {
        let config = YoloConfig::default();
        assert_eq!(config.input_size, (640, 640));
        assert_eq!(config.confidence_threshold, 0.25);
        assert_eq!(config.iou_threshold, 0.45);
        assert_eq!(config.max_detections, 300);
        assert!(!config.per_class_nms);
    }

    #[test]
    fn test_yolo_config_builder() {
        let config = YoloConfig::new()
            .with_version(YoloVersion::V5)
            .with_input_size(416, 416)
            .with_confidence_threshold(0.5)
            .with_iou_threshold(0.4)
            .with_max_detections(100)
            .with_per_class_nms(true);

        assert_eq!(config.version, YoloVersion::V5);
        assert_eq!(config.input_size, (416, 416));
        assert_eq!(config.confidence_threshold, 0.5);
        assert_eq!(config.iou_threshold, 0.4);
        assert_eq!(config.max_detections, 100);
        assert!(config.per_class_nms);
    }

    #[test]
    fn test_yolo_version() {
        assert_eq!(YoloVersion::V5, YoloVersion::V5);
        assert_ne!(YoloVersion::V5, YoloVersion::V8);
    }

    #[test]
    fn test_invalid_dimensions() {
        // This test would require an actual YOLO model
        // For now, just test configuration
        let config = YoloConfig::default();
        assert_eq!(config.input_size, (640, 640));
    }

    #[test]
    fn test_class_names_custom() {
        let custom_names = vec!["cat".to_string(), "dog".to_string()];
        let config = YoloConfig::new().with_class_names(custom_names.clone());
        assert_eq!(config.class_names, Some(custom_names));
    }

    #[test]
    fn test_coco_class_names_specific_entries() {
        let names = coco_class_names();
        assert_eq!(names[14], "bird");
        assert_eq!(names[15], "cat");
        assert_eq!(names[16], "dog");
        assert_eq!(names[39], "bottle");
        assert_eq!(names[56], "chair");
    }

    #[test]
    fn test_yolo_config_execution_providers() {
        let config =
            YoloConfig::new().with_execution_providers(vec!["CUDAExecutionProvider".to_string()]);
        assert_eq!(config.execution_providers[0], "CUDAExecutionProvider");
    }

    #[test]
    fn test_yolo_config_default_providers() {
        let config = YoloConfig::default();
        assert_eq!(config.execution_providers.len(), 1);
        assert_eq!(config.execution_providers[0], "CPUExecutionProvider");
    }

    #[test]
    fn test_yolo_version_equality() {
        assert_eq!(YoloVersion::V8, YoloVersion::V8);
        assert_eq!(YoloVersion::V5, YoloVersion::V5);
    }

    #[test]
    fn test_yolo_config_max_detections() {
        let config = YoloConfig::new().with_max_detections(50);
        assert_eq!(config.max_detections, 50);

        let config2 = YoloConfig::new().with_max_detections(1000);
        assert_eq!(config2.max_detections, 1000);
    }

    #[test]
    fn test_coco_class_names_no_duplicates() {
        let names = coco_class_names();
        let mut seen = std::collections::HashSet::new();
        for name in &names {
            assert!(seen.insert(name.as_str()), "Duplicate class name: {name}");
        }
    }

    #[test]
    fn test_coco_class_names_non_empty() {
        let names = coco_class_names();
        for name in &names {
            assert!(!name.is_empty(), "Class name should not be empty");
        }
    }

    #[test]
    fn test_yolo_version_v9() {
        let config = YoloConfig::new().with_version(YoloVersion::V9);
        assert_eq!(config.version, YoloVersion::V9);
        assert_ne!(config.version, YoloVersion::V8);
        assert_ne!(config.version, YoloVersion::V5);
    }

    #[test]
    fn test_input_resolution_fixed() {
        let res = InputResolution::Fixed(640, 640);
        assert_eq!(res.compute_size(1920, 1080), (640, 640));
        assert_eq!(res.compute_size(100, 100), (640, 640));
    }

    #[test]
    fn test_input_resolution_dynamic_downscale() {
        let res = InputResolution::Dynamic {
            min_size: 320,
            max_size: 1280,
            stride: 32,
        };
        // 1920x1080 -> scale down to max 1280
        let (w, h) = res.compute_size(1920, 1080);
        assert_eq!(w % 32, 0, "Width must be aligned to stride");
        assert_eq!(h % 32, 0, "Height must be aligned to stride");
        assert!(w <= 1280);
        assert!(h <= 1280);
    }

    #[test]
    fn test_input_resolution_dynamic_upscale() {
        let res = InputResolution::Dynamic {
            min_size: 320,
            max_size: 1280,
            stride: 32,
        };
        // 100x100 -> scale up to min 320
        let (w, h) = res.compute_size(100, 100);
        assert!(w >= 320);
        assert!(h >= 320);
        assert_eq!(w % 32, 0);
        assert_eq!(h % 32, 0);
    }

    #[test]
    fn test_input_resolution_dynamic_passthrough() {
        let res = InputResolution::Dynamic {
            min_size: 320,
            max_size: 1280,
            stride: 32,
        };
        // 640x480 -> no scaling needed, just align
        let (w, h) = res.compute_size(640, 480);
        assert_eq!(w % 32, 0);
        assert_eq!(h % 32, 0);
        assert!(w >= 320 && w <= 1280);
        assert!(h >= 320 && h <= 1280);
    }

    #[test]
    fn test_input_resolution_default() {
        let res = InputResolution::default();
        assert_eq!(res, InputResolution::Fixed(640, 640));
    }

    #[test]
    fn test_config_with_input_resolution() {
        let config = YoloConfig::new().with_input_resolution(InputResolution::Dynamic {
            min_size: 320,
            max_size: 1280,
            stride: 32,
        });
        assert!(config.input_resolution.is_some());

        let (w, h) = config.effective_input_size(800, 600);
        assert_eq!(w % 32, 0);
        assert_eq!(h % 32, 0);
    }

    #[test]
    fn test_effective_input_size_without_dynamic() {
        let config = YoloConfig::new().with_input_size(416, 416);
        assert_eq!(config.effective_input_size(1920, 1080), (416, 416));
    }

    #[test]
    fn test_yolo_v9_config_full() {
        let config = YoloConfig::new()
            .with_version(YoloVersion::V9)
            .with_input_resolution(InputResolution::Dynamic {
                min_size: 320,
                max_size: 1280,
                stride: 32,
            })
            .with_confidence_threshold(0.3)
            .with_iou_threshold(0.5)
            .with_max_detections(200)
            .with_per_class_nms(true);

        assert_eq!(config.version, YoloVersion::V9);
        assert!(config.input_resolution.is_some());
        assert_eq!(config.confidence_threshold, 0.3);
        assert_eq!(config.iou_threshold, 0.5);
        assert_eq!(config.max_detections, 200);
        assert!(config.per_class_nms);
    }
}
