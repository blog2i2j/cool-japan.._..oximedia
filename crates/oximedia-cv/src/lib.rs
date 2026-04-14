//! Computer vision module for `OxiMedia`.
//!
//! `oximedia-cv` provides computer vision algorithms and image processing
//! capabilities for the `OxiMedia` multimedia framework. This includes:
//!
//! - **Image processing**: Resize, color conversion, filtering, edge detection
//! - **Detection**: Face detection, motion detection, object detection, corner detection
//! - **Transforms**: Affine and perspective transformations
//! - **Enhancement**: Super-resolution and denoising using AI models
//! - **Machine Learning**: ONNX Runtime integration for ML model inference
//! - **Tracking**: Optical flow, feature tracking, object tracking
//! - **Stabilization**: Video stabilization with motion smoothing
//! - **Scene detection**: Video scene changes and shot boundary detection
//! - **Quality**: Video quality assessment metrics (PSNR, SSIM, VMAF)
//! - **Interpolation**: Frame interpolation using optical flow for frame rate conversion
//! - **Chroma key**: Green screen and blue screen processing with spill suppression
//! - **Content-aware scaling**: Seam carving and intelligent resizing
//! - **Interlace detection**: Interlacing and telecine detection with IVTC recommendations
//! - **Motion blur**: Motion blur synthesis and removal with deconvolution algorithms
//! - **Fingerprinting**: Perceptual content fingerprinting for video and audio
//!
//! # Modules
//!
//! - [`image`]: Image processing operations (resize, filter, histogram, etc.)
//! - [`detect`]: Detection algorithms (face, motion, object, corner)
//! - [`transform`]: Geometric transformations (affine, perspective)
//! - [`enhance`]: Image enhancement (super-resolution, denoising)
//! - `ml`: Machine learning and ONNX Runtime integration
//! - [`tracking`]: Video tracking and optical flow (incl. enhanced SORT with [`tracking::SortTrackerV2`] / [`tracking::LkTracker`])
//! - [`stabilize`]: Video stabilization with motion smoothing
//! - [`scene`]: Video scene detection and shot boundary detection
//! - [`quality`]: Video quality metrics (PSNR, SSIM, VMAF, temporal)
//! - [`interpolate`]: Frame interpolation using optical flow for smooth motion
//! - [`chroma_key`]: Chroma keying (green screen) with spill suppression and compositing
//! - [`scale`]: Content-aware scaling using seam carving and saliency detection
//! - [`interlace`]: Interlacing and telecine detection for video content analysis
//! - [`motion_blur`]: Motion blur synthesis and removal with deconvolution
//! - [`fingerprint`]: Perceptual fingerprinting for content identification and matching
//!
//! # Example
//!
//! ```
//! use oximedia_cv::image::{ResizeMethod, ColorSpace};
//! use oximedia_cv::detect::BoundingBox;
//! use oximedia_cv::transform::AffineTransform;
//!
//! // Example usage
//! let bbox = BoundingBox::new(10.0, 20.0, 100.0, 150.0);
//! assert!(bbox.area() > 0.0);
//! ```

#![warn(missing_docs)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::similar_names)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::unused_self)]
#![allow(clippy::items_after_statements)]
#![allow(clippy::manual_memcpy)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::trivially_copy_pass_by_ref)]
#![allow(clippy::single_match_else)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::manual_swap)]
#![allow(clippy::doc_markdown)]
#![allow(dead_code)]
#![allow(unused_imports, unused_variables)]
#![allow(unused_mut)]

pub mod action_recognition;
pub mod background_subtraction;
pub mod blob_detector;
pub mod bounding_box;
pub mod chroma_key;
pub mod color_cluster;
pub mod contour;
pub mod deep_sort;
pub mod depth_estimation;
pub mod detect;
pub mod enhance;
pub mod error;
pub mod feature_extract;
pub mod feature_match;
pub mod fingerprint;
pub mod histogram_backproject;
pub mod hough_transform;
pub mod image;
pub mod instance_segmentation;
pub mod interlace;
pub mod interpolate;
pub mod keypoint;
pub mod lane_detect;
pub mod lens_distortion;
#[cfg(feature = "onnx")]
pub mod ml;
pub mod morphology;
pub mod motion_blur;
pub mod motion_vector;
pub mod obj_tracking;
pub mod optical_flow_field;
pub mod panorama_stitch;
pub mod pose_estimation;
pub mod quality;
pub mod scale;
pub mod scene;
pub mod segmentation;
pub mod stabilize;
pub mod style_transfer;
pub mod superpixel;
pub mod text_detect;
pub mod texture_analysis;
pub mod tracking;
pub mod transform;
pub mod video_matting;

// Re-export commonly used items at crate root
pub use error::{CvError, CvResult};
