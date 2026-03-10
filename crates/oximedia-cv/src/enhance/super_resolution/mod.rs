//! Video Super-Resolution using Neural Networks.
//!
//! This module provides AI-powered super-resolution for video upscaling using various
//! neural network models via ONNX Runtime. Supports multiple quality modes, temporal
//! consistency, and video-specific optimizations.
//!
//! # Supported Models
//!
//! - **ESRGAN** (Enhanced Super-Resolution GAN) - High quality photo upscaling
//! - **Real-ESRGAN** - Practical real-world image restoration and enhancement
//! - **EDSR** (Enhanced Deep Residual Networks) - Balanced quality and performance
//! - **SRCNN** (Super-Resolution CNN) - Fast, lightweight upscaling
//! - **VDSR** (Very Deep Super-Resolution) - Deep network with residual learning
//!
//! # Features
//!
//! - Multiple upscaling factors (2x, 4x, 8x)
//! - Tile-based processing for large images/frames
//! - Temporal consistency for video
//! - Frame buffering and motion-aware processing
//! - YUV color space support
//! - Edge enhancement and artifact reduction
//! - GPU acceleration via ONNX Runtime
//! - Model caching for efficient batch processing
//! - Quality modes (Fast, Balanced, High Quality, Animation)
//!
//! # Example
//!
//! ```no_run
//! use oximedia_cv::enhance::{SuperResolutionModel, UpscaleFactor, QualityMode};
//!
//! // Create a model with quality mode
//! let model = SuperResolutionModel::from_quality_mode(
//!     QualityMode::HighQuality,
//!     UpscaleFactor::X4,
//! )?;
//!
//! // Upscale an image
//! let input_image = vec![0u8; 256 * 256 * 3];
//! let upscaled = model.upscale(&input_image, 256, 256)?;
//! # Ok::<(), oximedia_cv::error::CvError>(())
//! ```

#![forbid(unsafe_code)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::too_many_lines)]

pub mod cache;
pub mod esrgan;
pub mod model;
pub mod types;
pub mod utils;
pub mod video;

// Re-export types
pub use cache::ModelCache;
pub use esrgan::{BatchUpscaler, EsrganUpscaler};
pub use model::SuperResolutionModel;
pub use types::{
    ChromaUpscaleMode, ColorSpace, ModelType, ProcessingOptions, ProgressCallback, QualityMode,
    TileConfig, UpscaleFactor,
};
pub use video::{MotionEstimator, TemporalFilter, VideoFrame, VideoSuperResolution};
