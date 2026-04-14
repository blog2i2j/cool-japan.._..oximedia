//! Image enhancement module.
//!
//! This module provides advanced image enhancement algorithms including:
//!
//! - **Super-resolution**: AI-powered upscaling using multiple neural network models
//! - **Denoising**: Neural network-based image denoising
//!
//! # Features
//!
//! ## Super-Resolution
//!
//! The `super_resolution` module provides video super-resolution with:
//! - Multiple model types (ESRGAN, Real-ESRGAN, EDSR, SRCNN, VDSR)
//! - Multiple upscaling factors (2x, 4x, 8x)
//! - Quality modes (Fast, Balanced, High Quality, Animation)
//! - Video-specific features:
//!   - Temporal consistency filtering
//!   - Motion-aware processing
//!   - Frame buffering
//! - YUV color space support
//! - Pre/post processing (denoising, edge enhancement, artifact reduction)
//! - Tile-based processing for large frames
//! - Model caching for efficient batch processing
//! - GPU acceleration via ONNX Runtime
//!
//! ## Denoising
//!
//! The `denoising` module provides CNN-based denoising with:
//! - Blind denoising (automatic noise estimation)
//! - Noise-level-aware denoising
//! - Color and luminance denoising control
//! - Tile-based processing
//!
//! # Example
//!
//! ```
//! use oximedia_cv::enhance::{SuperResolutionEnhancer, UpscaleMode};
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // CPU-based bicubic upscaling (no ONNX required)
//! let input = vec![128u8; 64 * 64 * 3];
//! let enhancer = SuperResolutionEnhancer::new(UpscaleMode::Bicubic2x);
//! let upscaled = enhancer.upscale(&input, 64, 64)?;
//! assert_eq!(upscaled.len(), 128 * 128 * 3);
//! Ok(())
//! }
//! ```
//!
//! ```no_run
//! # #[cfg(feature = "onnx")]
//! # fn onnx_example() -> oximedia_cv::error::CvResult<()> {
//! use oximedia_cv::enhance::{
//!     SuperResolutionModel, ModelType, UpscaleFactor, QualityMode,
//!     VideoSuperResolution, VideoFrame, NeuralDenoiser,
//! };
//!
//! // Super-resolution with quality mode (requires 'onnx' feature)
//! let mut model = SuperResolutionModel::from_quality_mode(
//!     QualityMode::HighQuality,
//!     UpscaleFactor::X4,
//! )?;
//! let input = vec![0u8; 256 * 256 * 3];
//! let upscaled = model.upscale(&input, 256, 256)?;
//! # Ok(())
//! # }
//! ```

pub mod cpu_upscale;
#[cfg(feature = "onnx")]
pub mod denoising;
pub mod sharpening;
#[cfg(feature = "onnx")]
pub mod super_resolution;
pub mod temporal_denoising;

// Re-export CPU upscaling items (always available)
pub use cpu_upscale::{
    apply_unsharp_mask, calculate_mse, calculate_psnr, calculate_ssim, cubic_weight,
    upscale_bicubic, upscale_bilinear_rgb, upscale_nearest, SuperResolutionEnhancer, UpscaleMode,
};

// Re-export commonly used items
#[cfg(feature = "onnx")]
pub use denoising::{
    noise_estimation, BatchDenoiser, DenoisingConfig, DenoisingProgressCallback, NeuralDenoiser,
    NoiseLevel,
};
#[cfg(feature = "onnx")]
pub use super_resolution::{
    utils, BatchUpscaler, ChromaUpscaleMode, ColorSpace, EsrganUpscaler, ModelCache, ModelType,
    MotionEstimator, ProcessingOptions, ProgressCallback, QualityMode, SuperResolutionModel,
    TemporalFilter, TileConfig, UpscaleFactor, VideoFrame, VideoSuperResolution,
};
