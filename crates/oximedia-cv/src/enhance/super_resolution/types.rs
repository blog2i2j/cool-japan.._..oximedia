//! Type definitions for super-resolution: enums, configs, and options.

use crate::error::{CvError, CvResult};

/// Upscale factor for super-resolution.
///
/// Determines the scaling factor applied to the input image.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UpscaleFactor {
    /// 2x upscaling (output is 2x larger in each dimension).
    X2,
    /// 4x upscaling (output is 4x larger in each dimension).
    X4,
    /// 8x upscaling (output is 8x larger in each dimension).
    X8,
}

impl UpscaleFactor {
    /// Get the numeric scale factor.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::enhance::UpscaleFactor;
    ///
    /// assert_eq!(UpscaleFactor::X2.scale(), 2);
    /// assert_eq!(UpscaleFactor::X4.scale(), 4);
    /// assert_eq!(UpscaleFactor::X8.scale(), 8);
    /// ```
    #[must_use]
    pub const fn scale(&self) -> u32 {
        match self {
            Self::X2 => 2,
            Self::X4 => 4,
            Self::X8 => 8,
        }
    }
}

/// Type of super-resolution neural network model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelType {
    /// ESRGAN (Enhanced Super-Resolution GAN) - High quality photo upscaling.
    ESRGAN,
    /// Real-ESRGAN - Practical real-world image restoration.
    RealESRGAN,
    /// EDSR (Enhanced Deep Residual Networks) - Balanced quality and speed.
    EDSR,
    /// SRCNN (Super-Resolution CNN) - Fast, lightweight model.
    SRCNN,
    /// VDSR (Very Deep Super-Resolution) - Deep network with residual learning.
    VDSR,
}

impl ModelType {
    /// Get a human-readable name for the model type.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::ESRGAN => "ESRGAN",
            Self::RealESRGAN => "Real-ESRGAN",
            Self::EDSR => "EDSR",
            Self::SRCNN => "SRCNN",
            Self::VDSR => "VDSR",
        }
    }

    /// Get typical input normalization range for this model.
    #[must_use]
    pub const fn normalization_range(&self) -> (f32, f32) {
        match self {
            Self::ESRGAN | Self::RealESRGAN | Self::SRCNN | Self::VDSR => (0.0, 1.0),
            Self::EDSR => (0.0, 255.0),
        }
    }

    /// Get whether this model expects mean subtraction.
    #[must_use]
    pub const fn uses_mean_subtraction(&self) -> bool {
        matches!(self, Self::EDSR)
    }

    /// Get RGB mean values for mean subtraction (if applicable).
    #[must_use]
    pub const fn rgb_mean(&self) -> [f32; 3] {
        match self {
            Self::EDSR => [0.4488, 0.4371, 0.4040],
            _ => [0.0, 0.0, 0.0],
        }
    }
}

/// Quality mode for super-resolution.
///
/// Determines the trade-off between quality, speed, and resource usage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualityMode {
    /// Fast mode - Uses lightweight models (SRCNN), good for real-time processing.
    Fast,
    /// Balanced mode - Uses medium-complexity models (EDSR), good balance of quality/speed.
    Balanced,
    /// High quality mode - Uses complex models (Real-ESRGAN), best quality.
    HighQuality,
    /// Animation mode - Optimized for anime/cartoon content.
    Animation,
}

impl QualityMode {
    /// Get the recommended model type for this quality mode.
    #[must_use]
    pub const fn recommended_model(&self) -> ModelType {
        match self {
            Self::Fast => ModelType::SRCNN,
            Self::Balanced => ModelType::EDSR,
            Self::HighQuality | Self::Animation => ModelType::RealESRGAN,
        }
    }

    /// Get the recommended tile size for this quality mode.
    #[must_use]
    pub const fn recommended_tile_size(&self) -> u32 {
        match self {
            Self::Fast => 512,
            Self::Balanced => 256,
            Self::HighQuality | Self::Animation => 128,
        }
    }
}

/// Color space for image processing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorSpace {
    /// RGB color space (3 channels, no subsampling).
    RGB,
    /// YUV 4:2:0 (Y full resolution, U/V subsampled 2x).
    YUV420,
    /// YUV 4:4:4 (all channels full resolution).
    YUV444,
}

impl ColorSpace {
    /// Get the number of channels for this color space.
    #[must_use]
    pub const fn num_channels(&self) -> usize {
        match self {
            Self::RGB | Self::YUV420 | Self::YUV444 => 3,
        }
    }

    /// Check if chroma channels are subsampled.
    #[must_use]
    pub const fn is_subsampled(&self) -> bool {
        matches!(self, Self::YUV420)
    }
}

/// Chroma upscaling mode for YUV processing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChromaUpscaleMode {
    /// Upscale luma only, use simple interpolation for chroma.
    LumaOnly,
    /// Upscale all channels separately.
    Separate,
    /// Upscale all channels jointly (convert to RGB first).
    Joint,
}

/// Configuration for tile-based processing.
#[derive(Debug, Clone, Copy)]
pub struct TileConfig {
    /// Size of each tile (width and height).
    pub tile_size: u32,
    /// Padding around each tile to reduce seam artifacts.
    pub tile_padding: u32,
    /// Feathering width for blending overlapping regions.
    pub feather_width: u32,
}

impl Default for TileConfig {
    fn default() -> Self {
        Self {
            tile_size: 256,
            tile_padding: 16,
            feather_width: 8,
        }
    }
}

impl TileConfig {
    /// Create a new tile configuration.
    ///
    /// # Arguments
    ///
    /// * `tile_size` - Size of each tile (must be >= 64)
    /// * `tile_padding` - Padding around each tile (must be <= tile_size / 4)
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::enhance::TileConfig;
    ///
    /// let config = TileConfig::new(512, 32)?;
    /// assert_eq!(config.tile_size, 512);
    /// # Ok::<(), oximedia_cv::error::CvError>(())
    /// ```
    pub fn new(tile_size: u32, tile_padding: u32) -> CvResult<Self> {
        if tile_size < 64 {
            return Err(CvError::invalid_parameter(
                "tile_size",
                format!("{tile_size} (must be >= 64)"),
            ));
        }
        if tile_padding > tile_size / 4 {
            return Err(CvError::invalid_parameter(
                "tile_padding",
                format!("{tile_padding} (must be <= tile_size / 4)"),
            ));
        }
        Ok(Self {
            tile_size,
            tile_padding,
            feather_width: tile_padding.min(16),
        })
    }

    /// Get the effective tile size including padding.
    #[must_use]
    pub const fn padded_size(&self) -> u32 {
        self.tile_size + 2 * self.tile_padding
    }
}

/// Progress callback function type.
///
/// Called periodically during processing with progress information.
/// Returns `true` to continue processing, `false` to abort.
pub type ProgressCallback = Box<dyn Fn(usize, usize) -> bool + Send + Sync>;

/// Processing options for super-resolution.
#[derive(Debug, Clone)]
pub struct ProcessingOptions {
    /// Enable edge enhancement post-processing.
    pub edge_enhancement: bool,
    /// Enable artifact reduction post-processing.
    pub artifact_reduction: bool,
    /// Enable denoising before upscaling.
    pub denoise: bool,
    /// Chroma upscaling mode for YUV inputs.
    pub chroma_upscale: ChromaUpscaleMode,
    /// Sharpness enhancement amount (0.0 = none, 1.0 = maximum).
    pub sharpness: f32,
    /// Color space for processing.
    pub color_space: ColorSpace,
}

impl Default for ProcessingOptions {
    fn default() -> Self {
        Self {
            edge_enhancement: false,
            artifact_reduction: true,
            denoise: false,
            chroma_upscale: ChromaUpscaleMode::Joint,
            sharpness: 0.0,
            color_space: ColorSpace::RGB,
        }
    }
}

impl ProcessingOptions {
    /// Create new processing options with all enhancements enabled.
    #[must_use]
    pub fn enhanced() -> Self {
        Self {
            edge_enhancement: true,
            artifact_reduction: true,
            denoise: true,
            chroma_upscale: ChromaUpscaleMode::Joint,
            sharpness: 0.3,
            color_space: ColorSpace::RGB,
        }
    }

    /// Create new processing options for fast processing.
    #[must_use]
    pub fn fast() -> Self {
        Self {
            edge_enhancement: false,
            artifact_reduction: false,
            denoise: false,
            chroma_upscale: ChromaUpscaleMode::LumaOnly,
            sharpness: 0.0,
            color_space: ColorSpace::RGB,
        }
    }

    /// Create new processing options for video processing.
    #[must_use]
    pub fn video() -> Self {
        Self {
            edge_enhancement: false,
            artifact_reduction: true,
            denoise: true,
            chroma_upscale: ChromaUpscaleMode::Separate,
            sharpness: 0.1,
            color_space: ColorSpace::YUV420,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_upscale_factor() {
        assert_eq!(UpscaleFactor::X2.scale(), 2);
        assert_eq!(UpscaleFactor::X4.scale(), 4);
        assert_eq!(UpscaleFactor::X8.scale(), 8);
    }

    #[test]
    fn test_model_type_properties() {
        assert_eq!(ModelType::ESRGAN.name(), "ESRGAN");
        assert_eq!(ModelType::RealESRGAN.name(), "Real-ESRGAN");
        assert_eq!(ModelType::EDSR.name(), "EDSR");
        assert_eq!(ModelType::SRCNN.name(), "SRCNN");
        assert_eq!(ModelType::VDSR.name(), "VDSR");

        assert_eq!(ModelType::ESRGAN.normalization_range(), (0.0, 1.0));
        assert_eq!(ModelType::EDSR.normalization_range(), (0.0, 255.0));

        assert!(!ModelType::ESRGAN.uses_mean_subtraction());
        assert!(ModelType::EDSR.uses_mean_subtraction());
    }

    #[test]
    fn test_quality_mode() {
        assert_eq!(QualityMode::Fast.recommended_model(), ModelType::SRCNN);
        assert_eq!(QualityMode::Balanced.recommended_model(), ModelType::EDSR);
        assert_eq!(
            QualityMode::HighQuality.recommended_model(),
            ModelType::RealESRGAN
        );
        assert_eq!(
            QualityMode::Animation.recommended_model(),
            ModelType::RealESRGAN
        );

        assert_eq!(QualityMode::Fast.recommended_tile_size(), 512);
        assert_eq!(QualityMode::Balanced.recommended_tile_size(), 256);
    }

    #[test]
    fn test_color_space() {
        assert_eq!(ColorSpace::RGB.num_channels(), 3);
        assert_eq!(ColorSpace::YUV420.num_channels(), 3);
        assert_eq!(ColorSpace::YUV444.num_channels(), 3);

        assert!(!ColorSpace::RGB.is_subsampled());
        assert!(ColorSpace::YUV420.is_subsampled());
        assert!(!ColorSpace::YUV444.is_subsampled());
    }

    #[test]
    fn test_processing_options() {
        let default_opts = ProcessingOptions::default();
        assert!(!default_opts.edge_enhancement);
        assert!(default_opts.artifact_reduction);
        assert!(!default_opts.denoise);

        let enhanced_opts = ProcessingOptions::enhanced();
        assert!(enhanced_opts.edge_enhancement);
        assert!(enhanced_opts.artifact_reduction);
        assert!(enhanced_opts.denoise);

        let fast_opts = ProcessingOptions::fast();
        assert!(!fast_opts.edge_enhancement);
        assert!(!fast_opts.artifact_reduction);
        assert!(!fast_opts.denoise);
    }

    #[test]
    fn test_tile_config_default() {
        let config = TileConfig::default();
        assert_eq!(config.tile_size, 256);
        assert_eq!(config.tile_padding, 16);
    }

    #[test]
    fn test_tile_config_new() {
        let config = TileConfig::new(512, 32).expect("valid tile config");
        assert_eq!(config.tile_size, 512);
        assert_eq!(config.tile_padding, 32);
        assert_eq!(config.padded_size(), 576);
    }

    #[test]
    fn test_tile_config_invalid_size() {
        let result = TileConfig::new(32, 8);
        assert!(result.is_err());
    }

    #[test]
    fn test_tile_config_invalid_padding() {
        let result = TileConfig::new(256, 100);
        assert!(result.is_err());
    }

    #[test]
    fn test_chroma_upscale_modes() {
        let _luma_only = ChromaUpscaleMode::LumaOnly;
        let _separate = ChromaUpscaleMode::Separate;
        let _joint = ChromaUpscaleMode::Joint;
    }
}
