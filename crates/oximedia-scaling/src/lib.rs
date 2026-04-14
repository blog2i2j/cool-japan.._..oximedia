//! Professional video scaling operations for OxiMedia
//!
//! This crate provides high-quality video scaling functionality including:
//! - Bilinear and bicubic interpolation
//! - Lanczos filtering
//! - Aspect ratio preservation
//! - Batch scaling operations

#![warn(missing_docs, rust_2018_idioms, unreachable_pub, unsafe_code)]

pub mod adaptive_scaling;
pub mod aspect_preserve;
pub mod aspect_ratio;
pub mod batch_scale;
pub mod bicubic;
pub mod chroma_scale;
pub mod content_aware_scale;
pub mod crop;
pub mod crop_scale;
pub mod deinterlace;
pub mod ewa_resample;
pub mod field_scale;
pub mod half_pixel;
pub mod lanczos;
pub mod nearest_neighbor;
pub mod pad;
pub mod pad_scale;
pub mod perceptual_sharpening;
pub mod quality_metric;
pub mod quality_metrics;
pub mod resampler;
pub mod resolution_ladder;
pub mod roi_scale;
pub mod scale_config;
pub mod scale_filter;
pub mod scale_pipeline;
pub mod sharpness_scale;
pub mod super_res;
pub mod super_resolution;
pub mod thumbnail;
pub mod tile;

/// Seam carving with forward energy for content-aware image resizing.
pub mod seam_carve;

/// SIMD-accelerated pixel interpolation for resize operations.
#[allow(unsafe_code)]
pub mod simd_interp;

/// Aspect-ratio-preserving crop helper.
pub mod aspect_ratio_crop;

/// Edge-directed interpolation (NEDI-like) for improved diagonal edge rendering.
pub mod edge_directed_interpolation;

/// Film grain removal before scaling and re-synthesis at target resolution.
pub mod film_grain_scale;

/// PQ/HLG tone-mapping during resolution changes.
pub mod hdr_scaling;

/// Multi-pass scaling for extreme scale ratios.
pub mod multi_pass_scale;

/// Internal format negotiation helpers.
pub mod negotiate;

/// Lightweight neural 2x/4x upscaling.
pub mod neural_upscale;

/// Internal padding utilities.
pub mod padding;

/// Parallel (rayon) row processing for VideoScaler.
pub mod parallel_scale;

/// PSNR/SSIM quality regression helpers.
pub mod quality_regression;

/// Resolution recommendation engine.
pub mod resolution_recommender;

/// Ring-buffer row cache for vertical filter passes.
pub mod ring_buffer_cache;

/// Fast low-quality preview scaling.
pub mod scale_preview;

/// Sharpness analysis helpers.
pub mod sharpness;

/// Temporal scaling / frame-rate conversion.
pub mod temporal_scaling;

/// Thumbnail generation pipeline.
pub mod thumbnail_generator;

/// Watermark-safe scaling that preserves watermark positions.
pub mod watermark_safe_scale;

// Re-exports from new modules for ergonomic access.
pub use ewa_resample::{lanczos_kernel, mitchell_filter, sinc, EwaFilter, EwaResampler};
pub use half_pixel::{
    bilinear_interp, cubic_interp, cubic_interp_2d, CoordinateMapper, HalfPixelMode, ScaleInterp,
    ScaleKernel,
};
pub use nearest_neighbor::{NearestNeighborConfig, NearestNeighborScaler};
pub use perceptual_sharpening::{
    gaussian_blur_1d, local_laplacian, sharpen, AdaptiveSharpener, CasSharpener, SharpnessMode,
    UnsharpMask,
};
pub use resolution_ladder::{
    compute_optimal_ladder, ContentDifficultyScore, OptimalRung, PerTitleLadder, PerceptualLadder,
    QualityTarget, RungSelector,
};
pub use seam_carve::{EnergyFunction, ScalingError, SeamCarver, SeamCarvingConfig};

use serde::{Deserialize, Serialize};
use std::fmt;

/// Video scaling mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScalingMode {
    /// Bilinear interpolation - fast and reasonable quality
    Bilinear,
    /// Bicubic interpolation - better quality
    Bicubic,
    /// Lanczos filtering - highest quality
    Lanczos,
    /// Nearest-neighbor - no interpolation, ideal for pixel art and retro content
    NearestNeighbor,
}

impl fmt::Display for ScalingMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bilinear => write!(f, "Bilinear"),
            Self::Bicubic => write!(f, "Bicubic"),
            Self::Lanczos => write!(f, "Lanczos"),
            Self::NearestNeighbor => write!(f, "NearestNeighbor"),
        }
    }
}

/// Aspect ratio preservation mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AspectRatioMode {
    /// Stretch to fill target dimensions
    Stretch,
    /// Preserve aspect ratio with letterboxing
    Letterbox,
    /// Preserve aspect ratio with cropping
    Crop,
}

/// Pixel Aspect Ratio — the ratio of a pixel's displayed width to its height.
///
/// Square pixels have PAR 1:1. Common broadcast PARs include 10:11 (NTSC 4:3),
/// 40:33 (NTSC 16:9), 12:11 (PAL 4:3), and 16:11 (PAL 16:9).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PixelAspectRatio {
    /// Numerator (horizontal component).
    pub num: u32,
    /// Denominator (vertical component).
    pub den: u32,
}

impl PixelAspectRatio {
    /// Create a new PAR. Denominator is clamped to a minimum of 1.
    pub fn new(num: u32, den: u32) -> Self {
        Self {
            num,
            den: den.max(1),
        }
    }

    /// Square pixel (1:1).
    pub fn square() -> Self {
        Self { num: 1, den: 1 }
    }

    /// NTSC 4:3 anamorphic PAR (10:11).
    pub fn ntsc_4_3() -> Self {
        Self { num: 10, den: 11 }
    }

    /// NTSC 16:9 anamorphic PAR (40:33).
    pub fn ntsc_16_9() -> Self {
        Self { num: 40, den: 33 }
    }

    /// PAL 4:3 anamorphic PAR (12:11).
    pub fn pal_4_3() -> Self {
        Self { num: 12, den: 11 }
    }

    /// PAL 16:9 anamorphic PAR (16:11).
    pub fn pal_16_9() -> Self {
        Self { num: 16, den: 11 }
    }

    /// Returns the PAR as a floating-point value.
    pub fn to_float(&self) -> f64 {
        self.num as f64 / self.den as f64
    }

    /// Returns true if this PAR represents square pixels.
    pub fn is_square(&self) -> bool {
        self.num == self.den
    }
}

impl Default for PixelAspectRatio {
    fn default() -> Self {
        Self::square()
    }
}

impl fmt::Display for PixelAspectRatio {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.num, self.den)
    }
}

/// Video scaling parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingParams {
    /// Target width
    pub width: u32,
    /// Target height
    pub height: u32,
    /// Scaling mode
    pub mode: ScalingMode,
    /// Aspect ratio preservation
    pub aspect_ratio: AspectRatioMode,
    /// Source pixel aspect ratio (for non-square pixel correction).
    pub source_par: PixelAspectRatio,
}

impl ScalingParams {
    /// Create new scaling parameters
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            mode: ScalingMode::Lanczos,
            aspect_ratio: AspectRatioMode::Letterbox,
            source_par: PixelAspectRatio::square(),
        }
    }

    /// Set scaling mode
    pub fn with_mode(mut self, mode: ScalingMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set aspect ratio mode
    pub fn with_aspect_ratio(mut self, mode: AspectRatioMode) -> Self {
        self.aspect_ratio = mode;
        self
    }

    /// Set source pixel aspect ratio for PAR correction.
    ///
    /// When a non-square PAR is set, `VideoScaler::calculate_dimensions` will
    /// convert the source's storage aspect ratio to its display aspect ratio
    /// before computing the output size.
    pub fn with_source_par(mut self, par: PixelAspectRatio) -> Self {
        self.source_par = par;
        self
    }
}

/// Scaler for video operations
pub struct VideoScaler {
    params: ScalingParams,
}

impl VideoScaler {
    /// Create a new video scaler with given parameters
    pub fn new(params: ScalingParams) -> Self {
        Self { params }
    }

    /// Get scaling parameters
    pub fn params(&self) -> &ScalingParams {
        &self.params
    }

    /// Calculate output dimensions preserving aspect ratio.
    ///
    /// When a non-square source pixel aspect ratio (PAR) is configured, the
    /// storage dimensions are first converted to display dimensions before
    /// computing the scaled output. This correctly handles anamorphic content
    /// such as NTSC/PAL SD broadcasts.
    pub fn calculate_dimensions(&self, src_width: u32, src_height: u32) -> (u32, u32) {
        self.calculate_dimensions_with_par(src_width, src_height, &self.params.source_par)
    }

    /// Calculate output dimensions with an explicit PAR override.
    ///
    /// The Display Aspect Ratio (DAR) is computed as:
    /// ```text
    /// DAR = (src_width × PAR_num) / (src_height × PAR_den)
    /// ```
    /// The DAR is then used as the source aspect ratio for letterbox/crop scaling.
    pub fn calculate_dimensions_with_par(
        &self,
        src_width: u32,
        src_height: u32,
        par: &PixelAspectRatio,
    ) -> (u32, u32) {
        match self.params.aspect_ratio {
            AspectRatioMode::Stretch => (self.params.width, self.params.height),
            AspectRatioMode::Letterbox | AspectRatioMode::Crop => {
                // Compute the Display Aspect Ratio incorporating PAR.
                let display_width = src_width as f64 * par.to_float();
                let src_aspect = display_width / src_height as f64;
                let dst_aspect = self.params.width as f64 / self.params.height as f64;

                if (src_aspect - dst_aspect).abs() < f64::EPSILON {
                    (self.params.width, self.params.height)
                } else if src_aspect > dst_aspect {
                    // Source display is wider, fit height, scale width proportionally
                    let w = (self.params.height as f64 * src_aspect) as u32;
                    (w, self.params.height)
                } else {
                    // Source display is taller, fit width, scale height proportionally
                    let h = (self.params.width as f64 / src_aspect) as u32;
                    (self.params.width, h)
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scaling_params_creation() {
        let params = ScalingParams::new(1920, 1080);
        assert_eq!(params.width, 1920);
        assert_eq!(params.height, 1080);
        assert_eq!(params.mode, ScalingMode::Lanczos);
        assert!(params.source_par.is_square());
    }

    #[test]
    fn test_scaling_mode_with_builder() {
        let params = ScalingParams::new(1920, 1080)
            .with_mode(ScalingMode::Bicubic)
            .with_aspect_ratio(AspectRatioMode::Crop);

        assert_eq!(params.mode, ScalingMode::Bicubic);
        assert_eq!(params.aspect_ratio, AspectRatioMode::Crop);
    }

    #[test]
    fn test_scaling_mode_nearest_neighbor() {
        let params = ScalingParams::new(640, 480).with_mode(ScalingMode::NearestNeighbor);
        assert_eq!(params.mode, ScalingMode::NearestNeighbor);
    }

    #[test]
    fn test_scaler_creation() {
        let params = ScalingParams::new(1920, 1080);
        let scaler = VideoScaler::new(params);
        assert_eq!(scaler.params().width, 1920);
    }

    #[test]
    fn test_calculate_dimensions_stretch() {
        let params = ScalingParams::new(1920, 1080).with_aspect_ratio(AspectRatioMode::Stretch);
        let scaler = VideoScaler::new(params);
        let (w, h) = scaler.calculate_dimensions(3840, 2160);
        assert_eq!((w, h), (1920, 1080));
    }

    #[test]
    fn test_calculate_dimensions_letterbox() {
        let params = ScalingParams::new(1920, 1080).with_aspect_ratio(AspectRatioMode::Letterbox);
        let scaler = VideoScaler::new(params);
        // 4:3 video scaled to 16:9 - narrower aspect ratio
        // 4:3 = 1.333, 16:9 = 1.777
        // Source is narrower, so fit to width and scale height down
        let (w, h) = scaler.calculate_dimensions(1024, 768);
        assert_eq!(w, 1920);
        // Height should be scaled proportionally: 1920 / (1024/768) = 1440
        assert_eq!(h, 1440);
    }

    #[test]
    fn test_scaling_mode_display() {
        assert_eq!(ScalingMode::Bilinear.to_string(), "Bilinear");
        assert_eq!(ScalingMode::Bicubic.to_string(), "Bicubic");
        assert_eq!(ScalingMode::Lanczos.to_string(), "Lanczos");
        assert_eq!(ScalingMode::NearestNeighbor.to_string(), "NearestNeighbor");
    }

    // ── PAR correction tests ────────────────────────────────────────────────

    #[test]
    fn test_par_square_default() {
        let par = PixelAspectRatio::default();
        assert!(par.is_square());
        assert!((par.to_float() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_par_ntsc_4_3() {
        let par = PixelAspectRatio::ntsc_4_3();
        assert_eq!(par.num, 10);
        assert_eq!(par.den, 11);
        assert!(!par.is_square());
        assert!((par.to_float() - 10.0 / 11.0).abs() < 1e-6);
    }

    #[test]
    fn test_par_display() {
        let par = PixelAspectRatio::new(16, 11);
        assert_eq!(par.to_string(), "16:11");
    }

    #[test]
    fn test_par_zero_den_clamped() {
        let par = PixelAspectRatio::new(1, 0);
        assert_eq!(par.den, 1);
    }

    #[test]
    fn test_calculate_dimensions_square_par_same_as_no_par() {
        // With square PAR, behavior should be identical to default
        let params = ScalingParams::new(1920, 1080)
            .with_aspect_ratio(AspectRatioMode::Letterbox)
            .with_source_par(PixelAspectRatio::square());
        let scaler = VideoScaler::new(params);
        let (w, h) = scaler.calculate_dimensions(1024, 768);
        assert_eq!(w, 1920);
        assert_eq!(h, 1440);
    }

    #[test]
    fn test_calculate_dimensions_ntsc_par_correction() {
        // NTSC SD 720x480 with 10:11 PAR should display as ~654x480 (4:3 DAR)
        // DAR = 720 * (10/11) / 480 = 654.5 / 480 = 1.3636...
        let params = ScalingParams::new(1920, 1080)
            .with_aspect_ratio(AspectRatioMode::Letterbox)
            .with_source_par(PixelAspectRatio::ntsc_4_3());
        let scaler = VideoScaler::new(params);
        let (w, h) = scaler.calculate_dimensions(720, 480);
        // DAR = 720*10/11 / 480 = 1.3636... < 1.7778 (16:9)
        // Source is narrower (taller), so fit width -> h = 1920 / 1.3636 = 1408
        assert_eq!(w, 1920);
        assert!(
            h > 1080,
            "height {h} should exceed 1080 for 4:3 content in 16:9 target"
        );
    }

    #[test]
    fn test_calculate_dimensions_wide_par_correction() {
        // 720x480 with 40:33 PAR → DAR = 720*40/33 / 480 = 872.7/480 = 1.818 ≈ 16:9
        let params = ScalingParams::new(1920, 1080)
            .with_aspect_ratio(AspectRatioMode::Letterbox)
            .with_source_par(PixelAspectRatio::ntsc_16_9());
        let scaler = VideoScaler::new(params);
        let (w, h) = scaler.calculate_dimensions(720, 480);
        // DAR ≈ 1.818 which is close to 16:9 = 1.778
        // Source is wider → fit height → w = 1080 * 1.818 = 1963
        assert!(w >= 1920, "width {w} should be near 1920");
        assert_eq!(h, 1080);
    }

    #[test]
    fn test_calculate_dimensions_stretch_ignores_par() {
        // Stretch mode should not be affected by PAR
        let params = ScalingParams::new(1920, 1080)
            .with_aspect_ratio(AspectRatioMode::Stretch)
            .with_source_par(PixelAspectRatio::ntsc_4_3());
        let scaler = VideoScaler::new(params);
        let (w, h) = scaler.calculate_dimensions(720, 480);
        assert_eq!((w, h), (1920, 1080));
    }

    #[test]
    fn test_calculate_dimensions_with_par_override() {
        let params = ScalingParams::new(1920, 1080).with_aspect_ratio(AspectRatioMode::Letterbox);
        let scaler = VideoScaler::new(params);
        let par = PixelAspectRatio::pal_4_3();
        let (w, h) = scaler.calculate_dimensions_with_par(720, 576, &par);
        // DAR = 720 * 12/11 / 576 = 785.45 / 576 = 1.3637... (4:3)
        // Source narrower than 16:9 → fit width → h > 1080
        assert_eq!(w, 1920);
        assert!(h > 1080);
    }

    #[test]
    fn test_par_pal_16_9() {
        let par = PixelAspectRatio::pal_16_9();
        assert_eq!(par.num, 16);
        assert_eq!(par.den, 11);
    }
}
