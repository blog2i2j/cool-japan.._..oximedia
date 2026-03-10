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
pub mod field_scale;
pub mod lanczos;
pub mod pad;
pub mod pad_scale;
pub mod quality_metric;
pub mod quality_metrics;
pub mod resampler;
pub mod resolution_ladder;
pub mod roi_scale;
pub mod scale_config;
pub mod scale_filter;
pub mod scale_pipeline;
pub mod sharpness_scale;
pub mod super_resolution;
pub mod thumbnail;
pub mod tile;

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
}

impl fmt::Display for ScalingMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bilinear => write!(f, "Bilinear"),
            Self::Bicubic => write!(f, "Bicubic"),
            Self::Lanczos => write!(f, "Lanczos"),
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
}

impl ScalingParams {
    /// Create new scaling parameters
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            mode: ScalingMode::Lanczos,
            aspect_ratio: AspectRatioMode::Letterbox,
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

    /// Calculate output dimensions preserving aspect ratio
    pub fn calculate_dimensions(&self, src_width: u32, src_height: u32) -> (u32, u32) {
        match self.params.aspect_ratio {
            AspectRatioMode::Stretch => (self.params.width, self.params.height),
            AspectRatioMode::Letterbox | AspectRatioMode::Crop => {
                let src_aspect = src_width as f64 / src_height as f64;
                let dst_aspect = self.params.width as f64 / self.params.height as f64;

                if (src_aspect - dst_aspect).abs() < f64::EPSILON {
                    (self.params.width, self.params.height)
                } else if src_aspect > dst_aspect {
                    // Source is wider, fit height, scale width proportionally
                    let w = (self.params.height as f64 * src_aspect) as u32;
                    (w, self.params.height)
                } else {
                    // Source is taller, fit width, scale height proportionally
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
    }
}
