// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Image format handling and properties.

use super::ImageFormat;
use crate::{ConversionError, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Image format properties.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageProperties {
    /// Image format
    pub format: ImageFormat,
    /// Width in pixels
    pub width: u32,
    /// Height in pixels
    pub height: u32,
    /// Bit depth per channel
    pub bit_depth: u32,
    /// Has alpha channel
    pub has_alpha: bool,
    /// Color space
    pub color_space: ImageColorSpace,
    /// ICC profile data
    pub icc_profile: Option<Vec<u8>>,
}

/// Image color space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImageColorSpace {
    /// sRGB color space
    Srgb,
    /// Linear RGB
    LinearRgb,
    /// Adobe RGB (1998)
    AdobeRgb,
    /// `ProPhoto` RGB
    ProPhotoRgb,
    /// Grayscale
    Grayscale,
}

/// Image format detector.
#[derive(Debug, Clone)]
pub struct ImageFormatDetector;

impl ImageFormatDetector {
    /// Create a new image format detector.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Detect image format from file.
    ///
    /// If the file exists, the magic bytes are read to determine the format.
    /// For non-existent paths (e.g. in pipeline configuration), the extension
    /// is used as a fallback.
    pub fn detect(&self, path: &Path) -> Result<ImageProperties> {
        let extension = path.extension().and_then(|e| e.to_str()).unwrap_or("");

        // Attempt magic-byte detection when the file is readable, falling
        // back to extension-based detection for non-existent paths.
        let format_opt: Option<ImageFormat> = if path.exists() {
            self.detect_from_magic(path)
                .or_else(|| self.format_from_extension(extension))
        } else {
            self.format_from_extension(extension)
        };

        let format = format_opt.ok_or_else(|| {
            ConversionError::UnsupportedCodec(format!("Unsupported image format: {extension}"))
        })?;

        // Derive per-format defaults for bit depth and alpha.
        let (bit_depth, has_alpha) = match format {
            ImageFormat::Png => (8, true),
            ImageFormat::Webp => (8, true),
            ImageFormat::Tiff => (8, false),
            ImageFormat::Dpx => (10, false),
            ImageFormat::Exr => (16, true),
        };

        Ok(ImageProperties {
            format,
            width: 0,
            height: 0,
            bit_depth,
            has_alpha,
            color_space: ImageColorSpace::Srgb,
            icc_profile: None,
        })
    }

    /// Identify format from magic bytes at the start of the file.
    fn detect_from_magic(&self, path: &Path) -> Option<ImageFormat> {
        use std::io::Read;
        let mut buf = [0u8; 12];
        let mut f = std::fs::File::open(path).ok()?;
        let n = f.read(&mut buf).ok()?;
        if n < 4 {
            return None;
        }
        // PNG: \x89PNG
        if buf[0] == 0x89 && &buf[1..4] == b"PNG" {
            return Some(ImageFormat::Png);
        }
        // RIFF/WEBP
        if &buf[0..4] == b"RIFF" && n >= 12 && &buf[8..12] == b"WEBP" {
            return Some(ImageFormat::Webp);
        }
        // TIFF little-endian (II) or big-endian (MM)
        if (&buf[0..4] == b"II\x2a\x00") || (&buf[0..4] == b"MM\x00\x2a") {
            return Some(ImageFormat::Tiff);
        }
        // DPX: SDPX or XPDS
        if &buf[0..4] == b"SDPX" || &buf[0..4] == b"XPDS" {
            return Some(ImageFormat::Dpx);
        }
        // OpenEXR: 0x762f3101
        if buf[0] == 0x76 && buf[1] == 0x2f && buf[2] == 0x31 && buf[3] == 0x01 {
            return Some(ImageFormat::Exr);
        }
        None
    }

    /// Map a file extension string to an `ImageFormat`.
    fn format_from_extension(&self, extension: &str) -> Option<ImageFormat> {
        match extension.to_lowercase().as_str() {
            "png" => Some(ImageFormat::Png),
            "webp" => Some(ImageFormat::Webp),
            "tif" | "tiff" => Some(ImageFormat::Tiff),
            "dpx" => Some(ImageFormat::Dpx),
            "exr" => Some(ImageFormat::Exr),
            _ => None,
        }
    }

    /// Check if file is an image.
    #[must_use]
    pub fn is_image(&self, path: &Path) -> bool {
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            matches!(
                ext.to_lowercase().as_str(),
                "png"
                    | "webp"
                    | "tif"
                    | "tiff"
                    | "dpx"
                    | "exr"
                    | "jpg"
                    | "jpeg"
                    | "bmp"
                    | "gif"
                    | "psd"
            )
        } else {
            false
        }
    }
}

impl Default for ImageFormatDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Image format validator.
#[derive(Debug, Clone)]
pub struct ImageFormatValidator;

impl ImageFormatValidator {
    /// Create a new validator.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Validate image resolution.
    pub fn validate_resolution(&self, width: u32, height: u32) -> Result<()> {
        const MAX_WIDTH: u32 = 16384;
        const MAX_HEIGHT: u32 = 16384;
        const MIN_WIDTH: u32 = 1;
        const MIN_HEIGHT: u32 = 1;

        if !(MIN_WIDTH..=MAX_WIDTH).contains(&width) {
            return Err(ConversionError::InvalidInput(format!(
                "Image width {width} is outside valid range {MIN_WIDTH}-{MAX_WIDTH}"
            )));
        }

        if !(MIN_HEIGHT..=MAX_HEIGHT).contains(&height) {
            return Err(ConversionError::InvalidInput(format!(
                "Image height {height} is outside valid range {MIN_HEIGHT}-{MAX_HEIGHT}"
            )));
        }

        Ok(())
    }

    /// Validate bit depth for format.
    pub fn validate_bit_depth(&self, format: ImageFormat, bit_depth: u32) -> Result<()> {
        let valid_depths = match format {
            ImageFormat::Png => vec![8, 16],
            ImageFormat::Webp => vec![8],
            ImageFormat::Tiff => vec![8, 16, 32],
            ImageFormat::Dpx => vec![10, 12, 16],
            ImageFormat::Exr => vec![16, 32],
        };

        if valid_depths.contains(&bit_depth) {
            Ok(())
        } else {
            Err(ConversionError::InvalidInput(format!(
                "Bit depth {bit_depth} is not valid for format {format}"
            )))
        }
    }
}

impl Default for ImageFormatValidator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_format_detector() {
        let detector = ImageFormatDetector::new();

        assert!(detector.is_image(Path::new("test.png")));
        assert!(detector.is_image(Path::new("test.webp")));
        assert!(detector.is_image(Path::new("test.tiff")));
        assert!(!detector.is_image(Path::new("test.mp4")));
    }

    #[test]
    fn test_detect_format_from_extension() {
        let detector = ImageFormatDetector::new();

        let result = detector.detect(Path::new("test.png"));
        assert!(result.is_ok());
        assert_eq!(result.unwrap().format, ImageFormat::Png);

        let result = detector.detect(Path::new("test.webp"));
        assert!(result.is_ok());
        assert_eq!(result.unwrap().format, ImageFormat::Webp);
    }

    #[test]
    fn test_image_format_validator() {
        let validator = ImageFormatValidator::new();

        assert!(validator.validate_resolution(1920, 1080).is_ok());
        assert!(validator.validate_resolution(0, 100).is_err());
        assert!(validator.validate_resolution(20000, 100).is_err());

        assert!(validator.validate_bit_depth(ImageFormat::Png, 8).is_ok());
        assert!(validator.validate_bit_depth(ImageFormat::Png, 16).is_ok());
        assert!(validator.validate_bit_depth(ImageFormat::Png, 32).is_err());

        assert!(validator.validate_bit_depth(ImageFormat::Exr, 16).is_ok());
        assert!(validator.validate_bit_depth(ImageFormat::Exr, 32).is_ok());
        assert!(validator.validate_bit_depth(ImageFormat::Exr, 8).is_err());
    }
}
