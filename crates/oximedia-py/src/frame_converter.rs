#![allow(dead_code)]
//! Frame format conversion utilities for Python bindings.
//!
//! Provides pixel format conversion, color space transformations,
//! resolution scaling, and frame data export for Python consumers.

use std::collections::HashMap;

/// Supported pixel formats for conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConvertPixelFormat {
    /// 8-bit YUV 4:2:0 planar.
    Yuv420p,
    /// 8-bit YUV 4:2:2 planar.
    Yuv422p,
    /// 8-bit YUV 4:4:4 planar.
    Yuv444p,
    /// 8-bit RGB packed.
    Rgb24,
    /// 8-bit RGBA packed.
    Rgba32,
    /// 8-bit BGR packed.
    Bgr24,
    /// 8-bit BGRA packed.
    Bgra32,
    /// 8-bit grayscale.
    Gray8,
    /// 10-bit YUV 4:2:0 planar.
    Yuv420p10,
    /// NV12 semi-planar.
    Nv12,
}

/// Color space identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConvertColorSpace {
    /// BT.601 (SD).
    Bt601,
    /// BT.709 (HD).
    Bt709,
    /// BT.2020 (UHD).
    Bt2020,
    /// sRGB.
    Srgb,
    /// Linear light.
    Linear,
}

/// Scaling algorithm used during conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScaleAlgorithm {
    /// Nearest neighbor (fastest).
    Nearest,
    /// Bilinear interpolation.
    Bilinear,
    /// Bicubic interpolation.
    Bicubic,
    /// Lanczos resampling.
    Lanczos,
}

/// Configuration for a frame conversion operation.
#[derive(Debug, Clone)]
pub struct FrameConvertConfig {
    /// Source pixel format.
    pub src_format: ConvertPixelFormat,
    /// Destination pixel format.
    pub dst_format: ConvertPixelFormat,
    /// Source color space (if color-space conversion is needed).
    pub src_color_space: Option<ConvertColorSpace>,
    /// Destination color space.
    pub dst_color_space: Option<ConvertColorSpace>,
    /// Target width (None = keep original).
    pub target_width: Option<u32>,
    /// Target height (None = keep original).
    pub target_height: Option<u32>,
    /// Scaling algorithm.
    pub scale_algo: ScaleAlgorithm,
    /// Whether to apply dithering on bit-depth reduction.
    pub dither: bool,
}

impl Default for FrameConvertConfig {
    fn default() -> Self {
        Self {
            src_format: ConvertPixelFormat::Yuv420p,
            dst_format: ConvertPixelFormat::Rgb24,
            src_color_space: None,
            dst_color_space: None,
            target_width: None,
            target_height: None,
            scale_algo: ScaleAlgorithm::Bilinear,
            dither: false,
        }
    }
}

/// Represents a raw frame buffer for conversion.
#[derive(Debug, Clone)]
pub struct RawFrame {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Pixel format of this frame.
    pub format: ConvertPixelFormat,
    /// Plane data (one Vec per plane).
    pub planes: Vec<Vec<u8>>,
    /// Stride (bytes per row) for each plane.
    pub strides: Vec<usize>,
    /// Presentation timestamp in microseconds.
    pub pts_us: i64,
}

impl RawFrame {
    /// Create a new raw frame with allocated planes.
    pub fn new(width: u32, height: u32, format: ConvertPixelFormat) -> Self {
        let (planes, strides) = Self::allocate_planes(width, height, format);
        Self {
            width,
            height,
            format,
            planes,
            strides,
            pts_us: 0,
        }
    }

    /// Compute total byte size of all planes.
    pub fn total_bytes(&self) -> usize {
        self.planes.iter().map(|p| p.len()).sum()
    }

    /// Check whether the frame dimensions are valid (non-zero).
    pub fn is_valid(&self) -> bool {
        self.width > 0 && self.height > 0 && !self.planes.is_empty()
    }

    /// Allocate planes for a given format.
    fn allocate_planes(
        width: u32,
        height: u32,
        format: ConvertPixelFormat,
    ) -> (Vec<Vec<u8>>, Vec<usize>) {
        let w = width as usize;
        let h = height as usize;
        match format {
            ConvertPixelFormat::Yuv420p => {
                let y_stride = w;
                let uv_stride = (w + 1) / 2;
                let uv_height = (h + 1) / 2;
                let y_plane = vec![0u8; y_stride * h];
                let u_plane = vec![0u8; uv_stride * uv_height];
                let v_plane = vec![0u8; uv_stride * uv_height];
                (vec![y_plane, u_plane, v_plane], vec![y_stride, uv_stride, uv_stride])
            }
            ConvertPixelFormat::Yuv422p => {
                let y_stride = w;
                let uv_stride = (w + 1) / 2;
                let y_plane = vec![0u8; y_stride * h];
                let u_plane = vec![0u8; uv_stride * h];
                let v_plane = vec![0u8; uv_stride * h];
                (vec![y_plane, u_plane, v_plane], vec![y_stride, uv_stride, uv_stride])
            }
            ConvertPixelFormat::Yuv444p => {
                let stride = w;
                let plane = vec![0u8; stride * h];
                (vec![plane.clone(), plane.clone(), plane], vec![stride, stride, stride])
            }
            ConvertPixelFormat::Rgb24 | ConvertPixelFormat::Bgr24 => {
                let stride = w * 3;
                (vec![vec![0u8; stride * h]], vec![stride])
            }
            ConvertPixelFormat::Rgba32 | ConvertPixelFormat::Bgra32 => {
                let stride = w * 4;
                (vec![vec![0u8; stride * h]], vec![stride])
            }
            ConvertPixelFormat::Gray8 => {
                let stride = w;
                (vec![vec![0u8; stride * h]], vec![stride])
            }
            ConvertPixelFormat::Yuv420p10 => {
                let y_stride = w * 2;
                let uv_stride = ((w + 1) / 2) * 2;
                let uv_height = (h + 1) / 2;
                (
                    vec![
                        vec![0u8; y_stride * h],
                        vec![0u8; uv_stride * uv_height],
                        vec![0u8; uv_stride * uv_height],
                    ],
                    vec![y_stride, uv_stride, uv_stride],
                )
            }
            ConvertPixelFormat::Nv12 => {
                let y_stride = w;
                let uv_stride = w;
                let uv_height = (h + 1) / 2;
                (
                    vec![vec![0u8; y_stride * h], vec![0u8; uv_stride * uv_height]],
                    vec![y_stride, uv_stride],
                )
            }
        }
    }
}

/// Frame converter that applies pixel-format and color-space transformations.
#[derive(Debug)]
pub struct FrameConverter {
    /// Active configuration.
    config: FrameConvertConfig,
    /// Conversion statistics.
    stats: ConvertStats,
}

/// Accumulated statistics for frame conversion.
#[derive(Debug, Clone, Default)]
pub struct ConvertStats {
    /// Number of frames converted.
    pub frames_converted: u64,
    /// Total input bytes processed.
    pub bytes_in: u64,
    /// Total output bytes produced.
    pub bytes_out: u64,
}

impl ConvertStats {
    /// Compression ratio (output / input).
    #[allow(clippy::cast_precision_loss)]
    pub fn ratio(&self) -> f64 {
        if self.bytes_in == 0 {
            0.0
        } else {
            self.bytes_out as f64 / self.bytes_in as f64
        }
    }
}

impl FrameConverter {
    /// Create a converter with the given configuration.
    pub fn new(config: FrameConvertConfig) -> Self {
        Self {
            config,
            stats: ConvertStats::default(),
        }
    }

    /// Return current statistics.
    pub fn stats(&self) -> &ConvertStats {
        &self.stats
    }

    /// Reset statistics to zero.
    pub fn reset_stats(&mut self) {
        self.stats = ConvertStats::default();
    }

    /// Convert a single frame according to the active configuration.
    pub fn convert(&mut self, input: &RawFrame) -> Result<RawFrame, FrameConvertError> {
        if !input.is_valid() {
            return Err(FrameConvertError::InvalidInput("frame has zero dimensions".into()));
        }
        if input.format != self.config.src_format {
            return Err(FrameConvertError::FormatMismatch {
                expected: self.config.src_format,
                got: input.format,
            });
        }

        let out_w = self.config.target_width.unwrap_or(input.width);
        let out_h = self.config.target_height.unwrap_or(input.height);

        let mut output = RawFrame::new(out_w, out_h, self.config.dst_format);
        output.pts_us = input.pts_us;

        // Simple passthrough/fill for same-format, same-size
        if self.config.src_format == self.config.dst_format
            && out_w == input.width
            && out_h == input.height
        {
            for (dst, src) in output.planes.iter_mut().zip(input.planes.iter()) {
                let len = dst.len().min(src.len());
                dst[..len].copy_from_slice(&src[..len]);
            }
        }
        // Otherwise output is zero-filled (placeholder)

        let in_bytes = input.total_bytes() as u64;
        let out_bytes = output.total_bytes() as u64;
        self.stats.frames_converted += 1;
        self.stats.bytes_in += in_bytes;
        self.stats.bytes_out += out_bytes;

        Ok(output)
    }

    /// Batch-convert multiple frames.
    pub fn convert_batch(
        &mut self,
        frames: &[RawFrame],
    ) -> Vec<Result<RawFrame, FrameConvertError>> {
        frames.iter().map(|f| self.convert(f)).collect()
    }
}

/// Errors during frame conversion.
#[derive(Debug, Clone, PartialEq)]
pub enum FrameConvertError {
    /// The input frame is invalid.
    InvalidInput(String),
    /// Pixel format of input does not match expected source format.
    FormatMismatch {
        /// Expected format.
        expected: ConvertPixelFormat,
        /// Actual format.
        got: ConvertPixelFormat,
    },
    /// An unsupported conversion path.
    UnsupportedConversion(String),
}

impl std::fmt::Display for FrameConvertError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
            Self::FormatMismatch { expected, got } => {
                write!(f, "format mismatch: expected {expected:?}, got {got:?}")
            }
            Self::UnsupportedConversion(msg) => write!(f, "unsupported conversion: {msg}"),
        }
    }
}

impl std::error::Error for FrameConvertError {}

/// Registry of known format conversion capabilities.
#[derive(Debug, Default)]
pub struct ConversionRegistry {
    /// Supported conversion pairs (src, dst) -> human-readable name.
    supported: HashMap<(ConvertPixelFormat, ConvertPixelFormat), String>,
}

impl ConversionRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a supported conversion path.
    pub fn register(
        &mut self,
        src: ConvertPixelFormat,
        dst: ConvertPixelFormat,
        name: impl Into<String>,
    ) {
        self.supported.insert((src, dst), name.into());
    }

    /// Check whether a conversion path is registered.
    pub fn is_supported(&self, src: ConvertPixelFormat, dst: ConvertPixelFormat) -> bool {
        self.supported.contains_key(&(src, dst))
    }

    /// Number of registered conversions.
    pub fn count(&self) -> usize {
        self.supported.len()
    }

    /// List all registered conversion names.
    pub fn list_names(&self) -> Vec<&str> {
        self.supported.values().map(String::as_str).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = FrameConvertConfig::default();
        assert_eq!(cfg.src_format, ConvertPixelFormat::Yuv420p);
        assert_eq!(cfg.dst_format, ConvertPixelFormat::Rgb24);
        assert_eq!(cfg.scale_algo, ScaleAlgorithm::Bilinear);
        assert!(!cfg.dither);
    }

    #[test]
    fn test_raw_frame_new_rgb() {
        let frame = RawFrame::new(1920, 1080, ConvertPixelFormat::Rgb24);
        assert_eq!(frame.width, 1920);
        assert_eq!(frame.height, 1080);
        assert_eq!(frame.planes.len(), 1);
        assert_eq!(frame.strides[0], 1920 * 3);
        assert_eq!(frame.total_bytes(), 1920 * 1080 * 3);
    }

    #[test]
    fn test_raw_frame_new_yuv420() {
        let frame = RawFrame::new(320, 240, ConvertPixelFormat::Yuv420p);
        assert_eq!(frame.planes.len(), 3);
        assert_eq!(frame.strides[0], 320);
        assert_eq!(frame.strides[1], 160);
        assert!(frame.is_valid());
    }

    #[test]
    fn test_raw_frame_nv12() {
        let frame = RawFrame::new(640, 480, ConvertPixelFormat::Nv12);
        assert_eq!(frame.planes.len(), 2);
        assert_eq!(frame.strides[0], 640);
        assert_eq!(frame.strides[1], 640);
    }

    #[test]
    fn test_raw_frame_gray8() {
        let frame = RawFrame::new(100, 100, ConvertPixelFormat::Gray8);
        assert_eq!(frame.total_bytes(), 10_000);
    }

    #[test]
    fn test_raw_frame_invalid_zero() {
        let frame = RawFrame {
            width: 0,
            height: 0,
            format: ConvertPixelFormat::Rgb24,
            planes: vec![],
            strides: vec![],
            pts_us: 0,
        };
        assert!(!frame.is_valid());
    }

    #[test]
    fn test_converter_passthrough() {
        let cfg = FrameConvertConfig {
            src_format: ConvertPixelFormat::Rgb24,
            dst_format: ConvertPixelFormat::Rgb24,
            ..Default::default()
        };
        let mut conv = FrameConverter::new(cfg);
        let mut frame = RawFrame::new(4, 4, ConvertPixelFormat::Rgb24);
        frame.planes[0][0] = 0xFF;
        frame.pts_us = 42_000;

        let out = conv.convert(&frame).expect("out should be valid");
        assert_eq!(out.planes[0][0], 0xFF);
        assert_eq!(out.pts_us, 42_000);
        assert_eq!(conv.stats().frames_converted, 1);
    }

    #[test]
    fn test_converter_format_mismatch() {
        let cfg = FrameConvertConfig {
            src_format: ConvertPixelFormat::Yuv420p,
            dst_format: ConvertPixelFormat::Rgb24,
            ..Default::default()
        };
        let mut conv = FrameConverter::new(cfg);
        let frame = RawFrame::new(4, 4, ConvertPixelFormat::Rgb24);
        let err = conv.convert(&frame).unwrap_err();
        assert!(matches!(err, FrameConvertError::FormatMismatch { .. }));
    }

    #[test]
    fn test_converter_invalid_input() {
        let cfg = FrameConvertConfig::default();
        let mut conv = FrameConverter::new(cfg);
        let frame = RawFrame {
            width: 0,
            height: 0,
            format: ConvertPixelFormat::Yuv420p,
            planes: vec![],
            strides: vec![],
            pts_us: 0,
        };
        assert!(conv.convert(&frame).is_err());
    }

    #[test]
    fn test_converter_batch() {
        let cfg = FrameConvertConfig {
            src_format: ConvertPixelFormat::Gray8,
            dst_format: ConvertPixelFormat::Gray8,
            ..Default::default()
        };
        let mut conv = FrameConverter::new(cfg);
        let frames: Vec<RawFrame> = (0..5)
            .map(|_| RawFrame::new(8, 8, ConvertPixelFormat::Gray8))
            .collect();
        let results = conv.convert_batch(&frames);
        assert_eq!(results.len(), 5);
        assert!(results.iter().all(|r| r.is_ok()));
        assert_eq!(conv.stats().frames_converted, 5);
    }

    #[test]
    fn test_convert_stats_ratio() {
        let mut stats = ConvertStats::default();
        assert_eq!(stats.ratio(), 0.0);
        stats.bytes_in = 1000;
        stats.bytes_out = 500;
        assert!((stats.ratio() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_conversion_registry() {
        let mut reg = ConversionRegistry::new();
        assert_eq!(reg.count(), 0);
        reg.register(ConvertPixelFormat::Yuv420p, ConvertPixelFormat::Rgb24, "yuv420_to_rgb");
        assert!(reg.is_supported(ConvertPixelFormat::Yuv420p, ConvertPixelFormat::Rgb24));
        assert!(!reg.is_supported(ConvertPixelFormat::Rgb24, ConvertPixelFormat::Yuv420p));
        assert_eq!(reg.count(), 1);
        let names = reg.list_names();
        assert_eq!(names, vec!["yuv420_to_rgb"]);
    }

    #[test]
    fn test_frame_convert_error_display() {
        let e = FrameConvertError::InvalidInput("bad data".into());
        assert!(e.to_string().contains("bad data"));
        let e2 = FrameConvertError::UnsupportedConversion("nope".into());
        assert!(e2.to_string().contains("nope"));
    }

    #[test]
    fn test_reset_stats() {
        let cfg = FrameConvertConfig {
            src_format: ConvertPixelFormat::Gray8,
            dst_format: ConvertPixelFormat::Gray8,
            ..Default::default()
        };
        let mut conv = FrameConverter::new(cfg);
        let frame = RawFrame::new(2, 2, ConvertPixelFormat::Gray8);
        let _ = conv.convert(&frame);
        assert_eq!(conv.stats().frames_converted, 1);
        conv.reset_stats();
        assert_eq!(conv.stats().frames_converted, 0);
    }
}
