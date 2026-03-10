//! Output formatting for decoded frames.
//!
//! This module handles final output formatting including pixel format
//! conversion, bit depth handling, and frame metadata attachment.

#![forbid(unsafe_code)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::items_after_statements)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::identity_op)]
#![allow(clippy::range_plus_one)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::useless_conversion)]
#![allow(clippy::redundant_closure_for_method_calls)]
#![allow(clippy::single_match_else)]
#![allow(dead_code)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::unused_self)]
#![allow(clippy::trivially_copy_pass_by_ref)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::similar_names)]
#![allow(clippy::match_same_arms)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::comparison_chain)]
#![allow(clippy::if_then_some_else_none)]

use super::{FrameBuffer, PlaneBuffer, ReconstructResult};

// =============================================================================
// Output Format
// =============================================================================

/// Output pixel format.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OutputFormat {
    /// YUV planar (same as internal).
    #[default]
    YuvPlanar,
    /// YUV semi-planar (NV12/NV21).
    YuvSemiPlanar,
    /// RGB interleaved.
    Rgb,
    /// RGBA interleaved.
    Rgba,
    /// BGR interleaved.
    Bgr,
    /// BGRA interleaved.
    Bgra,
}

impl OutputFormat {
    /// Get number of output planes.
    #[must_use]
    pub const fn num_planes(self) -> usize {
        match self {
            Self::YuvPlanar => 3,
            Self::YuvSemiPlanar => 2,
            Self::Rgb | Self::Bgr => 1,
            Self::Rgba | Self::Bgra => 1,
        }
    }

    /// Get bytes per pixel for packed formats.
    #[must_use]
    pub const fn bytes_per_pixel(self) -> usize {
        match self {
            Self::YuvPlanar | Self::YuvSemiPlanar => 1,
            Self::Rgb | Self::Bgr => 3,
            Self::Rgba | Self::Bgra => 4,
        }
    }

    /// Check if this is a planar format.
    #[must_use]
    pub const fn is_planar(self) -> bool {
        matches!(self, Self::YuvPlanar | Self::YuvSemiPlanar)
    }

    /// Check if this is an RGB format.
    #[must_use]
    pub const fn is_rgb(self) -> bool {
        matches!(self, Self::Rgb | Self::Rgba | Self::Bgr | Self::Bgra)
    }
}

// =============================================================================
// Output Configuration
// =============================================================================

/// Configuration for output formatting.
#[derive(Clone, Debug)]
pub struct OutputConfig {
    /// Output format.
    pub format: OutputFormat,
    /// Output bit depth.
    pub bit_depth: u8,
    /// Dither when reducing bit depth.
    pub dither: bool,
    /// Full range output (0-255 vs 16-235).
    pub full_range: bool,
    /// Output width (for scaling).
    pub width: Option<u32>,
    /// Output height (for scaling).
    pub height: Option<u32>,
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            format: OutputFormat::YuvPlanar,
            bit_depth: 8,
            dither: false,
            full_range: false,
            width: None,
            height: None,
        }
    }
}

impl OutputConfig {
    /// Create a new output configuration.
    #[must_use]
    pub fn new(format: OutputFormat) -> Self {
        Self {
            format,
            ..Default::default()
        }
    }

    /// Set output bit depth.
    #[must_use]
    pub const fn with_bit_depth(mut self, bit_depth: u8) -> Self {
        self.bit_depth = bit_depth;
        self
    }

    /// Enable dithering.
    #[must_use]
    pub const fn with_dither(mut self) -> Self {
        self.dither = true;
        self
    }

    /// Set full range output.
    #[must_use]
    pub const fn with_full_range(mut self) -> Self {
        self.full_range = true;
        self
    }

    /// Set output dimensions.
    #[must_use]
    pub const fn with_dimensions(mut self, width: u32, height: u32) -> Self {
        self.width = Some(width);
        self.height = Some(height);
        self
    }
}

// =============================================================================
// Output Buffer
// =============================================================================

/// Output buffer for formatted frames.
#[derive(Clone, Debug)]
pub struct OutputBuffer {
    /// Output data planes.
    planes: Vec<Vec<u8>>,
    /// Width.
    width: u32,
    /// Height.
    height: u32,
    /// Output format.
    format: OutputFormat,
    /// Bit depth.
    bit_depth: u8,
    /// Timestamp.
    timestamp: i64,
}

impl OutputBuffer {
    /// Create a new output buffer.
    #[must_use]
    pub fn new(width: u32, height: u32, format: OutputFormat, bit_depth: u8) -> Self {
        let bytes_per_sample = if bit_depth > 8 { 2 } else { 1 };

        let planes = match format {
            OutputFormat::YuvPlanar => {
                vec![
                    vec![0u8; width as usize * height as usize * bytes_per_sample],
                    vec![0u8; (width as usize / 2) * (height as usize / 2) * bytes_per_sample],
                    vec![0u8; (width as usize / 2) * (height as usize / 2) * bytes_per_sample],
                ]
            }
            OutputFormat::YuvSemiPlanar => {
                vec![
                    vec![0u8; width as usize * height as usize * bytes_per_sample],
                    vec![0u8; (width as usize / 2) * (height as usize / 2) * 2 * bytes_per_sample],
                ]
            }
            OutputFormat::Rgb | OutputFormat::Bgr => {
                vec![vec![
                    0u8;
                    width as usize * height as usize * 3 * bytes_per_sample
                ]]
            }
            OutputFormat::Rgba | OutputFormat::Bgra => {
                vec![vec![
                    0u8;
                    width as usize * height as usize * 4 * bytes_per_sample
                ]]
            }
        };

        Self {
            planes,
            width,
            height,
            format,
            bit_depth,
            timestamp: 0,
        }
    }

    /// Get width.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Get height.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Get format.
    #[must_use]
    pub const fn format(&self) -> OutputFormat {
        self.format
    }

    /// Get bit depth.
    #[must_use]
    pub const fn bit_depth(&self) -> u8 {
        self.bit_depth
    }

    /// Get timestamp.
    #[must_use]
    pub const fn timestamp(&self) -> i64 {
        self.timestamp
    }

    /// Set timestamp.
    pub fn set_timestamp(&mut self, timestamp: i64) {
        self.timestamp = timestamp;
    }

    /// Get plane data.
    #[must_use]
    pub fn plane(&self, index: usize) -> &[u8] {
        self.planes.get(index).map_or(&[], Vec::as_slice)
    }

    /// Get mutable plane data.
    pub fn plane_mut(&mut self, index: usize) -> &mut [u8] {
        self.planes
            .get_mut(index)
            .map_or(&mut [], Vec::as_mut_slice)
    }

    /// Get all planes.
    #[must_use]
    pub fn planes(&self) -> &[Vec<u8>] {
        &self.planes
    }

    /// Get total size in bytes.
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        self.planes.iter().map(Vec::len).sum()
    }
}

// =============================================================================
// Output Formatter
// =============================================================================

/// Output formatter for converting decoded frames.
#[derive(Debug)]
pub struct OutputFormatter {
    /// Current configuration.
    config: OutputConfig,
    /// Dither state.
    dither_state: u32,
}

impl Default for OutputFormatter {
    fn default() -> Self {
        Self::new()
    }
}

impl OutputFormatter {
    /// Create a new output formatter.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: OutputConfig::default(),
            dither_state: 0,
        }
    }

    /// Create with configuration.
    #[must_use]
    pub fn with_config(config: OutputConfig) -> Self {
        Self {
            config,
            dither_state: 0,
        }
    }

    /// Set configuration.
    pub fn set_config(&mut self, config: OutputConfig) {
        self.config = config;
    }

    /// Get current configuration.
    #[must_use]
    pub fn config(&self) -> &OutputConfig {
        &self.config
    }

    /// Format a frame to the configured output format.
    ///
    /// # Errors
    ///
    /// Returns error if formatting fails.
    pub fn format(&mut self, frame: &FrameBuffer) -> ReconstructResult<OutputBuffer> {
        let width = self.config.width.unwrap_or(frame.width());
        let height = self.config.height.unwrap_or(frame.height());

        let mut output =
            OutputBuffer::new(width, height, self.config.format, self.config.bit_depth);
        output.set_timestamp(frame.timestamp());

        match self.config.format {
            OutputFormat::YuvPlanar => self.format_yuv_planar(frame, &mut output)?,
            OutputFormat::YuvSemiPlanar => self.format_yuv_semi_planar(frame, &mut output)?,
            OutputFormat::Rgb => self.format_rgb(frame, &mut output, false, false)?,
            OutputFormat::Rgba => self.format_rgb(frame, &mut output, true, false)?,
            OutputFormat::Bgr => self.format_rgb(frame, &mut output, false, true)?,
            OutputFormat::Bgra => self.format_rgb(frame, &mut output, true, true)?,
        }

        Ok(output)
    }

    /// Format to YUV planar.
    fn format_yuv_planar(
        &mut self,
        frame: &FrameBuffer,
        output: &mut OutputBuffer,
    ) -> ReconstructResult<()> {
        // Copy Y plane
        self.copy_plane(frame.y_plane(), output.plane_mut(0));

        // Copy U plane
        if let Some(u) = frame.u_plane() {
            self.copy_plane(u, output.plane_mut(1));
        }

        // Copy V plane
        if let Some(v) = frame.v_plane() {
            self.copy_plane(v, output.plane_mut(2));
        }

        Ok(())
    }

    /// Format to YUV semi-planar (NV12).
    fn format_yuv_semi_planar(
        &mut self,
        frame: &FrameBuffer,
        output: &mut OutputBuffer,
    ) -> ReconstructResult<()> {
        // Copy Y plane
        self.copy_plane(frame.y_plane(), output.plane_mut(0));

        // Interleave UV
        if let (Some(u), Some(v)) = (frame.u_plane(), frame.v_plane()) {
            let uv_plane = output.plane_mut(1);
            let u_data = u.data();
            let v_data = v.data();

            let mut dst_idx = 0;
            for (u_val, v_val) in u_data.iter().zip(v_data.iter()) {
                let u_byte = self.convert_sample(*u_val, frame.bit_depth(), self.config.bit_depth);
                let v_byte = self.convert_sample(*v_val, frame.bit_depth(), self.config.bit_depth);

                if dst_idx + 1 < uv_plane.len() {
                    uv_plane[dst_idx] = u_byte;
                    uv_plane[dst_idx + 1] = v_byte;
                    dst_idx += 2;
                }
            }
        }

        Ok(())
    }

    /// Format to RGB.
    fn format_rgb(
        &mut self,
        frame: &FrameBuffer,
        output: &mut OutputBuffer,
        with_alpha: bool,
        bgr_order: bool,
    ) -> ReconstructResult<()> {
        let width = output.width() as usize;
        let height = output.height() as usize;
        let bpp = if with_alpha { 4 } else { 3 };
        let rgb_data = output.plane_mut(0);

        let y_plane = frame.y_plane();
        let u_plane = frame.u_plane();
        let v_plane = frame.v_plane();

        for y in 0..height {
            for x in 0..width {
                // Get YUV values
                let y_val = y_plane.get(x as u32, y as u32);
                let (u_val, v_val) = if let (Some(u), Some(v)) = (u_plane, v_plane) {
                    let cx = (x / 2) as u32;
                    let cy = (y / 2) as u32;
                    (u.get(cx, cy), v.get(cx, cy))
                } else {
                    (128, 128)
                };

                // Convert to RGB using BT.709 coefficients
                let (r, g, b) = yuv_to_rgb(y_val, u_val, v_val, frame.bit_depth());

                // Write to output
                let idx = (y * width + x) * bpp;
                if idx + bpp <= rgb_data.len() {
                    if bgr_order {
                        rgb_data[idx] = b;
                        rgb_data[idx + 1] = g;
                        rgb_data[idx + 2] = r;
                    } else {
                        rgb_data[idx] = r;
                        rgb_data[idx + 1] = g;
                        rgb_data[idx + 2] = b;
                    }
                    if with_alpha {
                        rgb_data[idx + 3] = 255;
                    }
                }
            }
        }

        Ok(())
    }

    /// Copy a plane with bit depth conversion.
    fn copy_plane(&mut self, src: &PlaneBuffer, dst: &mut [u8]) {
        let src_bd = src.bit_depth();
        let dst_bd = self.config.bit_depth;
        let src_data = src.data();

        if self.config.dither && dst_bd < src_bd {
            // Dithered conversion
            for (i, &sample) in src_data.iter().enumerate() {
                if i < dst.len() {
                    dst[i] = self.convert_sample_dithered(sample, src_bd, dst_bd);
                }
            }
        } else {
            // Direct conversion
            for (i, &sample) in src_data.iter().enumerate() {
                if i < dst.len() {
                    dst[i] = self.convert_sample(sample, src_bd, dst_bd);
                }
            }
        }
    }

    /// Convert a sample between bit depths.
    fn convert_sample(&self, sample: i16, src_bd: u8, dst_bd: u8) -> u8 {
        let src_max = (1i32 << src_bd) - 1;
        let dst_max = (1i32 << dst_bd) - 1;

        let clamped = (i32::from(sample)).clamp(0, src_max);

        if src_bd == dst_bd {
            clamped as u8
        } else if dst_bd < src_bd {
            // Scale down
            let shift = src_bd - dst_bd;
            (clamped >> shift) as u8
        } else {
            // Scale up
            let shift = dst_bd - src_bd;
            ((clamped << shift) | (clamped >> (src_bd - shift))).min(dst_max) as u8
        }
    }

    /// Convert a sample with dithering.
    fn convert_sample_dithered(&mut self, sample: i16, src_bd: u8, dst_bd: u8) -> u8 {
        let src_max = (1i32 << src_bd) - 1;

        let clamped = (i32::from(sample)).clamp(0, src_max);
        let shift = src_bd.saturating_sub(dst_bd);

        if shift == 0 {
            return clamped as u8;
        }

        // Simple ordered dither
        let dither = (self.dither_state & ((1 << shift) - 1)) as i32;
        self.dither_state = self
            .dither_state
            .wrapping_mul(1664525)
            .wrapping_add(1013904223);

        let result = (clamped + dither) >> shift;
        result.min(255) as u8
    }
}

/// Convert YUV to RGB using BT.709 coefficients.
fn yuv_to_rgb(y: i16, u: i16, v: i16, bd: u8) -> (u8, u8, u8) {
    let half = 1i32 << (bd - 1);

    let y_val = i32::from(y);
    let u_val = i32::from(u) - half;
    let v_val = i32::from(v) - half;

    // BT.709 coefficients (scaled by 256)
    let r = y_val + ((359 * v_val) >> 8);
    let g = y_val - ((88 * u_val + 183 * v_val) >> 8);
    let b = y_val + ((454 * u_val) >> 8);

    // Scale to 8-bit if needed
    let shift = bd.saturating_sub(8);

    (
        ((r >> shift).clamp(0, 255)) as u8,
        ((g >> shift).clamp(0, 255)) as u8,
        ((b >> shift).clamp(0, 255)) as u8,
    )
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reconstruct::ChromaSubsampling;

    #[test]
    fn test_output_format() {
        assert_eq!(OutputFormat::YuvPlanar.num_planes(), 3);
        assert_eq!(OutputFormat::YuvSemiPlanar.num_planes(), 2);
        assert_eq!(OutputFormat::Rgb.num_planes(), 1);
        assert_eq!(OutputFormat::Rgba.num_planes(), 1);

        assert_eq!(OutputFormat::Rgb.bytes_per_pixel(), 3);
        assert_eq!(OutputFormat::Rgba.bytes_per_pixel(), 4);

        assert!(OutputFormat::YuvPlanar.is_planar());
        assert!(!OutputFormat::Rgb.is_planar());
        assert!(OutputFormat::Rgb.is_rgb());
    }

    #[test]
    fn test_output_config() {
        let config = OutputConfig::new(OutputFormat::Rgb)
            .with_bit_depth(8)
            .with_full_range()
            .with_dither()
            .with_dimensions(1920, 1080);

        assert_eq!(config.format, OutputFormat::Rgb);
        assert_eq!(config.bit_depth, 8);
        assert!(config.full_range);
        assert!(config.dither);
        assert_eq!(config.width, Some(1920));
        assert_eq!(config.height, Some(1080));
    }

    #[test]
    fn test_output_buffer() {
        let buffer = OutputBuffer::new(1920, 1080, OutputFormat::YuvPlanar, 8);

        assert_eq!(buffer.width(), 1920);
        assert_eq!(buffer.height(), 1080);
        assert_eq!(buffer.format(), OutputFormat::YuvPlanar);
        assert_eq!(buffer.bit_depth(), 8);

        // Check plane sizes
        assert_eq!(buffer.plane(0).len(), 1920 * 1080);
        assert_eq!(buffer.plane(1).len(), 960 * 540);
        assert_eq!(buffer.plane(2).len(), 960 * 540);
    }

    #[test]
    fn test_output_buffer_rgb() {
        let buffer = OutputBuffer::new(64, 64, OutputFormat::Rgb, 8);
        assert_eq!(buffer.plane(0).len(), 64 * 64 * 3);
    }

    #[test]
    fn test_output_buffer_rgba() {
        let buffer = OutputBuffer::new(64, 64, OutputFormat::Rgba, 8);
        assert_eq!(buffer.plane(0).len(), 64 * 64 * 4);
    }

    #[test]
    fn test_output_formatter_creation() {
        let formatter = OutputFormatter::new();
        assert_eq!(formatter.config().format, OutputFormat::YuvPlanar);
    }

    #[test]
    fn test_output_formatter_with_config() {
        let config = OutputConfig::new(OutputFormat::Rgb);
        let formatter = OutputFormatter::with_config(config);
        assert_eq!(formatter.config().format, OutputFormat::Rgb);
    }

    #[test]
    fn test_output_formatter_format_yuv() {
        let frame = FrameBuffer::new(64, 64, 8, ChromaSubsampling::Cs420);
        let mut formatter = OutputFormatter::new();

        let output = formatter.format(&frame).expect("should succeed");
        assert_eq!(output.width(), 64);
        assert_eq!(output.height(), 64);
        assert_eq!(output.format(), OutputFormat::YuvPlanar);
    }

    #[test]
    fn test_output_formatter_format_rgb() {
        let frame = FrameBuffer::new(64, 64, 8, ChromaSubsampling::Cs420);
        let config = OutputConfig::new(OutputFormat::Rgb);
        let mut formatter = OutputFormatter::with_config(config);

        let output = formatter.format(&frame).expect("should succeed");
        assert_eq!(output.format(), OutputFormat::Rgb);
        assert_eq!(output.plane(0).len(), 64 * 64 * 3);
    }

    #[test]
    fn test_yuv_to_rgb() {
        // Black
        let (r, g, b) = yuv_to_rgb(0, 128, 128, 8);
        assert!(r < 10 && g < 10 && b < 10);

        // White
        let (r, g, b) = yuv_to_rgb(255, 128, 128, 8);
        assert!(r > 245 && g > 245 && b > 245);
    }

    #[test]
    fn test_convert_sample() {
        let formatter = OutputFormatter::new();

        // Same bit depth
        assert_eq!(formatter.convert_sample(128, 8, 8), 128);

        // Scale down
        assert_eq!(formatter.convert_sample(512, 10, 8), 128);

        // Scale down from 12-bit to 8-bit
        assert_eq!(formatter.convert_sample(2048, 12, 8), 128);
    }

    #[test]
    fn test_convert_sample_clamping() {
        let formatter = OutputFormatter::new();

        // Negative value should clamp to 0
        assert_eq!(formatter.convert_sample(-10, 8, 8), 0);

        // Over max should clamp
        assert_eq!(formatter.convert_sample(300, 8, 8), 255);
    }
}
