//! Colorspace conversion filter.
//!
//! This filter converts video frames between different color spaces (YUV/RGB)
//! and handles chroma subsampling conversions.

#![forbid(unsafe_code)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::similar_names)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::match_same_arms)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::unused_self)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::bool_to_int_with_if)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::map_unwrap_or)]
#![allow(clippy::no_effect_underscore_binding)]
#![allow(clippy::unreadable_literal)]
#![allow(dead_code)]

use crate::error::{GraphError, GraphResult};
use crate::frame::FilterFrame;
use crate::node::{Node, NodeId, NodeState, NodeType};
use crate::port::{InputPort, OutputPort, PortFormat, PortId, PortType, VideoPortFormat};
use oximedia_codec::{ColorInfo, MatrixCoefficients, Plane, VideoFrame};
use oximedia_core::PixelFormat;

/// Color matrix coefficients for YUV <-> RGB conversion.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColorMatrix {
    /// RGB to Y coefficients (kr, kg, kb).
    pub rgb_to_y: [f64; 3],
    /// Scale factor for Cb calculation.
    pub cb_scale: f64,
    /// Scale factor for Cr calculation.
    pub cr_scale: f64,
}

impl ColorMatrix {
    /// BT.601 (SD) color matrix.
    pub const BT601: Self = Self {
        rgb_to_y: [0.299, 0.587, 0.114],
        cb_scale: 1.772,
        cr_scale: 1.402,
    };

    /// BT.709 (HD) color matrix.
    pub const BT709: Self = Self {
        rgb_to_y: [0.2126, 0.7152, 0.0722],
        cb_scale: 1.8556,
        cr_scale: 1.5748,
    };

    /// BT.2020 (UHD/HDR) color matrix.
    pub const BT2020: Self = Self {
        rgb_to_y: [0.2627, 0.6780, 0.0593],
        cb_scale: 1.8814,
        cr_scale: 1.4746,
    };

    /// Get the appropriate matrix for the given matrix coefficients.
    #[must_use]
    pub fn from_matrix_coefficients(coeffs: MatrixCoefficients) -> Self {
        match coeffs {
            MatrixCoefficients::Bt709 => Self::BT709,
            MatrixCoefficients::Bt470Bg | MatrixCoefficients::Smpte170M => Self::BT601,
            MatrixCoefficients::Bt2020Ncl | MatrixCoefficients::Bt2020Cl => Self::BT2020,
            _ => Self::BT709, // Default to BT.709
        }
    }

    /// Convert RGB to YUV.
    #[must_use]
    pub fn rgb_to_yuv(&self, r: u8, g: u8, b: u8, full_range: bool) -> (u8, u8, u8) {
        let r = r as f64;
        let g = g as f64;
        let b = b as f64;

        let y = self.rgb_to_y[0] * r + self.rgb_to_y[1] * g + self.rgb_to_y[2] * b;
        let cb = (b - y) / self.cb_scale + 128.0;
        let cr = (r - y) / self.cr_scale + 128.0;

        let (y_min, y_max) = if full_range {
            (0.0, 255.0)
        } else {
            (16.0, 235.0)
        };
        let (c_min, c_max) = if full_range {
            (0.0, 255.0)
        } else {
            (16.0, 240.0)
        };

        let y = if full_range {
            y
        } else {
            y * (235.0 - 16.0) / 255.0 + 16.0
        };

        (
            y.clamp(y_min, y_max).round() as u8,
            cb.clamp(c_min, c_max).round() as u8,
            cr.clamp(c_min, c_max).round() as u8,
        )
    }

    /// Convert YUV to RGB.
    #[must_use]
    pub fn yuv_to_rgb(&self, y: u8, u: u8, v: u8, full_range: bool) -> (u8, u8, u8) {
        let y = if full_range {
            y as f64
        } else {
            ((y as f64) - 16.0) * 255.0 / (235.0 - 16.0)
        };

        let cb = if full_range {
            (u as f64) - 128.0
        } else {
            ((u as f64) - 128.0) * 255.0 / (240.0 - 16.0)
        };

        let cr = if full_range {
            (v as f64) - 128.0
        } else {
            ((v as f64) - 128.0) * 255.0 / (240.0 - 16.0)
        };

        let r = y + self.cr_scale * cr;
        let g = y
            - (self.rgb_to_y[0] / self.rgb_to_y[1]) * self.cr_scale * cr
            - (self.rgb_to_y[2] / self.rgb_to_y[1]) * self.cb_scale * cb;
        let b = y + self.cb_scale * cb;

        (
            r.clamp(0.0, 255.0).round() as u8,
            g.clamp(0.0, 255.0).round() as u8,
            b.clamp(0.0, 255.0).round() as u8,
        )
    }
}

impl Default for ColorMatrix {
    fn default() -> Self {
        Self::BT709
    }
}

/// Chroma sampling format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ChromaFormat {
    /// 4:4:4 - Full chroma resolution.
    Yuv444,
    /// 4:2:2 - Half horizontal chroma resolution.
    Yuv422,
    /// 4:2:0 - Half horizontal and vertical chroma resolution.
    #[default]
    Yuv420,
}

impl ChromaFormat {
    /// Get the subsampling ratios (horizontal, vertical).
    #[must_use]
    pub fn subsampling(&self) -> (u32, u32) {
        match self {
            Self::Yuv444 => (1, 1),
            Self::Yuv422 => (2, 1),
            Self::Yuv420 => (2, 2),
        }
    }

    /// Get from pixel format.
    #[must_use]
    pub fn from_pixel_format(format: PixelFormat) -> Option<Self> {
        match format {
            PixelFormat::Yuv444p => Some(Self::Yuv444),
            PixelFormat::Yuv422p => Some(Self::Yuv422),
            PixelFormat::Yuv420p | PixelFormat::Yuv420p10le | PixelFormat::Yuv420p12le => {
                Some(Self::Yuv420)
            }
            _ => None,
        }
    }
}

/// Configuration for color conversion.
#[derive(Clone, Debug)]
pub struct ColorConvertConfig {
    /// Target pixel format.
    pub target_format: PixelFormat,
    /// Color matrix for conversions.
    pub matrix: ColorMatrix,
    /// Output full range (vs limited range).
    pub full_range: bool,
    /// Dithering for bit depth reduction.
    pub dither: bool,
}

impl ColorConvertConfig {
    /// Create a new configuration for converting to a target format.
    #[must_use]
    pub fn new(target_format: PixelFormat) -> Self {
        Self {
            target_format,
            matrix: ColorMatrix::default(),
            full_range: false,
            dither: false,
        }
    }

    /// Set the color matrix.
    #[must_use]
    pub fn with_matrix(mut self, matrix: ColorMatrix) -> Self {
        self.matrix = matrix;
        self
    }

    /// Set full range output.
    #[must_use]
    pub fn with_full_range(mut self, full_range: bool) -> Self {
        self.full_range = full_range;
        self
    }

    /// Enable dithering.
    #[must_use]
    pub fn with_dither(mut self, dither: bool) -> Self {
        self.dither = dither;
        self
    }
}

/// Color conversion filter.
///
/// Converts between YUV and RGB color spaces, handles chroma subsampling,
/// and manages color range conversions.
///
/// # Example
///
/// ```ignore
/// use oximedia_graph::filters::video::{ColorConvertFilter, ColorConvertConfig, ColorMatrix};
/// use oximedia_graph::node::NodeId;
/// use oximedia_core::PixelFormat;
///
/// let config = ColorConvertConfig::new(PixelFormat::Rgb24)
///     .with_matrix(ColorMatrix::BT709)
///     .with_full_range(true);
///
/// let filter = ColorConvertFilter::new(NodeId(0), "yuv_to_rgb", config);
/// ```
pub struct ColorConvertFilter {
    id: NodeId,
    name: String,
    state: NodeState,
    inputs: Vec<InputPort>,
    outputs: Vec<OutputPort>,
    config: ColorConvertConfig,
}

impl ColorConvertFilter {
    /// Create a new color conversion filter.
    #[must_use]
    pub fn new(id: NodeId, name: impl Into<String>, config: ColorConvertConfig) -> Self {
        let output_format = PortFormat::Video(VideoPortFormat::new(config.target_format));

        Self {
            id,
            name: name.into(),
            state: NodeState::Idle,
            inputs: vec![InputPort::new(PortId(0), "input", PortType::Video)
                .with_format(PortFormat::Video(VideoPortFormat::any()))],
            outputs: vec![
                OutputPort::new(PortId(0), "output", PortType::Video).with_format(output_format)
            ],
            config,
        }
    }

    /// Get the current configuration.
    #[must_use]
    pub fn config(&self) -> &ColorConvertConfig {
        &self.config
    }

    /// Convert YUV to RGB.
    fn yuv_to_rgb(&self, input: &VideoFrame) -> VideoFrame {
        let width = input.width as usize;
        let height = input.height as usize;

        let mut output = VideoFrame::new(self.config.target_format, input.width, input.height);
        output.timestamp = input.timestamp;
        output.frame_type = input.frame_type;
        output.color_info = ColorInfo {
            full_range: self.config.full_range,
            ..input.color_info
        };

        let y_plane = input.planes.first();
        let u_plane = input.planes.get(1);
        let v_plane = input.planes.get(2);

        let (h_sub, v_sub) = input.format.chroma_subsampling();
        let bpp = if self.config.target_format == PixelFormat::Rgba32 {
            4
        } else {
            3
        };

        let mut rgb_data = vec![0u8; width * height * bpp];

        for y in 0..height {
            for x in 0..width {
                let y_val = y_plane
                    .map(|p| p.row(y).get(x).copied().unwrap_or(16))
                    .unwrap_or(16);

                let chroma_x = x / h_sub as usize;
                let chroma_y = y / v_sub as usize;

                let u_val = u_plane
                    .map(|p| p.row(chroma_y).get(chroma_x).copied().unwrap_or(128))
                    .unwrap_or(128);
                let v_val = v_plane
                    .map(|p| p.row(chroma_y).get(chroma_x).copied().unwrap_or(128))
                    .unwrap_or(128);

                let (r, g, b) =
                    self.config
                        .matrix
                        .yuv_to_rgb(y_val, u_val, v_val, input.color_info.full_range);

                let offset = (y * width + x) * bpp;
                rgb_data[offset] = r;
                rgb_data[offset + 1] = g;
                rgb_data[offset + 2] = b;
                if bpp == 4 {
                    rgb_data[offset + 3] = 255; // Alpha
                }
            }
        }

        output.planes.push(Plane::new(rgb_data, width * bpp));
        output
    }

    /// Convert RGB to YUV.
    fn rgb_to_yuv(&self, input: &VideoFrame) -> VideoFrame {
        let width = input.width as usize;
        let height = input.height as usize;

        let mut output = VideoFrame::new(self.config.target_format, input.width, input.height);
        output.timestamp = input.timestamp;
        output.frame_type = input.frame_type;
        output.color_info = ColorInfo {
            full_range: self.config.full_range,
            ..input.color_info
        };

        let src_plane = input.planes.first();
        let src_bpp = if input.format == PixelFormat::Rgba32 {
            4
        } else {
            3
        };

        let (h_sub, v_sub) = self.config.target_format.chroma_subsampling();
        let chroma_width = width / h_sub as usize;
        let chroma_height = height / v_sub as usize;

        let mut y_data = vec![0u8; width * height];
        let mut u_data = vec![0u8; chroma_width * chroma_height];
        let mut v_data = vec![0u8; chroma_width * chroma_height];

        // Convert all pixels
        for y in 0..height {
            for x in 0..width {
                let (r, g, b) = if let Some(plane) = src_plane {
                    let row = plane.row(y);
                    let offset = x * src_bpp;
                    (
                        row.get(offset).copied().unwrap_or(0),
                        row.get(offset + 1).copied().unwrap_or(0),
                        row.get(offset + 2).copied().unwrap_or(0),
                    )
                } else {
                    (0, 0, 0)
                };

                let (y_val, u_val, v_val) =
                    self.config
                        .matrix
                        .rgb_to_yuv(r, g, b, self.config.full_range);

                y_data[y * width + x] = y_val;

                // Accumulate chroma for subsampling
                if x % h_sub as usize == 0 && y % v_sub as usize == 0 {
                    let chroma_x = x / h_sub as usize;
                    let chroma_y = y / v_sub as usize;
                    u_data[chroma_y * chroma_width + chroma_x] = u_val;
                    v_data[chroma_y * chroma_width + chroma_x] = v_val;
                }
            }
        }

        output.planes.push(Plane::new(y_data, width));
        output.planes.push(Plane::new(u_data, chroma_width));
        output.planes.push(Plane::new(v_data, chroma_width));

        output
    }

    /// Convert between YUV formats (chroma resampling).
    fn convert_yuv_format(&self, input: &VideoFrame) -> VideoFrame {
        let width = input.width as usize;
        let height = input.height as usize;

        let mut output = VideoFrame::new(self.config.target_format, input.width, input.height);
        output.timestamp = input.timestamp;
        output.frame_type = input.frame_type;
        output.color_info = input.color_info;

        // Copy Y plane unchanged
        if let Some(y_plane) = input.planes.first() {
            output.planes.push(y_plane.clone());
        }

        let src_chroma = ChromaFormat::from_pixel_format(input.format);
        let dst_chroma = ChromaFormat::from_pixel_format(self.config.target_format);

        if src_chroma == dst_chroma {
            // Same chroma format, just copy
            for plane in input.planes.iter().skip(1) {
                output.planes.push(plane.clone());
            }
        } else {
            // Resample chroma
            let (src_h, src_v) = src_chroma.map(|c| c.subsampling()).unwrap_or((1, 1));
            let (dst_h, dst_v) = dst_chroma.map(|c| c.subsampling()).unwrap_or((1, 1));

            let dst_chroma_width = width / dst_h as usize;
            let _dst_chroma_height = height / dst_v as usize;

            for plane_idx in 1..=2 {
                if let Some(src_plane) = input.planes.get(plane_idx) {
                    let resampled =
                        resample_chroma(src_plane, width, height, src_h, src_v, dst_h, dst_v);
                    output.planes.push(Plane::new(resampled, dst_chroma_width));
                }
            }
        }

        output
    }

    /// Convert a frame to the target format.
    fn convert_frame(&self, input: &VideoFrame) -> GraphResult<VideoFrame> {
        let src_is_yuv = input.format.is_yuv();
        let dst_is_yuv = self.config.target_format.is_yuv();

        let output = if src_is_yuv && !dst_is_yuv {
            // YUV -> RGB
            self.yuv_to_rgb(input)
        } else if !src_is_yuv && dst_is_yuv {
            // RGB -> YUV
            self.rgb_to_yuv(input)
        } else if src_is_yuv && dst_is_yuv {
            // YUV -> YUV (chroma resampling)
            self.convert_yuv_format(input)
        } else {
            // RGB -> RGB (just copy or format change)
            let mut output = VideoFrame::new(self.config.target_format, input.width, input.height);
            output.timestamp = input.timestamp;
            output.frame_type = input.frame_type;
            output.color_info = input.color_info;
            if let Some(plane) = input.planes.first() {
                output.planes.push(plane.clone());
            }
            output
        };

        Ok(output)
    }
}

/// Resample chroma plane.
fn resample_chroma(
    src: &Plane,
    width: usize,
    height: usize,
    src_h_sub: u32,
    src_v_sub: u32,
    dst_h_sub: u32,
    dst_v_sub: u32,
) -> Vec<u8> {
    let dst_width = width / dst_h_sub as usize;
    let dst_height = height / dst_v_sub as usize;
    let src_width = width / src_h_sub as usize;

    let mut dst = vec![128u8; dst_width * dst_height];

    for dy in 0..dst_height {
        for dx in 0..dst_width {
            // Map destination position to source position
            let sx = (dx * dst_h_sub as usize) / src_h_sub as usize;
            let sy = (dy * dst_v_sub as usize) / src_v_sub as usize;

            let sx = sx.min(src_width.saturating_sub(1));
            let src_row = src.row(sy);

            dst[dy * dst_width + dx] = src_row.get(sx).copied().unwrap_or(128);
        }
    }

    dst
}

impl Node for ColorConvertFilter {
    fn id(&self) -> NodeId {
        self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn node_type(&self) -> NodeType {
        NodeType::Filter
    }

    fn state(&self) -> NodeState {
        self.state
    }

    fn set_state(&mut self, state: NodeState) -> GraphResult<()> {
        if !self.state.can_transition_to(state) {
            return Err(GraphError::InvalidStateTransition {
                node: self.id,
                from: self.state.to_string(),
                to: state.to_string(),
            });
        }
        self.state = state;
        Ok(())
    }

    fn inputs(&self) -> &[InputPort] {
        &self.inputs
    }

    fn outputs(&self) -> &[OutputPort] {
        &self.outputs
    }

    fn process(&mut self, input: Option<FilterFrame>) -> GraphResult<Option<FilterFrame>> {
        match input {
            Some(FilterFrame::Video(frame)) => {
                if frame.format == self.config.target_format {
                    // No conversion needed
                    Ok(Some(FilterFrame::Video(frame)))
                } else {
                    let converted = self.convert_frame(&frame)?;
                    Ok(Some(FilterFrame::Video(converted)))
                }
            }
            Some(_) => Err(GraphError::PortTypeMismatch {
                expected: "Video".to_string(),
                actual: "Audio".to_string(),
            }),
            None => Ok(None),
        }
    }
}

/// Upsample chroma from 4:2:0 to 4:4:4.
#[must_use]
pub fn upsample_420_to_444(
    u_plane: &Plane,
    v_plane: &Plane,
    width: u32,
    height: u32,
) -> (Vec<u8>, Vec<u8>) {
    let full_size = (width * height) as usize;
    let mut u_out = vec![128u8; full_size];
    let mut v_out = vec![128u8; full_size];

    for y in 0..height as usize {
        for x in 0..width as usize {
            let src_x = x / 2;
            let src_y = y / 2;

            let u_row = u_plane.row(src_y);
            let v_row = v_plane.row(src_y);

            u_out[y * width as usize + x] = u_row.get(src_x).copied().unwrap_or(128);
            v_out[y * width as usize + x] = v_row.get(src_x).copied().unwrap_or(128);
        }
    }

    (u_out, v_out)
}

/// Downsample chroma from 4:4:4 to 4:2:0.
#[must_use]
pub fn downsample_444_to_420(
    u_plane: &Plane,
    v_plane: &Plane,
    width: u32,
    height: u32,
) -> (Vec<u8>, Vec<u8>) {
    let chroma_width = width / 2;
    let chroma_height = height / 2;
    let chroma_size = (chroma_width * chroma_height) as usize;

    let mut u_out = vec![128u8; chroma_size];
    let mut v_out = vec![128u8; chroma_size];

    for cy in 0..chroma_height as usize {
        for cx in 0..chroma_width as usize {
            // Average 2x2 block
            let mut u_sum = 0u32;
            let mut v_sum = 0u32;

            for dy in 0..2 {
                for dx in 0..2 {
                    let sx = cx * 2 + dx;
                    let sy = cy * 2 + dy;

                    let u_row = u_plane.row(sy);
                    let v_row = v_plane.row(sy);

                    u_sum += u_row.get(sx).copied().unwrap_or(128) as u32;
                    v_sum += v_row.get(sx).copied().unwrap_or(128) as u32;
                }
            }

            u_out[cy * chroma_width as usize + cx] = (u_sum / 4) as u8;
            v_out[cy * chroma_width as usize + cx] = (v_sum / 4) as u8;
        }
    }

    (u_out, v_out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_yuv_frame(width: u32, height: u32) -> VideoFrame {
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, width, height);
        frame.allocate();

        // Fill Y plane with gradient
        if let Some(plane) = frame.planes.get_mut(0) {
            let mut data = vec![0u8; width as usize * height as usize];
            for y in 0..height as usize {
                for x in 0..width as usize {
                    data[y * width as usize + x] = ((x + y) % 256) as u8;
                }
            }
            *plane = Plane::new(data, width as usize);
        }

        // Fill U and V planes with neutral gray
        let chroma_width = width / 2;
        let chroma_height = height / 2;
        let chroma_size = (chroma_width * chroma_height) as usize;

        if let Some(plane) = frame.planes.get_mut(1) {
            *plane = Plane::new(vec![128u8; chroma_size], chroma_width as usize);
        }
        if let Some(plane) = frame.planes.get_mut(2) {
            *plane = Plane::new(vec![128u8; chroma_size], chroma_width as usize);
        }

        frame
    }

    fn create_test_rgb_frame(width: u32, height: u32) -> VideoFrame {
        let mut frame = VideoFrame::new(PixelFormat::Rgb24, width, height);

        let mut data = vec![0u8; (width * height * 3) as usize];
        for y in 0..height as usize {
            for x in 0..width as usize {
                let offset = (y * width as usize + x) * 3;
                data[offset] = (x % 256) as u8; // R
                data[offset + 1] = (y % 256) as u8; // G
                data[offset + 2] = 128; // B
            }
        }

        frame.planes.push(Plane::new(data, (width * 3) as usize));
        frame
    }

    #[test]
    fn test_color_matrix_bt601() {
        let matrix = ColorMatrix::BT601;
        assert!((matrix.rgb_to_y[0] - 0.299).abs() < 0.001);
        assert!((matrix.rgb_to_y[1] - 0.587).abs() < 0.001);
        assert!((matrix.rgb_to_y[2] - 0.114).abs() < 0.001);
    }

    #[test]
    fn test_color_matrix_bt709() {
        let matrix = ColorMatrix::BT709;
        assert!((matrix.rgb_to_y[0] - 0.2126).abs() < 0.001);
    }

    #[test]
    fn test_rgb_to_yuv_conversion() {
        let matrix = ColorMatrix::BT601;

        // White
        let (y, u, v) = matrix.rgb_to_yuv(255, 255, 255, true);
        assert!(y > 250);
        assert!((u as i32 - 128).abs() < 5);
        assert!((v as i32 - 128).abs() < 5);

        // Black
        let (y, u, v) = matrix.rgb_to_yuv(0, 0, 0, true);
        assert!(y < 5);
        assert!((u as i32 - 128).abs() < 5);
        assert!((v as i32 - 128).abs() < 5);
    }

    #[test]
    fn test_yuv_to_rgb_conversion() {
        let matrix = ColorMatrix::BT601;

        // Test roundtrip
        let (y, u, v) = matrix.rgb_to_yuv(128, 64, 192, true);
        let (r, g, b) = matrix.yuv_to_rgb(y, u, v, true);

        assert!((r as i32 - 128).abs() < 5);
        assert!((g as i32 - 64).abs() < 5);
        assert!((b as i32 - 192).abs() < 5);
    }

    #[test]
    fn test_chroma_format_subsampling() {
        assert_eq!(ChromaFormat::Yuv444.subsampling(), (1, 1));
        assert_eq!(ChromaFormat::Yuv422.subsampling(), (2, 1));
        assert_eq!(ChromaFormat::Yuv420.subsampling(), (2, 2));
    }

    #[test]
    fn test_chroma_format_from_pixel_format() {
        assert_eq!(
            ChromaFormat::from_pixel_format(PixelFormat::Yuv420p),
            Some(ChromaFormat::Yuv420)
        );
        assert_eq!(
            ChromaFormat::from_pixel_format(PixelFormat::Yuv422p),
            Some(ChromaFormat::Yuv422)
        );
        assert_eq!(ChromaFormat::from_pixel_format(PixelFormat::Rgb24), None);
    }

    #[test]
    fn test_color_convert_config() {
        let config = ColorConvertConfig::new(PixelFormat::Rgb24)
            .with_matrix(ColorMatrix::BT709)
            .with_full_range(true)
            .with_dither(true);

        assert_eq!(config.target_format, PixelFormat::Rgb24);
        assert!(config.full_range);
        assert!(config.dither);
    }

    #[test]
    fn test_color_convert_filter_creation() {
        let config = ColorConvertConfig::new(PixelFormat::Rgb24);
        let filter = ColorConvertFilter::new(NodeId(0), "convert", config);

        assert_eq!(filter.id(), NodeId(0));
        assert_eq!(filter.name(), "convert");
        assert_eq!(filter.node_type(), NodeType::Filter);
    }

    #[test]
    fn test_yuv_to_rgb_filter() {
        let config = ColorConvertConfig::new(PixelFormat::Rgb24);
        let mut filter = ColorConvertFilter::new(NodeId(0), "convert", config);

        let input = create_test_yuv_frame(64, 48);
        let result = filter
            .process(Some(FilterFrame::Video(input)))
            .expect("operation should succeed")
            .expect("operation should succeed");

        if let FilterFrame::Video(frame) = result {
            assert_eq!(frame.format, PixelFormat::Rgb24);
            assert_eq!(frame.width, 64);
            assert_eq!(frame.height, 48);
            assert_eq!(frame.planes.len(), 1);
        } else {
            panic!("Expected video frame");
        }
    }

    #[test]
    fn test_rgb_to_yuv_filter() {
        let config = ColorConvertConfig::new(PixelFormat::Yuv420p);
        let mut filter = ColorConvertFilter::new(NodeId(0), "convert", config);

        let input = create_test_rgb_frame(64, 48);
        let result = filter
            .process(Some(FilterFrame::Video(input)))
            .expect("operation should succeed")
            .expect("operation should succeed");

        if let FilterFrame::Video(frame) = result {
            assert_eq!(frame.format, PixelFormat::Yuv420p);
            assert_eq!(frame.planes.len(), 3);
        } else {
            panic!("Expected video frame");
        }
    }

    #[test]
    fn test_same_format_passthrough() {
        let config = ColorConvertConfig::new(PixelFormat::Yuv420p);
        let mut filter = ColorConvertFilter::new(NodeId(0), "convert", config);

        let input = create_test_yuv_frame(64, 48);
        let result = filter
            .process(Some(FilterFrame::Video(input)))
            .expect("operation should succeed")
            .expect("operation should succeed");

        if let FilterFrame::Video(frame) = result {
            assert_eq!(frame.format, PixelFormat::Yuv420p);
        } else {
            panic!("Expected video frame");
        }
    }

    #[test]
    fn test_upsample_420_to_444() {
        let u_data = vec![100u8; 32 * 24];
        let v_data = vec![150u8; 32 * 24];
        let u_plane = Plane::new(u_data, 32);
        let v_plane = Plane::new(v_data, 32);

        let (u_out, v_out) = upsample_420_to_444(&u_plane, &v_plane, 64, 48);

        assert_eq!(u_out.len(), 64 * 48);
        assert_eq!(v_out.len(), 64 * 48);
        assert_eq!(u_out[0], 100);
        assert_eq!(v_out[0], 150);
    }

    #[test]
    fn test_downsample_444_to_420() {
        let u_data = vec![100u8; 64 * 48];
        let v_data = vec![150u8; 64 * 48];
        let u_plane = Plane::new(u_data, 64);
        let v_plane = Plane::new(v_data, 64);

        let (u_out, v_out) = downsample_444_to_420(&u_plane, &v_plane, 64, 48);

        assert_eq!(u_out.len(), 32 * 24);
        assert_eq!(v_out.len(), 32 * 24);
        assert_eq!(u_out[0], 100);
        assert_eq!(v_out[0], 150);
    }

    #[test]
    fn test_node_state_transitions() {
        let config = ColorConvertConfig::new(PixelFormat::Rgb24);
        let mut filter = ColorConvertFilter::new(NodeId(0), "convert", config);

        assert_eq!(filter.state(), NodeState::Idle);
        filter
            .set_state(NodeState::Processing)
            .expect("set_state should succeed");
        assert_eq!(filter.state(), NodeState::Processing);
    }

    #[test]
    fn test_process_none_input() {
        let config = ColorConvertConfig::new(PixelFormat::Rgb24);
        let mut filter = ColorConvertFilter::new(NodeId(0), "convert", config);

        let result = filter.process(None).expect("process should succeed");
        assert!(result.is_none());
    }

    #[test]
    fn test_matrix_from_coefficients() {
        let matrix = ColorMatrix::from_matrix_coefficients(MatrixCoefficients::Bt709);
        assert!((matrix.rgb_to_y[0] - 0.2126).abs() < 0.001);

        let matrix = ColorMatrix::from_matrix_coefficients(MatrixCoefficients::Smpte170M);
        assert!((matrix.rgb_to_y[0] - 0.299).abs() < 0.001);

        let matrix = ColorMatrix::from_matrix_coefficients(MatrixCoefficients::Bt2020Ncl);
        assert!((matrix.rgb_to_y[0] - 0.2627).abs() < 0.001);
    }
}
