//! 3D LUT (Look-Up Table) color grading filter.
//!
//! This filter applies 3D LUTs for professional color grading and color correction.
//! It supports multiple file formats, interpolation methods, and advanced features
//! like LUT composition, HDR support, and shaper LUTs.
//!
//! # Features
//!
//! - **3D LUT Support:**
//!   - Trilinear interpolation
//!   - Tetrahedral interpolation
//!   - Configurable cube sizes (17³, 33³, 65³)
//!   - LUT composition (chain multiple LUTs)
//!
//! - **File Format Support:**
//!   - .cube format (Adobe/DaVinci Resolve)
//!   - .3dl format (Autodesk/Lustre)
//!   - .csp format (Cinespace)
//!   - CSV/text formats
//!
//! - **1D LUT Support:**
//!   - Linear interpolation
//!   - Per-channel 1D LUTs
//!   - Gamma curves
//!
//! - **Operations:**
//!   - LUT generation from formulas
//!   - Identity LUT creation
//!   - LUT inversion
//!   - LUT analysis and validation
//!
//! - **Advanced Features:**
//!   - HDR LUT support
//!   - Log/linear space handling
//!   - Shaper LUTs (1D pre-LUT)
//!   - GPU acceleration hooks
//!
//! # Example
//!
//! ```ignore
//! use oximedia_graph::filters::video::{Lut3dFilter, Lut3dConfig, LutInterpolation};
//! use oximedia_graph::node::NodeId;
//!
//! let config = Lut3dConfig::new()
//!     .with_file("colorgrade.cube")
//!     .with_interpolation(LutInterpolation::Tetrahedral);
//!
//! let filter = Lut3dFilter::new(NodeId(0), "lut3d", config);
//! ```

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
#![allow(clippy::too_many_arguments)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::struct_excessive_bools)]
#![allow(dead_code)]

mod cache;
mod io;
mod lut1d;
mod lut3d;
mod types;

use std::path::Path;

use crate::error::{GraphError, GraphResult};
use crate::frame::FilterFrame;
use crate::node::{Node, NodeId, NodeState, NodeType};
use crate::port::{InputPort, OutputPort, PortFormat, PortId, PortType, VideoPortFormat};
use oximedia_codec::{ColorInfo, Plane, VideoFrame};
use oximedia_core::PixelFormat;

// Public re-exports
pub use cache::{procedural, utils, CacheStats, LutAnalysis, LutCache};
pub use io::{
    export_3dl_file, export_csv_file, export_cube_file, load_lut_file, parse_3dl_file,
    parse_csv_file, parse_cube_file,
};
pub use lut1d::Lut1d;
pub use lut3d::Lut3d;
pub use types::{
    ColorChannel, GpuLutHints, LutBlendMode, LutColorSpace, LutFormat, LutInterpolation, LutSize,
    RgbColor,
};

/// Configuration for 3D LUT filter.
#[derive(Clone, Debug)]
pub struct Lut3dConfig {
    /// Path to LUT file (optional, can use programmatic LUT).
    pub lut_file: Option<String>,
    /// Interpolation method.
    pub interpolation: LutInterpolation,
    /// Color space for LUT processing.
    pub color_space: LutColorSpace,
    /// Optional shaper LUT (1D pre-LUT).
    pub shaper_lut: Option<Lut1d>,
    /// LUT strength/mix (0.0 = bypass, 1.0 = full strength).
    pub strength: f64,
    /// Target output format.
    pub target_format: PixelFormat,
}

impl Lut3dConfig {
    /// Create a new LUT configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            lut_file: None,
            interpolation: LutInterpolation::default(),
            color_space: LutColorSpace::default(),
            shaper_lut: None,
            strength: 1.0,
            target_format: PixelFormat::Rgb24,
        }
    }

    /// Set the LUT file path.
    #[must_use]
    pub fn with_file(mut self, path: impl Into<String>) -> Self {
        self.lut_file = Some(path.into());
        self
    }

    /// Set the interpolation method.
    #[must_use]
    pub fn with_interpolation(mut self, interpolation: LutInterpolation) -> Self {
        self.interpolation = interpolation;
        self
    }

    /// Set the color space.
    #[must_use]
    pub fn with_color_space(mut self, color_space: LutColorSpace) -> Self {
        self.color_space = color_space;
        self
    }

    /// Set the shaper LUT.
    #[must_use]
    pub fn with_shaper(mut self, shaper: Lut1d) -> Self {
        self.shaper_lut = Some(shaper);
        self
    }

    /// Set the LUT strength.
    #[must_use]
    pub fn with_strength(mut self, strength: f64) -> Self {
        self.strength = strength.clamp(0.0, 1.0);
        self
    }

    /// Set the target output format.
    #[must_use]
    pub fn with_target_format(mut self, format: PixelFormat) -> Self {
        self.target_format = format;
        self
    }
}

impl Default for Lut3dConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// 3D LUT filter node.
pub struct Lut3dFilter {
    id: NodeId,
    name: String,
    state: NodeState,
    inputs: Vec<InputPort>,
    outputs: Vec<OutputPort>,
    config: Lut3dConfig,
    lut: Lut3d,
}

impl Lut3dFilter {
    /// Create a new 3D LUT filter.
    pub fn new(id: NodeId, name: impl Into<String>, config: Lut3dConfig) -> Result<Self, String> {
        // Load or create LUT
        let lut = if let Some(ref path) = config.lut_file {
            load_lut_file(Path::new(path))?
        } else {
            // Create identity LUT if no file specified
            Lut3d::identity(33)
        };

        // Validate LUT
        let warnings = lut.validate();
        if !warnings.is_empty() {
            return Err(format!("LUT validation failed: {}", warnings.join(", ")));
        }

        let output_format = PortFormat::Video(VideoPortFormat::new(config.target_format));

        Ok(Self {
            id,
            name: name.into(),
            state: NodeState::Idle,
            inputs: vec![InputPort::new(PortId(0), "input", PortType::Video)
                .with_format(PortFormat::Video(VideoPortFormat::any()))],
            outputs: vec![
                OutputPort::new(PortId(0), "output", PortType::Video).with_format(output_format)
            ],
            config,
            lut,
        })
    }

    /// Create with an existing LUT.
    #[must_use]
    pub fn with_lut(id: NodeId, name: impl Into<String>, config: Lut3dConfig, lut: Lut3d) -> Self {
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
            lut,
        }
    }

    /// Get the current configuration.
    #[must_use]
    pub fn config(&self) -> &Lut3dConfig {
        &self.config
    }

    /// Get the current LUT.
    #[must_use]
    pub fn lut(&self) -> &Lut3d {
        &self.lut
    }

    /// Apply LUT to a single pixel.
    fn apply_lut_to_pixel(&self, r: u8, g: u8, b: u8) -> (u8, u8, u8) {
        // Convert to normalized RGB
        let mut color = RgbColor::from_u8(r, g, b);

        // Apply color space transform to linear/log
        color = RgbColor::new(
            self.config.color_space.inverse(color.r),
            self.config.color_space.inverse(color.g),
            self.config.color_space.inverse(color.b),
        );

        // Apply shaper LUT if present
        if let Some(ref shaper) = self.config.shaper_lut {
            color = shaper.apply(color);
        }

        // Apply 3D LUT with selected interpolation
        let lut_output = match self.config.interpolation {
            LutInterpolation::Nearest => self.lut.apply_nearest(color),
            LutInterpolation::Trilinear => self.lut.apply_trilinear(color),
            LutInterpolation::Tetrahedral => self.lut.apply_tetrahedral(color),
        };

        // Mix with original based on strength
        let mixed = if (self.config.strength - 1.0).abs() < 0.001 {
            lut_output
        } else {
            color.lerp(&lut_output, self.config.strength)
        };

        // Apply color space transform back
        let output = RgbColor::new(
            self.config.color_space.forward(mixed.r),
            self.config.color_space.forward(mixed.g),
            self.config.color_space.forward(mixed.b),
        );

        output.clamp().to_u8()
    }

    /// Convert YUV to RGB.
    fn yuv_to_rgb(&self, frame: &VideoFrame) -> Vec<u8> {
        let width = frame.width as usize;
        let height = frame.height as usize;

        let y_plane = frame.planes.first();
        let u_plane = frame.planes.get(1);
        let v_plane = frame.planes.get(2);

        let (h_sub, v_sub) = frame.format.chroma_subsampling();
        let mut rgb_data = vec![0u8; width * height * 3];

        for y in 0..height {
            for x in 0..width {
                let y_val = y_plane
                    .map(|p| p.row(y).get(x).copied().unwrap_or(16))
                    .unwrap_or(16) as f64;

                let chroma_x = x / h_sub as usize;
                let chroma_y = y / v_sub as usize;

                let u_val = u_plane
                    .map(|p| p.row(chroma_y).get(chroma_x).copied().unwrap_or(128))
                    .unwrap_or(128) as f64;
                let v_val = v_plane
                    .map(|p| p.row(chroma_y).get(chroma_x).copied().unwrap_or(128))
                    .unwrap_or(128) as f64;

                // YUV to RGB conversion (limited range)
                let y_norm = (y_val - 16.0) * 255.0 / 219.0;
                let cb = (u_val - 128.0) * 255.0 / 224.0;
                let cr = (v_val - 128.0) * 255.0 / 224.0;

                let r = y_norm + 1.5748 * cr;
                let g = y_norm - 0.1873 * cb - 0.4681 * cr;
                let b = y_norm + 1.8556 * cb;

                let offset = (y * width + x) * 3;
                rgb_data[offset] = r.clamp(0.0, 255.0) as u8;
                rgb_data[offset + 1] = g.clamp(0.0, 255.0) as u8;
                rgb_data[offset + 2] = b.clamp(0.0, 255.0) as u8;
            }
        }

        rgb_data
    }

    /// Convert RGB to YUV.
    fn rgb_to_yuv(
        &self,
        rgb_data: &[u8],
        width: usize,
        height: usize,
    ) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        // BT.709 matrix for RGB to YUV
        const KR: f64 = 0.2126;
        const KB: f64 = 0.0722;

        let mut y_data = vec![0u8; width * height];
        let chroma_width = width / 2;
        let chroma_height = height / 2;
        let mut u_data = vec![128u8; chroma_width * chroma_height];
        let mut v_data = vec![128u8; chroma_width * chroma_height];

        for y in 0..height {
            for x in 0..width {
                let offset = (y * width + x) * 3;
                let r = rgb_data[offset] as f64;
                let g = rgb_data[offset + 1] as f64;
                let b = rgb_data[offset + 2] as f64;

                // RGB to YUV conversion (limited range)
                let y_val = KR * r + (1.0 - KR - KB) * g + KB * b;
                let cb = (b - y_val) / 1.8556;
                let cr = (r - y_val) / 1.5748;

                let y_out = y_val * 219.0 / 255.0 + 16.0;
                let cb_out = cb * 224.0 / 255.0 + 128.0;
                let cr_out = cr * 224.0 / 255.0 + 128.0;

                y_data[y * width + x] = y_out.clamp(16.0, 235.0) as u8;

                // Subsample chroma (4:2:0)
                if x % 2 == 0 && y % 2 == 0 {
                    let chroma_x = x / 2;
                    let chroma_y = y / 2;
                    u_data[chroma_y * chroma_width + chroma_x] = cb_out.clamp(16.0, 240.0) as u8;
                    v_data[chroma_y * chroma_width + chroma_x] = cr_out.clamp(16.0, 240.0) as u8;
                }
            }
        }

        (y_data, u_data, v_data)
    }

    /// Process RGB frame.
    fn process_rgb(&self, input: &VideoFrame) -> GraphResult<VideoFrame> {
        let width = input.width as usize;
        let height = input.height as usize;

        let src_plane = input
            .planes
            .first()
            .ok_or_else(|| GraphError::ConfigurationError("Missing RGB plane".to_string()))?;

        let src_bpp = if input.format == PixelFormat::Rgba32 {
            4
        } else {
            3
        };

        let mut output_rgb = vec![0u8; width * height * 3];

        // Process each pixel
        for y in 0..height {
            for x in 0..width {
                let row = src_plane.row(y);
                let offset = x * src_bpp;

                let r = row.get(offset).copied().unwrap_or(0);
                let g = row.get(offset + 1).copied().unwrap_or(0);
                let b = row.get(offset + 2).copied().unwrap_or(0);

                let (r_out, g_out, b_out) = self.apply_lut_to_pixel(r, g, b);

                let out_offset = (y * width + x) * 3;
                output_rgb[out_offset] = r_out;
                output_rgb[out_offset + 1] = g_out;
                output_rgb[out_offset + 2] = b_out;
            }
        }

        // Convert to target format
        if self.config.target_format.is_yuv() {
            let (y_data, u_data, v_data) = self.rgb_to_yuv(&output_rgb, width, height);

            let mut output = VideoFrame::new(self.config.target_format, input.width, input.height);
            output.timestamp = input.timestamp;
            output.frame_type = input.frame_type;
            output.color_info = ColorInfo {
                full_range: false,
                ..input.color_info
            };

            let chroma_width = width / 2;
            output.planes.push(Plane::new(y_data, width));
            output.planes.push(Plane::new(u_data, chroma_width));
            output.planes.push(Plane::new(v_data, chroma_width));

            Ok(output)
        } else {
            let mut output = VideoFrame::new(self.config.target_format, input.width, input.height);
            output.timestamp = input.timestamp;
            output.frame_type = input.frame_type;
            output.color_info = input.color_info;
            output.planes.push(Plane::new(output_rgb, width * 3));

            Ok(output)
        }
    }

    /// Process YUV frame.
    fn process_yuv(&self, input: &VideoFrame) -> GraphResult<VideoFrame> {
        let width = input.width as usize;
        let height = input.height as usize;

        // Convert to RGB
        let rgb_data = self.yuv_to_rgb(input);

        // Process each pixel
        let mut output_rgb = vec![0u8; width * height * 3];

        for y in 0..height {
            for x in 0..width {
                let offset = (y * width + x) * 3;
                let r = rgb_data[offset];
                let g = rgb_data[offset + 1];
                let b = rgb_data[offset + 2];

                let (r_out, g_out, b_out) = self.apply_lut_to_pixel(r, g, b);

                output_rgb[offset] = r_out;
                output_rgb[offset + 1] = g_out;
                output_rgb[offset + 2] = b_out;
            }
        }

        // Convert to target format
        if self.config.target_format.is_yuv() {
            let (y_data, u_data, v_data) = self.rgb_to_yuv(&output_rgb, width, height);

            let mut output = VideoFrame::new(self.config.target_format, input.width, input.height);
            output.timestamp = input.timestamp;
            output.frame_type = input.frame_type;
            output.color_info = ColorInfo {
                full_range: false,
                ..input.color_info
            };

            let chroma_width = width / 2;
            output.planes.push(Plane::new(y_data, width));
            output.planes.push(Plane::new(u_data, chroma_width));
            output.planes.push(Plane::new(v_data, chroma_width));

            Ok(output)
        } else {
            let mut output = VideoFrame::new(self.config.target_format, input.width, input.height);
            output.timestamp = input.timestamp;
            output.frame_type = input.frame_type;
            output.color_info = input.color_info;
            output.planes.push(Plane::new(output_rgb, width * 3));

            Ok(output)
        }
    }

    /// Apply LUT to a frame.
    fn apply_lut(&self, input: &VideoFrame) -> GraphResult<VideoFrame> {
        if input.format.is_yuv() {
            self.process_yuv(input)
        } else {
            self.process_rgb(input)
        }
    }
}

impl Node for Lut3dFilter {
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
                let processed = self.apply_lut(&frame)?;
                Ok(Some(FilterFrame::Video(processed)))
            }
            Some(_) => Err(GraphError::PortTypeMismatch {
                expected: "Video".to_string(),
                actual: "Audio".to_string(),
            }),
            None => Ok(None),
        }
    }
}
