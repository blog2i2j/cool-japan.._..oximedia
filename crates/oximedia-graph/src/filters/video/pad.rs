//! Video padding filter.
//!
//! This filter adds padding (borders) around video frames, useful for
//! letterboxing, pillarboxing, and aspect ratio adjustment.

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
use oximedia_codec::{Plane, VideoFrame};

/// Color for padding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PadColor {
    /// Black (Y=0 or 16 for limited range, UV=128).
    Black,
    /// White (Y=255 or 235 for limited range, UV=128).
    White,
    /// Gray (Y=128, UV=128).
    Gray,
    /// Custom YUV color.
    YuvColor {
        /// Y (luma) component.
        y: u8,
        /// U (Cb) component.
        u: u8,
        /// V (Cr) component.
        v: u8,
    },
    /// Custom RGB color (will be converted to YUV internally).
    RgbColor {
        /// Red component.
        r: u8,
        /// Green component.
        g: u8,
        /// Blue component.
        b: u8,
    },
}

#[allow(clippy::derivable_impls)]
impl Default for PadColor {
    fn default() -> Self {
        Self::Black
    }
}

impl PadColor {
    /// Get the YUV values for this color.
    #[must_use]
    pub fn to_yuv(&self, full_range: bool) -> (u8, u8, u8) {
        match self {
            Self::Black => {
                if full_range {
                    (0, 128, 128)
                } else {
                    (16, 128, 128)
                }
            }
            Self::White => {
                if full_range {
                    (255, 128, 128)
                } else {
                    (235, 128, 128)
                }
            }
            Self::Gray => (128, 128, 128),
            Self::YuvColor { y, u, v } => (*y, *u, *v),
            Self::RgbColor { r, g, b } => rgb_to_yuv(*r, *g, *b),
        }
    }

    /// Create a custom YUV color.
    #[must_use]
    pub fn yuv(y: u8, u: u8, v: u8) -> Self {
        Self::YuvColor { y, u, v }
    }

    /// Create a custom RGB color.
    #[must_use]
    pub fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self::RgbColor { r, g, b }
    }
}

/// Convert RGB to YUV (BT.601 full range).
fn rgb_to_yuv(r: u8, g: u8, b: u8) -> (u8, u8, u8) {
    let r = r as f64;
    let g = g as f64;
    let b = b as f64;

    let y = 0.299 * r + 0.587 * g + 0.114 * b;
    let u = -0.169 * r - 0.331 * g + 0.500 * b + 128.0;
    let v = 0.500 * r - 0.419 * g - 0.081 * b + 128.0;

    (
        y.round().clamp(0.0, 255.0) as u8,
        u.round().clamp(0.0, 255.0) as u8,
        v.round().clamp(0.0, 255.0) as u8,
    )
}

/// Configuration for the pad filter.
#[derive(Clone, Debug)]
pub struct PadConfig {
    /// Padding on the left side (pixels).
    pub left: u32,
    /// Padding on the top side (pixels).
    pub top: u32,
    /// Padding on the right side (pixels).
    pub right: u32,
    /// Padding on the bottom side (pixels).
    pub bottom: u32,
    /// Color for padding.
    pub color: PadColor,
    /// Target width (if auto-padding is enabled).
    pub target_width: Option<u32>,
    /// Target height (if auto-padding is enabled).
    pub target_height: Option<u32>,
    /// Target aspect ratio for auto-padding.
    pub target_aspect: Option<f64>,
}

impl PadConfig {
    /// Create a new pad configuration with explicit padding values.
    #[must_use]
    pub fn new(left: u32, top: u32, right: u32, bottom: u32) -> Self {
        Self {
            left,
            top,
            right,
            bottom,
            color: PadColor::default(),
            target_width: None,
            target_height: None,
            target_aspect: None,
        }
    }

    /// Create padding to reach a target size.
    #[must_use]
    pub fn to_size(target_width: u32, target_height: u32) -> Self {
        Self {
            left: 0,
            top: 0,
            right: 0,
            bottom: 0,
            color: PadColor::default(),
            target_width: Some(target_width),
            target_height: Some(target_height),
            target_aspect: None,
        }
    }

    /// Create padding for a target aspect ratio (letterbox/pillarbox).
    #[must_use]
    pub fn for_aspect(aspect: f64) -> Self {
        Self {
            left: 0,
            top: 0,
            right: 0,
            bottom: 0,
            color: PadColor::default(),
            target_width: None,
            target_height: None,
            target_aspect: Some(aspect),
        }
    }

    /// Set the padding color.
    #[must_use]
    pub fn with_color(mut self, color: PadColor) -> Self {
        self.color = color;
        self
    }

    /// Calculate the actual padding values for given source dimensions.
    #[must_use]
    pub fn calculate_padding(&self, src_width: u32, src_height: u32) -> PadValues {
        if let (Some(target_w), Some(target_h)) = (self.target_width, self.target_height) {
            // Pad to exact target size (centered)
            let h_pad = target_w.saturating_sub(src_width);
            let v_pad = target_h.saturating_sub(src_height);

            PadValues {
                left: h_pad / 2,
                top: v_pad / 2,
                right: h_pad - (h_pad / 2),
                bottom: v_pad - (v_pad / 2),
            }
        } else if let Some(target_aspect) = self.target_aspect {
            // Pad to match aspect ratio
            calculate_aspect_padding(src_width, src_height, target_aspect)
        } else {
            // Use explicit padding values
            PadValues {
                left: self.left,
                top: self.top,
                right: self.right,
                bottom: self.bottom,
            }
        }
    }

    /// Get the output dimensions for given source dimensions.
    #[must_use]
    pub fn output_dimensions(&self, src_width: u32, src_height: u32) -> (u32, u32) {
        let pad = self.calculate_padding(src_width, src_height);
        (
            src_width + pad.left + pad.right,
            src_height + pad.top + pad.bottom,
        )
    }
}

/// Calculated padding values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PadValues {
    /// Left padding.
    pub left: u32,
    /// Top padding.
    pub top: u32,
    /// Right padding.
    pub right: u32,
    /// Bottom padding.
    pub bottom: u32,
}

impl PadValues {
    /// Get total horizontal padding.
    #[must_use]
    pub fn horizontal(&self) -> u32 {
        self.left + self.right
    }

    /// Get total vertical padding.
    #[must_use]
    pub fn vertical(&self) -> u32 {
        self.top + self.bottom
    }

    /// Check if any padding is applied.
    #[must_use]
    pub fn has_padding(&self) -> bool {
        self.left > 0 || self.top > 0 || self.right > 0 || self.bottom > 0
    }

    /// Scale padding values for chroma planes.
    #[must_use]
    pub fn scale_for_chroma(&self, h_ratio: u32, v_ratio: u32) -> Self {
        Self {
            left: self.left / h_ratio,
            top: self.top / v_ratio,
            right: self.right / h_ratio,
            bottom: self.bottom / v_ratio,
        }
    }
}

/// Calculate padding to achieve target aspect ratio.
fn calculate_aspect_padding(src_width: u32, src_height: u32, target_aspect: f64) -> PadValues {
    let src_aspect = src_width as f64 / src_height as f64;

    if src_aspect > target_aspect {
        // Source is wider than target, add vertical padding (letterbox)
        let target_height = (src_width as f64 / target_aspect).round() as u32;
        let v_pad = target_height.saturating_sub(src_height);
        PadValues {
            left: 0,
            top: v_pad / 2,
            right: 0,
            bottom: v_pad - (v_pad / 2),
        }
    } else {
        // Source is taller than target, add horizontal padding (pillarbox)
        let target_width = (src_height as f64 * target_aspect).round() as u32;
        let h_pad = target_width.saturating_sub(src_width);
        PadValues {
            left: h_pad / 2,
            top: 0,
            right: h_pad - (h_pad / 2),
            bottom: 0,
        }
    }
}

/// Video padding filter.
///
/// Adds padding around video frames for letterboxing, pillarboxing,
/// or reaching a target resolution.
///
/// # Example
///
/// ```ignore
/// use oximedia_graph::filters::video::{PadFilter, PadConfig, PadColor};
/// use oximedia_graph::node::NodeId;
///
/// // Create a letterbox for 16:9 aspect ratio
/// let config = PadConfig::for_aspect(16.0 / 9.0)
///     .with_color(PadColor::Black);
///
/// let filter = PadFilter::new(NodeId(0), "letterbox", config);
/// ```
pub struct PadFilter {
    id: NodeId,
    name: String,
    state: NodeState,
    inputs: Vec<InputPort>,
    outputs: Vec<OutputPort>,
    config: PadConfig,
}

impl PadFilter {
    /// Create a new pad filter.
    #[must_use]
    pub fn new(id: NodeId, name: impl Into<String>, config: PadConfig) -> Self {
        Self {
            id,
            name: name.into(),
            state: NodeState::Idle,
            inputs: vec![InputPort::new(PortId(0), "input", PortType::Video)
                .with_format(PortFormat::Video(VideoPortFormat::any()))],
            outputs: vec![OutputPort::new(PortId(0), "output", PortType::Video)
                .with_format(PortFormat::Video(VideoPortFormat::any()))],
            config,
        }
    }

    /// Get the current configuration.
    #[must_use]
    pub fn config(&self) -> &PadConfig {
        &self.config
    }

    /// Update the pad configuration.
    pub fn set_config(&mut self, config: PadConfig) {
        self.config = config;
    }

    /// Pad a single plane.
    fn pad_plane(
        &self,
        src: &Plane,
        src_width: u32,
        src_height: u32,
        pad: &PadValues,
        fill: u8,
    ) -> Plane {
        let dst_width = src_width + pad.left + pad.right;
        let dst_height = src_height + pad.top + pad.bottom;
        let mut dst_data = vec![fill; dst_width as usize * dst_height as usize];

        // Copy source data into padded area
        for y in 0..src_height as usize {
            let src_row = src.row(y);
            let dst_y = y + pad.top as usize;
            let dst_start = dst_y * dst_width as usize + pad.left as usize;

            for x in 0..src_width as usize {
                dst_data[dst_start + x] = src_row.get(x).copied().unwrap_or(fill);
            }
        }

        Plane::new(dst_data, dst_width as usize)
    }

    /// Pad a video frame.
    fn pad_frame(&self, input: &VideoFrame) -> VideoFrame {
        let pad = self.config.calculate_padding(input.width, input.height);
        let (dst_width, dst_height) = self.config.output_dimensions(input.width, input.height);

        let mut output = VideoFrame::new(input.format, dst_width, dst_height);
        output.timestamp = input.timestamp;
        output.frame_type = input.frame_type;
        output.color_info = input.color_info;

        let (y_fill, u_fill, v_fill) = self.config.color.to_yuv(input.color_info.full_range);

        for (i, src_plane) in input.planes.iter().enumerate() {
            let (src_w, src_h) = input.plane_dimensions(i);

            let (plane_pad, fill) = if i > 0 && input.format.is_yuv() {
                let (h_ratio, v_ratio) = input.format.chroma_subsampling();
                let fill = if i == 1 { u_fill } else { v_fill };
                (pad.scale_for_chroma(h_ratio, v_ratio), fill)
            } else {
                (pad, y_fill)
            };

            let plane = self.pad_plane(src_plane, src_w, src_h, &plane_pad, fill);
            output.planes.push(plane);
        }

        output
    }
}

impl Node for PadFilter {
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
                let padded = self.pad_frame(&frame);
                Ok(Some(FilterFrame::Video(padded)))
            }
            Some(_) => Err(GraphError::PortTypeMismatch {
                expected: "Video".to_string(),
                actual: "Audio".to_string(),
            }),
            None => Ok(None),
        }
    }
}

/// Create a letterbox configuration for 16:9 content.
#[must_use]
pub fn letterbox_16_9() -> PadConfig {
    PadConfig::for_aspect(16.0 / 9.0).with_color(PadColor::Black)
}

/// Create a letterbox configuration for 4:3 content.
#[must_use]
pub fn letterbox_4_3() -> PadConfig {
    PadConfig::for_aspect(4.0 / 3.0).with_color(PadColor::Black)
}

/// Create a letterbox configuration for 2.35:1 (CinemaScope) content.
#[must_use]
pub fn letterbox_cinemascope() -> PadConfig {
    PadConfig::for_aspect(2.35).with_color(PadColor::Black)
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_core::PixelFormat;

    fn create_test_frame(width: u32, height: u32) -> VideoFrame {
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, width, height);
        frame.allocate();

        // Fill Y plane with mid-gray
        if let Some(plane) = frame.planes.get_mut(0) {
            let data = vec![128u8; width as usize * height as usize];
            *plane = Plane::new(data, width as usize);
        }

        frame
    }

    #[test]
    fn test_pad_color_to_yuv() {
        let (y, u, v) = PadColor::Black.to_yuv(true);
        assert_eq!((y, u, v), (0, 128, 128));

        let (y, u, v) = PadColor::Black.to_yuv(false);
        assert_eq!((y, u, v), (16, 128, 128));

        let (y, u, v) = PadColor::White.to_yuv(true);
        assert_eq!((y, u, v), (255, 128, 128));

        let (y, u, v) = PadColor::Gray.to_yuv(true);
        assert_eq!((y, u, v), (128, 128, 128));
    }

    #[test]
    fn test_pad_color_custom_yuv() {
        let color = PadColor::yuv(100, 150, 200);
        let (y, u, v) = color.to_yuv(true);
        assert_eq!((y, u, v), (100, 150, 200));
    }

    #[test]
    fn test_pad_color_rgb() {
        let color = PadColor::rgb(255, 0, 0); // Red
        let (y, u, v) = color.to_yuv(true);
        // Red in YUV should have high Y, low U, high V
        assert!(y > 50);
        assert!(u < 128);
        assert!(v > 128);
    }

    #[test]
    fn test_rgb_to_yuv() {
        // White
        let (y, u, v) = rgb_to_yuv(255, 255, 255);
        assert_eq!(y, 255);
        assert_eq!(u, 128);
        assert_eq!(v, 128);

        // Black
        let (y, u, v) = rgb_to_yuv(0, 0, 0);
        assert_eq!(y, 0);
        assert_eq!(u, 128);
        assert_eq!(v, 128);
    }

    #[test]
    fn test_pad_config_explicit() {
        let config = PadConfig::new(10, 20, 30, 40);
        assert_eq!(config.left, 10);
        assert_eq!(config.top, 20);
        assert_eq!(config.right, 30);
        assert_eq!(config.bottom, 40);
    }

    #[test]
    fn test_pad_config_to_size() {
        let config = PadConfig::to_size(1920, 1080);
        let pad = config.calculate_padding(1280, 720);

        let dst_width = 1280 + pad.left + pad.right;
        let dst_height = 720 + pad.top + pad.bottom;

        assert_eq!(dst_width, 1920);
        assert_eq!(dst_height, 1080);
    }

    #[test]
    fn test_pad_config_for_aspect() {
        // 4:3 to 16:9 (pillarbox)
        let config = PadConfig::for_aspect(16.0 / 9.0);
        let pad = config.calculate_padding(640, 480);

        assert!(pad.left > 0);
        assert!(pad.right > 0);
        assert_eq!(pad.top, 0);
        assert_eq!(pad.bottom, 0);

        // 16:9 to 4:3 (letterbox)
        let config = PadConfig::for_aspect(4.0 / 3.0);
        let pad = config.calculate_padding(1920, 1080);

        assert_eq!(pad.left, 0);
        assert_eq!(pad.right, 0);
        assert!(pad.top > 0);
        assert!(pad.bottom > 0);
    }

    #[test]
    fn test_pad_values_methods() {
        let pad = PadValues {
            left: 10,
            top: 20,
            right: 30,
            bottom: 40,
        };

        assert_eq!(pad.horizontal(), 40);
        assert_eq!(pad.vertical(), 60);
        assert!(pad.has_padding());

        let no_pad = PadValues {
            left: 0,
            top: 0,
            right: 0,
            bottom: 0,
        };
        assert!(!no_pad.has_padding());
    }

    #[test]
    fn test_pad_values_scale_for_chroma() {
        let pad = PadValues {
            left: 100,
            top: 200,
            right: 100,
            bottom: 200,
        };

        let scaled = pad.scale_for_chroma(2, 2);
        assert_eq!(scaled.left, 50);
        assert_eq!(scaled.top, 100);
        assert_eq!(scaled.right, 50);
        assert_eq!(scaled.bottom, 100);
    }

    #[test]
    fn test_pad_filter_creation() {
        let config = PadConfig::new(10, 20, 10, 20);
        let filter = PadFilter::new(NodeId(0), "pad", config);

        assert_eq!(filter.id(), NodeId(0));
        assert_eq!(filter.name(), "pad");
        assert_eq!(filter.node_type(), NodeType::Filter);
    }

    #[test]
    fn test_pad_filter_process() {
        let config = PadConfig::new(10, 20, 10, 20);
        let mut filter = PadFilter::new(NodeId(0), "pad", config);

        let input = create_test_frame(640, 480);
        let result = filter
            .process(Some(FilterFrame::Video(input)))
            .expect("operation should succeed")
            .expect("operation should succeed");

        if let FilterFrame::Video(frame) = result {
            assert_eq!(frame.width, 660); // 640 + 10 + 10
            assert_eq!(frame.height, 520); // 480 + 20 + 20
        } else {
            panic!("Expected video frame");
        }
    }

    #[test]
    fn test_letterbox_presets() {
        let config = letterbox_16_9();
        assert!(config.target_aspect.is_some());

        let config = letterbox_4_3();
        assert!(config.target_aspect.is_some());

        let config = letterbox_cinemascope();
        assert!(config.target_aspect.is_some());
    }

    #[test]
    fn test_node_state_transitions() {
        let config = PadConfig::new(10, 10, 10, 10);
        let mut filter = PadFilter::new(NodeId(0), "pad", config);

        assert_eq!(filter.state(), NodeState::Idle);
        filter
            .set_state(NodeState::Processing)
            .expect("set_state should succeed");
        assert_eq!(filter.state(), NodeState::Processing);
    }

    #[test]
    fn test_process_none_input() {
        let config = PadConfig::new(10, 10, 10, 10);
        let mut filter = PadFilter::new(NodeId(0), "pad", config);

        let result = filter.process(None).expect("process should succeed");
        assert!(result.is_none());
    }

    #[test]
    fn test_output_dimensions() {
        let config = PadConfig::to_size(1920, 1080);
        let (w, h) = config.output_dimensions(1280, 720);
        assert_eq!(w, 1920);
        assert_eq!(h, 1080);
    }
}
