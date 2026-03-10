//! Video overlay/compositing filter.
//!
//! This filter composites one video stream over another, supporting various
//! blend modes, positioning, and alpha blending.

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
use oximedia_core::PixelFormat;

/// Blend mode for compositing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum BlendMode {
    /// Normal blend (overlay on top with alpha).
    #[default]
    Normal,
    /// Additive blend (lighten).
    Add,
    /// Multiplicative blend (darken).
    Multiply,
    /// Screen blend (lighten, opposite of multiply).
    Screen,
    /// Overlay blend (combination of multiply and screen).
    Overlay,
    /// Darken (take minimum).
    Darken,
    /// Lighten (take maximum).
    Lighten,
    /// Difference blend.
    Difference,
    /// Exclusion blend.
    Exclusion,
}

impl BlendMode {
    /// Apply this blend mode to two pixel values.
    ///
    /// Both values should be in the range 0.0-1.0.
    #[must_use]
    pub fn blend(&self, base: f64, overlay: f64) -> f64 {
        match self {
            Self::Normal => overlay,
            Self::Add => (base + overlay).min(1.0),
            Self::Multiply => base * overlay,
            Self::Screen => 1.0 - (1.0 - base) * (1.0 - overlay),
            Self::Overlay => {
                if base < 0.5 {
                    2.0 * base * overlay
                } else {
                    1.0 - 2.0 * (1.0 - base) * (1.0 - overlay)
                }
            }
            Self::Darken => base.min(overlay),
            Self::Lighten => base.max(overlay),
            Self::Difference => (base - overlay).abs(),
            Self::Exclusion => base + overlay - 2.0 * base * overlay,
        }
    }

    /// Apply blend with alpha.
    #[must_use]
    pub fn blend_with_alpha(&self, base: f64, overlay: f64, alpha: f64) -> f64 {
        let blended = self.blend(base, overlay);
        base * (1.0 - alpha) + blended * alpha
    }
}

/// Position alignment for overlay.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Alignment {
    /// Top-left corner.
    #[default]
    TopLeft,
    /// Top-center.
    TopCenter,
    /// Top-right corner.
    TopRight,
    /// Center-left.
    CenterLeft,
    /// Center.
    Center,
    /// Center-right.
    CenterRight,
    /// Bottom-left corner.
    BottomLeft,
    /// Bottom-center.
    BottomCenter,
    /// Bottom-right corner.
    BottomRight,
}

impl Alignment {
    /// Calculate position offset for alignment.
    #[must_use]
    pub fn offset(
        &self,
        container_w: u32,
        container_h: u32,
        content_w: u32,
        content_h: u32,
    ) -> (i32, i32) {
        let h_offset = match self {
            Self::TopLeft | Self::CenterLeft | Self::BottomLeft => 0,
            Self::TopCenter | Self::Center | Self::BottomCenter => {
                (container_w.saturating_sub(content_w) / 2) as i32
            }
            Self::TopRight | Self::CenterRight | Self::BottomRight => {
                container_w.saturating_sub(content_w) as i32
            }
        };

        let v_offset = match self {
            Self::TopLeft | Self::TopCenter | Self::TopRight => 0,
            Self::CenterLeft | Self::Center | Self::CenterRight => {
                (container_h.saturating_sub(content_h) / 2) as i32
            }
            Self::BottomLeft | Self::BottomCenter | Self::BottomRight => {
                container_h.saturating_sub(content_h) as i32
            }
        };

        (h_offset, v_offset)
    }
}

/// Configuration for the overlay filter.
#[derive(Clone, Debug)]
pub struct OverlayConfig {
    /// X position of overlay (relative to base).
    pub x: i32,
    /// Y position of overlay (relative to base).
    pub y: i32,
    /// Alignment mode.
    pub alignment: Alignment,
    /// Blend mode.
    pub blend_mode: BlendMode,
    /// Global alpha (0.0 = transparent, 1.0 = opaque).
    pub alpha: f64,
    /// Use overlay's alpha channel if available.
    pub use_alpha_channel: bool,
    /// Enable alpha premultiplication.
    pub premultiplied_alpha: bool,
}

impl Default for OverlayConfig {
    fn default() -> Self {
        Self {
            x: 0,
            y: 0,
            alignment: Alignment::default(),
            blend_mode: BlendMode::default(),
            alpha: 1.0,
            use_alpha_channel: true,
            premultiplied_alpha: false,
        }
    }
}

impl OverlayConfig {
    /// Create a new overlay configuration.
    #[must_use]
    pub fn new(x: i32, y: i32) -> Self {
        Self {
            x,
            y,
            ..Default::default()
        }
    }

    /// Create a centered overlay configuration.
    #[must_use]
    pub fn centered() -> Self {
        Self {
            alignment: Alignment::Center,
            ..Default::default()
        }
    }

    /// Set the position.
    #[must_use]
    pub fn with_position(mut self, x: i32, y: i32) -> Self {
        self.x = x;
        self.y = y;
        self
    }

    /// Set the alignment.
    #[must_use]
    pub fn with_alignment(mut self, alignment: Alignment) -> Self {
        self.alignment = alignment;
        self
    }

    /// Set the blend mode.
    #[must_use]
    pub fn with_blend_mode(mut self, mode: BlendMode) -> Self {
        self.blend_mode = mode;
        self
    }

    /// Set the global alpha.
    #[must_use]
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha.clamp(0.0, 1.0);
        self
    }

    /// Enable or disable alpha channel usage.
    #[must_use]
    pub fn with_use_alpha_channel(mut self, use_alpha: bool) -> Self {
        self.use_alpha_channel = use_alpha;
        self
    }

    /// Calculate the actual position for overlay.
    #[must_use]
    pub fn calculate_position(
        &self,
        base_w: u32,
        base_h: u32,
        overlay_w: u32,
        overlay_h: u32,
    ) -> (i32, i32) {
        let (align_x, align_y) = self.alignment.offset(base_w, base_h, overlay_w, overlay_h);
        (align_x + self.x, align_y + self.y)
    }
}

/// Video overlay filter.
///
/// Composites an overlay video onto a base video with various blend modes
/// and positioning options.
///
/// # Example
///
/// ```ignore
/// use oximedia_graph::filters::video::{OverlayFilter, OverlayConfig, BlendMode, Alignment};
/// use oximedia_graph::node::NodeId;
///
/// // Create a centered overlay with 50% opacity
/// let config = OverlayConfig::centered()
///     .with_blend_mode(BlendMode::Normal)
///     .with_alpha(0.5);
///
/// let filter = OverlayFilter::new(NodeId(0), "overlay", config);
/// ```
pub struct OverlayFilter {
    id: NodeId,
    name: String,
    state: NodeState,
    inputs: Vec<InputPort>,
    outputs: Vec<OutputPort>,
    config: OverlayConfig,
    /// Buffered base frame.
    base_frame: Option<VideoFrame>,
    /// Buffered overlay frame.
    overlay_frame: Option<VideoFrame>,
}

impl OverlayFilter {
    /// Create a new overlay filter.
    #[must_use]
    pub fn new(id: NodeId, name: impl Into<String>, config: OverlayConfig) -> Self {
        Self {
            id,
            name: name.into(),
            state: NodeState::Idle,
            inputs: vec![
                InputPort::new(PortId(0), "base", PortType::Video)
                    .with_format(PortFormat::Video(VideoPortFormat::any())),
                InputPort::new(PortId(1), "overlay", PortType::Video)
                    .with_format(PortFormat::Video(VideoPortFormat::any())),
            ],
            outputs: vec![OutputPort::new(PortId(0), "output", PortType::Video)
                .with_format(PortFormat::Video(VideoPortFormat::any()))],
            config,
            base_frame: None,
            overlay_frame: None,
        }
    }

    /// Get the current configuration.
    #[must_use]
    pub fn config(&self) -> &OverlayConfig {
        &self.config
    }

    /// Update the configuration.
    pub fn set_config(&mut self, config: OverlayConfig) {
        self.config = config;
    }

    /// Set the base frame.
    pub fn set_base_frame(&mut self, frame: VideoFrame) {
        self.base_frame = Some(frame);
    }

    /// Set the overlay frame.
    pub fn set_overlay_frame(&mut self, frame: VideoFrame) {
        self.overlay_frame = Some(frame);
    }

    /// Composite the overlay onto the base frame.
    fn composite(&self, base: &VideoFrame, overlay: &VideoFrame) -> VideoFrame {
        let mut output = base.clone();

        let (pos_x, pos_y) =
            self.config
                .calculate_position(base.width, base.height, overlay.width, overlay.height);

        // Handle YUV formats
        if base.format.is_yuv() && overlay.format.is_yuv() {
            self.composite_yuv(&mut output, overlay, pos_x, pos_y);
        } else {
            // For simplicity, handle RGB/RGBA compositing
            self.composite_rgb(&mut output, overlay, pos_x, pos_y);
        }

        output
    }

    /// Composite YUV frames.
    fn composite_yuv(&self, output: &mut VideoFrame, overlay: &VideoFrame, pos_x: i32, pos_y: i32) {
        let format = output.format;
        let (h_sub, v_sub) = format.chroma_subsampling();

        // Pre-calculate dimensions for each plane
        let plane_infos: Vec<_> = (0..output.planes.len().min(overlay.planes.len()))
            .map(|plane_idx| {
                let (base_w, base_h) = if plane_idx == 0 {
                    (output.width, output.height)
                } else {
                    (output.width / h_sub, output.height / v_sub)
                };
                let (overlay_w, overlay_h) = if plane_idx == 0 {
                    (overlay.width, overlay.height)
                } else {
                    (overlay.width / h_sub, overlay.height / v_sub)
                };
                let (scale_x, scale_y) = if plane_idx > 0 {
                    (h_sub as i32, v_sub as i32)
                } else {
                    (1, 1)
                };
                (base_w, base_h, overlay_w, overlay_h, scale_x, scale_y)
            })
            .collect();

        for (plane_idx, (base_plane, overlay_plane)) in output
            .planes
            .iter_mut()
            .zip(overlay.planes.iter())
            .enumerate()
        {
            let (base_w, base_h, overlay_w, overlay_h, scale_x, scale_y) = plane_infos[plane_idx];

            let plane_pos_x = pos_x / scale_x;
            let plane_pos_y = pos_y / scale_y;

            // Clone and modify data
            let mut new_data = base_plane.data.to_vec();

            for oy in 0..overlay_h as i32 {
                let by = plane_pos_y + oy;
                if by < 0 || by >= base_h as i32 {
                    continue;
                }

                for ox in 0..overlay_w as i32 {
                    let bx = plane_pos_x + ox;
                    if bx < 0 || bx >= base_w as i32 {
                        continue;
                    }

                    let base_idx = (by as usize) * base_w as usize + bx as usize;
                    let overlay_idx = (oy as usize) * overlay_plane.stride + ox as usize;

                    let base_val = new_data.get(base_idx).copied().unwrap_or(128) as f64 / 255.0;
                    let overlay_val =
                        overlay_plane.data.get(overlay_idx).copied().unwrap_or(128) as f64 / 255.0;

                    let blended = self.config.blend_mode.blend_with_alpha(
                        base_val,
                        overlay_val,
                        self.config.alpha,
                    );

                    new_data[base_idx] = (blended * 255.0).round().clamp(0.0, 255.0) as u8;
                }
            }

            *base_plane = Plane::new(new_data, base_plane.stride);
        }
    }

    /// Composite RGB/RGBA frames.
    fn composite_rgb(&self, output: &mut VideoFrame, overlay: &VideoFrame, pos_x: i32, pos_y: i32) {
        if output.planes.is_empty() || overlay.planes.is_empty() {
            return;
        }

        let base_plane = &output.planes[0];
        let overlay_plane = &overlay.planes[0];

        let base_bpp = if output.format == PixelFormat::Rgba32 {
            4
        } else {
            3
        };
        let overlay_bpp = if overlay.format == PixelFormat::Rgba32 {
            4
        } else {
            3
        };

        let mut new_data = base_plane.data.to_vec();

        for oy in 0..overlay.height as i32 {
            let by = pos_y + oy;
            if by < 0 || by >= output.height as i32 {
                continue;
            }

            for ox in 0..overlay.width as i32 {
                let bx = pos_x + ox;
                if bx < 0 || bx >= output.width as i32 {
                    continue;
                }

                let base_idx = (by as usize * output.width as usize + bx as usize) * base_bpp;
                let overlay_idx =
                    (oy as usize * overlay.width as usize + ox as usize) * overlay_bpp;

                // Get alpha
                let alpha = if self.config.use_alpha_channel && overlay_bpp == 4 {
                    let overlay_alpha = overlay_plane
                        .data
                        .get(overlay_idx + 3)
                        .copied()
                        .unwrap_or(255) as f64
                        / 255.0;
                    overlay_alpha * self.config.alpha
                } else {
                    self.config.alpha
                };

                // Blend each channel
                for c in 0..3 {
                    let base_val = new_data.get(base_idx + c).copied().unwrap_or(0) as f64 / 255.0;
                    let overlay_val = overlay_plane
                        .data
                        .get(overlay_idx + c)
                        .copied()
                        .unwrap_or(0) as f64
                        / 255.0;

                    let blended =
                        self.config
                            .blend_mode
                            .blend_with_alpha(base_val, overlay_val, alpha);

                    new_data[base_idx + c] = (blended * 255.0).round().clamp(0.0, 255.0) as u8;
                }

                // Handle output alpha
                if base_bpp == 4 {
                    let base_alpha =
                        new_data.get(base_idx + 3).copied().unwrap_or(255) as f64 / 255.0;
                    let out_alpha = base_alpha + alpha * (1.0 - base_alpha);
                    new_data[base_idx + 3] = (out_alpha * 255.0).round().clamp(0.0, 255.0) as u8;
                }
            }
        }

        output.planes[0] = Plane::new(new_data, base_plane.stride);
    }

    /// Process when both frames are available.
    fn try_composite(&mut self) -> Option<VideoFrame> {
        match (self.base_frame.take(), self.overlay_frame.take()) {
            (Some(base), Some(overlay)) => Some(self.composite(&base, &overlay)),
            (Some(base), None) => Some(base), // No overlay, pass through base
            (None, Some(_)) => None,          // No base, cannot output
            (None, None) => None,
        }
    }
}

impl Node for OverlayFilter {
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
        // Note: This filter has 2 inputs. In a real implementation,
        // the graph runtime would manage multiple inputs.
        // For simplicity, we treat the input as the base frame.
        match input {
            Some(FilterFrame::Video(frame)) => {
                self.base_frame = Some(frame);

                // If we have both frames, composite
                if self.overlay_frame.is_some() {
                    Ok(self.try_composite().map(FilterFrame::Video))
                } else {
                    // Pass through base if no overlay
                    Ok(self.base_frame.take().map(FilterFrame::Video))
                }
            }
            Some(_) => Err(GraphError::PortTypeMismatch {
                expected: "Video".to_string(),
                actual: "Audio".to_string(),
            }),
            None => Ok(None),
        }
    }

    fn reset(&mut self) -> GraphResult<()> {
        self.base_frame = None;
        self.overlay_frame = None;
        self.set_state(NodeState::Idle)
    }
}

/// Helper function to create a simple color overlay.
#[must_use]
pub fn create_color_overlay(width: u32, height: u32, r: u8, g: u8, b: u8, alpha: u8) -> VideoFrame {
    let mut frame = VideoFrame::new(PixelFormat::Rgba32, width, height);

    let size = (width * height * 4) as usize;
    let mut data = vec![0u8; size];

    for i in (0..size).step_by(4) {
        data[i] = r;
        data[i + 1] = g;
        data[i + 2] = b;
        data[i + 3] = alpha;
    }

    frame.planes.push(Plane::new(data, (width * 4) as usize));
    frame
}

/// Helper function to create a gradient overlay.
#[must_use]
pub fn create_gradient_overlay(
    width: u32,
    height: u32,
    start_color: (u8, u8, u8),
    end_color: (u8, u8, u8),
    horizontal: bool,
) -> VideoFrame {
    let mut frame = VideoFrame::new(PixelFormat::Rgba32, width, height);

    let size = (width * height * 4) as usize;
    let mut data = vec![0u8; size];

    for y in 0..height as usize {
        for x in 0..width as usize {
            let t = if horizontal {
                x as f64 / (width as f64 - 1.0).max(1.0)
            } else {
                y as f64 / (height as f64 - 1.0).max(1.0)
            };

            let r = (start_color.0 as f64 * (1.0 - t) + end_color.0 as f64 * t).round() as u8;
            let g = (start_color.1 as f64 * (1.0 - t) + end_color.1 as f64 * t).round() as u8;
            let b = (start_color.2 as f64 * (1.0 - t) + end_color.2 as f64 * t).round() as u8;

            let idx = (y * width as usize + x) * 4;
            data[idx] = r;
            data[idx + 1] = g;
            data[idx + 2] = b;
            data[idx + 3] = 255;
        }
    }

    frame.planes.push(Plane::new(data, (width * 4) as usize));
    frame
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_yuv_frame(width: u32, height: u32, fill_y: u8) -> VideoFrame {
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, width, height);
        frame.allocate();

        if let Some(plane) = frame.planes.get_mut(0) {
            let data = vec![fill_y; (width * height) as usize];
            *plane = Plane::new(data, width as usize);
        }

        frame
    }

    fn create_test_rgb_frame(width: u32, height: u32, r: u8, g: u8, b: u8) -> VideoFrame {
        let mut frame = VideoFrame::new(PixelFormat::Rgb24, width, height);

        let size = (width * height * 3) as usize;
        let mut data = vec![0u8; size];

        for i in (0..size).step_by(3) {
            data[i] = r;
            data[i + 1] = g;
            data[i + 2] = b;
        }

        frame.planes.push(Plane::new(data, (width * 3) as usize));
        frame
    }

    #[test]
    fn test_blend_modes() {
        // Normal blend
        assert!((BlendMode::Normal.blend(0.5, 0.8) - 0.8).abs() < 0.001);

        // Add blend
        assert!((BlendMode::Add.blend(0.6, 0.5) - 1.0).abs() < 0.001); // Clamped

        // Multiply blend
        assert!((BlendMode::Multiply.blend(0.5, 0.5) - 0.25).abs() < 0.001);

        // Screen blend
        assert!((BlendMode::Screen.blend(0.5, 0.5) - 0.75).abs() < 0.001);

        // Darken blend
        assert!((BlendMode::Darken.blend(0.3, 0.7) - 0.3).abs() < 0.001);

        // Lighten blend
        assert!((BlendMode::Lighten.blend(0.3, 0.7) - 0.7).abs() < 0.001);

        // Difference blend
        assert!((BlendMode::Difference.blend(0.8, 0.3) - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_blend_with_alpha() {
        let result = BlendMode::Normal.blend_with_alpha(0.2, 0.8, 0.5);
        // 0.2 * 0.5 + 0.8 * 0.5 = 0.1 + 0.4 = 0.5
        assert!((result - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_alignment_offset() {
        let (x, y) = Alignment::Center.offset(100, 100, 20, 20);
        assert_eq!(x, 40);
        assert_eq!(y, 40);

        let (x, y) = Alignment::TopLeft.offset(100, 100, 20, 20);
        assert_eq!(x, 0);
        assert_eq!(y, 0);

        let (x, y) = Alignment::BottomRight.offset(100, 100, 20, 20);
        assert_eq!(x, 80);
        assert_eq!(y, 80);
    }

    #[test]
    fn test_overlay_config() {
        let config = OverlayConfig::new(10, 20)
            .with_alignment(Alignment::Center)
            .with_blend_mode(BlendMode::Multiply)
            .with_alpha(0.75)
            .with_use_alpha_channel(false);

        assert_eq!(config.x, 10);
        assert_eq!(config.y, 20);
        assert_eq!(config.alignment, Alignment::Center);
        assert_eq!(config.blend_mode, BlendMode::Multiply);
        assert!((config.alpha - 0.75).abs() < 0.001);
        assert!(!config.use_alpha_channel);
    }

    #[test]
    fn test_overlay_config_centered() {
        let config = OverlayConfig::centered();
        assert_eq!(config.alignment, Alignment::Center);
    }

    #[test]
    fn test_calculate_position() {
        let config = OverlayConfig::new(10, 20).with_alignment(Alignment::Center);
        let (x, y) = config.calculate_position(100, 100, 20, 20);

        // Center offset (40, 40) + position (10, 20) = (50, 60)
        assert_eq!(x, 50);
        assert_eq!(y, 60);
    }

    #[test]
    fn test_overlay_filter_creation() {
        let config = OverlayConfig::default();
        let filter = OverlayFilter::new(NodeId(0), "overlay", config);

        assert_eq!(filter.id(), NodeId(0));
        assert_eq!(filter.name(), "overlay");
        assert_eq!(filter.node_type(), NodeType::Filter);
        assert_eq!(filter.inputs().len(), 2); // Base and overlay inputs
        assert_eq!(filter.outputs().len(), 1);
    }

    #[test]
    fn test_composite_yuv() {
        let config = OverlayConfig::new(0, 0).with_alpha(0.5);
        let filter = OverlayFilter::new(NodeId(0), "overlay", config);

        let base = create_test_yuv_frame(64, 48, 100);
        let overlay = create_test_yuv_frame(32, 24, 200);

        let result = filter.composite(&base, &overlay);

        assert_eq!(result.width, 64);
        assert_eq!(result.height, 48);
    }

    #[test]
    fn test_composite_rgb() {
        let config = OverlayConfig::new(0, 0).with_alpha(1.0);
        let filter = OverlayFilter::new(NodeId(0), "overlay", config);

        let base = create_test_rgb_frame(64, 48, 0, 0, 0);
        let overlay = create_test_rgb_frame(32, 24, 255, 255, 255);

        let result = filter.composite(&base, &overlay);

        assert_eq!(result.width, 64);
        assert_eq!(result.height, 48);
    }

    #[test]
    fn test_create_color_overlay() {
        let overlay = create_color_overlay(64, 48, 255, 0, 0, 128);

        assert_eq!(overlay.width, 64);
        assert_eq!(overlay.height, 48);
        assert_eq!(overlay.format, PixelFormat::Rgba32);
        assert!(!overlay.planes.is_empty());

        // Check first pixel
        let data = &overlay.planes[0].data;
        assert_eq!(data[0], 255); // R
        assert_eq!(data[1], 0); // G
        assert_eq!(data[2], 0); // B
        assert_eq!(data[3], 128); // A
    }

    #[test]
    fn test_create_gradient_overlay() {
        let overlay = create_gradient_overlay(64, 48, (255, 0, 0), (0, 0, 255), true);

        assert_eq!(overlay.width, 64);
        assert_eq!(overlay.height, 48);
        assert_eq!(overlay.format, PixelFormat::Rgba32);

        // Check gradient: first pixel should be red-ish, last should be blue-ish
        let data = &overlay.planes[0].data;
        assert!(data[0] > 200); // R high at start
        let last_idx = ((64 * 48 - 1) * 4) as usize;
        assert!(data[last_idx + 2] > 200); // B high at end
    }

    #[test]
    fn test_process_with_base_only() {
        let config = OverlayConfig::default();
        let mut filter = OverlayFilter::new(NodeId(0), "overlay", config);

        let base = create_test_yuv_frame(64, 48, 100);
        let result = filter
            .process(Some(FilterFrame::Video(base)))
            .expect("operation should succeed")
            .expect("operation should succeed");

        // Should pass through base when no overlay
        assert!(matches!(result, FilterFrame::Video(_)));
    }

    #[test]
    fn test_process_with_both_frames() {
        let config = OverlayConfig::new(0, 0);
        let mut filter = OverlayFilter::new(NodeId(0), "overlay", config);

        // Set overlay first
        let overlay = create_test_yuv_frame(32, 24, 200);
        filter.set_overlay_frame(overlay);

        // Then process base
        let base = create_test_yuv_frame(64, 48, 100);
        let result = filter
            .process(Some(FilterFrame::Video(base)))
            .expect("operation should succeed")
            .expect("operation should succeed");

        assert!(matches!(result, FilterFrame::Video(_)));
    }

    #[test]
    fn test_node_state_transitions() {
        let config = OverlayConfig::default();
        let mut filter = OverlayFilter::new(NodeId(0), "overlay", config);

        assert_eq!(filter.state(), NodeState::Idle);
        filter
            .set_state(NodeState::Processing)
            .expect("set_state should succeed");
        assert_eq!(filter.state(), NodeState::Processing);
    }

    #[test]
    fn test_process_none_input() {
        let config = OverlayConfig::default();
        let mut filter = OverlayFilter::new(NodeId(0), "overlay", config);

        let result = filter.process(None).expect("process should succeed");
        assert!(result.is_none());
    }

    #[test]
    fn test_reset() {
        let config = OverlayConfig::default();
        let mut filter = OverlayFilter::new(NodeId(0), "overlay", config);

        filter.set_base_frame(create_test_yuv_frame(64, 48, 100));
        filter.set_overlay_frame(create_test_yuv_frame(32, 24, 200));

        filter.reset().expect("reset should succeed");

        assert!(filter.base_frame.is_none());
        assert!(filter.overlay_frame.is_none());
    }

    #[test]
    fn test_overlay_blend_mode() {
        // Test overlay blend mode (combination)
        let result = BlendMode::Overlay.blend(0.25, 0.5);
        // For base < 0.5: 2 * base * overlay = 2 * 0.25 * 0.5 = 0.25
        assert!((result - 0.25).abs() < 0.001);

        let result = BlendMode::Overlay.blend(0.75, 0.5);
        // For base >= 0.5: 1 - 2 * (1-base) * (1-overlay) = 1 - 2 * 0.25 * 0.5 = 0.75
        assert!((result - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_exclusion_blend() {
        let result = BlendMode::Exclusion.blend(0.5, 0.5);
        // base + overlay - 2 * base * overlay = 0.5 + 0.5 - 2 * 0.25 = 0.5
        assert!((result - 0.5).abs() < 0.001);
    }
}
