//! Video scaling filter.
//!
//! This filter rescales video frames to a target resolution using various
//! resampling algorithms including Lanczos, Bicubic, Bilinear, and Nearest Neighbor.

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

use std::f64::consts::PI;

use crate::error::{GraphError, GraphResult};
use crate::frame::FilterFrame;
use crate::node::{Node, NodeId, NodeState, NodeType};
use crate::port::{InputPort, OutputPort, PortFormat, PortId, PortType, VideoPortFormat};
use oximedia_codec::{Plane, VideoFrame};

/// Scaling algorithm for image resampling.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ScaleAlgorithm {
    /// Nearest neighbor - fastest, lowest quality, good for pixel art.
    Nearest,
    /// Bilinear interpolation - fast, moderate quality.
    Bilinear,
    /// Bicubic interpolation using Mitchell-Netravali coefficients.
    #[default]
    Bicubic,
    /// Bicubic interpolation using Catmull-Rom spline.
    CatmullRom,
    /// Lanczos-2 - high quality, 2-tap sinc window.
    Lanczos2,
    /// Lanczos-3 - higher quality, 3-tap sinc window.
    Lanczos3,
    /// Lanczos-4 - highest quality, 4-tap sinc window.
    Lanczos4,
}

impl ScaleAlgorithm {
    /// Get the filter support (radius in source pixels).
    #[must_use]
    pub fn support(&self) -> f64 {
        match self {
            Self::Nearest => 0.5,
            Self::Bilinear => 1.0,
            Self::Bicubic | Self::CatmullRom => 2.0,
            Self::Lanczos2 => 2.0,
            Self::Lanczos3 => 3.0,
            Self::Lanczos4 => 4.0,
        }
    }

    /// Calculate the kernel value at position x.
    #[must_use]
    pub fn kernel(&self, x: f64) -> f64 {
        let x = x.abs();
        match self {
            Self::Nearest => {
                if x < 0.5 {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Bilinear => bilinear_kernel(x),
            Self::Bicubic => mitchell_netravali_kernel(x),
            Self::CatmullRom => catmull_rom_kernel(x),
            Self::Lanczos2 => lanczos_kernel(x, 2.0),
            Self::Lanczos3 => lanczos_kernel(x, 3.0),
            Self::Lanczos4 => lanczos_kernel(x, 4.0),
        }
    }
}

/// Bilinear interpolation kernel.
fn bilinear_kernel(x: f64) -> f64 {
    if x < 1.0 {
        1.0 - x
    } else {
        0.0
    }
}

/// Mitchell-Netravali bicubic kernel with B=1/3, C=1/3.
fn mitchell_netravali_kernel(x: f64) -> f64 {
    const B: f64 = 1.0 / 3.0;
    const C: f64 = 1.0 / 3.0;

    let x2 = x * x;
    let x3 = x2 * x;

    if x < 1.0 {
        ((12.0 - 9.0 * B - 6.0 * C) * x3 + (-18.0 + 12.0 * B + 6.0 * C) * x2 + (6.0 - 2.0 * B))
            / 6.0
    } else if x < 2.0 {
        ((-B - 6.0 * C) * x3
            + (6.0 * B + 30.0 * C) * x2
            + (-12.0 * B - 48.0 * C) * x
            + (8.0 * B + 24.0 * C))
            / 6.0
    } else {
        0.0
    }
}

/// Catmull-Rom bicubic kernel (B=0, C=0.5).
fn catmull_rom_kernel(x: f64) -> f64 {
    let x2 = x * x;
    let x3 = x2 * x;

    if x < 1.0 {
        1.5 * x3 - 2.5 * x2 + 1.0
    } else if x < 2.0 {
        -0.5 * x3 + 2.5 * x2 - 4.0 * x + 2.0
    } else {
        0.0
    }
}

/// Lanczos windowed sinc kernel.
fn lanczos_kernel(x: f64, a: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else if x < a {
        sinc(x) * sinc(x / a)
    } else {
        0.0
    }
}

/// Normalized sinc function.
fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else {
        let pix = PI * x;
        pix.sin() / pix
    }
}

/// Configuration for the scale filter.
#[derive(Clone, Debug)]
pub struct ScaleConfig {
    /// Target width.
    pub width: u32,
    /// Target height.
    pub height: u32,
    /// Scaling algorithm.
    pub algorithm: ScaleAlgorithm,
    /// Enable anti-aliasing for downscaling.
    pub antialias: bool,
    /// Preserve aspect ratio (letterbox/pillarbox as needed).
    pub preserve_aspect: bool,
}

impl ScaleConfig {
    /// Create a new scale configuration.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            algorithm: ScaleAlgorithm::default(),
            antialias: true,
            preserve_aspect: false,
        }
    }

    /// Set the scaling algorithm.
    #[must_use]
    pub fn with_algorithm(mut self, algorithm: ScaleAlgorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Enable or disable anti-aliasing.
    #[must_use]
    pub fn with_antialias(mut self, enabled: bool) -> Self {
        self.antialias = enabled;
        self
    }

    /// Enable or disable aspect ratio preservation.
    #[must_use]
    pub fn with_preserve_aspect(mut self, enabled: bool) -> Self {
        self.preserve_aspect = enabled;
        self
    }
}

/// Video scaling filter.
///
/// Rescales video frames to a target resolution using configurable
/// resampling algorithms.
///
/// # Example
///
/// ```ignore
/// use oximedia_graph::filters::video::{ScaleFilter, ScaleConfig, ScaleAlgorithm};
/// use oximedia_graph::node::NodeId;
///
/// let config = ScaleConfig::new(1280, 720)
///     .with_algorithm(ScaleAlgorithm::Lanczos3)
///     .with_antialias(true);
///
/// let filter = ScaleFilter::new(NodeId(0), "scale", config);
/// ```
pub struct ScaleFilter {
    id: NodeId,
    name: String,
    state: NodeState,
    inputs: Vec<InputPort>,
    outputs: Vec<OutputPort>,
    config: ScaleConfig,
    /// Precomputed horizontal filter coefficients.
    h_coefficients: Vec<FilterCoefficients>,
    /// Precomputed vertical filter coefficients.
    v_coefficients: Vec<FilterCoefficients>,
    /// Source dimensions (cached for coefficient reuse).
    cached_src_dims: Option<(u32, u32)>,
}

/// Filter coefficients for a single output pixel.
#[derive(Clone, Debug)]
struct FilterCoefficients {
    /// Starting position in source.
    start: usize,
    /// Coefficient weights.
    weights: Vec<f64>,
}

impl ScaleFilter {
    /// Create a new scale filter.
    #[must_use]
    pub fn new(id: NodeId, name: impl Into<String>, config: ScaleConfig) -> Self {
        let output_format =
            PortFormat::Video(VideoPortFormat::any().with_dimensions(config.width, config.height));

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
            h_coefficients: Vec::new(),
            v_coefficients: Vec::new(),
            cached_src_dims: None,
        }
    }

    /// Get the current configuration.
    #[must_use]
    pub fn config(&self) -> &ScaleConfig {
        &self.config
    }

    /// Update the target dimensions.
    pub fn set_dimensions(&mut self, width: u32, height: u32) {
        if self.config.width != width || self.config.height != height {
            self.config.width = width;
            self.config.height = height;
            self.cached_src_dims = None;
            self.h_coefficients.clear();
            self.v_coefficients.clear();
        }
    }

    /// Precompute filter coefficients for a given source and target size.
    fn compute_coefficients(&mut self, src_width: u32, src_height: u32) {
        if self.cached_src_dims == Some((src_width, src_height)) {
            return;
        }

        self.h_coefficients =
            Self::compute_1d_coefficients(src_width, self.config.width, &self.config);
        self.v_coefficients =
            Self::compute_1d_coefficients(src_height, self.config.height, &self.config);
        self.cached_src_dims = Some((src_width, src_height));
    }

    /// Compute 1D filter coefficients.
    fn compute_1d_coefficients(
        src_size: u32,
        dst_size: u32,
        config: &ScaleConfig,
    ) -> Vec<FilterCoefficients> {
        let mut coefficients = Vec::with_capacity(dst_size as usize);
        let scale = src_size as f64 / dst_size as f64;
        let algorithm = config.algorithm;

        // For downscaling with antialiasing, expand the filter support
        let filter_scale = if config.antialias && scale > 1.0 {
            scale
        } else {
            1.0
        };

        let support = algorithm.support() * filter_scale;

        for dst_pos in 0..dst_size {
            let center = (dst_pos as f64 + 0.5) * scale - 0.5;
            let start = ((center - support).floor() as i64).max(0) as usize;
            let end = ((center + support).ceil() as i64).min(src_size as i64) as usize;

            let mut weights = Vec::with_capacity(end - start);
            let mut sum = 0.0;

            for src_pos in start..end {
                let distance = (src_pos as f64 - center) / filter_scale;
                let weight = algorithm.kernel(distance);
                weights.push(weight);
                sum += weight;
            }

            // Normalize weights
            if sum != 0.0 {
                for w in &mut weights {
                    *w /= sum;
                }
            }

            coefficients.push(FilterCoefficients { start, weights });
        }

        coefficients
    }

    /// Scale a single plane.
    #[allow(clippy::too_many_arguments)]
    fn scale_plane(
        &self,
        src: &Plane,
        src_width: u32,
        src_height: u32,
        dst_width: u32,
        dst_height: u32,
    ) -> Plane {
        // Create intermediate buffer for horizontal pass
        let mut intermediate = vec![0.0f64; dst_width as usize * src_height as usize];

        // Horizontal pass
        for y in 0..src_height as usize {
            let src_row = src.row(y);
            for (x, coef) in self.h_coefficients.iter().enumerate() {
                let mut sum = 0.0;
                for (i, &weight) in coef.weights.iter().enumerate() {
                    let src_x = (coef.start + i).min(src_width as usize - 1);
                    sum += src_row.get(src_x).copied().unwrap_or(0) as f64 * weight;
                }
                intermediate[y * dst_width as usize + x] = sum;
            }
        }

        // Vertical pass
        let mut dst_data = vec![0u8; dst_width as usize * dst_height as usize];

        for y in 0..dst_height as usize {
            let coef = &self.v_coefficients[y];
            for x in 0..dst_width as usize {
                let mut sum = 0.0;
                for (i, &weight) in coef.weights.iter().enumerate() {
                    let src_y = (coef.start + i).min(src_height as usize - 1);
                    sum += intermediate[src_y * dst_width as usize + x] * weight;
                }
                dst_data[y * dst_width as usize + x] = sum.round().clamp(0.0, 255.0) as u8;
            }
        }

        Plane::new(dst_data, dst_width as usize)
    }

    /// Scale a video frame.
    fn scale_frame(&mut self, input: &VideoFrame) -> VideoFrame {
        self.compute_coefficients(input.width, input.height);

        let mut output = VideoFrame::new(input.format, self.config.width, self.config.height);
        output.timestamp = input.timestamp;
        output.frame_type = input.frame_type;
        output.color_info = input.color_info;

        // Scale each plane
        for (i, src_plane) in input.planes.iter().enumerate() {
            let (src_w, src_h) = input.plane_dimensions(i);
            let (dst_w, dst_h) = output.plane_dimensions(i);

            // For chroma planes, we need to compute separate coefficients
            if i > 0 && input.format.is_yuv() {
                let old_h = self.h_coefficients.clone();
                let old_v = self.v_coefficients.clone();
                let old_cached = self.cached_src_dims;

                self.h_coefficients = Self::compute_1d_coefficients(src_w, dst_w, &self.config);
                self.v_coefficients = Self::compute_1d_coefficients(src_h, dst_h, &self.config);

                let plane = self.scale_plane(src_plane, src_w, src_h, dst_w, dst_h);
                output.planes.push(plane);

                self.h_coefficients = old_h;
                self.v_coefficients = old_v;
                self.cached_src_dims = old_cached;
            } else {
                let plane = self.scale_plane(src_plane, src_w, src_h, dst_w, dst_h);
                output.planes.push(plane);
            }
        }

        output
    }
}

impl Node for ScaleFilter {
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
                let scaled = self.scale_frame(&frame);
                Ok(Some(FilterFrame::Video(scaled)))
            }
            Some(_) => Err(GraphError::PortTypeMismatch {
                expected: "Video".to_string(),
                actual: "Audio".to_string(),
            }),
            None => Ok(None),
        }
    }
}

/// Nearest neighbor scaler for fast, low-quality scaling.
#[derive(Debug)]
pub struct NearestNeighborScaler {
    dst_width: u32,
    dst_height: u32,
}

impl NearestNeighborScaler {
    /// Create a new nearest neighbor scaler.
    #[must_use]
    pub fn new(dst_width: u32, dst_height: u32) -> Self {
        Self {
            dst_width,
            dst_height,
        }
    }

    /// Scale a plane using nearest neighbor interpolation.
    #[must_use]
    pub fn scale_plane(&self, src: &Plane, src_width: u32, src_height: u32) -> Plane {
        let mut dst_data = vec![0u8; self.dst_width as usize * self.dst_height as usize];

        let x_ratio = src_width as f64 / self.dst_width as f64;
        let y_ratio = src_height as f64 / self.dst_height as f64;

        for y in 0..self.dst_height as usize {
            let src_y = ((y as f64 + 0.5) * y_ratio).floor() as usize;
            let src_y = src_y.min(src_height as usize - 1);
            let src_row = src.row(src_y);

            for x in 0..self.dst_width as usize {
                let src_x = ((x as f64 + 0.5) * x_ratio).floor() as usize;
                let src_x = src_x.min(src_width as usize - 1);
                dst_data[y * self.dst_width as usize + x] =
                    src_row.get(src_x).copied().unwrap_or(0);
            }
        }

        Plane::new(dst_data, self.dst_width as usize)
    }
}

/// Bilinear scaler for moderate quality scaling.
#[derive(Debug)]
pub struct BilinearScaler {
    dst_width: u32,
    dst_height: u32,
}

impl BilinearScaler {
    /// Create a new bilinear scaler.
    #[must_use]
    pub fn new(dst_width: u32, dst_height: u32) -> Self {
        Self {
            dst_width,
            dst_height,
        }
    }

    /// Scale a plane using bilinear interpolation.
    #[must_use]
    pub fn scale_plane(&self, src: &Plane, src_width: u32, src_height: u32) -> Plane {
        let mut dst_data = vec![0u8; self.dst_width as usize * self.dst_height as usize];

        let x_ratio = (src_width as f64 - 1.0) / (self.dst_width as f64 - 1.0).max(1.0);
        let y_ratio = (src_height as f64 - 1.0) / (self.dst_height as f64 - 1.0).max(1.0);

        for y in 0..self.dst_height as usize {
            let src_y = y as f64 * y_ratio;
            let y0 = src_y.floor() as usize;
            let y1 = (y0 + 1).min(src_height as usize - 1);
            let y_frac = src_y - y0 as f64;

            let row0 = src.row(y0);
            let row1 = src.row(y1);

            for x in 0..self.dst_width as usize {
                let src_x = x as f64 * x_ratio;
                let x0 = src_x.floor() as usize;
                let x1 = (x0 + 1).min(src_width as usize - 1);
                let x_frac = src_x - x0 as f64;

                let p00 = row0.get(x0).copied().unwrap_or(0) as f64;
                let p10 = row0.get(x1).copied().unwrap_or(0) as f64;
                let p01 = row1.get(x0).copied().unwrap_or(0) as f64;
                let p11 = row1.get(x1).copied().unwrap_or(0) as f64;

                let top = p00 * (1.0 - x_frac) + p10 * x_frac;
                let bottom = p01 * (1.0 - x_frac) + p11 * x_frac;
                let value = top * (1.0 - y_frac) + bottom * y_frac;

                dst_data[y * self.dst_width as usize + x] = value.round().clamp(0.0, 255.0) as u8;
            }
        }

        Plane::new(dst_data, self.dst_width as usize)
    }
}

/// Calculate the aspect ratio preserving dimensions.
#[must_use]
pub fn calculate_aspect_fit(
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
) -> (u32, u32) {
    let src_aspect = src_width as f64 / src_height as f64;
    let dst_aspect = dst_width as f64 / dst_height as f64;

    if src_aspect > dst_aspect {
        // Width limited
        let new_height = (dst_width as f64 / src_aspect).round() as u32;
        (dst_width, new_height)
    } else {
        // Height limited
        let new_width = (dst_height as f64 * src_aspect).round() as u32;
        (new_width, dst_height)
    }
}

/// Calculate the aspect ratio preserving dimensions for fill mode.
#[must_use]
pub fn calculate_aspect_fill(
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
) -> (u32, u32) {
    let src_aspect = src_width as f64 / src_height as f64;
    let dst_aspect = dst_width as f64 / dst_height as f64;

    if src_aspect < dst_aspect {
        // Width limited
        let new_height = (dst_width as f64 / src_aspect).round() as u32;
        (dst_width, new_height)
    } else {
        // Height limited
        let new_width = (dst_height as f64 * src_aspect).round() as u32;
        (new_width, dst_height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_core::PixelFormat;

    fn create_test_frame(width: u32, height: u32) -> VideoFrame {
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, width, height);
        frame.allocate();

        // Fill with a gradient pattern
        if let Some(plane) = frame.planes.get_mut(0) {
            let mut data = vec![0u8; width as usize * height as usize];
            for y in 0..height as usize {
                for x in 0..width as usize {
                    data[y * width as usize + x] = ((x + y) % 256) as u8;
                }
            }
            *plane = Plane::new(data, width as usize);
        }

        frame
    }

    #[test]
    fn test_scale_filter_creation() {
        let config = ScaleConfig::new(1280, 720)
            .with_algorithm(ScaleAlgorithm::Lanczos3)
            .with_antialias(true);

        let filter = ScaleFilter::new(NodeId(0), "scale", config);

        assert_eq!(filter.id(), NodeId(0));
        assert_eq!(filter.name(), "scale");
        assert_eq!(filter.config().width, 1280);
        assert_eq!(filter.config().height, 720);
        assert_eq!(filter.config().algorithm, ScaleAlgorithm::Lanczos3);
    }

    #[test]
    fn test_scale_algorithms() {
        assert_eq!(ScaleAlgorithm::Nearest.support(), 0.5);
        assert_eq!(ScaleAlgorithm::Bilinear.support(), 1.0);
        assert_eq!(ScaleAlgorithm::Bicubic.support(), 2.0);
        assert_eq!(ScaleAlgorithm::Lanczos2.support(), 2.0);
        assert_eq!(ScaleAlgorithm::Lanczos3.support(), 3.0);
        assert_eq!(ScaleAlgorithm::Lanczos4.support(), 4.0);
    }

    #[test]
    fn test_kernel_values() {
        // Nearest at center should be 1
        assert!((ScaleAlgorithm::Nearest.kernel(0.0) - 1.0).abs() < 0.001);
        assert!((ScaleAlgorithm::Nearest.kernel(0.6) - 0.0).abs() < 0.001);

        // Bilinear at center should be 1
        assert!((ScaleAlgorithm::Bilinear.kernel(0.0) - 1.0).abs() < 0.001);
        assert!((ScaleAlgorithm::Bilinear.kernel(0.5) - 0.5).abs() < 0.001);
        assert!((ScaleAlgorithm::Bilinear.kernel(1.0) - 0.0).abs() < 0.001);

        // Lanczos at center should be 1
        assert!((ScaleAlgorithm::Lanczos3.kernel(0.0) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_scale_downscale() {
        let config = ScaleConfig::new(80, 60);
        let mut filter = ScaleFilter::new(NodeId(0), "scale", config);

        let input = create_test_frame(160, 120);
        let result = filter.scale_frame(&input);

        assert_eq!(result.width, 80);
        assert_eq!(result.height, 60);
        assert_eq!(result.planes.len(), input.planes.len());
    }

    #[test]
    fn test_scale_upscale() {
        let config = ScaleConfig::new(320, 240);
        let mut filter = ScaleFilter::new(NodeId(0), "scale", config);

        let input = create_test_frame(160, 120);
        let result = filter.scale_frame(&input);

        assert_eq!(result.width, 320);
        assert_eq!(result.height, 240);
    }

    #[test]
    fn test_nearest_neighbor_scaler() {
        let scaler = NearestNeighborScaler::new(320, 240);

        let src_data = vec![128u8; 640 * 480];
        let src_plane = Plane::new(src_data, 640);

        let result = scaler.scale_plane(&src_plane, 640, 480);
        assert_eq!(result.stride, 320);
    }

    #[test]
    fn test_bilinear_scaler() {
        let scaler = BilinearScaler::new(320, 240);

        let src_data = vec![128u8; 640 * 480];
        let src_plane = Plane::new(src_data, 640);

        let result = scaler.scale_plane(&src_plane, 640, 480);
        assert_eq!(result.stride, 320);
    }

    #[test]
    fn test_aspect_fit() {
        // 16:9 source into 4:3 container
        let (w, h) = calculate_aspect_fit(1920, 1080, 640, 480);
        assert_eq!(w, 640);
        assert!(h <= 480);

        // 4:3 source into 16:9 container
        let (w, h) = calculate_aspect_fit(640, 480, 1920, 1080);
        assert!(w <= 1920);
        assert_eq!(h, 1080);
    }

    #[test]
    fn test_aspect_fill() {
        // 16:9 source into 4:3 container (will crop width)
        let (w, h) = calculate_aspect_fill(1920, 1080, 640, 480);
        assert!(w >= 640 || h >= 480);
    }

    #[test]
    fn test_sinc_function() {
        assert!((sinc(0.0) - 1.0).abs() < 0.001);
        // sinc(1) should be 0
        assert!(sinc(1.0).abs() < 0.001);
    }

    #[test]
    fn test_node_trait_implementation() {
        let config = ScaleConfig::new(1280, 720);
        let mut filter = ScaleFilter::new(NodeId(42), "test_scale", config);

        assert_eq!(filter.node_type(), NodeType::Filter);
        assert_eq!(filter.state(), NodeState::Idle);
        assert_eq!(filter.inputs().len(), 1);
        assert_eq!(filter.outputs().len(), 1);

        filter
            .set_state(NodeState::Processing)
            .expect("set_state should succeed");
        assert_eq!(filter.state(), NodeState::Processing);
    }

    #[test]
    fn test_process_none_input() {
        let config = ScaleConfig::new(1280, 720);
        let mut filter = ScaleFilter::new(NodeId(0), "scale", config);

        let result = filter.process(None).expect("process should succeed");
        assert!(result.is_none());
    }

    #[test]
    fn test_scale_config_builder() {
        let config = ScaleConfig::new(1920, 1080)
            .with_algorithm(ScaleAlgorithm::CatmullRom)
            .with_antialias(false)
            .with_preserve_aspect(true);

        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
        assert_eq!(config.algorithm, ScaleAlgorithm::CatmullRom);
        assert!(!config.antialias);
        assert!(config.preserve_aspect);
    }
}
