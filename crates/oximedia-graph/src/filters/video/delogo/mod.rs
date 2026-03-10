//! Video delogo filter.
//!
//! This filter removes logos and watermarks from video frames using various
//! techniques including blur, inpainting, texture synthesis, and temporal
//! interpolation.
//!
//! # Features
//!
//! - Manual region specification with bounding boxes
//! - Template matching for automatic logo detection
//! - Logo tracking across frames
//! - Multi-logo support
//! - Multiple removal techniques:
//!   - Simple blur
//!   - PDE-based inpainting (Navier-Stokes)
//!   - Fast marching method
//!   - Exemplar-based inpainting
//!   - Patch-based texture synthesis
//!   - Edge-aware interpolation
//!   - Temporal coherence using neighboring frames
//! - Alpha blending with configurable strength
//! - Feathered edges for smooth transitions
//! - Adaptive blending based on content
//! - Semi-transparent logo handling
//!
//! # Example
//!
//! ```ignore
//! use oximedia_graph::filters::video::{DelogoFilter, DelogoConfig, DelogoMethod, Rectangle};
//! use oximedia_graph::node::NodeId;
//!
//! // Create a delogo filter for a watermark in the top-right corner
//! let region = Rectangle::new(1600, 50, 200, 100);
//! let config = DelogoConfig::new(region, DelogoMethod::Inpainting)
//!     .with_feather(10)
//!     .with_strength(1.0);
//!
//! let filter = DelogoFilter::new(NodeId(0), "delogo", config);
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
#![allow(dead_code)]

use std::collections::VecDeque;

use crate::error::{GraphError, GraphResult};
use crate::frame::FilterFrame;
use crate::node::{Node, NodeId, NodeState, NodeType};
use crate::port::{InputPort, OutputPort, PortFormat, PortId, PortType, VideoPortFormat};
use oximedia_codec::{Plane, VideoFrame};

pub mod advanced_inpainting;
pub mod algorithms;
pub mod color;
pub mod detection;
pub mod inpainting;
pub mod mask;
pub mod metrics;

/// Rectangle region for logo specification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rectangle {
    /// X coordinate of top-left corner.
    pub x: u32,
    /// Y coordinate of top-left corner.
    pub y: u32,
    /// Width of the rectangle.
    pub width: u32,
    /// Height of the rectangle.
    pub height: u32,
}

impl Rectangle {
    /// Create a new rectangle.
    #[must_use]
    pub fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Get the right edge coordinate.
    #[must_use]
    pub fn right(&self) -> u32 {
        self.x + self.width
    }

    /// Get the bottom edge coordinate.
    #[must_use]
    pub fn bottom(&self) -> u32 {
        self.y + self.height
    }

    /// Check if a point is inside the rectangle.
    #[must_use]
    pub fn contains(&self, x: u32, y: u32) -> bool {
        x >= self.x && x < self.right() && y >= self.y && y < self.bottom()
    }

    /// Expand the rectangle by a given amount.
    #[must_use]
    pub fn expand(&self, amount: u32) -> Self {
        Self {
            x: self.x.saturating_sub(amount),
            y: self.y.saturating_sub(amount),
            width: self.width + amount * 2,
            height: self.height + amount * 2,
        }
    }

    /// Clamp the rectangle to fit within given dimensions.
    #[must_use]
    pub fn clamp(&self, max_width: u32, max_height: u32) -> Self {
        let x = self.x.min(max_width.saturating_sub(1));
        let y = self.y.min(max_height.saturating_sub(1));
        let width = self.width.min(max_width.saturating_sub(x));
        let height = self.height.min(max_height.saturating_sub(y));

        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Get the area of the rectangle.
    #[must_use]
    pub fn area(&self) -> u32 {
        self.width * self.height
    }

    /// Scale the rectangle for chroma planes.
    #[must_use]
    pub fn scale_for_chroma(&self, h_ratio: u32, v_ratio: u32) -> Self {
        Self {
            x: self.x / h_ratio,
            y: self.y / v_ratio,
            width: self.width / h_ratio,
            height: self.height / v_ratio,
        }
    }
}

/// Logo removal method.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DelogoMethod {
    /// Simple Gaussian blur.
    Blur,
    /// PDE-based inpainting (Navier-Stokes).
    #[default]
    Inpainting,
    /// Fast marching method inpainting.
    FastMarching,
    /// Exemplar-based inpainting (patch matching).
    ExemplarBased,
    /// Patch-based texture synthesis.
    TextureSynthesis,
    /// Edge-aware interpolation.
    EdgeAware,
    /// Temporal interpolation using neighboring frames.
    TemporalInterpolation,
}

/// Logo detection mode.
#[derive(Clone, Debug, PartialEq)]
pub enum LogoDetection {
    /// Manual specification of logo region.
    Manual(Rectangle),
    /// Template matching with a reference template.
    Template {
        /// Template image data.
        template: Vec<u8>,
        /// Template width.
        width: u32,
        /// Template height.
        height: u32,
        /// Detection threshold (0.0-1.0).
        threshold: f32,
    },
    /// Automatic detection and tracking.
    Automatic {
        /// Initial search region.
        search_region: Rectangle,
        /// Detection sensitivity.
        sensitivity: f32,
    },
}

/// Configuration for the delogo filter.
#[derive(Clone, Debug)]
pub struct DelogoConfig {
    /// Logo regions to remove (supports multiple logos).
    pub regions: Vec<Rectangle>,
    /// Detection mode for automatic logo detection.
    pub detection: Option<LogoDetection>,
    /// Removal method.
    pub method: DelogoMethod,
    /// Blend strength (0.0 = no effect, 1.0 = full removal).
    pub strength: f32,
    /// Feather radius for edge blending (pixels).
    pub feather: u32,
    /// Enable temporal coherence.
    pub temporal_coherence: bool,
    /// Number of frames to use for temporal processing.
    pub temporal_radius: usize,
    /// Enable edge preservation.
    pub preserve_edges: bool,
    /// Inpainting iterations (for iterative methods).
    pub iterations: u32,
    /// Patch size for texture synthesis methods.
    pub patch_size: u32,
}

impl DelogoConfig {
    /// Create a new delogo configuration with a single region.
    #[must_use]
    pub fn new(region: Rectangle, method: DelogoMethod) -> Self {
        Self {
            regions: vec![region],
            detection: None,
            method,
            strength: 1.0,
            feather: 5,
            temporal_coherence: false,
            temporal_radius: 2,
            preserve_edges: true,
            iterations: 100,
            patch_size: 7,
        }
    }

    /// Create a configuration with multiple regions.
    #[must_use]
    pub fn with_regions(regions: Vec<Rectangle>, method: DelogoMethod) -> Self {
        Self {
            regions,
            detection: None,
            method,
            strength: 1.0,
            feather: 5,
            temporal_coherence: false,
            temporal_radius: 2,
            preserve_edges: true,
            iterations: 100,
            patch_size: 7,
        }
    }

    /// Enable automatic logo detection.
    #[must_use]
    pub fn with_detection(mut self, detection: LogoDetection) -> Self {
        self.detection = Some(detection);
        self
    }

    /// Set the blend strength.
    #[must_use]
    pub fn with_strength(mut self, strength: f32) -> Self {
        self.strength = strength.clamp(0.0, 1.0);
        self
    }

    /// Set the feather radius.
    #[must_use]
    pub fn with_feather(mut self, feather: u32) -> Self {
        self.feather = feather;
        self
    }

    /// Enable temporal coherence.
    #[must_use]
    pub fn with_temporal_coherence(mut self, radius: usize) -> Self {
        self.temporal_coherence = true;
        self.temporal_radius = radius.max(1);
        self
    }

    /// Set edge preservation mode.
    #[must_use]
    pub fn with_edge_preservation(mut self, enabled: bool) -> Self {
        self.preserve_edges = enabled;
        self
    }

    /// Set the number of inpainting iterations.
    #[must_use]
    pub fn with_iterations(mut self, iterations: u32) -> Self {
        self.iterations = iterations.max(1);
        self
    }

    /// Set the patch size for texture synthesis.
    #[must_use]
    pub fn with_patch_size(mut self, size: u32) -> Self {
        self.patch_size = size.clamp(3, 64);
        self
    }
}

/// Video delogo filter.
///
/// Removes logos and watermarks from video frames using various advanced
/// techniques. Supports multiple logos, temporal coherence, and edge-aware
/// processing for natural-looking results.
pub struct DelogoFilter {
    pub(crate) id: NodeId,
    pub(crate) name: String,
    pub(crate) state: NodeState,
    pub(crate) inputs: Vec<InputPort>,
    pub(crate) outputs: Vec<OutputPort>,
    pub(crate) config: DelogoConfig,
    /// Frame buffer for temporal processing.
    pub(crate) frame_buffer: VecDeque<VideoFrame>,
    /// Logo tracker for automatic detection.
    pub(crate) tracker: Option<LogoTracker>,
}

impl DelogoFilter {
    /// Create a new delogo filter.
    #[must_use]
    pub fn new(id: NodeId, name: impl Into<String>, config: DelogoConfig) -> Self {
        Self {
            id,
            name: name.into(),
            state: NodeState::Idle,
            inputs: vec![InputPort::new(PortId(0), "input", PortType::Video)
                .with_format(PortFormat::Video(VideoPortFormat::any()))],
            outputs: vec![OutputPort::new(PortId(0), "output", PortType::Video)
                .with_format(PortFormat::Video(VideoPortFormat::any()))],
            config,
            frame_buffer: VecDeque::new(),
            tracker: None,
        }
    }

    /// Get the current configuration.
    #[must_use]
    pub fn config(&self) -> &DelogoConfig {
        &self.config
    }

    /// Update the configuration.
    pub fn set_config(&mut self, config: DelogoConfig) {
        self.config = config;
        self.frame_buffer.clear();
        self.tracker = None;
    }

    /// Add a logo region.
    pub fn add_region(&mut self, region: Rectangle) {
        self.config.regions.push(region);
    }

    /// Clear all logo regions.
    pub fn clear_regions(&mut self) {
        self.config.regions.clear();
    }

    /// Process a single frame.
    pub(crate) fn process_frame(&mut self, mut frame: VideoFrame) -> GraphResult<VideoFrame> {
        // Update temporal buffer
        if self.config.temporal_coherence {
            self.frame_buffer.push_back(frame.clone());
            if self.frame_buffer.len() > self.config.temporal_radius * 2 + 1 {
                self.frame_buffer.pop_front();
            }
        }

        // Detect logos if automatic detection is enabled
        let detected_regions = if let Some(detection) = self.config.detection.clone() {
            self.detect_logos(&frame, &detection)
        } else {
            None
        };

        if let Some(regions) = detected_regions {
            self.config.regions = regions;
        }

        // Process each logo region
        let regions = self.config.regions.clone();
        for region in &regions {
            let clamped = region.clamp(frame.width, frame.height);
            self.remove_logo(&mut frame, &clamped)?;
        }

        Ok(frame)
    }

    /// Remove a logo from a specific region.
    fn remove_logo(&self, frame: &mut VideoFrame, region: &Rectangle) -> GraphResult<()> {
        // Get format information
        let (h_sub, v_sub) = frame.format.chroma_subsampling();

        // Process each plane
        for (plane_idx, plane) in frame.planes.iter_mut().enumerate() {
            let plane_region = if plane_idx > 0 && frame.format.is_yuv() {
                region.scale_for_chroma(h_sub, v_sub)
            } else {
                *region
            };

            let (plane_width, plane_height) = if plane_idx == 0 {
                (frame.width, frame.height)
            } else {
                (frame.width / h_sub, frame.height / v_sub)
            };

            self.process_plane(plane, &plane_region, plane_width, plane_height)?;
        }

        Ok(())
    }

    /// Process a single plane for logo removal.
    fn process_plane(
        &self,
        plane: &mut Plane,
        region: &Rectangle,
        width: u32,
        height: u32,
    ) -> GraphResult<()> {
        // Create a working buffer
        let mut data = plane.data.to_vec();

        match self.config.method {
            DelogoMethod::Blur => {
                self.apply_blur(&mut data, region, width, height);
            }
            DelogoMethod::Inpainting => {
                self.apply_inpainting(&mut data, region, width, height);
            }
            DelogoMethod::FastMarching => {
                self.apply_fast_marching(&mut data, region, width, height);
            }
            DelogoMethod::ExemplarBased => {
                self.apply_exemplar_based(&mut data, region, width, height);
            }
            DelogoMethod::TextureSynthesis => {
                self.apply_texture_synthesis(&mut data, region, width, height);
            }
            DelogoMethod::EdgeAware => {
                self.apply_edge_aware(&mut data, region, width, height);
            }
            DelogoMethod::TemporalInterpolation => {
                self.apply_temporal_interpolation(&mut data, region, width, height);
            }
        }

        // Apply feathering/blending
        if self.config.feather > 0 {
            self.apply_feathering(&mut data, plane, region, width, height);
        }

        // Update plane data
        *plane = Plane::new(data, plane.stride);

        Ok(())
    }
}

impl Node for DelogoFilter {
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
                let processed = self.process_frame(frame)?;
                Ok(Some(FilterFrame::Video(processed)))
            }
            Some(_) => Err(GraphError::PortTypeMismatch {
                expected: "Video".to_string(),
                actual: "Audio".to_string(),
            }),
            None => Ok(None),
        }
    }

    fn reset(&mut self) -> GraphResult<()> {
        self.frame_buffer.clear();
        self.tracker = None;
        self.set_state(NodeState::Idle)
    }
}

/// Logo tracker for multi-frame tracking.
#[derive(Debug)]
pub(crate) struct LogoTracker {
    /// Tracked regions.
    pub(crate) regions: Vec<TrackedRegion>,
    /// Maximum tracking distance per frame.
    pub(crate) max_distance: f32,
}

impl LogoTracker {
    /// Create a new logo tracker.
    #[must_use]
    pub fn new() -> Self {
        Self {
            regions: Vec::new(),
            max_distance: 20.0,
        }
    }

    /// Update tracked regions with new detections.
    pub fn update(&mut self, detected: Vec<Rectangle>) {
        // Simple tracking: match by proximity
        for region in detected {
            let mut matched = false;

            for tracked in &mut self.regions {
                if tracked.matches(&region, self.max_distance) {
                    tracked.update(region);
                    matched = true;
                    break;
                }
            }

            if !matched {
                self.regions.push(TrackedRegion::new(region));
            }
        }

        // Remove stale tracks
        self.regions.retain(|r| r.confidence > 0.3);
    }

    /// Get current tracked regions.
    #[must_use]
    pub fn regions(&self) -> Vec<Rectangle> {
        self.regions.iter().map(|r| r.region).collect()
    }
}

/// A tracked logo region.
#[derive(Debug)]
pub(crate) struct TrackedRegion {
    /// Current region.
    pub(crate) region: Rectangle,
    /// Tracking confidence (0.0-1.0).
    pub(crate) confidence: f32,
    /// Number of frames tracked.
    pub(crate) age: u32,
}

impl TrackedRegion {
    /// Create a new tracked region.
    #[must_use]
    pub fn new(region: Rectangle) -> Self {
        Self {
            region,
            confidence: 1.0,
            age: 0,
        }
    }

    /// Check if a detection matches this tracked region.
    #[must_use]
    pub fn matches(&self, other: &Rectangle, max_distance: f32) -> bool {
        let dx = (self.region.x as f32 - other.x as f32).abs();
        let dy = (self.region.y as f32 - other.y as f32).abs();
        let distance = (dx * dx + dy * dy).sqrt();

        distance < max_distance
    }

    /// Update with a new detection.
    pub fn update(&mut self, region: Rectangle) {
        // Smooth position update
        let alpha = 0.3;
        self.region.x = (self.region.x as f32 * (1.0 - alpha) + region.x as f32 * alpha) as u32;
        self.region.y = (self.region.y as f32 * (1.0 - alpha) + region.y as f32 * alpha) as u32;
        self.region.width =
            (self.region.width as f32 * (1.0 - alpha) + region.width as f32 * alpha) as u32;
        self.region.height =
            (self.region.height as f32 * (1.0 - alpha) + region.height as f32 * alpha) as u32;

        self.confidence = (self.confidence * 0.9 + 0.1).min(1.0);
        self.age += 1;
    }
}

/// Ordered float for priority queue.
#[derive(Debug, Clone, Copy)]
pub(crate) struct OrderedFloat(pub f32, pub usize);

impl PartialEq for OrderedFloat {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for OrderedFloat {}

impl PartialOrd for OrderedFloat {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Create a Gaussian kernel for blurring.
pub(crate) fn create_gaussian_kernel(radius: usize, sigma: f32) -> Vec<f32> {
    let size = radius * 2 + 1;
    let mut kernel = vec![0.0f32; size * size];
    let mut sum = 0.0f32;

    let sigma_sq = sigma * sigma;

    for y in 0..size {
        for x in 0..size {
            let dx = x as f32 - radius as f32;
            let dy = y as f32 - radius as f32;
            let dist_sq = dx * dx + dy * dy;

            let val = (-dist_sq / (2.0 * sigma_sq)).exp();
            kernel[y * size + x] = val;
            sum += val;
        }
    }

    // Normalize
    for val in &mut kernel {
        *val /= sum;
    }

    kernel
}
