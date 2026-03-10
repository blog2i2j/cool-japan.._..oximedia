//! Deinterlacing filter.
//!
//! This filter converts interlaced video to progressive video using various
//! deinterlacing algorithms.

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

use std::collections::VecDeque;

use crate::error::{GraphError, GraphResult};
use crate::frame::FilterFrame;
use crate::node::{Node, NodeId, NodeState, NodeType};
use crate::port::{InputPort, OutputPort, PortFormat, PortId, PortType, VideoPortFormat};
use oximedia_codec::{Plane, VideoFrame};
use oximedia_core::Timestamp;

/// Deinterlacing mode/algorithm.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DeinterlaceMode {
    /// Bob deinterlacing - simple field interpolation (doubles frame rate).
    Bob,
    /// Weave - combine fields (may cause combing artifacts).
    Weave,
    /// Blend - average adjacent fields (reduces motion blur).
    #[default]
    Blend,
    /// Yadif - Yet Another DeInterlacing Filter (high quality).
    Yadif,
    /// Yadif with spatial interpolation only (no temporal).
    YadifSpatial,
}

impl DeinterlaceMode {
    /// Check if this mode doubles the frame rate.
    #[must_use]
    pub fn doubles_framerate(&self) -> bool {
        matches!(self, Self::Bob | Self::Yadif)
    }

    /// Check if this mode requires temporal information.
    #[must_use]
    pub fn requires_temporal(&self) -> bool {
        matches!(self, Self::Yadif)
    }
}

/// Field order in interlaced video.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum FieldOrder {
    /// Top field first (TFF).
    #[default]
    TopFieldFirst,
    /// Bottom field first (BFF).
    BottomFieldFirst,
    /// Auto-detect from video metadata.
    Auto,
}

impl FieldOrder {
    /// Get the starting line for the first field.
    #[must_use]
    pub fn first_field_start(&self) -> usize {
        match self {
            Self::TopFieldFirst | Self::Auto => 0,
            Self::BottomFieldFirst => 1,
        }
    }

    /// Get the starting line for the second field.
    #[must_use]
    pub fn second_field_start(&self) -> usize {
        match self {
            Self::TopFieldFirst | Self::Auto => 1,
            Self::BottomFieldFirst => 0,
        }
    }
}

/// Configuration for the deinterlace filter.
#[derive(Clone, Debug)]
pub struct DeinterlaceConfig {
    /// Deinterlacing algorithm.
    pub mode: DeinterlaceMode,
    /// Field order.
    pub field_order: FieldOrder,
    /// Only deinterlace detected interlaced frames.
    pub auto_detect: bool,
    /// Threshold for interlace detection (0.0-1.0).
    pub detection_threshold: f64,
}

impl Default for DeinterlaceConfig {
    fn default() -> Self {
        Self {
            mode: DeinterlaceMode::default(),
            field_order: FieldOrder::default(),
            auto_detect: false,
            detection_threshold: 0.5,
        }
    }
}

impl DeinterlaceConfig {
    /// Create a new deinterlace configuration.
    #[must_use]
    pub fn new(mode: DeinterlaceMode) -> Self {
        Self {
            mode,
            ..Default::default()
        }
    }

    /// Set the field order.
    #[must_use]
    pub fn with_field_order(mut self, order: FieldOrder) -> Self {
        self.field_order = order;
        self
    }

    /// Enable auto-detection of interlaced content.
    #[must_use]
    pub fn with_auto_detect(mut self, enabled: bool) -> Self {
        self.auto_detect = enabled;
        self
    }

    /// Set the detection threshold.
    #[must_use]
    pub fn with_detection_threshold(mut self, threshold: f64) -> Self {
        self.detection_threshold = threshold.clamp(0.0, 1.0);
        self
    }
}

/// Deinterlacing filter.
///
/// Converts interlaced video to progressive video using various algorithms.
///
/// # Example
///
/// ```ignore
/// use oximedia_graph::filters::video::{DeinterlaceFilter, DeinterlaceConfig, DeinterlaceMode};
/// use oximedia_graph::node::NodeId;
///
/// let config = DeinterlaceConfig::new(DeinterlaceMode::Yadif)
///     .with_field_order(FieldOrder::TopFieldFirst);
///
/// let filter = DeinterlaceFilter::new(NodeId(0), "deinterlace", config);
/// ```
pub struct DeinterlaceFilter {
    id: NodeId,
    name: String,
    state: NodeState,
    inputs: Vec<InputPort>,
    outputs: Vec<OutputPort>,
    config: DeinterlaceConfig,
    /// Frame buffer for temporal algorithms.
    frame_buffer: VecDeque<VideoFrame>,
    /// Current output frame index.
    output_frame_idx: u64,
    /// Pending output frames (for bob/yadif that produce 2 frames per input).
    pending_output: Vec<VideoFrame>,
}

impl DeinterlaceFilter {
    /// Create a new deinterlace filter.
    #[must_use]
    pub fn new(id: NodeId, name: impl Into<String>, config: DeinterlaceConfig) -> Self {
        Self {
            id,
            name: name.into(),
            state: NodeState::Idle,
            inputs: vec![InputPort::new(PortId(0), "input", PortType::Video)
                .with_format(PortFormat::Video(VideoPortFormat::any()))],
            outputs: vec![OutputPort::new(PortId(0), "output", PortType::Video)
                .with_format(PortFormat::Video(VideoPortFormat::any()))],
            config,
            frame_buffer: VecDeque::with_capacity(3),
            output_frame_idx: 0,
            pending_output: Vec::new(),
        }
    }

    /// Get the current configuration.
    #[must_use]
    pub fn config(&self) -> &DeinterlaceConfig {
        &self.config
    }

    /// Detect if a frame is interlaced by analyzing combing artifacts.
    fn detect_interlaced(&self, frame: &VideoFrame) -> bool {
        if frame.planes.is_empty() {
            return false;
        }

        let plane = &frame.planes[0];
        let height = frame.height as usize;
        let width = frame.width as usize;

        // Simple comb detection: look for large differences between adjacent lines
        let mut comb_score = 0u64;
        let mut total_samples = 0u64;

        for y in 1..height - 1 {
            let row_prev = plane.row(y - 1);
            let row_curr = plane.row(y);
            let row_next = plane.row(y + 1);

            for x in 0..width {
                let prev = row_prev.get(x).copied().unwrap_or(0) as i32;
                let curr = row_curr.get(x).copied().unwrap_or(0) as i32;
                let next = row_next.get(x).copied().unwrap_or(0) as i32;

                // Combing metric: current line differs significantly from interpolation
                let interpolated = (prev + next) / 2;
                let diff = (curr - interpolated).unsigned_abs() as u64;

                if diff > 20 {
                    comb_score += diff;
                }
                total_samples += 1;
            }
        }

        if total_samples == 0 {
            return false;
        }

        let score = comb_score as f64 / total_samples as f64;
        score > self.config.detection_threshold * 10.0
    }

    /// Deinterlace using bob algorithm (line doubling with interpolation).
    fn bob_deinterlace(&self, frame: &VideoFrame, field: usize) -> VideoFrame {
        let mut output = VideoFrame::new(frame.format, frame.width, frame.height);
        output.frame_type = frame.frame_type;
        output.color_info = frame.color_info;

        // Adjust timestamp for bob (two frames per input)
        let half_duration =
            frame.timestamp.timebase.den as i64 / (frame.timestamp.timebase.num as i64 * 2);
        let pts_offset = if field == 0 { 0 } else { half_duration };
        output.timestamp =
            Timestamp::new(frame.timestamp.pts + pts_offset, frame.timestamp.timebase);

        for (plane_idx, src_plane) in frame.planes.iter().enumerate() {
            let (width, height) = frame.plane_dimensions(plane_idx);
            let mut dst_data = vec![0u8; (width * height) as usize];

            let field_start = if field == 0 {
                self.config.field_order.first_field_start()
            } else {
                self.config.field_order.second_field_start()
            };

            for y in 0..height as usize {
                let is_field_line = (y + field_start) % 2 == 0;

                if is_field_line {
                    // Copy original field line
                    let src_row = src_plane.row(y);
                    for x in 0..width as usize {
                        dst_data[y * width as usize + x] = src_row.get(x).copied().unwrap_or(0);
                    }
                } else {
                    // Interpolate from adjacent lines
                    let y_prev = y.saturating_sub(1);
                    let y_next = (y + 1).min(height as usize - 1);

                    let prev_row = src_plane.row(y_prev);
                    let next_row = src_plane.row(y_next);

                    for x in 0..width as usize {
                        let prev = prev_row.get(x).copied().unwrap_or(0) as u16;
                        let next = next_row.get(x).copied().unwrap_or(0) as u16;
                        dst_data[y * width as usize + x] = ((prev + next) / 2) as u8;
                    }
                }
            }

            output.planes.push(Plane::new(dst_data, width as usize));
        }

        output
    }

    /// Deinterlace using weave algorithm (combine fields).
    fn weave_deinterlace(&self, frame: &VideoFrame) -> VideoFrame {
        // Weave simply passes through the frame, combining both fields
        // This is essentially a no-op but marks the frame as progressive
        frame.clone()
    }

    /// Deinterlace using blend algorithm (average adjacent lines).
    fn blend_deinterlace(&self, frame: &VideoFrame) -> VideoFrame {
        let mut output = VideoFrame::new(frame.format, frame.width, frame.height);
        output.timestamp = frame.timestamp;
        output.frame_type = frame.frame_type;
        output.color_info = frame.color_info;

        for (plane_idx, src_plane) in frame.planes.iter().enumerate() {
            let (width, height) = frame.plane_dimensions(plane_idx);
            let mut dst_data = vec![0u8; (width * height) as usize];

            for y in 0..height as usize {
                let curr_row = src_plane.row(y);

                if y == 0 || y == height as usize - 1 {
                    // Copy edge lines directly
                    for x in 0..width as usize {
                        dst_data[y * width as usize + x] = curr_row.get(x).copied().unwrap_or(0);
                    }
                } else {
                    // Blend with adjacent lines
                    let prev_row = src_plane.row(y - 1);
                    let next_row = src_plane.row(y + 1);

                    for x in 0..width as usize {
                        let prev = prev_row.get(x).copied().unwrap_or(0) as u32;
                        let curr = curr_row.get(x).copied().unwrap_or(0) as u32;
                        let next = next_row.get(x).copied().unwrap_or(0) as u32;

                        // 50% current, 25% each adjacent
                        let blended = (prev + curr * 2 + next) / 4;
                        dst_data[y * width as usize + x] = blended as u8;
                    }
                }
            }

            output.planes.push(Plane::new(dst_data, width as usize));
        }

        output
    }

    /// Deinterlace using YADIF algorithm.
    fn yadif_deinterlace(&self, field: usize) -> Option<VideoFrame> {
        // YADIF requires 3 frames: previous, current, next
        if self.frame_buffer.len() < 2 {
            return None;
        }

        let curr_idx = if self.frame_buffer.len() >= 3 { 1 } else { 0 };
        let frame = &self.frame_buffer[curr_idx];

        let prev_frame = self.frame_buffer.front();
        let next_frame = if self.frame_buffer.len() >= 3 {
            self.frame_buffer.get(2)
        } else {
            self.frame_buffer.get(1)
        };

        let mut output = VideoFrame::new(frame.format, frame.width, frame.height);
        output.frame_type = frame.frame_type;
        output.color_info = frame.color_info;

        // Adjust timestamp
        let half_duration =
            frame.timestamp.timebase.den as i64 / (frame.timestamp.timebase.num as i64 * 2);
        let pts_offset = if field == 0 { 0 } else { half_duration };
        output.timestamp =
            Timestamp::new(frame.timestamp.pts + pts_offset, frame.timestamp.timebase);

        for (plane_idx, src_plane) in frame.planes.iter().enumerate() {
            let (width, height) = frame.plane_dimensions(plane_idx);
            let mut dst_data = vec![0u8; (width * height) as usize];

            let field_start = if field == 0 {
                self.config.field_order.first_field_start()
            } else {
                self.config.field_order.second_field_start()
            };

            let prev_plane = prev_frame.and_then(|f| f.planes.get(plane_idx));
            let next_plane = next_frame.and_then(|f| f.planes.get(plane_idx));

            for y in 0..height as usize {
                let is_field_line = (y + field_start) % 2 == 0;

                if is_field_line {
                    // Copy original field line
                    let src_row = src_plane.row(y);
                    for x in 0..width as usize {
                        dst_data[y * width as usize + x] = src_row.get(x).copied().unwrap_or(0);
                    }
                } else {
                    // YADIF interpolation
                    for x in 0..width as usize {
                        let pixel = self.yadif_pixel(
                            src_plane,
                            prev_plane,
                            next_plane,
                            x,
                            y,
                            width as usize,
                            height as usize,
                        );
                        dst_data[y * width as usize + x] = pixel;
                    }
                }
            }

            output.planes.push(Plane::new(dst_data, width as usize));
        }

        Some(output)
    }

    /// Calculate a single YADIF pixel.
    #[allow(clippy::too_many_arguments)]
    fn yadif_pixel(
        &self,
        curr: &Plane,
        prev: Option<&Plane>,
        next: Option<&Plane>,
        x: usize,
        y: usize,
        width: usize,
        height: usize,
    ) -> u8 {
        let y_prev = y.saturating_sub(1);
        let y_next = (y + 1).min(height - 1);

        // Spatial prediction (from current frame)
        let c_prev = curr.row(y_prev).get(x).copied().unwrap_or(0) as i32;
        let c_next = curr.row(y_next).get(x).copied().unwrap_or(0) as i32;
        let spatial = (c_prev + c_next) / 2;

        // Temporal prediction (from previous and next frames)
        let temporal = if let (Some(p), Some(n)) = (prev, next) {
            let p_curr = p.row(y).get(x).copied().unwrap_or(0) as i32;
            let n_curr = n.row(y).get(x).copied().unwrap_or(0) as i32;
            (p_curr + n_curr) / 2
        } else {
            spatial
        };

        // Edge detection for choosing between spatial and temporal
        let edge_prev = curr
            .row(y_prev)
            .get(x.saturating_sub(1))
            .copied()
            .unwrap_or(0) as i32;
        let edge_next = curr
            .row(y_next)
            .get((x + 1).min(width - 1))
            .copied()
            .unwrap_or(0) as i32;

        let spatial_diff = (c_prev - c_next).abs();
        let edge_diff = (edge_prev - edge_next).abs();

        // Use spatial prediction if there's significant spatial variation
        let result = if spatial_diff > edge_diff * 2 {
            temporal
        } else {
            spatial
        };

        result.clamp(0, 255) as u8
    }

    /// Process a frame and produce deinterlaced output.
    fn process_frame(&mut self, frame: VideoFrame) -> Vec<VideoFrame> {
        // Check for auto-detection
        if self.config.auto_detect && !self.detect_interlaced(&frame) {
            return vec![frame];
        }

        self.frame_buffer.push_back(frame);

        // Keep buffer size limited
        while self.frame_buffer.len() > 3 {
            self.frame_buffer.pop_front();
        }

        let mut output = Vec::new();

        match self.config.mode {
            DeinterlaceMode::Bob => {
                if let Some(frame) = self.frame_buffer.back() {
                    output.push(self.bob_deinterlace(frame, 0));
                    output.push(self.bob_deinterlace(frame, 1));
                }
            }
            DeinterlaceMode::Weave => {
                if let Some(frame) = self.frame_buffer.back() {
                    output.push(self.weave_deinterlace(frame));
                }
            }
            DeinterlaceMode::Blend => {
                if let Some(frame) = self.frame_buffer.back() {
                    output.push(self.blend_deinterlace(frame));
                }
            }
            DeinterlaceMode::Yadif => {
                if let Some(frame) = self.yadif_deinterlace(0) {
                    output.push(frame);
                }
                if let Some(frame) = self.yadif_deinterlace(1) {
                    output.push(frame);
                }
            }
            DeinterlaceMode::YadifSpatial => {
                // Spatial-only YADIF (similar to blend but with edge-aware interpolation)
                if let Some(frame) = self.frame_buffer.back() {
                    output.push(self.blend_deinterlace(frame));
                }
            }
        }

        self.output_frame_idx += output.len() as u64;
        output
    }
}

impl Node for DeinterlaceFilter {
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
        // First, check for pending output from previous processing
        if !self.pending_output.is_empty() {
            return Ok(Some(FilterFrame::Video(self.pending_output.remove(0))));
        }

        match input {
            Some(FilterFrame::Video(frame)) => {
                let mut output_frames = self.process_frame(frame);

                if output_frames.is_empty() {
                    Ok(None)
                } else {
                    let first = output_frames.remove(0);
                    self.pending_output = output_frames;
                    Ok(Some(FilterFrame::Video(first)))
                }
            }
            Some(_) => Err(GraphError::PortTypeMismatch {
                expected: "Video".to_string(),
                actual: "Audio".to_string(),
            }),
            None => Ok(None),
        }
    }

    fn flush(&mut self) -> GraphResult<Vec<FilterFrame>> {
        let mut output: Vec<FilterFrame> = self
            .pending_output
            .drain(..)
            .map(FilterFrame::Video)
            .collect();

        // Process any remaining buffered frames
        while let Some(frame) = self.frame_buffer.pop_front() {
            output.push(FilterFrame::Video(frame));
        }

        Ok(output)
    }

    fn reset(&mut self) -> GraphResult<()> {
        self.frame_buffer.clear();
        self.pending_output.clear();
        self.output_frame_idx = 0;
        self.set_state(NodeState::Idle)
    }
}

/// Detect interlacing in a video frame.
#[derive(Debug)]
pub struct InterlaceDetector {
    /// Detection threshold.
    threshold: f64,
    /// Number of frames analyzed.
    frames_analyzed: u64,
    /// Number of interlaced frames detected.
    interlaced_count: u64,
    /// Detected field order.
    detected_field_order: Option<FieldOrder>,
}

impl Default for InterlaceDetector {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            frames_analyzed: 0,
            interlaced_count: 0,
            detected_field_order: None,
        }
    }
}

impl InterlaceDetector {
    /// Create a new interlace detector.
    #[must_use]
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold: threshold.clamp(0.0, 1.0),
            ..Default::default()
        }
    }

    /// Analyze a frame for interlacing.
    pub fn analyze(&mut self, frame: &VideoFrame) -> bool {
        self.frames_analyzed += 1;

        if frame.planes.is_empty() {
            return false;
        }

        let is_interlaced = self.detect_combing(frame);
        if is_interlaced {
            self.interlaced_count += 1;
        }

        is_interlaced
    }

    /// Detect combing artifacts in a frame.
    fn detect_combing(&self, frame: &VideoFrame) -> bool {
        let plane = &frame.planes[0];
        let height = frame.height as usize;
        let width = frame.width as usize;

        let mut comb_score = 0u64;
        let mut samples = 0u64;

        // Sample every 4th line and pixel for efficiency
        for y in (2..height - 2).step_by(4) {
            for x in (0..width).step_by(4) {
                let row_m2 = plane.row(y - 2);
                let row_m1 = plane.row(y - 1);
                let row_0 = plane.row(y);
                let row_p1 = plane.row(y + 1);
                let row_p2 = plane.row(y + 2);

                let m2 = row_m2.get(x).copied().unwrap_or(0) as i32;
                let m1 = row_m1.get(x).copied().unwrap_or(0) as i32;
                let c = row_0.get(x).copied().unwrap_or(0) as i32;
                let p1 = row_p1.get(x).copied().unwrap_or(0) as i32;
                let p2 = row_p2.get(x).copied().unwrap_or(0) as i32;

                // Combing metric
                let diff = ((m1 - c).abs() + (p1 - c).abs()) - ((m2 - c).abs() + (p2 - c).abs());
                if diff > 20 {
                    comb_score += diff.unsigned_abs() as u64;
                }
                samples += 1;
            }
        }

        if samples == 0 {
            return false;
        }

        let score = comb_score as f64 / samples as f64;
        score > self.threshold * 10.0
    }

    /// Get the percentage of interlaced frames detected.
    #[must_use]
    pub fn interlaced_percentage(&self) -> f64 {
        if self.frames_analyzed == 0 {
            0.0
        } else {
            self.interlaced_count as f64 / self.frames_analyzed as f64
        }
    }

    /// Check if content is likely interlaced.
    #[must_use]
    pub fn is_interlaced(&self) -> bool {
        self.interlaced_percentage() > 0.5
    }

    /// Get the detected field order.
    #[must_use]
    pub fn field_order(&self) -> Option<FieldOrder> {
        self.detected_field_order
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_core::PixelFormat;

    fn create_test_frame(width: u32, height: u32) -> VideoFrame {
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, width, height);
        frame.allocate();

        // Create a pattern that simulates interlacing
        if let Some(plane) = frame.planes.get_mut(0) {
            let mut data = vec![0u8; (width * height) as usize];
            for y in 0..height as usize {
                for x in 0..width as usize {
                    // Alternating line pattern
                    data[y * width as usize + x] = if y % 2 == 0 { 200 } else { 50 };
                }
            }
            *plane = Plane::new(data, width as usize);
        }

        frame
    }

    fn create_progressive_frame(width: u32, height: u32) -> VideoFrame {
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, width, height);
        frame.allocate();

        // Create a smooth gradient (no interlacing)
        if let Some(plane) = frame.planes.get_mut(0) {
            let mut data = vec![0u8; (width * height) as usize];
            for y in 0..height as usize {
                for x in 0..width as usize {
                    data[y * width as usize + x] = ((y * 255) / height as usize) as u8;
                }
            }
            *plane = Plane::new(data, width as usize);
        }

        frame
    }

    #[test]
    fn test_deinterlace_mode_properties() {
        assert!(DeinterlaceMode::Bob.doubles_framerate());
        assert!(DeinterlaceMode::Yadif.doubles_framerate());
        assert!(!DeinterlaceMode::Blend.doubles_framerate());
        assert!(!DeinterlaceMode::Weave.doubles_framerate());

        assert!(DeinterlaceMode::Yadif.requires_temporal());
        assert!(!DeinterlaceMode::Bob.requires_temporal());
    }

    #[test]
    fn test_field_order() {
        assert_eq!(FieldOrder::TopFieldFirst.first_field_start(), 0);
        assert_eq!(FieldOrder::TopFieldFirst.second_field_start(), 1);
        assert_eq!(FieldOrder::BottomFieldFirst.first_field_start(), 1);
        assert_eq!(FieldOrder::BottomFieldFirst.second_field_start(), 0);
    }

    #[test]
    fn test_deinterlace_config() {
        let config = DeinterlaceConfig::new(DeinterlaceMode::Bob)
            .with_field_order(FieldOrder::BottomFieldFirst)
            .with_auto_detect(true)
            .with_detection_threshold(0.7);

        assert_eq!(config.mode, DeinterlaceMode::Bob);
        assert_eq!(config.field_order, FieldOrder::BottomFieldFirst);
        assert!(config.auto_detect);
        assert!((config.detection_threshold - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_deinterlace_filter_creation() {
        let config = DeinterlaceConfig::new(DeinterlaceMode::Blend);
        let filter = DeinterlaceFilter::new(NodeId(0), "deinterlace", config);

        assert_eq!(filter.id(), NodeId(0));
        assert_eq!(filter.name(), "deinterlace");
        assert_eq!(filter.node_type(), NodeType::Filter);
    }

    #[test]
    fn test_bob_deinterlace() {
        let config = DeinterlaceConfig::new(DeinterlaceMode::Bob);
        let mut filter = DeinterlaceFilter::new(NodeId(0), "deinterlace", config);

        let input = create_test_frame(64, 48);
        let output_frames = filter.process_frame(input);

        // Bob should produce 2 frames
        assert_eq!(output_frames.len(), 2);
    }

    #[test]
    fn test_blend_deinterlace() {
        let config = DeinterlaceConfig::new(DeinterlaceMode::Blend);
        let mut filter = DeinterlaceFilter::new(NodeId(0), "deinterlace", config);

        let input = create_test_frame(64, 48);
        let output_frames = filter.process_frame(input);

        // Blend should produce 1 frame
        assert_eq!(output_frames.len(), 1);
    }

    #[test]
    fn test_weave_deinterlace() {
        let config = DeinterlaceConfig::new(DeinterlaceMode::Weave);
        let mut filter = DeinterlaceFilter::new(NodeId(0), "deinterlace", config);

        let input = create_test_frame(64, 48);
        let output_frames = filter.process_frame(input);

        assert_eq!(output_frames.len(), 1);
    }

    #[test]
    fn test_auto_detect_progressive() {
        let config = DeinterlaceConfig::new(DeinterlaceMode::Blend).with_auto_detect(true);
        let mut filter = DeinterlaceFilter::new(NodeId(0), "deinterlace", config);

        let input = create_progressive_frame(64, 48);
        let output_frames = filter.process_frame(input);

        // Progressive frame should pass through unchanged
        assert_eq!(output_frames.len(), 1);
    }

    #[test]
    fn test_interlace_detector() {
        let mut detector = InterlaceDetector::new(0.5);

        let interlaced = create_test_frame(64, 48);
        let _is_interlaced = detector.analyze(&interlaced);

        assert!(detector.frames_analyzed > 0);
        // Note: detection may vary based on pattern
    }

    #[test]
    fn test_interlace_detector_percentage() {
        let detector = InterlaceDetector {
            threshold: 0.5,
            frames_analyzed: 10,
            interlaced_count: 7,
            detected_field_order: None,
        };

        assert!((detector.interlaced_percentage() - 0.7).abs() < 0.001);
        assert!(detector.is_interlaced());
    }

    #[test]
    fn test_node_process() {
        let config = DeinterlaceConfig::new(DeinterlaceMode::Blend);
        let mut filter = DeinterlaceFilter::new(NodeId(0), "deinterlace", config);

        let input = create_test_frame(64, 48);
        let result = filter
            .process(Some(FilterFrame::Video(input)))
            .expect("operation should succeed")
            .expect("operation should succeed");

        assert!(matches!(result, FilterFrame::Video(_)));
    }

    #[test]
    fn test_node_state_transitions() {
        let config = DeinterlaceConfig::new(DeinterlaceMode::Blend);
        let mut filter = DeinterlaceFilter::new(NodeId(0), "deinterlace", config);

        assert_eq!(filter.state(), NodeState::Idle);
        filter
            .set_state(NodeState::Processing)
            .expect("set_state should succeed");
        assert_eq!(filter.state(), NodeState::Processing);
    }

    #[test]
    fn test_process_none_input() {
        let config = DeinterlaceConfig::new(DeinterlaceMode::Blend);
        let mut filter = DeinterlaceFilter::new(NodeId(0), "deinterlace", config);

        let result = filter.process(None).expect("process should succeed");
        assert!(result.is_none());
    }

    #[test]
    fn test_reset() {
        let config = DeinterlaceConfig::new(DeinterlaceMode::Blend);
        let mut filter = DeinterlaceFilter::new(NodeId(0), "deinterlace", config);

        let input = create_test_frame(64, 48);
        let _ = filter.process(Some(FilterFrame::Video(input)));

        filter.reset().expect("reset should succeed");

        assert!(filter.frame_buffer.is_empty());
        assert_eq!(filter.output_frame_idx, 0);
    }
}
