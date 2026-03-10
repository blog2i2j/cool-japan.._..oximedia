//! Frame rate adjustment filter.
//!
//! This filter adjusts the frame rate of video streams by dropping, duplicating,
//! or blending frames as needed.

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
use oximedia_core::{Rational, Timestamp};

/// Frame rate adjustment mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum FpsMode {
    /// Drop frames when downsampling, duplicate when upsampling.
    #[default]
    DropDuplicate,
    /// Drop frames only (never duplicate).
    Drop,
    /// Duplicate frames only (never drop).
    Duplicate,
    /// Blend adjacent frames for smooth interpolation.
    Blend,
    /// Variable frame rate passthrough (adjust timestamps only).
    Vfr,
}

impl FpsMode {
    /// Check if this mode allows frame dropping.
    #[must_use]
    pub fn allows_drop(&self) -> bool {
        matches!(self, Self::DropDuplicate | Self::Drop | Self::Blend)
    }

    /// Check if this mode allows frame duplication.
    #[must_use]
    pub fn allows_duplicate(&self) -> bool {
        matches!(self, Self::DropDuplicate | Self::Duplicate | Self::Blend)
    }
}

/// Configuration for the FPS filter.
#[derive(Clone, Debug)]
pub struct FpsConfig {
    /// Target frame rate numerator.
    pub fps_num: u32,
    /// Target frame rate denominator.
    pub fps_den: u32,
    /// Frame rate adjustment mode.
    pub mode: FpsMode,
    /// Round timestamps to nearest frame (vs floor).
    pub round: bool,
    /// Start time offset in timebase units.
    pub start_time: i64,
    /// End-of-stream handling: output remaining buffered frames.
    pub eof_action: EofAction,
}

impl FpsConfig {
    /// Create a new FPS configuration with the given target frame rate.
    #[must_use]
    pub fn new(fps_num: u32, fps_den: u32) -> Self {
        Self {
            fps_num,
            fps_den,
            mode: FpsMode::default(),
            round: true,
            start_time: 0,
            eof_action: EofAction::Pass,
        }
    }

    /// Create a configuration for common frame rates.
    #[must_use]
    pub fn from_rate(fps: f64) -> Self {
        let (num, den) = rational_from_float(fps);
        Self::new(num, den)
    }

    /// Create a 24 fps configuration.
    #[must_use]
    pub fn fps_24() -> Self {
        Self::new(24, 1)
    }

    /// Create a 25 fps configuration (PAL).
    #[must_use]
    pub fn fps_25() -> Self {
        Self::new(25, 1)
    }

    /// Create a 30 fps configuration.
    #[must_use]
    pub fn fps_30() -> Self {
        Self::new(30, 1)
    }

    /// Create a 29.97 fps configuration (NTSC).
    #[must_use]
    pub fn fps_29_97() -> Self {
        Self::new(30000, 1001)
    }

    /// Create a 60 fps configuration.
    #[must_use]
    pub fn fps_60() -> Self {
        Self::new(60, 1)
    }

    /// Create a 59.94 fps configuration (NTSC).
    #[must_use]
    pub fn fps_59_94() -> Self {
        Self::new(60000, 1001)
    }

    /// Set the adjustment mode.
    #[must_use]
    pub fn with_mode(mut self, mode: FpsMode) -> Self {
        self.mode = mode;
        self
    }

    /// Enable or disable rounding.
    #[must_use]
    pub fn with_round(mut self, round: bool) -> Self {
        self.round = round;
        self
    }

    /// Set the start time offset.
    #[must_use]
    pub fn with_start_time(mut self, start_time: i64) -> Self {
        self.start_time = start_time;
        self
    }

    /// Set the EOF action.
    #[must_use]
    pub fn with_eof_action(mut self, action: EofAction) -> Self {
        self.eof_action = action;
        self
    }

    /// Get the target frame rate as a float.
    #[must_use]
    pub fn fps(&self) -> f64 {
        self.fps_num as f64 / self.fps_den as f64
    }

    /// Get the frame duration in the given timebase.
    #[must_use]
    pub fn frame_duration(&self, timebase: Rational) -> i64 {
        let duration_sec = self.fps_den as f64 / self.fps_num as f64;
        let tb_rate = timebase.den as f64 / timebase.num as f64;
        (duration_sec * tb_rate).round() as i64
    }
}

/// End-of-stream action.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum EofAction {
    /// Pass through remaining frames.
    #[default]
    Pass,
    /// Repeat last frame until expected end.
    Repeat,
    /// Discard remaining frames.
    Discard,
}

/// Convert a floating point frame rate to a rational approximation.
fn rational_from_float(fps: f64) -> (u32, u32) {
    const COMMON_RATES: [(f64, u32, u32); 10] = [
        (23.976, 24000, 1001),
        (24.0, 24, 1),
        (25.0, 25, 1),
        (29.97, 30000, 1001),
        (30.0, 30, 1),
        (50.0, 50, 1),
        (59.94, 60000, 1001),
        (60.0, 60, 1),
        (120.0, 120, 1),
        (144.0, 144, 1),
    ];

    // Check for common rates first
    for (rate, num, den) in COMMON_RATES {
        if (fps - rate).abs() < 0.01 {
            return (num, den);
        }
    }

    // Fall back to simple integer if close enough
    let int_fps = fps.round() as u32;
    if (fps - int_fps as f64).abs() < 0.01 {
        return (int_fps, 1);
    }

    // Use continued fraction approximation
    let (num, den) = continued_fraction(fps, 1000000);
    (num as u32, den as u32)
}

/// Continued fraction approximation of a float.
fn continued_fraction(value: f64, max_den: i64) -> (i64, i64) {
    let mut n0 = 0i64;
    let mut d0 = 1i64;
    let mut n1 = 1i64;
    let mut d1 = 0i64;

    let mut x = value;
    loop {
        let a = x.floor() as i64;
        let n2 = a * n1 + n0;
        let d2 = a * d1 + d0;

        if d2 > max_den {
            break;
        }

        n0 = n1;
        d0 = d1;
        n1 = n2;
        d1 = d2;

        let rem = x - a as f64;
        if rem.abs() < 1e-10 {
            break;
        }
        x = 1.0 / rem;
    }

    (n1, d1)
}

/// Frame rate adjustment filter.
///
/// Adjusts video frame rate by dropping, duplicating, or blending frames.
///
/// # Example
///
/// ```ignore
/// use oximedia_graph::filters::video::{FpsFilter, FpsConfig, FpsMode};
/// use oximedia_graph::node::NodeId;
///
/// // Convert to 30 fps with blending
/// let config = FpsConfig::fps_30()
///     .with_mode(FpsMode::Blend);
///
/// let filter = FpsFilter::new(NodeId(0), "fps", config);
/// ```
pub struct FpsFilter {
    id: NodeId,
    name: String,
    state: NodeState,
    inputs: Vec<InputPort>,
    outputs: Vec<OutputPort>,
    config: FpsConfig,
    /// Buffer of recent frames for interpolation.
    frame_buffer: VecDeque<VideoFrame>,
    /// Current output frame index.
    output_frame_idx: u64,
    /// Last input frame PTS.
    last_input_pts: Option<i64>,
    /// Input timebase.
    input_timebase: Rational,
    /// Frames dropped count.
    frames_dropped: u64,
    /// Frames duplicated count.
    frames_duplicated: u64,
}

impl FpsFilter {
    /// Create a new FPS filter.
    #[must_use]
    pub fn new(id: NodeId, name: impl Into<String>, config: FpsConfig) -> Self {
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
            last_input_pts: None,
            input_timebase: Rational::new(1, 1000),
            frames_dropped: 0,
            frames_duplicated: 0,
        }
    }

    /// Get the current configuration.
    #[must_use]
    pub fn config(&self) -> &FpsConfig {
        &self.config
    }

    /// Get the number of frames dropped.
    #[must_use]
    pub fn frames_dropped(&self) -> u64 {
        self.frames_dropped
    }

    /// Get the number of frames duplicated.
    #[must_use]
    pub fn frames_duplicated(&self) -> u64 {
        self.frames_duplicated
    }

    /// Calculate the expected PTS for an output frame index.
    fn expected_pts(&self, frame_idx: u64) -> i64 {
        let frame_duration = self.config.frame_duration(self.input_timebase);
        self.config.start_time + (frame_idx as i64 * frame_duration)
    }

    /// Find the nearest input frame to the target PTS.
    fn find_nearest_frame(&self, target_pts: i64) -> Option<&VideoFrame> {
        self.frame_buffer.iter().min_by_key(|f| {
            let pts = f.timestamp.pts;
            (pts - target_pts).abs()
        })
    }

    /// Blend two frames together.
    fn blend_frames(
        &self,
        frame1: &VideoFrame,
        frame2: &VideoFrame,
        blend_factor: f64,
    ) -> VideoFrame {
        let mut output = frame1.clone();

        for (i, (p1, p2)) in frame1.planes.iter().zip(frame2.planes.iter()).enumerate() {
            let (w, h) = frame1.plane_dimensions(i);
            let size = (w * h) as usize;
            let mut blended_data = vec![0u8; size];

            for j in 0..size {
                let v1 = p1.data.get(j).copied().unwrap_or(0) as f64;
                let v2 = p2.data.get(j).copied().unwrap_or(0) as f64;
                let blended = v1 * (1.0 - blend_factor) + v2 * blend_factor;
                blended_data[j] = blended.round().clamp(0.0, 255.0) as u8;
            }

            output.planes[i] = Plane::new(blended_data, p1.stride);
        }

        output
    }

    /// Process a frame and potentially produce output.
    fn process_frame(&mut self, input: VideoFrame) -> GraphResult<Vec<VideoFrame>> {
        // Update input timebase
        self.input_timebase = input.timestamp.timebase;

        let input_pts = input.timestamp.pts;
        self.frame_buffer.push_back(input);

        // Keep only the last few frames for interpolation
        while self.frame_buffer.len() > 3 {
            self.frame_buffer.pop_front();
        }

        let mut output_frames = Vec::new();

        // Generate output frames
        loop {
            let target_pts = self.expected_pts(self.output_frame_idx);

            // Check if we have enough frames
            if let Some(latest) = self.frame_buffer.back() {
                if latest.timestamp.pts < target_pts && self.last_input_pts.is_none() {
                    // Need more input frames
                    break;
                }
            } else {
                break;
            }

            match self.config.mode {
                FpsMode::DropDuplicate | FpsMode::Drop | FpsMode::Duplicate => {
                    if let Some(nearest) = self.find_nearest_frame(target_pts) {
                        let nearest_pts = nearest.timestamp.pts;
                        let frame_duration = self.config.frame_duration(self.input_timebase);

                        // Check if we should output this frame
                        let should_output = if self.config.round {
                            (nearest_pts - target_pts).abs() <= frame_duration / 2
                        } else {
                            nearest_pts <= target_pts + frame_duration
                        };

                        if should_output {
                            let mut output = nearest.clone();
                            output.timestamp = Timestamp::new(target_pts, self.input_timebase);
                            output_frames.push(output);

                            // Track duplicates
                            if let Some(last_pts) = self.last_input_pts {
                                if nearest_pts == last_pts {
                                    self.frames_duplicated += 1;
                                }
                            }
                        } else if self.config.mode.allows_duplicate() {
                            // Duplicate last frame
                            if let Some(last) = self.frame_buffer.back() {
                                let mut output = last.clone();
                                output.timestamp = Timestamp::new(target_pts, self.input_timebase);
                                output_frames.push(output);
                                self.frames_duplicated += 1;
                            }
                        } else {
                            self.frames_dropped += 1;
                        }
                    }
                }
                FpsMode::Blend => {
                    // Find two frames to blend
                    let mut prev_frame: Option<&VideoFrame> = None;
                    let mut next_frame: Option<&VideoFrame> = None;

                    for frame in &self.frame_buffer {
                        if frame.timestamp.pts <= target_pts {
                            prev_frame = Some(frame);
                        }
                        if frame.timestamp.pts >= target_pts && next_frame.is_none() {
                            next_frame = Some(frame);
                        }
                    }

                    match (prev_frame, next_frame) {
                        (Some(prev), Some(next)) => {
                            let prev_pts = prev.timestamp.pts;
                            let next_pts = next.timestamp.pts;

                            if prev_pts == next_pts {
                                // Same frame, no blending needed
                                let mut output = prev.clone();
                                output.timestamp = Timestamp::new(target_pts, self.input_timebase);
                                output_frames.push(output);
                            } else {
                                // Blend between frames
                                let blend_factor =
                                    (target_pts - prev_pts) as f64 / (next_pts - prev_pts) as f64;
                                let blend_factor = blend_factor.clamp(0.0, 1.0);

                                let mut blended = self.blend_frames(prev, next, blend_factor);
                                blended.timestamp = Timestamp::new(target_pts, self.input_timebase);
                                output_frames.push(blended);
                            }
                        }
                        (Some(prev), None) => {
                            // Only have previous frame, duplicate
                            let mut output = prev.clone();
                            output.timestamp = Timestamp::new(target_pts, self.input_timebase);
                            output_frames.push(output);
                            self.frames_duplicated += 1;
                        }
                        _ => {
                            // Need more input
                            break;
                        }
                    }
                }
                FpsMode::Vfr => {
                    // Pass through with adjusted timestamp
                    if let Some(frame) = self.frame_buffer.back() {
                        let mut output = frame.clone();
                        output.timestamp = Timestamp::new(target_pts, self.input_timebase);
                        output_frames.push(output);
                    }
                }
            }

            self.output_frame_idx += 1;

            // Limit output frames per call
            if output_frames.len() >= 10 {
                break;
            }
        }

        self.last_input_pts = Some(input_pts);
        Ok(output_frames)
    }
}

impl Node for FpsFilter {
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
                let output_frames = self.process_frame(frame)?;
                // Return the first output frame; additional frames would need buffering
                Ok(output_frames.into_iter().next().map(FilterFrame::Video))
            }
            Some(_) => Err(GraphError::PortTypeMismatch {
                expected: "Video".to_string(),
                actual: "Audio".to_string(),
            }),
            None => Ok(None),
        }
    }

    fn flush(&mut self) -> GraphResult<Vec<FilterFrame>> {
        let mut output = Vec::new();

        match self.config.eof_action {
            EofAction::Pass => {
                // Output remaining buffered frames
                for frame in self.frame_buffer.drain(..) {
                    output.push(FilterFrame::Video(frame));
                }
            }
            EofAction::Repeat => {
                // Repeat last frame until expected end
                if let Some(last) = self.frame_buffer.back().cloned() {
                    let target_pts = self.expected_pts(self.output_frame_idx);
                    let mut frame = last;
                    frame.timestamp = Timestamp::new(target_pts, self.input_timebase);
                    output.push(FilterFrame::Video(frame));
                }
            }
            EofAction::Discard => {
                // Discard remaining frames
                self.frame_buffer.clear();
            }
        }

        Ok(output)
    }

    fn reset(&mut self) -> GraphResult<()> {
        self.frame_buffer.clear();
        self.output_frame_idx = 0;
        self.last_input_pts = None;
        self.frames_dropped = 0;
        self.frames_duplicated = 0;
        self.set_state(NodeState::Idle)
    }
}

/// Calculate the frame rate from a video stream's timestamps.
#[allow(dead_code)]
pub struct FrameRateDetector {
    /// Collected frame timestamps.
    timestamps: Vec<i64>,
    /// Detected frame rate (num, den).
    detected_rate: Option<(u32, u32)>,
    /// Minimum frames needed for detection.
    min_frames: usize,
}

impl Default for FrameRateDetector {
    fn default() -> Self {
        Self {
            timestamps: Vec::new(),
            detected_rate: None,
            min_frames: 10,
        }
    }
}

impl FrameRateDetector {
    /// Create a new frame rate detector.
    #[must_use]
    pub fn new(min_frames: usize) -> Self {
        Self {
            timestamps: Vec::new(),
            detected_rate: None,
            min_frames,
        }
    }

    /// Add a frame timestamp.
    pub fn add_timestamp(&mut self, pts: i64) {
        self.timestamps.push(pts);

        if self.timestamps.len() >= self.min_frames && self.detected_rate.is_none() {
            self.detect();
        }
    }

    /// Detect the frame rate from collected timestamps.
    fn detect(&mut self) {
        if self.timestamps.len() < 2 {
            return;
        }

        // Calculate average frame duration
        let mut total_duration = 0i64;
        for i in 1..self.timestamps.len() {
            total_duration += self.timestamps[i] - self.timestamps[i - 1];
        }

        let avg_duration = total_duration as f64 / (self.timestamps.len() - 1) as f64;

        // Assuming 1000 timebase (ms), convert to fps
        let fps = 1000.0 / avg_duration;
        let (num, den) = rational_from_float(fps);
        self.detected_rate = Some((num, den));
    }

    /// Get the detected frame rate.
    #[must_use]
    pub fn frame_rate(&self) -> Option<(u32, u32)> {
        self.detected_rate
    }

    /// Get the detected frame rate as a float.
    #[must_use]
    pub fn fps(&self) -> Option<f64> {
        self.detected_rate.map(|(num, den)| num as f64 / den as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_frame(pts: i64) -> VideoFrame {
        use oximedia_core::PixelFormat;

        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 64, 48);
        frame.timestamp = Timestamp::new(pts, Rational::new(1, 1000));
        frame.allocate();
        frame
    }

    #[test]
    fn test_fps_mode_properties() {
        assert!(FpsMode::DropDuplicate.allows_drop());
        assert!(FpsMode::DropDuplicate.allows_duplicate());
        assert!(FpsMode::Drop.allows_drop());
        assert!(!FpsMode::Drop.allows_duplicate());
        assert!(!FpsMode::Duplicate.allows_drop());
        assert!(FpsMode::Duplicate.allows_duplicate());
    }

    #[test]
    fn test_fps_config_creation() {
        let config = FpsConfig::new(30, 1);
        assert_eq!(config.fps_num, 30);
        assert_eq!(config.fps_den, 1);
        assert!((config.fps() - 30.0).abs() < 0.001);
    }

    #[test]
    fn test_fps_config_presets() {
        assert!((FpsConfig::fps_24().fps() - 24.0).abs() < 0.001);
        assert!((FpsConfig::fps_25().fps() - 25.0).abs() < 0.001);
        assert!((FpsConfig::fps_30().fps() - 30.0).abs() < 0.001);
        assert!((FpsConfig::fps_29_97().fps() - 29.97).abs() < 0.01);
        assert!((FpsConfig::fps_60().fps() - 60.0).abs() < 0.001);
        assert!((FpsConfig::fps_59_94().fps() - 59.94).abs() < 0.01);
    }

    #[test]
    fn test_fps_config_from_rate() {
        let config = FpsConfig::from_rate(23.976);
        assert_eq!(config.fps_num, 24000);
        assert_eq!(config.fps_den, 1001);

        let config = FpsConfig::from_rate(30.0);
        assert_eq!(config.fps_num, 30);
        assert_eq!(config.fps_den, 1);
    }

    #[test]
    fn test_fps_config_frame_duration() {
        let config = FpsConfig::fps_30();
        let duration = config.frame_duration(Rational::new(1, 1000));
        // 30 fps = ~33.33ms per frame
        assert!((duration - 33).abs() <= 1);
    }

    #[test]
    fn test_rational_from_float() {
        assert_eq!(rational_from_float(24.0), (24, 1));
        assert_eq!(rational_from_float(29.97), (30000, 1001));
        assert_eq!(rational_from_float(59.94), (60000, 1001));
    }

    #[test]
    fn test_fps_filter_creation() {
        let config = FpsConfig::fps_30();
        let filter = FpsFilter::new(NodeId(0), "fps", config);

        assert_eq!(filter.id(), NodeId(0));
        assert_eq!(filter.name(), "fps");
        assert_eq!(filter.node_type(), NodeType::Filter);
    }

    #[test]
    fn test_fps_filter_process() {
        let config = FpsConfig::fps_30().with_mode(FpsMode::DropDuplicate);
        let mut filter = FpsFilter::new(NodeId(0), "fps", config);

        // Process a few frames
        for i in 0..5 {
            let frame = create_test_frame(i * 40); // ~25 fps input
            let _ = filter.process(Some(FilterFrame::Video(frame)));
        }

        // Check that we've processed frames
        assert!(filter.output_frame_idx > 0);
    }

    #[test]
    fn test_fps_filter_statistics() {
        let config = FpsConfig::fps_30();
        let filter = FpsFilter::new(NodeId(0), "fps", config);

        assert_eq!(filter.frames_dropped(), 0);
        assert_eq!(filter.frames_duplicated(), 0);
    }

    #[test]
    fn test_fps_filter_reset() {
        let config = FpsConfig::fps_30();
        let mut filter = FpsFilter::new(NodeId(0), "fps", config);

        // Process some frames
        for i in 0..3 {
            let frame = create_test_frame(i * 33);
            let _ = filter.process(Some(FilterFrame::Video(frame)));
        }

        // Reset
        filter.reset().expect("reset should succeed");

        assert_eq!(filter.output_frame_idx, 0);
        assert!(filter.frame_buffer.is_empty());
    }

    #[test]
    fn test_fps_filter_flush() {
        let config = FpsConfig::fps_30().with_eof_action(EofAction::Pass);
        let mut filter = FpsFilter::new(NodeId(0), "fps", config);

        // Add a frame to the buffer
        let frame = create_test_frame(0);
        let _ = filter.process(Some(FilterFrame::Video(frame)));

        // Flush
        let flushed = filter.flush().expect("flush should succeed");
        assert!(!flushed.is_empty());
    }

    #[test]
    fn test_frame_rate_detector() {
        let mut detector = FrameRateDetector::new(5);

        // Add timestamps at 30 fps (33.33ms interval)
        for i in 0..10 {
            detector.add_timestamp(i * 33);
        }

        let fps = detector.fps().expect("fps should succeed");
        assert!((fps - 30.0).abs() < 1.0);
    }

    #[test]
    fn test_node_state_transitions() {
        let config = FpsConfig::fps_30();
        let mut filter = FpsFilter::new(NodeId(0), "fps", config);

        assert_eq!(filter.state(), NodeState::Idle);
        filter
            .set_state(NodeState::Processing)
            .expect("set_state should succeed");
        assert_eq!(filter.state(), NodeState::Processing);
    }

    #[test]
    fn test_process_none_input() {
        let config = FpsConfig::fps_30();
        let mut filter = FpsFilter::new(NodeId(0), "fps", config);

        let result = filter.process(None).expect("process should succeed");
        assert!(result.is_none());
    }

    #[test]
    fn test_continued_fraction() {
        let (num, den) = continued_fraction(29.97, 10000);
        let result = num as f64 / den as f64;
        assert!((result - 29.97).abs() < 0.01);
    }

    #[test]
    fn test_eof_actions() {
        // Test Pass action
        let config = FpsConfig::fps_30().with_eof_action(EofAction::Pass);
        let mut filter = FpsFilter::new(NodeId(0), "fps", config);
        let _ = filter.process(Some(FilterFrame::Video(create_test_frame(0))));
        let flushed = filter.flush().expect("flush should succeed");
        assert!(!flushed.is_empty());

        // Test Discard action
        let config = FpsConfig::fps_30().with_eof_action(EofAction::Discard);
        let mut filter = FpsFilter::new(NodeId(0), "fps", config);
        let _ = filter.process(Some(FilterFrame::Video(create_test_frame(0))));
        let flushed = filter.flush().expect("flush should succeed");
        assert!(flushed.is_empty());
    }
}
