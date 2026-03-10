//! Null sink for video frames.
//!
//! This sink discards all incoming frames. It's useful for benchmarking
//! encoder/decoder performance without the overhead of writing to disk.

use crate::error::{GraphError, GraphResult};
use crate::frame::FilterFrame;
use crate::node::{Node, NodeId, NodeState, NodeType};
use crate::port::{InputPort, OutputPort, PortFormat, PortId, PortType, VideoPortFormat};

/// A sink that discards all incoming video frames.
///
/// This is useful for:
/// - Benchmarking without I/O overhead
/// - Testing graph connectivity
/// - Counting frames processed
pub struct NullSink {
    id: NodeId,
    name: String,
    state: NodeState,
    inputs: Vec<InputPort>,
    outputs: Vec<OutputPort>,
    frames_received: u64,
}

impl NullSink {
    /// Create a new null sink.
    #[must_use]
    pub fn new(id: NodeId, name: impl Into<String>) -> Self {
        let video_format = PortFormat::Video(VideoPortFormat::any());

        Self {
            id,
            name: name.into(),
            state: NodeState::Idle,
            inputs: vec![
                InputPort::new(PortId(0), "input", PortType::Video).with_format(video_format)
            ],
            outputs: Vec::new(),
            frames_received: 0,
        }
    }

    /// Get the number of frames received.
    #[must_use]
    pub fn frames_received(&self) -> u64 {
        self.frames_received
    }

    /// Reset the frame counter.
    pub fn reset_counter(&mut self) {
        self.frames_received = 0;
    }
}

impl Node for NullSink {
    fn id(&self) -> NodeId {
        self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn node_type(&self) -> NodeType {
        NodeType::Sink
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
        if let Some(frame) = input {
            if frame.is_video() {
                self.frames_received += 1;
            }
        }
        // Sink produces no output
        Ok(None)
    }

    fn reset(&mut self) -> GraphResult<()> {
        self.frames_received = 0;
        self.set_state(NodeState::Idle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_codec::VideoFrame;
    use oximedia_core::PixelFormat;

    #[test]
    fn test_null_sink_creation() {
        let sink = NullSink::new(NodeId(0), "null");

        assert_eq!(sink.id(), NodeId(0));
        assert_eq!(sink.name(), "null");
        assert_eq!(sink.node_type(), NodeType::Sink);
        assert_eq!(sink.frames_received(), 0);
    }

    #[test]
    fn test_null_sink_ports() {
        let sink = NullSink::new(NodeId(0), "null");

        // Should have one input, no outputs
        assert_eq!(sink.inputs().len(), 1);
        assert_eq!(sink.outputs().len(), 0);
    }

    #[test]
    fn test_null_sink_process() {
        let mut sink = NullSink::new(NodeId(0), "null");

        // Process some frames
        for _ in 0..5 {
            let video = VideoFrame::new(PixelFormat::Yuv420p, 1920, 1080);
            let frame = FilterFrame::Video(video);
            let result = sink.process(Some(frame)).expect("process should succeed");
            // Sink should produce no output
            assert!(result.is_none());
        }

        assert_eq!(sink.frames_received(), 5);
    }

    #[test]
    fn test_null_sink_reset() {
        let mut sink = NullSink::new(NodeId(0), "null");

        // Process a frame
        let video = VideoFrame::new(PixelFormat::Yuv420p, 1920, 1080);
        sink.process(Some(FilterFrame::Video(video)))
            .expect("operation should succeed");
        assert_eq!(sink.frames_received(), 1);

        // Reset
        sink.reset().expect("reset should succeed");
        assert_eq!(sink.frames_received(), 0);
        assert_eq!(sink.state(), NodeState::Idle);
    }

    #[test]
    fn test_null_sink_no_input() {
        let mut sink = NullSink::new(NodeId(0), "null");

        let result = sink.process(None).expect("process should succeed");
        assert!(result.is_none());
        assert_eq!(sink.frames_received(), 0);
    }
}
