//! Passthrough video filter.
//!
//! This filter simply passes video frames through unchanged. It's useful for
//! testing and as a template for more complex filters.

use crate::error::{GraphError, GraphResult};
use crate::frame::FilterFrame;
use crate::node::{Node, NodeId, NodeState, NodeType};
use crate::port::{InputPort, OutputPort, PortFormat, PortId, PortType, VideoPortFormat};

/// A video filter that passes frames through unchanged.
///
/// This filter is useful for:
/// - Testing graph connectivity
/// - Serving as a template for custom filters
/// - Acting as a source node when configured as such
pub struct PassthroughFilter {
    id: NodeId,
    name: String,
    state: NodeState,
    node_type: NodeType,
    inputs: Vec<InputPort>,
    outputs: Vec<OutputPort>,
}

impl PassthroughFilter {
    /// Create a new passthrough filter.
    #[must_use]
    pub fn new(id: NodeId, name: impl Into<String>) -> Self {
        let video_format = PortFormat::Video(VideoPortFormat::any());

        Self {
            id,
            name: name.into(),
            state: NodeState::Idle,
            node_type: NodeType::Filter,
            inputs: vec![InputPort::new(PortId(0), "input", PortType::Video)
                .with_format(video_format.clone())],
            outputs: vec![
                OutputPort::new(PortId(0), "output", PortType::Video).with_format(video_format)
            ],
        }
    }

    /// Create a passthrough filter configured as a source node.
    ///
    /// Source nodes have no required inputs and are entry points to the graph.
    #[must_use]
    pub fn new_source(id: NodeId, name: impl Into<String>) -> Self {
        let video_format = PortFormat::Video(VideoPortFormat::any());

        Self {
            id,
            name: name.into(),
            state: NodeState::Idle,
            node_type: NodeType::Source,
            inputs: vec![InputPort::new(PortId(0), "input", PortType::Video)
                .with_format(video_format.clone())
                .optional()],
            outputs: vec![
                OutputPort::new(PortId(0), "output", PortType::Video).with_format(video_format)
            ],
        }
    }
}

impl Node for PassthroughFilter {
    fn id(&self) -> NodeId {
        self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn node_type(&self) -> NodeType {
        self.node_type
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
        // Passthrough: simply return the input
        match input {
            Some(frame) if frame.is_video() => Ok(Some(frame)),
            Some(_) => Err(GraphError::PortTypeMismatch {
                expected: "Video".to_string(),
                actual: "Audio".to_string(),
            }),
            None => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_codec::VideoFrame;
    use oximedia_core::PixelFormat;

    #[test]
    fn test_passthrough_creation() {
        let filter = PassthroughFilter::new(NodeId(42), "test_filter");

        assert_eq!(filter.id(), NodeId(42));
        assert_eq!(filter.name(), "test_filter");
        assert_eq!(filter.node_type(), NodeType::Filter);
        assert_eq!(filter.state(), NodeState::Idle);
    }

    #[test]
    fn test_passthrough_source() {
        let filter = PassthroughFilter::new_source(NodeId(0), "source");

        assert_eq!(filter.node_type(), NodeType::Source);
        assert!(!filter.inputs()[0].required);
    }

    #[test]
    fn test_passthrough_ports() {
        let filter = PassthroughFilter::new(NodeId(0), "test");

        assert_eq!(filter.inputs().len(), 1);
        assert_eq!(filter.outputs().len(), 1);

        let input = &filter.inputs()[0];
        assert_eq!(input.port_type, PortType::Video);
        assert!(input.required);

        let output = &filter.outputs()[0];
        assert_eq!(output.port_type, PortType::Video);
    }

    #[test]
    fn test_passthrough_process() {
        let mut filter = PassthroughFilter::new(NodeId(0), "test");

        // Create a test frame
        let video = VideoFrame::new(PixelFormat::Yuv420p, 1920, 1080);
        let frame = FilterFrame::Video(video);

        // Process should pass through
        let result = filter.process(Some(frame)).expect("process should succeed");
        assert!(result.is_some());
        assert!(result.expect("value should be valid").is_video());
    }

    #[test]
    fn test_passthrough_no_input() {
        let mut filter = PassthroughFilter::new(NodeId(0), "test");

        // No input should produce no output
        let result = filter.process(None).expect("process should succeed");
        assert!(result.is_none());
    }

    #[test]
    fn test_state_transitions() {
        let mut filter = PassthroughFilter::new(NodeId(0), "test");

        // Valid transitions
        assert!(filter.set_state(NodeState::Processing).is_ok());
        assert_eq!(filter.state(), NodeState::Processing);

        assert!(filter.set_state(NodeState::Idle).is_ok());
        assert_eq!(filter.state(), NodeState::Idle);

        assert!(filter.set_state(NodeState::Done).is_ok());
        assert_eq!(filter.state(), NodeState::Done);

        // Invalid transition from Done to Processing
        assert!(filter.set_state(NodeState::Processing).is_err());
    }
}
