//! Passthrough audio filter.
//!
//! This filter simply passes audio frames through unchanged. It's useful for
//! testing and as a template for more complex audio filters.

use crate::error::{GraphError, GraphResult};
use crate::frame::FilterFrame;
use crate::node::{Node, NodeId, NodeState, NodeType};
use crate::port::{AudioPortFormat, InputPort, OutputPort, PortFormat, PortId, PortType};

/// An audio filter that passes frames through unchanged.
///
/// This filter is useful for:
/// - Testing graph connectivity with audio streams
/// - Serving as a template for custom audio filters
/// - Acting as a source node when configured as such
pub struct AudioPassthrough {
    id: NodeId,
    name: String,
    state: NodeState,
    node_type: NodeType,
    inputs: Vec<InputPort>,
    outputs: Vec<OutputPort>,
}

impl AudioPassthrough {
    /// Create a new audio passthrough filter.
    #[must_use]
    pub fn new(id: NodeId, name: impl Into<String>) -> Self {
        let audio_format = PortFormat::Audio(AudioPortFormat::any());

        Self {
            id,
            name: name.into(),
            state: NodeState::Idle,
            node_type: NodeType::Filter,
            inputs: vec![InputPort::new(PortId(0), "input", PortType::Audio)
                .with_format(audio_format.clone())],
            outputs: vec![
                OutputPort::new(PortId(0), "output", PortType::Audio).with_format(audio_format)
            ],
        }
    }

    /// Create an audio passthrough filter configured as a source node.
    #[must_use]
    pub fn new_source(id: NodeId, name: impl Into<String>) -> Self {
        let audio_format = PortFormat::Audio(AudioPortFormat::any());

        Self {
            id,
            name: name.into(),
            state: NodeState::Idle,
            node_type: NodeType::Source,
            inputs: vec![InputPort::new(PortId(0), "input", PortType::Audio)
                .with_format(audio_format.clone())
                .optional()],
            outputs: vec![
                OutputPort::new(PortId(0), "output", PortType::Audio).with_format(audio_format)
            ],
        }
    }

    /// Create an audio passthrough filter configured as a sink node.
    #[must_use]
    pub fn new_sink(id: NodeId, name: impl Into<String>) -> Self {
        let audio_format = PortFormat::Audio(AudioPortFormat::any());

        Self {
            id,
            name: name.into(),
            state: NodeState::Idle,
            node_type: NodeType::Sink,
            inputs: vec![
                InputPort::new(PortId(0), "input", PortType::Audio).with_format(audio_format)
            ],
            outputs: Vec::new(),
        }
    }
}

impl Node for AudioPassthrough {
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
        match input {
            Some(frame) if frame.is_audio() => {
                // For sink nodes, consume the frame and produce no output
                if self.node_type == NodeType::Sink {
                    Ok(None)
                } else {
                    Ok(Some(frame))
                }
            }
            Some(_) => Err(GraphError::PortTypeMismatch {
                expected: "Audio".to_string(),
                actual: "Video".to_string(),
            }),
            None => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_audio::{AudioFrame, ChannelLayout};
    use oximedia_core::SampleFormat;

    #[test]
    fn test_audio_passthrough_creation() {
        let filter = AudioPassthrough::new(NodeId(42), "test_filter");

        assert_eq!(filter.id(), NodeId(42));
        assert_eq!(filter.name(), "test_filter");
        assert_eq!(filter.node_type(), NodeType::Filter);
        assert_eq!(filter.state(), NodeState::Idle);
    }

    #[test]
    fn test_audio_passthrough_source() {
        let filter = AudioPassthrough::new_source(NodeId(0), "source");

        assert_eq!(filter.node_type(), NodeType::Source);
        assert!(!filter.inputs()[0].required);
    }

    #[test]
    fn test_audio_passthrough_sink() {
        let filter = AudioPassthrough::new_sink(NodeId(0), "sink");

        assert_eq!(filter.node_type(), NodeType::Sink);
        assert!(filter.outputs().is_empty());
    }

    #[test]
    fn test_audio_passthrough_ports() {
        let filter = AudioPassthrough::new(NodeId(0), "test");

        assert_eq!(filter.inputs().len(), 1);
        assert_eq!(filter.outputs().len(), 1);

        let input = &filter.inputs()[0];
        assert_eq!(input.port_type, PortType::Audio);

        let output = &filter.outputs()[0];
        assert_eq!(output.port_type, PortType::Audio);
    }

    #[test]
    fn test_audio_passthrough_process() {
        let mut filter = AudioPassthrough::new(NodeId(0), "test");

        // Create a test frame
        let audio = AudioFrame::new(SampleFormat::F32, 48000, ChannelLayout::Stereo);
        let frame = FilterFrame::Audio(audio);

        // Process should pass through
        let result = filter.process(Some(frame)).expect("process should succeed");
        assert!(result.is_some());
        assert!(result.expect("value should be valid").is_audio());
    }

    #[test]
    fn test_audio_passthrough_sink_process() {
        let mut filter = AudioPassthrough::new_sink(NodeId(0), "sink");

        // Create a test frame
        let audio = AudioFrame::new(SampleFormat::F32, 48000, ChannelLayout::Stereo);
        let frame = FilterFrame::Audio(audio);

        // Sink should consume frame and produce no output
        let result = filter.process(Some(frame)).expect("process should succeed");
        assert!(result.is_none());
    }

    #[test]
    fn test_audio_passthrough_no_input() {
        let mut filter = AudioPassthrough::new(NodeId(0), "test");

        let result = filter.process(None).expect("process should succeed");
        assert!(result.is_none());
    }

    #[test]
    fn test_state_transitions() {
        let mut filter = AudioPassthrough::new(NodeId(0), "test");

        assert!(filter.set_state(NodeState::Processing).is_ok());
        assert_eq!(filter.state(), NodeState::Processing);

        assert!(filter.set_state(NodeState::Done).is_ok());
        assert_eq!(filter.state(), NodeState::Done);
    }
}
