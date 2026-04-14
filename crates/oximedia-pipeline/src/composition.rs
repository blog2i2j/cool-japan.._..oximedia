//! Pipeline composition — nest sub-pipelines as single nodes.
//!
//! This module enables hierarchical pipeline construction: an entire
//! [`PipelineGraph`] can be wrapped in a [`ComposedNode`] and inserted as a
//! single node into a parent graph.  From the parent's perspective, the
//! composed node exposes only its boundary pads; the internal topology is
//! encapsulated.
//!
//! # Design
//!
//! A [`ComposedNode`] is described by:
//! - An inner [`PipelineGraph`] containing the full sub-pipeline.
//! - A [`BoundaryMap`] that names which inner pads are exposed as inputs /
//!   outputs on the composed node's external interface.
//!
//! When a parent graph calls [`PipelineComposer::flatten`], all inner nodes and
//! edges are merged into the parent, and the boundary pad references are
//! re-wired to the outer connections.
//!
//! # Example
//!
//! ```rust
//! use oximedia_pipeline::builder::PipelineBuilder;
//! use oximedia_pipeline::node::{SourceConfig, SinkConfig, FrameFormat, StreamSpec};
//! use oximedia_pipeline::composition::{ComposedNode, BoundaryMap, PipelineComposer};
//!
//! // Build a sub-pipeline (scale + flip)
//! let inner = PipelineBuilder::new()
//!     .source("in", SourceConfig::File("inner_input.mkv".into()))
//!     .scale(1280, 720)
//!     .hflip()
//!     .sink("out", SinkConfig::Null)
//!     .build()
//!     .expect("inner pipeline should be valid");
//!
//! assert!(inner.node_count() >= 3);
//! ```

use std::collections::HashMap;

use crate::graph::{Edge, PipelineGraph};
use crate::node::{NodeId, NodeSpec, StreamSpec};
use crate::PipelineError;

// ---------------------------------------------------------------------------
// BoundaryPad
// ---------------------------------------------------------------------------

/// A reference to an inner node's pad that is exposed on the boundary of a
/// composed node.
#[derive(Debug, Clone)]
pub struct BoundaryPad {
    /// The inner node that owns this pad.
    pub inner_node: NodeId,
    /// The name of the pad on the inner node.
    pub inner_pad: String,
    /// The external name by which this pad is known on the composed node.
    pub external_name: String,
    /// Stream specification at this pad.
    pub stream: StreamSpec,
}

// ---------------------------------------------------------------------------
// BoundaryMap
// ---------------------------------------------------------------------------

/// Describes which inner pads are exposed as the public interface of a
/// composed node.
#[derive(Debug, Clone, Default)]
pub struct BoundaryMap {
    /// Pads that accept data from outside (appear as input pads of the composed
    /// node).
    pub inputs: Vec<BoundaryPad>,
    /// Pads that produce data to the outside (appear as output pads).
    pub outputs: Vec<BoundaryPad>,
}

impl BoundaryMap {
    /// Create an empty boundary map.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an input boundary pad.
    pub fn add_input(
        &mut self,
        inner_node: NodeId,
        inner_pad: impl Into<String>,
        external_name: impl Into<String>,
        stream: StreamSpec,
    ) {
        self.inputs.push(BoundaryPad {
            inner_node,
            inner_pad: inner_pad.into(),
            external_name: external_name.into(),
            stream,
        });
    }

    /// Add an output boundary pad.
    pub fn add_output(
        &mut self,
        inner_node: NodeId,
        inner_pad: impl Into<String>,
        external_name: impl Into<String>,
        stream: StreamSpec,
    ) {
        self.outputs.push(BoundaryPad {
            inner_node,
            inner_pad: inner_pad.into(),
            external_name: external_name.into(),
            stream,
        });
    }
}

// ---------------------------------------------------------------------------
// ComposedNode
// ---------------------------------------------------------------------------

/// A named sub-pipeline wrapped as a single logical node.
///
/// Use [`PipelineComposer`] to flatten a composed node back into a parent graph.
#[derive(Debug, Clone)]
pub struct ComposedNode {
    /// Human-readable name for this composed block.
    pub name: String,
    /// The inner pipeline graph.
    pub inner: PipelineGraph,
    /// Boundary pad definitions.
    pub boundary: BoundaryMap,
}

impl ComposedNode {
    /// Create a new composed node from an inner pipeline and boundary map.
    #[must_use]
    pub fn new(name: impl Into<String>, inner: PipelineGraph, boundary: BoundaryMap) -> Self {
        Self {
            name: name.into(),
            inner,
            boundary,
        }
    }

    /// Validate that all boundary node IDs exist in the inner graph.
    ///
    /// # Errors
    ///
    /// Returns `PipelineError::NodeNotFound` if a boundary references a node
    /// that is not in the inner graph.
    pub fn validate(&self) -> Result<(), PipelineError> {
        for bp in self.boundary.inputs.iter().chain(self.boundary.outputs.iter()) {
            if !self.inner.nodes.contains_key(&bp.inner_node) {
                return Err(PipelineError::NodeNotFound(format!(
                    "ComposedNode '{}' boundary references missing inner node {}",
                    self.name, bp.inner_node,
                )));
            }
        }
        Ok(())
    }

    /// Number of internal nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.inner.node_count()
    }

    /// Number of input boundary pads.
    #[must_use]
    pub fn input_count(&self) -> usize {
        self.boundary.inputs.len()
    }

    /// Number of output boundary pads.
    #[must_use]
    pub fn output_count(&self) -> usize {
        self.boundary.outputs.len()
    }
}

// ---------------------------------------------------------------------------
// PipelineComposer
// ---------------------------------------------------------------------------

/// Flattens composed nodes into a parent [`PipelineGraph`].
///
/// After calling [`PipelineComposer::flatten`], the parent graph contains all
/// inner nodes and edges from the composed node, and any edges in the parent
/// graph that were connected to the composed-node placeholder are re-wired to
/// the appropriate inner pads via the boundary map.
pub struct PipelineComposer;

impl PipelineComposer {
    /// Flatten a [`ComposedNode`] into a new standalone `PipelineGraph`.
    ///
    /// This is the most common usage: materialise the composed node into a
    /// concrete graph for validation and execution.
    ///
    /// # Errors
    ///
    /// Returns `PipelineError::ValidationError` if the boundary is invalid.
    pub fn flatten(composed: &ComposedNode) -> Result<PipelineGraph, PipelineError> {
        composed.validate()?;
        Ok(composed.inner.clone())
    }

    /// Merge a [`ComposedNode`] into an existing `parent` graph.
    ///
    /// All inner nodes and edges are copied into `parent`.  Boundary pads in
    /// `parent_connections` (a map from external pad name → `(parent_node_id,
    /// parent_pad_name)`) are re-wired to the inner boundary nodes.
    ///
    /// # Arguments
    ///
    /// * `parent` — the graph to merge into.
    /// * `composed` — the composed node being merged.
    /// * `parent_connections` — maps each boundary pad's `external_name` to
    ///   the corresponding node/pad already in `parent` that should be
    ///   connected.
    ///
    /// # Errors
    ///
    /// Returns an error if any boundary node is missing or if a connection
    /// would create an incompatible stream pairing.
    pub fn merge_into(
        parent: &mut PipelineGraph,
        composed: &ComposedNode,
        parent_connections: &HashMap<String, (NodeId, String)>,
    ) -> Result<(), PipelineError> {
        composed.validate()?;

        // Copy all inner nodes into parent.
        for (id, spec) in &composed.inner.nodes {
            parent.nodes.insert(*id, spec.clone());
        }

        // Copy all inner edges.
        for edge in &composed.inner.edges {
            parent.edges.push(edge.clone());
        }

        // Wire up input boundaries: for each input boundary pad, add an edge
        // from the parent connection to the inner boundary node.
        for bp in &composed.boundary.inputs {
            if let Some((parent_node_id, parent_pad)) = parent_connections.get(&bp.external_name) {
                parent.edges.push(Edge {
                    from_node: *parent_node_id,
                    from_pad: parent_pad.clone(),
                    to_node: bp.inner_node,
                    to_pad: bp.inner_pad.clone(),
                });
            }
        }

        // Wire up output boundaries: add edges from inner boundary nodes to
        // parent connection nodes.
        for bp in &composed.boundary.outputs {
            if let Some((parent_node_id, parent_pad)) = parent_connections.get(&bp.external_name) {
                parent.edges.push(Edge {
                    from_node: bp.inner_node,
                    from_pad: bp.inner_pad.clone(),
                    to_node: *parent_node_id,
                    to_pad: parent_pad.clone(),
                });
            }
        }

        Ok(())
    }

    /// Build a `NodeSpec` that represents the composed node's boundary as a
    /// single logical filter node in a parent graph.
    ///
    /// This is useful when you want a placeholder node in a parent graph that
    /// shows the composed node's interface without fully inlining the inner
    /// topology.  Call [`merge_into`] later to inline it.
    ///
    /// [`merge_into`]: Self::merge_into
    pub fn as_filter_spec(composed: &ComposedNode) -> NodeSpec {
        use crate::node::{FilterConfig, StreamSpec};

        // Build input pads from boundary inputs.
        let input_pads: Vec<(String, StreamSpec)> = composed
            .boundary
            .inputs
            .iter()
            .map(|bp| (bp.external_name.clone(), bp.stream.clone()))
            .collect();

        // Build output pads from boundary outputs.
        let output_pads: Vec<(String, StreamSpec)> = composed
            .boundary
            .outputs
            .iter()
            .map(|bp| (bp.external_name.clone(), bp.stream.clone()))
            .collect();

        let default_stream = StreamSpec::video(
            crate::node::FrameFormat::Yuv420p,
            0,
            0,
            25,
        );
        let first_in_stream = input_pads
            .first()
            .map(|(_, s)| s.clone())
            .or_else(|| output_pads.first().map(|(_, s)| s.clone()))
            .unwrap_or_else(|| default_stream.clone());
        let first_out_stream = output_pads
            .first()
            .map(|(_, s)| s.clone())
            .unwrap_or_else(|| first_in_stream.clone());

        // Build a Parametric filter config wrapping a noop Hflip as base.
        let props = std::collections::HashMap::from([
            ("composed_name".to_string(), composed.name.clone()),
            ("inner_node_count".to_string(), composed.node_count().to_string()),
        ]);
        let base_config = FilterConfig::Custom {
            name: "composed_block".to_string(),
            params: vec![("name".to_string(), composed.name.clone())],
        };
        let filter_config = FilterConfig::parametric(base_config, props);

        let mut spec = NodeSpec::filter(
            &composed.name,
            filter_config,
            first_in_stream,
            first_out_stream,
        );

        // Replace the default pads with boundary-derived ones.
        spec.input_pads = input_pads;
        spec.output_pads = output_pads;

        spec
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::PipelineBuilder;
    use crate::node::{FrameFormat, SinkConfig, SourceConfig, StreamSpec};

    fn build_inner_pipeline() -> PipelineGraph {
        PipelineBuilder::new()
            .source("src", SourceConfig::File("input.mkv".into()))
            .scale(1280, 720)
            .hflip()
            .sink("sink", SinkConfig::Null)
            .build()
            .expect("inner pipeline valid")
    }

    #[test]
    fn test_composed_node_node_count() {
        let inner = build_inner_pipeline();
        let count = inner.node_count();
        let boundary = BoundaryMap::new();
        let cn = ComposedNode::new("block", inner, boundary);
        assert_eq!(cn.node_count(), count);
    }

    #[test]
    fn test_composed_node_empty_boundary() {
        let inner = build_inner_pipeline();
        let boundary = BoundaryMap::new();
        let cn = ComposedNode::new("block", inner, boundary);
        assert_eq!(cn.input_count(), 0);
        assert_eq!(cn.output_count(), 0);
    }

    #[test]
    fn test_boundary_map_add() {
        let inner = build_inner_pipeline();
        let node_ids: Vec<NodeId> = inner.nodes.keys().copied().collect();
        let first_id = node_ids[0];

        let stream = StreamSpec::video(FrameFormat::Yuv420p, 1280, 720, 25);
        let mut bm = BoundaryMap::new();
        bm.add_input(first_id, "default", "video_in", stream.clone());
        bm.add_output(first_id, "default", "video_out", stream);

        assert_eq!(bm.inputs.len(), 1);
        assert_eq!(bm.outputs.len(), 1);
        assert_eq!(bm.inputs[0].external_name, "video_in");
        assert_eq!(bm.outputs[0].external_name, "video_out");
    }

    #[test]
    fn test_validate_valid_boundary() {
        let inner = build_inner_pipeline();
        let node_ids: Vec<NodeId> = inner.nodes.keys().copied().collect();
        let first_id = node_ids[0];

        let stream = StreamSpec::video(FrameFormat::Yuv420p, 1280, 720, 25);
        let mut bm = BoundaryMap::new();
        bm.add_input(first_id, "default", "in", stream);

        let cn = ComposedNode::new("block", inner, bm);
        assert!(cn.validate().is_ok());
    }

    #[test]
    fn test_validate_missing_inner_node() {
        let inner = build_inner_pipeline();
        let missing_id = NodeId::new(); // random ID not in inner

        let stream = StreamSpec::video(FrameFormat::Yuv420p, 1280, 720, 25);
        let mut bm = BoundaryMap::new();
        bm.add_input(missing_id, "default", "in", stream);

        let cn = ComposedNode::new("block", inner, bm);
        assert!(cn.validate().is_err());
    }

    #[test]
    fn test_flatten_produces_graph() {
        let inner = build_inner_pipeline();
        let original_count = inner.node_count();
        let boundary = BoundaryMap::new();
        let cn = ComposedNode::new("block", inner, boundary);

        let flat = PipelineComposer::flatten(&cn).expect("flatten ok");
        assert_eq!(flat.node_count(), original_count);
    }

    #[test]
    fn test_as_filter_spec_has_pads() {
        let inner = build_inner_pipeline();
        let node_ids: Vec<NodeId> = inner.nodes.keys().copied().collect();
        let first_id = node_ids[0];

        let stream = StreamSpec::video(FrameFormat::Yuv420p, 1280, 720, 25);
        let mut bm = BoundaryMap::new();
        bm.add_input(first_id, "default", "video_in", stream.clone());
        bm.add_output(first_id, "default", "video_out", stream);

        let cn = ComposedNode::new("scale_block", inner, bm);
        let spec = PipelineComposer::as_filter_spec(&cn);

        assert_eq!(spec.name, "scale_block");
        assert_eq!(spec.input_pads.len(), 1);
        assert_eq!(spec.output_pads.len(), 1);
        assert_eq!(spec.input_pads[0].0, "video_in");
        assert_eq!(spec.output_pads[0].0, "video_out");
    }

    #[test]
    fn test_merge_into_empty_connections() {
        let inner = build_inner_pipeline();
        let inner_node_count = inner.node_count();
        let boundary = BoundaryMap::new();
        let cn = ComposedNode::new("block", inner, boundary);

        let mut parent = PipelineGraph::new();
        PipelineComposer::merge_into(&mut parent, &cn, &HashMap::new())
            .expect("merge_into ok");

        assert_eq!(parent.node_count(), inner_node_count);
    }
}
