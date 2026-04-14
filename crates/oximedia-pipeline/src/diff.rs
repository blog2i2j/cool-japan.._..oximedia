//! Pipeline diff: compute structural changes between two pipeline graphs.
//!
//! [`PipelineDiff`] compares two [`PipelineGraph`] instances and produces a
//! list of [`PipelineChange`] values describing each structural difference:
//!
//! - Nodes added to `b` that were not in `a`.
//! - Nodes removed from `a` that are absent in `b`.
//! - Nodes renamed (same id, different name) — reported as a [`PipelineChange::NodeRenamed`].
//! - Nodes whose type changed — reported as [`PipelineChange::NodeTypeChanged`].
//! - Edges added to `b` that were not in `a`.
//! - Edges removed from `a` that are absent in `b`.
//!
//! Comparison is performed by [`NodeId`] (UUID) for nodes and by
//! `(from_node, from_pad, to_node, to_pad)` for edges.
//!
//! # Example
//!
//! ```rust
//! use oximedia_pipeline::builder::PipelineBuilder;
//! use oximedia_pipeline::diff::PipelineDiff;
//! use oximedia_pipeline::node::SourceConfig;
//!
//! let a = PipelineBuilder::new()
//!     .source("src", SourceConfig::File("in.mkv".into()))
//!     .scale(1280, 720)
//!     .sink("out", oximedia_pipeline::node::SinkConfig::File("out.mkv".into()))
//!     .build()
//!     .expect("valid");
//!
//! let b = PipelineBuilder::new()
//!     .source("src", SourceConfig::File("in.mkv".into()))
//!     .sink("out", oximedia_pipeline::node::SinkConfig::File("out.mkv".into()))
//!     .build()
//!     .expect("valid");
//!
//! let changes = PipelineDiff::compute(&a, &b);
//! // b has fewer nodes (no scale) — so some removals or additions will be present.
//! // The graphs use different NodeIds so exact counts depend on UUIDs.
//! let _ = changes; // structural check passes
//! ```

use crate::graph::{Edge, PipelineGraph};
use crate::node::{NodeId, NodeType};

// ---------------------------------------------------------------------------
// PipelineChange
// ---------------------------------------------------------------------------

/// A single structural change between two pipeline configurations.
#[derive(Debug, Clone, PartialEq)]
pub enum PipelineChange {
    /// A node present in `b` was absent in `a`.
    NodeAdded {
        /// The new node's identifier.
        id: NodeId,
        /// The new node's name.
        name: String,
        /// Human-readable description of the node type.
        node_type: String,
    },
    /// A node present in `a` is absent in `b`.
    NodeRemoved {
        /// The removed node's identifier.
        id: NodeId,
        /// The removed node's name.
        name: String,
        /// Human-readable description of the node type.
        node_type: String,
    },
    /// A node with the same `id` has a different name in `b`.
    NodeRenamed {
        /// The node identifier.
        id: NodeId,
        /// Name in `a`.
        old_name: String,
        /// Name in `b`.
        new_name: String,
    },
    /// A node with the same `id` has a different [`NodeType`] in `b`.
    NodeTypeChanged {
        /// The node identifier.
        id: NodeId,
        /// Node name (from `b`).
        name: String,
        /// Type description in `a`.
        old_type: String,
        /// Type description in `b`.
        new_type: String,
    },
    /// An edge present in `b` was absent in `a`.
    EdgeAdded {
        /// The added edge.
        edge: Edge,
    },
    /// An edge present in `a` is absent in `b`.
    EdgeRemoved {
        /// The removed edge.
        edge: Edge,
    },
}

impl PipelineChange {
    /// A short human-readable label for this change.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            PipelineChange::NodeAdded { .. } => "node_added",
            PipelineChange::NodeRemoved { .. } => "node_removed",
            PipelineChange::NodeRenamed { .. } => "node_renamed",
            PipelineChange::NodeTypeChanged { .. } => "node_type_changed",
            PipelineChange::EdgeAdded { .. } => "edge_added",
            PipelineChange::EdgeRemoved { .. } => "edge_removed",
        }
    }

    /// Whether this change affects a node (vs an edge).
    #[must_use]
    pub fn is_node_change(&self) -> bool {
        matches!(
            self,
            PipelineChange::NodeAdded { .. }
                | PipelineChange::NodeRemoved { .. }
                | PipelineChange::NodeRenamed { .. }
                | PipelineChange::NodeTypeChanged { .. }
        )
    }

    /// Whether this change affects an edge.
    #[must_use]
    pub fn is_edge_change(&self) -> bool {
        matches!(
            self,
            PipelineChange::EdgeAdded { .. } | PipelineChange::EdgeRemoved { .. }
        )
    }
}

// ---------------------------------------------------------------------------
// PipelineDiff
// ---------------------------------------------------------------------------

/// Computes the structural diff between two [`PipelineGraph`] instances.
pub struct PipelineDiff;

impl PipelineDiff {
    /// Compute all changes between pipeline graph `a` and `b`.
    ///
    /// Returns a `Vec<PipelineChange>` in a deterministic order:
    /// node changes first (added, removed, renamed, type-changed), then edge
    /// changes (added, removed).
    #[must_use]
    pub fn compute(a: &PipelineGraph, b: &PipelineGraph) -> Vec<PipelineChange> {
        let mut changes: Vec<PipelineChange> = Vec::new();

        // ── Node diff ────────────────────────────────────────────────────────

        // Nodes in `b` but not in `a` → added.
        for (id, spec_b) in &b.nodes {
            if !a.nodes.contains_key(id) {
                changes.push(PipelineChange::NodeAdded {
                    id: *id,
                    name: spec_b.name.clone(),
                    node_type: node_type_label(&spec_b.node_type),
                });
            }
        }

        // Nodes in `a` but not in `b` → removed.
        for (id, spec_a) in &a.nodes {
            if !b.nodes.contains_key(id) {
                changes.push(PipelineChange::NodeRemoved {
                    id: *id,
                    name: spec_a.name.clone(),
                    node_type: node_type_label(&spec_a.node_type),
                });
            }
        }

        // Nodes present in both `a` and `b` — check for renames / type changes.
        for (id, spec_a) in &a.nodes {
            if let Some(spec_b) = b.nodes.get(id) {
                if spec_a.name != spec_b.name {
                    changes.push(PipelineChange::NodeRenamed {
                        id: *id,
                        old_name: spec_a.name.clone(),
                        new_name: spec_b.name.clone(),
                    });
                }
                let ta = node_type_label(&spec_a.node_type);
                let tb = node_type_label(&spec_b.node_type);
                if ta != tb {
                    changes.push(PipelineChange::NodeTypeChanged {
                        id: *id,
                        name: spec_b.name.clone(),
                        old_type: ta,
                        new_type: tb,
                    });
                }
            }
        }

        // ── Edge diff ────────────────────────────────────────────────────────

        // Edges in `b` but not in `a` → added.
        for edge_b in &b.edges {
            if !a.edges.iter().any(|e| edges_equal(e, edge_b)) {
                changes.push(PipelineChange::EdgeAdded { edge: edge_b.clone() });
            }
        }

        // Edges in `a` but not in `b` → removed.
        for edge_a in &a.edges {
            if !b.edges.iter().any(|e| edges_equal(e, edge_a)) {
                changes.push(PipelineChange::EdgeRemoved { edge: edge_a.clone() });
            }
        }

        changes
    }

    /// Return only node-level changes between `a` and `b`.
    #[must_use]
    pub fn node_changes(a: &PipelineGraph, b: &PipelineGraph) -> Vec<PipelineChange> {
        Self::compute(a, b)
            .into_iter()
            .filter(|c| c.is_node_change())
            .collect()
    }

    /// Return only edge-level changes between `a` and `b`.
    #[must_use]
    pub fn edge_changes(a: &PipelineGraph, b: &PipelineGraph) -> Vec<PipelineChange> {
        Self::compute(a, b)
            .into_iter()
            .filter(|c| c.is_edge_change())
            .collect()
    }

    /// Return `true` if `a` and `b` are structurally identical.
    #[must_use]
    pub fn is_identical(a: &PipelineGraph, b: &PipelineGraph) -> bool {
        Self::compute(a, b).is_empty()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Return a short label string for a [`NodeType`] variant.
fn node_type_label(nt: &NodeType) -> String {
    match nt {
        NodeType::Source(_) => "source".to_string(),
        NodeType::Sink(_) => "sink".to_string(),
        NodeType::Filter(f) => format!("filter({})", filter_label(f)),
        NodeType::Split => "split".to_string(),
        NodeType::Merge => "merge".to_string(),
        NodeType::Null => "null".to_string(),
        NodeType::Conditional(_) => "conditional".to_string(),
    }
}

/// Return a short label for a [`FilterConfig`] variant.
fn filter_label(f: &crate::node::FilterConfig) -> &'static str {
    match f {
        crate::node::FilterConfig::Scale { .. } => "scale",
        crate::node::FilterConfig::Crop { .. } => "crop",
        crate::node::FilterConfig::Trim { .. } => "trim",
        crate::node::FilterConfig::Volume { .. } => "volume",
        crate::node::FilterConfig::Fps { .. } => "fps",
        crate::node::FilterConfig::Format(_) => "format",
        crate::node::FilterConfig::Overlay => "overlay",
        crate::node::FilterConfig::Concat { .. } => "concat",
        crate::node::FilterConfig::Pad { .. } => "pad",
        crate::node::FilterConfig::Hflip => "hflip",
        crate::node::FilterConfig::Vflip => "vflip",
        crate::node::FilterConfig::Transpose(_) => "transpose",
        crate::node::FilterConfig::Custom { .. } => "custom",
        crate::node::FilterConfig::Parametric { .. } => "parametric",
    }
}

/// Check structural equality of two edges (ignoring graph identity).
fn edges_equal(a: &Edge, b: &Edge) -> bool {
    a.from_node == b.from_node
        && a.from_pad == b.from_pad
        && a.to_node == b.to_node
        && a.to_pad == b.to_pad
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::PipelineBuilder;
    use crate::graph::PipelineGraph;
    use crate::node::{NodeId, NodeSpec, NodeType, SinkConfig, SourceConfig, StreamKind, StreamSpec};

    fn video_spec() -> StreamSpec {
        StreamSpec {
            kind: StreamKind::Video,
            format: crate::node::FrameFormat::Yuv420p,
            width: None,
            height: None,
            sample_rate: None,
            channels: None,
            time_base: (1, 25),
        }
    }

    #[test]
    fn test_identical_graphs_no_changes() {
        let a = PipelineBuilder::new()
            .source("src", SourceConfig::File("in.mkv".into()))
            .sink("sink", SinkConfig::File("out.mkv".into()))
            .build()
            .expect("valid");

        assert!(
            PipelineDiff::is_identical(&a, &a),
            "graph compared to itself should have no changes"
        );
    }

    #[test]
    fn test_added_node() {
        let a = PipelineGraph::new();
        let mut b = PipelineGraph::new();

        let id = NodeId::new();
        b.nodes.insert(
            id,
            NodeSpec::new("extra", NodeType::Null, vec![], vec![]),
        );

        let changes = PipelineDiff::compute(&a, &b);
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].label(), "node_added");
    }

    #[test]
    fn test_removed_node() {
        let mut a = PipelineGraph::new();
        let b = PipelineGraph::new();

        let id = NodeId::new();
        a.nodes.insert(
            id,
            NodeSpec::new("gone", NodeType::Null, vec![], vec![]),
        );

        let changes = PipelineDiff::compute(&a, &b);
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].label(), "node_removed");
    }

    #[test]
    fn test_renamed_node() {
        let id = NodeId::new();
        let spec_a = NodeSpec {
            id,
            name: "old_name".to_string(),
            node_type: NodeType::Null,
            input_pads: vec![],
            output_pads: vec![],
        };
        let spec_b = NodeSpec {
            id,
            name: "new_name".to_string(),
            node_type: NodeType::Null,
            input_pads: vec![],
            output_pads: vec![],
        };

        let mut a = PipelineGraph::new();
        let mut b = PipelineGraph::new();
        a.nodes.insert(id, spec_a);
        b.nodes.insert(id, spec_b);

        let changes = PipelineDiff::compute(&a, &b);
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].label(), "node_renamed");
    }

    #[test]
    fn test_node_changes_filter() {
        let mut a = PipelineGraph::new();
        let b = PipelineGraph::new();
        a.nodes.insert(NodeId::new(), NodeSpec::new("x", NodeType::Null, vec![], vec![]));
        let node_ch = PipelineDiff::node_changes(&a, &b);
        let edge_ch = PipelineDiff::edge_changes(&a, &b);
        assert_eq!(node_ch.len(), 1);
        assert!(edge_ch.is_empty());
    }

    #[test]
    fn test_edge_added() {
        let id_a = NodeId::new();
        let id_b = NodeId::new();

        let mut a = PipelineGraph::new();
        let mut b = PipelineGraph::new();

        for g in [&mut a, &mut b] {
            g.nodes.insert(id_a, NodeSpec::new("n1", NodeType::Null, vec![], vec![("out".into(), video_spec())]));
            g.nodes.insert(id_b, NodeSpec::new("n2", NodeType::Null, vec![("in".into(), video_spec())], vec![]));
        }

        // Add an edge only to `b`.
        b.edges.push(Edge {
            from_node: id_a,
            from_pad: "out".to_string(),
            to_node: id_b,
            to_pad: "in".to_string(),
        });

        let changes = PipelineDiff::compute(&a, &b);
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].label(), "edge_added");
    }
}
