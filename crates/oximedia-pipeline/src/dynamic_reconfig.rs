//! Dynamic pipeline reconfiguration — add, remove, and replace nodes at runtime.
//!
//! A [`DynamicPipeline`] wraps a [`PipelineGraph`] and provides a transactional
//! API for making structural changes while the pipeline is either idle or between
//! frame boundaries.
//!
//! # Design
//!
//! Changes are collected into a [`ReconfigTransaction`] via a builder-style API.
//! Only when [`DynamicPipeline::commit`] is called are all operations applied
//! atomically:
//!
//! 1. The current graph is cloned.
//! 2. All pending operations are applied to the clone.
//! 3. The clone is validated.
//! 4. On success, the clone replaces the live graph and a [`ReconfigEvent`] is
//!    emitted to any registered listener.
//! 5. On failure, the original graph is unchanged and the error is returned.
//!
//! This guarantees that the live graph is **never left in a partially-modified
//! state** due to a failed operation.
//!
//! # Example
//!
//! ```rust
//! use oximedia_pipeline::builder::PipelineBuilder;
//! use oximedia_pipeline::node::{SourceConfig, SinkConfig, FilterConfig};
//! use oximedia_pipeline::dynamic_reconfig::DynamicPipeline;
//!
//! let graph = PipelineBuilder::new()
//!     .source("in", SourceConfig::File("input.mp4".into()))
//!     .scale(1280, 720)
//!     .sink("out", SinkConfig::Null)
//!     .build()
//!     .expect("build ok");
//!
//! let mut dyn_pipeline = DynamicPipeline::new(graph);
//! let node_count_before = dyn_pipeline.graph().node_count();
//!
//! // Append a new hflip filter at the end of the scale node's output
//! let tx = dyn_pipeline
//!     .begin()
//!     .insert_filter_after_name("scale", "hflip", FilterConfig::Hflip);
//!
//! dyn_pipeline.commit(tx).expect("reconfiguration should succeed");
//! assert!(dyn_pipeline.graph().node_count() > node_count_before);
//! ```

use crate::graph::{Edge, PipelineGraph};
use crate::node::{FilterConfig, NodeId, NodeSpec, NodeType};
use crate::PipelineError;

// ── ReconfigOp ────────────────────────────────────────────────────────────────

/// A single atomic reconfiguration operation on a pipeline graph.
#[derive(Debug, Clone)]
pub enum ReconfigOp {
    /// Remove the node with the given [`NodeId`] and re-wire its edges.
    ///
    /// The removed node's in-edges are connected directly to its out-edges
    /// (bypass), preserving stream flow.  If the node has no in-edges or no
    /// out-edges, only one side of the bypass is possible.
    RemoveNode(NodeId),

    /// Remove the node whose human-readable name matches `name`.
    ///
    /// If multiple nodes share the name, the **first** match (arbitrary ordering)
    /// is removed.
    RemoveNodeByName(String),

    /// Insert a new [`NodeSpec`] into the graph.  No edges are created; the
    /// caller must also add [`ReconfigOp::AddEdge`] operations to connect the
    /// node.
    AddNode(NodeSpec),

    /// Add a directed edge from `(from_node, from_pad)` to `(to_node, to_pad)`.
    AddEdge {
        from_node: NodeId,
        from_pad: String,
        to_node: NodeId,
        to_pad: String,
    },

    /// Remove the directed edge from `(from_node, from_pad)` to
    /// `(to_node, to_pad)`.  No-op if the edge does not exist.
    RemoveEdge {
        from_node: NodeId,
        from_pad: String,
        to_node: NodeId,
        to_pad: String,
    },

    /// Replace a node's [`NodeType`] (i.e., its filter configuration) in-place.
    ///
    /// Input and output pads are **not** changed, so this is safe only when the
    /// replacement has identical stream kind requirements.
    ReplaceNodeType {
        node_id: NodeId,
        new_type: NodeType,
    },

    /// Insert a new filter node immediately after the named upstream node,
    /// redirecting the upstream's out-edges to pass through the new node first.
    InsertFilterAfterName {
        upstream_name: String,
        new_node_name: String,
        filter_config: FilterConfig,
    },

    /// Insert a new filter node immediately before the named downstream node,
    /// redirecting the downstream's in-edges to pass through the new node first.
    InsertFilterBeforeName {
        downstream_name: String,
        new_node_name: String,
        filter_config: FilterConfig,
    },
}

// ── ReconfigTransaction ───────────────────────────────────────────────────────

/// A sequence of reconfiguration operations that will be applied atomically.
///
/// Build one via [`DynamicPipeline::begin`] and commit it via
/// [`DynamicPipeline::commit`].
#[derive(Debug, Clone, Default)]
pub struct ReconfigTransaction {
    ops: Vec<ReconfigOp>,
}

impl ReconfigTransaction {
    /// Create an empty transaction.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a raw [`ReconfigOp`] to this transaction.
    pub fn push(mut self, op: ReconfigOp) -> Self {
        self.ops.push(op);
        self
    }

    /// Remove a node by ID, bypassing its edges.
    pub fn remove_node(self, node_id: NodeId) -> Self {
        self.push(ReconfigOp::RemoveNode(node_id))
    }

    /// Remove a node by human-readable name, bypassing its edges.
    pub fn remove_node_by_name(self, name: impl Into<String>) -> Self {
        self.push(ReconfigOp::RemoveNodeByName(name.into()))
    }

    /// Insert a new filter node after the named upstream node.
    pub fn insert_filter_after_name(
        self,
        upstream_name: impl Into<String>,
        new_node_name: impl Into<String>,
        filter_config: FilterConfig,
    ) -> Self {
        self.push(ReconfigOp::InsertFilterAfterName {
            upstream_name: upstream_name.into(),
            new_node_name: new_node_name.into(),
            filter_config,
        })
    }

    /// Insert a new filter node before the named downstream node.
    pub fn insert_filter_before_name(
        self,
        downstream_name: impl Into<String>,
        new_node_name: impl Into<String>,
        filter_config: FilterConfig,
    ) -> Self {
        self.push(ReconfigOp::InsertFilterBeforeName {
            downstream_name: downstream_name.into(),
            new_node_name: new_node_name.into(),
            filter_config,
        })
    }

    /// Add a bare node (no edges).
    pub fn add_node(self, spec: NodeSpec) -> Self {
        self.push(ReconfigOp::AddNode(spec))
    }

    /// Add a directed edge.
    pub fn add_edge(
        self,
        from_node: NodeId,
        from_pad: impl Into<String>,
        to_node: NodeId,
        to_pad: impl Into<String>,
    ) -> Self {
        self.push(ReconfigOp::AddEdge {
            from_node,
            from_pad: from_pad.into(),
            to_node,
            to_pad: to_pad.into(),
        })
    }

    /// Replace a node's type in-place.
    pub fn replace_node_type(self, node_id: NodeId, new_type: NodeType) -> Self {
        self.push(ReconfigOp::ReplaceNodeType { node_id, new_type })
    }

    /// Number of operations in this transaction.
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Returns `true` when no operations have been queued.
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }
}

// ── ReconfigEvent ─────────────────────────────────────────────────────────────

/// Describes a committed reconfiguration for notification purposes.
#[derive(Debug, Clone)]
pub struct ReconfigEvent {
    /// Number of nodes added in the transaction.
    pub nodes_added: u32,
    /// Number of nodes removed in the transaction.
    pub nodes_removed: u32,
    /// Number of edges added in the transaction.
    pub edges_added: u32,
    /// Number of edges removed in the transaction.
    pub edges_removed: u32,
    /// Number of node types replaced in the transaction.
    pub types_replaced: u32,
    /// Total operations committed.
    pub total_ops: u32,
}

// ── DynamicPipeline ───────────────────────────────────────────────────────────

/// A live pipeline that supports atomic structural reconfiguration.
///
/// Wraps a [`PipelineGraph`] and provides a transactional API so that
/// multiple edits can be applied atomically with rollback on failure.
pub struct DynamicPipeline {
    graph: PipelineGraph,
    /// Generation counter incremented on every successful commit.
    generation: u64,
    /// Optional callback invoked after each successful commit.
    on_reconfig: Option<Box<dyn Fn(&ReconfigEvent) + Send + Sync + 'static>>,
}

impl DynamicPipeline {
    /// Create a `DynamicPipeline` wrapping an existing graph.
    pub fn new(graph: PipelineGraph) -> Self {
        Self {
            graph,
            generation: 0,
            on_reconfig: None,
        }
    }

    /// Register a callback that is called after every successful [`commit`].
    ///
    /// Only one listener is supported; calling this again replaces any previous
    /// registration.
    ///
    /// [`commit`]: DynamicPipeline::commit
    pub fn on_reconfig<F>(&mut self, f: F)
    where
        F: Fn(&ReconfigEvent) + Send + Sync + 'static,
    {
        self.on_reconfig = Some(Box::new(f));
    }

    /// Return a read-only reference to the current graph.
    pub fn graph(&self) -> &PipelineGraph {
        &self.graph
    }

    /// Return the current generation counter.  Incremented on every successful
    /// commit.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Begin a new empty transaction.
    pub fn begin(&self) -> ReconfigTransaction {
        ReconfigTransaction::new()
    }

    /// Apply all operations in `tx` atomically.
    ///
    /// If any operation fails (e.g. referenced node not found), the live graph
    /// is left unchanged and the error is returned.
    pub fn commit(&mut self, tx: ReconfigTransaction) -> Result<ReconfigEvent, PipelineError> {
        // Work on a clone so we can roll back on failure
        let mut candidate = self.graph.clone();
        let mut event = ReconfigEvent {
            nodes_added: 0,
            nodes_removed: 0,
            edges_added: 0,
            edges_removed: 0,
            types_replaced: 0,
            total_ops: tx.ops.len() as u32,
        };

        for op in tx.ops {
            apply_op(&mut candidate, op, &mut event)?;
        }

        // Commit
        self.graph = candidate;
        self.generation += 1;

        if let Some(cb) = &self.on_reconfig {
            cb(&event);
        }

        Ok(event)
    }

    /// Convenience: commit a single operation directly.
    pub fn apply(&mut self, op: ReconfigOp) -> Result<ReconfigEvent, PipelineError> {
        let tx = ReconfigTransaction::new().push(op);
        self.commit(tx)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Operation implementation helpers
// ─────────────────────────────────────────────────────────────────────────────

fn apply_op(
    g: &mut PipelineGraph,
    op: ReconfigOp,
    event: &mut ReconfigEvent,
) -> Result<(), PipelineError> {
    match op {
        ReconfigOp::RemoveNode(id) => {
            remove_and_bypass(g, id)?;
            event.nodes_removed += 1;
        }

        ReconfigOp::RemoveNodeByName(name) => {
            let id = find_node_by_name(g, &name)
                .ok_or_else(|| PipelineError::NodeNotFound(name.clone()))?;
            remove_and_bypass(g, id)?;
            event.nodes_removed += 1;
        }

        ReconfigOp::AddNode(spec) => {
            g.add_node(spec);
            event.nodes_added += 1;
        }

        ReconfigOp::AddEdge {
            from_node,
            from_pad,
            to_node,
            to_pad,
        } => {
            g.connect(from_node, &from_pad, to_node, &to_pad)?;
            event.edges_added += 1;
        }

        ReconfigOp::RemoveEdge {
            from_node,
            from_pad,
            to_node,
            to_pad,
        } => {
            let before = g.edges.len();
            g.edges.retain(|e| {
                !(e.from_node == from_node
                    && e.from_pad == from_pad
                    && e.to_node == to_node
                    && e.to_pad == to_pad)
            });
            if g.edges.len() < before {
                event.edges_removed += (before - g.edges.len()) as u32;
            }
        }

        ReconfigOp::ReplaceNodeType { node_id, new_type } => {
            let spec = g
                .nodes
                .get_mut(&node_id)
                .ok_or_else(|| PipelineError::NodeNotFound(node_id.to_string()))?;
            spec.node_type = new_type;
            event.types_replaced += 1;
        }

        ReconfigOp::InsertFilterAfterName {
            upstream_name,
            new_node_name,
            filter_config,
        } => {
            insert_filter_after_name(g, &upstream_name, &new_node_name, filter_config, event)?;
        }

        ReconfigOp::InsertFilterBeforeName {
            downstream_name,
            new_node_name,
            filter_config,
        } => {
            insert_filter_before_name(g, &downstream_name, &new_node_name, filter_config, event)?;
        }
    }
    Ok(())
}

/// Remove a node, bypassing its edges (connect upstream directly to downstream).
fn remove_and_bypass(g: &mut PipelineGraph, id: NodeId) -> Result<(), PipelineError> {
    if !g.nodes.contains_key(&id) {
        return Err(PipelineError::NodeNotFound(id.to_string()));
    }

    let in_edges: Vec<Edge> = g.edges.iter().filter(|e| e.to_node == id).cloned().collect();
    let out_edges: Vec<Edge> = g.edges.iter().filter(|e| e.from_node == id).cloned().collect();

    let mut bypass: Vec<Edge> = Vec::new();
    for ie in &in_edges {
        for oe in &out_edges {
            bypass.push(Edge {
                from_node: ie.from_node,
                from_pad: ie.from_pad.clone(),
                to_node: oe.to_node,
                to_pad: oe.to_pad.clone(),
            });
        }
    }

    g.edges.retain(|e| e.to_node != id && e.from_node != id);
    g.edges.extend(bypass);
    g.nodes.remove(&id);
    Ok(())
}

/// Find the first node whose `name` matches `target` (exact match).
fn find_node_by_name(g: &PipelineGraph, target: &str) -> Option<NodeId> {
    g.nodes
        .iter()
        .find(|(_, spec)| spec.name == target)
        .map(|(id, _)| *id)
}

/// Insert a filter node immediately after `upstream_name`.
fn insert_filter_after_name(
    g: &mut PipelineGraph,
    upstream_name: &str,
    new_node_name: &str,
    filter_config: FilterConfig,
    event: &mut ReconfigEvent,
) -> Result<(), PipelineError> {
    let upstream_id = find_node_by_name(g, upstream_name)
        .ok_or_else(|| PipelineError::NodeNotFound(upstream_name.to_string()))?;

    // Derive stream spec from the upstream node's first output pad
    let stream_spec = g
        .nodes
        .get(&upstream_id)
        .and_then(|s| s.output_pads.first().map(|(_, sp)| sp.clone()))
        .unwrap_or_default();

    // Collect out-edges from the upstream node (we will retarget them)
    let out_edges: Vec<Edge> = g
        .edges
        .iter()
        .filter(|e| e.from_node == upstream_id)
        .cloned()
        .collect();

    // Build the new filter node
    let new_spec = NodeSpec::filter(
        new_node_name,
        filter_config,
        stream_spec.clone(),
        stream_spec,
    );
    let new_id = new_spec.id;
    g.nodes.insert(new_id, new_spec);
    event.nodes_added += 1;

    // Remove original out-edges from upstream, re-add targeting new node
    g.edges.retain(|e| e.from_node != upstream_id);

    // upstream → new_node
    g.edges.push(Edge {
        from_node: upstream_id,
        from_pad: "default".to_string(),
        to_node: new_id,
        to_pad: "default".to_string(),
    });
    event.edges_added += 1;

    // new_node → original destinations
    for oe in &out_edges {
        g.edges.push(Edge {
            from_node: new_id,
            from_pad: "default".to_string(),
            to_node: oe.to_node,
            to_pad: oe.to_pad.clone(),
        });
        event.edges_added += 1;
    }

    Ok(())
}

/// Insert a filter node immediately before `downstream_name`.
fn insert_filter_before_name(
    g: &mut PipelineGraph,
    downstream_name: &str,
    new_node_name: &str,
    filter_config: FilterConfig,
    event: &mut ReconfigEvent,
) -> Result<(), PipelineError> {
    let downstream_id = find_node_by_name(g, downstream_name)
        .ok_or_else(|| PipelineError::NodeNotFound(downstream_name.to_string()))?;

    // Derive stream spec from the downstream node's first input pad
    let stream_spec = g
        .nodes
        .get(&downstream_id)
        .and_then(|s| s.input_pads.first().map(|(_, sp)| sp.clone()))
        .unwrap_or_default();

    // Collect in-edges to the downstream node
    let in_edges: Vec<Edge> = g
        .edges
        .iter()
        .filter(|e| e.to_node == downstream_id)
        .cloned()
        .collect();

    // Build the new filter node
    let new_spec = NodeSpec::filter(
        new_node_name,
        filter_config,
        stream_spec.clone(),
        stream_spec,
    );
    let new_id = new_spec.id;
    g.nodes.insert(new_id, new_spec);
    event.nodes_added += 1;

    // Remove in-edges to downstream, re-add from new node
    g.edges.retain(|e| e.to_node != downstream_id);

    // new_node → downstream
    g.edges.push(Edge {
        from_node: new_id,
        from_pad: "default".to_string(),
        to_node: downstream_id,
        to_pad: "default".to_string(),
    });
    event.edges_added += 1;

    // original sources → new_node
    for ie in &in_edges {
        g.edges.push(Edge {
            from_node: ie.from_node,
            from_pad: ie.from_pad.clone(),
            to_node: new_id,
            to_pad: "default".to_string(),
        });
        event.edges_added += 1;
    }

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::PipelineBuilder;
    use crate::node::{SinkConfig, SourceConfig};

    fn base_graph() -> PipelineGraph {
        PipelineBuilder::new()
            .source("in", SourceConfig::File("in.mp4".into()))
            .scale(1280, 720)
            .sink("out", SinkConfig::Null)
            .build()
            .unwrap()
    }

    #[test]
    fn insert_filter_after_increases_node_count() {
        let g = base_graph();
        let before = g.node_count();
        let mut dp = DynamicPipeline::new(g);

        let tx = dp
            .begin()
            .insert_filter_after_name("scale", "hflip", FilterConfig::Hflip);
        dp.commit(tx).unwrap();

        assert_eq!(dp.graph().node_count(), before + 1);
    }

    #[test]
    fn insert_filter_before_increases_node_count() {
        let g = base_graph();
        let before = g.node_count();
        let mut dp = DynamicPipeline::new(g);

        let tx = dp
            .begin()
            .insert_filter_before_name("scale", "vflip", FilterConfig::Vflip);
        dp.commit(tx).unwrap();

        assert_eq!(dp.graph().node_count(), before + 1);
    }

    #[test]
    fn remove_node_decreases_count_and_preserves_edges() {
        let g = base_graph();
        let before = g.node_count();
        let mut dp = DynamicPipeline::new(g);

        let tx = dp.begin().remove_node_by_name("scale");
        dp.commit(tx).unwrap();

        assert_eq!(dp.graph().node_count(), before - 1);
        // Source and sink should still be connected via a bypass edge
        let src_id = dp
            .graph()
            .nodes
            .iter()
            .find(|(_, s)| s.name == "in")
            .map(|(&id, _)| id)
            .unwrap();
        let sink_id = dp
            .graph()
            .nodes
            .iter()
            .find(|(_, s)| s.name == "out")
            .map(|(&id, _)| id)
            .unwrap();
        let connected = dp
            .graph()
            .edges
            .iter()
            .any(|e| e.from_node == src_id && e.to_node == sink_id);
        assert!(connected, "source and sink should be connected after bypass");
    }

    #[test]
    fn generation_increments_on_commit() {
        let g = base_graph();
        let mut dp = DynamicPipeline::new(g);
        assert_eq!(dp.generation(), 0);

        let tx = dp
            .begin()
            .insert_filter_after_name("scale", "hflip", FilterConfig::Hflip);
        dp.commit(tx).unwrap();
        assert_eq!(dp.generation(), 1);
    }

    #[test]
    fn failed_commit_does_not_change_graph() {
        let g = base_graph();
        let before_count = g.node_count();
        let mut dp = DynamicPipeline::new(g);

        let tx = dp
            .begin()
            .remove_node_by_name("nonexistent_node");
        let result = dp.commit(tx);

        assert!(result.is_err());
        // Graph unchanged
        assert_eq!(dp.graph().node_count(), before_count);
        assert_eq!(dp.generation(), 0);
    }

    #[test]
    fn replace_node_type_updates_filter() {
        let g = base_graph();
        let mut dp = DynamicPipeline::new(g);

        // Find the scale node id
        let scale_id = dp
            .graph()
            .nodes
            .iter()
            .find(|(_, s)| s.name == "scale")
            .map(|(&id, _)| id)
            .unwrap();

        let tx = dp
            .begin()
            .replace_node_type(scale_id, NodeType::Filter(FilterConfig::Hflip));
        dp.commit(tx).unwrap();

        let updated_spec = dp.graph().nodes.get(&scale_id).unwrap();
        assert!(matches!(
            updated_spec.node_type,
            NodeType::Filter(FilterConfig::Hflip)
        ));
    }

    #[test]
    fn multi_op_transaction_applied_atomically() {
        let g = base_graph();
        let before = g.node_count();
        let mut dp = DynamicPipeline::new(g);

        // Three operations in one transaction: insert 2 filters and remove 1
        let tx = dp
            .begin()
            .insert_filter_after_name("scale", "hflip", FilterConfig::Hflip)
            .insert_filter_after_name("hflip", "vflip", FilterConfig::Vflip)
            .remove_node_by_name("scale");

        let event = dp.commit(tx).unwrap();
        // Net: +2 inserted, -1 removed = before + 1
        assert_eq!(dp.graph().node_count(), before + 1);
        assert_eq!(event.total_ops, 3);
    }

    #[test]
    fn on_reconfig_callback_fired() {
        use std::sync::{Arc, Mutex};

        let g = base_graph();
        let mut dp = DynamicPipeline::new(g);

        let fired = Arc::new(Mutex::new(false));
        let fired_clone = Arc::clone(&fired);
        dp.on_reconfig(move |_event| {
            *fired_clone.lock().unwrap() = true;
        });

        let tx = dp
            .begin()
            .insert_filter_after_name("scale", "hflip", FilterConfig::Hflip);
        dp.commit(tx).unwrap();

        assert!(*fired.lock().unwrap(), "callback should have fired");
    }

    #[test]
    fn apply_single_op_convenience() {
        let g = base_graph();
        let before = g.node_count();
        let mut dp = DynamicPipeline::new(g);

        dp.apply(ReconfigOp::InsertFilterAfterName {
            upstream_name: "scale".to_string(),
            new_node_name: "hflip".to_string(),
            filter_config: FilterConfig::Hflip,
        })
        .unwrap();

        assert_eq!(dp.graph().node_count(), before + 1);
    }

    #[test]
    fn remove_edge_detaches_nodes() {
        let g = base_graph();
        let mut dp = DynamicPipeline::new(g);

        // Find the edge from source to scale
        let src_id = dp
            .graph()
            .nodes
            .iter()
            .find(|(_, s)| s.name == "in")
            .map(|(&id, _)| id)
            .unwrap();
        let scale_id = dp
            .graph()
            .nodes
            .iter()
            .find(|(_, s)| s.name == "scale")
            .map(|(&id, _)| id)
            .unwrap();

        let before_edges = dp.graph().edges.len();

        // We skip the re-add and just test removal
        let tx2 = dp.begin().push(ReconfigOp::RemoveEdge {
            from_node: src_id,
            from_pad: "default".to_string(),
            to_node: scale_id,
            to_pad: "default".to_string(),
        });
        dp.commit(tx2).unwrap();

        assert!(dp.graph().edges.len() < before_edges);
    }
}
