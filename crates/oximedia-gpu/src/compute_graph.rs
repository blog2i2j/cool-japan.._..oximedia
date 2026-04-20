//! GPU compute graph — a typed, topologically-ordered execution planner.
//!
//! Models the task graph used internally by modern GPU driver stacks (e.g.
//! DirectX 12 command lists, Vulkan render graphs, WebGPU compute passes).
//!
//! # Node types
//!
//! | Variant | Meaning |
//! |---------|---------|
//! | [`NodeKind::Kernel`] | Dispatch a compute shader / kernel. |
//! | [`NodeKind::Copy`] | Transfer data between buffers (DMA-style). |
//! | [`NodeKind::Barrier`] | Memory / execution barrier between stages. |
//!
//! # Workflow
//!
//! 1. Create a [`ComputeGraph`].
//! 2. Add nodes via [`ComputeGraph::add_node`].
//! 3. Add resource bindings via [`ComputeGraph::bind_resource`].
//! 4. Add edges via [`ComputeGraph::add_edge`].
//! 5. Call [`ComputeGraph::execution_order`] to obtain a valid ordering.
//! 6. Optionally call [`ComputeGraph::validate`] to check for binding issues.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use thiserror::Error;

// ─── Error ────────────────────────────────────────────────────────────────────

/// Errors produced by the compute graph.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum GraphError {
    /// A node with this ID does not exist.
    #[error("Node not found: {0}")]
    NodeNotFound(u32),
    /// A node with this ID has already been added.
    #[error("Duplicate node ID: {0}")]
    DuplicateNode(u32),
    /// Adding this edge would introduce a cycle.
    #[error("Edge from node {from} to node {to} would create a cycle")]
    CyclicEdge { from: u32, to: u32 },
    /// The graph contains a cycle (defensive check during ordering).
    #[error("Compute graph contains a cycle; cannot determine execution order")]
    CycleDetected,
    /// A required resource binding is missing.
    #[error("Node {node_id} is missing required resource binding '{resource}'")]
    MissingBinding { node_id: u32, resource: String },
    /// A resource is bound to an incompatible node type.
    #[error("Resource '{resource}' cannot be bound to a {node_kind:?} node")]
    IncompatibleBinding {
        resource: String,
        node_kind: NodeKind,
    },
}

// ─── NodeKind ─────────────────────────────────────────────────────────────────

/// The functional category of a compute graph node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NodeKind {
    /// A compute kernel dispatch.
    Kernel {
        /// Shader entry-point label.
        entry_point: String,
        /// Number of thread groups along X, Y, Z axes.
        dispatch: [u32; 3],
    },
    /// A buffer-to-buffer copy operation.
    Copy {
        /// Identifier of the source buffer.
        src_buffer: u32,
        /// Identifier of the destination buffer.
        dst_buffer: u32,
        /// Number of bytes to copy.
        byte_count: usize,
    },
    /// An execution or memory barrier.
    Barrier {
        /// Stage that must complete before the barrier.
        src_stage: PipelineStageFlags,
        /// Stage that must wait after the barrier.
        dst_stage: PipelineStageFlags,
    },
}

// ─── PipelineStageFlags ───────────────────────────────────────────────────────

/// Bit flags representing pipeline stages for barrier nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PipelineStageFlags(pub u32);

impl PipelineStageFlags {
    /// No stage.
    pub const NONE: Self = Self(0);
    /// Compute shader stage.
    pub const COMPUTE_SHADER: Self = Self(1 << 0);
    /// Transfer / DMA stage.
    pub const TRANSFER: Self = Self(1 << 1);
    /// Host-side (CPU) read/write stage.
    pub const HOST: Self = Self(1 << 2);
    /// All stages (convenience sentinel).
    pub const ALL: Self = Self(0xFFFF_FFFF);

    /// Return `true` if `other` is a subset of `self`.
    #[must_use]
    pub fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    /// Bitwise OR of two flag sets.
    #[must_use]
    pub fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

// ─── ResourceBinding ──────────────────────────────────────────────────────────

/// A named resource bound to a specific node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceBinding {
    /// Logical binding name (e.g. `"input_buffer"`, `"output_texture"`).
    pub name: String,
    /// Buffer or texture ID this binding points to.
    pub resource_id: u32,
    /// Access mode.
    pub access: ResourceAccess,
}

/// Read / write access mode for a resource binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceAccess {
    /// The node only reads from this resource.
    ReadOnly,
    /// The node only writes to this resource.
    WriteOnly,
    /// The node both reads and writes to this resource.
    ReadWrite,
}

// ─── GraphNode ────────────────────────────────────────────────────────────────

/// A single node in the compute graph.
#[derive(Debug, Clone)]
pub struct GraphNode {
    /// Unique ID within the graph.
    pub id: u32,
    /// Human-readable label for profiling / debugging.
    pub label: String,
    /// Functional type of this node.
    pub kind: NodeKind,
    /// Resource bindings declared for this node.
    pub bindings: Vec<ResourceBinding>,
}

impl GraphNode {
    /// Construct a new `GraphNode` with no bindings.
    #[must_use]
    pub fn new(id: u32, label: impl Into<String>, kind: NodeKind) -> Self {
        Self {
            id,
            label: label.into(),
            kind,
            bindings: Vec::new(),
        }
    }
}

// ─── ExecutionPlan ────────────────────────────────────────────────────────────

/// The result of topological ordering: an ordered list of node IDs.
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    /// Node IDs in valid execution order (dependencies first).
    pub order: Vec<u32>,
    /// Estimated total dispatch work (sum of kernel thread groups).
    pub total_dispatch_groups: u64,
    /// Number of barrier nodes in the graph.
    pub barrier_count: usize,
    /// Number of copy nodes in the graph.
    pub copy_count: usize,
}

// ─── ComputeGraph ─────────────────────────────────────────────────────────────

/// Directed acyclic graph of compute nodes with resource binding management.
pub struct ComputeGraph {
    /// All nodes, keyed by ID.
    nodes: BTreeMap<u32, GraphNode>,
    /// Forward adjacency: `adj[a]` = set of nodes that must run *after* `a`.
    adj: BTreeMap<u32, BTreeSet<u32>>,
    /// Reverse adjacency: `radj[b]` = set of nodes that `b` depends on.
    radj: BTreeMap<u32, BTreeSet<u32>>,
}

impl ComputeGraph {
    /// Create an empty compute graph.
    #[must_use]
    pub fn new() -> Self {
        Self {
            nodes: BTreeMap::new(),
            adj: BTreeMap::new(),
            radj: BTreeMap::new(),
        }
    }

    /// Add a node to the graph.
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::DuplicateNode`] if a node with the same ID already
    /// exists.
    pub fn add_node(&mut self, node: GraphNode) -> Result<(), GraphError> {
        if self.nodes.contains_key(&node.id) {
            return Err(GraphError::DuplicateNode(node.id));
        }
        let id = node.id;
        self.nodes.insert(id, node);
        self.adj.entry(id).or_default();
        self.radj.entry(id).or_default();
        Ok(())
    }

    /// Attach a resource binding to an existing node.
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::NodeNotFound`] if the node does not exist.
    /// Returns [`GraphError::IncompatibleBinding`] if the resource type is not
    /// appropriate for the node kind (e.g. binding a buffer to a `Barrier`).
    pub fn bind_resource(
        &mut self,
        node_id: u32,
        binding: ResourceBinding,
    ) -> Result<(), GraphError> {
        let node = self
            .nodes
            .get_mut(&node_id)
            .ok_or(GraphError::NodeNotFound(node_id))?;
        // Barrier nodes do not consume buffers.
        if matches!(node.kind, NodeKind::Barrier { .. }) {
            return Err(GraphError::IncompatibleBinding {
                resource: binding.name,
                node_kind: NodeKind::Barrier {
                    src_stage: PipelineStageFlags::NONE,
                    dst_stage: PipelineStageFlags::NONE,
                },
            });
        }
        node.bindings.push(binding);
        Ok(())
    }

    /// Add a directed edge: `from` → `to` (node `from` must execute before `to`).
    ///
    /// # Errors
    ///
    /// * [`GraphError::NodeNotFound`] if either node is missing.
    /// * [`GraphError::CyclicEdge`] if the edge would introduce a cycle.
    pub fn add_edge(&mut self, from: u32, to: u32) -> Result<(), GraphError> {
        if !self.nodes.contains_key(&from) {
            return Err(GraphError::NodeNotFound(from));
        }
        if !self.nodes.contains_key(&to) {
            return Err(GraphError::NodeNotFound(to));
        }
        // Cycle check: can `from` already be reached *from* `to`?
        if self.is_reachable(to, from) {
            return Err(GraphError::CyclicEdge { from, to });
        }
        self.adj.entry(from).or_default().insert(to);
        self.radj.entry(to).or_default().insert(from);
        Ok(())
    }

    /// Compute a valid [`ExecutionPlan`] via topological ordering (Kahn's
    /// algorithm with deterministic tie-breaking by node ID).
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::CycleDetected`] if the graph contains a cycle
    /// (defensive; the `add_edge` invariant should prevent this).
    pub fn execution_order(&self) -> Result<ExecutionPlan, GraphError> {
        let mut in_degree: BTreeMap<u32, usize> = self
            .nodes
            .keys()
            .map(|&id| (id, self.radj[&id].len()))
            .collect();

        let mut ready: BTreeSet<u32> = in_degree
            .iter()
            .filter_map(|(&id, &deg)| if deg == 0 { Some(id) } else { None })
            .collect();

        let mut order = Vec::with_capacity(self.nodes.len());

        while let Some(&next) = ready.iter().next() {
            ready.remove(&next);
            order.push(next);
            for &successor in self
                .adj
                .get(&next)
                .map_or(&BTreeSet::new() as &BTreeSet<u32>, |s| s)
            {
                let deg = in_degree.entry(successor).or_insert(0);
                *deg = deg.saturating_sub(1);
                if *deg == 0 {
                    ready.insert(successor);
                }
            }
        }

        if order.len() != self.nodes.len() {
            return Err(GraphError::CycleDetected);
        }

        // Build plan metrics.
        let mut total_dispatch_groups: u64 = 0;
        let mut barrier_count = 0usize;
        let mut copy_count = 0usize;

        for &id in &order {
            if let Some(node) = self.nodes.get(&id) {
                match &node.kind {
                    NodeKind::Kernel { dispatch, .. } => {
                        total_dispatch_groups +=
                            dispatch.iter().map(|&d| u64::from(d)).product::<u64>();
                    }
                    NodeKind::Copy { .. } => copy_count += 1,
                    NodeKind::Barrier { .. } => barrier_count += 1,
                }
            }
        }

        Ok(ExecutionPlan {
            order,
            total_dispatch_groups,
            barrier_count,
            copy_count,
        })
    }

    /// Validate that every `Kernel` and `Copy` node has at least one resource
    /// binding and that no required named bindings are missing.
    ///
    /// Barrier nodes are not checked (they do not use resource bindings).
    ///
    /// # Errors
    ///
    /// Returns the first [`GraphError::MissingBinding`] encountered.
    pub fn validate(&self) -> Result<(), GraphError> {
        for node in self.nodes.values() {
            match &node.kind {
                NodeKind::Kernel { .. } | NodeKind::Copy { .. } => {
                    if node.bindings.is_empty() {
                        return Err(GraphError::MissingBinding {
                            node_id: node.id,
                            resource: "<any>".to_string(),
                        });
                    }
                }
                NodeKind::Barrier { .. } => {} // no bindings required
            }
        }
        Ok(())
    }

    /// Number of nodes in the graph.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of directed edges in the graph.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.adj.values().map(|s| s.len()).sum()
    }

    /// Retrieve a node by ID.
    #[must_use]
    pub fn node(&self, id: u32) -> Option<&GraphNode> {
        self.nodes.get(&id)
    }

    /// Return the IDs of all direct predecessors (nodes that must run before
    /// `node_id`).
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::NodeNotFound`] if the ID is not registered.
    pub fn predecessors(&self, node_id: u32) -> Result<Vec<u32>, GraphError> {
        if !self.nodes.contains_key(&node_id) {
            return Err(GraphError::NodeNotFound(node_id));
        }
        Ok(self
            .radj
            .get(&node_id)
            .map_or(vec![], |s| s.iter().copied().collect()))
    }

    /// Return the IDs of all direct successors (nodes that must run after
    /// `node_id`).
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::NodeNotFound`] if the ID is not registered.
    pub fn successors(&self, node_id: u32) -> Result<Vec<u32>, GraphError> {
        if !self.nodes.contains_key(&node_id) {
            return Err(GraphError::NodeNotFound(node_id));
        }
        Ok(self
            .adj
            .get(&node_id)
            .map_or(vec![], |s| s.iter().copied().collect()))
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// BFS reachability following *forward* edges.
    fn is_reachable(&self, start: u32, target: u32) -> bool {
        if start == target {
            return true;
        }
        let mut visited = BTreeSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(start);
        while let Some(cur) = queue.pop_front() {
            if visited.contains(&cur) {
                continue;
            }
            visited.insert(cur);
            if let Some(succs) = self.adj.get(&cur) {
                for &s in succs {
                    if s == target {
                        return true;
                    }
                    queue.push_back(s);
                }
            }
        }
        false
    }
}

impl Default for ComputeGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn kernel_node(id: u32, dispatch: [u32; 3]) -> GraphNode {
        GraphNode::new(
            id,
            format!("kernel_{id}"),
            NodeKind::Kernel {
                entry_point: format!("main_{id}"),
                dispatch,
            },
        )
    }

    fn copy_node(id: u32, src: u32, dst: u32, bytes: usize) -> GraphNode {
        GraphNode::new(
            id,
            format!("copy_{id}"),
            NodeKind::Copy {
                src_buffer: src,
                dst_buffer: dst,
                byte_count: bytes,
            },
        )
    }

    fn barrier_node(id: u32) -> GraphNode {
        GraphNode::new(
            id,
            format!("barrier_{id}"),
            NodeKind::Barrier {
                src_stage: PipelineStageFlags::COMPUTE_SHADER,
                dst_stage: PipelineStageFlags::TRANSFER,
            },
        )
    }

    fn simple_binding(name: &str, resource_id: u32) -> ResourceBinding {
        ResourceBinding {
            name: name.to_string(),
            resource_id,
            access: ResourceAccess::ReadWrite,
        }
    }

    // ── PipelineStageFlags ────────────────────────────────────────────────────

    #[test]
    fn test_pipeline_stage_contains() {
        let combined = PipelineStageFlags::COMPUTE_SHADER.union(PipelineStageFlags::TRANSFER);
        assert!(combined.contains(PipelineStageFlags::COMPUTE_SHADER));
        assert!(combined.contains(PipelineStageFlags::TRANSFER));
        assert!(!combined.contains(PipelineStageFlags::HOST));
    }

    #[test]
    fn test_pipeline_stage_all_contains_any() {
        assert!(PipelineStageFlags::ALL.contains(PipelineStageFlags::COMPUTE_SHADER));
        assert!(PipelineStageFlags::ALL.contains(PipelineStageFlags::HOST));
    }

    // ── ComputeGraph – construction ───────────────────────────────────────────

    #[test]
    fn test_add_node_and_count() -> Result<(), GraphError> {
        let mut g = ComputeGraph::new();
        g.add_node(kernel_node(1, [4, 1, 1]))?;
        g.add_node(barrier_node(2))?;
        assert_eq!(g.node_count(), 2);
        Ok(())
    }

    #[test]
    fn test_add_duplicate_node_error() -> Result<(), GraphError> {
        let mut g = ComputeGraph::new();
        g.add_node(kernel_node(1, [1, 1, 1]))?;
        let err = g.add_node(kernel_node(1, [2, 2, 2]));
        assert!(matches!(err, Err(GraphError::DuplicateNode(1))));
        Ok(())
    }

    #[test]
    fn test_add_edge_increments_count() -> Result<(), GraphError> {
        let mut g = ComputeGraph::new();
        g.add_node(kernel_node(1, [1, 1, 1]))?;
        g.add_node(kernel_node(2, [1, 1, 1]))?;
        g.add_edge(1, 2)?;
        assert_eq!(g.edge_count(), 1);
        Ok(())
    }

    #[test]
    fn test_add_edge_unknown_node_error() -> Result<(), GraphError> {
        let mut g = ComputeGraph::new();
        g.add_node(kernel_node(1, [1, 1, 1]))?;
        assert!(matches!(
            g.add_edge(1, 99),
            Err(GraphError::NodeNotFound(99))
        ));
        Ok(())
    }

    #[test]
    fn test_add_cyclic_edge_error() -> Result<(), GraphError> {
        let mut g = ComputeGraph::new();
        g.add_node(kernel_node(1, [1, 1, 1]))?;
        g.add_node(kernel_node(2, [1, 1, 1]))?;
        g.add_edge(1, 2)?;
        let err = g.add_edge(2, 1);
        assert!(matches!(
            err,
            Err(GraphError::CyclicEdge { from: 2, to: 1 })
        ));
        Ok(())
    }

    // ── execution_order ───────────────────────────────────────────────────────

    #[test]
    fn test_execution_order_single_node() -> Result<(), GraphError> {
        let mut g = ComputeGraph::new();
        g.add_node(kernel_node(5, [8, 1, 1]))?;
        let plan = g.execution_order()?;
        assert_eq!(plan.order, vec![5]);
        assert_eq!(plan.total_dispatch_groups, 8);
        Ok(())
    }

    #[test]
    fn test_execution_order_linear_chain() -> Result<(), GraphError> {
        let mut g = ComputeGraph::new();
        for id in [1, 2, 3] {
            g.add_node(kernel_node(id, [2, 1, 1]))?;
        }
        g.add_edge(1, 2)?;
        g.add_edge(2, 3)?;
        let plan = g.execution_order()?;
        assert_eq!(plan.order, vec![1, 2, 3]);
        assert_eq!(plan.total_dispatch_groups, 6);
        Ok(())
    }

    #[test]
    fn test_execution_order_with_barrier_and_copy() -> Result<(), GraphError> {
        // kernel → barrier → copy
        let mut g = ComputeGraph::new();
        g.add_node(kernel_node(1, [4, 4, 1]))?;
        g.add_node(barrier_node(2))?;
        g.add_node(copy_node(3, 0, 1, 1024))?;
        g.add_edge(1, 2)?;
        g.add_edge(2, 3)?;
        let plan = g.execution_order()?;
        assert_eq!(plan.order, vec![1, 2, 3]);
        assert_eq!(plan.barrier_count, 1);
        assert_eq!(plan.copy_count, 1);
        assert_eq!(plan.total_dispatch_groups, 16);
        Ok(())
    }

    #[test]
    fn test_execution_order_independent_nodes_sorted_by_id() -> Result<(), GraphError> {
        let mut g = ComputeGraph::new();
        for id in [5, 3, 1] {
            g.add_node(kernel_node(id, [1, 1, 1]))?;
        }
        let plan = g.execution_order()?;
        assert_eq!(plan.order, vec![1, 3, 5]);
        Ok(())
    }

    // ── resource bindings ─────────────────────────────────────────────────────

    #[test]
    fn test_bind_resource_to_kernel() -> Result<(), GraphError> {
        let mut g = ComputeGraph::new();
        g.add_node(kernel_node(1, [1, 1, 1]))?;
        g.bind_resource(1, simple_binding("input", 10))?;
        let node = g.node(1).ok_or(GraphError::NodeNotFound(1))?;
        assert_eq!(node.bindings.len(), 1);
        Ok(())
    }

    #[test]
    fn test_bind_resource_to_barrier_fails() -> Result<(), GraphError> {
        let mut g = ComputeGraph::new();
        g.add_node(barrier_node(1))?;
        let err = g.bind_resource(1, simple_binding("buf", 0));
        assert!(matches!(err, Err(GraphError::IncompatibleBinding { .. })));
        Ok(())
    }

    #[test]
    fn test_bind_resource_unknown_node_fails() -> Result<(), GraphError> {
        let mut g = ComputeGraph::new();
        let err = g.bind_resource(99, simple_binding("buf", 0));
        assert!(matches!(err, Err(GraphError::NodeNotFound(99))));
        Ok(())
    }

    // ── validate ─────────────────────────────────────────────────────────────

    #[test]
    fn test_validate_passes_when_all_bound() -> Result<(), GraphError> {
        let mut g = ComputeGraph::new();
        let mut n = kernel_node(1, [1, 1, 1]);
        n.bindings.push(simple_binding("buf", 0));
        g.add_node(n)?;
        g.add_node(barrier_node(2))?;
        g.add_edge(1, 2)?;
        assert!(g.validate().is_ok());
        Ok(())
    }

    #[test]
    fn test_validate_fails_when_kernel_has_no_bindings() -> Result<(), GraphError> {
        let mut g = ComputeGraph::new();
        g.add_node(kernel_node(1, [1, 1, 1]))?;
        assert!(matches!(
            g.validate(),
            Err(GraphError::MissingBinding { node_id: 1, .. })
        ));
        Ok(())
    }

    // ── predecessors / successors ─────────────────────────────────────────────

    #[test]
    fn test_predecessors_and_successors() -> Result<(), GraphError> {
        let mut g = ComputeGraph::new();
        for id in [1, 2, 3] {
            g.add_node(kernel_node(id, [1, 1, 1]))?;
        }
        g.add_edge(1, 3)?;
        g.add_edge(2, 3)?;
        let mut preds = g.predecessors(3)?;
        preds.sort_unstable();
        assert_eq!(preds, vec![1, 2]);
        let succs_1 = g.successors(1)?;
        assert_eq!(succs_1, vec![3]);
        Ok(())
    }

    #[test]
    fn test_predecessors_unknown_node_error() -> Result<(), GraphError> {
        let g = ComputeGraph::new();
        assert!(matches!(
            g.predecessors(42),
            Err(GraphError::NodeNotFound(42))
        ));
        Ok(())
    }
}
