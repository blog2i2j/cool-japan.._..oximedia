//! Clock hierarchy graph for visualising synchronisation relationships.
//!
//! In a complex broadcast or data-centre facility, dozens or hundreds of
//! clocks form a tree rooted at one or more grandmasters.  This module
//! provides a directed graph representation of the clock hierarchy, allowing:
//!
//! - Building the tree from parent-child relationships.
//! - Computing levels (depth from root).
//! - Serialising to DOT format for visualisation with Graphviz.
//! - Detecting cycles and disconnected sub-trees.
//! - Collecting path-delay statistics along each branch.
//!
//! # Example
//! ```rust
//! use oximedia_timesync::clock_graph::{ClockGraph, ClockNode, ClockRole};
//!
//! let mut graph = ClockGraph::new();
//! let gm_id = graph.add_node(ClockNode::new("GM1".to_string(), ClockRole::Grandmaster));
//! let bc_id = graph.add_node(ClockNode::new("BC1".to_string(), ClockRole::BoundaryClock));
//! let oc_id = graph.add_node(ClockNode::new("OC1".to_string(), ClockRole::OrdinaryClock));
//!
//! graph.add_edge(gm_id, bc_id, 500).expect("valid edge");
//! graph.add_edge(bc_id, oc_id, 800).expect("valid edge");
//!
//! let dot = graph.to_dot();
//! assert!(dot.contains("GM1"));
//! ```

use crate::error::{TimeSyncError, TimeSyncResult};
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

// ---------------------------------------------------------------------------
// ClockRole
// ---------------------------------------------------------------------------

/// The role of a clock node in the PTP hierarchy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ClockRole {
    /// IEEE 1588 Grandmaster Clock.
    Grandmaster,
    /// IEEE 1588 Boundary Clock.
    BoundaryClock,
    /// IEEE 1588 Ordinary Clock (end device).
    OrdinaryClock,
    /// IEEE 1588 Transparent Clock (passthrough).
    TransparentClock,
    /// gPTP (IEEE 802.1AS) master.
    GptpMaster,
    /// gPTP (IEEE 802.1AS) slave.
    GptpSlave,
    /// NTP server (stratum 1/2).
    NtpServer,
    /// NTP client.
    NtpClient,
    /// Unknown / undetermined.
    Unknown,
}

impl ClockRole {
    /// Returns a short label for DOT rendering.
    #[must_use]
    pub fn dot_label(&self) -> &'static str {
        match self {
            Self::Grandmaster => "GM",
            Self::BoundaryClock => "BC",
            Self::OrdinaryClock => "OC",
            Self::TransparentClock => "TC",
            Self::GptpMaster => "gPTP-M",
            Self::GptpSlave => "gPTP-S",
            Self::NtpServer => "NTP-SRV",
            Self::NtpClient => "NTP-CLI",
            Self::Unknown => "?",
        }
    }

    /// Returns `true` for roles that can act as a sync source.
    #[must_use]
    pub fn is_source(&self) -> bool {
        matches!(
            self,
            Self::Grandmaster | Self::BoundaryClock | Self::GptpMaster | Self::NtpServer
        )
    }
}

impl fmt::Display for ClockRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.dot_label())
    }
}

// ---------------------------------------------------------------------------
// NodeId
// ---------------------------------------------------------------------------

/// Opaque identifier for a node in the clock graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(usize);

// ---------------------------------------------------------------------------
// ClockNode
// ---------------------------------------------------------------------------

/// A single clock device in the graph.
#[derive(Debug, Clone)]
pub struct ClockNode {
    /// Human-readable name (hostname, chassis ID, etc.)
    pub name: String,
    /// Role of this clock.
    pub role: ClockRole,
    /// Offset from its parent in nanoseconds (if known).
    pub offset_ns: Option<i64>,
    /// Depth level from root (0 = grandmaster, 1 = first-level slave, ...).
    pub level: usize,
    /// Optional user metadata string.
    pub metadata: String,
}

impl ClockNode {
    /// Creates a new clock node with the given name and role.
    #[must_use]
    pub fn new(name: String, role: ClockRole) -> Self {
        Self {
            name,
            role,
            offset_ns: None,
            level: 0,
            metadata: String::new(),
        }
    }

    /// Adds a metadata string.
    #[must_use]
    pub fn with_metadata(mut self, metadata: &str) -> Self {
        self.metadata = metadata.to_string();
        self
    }

    /// Sets the offset from parent.
    #[must_use]
    pub fn with_offset_ns(mut self, offset_ns: i64) -> Self {
        self.offset_ns = Some(offset_ns);
        self
    }
}

// ---------------------------------------------------------------------------
// ClockEdge
// ---------------------------------------------------------------------------

/// A directed sync relationship from a source clock to a slave clock.
#[derive(Debug, Clone)]
pub struct ClockEdge {
    /// Source node.
    pub from: NodeId,
    /// Sink (slave) node.
    pub to: NodeId,
    /// One-way path delay in nanoseconds (0 if unknown).
    pub path_delay_ns: u64,
}

// ---------------------------------------------------------------------------
// ClockGraph
// ---------------------------------------------------------------------------

/// Directed graph representing the clock synchronisation hierarchy.
pub struct ClockGraph {
    nodes: HashMap<NodeId, ClockNode>,
    edges: Vec<ClockEdge>,
    next_id: usize,
}

impl ClockGraph {
    /// Creates an empty clock graph.
    #[must_use]
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            next_id: 0,
        }
    }

    /// Adds a node and returns its [`NodeId`].
    pub fn add_node(&mut self, node: ClockNode) -> NodeId {
        let id = NodeId(self.next_id);
        self.next_id += 1;
        self.nodes.insert(id, node);
        id
    }

    /// Adds a directed sync edge from `from` to `to` with the given path delay.
    ///
    /// Returns an error if either node does not exist or if the edge would
    /// create a cycle.
    pub fn add_edge(&mut self, from: NodeId, to: NodeId, path_delay_ns: u64) -> TimeSyncResult<()> {
        if !self.nodes.contains_key(&from) {
            return Err(TimeSyncError::InvalidConfig(format!(
                "clock_graph: source node {:?} not found",
                from
            )));
        }
        if !self.nodes.contains_key(&to) {
            return Err(TimeSyncError::InvalidConfig(format!(
                "clock_graph: sink node {:?} not found",
                to
            )));
        }
        // Check for self-loop
        if from == to {
            return Err(TimeSyncError::InvalidConfig(
                "clock_graph: self-loop edge is not allowed".to_string(),
            ));
        }
        // Cycle detection: would adding from→to create a cycle?
        // i.e. can we reach `from` from `to` in the current graph?
        if self.is_reachable(to, from) {
            return Err(TimeSyncError::InvalidConfig(format!(
                "clock_graph: adding edge {:?}→{:?} would create a cycle",
                from, to
            )));
        }

        self.edges.push(ClockEdge {
            from,
            to,
            path_delay_ns,
        });
        Ok(())
    }

    /// Returns the number of nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Returns the number of edges.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Returns a reference to a node by ID.
    #[must_use]
    pub fn node(&self, id: NodeId) -> Option<&ClockNode> {
        self.nodes.get(&id)
    }

    /// Returns a mutable reference to a node by ID.
    pub fn node_mut(&mut self, id: NodeId) -> Option<&mut ClockNode> {
        self.nodes.get_mut(&id)
    }

    /// Recomputes the `level` field of every node using BFS from root nodes
    /// (nodes with no incoming edges).
    ///
    /// Nodes unreachable from any root keep `level = 0`.
    pub fn compute_levels(&mut self) {
        // Find roots: nodes with no incoming edges.
        let has_parent: HashSet<NodeId> = self.edges.iter().map(|e| e.to).collect();
        let roots: Vec<NodeId> = self
            .nodes
            .keys()
            .filter(|id| !has_parent.contains(id))
            .copied()
            .collect();

        // BFS from each root.
        let mut level_map: HashMap<NodeId, usize> = HashMap::new();
        let mut queue: VecDeque<(NodeId, usize)> = roots.into_iter().map(|id| (id, 0)).collect();

        while let Some((node_id, level)) = queue.pop_front() {
            let entry = level_map.entry(node_id).or_insert(level);
            if *entry > level {
                *entry = level;
            }
            // Enqueue children
            for edge in self.edges.iter().filter(|e| e.from == node_id) {
                if !level_map.contains_key(&edge.to) {
                    queue.push_back((edge.to, level + 1));
                }
            }
        }

        // Apply levels to nodes.
        for (id, node) in self.nodes.iter_mut() {
            node.level = level_map.get(id).copied().unwrap_or(0);
        }
    }

    /// Returns the path-delay sum from a root to `target`, or `None` if the
    /// node is not reachable.
    #[must_use]
    pub fn total_path_delay_ns(&self, target: NodeId) -> Option<u64> {
        // BFS / Dijkstra over accumulating path delay.
        let has_parent: HashSet<NodeId> = self.edges.iter().map(|e| e.to).collect();
        let roots: Vec<NodeId> = self
            .nodes
            .keys()
            .filter(|id| !has_parent.contains(id))
            .copied()
            .collect();

        let mut queue: VecDeque<(NodeId, u64)> = roots.into_iter().map(|id| (id, 0)).collect();
        let mut visited: HashSet<NodeId> = HashSet::new();

        while let Some((node_id, delay)) = queue.pop_front() {
            if node_id == target {
                return Some(delay);
            }
            if visited.contains(&node_id) {
                continue;
            }
            visited.insert(node_id);
            for edge in self.edges.iter().filter(|e| e.from == node_id) {
                queue.push_back((edge.to, delay.saturating_add(edge.path_delay_ns)));
            }
        }
        None
    }

    /// Returns nodes at a given level (depth from root).
    ///
    /// `compute_levels()` must be called first for meaningful results.
    #[must_use]
    pub fn nodes_at_level(&self, level: usize) -> Vec<NodeId> {
        self.nodes
            .iter()
            .filter(|(_, n)| n.level == level)
            .map(|(id, _)| *id)
            .collect()
    }

    /// Returns the direct children of `parent_id`.
    #[must_use]
    pub fn children_of(&self, parent_id: NodeId) -> Vec<NodeId> {
        self.edges
            .iter()
            .filter(|e| e.from == parent_id)
            .map(|e| e.to)
            .collect()
    }

    /// Returns the parent of `child_id` (first matching edge), if any.
    #[must_use]
    pub fn parent_of(&self, child_id: NodeId) -> Option<NodeId> {
        self.edges.iter().find(|e| e.to == child_id).map(|e| e.from)
    }

    /// Serialises the graph to Graphviz DOT format.
    ///
    /// Each node is labelled with its name and role.  Edges are annotated with
    /// path delay in nanoseconds.
    #[must_use]
    pub fn to_dot(&self) -> String {
        let mut out =
            String::from("digraph clock_hierarchy {\n  rankdir=TB;\n  node [shape=box];\n");

        // Emit nodes
        for (id, node) in &self.nodes {
            let label = format!(
                "{} [{}]{}",
                node.name,
                node.role,
                if let Some(off) = node.offset_ns {
                    format!("\\noffset={}ns", off)
                } else {
                    String::new()
                }
            );
            let shape = match node.role {
                ClockRole::Grandmaster => "diamond",
                ClockRole::BoundaryClock => "box",
                ClockRole::TransparentClock => "parallelogram",
                ClockRole::NtpServer | ClockRole::NtpClient => "ellipse",
                _ => "box",
            };
            out.push_str(&format!(
                "  n{} [label=\"{}\", shape={}];\n",
                id.0, label, shape
            ));
        }

        // Emit edges
        for edge in &self.edges {
            out.push_str(&format!(
                "  n{} -> n{} [label=\"{}ns\"];\n",
                edge.from.0, edge.to.0, edge.path_delay_ns
            ));
        }

        out.push_str("}\n");
        out
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Returns `true` if node `target` is reachable from `start` following
    /// directed edges.
    fn is_reachable(&self, start: NodeId, target: NodeId) -> bool {
        let mut visited: HashSet<NodeId> = HashSet::new();
        let mut queue: VecDeque<NodeId> = VecDeque::new();
        queue.push_back(start);
        while let Some(id) = queue.pop_front() {
            if id == target {
                return true;
            }
            if visited.contains(&id) {
                continue;
            }
            visited.insert(id);
            for edge in self.edges.iter().filter(|e| e.from == id) {
                queue.push_back(edge.to);
            }
        }
        false
    }
}

impl Default for ClockGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_graph() -> (ClockGraph, NodeId, NodeId, NodeId) {
        let mut g = ClockGraph::new();
        let gm = g.add_node(ClockNode::new("GM1".to_string(), ClockRole::Grandmaster));
        let bc = g.add_node(ClockNode::new("BC1".to_string(), ClockRole::BoundaryClock));
        let oc = g.add_node(ClockNode::new("OC1".to_string(), ClockRole::OrdinaryClock));
        g.add_edge(gm, bc, 500).expect("valid GM→BC");
        g.add_edge(bc, oc, 800).expect("valid BC→OC");
        (g, gm, bc, oc)
    }

    #[test]
    fn test_graph_node_count() {
        let (g, _, _, _) = simple_graph();
        assert_eq!(g.node_count(), 3);
        assert_eq!(g.edge_count(), 2);
    }

    #[test]
    fn test_graph_compute_levels() {
        let (mut g, gm, bc, oc) = simple_graph();
        g.compute_levels();
        assert_eq!(g.node(gm).map(|n| n.level), Some(0));
        assert_eq!(g.node(bc).map(|n| n.level), Some(1));
        assert_eq!(g.node(oc).map(|n| n.level), Some(2));
    }

    #[test]
    fn test_graph_total_path_delay() {
        let (g, _, _, oc) = simple_graph();
        // GM→BC: 500 ns, BC→OC: 800 ns → total = 1300 ns
        let delay = g.total_path_delay_ns(oc);
        assert_eq!(delay, Some(1300));
    }

    #[test]
    fn test_graph_children_of() {
        let (g, gm, bc, _) = simple_graph();
        let children = g.children_of(gm);
        assert_eq!(children, vec![bc]);
    }

    #[test]
    fn test_graph_parent_of() {
        let (g, gm, bc, _) = simple_graph();
        assert_eq!(g.parent_of(bc), Some(gm));
        assert_eq!(g.parent_of(gm), None);
    }

    #[test]
    fn test_graph_nodes_at_level() {
        let (mut g, gm, bc, oc) = simple_graph();
        let _ = (gm, bc, oc); // silence unused
        g.compute_levels();
        let level0 = g.nodes_at_level(0);
        let level1 = g.nodes_at_level(1);
        assert_eq!(level0.len(), 1);
        assert_eq!(level1.len(), 1);
    }

    #[test]
    fn test_graph_to_dot() {
        let (g, _, _, _) = simple_graph();
        let dot = g.to_dot();
        assert!(dot.starts_with("digraph clock_hierarchy"));
        assert!(dot.contains("GM1"));
        assert!(dot.contains("BC1"));
        assert!(dot.contains("OC1"));
        assert!(dot.contains("500ns"));
    }

    #[test]
    fn test_graph_cycle_detection() {
        let mut g = ClockGraph::new();
        let a = g.add_node(ClockNode::new("A".to_string(), ClockRole::OrdinaryClock));
        let b = g.add_node(ClockNode::new("B".to_string(), ClockRole::OrdinaryClock));
        g.add_edge(a, b, 100).expect("valid A→B");
        // Adding B→A would create a cycle
        let result = g.add_edge(b, a, 100);
        assert!(result.is_err(), "cycle should be rejected");
    }

    #[test]
    fn test_graph_self_loop_rejected() {
        let mut g = ClockGraph::new();
        let a = g.add_node(ClockNode::new("A".to_string(), ClockRole::OrdinaryClock));
        assert!(g.add_edge(a, a, 0).is_err());
    }

    #[test]
    fn test_graph_unknown_node_rejected() {
        let mut g = ClockGraph::new();
        let a = g.add_node(ClockNode::new("A".to_string(), ClockRole::OrdinaryClock));
        let missing = NodeId(999);
        assert!(g.add_edge(a, missing, 0).is_err());
    }

    #[test]
    fn test_clock_role_is_source() {
        assert!(ClockRole::Grandmaster.is_source());
        assert!(ClockRole::BoundaryClock.is_source());
        assert!(!ClockRole::OrdinaryClock.is_source());
        assert!(!ClockRole::NtpClient.is_source());
    }

    #[test]
    fn test_clock_role_display() {
        assert_eq!(format!("{}", ClockRole::Grandmaster), "GM");
        assert_eq!(format!("{}", ClockRole::BoundaryClock), "BC");
    }
}
