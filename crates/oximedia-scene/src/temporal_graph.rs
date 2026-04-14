//! Temporal graph module for connecting scene analysis results across time.
//!
//! Builds a directed graph where nodes represent scene segments and edges encode
//! temporal relationships (sequential, parallel, flashback, flash-forward). The
//! graph enables narrative structure analysis, story arc detection, and timeline
//! visualisation — all via lightweight, patent-free graph algorithms.
//!
//! # Algorithms
//! - Topological sort (Kahn's algorithm) for narrative ordering
//! - BFS/DFS for reachability and path finding
//! - Narrative arc detection via scene-energy profiling (three-act structure)

use crate::error::{SceneError, SceneResult};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

/// Unique identifier for a scene node.
pub type NodeId = u64;

/// Type of temporal edge between two scene nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EdgeKind {
    /// One scene directly follows the other in real time.
    Sequential,
    /// Scenes happen concurrently (e.g. split-screen or parallel storylines).
    Parallel,
    /// The destination scene is a flashback relative to the source.
    Flashback,
    /// The destination scene is a flash-forward / prolepsis.
    FlashForward,
    /// Thematic connection (same location, character, or motif).
    Thematic,
}

impl EdgeKind {
    /// Human-readable label.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Sequential => "sequential",
            Self::Parallel => "parallel",
            Self::Flashback => "flashback",
            Self::FlashForward => "flash-forward",
            Self::Thematic => "thematic",
        }
    }

    /// True if this edge type implies a forward temporal direction.
    #[must_use]
    pub const fn is_forward(&self) -> bool {
        matches!(self, Self::Sequential | Self::FlashForward | Self::Thematic)
    }
}

/// A directed edge in the temporal graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalEdge {
    /// Source node identifier.
    pub from: NodeId,
    /// Destination node identifier.
    pub to: NodeId,
    /// Edge relationship type.
    pub kind: EdgeKind,
    /// Optional weight (e.g. transition strength 0.0–1.0).
    pub weight: f32,
}

impl TemporalEdge {
    /// Create a new edge.
    #[must_use]
    pub fn new(from: NodeId, to: NodeId, kind: EdgeKind, weight: f32) -> Self {
        Self {
            from,
            to,
            kind,
            weight: weight.clamp(0.0, 1.0),
        }
    }
}

/// Narrative phase of a scene node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NarrativePhase {
    /// Setup / introduction.
    Setup,
    /// Rising action / inciting incident.
    RisingAction,
    /// Climax.
    Climax,
    /// Falling action / resolution.
    FallingAction,
    /// Denouement / epilogue.
    Denouement,
    /// Not yet classified.
    Unclassified,
}

impl NarrativePhase {
    /// Human-readable label.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Setup => "setup",
            Self::RisingAction => "rising action",
            Self::Climax => "climax",
            Self::FallingAction => "falling action",
            Self::Denouement => "denouement",
            Self::Unclassified => "unclassified",
        }
    }
}

/// A node in the temporal scene graph representing one scene segment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneNode {
    /// Unique node identifier.
    pub id: NodeId,
    /// Start time in seconds.
    pub start_time: f64,
    /// End time in seconds.
    pub end_time: f64,
    /// Optional human-readable label (e.g. scene name or shot description).
    pub label: Option<String>,
    /// Normalised dramatic energy of this scene (0.0–1.0).
    pub dramatic_energy: f32,
    /// Narrative phase classification.
    pub phase: NarrativePhase,
    /// Arbitrary metadata key-value pairs.
    pub metadata: HashMap<String, String>,
}

impl SceneNode {
    /// Create a new scene node.
    pub fn new(id: NodeId, start_time: f64, end_time: f64) -> SceneResult<Self> {
        if end_time <= start_time {
            return Err(SceneError::InvalidParameter(format!(
                "end_time ({end_time}) must be > start_time ({start_time})"
            )));
        }
        Ok(Self {
            id,
            start_time,
            end_time,
            label: None,
            dramatic_energy: 0.0,
            phase: NarrativePhase::Unclassified,
            metadata: HashMap::new(),
        })
    }

    /// Duration of this scene in seconds.
    #[must_use]
    pub fn duration(&self) -> f64 {
        self.end_time - self.start_time
    }

    /// Set a human-readable label.
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Set dramatic energy.
    #[must_use]
    pub fn with_energy(mut self, energy: f32) -> Self {
        self.dramatic_energy = energy.clamp(0.0, 1.0);
        self
    }

    /// Set narrative phase.
    #[must_use]
    pub fn with_phase(mut self, phase: NarrativePhase) -> Self {
        self.phase = phase;
        self
    }

    /// Insert a metadata key-value pair.
    pub fn insert_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }
}

/// Result of a topological sort.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologicalOrder {
    /// Node IDs in topological order (source nodes first).
    pub order: Vec<NodeId>,
    /// True if the graph is a DAG (no cycles detected).
    pub is_dag: bool,
}

/// Summary of narrative arc analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NarrativeArc {
    /// Total duration in seconds.
    pub total_duration: f64,
    /// Node IDs in each narrative phase.
    pub phases: HashMap<String, Vec<NodeId>>,
    /// Index (0-based) of the peak dramatic energy node.
    pub climax_node: Option<NodeId>,
    /// Mean dramatic energy across all nodes.
    pub mean_energy: f32,
}

/// A directed temporal scene graph.
///
/// Nodes represent scene segments; edges encode temporal and narrative relationships.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalGraph {
    nodes: HashMap<NodeId, SceneNode>,
    /// Adjacency list: from → list of edge indices.
    adj: HashMap<NodeId, Vec<usize>>,
    edges: Vec<TemporalEdge>,
    next_id: NodeId,
}

impl Default for TemporalGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl TemporalGraph {
    /// Create an empty temporal graph.
    #[must_use]
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            adj: HashMap::new(),
            edges: Vec::new(),
            next_id: 1,
        }
    }

    // ── node management ───────────────────────────────────────────────────────

    /// Generate a unique node ID.
    pub fn next_node_id(&mut self) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Add a scene node to the graph. Duplicate IDs are rejected.
    pub fn add_node(&mut self, node: SceneNode) -> SceneResult<()> {
        if self.nodes.contains_key(&node.id) {
            return Err(SceneError::InvalidParameter(format!(
                "node {} already exists",
                node.id
            )));
        }
        self.adj.entry(node.id).or_default();
        self.nodes.insert(node.id, node);
        Ok(())
    }

    /// Remove a node and all its incident edges.
    pub fn remove_node(&mut self, id: NodeId) -> SceneResult<SceneNode> {
        let node = self
            .nodes
            .remove(&id)
            .ok_or_else(|| SceneError::InvalidParameter(format!("node {id} not found")))?;
        self.adj.remove(&id);
        // Remove edges involving this node from adjacency lists.
        for adj_list in self.adj.values_mut() {
            adj_list.retain(|&ei| self.edges[ei].from != id && self.edges[ei].to != id);
        }
        Ok(node)
    }

    /// Get an immutable reference to a node.
    #[must_use]
    pub fn node(&self, id: NodeId) -> Option<&SceneNode> {
        self.nodes.get(&id)
    }

    /// Get a mutable reference to a node.
    pub fn node_mut(&mut self, id: NodeId) -> Option<&mut SceneNode> {
        self.nodes.get_mut(&id)
    }

    /// Number of nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    // ── edge management ───────────────────────────────────────────────────────

    /// Add a directed edge between two existing nodes.
    pub fn add_edge(&mut self, edge: TemporalEdge) -> SceneResult<usize> {
        if !self.nodes.contains_key(&edge.from) {
            return Err(SceneError::InvalidParameter(format!(
                "source node {} not found",
                edge.from
            )));
        }
        if !self.nodes.contains_key(&edge.to) {
            return Err(SceneError::InvalidParameter(format!(
                "destination node {} not found",
                edge.to
            )));
        }
        let idx = self.edges.len();
        self.adj.entry(edge.from).or_default().push(idx);
        self.edges.push(edge);
        Ok(idx)
    }

    /// Number of edges.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Edges leaving a node.
    #[must_use]
    pub fn edges_from(&self, id: NodeId) -> Vec<&TemporalEdge> {
        self.adj
            .get(&id)
            .map(|indices| indices.iter().map(|&i| &self.edges[i]).collect())
            .unwrap_or_default()
    }

    // ── graph algorithms ──────────────────────────────────────────────────────

    /// Topological sort of sequential/thematic edges using Kahn's algorithm.
    ///
    /// Only edges with `is_forward() == true` contribute to ordering.
    /// Returns `TopologicalOrder::is_dag == false` if a cycle is detected.
    #[must_use]
    pub fn topological_sort(&self) -> TopologicalOrder {
        // Build in-degree map over forward edges only.
        let mut in_degree: HashMap<NodeId, usize> = self.nodes.keys().map(|&id| (id, 0)).collect();

        for edge in &self.edges {
            if edge.kind.is_forward() {
                *in_degree.entry(edge.to).or_insert(0) += 1;
            }
        }

        let mut queue: VecDeque<NodeId> = in_degree
            .iter()
            .filter_map(|(&id, &deg)| if deg == 0 { Some(id) } else { None })
            .collect();

        // Sort for determinism.
        let mut queue_vec: Vec<NodeId> = queue.drain(..).collect();
        queue_vec.sort_unstable();
        queue.extend(queue_vec);

        let mut order: Vec<NodeId> = Vec::with_capacity(self.nodes.len());

        while let Some(id) = queue.pop_front() {
            order.push(id);
            if let Some(adj_indices) = self.adj.get(&id) {
                let mut neighbours: Vec<NodeId> = adj_indices
                    .iter()
                    .filter_map(|&ei| {
                        let e = &self.edges[ei];
                        if e.kind.is_forward() {
                            Some(e.to)
                        } else {
                            None
                        }
                    })
                    .collect();
                neighbours.sort_unstable();
                for nb in neighbours {
                    let deg = in_degree.entry(nb).or_insert(0);
                    *deg = deg.saturating_sub(1);
                    if *deg == 0 {
                        queue.push_back(nb);
                    }
                }
            }
        }

        let is_dag = order.len() == self.nodes.len();
        TopologicalOrder { order, is_dag }
    }

    /// BFS reachability: all nodes reachable from `start` over any edge kind.
    #[must_use]
    pub fn reachable_from(&self, start: NodeId) -> HashSet<NodeId> {
        let mut visited = HashSet::new();
        if !self.nodes.contains_key(&start) {
            return visited;
        }
        let mut queue = VecDeque::new();
        queue.push_back(start);
        visited.insert(start);
        while let Some(id) = queue.pop_front() {
            if let Some(adj_indices) = self.adj.get(&id) {
                for &ei in adj_indices {
                    let nb = self.edges[ei].to;
                    if visited.insert(nb) {
                        queue.push_back(nb);
                    }
                }
            }
        }
        visited
    }

    /// Find the shortest path (fewest edges) from `start` to `end` via BFS.
    /// Returns `None` if no path exists.
    #[must_use]
    pub fn shortest_path(&self, start: NodeId, end: NodeId) -> Option<Vec<NodeId>> {
        if !self.nodes.contains_key(&start) || !self.nodes.contains_key(&end) {
            return None;
        }
        if start == end {
            return Some(vec![start]);
        }
        let mut visited = HashMap::new();
        visited.insert(start, start); // node → predecessor
        let mut queue = VecDeque::new();
        queue.push_back(start);
        while let Some(id) = queue.pop_front() {
            if let Some(adj_indices) = self.adj.get(&id) {
                for &ei in adj_indices {
                    let nb = self.edges[ei].to;
                    if let std::collections::hash_map::Entry::Vacant(e) = visited.entry(nb) {
                        e.insert(id);
                        if nb == end {
                            // Reconstruct path
                            let mut path = vec![end];
                            let mut cur = end;
                            while cur != start {
                                cur = visited[&cur];
                                path.push(cur);
                            }
                            path.reverse();
                            return Some(path);
                        }
                        queue.push_back(nb);
                    }
                }
            }
        }
        None
    }

    // ── narrative analysis ────────────────────────────────────────────────────

    /// Classify nodes into narrative phases using a three-act energy model.
    ///
    /// The approach: sort nodes by start time, split into acts proportionally,
    /// and label by local vs global energy.  The node with peak energy is the
    /// climax.
    pub fn assign_narrative_phases(&mut self) {
        if self.nodes.is_empty() {
            return;
        }

        // Sort nodes by start time.
        let mut sorted: Vec<NodeId> = self.nodes.keys().cloned().collect();
        sorted.sort_by(|&a, &b| {
            let ta = self.nodes[&a].start_time;
            let tb = self.nodes[&b].start_time;
            ta.partial_cmp(&tb).unwrap_or(std::cmp::Ordering::Equal)
        });

        let n = sorted.len();
        // Three-act split: ~25% setup, ~50% development, ~25% resolution.
        let act1_end = (n as f32 * 0.25).ceil() as usize;
        let act2_end = (n as f32 * 0.75).ceil() as usize;

        // Find peak energy index for climax.
        let climax_idx = sorted
            .iter()
            .enumerate()
            .max_by(|(_, &a), (_, &b)| {
                self.nodes[&a]
                    .dramatic_energy
                    .partial_cmp(&self.nodes[&b].dramatic_energy)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(act2_end.saturating_sub(1));

        for (i, &id) in sorted.iter().enumerate() {
            let phase = if i == climax_idx && i >= act1_end {
                NarrativePhase::Climax
            } else if i < act1_end {
                if i == 0 {
                    NarrativePhase::Setup
                } else {
                    NarrativePhase::RisingAction
                }
            } else if i < act2_end {
                NarrativePhase::RisingAction
            } else if i == n.saturating_sub(1) {
                NarrativePhase::Denouement
            } else {
                NarrativePhase::FallingAction
            };
            if let Some(node) = self.nodes.get_mut(&id) {
                node.phase = phase;
            }
        }
    }

    /// Build a narrative arc summary.
    #[must_use]
    pub fn narrative_arc(&self) -> NarrativeArc {
        let total_duration: f64 = self
            .nodes
            .values()
            .map(|n| n.end_time)
            .fold(f64::NEG_INFINITY, f64::max)
            - self
                .nodes
                .values()
                .map(|n| n.start_time)
                .fold(f64::INFINITY, f64::min);

        let total_duration = total_duration.max(0.0);

        let mut phases: HashMap<String, Vec<NodeId>> = HashMap::new();
        let mut climax_node = None;
        let mut peak_energy = -1.0f32;
        let mut energy_sum = 0.0f32;

        for node in self.nodes.values() {
            phases
                .entry(node.phase.label().to_string())
                .or_default()
                .push(node.id);
            if node.dramatic_energy > peak_energy {
                peak_energy = node.dramatic_energy;
                climax_node = Some(node.id);
            }
            energy_sum += node.dramatic_energy;
        }

        let mean_energy = if self.nodes.is_empty() {
            0.0
        } else {
            energy_sum / self.nodes.len() as f32
        };

        NarrativeArc {
            total_duration,
            phases,
            climax_node,
            mean_energy,
        }
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn build_linear_graph() -> (TemporalGraph, Vec<NodeId>) {
        let mut g = TemporalGraph::new();
        let ids: Vec<NodeId> = (1..=5).collect();
        for &id in &ids {
            let node = SceneNode::new(id, id as f64 - 1.0, id as f64)
                .unwrap()
                .with_energy((id as f32) * 0.15);
            g.add_node(node).unwrap();
        }
        // Sequential chain: 1→2→3→4→5
        for i in 0..ids.len() - 1 {
            let e = TemporalEdge::new(ids[i], ids[i + 1], EdgeKind::Sequential, 1.0);
            g.add_edge(e).unwrap();
        }
        (g, ids)
    }

    #[test]
    fn test_add_node_duplicate_rejected() {
        let mut g = TemporalGraph::new();
        let node1 = SceneNode::new(1, 0.0, 1.0).unwrap();
        let node2 = SceneNode::new(1, 1.0, 2.0).unwrap();
        assert!(g.add_node(node1).is_ok());
        assert!(g.add_node(node2).is_err());
    }

    #[test]
    fn test_edge_count() {
        let (g, _) = build_linear_graph();
        assert_eq!(g.edge_count(), 4);
        assert_eq!(g.node_count(), 5);
    }

    #[test]
    fn test_topological_sort_linear() {
        let (g, ids) = build_linear_graph();
        let topo = g.topological_sort();
        assert!(topo.is_dag);
        // Linear chain must appear in order
        for (i, id) in ids.iter().enumerate() {
            assert_eq!(topo.order[i], *id);
        }
    }

    #[test]
    fn test_reachable_from() {
        let (g, ids) = build_linear_graph();
        let reachable = g.reachable_from(ids[0]);
        // From node 1 everything is reachable
        assert_eq!(reachable.len(), 5);
        let reachable_from_last = g.reachable_from(*ids.last().unwrap());
        // Last node has no outgoing edges
        assert_eq!(reachable_from_last.len(), 1);
    }

    #[test]
    fn test_shortest_path() {
        let (g, ids) = build_linear_graph();
        let path = g.shortest_path(ids[0], ids[4]);
        assert!(path.is_some());
        let p = path.unwrap();
        assert_eq!(p.first(), Some(&ids[0]));
        assert_eq!(p.last(), Some(&ids[4]));
        assert_eq!(p.len(), 5);
    }

    #[test]
    fn test_shortest_path_same_node() {
        let mut g = TemporalGraph::new();
        g.add_node(SceneNode::new(1, 0.0, 1.0).unwrap()).unwrap();
        let path = g.shortest_path(1, 1);
        assert_eq!(path, Some(vec![1]));
    }

    #[test]
    fn test_narrative_phases_assigned() {
        let mut g = TemporalGraph::new();
        for i in 1..=6u64 {
            let node = SceneNode::new(i, i as f64 - 1.0, i as f64)
                .unwrap()
                .with_energy(if i == 4 { 0.95 } else { 0.3 });
            g.add_node(node).unwrap();
        }
        g.assign_narrative_phases();
        // Node 1 must be Setup
        assert_eq!(g.node(1).unwrap().phase, NarrativePhase::Setup);
        // Node 4 (highest energy) must be Climax
        assert_eq!(g.node(4).unwrap().phase, NarrativePhase::Climax);
        // Last node must be Denouement
        assert_eq!(g.node(6).unwrap().phase, NarrativePhase::Denouement);
    }

    #[test]
    fn test_narrative_arc_summary() {
        let (mut g, _) = build_linear_graph();
        g.assign_narrative_phases();
        let arc = g.narrative_arc();
        assert!(arc.total_duration > 0.0);
        assert!(arc.climax_node.is_some());
        assert!(arc.mean_energy > 0.0);
    }

    #[test]
    fn test_scene_node_invalid_times() {
        assert!(SceneNode::new(1, 5.0, 3.0).is_err());
        assert!(SceneNode::new(1, 5.0, 5.0).is_err());
    }

    #[test]
    fn test_edge_kind_labels_and_direction() {
        assert!(EdgeKind::Sequential.is_forward());
        assert!(!EdgeKind::Flashback.is_forward());
        assert!(!EdgeKind::Parallel.is_forward());
        assert!(EdgeKind::FlashForward.is_forward());
        assert!(EdgeKind::Thematic.is_forward());
        assert_eq!(EdgeKind::Flashback.label(), "flashback");
    }
}
