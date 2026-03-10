//! GPU processing pipeline management
//!
//! Provides a directed-acyclic-graph (DAG) style pipeline for composing GPU
//! processing stages. Pipeline nodes are connected via edges; the pipeline
//! validates that the graph is acyclic before execution.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// A stage in the GPU processing pipeline
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PipelineStage {
    /// Decode compressed media
    Decode,
    /// Colour-space conversion (e.g., YUV → RGB)
    Colorspace,
    /// Image filter (blur, sharpen, …)
    Filter,
    /// Encode to compressed output
    Encode,
    /// Render to display surface
    Display,
}

impl std::fmt::Display for PipelineStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Decode => write!(f, "Decode"),
            Self::Colorspace => write!(f, "Colorspace"),
            Self::Filter => write!(f, "Filter"),
            Self::Encode => write!(f, "Encode"),
            Self::Display => write!(f, "Display"),
        }
    }
}

/// A single node in the GPU pipeline
#[derive(Debug, Clone)]
pub struct PipelineNode {
    /// Unique identifier for this node
    pub id: u64,
    /// The processing stage this node represents
    pub stage: PipelineStage,
    /// Human-readable name
    pub name: String,
    /// Number of input connections
    pub input_count: usize,
    /// Number of output connections
    pub output_count: usize,
}

impl PipelineNode {
    /// Create a new pipeline node
    pub fn new(id: u64, stage: PipelineStage, name: impl Into<String>) -> Self {
        Self {
            id,
            stage,
            name: name.into(),
            input_count: 0,
            output_count: 0,
        }
    }
}

/// A directed-acyclic-graph GPU processing pipeline
#[derive(Debug, Clone)]
pub struct GpuPipeline {
    nodes: Vec<PipelineNode>,
    edges: Vec<(u64, u64)>,
    active: bool,
}

impl GpuPipeline {
    /// Create a new empty pipeline
    #[must_use]
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            active: false,
        }
    }

    /// Add a node to the pipeline; returns the node id
    pub fn add_node(&mut self, mut node: PipelineNode) -> u64 {
        let id = node.id;
        node.input_count = 0;
        node.output_count = 0;
        self.nodes.push(node);
        id
    }

    /// Connect two nodes by id (from → to)
    ///
    /// # Errors
    ///
    /// Returns an error if either node does not exist or if the connection
    /// would create a cycle.
    pub fn connect(&mut self, from: u64, to: u64) -> Result<(), String> {
        if self.find_node(from).is_none() {
            return Err(format!("Source node {from} not found"));
        }
        if self.find_node(to).is_none() {
            return Err(format!("Target node {to} not found"));
        }
        if from == to {
            return Err("Self-loop not allowed".to_string());
        }
        // Check for duplicate edge
        if self.edges.contains(&(from, to)) {
            return Err(format!("Edge ({from}, {to}) already exists"));
        }
        // Tentatively add and check for cycle
        self.edges.push((from, to));
        if self.has_cycle() {
            self.edges.pop();
            return Err(format!("Adding edge ({from}, {to}) would create a cycle"));
        }
        // Update port counts
        if let Some(n) = self.nodes.iter_mut().find(|n| n.id == from) {
            n.output_count += 1;
        }
        if let Some(n) = self.nodes.iter_mut().find(|n| n.id == to) {
            n.input_count += 1;
        }
        Ok(())
    }

    /// Validate the pipeline (no isolated sinks without a source, etc.)
    ///
    /// # Errors
    ///
    /// Returns an error describing the first validation problem found.
    pub fn validate(&self) -> Result<(), String> {
        if self.nodes.is_empty() {
            return Err("Pipeline has no nodes".to_string());
        }
        if self.has_cycle() {
            return Err("Pipeline contains a cycle".to_string());
        }
        Ok(())
    }

    /// Number of nodes in the pipeline
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Returns `true` if the pipeline is valid (non-empty, acyclic)
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.validate().is_ok()
    }

    /// Activate the pipeline for processing
    pub fn activate(&mut self) {
        self.active = true;
    }

    /// Deactivate the pipeline
    pub fn deactivate(&mut self) {
        self.active = false;
    }

    /// Whether the pipeline is currently active
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Access the node list
    #[must_use]
    pub fn nodes(&self) -> &[PipelineNode] {
        &self.nodes
    }

    /// Access the edge list
    #[must_use]
    pub fn edges(&self) -> &[(u64, u64)] {
        &self.edges
    }

    // ----- private helpers -----

    fn find_node(&self, id: u64) -> Option<&PipelineNode> {
        self.nodes.iter().find(|n| n.id == id)
    }

    /// Cycle detection via DFS
    fn has_cycle(&self) -> bool {
        let node_ids: Vec<u64> = self.nodes.iter().map(|n| n.id).collect();
        let mut visited = std::collections::HashSet::new();
        let mut stack = std::collections::HashSet::new();

        for &id in &node_ids {
            if self.dfs_cycle(id, &mut visited, &mut stack) {
                return true;
            }
        }
        false
    }

    fn dfs_cycle(
        &self,
        node: u64,
        visited: &mut std::collections::HashSet<u64>,
        stack: &mut std::collections::HashSet<u64>,
    ) -> bool {
        if stack.contains(&node) {
            return true;
        }
        if visited.contains(&node) {
            return false;
        }
        visited.insert(node);
        stack.insert(node);
        for &(from, to) in &self.edges {
            if from == node && self.dfs_cycle(to, visited, stack) {
                return true;
            }
        }
        stack.remove(&node);
        false
    }
}

impl Default for GpuPipeline {
    fn default() -> Self {
        Self::new()
    }
}

/// Aggregated performance metrics for a pipeline
#[derive(Debug, Clone)]
pub struct PipelineMetrics {
    /// Total frames successfully processed
    pub frames_processed: u64,
    /// Average frame processing latency in milliseconds
    pub avg_latency_ms: f64,
    /// Number of frames dropped due to backpressure / overflow
    pub dropped_frames: u64,
    /// GPU utilisation in [0.0, 1.0]
    pub utilization: f64,
}

impl PipelineMetrics {
    /// Create a zeroed metrics record
    #[must_use]
    pub fn new() -> Self {
        Self {
            frames_processed: 0,
            avg_latency_ms: 0.0,
            dropped_frames: 0,
            utilization: 0.0,
        }
    }

    /// Record a new frame with the given latency
    pub fn record_frame(&mut self, latency_ms: f64) {
        let n = self.frames_processed as f64;
        self.avg_latency_ms = (self.avg_latency_ms * n + latency_ms) / (n + 1.0);
        self.frames_processed += 1;
    }

    /// Record a dropped frame
    pub fn record_drop(&mut self) {
        self.dropped_frames += 1;
    }

    /// Drop rate in [0.0, 1.0]
    #[must_use]
    pub fn drop_rate(&self) -> f64 {
        let total = self.frames_processed + self.dropped_frames;
        if total == 0 {
            0.0
        } else {
            self.dropped_frames as f64 / total as f64
        }
    }
}

impl Default for PipelineMetrics {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// Unit tests
// ============================================================
#[cfg(test)]
mod tests {
    use super::*;

    fn make_node(id: u64, stage: PipelineStage) -> PipelineNode {
        PipelineNode::new(id, stage, format!("node_{id}"))
    }

    #[test]
    fn test_pipeline_new_is_empty() {
        let p = GpuPipeline::new();
        assert_eq!(p.node_count(), 0);
        assert!(!p.is_active());
    }

    #[test]
    fn test_add_node_returns_id() {
        let mut p = GpuPipeline::new();
        let id = p.add_node(make_node(42, PipelineStage::Decode));
        assert_eq!(id, 42);
        assert_eq!(p.node_count(), 1);
    }

    #[test]
    fn test_connect_nodes_ok() {
        let mut p = GpuPipeline::new();
        p.add_node(make_node(1, PipelineStage::Decode));
        p.add_node(make_node(2, PipelineStage::Colorspace));
        assert!(p.connect(1, 2).is_ok());
        assert_eq!(p.edges().len(), 1);
    }

    #[test]
    fn test_connect_missing_node_err() {
        let mut p = GpuPipeline::new();
        p.add_node(make_node(1, PipelineStage::Decode));
        assert!(p.connect(1, 99).is_err());
    }

    #[test]
    fn test_connect_self_loop_err() {
        let mut p = GpuPipeline::new();
        p.add_node(make_node(1, PipelineStage::Filter));
        assert!(p.connect(1, 1).is_err());
    }

    #[test]
    fn test_connect_duplicate_edge_err() {
        let mut p = GpuPipeline::new();
        p.add_node(make_node(1, PipelineStage::Decode));
        p.add_node(make_node(2, PipelineStage::Encode));
        p.connect(1, 2).expect("pipeline connection should succeed");
        assert!(p.connect(1, 2).is_err());
    }

    #[test]
    fn test_connect_cycle_detected() {
        let mut p = GpuPipeline::new();
        p.add_node(make_node(1, PipelineStage::Decode));
        p.add_node(make_node(2, PipelineStage::Filter));
        p.add_node(make_node(3, PipelineStage::Encode));
        p.connect(1, 2).expect("pipeline connection should succeed");
        p.connect(2, 3).expect("pipeline connection should succeed");
        assert!(p.connect(3, 1).is_err());
    }

    #[test]
    fn test_validate_empty_err() {
        let p = GpuPipeline::new();
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_validate_single_node_ok() {
        let mut p = GpuPipeline::new();
        p.add_node(make_node(1, PipelineStage::Display));
        assert!(p.validate().is_ok());
        assert!(p.is_valid());
    }

    #[test]
    fn test_activate_deactivate() {
        let mut p = GpuPipeline::new();
        p.activate();
        assert!(p.is_active());
        p.deactivate();
        assert!(!p.is_active());
    }

    #[test]
    fn test_port_counts_updated() {
        let mut p = GpuPipeline::new();
        p.add_node(make_node(1, PipelineStage::Decode));
        p.add_node(make_node(2, PipelineStage::Encode));
        p.connect(1, 2).expect("pipeline connection should succeed");
        let n1 = p
            .nodes()
            .iter()
            .find(|n| n.id == 1)
            .expect("find should return a result");
        let n2 = p
            .nodes()
            .iter()
            .find(|n| n.id == 2)
            .expect("find should return a result");
        assert_eq!(n1.output_count, 1);
        assert_eq!(n2.input_count, 1);
    }

    #[test]
    fn test_metrics_record_frame() {
        let mut m = PipelineMetrics::new();
        m.record_frame(10.0);
        m.record_frame(20.0);
        assert_eq!(m.frames_processed, 2);
        assert!((m.avg_latency_ms - 15.0).abs() < 1e-9);
    }

    #[test]
    fn test_metrics_drop_rate() {
        let mut m = PipelineMetrics::new();
        m.record_frame(5.0);
        m.record_drop();
        assert!((m.drop_rate() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_stage_display() {
        assert_eq!(PipelineStage::Decode.to_string(), "Decode");
        assert_eq!(PipelineStage::Display.to_string(), "Display");
    }
}
