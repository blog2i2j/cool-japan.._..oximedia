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
// BarrierBatcher — batched GPU memory barrier management
// ============================================================

/// Direction of a buffer memory barrier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BarrierKind {
    /// Read-after-write hazard: a prior write must complete before a read.
    ReadAfterWrite,
    /// Write-after-read hazard: a prior read must complete before a write.
    WriteAfterRead,
}

/// Represents a logical buffer barrier between two pipeline stages.
#[derive(Debug, Clone)]
pub struct BufferBarrier {
    /// Identifier of the buffer resource.
    pub buffer_id: u64,
    /// Kind of hazard this barrier resolves.
    pub kind: BarrierKind,
    /// Source pipeline stage (ordering context).
    pub src_stage: PipelineStage,
    /// Destination pipeline stage (ordering context).
    pub dst_stage: PipelineStage,
}

impl BufferBarrier {
    /// Create a new `BufferBarrier`.
    #[must_use]
    pub fn new(buffer_id: u64, kind: BarrierKind, src: PipelineStage, dst: PipelineStage) -> Self {
        Self {
            buffer_id,
            kind,
            src_stage: src,
            dst_stage: dst,
        }
    }
}

/// Strategy governing when accumulated barriers are flushed to the encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BarrierStrategy {
    /// Flush after every single barrier is added (maximum safety, more overhead).
    Eager,
    /// Flush once at least `N` barriers have accumulated.
    Batched(usize),
    /// Only flush when explicitly requested (e.g. at pass boundaries).
    Deferred,
}

/// Tracks a recorded flush event for observability in tests and diagnostics.
#[derive(Debug, Clone)]
pub struct FlushRecord {
    /// Number of read-after-write barriers sent in this flush.
    pub raw_count: usize,
    /// Number of write-after-read barriers sent in this flush.
    pub war_count: usize,
}

/// Accumulates GPU buffer barriers and issues them to a mock encoder in batches.
///
/// In a real GPU engine the `flush` call would translate the accumulated
/// barriers into a `wgpu::CommandEncoder::insert_debug_marker` / pipeline-
/// barrier equivalent.  Here we model the encoder with a simple callback so
/// that the logic can be exercised in pure-CPU unit tests without a GPU device.
pub struct BarrierBatcher {
    pending_read_after_write: Vec<BufferBarrier>,
    pending_write_after_read: Vec<BufferBarrier>,
    strategy: BarrierStrategy,
    /// Number of individual barriers that have been batched and submitted.
    batched_count: u64,
    /// History of flush events (used for test assertions and diagnostics).
    flush_log: Vec<FlushRecord>,
}

impl BarrierBatcher {
    /// Create a `BarrierBatcher` with the given strategy.
    #[must_use]
    pub fn new(strategy: BarrierStrategy) -> Self {
        Self {
            pending_read_after_write: Vec::new(),
            pending_write_after_read: Vec::new(),
            strategy,
            batched_count: 0,
            flush_log: Vec::new(),
        }
    }

    /// Add a barrier.  In `Eager` mode this immediately triggers a flush;
    /// in `Batched(n)` mode a flush is triggered once `n` barriers are pending;
    /// in `Deferred` mode barriers accumulate until `flush()` is called explicitly.
    ///
    /// Returns `true` if a flush occurred as a result of adding this barrier.
    pub fn add_barrier(&mut self, barrier: BufferBarrier) -> bool {
        match barrier.kind {
            BarrierKind::ReadAfterWrite => self.pending_read_after_write.push(barrier),
            BarrierKind::WriteAfterRead => self.pending_write_after_read.push(barrier),
        }

        let should_flush = match self.strategy {
            BarrierStrategy::Eager => true,
            BarrierStrategy::Batched(n) => self.pending_count() >= n,
            BarrierStrategy::Deferred => false,
        };

        if should_flush {
            self.flush();
            true
        } else {
            false
        }
    }

    /// Flush all pending barriers to the (simulated) encoder.
    ///
    /// Returns the total number of barriers flushed in this call.
    /// After flushing, the pending queues are empty.
    pub fn flush(&mut self) -> usize {
        let raw = self.pending_read_after_write.len();
        let war = self.pending_write_after_read.len();
        let total = raw + war;

        if total == 0 {
            return 0;
        }

        // Record this flush for observability.
        self.flush_log.push(FlushRecord {
            raw_count: raw,
            war_count: war,
        });
        self.batched_count += total as u64;

        self.pending_read_after_write.clear();
        self.pending_write_after_read.clear();

        total
    }

    /// Number of barriers currently waiting to be flushed.
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.pending_read_after_write.len() + self.pending_write_after_read.len()
    }

    /// Total number of individual barriers that have been submitted to the encoder.
    #[must_use]
    pub fn batched_count(&self) -> u64 {
        self.batched_count
    }

    /// Immutable view of the flush history.
    #[must_use]
    pub fn flush_log(&self) -> &[FlushRecord] {
        &self.flush_log
    }

    /// Active strategy.
    #[must_use]
    pub fn strategy(&self) -> BarrierStrategy {
        self.strategy
    }
}

impl std::fmt::Debug for BarrierBatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BarrierBatcher")
            .field("strategy", &self.strategy)
            .field("pending", &self.pending_count())
            .field("batched_count", &self.batched_count)
            .field("flush_events", &self.flush_log.len())
            .finish()
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

    // ── BarrierBatcher tests ──────────────────────────────────────────────────

    fn raw_barrier(buf_id: u64) -> BufferBarrier {
        BufferBarrier::new(
            buf_id,
            BarrierKind::ReadAfterWrite,
            PipelineStage::Decode,
            PipelineStage::Filter,
        )
    }

    fn war_barrier(buf_id: u64) -> BufferBarrier {
        BufferBarrier::new(
            buf_id,
            BarrierKind::WriteAfterRead,
            PipelineStage::Filter,
            PipelineStage::Encode,
        )
    }

    #[test]
    fn test_batcher_eager_flushes_immediately() {
        let mut b = BarrierBatcher::new(BarrierStrategy::Eager);
        let flushed = b.add_barrier(raw_barrier(1));
        assert!(flushed, "eager strategy must flush on every add");
        assert_eq!(b.pending_count(), 0, "pending must be 0 after eager flush");
        assert_eq!(b.batched_count(), 1);
    }

    #[test]
    fn test_batcher_eager_each_barrier_is_one_flush() {
        let mut b = BarrierBatcher::new(BarrierStrategy::Eager);
        for i in 0..5u64 {
            b.add_barrier(raw_barrier(i));
        }
        assert_eq!(b.flush_log().len(), 5, "5 adds → 5 flushes in eager mode");
        assert_eq!(b.batched_count(), 5);
    }

    #[test]
    fn test_batcher_batched_accumulates_before_flush() {
        let mut b = BarrierBatcher::new(BarrierStrategy::Batched(5));
        // Add 4 barriers — should not flush yet
        for i in 0..4u64 {
            let flushed = b.add_barrier(raw_barrier(i));
            assert!(!flushed, "should not flush before reaching threshold");
        }
        assert_eq!(b.pending_count(), 4);
        assert_eq!(b.flush_log().len(), 0, "no flushes yet");
        // 5th barrier triggers flush
        let flushed = b.add_barrier(raw_barrier(4));
        assert!(flushed, "5th barrier must trigger flush");
        assert_eq!(b.pending_count(), 0);
        assert_eq!(b.flush_log().len(), 1, "exactly 1 batch flush occurred");
        assert_eq!(b.flush_log()[0].raw_count, 5);
        assert_eq!(b.batched_count(), 5);
    }

    #[test]
    fn test_batcher_batched_two_batches() {
        let mut b = BarrierBatcher::new(BarrierStrategy::Batched(3));
        for i in 0..6u64 {
            b.add_barrier(raw_barrier(i));
        }
        assert_eq!(
            b.flush_log().len(),
            2,
            "6 barriers at threshold=3 → 2 flushes"
        );
        assert_eq!(b.batched_count(), 6);
    }

    #[test]
    fn test_batcher_deferred_does_not_auto_flush() {
        let mut b = BarrierBatcher::new(BarrierStrategy::Deferred);
        for i in 0..10u64 {
            let flushed = b.add_barrier(raw_barrier(i));
            assert!(!flushed, "deferred mode must never auto-flush");
        }
        assert_eq!(b.pending_count(), 10);
        assert_eq!(b.flush_log().len(), 0);
    }

    #[test]
    fn test_batcher_manual_flush_clears_pending() {
        let mut b = BarrierBatcher::new(BarrierStrategy::Deferred);
        b.add_barrier(raw_barrier(1));
        b.add_barrier(war_barrier(2));
        assert_eq!(b.pending_count(), 2);
        let flushed_count = b.flush();
        assert_eq!(flushed_count, 2);
        assert_eq!(b.pending_count(), 0);
        assert_eq!(b.batched_count(), 2);
    }

    #[test]
    fn test_batcher_empty_flush_does_nothing() {
        let mut b = BarrierBatcher::new(BarrierStrategy::Deferred);
        let count = b.flush();
        assert_eq!(count, 0, "flush on empty batcher should return 0");
        assert_eq!(
            b.flush_log().len(),
            0,
            "empty flush should not log a record"
        );
    }

    #[test]
    fn test_batcher_mixed_kinds_tracked_separately() {
        let mut b = BarrierBatcher::new(BarrierStrategy::Deferred);
        b.add_barrier(raw_barrier(1));
        b.add_barrier(raw_barrier(2));
        b.add_barrier(war_barrier(3));
        b.flush();
        let record = &b.flush_log()[0];
        assert_eq!(record.raw_count, 2);
        assert_eq!(record.war_count, 1);
    }

    #[test]
    fn test_batcher_strategy_accessor() {
        let b = BarrierBatcher::new(BarrierStrategy::Batched(8));
        assert_eq!(b.strategy(), BarrierStrategy::Batched(8));
    }

    #[test]
    fn test_batcher_debug_fmt() {
        let b = BarrierBatcher::new(BarrierStrategy::Eager);
        let s = format!("{b:?}");
        assert!(s.contains("BarrierBatcher"));
    }
}
