//! Execution context for filter graphs.
//!
//! The context provides runtime state and statistics for graph execution.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::error::GraphResult;
use crate::frame::FramePool;
use crate::graph::FilterGraph;
use crate::node::NodeId;

/// Execution context for running filter graphs.
#[allow(dead_code)]
pub struct GraphContext {
    /// The filter graph being executed.
    graph: FilterGraph,
    /// Frame pool for allocation reuse.
    frame_pool: FramePool,
    /// Processing statistics.
    stats: ProcessingStats,
    /// Per-node statistics.
    node_stats: HashMap<NodeId, NodeStats>,
    /// Whether the context is currently running.
    running: bool,
    /// Start time of processing.
    start_time: Option<Instant>,
}

impl GraphContext {
    /// Create a new execution context for a graph.
    #[must_use]
    pub fn new(graph: FilterGraph) -> Self {
        let node_stats = graph
            .node_ids()
            .into_iter()
            .map(|id| (id, NodeStats::default()))
            .collect();

        Self {
            graph,
            frame_pool: FramePool::default(),
            stats: ProcessingStats::default(),
            node_stats,
            running: false,
            start_time: None,
        }
    }

    /// Get a reference to the underlying graph.
    #[must_use]
    pub fn graph(&self) -> &FilterGraph {
        &self.graph
    }

    /// Get a mutable reference to the underlying graph.
    pub fn graph_mut(&mut self) -> &mut FilterGraph {
        &mut self.graph
    }

    /// Get the frame pool.
    #[must_use]
    pub fn frame_pool(&self) -> &FramePool {
        &self.frame_pool
    }

    /// Get mutable access to the frame pool.
    pub fn frame_pool_mut(&mut self) -> &mut FramePool {
        &mut self.frame_pool
    }

    /// Get processing statistics.
    #[must_use]
    pub fn stats(&self) -> &ProcessingStats {
        &self.stats
    }

    /// Get statistics for a specific node.
    #[must_use]
    pub fn node_stats(&self, id: NodeId) -> Option<&NodeStats> {
        self.node_stats.get(&id)
    }

    /// Check if the context is currently running.
    #[must_use]
    pub fn is_running(&self) -> bool {
        self.running
    }

    /// Initialize the context for processing.
    pub fn initialize(&mut self) -> GraphResult<()> {
        self.graph.initialize()?;
        self.stats = ProcessingStats::default();
        for stats in self.node_stats.values_mut() {
            *stats = NodeStats::default();
        }
        self.running = false;
        self.start_time = None;
        Ok(())
    }

    /// Start processing.
    pub fn start(&mut self) -> GraphResult<()> {
        if !self.running {
            self.running = true;
            self.start_time = Some(Instant::now());
        }
        Ok(())
    }

    /// Process one step.
    pub fn step(&mut self) -> GraphResult<bool> {
        if !self.running {
            self.start()?;
        }

        let step_start = Instant::now();
        let processed = self.graph.process_step()?;
        let step_duration = step_start.elapsed();

        if processed {
            self.stats.frames_processed += 1;
            self.stats.total_processing_time += step_duration;
        }

        Ok(processed)
    }

    /// Stop processing.
    pub fn stop(&mut self) -> GraphResult<()> {
        if self.running {
            self.running = false;
            if let Some(start) = self.start_time.take() {
                self.stats.wall_clock_time = start.elapsed();
            }
        }
        Ok(())
    }

    /// Reset the context.
    pub fn reset(&mut self) -> GraphResult<()> {
        self.stop()?;
        self.graph.reset()?;
        self.frame_pool.clear();
        self.stats = ProcessingStats::default();
        for stats in self.node_stats.values_mut() {
            *stats = NodeStats::default();
        }
        Ok(())
    }

    /// Flush all pending frames.
    pub fn flush(&mut self) -> GraphResult<()> {
        let _ = self.graph.flush()?;
        Ok(())
    }

    /// Run the graph until completion or error.
    pub fn run_to_completion(&mut self) -> GraphResult<()> {
        self.initialize()?;
        self.start()?;

        loop {
            if !self.step()? {
                break;
            }
        }

        self.flush()?;
        self.stop()?;
        Ok(())
    }
}

/// Statistics for graph processing.
#[derive(Clone, Debug, Default)]
pub struct ProcessingStats {
    /// Total frames processed.
    pub frames_processed: u64,
    /// Total processing time (not wall clock).
    pub total_processing_time: Duration,
    /// Wall clock time from start to finish.
    pub wall_clock_time: Duration,
    /// Number of dropped frames.
    pub frames_dropped: u64,
    /// Number of errors encountered.
    pub errors: u64,
}

impl ProcessingStats {
    /// Get frames per second based on wall clock time.
    #[must_use]
    pub fn fps(&self) -> f64 {
        if self.wall_clock_time.is_zero() {
            0.0
        } else {
            #[allow(clippy::cast_precision_loss)]
            let result = self.frames_processed as f64 / self.wall_clock_time.as_secs_f64();
            result
        }
    }

    /// Get average processing time per frame.
    #[must_use]
    pub fn avg_frame_time(&self) -> Duration {
        if self.frames_processed == 0 {
            Duration::ZERO
        } else {
            self.total_processing_time / self.frames_processed as u32
        }
    }

    /// Get processing efficiency (processing time / wall clock time).
    #[must_use]
    pub fn efficiency(&self) -> f64 {
        if self.wall_clock_time.is_zero() {
            0.0
        } else {
            self.total_processing_time.as_secs_f64() / self.wall_clock_time.as_secs_f64()
        }
    }
}

/// Statistics for individual node processing.
#[derive(Clone, Debug, Default)]
pub struct NodeStats {
    /// Frames processed by this node.
    pub frames_processed: u64,
    /// Total processing time for this node.
    pub processing_time: Duration,
    /// Frames dropped by this node.
    pub frames_dropped: u64,
    /// Frames buffered in this node.
    pub frames_buffered: u64,
}

impl NodeStats {
    /// Get average processing time per frame.
    #[must_use]
    pub fn avg_frame_time(&self) -> Duration {
        if self.frames_processed == 0 {
            Duration::ZERO
        } else {
            self.processing_time / self.frames_processed as u32
        }
    }
}

/// Placeholder for thread pool integration.
///
/// This will be expanded to support parallel processing of independent
/// branches in the filter graph.
#[allow(dead_code)]
pub struct ThreadPoolConfig {
    /// Number of worker threads.
    pub num_threads: usize,
    /// Stack size per thread in bytes.
    pub stack_size: Option<usize>,
    /// Thread name prefix.
    pub name_prefix: String,
}

impl Default for ThreadPoolConfig {
    fn default() -> Self {
        Self {
            num_threads: num_cpus(),
            stack_size: None,
            name_prefix: "oximedia-worker".to_string(),
        }
    }
}

impl ThreadPoolConfig {
    /// Create a new thread pool configuration.
    #[must_use]
    pub fn new(num_threads: usize) -> Self {
        Self {
            num_threads,
            ..Default::default()
        }
    }

    /// Set the stack size.
    #[must_use]
    pub fn with_stack_size(mut self, size: usize) -> Self {
        self.stack_size = Some(size);
        self
    }

    /// Set the name prefix.
    #[must_use]
    pub fn with_name_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.name_prefix = prefix.into();
        self
    }
}

/// Get the number of available CPUs.
fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filters::video::{NullSink, PassthroughFilter};
    use crate::graph::GraphBuilder;
    use crate::node::NodeId;
    use crate::port::PortId;

    fn create_simple_graph() -> FilterGraph {
        let source = PassthroughFilter::new_source(NodeId(0), "source");
        let sink = NullSink::new(NodeId(0), "sink");

        let (builder, source_id) = GraphBuilder::new().add_node(Box::new(source));
        let (builder, sink_id) = builder.add_node(Box::new(sink));

        builder
            .connect(source_id, PortId(0), sink_id, PortId(0))
            .expect("operation should succeed")
            .build()
            .expect("operation should succeed")
    }

    #[test]
    fn test_context_creation() {
        let graph = create_simple_graph();
        let context = GraphContext::new(graph);

        assert!(!context.is_running());
        assert_eq!(context.stats().frames_processed, 0);
    }

    #[test]
    fn test_context_initialize() {
        let graph = create_simple_graph();
        let mut context = GraphContext::new(graph);

        context.initialize().expect("initialize should succeed");
        assert!(!context.is_running());
    }

    #[test]
    fn test_context_start_stop() {
        let graph = create_simple_graph();
        let mut context = GraphContext::new(graph);

        context.initialize().expect("initialize should succeed");
        context.start().expect("start should succeed");
        assert!(context.is_running());

        context.stop().expect("stop should succeed");
        assert!(!context.is_running());
    }

    #[test]
    fn test_context_reset() {
        let graph = create_simple_graph();
        let mut context = GraphContext::new(graph);

        context.initialize().expect("initialize should succeed");
        context.start().expect("start should succeed");
        context.reset().expect("reset should succeed");

        assert!(!context.is_running());
        assert_eq!(context.stats().frames_processed, 0);
    }

    #[test]
    fn test_processing_stats() {
        let mut stats = ProcessingStats::default();
        stats.frames_processed = 100;
        stats.wall_clock_time = Duration::from_secs(1);
        stats.total_processing_time = Duration::from_millis(500);

        assert!((stats.fps() - 100.0).abs() < 0.01);
        assert_eq!(stats.avg_frame_time(), Duration::from_millis(5));
        assert!((stats.efficiency() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_thread_pool_config() {
        let config = ThreadPoolConfig::new(4)
            .with_stack_size(1024 * 1024)
            .with_name_prefix("test");

        assert_eq!(config.num_threads, 4);
        assert_eq!(config.stack_size, Some(1024 * 1024));
        assert_eq!(config.name_prefix, "test");
    }

    #[test]
    fn test_num_cpus() {
        let cpus = num_cpus();
        assert!(cpus >= 1);
    }
}
