//! Cached signal-flow-graph validation.
//!
//! [`ValidateCache`] wraps a [`SignalFlowGraph`] and caches the result of
//! [`SignalFlowGraph::validate`].  The cache is invalidated whenever the
//! topology changes (node added/removed, edge added/removed).  Read-only
//! queries do **not** invalidate the cache, so repeated `validate()` calls
//! on a stable graph are essentially free.
//!
//! # Example
//!
//! ```
//! use oximedia_routing::validate_cache::ValidateCache;
//! use oximedia_routing::flow::{FlowEdge, SignalFlowGraph};
//!
//! let mut vc = ValidateCache::new();
//! let src = vc.add_input("Mic".into(), 1);
//! let dst = vc.add_output("Mon".into(), 1);
//! vc.connect(src, dst, FlowEdge::default()).expect("connect");
//!
//! // First call computes validation from scratch.
//! let r1 = vc.validate();
//! assert!(r1.is_valid);
//!
//! // Second call returns the cached result without re-computing.
//! let r2 = vc.validate();
//! assert_eq!(r1.errors.len(), r2.errors.len());
//! ```

#![allow(dead_code)]

use crate::flow::{FlowEdge, FlowError, NodeType, SignalFlowGraph, ValidationResult};
use petgraph::graph::NodeIndex;

/// A topology-version counter.
type Version = u64;

/// Wraps [`SignalFlowGraph`] with a validation result cache.
///
/// Every mutating method bumps an internal version counter.  `validate()`
/// re-computes only when the version has changed since the last call.
#[derive(Debug)]
pub struct ValidateCache {
    graph: SignalFlowGraph,
    /// Monotonically increasing topology version.
    version: Version,
    /// Cached validation result together with the version at which it was
    /// computed.
    cached: Option<(Version, ValidationResult)>,
    /// Total number of cache hits (for diagnostics).
    cache_hits: u64,
    /// Total number of cache misses.
    cache_misses: u64,
}

impl Default for ValidateCache {
    fn default() -> Self {
        Self::new()
    }
}

impl ValidateCache {
    // ------------------------------------------------------------------
    // Construction
    // ------------------------------------------------------------------

    /// Creates a new cache wrapping an empty graph.
    pub fn new() -> Self {
        Self {
            graph: SignalFlowGraph::new(),
            version: 0,
            cached: None,
            cache_hits: 0,
            cache_misses: 0,
        }
    }

    /// Creates a cache from an existing graph.
    pub fn from_graph(graph: SignalFlowGraph) -> Self {
        Self {
            graph,
            version: 1, // consider it already mutated
            cached: None,
            cache_hits: 0,
            cache_misses: 0,
        }
    }

    // ------------------------------------------------------------------
    // Mutating helpers (each bumps the version)
    // ------------------------------------------------------------------

    fn bump(&mut self) {
        self.version = self.version.wrapping_add(1);
    }

    /// Adds an input node to the graph.
    pub fn add_input(&mut self, label: String, channels: u8) -> NodeIndex {
        self.bump();
        self.graph.add_input(label, channels)
    }

    /// Adds an output node to the graph.
    pub fn add_output(&mut self, label: String, channels: u8) -> NodeIndex {
        self.bump();
        self.graph.add_output(label, channels)
    }

    /// Adds a processor node.
    pub fn add_processor(&mut self, label: String, processor_type: String) -> NodeIndex {
        self.bump();
        self.graph.add_processor(label, processor_type)
    }

    /// Adds a bus node.
    pub fn add_bus(&mut self, label: String, channels: u8) -> NodeIndex {
        self.bump();
        self.graph.add_bus(label, channels)
    }

    /// Connects two nodes.
    pub fn connect(
        &mut self,
        from: NodeIndex,
        to: NodeIndex,
        edge: FlowEdge,
    ) -> Result<(), FlowError> {
        self.bump();
        self.graph.connect(from, to, edge)
    }

    /// Disconnects two nodes.
    pub fn disconnect(&mut self, from: NodeIndex, to: NodeIndex) -> Result<(), FlowError> {
        self.bump();
        self.graph.disconnect(from, to)
    }

    /// Removes a node from the graph.
    pub fn remove_node(&mut self, node: NodeIndex) -> Result<NodeType, FlowError> {
        self.bump();
        self.graph.remove_node(node)
    }

    /// Clears the entire graph.
    pub fn clear(&mut self) {
        self.bump();
        self.graph.clear();
    }

    /// Explicitly invalidates the cache (e.g. after external changes).
    pub fn invalidate(&mut self) {
        self.bump();
    }

    // ------------------------------------------------------------------
    // Read-only queries (never invalidate)
    // ------------------------------------------------------------------

    /// Returns a reference to the underlying graph.
    pub fn graph(&self) -> &SignalFlowGraph {
        &self.graph
    }

    /// Number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Number of edges in the graph.
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Current topology version number.
    pub fn version(&self) -> Version {
        self.version
    }

    /// Number of cache hits since creation.
    pub fn cache_hits(&self) -> u64 {
        self.cache_hits
    }

    /// Number of cache misses since creation.
    pub fn cache_misses(&self) -> u64 {
        self.cache_misses
    }

    /// Hit-rate as a fraction in [0, 1].  Returns 0.0 when no queries
    /// have been made.
    pub fn hit_rate(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            return 0.0;
        }
        self.cache_hits as f64 / total as f64
    }

    /// Returns `true` if the cached result is still valid (version has not
    /// changed since the last `validate()` call).
    pub fn is_cache_valid(&self) -> bool {
        self.cached
            .as_ref()
            .map_or(false, |(v, _)| *v == self.version)
    }

    // ------------------------------------------------------------------
    // Cached validation
    // ------------------------------------------------------------------

    /// Returns the validation result for the current topology.
    ///
    /// If the topology has not changed since the last call, the cached result
    /// is returned directly (cache hit).  Otherwise the graph is re-validated
    /// and the new result is cached.
    pub fn validate(&mut self) -> ValidationResult {
        if let Some((v, ref result)) = self.cached {
            if v == self.version {
                self.cache_hits += 1;
                return result.clone();
            }
        }

        self.cache_misses += 1;
        let result = self.graph.validate();
        self.cached = Some((self.version, result.clone()));
        result
    }

    /// Resets hit/miss counters.
    pub fn reset_stats(&mut self) {
        self.cache_hits = 0;
        self.cache_misses = 0;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flow::FlowEdge;

    fn simple_graph() -> ValidateCache {
        let mut vc = ValidateCache::new();
        let src = vc.add_input("Mic".into(), 1);
        let dst = vc.add_output("Mon".into(), 1);
        vc.connect(src, dst, FlowEdge::default())
            .expect("connect should succeed");
        vc
    }

    #[test]
    fn test_validate_returns_valid_for_simple_graph() {
        let mut vc = simple_graph();
        let result = vc.validate();
        assert!(result.is_valid);
    }

    #[test]
    fn test_cache_hit_on_second_call() {
        let mut vc = simple_graph();
        let _r1 = vc.validate();
        assert_eq!(vc.cache_misses(), 1);
        assert_eq!(vc.cache_hits(), 0);

        let _r2 = vc.validate();
        assert_eq!(vc.cache_misses(), 1);
        assert_eq!(vc.cache_hits(), 1);
    }

    #[test]
    fn test_cache_invalidated_on_add_input() {
        let mut vc = simple_graph();
        let _r1 = vc.validate();
        assert!(vc.is_cache_valid());

        vc.add_input("Extra".into(), 2);
        assert!(!vc.is_cache_valid());
    }

    #[test]
    fn test_cache_invalidated_on_add_output() {
        let mut vc = simple_graph();
        let _r1 = vc.validate();
        vc.add_output("Extra".into(), 2);
        assert!(!vc.is_cache_valid());
    }

    #[test]
    fn test_cache_invalidated_on_connect() {
        let mut vc = ValidateCache::new();
        let a = vc.add_input("A".into(), 1);
        let b = vc.add_output("B".into(), 1);
        let _r1 = vc.validate();
        assert!(vc.is_cache_valid());

        vc.connect(a, b, FlowEdge::default())
            .expect("connect should succeed");
        assert!(!vc.is_cache_valid());
    }

    #[test]
    fn test_cache_invalidated_on_disconnect() {
        let mut vc = simple_graph();
        let _r1 = vc.validate();

        // Find the two nodes we connected
        let inputs = vc.graph().get_all_inputs();
        let outputs = vc.graph().get_all_outputs();
        if let (Some(&src), Some(&dst)) = (inputs.first(), outputs.first()) {
            vc.disconnect(src, dst).expect("disconnect should succeed");
            assert!(!vc.is_cache_valid());
        }
    }

    #[test]
    fn test_cache_invalidated_on_clear() {
        let mut vc = simple_graph();
        let _r1 = vc.validate();
        vc.clear();
        assert!(!vc.is_cache_valid());
    }

    #[test]
    fn test_explicit_invalidate() {
        let mut vc = simple_graph();
        let _r1 = vc.validate();
        assert!(vc.is_cache_valid());
        vc.invalidate();
        assert!(!vc.is_cache_valid());
    }

    #[test]
    fn test_from_graph() {
        let mut g = SignalFlowGraph::new();
        let a = g.add_input("Mic".into(), 1);
        let b = g.add_output("Mon".into(), 1);
        g.connect(a, b, FlowEdge::default())
            .expect("connect should succeed");

        let mut vc = ValidateCache::from_graph(g);
        assert!(!vc.is_cache_valid()); // no cached result yet
        let r = vc.validate();
        assert!(r.is_valid);
        assert!(vc.is_cache_valid());
    }

    #[test]
    fn test_hit_rate_no_queries() {
        let vc = ValidateCache::new();
        assert!((vc.hit_rate() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_hit_rate_all_hits() {
        let mut vc = simple_graph();
        let _r1 = vc.validate(); // miss
        let _r2 = vc.validate(); // hit
        let _r3 = vc.validate(); // hit
                                 // 2 hits / 3 total = 0.666...
        assert!((vc.hit_rate() - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_reset_stats() {
        let mut vc = simple_graph();
        let _r1 = vc.validate();
        let _r2 = vc.validate();
        assert!(vc.cache_hits() > 0 || vc.cache_misses() > 0);
        vc.reset_stats();
        assert_eq!(vc.cache_hits(), 0);
        assert_eq!(vc.cache_misses(), 0);
    }

    #[test]
    fn test_version_increases_monotonically() {
        let mut vc = ValidateCache::new();
        let v0 = vc.version();
        vc.add_input("A".into(), 1);
        let v1 = vc.version();
        vc.add_output("B".into(), 1);
        let v2 = vc.version();
        assert!(v1 > v0);
        assert!(v2 > v1);
    }

    #[test]
    fn test_node_and_edge_count_passthrough() {
        let vc = simple_graph();
        assert_eq!(vc.node_count(), 2);
        assert_eq!(vc.edge_count(), 1);
    }

    #[test]
    fn test_add_processor_invalidates() {
        let mut vc = simple_graph();
        let _r1 = vc.validate();
        vc.add_processor("EQ".into(), "equalizer".into());
        assert!(!vc.is_cache_valid());
    }

    #[test]
    fn test_add_bus_invalidates() {
        let mut vc = simple_graph();
        let _r1 = vc.validate();
        vc.add_bus("Mix".into(), 2);
        assert!(!vc.is_cache_valid());
    }

    #[test]
    fn test_remove_node_invalidates() {
        let mut vc = ValidateCache::new();
        let a = vc.add_input("A".into(), 1);
        let _r = vc.validate();
        assert!(vc.is_cache_valid());
        vc.remove_node(a).expect("remove should succeed");
        assert!(!vc.is_cache_valid());
    }

    #[test]
    fn test_default_trait() {
        let vc = ValidateCache::default();
        assert_eq!(vc.node_count(), 0);
        assert_eq!(vc.edge_count(), 0);
    }
}
