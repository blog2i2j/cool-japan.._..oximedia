//! Zero-latency path optimization for live monitoring chains.
//!
//! In live production, monitoring paths must have the absolute minimum
//! latency.  This module analyses a routing graph and finds paths that
//! avoid high-latency processing stages (e.g. sample-rate converters,
//! look-ahead limiters, or network hops with buffering).
//!
//! # Concepts
//!
//! * **Processing node** — any device or stage with a known latency cost
//!   (measured in samples at a reference rate).
//! * **Direct path** — a source→destination route that avoids all
//!   processing nodes whose latency exceeds a configurable threshold.
//! * **Latency budget** — the maximum acceptable end-to-end latency for
//!   a given path.
//!
//! # Example
//!
//! ```
//! use oximedia_routing::zero_latency::{ZeroLatencyOptimizer, ProcessingNode, MonitorPath};
//!
//! let mut opt = ZeroLatencyOptimizer::new(48_000);
//!
//! // Register processing stages.
//! opt.add_node(ProcessingNode::new("mic_pre", 0));       // zero-latency preamp
//! opt.add_node(ProcessingNode::new("src_44", 512));      // SRC adds 512 samples
//! opt.add_node(ProcessingNode::new("monitor", 0));       // monitor out
//!
//! // Direct connection (bypasses SRC).
//! opt.add_link("mic_pre", "monitor", 0);
//! // Through SRC.
//! opt.add_link("mic_pre", "src_44", 0);
//! opt.add_link("src_44", "monitor", 0);
//!
//! // Find the lowest-latency path.
//! let path = opt.find_lowest_latency("mic_pre", "monitor");
//! assert!(path.is_some());
//! let p = path.expect("path exists");
//! assert_eq!(p.total_latency_samples, 0);
//! ```

#![allow(dead_code)]

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

use serde::{Deserialize, Serialize};

/// A processing node in the routing topology.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingNode {
    /// Unique node identifier.
    pub id: String,
    /// Latency introduced by this node in samples.
    pub latency_samples: u64,
    /// Whether this node can be bypassed.
    pub bypassable: bool,
    /// Human-readable label.
    pub label: Option<String>,
    /// Node category (for filtering).
    pub category: NodeCategory,
}

/// Category of processing node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NodeCategory {
    /// Source (microphone, line in, etc.).
    Source,
    /// Destination (monitor, headphone, etc.).
    Destination,
    /// Digital signal processing (EQ, compressor, etc.).
    Dsp,
    /// Sample-rate converter.
    SampleRateConverter,
    /// Network hop (Dante, AES67, etc.).
    Network,
    /// Hardware I/O (ADC, DAC).
    HardwareIo,
    /// Other / generic.
    Other,
}

impl ProcessingNode {
    /// Creates a new node with zero latency and category `Other`.
    pub fn new(id: impl Into<String>, latency_samples: u64) -> Self {
        Self {
            id: id.into(),
            latency_samples,
            bypassable: false,
            label: None,
            category: NodeCategory::Other,
        }
    }

    /// Sets the node category.
    pub fn with_category(mut self, category: NodeCategory) -> Self {
        self.category = category;
        self
    }

    /// Marks the node as bypassable.
    pub fn with_bypassable(mut self, bypassable: bool) -> Self {
        self.bypassable = bypassable;
        self
    }

    /// Sets a human-readable label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Latency in milliseconds at the given sample rate.
    pub fn latency_ms(&self, sample_rate: u32) -> f64 {
        if sample_rate == 0 {
            return 0.0;
        }
        self.latency_samples as f64 / sample_rate as f64 * 1000.0
    }
}

/// A link between two nodes with optional additional latency (e.g. cable
/// propagation, buffer).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Link {
    /// Source node id.
    pub from: String,
    /// Destination node id.
    pub to: String,
    /// Additional latency introduced by the link itself (samples).
    pub link_latency_samples: u64,
}

/// A discovered path from source to destination.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitorPath {
    /// Ordered list of node ids from source to destination.
    pub nodes: Vec<String>,
    /// Total latency in samples.
    pub total_latency_samples: u64,
    /// Total latency in milliseconds at the reference sample rate.
    pub total_latency_ms: f64,
    /// Number of hops.
    pub hop_count: usize,
}

/// Entry in the Dijkstra priority queue.
#[derive(Debug, Clone, Eq, PartialEq)]
struct QueueEntry {
    cost: u64,
    node_id: String,
}

impl Ord for QueueEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: reverse ordering.
        other
            .cost
            .cmp(&self.cost)
            .then_with(|| self.node_id.cmp(&other.node_id))
    }
}

impl PartialOrd for QueueEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Zero-latency path optimizer.
///
/// Analyses a graph of processing nodes and links to find the path with
/// the absolute lowest latency between any source and destination.
#[derive(Debug, Clone)]
pub struct ZeroLatencyOptimizer {
    /// Reference sample rate for ms conversion.
    sample_rate: u32,
    /// All registered nodes keyed by id.
    nodes: HashMap<String, ProcessingNode>,
    /// Adjacency list: from_id → Vec<(to_id, link_latency)>.
    adjacency: HashMap<String, Vec<(String, u64)>>,
    /// Maximum latency budget in samples (0 = unlimited).
    max_latency_samples: u64,
    /// Categories to avoid when finding paths.
    avoid_categories: HashSet<NodeCategory>,
}

impl ZeroLatencyOptimizer {
    /// Creates a new optimizer with the given reference sample rate.
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            nodes: HashMap::new(),
            adjacency: HashMap::new(),
            max_latency_samples: 0,
            avoid_categories: HashSet::new(),
        }
    }

    /// Sets a maximum latency budget in samples.
    pub fn with_max_latency(mut self, samples: u64) -> Self {
        self.max_latency_samples = samples;
        self
    }

    /// Adds a node category to avoid when routing.
    pub fn avoid_category(&mut self, category: NodeCategory) {
        self.avoid_categories.insert(category);
    }

    /// Registers a processing node.
    pub fn add_node(&mut self, node: ProcessingNode) {
        let id = node.id.clone();
        self.nodes.insert(id.clone(), node);
        self.adjacency.entry(id).or_default();
    }

    /// Adds a directional link between two nodes.
    pub fn add_link(
        &mut self,
        from: impl Into<String>,
        to: impl Into<String>,
        link_latency_samples: u64,
    ) {
        let from = from.into();
        let to = to.into();
        self.adjacency
            .entry(from.clone())
            .or_default()
            .push((to, link_latency_samples));
    }

    /// Adds a bidirectional link.
    pub fn add_bidirectional_link(
        &mut self,
        a: impl Into<String> + Clone,
        b: impl Into<String> + Clone,
        link_latency_samples: u64,
    ) {
        let a_str: String = a.into();
        let b_str: String = b.into();
        self.add_link(a_str.clone(), b_str.clone(), link_latency_samples);
        self.add_link(b_str, a_str, link_latency_samples);
    }

    /// Number of registered nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Reference sample rate.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Returns node info by id.
    pub fn get_node(&self, id: &str) -> Option<&ProcessingNode> {
        self.nodes.get(id)
    }

    /// Finds the path with the lowest total latency from `source` to `dest`.
    ///
    /// Uses Dijkstra's algorithm where edge cost = destination node latency
    /// + link latency.  Nodes in avoided categories are skipped (unless they
    /// are the source or destination).
    pub fn find_lowest_latency(&self, source: &str, dest: &str) -> Option<MonitorPath> {
        if !self.nodes.contains_key(source) || !self.nodes.contains_key(dest) {
            return None;
        }

        let mut dist: HashMap<String, u64> = HashMap::new();
        let mut prev: HashMap<String, String> = HashMap::new();
        let mut heap = BinaryHeap::new();

        let src_latency = self.nodes.get(source).map_or(0, |n| n.latency_samples);
        dist.insert(source.to_string(), src_latency);
        heap.push(QueueEntry {
            cost: src_latency,
            node_id: source.to_string(),
        });

        while let Some(QueueEntry { cost, node_id }) = heap.pop() {
            if node_id == dest {
                break;
            }

            if let Some(&best) = dist.get(&node_id) {
                if cost > best {
                    continue;
                }
            }

            let neighbors = match self.adjacency.get(&node_id) {
                Some(v) => v.clone(),
                None => continue,
            };

            for (next_id, link_lat) in &neighbors {
                // Skip avoided categories (unless it's the destination).
                if next_id != dest {
                    if let Some(next_node) = self.nodes.get(next_id.as_str()) {
                        if self.avoid_categories.contains(&next_node.category) {
                            continue;
                        }
                    }
                }

                let next_node_lat = self
                    .nodes
                    .get(next_id.as_str())
                    .map_or(0, |n| n.latency_samples);
                let new_cost = cost + link_lat + next_node_lat;

                // Respect budget.
                if self.max_latency_samples > 0 && new_cost > self.max_latency_samples {
                    continue;
                }

                let current_best = dist.get(next_id.as_str()).copied().unwrap_or(u64::MAX);
                if new_cost < current_best {
                    dist.insert(next_id.clone(), new_cost);
                    prev.insert(next_id.clone(), node_id.clone());
                    heap.push(QueueEntry {
                        cost: new_cost,
                        node_id: next_id.clone(),
                    });
                }
            }
        }

        // Reconstruct path.
        if !dist.contains_key(dest) {
            return None;
        }

        let total = dist.get(dest).copied().unwrap_or(0);
        let mut path = Vec::new();
        let mut cur = dest.to_string();
        loop {
            path.push(cur.clone());
            match prev.get(&cur) {
                Some(p) => cur = p.clone(),
                None => break,
            }
        }
        path.reverse();

        if path.first().map(|s| s.as_str()) != Some(source) {
            return None;
        }

        let latency_ms = if self.sample_rate > 0 {
            total as f64 / self.sample_rate as f64 * 1000.0
        } else {
            0.0
        };

        Some(MonitorPath {
            hop_count: path.len().saturating_sub(1),
            nodes: path,
            total_latency_samples: total,
            total_latency_ms: latency_ms,
        })
    }

    /// Finds all paths from `source` to `dest` within the latency budget.
    ///
    /// Returns paths sorted by latency (lowest first), up to `max_results`.
    pub fn find_all_paths(&self, source: &str, dest: &str, max_results: usize) -> Vec<MonitorPath> {
        let mut results = Vec::new();
        let mut visited = HashSet::new();
        let mut current_path = Vec::new();

        self.dfs_paths(
            source,
            dest,
            &mut visited,
            &mut current_path,
            0,
            &mut results,
            max_results * 4, // collect extra, then sort and truncate
        );

        results.sort_by_key(|p| p.total_latency_samples);
        results.truncate(max_results);
        results
    }

    fn dfs_paths(
        &self,
        current: &str,
        dest: &str,
        visited: &mut HashSet<String>,
        path: &mut Vec<String>,
        latency: u64,
        results: &mut Vec<MonitorPath>,
        max_collect: usize,
    ) {
        if results.len() >= max_collect {
            return;
        }

        visited.insert(current.to_string());
        path.push(current.to_string());

        let node_lat = self.nodes.get(current).map_or(0, |n| n.latency_samples);
        let total = latency + node_lat;

        if self.max_latency_samples > 0 && total > self.max_latency_samples {
            path.pop();
            visited.remove(current);
            return;
        }

        if current == dest {
            let latency_ms = if self.sample_rate > 0 {
                total as f64 / self.sample_rate as f64 * 1000.0
            } else {
                0.0
            };
            results.push(MonitorPath {
                nodes: path.clone(),
                total_latency_samples: total,
                total_latency_ms: latency_ms,
                hop_count: path.len().saturating_sub(1),
            });
        } else if let Some(neighbors) = self.adjacency.get(current) {
            for (next_id, link_lat) in neighbors {
                if visited.contains(next_id.as_str()) {
                    continue;
                }
                // Skip avoided categories (unless destination).
                if next_id != dest {
                    if let Some(node) = self.nodes.get(next_id.as_str()) {
                        if self.avoid_categories.contains(&node.category) {
                            continue;
                        }
                    }
                }
                self.dfs_paths(
                    next_id,
                    dest,
                    visited,
                    path,
                    total + link_lat,
                    results,
                    max_collect,
                );
            }
        }

        path.pop();
        visited.remove(current);
    }

    /// Returns `true` if any path exists from source to dest.
    pub fn is_reachable(&self, source: &str, dest: &str) -> bool {
        self.find_lowest_latency(source, dest).is_some()
    }

    /// Total latency of the lowest-latency path in milliseconds, or `None`.
    pub fn lowest_latency_ms(&self, source: &str, dest: &str) -> Option<f64> {
        self.find_lowest_latency(source, dest)
            .map(|p| p.total_latency_ms)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn build_simple() -> ZeroLatencyOptimizer {
        let mut opt = ZeroLatencyOptimizer::new(48_000);
        opt.add_node(ProcessingNode::new("mic", 0).with_category(NodeCategory::Source));
        opt.add_node(ProcessingNode::new("monitor", 0).with_category(NodeCategory::Destination));
        opt.add_link("mic", "monitor", 0);
        opt
    }

    #[test]
    fn test_direct_zero_latency_path() {
        let opt = build_simple();
        let path = opt.find_lowest_latency("mic", "monitor");
        assert!(path.is_some());
        let p = path.expect("path should exist");
        assert_eq!(p.total_latency_samples, 0);
        assert_eq!(p.hop_count, 1);
    }

    #[test]
    fn test_path_with_processing_latency() {
        let mut opt = ZeroLatencyOptimizer::new(48_000);
        opt.add_node(ProcessingNode::new("mic", 0));
        opt.add_node(ProcessingNode::new("eq", 64));
        opt.add_node(ProcessingNode::new("monitor", 0));
        opt.add_link("mic", "eq", 0);
        opt.add_link("eq", "monitor", 0);
        let p = opt
            .find_lowest_latency("mic", "monitor")
            .expect("path should exist");
        assert_eq!(p.total_latency_samples, 64);
    }

    #[test]
    fn test_chooses_lower_latency_path() {
        let mut opt = ZeroLatencyOptimizer::new(48_000);
        opt.add_node(ProcessingNode::new("mic", 0));
        opt.add_node(ProcessingNode::new("src", 512)); // high latency
        opt.add_node(ProcessingNode::new("direct", 0)); // zero latency
        opt.add_node(ProcessingNode::new("monitor", 0));

        // Path through SRC: mic -> src -> monitor (512 samples)
        opt.add_link("mic", "src", 0);
        opt.add_link("src", "monitor", 0);

        // Direct path: mic -> direct -> monitor (0 samples)
        opt.add_link("mic", "direct", 0);
        opt.add_link("direct", "monitor", 0);

        let p = opt
            .find_lowest_latency("mic", "monitor")
            .expect("path should exist");
        assert_eq!(p.total_latency_samples, 0);
        assert!(p.nodes.contains(&"direct".to_string()));
    }

    #[test]
    fn test_link_latency_included() {
        let mut opt = ZeroLatencyOptimizer::new(48_000);
        opt.add_node(ProcessingNode::new("a", 0));
        opt.add_node(ProcessingNode::new("b", 0));
        opt.add_link("a", "b", 128); // cable delay
        let p = opt
            .find_lowest_latency("a", "b")
            .expect("path should exist");
        assert_eq!(p.total_latency_samples, 128);
    }

    #[test]
    fn test_no_path_returns_none() {
        let mut opt = ZeroLatencyOptimizer::new(48_000);
        opt.add_node(ProcessingNode::new("a", 0));
        opt.add_node(ProcessingNode::new("b", 0));
        // No link between them.
        assert!(opt.find_lowest_latency("a", "b").is_none());
    }

    #[test]
    fn test_nonexistent_node_returns_none() {
        let opt = build_simple();
        assert!(opt.find_lowest_latency("ghost", "monitor").is_none());
        assert!(opt.find_lowest_latency("mic", "ghost").is_none());
    }

    #[test]
    fn test_avoid_category() {
        let mut opt = ZeroLatencyOptimizer::new(48_000);
        opt.add_node(ProcessingNode::new("mic", 0));
        opt.add_node(
            ProcessingNode::new("src", 0).with_category(NodeCategory::SampleRateConverter),
        );
        opt.add_node(ProcessingNode::new("direct", 10));
        opt.add_node(ProcessingNode::new("monitor", 0));

        opt.add_link("mic", "src", 0);
        opt.add_link("src", "monitor", 0);
        opt.add_link("mic", "direct", 0);
        opt.add_link("direct", "monitor", 0);

        opt.avoid_category(NodeCategory::SampleRateConverter);

        let p = opt
            .find_lowest_latency("mic", "monitor")
            .expect("path should exist");
        // Should go through 'direct' (latency 10) not 'src' (latency 0 but avoided).
        assert!(p.nodes.contains(&"direct".to_string()));
        assert!(!p.nodes.contains(&"src".to_string()));
    }

    #[test]
    fn test_max_latency_budget() {
        let mut opt = ZeroLatencyOptimizer::new(48_000);
        opt = opt.with_max_latency(100);
        opt.add_node(ProcessingNode::new("a", 0));
        opt.add_node(ProcessingNode::new("b", 200)); // exceeds budget
        opt.add_link("a", "b", 0);

        assert!(opt.find_lowest_latency("a", "b").is_none());
    }

    #[test]
    fn test_latency_ms_conversion() {
        let node = ProcessingNode::new("test", 48_000);
        assert!((node.latency_ms(48_000) - 1000.0).abs() < 1e-6);
    }

    #[test]
    fn test_find_all_paths() {
        let mut opt = ZeroLatencyOptimizer::new(48_000);
        opt.add_node(ProcessingNode::new("mic", 0));
        opt.add_node(ProcessingNode::new("path_a", 10));
        opt.add_node(ProcessingNode::new("path_b", 20));
        opt.add_node(ProcessingNode::new("monitor", 0));

        opt.add_link("mic", "path_a", 0);
        opt.add_link("path_a", "monitor", 0);
        opt.add_link("mic", "path_b", 0);
        opt.add_link("path_b", "monitor", 0);

        let paths = opt.find_all_paths("mic", "monitor", 10);
        assert_eq!(paths.len(), 2);
        // Should be sorted by latency.
        assert!(paths[0].total_latency_samples <= paths[1].total_latency_samples);
    }

    #[test]
    fn test_is_reachable() {
        let opt = build_simple();
        assert!(opt.is_reachable("mic", "monitor"));
        assert!(!opt.is_reachable("monitor", "mic")); // unidirectional
    }

    #[test]
    fn test_bidirectional_link() {
        let mut opt = ZeroLatencyOptimizer::new(48_000);
        opt.add_node(ProcessingNode::new("a", 0));
        opt.add_node(ProcessingNode::new("b", 0));
        opt.add_bidirectional_link("a", "b", 0);
        assert!(opt.is_reachable("a", "b"));
        assert!(opt.is_reachable("b", "a"));
    }

    #[test]
    fn test_node_count() {
        let opt = build_simple();
        assert_eq!(opt.node_count(), 2);
    }

    #[test]
    fn test_get_node() {
        let opt = build_simple();
        let n = opt.get_node("mic");
        assert!(n.is_some());
        assert_eq!(n.expect("should exist").category, NodeCategory::Source);
    }

    #[test]
    fn test_lowest_latency_ms() {
        let mut opt = ZeroLatencyOptimizer::new(48_000);
        opt.add_node(ProcessingNode::new("a", 48)); // 1ms
        opt.add_node(ProcessingNode::new("b", 0));
        opt.add_link("a", "b", 0);
        let ms = opt.lowest_latency_ms("a", "b");
        assert!(ms.is_some());
        assert!((ms.expect("should exist") - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_node_with_label() {
        let node = ProcessingNode::new("eq", 32).with_label("Parametric EQ");
        assert_eq!(node.label.as_deref(), Some("Parametric EQ"));
    }

    #[test]
    fn test_node_bypassable() {
        let node = ProcessingNode::new("comp", 64).with_bypassable(true);
        assert!(node.bypassable);
    }

    #[test]
    fn test_zero_sample_rate_latency_ms() {
        let node = ProcessingNode::new("x", 100);
        assert!((node.latency_ms(0) - 0.0).abs() < f64::EPSILON);
    }
}
