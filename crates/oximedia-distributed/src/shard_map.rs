#![allow(dead_code)]
//! Shard mapping for distributing data across nodes using consistent hashing.
//!
//! This module provides a virtual-node-based consistent hashing ring that maps
//! arbitrary keys to physical node assignments. It supports:
//! - Adding/removing nodes with automatic rebalancing
//! - Configurable virtual node count per physical node
//! - Key-to-node lookups with O(log n) performance
//! - Shard statistics and load factor computation

use std::collections::{BTreeMap, HashMap};
use std::fmt;

/// Identifier for a physical node in the cluster.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NodeId {
    /// Unique string identifier for the node.
    pub id: String,
    /// Human-readable label.
    pub label: String,
    /// Weight factor for virtual node count (1.0 = normal).
    pub weight: u32,
}

impl NodeId {
    /// Create a new node identifier.
    #[must_use]
    pub fn new(id: &str, label: &str) -> Self {
        Self {
            id: id.to_string(),
            label: label.to_string(),
            weight: 100,
        }
    }

    /// Create a node with a custom weight (percentage; 100 = normal).
    #[must_use]
    pub fn with_weight(mut self, weight: u32) -> Self {
        self.weight = weight;
        self
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}({})", self.label, self.id)
    }
}

/// A virtual node on the hash ring.
#[derive(Debug, Clone)]
struct VNode {
    /// Hash position on the ring.
    hash: u64,
    /// The physical node this virtual node belongs to.
    node_id: String,
    /// Virtual node index.
    vnode_index: u32,
}

/// Configuration for the shard map.
#[derive(Debug, Clone)]
pub struct ShardMapConfig {
    /// Base number of virtual nodes per physical node.
    pub vnodes_per_node: u32,
    /// Whether to use node weight for virtual node count scaling.
    pub use_weights: bool,
}

impl Default for ShardMapConfig {
    fn default() -> Self {
        Self {
            vnodes_per_node: 150,
            use_weights: true,
        }
    }
}

/// Consistent hash ring for shard mapping.
#[derive(Debug, Clone)]
pub struct ShardMap {
    /// The hash ring: hash -> `node_id`.
    ring: BTreeMap<u64, String>,
    /// Registered physical nodes.
    nodes: HashMap<String, NodeId>,
    /// Configuration.
    config: ShardMapConfig,
}

impl ShardMap {
    /// Create a new empty shard map with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            ring: BTreeMap::new(),
            nodes: HashMap::new(),
            config: ShardMapConfig::default(),
        }
    }

    /// Create a shard map with custom configuration.
    #[must_use]
    pub fn with_config(config: ShardMapConfig) -> Self {
        Self {
            ring: BTreeMap::new(),
            nodes: HashMap::new(),
            config,
        }
    }

    /// Add a node to the ring.
    pub fn add_node(&mut self, node: NodeId) {
        let vnode_count = self.effective_vnodes(&node);
        for i in 0..vnode_count {
            let key = format!("{}:vnode:{}", node.id, i);
            let hash = Self::hash_key(&key);
            self.ring.insert(hash, node.id.clone());
        }
        self.nodes.insert(node.id.clone(), node);
    }

    /// Remove a node from the ring.
    pub fn remove_node(&mut self, node_id: &str) -> bool {
        if let Some(node) = self.nodes.remove(node_id) {
            let vnode_count = self.effective_vnodes(&node);
            for i in 0..vnode_count {
                let key = format!("{}:vnode:{}", node.id, i);
                let hash = Self::hash_key(&key);
                self.ring.remove(&hash);
            }
            true
        } else {
            false
        }
    }

    /// Look up which node a key maps to.
    #[must_use]
    pub fn lookup(&self, key: &str) -> Option<&str> {
        if self.ring.is_empty() {
            return None;
        }
        let hash = Self::hash_key(key);
        // Find the first node at or after the hash
        if let Some((_h, node_id)) = self.ring.range(hash..).next() {
            return Some(node_id.as_str());
        }
        // Wrap around to the first node in the ring
        self.ring.values().next().map(std::string::String::as_str)
    }

    /// Get the number of physical nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get the total number of virtual nodes on the ring.
    #[must_use]
    pub fn vnode_count(&self) -> usize {
        self.ring.len()
    }

    /// Check if the ring contains a specific node.
    #[must_use]
    pub fn has_node(&self, node_id: &str) -> bool {
        self.nodes.contains_key(node_id)
    }

    /// Get all registered node IDs.
    #[must_use]
    pub fn node_ids(&self) -> Vec<&str> {
        self.nodes.keys().map(std::string::String::as_str).collect()
    }

    /// Compute load distribution: how many virtual nodes each physical node owns.
    #[must_use]
    pub fn load_distribution(&self) -> HashMap<String, usize> {
        let mut dist: HashMap<String, usize> = HashMap::new();
        for node_id in self.ring.values() {
            *dist.entry(node_id.clone()).or_insert(0) += 1;
        }
        dist
    }

    /// Compute the load factor (std dev / mean of vnode counts).
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn load_factor(&self) -> f64 {
        let dist = self.load_distribution();
        if dist.is_empty() {
            return 0.0;
        }
        let counts: Vec<f64> = dist.values().map(|&c| c as f64).collect();
        let mean = counts.iter().sum::<f64>() / counts.len() as f64;
        if mean == 0.0 {
            return 0.0;
        }
        let variance = counts.iter().map(|c| (c - mean).powi(2)).sum::<f64>() / counts.len() as f64;
        variance.sqrt() / mean
    }

    /// Compute effective virtual node count for a given physical node.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    fn effective_vnodes(&self, node: &NodeId) -> u32 {
        if self.config.use_weights {
            let scaled = f64::from(self.config.vnodes_per_node) * (f64::from(node.weight) / 100.0);
            scaled.round() as u32
        } else {
            self.config.vnodes_per_node
        }
    }

    /// Simple FNV-1a-style hash for deterministic key hashing.
    fn hash_key(key: &str) -> u64 {
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for byte in key.as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0100_0000_01b3);
        }
        hash
    }
}

impl Default for ShardMap {
    fn default() -> Self {
        Self::new()
    }
}

/// Shard assignment result for a batch of keys.
#[derive(Debug, Clone)]
pub struct BatchAssignment {
    /// Map of key -> assigned `node_id`.
    pub assignments: HashMap<String, String>,
    /// Keys that could not be assigned (empty ring).
    pub unassigned: Vec<String>,
}

impl BatchAssignment {
    /// Create a new empty batch assignment.
    #[must_use]
    pub fn new() -> Self {
        Self {
            assignments: HashMap::new(),
            unassigned: Vec::new(),
        }
    }

    /// Number of successfully assigned keys.
    #[must_use]
    pub fn assigned_count(&self) -> usize {
        self.assignments.len()
    }

    /// Number of unassigned keys.
    #[must_use]
    pub fn unassigned_count(&self) -> usize {
        self.unassigned.len()
    }
}

impl Default for BatchAssignment {
    fn default() -> Self {
        Self::new()
    }
}

/// Assign a batch of keys to nodes using the given shard map.
#[must_use]
pub fn batch_assign(shard_map: &ShardMap, keys: &[&str]) -> BatchAssignment {
    let mut result = BatchAssignment::new();
    for &key in keys {
        if let Some(node_id) = shard_map.lookup(key) {
            result
                .assignments
                .insert(key.to_string(), node_id.to_string());
        } else {
            result.unassigned.push(key.to_string());
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_id_creation() {
        let node = NodeId::new("node-1", "Worker 1");
        assert_eq!(node.id, "node-1");
        assert_eq!(node.label, "Worker 1");
        assert_eq!(node.weight, 100);
    }

    #[test]
    fn test_node_id_with_weight() {
        let node = NodeId::new("n1", "N1").with_weight(200);
        assert_eq!(node.weight, 200);
    }

    #[test]
    fn test_node_id_display() {
        let node = NodeId::new("n1", "Worker");
        assert_eq!(node.to_string(), "Worker(n1)");
    }

    #[test]
    fn test_shard_map_empty() {
        let sm = ShardMap::new();
        assert_eq!(sm.node_count(), 0);
        assert_eq!(sm.vnode_count(), 0);
        assert_eq!(sm.lookup("any-key"), None);
    }

    #[test]
    fn test_shard_map_add_node() {
        let mut sm = ShardMap::new();
        sm.add_node(NodeId::new("n1", "Node 1"));
        assert_eq!(sm.node_count(), 1);
        assert!(sm.has_node("n1"));
        assert!(sm.vnode_count() > 0);
    }

    #[test]
    fn test_shard_map_remove_node() {
        let mut sm = ShardMap::new();
        sm.add_node(NodeId::new("n1", "Node 1"));
        assert!(sm.remove_node("n1"));
        assert_eq!(sm.node_count(), 0);
        assert_eq!(sm.vnode_count(), 0);
        assert!(!sm.has_node("n1"));
    }

    #[test]
    fn test_shard_map_remove_nonexistent() {
        let mut sm = ShardMap::new();
        assert!(!sm.remove_node("nonexistent"));
    }

    #[test]
    fn test_shard_map_lookup_single_node() {
        let mut sm = ShardMap::new();
        sm.add_node(NodeId::new("n1", "Node 1"));
        // With a single node, all keys must map to it
        assert_eq!(sm.lookup("key-a"), Some("n1"));
        assert_eq!(sm.lookup("key-b"), Some("n1"));
        assert_eq!(sm.lookup("key-c"), Some("n1"));
    }

    #[test]
    fn test_shard_map_lookup_deterministic() {
        let mut sm = ShardMap::new();
        sm.add_node(NodeId::new("n1", "Node 1"));
        sm.add_node(NodeId::new("n2", "Node 2"));
        let result1 = sm
            .lookup("my-key")
            .expect("lookup should succeed")
            .to_string();
        let result2 = sm
            .lookup("my-key")
            .expect("lookup should succeed")
            .to_string();
        assert_eq!(result1, result2);
    }

    #[test]
    fn test_shard_map_distribution() {
        let mut sm = ShardMap::new();
        sm.add_node(NodeId::new("n1", "N1"));
        sm.add_node(NodeId::new("n2", "N2"));
        sm.add_node(NodeId::new("n3", "N3"));

        let dist = sm.load_distribution();
        assert_eq!(dist.len(), 3);
        // Each node should have roughly vnodes_per_node vnodes
        for count in dist.values() {
            assert!(*count > 0);
        }
    }

    #[test]
    fn test_shard_map_load_factor() {
        let mut sm = ShardMap::new();
        sm.add_node(NodeId::new("n1", "N1"));
        sm.add_node(NodeId::new("n2", "N2"));
        sm.add_node(NodeId::new("n3", "N3"));
        let lf = sm.load_factor();
        // Load factor should be small for equal-weight nodes
        assert!(lf < 0.5, "load factor too high: {}", lf);
    }

    #[test]
    fn test_shard_map_load_factor_empty() {
        let sm = ShardMap::new();
        assert_eq!(sm.load_factor(), 0.0);
    }

    #[test]
    fn test_shard_map_node_ids() {
        let mut sm = ShardMap::new();
        sm.add_node(NodeId::new("a", "A"));
        sm.add_node(NodeId::new("b", "B"));
        let mut ids = sm.node_ids();
        ids.sort();
        assert_eq!(ids, vec!["a", "b"]);
    }

    #[test]
    fn test_weighted_nodes() {
        let mut sm = ShardMap::with_config(ShardMapConfig {
            vnodes_per_node: 100,
            use_weights: true,
        });
        sm.add_node(NodeId::new("n1", "N1").with_weight(100));
        sm.add_node(NodeId::new("n2", "N2").with_weight(200));

        let dist = sm.load_distribution();
        let n1_count = dist.get("n1").copied().unwrap_or(0);
        let n2_count = dist.get("n2").copied().unwrap_or(0);
        // n2 should have roughly twice as many vnodes as n1
        assert!(
            n2_count > n1_count,
            "n2={} should be > n1={}",
            n2_count,
            n1_count
        );
    }

    #[test]
    fn test_batch_assign() {
        let mut sm = ShardMap::new();
        sm.add_node(NodeId::new("n1", "N1"));
        sm.add_node(NodeId::new("n2", "N2"));

        let keys = vec!["key1", "key2", "key3"];
        let result = batch_assign(&sm, &keys);
        assert_eq!(result.assigned_count(), 3);
        assert_eq!(result.unassigned_count(), 0);
    }

    #[test]
    fn test_batch_assign_empty_ring() {
        let sm = ShardMap::new();
        let keys = vec!["key1", "key2"];
        let result = batch_assign(&sm, &keys);
        assert_eq!(result.assigned_count(), 0);
        assert_eq!(result.unassigned_count(), 2);
    }

    #[test]
    fn test_default_config() {
        let config = ShardMapConfig::default();
        assert_eq!(config.vnodes_per_node, 150);
        assert!(config.use_weights);
    }

    #[test]
    fn test_shard_map_default_trait() {
        let sm = ShardMap::default();
        assert_eq!(sm.node_count(), 0);
    }
}
