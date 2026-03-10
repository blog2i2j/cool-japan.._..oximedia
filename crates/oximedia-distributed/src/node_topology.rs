#![allow(dead_code)]
//! Network topology awareness for data-local task scheduling.
//!
//! Models the physical and logical topology of a distributed cluster so that
//! the scheduler can prefer nodes that are close to the data, minimising
//! network transfer costs.

use std::collections::{HashMap, HashSet};
use std::fmt;

/// Identifies a location tier in the topology hierarchy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LocationTier {
    /// Same physical host (local).
    Host,
    /// Same rack / network switch.
    Rack,
    /// Same data-centre / availability zone.
    DataCenter,
    /// Same geographic region.
    Region,
    /// Different region (cross-region).
    Remote,
}

impl LocationTier {
    /// Relative cost weight (lower is better).
    #[must_use]
    pub fn cost_weight(self) -> u32 {
        match self {
            Self::Host => 0,
            Self::Rack => 1,
            Self::DataCenter => 5,
            Self::Region => 20,
            Self::Remote => 100,
        }
    }
}

impl fmt::Display for LocationTier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Host => write!(f, "Host"),
            Self::Rack => write!(f, "Rack"),
            Self::DataCenter => write!(f, "DataCenter"),
            Self::Region => write!(f, "Region"),
            Self::Remote => write!(f, "Remote"),
        }
    }
}

/// Physical location descriptor for a node.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NodeLocation {
    /// Region label (e.g. "us-east-1").
    pub region: String,
    /// Data-centre / availability zone.
    pub data_center: String,
    /// Rack identifier.
    pub rack: String,
    /// Hostname.
    pub host: String,
}

impl NodeLocation {
    /// Create a new location.
    pub fn new(
        region: impl Into<String>,
        data_center: impl Into<String>,
        rack: impl Into<String>,
        host: impl Into<String>,
    ) -> Self {
        Self {
            region: region.into(),
            data_center: data_center.into(),
            rack: rack.into(),
            host: host.into(),
        }
    }

    /// Determine the tier of proximity between two locations.
    #[must_use]
    pub fn tier_to(&self, other: &Self) -> LocationTier {
        if self.host == other.host
            && self.rack == other.rack
            && self.data_center == other.data_center
            && self.region == other.region
        {
            LocationTier::Host
        } else if self.rack == other.rack
            && self.data_center == other.data_center
            && self.region == other.region
        {
            LocationTier::Rack
        } else if self.data_center == other.data_center {
            LocationTier::DataCenter
        } else if self.region == other.region {
            LocationTier::Region
        } else {
            LocationTier::Remote
        }
    }

    /// Cost weight of transferring data to another location.
    #[must_use]
    pub fn cost_to(&self, other: &Self) -> u32 {
        self.tier_to(other).cost_weight()
    }
}

/// A node registered in the topology.
#[derive(Debug, Clone)]
pub struct TopologyNode {
    /// Unique node identifier.
    pub node_id: String,
    /// Physical location.
    pub location: NodeLocation,
    /// Whether the node is currently available.
    pub available: bool,
    /// Set of data block IDs this node holds locally.
    pub local_data: HashSet<String>,
}

impl TopologyNode {
    /// Create a new topology node.
    pub fn new(node_id: impl Into<String>, location: NodeLocation) -> Self {
        Self {
            node_id: node_id.into(),
            location,
            available: true,
            local_data: HashSet::new(),
        }
    }

    /// Mark data as locally available on this node.
    pub fn add_local_data(&mut self, data_id: impl Into<String>) {
        self.local_data.insert(data_id.into());
    }

    /// Remove a data block reference.
    pub fn remove_local_data(&mut self, data_id: &str) {
        self.local_data.remove(data_id);
    }

    /// Check whether this node has a specific data block.
    #[must_use]
    pub fn has_data(&self, data_id: &str) -> bool {
        self.local_data.contains(data_id)
    }
}

/// The cluster topology manager.
#[derive(Debug, Clone)]
pub struct TopologyManager {
    /// All registered nodes.
    nodes: HashMap<String, TopologyNode>,
}

impl TopologyManager {
    /// Create an empty topology.
    #[must_use]
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
        }
    }

    /// Register a node.
    pub fn add_node(&mut self, node: TopologyNode) {
        self.nodes.insert(node.node_id.clone(), node);
    }

    /// Remove a node by ID.
    pub fn remove_node(&mut self, node_id: &str) -> Option<TopologyNode> {
        self.nodes.remove(node_id)
    }

    /// Get a node by ID.
    #[must_use]
    pub fn get_node(&self, node_id: &str) -> Option<&TopologyNode> {
        self.nodes.get(node_id)
    }

    /// Get a mutable reference to a node.
    pub fn get_node_mut(&mut self, node_id: &str) -> Option<&mut TopologyNode> {
        self.nodes.get_mut(node_id)
    }

    /// Number of nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// List all node IDs.
    #[must_use]
    pub fn node_ids(&self) -> Vec<&str> {
        self.nodes.keys().map(std::string::String::as_str).collect()
    }

    /// Find nodes that have a specific data block locally, sorted by availability.
    #[must_use]
    pub fn nodes_with_data(&self, data_id: &str) -> Vec<&TopologyNode> {
        let mut nodes: Vec<&TopologyNode> = self
            .nodes
            .values()
            .filter(|n| n.has_data(data_id) && n.available)
            .collect();
        // Stable sort: available nodes first (already filtered), then by node_id for determinism
        nodes.sort_by(|a, b| a.node_id.cmp(&b.node_id));
        nodes
    }

    /// Rank candidate nodes for a task that needs `data_id`, preferring
    /// nodes closest to `reference_location`.
    ///
    /// Returns node IDs sorted by ascending transfer cost.
    #[must_use]
    pub fn rank_by_locality(
        &self,
        data_id: &str,
        reference_location: &NodeLocation,
    ) -> Vec<(String, u32)> {
        let mut candidates: Vec<(String, u32)> = self
            .nodes
            .values()
            .filter(|n| n.available)
            .map(|n| {
                let mut cost = reference_location.cost_to(&n.location);
                // Bonus: if the node already has the data, cost is even lower
                if n.has_data(data_id) {
                    cost = cost.saturating_sub(1);
                }
                (n.node_id.clone(), cost)
            })
            .collect();
        candidates.sort_by_key(|&(_, cost)| cost);
        candidates
    }

    /// Get all available nodes in a given region.
    #[must_use]
    pub fn nodes_in_region(&self, region: &str) -> Vec<&TopologyNode> {
        self.nodes
            .values()
            .filter(|n| n.location.region == region && n.available)
            .collect()
    }

    /// Get all available nodes in a given data centre.
    #[must_use]
    pub fn nodes_in_data_center(&self, dc: &str) -> Vec<&TopologyNode> {
        self.nodes
            .values()
            .filter(|n| n.location.data_center == dc && n.available)
            .collect()
    }

    /// Get all available nodes on a given rack.
    #[must_use]
    pub fn nodes_in_rack(&self, rack: &str) -> Vec<&TopologyNode> {
        self.nodes
            .values()
            .filter(|n| n.location.rack == rack && n.available)
            .collect()
    }

    /// Set a node's availability.
    pub fn set_available(&mut self, node_id: &str, available: bool) {
        if let Some(node) = self.nodes.get_mut(node_id) {
            node.available = available;
        }
    }

    /// List distinct regions.
    #[must_use]
    pub fn regions(&self) -> Vec<String> {
        let mut set: HashSet<String> = HashSet::new();
        for n in self.nodes.values() {
            set.insert(n.location.region.clone());
        }
        let mut regions: Vec<String> = set.into_iter().collect();
        regions.sort();
        regions
    }
}

impl Default for TopologyManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn loc(region: &str, dc: &str, rack: &str, host: &str) -> NodeLocation {
        NodeLocation::new(region, dc, rack, host)
    }

    #[test]
    fn test_tier_same_host() {
        let a = loc("us", "dc1", "r1", "h1");
        let b = loc("us", "dc1", "r1", "h1");
        assert_eq!(a.tier_to(&b), LocationTier::Host);
    }

    #[test]
    fn test_tier_same_rack() {
        let a = loc("us", "dc1", "r1", "h1");
        let b = loc("us", "dc1", "r1", "h2");
        assert_eq!(a.tier_to(&b), LocationTier::Rack);
    }

    #[test]
    fn test_tier_same_dc() {
        let a = loc("us", "dc1", "r1", "h1");
        let b = loc("us", "dc1", "r2", "h3");
        assert_eq!(a.tier_to(&b), LocationTier::DataCenter);
    }

    #[test]
    fn test_tier_same_region() {
        let a = loc("us", "dc1", "r1", "h1");
        let b = loc("us", "dc2", "r5", "h9");
        assert_eq!(a.tier_to(&b), LocationTier::Region);
    }

    #[test]
    fn test_tier_remote() {
        let a = loc("us", "dc1", "r1", "h1");
        let b = loc("eu", "dc3", "r1", "h1");
        assert_eq!(a.tier_to(&b), LocationTier::Remote);
    }

    #[test]
    fn test_cost_ordering() {
        assert!(LocationTier::Host.cost_weight() < LocationTier::Rack.cost_weight());
        assert!(LocationTier::Rack.cost_weight() < LocationTier::DataCenter.cost_weight());
        assert!(LocationTier::DataCenter.cost_weight() < LocationTier::Region.cost_weight());
        assert!(LocationTier::Region.cost_weight() < LocationTier::Remote.cost_weight());
    }

    #[test]
    fn test_topology_manager_add_remove() {
        let mut mgr = TopologyManager::new();
        let node = TopologyNode::new("n1", loc("us", "dc1", "r1", "h1"));
        mgr.add_node(node);
        assert_eq!(mgr.node_count(), 1);
        mgr.remove_node("n1");
        assert_eq!(mgr.node_count(), 0);
    }

    #[test]
    fn test_nodes_with_data() {
        let mut mgr = TopologyManager::new();
        let mut n1 = TopologyNode::new("n1", loc("us", "dc1", "r1", "h1"));
        n1.add_local_data("block-42");
        let n2 = TopologyNode::new("n2", loc("us", "dc1", "r1", "h2"));
        mgr.add_node(n1);
        mgr.add_node(n2);
        let holders = mgr.nodes_with_data("block-42");
        assert_eq!(holders.len(), 1);
        assert_eq!(holders[0].node_id, "n1");
    }

    #[test]
    fn test_rank_by_locality() {
        let mut mgr = TopologyManager::new();
        let mut n_local = TopologyNode::new("local", loc("us", "dc1", "r1", "h1"));
        n_local.add_local_data("data-1");
        let n_remote = TopologyNode::new("remote", loc("eu", "dc3", "r1", "h9"));
        mgr.add_node(n_local);
        mgr.add_node(n_remote);
        let ref_loc = loc("us", "dc1", "r1", "h1");
        let ranked = mgr.rank_by_locality("data-1", &ref_loc);
        assert_eq!(ranked[0].0, "local");
        assert!(ranked[0].1 < ranked[1].1);
    }

    #[test]
    fn test_set_available() {
        let mut mgr = TopologyManager::new();
        mgr.add_node(TopologyNode::new("n1", loc("us", "dc1", "r1", "h1")));
        mgr.set_available("n1", false);
        assert!(!mgr.get_node("n1").expect("node should exist").available);
        mgr.set_available("n1", true);
        assert!(mgr.get_node("n1").expect("node should exist").available);
    }

    #[test]
    fn test_nodes_in_region() {
        let mut mgr = TopologyManager::new();
        mgr.add_node(TopologyNode::new("n1", loc("us", "dc1", "r1", "h1")));
        mgr.add_node(TopologyNode::new("n2", loc("eu", "dc2", "r1", "h1")));
        assert_eq!(mgr.nodes_in_region("us").len(), 1);
        assert_eq!(mgr.nodes_in_region("eu").len(), 1);
    }

    #[test]
    fn test_nodes_in_data_center() {
        let mut mgr = TopologyManager::new();
        mgr.add_node(TopologyNode::new("n1", loc("us", "dc1", "r1", "h1")));
        mgr.add_node(TopologyNode::new("n2", loc("us", "dc1", "r2", "h2")));
        mgr.add_node(TopologyNode::new("n3", loc("us", "dc2", "r1", "h3")));
        assert_eq!(mgr.nodes_in_data_center("dc1").len(), 2);
    }

    #[test]
    fn test_regions_list() {
        let mut mgr = TopologyManager::new();
        mgr.add_node(TopologyNode::new("n1", loc("us", "dc1", "r1", "h1")));
        mgr.add_node(TopologyNode::new("n2", loc("eu", "dc2", "r1", "h1")));
        mgr.add_node(TopologyNode::new("n3", loc("us", "dc3", "r1", "h2")));
        let regions = mgr.regions();
        assert_eq!(regions, vec!["eu", "us"]);
    }

    #[test]
    fn test_location_tier_display() {
        assert_eq!(LocationTier::Host.to_string(), "Host");
        assert_eq!(LocationTier::Remote.to_string(), "Remote");
    }

    #[test]
    fn test_node_local_data_operations() {
        let mut node = TopologyNode::new("n1", loc("us", "dc1", "r1", "h1"));
        node.add_local_data("d1");
        node.add_local_data("d2");
        assert!(node.has_data("d1"));
        assert!(node.has_data("d2"));
        node.remove_local_data("d1");
        assert!(!node.has_data("d1"));
        assert!(node.has_data("d2"));
    }
}
