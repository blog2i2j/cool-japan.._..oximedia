//! Physical farm topology — rack and machine organisation.
//!
//! [`FarmTopology`] maintains a flat registry of [`FarmNode`]s describing
//! where each worker lives in the physical (or virtual) data-centre layout.
//! This topology is used by locality-aware schedulers to prefer workers in
//! the same rack as the job's data source, reducing cross-rack bandwidth.
//!
//! # Example
//!
//! ```
//! use oximedia_farm::topology::{FarmNode, FarmTopology};
//!
//! let mut topo = FarmTopology::new();
//! topo.add_node(FarmNode::new("rack-01", "host-a", 1));
//! topo.add_node(FarmNode::new("rack-01", "host-b", 2));
//! topo.add_node(FarmNode::new("rack-02", "host-c", 3));
//!
//! let in_rack1 = topo.nodes_in_rack("rack-01");
//! assert_eq!(in_rack1.len(), 2);
//! assert_eq!(topo.worker_count(), 3);
//! ```

/// A single worker node with its physical location metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FarmNode {
    /// Rack identifier (e.g. `"rack-01"`).
    pub rack: String,
    /// Machine hostname within the rack (e.g. `"encode-node-7"`).
    pub machine: String,
    /// Logical worker ID — must be unique within the topology.
    pub worker_id: u64,
}

impl FarmNode {
    /// Create a new node entry.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_farm::topology::FarmNode;
    ///
    /// let node = FarmNode::new("rack-03", "enc-node-12", 99);
    /// assert_eq!(node.rack, "rack-03");
    /// assert_eq!(node.worker_id, 99);
    /// ```
    #[must_use]
    pub fn new(rack: &str, machine: &str, worker_id: u64) -> Self {
        Self {
            rack: rack.to_string(),
            machine: machine.to_string(),
            worker_id,
        }
    }

    /// Return a human-readable location string `"rack/machine"`.
    #[must_use]
    pub fn location(&self) -> String {
        format!("{}/{}", self.rack, self.machine)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FarmTopology
// ─────────────────────────────────────────────────────────────────────────────

/// Registry of all worker nodes and their physical locations.
///
/// Duplicate `worker_id` values are silently ignored on insertion to prevent
/// inconsistent state.
#[derive(Debug, Clone, Default)]
pub struct FarmTopology {
    nodes: Vec<FarmNode>,
}

impl FarmTopology {
    /// Create an empty topology.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a node.
    ///
    /// If a node with the same `worker_id` already exists in the topology,
    /// this call is a no-op (the existing entry is preserved).
    pub fn add_node(&mut self, node: FarmNode) {
        // Deduplicate by worker_id
        if self.nodes.iter().any(|n| n.worker_id == node.worker_id) {
            return;
        }
        self.nodes.push(node);
    }

    /// Remove the node with `worker_id`, returning the removed node if found.
    pub fn remove_node(&mut self, worker_id: u64) -> Option<FarmNode> {
        if let Some(pos) = self.nodes.iter().position(|n| n.worker_id == worker_id) {
            Some(self.nodes.remove(pos))
        } else {
            None
        }
    }

    /// Return all nodes in the given rack (case-sensitive).
    #[must_use]
    pub fn nodes_in_rack(&self, rack: &str) -> Vec<&FarmNode> {
        self.nodes.iter().filter(|n| n.rack == rack).collect()
    }

    /// Return all nodes on the given machine hostname (case-sensitive).
    #[must_use]
    pub fn nodes_on_machine(&self, machine: &str) -> Vec<&FarmNode> {
        self.nodes.iter().filter(|n| n.machine == machine).collect()
    }

    /// Look up a node by `worker_id`.
    #[must_use]
    pub fn find_node(&self, worker_id: u64) -> Option<&FarmNode> {
        self.nodes.iter().find(|n| n.worker_id == worker_id)
    }

    /// Return the names of all unique racks present in the topology.
    #[must_use]
    pub fn rack_names(&self) -> Vec<&str> {
        let mut seen: Vec<&str> = Vec::new();
        for node in &self.nodes {
            if !seen.contains(&&*node.rack) {
                seen.push(&node.rack);
            }
        }
        seen
    }

    /// Total number of registered worker nodes.
    #[must_use]
    pub fn worker_count(&self) -> usize {
        self.nodes.len()
    }

    /// Returns `true` when no nodes have been registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Iterate over all nodes.
    pub fn iter(&self) -> impl Iterator<Item = &FarmNode> {
        self.nodes.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn basic_topology() -> FarmTopology {
        let mut t = FarmTopology::new();
        t.add_node(FarmNode::new("rack-01", "host-a", 1));
        t.add_node(FarmNode::new("rack-01", "host-b", 2));
        t.add_node(FarmNode::new("rack-02", "host-c", 3));
        t
    }

    #[test]
    fn test_new_is_empty() {
        let t = FarmTopology::new();
        assert!(t.is_empty());
        assert_eq!(t.worker_count(), 0);
    }

    #[test]
    fn test_add_node_and_count() {
        let t = basic_topology();
        assert_eq!(t.worker_count(), 3);
    }

    #[test]
    fn test_duplicate_worker_id_ignored() {
        let mut t = FarmTopology::new();
        t.add_node(FarmNode::new("rack-01", "host-a", 1));
        t.add_node(FarmNode::new("rack-02", "host-x", 1)); // duplicate id
        assert_eq!(t.worker_count(), 1);
    }

    #[test]
    fn test_nodes_in_rack() {
        let t = basic_topology();
        let rack1 = t.nodes_in_rack("rack-01");
        assert_eq!(rack1.len(), 2);
        assert!(rack1.iter().all(|n| n.rack == "rack-01"));
    }

    #[test]
    fn test_nodes_in_nonexistent_rack_empty() {
        let t = basic_topology();
        assert!(t.nodes_in_rack("rack-99").is_empty());
    }

    #[test]
    fn test_find_node_exists() {
        let t = basic_topology();
        let node = t.find_node(2);
        assert!(node.is_some());
        assert_eq!(node.expect("node should exist").machine, "host-b");
    }

    #[test]
    fn test_find_node_missing() {
        let t = basic_topology();
        assert!(t.find_node(999).is_none());
    }

    #[test]
    fn test_remove_node() {
        let mut t = basic_topology();
        let removed = t.remove_node(2);
        assert!(removed.is_some());
        assert_eq!(t.worker_count(), 2);
        assert!(t.find_node(2).is_none());
    }

    #[test]
    fn test_rack_names_unique() {
        let t = basic_topology();
        let racks = t.rack_names();
        assert_eq!(racks.len(), 2);
        assert!(racks.contains(&"rack-01"));
        assert!(racks.contains(&"rack-02"));
    }

    #[test]
    fn test_node_location_string() {
        let n = FarmNode::new("rack-01", "host-a", 1);
        assert_eq!(n.location(), "rack-01/host-a");
    }

    #[test]
    fn test_nodes_on_machine() {
        let t = basic_topology();
        let machines = t.nodes_on_machine("host-a");
        assert_eq!(machines.len(), 1);
    }
}
