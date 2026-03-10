//! Distributed cluster management.
//!
//! Provides types for managing a cluster of nodes in the distributed
//! encoding system, including roles, health status, and topology.

#![allow(dead_code)]

/// Role of a node in the cluster.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeRole {
    /// The current cluster leader.
    Leader,
    /// A regular member that follows the leader.
    Follower,
    /// A node seeking election to become leader.
    Candidate,
    /// A read-only observer that does not participate in elections.
    Observer,
}

impl NodeRole {
    /// Returns true if this node can participate in voting.
    #[must_use]
    pub fn can_vote(&self) -> bool {
        matches!(self, Self::Leader | Self::Follower | Self::Candidate)
    }

    /// Returns true if this node is the cluster leader.
    #[must_use]
    pub fn is_leader(&self) -> bool {
        matches!(self, Self::Leader)
    }

    /// Returns a human-readable name for the role.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::Leader => "Leader",
            Self::Follower => "Follower",
            Self::Candidate => "Candidate",
            Self::Observer => "Observer",
        }
    }
}

impl std::fmt::Display for NodeRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Health status of a cluster node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeHealth {
    /// Node is fully operational.
    Healthy,
    /// Node is operational but degraded (e.g. high load).
    Degraded,
    /// Node cannot be reached.
    Unreachable,
    /// Node is gracefully shutting down and not accepting new work.
    Draining,
}

impl NodeHealth {
    /// Returns true if the node is able to accept work.
    #[must_use]
    pub fn is_active(&self) -> bool {
        matches!(self, Self::Healthy | Self::Degraded)
    }

    /// Returns a human-readable description of the health status.
    #[must_use]
    pub fn description(&self) -> &str {
        match self {
            Self::Healthy => "Healthy",
            Self::Degraded => "Degraded",
            Self::Unreachable => "Unreachable",
            Self::Draining => "Draining",
        }
    }
}

impl std::fmt::Display for NodeHealth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.description())
    }
}

/// A node in the distributed cluster.
#[derive(Debug, Clone)]
pub struct ClusterNode {
    /// Unique node identifier.
    pub node_id: String,
    /// Network address of the node (e.g. "192.168.1.10:50051").
    pub address: String,
    /// Current role of the node.
    pub role: NodeRole,
    /// Current health status.
    pub health: NodeHealth,
    /// Unix epoch timestamp of the last received heartbeat.
    pub last_heartbeat_epoch: u64,
}

impl ClusterNode {
    /// Create a new cluster node.
    #[must_use]
    pub fn new(
        node_id: impl Into<String>,
        address: impl Into<String>,
        role: NodeRole,
        health: NodeHealth,
        last_heartbeat_epoch: u64,
    ) -> Self {
        Self {
            node_id: node_id.into(),
            address: address.into(),
            role,
            health,
            last_heartbeat_epoch,
        }
    }

    /// Returns true if the node's last heartbeat is older than `timeout_secs` seconds.
    #[must_use]
    pub fn is_stale(&self, now_epoch: u64, timeout_secs: u64) -> bool {
        now_epoch.saturating_sub(self.last_heartbeat_epoch) > timeout_secs
    }

    /// Returns true if this node can participate in voting.
    #[must_use]
    pub fn can_vote(&self) -> bool {
        self.role.can_vote()
    }

    /// Returns true if this node is healthy and active.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.health.is_active()
    }
}

/// The full topology of the distributed cluster.
#[derive(Debug, Default)]
pub struct ClusterTopology {
    /// All nodes known to be in the cluster.
    pub nodes: Vec<ClusterNode>,
}

impl ClusterTopology {
    /// Create a new empty cluster topology.
    #[must_use]
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    /// Add a node to the topology.
    ///
    /// If a node with the same `node_id` already exists it is replaced.
    pub fn add_node(&mut self, node: ClusterNode) {
        if let Some(existing) = self.nodes.iter_mut().find(|n| n.node_id == node.node_id) {
            *existing = node;
        } else {
            self.nodes.push(node);
        }
    }

    /// Find the current leader node, if any.
    #[must_use]
    pub fn find_leader(&self) -> Option<&ClusterNode> {
        self.nodes.iter().find(|n| n.role.is_leader())
    }

    /// Returns all nodes that are currently healthy or degraded (active).
    #[must_use]
    pub fn healthy_nodes(&self) -> Vec<&ClusterNode> {
        self.nodes.iter().filter(|n| n.health.is_active()).collect()
    }

    /// Returns the quorum size (majority of voting nodes).
    ///
    /// Quorum = ⌊N/2⌋ + 1 where N is the number of voting nodes.
    #[must_use]
    pub fn quorum_size(&self) -> usize {
        let voters = self.nodes.iter().filter(|n| n.can_vote()).count();
        voters / 2 + 1
    }

    /// Returns true if enough healthy voting nodes exist for a quorum.
    #[must_use]
    pub fn has_quorum(&self) -> bool {
        let healthy_voters = self
            .nodes
            .iter()
            .filter(|n| n.can_vote() && n.health.is_active())
            .count();
        healthy_voters >= self.quorum_size()
    }

    /// Returns the total number of nodes in the topology.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Remove a node from the topology by ID.
    ///
    /// Returns true if a node was removed.
    pub fn remove_node(&mut self, node_id: &str) -> bool {
        let before = self.nodes.len();
        self.nodes.retain(|n| n.node_id != node_id);
        self.nodes.len() < before
    }

    /// Find a node by its ID.
    #[must_use]
    pub fn find_by_id(&self, node_id: &str) -> Option<&ClusterNode> {
        self.nodes.iter().find(|n| n.node_id == node_id)
    }

    /// Returns all stale nodes whose last heartbeat exceeds `timeout_secs`.
    #[must_use]
    pub fn stale_nodes(&self, now_epoch: u64, timeout_secs: u64) -> Vec<&ClusterNode> {
        self.nodes
            .iter()
            .filter(|n| n.is_stale(now_epoch, timeout_secs))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_node(id: &str, role: NodeRole, health: NodeHealth, ts: u64) -> ClusterNode {
        ClusterNode::new(id, format!("10.0.0.1:{}", id), role, health, ts)
    }

    #[test]
    fn test_node_role_can_vote() {
        assert!(NodeRole::Leader.can_vote());
        assert!(NodeRole::Follower.can_vote());
        assert!(NodeRole::Candidate.can_vote());
        assert!(!NodeRole::Observer.can_vote());
    }

    #[test]
    fn test_node_role_is_leader() {
        assert!(NodeRole::Leader.is_leader());
        assert!(!NodeRole::Follower.is_leader());
        assert!(!NodeRole::Candidate.is_leader());
        assert!(!NodeRole::Observer.is_leader());
    }

    #[test]
    fn test_node_role_display() {
        assert_eq!(NodeRole::Leader.to_string(), "Leader");
        assert_eq!(NodeRole::Follower.to_string(), "Follower");
        assert_eq!(NodeRole::Observer.to_string(), "Observer");
    }

    #[test]
    fn test_node_health_is_active() {
        assert!(NodeHealth::Healthy.is_active());
        assert!(NodeHealth::Degraded.is_active());
        assert!(!NodeHealth::Unreachable.is_active());
        assert!(!NodeHealth::Draining.is_active());
    }

    #[test]
    fn test_node_health_display() {
        assert_eq!(NodeHealth::Healthy.to_string(), "Healthy");
        assert_eq!(NodeHealth::Unreachable.to_string(), "Unreachable");
    }

    #[test]
    fn test_cluster_node_is_stale() {
        let node = make_node("n1", NodeRole::Follower, NodeHealth::Healthy, 1000);
        // timeout = 30 seconds
        assert!(!node.is_stale(1020, 30)); // only 20s elapsed
        assert!(node.is_stale(1031, 30)); // 31s elapsed
    }

    #[test]
    fn test_cluster_topology_add_node() {
        let mut topo = ClusterTopology::new();
        topo.add_node(make_node("n1", NodeRole::Leader, NodeHealth::Healthy, 100));
        topo.add_node(make_node(
            "n2",
            NodeRole::Follower,
            NodeHealth::Healthy,
            100,
        ));
        assert_eq!(topo.node_count(), 2);
    }

    #[test]
    fn test_cluster_topology_add_node_replaces_existing() {
        let mut topo = ClusterTopology::new();
        topo.add_node(make_node(
            "n1",
            NodeRole::Follower,
            NodeHealth::Healthy,
            100,
        ));
        topo.add_node(make_node("n1", NodeRole::Leader, NodeHealth::Degraded, 200));
        assert_eq!(topo.node_count(), 1);
        assert_eq!(topo.nodes[0].role, NodeRole::Leader);
    }

    #[test]
    fn test_cluster_topology_find_leader() {
        let mut topo = ClusterTopology::new();
        topo.add_node(make_node("n1", NodeRole::Leader, NodeHealth::Healthy, 100));
        topo.add_node(make_node(
            "n2",
            NodeRole::Follower,
            NodeHealth::Healthy,
            100,
        ));

        let leader = topo.find_leader();
        assert!(leader.is_some());
        assert_eq!(leader.expect("leader should exist").node_id, "n1");
    }

    #[test]
    fn test_cluster_topology_no_leader() {
        let mut topo = ClusterTopology::new();
        topo.add_node(make_node(
            "n1",
            NodeRole::Follower,
            NodeHealth::Healthy,
            100,
        ));
        assert!(topo.find_leader().is_none());
    }

    #[test]
    fn test_cluster_topology_healthy_nodes() {
        let mut topo = ClusterTopology::new();
        topo.add_node(make_node("n1", NodeRole::Leader, NodeHealth::Healthy, 100));
        topo.add_node(make_node(
            "n2",
            NodeRole::Follower,
            NodeHealth::Degraded,
            100,
        ));
        topo.add_node(make_node(
            "n3",
            NodeRole::Follower,
            NodeHealth::Unreachable,
            100,
        ));

        let healthy = topo.healthy_nodes();
        assert_eq!(healthy.len(), 2);
    }

    #[test]
    fn test_cluster_topology_quorum_size() {
        let mut topo = ClusterTopology::new();
        // 3 voters → quorum = 2
        topo.add_node(make_node("n1", NodeRole::Leader, NodeHealth::Healthy, 100));
        topo.add_node(make_node(
            "n2",
            NodeRole::Follower,
            NodeHealth::Healthy,
            100,
        ));
        topo.add_node(make_node(
            "n3",
            NodeRole::Follower,
            NodeHealth::Healthy,
            100,
        ));
        // 1 observer (non-voter)
        topo.add_node(make_node(
            "n4",
            NodeRole::Observer,
            NodeHealth::Healthy,
            100,
        ));

        assert_eq!(topo.quorum_size(), 2);
    }

    #[test]
    fn test_cluster_topology_has_quorum() {
        let mut topo = ClusterTopology::new();
        topo.add_node(make_node("n1", NodeRole::Leader, NodeHealth::Healthy, 100));
        topo.add_node(make_node(
            "n2",
            NodeRole::Follower,
            NodeHealth::Healthy,
            100,
        ));
        topo.add_node(make_node(
            "n3",
            NodeRole::Follower,
            NodeHealth::Unreachable,
            100,
        ));

        // 3 voters, quorum = 2; only 2 healthy voters → has quorum
        assert!(topo.has_quorum());
    }

    #[test]
    fn test_cluster_topology_no_quorum() {
        let mut topo = ClusterTopology::new();
        topo.add_node(make_node(
            "n1",
            NodeRole::Leader,
            NodeHealth::Unreachable,
            100,
        ));
        topo.add_node(make_node(
            "n2",
            NodeRole::Follower,
            NodeHealth::Unreachable,
            100,
        ));
        topo.add_node(make_node(
            "n3",
            NodeRole::Follower,
            NodeHealth::Healthy,
            100,
        ));

        // 3 voters, quorum = 2; only 1 healthy → no quorum
        assert!(!topo.has_quorum());
    }

    #[test]
    fn test_cluster_topology_remove_node() {
        let mut topo = ClusterTopology::new();
        topo.add_node(make_node("n1", NodeRole::Leader, NodeHealth::Healthy, 100));
        topo.add_node(make_node(
            "n2",
            NodeRole::Follower,
            NodeHealth::Healthy,
            100,
        ));

        assert!(topo.remove_node("n1"));
        assert_eq!(topo.node_count(), 1);
        assert!(!topo.remove_node("MISSING"));
    }

    #[test]
    fn test_cluster_topology_stale_nodes() {
        let mut topo = ClusterTopology::new();
        topo.add_node(make_node("n1", NodeRole::Leader, NodeHealth::Healthy, 1000));
        topo.add_node(make_node(
            "n2",
            NodeRole::Follower,
            NodeHealth::Healthy,
            900,
        ));

        // At time=1025 with 30s timeout: n2 (ts=900) is stale (125s elapsed), n1 (ts=1000, 25s) is not
        let stale = topo.stale_nodes(1025, 30);
        assert_eq!(stale.len(), 1);
        assert_eq!(stale[0].node_id, "n2");
    }
}
