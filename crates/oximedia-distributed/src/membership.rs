//! Cluster membership management.
//!
//! Tracks the set of active nodes in a distributed cluster, supporting dynamic
//! add and remove operations.  In a production Raft deployment membership
//! changes would be propagated as special log entries; this module provides the
//! authoritative in-memory view of the current configuration.

/// Dynamic cluster membership.
///
/// Maintains a deduplicated, ordered list of node IDs.  Nodes are identified
/// by `u64` IDs (typically sequential integers or hash-based identifiers).
#[derive(Debug, Clone)]
pub struct ClusterMembership {
    /// Current set of member node IDs (kept sorted for determinism).
    nodes: Vec<u64>,
}

impl ClusterMembership {
    /// Create a new `ClusterMembership` with the given initial set of nodes.
    ///
    /// Duplicate IDs in `nodes` are silently deduplicated.
    #[must_use]
    pub fn new(nodes: Vec<u64>) -> Self {
        let mut deduped = nodes;
        deduped.sort_unstable();
        deduped.dedup();
        Self { nodes: deduped }
    }

    /// Add a node to the cluster.
    ///
    /// If `id` is already present, this is a no-op.
    pub fn add_node(&mut self, id: u64) {
        if !self.nodes.contains(&id) {
            self.nodes.push(id);
            self.nodes.sort_unstable();
        }
    }

    /// Remove a node from the cluster.
    ///
    /// If `id` is not present, this is a no-op.
    pub fn remove_node(&mut self, id: u64) {
        self.nodes.retain(|&n| n != id);
    }

    /// Return the current member list in ascending order.
    #[must_use]
    pub fn members(&self) -> &[u64] {
        &self.nodes
    }

    /// Return the number of members in the cluster.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Return `true` if there are no members.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Return `true` if `id` is currently a member.
    #[must_use]
    pub fn contains(&self, id: u64) -> bool {
        self.nodes.binary_search(&id).is_ok()
    }

    /// Quorum size: the minimum number of nodes required for a majority
    /// decision (i.e., `floor(n / 2) + 1`).
    #[must_use]
    pub fn quorum_size(&self) -> usize {
        self.nodes.len() / 2 + 1
    }

    /// Replace the entire membership set.
    ///
    /// Equivalent to constructing a new `ClusterMembership` with the given
    /// nodes.
    pub fn set_members(&mut self, nodes: Vec<u64>) {
        self.nodes = nodes;
        self.nodes.sort_unstable();
        self.nodes.dedup();
    }
}

impl Default for ClusterMembership {
    fn default() -> Self {
        Self::new(Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_deduplicates_and_sorts() {
        let m = ClusterMembership::new(vec![3, 1, 2, 1, 3]);
        assert_eq!(m.members(), &[1, 2, 3]);
    }

    #[test]
    fn test_add_node() {
        let mut m = ClusterMembership::new(vec![1, 2]);
        m.add_node(3);
        assert_eq!(m.members(), &[1, 2, 3]);
    }

    #[test]
    fn test_add_duplicate_node_is_noop() {
        let mut m = ClusterMembership::new(vec![1, 2]);
        m.add_node(2);
        assert_eq!(m.members(), &[1, 2]);
    }

    #[test]
    fn test_remove_node() {
        let mut m = ClusterMembership::new(vec![1, 2, 3]);
        m.remove_node(2);
        assert_eq!(m.members(), &[1, 3]);
    }

    #[test]
    fn test_remove_absent_node_is_noop() {
        let mut m = ClusterMembership::new(vec![1, 2]);
        m.remove_node(99);
        assert_eq!(m.members(), &[1, 2]);
    }

    #[test]
    fn test_contains() {
        let m = ClusterMembership::new(vec![10, 20, 30]);
        assert!(m.contains(20));
        assert!(!m.contains(25));
    }

    #[test]
    fn test_len_and_is_empty() {
        let mut m = ClusterMembership::new(vec![]);
        assert!(m.is_empty());
        assert_eq!(m.len(), 0);
        m.add_node(5);
        assert!(!m.is_empty());
        assert_eq!(m.len(), 1);
    }

    #[test]
    fn test_quorum_size() {
        // 1 node → quorum 1
        assert_eq!(ClusterMembership::new(vec![1]).quorum_size(), 1);
        // 2 nodes → quorum 2
        assert_eq!(ClusterMembership::new(vec![1, 2]).quorum_size(), 2);
        // 3 nodes → quorum 2
        assert_eq!(ClusterMembership::new(vec![1, 2, 3]).quorum_size(), 2);
        // 5 nodes → quorum 3
        assert_eq!(ClusterMembership::new(vec![1, 2, 3, 4, 5]).quorum_size(), 3);
    }

    #[test]
    fn test_set_members_replaces_all() {
        let mut m = ClusterMembership::new(vec![1, 2, 3]);
        m.set_members(vec![10, 20]);
        assert_eq!(m.members(), &[10, 20]);
    }

    #[test]
    fn test_membership_maintained_sorted_after_add() {
        let mut m = ClusterMembership::new(vec![5, 10]);
        m.add_node(7);
        assert_eq!(m.members(), &[5, 7, 10]);
    }

    #[test]
    fn test_add_and_remove_sequence() {
        let mut m = ClusterMembership::default();
        for id in [3, 1, 4, 1, 5, 9, 2, 6] {
            m.add_node(id);
        }
        // After dedup: 1,2,3,4,5,6,9
        assert_eq!(m.members(), &[1, 2, 3, 4, 5, 6, 9]);

        m.remove_node(4);
        m.remove_node(9);
        assert_eq!(m.members(), &[1, 2, 3, 5, 6]);
    }
}
