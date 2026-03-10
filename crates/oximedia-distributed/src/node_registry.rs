//! Node registry for cluster membership management.
//!
//! Tracks all known nodes in the distributed cluster, their capabilities,
//! roles, and current status.  Provides efficient lookup, filtering by role,
//! and node lifecycle management.

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// NodeRole
// ---------------------------------------------------------------------------

/// The role a node plays in the cluster.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NodeRole {
    /// Central coordinator node.
    Coordinator,
    /// General-purpose compute/encoding worker.
    Worker,
    /// Storage node (not involved in compute).
    Storage,
    /// Gateway / load-balancer node.
    Gateway,
}

impl NodeRole {
    /// Returns the human-readable name of this role.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Coordinator => "coordinator",
            Self::Worker => "worker",
            Self::Storage => "storage",
            Self::Gateway => "gateway",
        }
    }
}

// ---------------------------------------------------------------------------
// NodeStatus
// ---------------------------------------------------------------------------

/// Operational status of a registered node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeStatus {
    /// Node is healthy and available.
    Healthy,
    /// Node is available but operating in a degraded state.
    Degraded,
    /// Node is temporarily suspended (maintenance mode).
    Suspended,
    /// Node has been removed from the active cluster.
    Removed,
}

impl NodeStatus {
    /// Returns `true` if the node can accept work.
    #[must_use]
    pub fn is_available(self) -> bool {
        matches!(self, Self::Healthy | Self::Degraded)
    }
}

// ---------------------------------------------------------------------------
// NodeInfo
// ---------------------------------------------------------------------------

/// Information about a registered cluster node.
#[derive(Debug, Clone)]
pub struct NodeInfo {
    /// Unique node identifier.
    pub id: String,
    /// Network address (host:port).
    pub address: String,
    /// Assigned role in the cluster.
    pub role: NodeRole,
    /// Current operational status.
    pub status: NodeStatus,
    /// Number of CPU cores.
    pub cpu_cores: u32,
    /// Total RAM in megabytes.
    pub memory_mb: u32,
    /// Whether this node has GPU acceleration.
    pub has_gpu: bool,
    /// Unix epoch ms when the node was registered.
    pub registered_at_ms: u64,
    /// Unix epoch ms of the last status update.
    pub last_seen_ms: u64,
    /// Arbitrary tags (e.g., "av1", "region:eu-west-1").
    pub tags: Vec<String>,
}

impl NodeInfo {
    /// Create a new healthy node registration.
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        address: impl Into<String>,
        role: NodeRole,
        cpu_cores: u32,
        memory_mb: u32,
        has_gpu: bool,
        now_ms: u64,
    ) -> Self {
        Self {
            id: id.into(),
            address: address.into(),
            role,
            status: NodeStatus::Healthy,
            cpu_cores,
            memory_mb,
            has_gpu,
            registered_at_ms: now_ms,
            last_seen_ms: now_ms,
            tags: Vec::new(),
        }
    }

    /// Add a tag to this node.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Update the last-seen timestamp and optionally the status.
    pub fn touch(&mut self, now_ms: u64, status: NodeStatus) {
        self.last_seen_ms = now_ms;
        self.status = status;
    }

    /// Returns `true` if the node can accept work.
    #[must_use]
    pub fn is_available(&self) -> bool {
        self.status.is_available()
    }

    /// Returns `true` if the node has a specific tag.
    #[must_use]
    pub fn has_tag(&self, tag: &str) -> bool {
        self.tags.iter().any(|t| t == tag)
    }

    /// Age of the registration in milliseconds.
    #[must_use]
    pub fn age_ms(&self, now_ms: u64) -> u64 {
        now_ms.saturating_sub(self.registered_at_ms)
    }
}

// ---------------------------------------------------------------------------
// NodeRegistry
// ---------------------------------------------------------------------------

/// Central registry of all cluster nodes.
///
/// Supports fast lookup by ID and filtered queries by role/status/tag.
#[derive(Debug, Default)]
pub struct NodeRegistry {
    nodes: Vec<NodeInfo>,
}

impl NodeRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new node.  If a node with the same ID already exists, it is
    /// replaced.
    pub fn register(&mut self, node: NodeInfo) {
        if let Some(existing) = self.nodes.iter_mut().find(|n| n.id == node.id) {
            *existing = node;
        } else {
            self.nodes.push(node);
        }
    }

    /// Remove a node by ID, returning it if found.
    pub fn deregister(&mut self, id: &str) -> Option<NodeInfo> {
        if let Some(pos) = self.nodes.iter().position(|n| n.id == id) {
            Some(self.nodes.remove(pos))
        } else {
            None
        }
    }

    /// Look up a node by ID.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<&NodeInfo> {
        self.nodes.iter().find(|n| n.id == id)
    }

    /// Mutable access to a node by ID.
    #[must_use]
    pub fn get_mut(&mut self, id: &str) -> Option<&mut NodeInfo> {
        self.nodes.iter_mut().find(|n| n.id == id)
    }

    /// Total number of registered nodes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns `true` if the registry has no nodes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// All nodes with a given role.
    #[must_use]
    pub fn by_role(&self, role: NodeRole) -> Vec<&NodeInfo> {
        self.nodes.iter().filter(|n| n.role == role).collect()
    }

    /// All nodes with a given status.
    #[must_use]
    pub fn by_status(&self, status: NodeStatus) -> Vec<&NodeInfo> {
        self.nodes.iter().filter(|n| n.status == status).collect()
    }

    /// All available (Healthy or Degraded) worker nodes.
    #[must_use]
    pub fn available_workers(&self) -> Vec<&NodeInfo> {
        self.nodes
            .iter()
            .filter(|n| n.role == NodeRole::Worker && n.is_available())
            .collect()
    }

    /// All nodes that have a specific tag.
    #[must_use]
    pub fn by_tag(&self, tag: &str) -> Vec<&NodeInfo> {
        self.nodes.iter().filter(|n| n.has_tag(tag)).collect()
    }

    /// Mark a node as removed (soft delete).
    pub fn remove_node(&mut self, id: &str, now_ms: u64) {
        if let Some(n) = self.get_mut(id) {
            n.touch(now_ms, NodeStatus::Removed);
        }
    }

    /// Evict nodes that have not been seen within `ttl_ms` of `now_ms`.
    ///
    /// Returns the IDs of evicted nodes.
    pub fn evict_stale(&mut self, now_ms: u64, ttl_ms: u64) -> Vec<String> {
        let cutoff = now_ms.saturating_sub(ttl_ms);
        let mut evicted = Vec::new();
        self.nodes.retain(|n| {
            if n.last_seen_ms < cutoff {
                evicted.push(n.id.clone());
                false
            } else {
                true
            }
        });
        evicted
    }

    /// Summary counts: `(total, healthy, degraded, suspended, removed)`.
    #[must_use]
    pub fn status_summary(&self) -> (usize, usize, usize, usize, usize) {
        let total = self.nodes.len();
        let healthy = self.by_status(NodeStatus::Healthy).len();
        let degraded = self.by_status(NodeStatus::Degraded).len();
        let suspended = self.by_status(NodeStatus::Suspended).len();
        let removed = self.by_status(NodeStatus::Removed).len();
        (total, healthy, degraded, suspended, removed)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn worker(id: &str, now_ms: u64) -> NodeInfo {
        NodeInfo::new(
            id,
            format!("10.0.0.1:5000"),
            NodeRole::Worker,
            8,
            16_384,
            false,
            now_ms,
        )
    }

    fn gpu_worker(id: &str, now_ms: u64) -> NodeInfo {
        NodeInfo::new(
            id,
            format!("10.0.0.2:5000"),
            NodeRole::Worker,
            16,
            32_768,
            true,
            now_ms,
        )
        .with_tag("gpu")
    }

    fn coordinator(id: &str, now_ms: u64) -> NodeInfo {
        NodeInfo::new(
            id,
            format!("10.0.0.3:5000"),
            NodeRole::Coordinator,
            4,
            8_192,
            false,
            now_ms,
        )
    }

    // ── NodeRole ─────────────────────────────────────────────────────────

    #[test]
    fn test_node_role_as_str() {
        assert_eq!(NodeRole::Worker.as_str(), "worker");
        assert_eq!(NodeRole::Coordinator.as_str(), "coordinator");
        assert_eq!(NodeRole::Storage.as_str(), "storage");
        assert_eq!(NodeRole::Gateway.as_str(), "gateway");
    }

    // ── NodeStatus ───────────────────────────────────────────────────────

    #[test]
    fn test_healthy_is_available() {
        assert!(NodeStatus::Healthy.is_available());
    }

    #[test]
    fn test_degraded_is_available() {
        assert!(NodeStatus::Degraded.is_available());
    }

    #[test]
    fn test_suspended_not_available() {
        assert!(!NodeStatus::Suspended.is_available());
    }

    #[test]
    fn test_removed_not_available() {
        assert!(!NodeStatus::Removed.is_available());
    }

    // ── NodeInfo ─────────────────────────────────────────────────────────

    #[test]
    fn test_node_info_initial_status_healthy() {
        let n = worker("n0", 1000);
        assert_eq!(n.status, NodeStatus::Healthy);
    }

    #[test]
    fn test_node_info_has_tag() {
        let n = gpu_worker("n0", 1000);
        assert!(n.has_tag("gpu"));
        assert!(!n.has_tag("av1"));
    }

    #[test]
    fn test_node_info_age_ms() {
        let n = worker("n0", 1000);
        assert_eq!(n.age_ms(3000), 2000);
    }

    #[test]
    fn test_node_info_touch_updates_status() {
        let mut n = worker("n0", 1000);
        n.touch(2000, NodeStatus::Degraded);
        assert_eq!(n.status, NodeStatus::Degraded);
        assert_eq!(n.last_seen_ms, 2000);
    }

    // ── NodeRegistry ─────────────────────────────────────────────────────

    #[test]
    fn test_registry_empty() {
        let reg = NodeRegistry::new();
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);
    }

    #[test]
    fn test_registry_register_and_get() {
        let mut reg = NodeRegistry::new();
        reg.register(worker("n0", 1000));
        let n = reg.get("n0").expect("get should return a value");
        assert_eq!(n.id, "n0");
    }

    #[test]
    fn test_registry_register_replaces_existing() {
        let mut reg = NodeRegistry::new();
        reg.register(worker("n0", 1000));
        reg.register(gpu_worker("n0", 2000)); // same ID → replace
        assert_eq!(reg.len(), 1);
        assert!(reg.get("n0").expect("get should return a value").has_gpu);
    }

    #[test]
    fn test_registry_deregister() {
        let mut reg = NodeRegistry::new();
        reg.register(worker("n0", 1000));
        let removed = reg.deregister("n0");
        assert!(removed.is_some());
        assert!(reg.is_empty());
    }

    #[test]
    fn test_registry_by_role() {
        let mut reg = NodeRegistry::new();
        reg.register(worker("w0", 1000));
        reg.register(worker("w1", 1000));
        reg.register(coordinator("c0", 1000));
        assert_eq!(reg.by_role(NodeRole::Worker).len(), 2);
        assert_eq!(reg.by_role(NodeRole::Coordinator).len(), 1);
    }

    #[test]
    fn test_registry_available_workers() {
        let mut reg = NodeRegistry::new();
        let mut w1 = worker("w0", 1000);
        w1.status = NodeStatus::Suspended;
        reg.register(w1);
        reg.register(worker("w1", 1000));
        assert_eq!(reg.available_workers().len(), 1);
    }

    #[test]
    fn test_registry_by_tag() {
        let mut reg = NodeRegistry::new();
        reg.register(gpu_worker("g0", 1000));
        reg.register(worker("w0", 1000));
        assert_eq!(reg.by_tag("gpu").len(), 1);
    }

    #[test]
    fn test_registry_remove_node_soft_delete() {
        let mut reg = NodeRegistry::new();
        reg.register(worker("w0", 1000));
        reg.remove_node("w0", 2000);
        assert_eq!(
            reg.get("w0").expect("get should return a value").status,
            NodeStatus::Removed
        );
        assert!(!reg
            .get("w0")
            .expect("get should return a value")
            .is_available());
    }

    #[test]
    fn test_registry_evict_stale() {
        let mut reg = NodeRegistry::new();
        reg.register(worker("old", 100));
        reg.register(worker("fresh", 5000));
        let evicted = reg.evict_stale(6000, 1000); // cutoff = 5000 → old removed
        assert!(evicted.contains(&"old".to_string()));
        assert_eq!(reg.len(), 1);
    }

    #[test]
    fn test_registry_status_summary() {
        let mut reg = NodeRegistry::new();
        reg.register(worker("w0", 1000));
        reg.register(worker("w1", 1000));
        let (total, healthy, _, _, _) = reg.status_summary();
        assert_eq!(total, 2);
        assert_eq!(healthy, 2);
    }
}
