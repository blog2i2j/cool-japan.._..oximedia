//! Cluster discovery for auto-detecting batch workers on the network.
//!
//! Provides a registry of known worker nodes, health-check probing,
//! capability-based filtering (available CPU cores, GPU support, etc.),
//! and a simple peer-announcement protocol via UDP multicast.
//!
//! # Architecture
//!
//! A [`WorkerNode`] represents a single batch processing worker.  The
//! [`ClusterRegistry`] holds all known nodes and exposes methods for:
//!
//! - Manual registration and deregistration.
//! - Health-check evaluation (last-seen timestamp + timeout).
//! - Capability-based node selection.
//! - Serialisation to JSON for persistence or peer exchange.
//!
//! The optional [`DiscoveryAnnouncement`] type encodes the UDP datagram
//! payload used for zero-config peer discovery via LAN multicast.

#![allow(dead_code)]

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;

// ---------------------------------------------------------------------------
// WorkerCapabilities
// ---------------------------------------------------------------------------

/// Hardware and software capabilities advertised by a worker node.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorkerCapabilities {
    /// Number of logical CPU cores available for batch work.
    pub cpu_cores: u32,
    /// Available memory in megabytes.
    pub memory_mb: u64,
    /// Whether the node has GPU acceleration.
    pub has_gpu: bool,
    /// Number of GPU devices (0 if none).
    pub gpu_count: u32,
    /// Total VRAM across all GPUs in megabytes.
    pub vram_mb: u64,
    /// Whether the node supports NVENC/hardware video encoding.
    pub hardware_encode: bool,
    /// Whether the node supports hardware video decoding.
    pub hardware_decode: bool,
    /// Supported job types (e.g. `["transcode", "thumbnail", "analysis"]`).
    pub supported_job_types: Vec<String>,
    /// Maximum number of concurrent jobs this node can handle.
    pub max_concurrent_jobs: u32,
    /// Platform identifier (e.g. "linux/amd64", "darwin/arm64").
    pub platform: String,
}

impl Default for WorkerCapabilities {
    fn default() -> Self {
        Self {
            cpu_cores: 4,
            memory_mb: 8192,
            has_gpu: false,
            gpu_count: 0,
            vram_mb: 0,
            hardware_encode: false,
            hardware_decode: false,
            supported_job_types: vec!["transcode".to_string()],
            max_concurrent_jobs: 4,
            platform: "unknown".to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// WorkerStatus
// ---------------------------------------------------------------------------

/// Lifecycle state of a worker node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkerStatus {
    /// Node has recently responded to a health check.
    Healthy,
    /// Node has not responded within the health-check timeout.
    Unreachable,
    /// Node is draining: accepting no new work, finishing existing jobs.
    Draining,
    /// Node is in maintenance mode and should not receive jobs.
    Maintenance,
    /// Node has been deregistered from the cluster.
    Deregistered,
}

impl std::fmt::Display for WorkerStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Healthy => write!(f, "healthy"),
            Self::Unreachable => write!(f, "unreachable"),
            Self::Draining => write!(f, "draining"),
            Self::Maintenance => write!(f, "maintenance"),
            Self::Deregistered => write!(f, "deregistered"),
        }
    }
}

// ---------------------------------------------------------------------------
// WorkerNode
// ---------------------------------------------------------------------------

/// A single worker node in the cluster.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerNode {
    /// Unique node identifier (typically a UUID string).
    pub node_id: String,
    /// Human-readable display name.
    pub display_name: String,
    /// TCP/UDP address at which the node's batch API is reachable.
    pub address: SocketAddr,
    /// Hardware and software capabilities.
    pub capabilities: WorkerCapabilities,
    /// Current lifecycle status.
    pub status: WorkerStatus,
    /// Unix timestamp (seconds) of the last successful health response.
    pub last_seen_secs: u64,
    /// Number of jobs currently running on this node.
    pub active_jobs: u32,
    /// Software version string of the batch agent on this node.
    pub agent_version: String,
    /// Arbitrary metadata (region, rack, owner, etc.).
    pub metadata: HashMap<String, String>,
}

impl WorkerNode {
    /// Create a new healthy worker node.
    #[must_use]
    pub fn new(
        node_id: impl Into<String>,
        display_name: impl Into<String>,
        address: SocketAddr,
        capabilities: WorkerCapabilities,
    ) -> Self {
        Self {
            node_id: node_id.into(),
            display_name: display_name.into(),
            address,
            capabilities,
            status: WorkerStatus::Healthy,
            last_seen_secs: current_timestamp(),
            active_jobs: 0,
            agent_version: "0.0.0".to_string(),
            metadata: HashMap::new(),
        }
    }

    /// Update `last_seen_secs` to now and set status to `Healthy`.
    pub fn mark_seen(&mut self) {
        self.last_seen_secs = current_timestamp();
        self.status = WorkerStatus::Healthy;
    }

    /// How many seconds ago this node was last seen (0 if the clock went backwards).
    #[must_use]
    pub fn age_secs(&self) -> u64 {
        current_timestamp().saturating_sub(self.last_seen_secs)
    }

    /// Returns `true` if the node can accept at least one more job.
    #[must_use]
    pub fn has_capacity(&self) -> bool {
        self.status == WorkerStatus::Healthy
            && self.active_jobs < self.capabilities.max_concurrent_jobs
    }

    /// Returns `true` if the node supports the given job type.
    #[must_use]
    pub fn supports_job_type(&self, job_type: &str) -> bool {
        self.capabilities
            .supported_job_types
            .iter()
            .any(|t| t == job_type)
    }

    /// Returns the number of free job slots.
    #[must_use]
    pub fn free_slots(&self) -> u32 {
        self.capabilities
            .max_concurrent_jobs
            .saturating_sub(self.active_jobs)
    }

    /// Add metadata and return self (builder style).
    #[must_use]
    pub fn with_meta(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

// ---------------------------------------------------------------------------
// DiscoveryAnnouncement
// ---------------------------------------------------------------------------

/// UDP multicast datagram payload for zero-config cluster peer discovery.
///
/// When a worker starts it broadcasts an [`DiscoveryAnnouncement`] to the
/// cluster multicast group.  Other nodes and the coordinator deserialise
/// the datagram and add the sender to their registries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryAnnouncement {
    /// Unique node identifier.
    pub node_id: String,
    /// Human-readable name.
    pub display_name: String,
    /// The node's batch API address.
    pub address: SocketAddr,
    /// Node capabilities.
    pub capabilities: WorkerCapabilities,
    /// Agent software version.
    pub agent_version: String,
    /// Announcement type: "join" or "leave".
    pub announcement_type: AnnouncementType,
    /// Unix timestamp (seconds) when the announcement was generated.
    pub timestamp_secs: u64,
}

/// Whether the node is joining or leaving the cluster.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnnouncementType {
    /// Node is joining (or re-joining) the cluster.
    Join,
    /// Node is leaving the cluster gracefully.
    Leave,
}

impl DiscoveryAnnouncement {
    /// Create a join announcement.
    #[must_use]
    pub fn join(node: &WorkerNode) -> Self {
        Self {
            node_id: node.node_id.clone(),
            display_name: node.display_name.clone(),
            address: node.address,
            capabilities: node.capabilities.clone(),
            agent_version: node.agent_version.clone(),
            announcement_type: AnnouncementType::Join,
            timestamp_secs: current_timestamp(),
        }
    }

    /// Create a leave announcement.
    #[must_use]
    pub fn leave(node: &WorkerNode) -> Self {
        Self {
            node_id: node.node_id.clone(),
            display_name: node.display_name.clone(),
            address: node.address,
            capabilities: node.capabilities.clone(),
            agent_version: node.agent_version.clone(),
            announcement_type: AnnouncementType::Leave,
            timestamp_secs: current_timestamp(),
        }
    }

    /// Serialise to JSON bytes for transmission.
    ///
    /// # Errors
    ///
    /// Returns an error if serialisation fails.
    pub fn to_bytes(&self) -> crate::error::Result<Vec<u8>> {
        serde_json::to_vec(self).map_err(crate::error::BatchError::SerializationError)
    }

    /// Deserialise from JSON bytes received over the network.
    ///
    /// # Errors
    ///
    /// Returns an error if deserialisation fails.
    pub fn from_bytes(bytes: &[u8]) -> crate::error::Result<Self> {
        serde_json::from_slice(bytes).map_err(crate::error::BatchError::SerializationError)
    }
}

// ---------------------------------------------------------------------------
// ClusterRegistry
// ---------------------------------------------------------------------------

/// Configuration for the cluster registry health-check behaviour.
#[derive(Debug, Clone)]
pub struct RegistryConfig {
    /// A node is considered `Unreachable` if its `last_seen_secs` is more
    /// than `health_timeout_secs` seconds in the past.
    pub health_timeout_secs: u64,
    /// Maximum number of nodes the registry will hold before it stops
    /// accepting new registrations.  0 means unlimited.
    pub max_nodes: usize,
}

impl Default for RegistryConfig {
    fn default() -> Self {
        Self {
            health_timeout_secs: 30,
            max_nodes: 0,
        }
    }
}

/// Thread-safe registry of all known worker nodes in the cluster.
#[derive(Debug)]
pub struct ClusterRegistry {
    nodes: RwLock<HashMap<String, WorkerNode>>,
    config: RegistryConfig,
}

impl ClusterRegistry {
    /// Create a new registry with the given configuration.
    #[must_use]
    pub fn new(config: RegistryConfig) -> Self {
        Self {
            nodes: RwLock::new(HashMap::new()),
            config,
        }
    }

    /// Create a registry with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(RegistryConfig::default())
    }

    // -----------------------------------------------------------------------
    // Mutation
    // -----------------------------------------------------------------------

    /// Register a new worker node, or update an existing one.
    ///
    /// Returns `true` if this is a new node, `false` if it was an update.
    pub fn register(&self, node: WorkerNode) -> bool {
        let is_new = !self.nodes.read().contains_key(&node.node_id);
        if self.config.max_nodes > 0 {
            let count = self.nodes.read().len();
            if is_new && count >= self.config.max_nodes {
                return false;
            }
        }
        self.nodes.write().insert(node.node_id.clone(), node);
        is_new
    }

    /// Register a node from a discovery announcement.
    ///
    /// Returns `true` if this is a new node.
    pub fn handle_announcement(&self, announcement: DiscoveryAnnouncement) -> bool {
        match announcement.announcement_type {
            AnnouncementType::Join => {
                let node = WorkerNode::new(
                    announcement.node_id,
                    announcement.display_name,
                    announcement.address,
                    announcement.capabilities,
                );
                self.register(node)
            }
            AnnouncementType::Leave => {
                self.deregister(&announcement.node_id);
                false
            }
        }
    }

    /// Deregister a worker node by its ID.
    ///
    /// Returns `true` if the node was found and removed.
    pub fn deregister(&self, node_id: &str) -> bool {
        self.nodes.write().remove(node_id).is_some()
    }

    /// Mark a node as having been seen now (health-check passed).
    ///
    /// Returns `true` if the node exists.
    pub fn mark_seen(&self, node_id: &str) -> bool {
        if let Some(node) = self.nodes.write().get_mut(node_id) {
            node.mark_seen();
            true
        } else {
            false
        }
    }

    /// Update the active job count for a node.
    ///
    /// Returns `true` if the node exists.
    pub fn update_active_jobs(&self, node_id: &str, active_jobs: u32) -> bool {
        if let Some(node) = self.nodes.write().get_mut(node_id) {
            node.active_jobs = active_jobs;
            true
        } else {
            false
        }
    }

    /// Set the status of a node.
    ///
    /// Returns `true` if the node exists.
    pub fn set_status(&self, node_id: &str, status: WorkerStatus) -> bool {
        if let Some(node) = self.nodes.write().get_mut(node_id) {
            node.status = status;
            true
        } else {
            false
        }
    }

    /// Run a health sweep: mark nodes whose last-seen timestamp exceeds
    /// `health_timeout_secs` as `Unreachable`.
    ///
    /// Returns the number of nodes that transitioned to `Unreachable`.
    pub fn health_sweep(&self) -> usize {
        let timeout = self.config.health_timeout_secs;
        let mut stale = 0usize;
        for node in self.nodes.write().values_mut() {
            if node.status == WorkerStatus::Healthy && node.age_secs() > timeout {
                node.status = WorkerStatus::Unreachable;
                stale += 1;
            }
        }
        stale
    }

    /// Remove all nodes whose status is `Deregistered` or `Unreachable`.
    ///
    /// Returns the number of nodes purged.
    pub fn purge_dead_nodes(&self) -> usize {
        let mut nodes = self.nodes.write();
        let before = nodes.len();
        nodes.retain(|_, n| {
            !matches!(n.status, WorkerStatus::Deregistered | WorkerStatus::Unreachable)
        });
        before - nodes.len()
    }

    // -----------------------------------------------------------------------
    // Query
    // -----------------------------------------------------------------------

    /// Total number of registered nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.read().len()
    }

    /// Whether the registry has no nodes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.read().is_empty()
    }

    /// Look up a node by ID.
    #[must_use]
    pub fn get(&self, node_id: &str) -> Option<WorkerNode> {
        self.nodes.read().get(node_id).cloned()
    }

    /// List all nodes.
    #[must_use]
    pub fn all_nodes(&self) -> Vec<WorkerNode> {
        self.nodes.read().values().cloned().collect()
    }

    /// List nodes with `Healthy` status.
    #[must_use]
    pub fn healthy_nodes(&self) -> Vec<WorkerNode> {
        self.nodes
            .read()
            .values()
            .filter(|n| n.status == WorkerStatus::Healthy)
            .cloned()
            .collect()
    }

    /// List nodes that have capacity for at least one more job.
    #[must_use]
    pub fn available_nodes(&self) -> Vec<WorkerNode> {
        self.nodes
            .read()
            .values()
            .filter(|n| n.has_capacity())
            .cloned()
            .collect()
    }

    /// List nodes capable of handling a specific job type.
    #[must_use]
    pub fn nodes_for_job_type(&self, job_type: &str) -> Vec<WorkerNode> {
        self.nodes
            .read()
            .values()
            .filter(|n| n.has_capacity() && n.supports_job_type(job_type))
            .cloned()
            .collect()
    }

    /// List nodes that have GPU acceleration and are available.
    #[must_use]
    pub fn gpu_nodes(&self) -> Vec<WorkerNode> {
        self.nodes
            .read()
            .values()
            .filter(|n| n.has_capacity() && n.capabilities.has_gpu)
            .cloned()
            .collect()
    }

    /// Select the least-loaded available node (fewest active jobs), optionally
    /// filtered by job type.  Returns `None` if no suitable node exists.
    #[must_use]
    pub fn least_loaded_node(&self, job_type: Option<&str>) -> Option<WorkerNode> {
        self.nodes
            .read()
            .values()
            .filter(|n| {
                n.has_capacity()
                    && job_type.map_or(true, |jt| n.supports_job_type(jt))
            })
            .min_by_key(|n| n.active_jobs)
            .cloned()
    }

    /// Cluster-level summary statistics.
    #[must_use]
    pub fn stats(&self) -> ClusterStats {
        let nodes = self.nodes.read();
        let total = nodes.len();
        let healthy = nodes.values().filter(|n| n.status == WorkerStatus::Healthy).count();
        let total_cpu: u32 = nodes
            .values()
            .filter(|n| n.status == WorkerStatus::Healthy)
            .map(|n| n.capabilities.cpu_cores)
            .sum();
        let total_memory_mb: u64 = nodes
            .values()
            .filter(|n| n.status == WorkerStatus::Healthy)
            .map(|n| n.capabilities.memory_mb)
            .sum();
        let total_active_jobs: u32 = nodes.values().map(|n| n.active_jobs).sum();
        let total_capacity: u32 = nodes
            .values()
            .filter(|n| n.status == WorkerStatus::Healthy)
            .map(|n| n.capabilities.max_concurrent_jobs)
            .sum();
        let gpu_count = nodes
            .values()
            .filter(|n| n.status == WorkerStatus::Healthy && n.capabilities.has_gpu)
            .count();

        ClusterStats {
            total_nodes: total,
            healthy_nodes: healthy,
            total_cpu_cores: total_cpu,
            total_memory_mb,
            total_active_jobs,
            total_job_capacity: total_capacity,
            gpu_nodes: gpu_count,
        }
    }
}

impl Default for ClusterRegistry {
    fn default() -> Self {
        Self::with_defaults()
    }
}

/// Cluster-level summary statistics.
#[derive(Debug, Clone)]
pub struct ClusterStats {
    /// Total number of registered nodes (all statuses).
    pub total_nodes: usize,
    /// Number of nodes with `Healthy` status.
    pub healthy_nodes: usize,
    /// Total CPU cores across all healthy nodes.
    pub total_cpu_cores: u32,
    /// Total memory in megabytes across all healthy nodes.
    pub total_memory_mb: u64,
    /// Total number of active jobs across all nodes.
    pub total_active_jobs: u32,
    /// Total job slots across all healthy nodes.
    pub total_job_capacity: u32,
    /// Number of healthy nodes with GPU support.
    pub gpu_nodes: usize,
}

impl ClusterStats {
    /// Overall cluster utilisation as a fraction 0.0–1.0.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn utilisation(&self) -> f64 {
        if self.total_job_capacity == 0 {
            return 0.0;
        }
        self.total_active_jobs as f64 / self.total_job_capacity as f64
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_addr(port: u16) -> SocketAddr {
        format!("127.0.0.1:{port}").parse().expect("valid addr")
    }

    fn make_node(id: &str, port: u16) -> WorkerNode {
        WorkerNode::new(id, id, make_addr(port), WorkerCapabilities::default())
    }

    #[test]
    fn test_register_and_count() {
        let reg = ClusterRegistry::with_defaults();
        assert!(reg.is_empty());
        reg.register(make_node("node-1", 7001));
        assert_eq!(reg.node_count(), 1);
    }

    #[test]
    fn test_register_is_new() {
        let reg = ClusterRegistry::with_defaults();
        assert!(reg.register(make_node("n1", 7001)));
        // Second registration with same ID is an update, not new.
        assert!(!reg.register(make_node("n1", 7001)));
    }

    #[test]
    fn test_deregister() {
        let reg = ClusterRegistry::with_defaults();
        reg.register(make_node("n1", 7001));
        assert!(reg.deregister("n1"));
        assert!(!reg.deregister("n1"));
        assert!(reg.is_empty());
    }

    #[test]
    fn test_get_node() {
        let reg = ClusterRegistry::with_defaults();
        reg.register(make_node("n1", 7001));
        let node = reg.get("n1").expect("should find node");
        assert_eq!(node.node_id, "n1");
        assert!(reg.get("nope").is_none());
    }

    #[test]
    fn test_healthy_nodes() {
        let reg = ClusterRegistry::with_defaults();
        reg.register(make_node("n1", 7001));
        let mut n2 = make_node("n2", 7002);
        n2.status = WorkerStatus::Unreachable;
        reg.register(n2);
        let healthy = reg.healthy_nodes();
        assert_eq!(healthy.len(), 1);
        assert_eq!(healthy[0].node_id, "n1");
    }

    #[test]
    fn test_health_sweep_marks_stale() {
        let config = RegistryConfig {
            health_timeout_secs: 1,
            max_nodes: 0,
        };
        let reg = ClusterRegistry::new(config);
        let mut node = make_node("n1", 7001);
        // Backdate last_seen so it's stale.
        node.last_seen_secs = current_timestamp().saturating_sub(10);
        reg.register(node);
        let stale = reg.health_sweep();
        assert_eq!(stale, 1);
        let n = reg.get("n1").expect("node should still exist");
        assert_eq!(n.status, WorkerStatus::Unreachable);
    }

    #[test]
    fn test_mark_seen_resets_status() {
        let reg = ClusterRegistry::with_defaults();
        let mut node = make_node("n1", 7001);
        node.status = WorkerStatus::Unreachable;
        reg.register(node);
        assert!(reg.mark_seen("n1"));
        let n = reg.get("n1").expect("node");
        assert_eq!(n.status, WorkerStatus::Healthy);
    }

    #[test]
    fn test_least_loaded_node_returns_min_active() {
        let reg = ClusterRegistry::with_defaults();
        let mut n1 = make_node("n1", 7001);
        n1.active_jobs = 3;
        let mut n2 = make_node("n2", 7002);
        n2.active_jobs = 1;
        reg.register(n1);
        reg.register(n2);
        let selected = reg.least_loaded_node(None).expect("should find node");
        assert_eq!(selected.node_id, "n2");
    }

    #[test]
    fn test_nodes_for_job_type() {
        let reg = ClusterRegistry::with_defaults();
        let mut caps = WorkerCapabilities::default();
        caps.supported_job_types = vec!["transcode".to_string(), "thumbnail".to_string()];
        let n1 = WorkerNode::new("n1", "n1", make_addr(7001), caps.clone());

        let mut caps2 = WorkerCapabilities::default();
        caps2.supported_job_types = vec!["analysis".to_string()];
        let n2 = WorkerNode::new("n2", "n2", make_addr(7002), caps2);

        reg.register(n1);
        reg.register(n2);

        let transcode_nodes = reg.nodes_for_job_type("transcode");
        assert_eq!(transcode_nodes.len(), 1);
        assert_eq!(transcode_nodes[0].node_id, "n1");

        let analysis_nodes = reg.nodes_for_job_type("analysis");
        assert_eq!(analysis_nodes.len(), 1);
    }

    #[test]
    fn test_cluster_stats() {
        let reg = ClusterRegistry::with_defaults();
        reg.register(make_node("n1", 7001));
        reg.register(make_node("n2", 7002));
        let stats = reg.stats();
        assert_eq!(stats.total_nodes, 2);
        assert_eq!(stats.healthy_nodes, 2);
        assert_eq!(stats.total_cpu_cores, 8); // 2 × 4 cores default
    }

    #[test]
    fn test_max_nodes_limit() {
        let config = RegistryConfig {
            health_timeout_secs: 30,
            max_nodes: 2,
        };
        let reg = ClusterRegistry::new(config);
        assert!(reg.register(make_node("n1", 7001)));
        assert!(reg.register(make_node("n2", 7002)));
        // Third node should be rejected.
        assert!(!reg.register(make_node("n3", 7003)));
        assert_eq!(reg.node_count(), 2);
    }

    #[test]
    fn test_purge_dead_nodes() {
        let reg = ClusterRegistry::with_defaults();
        let mut n1 = make_node("n1", 7001);
        n1.status = WorkerStatus::Unreachable;
        reg.register(n1);
        reg.register(make_node("n2", 7002));
        let purged = reg.purge_dead_nodes();
        assert_eq!(purged, 1);
        assert_eq!(reg.node_count(), 1);
    }

    #[test]
    fn test_announcement_join_registers_node() {
        let reg = ClusterRegistry::with_defaults();
        let node = make_node("n1", 7001);
        let announcement = DiscoveryAnnouncement::join(&node);
        assert!(reg.handle_announcement(announcement));
        assert_eq!(reg.node_count(), 1);
    }

    #[test]
    fn test_announcement_leave_removes_node() {
        let reg = ClusterRegistry::with_defaults();
        let node = make_node("n1", 7001);
        let join = DiscoveryAnnouncement::join(&node);
        reg.handle_announcement(join);
        assert_eq!(reg.node_count(), 1);

        let leave = DiscoveryAnnouncement::leave(&node);
        reg.handle_announcement(leave);
        assert_eq!(reg.node_count(), 0);
    }

    #[test]
    fn test_announcement_roundtrip_serialisation() {
        let node = make_node("n1", 7001);
        let ann = DiscoveryAnnouncement::join(&node);
        let bytes = ann.to_bytes().expect("serialise");
        let ann2 = DiscoveryAnnouncement::from_bytes(&bytes).expect("deserialise");
        assert_eq!(ann2.node_id, "n1");
        assert_eq!(ann2.announcement_type, AnnouncementType::Join);
    }

    #[test]
    fn test_worker_node_has_capacity() {
        let mut node = make_node("n1", 7001);
        node.active_jobs = node.capabilities.max_concurrent_jobs;
        assert!(!node.has_capacity());
        node.active_jobs = 0;
        assert!(node.has_capacity());
    }

    #[test]
    fn test_worker_node_free_slots() {
        let mut node = make_node("n1", 7001);
        node.capabilities.max_concurrent_jobs = 8;
        node.active_jobs = 3;
        assert_eq!(node.free_slots(), 5);
    }

    #[test]
    fn test_cluster_stats_utilisation() {
        let stats = ClusterStats {
            total_nodes: 1,
            healthy_nodes: 1,
            total_cpu_cores: 4,
            total_memory_mb: 8192,
            total_active_jobs: 2,
            total_job_capacity: 4,
            gpu_nodes: 0,
        };
        assert!((stats.utilisation() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_all_nodes_list() {
        let reg = ClusterRegistry::with_defaults();
        reg.register(make_node("n1", 7001));
        reg.register(make_node("n2", 7002));
        let nodes = reg.all_nodes();
        assert_eq!(nodes.len(), 2);
    }

    #[test]
    fn test_gpu_nodes_filter() {
        let reg = ClusterRegistry::with_defaults();
        let mut gpu_caps = WorkerCapabilities::default();
        gpu_caps.has_gpu = true;
        gpu_caps.gpu_count = 2;
        let gpu_node = WorkerNode::new("gpu1", "GPU Node", make_addr(7010), gpu_caps);
        reg.register(gpu_node);
        reg.register(make_node("cpu1", 7011));
        let gpu_nodes = reg.gpu_nodes();
        assert_eq!(gpu_nodes.len(), 1);
        assert_eq!(gpu_nodes[0].node_id, "gpu1");
    }
}
