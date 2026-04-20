//! Cluster support for distributed live streaming.
//!
//! This module provides clustering capabilities including:
//! - Multi-node deployment
//! - Load balancing
//! - Stream replication
//! - Health monitoring
//! - Automatic failover

use crate::error::NetResult;
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;
use uuid::Uuid;

/// Node state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeState {
    /// Node is initializing.
    Initializing,

    /// Node is active and healthy.
    Active,

    /// Node is degraded (partial functionality).
    Degraded,

    /// Node is unhealthy.
    Unhealthy,

    /// Node is shutting down.
    ShuttingDown,

    /// Node is offline.
    Offline,
}

/// Node role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeRole {
    /// Primary node (handles writes).
    Primary,

    /// Secondary node (read replicas).
    Secondary,

    /// Edge node (CDN edge).
    Edge,
}

/// Cluster node information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    /// Node ID.
    pub id: Uuid,

    /// Node name.
    pub name: String,

    /// Node address.
    pub address: SocketAddr,

    /// Node role.
    pub role: NodeRole,

    /// Node state.
    pub state: NodeState,

    /// Node region/datacenter.
    pub region: String,

    /// Active stream count.
    pub stream_count: usize,

    /// Active viewer count.
    pub viewer_count: usize,

    /// CPU usage (percentage).
    pub cpu_usage: f64,

    /// Memory usage (percentage).
    pub memory_usage: f64,

    /// Network bandwidth usage (bytes/sec).
    pub bandwidth_usage: u64,

    /// Last heartbeat time.
    pub last_heartbeat: DateTime<Utc>,

    /// Node start time.
    pub start_time: DateTime<Utc>,

    /// Node version.
    pub version: String,
}

impl NodeInfo {
    /// Creates a new node info.
    #[must_use]
    pub fn new(name: impl Into<String>, address: SocketAddr, region: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            address,
            role: NodeRole::Secondary,
            state: NodeState::Initializing,
            region: region.into(),
            stream_count: 0,
            viewer_count: 0,
            cpu_usage: 0.0,
            memory_usage: 0.0,
            bandwidth_usage: 0,
            last_heartbeat: Utc::now(),
            start_time: Utc::now(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    /// Checks if the node is healthy.
    #[must_use]
    pub fn is_healthy(&self) -> bool {
        self.state == NodeState::Active || self.state == NodeState::Degraded
    }

    /// Checks if heartbeat is recent.
    #[must_use]
    pub fn has_recent_heartbeat(&self, timeout: Duration) -> bool {
        if let Ok(elapsed) = (Utc::now() - self.last_heartbeat).to_std() {
            elapsed < timeout
        } else {
            false
        }
    }
}

/// Cluster configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterConfig {
    /// Node name.
    pub node_name: String,

    /// Node address.
    pub node_address: SocketAddr,

    /// Region/datacenter.
    pub region: String,

    /// Seed nodes for cluster discovery.
    pub seed_nodes: Vec<SocketAddr>,

    /// Heartbeat interval.
    pub heartbeat_interval: Duration,

    /// Heartbeat timeout.
    pub heartbeat_timeout: Duration,

    /// Enable stream replication.
    pub enable_replication: bool,

    /// Replication factor.
    pub replication_factor: usize,

    /// Enable automatic failover.
    pub enable_failover: bool,
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self {
            node_name: hostname::get()
                .ok()
                .and_then(|h| h.into_string().ok())
                .unwrap_or_else(|| "unknown".to_string()),
            node_address: SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), 8080),
            region: "default".to_string(),
            seed_nodes: Vec::new(),
            heartbeat_interval: Duration::from_secs(5),
            heartbeat_timeout: Duration::from_secs(15),
            enable_replication: true,
            replication_factor: 2,
            enable_failover: true,
        }
    }
}

/// Cluster node.
pub struct ClusterNode {
    /// Local node information.
    local_info: RwLock<NodeInfo>,

    /// Cluster configuration.
    config: ClusterConfig,

    /// Known nodes in the cluster.
    nodes: RwLock<HashMap<Uuid, NodeInfo>>,

    /// Stream assignments (stream_id -> node_id).
    stream_assignments: RwLock<HashMap<Uuid, Vec<Uuid>>>,
}

impl ClusterNode {
    /// Creates a new cluster node.
    ///
    /// # Errors
    ///
    /// Returns an error if initialization fails.
    pub async fn new(config: ClusterConfig) -> NetResult<Self> {
        let local_info = NodeInfo::new(&config.node_name, config.node_address, &config.region);

        Ok(Self {
            local_info: RwLock::new(local_info),
            config,
            nodes: RwLock::new(HashMap::new()),
            stream_assignments: RwLock::new(HashMap::new()),
        })
    }

    /// Starts the cluster node.
    ///
    /// # Errors
    ///
    /// Returns an error if startup fails.
    pub async fn start(&self) -> NetResult<()> {
        // Set state to active
        {
            let mut info = self.local_info.write();
            info.state = NodeState::Active;
        }

        // Start heartbeat task
        self.start_heartbeat_task();

        // Start health check task
        self.start_health_check_task();

        // Discover cluster nodes
        self.discover_nodes().await?;

        Ok(())
    }

    /// Discovers nodes in the cluster.
    async fn discover_nodes(&self) -> NetResult<()> {
        // In production, this would contact seed nodes to discover the cluster
        // For now, we'll just add seed nodes as known nodes

        for seed_addr in &self.config.seed_nodes {
            let node_info =
                NodeInfo::new(format!("node-{seed_addr}"), *seed_addr, &self.config.region);

            let mut nodes = self.nodes.write();
            nodes.insert(node_info.id, node_info);
        }

        Ok(())
    }

    /// Starts heartbeat task.
    fn start_heartbeat_task(&self) {
        let interval = self.config.heartbeat_interval;
        let local_info_data = self.local_info.read().clone();
        let local_info = Arc::new(RwLock::new(local_info_data));

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(interval);

            loop {
                interval.tick().await;

                // Update heartbeat timestamp
                {
                    let mut info = local_info.write();
                    info.last_heartbeat = Utc::now();
                }

                // Send heartbeat to other nodes
                // (Would be implemented with actual networking)
            }
        });
    }

    /// Starts health check task.
    fn start_health_check_task(&self) {
        let timeout = self.config.heartbeat_timeout;
        let nodes_map: HashMap<Uuid, NodeInfo> = self.nodes.read().clone();
        let nodes = Arc::new(RwLock::new(nodes_map));

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(5));

            loop {
                interval.tick().await;

                // Check node health
                let mut nodes = nodes.write();
                for (_, node) in nodes.iter_mut() {
                    if !node.has_recent_heartbeat(timeout) {
                        node.state = NodeState::Unhealthy;
                    }
                }
            }
        });
    }

    /// Assigns a stream to nodes.
    pub fn assign_stream(&self, stream_id: Uuid) -> Vec<Uuid> {
        let nodes = self.nodes.read();

        // Select healthy nodes
        let healthy_nodes: Vec<_> = nodes.values().filter(|n| n.is_healthy()).collect();

        if healthy_nodes.is_empty() {
            return vec![self.local_info.read().id];
        }

        // Simple round-robin assignment
        let count = self.config.replication_factor.min(healthy_nodes.len());
        let assigned: Vec<Uuid> = healthy_nodes.iter().take(count).map(|n| n.id).collect();

        // Store assignment
        {
            let mut assignments = self.stream_assignments.write();
            assignments.insert(stream_id, assigned.clone());
        }

        assigned
    }

    /// Gets nodes assigned to a stream.
    #[must_use]
    pub fn get_stream_nodes(&self, stream_id: Uuid) -> Option<Vec<Uuid>> {
        let assignments = self.stream_assignments.read();
        assignments.get(&stream_id).cloned()
    }

    /// Updates local node metrics.
    pub fn update_metrics(&self, stream_count: usize, viewer_count: usize) {
        let mut info = self.local_info.write();
        info.stream_count = stream_count;
        info.viewer_count = viewer_count;

        // Update CPU and memory usage (would use actual metrics)
        info.cpu_usage = 0.0;
        info.memory_usage = 0.0;
    }

    /// Gets local node information.
    #[must_use]
    pub fn local_info(&self) -> NodeInfo {
        self.local_info.read().clone()
    }

    /// Gets all cluster nodes.
    #[must_use]
    pub fn cluster_nodes(&self) -> Vec<NodeInfo> {
        let nodes = self.nodes.read();
        nodes.values().cloned().collect()
    }

    /// Gets healthy nodes.
    #[must_use]
    pub fn healthy_nodes(&self) -> Vec<NodeInfo> {
        let nodes = self.nodes.read();
        nodes.values().filter(|n| n.is_healthy()).cloned().collect()
    }

    /// Selects best node for a viewer based on region.
    #[must_use]
    pub fn select_node_for_viewer(&self, viewer_region: Option<&str>) -> Option<NodeInfo> {
        let nodes = self.nodes.read();
        let healthy: Vec<_> = nodes.values().filter(|n| n.is_healthy()).collect();

        if healthy.is_empty() {
            return Some(self.local_info.read().clone());
        }

        // Prefer nodes in same region
        if let Some(region) = viewer_region {
            if let Some(node) = healthy.iter().find(|n| n.region == region) {
                return Some((*node).clone());
            }
        }

        // Select node with lowest load
        healthy
            .iter()
            .min_by_key(|n| n.viewer_count)
            .map(|n| (*n).clone())
    }

    /// Shuts down the cluster node.
    pub async fn shutdown(&self) -> NetResult<()> {
        let mut info = self.local_info.write();
        info.state = NodeState::ShuttingDown;

        // Notify other nodes (would be implemented)

        info.state = NodeState::Offline;

        Ok(())
    }
}

/// Hostname helper module.
mod hostname {
    use std::ffi::OsString;
    use std::io;

    #[must_use]
    pub fn get() -> io::Result<OsString> {
        // Simple hostname implementation
        Ok(OsString::from("localhost"))
    }
}
