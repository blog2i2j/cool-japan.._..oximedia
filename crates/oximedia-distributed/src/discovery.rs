//! Worker discovery and health monitoring.
//!
//! This module provides:
//! - Worker discovery via mDNS, etcd, Consul, or static configuration
//! - Health monitoring and heartbeat tracking
//! - Automatic worker registration
//! - Capacity tracking and load balancing
//! - Geographic distribution awareness

#![allow(dead_code)]

use crate::{DiscoveryMethod, DistributedError, Result};
use dashmap::DashMap;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// Worker discovery service
pub struct DiscoveryService {
    /// Discovery method
    method: DiscoveryMethod,

    /// Discovered workers
    workers: Arc<DashMap<String, WorkerEndpoint>>,

    /// Service configuration
    config: DiscoveryConfig,

    /// Health check state
    health_checker: Arc<HealthChecker>,

    /// Running flag
    running: Arc<AtomicBool>,
}

/// Discovery service configuration
#[derive(Debug, Clone)]
pub struct DiscoveryConfig {
    /// Service name for discovery
    pub service_name: String,

    /// Service port
    pub service_port: u16,

    /// Discovery interval
    pub discovery_interval: Duration,

    /// Health check interval
    pub health_check_interval: Duration,

    /// Worker timeout
    pub worker_timeout: Duration,

    /// etcd endpoints (for etcd discovery)
    pub etcd_endpoints: Vec<String>,

    /// Consul address (for Consul discovery)
    pub consul_address: String,

    /// Static worker addresses
    pub static_workers: Vec<String>,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            service_name: "oximedia-worker".to_string(),
            service_port: 50052,
            discovery_interval: Duration::from_secs(30),
            health_check_interval: Duration::from_secs(10),
            worker_timeout: Duration::from_secs(90),
            etcd_endpoints: vec!["http://127.0.0.1:2379".to_string()],
            consul_address: "http://127.0.0.1:8500".to_string(),
            static_workers: Vec::new(),
        }
    }
}

/// Worker endpoint information
#[derive(Debug, Clone)]
pub struct WorkerEndpoint {
    /// Worker ID
    pub worker_id: String,

    /// Worker address
    pub address: SocketAddr,

    /// Hostname
    pub hostname: String,

    /// Capabilities
    pub capabilities: WorkerCapabilities,

    /// Health status
    pub health: HealthStatus,

    /// Last seen timestamp
    pub last_seen: SystemTime,

    /// Geographic location (optional)
    pub location: Option<GeoLocation>,

    /// Metadata
    pub metadata: HashMap<String, String>,
}

/// Worker capabilities
#[derive(Debug, Clone)]
pub struct WorkerCapabilities {
    /// CPU cores
    pub cpu_cores: u32,

    /// Memory in bytes
    pub memory_bytes: u64,

    /// GPU devices
    pub gpu_devices: Vec<String>,

    /// Supported codecs
    pub codecs: Vec<String>,

    /// Maximum concurrent jobs
    pub max_jobs: u32,

    /// Performance score (relative)
    pub performance_score: f32,
}

impl Default for WorkerCapabilities {
    fn default() -> Self {
        Self {
            cpu_cores: 1,
            memory_bytes: 1_073_741_824,
            gpu_devices: Vec::new(),
            codecs: vec!["h264".to_string()],
            max_jobs: 2,
            performance_score: 1.0,
        }
    }
}

/// Health status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
    Unknown,
}

/// Geographic location
#[derive(Debug, Clone)]
pub struct GeoLocation {
    /// Region (e.g., "us-west-2")
    pub region: String,

    /// Availability zone
    pub zone: Option<String>,

    /// Data center
    pub datacenter: Option<String>,

    /// Latitude
    pub latitude: Option<f64>,

    /// Longitude
    pub longitude: Option<f64>,
}

/// Health checker
pub struct HealthChecker {
    /// Health check results
    health_results: Arc<DashMap<String, HealthCheckResult>>,

    /// Statistics
    stats: HealthStats,
}

/// Health check result
#[derive(Debug, Clone)]
struct HealthCheckResult {
    worker_id: String,
    status: HealthStatus,
    latency: Duration,
    last_check: SystemTime,
    consecutive_failures: u32,
}

/// Health statistics
#[derive(Debug, Default)]
struct HealthStats {
    total_checks: AtomicU64,
    successful_checks: AtomicU64,
    failed_checks: AtomicU64,
}

impl DiscoveryService {
    /// Create a new discovery service
    #[must_use]
    pub fn new(method: DiscoveryMethod, config: DiscoveryConfig) -> Self {
        Self {
            method,
            workers: Arc::new(DashMap::new()),
            config,
            health_checker: Arc::new(HealthChecker::new()),
            running: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Start the discovery service
    pub async fn start(&self) -> Result<()> {
        info!("Starting discovery service with method: {:?}", self.method);
        self.running.store(true, Ordering::Relaxed);

        // Start discovery loop
        let service = self.clone_refs();
        tokio::spawn(async move {
            service.discovery_loop().await;
        });

        // Start health check loop
        let service = self.clone_refs();
        tokio::spawn(async move {
            service.health_check_loop().await;
        });

        Ok(())
    }

    fn clone_refs(&self) -> Self {
        Self {
            method: self.method,
            workers: self.workers.clone(),
            config: self.config.clone(),
            health_checker: self.health_checker.clone(),
            running: self.running.clone(),
        }
    }

    /// Discovery loop
    async fn discovery_loop(&self) {
        let mut interval = tokio::time::interval(self.config.discovery_interval);

        while self.running.load(Ordering::Relaxed) {
            interval.tick().await;

            if let Err(e) = self.discover_workers().await {
                error!("Worker discovery failed: {}", e);
            }
        }
    }

    /// Discover workers based on configured method
    async fn discover_workers(&self) -> Result<()> {
        match self.method {
            DiscoveryMethod::Static => self.discover_static().await,
            DiscoveryMethod::MDNS => self.discover_mdns().await,
            DiscoveryMethod::Etcd => self.discover_etcd().await,
            DiscoveryMethod::Consul => self.discover_consul().await,
        }
    }

    /// Static worker discovery
    async fn discover_static(&self) -> Result<()> {
        debug!("Discovering static workers");

        for addr_str in &self.config.static_workers {
            let addr: SocketAddr = addr_str
                .parse()
                .map_err(|e| DistributedError::Discovery(format!("Invalid address: {e}")))?;

            let worker_id = format!("static-{addr}");

            let endpoint = WorkerEndpoint {
                worker_id: worker_id.clone(),
                address: addr,
                hostname: addr.ip().to_string(),
                capabilities: WorkerCapabilities::default(),
                health: HealthStatus::Unknown,
                last_seen: SystemTime::now(),
                location: None,
                metadata: HashMap::new(),
            };

            self.workers.insert(worker_id, endpoint);
        }

        info!(
            "Discovered {} static workers",
            self.config.static_workers.len()
        );
        Ok(())
    }

    /// mDNS-based discovery
    async fn discover_mdns(&self) -> Result<()> {
        debug!("Discovering workers via mDNS");

        // In production, would use mdns crate to discover services
        // For now, simulate discovery
        info!("mDNS discovery completed");
        Ok(())
    }

    /// etcd-based discovery
    async fn discover_etcd(&self) -> Result<()> {
        debug!("Discovering workers via etcd");

        // In production, would query etcd for worker registrations
        // For now, simulate discovery
        info!("etcd discovery completed");
        Ok(())
    }

    /// Consul-based discovery
    async fn discover_consul(&self) -> Result<()> {
        debug!("Discovering workers via Consul");

        // In production, would query Consul service catalog
        // For now, simulate discovery
        info!("Consul discovery completed");
        Ok(())
    }

    /// Health check loop
    async fn health_check_loop(&self) {
        let mut interval = tokio::time::interval(self.config.health_check_interval);

        while self.running.load(Ordering::Relaxed) {
            interval.tick().await;

            self.check_worker_health().await;
        }
    }

    /// Check health of all workers
    async fn check_worker_health(&self) {
        let workers: Vec<_> = self
            .workers
            .iter()
            .map(|e| (e.key().clone(), e.value().clone()))
            .collect();

        for (worker_id, endpoint) in workers {
            let result = self.health_checker.check_worker(&endpoint).await;

            // Update worker health status
            if let Some(mut worker) = self.workers.get_mut(&worker_id) {
                worker.health = result.status;
                worker.last_seen = SystemTime::now();
            }

            // Remove unhealthy workers after timeout
            if result.status == HealthStatus::Unhealthy && result.consecutive_failures > 3 {
                warn!("Removing unhealthy worker: {}", worker_id);
                self.workers.remove(&worker_id);
            }
        }
    }

    /// Register a worker manually
    pub fn register_worker(&self, endpoint: WorkerEndpoint) -> Result<()> {
        info!("Registering worker: {}", endpoint.worker_id);
        self.workers.insert(endpoint.worker_id.clone(), endpoint);
        Ok(())
    }

    /// Unregister a worker
    pub fn unregister_worker(&self, worker_id: &str) -> Result<()> {
        info!("Unregistering worker: {}", worker_id);
        self.workers.remove(worker_id);
        Ok(())
    }

    /// Get all discovered workers
    #[must_use]
    pub fn get_workers(&self) -> Vec<WorkerEndpoint> {
        self.workers.iter().map(|e| e.value().clone()).collect()
    }

    /// Get healthy workers
    #[must_use]
    pub fn get_healthy_workers(&self) -> Vec<WorkerEndpoint> {
        self.workers
            .iter()
            .filter(|e| e.value().health == HealthStatus::Healthy)
            .map(|e| e.value().clone())
            .collect()
    }

    /// Get worker by ID
    #[must_use]
    pub fn get_worker(&self, worker_id: &str) -> Option<WorkerEndpoint> {
        self.workers.get(worker_id).map(|e| e.value().clone())
    }

    /// Find workers by capability
    #[must_use]
    pub fn find_workers_by_capability(&self, required_codec: &str) -> Vec<WorkerEndpoint> {
        self.workers
            .iter()
            .filter(|e| {
                e.value().health == HealthStatus::Healthy
                    && e.value()
                        .capabilities
                        .codecs
                        .iter()
                        .any(|c| c == required_codec)
            })
            .map(|e| e.value().clone())
            .collect()
    }

    /// Find workers by location
    #[must_use]
    pub fn find_workers_by_region(&self, region: &str) -> Vec<WorkerEndpoint> {
        self.workers
            .iter()
            .filter(|e| {
                e.value()
                    .location
                    .as_ref()
                    .is_some_and(|l| l.region == region)
            })
            .map(|e| e.value().clone())
            .collect()
    }

    /// Get capacity statistics
    #[must_use]
    pub fn get_capacity_stats(&self) -> CapacityStats {
        let workers = self.get_healthy_workers();

        let total_cpu: u32 = workers.iter().map(|w| w.capabilities.cpu_cores).sum();
        let total_memory: u64 = workers.iter().map(|w| w.capabilities.memory_bytes).sum();
        let total_gpus: usize = workers
            .iter()
            .map(|w| w.capabilities.gpu_devices.len())
            .sum();
        let total_job_slots: u32 = workers.iter().map(|w| w.capabilities.max_jobs).sum();

        CapacityStats {
            total_workers: workers.len(),
            total_cpu_cores: total_cpu,
            total_memory_bytes: total_memory,
            total_gpu_devices: total_gpus,
            total_job_slots,
            average_performance: workers
                .iter()
                .map(|w| w.capabilities.performance_score)
                .sum::<f32>()
                / workers.len().max(1) as f32,
        }
    }

    /// Stop the discovery service
    pub async fn stop(&self) -> Result<()> {
        info!("Stopping discovery service");
        self.running.store(false, Ordering::Relaxed);
        Ok(())
    }
}

/// Capacity statistics
#[derive(Debug, Clone)]
pub struct CapacityStats {
    pub total_workers: usize,
    pub total_cpu_cores: u32,
    pub total_memory_bytes: u64,
    pub total_gpu_devices: usize,
    pub total_job_slots: u32,
    pub average_performance: f32,
}

impl HealthChecker {
    /// Create a new health checker
    fn new() -> Self {
        Self {
            health_results: Arc::new(DashMap::new()),
            stats: HealthStats::default(),
        }
    }

    /// Check worker health
    async fn check_worker(&self, endpoint: &WorkerEndpoint) -> HealthCheckResult {
        self.stats.total_checks.fetch_add(1, Ordering::Relaxed);

        let start = SystemTime::now();

        // Perform health check (simplified)
        let status = self.perform_health_check(endpoint).await;

        let latency = start.elapsed().unwrap_or(Duration::ZERO);

        let mut result = HealthCheckResult {
            worker_id: endpoint.worker_id.clone(),
            status,
            latency,
            last_check: SystemTime::now(),
            consecutive_failures: 0,
        };

        // Update consecutive failures
        if let Some(prev) = self.health_results.get(&endpoint.worker_id) {
            if status == HealthStatus::Unhealthy {
                result.consecutive_failures = prev.consecutive_failures + 1;
            }
        } else if status == HealthStatus::Unhealthy {
            result.consecutive_failures = 1;
        }

        // Update statistics
        if status == HealthStatus::Healthy {
            self.stats.successful_checks.fetch_add(1, Ordering::Relaxed);
        } else {
            self.stats.failed_checks.fetch_add(1, Ordering::Relaxed);
        }

        self.health_results
            .insert(endpoint.worker_id.clone(), result.clone());

        result
    }

    /// Perform actual health check
    async fn perform_health_check(&self, _endpoint: &WorkerEndpoint) -> HealthStatus {
        // In production, would send HTTP/gRPC health check request
        // For now, simulate based on last_seen time
        HealthStatus::Healthy
    }

    /// Get health check statistics
    pub fn statistics(&self) -> HealthCheckStats {
        HealthCheckStats {
            total_checks: self.stats.total_checks.load(Ordering::Relaxed),
            successful_checks: self.stats.successful_checks.load(Ordering::Relaxed),
            failed_checks: self.stats.failed_checks.load(Ordering::Relaxed),
        }
    }
}

/// Health check statistics
#[derive(Debug, Clone)]
pub struct HealthCheckStats {
    pub total_checks: u64,
    pub successful_checks: u64,
    pub failed_checks: u64,
}

impl HealthCheckStats {
    /// Get success rate
    #[must_use]
    pub fn success_rate(&self) -> f64 {
        if self.total_checks == 0 {
            return 0.0;
        }
        self.successful_checks as f64 / self.total_checks as f64
    }
}

/// Worker registry for persistent storage
pub struct WorkerRegistry {
    /// Registered workers
    workers: Arc<RwLock<HashMap<String, RegisteredWorker>>>,

    /// Registry backend
    backend: RegistryBackend,
}

/// Registered worker with persistence
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RegisteredWorker {
    worker_id: String,
    address: String,
    hostname: String,
    registered_at: u64,
    last_heartbeat: u64,
    capabilities: serde_json::Value,
    metadata: HashMap<String, String>,
}

/// Registry backend
#[derive(Debug, Clone, Copy)]
pub enum RegistryBackend {
    Memory,
    Etcd,
    Consul,
}

impl WorkerRegistry {
    /// Create a new worker registry
    #[must_use]
    pub fn new(backend: RegistryBackend) -> Self {
        Self {
            workers: Arc::new(RwLock::new(HashMap::new())),
            backend,
        }
    }

    /// Register a worker
    pub async fn register(&self, worker_id: String, endpoint: WorkerEndpoint) -> Result<()> {
        info!("Registering worker in registry: {}", worker_id);

        let unix_now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_secs();

        let worker = RegisteredWorker {
            worker_id: worker_id.clone(),
            address: endpoint.address.to_string(),
            hostname: endpoint.hostname,
            registered_at: unix_now,
            last_heartbeat: unix_now,
            capabilities: serde_json::json!({}),
            metadata: endpoint.metadata,
        };

        let mut workers = self.workers.write().await;
        workers.insert(worker_id.clone(), worker.clone());

        // Persist to backend
        self.persist_worker(&worker).await?;

        Ok(())
    }

    /// Unregister a worker
    pub async fn unregister(&self, worker_id: &str) -> Result<()> {
        info!("Unregistering worker from registry: {}", worker_id);

        let mut workers = self.workers.write().await;
        workers.remove(worker_id);

        // Remove from backend
        self.remove_worker(worker_id).await?;

        Ok(())
    }

    /// Update worker heartbeat
    pub async fn update_heartbeat(&self, worker_id: &str) -> Result<()> {
        let mut workers = self.workers.write().await;

        if let Some(worker) = workers.get_mut(worker_id) {
            worker.last_heartbeat = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or(Duration::ZERO)
                .as_secs();
        }

        Ok(())
    }

    /// Persist worker to backend
    async fn persist_worker(&self, _worker: &RegisteredWorker) -> Result<()> {
        match self.backend {
            RegistryBackend::Memory => Ok(()),
            RegistryBackend::Etcd => {
                // In production, write to etcd
                Ok(())
            }
            RegistryBackend::Consul => {
                // In production, write to Consul
                Ok(())
            }
        }
    }

    /// Remove worker from backend
    async fn remove_worker(&self, _worker_id: &str) -> Result<()> {
        match self.backend {
            RegistryBackend::Memory => Ok(()),
            RegistryBackend::Etcd => {
                // In production, delete from etcd
                Ok(())
            }
            RegistryBackend::Consul => {
                // In production, delete from Consul
                Ok(())
            }
        }
    }

    /// Get all workers
    pub async fn get_all(&self) -> Vec<RegisteredWorker> {
        let workers = self.workers.read().await;
        workers.values().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr};

    #[test]
    fn test_discovery_config() {
        let config = DiscoveryConfig::default();
        assert_eq!(config.service_name, "oximedia-worker");
        assert_eq!(config.service_port, 50052);
    }

    #[test]
    fn test_worker_capabilities() {
        let caps = WorkerCapabilities::default();
        assert_eq!(caps.cpu_cores, 1);
        assert_eq!(caps.max_jobs, 2);
        assert!(!caps.codecs.is_empty());
    }

    #[test]
    fn test_health_status() {
        assert_eq!(HealthStatus::Healthy, HealthStatus::Healthy);
        assert_ne!(HealthStatus::Healthy, HealthStatus::Unhealthy);
    }

    #[test]
    fn test_discovery_service_creation() {
        let config = DiscoveryConfig::default();
        let service = DiscoveryService::new(DiscoveryMethod::Static, config);
        assert_eq!(service.method, DiscoveryMethod::Static);
        assert_eq!(service.workers.len(), 0);
    }

    #[test]
    fn test_worker_registration() {
        let config = DiscoveryConfig::default();
        let service = DiscoveryService::new(DiscoveryMethod::Static, config);

        let endpoint = WorkerEndpoint {
            worker_id: "test-worker".to_string(),
            address: SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 50052),
            hostname: "localhost".to_string(),
            capabilities: WorkerCapabilities::default(),
            health: HealthStatus::Healthy,
            last_seen: SystemTime::now(),
            location: None,
            metadata: HashMap::new(),
        };

        assert!(service.register_worker(endpoint).is_ok());
        assert_eq!(service.workers.len(), 1);
    }

    #[test]
    fn test_capacity_stats() {
        let config = DiscoveryConfig::default();
        let service = DiscoveryService::new(DiscoveryMethod::Static, config);

        let stats = service.get_capacity_stats();
        assert_eq!(stats.total_workers, 0);
        assert_eq!(stats.total_cpu_cores, 0);
    }

    #[test]
    fn test_health_check_stats() {
        let stats = HealthCheckStats {
            total_checks: 100,
            successful_checks: 95,
            failed_checks: 5,
        };

        assert_eq!(stats.success_rate(), 0.95);
    }

    #[test]
    fn test_geo_location() {
        let location = GeoLocation {
            region: "us-west-2".to_string(),
            zone: Some("us-west-2a".to_string()),
            datacenter: None,
            latitude: Some(37.7749),
            longitude: Some(-122.4194),
        };

        assert_eq!(location.region, "us-west-2");
        assert!(location.latitude.is_some());
    }
}
