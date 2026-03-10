#![allow(dead_code)]
//! Worker pool management with grouping and tagging.
//!
//! Provides a logical grouping layer on top of individual workers:
//! - Worker groups (pools) with shared properties and tags
//! - Pool-level capacity tracking and utilization metrics
//! - Worker assignment and removal from pools
//! - Pool-based job routing (match job requirements to pool capabilities)
//! - Drain and maintenance mode for individual pools

use std::collections::{HashMap, HashSet};

/// Unique identifier for a worker pool.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct PoolId(pub String);

impl PoolId {
    /// Create a new pool identifier.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Get the pool ID as a string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for PoolId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Status of a worker pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum PoolStatus {
    /// Pool is active and accepting jobs.
    Active,
    /// Pool is draining; existing jobs complete but no new ones accepted.
    Draining,
    /// Pool is in maintenance mode; no jobs accepted.
    Maintenance,
    /// Pool is disabled.
    Disabled,
}

impl std::fmt::Display for PoolStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Active => write!(f, "Active"),
            Self::Draining => write!(f, "Draining"),
            Self::Maintenance => write!(f, "Maintenance"),
            Self::Disabled => write!(f, "Disabled"),
        }
    }
}

/// Error type for worker pool operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PoolError {
    /// Pool not found.
    PoolNotFound(String),
    /// Pool already exists.
    PoolAlreadyExists(String),
    /// Worker not found in pool.
    WorkerNotFound(String),
    /// Worker already in pool.
    WorkerAlreadyInPool(String),
    /// Pool is not accepting jobs.
    PoolNotAccepting(String),
    /// Pool capacity exceeded.
    CapacityExceeded {
        /// The pool id.
        pool_id: String,
        /// Maximum workers allowed.
        max: usize,
    },
}

impl std::fmt::Display for PoolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PoolNotFound(id) => write!(f, "pool not found: {id}"),
            Self::PoolAlreadyExists(id) => write!(f, "pool already exists: {id}"),
            Self::WorkerNotFound(id) => write!(f, "worker not found: {id}"),
            Self::WorkerAlreadyInPool(id) => write!(f, "worker already in pool: {id}"),
            Self::PoolNotAccepting(id) => write!(f, "pool not accepting jobs: {id}"),
            Self::CapacityExceeded { pool_id, max } => {
                write!(f, "pool {pool_id} capacity exceeded (max {max})")
            }
        }
    }
}

impl std::error::Error for PoolError {}

/// Result type for pool operations.
pub type Result<T> = std::result::Result<T, PoolError>;

/// A single worker pool definition.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WorkerPool {
    /// Pool identifier.
    pub id: PoolId,
    /// Human-readable name.
    pub name: String,
    /// Pool status.
    pub status: PoolStatus,
    /// Tags describing pool capabilities (e.g., "gpu", "high-memory").
    pub tags: HashSet<String>,
    /// Maximum number of workers in this pool (0 = unlimited).
    pub max_workers: usize,
    /// Worker IDs currently assigned to this pool.
    pub workers: HashSet<String>,
    /// Priority weight for job routing (higher = preferred).
    pub priority_weight: u32,
}

impl WorkerPool {
    /// Create a new active pool.
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: PoolId::new(id),
            name: name.into(),
            status: PoolStatus::Active,
            tags: HashSet::new(),
            max_workers: 0,
            workers: HashSet::new(),
            priority_weight: 100,
        }
    }

    /// Set the maximum worker count.
    #[must_use]
    pub fn with_max_workers(mut self, max: usize) -> Self {
        self.max_workers = max;
        self
    }

    /// Add a tag to the pool.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.insert(tag.into());
        self
    }

    /// Set the priority weight.
    #[must_use]
    pub fn with_priority(mut self, weight: u32) -> Self {
        self.priority_weight = weight;
        self
    }

    /// Check if the pool is accepting new work.
    #[must_use]
    pub fn is_accepting(&self) -> bool {
        self.status == PoolStatus::Active
    }

    /// Get the current number of workers.
    #[must_use]
    pub fn worker_count(&self) -> usize {
        self.workers.len()
    }

    /// Check if the pool has room for another worker.
    #[must_use]
    pub fn has_capacity(&self) -> bool {
        self.max_workers == 0 || self.workers.len() < self.max_workers
    }

    /// Check if the pool has a specific tag.
    #[must_use]
    pub fn has_tag(&self, tag: &str) -> bool {
        self.tags.contains(tag)
    }

    /// Check if the pool has ALL of the required tags.
    #[must_use]
    pub fn has_all_tags(&self, required: &[String]) -> bool {
        required.iter().all(|t| self.tags.contains(t))
    }

    /// Add a worker to this pool.
    ///
    /// # Errors
    ///
    /// Returns `PoolError::WorkerAlreadyInPool` if the worker is already in this pool,
    /// or `PoolError::CapacityExceeded` if the pool is full.
    pub fn add_worker(&mut self, worker_id: impl Into<String>) -> Result<()> {
        let wid = worker_id.into();
        if self.workers.contains(&wid) {
            return Err(PoolError::WorkerAlreadyInPool(wid));
        }
        if !self.has_capacity() {
            return Err(PoolError::CapacityExceeded {
                pool_id: self.id.0.clone(),
                max: self.max_workers,
            });
        }
        self.workers.insert(wid);
        Ok(())
    }

    /// Remove a worker from this pool.
    ///
    /// # Errors
    ///
    /// Returns `PoolError::WorkerNotFound` if the worker is not in this pool.
    pub fn remove_worker(&mut self, worker_id: &str) -> Result<()> {
        if !self.workers.remove(worker_id) {
            return Err(PoolError::WorkerNotFound(worker_id.to_string()));
        }
        Ok(())
    }
}

/// Manages multiple worker pools.
#[derive(Debug, Default)]
pub struct PoolManager {
    /// Map of pool ID to pool.
    pools: HashMap<String, WorkerPool>,
}

impl PoolManager {
    /// Create an empty pool manager.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a new pool.
    ///
    /// # Errors
    ///
    /// Returns `PoolError::PoolAlreadyExists` if a pool with the same ID exists.
    pub fn add_pool(&mut self, pool: WorkerPool) -> Result<()> {
        if self.pools.contains_key(&pool.id.0) {
            return Err(PoolError::PoolAlreadyExists(pool.id.0.clone()));
        }
        self.pools.insert(pool.id.0.clone(), pool);
        Ok(())
    }

    /// Get a pool by ID.
    #[must_use]
    pub fn get_pool(&self, pool_id: &str) -> Option<&WorkerPool> {
        self.pools.get(pool_id)
    }

    /// Get a mutable reference to a pool by ID.
    pub fn get_pool_mut(&mut self, pool_id: &str) -> Option<&mut WorkerPool> {
        self.pools.get_mut(pool_id)
    }

    /// Remove a pool.
    pub fn remove_pool(&mut self, pool_id: &str) -> Option<WorkerPool> {
        self.pools.remove(pool_id)
    }

    /// Get the number of pools.
    #[must_use]
    pub fn pool_count(&self) -> usize {
        self.pools.len()
    }

    /// Find all pools that have the specified tags and are currently accepting.
    #[must_use]
    pub fn find_pools_by_tags(&self, required_tags: &[String]) -> Vec<&WorkerPool> {
        self.pools
            .values()
            .filter(|p| p.is_accepting() && p.has_all_tags(required_tags))
            .collect()
    }

    /// Get the total number of workers across all pools.
    #[must_use]
    pub fn total_workers(&self) -> usize {
        self.pools.values().map(WorkerPool::worker_count).sum()
    }

    /// Set the status of a pool.
    ///
    /// # Errors
    ///
    /// Returns `PoolError::PoolNotFound` if the pool does not exist.
    pub fn set_pool_status(&mut self, pool_id: &str, status: PoolStatus) -> Result<()> {
        let pool = self
            .pools
            .get_mut(pool_id)
            .ok_or_else(|| PoolError::PoolNotFound(pool_id.to_string()))?;
        pool.status = status;
        Ok(())
    }

    /// List all pool IDs.
    pub fn list_pool_ids(&self) -> Vec<&str> {
        self.pools.keys().map(String::as_str).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_creation() {
        let pool = WorkerPool::new("gpu-pool", "GPU Workers");
        assert_eq!(pool.id.as_str(), "gpu-pool");
        assert_eq!(pool.name, "GPU Workers");
        assert_eq!(pool.status, PoolStatus::Active);
        assert_eq!(pool.worker_count(), 0);
    }

    #[test]
    fn test_pool_with_builders() {
        let pool = WorkerPool::new("p1", "Pool 1")
            .with_max_workers(10)
            .with_tag("gpu")
            .with_tag("high-memory")
            .with_priority(200);
        assert_eq!(pool.max_workers, 10);
        assert!(pool.has_tag("gpu"));
        assert!(pool.has_tag("high-memory"));
        assert_eq!(pool.priority_weight, 200);
    }

    #[test]
    fn test_add_remove_worker() {
        let mut pool = WorkerPool::new("p1", "Pool 1").with_max_workers(2);
        pool.add_worker("w1").expect("add_worker should succeed");
        pool.add_worker("w2").expect("add_worker should succeed");
        assert_eq!(pool.worker_count(), 2);
        assert!(!pool.has_capacity());
        pool.remove_worker("w1")
            .expect("remove_worker should succeed");
        assert_eq!(pool.worker_count(), 1);
        assert!(pool.has_capacity());
    }

    #[test]
    fn test_add_worker_duplicate() {
        let mut pool = WorkerPool::new("p1", "Pool 1");
        pool.add_worker("w1").expect("add_worker should succeed");
        let err = pool.add_worker("w1").unwrap_err();
        assert_eq!(err, PoolError::WorkerAlreadyInPool("w1".to_string()));
    }

    #[test]
    fn test_add_worker_capacity_exceeded() {
        let mut pool = WorkerPool::new("p1", "Pool 1").with_max_workers(1);
        pool.add_worker("w1").expect("add_worker should succeed");
        let err = pool.add_worker("w2").unwrap_err();
        assert_eq!(
            err,
            PoolError::CapacityExceeded {
                pool_id: "p1".to_string(),
                max: 1,
            }
        );
    }

    #[test]
    fn test_remove_worker_not_found() {
        let mut pool = WorkerPool::new("p1", "Pool 1");
        let err = pool.remove_worker("w_none").unwrap_err();
        assert_eq!(err, PoolError::WorkerNotFound("w_none".to_string()));
    }

    #[test]
    fn test_pool_status_display() {
        assert_eq!(PoolStatus::Active.to_string(), "Active");
        assert_eq!(PoolStatus::Draining.to_string(), "Draining");
        assert_eq!(PoolStatus::Maintenance.to_string(), "Maintenance");
        assert_eq!(PoolStatus::Disabled.to_string(), "Disabled");
    }

    #[test]
    fn test_pool_accepting() {
        let mut pool = WorkerPool::new("p1", "Pool 1");
        assert!(pool.is_accepting());
        pool.status = PoolStatus::Draining;
        assert!(!pool.is_accepting());
    }

    #[test]
    fn test_has_all_tags() {
        let pool = WorkerPool::new("p1", "Pool 1")
            .with_tag("gpu")
            .with_tag("fast");
        assert!(pool.has_all_tags(&["gpu".to_string(), "fast".to_string()]));
        assert!(!pool.has_all_tags(&["gpu".to_string(), "ssd".to_string()]));
    }

    #[test]
    fn test_pool_manager_add_and_get() {
        let mut mgr = PoolManager::new();
        mgr.add_pool(WorkerPool::new("p1", "Pool 1"))
            .expect("failed to create");
        assert_eq!(mgr.pool_count(), 1);
        assert!(mgr.get_pool("p1").is_some());
        assert!(mgr.get_pool("p2").is_none());
    }

    #[test]
    fn test_pool_manager_duplicate() {
        let mut mgr = PoolManager::new();
        mgr.add_pool(WorkerPool::new("p1", "Pool 1"))
            .expect("failed to create");
        let err = mgr
            .add_pool(WorkerPool::new("p1", "Pool 1 dup"))
            .unwrap_err();
        assert_eq!(err, PoolError::PoolAlreadyExists("p1".to_string()));
    }

    #[test]
    fn test_pool_manager_find_by_tags() {
        let mut mgr = PoolManager::new();
        mgr.add_pool(WorkerPool::new("gpu", "GPU").with_tag("gpu"))
            .expect("operation should succeed");
        mgr.add_pool(WorkerPool::new("cpu", "CPU").with_tag("cpu"))
            .expect("operation should succeed");
        let found = mgr.find_pools_by_tags(&["gpu".to_string()]);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].id.as_str(), "gpu");
    }

    #[test]
    fn test_pool_manager_total_workers() {
        let mut mgr = PoolManager::new();
        let mut p1 = WorkerPool::new("p1", "P1");
        p1.add_worker("w1").expect("add_worker should succeed");
        p1.add_worker("w2").expect("add_worker should succeed");
        let mut p2 = WorkerPool::new("p2", "P2");
        p2.add_worker("w3").expect("add_worker should succeed");
        mgr.add_pool(p1).expect("add_pool should succeed");
        mgr.add_pool(p2).expect("add_pool should succeed");
        assert_eq!(mgr.total_workers(), 3);
    }

    #[test]
    fn test_pool_manager_set_status() {
        let mut mgr = PoolManager::new();
        mgr.add_pool(WorkerPool::new("p1", "P1"))
            .expect("failed to create");
        mgr.set_pool_status("p1", PoolStatus::Draining)
            .expect("set_pool_status should succeed");
        assert_eq!(
            mgr.get_pool("p1").expect("get_pool should succeed").status,
            PoolStatus::Draining
        );
    }

    #[test]
    fn test_pool_error_display() {
        let err = PoolError::PoolNotFound("missing".to_string());
        assert_eq!(err.to_string(), "pool not found: missing");
    }
}
