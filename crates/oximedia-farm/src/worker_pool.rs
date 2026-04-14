#![allow(dead_code)]
//! Worker pool management with grouping and tagging.
//!
//! Provides a logical grouping layer on top of individual workers:
//! - Worker groups (pools) with shared properties and tags
//! - Pool-level capacity tracking and utilization metrics
//! - Worker assignment and removal from pools
//! - Pool-based job routing (match job requirements to pool capabilities)
//! - Drain and maintenance mode for individual pools
//!
//! # Auto-scaling
//!
//! `PoolManager` supports queue-depth-driven auto-scaling via
//! [`PoolManager::evaluate_scaling`] and [`PoolManager::apply_scale_decision`].
//! A [`PoolAutoScaleConfig`] per pool specifies the thresholds, step sizes, and
//! a cooldown period to prevent oscillation.

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

// ── Auto-scaling ──────────────────────────────────────────────────────────────

/// Configuration for queue-depth-driven auto-scaling of a single pool.
///
/// The scaling decision is based on the ratio of pending queue depth to the
/// current number of active workers.  If the ratio exceeds `scale_up_threshold`,
/// the pool should grow; if it falls below `scale_down_threshold`, the pool
/// should shrink.
#[derive(Debug, Clone, PartialEq)]
pub struct PoolAutoScaleConfig {
    /// Minimum number of workers this pool should maintain.
    pub min_workers: usize,
    /// Maximum number of workers this pool may ever hold (`0` = no cap).
    pub max_workers: usize,
    /// Queue-depth-per-worker ratio above which a scale-up is recommended.
    pub scale_up_threshold: f64,
    /// Queue-depth-per-worker ratio below which a scale-down is recommended.
    pub scale_down_threshold: f64,
    /// Number of worker slots to add per scale-up action.
    pub scale_up_step: usize,
    /// Number of worker slots to remove per scale-down action.
    pub scale_down_step: usize,
    /// Minimum seconds between consecutive scale actions (prevents oscillation).
    pub cooldown_secs: u64,
}

impl PoolAutoScaleConfig {
    /// Sensible defaults: min=1, max=unlimited, thresholds 2.0/0.5, step=1, cooldown=60 s.
    #[must_use]
    pub fn default_config() -> Self {
        Self {
            min_workers: 1,
            max_workers: 0,
            scale_up_threshold: 2.0,
            scale_down_threshold: 0.5,
            scale_up_step: 1,
            scale_down_step: 1,
            cooldown_secs: 60,
        }
    }
}

impl Default for PoolAutoScaleConfig {
    fn default() -> Self {
        Self::default_config()
    }
}

/// The scaling action recommended by [`PoolManager::evaluate_scaling`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolScaleDecision {
    /// No change needed; the pool is appropriately sized.
    NoChange,
    /// Increase pool capacity by `n` worker slots.
    ScaleUp(usize),
    /// Decrease pool capacity by `n` worker slots.
    ScaleDown(usize),
}

/// Per-pool auto-scale runtime state (cooldown tracking).
#[derive(Debug, Default, Clone)]
struct AutoScaleState {
    /// Epoch-seconds timestamp of the last scale action (0 = never scaled).
    last_action_epoch: u64,
}

/// Manages multiple worker pools.
#[derive(Debug, Default)]
pub struct PoolManager {
    /// Map of pool ID to pool.
    pools: HashMap<String, WorkerPool>,
    /// Per-pool auto-scale configuration.
    autoscale_configs: HashMap<String, PoolAutoScaleConfig>,
    /// Per-pool auto-scale runtime state.
    autoscale_state: HashMap<String, AutoScaleState>,
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
        self.autoscale_configs.remove(pool_id);
        self.autoscale_state.remove(pool_id);
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

    // ── Auto-scaling ──────────────────────────────────────────────────────────

    /// Attach an auto-scale configuration to a pool.
    ///
    /// # Errors
    ///
    /// Returns `PoolError::PoolNotFound` if the pool does not exist.
    pub fn set_autoscale_config(
        &mut self,
        pool_id: &str,
        config: PoolAutoScaleConfig,
    ) -> Result<()> {
        if !self.pools.contains_key(pool_id) {
            return Err(PoolError::PoolNotFound(pool_id.to_string()));
        }
        self.autoscale_configs.insert(pool_id.to_string(), config);
        self.autoscale_state.entry(pool_id.to_string()).or_default();
        Ok(())
    }

    /// Evaluate whether a pool should scale up or down given the current
    /// `queue_depth` (number of pending jobs) and `now_secs` (epoch seconds).
    ///
    /// Returns [`PoolScaleDecision::NoChange`] if:
    /// - the pool has no auto-scale config,
    /// - the pool does not exist,
    /// - the cooldown period has not elapsed since the last scale action, or
    /// - the current worker count already satisfies the thresholds.
    ///
    /// The decision is based solely on the *current* state; callers must invoke
    /// [`Self::apply_scale_decision`] to actually mutate the pool.
    #[must_use]
    pub fn evaluate_scaling(
        &self,
        pool_id: &str,
        queue_depth: usize,
        now_secs: u64,
    ) -> PoolScaleDecision {
        let config = match self.autoscale_configs.get(pool_id) {
            Some(c) => c,
            None => return PoolScaleDecision::NoChange,
        };
        let pool = match self.pools.get(pool_id) {
            Some(p) => p,
            None => return PoolScaleDecision::NoChange,
        };
        let state = match self.autoscale_state.get(pool_id) {
            Some(s) => s,
            None => return PoolScaleDecision::NoChange,
        };

        // Enforce cooldown.
        if state.last_action_epoch > 0
            && now_secs.saturating_sub(state.last_action_epoch) < config.cooldown_secs
        {
            return PoolScaleDecision::NoChange;
        }

        let worker_count = pool.worker_count();

        // Avoid division by zero: treat 0 workers as 1 for ratio calculation so
        // that a non-zero queue depth always triggers a scale-up.
        let effective_workers = worker_count.max(1) as f64;
        let ratio = queue_depth as f64 / effective_workers;

        if ratio > config.scale_up_threshold {
            // Do not exceed max_workers (if capped).
            let headroom = if config.max_workers == 0 {
                config.scale_up_step
            } else {
                config.max_workers.saturating_sub(worker_count)
            };
            if headroom == 0 {
                return PoolScaleDecision::NoChange;
            }
            let step = config.scale_up_step.min(headroom);
            PoolScaleDecision::ScaleUp(step)
        } else if ratio < config.scale_down_threshold {
            // Do not go below min_workers.
            let removable = worker_count.saturating_sub(config.min_workers);
            if removable == 0 {
                return PoolScaleDecision::NoChange;
            }
            let step = config.scale_down_step.min(removable);
            PoolScaleDecision::ScaleDown(step)
        } else {
            PoolScaleDecision::NoChange
        }
    }

    /// Apply a previously evaluated `PoolScaleDecision` to the pool.
    ///
    /// - `ScaleUp(n)` — adds `n` placeholder worker IDs (`autoscale-<pool>-<seq>`) to
    ///   the pool's worker set and updates the pool's `max_workers` limit if needed.
    /// - `ScaleDown(n)` — transitions `n` workers to Draining by removing them from
    ///   the worker set (the actual worker processes are responsible for graceful
    ///   shutdown; this only updates the logical pool accounting).
    /// - `NoChange` — no-op.
    ///
    /// Records the current time for cooldown enforcement.
    ///
    /// # Errors
    ///
    /// Returns `PoolError::PoolNotFound` if the pool does not exist.
    pub fn apply_scale_decision(
        &mut self,
        pool_id: &str,
        decision: &PoolScaleDecision,
        now_secs: u64,
    ) -> Result<()> {
        if !self.pools.contains_key(pool_id) {
            return Err(PoolError::PoolNotFound(pool_id.to_string()));
        }

        match decision {
            PoolScaleDecision::NoChange => {}
            PoolScaleDecision::ScaleUp(n) => {
                let pool = self
                    .pools
                    .get_mut(pool_id)
                    .ok_or_else(|| PoolError::PoolNotFound(pool_id.to_string()))?;
                let start_seq = pool.workers.len();
                for i in 0..*n {
                    let wid = format!("autoscale-{pool_id}-{}", start_seq + i);
                    // Temporarily raise max_workers if the pool is capped and the
                    // auto-scale config explicitly asks for more slots.
                    if pool.max_workers > 0 && pool.workers.len() >= pool.max_workers {
                        pool.max_workers += 1;
                    }
                    // Best-effort: ignore duplicate-worker errors from repeated calls.
                    let _ = pool.add_worker(&wid);
                }
                self.autoscale_state
                    .entry(pool_id.to_string())
                    .or_default()
                    .last_action_epoch = now_secs;
            }
            PoolScaleDecision::ScaleDown(n) => {
                let pool = self
                    .pools
                    .get_mut(pool_id)
                    .ok_or_else(|| PoolError::PoolNotFound(pool_id.to_string()))?;
                // Collect worker IDs to drain — take the last `n` from a sorted list
                // so the behaviour is deterministic in tests.
                let mut worker_ids: Vec<String> = pool.workers.iter().cloned().collect();
                worker_ids.sort();
                let drain_start = worker_ids.len().saturating_sub(*n);
                let to_drain: Vec<String> = worker_ids.drain(drain_start..).collect();
                for wid in to_drain {
                    let _ = pool.remove_worker(&wid);
                }
                self.autoscale_state
                    .entry(pool_id.to_string())
                    .or_default()
                    .last_action_epoch = now_secs;
            }
        }

        Ok(())
    }

    /// Return the auto-scale config for a pool, if any.
    #[must_use]
    pub fn autoscale_config(&self, pool_id: &str) -> Option<&PoolAutoScaleConfig> {
        self.autoscale_configs.get(pool_id)
    }

    /// Return the epoch-seconds timestamp of the last scale action for a pool.
    /// Returns `0` if the pool has never been scaled or has no config.
    #[must_use]
    pub fn last_scale_action_epoch(&self, pool_id: &str) -> u64 {
        self.autoscale_state
            .get(pool_id)
            .map(|s| s.last_action_epoch)
            .unwrap_or(0)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // --- WorkerPool basic tests ---

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
        pool.add_worker("w1").expect("add_worker failed");
        pool.add_worker("w2").expect("add_worker failed");
        assert_eq!(pool.worker_count(), 2);
        assert!(!pool.has_capacity());
        pool.remove_worker("w1").expect("remove_worker failed");
        assert_eq!(pool.worker_count(), 1);
        assert!(pool.has_capacity());
    }

    #[test]
    fn test_add_worker_duplicate() {
        let mut pool = WorkerPool::new("p1", "Pool 1");
        pool.add_worker("w1").expect("add_worker failed");
        let err = pool.add_worker("w1").expect_err("expected error");
        assert_eq!(err, PoolError::WorkerAlreadyInPool("w1".to_string()));
    }

    #[test]
    fn test_add_worker_capacity_exceeded() {
        let mut pool = WorkerPool::new("p1", "Pool 1").with_max_workers(1);
        pool.add_worker("w1").expect("add_worker failed");
        let err = pool.add_worker("w2").expect_err("expected error");
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
        let err = pool.remove_worker("w_none").expect_err("expected error");
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

    // --- PoolManager basic tests ---

    #[test]
    fn test_pool_manager_add_and_get() {
        let mut mgr = PoolManager::new();
        mgr.add_pool(WorkerPool::new("p1", "Pool 1"))
            .expect("add_pool failed");
        assert_eq!(mgr.pool_count(), 1);
        assert!(mgr.get_pool("p1").is_some());
        assert!(mgr.get_pool("p2").is_none());
    }

    #[test]
    fn test_pool_manager_duplicate() {
        let mut mgr = PoolManager::new();
        mgr.add_pool(WorkerPool::new("p1", "Pool 1"))
            .expect("add_pool failed");
        let err = mgr
            .add_pool(WorkerPool::new("p1", "Pool 1 dup"))
            .expect_err("expected error");
        assert_eq!(err, PoolError::PoolAlreadyExists("p1".to_string()));
    }

    #[test]
    fn test_pool_manager_find_by_tags() {
        let mut mgr = PoolManager::new();
        mgr.add_pool(WorkerPool::new("gpu", "GPU").with_tag("gpu"))
            .expect("add_pool failed");
        mgr.add_pool(WorkerPool::new("cpu", "CPU").with_tag("cpu"))
            .expect("add_pool failed");
        let found = mgr.find_pools_by_tags(&["gpu".to_string()]);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].id.as_str(), "gpu");
    }

    #[test]
    fn test_pool_manager_total_workers() {
        let mut mgr = PoolManager::new();
        let mut p1 = WorkerPool::new("p1", "P1");
        p1.add_worker("w1").expect("add_worker failed");
        p1.add_worker("w2").expect("add_worker failed");
        let mut p2 = WorkerPool::new("p2", "P2");
        p2.add_worker("w3").expect("add_worker failed");
        mgr.add_pool(p1).expect("add_pool failed");
        mgr.add_pool(p2).expect("add_pool failed");
        assert_eq!(mgr.total_workers(), 3);
    }

    #[test]
    fn test_pool_manager_set_status() {
        let mut mgr = PoolManager::new();
        mgr.add_pool(WorkerPool::new("p1", "P1"))
            .expect("add_pool failed");
        mgr.set_pool_status("p1", PoolStatus::Draining)
            .expect("set_pool_status failed");
        assert_eq!(
            mgr.get_pool("p1").expect("pool not found").status,
            PoolStatus::Draining
        );
    }

    #[test]
    fn test_pool_error_display() {
        let err = PoolError::PoolNotFound("missing".to_string());
        assert_eq!(err.to_string(), "pool not found: missing");
    }

    // --- Auto-scaling tests ---

    fn make_pool_with_workers(id: &str, worker_count: usize) -> WorkerPool {
        let mut p = WorkerPool::new(id, id);
        for i in 0..worker_count {
            p.add_worker(format!("w{i}")).expect("add_worker failed");
        }
        p
    }

    #[test]
    fn test_autoscale_no_config_returns_no_change() {
        let mut mgr = PoolManager::new();
        mgr.add_pool(make_pool_with_workers("p1", 2))
            .expect("add_pool failed");
        // No auto-scale config set.
        let decision = mgr.evaluate_scaling("p1", 10, 1000);
        assert_eq!(decision, PoolScaleDecision::NoChange);
    }

    #[test]
    fn test_autoscale_scale_up_when_ratio_exceeds_threshold() {
        let mut mgr = PoolManager::new();
        mgr.add_pool(make_pool_with_workers("p1", 2))
            .expect("add_pool failed");
        let config = PoolAutoScaleConfig {
            min_workers: 1,
            max_workers: 0,
            scale_up_threshold: 2.0,
            scale_down_threshold: 0.5,
            scale_up_step: 1,
            scale_down_step: 1,
            cooldown_secs: 0,
        };
        mgr.set_autoscale_config("p1", config)
            .expect("set_autoscale_config failed");
        // 2 workers, 10 queued → ratio=5.0 > 2.0 → ScaleUp
        let decision = mgr.evaluate_scaling("p1", 10, 0);
        assert_eq!(decision, PoolScaleDecision::ScaleUp(1));
    }

    #[test]
    fn test_autoscale_scale_down_when_ratio_below_threshold() {
        let mut mgr = PoolManager::new();
        mgr.add_pool(make_pool_with_workers("p1", 4))
            .expect("add_pool failed");
        let config = PoolAutoScaleConfig {
            min_workers: 1,
            max_workers: 0,
            scale_up_threshold: 2.0,
            scale_down_threshold: 0.5,
            scale_up_step: 1,
            scale_down_step: 1,
            cooldown_secs: 0,
        };
        mgr.set_autoscale_config("p1", config)
            .expect("set_autoscale_config failed");
        // 4 workers, 1 queued → ratio=0.25 < 0.5 → ScaleDown
        let decision = mgr.evaluate_scaling("p1", 1, 0);
        assert_eq!(decision, PoolScaleDecision::ScaleDown(1));
    }

    #[test]
    fn test_autoscale_no_change_within_band() {
        let mut mgr = PoolManager::new();
        mgr.add_pool(make_pool_with_workers("p1", 4))
            .expect("add_pool failed");
        let config = PoolAutoScaleConfig::default_config();
        mgr.set_autoscale_config("p1", config)
            .expect("set_autoscale_config failed");
        // 4 workers, 6 queued → ratio=1.5; 0.5 < 1.5 < 2.0 → NoChange
        let decision = mgr.evaluate_scaling("p1", 6, 0);
        assert_eq!(decision, PoolScaleDecision::NoChange);
    }

    #[test]
    fn test_autoscale_cooldown_prevents_rapid_scaling() {
        let mut mgr = PoolManager::new();
        mgr.add_pool(make_pool_with_workers("p1", 2))
            .expect("add_pool failed");
        let config = PoolAutoScaleConfig {
            cooldown_secs: 60,
            ..PoolAutoScaleConfig::default_config()
        };
        mgr.set_autoscale_config("p1", config)
            .expect("set_autoscale_config failed");
        // First scale action at t=100.
        let d = mgr.evaluate_scaling("p1", 20, 100);
        assert_eq!(d, PoolScaleDecision::ScaleUp(1));
        mgr.apply_scale_decision("p1", &d, 100)
            .expect("apply_scale_decision failed");
        // At t=140 (only 40 s later, inside the 60-s cooldown) → NoChange.
        let d2 = mgr.evaluate_scaling("p1", 20, 140);
        assert_eq!(d2, PoolScaleDecision::NoChange);
        // At t=161 (cooldown elapsed) → ScaleUp again.
        let d3 = mgr.evaluate_scaling("p1", 20, 161);
        assert_eq!(d3, PoolScaleDecision::ScaleUp(1));
    }

    #[test]
    fn test_autoscale_apply_scale_up_adds_workers() {
        let mut mgr = PoolManager::new();
        mgr.add_pool(make_pool_with_workers("p1", 2))
            .expect("add_pool failed");
        let config = PoolAutoScaleConfig {
            scale_up_step: 2,
            cooldown_secs: 0,
            ..PoolAutoScaleConfig::default_config()
        };
        mgr.set_autoscale_config("p1", config)
            .expect("set_autoscale_config failed");
        let before = mgr.get_pool("p1").expect("pool not found").worker_count();
        mgr.apply_scale_decision("p1", &PoolScaleDecision::ScaleUp(2), 0)
            .expect("apply_scale_decision failed");
        let after = mgr.get_pool("p1").expect("pool not found").worker_count();
        assert_eq!(after, before + 2);
    }

    #[test]
    fn test_autoscale_apply_scale_down_removes_workers() {
        let mut mgr = PoolManager::new();
        mgr.add_pool(make_pool_with_workers("p1", 4))
            .expect("add_pool failed");
        let config = PoolAutoScaleConfig::default_config();
        mgr.set_autoscale_config("p1", config)
            .expect("set_autoscale_config failed");
        let before = mgr.get_pool("p1").expect("pool not found").worker_count();
        mgr.apply_scale_decision("p1", &PoolScaleDecision::ScaleDown(2), 0)
            .expect("apply_scale_decision failed");
        let after = mgr.get_pool("p1").expect("pool not found").worker_count();
        assert_eq!(after, before - 2);
    }

    #[test]
    fn test_autoscale_respects_min_workers() {
        let mut mgr = PoolManager::new();
        mgr.add_pool(make_pool_with_workers("p1", 2))
            .expect("add_pool failed");
        let config = PoolAutoScaleConfig {
            min_workers: 2,
            scale_down_threshold: 0.5,
            scale_down_step: 1,
            cooldown_secs: 0,
            ..PoolAutoScaleConfig::default_config()
        };
        mgr.set_autoscale_config("p1", config)
            .expect("set_autoscale_config failed");
        // 2 workers, 0 queued → ratio=0 < 0.5, but min=2, removable=0 → NoChange.
        let decision = mgr.evaluate_scaling("p1", 0, 0);
        assert_eq!(decision, PoolScaleDecision::NoChange);
    }

    #[test]
    fn test_autoscale_respects_max_workers() {
        let mut mgr = PoolManager::new();
        mgr.add_pool(make_pool_with_workers("p1", 3))
            .expect("add_pool failed");
        let config = PoolAutoScaleConfig {
            max_workers: 3, // already at max
            scale_up_threshold: 1.0,
            scale_up_step: 1,
            cooldown_secs: 0,
            ..PoolAutoScaleConfig::default_config()
        };
        mgr.set_autoscale_config("p1", config)
            .expect("set_autoscale_config failed");
        // Ratio is high, but pool is at max_workers → NoChange.
        let decision = mgr.evaluate_scaling("p1", 100, 0);
        assert_eq!(decision, PoolScaleDecision::NoChange);
    }

    #[test]
    fn test_autoscale_last_scale_epoch_updated() {
        let mut mgr = PoolManager::new();
        mgr.add_pool(make_pool_with_workers("p1", 2))
            .expect("add_pool failed");
        let config = PoolAutoScaleConfig::default_config();
        mgr.set_autoscale_config("p1", config)
            .expect("set_autoscale_config failed");
        assert_eq!(mgr.last_scale_action_epoch("p1"), 0);
        mgr.apply_scale_decision("p1", &PoolScaleDecision::ScaleUp(1), 9999)
            .expect("apply_scale_decision failed");
        assert_eq!(mgr.last_scale_action_epoch("p1"), 9999);
    }

    #[test]
    fn test_autoscale_no_change_decision_does_not_update_epoch() {
        let mut mgr = PoolManager::new();
        mgr.add_pool(make_pool_with_workers("p1", 2))
            .expect("add_pool failed");
        let config = PoolAutoScaleConfig::default_config();
        mgr.set_autoscale_config("p1", config)
            .expect("set_autoscale_config failed");
        mgr.apply_scale_decision("p1", &PoolScaleDecision::NoChange, 5000)
            .expect("apply_scale_decision failed");
        assert_eq!(mgr.last_scale_action_epoch("p1"), 0);
    }
}
