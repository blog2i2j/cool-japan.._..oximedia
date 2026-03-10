#![allow(dead_code)]
//! GPU fence pool for efficient synchronization primitive reuse.
//!
//! This module provides a pooled allocation strategy for GPU fences,
//! reducing the overhead of creating and destroying fence objects
//! on every frame submission.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Unique identifier for a pooled fence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FenceId(
    /// Inner identifier value.
    pub u64,
);

/// Current status of a fence in the pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FenceStatus {
    /// Fence is available for reuse.
    Available,
    /// Fence has been submitted and is pending GPU completion.
    Pending,
    /// Fence has been signaled by the GPU.
    Signaled,
    /// Fence encountered an error.
    Error,
}

impl std::fmt::Display for FenceStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Available => write!(f, "Available"),
            Self::Pending => write!(f, "Pending"),
            Self::Signaled => write!(f, "Signaled"),
            Self::Error => write!(f, "Error"),
        }
    }
}

/// A single fence entry in the pool.
#[derive(Debug, Clone)]
pub struct PooledFence {
    /// Unique identifier.
    pub id: FenceId,
    /// Current status of this fence.
    pub status: FenceStatus,
    /// Timestamp when the fence was submitted.
    pub submit_time: Option<Instant>,
    /// Timestamp when the fence was signaled.
    pub signal_time: Option<Instant>,
    /// Number of times this fence has been recycled.
    pub recycle_count: u64,
}

impl PooledFence {
    /// Create a new pooled fence with the given ID.
    #[must_use]
    pub fn new(id: FenceId) -> Self {
        Self {
            id,
            status: FenceStatus::Available,
            submit_time: None,
            signal_time: None,
            recycle_count: 0,
        }
    }

    /// Return the latency between submit and signal, if both are recorded.
    #[must_use]
    pub fn latency(&self) -> Option<Duration> {
        match (self.submit_time, self.signal_time) {
            (Some(submit), Some(signal)) => Some(signal.duration_since(submit)),
            _ => None,
        }
    }

    /// Check if the fence is currently available for reuse.
    #[must_use]
    pub fn is_available(&self) -> bool {
        self.status == FenceStatus::Available
    }

    /// Check if the fence is pending GPU completion.
    #[must_use]
    pub fn is_pending(&self) -> bool {
        self.status == FenceStatus::Pending
    }
}

/// Configuration for the fence pool.
#[derive(Debug, Clone)]
pub struct FencePoolConfig {
    /// Initial number of fences to pre-allocate.
    pub initial_size: usize,
    /// Maximum number of fences allowed in the pool.
    pub max_size: usize,
    /// Timeout for fence wait operations.
    pub wait_timeout: Duration,
    /// Whether to auto-grow the pool when exhausted.
    pub auto_grow: bool,
    /// Batch size for growing the pool.
    pub grow_batch_size: usize,
}

impl Default for FencePoolConfig {
    fn default() -> Self {
        Self {
            initial_size: 16,
            max_size: 256,
            wait_timeout: Duration::from_secs(5),
            auto_grow: true,
            grow_batch_size: 8,
        }
    }
}

/// Statistics about fence pool usage.
#[derive(Debug, Clone, Default)]
pub struct FencePoolStats {
    /// Total number of fences in the pool.
    pub total_fences: usize,
    /// Number of fences currently available.
    pub available_count: usize,
    /// Number of fences currently pending.
    pub pending_count: usize,
    /// Number of fences currently signaled.
    pub signaled_count: usize,
    /// Total number of allocations performed.
    pub total_allocations: u64,
    /// Total number of recycles performed.
    pub total_recycles: u64,
    /// Number of times the pool had to grow.
    pub grow_events: u64,
    /// Average latency of signaled fences in microseconds.
    pub avg_latency_us: f64,
}

impl FencePoolStats {
    /// Return the utilization ratio of the pool (pending / total).
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn utilization(&self) -> f64 {
        if self.total_fences == 0 {
            return 0.0;
        }
        self.pending_count as f64 / self.total_fences as f64
    }
}

/// A pool of reusable GPU fence objects.
///
/// Manages fence lifecycle to avoid repeated allocation/deallocation overhead.
pub struct FencePool {
    /// All fences in the pool.
    fences: Vec<PooledFence>,
    /// Queue of available fence indices.
    available: VecDeque<usize>,
    /// Configuration for this pool.
    config: FencePoolConfig,
    /// Counter for generating unique fence IDs.
    next_id: AtomicU64,
    /// Total allocations performed.
    total_allocations: u64,
    /// Total recycles performed.
    total_recycles: u64,
    /// Number of grow events.
    grow_events: u64,
}

impl FencePool {
    /// Create a new fence pool with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(FencePoolConfig::default())
    }

    /// Create a new fence pool with the given configuration.
    #[must_use]
    pub fn with_config(config: FencePoolConfig) -> Self {
        let mut pool = Self {
            fences: Vec::with_capacity(config.initial_size),
            available: VecDeque::with_capacity(config.initial_size),
            next_id: AtomicU64::new(0),
            total_allocations: 0,
            total_recycles: 0,
            grow_events: 0,
            config,
        };
        let initial = pool.config.initial_size;
        pool.grow(initial);
        pool
    }

    /// Grow the pool by the specified number of fences.
    fn grow(&mut self, count: usize) {
        let max = self.config.max_size;
        let current = self.fences.len();
        let actual_count = count.min(max.saturating_sub(current));
        for _ in 0..actual_count {
            let id = FenceId(self.next_id.fetch_add(1, Ordering::Relaxed));
            let fence = PooledFence::new(id);
            let index = self.fences.len();
            self.fences.push(fence);
            self.available.push_back(index);
        }
        if actual_count > 0 {
            self.grow_events += 1;
        }
    }

    /// Acquire a fence from the pool. Returns `None` if no fences are available
    /// and auto-grow is disabled or the pool is at maximum size.
    pub fn acquire(&mut self) -> Option<FenceId> {
        if self.available.is_empty() && self.config.auto_grow {
            let batch = self.config.grow_batch_size;
            self.grow(batch);
        }
        let index = self.available.pop_front()?;
        let fence = &mut self.fences[index];
        fence.status = FenceStatus::Pending;
        fence.submit_time = Some(Instant::now());
        fence.signal_time = None;
        self.total_allocations += 1;
        Some(fence.id)
    }

    /// Signal that a fence has completed on the GPU.
    pub fn signal(&mut self, id: FenceId) -> bool {
        if let Some(fence) = self.fences.iter_mut().find(|f| f.id == id) {
            fence.status = FenceStatus::Signaled;
            fence.signal_time = Some(Instant::now());
            true
        } else {
            false
        }
    }

    /// Release a fence back to the pool for reuse.
    pub fn release(&mut self, id: FenceId) -> bool {
        if let Some((index, fence)) = self.fences.iter_mut().enumerate().find(|(_, f)| f.id == id) {
            fence.status = FenceStatus::Available;
            fence.submit_time = None;
            fence.signal_time = None;
            fence.recycle_count += 1;
            self.available.push_back(index);
            self.total_recycles += 1;
            true
        } else {
            false
        }
    }

    /// Get the current status of a fence.
    pub fn status(&self, id: FenceId) -> Option<FenceStatus> {
        self.fences.iter().find(|f| f.id == id).map(|f| f.status)
    }

    /// Return all currently pending fence IDs.
    pub fn pending_fences(&self) -> Vec<FenceId> {
        self.fences
            .iter()
            .filter(|f| f.status == FenceStatus::Pending)
            .map(|f| f.id)
            .collect()
    }

    /// Signal all pending fences and release them back to the pool.
    pub fn flush_all(&mut self) {
        let pending_ids: Vec<FenceId> = self.pending_fences();
        for id in &pending_ids {
            self.signal(*id);
        }
        let signaled_ids: Vec<FenceId> = self
            .fences
            .iter()
            .filter(|f| f.status == FenceStatus::Signaled)
            .map(|f| f.id)
            .collect();
        for id in signaled_ids {
            self.release(id);
        }
    }

    /// Return the total number of fences in the pool.
    pub fn total_count(&self) -> usize {
        self.fences.len()
    }

    /// Return the number of available fences.
    pub fn available_count(&self) -> usize {
        self.available.len()
    }

    /// Compute pool statistics.
    #[allow(clippy::cast_precision_loss)]
    pub fn stats(&self) -> FencePoolStats {
        let pending_count = self
            .fences
            .iter()
            .filter(|f| f.status == FenceStatus::Pending)
            .count();
        let signaled_count = self
            .fences
            .iter()
            .filter(|f| f.status == FenceStatus::Signaled)
            .count();

        let latencies: Vec<f64> = self
            .fences
            .iter()
            .filter_map(PooledFence::latency)
            .map(|d| d.as_micros() as f64)
            .collect();
        let avg_latency_us = if latencies.is_empty() {
            0.0
        } else {
            latencies.iter().sum::<f64>() / latencies.len() as f64
        };

        FencePoolStats {
            total_fences: self.fences.len(),
            available_count: self.available.len(),
            pending_count,
            signaled_count,
            total_allocations: self.total_allocations,
            total_recycles: self.total_recycles,
            grow_events: self.grow_events,
            avg_latency_us,
        }
    }

    /// Reset the pool, marking all fences as available.
    pub fn reset(&mut self) {
        self.available.clear();
        for (i, fence) in self.fences.iter_mut().enumerate() {
            fence.status = FenceStatus::Available;
            fence.submit_time = None;
            fence.signal_time = None;
            self.available.push_back(i);
        }
    }

    /// Return the wait timeout configured for this pool.
    pub fn wait_timeout(&self) -> Duration {
        self.config.wait_timeout
    }

    /// Check whether the pool has fences available without growing.
    pub fn has_available(&self) -> bool {
        !self.available.is_empty()
    }
}

impl Default for FencePool {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_default_pool() {
        let pool = FencePool::new();
        assert_eq!(pool.total_count(), 16);
        assert_eq!(pool.available_count(), 16);
    }

    #[test]
    fn test_create_with_config() {
        let config = FencePoolConfig {
            initial_size: 4,
            max_size: 32,
            auto_grow: true,
            grow_batch_size: 4,
            ..Default::default()
        };
        let pool = FencePool::with_config(config);
        assert_eq!(pool.total_count(), 4);
        assert_eq!(pool.available_count(), 4);
    }

    #[test]
    fn test_acquire_fence() {
        let mut pool = FencePool::new();
        let id = pool.acquire();
        assert!(id.is_some());
        assert_eq!(pool.available_count(), 15);
    }

    #[test]
    fn test_signal_fence() {
        let mut pool = FencePool::new();
        let id = pool.acquire().expect("fence acquire should succeed");
        assert!(pool.signal(id));
        assert_eq!(pool.status(id), Some(FenceStatus::Signaled));
    }

    #[test]
    fn test_release_fence() {
        let mut pool = FencePool::new();
        let initial_available = pool.available_count();
        let id = pool.acquire().expect("fence acquire should succeed");
        assert_eq!(pool.available_count(), initial_available - 1);
        pool.signal(id);
        pool.release(id);
        assert_eq!(pool.available_count(), initial_available);
        assert_eq!(pool.status(id), Some(FenceStatus::Available));
    }

    #[test]
    fn test_pending_fences() {
        let mut pool = FencePool::new();
        let id1 = pool.acquire().expect("fence acquire should succeed");
        let id2 = pool.acquire().expect("fence acquire should succeed");
        let pending = pool.pending_fences();
        assert_eq!(pending.len(), 2);
        assert!(pending.contains(&id1));
        assert!(pending.contains(&id2));
    }

    #[test]
    fn test_flush_all() {
        let mut pool = FencePool::new();
        let _id1 = pool.acquire().expect("fence acquire should succeed");
        let _id2 = pool.acquire().expect("fence acquire should succeed");
        assert_eq!(pool.pending_fences().len(), 2);
        pool.flush_all();
        assert_eq!(pool.pending_fences().len(), 0);
        assert_eq!(pool.available_count(), 16);
    }

    #[test]
    fn test_auto_grow() {
        let config = FencePoolConfig {
            initial_size: 2,
            max_size: 10,
            auto_grow: true,
            grow_batch_size: 3,
            ..Default::default()
        };
        let mut pool = FencePool::with_config(config);
        let _id1 = pool.acquire().expect("fence acquire should succeed");
        let _id2 = pool.acquire().expect("fence acquire should succeed");
        // Pool exhausted, should auto-grow
        let id3 = pool.acquire();
        assert!(id3.is_some());
        assert!(pool.total_count() > 2);
    }

    #[test]
    fn test_max_size_limit() {
        let config = FencePoolConfig {
            initial_size: 2,
            max_size: 3,
            auto_grow: true,
            grow_batch_size: 10,
            ..Default::default()
        };
        let mut pool = FencePool::with_config(config);
        let _id1 = pool.acquire().expect("fence acquire should succeed");
        let _id2 = pool.acquire().expect("fence acquire should succeed");
        let _id3 = pool.acquire();
        // Should not exceed max
        assert!(pool.total_count() <= 3);
    }

    #[test]
    fn test_no_auto_grow() {
        let config = FencePoolConfig {
            initial_size: 1,
            max_size: 10,
            auto_grow: false,
            ..Default::default()
        };
        let mut pool = FencePool::with_config(config);
        let _id = pool.acquire().expect("fence acquire should succeed");
        let id2 = pool.acquire();
        assert!(id2.is_none());
    }

    #[test]
    fn test_stats() {
        let mut pool = FencePool::new();
        let id1 = pool.acquire().expect("fence acquire should succeed");
        let _id2 = pool.acquire().expect("fence acquire should succeed");
        pool.signal(id1);
        let stats = pool.stats();
        assert_eq!(stats.total_fences, 16);
        assert_eq!(stats.pending_count, 1);
        assert_eq!(stats.signaled_count, 1);
        assert_eq!(stats.total_allocations, 2);
    }

    #[test]
    fn test_stats_utilization() {
        let stats = FencePoolStats {
            total_fences: 10,
            pending_count: 5,
            ..Default::default()
        };
        let util = stats.utilization();
        assert!((util - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_stats_utilization_empty() {
        let stats = FencePoolStats::default();
        assert!((stats.utilization() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_reset_pool() {
        let mut pool = FencePool::new();
        let _id1 = pool.acquire().expect("fence acquire should succeed");
        let _id2 = pool.acquire().expect("fence acquire should succeed");
        pool.reset();
        assert_eq!(pool.available_count(), pool.total_count());
    }

    #[test]
    fn test_fence_display() {
        assert_eq!(format!("{}", FenceStatus::Available), "Available");
        assert_eq!(format!("{}", FenceStatus::Pending), "Pending");
        assert_eq!(format!("{}", FenceStatus::Signaled), "Signaled");
        assert_eq!(format!("{}", FenceStatus::Error), "Error");
    }

    #[test]
    fn test_pooled_fence_latency() {
        let mut fence = PooledFence::new(FenceId(0));
        assert!(fence.latency().is_none());
        fence.submit_time = Some(Instant::now());
        assert!(fence.latency().is_none());
        fence.signal_time = Some(Instant::now());
        assert!(fence.latency().is_some());
    }

    #[test]
    fn test_wait_timeout() {
        let pool = FencePool::new();
        assert_eq!(pool.wait_timeout(), Duration::from_secs(5));
    }

    #[test]
    fn test_has_available() {
        let config = FencePoolConfig {
            initial_size: 1,
            max_size: 1,
            auto_grow: false,
            ..Default::default()
        };
        let mut pool = FencePool::with_config(config);
        assert!(pool.has_available());
        let _id = pool.acquire().expect("fence acquire should succeed");
        assert!(!pool.has_available());
    }
}
