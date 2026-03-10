#![allow(dead_code)]
//! Resource quota management for distributed workers.
//!
//! Tracks and enforces CPU, memory, and GPU quotas per worker or per tenant,
//! preventing any single consumer from monopolising shared cluster resources.

use std::collections::HashMap;
use std::fmt;

/// A set of resource limits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResourceLimits {
    /// Maximum CPU cores (fractional allowed, e.g. 2.5).
    pub max_cpu_cores: f64,
    /// Maximum memory in bytes.
    pub max_memory_bytes: u64,
    /// Maximum GPU units (fractional allowed).
    pub max_gpu_units: f64,
    /// Maximum concurrent tasks.
    pub max_tasks: u32,
    /// Maximum network bandwidth in bytes/sec.
    pub max_bandwidth_bps: u64,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_cpu_cores: 4.0,
            max_memory_bytes: 8 * 1024 * 1024 * 1024, // 8 GiB
            max_gpu_units: 1.0,
            max_tasks: 8,
            max_bandwidth_bps: 1_000_000_000, // 1 Gbps
        }
    }
}

/// Current usage snapshot for a single consumer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResourceUsage {
    /// CPU cores currently used.
    pub cpu_cores: f64,
    /// Memory bytes currently used.
    pub memory_bytes: u64,
    /// GPU units currently used.
    pub gpu_units: f64,
    /// Number of active tasks.
    pub active_tasks: u32,
    /// Current bandwidth usage in bytes/sec.
    pub bandwidth_bps: u64,
}

impl Default for ResourceUsage {
    fn default() -> Self {
        Self {
            cpu_cores: 0.0,
            memory_bytes: 0,
            gpu_units: 0.0,
            active_tasks: 0,
            bandwidth_bps: 0,
        }
    }
}

impl ResourceUsage {
    /// Create a zero-usage snapshot.
    #[must_use]
    pub fn zero() -> Self {
        Self::default()
    }
}

/// Reason a quota check failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuotaViolation {
    /// CPU limit exceeded.
    CpuExceeded,
    /// Memory limit exceeded.
    MemoryExceeded,
    /// GPU limit exceeded.
    GpuExceeded,
    /// Task count limit exceeded.
    TasksExceeded,
    /// Bandwidth limit exceeded.
    BandwidthExceeded,
}

impl fmt::Display for QuotaViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CpuExceeded => write!(f, "CPU quota exceeded"),
            Self::MemoryExceeded => write!(f, "Memory quota exceeded"),
            Self::GpuExceeded => write!(f, "GPU quota exceeded"),
            Self::TasksExceeded => write!(f, "Task count quota exceeded"),
            Self::BandwidthExceeded => write!(f, "Bandwidth quota exceeded"),
        }
    }
}

/// A resource request describing what a new task needs.
#[derive(Debug, Clone, Copy)]
pub struct ResourceRequest {
    /// CPU cores required.
    pub cpu_cores: f64,
    /// Memory bytes required.
    pub memory_bytes: u64,
    /// GPU units required.
    pub gpu_units: f64,
    /// Bandwidth required (bytes/sec).
    pub bandwidth_bps: u64,
}

impl ResourceRequest {
    /// Create a CPU-only request.
    #[must_use]
    pub fn cpu_only(cores: f64) -> Self {
        Self {
            cpu_cores: cores,
            memory_bytes: 0,
            gpu_units: 0.0,
            bandwidth_bps: 0,
        }
    }

    /// Create a memory-only request.
    #[must_use]
    pub fn memory_only(bytes: u64) -> Self {
        Self {
            cpu_cores: 0.0,
            memory_bytes: bytes,
            gpu_units: 0.0,
            bandwidth_bps: 0,
        }
    }
}

/// Manages quotas and usage for multiple consumers (workers / tenants).
#[derive(Debug, Clone)]
pub struct QuotaManager {
    /// Per-consumer limits.
    limits: HashMap<String, ResourceLimits>,
    /// Per-consumer current usage.
    usage: HashMap<String, ResourceUsage>,
    /// Global (cluster-wide) limits (optional).
    global_limits: Option<ResourceLimits>,
    /// Global usage accumulator.
    global_usage: ResourceUsage,
}

impl QuotaManager {
    /// Create a new manager with no consumers.
    #[must_use]
    pub fn new() -> Self {
        Self {
            limits: HashMap::new(),
            usage: HashMap::new(),
            global_limits: None,
            global_usage: ResourceUsage::zero(),
        }
    }

    /// Set global cluster-wide limits.
    pub fn set_global_limits(&mut self, limits: ResourceLimits) {
        self.global_limits = Some(limits);
    }

    /// Register a consumer with specific limits.
    pub fn register(&mut self, consumer_id: impl Into<String>, limits: ResourceLimits) {
        let id = consumer_id.into();
        self.limits.insert(id.clone(), limits);
        self.usage.entry(id).or_insert_with(ResourceUsage::zero);
    }

    /// Remove a consumer.
    pub fn unregister(&mut self, consumer_id: &str) {
        if let Some(u) = self.usage.remove(consumer_id) {
            self.subtract_global(&u);
        }
        self.limits.remove(consumer_id);
    }

    /// Check whether a resource request can be satisfied for a consumer.
    ///
    /// Returns `Ok(())` if within quota, or `Err(violations)` listing which
    /// limits would be exceeded.
    #[allow(clippy::cast_precision_loss)]
    pub fn check(
        &self,
        consumer_id: &str,
        request: &ResourceRequest,
    ) -> Result<(), Vec<QuotaViolation>> {
        let limits = match self.limits.get(consumer_id) {
            Some(l) => l,
            None => return Err(vec![QuotaViolation::TasksExceeded]),
        };
        let usage = self.usage.get(consumer_id).copied().unwrap_or_default();
        let mut violations = Vec::new();

        if usage.cpu_cores + request.cpu_cores > limits.max_cpu_cores {
            violations.push(QuotaViolation::CpuExceeded);
        }
        if usage.memory_bytes + request.memory_bytes > limits.max_memory_bytes {
            violations.push(QuotaViolation::MemoryExceeded);
        }
        if usage.gpu_units + request.gpu_units > limits.max_gpu_units {
            violations.push(QuotaViolation::GpuExceeded);
        }
        if usage.active_tasks + 1 > limits.max_tasks {
            violations.push(QuotaViolation::TasksExceeded);
        }
        if usage.bandwidth_bps + request.bandwidth_bps > limits.max_bandwidth_bps {
            violations.push(QuotaViolation::BandwidthExceeded);
        }

        if violations.is_empty() {
            Ok(())
        } else {
            Err(violations)
        }
    }

    /// Acquire resources for a consumer (updates usage).
    ///
    /// Returns `Err` if quota would be exceeded.
    pub fn acquire(
        &mut self,
        consumer_id: &str,
        request: &ResourceRequest,
    ) -> Result<(), Vec<QuotaViolation>> {
        self.check(consumer_id, request)?;
        let usage = self
            .usage
            .get_mut(consumer_id)
            .ok_or_else(|| vec![QuotaViolation::TasksExceeded])?;
        usage.cpu_cores += request.cpu_cores;
        usage.memory_bytes += request.memory_bytes;
        usage.gpu_units += request.gpu_units;
        usage.active_tasks += 1;
        usage.bandwidth_bps += request.bandwidth_bps;

        self.global_usage.cpu_cores += request.cpu_cores;
        self.global_usage.memory_bytes += request.memory_bytes;
        self.global_usage.gpu_units += request.gpu_units;
        self.global_usage.active_tasks += 1;
        self.global_usage.bandwidth_bps += request.bandwidth_bps;
        Ok(())
    }

    /// Release resources for a consumer.
    pub fn release(&mut self, consumer_id: &str, request: &ResourceRequest) {
        if let Some(usage) = self.usage.get_mut(consumer_id) {
            usage.cpu_cores = (usage.cpu_cores - request.cpu_cores).max(0.0);
            usage.memory_bytes = usage.memory_bytes.saturating_sub(request.memory_bytes);
            usage.gpu_units = (usage.gpu_units - request.gpu_units).max(0.0);
            usage.active_tasks = usage.active_tasks.saturating_sub(1);
            usage.bandwidth_bps = usage.bandwidth_bps.saturating_sub(request.bandwidth_bps);

            self.global_usage.cpu_cores =
                (self.global_usage.cpu_cores - request.cpu_cores).max(0.0);
            self.global_usage.memory_bytes = self
                .global_usage
                .memory_bytes
                .saturating_sub(request.memory_bytes);
            self.global_usage.gpu_units =
                (self.global_usage.gpu_units - request.gpu_units).max(0.0);
            self.global_usage.active_tasks = self.global_usage.active_tasks.saturating_sub(1);
            self.global_usage.bandwidth_bps = self
                .global_usage
                .bandwidth_bps
                .saturating_sub(request.bandwidth_bps);
        }
    }

    /// Get current usage for a consumer.
    #[must_use]
    pub fn usage(&self, consumer_id: &str) -> Option<&ResourceUsage> {
        self.usage.get(consumer_id)
    }

    /// Get limits for a consumer.
    #[must_use]
    pub fn limits(&self, consumer_id: &str) -> Option<&ResourceLimits> {
        self.limits.get(consumer_id)
    }

    /// Get global usage.
    #[must_use]
    pub fn global_usage(&self) -> &ResourceUsage {
        &self.global_usage
    }

    /// Number of registered consumers.
    #[must_use]
    pub fn consumer_count(&self) -> usize {
        self.limits.len()
    }

    /// Utilisation ratio for a consumer's CPU (0.0 - 1.0+).
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn cpu_utilisation(&self, consumer_id: &str) -> Option<f64> {
        let limits = self.limits.get(consumer_id)?;
        let usage = self.usage.get(consumer_id)?;
        if limits.max_cpu_cores > 0.0 {
            Some(usage.cpu_cores / limits.max_cpu_cores)
        } else {
            Some(0.0)
        }
    }

    /// Utilisation ratio for a consumer's memory (0.0 - 1.0+).
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn memory_utilisation(&self, consumer_id: &str) -> Option<f64> {
        let limits = self.limits.get(consumer_id)?;
        let usage = self.usage.get(consumer_id)?;
        if limits.max_memory_bytes > 0 {
            Some(usage.memory_bytes as f64 / limits.max_memory_bytes as f64)
        } else {
            Some(0.0)
        }
    }

    /// Subtract usage from global totals (helper for unregister).
    fn subtract_global(&mut self, u: &ResourceUsage) {
        self.global_usage.cpu_cores = (self.global_usage.cpu_cores - u.cpu_cores).max(0.0);
        self.global_usage.memory_bytes = self
            .global_usage
            .memory_bytes
            .saturating_sub(u.memory_bytes);
        self.global_usage.gpu_units = (self.global_usage.gpu_units - u.gpu_units).max(0.0);
        self.global_usage.active_tasks = self
            .global_usage
            .active_tasks
            .saturating_sub(u.active_tasks);
        self.global_usage.bandwidth_bps = self
            .global_usage
            .bandwidth_bps
            .saturating_sub(u.bandwidth_bps);
    }
}

impl Default for QuotaManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_limits() {
        let l = ResourceLimits::default();
        assert!((l.max_cpu_cores - 4.0).abs() < f64::EPSILON);
        assert_eq!(l.max_tasks, 8);
    }

    #[test]
    fn test_zero_usage() {
        let u = ResourceUsage::zero();
        assert!((u.cpu_cores - 0.0).abs() < f64::EPSILON);
        assert_eq!(u.memory_bytes, 0);
        assert_eq!(u.active_tasks, 0);
    }

    #[test]
    fn test_register_and_count() {
        let mut mgr = QuotaManager::new();
        mgr.register("w1", ResourceLimits::default());
        mgr.register("w2", ResourceLimits::default());
        assert_eq!(mgr.consumer_count(), 2);
    }

    #[test]
    fn test_unregister() {
        let mut mgr = QuotaManager::new();
        mgr.register("w1", ResourceLimits::default());
        mgr.unregister("w1");
        assert_eq!(mgr.consumer_count(), 0);
    }

    #[test]
    fn test_check_within_quota() {
        let mut mgr = QuotaManager::new();
        mgr.register("w1", ResourceLimits::default());
        let req = ResourceRequest::cpu_only(1.0);
        assert!(mgr.check("w1", &req).is_ok());
    }

    #[test]
    fn test_check_exceeds_cpu() {
        let mut mgr = QuotaManager::new();
        let limits = ResourceLimits {
            max_cpu_cores: 2.0,
            ..Default::default()
        };
        mgr.register("w1", limits);
        let req = ResourceRequest::cpu_only(3.0);
        let err = mgr.check("w1", &req).unwrap_err();
        assert!(err.contains(&QuotaViolation::CpuExceeded));
    }

    #[test]
    fn test_check_exceeds_memory() {
        let mut mgr = QuotaManager::new();
        let limits = ResourceLimits {
            max_memory_bytes: 1000,
            ..Default::default()
        };
        mgr.register("w1", limits);
        let req = ResourceRequest::memory_only(2000);
        let err = mgr.check("w1", &req).unwrap_err();
        assert!(err.contains(&QuotaViolation::MemoryExceeded));
    }

    #[test]
    fn test_acquire_and_release() {
        let mut mgr = QuotaManager::new();
        mgr.register("w1", ResourceLimits::default());
        let req = ResourceRequest::cpu_only(1.0);
        assert!(mgr.acquire("w1", &req).is_ok());
        let usage = mgr.usage("w1").expect("usage should be available");
        assert!((usage.cpu_cores - 1.0).abs() < f64::EPSILON);
        assert_eq!(usage.active_tasks, 1);

        mgr.release("w1", &req);
        let usage = mgr.usage("w1").expect("usage should be available");
        assert!((usage.cpu_cores - 0.0).abs() < f64::EPSILON);
        assert_eq!(usage.active_tasks, 0);
    }

    #[test]
    fn test_acquire_denied_when_full() {
        let mut mgr = QuotaManager::new();
        let limits = ResourceLimits {
            max_tasks: 1,
            ..Default::default()
        };
        mgr.register("w1", limits);
        let req = ResourceRequest::cpu_only(0.5);
        assert!(mgr.acquire("w1", &req).is_ok());
        let err = mgr.acquire("w1", &req).unwrap_err();
        assert!(err.contains(&QuotaViolation::TasksExceeded));
    }

    #[test]
    fn test_global_usage_tracking() {
        let mut mgr = QuotaManager::new();
        mgr.register("w1", ResourceLimits::default());
        mgr.register("w2", ResourceLimits::default());
        let req = ResourceRequest::cpu_only(1.0);
        mgr.acquire("w1", &req)
            .expect("resource acquisition should succeed");
        mgr.acquire("w2", &req)
            .expect("resource acquisition should succeed");
        assert!((mgr.global_usage().cpu_cores - 2.0).abs() < f64::EPSILON);
        assert_eq!(mgr.global_usage().active_tasks, 2);
    }

    #[test]
    fn test_cpu_utilisation() {
        let mut mgr = QuotaManager::new();
        let limits = ResourceLimits {
            max_cpu_cores: 4.0,
            ..Default::default()
        };
        mgr.register("w1", limits);
        let req = ResourceRequest::cpu_only(2.0);
        mgr.acquire("w1", &req)
            .expect("resource acquisition should succeed");
        let util = mgr
            .cpu_utilisation("w1")
            .expect("utilisation should be available");
        assert!((util - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_memory_utilisation() {
        let mut mgr = QuotaManager::new();
        let limits = ResourceLimits {
            max_memory_bytes: 1000,
            ..Default::default()
        };
        mgr.register("w1", limits);
        let req = ResourceRequest::memory_only(250);
        mgr.acquire("w1", &req)
            .expect("resource acquisition should succeed");
        let util = mgr
            .memory_utilisation("w1")
            .expect("utilisation should be available");
        assert!((util - 0.25).abs() < f64::EPSILON);
    }

    #[test]
    fn test_quota_violation_display() {
        assert_eq!(
            QuotaViolation::CpuExceeded.to_string(),
            "CPU quota exceeded"
        );
        assert_eq!(
            QuotaViolation::MemoryExceeded.to_string(),
            "Memory quota exceeded"
        );
    }

    #[test]
    fn test_unknown_consumer_check_fails() {
        let mgr = QuotaManager::new();
        let req = ResourceRequest::cpu_only(1.0);
        assert!(mgr.check("unknown", &req).is_err());
    }
}
