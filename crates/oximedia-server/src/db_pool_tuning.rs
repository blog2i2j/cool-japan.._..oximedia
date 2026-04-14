//! Database connection pool size tuning with CPU-aware defaults.
//!
//! Derives sensible connection pool limits from CPU core count and workload
//! characteristics. The approach follows the "pool-size = C * (N + 1)" heuristic
//! (where C is the number of CPUs and N is the wait-ratio) popularised by HikariCP
//! and adapted for async Rust / SQLite + Postgres workloads.
//!
//! # Example
//!
//! ```rust
//! use oximedia_server::db_pool_tuning::{PoolTuningProfile, PoolTuner};
//!
//! let tuner = PoolTuner::new(PoolTuningProfile::default());
//! let cfg = tuner.derive_config(num_cpus());
//! assert!(cfg.max_connections >= cfg.min_connections);
//!
//! fn num_cpus() -> usize { 4 }
//! ```

#![allow(dead_code)]

use std::time::Duration;

// ── Pool workload profile ─────────────────────────────────────────────────────

/// Describes the ratio of I/O-wait to CPU-computation in each database query.
///
/// The tuner uses this to scale the pool size: a heavier I/O workload benefits
/// from more concurrent connections because each connection spends most of its
/// time waiting on the database rather than using the CPU.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WorkloadProfile {
    /// Mostly short reads / writes with minimal disk I/O.
    CpuBound,
    /// Mix of reads and writes, some full-text search or JSON queries.
    Balanced,
    /// Long analytical queries, aggregations, big media metadata scans.
    IoHeavy,
}

impl WorkloadProfile {
    /// Returns the empirically-chosen I/O-wait multiplier for pool sizing.
    ///
    /// CpuBound → 1.0 (pool ≈ CPUs), IoHeavy → 3.0 (pool ≈ 3 × CPUs).
    pub fn wait_multiplier(self) -> f64 {
        match self {
            Self::CpuBound => 1.0,
            Self::Balanced => 2.0,
            Self::IoHeavy => 3.0,
        }
    }
}

impl Default for WorkloadProfile {
    fn default() -> Self {
        Self::Balanced
    }
}

// ── Tuning profile ────────────────────────────────────────────────────────────

/// High-level pool tuning policy passed to [`PoolTuner`].
#[derive(Debug, Clone)]
pub struct PoolTuningProfile {
    /// Workload mix that drives multiplier selection.
    pub workload: WorkloadProfile,
    /// Floor: never allocate fewer than this many connections.
    pub min_floor: u32,
    /// Ceiling: never allocate more than this many connections regardless of CPU count.
    pub max_ceiling: u32,
    /// Idle connections are closed after this duration.
    pub idle_timeout: Duration,
    /// New connection must be acquired within this duration or the request errors.
    pub acquire_timeout: Duration,
    /// Total connection lifetime before forced recycle (guards against memory leaks).
    pub max_lifetime: Duration,
    /// Whether to enable connection-level health checks (test-on-acquire).
    pub health_check_enabled: bool,
}

impl Default for PoolTuningProfile {
    fn default() -> Self {
        Self {
            workload: WorkloadProfile::default(),
            min_floor: 2,
            max_ceiling: 64,
            idle_timeout: Duration::from_secs(600),
            acquire_timeout: Duration::from_secs(30),
            max_lifetime: Duration::from_secs(3600),
            health_check_enabled: true,
        }
    }
}

// ── Derived pool configuration ────────────────────────────────────────────────

/// Concrete pool limits derived by [`PoolTuner::derive_config`].
#[derive(Debug, Clone, PartialEq)]
pub struct DerivedPoolConfig {
    /// Minimum number of idle connections maintained in the pool.
    pub min_connections: u32,
    /// Maximum number of open connections allowed simultaneously.
    pub max_connections: u32,
    /// Time after which idle connections are closed.
    pub idle_timeout: Duration,
    /// Maximum time to wait for a connection from the pool.
    pub acquire_timeout: Duration,
    /// Maximum total lifetime for a connection.
    pub max_lifetime: Duration,
    /// Whether test-on-acquire health-ping is enabled.
    pub health_check_enabled: bool,
    /// Informational: the CPU count used during derivation.
    pub derived_from_cpus: usize,
}

impl DerivedPoolConfig {
    /// Validates that the derived configuration is internally consistent.
    ///
    /// Returns `Err` with a description if any constraint is violated.
    pub fn validate(&self) -> Result<(), String> {
        if self.min_connections > self.max_connections {
            return Err(format!(
                "min_connections ({}) must be ≤ max_connections ({})",
                self.min_connections, self.max_connections
            ));
        }
        if self.max_connections == 0 {
            return Err("max_connections must be > 0".to_string());
        }
        if self.acquire_timeout.is_zero() {
            return Err("acquire_timeout must be > 0".to_string());
        }
        Ok(())
    }

    /// Returns the headroom between min and max connections.
    pub fn burst_capacity(&self) -> u32 {
        self.max_connections.saturating_sub(self.min_connections)
    }
}

// ── Pool tuner ────────────────────────────────────────────────────────────────

/// Derives connection pool configuration from CPU count and a tuning profile.
#[derive(Debug, Clone)]
pub struct PoolTuner {
    profile: PoolTuningProfile,
}

impl PoolTuner {
    /// Creates a new tuner with the given profile.
    pub fn new(profile: PoolTuningProfile) -> Self {
        Self { profile }
    }

    /// Derives a [`DerivedPoolConfig`] for a system with `cpu_count` logical CPUs.
    ///
    /// Formula: `max = clamp(round(cpus * (wait_mult + 1)), min_floor, max_ceiling)`,
    ///          `min = max(1, max / 4)`.
    pub fn derive_config(&self, cpu_count: usize) -> DerivedPoolConfig {
        let cpus = cpu_count.max(1) as f64;
        let multiplier = self.profile.workload.wait_multiplier() + 1.0;
        let raw_max = (cpus * multiplier).round() as u32;
        let max_connections = raw_max
            .max(self.profile.min_floor)
            .min(self.profile.max_ceiling);
        let min_connections = (max_connections / 4).max(self.profile.min_floor.min(1));

        DerivedPoolConfig {
            min_connections,
            max_connections,
            idle_timeout: self.profile.idle_timeout,
            acquire_timeout: self.profile.acquire_timeout,
            max_lifetime: self.profile.max_lifetime,
            health_check_enabled: self.profile.health_check_enabled,
            derived_from_cpus: cpu_count,
        }
    }

    /// Returns the profile used by this tuner.
    pub fn profile(&self) -> &PoolTuningProfile {
        &self.profile
    }
}

impl Default for PoolTuner {
    fn default() -> Self {
        Self::new(PoolTuningProfile::default())
    }
}

// ── Runtime helper ────────────────────────────────────────────────────────────

/// Returns the number of logical CPUs on the current system.
///
/// Uses `std::thread::available_parallelism` with a fallback to 1.
pub fn logical_cpu_count() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// Derives a pool configuration using the given profile and the actual CPU count.
pub fn auto_tune(profile: PoolTuningProfile) -> DerivedPoolConfig {
    let cpus = logical_cpu_count();
    PoolTuner::new(profile).derive_config(cpus)
}

// ── Pool statistics ───────────────────────────────────────────────────────────

/// Live statistics snapshot from a running connection pool.
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    /// Number of connections currently idle.
    pub idle: u32,
    /// Number of connections currently in use.
    pub active: u32,
    /// Total number of connections (idle + active).
    pub total: u32,
    /// Total number of connection acquisition requests so far.
    pub total_acquisitions: u64,
    /// Number of acquisition requests that timed out.
    pub timeout_count: u64,
    /// Number of connections recycled due to max_lifetime expiry.
    pub recycled_count: u64,
}

impl PoolStats {
    /// Computes the pool utilisation as a fraction in `[0.0, 1.0]`.
    ///
    /// Returns `0.0` if `total == 0` to avoid division by zero.
    pub fn utilisation(&self, max_connections: u32) -> f64 {
        if max_connections == 0 {
            return 0.0;
        }
        self.active as f64 / max_connections as f64
    }

    /// Returns `true` if the pool is at or above its configured maximum.
    pub fn is_saturated(&self, max_connections: u32) -> bool {
        self.total >= max_connections
    }
}

// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workload_profile_multipliers() {
        assert!((WorkloadProfile::CpuBound.wait_multiplier() - 1.0).abs() < f64::EPSILON);
        assert!((WorkloadProfile::Balanced.wait_multiplier() - 2.0).abs() < f64::EPSILON);
        assert!((WorkloadProfile::IoHeavy.wait_multiplier() - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_derive_config_basic() {
        let tuner = PoolTuner::default();
        let cfg = tuner.derive_config(4);
        assert!(cfg.max_connections >= cfg.min_connections);
        assert!(cfg.max_connections > 0);
        assert_eq!(cfg.derived_from_cpus, 4);
    }

    #[test]
    fn test_derive_config_respects_ceiling() {
        let profile = PoolTuningProfile {
            max_ceiling: 8,
            ..Default::default()
        };
        let tuner = PoolTuner::new(profile);
        let cfg = tuner.derive_config(100);
        assert!(cfg.max_connections <= 8);
    }

    #[test]
    fn test_derive_config_respects_floor() {
        let profile = PoolTuningProfile {
            min_floor: 5,
            ..Default::default()
        };
        let tuner = PoolTuner::new(profile);
        let cfg = tuner.derive_config(1);
        assert!(cfg.max_connections >= 5);
    }

    #[test]
    fn test_derived_pool_config_validation_ok() {
        let tuner = PoolTuner::default();
        let cfg = tuner.derive_config(4);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_derived_pool_config_validation_min_gt_max() {
        let cfg = DerivedPoolConfig {
            min_connections: 10,
            max_connections: 5,
            idle_timeout: Duration::from_secs(600),
            acquire_timeout: Duration::from_secs(30),
            max_lifetime: Duration::from_secs(3600),
            health_check_enabled: true,
            derived_from_cpus: 4,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_derived_pool_config_zero_max_errors() {
        let cfg = DerivedPoolConfig {
            min_connections: 0,
            max_connections: 0,
            idle_timeout: Duration::from_secs(600),
            acquire_timeout: Duration::from_secs(30),
            max_lifetime: Duration::from_secs(3600),
            health_check_enabled: false,
            derived_from_cpus: 1,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_burst_capacity() {
        let cfg = DerivedPoolConfig {
            min_connections: 3,
            max_connections: 12,
            idle_timeout: Duration::from_secs(600),
            acquire_timeout: Duration::from_secs(30),
            max_lifetime: Duration::from_secs(3600),
            health_check_enabled: true,
            derived_from_cpus: 4,
        };
        assert_eq!(cfg.burst_capacity(), 9);
    }

    #[test]
    fn test_pool_stats_utilisation() {
        let stats = PoolStats {
            idle: 2,
            active: 6,
            total: 8,
            ..Default::default()
        };
        let util = stats.utilisation(10);
        assert!((util - 0.6).abs() < 1e-9);
    }

    #[test]
    fn test_pool_stats_utilisation_zero_max() {
        let stats = PoolStats::default();
        assert!((stats.utilisation(0) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_pool_stats_is_saturated() {
        let stats = PoolStats {
            total: 10,
            ..Default::default()
        };
        assert!(stats.is_saturated(10));
        assert!(!stats.is_saturated(11));
    }

    #[test]
    fn test_io_heavy_produces_larger_pool_than_cpu_bound() {
        let io_tuner = PoolTuner::new(PoolTuningProfile {
            workload: WorkloadProfile::IoHeavy,
            ..Default::default()
        });
        let cpu_tuner = PoolTuner::new(PoolTuningProfile {
            workload: WorkloadProfile::CpuBound,
            ..Default::default()
        });
        let io_cfg = io_tuner.derive_config(4);
        let cpu_cfg = cpu_tuner.derive_config(4);
        assert!(io_cfg.max_connections >= cpu_cfg.max_connections);
    }

    #[test]
    fn test_logical_cpu_count_positive() {
        assert!(logical_cpu_count() >= 1);
    }

    #[test]
    fn test_auto_tune_returns_valid_config() {
        let cfg = auto_tune(PoolTuningProfile::default());
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_single_cpu_produces_valid_config() {
        let tuner = PoolTuner::default();
        let cfg = tuner.derive_config(1);
        assert!(cfg.validate().is_ok());
        assert_eq!(cfg.derived_from_cpus, 1);
    }

    #[test]
    fn test_profile_accessor() {
        let profile = PoolTuningProfile {
            workload: WorkloadProfile::IoHeavy,
            max_ceiling: 32,
            ..Default::default()
        };
        let tuner = PoolTuner::new(profile.clone());
        assert_eq!(tuner.profile().max_ceiling, 32);
    }
}
