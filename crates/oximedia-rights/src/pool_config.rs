//! Database connection-pool configuration and health monitoring.
//!
//! The `RightsManager` currently creates a single SQLite connection on
//! construction.  This module provides the configuration types and health
//! structures needed to migrate to a pooled model (e.g. `sqlx::SqlitePool`),
//! and tracks pool health metrics so operators can detect connection exhaustion
//! or excessive wait times.
//!
//! # Design
//!
//! The module is intentionally **database-agnostic** at the type level:
//! `PoolConfig` holds the parameters that would be passed to a pool builder;
//! `PoolHealth` is a plain-old-data snapshot reported by a pool monitor.
//! This allows wasm32 targets (which cannot link to SQLite) to compile the
//! configuration types without pulling in native database libraries.
//!
//! # Usage
//!
//! ```rust
//! use oximedia_rights::pool_config::{PoolConfig, PoolHealth, PoolMonitor};
//!
//! // Build the desired configuration.
//! let config = PoolConfig::builder()
//!     .max_connections(10)
//!     .min_idle(2)
//!     .connect_timeout_ms(3_000)
//!     .idle_timeout_ms(60_000)
//!     .max_lifetime_ms(300_000)
//!     .build();
//!
//! assert_eq!(config.max_connections, 10);
//! assert!(config.validate().is_ok());
//!
//! // Simulate pool health reporting.
//! let health = PoolHealth {
//!     active: 3,
//!     idle: 2,
//!     max: 10,
//!     waiting: 0,
//!     total_acquired: 50,
//!     total_released: 47,
//!     total_timed_out: 0,
//!     avg_wait_ms: 1.2,
//! };
//! let mut monitor = PoolMonitor::new(config);
//! monitor.record_snapshot(health, 1_000_000);
//! assert!(monitor.is_healthy(1_000_000));
//! ```

#![allow(dead_code)]

// ── PoolConfig ────────────────────────────────────────────────────────────────

/// Configuration for a database connection pool.
///
/// Construct using [`PoolConfigBuilder`] via [`PoolConfig::builder()`].
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Maximum number of connections in the pool.
    pub max_connections: u32,
    /// Minimum number of idle connections to maintain.
    pub min_idle: u32,
    /// Maximum time (ms) to wait for a connection before returning an error.
    pub connect_timeout_ms: u64,
    /// Time (ms) after which an idle connection is closed.
    pub idle_timeout_ms: u64,
    /// Maximum total lifetime (ms) of a connection, regardless of idle time.
    pub max_lifetime_ms: u64,
    /// Whether to test connections with a `SELECT 1` before handing them out.
    pub test_before_acquire: bool,
    /// Maximum number of pending requests allowed before new ones are
    /// rejected immediately.
    pub max_waiting: u32,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            max_connections: 5,
            min_idle: 1,
            connect_timeout_ms: 5_000,
            idle_timeout_ms: 600_000,  // 10 min
            max_lifetime_ms: 3_600_000, // 1 h
            test_before_acquire: false,
            max_waiting: 32,
        }
    }
}

impl PoolConfig {
    /// Return a [`PoolConfigBuilder`] for fluent construction.
    #[must_use]
    pub fn builder() -> PoolConfigBuilder {
        PoolConfigBuilder::default()
    }

    /// Validate the configuration, returning `Err` if any constraint is
    /// violated.
    pub fn validate(&self) -> crate::Result<()> {
        if self.max_connections == 0 {
            return Err(crate::RightsError::InvalidOperation(
                "max_connections must be > 0".to_string(),
            ));
        }
        if self.min_idle > self.max_connections {
            return Err(crate::RightsError::InvalidOperation(format!(
                "min_idle ({}) must not exceed max_connections ({})",
                self.min_idle, self.max_connections
            )));
        }
        if self.connect_timeout_ms == 0 {
            return Err(crate::RightsError::InvalidOperation(
                "connect_timeout_ms must be > 0".to_string(),
            ));
        }
        Ok(())
    }

    /// Fraction of the pool currently reserved (active / max).  Returns 0.0
    /// when `max_connections` is 0.
    #[must_use]
    pub fn utilisation(&self, active: u32) -> f64 {
        if self.max_connections == 0 {
            return 0.0;
        }
        active as f64 / self.max_connections as f64
    }

    /// Whether the pool is considered saturated at the given active count.
    /// The threshold is 90% of `max_connections`.
    #[must_use]
    pub fn is_saturated(&self, active: u32) -> bool {
        self.utilisation(active) >= 0.9
    }
}

// ── PoolConfigBuilder ─────────────────────────────────────────────────────────

/// Fluent builder for [`PoolConfig`].
#[derive(Debug, Default)]
pub struct PoolConfigBuilder {
    inner: PoolConfig,
}

impl PoolConfigBuilder {
    /// Set maximum number of connections.
    #[must_use]
    pub fn max_connections(mut self, n: u32) -> Self {
        self.inner.max_connections = n;
        self
    }

    /// Set minimum idle connections.
    #[must_use]
    pub fn min_idle(mut self, n: u32) -> Self {
        self.inner.min_idle = n;
        self
    }

    /// Set connection acquisition timeout (milliseconds).
    #[must_use]
    pub fn connect_timeout_ms(mut self, ms: u64) -> Self {
        self.inner.connect_timeout_ms = ms;
        self
    }

    /// Set idle connection lifetime (milliseconds).
    #[must_use]
    pub fn idle_timeout_ms(mut self, ms: u64) -> Self {
        self.inner.idle_timeout_ms = ms;
        self
    }

    /// Set maximum connection lifetime (milliseconds).
    #[must_use]
    pub fn max_lifetime_ms(mut self, ms: u64) -> Self {
        self.inner.max_lifetime_ms = ms;
        self
    }

    /// Enable or disable connection health testing before acquisition.
    #[must_use]
    pub fn test_before_acquire(mut self, yes: bool) -> Self {
        self.inner.test_before_acquire = yes;
        self
    }

    /// Set the maximum number of pending connection requests.
    #[must_use]
    pub fn max_waiting(mut self, n: u32) -> Self {
        self.inner.max_waiting = n;
        self
    }

    /// Finalise and return the [`PoolConfig`].
    #[must_use]
    pub fn build(self) -> PoolConfig {
        self.inner
    }
}

// ── PoolHealth ────────────────────────────────────────────────────────────────

/// A point-in-time snapshot of pool health metrics.
#[derive(Debug, Clone)]
pub struct PoolHealth {
    /// Number of connections currently in use.
    pub active: u32,
    /// Number of idle (available) connections.
    pub idle: u32,
    /// Pool maximum (from [`PoolConfig::max_connections`]).
    pub max: u32,
    /// Number of callers waiting for a connection.
    pub waiting: u32,
    /// Cumulative count of successful connection acquisitions.
    pub total_acquired: u64,
    /// Cumulative count of connections returned to the pool.
    pub total_released: u64,
    /// Cumulative count of acquisition attempts that timed out.
    pub total_timed_out: u64,
    /// Rolling average wait time for acquisitions (milliseconds).
    pub avg_wait_ms: f64,
}

impl PoolHealth {
    /// Total connections (active + idle).
    #[must_use]
    pub fn total(&self) -> u32 {
        self.active.saturating_add(self.idle)
    }

    /// Pool utilisation fraction (active / max). Returns 0.0 if max is 0.
    #[must_use]
    pub fn utilisation(&self) -> f64 {
        if self.max == 0 {
            return 0.0;
        }
        self.active as f64 / self.max as f64
    }

    /// Whether any connections are waiting for acquisition.
    #[must_use]
    pub fn has_waiters(&self) -> bool {
        self.waiting > 0
    }

    /// Whether the pool has any connections in flight (i.e. is not empty).
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.active > 0
    }
}

// ── HealthThresholds ──────────────────────────────────────────────────────────

/// Thresholds used by [`PoolMonitor`] to decide whether the pool is healthy.
#[derive(Debug, Clone)]
pub struct HealthThresholds {
    /// Average wait time (ms) above which the pool is considered degraded.
    pub max_avg_wait_ms: f64,
    /// Utilisation fraction above which the pool is considered saturated.
    pub max_utilisation: f64,
    /// Timeout count above which the pool is considered unhealthy.
    pub max_total_timed_out: u64,
}

impl Default for HealthThresholds {
    fn default() -> Self {
        Self {
            max_avg_wait_ms: 100.0,
            max_utilisation: 0.9,
            max_total_timed_out: 10,
        }
    }
}

// ── PoolMonitor ───────────────────────────────────────────────────────────────

/// Tracks pool health snapshots over time and exposes health-check logic.
///
/// Snapshots are retained in insertion order.  For long-running processes
/// callers should call [`trim_snapshots`](PoolMonitor::trim_snapshots)
/// periodically.
#[derive(Debug)]
pub struct PoolMonitor {
    config: PoolConfig,
    thresholds: HealthThresholds,
    snapshots: Vec<(u64, PoolHealth)>, // (timestamp_ms, health)
}

impl PoolMonitor {
    /// Create a new monitor for the given pool configuration.
    #[must_use]
    pub fn new(config: PoolConfig) -> Self {
        Self {
            config,
            thresholds: HealthThresholds::default(),
            snapshots: Vec::new(),
        }
    }

    /// Override the default health thresholds.
    #[must_use]
    pub fn with_thresholds(mut self, thresholds: HealthThresholds) -> Self {
        self.thresholds = thresholds;
        self
    }

    /// Record a new health snapshot at `timestamp_ms`.
    pub fn record_snapshot(&mut self, health: PoolHealth, timestamp_ms: u64) {
        self.snapshots.push((timestamp_ms, health));
    }

    /// Remove snapshots older than `older_than_ms` milliseconds before `now`.
    pub fn trim_snapshots(&mut self, now: u64, older_than_ms: u64) {
        let cutoff = now.saturating_sub(older_than_ms);
        self.snapshots.retain(|(ts, _)| *ts >= cutoff);
    }

    /// The most recent health snapshot, or `None` if no snapshots exist.
    #[must_use]
    pub fn latest(&self) -> Option<&PoolHealth> {
        self.snapshots.last().map(|(_, h)| h)
    }

    /// Number of recorded snapshots.
    #[must_use]
    pub fn snapshot_count(&self) -> usize {
        self.snapshots.len()
    }

    /// Whether the pool is considered healthy at `now` based on the latest
    /// snapshot and the configured thresholds.
    ///
    /// Returns `false` if no snapshots have been recorded.
    #[must_use]
    pub fn is_healthy(&self, _now: u64) -> bool {
        let Some(h) = self.latest() else { return false };
        h.avg_wait_ms <= self.thresholds.max_avg_wait_ms
            && h.utilisation() <= self.thresholds.max_utilisation
            && h.total_timed_out <= self.thresholds.max_total_timed_out
    }

    /// Pool utilisation from the latest snapshot. Returns `None` if no
    /// snapshots exist.
    #[must_use]
    pub fn current_utilisation(&self) -> Option<f64> {
        self.latest().map(|h| h.utilisation())
    }

    /// Total acquisition timeouts across all recorded snapshots.
    #[must_use]
    pub fn total_timeouts(&self) -> u64 {
        self.snapshots
            .iter()
            .map(|(_, h)| h.total_timed_out)
            .max()
            .unwrap_or(0)
    }

    /// Rolling average wait time (ms) computed across the recorded snapshots,
    /// or `None` if no snapshots exist.
    #[must_use]
    pub fn rolling_avg_wait_ms(&self) -> Option<f64> {
        if self.snapshots.is_empty() {
            return None;
        }
        let sum: f64 = self.snapshots.iter().map(|(_, h)| h.avg_wait_ms).sum();
        Some(sum / self.snapshots.len() as f64)
    }

    /// Access the pool configuration used by this monitor.
    #[must_use]
    pub fn config(&self) -> &PoolConfig {
        &self.config
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn healthy_snapshot(active: u32, max: u32) -> PoolHealth {
        PoolHealth {
            active,
            idle: max - active,
            max,
            waiting: 0,
            total_acquired: 100,
            total_released: 97,
            total_timed_out: 0,
            avg_wait_ms: 2.5,
        }
    }

    // ── PoolConfig ──

    #[test]
    fn test_config_default_values() {
        let cfg = PoolConfig::default();
        assert_eq!(cfg.max_connections, 5);
        assert_eq!(cfg.min_idle, 1);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_builder_roundtrip() {
        let cfg = PoolConfig::builder()
            .max_connections(20)
            .min_idle(4)
            .connect_timeout_ms(2_000)
            .idle_timeout_ms(30_000)
            .max_lifetime_ms(120_000)
            .test_before_acquire(true)
            .max_waiting(16)
            .build();
        assert_eq!(cfg.max_connections, 20);
        assert_eq!(cfg.min_idle, 4);
        assert!(cfg.test_before_acquire);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_validate_zero_max_connections() {
        let cfg = PoolConfig::builder().max_connections(0).build();
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validate_min_idle_exceeds_max() {
        let cfg = PoolConfig::builder()
            .max_connections(3)
            .min_idle(5)
            .build();
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validate_zero_timeout() {
        let cfg = PoolConfig::builder()
            .max_connections(5)
            .connect_timeout_ms(0)
            .build();
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_utilisation() {
        let cfg = PoolConfig::builder().max_connections(10).build();
        let u = cfg.utilisation(7);
        assert!((u - 0.7).abs() < 1e-9);
    }

    #[test]
    fn test_config_is_saturated() {
        let cfg = PoolConfig::builder().max_connections(10).build();
        assert!(!cfg.is_saturated(8));  // 80% — not saturated
        assert!(cfg.is_saturated(9));   // 90% — saturated
    }

    // ── PoolHealth ──

    #[test]
    fn test_health_total() {
        let h = healthy_snapshot(3, 10);
        assert_eq!(h.total(), 10); // active(3) + idle(7)
    }

    #[test]
    fn test_health_utilisation() {
        let h = healthy_snapshot(4, 10);
        assert!((h.utilisation() - 0.4).abs() < 1e-9);
    }

    #[test]
    fn test_health_has_waiters() {
        let mut h = healthy_snapshot(3, 10);
        assert!(!h.has_waiters());
        h.waiting = 2;
        assert!(h.has_waiters());
    }

    // ── PoolMonitor ──

    #[test]
    fn test_monitor_no_snapshots_unhealthy() {
        let monitor = PoolMonitor::new(PoolConfig::default());
        assert!(!monitor.is_healthy(1_000));
    }

    #[test]
    fn test_monitor_healthy_snapshot() {
        let mut monitor = PoolMonitor::new(PoolConfig::default());
        monitor.record_snapshot(healthy_snapshot(2, 5), 1_000);
        assert!(monitor.is_healthy(1_000));
    }

    #[test]
    fn test_monitor_unhealthy_high_wait() {
        let mut monitor = PoolMonitor::new(PoolConfig::default());
        let mut h = healthy_snapshot(2, 5);
        h.avg_wait_ms = 999.0; // above threshold of 100 ms
        monitor.record_snapshot(h, 1_000);
        assert!(!monitor.is_healthy(1_000));
    }

    #[test]
    fn test_monitor_unhealthy_too_many_timeouts() {
        let mut monitor = PoolMonitor::new(PoolConfig::default());
        let mut h = healthy_snapshot(2, 5);
        h.total_timed_out = 50; // above default threshold of 10
        monitor.record_snapshot(h, 1_000);
        assert!(!monitor.is_healthy(1_000));
    }

    #[test]
    fn test_monitor_trim_snapshots() {
        let mut monitor = PoolMonitor::new(PoolConfig::default());
        monitor.record_snapshot(healthy_snapshot(1, 5), 100);
        monitor.record_snapshot(healthy_snapshot(2, 5), 200);
        monitor.record_snapshot(healthy_snapshot(3, 5), 300);
        // Keep only snapshots within 50 ms of now=300
        monitor.trim_snapshots(300, 50);
        assert_eq!(monitor.snapshot_count(), 1);
    }

    #[test]
    fn test_monitor_rolling_avg_wait() {
        let mut monitor = PoolMonitor::new(PoolConfig::default());
        let mut h1 = healthy_snapshot(1, 5);
        h1.avg_wait_ms = 10.0;
        let mut h2 = healthy_snapshot(2, 5);
        h2.avg_wait_ms = 20.0;
        monitor.record_snapshot(h1, 100);
        monitor.record_snapshot(h2, 200);
        let avg = monitor.rolling_avg_wait_ms().expect("avg should be Some");
        assert!((avg - 15.0).abs() < 1e-9);
    }

    #[test]
    fn test_monitor_current_utilisation() {
        let mut monitor = PoolMonitor::new(PoolConfig::default());
        monitor.record_snapshot(healthy_snapshot(3, 5), 100);
        // active=3, max=5 → 60%
        let u = monitor.current_utilisation().expect("utilisation should be Some");
        assert!((u - 0.6).abs() < 1e-9);
    }

    #[test]
    fn test_monitor_total_timeouts() {
        let mut monitor = PoolMonitor::new(PoolConfig::default());
        let mut h = healthy_snapshot(1, 5);
        h.total_timed_out = 7;
        monitor.record_snapshot(h, 100);
        assert_eq!(monitor.total_timeouts(), 7);
    }
}
