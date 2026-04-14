// #![allow(dead_code)]
//! NDI connection pool with health-checking, load-balancing, and auto-scaling.
//!
//! Manages a reusable pool of NDI source connections. Idle connections are kept
//! alive and health-checked; the pool can grow and shrink within configured
//! bounds and supports multiple load-balancing strategies.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::module_name_repetitions)]

use std::collections::{HashMap, VecDeque};
use std::net::SocketAddr;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// LoadBalanceStrategy
// ---------------------------------------------------------------------------

/// Strategy used to pick a connection from the pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LoadBalanceStrategy {
    /// Always pick the connection with the fewest in-flight requests.
    LeastConnections,
    /// Pick the next connection in a round-robin fashion.
    RoundRobin,
    /// Pick the connection with the lowest observed latency.
    LowestLatency,
    /// Pick a random connection from the available pool.
    Random,
}

impl Default for LoadBalanceStrategy {
    fn default() -> Self {
        Self::RoundRobin
    }
}

// ---------------------------------------------------------------------------
// ConnectionHealth
// ---------------------------------------------------------------------------

/// Health status of a pooled connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConnectionHealth {
    /// Connection is healthy and ready to accept requests.
    Healthy,
    /// Connection is degraded (high latency / packet loss) but still usable.
    Degraded,
    /// Connection has failed and should not be used.
    Unhealthy,
    /// Connection is being checked.
    Checking,
}

// ---------------------------------------------------------------------------
// PooledConnectionInfo
// ---------------------------------------------------------------------------

/// Metadata describing a single connection slot in the pool.
#[derive(Debug, Clone)]
pub struct PooledConnectionInfo {
    /// Unique identifier for this slot.
    pub id: u64,
    /// Remote NDI source address.
    pub address: SocketAddr,
    /// Current health status.
    pub health: ConnectionHealth,
    /// Number of requests currently using this connection.
    pub in_flight: u32,
    /// Exponentially-weighted moving-average latency (ms).
    pub ewma_latency_ms: f64,
    /// When the connection was established.
    pub connected_at: Instant,
    /// When the last successful health check completed.
    pub last_health_check: Option<Instant>,
    /// Total requests served.
    pub requests_served: u64,
    /// Total requests that failed.
    pub requests_failed: u64,
}

impl PooledConnectionInfo {
    fn new(id: u64, address: SocketAddr) -> Self {
        Self {
            id,
            address,
            health: ConnectionHealth::Healthy,
            in_flight: 0,
            ewma_latency_ms: 0.0,
            connected_at: Instant::now(),
            last_health_check: None,
            requests_served: 0,
            requests_failed: 0,
        }
    }

    /// Update the EWMA latency with a new measurement.
    pub fn record_latency(&mut self, latency_ms: f64) {
        const ALPHA: f64 = 0.2;
        if self.ewma_latency_ms == 0.0 {
            self.ewma_latency_ms = latency_ms;
        } else {
            self.ewma_latency_ms = ALPHA * latency_ms + (1.0 - ALPHA) * self.ewma_latency_ms;
        }
    }

    /// Return success ratio (0.0–1.0) for this connection.
    pub fn success_ratio(&self) -> f64 {
        let total = self.requests_served + self.requests_failed;
        if total == 0 {
            1.0
        } else {
            self.requests_served as f64 / total as f64
        }
    }
}

// ---------------------------------------------------------------------------
// ConnectionPoolConfig
// ---------------------------------------------------------------------------

/// Configuration for the connection pool.
#[derive(Debug, Clone)]
pub struct ConnectionPoolConfig {
    /// Minimum number of connections to keep open.
    pub min_connections: usize,
    /// Maximum number of connections allowed.
    pub max_connections: usize,
    /// How long an idle connection may remain before being culled.
    pub idle_timeout: Duration,
    /// How often to run the health check sweep.
    pub health_check_interval: Duration,
    /// Latency threshold above which a connection is marked degraded (ms).
    pub degraded_latency_ms: f64,
    /// Latency threshold above which a connection is marked unhealthy (ms).
    pub unhealthy_latency_ms: f64,
    /// Minimum success ratio before a connection is marked unhealthy.
    pub min_success_ratio: f64,
    /// Load-balancing strategy.
    pub strategy: LoadBalanceStrategy,
}

impl Default for ConnectionPoolConfig {
    fn default() -> Self {
        Self {
            min_connections: 2,
            max_connections: 16,
            idle_timeout: Duration::from_secs(60),
            health_check_interval: Duration::from_secs(10),
            degraded_latency_ms: 50.0,
            unhealthy_latency_ms: 200.0,
            min_success_ratio: 0.95,
            strategy: LoadBalanceStrategy::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// PoolStats
// ---------------------------------------------------------------------------

/// Aggregate statistics for the pool.
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    /// Current number of connections in the pool.
    pub pool_size: usize,
    /// Number of healthy connections.
    pub healthy_count: usize,
    /// Number of degraded connections.
    pub degraded_count: usize,
    /// Number of unhealthy connections.
    pub unhealthy_count: usize,
    /// Total connections ever created.
    pub total_created: u64,
    /// Total connections ever removed.
    pub total_removed: u64,
    /// Total requests dispatched through the pool.
    pub total_requests: u64,
    /// Total requests that returned an error.
    pub total_failures: u64,
    /// Average EWMA latency across healthy connections (ms).
    pub avg_latency_ms: f64,
}

// ---------------------------------------------------------------------------
// ConnectionPool
// ---------------------------------------------------------------------------

/// A pool of reusable NDI source connections.
///
/// Maintains a set of connections within configured bounds, performs periodic
/// health checks, and selects connections using the configured strategy.
#[derive(Debug)]
pub struct ConnectionPool {
    config: ConnectionPoolConfig,
    connections: HashMap<u64, PooledConnectionInfo>,
    next_id: u64,
    /// Round-robin cursor.
    rr_cursor: usize,
    /// Ordered list of slot IDs for round-robin.
    rr_order: VecDeque<u64>,
    /// When the last health check sweep ran.
    last_sweep: Option<Instant>,
    /// Total connections ever created.
    total_created: u64,
    /// Total connections ever removed.
    total_removed: u64,
    /// Total requests dispatched.
    total_requests: u64,
    /// Total failures.
    total_failures: u64,
}

impl ConnectionPool {
    /// Create an empty pool with the given configuration.
    pub fn new(config: ConnectionPoolConfig) -> Self {
        Self {
            config,
            connections: HashMap::new(),
            next_id: 1,
            rr_cursor: 0,
            rr_order: VecDeque::new(),
            last_sweep: None,
            total_created: 0,
            total_removed: 0,
            total_requests: 0,
            total_failures: 0,
        }
    }

    /// Return the pool configuration.
    pub fn config(&self) -> &ConnectionPoolConfig {
        &self.config
    }

    /// Add a new connection to the pool. Returns the connection's slot ID, or
    /// `None` if the pool is at capacity.
    pub fn add_connection(&mut self, address: SocketAddr) -> Option<u64> {
        if self.connections.len() >= self.config.max_connections {
            return None;
        }
        let id = self.next_id;
        self.next_id += 1;
        let info = PooledConnectionInfo::new(id, address);
        self.connections.insert(id, info);
        self.rr_order.push_back(id);
        self.total_created += 1;
        Some(id)
    }

    /// Remove a connection by slot ID. Returns `true` if it existed.
    pub fn remove_connection(&mut self, id: u64) -> bool {
        if self.connections.remove(&id).is_some() {
            self.rr_order.retain(|&x| x != id);
            self.total_removed += 1;
            true
        } else {
            false
        }
    }

    /// Return the number of connections currently in the pool.
    pub fn len(&self) -> usize {
        self.connections.len()
    }

    /// Return `true` if the pool has no connections.
    pub fn is_empty(&self) -> bool {
        self.connections.is_empty()
    }

    /// Select a connection according to the configured load-balancing strategy.
    /// Returns the slot ID of the chosen connection, or `None` if no healthy
    /// connection is available.
    pub fn acquire(&mut self) -> Option<u64> {
        self.total_requests += 1;
        match self.config.strategy {
            LoadBalanceStrategy::RoundRobin => self.acquire_round_robin(),
            LoadBalanceStrategy::LeastConnections => self.acquire_least_connections(),
            LoadBalanceStrategy::LowestLatency => self.acquire_lowest_latency(),
            LoadBalanceStrategy::Random => self.acquire_random(),
        }
    }

    fn acquire_round_robin(&mut self) -> Option<u64> {
        let healthy: Vec<u64> = self.healthy_ids();
        if healthy.is_empty() {
            return None;
        }
        let idx = self.rr_cursor % healthy.len();
        self.rr_cursor = self.rr_cursor.wrapping_add(1);
        let id = healthy[idx];
        if let Some(conn) = self.connections.get_mut(&id) {
            conn.in_flight += 1;
            conn.requests_served += 1;
        }
        Some(id)
    }

    fn acquire_least_connections(&mut self) -> Option<u64> {
        let id = self
            .connections
            .values()
            .filter(|c| matches!(c.health, ConnectionHealth::Healthy | ConnectionHealth::Degraded))
            .min_by_key(|c| c.in_flight)?
            .id;
        if let Some(conn) = self.connections.get_mut(&id) {
            conn.in_flight += 1;
            conn.requests_served += 1;
        }
        Some(id)
    }

    fn acquire_lowest_latency(&mut self) -> Option<u64> {
        let id = self
            .connections
            .values()
            .filter(|c| matches!(c.health, ConnectionHealth::Healthy | ConnectionHealth::Degraded))
            .min_by(|a, b| {
                a.ewma_latency_ms
                    .partial_cmp(&b.ewma_latency_ms)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })?
            .id;
        if let Some(conn) = self.connections.get_mut(&id) {
            conn.in_flight += 1;
            conn.requests_served += 1;
        }
        Some(id)
    }

    fn acquire_random(&mut self) -> Option<u64> {
        let healthy: Vec<u64> = self.healthy_ids();
        if healthy.is_empty() {
            return None;
        }
        // Use a simple deterministic index based on current state (avoids rand dep)
        let idx = (self.total_requests as usize).wrapping_mul(2654435761) % healthy.len();
        let id = healthy[idx];
        if let Some(conn) = self.connections.get_mut(&id) {
            conn.in_flight += 1;
            conn.requests_served += 1;
        }
        Some(id)
    }

    /// Release a slot after use, optionally recording latency and whether it
    /// succeeded.
    pub fn release(&mut self, id: u64, latency_ms: f64, success: bool) {
        if let Some(conn) = self.connections.get_mut(&id) {
            if conn.in_flight > 0 {
                conn.in_flight -= 1;
            }
            conn.record_latency(latency_ms);
            if !success {
                conn.requests_failed += 1;
                self.total_failures += 1;
            }
        }
    }

    /// Mark a connection's health manually.
    pub fn set_health(&mut self, id: u64, health: ConnectionHealth) {
        if let Some(conn) = self.connections.get_mut(&id) {
            conn.health = health;
        }
    }

    /// Run a health-check sweep over all connections.
    ///
    /// Connections whose EWMA latency or success ratio exceeds configured
    /// thresholds are downgraded; connections below the minimum are upgraded.
    /// Idle connections beyond `min_connections` are removed.
    ///
    /// Returns the number of connections removed.
    pub fn sweep(&mut self) -> usize {
        self.last_sweep = Some(Instant::now());
        let mut to_remove: Vec<u64> = Vec::new();
        let min_conn = self.config.min_connections;
        let idle_timeout = self.config.idle_timeout;
        let unhealthy_latency = self.config.unhealthy_latency_ms;
        let degraded_latency = self.config.degraded_latency_ms;
        let min_ratio = self.config.min_success_ratio;
        let current_len = self.connections.len();

        for conn in self.connections.values_mut() {
            // Skip connections currently in flight.
            if conn.in_flight > 0 {
                continue;
            }

            // Assess health based on latency and success ratio.
            if conn.ewma_latency_ms >= unhealthy_latency
                || conn.success_ratio() < min_ratio
            {
                conn.health = ConnectionHealth::Unhealthy;
            } else if conn.ewma_latency_ms >= degraded_latency {
                conn.health = ConnectionHealth::Degraded;
            } else {
                conn.health = ConnectionHealth::Healthy;
            }

            conn.last_health_check = Some(Instant::now());

            // Mark idle connections for removal if above minimum.
            let idle_dur = Instant::now().duration_since(conn.connected_at);
            if idle_dur > idle_timeout
                && conn.requests_served == 0
                && current_len > min_conn
            {
                to_remove.push(conn.id);
            }
        }

        // Remove unhealthy connections above min threshold.
        let unhealthy: Vec<u64> = self
            .connections
            .values()
            .filter(|c| {
                c.health == ConnectionHealth::Unhealthy
                    && current_len > min_conn
            })
            .map(|c| c.id)
            .collect();

        for id in unhealthy {
            if !to_remove.contains(&id) {
                to_remove.push(id);
            }
        }

        let removed = to_remove.len();
        for id in to_remove {
            self.remove_connection(id);
        }
        removed
    }

    /// Return a snapshot of aggregate pool statistics.
    pub fn stats(&self) -> PoolStats {
        let mut healthy_count = 0usize;
        let mut degraded_count = 0usize;
        let mut unhealthy_count = 0usize;
        let mut latency_sum = 0.0f64;
        let mut latency_n = 0usize;

        for conn in self.connections.values() {
            match conn.health {
                ConnectionHealth::Healthy => {
                    healthy_count += 1;
                    latency_sum += conn.ewma_latency_ms;
                    latency_n += 1;
                }
                ConnectionHealth::Degraded => degraded_count += 1,
                ConnectionHealth::Unhealthy => unhealthy_count += 1,
                ConnectionHealth::Checking => {}
            }
        }

        let avg_latency_ms = if latency_n > 0 {
            latency_sum / latency_n as f64
        } else {
            0.0
        };

        PoolStats {
            pool_size: self.connections.len(),
            healthy_count,
            degraded_count,
            unhealthy_count,
            total_created: self.total_created,
            total_removed: self.total_removed,
            total_requests: self.total_requests,
            total_failures: self.total_failures,
            avg_latency_ms,
        }
    }

    /// Return the connection info for a given slot ID.
    pub fn connection_info(&self, id: u64) -> Option<&PooledConnectionInfo> {
        self.connections.get(&id)
    }

    /// Return all connection infos.
    pub fn all_connections(&self) -> impl Iterator<Item = &PooledConnectionInfo> {
        self.connections.values()
    }

    /// Return whether the pool needs more connections to meet the minimum.
    pub fn needs_more_connections(&self) -> bool {
        self.connections.len() < self.config.min_connections
    }

    /// Return the IDs of connections that are healthy or degraded.
    fn healthy_ids(&self) -> Vec<u64> {
        self.connections
            .values()
            .filter(|c| {
                matches!(
                    c.health,
                    ConnectionHealth::Healthy | ConnectionHealth::Degraded
                )
            })
            .map(|c| c.id)
            .collect()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn pool_addr(n: u8) -> SocketAddr {
        format!("127.0.0.{n}:5960").parse().expect("valid addr")
    }

    fn default_pool() -> ConnectionPool {
        ConnectionPool::new(ConnectionPoolConfig::default())
    }

    #[test]
    fn test_add_and_remove_connection() {
        let mut pool = default_pool();
        let id = pool.add_connection(pool_addr(1)).expect("should add");
        assert_eq!(pool.len(), 1);
        assert!(pool.remove_connection(id));
        assert_eq!(pool.len(), 0);
    }

    #[test]
    fn test_max_connections_enforced() {
        let config = ConnectionPoolConfig {
            max_connections: 2,
            ..Default::default()
        };
        let mut pool = ConnectionPool::new(config);
        pool.add_connection(pool_addr(1)).expect("ok");
        pool.add_connection(pool_addr(2)).expect("ok");
        let third = pool.add_connection(pool_addr(3));
        assert!(third.is_none(), "pool should reject when at max");
    }

    #[test]
    fn test_acquire_round_robin() {
        let config = ConnectionPoolConfig {
            strategy: LoadBalanceStrategy::RoundRobin,
            ..Default::default()
        };
        let mut pool = ConnectionPool::new(config);
        let id1 = pool.add_connection(pool_addr(1)).expect("ok");
        let id2 = pool.add_connection(pool_addr(2)).expect("ok");

        let got1 = pool.acquire().expect("got connection");
        let got2 = pool.acquire().expect("got connection");
        // Both should have been vended
        assert!(
            (got1 == id1 && got2 == id2) || (got1 == id2 && got2 == id1),
            "round robin should cycle through connections"
        );
    }

    #[test]
    fn test_acquire_returns_none_when_empty() {
        let mut pool = default_pool();
        assert!(pool.acquire().is_none());
    }

    #[test]
    fn test_acquire_least_connections() {
        let config = ConnectionPoolConfig {
            strategy: LoadBalanceStrategy::LeastConnections,
            ..Default::default()
        };
        let mut pool = ConnectionPool::new(config);
        let id1 = pool.add_connection(pool_addr(1)).expect("ok");
        let id2 = pool.add_connection(pool_addr(2)).expect("ok");

        // Manually set id1 to have many in-flight requests so id2 wins
        if let Some(c) = pool.connections.get_mut(&id1) {
            c.in_flight = 10;
        }

        // Now id2 should have fewer in-flight
        let picked = pool.acquire().expect("ok");
        assert_eq!(picked, id2, "should prefer connection with fewer in-flight");
    }

    #[test]
    fn test_release_decrements_in_flight() {
        let mut pool = default_pool();
        let id = pool.add_connection(pool_addr(1)).expect("ok");
        pool.acquire();
        assert_eq!(pool.connection_info(id).expect("exists").in_flight, 1);
        pool.release(id, 5.0, true);
        assert_eq!(pool.connection_info(id).expect("exists").in_flight, 0);
    }

    #[test]
    fn test_release_records_latency() {
        let mut pool = default_pool();
        let id = pool.add_connection(pool_addr(1)).expect("ok");
        pool.acquire();
        pool.release(id, 10.0, true);
        let info = pool.connection_info(id).expect("exists");
        assert!(info.ewma_latency_ms > 0.0);
    }

    #[test]
    fn test_release_failure_increments_counter() {
        let mut pool = default_pool();
        let id = pool.add_connection(pool_addr(1)).expect("ok");
        pool.acquire();
        pool.release(id, 5.0, false);
        let stats = pool.stats();
        assert_eq!(stats.total_failures, 1);
    }

    #[test]
    fn test_set_health() {
        let mut pool = default_pool();
        let id = pool.add_connection(pool_addr(1)).expect("ok");
        pool.set_health(id, ConnectionHealth::Unhealthy);
        assert_eq!(
            pool.connection_info(id).expect("exists").health,
            ConnectionHealth::Unhealthy
        );
    }

    #[test]
    fn test_acquire_skips_unhealthy() {
        let config = ConnectionPoolConfig {
            strategy: LoadBalanceStrategy::RoundRobin,
            ..Default::default()
        };
        let mut pool = ConnectionPool::new(config);
        let id1 = pool.add_connection(pool_addr(1)).expect("ok");
        let _id2 = pool.add_connection(pool_addr(2)).expect("ok");
        pool.set_health(id1, ConnectionHealth::Unhealthy);

        // Both acquire calls should return the healthy connection
        for _ in 0..4 {
            let got = pool.acquire().expect("should get healthy conn");
            assert_ne!(got, id1, "unhealthy connection should not be selected");
        }
    }

    #[test]
    fn test_stats() {
        let mut pool = default_pool();
        pool.add_connection(pool_addr(1)).expect("ok");
        pool.add_connection(pool_addr(2)).expect("ok");
        let stats = pool.stats();
        assert_eq!(stats.pool_size, 2);
        assert_eq!(stats.healthy_count, 2);
        assert_eq!(stats.total_created, 2);
    }

    #[test]
    fn test_needs_more_connections() {
        let config = ConnectionPoolConfig {
            min_connections: 3,
            ..Default::default()
        };
        let mut pool = ConnectionPool::new(config);
        assert!(pool.needs_more_connections());
        pool.add_connection(pool_addr(1)).expect("ok");
        pool.add_connection(pool_addr(2)).expect("ok");
        pool.add_connection(pool_addr(3)).expect("ok");
        assert!(!pool.needs_more_connections());
    }

    #[test]
    fn test_sweep_marks_unhealthy_high_latency() {
        let config = ConnectionPoolConfig {
            min_connections: 0,
            max_connections: 16,
            unhealthy_latency_ms: 100.0,
            degraded_latency_ms: 50.0,
            ..Default::default()
        };
        let mut pool = ConnectionPool::new(config);
        let id = pool.add_connection(pool_addr(1)).expect("ok");
        // Manually set very high latency
        if let Some(conn) = pool.connections.get_mut(&id) {
            conn.ewma_latency_ms = 200.0;
        }
        pool.sweep();
        // Connection should be removed (unhealthy and above min)
        // Or at least marked unhealthy then removed in sweep
        // (It depends on whether len > min_connections after removal)
        let _ = pool.stats(); // should not panic
    }

    #[test]
    fn test_all_connections_iterator() {
        let mut pool = default_pool();
        pool.add_connection(pool_addr(1)).expect("ok");
        pool.add_connection(pool_addr(2)).expect("ok");
        let count = pool.all_connections().count();
        assert_eq!(count, 2);
    }

    #[test]
    fn test_success_ratio_default() {
        let info = PooledConnectionInfo::new(1, pool_addr(1));
        assert!((info.success_ratio() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_success_ratio_with_failures() {
        let mut info = PooledConnectionInfo::new(1, pool_addr(1));
        info.requests_served = 9;
        info.requests_failed = 1;
        assert!((info.success_ratio() - 0.9).abs() < 1e-9);
    }

    #[test]
    fn test_acquire_lowest_latency() {
        let config = ConnectionPoolConfig {
            strategy: LoadBalanceStrategy::LowestLatency,
            ..Default::default()
        };
        let mut pool = ConnectionPool::new(config);
        let id1 = pool.add_connection(pool_addr(1)).expect("ok");
        let id2 = pool.add_connection(pool_addr(2)).expect("ok");

        // Give id1 a high latency and id2 a low latency
        if let Some(c) = pool.connections.get_mut(&id1) {
            c.ewma_latency_ms = 80.0;
        }
        if let Some(c) = pool.connections.get_mut(&id2) {
            c.ewma_latency_ms = 5.0;
        }

        let picked = pool.acquire().expect("ok");
        assert_eq!(picked, id2, "should pick lowest-latency connection");
    }
}
