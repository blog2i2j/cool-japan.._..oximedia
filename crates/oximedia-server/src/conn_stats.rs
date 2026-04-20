//! Connection pool statistics and monitoring.

#![allow(dead_code)]

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// The lifecycle state of a pooled connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionState {
    /// Connection is available and ready to use.
    Idle,
    /// Connection is currently being used by a request.
    Active,
    /// Connection is being established.
    Connecting,
    /// Connection has been invalidated / closed.
    Closed,
}

impl ConnectionState {
    /// Returns `true` if the connection can be acquired (only `Idle`).
    pub fn is_usable(self) -> bool {
        self == Self::Idle
    }

    /// Returns `true` for non-closed states.
    pub fn is_alive(self) -> bool {
        self != Self::Closed
    }
}

/// Lightweight handle representing a connection in the pool.
#[derive(Debug, Clone)]
pub struct PooledConnection {
    /// Unique connection identifier within the pool.
    pub id: u64,
    /// Remote address (IP:port) string.
    pub remote_addr: String,
    /// Current lifecycle state.
    pub state: ConnectionState,
    /// Monotonic timestamp of when this connection was last used.
    last_used: Instant,
    /// Total number of requests served by this connection.
    pub request_count: u64,
}

impl PooledConnection {
    /// Creates a new `PooledConnection` in `Idle` state.
    pub fn new(id: u64, remote_addr: impl Into<String>) -> Self {
        Self {
            id,
            remote_addr: remote_addr.into(),
            state: ConnectionState::Idle,
            last_used: Instant::now(),
            request_count: 0,
        }
    }

    /// Returns `true` when the connection is in `Idle` state.
    pub fn is_idle(&self) -> bool {
        self.state == ConnectionState::Idle
    }

    /// Returns how long the connection has been idle (only meaningful when `Idle`).
    pub fn idle_duration(&self) -> Duration {
        self.last_used.elapsed()
    }

    /// Marks the connection as acquired (transitions to `Active`).
    pub fn acquire(&mut self) {
        self.state = ConnectionState::Active;
        self.last_used = Instant::now();
    }

    /// Marks the connection as returned to the pool (transitions to `Idle`).
    pub fn release(&mut self) {
        self.state = ConnectionState::Idle;
        self.request_count += 1;
        self.last_used = Instant::now();
    }

    /// Closes the connection.
    pub fn close(&mut self) {
        self.state = ConnectionState::Closed;
    }
}

/// A simple connection pool backed by an in-memory map.
#[derive(Debug)]
pub struct ConnectionPool {
    connections: HashMap<u64, PooledConnection>,
    max_size: usize,
    next_id: u64,
}

impl ConnectionPool {
    /// Creates a new pool with the given maximum number of connections.
    pub fn new(max_size: usize) -> Self {
        Self {
            connections: HashMap::new(),
            max_size,
            next_id: 1,
        }
    }

    /// Allocates a new connection slot for `remote_addr`.
    ///
    /// Returns `None` if the pool is at capacity.
    pub fn acquire(&mut self, remote_addr: impl Into<String>) -> Option<u64> {
        if self.connections.len() >= self.max_size {
            return None;
        }
        let id = self.next_id;
        self.next_id += 1;
        let mut conn = PooledConnection::new(id, remote_addr);
        conn.acquire();
        self.connections.insert(id, conn);
        Some(id)
    }

    /// Returns a connection back to the idle pool.
    ///
    /// Returns `false` if the connection ID is not known.
    pub fn release(&mut self, id: u64) -> bool {
        if let Some(conn) = self.connections.get_mut(&id) {
            conn.release();
            true
        } else {
            false
        }
    }

    /// Closes and removes a connection from the pool.
    pub fn remove(&mut self, id: u64) -> bool {
        if let Some(conn) = self.connections.get_mut(&id) {
            conn.close();
        }
        self.connections.remove(&id).is_some()
    }

    /// Returns the number of currently idle (available) connections.
    pub fn available_count(&self) -> usize {
        self.connections.values().filter(|c| c.is_idle()).count()
    }

    /// Returns the number of active (in-use) connections.
    pub fn active_count(&self) -> usize {
        self.connections
            .values()
            .filter(|c| c.state == ConnectionState::Active)
            .count()
    }

    /// Returns the total number of connections in the pool (any state).
    pub fn total_count(&self) -> usize {
        self.connections.len()
    }

    /// Removes all connections that have been idle longer than `timeout`.
    pub fn evict_stale(&mut self, timeout: Duration) {
        self.connections
            .retain(|_, c| !(c.is_idle() && c.idle_duration() >= timeout));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_connection_state_is_usable() {
        assert!(ConnectionState::Idle.is_usable());
        assert!(!ConnectionState::Active.is_usable());
        assert!(!ConnectionState::Closed.is_usable());
    }

    #[test]
    fn test_connection_state_is_alive() {
        assert!(ConnectionState::Idle.is_alive());
        assert!(ConnectionState::Active.is_alive());
        assert!(!ConnectionState::Closed.is_alive());
    }

    #[test]
    fn test_pooled_connection_new_is_idle() {
        let conn = PooledConnection::new(1, "127.0.0.1:8080");
        assert!(conn.is_idle());
        assert_eq!(conn.request_count, 0);
    }

    #[test]
    fn test_pooled_connection_acquire() {
        let mut conn = PooledConnection::new(1, "127.0.0.1:8080");
        conn.acquire();
        assert_eq!(conn.state, ConnectionState::Active);
    }

    #[test]
    fn test_pooled_connection_release_increments_count() {
        let mut conn = PooledConnection::new(2, "10.0.0.1:9000");
        conn.acquire();
        conn.release();
        assert!(conn.is_idle());
        assert_eq!(conn.request_count, 1);
    }

    #[test]
    fn test_pooled_connection_close() {
        let mut conn = PooledConnection::new(3, "192.168.1.1:80");
        conn.close();
        assert_eq!(conn.state, ConnectionState::Closed);
        assert!(!conn.is_idle());
    }

    #[test]
    fn test_pool_acquire_returns_id() {
        let mut pool = ConnectionPool::new(5);
        let id = pool.acquire("host1:8080");
        assert!(id.is_some());
        assert_eq!(pool.active_count(), 1);
    }

    #[test]
    fn test_pool_max_size() {
        let mut pool = ConnectionPool::new(2);
        assert!(pool.acquire("a:1").is_some());
        assert!(pool.acquire("b:2").is_some());
        assert!(pool.acquire("c:3").is_none()); // at capacity
    }

    #[test]
    fn test_pool_release() {
        let mut pool = ConnectionPool::new(4);
        let id = pool.acquire("host:80").expect("should succeed in test");
        assert_eq!(pool.active_count(), 1);
        pool.release(id);
        assert_eq!(pool.available_count(), 1);
        assert_eq!(pool.active_count(), 0);
    }

    #[test]
    fn test_pool_release_unknown_id() {
        let mut pool = ConnectionPool::new(4);
        assert!(!pool.release(9999));
    }

    #[test]
    fn test_pool_remove() {
        let mut pool = ConnectionPool::new(4);
        let id = pool.acquire("host:80").expect("should succeed in test");
        assert!(pool.remove(id));
        assert_eq!(pool.total_count(), 0);
    }

    #[test]
    fn test_pool_available_count() {
        let mut pool = ConnectionPool::new(10);
        let id1 = pool.acquire("a:1").expect("should succeed in test");
        let _id2 = pool.acquire("b:2").expect("should succeed in test");
        pool.release(id1);
        assert_eq!(pool.available_count(), 1);
        assert_eq!(pool.active_count(), 1);
    }

    #[test]
    fn test_pool_evict_stale_immediate() {
        let mut pool = ConnectionPool::new(10);
        let id = pool.acquire("host:80").expect("should succeed in test");
        pool.release(id); // now idle
                          // Evict with zero timeout — released connections are immediately stale
        pool.evict_stale(Duration::from_secs(0));
        assert_eq!(pool.total_count(), 0);
    }

    #[test]
    fn test_pool_evict_stale_keeps_active() {
        let mut pool = ConnectionPool::new(10);
        let _id = pool.acquire("host:80").expect("should succeed in test"); // stays Active
        pool.evict_stale(Duration::from_secs(0));
        // Active connections must not be evicted
        assert_eq!(pool.total_count(), 1);
    }
}
