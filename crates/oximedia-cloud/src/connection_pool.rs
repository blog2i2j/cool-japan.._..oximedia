// connection_pool.rs — Generic connection pool for cloud storage providers.
//
// Provides:
//   - `ConnectionPool<T>` — generic bounded pool with idle-timeout and acquire-timeout
//   - `PooledConnection<T>` — RAII guard that returns the connection on drop
//   - `PoolStats` — snapshot of pool telemetry
#![allow(dead_code)]

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;

use crate::error::{CloudError, Result};

// ---------------------------------------------------------------------------
// PoolStats
// ---------------------------------------------------------------------------

/// A point-in-time snapshot of pool utilisation metrics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PoolStats {
    /// Total slots in the pool (idle + active + pending creation).
    pub total: usize,
    /// Connections currently sitting idle in the pool.
    pub idle: usize,
    /// Connections currently checked out by callers.
    pub active: usize,
    /// Number of successful `acquire` calls lifetime total.
    pub acquired_total: u64,
    /// Number of connections that were discarded due to idle timeout.
    pub evicted_total: u64,
}

impl PoolStats {
    /// Convenience: return the active count.
    #[must_use]
    pub fn active(&self) -> usize {
        self.active
    }
}

// ---------------------------------------------------------------------------
// ConnectionFactory
// ---------------------------------------------------------------------------

/// Trait that the pool uses to create and validate connections.
///
/// Implement this for each concrete connection type.
pub trait ConnectionFactory<T>: Send + Sync {
    /// Create a fresh connection, returning an error on failure.
    fn create(&self) -> Result<T>;

    /// Validate an existing connection before it is re-used.
    ///
    /// Returns `true` when the connection is still usable.
    fn is_valid(&self, conn: &T) -> bool;

    /// Perform any teardown when a connection is discarded.
    fn destroy(&self, conn: T) {
        drop(conn);
    }
}

// ---------------------------------------------------------------------------
// Internal slot
// ---------------------------------------------------------------------------

struct Slot<T> {
    /// `None` only during the brief eviction window when we drain stale conns.
    conn: Option<T>,
    /// When this connection was last returned to the pool.
    idle_since: Instant,
}

impl<T> Slot<T> {
    fn new(conn: T) -> Self {
        Self {
            conn: Some(conn),
            idle_since: Instant::now(),
        }
    }

    fn is_stale(&self, timeout: Duration) -> bool {
        Instant::now().duration_since(self.idle_since) > timeout
    }
}

// ---------------------------------------------------------------------------
// ConnectionPool<T>
// ---------------------------------------------------------------------------

struct PoolInner<T> {
    idle: VecDeque<Slot<T>>,
    /// Number of connections currently checked out.
    active: usize,
    /// Lifetime counters.
    acquired_total: u64,
    evicted_total: u64,
}

/// A generic, bounded connection pool.
///
/// ## Lifecycle
///
/// 1. Call `acquire()` to obtain a `PooledConnection<T>`.
/// 2. Use the connection via `Deref`/`DerefMut`.
/// 3. When the `PooledConnection` is dropped, the underlying connection is
///    returned to the pool (or discarded if the pool is full / the connection
///    has gone invalid).
pub struct ConnectionPool<T> {
    inner: Arc<Mutex<PoolInner<T>>>,
    factory: Arc<dyn ConnectionFactory<T>>,
    max_size: usize,
    idle_timeout: Duration,
    acquire_timeout: Duration,
}

impl<T: Send + 'static> ConnectionPool<T> {
    /// Create a new pool.
    ///
    /// - `max_size` — maximum number of connections (idle + active).
    /// - `idle_timeout` — how long an idle connection may sit before eviction.
    /// - `acquire_timeout` — how long `acquire` spins before returning
    ///   [`CloudError::Timeout`].
    pub fn new(
        factory: impl ConnectionFactory<T> + 'static,
        max_size: usize,
        idle_timeout: Duration,
        acquire_timeout: Duration,
    ) -> Result<Self> {
        if max_size == 0 {
            return Err(CloudError::InvalidConfig(
                "max_size must be at least 1".into(),
            ));
        }
        Ok(Self {
            inner: Arc::new(Mutex::new(PoolInner {
                idle: VecDeque::new(),
                active: 0,
                acquired_total: 0,
                evicted_total: 0,
            })),
            factory: Arc::new(factory),
            max_size,
            idle_timeout,
            acquire_timeout,
        })
    }

    /// Acquire a connection from the pool.
    ///
    /// The method first evicts stale idle connections, then either re-uses an
    /// idle one or creates a new one.  If the pool is at `max_size` and all
    /// connections are active, it spins (briefly yielding between polls) until
    /// one is returned or `acquire_timeout` elapses.
    pub fn acquire(&self) -> Result<PooledConnection<T>> {
        let deadline = Instant::now() + self.acquire_timeout;
        loop {
            // Collect stale connections outside the lock to call `destroy`.
            let stale = self.drain_stale();
            for conn in stale {
                self.factory.destroy(conn);
            }

            // Phase 1: drain invalid idle connections and collect candidates
            // under the lock; then validate them without holding the lock.
            let candidate: Option<T>;
            let mut invalid_conns: Vec<T> = Vec::new();
            {
                let mut guard = self.inner.lock();
                let mut found: Option<T> = None;

                while let Some(mut slot) = guard.idle.pop_front() {
                    if let Some(conn) = slot.conn.take() {
                        if self.factory.is_valid(&conn) {
                            found = Some(conn);
                            break;
                        } else {
                            guard.evicted_total += 1;
                            invalid_conns.push(conn);
                        }
                    }
                }
                candidate = found;
            }

            // Destroy invalid connections without holding the lock.
            for conn in invalid_conns {
                self.factory.destroy(conn);
            }

            if let Some(conn) = candidate {
                let mut guard = self.inner.lock();
                guard.active += 1;
                guard.acquired_total += 1;
                drop(guard);
                return Ok(PooledConnection {
                    conn: Some(conn),
                    pool: Arc::clone(&self.inner),
                    factory: Arc::clone(&self.factory),
                    max_size: self.max_size,
                });
            }

            // Phase 2: try to create a new connection if under the limit.
            let can_create = {
                let guard = self.inner.lock();
                guard.idle.len() + guard.active < self.max_size
            };

            if can_create {
                match self.factory.create() {
                    Ok(conn) => {
                        let mut guard = self.inner.lock();
                        guard.active += 1;
                        guard.acquired_total += 1;
                        drop(guard);
                        return Ok(PooledConnection {
                            conn: Some(conn),
                            pool: Arc::clone(&self.inner),
                            factory: Arc::clone(&self.factory),
                            max_size: self.max_size,
                        });
                    }
                    Err(e) => return Err(e),
                }
            }
            // Pool exhausted — spin.

            if Instant::now() >= deadline {
                return Err(CloudError::Timeout(format!(
                    "pool acquire timed out after {:?}",
                    self.acquire_timeout
                )));
            }
            // Brief yield to avoid busy-looping.
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    /// Return a point-in-time snapshot of pool statistics.
    pub fn stats(&self) -> PoolStats {
        let guard = self.inner.lock();
        PoolStats {
            total: guard.idle.len() + guard.active,
            idle: guard.idle.len(),
            active: guard.active,
            acquired_total: guard.acquired_total,
            evicted_total: guard.evicted_total,
        }
    }

    /// Manually trigger eviction of idle-timeout connections.
    pub fn evict_idle(&self) {
        let stale = self.drain_stale();
        for conn in stale {
            self.factory.destroy(conn);
        }
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    /// Remove stale idle slots under the lock, returning the drained
    /// connections so the caller can `destroy` them without holding the lock.
    fn drain_stale(&self) -> Vec<T> {
        let timeout = self.idle_timeout;
        let mut guard = self.inner.lock();

        // Partition idle queue: keep fresh slots, collect stale ones.
        // We cannot mutate `guard.evicted_total` while `guard.idle.drain` holds
        // a borrow on `guard`, so we count stale items first and update after.
        let all_slots: Vec<Slot<T>> = guard.idle.drain(..).collect();
        let mut fresh: VecDeque<Slot<T>> = VecDeque::with_capacity(all_slots.len());
        let mut stale: Vec<T> = Vec::new();
        let mut evicted_count: u64 = 0;

        for mut slot in all_slots {
            if slot.is_stale(timeout) {
                if let Some(conn) = slot.conn.take() {
                    stale.push(conn);
                    evicted_count += 1;
                }
            } else {
                fresh.push_back(slot);
            }
        }

        guard.idle = fresh;
        guard.evicted_total += evicted_count;
        stale
    }
}

// ---------------------------------------------------------------------------
// PooledConnection<T>
// ---------------------------------------------------------------------------

/// An RAII guard that holds a single connection checked out of a `ConnectionPool`.
///
/// When dropped, the connection is returned to the pool if it is still valid
/// and the pool has capacity; otherwise it is destroyed via the factory.
pub struct PooledConnection<T> {
    conn: Option<T>,
    pool: Arc<Mutex<PoolInner<T>>>,
    factory: Arc<dyn ConnectionFactory<T>>,
    max_size: usize,
}

impl<T> PooledConnection<T> {
    /// Access the underlying connection.
    pub fn get(&self) -> &T {
        self.conn
            .as_ref()
            .expect("connection is always present while guard is alive")
    }

    /// Access the underlying connection mutably.
    pub fn get_mut(&mut self) -> &mut T {
        self.conn
            .as_mut()
            .expect("connection is always present while guard is alive")
    }
}

impl<T> std::ops::Deref for PooledConnection<T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.get()
    }
}

impl<T> std::ops::DerefMut for PooledConnection<T> {
    fn deref_mut(&mut self) -> &mut T {
        self.get_mut()
    }
}

impl<T> Drop for PooledConnection<T> {
    fn drop(&mut self) {
        let conn = match self.conn.take() {
            Some(c) => c,
            None => return,
        };

        let should_return = self.factory.is_valid(&conn);
        if should_return {
            let mut guard = self.pool.lock();
            guard.active = guard.active.saturating_sub(1);
            let total = guard.idle.len() + guard.active;
            if total < self.max_size {
                guard.idle.push_back(Slot::new(conn));
                return;
            }
            // Pool is over capacity (shouldn't normally happen); fall through.
        } else {
            let mut guard = self.pool.lock();
            guard.active = guard.active.saturating_sub(1);
        }
        self.factory.destroy(conn);
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for PooledConnection<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PooledConnection")
            .field("conn", &self.conn)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // -----------------------------------------------------------------------
    // Minimal test connection

    #[derive(Debug)]
    struct Counter(usize);

    static ALIVE: AtomicUsize = AtomicUsize::new(0);

    struct CounterFactory {
        next_id: Mutex<usize>,
        /// If `Some(n)`, `create` errors for call index >= n.
        fail_after: Option<usize>,
        created: AtomicUsize,
        valid: bool,
    }

    impl CounterFactory {
        fn new() -> Self {
            Self {
                next_id: Mutex::new(0),
                fail_after: None,
                created: AtomicUsize::new(0),
                valid: true,
            }
        }

        fn with_fail_after(mut self, n: usize) -> Self {
            self.fail_after = Some(n);
            self
        }

        fn with_always_invalid(mut self) -> Self {
            self.valid = false;
            self
        }
    }

    impl ConnectionFactory<Counter> for CounterFactory {
        fn create(&self) -> Result<Counter> {
            let n = self.created.fetch_add(1, Ordering::SeqCst);
            if let Some(limit) = self.fail_after {
                if n >= limit {
                    return Err(CloudError::Network("simulated failure".into()));
                }
            }
            let mut id = self.next_id.lock();
            let conn = Counter(*id);
            *id += 1;
            ALIVE.fetch_add(1, Ordering::SeqCst);
            Ok(conn)
        }

        fn is_valid(&self, _conn: &Counter) -> bool {
            self.valid
        }

        fn destroy(&self, _conn: Counter) {
            ALIVE.fetch_sub(1, Ordering::SeqCst);
        }
    }

    fn make_pool(max: usize) -> ConnectionPool<Counter> {
        ConnectionPool::new(
            CounterFactory::new(),
            max,
            Duration::from_secs(60),
            Duration::from_millis(200),
        )
        .expect("pool creation must succeed")
    }

    // -----------------------------------------------------------------------

    #[test]
    fn test_pool_creation_invalid_max_size() {
        let result = ConnectionPool::new(
            CounterFactory::new(),
            0,
            Duration::from_secs(60),
            Duration::from_millis(100),
        );
        assert!(result.is_err(), "zero max_size should be rejected");
    }

    #[test]
    fn test_pool_acquire_creates_connection() {
        let pool = make_pool(4);
        let conn = pool.acquire().expect("acquire must succeed");
        assert_eq!(conn.0, 0);
        let stats = pool.stats();
        assert_eq!(stats.active, 1);
        assert_eq!(stats.idle, 0);
    }

    #[test]
    fn test_pool_returns_connection_on_drop() {
        let pool = make_pool(4);
        {
            let _conn = pool.acquire().expect("acquire");
            assert_eq!(pool.stats().active, 1);
        } // drop
        let stats = pool.stats();
        assert_eq!(stats.active, 0);
        assert_eq!(stats.idle, 1);
    }

    #[test]
    fn test_pool_reuses_idle_connection() {
        let pool = make_pool(4);
        {
            let _c = pool.acquire().expect("acquire");
        }
        // Second acquire should reuse the idle connection (id 0).
        let conn = pool.acquire().expect("acquire");
        assert_eq!(conn.0, 0, "should reuse the idle connection");
        assert_eq!(pool.stats().acquired_total, 2);
    }

    #[test]
    fn test_pool_max_size_respected() {
        let pool = make_pool(2);
        let _c1 = pool.acquire().expect("first acquire");
        let _c2 = pool.acquire().expect("second acquire");
        assert_eq!(pool.stats().active, 2);
        // Third acquire should time out.
        let err = pool.acquire();
        assert!(err.is_err(), "pool should be exhausted");
        match err.unwrap_err() {
            CloudError::Timeout(_) => {}
            other => panic!("expected Timeout, got {:?}", other),
        }
    }

    #[test]
    fn test_pool_stats_active_idle() {
        let pool = make_pool(4);
        let c1 = pool.acquire().expect("c1");
        let _c2 = pool.acquire().expect("c2");
        drop(c1);
        let stats = pool.stats();
        assert_eq!(stats.idle, 1);
        assert_eq!(stats.active, 1);
        assert_eq!(stats.total, 2);
    }

    #[test]
    fn test_pool_evict_idle() {
        let pool = ConnectionPool::new(
            CounterFactory::new(),
            4,
            Duration::from_millis(1), // very short timeout
            Duration::from_millis(200),
        )
        .expect("pool");

        let c = pool.acquire().expect("acquire");
        drop(c);
        assert_eq!(pool.stats().idle, 1);

        std::thread::sleep(Duration::from_millis(10));
        pool.evict_idle();
        let stats = pool.stats();
        assert_eq!(stats.idle, 0);
        assert_eq!(stats.evicted_total, 1);
    }

    #[test]
    fn test_pool_acquire_factory_error_propagated() {
        let pool = ConnectionPool::new(
            CounterFactory::new().with_fail_after(0),
            4,
            Duration::from_secs(60),
            Duration::from_millis(100),
        )
        .expect("pool");
        let err = pool.acquire();
        assert!(err.is_err());
        match err.unwrap_err() {
            CloudError::Network(_) => {}
            other => panic!("expected Network error, got {:?}", other),
        }
    }

    #[test]
    fn test_pooled_connection_deref() {
        let pool = make_pool(2);
        let conn = pool.acquire().expect("acquire");
        // Access via Deref
        let _id: usize = conn.0;
    }

    #[test]
    fn test_pool_multiple_acquire_drop_cycles() {
        let pool = make_pool(2);
        for _ in 0..8 {
            let c = pool.acquire().expect("acquire in loop");
            drop(c);
        }
        let stats = pool.stats();
        // All connections should be back in idle pool.
        assert_eq!(stats.active, 0);
        assert_eq!(stats.acquired_total, 8);
    }

    #[test]
    fn test_pool_invalid_connection_not_returned() {
        // A factory where is_valid always returns false.
        let pool = ConnectionPool::new(
            CounterFactory::new().with_always_invalid(),
            4,
            Duration::from_secs(60),
            Duration::from_millis(200),
        )
        .expect("pool");

        let c = pool.acquire().expect("first acquire should succeed");
        drop(c);
        // The invalid connection should not have been returned to idle.
        assert_eq!(pool.stats().idle, 0);
    }
}
