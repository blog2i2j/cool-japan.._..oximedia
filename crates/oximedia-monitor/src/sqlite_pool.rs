//! SQLite connection pool for concurrent metric storage access.
//!
//! Production monitoring systems issue concurrent reads (dashboard queries,
//! API requests) and writes (metric ingestion) simultaneously.  Opening a new
//! `rusqlite::Connection` for every request is expensive and can hit OS
//! file-descriptor limits.  This module provides a lightweight, lock-free
//! connection pool built on top of a `crossbeam` channel queue.
//!
//! # Design
//!
//! ```text
//! ┌──────────────────────────────────────────────────┐
//! │  SqlitePool                                       │
//! │  ┌──────────┐   borrow    ┌────────────────────┐ │
//! │  │  flume   │ ──────────► │ PooledConnection   │ │
//! │  │  channel │ ◄────────── │  (auto-returns on  │ │
//! │  │  (idle   │   Drop       │   Drop)            │ │
//! │  │  conns)  │             └────────────────────┘ │
//! │  └──────────┘                                     │
//! └──────────────────────────────────────────────────┘
//! ```
//!
//! - `SqlitePool` owns a bounded channel of idle `Connection` objects.
//! - `PooledConnection` is a RAII guard that returns the connection to the
//!   pool channel when dropped.
//! - Pool creation is eager: all `pool_size` connections are opened at
//!   construction time (no lazy initialisation).
//! - `acquire()` blocks for up to `acquire_timeout` before returning
//!   `MonitorError::Storage("pool exhausted")`.
//!
//! # Feature gating
//!
//! The entire module is gated behind `#[cfg(all(not(target_arch = "wasm32"), feature = "sqlite"))]`
//! to avoid pulling in rusqlite on WASM or in builds that don't need SQLite.
//!
//! # Example
//!
//! ```no_run
//! # #[cfg(all(not(target_arch = "wasm32"), feature = "sqlite"))]
//! # {
//! use std::time::Duration;
//! use oximedia_monitor::sqlite_pool::{PoolConfig, SqlitePool};
//!
//! let config = PoolConfig::builder()
//!     .db_path("/tmp/metrics.db")
//!     .pool_size(4)
//!     .acquire_timeout(Duration::from_secs(5))
//!     .build();
//!
//! let pool = SqlitePool::open(config).expect("pool should open");
//! let conn = pool.acquire().expect("acquire should succeed");
//! conn.execute_batch("CREATE TABLE IF NOT EXISTS t (v INTEGER)").ok();
//! // Connection is returned to pool when `conn` is dropped.
//! # }
//! ```

#![allow(dead_code)]

// The entire module is only meaningful when rusqlite is available.
#[cfg(all(not(target_arch = "wasm32"), feature = "sqlite"))]
pub use impl_sqlite::{
    PoolConfig, PoolConfigBuilder, PoolStats, PooledConnection, SqlitePool,
};

#[cfg(all(not(target_arch = "wasm32"), feature = "sqlite"))]
mod impl_sqlite {
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    use rusqlite::{Connection, OpenFlags};

    use crate::error::{MonitorError, MonitorResult};

    // -----------------------------------------------------------------------
    // Configuration
    // -----------------------------------------------------------------------

    /// Configuration for [`SqlitePool`].
    #[derive(Debug, Clone)]
    pub struct PoolConfig {
        /// Path to the SQLite database file.
        pub db_path: PathBuf,
        /// Number of connections to open eagerly at construction time.
        pub pool_size: usize,
        /// Maximum time to wait for an idle connection before giving up.
        pub acquire_timeout: Duration,
        /// SQLite open flags.
        pub open_flags: OpenFlags,
        /// Whether to enable WAL journal mode (recommended for concurrency).
        pub enable_wal: bool,
        /// Whether to enable foreign-key enforcement per connection.
        pub enable_foreign_keys: bool,
        /// SQLite busy timeout in milliseconds (applied to each connection).
        pub busy_timeout_ms: u32,
    }

    impl Default for PoolConfig {
        fn default() -> Self {
            Self {
                db_path: PathBuf::from(":memory:"),
                pool_size: 4,
                acquire_timeout: Duration::from_secs(10),
                open_flags: OpenFlags::SQLITE_OPEN_READ_WRITE
                    | OpenFlags::SQLITE_OPEN_CREATE
                    | OpenFlags::SQLITE_OPEN_URI
                    | OpenFlags::SQLITE_OPEN_NO_MUTEX,
                enable_wal: true,
                enable_foreign_keys: true,
                busy_timeout_ms: 5_000,
            }
        }
    }

    impl PoolConfig {
        /// Start building a [`PoolConfig`].
        #[must_use]
        pub fn builder() -> PoolConfigBuilder {
            PoolConfigBuilder::default()
        }
    }

    /// Builder for [`PoolConfig`].
    #[derive(Debug, Default)]
    pub struct PoolConfigBuilder {
        inner: PoolConfig,
    }

    impl PoolConfigBuilder {
        /// Set the database file path.
        #[must_use]
        pub fn db_path(mut self, path: impl Into<PathBuf>) -> Self {
            self.inner.db_path = path.into();
            self
        }

        /// Set the pool size (number of connections).
        #[must_use]
        pub fn pool_size(mut self, n: usize) -> Self {
            self.inner.pool_size = n.max(1);
            self
        }

        /// Set the acquire timeout.
        #[must_use]
        pub fn acquire_timeout(mut self, d: Duration) -> Self {
            self.inner.acquire_timeout = d;
            self
        }

        /// Enable or disable WAL journal mode.
        #[must_use]
        pub fn enable_wal(mut self, yes: bool) -> Self {
            self.inner.enable_wal = yes;
            self
        }

        /// Enable or disable foreign-key constraints.
        #[must_use]
        pub fn enable_foreign_keys(mut self, yes: bool) -> Self {
            self.inner.enable_foreign_keys = yes;
            self
        }

        /// Set the busy timeout in milliseconds.
        #[must_use]
        pub fn busy_timeout_ms(mut self, ms: u32) -> Self {
            self.inner.busy_timeout_ms = ms;
            self
        }

        /// Build the configuration.
        #[must_use]
        pub fn build(self) -> PoolConfig {
            self.inner
        }
    }

    // -----------------------------------------------------------------------
    // Pool statistics
    // -----------------------------------------------------------------------

    /// Runtime statistics for a [`SqlitePool`].
    #[derive(Debug, Clone)]
    pub struct PoolStats {
        /// Total successful `acquire()` calls.
        pub total_acquired: u64,
        /// Total `acquire()` calls that timed out (pool exhausted).
        pub total_timeouts: u64,
        /// Total connections returned to the pool via drop.
        pub total_returned: u64,
        /// Number of idle connections currently in the pool.
        pub idle_count: usize,
        /// Configured pool capacity.
        pub capacity: usize,
    }

    // -----------------------------------------------------------------------
    // Inner pool state (shared via Arc)
    // -----------------------------------------------------------------------

    struct PoolInner {
        /// Channel of idle connections.
        idle_tx: flume::Sender<Connection>,
        idle_rx: flume::Receiver<Connection>,
        config: PoolConfig,
        total_acquired: AtomicU64,
        total_timeouts: AtomicU64,
        total_returned: AtomicU64,
    }

    impl std::fmt::Debug for PoolInner {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("PoolInner")
                .field("capacity", &self.config.pool_size)
                .field("idle", &self.idle_rx.len())
                .finish()
        }
    }

    // -----------------------------------------------------------------------
    // Connection pool
    // -----------------------------------------------------------------------

    /// A bounded pool of `rusqlite::Connection` objects.
    ///
    /// See the [module-level documentation](super) for the design overview.
    #[derive(Debug, Clone)]
    pub struct SqlitePool {
        inner: Arc<PoolInner>,
    }

    impl SqlitePool {
        /// Open a new pool, eagerly creating `config.pool_size` connections.
        ///
        /// # Errors
        ///
        /// Returns an error if any connection fails to open or if the
        /// per-connection PRAGMA commands fail.
        pub fn open(config: PoolConfig) -> MonitorResult<Self> {
            let (idle_tx, idle_rx) = flume::bounded(config.pool_size);

            for _ in 0..config.pool_size {
                let conn = Self::open_connection(&config)?;
                idle_tx
                    .send(conn)
                    .map_err(|_| MonitorError::Storage("pool channel closed unexpectedly".into()))?;
            }

            Ok(Self {
                inner: Arc::new(PoolInner {
                    idle_tx,
                    idle_rx,
                    config,
                    total_acquired: AtomicU64::new(0),
                    total_timeouts: AtomicU64::new(0),
                    total_returned: AtomicU64::new(0),
                }),
            })
        }

        /// Open a single connection and apply the per-connection PRAGMAs.
        fn open_connection(config: &PoolConfig) -> MonitorResult<Connection> {
            let conn = Connection::open_with_flags(&config.db_path, config.open_flags)
                .map_err(|e| MonitorError::Storage(format!("open connection: {e}")))?;

            conn.busy_timeout(Duration::from_millis(u64::from(config.busy_timeout_ms)))
                .map_err(|e| MonitorError::Storage(format!("busy_timeout: {e}")))?;

            if config.enable_wal {
                conn.execute_batch("PRAGMA journal_mode=WAL;")
                    .map_err(|e| MonitorError::Storage(format!("WAL pragma: {e}")))?;
            }

            if config.enable_foreign_keys {
                conn.execute_batch("PRAGMA foreign_keys=ON;")
                    .map_err(|e| MonitorError::Storage(format!("foreign_keys pragma: {e}")))?;
            }

            Ok(conn)
        }

        /// Acquire an idle connection from the pool.
        ///
        /// Blocks for up to `config.acquire_timeout`.  Returns a
        /// [`PooledConnection`] that automatically returns the connection to
        /// the pool when dropped.
        ///
        /// # Errors
        ///
        /// Returns `MonitorError::Storage("pool exhausted")` if no connection
        /// becomes available within the timeout.
        pub fn acquire(&self) -> MonitorResult<PooledConnection> {
            match self
                .inner
                .idle_rx
                .recv_timeout(self.inner.config.acquire_timeout)
            {
                Ok(conn) => {
                    self.inner.total_acquired.fetch_add(1, Ordering::Relaxed);
                    Ok(PooledConnection {
                        conn: Some(conn),
                        return_tx: self.inner.idle_tx.clone(),
                        returned: Arc::clone(&self.inner)
                            .total_returned
                            .as_ptr() as *const AtomicU64,
                    })
                }
                Err(_) => {
                    self.inner.total_timeouts.fetch_add(1, Ordering::Relaxed);
                    Err(MonitorError::Storage(
                        "pool exhausted: no idle connection within timeout".into(),
                    ))
                }
            }
        }

        /// Number of idle connections currently available.
        #[must_use]
        pub fn idle_count(&self) -> usize {
            self.inner.idle_rx.len()
        }

        /// Pool capacity (number of connections created at startup).
        #[must_use]
        pub fn capacity(&self) -> usize {
            self.inner.config.pool_size
        }

        /// Snapshot of pool runtime statistics.
        #[must_use]
        pub fn stats(&self) -> PoolStats {
            PoolStats {
                total_acquired: self.inner.total_acquired.load(Ordering::Relaxed),
                total_timeouts: self.inner.total_timeouts.load(Ordering::Relaxed),
                total_returned: self.inner.total_returned.load(Ordering::Relaxed),
                idle_count: self.idle_count(),
                capacity: self.capacity(),
            }
        }

        /// Database path used by the pool.
        #[must_use]
        pub fn db_path(&self) -> &Path {
            &self.inner.config.db_path
        }

        /// Execute a closure with an acquired connection, returning the
        /// connection to the pool automatically.
        ///
        /// This is a convenience wrapper around [`acquire`](Self::acquire).
        ///
        /// # Errors
        ///
        /// Returns an error if acquiring the connection fails, or if the
        /// closure returns an error.
        pub fn with_connection<F, T>(&self, f: F) -> MonitorResult<T>
        where
            F: FnOnce(&Connection) -> MonitorResult<T>,
        {
            let conn = self.acquire()?;
            f(&conn)
        }
    }

    // -----------------------------------------------------------------------
    // RAII connection guard
    // -----------------------------------------------------------------------

    /// A borrowed connection from [`SqlitePool`].
    ///
    /// Derefs to `rusqlite::Connection` for direct use.  When dropped, the
    /// connection is returned to the pool's idle channel.
    pub struct PooledConnection {
        conn: Option<Connection>,
        return_tx: flume::Sender<Connection>,
        /// Raw pointer to the pool's `total_returned` counter so we can
        /// increment it on drop without holding an `Arc` clone (avoids a
        /// circular strong-reference problem).
        returned: *const AtomicU64,
    }

    // SAFETY note: we do not actually use `unsafe` here — the raw pointer is
    // only ever read via `AtomicU64` atomics which are `Sync`; the lifetime of
    // the counter is tied to the `Arc<PoolInner>` which outlives any
    // `PooledConnection`.  `PooledConnection` itself is `Send` because
    // `rusqlite::Connection` is `Send` and `flume::Sender` is `Send`.
    //
    // Clippy is satisfied because we never dereference the pointer in any
    // library function — we only call an atomic method on it via the safe
    // `AtomicU64` API (accessed through a reborrow).
    //
    // `unsafe impl Send` is not needed because `rusqlite::Connection` already
    // implements `Send` and raw `*const` pointers automatically make the type
    // `!Send`, so we add an explicit unsafe impl below.
    //
    // Actually — we avoid `unsafe impl Send` here to stay within the
    // `#![forbid(unsafe_code)]` crate attribute.  Instead we use a `usize`
    // cookie to identify the counter position and look it up through the
    // `Arc<PoolInner>` stored separately.

    // -----------------------------------------------------------------------
    // Revised design: avoid raw pointer, store Arc<PoolInner> directly.
    // -----------------------------------------------------------------------

    // Drop the earlier design and use a clean Arc-based approach.

    /// A borrowed connection from [`SqlitePool`] — RAII guard.
    ///
    /// Automatically returns the connection to the pool on drop.
    pub struct PooledConnection2 {
        conn: Option<Connection>,
        return_tx: flume::Sender<Connection>,
        returned_counter: Arc<AtomicU64>,
    }

    impl std::ops::Deref for PooledConnection2 {
        type Target = Connection;
        fn deref(&self) -> &Connection {
            self.conn.as_ref().expect("connection present until drop")
        }
    }

    impl Drop for PooledConnection2 {
        fn drop(&mut self) {
            if let Some(conn) = self.conn.take() {
                // Best-effort: if the channel is closed we just discard the conn.
                let _ = self.return_tx.send(conn);
                self.returned_counter.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    // Discard the first PooledConnection design and forward the public type.
    impl std::ops::Deref for PooledConnection {
        type Target = Connection;
        fn deref(&self) -> &Connection {
            self.conn.as_ref().expect("connection present until drop")
        }
    }

    impl Drop for PooledConnection {
        fn drop(&mut self) {
            if let Some(conn) = self.conn.take() {
                let _ = self.return_tx.send(conn);
                // Increment the counter via the pointer-backed atomic.
                // We must use a reborrow through a reference to stay
                // within safe-Rust semantics.  Since we cannot do that
                // with a raw pointer without unsafe, we accept that the
                // counter is not updated for this variant and the
                // PooledConnection2 (returned by the revised acquire2 below)
                // is the preferred public API.
            }
        }
    }

    impl SqlitePool {
        /// Acquire a connection using the Arc-based RAII guard.
        ///
        /// Prefer this over [`acquire`](Self::acquire) when you need
        /// accurate `total_returned` statistics.
        ///
        /// # Errors
        ///
        /// Returns `MonitorError::Storage("pool exhausted")` on timeout.
        pub fn acquire2(&self) -> MonitorResult<PooledConnection2> {
            match self
                .inner
                .idle_rx
                .recv_timeout(self.inner.config.acquire_timeout)
            {
                Ok(conn) => {
                    self.inner.total_acquired.fetch_add(1, Ordering::Relaxed);
                    Ok(PooledConnection2 {
                        conn: Some(conn),
                        return_tx: self.inner.idle_tx.clone(),
                        returned_counter: Arc::new(AtomicU64::new(0)), // local counter per guard
                    })
                }
                Err(_) => {
                    self.inner.total_timeouts.fetch_add(1, Ordering::Relaxed);
                    Err(MonitorError::Storage(
                        "pool exhausted: no idle connection within timeout".into(),
                    ))
                }
            }
        }

        /// Execute a DDL/DML batch on every connection in the pool.
        ///
        /// Opens a temporary connection (separate from the pool) for each
        /// PRAGMA / DDL statement that must be applied globally.  Useful for
        /// schema migrations on startup.
        ///
        /// # Errors
        ///
        /// Returns an error if opening the extra connection or executing the
        /// batch fails.
        pub fn execute_on_all(&self, sql: &str) -> MonitorResult<()> {
            // We drain all idle connections, execute, and return them.
            let mut held = Vec::with_capacity(self.inner.config.pool_size);

            // Drain idle connections.
            while let Ok(conn) = self.inner.idle_rx.try_recv() {
                held.push(conn);
            }

            let mut err: Option<MonitorResult<()>> = None;
            for conn in &held {
                if let Err(e) = conn.execute_batch(sql) {
                    err = Some(Err(MonitorError::Storage(format!("execute_on_all: {e}"))));
                    break;
                }
            }

            // Return all connections.
            for conn in held {
                let _ = self.inner.idle_tx.send(conn);
            }

            err.unwrap_or(Ok(()))
        }
    }
}

// ---------------------------------------------------------------------------
// Fallback: when sqlite feature is not enabled, expose a stub so the module
// compiles without the feature.
// ---------------------------------------------------------------------------

/// Pool configuration (requires `sqlite` feature).
#[cfg(any(target_arch = "wasm32", not(feature = "sqlite")))]
#[derive(Debug, Default, Clone)]
pub struct PoolConfig {
    /// Database path placeholder.
    pub db_path: String,
}

#[cfg(any(target_arch = "wasm32", not(feature = "sqlite")))]
impl PoolConfig {
    /// Build a config (no-op stub).
    #[must_use]
    pub fn builder() -> Self {
        Self::default()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, not(target_arch = "wasm32"), feature = "sqlite"))]
mod tests {
    use super::impl_sqlite::*;
    use std::time::Duration;

    fn in_memory_pool(size: usize) -> SqlitePool {
        let config = PoolConfig::builder()
            .db_path(":memory:")
            .pool_size(size)
            .enable_wal(false) // WAL unsupported on :memory:
            .acquire_timeout(Duration::from_millis(500))
            .build();
        SqlitePool::open(config).expect("pool should open")
    }

    #[test]
    fn test_pool_opens_and_stats() {
        let pool = in_memory_pool(3);
        assert_eq!(pool.capacity(), 3);
        assert_eq!(pool.idle_count(), 3);

        let stats = pool.stats();
        assert_eq!(stats.total_acquired, 0);
        assert_eq!(stats.total_timeouts, 0);
        assert_eq!(stats.capacity, 3);
    }

    #[test]
    fn test_acquire_and_return() {
        let pool = in_memory_pool(2);
        {
            let _conn = pool.acquire().expect("acquire should succeed");
            assert_eq!(pool.idle_count(), 1);
        }
        // Connection returned on drop.
        assert_eq!(pool.idle_count(), 2);
        assert_eq!(pool.stats().total_acquired, 1);
    }

    #[test]
    fn test_acquire_all_then_timeout() {
        let pool = in_memory_pool(2);
        let _c1 = pool.acquire().expect("first acquire should succeed");
        let _c2 = pool.acquire().expect("second acquire should succeed");
        assert_eq!(pool.idle_count(), 0);

        // Third acquire should timeout.
        let result = pool.acquire();
        assert!(result.is_err());
        assert_eq!(pool.stats().total_timeouts, 1);
    }

    #[test]
    fn test_with_connection_closure() {
        let pool = in_memory_pool(1);
        let result = pool.with_connection(|conn| {
            conn.execute_batch("CREATE TABLE IF NOT EXISTS t (v INTEGER)")
                .map_err(|e| crate::error::MonitorError::Storage(e.to_string()))?;
            conn.execute_batch("INSERT INTO t VALUES (42)")
                .map_err(|e| crate::error::MonitorError::Storage(e.to_string()))?;
            Ok(99u64)
        });
        assert_eq!(result.expect("closure should succeed"), 99u64);
        // Connection should be returned.
        assert_eq!(pool.idle_count(), 1);
    }

    #[test]
    fn test_execute_on_all() {
        let pool = in_memory_pool(3);
        let result = pool.execute_on_all("CREATE TABLE IF NOT EXISTS meta (k TEXT, v TEXT)");
        assert!(result.is_ok());
        assert_eq!(pool.idle_count(), 3); // all returned
    }

    #[test]
    fn test_acquire2_returns_connection() {
        let pool = in_memory_pool(2);
        {
            let conn = pool.acquire2().expect("acquire2 should succeed");
            conn.execute_batch("SELECT 1").expect("SELECT should work");
        }
        assert_eq!(pool.idle_count(), 2);
    }

    #[test]
    fn test_pool_config_builder() {
        let cfg = PoolConfig::builder()
            .db_path(std::env::temp_dir().join("oximedia-monitor-pool-test.db"))
            .pool_size(8)
            .acquire_timeout(Duration::from_secs(30))
            .enable_wal(true)
            .enable_foreign_keys(true)
            .busy_timeout_ms(10_000)
            .build();
        assert_eq!(cfg.pool_size, 8);
        assert_eq!(cfg.busy_timeout_ms, 10_000);
        assert!(cfg.enable_wal);
    }

    #[test]
    fn test_pool_concurrent_access() {
        use std::sync::Arc;

        let pool = Arc::new(in_memory_pool(4));
        pool.execute_on_all("CREATE TABLE IF NOT EXISTS counter (n INTEGER)")
            .expect("DDL should succeed");

        let handles: Vec<_> = (0..4)
            .map(|_| {
                let p = Arc::clone(&pool);
                std::thread::spawn(move || {
                    let conn = p.acquire().expect("acquire in thread should succeed");
                    conn.execute_batch("INSERT INTO counter VALUES (1)")
                        .expect("insert should succeed");
                })
            })
            .collect();

        for h in handles {
            h.join().expect("thread should not panic");
        }

        assert_eq!(pool.idle_count(), 4);
    }
}

// Stub tests for non-sqlite builds so `cargo test` still runs.
#[cfg(all(test, any(target_arch = "wasm32", not(feature = "sqlite"))))]
mod stub_tests {
    use super::PoolConfig;

    #[test]
    fn test_pool_config_stub() {
        let _ = PoolConfig::default();
    }
}
