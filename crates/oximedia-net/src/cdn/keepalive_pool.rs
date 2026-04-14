//! HTTP keep-alive connection pool for CDN edge servers.
//!
//! This module implements a per-host HTTP/1.1 keep-alive connection pool that
//! enables the CDN module to reuse TCP connections when fetching or uploading
//! segments to edge servers, dramatically reducing latency caused by TCP
//! connection setup and TLS handshake overhead.
//!
//! ## Design
//!
//! ```text
//!  CdnKeepalivePool
//!    ├── host "cdn1.example.com" ── [conn_a, conn_b]  (idle pool)
//!    ├── host "cdn2.example.com" ── [conn_c]
//!    └── host "origin.example.com" ── []
//! ```
//!
//! ### Key properties
//!
//! - **Per-host pools** — connections are keyed by `(scheme, host, port)`.
//! - **Max-idle control** — each host slot holds at most `max_idle_per_host`
//!   connections; excess connections are dropped.
//! - **Max-total control** — the pool enforces a global ceiling across all hosts.
//! - **Idle timeout** — connections idle longer than `idle_timeout` are discarded
//!   rather than returned to the pool (avoids using half-closed connections).
//! - **Acquire/release** — callers acquire a [`PooledConn`] token; when dropped,
//!   the token returns the connection metadata to the pool.
//! - **Statistics** — per-host and aggregate statistics are tracked.
//!
//! The implementation is synchronous (using `parking_lot::Mutex`) and
//! intentionally does not hold actual TCP sockets, since socket lifecycle
//! management belongs to the calling HTTP client.  Instead, the pool manages
//! *connection descriptors* — lightweight tokens carrying the metadata needed
//! for HTTP/1.1 request pipelining decisions (keep-alive state, request count,
//! etc.).

#![allow(dead_code)]

use parking_lot::Mutex;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::error::{NetError, NetResult};

// ─── Host Key ─────────────────────────────────────────────────────────────────

/// A normalised (scheme, host, port) triple used as the pool key.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct HostKey {
    /// URI scheme: `"https"` or `"http"`.
    pub scheme: String,
    /// Hostname or IP address (lower-cased).
    pub host: String,
    /// TCP port number.
    pub port: u16,
}

impl HostKey {
    /// Parses a URL string into a [`HostKey`].
    ///
    /// # Errors
    ///
    /// Returns [`NetError::InvalidUrl`] if the URL is malformed or missing a host.
    pub fn from_url(url: &str) -> NetResult<Self> {
        // Minimal URL parser: scheme://host[:port]/path
        let (scheme, rest) = url
            .split_once("://")
            .ok_or_else(|| NetError::invalid_url(format!("missing scheme in URL: {url}")))?;

        let authority = rest.split('/').next().unwrap_or(rest);

        let (host_part, port) = if let Some((h, p)) = authority.rsplit_once(':') {
            let port: u16 = p
                .parse()
                .map_err(|_| NetError::invalid_url(format!("invalid port in URL: {url}")))?;
            (h.to_owned(), port)
        } else {
            let default_port = match scheme {
                "https" => 443,
                "http" => 80,
                _ => {
                    return Err(NetError::invalid_url(format!(
                        "unknown scheme '{}' in URL: {}",
                        scheme, url
                    )));
                }
            };
            (authority.to_owned(), default_port)
        };

        Ok(Self {
            scheme: scheme.to_ascii_lowercase(),
            host: host_part.to_ascii_lowercase(),
            port,
        })
    }

    /// Returns a display string in the form `scheme://host:port`.
    #[must_use]
    pub fn display(&self) -> String {
        format!("{}://{}:{}", self.scheme, self.host, self.port)
    }
}

// ─── Connection Descriptor ─────────────────────────────────────────────────────

/// A lightweight descriptor for an HTTP connection being managed by the pool.
///
/// The descriptor does not hold an actual socket — the caller is responsible
/// for the real TCP/TLS connection.  The descriptor carries the metadata needed
/// for keep-alive decisions.
#[derive(Debug, Clone)]
pub struct ConnDescriptor {
    /// Unique connection identifier within the pool.
    pub id: u64,
    /// Host this connection belongs to.
    pub key: HostKey,
    /// How many HTTP requests have been sent on this connection.
    pub request_count: u32,
    /// Maximum number of requests before the connection is retired
    /// (aligns with typical server `keep-alive max` settings).
    pub max_requests: u32,
    /// When this descriptor was first created.
    pub created_at: Instant,
    /// When this descriptor was last used (request sent / response received).
    pub last_used: Instant,
    /// Whether the peer signalled `Connection: close` on the last response.
    pub peer_close_requested: bool,
}

impl ConnDescriptor {
    /// Creates a new descriptor with a unique id.
    #[must_use]
    pub fn new(id: u64, key: HostKey, max_requests: u32) -> Self {
        let now = Instant::now();
        Self {
            id,
            key,
            request_count: 0,
            max_requests,
            created_at: now,
            last_used: now,
            peer_close_requested: false,
        }
    }

    /// Returns `true` if this connection is still eligible for reuse.
    #[must_use]
    pub fn is_reusable(&self, idle_timeout: Duration) -> bool {
        if self.peer_close_requested {
            return false;
        }
        if self.request_count >= self.max_requests {
            return false;
        }
        self.last_used.elapsed() < idle_timeout
    }

    /// Marks one more request having been issued on this connection.
    pub fn record_request(&mut self) {
        self.request_count = self.request_count.saturating_add(1);
        self.last_used = Instant::now();
    }
}

// ─── Pool Configuration ────────────────────────────────────────────────────────

/// Configuration for [`CdnKeepalivePool`].
#[derive(Debug, Clone)]
pub struct KeepalivePoolConfig {
    /// Maximum idle connections stored per host.
    pub max_idle_per_host: usize,
    /// Maximum total connections across all hosts.
    pub max_total_connections: usize,
    /// How long an idle connection is kept before being discarded.
    pub idle_timeout: Duration,
    /// Maximum requests per connection before it is retired.
    pub max_requests_per_conn: u32,
    /// Whether to prefer reusing an existing connection over opening a new one.
    pub prefer_reuse: bool,
}

impl Default for KeepalivePoolConfig {
    fn default() -> Self {
        Self {
            max_idle_per_host: 8,
            max_total_connections: 64,
            idle_timeout: Duration::from_secs(90),
            max_requests_per_conn: 1000,
            prefer_reuse: true,
        }
    }
}

impl KeepalivePoolConfig {
    /// Creates a conservative config suitable for low-traffic origin servers.
    #[must_use]
    pub fn conservative() -> Self {
        Self {
            max_idle_per_host: 2,
            max_total_connections: 16,
            idle_timeout: Duration::from_secs(30),
            max_requests_per_conn: 200,
            prefer_reuse: true,
        }
    }

    /// Creates a high-throughput config for CDN edge nodes serving many clients.
    #[must_use]
    pub fn high_throughput() -> Self {
        Self {
            max_idle_per_host: 32,
            max_total_connections: 256,
            idle_timeout: Duration::from_secs(120),
            max_requests_per_conn: 5000,
            prefer_reuse: true,
        }
    }
}

// ─── Pool Statistics ──────────────────────────────────────────────────────────

/// Aggregate pool statistics.
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    /// Total connections ever created by the pool.
    pub connections_created: u64,
    /// Total connections reused (cache hit).
    pub connections_reused: u64,
    /// Total connections discarded (idle timeout, max-requests, peer close).
    pub connections_discarded: u64,
    /// Current number of idle descriptors across all hosts.
    pub idle_count: usize,
    /// Current number of active (checked-out) descriptors.
    pub active_count: usize,
}

impl PoolStats {
    /// Returns the cache-hit ratio (0.0–1.0).
    #[must_use]
    pub fn reuse_ratio(&self) -> f64 {
        let total = self.connections_created + self.connections_reused;
        if total == 0 {
            return 0.0;
        }
        self.connections_reused as f64 / total as f64
    }
}

// ─── Pool Inner ───────────────────────────────────────────────────────────────

/// Shared inner state of the pool.
struct PoolInner {
    /// Idle connection descriptors keyed by host.
    idle: HashMap<HostKey, VecDeque<ConnDescriptor>>,
    /// Count of active (checked-out) connections.
    active_count: usize,
    /// Running statistics.
    stats: PoolStats,
    /// Monotonically increasing connection id counter.
    next_id: u64,
    /// Pool configuration (copied in for convenient access).
    config: KeepalivePoolConfig,
}

impl PoolInner {
    fn new(config: KeepalivePoolConfig) -> Self {
        Self {
            idle: HashMap::new(),
            active_count: 0,
            stats: PoolStats::default(),
            next_id: 1,
            config,
        }
    }

    /// Total idle connections across all hosts.
    fn total_idle(&self) -> usize {
        self.idle.values().map(VecDeque::len).sum()
    }

    /// Attempts to acquire a reusable descriptor for `key`.
    ///
    /// Returns `None` if no suitable idle descriptor is found.
    fn acquire_idle(&mut self, key: &HostKey) -> Option<ConnDescriptor> {
        let idle_timeout = self.config.idle_timeout;
        let queue = self.idle.get_mut(key)?;

        // Drain stale entries from the front.
        while let Some(front) = queue.front() {
            if front.is_reusable(idle_timeout) {
                break;
            }
            queue.pop_front();
            self.stats.connections_discarded += 1;
        }

        if let Some(mut desc) = queue.pop_front() {
            desc.record_request();
            self.active_count += 1;
            self.stats.connections_reused += 1;
            self.stats.idle_count = self.total_idle();
            self.stats.active_count = self.active_count;
            Some(desc)
        } else {
            None
        }
    }

    /// Creates a new descriptor for `key`.
    ///
    /// # Errors
    ///
    /// Returns [`NetError::Buffer`] if the total connection limit would be exceeded.
    fn create_new(&mut self, key: HostKey) -> NetResult<ConnDescriptor> {
        let total = self.total_idle() + self.active_count;
        if total >= self.config.max_total_connections {
            return Err(NetError::buffer(format!(
                "keep-alive pool exhausted: {} connections (max {})",
                total, self.config.max_total_connections
            )));
        }
        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);
        let mut desc = ConnDescriptor::new(id, key, self.config.max_requests_per_conn);
        desc.record_request();
        self.active_count += 1;
        self.stats.connections_created += 1;
        self.stats.active_count = self.active_count;
        Ok(desc)
    }

    /// Returns a descriptor back to the idle pool after use.
    fn release(&mut self, mut desc: ConnDescriptor) {
        self.active_count = self.active_count.saturating_sub(1);

        // Do not return connections that should not be reused.
        if !desc.is_reusable(self.config.idle_timeout) {
            self.stats.connections_discarded += 1;
            self.stats.active_count = self.active_count;
            self.stats.idle_count = self.total_idle();
            return;
        }

        // Update last_used before returning to idle.
        desc.last_used = Instant::now();
        let key = desc.key.clone();
        let queue = self.idle.entry(key).or_default();

        if queue.len() < self.config.max_idle_per_host {
            queue.push_back(desc);
            self.stats.idle_count = self.total_idle();
        } else {
            // Pool full for this host — discard.
            self.stats.connections_discarded += 1;
        }
        self.stats.active_count = self.active_count;
    }

    /// Removes all idle connections for `key`.
    fn evict_host(&mut self, key: &HostKey) {
        if let Some(queue) = self.idle.remove(key) {
            self.stats.connections_discarded += queue.len() as u64;
        }
        self.stats.idle_count = self.total_idle();
    }

    /// Removes all idle connections older than the configured idle timeout.
    fn purge_stale(&mut self) {
        let timeout = self.config.idle_timeout;
        let mut discarded: u64 = 0;
        for queue in self.idle.values_mut() {
            let before = queue.len();
            queue.retain(|d| d.last_used.elapsed() < timeout);
            discarded += (before - queue.len()) as u64;
        }
        self.stats.connections_discarded += discarded;
        self.stats.idle_count = self.total_idle();
    }
}

// ─── Public Pool ─────────────────────────────────────────────────────────────

/// A thread-safe HTTP keep-alive connection descriptor pool for CDN edge servers.
///
/// Callers acquire a [`ConnDescriptor`] via [`CdnKeepalivePool::acquire`], use the
/// connection, then return it via [`CdnKeepalivePool::release`].
///
/// If `peer_close_requested` is set on the returned descriptor, the connection
/// should be closed rather than returned to the pool.
///
/// # Example
///
/// ```
/// use oximedia_net::cdn::keepalive_pool::{CdnKeepalivePool, KeepalivePoolConfig, HostKey};
///
/// let pool = CdnKeepalivePool::new(KeepalivePoolConfig::default());
/// let key = HostKey::from_url("https://cdn.example.com/segments/seg1.ts").unwrap();
/// let desc = pool.acquire(key.clone()).unwrap();
/// // … use the connection …
/// pool.release(desc);
///
/// let stats = pool.stats();
/// assert_eq!(stats.connections_created, 1);
/// ```
#[derive(Clone)]
pub struct CdnKeepalivePool {
    inner: Arc<Mutex<PoolInner>>,
}

impl CdnKeepalivePool {
    /// Creates a new pool with the given configuration.
    #[must_use]
    pub fn new(config: KeepalivePoolConfig) -> Self {
        Self {
            inner: Arc::new(Mutex::new(PoolInner::new(config))),
        }
    }

    /// Creates a pool with default configuration.
    #[must_use]
    pub fn default_pool() -> Self {
        Self::new(KeepalivePoolConfig::default())
    }

    /// Acquires a connection descriptor for `key`.
    ///
    /// If an idle, reusable descriptor exists it is returned (cache hit).
    /// Otherwise a new descriptor is created (cache miss), provided the
    /// pool has not reached its total connection limit.
    ///
    /// # Errors
    ///
    /// Returns [`NetError::Buffer`] if the total connection limit is reached.
    pub fn acquire(&self, key: HostKey) -> NetResult<ConnDescriptor> {
        let mut inner = self.inner.lock();

        if inner.config.prefer_reuse {
            if let Some(desc) = inner.acquire_idle(&key) {
                return Ok(desc);
            }
        }

        inner.create_new(key)
    }

    /// Returns a descriptor to the idle pool.
    ///
    /// If the descriptor should not be reused (e.g., `peer_close_requested`
    /// is set, or it has reached `max_requests`) it is silently discarded.
    pub fn release(&self, desc: ConnDescriptor) {
        self.inner.lock().release(desc);
    }

    /// Removes all idle connections for the given host key.
    ///
    /// Useful when a CDN provider is detected as unhealthy and all connections
    /// to it must be invalidated.
    pub fn evict_host(&self, key: &HostKey) {
        self.inner.lock().evict_host(key);
    }

    /// Removes all idle connections that have exceeded the idle timeout.
    ///
    /// This should be called periodically (e.g., every 30 seconds) from a
    /// background task to prevent stale descriptor accumulation.
    pub fn purge_stale(&self) {
        self.inner.lock().purge_stale();
    }

    /// Returns a snapshot of current pool statistics.
    #[must_use]
    pub fn stats(&self) -> PoolStats {
        self.inner.lock().stats.clone()
    }

    /// Returns the current number of idle descriptors.
    #[must_use]
    pub fn idle_count(&self) -> usize {
        self.inner.lock().total_idle()
    }

    /// Returns the current number of checked-out (active) descriptors.
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.inner.lock().active_count
    }
}

impl std::fmt::Debug for CdnKeepalivePool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = self.inner.lock();
        f.debug_struct("CdnKeepalivePool")
            .field("idle", &inner.total_idle())
            .field("active", &inner.active_count)
            .finish()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pool() -> CdnKeepalivePool {
        CdnKeepalivePool::new(KeepalivePoolConfig {
            max_idle_per_host: 4,
            max_total_connections: 10,
            idle_timeout: Duration::from_secs(60),
            max_requests_per_conn: 100,
            prefer_reuse: true,
        })
    }

    #[test]
    fn test_host_key_from_url_https() {
        let key = HostKey::from_url("https://cdn.example.com/path/seg.ts").unwrap();
        assert_eq!(key.scheme, "https");
        assert_eq!(key.host, "cdn.example.com");
        assert_eq!(key.port, 443);
    }

    #[test]
    fn test_host_key_from_url_http_custom_port() {
        let key = HostKey::from_url("http://origin.example.com:8080/live/").unwrap();
        assert_eq!(key.scheme, "http");
        assert_eq!(key.port, 8080);
    }

    #[test]
    fn test_host_key_from_url_invalid_scheme() {
        let err = HostKey::from_url("ftp://files.example.com/data").unwrap_err();
        assert!(matches!(err, NetError::InvalidUrl(_)));
    }

    #[test]
    fn test_host_key_from_url_missing_scheme() {
        let err = HostKey::from_url("no-scheme-here").unwrap_err();
        assert!(matches!(err, NetError::InvalidUrl(_)));
    }

    #[test]
    fn test_acquire_creates_new_descriptor() {
        let pool = make_pool();
        let key = HostKey::from_url("https://cdn1.example.com/seg.ts").unwrap();
        let desc = pool.acquire(key).unwrap();
        assert_eq!(desc.request_count, 1);
        assert_eq!(pool.active_count(), 1);
        assert_eq!(pool.idle_count(), 0);
    }

    #[test]
    fn test_release_and_reuse() {
        let pool = make_pool();
        let key = HostKey::from_url("https://cdn1.example.com/seg.ts").unwrap();

        let desc = pool.acquire(key.clone()).unwrap();
        let id = desc.id;
        pool.release(desc);

        assert_eq!(pool.idle_count(), 1);
        assert_eq!(pool.active_count(), 0);

        // Acquiring again should reuse the same descriptor.
        let desc2 = pool.acquire(key).unwrap();
        assert_eq!(desc2.id, id, "should reuse the same descriptor");
        assert_eq!(desc2.request_count, 2);

        let stats = pool.stats();
        assert_eq!(stats.connections_created, 1);
        assert_eq!(stats.connections_reused, 1);
    }

    #[test]
    fn test_pool_exhaustion_returns_error() {
        let pool = CdnKeepalivePool::new(KeepalivePoolConfig {
            max_idle_per_host: 1,
            max_total_connections: 2,
            idle_timeout: Duration::from_secs(60),
            max_requests_per_conn: 100,
            prefer_reuse: true,
        });

        let key = HostKey::from_url("https://cdn.example.com/seg.ts").unwrap();
        let _d1 = pool.acquire(key.clone()).unwrap();
        let _d2 = pool.acquire(key.clone()).unwrap();
        // Third acquire should fail.
        let err = pool.acquire(key).unwrap_err();
        assert!(matches!(err, NetError::Buffer(_)));
    }

    #[test]
    fn test_peer_close_not_returned_to_pool() {
        let pool = make_pool();
        let key = HostKey::from_url("https://cdn.example.com/seg.ts").unwrap();
        let mut desc = pool.acquire(key).unwrap();
        desc.peer_close_requested = true;
        pool.release(desc);

        // Should not appear in idle pool.
        assert_eq!(pool.idle_count(), 0);
        let stats = pool.stats();
        assert_eq!(stats.connections_discarded, 1);
    }

    #[test]
    fn test_evict_host_clears_idle() {
        let pool = make_pool();
        let key = HostKey::from_url("https://cdn.example.com/seg.ts").unwrap();
        let desc = pool.acquire(key.clone()).unwrap();
        pool.release(desc);
        assert_eq!(pool.idle_count(), 1);

        pool.evict_host(&key);
        assert_eq!(pool.idle_count(), 0);
    }

    #[test]
    fn test_max_idle_per_host_enforced() {
        let pool = CdnKeepalivePool::new(KeepalivePoolConfig {
            max_idle_per_host: 2,
            max_total_connections: 20,
            idle_timeout: Duration::from_secs(60),
            max_requests_per_conn: 100,
            prefer_reuse: false, // force new connections
        });

        let key = HostKey::from_url("https://cdn.example.com/seg.ts").unwrap();
        let d1 = pool.acquire(key.clone()).unwrap();
        let d2 = pool.acquire(key.clone()).unwrap();
        let d3 = pool.acquire(key.clone()).unwrap();

        // Return all three; only 2 (max_idle_per_host) should remain.
        pool.release(d1);
        pool.release(d2);
        pool.release(d3);

        assert_eq!(pool.idle_count(), 2);
        let stats = pool.stats();
        assert_eq!(stats.connections_discarded, 1);
    }

    #[test]
    fn test_purge_stale_removes_expired() {
        let pool = CdnKeepalivePool::new(KeepalivePoolConfig {
            max_idle_per_host: 4,
            max_total_connections: 10,
            // Zero timeout so all idle connections are immediately stale.
            idle_timeout: Duration::ZERO,
            max_requests_per_conn: 100,
            prefer_reuse: true,
        });

        let key = HostKey::from_url("https://cdn.example.com/seg.ts").unwrap();
        let desc = pool.acquire(key).unwrap();
        pool.release(desc);

        // After purge the idle slot should be gone.
        pool.purge_stale();
        assert_eq!(pool.idle_count(), 0);
    }

    #[test]
    fn test_stats_reuse_ratio() {
        let pool = make_pool();
        let key = HostKey::from_url("https://cdn.example.com/seg.ts").unwrap();

        // Create, release, reuse.
        let d = pool.acquire(key.clone()).unwrap();
        pool.release(d);
        let _d2 = pool.acquire(key).unwrap();

        let stats = pool.stats();
        // 1 created + 1 reused → 50 % reuse ratio.
        let ratio = stats.reuse_ratio();
        assert!((ratio - 0.5_f64).abs() < 1e-9);
    }

    #[test]
    fn test_conn_descriptor_is_reusable() {
        let key = HostKey::from_url("https://cdn.example.com").unwrap();
        let mut desc = ConnDescriptor::new(1, key, 10);
        desc.record_request();

        assert!(desc.is_reusable(Duration::from_secs(60)));

        // Exhaust request limit.
        desc.max_requests = 1;
        assert!(!desc.is_reusable(Duration::from_secs(60)));
    }

    #[test]
    fn test_host_key_display() {
        let key = HostKey::from_url("https://cdn.example.com/path").unwrap();
        assert_eq!(key.display(), "https://cdn.example.com:443");
    }
}
