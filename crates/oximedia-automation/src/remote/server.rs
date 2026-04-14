//! Remote control server.

use crate::remote::api::ApiRouter;
use crate::remote::websocket::WebSocketHandler;
use crate::Result;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::RwLock;
use tracing::{info, warn};

// ─────────────────────────────────────────────────────────────────────────────
// HTTP session pool
// ─────────────────────────────────────────────────────────────────────────────

/// A pooled HTTP session handle.
///
/// Each session tracks its identity, creation time, and the number of
/// requests it has served.  The network layer is simulated — no actual
/// connections are created so the pool can be exercised in unit tests.
#[derive(Debug)]
pub struct HttpSession {
    /// Unique session identifier.
    pub id: u64,
    /// Monotonic instant at which this session was created.
    pub created_at: std::time::Instant,
    /// Number of HTTP requests served by this session.
    pub requests_served: u64,
}

/// Connection pool for HTTP sessions.
///
/// Maintains a bounded queue of idle sessions.  When a session is acquired
/// the pool returns an existing idle session (a "reuse") if one is available,
/// otherwise it creates a fresh session.  When a session is released it is
/// returned to the idle queue unless it has exceeded its maximum lifetime.
///
/// All operations on the pool are `O(1)` amortised and do **not** require an
/// async runtime — the internal lock is a `std::sync::Mutex` so the pool can
/// be used from both synchronous and asynchronous contexts.
pub struct HttpSessionPool {
    idle: Mutex<VecDeque<HttpSession>>,
    max_idle: usize,
    max_lifetime_secs: u64,
    next_id: AtomicU64,
    total_created: AtomicU64,
    total_reused: AtomicU64,
}

impl HttpSessionPool {
    /// Create a new pool.
    ///
    /// * `max_idle` — maximum number of sessions to keep in the idle queue.
    /// * `max_lifetime_secs` — sessions older than this are discarded on
    ///   release rather than being returned to the pool.
    pub fn new(max_idle: usize, max_lifetime_secs: u64) -> Self {
        Self {
            idle: Mutex::new(VecDeque::with_capacity(max_idle)),
            max_idle,
            max_lifetime_secs,
            next_id: AtomicU64::new(1),
            total_created: AtomicU64::new(0),
            total_reused: AtomicU64::new(0),
        }
    }

    /// Acquire a session from the pool.
    ///
    /// If an unexpired idle session is available it is popped from the queue
    /// and returned (counted as a reuse).  Otherwise a new session is created.
    pub fn acquire(&self) -> HttpSession {
        // Evict any expired entries at the front of the queue.
        let mut idle = self.idle.lock().expect("HttpSessionPool mutex poisoned");
        while let Some(front) = idle.front() {
            let age_secs = front.created_at.elapsed().as_secs();
            if age_secs >= self.max_lifetime_secs {
                idle.pop_front();
            } else {
                break;
            }
        }

        if let Some(session) = idle.pop_front() {
            self.total_reused.fetch_add(1, Ordering::Relaxed);
            session
        } else {
            drop(idle);
            let id = self.next_id.fetch_add(1, Ordering::Relaxed);
            self.total_created.fetch_add(1, Ordering::Relaxed);
            HttpSession {
                id,
                created_at: std::time::Instant::now(),
                requests_served: 0,
            }
        }
    }

    /// Release a session back to the pool.
    ///
    /// The session is discarded if it has exceeded `max_lifetime_secs` or if
    /// the idle queue is already full.
    pub fn release(&self, mut session: HttpSession) {
        let age_secs = session.created_at.elapsed().as_secs();
        if age_secs >= self.max_lifetime_secs {
            // Session expired — discard it.
            return;
        }

        let mut idle = self.idle.lock().expect("HttpSessionPool mutex poisoned");
        if idle.len() < self.max_idle {
            session.requests_served += 1;
            idle.push_back(session);
        }
        // else: pool is full — discard the session.
    }

    /// Return the number of sessions currently sitting idle in the pool.
    pub fn idle_count(&self) -> usize {
        self.idle
            .lock()
            .expect("HttpSessionPool mutex poisoned")
            .len()
    }

    /// Total number of sessions created (new allocations, not reuses).
    pub fn total_created(&self) -> u64 {
        self.total_created.load(Ordering::Relaxed)
    }

    /// Total number of times an existing idle session was reused.
    pub fn total_reused(&self) -> u64 {
        self.total_reused.load(Ordering::Relaxed)
    }

    /// Pool hit rate: `reused / (created + reused)`.
    ///
    /// Returns `0.0` when no sessions have been acquired yet.
    pub fn hit_rate(&self) -> f64 {
        let reused = self.total_reused.load(Ordering::Relaxed);
        let created = self.total_created.load(Ordering::Relaxed);
        let total = created + reused;
        if total == 0 {
            0.0
        } else {
            reused as f64 / total as f64
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Remote server configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Remote server configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteConfig {
    /// Server bind address
    pub bind_address: String,
    /// Server port
    pub port: u16,
    /// Enable authentication
    pub require_auth: bool,
    /// Enable WebSocket
    pub enable_websocket: bool,
}

impl Default for RemoteConfig {
    fn default() -> Self {
        Self {
            bind_address: "0.0.0.0".to_string(),
            port: 8080,
            require_auth: true,
            enable_websocket: true,
        }
    }
}

/// Remote control server.
#[allow(dead_code)]
pub struct RemoteServer {
    config: RemoteConfig,
    api_router: ApiRouter,
    websocket_handler: Option<WebSocketHandler>,
    running: Arc<RwLock<bool>>,
}

impl RemoteServer {
    /// Create a new remote control server.
    pub fn new(config: RemoteConfig) -> Self {
        info!(
            "Creating remote control server at {}:{}",
            config.bind_address, config.port
        );

        let websocket_handler = if config.enable_websocket {
            Some(WebSocketHandler::new())
        } else {
            None
        };

        Self {
            config: config.clone(),
            api_router: ApiRouter::new(config),
            websocket_handler,
            running: Arc::new(RwLock::new(false)),
        }
    }

    /// Start the remote control server.
    pub async fn start(&mut self) -> Result<()> {
        info!("Starting remote control server");

        {
            let mut running = self.running.write().await;
            *running = true;
        }

        // In a real implementation, this would:
        // 1. Start the HTTP/REST API server (using axum)
        // 2. Start the WebSocket server if enabled
        // 3. Set up authentication middleware
        // 4. Configure CORS
        // 5. Start metrics endpoint

        Ok(())
    }

    /// Stop the remote control server.
    pub async fn stop(&mut self) -> Result<()> {
        info!("Stopping remote control server");

        let mut running = self.running.write().await;
        *running = false;

        Ok(())
    }

    /// Check if server is running.
    pub async fn is_running(&self) -> bool {
        *self.running.read().await
    }

    /// Get server address.
    pub fn address(&self) -> String {
        format!("{}:{}", self.config.bind_address, self.config.port)
    }

    /// Broadcast WebSocket message.
    pub async fn broadcast(&self, message: String) -> Result<()> {
        if let Some(ref handler) = self.websocket_handler {
            handler.broadcast(message).await?;
        } else {
            warn!("WebSocket not enabled, cannot broadcast message");
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_remote_server_creation() {
        let config = RemoteConfig::default();
        let server = RemoteServer::new(config);
        assert_eq!(server.address(), "0.0.0.0:8080");
    }

    #[tokio::test]
    async fn test_server_lifecycle() {
        let config = RemoteConfig::default();
        let mut server = RemoteServer::new(config);

        assert!(!server.is_running().await);
        server.start().await.expect("operation should succeed");
        assert!(server.is_running().await);
        server.stop().await.expect("operation should succeed");
        assert!(!server.is_running().await);
    }

    // ─── HttpSessionPool tests ──────────────────────────────────────────────

    #[test]
    fn pool_acquire_creates_new_session_when_empty() {
        let pool = HttpSessionPool::new(4, 60);
        let session = pool.acquire();
        assert!(session.id >= 1, "session id should be non-zero");
        assert_eq!(pool.total_created(), 1);
        assert_eq!(pool.total_reused(), 0);
    }

    #[test]
    fn pool_release_then_acquire_reuses_session() {
        let pool = HttpSessionPool::new(4, 60);
        let s = pool.acquire();
        let first_id = s.id;
        pool.release(s);
        assert_eq!(pool.idle_count(), 1);

        let s2 = pool.acquire();
        assert_eq!(s2.id, first_id, "same session should be reused");
        assert_eq!(pool.total_created(), 1);
        assert_eq!(pool.total_reused(), 1);
        assert_eq!(pool.idle_count(), 0);
    }

    #[test]
    fn pool_idle_count_tracks_correctly() {
        let pool = HttpSessionPool::new(4, 60);
        let s1 = pool.acquire();
        let s2 = pool.acquire();
        assert_eq!(pool.idle_count(), 0);

        pool.release(s1);
        assert_eq!(pool.idle_count(), 1);

        pool.release(s2);
        assert_eq!(pool.idle_count(), 2);
    }

    #[test]
    fn pool_does_not_exceed_max_idle() {
        let pool = HttpSessionPool::new(2, 60);
        let s1 = pool.acquire();
        let s2 = pool.acquire();
        let s3 = pool.acquire();

        pool.release(s1);
        pool.release(s2);
        // Pool is full; s3 should be discarded.
        pool.release(s3);

        assert_eq!(pool.idle_count(), 2, "idle must not exceed max_idle");
    }

    #[test]
    fn pool_expired_session_not_reused() {
        // max_lifetime_secs = 0 means every session expires immediately.
        let pool = HttpSessionPool::new(4, 0);
        let s = pool.acquire();
        pool.release(s);
        // The session should have been discarded since it's already expired.
        assert_eq!(pool.idle_count(), 0, "expired session must not be kept");

        // Next acquire must create a new session.
        let _s2 = pool.acquire();
        assert_eq!(pool.total_created(), 2, "new session must be created");
        assert_eq!(pool.total_reused(), 0);
    }

    #[test]
    fn pool_hit_rate_is_zero_initially() {
        let pool = HttpSessionPool::new(4, 60);
        assert_eq!(pool.hit_rate(), 0.0);
    }

    #[test]
    fn pool_hit_rate_correct_after_mixed_acquisitions() {
        let pool = HttpSessionPool::new(4, 60);
        // 1 create, then reuse 3 times → hit_rate = 3 / (1+3) = 0.75
        let s = pool.acquire(); // created
        pool.release(s);
        let s = pool.acquire(); // reused
        pool.release(s);
        let s = pool.acquire(); // reused
        pool.release(s);
        let s = pool.acquire(); // reused
        pool.release(s);

        let hr = pool.hit_rate();
        assert!(
            (hr - 0.75).abs() < 1e-9,
            "expected hit_rate 0.75 but got {hr}"
        );
    }

    #[test]
    fn pool_requests_served_increments_on_release() {
        let pool = HttpSessionPool::new(4, 60);
        let s = pool.acquire();
        assert_eq!(s.requests_served, 0);
        pool.release(s);

        let s2 = pool.acquire();
        assert_eq!(
            s2.requests_served, 1,
            "requests_served should be 1 after one release"
        );
        pool.release(s2);

        let s3 = pool.acquire();
        assert_eq!(
            s3.requests_served, 2,
            "requests_served should be 2 after two releases"
        );
        let _ = s3; // extend lifetime to end of test scope
    }

    #[test]
    fn pool_concurrent_access_from_multiple_threads() {
        use std::sync::Arc;
        use std::thread;

        let pool = Arc::new(HttpSessionPool::new(8, 120));
        let mut handles = Vec::new();

        for _ in 0..4 {
            let p = Arc::clone(&pool);
            let h = thread::spawn(move || {
                for _ in 0..25 {
                    let s = p.acquire();
                    // Simulate a short amount of work.
                    p.release(s);
                }
            });
            handles.push(h);
        }

        for h in handles {
            h.join().expect("thread should not panic");
        }

        // 4 threads × 25 iterations = 100 total acquires.
        let total = pool.total_created() + pool.total_reused();
        assert_eq!(total, 100, "total acquires should be 100");
    }

    #[test]
    fn pool_total_created_grows_when_idle_empty() {
        let pool = HttpSessionPool::new(1, 60);
        // Acquire 5 without releasing — all must be new creations.
        let sessions: Vec<_> = (0..5).map(|_| pool.acquire()).collect();
        assert_eq!(pool.total_created(), 5);
        assert_eq!(pool.total_reused(), 0);
        drop(sessions);
    }
}
