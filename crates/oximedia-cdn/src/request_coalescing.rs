//! Request coalescing — deduplicate concurrent origin fetches for the same
//! resource, collapsing multiple simultaneous cache-miss requests into a single
//! upstream call and broadcasting the response to all waiting callers.
//!
//! # Overview
//!
//! When many clients simultaneously request content that is not yet in cache, a
//! CDN without coalescing would issue one origin fetch per client, overloading
//! the origin and wasting bandwidth.  [`CoalescingRegistry`] tracks in-flight
//! fetches keyed by a normalised request key and returns a
//! [`CoalesceHandle`] that the caller uses to either:
//!
//! - **Lead** the fetch (first waiter) — perform the actual origin request and
//!   call [`CoalesceHandle::complete`] or [`CoalesceHandle::fail`].
//! - **Follow** the fetch (subsequent waiters) — block on the shared result via
//!   [`CoalesceHandle::wait`].
//!
//! The implementation is purely synchronous and lock-based so it works without
//! an async runtime.  Each in-flight slot uses a `std::sync::Condvar` for
//! efficient blocking.
//!
//! # Coalescing key normalisation
//!
//! Keys are normalised to remove query parameters that should not differentiate
//! cache variants (configurable via [`CoalescingConfig::ignored_query_params`]).

use std::collections::HashMap;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use thiserror::Error;

// ─── Error ────────────────────────────────────────────────────────────────────

/// Errors from the coalescing subsystem.
#[derive(Debug, Error, Clone)]
pub enum CoalescingError {
    /// The leader fetch failed — propagated to all followers.
    #[error("origin fetch failed: {0}")]
    FetchFailed(String),
    /// Waiting for a coalesced result timed out.
    #[error("coalescing wait timed out after {0:?}")]
    Timeout(Duration),
    /// The registry is at capacity — no new in-flight slots available.
    #[error("coalescing registry full (capacity {0})")]
    RegistryFull(usize),
}

// ─── CoalescingConfig ─────────────────────────────────────────────────────────

/// Configuration for the coalescing registry.
#[derive(Debug, Clone)]
pub struct CoalescingConfig {
    /// Maximum number of concurrent in-flight coalescing slots.
    pub max_in_flight: usize,
    /// Timeout for followers waiting on a coalesced result.
    pub wait_timeout: Duration,
    /// Query-string parameter names that should be stripped before coalescing.
    /// E.g. `["token", "sig"]` → `"/video.mp4?token=abc"` coalesces with
    /// `"/video.mp4?token=xyz"`.
    pub ignored_query_params: Vec<String>,
}

impl Default for CoalescingConfig {
    fn default() -> Self {
        Self {
            max_in_flight: 1024,
            wait_timeout: Duration::from_secs(30),
            ignored_query_params: Vec::new(),
        }
    }
}

impl CoalescingConfig {
    /// Set the maximum in-flight slots.
    pub fn with_max_in_flight(mut self, n: usize) -> Self {
        self.max_in_flight = n;
        self
    }

    /// Set the follower wait timeout.
    pub fn with_wait_timeout(mut self, timeout: Duration) -> Self {
        self.wait_timeout = timeout;
        self
    }

    /// Add query parameter names to ignore during key normalisation.
    pub fn with_ignored_params(
        mut self,
        params: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        for p in params {
            self.ignored_query_params.push(p.into());
        }
        self
    }
}

// ─── SlotState ────────────────────────────────────────────────────────────────

/// State of an in-flight coalescing slot.
#[derive(Debug, Clone)]
enum SlotState {
    /// Origin fetch is in progress.
    InFlight,
    /// Origin fetch completed successfully — contains the response payload.
    Done(Vec<u8>),
    /// Origin fetch failed — contains the error message.
    Failed(String),
}

/// Shared slot protected by a `Mutex` and signalled via `Condvar`.
struct Slot {
    state: Mutex<SlotState>,
    signal: Condvar,
    /// When this slot was created.
    created_at: Instant,
    /// Number of requests coalesced into this slot (leader + followers).
    coalesced_count: Mutex<usize>,
}

impl Slot {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(SlotState::InFlight),
            signal: Condvar::new(),
            created_at: Instant::now(),
            coalesced_count: Mutex::new(1),
        })
    }
}

// ─── CoalesceRole ─────────────────────────────────────────────────────────────

/// Whether the caller is the leader or a follower for this coalescing slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoalesceRole {
    /// This caller will perform the actual origin fetch.
    Leader,
    /// This caller will wait for the leader to complete the fetch.
    Follower,
}

// ─── CoalesceHandle ──────────────────────────────────────────────────────────

/// Handle returned by [`CoalescingRegistry::acquire`].
///
/// - **Leaders** should call [`complete`](CoalesceHandle::complete) on success
///   or [`fail`](CoalesceHandle::fail) on failure.  Both calls release all
///   waiting followers.
/// - **Followers** should call [`wait`](CoalesceHandle::wait) to block until
///   the leader finishes.
#[derive(Clone)]
pub struct CoalesceHandle {
    /// Normalised key used to index this slot.
    pub key: String,
    /// Role of the caller holding this handle.
    pub role: CoalesceRole,
    slot: Arc<Slot>,
    wait_timeout: Duration,
    registry_ref: Arc<RegistryInner>,
}

impl std::fmt::Debug for CoalesceHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CoalesceHandle")
            .field("key", &self.key)
            .field("role", &self.role)
            .field("wait_timeout", &self.wait_timeout)
            .finish()
    }
}

impl CoalesceHandle {
    /// (Leader) Signal a successful fetch with `payload`.
    ///
    /// All followers waiting on this slot will be woken and receive a clone of
    /// the payload.
    pub fn complete(self, payload: Vec<u8>) {
        {
            let mut state = self.slot.state.lock().unwrap_or_else(|e| e.into_inner());
            *state = SlotState::Done(payload);
        }
        self.slot.signal.notify_all();
        self.registry_ref.remove(&self.key);
    }

    /// (Leader) Signal a fetch failure with a message.
    ///
    /// All waiting followers will be woken and receive `CoalescingError::FetchFailed`.
    pub fn fail(self, reason: impl Into<String>) {
        {
            let mut state = self.slot.state.lock().unwrap_or_else(|e| e.into_inner());
            *state = SlotState::Failed(reason.into());
        }
        self.slot.signal.notify_all();
        self.registry_ref.remove(&self.key);
    }

    /// (Follower) Wait for the leader to complete the fetch.
    ///
    /// Returns the payload on success, or a [`CoalescingError`] on failure or
    /// timeout.
    pub fn wait(self) -> Result<Vec<u8>, CoalescingError> {
        let timeout = self.wait_timeout;
        let deadline = Instant::now() + timeout;

        let mut guard = self.slot.state.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            match &*guard {
                SlotState::Done(payload) => return Ok(payload.clone()),
                SlotState::Failed(reason) => {
                    return Err(CoalescingError::FetchFailed(reason.clone()))
                }
                SlotState::InFlight => {}
            }
            let remaining = deadline
                .checked_duration_since(Instant::now())
                .unwrap_or(Duration::ZERO);
            if remaining.is_zero() {
                return Err(CoalescingError::Timeout(timeout));
            }
            let (new_guard, result) = self
                .slot
                .signal
                .wait_timeout(guard, remaining)
                .unwrap_or_else(|e| e.into_inner());
            guard = new_guard;
            if result.timed_out() {
                // Double-check: the signal might have been missed just at the deadline.
                match &*guard {
                    SlotState::Done(payload) => return Ok(payload.clone()),
                    SlotState::Failed(reason) => {
                        return Err(CoalescingError::FetchFailed(reason.clone()))
                    }
                    SlotState::InFlight => {
                        return Err(CoalescingError::Timeout(timeout));
                    }
                }
            }
        }
    }

    /// Age of the underlying slot since it was created.
    pub fn slot_age(&self) -> Duration {
        self.slot.created_at.elapsed()
    }

    /// Number of requests coalesced into this slot (includes this caller).
    pub fn coalesced_count(&self) -> usize {
        *self.slot.coalesced_count.lock().unwrap_or_else(|e| e.into_inner())
    }
}

// ─── RegistryInner ───────────────────────────────────────────────────────────

struct RegistryInner {
    slots: Mutex<HashMap<String, Arc<Slot>>>,
    max_in_flight: usize,
}

impl RegistryInner {
    fn remove(&self, key: &str) {
        if let Ok(mut slots) = self.slots.lock() {
            slots.remove(key);
        }
    }
}

// ─── CoalescingRegistry ──────────────────────────────────────────────────────

/// Thread-safe registry that deduplicates concurrent origin fetches.
pub struct CoalescingRegistry {
    inner: Arc<RegistryInner>,
    wait_timeout: Duration,
    ignored_params: Vec<String>,
}

impl CoalescingRegistry {
    /// Create a new registry with the given configuration.
    pub fn new(config: CoalescingConfig) -> Self {
        Self {
            inner: Arc::new(RegistryInner {
                slots: Mutex::new(HashMap::new()),
                max_in_flight: config.max_in_flight,
            }),
            wait_timeout: config.wait_timeout,
            ignored_params: config.ignored_query_params,
        }
    }

    /// Acquire a coalescing slot for `raw_key`.
    ///
    /// - Returns `(Leader, handle)` if this is the first request for the key.
    /// - Returns `(Follower, handle)` if a fetch is already in progress.
    ///
    /// The caller is responsible for checking the role and acting accordingly.
    pub fn acquire(&self, raw_key: &str) -> Result<CoalesceHandle, CoalescingError> {
        let key = self.normalise(raw_key);
        let mut slots = self.inner.slots.lock().unwrap_or_else(|e| e.into_inner());

        if let Some(existing) = slots.get(&key) {
            // Increment coalesced count.
            if let Ok(mut count) = existing.coalesced_count.lock() {
                *count += 1;
            }
            return Ok(CoalesceHandle {
                key,
                role: CoalesceRole::Follower,
                slot: Arc::clone(existing),
                wait_timeout: self.wait_timeout,
                registry_ref: Arc::clone(&self.inner),
            });
        }

        if slots.len() >= self.inner.max_in_flight {
            return Err(CoalescingError::RegistryFull(self.inner.max_in_flight));
        }

        let slot = Slot::new();
        slots.insert(key.clone(), Arc::clone(&slot));
        Ok(CoalesceHandle {
            key,
            role: CoalesceRole::Leader,
            slot,
            wait_timeout: self.wait_timeout,
            registry_ref: Arc::clone(&self.inner),
        })
    }

    /// Number of currently in-flight coalescing slots.
    pub fn in_flight_count(&self) -> usize {
        self.inner.slots.lock().map(|s| s.len()).unwrap_or(0)
    }

    /// Normalise a request key by stripping ignored query parameters.
    ///
    /// Splits on `?` and removes known-ignorable params from the query string.
    /// The remaining params are sorted for deterministic key generation.
    pub fn normalise(&self, raw_key: &str) -> String {
        if self.ignored_params.is_empty() {
            return raw_key.to_string();
        }
        match raw_key.split_once('?') {
            None => raw_key.to_string(),
            Some((path, query)) => {
                let mut params: Vec<&str> = query
                    .split('&')
                    .filter(|kv| {
                        let name = kv.split('=').next().unwrap_or(*kv);
                        !self.ignored_params.iter().any(|ip| ip == name)
                    })
                    .collect();
                params.sort_unstable();
                if params.is_empty() {
                    path.to_string()
                } else {
                    format!("{}?{}", path, params.join("&"))
                }
            }
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    // 1. First acquire is Leader
    #[test]
    fn test_first_acquire_is_leader() {
        let registry = CoalescingRegistry::new(CoalescingConfig::default());
        let handle = registry.acquire("/video.mp4").expect("handle");
        assert_eq!(handle.role, CoalesceRole::Leader);
        assert_eq!(registry.in_flight_count(), 1);
        // Clean up.
        handle.complete(b"ok".to_vec());
    }

    // 2. Second acquire for same key is Follower
    #[test]
    fn test_second_acquire_is_follower() {
        let registry = CoalescingRegistry::new(CoalescingConfig::default());
        let leader = registry.acquire("/video.mp4").expect("leader");
        let follower = registry.acquire("/video.mp4").expect("follower");
        assert_eq!(follower.role, CoalesceRole::Follower);
        leader.complete(b"data".to_vec());
    }

    // 3. complete delivers payload to follower
    #[test]
    fn test_complete_delivers_payload() {
        let registry = Arc::new(CoalescingRegistry::new(CoalescingConfig::default()));
        let reg_clone = Arc::clone(&registry);
        let leader_handle = registry.acquire("/segment.ts").expect("leader");
        let follower_handle = registry.acquire("/segment.ts").expect("follower");

        let payload = b"segment data".to_vec();
        let payload_clone = payload.clone();

        let t = thread::spawn(move || follower_handle.wait().expect("payload"));

        // Give the thread a moment to block.
        thread::sleep(Duration::from_millis(5));
        leader_handle.complete(payload_clone);
        let received = t.join().expect("thread ok");
        drop(reg_clone);
        assert_eq!(received, payload);
    }

    // 4. fail propagates error to follower
    #[test]
    fn test_fail_propagates_error() {
        let registry = Arc::new(CoalescingRegistry::new(CoalescingConfig::default()));
        let leader = registry.acquire("/bad.mp4").expect("leader");
        let follower = registry.acquire("/bad.mp4").expect("follower");

        let t = thread::spawn(move || follower.wait());
        thread::sleep(Duration::from_millis(5));
        leader.fail("origin 503");
        let result = t.join().expect("thread ok");
        assert!(matches!(result, Err(CoalescingError::FetchFailed(_))));
    }

    // 5. Slot removed after completion
    #[test]
    fn test_slot_removed_after_complete() {
        let registry = CoalescingRegistry::new(CoalescingConfig::default());
        let handle = registry.acquire("/x.mp4").expect("leader");
        assert_eq!(registry.in_flight_count(), 1);
        handle.complete(vec![]);
        assert_eq!(registry.in_flight_count(), 0);
    }

    // 6. Slot removed after failure
    #[test]
    fn test_slot_removed_after_fail() {
        let registry = CoalescingRegistry::new(CoalescingConfig::default());
        let handle = registry.acquire("/x.mp4").expect("leader");
        handle.fail("err");
        assert_eq!(registry.in_flight_count(), 0);
    }

    // 7. coalesced_count increments for each waiter
    #[test]
    fn test_coalesced_count() {
        let registry = CoalescingRegistry::new(CoalescingConfig::default());
        let leader = registry.acquire("/v.mp4").expect("leader");
        assert_eq!(leader.coalesced_count(), 1);
        let _f1 = registry.acquire("/v.mp4").expect("follower 1");
        assert_eq!(leader.coalesced_count(), 2);
        let _f2 = registry.acquire("/v.mp4").expect("follower 2");
        assert_eq!(leader.coalesced_count(), 3);
        leader.complete(vec![]);
    }

    // 8. RegistryFull error when at capacity
    #[test]
    fn test_registry_full_error() {
        let config = CoalescingConfig::default().with_max_in_flight(2);
        let registry = CoalescingRegistry::new(config);
        let h1 = registry.acquire("/a").expect("slot 1");
        let h2 = registry.acquire("/b").expect("slot 2");
        let err = registry.acquire("/c").unwrap_err();
        assert!(matches!(err, CoalescingError::RegistryFull(2)));
        h1.complete(vec![]);
        h2.complete(vec![]);
    }

    // 9. Normalise strips ignored params
    #[test]
    fn test_normalise_strips_ignored_params() {
        let config = CoalescingConfig::default().with_ignored_params(["token", "sig"]);
        let registry = CoalescingRegistry::new(config);
        let k1 = registry.normalise("/video.mp4?token=abc&quality=720");
        let k2 = registry.normalise("/video.mp4?token=xyz&quality=720");
        assert_eq!(k1, k2, "different tokens should normalise to same key");
    }

    // 10. Normalise keeps non-ignored params
    #[test]
    fn test_normalise_keeps_other_params() {
        let config = CoalescingConfig::default().with_ignored_params(["sig"]);
        let registry = CoalescingRegistry::new(config);
        let k1 = registry.normalise("/v.mp4?quality=720&sig=abc");
        let k2 = registry.normalise("/v.mp4?quality=1080&sig=abc");
        assert_ne!(k1, k2, "different quality should produce different keys");
    }

    // 11. Normalise: no query string
    #[test]
    fn test_normalise_no_query() {
        let registry = CoalescingRegistry::new(CoalescingConfig::default());
        let k = registry.normalise("/plain/path");
        assert_eq!(k, "/plain/path");
    }

    // 12. Normalise: all params stripped → no trailing ?
    #[test]
    fn test_normalise_all_params_stripped() {
        let config = CoalescingConfig::default().with_ignored_params(["token"]);
        let registry = CoalescingRegistry::new(config);
        let k = registry.normalise("/video.mp4?token=abc");
        assert_eq!(k, "/video.mp4");
    }

    // 13. Wait timeout
    #[test]
    fn test_wait_timeout() {
        let config = CoalescingConfig::default().with_wait_timeout(Duration::from_millis(30));
        let registry = CoalescingRegistry::new(config);
        let _leader = registry.acquire("/slow.mp4").expect("leader");
        let follower = registry.acquire("/slow.mp4").expect("follower");
        // Don't complete the leader — follower should time out.
        let result = follower.wait();
        assert!(
            matches!(result, Err(CoalescingError::Timeout(_))),
            "expected timeout, got: {result:?}"
        );
        // Clean up leader slot manually.
        let _ = _leader;
    }

    // 14. Different keys get independent slots
    #[test]
    fn test_different_keys_independent() {
        let registry = CoalescingRegistry::new(CoalescingConfig::default());
        let h1 = registry.acquire("/a.mp4").expect("h1");
        let h2 = registry.acquire("/b.mp4").expect("h2");
        assert_eq!(h1.role, CoalesceRole::Leader);
        assert_eq!(h2.role, CoalesceRole::Leader);
        assert_eq!(registry.in_flight_count(), 2);
        h1.complete(vec![1]);
        h2.complete(vec![2]);
    }

    // 15. CoalescingConfig builders
    #[test]
    fn test_coalescing_config_builders() {
        let cfg = CoalescingConfig::default()
            .with_max_in_flight(512)
            .with_wait_timeout(Duration::from_secs(10))
            .with_ignored_params(["token"]);
        assert_eq!(cfg.max_in_flight, 512);
        assert_eq!(cfg.wait_timeout, Duration::from_secs(10));
        assert_eq!(cfg.ignored_query_params, vec!["token"]);
    }

    // 16. Multiple followers receive the same payload
    #[test]
    fn test_multiple_followers_same_payload() {
        let registry = Arc::new(CoalescingRegistry::new(CoalescingConfig::default()));
        let leader = registry.acquire("/multi.mp4").expect("leader");
        let f1 = registry.acquire("/multi.mp4").expect("f1");
        let f2 = registry.acquire("/multi.mp4").expect("f2");
        let f3 = registry.acquire("/multi.mp4").expect("f3");

        let payload = b"shared response".to_vec();
        let pc = payload.clone();

        let t1 = thread::spawn(move || f1.wait());
        let t2 = thread::spawn(move || f2.wait());
        let t3 = thread::spawn(move || f3.wait());

        thread::sleep(Duration::from_millis(5));
        leader.complete(pc);

        let r1 = t1.join().expect("t1").expect("r1");
        let r2 = t2.join().expect("t2").expect("r2");
        let r3 = t3.join().expect("t3").expect("r3");
        assert_eq!(r1, payload);
        assert_eq!(r2, payload);
        assert_eq!(r3, payload);
    }

    // 17. slot_age is non-negative
    #[test]
    fn test_slot_age() {
        let registry = CoalescingRegistry::new(CoalescingConfig::default());
        let handle = registry.acquire("/age-test.mp4").expect("handle");
        let age = handle.slot_age();
        assert!(age < Duration::from_secs(5));
        handle.complete(vec![]);
    }

    // 18. After complete, new acquire for same key creates new slot
    #[test]
    fn test_new_slot_after_complete() {
        let registry = CoalescingRegistry::new(CoalescingConfig::default());
        let h1 = registry.acquire("/reuse.mp4").expect("h1");
        h1.complete(b"first".to_vec());

        // New acquire should be a fresh Leader slot.
        let h2 = registry.acquire("/reuse.mp4").expect("h2");
        assert_eq!(h2.role, CoalesceRole::Leader);
        h2.complete(b"second".to_vec());
    }

    // 19. Normalise sorts remaining params deterministically
    #[test]
    fn test_normalise_sorted_params() {
        let config = CoalescingConfig::default().with_ignored_params(["sig"]);
        let registry = CoalescingRegistry::new(config);
        let k1 = registry.normalise("/v.mp4?b=2&a=1");
        let k2 = registry.normalise("/v.mp4?a=1&b=2");
        assert_eq!(k1, k2, "param order should not matter after normalisation");
    }

    // 20. in_flight_count reflects active slots only
    #[test]
    fn test_in_flight_count() {
        let registry = CoalescingRegistry::new(CoalescingConfig::default());
        assert_eq!(registry.in_flight_count(), 0);
        let h1 = registry.acquire("/a").expect("h1");
        let h2 = registry.acquire("/b").expect("h2");
        assert_eq!(registry.in_flight_count(), 2);
        h1.fail("err");
        assert_eq!(registry.in_flight_count(), 1);
        h2.complete(vec![]);
        assert_eq!(registry.in_flight_count(), 0);
    }
}
