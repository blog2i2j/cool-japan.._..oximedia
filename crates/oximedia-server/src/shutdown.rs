//! Graceful server shutdown coordination.
//!
//! Provides a [`ShutdownSignal`](crate::shutdown::ShutdownSignal) that allows one task to signal shutdown to
//! others and wait for all in-flight connections to drain before the process
//! exits.
//!
//! The implementation is entirely synchronous / CPU-bound so it can be used
//! from both async (`tokio`) and blocking contexts.  Real production code
//! would typically replace the polling loop with `tokio::select!` or a
//! `tokio::sync::Notify`, but this CPU-stub version keeps the crate
//! dependency surface minimal.
//!
//! # Example
//!
//! ```rust
//! use oximedia_server::shutdown::ShutdownSignal;
//!
//! let mut sig = ShutdownSignal::new();
//! assert!(!sig.is_signalled());
//! sig.signal();
//! assert!(sig.is_signalled());
//! // Simulate zero active connections → wait_for_idle returns immediately.
//! assert!(sig.wait_for_idle(0, 1000));
//! ```

#![allow(dead_code)]

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

// ── ShutdownSignal ────────────────────────────────────────────────────────────

/// Shared shutdown state, cheaply cloneable across threads.
#[derive(Clone)]
pub struct ShutdownSignal {
    /// Set to `true` when shutdown has been requested.
    signalled: Arc<AtomicBool>,
    /// Number of active connections tracked externally.
    active_connections: Arc<AtomicU64>,
}

impl std::fmt::Debug for ShutdownSignal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShutdownSignal")
            .field("signalled", &self.signalled.load(Ordering::Relaxed))
            .field(
                "active_connections",
                &self.active_connections.load(Ordering::Relaxed),
            )
            .finish()
    }
}

impl ShutdownSignal {
    /// Create a new, un-signalled shutdown signal.
    #[must_use]
    pub fn new() -> Self {
        Self {
            signalled: Arc::new(AtomicBool::new(false)),
            active_connections: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Send the shutdown signal.
    ///
    /// All clones of this signal will observe `is_signalled() == true` after
    /// this call.
    pub fn signal(&self) {
        self.signalled.store(true, Ordering::Release);
    }

    /// Returns `true` if shutdown has been requested.
    #[must_use]
    pub fn is_signalled(&self) -> bool {
        self.signalled.load(Ordering::Acquire)
    }

    /// Set the number of active connections on the shared counter.
    ///
    /// In production, each accepted connection would increment this on
    /// arrival and decrement on close.  For testing, the caller can set
    /// it directly.
    pub fn set_active_connections(&self, count: u64) {
        self.active_connections.store(count, Ordering::Release);
    }

    /// Decrement the active connection counter by one (saturating at zero).
    pub fn connection_closed(&self) {
        // Saturating decrement: load, compute max(v-1, 0), CAS.
        let mut cur = self.active_connections.load(Ordering::Acquire);
        loop {
            if cur == 0 {
                return;
            }
            match self.active_connections.compare_exchange_weak(
                cur,
                cur - 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return,
                Err(actual) => cur = actual,
            }
        }
    }

    /// Increment the active connection counter by one.
    pub fn connection_accepted(&self) {
        self.active_connections.fetch_add(1, Ordering::AcqRel);
    }

    /// Return the current active connection count.
    #[must_use]
    pub fn active_count(&self) -> u64 {
        self.active_connections.load(Ordering::Acquire)
    }

    /// Wait until `active_connections` drops to zero or `timeout_ms`
    /// milliseconds elapse.
    ///
    /// * `active_connections` – Snapshot of the caller's connection count at
    ///   the time of the call (used as an override if > 0; the internal atomic
    ///   is used otherwise).
    /// * `timeout_ms`         – Maximum wait time in milliseconds.
    ///
    /// Returns `true` if all connections drained within the timeout, `false`
    /// if the timeout was reached with connections still active.
    pub fn wait_for_idle(&self, active_connections: u64, timeout_ms: u64) -> bool {
        // Use the caller-supplied count if it overrides our internal counter.
        if active_connections == 0 && self.active_count() == 0 {
            return true;
        }

        let deadline = Instant::now() + Duration::from_millis(timeout_ms);
        let poll_interval = Duration::from_millis(1);

        loop {
            let remaining = active_connections.saturating_add(self.active_count());
            if remaining == 0 {
                return true;
            }
            if Instant::now() >= deadline {
                return false;
            }
            std::thread::sleep(poll_interval.min(Duration::from_millis(timeout_ms / 10 + 1)));
        }
    }

    /// Reset the signal and clear the connection counter (useful in tests).
    pub fn reset(&self) {
        self.signalled.store(false, Ordering::Release);
        self.active_connections.store(0, Ordering::Release);
    }
}

impl Default for ShutdownSignal {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_not_signalled() {
        let sig = ShutdownSignal::new();
        assert!(!sig.is_signalled());
    }

    #[test]
    fn test_signal_sets_flag() {
        let sig = ShutdownSignal::new();
        sig.signal();
        assert!(sig.is_signalled());
    }

    #[test]
    fn test_clone_sees_signal() {
        let sig = ShutdownSignal::new();
        let clone = sig.clone();
        sig.signal();
        assert!(clone.is_signalled());
    }

    #[test]
    fn test_wait_for_idle_zero_connections_immediate() {
        let sig = ShutdownSignal::new();
        assert!(sig.wait_for_idle(0, 100));
    }

    #[test]
    fn test_wait_for_idle_timeout_with_connections() {
        let sig = ShutdownSignal::new();
        sig.set_active_connections(5);
        // 10 ms timeout, connections never drop → should return false.
        let result = sig.wait_for_idle(5, 10);
        assert!(!result);
    }

    #[test]
    fn test_connection_accepted_and_closed() {
        let sig = ShutdownSignal::new();
        sig.connection_accepted();
        sig.connection_accepted();
        assert_eq!(sig.active_count(), 2);
        sig.connection_closed();
        assert_eq!(sig.active_count(), 1);
    }

    #[test]
    fn test_connection_closed_saturating() {
        let sig = ShutdownSignal::new();
        // Should not underflow.
        sig.connection_closed();
        assert_eq!(sig.active_count(), 0);
    }

    #[test]
    fn test_reset_clears_state() {
        let sig = ShutdownSignal::new();
        sig.signal();
        sig.set_active_connections(10);
        sig.reset();
        assert!(!sig.is_signalled());
        assert_eq!(sig.active_count(), 0);
    }

    #[test]
    fn test_set_active_connections() {
        let sig = ShutdownSignal::new();
        sig.set_active_connections(42);
        assert_eq!(sig.active_count(), 42);
    }

    #[test]
    fn test_debug_format() {
        let sig = ShutdownSignal::new();
        let s = format!("{sig:?}");
        assert!(s.contains("ShutdownSignal"), "debug={s}");
    }
}
