//! Optimistic locking with version-based conflict detection.
//!
//! Provides [`OptimisticLock`] — a lightweight guard that prevents
//! lost-update anomalies by requiring callers to present the current version
//! before any mutation is accepted.  Concurrent writers each read the current
//! version, compute their change locally, then attempt to commit by supplying
//! the version they read.  If another writer already incremented the version,
//! the update is rejected and the caller must retry.
//!
//! # Example
//!
//! ```
//! use oximedia_collab::opt_lock::OptimisticLock;
//!
//! let mut lock = OptimisticLock::new(0);
//!
//! // First writer — succeeds, version becomes 1.
//! assert!(lock.try_update(0, 1));
//! assert_eq!(lock.version(), 1);
//!
//! // Stale writer — fails because expected version (0) no longer matches.
//! assert!(!lock.try_update(0, 2));
//! assert_eq!(lock.version(), 1);
//! ```

#![allow(dead_code)]

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

// ─────────────────────────────────────────────────────────────────────────────
// OptimisticLock
// ─────────────────────────────────────────────────────────────────────────────

/// A version-stamped optimistic lock.
///
/// Callers read the current [`version`](Self::version), compute their change,
/// then call [`try_update`](Self::try_update) supplying the version they
/// observed.  The update succeeds only when the stored version still matches
/// the expected value; otherwise it returns `false` and the caller should
/// re-read and retry.
///
/// The internal counter is stored in an `Arc<AtomicU64>` so that
/// `OptimisticLock` is `Clone` and multiple handles can observe the same
/// version without sharing a `&mut` reference.
#[derive(Debug, Clone)]
pub struct OptimisticLock {
    inner: Arc<AtomicU64>,
}

impl OptimisticLock {
    /// Create a new `OptimisticLock` initialised to `version`.
    pub fn new(version: u64) -> Self {
        Self {
            inner: Arc::new(AtomicU64::new(version)),
        }
    }

    /// Return the current version.
    ///
    /// Uses `Acquire` ordering so subsequent reads of shared data cannot be
    /// reordered before this load.
    pub fn version(&self) -> u64 {
        self.inner.load(Ordering::Acquire)
    }

    /// Attempt to update the version from `expected_ver` to `new_ver`.
    ///
    /// Returns `true` when the CAS succeeds (i.e., the stored version equalled
    /// `expected_ver` at the moment of the swap and has been atomically
    /// replaced with `new_ver`).  Returns `false` when the stored version no
    /// longer matches `expected_ver` — indicating a concurrent update occurred
    /// and the caller must retry.
    ///
    /// Uses `AcqRel` / `Acquire` ordering on success / failure respectively.
    pub fn try_update(&self, expected_ver: u64, new_ver: u64) -> bool {
        self.inner
            .compare_exchange(expected_ver, new_ver, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }

    /// Force-set the version without a CAS check.
    ///
    /// This is intended for administrative resets only; ordinary code should
    /// use [`try_update`](Self::try_update).
    pub fn force_set(&self, version: u64) {
        self.inner.store(version, Ordering::Release);
    }

    /// Increment the version by one unconditionally, returning the new value.
    ///
    /// Useful for internal bookkeeping where no concurrent access is expected.
    pub fn bump(&self) -> u64 {
        self.inner.fetch_add(1, Ordering::AcqRel) + 1
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_version() {
        let lock = OptimisticLock::new(42);
        assert_eq!(lock.version(), 42);
    }

    #[test]
    fn test_new_zero_version() {
        let lock = OptimisticLock::new(0);
        assert_eq!(lock.version(), 0);
    }

    #[test]
    fn test_try_update_success() {
        let lock = OptimisticLock::new(0);
        let ok = lock.try_update(0, 1);
        assert!(ok, "update with correct expected version should succeed");
        assert_eq!(lock.version(), 1);
    }

    #[test]
    fn test_try_update_failure_stale_version() {
        let lock = OptimisticLock::new(0);
        lock.try_update(0, 1); // advances to v1
        let ok = lock.try_update(0, 2); // still thinks it's v0 — stale
        assert!(!ok, "update with stale expected version should fail");
        assert_eq!(lock.version(), 1, "version must not change on failure");
    }

    #[test]
    fn test_try_update_sequential_chain() {
        let lock = OptimisticLock::new(0);
        assert!(lock.try_update(0, 1));
        assert!(lock.try_update(1, 2));
        assert!(lock.try_update(2, 3));
        assert_eq!(lock.version(), 3);
    }

    #[test]
    fn test_force_set() {
        let lock = OptimisticLock::new(5);
        lock.force_set(100);
        assert_eq!(lock.version(), 100);
    }

    #[test]
    fn test_bump_increments_version() {
        let lock = OptimisticLock::new(10);
        let v = lock.bump();
        assert_eq!(v, 11);
        assert_eq!(lock.version(), 11);
    }

    #[test]
    fn test_clone_shares_state() {
        let lock = OptimisticLock::new(0);
        let clone = lock.clone();
        lock.try_update(0, 7);
        assert_eq!(clone.version(), 7, "clone must reflect updates to the original");
    }

    #[test]
    fn test_multiple_try_update_only_one_wins() {
        // Simulate two writers both reading version 0 and trying to commit.
        let lock = OptimisticLock::new(0);
        let a = lock.try_update(0, 1);
        let b = lock.try_update(0, 2); // loses: version is now 1
        assert!(a);
        assert!(!b);
        assert_eq!(lock.version(), 1);
    }

    #[test]
    fn test_try_update_same_version_idempotent_value() {
        // Updating to the same version value is allowed as long as expected matches.
        let lock = OptimisticLock::new(5);
        assert!(lock.try_update(5, 5));
        assert_eq!(lock.version(), 5);
    }

    #[test]
    fn test_version_after_multiple_bumps() {
        let lock = OptimisticLock::new(0);
        lock.bump();
        lock.bump();
        lock.bump();
        assert_eq!(lock.version(), 3);
    }
}
