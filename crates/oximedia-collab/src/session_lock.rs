//! Session-level resource locking for collaborative editing.
//!
//! Prevents concurrent edits to the same resource (clip, track, effect, etc.)
//! by maintaining short-lived, user-scoped locks with expiry support.

#![allow(dead_code)]

use std::collections::HashMap;

/// The scope of a session lock.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LockScope {
    /// Lock on a specific clip identified by its ID.
    Clip(String),
    /// Lock on an entire track.
    Track(String),
    /// Lock on a timeline segment (track + time range label).
    Segment { track: String, label: String },
    /// Lock on an effect applied to a clip.
    Effect { clip: String, effect_name: String },
    /// Lock on the project-level settings.
    ProjectSettings,
}

impl LockScope {
    /// Returns a stable string key for use in hash maps.
    #[must_use]
    pub fn key(&self) -> String {
        match self {
            Self::Clip(id) => format!("clip:{id}"),
            Self::Track(id) => format!("track:{id}"),
            Self::Segment { track, label } => format!("seg:{track}:{label}"),
            Self::Effect { clip, effect_name } => format!("effect:{clip}:{effect_name}"),
            Self::ProjectSettings => "project_settings".to_string(),
        }
    }
}

/// An individual session lock held by a user.
#[derive(Debug, Clone)]
pub struct SessionLock {
    /// The user (or agent) holding this lock.
    pub holder_id: String,
    /// The locked resource scope.
    pub scope: LockScope,
    /// Unix timestamp (seconds) when the lock was acquired.
    pub acquired_at: u64,
    /// Lifetime of the lock in seconds (0 = never expires).
    pub ttl_secs: u64,
}

impl SessionLock {
    /// Creates a new lock record.
    #[must_use]
    pub fn new(holder_id: String, scope: LockScope, acquired_at: u64, ttl_secs: u64) -> Self {
        Self {
            holder_id,
            scope,
            acquired_at,
            ttl_secs,
        }
    }

    /// Returns `true` if this lock has expired at `now_secs`.
    ///
    /// A lock with `ttl_secs == 0` never expires.
    #[must_use]
    pub fn is_expired_at(&self, now_secs: u64) -> bool {
        if self.ttl_secs == 0 {
            return false;
        }
        now_secs >= self.acquired_at.saturating_add(self.ttl_secs)
    }

    /// Returns the Unix timestamp when this lock expires, or `None` if it never
    /// expires.
    #[must_use]
    pub fn expires_at(&self) -> Option<u64> {
        if self.ttl_secs == 0 {
            None
        } else {
            Some(self.acquired_at.saturating_add(self.ttl_secs))
        }
    }

    /// Returns how many seconds remain until expiry given `now_secs`, or `None`
    /// if the lock never expires or is already expired.
    #[must_use]
    pub fn remaining_secs(&self, now_secs: u64) -> Option<u64> {
        let exp = self.expires_at()?;
        if now_secs >= exp {
            None
        } else {
            Some(exp - now_secs)
        }
    }
}

/// Error type for lock operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LockError {
    /// The resource is already locked by another holder.
    AlreadyLocked { holder_id: String },
    /// No lock was found for the given scope.
    NotFound,
    /// The caller does not hold the lock.
    NotHolder,
}

impl std::fmt::Display for LockError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AlreadyLocked { holder_id } => write!(f, "resource locked by {holder_id}"),
            Self::NotFound => write!(f, "lock not found"),
            Self::NotHolder => write!(f, "caller does not hold the lock"),
        }
    }
}

/// Manages all session locks.
#[derive(Debug, Default)]
pub struct LockManager {
    /// Mapping from scope key to active lock.
    locks: HashMap<String, SessionLock>,
}

impl LockManager {
    /// Creates an empty lock manager.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Attempts to acquire a lock on `scope` for `holder_id` at `now_secs`.
    ///
    /// Returns `Ok(())` on success.  If the scope is already locked (and the
    /// existing lock has not expired), returns `Err(LockError::AlreadyLocked)`.
    pub fn acquire(
        &mut self,
        holder_id: String,
        scope: LockScope,
        now_secs: u64,
        ttl_secs: u64,
    ) -> Result<(), LockError> {
        let key = scope.key();
        if let Some(existing) = self.locks.get(&key) {
            if !existing.is_expired_at(now_secs) {
                return Err(LockError::AlreadyLocked {
                    holder_id: existing.holder_id.clone(),
                });
            }
        }
        self.locks
            .insert(key, SessionLock::new(holder_id, scope, now_secs, ttl_secs));
        Ok(())
    }

    /// Releases a lock held by `holder_id` on `scope`.
    ///
    /// Returns `Err(LockError::NotFound)` if no lock exists for the scope, or
    /// `Err(LockError::NotHolder)` if another user holds it.
    pub fn release(&mut self, holder_id: &str, scope: &LockScope) -> Result<(), LockError> {
        let key = scope.key();
        match self.locks.get(&key) {
            None => Err(LockError::NotFound),
            Some(lock) if lock.holder_id != holder_id => Err(LockError::NotHolder),
            Some(_) => {
                self.locks.remove(&key);
                Ok(())
            }
        }
    }

    /// Returns `true` if `scope` is currently locked (and not expired) at `now_secs`.
    #[must_use]
    pub fn is_locked(&self, scope: &LockScope, now_secs: u64) -> bool {
        self.locks
            .get(&scope.key())
            .map(|l| !l.is_expired_at(now_secs))
            .unwrap_or(false)
    }

    /// Returns the lock for `scope` if it exists and has not expired.
    #[must_use]
    pub fn get_lock(&self, scope: &LockScope, now_secs: u64) -> Option<&SessionLock> {
        self.locks
            .get(&scope.key())
            .filter(|l| !l.is_expired_at(now_secs))
    }

    /// Evicts all expired locks (at `now_secs`).  Returns the number evicted.
    pub fn evict_expired(&mut self, now_secs: u64) -> usize {
        let before = self.locks.len();
        self.locks.retain(|_, l| !l.is_expired_at(now_secs));
        before - self.locks.len()
    }

    /// Returns the number of currently held locks (including possibly expired ones).
    #[must_use]
    pub fn lock_count(&self) -> usize {
        self.locks.len()
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn clip_scope(id: &str) -> LockScope {
        LockScope::Clip(id.to_string())
    }

    #[test]
    fn lock_scope_key_clip() {
        assert_eq!(clip_scope("c1").key(), "clip:c1");
    }

    #[test]
    fn lock_scope_key_track() {
        assert_eq!(LockScope::Track("t1".into()).key(), "track:t1");
    }

    #[test]
    fn lock_scope_key_project_settings() {
        assert_eq!(LockScope::ProjectSettings.key(), "project_settings");
    }

    #[test]
    fn session_lock_not_expired_permanent() {
        let lock = SessionLock::new("u1".into(), clip_scope("c1"), 100, 0);
        assert!(!lock.is_expired_at(u64::MAX));
    }

    #[test]
    fn session_lock_expired() {
        let lock = SessionLock::new("u1".into(), clip_scope("c1"), 100, 50);
        assert!(!lock.is_expired_at(149));
        assert!(lock.is_expired_at(150));
    }

    #[test]
    fn session_lock_remaining_secs() {
        let lock = SessionLock::new("u1".into(), clip_scope("c1"), 100, 60);
        assert_eq!(lock.remaining_secs(110), Some(50));
        assert_eq!(lock.remaining_secs(160), None); // expired
    }

    #[test]
    fn session_lock_expires_at() {
        let lock = SessionLock::new("u1".into(), clip_scope("c1"), 100, 30);
        assert_eq!(lock.expires_at(), Some(130));
    }

    #[test]
    fn manager_acquire_success() {
        let mut mgr = LockManager::new();
        mgr.acquire("u1".into(), clip_scope("c1"), 0, 60)
            .expect("collab test operation should succeed");
        assert!(mgr.is_locked(&clip_scope("c1"), 0));
    }

    #[test]
    fn manager_acquire_conflict() {
        let mut mgr = LockManager::new();
        mgr.acquire("u1".into(), clip_scope("c1"), 0, 60)
            .expect("collab test operation should succeed");
        let err = mgr
            .acquire("u2".into(), clip_scope("c1"), 0, 60)
            .unwrap_err();
        assert_eq!(
            err,
            LockError::AlreadyLocked {
                holder_id: "u1".to_string()
            }
        );
    }

    #[test]
    fn manager_acquire_after_expiry() {
        let mut mgr = LockManager::new();
        mgr.acquire("u1".into(), clip_scope("c1"), 0, 10)
            .expect("collab test operation should succeed");
        // Lock expired at t=10; u2 can now acquire at t=20
        mgr.acquire("u2".into(), clip_scope("c1"), 20, 60)
            .expect("collab test operation should succeed");
        let lock = mgr
            .get_lock(&clip_scope("c1"), 20)
            .expect("collab test operation should succeed");
        assert_eq!(lock.holder_id, "u2");
    }

    #[test]
    fn manager_release_success() {
        let mut mgr = LockManager::new();
        mgr.acquire("u1".into(), clip_scope("c1"), 0, 60)
            .expect("collab test operation should succeed");
        mgr.release("u1", &clip_scope("c1"))
            .expect("collab test operation should succeed");
        assert!(!mgr.is_locked(&clip_scope("c1"), 0));
    }

    #[test]
    fn manager_release_not_holder() {
        let mut mgr = LockManager::new();
        mgr.acquire("u1".into(), clip_scope("c1"), 0, 60)
            .expect("collab test operation should succeed");
        let err = mgr.release("u2", &clip_scope("c1")).unwrap_err();
        assert_eq!(err, LockError::NotHolder);
    }

    #[test]
    fn manager_evict_expired() {
        let mut mgr = LockManager::new();
        mgr.acquire("u1".into(), clip_scope("c1"), 0, 10)
            .expect("collab test operation should succeed");
        mgr.acquire("u2".into(), clip_scope("c2"), 0, 0)
            .expect("collab test operation should succeed"); // permanent
        let evicted = mgr.evict_expired(50);
        assert_eq!(evicted, 1);
        assert_eq!(mgr.lock_count(), 1);
    }

    #[test]
    fn lock_error_display() {
        let e = LockError::AlreadyLocked {
            holder_id: "u1".into(),
        };
        assert!(e.to_string().contains("u1"));
    }
}
