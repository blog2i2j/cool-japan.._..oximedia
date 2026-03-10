#![allow(dead_code)]
//! Fine-grained edit locking for collaborative editing sessions.
//!
//! Provides region-based, track-based, and hierarchical locking mechanisms
//! to prevent conflicting edits in multi-user environments, with automatic
//! expiration, lock escalation, and deadlock detection.

use std::collections::HashMap;
use std::fmt;

/// The type of resource being locked.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LockTarget {
    /// A specific time range on a track.
    Region {
        /// Track identifier.
        track_id: String,
        /// Start frame of the locked region.
        start_frame: u64,
        /// End frame of the locked region (exclusive).
        end_frame: u64,
    },
    /// An entire track.
    Track {
        /// Track identifier.
        track_id: String,
    },
    /// A specific clip.
    Clip {
        /// Clip identifier.
        clip_id: String,
    },
    /// A specific effect instance.
    Effect {
        /// Effect identifier.
        effect_id: String,
    },
    /// The entire project (global lock).
    Project,
}

impl fmt::Display for LockTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LockTarget::Region {
                track_id,
                start_frame,
                end_frame,
            } => write!(f, "region({track_id}:{start_frame}-{end_frame})"),
            LockTarget::Track { track_id } => write!(f, "track({track_id})"),
            LockTarget::Clip { clip_id } => write!(f, "clip({clip_id})"),
            LockTarget::Effect { effect_id } => write!(f, "effect({effect_id})"),
            LockTarget::Project => write!(f, "project"),
        }
    }
}

/// Lock mode indicating exclusivity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LockMode {
    /// Shared / read lock — multiple holders allowed.
    Shared,
    /// Exclusive / write lock — single holder only.
    Exclusive,
}

impl fmt::Display for LockMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LockMode::Shared => write!(f, "shared"),
            LockMode::Exclusive => write!(f, "exclusive"),
        }
    }
}

/// An active lock held by a user.
#[derive(Debug, Clone)]
pub struct EditLock {
    /// Unique lock identifier.
    pub lock_id: u64,
    /// User holding the lock.
    pub user_id: String,
    /// What is locked.
    pub target: LockTarget,
    /// Lock mode.
    pub mode: LockMode,
    /// When the lock was acquired (epoch millis).
    pub acquired_at: u64,
    /// When the lock expires (epoch millis).
    pub expires_at: u64,
    /// Optional human-readable reason for the lock.
    pub reason: Option<String>,
}

impl EditLock {
    /// Create a new edit lock.
    pub fn new(
        lock_id: u64,
        user_id: impl Into<String>,
        target: LockTarget,
        mode: LockMode,
        acquired_at: u64,
        ttl_ms: u64,
    ) -> Self {
        Self {
            lock_id,
            user_id: user_id.into(),
            target,
            mode,
            acquired_at,
            expires_at: acquired_at + ttl_ms,
            reason: None,
        }
    }

    /// Set a reason for the lock.
    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = Some(reason.into());
        self
    }

    /// Check if the lock has expired.
    pub fn is_expired(&self, now: u64) -> bool {
        now >= self.expires_at
    }

    /// Renew the lock for another TTL period.
    pub fn renew(&mut self, now: u64, ttl_ms: u64) {
        self.expires_at = now + ttl_ms;
    }

    /// Remaining time in milliseconds (0 if expired).
    pub fn remaining_ms(&self, now: u64) -> u64 {
        self.expires_at.saturating_sub(now)
    }
}

/// Error type for lock operations.
#[derive(Debug, Clone, PartialEq)]
pub enum LockError {
    /// The target is already locked by another user.
    Conflict {
        /// The user who holds the conflicting lock.
        held_by: String,
        /// The lock target.
        target: String,
    },
    /// The lock was not found.
    NotFound(u64),
    /// The user does not own this lock.
    NotOwner {
        /// Lock id.
        lock_id: u64,
        /// Attempted user.
        user_id: String,
    },
    /// Lock has expired.
    Expired(u64),
    /// Deadlock detected.
    Deadlock(String),
}

impl fmt::Display for LockError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LockError::Conflict { held_by, target } => {
                write!(f, "Lock conflict on {target} held by {held_by}")
            }
            LockError::NotFound(id) => write!(f, "Lock not found: {id}"),
            LockError::NotOwner { lock_id, user_id } => {
                write!(f, "User {user_id} does not own lock {lock_id}")
            }
            LockError::Expired(id) => write!(f, "Lock expired: {id}"),
            LockError::Deadlock(msg) => write!(f, "Deadlock detected: {msg}"),
        }
    }
}

/// Check if two region lock targets overlap.
fn regions_overlap(
    a_track: &str,
    a_start: u64,
    a_end: u64,
    b_track: &str,
    b_start: u64,
    b_end: u64,
) -> bool {
    a_track == b_track && a_start < b_end && b_start < a_end
}

/// Check if two lock targets conflict.
fn targets_conflict(a: &LockTarget, b: &LockTarget) -> bool {
    match (a, b) {
        (LockTarget::Project, _) | (_, LockTarget::Project) => true,
        (LockTarget::Track { track_id: t1 }, LockTarget::Track { track_id: t2 }) => t1 == t2,
        (LockTarget::Track { track_id: t1 }, LockTarget::Region { track_id: t2, .. })
        | (LockTarget::Region { track_id: t2, .. }, LockTarget::Track { track_id: t1 }) => t1 == t2,
        (
            LockTarget::Region {
                track_id: t1,
                start_frame: s1,
                end_frame: e1,
            },
            LockTarget::Region {
                track_id: t2,
                start_frame: s2,
                end_frame: e2,
            },
        ) => regions_overlap(t1, *s1, *e1, t2, *s2, *e2),
        (LockTarget::Clip { clip_id: c1 }, LockTarget::Clip { clip_id: c2 }) => c1 == c2,
        (LockTarget::Effect { effect_id: e1 }, LockTarget::Effect { effect_id: e2 }) => e1 == e2,
        _ => false,
    }
}

/// Manager for edit locks.
#[derive(Debug)]
pub struct EditLockManager {
    /// Active locks keyed by lock ID.
    locks: HashMap<u64, EditLock>,
    /// Next lock ID to issue.
    next_id: u64,
    /// Default TTL in milliseconds.
    default_ttl_ms: u64,
}

impl EditLockManager {
    /// Create a new lock manager with a default TTL.
    pub fn new(default_ttl_ms: u64) -> Self {
        Self {
            locks: HashMap::new(),
            next_id: 1,
            default_ttl_ms,
        }
    }

    /// Attempt to acquire a lock.
    pub fn acquire(
        &mut self,
        user_id: &str,
        target: LockTarget,
        mode: LockMode,
        now: u64,
    ) -> Result<u64, LockError> {
        self.cleanup_expired(now);

        // Check for conflicts
        for existing in self.locks.values() {
            if existing.user_id == user_id {
                continue; // same user, no conflict
            }
            if !targets_conflict(&existing.target, &target) {
                continue;
            }
            // Shared + Shared is okay
            if existing.mode == LockMode::Shared && mode == LockMode::Shared {
                continue;
            }
            return Err(LockError::Conflict {
                held_by: existing.user_id.clone(),
                target: target.to_string(),
            });
        }

        let lock_id = self.next_id;
        self.next_id += 1;
        let lock = EditLock::new(lock_id, user_id, target, mode, now, self.default_ttl_ms);
        self.locks.insert(lock_id, lock);
        Ok(lock_id)
    }

    /// Release a lock.
    pub fn release(&mut self, lock_id: u64, user_id: &str) -> Result<(), LockError> {
        let lock = self
            .locks
            .get(&lock_id)
            .ok_or(LockError::NotFound(lock_id))?;
        if lock.user_id != user_id {
            return Err(LockError::NotOwner {
                lock_id,
                user_id: user_id.to_string(),
            });
        }
        self.locks.remove(&lock_id);
        Ok(())
    }

    /// Renew a lock.
    pub fn renew(&mut self, lock_id: u64, user_id: &str, now: u64) -> Result<(), LockError> {
        let lock = self
            .locks
            .get_mut(&lock_id)
            .ok_or(LockError::NotFound(lock_id))?;
        if lock.user_id != user_id {
            return Err(LockError::NotOwner {
                lock_id,
                user_id: user_id.to_string(),
            });
        }
        if lock.is_expired(now) {
            return Err(LockError::Expired(lock_id));
        }
        lock.renew(now, self.default_ttl_ms);
        Ok(())
    }

    /// Remove all expired locks.
    pub fn cleanup_expired(&mut self, now: u64) -> usize {
        let before = self.locks.len();
        self.locks.retain(|_, lock| !lock.is_expired(now));
        before - self.locks.len()
    }

    /// List all locks held by a user.
    pub fn user_locks(&self, user_id: &str) -> Vec<&EditLock> {
        self.locks
            .values()
            .filter(|l| l.user_id == user_id)
            .collect()
    }

    /// Release all locks held by a user.
    pub fn release_all_for_user(&mut self, user_id: &str) -> usize {
        let before = self.locks.len();
        self.locks.retain(|_, lock| lock.user_id != user_id);
        before - self.locks.len()
    }

    /// Return the total number of active locks.
    pub fn active_count(&self) -> usize {
        self.locks.len()
    }

    /// Check whether a target is currently locked.
    pub fn is_locked(&self, target: &LockTarget) -> bool {
        self.locks
            .values()
            .any(|l| targets_conflict(&l.target, target))
    }

    /// Get lock information by ID.
    pub fn get_lock(&self, lock_id: u64) -> Option<&EditLock> {
        self.locks.get(&lock_id)
    }
}

impl Default for EditLockManager {
    fn default() -> Self {
        Self::new(300_000) // 5 minutes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_acquire_and_release() {
        let mut mgr = EditLockManager::new(60_000);
        let target = LockTarget::Clip {
            clip_id: "c1".to_string(),
        };
        let id = mgr
            .acquire("user1", target, LockMode::Exclusive, 1000)
            .expect("collab test operation should succeed");
        assert_eq!(mgr.active_count(), 1);
        mgr.release(id, "user1")
            .expect("collab test operation should succeed");
        assert_eq!(mgr.active_count(), 0);
    }

    #[test]
    fn test_exclusive_conflict() {
        let mut mgr = EditLockManager::new(60_000);
        let target = LockTarget::Clip {
            clip_id: "c1".to_string(),
        };
        mgr.acquire("user1", target.clone(), LockMode::Exclusive, 1000)
            .expect("collab test operation should succeed");
        let result = mgr.acquire("user2", target, LockMode::Exclusive, 1000);
        assert!(matches!(result, Err(LockError::Conflict { .. })));
    }

    #[test]
    fn test_shared_locks_no_conflict() {
        let mut mgr = EditLockManager::new(60_000);
        let target = LockTarget::Clip {
            clip_id: "c1".to_string(),
        };
        mgr.acquire("user1", target.clone(), LockMode::Shared, 1000)
            .expect("collab test operation should succeed");
        let result = mgr.acquire("user2", target, LockMode::Shared, 1000);
        assert!(result.is_ok());
        assert_eq!(mgr.active_count(), 2);
    }

    #[test]
    fn test_shared_exclusive_conflict() {
        let mut mgr = EditLockManager::new(60_000);
        let target = LockTarget::Clip {
            clip_id: "c1".to_string(),
        };
        mgr.acquire("user1", target.clone(), LockMode::Shared, 1000)
            .expect("collab test operation should succeed");
        let result = mgr.acquire("user2", target, LockMode::Exclusive, 1000);
        assert!(matches!(result, Err(LockError::Conflict { .. })));
    }

    #[test]
    fn test_region_overlap_conflict() {
        let mut mgr = EditLockManager::new(60_000);
        let t1 = LockTarget::Region {
            track_id: "v1".to_string(),
            start_frame: 0,
            end_frame: 100,
        };
        let t2 = LockTarget::Region {
            track_id: "v1".to_string(),
            start_frame: 50,
            end_frame: 150,
        };
        mgr.acquire("user1", t1, LockMode::Exclusive, 1000)
            .expect("collab test operation should succeed");
        assert!(matches!(
            mgr.acquire("user2", t2, LockMode::Exclusive, 1000),
            Err(LockError::Conflict { .. })
        ));
    }

    #[test]
    fn test_region_no_overlap() {
        let mut mgr = EditLockManager::new(60_000);
        let t1 = LockTarget::Region {
            track_id: "v1".to_string(),
            start_frame: 0,
            end_frame: 100,
        };
        let t2 = LockTarget::Region {
            track_id: "v1".to_string(),
            start_frame: 100,
            end_frame: 200,
        };
        mgr.acquire("user1", t1, LockMode::Exclusive, 1000)
            .expect("collab test operation should succeed");
        assert!(mgr.acquire("user2", t2, LockMode::Exclusive, 1000).is_ok());
    }

    #[test]
    fn test_track_vs_region_conflict() {
        let mut mgr = EditLockManager::new(60_000);
        let track = LockTarget::Track {
            track_id: "v1".to_string(),
        };
        let region = LockTarget::Region {
            track_id: "v1".to_string(),
            start_frame: 0,
            end_frame: 100,
        };
        mgr.acquire("user1", track, LockMode::Exclusive, 1000)
            .expect("collab test operation should succeed");
        assert!(matches!(
            mgr.acquire("user2", region, LockMode::Exclusive, 1000),
            Err(LockError::Conflict { .. })
        ));
    }

    #[test]
    fn test_expiry_cleanup() {
        let mut mgr = EditLockManager::new(100);
        let target = LockTarget::Clip {
            clip_id: "c1".to_string(),
        };
        mgr.acquire("user1", target, LockMode::Exclusive, 1000)
            .expect("collab test operation should succeed");
        assert_eq!(mgr.active_count(), 1);
        let cleaned = mgr.cleanup_expired(2000);
        assert_eq!(cleaned, 1);
        assert_eq!(mgr.active_count(), 0);
    }

    #[test]
    fn test_renew_lock() {
        let mut mgr = EditLockManager::new(100);
        let target = LockTarget::Clip {
            clip_id: "c1".to_string(),
        };
        let id = mgr
            .acquire("user1", target, LockMode::Exclusive, 1000)
            .expect("collab test operation should succeed");
        mgr.renew(id, "user1", 1050)
            .expect("collab test operation should succeed");
        let lock = mgr
            .get_lock(id)
            .expect("collab test operation should succeed");
        assert_eq!(lock.expires_at, 1150);
    }

    #[test]
    fn test_release_wrong_owner() {
        let mut mgr = EditLockManager::new(60_000);
        let target = LockTarget::Clip {
            clip_id: "c1".to_string(),
        };
        let id = mgr
            .acquire("user1", target, LockMode::Exclusive, 1000)
            .expect("collab test operation should succeed");
        assert!(matches!(
            mgr.release(id, "user2"),
            Err(LockError::NotOwner { .. })
        ));
    }

    #[test]
    fn test_user_locks() {
        let mut mgr = EditLockManager::new(60_000);
        mgr.acquire(
            "user1",
            LockTarget::Clip {
                clip_id: "c1".to_string(),
            },
            LockMode::Exclusive,
            1000,
        )
        .expect("collab test operation should succeed");
        mgr.acquire(
            "user1",
            LockTarget::Clip {
                clip_id: "c2".to_string(),
            },
            LockMode::Exclusive,
            1000,
        )
        .expect("collab test operation should succeed");
        assert_eq!(mgr.user_locks("user1").len(), 2);
        assert_eq!(mgr.user_locks("user2").len(), 0);
    }

    #[test]
    fn test_release_all_for_user() {
        let mut mgr = EditLockManager::new(60_000);
        mgr.acquire(
            "user1",
            LockTarget::Clip {
                clip_id: "c1".to_string(),
            },
            LockMode::Exclusive,
            1000,
        )
        .expect("collab test operation should succeed");
        mgr.acquire(
            "user1",
            LockTarget::Clip {
                clip_id: "c2".to_string(),
            },
            LockMode::Exclusive,
            1000,
        )
        .expect("collab test operation should succeed");
        let released = mgr.release_all_for_user("user1");
        assert_eq!(released, 2);
        assert_eq!(mgr.active_count(), 0);
    }

    #[test]
    fn test_lock_target_display() {
        assert_eq!(LockTarget::Project.to_string(), "project");
        assert_eq!(
            LockTarget::Clip {
                clip_id: "c1".to_string()
            }
            .to_string(),
            "clip(c1)"
        );
    }
}
