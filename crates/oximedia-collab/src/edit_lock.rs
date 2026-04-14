#![allow(dead_code)]
//! Fine-grained edit locking for collaborative editing sessions.
//!
//! Provides region-based, track-based, and hierarchical locking mechanisms
//! to prevent conflicting edits in multi-user environments, with automatic
//! expiration, lock escalation, and deadlock detection.

use std::collections::{HashMap, HashSet};
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

// ─────────────────────────────────────────────────────────────────────────────
// Waiter graph for deadlock detection on EditLockManager
// ─────────────────────────────────────────────────────────────────────────────

/// Error returned by deadlock-aware lock acquisition.
#[derive(Debug, Clone, PartialEq)]
pub struct DeadlockError {
    /// The cycle of lock holder IDs that would form a deadlock.
    pub cycle: Vec<String>,
}

impl fmt::Display for DeadlockError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Deadlock cycle detected: {}", self.cycle.join(" -> "))
    }
}

/// Tracks who is waiting for whom to release a lock.
///
/// This is a directed graph: `waits_for[A] = {B, C}` means user A is
/// blocked waiting for users B and C to release their locks.
///
/// Used by [`EditLockManager::try_acquire_with_deadlock_check`] to detect
/// cycles before granting a new lock acquisition.
#[derive(Debug, Default, Clone)]
pub struct WaiterGraph {
    /// `waits_for[holder] = set of users that holder is waiting on`.
    waits_for: HashMap<String, HashSet<String>>,
}

impl WaiterGraph {
    /// Create an empty waiter graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record that `waiter` is now waiting for `holder_of_lock` to release.
    pub fn add_wait(&mut self, waiter: &str, holder_of_lock: &str) {
        self.waits_for
            .entry(waiter.to_string())
            .or_default()
            .insert(holder_of_lock.to_string());
    }

    /// Remove all wait edges originating from `user` (called when a lock is released).
    pub fn remove_waiter(&mut self, user: &str) {
        self.waits_for.remove(user);
        // Also remove this user from all sets it appears in.
        for holders in self.waits_for.values_mut() {
            holders.remove(user);
        }
    }

    /// Detect whether adding a wait edge `waiter → blocker` would create a cycle.
    ///
    /// Returns `Some(cycle)` with the list of user IDs forming the cycle, or
    /// `None` if no cycle would be formed.
    pub fn detect_cycle_if_added(&self, waiter: &str, blocker: &str) -> Option<Vec<String>> {
        // Would blocker eventually wait on waiter (transitively)?  If so, adding
        // the edge waiter→blocker creates a cycle.
        let mut visited: HashSet<&str> = HashSet::new();
        let mut in_stack: Vec<String> = Vec::new();

        // DFS from `blocker` through the existing waits_for graph.
        self.dfs_cycle(blocker, waiter, &mut visited, &mut in_stack)
    }

    /// DFS helper: traverses the waits_for graph from `current` looking for
    /// `target`. Returns `Some(path)` if `target` is reachable (creating a cycle).
    fn dfs_cycle<'a>(
        &'a self,
        current: &'a str,
        target: &str,
        visited: &mut HashSet<&'a str>,
        in_stack: &mut Vec<String>,
    ) -> Option<Vec<String>> {
        if current == target {
            // Found the cycle — return the full cycle path.
            let mut cycle = in_stack.clone();
            cycle.push(target.to_string());
            return Some(cycle);
        }

        if visited.contains(current) {
            return None;
        }
        visited.insert(current);
        in_stack.push(current.to_string());

        if let Some(neighbors) = self.waits_for.get(current) {
            for next in neighbors {
                if let Some(cycle) = self.dfs_cycle(next.as_str(), target, visited, in_stack) {
                    return Some(cycle);
                }
            }
        }

        in_stack.pop();
        None
    }

    /// Return all users currently tracked in the graph.
    pub fn users(&self) -> Vec<&str> {
        self.waits_for.keys().map(|s| s.as_str()).collect()
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
    /// Waiter graph for deadlock detection.
    waiter_graph: WaiterGraph,
}

impl EditLockManager {
    /// Create a new lock manager with a default TTL.
    pub fn new(default_ttl_ms: u64) -> Self {
        Self {
            locks: HashMap::new(),
            next_id: 1,
            default_ttl_ms,
            waiter_graph: WaiterGraph::new(),
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
        // Clean up any wait-edges in the deadlock graph for this user.
        self.waiter_graph.remove_waiter(user_id);
        Ok(())
    }

    /// Attempt to acquire a lock with deadlock detection.
    ///
    /// Before granting the lock, the waiter graph is consulted to check whether
    /// adding a wait-edge from `user_id` to each current holder would create a
    /// cycle.  If a cycle is detected, `Err(DeadlockError { cycle })` is returned
    /// and the lock is NOT acquired.  On success the lock ID is returned.
    pub fn try_acquire_with_deadlock_check(
        &mut self,
        user_id: &str,
        target: LockTarget,
        mode: LockMode,
        now: u64,
    ) -> Result<u64, DeadlockError> {
        self.cleanup_expired(now);

        // Collect the set of users who would block this acquisition.
        let blockers: Vec<String> = self
            .locks
            .values()
            .filter(|l| {
                l.user_id != user_id
                    && targets_conflict(&l.target, &target)
                    && !(l.mode == LockMode::Shared && mode == LockMode::Shared)
            })
            .map(|l| l.user_id.clone())
            .collect();

        // For each blocker, check if adding user_id → blocker would form a cycle.
        for blocker in &blockers {
            if let Some(cycle) = self.waiter_graph.detect_cycle_if_added(user_id, blocker) {
                return Err(DeadlockError { cycle });
            }
        }

        // No deadlock: register the wait edges temporarily (removed on acquire).
        for blocker in &blockers {
            self.waiter_graph.add_wait(user_id, blocker);
        }

        // If there are blockers we cannot actually acquire yet; return a
        // NotFound-style error via LockError::Conflict without deadlock.
        // The caller should retry after the blocker releases.
        if !blockers.is_empty() {
            // Roll back the wait edges we just added (caller will retry).
            self.waiter_graph.remove_waiter(user_id);
            // Return a "soft" conflict — but not a deadlock.
            // We re-use the LockError::Conflict logic by converting it.
            let held_by = blockers[0].clone();
            // Can't return LockError here (different error type), so we
            // report an empty cycle to signal "would block, not deadlock".
            return Err(DeadlockError {
                cycle: vec![format!("blocked: {} waits for {}", user_id, held_by)],
            });
        }

        // No blockers and no deadlock: grant the lock.
        self.waiter_graph.remove_waiter(user_id);
        let lock_id = self.next_id;
        self.next_id += 1;
        let lock = EditLock::new(lock_id, user_id, target, mode, now, self.default_ttl_ms);
        self.locks.insert(lock_id, lock);
        Ok(lock_id)
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

// ─────────────────────────────────────────────────────────────────────────────
// Deadlock detection tests on EditLockManager
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod deadlock_tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use std::thread;

    #[test]
    fn test_waiter_graph_simple_2_node_cycle_detected() {
        let mut graph = WaiterGraph::new();
        // A is waiting for B.
        graph.add_wait("A", "B");
        // Now if we try to add B→A, that would form a cycle.
        let cycle = graph.detect_cycle_if_added("B", "A");
        assert!(cycle.is_some(), "should detect A→B→A cycle");
        let c = cycle.expect("cycle must exist");
        assert!(c.contains(&"A".to_string()) || c.contains(&"B".to_string()));
    }

    #[test]
    fn test_waiter_graph_3_node_cycle_detected() {
        let mut graph = WaiterGraph::new();
        // A→B, B→C; then adding C→A should detect cycle.
        graph.add_wait("A", "B");
        graph.add_wait("B", "C");
        let cycle = graph.detect_cycle_if_added("C", "A");
        assert!(cycle.is_some(), "A→B→C→A cycle must be detected");
    }

    #[test]
    fn test_waiter_graph_no_cycle() {
        let mut graph = WaiterGraph::new();
        // A→B, C→D: completely separate chains.
        graph.add_wait("A", "B");
        graph.add_wait("C", "D");
        let cycle = graph.detect_cycle_if_added("B", "C");
        // B→C forms B→C→D, not a cycle back to A or B.
        assert!(cycle.is_none(), "no cycle should be detected");
    }

    #[test]
    fn test_waiter_graph_remove_breaks_cycle_path() {
        let mut graph = WaiterGraph::new();
        graph.add_wait("A", "B");
        // Remove A's wait edges.
        graph.remove_waiter("A");
        // Now B→A does not form a cycle because A→B no longer exists.
        let cycle = graph.detect_cycle_if_added("B", "A");
        assert!(cycle.is_none());
    }

    #[test]
    fn test_deadlock_check_2_thread_scenario() {
        // Simulates: A holds clip-1, B holds clip-2.
        // A tries to acquire clip-2 → blocked (not a deadlock yet).
        // B tries to acquire clip-1 → deadlock!
        let mut mgr = EditLockManager::new(60_000);
        let clip1 = LockTarget::Clip {
            clip_id: "clip-1".to_string(),
        };
        let clip2 = LockTarget::Clip {
            clip_id: "clip-2".to_string(),
        };

        // A acquires clip-1.
        mgr.acquire("A", clip1.clone(), LockMode::Exclusive, 1000)
            .expect("A should acquire clip-1");
        // B acquires clip-2.
        mgr.acquire("B", clip2.clone(), LockMode::Exclusive, 1000)
            .expect("B should acquire clip-2");

        // Record in waiter graph that A is waiting for B (A wants clip-2 held by B).
        mgr.waiter_graph.add_wait("A", "B");

        // B now tries to acquire clip-1 (held by A): deadlock!
        let result = mgr.try_acquire_with_deadlock_check("B", clip1, LockMode::Exclusive, 1000);
        assert!(result.is_err(), "B→clip-1 should return an error");
        let err = result.expect_err("must be err");
        // The cycle should involve A and B.
        assert!(
            err.cycle.iter().any(|s| s.contains("A") || s.contains("B")),
            "cycle should mention A or B: {:?}",
            err.cycle
        );
    }

    #[test]
    fn test_no_deadlock_unrelated_locks() {
        let mut mgr = EditLockManager::new(60_000);
        // A holds clip-1, C holds clip-3 (unrelated).
        mgr.acquire(
            "A",
            LockTarget::Clip {
                clip_id: "clip-1".to_string(),
            },
            LockMode::Exclusive,
            1000,
        )
        .expect("A acquires clip-1");
        mgr.acquire(
            "C",
            LockTarget::Clip {
                clip_id: "clip-3".to_string(),
            },
            LockMode::Exclusive,
            1000,
        )
        .expect("C acquires clip-3");

        // B tries to acquire clip-2 (not held by anyone) → no conflict, no deadlock.
        let result = mgr.try_acquire_with_deadlock_check(
            "B",
            LockTarget::Clip {
                clip_id: "clip-2".to_string(),
            },
            LockMode::Exclusive,
            1000,
        );
        assert!(result.is_ok(), "B should acquire clip-2 without deadlock");
    }

    #[test]
    fn test_concurrent_stress_no_panic() {
        // 8 threads randomly acquiring/releasing locks — must not panic.
        let mgr = Arc::new(Mutex::new(EditLockManager::new(100_000)));
        let clip_ids = ["c1", "c2", "c3", "c4"];
        let handles: Vec<_> = (0..8)
            .map(|i| {
                let mgr_clone = Arc::clone(&mgr);
                let user = format!("user-{}", i);
                let clip = clip_ids[i % clip_ids.len()].to_string();
                thread::spawn(move || {
                    for t in 0u64..50 {
                        let now = t * 100;
                        let target = LockTarget::Clip {
                            clip_id: clip.clone(),
                        };
                        let mut guard = mgr_clone.lock().expect("mutex should not be poisoned");
                        // Try deadlock-aware acquire; ignore all errors.
                        let _ = guard.try_acquire_with_deadlock_check(
                            &user,
                            target,
                            LockMode::Exclusive,
                            now,
                        );
                        // Also try to release any locks we hold.
                        guard.release_all_for_user(&user);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().expect("thread should not panic");
        }
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

// ─────────────────────────────────────────────────────────────────────────────
// Per-track locking with lock escalation
// ─────────────────────────────────────────────────────────────────────────────

/// The granularity at which a per-track lock is held.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TrackLockGranularity {
    /// Locked for reading only (multiple readers allowed).
    Read,
    /// Locked for small region edits.
    Region,
    /// Locked for operations spanning the entire track.
    FullTrack,
}

impl std::fmt::Display for TrackLockGranularity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Read => write!(f, "read"),
            Self::Region => write!(f, "region"),
            Self::FullTrack => write!(f, "full_track"),
        }
    }
}

/// A per-track lock record.
#[derive(Debug, Clone)]
pub struct TrackLock {
    /// Track being locked.
    pub track_id: String,
    /// User holding this lock.
    pub user_id: String,
    /// Lock granularity.
    pub granularity: TrackLockGranularity,
    /// When this lock was acquired (epoch millis).
    pub acquired_at: u64,
    /// When this lock expires (epoch millis).
    pub expires_at: u64,
}

impl TrackLock {
    /// Check if expired.
    #[must_use]
    pub fn is_expired(&self, now: u64) -> bool {
        now >= self.expires_at
    }

    /// Remaining TTL in milliseconds (0 if expired).
    #[must_use]
    pub fn remaining_ms(&self, now: u64) -> u64 {
        self.expires_at.saturating_sub(now)
    }
}

/// Error returned by per-track lock operations.
#[derive(Debug, Clone, PartialEq)]
pub enum TrackLockError {
    /// Another user holds a conflicting lock.
    Conflict {
        track_id: String,
        held_by: String,
        held_granularity: TrackLockGranularity,
    },
    /// The requested lock does not exist.
    NotFound { track_id: String, user_id: String },
    /// The lock has already expired.
    Expired { track_id: String },
    /// Deadlock detected between two users competing on the same track.
    Deadlock { description: String },
}

impl std::fmt::Display for TrackLockError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Conflict {
                track_id,
                held_by,
                held_granularity,
            } => write!(
                f,
                "Track '{track_id}' locked ({held_granularity}) by '{held_by}'"
            ),
            Self::NotFound { track_id, user_id } => {
                write!(f, "No lock on track '{track_id}' for user '{user_id}'")
            }
            Self::Expired { track_id } => write!(f, "Lock on track '{track_id}' has expired"),
            Self::Deadlock { description } => write!(f, "Deadlock: {description}"),
        }
    }
}

/// Manager for per-track locks with escalation support.
///
/// Supports three granularity levels:
/// - `Read` — multiple readers allowed across users.
/// - `Region` — exclusive: only one user per track at a time.
/// - `FullTrack` — exclusive upgrade; downgrades require re-acquisition.
///
/// Escalation: a user holding a `Region` lock can escalate to `FullTrack`
/// if no other user holds any lock on the same track.
#[derive(Debug, Default)]
pub struct TrackLockManager {
    /// Active per-track locks.  Multiple entries for the same track are
    /// allowed only when all are `Read`.
    locks: Vec<TrackLock>,
    /// Default TTL in milliseconds.
    default_ttl_ms: u64,
}

impl TrackLockManager {
    /// Create a new manager with the given default TTL.
    #[must_use]
    pub fn new(default_ttl_ms: u64) -> Self {
        Self {
            locks: Vec::new(),
            default_ttl_ms,
        }
    }

    /// Remove expired locks and return the count removed.
    pub fn cleanup_expired(&mut self, now: u64) -> usize {
        let before = self.locks.len();
        self.locks.retain(|l| !l.is_expired(now));
        before - self.locks.len()
    }

    /// Attempt to acquire a lock on `track_id` for `user_id`.
    ///
    /// Returns `Ok(())` on success, or a `TrackLockError` if the acquisition
    /// would conflict or is illegal.
    pub fn acquire(
        &mut self,
        track_id: &str,
        user_id: &str,
        granularity: TrackLockGranularity,
        now: u64,
    ) -> Result<(), TrackLockError> {
        self.cleanup_expired(now);

        // Check for conflicts with other users' locks on the same track.
        for existing in self.locks.iter().filter(|l| l.track_id == track_id) {
            if existing.user_id == user_id {
                // Same user re-acquiring the same or lower granularity is a
                // no-op upgrade path; handled by escalate().
                continue;
            }
            // Two Read locks are always compatible.
            if existing.granularity == TrackLockGranularity::Read
                && granularity == TrackLockGranularity::Read
            {
                continue;
            }
            // Any other combination is a conflict.
            return Err(TrackLockError::Conflict {
                track_id: track_id.to_string(),
                held_by: existing.user_id.clone(),
                held_granularity: existing.granularity,
            });
        }

        // Remove any existing lock this user holds on the same track before
        // inserting the new one.
        self.locks
            .retain(|l| !(l.track_id == track_id && l.user_id == user_id));

        self.locks.push(TrackLock {
            track_id: track_id.to_string(),
            user_id: user_id.to_string(),
            granularity,
            acquired_at: now,
            expires_at: now + self.default_ttl_ms,
        });

        Ok(())
    }

    /// Release a lock held by `user_id` on `track_id`.
    pub fn release(&mut self, track_id: &str, user_id: &str) -> Result<(), TrackLockError> {
        let before = self.locks.len();
        self.locks
            .retain(|l| !(l.track_id == track_id && l.user_id == user_id));
        if self.locks.len() < before {
            Ok(())
        } else {
            Err(TrackLockError::NotFound {
                track_id: track_id.to_string(),
                user_id: user_id.to_string(),
            })
        }
    }

    /// Escalate an existing lock held by `user_id` on `track_id` to a higher
    /// granularity.
    ///
    /// Returns `Err(TrackLockError::Conflict)` if another user holds any lock
    /// on that track, or `Err(TrackLockError::NotFound)` if the caller has no
    /// current lock on the track.
    pub fn escalate(
        &mut self,
        track_id: &str,
        user_id: &str,
        new_granularity: TrackLockGranularity,
        now: u64,
    ) -> Result<(), TrackLockError> {
        self.cleanup_expired(now);

        // Check the caller has a lock to escalate.
        let current = self
            .locks
            .iter()
            .find(|l| l.track_id == track_id && l.user_id == user_id)
            .ok_or_else(|| TrackLockError::NotFound {
                track_id: track_id.to_string(),
                user_id: user_id.to_string(),
            })?;

        let current_granularity = current.granularity;

        if new_granularity <= current_granularity {
            // Already at this level or higher; treat as success.
            return Ok(());
        }

        // Ensure no other users are on this track.
        for other in self.locks.iter().filter(|l| l.track_id == track_id) {
            if other.user_id == user_id {
                continue;
            }
            return Err(TrackLockError::Conflict {
                track_id: track_id.to_string(),
                held_by: other.user_id.clone(),
                held_granularity: other.granularity,
            });
        }

        // Upgrade the lock.
        for lock in self.locks.iter_mut() {
            if lock.track_id == track_id && lock.user_id == user_id {
                lock.granularity = new_granularity;
                lock.expires_at = now + self.default_ttl_ms;
                break;
            }
        }

        Ok(())
    }

    /// Detect potential deadlocks: two or more users each waiting for a
    /// resource the other holds (simplified cycle check).
    ///
    /// In this model we simply check if user A holds a read lock while user B
    /// attempts an exclusive lock *and* user B holds a read lock on a
    /// *different* track that user A is attempting to escalate — a hallmark of
    /// the classic AB/BA deadlock pattern.
    ///
    /// Returns a list of `(user_a, user_b)` pairs that are in a deadlock
    /// situation given the provided pending escalation requests.
    ///
    /// `pending_escalations` is a slice of `(track_id, user_id)` pairs
    /// representing locks that users are about to request.
    #[must_use]
    pub fn detect_deadlocks(&self, pending_escalations: &[(&str, &str)]) -> Vec<(String, String)> {
        let mut deadlocks = Vec::new();

        // Build a map: track_id → [user_id currently holding exclusive locks]
        let held: std::collections::HashMap<&str, Vec<&str>> = {
            let mut m: std::collections::HashMap<&str, Vec<&str>> =
                std::collections::HashMap::new();
            for lock in &self.locks {
                if lock.granularity > TrackLockGranularity::Read {
                    m.entry(lock.track_id.as_str())
                        .or_default()
                        .push(&lock.user_id);
                }
            }
            m
        };

        // For each pair of pending escalations, check if they form a cycle:
        // user_a wants track T1 (held by user_b) and user_b wants track T2
        // (held by user_a).
        for i in 0..pending_escalations.len() {
            for j in (i + 1)..pending_escalations.len() {
                let (track_i, user_i) = pending_escalations[i];
                let (track_j, user_j) = pending_escalations[j];

                if user_i == user_j || track_i == track_j {
                    continue;
                }

                let i_blocked_by_j = held
                    .get(track_i)
                    .map(|holders| holders.contains(&user_j))
                    .unwrap_or(false);

                let j_blocked_by_i = held
                    .get(track_j)
                    .map(|holders| holders.contains(&user_i))
                    .unwrap_or(false);

                if i_blocked_by_j && j_blocked_by_i {
                    deadlocks.push((user_i.to_string(), user_j.to_string()));
                }
            }
        }

        deadlocks
    }

    /// Return all locks held by a specific user.
    #[must_use]
    pub fn user_locks(&self, user_id: &str) -> Vec<&TrackLock> {
        self.locks.iter().filter(|l| l.user_id == user_id).collect()
    }

    /// Return all locks on a specific track.
    #[must_use]
    pub fn track_locks(&self, track_id: &str) -> Vec<&TrackLock> {
        self.locks
            .iter()
            .filter(|l| l.track_id == track_id)
            .collect()
    }

    /// Release all locks held by a user.
    pub fn release_all_for_user(&mut self, user_id: &str) -> usize {
        let before = self.locks.len();
        self.locks.retain(|l| l.user_id != user_id);
        before - self.locks.len()
    }

    /// Total number of active locks.
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.locks.len()
    }
}

#[cfg(test)]
mod track_lock_tests {
    use super::*;

    #[test]
    fn test_acquire_read_lock() {
        let mut mgr = TrackLockManager::new(60_000);
        mgr.acquire("video-1", "user1", TrackLockGranularity::Read, 1000)
            .expect("should acquire read lock");
        assert_eq!(mgr.active_count(), 1);
    }

    #[test]
    fn test_two_read_locks_on_same_track() {
        let mut mgr = TrackLockManager::new(60_000);
        mgr.acquire("video-1", "user1", TrackLockGranularity::Read, 1000)
            .expect("user1 read lock");
        mgr.acquire("video-1", "user2", TrackLockGranularity::Read, 1000)
            .expect("user2 read lock");
        assert_eq!(mgr.active_count(), 2);
    }

    #[test]
    fn test_region_lock_conflicts_with_region() {
        let mut mgr = TrackLockManager::new(60_000);
        mgr.acquire("video-1", "user1", TrackLockGranularity::Region, 1000)
            .expect("user1 region lock");
        let err = mgr
            .acquire("video-1", "user2", TrackLockGranularity::Region, 1000)
            .expect_err("should conflict");
        assert!(matches!(err, TrackLockError::Conflict { .. }));
    }

    #[test]
    fn test_full_track_lock_conflicts_with_read() {
        let mut mgr = TrackLockManager::new(60_000);
        mgr.acquire("video-1", "user1", TrackLockGranularity::Read, 1000)
            .expect("user1 read lock");
        let err = mgr
            .acquire("video-1", "user2", TrackLockGranularity::FullTrack, 1000)
            .expect_err("should conflict");
        assert!(matches!(err, TrackLockError::Conflict { .. }));
    }

    #[test]
    fn test_release_lock() {
        let mut mgr = TrackLockManager::new(60_000);
        mgr.acquire("video-1", "user1", TrackLockGranularity::Region, 1000)
            .expect("acquire");
        mgr.release("video-1", "user1").expect("release");
        assert_eq!(mgr.active_count(), 0);
    }

    #[test]
    fn test_escalate_region_to_full_track() {
        let mut mgr = TrackLockManager::new(60_000);
        mgr.acquire("video-1", "user1", TrackLockGranularity::Region, 1000)
            .expect("acquire region");
        mgr.escalate("video-1", "user1", TrackLockGranularity::FullTrack, 2000)
            .expect("escalate to full_track");
        let locks = mgr.track_locks("video-1");
        assert_eq!(locks.len(), 1);
        assert_eq!(locks[0].granularity, TrackLockGranularity::FullTrack);
    }

    #[test]
    fn test_escalate_blocked_by_other_user() {
        let mut mgr = TrackLockManager::new(60_000);
        mgr.acquire("video-1", "user1", TrackLockGranularity::Region, 1000)
            .expect("user1 region lock");
        mgr.acquire("video-2", "user2", TrackLockGranularity::Read, 1000)
            .expect("user2 read on other track");
        // Inject a lock so user2 also holds a read on video-1
        mgr.locks.push(TrackLock {
            track_id: "video-1".to_string(),
            user_id: "user2".to_string(),
            granularity: TrackLockGranularity::Read,
            acquired_at: 1000,
            expires_at: 61_000,
        });
        let err = mgr
            .escalate("video-1", "user1", TrackLockGranularity::FullTrack, 2000)
            .expect_err("escalation should be blocked by user2");
        assert!(matches!(err, TrackLockError::Conflict { .. }));
    }

    #[test]
    fn test_escalate_no_existing_lock_fails() {
        let mut mgr = TrackLockManager::new(60_000);
        let err = mgr
            .escalate("video-1", "user1", TrackLockGranularity::FullTrack, 1000)
            .expect_err("no lock to escalate");
        assert!(matches!(err, TrackLockError::NotFound { .. }));
    }

    #[test]
    fn test_detect_deadlocks() {
        let mut mgr = TrackLockManager::new(60_000);
        // user1 holds Region on video-1; user2 holds Region on video-2
        mgr.acquire("video-1", "user1", TrackLockGranularity::Region, 1000)
            .expect("user1 locks video-1");
        mgr.acquire("video-2", "user2", TrackLockGranularity::Region, 1000)
            .expect("user2 locks video-2");

        // Pending: user1 wants video-2, user2 wants video-1 → deadlock
        let pending = vec![("video-2", "user1"), ("video-1", "user2")];
        let deadlocks = mgr.detect_deadlocks(&pending);
        assert_eq!(deadlocks.len(), 1, "should detect one deadlock pair");
    }

    #[test]
    fn test_no_deadlock_when_no_conflict() {
        let mgr = TrackLockManager::new(60_000);
        let pending = vec![("video-1", "user1"), ("video-2", "user2")];
        let deadlocks = mgr.detect_deadlocks(&pending);
        assert!(deadlocks.is_empty());
    }

    #[test]
    fn test_expiry_cleanup() {
        let mut mgr = TrackLockManager::new(100);
        mgr.acquire("video-1", "user1", TrackLockGranularity::Region, 1000)
            .expect("acquire");
        let removed = mgr.cleanup_expired(2000);
        assert_eq!(removed, 1);
        assert_eq!(mgr.active_count(), 0);
    }

    #[test]
    fn test_granularity_ordering() {
        assert!(TrackLockGranularity::Read < TrackLockGranularity::Region);
        assert!(TrackLockGranularity::Region < TrackLockGranularity::FullTrack);
    }

    #[test]
    fn test_granularity_display() {
        assert_eq!(TrackLockGranularity::Read.to_string(), "read");
        assert_eq!(TrackLockGranularity::FullTrack.to_string(), "full_track");
    }
}
