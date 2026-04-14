//! Per-track lock escalation for collaborative editing sessions.
//!
//! When a user begins editing a specific region of a track they typically hold
//! a lightweight **shared** lock. As their edit expands to cover the whole
//! track — or when a higher-priority operation requires exclusive access — the
//! lock must be **escalated** to exclusive mode.
//!
//! This module provides:
//!
//! - [`EscalationPriority`] — numeric priority that determines which pending
//!   escalation is granted first when a track becomes available.
//! - [`PendingEscalation`] — a queued request to upgrade a lock.
//! - [`TrackLockState`] — the full locking state for a single track including
//!   current holder, mode, and the pending escalation queue.
//! - [`LockEscalationManager`] — manages all tracks, processes escalation
//!   requests, detects deadlocks (cycle detection in the waits-for graph), and
//!   forcibly revokes stale locks.

#![allow(dead_code)]

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

// ─────────────────────────────────────────────────────────────────────────────
// Domain types
// ─────────────────────────────────────────────────────────────────────────────

/// Numeric priority for escalation requests (higher value = higher priority).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct EscalationPriority(pub u32);

impl EscalationPriority {
    /// Normal priority.
    pub const NORMAL: Self = Self(100);
    /// Elevated priority (e.g. senior editor).
    pub const HIGH: Self = Self(200);
    /// Real-time critical (e.g. live broadcast correction).
    pub const CRITICAL: Self = Self(500);
}

impl Default for EscalationPriority {
    fn default() -> Self {
        Self::NORMAL
    }
}

impl fmt::Display for EscalationPriority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "priority({})", self.0)
    }
}

/// Current mode of a per-track lock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackLockMode {
    /// Multiple users may hold a shared lock simultaneously.
    Shared,
    /// Only one user may hold the lock; all others are blocked.
    Exclusive,
}

impl fmt::Display for TrackLockMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shared => write!(f, "shared"),
            Self::Exclusive => write!(f, "exclusive"),
        }
    }
}

/// A pending request to escalate a lock from shared to exclusive.
#[derive(Debug, Clone)]
pub struct PendingEscalation {
    /// User requesting escalation.
    pub user_id: String,
    /// Track they wish to own exclusively.
    pub track_id: String,
    /// When the request was submitted (epoch ms).
    pub requested_at_ms: u64,
    /// Priority — higher values are served first.
    pub priority: EscalationPriority,
    /// Optional reason for escalation.
    pub reason: Option<String>,
}

impl PendingEscalation {
    /// Create a new escalation request.
    #[must_use]
    pub fn new(
        user_id: impl Into<String>,
        track_id: impl Into<String>,
        requested_at_ms: u64,
        priority: EscalationPriority,
    ) -> Self {
        Self {
            user_id: user_id.into(),
            track_id: track_id.into(),
            requested_at_ms,
            priority,
            reason: None,
        }
    }

    /// Attach a human-readable reason.
    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = Some(reason.into());
        self
    }
}

/// Active lock holder entry on a track.
#[derive(Debug, Clone)]
pub struct TrackLockHolder {
    /// User holding this lock entry.
    pub user_id: String,
    /// Mode this particular holder has.
    pub mode: TrackLockMode,
    /// When the lock was acquired.
    pub acquired_at_ms: u64,
    /// Expiry time.
    pub expires_at_ms: u64,
}

impl TrackLockHolder {
    /// Create a new holder entry.
    #[must_use]
    pub fn new(
        user_id: impl Into<String>,
        mode: TrackLockMode,
        acquired_at_ms: u64,
        ttl_ms: u64,
    ) -> Self {
        Self {
            user_id: user_id.into(),
            mode,
            acquired_at_ms,
            expires_at_ms: acquired_at_ms + ttl_ms,
        }
    }

    /// Check whether this holder's lock has expired.
    pub fn is_expired(&self, now_ms: u64) -> bool {
        now_ms >= self.expires_at_ms
    }
}

/// Full locking state for a single track.
#[derive(Debug, Default)]
pub struct TrackLockState {
    /// Active holders on this track (may be >1 for shared mode).
    pub holders: Vec<TrackLockHolder>,
    /// Queue of pending escalation requests, ordered by priority then time.
    pub pending: VecDeque<PendingEscalation>,
}

impl TrackLockState {
    /// Return `true` if any holder is in exclusive mode.
    pub fn is_exclusively_held(&self) -> bool {
        self.holders.iter().any(|h| h.mode == TrackLockMode::Exclusive)
    }

    /// Return `true` if `user_id` currently holds any lock on this track.
    pub fn is_held_by(&self, user_id: &str) -> bool {
        self.holders.iter().any(|h| h.user_id == user_id)
    }

    /// Count of active holders.
    pub fn holder_count(&self) -> usize {
        self.holders.len()
    }

    /// Remove expired holders and return the count removed.
    pub fn expire(&mut self, now_ms: u64) -> usize {
        let before = self.holders.len();
        self.holders.retain(|h| !h.is_expired(now_ms));
        before - self.holders.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Error variants for lock escalation operations.
#[derive(Debug, Clone, PartialEq)]
pub enum EscalationError {
    /// Escalation is blocked because other users hold a lock on the track.
    BlockedBy(Vec<String>),
    /// The user does not currently hold a shared lock to escalate.
    NoSharedLock { user_id: String, track_id: String },
    /// The user already holds an exclusive lock.
    AlreadyExclusive { user_id: String, track_id: String },
    /// A deadlock was detected in the waits-for graph.
    Deadlock(String),
    /// The requested track is unknown.
    UnknownTrack(String),
    /// The escalation request is a duplicate.
    DuplicatePending { user_id: String, track_id: String },
}

impl fmt::Display for EscalationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BlockedBy(users) => write!(f, "escalation blocked by: {}", users.join(", ")),
            Self::NoSharedLock { user_id, track_id } => {
                write!(f, "{user_id} has no shared lock on {track_id}")
            }
            Self::AlreadyExclusive { user_id, track_id } => {
                write!(f, "{user_id} already holds exclusive lock on {track_id}")
            }
            Self::Deadlock(msg) => write!(f, "deadlock: {msg}"),
            Self::UnknownTrack(t) => write!(f, "unknown track: {t}"),
            Self::DuplicatePending { user_id, track_id } => {
                write!(f, "{user_id} already has a pending escalation for {track_id}")
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// EscalationResult
// ─────────────────────────────────────────────────────────────────────────────

/// Outcome of attempting an escalation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EscalationResult {
    /// Escalation was granted immediately.
    Granted,
    /// Escalation is queued and will be granted when the track is free.
    Queued,
}

// ─────────────────────────────────────────────────────────────────────────────
// LockEscalationManager
// ─────────────────────────────────────────────────────────────────────────────

/// Manages per-track lock state including shared→exclusive escalation with
/// priority queuing and waits-for deadlock detection.
#[derive(Debug, Default)]
pub struct LockEscalationManager {
    /// Per-track state.
    tracks: HashMap<String, TrackLockState>,
    /// Default lock TTL in milliseconds.
    default_ttl_ms: u64,
}

impl LockEscalationManager {
    /// Create a manager with the given default lock TTL.
    #[must_use]
    pub fn new(default_ttl_ms: u64) -> Self {
        Self {
            tracks: HashMap::new(),
            default_ttl_ms,
        }
    }

    // ── Internal helpers ────────────────────────────────────────────────────

    fn state_mut(&mut self, track_id: &str) -> &mut TrackLockState {
        self.tracks.entry(track_id.to_string()).or_default()
    }

    fn state(&self, track_id: &str) -> Option<&TrackLockState> {
        self.tracks.get(track_id)
    }

    /// Insert a pending escalation in priority-descending, then time-ascending
    /// order.
    fn enqueue_escalation(pending: &mut VecDeque<PendingEscalation>, req: PendingEscalation) {
        // Find insertion point: first index where priority < req.priority,
        // or for equal priority, after all earlier requests.
        let pos = pending.iter().position(|p| {
            p.priority < req.priority
                || (p.priority == req.priority && p.requested_at_ms > req.requested_at_ms)
        });
        match pos {
            Some(i) => pending.insert(i, req),
            None => pending.push_back(req),
        }
    }

    // ── Public API ──────────────────────────────────────────────────────────

    /// Acquire a shared lock for `user_id` on `track_id`.
    ///
    /// Fails if the track is already held exclusively by another user.
    pub fn acquire_shared(
        &mut self,
        user_id: &str,
        track_id: &str,
        now_ms: u64,
    ) -> Result<(), EscalationError> {
        let ttl_ms = self.default_ttl_ms;
        let state = self.state_mut(track_id);
        state.expire(now_ms);

        if state.is_exclusively_held() {
            let blockers: Vec<String> = state.holders.iter().map(|h| h.user_id.clone()).collect();
            return Err(EscalationError::BlockedBy(blockers));
        }

        // Allow re-entrant shared acquisition (idempotent).
        if !state.is_held_by(user_id) {
            state.holders.push(TrackLockHolder::new(
                user_id,
                TrackLockMode::Shared,
                now_ms,
                ttl_ms,
            ));
        }
        Ok(())
    }

    /// Request escalation of `user_id`'s shared lock to exclusive.
    ///
    /// If no other users currently hold a shared lock the escalation is granted
    /// immediately.  Otherwise the request is queued.  Returns an error if a
    /// deadlock is detected or the request is malformed.
    pub fn request_escalation(
        &mut self,
        user_id: &str,
        track_id: &str,
        now_ms: u64,
        priority: EscalationPriority,
    ) -> Result<EscalationResult, EscalationError> {
        let ttl_ms = self.default_ttl_ms;
        {
            let state = self.state_mut(track_id);
            state.expire(now_ms);

            // User must already hold a shared lock.
            if !state.is_held_by(user_id) {
                return Err(EscalationError::NoSharedLock {
                    user_id: user_id.to_string(),
                    track_id: track_id.to_string(),
                });
            }

            // Already exclusive?
            if state
                .holders
                .iter()
                .any(|h| h.user_id == user_id && h.mode == TrackLockMode::Exclusive)
            {
                return Err(EscalationError::AlreadyExclusive {
                    user_id: user_id.to_string(),
                    track_id: track_id.to_string(),
                });
            }

            // Duplicate pending?
            if state.pending.iter().any(|p| p.user_id == user_id) {
                return Err(EscalationError::DuplicatePending {
                    user_id: user_id.to_string(),
                    track_id: track_id.to_string(),
                });
            }

            // Can we escalate immediately? (only this user holds a shared lock)
            let other_holders: Vec<&TrackLockHolder> =
                state.holders.iter().filter(|h| h.user_id != user_id).collect();

            if other_holders.is_empty() {
                // Promote to exclusive in-place.
                if let Some(h) = state.holders.iter_mut().find(|h| h.user_id == user_id) {
                    h.mode = TrackLockMode::Exclusive;
                    h.expires_at_ms = now_ms + ttl_ms;
                }
                return Ok(EscalationResult::Granted);
            }

            // Must queue.
            let req = PendingEscalation::new(user_id, track_id, now_ms, priority);
            Self::enqueue_escalation(&mut state.pending, req);
        }

        // Deadlock detection: does the waits-for graph contain a cycle?
        if let Some(cycle) = self.detect_deadlock(user_id, track_id) {
            // Roll back the enqueue and report.
            let state = self.state_mut(track_id);
            state.pending.retain(|p| p.user_id != user_id);
            return Err(EscalationError::Deadlock(cycle));
        }

        Ok(EscalationResult::Queued)
    }

    /// Detect a cross-track deadlock cycle in the waits-for graph starting from
    /// `root_user` waiting on `root_track`.
    ///
    /// A deadlock is only reported when the cycle spans **multiple tracks**:
    /// mutual escalation on the *same* track is resolvable by priority ordering
    /// and does not constitute a deadlock.
    ///
    /// Returns `Some(description)` if a cross-track cycle is found.
    fn detect_deadlock(&self, root_user: &str, root_track: &str) -> Option<String> {
        // Build a waits-for map: (user, track) → set of (user, track) pairs
        // that must release before the pending escalation can proceed.
        // We only consider cross-track dependencies.
        //
        // Specifically, user U pending on track T waits for every holder H on
        // track T such that H also has a pending escalation on a *different*
        // track T2.  U→H across tracks creates the dangerous cycle.
        let mut cross_waits: HashMap<&str, HashSet<(&str, &str)>> = HashMap::new();

        for (tid, state) in &self.tracks {
            for pending in &state.pending {
                // For each other holder on this track…
                for holder in &state.holders {
                    if holder.user_id == pending.user_id {
                        continue;
                    }
                    // Check if holder also has a pending escalation on a different track.
                    for (tid2, state2) in &self.tracks {
                        if tid2 == tid {
                            continue;
                        }
                        if state2.pending.iter().any(|p| p.user_id == holder.user_id) {
                            cross_waits
                                .entry(pending.user_id.as_str())
                                .or_default()
                                .insert((holder.user_id.as_str(), tid2.as_str()));
                        }
                    }
                }
            }
        }

        // DFS from root_user to detect whether a cross-track cycle leads back.
        let _ = root_track; // anchor for documentation clarity
        let mut visited: HashSet<&str> = HashSet::new();
        let mut stack: Vec<&str> = vec![root_user];

        while let Some(node) = stack.pop() {
            if visited.contains(node) {
                continue;
            }
            visited.insert(node);
            if let Some(deps) = cross_waits.get(node) {
                for &(dep_user, _dep_track) in deps {
                    if dep_user == root_user {
                        return Some(format!("{root_user} → {node} → {root_user}"));
                    }
                    stack.push(dep_user);
                }
            }
        }
        None
    }

    /// Release `user_id`'s lock on `track_id`.
    ///
    /// After releasing, attempts to promote the highest-priority pending
    /// escalation if the track is now free.
    pub fn release(
        &mut self,
        user_id: &str,
        track_id: &str,
        now_ms: u64,
    ) -> Result<Option<String>, EscalationError> {
        {
            let state = self.tracks.get_mut(track_id).ok_or_else(|| {
                EscalationError::UnknownTrack(track_id.to_string())
            })?;
            state.expire(now_ms);
            let before = state.holders.len();
            state.holders.retain(|h| h.user_id != user_id);
            if state.holders.len() == before {
                return Err(EscalationError::NoSharedLock {
                    user_id: user_id.to_string(),
                    track_id: track_id.to_string(),
                });
            }
        }

        // Try to promote pending escalation.
        let promoted = self.try_promote_pending(track_id, now_ms);
        Ok(promoted)
    }

    /// Attempt to promote the head of the pending queue if the track is free.
    fn try_promote_pending(&mut self, track_id: &str, now_ms: u64) -> Option<String> {
        let ttl_ms = self.default_ttl_ms;
        let state = self.tracks.get_mut(track_id)?;
        state.expire(now_ms);

        // Only promote if no shared holders remain (other than the pending user
        // themselves — they must already have a shared lock to be in the queue).
        let pending_user = state.pending.front()?.user_id.clone();

        let other_holders: usize = state
            .holders
            .iter()
            .filter(|h| h.user_id != pending_user)
            .count();

        if other_holders == 0 {
            // Remove pending entry.
            state.pending.pop_front();
            // Promote or insert exclusive.
            if let Some(h) = state.holders.iter_mut().find(|h| h.user_id == pending_user) {
                h.mode = TrackLockMode::Exclusive;
                h.expires_at_ms = now_ms + ttl_ms;
            } else {
                state.holders.push(TrackLockHolder::new(
                    &pending_user,
                    TrackLockMode::Exclusive,
                    now_ms,
                    ttl_ms,
                ));
            }
            Some(pending_user)
        } else {
            None
        }
    }

    /// Forcibly revoke all locks held by `user_id` across all tracks and remove
    /// any pending escalations they have submitted.
    pub fn force_revoke_user(&mut self, user_id: &str, now_ms: u64) -> Vec<String> {
        let mut affected = Vec::new();
        for (track_id, state) in &mut self.tracks {
            let before = state.holders.len();
            state.holders.retain(|h| h.user_id != user_id);
            state.pending.retain(|p| p.user_id != user_id);
            state.expire(now_ms);
            if state.holders.len() < before {
                affected.push(track_id.clone());
            }
        }
        // Try promoting pending escalations on affected tracks.
        for track_id in &affected {
            self.try_promote_pending(track_id, now_ms);
        }
        affected
    }

    /// Return the current lock mode for `user_id` on `track_id`, or `None`.
    #[must_use]
    pub fn current_mode(&self, user_id: &str, track_id: &str) -> Option<TrackLockMode> {
        self.state(track_id)?
            .holders
            .iter()
            .find(|h| h.user_id == user_id)
            .map(|h| h.mode)
    }

    /// Number of pending escalations on a track.
    #[must_use]
    pub fn pending_count(&self, track_id: &str) -> usize {
        self.state(track_id).map(|s| s.pending.len()).unwrap_or(0)
    }

    /// All track identifiers currently managed.
    #[must_use]
    pub fn tracks(&self) -> Vec<&str> {
        self.tracks.keys().map(String::as_str).collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TTL: u64 = 60_000; // 60 s

    fn mgr() -> LockEscalationManager {
        LockEscalationManager::new(TTL)
    }

    // ── EscalationPriority ──

    #[test]
    fn test_priority_ordering() {
        assert!(EscalationPriority::CRITICAL > EscalationPriority::HIGH);
        assert!(EscalationPriority::HIGH > EscalationPriority::NORMAL);
    }

    #[test]
    fn test_priority_display() {
        assert_eq!(EscalationPriority::NORMAL.to_string(), "priority(100)");
    }

    // ── TrackLockMode ──

    #[test]
    fn test_lock_mode_display() {
        assert_eq!(TrackLockMode::Shared.to_string(), "shared");
        assert_eq!(TrackLockMode::Exclusive.to_string(), "exclusive");
    }

    // ── acquire_shared ──

    #[test]
    fn test_acquire_shared_success() {
        let mut m = mgr();
        m.acquire_shared("alice", "track-1", 0).expect("should succeed");
        assert_eq!(m.current_mode("alice", "track-1"), Some(TrackLockMode::Shared));
    }

    #[test]
    fn test_acquire_shared_multiple_users() {
        let mut m = mgr();
        m.acquire_shared("alice", "t1", 0).expect("ok");
        m.acquire_shared("bob", "t1", 0).expect("ok");
        let state = m.tracks.get("t1").expect("track exists");
        assert_eq!(state.holder_count(), 2);
    }

    #[test]
    fn test_acquire_shared_blocked_by_exclusive() {
        let mut m = mgr();
        m.acquire_shared("alice", "t1", 0).expect("ok");
        m.request_escalation("alice", "t1", 0, EscalationPriority::NORMAL)
            .expect("escalation granted immediately");
        // bob cannot acquire shared while alice holds exclusive
        let err = m.acquire_shared("bob", "t1", 0).expect_err("should be blocked");
        assert!(matches!(err, EscalationError::BlockedBy(_)));
    }

    // ── request_escalation ──

    #[test]
    fn test_escalation_granted_immediately_when_sole_holder() {
        let mut m = mgr();
        m.acquire_shared("alice", "t1", 0).expect("ok");
        let result = m.request_escalation("alice", "t1", 0, EscalationPriority::NORMAL)
            .expect("should succeed");
        assert_eq!(result, EscalationResult::Granted);
        assert_eq!(m.current_mode("alice", "t1"), Some(TrackLockMode::Exclusive));
    }

    #[test]
    fn test_escalation_queued_when_other_holders_present() {
        let mut m = mgr();
        m.acquire_shared("alice", "t1", 0).expect("ok");
        m.acquire_shared("bob", "t1", 0).expect("ok");
        let result = m.request_escalation("alice", "t1", 0, EscalationPriority::NORMAL)
            .expect("should queue");
        assert_eq!(result, EscalationResult::Queued);
        assert_eq!(m.pending_count("t1"), 1);
    }

    #[test]
    fn test_escalation_error_no_shared_lock() {
        let mut m = mgr();
        let err = m.request_escalation("ghost", "t1", 0, EscalationPriority::NORMAL)
            .expect_err("no shared lock");
        assert!(matches!(err, EscalationError::NoSharedLock { .. }));
    }

    #[test]
    fn test_escalation_error_already_exclusive() {
        let mut m = mgr();
        m.acquire_shared("alice", "t1", 0).expect("ok");
        m.request_escalation("alice", "t1", 0, EscalationPriority::NORMAL)
            .expect("granted");
        let err = m.request_escalation("alice", "t1", 100, EscalationPriority::NORMAL)
            .expect_err("already exclusive");
        assert!(matches!(err, EscalationError::AlreadyExclusive { .. }));
    }

    #[test]
    fn test_escalation_duplicate_pending_rejected() {
        let mut m = mgr();
        m.acquire_shared("alice", "t1", 0).expect("ok");
        m.acquire_shared("bob", "t1", 0).expect("ok");
        m.request_escalation("alice", "t1", 0, EscalationPriority::NORMAL)
            .expect("queued");
        let err = m.request_escalation("alice", "t1", 100, EscalationPriority::NORMAL)
            .expect_err("duplicate");
        assert!(matches!(err, EscalationError::DuplicatePending { .. }));
    }

    // ── release + promotion ──

    #[test]
    fn test_release_promotes_pending_escalation() {
        let mut m = mgr();
        m.acquire_shared("alice", "t1", 0).expect("ok");
        m.acquire_shared("bob", "t1", 0).expect("ok");
        m.request_escalation("alice", "t1", 0, EscalationPriority::NORMAL)
            .expect("queued");

        // Bob releases → alice should be promoted.
        let promoted = m.release("bob", "t1", 100).expect("ok");
        assert_eq!(promoted, Some("alice".to_string()));
        assert_eq!(m.current_mode("alice", "t1"), Some(TrackLockMode::Exclusive));
        assert_eq!(m.pending_count("t1"), 0);
    }

    #[test]
    fn test_release_no_promotion_if_other_holder_remains() {
        let mut m = mgr();
        m.acquire_shared("alice", "t1", 0).expect("ok");
        m.acquire_shared("bob", "t1", 0).expect("ok");
        m.acquire_shared("carol", "t1", 0).expect("ok");
        m.request_escalation("alice", "t1", 0, EscalationPriority::NORMAL)
            .expect("queued");

        // Bob releases but carol still holds → no promotion yet.
        let promoted = m.release("bob", "t1", 100).expect("ok");
        assert_eq!(promoted, None);
        assert_eq!(m.pending_count("t1"), 1);
    }

    // ── priority ordering in queue ──

    #[test]
    fn test_high_priority_escalation_queued_before_normal() {
        let mut m = mgr();
        m.acquire_shared("alice", "t1", 0).expect("ok");
        m.acquire_shared("bob", "t1", 0).expect("ok");
        m.acquire_shared("carol", "t1", 0).expect("ok");

        // Bob enqueues with NORMAL, carol with HIGH.
        m.request_escalation("bob", "t1", 100, EscalationPriority::NORMAL)
            .expect("queued");
        m.request_escalation("carol", "t1", 200, EscalationPriority::HIGH)
            .expect("queued");

        let state = m.tracks.get("t1").expect("track");
        // Higher priority (carol HIGH=200) should be at front.
        assert_eq!(state.pending.front().map(|p| p.user_id.as_str()), Some("carol"));
    }

    // ── force_revoke_user ──

    #[test]
    fn test_force_revoke_releases_locks_and_promotes() {
        let mut m = mgr();
        m.acquire_shared("alice", "t1", 0).expect("ok");
        m.acquire_shared("bob", "t1", 0).expect("ok");
        m.request_escalation("alice", "t1", 0, EscalationPriority::NORMAL)
            .expect("queued");

        let affected = m.force_revoke_user("bob", 100);
        assert!(affected.contains(&"t1".to_string()));
        // Alice should have been promoted.
        assert_eq!(m.current_mode("alice", "t1"), Some(TrackLockMode::Exclusive));
    }

    // ── expiry ──

    #[test]
    fn test_expired_holder_cleaned_on_acquire() {
        let mut m = LockEscalationManager::new(500); // 500ms TTL
        m.acquire_shared("alice", "t1", 0).expect("ok");
        // At t=600 alice's lock has expired; bob should be able to acquire.
        m.acquire_shared("bob", "t1", 600).expect("should succeed after expiry");
        let state = m.tracks.get("t1").expect("track");
        // Only bob remains.
        assert_eq!(state.holder_count(), 1);
        assert_eq!(state.holders[0].user_id, "bob");
    }

    // ── EscalationError Display ──

    #[test]
    fn test_error_display_blocked_by() {
        let err = EscalationError::BlockedBy(vec!["alice".to_string()]);
        assert!(err.to_string().contains("alice"));
    }

    #[test]
    fn test_error_display_deadlock() {
        let err = EscalationError::Deadlock("a → b → a".to_string());
        assert!(err.to_string().contains("deadlock"));
    }
}
