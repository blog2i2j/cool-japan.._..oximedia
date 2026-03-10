#![allow(dead_code)]
//! Asset locking for collaborative editing.
//!
//! Provides `Read`, `Write`, and `Exclusive` lock types with a `LockManager`
//! that enforces compatibility rules to prevent conflicting concurrent access.

use chrono::{DateTime, Utc};
use std::collections::HashMap;

/// The kind of lock being requested or held.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LockType {
    /// Shared read lock – multiple holders permitted simultaneously.
    Read,
    /// Exclusive write lock – only one holder, no concurrent reads.
    Write,
    /// Fully exclusive lock – no other lock of any type may coexist.
    Exclusive,
}

impl LockType {
    /// Returns `true` if this lock type allows other locks to coexist concurrently.
    pub fn allows_concurrent(&self) -> bool {
        matches!(self, LockType::Read)
    }

    /// Returns `true` if `self` is compatible with `other` being held simultaneously.
    pub fn is_compatible_with(&self, other: LockType) -> bool {
        match (self, other) {
            (LockType::Read, LockType::Read) => true,
            _ => false,
        }
    }

    /// Human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            LockType::Read => "Read",
            LockType::Write => "Write",
            LockType::Exclusive => "Exclusive",
        }
    }
}

/// A lock held by a user on a particular asset.
#[derive(Debug, Clone)]
pub struct AssetLock {
    /// Unique lock token.
    pub token: String,
    /// Asset being locked.
    pub asset_id: String,
    /// User who holds the lock.
    pub holder_id: String,
    /// Type of lock.
    pub lock_type: LockType,
    /// When the lock was acquired.
    pub acquired_at: DateTime<Utc>,
    /// Optional TTL expiry; `None` means it does not auto-expire.
    pub expires_at: Option<DateTime<Utc>>,
}

impl AssetLock {
    /// Create a new lock.
    pub fn new(
        token: impl Into<String>,
        asset_id: impl Into<String>,
        holder_id: impl Into<String>,
        lock_type: LockType,
    ) -> Self {
        Self {
            token: token.into(),
            asset_id: asset_id.into(),
            holder_id: holder_id.into(),
            lock_type,
            acquired_at: Utc::now(),
            expires_at: None,
        }
    }

    /// Returns `true` if `user_id` is the holder of this lock.
    pub fn is_held_by(&self, user_id: &str) -> bool {
        self.holder_id == user_id
    }

    /// Returns `true` when the lock has expired (only when `expires_at` is set).
    pub fn is_expired(&self) -> bool {
        self.expires_at.map_or(false, |exp| Utc::now() >= exp)
    }
}

/// Manages all active locks across assets.
///
/// Locks are indexed by asset_id, allowing fast compatibility checks.
#[derive(Debug, Default)]
pub struct LockManager {
    /// Active locks keyed by token.
    locks_by_token: HashMap<String, AssetLock>,
    /// Mapping from asset_id to a list of active lock tokens.
    asset_index: HashMap<String, Vec<String>>,
    /// Monotonically-increasing counter for token generation.
    next_token: u64,
}

impl LockManager {
    /// Create an empty lock manager.
    pub fn new() -> Self {
        Self::default()
    }

    /// Attempt to acquire a lock of `lock_type` on `asset_id` for `user_id`.
    ///
    /// Returns the lock token on success, or an `Err` describing why the
    /// request was incompatible with an existing lock.
    pub fn acquire(
        &mut self,
        asset_id: &str,
        user_id: &str,
        lock_type: LockType,
    ) -> Result<String, String> {
        // Remove expired locks first
        self.purge_expired(asset_id);

        if let Some(tokens) = self.asset_index.get(asset_id) {
            for token in tokens {
                if let Some(existing) = self.locks_by_token.get(token.as_str()) {
                    if !lock_type.is_compatible_with(existing.lock_type)
                        || !existing.lock_type.is_compatible_with(lock_type)
                    {
                        return Err(format!(
                            "Asset '{}' already has an incompatible {} lock held by '{}'",
                            asset_id,
                            existing.lock_type.name(),
                            existing.holder_id
                        ));
                    }
                }
            }
        }

        self.next_token += 1;
        let token = format!("lock_{}", self.next_token);
        let lock = AssetLock::new(&token, asset_id, user_id, lock_type);

        self.asset_index
            .entry(asset_id.to_string())
            .or_default()
            .push(token.clone());
        self.locks_by_token.insert(token.clone(), lock);

        Ok(token)
    }

    /// Release a lock by its token. Returns `true` if the lock existed.
    pub fn release(&mut self, token: &str) -> bool {
        if let Some(lock) = self.locks_by_token.remove(token) {
            if let Some(tokens) = self.asset_index.get_mut(&lock.asset_id) {
                tokens.retain(|t| t != token);
            }
            true
        } else {
            false
        }
    }

    /// Returns `true` if `asset_id` currently has at least one active (non-expired) lock.
    pub fn is_locked(&self, asset_id: &str) -> bool {
        self.asset_index.get(asset_id).map_or(false, |tokens| {
            tokens.iter().any(|t| {
                self.locks_by_token
                    .get(t.as_str())
                    .map_or(false, |l| !l.is_expired())
            })
        })
    }

    /// Retrieve a lock by its token.
    pub fn get_lock(&self, token: &str) -> Option<&AssetLock> {
        self.locks_by_token.get(token)
    }

    /// Purge expired locks for a specific asset from internal state.
    fn purge_expired(&mut self, asset_id: &str) {
        let expired_tokens: Vec<String> = self
            .asset_index
            .get(asset_id)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .filter(|t| {
                self.locks_by_token
                    .get(t.as_str())
                    .map_or(true, |l| l.is_expired())
            })
            .collect();

        for token in &expired_tokens {
            self.locks_by_token.remove(token.as_str());
        }
        if let Some(tokens) = self.asset_index.get_mut(asset_id) {
            tokens.retain(|t| !expired_tokens.contains(t));
        }
    }

    /// Total active (non-expired) lock count across all assets.
    pub fn active_lock_count(&self) -> usize {
        self.locks_by_token
            .values()
            .filter(|l| !l.is_expired())
            .count()
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    // LockType tests

    #[test]
    fn test_read_allows_concurrent() {
        assert!(LockType::Read.allows_concurrent());
    }

    #[test]
    fn test_write_does_not_allow_concurrent() {
        assert!(!LockType::Write.allows_concurrent());
    }

    #[test]
    fn test_exclusive_does_not_allow_concurrent() {
        assert!(!LockType::Exclusive.allows_concurrent());
    }

    #[test]
    fn test_read_compatible_with_read() {
        assert!(LockType::Read.is_compatible_with(LockType::Read));
    }

    #[test]
    fn test_write_incompatible_with_read() {
        assert!(!LockType::Write.is_compatible_with(LockType::Read));
    }

    #[test]
    fn test_lock_type_names_non_empty() {
        for lt in &[LockType::Read, LockType::Write, LockType::Exclusive] {
            assert!(!lt.name().is_empty());
        }
    }

    // AssetLock tests

    #[test]
    fn test_is_held_by() {
        let l = AssetLock::new("t1", "asset1", "user1", LockType::Write);
        assert!(l.is_held_by("user1"));
        assert!(!l.is_held_by("user2"));
    }

    #[test]
    fn test_not_expired_by_default() {
        let l = AssetLock::new("t1", "asset1", "user1", LockType::Read);
        assert!(!l.is_expired());
    }

    #[test]
    fn test_expired_when_past_expiry() {
        let mut l = AssetLock::new("t1", "asset1", "user1", LockType::Read);
        l.expires_at = Some(Utc::now() - Duration::seconds(1));
        assert!(l.is_expired());
    }

    // LockManager tests

    #[test]
    fn test_acquire_write_lock_on_free_asset() {
        let mut mgr = LockManager::new();
        let token = mgr
            .acquire("a1", "u1", LockType::Write)
            .expect("collab test operation should succeed");
        assert!(!token.is_empty());
        assert!(mgr.is_locked("a1"));
    }

    #[test]
    fn test_acquire_two_read_locks() {
        let mut mgr = LockManager::new();
        assert!(mgr.acquire("a1", "u1", LockType::Read).is_ok());
        assert!(mgr.acquire("a1", "u2", LockType::Read).is_ok());
    }

    #[test]
    fn test_acquire_write_blocked_by_existing_write() {
        let mut mgr = LockManager::new();
        mgr.acquire("a1", "u1", LockType::Write)
            .expect("collab test operation should succeed");
        let result = mgr.acquire("a1", "u2", LockType::Write);
        assert!(result.is_err());
    }

    #[test]
    fn test_acquire_read_blocked_by_write() {
        let mut mgr = LockManager::new();
        mgr.acquire("a1", "u1", LockType::Write)
            .expect("collab test operation should succeed");
        let result = mgr.acquire("a1", "u2", LockType::Read);
        assert!(result.is_err());
    }

    #[test]
    fn test_release_lock() {
        let mut mgr = LockManager::new();
        let token = mgr
            .acquire("a1", "u1", LockType::Write)
            .expect("collab test operation should succeed");
        assert!(mgr.release(&token));
        assert!(!mgr.is_locked("a1"));
    }

    #[test]
    fn test_release_nonexistent_returns_false() {
        let mut mgr = LockManager::new();
        assert!(!mgr.release("fake_token"));
    }

    #[test]
    fn test_is_locked_false_on_unknown_asset() {
        let mgr = LockManager::new();
        assert!(!mgr.is_locked("unknown_asset"));
    }

    #[test]
    fn test_active_lock_count() {
        let mut mgr = LockManager::new();
        mgr.acquire("a1", "u1", LockType::Write)
            .expect("collab test operation should succeed");
        mgr.acquire("a2", "u2", LockType::Read)
            .expect("collab test operation should succeed");
        assert_eq!(mgr.active_lock_count(), 2);
    }

    #[test]
    fn test_exclusive_blocks_all_others() {
        let mut mgr = LockManager::new();
        mgr.acquire("a1", "u1", LockType::Exclusive)
            .expect("collab test operation should succeed");
        assert!(mgr.acquire("a1", "u2", LockType::Read).is_err());
        assert!(mgr.acquire("a1", "u2", LockType::Write).is_err());
    }
}
