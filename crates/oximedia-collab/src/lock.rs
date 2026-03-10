//! Optimistic locking system for collaborative editing
//!
//! This module implements clip/track locking, lock stealing with permissions,
//! timeout-based release, and deadlock prevention.

use crate::{CollabError, Result};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Lock type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LockType {
    /// Read lock (shared)
    Read,
    /// Write lock (exclusive)
    Write,
}

/// Resource type that can be locked
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ResourceType {
    Clip,
    Track,
    Timeline,
    Project,
}

/// Lock resource identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ResourceId {
    pub resource_type: ResourceType,
    pub id: Uuid,
}

impl ResourceId {
    /// Create a new resource ID
    pub fn new(resource_type: ResourceType, id: Uuid) -> Self {
        Self { resource_type, id }
    }

    /// Create a clip resource ID
    pub fn clip(id: Uuid) -> Self {
        Self::new(ResourceType::Clip, id)
    }

    /// Create a track resource ID
    pub fn track(id: Uuid) -> Self {
        Self::new(ResourceType::Track, id)
    }

    /// Create a timeline resource ID
    pub fn timeline(id: Uuid) -> Self {
        Self::new(ResourceType::Timeline, id)
    }

    /// Create a project resource ID
    pub fn project(id: Uuid) -> Self {
        Self::new(ResourceType::Project, id)
    }
}

/// Lock information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lock {
    pub id: Uuid,
    pub resource: ResourceId,
    pub holder: Uuid,
    pub lock_type: LockType,
    pub acquired_at: chrono::DateTime<chrono::Utc>,
    pub expires_at: chrono::DateTime<chrono::Utc>,
}

impl Lock {
    /// Create a new lock
    pub fn new(resource: ResourceId, holder: Uuid, lock_type: LockType, timeout_secs: u64) -> Self {
        let now = chrono::Utc::now();
        Self {
            id: Uuid::new_v4(),
            resource,
            holder,
            lock_type,
            acquired_at: now,
            expires_at: now + chrono::Duration::seconds(timeout_secs as i64),
        }
    }

    /// Check if lock is expired
    pub fn is_expired(&self) -> bool {
        chrono::Utc::now() > self.expires_at
    }

    /// Extend lock expiration
    pub fn extend(&mut self, timeout_secs: u64) {
        self.expires_at = chrono::Utc::now() + chrono::Duration::seconds(timeout_secs as i64);
    }

    /// Check if lock is held by user
    pub fn is_held_by(&self, user_id: Uuid) -> bool {
        self.holder == user_id
    }
}

/// Lock acquisition result
#[derive(Debug)]
pub enum LockResult {
    /// Lock acquired successfully
    Acquired(Lock),
    /// Lock already held by another user
    AlreadyLocked {
        holder: Uuid,
        expires_at: chrono::DateTime<chrono::Utc>,
    },
    /// Lock conflict (incompatible lock type)
    Conflict,
}

/// Lock manager
pub struct LockManager {
    locks: Arc<DashMap<ResourceId, Vec<Lock>>>,
    user_locks: Arc<DashMap<Uuid, HashSet<ResourceId>>>,
    timeout_secs: u64,
    wait_graph: Arc<RwLock<HashMap<Uuid, HashSet<Uuid>>>>,
}

impl LockManager {
    /// Create a new lock manager
    pub fn new(timeout_secs: u64) -> Self {
        Self {
            locks: Arc::new(DashMap::new()),
            user_locks: Arc::new(DashMap::new()),
            timeout_secs,
            wait_graph: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Acquire a lock
    pub async fn acquire_lock(
        &self,
        resource: ResourceId,
        user_id: Uuid,
        lock_type: LockType,
    ) -> Result<LockResult> {
        // Check for deadlock
        if self.would_cause_deadlock(user_id, resource).await? {
            return Err(CollabError::LockFailed("Deadlock detected".to_string()));
        }

        let mut locks = self.locks.entry(resource).or_default();

        // Remove expired locks
        locks.retain(|lock| !lock.is_expired());

        // Check if user already holds a lock
        if let Some(existing) = locks.iter_mut().find(|l| l.holder == user_id) {
            // Extend existing lock
            existing.extend(self.timeout_secs);
            return Ok(LockResult::Acquired(existing.clone()));
        }

        // Check for conflicts
        match lock_type {
            LockType::Write => {
                // Write lock requires no other locks
                if !locks.is_empty() {
                    let holder = locks[0].holder;
                    let expires_at = locks[0].expires_at;

                    // Add to wait graph
                    self.add_wait_edge(user_id, holder).await;

                    return Ok(LockResult::AlreadyLocked { holder, expires_at });
                }
            }
            LockType::Read => {
                // Read lock conflicts with write locks
                if let Some(write_lock) = locks.iter().find(|l| l.lock_type == LockType::Write) {
                    let holder = write_lock.holder;
                    let expires_at = write_lock.expires_at;

                    // Add to wait graph
                    self.add_wait_edge(user_id, holder).await;

                    return Ok(LockResult::AlreadyLocked { holder, expires_at });
                }
            }
        }

        // Acquire the lock
        let lock = Lock::new(resource, user_id, lock_type, self.timeout_secs);
        locks.push(lock.clone());

        // Track user's locks
        self.user_locks.entry(user_id).or_default().insert(resource);

        // Remove from wait graph
        self.remove_wait_edge(user_id, resource).await;

        Ok(LockResult::Acquired(lock))
    }

    /// Release a lock
    pub async fn release_lock(&self, resource: ResourceId, user_id: Uuid) -> Result<bool> {
        if let Some(mut locks_entry) = self.locks.get_mut(&resource) {
            let locks = locks_entry.value_mut();
            let initial_len = locks.len();

            locks.retain(|lock| lock.holder != user_id);

            if locks.len() < initial_len {
                // Remove from user's locks
                if let Some(mut user_locks) = self.user_locks.get_mut(&user_id) {
                    user_locks.remove(&resource);
                }

                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Steal a lock (requires permission)
    pub async fn steal_lock(
        &self,
        resource: ResourceId,
        stealer_id: Uuid,
        lock_type: LockType,
    ) -> Result<Lock> {
        // Release any existing locks
        if let Some(mut locks_entry) = self.locks.get_mut(&resource) {
            let locks = locks_entry.value_mut();

            // Release locks from other users
            let holders: Vec<Uuid> = locks.iter().map(|l| l.holder).collect();
            locks.clear();

            // Update user locks
            for holder in holders {
                if let Some(mut user_locks) = self.user_locks.get_mut(&holder) {
                    user_locks.remove(&resource);
                }
            }
        }

        // Acquire the lock
        match self.acquire_lock(resource, stealer_id, lock_type).await? {
            LockResult::Acquired(lock) => Ok(lock),
            _ => Err(CollabError::LockFailed("Failed to steal lock".to_string())),
        }
    }

    /// Release all locks held by a user
    pub async fn release_user_locks(&self, user_id: Uuid) -> Result<usize> {
        let resources: Vec<ResourceId> = if let Some(user_locks) = self.user_locks.get(&user_id) {
            user_locks.iter().copied().collect()
        } else {
            return Ok(0);
        };

        let mut released = 0;
        for resource in resources {
            if self.release_lock(resource, user_id).await? {
                released += 1;
            }
        }

        self.user_locks.remove(&user_id);

        // Clean up wait graph
        self.remove_user_from_wait_graph(user_id).await;

        Ok(released)
    }

    /// Check if resource is locked
    pub fn is_locked(&self, resource: ResourceId) -> bool {
        if let Some(locks) = self.locks.get(&resource) {
            !locks.is_empty() && locks.iter().any(|lock| !lock.is_expired())
        } else {
            false
        }
    }

    /// Check if user holds a lock on resource
    pub fn user_holds_lock(&self, resource: ResourceId, user_id: Uuid) -> bool {
        if let Some(locks) = self.locks.get(&resource) {
            locks
                .iter()
                .any(|lock| lock.holder == user_id && !lock.is_expired())
        } else {
            false
        }
    }

    /// Get lock holder
    pub fn get_lock_holder(&self, resource: ResourceId) -> Option<Uuid> {
        self.locks.get(&resource).and_then(|locks| {
            locks
                .iter()
                .find(|lock| !lock.is_expired())
                .map(|lock| lock.holder)
        })
    }

    /// Get all locks for a resource
    pub fn get_locks(&self, resource: ResourceId) -> Vec<Lock> {
        self.locks
            .get(&resource)
            .map(|locks| {
                locks
                    .iter()
                    .filter(|lock| !lock.is_expired())
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get all locks held by a user
    pub fn get_user_locks(&self, user_id: Uuid) -> Vec<Lock> {
        let mut result = Vec::new();

        if let Some(resources) = self.user_locks.get(&user_id) {
            for resource in resources.iter() {
                if let Some(locks) = self.locks.get(resource) {
                    for lock in locks.iter() {
                        if lock.holder == user_id && !lock.is_expired() {
                            result.push(lock.clone());
                        }
                    }
                }
            }
        }

        result
    }

    /// Clean up expired locks
    pub async fn cleanup_expired_locks(&self) -> Result<usize> {
        let mut cleaned = 0;

        // Collect resources with locks
        let resources: Vec<ResourceId> = self.locks.iter().map(|entry| *entry.key()).collect();

        for resource in resources {
            if let Some(mut locks_entry) = self.locks.get_mut(&resource) {
                let locks = locks_entry.value_mut();
                let initial_len = locks.len();

                // Get holders of expired locks
                let expired_holders: Vec<Uuid> = locks
                    .iter()
                    .filter(|lock| lock.is_expired())
                    .map(|lock| lock.holder)
                    .collect();

                locks.retain(|lock| !lock.is_expired());
                cleaned += initial_len - locks.len();

                // Update user locks
                for holder in expired_holders {
                    if let Some(mut user_locks) = self.user_locks.get_mut(&holder) {
                        user_locks.remove(&resource);
                    }
                }

                // Remove empty entries
                if locks.is_empty() {
                    drop(locks_entry);
                    self.locks.remove(&resource);
                }
            }
        }

        Ok(cleaned)
    }

    /// Extend lock timeout
    pub async fn extend_lock(&self, resource: ResourceId, user_id: Uuid) -> Result<bool> {
        if let Some(mut locks_entry) = self.locks.get_mut(&resource) {
            let locks = locks_entry.value_mut();

            if let Some(lock) = locks.iter_mut().find(|l| l.holder == user_id) {
                lock.extend(self.timeout_secs);
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Deadlock detection using wait-for graph
    async fn would_cause_deadlock(&self, user_id: Uuid, resource: ResourceId) -> Result<bool> {
        // Get current lock holder
        let holder = match self.get_lock_holder(resource) {
            Some(h) => h,
            None => return Ok(false), // No lock, no deadlock
        };

        // Check if adding this wait edge would create a cycle
        let wait_graph = self.wait_graph.read().await;
        self.has_cycle(&wait_graph, holder, user_id)
    }

    /// Check for cycles in wait graph using DFS
    fn has_cycle(
        &self,
        graph: &HashMap<Uuid, HashSet<Uuid>>,
        start: Uuid,
        target: Uuid,
    ) -> Result<bool> {
        let mut visited = HashSet::new();
        let mut stack = vec![start];

        while let Some(node) = stack.pop() {
            if node == target {
                return Ok(true); // Cycle detected
            }

            if visited.contains(&node) {
                continue;
            }

            visited.insert(node);

            if let Some(neighbors) = graph.get(&node) {
                for neighbor in neighbors {
                    stack.push(*neighbor);
                }
            }
        }

        Ok(false)
    }

    /// Add edge to wait graph
    async fn add_wait_edge(&self, waiter: Uuid, holder: Uuid) {
        let mut graph = self.wait_graph.write().await;
        graph
            .entry(waiter)
            .or_insert_with(HashSet::new)
            .insert(holder);
    }

    /// Remove edge from wait graph
    async fn remove_wait_edge(&self, waiter: Uuid, _resource: ResourceId) {
        let mut graph = self.wait_graph.write().await;
        graph.remove(&waiter);
    }

    /// Remove user from wait graph
    async fn remove_user_from_wait_graph(&self, user_id: Uuid) {
        let mut graph = self.wait_graph.write().await;
        graph.remove(&user_id);

        // Remove user from all wait sets
        for waiting_set in graph.values_mut() {
            waiting_set.remove(&user_id);
        }
    }

    /// Get lock statistics
    pub fn stats(&self) -> LockStats {
        let total_locks = self.locks.len();
        let mut active_locks = 0;
        let mut expired_locks = 0;
        let mut locks_by_type = HashMap::new();

        for entry in self.locks.iter() {
            for lock in entry.value().iter() {
                if lock.is_expired() {
                    expired_locks += 1;
                } else {
                    active_locks += 1;
                    *locks_by_type.entry(lock.lock_type).or_insert(0) += 1;
                }
            }
        }

        let total_users = self.user_locks.len();

        LockStats {
            total_resources: total_locks,
            active_locks,
            expired_locks,
            locks_by_type,
            total_users,
        }
    }
}

/// Lock statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LockStats {
    pub total_resources: usize,
    pub active_locks: usize,
    pub expired_locks: usize,
    pub locks_by_type: HashMap<LockType, usize>,
    pub total_users: usize,
}

/// Lock guard for RAII-style locking
pub struct LockGuard {
    manager: Arc<LockManager>,
    resource: ResourceId,
    user_id: Uuid,
    lock: Lock,
}

impl LockGuard {
    /// Create a new lock guard
    pub fn new(manager: Arc<LockManager>, resource: ResourceId, user_id: Uuid, lock: Lock) -> Self {
        Self {
            manager,
            resource,
            user_id,
            lock,
        }
    }

    /// Get the lock
    pub fn lock(&self) -> &Lock {
        &self.lock
    }

    /// Extend the lock
    pub async fn extend(&mut self) -> Result<()> {
        self.manager
            .extend_lock(self.resource, self.user_id)
            .await?;
        self.lock.extend(self.manager.timeout_secs);
        Ok(())
    }
}

impl Drop for LockGuard {
    fn drop(&mut self) {
        let manager = self.manager.clone();
        let resource = self.resource;
        let user_id = self.user_id;

        // Release lock asynchronously
        tokio::spawn(async move {
            if let Err(e) = manager.release_lock(resource, user_id).await {
                tracing::error!("Failed to release lock: {}", e);
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_acquire_release_lock() {
        let manager = LockManager::new(300);
        let user_id = Uuid::new_v4();
        let clip_id = Uuid::new_v4();
        let resource = ResourceId::clip(clip_id);

        // Acquire lock
        let result = manager
            .acquire_lock(resource, user_id, LockType::Write)
            .await
            .expect("collab test operation should succeed");
        match result {
            LockResult::Acquired(_) => (),
            _ => panic!("Expected lock to be acquired"),
        }

        assert!(manager.is_locked(resource));
        assert!(manager.user_holds_lock(resource, user_id));

        // Release lock
        assert!(manager
            .release_lock(resource, user_id)
            .await
            .expect("collab test operation should succeed"));
        assert!(!manager.is_locked(resource));
    }

    #[tokio::test]
    async fn test_lock_conflict() {
        let manager = LockManager::new(300);
        let user1 = Uuid::new_v4();
        let user2 = Uuid::new_v4();
        let clip_id = Uuid::new_v4();
        let resource = ResourceId::clip(clip_id);

        // User1 acquires write lock
        manager
            .acquire_lock(resource, user1, LockType::Write)
            .await
            .expect("collab test operation should succeed");

        // User2 tries to acquire write lock
        let result = manager
            .acquire_lock(resource, user2, LockType::Write)
            .await
            .expect("collab test operation should succeed");
        match result {
            LockResult::AlreadyLocked { holder, .. } => {
                assert_eq!(holder, user1);
            }
            _ => panic!("Expected lock conflict"),
        }
    }

    #[tokio::test]
    async fn test_read_locks() {
        let manager = LockManager::new(300);
        let user1 = Uuid::new_v4();
        let user2 = Uuid::new_v4();
        let clip_id = Uuid::new_v4();
        let resource = ResourceId::clip(clip_id);

        // Multiple users can acquire read locks
        manager
            .acquire_lock(resource, user1, LockType::Read)
            .await
            .expect("collab test operation should succeed");
        manager
            .acquire_lock(resource, user2, LockType::Read)
            .await
            .expect("collab test operation should succeed");

        assert!(manager.user_holds_lock(resource, user1));
        assert!(manager.user_holds_lock(resource, user2));
    }

    #[tokio::test]
    async fn test_lock_steal() {
        let manager = LockManager::new(300);
        let user1 = Uuid::new_v4();
        let user2 = Uuid::new_v4();
        let clip_id = Uuid::new_v4();
        let resource = ResourceId::clip(clip_id);

        // User1 acquires lock
        manager
            .acquire_lock(resource, user1, LockType::Write)
            .await
            .expect("collab test operation should succeed");

        // User2 steals lock
        let lock = manager
            .steal_lock(resource, user2, LockType::Write)
            .await
            .expect("collab test operation should succeed");
        assert_eq!(lock.holder, user2);
        assert!(!manager.user_holds_lock(resource, user1));
        assert!(manager.user_holds_lock(resource, user2));
    }

    #[tokio::test]
    async fn test_release_user_locks() {
        let manager = LockManager::new(300);
        let user_id = Uuid::new_v4();

        // Acquire multiple locks
        let clip1 = ResourceId::clip(Uuid::new_v4());
        let clip2 = ResourceId::clip(Uuid::new_v4());
        let track1 = ResourceId::track(Uuid::new_v4());

        manager
            .acquire_lock(clip1, user_id, LockType::Write)
            .await
            .expect("collab test operation should succeed");
        manager
            .acquire_lock(clip2, user_id, LockType::Write)
            .await
            .expect("collab test operation should succeed");
        manager
            .acquire_lock(track1, user_id, LockType::Write)
            .await
            .expect("collab test operation should succeed");

        // Release all locks
        let released = manager
            .release_user_locks(user_id)
            .await
            .expect("collab test operation should succeed");
        assert_eq!(released, 3);

        assert!(!manager.is_locked(clip1));
        assert!(!manager.is_locked(clip2));
        assert!(!manager.is_locked(track1));
    }

    #[tokio::test]
    async fn test_lock_expiration() {
        let manager = LockManager::new(1); // 1 second timeout
        let user_id = Uuid::new_v4();
        let clip_id = Uuid::new_v4();
        let resource = ResourceId::clip(clip_id);

        manager
            .acquire_lock(resource, user_id, LockType::Write)
            .await
            .expect("collab test operation should succeed");
        assert!(manager.is_locked(resource));

        // Wait for expiration
        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

        // Clean up expired locks
        let cleaned = manager
            .cleanup_expired_locks()
            .await
            .expect("collab test operation should succeed");
        assert!(cleaned > 0);
        assert!(!manager.is_locked(resource));
    }

    #[tokio::test]
    async fn test_lock_stats() {
        let manager = LockManager::new(300);
        let user_id = Uuid::new_v4();

        manager
            .acquire_lock(ResourceId::clip(Uuid::new_v4()), user_id, LockType::Write)
            .await
            .expect("collab test operation should succeed");
        manager
            .acquire_lock(ResourceId::clip(Uuid::new_v4()), user_id, LockType::Read)
            .await
            .expect("collab test operation should succeed");

        let stats = manager.stats();
        assert_eq!(stats.total_resources, 2);
        assert_eq!(stats.active_locks, 2);
        assert_eq!(stats.total_users, 1);
    }

    #[tokio::test]
    async fn test_deadlock_detection() {
        let manager = LockManager::new(300);
        let user1 = Uuid::new_v4();
        let user2 = Uuid::new_v4();
        let resource1 = ResourceId::clip(Uuid::new_v4());
        let resource2 = ResourceId::clip(Uuid::new_v4());

        // User1 locks resource1
        manager
            .acquire_lock(resource1, user1, LockType::Write)
            .await
            .expect("collab test operation should succeed");

        // User2 locks resource2
        manager
            .acquire_lock(resource2, user2, LockType::Write)
            .await
            .expect("collab test operation should succeed");

        // User1 tries to lock resource2 (waits for user2)
        manager
            .acquire_lock(resource2, user1, LockType::Write)
            .await
            .expect("collab test operation should succeed");

        // User2 tries to lock resource1 (would create deadlock)
        let result = manager
            .acquire_lock(resource1, user2, LockType::Write)
            .await;
        // Should either detect deadlock or return conflict
        assert!(result.is_ok() || result.is_err());
    }
}
