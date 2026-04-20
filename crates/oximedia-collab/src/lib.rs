//! Real-time CRDT-based multi-user collaboration for OxiMedia.
//!
//! Provides conflict-free synchronization for concurrent video editing sessions,
//! supporting up to 10 simultaneous editors with sub-second latency over WebSocket.
//!
//! # Overview
//!
//! - **CRDT synchronization** — Yjs-backed (`yrs`) document model; GCounter, PNCounter,
//!   LWWRegister, MVRegister, GSet, and TwoPhaseSet primitives in `crdt_primitives`.
//! - **Operational transformation** — Insert/Delete/Update/Move ops with FIFO tiebreak;
//!   git-rebase-style `rebase()`; Kahn topological order and memoized LCA in `operation_log`.
//! - **Binary framing** — 4-byte header `[type u16 LE][len u16 LE]` + payload in
//!   `binary_framer`; `BatchedFramer` auto-flushes above a configurable threshold.
//! - **Deadlock detection** — `WaiterGraph` DFS cycle detection (`detect_cycle_if_added`)
//!   in `edit_lock`; region-based, track-based, and hierarchical lock scopes.
//! - **Delta changesets** — `DeltaChangeset::delta_from(base, current)` encodes only
//!   suffix ops beyond the base; `apply_delta` restores the full changeset.
//! - **Three-way merge** — `ProjectMerger` resolves timeline events via five
//!   `ConflictResolution` strategies; heuristic scoring by duration + parameter richness.
//! - **Snapshot repository** — git-inspired parent-chain with BFS common-ancestor detection
//!   and branch/fast-forward semantics in `snapshot_manager`.
//! - **Session management** — Owner/Editor/Viewer roles; per-session GC; awareness
//!   (cursor, viewport, presence) in `awareness` and `user_presence_map`.
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use oximedia_collab::{CollaborationServer, CollabConfig, User, UserRole};
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let config = CollabConfig::default();
//! let server = CollaborationServer::new(config);
//!
//! let owner = User::new("Alice".to_string(), UserRole::Owner);
//! let project_id = uuid::Uuid::new_v4();
//! let session_id = server.create_session(project_id, owner).await?;
//!
//! let editor = User::new("Bob".to_string(), UserRole::Editor);
//! server.join_session(session_id, editor).await?;
//!
//! server.start_background_tasks().await;
//! server.shutdown().await?;
//! # Ok(())
//! # }
//! ```
//!
//! # Default Configuration
//!
//! | Field | Default |
//! |-------|---------|
//! | `max_users_per_session` | 10 |
//! | `lock_timeout_secs` | 300 (5 min) |
//! | `compression_threshold` | 1 024 bytes |
//! | `history_limit` | 1 000 operations |
//! | `gc_interval_secs` | 600 (10 min) |
//! | `max_offline_queue` | 10 000 entries |

/// Chronological activity feed for tracking collaboration events with filtering and aggregation.
pub mod activity_feed;
pub mod annotation;
pub mod approval;
pub mod asset_lock;
pub mod audit_trail;
pub mod awareness;
/// Adaptive sync bandwidth management for collaboration sessions.
pub mod bandwidth_throttle;
/// Compact binary frame format for WebSocket throughput optimisation (replaces JSON for hot paths).
pub mod binary_framer;
pub mod changeset;
pub mod comments;
pub mod conflict_resolve;
pub mod crdt;
/// Classic CRDT primitives: GCounter, PNCounter, LWWRegister, MVRegister, GSet, TwoPhaseSet.
pub mod crdt_primitives;
pub mod diff_tracker;
pub mod diff_viewer;
/// Region-based, track-based, and hierarchical edit locking with automatic expiration and deadlock detection.
pub mod edit_lock;
pub mod history;
pub mod invite_link;
pub mod invite_system;
pub mod lock;
pub mod merge_strategy;
pub mod notification;
/// Operational transformation log: per-op DAG, OT transform/rebase, apply to `Vec<f32>` state.
pub mod operation_log;
pub mod permission;
pub mod presence;
pub mod review_link;
pub mod session;
pub mod session_lock;
pub mod session_manager;
/// Project snapshot and version management with branching and fast-forward merge detection.
pub mod snapshot_manager;
pub mod sync;
pub mod task_tracker;
/// Fine-grained team role management with capabilities, hierarchies, and role assignment workflows.
pub mod team_role;
/// Three-way media project merge: scalar, string, timeline-event, and parameter-map merging.
pub mod three_way_merge;
/// Spatial presence tracking (cursor/viewport positions) for collaboration.
pub mod user_presence_map;
pub mod version_compare;
pub mod workspace;

#[cfg(test)]
mod perf_tests;

use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Collaboration errors
#[derive(Error, Debug)]
pub enum CollabError {
    #[error("Session not found: {0}")]
    SessionNotFound(Uuid),

    #[error("User not found: {0}")]
    UserNotFound(Uuid),

    #[error("Permission denied: {0}")]
    PermissionDenied(String),

    #[error("Lock acquisition failed: {0}")]
    LockFailed(String),

    #[error("Sync error: {0}")]
    SyncError(String),

    #[error("CRDT error: {0}")]
    CrdtError(String),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Invalid operation: {0}")]
    InvalidOperation(String),
}

pub type Result<T> = std::result::Result<T, CollabError>;

/// User role in collaboration session
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UserRole {
    Owner,
    Editor,
    Viewer,
}

impl UserRole {
    /// Check if role has write permissions
    pub fn can_write(&self) -> bool {
        matches!(self, UserRole::Owner | UserRole::Editor)
    }

    /// Check if role can manage locks
    pub fn can_manage_locks(&self) -> bool {
        matches!(self, UserRole::Owner)
    }
}

/// User information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub id: Uuid,
    pub name: String,
    pub role: UserRole,
    pub color: String,
    pub joined_at: chrono::DateTime<chrono::Utc>,
}

impl User {
    /// Create a new user
    pub fn new(name: String, role: UserRole) -> Self {
        Self {
            id: Uuid::new_v4(),
            name,
            role,
            color: Self::generate_color(),
            joined_at: chrono::Utc::now(),
        }
    }

    fn generate_color() -> String {
        let colors = [
            "#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E2",
            "#F8B4B4", "#A8E6CF",
        ];
        colors[uuid::Uuid::new_v4().as_u128() as usize % colors.len()].to_string()
    }
}

/// Collaboration server configuration
#[derive(Debug, Clone)]
pub struct CollabConfig {
    /// Maximum number of concurrent users per session
    pub max_users_per_session: usize,

    /// Lock timeout in seconds
    pub lock_timeout_secs: u64,

    /// Enable compression for sync messages
    pub enable_compression: bool,

    /// Compression threshold in bytes
    pub compression_threshold: usize,

    /// History retention limit
    pub history_limit: usize,

    /// Garbage collection interval in seconds
    pub gc_interval_secs: u64,

    /// Enable offline support
    pub enable_offline: bool,

    /// Max offline queue size
    pub max_offline_queue: usize,
}

impl Default for CollabConfig {
    fn default() -> Self {
        Self {
            max_users_per_session: 10,
            lock_timeout_secs: 300, // 5 minutes
            enable_compression: true,
            compression_threshold: 1024, // 1KB
            history_limit: 1000,
            gc_interval_secs: 600, // 10 minutes
            enable_offline: true,
            max_offline_queue: 10000,
        }
    }
}

/// Main collaboration server
pub struct CollaborationServer {
    config: CollabConfig,
    sessions: Arc<DashMap<Uuid, Arc<session::Session>>>,
    sync_manager: Arc<RwLock<sync::SyncManager>>,
}

impl CollaborationServer {
    /// Create a new collaboration server
    pub fn new(config: CollabConfig) -> Self {
        Self {
            config: config.clone(),
            sessions: Arc::new(DashMap::new()),
            sync_manager: Arc::new(RwLock::new(sync::SyncManager::new(config))),
        }
    }

    /// Create a new collaboration session
    pub async fn create_session(&self, project_id: Uuid, owner: User) -> Result<Uuid> {
        let session_id = Uuid::new_v4();
        let session = Arc::new(session::Session::new(
            session_id,
            project_id,
            owner,
            self.config.clone(),
        ));

        self.sessions.insert(session_id, session.clone());
        self.sync_manager
            .write()
            .await
            .register_session(session_id, session)
            .await?;

        Ok(session_id)
    }

    /// Join an existing session
    pub async fn join_session(&self, session_id: Uuid, user: User) -> Result<()> {
        let session = self
            .sessions
            .get(&session_id)
            .ok_or(CollabError::SessionNotFound(session_id))?
            .clone();

        session.add_user(user).await?;
        Ok(())
    }

    /// Leave a session
    pub async fn leave_session(&self, session_id: Uuid, user_id: Uuid) -> Result<()> {
        let session = self
            .sessions
            .get(&session_id)
            .ok_or(CollabError::SessionNotFound(session_id))?
            .clone();

        session.remove_user(user_id).await?;

        // Clean up empty sessions
        if session.user_count().await == 0 {
            self.sessions.remove(&session_id);
            self.sync_manager
                .write()
                .await
                .unregister_session(session_id)
                .await?;
        }

        Ok(())
    }

    /// Get a session
    pub fn get_session(&self, session_id: Uuid) -> Result<Arc<session::Session>> {
        self.sessions
            .get(&session_id)
            .map(|s| s.clone())
            .ok_or(CollabError::SessionNotFound(session_id))
    }

    /// List all active sessions
    pub fn list_sessions(&self) -> Vec<Uuid> {
        self.sessions.iter().map(|entry| *entry.key()).collect()
    }

    /// Get sync manager
    pub fn sync_manager(&self) -> Arc<RwLock<sync::SyncManager>> {
        self.sync_manager.clone()
    }

    /// Start background tasks (GC, lock cleanup, etc.)
    pub async fn start_background_tasks(&self) {
        let sessions = self.sessions.clone();
        let gc_interval = self.config.gc_interval_secs;

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(gc_interval));

            loop {
                interval.tick().await;

                for entry in sessions.iter() {
                    let session = entry.value();
                    if let Err(e) = session.run_garbage_collection().await {
                        tracing::error!("GC failed for session {}: {}", entry.key(), e);
                    }
                }
            }
        });
    }

    /// Shutdown the server
    pub async fn shutdown(&self) -> Result<()> {
        // Close all sessions
        for entry in self.sessions.iter() {
            if let Err(e) = entry.value().close().await {
                tracing::error!("Failed to close session {}: {}", entry.key(), e);
            }
        }

        self.sessions.clear();
        self.sync_manager.write().await.shutdown().await?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_create_session() {
        let server = CollaborationServer::new(CollabConfig::default());
        let owner = User::new("Alice".to_string(), UserRole::Owner);
        let project_id = Uuid::new_v4();

        let session_id = server
            .create_session(project_id, owner)
            .await
            .expect("collab test operation should succeed");
        assert!(server.get_session(session_id).is_ok());
    }

    #[tokio::test]
    async fn test_join_leave_session() {
        let server = CollaborationServer::new(CollabConfig::default());
        let owner = User::new("Alice".to_string(), UserRole::Owner);
        let project_id = Uuid::new_v4();

        let session_id = server
            .create_session(project_id, owner.clone())
            .await
            .expect("collab test operation should succeed");

        let editor = User::new("Bob".to_string(), UserRole::Editor);
        server
            .join_session(session_id, editor.clone())
            .await
            .expect("collab test operation should succeed");

        let session = server
            .get_session(session_id)
            .expect("collab test operation should succeed");
        assert_eq!(session.user_count().await, 2);

        server
            .leave_session(session_id, editor.id)
            .await
            .expect("collab test operation should succeed");
        assert_eq!(session.user_count().await, 1);
    }

    #[test]
    fn test_user_role_permissions() {
        assert!(UserRole::Owner.can_write());
        assert!(UserRole::Editor.can_write());
        assert!(!UserRole::Viewer.can_write());

        assert!(UserRole::Owner.can_manage_locks());
        assert!(!UserRole::Editor.can_manage_locks());
        assert!(!UserRole::Viewer.can_manage_locks());
    }
}
