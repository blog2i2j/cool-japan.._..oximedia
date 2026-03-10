//! Session management for collaborative editing
//!
//! This module handles user sessions, presence tracking, cursor synchronization,
//! and permission enforcement.

use crate::awareness::AwarenessManager;
use crate::crdt::{CrdtDocument, TimelineOp};
use crate::history::HistoryManager;
use crate::lock::LockManager;
use crate::{CollabConfig, CollabError, Result, User, UserRole};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Cursor position in timeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CursorPosition {
    pub user_id: Uuid,
    pub timestamp: f64,
    pub track_id: Option<Uuid>,
    pub clip_id: Option<Uuid>,
}

/// Selection range in timeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionRange {
    pub user_id: Uuid,
    pub start_time: f64,
    pub end_time: f64,
    pub track_ids: Vec<Uuid>,
    pub clip_ids: Vec<Uuid>,
}

/// User presence state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserPresence {
    pub user: User,
    pub cursor: Option<CursorPosition>,
    pub selection: Option<SelectionRange>,
    pub is_active: bool,
    pub last_activity: chrono::DateTime<chrono::Utc>,
}

impl UserPresence {
    /// Create new user presence
    pub fn new(user: User) -> Self {
        Self {
            user,
            cursor: None,
            selection: None,
            is_active: true,
            last_activity: chrono::Utc::now(),
        }
    }

    /// Update cursor position
    pub fn update_cursor(&mut self, cursor: CursorPosition) {
        self.cursor = Some(cursor);
        self.last_activity = chrono::Utc::now();
    }

    /// Update selection
    pub fn update_selection(&mut self, selection: SelectionRange) {
        self.selection = Some(selection);
        self.last_activity = chrono::Utc::now();
    }

    /// Mark as active
    pub fn mark_active(&mut self) {
        self.is_active = true;
        self.last_activity = chrono::Utc::now();
    }

    /// Mark as inactive
    pub fn mark_inactive(&mut self) {
        self.is_active = false;
    }

    /// Check if user has been inactive for too long
    pub fn is_stale(&self, timeout_secs: u64) -> bool {
        let now = chrono::Utc::now();
        let elapsed = now.signed_duration_since(self.last_activity);
        elapsed.num_seconds() as u64 > timeout_secs
    }
}

/// Session permissions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionPermissions {
    pub can_edit: bool,
    pub can_lock: bool,
    pub can_invite: bool,
    pub can_kick: bool,
}

impl SessionPermissions {
    /// Get permissions for a role
    pub fn for_role(role: UserRole) -> Self {
        match role {
            UserRole::Owner => Self {
                can_edit: true,
                can_lock: true,
                can_invite: true,
                can_kick: true,
            },
            UserRole::Editor => Self {
                can_edit: true,
                can_lock: false,
                can_invite: false,
                can_kick: false,
            },
            UserRole::Viewer => Self {
                can_edit: false,
                can_lock: false,
                can_invite: false,
                can_kick: false,
            },
        }
    }
}

/// Collaboration session
pub struct Session {
    pub id: Uuid,
    pub project_id: Uuid,
    config: CollabConfig,
    users: Arc<DashMap<Uuid, UserPresence>>,
    document: Arc<CrdtDocument>,
    awareness: Arc<RwLock<AwarenessManager>>,
    history: Arc<RwLock<HistoryManager>>,
    locks: Arc<LockManager>,
    created_at: chrono::DateTime<chrono::Utc>,
    metadata: Arc<RwLock<HashMap<String, serde_json::Value>>>,
}

impl Session {
    /// Create a new session
    pub fn new(id: Uuid, project_id: Uuid, owner: User, config: CollabConfig) -> Self {
        let users = Arc::new(DashMap::new());
        let presence = UserPresence::new(owner.clone());
        users.insert(owner.id, presence);

        let document = Arc::new(CrdtDocument::new(id));
        let awareness = Arc::new(RwLock::new(AwarenessManager::new(id)));
        let history = Arc::new(RwLock::new(HistoryManager::new(config.history_limit)));
        let locks = Arc::new(LockManager::new(config.lock_timeout_secs));

        Self {
            id,
            project_id,
            config,
            users,
            document,
            awareness,
            history,
            locks,
            created_at: chrono::Utc::now(),
            metadata: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Add a user to the session
    pub async fn add_user(&self, user: User) -> Result<()> {
        // Check max users
        if self.users.len() >= self.config.max_users_per_session {
            return Err(CollabError::InvalidOperation("Session is full".to_string()));
        }

        let presence = UserPresence::new(user.clone());
        self.users.insert(user.id, presence);

        // Add to awareness
        self.awareness.write().await.add_user(user.clone()).await?;

        tracing::info!("User {} joined session {}", user.name, self.id);
        Ok(())
    }

    /// Remove a user from the session
    pub async fn remove_user(&self, user_id: Uuid) -> Result<()> {
        self.users.remove(&user_id);

        // Remove from awareness
        self.awareness.write().await.remove_user(user_id).await?;

        // Release all locks held by user
        self.locks.release_user_locks(user_id).await?;

        tracing::info!("User {} left session {}", user_id, self.id);
        Ok(())
    }

    /// Get user presence
    pub fn get_user_presence(&self, user_id: Uuid) -> Option<UserPresence> {
        self.users.get(&user_id).map(|entry| entry.value().clone())
    }

    /// Get all active users
    pub fn get_active_users(&self) -> Vec<UserPresence> {
        self.users
            .iter()
            .filter(|entry| entry.value().is_active)
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Get all users
    pub fn get_all_users(&self) -> Vec<UserPresence> {
        self.users
            .iter()
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Update user cursor
    pub async fn update_cursor(&self, user_id: Uuid, cursor: CursorPosition) -> Result<()> {
        if let Some(mut presence) = self.users.get_mut(&user_id) {
            presence.update_cursor(cursor.clone());
        }

        // Update awareness
        self.awareness
            .write()
            .await
            .update_cursor(user_id, cursor)
            .await?;

        Ok(())
    }

    /// Update user selection
    pub async fn update_selection(&self, user_id: Uuid, selection: SelectionRange) -> Result<()> {
        if let Some(mut presence) = self.users.get_mut(&user_id) {
            presence.update_selection(selection.clone());
        }

        // Update awareness
        self.awareness
            .write()
            .await
            .update_selection(user_id, selection)
            .await?;

        Ok(())
    }

    /// Mark user as active
    pub fn mark_user_active(&self, user_id: Uuid) {
        if let Some(mut presence) = self.users.get_mut(&user_id) {
            presence.mark_active();
        }
    }

    /// Mark user as inactive
    pub fn mark_user_inactive(&self, user_id: Uuid) {
        if let Some(mut presence) = self.users.get_mut(&user_id) {
            presence.mark_inactive();
        }
    }

    /// Clean up stale users
    pub async fn cleanup_stale_users(&self) -> Result<()> {
        let timeout = self.config.lock_timeout_secs * 2; // Double lock timeout for presence
        let stale_users: Vec<Uuid> = self
            .users
            .iter()
            .filter(|entry| entry.value().is_stale(timeout))
            .map(|entry| *entry.key())
            .collect();

        for user_id in stale_users {
            tracing::info!("Removing stale user {} from session {}", user_id, self.id);
            self.remove_user(user_id).await?;
        }

        Ok(())
    }

    /// Apply an operation
    pub async fn apply_operation(&self, user_id: Uuid, op: TimelineOp) -> Result<()> {
        // Check permissions
        let user = self
            .users
            .get(&user_id)
            .ok_or(CollabError::UserNotFound(user_id))?;

        let permissions = SessionPermissions::for_role(user.user.role);
        if !permissions.can_edit {
            return Err(CollabError::PermissionDenied(
                "User cannot edit".to_string(),
            ));
        }

        // Apply to document
        self.document.apply_operation(op.clone()).await?;

        // Add to history
        self.history.write().await.add_operation(op.clone()).await?;

        // Mark user as active
        self.mark_user_active(user_id);

        Ok(())
    }

    /// Undo last operation
    pub async fn undo(&self, user_id: Uuid) -> Result<Option<TimelineOp>> {
        // Check permissions
        let user = self
            .users
            .get(&user_id)
            .ok_or(CollabError::UserNotFound(user_id))?;

        let permissions = SessionPermissions::for_role(user.user.role);
        if !permissions.can_edit {
            return Err(CollabError::PermissionDenied(
                "User cannot edit".to_string(),
            ));
        }

        // Undo from history
        let undo_op = self.history.write().await.undo(user_id).await?;

        // Mark user as active
        self.mark_user_active(user_id);

        Ok(undo_op)
    }

    /// Redo last undone operation
    pub async fn redo(&self, user_id: Uuid) -> Result<Option<TimelineOp>> {
        // Check permissions
        let user = self
            .users
            .get(&user_id)
            .ok_or(CollabError::UserNotFound(user_id))?;

        let permissions = SessionPermissions::for_role(user.user.role);
        if !permissions.can_edit {
            return Err(CollabError::PermissionDenied(
                "User cannot edit".to_string(),
            ));
        }

        // Redo from history
        let redo_op = self.history.write().await.redo(user_id).await?;

        // Mark user as active
        self.mark_user_active(user_id);

        Ok(redo_op)
    }

    /// Get user count
    pub async fn user_count(&self) -> usize {
        self.users.len()
    }

    /// Get document
    pub fn document(&self) -> Arc<CrdtDocument> {
        self.document.clone()
    }

    /// Get awareness manager
    pub fn awareness(&self) -> Arc<RwLock<AwarenessManager>> {
        self.awareness.clone()
    }

    /// Get history manager
    pub fn history(&self) -> Arc<RwLock<HistoryManager>> {
        self.history.clone()
    }

    /// Get lock manager
    pub fn locks(&self) -> Arc<LockManager> {
        self.locks.clone()
    }

    /// Check if user can perform action
    pub fn check_permission(&self, user_id: Uuid, action: &str) -> Result<()> {
        let user = self
            .users
            .get(&user_id)
            .ok_or(CollabError::UserNotFound(user_id))?;

        let permissions = SessionPermissions::for_role(user.user.role);

        match action {
            "edit" if !permissions.can_edit => {
                Err(CollabError::PermissionDenied("Cannot edit".to_string()))
            }
            "lock" if !permissions.can_lock => {
                Err(CollabError::PermissionDenied("Cannot lock".to_string()))
            }
            "invite" if !permissions.can_invite => {
                Err(CollabError::PermissionDenied("Cannot invite".to_string()))
            }
            "kick" if !permissions.can_kick => {
                Err(CollabError::PermissionDenied("Cannot kick".to_string()))
            }
            _ => Ok(()),
        }
    }

    /// Kick a user from the session
    pub async fn kick_user(&self, kicker_id: Uuid, user_id: Uuid) -> Result<()> {
        self.check_permission(kicker_id, "kick")?;

        // Cannot kick owner
        // Note: we must drop the DashMap read guard before calling remove_user,
        // which acquires a write lock on the same shard, to avoid deadlock.
        let is_owner = {
            let user = self
                .users
                .get(&user_id)
                .ok_or(CollabError::UserNotFound(user_id))?;
            user.user.role == UserRole::Owner
        };

        if is_owner {
            return Err(CollabError::PermissionDenied(
                "Cannot kick owner".to_string(),
            ));
        }

        self.remove_user(user_id).await
    }

    /// Change user role
    pub async fn change_user_role(
        &self,
        changer_id: Uuid,
        user_id: Uuid,
        new_role: UserRole,
    ) -> Result<()> {
        // Only owner can change roles
        let changer = self
            .users
            .get(&changer_id)
            .ok_or(CollabError::UserNotFound(changer_id))?;

        if changer.user.role != UserRole::Owner {
            return Err(CollabError::PermissionDenied(
                "Only owner can change roles".to_string(),
            ));
        }

        // Update user role
        if let Some(mut user) = self.users.get_mut(&user_id) {
            user.user.role = new_role;
        }

        Ok(())
    }

    /// Set session metadata
    pub async fn set_metadata(&self, key: String, value: serde_json::Value) -> Result<()> {
        self.metadata.write().await.insert(key, value);
        Ok(())
    }

    /// Get session metadata
    pub async fn get_metadata(&self, key: &str) -> Option<serde_json::Value> {
        self.metadata.read().await.get(key).cloned()
    }

    /// Run garbage collection
    pub async fn run_garbage_collection(&self) -> Result<()> {
        // Clean up stale users
        self.cleanup_stale_users().await?;

        // Clean up document
        self.document
            .garbage_collect(self.config.history_limit)
            .await?;

        // Clean up history
        self.history
            .write()
            .await
            .compact(self.config.history_limit)
            .await?;

        // Clean up locks
        self.locks.cleanup_expired_locks().await?;

        Ok(())
    }

    /// Close the session
    pub async fn close(&self) -> Result<()> {
        // Remove all users
        let user_ids: Vec<Uuid> = self.users.iter().map(|entry| *entry.key()).collect();
        for user_id in user_ids {
            self.remove_user(user_id).await?;
        }

        // Clear all data
        self.document.clear().await?;

        Ok(())
    }

    /// Get session info
    pub async fn info(&self) -> SessionInfo {
        SessionInfo {
            id: self.id,
            project_id: self.project_id,
            user_count: self.users.len(),
            active_user_count: self.get_active_users().len(),
            created_at: self.created_at,
            operation_count: self.document.operation_count().await,
        }
    }
}

/// Session information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionInfo {
    pub id: Uuid,
    pub project_id: Uuid,
    pub user_count: usize,
    pub active_user_count: usize,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub operation_count: usize,
}

/// Session manager for handling multiple sessions
pub struct SessionManager {
    sessions: Arc<DashMap<Uuid, Arc<Session>>>,
    config: CollabConfig,
}

impl SessionManager {
    /// Create a new session manager
    pub fn new(config: CollabConfig) -> Self {
        Self {
            sessions: Arc::new(DashMap::new()),
            config,
        }
    }

    /// Create a new session
    pub async fn create_session(&self, project_id: Uuid, owner: User) -> Result<Arc<Session>> {
        let session_id = Uuid::new_v4();
        let session = Arc::new(Session::new(
            session_id,
            project_id,
            owner,
            self.config.clone(),
        ));

        self.sessions.insert(session_id, session.clone());
        Ok(session)
    }

    /// Get a session
    pub fn get_session(&self, session_id: Uuid) -> Option<Arc<Session>> {
        self.sessions
            .get(&session_id)
            .map(|entry| entry.value().clone())
    }

    /// Remove a session
    pub fn remove_session(&self, session_id: Uuid) {
        self.sessions.remove(&session_id);
    }

    /// List all sessions
    pub fn list_sessions(&self) -> Vec<SessionInfo> {
        let sessions = self.sessions.clone();
        let mut infos = Vec::new();

        for entry in sessions.iter() {
            let session = entry.value();
            let rt = tokio::runtime::Handle::current();
            let info = rt.block_on(session.info());
            infos.push(info);
        }

        infos
    }

    /// Run garbage collection on all sessions
    pub async fn garbage_collect_all(&self) -> Result<()> {
        for entry in self.sessions.iter() {
            entry.value().run_garbage_collection().await?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_session_creation() {
        let owner = User::new("Alice".to_string(), UserRole::Owner);
        let session = Session::new(
            Uuid::new_v4(),
            Uuid::new_v4(),
            owner,
            CollabConfig::default(),
        );

        assert_eq!(session.user_count().await, 1);
    }

    #[tokio::test]
    async fn test_add_remove_user() {
        let owner = User::new("Alice".to_string(), UserRole::Owner);
        let session = Session::new(
            Uuid::new_v4(),
            Uuid::new_v4(),
            owner,
            CollabConfig::default(),
        );

        let editor = User::new("Bob".to_string(), UserRole::Editor);
        session
            .add_user(editor.clone())
            .await
            .expect("collab test operation should succeed");
        assert_eq!(session.user_count().await, 2);

        session
            .remove_user(editor.id)
            .await
            .expect("collab test operation should succeed");
        assert_eq!(session.user_count().await, 1);
    }

    #[tokio::test]
    async fn test_permissions() {
        let owner = User::new("Alice".to_string(), UserRole::Owner);
        let session = Session::new(
            Uuid::new_v4(),
            Uuid::new_v4(),
            owner.clone(),
            CollabConfig::default(),
        );

        let viewer = User::new("Bob".to_string(), UserRole::Viewer);
        session
            .add_user(viewer.clone())
            .await
            .expect("collab test operation should succeed");

        // Owner can edit
        assert!(session.check_permission(owner.id, "edit").is_ok());

        // Viewer cannot edit
        assert!(session.check_permission(viewer.id, "edit").is_err());
    }

    #[tokio::test]
    async fn test_kick_user() {
        let owner = User::new("Alice".to_string(), UserRole::Owner);
        let session = Session::new(
            Uuid::new_v4(),
            Uuid::new_v4(),
            owner.clone(),
            CollabConfig::default(),
        );

        let editor = User::new("Bob".to_string(), UserRole::Editor);
        session
            .add_user(editor.clone())
            .await
            .expect("collab test operation should succeed");

        // Owner can kick editor
        session
            .kick_user(owner.id, editor.id)
            .await
            .expect("collab test operation should succeed");
        assert_eq!(session.user_count().await, 1);
    }

    #[tokio::test]
    async fn test_session_manager() {
        let manager = SessionManager::new(CollabConfig::default());
        let owner = User::new("Alice".to_string(), UserRole::Owner);

        let session = manager
            .create_session(Uuid::new_v4(), owner)
            .await
            .expect("collab test operation should succeed");
        let session_id = session.id;

        assert!(manager.get_session(session_id).is_some());

        manager.remove_session(session_id);
        assert!(manager.get_session(session_id).is_none());
    }
}
