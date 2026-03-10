//! Awareness protocol for user state broadcasting
//!
//! This module implements the Yjs awareness protocol for tracking user
//! cursors, selections, colors, and ephemeral state.

use crate::session::{CursorPosition, SelectionRange};
use crate::{CollabError, Result, User};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// User awareness state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserAwareness {
    pub user: User,
    pub cursor: Option<CursorPosition>,
    pub selection: Option<SelectionRange>,
    pub viewport: Option<ViewportState>,
    pub last_update: chrono::DateTime<chrono::Utc>,
}

impl UserAwareness {
    /// Create new user awareness
    pub fn new(user: User) -> Self {
        Self {
            user,
            cursor: None,
            selection: None,
            viewport: None,
            last_update: chrono::Utc::now(),
        }
    }

    /// Update cursor position
    pub fn set_cursor(&mut self, cursor: CursorPosition) {
        self.cursor = Some(cursor);
        self.last_update = chrono::Utc::now();
    }

    /// Update selection
    pub fn set_selection(&mut self, selection: SelectionRange) {
        self.selection = Some(selection);
        self.last_update = chrono::Utc::now();
    }

    /// Update viewport
    pub fn set_viewport(&mut self, viewport: ViewportState) {
        self.viewport = Some(viewport);
        self.last_update = chrono::Utc::now();
    }

    /// Clear cursor
    pub fn clear_cursor(&mut self) {
        self.cursor = None;
        self.last_update = chrono::Utc::now();
    }

    /// Clear selection
    pub fn clear_selection(&mut self) {
        self.selection = None;
        self.last_update = chrono::Utc::now();
    }

    /// Check if state is stale
    pub fn is_stale(&self, timeout_secs: u64) -> bool {
        let now = chrono::Utc::now();
        let elapsed = now.signed_duration_since(self.last_update);
        elapsed.num_seconds() as u64 > timeout_secs
    }
}

/// Viewport state (what the user is currently viewing)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViewportState {
    pub start_time: f64,
    pub end_time: f64,
    pub zoom_level: f64,
    pub visible_tracks: Vec<Uuid>,
}

/// Awareness update type
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AwarenessUpdate {
    /// User joined
    Join { user: User },

    /// User left
    Leave { user_id: Uuid },

    /// Cursor update
    Cursor {
        user_id: Uuid,
        cursor: CursorPosition,
    },

    /// Selection update
    Selection {
        user_id: Uuid,
        selection: SelectionRange,
    },

    /// Viewport update
    Viewport {
        user_id: Uuid,
        viewport: ViewportState,
    },

    /// Full state sync
    FullState {
        states: HashMap<Uuid, UserAwareness>,
    },
}

/// Awareness manager
pub struct AwarenessManager {
    #[allow(dead_code)]
    session_id: Uuid,
    states: Arc<RwLock<HashMap<Uuid, UserAwareness>>>,
    subscribers: Arc<RwLock<Vec<Box<dyn AwarenessSubscriber>>>>,
}

impl AwarenessManager {
    /// Create a new awareness manager
    pub fn new(session_id: Uuid) -> Self {
        Self {
            session_id,
            states: Arc::new(RwLock::new(HashMap::new())),
            subscribers: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Add a user
    pub async fn add_user(&self, user: User) -> Result<()> {
        let awareness = UserAwareness::new(user.clone());
        self.states.write().await.insert(user.id, awareness);

        // Notify subscribers
        self.notify(AwarenessUpdate::Join { user }).await;

        Ok(())
    }

    /// Remove a user
    pub async fn remove_user(&self, user_id: Uuid) -> Result<()> {
        self.states.write().await.remove(&user_id);

        // Notify subscribers
        self.notify(AwarenessUpdate::Leave { user_id }).await;

        Ok(())
    }

    /// Update cursor position
    pub async fn update_cursor(&self, user_id: Uuid, cursor: CursorPosition) -> Result<()> {
        if let Some(state) = self.states.write().await.get_mut(&user_id) {
            state.set_cursor(cursor.clone());

            // Notify subscribers
            self.notify(AwarenessUpdate::Cursor { user_id, cursor })
                .await;
        }

        Ok(())
    }

    /// Update selection
    pub async fn update_selection(&self, user_id: Uuid, selection: SelectionRange) -> Result<()> {
        if let Some(state) = self.states.write().await.get_mut(&user_id) {
            state.set_selection(selection.clone());

            // Notify subscribers
            self.notify(AwarenessUpdate::Selection { user_id, selection })
                .await;
        }

        Ok(())
    }

    /// Update viewport
    pub async fn update_viewport(&self, user_id: Uuid, viewport: ViewportState) -> Result<()> {
        if let Some(state) = self.states.write().await.get_mut(&user_id) {
            state.set_viewport(viewport.clone());

            // Notify subscribers
            self.notify(AwarenessUpdate::Viewport { user_id, viewport })
                .await;
        }

        Ok(())
    }

    /// Clear cursor
    pub async fn clear_cursor(&self, user_id: Uuid) -> Result<()> {
        if let Some(state) = self.states.write().await.get_mut(&user_id) {
            state.clear_cursor();
        }

        Ok(())
    }

    /// Clear selection
    pub async fn clear_selection(&self, user_id: Uuid) -> Result<()> {
        if let Some(state) = self.states.write().await.get_mut(&user_id) {
            state.clear_selection();
        }

        Ok(())
    }

    /// Get user awareness state
    pub async fn get_state(&self, user_id: Uuid) -> Option<UserAwareness> {
        self.states.read().await.get(&user_id).cloned()
    }

    /// Get all awareness states
    pub async fn get_all_states(&self) -> HashMap<Uuid, UserAwareness> {
        self.states.read().await.clone()
    }

    /// Get cursors for all users
    pub async fn get_all_cursors(&self) -> HashMap<Uuid, CursorPosition> {
        self.states
            .read()
            .await
            .iter()
            .filter_map(|(id, state)| state.cursor.as_ref().map(|cursor| (*id, cursor.clone())))
            .collect()
    }

    /// Get selections for all users
    pub async fn get_all_selections(&self) -> HashMap<Uuid, SelectionRange> {
        self.states
            .read()
            .await
            .iter()
            .filter_map(|(id, state)| state.selection.as_ref().map(|sel| (*id, sel.clone())))
            .collect()
    }

    /// Subscribe to awareness updates
    pub async fn subscribe(&self, subscriber: Box<dyn AwarenessSubscriber>) {
        self.subscribers.write().await.push(subscriber);
    }

    /// Notify all subscribers
    async fn notify(&self, update: AwarenessUpdate) {
        let subscribers = self.subscribers.read().await;
        for subscriber in subscribers.iter() {
            subscriber.on_awareness_update(&update).await;
        }
    }

    /// Encode awareness state for synchronization
    pub async fn encode_state(&self) -> Result<Vec<u8>> {
        let states = self.get_all_states().await;
        let encoded = serde_json::to_vec(&states)
            .map_err(|e| CollabError::SyncError(format!("Failed to encode awareness: {}", e)))?;
        Ok(encoded)
    }

    /// Apply awareness update from network
    pub async fn apply_update(&self, data: &[u8]) -> Result<()> {
        let update: AwarenessUpdate = serde_json::from_slice(data)
            .map_err(|e| CollabError::SyncError(format!("Failed to decode awareness: {}", e)))?;

        match update {
            AwarenessUpdate::Join { user } => {
                self.add_user(user).await?;
            }
            AwarenessUpdate::Leave { user_id } => {
                self.remove_user(user_id).await?;
            }
            AwarenessUpdate::Cursor { user_id, cursor } => {
                self.update_cursor(user_id, cursor).await?;
            }
            AwarenessUpdate::Selection { user_id, selection } => {
                self.update_selection(user_id, selection).await?;
            }
            AwarenessUpdate::Viewport { user_id, viewport } => {
                self.update_viewport(user_id, viewport).await?;
            }
            AwarenessUpdate::FullState { states } => {
                *self.states.write().await = states;
            }
        }

        Ok(())
    }

    /// Clean up stale states
    pub async fn cleanup_stale(&self, timeout_secs: u64) -> Result<()> {
        let stale_users: Vec<Uuid> = self
            .states
            .read()
            .await
            .iter()
            .filter(|(_, state)| state.is_stale(timeout_secs))
            .map(|(id, _)| *id)
            .collect();

        for user_id in stale_users {
            self.remove_user(user_id).await?;
        }

        Ok(())
    }

    /// Get user count
    pub async fn user_count(&self) -> usize {
        self.states.read().await.len()
    }

    /// Clear all states
    pub async fn clear(&self) {
        self.states.write().await.clear();
    }
}

/// Trait for awareness subscribers
#[async_trait::async_trait]
pub trait AwarenessSubscriber: Send + Sync {
    /// Called when awareness is updated
    async fn on_awareness_update(&self, update: &AwarenessUpdate);
}

/// Helper for rendering cursors
pub struct CursorRenderer {
    colors: Arc<RwLock<HashMap<Uuid, String>>>,
}

impl CursorRenderer {
    /// Create a new cursor renderer
    pub fn new() -> Self {
        Self {
            colors: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Set user color
    pub async fn set_color(&self, user_id: Uuid, color: String) {
        self.colors.write().await.insert(user_id, color);
    }

    /// Get user color
    pub async fn get_color(&self, user_id: Uuid) -> Option<String> {
        self.colors.read().await.get(&user_id).cloned()
    }

    /// Render cursor as SVG
    pub async fn render_cursor(
        &self,
        user_id: Uuid,
        cursor: &CursorPosition,
        username: &str,
    ) -> String {
        let color = self
            .get_color(user_id)
            .await
            .unwrap_or_else(|| "#000000".to_string());

        format!(
            r#"<g class="cursor" data-user="{user_id}">
                <line x1="{x}" y1="0" x2="{x}" y2="100%" stroke="{color}" stroke-width="2"/>
                <text x="{x}" y="20" fill="{color}" font-size="12">{username}</text>
            </g>"#,
            user_id = user_id,
            x = cursor.timestamp * 100.0, // Scale to pixels
            color = color,
            username = username
        )
    }

    /// Render selection as SVG
    pub async fn render_selection(&self, user_id: Uuid, selection: &SelectionRange) -> String {
        let color = self
            .get_color(user_id)
            .await
            .unwrap_or_else(|| "#000000".to_string());

        let x1 = selection.start_time * 100.0;
        let x2 = selection.end_time * 100.0;

        format!(
            r#"<rect x="{x1}" y="0" width="{width}" height="100%" fill="{color}" opacity="0.2"/>"#,
            x1 = x1,
            width = x2 - x1,
            color = color
        )
    }
}

impl Default for CursorRenderer {
    fn default() -> Self {
        Self::new()
    }
}

/// Awareness state exporter
pub struct AwarenessExporter;

impl AwarenessExporter {
    /// Export awareness to JSON
    pub fn to_json(states: &HashMap<Uuid, UserAwareness>) -> Result<String> {
        serde_json::to_string_pretty(states).map_err(CollabError::SerializationError)
    }

    /// Import awareness from JSON
    pub fn from_json(json: &str) -> Result<HashMap<Uuid, UserAwareness>> {
        serde_json::from_str(json).map_err(CollabError::SerializationError)
    }

    /// Export to binary format
    pub fn to_binary(states: &HashMap<Uuid, UserAwareness>) -> Result<Vec<u8>> {
        let compat = oxicode::serde::Compat(states);
        oxicode::encode_to_vec(&compat)
            .map_err(|e| CollabError::SyncError(format!("Binary encoding failed: {}", e)))
    }

    /// Import from binary format
    pub fn from_binary(data: &[u8]) -> Result<HashMap<Uuid, UserAwareness>> {
        let (compat, _): (oxicode::serde::Compat<HashMap<Uuid, UserAwareness>>, _) =
            oxicode::decode_from_slice(data)
                .map_err(|e| CollabError::SyncError(format!("Binary decoding failed: {}", e)))?;
        Ok(compat.0)
    }
}

/// Presence tracker for detecting user activity
pub struct PresenceTracker {
    last_activity: Arc<RwLock<HashMap<Uuid, chrono::DateTime<chrono::Utc>>>>,
    timeout_secs: u64,
}

impl PresenceTracker {
    /// Create a new presence tracker
    pub fn new(timeout_secs: u64) -> Self {
        Self {
            last_activity: Arc::new(RwLock::new(HashMap::new())),
            timeout_secs,
        }
    }

    /// Record user activity
    pub async fn record_activity(&self, user_id: Uuid) {
        self.last_activity
            .write()
            .await
            .insert(user_id, chrono::Utc::now());
    }

    /// Check if user is active
    pub async fn is_active(&self, user_id: Uuid) -> bool {
        if let Some(last) = self.last_activity.read().await.get(&user_id) {
            let now = chrono::Utc::now();
            let elapsed = now.signed_duration_since(*last);
            elapsed.num_seconds() as u64 <= self.timeout_secs
        } else {
            false
        }
    }

    /// Get all active users
    pub async fn get_active_users(&self) -> Vec<Uuid> {
        let activities = self.last_activity.read().await;
        let now = chrono::Utc::now();

        activities
            .iter()
            .filter(|(_, last)| {
                let elapsed = now.signed_duration_since(**last);
                elapsed.num_seconds() as u64 <= self.timeout_secs
            })
            .map(|(id, _)| *id)
            .collect()
    }

    /// Clean up inactive users
    pub async fn cleanup_inactive(&self) {
        let now = chrono::Utc::now();
        let mut activities = self.last_activity.write().await;

        activities.retain(|_, last| {
            let elapsed = now.signed_duration_since(*last);
            elapsed.num_seconds() as u64 <= self.timeout_secs
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::UserRole;

    #[tokio::test]
    async fn test_awareness_manager() {
        let manager = AwarenessManager::new(Uuid::new_v4());
        let user = User::new("Alice".to_string(), UserRole::Owner);

        manager
            .add_user(user.clone())
            .await
            .expect("collab test operation should succeed");
        assert_eq!(manager.user_count().await, 1);

        let state = manager
            .get_state(user.id)
            .await
            .expect("collab test operation should succeed");
        assert_eq!(state.user.name, "Alice");

        manager
            .remove_user(user.id)
            .await
            .expect("collab test operation should succeed");
        assert_eq!(manager.user_count().await, 0);
    }

    #[tokio::test]
    async fn test_cursor_update() {
        let manager = AwarenessManager::new(Uuid::new_v4());
        let user = User::new("Alice".to_string(), UserRole::Owner);

        manager
            .add_user(user.clone())
            .await
            .expect("collab test operation should succeed");

        let cursor = CursorPosition {
            user_id: user.id,
            timestamp: 10.0,
            track_id: None,
            clip_id: None,
        };

        manager
            .update_cursor(user.id, cursor.clone())
            .await
            .expect("collab test operation should succeed");

        let state = manager
            .get_state(user.id)
            .await
            .expect("collab test operation should succeed");
        assert!(state.cursor.is_some());
        assert_eq!(
            state
                .cursor
                .expect("collab test operation should succeed")
                .timestamp,
            10.0
        );
    }

    #[tokio::test]
    async fn test_selection_update() {
        let manager = AwarenessManager::new(Uuid::new_v4());
        let user = User::new("Alice".to_string(), UserRole::Owner);

        manager
            .add_user(user.clone())
            .await
            .expect("collab test operation should succeed");

        let selection = SelectionRange {
            user_id: user.id,
            start_time: 5.0,
            end_time: 15.0,
            track_ids: vec![],
            clip_ids: vec![],
        };

        manager
            .update_selection(user.id, selection.clone())
            .await
            .expect("collab test operation should succeed");

        let state = manager
            .get_state(user.id)
            .await
            .expect("collab test operation should succeed");
        assert!(state.selection.is_some());
    }

    #[tokio::test]
    async fn test_cursor_renderer() {
        let renderer = CursorRenderer::new();
        let user_id = Uuid::new_v4();

        renderer.set_color(user_id, "#FF0000".to_string()).await;

        let cursor = CursorPosition {
            user_id,
            timestamp: 10.0,
            track_id: None,
            clip_id: None,
        };

        let svg = renderer.render_cursor(user_id, &cursor, "Alice").await;
        assert!(svg.contains("FF0000"));
        assert!(svg.contains("Alice"));
    }

    #[tokio::test]
    async fn test_presence_tracker() {
        let tracker = PresenceTracker::new(60);
        let user_id = Uuid::new_v4();

        tracker.record_activity(user_id).await;
        assert!(tracker.is_active(user_id).await);

        let active_users = tracker.get_active_users().await;
        assert_eq!(active_users.len(), 1);
        assert_eq!(active_users[0], user_id);
    }

    #[tokio::test]
    async fn test_awareness_export() {
        let mut states = HashMap::new();
        let user = User::new("Alice".to_string(), UserRole::Owner);
        let awareness = UserAwareness::new(user.clone());
        states.insert(user.id, awareness);

        let json =
            AwarenessExporter::to_json(&states).expect("collab test operation should succeed");
        let imported =
            AwarenessExporter::from_json(&json).expect("collab test operation should succeed");
        assert_eq!(states.len(), imported.len());

        let binary =
            AwarenessExporter::to_binary(&states).expect("collab test operation should succeed");
        let imported =
            AwarenessExporter::from_binary(&binary).expect("collab test operation should succeed");
        assert_eq!(states.len(), imported.len());
    }
}
