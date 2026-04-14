//! Real-time synchronization protocol
//!
//! This module handles WebSocket-based communication, delta encoding,
//! compression, and offline support.

use crate::session::Session;
use crate::{CollabConfig, CollabError, Result};
use oxiarc_deflate::{gzip_compress, gzip_decompress};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;
use yrs::StateVector;

/// Sync message types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SyncMessage {
    /// Request sync state
    SyncStep1 { state_vector: Vec<u8> },

    /// Send missing updates
    SyncStep2 { update: Vec<u8> },

    /// Send incremental update
    Update { update: Vec<u8> },

    /// Awareness update (cursor, selection, etc.)
    Awareness { state: Vec<u8> },

    /// Query sync state
    QueryState,

    /// Ping for keepalive
    Ping,

    /// Pong response
    Pong,

    /// Error message
    Error { message: String },
}

/// Compressed sync message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedMessage {
    pub compressed: bool,
    pub data: Vec<u8>,
}

impl CompressedMessage {
    /// Create a new message, optionally compressing it
    pub fn new(data: Vec<u8>, compress: bool, threshold: usize) -> Result<Self> {
        if compress && data.len() > threshold {
            let compressed_data = compress_data(&data)?;
            Ok(Self {
                compressed: true,
                data: compressed_data,
            })
        } else {
            Ok(Self {
                compressed: false,
                data,
            })
        }
    }

    /// Decompress and deserialize the message
    pub fn deserialize(&self) -> Result<SyncMessage> {
        let data = if self.compressed {
            decompress_data(&self.data)?
        } else {
            self.data.clone()
        };

        serde_json::from_slice(&data)
            .map_err(|e| CollabError::SyncError(format!("Deserialization failed: {}", e)))
    }

    /// Serialize and optionally compress a message
    pub fn from_message(msg: &SyncMessage, compress: bool, threshold: usize) -> Result<Self> {
        let data = serde_json::to_vec(msg)
            .map_err(|e| CollabError::SyncError(format!("Serialization failed: {}", e)))?;

        Self::new(data, compress, threshold)
    }
}

/// Compress data using gzip
fn compress_data(data: &[u8]) -> Result<Vec<u8>> {
    gzip_compress(data, 6).map_err(|e| CollabError::SyncError(format!("Compression failed: {}", e)))
}

/// Decompress data using gzip
fn decompress_data(data: &[u8]) -> Result<Vec<u8>> {
    gzip_decompress(data)
        .map_err(|e| CollabError::SyncError(format!("Decompression failed: {}", e)))
}

/// Sync connection state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionState {
    Connected,
    Disconnected,
    Syncing,
    Error,
}

/// Client connection for synchronization
pub struct SyncConnection {
    pub id: Uuid,
    pub user_id: Uuid,
    pub session_id: Uuid,
    state: Arc<RwLock<ConnectionState>>,
    pending_messages: Arc<RwLock<VecDeque<CompressedMessage>>>,
    last_activity: Arc<RwLock<chrono::DateTime<chrono::Utc>>>,
    config: CollabConfig,
}

impl SyncConnection {
    /// Create a new sync connection
    pub fn new(user_id: Uuid, session_id: Uuid, config: CollabConfig) -> Self {
        Self {
            id: Uuid::new_v4(),
            user_id,
            session_id,
            state: Arc::new(RwLock::new(ConnectionState::Connected)),
            pending_messages: Arc::new(RwLock::new(VecDeque::new())),
            last_activity: Arc::new(RwLock::new(chrono::Utc::now())),
            config,
        }
    }

    /// Get connection state
    pub async fn state(&self) -> ConnectionState {
        *self.state.read().await
    }

    /// Set connection state
    pub async fn set_state(&self, state: ConnectionState) {
        *self.state.write().await = state;
    }

    /// Send a message
    pub async fn send_message(&self, message: SyncMessage) -> Result<CompressedMessage> {
        let compressed = CompressedMessage::from_message(
            &message,
            self.config.enable_compression,
            self.config.compression_threshold,
        )?;

        // If offline, queue the message
        if *self.state.read().await == ConnectionState::Disconnected {
            if self.config.enable_offline {
                let mut queue = self.pending_messages.write().await;
                if queue.len() < self.config.max_offline_queue {
                    queue.push_back(compressed.clone());
                } else {
                    return Err(CollabError::SyncError("Offline queue full".to_string()));
                }
            } else {
                return Err(CollabError::SyncError("Connection offline".to_string()));
            }
        }

        self.update_activity().await;
        Ok(compressed)
    }

    /// Receive and process a message
    pub async fn receive_message(&self, compressed: CompressedMessage) -> Result<SyncMessage> {
        self.update_activity().await;
        compressed.deserialize()
    }

    /// Flush pending messages
    pub async fn flush_pending(&self) -> Result<Vec<CompressedMessage>> {
        let mut queue = self.pending_messages.write().await;
        let messages: Vec<_> = queue.drain(..).collect();
        Ok(messages)
    }

    /// Update last activity timestamp
    async fn update_activity(&self) {
        *self.last_activity.write().await = chrono::Utc::now();
    }

    /// Check if connection is stale
    pub async fn is_stale(&self, timeout_secs: u64) -> bool {
        let now = chrono::Utc::now();
        let last = *self.last_activity.read().await;
        let elapsed = now.signed_duration_since(last);
        elapsed.num_seconds() as u64 > timeout_secs
    }

    /// Disconnect
    pub async fn disconnect(&self) {
        self.set_state(ConnectionState::Disconnected).await;
    }

    /// Reconnect
    pub async fn reconnect(&self) -> Result<Vec<CompressedMessage>> {
        self.set_state(ConnectionState::Connected).await;
        self.flush_pending().await
    }
}

/// Synchronization manager
pub struct SyncManager {
    connections: Arc<RwLock<HashMap<Uuid, Arc<SyncConnection>>>>,
    sessions: Arc<RwLock<HashMap<Uuid, Arc<Session>>>>,
    config: CollabConfig,
}

impl SyncManager {
    /// Create a new sync manager
    pub fn new(config: CollabConfig) -> Self {
        Self {
            connections: Arc::new(RwLock::new(HashMap::new())),
            sessions: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }

    /// Register a session
    pub async fn register_session(&self, session_id: Uuid, session: Arc<Session>) -> Result<()> {
        self.sessions.write().await.insert(session_id, session);
        Ok(())
    }

    /// Unregister a session
    pub async fn unregister_session(&self, session_id: Uuid) -> Result<()> {
        // Disconnect all connections for this session
        let connections: Vec<Uuid> = self
            .connections
            .read()
            .await
            .iter()
            .filter(|(_, conn)| {
                let rt = tokio::runtime::Handle::current();
                rt.block_on(async { conn.session_id == session_id })
            })
            .map(|(id, _)| *id)
            .collect();

        for conn_id in connections {
            self.disconnect(conn_id).await?;
        }

        self.sessions.write().await.remove(&session_id);
        Ok(())
    }

    /// Create a new connection
    pub async fn connect(&self, user_id: Uuid, session_id: Uuid) -> Result<Arc<SyncConnection>> {
        // Check if session exists
        let session = self
            .sessions
            .read()
            .await
            .get(&session_id)
            .cloned()
            .ok_or(CollabError::SessionNotFound(session_id))?;

        // Create connection
        let connection = Arc::new(SyncConnection::new(
            user_id,
            session_id,
            self.config.clone(),
        ));

        self.connections
            .write()
            .await
            .insert(connection.id, connection.clone());

        // Send initial sync
        self.send_initial_sync(&connection, &session).await?;

        Ok(connection)
    }

    /// Disconnect a connection
    pub async fn disconnect(&self, connection_id: Uuid) -> Result<()> {
        if let Some(conn) = self.connections.write().await.remove(&connection_id) {
            conn.disconnect().await;
        }
        Ok(())
    }

    /// Handle sync message
    #[allow(clippy::too_many_arguments)]
    pub async fn handle_message(
        &self,
        connection_id: Uuid,
        message: SyncMessage,
    ) -> Result<Option<SyncMessage>> {
        let conn = self
            .connections
            .read()
            .await
            .get(&connection_id)
            .cloned()
            .ok_or(CollabError::SyncError("Connection not found".to_string()))?;

        let session = self
            .sessions
            .read()
            .await
            .get(&conn.session_id)
            .cloned()
            .ok_or(CollabError::SessionNotFound(conn.session_id))?;

        match message {
            SyncMessage::SyncStep1 { state_vector } => {
                self.handle_sync_step1(&conn, &session, &state_vector).await
            }
            SyncMessage::Update { update } => self.handle_update(&conn, &session, &update).await,
            SyncMessage::Awareness { state } => {
                self.handle_awareness(&conn, &session, &state).await
            }
            SyncMessage::QueryState => self.handle_query_state(&conn, &session).await,
            SyncMessage::Ping => Ok(Some(SyncMessage::Pong)),
            _ => Ok(None),
        }
    }

    /// Send initial sync to new connection
    async fn send_initial_sync(&self, conn: &SyncConnection, session: &Session) -> Result<()> {
        let doc = session.document();
        let state_vector = StateVector::default();
        let update = doc.get_update(&state_vector)?;

        let msg = SyncMessage::SyncStep2 { update };
        conn.send_message(msg).await?;

        Ok(())
    }

    /// Handle sync step 1 (client sends state vector)
    async fn handle_sync_step1(
        &self,
        _conn: &SyncConnection,
        session: &Session,
        state_vector: &[u8],
    ) -> Result<Option<SyncMessage>> {
        use yrs::updates::decoder::Decode;
        let doc = session.document();

        // Decode state vector
        let sv = StateVector::decode_v1(state_vector)
            .map_err(|e| CollabError::SyncError(format!("Failed to decode state vector: {}", e)))?;

        // Get missing updates
        let update = doc.get_update(&sv)?;

        Ok(Some(SyncMessage::SyncStep2 { update }))
    }

    /// Handle update message
    async fn handle_update(
        &self,
        _conn: &SyncConnection,
        session: &Session,
        update: &[u8],
    ) -> Result<Option<SyncMessage>> {
        let doc = session.document();
        doc.apply_update(update)?;

        // Broadcast to other connections
        self.broadcast_update(session.id, update).await?;

        Ok(None)
    }

    /// Handle awareness message
    async fn handle_awareness(
        &self,
        _conn: &SyncConnection,
        session: &Session,
        state: &[u8],
    ) -> Result<Option<SyncMessage>> {
        // Update awareness state
        let awareness_mgr = session.awareness();
        awareness_mgr.write().await.apply_update(state).await?;

        // Broadcast to other connections
        self.broadcast_awareness(session.id, state).await?;

        Ok(None)
    }

    /// Handle query state message
    async fn handle_query_state(
        &self,
        _conn: &SyncConnection,
        session: &Session,
    ) -> Result<Option<SyncMessage>> {
        let doc = session.document();
        let state_vector = StateVector::default();
        let update = doc.get_update(&state_vector)?;

        Ok(Some(SyncMessage::SyncStep2 { update }))
    }

    /// Broadcast update to all connections in a session
    async fn broadcast_update(&self, session_id: Uuid, update: &[u8]) -> Result<()> {
        let connections = self.connections.read().await;

        for conn in connections.values() {
            if conn.session_id == session_id {
                let msg = SyncMessage::Update {
                    update: update.to_vec(),
                };
                conn.send_message(msg).await?;
            }
        }

        Ok(())
    }

    /// Broadcast awareness to all connections in a session
    async fn broadcast_awareness(&self, session_id: Uuid, state: &[u8]) -> Result<()> {
        let connections = self.connections.read().await;

        for conn in connections.values() {
            if conn.session_id == session_id {
                let msg = SyncMessage::Awareness {
                    state: state.to_vec(),
                };
                conn.send_message(msg).await?;
            }
        }

        Ok(())
    }

    /// Clean up stale connections
    pub async fn cleanup_stale_connections(&self) -> Result<()> {
        let timeout = self.config.lock_timeout_secs;
        let stale_ids: Vec<Uuid> = {
            let connections = self.connections.read().await;
            let mut ids = Vec::new();

            for (id, conn) in connections.iter() {
                if conn.is_stale(timeout).await {
                    ids.push(*id);
                }
            }

            ids
        };

        for id in stale_ids {
            tracing::info!("Disconnecting stale connection {}", id);
            self.disconnect(id).await?;
        }

        Ok(())
    }

    /// Get connection count for a session
    pub async fn connection_count(&self, session_id: Uuid) -> usize {
        self.connections
            .read()
            .await
            .values()
            .filter(|conn| conn.session_id == session_id)
            .count()
    }

    /// Shutdown sync manager
    pub async fn shutdown(&self) -> Result<()> {
        let connection_ids: Vec<Uuid> = self.connections.read().await.keys().copied().collect();

        for id in connection_ids {
            self.disconnect(id).await?;
        }

        self.sessions.write().await.clear();

        Ok(())
    }
}

/// Delta encoder for efficient synchronization
pub struct DeltaEncoder {
    last_state: Arc<RwLock<Option<Vec<u8>>>>,
}

impl DeltaEncoder {
    /// Create a new delta encoder
    pub fn new() -> Self {
        Self {
            last_state: Arc::new(RwLock::new(None)),
        }
    }

    /// Encode a delta between last state and current state
    pub async fn encode_delta(&self, current: &[u8]) -> Vec<u8> {
        let last = self.last_state.read().await;

        if let Some(last_bytes) = last.as_ref() {
            // Simple delta: just send what's different
            if last_bytes == current {
                return Vec::new(); // No changes
            }
        }

        // Update last state
        drop(last);
        *self.last_state.write().await = Some(current.to_vec());

        current.to_vec()
    }

    /// Reset encoder state
    pub async fn reset(&self) {
        *self.last_state.write().await = None;
    }
}

impl Default for DeltaEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Change queue for offline support
pub struct ChangeQueue {
    queue: Arc<RwLock<VecDeque<SyncMessage>>>,
    max_size: usize,
}

impl ChangeQueue {
    /// Create a new change queue
    pub fn new(max_size: usize) -> Self {
        Self {
            queue: Arc::new(RwLock::new(VecDeque::new())),
            max_size,
        }
    }

    /// Add a change to the queue
    pub async fn push(&self, message: SyncMessage) -> Result<()> {
        let mut queue = self.queue.write().await;

        if queue.len() >= self.max_size {
            // Remove oldest message
            queue.pop_front();
        }

        queue.push_back(message);
        Ok(())
    }

    /// Get all pending changes
    pub async fn drain(&self) -> Vec<SyncMessage> {
        let mut queue = self.queue.write().await;
        queue.drain(..).collect()
    }

    /// Get queue size
    pub async fn len(&self) -> usize {
        self.queue.read().await.len()
    }

    /// Check if queue is empty
    pub async fn is_empty(&self) -> bool {
        self.queue.read().await.is_empty()
    }

    /// Clear the queue
    pub async fn clear(&self) {
        self.queue.write().await.clear();
    }
}

// ---------------------------------------------------------------------------
// Offline edit queue with automatic conflict resolution on reconnect
// ---------------------------------------------------------------------------

/// Conflict resolution strategy applied when an offline edit queue is flushed
/// on reconnect.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OfflineConflictStrategy {
    /// Server state wins — local edits that conflict are dropped.
    ServerWins,
    /// Client (local) edits win — conflicting server state is overwritten.
    ClientWins,
    /// Merge by interleaving: apply local edits in order, skipping any whose
    /// logical clock is older than the server's latest clock for that key.
    MergeByTimestamp,
}

/// Stamped local edit stored while offline.
#[derive(Debug, Clone)]
pub struct OfflineEdit {
    /// Monotonically increasing sequence number (local).
    pub seq: u64,
    /// Logical clock at submission time.
    pub logical_clock: u64,
    /// An opaque key identifying the region/object this edit targets
    /// (e.g. "track:1:0-5000").
    pub target_key: String,
    /// The sync message payload.
    pub message: SyncMessage,
    /// Whether this edit has been acknowledged by the server.
    pub acknowledged: bool,
}

/// Result of flushing the offline queue on reconnect.
#[derive(Debug)]
pub struct ReconnectResult {
    /// Edits that were successfully applied (forwarded to server).
    pub applied: Vec<OfflineEdit>,
    /// Edits that were dropped because of conflicts.
    pub dropped: Vec<OfflineEdit>,
    /// Edits that were merged (modified before sending).
    pub merged: Vec<OfflineEdit>,
}

/// Queue for edits made while the client is disconnected.
///
/// On reconnect, the queue is flushed and each edit is checked against the
/// server's latest state using the configured [`OfflineConflictStrategy`].
#[derive(Debug)]
pub struct OfflineEditQueue {
    /// Queued edits in submission order.
    edits: Vec<OfflineEdit>,
    /// Maximum number of edits to buffer.
    max_size: usize,
    /// Next sequence number.
    next_seq: u64,
    /// Conflict resolution strategy.
    strategy: OfflineConflictStrategy,
    /// Server-side logical clocks per target key (refreshed on reconnect).
    server_clocks: HashMap<String, u64>,
}

impl OfflineEditQueue {
    /// Create a new offline edit queue.
    pub fn new(max_size: usize, strategy: OfflineConflictStrategy) -> Self {
        Self {
            edits: Vec::new(),
            max_size,
            next_seq: 0,
            strategy,
            server_clocks: HashMap::new(),
        }
    }

    /// Enqueue an edit made while offline.
    ///
    /// Returns `Ok(seq)` on success or an error if the queue is full.
    pub fn enqueue(
        &mut self,
        logical_clock: u64,
        target_key: impl Into<String>,
        message: SyncMessage,
    ) -> std::result::Result<u64, String> {
        if self.edits.len() >= self.max_size {
            return Err("Offline edit queue is full".to_string());
        }
        let seq = self.next_seq;
        self.next_seq += 1;
        self.edits.push(OfflineEdit {
            seq,
            logical_clock,
            target_key: target_key.into(),
            message,
            acknowledged: false,
        });
        Ok(seq)
    }

    /// Number of pending edits.
    #[must_use]
    pub fn len(&self) -> usize {
        self.edits.len()
    }

    /// Whether the queue is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.edits.is_empty()
    }

    /// Update the server-side logical clock for a given target key.
    /// Called when the client receives the server's state on reconnect.
    pub fn update_server_clock(&mut self, target_key: impl Into<String>, clock: u64) {
        self.server_clocks.insert(target_key.into(), clock);
    }

    /// Flush the queue, resolving conflicts against the server clocks.
    ///
    /// This consumes all pending edits and returns a [`ReconnectResult`]
    /// describing which edits were applied, dropped, or merged.
    pub fn flush(&mut self) -> ReconnectResult {
        let edits = std::mem::take(&mut self.edits);
        let mut result = ReconnectResult {
            applied: Vec::new(),
            dropped: Vec::new(),
            merged: Vec::new(),
        };

        for edit in edits {
            let server_clock = self
                .server_clocks
                .get(&edit.target_key)
                .copied()
                .unwrap_or(0);
            let has_conflict = server_clock > 0 && edit.logical_clock <= server_clock;

            if !has_conflict {
                // No conflict — apply directly.
                result.applied.push(edit);
            } else {
                match self.strategy {
                    OfflineConflictStrategy::ServerWins => {
                        // Drop the local edit.
                        result.dropped.push(edit);
                    }
                    OfflineConflictStrategy::ClientWins => {
                        // Force-apply the local edit.
                        result.applied.push(edit);
                    }
                    OfflineConflictStrategy::MergeByTimestamp => {
                        // Attempt merge: if the local clock equals the server
                        // clock, treat as a merge (both changed at the same
                        // logical time).  Otherwise drop.
                        if edit.logical_clock == server_clock {
                            let mut merged_edit = edit;
                            merged_edit.acknowledged = false;
                            result.merged.push(merged_edit);
                        } else {
                            result.dropped.push(edit);
                        }
                    }
                }
            }
        }

        result
    }

    /// Clear the queue without flushing.
    pub fn clear(&mut self) {
        self.edits.clear();
    }

    /// Get a reference to all pending edits.
    #[must_use]
    pub fn pending(&self) -> &[OfflineEdit] {
        &self.edits
    }

    /// Get the configured strategy.
    #[must_use]
    pub fn strategy(&self) -> OfflineConflictStrategy {
        self.strategy
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{User, UserRole};

    #[tokio::test]
    async fn test_compression() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let compressed = compress_data(&data).expect("collab test operation should succeed");
        let decompressed =
            decompress_data(&compressed).expect("collab test operation should succeed");
        assert_eq!(data, decompressed);
    }

    #[tokio::test]
    async fn test_compressed_message() {
        let msg = SyncMessage::Ping;
        let compressed = CompressedMessage::from_message(&msg, true, 0)
            .expect("collab test operation should succeed");
        let decompressed = compressed
            .deserialize()
            .expect("collab test operation should succeed");

        match decompressed {
            SyncMessage::Ping => (),
            _ => panic!("Wrong message type"),
        }
    }

    #[tokio::test]
    async fn test_sync_connection() {
        let conn = SyncConnection::new(Uuid::new_v4(), Uuid::new_v4(), CollabConfig::default());

        assert_eq!(conn.state().await, ConnectionState::Connected);

        conn.disconnect().await;
        assert_eq!(conn.state().await, ConnectionState::Disconnected);
    }

    #[tokio::test]
    async fn test_change_queue() {
        let queue = ChangeQueue::new(3);

        queue
            .push(SyncMessage::Ping)
            .await
            .expect("collab test operation should succeed");
        queue
            .push(SyncMessage::Pong)
            .await
            .expect("collab test operation should succeed");
        queue
            .push(SyncMessage::QueryState)
            .await
            .expect("collab test operation should succeed");

        assert_eq!(queue.len().await, 3);

        // Should remove oldest when full
        queue
            .push(SyncMessage::Ping)
            .await
            .expect("collab test operation should succeed");
        assert_eq!(queue.len().await, 3);

        let messages = queue.drain().await;
        assert_eq!(messages.len(), 3);
        assert!(queue.is_empty().await);
    }

    #[tokio::test]
    async fn test_delta_encoder() {
        let encoder = DeltaEncoder::new();

        let data1 = vec![1, 2, 3];
        let delta1 = encoder.encode_delta(&data1).await;
        assert_eq!(delta1, data1);

        // Same data should return empty delta
        let delta2 = encoder.encode_delta(&data1).await;
        assert!(delta2.is_empty());

        // Different data should return full data
        let data2 = vec![4, 5, 6];
        let delta3 = encoder.encode_delta(&data2).await;
        assert_eq!(delta3, data2);
    }

    #[tokio::test]
    async fn test_sync_manager() {
        let config = CollabConfig::default();
        let manager = SyncManager::new(config.clone());

        let owner = User::new("Alice".to_string(), UserRole::Owner);
        let session = Arc::new(crate::session::Session::new(
            Uuid::new_v4(),
            Uuid::new_v4(),
            owner.clone(),
            config,
        ));

        manager
            .register_session(session.id, session.clone())
            .await
            .expect("collab test operation should succeed");

        let conn = manager
            .connect(owner.id, session.id)
            .await
            .expect("collab test operation should succeed");
        assert_eq!(manager.connection_count(session.id).await, 1);

        manager
            .disconnect(conn.id)
            .await
            .expect("collab test operation should succeed");
        assert_eq!(manager.connection_count(session.id).await, 0);
    }
}

/// Reconnection strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReconnectionStrategy {
    /// No automatic reconnection
    None,
    /// Immediate reconnection
    Immediate,
    /// Exponential backoff
    ExponentialBackoff {
        initial_delay_ms: u64,
        max_delay_ms: u64,
    },
    /// Fixed interval
    FixedInterval { interval_ms: u64 },
}

impl Default for ReconnectionStrategy {
    fn default() -> Self {
        Self::ExponentialBackoff {
            initial_delay_ms: 1000,
            max_delay_ms: 30000,
        }
    }
}

/// Connection statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConnectionStats {
    pub messages_sent: u64,
    pub messages_received: u64,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub reconnection_count: u32,
    pub last_ping_ms: Option<u64>,
    pub average_latency_ms: f64,
}

impl ConnectionStats {
    /// Record sent message
    pub fn record_sent(&mut self, size: usize) {
        self.messages_sent += 1;
        self.bytes_sent += size as u64;
    }

    /// Record received message
    pub fn record_received(&mut self, size: usize) {
        self.messages_received += 1;
        self.bytes_received += size as u64;
    }

    /// Update latency measurement
    pub fn update_latency(&mut self, latency_ms: u64) {
        self.last_ping_ms = Some(latency_ms);

        // Exponential moving average
        let alpha = 0.2;
        self.average_latency_ms =
            alpha * (latency_ms as f64) + (1.0 - alpha) * self.average_latency_ms;
    }

    /// Record reconnection
    pub fn record_reconnection(&mut self) {
        self.reconnection_count += 1;
    }
}

/// Reconnection manager
pub struct ReconnectionManager {
    strategy: ReconnectionStrategy,
    attempt_count: Arc<RwLock<u32>>,
    last_attempt: Arc<RwLock<Option<chrono::DateTime<chrono::Utc>>>>,
}

impl ReconnectionManager {
    /// Create a new reconnection manager
    pub fn new(strategy: ReconnectionStrategy) -> Self {
        Self {
            strategy,
            attempt_count: Arc::new(RwLock::new(0)),
            last_attempt: Arc::new(RwLock::new(None)),
        }
    }

    /// Calculate next reconnection delay
    pub async fn next_delay(&self) -> Option<tokio::time::Duration> {
        match self.strategy {
            ReconnectionStrategy::None => None,
            ReconnectionStrategy::Immediate => Some(tokio::time::Duration::from_millis(0)),
            ReconnectionStrategy::ExponentialBackoff {
                initial_delay_ms,
                max_delay_ms,
            } => {
                let attempt = *self.attempt_count.read().await;
                let delay_ms = (initial_delay_ms * 2u64.pow(attempt)).min(max_delay_ms);
                Some(tokio::time::Duration::from_millis(delay_ms))
            }
            ReconnectionStrategy::FixedInterval { interval_ms } => {
                Some(tokio::time::Duration::from_millis(interval_ms))
            }
        }
    }

    /// Record reconnection attempt
    pub async fn record_attempt(&self) {
        *self.attempt_count.write().await += 1;
        *self.last_attempt.write().await = Some(chrono::Utc::now());
    }

    /// Reset reconnection counter
    pub async fn reset(&self) {
        *self.attempt_count.write().await = 0;
        *self.last_attempt.write().await = None;
    }

    /// Get attempt count
    pub async fn attempts(&self) -> u32 {
        *self.attempt_count.read().await
    }
}

/// Message batching manager
pub struct MessageBatcher {
    batch: Arc<RwLock<Vec<SyncMessage>>>,
    max_batch_size: usize,
    flush_interval_ms: u64,
}

impl MessageBatcher {
    /// Create a new message batcher
    pub fn new(max_batch_size: usize, flush_interval_ms: u64) -> Self {
        Self {
            batch: Arc::new(RwLock::new(Vec::new())),
            max_batch_size,
            flush_interval_ms,
        }
    }

    /// Add message to batch
    pub async fn add(&self, message: SyncMessage) -> Option<Vec<SyncMessage>> {
        let mut batch = self.batch.write().await;
        batch.push(message);

        if batch.len() >= self.max_batch_size {
            Some(std::mem::take(&mut *batch))
        } else {
            None
        }
    }

    /// Flush current batch
    pub async fn flush(&self) -> Vec<SyncMessage> {
        let mut batch = self.batch.write().await;
        std::mem::take(&mut *batch)
    }

    /// Start automatic flushing
    pub fn start_auto_flush(&self) -> tokio::task::JoinHandle<()> {
        let batch = self.batch.clone();
        let interval_ms = self.flush_interval_ms;

        tokio::spawn(async move {
            let mut interval =
                tokio::time::interval(tokio::time::Duration::from_millis(interval_ms));

            loop {
                interval.tick().await;
                batch.write().await.clear();
            }
        })
    }

    /// Get batch size
    pub async fn size(&self) -> usize {
        self.batch.read().await.len()
    }
}

/// Heartbeat manager for connection monitoring
pub struct HeartbeatManager {
    interval_ms: u64,
    timeout_ms: u64,
    last_heartbeat: Arc<RwLock<chrono::DateTime<chrono::Utc>>>,
}

impl HeartbeatManager {
    /// Create a new heartbeat manager
    pub fn new(interval_ms: u64, timeout_ms: u64) -> Self {
        Self {
            interval_ms,
            timeout_ms,
            last_heartbeat: Arc::new(RwLock::new(chrono::Utc::now())),
        }
    }

    /// Record heartbeat
    pub async fn record_heartbeat(&self) {
        *self.last_heartbeat.write().await = chrono::Utc::now();
    }

    /// Check if connection is alive
    pub async fn is_alive(&self) -> bool {
        let last = *self.last_heartbeat.read().await;
        let now = chrono::Utc::now();
        let elapsed = now.signed_duration_since(last);

        elapsed.num_milliseconds() < self.timeout_ms as i64
    }

    /// Start heartbeat loop
    pub fn start_heartbeat_loop<F>(&self, mut send_ping: F) -> tokio::task::JoinHandle<()>
    where
        F: FnMut() + Send + 'static,
    {
        let interval_ms = self.interval_ms;

        tokio::spawn(async move {
            let mut interval =
                tokio::time::interval(tokio::time::Duration::from_millis(interval_ms));

            loop {
                interval.tick().await;
                send_ping();
            }
        })
    }

    /// Get time since last heartbeat
    pub async fn time_since_last(&self) -> chrono::Duration {
        let last = *self.last_heartbeat.read().await;
        let now = chrono::Utc::now();
        now.signed_duration_since(last)
    }
}

/// Type alias for bandwidth samples.
type BandwidthSamples = Arc<RwLock<VecDeque<(chrono::DateTime<chrono::Utc>, usize)>>>;

/// Bandwidth monitor
pub struct BandwidthMonitor {
    window_size_ms: u64,
    samples: BandwidthSamples,
}

impl BandwidthMonitor {
    /// Create a new bandwidth monitor
    pub fn new(window_size_ms: u64) -> Self {
        Self {
            window_size_ms,
            samples: Arc::new(RwLock::new(VecDeque::new())),
        }
    }

    /// Record data transfer
    pub async fn record(&self, bytes: usize) {
        let mut samples = self.samples.write().await;
        let now = chrono::Utc::now();

        samples.push_back((now, bytes));

        // Remove old samples outside window
        let cutoff = now - chrono::Duration::milliseconds(self.window_size_ms as i64);
        while let Some((timestamp, _)) = samples.front() {
            if *timestamp < cutoff {
                samples.pop_front();
            } else {
                break;
            }
        }
    }

    /// Get current bandwidth in bytes per second
    pub async fn current_bandwidth(&self) -> f64 {
        let samples = self.samples.read().await;

        if samples.is_empty() {
            return 0.0;
        }

        let total_bytes: usize = samples.iter().map(|(_, bytes)| bytes).sum();
        let window_secs = self.window_size_ms as f64 / 1000.0;

        total_bytes as f64 / window_secs
    }

    /// Get average message size
    pub async fn average_message_size(&self) -> f64 {
        let samples = self.samples.read().await;

        if samples.is_empty() {
            return 0.0;
        }

        let total_bytes: usize = samples.iter().map(|(_, bytes)| bytes).sum();
        total_bytes as f64 / samples.len() as f64
    }

    /// Clear samples
    pub async fn clear(&self) {
        self.samples.write().await.clear();
    }
}

/// Connection pool manager
pub struct ConnectionPool {
    connections: Arc<RwLock<HashMap<Uuid, Arc<SyncConnection>>>>,
    max_connections: usize,
}

impl ConnectionPool {
    /// Create a new connection pool
    pub fn new(max_connections: usize) -> Self {
        Self {
            connections: Arc::new(RwLock::new(HashMap::new())),
            max_connections,
        }
    }

    /// Add connection to pool
    pub async fn add(&self, connection: Arc<SyncConnection>) -> Result<()> {
        let mut connections = self.connections.write().await;

        if connections.len() >= self.max_connections {
            return Err(CollabError::SyncError("Connection pool full".to_string()));
        }

        connections.insert(connection.id, connection);
        Ok(())
    }

    /// Remove connection from pool
    pub async fn remove(&self, connection_id: Uuid) -> Option<Arc<SyncConnection>> {
        self.connections.write().await.remove(&connection_id)
    }

    /// Get connection from pool
    pub async fn get(&self, connection_id: Uuid) -> Option<Arc<SyncConnection>> {
        self.connections.read().await.get(&connection_id).cloned()
    }

    /// Get all connections for a session
    pub async fn get_by_session(&self, session_id: Uuid) -> Vec<Arc<SyncConnection>> {
        self.connections
            .read()
            .await
            .values()
            .filter(|conn| conn.session_id == session_id)
            .cloned()
            .collect()
    }

    /// Get connection count
    pub async fn size(&self) -> usize {
        self.connections.read().await.len()
    }

    /// Clear all connections
    pub async fn clear(&self) {
        self.connections.write().await.clear();
    }
}

/// Message throttling manager
pub struct ThrottleManager {
    max_messages_per_second: u32,
    message_times: Arc<RwLock<VecDeque<chrono::DateTime<chrono::Utc>>>>,
}

impl ThrottleManager {
    /// Create a new throttle manager
    pub fn new(max_messages_per_second: u32) -> Self {
        Self {
            max_messages_per_second,
            message_times: Arc::new(RwLock::new(VecDeque::new())),
        }
    }

    /// Check if message can be sent
    pub async fn can_send(&self) -> bool {
        let mut times = self.message_times.write().await;
        let now = chrono::Utc::now();
        let one_second_ago = now - chrono::Duration::seconds(1);

        // Remove old timestamps
        while let Some(time) = times.front() {
            if *time < one_second_ago {
                times.pop_front();
            } else {
                break;
            }
        }

        times.len() < self.max_messages_per_second as usize
    }

    /// Record message sent
    pub async fn record_send(&self) {
        let mut times = self.message_times.write().await;
        times.push_back(chrono::Utc::now());
    }

    /// Get current message rate
    pub async fn current_rate(&self) -> u32 {
        let times = self.message_times.read().await;
        times.len() as u32
    }

    /// Wait until message can be sent
    pub async fn wait_if_needed(&self) {
        while !self.can_send().await {
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        }
    }
}

/// Quality of Service monitor
pub struct QoSMonitor {
    packet_loss: Arc<RwLock<f64>>,
    jitter_ms: Arc<RwLock<f64>>,
    latencies: Arc<RwLock<VecDeque<u64>>>,
    max_samples: usize,
}

impl QoSMonitor {
    /// Create a new QoS monitor
    pub fn new(max_samples: usize) -> Self {
        Self {
            packet_loss: Arc::new(RwLock::new(0.0)),
            jitter_ms: Arc::new(RwLock::new(0.0)),
            latencies: Arc::new(RwLock::new(VecDeque::new())),
            max_samples,
        }
    }

    /// Record latency sample
    pub async fn record_latency(&self, latency_ms: u64) {
        let mut latencies = self.latencies.write().await;

        latencies.push_back(latency_ms);

        if latencies.len() > self.max_samples {
            latencies.pop_front();
        }

        // Calculate jitter (variance in latency)
        if latencies.len() >= 2 {
            let mut total_diff = 0.0;
            let mut prev = latencies[0];

            for &latency in latencies.iter().skip(1) {
                total_diff += (latency as i64 - prev as i64).abs() as f64;
                prev = latency;
            }

            *self.jitter_ms.write().await = total_diff / (latencies.len() - 1) as f64;
        }
    }

    /// Record packet loss
    pub async fn record_packet_loss(&self, loss_percentage: f64) {
        *self.packet_loss.write().await = loss_percentage;
    }

    /// Get average latency
    pub async fn average_latency(&self) -> f64 {
        let latencies = self.latencies.read().await;

        if latencies.is_empty() {
            return 0.0;
        }

        let sum: u64 = latencies.iter().sum();
        sum as f64 / latencies.len() as f64
    }

    /// Get current jitter
    pub async fn jitter(&self) -> f64 {
        *self.jitter_ms.read().await
    }

    /// Get packet loss
    pub async fn packet_loss(&self) -> f64 {
        *self.packet_loss.read().await
    }

    /// Get connection quality score (0-100)
    pub async fn quality_score(&self) -> f64 {
        let latency = self.average_latency().await;
        let jitter = self.jitter().await;
        let loss = self.packet_loss().await;

        // Simple scoring algorithm
        let latency_score = (1.0 - (latency / 1000.0).min(1.0)) * 40.0;
        let jitter_score = (1.0 - (jitter / 100.0).min(1.0)) * 30.0;
        let loss_score = (1.0 - loss) * 30.0;

        latency_score + jitter_score + loss_score
    }
}

/// Type alias for sync coordinator sessions.
type SyncSessions = Arc<RwLock<HashMap<Uuid, Arc<RwLock<Vec<Arc<SyncConnection>>>>>>>;

/// Sync coordinator for managing multiple sync sessions
pub struct SyncCoordinator {
    sessions: SyncSessions,
    #[allow(dead_code)]
    config: CollabConfig,
}

impl SyncCoordinator {
    /// Create a new sync coordinator
    pub fn new(config: CollabConfig) -> Self {
        Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }

    /// Register a session
    pub async fn register_session(&self, session_id: Uuid) -> Result<()> {
        let mut sessions = self.sessions.write().await;
        sessions.insert(session_id, Arc::new(RwLock::new(Vec::new())));
        Ok(())
    }

    /// Add connection to session
    pub async fn add_connection(
        &self,
        session_id: Uuid,
        connection: Arc<SyncConnection>,
    ) -> Result<()> {
        let sessions = self.sessions.read().await;

        let connections = sessions
            .get(&session_id)
            .ok_or(CollabError::SessionNotFound(session_id))?;

        connections.write().await.push(connection);
        Ok(())
    }

    /// Broadcast message to all connections in session
    pub async fn broadcast(&self, session_id: Uuid, message: SyncMessage) -> Result<usize> {
        let sessions = self.sessions.read().await;

        let connections = sessions
            .get(&session_id)
            .ok_or(CollabError::SessionNotFound(session_id))?;

        let connections = connections.read().await;
        let mut sent = 0;

        for conn in connections.iter() {
            if conn.state().await == ConnectionState::Connected {
                conn.send_message(message.clone()).await?;
                sent += 1;
            }
        }

        Ok(sent)
    }

    /// Get connection count for session
    pub async fn connection_count(&self, session_id: Uuid) -> usize {
        let sessions = self.sessions.read().await;

        match sessions.get(&session_id) {
            Some(conns) => conns.read().await.len(),
            None => 0,
        }
    }

    /// Remove session
    pub async fn remove_session(&self, session_id: Uuid) -> Result<()> {
        self.sessions.write().await.remove(&session_id);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Rate limiting
// ---------------------------------------------------------------------------

/// Error returned when the sync rate limit is exceeded.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RateLimitError {
    /// The token bucket is exhausted; the operation should be retried later.
    #[error("Sync rate limit exceeded: bucket empty (rate={rate}/s, burst={burst})")]
    BucketExhausted {
        /// Configured steady-state rate in operations per second.
        rate: u32,
        /// Configured burst capacity.
        burst: u32,
    },
}

/// Token-bucket rate limiter for sync operations.
///
/// Tokens refill at `rate` per second up to `burst` capacity.
/// Each call to [`TokenBucket::check_and_consume`] attempts to withdraw
/// one token without blocking.
pub struct TokenBucket {
    /// Maximum number of tokens that can accumulate (burst capacity).
    burst: u32,
    /// Refill rate in tokens per second.
    rate: u32,
    /// Current token count (may be fractional internally).
    tokens: f64,
    /// Last refill instant.
    last_refill: std::time::Instant,
}

impl TokenBucket {
    /// Create a new token bucket.
    ///
    /// # Parameters
    /// - `rate`: steady-state operations per second.
    /// - `burst`: maximum number of tokens that can accumulate.
    pub fn new(rate: u32, burst: u32) -> Self {
        Self {
            burst,
            rate,
            tokens: burst as f64,
            last_refill: std::time::Instant::now(),
        }
    }

    /// Refill tokens based on elapsed wall-clock time.
    fn refill(&mut self) {
        let now = std::time::Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        self.tokens = (self.tokens + elapsed * self.rate as f64).min(self.burst as f64);
        self.last_refill = now;
    }

    /// Non-blocking attempt to consume one token.
    ///
    /// Returns `Ok(())` if a token was available, or
    /// [`RateLimitError::BucketExhausted`] if the bucket is empty.
    pub fn check_and_consume(&mut self) -> std::result::Result<(), RateLimitError> {
        self.refill();
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            Ok(())
        } else {
            Err(RateLimitError::BucketExhausted {
                rate: self.rate,
                burst: self.burst,
            })
        }
    }

    /// Current token count (for inspection/testing).
    pub fn tokens(&self) -> f64 {
        self.tokens
    }
}

/// Rate limiter for sync operation dispatch.
///
/// Wraps a [`TokenBucket`] and exposes a higher-level API that maps
/// rate-limit failures to [`CollabError::SyncError`].
pub struct SyncRateLimiter {
    bucket: parking_lot::Mutex<TokenBucket>,
}

impl SyncRateLimiter {
    /// Create a new rate limiter.
    ///
    /// # Parameters
    /// - `rate`: steady-state operations per second allowed per connection.
    /// - `burst`: maximum burst capacity.
    pub fn new(rate: u32, burst: u32) -> Self {
        Self {
            bucket: parking_lot::Mutex::new(TokenBucket::new(rate, burst)),
        }
    }

    /// Check and consume one rate-limit token.
    ///
    /// Returns `Ok(())` if the operation is allowed, or a
    /// [`CollabError::SyncError`] if the limit is exceeded.
    pub fn check_and_consume(&self) -> Result<()> {
        self.bucket
            .lock()
            .check_and_consume()
            .map_err(|e| CollabError::SyncError(e.to_string()))
    }

    /// Expose the underlying [`RateLimitError`] directly (for callers that
    /// want to distinguish rate-limit failures from other sync errors).
    pub fn check_and_consume_raw(&self) -> std::result::Result<(), RateLimitError> {
        self.bucket.lock().check_and_consume()
    }
}

#[cfg(test)]
mod extended_tests {
    use super::*;

    #[tokio::test]
    async fn test_reconnection_manager() {
        let manager = ReconnectionManager::new(ReconnectionStrategy::ExponentialBackoff {
            initial_delay_ms: 100,
            max_delay_ms: 1000,
        });

        manager.record_attempt().await;
        assert_eq!(manager.attempts().await, 1);

        let delay = manager
            .next_delay()
            .await
            .expect("collab test operation should succeed");
        assert!(delay.as_millis() >= 100);

        manager.reset().await;
        assert_eq!(manager.attempts().await, 0);
    }

    #[tokio::test]
    async fn test_message_batcher() {
        let batcher = MessageBatcher::new(3, 1000);

        batcher.add(SyncMessage::Ping).await;
        batcher.add(SyncMessage::Pong).await;

        assert_eq!(batcher.size().await, 2);

        let batch = batcher.add(SyncMessage::QueryState).await;
        assert!(batch.is_some());
        assert_eq!(
            batch.expect("collab test operation should succeed").len(),
            3
        );
    }

    #[tokio::test]
    async fn test_heartbeat_manager() {
        let manager = HeartbeatManager::new(1000, 5000);

        assert!(manager.is_alive().await);

        manager.record_heartbeat().await;
        assert!(manager.is_alive().await);

        let time_since = manager.time_since_last().await;
        assert!(time_since.num_milliseconds() < 100);
    }

    #[tokio::test]
    async fn test_bandwidth_monitor() {
        let monitor = BandwidthMonitor::new(1000);

        monitor.record(1000).await;
        monitor.record(2000).await;
        monitor.record(1500).await;

        let bandwidth = monitor.current_bandwidth().await;
        assert!(bandwidth > 0.0);

        let avg_size = monitor.average_message_size().await;
        assert_eq!(avg_size, 1500.0);
    }

    #[tokio::test]
    async fn test_connection_pool() {
        let pool = ConnectionPool::new(5);
        let conn = Arc::new(SyncConnection::new(
            Uuid::new_v4(),
            Uuid::new_v4(),
            CollabConfig::default(),
        ));

        pool.add(conn.clone())
            .await
            .expect("collab test operation should succeed");
        assert_eq!(pool.size().await, 1);

        let retrieved = pool
            .get(conn.id)
            .await
            .expect("collab test operation should succeed");
        assert_eq!(retrieved.id, conn.id);
    }

    #[tokio::test]
    async fn test_throttle_manager() {
        let throttle = ThrottleManager::new(10);

        assert!(throttle.can_send().await);

        for _ in 0..10 {
            throttle.record_send().await;
        }

        assert!(!throttle.can_send().await);
        assert_eq!(throttle.current_rate().await, 10);
    }

    #[tokio::test]
    async fn test_qos_monitor() {
        let monitor = QoSMonitor::new(100);

        monitor.record_latency(50).await;
        monitor.record_latency(60).await;
        monitor.record_latency(55).await;

        let avg_latency = monitor.average_latency().await;
        assert!((avg_latency - 55.0).abs() < 5.0);

        let quality = monitor.quality_score().await;
        assert!(quality > 0.0 && quality <= 100.0);
    }

    #[tokio::test]
    async fn test_sync_coordinator() {
        let coordinator = SyncCoordinator::new(CollabConfig::default());
        let session_id = Uuid::new_v4();

        coordinator
            .register_session(session_id)
            .await
            .expect("collab test operation should succeed");

        let conn = Arc::new(SyncConnection::new(
            Uuid::new_v4(),
            session_id,
            CollabConfig::default(),
        ));

        coordinator
            .add_connection(session_id, conn)
            .await
            .expect("collab test operation should succeed");
        assert_eq!(coordinator.connection_count(session_id).await, 1);
    }

    // ---- OfflineEditQueue tests ----

    #[tokio::test]
    async fn test_offline_queue_enqueue_and_len() {
        let mut q = OfflineEditQueue::new(100, OfflineConflictStrategy::ServerWins);
        let seq = q.enqueue(1, "track:0:0-1000", SyncMessage::Ping);
        assert!(seq.is_ok());
        assert_eq!(q.len(), 1);
        assert!(!q.is_empty());
    }

    #[tokio::test]
    async fn test_offline_queue_full_rejects() {
        let mut q = OfflineEditQueue::new(2, OfflineConflictStrategy::ServerWins);
        assert!(q.enqueue(1, "k1", SyncMessage::Ping).is_ok());
        assert!(q.enqueue(2, "k2", SyncMessage::Pong).is_ok());
        assert!(q.enqueue(3, "k3", SyncMessage::QueryState).is_err());
    }

    #[tokio::test]
    async fn test_offline_flush_no_conflict() {
        let mut q = OfflineEditQueue::new(100, OfflineConflictStrategy::ServerWins);
        q.enqueue(5, "track:0", SyncMessage::Ping)
            .expect("should enqueue");
        q.enqueue(6, "track:1", SyncMessage::Pong)
            .expect("should enqueue");
        // No server clocks set → no conflict
        let result = q.flush();
        assert_eq!(result.applied.len(), 2);
        assert!(result.dropped.is_empty());
        assert!(result.merged.is_empty());
        assert!(q.is_empty());
    }

    #[tokio::test]
    async fn test_offline_flush_server_wins_drops_stale() {
        let mut q = OfflineEditQueue::new(100, OfflineConflictStrategy::ServerWins);
        q.enqueue(3, "track:0", SyncMessage::Ping)
            .expect("should enqueue");
        q.enqueue(10, "track:1", SyncMessage::Pong)
            .expect("should enqueue");
        // Server's clock for track:0 is 5 → local edit at 3 is stale
        q.update_server_clock("track:0", 5);
        let result = q.flush();
        assert_eq!(result.applied.len(), 1);
        assert_eq!(result.dropped.len(), 1);
        assert_eq!(result.dropped[0].target_key, "track:0");
    }

    #[tokio::test]
    async fn test_offline_flush_client_wins_keeps_all() {
        let mut q = OfflineEditQueue::new(100, OfflineConflictStrategy::ClientWins);
        q.enqueue(3, "track:0", SyncMessage::Ping)
            .expect("should enqueue");
        q.update_server_clock("track:0", 5);
        let result = q.flush();
        assert_eq!(result.applied.len(), 1);
        assert!(result.dropped.is_empty());
    }

    #[tokio::test]
    async fn test_offline_flush_merge_by_timestamp_equal_clock() {
        let mut q = OfflineEditQueue::new(100, OfflineConflictStrategy::MergeByTimestamp);
        q.enqueue(5, "track:0", SyncMessage::Ping)
            .expect("should enqueue");
        q.update_server_clock("track:0", 5); // same clock → merge
        let result = q.flush();
        assert!(result.applied.is_empty());
        assert!(result.dropped.is_empty());
        assert_eq!(result.merged.len(), 1);
    }

    #[tokio::test]
    async fn test_offline_flush_merge_by_timestamp_older_drops() {
        let mut q = OfflineEditQueue::new(100, OfflineConflictStrategy::MergeByTimestamp);
        q.enqueue(3, "track:0", SyncMessage::Ping)
            .expect("should enqueue");
        q.update_server_clock("track:0", 5); // local is older → drop
        let result = q.flush();
        assert!(result.applied.is_empty());
        assert_eq!(result.dropped.len(), 1);
        assert!(result.merged.is_empty());
    }

    #[tokio::test]
    async fn test_offline_queue_clear() {
        let mut q = OfflineEditQueue::new(100, OfflineConflictStrategy::ServerWins);
        q.enqueue(1, "k", SyncMessage::Ping)
            .expect("should enqueue");
        q.clear();
        assert!(q.is_empty());
    }

    #[tokio::test]
    async fn test_offline_queue_pending_view() {
        let mut q = OfflineEditQueue::new(100, OfflineConflictStrategy::ServerWins);
        q.enqueue(1, "k1", SyncMessage::Ping)
            .expect("should enqueue");
        q.enqueue(2, "k2", SyncMessage::Pong)
            .expect("should enqueue");
        assert_eq!(q.pending().len(), 2);
        assert_eq!(q.pending()[0].seq, 0);
        assert_eq!(q.pending()[1].seq, 1);
    }

    #[tokio::test]
    async fn test_offline_queue_strategy() {
        let q = OfflineEditQueue::new(10, OfflineConflictStrategy::ClientWins);
        assert_eq!(q.strategy(), OfflineConflictStrategy::ClientWins);
    }

    // ---- SyncRateLimiter / TokenBucket tests ----

    #[test]
    fn test_rate_limiter_allows_burst() {
        // Burst of 5 means we can fire 5 ops immediately.
        let limiter = SyncRateLimiter::new(10, 5);
        for i in 0..5 {
            assert!(
                limiter.check_and_consume().is_ok(),
                "burst op {} should succeed",
                i
            );
        }
        // 6th attempt must fail — bucket is empty.
        assert!(limiter.check_and_consume().is_err());
    }

    #[test]
    fn test_rate_limiter_error_kind() {
        let limiter = SyncRateLimiter::new(2, 1);
        limiter.check_and_consume().expect("first should succeed");
        let err = limiter
            .check_and_consume_raw()
            .expect_err("second should fail");
        let RateLimitError::BucketExhausted { rate, burst } = err;
        assert_eq!(rate, 2);
        assert_eq!(burst, 1);
    }

    #[test]
    fn test_token_bucket_starts_full() {
        let bucket = TokenBucket::new(10, 20);
        assert!((bucket.tokens() - 20.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_token_bucket_refills_over_time() {
        let mut bucket = TokenBucket::new(1000, 10);
        // Drain all tokens.
        for _ in 0..10 {
            bucket.check_and_consume().expect("drain");
        }
        assert!(bucket.check_and_consume().is_err());

        // Wait 20 ms (should add ~20 tokens at 1000/s) and retry.
        std::thread::sleep(std::time::Duration::from_millis(20));
        assert!(bucket.check_and_consume().is_ok());
    }
}
