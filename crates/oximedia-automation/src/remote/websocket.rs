//! WebSocket real-time updates.

use crate::{AutomationError, Result};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use tracing::{debug, info};

/// WebSocket message type.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WebSocketMessage {
    /// Status update
    #[serde(rename = "status")]
    Status {
        /// Status data payload
        data: serde_json::Value,
    },

    /// Event notification
    #[serde(rename = "event")]
    Event {
        /// Event name
        event: String,
        /// Event data payload
        data: serde_json::Value,
    },

    /// Metrics update
    #[serde(rename = "metrics")]
    Metrics {
        /// Metrics data payload
        data: serde_json::Value,
    },

    /// Alert notification
    #[serde(rename = "alert")]
    Alert {
        /// Alert severity level
        severity: String,
        /// Alert message text
        message: String,
    },

    /// Ping
    #[serde(rename = "ping")]
    Ping,

    /// Pong
    #[serde(rename = "pong")]
    Pong,
}

/// WebSocket connection.
pub struct WebSocketConnection {
    id: String,
    tx: mpsc::UnboundedSender<String>,
}

impl WebSocketConnection {
    /// Create a new WebSocket connection.
    pub fn new(id: String, tx: mpsc::UnboundedSender<String>) -> Self {
        Self { id, tx }
    }

    /// Send message to connection.
    pub fn send(&self, message: String) -> Result<()> {
        self.tx
            .send(message)
            .map_err(|_| AutomationError::RemoteControl("Failed to send message".to_string()))?;
        Ok(())
    }

    /// Get connection ID.
    pub fn id(&self) -> &str {
        &self.id
    }
}

/// WebSocket handler.
pub struct WebSocketHandler {
    connections: Arc<RwLock<Vec<WebSocketConnection>>>,
}

impl WebSocketHandler {
    /// Create a new WebSocket handler.
    pub fn new() -> Self {
        info!("Creating WebSocket handler");

        Self {
            connections: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Add a new connection.
    pub async fn add_connection(&self, id: String, tx: mpsc::UnboundedSender<String>) {
        debug!("Adding WebSocket connection: {}", id);

        let connection = WebSocketConnection::new(id, tx);

        let mut connections = self.connections.write().await;
        connections.push(connection);
    }

    /// Remove a connection.
    pub async fn remove_connection(&self, id: &str) {
        debug!("Removing WebSocket connection: {}", id);

        let mut connections = self.connections.write().await;
        connections.retain(|conn| conn.id() != id);
    }

    /// Broadcast message to all connections.
    pub async fn broadcast(&self, message: String) -> Result<()> {
        debug!(
            "Broadcasting message to {} connections",
            self.connections.read().await.len()
        );

        let connections = self.connections.read().await;

        for connection in connections.iter() {
            if let Err(e) = connection.send(message.clone()) {
                debug!("Failed to send to connection {}: {}", connection.id(), e);
            }
        }

        Ok(())
    }

    /// Send message to specific connection.
    pub async fn send_to(&self, id: &str, message: String) -> Result<()> {
        let connections = self.connections.read().await;

        for connection in connections.iter() {
            if connection.id() == id {
                return connection.send(message);
            }
        }

        Err(AutomationError::NotFound(format!("Connection {id}")))
    }

    /// Get number of active connections.
    pub async fn connection_count(&self) -> usize {
        self.connections.read().await.len()
    }

    /// Broadcast status update.
    pub async fn broadcast_status(&self, data: serde_json::Value) -> Result<()> {
        let message = WebSocketMessage::Status { data };
        let json = serde_json::to_string(&message)?;
        self.broadcast(json).await
    }

    /// Broadcast event notification.
    pub async fn broadcast_event(&self, event: String, data: serde_json::Value) -> Result<()> {
        let message = WebSocketMessage::Event { event, data };
        let json = serde_json::to_string(&message)?;
        self.broadcast(json).await
    }

    /// Broadcast metrics update.
    pub async fn broadcast_metrics(&self, data: serde_json::Value) -> Result<()> {
        let message = WebSocketMessage::Metrics { data };
        let json = serde_json::to_string(&message)?;
        self.broadcast(json).await
    }

    /// Broadcast alert.
    pub async fn broadcast_alert(&self, severity: String, message: String) -> Result<()> {
        let msg = WebSocketMessage::Alert { severity, message };
        let json = serde_json::to_string(&msg)?;
        self.broadcast(json).await
    }
}

impl Default for WebSocketHandler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_websocket_handler() {
        let handler = WebSocketHandler::new();
        assert_eq!(handler.connection_count().await, 0);

        let (tx, _rx) = mpsc::unbounded_channel();
        handler.add_connection("conn1".to_string(), tx).await;

        assert_eq!(handler.connection_count().await, 1);

        handler.remove_connection("conn1").await;
        assert_eq!(handler.connection_count().await, 0);
    }

    #[tokio::test]
    async fn test_broadcast() {
        let handler = WebSocketHandler::new();

        let (tx, mut rx) = mpsc::unbounded_channel();
        handler.add_connection("conn1".to_string(), tx).await;

        handler
            .broadcast("test message".to_string())
            .await
            .expect("operation should succeed");

        let received = rx.recv().await.expect("recv should succeed");
        assert_eq!(received, "test message");
    }

    #[test]
    fn test_websocket_message_serialization() {
        let message = WebSocketMessage::Status {
            data: serde_json::json!({"status": "ok"}),
        };

        let json = serde_json::to_string(&message).expect("to_string should succeed");
        assert!(json.contains("\"type\":\"status\""));
    }
}
