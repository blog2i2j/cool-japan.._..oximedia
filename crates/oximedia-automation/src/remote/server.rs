//! Remote control server.

use crate::remote::api::ApiRouter;
use crate::remote::websocket::WebSocketHandler;
use crate::Result;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn};

/// Remote server configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteConfig {
    /// Server bind address
    pub bind_address: String,
    /// Server port
    pub port: u16,
    /// Enable authentication
    pub require_auth: bool,
    /// Enable WebSocket
    pub enable_websocket: bool,
}

impl Default for RemoteConfig {
    fn default() -> Self {
        Self {
            bind_address: "0.0.0.0".to_string(),
            port: 8080,
            require_auth: true,
            enable_websocket: true,
        }
    }
}

/// Remote control server.
#[allow(dead_code)]
pub struct RemoteServer {
    config: RemoteConfig,
    api_router: ApiRouter,
    websocket_handler: Option<WebSocketHandler>,
    running: Arc<RwLock<bool>>,
}

impl RemoteServer {
    /// Create a new remote control server.
    pub fn new(config: RemoteConfig) -> Self {
        info!(
            "Creating remote control server at {}:{}",
            config.bind_address, config.port
        );

        let websocket_handler = if config.enable_websocket {
            Some(WebSocketHandler::new())
        } else {
            None
        };

        Self {
            config: config.clone(),
            api_router: ApiRouter::new(config),
            websocket_handler,
            running: Arc::new(RwLock::new(false)),
        }
    }

    /// Start the remote control server.
    pub async fn start(&mut self) -> Result<()> {
        info!("Starting remote control server");

        {
            let mut running = self.running.write().await;
            *running = true;
        }

        // In a real implementation, this would:
        // 1. Start the HTTP/REST API server (using axum)
        // 2. Start the WebSocket server if enabled
        // 3. Set up authentication middleware
        // 4. Configure CORS
        // 5. Start metrics endpoint

        Ok(())
    }

    /// Stop the remote control server.
    pub async fn stop(&mut self) -> Result<()> {
        info!("Stopping remote control server");

        let mut running = self.running.write().await;
        *running = false;

        Ok(())
    }

    /// Check if server is running.
    pub async fn is_running(&self) -> bool {
        *self.running.read().await
    }

    /// Get server address.
    pub fn address(&self) -> String {
        format!("{}:{}", self.config.bind_address, self.config.port)
    }

    /// Broadcast WebSocket message.
    pub async fn broadcast(&self, message: String) -> Result<()> {
        if let Some(ref handler) = self.websocket_handler {
            handler.broadcast(message).await?;
        } else {
            warn!("WebSocket not enabled, cannot broadcast message");
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_remote_server_creation() {
        let config = RemoteConfig::default();
        let server = RemoteServer::new(config);
        assert_eq!(server.address(), "0.0.0.0:8080");
    }

    #[tokio::test]
    async fn test_server_lifecycle() {
        let config = RemoteConfig::default();
        let mut server = RemoteServer::new(config);

        assert!(!server.is_running().await);
        server.start().await.expect("operation should succeed");
        assert!(server.is_running().await);
        server.stop().await.expect("operation should succeed");
        assert!(!server.is_running().await);
    }
}
