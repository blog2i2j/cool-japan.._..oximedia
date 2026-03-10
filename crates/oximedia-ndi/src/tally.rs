//! NDI tally light protocol
//!
//! This module implements the tally light protocol for NDI, which allows
//! receivers to indicate to sources whether they are on program, preview,
//! or neither.
#![allow(dead_code)]

use crate::{NdiError, Result};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{mpsc, Notify};
use tokio::task::JoinHandle;
use tracing::{debug, error, info, trace, warn};

/// Tally state for a source
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct TallyState {
    /// Whether this source is on program (red tally)
    pub program: bool,

    /// Whether this source is on preview (green tally)
    pub preview: bool,
}

impl TallyState {
    /// Create a new tally state
    pub fn new(program: bool, preview: bool) -> Self {
        Self { program, preview }
    }

    /// Create a tally state for program (red tally)
    pub fn program() -> Self {
        Self {
            program: true,
            preview: false,
        }
    }

    /// Create a tally state for preview (green tally)
    pub fn preview() -> Self {
        Self {
            program: false,
            preview: true,
        }
    }

    /// Create a tally state for off (no tally)
    pub fn off() -> Self {
        Self {
            program: false,
            preview: false,
        }
    }

    /// Check if any tally is active
    pub fn is_active(&self) -> bool {
        self.program || self.preview
    }

    /// Combine with another tally state (OR operation)
    pub fn combine(&self, other: &Self) -> Self {
        Self {
            program: self.program || other.program,
            preview: self.preview || other.preview,
        }
    }

    /// Encode to a byte
    pub fn encode(&self) -> u8 {
        let mut byte = 0u8;
        if self.program {
            byte |= 0x01;
        }
        if self.preview {
            byte |= 0x02;
        }
        byte
    }

    /// Decode from a byte
    pub fn decode(byte: u8) -> Self {
        Self {
            program: (byte & 0x01) != 0,
            preview: (byte & 0x02) != 0,
        }
    }
}

/// Tally message
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TallyMessage {
    /// Source name
    source_name: String,

    /// Tally state
    state: TallyState,

    /// Timestamp
    timestamp: i64,
}

impl TallyMessage {
    /// Create a new tally message
    fn new(source_name: String, state: TallyState) -> Self {
        Self {
            source_name,
            state,
            timestamp: chrono::Utc::now().timestamp_micros(),
        }
    }

    /// Encode to JSON bytes
    fn encode(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(self)
            .map_err(|e| NdiError::Protocol(format!("Failed to encode tally message: {}", e)))
    }

    /// Decode from JSON bytes
    fn decode(data: &[u8]) -> Result<Self> {
        serde_json::from_slice(data)
            .map_err(|e| NdiError::Protocol(format!("Failed to decode tally message: {}", e)))
    }
}

/// Tally server
///
/// Listens for tally updates from control systems and distributes them to NDI sources
pub struct TallyServer {
    /// Listening address
    address: SocketAddr,

    /// Tally states for each source
    states: Arc<RwLock<HashMap<String, TallyState>>>,

    /// State change callback
    callback: Arc<RwLock<Option<Box<dyn Fn(String, TallyState) + Send + Sync>>>>,

    /// Shutdown notify
    shutdown: Arc<Notify>,

    /// Server task handle
    task_handle: Arc<RwLock<Option<JoinHandle<()>>>>,
}

impl TallyServer {
    /// Create a new tally server
    pub async fn new(address: SocketAddr) -> Result<Self> {
        let server = Self {
            address,
            states: Arc::new(RwLock::new(HashMap::new())),
            callback: Arc::new(RwLock::new(None)),
            shutdown: Arc::new(Notify::new()),
            task_handle: Arc::new(RwLock::new(None)),
        };

        server.start().await?;
        Ok(server)
    }

    /// Start the tally server
    async fn start(&self) -> Result<()> {
        let listener = TcpListener::bind(self.address)
            .await
            .map_err(|e| NdiError::Network(e))?;

        info!("Tally server listening on {}", self.address);

        let states = self.states.clone();
        let callback = self.callback.clone();
        let shutdown = self.shutdown.clone();

        let handle = tokio::spawn(async move {
            loop {
                tokio::select! {
                    result = listener.accept() => {
                        match result {
                            Ok((stream, addr)) => {
                                debug!("New tally connection from {}", addr);
                                let states = states.clone();
                                let callback = callback.clone();

                                tokio::spawn(async move {
                                    if let Err(e) = Self::handle_connection(stream, states, callback).await {
                                        warn!("Tally connection error from {}: {}", addr, e);
                                    }
                                });
                            }
                            Err(e) => {
                                error!("Failed to accept tally connection: {}", e);
                            }
                        }
                    }
                    _ = shutdown.notified() => {
                        info!("Tally server shutting down");
                        break;
                    }
                }
            }
        });

        *self.task_handle.write() = Some(handle);
        Ok(())
    }

    /// Handle a tally connection
    async fn handle_connection(
        mut stream: TcpStream,
        states: Arc<RwLock<HashMap<String, TallyState>>>,
        callback: Arc<RwLock<Option<Box<dyn Fn(String, TallyState) + Send + Sync>>>>,
    ) -> Result<()> {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let mut buffer = vec![0u8; 4096];

        loop {
            let n = stream
                .read(&mut buffer)
                .await
                .map_err(|e| NdiError::Network(e))?;

            if n == 0 {
                break;
            }

            let message = TallyMessage::decode(&buffer[..n])?;
            trace!("Received tally update: {:?}", message);

            // Update state
            states
                .write()
                .insert(message.source_name.clone(), message.state);

            // Call callback
            if let Some(cb) = callback.read().as_ref() {
                cb(message.source_name.clone(), message.state);
            }

            // Send acknowledgment
            stream
                .write_all(b"OK\n")
                .await
                .map_err(|e| NdiError::Network(e))?;
        }

        Ok(())
    }

    /// Set a callback for state changes
    pub fn set_callback<F>(&self, callback: F)
    where
        F: Fn(String, TallyState) + Send + Sync + 'static,
    {
        *self.callback.write() = Some(Box::new(callback));
    }

    /// Get the tally state for a source
    pub fn get_state(&self, source_name: &str) -> Option<TallyState> {
        self.states.read().get(source_name).copied()
    }

    /// Set the tally state for a source
    pub fn set_state(&self, source_name: String, state: TallyState) {
        self.states.write().insert(source_name.clone(), state);

        // Call callback
        if let Some(cb) = self.callback.read().as_ref() {
            cb(source_name, state);
        }
    }

    /// Get all tally states
    pub fn get_all_states(&self) -> HashMap<String, TallyState> {
        self.states.read().clone()
    }

    /// Clear the tally state for a source
    pub fn clear_state(&self, source_name: &str) {
        self.states.write().remove(source_name);
    }

    /// Clear all tally states
    pub fn clear_all_states(&self) {
        self.states.write().clear();
    }

    /// Get the listening address
    pub fn address(&self) -> SocketAddr {
        self.address
    }

    /// Shutdown the tally server
    pub async fn shutdown(&self) {
        info!("Shutting down tally server");
        self.shutdown.notify_waiters();

        if let Some(handle) = self.task_handle.write().take() {
            let _ = handle.await;
        }
    }
}

impl Drop for TallyServer {
    fn drop(&mut self) {
        self.shutdown.notify_waiters();
    }
}

/// Tally client
///
/// Sends tally updates to a tally server
pub struct TallyClient {
    /// Server address
    server_address: SocketAddr,

    /// Message queue
    message_queue: mpsc::UnboundedSender<TallyMessage>,

    /// Client task handle
    task_handle: Arc<RwLock<Option<JoinHandle<()>>>>,

    /// Shutdown notify
    shutdown: Arc<Notify>,
}

impl TallyClient {
    /// Create a new tally client
    pub fn new(server_address: SocketAddr) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        let shutdown = Arc::new(Notify::new());

        let client = Self {
            server_address,
            message_queue: tx,
            task_handle: Arc::new(RwLock::new(None)),
            shutdown: shutdown.clone(),
        };

        // Start sender task
        let handle = tokio::spawn(Self::sender_task(server_address, rx, shutdown));
        *client.task_handle.write() = Some(handle);

        client
    }

    /// Sender task
    async fn sender_task(
        server_address: SocketAddr,
        mut rx: mpsc::UnboundedReceiver<TallyMessage>,
        shutdown: Arc<Notify>,
    ) {
        loop {
            tokio::select! {
                message = rx.recv() => {
                    match message {
                        Some(msg) => {
                            if let Err(e) = Self::send_message(server_address, &msg).await {
                                warn!("Failed to send tally message: {}", e);
                            }
                        }
                        None => {
                            debug!("Tally message queue closed");
                            break;
                        }
                    }
                }
                _ = shutdown.notified() => {
                    info!("Tally client shutting down");
                    break;
                }
            }
        }
    }

    /// Send a tally message
    async fn send_message(server_address: SocketAddr, message: &TallyMessage) -> Result<()> {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let mut stream = TcpStream::connect(server_address)
            .await
            .map_err(|e| NdiError::Network(e))?;

        let data = message.encode()?;
        stream
            .write_all(&data)
            .await
            .map_err(|e| NdiError::Network(e))?;

        // Wait for acknowledgment
        let mut buffer = vec![0u8; 8];
        let _n = stream
            .read(&mut buffer)
            .await
            .map_err(|e| NdiError::Network(e))?;

        Ok(())
    }

    /// Update tally state for a source
    pub fn update_tally(&self, source_name: String, state: TallyState) -> Result<()> {
        let message = TallyMessage::new(source_name, state);
        self.message_queue
            .send(message)
            .map_err(|_| NdiError::Protocol("Failed to queue tally message".to_string()))
    }

    /// Shutdown the tally client
    pub async fn shutdown(&self) {
        info!("Shutting down tally client");
        self.shutdown.notify_waiters();

        if let Some(handle) = self.task_handle.write().take() {
            let _ = handle.await;
        }
    }
}

impl Drop for TallyClient {
    fn drop(&mut self) {
        self.shutdown.notify_waiters();
    }
}

/// Tally aggregator
///
/// Aggregates tally states from multiple receivers
pub struct TallyAggregator {
    /// Individual tally states
    states: Arc<RwLock<HashMap<String, TallyState>>>,
}

impl TallyAggregator {
    /// Create a new tally aggregator
    pub fn new() -> Self {
        Self {
            states: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Update tally state from a receiver
    pub fn update(&self, receiver_id: String, state: TallyState) {
        self.states.write().insert(receiver_id, state);
    }

    /// Remove a receiver's tally state
    pub fn remove(&self, receiver_id: &str) {
        self.states.write().remove(receiver_id);
    }

    /// Get the combined tally state
    pub fn get_combined(&self) -> TallyState {
        let states = self.states.read();
        let mut combined = TallyState::off();

        for state in states.values() {
            combined = combined.combine(state);
        }

        combined
    }

    /// Get all individual tally states
    pub fn get_all(&self) -> HashMap<String, TallyState> {
        self.states.read().clone()
    }

    /// Clear all tally states
    pub fn clear(&self) {
        self.states.write().clear();
    }
}

impl Default for TallyAggregator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tally_state() {
        let state = TallyState::program();
        assert!(state.program);
        assert!(!state.preview);
        assert!(state.is_active());

        let state = TallyState::preview();
        assert!(!state.program);
        assert!(state.preview);
        assert!(state.is_active());

        let state = TallyState::off();
        assert!(!state.program);
        assert!(!state.preview);
        assert!(!state.is_active());
    }

    #[test]
    fn test_tally_state_combine() {
        let state1 = TallyState::program();
        let state2 = TallyState::preview();
        let combined = state1.combine(&state2);

        assert!(combined.program);
        assert!(combined.preview);
    }

    #[test]
    fn test_tally_state_encode_decode() {
        let state = TallyState::new(true, false);
        let encoded = state.encode();
        let decoded = TallyState::decode(encoded);

        assert_eq!(state, decoded);

        let state = TallyState::new(false, true);
        let encoded = state.encode();
        let decoded = TallyState::decode(encoded);

        assert_eq!(state, decoded);

        let state = TallyState::new(true, true);
        let encoded = state.encode();
        let decoded = TallyState::decode(encoded);

        assert_eq!(state, decoded);
    }

    #[test]
    fn test_tally_aggregator() {
        let aggregator = TallyAggregator::new();

        aggregator.update("receiver1".to_string(), TallyState::program());
        aggregator.update("receiver2".to_string(), TallyState::preview());

        let combined = aggregator.get_combined();
        assert!(combined.program);
        assert!(combined.preview);

        aggregator.remove("receiver1");
        let combined = aggregator.get_combined();
        assert!(!combined.program);
        assert!(combined.preview);

        aggregator.clear();
        let combined = aggregator.get_combined();
        assert!(!combined.program);
        assert!(!combined.preview);
    }

    #[test]
    fn test_tally_message() {
        let msg = TallyMessage::new("Test Source".to_string(), TallyState::program());
        let encoded = msg.encode().expect("encoding should succeed");
        let decoded = TallyMessage::decode(&encoded).expect("unexpected None/Err");

        assert_eq!(msg.source_name, decoded.source_name);
        assert_eq!(msg.state, decoded.state);
    }
}
