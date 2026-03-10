//! NDI connection configuration and connection management.
//!
//! Provides types for configuring NDI connections (receiver and sender modes),
//! and a `ConnectionManager` that tracks active connections and their states.

#![allow(dead_code)]

use std::collections::HashMap;
use std::net::SocketAddr;
use std::time::{Duration, Instant};

/// The operational mode for an NDI connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConnectionMode {
    /// Full-quality receive mode (highest bandwidth).
    HighQuality,
    /// Low-bandwidth proxy receive mode.
    LowBandwidth,
    /// Audio-only receive mode.
    AudioOnly,
    /// Sender mode: broadcasting frames to the network.
    Sender,
}

impl ConnectionMode {
    /// Returns true if this mode involves receiving video.
    pub fn receives_video(self) -> bool {
        matches!(self, Self::HighQuality | Self::LowBandwidth)
    }

    /// Returns true if this mode involves receiving or sending audio.
    pub fn handles_audio(self) -> bool {
        matches!(
            self,
            Self::HighQuality | Self::LowBandwidth | Self::AudioOnly | Self::Sender
        )
    }

    /// Approximate maximum bandwidth in Mbit/s for this mode (0 = unlimited).
    pub fn max_bandwidth_mbps(self) -> u32 {
        match self {
            Self::HighQuality => 0,
            Self::LowBandwidth => 100,
            Self::AudioOnly => 5,
            Self::Sender => 0,
        }
    }
}

/// Configuration for a single NDI connection.
#[derive(Debug, Clone)]
pub struct NdiConnectionConfig {
    /// The remote endpoint address.
    pub remote_addr: SocketAddr,
    /// Connection mode.
    pub mode: ConnectionMode,
    /// Connection timeout.
    pub connect_timeout: Duration,
    /// How long to wait before retrying a failed connection.
    pub retry_interval: Duration,
    /// Maximum number of reconnect attempts (0 = unlimited).
    pub max_retries: u32,
    /// Enable keepalive pings on the connection.
    pub keepalive: bool,
    /// Receive buffer size in frames.
    pub buffer_frames: usize,
}

impl NdiConnectionConfig {
    /// Create a new configuration for the given address and mode.
    pub fn new(remote_addr: SocketAddr, mode: ConnectionMode) -> Self {
        Self {
            remote_addr,
            mode,
            connect_timeout: Duration::from_secs(10),
            retry_interval: Duration::from_secs(2),
            max_retries: 5,
            keepalive: true,
            buffer_frames: 8,
        }
    }

    /// Set the connection timeout.
    pub fn with_connect_timeout(mut self, timeout: Duration) -> Self {
        self.connect_timeout = timeout;
        self
    }

    /// Set the retry interval.
    pub fn with_retry_interval(mut self, interval: Duration) -> Self {
        self.retry_interval = interval;
        self
    }

    /// Set the maximum number of retries.
    pub fn with_max_retries(mut self, max: u32) -> Self {
        self.max_retries = max;
        self
    }

    /// Set whether keepalive is enabled.
    pub fn with_keepalive(mut self, enabled: bool) -> Self {
        self.keepalive = enabled;
        self
    }

    /// Set the receive buffer frame count.
    pub fn with_buffer_frames(mut self, frames: usize) -> Self {
        self.buffer_frames = frames;
        self
    }
}

/// The current state of a managed NDI connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionState {
    /// Not yet connected.
    Idle,
    /// Currently attempting to connect.
    Connecting,
    /// Successfully connected.
    Connected,
    /// Connection lost; attempting to reconnect.
    Reconnecting,
    /// Permanently disconnected (max retries exceeded or manual close).
    Closed,
}

impl ConnectionState {
    /// Returns true if the connection is actively usable.
    pub fn is_active(self) -> bool {
        self == Self::Connected
    }

    /// Returns true if the connection should be retried.
    pub fn should_retry(self) -> bool {
        matches!(self, Self::Reconnecting)
    }
}

/// Internal record for a managed connection.
#[derive(Debug)]
struct ConnectionRecord {
    config: NdiConnectionConfig,
    state: ConnectionState,
    retry_count: u32,
    last_state_change: Instant,
    connected_at: Option<Instant>,
}

impl ConnectionRecord {
    fn new(config: NdiConnectionConfig) -> Self {
        Self {
            config,
            state: ConnectionState::Idle,
            retry_count: 0,
            last_state_change: Instant::now(),
            connected_at: None,
        }
    }

    fn set_state(&mut self, new_state: ConnectionState) {
        self.state = new_state;
        self.last_state_change = Instant::now();
        if new_state == ConnectionState::Connected {
            self.connected_at = Some(Instant::now());
        }
    }
}

/// Manages multiple NDI connections identified by a string key.
#[derive(Debug, Default)]
pub struct ConnectionManager {
    connections: HashMap<String, ConnectionRecord>,
}

impl ConnectionManager {
    /// Create a new, empty connection manager.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new connection configuration under the given key.
    ///
    /// If a connection with that key already exists it is replaced.
    pub fn register(&mut self, key: impl Into<String>, config: NdiConnectionConfig) {
        self.connections
            .insert(key.into(), ConnectionRecord::new(config));
    }

    /// Remove a connection record, returning whether it existed.
    pub fn deregister(&mut self, key: &str) -> bool {
        self.connections.remove(key).is_some()
    }

    /// Mark a connection as connected.
    pub fn set_connected(&mut self, key: &str) -> bool {
        if let Some(rec) = self.connections.get_mut(key) {
            rec.set_state(ConnectionState::Connected);
            rec.retry_count = 0;
            true
        } else {
            false
        }
    }

    /// Mark a connection as disconnected (triggers retry if retries remain).
    ///
    /// Returns the new state after applying retry logic.
    pub fn set_disconnected(&mut self, key: &str) -> Option<ConnectionState> {
        let rec = self.connections.get_mut(key)?;
        rec.retry_count += 1;
        let max = rec.config.max_retries;
        if max > 0 && rec.retry_count >= max {
            rec.set_state(ConnectionState::Closed);
        } else {
            rec.set_state(ConnectionState::Reconnecting);
        }
        Some(rec.state)
    }

    /// Mark a connection as closed (permanent).
    pub fn close(&mut self, key: &str) -> bool {
        if let Some(rec) = self.connections.get_mut(key) {
            rec.set_state(ConnectionState::Closed);
            true
        } else {
            false
        }
    }

    /// Get the current state of a connection.
    pub fn state(&self, key: &str) -> Option<ConnectionState> {
        self.connections.get(key).map(|r| r.state)
    }

    /// Returns the number of connections currently in the `Connected` state.
    pub fn active_count(&self) -> usize {
        self.connections
            .values()
            .filter(|r| r.state.is_active())
            .count()
    }

    /// Returns the total number of registered connections.
    pub fn total_count(&self) -> usize {
        self.connections.len()
    }

    /// Returns keys of all connections that need to be retried.
    pub fn pending_retry_keys(&self) -> Vec<String> {
        self.connections
            .iter()
            .filter(|(_, r)| r.state.should_retry())
            .map(|(k, _)| k.clone())
            .collect()
    }

    /// Returns the uptime of a connected connection, or `None` if not connected.
    pub fn uptime(&self, key: &str) -> Option<Duration> {
        self.connections
            .get(key)
            .and_then(|r| r.connected_at)
            .map(|t| t.elapsed())
    }

    /// Get the config for a registered connection.
    pub fn config(&self, key: &str) -> Option<&NdiConnectionConfig> {
        self.connections.get(key).map(|r| &r.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::SocketAddr;

    fn test_addr() -> SocketAddr {
        "127.0.0.1:5960".parse().expect("expected valid parse")
    }

    fn make_config(mode: ConnectionMode) -> NdiConnectionConfig {
        NdiConnectionConfig::new(test_addr(), mode)
    }

    #[test]
    fn test_connection_mode_receives_video() {
        assert!(ConnectionMode::HighQuality.receives_video());
        assert!(ConnectionMode::LowBandwidth.receives_video());
        assert!(!ConnectionMode::AudioOnly.receives_video());
        assert!(!ConnectionMode::Sender.receives_video());
    }

    #[test]
    fn test_connection_mode_handles_audio() {
        assert!(ConnectionMode::AudioOnly.handles_audio());
        assert!(ConnectionMode::Sender.handles_audio());
        assert!(ConnectionMode::HighQuality.handles_audio());
    }

    #[test]
    fn test_connection_mode_bandwidth() {
        assert_eq!(ConnectionMode::LowBandwidth.max_bandwidth_mbps(), 100);
        assert_eq!(ConnectionMode::AudioOnly.max_bandwidth_mbps(), 5);
        assert_eq!(ConnectionMode::HighQuality.max_bandwidth_mbps(), 0);
    }

    #[test]
    fn test_config_builder() {
        let cfg = make_config(ConnectionMode::HighQuality)
            .with_connect_timeout(Duration::from_secs(5))
            .with_retry_interval(Duration::from_secs(1))
            .with_max_retries(3)
            .with_keepalive(false)
            .with_buffer_frames(4);
        assert_eq!(cfg.connect_timeout, Duration::from_secs(5));
        assert_eq!(cfg.max_retries, 3);
        assert!(!cfg.keepalive);
        assert_eq!(cfg.buffer_frames, 4);
    }

    #[test]
    fn test_manager_register_and_state() {
        let mut mgr = ConnectionManager::new();
        mgr.register("cam1", make_config(ConnectionMode::HighQuality));
        assert_eq!(mgr.state("cam1"), Some(ConnectionState::Idle));
        assert_eq!(mgr.total_count(), 1);
    }

    #[test]
    fn test_manager_deregister() {
        let mut mgr = ConnectionManager::new();
        mgr.register("cam1", make_config(ConnectionMode::HighQuality));
        assert!(mgr.deregister("cam1"));
        assert!(!mgr.deregister("cam1")); // second call returns false
        assert_eq!(mgr.total_count(), 0);
    }

    #[test]
    fn test_manager_set_connected() {
        let mut mgr = ConnectionManager::new();
        mgr.register("cam1", make_config(ConnectionMode::HighQuality));
        assert!(mgr.set_connected("cam1"));
        assert_eq!(mgr.state("cam1"), Some(ConnectionState::Connected));
        assert_eq!(mgr.active_count(), 1);
    }

    #[test]
    fn test_manager_set_disconnected_triggers_retry() {
        let mut mgr = ConnectionManager::new();
        let cfg = make_config(ConnectionMode::HighQuality).with_max_retries(3);
        mgr.register("cam1", cfg);
        mgr.set_connected("cam1");
        let new_state = mgr
            .set_disconnected("cam1")
            .expect("expected state transition");
        assert_eq!(new_state, ConnectionState::Reconnecting);
    }

    #[test]
    fn test_manager_set_disconnected_closes_after_max_retries() {
        let mut mgr = ConnectionManager::new();
        let cfg = make_config(ConnectionMode::HighQuality).with_max_retries(2);
        mgr.register("cam1", cfg);
        mgr.set_connected("cam1");
        mgr.set_disconnected("cam1"); // retry_count = 1
        let state = mgr
            .set_disconnected("cam1")
            .expect("expected state transition"); // retry_count = 2 >= 2
        assert_eq!(state, ConnectionState::Closed);
    }

    #[test]
    fn test_manager_pending_retry_keys() {
        let mut mgr = ConnectionManager::new();
        mgr.register(
            "cam1",
            make_config(ConnectionMode::HighQuality).with_max_retries(10),
        );
        mgr.register(
            "cam2",
            make_config(ConnectionMode::LowBandwidth).with_max_retries(10),
        );
        mgr.set_connected("cam1");
        mgr.set_disconnected("cam1");
        let keys = mgr.pending_retry_keys();
        assert!(keys.contains(&"cam1".to_string()));
        assert!(!keys.contains(&"cam2".to_string()));
    }

    #[test]
    fn test_manager_uptime_none_when_not_connected() {
        let mut mgr = ConnectionManager::new();
        mgr.register("cam1", make_config(ConnectionMode::HighQuality));
        assert!(mgr.uptime("cam1").is_none());
    }

    #[test]
    fn test_manager_uptime_some_when_connected() {
        let mut mgr = ConnectionManager::new();
        mgr.register("cam1", make_config(ConnectionMode::HighQuality));
        mgr.set_connected("cam1");
        assert!(mgr.uptime("cam1").is_some());
    }

    #[test]
    fn test_connection_state_is_active() {
        assert!(ConnectionState::Connected.is_active());
        assert!(!ConnectionState::Idle.is_active());
        assert!(!ConnectionState::Reconnecting.is_active());
        assert!(!ConnectionState::Closed.is_active());
    }

    #[test]
    fn test_manager_config_retrieval() {
        let mut mgr = ConnectionManager::new();
        mgr.register("cam1", make_config(ConnectionMode::AudioOnly));
        let cfg = mgr.config("cam1").expect("expected config to exist");
        assert_eq!(cfg.mode, ConnectionMode::AudioOnly);
    }

    #[test]
    fn test_manager_close() {
        let mut mgr = ConnectionManager::new();
        mgr.register("cam1", make_config(ConnectionMode::HighQuality));
        mgr.set_connected("cam1");
        assert!(mgr.close("cam1"));
        assert_eq!(mgr.state("cam1"), Some(ConnectionState::Closed));
        assert_eq!(mgr.active_count(), 0);
    }
}
