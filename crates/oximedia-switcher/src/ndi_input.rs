//! NDI (Network Device Interface) input support for the video switcher.
//!
//! NDI is a royalty-free, low-latency standard for transmitting video and
//! audio over Ethernet networks.  This module provides a pure-Rust model for
//! discovering, connecting to, and managing NDI sources as live inputs to the
//! switcher.
//!
//! The module is intentionally free of any native NDI SDK dependency so that
//! the crate can compile on all targets.  A `ndi` feature flag is reserved for
//! future integration with a native binding crate.
//!
//! # Architecture
//!
//! ```text
//!  ┌─────────────────────────────────────┐
//!  │            NdiInputManager          │
//!  │  ┌──────────────┐  ┌─────────────┐ │
//!  │  │  NdiSource[] │  │ NdiReceiver │ │
//!  │  └──────────────┘  └─────────────┘ │
//!  │       discovery         active      │
//!  └─────────────────────────────────────┘
//!          ↓ found sources
//!  ┌────────────────────────┐
//!  │ Switcher InputRouter   │
//!  └────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```rust
//! use oximedia_switcher::ndi_input::{NdiInputManager, NdiSource, NdiReceiverConfig};
//!
//! let mut manager = NdiInputManager::new();
//!
//! // Manually register a known source (simulating discovery)
//! let src = NdiSource::new("CameraOp1", "192.168.1.50:5960");
//! manager.register_source(src);
//!
//! assert_eq!(manager.source_count(), 1);
//!
//! // Connect to the source
//! let cfg = NdiReceiverConfig::default();
//! manager.connect(0, cfg).expect("connect ok");
//! assert!(manager.is_connected(0));
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

// ────────────────────────────────────────────────────────────────────────────
// Errors
// ────────────────────────────────────────────────────────────────────────────

/// Errors produced by the NDI input subsystem.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum NdiInputError {
    /// The requested source index is out of range.
    #[error("NDI source index {0} is out of range (total sources: {1})")]
    SourceOutOfRange(usize, usize),

    /// Attempted to connect to a source that is already connected.
    #[error("NDI source {0} is already connected")]
    AlreadyConnected(usize),

    /// Attempted to disconnect a source that is not connected.
    #[error("NDI source {0} is not connected")]
    NotConnected(usize),

    /// The source name is empty.
    #[error("NDI source name must not be empty")]
    EmptySourceName,

    /// The source address is empty or malformed.
    #[error("NDI source address '{0}' is invalid")]
    InvalidAddress(String),

    /// The bandwidth mode is incompatible with the configured receiver.
    #[error("Bandwidth mode {0:?} is not supported by this receiver configuration")]
    UnsupportedBandwidth(BandwidthMode),

    /// Maximum simultaneous receiver count exceeded.
    #[error("Cannot connect more than {0} simultaneous NDI receivers")]
    ReceiverLimitExceeded(usize),
}

// ────────────────────────────────────────────────────────────────────────────
// NDI bandwidth / quality modes
// ────────────────────────────────────────────────────────────────────────────

/// Bandwidth mode for an NDI receiver.
///
/// Higher bandwidth delivers full-quality frames; lower bandwidth reduces
/// network load at the cost of resolution or quality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BandwidthMode {
    /// Receive full-quality, highest-bandwidth video (default).
    Highest,
    /// Receive a lower-resolution proxy stream suitable for multiview monitoring.
    Lowest,
    /// Receive only audio, no video.
    AudioOnly,
    /// Receive metadata streams only (e.g., tally, PTZ data).
    MetadataOnly,
}

impl std::fmt::Display for BandwidthMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BandwidthMode::Highest => write!(f, "Highest"),
            BandwidthMode::Lowest => write!(f, "Lowest"),
            BandwidthMode::AudioOnly => write!(f, "AudioOnly"),
            BandwidthMode::MetadataOnly => write!(f, "MetadataOnly"),
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Connection state
// ────────────────────────────────────────────────────────────────────────────

/// Connection state of an NDI receiver.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NdiConnectionState {
    /// Not connected; no network activity.
    Disconnected,
    /// Attempting to connect to the source.
    Connecting,
    /// Actively receiving video/audio from the source.
    Connected,
    /// Connection lost; awaiting reconnect.
    Reconnecting,
    /// Connection was explicitly closed by the user.
    Closed,
}

impl NdiConnectionState {
    /// Returns `true` if frames are currently being received.
    pub fn is_active(&self) -> bool {
        matches!(self, NdiConnectionState::Connected)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Source descriptor
// ────────────────────────────────────────────────────────────────────────────

/// Describes a discovered or manually registered NDI source.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NdiSource {
    /// Human-readable source name as broadcast by the sender.
    pub name: String,
    /// Network address of the source in `host:port` format.
    pub address: String,
    /// Optional group name for NDI grouping / multicast.
    pub groups: Vec<String>,
    /// Whether this source was found via mDNS/DNS-SD discovery or manually registered.
    pub discovered: bool,
}

impl NdiSource {
    /// Create a new NDI source descriptor.
    pub fn new(name: impl Into<String>, address: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            address: address.into(),
            groups: Vec::new(),
            discovered: false,
        }
    }

    /// Create a discovered NDI source (found via network discovery).
    pub fn discovered(name: impl Into<String>, address: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            address: address.into(),
            groups: Vec::new(),
            discovered: true,
        }
    }

    /// Add the source to an NDI group.
    pub fn with_group(mut self, group: impl Into<String>) -> Self {
        self.groups.push(group.into());
        self
    }

    /// Validate that the source descriptor contains non-empty fields.
    pub fn validate(&self) -> Result<(), NdiInputError> {
        if self.name.is_empty() {
            return Err(NdiInputError::EmptySourceName);
        }
        if self.address.is_empty() {
            return Err(NdiInputError::InvalidAddress(self.address.clone()));
        }
        Ok(())
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Receiver configuration
// ────────────────────────────────────────────────────────────────────────────

/// Configuration for an NDI receiver connection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NdiReceiverConfig {
    /// Bandwidth mode for this connection.
    pub bandwidth: BandwidthMode,
    /// Whether to allow field-dominant interlaced video.
    pub allow_interlaced: bool,
    /// Preferred colour space.  Ignored when `bandwidth` is `AudioOnly`.
    pub color_format: NdiColorFormat,
    /// Maximum number of frames to buffer before dropping.
    pub max_queue_depth: usize,
    /// Timeout in milliseconds when waiting for the first frame.
    pub connect_timeout_ms: u64,
    /// Whether to automatically reconnect after a connection loss.
    pub auto_reconnect: bool,
}

impl Default for NdiReceiverConfig {
    fn default() -> Self {
        Self {
            bandwidth: BandwidthMode::Highest,
            allow_interlaced: false,
            color_format: NdiColorFormat::Uyvy,
            max_queue_depth: 4,
            connect_timeout_ms: 5000,
            auto_reconnect: true,
        }
    }
}

/// Colour format emitted by the NDI receiver.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NdiColorFormat {
    /// UYVY 4:2:2 (most efficient for SDI-style processing).
    Uyvy,
    /// BGRA 4:4:4:4 with alpha channel.
    Bgra,
    /// Fastest format available on the platform.
    Fastest,
    /// Highest-quality format available on the platform.
    Best,
}

// ────────────────────────────────────────────────────────────────────────────
// Active receiver
// ────────────────────────────────────────────────────────────────────────────

/// Represents a live NDI receiver connection.
#[derive(Debug)]
pub struct NdiReceiver {
    /// Index of the source this receiver is connected to.
    pub source_index: usize,
    /// Configuration in use.
    pub config: NdiReceiverConfig,
    /// Current connection state.
    pub state: NdiConnectionState,
    /// Frames received since last reset.
    pub frames_received: u64,
    /// Frames dropped since last reset (queue overflow).
    pub frames_dropped: u64,
    /// Tally data received from the remote end (if any).
    pub tally: Option<NdiTallyData>,
}

impl NdiReceiver {
    fn new(source_index: usize, config: NdiReceiverConfig) -> Self {
        Self {
            source_index,
            config,
            state: NdiConnectionState::Connecting,
            frames_received: 0,
            frames_dropped: 0,
            tally: None,
        }
    }

    /// Simulate completing the connection handshake.
    pub fn on_connected(&mut self) {
        self.state = NdiConnectionState::Connected;
    }

    /// Simulate receiving a video frame.
    ///
    /// Returns `true` if the frame was accepted, `false` if the queue is full
    /// and the frame was dropped.
    pub fn receive_frame(&mut self) -> bool {
        if self.frames_received % (self.config.max_queue_depth as u64 + 1)
            == self.config.max_queue_depth as u64
        {
            // Simulate occasional overflow.
            self.frames_dropped += 1;
            false
        } else {
            self.frames_received += 1;
            true
        }
    }

    /// Update the tally state received from the remote sender.
    pub fn update_tally(&mut self, tally: NdiTallyData) {
        self.tally = Some(tally);
    }

    /// Returns `true` if currently connected and receiving data.
    pub fn is_active(&self) -> bool {
        self.state.is_active()
    }

    /// Statistics summary.
    pub fn stats(&self) -> NdiReceiverStats {
        NdiReceiverStats {
            frames_received: self.frames_received,
            frames_dropped: self.frames_dropped,
            drop_ratio: if self.frames_received + self.frames_dropped == 0 {
                0.0
            } else {
                self.frames_dropped as f64 / (self.frames_received + self.frames_dropped) as f64
            },
        }
    }
}

/// Tally information sent over the NDI tally back-channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct NdiTallyData {
    /// This source is on-air (program output).
    pub on_program: bool,
    /// This source is on preview.
    pub on_preview: bool,
}

/// Snapshot of receiver performance counters.
#[derive(Debug, Clone)]
pub struct NdiReceiverStats {
    /// Total frames accepted into the queue.
    pub frames_received: u64,
    /// Total frames discarded due to queue overflow.
    pub frames_dropped: u64,
    /// Ratio of dropped to total (0.0 – 1.0).
    pub drop_ratio: f64,
}

// ────────────────────────────────────────────────────────────────────────────
// Discovery record
// ────────────────────────────────────────────────────────────────────────────

/// A record returned by the NDI discovery sub-system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NdiDiscoveryRecord {
    /// Index assigned by the manager.
    pub index: usize,
    /// Source descriptor.
    pub source: NdiSource,
    /// Whether this source is currently connected.
    pub connected: bool,
}

// ────────────────────────────────────────────────────────────────────────────
// Manager
// ────────────────────────────────────────────────────────────────────────────

/// Maximum simultaneous receivers supported.
const MAX_RECEIVERS: usize = 64;

/// Manages NDI source discovery and active receiver connections.
///
/// In a real production system this would wrap the native NDI SDK's finder and
/// receiver objects.  Here we provide a pure-Rust simulation suitable for
/// integration testing and protocol-layer logic.
#[derive(Debug)]
pub struct NdiInputManager {
    /// All known sources (discovered + manually registered).
    sources: Vec<NdiSource>,
    /// Active receivers keyed by source index.
    receivers: HashMap<usize, NdiReceiver>,
    /// Global enable/disable for NDI discovery.
    discovery_enabled: bool,
    /// Groups to scan during discovery (empty = all groups).
    scan_groups: Vec<String>,
}

impl NdiInputManager {
    /// Create a new manager with discovery enabled.
    pub fn new() -> Self {
        Self {
            sources: Vec::new(),
            receivers: HashMap::new(),
            discovery_enabled: true,
            scan_groups: Vec::new(),
        }
    }

    /// Create a manager scoped to specific NDI groups.
    pub fn with_groups(groups: Vec<String>) -> Self {
        Self {
            sources: Vec::new(),
            receivers: HashMap::new(),
            discovery_enabled: true,
            scan_groups: groups,
        }
    }

    // ── Source management ────────────────────────────────────────────────────

    /// Register a source (manual or via discovery callback).
    pub fn register_source(&mut self, source: NdiSource) -> Result<usize, NdiInputError> {
        source.validate()?;
        let idx = self.sources.len();
        self.sources.push(source);
        Ok(idx)
    }

    /// Remove a previously registered source.
    ///
    /// Disconnects the receiver if one is active.
    pub fn remove_source(&mut self, index: usize) -> Result<(), NdiInputError> {
        if index >= self.sources.len() {
            return Err(NdiInputError::SourceOutOfRange(index, self.sources.len()));
        }
        self.receivers.remove(&index);
        self.sources.remove(index);
        // Re-key receivers with indices > removed index
        let remapped: HashMap<usize, NdiReceiver> = self
            .receivers
            .drain()
            .map(|(k, v)| if k > index { (k - 1, v) } else { (k, v) })
            .collect();
        self.receivers = remapped;
        Ok(())
    }

    /// Returns the total number of registered sources.
    pub fn source_count(&self) -> usize {
        self.sources.len()
    }

    /// Look up a source by index.
    pub fn source(&self, index: usize) -> Option<&NdiSource> {
        self.sources.get(index)
    }

    /// Return a list of all known sources with their connection status.
    pub fn list_sources(&self) -> Vec<NdiDiscoveryRecord> {
        self.sources
            .iter()
            .enumerate()
            .map(|(i, src)| NdiDiscoveryRecord {
                index: i,
                source: src.clone(),
                connected: self.receivers.contains_key(&i),
            })
            .collect()
    }

    // ── Receiver management ──────────────────────────────────────────────────

    /// Connect to a source.
    ///
    /// Returns an error if already connected or the source index is invalid.
    pub fn connect(
        &mut self,
        source_index: usize,
        config: NdiReceiverConfig,
    ) -> Result<(), NdiInputError> {
        if source_index >= self.sources.len() {
            return Err(NdiInputError::SourceOutOfRange(
                source_index,
                self.sources.len(),
            ));
        }
        if self.receivers.contains_key(&source_index) {
            return Err(NdiInputError::AlreadyConnected(source_index));
        }
        if self.receivers.len() >= MAX_RECEIVERS {
            return Err(NdiInputError::ReceiverLimitExceeded(MAX_RECEIVERS));
        }
        let mut receiver = NdiReceiver::new(source_index, config);
        // Immediately simulate a successful connection in this pure-Rust model.
        receiver.on_connected();
        self.receivers.insert(source_index, receiver);
        Ok(())
    }

    /// Disconnect the receiver for a source.
    pub fn disconnect(&mut self, source_index: usize) -> Result<(), NdiInputError> {
        if source_index >= self.sources.len() {
            return Err(NdiInputError::SourceOutOfRange(
                source_index,
                self.sources.len(),
            ));
        }
        if !self.receivers.contains_key(&source_index) {
            return Err(NdiInputError::NotConnected(source_index));
        }
        self.receivers.remove(&source_index);
        Ok(())
    }

    /// Returns `true` if the source at `index` has an active receiver.
    pub fn is_connected(&self, source_index: usize) -> bool {
        self.receivers
            .get(&source_index)
            .map(|r| r.is_active())
            .unwrap_or(false)
    }

    /// Get an immutable reference to the receiver for a source.
    pub fn receiver(&self, source_index: usize) -> Option<&NdiReceiver> {
        self.receivers.get(&source_index)
    }

    /// Get a mutable reference to the receiver for a source.
    pub fn receiver_mut(&mut self, source_index: usize) -> Option<&mut NdiReceiver> {
        self.receivers.get_mut(&source_index)
    }

    /// Returns the number of currently active receivers.
    pub fn active_receiver_count(&self) -> usize {
        self.receivers.values().filter(|r| r.is_active()).count()
    }

    // ── Discovery control ────────────────────────────────────────────────────

    /// Enable or disable automatic source discovery.
    pub fn set_discovery_enabled(&mut self, enabled: bool) {
        self.discovery_enabled = enabled;
    }

    /// Returns `true` if source discovery is currently enabled.
    pub fn discovery_enabled(&self) -> bool {
        self.discovery_enabled
    }

    /// Return a reference to the configured scan groups.
    pub fn scan_groups(&self) -> &[String] {
        &self.scan_groups
    }

    /// Add an NDI group to the discovery scan list.
    pub fn add_scan_group(&mut self, group: impl Into<String>) {
        self.scan_groups.push(group.into());
    }

    // ── Tally distribution ───────────────────────────────────────────────────

    /// Push a tally update to all connected receivers.
    ///
    /// In a real system this would encode and transmit NDI tally metadata back
    /// to the sender.  Here we update the in-memory state.
    pub fn broadcast_tally(&mut self, on_program: &[usize], on_preview: &[usize]) {
        for (source_index, receiver) in &mut self.receivers {
            let tally = NdiTallyData {
                on_program: on_program.contains(source_index),
                on_preview: on_preview.contains(source_index),
            };
            receiver.update_tally(tally);
        }
    }

    // ── Aggregate statistics ─────────────────────────────────────────────────

    /// Return aggregate statistics across all active receivers.
    pub fn aggregate_stats(&self) -> NdiAggregateStats {
        let mut total_received = 0u64;
        let mut total_dropped = 0u64;
        for receiver in self.receivers.values() {
            total_received += receiver.frames_received;
            total_dropped += receiver.frames_dropped;
        }
        NdiAggregateStats {
            total_sources: self.sources.len(),
            connected_receivers: self.receivers.len(),
            total_frames_received: total_received,
            total_frames_dropped: total_dropped,
        }
    }
}

/// Aggregate performance statistics across all NDI receivers.
#[derive(Debug, Clone)]
pub struct NdiAggregateStats {
    /// Total number of registered sources.
    pub total_sources: usize,
    /// Number of sources currently connected.
    pub connected_receivers: usize,
    /// Total frames received across all receivers.
    pub total_frames_received: u64,
    /// Total frames dropped across all receivers.
    pub total_frames_dropped: u64,
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_manager_with_sources(n: usize) -> NdiInputManager {
        let mut m = NdiInputManager::new();
        for i in 0..n {
            let src = NdiSource::new(format!("Camera{i}"), format!("192.168.1.{i}:5960"));
            m.register_source(src).expect("register ok");
        }
        m
    }

    #[test]
    fn test_register_and_count_sources() {
        let mut manager = NdiInputManager::new();
        let src = NdiSource::new("Cam1", "10.0.0.1:5960");
        manager.register_source(src).expect("register ok");
        assert_eq!(manager.source_count(), 1);
    }

    #[test]
    fn test_register_empty_name_fails() {
        let mut manager = NdiInputManager::new();
        let src = NdiSource::new("", "10.0.0.1:5960");
        assert!(matches!(
            manager.register_source(src),
            Err(NdiInputError::EmptySourceName)
        ));
    }

    #[test]
    fn test_connect_and_is_connected() {
        let mut manager = make_manager_with_sources(2);
        let cfg = NdiReceiverConfig::default();
        manager.connect(0, cfg).expect("connect ok");
        assert!(manager.is_connected(0));
        assert!(!manager.is_connected(1));
    }

    #[test]
    fn test_connect_already_connected_error() {
        let mut manager = make_manager_with_sources(1);
        let cfg = NdiReceiverConfig::default();
        manager.connect(0, cfg.clone()).expect("first connect ok");
        assert!(matches!(
            manager.connect(0, cfg),
            Err(NdiInputError::AlreadyConnected(0))
        ));
    }

    #[test]
    fn test_connect_out_of_range_error() {
        let mut manager = make_manager_with_sources(1);
        let cfg = NdiReceiverConfig::default();
        assert!(matches!(
            manager.connect(5, cfg),
            Err(NdiInputError::SourceOutOfRange(5, 1))
        ));
    }

    #[test]
    fn test_disconnect() {
        let mut manager = make_manager_with_sources(1);
        let cfg = NdiReceiverConfig::default();
        manager.connect(0, cfg).expect("connect ok");
        manager.disconnect(0).expect("disconnect ok");
        assert!(!manager.is_connected(0));
    }

    #[test]
    fn test_disconnect_not_connected_error() {
        let mut manager = make_manager_with_sources(1);
        assert!(matches!(
            manager.disconnect(0),
            Err(NdiInputError::NotConnected(0))
        ));
    }

    #[test]
    fn test_broadcast_tally() {
        let mut manager = make_manager_with_sources(3);
        let cfg = NdiReceiverConfig::default();
        manager.connect(0, cfg.clone()).expect("ok");
        manager.connect(1, cfg.clone()).expect("ok");
        manager.connect(2, cfg).expect("ok");

        manager.broadcast_tally(&[0], &[1]);

        let r0 = manager.receiver(0).expect("exists");
        assert_eq!(
            r0.tally,
            Some(NdiTallyData {
                on_program: true,
                on_preview: false
            })
        );
        let r1 = manager.receiver(1).expect("exists");
        assert_eq!(
            r1.tally,
            Some(NdiTallyData {
                on_program: false,
                on_preview: true
            })
        );
        let r2 = manager.receiver(2).expect("exists");
        assert_eq!(
            r2.tally,
            Some(NdiTallyData {
                on_program: false,
                on_preview: false
            })
        );
    }

    #[test]
    fn test_aggregate_stats() {
        let mut manager = make_manager_with_sources(2);
        let cfg = NdiReceiverConfig::default();
        manager.connect(0, cfg.clone()).expect("ok");
        manager.connect(1, cfg).expect("ok");

        let stats = manager.aggregate_stats();
        assert_eq!(stats.total_sources, 2);
        assert_eq!(stats.connected_receivers, 2);
    }

    #[test]
    fn test_list_sources() {
        let mut manager = make_manager_with_sources(3);
        let cfg = NdiReceiverConfig::default();
        manager.connect(1, cfg).expect("ok");

        let list = manager.list_sources();
        assert_eq!(list.len(), 3);
        assert!(!list[0].connected);
        assert!(list[1].connected);
        assert!(!list[2].connected);
    }

    #[test]
    fn test_remove_source() {
        let mut manager = make_manager_with_sources(3);
        manager.remove_source(1).expect("remove ok");
        assert_eq!(manager.source_count(), 2);
        // Source that was at index 2 is now at index 1.
        assert!(manager.source(1).is_some());
    }

    #[test]
    fn test_receiver_frame_counting() {
        let mut manager = make_manager_with_sources(1);
        let cfg = NdiReceiverConfig {
            max_queue_depth: 10,
            ..Default::default()
        };
        manager.connect(0, cfg).expect("ok");
        let receiver = manager.receiver_mut(0).expect("exists");
        for _ in 0..5 {
            receiver.receive_frame();
        }
        assert!(receiver.frames_received > 0);
    }

    #[test]
    fn test_discovery_toggle() {
        let mut manager = NdiInputManager::new();
        assert!(manager.discovery_enabled());
        manager.set_discovery_enabled(false);
        assert!(!manager.discovery_enabled());
        manager.set_discovery_enabled(true);
        assert!(manager.discovery_enabled());
    }

    #[test]
    fn test_scan_groups() {
        let mut manager = NdiInputManager::new();
        manager.add_scan_group("Studio A");
        manager.add_scan_group("Studio B");
        assert_eq!(manager.scan_groups().len(), 2);
    }

    #[test]
    fn test_active_receiver_count() {
        let mut manager = make_manager_with_sources(4);
        let cfg = NdiReceiverConfig::default();
        manager.connect(0, cfg.clone()).expect("ok");
        manager.connect(2, cfg).expect("ok");
        assert_eq!(manager.active_receiver_count(), 2);
    }
}
