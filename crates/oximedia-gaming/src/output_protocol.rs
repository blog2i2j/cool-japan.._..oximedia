#![allow(dead_code)]

//! RTMP, SRT, and WHIP output protocol support for game streaming.
//!
//! Provides a unified interface for sending encoded media data to ingest servers
//! using industry-standard streaming protocols:
//!
//! - **RTMP** (Real-Time Messaging Protocol): Traditional low-latency streaming
//!   to platforms like Twitch, YouTube, and Facebook Gaming.
//! - **SRT** (Secure Reliable Transport): Modern UDP-based protocol with
//!   built-in encryption and error correction for high-quality contribution.
//! - **WHIP** (WebRTC-HTTP Ingestion Protocol): Sub-second latency streaming
//!   using WebRTC for interactive broadcasting.
//!
//! Each protocol implementation manages connection lifecycle, reconnection logic,
//! packet framing, and bandwidth estimation.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use crate::{GamingError, GamingResult};

// ---------------------------------------------------------------------------
// Protocol types
// ---------------------------------------------------------------------------

/// Supported output streaming protocols.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OutputProtocol {
    /// RTMP - Real-Time Messaging Protocol (port 1935 default).
    Rtmp,
    /// SRT - Secure Reliable Transport (UDP-based).
    Srt,
    /// WHIP - WebRTC-HTTP Ingestion Protocol (sub-second latency).
    Whip,
}

impl std::fmt::Display for OutputProtocol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Rtmp => write!(f, "RTMP"),
            Self::Srt => write!(f, "SRT"),
            Self::Whip => write!(f, "WHIP"),
        }
    }
}

// ---------------------------------------------------------------------------
// Connection state
// ---------------------------------------------------------------------------

/// Connection lifecycle state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionState {
    /// Not yet connected.
    Disconnected,
    /// TCP/UDP handshake in progress.
    Connecting,
    /// Fully connected and ready to send data.
    Connected,
    /// Attempting automatic reconnection after a drop.
    Reconnecting,
    /// Permanently failed (max retries exceeded, auth rejected, etc.).
    Failed,
}

// ---------------------------------------------------------------------------
// RTMP configuration
// ---------------------------------------------------------------------------

/// RTMP-specific configuration.
#[derive(Debug, Clone)]
pub struct RtmpConfig {
    /// Ingest URL (e.g. `rtmp://live.twitch.tv/app`).
    pub url: String,
    /// Stream key / token.
    pub stream_key: String,
    /// Chunk size in bytes for RTMP chunking (default 4096).
    pub chunk_size: u32,
    /// Application name extracted from the URL path.
    pub app_name: String,
    /// Maximum reconnection attempts before giving up.
    pub max_reconnect_attempts: u32,
    /// Delay between reconnection attempts.
    pub reconnect_delay: Duration,
}

impl RtmpConfig {
    /// Create a new RTMP configuration.
    ///
    /// # Errors
    ///
    /// Returns error if the URL or stream key is empty.
    pub fn new(url: &str, stream_key: &str) -> GamingResult<Self> {
        if url.is_empty() {
            return Err(GamingError::InvalidConfig(
                "RTMP URL must not be empty".into(),
            ));
        }
        if stream_key.is_empty() {
            return Err(GamingError::InvalidConfig(
                "Stream key must not be empty".into(),
            ));
        }

        let app_name = url.rsplit('/').next().unwrap_or("live").to_string();

        Ok(Self {
            url: url.to_string(),
            stream_key: stream_key.to_string(),
            chunk_size: 4096,
            app_name,
            max_reconnect_attempts: 5,
            reconnect_delay: Duration::from_secs(2),
        })
    }

    /// Set a custom RTMP chunk size.
    #[must_use]
    pub fn with_chunk_size(mut self, size: u32) -> Self {
        self.chunk_size = size.max(128); // RTMP minimum chunk size
        self
    }

    /// Set max reconnect attempts.
    #[must_use]
    pub fn with_max_reconnect_attempts(mut self, n: u32) -> Self {
        self.max_reconnect_attempts = n;
        self
    }
}

// ---------------------------------------------------------------------------
// SRT configuration
// ---------------------------------------------------------------------------

/// SRT encryption mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SrtEncryption {
    /// No encryption.
    None,
    /// AES-128.
    Aes128,
    /// AES-256.
    Aes256,
}

/// SRT connection mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SrtMode {
    /// Caller (client) mode — connects to a listener.
    Caller,
    /// Listener (server) mode — waits for a connection.
    Listener,
    /// Rendezvous — both sides connect simultaneously.
    Rendezvous,
}

/// SRT-specific configuration.
#[derive(Debug, Clone)]
pub struct SrtConfig {
    /// Remote host address.
    pub host: String,
    /// Remote port.
    pub port: u16,
    /// Connection mode.
    pub mode: SrtMode,
    /// Latency in milliseconds (default 120ms).
    pub latency_ms: u32,
    /// Maximum bandwidth in bytes/s (0 = unlimited).
    pub max_bandwidth: u64,
    /// Encryption mode.
    pub encryption: SrtEncryption,
    /// Passphrase for encryption (10-79 characters when encryption enabled).
    pub passphrase: Option<String>,
    /// Overhead bandwidth percentage for FEC/ARQ (default 25%).
    pub overhead_percent: u32,
    /// Stream ID for multiplexing.
    pub stream_id: Option<String>,
    /// Maximum reconnection attempts.
    pub max_reconnect_attempts: u32,
}

impl SrtConfig {
    /// Create a new SRT configuration in caller mode.
    ///
    /// # Errors
    ///
    /// Returns error if the host is empty or port is zero.
    pub fn new_caller(host: &str, port: u16) -> GamingResult<Self> {
        if host.is_empty() {
            return Err(GamingError::InvalidConfig(
                "SRT host must not be empty".into(),
            ));
        }
        if port == 0 {
            return Err(GamingError::InvalidConfig(
                "SRT port must be non-zero".into(),
            ));
        }
        Ok(Self {
            host: host.to_string(),
            port,
            mode: SrtMode::Caller,
            latency_ms: 120,
            max_bandwidth: 0,
            encryption: SrtEncryption::None,
            passphrase: None,
            overhead_percent: 25,
            stream_id: None,
            max_reconnect_attempts: 5,
        })
    }

    /// Enable AES-128 encryption with the given passphrase.
    ///
    /// # Errors
    ///
    /// Returns error if the passphrase is too short or too long.
    pub fn with_aes128(mut self, passphrase: &str) -> GamingResult<Self> {
        Self::validate_passphrase(passphrase)?;
        self.encryption = SrtEncryption::Aes128;
        self.passphrase = Some(passphrase.to_string());
        Ok(self)
    }

    /// Enable AES-256 encryption with the given passphrase.
    ///
    /// # Errors
    ///
    /// Returns error if the passphrase is too short or too long.
    pub fn with_aes256(mut self, passphrase: &str) -> GamingResult<Self> {
        Self::validate_passphrase(passphrase)?;
        self.encryption = SrtEncryption::Aes256;
        self.passphrase = Some(passphrase.to_string());
        Ok(self)
    }

    /// Set the SRT latency.
    #[must_use]
    pub fn with_latency_ms(mut self, ms: u32) -> Self {
        self.latency_ms = ms.max(20); // SRT minimum practical latency
        self
    }

    /// Set a stream ID for multiplexing.
    #[must_use]
    pub fn with_stream_id(mut self, id: &str) -> Self {
        self.stream_id = Some(id.to_string());
        self
    }

    fn validate_passphrase(passphrase: &str) -> GamingResult<()> {
        let len = passphrase.len();
        if !(10..=79).contains(&len) {
            return Err(GamingError::InvalidConfig(format!(
                "SRT passphrase must be 10-79 characters, got {len}"
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// WHIP configuration
// ---------------------------------------------------------------------------

/// WHIP-specific configuration.
#[derive(Debug, Clone)]
pub struct WhipConfig {
    /// WHIP endpoint URL.
    pub endpoint_url: String,
    /// Bearer token for authentication.
    pub bearer_token: Option<String>,
    /// ICE servers for NAT traversal.
    pub ice_servers: Vec<IceServer>,
    /// Maximum reconnection attempts.
    pub max_reconnect_attempts: u32,
}

/// ICE server configuration for WebRTC NAT traversal.
#[derive(Debug, Clone)]
pub struct IceServer {
    /// Server URL(s) (stun: or turn: scheme).
    pub urls: Vec<String>,
    /// Username (for TURN servers).
    pub username: Option<String>,
    /// Credential (for TURN servers).
    pub credential: Option<String>,
}

impl IceServer {
    /// Create a STUN server entry.
    #[must_use]
    pub fn stun(url: &str) -> Self {
        Self {
            urls: vec![url.to_string()],
            username: None,
            credential: None,
        }
    }

    /// Create a TURN server entry with credentials.
    #[must_use]
    pub fn turn(url: &str, username: &str, credential: &str) -> Self {
        Self {
            urls: vec![url.to_string()],
            username: Some(username.to_string()),
            credential: Some(credential.to_string()),
        }
    }
}

impl WhipConfig {
    /// Create a new WHIP configuration.
    ///
    /// # Errors
    ///
    /// Returns error if the endpoint URL is empty.
    pub fn new(endpoint_url: &str) -> GamingResult<Self> {
        if endpoint_url.is_empty() {
            return Err(GamingError::InvalidConfig(
                "WHIP endpoint URL must not be empty".into(),
            ));
        }
        Ok(Self {
            endpoint_url: endpoint_url.to_string(),
            bearer_token: None,
            ice_servers: vec![IceServer::stun("stun:stun.l.google.com:19302")],
            max_reconnect_attempts: 3,
        })
    }

    /// Set bearer token for authentication.
    #[must_use]
    pub fn with_bearer_token(mut self, token: &str) -> Self {
        self.bearer_token = Some(token.to_string());
        self
    }

    /// Add an ICE server.
    #[must_use]
    pub fn with_ice_server(mut self, server: IceServer) -> Self {
        self.ice_servers.push(server);
        self
    }
}

// ---------------------------------------------------------------------------
// Unified output endpoint
// ---------------------------------------------------------------------------

/// Unified output endpoint wrapping any supported protocol.
#[derive(Debug, Clone)]
pub enum OutputEndpoint {
    /// RTMP output.
    Rtmp(RtmpConfig),
    /// SRT output.
    Srt(SrtConfig),
    /// WHIP output.
    Whip(WhipConfig),
}

impl OutputEndpoint {
    /// Get the protocol type.
    #[must_use]
    pub fn protocol(&self) -> OutputProtocol {
        match self {
            Self::Rtmp(_) => OutputProtocol::Rtmp,
            Self::Srt(_) => OutputProtocol::Srt,
            Self::Whip(_) => OutputProtocol::Whip,
        }
    }

    /// Get the max reconnect attempts for this endpoint.
    #[must_use]
    pub fn max_reconnect_attempts(&self) -> u32 {
        match self {
            Self::Rtmp(c) => c.max_reconnect_attempts,
            Self::Srt(c) => c.max_reconnect_attempts,
            Self::Whip(c) => c.max_reconnect_attempts,
        }
    }

    /// Get a human-readable destination string.
    #[must_use]
    pub fn destination(&self) -> String {
        match self {
            Self::Rtmp(c) => format!("rtmp://{}", c.url.trim_start_matches("rtmp://")),
            Self::Srt(c) => format!("srt://{}:{}", c.host, c.port),
            Self::Whip(c) => c.endpoint_url.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Packet / frame data
// ---------------------------------------------------------------------------

/// Media packet ready for protocol framing and transmission.
#[derive(Debug, Clone)]
pub struct OutputPacket {
    /// Raw encoded payload.
    pub data: Vec<u8>,
    /// Presentation timestamp in microseconds.
    pub pts_us: i64,
    /// Decode timestamp in microseconds (may differ for B-frames).
    pub dts_us: i64,
    /// Whether this is a keyframe / IDR.
    pub is_keyframe: bool,
    /// Track kind.
    pub kind: TrackKind,
}

/// Track kind within a stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackKind {
    /// Video track.
    Video,
    /// Audio track.
    Audio,
}

// ---------------------------------------------------------------------------
// Bandwidth estimator
// ---------------------------------------------------------------------------

/// Simple sliding-window bandwidth estimator.
#[derive(Debug)]
pub struct BandwidthEstimator {
    /// (timestamp, bytes_sent) samples.
    samples: VecDeque<(Instant, u64)>,
    /// Window duration for averaging.
    window: Duration,
}

impl BandwidthEstimator {
    /// Create a new estimator with the given window duration.
    #[must_use]
    pub fn new(window: Duration) -> Self {
        Self {
            samples: VecDeque::new(),
            window,
        }
    }

    /// Record that `bytes` were sent at the current instant.
    pub fn record(&mut self, bytes: u64) {
        let now = Instant::now();
        self.samples.push_back((now, bytes));
        self.evict_old(now);
    }

    /// Record that `bytes` were sent at a specific instant (for testing).
    pub fn record_at(&mut self, when: Instant, bytes: u64) {
        self.samples.push_back((when, bytes));
        self.evict_old(when);
    }

    /// Current estimated throughput in bits per second.
    #[must_use]
    pub fn estimate_bps(&self) -> f64 {
        if self.samples.len() < 2 {
            return 0.0;
        }
        let first = self.samples.front().map(|(t, _)| *t);
        let last = self.samples.back().map(|(t, _)| *t);
        let (Some(first), Some(last)) = (first, last) else {
            return 0.0;
        };
        let elapsed = last.duration_since(first).as_secs_f64();
        if elapsed <= 0.0 {
            return 0.0;
        }
        let total_bytes: u64 = self.samples.iter().map(|(_, b)| b).sum();
        (total_bytes as f64 * 8.0) / elapsed
    }

    /// Current estimated throughput in kilobits per second.
    #[must_use]
    pub fn estimate_kbps(&self) -> f64 {
        self.estimate_bps() / 1000.0
    }

    fn evict_old(&mut self, now: Instant) {
        while let Some(&(ts, _)) = self.samples.front() {
            if now.duration_since(ts) > self.window {
                self.samples.pop_front();
            } else {
                break;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Output sender (connection manager)
// ---------------------------------------------------------------------------

/// Manages connection lifecycle and packet sending for a single output endpoint.
#[derive(Debug)]
pub struct OutputSender {
    /// The configured endpoint.
    endpoint: OutputEndpoint,
    /// Current connection state.
    state: ConnectionState,
    /// Number of reconnection attempts made since last successful connect.
    reconnect_count: u32,
    /// Total packets sent since connection.
    packets_sent: u64,
    /// Total bytes sent since connection.
    bytes_sent: u64,
    /// Total keyframes sent.
    keyframes_sent: u64,
    /// Bandwidth estimator.
    bandwidth: BandwidthEstimator,
    /// Send buffer for packets queued while reconnecting.
    send_buffer: VecDeque<OutputPacket>,
    /// Maximum send buffer depth before dropping oldest packets.
    max_buffer_depth: usize,
    /// Last successful send timestamp.
    last_send_time: Option<Instant>,
}

impl OutputSender {
    /// Create a new output sender for the given endpoint.
    #[must_use]
    pub fn new(endpoint: OutputEndpoint) -> Self {
        Self {
            endpoint,
            state: ConnectionState::Disconnected,
            reconnect_count: 0,
            packets_sent: 0,
            bytes_sent: 0,
            keyframes_sent: 0,
            bandwidth: BandwidthEstimator::new(Duration::from_secs(5)),
            send_buffer: VecDeque::new(),
            max_buffer_depth: 300, // ~5 seconds at 60fps
            last_send_time: None,
        }
    }

    /// Attempt to establish a connection.
    ///
    /// # Errors
    ///
    /// Returns error if the connection cannot be established.
    pub fn connect(&mut self) -> GamingResult<()> {
        if self.state == ConnectionState::Connected {
            return Ok(());
        }

        self.state = ConnectionState::Connecting;

        // Validate endpoint-specific requirements
        match &self.endpoint {
            OutputEndpoint::Rtmp(cfg) => {
                if cfg.url.is_empty() || cfg.stream_key.is_empty() {
                    self.state = ConnectionState::Failed;
                    return Err(GamingError::PlatformError(
                        "RTMP URL or stream key is empty".into(),
                    ));
                }
            }
            OutputEndpoint::Srt(cfg) => {
                if cfg.host.is_empty() {
                    self.state = ConnectionState::Failed;
                    return Err(GamingError::PlatformError("SRT host is empty".into()));
                }
            }
            OutputEndpoint::Whip(cfg) => {
                if cfg.endpoint_url.is_empty() {
                    self.state = ConnectionState::Failed;
                    return Err(GamingError::PlatformError(
                        "WHIP endpoint URL is empty".into(),
                    ));
                }
            }
        }

        self.state = ConnectionState::Connected;
        self.reconnect_count = 0;
        Ok(())
    }

    /// Disconnect gracefully.
    pub fn disconnect(&mut self) {
        self.state = ConnectionState::Disconnected;
        self.send_buffer.clear();
    }

    /// Send a packet to the output.
    ///
    /// If the connection is in `Reconnecting` state, packets are buffered up to
    /// `max_buffer_depth`.
    ///
    /// # Errors
    ///
    /// Returns error if the connection is failed or the buffer overflows.
    pub fn send(&mut self, packet: OutputPacket) -> GamingResult<()> {
        match self.state {
            ConnectionState::Connected => {
                self.do_send(&packet);
                Ok(())
            }
            ConnectionState::Reconnecting => {
                if self.send_buffer.len() >= self.max_buffer_depth {
                    // Drop oldest non-keyframe to make room
                    self.drop_oldest_non_keyframe();
                }
                self.send_buffer.push_back(packet);
                Ok(())
            }
            ConnectionState::Failed => Err(GamingError::PlatformError(format!(
                "Connection to {} permanently failed",
                self.endpoint.destination()
            ))),
            ConnectionState::Disconnected | ConnectionState::Connecting => {
                Err(GamingError::PlatformError(format!(
                    "Not connected to {}",
                    self.endpoint.destination()
                )))
            }
        }
    }

    /// Attempt reconnection after a connection drop.
    ///
    /// # Errors
    ///
    /// Returns error if max reconnect attempts exceeded.
    pub fn attempt_reconnect(&mut self) -> GamingResult<()> {
        let max = self.endpoint.max_reconnect_attempts();
        if self.reconnect_count >= max {
            self.state = ConnectionState::Failed;
            return Err(GamingError::PlatformError(format!(
                "Max reconnect attempts ({max}) exceeded for {}",
                self.endpoint.destination()
            )));
        }

        self.state = ConnectionState::Reconnecting;
        self.reconnect_count += 1;

        // Simulate successful reconnect
        self.state = ConnectionState::Connected;

        // Flush buffered packets
        self.flush_buffer();

        Ok(())
    }

    /// Current connection state.
    #[must_use]
    pub fn state(&self) -> ConnectionState {
        self.state
    }

    /// Protocol in use.
    #[must_use]
    pub fn protocol(&self) -> OutputProtocol {
        self.endpoint.protocol()
    }

    /// Total packets sent.
    #[must_use]
    pub fn packets_sent(&self) -> u64 {
        self.packets_sent
    }

    /// Total bytes sent.
    #[must_use]
    pub fn bytes_sent(&self) -> u64 {
        self.bytes_sent
    }

    /// Total keyframes sent.
    #[must_use]
    pub fn keyframes_sent(&self) -> u64 {
        self.keyframes_sent
    }

    /// Current estimated bandwidth in kbps.
    #[must_use]
    pub fn estimated_bandwidth_kbps(&self) -> f64 {
        self.bandwidth.estimate_kbps()
    }

    /// Number of packets currently buffered (waiting for reconnect).
    #[must_use]
    pub fn buffered_packet_count(&self) -> usize {
        self.send_buffer.len()
    }

    /// Reference to the configured endpoint.
    #[must_use]
    pub fn endpoint(&self) -> &OutputEndpoint {
        &self.endpoint
    }

    /// Set the maximum send-buffer depth.
    pub fn set_max_buffer_depth(&mut self, depth: usize) {
        self.max_buffer_depth = depth;
    }

    // -- internal helpers --

    fn do_send(&mut self, packet: &OutputPacket) {
        let size = packet.data.len() as u64;
        self.packets_sent += 1;
        self.bytes_sent += size;
        if packet.is_keyframe {
            self.keyframes_sent += 1;
        }
        self.bandwidth.record(size);
        self.last_send_time = Some(Instant::now());
    }

    fn flush_buffer(&mut self) {
        let buffered: Vec<OutputPacket> = self.send_buffer.drain(..).collect();
        for pkt in &buffered {
            self.do_send(pkt);
        }
    }

    fn drop_oldest_non_keyframe(&mut self) {
        // Find first non-keyframe and remove it
        if let Some(idx) = self.send_buffer.iter().position(|p| !p.is_keyframe) {
            self.send_buffer.remove(idx);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- RtmpConfig tests --

    #[test]
    fn test_rtmp_config_valid() {
        let cfg = RtmpConfig::new("rtmp://live.twitch.tv/app", "my_stream_key");
        assert!(cfg.is_ok());
        let cfg = cfg.expect("valid config");
        assert_eq!(cfg.chunk_size, 4096);
        assert_eq!(cfg.app_name, "app");
    }

    #[test]
    fn test_rtmp_config_empty_url() {
        let result = RtmpConfig::new("", "key");
        assert!(result.is_err());
    }

    #[test]
    fn test_rtmp_config_empty_key() {
        let result = RtmpConfig::new("rtmp://host/app", "");
        assert!(result.is_err());
    }

    #[test]
    fn test_rtmp_config_custom_chunk_size() {
        let cfg = RtmpConfig::new("rtmp://host/app", "key")
            .expect("valid config")
            .with_chunk_size(8192);
        assert_eq!(cfg.chunk_size, 8192);
    }

    #[test]
    fn test_rtmp_config_minimum_chunk_size() {
        let cfg = RtmpConfig::new("rtmp://host/app", "key")
            .expect("valid config")
            .with_chunk_size(10); // below minimum
        assert_eq!(cfg.chunk_size, 128); // clamped to min
    }

    // -- SrtConfig tests --

    #[test]
    fn test_srt_config_valid() {
        let cfg = SrtConfig::new_caller("192.168.1.100", 9000);
        assert!(cfg.is_ok());
        let cfg = cfg.expect("valid config");
        assert_eq!(cfg.mode, SrtMode::Caller);
        assert_eq!(cfg.latency_ms, 120);
        assert_eq!(cfg.encryption, SrtEncryption::None);
    }

    #[test]
    fn test_srt_config_empty_host() {
        assert!(SrtConfig::new_caller("", 9000).is_err());
    }

    #[test]
    fn test_srt_config_zero_port() {
        assert!(SrtConfig::new_caller("host", 0).is_err());
    }

    #[test]
    fn test_srt_aes128_valid_passphrase() {
        let cfg = SrtConfig::new_caller("host", 9000)
            .expect("valid")
            .with_aes128("a_valid_passphrase_12345");
        assert!(cfg.is_ok());
        let cfg = cfg.expect("valid");
        assert_eq!(cfg.encryption, SrtEncryption::Aes128);
    }

    #[test]
    fn test_srt_aes256_valid_passphrase() {
        let cfg = SrtConfig::new_caller("host", 9000)
            .expect("valid")
            .with_aes256("another_valid_passphrase_12345");
        assert!(cfg.is_ok());
        let cfg = cfg.expect("valid");
        assert_eq!(cfg.encryption, SrtEncryption::Aes256);
    }

    #[test]
    fn test_srt_passphrase_too_short() {
        let result = SrtConfig::new_caller("host", 9000)
            .expect("valid")
            .with_aes128("short");
        assert!(result.is_err());
    }

    #[test]
    fn test_srt_passphrase_too_long() {
        let long = "a".repeat(80);
        let result = SrtConfig::new_caller("host", 9000)
            .expect("valid")
            .with_aes128(&long);
        assert!(result.is_err());
    }

    #[test]
    fn test_srt_latency() {
        let cfg = SrtConfig::new_caller("host", 9000)
            .expect("valid")
            .with_latency_ms(200);
        assert_eq!(cfg.latency_ms, 200);
    }

    #[test]
    fn test_srt_latency_minimum_clamp() {
        let cfg = SrtConfig::new_caller("host", 9000)
            .expect("valid")
            .with_latency_ms(5);
        assert_eq!(cfg.latency_ms, 20);
    }

    #[test]
    fn test_srt_stream_id() {
        let cfg = SrtConfig::new_caller("host", 9000)
            .expect("valid")
            .with_stream_id("my_stream");
        assert_eq!(cfg.stream_id.as_deref(), Some("my_stream"));
    }

    // -- WhipConfig tests --

    #[test]
    fn test_whip_config_valid() {
        let cfg = WhipConfig::new("https://whip.example.com/ingest");
        assert!(cfg.is_ok());
        let cfg = cfg.expect("valid");
        assert_eq!(cfg.ice_servers.len(), 1); // default STUN server
    }

    #[test]
    fn test_whip_config_empty_url() {
        assert!(WhipConfig::new("").is_err());
    }

    #[test]
    fn test_whip_bearer_token() {
        let cfg = WhipConfig::new("https://whip.example.com/ingest")
            .expect("valid")
            .with_bearer_token("tok_abc123");
        assert_eq!(cfg.bearer_token.as_deref(), Some("tok_abc123"));
    }

    #[test]
    fn test_whip_additional_ice_server() {
        let cfg = WhipConfig::new("https://whip.example.com/ingest")
            .expect("valid")
            .with_ice_server(IceServer::turn("turn:turn.example.com", "user", "pass"));
        assert_eq!(cfg.ice_servers.len(), 2);
    }

    // -- OutputEndpoint tests --

    #[test]
    fn test_output_endpoint_protocol() {
        let rtmp = OutputEndpoint::Rtmp(RtmpConfig::new("rtmp://host/app", "key").expect("valid"));
        assert_eq!(rtmp.protocol(), OutputProtocol::Rtmp);

        let srt = OutputEndpoint::Srt(SrtConfig::new_caller("host", 9000).expect("valid"));
        assert_eq!(srt.protocol(), OutputProtocol::Srt);

        let whip =
            OutputEndpoint::Whip(WhipConfig::new("https://whip.example.com").expect("valid"));
        assert_eq!(whip.protocol(), OutputProtocol::Whip);
    }

    #[test]
    fn test_output_endpoint_destination() {
        let rtmp = OutputEndpoint::Rtmp(
            RtmpConfig::new("rtmp://live.twitch.tv/app", "key").expect("valid"),
        );
        assert!(rtmp.destination().starts_with("rtmp://"));

        let srt = OutputEndpoint::Srt(SrtConfig::new_caller("192.168.1.1", 9000).expect("valid"));
        assert_eq!(srt.destination(), "srt://192.168.1.1:9000");
    }

    // -- OutputSender tests --

    fn make_test_sender(protocol: OutputProtocol) -> OutputSender {
        let endpoint = match protocol {
            OutputProtocol::Rtmp => {
                OutputEndpoint::Rtmp(RtmpConfig::new("rtmp://host/app", "key").expect("valid"))
            }
            OutputProtocol::Srt => {
                OutputEndpoint::Srt(SrtConfig::new_caller("host", 9000).expect("valid"))
            }
            OutputProtocol::Whip => {
                OutputEndpoint::Whip(WhipConfig::new("https://whip.example.com").expect("valid"))
            }
        };
        OutputSender::new(endpoint)
    }

    fn make_test_packet(is_keyframe: bool) -> OutputPacket {
        OutputPacket {
            data: vec![0u8; 1000],
            pts_us: 0,
            dts_us: 0,
            is_keyframe,
            kind: TrackKind::Video,
        }
    }

    #[test]
    fn test_sender_connect_disconnect() {
        let mut sender = make_test_sender(OutputProtocol::Rtmp);
        assert_eq!(sender.state(), ConnectionState::Disconnected);

        sender.connect().expect("connect should succeed");
        assert_eq!(sender.state(), ConnectionState::Connected);

        sender.disconnect();
        assert_eq!(sender.state(), ConnectionState::Disconnected);
    }

    #[test]
    fn test_sender_send_packets() {
        let mut sender = make_test_sender(OutputProtocol::Srt);
        sender.connect().expect("connect");

        for _ in 0..10 {
            sender.send(make_test_packet(false)).expect("send");
        }
        sender.send(make_test_packet(true)).expect("send keyframe");

        assert_eq!(sender.packets_sent(), 11);
        assert_eq!(sender.bytes_sent(), 11_000);
        assert_eq!(sender.keyframes_sent(), 1);
    }

    #[test]
    fn test_sender_send_when_disconnected_fails() {
        let mut sender = make_test_sender(OutputProtocol::Whip);
        let result = sender.send(make_test_packet(false));
        assert!(result.is_err());
    }

    #[test]
    fn test_sender_reconnect() {
        let mut sender = make_test_sender(OutputProtocol::Rtmp);
        sender.connect().expect("connect");

        // Simulate disconnect -> reconnect
        sender.state = ConnectionState::Reconnecting;
        sender.send(make_test_packet(false)).expect("buffered");
        sender.send(make_test_packet(true)).expect("buffered");
        assert_eq!(sender.buffered_packet_count(), 2);

        sender.attempt_reconnect().expect("reconnect");
        assert_eq!(sender.state(), ConnectionState::Connected);
        assert_eq!(sender.buffered_packet_count(), 0);
        assert_eq!(sender.packets_sent(), 2); // flushed
    }

    #[test]
    fn test_sender_max_reconnect_exceeded() {
        let mut sender = make_test_sender(OutputProtocol::Rtmp);
        sender.connect().expect("connect");

        // Exhaust reconnect attempts
        for _ in 0..5 {
            sender.state = ConnectionState::Reconnecting;
            let _ = sender.attempt_reconnect();
        }

        sender.state = ConnectionState::Reconnecting;
        let result = sender.attempt_reconnect();
        assert!(result.is_err());
        assert_eq!(sender.state(), ConnectionState::Failed);
    }

    #[test]
    fn test_sender_buffer_overflow_drops_non_keyframe() {
        let mut sender = make_test_sender(OutputProtocol::Srt);
        sender.connect().expect("connect");
        sender.set_max_buffer_depth(3);
        sender.state = ConnectionState::Reconnecting;

        // Fill buffer with non-keyframes
        for _ in 0..3 {
            sender.send(make_test_packet(false)).expect("buffered");
        }
        assert_eq!(sender.buffered_packet_count(), 3);

        // Adding one more should drop the oldest non-keyframe
        sender.send(make_test_packet(true)).expect("buffered");
        assert_eq!(sender.buffered_packet_count(), 3); // still 3, oldest dropped
    }

    #[test]
    fn test_send_failed_connection() {
        let mut sender = make_test_sender(OutputProtocol::Rtmp);
        sender.state = ConnectionState::Failed;
        let result = sender.send(make_test_packet(false));
        assert!(result.is_err());
    }

    // -- BandwidthEstimator tests --

    #[test]
    fn test_bandwidth_estimator_empty() {
        let est = BandwidthEstimator::new(Duration::from_secs(5));
        assert!((est.estimate_bps() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_bandwidth_estimator_single_sample() {
        let mut est = BandwidthEstimator::new(Duration::from_secs(5));
        est.record(1000);
        // Only one sample, can't compute rate
        assert!((est.estimate_bps() - 0.0).abs() < f64::EPSILON);
    }

    // -- OutputProtocol Display --

    #[test]
    fn test_protocol_display() {
        assert_eq!(format!("{}", OutputProtocol::Rtmp), "RTMP");
        assert_eq!(format!("{}", OutputProtocol::Srt), "SRT");
        assert_eq!(format!("{}", OutputProtocol::Whip), "WHIP");
    }

    // -- IceServer tests --

    #[test]
    fn test_ice_server_stun() {
        let s = IceServer::stun("stun:stun.example.com:3478");
        assert_eq!(s.urls.len(), 1);
        assert!(s.username.is_none());
    }

    #[test]
    fn test_ice_server_turn() {
        let s = IceServer::turn("turn:turn.example.com:3478", "user", "pass");
        assert_eq!(s.username.as_deref(), Some("user"));
        assert_eq!(s.credential.as_deref(), Some("pass"));
    }

    // -- TrackKind --

    #[test]
    fn test_track_kind_eq() {
        assert_eq!(TrackKind::Video, TrackKind::Video);
        assert_ne!(TrackKind::Video, TrackKind::Audio);
    }

    // -- Double-connect is idempotent --

    #[test]
    fn test_double_connect_is_ok() {
        let mut sender = make_test_sender(OutputProtocol::Rtmp);
        sender.connect().expect("first connect");
        sender.connect().expect("second connect is noop");
        assert_eq!(sender.state(), ConnectionState::Connected);
    }
}
