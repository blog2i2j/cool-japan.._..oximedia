//! High-level SRT connection configuration and state machine.
//!
//! This module provides ergonomic builder types for the three SRT connection
//! modes (Caller, Listener, Rendezvous) and a lightweight connection state
//! machine that can be driven without an actual UDP socket.
//!
//! The low-level packet/socket layer is in [`crate::srt`]; this module
//! exposes a higher-level API suitable for configuration files, CLIs, and
//! integration tests.
//!
//! # Example
//!
//! ```
//! use oximedia_net::srt_config::{SrtConnectionConfig, SrtState};
//!
//! let mut conn = SrtConnectionConfig::caller("stream.example.com", 9000)
//!     .with_latency(120)
//!     .with_passphrase("s3cr3t")
//!     .build();
//!
//! assert_eq!(conn.state(), &SrtState::Disconnected);
//! conn.connect().expect("transitions to Connected");
//! assert_eq!(conn.state(), &SrtState::Connected);
//! conn.simulate_send(1316);
//! conn.disconnect();
//! assert_eq!(conn.state(), &SrtState::Closed);
//! ```

// ─── Connection Mode ──────────────────────────────────────────────────────────

/// SRT connection establishment mode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SrtConnectionMode {
    /// Caller: initiates connection to a known listener.
    Caller {
        /// Hostname or IP address of the listener.
        target_host: String,
        /// UDP port of the listener.
        target_port: u16,
    },
    /// Listener: binds a local port and waits for callers.
    Listener {
        /// Local UDP port to bind.
        bind_port: u16,
        /// Maximum number of queued incoming connections.
        backlog: u32,
    },
    /// Rendezvous: both peers connect to each other simultaneously.
    ///
    /// Enables NAT traversal — both sides punch through their NATs.
    Rendezvous {
        /// Local UDP port to bind.
        local_port: u16,
        /// Remote peer's hostname or IP address.
        remote_host: String,
        /// Remote peer's UDP port.
        remote_port: u16,
    },
}

impl SrtConnectionMode {
    /// Returns the human-readable mode name.
    #[must_use]
    pub fn mode_name(&self) -> &str {
        match self {
            Self::Caller { .. } => "caller",
            Self::Listener { .. } => "listener",
            Self::Rendezvous { .. } => "rendezvous",
        }
    }

    /// Returns `true` for modes that initiate an outbound connection.
    ///
    /// Both `Caller` and `Rendezvous` send the first packet; `Listener` waits.
    #[must_use]
    pub fn is_outbound(&self) -> bool {
        matches!(self, Self::Caller { .. } | Self::Rendezvous { .. })
    }

    /// Validate the mode parameters.
    ///
    /// # Errors
    ///
    /// - Caller: `target_port` must be > 0.
    /// - Listener: `bind_port` must be > 0, `backlog` must be > 0.
    /// - Rendezvous: `local_port` and `remote_port` must be > 0.
    pub fn validate(&self) -> Result<(), String> {
        match self {
            Self::Caller { target_port, .. } => {
                if *target_port == 0 {
                    return Err("Caller: target_port must be greater than 0".to_owned());
                }
            }
            Self::Listener { bind_port, backlog } => {
                if *bind_port == 0 {
                    return Err("Listener: bind_port must be greater than 0".to_owned());
                }
                if *backlog == 0 {
                    return Err("Listener: backlog must be greater than 0".to_owned());
                }
            }
            Self::Rendezvous {
                local_port,
                remote_port,
                ..
            } => {
                if *local_port == 0 {
                    return Err("Rendezvous: local_port must be greater than 0".to_owned());
                }
                if *remote_port == 0 {
                    return Err("Rendezvous: remote_port must be greater than 0".to_owned());
                }
            }
        }
        Ok(())
    }
}

// ─── SrtConnectionConfig ─────────────────────────────────────────────────────

/// High-level SRT connection configuration.
///
/// Use the constructor methods [`SrtConnectionConfig::caller`],
/// [`SrtConnectionConfig::listener`], or [`SrtConnectionConfig::rendezvous`]
/// as starting points, then chain builder methods.
#[derive(Debug, Clone)]
pub struct SrtConnectionConfig {
    /// Connection establishment mode.
    pub mode: SrtConnectionMode,
    /// SRT receive/send latency buffer in milliseconds (20–8000 ms).
    pub latency_ms: u32,
    /// Optional maximum bandwidth limit in bits per second.
    pub max_bandwidth_bps: Option<u64>,
    /// Optional AES passphrase for payload encryption.
    pub passphrase: Option<String>,
    /// Optional stream ID for connection multiplexing.
    pub stream_id: Option<String>,
    /// TCP-style connect timeout in milliseconds.
    pub connect_timeout_ms: u32,
}

impl SrtConnectionConfig {
    fn new_with_mode(mode: SrtConnectionMode) -> Self {
        Self {
            mode,
            latency_ms: 120,
            max_bandwidth_bps: None,
            passphrase: None,
            stream_id: None,
            connect_timeout_ms: 3000,
        }
    }

    /// Create a Caller configuration targeting the given host and port.
    #[must_use]
    pub fn caller(host: impl Into<String>, port: u16) -> Self {
        Self::new_with_mode(SrtConnectionMode::Caller {
            target_host: host.into(),
            target_port: port,
        })
    }

    /// Create a Listener configuration bound to `port` with a backlog of 5.
    #[must_use]
    pub fn listener(port: u16) -> Self {
        Self::new_with_mode(SrtConnectionMode::Listener {
            bind_port: port,
            backlog: 5,
        })
    }

    /// Create a Rendezvous configuration.
    #[must_use]
    pub fn rendezvous(local_port: u16, remote_host: impl Into<String>, remote_port: u16) -> Self {
        Self::new_with_mode(SrtConnectionMode::Rendezvous {
            local_port,
            remote_host: remote_host.into(),
            remote_port,
        })
    }

    /// Override the SRT latency buffer.
    #[must_use]
    pub fn with_latency(mut self, ms: u32) -> Self {
        self.latency_ms = ms;
        self
    }

    /// Set an encryption passphrase.
    #[must_use]
    pub fn with_passphrase(mut self, pass: impl Into<String>) -> Self {
        self.passphrase = Some(pass.into());
        self
    }

    /// Set a stream ID for server-side multiplexing.
    #[must_use]
    pub fn with_stream_id(mut self, id: impl Into<String>) -> Self {
        self.stream_id = Some(id.into());
        self
    }

    /// Set a maximum bandwidth cap.
    #[must_use]
    pub fn with_max_bandwidth(mut self, bps: u64) -> Self {
        self.max_bandwidth_bps = Some(bps);
        self
    }

    /// Build an [`SrtConnection`] from this configuration.
    #[must_use]
    pub fn build(self) -> SrtConnection {
        SrtConnection::new(self)
    }

    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`SrtConnectionMode::validate`].
    pub fn validate(&self) -> Result<(), String> {
        self.mode.validate()
    }
}

// ─── SrtState ─────────────────────────────────────────────────────────────────

/// State machine states for an SRT connection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SrtState {
    /// Initial state; no handshake has been started.
    Disconnected,
    /// Handshake is in progress (SYN/induction/conclusion exchange).
    Connecting,
    /// Connection is established; data transfer is possible.
    Connected,
    /// The connection was interrupted (packet loss > threshold, timeout, etc.).
    Broken,
    /// The connection was cleanly closed.
    Closed,
}

// ─── SrtConnection ────────────────────────────────────────────────────────────

/// Lightweight SRT connection state machine.
///
/// Simulates connection lifecycle and tracks traffic statistics without
/// requiring an actual UDP socket.  Useful for unit testing and protocol
/// logic validation.
pub struct SrtConnection {
    config: SrtConnectionConfig,
    state: SrtState,
    bytes_sent: u64,
    bytes_received: u64,
    packets_lost: u64,
}

impl SrtConnection {
    /// Create a new connection in [`SrtState::Disconnected`].
    #[must_use]
    pub fn new(config: SrtConnectionConfig) -> Self {
        Self {
            config,
            state: SrtState::Disconnected,
            bytes_sent: 0,
            bytes_received: 0,
            packets_lost: 0,
        }
    }

    /// Attempt to connect.
    ///
    /// Transitions: `Disconnected` → `Connecting` → `Connected`.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the configuration is invalid or if the current state
    /// does not permit a new connection attempt (already `Connected` or `Closed`).
    pub fn connect(&mut self) -> Result<(), String> {
        match self.state {
            SrtState::Disconnected | SrtState::Broken => {}
            SrtState::Connecting => {
                return Err("already connecting".to_owned());
            }
            SrtState::Connected => {
                return Err("already connected".to_owned());
            }
            SrtState::Closed => {
                return Err("connection is closed; create a new SrtConnection".to_owned());
            }
        }

        self.config.validate()?;
        self.state = SrtState::Connecting;
        // Simulate successful handshake completion (no real I/O).
        self.state = SrtState::Connected;
        Ok(())
    }

    /// Cleanly close the connection.
    ///
    /// Transitions to [`SrtState::Closed`] from any state.
    pub fn disconnect(&mut self) {
        self.state = SrtState::Closed;
    }

    /// Returns a reference to the current connection state.
    #[must_use]
    pub fn state(&self) -> &SrtState {
        &self.state
    }

    /// Returns a reference to the configuration.
    #[must_use]
    pub fn config(&self) -> &SrtConnectionConfig {
        &self.config
    }

    /// Simulate sending `bytes` over the connection, updating statistics.
    ///
    /// Does not require the connection to be in `Connected` state so that
    /// unit tests can drive statistics independently of state.
    pub fn simulate_send(&mut self, bytes: u64) {
        self.bytes_sent = self.bytes_sent.saturating_add(bytes);
    }

    /// Simulate receiving `bytes` over the connection, updating statistics.
    pub fn simulate_receive(&mut self, bytes: u64) {
        self.bytes_received = self.bytes_received.saturating_add(bytes);
    }

    /// Simulate a packet loss event, incrementing the lost-packet counter.
    pub fn simulate_loss(&mut self, packets: u64) {
        self.packets_lost = self.packets_lost.saturating_add(packets);
    }

    /// Returns the total bytes sent.
    #[must_use]
    pub fn bytes_sent(&self) -> u64 {
        self.bytes_sent
    }

    /// Returns the total bytes received.
    #[must_use]
    pub fn bytes_received(&self) -> u64 {
        self.bytes_received
    }

    /// Packet loss rate: `packets_lost / (packets_sent + packets_received)`.
    ///
    /// Treats every 1316 bytes as one packet (SRT default MTU).
    /// Returns `0.0` if no traffic has occurred.
    #[must_use]
    pub fn packet_loss_rate(&self) -> f64 {
        const MTU: u64 = 1316;
        let sent_pkts = self.bytes_sent.saturating_add(MTU - 1) / MTU;
        let recv_pkts = self.bytes_received.saturating_add(MTU - 1) / MTU;
        let total = sent_pkts + recv_pkts;
        if total == 0 {
            return 0.0;
        }
        self.packets_lost as f64 / total as f64
    }
}

// ─── SrtStreamConfig ─────────────────────────────────────────────────────────

/// High-level stream-level configuration for an SRT connection.
///
/// Wraps [`SrtConnectionConfig`] with convenience constructors focused on
/// the transport stream parameters (latency, encryption, stream ID).
///
/// # Example
///
/// ```
/// use oximedia_net::srt_config::{SrtConnectionMode, SrtStreamConfig};
///
/// let mode = SrtConnectionMode::Caller {
///     target_host: "ingest.example.com".to_owned(),
///     target_port: 9000,
/// };
/// let cfg = SrtStreamConfig::with_mode(mode)
///     .with_latency(200)
///     .with_passphrase("s3cr3t");
/// let url = cfg.connection_string();
/// assert!(url.starts_with("srt://"));
/// assert!(url.contains("mode=caller"));
/// assert!(url.contains("latency=200"));
/// ```
#[derive(Debug, Clone)]
pub struct SrtStreamConfig {
    /// Underlying low-level connection config.
    inner: SrtConnectionConfig,
}

impl SrtStreamConfig {
    /// Create a stream config from an explicit [`SrtConnectionMode`].
    #[must_use]
    pub fn with_mode(mode: SrtConnectionMode) -> Self {
        Self {
            inner: SrtConnectionConfig::new_with_mode(mode),
        }
    }

    /// Override the latency in milliseconds.
    #[must_use]
    pub fn with_latency(mut self, ms: u32) -> Self {
        self.inner.latency_ms = ms;
        self
    }

    /// Set an encryption passphrase.
    #[must_use]
    pub fn with_passphrase(mut self, pass: impl Into<String>) -> Self {
        self.inner.passphrase = Some(pass.into());
        self
    }

    /// Set a stream ID for server-side multiplexing.
    #[must_use]
    pub fn with_stream_id(mut self, id: impl Into<String>) -> Self {
        self.inner.stream_id = Some(id.into());
        self
    }

    /// Returns a reference to the underlying [`SrtConnectionConfig`].
    #[must_use]
    pub fn config(&self) -> &SrtConnectionConfig {
        &self.inner
    }

    /// Build the underlying [`SrtConnection`].
    #[must_use]
    pub fn build(self) -> SrtConnection {
        self.inner.build()
    }

    /// Format the connection parameters as an SRT URL.
    ///
    /// Delegates to the module-level [`connection_string`] function.
    #[must_use]
    pub fn connection_string(&self) -> String {
        connection_string(&self.inner)
    }
}

// ─── connection_string helper ─────────────────────────────────────────────────

/// Format an [`SrtConnectionConfig`] as an SRT URL string.
///
/// The URL format follows the SRT URI conventions:
///
/// ```text
/// srt://host:port?mode=caller&latency=N[&passphrase=...][&streamid=...]
/// ```
///
/// - **Caller**: `srt://target_host:target_port?mode=caller&latency=N`
/// - **Listener**: `srt://0.0.0.0:port?mode=listener&latency=N`
/// - **Rendezvous**: `srt://remote_host:remote_port?mode=rendezvous&latency=N`
///
/// The `passphrase` query parameter is included only when the config sets one.
/// The `streamid` query parameter is included only when set.
#[must_use]
pub fn connection_string(config: &SrtConnectionConfig) -> String {
    let (host, port, mode_name) = match &config.mode {
        SrtConnectionMode::Caller {
            target_host,
            target_port,
        } => (target_host.as_str(), *target_port, "caller"),
        SrtConnectionMode::Listener { bind_port, .. } => ("0.0.0.0", *bind_port, "listener"),
        SrtConnectionMode::Rendezvous {
            remote_host,
            remote_port,
            ..
        } => (remote_host.as_str(), *remote_port, "rendezvous"),
    };

    let mut url = format!(
        "srt://{host}:{port}?mode={mode_name}&latency={}",
        config.latency_ms
    );

    if let Some(ref pass) = config.passphrase {
        url.push_str(&format!("&passphrase={pass}"));
    }
    if let Some(ref sid) = config.stream_id {
        url.push_str(&format!("&streamid={sid}"));
    }

    url
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // 1. mode_name correct for each variant
    #[test]
    fn test_mode_name_caller() {
        let m = SrtConnectionMode::Caller {
            target_host: "host".to_owned(),
            target_port: 9000,
        };
        assert_eq!(m.mode_name(), "caller");
    }

    #[test]
    fn test_mode_name_listener() {
        let m = SrtConnectionMode::Listener {
            bind_port: 9000,
            backlog: 5,
        };
        assert_eq!(m.mode_name(), "listener");
    }

    #[test]
    fn test_mode_name_rendezvous() {
        let m = SrtConnectionMode::Rendezvous {
            local_port: 9000,
            remote_host: "peer".to_owned(),
            remote_port: 9001,
        };
        assert_eq!(m.mode_name(), "rendezvous");
    }

    // 2. validate catches zero port
    #[test]
    fn test_validate_caller_zero_port() {
        let m = SrtConnectionMode::Caller {
            target_host: "host".to_owned(),
            target_port: 0,
        };
        assert!(m.validate().is_err());
    }

    #[test]
    fn test_validate_listener_zero_port() {
        let m = SrtConnectionMode::Listener {
            bind_port: 0,
            backlog: 5,
        };
        assert!(m.validate().is_err());
    }

    #[test]
    fn test_validate_listener_zero_backlog() {
        let m = SrtConnectionMode::Listener {
            bind_port: 9000,
            backlog: 0,
        };
        assert!(m.validate().is_err());
    }

    #[test]
    fn test_validate_rendezvous_zero_local() {
        let m = SrtConnectionMode::Rendezvous {
            local_port: 0,
            remote_host: "h".to_owned(),
            remote_port: 9001,
        };
        assert!(m.validate().is_err());
    }

    // 3. caller.is_outbound = true
    #[test]
    fn test_caller_is_outbound() {
        let m = SrtConnectionMode::Caller {
            target_host: "h".to_owned(),
            target_port: 9000,
        };
        assert!(m.is_outbound());
    }

    // 4. listener.is_outbound = false
    #[test]
    fn test_listener_is_not_outbound() {
        let m = SrtConnectionMode::Listener {
            bind_port: 9000,
            backlog: 5,
        };
        assert!(!m.is_outbound());
    }

    // 5. rendezvous.is_outbound = true
    #[test]
    fn test_rendezvous_is_outbound() {
        let m = SrtConnectionMode::Rendezvous {
            local_port: 9000,
            remote_host: "h".to_owned(),
            remote_port: 9001,
        };
        assert!(m.is_outbound());
    }

    // 6. SrtConnection starts Disconnected
    #[test]
    fn test_connection_starts_disconnected() {
        let conn = SrtConnectionConfig::caller("host", 9000).build();
        assert_eq!(conn.state(), &SrtState::Disconnected);
    }

    // 7. connect transitions to Connected
    #[test]
    fn test_connect_transitions() {
        let mut conn = SrtConnectionConfig::caller("host", 9000).build();
        conn.connect().expect("connect should succeed");
        assert_eq!(conn.state(), &SrtState::Connected);
    }

    // 8. disconnect transitions to Closed
    #[test]
    fn test_disconnect() {
        let mut conn = SrtConnectionConfig::caller("host", 9000).build();
        conn.connect().expect("connect");
        conn.disconnect();
        assert_eq!(conn.state(), &SrtState::Closed);
    }

    // 9. simulate_send increments bytes_sent
    #[test]
    fn test_simulate_send() {
        let mut conn = SrtConnectionConfig::caller("host", 9000).build();
        conn.simulate_send(1316);
        conn.simulate_send(1316);
        assert_eq!(conn.bytes_sent(), 2632);
    }

    // 10. simulate_receive increments bytes_received
    #[test]
    fn test_simulate_receive() {
        let mut conn = SrtConnectionConfig::listener(9000).build();
        conn.simulate_receive(4096);
        assert_eq!(conn.bytes_received(), 4096);
    }

    // 11. packet_loss_rate is 0 with no traffic
    #[test]
    fn test_packet_loss_rate_no_traffic() {
        let conn = SrtConnectionConfig::caller("host", 9000).build();
        assert!((conn.packet_loss_rate() - 0.0).abs() < f64::EPSILON);
    }

    // 12. packet_loss_rate computed correctly
    #[test]
    fn test_packet_loss_rate() {
        let mut conn = SrtConnectionConfig::rendezvous(9000, "peer", 9001).build();
        conn.simulate_send(1316); // 1 packet
        conn.simulate_loss(1);
        let rate = conn.packet_loss_rate();
        // 1 lost / 1 sent = 1.0
        assert!((rate - 1.0).abs() < 1e-9);
    }

    // 13. builder with_latency
    #[test]
    fn test_with_latency() {
        let cfg = SrtConnectionConfig::caller("host", 9000).with_latency(200);
        assert_eq!(cfg.latency_ms, 200);
    }

    // 14. builder with_passphrase
    #[test]
    fn test_with_passphrase() {
        let cfg = SrtConnectionConfig::caller("host", 9000).with_passphrase("secret");
        assert_eq!(cfg.passphrase.as_deref(), Some("secret"));
    }

    // 15. connect on closed returns error
    #[test]
    fn test_connect_on_closed_errors() {
        let mut conn = SrtConnectionConfig::caller("host", 9000).build();
        conn.disconnect();
        let result = conn.connect();
        assert!(result.is_err());
    }

    // 16. connection_string for Caller mode
    #[test]
    fn test_connection_string_caller() {
        let cfg = SrtConnectionConfig::caller("stream.example.com", 9000).with_latency(120);
        let url = connection_string(&cfg);
        assert!(url.starts_with("srt://stream.example.com:9000"));
        assert!(url.contains("mode=caller"));
        assert!(url.contains("latency=120"));
    }

    // 17. connection_string for Listener mode
    #[test]
    fn test_connection_string_listener() {
        let cfg = SrtConnectionConfig::listener(9000).with_latency(80);
        let url = connection_string(&cfg);
        assert!(url.starts_with("srt://0.0.0.0:9000"));
        assert!(url.contains("mode=listener"));
        assert!(url.contains("latency=80"));
    }

    // 18. connection_string for Rendezvous mode
    #[test]
    fn test_connection_string_rendezvous() {
        let cfg = SrtConnectionConfig::rendezvous(9000, "peer.example.com", 9001).with_latency(200);
        let url = connection_string(&cfg);
        assert!(url.contains("mode=rendezvous"));
        assert!(url.contains("peer.example.com:9001"));
    }

    // 19. connection_string includes passphrase when set
    #[test]
    fn test_connection_string_passphrase() {
        let cfg = SrtConnectionConfig::caller("host", 9000).with_passphrase("secret123");
        let url = connection_string(&cfg);
        assert!(url.contains("passphrase=secret123"));
    }

    // 20. connection_string includes streamid when set
    #[test]
    fn test_connection_string_stream_id() {
        let cfg = SrtConnectionConfig::caller("host", 9000)
            .with_stream_id("#!::r=live/stream1,m=publish");
        let url = connection_string(&cfg);
        assert!(url.contains("streamid="));
    }

    // 21. SrtStreamConfig::with_mode caller creates correct URL
    #[test]
    fn test_srt_stream_config_with_mode_caller() {
        let mode = SrtConnectionMode::Caller {
            target_host: "ingest.example.com".to_owned(),
            target_port: 5000,
        };
        let cfg = SrtStreamConfig::with_mode(mode).with_latency(150);
        let url = cfg.connection_string();
        assert!(url.contains("ingest.example.com:5000"));
        assert!(url.contains("mode=caller"));
        assert!(url.contains("latency=150"));
    }

    // 22. SrtStreamConfig builds an SrtConnection
    #[test]
    fn test_srt_stream_config_build() {
        let mode = SrtConnectionMode::Listener {
            bind_port: 9000,
            backlog: 5,
        };
        let conn = SrtStreamConfig::with_mode(mode).build();
        assert_eq!(conn.state(), &SrtState::Disconnected);
    }

    // 23. connection_string no passphrase or streamid when not set
    #[test]
    fn test_connection_string_no_extras() {
        let cfg = SrtConnectionConfig::caller("host", 9000);
        let url = connection_string(&cfg);
        assert!(!url.contains("passphrase="));
        assert!(!url.contains("streamid="));
    }
}
