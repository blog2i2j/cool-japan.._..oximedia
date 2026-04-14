//! Live SRT ingest stub.
//!
//! Provides configuration, session tracking, and statistics for SRT (Secure
//! Reliable Transport) live ingest.  The actual SRT socket I/O is intentionally
//! stubbed out; this module focuses on the session-management layer that a full
//! implementation would sit above.

#![allow(dead_code)]

use std::collections::HashMap;

// ── Configuration ─────────────────────────────────────────────────────────────

/// Configuration for an SRT ingest listener.
#[derive(Debug, Clone)]
pub struct SrtIngestConfig {
    /// UDP port to listen on for incoming SRT connections.
    pub port: u16,
    /// Optional stream-ID passphrase for encryption.
    pub passphrase: Option<String>,
    /// Target latency in milliseconds (SRT `SRTO_LATENCY`).
    pub latency_ms: u32,
    /// Maximum receive bandwidth in Mbit/s (`SRTO_MAXBW`).
    pub max_bw_mbps: f32,
}

impl Default for SrtIngestConfig {
    fn default() -> Self {
        Self {
            port: 9998,
            passphrase: None,
            latency_ms: 200,
            max_bw_mbps: 100.0,
        }
    }
}

impl SrtIngestConfig {
    /// Returns `true` if the config has a non-empty passphrase.
    pub fn is_encrypted(&self) -> bool {
        self.passphrase
            .as_deref()
            .map(|p| !p.is_empty())
            .unwrap_or(false)
    }
}

// ── Session ───────────────────────────────────────────────────────────────────

/// An active (or historical) SRT ingest session.
#[derive(Debug, Clone)]
pub struct IngestSession {
    /// Unique session identifier.
    pub id: String,
    /// Remote client address (IP:port).
    pub client_addr: String,
    /// Unix epoch milliseconds when the session was opened.
    pub started_at_ms: u64,
    /// Total bytes received on this session.
    pub bytes_received: u64,
    /// Number of lost packets reported by SRT.
    pub packets_lost: u64,
}

impl IngestSession {
    /// Approximate number of received packets (assuming 1316-byte MPEG-TS payload).
    fn received_packets(&self) -> u64 {
        self.bytes_received / 1316
    }

    /// Duration of the session in milliseconds, given the current wall-clock.
    pub fn duration_ms(&self, now_ms: u64) -> u64 {
        now_ms.saturating_sub(self.started_at_ms)
    }

    /// Estimated throughput in kbit/s given the elapsed time.
    pub fn throughput_kbps(&self, now_ms: u64) -> f64 {
        let elapsed_ms = self.duration_ms(now_ms);
        if elapsed_ms == 0 {
            return 0.0;
        }
        (self.bytes_received as f64 * 8.0) / elapsed_ms as f64
    }
}

// ── LCG-based session ID generator ───────────────────────────────────────────

/// A minimal linear-congruential generator used to produce UUID-like session IDs
/// without external dependencies.
struct LcgIdGen {
    state: u64,
}

impl LcgIdGen {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Advances the LCG state and returns the next value.
    fn next(&mut self) -> u64 {
        // Parameters from Knuth's MMIX
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    /// Generates a UUID-like hex string with dashes.
    fn gen_id(&mut self, counter: u64) -> String {
        let a = self.next() ^ counter;
        let b = self.next() ^ counter.wrapping_add(1);
        // Format as xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx (version 4-ish)
        let p1 = (a >> 32) as u32;
        let p2 = ((a >> 16) & 0xFFFF) as u16;
        let p3 = (a & 0x0FFF) as u16 | 0x4000u16; // version 4
        let p4 = ((b >> 48) & 0x3FFF) as u16 | 0x8000u16; // variant bits
        let p5 = b & 0xFFFF_FFFF_FFFFu64;
        format!("{:08x}-{:04x}-{:04x}-{:04x}-{:012x}", p1, p2, p3, p4, p5)
    }
}

// ── SRT ingest server ─────────────────────────────────────────────────────────

/// Stub SRT ingest server that manages sessions without real socket I/O.
pub struct SrtIngestServer {
    /// Server configuration.
    pub config: SrtIngestConfig,
    /// Active and historical sessions keyed by session ID.
    sessions: HashMap<String, IngestSession>,
    /// ID generator.
    id_gen: LcgIdGen,
    /// Monotonically increasing session counter (used to seed the ID generator).
    session_counter: u64,
}

impl SrtIngestServer {
    /// Creates a new ingest server with the given configuration.
    pub fn new(config: SrtIngestConfig) -> Self {
        // Seed the LCG with the port number for deterministic but varied output.
        let seed = config.port as u64 ^ 0xDEAD_BEEF_1337_0042;
        Self {
            config,
            sessions: HashMap::new(),
            id_gen: LcgIdGen::new(seed),
            session_counter: 0,
        }
    }

    /// Simulates accepting a new SRT session from `client_addr` at `now_ms`.
    ///
    /// Returns the newly generated session ID.
    pub fn accept_session(&mut self, client_addr: String, now_ms: u64) -> String {
        self.session_counter = self.session_counter.wrapping_add(1);
        let id = self.id_gen.gen_id(self.session_counter);
        let session = IngestSession {
            id: id.clone(),
            client_addr,
            started_at_ms: now_ms,
            bytes_received: 0,
            packets_lost: 0,
        };
        self.sessions.insert(id.clone(), session);
        id
    }

    /// Updates statistics for the session identified by `id`.
    ///
    /// `bytes` is the number of additional bytes received since the last update.
    /// `lost` is the number of additional lost packets since the last update.
    pub fn update_session(&mut self, id: &str, bytes: u64, lost: u64) {
        if let Some(session) = self.sessions.get_mut(id) {
            session.bytes_received = session.bytes_received.saturating_add(bytes);
            session.packets_lost = session.packets_lost.saturating_add(lost);
        }
    }

    /// Returns a reference to the session with `id`, or `None` if not found.
    pub fn session_stats(&self, id: &str) -> Option<&IngestSession> {
        self.sessions.get(id)
    }

    /// Calculates the packet loss rate for a session.
    ///
    /// Returns `lost / (received_packets + lost)`.  Returns `0.0` if no
    /// packets have been seen or the session does not exist.
    pub fn packet_loss_rate(&self, id: &str) -> f32 {
        let session = match self.sessions.get(id) {
            Some(s) => s,
            None => return 0.0,
        };
        let received = session.received_packets();
        let total = received.saturating_add(session.packets_lost);
        if total == 0 {
            return 0.0;
        }
        session.packets_lost as f32 / total as f32
    }

    /// Removes a session by ID.  Returns `true` if it existed.
    pub fn remove_session(&mut self, id: &str) -> bool {
        self.sessions.remove(id).is_some()
    }

    /// Returns the number of active sessions.
    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }

    /// Returns the IDs of all sessions.
    pub fn session_ids(&self) -> Vec<&str> {
        self.sessions.keys().map(String::as_str).collect()
    }
}

// ── LiveIngestHandler ─────────────────────────────────────────────────────────

/// Simplified live ingest handler with RTMP connection management.
///
/// Wraps [`SrtIngestServer`] with an RTMP-oriented API matching the TODO spec:
///
/// ```rust
/// use oximedia_server::live_ingest::LiveIngestHandler;
///
/// let mut handler = LiveIngestHandler::new(1935);
/// let result = handler.handle_rtmp_connect("client-1", "live/stream");
/// assert!(result.is_ok());
/// ```
pub struct LiveIngestHandler {
    /// Listening port.
    pub port: u16,
    /// RTMP app name → session ID map.
    connections: HashMap<String, String>,
    /// Connection counter for ID generation.
    counter: u64,
}

impl LiveIngestHandler {
    /// Create a new RTMP ingest handler on the given port.
    #[must_use]
    pub fn new(port: u16) -> Self {
        Self {
            port,
            connections: HashMap::new(),
            counter: 0,
        }
    }

    /// Handle an RTMP client connecting to an application.
    ///
    /// * `client_id` – Unique identifier for the connecting client.
    /// * `app`       – RTMP application name (e.g. `"live/stream"`).
    ///
    /// Returns `Ok(())` on success, or `Err(String)` if the app name is empty
    /// or the connection slot is already occupied.
    pub fn handle_rtmp_connect(&mut self, client_id: &str, app: &str) -> Result<(), String> {
        if app.is_empty() {
            return Err("RTMP app name must not be empty".to_string());
        }
        if client_id.is_empty() {
            return Err("client_id must not be empty".to_string());
        }
        self.counter = self.counter.wrapping_add(1);
        let session_id = format!("rtmp-{}-{}-{}", self.port, self.counter, client_id);
        self.connections.insert(app.to_string(), session_id);
        Ok(())
    }

    /// Disconnect an RTMP client from an application.
    ///
    /// Returns `true` if a connection for `app` was found and removed.
    pub fn handle_rtmp_disconnect(&mut self, app: &str) -> bool {
        self.connections.remove(app).is_some()
    }

    /// Return the session ID for the given app, if connected.
    #[must_use]
    pub fn session_for_app(&self, app: &str) -> Option<&str> {
        self.connections.get(app).map(String::as_str)
    }

    /// Number of active RTMP connections.
    #[must_use]
    pub fn connection_count(&self) -> usize {
        self.connections.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_server() -> SrtIngestServer {
        SrtIngestServer::new(SrtIngestConfig::default())
    }

    // SrtIngestConfig tests

    #[test]
    fn test_config_default_values() {
        let cfg = SrtIngestConfig::default();
        assert_eq!(cfg.port, 9998);
        assert_eq!(cfg.latency_ms, 200);
        assert!(!cfg.is_encrypted());
    }

    #[test]
    fn test_config_with_passphrase() {
        let cfg = SrtIngestConfig {
            passphrase: Some("s3cr3t".to_string()),
            ..Default::default()
        };
        assert!(cfg.is_encrypted());
    }

    #[test]
    fn test_config_empty_passphrase_not_encrypted() {
        let cfg = SrtIngestConfig {
            passphrase: Some(String::new()),
            ..Default::default()
        };
        assert!(!cfg.is_encrypted());
    }

    // accept_session tests

    #[test]
    fn test_accept_session_returns_nonempty_id() {
        let mut srv = default_server();
        let id = srv.accept_session("192.168.1.1:9000".to_string(), 1_000_000);
        assert!(!id.is_empty());
        assert_eq!(srv.session_count(), 1);
    }

    #[test]
    fn test_accept_session_ids_are_unique() {
        let mut srv = default_server();
        let id1 = srv.accept_session("10.0.0.1:1234".to_string(), 0);
        let id2 = srv.accept_session("10.0.0.2:1234".to_string(), 0);
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_session_stats_found() {
        let mut srv = default_server();
        let id = srv.accept_session("127.0.0.1:5000".to_string(), 5_000);
        let stats = srv.session_stats(&id).expect("session should exist");
        assert_eq!(stats.client_addr, "127.0.0.1:5000");
        assert_eq!(stats.started_at_ms, 5_000);
        assert_eq!(stats.bytes_received, 0);
    }

    #[test]
    fn test_session_stats_not_found() {
        let srv = default_server();
        assert!(srv.session_stats("nonexistent").is_none());
    }

    // update_session tests

    #[test]
    fn test_update_session_accumulates_bytes() {
        let mut srv = default_server();
        let id = srv.accept_session("1.2.3.4:9000".to_string(), 0);
        srv.update_session(&id, 1024, 0);
        srv.update_session(&id, 2048, 0);
        let stats = srv.session_stats(&id).expect("should exist");
        assert_eq!(stats.bytes_received, 3072);
    }

    #[test]
    fn test_update_session_accumulates_lost() {
        let mut srv = default_server();
        let id = srv.accept_session("1.2.3.4:9000".to_string(), 0);
        srv.update_session(&id, 10 * 1316, 2);
        srv.update_session(&id, 10 * 1316, 3);
        let stats = srv.session_stats(&id).expect("should exist");
        assert_eq!(stats.packets_lost, 5);
    }

    #[test]
    fn test_update_nonexistent_session_is_noop() {
        let mut srv = default_server();
        // Should not panic
        srv.update_session("ghost", 9999, 9999);
        assert_eq!(srv.session_count(), 0);
    }

    // packet_loss_rate tests

    #[test]
    fn test_packet_loss_rate_zero_when_no_packets() {
        let mut srv = default_server();
        let id = srv.accept_session("1.2.3.4:9000".to_string(), 0);
        assert!((srv.packet_loss_rate(&id)).abs() < 1e-6);
    }

    #[test]
    fn test_packet_loss_rate_calculated_correctly() {
        let mut srv = default_server();
        let id = srv.accept_session("1.2.3.4:9000".to_string(), 0);
        // 9 * 1316 bytes received -> 9 received packets; 1 lost -> rate = 1/10 = 0.1
        srv.update_session(&id, 9 * 1316, 1);
        let rate = srv.packet_loss_rate(&id);
        let expected = 1.0_f32 / 10.0_f32;
        assert!(
            (rate - expected).abs() < 1e-5,
            "rate={} expected={}",
            rate,
            expected
        );
    }

    #[test]
    fn test_packet_loss_rate_nonexistent_returns_zero() {
        let srv = default_server();
        assert!((srv.packet_loss_rate("ghost")).abs() < 1e-6);
    }

    // remove_session tests

    #[test]
    fn test_remove_session() {
        let mut srv = default_server();
        let id = srv.accept_session("1.2.3.4:9000".to_string(), 0);
        assert!(srv.remove_session(&id));
        assert_eq!(srv.session_count(), 0);
    }

    #[test]
    fn test_remove_nonexistent_session() {
        let mut srv = default_server();
        assert!(!srv.remove_session("ghost"));
    }

    // IngestSession helpers

    #[test]
    fn test_session_duration_ms() {
        let session = IngestSession {
            id: "x".to_string(),
            client_addr: "127.0.0.1:1".to_string(),
            started_at_ms: 1000,
            bytes_received: 0,
            packets_lost: 0,
        };
        assert_eq!(session.duration_ms(1500), 500);
    }

    #[test]
    fn test_session_throughput_zero_elapsed() {
        let session = IngestSession {
            id: "x".to_string(),
            client_addr: "127.0.0.1:1".to_string(),
            started_at_ms: 1000,
            bytes_received: 10000,
            packets_lost: 0,
        };
        // Same timestamp as started_at -> elapsed = 0 -> 0 kbps
        assert!((session.throughput_kbps(1000)).abs() < 1e-9);
    }

    // ── LiveIngestHandler tests ───────────────────────────────────────────────

    #[test]
    fn test_rtmp_connect_ok() {
        let mut h = LiveIngestHandler::new(1935);
        assert!(h.handle_rtmp_connect("client-1", "live/stream").is_ok());
        assert_eq!(h.connection_count(), 1);
    }

    #[test]
    fn test_rtmp_connect_empty_app_errors() {
        let mut h = LiveIngestHandler::new(1935);
        assert!(h.handle_rtmp_connect("client-1", "").is_err());
    }

    #[test]
    fn test_rtmp_connect_empty_client_errors() {
        let mut h = LiveIngestHandler::new(1935);
        assert!(h.handle_rtmp_connect("", "live/stream").is_err());
    }

    #[test]
    fn test_rtmp_disconnect() {
        let mut h = LiveIngestHandler::new(1935);
        h.handle_rtmp_connect("c", "live/a").unwrap();
        assert!(h.handle_rtmp_disconnect("live/a"));
        assert_eq!(h.connection_count(), 0);
    }

    #[test]
    fn test_rtmp_session_for_app() {
        let mut h = LiveIngestHandler::new(1935);
        h.handle_rtmp_connect("c1", "live/b").unwrap();
        assert!(h.session_for_app("live/b").is_some());
        assert!(h.session_for_app("live/unknown").is_none());
    }

    // LcgIdGen tests (via accept_session)

    #[test]
    fn test_multiple_accepts_produce_valid_uuids() {
        let mut srv = default_server();
        for i in 0..10u64 {
            let id = srv.accept_session(format!("10.0.0.{}:9000", i), i * 100);
            // UUID-like: 8-4-4-4-12 hex chars separated by '-'
            let parts: Vec<&str> = id.split('-').collect();
            assert_eq!(parts.len(), 5, "id={}", id);
            assert_eq!(parts[0].len(), 8);
            assert_eq!(parts[1].len(), 4);
            assert_eq!(parts[2].len(), 4);
            assert_eq!(parts[3].len(), 4);
            assert_eq!(parts[4].len(), 12);
        }
    }
}
