#![allow(dead_code)]
//! Proxy streaming — stream proxies over the network without local download.
//!
//! This module models a proxy streaming system where clients can request byte
//! ranges of proxy media files from a streaming server.  It supports chunked
//! delivery, bandwidth throttling, and session management.
//!
//! The implementation is purely in-memory and does not perform real network I/O.
//! It provides the data structures and logic that a real HTTP/2 or QUIC-based
//! streaming server would use internally.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Streaming protocol
// ---------------------------------------------------------------------------

/// Supported streaming protocols.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StreamProtocol {
    /// HTTP/1.1 byte-range requests.
    Http1,
    /// HTTP/2 multiplexed streams.
    Http2,
    /// QUIC-based low-latency streaming.
    Quic,
}

impl StreamProtocol {
    /// Human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Http1 => "HTTP/1.1",
            Self::Http2 => "HTTP/2",
            Self::Quic => "QUIC",
        }
    }
}

// ---------------------------------------------------------------------------
// Byte range
// ---------------------------------------------------------------------------

/// A byte range within a file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ByteRange {
    /// Start offset (inclusive).
    pub start: u64,
    /// End offset (exclusive).
    pub end: u64,
}

impl ByteRange {
    /// Create a new byte range.
    pub fn new(start: u64, end: u64) -> Self {
        let (s, e) = if start <= end { (start, end) } else { (end, start) };
        Self { start: s, end: e }
    }

    /// Length of the range in bytes.
    pub fn len(&self) -> u64 {
        self.end.saturating_sub(self.start)
    }

    /// Whether the range is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether this range overlaps with another.
    pub fn overlaps(&self, other: &ByteRange) -> bool {
        self.start < other.end && other.start < self.end
    }

    /// Split this range into chunks of at most `chunk_size` bytes.
    pub fn chunks(&self, chunk_size: u64) -> Vec<ByteRange> {
        if chunk_size == 0 || self.is_empty() {
            return Vec::new();
        }
        let mut result = Vec::new();
        let mut offset = self.start;
        while offset < self.end {
            let end = (offset + chunk_size).min(self.end);
            result.push(ByteRange::new(offset, end));
            offset = end;
        }
        result
    }
}

// ---------------------------------------------------------------------------
// Streamable proxy
// ---------------------------------------------------------------------------

/// A proxy file registered for streaming.
#[derive(Debug, Clone)]
pub struct StreamableProxy {
    /// Unique proxy identifier.
    pub id: String,
    /// File path on the server.
    pub path: String,
    /// Total size in bytes.
    pub size_bytes: u64,
    /// MIME type (e.g. "video/mp4").
    pub mime_type: String,
    /// Recommended chunk size for streaming.
    pub chunk_size: u64,
    /// Number of times this proxy has been streamed.
    pub stream_count: u64,
    /// Total bytes served.
    pub bytes_served: u64,
}

impl StreamableProxy {
    /// Create a new streamable proxy.
    pub fn new(
        id: impl Into<String>,
        path: impl Into<String>,
        size_bytes: u64,
        mime_type: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            path: path.into(),
            size_bytes,
            mime_type: mime_type.into(),
            chunk_size: 1024 * 1024, // 1 MB default
            stream_count: 0,
            bytes_served: 0,
        }
    }

    /// Set the chunk size.
    pub fn with_chunk_size(mut self, size: u64) -> Self {
        self.chunk_size = size;
        self
    }

    /// Get the full-file byte range.
    pub fn full_range(&self) -> ByteRange {
        ByteRange::new(0, self.size_bytes)
    }

    /// Get chunks for the full file.
    pub fn file_chunks(&self) -> Vec<ByteRange> {
        self.full_range().chunks(self.chunk_size)
    }
}

// ---------------------------------------------------------------------------
// Streaming session
// ---------------------------------------------------------------------------

/// State of a streaming session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionState {
    /// Session is active and streaming.
    Active,
    /// Session is paused.
    Paused,
    /// Session completed (all requested data delivered).
    Completed,
    /// Session was aborted by the client.
    Aborted,
    /// Session timed out.
    TimedOut,
}

/// A streaming session for a single client.
#[derive(Debug, Clone)]
pub struct StreamSession {
    /// Session identifier.
    pub id: String,
    /// Proxy being streamed.
    pub proxy_id: String,
    /// Protocol in use.
    pub protocol: StreamProtocol,
    /// Bandwidth limit in bytes/second (0 = unlimited).
    pub bandwidth_limit_bps: u64,
    /// Bytes delivered so far.
    pub bytes_delivered: u64,
    /// Total bytes requested.
    pub bytes_requested: u64,
    /// Current session state.
    pub state: SessionState,
    /// Chunks delivered.
    pub chunks_delivered: u32,
    /// Total chunks expected.
    pub total_chunks: u32,
}

impl StreamSession {
    /// Create a new active session.
    pub fn new(
        id: impl Into<String>,
        proxy_id: impl Into<String>,
        protocol: StreamProtocol,
    ) -> Self {
        Self {
            id: id.into(),
            proxy_id: proxy_id.into(),
            protocol,
            bandwidth_limit_bps: 0,
            bytes_delivered: 0,
            bytes_requested: 0,
            state: SessionState::Active,
            chunks_delivered: 0,
            total_chunks: 0,
        }
    }

    /// Set bandwidth limit.
    pub fn with_bandwidth_limit(mut self, bps: u64) -> Self {
        self.bandwidth_limit_bps = bps;
        self
    }

    /// Record delivery of a chunk.
    #[allow(clippy::cast_possible_truncation)]
    pub fn deliver_chunk(&mut self, bytes: u64) {
        self.bytes_delivered += bytes;
        self.chunks_delivered += 1;
        if self.total_chunks > 0 && self.chunks_delivered >= self.total_chunks {
            self.state = SessionState::Completed;
        }
    }

    /// Progress as a percentage (0.0 to 100.0).
    #[allow(clippy::cast_precision_loss)]
    pub fn progress_percent(&self) -> f64 {
        if self.bytes_requested == 0 {
            return 0.0;
        }
        (self.bytes_delivered as f64 / self.bytes_requested as f64 * 100.0).min(100.0)
    }

    /// Estimated time remaining in seconds at current bandwidth limit.
    #[allow(clippy::cast_precision_loss)]
    pub fn estimated_remaining_secs(&self) -> f64 {
        if self.bandwidth_limit_bps == 0 {
            return 0.0; // unlimited = instant
        }
        let remaining = self.bytes_requested.saturating_sub(self.bytes_delivered);
        remaining as f64 / self.bandwidth_limit_bps as f64
    }

    /// Whether the session is in a terminal state.
    pub fn is_terminal(&self) -> bool {
        matches!(
            self.state,
            SessionState::Completed | SessionState::Aborted | SessionState::TimedOut
        )
    }
}

// ---------------------------------------------------------------------------
// Streaming server
// ---------------------------------------------------------------------------

/// In-memory proxy streaming server.
pub struct ProxyStreamingServer {
    /// Registered proxies.
    proxies: HashMap<String, StreamableProxy>,
    /// Active and completed sessions.
    sessions: HashMap<String, StreamSession>,
    /// Next session ID counter.
    next_session_id: u64,
    /// Total bytes served across all sessions.
    total_bytes_served: u64,
    /// Maximum concurrent sessions (0 = unlimited).
    max_concurrent: usize,
}

impl ProxyStreamingServer {
    /// Create a new streaming server.
    pub fn new() -> Self {
        Self {
            proxies: HashMap::new(),
            sessions: HashMap::new(),
            next_session_id: 1,
            total_bytes_served: 0,
            max_concurrent: 0,
        }
    }

    /// Set maximum concurrent sessions.
    pub fn with_max_concurrent(mut self, max: usize) -> Self {
        self.max_concurrent = max;
        self
    }

    /// Register a proxy for streaming.
    pub fn register_proxy(&mut self, proxy: StreamableProxy) {
        self.proxies.insert(proxy.id.clone(), proxy);
    }

    /// Remove a proxy (only if no active sessions reference it).
    pub fn remove_proxy(&mut self, id: &str) -> bool {
        let has_active = self
            .sessions
            .values()
            .any(|s| s.proxy_id == id && !s.is_terminal());
        if has_active {
            return false;
        }
        self.proxies.remove(id).is_some()
    }

    /// Start a new streaming session.
    ///
    /// Returns the session ID on success, or an error string on failure.
    #[allow(clippy::cast_possible_truncation)]
    pub fn start_session(
        &mut self,
        proxy_id: &str,
        protocol: StreamProtocol,
        range: Option<ByteRange>,
    ) -> Result<String, String> {
        // Check proxy exists
        let proxy = self
            .proxies
            .get(proxy_id)
            .ok_or_else(|| format!("proxy not found: {proxy_id}"))?;

        // Check concurrent limit
        if self.max_concurrent > 0 {
            let active = self
                .sessions
                .values()
                .filter(|s| !s.is_terminal())
                .count();
            if active >= self.max_concurrent {
                return Err("max concurrent sessions reached".to_string());
            }
        }

        let effective_range = range.unwrap_or_else(|| proxy.full_range());
        let chunk_count = proxy
            .full_range()
            .chunks(proxy.chunk_size)
            .len();

        let session_id = format!("sess_{}", self.next_session_id);
        self.next_session_id += 1;

        let mut session = StreamSession::new(&session_id, proxy_id, protocol);
        session.bytes_requested = effective_range.len();
        session.total_chunks = chunk_count as u32;

        self.sessions.insert(session_id.clone(), session);

        // Update proxy stats
        if let Some(p) = self.proxies.get_mut(proxy_id) {
            p.stream_count += 1;
        }

        Ok(session_id)
    }

    /// Deliver a chunk to a session.
    pub fn deliver_chunk(&mut self, session_id: &str, bytes: u64) -> bool {
        if let Some(session) = self.sessions.get_mut(session_id) {
            if session.is_terminal() {
                return false;
            }
            session.deliver_chunk(bytes);
            self.total_bytes_served += bytes;

            // Update proxy bytes served
            if let Some(p) = self.proxies.get_mut(&session.proxy_id) {
                p.bytes_served += bytes;
            }
            true
        } else {
            false
        }
    }

    /// Abort a session.
    pub fn abort_session(&mut self, session_id: &str) -> bool {
        if let Some(session) = self.sessions.get_mut(session_id) {
            if session.is_terminal() {
                return false;
            }
            session.state = SessionState::Aborted;
            true
        } else {
            false
        }
    }

    /// Pause a session.
    pub fn pause_session(&mut self, session_id: &str) -> bool {
        if let Some(session) = self.sessions.get_mut(session_id) {
            if session.state == SessionState::Active {
                session.state = SessionState::Paused;
                return true;
            }
        }
        false
    }

    /// Resume a paused session.
    pub fn resume_session(&mut self, session_id: &str) -> bool {
        if let Some(session) = self.sessions.get_mut(session_id) {
            if session.state == SessionState::Paused {
                session.state = SessionState::Active;
                return true;
            }
        }
        false
    }

    /// Get a session by ID.
    pub fn get_session(&self, id: &str) -> Option<&StreamSession> {
        self.sessions.get(id)
    }

    /// Get a proxy by ID.
    pub fn get_proxy(&self, id: &str) -> Option<&StreamableProxy> {
        self.proxies.get(id)
    }

    /// Number of active (non-terminal) sessions.
    pub fn active_session_count(&self) -> usize {
        self.sessions
            .values()
            .filter(|s| !s.is_terminal())
            .count()
    }

    /// Total bytes served.
    pub fn total_bytes_served(&self) -> u64 {
        self.total_bytes_served
    }

    /// Number of registered proxies.
    pub fn proxy_count(&self) -> usize {
        self.proxies.len()
    }
}

impl Default for ProxyStreamingServer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_proxy(id: &str, size: u64) -> StreamableProxy {
        StreamableProxy::new(id, format!("/proxy/{id}.mp4"), size, "video/mp4")
    }

    #[test]
    fn test_byte_range_basics() {
        let r = ByteRange::new(10, 20);
        assert_eq!(r.len(), 10);
        assert!(!r.is_empty());
    }

    #[test]
    fn test_byte_range_reversed() {
        let r = ByteRange::new(20, 10);
        assert_eq!(r.start, 10);
        assert_eq!(r.end, 20);
    }

    #[test]
    fn test_byte_range_empty() {
        let r = ByteRange::new(5, 5);
        assert!(r.is_empty());
    }

    #[test]
    fn test_byte_range_overlap() {
        let a = ByteRange::new(0, 10);
        let b = ByteRange::new(5, 15);
        let c = ByteRange::new(10, 20);
        assert!(a.overlaps(&b));
        assert!(!a.overlaps(&c));
    }

    #[test]
    fn test_byte_range_chunks() {
        let r = ByteRange::new(0, 100);
        let chunks = r.chunks(30);
        assert_eq!(chunks.len(), 4); // 0-30, 30-60, 60-90, 90-100
        assert_eq!(chunks[3].len(), 10);
    }

    #[test]
    fn test_byte_range_chunks_zero_size() {
        let r = ByteRange::new(0, 100);
        assert!(r.chunks(0).is_empty());
    }

    #[test]
    fn test_streamable_proxy() {
        let p = make_proxy("p1", 1024 * 1024 * 10); // 10 MB
        assert_eq!(p.size_bytes, 10_485_760);
        assert_eq!(p.chunk_size, 1024 * 1024);
        let chunks = p.file_chunks();
        assert_eq!(chunks.len(), 10);
    }

    #[test]
    fn test_session_creation() {
        let s = StreamSession::new("s1", "p1", StreamProtocol::Http2);
        assert_eq!(s.state, SessionState::Active);
        assert!(!s.is_terminal());
    }

    #[test]
    fn test_session_deliver_chunk() {
        let mut s = StreamSession::new("s1", "p1", StreamProtocol::Http1);
        s.bytes_requested = 1000;
        s.total_chunks = 2;
        s.deliver_chunk(500);
        assert_eq!(s.chunks_delivered, 1);
        assert_eq!(s.bytes_delivered, 500);
        assert!((s.progress_percent() - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_session_auto_complete() {
        let mut s = StreamSession::new("s1", "p1", StreamProtocol::Http2);
        s.bytes_requested = 200;
        s.total_chunks = 2;
        s.deliver_chunk(100);
        s.deliver_chunk(100);
        assert_eq!(s.state, SessionState::Completed);
        assert!(s.is_terminal());
    }

    #[test]
    fn test_session_estimated_remaining() {
        let mut s = StreamSession::new("s1", "p1", StreamProtocol::Quic);
        s.bytes_requested = 1000;
        s.bandwidth_limit_bps = 100;
        s.deliver_chunk(500);
        let remaining = s.estimated_remaining_secs();
        assert!((remaining - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_server_register_and_start_session() {
        let mut server = ProxyStreamingServer::new();
        server.register_proxy(make_proxy("p1", 10_000));

        let session_id = server
            .start_session("p1", StreamProtocol::Http2, None)
            .expect("should start");
        assert!(session_id.starts_with("sess_"));
        assert_eq!(server.active_session_count(), 1);
    }

    #[test]
    fn test_server_proxy_not_found() {
        let mut server = ProxyStreamingServer::new();
        let result = server.start_session("missing", StreamProtocol::Http1, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_server_max_concurrent() {
        let mut server = ProxyStreamingServer::new().with_max_concurrent(1);
        server.register_proxy(make_proxy("p1", 1000));

        let _ = server
            .start_session("p1", StreamProtocol::Http2, None)
            .expect("first ok");
        let result = server.start_session("p1", StreamProtocol::Http2, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_server_deliver_and_track() {
        let mut server = ProxyStreamingServer::new();
        server.register_proxy(make_proxy("p1", 10_000));
        let sid = server
            .start_session("p1", StreamProtocol::Http1, None)
            .expect("ok");

        assert!(server.deliver_chunk(&sid, 5000));
        assert_eq!(server.total_bytes_served(), 5000);

        let p = server.get_proxy("p1").expect("proxy should exist");
        assert_eq!(p.bytes_served, 5000);
    }

    #[test]
    fn test_server_abort_session() {
        let mut server = ProxyStreamingServer::new();
        server.register_proxy(make_proxy("p1", 1000));
        let sid = server
            .start_session("p1", StreamProtocol::Http2, None)
            .expect("ok");

        assert!(server.abort_session(&sid));
        let s = server.get_session(&sid).expect("session should exist");
        assert_eq!(s.state, SessionState::Aborted);
        assert!(s.is_terminal());
    }

    #[test]
    fn test_server_pause_resume() {
        let mut server = ProxyStreamingServer::new();
        server.register_proxy(make_proxy("p1", 1000));
        let sid = server
            .start_session("p1", StreamProtocol::Quic, None)
            .expect("ok");

        assert!(server.pause_session(&sid));
        assert_eq!(
            server.get_session(&sid).expect("s").state,
            SessionState::Paused
        );
        assert!(server.resume_session(&sid));
        assert_eq!(
            server.get_session(&sid).expect("s").state,
            SessionState::Active
        );
    }

    #[test]
    fn test_server_remove_proxy_no_active() {
        let mut server = ProxyStreamingServer::new();
        server.register_proxy(make_proxy("p1", 1000));
        assert!(server.remove_proxy("p1"));
        assert_eq!(server.proxy_count(), 0);
    }

    #[test]
    fn test_server_remove_proxy_with_active_session() {
        let mut server = ProxyStreamingServer::new();
        server.register_proxy(make_proxy("p1", 1000));
        let _ = server
            .start_session("p1", StreamProtocol::Http1, None)
            .expect("ok");
        assert!(!server.remove_proxy("p1")); // blocked by active session
    }

    #[test]
    fn test_protocol_name() {
        assert_eq!(StreamProtocol::Http1.name(), "HTTP/1.1");
        assert_eq!(StreamProtocol::Http2.name(), "HTTP/2");
        assert_eq!(StreamProtocol::Quic.name(), "QUIC");
    }

    #[test]
    fn test_default_server() {
        let server = ProxyStreamingServer::default();
        assert_eq!(server.proxy_count(), 0);
        assert_eq!(server.active_session_count(), 0);
    }
}
