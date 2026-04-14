//! RTSP/RTP source serving for compatibility with standard media players.
//!
//! Implements a subset of RTSP 1.0 (RFC 2326) sufficient to serve live video
//! and audio streams to VLC, ffplay, and other standard RTSP clients.
//!
//! # Supported methods
//!
//! - `OPTIONS` — advertise supported methods
//! - `DESCRIBE` — return SDP description of available media
//! - `SETUP` — negotiate transport parameters (UDP unicast or multicast)
//! - `PLAY` — start media delivery
//! - `TEARDOWN` — end the session
//!
//! # Architecture
//!
//! ```text
//! RtspServer
//!   ├── RtspSession (per connected client)
//!   │     ├── state machine: Init → Ready → Playing → Teardown
//!   │     └── RtpTransportConfig (client-chosen port range)
//!   └── MediaSource (shared, caller-provided)
//! ```

#![allow(dead_code)]

use std::collections::HashMap;

/// RTSP session identifier (opaque string token returned to the client).
pub type RtspSessionId = String;

/// RTSP response status codes (RFC 2326 §7.1.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum RtspStatus {
    /// 200 OK
    Ok = 200,
    /// 400 Bad Request
    BadRequest = 400,
    /// 404 Not Found
    NotFound = 404,
    /// 405 Method Not Allowed
    MethodNotAllowed = 405,
    /// 454 Session Not Found
    SessionNotFound = 454,
    /// 455 Method Not Valid In This State
    MethodNotValidInState = 455,
    /// 500 Internal Server Error
    InternalError = 500,
    /// 501 Not Implemented
    NotImplemented = 501,
}

impl RtspStatus {
    /// Returns the standard reason phrase for this status code.
    #[must_use]
    pub fn reason_phrase(self) -> &'static str {
        match self {
            Self::Ok => "OK",
            Self::BadRequest => "Bad Request",
            Self::NotFound => "Not Found",
            Self::MethodNotAllowed => "Method Not Allowed",
            Self::SessionNotFound => "Session Not Found",
            Self::MethodNotValidInState => "Method Not Valid In This State",
            Self::InternalError => "Internal Server Error",
            Self::NotImplemented => "Not Implemented",
        }
    }
}

/// RTSP transport type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RtspTransport {
    /// Unicast UDP — client-selected port pair (RTP/RTCP).
    UdpUnicast {
        /// Client RTP port.
        client_rtp_port: u16,
        /// Client RTCP port.
        client_rtcp_port: u16,
        /// Server-assigned RTP source port.
        server_rtp_port: u16,
        /// Server-assigned RTCP source port.
        server_rtcp_port: u16,
    },
    /// Multicast UDP — server-chosen group address and port.
    UdpMulticast {
        /// Multicast group address.
        group_addr: String,
        /// RTP port.
        rtp_port: u16,
        /// Time-to-live.
        ttl: u8,
    },
    /// TCP interleaved (RTP over RTSP control connection).
    TcpInterleaved {
        /// Interleave channel for RTP.
        rtp_channel: u8,
        /// Interleave channel for RTCP.
        rtcp_channel: u8,
    },
}

/// State of an RTSP session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RtspSessionState {
    /// Freshly created; no `SETUP` received yet.
    Init,
    /// `SETUP` received; `PLAY` not yet received.
    Ready,
    /// `PLAY` received; media is flowing.
    Playing,
    /// `TEARDOWN` received; session is being closed.
    TearingDown,
}

/// An individual RTSP client session.
#[derive(Debug, Clone)]
pub struct RtspSession {
    /// Session identifier sent to the client in every response.
    pub id: RtspSessionId,
    /// Current session state.
    pub state: RtspSessionState,
    /// Negotiated transport parameters.
    pub transport: Option<RtspTransport>,
    /// The RTSP URL path this session is tracking.
    pub url_path: String,
    /// RTSP `CSeq` of the last request.
    pub last_cseq: u32,
    /// Session creation time (ms).
    pub created_at_ms: u64,
    /// Number of RTP packets delivered since `PLAY`.
    pub packets_sent: u64,
    /// Number of bytes delivered since `PLAY`.
    pub bytes_sent: u64,
}

impl RtspSession {
    /// Creates a new session in the `Init` state.
    #[must_use]
    pub fn new(id: RtspSessionId, url_path: String, created_at_ms: u64) -> Self {
        Self {
            id,
            state: RtspSessionState::Init,
            transport: None,
            url_path,
            last_cseq: 0,
            created_at_ms,
            packets_sent: 0,
            bytes_sent: 0,
        }
    }

    /// Returns `true` if media is currently flowing.
    #[must_use]
    pub fn is_playing(&self) -> bool {
        self.state == RtspSessionState::Playing
    }
}

/// Error type for RTSP server operations.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum RtspError {
    /// Session not found.
    #[error("RTSP session '{0}' not found")]
    SessionNotFound(String),
    /// Invalid method for current state.
    #[error("method not valid in state {state:?}: {method}")]
    InvalidState {
        /// Current state.
        state: RtspSessionState,
        /// The RTSP method attempted.
        method: String,
    },
    /// Requested resource not found.
    #[error("resource not found: {0}")]
    ResourceNotFound(String),
    /// Transport negotiation failed.
    #[error("transport error: {0}")]
    TransportError(String),
}

/// Result type for RTSP server operations.
pub type RtspResult<T> = Result<T, RtspError>;

/// An RTSP response message.
#[derive(Debug, Clone)]
pub struct RtspResponse {
    /// HTTP-style status code.
    pub status: RtspStatus,
    /// CSeq mirrored from the request.
    pub cseq: u32,
    /// Optional session identifier.
    pub session_id: Option<String>,
    /// Additional headers (key → value pairs).
    pub headers: HashMap<String, String>,
    /// Optional response body (e.g., SDP for `DESCRIBE`).
    pub body: Option<String>,
}

impl RtspResponse {
    /// Constructs a minimal OK response.
    #[must_use]
    pub fn ok(cseq: u32) -> Self {
        Self {
            status: RtspStatus::Ok,
            cseq,
            session_id: None,
            headers: HashMap::new(),
            body: None,
        }
    }

    /// Constructs a simple error response.
    #[must_use]
    pub fn error(status: RtspStatus, cseq: u32) -> Self {
        Self {
            status,
            cseq,
            session_id: None,
            headers: HashMap::new(),
            body: None,
        }
    }

    /// Serialises the response to a `String` following RTSP wire format.
    #[must_use]
    pub fn to_wire(&self) -> String {
        let mut s = format!(
            "RTSP/1.0 {} {}\r\nCSeq: {}\r\n",
            self.status as u16,
            self.status.reason_phrase(),
            self.cseq,
        );
        if let Some(sid) = &self.session_id {
            s.push_str(&format!("Session: {sid}\r\n"));
        }
        for (k, v) in &self.headers {
            s.push_str(&format!("{k}: {v}\r\n"));
        }
        if let Some(body) = &self.body {
            s.push_str(&format!("Content-Length: {}\r\n", body.len()));
            s.push_str("Content-Type: application/sdp\r\n");
            s.push_str("\r\n");
            s.push_str(body);
        } else {
            s.push_str("\r\n");
        }
        s
    }
}

/// Media track descriptor used to build SDP `DESCRIBE` responses.
#[derive(Debug, Clone)]
pub struct MediaTrack {
    /// Media type: `"video"` or `"audio"`.
    pub media_type: String,
    /// RTP payload type number.
    pub payload_type: u8,
    /// Codec name (e.g. `"VP9"`, `"opus"`).
    pub codec: String,
    /// Clock rate in Hz.
    pub clock_rate: u32,
    /// For audio: number of channels.
    pub channels: Option<u8>,
    /// Control URI suffix (e.g. `"track1"`).
    pub control: String,
}

impl MediaTrack {
    /// Renders this track as an SDP `m=` block.
    #[must_use]
    pub fn to_sdp_block(&self) -> String {
        let channel_str = self
            .channels
            .map(|c| format!("/{c}"))
            .unwrap_or_default();
        let port = if self.media_type == "video" { 0u16 } else { 0u16 };
        format!(
            "m={media} {port} RTP/AVP {pt}\r\n\
             a=rtpmap:{pt} {codec}/{rate}{ch}\r\n\
             a=control:{ctrl}\r\n",
            media = self.media_type,
            port = port,
            pt = self.payload_type,
            codec = self.codec,
            rate = self.clock_rate,
            ch = channel_str,
            ctrl = self.control,
        )
    }
}

/// An RTSP server that manages client sessions.
#[derive(Debug, Default)]
pub struct RtspServer {
    /// Active sessions by session ID.
    sessions: HashMap<RtspSessionId, RtspSession>,
    /// Available media tracks.
    tracks: Vec<MediaTrack>,
    /// Stream display name embedded in the SDP.
    stream_name: String,
    /// Next server-side RTP port to assign (odd for RTP, even+1 for RTCP).
    next_server_port: u16,
    /// Monotonic clock (ms), caller-supplied for testing.
    now_ms: u64,
}

impl RtspServer {
    /// Creates a new server with the given stream name and media tracks.
    #[must_use]
    pub fn new(stream_name: impl Into<String>, tracks: Vec<MediaTrack>) -> Self {
        Self {
            sessions: HashMap::new(),
            tracks,
            stream_name: stream_name.into(),
            next_server_port: 10000,
            now_ms: 0,
        }
    }

    /// Advances the internal clock.
    pub fn set_time_ms(&mut self, ms: u64) {
        self.now_ms = ms;
    }

    // ── Request handlers ─────────────────────────────────────────────────────

    /// Handles an `OPTIONS` request.
    #[must_use]
    pub fn handle_options(&self, cseq: u32) -> RtspResponse {
        let mut resp = RtspResponse::ok(cseq);
        resp.headers.insert(
            "Public".to_owned(),
            "OPTIONS, DESCRIBE, SETUP, PLAY, TEARDOWN".to_owned(),
        );
        resp
    }

    /// Handles a `DESCRIBE` request, returning an SDP body.
    #[must_use]
    pub fn handle_describe(&self, cseq: u32, url: &str) -> RtspResponse {
        let sdp = self.build_sdp(url);
        let mut resp = RtspResponse::ok(cseq);
        resp.body = Some(sdp);
        resp
    }

    /// Handles a `SETUP` request: creates or updates a session and assigns
    /// transport parameters.
    pub fn handle_setup(
        &mut self,
        cseq: u32,
        url: &str,
        session_id: Option<&str>,
        client_rtp_port: u16,
        client_rtcp_port: u16,
    ) -> RtspResponse {
        let srv_rtp = self.next_server_port;
        let srv_rtcp = srv_rtp + 1;
        self.next_server_port = srv_rtp + 2;

        let transport = RtspTransport::UdpUnicast {
            client_rtp_port,
            client_rtcp_port,
            server_rtp_port: srv_rtp,
            server_rtcp_port: srv_rtcp,
        };

        let sid = match session_id {
            Some(id) => id.to_owned(),
            None => format!("{:016x}", self.now_ms ^ (cseq as u64)),
        };

        let session = self
            .sessions
            .entry(sid.clone())
            .or_insert_with(|| RtspSession::new(sid.clone(), url.to_owned(), self.now_ms));
        session.transport = Some(transport.clone());
        session.last_cseq = cseq;

        if session.state == RtspSessionState::Init {
            session.state = RtspSessionState::Ready;
        }

        let transport_header = match &transport {
            RtspTransport::UdpUnicast {
                client_rtp_port,
                client_rtcp_port,
                server_rtp_port,
                server_rtcp_port,
            } => format!(
                "RTP/AVP;unicast;client_port={crtp}-{crtcp};server_port={srtp}-{srtcp}",
                crtp = client_rtp_port,
                crtcp = client_rtcp_port,
                srtp = server_rtp_port,
                srtcp = server_rtcp_port,
            ),
            _ => "RTP/AVP".to_owned(),
        };

        let mut resp = RtspResponse::ok(cseq);
        resp.session_id = Some(sid);
        resp.headers.insert("Transport".to_owned(), transport_header);
        resp
    }

    /// Handles a `PLAY` request.
    pub fn handle_play(&mut self, cseq: u32, session_id: &str) -> RtspResponse {
        match self.sessions.get_mut(session_id) {
            None => RtspResponse::error(RtspStatus::SessionNotFound, cseq),
            Some(sess) => {
                if sess.state != RtspSessionState::Ready && sess.state != RtspSessionState::Playing {
                    return RtspResponse::error(RtspStatus::MethodNotValidInState, cseq);
                }
                sess.state = RtspSessionState::Playing;
                sess.last_cseq = cseq;
                let mut resp = RtspResponse::ok(cseq);
                resp.session_id = Some(session_id.to_owned());
                resp.headers.insert("Range".to_owned(), "npt=0.000-".to_owned());
                resp
            }
        }
    }

    /// Handles a `TEARDOWN` request.
    pub fn handle_teardown(&mut self, cseq: u32, session_id: &str) -> RtspResponse {
        match self.sessions.get_mut(session_id) {
            None => RtspResponse::error(RtspStatus::SessionNotFound, cseq),
            Some(sess) => {
                sess.state = RtspSessionState::TearingDown;
                sess.last_cseq = cseq;
                let mut resp = RtspResponse::ok(cseq);
                resp.session_id = Some(session_id.to_owned());
                resp
            }
        }
    }

    // ── Accessors ────────────────────────────────────────────────────────────

    /// Returns a reference to a session.
    pub fn get_session(&self, id: &str) -> RtspResult<&RtspSession> {
        self.sessions
            .get(id)
            .ok_or_else(|| RtspError::SessionNotFound(id.to_owned()))
    }

    /// Returns the number of sessions currently in the `Playing` state.
    #[must_use]
    pub fn playing_count(&self) -> usize {
        self.sessions
            .values()
            .filter(|s| s.is_playing())
            .count()
    }

    /// Removes torn-down sessions.
    pub fn gc_sessions(&mut self) {
        self.sessions
            .retain(|_, s| s.state != RtspSessionState::TearingDown);
    }

    /// Records statistics for an outgoing RTP burst.
    pub fn record_sent(&mut self, session_id: &str, packets: u64, bytes: u64) {
        if let Some(sess) = self.sessions.get_mut(session_id) {
            sess.packets_sent += packets;
            sess.bytes_sent += bytes;
        }
    }

    // ── SDP building ─────────────────────────────────────────────────────────

    fn build_sdp(&self, base_url: &str) -> String {
        let mut sdp = format!(
            "v=0\r\n\
             o=- 0 0 IN IP4 0.0.0.0\r\n\
             s={name}\r\n\
             t=0 0\r\n\
             a=control:{url}\r\n",
            name = self.stream_name,
            url = base_url,
        );
        for track in &self.tracks {
            sdp.push_str(&track.to_sdp_block());
        }
        sdp
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_server() -> RtspServer {
        let tracks = vec![
            MediaTrack {
                media_type: "video".to_owned(),
                payload_type: 96,
                codec: "VP9".to_owned(),
                clock_rate: 90000,
                channels: None,
                control: "track1".to_owned(),
            },
            MediaTrack {
                media_type: "audio".to_owned(),
                payload_type: 97,
                codec: "opus".to_owned(),
                clock_rate: 48000,
                channels: Some(2),
                control: "track2".to_owned(),
            },
        ];
        RtspServer::new("Test Stream", tracks)
    }

    #[test]
    fn test_options_response() {
        let server = make_server();
        let resp = server.handle_options(1);
        assert_eq!(resp.status, RtspStatus::Ok);
        assert_eq!(resp.cseq, 1);
        assert!(resp.headers.get("Public").unwrap().contains("DESCRIBE"));
    }

    #[test]
    fn test_describe_returns_sdp() {
        let server = make_server();
        let resp = server.handle_describe(2, "rtsp://localhost/live");
        assert_eq!(resp.status, RtspStatus::Ok);
        let sdp = resp.body.unwrap();
        assert!(sdp.contains("m=video"));
        assert!(sdp.contains("VP9"));
        assert!(sdp.contains("m=audio"));
    }

    #[test]
    fn test_setup_creates_session() {
        let mut server = make_server();
        server.set_time_ms(1000);
        let resp = server.handle_setup(3, "rtsp://localhost/live/track1", None, 50000, 50001);
        assert_eq!(resp.status, RtspStatus::Ok);
        let sid = resp.session_id.unwrap();
        let sess = server.get_session(&sid).unwrap();
        assert_eq!(sess.state, RtspSessionState::Ready);
    }

    #[test]
    fn test_play_transitions_to_playing() {
        let mut server = make_server();
        let setup = server.handle_setup(1, "rtsp://localhost/live/track1", None, 50000, 50001);
        let sid = setup.session_id.unwrap();
        let resp = server.handle_play(2, &sid);
        assert_eq!(resp.status, RtspStatus::Ok);
        let sess = server.get_session(&sid).unwrap();
        assert!(sess.is_playing());
    }

    #[test]
    fn test_teardown() {
        let mut server = make_server();
        let setup = server.handle_setup(1, "rtsp://localhost/live/track1", None, 50000, 50001);
        let sid = setup.session_id.unwrap();
        server.handle_play(2, &sid);
        let resp = server.handle_teardown(3, &sid);
        assert_eq!(resp.status, RtspStatus::Ok);
        let sess = server.get_session(&sid).unwrap();
        assert_eq!(sess.state, RtspSessionState::TearingDown);
    }

    #[test]
    fn test_play_unknown_session_returns_error() {
        let mut server = make_server();
        let resp = server.handle_play(1, "nosession");
        assert_eq!(resp.status, RtspStatus::SessionNotFound);
    }

    #[test]
    fn test_playing_count() {
        let mut server = make_server();
        let setup = server.handle_setup(1, "rtsp://localhost/live/track1", None, 50000, 50001);
        let sid = setup.session_id.unwrap();
        assert_eq!(server.playing_count(), 0);
        server.handle_play(2, &sid);
        assert_eq!(server.playing_count(), 1);
    }

    #[test]
    fn test_gc_sessions() {
        let mut server = make_server();
        let setup = server.handle_setup(1, "rtsp://localhost/live/track1", None, 50000, 50001);
        let sid = setup.session_id.unwrap();
        server.handle_teardown(2, &sid);
        server.gc_sessions();
        assert!(server.get_session(&sid).is_err());
    }

    #[test]
    fn test_wire_format_ok() {
        let resp = RtspResponse::ok(5);
        let wire = resp.to_wire();
        assert!(wire.starts_with("RTSP/1.0 200 OK"));
        assert!(wire.contains("CSeq: 5"));
    }

    #[test]
    fn test_record_sent() {
        let mut server = make_server();
        let setup = server.handle_setup(1, "rtsp://localhost/live/track1", None, 50000, 50001);
        let sid = setup.session_id.unwrap();
        server.record_sent(&sid, 100, 153600);
        let sess = server.get_session(&sid).unwrap();
        assert_eq!(sess.packets_sent, 100);
        assert_eq!(sess.bytes_sent, 153600);
    }
}
