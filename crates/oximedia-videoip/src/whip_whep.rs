//! WHIP/WHEP protocol support for WebRTC-based media ingest and egress.
//!
//! **WHIP** (WebRTC-HTTP Ingest Protocol, draft-ietf-wish-whip) and
//! **WHEP** (WebRTC-HTTP Egress Protocol, draft-ietf-wish-whep) provide a
//! signalling path for WebRTC media sessions over plain HTTP.
//!
//! This module implements the signalling layer (SDP offer/answer exchange over
//! HTTP POST/GET) and the session state machine.  The actual media transport
//! runs over standard RTP/RTCP as defined by WebRTC.
//!
//! # Protocol overview
//!
//! ```text
//!  WHIP Ingest                         WHEP Egress
//!  ───────────                         ───────────
//!  Client                              Client
//!    POST /whip  SDP offer ──>           GET /whep  SDP offer ──>
//!    <── 201 Created  SDP answer         <── 200 OK  SDP answer
//!    (media flows)                        (media flows)
//!    DELETE /whip/<id>  ──>               DELETE /whep/<id>  ──>
//!    <── 200 OK                           <── 200 OK
//! ```

#![allow(dead_code)]

use std::collections::HashMap;

/// Unique session identifier (UUID-like string).
pub type SessionId = String;

/// Role of a WHIP/WHEP session endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SessionRole {
    /// WHIP ingesting client — pushes media to the server.
    WhipIngester,
    /// WHEP egress client — pulls media from the server.
    WhepViewer,
}

/// State of a WHIP/WHEP session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionState {
    /// Session offer has been received; waiting for answer.
    Offering,
    /// SDP answer has been sent; ICE/DTLS negotiation in progress.
    Negotiating,
    /// ICE connected; media flowing.
    Connected,
    /// Session is being torn down.
    Terminating,
    /// Session has been closed.
    Closed,
}

/// ICE candidate trickling entry.
#[derive(Debug, Clone)]
pub struct IceCandidate {
    /// The raw SDP `a=candidate:…` line value.
    pub candidate: String,
    /// SDP mid (media stream identification tag).
    pub sdp_mid: Option<String>,
    /// SDP m-line index.
    pub sdp_mline_index: Option<u16>,
}

/// An SDP body exchanged during WHIP/WHEP negotiation.
#[derive(Debug, Clone)]
pub struct SdpBody {
    /// Raw SDP text.
    pub sdp: String,
    /// SDP type: `"offer"` or `"answer"`.
    pub sdp_type: SdpType,
}

/// SDP message type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SdpType {
    /// SDP offer.
    Offer,
    /// SDP answer.
    Answer,
}

/// A WHIP/WHEP session entry maintained by the server.
#[derive(Debug, Clone)]
pub struct WhipWhepSession {
    /// Session identifier.
    pub id: SessionId,
    /// Session role.
    pub role: SessionRole,
    /// Current session state.
    pub state: SessionState,
    /// The remote SDP offer received from the client.
    pub remote_sdp: Option<SdpBody>,
    /// The local SDP answer produced by the server.
    pub local_sdp: Option<SdpBody>,
    /// Queued trickle-ICE candidates from the remote side.
    pub pending_candidates: Vec<IceCandidate>,
    /// Optional human-readable stream name.
    pub stream_name: Option<String>,
    /// Session creation timestamp (milliseconds since Unix epoch).
    pub created_at_ms: u64,
}

impl WhipWhepSession {
    /// Creates a new session in the `Offering` state.
    #[must_use]
    pub fn new(id: SessionId, role: SessionRole, created_at_ms: u64) -> Self {
        Self {
            id,
            role,
            state: SessionState::Offering,
            remote_sdp: None,
            local_sdp: None,
            pending_candidates: Vec::new(),
            stream_name: None,
            created_at_ms,
        }
    }

    /// Advances the session state to `Negotiating` after the local answer has been set.
    pub fn set_answer(&mut self, answer: SdpBody) {
        self.local_sdp = Some(answer);
        self.state = SessionState::Negotiating;
    }

    /// Marks the session as `Connected` (ICE/DTLS completed).
    pub fn mark_connected(&mut self) {
        self.state = SessionState::Connected;
    }

    /// Appends a trickle-ICE candidate.
    pub fn add_candidate(&mut self, candidate: IceCandidate) {
        self.pending_candidates.push(candidate);
    }

    /// Begins teardown.
    pub fn terminate(&mut self) {
        self.state = SessionState::Terminating;
    }

    /// Finalises teardown.
    pub fn close(&mut self) {
        self.state = SessionState::Closed;
    }

    /// Returns `true` when the session is active (not closed/terminating).
    #[must_use]
    pub fn is_active(&self) -> bool {
        !matches!(self.state, SessionState::Closed | SessionState::Terminating)
    }
}

/// Error type for WHIP/WHEP operations.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum WhipWhepError {
    /// Session not found.
    #[error("session '{0}' not found")]
    SessionNotFound(String),
    /// Session already exists.
    #[error("session '{0}' already exists")]
    SessionExists(String),
    /// Invalid SDP (e.g., missing required fields).
    #[error("invalid SDP: {0}")]
    InvalidSdp(String),
    /// Operation is not valid in the current state.
    #[error("invalid state transition from {current:?}: {reason}")]
    InvalidState {
        /// Current state.
        current: SessionState,
        /// Human-readable reason.
        reason: String,
    },
    /// Resource limit reached (e.g., max concurrent sessions).
    #[error("resource limit: {0}")]
    ResourceLimit(String),
}

/// Result type for WHIP/WHEP operations.
pub type WhipWhepResult<T> = Result<T, WhipWhepError>;

/// Server-side session manager for WHIP/WHEP protocols.
///
/// Maintains a registry of active sessions and implements the core
/// offer/answer exchange logic.  In a real deployment this would be
/// integrated with a WebRTC peer connection implementation; here the
/// server generates a minimal SDP answer mirroring the offered codecs.
#[derive(Debug, Default)]
pub struct WhipWhepServer {
    /// Active sessions indexed by session ID.
    sessions: HashMap<SessionId, WhipWhepSession>,
    /// Maximum number of concurrent sessions (0 = unlimited).
    max_sessions: usize,
    /// Monotonic millisecond clock (caller-supplied, enables deterministic tests).
    now_ms: u64,
}

impl WhipWhepServer {
    /// Creates a new server with an optional session limit.
    #[must_use]
    pub fn new(max_sessions: usize) -> Self {
        Self {
            sessions: HashMap::new(),
            max_sessions,
            now_ms: 0,
        }
    }

    /// Advances the internal clock (for testing / time-aware logic).
    pub fn set_time_ms(&mut self, ms: u64) {
        self.now_ms = ms;
    }

    // ── WHIP ingest ──────────────────────────────────────────────────────────

    /// Handles a WHIP HTTP `POST` request: creates a session, validates the SDP
    /// offer, and generates an SDP answer.
    ///
    /// Returns the session ID and the server-side SDP answer body.
    pub fn handle_whip_offer(
        &mut self,
        session_id: SessionId,
        offer_sdp: &str,
    ) -> WhipWhepResult<(SessionId, SdpBody)> {
        self.check_resource_limit()?;
        if self.sessions.contains_key(&session_id) {
            return Err(WhipWhepError::SessionExists(session_id));
        }

        validate_sdp_minimal(offer_sdp)?;

        let mut session = WhipWhepSession::new(
            session_id.clone(),
            SessionRole::WhipIngester,
            self.now_ms,
        );
        session.remote_sdp = Some(SdpBody {
            sdp: offer_sdp.to_owned(),
            sdp_type: SdpType::Offer,
        });

        let answer = generate_sdp_answer(offer_sdp, &session_id);
        session.set_answer(SdpBody {
            sdp: answer.clone(),
            sdp_type: SdpType::Answer,
        });

        self.sessions.insert(session_id.clone(), session);
        Ok((session_id, SdpBody { sdp: answer, sdp_type: SdpType::Answer }))
    }

    /// Handles a WHEP HTTP `POST` request: creates an egress session and
    /// generates an SDP answer.
    pub fn handle_whep_offer(
        &mut self,
        session_id: SessionId,
        offer_sdp: &str,
    ) -> WhipWhepResult<(SessionId, SdpBody)> {
        self.check_resource_limit()?;
        if self.sessions.contains_key(&session_id) {
            return Err(WhipWhepError::SessionExists(session_id));
        }

        validate_sdp_minimal(offer_sdp)?;

        let mut session = WhipWhepSession::new(
            session_id.clone(),
            SessionRole::WhepViewer,
            self.now_ms,
        );
        session.remote_sdp = Some(SdpBody {
            sdp: offer_sdp.to_owned(),
            sdp_type: SdpType::Offer,
        });

        let answer = generate_sdp_answer(offer_sdp, &session_id);
        session.set_answer(SdpBody {
            sdp: answer.clone(),
            sdp_type: SdpType::Answer,
        });

        self.sessions.insert(session_id.clone(), session);
        Ok((session_id, SdpBody { sdp: answer, sdp_type: SdpType::Answer }))
    }

    /// Handles a trickle-ICE `PATCH` request: appends a remote ICE candidate to
    /// the identified session.
    pub fn handle_ice_candidate(
        &mut self,
        session_id: &str,
        candidate: IceCandidate,
    ) -> WhipWhepResult<()> {
        let session = self.get_session_mut(session_id)?;
        if !session.is_active() {
            return Err(WhipWhepError::InvalidState {
                current: session.state.clone(),
                reason: "session is not active".to_owned(),
            });
        }
        session.add_candidate(candidate);
        Ok(())
    }

    /// Handles a `DELETE` request: tears down the session.
    pub fn handle_delete(&mut self, session_id: &str) -> WhipWhepResult<()> {
        let session = self.get_session_mut(session_id)?;
        session.terminate();
        session.close();
        Ok(())
    }

    /// Marks a session as `Connected` (called by the transport layer when ICE succeeds).
    pub fn mark_connected(&mut self, session_id: &str) -> WhipWhepResult<()> {
        let session = self.get_session_mut(session_id)?;
        session.mark_connected();
        Ok(())
    }

    // ── Session accessors ────────────────────────────────────────────────────

    /// Returns a reference to a session by ID.
    pub fn get_session(&self, id: &str) -> WhipWhepResult<&WhipWhepSession> {
        self.sessions
            .get(id)
            .ok_or_else(|| WhipWhepError::SessionNotFound(id.to_owned()))
    }

    /// Returns a mutable reference to a session by ID.
    pub fn get_session_mut(&mut self, id: &str) -> WhipWhepResult<&mut WhipWhepSession> {
        self.sessions
            .get_mut(id)
            .ok_or_else(|| WhipWhepError::SessionNotFound(id.to_owned()))
    }

    /// Returns a list of all active session IDs and their roles.
    #[must_use]
    pub fn active_sessions(&self) -> Vec<(&str, SessionRole)> {
        self.sessions
            .values()
            .filter(|s| s.is_active())
            .map(|s| (s.id.as_str(), s.role))
            .collect()
    }

    /// Total number of tracked sessions (including closed ones).
    #[must_use]
    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }

    /// Removes sessions that are in the `Closed` state to free memory.
    pub fn gc_closed_sessions(&mut self) {
        self.sessions.retain(|_, s| s.state != SessionState::Closed);
    }

    // ── Internal helpers ─────────────────────────────────────────────────────

    fn check_resource_limit(&self) -> WhipWhepResult<()> {
        if self.max_sessions > 0 {
            let active = self
                .sessions
                .values()
                .filter(|s| s.is_active())
                .count();
            if active >= self.max_sessions {
                return Err(WhipWhepError::ResourceLimit(format!(
                    "max {max} concurrent sessions reached",
                    max = self.max_sessions
                )));
            }
        }
        Ok(())
    }
}

// ── SDP helpers ──────────────────────────────────────────────────────────────

/// Validates that `sdp` is non-empty and contains the mandatory `v=` and `o=` lines.
fn validate_sdp_minimal(sdp: &str) -> WhipWhepResult<()> {
    if sdp.trim().is_empty() {
        return Err(WhipWhepError::InvalidSdp("empty SDP".to_owned()));
    }
    if !sdp.contains("v=") {
        return Err(WhipWhepError::InvalidSdp(
            "missing SDP version line (v=)".to_owned(),
        ));
    }
    if !sdp.contains("o=") {
        return Err(WhipWhepError::InvalidSdp(
            "missing SDP origin line (o=)".to_owned(),
        ));
    }
    Ok(())
}

/// Generates a minimal SDP answer that accepts all media sections from `offer_sdp`.
///
/// In production this would be produced by a WebRTC engine (e.g. `libwebrtc` or
/// `str0m`); here we produce a structurally valid but simplified answer suitable
/// for unit testing the signalling path.
fn generate_sdp_answer(offer_sdp: &str, session_id: &str) -> String {
    let mut lines: Vec<String> = Vec::new();
    lines.push("v=0".to_owned());
    lines.push(format!(
        "o=oximedia-videoip 0 1 IN IP4 0.0.0.0"
    ));
    lines.push(format!("s=OxiMedia WHIP/WHEP session {session_id}"));
    lines.push("t=0 0".to_owned());
    lines.push("a=group:BUNDLE".to_owned());

    // Mirror every m= section from the offer, changing `sendrecv`/`sendonly`
    // to the appropriate reciprocal direction.
    let mut in_media = false;
    let mut mid_index: u32 = 0;
    for raw_line in offer_sdp.lines() {
        let line = raw_line.trim();
        if line.starts_with("m=") {
            in_media = true;
            lines.push(line.to_owned());
            lines.push(format!("a=mid:{mid_index}"));
            mid_index += 1;
            lines.push("a=recvonly".to_owned());
        } else if in_media {
            if line.starts_with("a=sendrecv") || line.starts_with("a=sendonly") || line.starts_with("a=recvonly") {
                // Direction already set above; skip the offer's direction.
                continue;
            }
            if line.starts_with("a=mid:") {
                // Skip offer's mid; we already emitted our own.
                continue;
            }
            if !line.is_empty() && !line.starts_with("c=") {
                lines.push(line.to_owned());
            }
        }
    }

    lines.join("\r\n") + "\r\n"
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL_OFFER: &str = concat!(
        "v=0\r\n",
        "o=- 1234 1 IN IP4 127.0.0.1\r\n",
        "s=Test\r\n",
        "t=0 0\r\n",
        "m=video 9 UDP/TLS/RTP/SAVPF 96\r\n",
        "a=rtpmap:96 VP9/90000\r\n",
        "a=sendonly\r\n",
        "a=mid:0\r\n",
    );

    #[test]
    fn test_whip_offer_creates_session() {
        let mut server = WhipWhepServer::new(0);
        let (sid, answer) = server.handle_whip_offer("sess1".into(), MINIMAL_OFFER).unwrap();
        assert_eq!(sid, "sess1");
        assert_eq!(answer.sdp_type, SdpType::Answer);
        assert!(answer.sdp.contains("v=0"));
    }

    #[test]
    fn test_whep_offer_creates_session() {
        let mut server = WhipWhepServer::new(0);
        let (sid, answer) = server.handle_whep_offer("sess2".into(), MINIMAL_OFFER).unwrap();
        assert_eq!(sid, "sess2");
        assert!(answer.sdp.contains("m=video"));
    }

    #[test]
    fn test_duplicate_session_rejected() {
        let mut server = WhipWhepServer::new(0);
        server.handle_whip_offer("dup".into(), MINIMAL_OFFER).unwrap();
        let result = server.handle_whip_offer("dup".into(), MINIMAL_OFFER);
        assert!(matches!(result, Err(WhipWhepError::SessionExists(_))));
    }

    #[test]
    fn test_session_limit_enforced() {
        let mut server = WhipWhepServer::new(1);
        server.handle_whip_offer("s1".into(), MINIMAL_OFFER).unwrap();
        let result = server.handle_whip_offer("s2".into(), MINIMAL_OFFER);
        assert!(matches!(result, Err(WhipWhepError::ResourceLimit(_))));
    }

    #[test]
    fn test_delete_session() {
        let mut server = WhipWhepServer::new(0);
        server.handle_whip_offer("s1".into(), MINIMAL_OFFER).unwrap();
        server.handle_delete("s1").unwrap();
        let sess = server.get_session("s1").unwrap();
        assert_eq!(sess.state, SessionState::Closed);
    }

    #[test]
    fn test_gc_closed_sessions() {
        let mut server = WhipWhepServer::new(0);
        server.handle_whip_offer("s1".into(), MINIMAL_OFFER).unwrap();
        server.handle_delete("s1").unwrap();
        assert_eq!(server.session_count(), 1);
        server.gc_closed_sessions();
        assert_eq!(server.session_count(), 0);
    }

    #[test]
    fn test_ice_candidate_added() {
        let mut server = WhipWhepServer::new(0);
        server.handle_whip_offer("s1".into(), MINIMAL_OFFER).unwrap();
        let cand = IceCandidate {
            candidate: "candidate:1 1 UDP 2122252543 192.168.1.2 55000 typ host".to_owned(),
            sdp_mid: Some("0".to_owned()),
            sdp_mline_index: Some(0),
        };
        server.handle_ice_candidate("s1", cand).unwrap();
        let sess = server.get_session("s1").unwrap();
        assert_eq!(sess.pending_candidates.len(), 1);
    }

    #[test]
    fn test_mark_connected() {
        let mut server = WhipWhepServer::new(0);
        server.handle_whip_offer("s1".into(), MINIMAL_OFFER).unwrap();
        server.mark_connected("s1").unwrap();
        let sess = server.get_session("s1").unwrap();
        assert_eq!(sess.state, SessionState::Connected);
    }

    #[test]
    fn test_invalid_sdp_empty() {
        let mut server = WhipWhepServer::new(0);
        let result = server.handle_whip_offer("bad".into(), "");
        assert!(matches!(result, Err(WhipWhepError::InvalidSdp(_))));
    }

    #[test]
    fn test_session_not_found() {
        let server = WhipWhepServer::new(0);
        let result = server.get_session("nonexistent");
        assert!(matches!(result, Err(WhipWhepError::SessionNotFound(_))));
    }

    #[test]
    fn test_active_sessions_listing() {
        let mut server = WhipWhepServer::new(0);
        server.handle_whip_offer("s1".into(), MINIMAL_OFFER).unwrap();
        server.handle_whep_offer("s2".into(), MINIMAL_OFFER).unwrap();
        let active = server.active_sessions();
        assert_eq!(active.len(), 2);
    }

    #[test]
    fn test_subpixel_offset_negligible() {
        let o = SubPixelOffset2d { dx: 0.05, dy: 0.02 };
        assert!(o.is_negligible(0.1));
    }

    #[test]
    fn test_subpixel_offset_not_negligible() {
        let o = SubPixelOffset2d { dx: 0.15, dy: 0.02 };
        assert!(!o.is_negligible(0.1));
    }
}

/// A 2-D sub-pixel offset (re-exported for integration with stream compositing).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SubPixelOffset2d {
    /// Fractional horizontal offset.
    pub dx: f64,
    /// Fractional vertical offset.
    pub dy: f64,
}

impl SubPixelOffset2d {
    /// Returns `true` when both components are below `threshold`.
    #[must_use]
    pub fn is_negligible(&self, threshold: f64) -> bool {
        self.dx.abs() < threshold && self.dy.abs() < threshold
    }
}
