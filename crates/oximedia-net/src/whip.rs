//! WHIP/WHEP session signaling stubs.
//!
//! WHIP (WebRTC HTTP Ingest Protocol, draft-ietf-wish-whip) and
//! WHEP (WebRTC HTTP Egress Protocol, draft-ietf-wish-whep) provide a
//! simple SDP offer/answer exchange over HTTP for browser-based WebRTC.
//!
//! This module exposes lightweight **session objects** that generate SDP
//! offers, track session state, and parse answers — without making any real
//! network calls.
//!
//! For actual HTTP transport see [`crate::whip_whep::WhipClient`] and
//! [`crate::whip_whep::WhepClient`].
//!
//! # Example
//!
//! ```
//! use oximedia_net::whip::{WhipSession, WhipState};
//!
//! let mut session = WhipSession::new("https://ingest.example.com/whip/live");
//! assert_eq!(session.state, WhipState::Initial);
//!
//! let offer = session.generate_offer("H264", "opus");
//! assert!(offer.starts_with("v=0"));
//!
//! session.process_answer(
//!     "v=0\r\no=- 0 0 IN IP4 0.0.0.0\r\n",
//!     "https://ingest.example.com/whip/live/sess123",
//! );
//! assert_eq!(session.state, WhipState::Established);
//! assert!(session.is_active());
//! ```

use uuid::Uuid;

// ─── ICE credential helpers ───────────────────────────────────────────────────

fn ice_ufrag(seed: &str) -> String {
    let mut h: u32 = 0x811c_9dc5;
    for b in seed.as_bytes() {
        h ^= *b as u32;
        h = h.wrapping_mul(0x0100_0193);
    }
    let chars: &[u8] = b"abcdefghijklmnopqrstuvwxyz0123456789";
    let len = chars.len() as u32;
    (0..4)
        .map(|i| (chars[((h >> (i * 8)) % len) as usize]) as char)
        .collect()
}

fn ice_pwd(seed: &str) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in seed.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    let chars: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let len = chars.len() as u64;
    (0u64..24)
        .map(|i| {
            let mixed = h
                .wrapping_add(i.wrapping_mul(6_364_136_223_846_793_005))
                .wrapping_mul((i + 1).wrapping_mul(2_862_933_555_777_941_757));
            (chars[((mixed >> 33) % len) as usize]) as char
        })
        .collect()
}

// ─── WhipState ────────────────────────────────────────────────────────────────

/// State of a WHIP ingest session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WhipState {
    /// Session created; no offer has been sent.
    Initial,
    /// SDP offer has been sent to the WHIP endpoint.
    OfferSent,
    /// SDP answer received; media is flowing.
    Established,
    /// Session has been terminated.
    Terminated,
}

// ─── WhipSession ─────────────────────────────────────────────────────────────

/// WHIP ingest session for browser-based WebRTC publishing.
///
/// Manages the SDP offer/answer exchange lifecycle for a single
/// WHIP ingest endpoint.
#[derive(Debug, Clone)]
pub struct WhipSession {
    /// Unique session identifier (UUID v4).
    pub session_id: String,
    /// WHIP endpoint URL (the resource to POST the SDP offer to).
    pub endpoint_url: String,
    /// Server-assigned resource URL (available after [`WhipSession::process_answer`]).
    pub resource_url: Option<String>,
    /// Current session state.
    pub state: WhipState,
    /// ICE username fragment.
    pub ice_ufrag: String,
    /// ICE password.
    pub ice_pwd: String,
}

impl WhipSession {
    /// Create a new WHIP session targeting `endpoint_url`.
    ///
    /// Generates a random session ID and deterministic ICE credentials
    /// derived from the session ID.
    #[must_use]
    pub fn new(endpoint_url: impl Into<String>) -> Self {
        let session_id = Uuid::new_v4().to_string();
        let ufrag = ice_ufrag(&session_id);
        let pwd = ice_pwd(&session_id);
        Self {
            session_id,
            endpoint_url: endpoint_url.into(),
            resource_url: None,
            state: WhipState::Initial,
            ice_ufrag: ufrag,
            ice_pwd: pwd,
        }
    }

    /// Generate a minimal SDP offer for WHIP ingest.
    ///
    /// Includes `v=0`, session-level ICE credentials, and one video and one
    /// audio `m=` section for the given `video_codec` and `audio_codec`.
    /// The returned string starts with `v=0` as required by RFC 4566.
    ///
    /// Calling this method transitions the session to [`WhipState::OfferSent`].
    #[must_use]
    pub fn generate_offer(&mut self, video_codec: &str, audio_codec: &str) -> String {
        self.state = WhipState::OfferSent;
        let mut sdp = String::with_capacity(512);
        sdp.push_str("v=0\r\n");
        sdp.push_str(&format!("o=- {} 0 IN IP4 0.0.0.0\r\n", self.session_id));
        sdp.push_str("s=WHIP Ingest\r\n");
        sdp.push_str("t=0 0\r\n");
        sdp.push_str(&format!("a=ice-ufrag:{}\r\n", self.ice_ufrag));
        sdp.push_str(&format!("a=ice-pwd:{}\r\n", self.ice_pwd));
        sdp.push_str("a=fingerprint:sha-256 00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00\r\n");
        sdp.push_str("a=setup:actpass\r\n");
        // Video m= section
        sdp.push_str("m=video 9 UDP/TLS/RTP/SAVPF 96\r\n");
        sdp.push_str("c=IN IP4 0.0.0.0\r\n");
        sdp.push_str(&format!("a=rtpmap:96 {video_codec}/90000\r\n"));
        sdp.push_str("a=sendonly\r\n");
        sdp.push_str("a=mid:video\r\n");
        // Audio m= section
        sdp.push_str("m=audio 9 UDP/TLS/RTP/SAVPF 111\r\n");
        sdp.push_str("c=IN IP4 0.0.0.0\r\n");
        sdp.push_str(&format!("a=rtpmap:111 {audio_codec}/48000/2\r\n"));
        sdp.push_str("a=sendonly\r\n");
        sdp.push_str("a=mid:audio\r\n");
        sdp
    }

    /// Process the SDP answer returned by the WHIP server (HTTP 201 body).
    ///
    /// Sets `resource_url` and transitions to [`WhipState::Established`].
    pub fn process_answer(&mut self, _sdp_answer: &str, resource_url: impl Into<String>) {
        self.resource_url = Some(resource_url.into());
        self.state = WhipState::Established;
    }

    /// Terminate the session.
    ///
    /// Transitions to [`WhipState::Terminated`].
    pub fn terminate(&mut self) {
        self.state = WhipState::Terminated;
    }

    /// Returns `true` when the session is in [`WhipState::Established`].
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.state == WhipState::Established
    }
}

// ─── WhepState ────────────────────────────────────────────────────────────────

/// State of a WHEP egress session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WhepState {
    /// Session created; no request has been sent.
    Initial,
    /// Request (SDP offer) has been sent to the WHEP endpoint.
    Requesting,
    /// Session is established and media is playing.
    Playing,
    /// Session has been stopped.
    Stopped,
}

// ─── WhepSession ─────────────────────────────────────────────────────────────

/// WHEP egress session for browser-based WebRTC playback.
///
/// Manages the SDP offer/answer exchange lifecycle for a single
/// WHEP egress endpoint.
#[derive(Debug, Clone)]
pub struct WhepSession {
    /// Unique session identifier (UUID v4).
    pub session_id: String,
    /// WHEP endpoint URL.
    pub endpoint_url: String,
    /// Current session state.
    pub state: WhepState,
}

impl WhepSession {
    /// Create a new WHEP session targeting `endpoint_url`.
    #[must_use]
    pub fn new(endpoint_url: impl Into<String>) -> Self {
        Self {
            session_id: Uuid::new_v4().to_string(),
            endpoint_url: endpoint_url.into(),
            state: WhepState::Initial,
        }
    }

    /// Generate the HTTP request headers for the initial WHEP POST.
    ///
    /// Returns a list of `(header-name, header-value)` pairs per RFC 7230.
    #[must_use]
    pub fn generate_request_headers(&self) -> Vec<(String, String)> {
        vec![
            ("Content-Type".to_owned(), "application/sdp".to_owned()),
            ("Accept".to_owned(), "application/sdp".to_owned()),
        ]
    }

    /// Process an SDP offer received from the WHEP server.
    ///
    /// Transitions the session to [`WhepState::Playing`].
    pub fn process_offer(&mut self, _sdp_offer: &str) {
        self.state = WhepState::Playing;
    }

    /// Stop the session.
    ///
    /// Transitions to [`WhepState::Stopped`].
    pub fn stop(&mut self) {
        self.state = WhepState::Stopped;
    }

    /// Returns `true` when the session is in [`WhepState::Playing`].
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.state == WhepState::Playing
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // 1. WhipSession new — starts in Initial, has non-empty session_id
    #[test]
    fn test_whip_session_new() {
        let s = WhipSession::new("https://example.com/whip");
        assert_eq!(s.state, WhipState::Initial);
        assert!(!s.session_id.is_empty());
        assert!(!s.ice_ufrag.is_empty());
        assert!(!s.ice_pwd.is_empty());
    }

    // 2. generate_offer contains v=0 and video codec
    #[test]
    fn test_generate_offer_content() {
        let mut s = WhipSession::new("https://example.com/whip");
        let offer = s.generate_offer("H264", "opus");
        assert!(offer.starts_with("v=0"), "SDP must start with v=0");
        assert!(offer.contains("H264"));
        assert!(offer.contains("opus"));
    }

    // 3. generate_offer transitions state to OfferSent
    #[test]
    fn test_generate_offer_state_transition() {
        let mut s = WhipSession::new("https://example.com/whip");
        s.generate_offer("VP9", "opus");
        assert_eq!(s.state, WhipState::OfferSent);
    }

    // 4. process_answer sets resource_url and state to Established
    #[test]
    fn test_process_answer() {
        let mut s = WhipSession::new("https://example.com/whip");
        s.generate_offer("AV1", "opus");
        s.process_answer(
            "v=0\r\no=- 0 0 IN IP4 0.0.0.0\r\n",
            "https://example.com/whip/sess1",
        );
        assert_eq!(s.state, WhipState::Established);
        assert_eq!(
            s.resource_url.as_deref(),
            Some("https://example.com/whip/sess1")
        );
    }

    // 5. is_active true only when Established
    #[test]
    fn test_whip_is_active() {
        let mut s = WhipSession::new("https://example.com/whip");
        assert!(!s.is_active());
        s.generate_offer("VP8", "opus");
        assert!(!s.is_active());
        s.process_answer("v=0\r\n", "https://res/1");
        assert!(s.is_active());
        s.terminate();
        assert!(!s.is_active());
    }

    // 6. terminate transitions to Terminated
    #[test]
    fn test_terminate() {
        let mut s = WhipSession::new("https://example.com/whip");
        s.terminate();
        assert_eq!(s.state, WhipState::Terminated);
    }

    // 7. WhepSession generate_request_headers has Content-Type
    #[test]
    fn test_whep_request_headers() {
        let s = WhepSession::new("https://example.com/whep");
        let headers = s.generate_request_headers();
        let ct = headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case("Content-Type"));
        assert!(ct.is_some(), "must include Content-Type header");
        let (_, v) = ct.expect("checked above");
        assert_eq!(v, "application/sdp");
    }

    // 8. WhepSession is_active correct states
    #[test]
    fn test_whep_is_active() {
        let mut s = WhepSession::new("https://example.com/whep");
        assert!(!s.is_active());
        s.state = WhepState::Requesting;
        assert!(!s.is_active());
        s.process_offer("v=0\r\n");
        assert!(s.is_active());
        s.stop();
        assert!(!s.is_active());
    }

    // 9. WhepSession stop transitions to Stopped
    #[test]
    fn test_whep_stop() {
        let mut s = WhepSession::new("https://example.com/whep");
        s.process_offer("v=0\r\n");
        s.stop();
        assert_eq!(s.state, WhepState::Stopped);
    }

    // 10. Two sessions have different IDs
    #[test]
    fn test_unique_session_ids() {
        let s1 = WhipSession::new("https://example.com/whip");
        let s2 = WhipSession::new("https://example.com/whip");
        assert_ne!(s1.session_id, s2.session_id);
    }
}
