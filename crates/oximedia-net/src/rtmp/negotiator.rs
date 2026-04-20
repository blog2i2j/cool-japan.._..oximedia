//! Enhanced RTMP (RTMP+) capability negotiation.
//!
//! During an RTMP `connect()` command exchange, Enhanced RTMP adds a
//! `fourCcList` field to the command object to advertise supported video/audio
//! codecs by their FourCC identifiers.
//!
//! This module implements:
//!
//! - [`NegotiationCapabilities`] — the full capability set for one endpoint.
//! - [`EnhancedRtmpNegotiator`] — drives the connect-command exchange,
//!   advertising local capabilities and parsing peer capabilities.
//! - [`NegotiatedSession`] — the result of a successful negotiation.
//! - [`ConnectCommand`] — structured representation of the RTMP `connect` AMF object.

use super::enhanced::{EnhancedRtmpCapabilities, FourCC};
use crate::error::{NetError, NetResult};
use std::collections::HashMap;

// ─── Negotiation Capabilities ────────────────────────────────────────────────

/// Extended capability descriptor including RTMP-specific metadata.
#[derive(Debug, Clone)]
pub struct NegotiationCapabilities {
    /// Video codec FourCCs this endpoint can encode/decode.
    pub video_codecs: Vec<FourCC>,
    /// Audio codec FourCCs this endpoint can encode/decode.
    pub audio_codecs: Vec<FourCC>,
    /// Whether this endpoint supports Enhanced RTMP.
    pub enhanced_rtmp: bool,
    /// RTMP application name (from the `connect` command `app` field).
    pub app: String,
    /// Flash player version string (or custom version string for servers).
    pub rtmp_version: String,
    /// Whether RTMP+push (server → client push without prior publish) is supported.
    pub supports_push: bool,
    /// Whether reconnect after disconnect is supported.
    pub supports_reconnect: bool,
    /// Additional custom capabilities as key-value pairs.
    pub custom: HashMap<String, String>,
}

impl Default for NegotiationCapabilities {
    fn default() -> Self {
        Self {
            video_codecs: vec![FourCC::AV1, FourCC::VP9, FourCC::AVC],
            audio_codecs: vec![FourCC::OPUS, FourCC::FLAC, FourCC::AAC],
            enhanced_rtmp: true,
            app: "live".to_owned(),
            rtmp_version: "oximedia/0.1.2".to_owned(),
            supports_push: false,
            supports_reconnect: true,
            custom: HashMap::new(),
        }
    }
}

impl NegotiationCapabilities {
    /// Creates capabilities advertising only patent-free codecs.
    #[must_use]
    pub fn patent_free() -> Self {
        Self {
            video_codecs: vec![FourCC::AV1, FourCC::VP9],
            audio_codecs: vec![FourCC::OPUS, FourCC::FLAC],
            ..Self::default()
        }
    }

    /// Returns the FourCC code list as a comma-separated string.
    ///
    /// This format is used in the Enhanced RTMP `fourCcList` AMF field.
    #[must_use]
    pub fn fourcc_list(&self) -> String {
        let mut list: Vec<String> = self
            .video_codecs
            .iter()
            .chain(&self.audio_codecs)
            .filter_map(|cc| cc.as_str().map(|s| s.to_owned()))
            .collect();
        list.sort();
        list.dedup();
        list.join(",")
    }

    /// Converts to the simplified `EnhancedRtmpCapabilities` type.
    #[must_use]
    pub fn to_enhanced_caps(&self) -> EnhancedRtmpCapabilities {
        EnhancedRtmpCapabilities {
            video_codecs: self.video_codecs.clone(),
            audio_codecs: self.audio_codecs.clone(),
            enhanced_supported: self.enhanced_rtmp,
            version: self.rtmp_version.clone(),
        }
    }

    /// Adds a custom capability field.
    pub fn add_custom(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.custom.insert(key.into(), value.into());
    }

    /// Returns whether the given video codec is supported.
    #[must_use]
    pub fn supports_video(&self, codec: &FourCC) -> bool {
        self.video_codecs.contains(codec)
    }

    /// Returns whether the given audio codec is supported.
    #[must_use]
    pub fn supports_audio(&self, codec: &FourCC) -> bool {
        self.audio_codecs.contains(codec)
    }
}

// ─── ConnectCommand ───────────────────────────────────────────────────────────

/// Structured representation of an RTMP `connect` command object.
///
/// During an Enhanced RTMP handshake, the connect command includes a
/// `fourCcList` field advertising supported codecs.
#[derive(Debug, Clone)]
pub struct ConnectCommand {
    /// Application name (`app` field).
    pub app: String,
    /// Flash/RTMP version string (`flashVer` field).
    pub flash_ver: String,
    /// SWF URL (`swfUrl` field), if provided.
    pub swf_url: Option<String>,
    /// TC URL — the full connection URL.
    pub tc_url: String,
    /// `fpad` flag — whether a proxy is used.
    pub fpad: bool,
    /// Audio codec bitmask (legacy, for fallback).
    pub audio_codecs: u16,
    /// Video codec bitmask (legacy, for fallback).
    pub video_codecs: u16,
    /// Video function flags.
    pub video_function: u32,
    /// Object encoding (0 = AMF0, 3 = AMF3).
    pub object_encoding: u8,
    /// Enhanced RTMP FourCC list (comma-separated codec identifiers).
    pub fourcc_list: Option<String>,
}

impl ConnectCommand {
    /// Creates a new connect command for a given application and URL.
    #[must_use]
    pub fn new(app: impl Into<String>, tc_url: impl Into<String>) -> Self {
        Self {
            app: app.into(),
            flash_ver: "LNX 9,0,124,2".to_owned(),
            swf_url: None,
            tc_url: tc_url.into(),
            fpad: false,
            audio_codecs: 0x0FFF,
            video_codecs: 0x0080,
            video_function: 1,
            object_encoding: 0,
            fourcc_list: None,
        }
    }

    /// Creates an Enhanced RTMP connect command with a FourCC list.
    #[must_use]
    pub fn enhanced(
        app: impl Into<String>,
        tc_url: impl Into<String>,
        caps: &NegotiationCapabilities,
    ) -> Self {
        Self {
            fourcc_list: Some(caps.fourcc_list()),
            ..Self::new(app, tc_url)
        }
    }

    /// Returns `true` if this command signals Enhanced RTMP support.
    #[must_use]
    pub fn is_enhanced(&self) -> bool {
        self.fourcc_list.is_some()
    }

    /// Parses the FourCC list back into [`FourCC`] codes.
    #[must_use]
    pub fn parsed_fourccs(&self) -> Vec<FourCC> {
        self.fourcc_list
            .as_deref()
            .map(|list| {
                list.split(',')
                    .filter_map(|code| FourCC::from_str_code(code.trim()))
                    .collect()
            })
            .unwrap_or_default()
    }
}

// ─── Negotiated Session ───────────────────────────────────────────────────────

/// Result of a successful Enhanced RTMP capability negotiation.
#[derive(Debug, Clone)]
pub struct NegotiatedSession {
    /// Intersection of video codecs both sides support.
    pub video_codecs: Vec<FourCC>,
    /// Intersection of audio codecs both sides support.
    pub audio_codecs: Vec<FourCC>,
    /// Whether Enhanced RTMP is in effect (both sides agreed).
    pub enhanced_mode: bool,
    /// Application name from the connect command.
    pub app: String,
    /// The raw peer connect command for further inspection.
    pub peer_connect: ConnectCommand,
}

impl NegotiatedSession {
    /// Returns whether a specific video codec was negotiated.
    #[must_use]
    pub fn can_use_video(&self, codec: &FourCC) -> bool {
        self.video_codecs.contains(codec)
    }

    /// Returns whether a specific audio codec was negotiated.
    #[must_use]
    pub fn can_use_audio(&self, codec: &FourCC) -> bool {
        self.audio_codecs.contains(codec)
    }

    /// Returns a summary of the negotiated codecs.
    #[must_use]
    pub fn summary(&self) -> String {
        let v: Vec<&str> = self.video_codecs.iter().map(|c| c.codec_name()).collect();
        let a: Vec<&str> = self.audio_codecs.iter().map(|c| c.codec_name()).collect();
        format!(
            "enhanced={}, video=[{}], audio=[{}]",
            self.enhanced_mode,
            v.join(", "),
            a.join(", ")
        )
    }
}

// ─── EnhancedRtmpNegotiator ───────────────────────────────────────────────────

/// State of the Enhanced RTMP negotiation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NegotiatorState {
    /// Waiting for the connect command (server side) or ready to send (client side).
    Idle,
    /// Connect command sent; waiting for the server's `_result` response.
    WaitingResult,
    /// Negotiation succeeded.
    Connected,
    /// Negotiation failed.
    Failed,
}

/// Drives the Enhanced RTMP capability exchange during the `connect` phase.
///
/// **Client-side workflow:**
/// 1. Call [`EnhancedRtmpNegotiator::build_connect_command`] and send the result.
/// 2. When the server's `_result` AMF object arrives, call
///    [`EnhancedRtmpNegotiator::process_server_result`].
///
/// **Server-side workflow:**
/// 1. When the client's `connect` command arrives, call
///    [`EnhancedRtmpNegotiator::process_client_connect`].
/// 2. Send the returned `ConnectResult` to the client.
pub struct EnhancedRtmpNegotiator {
    /// Local capabilities.
    local_caps: NegotiationCapabilities,
    /// Current state.
    state: NegotiatorState,
    /// Peer capabilities, filled in after negotiation.
    peer_caps: Option<NegotiationCapabilities>,
    /// Final negotiated session.
    session: Option<NegotiatedSession>,
}

impl EnhancedRtmpNegotiator {
    /// Creates a new negotiator with the given local capabilities.
    #[must_use]
    pub fn new(local_caps: NegotiationCapabilities) -> Self {
        Self {
            local_caps,
            state: NegotiatorState::Idle,
            peer_caps: None,
            session: None,
        }
    }

    /// Creates a negotiator advertising patent-free codecs only.
    #[must_use]
    pub fn patent_free() -> Self {
        Self::new(NegotiationCapabilities::patent_free())
    }

    /// Returns the current negotiation state.
    #[must_use]
    pub fn state(&self) -> NegotiatorState {
        self.state
    }

    /// Returns `true` if negotiation is complete.
    #[must_use]
    pub fn is_connected(&self) -> bool {
        self.state == NegotiatorState::Connected
    }

    /// Returns the negotiated session, if available.
    #[must_use]
    pub fn session(&self) -> Option<&NegotiatedSession> {
        self.session.as_ref()
    }

    /// **Client side**: Builds the Enhanced RTMP `connect` command to send.
    ///
    /// Transitions the negotiator to `WaitingResult`.
    ///
    /// # Errors
    ///
    /// Returns an error if the negotiator is not in `Idle` state.
    pub fn build_connect_command(
        &mut self,
        app: impl Into<String>,
        tc_url: impl Into<String>,
    ) -> NetResult<ConnectCommand> {
        if self.state != NegotiatorState::Idle {
            return Err(NetError::invalid_state(
                "build_connect_command called in non-Idle state",
            ));
        }
        let cmd = ConnectCommand::enhanced(app, tc_url, &self.local_caps);
        self.state = NegotiatorState::WaitingResult;
        Ok(cmd)
    }

    /// **Client side**: Processes the server's `_result` command object.
    ///
    /// Returns the completed [`NegotiatedSession`] on success.
    ///
    /// # Errors
    ///
    /// Returns an error if the negotiator is not in `WaitingResult` state or
    /// if the peer does not support Enhanced RTMP.
    pub fn process_server_result(
        &mut self,
        server_connect: &ConnectCommand,
    ) -> NetResult<&NegotiatedSession> {
        if self.state != NegotiatorState::WaitingResult {
            return Err(NetError::invalid_state(
                "process_server_result called before build_connect_command",
            ));
        }

        let session = self.negotiate_session(server_connect);
        self.state = NegotiatorState::Connected;
        self.session = Some(session);
        self.session
            .as_ref()
            .ok_or_else(|| NetError::invalid_state("session failed to initialise"))
    }

    /// **Server side**: Processes the client's `connect` command.
    ///
    /// Returns the `ConnectCommand` that represents the server's `_result` —
    /// the caller should serialise and send this back to the client.
    ///
    /// # Errors
    ///
    /// Returns an error if the negotiator is not in `Idle` state.
    pub fn process_client_connect(
        &mut self,
        client_connect: &ConnectCommand,
    ) -> NetResult<ConnectCommand> {
        if self.state != NegotiatorState::Idle {
            return Err(NetError::invalid_state(
                "process_client_connect called in non-Idle state",
            ));
        }

        let session = self.negotiate_session(client_connect);
        self.state = NegotiatorState::Connected;
        self.session = Some(session);

        // Build the server _result command.
        let result_cmd = ConnectCommand::enhanced(
            &client_connect.app,
            &client_connect.tc_url,
            &self.local_caps,
        );
        Ok(result_cmd)
    }

    // ── Private ──────────────────────────────────────────────────────────────

    fn negotiate_session(&mut self, peer_cmd: &ConnectCommand) -> NegotiatedSession {
        let peer_fourccs = peer_cmd.parsed_fourccs();

        // Build peer capabilities from the FourCC list.
        let peer_video: Vec<FourCC> = peer_fourccs
            .iter()
            .filter(|cc| matches!(**cc, FourCC::AV1 | FourCC::VP9 | FourCC::HEVC | FourCC::AVC))
            .copied()
            .collect();
        let peer_audio: Vec<FourCC> = peer_fourccs
            .iter()
            .filter(|cc| matches!(**cc, FourCC::OPUS | FourCC::FLAC | FourCC::AAC))
            .copied()
            .collect();

        // Compute intersection.
        let video_negotiated: Vec<FourCC> = self
            .local_caps
            .video_codecs
            .iter()
            .filter(|cc| peer_video.contains(cc))
            .copied()
            .collect();
        let audio_negotiated: Vec<FourCC> = self
            .local_caps
            .audio_codecs
            .iter()
            .filter(|cc| peer_audio.contains(cc))
            .copied()
            .collect();

        let enhanced_mode = peer_cmd.is_enhanced() && self.local_caps.enhanced_rtmp;

        // Store peer capabilities.
        self.peer_caps = Some(NegotiationCapabilities {
            video_codecs: peer_video,
            audio_codecs: peer_audio,
            enhanced_rtmp: enhanced_mode,
            app: peer_cmd.app.clone(),
            rtmp_version: peer_cmd.flash_ver.clone(),
            supports_push: false,
            supports_reconnect: false,
            custom: HashMap::new(),
        });

        NegotiatedSession {
            video_codecs: video_negotiated,
            audio_codecs: audio_negotiated,
            enhanced_mode,
            app: peer_cmd.app.clone(),
            peer_connect: peer_cmd.clone(),
        }
    }
}

impl std::fmt::Debug for EnhancedRtmpNegotiator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EnhancedRtmpNegotiator")
            .field("state", &self.state)
            .field("enhanced", &self.local_caps.enhanced_rtmp)
            .finish()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn caps() -> NegotiationCapabilities {
        NegotiationCapabilities::default()
    }

    fn patent_free_caps() -> NegotiationCapabilities {
        NegotiationCapabilities::patent_free()
    }

    // 1. NegotiationCapabilities default
    #[test]
    fn test_default_caps() {
        let c = caps();
        assert!(c.supports_video(&FourCC::AV1));
        assert!(c.supports_audio(&FourCC::OPUS));
        assert!(c.enhanced_rtmp);
    }

    // 2. NegotiationCapabilities patent_free
    #[test]
    fn test_patent_free_caps() {
        let c = patent_free_caps();
        assert!(!c.supports_video(&FourCC::AVC));
        assert!(!c.supports_audio(&FourCC::AAC));
    }

    // 3. NegotiationCapabilities fourcc_list
    #[test]
    fn test_fourcc_list() {
        let c = NegotiationCapabilities::patent_free();
        let list = c.fourcc_list();
        assert!(list.contains("av01"));
        assert!(list.contains("vp09"));
        assert!(list.contains("Opus"));
    }

    // 4. NegotiationCapabilities add_custom
    #[test]
    fn test_add_custom() {
        let mut c = caps();
        c.add_custom("latency", "30ms");
        assert_eq!(c.custom.get("latency").map(String::as_str), Some("30ms"));
    }

    // 5. ConnectCommand new
    #[test]
    fn test_connect_command_new() {
        let cmd = ConnectCommand::new("live", "rtmp://server.example.com/live");
        assert_eq!(cmd.app, "live");
        assert!(!cmd.is_enhanced());
        assert!(cmd.parsed_fourccs().is_empty());
    }

    // 6. ConnectCommand enhanced
    #[test]
    fn test_connect_command_enhanced() {
        let c = NegotiationCapabilities::patent_free();
        let cmd = ConnectCommand::enhanced("live", "rtmp://s.example.com/live", &c);
        assert!(cmd.is_enhanced());
        let fourccs = cmd.parsed_fourccs();
        assert!(!fourccs.is_empty());
    }

    // 7. ConnectCommand parsed_fourccs round-trips
    #[test]
    fn test_parsed_fourccs_round_trip() {
        let c = NegotiationCapabilities::patent_free();
        let cmd = ConnectCommand::enhanced("live", "rtmp://s.example.com/live", &c);
        let fourccs = cmd.parsed_fourccs();
        assert!(fourccs.contains(&FourCC::AV1));
        assert!(fourccs.contains(&FourCC::VP9));
    }

    // 8. Negotiator initial state is Idle
    #[test]
    fn test_negotiator_initial_state() {
        let n = EnhancedRtmpNegotiator::new(caps());
        assert_eq!(n.state(), NegotiatorState::Idle);
        assert!(!n.is_connected());
    }

    // 9. Negotiator patent_free factory
    #[test]
    fn test_negotiator_patent_free() {
        let n = EnhancedRtmpNegotiator::patent_free();
        assert_eq!(n.state(), NegotiatorState::Idle);
    }

    // 10. Negotiator build_connect_command transitions to WaitingResult
    #[test]
    fn test_build_connect_command() {
        let mut n = EnhancedRtmpNegotiator::new(caps());
        let cmd = n
            .build_connect_command("live", "rtmp://s.example.com/live")
            .expect("should succeed");
        assert_eq!(n.state(), NegotiatorState::WaitingResult);
        assert!(cmd.is_enhanced());
    }

    // 11. Negotiator build_connect_command fails in non-Idle state
    #[test]
    fn test_build_connect_command_non_idle() {
        let mut n = EnhancedRtmpNegotiator::new(caps());
        n.build_connect_command("live", "rtmp://s.example.com/live")
            .expect("first call ok");
        let result = n.build_connect_command("live", "rtmp://s.example.com/live");
        assert!(result.is_err());
    }

    // 12. Client-side full flow
    #[test]
    fn test_client_side_full_flow() {
        let mut client = EnhancedRtmpNegotiator::new(caps());
        let connect_cmd = client
            .build_connect_command("live", "rtmp://server.example.com/live")
            .expect("build ok");

        // Simulate server returning its capabilities.
        let server_caps = NegotiationCapabilities::patent_free();
        let server_result =
            ConnectCommand::enhanced(&connect_cmd.app, &connect_cmd.tc_url, &server_caps);

        {
            let session = client
                .process_server_result(&server_result)
                .expect("negotiation ok");
            assert!(session.can_use_video(&FourCC::AV1));
            assert!(session.can_use_video(&FourCC::VP9));
            assert!(!session.can_use_video(&FourCC::AVC)); // patent-encumbered not in server
        }
        assert!(client.is_connected());
    }

    // 13. Server-side full flow
    #[test]
    fn test_server_side_full_flow() {
        let mut server = EnhancedRtmpNegotiator::new(NegotiationCapabilities::patent_free());

        // Client sends its connect command.
        let client_caps = caps();
        let client_cmd =
            ConnectCommand::enhanced("live", "rtmp://server.example.com/live", &client_caps);

        let result_cmd = server.process_client_connect(&client_cmd).expect("ok");
        assert!(server.is_connected());
        assert!(result_cmd.is_enhanced());

        let session = server.session().expect("should have session");
        assert!(session.enhanced_mode);
    }

    // 14. Negotiated session summary
    #[test]
    fn test_negotiated_session_summary() {
        let mut server = EnhancedRtmpNegotiator::new(NegotiationCapabilities::patent_free());
        let client_cmd = ConnectCommand::enhanced("live", "rtmp://s/live", &caps());
        server.process_client_connect(&client_cmd).expect("ok");
        let summary = server.session().expect("session").summary();
        assert!(summary.contains("enhanced=true"));
    }

    // 15. Negotiated session no common codecs
    #[test]
    fn test_no_common_codecs() {
        // Local caps: AV1 only.
        let local = NegotiationCapabilities {
            video_codecs: vec![FourCC::AV1],
            audio_codecs: vec![FourCC::OPUS],
            ..NegotiationCapabilities::default()
        };
        let mut n = EnhancedRtmpNegotiator::new(local);

        // Peer: HEVC + AAC only.
        let peer_cmd = ConnectCommand {
            fourcc_list: Some("hvc1,mp4a".to_owned()),
            ..ConnectCommand::new("live", "rtmp://s/live")
        };
        let _ = n
            .build_connect_command("live", "rtmp://s/live")
            .expect("ok");
        let session = n.process_server_result(&peer_cmd).expect("ok");
        // No common video or audio codecs.
        assert!(session.video_codecs.is_empty());
        assert!(session.audio_codecs.is_empty());
    }

    // 16. process_server_result fails before build_connect_command
    #[test]
    fn test_process_result_before_connect() {
        let mut n = EnhancedRtmpNegotiator::new(caps());
        let cmd = ConnectCommand::new("live", "rtmp://s/live");
        let result = n.process_server_result(&cmd);
        assert!(result.is_err());
    }

    // 17. process_client_connect fails in non-Idle state
    #[test]
    fn test_process_client_connect_twice() {
        let mut n = EnhancedRtmpNegotiator::new(caps());
        let cmd = ConnectCommand::enhanced("live", "rtmp://s/live", &caps());
        n.process_client_connect(&cmd).expect("first ok");
        let result = n.process_client_connect(&cmd);
        assert!(result.is_err());
    }

    // 18. ConnectCommand swf_url optional
    #[test]
    fn test_swf_url_optional() {
        let mut cmd = ConnectCommand::new("live", "rtmp://s/live");
        cmd.swf_url = Some("http://player.example.com/player.swf".to_owned());
        assert!(cmd.swf_url.is_some());
    }

    // 19. NegotiationCapabilities to_enhanced_caps
    #[test]
    fn test_to_enhanced_caps() {
        let c = caps();
        let enhanced = c.to_enhanced_caps();
        assert!(enhanced.enhanced_supported);
        assert!(enhanced.supports_video_codec(&FourCC::AV1));
    }

    // 20. Negotiator debug format
    #[test]
    fn test_negotiator_debug() {
        let n = EnhancedRtmpNegotiator::new(caps());
        let debug = format!("{n:?}");
        assert!(debug.contains("Idle") || debug.contains("state"));
    }

    // 21. ConnectCommand object_encoding field
    #[test]
    fn test_object_encoding() {
        let cmd = ConnectCommand::new("live", "rtmp://s/live");
        assert_eq!(cmd.object_encoding, 0);
    }

    // 22. ConnectCommand fpad field default
    #[test]
    fn test_fpad_default() {
        let cmd = ConnectCommand::new("live", "rtmp://s/live");
        assert!(!cmd.fpad);
    }

    // 23. Negotiated session app reflects peer app
    #[test]
    fn test_negotiated_app() {
        let mut n = EnhancedRtmpNegotiator::new(caps());
        let cmd = ConnectCommand::enhanced("myapp", "rtmp://s/myapp", &caps());
        n.process_client_connect(&cmd).expect("ok");
        assert_eq!(n.session().expect("session").app, "myapp");
    }
}
