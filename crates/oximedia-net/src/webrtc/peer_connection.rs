//! WebRTC Peer Connection implementation.
//!
//! This module provides the main PeerConnection API that ties together
//! ICE, DTLS, SCTP, and RTP/RTCP for complete WebRTC functionality.

#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]

use super::datachannel::{DataChannel, DataChannelConfig, DataChannelManager};
use super::dtls::{DtlsConfig, DtlsConnection, DtlsEndpoint, DtlsRole};
use super::ice::{IceCandidate, IceServer};
use super::ice_agent::{IceAgent, IceAgentConfig, IceConnectionState};
use super::rtcp::Packet as RtcpPacket;
use super::rtp::{Packet as RtpPacket, Session as RtpSession};
use super::sctp::Association;
use super::sdp::{Attribute, MediaDescription, MediaType, SessionDescription};
use crate::error::{NetError, NetResult};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;

/// Peer connection state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PeerConnectionState {
    /// New.
    New,
    /// Connecting.
    Connecting,
    /// Connected.
    Connected,
    /// Disconnected.
    Disconnected,
    /// Failed.
    Failed,
    /// Closed.
    Closed,
}

/// Signaling state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalingState {
    /// Stable.
    Stable,
    /// Have local offer.
    HaveLocalOffer,
    /// Have remote offer.
    HaveRemoteOffer,
    /// Have local answer.
    HaveLocalAnswer,
    /// Have remote answer.
    HaveRemoteAnswer,
    /// Closed.
    Closed,
}

/// SDP type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SdpType {
    /// Offer.
    Offer,
    /// Answer.
    Answer,
    /// Pranswer.
    Pranswer,
    /// Rollback.
    Rollback,
}

impl SdpType {
    /// Returns the string representation.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Offer => "offer",
            Self::Answer => "answer",
            Self::Pranswer => "pranswer",
            Self::Rollback => "rollback",
        }
    }
}

/// Session description.
#[derive(Debug, Clone)]
pub struct SessionDescriptionInit {
    /// SDP type.
    pub sdp_type: SdpType,
    /// SDP string.
    pub sdp: String,
}

impl SessionDescriptionInit {
    /// Creates a new session description.
    #[must_use]
    pub fn new(sdp_type: SdpType, sdp: impl Into<String>) -> Self {
        Self {
            sdp_type,
            sdp: sdp.into(),
        }
    }
}

/// Peer connection configuration.
#[derive(Debug, Clone, Default)]
pub struct PeerConnectionConfig {
    /// ICE servers.
    pub ice_servers: Vec<IceServer>,
    /// Bundle policy.
    pub bundle_policy: BundlePolicy,
    /// RTCP mux policy.
    pub rtcp_mux_policy: RtcpMuxPolicy,
}

impl PeerConnectionConfig {
    /// Creates a new configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds an ICE server.
    #[must_use]
    pub fn with_ice_server(mut self, server: IceServer) -> Self {
        self.ice_servers.push(server);
        self
    }
}

/// Bundle policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BundlePolicy {
    /// Balanced.
    Balanced,
    /// Max bundle.
    #[default]
    MaxBundle,
    /// Max compat.
    MaxCompat,
}

/// RTCP mux policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RtcpMuxPolicy {
    /// Require.
    #[default]
    Require,
}

/// Media track.
pub struct MediaTrack {
    /// Track ID.
    id: String,
    /// Track kind.
    kind: MediaType,
    /// RTP session.
    rtp_session: Arc<Mutex<RtpSession>>,
    /// RTCP sender.
    rtcp_tx: mpsc::UnboundedSender<RtcpPacket>,
}

impl MediaTrack {
    /// Creates a new media track.
    #[must_use]
    pub fn new(id: impl Into<String>, kind: MediaType, ssrc: u32) -> Self {
        let (rtcp_tx, _rtcp_rx) = mpsc::unbounded_channel();

        Self {
            id: id.into(),
            kind,
            rtp_session: Arc::new(Mutex::new(RtpSession::new(ssrc))),
            rtcp_tx,
        }
    }

    /// Gets the track ID.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Gets the track kind.
    #[must_use]
    pub const fn kind(&self) -> MediaType {
        self.kind
    }

    /// Sends RTP packet.
    pub async fn send_rtp(
        &self,
        payload_type: u8,
        timestamp: u32,
        payload: impl Into<bytes::Bytes>,
    ) -> NetResult<RtpPacket> {
        let packet = self
            .rtp_session
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .create_packet(payload_type, timestamp, payload);
        Ok(packet)
    }

    /// Sends RTCP packet.
    pub fn send_rtcp(&self, packet: RtcpPacket) -> NetResult<()> {
        self.rtcp_tx
            .send(packet)
            .map_err(|_| NetError::connection("RTCP channel closed"))?;
        Ok(())
    }

    /// Gets RTP statistics.
    #[must_use]
    pub fn stats(&self) -> super::rtp::Statistics {
        self.rtp_session
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .stats()
            .clone()
    }
}

/// Peer connection.
pub struct PeerConnection {
    /// Configuration.
    config: PeerConnectionConfig,
    /// Connection state.
    state: Arc<Mutex<PeerConnectionState>>,
    /// Signaling state.
    signaling_state: Arc<Mutex<SignalingState>>,
    /// ICE agent.
    ice_agent: Arc<Mutex<Option<IceAgent>>>,
    /// DTLS endpoint.
    dtls_endpoint: Arc<Mutex<Option<DtlsEndpoint>>>,
    /// DTLS connection.
    dtls_connection: Arc<Mutex<Option<Arc<DtlsConnection>>>>,
    /// SCTP association.
    sctp_association: Arc<Mutex<Option<Arc<Association>>>>,
    /// Data channel manager.
    dc_manager: Arc<Mutex<Option<DataChannelManager>>>,
    /// Media tracks.
    tracks: Arc<Mutex<Vec<Arc<MediaTrack>>>>,
    /// Local description.
    local_description: Arc<Mutex<Option<SessionDescription>>>,
    /// Remote description.
    remote_description: Arc<Mutex<Option<SessionDescription>>>,
    /// Pending local candidates.
    pending_local_candidates: Arc<Mutex<Vec<IceCandidate>>>,
}

impl PeerConnection {
    /// Creates a new peer connection.
    pub fn new(config: PeerConnectionConfig) -> NetResult<Self> {
        Ok(Self {
            config,
            state: Arc::new(Mutex::new(PeerConnectionState::New)),
            signaling_state: Arc::new(Mutex::new(SignalingState::Stable)),
            ice_agent: Arc::new(Mutex::new(None)),
            dtls_endpoint: Arc::new(Mutex::new(None)),
            dtls_connection: Arc::new(Mutex::new(None)),
            sctp_association: Arc::new(Mutex::new(None)),
            dc_manager: Arc::new(Mutex::new(None)),
            tracks: Arc::new(Mutex::new(Vec::new())),
            local_description: Arc::new(Mutex::new(None)),
            remote_description: Arc::new(Mutex::new(None)),
            pending_local_candidates: Arc::new(Mutex::new(Vec::new())),
        })
    }

    /// Creates an offer.
    pub async fn create_offer(&self) -> NetResult<SessionDescriptionInit> {
        *self
            .signaling_state
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = SignalingState::HaveLocalOffer;

        // Create ICE agent
        let ice_config = IceAgentConfig {
            ice_servers: self.config.ice_servers.clone(),
            controlling: true,
            ..Default::default()
        };

        let ice_agent = IceAgent::new(ice_config.clone());
        let local_candidates = ice_agent.gather_candidates().await?;

        *self.ice_agent.lock().unwrap_or_else(|e| e.into_inner()) = Some(ice_agent);
        *self
            .pending_local_candidates
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = local_candidates.clone();

        // Create DTLS config
        let dtls_config = DtlsConfig::new_self_signed(DtlsRole::Server)?;
        let fingerprint = dtls_config.fingerprint();

        // Build SDP
        let mut sdp = SessionDescription::new()
            .with_origin(format!("- {} 0 IN IP4 0.0.0.0", get_timestamp()))
            .with_session_name("WebRTC Session")
            .with_attribute(Attribute::new("group", "BUNDLE 0"));

        // Add data channel media
        let mut media = MediaDescription::data_channel(9)
            .with_format("webrtc-datachannel")
            .with_mid("0")
            .with_ice(ice_config.local_ufrag.clone(), ice_config.local_pwd.clone())
            .with_fingerprint(super::sdp::Fingerprint::new(
                fingerprint.algorithm,
                fingerprint.value,
            ))
            .with_rtcp_mux();

        media.setup = Some("actpass".to_string());

        // Add ICE candidates
        for candidate in &local_candidates {
            media
                .attributes
                .push(Attribute::new("candidate", candidate.to_sdp()));
        }

        sdp = sdp.with_media(media);

        let sdp_string = sdp.to_sdp();

        Ok(SessionDescriptionInit::new(SdpType::Offer, sdp_string))
    }

    /// Creates an answer.
    pub async fn create_answer(&self) -> NetResult<SessionDescriptionInit> {
        *self
            .signaling_state
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = SignalingState::HaveLocalAnswer;

        let remote_desc = self
            .remote_description
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let remote_desc = remote_desc
            .as_ref()
            .ok_or_else(|| NetError::invalid_state("No remote description"))?;

        // Extract remote ICE parameters
        let remote_media = remote_desc
            .media
            .first()
            .ok_or_else(|| NetError::protocol("No media in remote description"))?;

        let remote_ufrag = remote_media
            .ice_ufrag
            .clone()
            .ok_or_else(|| NetError::protocol("No ICE ufrag"))?;
        let remote_pwd = remote_media
            .ice_pwd
            .clone()
            .ok_or_else(|| NetError::protocol("No ICE pwd"))?;

        // Create ICE agent
        let ice_config = IceAgentConfig {
            ice_servers: self.config.ice_servers.clone(),
            controlling: false,
            remote_ufrag: Some(remote_ufrag),
            remote_pwd: Some(remote_pwd),
            ..Default::default()
        };

        let ice_agent = IceAgent::new(ice_config.clone());
        let local_candidates = ice_agent.gather_candidates().await?;

        *self.ice_agent.lock().unwrap_or_else(|e| e.into_inner()) = Some(ice_agent);
        *self
            .pending_local_candidates
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = local_candidates.clone();

        // Create DTLS config
        let dtls_config = DtlsConfig::new_self_signed(DtlsRole::Client)?;
        let fingerprint = dtls_config.fingerprint();

        // Build SDP answer
        let mut sdp = SessionDescription::new()
            .with_origin(format!("- {} 0 IN IP4 0.0.0.0", get_timestamp()))
            .with_session_name("WebRTC Session")
            .with_attribute(Attribute::new("group", "BUNDLE 0"));

        // Add data channel media
        let mut media = MediaDescription::data_channel(9)
            .with_format("webrtc-datachannel")
            .with_mid("0")
            .with_ice(ice_config.local_ufrag.clone(), ice_config.local_pwd.clone())
            .with_fingerprint(super::sdp::Fingerprint::new(
                fingerprint.algorithm,
                fingerprint.value,
            ))
            .with_rtcp_mux();

        media.setup = Some("active".to_string());

        // Add ICE candidates
        for candidate in &local_candidates {
            media
                .attributes
                .push(Attribute::new("candidate", candidate.to_sdp()));
        }

        sdp = sdp.with_media(media);

        let sdp_string = sdp.to_sdp();

        Ok(SessionDescriptionInit::new(SdpType::Answer, sdp_string))
    }

    /// Sets local description.
    pub async fn set_local_description(&self, desc: SessionDescriptionInit) -> NetResult<()> {
        let sdp = SessionDescription::parse(&desc.sdp)?;

        *self
            .local_description
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = Some(sdp);

        match desc.sdp_type {
            SdpType::Offer => {
                *self
                    .signaling_state
                    .lock()
                    .unwrap_or_else(|e| e.into_inner()) = SignalingState::HaveLocalOffer;
            }
            SdpType::Answer => {
                *self
                    .signaling_state
                    .lock()
                    .unwrap_or_else(|e| e.into_inner()) = SignalingState::Stable;
            }
            _ => {}
        }

        Ok(())
    }

    /// Sets remote description.
    pub async fn set_remote_description(&self, desc: SessionDescriptionInit) -> NetResult<()> {
        let sdp = SessionDescription::parse(&desc.sdp)?;

        // Extract remote ICE candidates
        for media in &sdp.media {
            for attr in &media.attributes {
                if attr.name == "candidate" {
                    if let Some(ref value) = attr.value {
                        if let Ok(candidate) = IceCandidate::parse(value) {
                            if let Some(ref ice_agent) =
                                *self.ice_agent.lock().unwrap_or_else(|e| e.into_inner())
                            {
                                ice_agent.add_remote_candidate(candidate);
                            }
                        }
                    }
                }
            }
        }

        *self
            .remote_description
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = Some(sdp);

        match desc.sdp_type {
            SdpType::Offer => {
                *self
                    .signaling_state
                    .lock()
                    .unwrap_or_else(|e| e.into_inner()) = SignalingState::HaveRemoteOffer;
            }
            SdpType::Answer => {
                *self
                    .signaling_state
                    .lock()
                    .unwrap_or_else(|e| e.into_inner()) = SignalingState::Stable;
                // Start connection process
                self.start_connection().await?;
            }
            _ => {}
        }

        Ok(())
    }

    /// Starts the connection process.
    async fn start_connection(&self) -> NetResult<()> {
        *self.state.lock().unwrap_or_else(|e| e.into_inner()) = PeerConnectionState::Connecting;

        // Perform ICE connectivity checks
        if let Some(ref ice_agent) = *self.ice_agent.lock().unwrap_or_else(|e| e.into_inner()) {
            ice_agent.check_connectivity().await?;

            if ice_agent.state() == IceConnectionState::Connected {
                // Get socket from ICE agent
                if let Some(socket) = ice_agent.socket() {
                    // Perform DTLS handshake
                    let dtls_config = DtlsConfig::new_self_signed(DtlsRole::Client)?;
                    let dtls_endpoint = DtlsEndpoint::new(dtls_config, socket);
                    let dtls_conn = dtls_endpoint.handshake().await?;

                    let dtls_conn = Arc::new(dtls_conn);
                    *self
                        .dtls_connection
                        .lock()
                        .unwrap_or_else(|e| e.into_inner()) = Some(dtls_conn.clone());

                    // Create SCTP association
                    let sctp_assoc = Arc::new(Association::new(5000, 5000));
                    *self
                        .sctp_association
                        .lock()
                        .unwrap_or_else(|e| e.into_inner()) = Some(sctp_assoc.clone());

                    // Create data channel manager
                    let dc_manager = DataChannelManager::new(sctp_assoc, dtls_conn);
                    *self.dc_manager.lock().unwrap_or_else(|e| e.into_inner()) = Some(dc_manager);

                    *self.state.lock().unwrap_or_else(|e| e.into_inner()) =
                        PeerConnectionState::Connected;
                }
            }
        }

        Ok(())
    }

    /// Adds an ICE candidate.
    pub async fn add_ice_candidate(&self, candidate: IceCandidate) -> NetResult<()> {
        if let Some(ref ice_agent) = *self.ice_agent.lock().unwrap_or_else(|e| e.into_inner()) {
            ice_agent.add_remote_candidate(candidate);
        }
        Ok(())
    }

    /// Creates a data channel.
    pub async fn create_data_channel(
        &self,
        label: impl Into<String>,
    ) -> NetResult<Arc<DataChannel>> {
        let config = DataChannelConfig::new(label);

        let dc_manager = self.dc_manager.lock().unwrap_or_else(|e| e.into_inner());
        let dc_manager = dc_manager
            .as_ref()
            .ok_or_else(|| NetError::invalid_state("Connection not established"))?;

        dc_manager.create_channel(config).await
    }

    /// Adds a media track.
    pub fn add_track(&self, track: Arc<MediaTrack>) {
        self.tracks
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(track);
    }

    /// Gets all tracks.
    #[must_use]
    pub fn tracks(&self) -> Vec<Arc<MediaTrack>> {
        self.tracks
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Gets the connection state.
    #[must_use]
    pub fn state(&self) -> PeerConnectionState {
        *self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Gets the signaling state.
    #[must_use]
    pub fn signaling_state(&self) -> SignalingState {
        *self
            .signaling_state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    }

    /// Gets the local description.
    #[must_use]
    pub fn local_description(&self) -> Option<SessionDescription> {
        self.local_description
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Gets the remote description.
    #[must_use]
    pub fn remote_description(&self) -> Option<SessionDescription> {
        self.remote_description
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Closes the peer connection.
    pub async fn close(&self) -> NetResult<()> {
        *self.state.lock().unwrap_or_else(|e| e.into_inner()) = PeerConnectionState::Closed;
        *self
            .signaling_state
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = SignalingState::Closed;

        Ok(())
    }
}

/// Gets the current timestamp in seconds.
fn get_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sdp_type() {
        assert_eq!(SdpType::Offer.as_str(), "offer");
        assert_eq!(SdpType::Answer.as_str(), "answer");
    }

    #[test]
    fn test_peer_connection_new() {
        let config = PeerConnectionConfig::new();
        let pc = PeerConnection::new(config).expect("should succeed in test");

        assert_eq!(pc.state(), PeerConnectionState::New);
        assert_eq!(pc.signaling_state(), SignalingState::Stable);
    }

    #[test]
    fn test_peer_connection_config() {
        let config = PeerConnectionConfig::new()
            .with_ice_server(IceServer::stun("stun:stun.example.com:3478"));

        assert_eq!(config.ice_servers.len(), 1);
    }

    #[test]
    fn test_media_track() {
        let track = MediaTrack::new("track1", MediaType::Audio, 12345);
        assert_eq!(track.id(), "track1");
        assert_eq!(track.kind(), MediaType::Audio);
    }

    #[test]
    fn test_session_description_init() {
        let desc = SessionDescriptionInit::new(SdpType::Offer, "v=0\r\n");
        assert_eq!(desc.sdp_type, SdpType::Offer);
        assert_eq!(desc.sdp, "v=0\r\n");
    }
}
