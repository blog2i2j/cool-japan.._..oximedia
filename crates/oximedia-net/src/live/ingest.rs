//! Stream ingest from RTMP and SRT sources.
//!
//! This module handles ingesting live streams from various sources
//! and converting them into media packets for the live server.

use super::{LiveStream, MediaPacket, MediaType, StreamRegistry};
use crate::error::{NetError, NetResult};
use crate::rtmp::{RtmpServer, RtmpServerConfig};
use bytes::Bytes;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::Arc;
use tokio::sync::mpsc;

/// Ingest source type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IngestSource {
    /// RTMP ingest.
    Rtmp,

    /// SRT ingest.
    Srt,

    /// WebRTC ingest.
    WebRtc,

    /// Direct packet injection.
    Direct,
}

/// Ingest configuration.
#[derive(Debug, Clone)]
pub struct IngestConfig {
    /// Enable RTMP ingest.
    pub enable_rtmp: bool,

    /// RTMP bind address.
    pub rtmp_bind_addr: SocketAddr,

    /// Enable SRT ingest.
    pub enable_srt: bool,

    /// SRT bind address.
    pub srt_bind_addr: SocketAddr,

    /// Enable WebRTC ingest.
    pub enable_webrtc: bool,

    /// WebRTC bind address.
    pub webrtc_bind_addr: SocketAddr,

    /// Maximum concurrent ingest sessions.
    pub max_sessions: usize,

    /// Buffer size for ingest packets.
    pub buffer_size: usize,
}

impl Default for IngestConfig {
    fn default() -> Self {
        Self {
            enable_rtmp: true,
            rtmp_bind_addr: SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), 1935),
            enable_srt: true,
            srt_bind_addr: SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), 9998),
            enable_webrtc: false,
            webrtc_bind_addr: SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), 8443),
            max_sessions: 100,
            buffer_size: 1000,
        }
    }
}

/// Ingest session state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IngestSessionState {
    /// Session is connecting.
    Connecting,

    /// Session is active.
    Active,

    /// Session is stopping.
    Stopping,

    /// Session has stopped.
    Stopped,

    /// Session encountered an error.
    Error,
}

/// Ingest session.
pub struct IngestSession {
    /// Session ID.
    pub session_id: String,

    /// Source type.
    pub source: IngestSource,

    /// Stream key.
    pub stream_key: String,

    /// Application name.
    pub app_name: String,

    /// Session state.
    state: RwLock<IngestSessionState>,

    /// Packet sender to live stream.
    packet_tx: mpsc::UnboundedSender<MediaPacket>,

    /// Associated live stream.
    live_stream: Arc<LiveStream>,

    /// Bytes ingested.
    bytes_ingested: RwLock<u64>,

    /// Packets ingested.
    packets_ingested: RwLock<u64>,
}

impl IngestSession {
    /// Creates a new ingest session.
    pub fn new(
        session_id: impl Into<String>,
        source: IngestSource,
        stream_key: impl Into<String>,
        app_name: impl Into<String>,
        live_stream: Arc<LiveStream>,
    ) -> Self {
        let (packet_tx, mut packet_rx) = mpsc::unbounded_channel();

        // Spawn packet forwarder
        let stream = Arc::clone(&live_stream);
        tokio::spawn(async move {
            while let Some(packet) = packet_rx.recv().await {
                if let Err(e) = stream.publish(packet) {
                    eprintln!("Failed to publish packet: {e}");
                    break;
                }
            }
        });

        Self {
            session_id: session_id.into(),
            source,
            stream_key: stream_key.into(),
            app_name: app_name.into(),
            state: RwLock::new(IngestSessionState::Connecting),
            packet_tx,
            live_stream,
            bytes_ingested: RwLock::new(0),
            packets_ingested: RwLock::new(0),
        }
    }

    /// Ingests a media packet.
    pub fn ingest_packet(&self, packet: MediaPacket) -> NetResult<()> {
        *self.bytes_ingested.write() += packet.data.len() as u64;
        *self.packets_ingested.write() += 1;

        self.packet_tx
            .send(packet)
            .map_err(|e| NetError::connection(format!("Failed to send packet: {e}")))?;

        Ok(())
    }

    /// Sets session state.
    pub fn set_state(&self, state: IngestSessionState) {
        *self.state.write() = state;

        if state == IngestSessionState::Active {
            self.live_stream.set_state(super::StreamState::Active);
        } else if state == IngestSessionState::Stopped {
            self.live_stream.set_state(super::StreamState::Stopped);
        }
    }

    /// Gets session state.
    #[must_use]
    pub fn state(&self) -> IngestSessionState {
        *self.state.read()
    }

    /// Gets bytes ingested.
    #[must_use]
    pub fn bytes_ingested(&self) -> u64 {
        *self.bytes_ingested.read()
    }

    /// Gets packets ingested.
    #[must_use]
    pub fn packets_ingested(&self) -> u64 {
        *self.packets_ingested.read()
    }

    /// Stops the session.
    pub fn stop(&self) {
        self.set_state(IngestSessionState::Stopping);
        self.set_state(IngestSessionState::Stopped);
    }
}

/// Ingest server managing all ingest sources.
pub struct IngestServer {
    /// Configuration.
    config: IngestConfig,

    /// Stream registry.
    registry: Arc<StreamRegistry>,

    /// Active sessions.
    sessions: RwLock<HashMap<String, Arc<IngestSession>>>,

    /// RTMP server.
    rtmp_server: Option<Arc<RtmpServer>>,
}

impl IngestServer {
    /// Creates a new ingest server.
    #[must_use]
    pub fn new(config: IngestConfig, registry: Arc<StreamRegistry>) -> Self {
        Self {
            config,
            registry,
            sessions: RwLock::new(HashMap::new()),
            rtmp_server: None,
        }
    }

    /// Starts the ingest server.
    ///
    /// # Errors
    ///
    /// Returns an error if any ingest source fails to start.
    pub async fn start(&mut self) -> NetResult<()> {
        // Start RTMP ingest
        if self.config.enable_rtmp {
            self.start_rtmp_ingest().await?;
        }

        // Start SRT ingest
        if self.config.enable_srt {
            self.start_srt_ingest().await?;
        }

        // Start WebRTC ingest
        if self.config.enable_webrtc {
            self.start_webrtc_ingest().await?;
        }

        Ok(())
    }

    /// Starts RTMP ingest.
    async fn start_rtmp_ingest(&mut self) -> NetResult<()> {
        let config = RtmpServerConfig {
            bind_address: self.config.rtmp_bind_addr.to_string(),
            ..Default::default()
        };

        // Create RTMP server with custom auth handler
        let sessions_clone: Arc<RwLock<HashMap<String, Arc<IngestSession>>>> =
            Arc::new(RwLock::new(self.sessions.read().clone()));

        let auth_handler = Arc::new(IngestAuthHandler {
            registry: Arc::clone(&self.registry),
            sessions: sessions_clone,
        });

        let rtmp_server = Arc::new(RtmpServer::new(config, auth_handler));
        self.rtmp_server = Some(rtmp_server);

        // Start RTMP server
        // In production, this would actually start the server
        // For now, we'll skip the actual server start to avoid blocking

        Ok(())
    }

    /// Starts SRT ingest.
    async fn start_srt_ingest(&self) -> NetResult<()> {
        // SRT ingest implementation
        // This would start an SRT server and handle incoming streams
        Ok(())
    }

    /// Starts WebRTC ingest.
    async fn start_webrtc_ingest(&self) -> NetResult<()> {
        // WebRTC ingest implementation
        // This would start a WebRTC signaling server
        Ok(())
    }

    /// Creates a new ingest session.
    pub fn create_session(
        &self,
        source: IngestSource,
        stream_key: impl Into<String>,
        app_name: impl Into<String>,
    ) -> NetResult<Arc<IngestSession>> {
        let stream_key = stream_key.into();
        let app_name = app_name.into();

        // Check session limit
        {
            let sessions = self.sessions.read();
            if sessions.len() >= self.config.max_sessions {
                return Err(NetError::invalid_state("Maximum session limit reached"));
            }
        }

        // Register or get live stream
        let live_stream = self.registry.register_stream(&stream_key, &app_name)?;

        // Create session
        let session_id = format!("{source:?}_{app_name}_{stream_key}");
        let session = Arc::new(IngestSession::new(
            &session_id,
            source,
            &stream_key,
            &app_name,
            live_stream,
        ));

        // Register session
        {
            let mut sessions = self.sessions.write();
            sessions.insert(session_id, session.clone());
        }

        Ok(session)
    }

    /// Removes an ingest session.
    pub fn remove_session(&self, session_id: &str) {
        let mut sessions = self.sessions.write();
        if let Some(session) = sessions.remove(session_id) {
            session.stop();
        }
    }

    /// Gets an ingest session.
    #[must_use]
    pub fn get_session(&self, session_id: &str) -> Option<Arc<IngestSession>> {
        let sessions = self.sessions.read();
        sessions.get(session_id).cloned()
    }

    /// Lists all active sessions.
    #[must_use]
    pub fn list_sessions(&self) -> Vec<String> {
        let sessions = self.sessions.read();
        sessions.keys().cloned().collect()
    }

    /// Returns session count.
    #[must_use]
    pub fn session_count(&self) -> usize {
        let sessions = self.sessions.read();
        sessions.len()
    }

    /// Stops the ingest server.
    pub fn stop(&self) {
        let sessions = self.sessions.read();
        for session in sessions.values() {
            session.stop();
        }
    }
}

/// Custom RTMP authentication handler for ingest.
struct IngestAuthHandler {
    registry: Arc<StreamRegistry>,
    sessions: Arc<RwLock<HashMap<String, Arc<IngestSession>>>>,
}

#[async_trait::async_trait]
impl crate::rtmp::AuthHandler for IngestAuthHandler {
    async fn authenticate_connect(
        &self,
        _app: &str,
        _tc_url: &str,
        _params: &HashMap<String, String>,
    ) -> crate::rtmp::AuthResult {
        crate::rtmp::AuthResult::Success
    }

    async fn authenticate_publish(
        &self,
        app: &str,
        stream_key: &str,
        _publish_type: crate::rtmp::PublishType,
    ) -> crate::rtmp::AuthResult {
        // Create ingest session
        match self.registry.register_stream(stream_key, app) {
            Ok(_live_stream) => crate::rtmp::AuthResult::Success,
            Err(e) => crate::rtmp::AuthResult::Failed(e.to_string()),
        }
    }

    async fn authenticate_play(&self, _app: &str, _stream_key: &str) -> crate::rtmp::AuthResult {
        crate::rtmp::AuthResult::Failed("Playback not supported on ingest server".to_string())
    }
}

/// Converts RTMP media packets to live stream packets.
pub fn convert_rtmp_packet(
    media_type: crate::rtmp::MediaPacketType,
    timestamp: u32,
    data: Bytes,
) -> MediaPacket {
    let packet_type = match media_type {
        crate::rtmp::MediaPacketType::Video => MediaType::Video,
        crate::rtmp::MediaPacketType::Audio => MediaType::Audio,
        crate::rtmp::MediaPacketType::Data => MediaType::Metadata,
    };

    MediaPacket::new(packet_type, u64::from(timestamp), data)
}
