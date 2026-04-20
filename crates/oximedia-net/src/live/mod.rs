//! Live HLS/DASH streaming server.
//!
//! This module provides a production-ready live streaming server supporting:
//! - HLS (HTTP Live Streaming) with LL-HLS (Low Latency)
//! - DASH (Dynamic Adaptive Streaming over HTTP)
//! - RTMP/SRT ingest
//! - Real-time transcoding and adaptive bitrate
//! - DVR/time-shifting functionality
//! - Authentication and access control
//! - Cluster support and CDN integration
//!
//! # Example
//!
//! ```ignore
//! use oximedia_net::live::{LiveServer, LiveServerConfig};
//! use std::net::SocketAddr;
//! use std::time::Duration;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = LiveServerConfig {
//!         bind_addr: "0.0.0.0:8080".parse()?,
//!         segment_duration: Duration::from_secs(2),
//!         dvr_window: Duration::from_secs(3600),
//!         ..Default::default()
//!     };
//!
//!     let server = LiveServer::new(config);
//!     server.start().await?;
//!
//!     Ok(())
//! }
//! ```

#![allow(dead_code)]

pub mod analytics;
pub mod auth;
pub mod cluster;
pub mod dash;
pub mod dvr;
pub mod hls;
pub mod ingest;
pub mod segment;
pub mod thumbnail;

use crate::error::{NetError, NetResult};
use bytes::Bytes;
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{broadcast, mpsc};
use uuid::Uuid;

pub use analytics::{Analytics, StreamMetrics, ViewerMetrics};
pub use auth::{AuthHandler, AuthResult, TokenValidator};
pub use cluster::{ClusterConfig, ClusterNode, NodeState};
pub use dash::server::DashServer;
pub use dvr::{DvrBuffer, DvrConfig};
pub use hls::server::HlsServer;
pub use ingest::{IngestConfig, IngestServer, IngestSource};
pub use segment::{MediaSegment, SegmentConfig, SegmentGenerator};
pub use thumbnail::ThumbnailGenerator;

/// Live server configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiveServerConfig {
    /// HTTP server bind address.
    pub bind_addr: SocketAddr,

    /// Segment duration for HLS/DASH.
    pub segment_duration: Duration,

    /// DVR window duration.
    pub dvr_window: Duration,

    /// Maximum number of concurrent streams.
    pub max_streams: usize,

    /// Enable HLS server.
    pub enable_hls: bool,

    /// Enable DASH server.
    pub enable_dash: bool,

    /// Enable LL-HLS (Low Latency HLS).
    pub enable_ll_hls: bool,

    /// Enable low latency DASH.
    pub enable_ll_dash: bool,

    /// HLS segment count in playlist.
    pub hls_segment_count: usize,

    /// DASH segment availability duration.
    pub dash_availability_duration: Duration,

    /// Enable DVR/time-shifting.
    pub enable_dvr: bool,

    /// Enable authentication.
    pub enable_auth: bool,

    /// Enable analytics.
    pub enable_analytics: bool,

    /// Enable thumbnail generation.
    pub enable_thumbnails: bool,

    /// Thumbnail interval.
    pub thumbnail_interval: Duration,

    /// CORS allowed origins.
    pub cors_origins: Vec<String>,

    /// Enable cluster mode.
    pub enable_cluster: bool,

    /// Cluster configuration.
    pub cluster_config: Option<ClusterConfig>,
}

impl Default for LiveServerConfig {
    fn default() -> Self {
        Self {
            bind_addr: SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), 8080),
            segment_duration: Duration::from_secs(2),
            dvr_window: Duration::from_secs(3600),
            max_streams: 1000,
            enable_hls: true,
            enable_dash: true,
            enable_ll_hls: true,
            enable_ll_dash: true,
            hls_segment_count: 6,
            dash_availability_duration: Duration::from_secs(60),
            enable_dvr: true,
            enable_auth: false,
            enable_analytics: true,
            enable_thumbnails: true,
            thumbnail_interval: Duration::from_secs(10),
            cors_origins: vec!["*".to_string()],
            enable_cluster: false,
            cluster_config: None,
        }
    }
}

/// Stream quality variant.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QualityVariant {
    /// Variant ID (e.g., "360p", "720p", "1080p").
    pub id: String,

    /// Bandwidth in bits per second.
    pub bandwidth: u64,

    /// Width in pixels.
    pub width: u32,

    /// Height in pixels.
    pub height: u32,

    /// Frame rate.
    pub framerate: f64,

    /// Video codec.
    pub video_codec: String,

    /// Audio codec.
    pub audio_codec: String,

    /// Video bitrate.
    pub video_bitrate: u64,

    /// Audio bitrate.
    pub audio_bitrate: u64,
}

impl QualityVariant {
    /// Creates a new quality variant.
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        width: u32,
        height: u32,
        video_bitrate: u64,
        audio_bitrate: u64,
    ) -> Self {
        let bandwidth = video_bitrate + audio_bitrate;
        Self {
            id: id.into(),
            bandwidth,
            width,
            height,
            framerate: 30.0,
            video_codec: "avc1.4d401f".to_string(),
            audio_codec: "mp4a.40.2".to_string(),
            video_bitrate,
            audio_bitrate,
        }
    }

    /// Returns a predefined set of quality variants.
    #[must_use]
    pub fn standard_variants() -> Vec<Self> {
        vec![
            Self::new("1080p", 1920, 1080, 4_500_000, 128_000),
            Self::new("720p", 1280, 720, 2_500_000, 128_000),
            Self::new("480p", 854, 480, 1_400_000, 96_000),
            Self::new("360p", 640, 360, 800_000, 96_000),
            Self::new("240p", 426, 240, 400_000, 64_000),
        ]
    }
}

/// Media type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MediaType {
    /// Video media.
    Video,
    /// Audio media.
    Audio,
    /// Metadata/subtitles.
    Metadata,
}

/// Media packet for streaming.
#[derive(Debug, Clone)]
pub struct MediaPacket {
    /// Packet media type.
    pub media_type: MediaType,

    /// Timestamp in milliseconds.
    pub timestamp: u64,

    /// Presentation timestamp.
    pub pts: u64,

    /// Decode timestamp.
    pub dts: u64,

    /// Duration in milliseconds.
    pub duration: u64,

    /// Is keyframe (for video).
    pub keyframe: bool,

    /// Payload data.
    pub data: Bytes,

    /// Quality variant ID.
    pub variant_id: Option<String>,
}

impl MediaPacket {
    /// Creates a new media packet.
    #[must_use]
    pub fn new(media_type: MediaType, timestamp: u64, data: Bytes) -> Self {
        Self {
            media_type,
            timestamp,
            pts: timestamp,
            dts: timestamp,
            duration: 0,
            keyframe: false,
            data,
            variant_id: None,
        }
    }

    /// Sets keyframe flag.
    #[must_use]
    pub const fn with_keyframe(mut self, keyframe: bool) -> Self {
        self.keyframe = keyframe;
        self
    }

    /// Sets variant ID.
    #[must_use]
    pub fn with_variant(mut self, variant_id: impl Into<String>) -> Self {
        self.variant_id = Some(variant_id.into());
        self
    }
}

/// Stream state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StreamState {
    /// Stream is initializing.
    Initializing,

    /// Stream is active and publishing.
    Active,

    /// Stream is paused.
    Paused,

    /// Stream is stopping.
    Stopping,

    /// Stream has stopped.
    Stopped,

    /// Stream encountered an error.
    Error,
}

/// Live stream information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamInfo {
    /// Unique stream ID.
    pub id: Uuid,

    /// Stream key/name.
    pub stream_key: String,

    /// Application name.
    pub app_name: String,

    /// Stream state.
    pub state: StreamState,

    /// Available quality variants.
    pub variants: Vec<QualityVariant>,

    /// Stream start time.
    pub start_time: DateTime<Utc>,

    /// Stream end time (if stopped).
    pub end_time: Option<DateTime<Utc>>,

    /// Current viewer count.
    pub viewer_count: u64,

    /// Peak viewer count.
    pub peak_viewer_count: u64,

    /// Total bytes ingested.
    pub bytes_ingested: u64,

    /// Total bytes served.
    pub bytes_served: u64,

    /// Metadata.
    pub metadata: HashMap<String, String>,
}

impl StreamInfo {
    /// Creates a new stream info.
    #[must_use]
    pub fn new(stream_key: impl Into<String>, app_name: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            stream_key: stream_key.into(),
            app_name: app_name.into(),
            state: StreamState::Initializing,
            variants: QualityVariant::standard_variants(),
            start_time: Utc::now(),
            end_time: None,
            viewer_count: 0,
            peak_viewer_count: 0,
            bytes_ingested: 0,
            bytes_served: 0,
            metadata: HashMap::new(),
        }
    }

    /// Returns the stream key path.
    #[must_use]
    pub fn key_path(&self) -> String {
        format!("{}/{}", self.app_name, self.stream_key)
    }
}

/// Active live stream.
pub struct LiveStream {
    /// Stream information.
    info: RwLock<StreamInfo>,

    /// Media packet broadcaster.
    media_tx: broadcast::Sender<MediaPacket>,

    /// DVR buffer.
    dvr_buffer: Option<Arc<RwLock<DvrBuffer>>>,

    /// Segment generator.
    segment_generator: Arc<segment::SegmentGenerator>,

    /// Analytics tracker.
    analytics: Option<Arc<Analytics>>,

    /// Thumbnail generator.
    thumbnail_gen: Option<Arc<ThumbnailGenerator>>,
}

impl LiveStream {
    /// Creates a new live stream.
    pub fn new(
        stream_key: impl Into<String>,
        app_name: impl Into<String>,
        config: &LiveServerConfig,
    ) -> Self {
        let info = StreamInfo::new(stream_key, app_name);
        let (media_tx, _) = broadcast::channel(1000);

        let dvr_buffer = if config.enable_dvr {
            Some(Arc::new(RwLock::new(DvrBuffer::new(DvrConfig {
                window_duration: config.dvr_window,
                segment_duration: config.segment_duration,
            }))))
        } else {
            None
        };

        let segment_config = SegmentConfig {
            duration: config.segment_duration,
            keyframe_interval: 60,
        };

        let segment_generator = Arc::new(segment::SegmentGenerator::new(segment_config));

        let analytics = if config.enable_analytics {
            Some(Arc::new(Analytics::new(info.id)))
        } else {
            None
        };

        let thumbnail_gen = if config.enable_thumbnails {
            Some(Arc::new(ThumbnailGenerator::new(config.thumbnail_interval)))
        } else {
            None
        };

        Self {
            info: RwLock::new(info),
            media_tx,
            dvr_buffer,
            segment_generator,
            analytics,
            thumbnail_gen,
        }
    }

    /// Returns stream information.
    #[must_use]
    pub fn info(&self) -> StreamInfo {
        self.info.read().clone()
    }

    /// Subscribes to media packets.
    #[must_use]
    pub fn subscribe(&self) -> broadcast::Receiver<MediaPacket> {
        self.media_tx.subscribe()
    }

    /// Publishes a media packet.
    pub fn publish(&self, packet: MediaPacket) -> NetResult<()> {
        // Update analytics
        if let Some(analytics) = &self.analytics {
            analytics.record_packet(&packet);
        }

        // Add to DVR buffer
        if let Some(dvr) = &self.dvr_buffer {
            dvr.write().add_packet(packet.clone());
        }

        // Generate segment
        self.segment_generator.add_packet(&packet)?;

        // Generate thumbnail
        if let Some(thumb_gen) = &self.thumbnail_gen {
            if packet.media_type == MediaType::Video && packet.keyframe {
                thumb_gen.generate_from_packet(&packet);
            }
        }

        // Broadcast to subscribers
        let _ = self.media_tx.send(packet);

        // Update stats
        let mut info = self.info.write();
        info.bytes_ingested += 1;

        Ok(())
    }

    /// Updates stream state.
    pub fn set_state(&self, state: StreamState) {
        let mut info = self.info.write();
        info.state = state;
        if state == StreamState::Stopped {
            info.end_time = Some(Utc::now());
        }
    }

    /// Increments viewer count.
    pub fn add_viewer(&self) {
        let mut info = self.info.write();
        info.viewer_count += 1;
        if info.viewer_count > info.peak_viewer_count {
            info.peak_viewer_count = info.viewer_count;
        }
    }

    /// Decrements viewer count.
    pub fn remove_viewer(&self) {
        let mut info = self.info.write();
        if info.viewer_count > 0 {
            info.viewer_count -= 1;
        }
    }

    /// Returns DVR buffer if enabled.
    #[must_use]
    pub fn dvr_buffer(&self) -> Option<Arc<RwLock<DvrBuffer>>> {
        self.dvr_buffer.clone()
    }

    /// Returns analytics if enabled.
    #[must_use]
    pub fn analytics(&self) -> Option<Arc<Analytics>> {
        self.analytics.clone()
    }
}

/// Stream registry managing all active streams.
pub struct StreamRegistry {
    /// Active streams (key: "app/stream_key").
    streams: RwLock<HashMap<String, Arc<LiveStream>>>,

    /// Configuration.
    config: LiveServerConfig,
}

impl StreamRegistry {
    /// Creates a new stream registry.
    #[must_use]
    pub fn new(config: LiveServerConfig) -> Self {
        Self {
            streams: RwLock::new(HashMap::new()),
            config,
        }
    }

    /// Registers a new stream.
    pub fn register_stream(
        &self,
        stream_key: impl Into<String>,
        app_name: impl Into<String>,
    ) -> NetResult<Arc<LiveStream>> {
        let stream_key = stream_key.into();
        let app_name = app_name.into();
        let key_path = format!("{app_name}/{stream_key}");

        let mut streams = self.streams.write();

        if streams.len() >= self.config.max_streams {
            return Err(NetError::invalid_state("Maximum stream limit reached"));
        }

        if streams.contains_key(&key_path) {
            return Err(NetError::invalid_state(format!(
                "Stream already exists: {key_path}"
            )));
        }

        let stream = Arc::new(LiveStream::new(&stream_key, &app_name, &self.config));
        streams.insert(key_path, stream.clone());

        Ok(stream)
    }

    /// Unregisters a stream.
    pub fn unregister_stream(&self, app_name: &str, stream_key: &str) {
        let key_path = format!("{app_name}/{stream_key}");
        let mut streams = self.streams.write();

        if let Some(stream) = streams.remove(&key_path) {
            stream.set_state(StreamState::Stopped);
        }
    }

    /// Gets a stream.
    #[must_use]
    pub fn get_stream(&self, app_name: &str, stream_key: &str) -> Option<Arc<LiveStream>> {
        let key_path = format!("{app_name}/{stream_key}");
        let streams = self.streams.read();
        streams.get(&key_path).cloned()
    }

    /// Lists all streams.
    #[must_use]
    pub fn list_streams(&self) -> Vec<StreamInfo> {
        let streams = self.streams.read();
        streams.values().map(|s| s.info()).collect()
    }

    /// Returns the number of active streams.
    #[must_use]
    pub fn stream_count(&self) -> usize {
        let streams = self.streams.read();
        streams.len()
    }
}

/// Main live streaming server.
pub struct LiveServer {
    /// Configuration.
    config: LiveServerConfig,

    /// Stream registry.
    registry: Arc<StreamRegistry>,

    /// Ingest server.
    ingest_server: Option<Arc<IngestServer>>,

    /// HLS server.
    hls_server: Option<Arc<HlsServer>>,

    /// DASH server.
    dash_server: Option<Arc<DashServer>>,

    /// Authentication handler.
    auth_handler: Option<Arc<dyn AuthHandler>>,

    /// Cluster node (if enabled).
    cluster_node: Option<Arc<ClusterNode>>,

    /// Shutdown signal.
    shutdown_tx: mpsc::Sender<()>,
    shutdown_rx: mpsc::Receiver<()>,
}

impl LiveServer {
    /// Creates a new live server.
    #[must_use]
    pub fn new(config: LiveServerConfig) -> Self {
        let registry = Arc::new(StreamRegistry::new(config.clone()));
        let (shutdown_tx, shutdown_rx) = mpsc::channel(1);

        Self {
            config,
            registry,
            ingest_server: None,
            hls_server: None,
            dash_server: None,
            auth_handler: None,
            cluster_node: None,
            shutdown_tx,
            shutdown_rx,
        }
    }

    /// Sets authentication handler.
    pub fn with_auth_handler(mut self, handler: Arc<dyn AuthHandler>) -> Self {
        self.auth_handler = Some(handler);
        self
    }

    /// Starts the live server.
    ///
    /// # Errors
    ///
    /// Returns an error if the server fails to start.
    pub async fn start(mut self) -> NetResult<()> {
        // Initialize cluster if enabled
        if self.config.enable_cluster {
            if let Some(cluster_config) = &self.config.cluster_config {
                let node = Arc::new(ClusterNode::new(cluster_config.clone()).await?);
                node.start().await?;
                self.cluster_node = Some(node);
            }
        }

        // Start HLS server
        if self.config.enable_hls {
            let hls_server = Arc::new(HlsServer::new(
                self.config.clone(),
                Arc::clone(&self.registry),
            ));
            self.hls_server = Some(hls_server);
        }

        // Start DASH server
        if self.config.enable_dash {
            let dash_server = Arc::new(DashServer::new(
                self.config.clone(),
                Arc::clone(&self.registry),
            ));
            self.dash_server = Some(dash_server);
        }

        // Start HTTP server
        self.start_http_server().await?;

        Ok(())
    }

    /// Starts the HTTP server.
    async fn start_http_server(&mut self) -> NetResult<()> {
        use hyper::server::conn::http1;
        use hyper::service::service_fn;
        use hyper_util::rt::TokioIo;
        use tokio::net::TcpListener;

        let listener = TcpListener::bind(self.config.bind_addr)
            .await
            .map_err(|e| NetError::connection(format!("Failed to bind: {e}")))?;

        let registry = Arc::clone(&self.registry);
        let hls_server = self.hls_server.clone();
        let dash_server = self.dash_server.clone();
        let config = self.config.clone();

        tokio::spawn(async move {
            loop {
                match listener.accept().await {
                    Ok((stream, _addr)) => {
                        let io = TokioIo::new(stream);
                        let registry = Arc::clone(&registry);
                        let hls = hls_server.clone();
                        let dash = dash_server.clone();
                        let cfg = config.clone();

                        tokio::spawn(async move {
                            let service = service_fn(move |req| {
                                handle_request(
                                    req,
                                    Arc::clone(&registry),
                                    hls.clone(),
                                    dash.clone(),
                                    cfg.clone(),
                                )
                            });

                            if let Err(e) =
                                http1::Builder::new().serve_connection(io, service).await
                            {
                                eprintln!("Error serving connection: {e}");
                            }
                        });
                    }
                    Err(e) => {
                        eprintln!("Accept error: {e}");
                    }
                }
            }
        });

        Ok(())
    }

    /// Returns the stream registry.
    #[must_use]
    pub fn registry(&self) -> &Arc<StreamRegistry> {
        &self.registry
    }

    /// Shuts down the server.
    pub async fn shutdown(&self) -> NetResult<()> {
        let _ = self.shutdown_tx.send(()).await;
        Ok(())
    }
}

/// Handles HTTP requests.
async fn handle_request(
    req: hyper::Request<hyper::body::Incoming>,
    registry: Arc<StreamRegistry>,
    hls_server: Option<Arc<HlsServer>>,
    dash_server: Option<Arc<DashServer>>,
    config: LiveServerConfig,
) -> Result<hyper::Response<http_body_util::Full<Bytes>>, hyper::Error> {
    use http_body_util::Full;

    let path = req.uri().path();

    // CORS headers
    let mut response = hyper::Response::builder();
    for origin in &config.cors_origins {
        response = response.header("Access-Control-Allow-Origin", origin);
    }
    response = response.header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    response = response.header("Access-Control-Allow-Headers", "Content-Type");

    // Handle OPTIONS
    if req.method() == hyper::Method::OPTIONS {
        return Ok(response
            .status(200)
            .body(Full::new(Bytes::new()))
            .unwrap_or_else(|_| hyper::Response::new(Full::new(Bytes::new()))));
    }

    // Route HLS requests
    if path.starts_with("/hls/") {
        if let Some(hls) = hls_server {
            return hls.handle_request(req, response).await;
        }
    }

    // Route DASH requests
    if path.starts_with("/dash/") {
        if let Some(dash) = dash_server {
            return dash.handle_request(req, response).await;
        }
    }

    // API endpoints
    if path.starts_with("/api/") {
        return handle_api_request(req, registry, response).await;
    }

    // 404
    Ok(response
        .status(404)
        .body(Full::new(Bytes::from("Not Found")))
        .unwrap_or_else(|_| hyper::Response::new(Full::new(Bytes::from("Not Found")))))
}

/// Handles API requests.
async fn handle_api_request(
    req: hyper::Request<hyper::body::Incoming>,
    registry: Arc<StreamRegistry>,
    response: hyper::http::response::Builder,
) -> Result<hyper::Response<http_body_util::Full<Bytes>>, hyper::Error> {
    use http_body_util::Full;

    let path = req.uri().path();

    if path == "/api/streams" && req.method() == hyper::Method::GET {
        let streams = registry.list_streams();
        let json = serde_json::to_string(&streams).unwrap_or_else(|_| String::from("[]"));

        return Ok(response
            .status(200)
            .header("Content-Type", "application/json")
            .body(Full::new(Bytes::from(json)))
            .unwrap_or_else(|_| hyper::Response::new(Full::new(Bytes::from("[]")))));
    }

    Ok(response
        .status(404)
        .body(Full::new(Bytes::from("Not Found")))
        .unwrap_or_else(|_| hyper::Response::new(Full::new(Bytes::from("Not Found")))))
}
