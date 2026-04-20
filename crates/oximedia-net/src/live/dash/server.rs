//! DASH server implementation.

use super::super::{LiveServerConfig, LiveStream, StreamRegistry};
use super::mpd::MpdBuilder;
use bytes::Bytes;
use http_body_util::Full;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

/// DASH server.
pub struct DashServer {
    /// Configuration.
    config: LiveServerConfig,

    /// Stream registry.
    registry: Arc<StreamRegistry>,

    /// Cached MPDs.
    mpd_cache: RwLock<HashMap<String, (Bytes, std::time::Instant)>>,

    /// Cache TTL.
    cache_ttl: Duration,
}

impl DashServer {
    /// Creates a new DASH server.
    #[must_use]
    pub fn new(config: LiveServerConfig, registry: Arc<StreamRegistry>) -> Self {
        Self {
            config,
            registry,
            mpd_cache: RwLock::new(HashMap::new()),
            cache_ttl: Duration::from_secs(1),
        }
    }

    /// Handles a DASH request.
    pub async fn handle_request(
        &self,
        req: hyper::Request<hyper::body::Incoming>,
        response: hyper::http::response::Builder,
    ) -> Result<hyper::Response<Full<Bytes>>, hyper::Error> {
        let path = req.uri().path();

        // Parse path: /dash/{app}/{stream_key}/{file}
        let parts: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();

        if parts.len() < 3 || parts[0] != "dash" {
            return Ok(response
                .status(400)
                .body(Full::new(Bytes::from("Invalid path")))
                .unwrap_or_else(|_| {
                    hyper::Response::new(Full::new(Bytes::from("Internal Server Error")))
                }));
        }

        let app_name = parts[1];
        let stream_key = parts[2];

        // Get stream
        let stream = match self.registry.get_stream(app_name, stream_key) {
            Some(s) => s,
            None => {
                return Ok(response
                    .status(404)
                    .body(Full::new(Bytes::from("Stream not found")))
                    .unwrap_or_else(|_| {
                        hyper::Response::new(Full::new(Bytes::from("Internal Server Error")))
                    }));
            }
        };

        if parts.len() == 3 || (parts.len() == 4 && parts[3] == "manifest.mpd") {
            // MPD request
            return self.serve_mpd(&stream, response).await;
        }

        let filename = parts[3];

        // Serve segment
        self.serve_segment(&stream, filename, response).await
    }

    /// Serves MPD (Media Presentation Description).
    async fn serve_mpd(
        &self,
        stream: &Arc<LiveStream>,
        response: hyper::http::response::Builder,
    ) -> Result<hyper::Response<Full<Bytes>>, hyper::Error> {
        let cache_key = format!("mpd_{}", stream.info().id);

        // Check cache
        {
            let cache = self.mpd_cache.read();
            if let Some((cached, timestamp)) = cache.get(&cache_key) {
                if timestamp.elapsed() < self.cache_ttl {
                    return Ok(response
                        .status(200)
                        .header("Content-Type", "application/dash+xml")
                        .header("Cache-Control", "max-age=1")
                        .body(Full::new(cached.clone()))
                        .unwrap_or_else(|_| {
                            hyper::Response::new(Full::new(Bytes::from("Internal Server Error")))
                        }));
                }
            }
        }

        // Build MPD
        let info = stream.info();
        let mut mpd_builder = MpdBuilder::new()
            .live()
            .min_buffer_time(Duration::from_secs(2))
            .availability_start_time(info.start_time);

        if self.config.enable_ll_dash {
            mpd_builder = mpd_builder.low_latency(Duration::from_millis(500));
        }

        // Add representations
        for variant in &info.variants {
            mpd_builder = mpd_builder.add_video_representation(
                variant.id.clone(),
                variant.bandwidth,
                variant.width,
                variant.height,
                format!("{},{}", variant.video_codec, variant.audio_codec),
            );
        }

        let mpd = mpd_builder.build();
        let bytes = Bytes::from(mpd);

        // Update cache
        {
            let mut cache = self.mpd_cache.write();
            cache.insert(cache_key, (bytes.clone(), std::time::Instant::now()));

            // Clean old entries
            cache.retain(|_, (_, ts)| ts.elapsed() < self.cache_ttl * 10);
        }

        // Add to viewer count
        stream.add_viewer();

        Ok(response
            .status(200)
            .header("Content-Type", "application/dash+xml")
            .header("Cache-Control", "max-age=1")
            .body(Full::new(bytes))
            .unwrap_or_else(|_| {
                hyper::Response::new(Full::new(Bytes::from("Internal Server Error")))
            }))
    }

    /// Serves media segment.
    async fn serve_segment(
        &self,
        stream: &Arc<LiveStream>,
        filename: &str,
        response: hyper::http::response::Builder,
    ) -> Result<hyper::Response<Full<Bytes>>, hyper::Error> {
        // Parse filename: init_{type}.mp4 or {type}_{sequence}_{uuid}.m4s

        if filename.starts_with("init_") {
            // Initialization segment
            if filename.contains("video") {
                if let Some(init_segment) = stream.segment_generator.get_video_init() {
                    return Ok(response
                        .status(200)
                        .header("Content-Type", "video/mp4")
                        .header("Cache-Control", "max-age=31536000")
                        .body(Full::new(init_segment.data.clone()))
                        .unwrap_or_else(|_| {
                            hyper::Response::new(Full::new(Bytes::from("Internal Server Error")))
                        }));
                }
            } else if filename.contains("audio") {
                if let Some(init_segment) = stream.segment_generator.get_audio_init() {
                    return Ok(response
                        .status(200)
                        .header("Content-Type", "audio/mp4")
                        .header("Cache-Control", "max-age=31536000")
                        .body(Full::new(init_segment.data.clone()))
                        .unwrap_or_else(|_| {
                            hyper::Response::new(Full::new(Bytes::from("Internal Server Error")))
                        }));
                }
            }
        } else if filename.starts_with("video_") {
            // Video segment
            if let Some(seq_str) = filename
                .strip_prefix("video_")
                .and_then(|s| s.split('_').next())
            {
                if let Ok(sequence) = seq_str.parse::<u64>() {
                    if let Some(segment) = stream.segment_generator.get_video_segment(sequence) {
                        return Ok(response
                            .status(200)
                            .header("Content-Type", "video/mp4")
                            .header("Cache-Control", "max-age=31536000")
                            .body(Full::new(segment.data.clone()))
                            .unwrap_or_else(|_| {
                                hyper::Response::new(Full::new(Bytes::from(
                                    "Internal Server Error",
                                )))
                            }));
                    }
                }
            }
        } else if filename.starts_with("audio_") {
            // Audio segment
            if let Some(seq_str) = filename
                .strip_prefix("audio_")
                .and_then(|s| s.split('_').next())
            {
                if let Ok(sequence) = seq_str.parse::<u64>() {
                    if let Some(segment) = stream.segment_generator.get_audio_segment(sequence) {
                        return Ok(response
                            .status(200)
                            .header("Content-Type", "audio/mp4")
                            .header("Cache-Control", "max-age=31536000")
                            .body(Full::new(segment.data.clone()))
                            .unwrap_or_else(|_| {
                                hyper::Response::new(Full::new(Bytes::from(
                                    "Internal Server Error",
                                )))
                            }));
                    }
                }
            }
        }

        Ok(response
            .status(404)
            .body(Full::new(Bytes::from("Segment not found")))
            .unwrap_or_else(|_| {
                hyper::Response::new(Full::new(Bytes::from("Internal Server Error")))
            }))
    }

    /// Clears MPD cache.
    pub fn clear_cache(&self) {
        let mut cache = self.mpd_cache.write();
        cache.clear();
    }
}
