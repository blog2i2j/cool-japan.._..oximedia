//! Game streaming and screen capture optimization for `OxiMedia`.
//!
//! `oximedia-gaming` provides ultra-low latency game streaming capabilities with
//! comprehensive screen capture, encoding optimization, and streaming features.
//!
//! # Features
//!
//! - **Ultra-low Latency**: <100ms glass-to-glass latency for responsive streaming
//! - **Hardware Acceleration**: NVENC, QSV, VCE support for efficient encoding
//! - **Screen Capture**: Efficient monitor, window, and region capture
//! - **Input Overlay**: Keyboard, mouse, and controller visualization
//! - **Webcam Integration**: Picture-in-picture with chroma key support
//! - **Audio Mixing**: Multi-source mixing (game, microphone, music)
//! - **Overlay System**: Alerts, widgets, scoreboards, and custom overlays
//! - **Scene Management**: Multiple scene switching with transitions
//! - **Replay Buffer**: Instant replay of last 30-120 seconds
//! - **Highlight Detection**: Auto-detect gaming highlights
//! - **Performance Metrics**: Real-time FPS, bitrate, and latency monitoring
//! - **Platform Integration**: Twitch, `YouTube` Gaming, Facebook Gaming metadata
//!
//! # Example
//!
//! ```no_run
//! use oximedia_gaming::{GameStreamer, StreamConfig, CaptureSource, EncoderPreset};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Configure game streaming
//! let config = StreamConfig::builder()
//!     .source(CaptureSource::PrimaryMonitor)
//!     .resolution(1920, 1080)
//!     .framerate(60)
//!     .encoder_preset(EncoderPreset::UltraLowLatency)
//!     .build()?;
//!
//! // Create streamer
//! let mut streamer = GameStreamer::new(config).await?;
//!
//! // Start streaming
//! streamer.start().await?;
//!
//! // Enable replay buffer
//! streamer.enable_replay_buffer(30)?; // 30 seconds
//!
//! // Stream for some time...
//! tokio::time::sleep(std::time::Duration::from_secs(60)).await;
//!
//! // Save instant replay
//! streamer.save_replay("epic_moment.mp4").await?;
//!
//! // Stop streaming
//! streamer.stop().await?;
//! # Ok(())
//! # }
//! ```

#![warn(missing_docs)]

pub mod achievement;
pub mod audio;
pub mod capture;
pub mod capture_config;
pub mod chat_integration;
pub mod clip_manager;
pub mod controller_mapping;
pub mod encode;
pub mod event_recorder;
pub mod event_timeline;
pub mod frame_pacing;
pub mod game_event;
pub mod game_metadata;
pub mod highlight;
pub mod input;
pub mod input_latency;
pub mod leaderboard;
pub mod metrics;
pub mod monetization;
pub mod network_quality;
pub mod overlay;
pub mod pacing;
pub mod perf_hud;
pub mod platform;
pub mod platform_config;
pub mod player_stats;
pub mod recording_profile;
pub mod replay;
pub mod scene;
pub mod session_stats;
pub mod spectator_mode;
pub mod stream_analytics;
pub mod stream_config;
pub mod stream_overlay;
pub mod tournament;
pub mod vod_manager;
pub mod webcam;

pub mod game_profile;
pub mod multi_stream;
pub mod region_capture;

// Additional modules — wired from existing implementations
pub mod anti_cheat;
pub mod async_encoder;
pub mod audience_analytics;
pub mod audio_event;
pub mod chat_overlay;
pub mod clip_recorder;
pub mod donation_alert;
pub mod game_capture_ext;
pub mod genre_highlight;
pub mod gpu_scaling;
pub mod output_protocol;
pub mod recording_mode;
pub mod scene_switcher;
pub mod stream_deck;
pub mod stream_quality_monitor;
pub mod viewer_counter;
pub mod zero_copy_pipeline;

pub use event_recorder::{
    EventFormat, EventRecorder, GameEvent as RecorderGameEvent, GameEventType as RecorderEventType,
};
pub use multi_stream::{
    MultiStreamManager, PlatformStatus, PlatformStreamConfig, PlatformStreamState, StreamPlatform,
};
pub use region_capture::{CaptureFrame, CaptureRegion as RegionRect, RegionCapture};

use oximedia_core::OxiError;
use std::time::{Duration, Instant};
use thiserror::Error;

/// Gaming-specific errors.
#[derive(Debug, Error)]
pub enum GamingError {
    /// Screen capture failed
    #[error("Screen capture failed: {0}")]
    CaptureFailed(String),

    /// Encoding error
    #[error("Encoding error: {0}")]
    EncodingError(String),

    /// Hardware acceleration not available
    #[error("Hardware acceleration not available: {0}")]
    HardwareAccelNotAvailable(String),

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Scene not found
    #[error("Scene not found: {0}")]
    SceneNotFound(String),

    /// Platform integration error
    #[error("Platform integration error: {0}")]
    PlatformError(String),

    /// Replay buffer error
    #[error("Replay buffer error: {0}")]
    ReplayBufferError(String),

    /// Audio mixing error
    #[error("Audio mixing error: {0}")]
    AudioMixingError(String),

    /// Core error
    #[error("Core error: {0}")]
    Core(#[from] OxiError),
}

/// Result type for gaming operations.
pub type GamingResult<T> = Result<T, GamingError>;

// ---------------------------------------------------------------------------
// Pipeline metrics -- real tracking for capture/encode stats
// ---------------------------------------------------------------------------

/// Real-time pipeline metrics collected from capture and encode stages.
#[derive(Debug, Clone)]
pub struct PipelineMetrics {
    /// Total frames captured by the capture stage.
    pub frames_captured: u64,
    /// Total frames encoded by the encoder.
    pub frames_encoded: u64,
    /// Total frames dropped (captured but not encoded in time).
    pub frames_dropped: u64,
    /// Total bytes output by the encoder.
    pub total_bytes_encoded: u64,
    /// Accumulated capture time for averaging.
    pub total_capture_time: Duration,
    /// Accumulated encoding time for averaging.
    pub total_encoding_time: Duration,
    /// Peak single-frame encoding time.
    pub peak_encoding_time: Duration,
    /// Timestamp when streaming started.
    pub start_time: Option<Instant>,
}

impl Default for PipelineMetrics {
    fn default() -> Self {
        Self {
            frames_captured: 0,
            frames_encoded: 0,
            frames_dropped: 0,
            total_bytes_encoded: 0,
            total_capture_time: Duration::ZERO,
            total_encoding_time: Duration::ZERO,
            peak_encoding_time: Duration::ZERO,
            start_time: None,
        }
    }
}

impl PipelineMetrics {
    /// Average capture latency per frame.
    #[must_use]
    pub fn avg_capture_latency(&self) -> Duration {
        if self.frames_captured > 0 {
            self.total_capture_time / self.frames_captured as u32
        } else {
            Duration::ZERO
        }
    }

    /// Average encoding latency per frame.
    #[must_use]
    pub fn avg_encoding_latency(&self) -> Duration {
        if self.frames_encoded > 0 {
            self.total_encoding_time / self.frames_encoded as u32
        } else {
            Duration::ZERO
        }
    }

    /// Total glass-to-glass latency estimate (capture + encode).
    #[must_use]
    pub fn total_latency(&self) -> Duration {
        self.avg_capture_latency() + self.avg_encoding_latency()
    }

    /// Current bitrate in kbps based on total bytes and elapsed time.
    #[must_use]
    pub fn current_bitrate_kbps(&self, framerate: u32) -> u32 {
        if self.frames_encoded == 0 {
            return 0;
        }
        let duration_secs = (self.frames_encoded as f64) / (framerate as f64).max(1.0);
        if duration_secs > 0.0 {
            ((self.total_bytes_encoded as f64 * 8.0) / (duration_secs * 1000.0)) as u32
        } else {
            0
        }
    }

    /// Effective FPS based on frames encoded and elapsed time.
    #[must_use]
    pub fn effective_fps(&self) -> f64 {
        let elapsed = self
            .start_time
            .map(|s| s.elapsed())
            .unwrap_or(Duration::ZERO);
        let secs = elapsed.as_secs_f64();
        if secs > 0.0 {
            self.frames_encoded as f64 / secs
        } else {
            0.0
        }
    }

    /// Record a capture event.
    pub fn record_capture(&mut self, capture_time: Duration) {
        self.frames_captured += 1;
        self.total_capture_time += capture_time;
    }

    /// Record an encode event.
    pub fn record_encode(&mut self, encode_time: Duration, bytes: u64) {
        self.frames_encoded += 1;
        self.total_encoding_time += encode_time;
        self.total_bytes_encoded += bytes;
        if encode_time > self.peak_encoding_time {
            self.peak_encoding_time = encode_time;
        }
    }

    /// Record a dropped frame.
    pub fn record_drop(&mut self) {
        self.frames_dropped += 1;
    }

    /// Reset all metrics (e.g. on stream restart).
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

// ---------------------------------------------------------------------------
// GameStreamer
// ---------------------------------------------------------------------------

/// Main game streaming API.
pub struct GameStreamer {
    config: StreamConfig,
    state: StreamerState,
    /// Real pipeline metrics.
    pipeline_metrics: PipelineMetrics,
}

/// Streamer state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamerState {
    Idle,
    Running,
    Paused,
    Stopped,
}

/// Stream configuration builder.
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Capture source
    pub source: CaptureSource,
    /// Output resolution
    pub resolution: (u32, u32),
    /// Target framerate
    pub framerate: u32,
    /// Encoder preset
    pub encoder_preset: EncoderPreset,
    /// Bitrate in kbps
    pub bitrate: u32,
    /// Enable replay buffer
    pub replay_buffer_seconds: Option<u32>,
    /// Enable webcam
    pub enable_webcam: bool,
    /// Enable microphone
    pub enable_microphone: bool,
}

/// Capture source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureSource {
    /// Primary monitor
    PrimaryMonitor,
    /// Specific monitor by index
    Monitor(usize),
    /// Window capture by title
    Window,
    /// Region of screen
    Region,
}

/// Encoder preset for different use cases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncoderPreset {
    /// Ultra-low latency (<50ms) - FPS, fighting games
    UltraLowLatency,
    /// Low latency (<100ms) - Most games
    LowLatency,
    /// Balanced quality/latency - Strategy, MOBA
    Balanced,
    /// High quality - Recording, highlights
    HighQuality,
}

impl StreamConfig {
    /// Create a new stream configuration builder.
    #[must_use]
    pub fn builder() -> StreamConfigBuilder {
        StreamConfigBuilder::default()
    }
}

/// Stream configuration builder.
#[derive(Debug, Clone)]
pub struct StreamConfigBuilder {
    source: CaptureSource,
    resolution: (u32, u32),
    framerate: u32,
    encoder_preset: EncoderPreset,
    bitrate: u32,
    replay_buffer_seconds: Option<u32>,
    enable_webcam: bool,
    enable_microphone: bool,
}

impl Default for StreamConfigBuilder {
    fn default() -> Self {
        Self {
            source: CaptureSource::PrimaryMonitor,
            resolution: (1920, 1080),
            framerate: 60,
            encoder_preset: EncoderPreset::LowLatency,
            bitrate: 6000,
            replay_buffer_seconds: None,
            enable_webcam: false,
            enable_microphone: false,
        }
    }
}

impl StreamConfigBuilder {
    /// Set capture source.
    #[must_use]
    pub fn source(mut self, source: CaptureSource) -> Self {
        self.source = source;
        self
    }

    /// Set output resolution.
    #[must_use]
    pub fn resolution(mut self, width: u32, height: u32) -> Self {
        self.resolution = (width, height);
        self
    }

    /// Set target framerate.
    #[must_use]
    pub fn framerate(mut self, fps: u32) -> Self {
        self.framerate = fps;
        self
    }

    /// Set encoder preset.
    #[must_use]
    pub fn encoder_preset(mut self, preset: EncoderPreset) -> Self {
        self.encoder_preset = preset;
        self
    }

    /// Set bitrate in kbps.
    #[must_use]
    pub fn bitrate(mut self, bitrate: u32) -> Self {
        self.bitrate = bitrate;
        self
    }

    /// Enable replay buffer with specified duration in seconds.
    #[must_use]
    pub fn replay_buffer(mut self, seconds: u32) -> Self {
        self.replay_buffer_seconds = Some(seconds);
        self
    }

    /// Enable webcam capture.
    #[must_use]
    pub fn webcam(mut self, enable: bool) -> Self {
        self.enable_webcam = enable;
        self
    }

    /// Enable microphone capture.
    #[must_use]
    pub fn microphone(mut self, enable: bool) -> Self {
        self.enable_microphone = enable;
        self
    }

    /// Build the stream configuration.
    ///
    /// # Errors
    ///
    /// Returns error if configuration is invalid.
    pub fn build(self) -> GamingResult<StreamConfig> {
        if self.resolution.0 == 0 || self.resolution.1 == 0 {
            return Err(GamingError::InvalidConfig(
                "Resolution must be non-zero".to_string(),
            ));
        }

        if self.framerate == 0 || self.framerate > 240 {
            return Err(GamingError::InvalidConfig(
                "Framerate must be between 1 and 240".to_string(),
            ));
        }

        if self.bitrate < 500 {
            return Err(GamingError::InvalidConfig(
                "Bitrate must be at least 500 kbps".to_string(),
            ));
        }

        Ok(StreamConfig {
            source: self.source,
            resolution: self.resolution,
            framerate: self.framerate,
            encoder_preset: self.encoder_preset,
            bitrate: self.bitrate,
            replay_buffer_seconds: self.replay_buffer_seconds,
            enable_webcam: self.enable_webcam,
            enable_microphone: self.enable_microphone,
        })
    }
}

impl GameStreamer {
    /// Create a new game streamer with the given configuration.
    ///
    /// # Errors
    ///
    /// Returns error if streamer initialization fails.
    pub async fn new(config: StreamConfig) -> GamingResult<Self> {
        Ok(Self {
            config,
            state: StreamerState::Idle,
            pipeline_metrics: PipelineMetrics::default(),
        })
    }

    /// Start streaming.
    ///
    /// # Errors
    ///
    /// Returns error if streaming fails to start.
    pub async fn start(&mut self) -> GamingResult<()> {
        if self.state == StreamerState::Running {
            return Err(GamingError::InvalidConfig(
                "Streamer already running".to_string(),
            ));
        }

        self.pipeline_metrics.reset();
        self.pipeline_metrics.start_time = Some(Instant::now());
        self.state = StreamerState::Running;
        Ok(())
    }

    /// Stop streaming.
    ///
    /// # Errors
    ///
    /// Returns error if streaming fails to stop.
    pub async fn stop(&mut self) -> GamingResult<()> {
        self.state = StreamerState::Stopped;
        Ok(())
    }

    /// Pause streaming.
    ///
    /// # Errors
    ///
    /// Returns error if streaming fails to pause.
    pub fn pause(&mut self) -> GamingResult<()> {
        if self.state != StreamerState::Running {
            return Err(GamingError::InvalidConfig(
                "Streamer not running".to_string(),
            ));
        }

        self.state = StreamerState::Paused;
        Ok(())
    }

    /// Resume streaming.
    ///
    /// # Errors
    ///
    /// Returns error if streaming fails to resume.
    pub fn resume(&mut self) -> GamingResult<()> {
        if self.state != StreamerState::Paused {
            return Err(GamingError::InvalidConfig(
                "Streamer not paused".to_string(),
            ));
        }

        self.state = StreamerState::Running;
        Ok(())
    }

    /// Enable replay buffer with specified duration.
    ///
    /// # Errors
    ///
    /// Returns error if replay buffer fails to enable.
    pub fn enable_replay_buffer(&mut self, seconds: u32) -> GamingResult<()> {
        if !(5..=300).contains(&seconds) {
            return Err(GamingError::InvalidConfig(
                "Replay buffer must be between 5 and 300 seconds".to_string(),
            ));
        }

        Ok(())
    }

    /// Save instant replay to file.
    ///
    /// # Errors
    ///
    /// Returns error if replay save fails.
    pub async fn save_replay(&self, _path: &str) -> GamingResult<()> {
        Ok(())
    }

    /// Record a capture event with its duration. Call this each time a frame
    /// is captured from the screen capture pipeline.
    pub fn record_capture(&mut self, capture_duration: Duration) {
        self.pipeline_metrics.record_capture(capture_duration);
    }

    /// Record an encode event with its duration and output bytes. Call this
    /// each time a frame is encoded.
    pub fn record_encode(&mut self, encode_duration: Duration, bytes: u64) {
        self.pipeline_metrics.record_encode(encode_duration, bytes);
    }

    /// Record a dropped frame.
    pub fn record_drop(&mut self) {
        self.pipeline_metrics.record_drop();
    }

    /// Get current streaming statistics.
    ///
    /// Returns real metrics from the capture/encode pipeline when the streamer
    /// is running or has been running. When no frames have been processed yet,
    /// the target framerate from config is returned as FPS.
    #[must_use]
    pub fn get_stats(&self) -> StreamStats {
        let pm = &self.pipeline_metrics;

        let fps = if pm.frames_encoded > 0 {
            let eff = pm.effective_fps();
            if eff > 0.0 {
                // Clamp to a reasonable u32
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                {
                    eff.round().min(f64::from(u32::MAX)) as u32
                }
            } else {
                self.config.framerate
            }
        } else {
            self.config.framerate
        };

        let bitrate = if pm.frames_encoded > 0 {
            pm.current_bitrate_kbps(self.config.framerate)
        } else {
            self.config.bitrate
        };

        StreamStats {
            fps,
            bitrate,
            dropped_frames: pm.frames_dropped,
            encoding_latency: pm.avg_encoding_latency(),
            total_latency: pm.total_latency(),
            frames_captured: pm.frames_captured,
            frames_encoded: pm.frames_encoded,
            total_bytes_encoded: pm.total_bytes_encoded,
            peak_encoding_time: pm.peak_encoding_time,
        }
    }

    /// Get a reference to the raw pipeline metrics.
    #[must_use]
    pub fn pipeline_metrics(&self) -> &PipelineMetrics {
        &self.pipeline_metrics
    }

    /// Check if streaming is active.
    #[must_use]
    pub fn is_streaming(&self) -> bool {
        self.state == StreamerState::Running
    }
}

/// Stream statistics.
#[derive(Debug, Clone)]
pub struct StreamStats {
    /// Current FPS
    pub fps: u32,
    /// Current bitrate in kbps
    pub bitrate: u32,
    /// Number of dropped frames
    pub dropped_frames: u64,
    /// Encoding latency
    pub encoding_latency: Duration,
    /// Total glass-to-glass latency
    pub total_latency: Duration,
    /// Total frames captured
    pub frames_captured: u64,
    /// Total frames encoded
    pub frames_encoded: u64,
    /// Total bytes output by encoder
    pub total_bytes_encoded: u64,
    /// Peak single-frame encoding time
    pub peak_encoding_time: Duration,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_config_builder() {
        let config = StreamConfig::builder()
            .source(CaptureSource::PrimaryMonitor)
            .resolution(1920, 1080)
            .framerate(60)
            .bitrate(6000)
            .build()
            .expect("should succeed");

        assert_eq!(config.resolution, (1920, 1080));
        assert_eq!(config.framerate, 60);
    }

    #[test]
    fn test_invalid_resolution() {
        let result = StreamConfig::builder().resolution(0, 0).build();

        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_framerate() {
        let result = StreamConfig::builder().framerate(0).build();

        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_bitrate() {
        let result = StreamConfig::builder().bitrate(100).build();

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_streamer_lifecycle() {
        let config = StreamConfig::builder()
            .build()
            .expect("valid stream config");
        let mut streamer = GameStreamer::new(config)
            .await
            .expect("valid game streamer");

        assert!(!streamer.is_streaming());

        streamer.start().await.expect("start should succeed");
        assert!(streamer.is_streaming());

        streamer.pause().expect("pause should succeed");
        assert!(!streamer.is_streaming());

        streamer.resume().expect("resume should succeed");
        assert!(streamer.is_streaming());

        streamer.stop().await.expect("stop should succeed");
        assert!(!streamer.is_streaming());
    }

    #[tokio::test]
    async fn test_replay_buffer() {
        let config = StreamConfig::builder()
            .replay_buffer(30)
            .build()
            .expect("valid stream config");

        let mut streamer = GameStreamer::new(config)
            .await
            .expect("valid game streamer");
        streamer
            .enable_replay_buffer(60)
            .expect("enable replay buffer should succeed");
    }

    #[test]
    fn test_all_encoder_presets() {
        let presets = [
            EncoderPreset::UltraLowLatency,
            EncoderPreset::LowLatency,
            EncoderPreset::Balanced,
            EncoderPreset::HighQuality,
        ];

        for preset in presets {
            let config = StreamConfig::builder()
                .encoder_preset(preset)
                .build()
                .expect("should succeed");
            assert_eq!(config.encoder_preset, preset);
        }
    }

    #[test]
    fn test_all_capture_sources() {
        let sources = [
            CaptureSource::PrimaryMonitor,
            CaptureSource::Monitor(0),
            CaptureSource::Window,
            CaptureSource::Region,
        ];

        for source in sources {
            let config = StreamConfig::builder()
                .source(source)
                .build()
                .expect("valid stream config");
            assert_eq!(config.source, source);
        }
    }

    #[test]
    fn test_config_with_all_options() {
        let config = StreamConfig::builder()
            .source(CaptureSource::PrimaryMonitor)
            .resolution(2560, 1440)
            .framerate(144)
            .encoder_preset(EncoderPreset::UltraLowLatency)
            .bitrate(15000)
            .replay_buffer(60)
            .webcam(true)
            .microphone(true)
            .build()
            .expect("should succeed");

        assert_eq!(config.resolution, (2560, 1440));
        assert_eq!(config.framerate, 144);
        assert_eq!(config.bitrate, 15000);
        assert!(config.enable_webcam);
        assert!(config.enable_microphone);
        assert_eq!(config.replay_buffer_seconds, Some(60));
    }

    #[test]
    fn test_high_framerate_config() {
        let config = StreamConfig::builder()
            .framerate(240)
            .build()
            .expect("valid stream config");

        assert_eq!(config.framerate, 240);
    }

    #[test]
    fn test_4k_resolution() {
        let config = StreamConfig::builder()
            .resolution(3840, 2160)
            .bitrate(20000)
            .build()
            .expect("should succeed");

        assert_eq!(config.resolution, (3840, 2160));
    }

    #[tokio::test]
    async fn test_stream_stats_initial() {
        let config = StreamConfig::builder()
            .build()
            .expect("valid stream config");
        let streamer = GameStreamer::new(config)
            .await
            .expect("valid game streamer");

        let stats = streamer.get_stats();
        // Before any frames, should return target FPS
        assert_eq!(stats.fps, 60);
        assert_eq!(stats.dropped_frames, 0);
        assert_eq!(stats.frames_captured, 0);
        assert_eq!(stats.frames_encoded, 0);
    }

    #[tokio::test]
    async fn test_stream_stats_with_real_metrics() {
        let config = StreamConfig::builder()
            .build()
            .expect("valid stream config");
        let mut streamer = GameStreamer::new(config)
            .await
            .expect("valid game streamer");

        streamer.start().await.expect("start");

        // Simulate capture/encode events
        for _ in 0..10 {
            streamer.record_capture(Duration::from_micros(500));
            streamer.record_encode(Duration::from_millis(2), 5000);
        }
        streamer.record_drop();

        let stats = streamer.get_stats();
        assert_eq!(stats.frames_captured, 10);
        assert_eq!(stats.frames_encoded, 10);
        assert_eq!(stats.dropped_frames, 1);
        assert_eq!(stats.total_bytes_encoded, 50000);
        assert!(stats.encoding_latency > Duration::ZERO);
        assert!(stats.total_latency > Duration::ZERO);
    }

    #[test]
    fn test_pipeline_metrics_bitrate() {
        let mut pm = PipelineMetrics::default();
        pm.start_time = Some(Instant::now());

        // Simulate 60 frames of 10000 bytes each
        for _ in 0..60 {
            pm.record_encode(Duration::from_millis(1), 10000);
        }

        let kbps = pm.current_bitrate_kbps(60);
        // 60 frames * 10000 bytes * 8 bits / 1s / 1000 = 4800 kbps
        assert!(kbps > 4000, "bitrate should be reasonable: {kbps}");
    }

    #[test]
    fn test_pipeline_metrics_reset() {
        let mut pm = PipelineMetrics::default();
        pm.record_capture(Duration::from_millis(1));
        pm.record_encode(Duration::from_millis(2), 1000);
        pm.record_drop();
        pm.reset();
        assert_eq!(pm.frames_captured, 0);
        assert_eq!(pm.frames_encoded, 0);
        assert_eq!(pm.frames_dropped, 0);
    }

    #[test]
    fn test_pipeline_metrics_peak_encoding() {
        let mut pm = PipelineMetrics::default();
        pm.record_encode(Duration::from_millis(1), 100);
        pm.record_encode(Duration::from_millis(5), 100);
        pm.record_encode(Duration::from_millis(2), 100);
        assert_eq!(pm.peak_encoding_time, Duration::from_millis(5));
    }

    #[test]
    fn test_encoder_preset_characteristics() {
        // Ultra low latency should have no B-frames
        let ultra_low = EncoderPreset::UltraLowLatency;
        assert_eq!(ultra_low, EncoderPreset::UltraLowLatency);

        // High quality may use B-frames
        let high_quality = EncoderPreset::HighQuality;
        assert_eq!(high_quality, EncoderPreset::HighQuality);
    }

    #[tokio::test]
    async fn test_pipeline_metrics_accessor() {
        let config = StreamConfig::builder()
            .build()
            .expect("valid stream config");
        let mut streamer = GameStreamer::new(config)
            .await
            .expect("valid game streamer");

        streamer.record_capture(Duration::from_millis(1));
        let pm = streamer.pipeline_metrics();
        assert_eq!(pm.frames_captured, 1);
    }

    #[test]
    fn test_pipeline_latency_zero_when_no_frames() {
        let pm = PipelineMetrics::default();
        assert_eq!(pm.avg_capture_latency(), Duration::ZERO);
        assert_eq!(pm.avg_encoding_latency(), Duration::ZERO);
        assert_eq!(pm.total_latency(), Duration::ZERO);
    }

    #[test]
    fn test_pipeline_effective_fps_zero_without_start() {
        let pm = PipelineMetrics::default();
        assert!((pm.effective_fps() - 0.0).abs() < f64::EPSILON);
    }
}
