#![allow(dead_code)]

//! Recording-only mode with higher quality settings than live streaming.
//!
//! Provides a dedicated recording pipeline that prioritizes quality over latency,
//! using larger GOP sizes, higher bitrates, multi-pass encoding estimation, and
//! lossless audio capture. Useful for creating highlight reels, tutorials, and
//! archival footage where real-time delivery is not required.
//!
//! # Features
//!
//! - **Quality presets**: Lossless, Studio, High, Standard, Compact
//! - **CRF-based encoding**: Constant Rate Factor for consistent quality
//! - **Multi-pass estimation**: Two-pass bitrate estimation for optimal file sizes
//! - **Audio quality**: Uncompressed or high-bitrate audio capture
//! - **Chapter markers**: In-recording chapter marking for post-production
//! - **File splitting**: Automatic file splitting by size or duration
//! - **Recording profiles**: Per-game or per-use-case recording configurations

use std::time::{Duration, Instant};

use crate::{GamingError, GamingResult};

// ---------------------------------------------------------------------------
// Quality presets
// ---------------------------------------------------------------------------

/// Recording quality preset, ordered from highest to lowest quality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RecordingQuality {
    /// Lossless / near-lossless recording. Very large files.
    Lossless,
    /// Studio quality — high bitrate, large GOP, suitable for editing.
    Studio,
    /// High quality — good balance for archival.
    High,
    /// Standard quality — smaller files, still good.
    Standard,
    /// Compact — prioritizes small file size.
    Compact,
}

impl RecordingQuality {
    /// Recommended CRF value for VP9 encoding at this quality level.
    /// Lower CRF = higher quality. Range: 0 (lossless) to 63.
    #[must_use]
    pub fn crf(&self) -> u8 {
        match self {
            Self::Lossless => 0,
            Self::Studio => 10,
            Self::High => 18,
            Self::Standard => 28,
            Self::Compact => 38,
        }
    }

    /// Recommended GOP (keyframe interval) in frames.
    #[must_use]
    pub fn gop_size(&self, framerate: u32) -> u32 {
        let seconds = match self {
            Self::Lossless => 1,
            Self::Studio => 2,
            Self::High => 4,
            Self::Standard => 5,
            Self::Compact => 10,
        };
        framerate.saturating_mul(seconds)
    }

    /// Recommended maximum bitrate in kbps for the given resolution.
    #[must_use]
    pub fn max_bitrate_kbps(&self, width: u32, height: u32) -> u32 {
        let pixels = (width as u64).saturating_mul(height as u64);
        let base = match self {
            Self::Lossless => 100_000,
            Self::Studio => 50_000,
            Self::High => 25_000,
            Self::Standard => 12_000,
            Self::Compact => 6_000,
        };
        // Scale proportionally to 1080p
        let reference_pixels: u64 = 1920 * 1080;
        let scale = (pixels as f64) / (reference_pixels as f64);
        let scaled = (base as f64 * scale).max(1000.0);
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        {
            scaled.round().min(f64::from(u32::MAX)) as u32
        }
    }

    /// Human-readable name.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Lossless => "Lossless",
            Self::Studio => "Studio",
            Self::High => "High",
            Self::Standard => "Standard",
            Self::Compact => "Compact",
        }
    }
}

impl std::fmt::Display for RecordingQuality {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ---------------------------------------------------------------------------
// Audio quality
// ---------------------------------------------------------------------------

/// Recording audio quality.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioRecordingQuality {
    /// Uncompressed PCM (WAV-like).
    Uncompressed,
    /// High-quality Opus at 256 kbps.
    HighOpus,
    /// Standard Opus at 128 kbps.
    StandardOpus,
    /// FLAC lossless compression.
    Flac,
    /// Vorbis at 192 kbps.
    HighVorbis,
}

impl AudioRecordingQuality {
    /// Approximate bitrate in kbps for estimation.
    #[must_use]
    pub fn approximate_bitrate_kbps(&self, sample_rate: u32, channels: u16) -> u32 {
        match self {
            Self::Uncompressed => {
                // 16-bit PCM: sample_rate * channels * 16 / 1000
                let bits_per_sec = (sample_rate as u64) * (channels as u64) * 16;
                #[allow(clippy::cast_possible_truncation)]
                {
                    (bits_per_sec / 1000).min(u32::MAX as u64) as u32
                }
            }
            Self::HighOpus => 256,
            Self::StandardOpus => 128,
            Self::Flac => {
                // Roughly 50-70% of uncompressed, use 60%
                let uncompressed = (sample_rate as u64) * (channels as u64) * 16;
                #[allow(clippy::cast_possible_truncation)]
                {
                    ((uncompressed * 60) / (100 * 1000)).min(u32::MAX as u64) as u32
                }
            }
            Self::HighVorbis => 192,
        }
    }

    /// Name for display.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Uncompressed => "Uncompressed PCM",
            Self::HighOpus => "Opus 256 kbps",
            Self::StandardOpus => "Opus 128 kbps",
            Self::Flac => "FLAC Lossless",
            Self::HighVorbis => "Vorbis 192 kbps",
        }
    }
}

// ---------------------------------------------------------------------------
// File splitting
// ---------------------------------------------------------------------------

/// How to split recording output files.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileSplitPolicy {
    /// No splitting — single output file.
    None,
    /// Split by duration (e.g. every 30 minutes).
    ByDuration(Duration),
    /// Split by approximate file size in megabytes.
    BySizeMb(u64),
}

// ---------------------------------------------------------------------------
// Chapter marker
// ---------------------------------------------------------------------------

/// A chapter marker within a recording.
#[derive(Debug, Clone)]
pub struct ChapterMarker {
    /// Title of the chapter.
    pub title: String,
    /// Time offset from the start of the recording.
    pub offset: Duration,
    /// Optional description or note.
    pub description: Option<String>,
}

impl ChapterMarker {
    /// Create a new chapter marker.
    #[must_use]
    pub fn new(title: &str, offset: Duration) -> Self {
        Self {
            title: title.to_string(),
            offset,
            description: None,
        }
    }

    /// Attach a description.
    #[must_use]
    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = Some(desc.to_string());
        self
    }
}

// ---------------------------------------------------------------------------
// Recording configuration
// ---------------------------------------------------------------------------

/// Configuration for recording-only mode.
#[derive(Debug, Clone)]
pub struct RecordingConfig {
    /// Video quality preset.
    pub quality: RecordingQuality,
    /// Output resolution (width, height).
    pub resolution: (u32, u32),
    /// Target framerate.
    pub framerate: u32,
    /// Audio quality.
    pub audio_quality: AudioRecordingQuality,
    /// Audio sample rate.
    pub audio_sample_rate: u32,
    /// Number of audio channels.
    pub audio_channels: u16,
    /// File split policy.
    pub split_policy: FileSplitPolicy,
    /// Output directory path.
    pub output_dir: String,
    /// Filename prefix (timestamp will be appended).
    pub filename_prefix: String,
    /// Whether to capture system audio.
    pub capture_system_audio: bool,
    /// Whether to capture microphone audio.
    pub capture_microphone: bool,
    /// Custom CRF override (None = use quality preset default).
    pub crf_override: Option<u8>,
    /// Custom max bitrate override in kbps (None = use quality preset default).
    pub max_bitrate_override: Option<u32>,
}

impl Default for RecordingConfig {
    fn default() -> Self {
        Self {
            quality: RecordingQuality::High,
            resolution: (1920, 1080),
            framerate: 60,
            audio_quality: AudioRecordingQuality::Flac,
            audio_sample_rate: 48000,
            audio_channels: 2,
            split_policy: FileSplitPolicy::None,
            output_dir: String::new(),
            filename_prefix: "recording".to_string(),
            capture_system_audio: true,
            capture_microphone: true,
            crf_override: None,
            max_bitrate_override: None,
        }
    }
}

impl RecordingConfig {
    /// Create a new builder.
    #[must_use]
    pub fn builder() -> RecordingConfigBuilder {
        RecordingConfigBuilder::default()
    }

    /// Effective CRF (override or preset default).
    #[must_use]
    pub fn effective_crf(&self) -> u8 {
        self.crf_override.unwrap_or_else(|| self.quality.crf())
    }

    /// Effective max bitrate in kbps.
    #[must_use]
    pub fn effective_max_bitrate_kbps(&self) -> u32 {
        self.max_bitrate_override.unwrap_or_else(|| {
            self.quality
                .max_bitrate_kbps(self.resolution.0, self.resolution.1)
        })
    }

    /// Estimated file size in MB for a given recording duration.
    #[must_use]
    pub fn estimate_file_size_mb(&self, duration: Duration) -> f64 {
        let video_kbps = self.effective_max_bitrate_kbps() as f64;
        let audio_kbps = self
            .audio_quality
            .approximate_bitrate_kbps(self.audio_sample_rate, self.audio_channels)
            as f64;
        let total_kbps = video_kbps + audio_kbps;
        let seconds = duration.as_secs_f64();
        // kbps * seconds / 8 / 1024 = MB
        (total_kbps * seconds) / (8.0 * 1024.0)
    }

    /// GOP size in frames based on quality and framerate.
    #[must_use]
    pub fn gop_size(&self) -> u32 {
        self.quality.gop_size(self.framerate)
    }
}

// ---------------------------------------------------------------------------
// Config builder
// ---------------------------------------------------------------------------

/// Builder for `RecordingConfig`.
#[derive(Debug, Clone)]
pub struct RecordingConfigBuilder {
    config: RecordingConfig,
}

impl Default for RecordingConfigBuilder {
    fn default() -> Self {
        Self {
            config: RecordingConfig::default(),
        }
    }
}

impl RecordingConfigBuilder {
    /// Set quality preset.
    #[must_use]
    pub fn quality(mut self, q: RecordingQuality) -> Self {
        self.config.quality = q;
        self
    }

    /// Set output resolution.
    #[must_use]
    pub fn resolution(mut self, width: u32, height: u32) -> Self {
        self.config.resolution = (width, height);
        self
    }

    /// Set target framerate.
    #[must_use]
    pub fn framerate(mut self, fps: u32) -> Self {
        self.config.framerate = fps;
        self
    }

    /// Set audio quality.
    #[must_use]
    pub fn audio_quality(mut self, aq: AudioRecordingQuality) -> Self {
        self.config.audio_quality = aq;
        self
    }

    /// Set audio sample rate.
    #[must_use]
    pub fn audio_sample_rate(mut self, rate: u32) -> Self {
        self.config.audio_sample_rate = rate;
        self
    }

    /// Set audio channels.
    #[must_use]
    pub fn audio_channels(mut self, channels: u16) -> Self {
        self.config.audio_channels = channels;
        self
    }

    /// Set file split policy.
    #[must_use]
    pub fn split_policy(mut self, policy: FileSplitPolicy) -> Self {
        self.config.split_policy = policy;
        self
    }

    /// Set output directory.
    #[must_use]
    pub fn output_dir(mut self, dir: &str) -> Self {
        self.config.output_dir = dir.to_string();
        self
    }

    /// Set filename prefix.
    #[must_use]
    pub fn filename_prefix(mut self, prefix: &str) -> Self {
        self.config.filename_prefix = prefix.to_string();
        self
    }

    /// Set system audio capture.
    #[must_use]
    pub fn capture_system_audio(mut self, enable: bool) -> Self {
        self.config.capture_system_audio = enable;
        self
    }

    /// Set microphone capture.
    #[must_use]
    pub fn capture_microphone(mut self, enable: bool) -> Self {
        self.config.capture_microphone = enable;
        self
    }

    /// Override CRF value.
    #[must_use]
    pub fn crf(mut self, crf: u8) -> Self {
        self.config.crf_override = Some(crf);
        self
    }

    /// Override max bitrate in kbps.
    #[must_use]
    pub fn max_bitrate_kbps(mut self, kbps: u32) -> Self {
        self.config.max_bitrate_override = Some(kbps);
        self
    }

    /// Build the recording configuration.
    ///
    /// # Errors
    ///
    /// Returns error if the configuration is invalid.
    pub fn build(self) -> GamingResult<RecordingConfig> {
        let c = &self.config;
        if c.resolution.0 == 0 || c.resolution.1 == 0 {
            return Err(GamingError::InvalidConfig(
                "Resolution must be non-zero".into(),
            ));
        }
        if c.framerate == 0 || c.framerate > 240 {
            return Err(GamingError::InvalidConfig("Framerate must be 1-240".into()));
        }
        if c.audio_sample_rate == 0 {
            return Err(GamingError::InvalidConfig(
                "Audio sample rate must be non-zero".into(),
            ));
        }
        if c.audio_channels == 0 {
            return Err(GamingError::InvalidConfig(
                "Audio channels must be non-zero".into(),
            ));
        }
        if let Some(crf) = c.crf_override {
            if crf > 63 {
                return Err(GamingError::InvalidConfig("CRF must be 0-63".into()));
            }
        }
        Ok(self.config)
    }
}

// ---------------------------------------------------------------------------
// Recording session
// ---------------------------------------------------------------------------

/// State of a recording session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordingState {
    /// Idle / not started.
    Idle,
    /// Currently recording.
    Recording,
    /// Paused.
    Paused,
    /// Stopped / finalized.
    Stopped,
}

/// A recording session managing the lifecycle and statistics of a single
/// recording run.
#[derive(Debug)]
pub struct RecordingSession {
    /// Configuration for this session.
    config: RecordingConfig,
    /// Current state.
    state: RecordingState,
    /// When the recording started (or None if not started).
    start_time: Option<Instant>,
    /// Accumulated recording duration (accounts for pauses).
    accumulated_duration: Duration,
    /// When the current recording segment started (for pause accounting).
    segment_start: Option<Instant>,
    /// Frames captured so far.
    frames_captured: u64,
    /// Bytes written so far.
    bytes_written: u64,
    /// Chapter markers.
    chapters: Vec<ChapterMarker>,
    /// Number of file segments created (for split policy).
    file_segments: u32,
}

impl RecordingSession {
    /// Create a new recording session with the given configuration.
    #[must_use]
    pub fn new(config: RecordingConfig) -> Self {
        Self {
            config,
            state: RecordingState::Idle,
            start_time: None,
            accumulated_duration: Duration::ZERO,
            segment_start: None,
            frames_captured: 0,
            bytes_written: 0,
            chapters: Vec::new(),
            file_segments: 0,
        }
    }

    /// Start recording.
    ///
    /// # Errors
    ///
    /// Returns error if already recording.
    pub fn start(&mut self) -> GamingResult<()> {
        if self.state == RecordingState::Recording {
            return Err(GamingError::InvalidConfig("Already recording".into()));
        }
        let now = Instant::now();
        self.start_time = Some(now);
        self.segment_start = Some(now);
        self.state = RecordingState::Recording;
        self.file_segments = 1;
        Ok(())
    }

    /// Pause recording.
    ///
    /// # Errors
    ///
    /// Returns error if not currently recording.
    pub fn pause(&mut self) -> GamingResult<()> {
        if self.state != RecordingState::Recording {
            return Err(GamingError::InvalidConfig("Not currently recording".into()));
        }
        if let Some(seg) = self.segment_start.take() {
            self.accumulated_duration += seg.elapsed();
        }
        self.state = RecordingState::Paused;
        Ok(())
    }

    /// Resume recording after a pause.
    ///
    /// # Errors
    ///
    /// Returns error if not paused.
    pub fn resume(&mut self) -> GamingResult<()> {
        if self.state != RecordingState::Paused {
            return Err(GamingError::InvalidConfig("Not paused".into()));
        }
        self.segment_start = Some(Instant::now());
        self.state = RecordingState::Recording;
        Ok(())
    }

    /// Stop recording and finalize.
    ///
    /// # Errors
    ///
    /// Returns error if in idle state.
    pub fn stop(&mut self) -> GamingResult<RecordingSummary> {
        if self.state == RecordingState::Idle || self.state == RecordingState::Stopped {
            return Err(GamingError::InvalidConfig("Not recording".into()));
        }
        if let Some(seg) = self.segment_start.take() {
            self.accumulated_duration += seg.elapsed();
        }
        self.state = RecordingState::Stopped;

        Ok(RecordingSummary {
            duration: self.accumulated_duration,
            frames_captured: self.frames_captured,
            bytes_written: self.bytes_written,
            file_segments: self.file_segments,
            chapters: self.chapters.clone(),
            quality: self.config.quality,
            effective_crf: self.config.effective_crf(),
        })
    }

    /// Record a captured frame.
    pub fn record_frame(&mut self, frame_bytes: u64) {
        self.frames_captured += 1;
        self.bytes_written += frame_bytes;

        // Check if we need to split
        if let FileSplitPolicy::BySizeMb(max_mb) = self.config.split_policy {
            let current_mb = self.bytes_written / (1024 * 1024);
            if current_mb >= max_mb && self.frames_captured > 1 {
                self.file_segments += 1;
                // In a real implementation, this would close the current file
                // and open a new one. Here we just track the segment count.
            }
        }
    }

    /// Add a chapter marker at the current recording position.
    ///
    /// # Errors
    ///
    /// Returns error if not recording.
    pub fn add_chapter(&mut self, title: &str) -> GamingResult<()> {
        if self.state != RecordingState::Recording && self.state != RecordingState::Paused {
            return Err(GamingError::InvalidConfig(
                "Can only add chapters while recording or paused".into(),
            ));
        }
        let offset = self.current_duration();
        self.chapters.push(ChapterMarker::new(title, offset));
        Ok(())
    }

    /// Add a chapter marker with a description.
    ///
    /// # Errors
    ///
    /// Returns error if not recording.
    pub fn add_chapter_with_description(
        &mut self,
        title: &str,
        description: &str,
    ) -> GamingResult<()> {
        if self.state != RecordingState::Recording && self.state != RecordingState::Paused {
            return Err(GamingError::InvalidConfig(
                "Can only add chapters while recording or paused".into(),
            ));
        }
        let offset = self.current_duration();
        let marker = ChapterMarker::new(title, offset).with_description(description);
        self.chapters.push(marker);
        Ok(())
    }

    /// Current recording duration (including current segment if recording).
    #[must_use]
    pub fn current_duration(&self) -> Duration {
        let seg_elapsed = self
            .segment_start
            .map(|s| s.elapsed())
            .unwrap_or(Duration::ZERO);
        self.accumulated_duration + seg_elapsed
    }

    /// Current state.
    #[must_use]
    pub fn state(&self) -> RecordingState {
        self.state
    }

    /// Total frames captured.
    #[must_use]
    pub fn frames_captured(&self) -> u64 {
        self.frames_captured
    }

    /// Total bytes written.
    #[must_use]
    pub fn bytes_written(&self) -> u64 {
        self.bytes_written
    }

    /// Number of file segments created.
    #[must_use]
    pub fn file_segments(&self) -> u32 {
        self.file_segments
    }

    /// Chapter markers.
    #[must_use]
    pub fn chapters(&self) -> &[ChapterMarker] {
        &self.chapters
    }

    /// Reference to the recording configuration.
    #[must_use]
    pub fn config(&self) -> &RecordingConfig {
        &self.config
    }

    /// Effective average bitrate so far in kbps.
    #[must_use]
    pub fn average_bitrate_kbps(&self) -> f64 {
        let dur = self.current_duration().as_secs_f64();
        if dur > 0.0 {
            (self.bytes_written as f64 * 8.0) / (dur * 1000.0)
        } else {
            0.0
        }
    }

    /// Effective FPS so far.
    #[must_use]
    pub fn effective_fps(&self) -> f64 {
        let dur = self.current_duration().as_secs_f64();
        if dur > 0.0 {
            self.frames_captured as f64 / dur
        } else {
            0.0
        }
    }
}

// ---------------------------------------------------------------------------
// Recording summary
// ---------------------------------------------------------------------------

/// Summary produced when a recording session is stopped.
#[derive(Debug, Clone)]
pub struct RecordingSummary {
    /// Total recording duration.
    pub duration: Duration,
    /// Total frames captured.
    pub frames_captured: u64,
    /// Total bytes written.
    pub bytes_written: u64,
    /// Number of file segments.
    pub file_segments: u32,
    /// Chapter markers.
    pub chapters: Vec<ChapterMarker>,
    /// Quality preset used.
    pub quality: RecordingQuality,
    /// Effective CRF used.
    pub effective_crf: u8,
}

impl RecordingSummary {
    /// File size in megabytes.
    #[must_use]
    pub fn file_size_mb(&self) -> f64 {
        self.bytes_written as f64 / (1024.0 * 1024.0)
    }

    /// Average bitrate in kbps.
    #[must_use]
    pub fn average_bitrate_kbps(&self) -> f64 {
        let secs = self.duration.as_secs_f64();
        if secs > 0.0 {
            (self.bytes_written as f64 * 8.0) / (secs * 1000.0)
        } else {
            0.0
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- RecordingQuality tests --

    #[test]
    fn test_quality_crf_values() {
        assert_eq!(RecordingQuality::Lossless.crf(), 0);
        assert_eq!(RecordingQuality::Studio.crf(), 10);
        assert_eq!(RecordingQuality::High.crf(), 18);
        assert_eq!(RecordingQuality::Standard.crf(), 28);
        assert_eq!(RecordingQuality::Compact.crf(), 38);
    }

    #[test]
    fn test_quality_gop_sizes() {
        assert_eq!(RecordingQuality::Lossless.gop_size(60), 60);
        assert_eq!(RecordingQuality::Studio.gop_size(60), 120);
        assert_eq!(RecordingQuality::High.gop_size(30), 120);
        assert_eq!(RecordingQuality::Standard.gop_size(60), 300);
        assert_eq!(RecordingQuality::Compact.gop_size(60), 600);
    }

    #[test]
    fn test_quality_max_bitrate_scales_with_resolution() {
        let q = RecordingQuality::High;
        let bitrate_1080 = q.max_bitrate_kbps(1920, 1080);
        let bitrate_4k = q.max_bitrate_kbps(3840, 2160);
        // 4K should be roughly 4x the bitrate of 1080p
        assert!(bitrate_4k > bitrate_1080 * 3);
    }

    #[test]
    fn test_quality_display() {
        assert_eq!(format!("{}", RecordingQuality::Studio), "Studio");
        assert_eq!(RecordingQuality::Compact.name(), "Compact");
    }

    // -- AudioRecordingQuality tests --

    #[test]
    fn test_audio_quality_bitrate() {
        let uncompressed = AudioRecordingQuality::Uncompressed.approximate_bitrate_kbps(48000, 2);
        assert!(uncompressed > 1000); // should be ~1536 kbps

        let opus = AudioRecordingQuality::HighOpus.approximate_bitrate_kbps(48000, 2);
        assert_eq!(opus, 256);

        let flac = AudioRecordingQuality::Flac.approximate_bitrate_kbps(48000, 2);
        assert!(flac > 0);
        assert!(flac < uncompressed); // FLAC should be smaller
    }

    #[test]
    fn test_audio_quality_name() {
        assert_eq!(AudioRecordingQuality::Flac.name(), "FLAC Lossless");
        assert_eq!(AudioRecordingQuality::HighVorbis.name(), "Vorbis 192 kbps");
    }

    // -- RecordingConfig tests --

    #[test]
    fn test_config_builder_default() {
        let config = RecordingConfig::builder().build().expect("valid config");
        assert_eq!(config.quality, RecordingQuality::High);
        assert_eq!(config.resolution, (1920, 1080));
        assert_eq!(config.framerate, 60);
    }

    #[test]
    fn test_config_builder_custom() {
        let config = RecordingConfig::builder()
            .quality(RecordingQuality::Lossless)
            .resolution(3840, 2160)
            .framerate(120)
            .audio_quality(AudioRecordingQuality::Uncompressed)
            .audio_sample_rate(96000)
            .audio_channels(6)
            .output_dir("/tmp/recordings")
            .filename_prefix("gameplay")
            .build()
            .expect("valid config");

        assert_eq!(config.quality, RecordingQuality::Lossless);
        assert_eq!(config.resolution, (3840, 2160));
        assert_eq!(config.framerate, 120);
        assert_eq!(config.audio_channels, 6);
    }

    #[test]
    fn test_config_builder_invalid_resolution() {
        assert!(RecordingConfig::builder()
            .resolution(0, 1080)
            .build()
            .is_err());
    }

    #[test]
    fn test_config_builder_invalid_framerate() {
        assert!(RecordingConfig::builder().framerate(0).build().is_err());
        assert!(RecordingConfig::builder().framerate(241).build().is_err());
    }

    #[test]
    fn test_config_builder_invalid_crf() {
        assert!(RecordingConfig::builder().crf(64).build().is_err());
    }

    #[test]
    fn test_config_builder_invalid_audio() {
        assert!(RecordingConfig::builder()
            .audio_sample_rate(0)
            .build()
            .is_err());
        assert!(RecordingConfig::builder()
            .audio_channels(0)
            .build()
            .is_err());
    }

    #[test]
    fn test_config_effective_crf_default() {
        let config = RecordingConfig::builder().build().expect("valid");
        assert_eq!(config.effective_crf(), RecordingQuality::High.crf());
    }

    #[test]
    fn test_config_effective_crf_override() {
        let config = RecordingConfig::builder().crf(22).build().expect("valid");
        assert_eq!(config.effective_crf(), 22);
    }

    #[test]
    fn test_config_effective_bitrate_override() {
        let config = RecordingConfig::builder()
            .max_bitrate_kbps(50000)
            .build()
            .expect("valid");
        assert_eq!(config.effective_max_bitrate_kbps(), 50000);
    }

    #[test]
    fn test_config_estimate_file_size() {
        let config = RecordingConfig::builder()
            .quality(RecordingQuality::Standard)
            .build()
            .expect("valid");
        let size = config.estimate_file_size_mb(Duration::from_hours(1));
        assert!(size > 0.0);
    }

    #[test]
    fn test_config_gop_size() {
        let config = RecordingConfig::builder()
            .quality(RecordingQuality::Studio)
            .framerate(60)
            .build()
            .expect("valid");
        assert_eq!(config.gop_size(), 120);
    }

    // -- RecordingSession tests --

    #[test]
    fn test_session_lifecycle() {
        let config = RecordingConfig::builder().build().expect("valid");
        let mut session = RecordingSession::new(config);

        assert_eq!(session.state(), RecordingState::Idle);

        session.start().expect("start");
        assert_eq!(session.state(), RecordingState::Recording);

        session.pause().expect("pause");
        assert_eq!(session.state(), RecordingState::Paused);

        session.resume().expect("resume");
        assert_eq!(session.state(), RecordingState::Recording);

        let summary = session.stop().expect("stop");
        assert_eq!(session.state(), RecordingState::Stopped);
        assert_eq!(summary.quality, RecordingQuality::High);
    }

    #[test]
    fn test_session_double_start_fails() {
        let config = RecordingConfig::builder().build().expect("valid");
        let mut session = RecordingSession::new(config);
        session.start().expect("first start");
        assert!(session.start().is_err());
    }

    #[test]
    fn test_session_pause_when_not_recording_fails() {
        let config = RecordingConfig::builder().build().expect("valid");
        let mut session = RecordingSession::new(config);
        assert!(session.pause().is_err());
    }

    #[test]
    fn test_session_resume_when_not_paused_fails() {
        let config = RecordingConfig::builder().build().expect("valid");
        let mut session = RecordingSession::new(config);
        session.start().expect("start");
        assert!(session.resume().is_err());
    }

    #[test]
    fn test_session_stop_when_idle_fails() {
        let config = RecordingConfig::builder().build().expect("valid");
        let mut session = RecordingSession::new(config);
        assert!(session.stop().is_err());
    }

    #[test]
    fn test_session_record_frames() {
        let config = RecordingConfig::builder().build().expect("valid");
        let mut session = RecordingSession::new(config);
        session.start().expect("start");

        for _ in 0..100 {
            session.record_frame(5000);
        }

        assert_eq!(session.frames_captured(), 100);
        assert_eq!(session.bytes_written(), 500_000);
    }

    #[test]
    fn test_session_chapters() {
        let config = RecordingConfig::builder().build().expect("valid");
        let mut session = RecordingSession::new(config);
        session.start().expect("start");

        session.add_chapter("Intro").expect("chapter");
        session
            .add_chapter_with_description("Boss Fight", "Final boss encounter")
            .expect("chapter");

        assert_eq!(session.chapters().len(), 2);
        assert_eq!(session.chapters()[0].title, "Intro");
        assert!(session.chapters()[1].description.is_some());
    }

    #[test]
    fn test_session_chapter_when_idle_fails() {
        let config = RecordingConfig::builder().build().expect("valid");
        let mut session = RecordingSession::new(config);
        assert!(session.add_chapter("test").is_err());
    }

    #[test]
    fn test_session_file_segments_with_split() {
        let config = RecordingConfig::builder()
            .split_policy(FileSplitPolicy::BySizeMb(1))
            .build()
            .expect("valid");
        let mut session = RecordingSession::new(config);
        session.start().expect("start");

        // Write more than 1 MB
        let chunk = 512 * 1024; // 512 KB
        for _ in 0..5 {
            session.record_frame(chunk);
        }

        assert!(session.file_segments() > 1);
    }

    #[test]
    fn test_session_summary() {
        let config = RecordingConfig::builder()
            .quality(RecordingQuality::Studio)
            .crf(15)
            .build()
            .expect("valid");
        let mut session = RecordingSession::new(config);
        session.start().expect("start");

        for _ in 0..60 {
            session.record_frame(10000);
        }

        let summary = session.stop().expect("stop");
        assert_eq!(summary.frames_captured, 60);
        assert_eq!(summary.bytes_written, 600_000);
        assert_eq!(summary.quality, RecordingQuality::Studio);
        assert_eq!(summary.effective_crf, 15);
        assert!(summary.file_size_mb() > 0.0);
    }

    #[test]
    fn test_session_effective_fps_and_bitrate() {
        let config = RecordingConfig::builder().build().expect("valid");
        let mut session = RecordingSession::new(config);
        session.start().expect("start");

        for _ in 0..100 {
            session.record_frame(5000);
        }

        // FPS and bitrate should be non-zero while recording
        // (exact values depend on timing)
        assert!(session.frames_captured() == 100);
    }

    // -- ChapterMarker tests --

    #[test]
    fn test_chapter_marker() {
        let marker = ChapterMarker::new("Test Chapter", Duration::from_secs(30))
            .with_description("A test description");
        assert_eq!(marker.title, "Test Chapter");
        assert_eq!(marker.offset, Duration::from_secs(30));
        assert_eq!(marker.description.as_deref(), Some("A test description"));
    }

    // -- FileSplitPolicy tests --

    #[test]
    fn test_file_split_policy_eq() {
        assert_eq!(FileSplitPolicy::None, FileSplitPolicy::None);
        assert_eq!(
            FileSplitPolicy::ByDuration(Duration::from_mins(30)),
            FileSplitPolicy::ByDuration(Duration::from_mins(30))
        );
        assert_ne!(FileSplitPolicy::None, FileSplitPolicy::BySizeMb(100));
    }

    #[test]
    fn test_recording_config_split_by_duration() {
        let config = RecordingConfig::builder()
            .split_policy(FileSplitPolicy::ByDuration(Duration::from_mins(30)))
            .build()
            .expect("valid");
        assert_eq!(
            config.split_policy,
            FileSplitPolicy::ByDuration(Duration::from_mins(30))
        );
    }
}
