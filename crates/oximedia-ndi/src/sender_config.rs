//! NDI sender configuration and sender registry.
//!
//! Provides types for configuring NDI senders with different output types
//! (video, audio, data) and a registry to manage multiple named senders.

#![allow(dead_code)]

use std::collections::HashMap;
use std::time::Duration;

/// Classifies the primary output type of an NDI sender.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SenderType {
    /// Full video + audio sender (most common).
    VideoAudio,
    /// Audio-only sender (e.g., return audio feeds).
    AudioOnly,
    /// Video-only sender (no embedded audio).
    VideoOnly,
    /// Data/metadata sender (no media essence).
    DataOnly,
    /// Low-bandwidth proxy sender (reduced quality for preview).
    Proxy,
}

impl SenderType {
    /// Returns true if this sender type produces video frames.
    pub fn has_video(self) -> bool {
        matches!(self, Self::VideoAudio | Self::VideoOnly | Self::Proxy)
    }

    /// Returns true if this sender type produces audio frames.
    pub fn has_audio(self) -> bool {
        matches!(self, Self::VideoAudio | Self::AudioOnly)
    }

    /// Returns a human-readable label.
    pub fn label(self) -> &'static str {
        match self {
            Self::VideoAudio => "video+audio",
            Self::AudioOnly => "audio-only",
            Self::VideoOnly => "video-only",
            Self::DataOnly => "data-only",
            Self::Proxy => "proxy",
        }
    }
}

/// Frame rate specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameRate {
    /// Numerator.
    pub num: u32,
    /// Denominator.
    pub den: u32,
}

impl FrameRate {
    /// Create a new frame rate.
    pub fn new(num: u32, den: u32) -> Self {
        Self { num, den }
    }

    /// 25 fps (PAL).
    pub fn fps25() -> Self {
        Self::new(25, 1)
    }

    /// 29.97 fps (NTSC drop-frame).
    pub fn fps2997() -> Self {
        Self::new(30000, 1001)
    }

    /// 30 fps.
    pub fn fps30() -> Self {
        Self::new(30, 1)
    }

    /// 50 fps.
    pub fn fps50() -> Self {
        Self::new(50, 1)
    }

    /// 59.94 fps.
    pub fn fps5994() -> Self {
        Self::new(60000, 1001)
    }

    /// 60 fps.
    pub fn fps60() -> Self {
        Self::new(60, 1)
    }

    /// Compute the floating-point frame rate value.
    #[allow(clippy::cast_precision_loss)]
    pub fn as_f64(self) -> f64 {
        self.num as f64 / self.den as f64
    }
}

impl std::fmt::Display for FrameRate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.den == 1 {
            write!(f, "{}fps", self.num)
        } else {
            write!(f, "{}/{}", self.num, self.den)
        }
    }
}

/// Quality preset for NDI sender encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualityPreset {
    /// Maximum quality — highest bandwidth.
    Highest,
    /// Balanced quality / bandwidth for normal production.
    Balanced,
    /// Reduced quality for bandwidth-constrained links.
    Proxy,
}

/// Complete configuration for an NDI sender.
#[derive(Debug, Clone)]
pub struct NdiSenderConfig {
    /// Human-readable name for this NDI source (visible to receivers).
    pub name: String,
    /// Optional group names this sender belongs to.
    pub groups: Vec<String>,
    /// The type of content this sender produces.
    pub sender_type: SenderType,
    /// Video frame rate.
    pub frame_rate: FrameRate,
    /// Video width in pixels.
    pub width: u32,
    /// Video height in pixels.
    pub height: u32,
    /// Audio sample rate in Hz.
    pub audio_sample_rate: u32,
    /// Number of audio channels.
    pub audio_channels: u16,
    /// Encoding quality preset.
    pub quality: QualityPreset,
    /// Enable asynchronous (non-blocking) sends.
    pub async_send: bool,
    /// Maximum send queue depth (frames).
    pub send_queue_depth: usize,
    /// Idle timeout: close connection if no data sent for this duration.
    pub idle_timeout: Option<Duration>,
}

impl NdiSenderConfig {
    /// Create a new configuration with sensible defaults.
    pub fn new(name: impl Into<String>, sender_type: SenderType) -> Self {
        Self {
            name: name.into(),
            groups: vec!["public".to_string()],
            sender_type,
            frame_rate: FrameRate::fps2997(),
            width: 1920,
            height: 1080,
            audio_sample_rate: 48000,
            audio_channels: 2,
            quality: QualityPreset::Balanced,
            async_send: true,
            send_queue_depth: 4,
            idle_timeout: Some(Duration::from_secs(30)),
        }
    }

    /// Set the video resolution.
    pub fn with_resolution(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    /// Set the video frame rate.
    pub fn with_frame_rate(mut self, fps: FrameRate) -> Self {
        self.frame_rate = fps;
        self
    }

    /// Add this sender to additional groups.
    pub fn with_group(mut self, group: impl Into<String>) -> Self {
        self.groups.push(group.into());
        self
    }

    /// Set the quality preset.
    pub fn with_quality(mut self, quality: QualityPreset) -> Self {
        self.quality = quality;
        self
    }

    /// Set the idle timeout.
    pub fn with_idle_timeout(mut self, timeout: Option<Duration>) -> Self {
        self.idle_timeout = timeout;
        self
    }

    /// Returns whether this config would produce a 4K output.
    pub fn is_4k(&self) -> bool {
        self.width >= 3840
    }

    /// Returns the approximate video bitrate in Mbit/s (very rough estimate).
    #[allow(clippy::cast_precision_loss)]
    pub fn estimated_bitrate_mbps(&self) -> f64 {
        let pixels_per_sec = self.width as f64 * self.height as f64 * self.frame_rate.as_f64();
        let bits_per_pixel = match self.quality {
            QualityPreset::Highest => 24.0,
            QualityPreset::Balanced => 12.0,
            QualityPreset::Proxy => 4.0,
        };
        pixels_per_sec * bits_per_pixel / 1_000_000.0
    }
}

impl Default for NdiSenderConfig {
    fn default() -> Self {
        Self::new("OxiMedia NDI Sender", SenderType::VideoAudio)
    }
}

/// Registry entry wrapping a config and its active status.
#[derive(Debug)]
struct SenderEntry {
    config: NdiSenderConfig,
    active: bool,
    /// Monotonically increasing version, bumped on each `update`.
    version: u32,
}

impl SenderEntry {
    fn new(config: NdiSenderConfig) -> Self {
        Self {
            config,
            active: false,
            version: 0,
        }
    }
}

/// Manages a collection of named NDI sender configurations.
#[derive(Debug, Default)]
pub struct SenderRegistry {
    senders: HashMap<String, SenderEntry>,
}

impl SenderRegistry {
    /// Create a new, empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new sender configuration.
    ///
    /// If a sender with the same name already exists, it is replaced.
    pub fn register(&mut self, config: NdiSenderConfig) {
        self.senders
            .insert(config.name.clone(), SenderEntry::new(config));
    }

    /// Remove a sender from the registry. Returns true if it existed.
    pub fn deregister(&mut self, name: &str) -> bool {
        self.senders.remove(name).is_some()
    }

    /// Update the configuration for an existing sender and bump its version.
    ///
    /// Returns false if the sender is not registered.
    pub fn update(&mut self, config: NdiSenderConfig) -> bool {
        if let Some(entry) = self.senders.get_mut(&config.name) {
            entry.version += 1;
            entry.config = config;
            true
        } else {
            false
        }
    }

    /// Mark a sender as active (currently streaming).
    pub fn set_active(&mut self, name: &str, active: bool) -> bool {
        if let Some(entry) = self.senders.get_mut(name) {
            entry.active = active;
            true
        } else {
            false
        }
    }

    /// Get a reference to a sender's configuration.
    pub fn get(&self, name: &str) -> Option<&NdiSenderConfig> {
        self.senders.get(name).map(|e| &e.config)
    }

    /// Returns whether a sender is currently active.
    pub fn is_active(&self, name: &str) -> bool {
        self.senders.get(name).map_or(false, |e| e.active)
    }

    /// Returns the config version number for a sender.
    pub fn version(&self, name: &str) -> Option<u32> {
        self.senders.get(name).map(|e| e.version)
    }

    /// Returns the total number of registered senders.
    pub fn count(&self) -> usize {
        self.senders.len()
    }

    /// Returns the number of currently active senders.
    pub fn active_count(&self) -> usize {
        self.senders.values().filter(|e| e.active).count()
    }

    /// Returns names of all active senders.
    pub fn active_names(&self) -> Vec<String> {
        self.senders
            .iter()
            .filter(|(_, e)| e.active)
            .map(|(k, _)| k.clone())
            .collect()
    }

    /// Returns names of all registered senders.
    pub fn all_names(&self) -> Vec<String> {
        self.senders.keys().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sender_type_has_video() {
        assert!(SenderType::VideoAudio.has_video());
        assert!(SenderType::VideoOnly.has_video());
        assert!(SenderType::Proxy.has_video());
        assert!(!SenderType::AudioOnly.has_video());
        assert!(!SenderType::DataOnly.has_video());
    }

    #[test]
    fn test_sender_type_has_audio() {
        assert!(SenderType::VideoAudio.has_audio());
        assert!(SenderType::AudioOnly.has_audio());
        assert!(!SenderType::VideoOnly.has_audio());
        assert!(!SenderType::DataOnly.has_audio());
    }

    #[test]
    fn test_sender_type_label() {
        assert_eq!(SenderType::VideoAudio.label(), "video+audio");
        assert_eq!(SenderType::Proxy.label(), "proxy");
        assert_eq!(SenderType::DataOnly.label(), "data-only");
    }

    #[test]
    fn test_frame_rate_fps25() {
        let fps = FrameRate::fps25();
        assert!((fps.as_f64() - 25.0).abs() < 1e-9);
    }

    #[test]
    fn test_frame_rate_fps2997() {
        let fps = FrameRate::fps2997();
        assert!((fps.as_f64() - (30000.0 / 1001.0)).abs() < 1e-9);
    }

    #[test]
    fn test_frame_rate_display() {
        assert_eq!(format!("{}", FrameRate::fps30()), "30fps");
        assert_eq!(format!("{}", FrameRate::fps2997()), "30000/1001");
    }

    #[test]
    fn test_sender_config_default() {
        let cfg = NdiSenderConfig::default();
        assert_eq!(cfg.sender_type, SenderType::VideoAudio);
        assert_eq!(cfg.width, 1920);
        assert_eq!(cfg.height, 1080);
        assert!(cfg.async_send);
    }

    #[test]
    fn test_sender_config_builder() {
        let cfg = NdiSenderConfig::new("TestSrc", SenderType::VideoOnly)
            .with_resolution(3840, 2160)
            .with_frame_rate(FrameRate::fps50())
            .with_group("studio")
            .with_quality(QualityPreset::Highest)
            .with_idle_timeout(None);
        assert_eq!(cfg.width, 3840);
        assert_eq!(cfg.height, 2160);
        assert_eq!(cfg.quality, QualityPreset::Highest);
        assert!(cfg.is_4k());
        assert!(cfg.groups.contains(&"studio".to_string()));
        assert!(cfg.idle_timeout.is_none());
    }

    #[test]
    fn test_sender_config_is_not_4k() {
        let cfg = NdiSenderConfig::default();
        assert!(!cfg.is_4k());
    }

    #[test]
    fn test_sender_config_estimated_bitrate() {
        let cfg = NdiSenderConfig::default(); // 1920x1080 @ ~30fps, balanced
        let bps = cfg.estimated_bitrate_mbps();
        assert!(bps > 0.0);
    }

    #[test]
    fn test_registry_register_and_get() {
        let mut reg = SenderRegistry::new();
        reg.register(NdiSenderConfig::new("Cam1", SenderType::VideoAudio));
        assert!(reg.get("Cam1").is_some());
        assert_eq!(reg.count(), 1);
    }

    #[test]
    fn test_registry_deregister() {
        let mut reg = SenderRegistry::new();
        reg.register(NdiSenderConfig::new("Cam1", SenderType::VideoAudio));
        assert!(reg.deregister("Cam1"));
        assert!(!reg.deregister("Cam1"));
        assert_eq!(reg.count(), 0);
    }

    #[test]
    fn test_registry_update_bumps_version() {
        let mut reg = SenderRegistry::new();
        reg.register(NdiSenderConfig::new("Cam1", SenderType::VideoAudio));
        assert_eq!(reg.version("Cam1"), Some(0));
        let cfg2 = NdiSenderConfig::new("Cam1", SenderType::VideoAudio).with_resolution(1280, 720);
        assert!(reg.update(cfg2));
        assert_eq!(reg.version("Cam1"), Some(1));
        assert_eq!(reg.get("Cam1").expect("expected key to exist").width, 1280);
    }

    #[test]
    fn test_registry_update_nonexistent_returns_false() {
        let mut reg = SenderRegistry::new();
        let result = reg.update(NdiSenderConfig::new("Ghost", SenderType::AudioOnly));
        assert!(!result);
    }

    #[test]
    fn test_registry_active_count() {
        let mut reg = SenderRegistry::new();
        reg.register(NdiSenderConfig::new("Cam1", SenderType::VideoAudio));
        reg.register(NdiSenderConfig::new("Cam2", SenderType::AudioOnly));
        reg.set_active("Cam1", true);
        assert_eq!(reg.active_count(), 1);
        assert!(reg.is_active("Cam1"));
        assert!(!reg.is_active("Cam2"));
    }

    #[test]
    fn test_registry_active_names() {
        let mut reg = SenderRegistry::new();
        reg.register(NdiSenderConfig::new("Cam1", SenderType::VideoAudio));
        reg.register(NdiSenderConfig::new("Cam2", SenderType::VideoAudio));
        reg.set_active("Cam2", true);
        let names = reg.active_names();
        assert!(names.contains(&"Cam2".to_string()));
        assert!(!names.contains(&"Cam1".to_string()));
    }

    #[test]
    fn test_registry_all_names() {
        let mut reg = SenderRegistry::new();
        reg.register(NdiSenderConfig::new("A", SenderType::VideoAudio));
        reg.register(NdiSenderConfig::new("B", SenderType::AudioOnly));
        let mut names = reg.all_names();
        names.sort();
        assert_eq!(names, vec!["A", "B"]);
    }
}
