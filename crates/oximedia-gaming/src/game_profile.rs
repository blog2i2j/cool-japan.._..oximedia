#![allow(dead_code)]

//! Game-specific optimization profiles for streaming and recording.
//!
//! Provides pre-configured encoder, capture, and bitrate settings tuned for
//! different game genres (FPS, MOBA, strategy, racing, etc.) and target
//! platforms (Twitch, YouTube, local recording).

use std::collections::HashMap;

/// Frames per second target.
pub type Fps = u32;

/// Bitrate in kilobits per second.
pub type BitrateKbps = u32;

/// Game genre classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GameGenre {
    /// First-person shooter (fast motion, low latency critical).
    Fps,
    /// Third-person action.
    ThirdPersonAction,
    /// Multiplayer online battle arena.
    Moba,
    /// Real-time strategy.
    Strategy,
    /// Racing / driving.
    Racing,
    /// Sports.
    Sports,
    /// Role-playing game.
    Rpg,
    /// Sandbox / creative.
    Sandbox,
    /// Retro / 2D / pixel art.
    Retro,
    /// Virtual reality.
    Vr,
}

/// Target streaming/recording platform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TargetPlatform {
    /// Twitch (RTMP, 6000 kbps max for non-partners).
    Twitch,
    /// YouTube Gaming (higher bitrate allowed).
    YouTube,
    /// Facebook Gaming.
    Facebook,
    /// Local file recording (no bitrate ceiling).
    LocalRecording,
    /// Custom / generic.
    Custom,
}

/// Encoder tuning hint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncoderTuning {
    /// Zero-latency tuning (no B-frames, single-pass).
    ZeroLatency,
    /// Low-latency with some look-ahead.
    LowLatency,
    /// Balanced quality / latency.
    Balanced,
    /// Film-quality (higher latency, best quality).
    Film,
}

/// Resolution preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Resolution {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

impl Resolution {
    /// Create a new resolution.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    /// 720p preset.
    #[must_use]
    pub fn hd() -> Self {
        Self::new(1280, 720)
    }

    /// 1080p preset.
    #[must_use]
    pub fn full_hd() -> Self {
        Self::new(1920, 1080)
    }

    /// 1440p preset.
    #[must_use]
    pub fn qhd() -> Self {
        Self::new(2560, 1440)
    }

    /// 4K preset.
    #[must_use]
    pub fn uhd() -> Self {
        Self::new(3840, 2160)
    }

    /// Total pixel count.
    #[must_use]
    pub fn pixel_count(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }
}

/// Complete optimization profile for a game/platform combination.
#[derive(Debug, Clone)]
pub struct GameProfile {
    /// Human-readable name.
    pub name: String,
    /// Game genre.
    pub genre: GameGenre,
    /// Target platform.
    pub platform: TargetPlatform,
    /// Output resolution.
    pub resolution: Resolution,
    /// Target framerate.
    pub fps: Fps,
    /// Target video bitrate.
    pub video_bitrate: BitrateKbps,
    /// Audio bitrate in kbps.
    pub audio_bitrate: BitrateKbps,
    /// Encoder tuning.
    pub tuning: EncoderTuning,
    /// Number of B-frames (0 for zero-latency).
    pub b_frames: u32,
    /// Key frame interval in seconds.
    pub keyframe_interval_secs: u32,
    /// Whether to use variable bitrate.
    pub use_vbr: bool,
}

impl GameProfile {
    /// Create a profile builder.
    #[must_use]
    pub fn builder(name: &str) -> GameProfileBuilder {
        GameProfileBuilder::new(name)
    }

    /// Estimate bandwidth usage in kilobits per second (video + audio).
    #[must_use]
    pub fn estimated_bandwidth(&self) -> BitrateKbps {
        self.video_bitrate + self.audio_bitrate
    }

    /// Check if this profile is suitable for low-bandwidth connections (< 4000 kbps total).
    #[must_use]
    pub fn is_low_bandwidth(&self) -> bool {
        self.estimated_bandwidth() < 4000
    }

    /// Estimate file size per hour of recording in megabytes.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn estimated_mb_per_hour(&self) -> f64 {
        let total_kbps = self.estimated_bandwidth();
        // kbps -> kB/s -> MB/h: kbps / 8 * 3600 / 1024
        total_kbps as f64 / 8.0 * 3600.0 / 1024.0
    }
}

/// Builder for `GameProfile`.
#[derive(Debug, Clone)]
pub struct GameProfileBuilder {
    name: String,
    genre: GameGenre,
    platform: TargetPlatform,
    resolution: Resolution,
    fps: Fps,
    video_bitrate: BitrateKbps,
    audio_bitrate: BitrateKbps,
    tuning: EncoderTuning,
    b_frames: u32,
    keyframe_interval_secs: u32,
    use_vbr: bool,
}

impl GameProfileBuilder {
    /// Create a new builder.
    #[must_use]
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            genre: GameGenre::Fps,
            platform: TargetPlatform::Twitch,
            resolution: Resolution::full_hd(),
            fps: 60,
            video_bitrate: 6000,
            audio_bitrate: 160,
            tuning: EncoderTuning::LowLatency,
            b_frames: 2,
            keyframe_interval_secs: 2,
            use_vbr: true,
        }
    }

    /// Set genre.
    #[must_use]
    pub fn genre(mut self, genre: GameGenre) -> Self {
        self.genre = genre;
        self
    }

    /// Set platform.
    #[must_use]
    pub fn platform(mut self, platform: TargetPlatform) -> Self {
        self.platform = platform;
        self
    }

    /// Set resolution.
    #[must_use]
    pub fn resolution(mut self, res: Resolution) -> Self {
        self.resolution = res;
        self
    }

    /// Set framerate.
    #[must_use]
    pub fn fps(mut self, fps: Fps) -> Self {
        self.fps = fps;
        self
    }

    /// Set video bitrate.
    #[must_use]
    pub fn video_bitrate(mut self, kbps: BitrateKbps) -> Self {
        self.video_bitrate = kbps;
        self
    }

    /// Set audio bitrate.
    #[must_use]
    pub fn audio_bitrate(mut self, kbps: BitrateKbps) -> Self {
        self.audio_bitrate = kbps;
        self
    }

    /// Set encoder tuning.
    #[must_use]
    pub fn tuning(mut self, tuning: EncoderTuning) -> Self {
        self.tuning = tuning;
        self
    }

    /// Set B-frame count.
    #[must_use]
    pub fn b_frames(mut self, count: u32) -> Self {
        self.b_frames = count;
        self
    }

    /// Set keyframe interval.
    #[must_use]
    pub fn keyframe_interval(mut self, secs: u32) -> Self {
        self.keyframe_interval_secs = secs;
        self
    }

    /// Set VBR mode.
    #[must_use]
    pub fn use_vbr(mut self, vbr: bool) -> Self {
        self.use_vbr = vbr;
        self
    }

    /// Build the profile.
    #[must_use]
    pub fn build(self) -> GameProfile {
        GameProfile {
            name: self.name,
            genre: self.genre,
            platform: self.platform,
            resolution: self.resolution,
            fps: self.fps,
            video_bitrate: self.video_bitrate,
            audio_bitrate: self.audio_bitrate,
            tuning: self.tuning,
            b_frames: self.b_frames,
            keyframe_interval_secs: self.keyframe_interval_secs,
            use_vbr: self.use_vbr,
        }
    }
}

/// Library of pre-built game profiles.
#[derive(Debug)]
pub struct ProfileLibrary {
    /// Stored profiles keyed by name.
    profiles: HashMap<String, GameProfile>,
}

impl Default for ProfileLibrary {
    fn default() -> Self {
        Self::new()
    }
}

impl ProfileLibrary {
    /// Create a library with default built-in profiles.
    #[must_use]
    pub fn new() -> Self {
        let mut lib = Self {
            profiles: HashMap::new(),
        };
        lib.register_defaults();
        lib
    }

    /// Register built-in profiles.
    fn register_defaults(&mut self) {
        self.add(
            GameProfile::builder("fps_twitch_1080p60")
                .genre(GameGenre::Fps)
                .platform(TargetPlatform::Twitch)
                .resolution(Resolution::full_hd())
                .fps(60)
                .video_bitrate(6000)
                .tuning(EncoderTuning::ZeroLatency)
                .b_frames(0)
                .build(),
        );

        self.add(
            GameProfile::builder("moba_twitch_1080p60")
                .genre(GameGenre::Moba)
                .platform(TargetPlatform::Twitch)
                .resolution(Resolution::full_hd())
                .fps(60)
                .video_bitrate(6000)
                .tuning(EncoderTuning::LowLatency)
                .b_frames(2)
                .build(),
        );

        self.add(
            GameProfile::builder("strategy_youtube_1440p30")
                .genre(GameGenre::Strategy)
                .platform(TargetPlatform::YouTube)
                .resolution(Resolution::qhd())
                .fps(30)
                .video_bitrate(9000)
                .tuning(EncoderTuning::Balanced)
                .b_frames(3)
                .build(),
        );

        self.add(
            GameProfile::builder("racing_local_4k60")
                .genre(GameGenre::Racing)
                .platform(TargetPlatform::LocalRecording)
                .resolution(Resolution::uhd())
                .fps(60)
                .video_bitrate(50_000)
                .tuning(EncoderTuning::Film)
                .b_frames(3)
                .build(),
        );

        self.add(
            GameProfile::builder("retro_twitch_720p60")
                .genre(GameGenre::Retro)
                .platform(TargetPlatform::Twitch)
                .resolution(Resolution::hd())
                .fps(60)
                .video_bitrate(3000)
                .tuning(EncoderTuning::LowLatency)
                .b_frames(0)
                .build(),
        );
    }

    /// Add a profile to the library.
    pub fn add(&mut self, profile: GameProfile) {
        self.profiles.insert(profile.name.clone(), profile);
    }

    /// Look up a profile by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&GameProfile> {
        self.profiles.get(name)
    }

    /// List all profile names.
    #[must_use]
    pub fn names(&self) -> Vec<&str> {
        self.profiles.keys().map(String::as_str).collect()
    }

    /// Find profiles matching a given genre.
    #[must_use]
    pub fn by_genre(&self, genre: GameGenre) -> Vec<&GameProfile> {
        self.profiles
            .values()
            .filter(|p| p.genre == genre)
            .collect()
    }

    /// Find profiles matching a given platform.
    #[must_use]
    pub fn by_platform(&self, platform: TargetPlatform) -> Vec<&GameProfile> {
        self.profiles
            .values()
            .filter(|p| p.platform == platform)
            .collect()
    }

    /// Return the number of profiles in the library.
    #[must_use]
    pub fn len(&self) -> usize {
        self.profiles.len()
    }

    /// Check if the library is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.profiles.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolution_presets() {
        assert_eq!(Resolution::hd().width, 1280);
        assert_eq!(Resolution::full_hd().width, 1920);
        assert_eq!(Resolution::qhd().width, 2560);
        assert_eq!(Resolution::uhd().width, 3840);
    }

    #[test]
    fn test_resolution_pixel_count() {
        let r = Resolution::full_hd();
        assert_eq!(r.pixel_count(), 1920 * 1080);
    }

    #[test]
    fn test_profile_builder() {
        let profile = GameProfile::builder("test")
            .genre(GameGenre::Fps)
            .platform(TargetPlatform::Twitch)
            .fps(120)
            .video_bitrate(8000)
            .build();
        assert_eq!(profile.name, "test");
        assert_eq!(profile.genre, GameGenre::Fps);
        assert_eq!(profile.fps, 120);
        assert_eq!(profile.video_bitrate, 8000);
    }

    #[test]
    fn test_estimated_bandwidth() {
        let profile = GameProfile::builder("bw_test")
            .video_bitrate(6000)
            .audio_bitrate(160)
            .build();
        assert_eq!(profile.estimated_bandwidth(), 6160);
    }

    #[test]
    fn test_is_low_bandwidth() {
        let low = GameProfile::builder("low")
            .video_bitrate(2000)
            .audio_bitrate(128)
            .build();
        assert!(low.is_low_bandwidth());

        let high = GameProfile::builder("high")
            .video_bitrate(8000)
            .audio_bitrate(320)
            .build();
        assert!(!high.is_low_bandwidth());
    }

    #[test]
    fn test_estimated_mb_per_hour() {
        let profile = GameProfile::builder("mbh")
            .video_bitrate(6000)
            .audio_bitrate(0)
            .build();
        let mb = profile.estimated_mb_per_hour();
        // 6000 kbps => 750 kB/s => 2700000 kB/h => ~2636 MB/h
        assert!(mb > 2600.0);
        assert!(mb < 2700.0);
    }

    #[test]
    fn test_profile_library_defaults() {
        let lib = ProfileLibrary::new();
        assert!(!lib.is_empty());
        assert!(lib.len() >= 5);
    }

    #[test]
    fn test_profile_library_get() {
        let lib = ProfileLibrary::new();
        let profile = lib.get("fps_twitch_1080p60");
        assert!(profile.is_some());
        let p = profile.expect("profile should exist");
        assert_eq!(p.genre, GameGenre::Fps);
        assert_eq!(p.tuning, EncoderTuning::ZeroLatency);
    }

    #[test]
    fn test_profile_library_by_genre() {
        let lib = ProfileLibrary::new();
        let fps_profiles = lib.by_genre(GameGenre::Fps);
        assert!(!fps_profiles.is_empty());
        for p in &fps_profiles {
            assert_eq!(p.genre, GameGenre::Fps);
        }
    }

    #[test]
    fn test_profile_library_by_platform() {
        let lib = ProfileLibrary::new();
        let twitch_profiles = lib.by_platform(TargetPlatform::Twitch);
        assert!(!twitch_profiles.is_empty());
        for p in &twitch_profiles {
            assert_eq!(p.platform, TargetPlatform::Twitch);
        }
    }

    #[test]
    fn test_profile_library_add_custom() {
        let mut lib = ProfileLibrary::new();
        let initial = lib.len();
        lib.add(
            GameProfile::builder("custom_profile")
                .genre(GameGenre::Vr)
                .platform(TargetPlatform::Custom)
                .fps(90)
                .video_bitrate(20_000)
                .build(),
        );
        assert_eq!(lib.len(), initial + 1);
        assert!(lib.get("custom_profile").is_some());
    }

    #[test]
    fn test_profile_library_names() {
        let lib = ProfileLibrary::new();
        let names = lib.names();
        assert!(!names.is_empty());
        assert!(names.contains(&"fps_twitch_1080p60"));
    }

    #[test]
    fn test_game_genre_variants() {
        let genres = [
            GameGenre::Fps,
            GameGenre::ThirdPersonAction,
            GameGenre::Moba,
            GameGenre::Strategy,
            GameGenre::Racing,
            GameGenre::Sports,
            GameGenre::Rpg,
            GameGenre::Sandbox,
            GameGenre::Retro,
            GameGenre::Vr,
        ];
        // All should be distinct
        for (i, g1) in genres.iter().enumerate() {
            for (j, g2) in genres.iter().enumerate() {
                if i != j {
                    assert_ne!(g1, g2);
                }
            }
        }
    }
}
