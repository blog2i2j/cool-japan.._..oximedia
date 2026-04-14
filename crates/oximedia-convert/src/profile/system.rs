// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Profile management system.

use crate::{ConversionOptions, ConversionSettings, MediaProperties, Result};

/// Predefined conversion profiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Profile {
    /// MP4/H.264 optimized for web playback
    WebOptimized,
    /// HLS/DASH streaming variants
    Streaming,
    /// Lossless preservation
    Archive,
    /// Small file size for email
    Email,
    /// Mobile-optimized format
    Mobile,
    /// `YouTube` upload optimization
    YouTube,
    /// Instagram video optimization
    Instagram,
    /// `TikTok` video optimization
    TikTok,
    /// Broadcast-compliant format
    Broadcast,
    /// Extract audio as MP3
    AudioMp3,
    /// Extract audio as FLAC
    AudioFlac,
    /// Extract audio as AAC
    AudioAac,
}

impl Profile {
    /// Get the name of the profile.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::WebOptimized => "Web Optimized",
            Self::Streaming => "Streaming",
            Self::Archive => "Archive",
            Self::Email => "Email",
            Self::Mobile => "Mobile",
            Self::YouTube => "YouTube",
            Self::Instagram => "Instagram",
            Self::TikTok => "TikTok",
            Self::Broadcast => "Broadcast",
            Self::AudioMp3 => "Audio MP3",
            Self::AudioFlac => "Audio FLAC",
            Self::AudioAac => "Audio AAC",
        }
    }

    /// Get the description of the profile.
    #[must_use]
    pub fn description(&self) -> &'static str {
        match self {
            Self::WebOptimized => "MP4 with H.264 video and AAC audio, optimized for web playback",
            Self::Streaming => "HLS/DASH variants for adaptive streaming",
            Self::Archive => "Lossless MKV format for long-term preservation",
            Self::Email => "Highly compressed MP4 for easy email sharing",
            Self::Mobile => "Optimized for mobile devices with lower bitrates",
            Self::YouTube => "Optimized for YouTube upload (1080p, high bitrate)",
            Self::Instagram => "Instagram-compliant format (max 60s, 1080x1080)",
            Self::TikTok => "TikTok-compliant format (vertical 9:16, up to 60s)",
            Self::Broadcast => "Broadcast-compliant MXF format",
            Self::AudioMp3 => "Extract audio as MP3 (192 kbps)",
            Self::AudioFlac => "Extract audio as FLAC (lossless)",
            Self::AudioAac => "Extract audio as AAC (256 kbps)",
        }
    }

    /// Apply this profile to create conversion settings.
    pub fn apply(
        &self,
        source: &MediaProperties,
        options: &ConversionOptions,
    ) -> Result<ConversionSettings> {
        match self {
            Self::WebOptimized => self.apply_web_optimized(source, options),
            Self::Streaming => self.apply_streaming(source, options),
            Self::Archive => self.apply_archive(source, options),
            Self::Email => self.apply_email(source, options),
            Self::Mobile => self.apply_mobile(source, options),
            Self::YouTube => self.apply_youtube(source, options),
            Self::Instagram => self.apply_instagram(source, options),
            Self::TikTok => self.apply_tiktok(source, options),
            Self::Broadcast => self.apply_broadcast(source, options),
            Self::AudioMp3 => self.apply_audio_mp3(source, options),
            Self::AudioFlac => self.apply_audio_flac(source, options),
            Self::AudioAac => self.apply_audio_aac(source, options),
        }
    }

    fn apply_web_optimized(
        &self,
        source: &MediaProperties,
        options: &ConversionOptions,
    ) -> Result<ConversionSettings> {
        let max_resolution = options.max_resolution.unwrap_or((1920, 1080));
        let resolution = calculate_resolution(source, max_resolution);

        Ok(ConversionSettings {
            format: "mp4".to_string(),
            video_codec: Some("h264".to_string()),
            audio_codec: Some("aac".to_string()),
            video_bitrate: options.target_bitrate.or(Some(5_000_000)),
            audio_bitrate: Some(192_000),
            resolution,
            frame_rate: source.frame_rate.or(Some(30.0)),
            parameters: vec![
                ("preset".to_string(), "medium".to_string()),
                ("movflags".to_string(), "+faststart".to_string()),
            ],
        })
    }

    fn apply_streaming(
        &self,
        source: &MediaProperties,
        _options: &ConversionOptions,
    ) -> Result<ConversionSettings> {
        Ok(ConversionSettings {
            format: "hls".to_string(),
            video_codec: Some("h264".to_string()),
            audio_codec: Some("aac".to_string()),
            video_bitrate: Some(4_000_000),
            audio_bitrate: Some(128_000),
            resolution: source.width.and_then(|w| source.height.map(|h| (w, h))),
            frame_rate: source.frame_rate,
            parameters: vec![
                ("hls_time".to_string(), "6".to_string()),
                ("hls_list_size".to_string(), "0".to_string()),
            ],
        })
    }

    fn apply_archive(
        &self,
        source: &MediaProperties,
        _options: &ConversionOptions,
    ) -> Result<ConversionSettings> {
        Ok(ConversionSettings {
            format: "mkv".to_string(),
            video_codec: Some("ffv1".to_string()),
            audio_codec: Some("flac".to_string()),
            video_bitrate: None,
            audio_bitrate: None,
            resolution: source.width.and_then(|w| source.height.map(|h| (w, h))),
            frame_rate: source.frame_rate,
            parameters: vec![
                ("level".to_string(), "3".to_string()),
                ("coder".to_string(), "1".to_string()),
            ],
        })
    }

    fn apply_email(
        &self,
        source: &MediaProperties,
        _options: &ConversionOptions,
    ) -> Result<ConversionSettings> {
        let resolution = calculate_resolution(source, (640, 480));

        Ok(ConversionSettings {
            format: "mp4".to_string(),
            video_codec: Some("h264".to_string()),
            audio_codec: Some("aac".to_string()),
            video_bitrate: Some(1_000_000),
            audio_bitrate: Some(96_000),
            resolution,
            frame_rate: Some(24.0),
            parameters: vec![
                ("preset".to_string(), "fast".to_string()),
                ("crf".to_string(), "28".to_string()),
            ],
        })
    }

    fn apply_mobile(
        &self,
        source: &MediaProperties,
        _options: &ConversionOptions,
    ) -> Result<ConversionSettings> {
        let resolution = calculate_resolution(source, (1280, 720));

        Ok(ConversionSettings {
            format: "mp4".to_string(),
            video_codec: Some("h264".to_string()),
            audio_codec: Some("aac".to_string()),
            video_bitrate: Some(2_500_000),
            audio_bitrate: Some(128_000),
            resolution,
            frame_rate: Some(30.0),
            parameters: vec![
                ("preset".to_string(), "medium".to_string()),
                ("profile".to_string(), "baseline".to_string()),
            ],
        })
    }

    fn apply_youtube(
        &self,
        source: &MediaProperties,
        _options: &ConversionOptions,
    ) -> Result<ConversionSettings> {
        let resolution = calculate_resolution(source, (1920, 1080));

        Ok(ConversionSettings {
            format: "mp4".to_string(),
            video_codec: Some("h264".to_string()),
            audio_codec: Some("aac".to_string()),
            video_bitrate: Some(8_000_000),
            audio_bitrate: Some(192_000),
            resolution,
            frame_rate: source.frame_rate.or(Some(30.0)),
            parameters: vec![
                ("preset".to_string(), "slow".to_string()),
                ("crf".to_string(), "18".to_string()),
            ],
        })
    }

    fn apply_instagram(
        &self,
        _source: &MediaProperties,
        _options: &ConversionOptions,
    ) -> Result<ConversionSettings> {
        Ok(ConversionSettings {
            format: "mp4".to_string(),
            video_codec: Some("h264".to_string()),
            audio_codec: Some("aac".to_string()),
            video_bitrate: Some(3_500_000),
            audio_bitrate: Some(128_000),
            resolution: Some((1080, 1080)),
            frame_rate: Some(30.0),
            parameters: vec![
                ("preset".to_string(), "medium".to_string()),
                ("pix_fmt".to_string(), "yuv420p".to_string()),
            ],
        })
    }

    fn apply_tiktok(
        &self,
        _source: &MediaProperties,
        _options: &ConversionOptions,
    ) -> Result<ConversionSettings> {
        Ok(ConversionSettings {
            format: "mp4".to_string(),
            video_codec: Some("h264".to_string()),
            audio_codec: Some("aac".to_string()),
            video_bitrate: Some(4_000_000),
            audio_bitrate: Some(128_000),
            resolution: Some((1080, 1920)),
            frame_rate: Some(30.0),
            parameters: vec![
                ("preset".to_string(), "medium".to_string()),
                ("pix_fmt".to_string(), "yuv420p".to_string()),
            ],
        })
    }

    fn apply_broadcast(
        &self,
        source: &MediaProperties,
        _options: &ConversionOptions,
    ) -> Result<ConversionSettings> {
        Ok(ConversionSettings {
            format: "mxf".to_string(),
            video_codec: Some("mpeg2video".to_string()),
            audio_codec: Some("pcm".to_string()),
            video_bitrate: Some(50_000_000),
            audio_bitrate: None,
            resolution: source.width.and_then(|w| source.height.map(|h| (w, h))),
            frame_rate: Some(25.0),
            parameters: vec![
                ("g".to_string(), "1".to_string()),
                ("intra".to_string(), "1".to_string()),
            ],
        })
    }

    fn apply_audio_mp3(
        &self,
        _source: &MediaProperties,
        _options: &ConversionOptions,
    ) -> Result<ConversionSettings> {
        Ok(ConversionSettings {
            format: "mp3".to_string(),
            video_codec: None,
            audio_codec: Some("mp3".to_string()),
            video_bitrate: None,
            audio_bitrate: Some(192_000),
            resolution: None,
            frame_rate: None,
            parameters: vec![],
        })
    }

    fn apply_audio_flac(
        &self,
        _source: &MediaProperties,
        _options: &ConversionOptions,
    ) -> Result<ConversionSettings> {
        Ok(ConversionSettings {
            format: "flac".to_string(),
            video_codec: None,
            audio_codec: Some("flac".to_string()),
            video_bitrate: None,
            audio_bitrate: None,
            resolution: None,
            frame_rate: None,
            parameters: vec![],
        })
    }

    fn apply_audio_aac(
        &self,
        _source: &MediaProperties,
        _options: &ConversionOptions,
    ) -> Result<ConversionSettings> {
        Ok(ConversionSettings {
            format: "m4a".to_string(),
            video_codec: None,
            audio_codec: Some("aac".to_string()),
            video_bitrate: None,
            audio_bitrate: Some(256_000),
            resolution: None,
            frame_rate: None,
            parameters: vec![],
        })
    }
}

/// System for managing conversion profiles.
#[derive(Debug, Clone)]
pub struct ProfileSystem {
    custom_profiles: Vec<CustomProfile>,
}

impl ProfileSystem {
    /// Create a new profile system.
    #[must_use]
    pub fn new() -> Self {
        Self {
            custom_profiles: Vec::new(),
        }
    }

    /// Get a profile by reference.
    pub fn get_profile<'a>(&self, profile: &'a Profile) -> Result<&'a Profile> {
        Ok(profile)
    }

    /// Add a custom profile.
    pub fn add_custom_profile(&mut self, profile: CustomProfile) {
        self.custom_profiles.push(profile);
    }

    /// Get a custom profile by name.
    #[must_use]
    pub fn get_custom_profile(&self, name: &str) -> Option<&CustomProfile> {
        self.custom_profiles.iter().find(|p| p.name == name)
    }

    /// List all available profiles.
    #[must_use]
    pub fn list_profiles(&self) -> Vec<&'static str> {
        vec![
            Profile::WebOptimized.name(),
            Profile::Streaming.name(),
            Profile::Archive.name(),
            Profile::Email.name(),
            Profile::Mobile.name(),
            Profile::YouTube.name(),
            Profile::Instagram.name(),
            Profile::TikTok.name(),
            Profile::Broadcast.name(),
            Profile::AudioMp3.name(),
            Profile::AudioFlac.name(),
            Profile::AudioAac.name(),
        ]
    }
}

impl Default for ProfileSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// A custom conversion profile.
#[derive(Debug, Clone)]
pub struct CustomProfile {
    /// Profile name
    pub name: String,
    /// Profile description
    pub description: String,
    /// Conversion settings
    pub settings: ConversionSettings,
}

fn calculate_resolution(source: &MediaProperties, max: (u32, u32)) -> Option<(u32, u32)> {
    let (src_width, src_height) = match (source.width, source.height) {
        (Some(w), Some(h)) => (w, h),
        _ => return None,
    };

    let (max_width, max_height) = max;

    if src_width <= max_width && src_height <= max_height {
        return Some((src_width, src_height));
    }

    let width_ratio = f64::from(max_width) / f64::from(src_width);
    let height_ratio = f64::from(max_height) / f64::from(src_height);
    let ratio = width_ratio.min(height_ratio);

    let new_width = (f64::from(src_width) * ratio) as u32;
    let new_height = (f64::from(src_height) * ratio) as u32;

    // Round to even numbers
    let new_width = new_width - (new_width % 2);
    let new_height = new_height - (new_height % 2);

    Some((new_width, new_height))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn create_test_properties() -> MediaProperties {
        MediaProperties {
            format: "mp4".to_string(),
            file_size: 1024 * 1024,
            duration: Some(Duration::from_secs(60)),
            width: Some(1920),
            height: Some(1080),
            video_codec: Some("h264".to_string()),
            audio_codec: Some("aac".to_string()),
            video_bitrate: Some(5_000_000),
            audio_bitrate: Some(128_000),
            frame_rate: Some(30.0),
            audio_sample_rate: Some(48000),
            audio_channels: Some(2),
        }
    }

    #[test]
    fn test_profile_names() {
        assert_eq!(Profile::WebOptimized.name(), "Web Optimized");
        assert_eq!(Profile::YouTube.name(), "YouTube");
    }

    #[test]
    fn test_web_optimized_profile() {
        let source = create_test_properties();
        let options = ConversionOptions::default();
        let settings = Profile::WebOptimized.apply(&source, &options).unwrap();

        assert_eq!(settings.format, "mp4");
        assert_eq!(settings.video_codec, Some("h264".to_string()));
        assert_eq!(settings.audio_codec, Some("aac".to_string()));
    }

    #[test]
    fn test_audio_profiles() {
        let source = create_test_properties();
        let options = ConversionOptions::default();

        let mp3 = Profile::AudioMp3.apply(&source, &options).unwrap();
        assert_eq!(mp3.format, "mp3");
        assert_eq!(mp3.video_codec, None);

        let flac = Profile::AudioFlac.apply(&source, &options).unwrap();
        assert_eq!(flac.format, "flac");
        assert_eq!(flac.audio_bitrate, None);
    }

    #[test]
    fn test_resolution_calculation() {
        let source = create_test_properties();
        let max = (1280, 720);

        let resolution = calculate_resolution(&source, max).unwrap();
        assert_eq!(resolution, (1280, 720));
    }

    #[test]
    fn test_profile_system() {
        let system = ProfileSystem::new();
        let profiles = system.list_profiles();

        assert!(!profiles.is_empty());
        assert!(profiles.contains(&"Web Optimized"));
    }

    #[test]
    fn test_custom_profile() {
        let mut system = ProfileSystem::new();
        let custom = CustomProfile {
            name: "Custom".to_string(),
            description: "Custom profile".to_string(),
            settings: ConversionSettings {
                format: "mp4".to_string(),
                video_codec: Some("h264".to_string()),
                audio_codec: Some("aac".to_string()),
                video_bitrate: Some(3_000_000),
                audio_bitrate: Some(192_000),
                resolution: None,
                frame_rate: None,
                parameters: vec![],
            },
        };

        system.add_custom_profile(custom);
        assert!(system.get_custom_profile("Custom").is_some());
    }
}
