// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Conversion presets for common use cases.
//!
//! This module provides pre-configured conversion settings for various platforms,
//! devices, and workflows.

pub mod archive;
pub mod broadcast;
pub mod device;
pub mod social_media;
pub mod web;

use crate::formats::{AudioCodec, ChannelLayout, ContainerFormat, VideoCodec};
use crate::{ConversionError, Result};
use serde::{Deserialize, Serialize};

/// Conversion preset.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Preset {
    /// Preset name
    pub name: String,
    /// Preset description
    pub description: String,
    /// Container format
    pub container: ContainerFormat,
    /// Video settings
    pub video: Option<VideoPresetSettings>,
    /// Audio settings
    pub audio: Option<AudioPresetSettings>,
}

/// Video preset settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VideoPresetSettings {
    /// Video codec
    pub codec: VideoCodec,
    /// Target width
    pub width: Option<u32>,
    /// Target height
    pub height: Option<u32>,
    /// Frame rate
    pub frame_rate: Option<f64>,
    /// Bitrate in bits per second
    pub bitrate: Option<u64>,
    /// Quality (CRF value)
    pub quality: Option<u32>,
    /// Two-pass encoding
    pub two_pass: bool,
    /// Encoding speed preset
    pub speed: EncodingSpeed,
}

/// Audio preset settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioPresetSettings {
    /// Audio codec
    pub codec: AudioCodec,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Channel layout
    pub channels: ChannelLayout,
    /// Bitrate in bits per second
    pub bitrate: Option<u64>,
}

/// Encoding speed preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EncodingSpeed {
    /// Fastest encoding, lower quality
    Fast,
    /// Balanced speed and quality
    Medium,
    /// Slower encoding, better quality
    Slow,
    /// Slowest encoding, best quality
    VerySlow,
}

impl Preset {
    /// Create a preset from a preset identifier.
    pub fn from_name(name: &str) -> Result<Self> {
        match name.to_lowercase().as_str() {
            // Web presets
            "youtube-1080p" | "youtube-hd" => web::youtube_1080p(),
            "youtube-720p" => web::youtube_720p(),
            "youtube-480p" | "youtube-sd" => web::youtube_480p(),
            "youtube-4k" | "youtube-2160p" => web::youtube_4k(),
            "vimeo-hq" => web::vimeo_hq(),
            "vimeo-hd" => web::vimeo_hd(),
            "vimeo-sd" => web::vimeo_sd(),
            "facebook-video" => web::facebook_video(),
            "facebook-story" => web::facebook_story(),
            "twitter-video" => web::twitter_video(),
            "instagram-feed" => web::instagram_feed(),
            "instagram-story" => web::instagram_story(),
            "instagram-reel" => web::instagram_reel(),
            "tiktok" => web::tiktok(),

            // Device presets
            "android-1080p" => device::android_1080p(),
            "android-720p" => device::android_720p(),
            "iphone" | "iphone-1080p" => device::iphone_1080p(),
            "ipad" | "ipad-1080p" => device::ipad_1080p(),
            "ps5" | "playstation5" => device::ps5(),
            "xbox" | "xbox-series" => device::xbox_series(),
            "switch" | "nintendo-switch" => device::nintendo_switch(),
            "smarttv-4k" => device::smart_tv_4k(),
            "smarttv-1080p" => device::smart_tv_1080p(),

            // Broadcast presets
            "broadcast-1080p-25" => broadcast::hd_1080p_25fps(),
            "broadcast-1080p-30" => broadcast::hd_1080p_30fps(),
            "broadcast-720p" => broadcast::hd_720p(),
            "broadcast-sd-pal" => broadcast::sd_pal(),
            "broadcast-sd-ntsc" => broadcast::sd_ntsc(),
            "broadcast-4k" => broadcast::uhd_4k(),

            // Archive presets
            "archive-lossless" => archive::lossless(),
            "archive-near-lossless" => archive::near_lossless(),
            "archive-intermediate" => archive::intermediate(),
            "archive-long-term" => archive::long_term(),

            _ => Err(ConversionError::InvalidProfile(format!(
                "Unknown preset: {name}"
            ))),
        }
    }

    /// Get all available preset names.
    #[must_use]
    pub fn available_presets() -> Vec<&'static str> {
        vec![
            // Web
            "youtube-1080p",
            "youtube-720p",
            "youtube-480p",
            "youtube-4k",
            "vimeo-hq",
            "vimeo-hd",
            "vimeo-sd",
            "facebook-video",
            "facebook-story",
            "twitter-video",
            "instagram-feed",
            "instagram-story",
            "instagram-reel",
            "tiktok",
            // Device
            "android-1080p",
            "android-720p",
            "iphone-1080p",
            "ipad-1080p",
            "ps5",
            "xbox-series",
            "nintendo-switch",
            "smarttv-4k",
            "smarttv-1080p",
            // Broadcast
            "broadcast-1080p-25",
            "broadcast-1080p-30",
            "broadcast-720p",
            "broadcast-sd-pal",
            "broadcast-sd-ntsc",
            "broadcast-4k",
            // Archive
            "archive-lossless",
            "archive-near-lossless",
            "archive-intermediate",
            "archive-long-term",
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preset_from_name() {
        assert!(Preset::from_name("youtube-1080p").is_ok());
        assert!(Preset::from_name("vimeo-hq").is_ok());
        assert!(Preset::from_name("android-1080p").is_ok());
        assert!(Preset::from_name("broadcast-1080p-25").is_ok());
        assert!(Preset::from_name("archive-lossless").is_ok());
        assert!(Preset::from_name("unknown-preset").is_err());
    }

    #[test]
    fn test_available_presets() {
        let presets = Preset::available_presets();
        assert!(presets.len() > 20);
        assert!(presets.contains(&"youtube-1080p"));
        assert!(presets.contains(&"iphone-1080p"));
    }

    #[test]
    fn test_youtube_preset_properties() {
        let preset = Preset::from_name("youtube-1080p").unwrap();
        assert!(preset.video.is_some());
        let video = preset.video.unwrap();
        assert_eq!(video.height, Some(1080));
    }
}
