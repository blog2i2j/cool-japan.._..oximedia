// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Built-in conversion presets.

use crate::profile::system::CustomProfile;
use crate::ConversionSettings;

/// Built-in profile presets.
#[derive(Debug, Clone)]
pub struct ProfilePresets;

impl ProfilePresets {
    /// Get all built-in presets.
    #[must_use]
    pub fn all() -> Vec<CustomProfile> {
        vec![
            Self::web_video_hd(),
            Self::web_video_sd(),
            Self::podcast_audio(),
            Self::music_high_quality(),
            Self::quick_preview(),
            Self::gif_animation(),
        ]
    }

    /// Web video in HD quality (1080p).
    #[must_use]
    pub fn web_video_hd() -> CustomProfile {
        CustomProfile {
            name: "Web Video HD".to_string(),
            description: "High quality web video (1080p, H.264)".to_string(),
            settings: ConversionSettings {
                format: "mp4".to_string(),
                video_codec: Some("h264".to_string()),
                audio_codec: Some("aac".to_string()),
                video_bitrate: Some(5_000_000),
                audio_bitrate: Some(192_000),
                resolution: Some((1920, 1080)),
                frame_rate: Some(30.0),
                parameters: vec![
                    ("preset".to_string(), "medium".to_string()),
                    ("crf".to_string(), "23".to_string()),
                ],
            },
        }
    }

    /// Web video in SD quality (480p).
    #[must_use]
    pub fn web_video_sd() -> CustomProfile {
        CustomProfile {
            name: "Web Video SD".to_string(),
            description: "Standard definition web video (480p, H.264)".to_string(),
            settings: ConversionSettings {
                format: "mp4".to_string(),
                video_codec: Some("h264".to_string()),
                audio_codec: Some("aac".to_string()),
                video_bitrate: Some(1_500_000),
                audio_bitrate: Some(128_000),
                resolution: Some((854, 480)),
                frame_rate: Some(30.0),
                parameters: vec![
                    ("preset".to_string(), "medium".to_string()),
                    ("crf".to_string(), "25".to_string()),
                ],
            },
        }
    }

    /// Podcast audio (mono, compressed).
    #[must_use]
    pub fn podcast_audio() -> CustomProfile {
        CustomProfile {
            name: "Podcast Audio".to_string(),
            description: "Optimized for podcast (mono, 64 kbps MP3)".to_string(),
            settings: ConversionSettings {
                format: "mp3".to_string(),
                video_codec: None,
                audio_codec: Some("mp3".to_string()),
                video_bitrate: None,
                audio_bitrate: Some(64_000),
                resolution: None,
                frame_rate: None,
                parameters: vec![
                    ("ac".to_string(), "1".to_string()), // mono
                ],
            },
        }
    }

    /// High quality music audio.
    #[must_use]
    pub fn music_high_quality() -> CustomProfile {
        CustomProfile {
            name: "Music High Quality".to_string(),
            description: "High quality stereo music (320 kbps MP3)".to_string(),
            settings: ConversionSettings {
                format: "mp3".to_string(),
                video_codec: None,
                audio_codec: Some("mp3".to_string()),
                video_bitrate: None,
                audio_bitrate: Some(320_000),
                resolution: None,
                frame_rate: None,
                parameters: vec![],
            },
        }
    }

    /// Quick preview (low quality, fast encoding).
    #[must_use]
    pub fn quick_preview() -> CustomProfile {
        CustomProfile {
            name: "Quick Preview".to_string(),
            description: "Fast encoding for quick preview (360p, low quality)".to_string(),
            settings: ConversionSettings {
                format: "mp4".to_string(),
                video_codec: Some("h264".to_string()),
                audio_codec: Some("aac".to_string()),
                video_bitrate: Some(800_000),
                audio_bitrate: Some(96_000),
                resolution: Some((640, 360)),
                frame_rate: Some(24.0),
                parameters: vec![
                    ("preset".to_string(), "ultrafast".to_string()),
                    ("crf".to_string(), "30".to_string()),
                ],
            },
        }
    }

    /// GIF animation.
    #[must_use]
    pub fn gif_animation() -> CustomProfile {
        CustomProfile {
            name: "GIF Animation".to_string(),
            description: "Convert video to GIF animation".to_string(),
            settings: ConversionSettings {
                format: "gif".to_string(),
                video_codec: Some("gif".to_string()),
                audio_codec: None,
                video_bitrate: None,
                audio_bitrate: None,
                resolution: Some((480, 270)),
                frame_rate: Some(15.0),
                parameters: vec![],
            },
        }
    }

    /// Get preset by name.
    #[must_use]
    pub fn get_by_name(name: &str) -> Option<CustomProfile> {
        Self::all().into_iter().find(|p| p.name == name)
    }

    /// List all preset names.
    #[must_use]
    pub fn list_names() -> Vec<String> {
        Self::all().into_iter().map(|p| p.name).collect()
    }

    /// Get presets by category.
    #[must_use]
    pub fn by_category(category: PresetCategory) -> Vec<CustomProfile> {
        match category {
            PresetCategory::Video => vec![
                Self::web_video_hd(),
                Self::web_video_sd(),
                Self::quick_preview(),
                Self::gif_animation(),
            ],
            PresetCategory::Audio => vec![Self::podcast_audio(), Self::music_high_quality()],
        }
    }
}

/// Categories for presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PresetCategory {
    /// Video presets
    Video,
    /// Audio presets
    Audio,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_presets() {
        let presets = ProfilePresets::all();
        assert!(!presets.is_empty());
        assert!(presets.len() >= 6);
    }

    #[test]
    fn test_web_video_hd() {
        let preset = ProfilePresets::web_video_hd();
        assert_eq!(preset.name, "Web Video HD");
        assert_eq!(preset.settings.format, "mp4");
        assert_eq!(preset.settings.resolution, Some((1920, 1080)));
    }

    #[test]
    fn test_podcast_audio() {
        let preset = ProfilePresets::podcast_audio();
        assert_eq!(preset.settings.format, "mp3");
        assert_eq!(preset.settings.video_codec, None);
        assert!(preset.settings.audio_bitrate.is_some());
    }

    #[test]
    fn test_get_by_name() {
        let preset = ProfilePresets::get_by_name("Web Video HD");
        assert!(preset.is_some());
        assert_eq!(preset.unwrap().name, "Web Video HD");

        let missing = ProfilePresets::get_by_name("Non-existent");
        assert!(missing.is_none());
    }

    #[test]
    fn test_list_names() {
        let names = ProfilePresets::list_names();
        assert!(!names.is_empty());
        assert!(names.contains(&"Web Video HD".to_string()));
        assert!(names.contains(&"Podcast Audio".to_string()));
    }

    #[test]
    fn test_by_category() {
        let video_presets = ProfilePresets::by_category(PresetCategory::Video);
        assert!(!video_presets.is_empty());
        assert!(video_presets
            .iter()
            .all(|p| p.settings.video_codec.is_some()));

        let audio_presets = ProfilePresets::by_category(PresetCategory::Audio);
        assert!(!audio_presets.is_empty());
        assert!(audio_presets
            .iter()
            .all(|p| p.settings.video_codec.is_none()));
    }

    #[test]
    fn test_quick_preview() {
        let preset = ProfilePresets::quick_preview();
        assert_eq!(preset.settings.resolution, Some((640, 360)));
        assert!(preset
            .settings
            .parameters
            .iter()
            .any(|(k, v)| k == "preset" && v == "ultrafast"));
    }

    #[test]
    fn test_gif_animation() {
        let preset = ProfilePresets::gif_animation();
        assert_eq!(preset.settings.format, "gif");
        assert_eq!(preset.settings.audio_codec, None);
    }
}
