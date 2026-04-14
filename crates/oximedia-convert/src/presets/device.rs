// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Device-specific conversion presets.

use super::{AudioPresetSettings, EncodingSpeed, Preset, VideoPresetSettings};
use crate::formats::{AudioCodec, ChannelLayout, ContainerFormat, VideoCodec};
use crate::Result;

/// Android 1080p preset.
pub fn android_1080p() -> Result<Preset> {
    Ok(Preset {
        name: "Android 1080p".to_string(),
        description: "Optimized for Android devices (1080p)".to_string(),
        container: ContainerFormat::Mp4,
        video: Some(VideoPresetSettings {
            codec: VideoCodec::Vp9,
            width: Some(1920),
            height: Some(1080),
            frame_rate: Some(30.0),
            bitrate: Some(6_000_000),
            quality: None,
            two_pass: false,
            speed: EncodingSpeed::Medium,
        }),
        audio: Some(AudioPresetSettings {
            codec: AudioCodec::Opus,
            sample_rate: 48000,
            channels: ChannelLayout::Stereo,
            bitrate: Some(192_000),
        }),
    })
}

/// Android 720p preset.
pub fn android_720p() -> Result<Preset> {
    Ok(Preset {
        name: "Android 720p".to_string(),
        description: "Optimized for Android devices (720p)".to_string(),
        container: ContainerFormat::Mp4,
        video: Some(VideoPresetSettings {
            codec: VideoCodec::Vp9,
            width: Some(1280),
            height: Some(720),
            frame_rate: Some(30.0),
            bitrate: Some(4_000_000),
            quality: None,
            two_pass: false,
            speed: EncodingSpeed::Fast,
        }),
        audio: Some(AudioPresetSettings {
            codec: AudioCodec::Opus,
            sample_rate: 48000,
            channels: ChannelLayout::Stereo,
            bitrate: Some(128_000),
        }),
    })
}

/// iPhone 1080p preset.
pub fn iphone_1080p() -> Result<Preset> {
    Ok(Preset {
        name: "iPhone 1080p".to_string(),
        description: "Optimized for iPhone (1080p)".to_string(),
        container: ContainerFormat::Mp4,
        video: Some(VideoPresetSettings {
            codec: VideoCodec::Vp9,
            width: Some(1920),
            height: Some(1080),
            frame_rate: Some(30.0),
            bitrate: Some(6_000_000),
            quality: None,
            two_pass: false,
            speed: EncodingSpeed::Medium,
        }),
        audio: Some(AudioPresetSettings {
            codec: AudioCodec::Opus,
            sample_rate: 48000,
            channels: ChannelLayout::Stereo,
            bitrate: Some(192_000),
        }),
    })
}

/// iPad 1080p preset.
pub fn ipad_1080p() -> Result<Preset> {
    Ok(Preset {
        name: "iPad 1080p".to_string(),
        description: "Optimized for iPad (1080p)".to_string(),
        container: ContainerFormat::Mp4,
        video: Some(VideoPresetSettings {
            codec: VideoCodec::Vp9,
            width: Some(1920),
            height: Some(1080),
            frame_rate: Some(30.0),
            bitrate: Some(8_000_000),
            quality: None,
            two_pass: false,
            speed: EncodingSpeed::Medium,
        }),
        audio: Some(AudioPresetSettings {
            codec: AudioCodec::Opus,
            sample_rate: 48000,
            channels: ChannelLayout::Stereo,
            bitrate: Some(256_000),
        }),
    })
}

/// `PlayStation` 5 preset.
pub fn ps5() -> Result<Preset> {
    Ok(Preset {
        name: "PlayStation 5".to_string(),
        description: "Optimized for PlayStation 5".to_string(),
        container: ContainerFormat::Mp4,
        video: Some(VideoPresetSettings {
            codec: VideoCodec::Vp9,
            width: Some(3840),
            height: Some(2160),
            frame_rate: Some(60.0),
            bitrate: Some(40_000_000),
            quality: None,
            two_pass: true,
            speed: EncodingSpeed::Medium,
        }),
        audio: Some(AudioPresetSettings {
            codec: AudioCodec::Opus,
            sample_rate: 48000,
            channels: ChannelLayout::Surround7_1,
            bitrate: Some(512_000),
        }),
    })
}

/// Xbox Series X/S preset.
pub fn xbox_series() -> Result<Preset> {
    Ok(Preset {
        name: "Xbox Series X/S".to_string(),
        description: "Optimized for Xbox Series X/S".to_string(),
        container: ContainerFormat::Mp4,
        video: Some(VideoPresetSettings {
            codec: VideoCodec::Vp9,
            width: Some(3840),
            height: Some(2160),
            frame_rate: Some(60.0),
            bitrate: Some(40_000_000),
            quality: None,
            two_pass: true,
            speed: EncodingSpeed::Medium,
        }),
        audio: Some(AudioPresetSettings {
            codec: AudioCodec::Opus,
            sample_rate: 48000,
            channels: ChannelLayout::Surround7_1,
            bitrate: Some(512_000),
        }),
    })
}

/// Nintendo Switch preset.
pub fn nintendo_switch() -> Result<Preset> {
    Ok(Preset {
        name: "Nintendo Switch".to_string(),
        description: "Optimized for Nintendo Switch".to_string(),
        container: ContainerFormat::Mp4,
        video: Some(VideoPresetSettings {
            codec: VideoCodec::Vp9,
            width: Some(1920),
            height: Some(1080),
            frame_rate: Some(30.0),
            bitrate: Some(8_000_000),
            quality: None,
            two_pass: false,
            speed: EncodingSpeed::Fast,
        }),
        audio: Some(AudioPresetSettings {
            codec: AudioCodec::Opus,
            sample_rate: 48000,
            channels: ChannelLayout::Stereo,
            bitrate: Some(192_000),
        }),
    })
}

/// Smart TV 4K preset.
pub fn smart_tv_4k() -> Result<Preset> {
    Ok(Preset {
        name: "Smart TV 4K".to_string(),
        description: "Optimized for 4K Smart TVs".to_string(),
        container: ContainerFormat::Mp4,
        video: Some(VideoPresetSettings {
            codec: VideoCodec::Vp9,
            width: Some(3840),
            height: Some(2160),
            frame_rate: Some(30.0),
            bitrate: Some(35_000_000),
            quality: None,
            two_pass: true,
            speed: EncodingSpeed::Slow,
        }),
        audio: Some(AudioPresetSettings {
            codec: AudioCodec::Opus,
            sample_rate: 48000,
            channels: ChannelLayout::Surround5_1,
            bitrate: Some(384_000),
        }),
    })
}

/// Smart TV 1080p preset.
pub fn smart_tv_1080p() -> Result<Preset> {
    Ok(Preset {
        name: "Smart TV 1080p".to_string(),
        description: "Optimized for 1080p Smart TVs".to_string(),
        container: ContainerFormat::Mp4,
        video: Some(VideoPresetSettings {
            codec: VideoCodec::Vp9,
            width: Some(1920),
            height: Some(1080),
            frame_rate: Some(30.0),
            bitrate: Some(10_000_000),
            quality: None,
            two_pass: false,
            speed: EncodingSpeed::Medium,
        }),
        audio: Some(AudioPresetSettings {
            codec: AudioCodec::Opus,
            sample_rate: 48000,
            channels: ChannelLayout::Surround5_1,
            bitrate: Some(384_000),
        }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_android_1080p() {
        let preset = android_1080p().unwrap();
        assert_eq!(preset.container, ContainerFormat::Mp4);
        let video = preset.video.unwrap();
        assert_eq!(video.width, Some(1920));
        assert_eq!(video.height, Some(1080));
    }

    #[test]
    fn test_gaming_consoles() {
        let ps5 = ps5().unwrap();
        let xbox = xbox_series().unwrap();
        let switch = nintendo_switch().unwrap();

        // PS5 and Xbox should be 4K
        assert_eq!(ps5.video.as_ref().unwrap().width, Some(3840));
        assert_eq!(xbox.video.as_ref().unwrap().width, Some(3840));

        // Switch should be 1080p
        assert_eq!(switch.video.as_ref().unwrap().width, Some(1920));
    }

    #[test]
    fn test_all_device_presets() {
        assert!(android_1080p().is_ok());
        assert!(android_720p().is_ok());
        assert!(iphone_1080p().is_ok());
        assert!(ipad_1080p().is_ok());
        assert!(ps5().is_ok());
        assert!(xbox_series().is_ok());
        assert!(nintendo_switch().is_ok());
        assert!(smart_tv_4k().is_ok());
        assert!(smart_tv_1080p().is_ok());
    }
}
