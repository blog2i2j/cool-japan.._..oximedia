// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Web platform conversion presets.

use super::{AudioPresetSettings, EncodingSpeed, Preset, VideoPresetSettings};
use crate::formats::{AudioCodec, ChannelLayout, ContainerFormat, VideoCodec};
use crate::Result;

/// `YouTube` 1080p preset.
pub fn youtube_1080p() -> Result<Preset> {
    Ok(Preset {
        name: "YouTube 1080p".to_string(),
        description: "Optimized for YouTube 1080p uploads".to_string(),
        container: ContainerFormat::Mp4,
        video: Some(VideoPresetSettings {
            codec: VideoCodec::Vp9,
            width: Some(1920),
            height: Some(1080),
            frame_rate: Some(30.0),
            bitrate: Some(8_000_000),
            quality: None,
            two_pass: true,
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

/// `YouTube` 720p preset.
pub fn youtube_720p() -> Result<Preset> {
    Ok(Preset {
        name: "YouTube 720p".to_string(),
        description: "Optimized for YouTube 720p uploads".to_string(),
        container: ContainerFormat::Mp4,
        video: Some(VideoPresetSettings {
            codec: VideoCodec::Vp9,
            width: Some(1280),
            height: Some(720),
            frame_rate: Some(30.0),
            bitrate: Some(5_000_000),
            quality: None,
            two_pass: true,
            speed: EncodingSpeed::Medium,
        }),
        audio: Some(AudioPresetSettings {
            codec: AudioCodec::Opus,
            sample_rate: 48000,
            channels: ChannelLayout::Stereo,
            bitrate: Some(128_000),
        }),
    })
}

/// `YouTube` 480p preset.
pub fn youtube_480p() -> Result<Preset> {
    Ok(Preset {
        name: "YouTube 480p".to_string(),
        description: "Optimized for YouTube 480p uploads".to_string(),
        container: ContainerFormat::Mp4,
        video: Some(VideoPresetSettings {
            codec: VideoCodec::Vp9,
            width: Some(854),
            height: Some(480),
            frame_rate: Some(30.0),
            bitrate: Some(2_500_000),
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

/// `YouTube` 4K preset.
pub fn youtube_4k() -> Result<Preset> {
    Ok(Preset {
        name: "YouTube 4K".to_string(),
        description: "Optimized for YouTube 4K (2160p) uploads".to_string(),
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
            channels: ChannelLayout::Stereo,
            bitrate: Some(192_000),
        }),
    })
}

/// Vimeo HQ preset.
pub fn vimeo_hq() -> Result<Preset> {
    Ok(Preset {
        name: "Vimeo HQ".to_string(),
        description: "Vimeo high quality preset".to_string(),
        container: ContainerFormat::Mp4,
        video: Some(VideoPresetSettings {
            codec: VideoCodec::Vp9,
            width: Some(1920),
            height: Some(1080),
            frame_rate: Some(30.0),
            bitrate: Some(10_000_000),
            quality: None,
            two_pass: true,
            speed: EncodingSpeed::Slow,
        }),
        audio: Some(AudioPresetSettings {
            codec: AudioCodec::Opus,
            sample_rate: 48000,
            channels: ChannelLayout::Stereo,
            bitrate: Some(256_000),
        }),
    })
}

/// Vimeo HD preset.
pub fn vimeo_hd() -> Result<Preset> {
    Ok(Preset {
        name: "Vimeo HD".to_string(),
        description: "Vimeo HD preset".to_string(),
        container: ContainerFormat::Mp4,
        video: Some(VideoPresetSettings {
            codec: VideoCodec::Vp9,
            width: Some(1280),
            height: Some(720),
            frame_rate: Some(30.0),
            bitrate: Some(5_000_000),
            quality: None,
            two_pass: true,
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

/// Vimeo SD preset.
pub fn vimeo_sd() -> Result<Preset> {
    Ok(Preset {
        name: "Vimeo SD".to_string(),
        description: "Vimeo standard definition preset".to_string(),
        container: ContainerFormat::Mp4,
        video: Some(VideoPresetSettings {
            codec: VideoCodec::Vp9,
            width: Some(640),
            height: Some(480),
            frame_rate: Some(30.0),
            bitrate: Some(2_000_000),
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

/// Facebook video post preset.
pub fn facebook_video() -> Result<Preset> {
    Ok(Preset {
        name: "Facebook Video".to_string(),
        description: "Optimized for Facebook video posts".to_string(),
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

/// Facebook story preset.
pub fn facebook_story() -> Result<Preset> {
    Ok(Preset {
        name: "Facebook Story".to_string(),
        description: "Optimized for Facebook stories (9:16 aspect ratio)".to_string(),
        container: ContainerFormat::Mp4,
        video: Some(VideoPresetSettings {
            codec: VideoCodec::Vp9,
            width: Some(1080),
            height: Some(1920),
            frame_rate: Some(30.0),
            bitrate: Some(3_000_000),
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

/// Twitter video preset.
pub fn twitter_video() -> Result<Preset> {
    Ok(Preset {
        name: "Twitter Video".to_string(),
        description: "Optimized for Twitter video tweets".to_string(),
        container: ContainerFormat::Mp4,
        video: Some(VideoPresetSettings {
            codec: VideoCodec::Vp9,
            width: Some(1280),
            height: Some(720),
            frame_rate: Some(30.0),
            bitrate: Some(5_000_000),
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

/// Instagram feed post preset.
pub fn instagram_feed() -> Result<Preset> {
    Ok(Preset {
        name: "Instagram Feed".to_string(),
        description: "Optimized for Instagram feed posts".to_string(),
        container: ContainerFormat::Mp4,
        video: Some(VideoPresetSettings {
            codec: VideoCodec::Vp9,
            width: Some(1080),
            height: Some(1080),
            frame_rate: Some(30.0),
            bitrate: Some(3_500_000),
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

/// Instagram story preset.
pub fn instagram_story() -> Result<Preset> {
    Ok(Preset {
        name: "Instagram Story".to_string(),
        description: "Optimized for Instagram stories (9:16 aspect ratio)".to_string(),
        container: ContainerFormat::Mp4,
        video: Some(VideoPresetSettings {
            codec: VideoCodec::Vp9,
            width: Some(1080),
            height: Some(1920),
            frame_rate: Some(30.0),
            bitrate: Some(3_000_000),
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

/// Instagram reel preset.
pub fn instagram_reel() -> Result<Preset> {
    Ok(Preset {
        name: "Instagram Reel".to_string(),
        description: "Optimized for Instagram reels (9:16 aspect ratio)".to_string(),
        container: ContainerFormat::Mp4,
        video: Some(VideoPresetSettings {
            codec: VideoCodec::Vp9,
            width: Some(1080),
            height: Some(1920),
            frame_rate: Some(30.0),
            bitrate: Some(3_500_000),
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

/// `TikTok` preset.
pub fn tiktok() -> Result<Preset> {
    Ok(Preset {
        name: "TikTok".to_string(),
        description: "Optimized for TikTok videos (9:16 aspect ratio)".to_string(),
        container: ContainerFormat::Mp4,
        video: Some(VideoPresetSettings {
            codec: VideoCodec::Vp9,
            width: Some(1080),
            height: Some(1920),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_youtube_1080p() {
        let preset = youtube_1080p().unwrap();
        assert_eq!(preset.container, ContainerFormat::Mp4);
        assert!(preset.video.is_some());
        let video = preset.video.unwrap();
        assert_eq!(video.width, Some(1920));
        assert_eq!(video.height, Some(1080));
        assert_eq!(video.codec, VideoCodec::Vp9);
    }

    #[test]
    fn test_youtube_4k() {
        let preset = youtube_4k().unwrap();
        let video = preset.video.unwrap();
        assert_eq!(video.width, Some(3840));
        assert_eq!(video.height, Some(2160));
    }

    #[test]
    fn test_instagram_story() {
        let preset = instagram_story().unwrap();
        let video = preset.video.unwrap();
        assert_eq!(video.width, Some(1080));
        assert_eq!(video.height, Some(1920)); // 9:16 aspect ratio
    }

    #[test]
    fn test_all_web_presets() {
        assert!(youtube_1080p().is_ok());
        assert!(youtube_720p().is_ok());
        assert!(youtube_480p().is_ok());
        assert!(youtube_4k().is_ok());
        assert!(vimeo_hq().is_ok());
        assert!(vimeo_hd().is_ok());
        assert!(vimeo_sd().is_ok());
        assert!(facebook_video().is_ok());
        assert!(facebook_story().is_ok());
        assert!(twitter_video().is_ok());
        assert!(instagram_feed().is_ok());
        assert!(instagram_story().is_ok());
        assert!(instagram_reel().is_ok());
        assert!(tiktok().is_ok());
    }
}
