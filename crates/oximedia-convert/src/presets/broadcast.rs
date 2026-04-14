// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Broadcast-standard conversion presets.

use super::{AudioPresetSettings, EncodingSpeed, Preset, VideoPresetSettings};
use crate::formats::{AudioCodec, ChannelLayout, ContainerFormat, VideoCodec};
use crate::Result;

/// HD 1080p 25fps preset (PAL regions).
pub fn hd_1080p_25fps() -> Result<Preset> {
    Ok(Preset {
        name: "Broadcast HD 1080p 25fps".to_string(),
        description: "Broadcast standard 1080p 25fps (PAL)".to_string(),
        container: ContainerFormat::MpegTs,
        video: Some(VideoPresetSettings {
            codec: VideoCodec::Av1,
            width: Some(1920),
            height: Some(1080),
            frame_rate: Some(25.0),
            bitrate: Some(50_000_000),
            quality: None,
            two_pass: true,
            speed: EncodingSpeed::VerySlow,
        }),
        audio: Some(AudioPresetSettings {
            codec: AudioCodec::Opus,
            sample_rate: 48000,
            channels: ChannelLayout::Stereo,
            bitrate: Some(256_000),
        }),
    })
}

/// HD 1080p 30fps preset (NTSC regions).
pub fn hd_1080p_30fps() -> Result<Preset> {
    Ok(Preset {
        name: "Broadcast HD 1080p 30fps".to_string(),
        description: "Broadcast standard 1080p 29.97fps (NTSC)".to_string(),
        container: ContainerFormat::MpegTs,
        video: Some(VideoPresetSettings {
            codec: VideoCodec::Av1,
            width: Some(1920),
            height: Some(1080),
            frame_rate: Some(29.97),
            bitrate: Some(50_000_000),
            quality: None,
            two_pass: true,
            speed: EncodingSpeed::VerySlow,
        }),
        audio: Some(AudioPresetSettings {
            codec: AudioCodec::Opus,
            sample_rate: 48000,
            channels: ChannelLayout::Stereo,
            bitrate: Some(256_000),
        }),
    })
}

/// HD 720p preset.
pub fn hd_720p() -> Result<Preset> {
    Ok(Preset {
        name: "Broadcast HD 720p".to_string(),
        description: "Broadcast standard 720p 59.94fps".to_string(),
        container: ContainerFormat::MpegTs,
        video: Some(VideoPresetSettings {
            codec: VideoCodec::Av1,
            width: Some(1280),
            height: Some(720),
            frame_rate: Some(59.94),
            bitrate: Some(40_000_000),
            quality: None,
            two_pass: true,
            speed: EncodingSpeed::VerySlow,
        }),
        audio: Some(AudioPresetSettings {
            codec: AudioCodec::Opus,
            sample_rate: 48000,
            channels: ChannelLayout::Stereo,
            bitrate: Some(256_000),
        }),
    })
}

/// SD PAL preset (576i).
pub fn sd_pal() -> Result<Preset> {
    Ok(Preset {
        name: "Broadcast SD PAL".to_string(),
        description: "Broadcast standard SD PAL (720x576, 25fps)".to_string(),
        container: ContainerFormat::MpegTs,
        video: Some(VideoPresetSettings {
            codec: VideoCodec::Vp9,
            width: Some(720),
            height: Some(576),
            frame_rate: Some(25.0),
            bitrate: Some(15_000_000),
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

/// SD NTSC preset (480i).
pub fn sd_ntsc() -> Result<Preset> {
    Ok(Preset {
        name: "Broadcast SD NTSC".to_string(),
        description: "Broadcast standard SD NTSC (720x480, 29.97fps)".to_string(),
        container: ContainerFormat::MpegTs,
        video: Some(VideoPresetSettings {
            codec: VideoCodec::Vp9,
            width: Some(720),
            height: Some(480),
            frame_rate: Some(29.97),
            bitrate: Some(15_000_000),
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

/// UHD 4K preset.
pub fn uhd_4k() -> Result<Preset> {
    Ok(Preset {
        name: "Broadcast UHD 4K".to_string(),
        description: "Broadcast standard UHD 4K (3840x2160)".to_string(),
        container: ContainerFormat::MpegTs,
        video: Some(VideoPresetSettings {
            codec: VideoCodec::Av1,
            width: Some(3840),
            height: Some(2160),
            frame_rate: Some(25.0),
            bitrate: Some(100_000_000),
            quality: None,
            two_pass: true,
            speed: EncodingSpeed::VerySlow,
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
    fn test_hd_1080p_25fps() {
        let preset = hd_1080p_25fps().unwrap();
        assert_eq!(preset.container, ContainerFormat::MpegTs);
        let video = preset.video.unwrap();
        assert_eq!(video.width, Some(1920));
        assert_eq!(video.height, Some(1080));
        assert_eq!(video.frame_rate, Some(25.0));
    }

    #[test]
    fn test_sd_pal_vs_ntsc() {
        let pal = sd_pal().unwrap();
        let ntsc = sd_ntsc().unwrap();

        let pal_video = pal.video.unwrap();
        let ntsc_video = ntsc.video.unwrap();

        assert_eq!(pal_video.height, Some(576));
        assert_eq!(ntsc_video.height, Some(480));
        assert_eq!(pal_video.frame_rate, Some(25.0));
        assert_eq!(ntsc_video.frame_rate, Some(29.97));
    }

    #[test]
    fn test_uhd_4k() {
        let preset = uhd_4k().unwrap();
        let video = preset.video.unwrap();
        assert_eq!(video.width, Some(3840));
        assert_eq!(video.height, Some(2160));
        assert_eq!(video.codec, VideoCodec::Av1);
    }

    #[test]
    fn test_all_broadcast_presets() {
        assert!(hd_1080p_25fps().is_ok());
        assert!(hd_1080p_30fps().is_ok());
        assert!(hd_720p().is_ok());
        assert!(sd_pal().is_ok());
        assert!(sd_ntsc().is_ok());
        assert!(uhd_4k().is_ok());
    }
}
