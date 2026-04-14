// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Archive and preservation conversion presets.

use super::{AudioPresetSettings, EncodingSpeed, Preset, VideoPresetSettings};
use crate::formats::{AudioCodec, ChannelLayout, ContainerFormat, VideoCodec};
use crate::Result;

/// Lossless archive preset.
pub fn lossless() -> Result<Preset> {
    Ok(Preset {
        name: "Archive Lossless".to_string(),
        description: "Lossless archival quality (no compression artifacts)".to_string(),
        container: ContainerFormat::Matroska,
        video: Some(VideoPresetSettings {
            codec: VideoCodec::Av1,
            width: None, // Preserve original
            height: None,
            frame_rate: None,
            bitrate: None,
            quality: Some(0), // Lossless
            two_pass: false,
            speed: EncodingSpeed::VerySlow,
        }),
        audio: Some(AudioPresetSettings {
            codec: AudioCodec::Flac,
            sample_rate: 48000,
            channels: ChannelLayout::Stereo,
            bitrate: None, // Lossless
        }),
    })
}

/// Near-lossless archive preset.
pub fn near_lossless() -> Result<Preset> {
    Ok(Preset {
        name: "Archive Near-Lossless".to_string(),
        description: "Near-lossless archival quality (visually lossless)".to_string(),
        container: ContainerFormat::Matroska,
        video: Some(VideoPresetSettings {
            codec: VideoCodec::Av1,
            width: None,
            height: None,
            frame_rate: None,
            bitrate: None,
            quality: Some(10), // Very high quality
            two_pass: false,
            speed: EncodingSpeed::VerySlow,
        }),
        audio: Some(AudioPresetSettings {
            codec: AudioCodec::Flac,
            sample_rate: 48000,
            channels: ChannelLayout::Stereo,
            bitrate: None,
        }),
    })
}

/// Intermediate codec preset (for editing).
pub fn intermediate() -> Result<Preset> {
    Ok(Preset {
        name: "Archive Intermediate".to_string(),
        description: "Intermediate codec for editing workflows".to_string(),
        container: ContainerFormat::Matroska,
        video: Some(VideoPresetSettings {
            codec: VideoCodec::Av1,
            width: None,
            height: None,
            frame_rate: None,
            bitrate: None,
            quality: Some(20), // High quality
            two_pass: false,
            speed: EncodingSpeed::Fast, // Faster for editing
        }),
        audio: Some(AudioPresetSettings {
            codec: AudioCodec::Pcm,
            sample_rate: 48000,
            channels: ChannelLayout::Stereo,
            bitrate: None,
        }),
    })
}

/// Long-term preservation preset.
pub fn long_term() -> Result<Preset> {
    Ok(Preset {
        name: "Archive Long-Term".to_string(),
        description: "Long-term preservation with maximum compatibility".to_string(),
        container: ContainerFormat::Matroska,
        video: Some(VideoPresetSettings {
            codec: VideoCodec::Av1,
            width: None,
            height: None,
            frame_rate: None,
            bitrate: None,
            quality: Some(5), // Very high quality
            two_pass: true,
            speed: EncodingSpeed::VerySlow,
        }),
        audio: Some(AudioPresetSettings {
            codec: AudioCodec::Flac,
            sample_rate: 48000,
            channels: ChannelLayout::Stereo,
            bitrate: None,
        }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lossless() {
        let preset = lossless().unwrap();
        assert_eq!(preset.container, ContainerFormat::Matroska);
        let video = preset.video.unwrap();
        assert_eq!(video.codec, VideoCodec::Av1);
        assert_eq!(video.quality, Some(0));
        let audio = preset.audio.unwrap();
        assert_eq!(audio.codec, AudioCodec::Flac);
    }

    #[test]
    fn test_intermediate() {
        let preset = intermediate().unwrap();
        let video = preset.video.unwrap();
        assert_eq!(video.speed, EncodingSpeed::Fast);
        let audio = preset.audio.unwrap();
        assert_eq!(audio.codec, AudioCodec::Pcm);
    }

    #[test]
    fn test_all_archive_presets() {
        assert!(lossless().is_ok());
        assert!(near_lossless().is_ok());
        assert!(intermediate().is_ok());
        assert!(long_term().is_ok());
    }
}
