// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Custom profile builder.

use crate::profile::system::CustomProfile;
use crate::{ConversionError, ConversionSettings, Result};

/// Builder for creating custom conversion profiles.
#[derive(Debug, Default)]
pub struct ProfileBuilder {
    name: Option<String>,
    description: Option<String>,
    format: Option<String>,
    video_codec: Option<String>,
    audio_codec: Option<String>,
    video_bitrate: Option<u64>,
    audio_bitrate: Option<u64>,
    resolution: Option<(u32, u32)>,
    frame_rate: Option<f64>,
    parameters: Vec<(String, String)>,
}

impl ProfileBuilder {
    /// Create a new profile builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the profile name.
    pub fn name<S: Into<String>>(mut self, name: S) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the profile description.
    pub fn description<S: Into<String>>(mut self, description: S) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Set the output format.
    pub fn format<S: Into<String>>(mut self, format: S) -> Self {
        self.format = Some(format.into());
        self
    }

    /// Set the video codec.
    pub fn video_codec<S: Into<String>>(mut self, codec: S) -> Self {
        self.video_codec = Some(codec.into());
        self
    }

    /// Set the audio codec.
    pub fn audio_codec<S: Into<String>>(mut self, codec: S) -> Self {
        self.audio_codec = Some(codec.into());
        self
    }

    /// Set the video bitrate.
    #[must_use]
    pub fn video_bitrate(mut self, bitrate: u64) -> Self {
        self.video_bitrate = Some(bitrate);
        self
    }

    /// Set the audio bitrate.
    #[must_use]
    pub fn audio_bitrate(mut self, bitrate: u64) -> Self {
        self.audio_bitrate = Some(bitrate);
        self
    }

    /// Set the resolution.
    #[must_use]
    pub fn resolution(mut self, width: u32, height: u32) -> Self {
        self.resolution = Some((width, height));
        self
    }

    /// Set the frame rate.
    #[must_use]
    pub fn frame_rate(mut self, fps: f64) -> Self {
        self.frame_rate = Some(fps);
        self
    }

    /// Add a custom parameter.
    pub fn parameter<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.parameters.push((key.into(), value.into()));
        self
    }

    /// Add multiple parameters.
    pub fn parameters<I, K, V>(mut self, params: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        for (k, v) in params {
            self.parameters.push((k.into(), v.into()));
        }
        self
    }

    /// Build the custom profile.
    pub fn build(self) -> Result<CustomProfile> {
        let name = self.name.ok_or_else(|| {
            ConversionError::InvalidProfile("Profile name is required".to_string())
        })?;

        let format = self.format.ok_or_else(|| {
            ConversionError::InvalidProfile("Output format is required".to_string())
        })?;

        Ok(CustomProfile {
            name,
            description: self
                .description
                .unwrap_or_else(|| "Custom profile".to_string()),
            settings: ConversionSettings {
                format,
                video_codec: self.video_codec,
                audio_codec: self.audio_codec,
                video_bitrate: self.video_bitrate,
                audio_bitrate: self.audio_bitrate,
                resolution: self.resolution,
                frame_rate: self.frame_rate,
                parameters: self.parameters,
            },
        })
    }

    /// Create a video-only profile builder.
    #[must_use]
    pub fn video_only() -> Self {
        Self::new()
    }

    /// Create an audio-only profile builder.
    #[must_use]
    pub fn audio_only() -> Self {
        Self::new()
    }

    /// Clone settings from an existing profile.
    #[must_use]
    pub fn from_profile(profile: &CustomProfile) -> Self {
        Self {
            name: Some(profile.name.clone()),
            description: Some(profile.description.clone()),
            format: Some(profile.settings.format.clone()),
            video_codec: profile.settings.video_codec.clone(),
            audio_codec: profile.settings.audio_codec.clone(),
            video_bitrate: profile.settings.video_bitrate,
            audio_bitrate: profile.settings.audio_bitrate,
            resolution: profile.settings.resolution,
            frame_rate: profile.settings.frame_rate,
            parameters: profile.settings.parameters.clone(),
        }
    }

    /// Create a builder for high quality video.
    #[must_use]
    pub fn high_quality_video() -> Self {
        Self::new()
            .format("mp4")
            .video_codec("h264")
            .audio_codec("aac")
            .video_bitrate(8_000_000)
            .audio_bitrate(192_000)
            .parameter("preset", "slow")
            .parameter("crf", "18")
    }

    /// Create a builder for low quality video (fast encoding).
    #[must_use]
    pub fn low_quality_video() -> Self {
        Self::new()
            .format("mp4")
            .video_codec("h264")
            .audio_codec("aac")
            .video_bitrate(1_000_000)
            .audio_bitrate(96_000)
            .parameter("preset", "ultrafast")
            .parameter("crf", "28")
    }

    /// Create a builder for high quality audio.
    #[must_use]
    pub fn high_quality_audio() -> Self {
        Self::new().format("flac").audio_codec("flac")
    }

    /// Create a builder for compressed audio.
    #[must_use]
    pub fn compressed_audio() -> Self {
        Self::new()
            .format("mp3")
            .audio_codec("mp3")
            .audio_bitrate(192_000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_basic() {
        let profile = ProfileBuilder::new()
            .name("Test Profile")
            .description("Test description")
            .format("mp4")
            .video_codec("h264")
            .audio_codec("aac")
            .build()
            .unwrap();

        assert_eq!(profile.name, "Test Profile");
        assert_eq!(profile.description, "Test description");
        assert_eq!(profile.settings.format, "mp4");
    }

    #[test]
    fn test_builder_missing_name() {
        let result = ProfileBuilder::new().format("mp4").build();

        assert!(result.is_err());
    }

    #[test]
    fn test_builder_missing_format() {
        let result = ProfileBuilder::new().name("Test").build();

        assert!(result.is_err());
    }

    #[test]
    fn test_builder_with_parameters() {
        let profile = ProfileBuilder::new()
            .name("Test")
            .format("mp4")
            .parameter("preset", "medium")
            .parameter("crf", "23")
            .build()
            .unwrap();

        assert_eq!(profile.settings.parameters.len(), 2);
        assert!(profile
            .settings
            .parameters
            .iter()
            .any(|(k, v)| k == "preset" && v == "medium"));
    }

    #[test]
    fn test_builder_with_resolution() {
        let profile = ProfileBuilder::new()
            .name("Test")
            .format("mp4")
            .resolution(1920, 1080)
            .build()
            .unwrap();

        assert_eq!(profile.settings.resolution, Some((1920, 1080)));
    }

    #[test]
    fn test_builder_with_bitrates() {
        let profile = ProfileBuilder::new()
            .name("Test")
            .format("mp4")
            .video_bitrate(5_000_000)
            .audio_bitrate(192_000)
            .build()
            .unwrap();

        assert_eq!(profile.settings.video_bitrate, Some(5_000_000));
        assert_eq!(profile.settings.audio_bitrate, Some(192_000));
    }

    #[test]
    fn test_high_quality_video() {
        let profile = ProfileBuilder::high_quality_video()
            .name("HQ Video")
            .build()
            .unwrap();

        assert_eq!(profile.settings.format, "mp4");
        assert_eq!(profile.settings.video_codec, Some("h264".to_string()));
        assert!(profile.settings.video_bitrate.unwrap() > 5_000_000);
    }

    #[test]
    fn test_low_quality_video() {
        let profile = ProfileBuilder::low_quality_video()
            .name("LQ Video")
            .build()
            .unwrap();

        assert_eq!(profile.settings.format, "mp4");
        assert!(profile.settings.video_bitrate.unwrap() < 2_000_000);
    }

    #[test]
    fn test_high_quality_audio() {
        let profile = ProfileBuilder::high_quality_audio()
            .name("HQ Audio")
            .build()
            .unwrap();

        assert_eq!(profile.settings.format, "flac");
        assert_eq!(profile.settings.audio_codec, Some("flac".to_string()));
    }

    #[test]
    fn test_from_profile() {
        let original = ProfileBuilder::new()
            .name("Original")
            .format("mp4")
            .video_codec("h264")
            .build()
            .unwrap();

        let cloned = ProfileBuilder::from_profile(&original)
            .name("Cloned")
            .build()
            .unwrap();

        assert_eq!(cloned.name, "Cloned");
        assert_eq!(cloned.settings.format, original.settings.format);
        assert_eq!(cloned.settings.video_codec, original.settings.video_codec);
    }
}
