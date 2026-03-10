// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Smart conversion features with automatic optimization.

use crate::formats::{AudioCodec, ContainerFormat, VideoCodec};
use crate::pipeline::{AudioSettings, BitrateMode, VideoSettings};
use crate::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Smart converter that automatically selects optimal settings.
#[derive(Debug, Clone)]
pub struct SmartConverter {
    analyzer: MediaAnalyzer,
    optimizer: SettingsOptimizer,
}

impl SmartConverter {
    /// Create a new smart converter.
    #[must_use]
    pub fn new() -> Self {
        Self {
            analyzer: MediaAnalyzer::new(),
            optimizer: SettingsOptimizer::new(),
        }
    }

    /// Analyze input and determine optimal conversion settings.
    pub async fn analyze_and_optimize(
        &self,
        input: &Path,
        target: ConversionTarget,
    ) -> Result<OptimizedSettings> {
        let analysis = self.analyzer.analyze(input).await?;
        self.optimizer.optimize(&analysis, target)
    }
}

impl Default for SmartConverter {
    fn default() -> Self {
        Self::new()
    }
}

/// Media analyzer for examining input files.
#[derive(Debug, Clone)]
pub struct MediaAnalyzer;

impl MediaAnalyzer {
    /// Create a new media analyzer.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Analyze media file.
    ///
    /// Reads file-level metadata (size, extension) and returns a best-effort
    /// `MediaAnalysis`. Full demux / codec probing requires the transcode
    /// pipeline and is deferred; the inferred fields default to common values.
    pub async fn analyze(&self, path: &Path) -> Result<MediaAnalysis> {
        let file_size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);

        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

        // Infer container-level media presence from extension.
        let (has_video, has_audio, video_codec, audio_codec) = match ext.to_lowercase().as_str() {
            "mp4" | "m4v" | "mov" => (true, true, Some(VideoCodec::Vp8), Some(AudioCodec::Opus)),
            "webm" => (true, true, Some(VideoCodec::Vp9), Some(AudioCodec::Opus)),
            "mkv" => (true, true, Some(VideoCodec::Av1), Some(AudioCodec::Opus)),
            "mp3" | "aac" | "ogg" | "flac" | "wav" => (false, true, None, Some(AudioCodec::Opus)),
            "png" | "jpg" | "jpeg" | "webp" | "tiff" | "tif" | "dpx" | "exr" => {
                (true, false, None, None)
            }
            _ => (true, true, Some(VideoCodec::Vp9), Some(AudioCodec::Opus)),
        };

        Ok(MediaAnalysis {
            has_video,
            has_audio,
            video_codec,
            audio_codec,
            // Resolution, frame rate, bitrate and duration require demuxing;
            // leave as None until the transcode pipeline is integrated.
            resolution: None,
            frame_rate: None,
            bitrate: None,
            duration_seconds: None,
            file_size,
            is_hdr: false,
            is_interlaced: false,
        })
    }
}

impl Default for MediaAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Media analysis result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MediaAnalysis {
    /// Has video stream
    pub has_video: bool,
    /// Has audio stream
    pub has_audio: bool,
    /// Video codec (if present)
    pub video_codec: Option<VideoCodec>,
    /// Audio codec (if present)
    pub audio_codec: Option<AudioCodec>,
    /// Video resolution (width, height)
    pub resolution: Option<(u32, u32)>,
    /// Frame rate
    pub frame_rate: Option<f64>,
    /// Bitrate in bits per second
    pub bitrate: Option<u64>,
    /// Duration in seconds
    pub duration_seconds: Option<f64>,
    /// File size in bytes
    pub file_size: u64,
    /// Is HDR content
    pub is_hdr: bool,
    /// Is interlaced
    pub is_interlaced: bool,
}

/// Conversion target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConversionTarget {
    /// Optimize for web streaming
    WebStreaming,
    /// Optimize for mobile devices
    Mobile,
    /// Optimize for maximum quality
    MaxQuality,
    /// Optimize for smallest file size
    MinSize,
    /// Optimize for fast encoding
    FastEncoding,
}

/// Settings optimizer.
#[derive(Debug, Clone)]
pub struct SettingsOptimizer;

impl SettingsOptimizer {
    /// Create a new settings optimizer.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Optimize settings based on analysis and target.
    pub fn optimize(
        &self,
        analysis: &MediaAnalysis,
        target: ConversionTarget,
    ) -> Result<OptimizedSettings> {
        let container = self.select_container(target);
        let video = if analysis.has_video {
            Some(self.optimize_video(analysis, target)?)
        } else {
            None
        };
        let audio = if analysis.has_audio {
            Some(self.optimize_audio(analysis, target)?)
        } else {
            None
        };

        Ok(OptimizedSettings {
            container,
            video,
            audio,
            rationale: self.generate_rationale(analysis, target),
        })
    }

    fn select_container(&self, target: ConversionTarget) -> ContainerFormat {
        match target {
            ConversionTarget::WebStreaming => ContainerFormat::Webm,
            ConversionTarget::Mobile => ContainerFormat::Mp4,
            ConversionTarget::MaxQuality => ContainerFormat::Matroska,
            ConversionTarget::MinSize => ContainerFormat::Webm,
            ConversionTarget::FastEncoding => ContainerFormat::Mp4,
        }
    }

    fn optimize_video(
        &self,
        analysis: &MediaAnalysis,
        target: ConversionTarget,
    ) -> Result<VideoSettings> {
        let codec = match target {
            ConversionTarget::WebStreaming | ConversionTarget::MinSize => VideoCodec::Vp9,
            ConversionTarget::Mobile | ConversionTarget::FastEncoding => VideoCodec::Vp8,
            ConversionTarget::MaxQuality => VideoCodec::Av1,
        };

        let bitrate = match target {
            ConversionTarget::MinSize => BitrateMode::Crf(45),
            ConversionTarget::FastEncoding => BitrateMode::Cbr(2_000_000),
            ConversionTarget::MaxQuality => BitrateMode::Crf(20),
            _ => BitrateMode::Crf(31),
        };

        Ok(VideoSettings {
            codec,
            resolution: analysis.resolution,
            frame_rate: analysis.frame_rate,
            bitrate,
            quality: None,
            two_pass: matches!(
                target,
                ConversionTarget::MaxQuality | ConversionTarget::WebStreaming
            ),
            speed: match target {
                ConversionTarget::FastEncoding => crate::pipeline::EncodingSpeed::Fast,
                ConversionTarget::MaxQuality => crate::pipeline::EncodingSpeed::VerySlow,
                _ => crate::pipeline::EncodingSpeed::Medium,
            },
            tone_map: analysis.is_hdr,
        })
    }

    fn optimize_audio(
        &self,
        _analysis: &MediaAnalysis,
        target: ConversionTarget,
    ) -> Result<AudioSettings> {
        let codec = match target {
            ConversionTarget::MaxQuality => AudioCodec::Flac,
            _ => AudioCodec::Opus,
        };

        let bitrate = match target {
            ConversionTarget::MinSize => 96_000,
            ConversionTarget::MaxQuality => 256_000,
            _ => 128_000,
        };

        Ok(AudioSettings {
            codec,
            sample_rate: 48000,
            channels: crate::formats::ChannelLayout::Stereo,
            bitrate: if codec == AudioCodec::Flac {
                None
            } else {
                Some(bitrate)
            },
            normalize: false,
            normalization_target: -23.0,
        })
    }

    fn generate_rationale(&self, _analysis: &MediaAnalysis, target: ConversionTarget) -> String {
        match target {
            ConversionTarget::WebStreaming => {
                "Optimized for web streaming with VP9 codec for good quality and browser compatibility"
            }
            ConversionTarget::Mobile => {
                "Optimized for mobile devices with efficient encoding and reasonable file size"
            }
            ConversionTarget::MaxQuality => {
                "Optimized for maximum quality using AV1 codec and high bitrate settings"
            }
            ConversionTarget::MinSize => {
                "Optimized for minimum file size using aggressive compression"
            }
            ConversionTarget::FastEncoding => {
                "Optimized for fast encoding with VP8 codec and single-pass encoding"
            }
        }
        .to_string()
    }
}

impl Default for SettingsOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Optimized conversion settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizedSettings {
    /// Container format
    pub container: ContainerFormat,
    /// Video settings
    pub video: Option<VideoSettings>,
    /// Audio settings
    pub audio: Option<AudioSettings>,
    /// Rationale for these settings
    pub rationale: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smart_converter() {
        // SmartConverter is a ZST; verify it can be constructed
        let _converter = SmartConverter::new();
        assert_eq!(std::mem::size_of::<SmartConverter>(), 0);
    }

    #[test]
    fn test_settings_optimizer() {
        let optimizer = SettingsOptimizer::new();
        let analysis = MediaAnalysis {
            has_video: true,
            has_audio: true,
            video_codec: Some(VideoCodec::Vp9),
            audio_codec: Some(AudioCodec::Opus),
            resolution: Some((1920, 1080)),
            frame_rate: Some(30.0),
            bitrate: Some(5_000_000),
            duration_seconds: Some(300.0),
            file_size: 625_000_000,
            is_hdr: false,
            is_interlaced: false,
        };

        let result = optimizer.optimize(&analysis, ConversionTarget::WebStreaming);
        assert!(result.is_ok());

        let settings = result.expect("optimization should succeed");
        assert_eq!(settings.container, ContainerFormat::Webm);
        assert!(settings.video.is_some());
        assert!(settings.audio.is_some());
    }

    #[test]
    fn test_all_conversion_targets() {
        let optimizer = SettingsOptimizer::new();
        let analysis = MediaAnalysis {
            has_video: true,
            has_audio: true,
            video_codec: Some(VideoCodec::Vp9),
            audio_codec: Some(AudioCodec::Opus),
            resolution: Some((1920, 1080)),
            frame_rate: Some(30.0),
            bitrate: Some(5_000_000),
            duration_seconds: Some(300.0),
            file_size: 625_000_000,
            is_hdr: false,
            is_interlaced: false,
        };

        assert!(optimizer
            .optimize(&analysis, ConversionTarget::WebStreaming)
            .is_ok());
        assert!(optimizer
            .optimize(&analysis, ConversionTarget::Mobile)
            .is_ok());
        assert!(optimizer
            .optimize(&analysis, ConversionTarget::MaxQuality)
            .is_ok());
        assert!(optimizer
            .optimize(&analysis, ConversionTarget::MinSize)
            .is_ok());
        assert!(optimizer
            .optimize(&analysis, ConversionTarget::FastEncoding)
            .is_ok());
    }
}
