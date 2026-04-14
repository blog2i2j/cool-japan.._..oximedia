// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Codec detection for media streams.

use crate::{ConversionError, Result};

/// Detector for media codecs.
#[derive(Debug, Clone)]
pub struct CodecDetector {
    video_codecs: Vec<CodecInfo>,
    audio_codecs: Vec<CodecInfo>,
}

impl CodecDetector {
    /// Create a new codec detector.
    #[must_use]
    pub fn new() -> Self {
        Self {
            video_codecs: Self::init_video_codecs(),
            audio_codecs: Self::init_audio_codecs(),
        }
    }

    /// Detect video codec from codec ID.
    #[must_use]
    pub fn detect_video_codec(&self, codec_id: &str) -> Option<&CodecInfo> {
        self.video_codecs
            .iter()
            .find(|c| c.id == codec_id || c.fourcc == Some(codec_id))
    }

    /// Detect audio codec from codec ID.
    #[must_use]
    pub fn detect_audio_codec(&self, codec_id: &str) -> Option<&CodecInfo> {
        self.audio_codecs.iter().find(|c| c.id == codec_id)
    }

    /// Get codec by name.
    #[must_use]
    pub fn get_codec_by_name(&self, name: &str) -> Option<&CodecInfo> {
        let name_lower = name.to_lowercase();

        self.video_codecs
            .iter()
            .chain(self.audio_codecs.iter())
            .find(|c| c.name.to_lowercase() == name_lower)
    }

    /// Check if a codec is supported for encoding.
    #[must_use]
    pub fn is_encoding_supported(&self, codec_id: &str) -> bool {
        self.detect_video_codec(codec_id)
            .or_else(|| self.detect_audio_codec(codec_id))
            .is_some_and(|c| c.encoding_supported)
    }

    /// Check if a codec is supported for decoding.
    #[must_use]
    pub fn is_decoding_supported(&self, codec_id: &str) -> bool {
        self.detect_video_codec(codec_id)
            .or_else(|| self.detect_audio_codec(codec_id))
            .is_some_and(|c| c.decoding_supported)
    }

    /// Get recommended encoder for a codec.
    pub fn get_recommended_encoder(&self, codec_type: CodecType) -> Result<&CodecInfo> {
        let codecs = match codec_type {
            CodecType::Video => &self.video_codecs,
            CodecType::Audio => &self.audio_codecs,
        };

        codecs
            .iter()
            .find(|c| c.encoding_supported && c.recommended)
            .ok_or_else(|| {
                ConversionError::UnsupportedCodec(format!(
                    "No recommended encoder for {codec_type:?}"
                ))
            })
    }

    fn init_video_codecs() -> Vec<CodecInfo> {
        vec![
            CodecInfo {
                id: "h264",
                name: "H.264/AVC",
                fourcc: Some("avc1"),
                media_type: CodecType::Video,
                encoding_supported: true,
                decoding_supported: true,
                recommended: true,
                quality_range: Some((18, 28)),
            },
            CodecInfo {
                id: "h265",
                name: "H.265/HEVC",
                fourcc: Some("hev1"),
                media_type: CodecType::Video,
                encoding_supported: true,
                decoding_supported: true,
                recommended: true,
                quality_range: Some((20, 30)),
            },
            CodecInfo {
                id: "vp8",
                name: "VP8",
                fourcc: Some("VP80"),
                media_type: CodecType::Video,
                encoding_supported: true,
                decoding_supported: true,
                recommended: false,
                quality_range: Some((4, 10)),
            },
            CodecInfo {
                id: "vp9",
                name: "VP9",
                fourcc: Some("vp09"),
                media_type: CodecType::Video,
                encoding_supported: true,
                decoding_supported: true,
                recommended: true,
                quality_range: Some((30, 40)),
            },
            CodecInfo {
                id: "av1",
                name: "AV1",
                fourcc: Some("av01"),
                media_type: CodecType::Video,
                encoding_supported: true,
                decoding_supported: true,
                recommended: true,
                quality_range: Some((30, 40)),
            },
            CodecInfo {
                id: "mpeg4",
                name: "MPEG-4 Part 2",
                fourcc: Some("mp4v"),
                media_type: CodecType::Video,
                encoding_supported: true,
                decoding_supported: true,
                recommended: false,
                quality_range: Some((2, 31)),
            },
            CodecInfo {
                id: "mpeg2video",
                name: "MPEG-2 Video",
                fourcc: Some("mpg2"),
                media_type: CodecType::Video,
                encoding_supported: true,
                decoding_supported: true,
                recommended: false,
                quality_range: None,
            },
        ]
    }

    fn init_audio_codecs() -> Vec<CodecInfo> {
        vec![
            CodecInfo {
                id: "aac",
                name: "AAC",
                fourcc: None,
                media_type: CodecType::Audio,
                encoding_supported: true,
                decoding_supported: true,
                recommended: true,
                quality_range: Some((128, 256)),
            },
            CodecInfo {
                id: "mp3",
                name: "MP3",
                fourcc: None,
                media_type: CodecType::Audio,
                encoding_supported: true,
                decoding_supported: true,
                recommended: false,
                quality_range: Some((128, 320)),
            },
            CodecInfo {
                id: "opus",
                name: "Opus",
                fourcc: None,
                media_type: CodecType::Audio,
                encoding_supported: true,
                decoding_supported: true,
                recommended: true,
                quality_range: Some((96, 256)),
            },
            CodecInfo {
                id: "vorbis",
                name: "Vorbis",
                fourcc: None,
                media_type: CodecType::Audio,
                encoding_supported: true,
                decoding_supported: true,
                recommended: false,
                quality_range: Some((128, 256)),
            },
            CodecInfo {
                id: "flac",
                name: "FLAC",
                fourcc: None,
                media_type: CodecType::Audio,
                encoding_supported: true,
                decoding_supported: true,
                recommended: true,
                quality_range: None,
            },
            CodecInfo {
                id: "pcm",
                name: "PCM",
                fourcc: None,
                media_type: CodecType::Audio,
                encoding_supported: true,
                decoding_supported: true,
                recommended: false,
                quality_range: None,
            },
        ]
    }
}

impl Default for CodecDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Information about a codec.
#[derive(Debug, Clone)]
pub struct CodecInfo {
    /// Codec identifier
    pub id: &'static str,
    /// Human-readable name
    pub name: &'static str,
    /// `FourCC` code (for video)
    pub fourcc: Option<&'static str>,
    /// Media type
    pub media_type: CodecType,
    /// Whether encoding is supported
    pub encoding_supported: bool,
    /// Whether decoding is supported
    pub decoding_supported: bool,
    /// Whether this is a recommended codec
    pub recommended: bool,
    /// Quality range (min, max) - CRF for video, kbps for audio
    pub quality_range: Option<(u32, u32)>,
}

impl CodecInfo {
    /// Get the default quality value.
    #[must_use]
    pub fn default_quality(&self) -> Option<u32> {
        self.quality_range.map(|(min, max)| (min + max) / 2)
    }

    /// Check if a quality value is within the valid range.
    #[must_use]
    pub fn is_valid_quality(&self, quality: u32) -> bool {
        match self.quality_range {
            Some((min, max)) => quality >= min && quality <= max,
            None => true,
        }
    }
}

/// Type of codec (video or audio).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodecType {
    /// Video codec
    Video,
    /// Audio codec
    Audio,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codec_detector_creation() {
        let detector = CodecDetector::new();
        assert!(!detector.video_codecs.is_empty());
        assert!(!detector.audio_codecs.is_empty());
    }

    #[test]
    fn test_video_codec_detection() {
        let detector = CodecDetector::new();

        let h264 = detector.detect_video_codec("h264");
        assert!(h264.is_some());
        assert_eq!(h264.unwrap().name, "H.264/AVC");

        let h265 = detector.detect_video_codec("h265");
        assert!(h265.is_some());
        assert_eq!(h265.unwrap().name, "H.265/HEVC");
    }

    #[test]
    fn test_audio_codec_detection() {
        let detector = CodecDetector::new();

        let aac = detector.detect_audio_codec("aac");
        assert!(aac.is_some());
        assert_eq!(aac.unwrap().name, "AAC");

        let mp3 = detector.detect_audio_codec("mp3");
        assert!(mp3.is_some());
        assert_eq!(mp3.unwrap().name, "MP3");
    }

    #[test]
    fn test_codec_support() {
        let detector = CodecDetector::new();

        assert!(detector.is_encoding_supported("h264"));
        assert!(detector.is_decoding_supported("h264"));
        assert!(detector.is_encoding_supported("aac"));
        assert!(detector.is_decoding_supported("aac"));
    }

    #[test]
    fn test_recommended_encoder() {
        let detector = CodecDetector::new();

        let video_encoder = detector.get_recommended_encoder(CodecType::Video);
        assert!(video_encoder.is_ok());
        assert!(video_encoder.unwrap().recommended);

        let audio_encoder = detector.get_recommended_encoder(CodecType::Audio);
        assert!(audio_encoder.is_ok());
        assert!(audio_encoder.unwrap().recommended);
    }

    #[test]
    fn test_codec_quality_range() {
        let detector = CodecDetector::new();

        let h264 = detector.detect_video_codec("h264").unwrap();
        assert!(h264.quality_range.is_some());
        assert!(h264.default_quality().is_some());
        assert!(h264.is_valid_quality(23));

        let aac = detector.detect_audio_codec("aac").unwrap();
        assert!(aac.quality_range.is_some());
        assert!(aac.is_valid_quality(192));
    }

    #[test]
    fn test_codec_by_name() {
        let detector = CodecDetector::new();

        let codec = detector.get_codec_by_name("H.264/AVC");
        assert!(codec.is_some());
        assert_eq!(codec.unwrap().id, "h264");
    }
}
