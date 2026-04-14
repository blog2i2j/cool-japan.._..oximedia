//! High-level media file property descriptors.
//!
//! [`MediaProperties`] aggregates the most commonly queried properties of a
//! media file — duration, stream counts, codec identifiers, resolution, and
//! bitrate — into a single, cheaply cloneable struct.
//!
//! The struct is intentionally kept format-agnostic.  Container-specific
//! metadata (MP4 `moov` atoms, MKV `Info` elements, etc.) is handled by the
//! `oximedia-container` crate; this module only defines the normalized,
//! codec-independent view.

#![allow(dead_code)]

use crate::types::{CodecId, PixelFormat, Rational, SampleFormat};
use serde::{Deserialize, Serialize};

/// Aspect ratio kind for display metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AspectRatio {
    /// Square pixels (PAR 1:1).
    Square,
    /// Non-square pixels with explicit numerator/denominator.
    NonSquare {
        /// Pixel aspect ratio numerator.
        numerator: u32,
        /// Pixel aspect ratio denominator.
        denominator: u32,
    },
}

impl AspectRatio {
    /// Convert to a floating-point ratio (DAR/SAR approximation).
    #[must_use]
    pub fn as_f64(self) -> f64 {
        match self {
            Self::Square => 1.0,
            Self::NonSquare {
                numerator,
                denominator,
            } => {
                if denominator == 0 {
                    1.0
                } else {
                    f64::from(numerator) / f64::from(denominator)
                }
            }
        }
    }
}

impl Default for AspectRatio {
    fn default() -> Self {
        Self::Square
    }
}

/// Video stream properties extracted from codec parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VideoProps {
    /// Pixel width of the encoded frame (coded width, may include padding).
    pub width: u32,
    /// Pixel height of the encoded frame (coded height, may include padding).
    pub height: u32,
    /// Display width after cropping and aspect-ratio correction.
    pub display_width: u32,
    /// Display height after cropping and aspect-ratio correction.
    pub display_height: u32,
    /// Frame rate as a rational number (e.g. 30000/1001 for 29.97 fps).
    pub frame_rate: Rational,
    /// Pixel format / chroma subsampling.
    pub pixel_format: PixelFormat,
    /// Bit depth per sample component (8, 10, 12, …).
    pub bit_depth: u8,
    /// Sample aspect ratio.
    pub sar: AspectRatio,
    /// Video codec.
    pub codec: CodecId,
    /// Average bitrate in bits per second (0 if unknown).
    pub bitrate_bps: u64,
}

impl VideoProps {
    /// Compute display aspect ratio as a floating-point approximation.
    #[must_use]
    pub fn display_aspect_ratio(&self) -> f64 {
        if self.display_height == 0 {
            return 0.0;
        }
        f64::from(self.display_width) / f64::from(self.display_height)
    }

    /// Return the frame rate as a floating-point value.
    #[must_use]
    pub fn fps(&self) -> f64 {
        self.frame_rate.to_f64()
    }
}

/// Audio stream properties extracted from codec parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioProps {
    /// Sample rate in Hz (e.g. 48000).
    pub sample_rate: u32,
    /// Number of audio channels (1 = mono, 2 = stereo, …).
    pub channels: u8,
    /// Sample format (PCM encoding type).
    pub sample_format: SampleFormat,
    /// Average bitrate in bits per second (0 if unknown).
    pub bitrate_bps: u64,
    /// Audio codec.
    pub codec: CodecId,
    /// Audio delay in milliseconds relative to the video stream (may be 0).
    pub delay_ms: i64,
}

impl AudioProps {
    /// Compute the bit depth implied by the sample format.
    #[must_use]
    pub fn bit_depth(&self) -> u8 {
        self.sample_format.bit_depth()
    }
}

/// Container-format hint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContainerHint {
    /// ISO Base Media / MPEG-4 Part 12 (MP4, M4V, MOV, …).
    IsoBaseMf,
    /// Matroska (MKV).
    Matroska,
    /// WebM (restricted Matroska).
    WebM,
    /// AVI.
    Avi,
    /// Ogg container.
    Ogg,
    /// MPEG Transport Stream.
    MpegTs,
    /// FLAC audio-only.
    Flac,
    /// WAV audio-only.
    Wav,
    /// Unknown or not determined.
    Unknown,
}

impl Default for ContainerHint {
    fn default() -> Self {
        Self::Unknown
    }
}

/// Aggregated media file properties.
///
/// This is the top-level type returned by container probers and used by repair
/// and transcode pipelines to decide which actions are needed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MediaProperties {
    /// Total duration of the file in seconds.  `None` if indeterminate.
    pub duration_secs: Option<f64>,
    /// Overall bitrate of all streams combined, in bits per second.
    pub total_bitrate_bps: u64,
    /// Number of video streams.
    pub video_stream_count: usize,
    /// Number of audio streams.
    pub audio_stream_count: usize,
    /// Number of subtitle streams.
    pub subtitle_stream_count: usize,
    /// Properties of the first (primary) video stream, if present.
    pub primary_video: Option<VideoProps>,
    /// Properties of the first (primary) audio stream, if present.
    pub primary_audio: Option<AudioProps>,
    /// Container format hint derived from the file extension / magic bytes.
    pub container: ContainerHint,
    /// Whether the file has a valid seek index.
    pub has_index: bool,
    /// Whether the file appears to be truncated.
    pub is_truncated: bool,
}

impl Default for MediaProperties {
    fn default() -> Self {
        Self {
            duration_secs: None,
            total_bitrate_bps: 0,
            video_stream_count: 0,
            audio_stream_count: 0,
            subtitle_stream_count: 0,
            primary_video: None,
            primary_audio: None,
            container: ContainerHint::Unknown,
            has_index: false,
            is_truncated: false,
        }
    }
}

impl MediaProperties {
    /// Create a new, empty [`MediaProperties`] descriptor.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Return `true` if the file has at least one video stream.
    #[must_use]
    pub fn has_video(&self) -> bool {
        self.video_stream_count > 0
    }

    /// Return `true` if the file has at least one audio stream.
    #[must_use]
    pub fn has_audio(&self) -> bool {
        self.audio_stream_count > 0
    }

    /// Return `true` if both a video and audio stream are present.
    #[must_use]
    pub fn is_av_content(&self) -> bool {
        self.has_video() && self.has_audio()
    }

    /// Return the frame rate of the primary video stream, or `None`.
    #[must_use]
    pub fn frame_rate(&self) -> Option<Rational> {
        self.primary_video.as_ref().map(|v| v.frame_rate)
    }

    /// Return the resolution of the primary video stream as `(width, height)`.
    #[must_use]
    pub fn resolution(&self) -> Option<(u32, u32)> {
        self.primary_video
            .as_ref()
            .map(|v| (v.display_width, v.display_height))
    }

    /// Return the sample rate of the primary audio stream, or `None`.
    #[must_use]
    pub fn sample_rate(&self) -> Option<u32> {
        self.primary_audio.as_ref().map(|a| a.sample_rate)
    }

    /// Total stream count (video + audio + subtitle).
    #[must_use]
    pub fn total_stream_count(&self) -> usize {
        self.video_stream_count + self.audio_stream_count + self.subtitle_stream_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{CodecId, PixelFormat, Rational, SampleFormat};

    fn sample_video_props() -> VideoProps {
        VideoProps {
            width: 1920,
            height: 1080,
            display_width: 1920,
            display_height: 1080,
            frame_rate: Rational::new(24000, 1001),
            pixel_format: PixelFormat::Yuv420p,
            bit_depth: 8,
            sar: AspectRatio::Square,
            codec: CodecId::Av1,
            bitrate_bps: 4_000_000,
        }
    }

    fn sample_audio_props() -> AudioProps {
        AudioProps {
            sample_rate: 48_000,
            channels: 2,
            sample_format: SampleFormat::F32,
            bitrate_bps: 192_000,
            codec: CodecId::Opus,
            delay_ms: 0,
        }
    }

    #[test]
    fn default_properties_empty() {
        let props = MediaProperties::default();
        assert!(!props.has_video());
        assert!(!props.has_audio());
        assert!(!props.is_av_content());
        assert_eq!(props.total_stream_count(), 0);
    }

    #[test]
    fn has_video_and_audio() {
        let props = MediaProperties {
            video_stream_count: 1,
            audio_stream_count: 2,
            primary_video: Some(sample_video_props()),
            primary_audio: Some(sample_audio_props()),
            ..Default::default()
        };
        assert!(props.has_video());
        assert!(props.has_audio());
        assert!(props.is_av_content());
        assert_eq!(props.total_stream_count(), 3);
    }

    #[test]
    fn resolution_from_video_props() {
        let props = MediaProperties {
            video_stream_count: 1,
            primary_video: Some(sample_video_props()),
            ..Default::default()
        };
        assert_eq!(props.resolution(), Some((1920, 1080)));
    }

    #[test]
    fn sample_rate_from_audio_props() {
        let props = MediaProperties {
            audio_stream_count: 1,
            primary_audio: Some(sample_audio_props()),
            ..Default::default()
        };
        assert_eq!(props.sample_rate(), Some(48_000));
    }

    #[test]
    fn frame_rate_from_video_props() {
        let props = MediaProperties {
            video_stream_count: 1,
            primary_video: Some(sample_video_props()),
            ..Default::default()
        };
        let fps = props.frame_rate().expect("fps").to_f64();
        assert!((fps - 23.976_f64).abs() < 0.001);
    }

    #[test]
    fn aspect_ratio_square() {
        let ar = AspectRatio::Square;
        assert_eq!(ar.as_f64(), 1.0);
    }

    #[test]
    fn aspect_ratio_non_square() {
        let ar = AspectRatio::NonSquare {
            numerator: 16,
            denominator: 9,
        };
        let r = ar.as_f64();
        assert!((r - 16.0 / 9.0).abs() < 1e-9);
    }

    #[test]
    fn aspect_ratio_zero_denominator() {
        let ar = AspectRatio::NonSquare {
            numerator: 4,
            denominator: 0,
        };
        assert_eq!(ar.as_f64(), 1.0);
    }

    #[test]
    fn display_aspect_ratio_video_props() {
        let vp = sample_video_props();
        let dar = vp.display_aspect_ratio();
        assert!((dar - 16.0 / 9.0).abs() < 1e-9);
    }

    #[test]
    fn total_stream_count_includes_subtitles() {
        let props = MediaProperties {
            video_stream_count: 1,
            audio_stream_count: 1,
            subtitle_stream_count: 3,
            ..Default::default()
        };
        assert_eq!(props.total_stream_count(), 5);
    }

    #[test]
    fn video_fps_approx() {
        let vp = sample_video_props();
        assert!((vp.fps() - 23.976).abs() < 0.001);
    }

    #[test]
    fn audio_bit_depth() {
        let ap = sample_audio_props();
        assert_eq!(ap.bit_depth(), SampleFormat::F32.bit_depth());
    }
}
