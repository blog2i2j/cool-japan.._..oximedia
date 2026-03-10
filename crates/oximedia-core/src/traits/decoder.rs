//! Decoder traits for video and audio.
//!
//! This module provides traits for implementing video and audio decoders.
//! These traits define the interface that all decoder implementations must follow.

use crate::error::OxiResult;
use crate::types::{CodecId, PixelFormat, SampleFormat, Timestamp};

/// Video frame produced by a video decoder.
///
/// Represents a decoded video frame with pixel data and metadata.
#[derive(Debug)]
pub struct VideoFrame {
    /// Pixel format of the frame data.
    pub format: PixelFormat,
    /// Width of the frame in pixels.
    pub width: u32,
    /// Height of the frame in pixels.
    pub height: u32,
    /// Timestamp information for this frame.
    pub timestamp: Timestamp,
    /// Plane data for the frame.
    /// For planar formats, contains one slice per plane.
    /// For packed formats, contains a single slice.
    pub planes: Vec<Vec<u8>>,
    /// Stride (bytes per row) for each plane.
    pub strides: Vec<usize>,
    /// Whether this frame is a keyframe.
    pub is_keyframe: bool,
}

impl VideoFrame {
    /// Creates a new video frame with the given parameters.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        format: PixelFormat,
        width: u32,
        height: u32,
        timestamp: Timestamp,
        planes: Vec<Vec<u8>>,
        strides: Vec<usize>,
        is_keyframe: bool,
    ) -> Self {
        Self {
            format,
            width,
            height,
            timestamp,
            planes,
            strides,
            is_keyframe,
        }
    }
}

/// Audio frame produced by an audio decoder.
///
/// Represents decoded audio samples with metadata.
#[derive(Debug)]
pub struct AudioFrame {
    /// Sample format of the audio data.
    pub format: SampleFormat,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Number of channels.
    pub channels: u16,
    /// Number of samples per channel in this frame.
    pub samples: usize,
    /// Timestamp information for this frame.
    pub timestamp: Timestamp,
    /// Audio data.
    /// For interleaved formats, contains a single buffer.
    /// For planar formats, contains one buffer per channel.
    pub data: Vec<Vec<u8>>,
}

impl AudioFrame {
    /// Creates a new audio frame with the given parameters.
    #[must_use]
    pub fn new(
        format: SampleFormat,
        sample_rate: u32,
        channels: u16,
        samples: usize,
        timestamp: Timestamp,
        data: Vec<Vec<u8>>,
    ) -> Self {
        Self {
            format,
            sample_rate,
            channels,
            samples,
            timestamp,
            data,
        }
    }

    /// Returns the duration of this frame in seconds.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn duration_seconds(&self) -> f64 {
        self.samples as f64 / f64::from(self.sample_rate)
    }
}

/// Trait for video decoder implementations.
///
/// Implementors decode compressed video data into raw video frames.
///
/// # Examples
///
/// ```ignore
/// use oximedia_core::traits::{VideoDecoder, VideoFrame};
///
/// fn decode_video(decoder: &mut impl VideoDecoder, data: &[u8]) -> Vec<VideoFrame> {
///     decoder.send_packet(data)?;
///     let mut frames = Vec::new();
///     while let Some(frame) = decoder.receive_frame()? {
///         frames.push(frame);
///     }
///     frames
/// }
/// ```
pub trait VideoDecoder {
    /// Returns the codec ID this decoder handles.
    fn codec_id(&self) -> CodecId;

    /// Returns the output pixel format.
    fn output_format(&self) -> PixelFormat;

    /// Returns the output dimensions (width, height).
    fn output_dimensions(&self) -> (u32, u32);

    /// Sends a compressed packet to the decoder.
    ///
    /// # Arguments
    ///
    /// * `data` - Compressed video data
    ///
    /// # Errors
    ///
    /// Returns an error if the packet is invalid or the decoder is in an error state.
    fn send_packet(&mut self, data: &[u8]) -> OxiResult<()>;

    /// Receives a decoded frame from the decoder.
    ///
    /// Returns `None` if no frame is currently available.
    /// Call repeatedly after `send_packet` to get all decoded frames.
    ///
    /// # Errors
    ///
    /// Returns an error if decoding fails.
    fn receive_frame(&mut self) -> OxiResult<Option<VideoFrame>>;

    /// Flushes the decoder, signaling end of stream.
    ///
    /// After calling flush, continue calling `receive_frame` to get
    /// any remaining buffered frames.
    ///
    /// # Errors
    ///
    /// Returns an error if flushing fails.
    fn flush(&mut self) -> OxiResult<()>;

    /// Resets the decoder state.
    ///
    /// Call this when seeking to a new position in the stream.
    ///
    /// # Errors
    ///
    /// Returns an error if reset fails.
    fn reset(&mut self) -> OxiResult<()>;
}

/// Trait for audio decoder implementations.
///
/// Implementors decode compressed audio data into raw audio frames.
///
/// # Examples
///
/// ```ignore
/// use oximedia_core::traits::{AudioDecoder, AudioFrame};
///
/// fn decode_audio(decoder: &mut impl AudioDecoder, data: &[u8]) -> Vec<AudioFrame> {
///     decoder.send_packet(data)?;
///     let mut frames = Vec::new();
///     while let Some(frame) = decoder.receive_frame()? {
///         frames.push(frame);
///     }
///     frames
/// }
/// ```
pub trait AudioDecoder {
    /// Returns the codec ID this decoder handles.
    fn codec_id(&self) -> CodecId;

    /// Returns the output sample format.
    fn output_format(&self) -> SampleFormat;

    /// Returns the sample rate in Hz.
    fn sample_rate(&self) -> u32;

    /// Returns the number of output channels.
    fn channels(&self) -> u16;

    /// Sends a compressed packet to the decoder.
    ///
    /// # Arguments
    ///
    /// * `data` - Compressed audio data
    ///
    /// # Errors
    ///
    /// Returns an error if the packet is invalid or the decoder is in an error state.
    fn send_packet(&mut self, data: &[u8]) -> OxiResult<()>;

    /// Receives a decoded frame from the decoder.
    ///
    /// Returns `None` if no frame is currently available.
    /// Call repeatedly after `send_packet` to get all decoded frames.
    ///
    /// # Errors
    ///
    /// Returns an error if decoding fails.
    fn receive_frame(&mut self) -> OxiResult<Option<AudioFrame>>;

    /// Flushes the decoder, signaling end of stream.
    ///
    /// After calling flush, continue calling `receive_frame` to get
    /// any remaining buffered frames.
    ///
    /// # Errors
    ///
    /// Returns an error if flushing fails.
    fn flush(&mut self) -> OxiResult<()>;

    /// Resets the decoder state.
    ///
    /// Call this when seeking to a new position in the stream.
    ///
    /// # Errors
    ///
    /// Returns an error if reset fails.
    fn reset(&mut self) -> OxiResult<()>;
}

/// Subtitle cue/event produced by a subtitle decoder.
///
/// Represents a single subtitle entry with timing and text/markup.
#[derive(Debug, Clone)]
pub struct SubtitleFrame {
    /// Start timestamp for this subtitle.
    pub start: Timestamp,
    /// End timestamp for this subtitle (duration).
    pub end: Timestamp,
    /// Text content of the subtitle.
    /// For text-based formats (SRT, `WebVTT`), contains the formatted text.
    /// For markup formats (ASS/SSA), contains the markup string.
    pub text: String,
    /// Optional subtitle layer (for overlapping subtitles).
    pub layer: u32,
    /// Optional subtitle position/alignment settings.
    pub settings: Option<SubtitleSettings>,
}

impl SubtitleFrame {
    /// Creates a new subtitle frame with the given parameters.
    #[must_use]
    pub fn new(start: Timestamp, end: Timestamp, text: impl Into<String>) -> Self {
        Self {
            start,
            end,
            text: text.into(),
            layer: 0,
            settings: None,
        }
    }

    /// Returns the duration of this subtitle in seconds.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn duration_seconds(&self) -> f64 {
        let start_sec = self.start.to_seconds();
        let end_sec = self.end.to_seconds();
        end_sec - start_sec
    }

    /// Sets the layer for this subtitle.
    #[must_use]
    pub const fn with_layer(mut self, layer: u32) -> Self {
        self.layer = layer;
        self
    }

    /// Sets the settings for this subtitle.
    #[must_use]
    pub fn with_settings(mut self, settings: SubtitleSettings) -> Self {
        self.settings = Some(settings);
        self
    }
}

/// Subtitle positioning and styling settings.
///
/// Provides optional positioning, alignment, and styling information
/// for subtitle rendering.
#[derive(Debug, Clone, Default)]
pub struct SubtitleSettings {
    /// Horizontal alignment (left, center, right).
    pub align_h: Option<HorizontalAlign>,
    /// Vertical alignment (top, middle, bottom).
    pub align_v: Option<VerticalAlign>,
    /// Position on screen (0.0 - 1.0).
    pub position: Option<(f32, f32)>,
    /// Size of subtitle region (width, height in 0.0 - 1.0).
    pub size: Option<(f32, f32)>,
}

/// Horizontal text alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HorizontalAlign {
    /// Left-aligned.
    Left,
    /// Center-aligned.
    Center,
    /// Right-aligned.
    Right,
}

/// Vertical text alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerticalAlign {
    /// Top-aligned.
    Top,
    /// Middle-aligned.
    Middle,
    /// Bottom-aligned.
    Bottom,
}

/// Trait for subtitle decoder implementations.
///
/// Implementors decode subtitle data into subtitle frames/cues.
///
/// # Examples
///
/// ```ignore
/// use oximedia_core::traits::{SubtitleDecoder, SubtitleFrame};
///
/// fn decode_subtitle(decoder: &mut impl SubtitleDecoder, data: &[u8]) -> Vec<SubtitleFrame> {
///     decoder.send_packet(data)?;
///     let mut frames = Vec::new();
///     while let Some(frame) = decoder.receive_frame()? {
///         frames.push(frame);
///     }
///     frames
/// }
/// ```
pub trait SubtitleDecoder {
    /// Returns the codec ID this decoder handles.
    fn codec_id(&self) -> CodecId;

    /// Sends a subtitle packet to the decoder.
    ///
    /// # Arguments
    ///
    /// * `data` - Subtitle data (text or binary)
    ///
    /// # Errors
    ///
    /// Returns an error if the packet is invalid or the decoder is in an error state.
    fn send_packet(&mut self, data: &[u8]) -> OxiResult<()>;

    /// Receives a decoded subtitle frame from the decoder.
    ///
    /// Returns `None` if no frame is currently available.
    /// Call repeatedly after `send_packet` to get all decoded frames.
    ///
    /// # Errors
    ///
    /// Returns an error if decoding fails.
    fn receive_frame(&mut self) -> OxiResult<Option<SubtitleFrame>>;

    /// Flushes the decoder, signaling end of stream.
    ///
    /// After calling flush, continue calling `receive_frame` to get
    /// any remaining buffered frames.
    ///
    /// # Errors
    ///
    /// Returns an error if flushing fails.
    fn flush(&mut self) -> OxiResult<()>;

    /// Resets the decoder state.
    ///
    /// Call this when seeking to a new position in the stream.
    ///
    /// # Errors
    ///
    /// Returns an error if reset fails.
    fn reset(&mut self) -> OxiResult<()>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Rational;

    #[test]
    fn test_video_frame_new() {
        let timestamp = Timestamp::new(0, Rational::new(1, 1000));
        let frame = VideoFrame::new(
            PixelFormat::Yuv420p,
            1920,
            1080,
            timestamp,
            vec![
                vec![0u8; 1920 * 1080],
                vec![0u8; 960 * 540],
                vec![0u8; 960 * 540],
            ],
            vec![1920, 960, 960],
            true,
        );

        assert_eq!(frame.format, PixelFormat::Yuv420p);
        assert_eq!(frame.width, 1920);
        assert_eq!(frame.height, 1080);
        assert!(frame.is_keyframe);
    }

    #[test]
    fn test_audio_frame_new() {
        let timestamp = Timestamp::new(0, Rational::new(1, 48000));
        let frame = AudioFrame::new(
            SampleFormat::F32,
            48000,
            2,
            1024,
            timestamp,
            vec![vec![0u8; 1024 * 2 * 4]],
        );

        assert_eq!(frame.format, SampleFormat::F32);
        assert_eq!(frame.sample_rate, 48000);
        assert_eq!(frame.channels, 2);
        assert_eq!(frame.samples, 1024);
    }

    #[test]
    fn test_audio_frame_duration() {
        let timestamp = Timestamp::new(0, Rational::new(1, 48000));
        let frame = AudioFrame::new(
            SampleFormat::F32,
            48000,
            2,
            48000, // 1 second of samples
            timestamp,
            vec![vec![0u8; 48000 * 2 * 4]],
        );

        assert!((frame.duration_seconds() - 1.0).abs() < f64::EPSILON);
    }
}
