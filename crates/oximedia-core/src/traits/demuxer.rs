//! Demuxer trait for container format parsing.
//!
//! This module provides the [`Demuxer`] trait for implementing container
//! format parsers (`WebM`, Ogg, etc.).

use crate::error::OxiResult;
use crate::types::{CodecId, MediaType, Rational, Timestamp};

/// Information about a media stream within a container.
///
/// Describes the properties of a single stream (video, audio, subtitle, etc.)
/// within a multimedia container.
#[derive(Clone, Debug)]
pub struct StreamInfo {
    /// Unique identifier for this stream within the container.
    pub index: u32,
    /// Type of media in this stream.
    pub media_type: MediaType,
    /// Codec used by this stream.
    pub codec_id: CodecId,
    /// Timebase for timestamps in this stream.
    pub timebase: Rational,
    /// Duration of the stream in timebase units (if known).
    pub duration: Option<i64>,
    /// Codec-specific extra data needed for decoder initialization.
    pub extra_data: Vec<u8>,
    /// Video-specific parameters (if this is a video stream).
    pub video: Option<VideoStreamInfo>,
    /// Audio-specific parameters (if this is an audio stream).
    pub audio: Option<AudioStreamInfo>,
}

/// Video-specific stream information.
#[derive(Clone, Debug)]
pub struct VideoStreamInfo {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Frame rate as a rational number.
    pub frame_rate: Option<Rational>,
    /// Pixel aspect ratio (None means square pixels).
    pub pixel_aspect_ratio: Option<Rational>,
}

/// Audio-specific stream information.
#[derive(Clone, Debug)]
pub struct AudioStreamInfo {
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Number of channels.
    pub channels: u16,
    /// Bits per sample (if applicable).
    pub bits_per_sample: Option<u16>,
}

/// A packet of compressed data from a demuxer.
///
/// Represents a chunk of compressed data for a single stream,
/// typically containing one or more compressed frames.
#[derive(Clone, Debug)]
pub struct Packet {
    /// Stream index this packet belongs to.
    pub stream_index: u32,
    /// Timestamp for this packet.
    pub timestamp: Timestamp,
    /// Compressed data.
    pub data: Vec<u8>,
    /// Whether this packet contains a keyframe.
    pub is_keyframe: bool,
}

impl Packet {
    /// Creates a new packet.
    #[must_use]
    pub fn new(stream_index: u32, timestamp: Timestamp, data: Vec<u8>, is_keyframe: bool) -> Self {
        Self {
            stream_index,
            timestamp,
            data,
            is_keyframe,
        }
    }
}

/// Container format information.
///
/// Metadata about the container format itself.
#[derive(Clone, Debug, Default)]
pub struct ContainerInfo {
    /// Container format name (e.g., "webm", "ogg").
    pub format_name: String,
    /// Total duration in seconds (if known).
    pub duration: Option<f64>,
    /// File size in bytes (if known).
    pub file_size: Option<u64>,
    /// Bit rate in bits per second (if known).
    pub bit_rate: Option<u64>,
    /// Whether the container supports seeking.
    pub seekable: bool,
}

/// Trait for demuxer implementations.
///
/// A demuxer parses a container format (`WebM`, Ogg, etc.) and extracts
/// compressed streams from it.
///
/// # Examples
///
/// ```ignore
/// use oximedia_core::traits::{Demuxer, Packet};
///
/// fn read_packets(demuxer: &mut impl Demuxer) -> Vec<Packet> {
///     let mut packets = Vec::new();
///     while let Ok(Some(packet)) = demuxer.read_packet() {
///         packets.push(packet);
///     }
///     packets
/// }
/// ```
pub trait Demuxer {
    /// Returns information about the container format.
    fn container_info(&self) -> &ContainerInfo;

    /// Returns the number of streams in the container.
    fn stream_count(&self) -> usize;

    /// Returns information about a specific stream.
    ///
    /// # Arguments
    ///
    /// * `index` - Stream index (0-based)
    ///
    /// # Panics
    ///
    /// May panic if the index is out of bounds.
    fn stream_info(&self, index: usize) -> &StreamInfo;

    /// Returns information about all streams.
    fn streams(&self) -> Vec<&StreamInfo>;

    /// Reads the next packet from the container.
    ///
    /// Returns `None` when end of stream is reached.
    ///
    /// # Errors
    ///
    /// Returns an error if reading fails.
    fn read_packet(&mut self) -> OxiResult<Option<Packet>>;

    /// Seeks to a specific position in the stream.
    ///
    /// # Arguments
    ///
    /// * `timestamp` - Target timestamp in seconds
    /// * `stream_index` - Optional stream to use for seeking (uses first video stream if None)
    ///
    /// # Errors
    ///
    /// Returns an error if seeking is not supported or fails.
    fn seek(&mut self, timestamp: f64, stream_index: Option<u32>) -> OxiResult<()>;

    /// Returns whether the demuxer supports seeking.
    fn is_seekable(&self) -> bool {
        self.container_info().seekable
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_info() {
        let info = StreamInfo {
            index: 0,
            media_type: MediaType::Video,
            codec_id: CodecId::Av1,
            timebase: Rational::new(1, 1000),
            duration: Some(60000),
            extra_data: vec![],
            video: Some(VideoStreamInfo {
                width: 1920,
                height: 1080,
                frame_rate: Some(Rational::new(30, 1)),
                pixel_aspect_ratio: None,
            }),
            audio: None,
        };

        assert_eq!(info.media_type, MediaType::Video);
        assert_eq!(info.codec_id, CodecId::Av1);
        let video = info.video.as_ref().expect("should have value");
        assert_eq!(video.width, 1920);
        assert_eq!(video.height, 1080);
    }

    #[test]
    fn test_packet_new() {
        let timestamp = Timestamp::new(1000, Rational::new(1, 1000));
        let packet = Packet::new(0, timestamp, vec![0u8; 1024], true);

        assert_eq!(packet.stream_index, 0);
        assert!(packet.is_keyframe);
        assert_eq!(packet.data.len(), 1024);
    }

    #[test]
    fn test_container_info_default() {
        let info = ContainerInfo::default();
        assert!(info.format_name.is_empty());
        assert!(info.duration.is_none());
        assert!(!info.seekable);
    }
}
