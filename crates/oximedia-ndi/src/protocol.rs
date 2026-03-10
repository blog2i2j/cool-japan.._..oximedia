//! NDI protocol implementation
//!
//! This module implements the core NDI protocol for frame transmission,
//! including headers, metadata, and connection management.
#![allow(dead_code)]

use crate::{AudioFormat, NdiError, Result, VideoFormat};
use bytes::{Buf, BufMut, Bytes, BytesMut};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Cursor;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tracing::{trace, warn};

/// NDI protocol magic number
const NDI_MAGIC: u32 = 0x4E444920; // "NDI "

/// NDI protocol version
const NDI_VERSION: u16 = 1;

/// Frame type identifiers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum NdiFrameType {
    /// Video frame
    Video = 0x01,
    /// Audio frame
    Audio = 0x02,
    /// Metadata packet
    Metadata = 0x03,
    /// Tally information
    Tally = 0x04,
    /// PTZ command
    Ptz = 0x05,
    /// Connection control
    Control = 0x06,
    /// Heartbeat/keepalive
    Heartbeat = 0x07,
}

impl TryFrom<u8> for NdiFrameType {
    type Error = NdiError;

    fn try_from(value: u8) -> Result<Self> {
        match value {
            0x01 => Ok(Self::Video),
            0x02 => Ok(Self::Audio),
            0x03 => Ok(Self::Metadata),
            0x04 => Ok(Self::Tally),
            0x05 => Ok(Self::Ptz),
            0x06 => Ok(Self::Control),
            0x07 => Ok(Self::Heartbeat),
            _ => Err(NdiError::Protocol(format!("Invalid frame type: {}", value))),
        }
    }
}

/// NDI frame header
#[derive(Debug, Clone)]
pub struct NdiFrameHeader {
    /// Frame type
    pub frame_type: NdiFrameType,

    /// Frame sequence number
    pub sequence: u32,

    /// Timestamp in microseconds
    pub timestamp: i64,

    /// Payload size in bytes
    pub payload_size: u32,

    /// Compression type (0 = none, 1 = SpeedHQ, 2 = H.264, etc.)
    pub compression: u8,

    /// Reserved flags
    pub flags: u16,
}

impl NdiFrameHeader {
    /// Create a new frame header
    pub fn new(frame_type: NdiFrameType, sequence: u32, timestamp: i64, payload_size: u32) -> Self {
        Self {
            frame_type,
            sequence,
            timestamp,
            payload_size,
            compression: 0,
            flags: 0,
        }
    }

    /// Encode the header to bytes
    pub fn encode(&self) -> Bytes {
        let mut buf = BytesMut::with_capacity(24);

        // Magic number
        buf.put_u32(NDI_MAGIC);

        // Version
        buf.put_u16(NDI_VERSION);

        // Frame type
        buf.put_u8(self.frame_type as u8);

        // Compression
        buf.put_u8(self.compression);

        // Sequence
        buf.put_u32(self.sequence);

        // Timestamp
        buf.put_i64(self.timestamp);

        // Payload size
        buf.put_u32(self.payload_size);

        // Flags
        buf.put_u16(self.flags);

        buf.freeze()
    }

    /// Decode a header from bytes
    pub fn decode(buf: &mut Cursor<&[u8]>) -> Result<Self> {
        if buf.remaining() < 24 {
            return Err(NdiError::Protocol(
                "Insufficient data for header".to_string(),
            ));
        }

        // Check magic number
        let magic = buf.get_u32();
        if magic != NDI_MAGIC {
            return Err(NdiError::Protocol(format!(
                "Invalid magic number: 0x{:08X}",
                magic
            )));
        }

        // Check version
        let version = buf.get_u16();
        if version != NDI_VERSION {
            warn!(
                "NDI version mismatch: expected {}, got {}",
                NDI_VERSION, version
            );
        }

        // Read frame type
        let frame_type = NdiFrameType::try_from(buf.get_u8())?;

        // Read compression
        let compression = buf.get_u8();

        // Read sequence
        let sequence = buf.get_u32();

        // Read timestamp
        let timestamp = buf.get_i64();

        // Read payload size
        let payload_size = buf.get_u32();

        // Read flags
        let flags = buf.get_u16();

        Ok(Self {
            frame_type,
            sequence,
            timestamp,
            payload_size,
            compression,
            flags,
        })
    }

    /// Get the header size in bytes
    pub const fn size() -> usize {
        24
    }
}

/// NDI video frame
#[derive(Debug, Clone)]
pub struct NdiVideoFrame {
    /// Frame header
    pub header: NdiFrameHeader,

    /// Video format
    pub format: VideoFormat,

    /// Frame data
    pub data: Bytes,

    /// Line stride in bytes
    pub stride: u32,
}

impl NdiVideoFrame {
    /// Create a new video frame
    pub fn new(
        sequence: u32,
        timestamp: i64,
        format: VideoFormat,
        data: Bytes,
        stride: u32,
    ) -> Self {
        let header =
            NdiFrameHeader::new(NdiFrameType::Video, sequence, timestamp, data.len() as u32);

        Self {
            header,
            format,
            data,
            stride,
        }
    }

    /// Encode the video frame
    pub fn encode(&self) -> Bytes {
        let mut buf = BytesMut::new();

        // Encode header
        buf.extend_from_slice(&self.header.encode());

        // Encode format
        buf.put_u32(self.format.width);
        buf.put_u32(self.format.height);
        buf.put_u32(self.format.fps_num);
        buf.put_u32(self.format.fps_den);
        buf.put_u8(u8::from(self.format.progressive));
        buf.put_u32(self.stride);

        // Encode data
        buf.extend_from_slice(&self.data);

        buf.freeze()
    }

    /// Decode a video frame
    pub fn decode(mut buf: Cursor<&[u8]>) -> Result<Self> {
        let header = NdiFrameHeader::decode(&mut buf)?;

        if header.frame_type != NdiFrameType::Video {
            return Err(NdiError::Protocol(format!(
                "Expected video frame, got {:?}",
                header.frame_type
            )));
        }

        // Decode format
        let width = buf.get_u32();
        let height = buf.get_u32();
        let fps_num = buf.get_u32();
        let fps_den = buf.get_u32();
        let progressive = buf.get_u8() != 0;
        let stride = buf.get_u32();

        let mut format = VideoFormat::new(width, height, fps_num, fps_den);
        format.progressive = progressive;

        // Decode data
        let remaining = buf.remaining();
        let mut data = vec![0u8; remaining];
        if remaining > 0 {
            Buf::copy_to_slice(&mut buf, &mut data);
        }

        Ok(Self {
            header,
            format,
            data: Bytes::from(data),
            stride,
        })
    }
}

/// NDI audio frame
#[derive(Debug, Clone)]
pub struct NdiAudioFrame {
    /// Frame header
    pub header: NdiFrameHeader,

    /// Audio format
    pub format: AudioFormat,

    /// Audio data
    pub data: Bytes,

    /// Number of samples
    pub num_samples: u32,
}

impl NdiAudioFrame {
    /// Create a new audio frame
    pub fn new(
        sequence: u32,
        timestamp: i64,
        format: AudioFormat,
        data: Bytes,
        num_samples: u32,
    ) -> Self {
        let header =
            NdiFrameHeader::new(NdiFrameType::Audio, sequence, timestamp, data.len() as u32);

        Self {
            header,
            format,
            data,
            num_samples,
        }
    }

    /// Encode the audio frame
    pub fn encode(&self) -> Bytes {
        let mut buf = BytesMut::new();

        // Encode header
        buf.extend_from_slice(&self.header.encode());

        // Encode format
        buf.put_u32(self.format.sample_rate);
        buf.put_u16(self.format.channels);
        buf.put_u16(self.format.bits_per_sample);
        buf.put_u32(self.num_samples);

        // Encode data
        buf.extend_from_slice(&self.data);

        buf.freeze()
    }

    /// Decode an audio frame
    pub fn decode(mut buf: Cursor<&[u8]>) -> Result<Self> {
        let header = NdiFrameHeader::decode(&mut buf)?;

        if header.frame_type != NdiFrameType::Audio {
            return Err(NdiError::Protocol(format!(
                "Expected audio frame, got {:?}",
                header.frame_type
            )));
        }

        // Decode format
        let sample_rate = buf.get_u32();
        let channels = buf.get_u16();
        let bits_per_sample = buf.get_u16();
        let num_samples = buf.get_u32();

        let format = AudioFormat::new(sample_rate, channels, bits_per_sample);

        // Decode data
        let remaining = buf.remaining();
        let mut data = vec![0u8; remaining];
        if remaining > 0 {
            Buf::copy_to_slice(&mut buf, &mut data);
        }

        Ok(Self {
            header,
            format,
            data: Bytes::from(data),
            num_samples,
        })
    }
}

/// NDI metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NdiMetadata {
    /// Metadata key-value pairs
    pub data: HashMap<String, String>,
}

impl NdiMetadata {
    /// Create a new empty metadata
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }

    /// Create metadata with initial data
    pub fn with_data(data: HashMap<String, String>) -> Self {
        Self { data }
    }

    /// Insert a key-value pair
    pub fn insert(&mut self, key: String, value: String) {
        self.data.insert(key, value);
    }

    /// Get a value by key
    pub fn get(&self, key: &str) -> Option<&String> {
        self.data.get(key)
    }

    /// Encode metadata to JSON bytes
    pub fn encode(&self, sequence: u32, timestamp: i64) -> Result<Bytes> {
        let json = serde_json::to_vec(&self.data)
            .map_err(|e| NdiError::Protocol(format!("Failed to encode metadata: {}", e)))?;

        let header = NdiFrameHeader::new(
            NdiFrameType::Metadata,
            sequence,
            timestamp,
            json.len() as u32,
        );

        let mut buf = BytesMut::new();
        buf.extend_from_slice(&header.encode());
        buf.extend_from_slice(&json);

        Ok(buf.freeze())
    }

    /// Decode metadata from bytes
    pub fn decode(mut buf: Cursor<&[u8]>) -> Result<(Self, NdiFrameHeader)> {
        let header = NdiFrameHeader::decode(&mut buf)?;

        if header.frame_type != NdiFrameType::Metadata {
            return Err(NdiError::Protocol(format!(
                "Expected metadata frame, got {:?}",
                header.frame_type
            )));
        }

        let remaining = buf.remaining();
        let mut json_data = vec![0u8; remaining];
        if remaining > 0 {
            Buf::copy_to_slice(&mut buf, &mut json_data);
        }

        let data: HashMap<String, String> = serde_json::from_slice(&json_data)
            .map_err(|e| NdiError::Protocol(format!("Failed to decode metadata: {}", e)))?;

        Ok((Self { data }, header))
    }
}

impl Default for NdiMetadata {
    fn default() -> Self {
        Self::new()
    }
}

/// NDI frame (enum of all frame types)
#[derive(Debug, Clone)]
pub enum NdiFrame {
    /// Video frame
    Video(NdiVideoFrame),
    /// Audio frame
    Audio(NdiAudioFrame),
    /// Metadata
    Metadata(NdiMetadata, NdiFrameHeader),
    /// Heartbeat
    Heartbeat(NdiFrameHeader),
}

impl NdiFrame {
    /// Get the frame type
    pub fn frame_type(&self) -> NdiFrameType {
        match self {
            Self::Video(_) => NdiFrameType::Video,
            Self::Audio(_) => NdiFrameType::Audio,
            Self::Metadata(_, _) => NdiFrameType::Metadata,
            Self::Heartbeat(_) => NdiFrameType::Heartbeat,
        }
    }

    /// Get the frame timestamp
    pub fn timestamp(&self) -> i64 {
        match self {
            Self::Video(f) => f.header.timestamp,
            Self::Audio(f) => f.header.timestamp,
            Self::Metadata(_, h) => h.timestamp,
            Self::Heartbeat(h) => h.timestamp,
        }
    }

    /// Get the frame sequence number
    pub fn sequence(&self) -> u32 {
        match self {
            Self::Video(f) => f.header.sequence,
            Self::Audio(f) => f.header.sequence,
            Self::Metadata(_, h) => h.sequence,
            Self::Heartbeat(h) => h.sequence,
        }
    }

    /// Encode the frame to bytes
    pub fn encode(&self) -> Result<Bytes> {
        match self {
            Self::Video(f) => Ok(f.encode()),
            Self::Audio(f) => Ok(f.encode()),
            Self::Metadata(m, h) => m.encode(h.sequence, h.timestamp),
            Self::Heartbeat(h) => Ok(h.encode()),
        }
    }

    /// Decode a frame from bytes
    pub fn decode(data: &[u8]) -> Result<Self> {
        let mut cursor = Cursor::new(data);
        let header = NdiFrameHeader::decode(&mut cursor)?;

        // Reset cursor
        let cursor = Cursor::new(data);

        match header.frame_type {
            NdiFrameType::Video => Ok(Self::Video(NdiVideoFrame::decode(cursor)?)),
            NdiFrameType::Audio => Ok(Self::Audio(NdiAudioFrame::decode(cursor)?)),
            NdiFrameType::Metadata => {
                let (metadata, header) = NdiMetadata::decode(cursor)?;
                Ok(Self::Metadata(metadata, header))
            }
            NdiFrameType::Heartbeat => Ok(Self::Heartbeat(header)),
            _ => Err(NdiError::Protocol(format!(
                "Unsupported frame type: {:?}",
                header.frame_type
            ))),
        }
    }
}

/// NDI connection manager
pub struct NdiConnection {
    /// TCP stream
    stream: TcpStream,

    /// Sequence number for sent frames
    sequence: u32,

    /// Read buffer
    read_buffer: BytesMut,
}

impl NdiConnection {
    /// Create a new NDI connection
    pub fn new(stream: TcpStream) -> Self {
        Self {
            stream,
            sequence: 0,
            read_buffer: BytesMut::with_capacity(65536),
        }
    }

    /// Send a frame
    pub async fn send_frame(&mut self, frame: &NdiFrame) -> Result<()> {
        let data = frame.encode()?;
        trace!(
            "Sending frame: {:?}, size: {}",
            frame.frame_type(),
            data.len()
        );

        self.stream
            .write_all(&data)
            .await
            .map_err(|e| NdiError::Network(e))?;

        self.stream
            .flush()
            .await
            .map_err(|e| NdiError::Network(e))?;

        Ok(())
    }

    /// Receive a frame
    pub async fn receive_frame(&mut self) -> Result<NdiFrame> {
        loop {
            // Try to parse a frame from the buffer
            if self.read_buffer.len() >= NdiFrameHeader::size() {
                let mut cursor = Cursor::new(&self.read_buffer[..]);
                match NdiFrameHeader::decode(&mut cursor) {
                    Ok(header) => {
                        let total_size = NdiFrameHeader::size() + header.payload_size as usize;

                        if self.read_buffer.len() >= total_size {
                            // We have a complete frame
                            let frame_data = self.read_buffer.split_to(total_size);
                            let frame = NdiFrame::decode(&frame_data)?;
                            trace!(
                                "Received frame: {:?}, size: {}",
                                frame.frame_type(),
                                total_size
                            );
                            return Ok(frame);
                        }
                    }
                    Err(_) => {
                        // Invalid header, skip one byte and try again
                        self.read_buffer.advance(1);
                        continue;
                    }
                }
            }

            // Need more data
            let mut temp_buf = vec![0u8; 8192];
            let n = self
                .stream
                .read(&mut temp_buf)
                .await
                .map_err(|e| NdiError::Network(e))?;

            if n == 0 {
                return Err(NdiError::ConnectionClosed);
            }

            self.read_buffer.extend_from_slice(&temp_buf[..n]);
        }
    }

    /// Receive a frame with timeout
    pub async fn receive_frame_timeout(&mut self, timeout: Duration) -> Result<NdiFrame> {
        tokio::time::timeout(timeout, self.receive_frame())
            .await
            .map_err(|_| NdiError::Timeout)?
    }

    /// Send a video frame
    pub async fn send_video_frame(
        &mut self,
        format: VideoFormat,
        data: Bytes,
        stride: u32,
        timestamp: i64,
    ) -> Result<()> {
        let frame = NdiVideoFrame::new(self.sequence, timestamp, format, data, stride);
        self.sequence = self.sequence.wrapping_add(1);
        self.send_frame(&NdiFrame::Video(frame)).await
    }

    /// Send an audio frame
    pub async fn send_audio_frame(
        &mut self,
        format: AudioFormat,
        data: Bytes,
        num_samples: u32,
        timestamp: i64,
    ) -> Result<()> {
        let frame = NdiAudioFrame::new(self.sequence, timestamp, format, data, num_samples);
        self.sequence = self.sequence.wrapping_add(1);
        self.send_frame(&NdiFrame::Audio(frame)).await
    }

    /// Send metadata
    pub async fn send_metadata(&mut self, metadata: NdiMetadata, timestamp: i64) -> Result<()> {
        let header = NdiFrameHeader::new(
            NdiFrameType::Metadata,
            self.sequence,
            timestamp,
            0, // Will be set by encode
        );
        self.sequence = self.sequence.wrapping_add(1);
        self.send_frame(&NdiFrame::Metadata(metadata, header)).await
    }

    /// Send a heartbeat
    pub async fn send_heartbeat(&mut self) -> Result<()> {
        let timestamp = current_timestamp();
        let header = NdiFrameHeader::new(NdiFrameType::Heartbeat, self.sequence, timestamp, 0);
        self.sequence = self.sequence.wrapping_add(1);
        self.send_frame(&NdiFrame::Heartbeat(header)).await
    }

    /// Get the local address of this connection
    pub fn local_addr(&self) -> std::io::Result<std::net::SocketAddr> {
        self.stream.local_addr()
    }

    /// Get the peer address of this connection
    pub fn peer_addr(&self) -> std::io::Result<std::net::SocketAddr> {
        self.stream.peer_addr()
    }

    /// Shutdown the connection
    pub async fn shutdown(&mut self) -> Result<()> {
        self.stream
            .shutdown()
            .await
            .map_err(|e| NdiError::Network(e))
    }
}

/// Get the current timestamp in microseconds since Unix epoch
pub fn current_timestamp() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as i64
}

/// Convert a timestamp to a `Duration` since Unix epoch
pub fn timestamp_to_duration(timestamp: i64) -> Duration {
    Duration::from_micros(timestamp as u64)
}

/// Frame synchronizer for audio/video sync
pub struct FrameSynchronizer {
    /// Video frames waiting for audio
    video_queue: Vec<NdiVideoFrame>,

    /// Audio frames waiting for video
    audio_queue: Vec<NdiAudioFrame>,

    /// Maximum sync offset in microseconds
    max_offset: i64,

    /// Last synchronized timestamp
    last_sync_timestamp: i64,
}

impl FrameSynchronizer {
    /// Create a new frame synchronizer
    pub fn new(max_offset_ms: i64) -> Self {
        Self {
            video_queue: Vec::new(),
            audio_queue: Vec::new(),
            max_offset: max_offset_ms * 1000,
            last_sync_timestamp: 0,
        }
    }

    /// Add a video frame
    pub fn add_video(&mut self, frame: NdiVideoFrame) {
        self.video_queue.push(frame);
        self.video_queue.sort_by_key(|f| f.header.timestamp);
    }

    /// Add an audio frame
    pub fn add_audio(&mut self, frame: NdiAudioFrame) {
        self.audio_queue.push(frame);
        self.audio_queue.sort_by_key(|f| f.header.timestamp);
    }

    /// Try to get synchronized frames
    pub fn get_synchronized(&mut self) -> Option<(Option<NdiVideoFrame>, Vec<NdiAudioFrame>)> {
        if self.video_queue.is_empty() {
            return None;
        }

        let video = self.video_queue.remove(0);
        let video_ts = video.header.timestamp;

        // Find audio frames that match this video frame
        let mut audio_frames = Vec::new();
        self.audio_queue.retain(|audio| {
            let diff = (audio.header.timestamp - video_ts).abs();
            if diff <= self.max_offset {
                audio_frames.push(audio.clone());
                false
            } else {
                true
            }
        });

        self.last_sync_timestamp = video_ts;
        Some((Some(video), audio_frames))
    }

    /// Clear old frames that are too far behind
    pub fn clear_old_frames(&mut self, current_timestamp: i64) {
        let threshold = current_timestamp - self.max_offset * 10;

        self.video_queue.retain(|f| f.header.timestamp > threshold);
        self.audio_queue.retain(|f| f.header.timestamp > threshold);
    }

    /// Get the number of queued video frames
    pub fn video_queue_len(&self) -> usize {
        self.video_queue.len()
    }

    /// Get the number of queued audio frames
    pub fn audio_queue_len(&self) -> usize {
        self.audio_queue.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_header_encode_decode() {
        let header = NdiFrameHeader::new(NdiFrameType::Video, 123, 456789, 1024);
        let encoded = header.encode();
        let mut cursor = Cursor::new(&encoded[..]);
        let decoded = NdiFrameHeader::decode(&mut cursor).expect("unexpected None/Err");

        assert_eq!(decoded.frame_type, NdiFrameType::Video);
        assert_eq!(decoded.sequence, 123);
        assert_eq!(decoded.timestamp, 456789);
        assert_eq!(decoded.payload_size, 1024);
    }

    #[test]
    fn test_metadata_encode_decode() {
        let mut metadata = NdiMetadata::new();
        metadata.insert("key1".to_string(), "value1".to_string());
        metadata.insert("key2".to_string(), "value2".to_string());

        let encoded = metadata.encode(1, 1000).expect("unexpected None/Err");
        let cursor = Cursor::new(&encoded[..]);
        let (decoded, header) = NdiMetadata::decode(cursor).expect("unexpected None/Err");

        assert_eq!(decoded.get("key1"), Some(&"value1".to_string()));
        assert_eq!(decoded.get("key2"), Some(&"value2".to_string()));
        assert_eq!(header.sequence, 1);
        assert_eq!(header.timestamp, 1000);
    }

    #[test]
    fn test_frame_synchronizer() {
        let mut sync = FrameSynchronizer::new(100);

        let video = NdiVideoFrame::new(1, 1000, VideoFormat::full_hd_30p(), Bytes::new(), 1920 * 2);
        let audio = NdiAudioFrame::new(1, 1050, AudioFormat::stereo_48k(), Bytes::new(), 480);

        sync.add_video(video);
        sync.add_audio(audio);

        let result = sync.get_synchronized();
        assert!(result.is_some());

        let (video, audio) = result.expect("expected successful result");
        assert!(video.is_some());
        assert_eq!(audio.len(), 1);
    }

    #[test]
    fn test_current_timestamp() {
        let ts1 = current_timestamp();
        std::thread::sleep(Duration::from_millis(10));
        let ts2 = current_timestamp();
        assert!(ts2 > ts1);
        assert!(ts2 - ts1 >= 10000); // At least 10ms = 10000us
    }
}
