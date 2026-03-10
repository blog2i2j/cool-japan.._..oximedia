//! FLAC native container demuxer.
//!
//! Parses the FLAC format as specified at <https://xiph.org/flac/format.html>.
//!
//! # Overview
//!
//! FLAC (Free Lossless Audio Codec) files consist of:
//! 1. A 4-byte magic number "fLaC"
//! 2. One or more metadata blocks (STREAMINFO is required and must be first)
//! 3. Audio frames containing compressed audio data
//!
//! # Supported Metadata Blocks
//!
//! - `STREAMINFO` - Required stream parameters (sample rate, channels, etc.)
//! - `VORBIS_COMMENT` - Tags (artist, title, etc.)
//! - `SEEKTABLE` - Sample-accurate seeking support
//! - `PADDING`, `APPLICATION`, `CUESHEET`, `PICTURE` - Parsed but not used
//!
//! # Example
//!
//! ```ignore
//! use oximedia_container::demux::FlacDemuxer;
//! use oximedia_io::FileSource;
//!
//! let source = FileSource::open("audio.flac").await?;
//! let mut demuxer = FlacDemuxer::new(source);
//!
//! let probe = demuxer.probe().await?;
//! println!("Detected: {:?}", probe.format);
//!
//! while let Ok(packet) = demuxer.read_packet().await {
//!     println!("Frame: {} bytes, {} samples",
//!              packet.size(), packet.duration().unwrap_or(0));
//! }
//! ```

mod frame;
pub mod metadata;

pub use frame::{ChannelAssignment, FrameHeader};
pub use metadata::{BlockType, MetadataBlock, StreamInfo as FlacStreamInfo, VorbisComment};

use async_trait::async_trait;
use bytes::Bytes;
use oximedia_core::{CodecId, OxiError, OxiResult, Rational, Timestamp};

use crate::demux::Demuxer;
use crate::{CodecParams, ContainerFormat, Metadata, Packet, PacketFlags, ProbeResult, StreamInfo};

/// FLAC magic number (`"fLaC"`).
pub const FLAC_MAGIC: &[u8; 4] = b"fLaC";

/// FLAC demuxer for parsing native FLAC container files.
///
/// Extracts compressed audio frames from FLAC files along with metadata
/// such as stream parameters, seek tables, and Vorbis comments.
///
/// # Example
///
/// ```ignore
/// use oximedia_container::demux::FlacDemuxer;
/// use oximedia_io::MemorySource;
///
/// let data = std::fs::read("audio.flac")?;
/// let source = MemorySource::new(data);
/// let mut demuxer = FlacDemuxer::new(source);
///
/// demuxer.probe().await?;
///
/// for stream in demuxer.streams() {
///     println!("Sample rate: {:?}", stream.codec_params.sample_rate);
/// }
/// ```
pub struct FlacDemuxer<R> {
    /// The underlying media source.
    source: R,

    /// Read buffer for frame data.
    buffer: Vec<u8>,

    /// Parsed `STREAMINFO` metadata block.
    stream_info: Option<FlacStreamInfo>,

    /// Parsed Vorbis comments (tags).
    vorbis_comments: Option<VorbisComment>,

    /// Stream information for the audio track.
    streams: Vec<StreamInfo>,

    /// Seek table for sample-accurate seeking.
    seek_table: Vec<SeekPoint>,

    /// Byte offset where metadata ends and audio frames begin.
    metadata_end: u64,

    /// Current position in the file (for tracking without seeking).
    position: u64,

    /// Current sample number.
    current_sample: u64,

    /// Whether end of file has been reached.
    eof: bool,

    /// Whether probe has been called.
    probed: bool,
}

/// Seek point for sample-accurate seeking.
///
/// FLAC files may contain a seek table with pre-computed seek points
/// for efficient random access to specific samples.
#[derive(Clone, Debug)]
pub struct SeekPoint {
    /// Sample number of the first sample in the target frame.
    pub sample: u64,

    /// Byte offset from the first audio frame.
    pub offset: u64,

    /// Number of samples in the target frame.
    pub samples: u16,
}

impl<R> FlacDemuxer<R> {
    /// Creates a new FLAC demuxer with the given source.
    ///
    /// After creation, call [`probe`](Demuxer::probe) to parse the file
    /// header and metadata before reading packets.
    #[must_use]
    pub fn new(source: R) -> Self {
        Self {
            source,
            buffer: Vec::with_capacity(16384),
            stream_info: None,
            vorbis_comments: None,
            streams: Vec::new(),
            seek_table: Vec::new(),
            metadata_end: 0,
            position: 0,
            current_sample: 0,
            eof: false,
            probed: false,
        }
    }

    /// Returns a reference to the underlying source.
    #[must_use]
    pub const fn source(&self) -> &R {
        &self.source
    }

    /// Returns a mutable reference to the underlying source.
    pub fn source_mut(&mut self) -> &mut R {
        &mut self.source
    }

    /// Consumes the demuxer and returns the underlying source.
    #[must_use]
    #[allow(dead_code)]
    pub fn into_source(self) -> R {
        self.source
    }

    /// Returns the stream info if available.
    #[must_use]
    pub fn stream_info(&self) -> Option<&FlacStreamInfo> {
        self.stream_info.as_ref()
    }

    /// Returns the Vorbis comments if available.
    #[must_use]
    pub fn vorbis_comments(&self) -> Option<&VorbisComment> {
        self.vorbis_comments.as_ref()
    }

    /// Returns the seek table.
    #[must_use]
    pub fn seek_table(&self) -> &[SeekPoint] {
        &self.seek_table
    }

    /// Returns the duration in seconds if known.
    #[must_use]
    pub fn duration_seconds(&self) -> Option<f64> {
        self.stream_info
            .as_ref()
            .and_then(FlacStreamInfo::duration_seconds)
    }

    /// Builds stream information from the parsed `STREAMINFO` block.
    fn build_stream_info(&mut self) {
        let Some(info) = &self.stream_info else {
            return;
        };

        // FLAC uses sample rate as the timebase denominator
        let timebase = Rational::new(1, i64::from(info.sample_rate));

        let mut stream = StreamInfo::new(0, CodecId::Flac, timebase);

        // Set duration in samples
        // FLAC total_samples is 36 bits max, safely fits in i64
        #[allow(clippy::cast_possible_wrap)]
        if info.total_samples > 0 {
            stream.duration = Some(info.total_samples as i64);
        }

        // Set codec parameters
        stream.codec_params = CodecParams::audio(info.sample_rate, info.channels);

        // Set metadata from Vorbis comments
        if let Some(comments) = &self.vorbis_comments {
            let mut metadata = Metadata::new();

            if let Some(title) = comments.get("TITLE") {
                metadata = metadata.with_title(title);
            }
            if let Some(artist) = comments.get("ARTIST") {
                metadata = metadata.with_artist(artist);
            }
            if let Some(album) = comments.get("ALBUM") {
                metadata = metadata.with_album(album);
            }

            // Add other tags as entries
            for (key, value) in &comments.comments {
                let key_upper = key.to_uppercase();
                if key_upper != "TITLE" && key_upper != "ARTIST" && key_upper != "ALBUM" {
                    metadata = metadata.with_entry(key.clone(), value.clone());
                }
            }

            stream.metadata = metadata;
        }

        self.streams.push(stream);
    }
}

#[async_trait]
impl<R: oximedia_io::MediaSource> Demuxer for FlacDemuxer<R> {
    #[allow(clippy::too_many_lines)]
    async fn probe(&mut self) -> OxiResult<ProbeResult> {
        if self.probed {
            return Ok(ProbeResult::new(ContainerFormat::Flac, 1.0));
        }

        // Read and verify magic number
        let mut magic = [0u8; 4];
        let n = self.source.read(&mut magic).await?;
        if n < 4 {
            return Err(OxiError::UnexpectedEof);
        }
        self.position = 4;

        if &magic != FLAC_MAGIC {
            return Err(OxiError::Parse {
                offset: 0,
                message: "Invalid FLAC magic number".into(),
            });
        }

        // Parse all metadata blocks
        loop {
            // Read block header (4 bytes)
            let mut header = [0u8; 4];
            let n = self.source.read(&mut header).await?;
            if n < 4 {
                return Err(OxiError::UnexpectedEof);
            }

            let is_last = header[0] & 0x80 != 0;
            let block_type = BlockType::from(header[0]);
            let length = u32::from_be_bytes([0, header[1], header[2], header[3]]) as usize;

            self.position += 4;

            // Read block data
            if self.buffer.len() < length {
                self.buffer.resize(length, 0);
            }
            let block_data = &mut self.buffer[..length];
            let mut read = 0;
            while read < length {
                let n = self.source.read(&mut block_data[read..]).await?;
                if n == 0 {
                    return Err(OxiError::UnexpectedEof);
                }
                read += n;
            }
            self.position += length as u64;

            // Parse block based on type
            match block_type {
                BlockType::StreamInfo => {
                    self.stream_info = Some(FlacStreamInfo::parse(block_data)?);
                }
                BlockType::VorbisComment => {
                    self.vorbis_comments = Some(VorbisComment::parse(block_data)?);
                }
                BlockType::SeekTable => {
                    // Parse seek table (18 bytes per entry)
                    let entry_count = length / 18;
                    for i in 0..entry_count {
                        let offset = i * 18;
                        let sample = u64::from_be_bytes([
                            block_data[offset],
                            block_data[offset + 1],
                            block_data[offset + 2],
                            block_data[offset + 3],
                            block_data[offset + 4],
                            block_data[offset + 5],
                            block_data[offset + 6],
                            block_data[offset + 7],
                        ]);

                        // Skip placeholder entries (sample number == 0xFFFFFFFFFFFFFFFF)
                        if sample == u64::MAX {
                            continue;
                        }

                        let frame_offset = u64::from_be_bytes([
                            block_data[offset + 8],
                            block_data[offset + 9],
                            block_data[offset + 10],
                            block_data[offset + 11],
                            block_data[offset + 12],
                            block_data[offset + 13],
                            block_data[offset + 14],
                            block_data[offset + 15],
                        ]);

                        let samples =
                            u16::from_be_bytes([block_data[offset + 16], block_data[offset + 17]]);

                        self.seek_table.push(SeekPoint {
                            sample,
                            offset: frame_offset,
                            samples,
                        });
                    }
                }
                BlockType::Padding
                | BlockType::Application
                | BlockType::CueSheet
                | BlockType::Picture
                | BlockType::Reserved => {
                    // Skip these blocks
                }
            }

            if is_last {
                break;
            }
        }

        // Verify we got STREAMINFO
        if self.stream_info.is_none() {
            return Err(OxiError::Parse {
                offset: 4,
                message: "Missing required STREAMINFO metadata block".into(),
            });
        }

        self.metadata_end = self.position;
        self.build_stream_info();
        self.probed = true;

        Ok(ProbeResult::new(ContainerFormat::Flac, 1.0))
    }

    async fn read_packet(&mut self) -> OxiResult<Packet> {
        if !self.probed {
            return Err(OxiError::InvalidData(
                "Must call probe() before read_packet()".into(),
            ));
        }

        if self.eof {
            return Err(OxiError::Eof);
        }

        let stream_info = self
            .stream_info
            .as_ref()
            .ok_or_else(|| OxiError::InvalidData("Missing STREAMINFO".into()))?;

        // Read potential frame header (we need at least 4 bytes to check sync)
        let mut header_buf = [0u8; 16];
        let n = self.source.read(&mut header_buf).await?;
        if n == 0 {
            self.eof = true;
            return Err(OxiError::Eof);
        }
        if n < 2 {
            self.eof = true;
            return Err(OxiError::Eof);
        }

        // Check for frame sync (0xFFF8 or 0xFFF9)
        let sync = (u16::from(header_buf[0]) << 8) | u16::from(header_buf[1]);
        if sync & 0xFFF8 != 0xFFF8 {
            // Try to find sync
            return Err(OxiError::Parse {
                offset: self.position,
                message: format!("Invalid frame sync: 0x{sync:04X}"),
            });
        }

        // Parse frame header to get block size
        let (frame_header, _header_len) = FrameHeader::parse(&header_buf[..n])?;
        let block_size = frame_header.block_size;

        // Estimate frame size based on stream info
        // Maximum frame size = block_size * channels * bits_per_sample / 8 + header + footer
        let max_frame_size = if stream_info.max_frame_size > 0 {
            stream_info.max_frame_size as usize
        } else {
            // Conservative estimate
            (block_size as usize) * (stream_info.channels as usize) * 4 + 256
        };

        // Read enough data for the frame
        if self.buffer.len() < max_frame_size + 16 {
            self.buffer.resize(max_frame_size + 16, 0);
        }

        // Copy header bytes we already read
        self.buffer[..n].copy_from_slice(&header_buf[..n]);

        // Read more data to complete the frame
        let mut total_read = n;
        if total_read < max_frame_size {
            let remaining = max_frame_size - total_read;
            let additional = self
                .source
                .read(&mut self.buffer[total_read..total_read + remaining])
                .await?;
            total_read += additional;
        }

        // Find frame end by looking for next sync or EOF
        // For simplicity, we use the header's block size to estimate
        // In a full implementation, we'd parse subframes and find CRC-16

        // Create packet with the frame data
        let frame_data =
            if stream_info.max_frame_size > 0 && total_read > stream_info.max_frame_size as usize {
                Bytes::copy_from_slice(&self.buffer[..stream_info.max_frame_size as usize])
            } else {
                Bytes::copy_from_slice(&self.buffer[..total_read])
            };

        // Create timestamp
        let timebase = Rational::new(1, i64::from(stream_info.sample_rate));
        // FLAC samples are 36 bits max, safely fits in i64
        #[allow(clippy::cast_possible_wrap)]
        let pts = self.current_sample as i64;
        let mut timestamp = Timestamp::new(pts, timebase);
        timestamp.duration = Some(i64::from(block_size));

        self.current_sample += u64::from(block_size);
        self.position += total_read as u64;

        // All FLAC frames are keyframes
        let packet = Packet::new(0, frame_data, timestamp, PacketFlags::KEYFRAME);

        Ok(packet)
    }

    fn streams(&self) -> &[StreamInfo] {
        &self.streams
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flac_magic() {
        assert_eq!(FLAC_MAGIC, b"fLaC");
    }

    #[test]
    fn test_seek_point() {
        let point = SeekPoint {
            sample: 44100,
            offset: 1024,
            samples: 4096,
        };
        assert_eq!(point.sample, 44100);
        assert_eq!(point.offset, 1024);
        assert_eq!(point.samples, 4096);
    }

    struct MockSource {
        data: Vec<u8>,
        pos: usize,
    }

    impl MockSource {
        fn new(data: Vec<u8>) -> Self {
            Self { data, pos: 0 }
        }
    }

    #[async_trait]
    impl oximedia_io::MediaSource for MockSource {
        async fn read(&mut self, buf: &mut [u8]) -> OxiResult<usize> {
            let remaining = self.data.len() - self.pos;
            let to_read = buf.len().min(remaining);
            buf[..to_read].copy_from_slice(&self.data[self.pos..self.pos + to_read]);
            self.pos += to_read;
            Ok(to_read)
        }

        async fn write_all(&mut self, _buf: &[u8]) -> OxiResult<()> {
            Err(OxiError::unsupported("MockSource does not support writing"))
        }

        async fn seek(&mut self, pos: std::io::SeekFrom) -> OxiResult<u64> {
            match pos {
                std::io::SeekFrom::Start(p) => {
                    self.pos = p as usize;
                }
                std::io::SeekFrom::Current(p) => {
                    self.pos = (self.pos as i64 + p) as usize;
                }
                std::io::SeekFrom::End(p) => {
                    self.pos = (self.data.len() as i64 + p) as usize;
                }
            }
            Ok(self.pos as u64)
        }

        fn len(&self) -> Option<u64> {
            Some(self.data.len() as u64)
        }

        fn is_seekable(&self) -> bool {
            true
        }

        fn position(&self) -> u64 {
            self.pos as u64
        }
    }

    #[tokio::test]
    async fn test_flac_demuxer_new() {
        let source = MockSource::new(Vec::new());
        let demuxer = FlacDemuxer::new(source);
        assert!(demuxer.streams().is_empty());
        assert!(demuxer.stream_info().is_none());
    }

    #[tokio::test]
    async fn test_flac_demuxer_invalid_magic() {
        let source = MockSource::new(b"RIFF".to_vec());
        let mut demuxer = FlacDemuxer::new(source);
        let result = demuxer.probe().await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_flac_demuxer_too_short() {
        let source = MockSource::new(b"fL".to_vec());
        let mut demuxer = FlacDemuxer::new(source);
        let result = demuxer.probe().await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_flac_demuxer_valid_header() {
        // Construct a minimal valid FLAC file with STREAMINFO
        let mut data = Vec::new();

        // Magic
        data.extend_from_slice(b"fLaC");

        // STREAMINFO block header (is_last=1, type=0, length=34)
        data.push(0x80); // is_last | type
        data.push(0x00); // length high byte
        data.push(0x00); // length mid byte
        data.push(0x22); // length low byte (34)

        // STREAMINFO data (34 bytes)
        // min_block_size = 4096
        data.push(0x10);
        data.push(0x00);
        // max_block_size = 4096
        data.push(0x10);
        data.push(0x00);
        // min_frame_size = 0 (3 bytes)
        data.push(0x00);
        data.push(0x00);
        data.push(0x00);
        // max_frame_size = 0 (3 bytes)
        data.push(0x00);
        data.push(0x00);
        data.push(0x00);
        // sample_rate = 44100, channels = 2, bits = 16, total_samples = 0
        // Bits 80-99: sample rate (20 bits) = 44100 = 0xAC44
        // Bits 100-102: channels - 1 (3 bits) = 1
        // Bits 103-107: bits per sample - 1 (5 bits) = 15
        // 44100 in 20 bits: 0x0AC44
        // 0x0A, 0xC4, 0x42
        data.push(0x0A);
        data.push(0xC4);
        data.push(0x42); // sample_rate tail | (channels-1) << 1
        data.push(0xF0); // (bits-1) << 4 | total_samples high nibble
                         // total_samples (36 bits, rest is 0)
        data.push(0x00);
        data.push(0x00);
        data.push(0x00);
        data.push(0x00);
        // MD5 (16 bytes)
        data.extend_from_slice(&[0u8; 16]);

        let source = MockSource::new(data);
        let mut demuxer = FlacDemuxer::new(source);
        let result = demuxer.probe().await;

        assert!(result.is_ok());
        let probe = result.expect("operation should succeed");
        assert_eq!(probe.format, ContainerFormat::Flac);
        assert!((probe.confidence - 1.0).abs() < f32::EPSILON);

        assert!(demuxer.stream_info().is_some());
        let info = demuxer.stream_info().expect("operation should succeed");
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
        assert_eq!(info.bits_per_sample, 16);

        assert_eq!(demuxer.streams().len(), 1);
    }
}
