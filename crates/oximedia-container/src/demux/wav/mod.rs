//! WAV/RIFF container demuxer.
//!
//! Implements the RIFF WAVE format for uncompressed and compressed audio.
//!
//! # Supported Formats
//!
//! - PCM (uncompressed integer samples, 8/16/24/32-bit)
//! - IEEE Float (32-bit and 64-bit floating point)
//! - A-law and mu-law companded audio
//! - `WAVE_FORMAT_EXTENSIBLE` for multi-channel audio
//!
//! # Example
//!
//! ```ignore
//! use oximedia_container::demux::WavDemuxer;
//! use oximedia_io::FileSource;
//!
//! let source = FileSource::open("audio.wav").await?;
//! let mut demuxer = WavDemuxer::new(source);
//!
//! let probe = demuxer.probe().await?;
//! println!("Format: {:?}", probe.format);
//!
//! while let Ok(packet) = demuxer.read_packet().await {
//!     println!("Packet: {} bytes", packet.size());
//! }
//! ```

mod chunks;

pub use chunks::{FmtChunk, FmtExtension, RiffChunk, WavFormat};

use async_trait::async_trait;
use bytes::Bytes;
use oximedia_core::{CodecId, OxiError, OxiResult, Rational, Timestamp};
use std::io::SeekFrom;

use crate::demux::Demuxer;
use crate::{CodecParams, ContainerFormat, Metadata, Packet, PacketFlags, ProbeResult, StreamInfo};

/// RIFF magic bytes.
pub const RIFF_MAGIC: &[u8; 4] = b"RIFF";
/// WAVE form type identifier.
pub const WAVE_FORM: &[u8; 4] = b"WAVE";
/// RF64 magic bytes (for files > 4GB).
pub const RF64_MAGIC: &[u8; 4] = b"RF64";
/// fmt chunk identifier.
const FMT_CHUNK: &[u8; 4] = b"fmt ";
/// data chunk identifier.
const DATA_CHUNK: &[u8; 4] = b"data";

/// Default packet size in samples (for reading).
const DEFAULT_PACKET_SAMPLES: u64 = 4096;

/// WAV demuxer.
///
/// Parses WAV/RIFF container format and extracts PCM audio packets.
///
/// # Lifecycle
///
/// 1. Create with `new(source)`
/// 2. Call `probe()` to parse headers and detect format
/// 3. Query `streams()` for audio stream information
/// 4. Call `read_packet()` in a loop to get PCM data packets
pub struct WavDemuxer<R> {
    /// Media source.
    source: R,
    /// Internal read buffer.
    buffer: Vec<u8>,
    /// Parsed format chunk.
    fmt: Option<FmtChunk>,
    /// Stream information.
    streams: Vec<StreamInfo>,

    /// Byte offset where data chunk starts.
    data_start: u64,
    /// Size of data chunk in bytes.
    data_size: u64,

    /// Current read position within data chunk.
    position: u64,
    /// Number of samples read so far.
    samples_read: u64,
    /// Whether end of file has been reached.
    eof: bool,
}

impl<R> WavDemuxer<R> {
    /// Creates a new WAV demuxer.
    ///
    /// After creation, call [`probe`](Demuxer::probe) to parse headers.
    #[must_use]
    pub fn new(source: R) -> Self {
        Self {
            source,
            buffer: Vec::with_capacity(8192),
            fmt: None,
            streams: Vec::new(),
            data_start: 0,
            data_size: 0,
            position: 0,
            samples_read: 0,
            eof: false,
        }
    }

    /// Returns a reference to the format chunk, if parsed.
    #[must_use]
    pub fn format_info(&self) -> Option<&FmtChunk> {
        self.fmt.as_ref()
    }

    /// Get total duration in seconds.
    ///
    /// Returns `None` if format info is not available or byte rate is zero.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn duration_seconds(&self) -> Option<f64> {
        let fmt = self.fmt.as_ref()?;
        if fmt.byte_rate == 0 {
            return None;
        }
        Some(self.data_size as f64 / f64::from(fmt.byte_rate))
    }

    /// Get total number of samples per channel.
    ///
    /// Returns `None` if format info is not available or block align is zero.
    #[must_use]
    pub fn total_samples(&self) -> Option<u64> {
        let fmt = self.fmt.as_ref()?;
        if fmt.block_align == 0 {
            return None;
        }
        Some(self.data_size / u64::from(fmt.block_align))
    }

    /// Returns the number of bytes remaining in the data chunk.
    #[must_use]
    pub fn bytes_remaining(&self) -> u64 {
        self.data_size.saturating_sub(self.position)
    }

    /// Returns true if all data has been read.
    #[must_use]
    pub fn is_eof(&self) -> bool {
        self.eof
    }
}

impl<R: oximedia_io::MediaSource> WavDemuxer<R> {
    /// Read exactly `n` bytes from source.
    async fn read_exact(&mut self, n: usize) -> OxiResult<Vec<u8>> {
        let mut buf = vec![0u8; n];
        let mut read = 0;
        while read < n {
            let chunk = self.source.read(&mut buf[read..]).await?;
            if chunk == 0 {
                return Err(OxiError::UnexpectedEof);
            }
            read += chunk;
        }
        Ok(buf)
    }

    /// Parse RIFF header and locate chunks.
    async fn parse_header(&mut self) -> OxiResult<()> {
        // Read RIFF header (12 bytes: "RIFF" + size + "WAVE")
        let header = self.read_exact(12).await?;

        // Validate RIFF magic
        if &header[0..4] != RIFF_MAGIC {
            return Err(OxiError::Parse {
                offset: 0,
                message: "Not a RIFF file".into(),
            });
        }

        // Validate WAVE form type
        if &header[8..12] != WAVE_FORM {
            return Err(OxiError::Parse {
                offset: 8,
                message: "Not a WAVE file".into(),
            });
        }

        // Parse chunks
        let mut offset = 12u64;
        let mut found_fmt = false;
        let mut found_data = false;

        while !found_data {
            // Read chunk header
            let chunk_header = match self.read_exact(RiffChunk::HEADER_SIZE).await {
                Ok(h) => h,
                Err(OxiError::UnexpectedEof) => break,
                Err(e) => return Err(e),
            };

            let chunk = RiffChunk::parse(&chunk_header)?;

            if chunk.is(FMT_CHUNK) {
                // Read fmt chunk data
                let fmt_data = self.read_exact(chunk.size as usize).await?;
                self.fmt = Some(FmtChunk::parse(&fmt_data)?);
                found_fmt = true;
            } else if chunk.is(DATA_CHUNK) {
                self.data_start = offset + RiffChunk::HEADER_SIZE as u64;
                self.data_size = u64::from(chunk.size);
                found_data = true;
            } else {
                // Skip unknown chunk
                let skip_size = u64::from(chunk.size);
                // Pad to even boundary
                let padded_size = if skip_size % 2 == 1 {
                    skip_size + 1
                } else {
                    skip_size
                };
                self.source
                    .seek(SeekFrom::Current(i64::try_from(padded_size).unwrap_or(0)))
                    .await?;
            }

            offset += RiffChunk::HEADER_SIZE as u64 + u64::from(chunk.size);
            // Pad to even boundary
            if chunk.size % 2 == 1 {
                offset += 1;
            }
        }

        if !found_fmt {
            return Err(OxiError::Parse {
                offset: 0,
                message: "Missing fmt chunk".into(),
            });
        }

        if !found_data {
            return Err(OxiError::Parse {
                offset: 0,
                message: "Missing data chunk".into(),
            });
        }

        Ok(())
    }

    /// Build stream info from parsed format.
    fn build_stream_info(&mut self) {
        let Some(fmt) = &self.fmt else {
            return;
        };

        // Create codec params
        let codec_params =
            CodecParams::audio(fmt.sample_rate, u8::try_from(fmt.channels).unwrap_or(2));

        // Calculate duration in samples
        let duration = if fmt.block_align > 0 {
            Some(i64::try_from(self.data_size / u64::from(fmt.block_align)).unwrap_or(0))
        } else {
            None
        };

        // Timebase is 1/sample_rate for sample-accurate timing
        let timebase = Rational::new(1, i64::from(fmt.sample_rate));

        let mut stream = StreamInfo::new(0, CodecId::Pcm, timebase);
        stream.duration = duration;
        stream.codec_params = codec_params;
        stream.metadata = Metadata::new();

        self.streams.push(stream);
    }
}

#[async_trait]
impl<R: oximedia_io::MediaSource> Demuxer for WavDemuxer<R> {
    async fn probe(&mut self) -> OxiResult<ProbeResult> {
        // Parse RIFF/WAVE header
        self.parse_header().await?;

        // Build stream info
        self.build_stream_info();

        // Seek to data start
        self.source.seek(SeekFrom::Start(self.data_start)).await?;
        self.position = 0;

        Ok(ProbeResult {
            format: ContainerFormat::Wav,
            confidence: 1.0,
        })
    }

    async fn read_packet(&mut self) -> OxiResult<Packet> {
        if self.eof {
            return Err(OxiError::Eof);
        }

        let Some(fmt) = &self.fmt else {
            return Err(OxiError::InvalidData(
                "Format not parsed. Call probe() first.".into(),
            ));
        };

        // Calculate bytes to read
        let samples_to_read = DEFAULT_PACKET_SAMPLES;
        let bytes_per_sample = u64::from(fmt.block_align);
        let bytes_to_read = samples_to_read * bytes_per_sample;

        // Limit to remaining data
        let remaining = self.data_size.saturating_sub(self.position);
        if remaining == 0 {
            self.eof = true;
            return Err(OxiError::Eof);
        }

        let actual_bytes = bytes_to_read.min(remaining);
        let actual_bytes_usize = usize::try_from(actual_bytes)
            .map_err(|_| OxiError::InvalidData("Invalid size".into()))?;

        // Read data
        self.buffer.resize(actual_bytes_usize, 0);
        let mut read = 0;
        while read < actual_bytes_usize {
            let chunk = self.source.read(&mut self.buffer[read..]).await?;
            if chunk == 0 {
                if read == 0 {
                    self.eof = true;
                    return Err(OxiError::Eof);
                }
                break;
            }
            read += chunk;
        }
        self.buffer.truncate(read);

        // Calculate timestamp
        let pts = i64::try_from(self.samples_read).unwrap_or(0);
        let samples_in_packet = if bytes_per_sample > 0 {
            read as u64 / bytes_per_sample
        } else {
            0
        };
        let duration = i64::try_from(samples_in_packet).ok();

        let timebase = Rational::new(1, i64::from(fmt.sample_rate));
        let timestamp = Timestamp::with_dts(pts, None, timebase, duration);

        // Update position
        self.position += read as u64;
        self.samples_read += samples_in_packet;

        // All PCM packets are keyframes
        Ok(Packet::new(
            0,
            Bytes::copy_from_slice(&self.buffer),
            timestamp,
            PacketFlags::KEYFRAME,
        ))
    }

    fn streams(&self) -> &[StreamInfo] {
        &self.streams
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wav_demuxer_new() {
        struct MockSource;
        let demuxer = WavDemuxer::new(MockSource);
        assert!(demuxer.fmt.is_none());
        assert!(demuxer.streams.is_empty());
        assert!(!demuxer.eof);
    }

    #[test]
    fn test_bytes_remaining() {
        struct MockSource;
        let mut demuxer = WavDemuxer::new(MockSource);
        demuxer.data_size = 1000;
        demuxer.position = 400;
        assert_eq!(demuxer.bytes_remaining(), 600);
    }

    #[test]
    fn test_duration_seconds() {
        struct MockSource;
        let mut demuxer = WavDemuxer::new(MockSource);
        demuxer.data_size = 176_400; // 1 second at 44.1kHz stereo 16-bit
        demuxer.fmt = Some(FmtChunk {
            format: WavFormat::Pcm,
            channels: 2,
            sample_rate: 44_100,
            byte_rate: 176_400,
            block_align: 4,
            bits_per_sample: 16,
            extension: None,
        });

        let duration = demuxer
            .duration_seconds()
            .expect("operation should succeed");
        assert!((duration - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_total_samples() {
        struct MockSource;
        let mut demuxer = WavDemuxer::new(MockSource);
        demuxer.data_size = 176_400;
        demuxer.fmt = Some(FmtChunk {
            format: WavFormat::Pcm,
            channels: 2,
            sample_rate: 44_100,
            byte_rate: 176_400,
            block_align: 4,
            bits_per_sample: 16,
            extension: None,
        });

        assert_eq!(demuxer.total_samples(), Some(44_100));
    }

    #[test]
    fn test_duration_zero_byte_rate() {
        struct MockSource;
        let mut demuxer = WavDemuxer::new(MockSource);
        demuxer.data_size = 1000;
        demuxer.fmt = Some(FmtChunk {
            format: WavFormat::Pcm,
            channels: 2,
            sample_rate: 44_100,
            byte_rate: 0, // Invalid
            block_align: 4,
            bits_per_sample: 16,
            extension: None,
        });

        assert!(demuxer.duration_seconds().is_none());
    }
}
