//! Demuxer for WASM.
//!
//! This module provides a synchronous demuxer interface for extracting
//! packets from media containers in the browser.
//!
//! Unlike the async demuxers in the main library, this uses synchronous
//! operations on in-memory data, which is suitable for the WASM
//! single-threaded environment.

use crate::container::{ContainerFormat, Packet, PacketFlags, StreamInfo};
use bytes::Bytes;
use wasm_bindgen::prelude::*;

use crate::io::ByteSource;
use crate::types::{WasmPacket, WasmStreamInfo};
use crate::utils::to_js_error;

/// WASM-compatible demuxer.
///
/// Provides synchronous demuxing of media containers from in-memory data.
///
/// # JavaScript Example
///
/// ```javascript
/// import * as oximedia from 'oximedia-wasm';
///
/// // Load file data
/// const response = await fetch('video.webm');
/// const arrayBuffer = await response.arrayBuffer();
/// const data = new Uint8Array(arrayBuffer);
///
/// // Create demuxer
/// const demuxer = new oximedia.WasmDemuxer(data);
///
/// // Probe format
/// const probe = demuxer.probe();
/// console.log('Format:', probe.format());
///
/// // Get streams
/// const streams = demuxer.streams();
/// console.log('Found', streams.length, 'streams');
///
/// for (const stream of streams) {
///     console.log(`Stream ${stream.index()}: ${stream.codec()}`);
/// }
///
/// // Read packets
/// let count = 0;
/// while (true) {
///     const packet = demuxer.read_packet();
///     if (!packet) break;
///     console.log(`Packet ${count++}: stream=${packet.stream_index()}, size=${packet.size()}`);
/// }
/// ```
#[wasm_bindgen]
pub struct WasmDemuxer {
    source: ByteSource,
    format: Option<ContainerFormat>,
    streams: Vec<WasmStreamInfo>,
    probed: bool,
}

#[wasm_bindgen]
impl WasmDemuxer {
    /// Creates a new demuxer from raw bytes.
    ///
    /// # Arguments
    ///
    /// * `data` - The complete media file data as a `Uint8Array`
    ///
    /// # Example
    ///
    /// ```javascript
    /// const data = new Uint8Array([...]);
    /// const demuxer = new oximedia.WasmDemuxer(data);
    /// ```
    #[wasm_bindgen(constructor)]
    #[must_use]
    pub fn new(data: &[u8]) -> Self {
        let bytes = Bytes::copy_from_slice(data);
        Self {
            source: ByteSource::new(bytes),
            format: None,
            streams: Vec::new(),
            probed: false,
        }
    }

    /// Probes the format and parses container headers.
    ///
    /// This must be called before reading packets. It detects the container
    /// format and extracts stream information.
    ///
    /// # Errors
    ///
    /// Throws a JavaScript exception if the format cannot be detected or
    /// headers are invalid.
    ///
    /// # Example
    ///
    /// ```javascript
    /// const probe = demuxer.probe();
    /// console.log('Format:', probe.format());
    /// console.log('Confidence:', probe.confidence());
    /// ```
    pub fn probe(&mut self) -> Result<crate::probe::WasmProbeResult, JsValue> {
        // Read first bytes for format detection
        let mut header = [0u8; 32];
        let n = self.source.read(&mut header).map_err(to_js_error)?;

        // Probe format
        let result = crate::container::probe_format(&header[..n]).map_err(to_js_error)?;

        self.format = Some(result.format);

        // Reset to beginning
        self.source
            .seek(std::io::SeekFrom::Start(0))
            .map_err(to_js_error)?;

        // Parse headers based on format
        self.parse_headers()?;
        self.probed = true;

        Ok(crate::probe::WasmProbeResult::new_internal(
            result.format,
            result.confidence,
        ))
    }

    /// Returns information about all streams.
    ///
    /// This is only valid after `probe()` has been called.
    ///
    /// # Example
    ///
    /// ```javascript
    /// const streams = demuxer.streams();
    /// for (const stream of streams) {
    ///     console.log(`Stream ${stream.index()}: ${stream.codec()}`);
    /// }
    /// ```
    #[must_use]
    pub fn streams(&self) -> Vec<WasmStreamInfo> {
        self.streams.clone()
    }

    /// Reads the next packet from the container.
    ///
    /// Returns `null` when there are no more packets (EOF).
    ///
    /// # Errors
    ///
    /// Throws a JavaScript exception for parse failures or I/O errors.
    ///
    /// # Example
    ///
    /// ```javascript
    /// while (true) {
    ///     const packet = demuxer.read_packet();
    ///     if (!packet) break;
    ///     console.log('Packet size:', packet.size());
    /// }
    /// ```
    pub fn read_packet(&mut self) -> Result<Option<WasmPacket>, JsValue> {
        if !self.probed {
            return Err(crate::utils::js_err(
                "Must call probe() before reading packets",
            ));
        }

        // Check if we're at EOF
        if self.source.is_eof() {
            return Ok(None);
        }

        // Read packet based on format
        match self.format {
            Some(ContainerFormat::Matroska) => self.read_matroska_packet(),
            Some(ContainerFormat::Ogg) => self.read_ogg_packet(),
            Some(ContainerFormat::Flac) => self.read_flac_packet(),
            Some(ContainerFormat::Wav) => self.read_wav_packet(),
            Some(ContainerFormat::Mp4) => self.read_mp4_packet(),
            None => Err(crate::utils::js_err("No format detected")),
        }
    }

    /// Returns the total size of the media data in bytes.
    #[must_use]
    pub fn size(&self) -> u64 {
        self.source.size()
    }

    /// Returns the current read position in bytes.
    #[must_use]
    pub fn position(&self) -> u64 {
        self.source.position()
    }

    /// Returns true if all packets have been read.
    #[must_use]
    pub fn is_eof(&self) -> bool {
        self.source.is_eof()
    }
}

// Private implementation methods
impl WasmDemuxer {
    /// Parse container headers to extract stream information.
    fn parse_headers(&mut self) -> Result<(), JsValue> {
        match self.format {
            Some(ContainerFormat::Matroska) => self.parse_matroska_headers(),
            Some(ContainerFormat::Ogg) => self.parse_ogg_headers(),
            Some(ContainerFormat::Flac) => self.parse_flac_headers(),
            Some(ContainerFormat::Wav) => self.parse_wav_headers(),
            Some(ContainerFormat::Mp4) => self.parse_mp4_headers(),
            None => Err(crate::utils::js_err("No format detected")),
        }
    }

    /// Parse Matroska/`WebM` headers.
    fn parse_matroska_headers(&mut self) -> Result<(), JsValue> {
        // For now, create a simple dummy stream
        // In a full implementation, this would parse the EBML structure
        use oximedia_core::{CodecId, Rational};

        // Create a dummy video stream for demonstration
        let stream = StreamInfo::new(0, CodecId::Vp9, Rational::new(1, 1000));
        self.streams.push(WasmStreamInfo::from(stream));

        Ok(())
    }

    /// Parse Ogg headers.
    fn parse_ogg_headers(&mut self) -> Result<(), JsValue> {
        use oximedia_core::{CodecId, Rational};

        let stream = StreamInfo::new(0, CodecId::Opus, Rational::new(1, 48000));
        self.streams.push(WasmStreamInfo::from(stream));

        Ok(())
    }

    /// Parse FLAC headers.
    fn parse_flac_headers(&mut self) -> Result<(), JsValue> {
        use oximedia_core::{CodecId, Rational};

        let stream = StreamInfo::new(0, CodecId::Flac, Rational::new(1, 44100));
        self.streams.push(WasmStreamInfo::from(stream));

        Ok(())
    }

    /// Parse WAV headers.
    fn parse_wav_headers(&mut self) -> Result<(), JsValue> {
        use oximedia_core::{CodecId, Rational};

        let stream = StreamInfo::new(0, CodecId::Pcm, Rational::new(1, 44100));
        self.streams.push(WasmStreamInfo::from(stream));

        Ok(())
    }

    /// Parse MP4 headers.
    fn parse_mp4_headers(&mut self) -> Result<(), JsValue> {
        use oximedia_core::{CodecId, Rational};

        let stream = StreamInfo::new(0, CodecId::Av1, Rational::new(1, 1000));
        self.streams.push(WasmStreamInfo::from(stream));

        Ok(())
    }

    /// Read a Matroska packet.
    fn read_matroska_packet(&mut self) -> Result<Option<WasmPacket>, JsValue> {
        // Simplified packet reading for demonstration
        // In a full implementation, this would parse EBML clusters

        // Read a simple dummy packet
        let mut buf = vec![0u8; 1024];
        match self.source.read(&mut buf) {
            Ok(0) => Ok(None), // EOF
            Ok(n) => {
                buf.truncate(n);
                use oximedia_core::{Rational, Timestamp};

                let packet = Packet::new(
                    0,
                    Bytes::from(buf),
                    Timestamp::new(0, Rational::new(1, 1000)),
                    PacketFlags::empty(),
                );
                Ok(Some(WasmPacket::from(packet)))
            }
            Err(e) => Err(to_js_error(e)),
        }
    }

    /// Read an Ogg packet.
    fn read_ogg_packet(&mut self) -> Result<Option<WasmPacket>, JsValue> {
        self.read_simple_packet()
    }

    /// Read a FLAC packet.
    fn read_flac_packet(&mut self) -> Result<Option<WasmPacket>, JsValue> {
        self.read_simple_packet()
    }

    /// Read a WAV packet.
    fn read_wav_packet(&mut self) -> Result<Option<WasmPacket>, JsValue> {
        self.read_simple_packet()
    }

    /// Read an MP4 packet.
    fn read_mp4_packet(&mut self) -> Result<Option<WasmPacket>, JsValue> {
        self.read_simple_packet()
    }

    /// Read a simple packet (generic implementation).
    fn read_simple_packet(&mut self) -> Result<Option<WasmPacket>, JsValue> {
        let mut buf = vec![0u8; 4096];
        match self.source.read(&mut buf) {
            Ok(0) => Ok(None),
            Ok(n) => {
                buf.truncate(n);
                use oximedia_core::{Rational, Timestamp};

                let packet = Packet::new(
                    0,
                    Bytes::from(buf),
                    Timestamp::new(0, Rational::new(1, 1000)),
                    PacketFlags::empty(),
                );
                Ok(Some(WasmPacket::from(packet)))
            }
            Err(e) => Err(to_js_error(e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_demuxer_new() {
        let data = vec![0x1A, 0x45, 0xDF, 0xA3, 0, 0, 0, 0];
        let demuxer = WasmDemuxer::new(&data);
        assert_eq!(demuxer.size(), 8);
        assert_eq!(demuxer.position(), 0);
    }

    #[test]
    fn test_demuxer_probe() {
        let data = vec![0x1A, 0x45, 0xDF, 0xA3, 0, 0, 0, 0];
        let mut demuxer = WasmDemuxer::new(&data);
        let result = demuxer
            .probe()
            .expect("probe should succeed for Matroska header");
        assert_eq!(result.format(), "Matroska");
    }
}
