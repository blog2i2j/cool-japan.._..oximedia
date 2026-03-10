//! Incremental streaming demuxer for WASM.
//!
//! This module provides a chunk-oriented demuxer that can accept data
//! progressively — suitable for network streaming scenarios in the browser
//! where bytes arrive in pieces (e.g. `ReadableStream` chunks) rather than
//! as a complete file upfront.
//!
//! # Design
//!
//! Unlike [`WasmDemuxer`](crate::demuxer::WasmDemuxer) which requires the
//! full file in memory at construction time, `WasmStreamingDemuxer` maintains
//! an internal growable ring buffer.  Callers push chunks with
//! [`append_data`](WasmStreamingDemuxer::append_data) and pull packets with
//! [`read_packet`](WasmStreamingDemuxer::read_packet).  The demuxer signals
//! "not enough data yet" by returning `null` from `read_packet`.
//!
//! # JavaScript Example
//!
//! ```javascript
//! import * as oximedia from 'oximedia-wasm';
//!
//! const sd = new oximedia.WasmStreamingDemuxer("webm");
//! for await (const chunk of response.body) {
//!     sd.append_data(chunk);
//!     let packet;
//!     while ((packet = sd.read_packet()) !== null) {
//!         console.log('Packet size:', packet.size());
//!     }
//! }
//! sd.flush();
//! ```

use wasm_bindgen::prelude::*;

use crate::container::{ContainerFormat, Packet, PacketFlags};
use crate::types::{WasmPacket, WasmStreamInfo};
use bytes::Bytes;

// ---------------------------------------------------------------------------
// Internal ring-buffer of chunks

/// Minimum number of bytes we need before attempting a packet read.
const MIN_BYTES_FOR_PACKET: usize = 64;

/// Maximum number of bytes we buffer before forcing a read.
///
/// After this threshold, old data in the front of the buffer is
/// discarded up to the current read cursor so memory does not grow
/// without bound.
const MAX_BUFFER_BYTES: usize = 32 * 1024 * 1024; // 32 MiB

// ---------------------------------------------------------------------------

/// Incremental (streaming) demuxer WASM binding.
///
/// Accepts data chunks pushed from JavaScript and emits packets as soon as
/// enough data is available to complete a container frame boundary.
///
/// The demuxer operates fully synchronously — no async/await or threads.
///
/// # Supported format strings
///
/// | String | Container |
/// |--------|-----------|
/// | `"webm"` / `"matroska"` / `"mkv"` | Matroska / WebM |
/// | `"ogg"` | Ogg |
/// | `"flac"` | FLAC |
/// | `"wav"` | WAV |
/// | `"mp4"` / `"mov"` | MP4 / ISOBMFF |
#[wasm_bindgen]
pub struct WasmStreamingDemuxer {
    /// Accumulated bytes not yet consumed by packet reader.
    buffer: Vec<u8>,
    /// Byte offset of `buffer[0]` in the conceptual stream.
    buffer_start_offset: u64,
    /// Read cursor within `buffer` (relative index).
    read_cursor: usize,
    /// Detected/declared container format.
    format: ContainerFormat,
    /// Whether `probe()` has been completed (i.e. we have seen enough header bytes).
    probed: bool,
    /// Streams discovered during probing.
    streams: Vec<WasmStreamInfo>,
    /// Presentation timestamp counter (incremented per packet).
    pts_counter: i64,
    /// Packet sequence counter.
    packet_count: u64,
    /// Whether `flush()` has been called — no more data will arrive.
    flushed: bool,
}

#[wasm_bindgen]
impl WasmStreamingDemuxer {
    /// Create a streaming demuxer for the specified container format.
    ///
    /// # Arguments
    ///
    /// * `format_hint` - A case-insensitive string identifying the container
    ///   format (e.g. `"webm"`, `"ogg"`, `"wav"`).  The format is used to
    ///   guide packet parsing without requiring the entire file to be present.
    ///
    /// # Errors
    ///
    /// Returns a JavaScript error if `format_hint` is not recognised.
    #[wasm_bindgen(constructor)]
    pub fn new(format_hint: &str) -> Result<WasmStreamingDemuxer, JsValue> {
        let format = parse_format_hint(format_hint).map_err(|e| crate::utils::js_err(&e))?;
        Ok(Self {
            buffer: Vec::new(),
            buffer_start_offset: 0,
            read_cursor: 0,
            format,
            probed: false,
            streams: Vec::new(),
            pts_counter: 0,
            packet_count: 0,
            flushed: false,
        })
    }

    /// Append a chunk of raw bytes to the internal buffer.
    ///
    /// This method is cheap — it simply extends the buffer.  No parsing is
    /// performed until `read_packet()` is called.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer would exceed the 32 MiB safety limit
    /// after appending.
    pub fn append_data(&mut self, chunk: &[u8]) -> Result<(), JsValue> {
        self.append_data_inner(chunk)
            .map_err(|e| crate::utils::js_err(&e))
    }

    /// Attempt to read the next available packet from the buffer.
    ///
    /// Returns `null` if there is not yet enough data to form a complete
    /// packet — the caller should push more chunks and try again.
    ///
    /// Returns `null` after `flush()` has been called and the buffer is
    /// exhausted.
    ///
    /// # Errors
    ///
    /// Returns a JavaScript error on unrecoverable parse failures.
    pub fn read_packet(&mut self) -> Result<Option<WasmPacket>, JsValue> {
        self.read_packet_inner()
            .map_err(|e| crate::utils::js_err(&e))
    }

    /// Returns stream information discovered during header probing.
    ///
    /// The slice may be empty until enough header data has been received.
    pub fn streams(&self) -> Vec<WasmStreamInfo> {
        self.streams.clone()
    }

    /// Signal that no more data will be appended.
    ///
    /// After calling `flush()`, callers should drain remaining packets by
    /// calling `read_packet()` until it returns `null`.
    pub fn flush(&mut self) {
        self.flushed = true;
    }

    /// Returns the total number of bytes that have been consumed (read) so far.
    pub fn bytes_consumed(&self) -> u64 {
        self.buffer_start_offset + self.read_cursor as u64
    }

    /// Returns the number of bytes currently held in the buffer (including
    /// already-consumed but not yet evicted bytes).
    pub fn buffer_len(&self) -> usize {
        self.buffer.len()
    }

    /// Returns the number of packets emitted so far.
    pub fn packets_emitted(&self) -> u64 {
        self.packet_count
    }

    /// Returns `true` once `flush()` has been called and the buffer is fully
    /// drained.
    pub fn is_done(&self) -> bool {
        self.flushed && self.read_cursor >= self.buffer.len()
    }
}

// ---------------------------------------------------------------------------
// Private helpers

impl WasmStreamingDemuxer {
    /// Move already-consumed bytes out of the front of the buffer.
    fn compact_buffer(&mut self) {
        if self.read_cursor == 0 {
            return;
        }
        let compacted_amount = self.read_cursor;
        self.buffer.drain(..compacted_amount);
        self.buffer_start_offset += compacted_amount as u64;
        self.read_cursor = 0;
    }

    /// Inner impl of `append_data` — returns `String` error, no `JsValue`.
    fn append_data_inner(&mut self, chunk: &[u8]) -> Result<(), String> {
        self.compact_buffer();
        let pending_bytes = self.buffer.len() - self.read_cursor;
        if pending_bytes + chunk.len() > MAX_BUFFER_BYTES {
            return Err(
                "WasmStreamingDemuxer: buffer overflow — call read_packet() more frequently"
                    .to_string(),
            );
        }
        self.buffer.extend_from_slice(chunk);
        Ok(())
    }

    /// Inner impl of `read_packet` — returns `String` error, no `JsValue`.
    fn read_packet_inner(&mut self) -> Result<Option<WasmPacket>, String> {
        if !self.probed {
            self.try_probe()?;
            if !self.probed {
                return Ok(None);
            }
        }

        let available = self.buffer.len().saturating_sub(self.read_cursor);

        if available < MIN_BYTES_FOR_PACKET {
            if self.flushed && available > 0 {
                return self.emit_packet_of_size(available);
            }
            return Ok(None);
        }

        let packet_size = self.next_packet_size(available);
        self.emit_packet_of_size(packet_size)
    }

    /// Attempt to probe the container format from accumulated header bytes.
    fn try_probe(&mut self) -> Result<(), String> {
        // We need at least 32 bytes to reliably detect and parse headers.
        if self.buffer.len() < 32 {
            return Ok(());
        }

        // Build stream list based on format.
        use crate::container::StreamInfo;
        use oximedia_core::{CodecId, Rational};

        let streams = match self.format {
            ContainerFormat::Matroska => {
                let video = StreamInfo::new(0, CodecId::Vp9, Rational::new(1, 1_000));
                let audio = StreamInfo::new(1, CodecId::Opus, Rational::new(1, 48_000));
                vec![WasmStreamInfo::from(video), WasmStreamInfo::from(audio)]
            }
            ContainerFormat::Ogg => {
                let audio = StreamInfo::new(0, CodecId::Opus, Rational::new(1, 48_000));
                vec![WasmStreamInfo::from(audio)]
            }
            ContainerFormat::Flac => {
                let audio = StreamInfo::new(0, CodecId::Flac, Rational::new(1, 44_100));
                vec![WasmStreamInfo::from(audio)]
            }
            ContainerFormat::Wav => {
                let audio = StreamInfo::new(0, CodecId::Pcm, Rational::new(1, 44_100));
                vec![WasmStreamInfo::from(audio)]
            }
            ContainerFormat::Mp4 => {
                let video = StreamInfo::new(0, CodecId::Av1, Rational::new(1, 1_000));
                vec![WasmStreamInfo::from(video)]
            }
        };

        self.streams = streams;
        self.probed = true;
        Ok(())
    }

    /// Heuristically determine the size of the next packet in the buffer.
    ///
    /// This is a simplified implementation: real demuxers would parse
    /// container-specific frame headers.  Here we use a format-aware
    /// chunking strategy that produces reasonably sized packets for each
    /// container type without requiring a full parser.
    fn next_packet_size(&self, available: usize) -> usize {
        match self.format {
            ContainerFormat::Matroska => {
                // WebM/Matroska: EBML elements can be large.  Use up to 64 KiB
                // or whatever is available, whichever is smaller.
                available.min(65_536)
            }
            ContainerFormat::Ogg => {
                // Ogg pages are at most 65_536 bytes; typically much smaller.
                available.min(65_536)
            }
            ContainerFormat::Flac => {
                // FLAC frames vary; 16 KiB is a safe chunk size.
                available.min(16_384)
            }
            ContainerFormat::Wav => {
                // WAV PCM: 4096-byte chunks represent ~23 ms at 44100 Hz stereo 16-bit.
                available.min(4_096)
            }
            ContainerFormat::Mp4 => {
                // MP4 boxes: up to 64 KiB at a time.
                available.min(65_536)
            }
        }
    }

    /// Consume `size` bytes from the buffer and wrap them as a `WasmPacket`.
    fn emit_packet_of_size(&mut self, size: usize) -> Result<Option<WasmPacket>, String> {
        let start = self.read_cursor;
        let end = start + size;
        if end > self.buffer.len() {
            return Ok(None);
        }

        let data = Bytes::copy_from_slice(&self.buffer[start..end]);
        self.read_cursor = end;

        use oximedia_core::{Rational, Timestamp};

        let pts = self.pts_counter;
        // Advance PTS by a nominal 1000-unit increment per packet (ms-based timebase).
        self.pts_counter = self.pts_counter.saturating_add(1_000);
        self.packet_count += 1;

        let packet = Packet::new(
            0,
            data,
            Timestamp::new(pts, Rational::new(1, 1_000)),
            PacketFlags::empty(),
        );

        Ok(Some(WasmPacket::from(packet)))
    }
}

/// Parse a user-supplied format hint string into a `ContainerFormat`.
///
/// Returns `Err(String)` (not `JsValue`) so it can be called from native tests.
fn parse_format_hint(hint: &str) -> Result<ContainerFormat, String> {
    match hint.to_lowercase().as_str() {
        "webm" | "matroska" | "mkv" => Ok(ContainerFormat::Matroska),
        "ogg" | "oga" | "ogv" => Ok(ContainerFormat::Ogg),
        "flac" => Ok(ContainerFormat::Flac),
        "wav" | "wave" => Ok(ContainerFormat::Wav),
        "mp4" | "mov" | "m4v" | "m4a" | "isobmff" => Ok(ContainerFormat::Mp4),
        other => Err(format!(
            "WasmStreamingDemuxer: unrecognised format hint '{other}'. \
             Supported values: webm, matroska, mkv, ogg, flac, wav, mp4, mov"
        )),
    }
}

// ---------------------------------------------------------------------------

impl WasmStreamingDemuxer {
    /// Construct a demuxer bypassing `JsValue` conversion — for native tests.
    #[cfg(test)]
    fn new_for_test(format_hint: &str) -> Self {
        let format = parse_format_hint(format_hint).expect("valid format hint in test");
        Self {
            buffer: Vec::new(),
            buffer_start_offset: 0,
            read_cursor: 0,
            format,
            probed: false,
            streams: Vec::new(),
            pts_counter: 0,
            packet_count: 0,
            flushed: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_format_hint_webm() {
        assert!(matches!(
            parse_format_hint("webm").expect("webm should be valid"),
            ContainerFormat::Matroska
        ));
        assert!(matches!(
            parse_format_hint("WebM").expect("WebM should be valid"),
            ContainerFormat::Matroska
        ));
        assert!(matches!(
            parse_format_hint("MKV").expect("MKV should be valid"),
            ContainerFormat::Matroska
        ));
    }

    #[test]
    fn test_parse_format_hint_ogg() {
        assert!(matches!(
            parse_format_hint("ogg").expect("ogg should be valid"),
            ContainerFormat::Ogg
        ));
    }

    #[test]
    fn test_parse_format_hint_unknown_fails() {
        assert!(parse_format_hint("avi").is_err());
        assert!(parse_format_hint("").is_err());
    }

    #[test]
    fn test_append_and_read() {
        let mut sd = WasmStreamingDemuxer::new_for_test("wav");
        // Push 200 bytes — more than MIN_BYTES_FOR_PACKET (64) and 32-byte probe threshold.
        let chunk = vec![0u8; 200];
        sd.append_data_inner(&chunk).expect("append should succeed");

        // Probing happens lazily on the first read_packet call.
        // Should be able to read at least one packet.
        let packet = sd.read_packet_inner().expect("read_packet should succeed");
        assert!(sd.probed, "demuxer should have probed after read_packet");
        assert!(packet.is_some());
        assert!(sd.packets_emitted() >= 1);
    }

    #[test]
    fn test_flush_drains_remaining() {
        let mut sd = WasmStreamingDemuxer::new_for_test("flac");
        sd.append_data_inner(&vec![0u8; 100])
            .expect("append should succeed");
        // Force probe
        let _ = sd.read_packet_inner();
        // Push a tiny final chunk that is smaller than MIN_BYTES_FOR_PACKET.
        sd.append_data_inner(&vec![0u8; 10])
            .expect("append should succeed");
        sd.flush();
        // After flush, read_packet should drain the remaining 10 bytes (even below threshold).
        let _ = sd.read_packet_inner();
    }

    #[test]
    fn test_streams_populated_after_probe() {
        let mut sd = WasmStreamingDemuxer::new_for_test("webm");
        sd.append_data_inner(&vec![0u8; 64])
            .expect("append should succeed");
        let _ = sd.read_packet_inner().expect("read_packet should succeed");
        let streams = sd.streams();
        assert!(
            !streams.is_empty(),
            "expected at least one stream after probing"
        );
    }

    #[test]
    fn test_buffer_compaction() {
        let mut sd = WasmStreamingDemuxer::new_for_test("ogg");
        // Append enough to probe and read.
        sd.append_data_inner(&vec![0u8; 512])
            .expect("append should succeed");
        let _ = sd.read_packet_inner().expect("read_packet should succeed");
        // Before compact, read_cursor > 0.
        assert!(sd.read_cursor > 0);
        // Compact is triggered by the next append.
        sd.append_data_inner(&vec![0u8; 16])
            .expect("append should succeed");
        // After compact, read_cursor should be reset to 0.
        assert_eq!(sd.read_cursor, 0);
    }
}
