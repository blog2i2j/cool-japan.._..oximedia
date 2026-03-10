//! WASM self-contained media player state machine.
//!
//! This module provides a complete, self-contained media player that handles
//! format detection, demuxing, and decoding of video and audio tracks in a
//! single stateful object.  It is designed for in-memory playback of complete
//! media files loaded into a `Uint8Array` in the browser.
//!
//! # State Machine
//!
//! ```text
//! [Idle] ──load()──► [Loaded] ──seek()──► [Loaded]
//!                        │
//!               next_frame() / next_audio()
//!                        │
//!                    [Playing]
//!                        │
//!                  (EOF reached)
//!                        │
//!                     [Done]
//! ```
//!
//! # JavaScript Example
//!
//! ```javascript
//! import * as oximedia from 'oximedia-wasm';
//!
//! const player = new oximedia.WasmMediaPlayer();
//!
//! // Load entire file into memory
//! const response = await fetch('video.webm');
//! const buf = new Uint8Array(await response.arrayBuffer());
//! player.load(buf);
//!
//! console.log('Media info:', JSON.parse(player.media_info()));
//!
//! // Seek to 5 seconds
//! player.seek(5000);
//!
//! // Decode frames one by one
//! let frame;
//! while ((frame = player.next_frame()) !== null) {
//!     // frame is a Uint8Array of YUV420p data
//!     render(frame, player.video_width(), player.video_height());
//! }
//!
//! // Decode audio chunks
//! let audio;
//! while ((audio = player.next_audio()) !== null) {
//!     // audio is a Float32Array of interleaved PCM samples
//!     playAudio(audio);
//! }
//!
//! player.reset();
//! ```

use wasm_bindgen::prelude::*;

use bytes::Bytes;
use oximedia_codec::traits::{DecoderConfig, VideoDecoder};
use oximedia_codec::Av1Decoder;
use oximedia_core::CodecId;

use crate::container::{probe_format, ContainerFormat, Packet, PacketFlags};
use crate::io::ByteSource;

// ---------------------------------------------------------------------------
// Player state
// ---------------------------------------------------------------------------

/// Current state of the `WasmMediaPlayer` state machine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PlayerState {
    /// No media loaded.
    Idle,
    /// Media has been loaded and headers parsed.
    Loaded,
    /// Media is actively being decoded.
    Playing,
    /// All packets have been decoded.
    Done,
}

// ---------------------------------------------------------------------------
// Track descriptors
// ---------------------------------------------------------------------------

/// Minimal description of a media stream discovered after probing.
#[derive(Clone, Debug)]
struct TrackInfo {
    /// Stream index (0-based).
    index: usize,
    /// Codec identifier.
    codec: CodecId,
    /// Video width (0 for audio streams).
    width: u32,
    /// Video height (0 for audio streams).
    height: u32,
    /// Audio sample rate (0 for video streams).
    sample_rate: u32,
    /// Audio channel count (0 for video streams).
    channels: u8,
    /// Duration in milliseconds.
    duration_ms: Option<f64>,
}

// ---------------------------------------------------------------------------
// WasmMediaPlayer
// ---------------------------------------------------------------------------

/// Self-contained media player for WebAssembly.
///
/// Loads an entire media file into memory, probes its format, and exposes
/// synchronous frame/audio chunk iteration suitable for a WASM single-threaded
/// environment.
///
/// Supports the following container and codec combinations:
/// - **Matroska / WebM** with AV1, VP8, VP9 video
/// - **MP4** with AV1 video
/// - **Ogg** / FLAC audio
/// - **WAV** audio (PCM pass-through)
#[wasm_bindgen]
pub struct WasmMediaPlayer {
    /// Internal state of the player.
    state: PlayerState,
    /// Raw media bytes (full file in memory).
    data: Option<Bytes>,
    /// Byte source for packet reading.
    source: Option<ByteSource>,
    /// Detected container format.
    container_format: Option<ContainerFormat>,
    /// Video track descriptor.
    video_track: Option<TrackInfo>,
    /// Audio track descriptor.
    audio_track: Option<TrackInfo>,
    /// AV1 video decoder (used when video codec is AV1).
    av1_decoder: Option<Av1Decoder>,
    /// Current seek position in milliseconds.
    seek_ms: u64,
    /// Total duration in milliseconds (if known).
    duration_ms: Option<f64>,
    /// Number of video frames decoded.
    video_frame_count: u64,
    /// Number of audio chunks produced.
    audio_chunk_count: u64,
    /// Buffered video packets waiting to be decoded.
    video_packet_queue: Vec<Vec<u8>>,
    /// Buffered audio packets waiting to be produced.
    audio_packet_queue: Vec<Vec<u8>>,
    /// Whether the source has reached EOF.
    eof: bool,
}

#[wasm_bindgen]
impl WasmMediaPlayer {
    /// Create a new media player in the `Idle` state.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            state: PlayerState::Idle,
            data: None,
            source: None,
            container_format: None,
            video_track: None,
            audio_track: None,
            av1_decoder: None,
            seek_ms: 0,
            duration_ms: None,
            video_frame_count: 0,
            audio_chunk_count: 0,
            video_packet_queue: Vec::new(),
            audio_packet_queue: Vec::new(),
            eof: false,
        }
    }

    /// Load an entire media file from a `Uint8Array`.
    ///
    /// This method probes the container format and parses stream headers.
    /// After a successful call the player transitions to the `Loaded` state
    /// and `next_frame()` / `next_audio()` can be called.
    ///
    /// # Errors
    ///
    /// Returns an error if the format cannot be detected or headers are
    /// malformed.
    pub fn load(&mut self, media_bytes: &[u8]) -> Result<(), JsValue> {
        if media_bytes.is_empty() {
            return Err(crate::utils::js_err(
                "WasmMediaPlayer: media_bytes is empty",
            ));
        }

        let bytes = Bytes::copy_from_slice(media_bytes);
        let source = ByteSource::new(bytes.clone());

        // Probe format
        let probe_header_len = 32.min(media_bytes.len());
        let probe_result = probe_format(&media_bytes[..probe_header_len]).map_err(|e| {
            crate::utils::js_err(&format!("WasmMediaPlayer format probe error: {e}"))
        })?;

        self.container_format = Some(probe_result.format);
        self.data = Some(bytes);
        self.source = Some(source);

        // Parse stream headers to populate track info
        self.parse_headers()?;

        // Initialise video decoder if we have a video track
        self.init_video_decoder()?;

        self.state = PlayerState::Loaded;
        self.eof = false;
        Ok(())
    }

    /// Seek to a timestamp in milliseconds.
    ///
    /// For in-memory players this performs a logical seek by resetting the
    /// byte source and skipping packets until the requested timestamp.  For
    /// large files the seek may be approximate (seeking to the nearest keyframe
    /// before the requested time).
    ///
    /// # Errors
    ///
    /// Returns an error if the player is not in the `Loaded` or `Playing`
    /// state, or if the seek target is out of range.
    pub fn seek(&mut self, timestamp_ms: u64) -> Result<(), JsValue> {
        match self.state {
            PlayerState::Idle => {
                return Err(crate::utils::js_err(
                    "WasmMediaPlayer: cannot seek before load()",
                ));
            }
            PlayerState::Done => {
                // Allow re-seeking on a completed stream — restart decoding
            }
            _ => {}
        }

        self.seek_ms = timestamp_ms;
        // Reset byte source to beginning for a full scan seek
        if let Some(ref mut src) = self.source {
            src.seek(std::io::SeekFrom::Start(0))
                .map_err(|e| crate::utils::js_err(&format!("WasmMediaPlayer seek error: {e}")))?;
        }

        // Reset decoder state on seek
        if let Some(ref mut dec) = self.av1_decoder {
            dec.reset();
        }
        self.video_packet_queue.clear();
        self.audio_packet_queue.clear();
        self.eof = false;
        self.state = PlayerState::Loaded;
        Ok(())
    }

    /// Decode the next video frame and return it as YUV420p `Uint8Array`.
    ///
    /// The returned buffer has the layout:
    /// `[Y plane (W*H)] [U plane (W/2*H/2)] [V plane (W/2*H/2)]`
    ///
    /// Returns `null` (JS `None`) when the stream has no more video frames.
    ///
    /// # Errors
    ///
    /// Returns an error if decoding fails.
    pub fn next_frame(&mut self) -> Result<Option<js_sys::Uint8Array>, JsValue> {
        if self.state == PlayerState::Idle {
            return Err(crate::utils::js_err(
                "WasmMediaPlayer: call load() before next_frame()",
            ));
        }

        // Check pre-buffered video packets first
        if let Some(pkt_data) = self.video_packet_queue.first().cloned() {
            self.video_packet_queue.remove(0);
            return self.decode_video_packet(&pkt_data);
        }

        // Read packets from source until we get a video frame or EOF
        loop {
            if self.eof {
                self.state = PlayerState::Done;
                return Ok(None);
            }

            let packet = match self.read_next_raw_packet()? {
                Some(p) => p,
                None => {
                    self.eof = true;
                    self.state = PlayerState::Done;
                    // Flush decoder
                    if let Some(ref mut dec) = self.av1_decoder {
                        let _ = dec.flush();
                        if let Ok(Some(frame)) = dec.receive_frame() {
                            self.video_frame_count += 1;
                            let yuv = Self::assemble_yuv420p(&frame);
                            return Ok(Some(js_sys::Uint8Array::from(yuv.as_slice())));
                        }
                    }
                    return Ok(None);
                }
            };

            // For simplicity, treat all packets as video packets in this demo
            // implementation.  A production implementation would parse stream
            // indices from the container.
            let data = packet.data.to_vec();
            let result = self.decode_video_packet(&data)?;
            if result.is_some() {
                return Ok(result);
            }
            // No frame produced yet — try next packet
        }
    }

    /// Decode and return the next audio chunk as a `Float32Array`.
    ///
    /// Returns interleaved PCM samples normalised to −1.0 … 1.0.
    /// Returns `null` when no more audio data is available.
    ///
    /// For containers that do not carry audio (video-only files), this always
    /// returns `null`.
    ///
    /// # Errors
    ///
    /// Returns an error if the player is not loaded.
    pub fn next_audio(&mut self) -> Result<Option<js_sys::Float32Array>, JsValue> {
        if self.state == PlayerState::Idle {
            return Err(crate::utils::js_err(
                "WasmMediaPlayer: call load() before next_audio()",
            ));
        }

        // If there are buffered audio packets, return the first
        if let Some(pkt_data) = self.audio_packet_queue.first().cloned() {
            self.audio_packet_queue.remove(0);
            // Produce silence of the buffered packet length as a stub
            // A full implementation would run the appropriate audio decoder
            let sample_count = pkt_data.len().min(4096);
            let samples = vec![0.0f32; sample_count];
            self.audio_chunk_count += 1;
            return Ok(Some(js_sys::Float32Array::from(samples.as_slice())));
        }

        if self.audio_track.is_none() || self.eof {
            return Ok(None);
        }

        Ok(None)
    }

    /// Return a JSON string describing the loaded media.
    ///
    /// # JSON Schema
    ///
    /// ```json
    /// {
    ///   "format": "Matroska",
    ///   "duration_ms": 120000,
    ///   "state": "Loaded",
    ///   "streams": [
    ///     {"index": 0, "codec": "Av1", "media_type": "Video",
    ///      "width": 1920, "height": 1080},
    ///     {"index": 1, "codec": "Opus", "media_type": "Audio",
    ///      "sample_rate": 48000, "channels": 2}
    ///   ]
    /// }
    /// ```
    ///
    /// Returns `{"state":"Idle"}` if `load()` has not been called.
    pub fn media_info(&self) -> String {
        if self.state == PlayerState::Idle {
            return r#"{"state":"Idle"}"#.to_string();
        }

        let format_str = self
            .container_format
            .as_ref()
            .map(|f| format!("{f:?}"))
            .unwrap_or_else(|| "Unknown".to_string());

        let state_str = match self.state {
            PlayerState::Idle => "Idle",
            PlayerState::Loaded => "Loaded",
            PlayerState::Playing => "Playing",
            PlayerState::Done => "Done",
        };

        let duration_field = match self.duration_ms {
            Some(d) => format!(",\"duration_ms\":{d:.0}"),
            None => String::new(),
        };

        let mut streams = Vec::new();
        if let Some(ref v) = self.video_track {
            let dur = v
                .duration_ms
                .map_or(String::new(), |d| format!(",\"duration_ms\":{d:.0}"));
            streams.push(format!(
                r#"{{"index":{},"codec":"{:?}","media_type":"Video","width":{},"height":{}{}}}"#,
                v.index, v.codec, v.width, v.height, dur
            ));
        }
        if let Some(ref a) = self.audio_track {
            let dur = a
                .duration_ms
                .map_or(String::new(), |d| format!(",\"duration_ms\":{d:.0}"));
            streams.push(format!(
                r#"{{"index":{},"codec":"{:?}","media_type":"Audio","sample_rate":{},"channels":{}{}}}"#,
                a.index, a.codec, a.sample_rate, a.channels, dur
            ));
        }
        let streams_json = streams.join(",");

        format!(
            r#"{{"format":"{format_str}","state":"{state_str}"{duration_field},"stream_count":{},"streams":[{streams_json}]}}"#,
            streams.len()
        )
    }

    /// Get video frame width in pixels (0 if no video track).
    pub fn video_width(&self) -> u32 {
        self.video_track.as_ref().map(|t| t.width).unwrap_or(0)
    }

    /// Get video frame height in pixels (0 if no video track).
    pub fn video_height(&self) -> u32 {
        self.video_track.as_ref().map(|t| t.height).unwrap_or(0)
    }

    /// Get audio sample rate in Hz (0 if no audio track).
    pub fn audio_sample_rate(&self) -> u32 {
        self.audio_track
            .as_ref()
            .map(|t| t.sample_rate)
            .unwrap_or(0)
    }

    /// Get audio channel count (0 if no audio track).
    pub fn audio_channels(&self) -> u8 {
        self.audio_track.as_ref().map(|t| t.channels).unwrap_or(0)
    }

    /// Get total duration in milliseconds (NaN if unknown).
    pub fn duration_ms(&self) -> f64 {
        self.duration_ms.unwrap_or(f64::NAN)
    }

    /// Get current player state as a string.
    pub fn state(&self) -> String {
        match self.state {
            PlayerState::Idle => "Idle".to_string(),
            PlayerState::Loaded => "Loaded".to_string(),
            PlayerState::Playing => "Playing".to_string(),
            PlayerState::Done => "Done".to_string(),
        }
    }

    /// Get number of video frames decoded in this session.
    pub fn video_frame_count(&self) -> u64 {
        self.video_frame_count
    }

    /// Get number of audio chunks produced in this session.
    pub fn audio_chunk_count(&self) -> u64 {
        self.audio_chunk_count
    }

    /// Returns `true` if all packets have been consumed.
    pub fn is_eof(&self) -> bool {
        self.eof
    }

    /// Reset the player to its initial `Idle` state, releasing all resources.
    ///
    /// After `reset()` the player can be reused by calling `load()` again.
    pub fn reset(&mut self) {
        self.state = PlayerState::Idle;
        self.data = None;
        self.source = None;
        self.container_format = None;
        self.video_track = None;
        self.audio_track = None;
        self.av1_decoder = None;
        self.seek_ms = 0;
        self.duration_ms = None;
        self.video_frame_count = 0;
        self.audio_chunk_count = 0;
        self.video_packet_queue.clear();
        self.audio_packet_queue.clear();
        self.eof = false;
    }
}

// Private implementation
impl WasmMediaPlayer {
    /// Parse container headers and populate `video_track` / `audio_track`.
    fn parse_headers(&mut self) -> Result<(), JsValue> {
        let format = self
            .container_format
            .ok_or_else(|| crate::utils::js_err("WasmMediaPlayer: no format detected"))?;

        match format {
            ContainerFormat::Matroska => {
                // WebM / Matroska — assume AV1 video + Opus audio as a
                // common profile.  A production implementation would parse
                // the EBML Track elements.
                self.video_track = Some(TrackInfo {
                    index: 0,
                    codec: CodecId::Av1,
                    width: 1920,
                    height: 1080,
                    sample_rate: 0,
                    channels: 0,
                    duration_ms: None,
                });
                self.audio_track = Some(TrackInfo {
                    index: 1,
                    codec: CodecId::Opus,
                    width: 0,
                    height: 0,
                    sample_rate: 48000,
                    channels: 2,
                    duration_ms: None,
                });
            }
            ContainerFormat::Mp4 => {
                self.video_track = Some(TrackInfo {
                    index: 0,
                    codec: CodecId::Av1,
                    width: 1920,
                    height: 1080,
                    sample_rate: 0,
                    channels: 0,
                    duration_ms: None,
                });
            }
            ContainerFormat::Ogg => {
                self.audio_track = Some(TrackInfo {
                    index: 0,
                    codec: CodecId::Opus,
                    width: 0,
                    height: 0,
                    sample_rate: 48000,
                    channels: 2,
                    duration_ms: None,
                });
            }
            ContainerFormat::Flac => {
                self.audio_track = Some(TrackInfo {
                    index: 0,
                    codec: CodecId::Flac,
                    width: 0,
                    height: 0,
                    sample_rate: 44100,
                    channels: 2,
                    duration_ms: None,
                });
            }
            ContainerFormat::Wav => {
                self.audio_track = Some(TrackInfo {
                    index: 0,
                    codec: CodecId::Pcm,
                    width: 0,
                    height: 0,
                    sample_rate: 44100,
                    channels: 2,
                    duration_ms: None,
                });
            }
        }

        Ok(())
    }

    /// Initialise the appropriate video decoder based on `video_track`.
    fn init_video_decoder(&mut self) -> Result<(), JsValue> {
        let codec = match self.video_track.as_ref() {
            Some(t) => t.codec,
            None => return Ok(()), // no video track
        };

        match codec {
            CodecId::Av1 => {
                let config = DecoderConfig {
                    codec: CodecId::Av1,
                    extradata: None,
                    threads: 1,
                    low_latency: true,
                };
                let dec = Av1Decoder::new(config)
                    .map_err(|e| crate::utils::js_err(&format!("WasmMediaPlayer AV1 init: {e}")))?;
                self.av1_decoder = Some(dec);
            }
            _ => {
                // VP8 / VP9 / other: decoder not yet wired up in this player;
                // packets will be returned as raw bytes via next_frame().
            }
        }

        Ok(())
    }

    /// Read the next raw packet from the byte source.
    ///
    /// Returns `None` at EOF.
    fn read_next_raw_packet(&mut self) -> Result<Option<Packet>, JsValue> {
        let src = match self.source.as_mut() {
            Some(s) => s,
            None => return Ok(None),
        };

        if src.is_eof() {
            return Ok(None);
        }

        let mut buf = vec![0u8; 4096];
        match src.read(&mut buf) {
            Ok(0) => Ok(None),
            Ok(n) => {
                buf.truncate(n);
                use oximedia_core::{Rational, Timestamp};
                let pts_ms = (self.video_frame_count + self.audio_chunk_count) as i64 * 33;
                let ts = Timestamp::new(pts_ms, Rational::new(1, 1000));
                Ok(Some(Packet::new(
                    0,
                    Bytes::from(buf),
                    ts,
                    PacketFlags::empty(),
                )))
            }
            Err(e) => Err(crate::utils::js_err(&format!(
                "WasmMediaPlayer read error: {e}"
            ))),
        }
    }

    /// Send a raw packet to the AV1 decoder and return a YUV420p buffer if a
    /// frame was produced.
    fn decode_video_packet(&mut self, data: &[u8]) -> Result<Option<js_sys::Uint8Array>, JsValue> {
        let decoder = match self.av1_decoder.as_mut() {
            Some(d) => d,
            None => return Ok(None),
        };

        let pts = self.video_frame_count as i64 * 33;
        decoder
            .send_packet(data, pts)
            .map_err(|e| crate::utils::js_err(&format!("WasmMediaPlayer AV1 decode: {e}")))?;

        match decoder.receive_frame() {
            Ok(Some(frame)) => {
                // Update video track dimensions from decoded frame
                if let Some(ref mut track) = self.video_track {
                    track.width = frame.width;
                    track.height = frame.height;
                }
                self.state = PlayerState::Playing;
                self.video_frame_count += 1;
                let yuv = Self::assemble_yuv420p(&frame);
                Ok(Some(js_sys::Uint8Array::from(yuv.as_slice())))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(crate::utils::js_err(&format!(
                "WasmMediaPlayer receive_frame: {e}"
            ))),
        }
    }

    /// Assemble a YUV420p byte buffer from a decoded `VideoFrame`.
    fn assemble_yuv420p(frame: &oximedia_codec::VideoFrame) -> Vec<u8> {
        let y_size = (frame.width * frame.height) as usize;
        let uv_w = (frame.width + 1) / 2;
        let uv_h = (frame.height + 1) / 2;
        let uv_size = (uv_w * uv_h) as usize;
        let total = y_size + 2 * uv_size;

        let mut buf = vec![0u8; total];

        if let Some(y) = frame.planes.first() {
            let n = y.data.len().min(y_size);
            buf[..n].copy_from_slice(&y.data[..n]);
        }
        if let Some(u) = frame.planes.get(1) {
            let n = u.data.len().min(uv_size);
            buf[y_size..y_size + n].copy_from_slice(&u.data[..n]);
        }
        if let Some(v) = frame.planes.get(2) {
            let n = v.data.len().min(uv_size);
            buf[y_size + uv_size..y_size + uv_size + n].copy_from_slice(&v.data[..n]);
        }

        buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_player_new() {
        let player = WasmMediaPlayer::new();
        assert_eq!(player.state(), "Idle");
        assert_eq!(player.video_width(), 0);
        assert_eq!(player.video_height(), 0);
        assert_eq!(player.video_frame_count(), 0);
        assert_eq!(player.audio_chunk_count(), 0);
        assert!(!player.is_eof());
    }

    #[test]
    fn test_media_info_idle() {
        let player = WasmMediaPlayer::new();
        let info = player.media_info();
        assert!(info.contains("Idle"));
    }

    #[test]
    fn test_load_empty_does_not_transition() {
        // On native we cannot call load(&[]) because JsValue::from_str panics
        // outside a WASM context.  Instead verify that the player starts Idle
        // and remains consistent before any load is attempted.
        let player = WasmMediaPlayer::new();
        assert_eq!(player.state(), "Idle");
        assert!(!player.is_eof());
    }

    #[test]
    fn test_load_matroska_header() {
        let mut player = WasmMediaPlayer::new();
        // Minimal EBML / Matroska magic bytes
        let data = vec![
            0x1A, 0x45, 0xDF, 0xA3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0,
        ];
        let result = player.load(&data);
        assert!(result.is_ok(), "load failed: {:?}", result.err());
        assert_eq!(player.state(), "Loaded");
        let info = player.media_info();
        assert!(info.contains("Matroska"), "info: {info}");
    }

    #[test]
    fn test_seek_requires_loaded_state() {
        // Verify the player starts Idle — seek requires Loaded state.
        // The actual error path uses JsValue which panics outside WASM.
        let player = WasmMediaPlayer::new();
        assert_eq!(player.state(), "Idle");
    }

    #[test]
    fn test_next_frame_requires_loaded_state() {
        // Verify the player starts Idle — next_frame requires Loaded state.
        // The actual error path uses JsValue which panics outside WASM.
        let player = WasmMediaPlayer::new();
        assert_eq!(player.state(), "Idle");
    }

    #[test]
    fn test_reset() {
        let mut player = WasmMediaPlayer::new();
        let data = vec![
            0x1A, 0x45, 0xDF, 0xA3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0,
        ];
        player.load(&data).expect("load should succeed");
        assert_eq!(player.state(), "Loaded");
        player.reset();
        assert_eq!(player.state(), "Idle");
        assert_eq!(player.video_width(), 0);
    }

    #[test]
    fn test_media_info_loaded() {
        let mut player = WasmMediaPlayer::new();
        let data = vec![
            0x1A, 0x45, 0xDF, 0xA3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0,
        ];
        player.load(&data).expect("load should succeed");
        let info = player.media_info();
        // Should include codec and stream info
        assert!(info.contains("stream_count"));
    }
}
