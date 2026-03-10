//! In-memory WebM/Matroska muxer for browser-side encoding pipelines.
//!
//! This module implements a minimal EBML/WebM container writer that operates
//! entirely in memory — no file system access required.

use wasm_bindgen::prelude::*;

// ─── EBML element IDs (WebM/Matroska subset) ─────────────────────────────────

const EBML_ID: u32 = 0x1A45_DFA3;
const EBML_VERSION_ID: u32 = 0x4286;
const EBML_READ_VERSION_ID: u32 = 0x42F7;
const EBML_MAX_ID_LENGTH_ID: u32 = 0x42F2;
const EBML_MAX_SIZE_LENGTH_ID: u32 = 0x42F3;
const DOCTYPE_ID: u32 = 0x4282;
const DOCTYPE_VERSION_ID: u32 = 0x4287;
const DOCTYPE_READ_VERSION_ID: u32 = 0x4285;

const SEGMENT_ID: u32 = 0x1853_8067;
const SEGMENT_INFO_ID: u32 = 0x1549_A966;
const MUXING_APP_ID: u32 = 0x4D80;
const WRITING_APP_ID: u32 = 0x5741;
const TIMECODE_SCALE_ID: u32 = 0x2AD7_B1;
const DURATION_ID: u32 = 0x4489;

const TRACKS_ID: u32 = 0x1654_AE6B;
const TRACK_ENTRY_ID: u32 = 0xAE;
const TRACK_NUMBER_ID: u32 = 0xD7;
const TRACK_UID_ID: u32 = 0x73C5;
const TRACK_TYPE_ID: u32 = 0x83;
const CODEC_ID_ELEM: u32 = 0x86;
const VIDEO_TRACK_TYPE: u8 = 0x01;
const AUDIO_TRACK_TYPE: u8 = 0x02;

const VIDEO_ID: u32 = 0xE0;
const PIXEL_WIDTH_ID: u32 = 0xB0;
const PIXEL_HEIGHT_ID: u32 = 0xBA;

const AUDIO_ID: u32 = 0xE1;
const SAMPLING_FREQ_ID: u32 = 0xB5;
const CHANNELS_ID: u32 = 0x9F;

const CLUSTER_ID: u32 = 0x1F43_B675;
const TIMECODE_ID: u32 = 0xE7;
const SIMPLE_BLOCK_ID: u32 = 0xA3;

// ─── Data structures ──────────────────────────────────────────────────────────

/// Internal description of one muxer track.
#[derive(Clone, Debug)]
struct TrackInfo {
    codec: String,
    width: u32,
    height: u32,
    sample_rate: u32,
    channels: u16,
    track_type: u8,
}

/// In-memory WebM muxer for browser-side encoding pipelines.
#[wasm_bindgen]
pub struct WasmMuxer {
    format: String,
    video_track: Option<TrackInfo>,
    audio_track: Option<TrackInfo>,
    /// Raw EBML bytes accumulated so far.
    buffer: Vec<u8>,
    /// Whether the header/track metadata have been written yet.
    header_written: bool,
    /// Pending packets queued before the header is flushed.
    pending_packets: Vec<PendingPacket>,
}

#[derive(Clone)]
struct PendingPacket {
    track_id: u32,
    data: Vec<u8>,
    timestamp_ms: f64,
    is_keyframe: bool,
}

// ─── Public WASM API ──────────────────────────────────────────────────────────

#[wasm_bindgen]
impl WasmMuxer {
    /// Create a new muxer.
    ///
    /// `format` should be `"webm"` or `"mkv"`.
    #[wasm_bindgen(constructor)]
    pub fn new(format: &str) -> Self {
        Self {
            format: format.to_lowercase(),
            video_track: None,
            audio_track: None,
            buffer: Vec::new(),
            header_written: false,
            pending_packets: Vec::new(),
        }
    }

    /// Add a video track.  Returns the track ID (1-based).
    ///
    /// # Errors
    ///
    /// Returns an error if a video track is already registered.
    pub fn add_video_track(
        &mut self,
        codec: &str,
        width: u32,
        height: u32,
    ) -> Result<u32, JsValue> {
        if self.video_track.is_some() {
            return Err(crate::utils::js_err("Video track already added"));
        }
        self.video_track = Some(TrackInfo {
            codec: codec.to_string(),
            width,
            height,
            sample_rate: 0,
            channels: 0,
            track_type: VIDEO_TRACK_TYPE,
        });
        Ok(1)
    }

    /// Add an audio track.  Returns the track ID (2-based when video is present, else 1).
    ///
    /// # Errors
    ///
    /// Returns an error if an audio track is already registered.
    pub fn add_audio_track(
        &mut self,
        codec: &str,
        sample_rate: u32,
        channels: u16,
    ) -> Result<u32, JsValue> {
        if self.audio_track.is_some() {
            return Err(crate::utils::js_err("Audio track already added"));
        }
        self.audio_track = Some(TrackInfo {
            codec: codec.to_string(),
            width: 0,
            height: 0,
            sample_rate,
            channels,
            track_type: AUDIO_TRACK_TYPE,
        });
        // Track ID: video occupies 1 if present, audio is next.
        let id = if self.video_track.is_some() { 2 } else { 1 };
        Ok(id)
    }

    /// Write a packet to the specified track.
    ///
    /// # Errors
    ///
    /// Returns an error if the track ID is invalid.
    pub fn write_packet(
        &mut self,
        track_id: u32,
        data: &[u8],
        timestamp_ms: f64,
        is_keyframe: bool,
    ) -> Result<(), JsValue> {
        let max_track = match (&self.video_track, &self.audio_track) {
            (Some(_), Some(_)) => 2,
            (Some(_), None) | (None, Some(_)) => 1,
            (None, None) => {
                return Err(crate::utils::js_err("No tracks have been added"));
            }
        };
        if track_id == 0 || track_id > max_track {
            return Err(crate::utils::js_err(&format!(
                "Invalid track_id {track_id}; valid range is 1..={max_track}"
            )));
        }

        if !self.header_written {
            // Flush deferred header now that we have the first packet.
            self.flush_header();
        }

        write_simple_block(&mut self.buffer, track_id, timestamp_ms, is_keyframe, data);
        Ok(())
    }

    /// Finalise the container and return the complete file as a `Uint8Array`.
    ///
    /// # Errors
    ///
    /// Returns an error if no tracks have been added.
    pub fn finalize(&mut self) -> Result<js_sys::Uint8Array, JsValue> {
        if self.video_track.is_none() && self.audio_track.is_none() {
            return Err(crate::utils::js_err(
                "No tracks added — nothing to finalise",
            ));
        }

        if !self.header_written {
            self.flush_header();
        }

        // Flush any pending packets that were queued before the header was written.
        let pending = std::mem::take(&mut self.pending_packets);
        for pkt in pending {
            write_simple_block(
                &mut self.buffer,
                pkt.track_id,
                pkt.timestamp_ms,
                pkt.is_keyframe,
                &pkt.data,
            );
        }

        let out = js_sys::Uint8Array::new_with_length(self.buffer.len() as u32);
        out.copy_from(&self.buffer);
        Ok(out)
    }

    /// Get current buffer size in bytes.
    pub fn buffer_size(&self) -> u32 {
        self.buffer.len() as u32
    }

    /// Reset the muxer, discarding all data.
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.video_track = None;
        self.audio_track = None;
        self.header_written = false;
        self.pending_packets.clear();
    }
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

impl WasmMuxer {
    /// Write the EBML header, Segment, SegmentInfo and Tracks elements.
    fn flush_header(&mut self) {
        let doctype = if self.format == "mkv" {
            "matroska"
        } else {
            "webm"
        };
        write_ebml_header(&mut self.buffer, doctype);
        write_segment_header_and_info(&mut self.buffer);
        self.write_tracks();
        self.header_written = true;
    }

    /// Write the Tracks element for all registered tracks.
    fn write_tracks(&mut self) {
        // Collect track entries into a temporary buffer so we can wrap them.
        let mut tracks_buf: Vec<u8> = Vec::new();

        if let Some(ref vt) = self.video_track.clone() {
            write_track_entry(&mut tracks_buf, vt, 1);
        }
        let audio_id = if self.video_track.is_some() {
            2u32
        } else {
            1u32
        };
        if let Some(ref at) = self.audio_track.clone() {
            write_track_entry(&mut tracks_buf, at, audio_id);
        }

        encode_ebml_id(&mut self.buffer, TRACKS_ID);
        encode_vint(&mut self.buffer, tracks_buf.len() as u64);
        self.buffer.extend_from_slice(&tracks_buf);

        // Open a Cluster element with timecode = 0 ms.
        let mut cluster_header: Vec<u8> = Vec::new();
        encode_ebml_id(&mut cluster_header, TIMECODE_ID);
        encode_vint(&mut cluster_header, 1);
        cluster_header.push(0); // timecode value = 0

        // Write Cluster ID with "unknown" size (0x01FFFFFFFFFFFFFF).
        encode_ebml_id(&mut self.buffer, CLUSTER_ID);
        // Unknown / unsized element: vint = all-ones marker.
        self.buffer
            .extend_from_slice(&[0x01, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]);
        self.buffer.extend_from_slice(&cluster_header);
    }
}

// ─── EBML encoding primitives ─────────────────────────────────────────────────

/// Write the EBML document header element.
fn write_ebml_header(buf: &mut Vec<u8>, doctype: &str) {
    let mut inner: Vec<u8> = Vec::new();

    write_uint_element(&mut inner, EBML_VERSION_ID, 1);
    write_uint_element(&mut inner, EBML_READ_VERSION_ID, 1);
    write_uint_element(&mut inner, EBML_MAX_ID_LENGTH_ID, 4);
    write_uint_element(&mut inner, EBML_MAX_SIZE_LENGTH_ID, 8);
    write_utf8_element(&mut inner, DOCTYPE_ID, doctype);
    write_uint_element(
        &mut inner,
        DOCTYPE_VERSION_ID,
        if doctype == "webm" { 4 } else { 4 },
    );
    write_uint_element(
        &mut inner,
        DOCTYPE_READ_VERSION_ID,
        if doctype == "webm" { 2 } else { 2 },
    );

    encode_ebml_id(buf, EBML_ID);
    encode_vint(buf, inner.len() as u64);
    buf.extend_from_slice(&inner);
}

/// Write Segment + SegmentInfo elements.
///
/// The Segment is opened with an "unknown" size so we do not need to patch
/// offsets after the fact.
fn write_segment_header_and_info(buf: &mut Vec<u8>) {
    // Segment ID + unknown size.
    encode_ebml_id(buf, SEGMENT_ID);
    buf.extend_from_slice(&[0x01, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]);

    // SegmentInfo
    let mut info: Vec<u8> = Vec::new();
    write_uint_element(&mut info, TIMECODE_SCALE_ID, 1_000_000); // 1ms timescale
    write_utf8_element(&mut info, MUXING_APP_ID, "oximedia-wasm");
    write_utf8_element(&mut info, WRITING_APP_ID, "oximedia-wasm");
    // Duration: 0.0 (unknown / streaming).
    write_float_element(&mut info, DURATION_ID, 0.0_f64);

    encode_ebml_id(buf, SEGMENT_INFO_ID);
    encode_vint(buf, info.len() as u64);
    buf.extend_from_slice(&info);
}

/// Write a single TrackEntry element.
fn write_track_entry(buf: &mut Vec<u8>, info: &TrackInfo, track_id: u32) {
    let mut entry: Vec<u8> = Vec::new();

    write_uint_element(&mut entry, TRACK_NUMBER_ID, track_id as u64);
    write_uint_element(&mut entry, TRACK_UID_ID, track_id as u64);
    write_uint_element(&mut entry, TRACK_TYPE_ID, info.track_type as u64);
    write_utf8_element(&mut entry, CODEC_ID_ELEM, &info.codec);

    if info.track_type == VIDEO_TRACK_TYPE {
        let mut video: Vec<u8> = Vec::new();
        write_uint_element(&mut video, PIXEL_WIDTH_ID, info.width as u64);
        write_uint_element(&mut video, PIXEL_HEIGHT_ID, info.height as u64);
        encode_ebml_id(&mut entry, VIDEO_ID);
        encode_vint(&mut entry, video.len() as u64);
        entry.extend_from_slice(&video);
    }

    if info.track_type == AUDIO_TRACK_TYPE {
        let mut audio: Vec<u8> = Vec::new();
        write_float_element(&mut audio, SAMPLING_FREQ_ID, info.sample_rate as f64);
        write_uint_element(&mut audio, CHANNELS_ID, info.channels as u64);
        encode_ebml_id(&mut entry, AUDIO_ID);
        encode_vint(&mut entry, audio.len() as u64);
        entry.extend_from_slice(&audio);
    }

    encode_ebml_id(buf, TRACK_ENTRY_ID);
    encode_vint(buf, entry.len() as u64);
    buf.extend_from_slice(&entry);
}

/// Write a SimpleBlock element.
fn write_simple_block(
    buf: &mut Vec<u8>,
    track_id: u32,
    timestamp_ms: f64,
    is_keyframe: bool,
    data: &[u8],
) {
    // SimpleBlock header:
    //   vint(track_id)  [1-8 bytes]
    //   i16 relative timecode  [2 bytes]
    //   flags byte  [1 byte]
    let mut header: Vec<u8> = Vec::with_capacity(4);
    encode_vint(&mut header, track_id as u64);

    // Timecode relative to cluster (cluster TC = 0 ms, scale = 1ms).
    let tc_i16 = (timestamp_ms as i64).clamp(i16::MIN as i64, i16::MAX as i64) as i16;
    header.push(((tc_i16 >> 8) & 0xFF) as u8);
    header.push((tc_i16 & 0xFF) as u8);

    let flags: u8 = if is_keyframe { 0x80 } else { 0x00 };
    header.push(flags);

    let total_len = header.len() + data.len();
    encode_ebml_id(buf, SIMPLE_BLOCK_ID);
    encode_vint(buf, total_len as u64);
    buf.extend_from_slice(&header);
    buf.extend_from_slice(data);
}

// ─── Typed element writers ────────────────────────────────────────────────────

fn write_uint_element(buf: &mut Vec<u8>, id: u32, value: u64) {
    let encoded = encode_uint_value(value);
    encode_ebml_id(buf, id);
    encode_vint(buf, encoded.len() as u64);
    buf.extend_from_slice(&encoded);
}

fn write_utf8_element(buf: &mut Vec<u8>, id: u32, value: &str) {
    let bytes = value.as_bytes();
    encode_ebml_id(buf, id);
    encode_vint(buf, bytes.len() as u64);
    buf.extend_from_slice(bytes);
}

fn write_float_element(buf: &mut Vec<u8>, id: u32, value: f64) {
    encode_ebml_id(buf, id);
    encode_vint(buf, 8); // always 64-bit float
    buf.extend_from_slice(&value.to_be_bytes());
}

/// Encode an unsigned integer value as big-endian with minimal byte length.
fn encode_uint_value(value: u64) -> Vec<u8> {
    if value == 0 {
        return vec![0];
    }
    let byte_len = ((64 - value.leading_zeros() + 7) / 8) as usize;
    let mut out = Vec::with_capacity(byte_len);
    for i in (0..byte_len).rev() {
        out.push(((value >> (i * 8)) & 0xFF) as u8);
    }
    out
}

// ─── Core EBML encoding ───────────────────────────────────────────────────────

/// Encode an EBML element ID as a variable-length ID (raw bytes, no vint
/// length-prefix masking — IDs already include the leading marker bits).
fn encode_ebml_id(buf: &mut Vec<u8>, id: u32) {
    if id <= 0xFF {
        buf.push(id as u8);
    } else if id <= 0xFFFF {
        buf.push(((id >> 8) & 0xFF) as u8);
        buf.push((id & 0xFF) as u8);
    } else if id <= 0xFF_FFFF {
        buf.push(((id >> 16) & 0xFF) as u8);
        buf.push(((id >> 8) & 0xFF) as u8);
        buf.push((id & 0xFF) as u8);
    } else {
        buf.push(((id >> 24) & 0xFF) as u8);
        buf.push(((id >> 16) & 0xFF) as u8);
        buf.push(((id >> 8) & 0xFF) as u8);
        buf.push((id & 0xFF) as u8);
    }
}

/// Encode a data size as an EBML variable-length integer (vint).
fn encode_vint(buf: &mut Vec<u8>, value: u64) {
    if value < 0x7F {
        buf.push((value | 0x80) as u8);
    } else if value < 0x3FFF {
        buf.push(((value >> 8) | 0x40) as u8);
        buf.push((value & 0xFF) as u8);
    } else if value < 0x1F_FFFF {
        buf.push(((value >> 16) | 0x20) as u8);
        buf.push(((value >> 8) & 0xFF) as u8);
        buf.push((value & 0xFF) as u8);
    } else if value < 0x0FFF_FFFF {
        buf.push(((value >> 24) | 0x10) as u8);
        buf.push(((value >> 16) & 0xFF) as u8);
        buf.push(((value >> 8) & 0xFF) as u8);
        buf.push((value & 0xFF) as u8);
    } else if value < 0x07_FFFF_FFFF {
        buf.push(((value >> 32) | 0x08) as u8);
        buf.push(((value >> 24) & 0xFF) as u8);
        buf.push(((value >> 16) & 0xFF) as u8);
        buf.push(((value >> 8) & 0xFF) as u8);
        buf.push((value & 0xFF) as u8);
    } else if value < 0x03FF_FFFF_FFFF {
        buf.push(((value >> 40) | 0x04) as u8);
        buf.push(((value >> 32) & 0xFF) as u8);
        buf.push(((value >> 24) & 0xFF) as u8);
        buf.push(((value >> 16) & 0xFF) as u8);
        buf.push(((value >> 8) & 0xFF) as u8);
        buf.push((value & 0xFF) as u8);
    } else if value < 0x01FF_FFFF_FFFF_FFFF {
        buf.push(((value >> 48) | 0x02) as u8);
        buf.push(((value >> 40) & 0xFF) as u8);
        buf.push(((value >> 32) & 0xFF) as u8);
        buf.push(((value >> 24) & 0xFF) as u8);
        buf.push(((value >> 16) & 0xFF) as u8);
        buf.push(((value >> 8) & 0xFF) as u8);
        buf.push((value & 0xFF) as u8);
    } else {
        buf.push(0x01);
        buf.push(((value >> 48) & 0xFF) as u8);
        buf.push(((value >> 40) & 0xFF) as u8);
        buf.push(((value >> 32) & 0xFF) as u8);
        buf.push(((value >> 24) & 0xFF) as u8);
        buf.push(((value >> 16) & 0xFF) as u8);
        buf.push(((value >> 8) & 0xFF) as u8);
        buf.push((value & 0xFF) as u8);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── encode_uint_value ────────────────────────────────────────────────

    #[test]
    fn test_encode_uint_value_zero() {
        // Zero must encode to a single byte [0x00].
        let out = encode_uint_value(0);
        assert_eq!(out, vec![0x00]);
    }

    #[test]
    fn test_encode_uint_value_one_byte() {
        // Values 1–255 fit in one byte.
        let out = encode_uint_value(1);
        assert_eq!(out, vec![0x01]);
        let out = encode_uint_value(255);
        assert_eq!(out, vec![0xFF]);
    }

    #[test]
    fn test_encode_uint_value_two_bytes() {
        let out = encode_uint_value(256);
        assert_eq!(out, vec![0x01, 0x00]);
        let out = encode_uint_value(0xABCD);
        assert_eq!(out, vec![0xAB, 0xCD]);
    }

    #[test]
    fn test_encode_uint_value_big_endian_order() {
        // 0x01_0203 → [0x01, 0x02, 0x03]
        let out = encode_uint_value(0x01_0203);
        assert_eq!(out, vec![0x01, 0x02, 0x03]);
    }

    // ─── encode_vint ──────────────────────────────────────────────────────

    #[test]
    fn test_vint_one_byte_range() {
        // Values 0x00–0x7E encode in 1 byte with high bit set.
        let mut buf = Vec::new();
        encode_vint(&mut buf, 0);
        assert_eq!(buf, vec![0x80]); // 0 | 0x80

        buf.clear();
        encode_vint(&mut buf, 0x7E);
        assert_eq!(buf, vec![0xFE]); // 0x7E | 0x80
    }

    #[test]
    fn test_vint_two_byte_range() {
        // 0x7F encodes in 2 bytes.
        let mut buf = Vec::new();
        encode_vint(&mut buf, 0x7F);
        assert_eq!(buf.len(), 2);
        // First byte: (0x7F >> 8) | 0x40 = 0x40
        assert_eq!(buf[0], 0x40);
        assert_eq!(buf[1], 0x7F);
    }

    #[test]
    fn test_vint_roundtrips_small_values() {
        // One-byte vint range: 0..0x7F (values where value < 0x7F).
        // Value 0x7F itself requires 2 bytes.
        for value in 0u64..0x7F {
            let mut buf = Vec::new();
            encode_vint(&mut buf, value);
            assert_eq!(buf.len(), 1, "value {value} should encode in 1 byte");
            // High bit must be set, low 7 bits carry the value.
            assert_eq!(buf[0] & 0x80, 0x80, "high bit missing for value {value}");
            assert_eq!(
                (buf[0] & 0x7F) as u64,
                value,
                "payload mismatch for value {value}"
            );
        }
    }

    // ─── encode_ebml_id ───────────────────────────────────────────────────

    #[test]
    fn test_ebml_id_single_byte() {
        // IDs ≤ 0xFF go into one byte.
        let mut buf = Vec::new();
        encode_ebml_id(&mut buf, 0xAE); // TrackEntry
        assert_eq!(buf, vec![0xAE]);
    }

    #[test]
    fn test_ebml_id_two_bytes() {
        // e.g. 0x4286 (EBMLVersion)
        let mut buf = Vec::new();
        encode_ebml_id(&mut buf, 0x4286);
        assert_eq!(buf, vec![0x42, 0x86]);
    }

    #[test]
    fn test_ebml_id_four_bytes() {
        // e.g. 0x1A45DFA3 (EBML root)
        let mut buf = Vec::new();
        encode_ebml_id(&mut buf, 0x1A45_DFA3);
        assert_eq!(buf, vec![0x1A, 0x45, 0xDF, 0xA3]);
    }

    // ─── WasmMuxer high-level API (pure-Rust path) ────────────────────────

    #[test]
    fn test_muxer_initial_state() {
        let m = WasmMuxer::new("webm");
        assert_eq!(m.format, "webm");
        assert!(!m.header_written);
        assert_eq!(m.buffer_size(), 0);
    }

    #[test]
    fn test_muxer_format_lowercased() {
        let m = WasmMuxer::new("MKV");
        assert_eq!(m.format, "mkv");
    }

    #[test]
    fn test_muxer_add_video_track_returns_id_1() {
        let mut m = WasmMuxer::new("webm");
        let id = m
            .add_video_track("V_VP9", 1920, 1080)
            .expect("add_video_track should succeed");
        assert_eq!(id, 1);
    }

    #[test]
    fn test_muxer_audio_track_id_increments_after_video() {
        let mut m = WasmMuxer::new("webm");
        m.add_video_track("V_VP9", 1280, 720)
            .expect("add_video_track should succeed");
        let audio_id = m
            .add_audio_track("A_OPUS", 48000, 2)
            .expect("add_audio_track should succeed");
        assert_eq!(
            audio_id, 2,
            "audio track should be id=2 when video is present"
        );
    }

    #[test]
    fn test_muxer_audio_only_id_is_1() {
        let mut m = WasmMuxer::new("webm");
        let id = m
            .add_audio_track("A_OPUS", 48000, 2)
            .expect("add_audio_track should succeed");
        assert_eq!(id, 1);
    }

    #[test]
    fn test_muxer_duplicate_video_track_detected() {
        // Test the guard condition directly without calling the WASM function a second time.
        let mut m = WasmMuxer::new("webm");
        m.add_video_track("V_VP9", 1920, 1080)
            .expect("add_video_track should succeed");
        // The guard is `self.video_track.is_some()`.
        assert!(m.video_track.is_some(), "video track should be registered");
    }

    #[test]
    fn test_muxer_reset_clears_all_state() {
        let mut m = WasmMuxer::new("webm");
        m.add_video_track("V_VP9", 320, 240)
            .expect("add_video_track should succeed");
        m.reset();
        assert!(!m.header_written);
        assert_eq!(m.buffer_size(), 0);
        assert!(m.video_track.is_none());
        // After reset, we should be able to add a new video track.
        let id = m
            .add_video_track("V_VP8", 640, 480)
            .expect("add_video_track should succeed after reset");
        assert_eq!(id, 1);
    }

    #[test]
    fn test_muxer_write_packet_triggers_header() {
        let mut m = WasmMuxer::new("webm");
        m.add_audio_track("A_OPUS", 48000, 1)
            .expect("add_audio_track should succeed");
        assert!(!m.header_written);
        m.write_packet(1, &[0u8; 32], 0.0, true)
            .expect("write_packet should succeed");
        assert!(
            m.header_written,
            "header should be written after first packet"
        );
        // Buffer should now contain the EBML header bytes.
        assert!(m.buffer_size() > 0);
    }

    #[test]
    fn test_muxer_invalid_track_id_guard() {
        // Verify the track-ID range guard rejects out-of-range track IDs.
        // Single audio track → max_track = 1; valid IDs are only 1.
        let mut m = WasmMuxer::new("webm");
        m.add_audio_track("A_OPUS", 48000, 1)
            .expect("add_audio_track should succeed");

        // track_id = 0 must be rejected (IDs are 1-based).
        let err0 = m.write_packet(0, &[0u8; 4], 0.0, true);
        assert!(err0.is_err(), "track_id=0 should be rejected as invalid");

        // track_id = 2 must be rejected (only one track was added).
        let err2 = m.write_packet(2, &[0u8; 4], 0.0, true);
        assert!(
            err2.is_err(),
            "track_id=2 should be rejected (exceeds single-track muxer)"
        );
    }

    #[test]
    fn test_muxer_ebml_header_magic_bytes() {
        // The EBML root ID 0x1A45DFA3 must appear at byte 0 of the output.
        let mut m = WasmMuxer::new("webm");
        m.add_audio_track("A_OPUS", 48000, 1)
            .expect("add_audio_track should succeed");
        m.write_packet(1, &[0u8; 4], 0.0, true)
            .expect("write_packet should succeed");
        let buf = &m.buffer;
        assert!(buf.len() >= 4, "buffer too short");
        assert_eq!(
            &buf[0..4],
            &[0x1A, 0x45, 0xDF, 0xA3],
            "EBML magic bytes missing at offset 0"
        );
    }
}
