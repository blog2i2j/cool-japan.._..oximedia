//! AVI muxer with OpenDML (>1 GB), PCM audio, H.264 and RGB24 support.
//!
//! Produces AVI 1.0 / OpenDML files with:
//!  - Primary `RIFF AVI ` chunk containing `hdrl`, `movi`, and `idx1`.
//!  - Secondary `RIFF AVIX` chunks when the file exceeds `riff_size_limit`.
//!  - Per-segment OpenDML `ix00`/`ix01` field-indexes and an `indx` super-index.
//!  - Optional PCM audio stream (`01wb` chunks).
//!  - Video codecs: MJPEG, H.264 (Annex-B pass-through), RGB24.

#![forbid(unsafe_code)]
#![allow(clippy::cast_possible_truncation)]

use crate::riff::{write_chunk, write_list};

// ============================================================================
// Constants
// ============================================================================

/// OpenDML-aligned 1 GiB RIFF size threshold.
pub const AVI_RIFF_SIZE_LIMIT: u64 = 1_073_741_312;

// ============================================================================
// Public configuration types
// ============================================================================

/// Audio configuration for an optional PCM audio stream.
#[derive(Clone, Debug)]
pub struct AudioConfig {
    /// Sample rate in Hz (e.g. 48000).
    pub sample_rate: u32,
    /// Number of channels (e.g. 2 for stereo).
    pub channels: u16,
    /// Bits per sample (e.g. 16).
    pub bits_per_sample: u16,
}

/// Video codec selector.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum VideoCodec {
    /// Motion JPEG (default).
    #[default]
    Mjpeg,
    /// H.264 Annex-B byte-stream pass-through.
    H264,
    /// Uncompressed BGR24, bottom-up row order.
    Rgb24,
}

// ============================================================================
// Error type
// ============================================================================

/// Errors returned by [`AviMjpegWriter`].
#[derive(Debug, thiserror::Error)]
pub enum AviError {
    /// A codec is not supported by this implementation.
    #[error("AVI muxer: unsupported codec: {0}")]
    UnsupportedCodec(String),

    /// File exceeded the configured size limit.
    #[error("AVI file exceeded 1 GB limit ({0} bytes); use OpenDML for larger files")]
    FileTooLarge(u64),
}

// ============================================================================
// Internal types
// ============================================================================

/// One entry in an OpenDML `ix##` field-index.
#[derive(Clone)]
struct IxEntry {
    /// Offset of this chunk's *payload* relative to the ix base offset.
    /// Bit 31 set = keyframe.
    offset: u32,
    /// Payload size in bytes (bit 31 may be set for keyframe marker).
    size: u32,
}

/// One entry in the `indx` super-index.
struct IndxEntry {
    qw_offset: u64,
    dw_size: u32,
    dw_duration: u32,
}

/// One entry in the legacy `idx1` table.
#[derive(Clone)]
struct Idx1Entry {
    ckid: [u8; 4],
    flags: u32,
    /// Offset relative to movi fourcc.
    offset: u32,
    size: u32,
}

/// One RIFF segment's metadata (frame indices + precomputed ix entries).
struct Segment {
    video_frame_range: core::ops::Range<usize>,
    audio_chunk_range: core::ops::Range<usize>,
    video_ix_entries: Vec<IxEntry>,
    audio_ix_entries: Vec<IxEntry>,
    idx1_entries: Vec<Idx1Entry>,
    /// Total bytes in the movi *payload* (all 00dc+01wb chunks with RIFF headers).
    movi_bytes: u64,
}

/// Absolute file positions for one RIFF segment's ix chunks.
struct SegmentLayout {
    /// Absolute file offset of the first byte of the movi *payload*
    /// (i.e. the byte immediately after the `movi` four-char code).
    movi_payload_start: u64,
    /// Absolute file offset of the `ix00` chunk header (`i`, `x`, `0`, `0` bytes).
    ix00_chunk_start: u64,
    /// Absolute file offset of the `ix01` chunk header (zero if no audio).
    ix01_chunk_start: u64,
}

// ============================================================================
// AVI writer
// ============================================================================

/// AVI muxer supporting OpenDML (>1 GB), PCM audio, H.264, and RGB24.
///
/// # Example
///
/// ```no_run
/// use oximedia_container::mux::avi::{AviMjpegWriter, VideoCodec};
///
/// let mut writer = AviMjpegWriter::new(640, 480, 30, 1);
/// writer.write_frame(vec![0xFF, 0xD8, 0xFF, 0xD9]).unwrap();
/// let avi_bytes = writer.finish().unwrap();
/// ```
pub struct AviMjpegWriter {
    width: u32,
    height: u32,
    fps_num: u32,
    fps_den: u32,
    codec: VideoCodec,
    audio: Option<AudioConfig>,
    /// `(frame_bytes, is_keyframe)` for each video frame.
    video_frames: Vec<(Vec<u8>, bool)>,
    /// Raw PCM bytes per audio chunk.
    audio_chunks: Vec<Vec<u8>>,
    riff_size_limit: u64,
}

impl AviMjpegWriter {
    /// Create a new writer with MJPEG video and no audio.
    ///
    /// `fps_num / fps_den` is the frame rate rational, e.g. `30, 1` for 30 fps.
    #[must_use]
    pub fn new(width: u32, height: u32, fps_num: u32, fps_den: u32) -> Self {
        Self {
            width,
            height,
            fps_num,
            fps_den,
            codec: VideoCodec::Mjpeg,
            audio: None,
            video_frames: Vec::new(),
            audio_chunks: Vec::new(),
            riff_size_limit: AVI_RIFF_SIZE_LIMIT,
        }
    }

    /// Set the video codec.
    #[must_use]
    pub fn with_video_codec(mut self, codec: VideoCodec) -> Self {
        self.codec = codec;
        self
    }

    /// Enable a PCM audio track.
    #[must_use]
    pub fn with_audio(mut self, cfg: AudioConfig) -> Self {
        self.audio = Some(cfg);
        self
    }

    /// Override the per-RIFF size threshold (intended for tests only).
    #[doc(hidden)]
    #[must_use]
    pub fn with_riff_size_limit(mut self, limit: u64) -> Self {
        self.riff_size_limit = limit;
        self
    }

    /// Append a video frame.
    ///
    /// For H.264 the bytes must be a complete Annex-B NAL unit stream.
    /// For RGB24 the bytes must be `width * height * 3` uncompressed BGR24.
    pub fn write_frame(&mut self, frame_bytes: Vec<u8>) -> Result<(), AviError> {
        let is_key = match &self.codec {
            VideoCodec::Mjpeg => true,
            VideoCodec::H264 => {
                // Heuristic: IDR slice = NAL type 5.
                frame_bytes
                    .windows(5)
                    .any(|w| w[0..4] == [0, 0, 0, 1] && (w[4] & 0x1F) == 5)
                    || frame_bytes
                        .windows(4)
                        .any(|w| w[0..3] == [0, 0, 1] && (w[3] & 0x1F) == 5)
            }
            VideoCodec::Rgb24 => true,
        };
        self.video_frames.push((frame_bytes, is_key));
        Ok(())
    }

    /// Append a PCM audio chunk (interleaved little-endian samples).
    ///
    /// Has no effect when no audio config has been provided.
    pub fn write_audio_chunk(&mut self, pcm_bytes: Vec<u8>) {
        if self.audio.is_some() {
            self.audio_chunks.push(pcm_bytes);
        }
    }

    /// Return the number of video frames queued.
    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.video_frames.len()
    }

    /// Finalize and return the AVI file bytes.
    pub fn finish(self) -> Result<Vec<u8>, AviError> {
        self.assemble()
    }

    // -----------------------------------------------------------------------
    // BITMAPINFOHEADER
    // -----------------------------------------------------------------------

    fn build_bitmapinfoheader(&self) -> Vec<u8> {
        let mut strf = Vec::with_capacity(40);
        let bi_size_image = match &self.codec {
            VideoCodec::Mjpeg | VideoCodec::Rgb24 => self.width * self.height * 3,
            VideoCodec::H264 => 0,
        };
        extend_u32_le(&mut strf, 40);
        extend_i32_le(&mut strf, self.width as i32);
        extend_i32_le(&mut strf, self.height as i32); // positive = bottom-up
        extend_u16_le(&mut strf, 1); // biPlanes
        extend_u16_le(&mut strf, 24); // biBitCount
        match &self.codec {
            VideoCodec::Mjpeg => strf.extend_from_slice(b"MJPG"),
            VideoCodec::H264 => strf.extend_from_slice(b"H264"),
            VideoCodec::Rgb24 => extend_u32_le(&mut strf, 0), // BI_RGB
        }
        extend_u32_le(&mut strf, bi_size_image);
        extend_i32_le(&mut strf, 0);
        extend_i32_le(&mut strf, 0);
        extend_u32_le(&mut strf, 0);
        extend_u32_le(&mut strf, 0);
        debug_assert_eq!(strf.len(), 40);
        strf
    }

    fn video_handler(&self) -> &[u8; 4] {
        match &self.codec {
            VideoCodec::Mjpeg => b"MJPG",
            VideoCodec::H264 => b"H264",
            VideoCodec::Rgb24 => b"\x00\x00\x00\x00",
        }
    }

    fn build_video_strh(&self, total_frames: u32, max_frame_bytes: u32) -> Vec<u8> {
        let mut strh = Vec::with_capacity(56);
        strh.extend_from_slice(b"vids");
        strh.extend_from_slice(self.video_handler());
        extend_u32_le(&mut strh, 0);
        extend_u16_le(&mut strh, 0);
        extend_u16_le(&mut strh, 0);
        extend_u32_le(&mut strh, 0);
        extend_u32_le(&mut strh, self.fps_den); // dwScale
        extend_u32_le(&mut strh, self.fps_num); // dwRate
        extend_u32_le(&mut strh, 0);
        extend_u32_le(&mut strh, total_frames);
        extend_u32_le(&mut strh, max_frame_bytes);
        extend_u32_le(&mut strh, 0xFFFF_FFFFu32);
        extend_u32_le(&mut strh, 0);
        extend_i16_le(&mut strh, 0);
        extend_i16_le(&mut strh, 0);
        extend_i16_le(&mut strh, self.width as i16);
        extend_i16_le(&mut strh, self.height as i16);
        debug_assert_eq!(strh.len(), 56);
        strh
    }

    fn build_audio_strh(cfg: &AudioConfig, total_audio_bytes: u32) -> Vec<u8> {
        let block_align = u32::from(cfg.channels) * u32::from(cfg.bits_per_sample / 8);
        let total_samples = total_audio_bytes.checked_div(block_align).unwrap_or(0);
        let mut strh = Vec::with_capacity(56);
        strh.extend_from_slice(b"auds");
        extend_u32_le(&mut strh, 0); // fccHandler = 0 for PCM
        extend_u32_le(&mut strh, 0);
        extend_u16_le(&mut strh, 0);
        extend_u16_le(&mut strh, 0);
        extend_u32_le(&mut strh, 0);
        extend_u32_le(&mut strh, 1); // dwScale
        extend_u32_le(&mut strh, cfg.sample_rate); // dwRate
        extend_u32_le(&mut strh, 0);
        extend_u32_le(&mut strh, total_samples); // dwLength
        extend_u32_le(&mut strh, 2048);
        extend_u32_le(&mut strh, 0xFFFF_FFFFu32);
        extend_u32_le(&mut strh, block_align); // dwSampleSize
        extend_i16_le(&mut strh, 0);
        extend_i16_le(&mut strh, 0);
        extend_i16_le(&mut strh, 0);
        extend_i16_le(&mut strh, 0);
        debug_assert_eq!(strh.len(), 56);
        strh
    }

    fn build_waveformatex(cfg: &AudioConfig) -> Vec<u8> {
        let block_align = cfg.channels * (cfg.bits_per_sample / 8);
        let avg_bytes_per_sec = u32::from(cfg.sample_rate) * u32::from(block_align);
        let mut wfx = Vec::with_capacity(18);
        extend_u16_le(&mut wfx, 1); // WAVE_FORMAT_PCM
        extend_u16_le(&mut wfx, cfg.channels);
        extend_u32_le(&mut wfx, cfg.sample_rate);
        extend_u32_le(&mut wfx, avg_bytes_per_sec);
        extend_u16_le(&mut wfx, block_align);
        extend_u16_le(&mut wfx, cfg.bits_per_sample);
        extend_u16_le(&mut wfx, 0); // cbSize
        debug_assert_eq!(wfx.len(), 18);
        wfx
    }

    // -----------------------------------------------------------------------
    // OpenDML index chunk builders
    // -----------------------------------------------------------------------

    /// Build an `ix##` field-index chunk payload.
    fn build_ix_payload(chunk_id: &[u8; 4], base_offset: u64, entries: &[IxEntry]) -> Vec<u8> {
        let n = entries.len() as u32;
        let mut p = Vec::with_capacity(24 + entries.len() * 8);
        extend_u16_le(&mut p, 2); // wLongsPerEntry
        p.push(1); // bIndexSubType = AVI_INDEX_IS_DATA
        p.push(1); // bIndexType = AVI_INDEX_OF_CHUNKS
        extend_u32_le(&mut p, n);
        p.extend_from_slice(chunk_id);
        extend_u64_le(&mut p, base_offset);
        extend_u32_le(&mut p, 0); // dwReserved
        for e in entries {
            extend_u32_le(&mut p, e.offset);
            extend_u32_le(&mut p, e.size);
        }
        p
    }

    /// Build an `indx` super-index chunk payload.
    fn build_indx_payload(chunk_id: &[u8; 4], entries: &[IndxEntry]) -> Vec<u8> {
        let n = entries.len() as u32;
        let mut p = Vec::with_capacity(24 + entries.len() * 16);
        extend_u16_le(&mut p, 4); // wLongsPerEntry
        p.push(0); // bIndexSubType
        p.push(0); // bIndexType = AVI_INDEX_OF_INDEXES
        extend_u32_le(&mut p, n);
        p.extend_from_slice(chunk_id);
        p.extend_from_slice(&[0u8; 12]); // dwReserved[3]
        for e in entries {
            extend_u64_le(&mut p, e.qw_offset);
            extend_u32_le(&mut p, e.dw_size);
            extend_u32_le(&mut p, e.dw_duration);
        }
        p
    }

    // -----------------------------------------------------------------------
    // Main assembly
    // -----------------------------------------------------------------------

    fn assemble(self) -> Result<Vec<u8>, AviError> {
        let total_frames = self.video_frames.len() as u32;
        let has_audio = self.audio.is_some();
        let num_streams: u32 = if has_audio { 2 } else { 1 };

        let max_video_bytes = self
            .video_frames
            .iter()
            .map(|(f, _)| f.len() as u32)
            .max()
            .unwrap_or(0);

        let total_audio_bytes: u32 = self
            .audio_chunks
            .iter()
            .map(|c| c.len() as u32)
            .fold(0u32, u32::saturating_add);

        let microsec_per_frame = if self.fps_num == 0 {
            33_333u32
        } else {
            (1_000_000u64 * u64::from(self.fps_den) / u64::from(self.fps_num)) as u32
        };

        let max_bytes_per_sec = self
            .width
            .saturating_mul(self.height)
            .saturating_mul(3)
            .saturating_mul(self.fps_num);

        // ---- strh/strf for video ----
        let video_strh = self.build_video_strh(total_frames, max_video_bytes);
        let video_strf = self.build_bitmapinfoheader();
        let mut video_strl_body = Vec::new();
        write_chunk(&mut video_strl_body, b"strh", &video_strh);
        write_chunk(&mut video_strl_body, b"strf", &video_strf);

        // ---- strh/strf for audio ----
        let mut audio_strl_body = Vec::new();
        if let Some(cfg) = &self.audio {
            let audio_strh = Self::build_audio_strh(cfg, total_audio_bytes);
            let audio_strf = Self::build_waveformatex(cfg);
            write_chunk(&mut audio_strl_body, b"strh", &audio_strh);
            write_chunk(&mut audio_strl_body, b"strf", &audio_strf);
        }

        // ---- avih ----
        let mut avih = Vec::with_capacity(56);
        extend_u32_le(&mut avih, microsec_per_frame);
        extend_u32_le(&mut avih, max_bytes_per_sec);
        extend_u32_le(&mut avih, 0);
        let avi_flags: u32 = if has_audio { 0x10 | 0x100 } else { 0x10 };
        extend_u32_le(&mut avih, avi_flags);
        extend_u32_le(&mut avih, total_frames);
        extend_u32_le(&mut avih, 0);
        extend_u32_le(&mut avih, num_streams);
        extend_u32_le(&mut avih, max_video_bytes);
        extend_u32_le(&mut avih, self.width);
        extend_u32_le(&mut avih, self.height);
        avih.extend_from_slice(&[0u8; 16]);
        debug_assert_eq!(avih.len(), 56);

        // ---- LIST odml + dmlh ----
        let mut dmlh = Vec::with_capacity(8);
        extend_u32_le(&mut dmlh, total_frames);
        extend_u32_le(&mut dmlh, 0);
        let mut odml_payload = Vec::new();
        write_chunk(&mut odml_payload, b"dmlh", &dmlh);

        // ---- Segment partitioning ----
        let segments =
            partition_segments(&self.video_frames, &self.audio_chunks, self.riff_size_limit);
        let num_segments = segments.len();

        // ---- Compute hdrl size (depends on num_segments for indx sizing) ----
        //
        // indx payload is fixed: 24 header + num_segments * 16 entries.
        let video_indx_payload_size = 24 + num_segments * 16;
        let audio_indx_payload_size = if has_audio { 24 + num_segments * 16 } else { 0 };

        // strl LIST = 8 (LIST+size) + 4 (strl) + body + 8 (indx hdr) + indx_payload
        let video_strl_list_total = 8 + 4 + video_strl_body.len() + 8 + video_indx_payload_size;
        let audio_strl_list_total = if has_audio {
            8 + 4 + audio_strl_body.len() + 8 + audio_indx_payload_size
        } else {
            0
        };
        // odml LIST = 8 + 4 + odml_payload.len()
        let odml_list_total = 8 + 4 + odml_payload.len();
        // hdrl content = avih chunk + video_strl + audio_strl + odml
        let hdrl_content_size =
            (8 + 56) + video_strl_list_total + audio_strl_list_total + odml_list_total;
        // hdrl LIST = 8 + 4 + hdrl_content_size
        let hdrl_list_total = 8 + 4 + hdrl_content_size;

        // File: RIFF(4)+size(4)+AVI (4) = 12, then hdrl LIST.
        // Primary movi LIST starts at: 12 + hdrl_list_total bytes.
        let primary_movi_list_start: u64 = 12 + hdrl_list_total as u64;

        // ---- Layout simulation ----
        let layouts = compute_layout(primary_movi_list_start, &segments, has_audio);

        // ---- Build indx entries ----
        let video_indx_entries: Vec<IndxEntry> = segments
            .iter()
            .zip(layouts.iter())
            .map(|(seg, lay)| {
                let ix_payload_size = 24 + seg.video_ix_entries.len() * 8;
                IndxEntry {
                    qw_offset: lay.ix00_chunk_start,
                    dw_size: (8 + ix_payload_size) as u32,
                    dw_duration: seg.video_ix_entries.len() as u32,
                }
            })
            .collect();

        let audio_indx_entries: Vec<IndxEntry> = if has_audio {
            segments
                .iter()
                .zip(layouts.iter())
                .map(|(seg, lay)| {
                    let ix_payload_size = 24 + seg.audio_ix_entries.len() * 8;
                    IndxEntry {
                        qw_offset: lay.ix01_chunk_start,
                        dw_size: (8 + ix_payload_size) as u32,
                        dw_duration: seg.audio_ix_entries.len() as u32,
                    }
                })
                .collect()
        } else {
            Vec::new()
        };

        let video_indx_payload = Self::build_indx_payload(b"00dc", &video_indx_entries);
        let audio_indx_payload = if has_audio {
            Self::build_indx_payload(b"01wb", &audio_indx_entries)
        } else {
            Vec::new()
        };

        debug_assert_eq!(video_indx_payload.len(), video_indx_payload_size);
        if has_audio {
            debug_assert_eq!(audio_indx_payload.len(), audio_indx_payload_size);
        }

        // ---- Build full hdrl with real indx ----
        let mut hdrl_content = Vec::with_capacity(hdrl_content_size);
        write_chunk(&mut hdrl_content, b"avih", &avih);

        let mut vstrl = video_strl_body;
        write_chunk(&mut vstrl, b"indx", &video_indx_payload);
        write_list(&mut hdrl_content, b"strl", &vstrl);

        if has_audio {
            let mut astrl = audio_strl_body;
            write_chunk(&mut astrl, b"indx", &audio_indx_payload);
            write_list(&mut hdrl_content, b"strl", &astrl);
        }
        write_list(&mut hdrl_content, b"odml", &odml_payload);
        debug_assert_eq!(hdrl_content.len(), hdrl_content_size);

        // ---- Build movi payloads ----
        let movi_payloads = build_movi_payloads(&self.video_frames, &self.audio_chunks, &segments);

        // ---- Build ix## payloads ----
        let ix00_payloads: Vec<Vec<u8>> = segments
            .iter()
            .zip(layouts.iter())
            .map(|(seg, lay)| {
                Self::build_ix_payload(b"00dc", lay.movi_payload_start, &seg.video_ix_entries)
            })
            .collect();

        let ix01_payloads: Vec<Vec<u8>> = if has_audio {
            segments
                .iter()
                .zip(layouts.iter())
                .map(|(seg, lay)| {
                    Self::build_ix_payload(b"01wb", lay.movi_payload_start, &seg.audio_ix_entries)
                })
                .collect()
        } else {
            Vec::new()
        };

        // ---- Build legacy idx1 for primary segment ----
        let idx1_payload = build_idx1_payload(&segments[0]);

        // ---- Assemble primary RIFF ----
        let mut primary_payload = Vec::new();
        write_list(&mut primary_payload, b"hdrl", &hdrl_content);
        write_list(&mut primary_payload, b"movi", &movi_payloads[0]);
        write_chunk(&mut primary_payload, b"ix00", &ix00_payloads[0]);
        if has_audio {
            write_chunk(&mut primary_payload, b"ix01", &ix01_payloads[0]);
        }
        write_chunk(&mut primary_payload, b"idx1", &idx1_payload);

        // ---- Assemble final output ----
        let mut out = Vec::new();
        out.extend_from_slice(b"RIFF");
        let primary_riff_size = 4u32 + primary_payload.len() as u32;
        extend_u32_le(&mut out, primary_riff_size);
        out.extend_from_slice(b"AVI ");
        out.extend_from_slice(&primary_payload);

        for seg_idx in 1..num_segments {
            let mut avix_payload = Vec::new();
            write_list(&mut avix_payload, b"movi", &movi_payloads[seg_idx]);
            write_chunk(&mut avix_payload, b"ix00", &ix00_payloads[seg_idx]);
            if has_audio {
                write_chunk(&mut avix_payload, b"ix01", &ix01_payloads[seg_idx]);
            }
            out.extend_from_slice(b"RIFF");
            let avix_riff_size = 4u32 + avix_payload.len() as u32;
            extend_u32_le(&mut out, avix_riff_size);
            out.extend_from_slice(b"AVIX");
            out.extend_from_slice(&avix_payload);
        }

        Ok(out)
    }
}

// ============================================================================
// Segment partitioning
// ============================================================================

/// RIFF-padded total size of a chunk with the given payload length.
/// Returns: fourcc(4) + size(4) + payload_len + optional_pad(0 or 1).
fn padded_chunk_total(payload_len: usize) -> u64 {
    let total = 8 + payload_len as u64;
    if payload_len % 2 == 1 {
        total + 1
    } else {
        total
    }
}

fn partition_segments(
    video_frames: &[(Vec<u8>, bool)],
    audio_chunks: &[Vec<u8>],
    limit: u64,
) -> Vec<Segment> {
    let mut segments: Vec<Segment> = Vec::new();

    let mut vid_start = 0usize;
    let mut aud_start = 0usize;
    let mut vid_count = 0usize;
    let mut aud_count = 0usize;
    let mut video_ix: Vec<IxEntry> = Vec::new();
    let mut audio_ix: Vec<IxEntry> = Vec::new();
    let mut idx1: Vec<Idx1Entry> = Vec::new();
    let mut movi_bytes: u64 = 0;
    let mut aud_idx = 0usize; // cursor into audio_chunks

    for (vi, (frame, is_key)) in video_frames.iter().enumerate() {
        let aud_chunk = audio_chunks.get(aud_idx);
        let audio_size = aud_chunk.map(|a| padded_chunk_total(a.len())).unwrap_or(0);
        let video_size = padded_chunk_total(frame.len());
        let needed = audio_size + video_size;

        // Overflow: flush current segment and start a new one.
        if movi_bytes + needed > limit && vid_count > 0 {
            segments.push(Segment {
                video_frame_range: vid_start..vid_start + vid_count,
                audio_chunk_range: aud_start..aud_start + aud_count,
                video_ix_entries: core::mem::take(&mut video_ix),
                audio_ix_entries: core::mem::take(&mut audio_ix),
                idx1_entries: core::mem::take(&mut idx1),
                movi_bytes,
            });
            vid_start = vi;
            aud_start = aud_idx;
            vid_count = 0;
            aud_count = 0;
            movi_bytes = 0;
        }

        // Add paired audio chunk first (interleaving convention).
        if let Some(audio) = aud_chunk {
            let audio_off = movi_bytes as u32 + 8; // offset to payload = chunk start + 8
            audio_ix.push(IxEntry {
                offset: audio_off,
                size: audio.len() as u32,
            });
            idx1.push(Idx1Entry {
                ckid: *b"01wb",
                flags: 0,
                offset: movi_bytes as u32 + 4, // offset relative to movi fourcc
                size: audio.len() as u32,
            });
            movi_bytes += padded_chunk_total(audio.len());
            aud_count += 1;
            aud_idx += 1;
        }

        // Add video chunk.
        let video_off = movi_bytes as u32 + 8;
        let keyframe_bit: u32 = if *is_key { 0x8000_0000 } else { 0 };
        video_ix.push(IxEntry {
            offset: video_off,
            size: frame.len() as u32 | keyframe_bit,
        });
        idx1.push(Idx1Entry {
            ckid: *b"00dc",
            flags: if *is_key { 0x10 } else { 0 },
            offset: movi_bytes as u32 + 4,
            size: frame.len() as u32,
        });
        movi_bytes += padded_chunk_total(frame.len());
        vid_count += 1;
    }

    // Drain remaining audio-only chunks.
    while aud_idx < audio_chunks.len() {
        let audio = &audio_chunks[aud_idx];
        let audio_size = padded_chunk_total(audio.len());
        if movi_bytes + audio_size > limit && aud_count > 0 {
            segments.push(Segment {
                video_frame_range: vid_start..vid_start + vid_count,
                audio_chunk_range: aud_start..aud_start + aud_count,
                video_ix_entries: core::mem::take(&mut video_ix),
                audio_ix_entries: core::mem::take(&mut audio_ix),
                idx1_entries: core::mem::take(&mut idx1),
                movi_bytes,
            });
            vid_start = video_frames.len();
            aud_start = aud_idx;
            vid_count = 0;
            aud_count = 0;
            movi_bytes = 0;
        }
        let audio_off = movi_bytes as u32 + 8;
        audio_ix.push(IxEntry {
            offset: audio_off,
            size: audio.len() as u32,
        });
        idx1.push(Idx1Entry {
            ckid: *b"01wb",
            flags: 0,
            offset: movi_bytes as u32 + 4,
            size: audio.len() as u32,
        });
        movi_bytes += audio_size;
        aud_count += 1;
        aud_idx += 1;
    }

    // Always push the final (possibly only) segment.
    segments.push(Segment {
        video_frame_range: vid_start..vid_start + vid_count,
        audio_chunk_range: aud_start..aud_start + aud_count,
        video_ix_entries: core::mem::take(&mut video_ix),
        audio_ix_entries: core::mem::take(&mut audio_ix),
        idx1_entries: core::mem::take(&mut idx1),
        movi_bytes,
    });

    segments
}

// ============================================================================
// Layout computation
// ============================================================================

/// Compute absolute file offsets for each segment's chunks.
fn compute_layout(
    primary_movi_list_start: u64,
    segments: &[Segment],
    has_audio: bool,
) -> Vec<SegmentLayout> {
    let mut layouts = Vec::with_capacity(segments.len());
    // cursor tracks the file position of the start of the next RIFF segment.
    // For the primary (index 0) this is where the movi LIST starts.
    // For secondary segments this is where the "RIFF" fourcc starts.
    let mut cursor = primary_movi_list_start;

    for (idx, seg) in segments.iter().enumerate() {
        // Absolute position of the movi LIST start for this segment.
        let movi_list_start = if idx == 0 {
            cursor
        } else {
            cursor + 12 // skip RIFF(4)+size(4)+AVIX(4)
        };

        // movi payload starts 12 bytes into the LIST (LIST+size+mowi_type).
        let movi_payload_start = movi_list_start + 12;

        // movi LIST total: LIST(4)+size(4)+movi(4)+payload.
        let movi_list_total = 8 + 4 + seg.movi_bytes;

        // ix00 chunk starts right after movi LIST.
        let ix00_chunk_start = movi_list_start + movi_list_total;
        let ix00_payload_size = 24 + seg.video_ix_entries.len() as u64 * 8;
        let ix00_total = 8 + ix00_payload_size;

        // ix01 chunk starts right after ix00.
        let ix01_chunk_start = ix00_chunk_start + ix00_total;
        let ix01_total = if has_audio {
            let ix01_payload_size = 24 + seg.audio_ix_entries.len() as u64 * 8;
            8 + ix01_payload_size
        } else {
            0
        };

        layouts.push(SegmentLayout {
            movi_payload_start,
            ix00_chunk_start,
            ix01_chunk_start,
        });

        // Advance cursor.
        if idx == 0 {
            // Primary: after movi + ix00 + ix01 + idx1.
            let idx1_total = 8 + seg.idx1_entries.len() as u64 * 16;
            cursor = ix01_chunk_start + ix01_total + idx1_total;
        } else {
            // Secondary: after the entire AVIX chunk.
            // AVIX = RIFF(4)+size(4)+AVIX(4)+movi+ix00+ix01
            let avix_content = 4 + movi_list_total + ix00_total + ix01_total;
            cursor = (movi_list_start - 12) + 8 + avix_content;
        }
    }

    layouts
}

// ============================================================================
// Movi payload builder
// ============================================================================

fn build_movi_payloads(
    video_frames: &[(Vec<u8>, bool)],
    audio_chunks: &[Vec<u8>],
    segments: &[Segment],
) -> Vec<Vec<u8>> {
    segments
        .iter()
        .map(|seg| {
            let mut movi = Vec::with_capacity(seg.movi_bytes as usize);
            let vend = seg.video_frame_range.end;
            let aend = seg.audio_chunk_range.end;
            let mut ai = seg.audio_chunk_range.start;

            for vi in seg.video_frame_range.start..vend {
                if ai < aend {
                    write_chunk(&mut movi, b"01wb", &audio_chunks[ai]);
                    ai += 1;
                }
                write_chunk(&mut movi, b"00dc", &video_frames[vi].0);
            }
            while ai < aend {
                write_chunk(&mut movi, b"01wb", &audio_chunks[ai]);
                ai += 1;
            }
            movi
        })
        .collect()
}

// ============================================================================
// idx1 builder
// ============================================================================

fn build_idx1_payload(seg: &Segment) -> Vec<u8> {
    let mut idx1 = Vec::with_capacity(seg.idx1_entries.len() * 16);
    for e in &seg.idx1_entries {
        idx1.extend_from_slice(&e.ckid);
        extend_u32_le(&mut idx1, e.flags);
        extend_u32_le(&mut idx1, e.offset);
        extend_u32_le(&mut idx1, e.size);
    }
    idx1
}

// ============================================================================
// Byte-extend helpers
// ============================================================================

#[inline]
fn extend_u64_le(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}

#[inline]
fn extend_u32_le(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

#[inline]
fn extend_u16_le(buf: &mut Vec<u8>, v: u16) {
    buf.extend_from_slice(&v.to_le_bytes());
}

#[inline]
fn extend_i16_le(buf: &mut Vec<u8>, v: i16) {
    buf.extend_from_slice(&v.to_le_bytes());
}

#[inline]
fn extend_i32_le(buf: &mut Vec<u8>, v: i32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

// ============================================================================
// Unit tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_jpeg(tag: u8) -> Vec<u8> {
        vec![0xFF, 0xD8, tag, 0xFF, 0xD9]
    }

    #[test]
    fn empty_writer_produces_valid_avi() {
        let writer = AviMjpegWriter::new(320, 240, 30, 1);
        let bytes = writer.finish().expect("finish should succeed");
        assert!(bytes.len() >= 12);
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"AVI ");
    }

    #[test]
    fn single_frame_roundtrip() {
        let mut writer = AviMjpegWriter::new(64, 48, 25, 1);
        writer.write_frame(minimal_jpeg(0xAA)).expect("write_frame");
        assert_eq!(writer.frame_count(), 1);
        let bytes = writer.finish().expect("finish");
        let has_idx1 = bytes.windows(4).any(|w| w == b"idx1");
        let has_00dc = bytes.windows(4).any(|w| w == b"00dc");
        assert!(has_idx1);
        assert!(has_00dc);
    }

    #[test]
    fn avi_error_file_too_large_message() {
        let err = AviError::FileTooLarge(2_000_000_000);
        let msg = err.to_string();
        assert!(msg.contains("1 GB"), "error message: {msg}");
    }

    #[test]
    fn microsec_per_frame_calculation() {
        let writer = AviMjpegWriter::new(1, 1, 30, 1);
        let bytes = writer.finish().expect("finish");
        // RIFF(4)+size(4)+AVI (4) = 12
        // LIST(4)+size(4)+hdrl(4) = 12
        // avih(4)+size(4) = 8 → microsec_per_frame at 12+12+8 = 32
        let us = u32::from_le_bytes([bytes[32], bytes[33], bytes[34], bytes[35]]);
        assert_eq!(us, 33_333);
    }
}
