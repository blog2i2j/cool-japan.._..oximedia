//! AVI demuxer with audio support and OpenDML multi-RIFF reading.
//!
//! Reads frames from AVI 1.0 and OpenDML files by:
//!  1. Parsing the `LIST hdrl` for stream info (video + audio).
//!  2. If `indx` super-index chunks are present, using `ix##` field indexes
//!     to locate frames across primary and secondary RIFFs.
//!  3. Falling back to linear movi scan (then idx1) when no super-index exists.
//!
//! # Supported stream types
//!
//! - `00dc` video chunks (MJPEG, H.264, RGB24)
//! - `01wb` audio chunks (PCM-LE)

#![forbid(unsafe_code)]
#![allow(clippy::cast_possible_truncation)]

use crate::DecodeSkipCursor;

// ============================================================================
// Error type
// ============================================================================

/// Errors returned by [`AviMjpegReader`].
#[derive(Debug, thiserror::Error)]
pub enum AviDemuxError {
    /// The input is too short to contain a valid RIFF header.
    #[error("data too short to be a valid AVI")]
    TooShort,

    /// The first 4 bytes are not `RIFF` or bytes 8–11 are not `AVI `.
    #[error("not a valid RIFF AVI signature")]
    InvalidSignature,

    /// No `LIST movi` chunk was found in the file.
    #[error("could not locate movi chunk")]
    NoMovi,

    /// The AVI `idx1` chunk is malformed (size not a multiple of 16).
    #[error("idx1 chunk has invalid size ({0} bytes; must be a multiple of 16)")]
    MalformedIdx1(u32),

    /// A chunk offset or length specified in idx1 extends past end-of-file.
    #[error("idx1 entry points outside file bounds (offset={0}, len={1})")]
    IndexOutOfBounds(u32, u32),

    /// The AVI contains a video stream that is not MJPEG.
    #[error("AVI contains non-MJPEG video stream")]
    UnsupportedCodec,
}

// ============================================================================
// Public audio info
// ============================================================================

/// Audio stream parameters parsed from the `auds` strf chunk.
#[derive(Clone, Debug)]
pub struct AviAudioFormat {
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Number of channels.
    pub channels: u16,
    /// Bits per sample.
    pub bits_per_sample: u16,
}

// ============================================================================
// Reader
// ============================================================================

/// AVI demuxer supporting MJPEG/H.264/RGB24 video and PCM audio.
///
/// Supports both AVI 1.0 (≤1 GB) and OpenDML multi-RIFF (>1 GB) files.
pub struct AviMjpegReader {
    data: Vec<u8>,
}

impl AviMjpegReader {
    /// Create a new reader, validating the RIFF AVI signature.
    ///
    /// # Errors
    ///
    /// Returns [`AviDemuxError::TooShort`] if `data` is shorter than 12 bytes.
    /// Returns [`AviDemuxError::InvalidSignature`] if the magic bytes are wrong.
    pub fn new(data: Vec<u8>) -> Result<Self, AviDemuxError> {
        if data.len() < 12 {
            return Err(AviDemuxError::TooShort);
        }
        if &data[0..4] != b"RIFF" {
            return Err(AviDemuxError::InvalidSignature);
        }
        if &data[8..12] != b"AVI " {
            return Err(AviDemuxError::InvalidSignature);
        }
        Ok(Self { data })
    }

    /// Parse the audio format from the `auds` strf chunk, if present.
    ///
    /// Returns `None` if no audio stream was found.
    pub fn audio_format(&self) -> Option<AviAudioFormat> {
        parse_audio_format(&self.data)
    }

    /// Extract all `00dc` video frame payloads in presentation order.
    ///
    /// Tries super-index (indx/ix##) first; falls back to idx1; then linear scan.
    ///
    /// # Errors
    ///
    /// Returns an error if the movi chunk is missing, or indexes are malformed.
    pub fn frames(&self) -> Result<Vec<Vec<u8>>, AviDemuxError> {
        // Try OpenDML super-index path.
        if let Some(frames) = self.frames_via_super_index() {
            return Ok(frames);
        }

        // Fall back to AVI 1.0 path (idx1 or linear scan).
        let (movi_list_offset, movi_payload_offset, movi_payload_len) =
            self.find_movi_in_primary()?;

        if let Some(idx1_offset) = self.find_idx1(movi_list_offset) {
            self.frames_via_idx1(idx1_offset, movi_list_offset)
        } else {
            self.frames_linear_scan(movi_payload_offset, movi_payload_len)
        }
    }

    /// Extract all `01wb` audio chunk payloads concatenated in order.
    ///
    /// Returns `None` if no audio data was found.
    pub fn audio_data(&self) -> Option<Vec<u8>> {
        let chunks = self
            .audio_chunks_via_super_index()
            .or_else(|| self.audio_chunks_linear());
        let chunks = chunks?;
        if chunks.is_empty() {
            return None;
        }
        let total: usize = chunks.iter().map(|c| c.len()).sum();
        let mut out = Vec::with_capacity(total);
        for c in chunks {
            out.extend_from_slice(&c);
        }
        Some(out)
    }

    /// Plans a sample-accurate seek cursor for the primary video stream.
    ///
    /// The returned cursor points to the nearest keyframe at or before
    /// `target_pts`, where PTS values are interpreted as 0-based frame numbers.
    ///
    /// # Errors
    ///
    /// Returns an error if the AVI container has no `movi` list or the index is malformed.
    pub fn seek_sample_accurate(&self, target_pts: u64) -> Result<DecodeSkipCursor, AviDemuxError> {
        let entries = if let Some(entries) = self.video_seek_entries_via_super_index() {
            entries
        } else {
            let (movi_list_offset, _, _) = self.find_movi_in_primary()?;
            let idx1_offset = self
                .find_idx1(movi_list_offset)
                .ok_or(AviDemuxError::NoMovi)?;
            self.video_seek_entries_via_idx1(idx1_offset, movi_list_offset)?
        };

        if entries.is_empty() {
            return Err(AviDemuxError::NoMovi);
        }

        let target_index = usize::try_from(target_pts).unwrap_or(usize::MAX);
        let bounded_target = target_index.min(entries.len().saturating_sub(1));
        let keyframe_index = (0..=bounded_target)
            .rev()
            .find(|&index| entries[index].is_keyframe)
            .unwrap_or(0);
        let skip_samples = u32::try_from(bounded_target.saturating_sub(keyframe_index))
            .map_err(|_| AviDemuxError::NoMovi)?;

        Ok(DecodeSkipCursor {
            byte_offset: u64::try_from(entries[keyframe_index].payload_offset).unwrap_or(u64::MAX),
            sample_index: keyframe_index,
            skip_samples,
            target_pts: i64::try_from(target_pts).unwrap_or(i64::MAX),
        })
    }

    // -----------------------------------------------------------------------
    // Audio format parsing
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    // OpenDML super-index path
    // -----------------------------------------------------------------------

    /// Try to extract video frames using OpenDML indx/ix## super-index.
    ///
    /// Returns `None` if no `indx` chunk is found (caller should fall back).
    fn frames_via_super_index(&self) -> Option<Vec<Vec<u8>>> {
        let video_indx_offset = find_indx_for_stream(&self.data, b"00dc")?;
        let indx_entries = parse_indx_entries(&self.data, video_indx_offset)?;
        let mut frames = Vec::new();
        for entry in &indx_entries {
            let ix_chunks = self.read_ix_chunk(entry.qw_offset as usize, b"00dc")?;
            frames.extend(ix_chunks);
        }
        Some(frames)
    }

    fn video_seek_entries_via_super_index(&self) -> Option<Vec<AviSeekEntry>> {
        let video_indx_offset = find_indx_for_stream(&self.data, b"00dc")?;
        let indx_entries = parse_indx_entries(&self.data, video_indx_offset)?;
        let mut entries = Vec::new();
        for entry in &indx_entries {
            entries.extend(self.read_ix_seek_entries(entry.qw_offset as usize, b"00dc")?);
        }
        Some(entries)
    }

    /// Try to extract audio chunks using OpenDML indx/ix## super-index.
    fn audio_chunks_via_super_index(&self) -> Option<Vec<Vec<u8>>> {
        let audio_indx_offset = find_indx_for_stream(&self.data, b"01wb")?;
        let indx_entries = parse_indx_entries(&self.data, audio_indx_offset)?;
        let mut chunks = Vec::new();
        for entry in &indx_entries {
            let ix_data = self.read_ix_chunk(entry.qw_offset as usize, b"01wb")?;
            chunks.extend(ix_data);
        }
        Some(chunks)
    }

    /// Read all chunk payloads from an `ix##` field-index at the given offset.
    fn read_ix_chunk(&self, ix_chunk_start: usize, ckid: &[u8; 4]) -> Option<Vec<Vec<u8>>> {
        let data = &self.data;
        if ix_chunk_start + 8 > data.len() {
            return None;
        }
        // Skip fourcc(4) + size(4).
        let payload_start = ix_chunk_start + 8;
        if payload_start + 24 > data.len() {
            return None;
        }

        // Parse header.
        // wLongsPerEntry(2) + bIndexSubType(1) + bIndexType(1) + nEntriesInUse(4) +
        // dwChunkId(4) + qwBaseOffset(8) + dwReserved(4) = 24 bytes
        let n_entries = read_u32_le(data, payload_start + 4) as usize;
        let base_offset = read_u64_le(data, payload_start + 12);
        let entries_start = payload_start + 24;

        if entries_start + n_entries * 8 > data.len() {
            return None;
        }

        let mut result = Vec::with_capacity(n_entries);
        for i in 0..n_entries {
            let e = entries_start + i * 8;
            // dwOffset: offset of chunk payload relative to qwBaseOffset.
            let offset = read_u32_le(data, e);
            // dwSize: payload size, bit 31 = keyframe flag.
            let size_raw = read_u32_le(data, e + 4);
            let payload_size = (size_raw & 0x7FFF_FFFF) as usize;

            // Absolute file position of the chunk payload.
            let abs_payload = (base_offset + u64::from(offset)) as usize;
            let abs_end = abs_payload.checked_add(payload_size)?;
            if abs_end > data.len() {
                continue;
            }
            // The ix entry offset points to the payload directly (past fourcc+size).
            // But we need to check if it's the right chunk type by peeking 8 bytes back.
            // Per OpenDML spec, qwBaseOffset is the absolute position of the `movi`
            // payload start.  Each ix entry's offset is relative to qwBaseOffset and
            // points to the *chunk payload* (past the 8-byte header).
            // So the actual chunk fourcc is at abs_payload - 8.
            let chunk_fourcc_pos = abs_payload.checked_sub(8)?;
            if chunk_fourcc_pos + 4 > data.len() {
                continue;
            }
            if &data[chunk_fourcc_pos..chunk_fourcc_pos + 4] == ckid {
                result.push(data[abs_payload..abs_end].to_vec());
            }
        }
        Some(result)
    }

    fn read_ix_seek_entries(
        &self,
        ix_chunk_start: usize,
        ckid: &[u8; 4],
    ) -> Option<Vec<AviSeekEntry>> {
        let data = &self.data;
        if ix_chunk_start + 8 > data.len() {
            return None;
        }
        let payload_start = ix_chunk_start + 8;
        if payload_start + 24 > data.len() {
            return None;
        }

        let n_entries = read_u32_le(data, payload_start + 4) as usize;
        let base_offset = read_u64_le(data, payload_start + 12);
        let entries_start = payload_start + 24;

        if entries_start + n_entries * 8 > data.len() {
            return None;
        }

        let mut result = Vec::with_capacity(n_entries);
        for i in 0..n_entries {
            let e = entries_start + i * 8;
            let offset = read_u32_le(data, e);
            let size_raw = read_u32_le(data, e + 4);
            let payload_size = (size_raw & 0x7FFF_FFFF) as usize;
            let abs_payload = (base_offset + u64::from(offset)) as usize;
            let abs_end = abs_payload.checked_add(payload_size)?;
            if abs_end > data.len() {
                continue;
            }
            let chunk_fourcc_pos = abs_payload.checked_sub(8)?;
            if chunk_fourcc_pos + 4 > data.len() {
                continue;
            }
            if &data[chunk_fourcc_pos..chunk_fourcc_pos + 4] == ckid {
                result.push(AviSeekEntry {
                    payload_offset: abs_payload,
                    is_keyframe: (size_raw & 0x8000_0000) == 0,
                });
            }
        }
        Some(result)
    }

    // -----------------------------------------------------------------------
    // Audio linear fallback
    // -----------------------------------------------------------------------

    fn audio_chunks_linear(&self) -> Option<Vec<Vec<u8>>> {
        // Collect 01wb chunks from all movi payloads (primary + secondary AVIX).
        let mut all_chunks = Vec::new();
        let data = &self.data;
        let mut riff_pos = 0usize;

        loop {
            if riff_pos + 12 > data.len() {
                break;
            }
            if &data[riff_pos..riff_pos + 4] != b"RIFF" {
                break;
            }
            let riff_type = &data[riff_pos + 8..riff_pos + 12];
            if riff_type != b"AVI " && riff_type != b"AVIX" {
                break;
            }

            // Find movi LIST in this RIFF segment.
            let riff_size = read_u32_le(data, riff_pos + 4) as usize;
            let riff_end = (riff_pos + 8 + riff_size).min(data.len());
            let search_start = riff_pos + 12;

            if let Some((_, payload_start, payload_len)) =
                find_movi_in_range(data, search_start, riff_end)
            {
                let mut pos = payload_start;
                let end = (payload_start + payload_len).min(data.len());
                while pos + 8 <= end {
                    let fourcc = &data[pos..pos + 4];
                    let size = read_u32_le(data, pos + 4) as usize;
                    if fourcc == b"01wb" {
                        let pstart = pos + 8;
                        let pend = pstart + size;
                        if pend <= end {
                            all_chunks.push(data[pstart..pend].to_vec());
                        }
                    }
                    let advance = 8 + size + (size % 2);
                    pos = match pos.checked_add(advance) {
                        Some(p) => p,
                        None => break,
                    };
                }
            }

            // Advance to next RIFF chunk.
            let advance = 8 + riff_size;
            riff_pos = match riff_pos.checked_add(advance) {
                Some(p) => p,
                None => break,
            };
        }

        if all_chunks.is_empty() {
            None
        } else {
            Some(all_chunks)
        }
    }

    // -----------------------------------------------------------------------
    // AVI 1.0 helpers
    // -----------------------------------------------------------------------

    /// Find the primary `LIST movi` chunk inside `RIFF AVI `.
    fn find_movi_in_primary(&self) -> Result<(usize, usize, usize), AviDemuxError> {
        let data = &self.data;
        let riff_size = read_u32_le(data, 4) as usize;
        let riff_end = (12 + riff_size).min(data.len());
        find_movi_in_range(data, 12, riff_end).ok_or(AviDemuxError::NoMovi)
    }

    /// Returns the byte offset of the `idx1` chunk after the given movi.
    fn find_idx1(&self, movi_list_offset: usize) -> Option<usize> {
        let movi_total = read_u32_le(&self.data, movi_list_offset + 4) as usize + 8;
        let after_movi = movi_list_offset + movi_total;
        let mut pos = after_movi;
        let data = &self.data;
        while pos + 8 <= data.len() {
            let fourcc = &data[pos..pos + 4];
            let size = read_u32_le(data, pos + 4) as usize;
            if fourcc == b"idx1" {
                return Some(pos);
            }
            let advance = 8 + size + (size % 2);
            pos = match pos.checked_add(advance) {
                Some(p) => p,
                None => break,
            };
        }
        None
    }

    /// Extract video frames using idx1.
    fn frames_via_idx1(
        &self,
        idx1_offset: usize,
        movi_list_offset: usize,
    ) -> Result<Vec<Vec<u8>>, AviDemuxError> {
        let data = &self.data;
        let idx1_size = read_u32_le(data, idx1_offset + 4);
        if idx1_size % 16 != 0 {
            return Err(AviDemuxError::MalformedIdx1(idx1_size));
        }
        let entries_start = idx1_offset + 8;
        let entry_count = idx1_size as usize / 16;
        let mut frames = Vec::with_capacity(entry_count);
        let movi_fourcc_pos = mowi_fourcc_pos(movi_list_offset);

        for i in 0..entry_count {
            let e = entries_start + i * 16;
            if e + 16 > data.len() {
                break;
            }
            if &data[e..e + 4] != b"00dc" {
                continue;
            }
            let chunk_offset = read_u32_le(data, e + 8);
            let chunk_len = read_u32_le(data, e + 12);
            let chunk_pos = movi_fourcc_pos
                .checked_add(chunk_offset as usize)
                .ok_or(AviDemuxError::IndexOutOfBounds(chunk_offset, chunk_len))?;
            let payload_start = chunk_pos + 8;
            let payload_end = payload_start
                .checked_add(chunk_len as usize)
                .ok_or(AviDemuxError::IndexOutOfBounds(chunk_offset, chunk_len))?;
            if payload_end > data.len() {
                return Err(AviDemuxError::IndexOutOfBounds(chunk_offset, chunk_len));
            }
            frames.push(data[payload_start..payload_end].to_vec());
        }
        Ok(frames)
    }

    fn video_seek_entries_via_idx1(
        &self,
        idx1_offset: usize,
        movi_list_offset: usize,
    ) -> Result<Vec<AviSeekEntry>, AviDemuxError> {
        let data = &self.data;
        let idx1_size = read_u32_le(data, idx1_offset + 4);
        if idx1_size % 16 != 0 {
            return Err(AviDemuxError::MalformedIdx1(idx1_size));
        }

        let entries_start = idx1_offset + 8;
        let entry_count = idx1_size as usize / 16;
        let movi_fourcc_pos = mowi_fourcc_pos(movi_list_offset);
        let mut entries = Vec::new();

        for i in 0..entry_count {
            let e = entries_start + i * 16;
            if e + 16 > data.len() {
                break;
            }
            if &data[e..e + 4] != b"00dc" {
                continue;
            }

            let flags = read_u32_le(data, e + 4);
            let chunk_offset = read_u32_le(data, e + 8);
            let chunk_len = read_u32_le(data, e + 12);
            let chunk_pos = movi_fourcc_pos
                .checked_add(chunk_offset as usize)
                .ok_or(AviDemuxError::IndexOutOfBounds(chunk_offset, chunk_len))?;
            let payload_offset = chunk_pos + 8;
            let payload_end = payload_offset
                .checked_add(chunk_len as usize)
                .ok_or(AviDemuxError::IndexOutOfBounds(chunk_offset, chunk_len))?;
            if payload_end > data.len() {
                return Err(AviDemuxError::IndexOutOfBounds(chunk_offset, chunk_len));
            }

            entries.push(AviSeekEntry {
                payload_offset,
                is_keyframe: (flags & 0x10) != 0,
            });
        }

        Ok(entries)
    }

    /// Extract video frames by linearly scanning the movi chunk.
    fn frames_linear_scan(
        &self,
        payload_start: usize,
        payload_len: usize,
    ) -> Result<Vec<Vec<u8>>, AviDemuxError> {
        let data = &self.data;
        let payload_end = (payload_start + payload_len).min(data.len());
        let mut pos = payload_start;
        let mut frames = Vec::new();
        while pos + 8 <= payload_end {
            let fourcc = &data[pos..pos + 4];
            let size = read_u32_le(data, pos + 4) as usize;
            if fourcc == b"00dc" {
                let frame_start = pos + 8;
                let frame_end = frame_start + size;
                if frame_end <= payload_end {
                    frames.push(data[frame_start..frame_end].to_vec());
                }
            }
            let advance = 8 + size + (size % 2);
            pos = match pos.checked_add(advance) {
                Some(p) => p,
                None => break,
            };
        }
        Ok(frames)
    }
}

// ============================================================================
// Audio format parsing helpers
// ============================================================================

fn parse_audio_format(data: &[u8]) -> Option<AviAudioFormat> {
    // Walk hdrl to find the audio strl (second strl in hdrl, or any strl with auds strh).
    let hdrl = find_list(data, 12, data.len(), b"hdrl")?;
    let (hdrl_payload_start, hdrl_payload_end) = hdrl;

    // Walk strl LIST chunks inside hdrl.
    let mut pos = hdrl_payload_start;
    while pos + 8 <= hdrl_payload_end {
        let fourcc = &data[pos..pos + 4];
        let size = read_u32_le(data, pos + 4) as usize;
        if fourcc == b"LIST" && pos + 12 <= hdrl_payload_end {
            let list_type = &data[pos + 8..pos + 12];
            if list_type == b"strl" {
                let strl_payload_start = pos + 12;
                let strl_payload_end = (pos + 8 + size).min(hdrl_payload_end);
                // Check if this strl has an `auds` strh.
                if strl_has_auds(data, strl_payload_start, strl_payload_end) {
                    return parse_strf_waveformatex(data, strl_payload_start, strl_payload_end);
                }
            }
        }
        let advance = 8 + size + (size % 2);
        pos = match pos.checked_add(advance) {
            Some(p) => p,
            None => break,
        };
    }
    None
}

/// Returns true if the strl payload contains a `strh` with `fccType = auds`.
fn strl_has_auds(data: &[u8], start: usize, end: usize) -> bool {
    let mut pos = start;
    while pos + 8 <= end {
        let fourcc = &data[pos..pos + 4];
        let size = read_u32_le(data, pos + 4) as usize;
        if fourcc == b"strh" && pos + 12 <= end {
            return &data[pos + 8..pos + 12] == b"auds";
        }
        let advance = 8 + size + (size % 2);
        pos = match pos.checked_add(advance) {
            Some(p) => p,
            None => break,
        };
    }
    false
}

/// Parse a `WAVEFORMATEX` from the `strf` chunk inside a strl payload.
fn parse_strf_waveformatex(
    data: &[u8],
    strl_start: usize,
    strl_end: usize,
) -> Option<AviAudioFormat> {
    let mut pos = strl_start;
    while pos + 8 <= strl_end {
        let fourcc = &data[pos..pos + 4];
        let size = read_u32_le(data, pos + 4) as usize;
        if fourcc == b"strf" {
            let payload_start = pos + 8;
            if payload_start + 14 <= strl_end {
                let channels = read_u16_le(data, payload_start + 2);
                let sample_rate = read_u32_le(data, payload_start + 4);
                let bits_per_sample = read_u16_le(data, payload_start + 14);
                return Some(AviAudioFormat {
                    sample_rate,
                    channels,
                    bits_per_sample,
                });
            }
        }
        let advance = 8 + size + (size % 2);
        pos = match pos.checked_add(advance) {
            Some(p) => p,
            None => break,
        };
    }
    None
}

// ============================================================================
// OpenDML super-index helpers
// ============================================================================

/// Parsed `indx` super-index entry.
struct SuperIndexEntry {
    qw_offset: u64,
    _dw_size: u32,
    _dw_duration: u32,
}

struct AviSeekEntry {
    payload_offset: usize,
    is_keyframe: bool,
}

/// Find the `indx` chunk for the given stream (by matching dwChunkId).
///
/// Searches inside all `strl` LIST chunks in `hdrl`.
fn find_indx_for_stream(data: &[u8], ckid: &[u8; 4]) -> Option<usize> {
    let (hdrl_start, hdrl_end) = find_list(data, 12, data.len(), b"hdrl")?;
    let mut pos = hdrl_start;
    while pos + 8 <= hdrl_end {
        let fourcc = &data[pos..pos + 4];
        let size = read_u32_le(data, pos + 4) as usize;
        if fourcc == b"LIST" && pos + 12 <= hdrl_end {
            let list_type = &data[pos + 8..pos + 12];
            if list_type == b"strl" {
                let strl_start = pos + 12;
                let strl_end = (pos + 8 + size).min(hdrl_end);
                if let Some(indx_off) = find_indx_with_ckid(data, strl_start, strl_end, ckid) {
                    return Some(indx_off);
                }
            }
        }
        let advance = 8 + size + (size % 2);
        pos = match pos.checked_add(advance) {
            Some(p) => p,
            None => break,
        };
    }
    None
}

/// Find an `indx` chunk within a strl whose dwChunkId matches `ckid`.
fn find_indx_with_ckid(data: &[u8], start: usize, end: usize, ckid: &[u8; 4]) -> Option<usize> {
    let mut pos = start;
    while pos + 8 <= end {
        let fourcc = &data[pos..pos + 4];
        let size = read_u32_le(data, pos + 4) as usize;
        if fourcc == b"indx" {
            // dwChunkId is at payload offset 8 (after wLongsPerEntry(2)+subtype(1)+type(1)+n(4)).
            let payload = pos + 8;
            if payload + 12 <= end && &data[payload + 8..payload + 12] == ckid {
                return Some(pos);
            }
        }
        let advance = 8 + size + (size % 2);
        pos = match pos.checked_add(advance) {
            Some(p) => p,
            None => break,
        };
    }
    None
}

/// Parse all entries from an `indx` chunk at `indx_chunk_offset`.
fn parse_indx_entries(data: &[u8], indx_chunk_offset: usize) -> Option<Vec<SuperIndexEntry>> {
    let payload_start = indx_chunk_offset + 8;
    if payload_start + 24 > data.len() {
        return None;
    }
    let n = read_u32_le(data, payload_start + 4) as usize;
    let entries_start = payload_start + 24;
    if entries_start + n * 16 > data.len() {
        return None;
    }
    let mut entries = Vec::with_capacity(n);
    for i in 0..n {
        let e = entries_start + i * 16;
        let qw_offset = read_u64_le(data, e);
        let dw_size = read_u32_le(data, e + 8);
        let dw_duration = read_u32_le(data, e + 12);
        entries.push(SuperIndexEntry {
            qw_offset,
            _dw_size: dw_size,
            _dw_duration: dw_duration,
        });
    }
    Some(entries)
}

// ============================================================================
// Chunk-walking helpers
// ============================================================================

/// Returns `(payload_start, payload_end)` for a LIST of the given type
/// found within `data[start..end]`.
fn find_list(data: &[u8], start: usize, end: usize, list_type: &[u8; 4]) -> Option<(usize, usize)> {
    let mut pos = start;
    while pos + 8 <= end {
        let fourcc = &data[pos..pos + 4];
        let size = read_u32_le(data, pos + 4) as usize;
        if fourcc == b"LIST" && pos + 12 <= end {
            let lt = &data[pos + 8..pos + 12];
            if lt == list_type {
                let payload_start = pos + 12;
                let payload_end = (pos + 8 + size).min(end);
                return Some((payload_start, payload_end));
            }
        }
        let advance = 8 + size + (size % 2);
        pos = match pos.checked_add(advance) {
            Some(p) => p,
            None => break,
        };
    }
    None
}

/// Returns `(list_offset, payload_start, payload_len)` for `LIST movi` within range.
fn find_movi_in_range(data: &[u8], start: usize, end: usize) -> Option<(usize, usize, usize)> {
    let mut pos = start;
    while pos + 8 <= end {
        let fourcc = &data[pos..pos + 4];
        let size = read_u32_le(data, pos + 4) as usize;
        if fourcc == b"LIST" && pos + 12 <= end {
            let list_type = &data[pos + 8..pos + 12];
            if list_type == b"movi" {
                let payload_start = pos + 12;
                let payload_len = size.saturating_sub(4);
                return Some((pos, payload_start, payload_len));
            }
        }
        let advance = 8 + size + (size % 2);
        pos = match pos.checked_add(advance) {
            Some(p) => p,
            None => break,
        };
    }
    None
}

/// Returns the absolute byte position of the `movi` four-char code in the file.
///
/// Layout: `LIST`(4) + size(4) + `mowi`(4) — so movi fourcc is at `movi_list_offset + 8`.
fn mowi_fourcc_pos(movi_list_offset: usize) -> usize {
    movi_list_offset + 8
}

// ============================================================================
// Byte-reading helpers
// ============================================================================

#[inline]
fn read_u32_le(data: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ])
}

#[inline]
fn read_u16_le(data: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([data[offset], data[offset + 1]])
}

#[inline]
fn read_u64_le(data: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
        data[offset + 4],
        data[offset + 5],
        data[offset + 6],
        data[offset + 7],
    ])
}

// ============================================================================
// Unit tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_too_short_input() {
        let result = AviMjpegReader::new(vec![0u8; 8]);
        assert!(matches!(result, Err(AviDemuxError::TooShort)));
    }

    #[test]
    fn rejects_invalid_signature() {
        let mut data = vec![0u8; 12];
        data[0..4].copy_from_slice(b"WAVE");
        let result = AviMjpegReader::new(data);
        assert!(matches!(result, Err(AviDemuxError::InvalidSignature)));
    }

    #[test]
    fn rejects_riff_non_avi() {
        let mut data = vec![0u8; 12];
        data[0..4].copy_from_slice(b"RIFF");
        data[8..12].copy_from_slice(b"WAVE");
        let result = AviMjpegReader::new(data);
        assert!(matches!(result, Err(AviDemuxError::InvalidSignature)));
    }

    #[test]
    fn empty_avi_no_movi() {
        let mut data = Vec::new();
        data.extend_from_slice(b"RIFF");
        let payload = b"AVI ";
        let riff_size = (payload.len() as u32).to_le_bytes();
        data.extend_from_slice(&riff_size);
        data.extend_from_slice(payload);
        let reader = AviMjpegReader::new(data).expect("signature valid");
        let result = reader.frames();
        assert!(matches!(result, Err(AviDemuxError::NoMovi)));
    }
}
