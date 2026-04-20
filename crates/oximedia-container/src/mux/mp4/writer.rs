//! MP4 muxer writer implementation.
//!
//! Generates ISOBMFF-compliant MP4 files with moov/mdat layout for
//! progressive MP4, or moov/moof+mdat for fragmented MP4 (fMP4).
//!
//! # Fragmented MP4 box layout
//!
//! ```text
//! [ftyp][moov[mvhd][mvex[trex…]][trak(empty stbl)…]]
//!   ([sidx][moof[mfhd][traf[tfhd][tfdt][trun]]][mdat])…
//! ```

#![forbid(unsafe_code)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use oximedia_core::{CodecId, MediaType, OxiError, OxiResult, Rational};

use super::av1c::build_av1c_from_extradata;
use crate::mux::cmaf::{
    build_empty_stco, build_empty_stsc, build_empty_stsz, build_empty_stts, build_mfhd, build_traf,
    write_box, write_full_box, write_u32_be, write_u64_be, FragSampleEntry, FragTrackRun,
};
use crate::{Packet, StreamInfo};

// ─── Constants ──────────────────────────────────────────────────────────────

/// Default timescale for the movie header (ticks per second).
const MOVIE_TIMESCALE: u32 = 1000;

/// Maximum supported tracks.
const MAX_TRACKS: usize = 16;

// ─── Configuration ──────────────────────────────────────────────────────────

/// MP4 fragment/muxing mode.
///
/// Controls whether the output is a standard progressive MP4 (single `moov`+`mdat`)
/// or a fragmented MP4 (`moov` init segment + repeating `sidx`+`moof`+`mdat` fragments).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mp4FragmentMode {
    /// Progressive MP4: single `moov` + single `mdat`.
    ///
    /// All sample metadata is collected in memory and written to `moov` at finalize time.
    Progressive,
    /// Fragmented MP4 (fMP4/CMAF-compatible): `moov` init segment + `sidx`+`moof`+`mdat` fragments.
    ///
    /// The `moov` init segment contains empty `stbl` tables and an `mvex`/`trex` extension box.
    /// Fragments are split on keyframe boundaries, honouring the requested target duration.
    ///
    /// `fragment_duration_ms` is the target fragment duration in milliseconds.  A value of 0
    /// means "one fragment per keyframe boundary".
    Fragmented {
        /// Target fragment duration in milliseconds.
        fragment_duration_ms: u32,
    },
}

impl Default for Mp4FragmentMode {
    fn default() -> Self {
        Self::Progressive
    }
}

/// Backward-compatible type alias — prefer [`Mp4FragmentMode`] in new code.
pub type Mp4Mode = Mp4FragmentMode;

/// Configuration for the MP4 muxer.
#[derive(Debug, Clone)]
pub struct Mp4Config {
    /// Muxing mode (progressive or fragmented).
    ///
    /// For fragmented mode use `Mp4FragmentMode::Fragmented { fragment_duration_ms }`.
    pub mode: Mp4FragmentMode,
    /// Major brand for ftyp box.
    pub major_brand: [u8; 4],
    /// Minor version for ftyp box.
    pub minor_version: u32,
    /// Compatible brands for ftyp box.
    pub compatible_brands: Vec<[u8; 4]>,
    /// Creation time (seconds since 1904-01-01).
    pub creation_time: u64,
    /// Modification time (seconds since 1904-01-01).
    pub modification_time: u64,
}

impl Default for Mp4Config {
    fn default() -> Self {
        Self {
            mode: Mp4FragmentMode::Progressive,
            major_brand: *b"isom",
            minor_version: 0x200,
            compatible_brands: vec![*b"isom", *b"iso6", *b"mp41"],
            creation_time: 0,
            modification_time: 0,
        }
    }
}

impl Mp4Config {
    /// Creates a new `Mp4Config` with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the muxing mode.
    #[must_use]
    pub fn with_mode(mut self, mode: Mp4FragmentMode) -> Self {
        self.mode = mode;
        self
    }

    /// Convenience builder: configure fragmented mode with the given target
    /// fragment duration in milliseconds.
    ///
    /// Equivalent to `.with_mode(Mp4FragmentMode::Fragmented { fragment_duration_ms: ms })`.
    #[must_use]
    pub fn with_fragmented(mut self, fragment_duration_ms: u32) -> Self {
        self.mode = Mp4FragmentMode::Fragmented {
            fragment_duration_ms,
        };
        self
    }

    /// Sets the major brand.
    #[must_use]
    pub const fn with_major_brand(mut self, brand: [u8; 4]) -> Self {
        self.major_brand = brand;
        self
    }

    /// Sets the minor version.
    #[must_use]
    pub const fn with_minor_version(mut self, version: u32) -> Self {
        self.minor_version = version;
        self
    }

    /// Adds a compatible brand.
    #[must_use]
    pub fn with_compatible_brand(mut self, brand: [u8; 4]) -> Self {
        self.compatible_brands.push(brand);
        self
    }
}

// ─── Sample Entry ───────────────────────────────────────────────────────────

/// Metadata for a single sample in a track.
#[derive(Debug, Clone)]
pub struct Mp4SampleEntry {
    /// Sample size in bytes.
    pub size: u32,
    /// Sample duration in the track's timescale.
    pub duration: u32,
    /// Composition time offset (PTS - DTS) in the track's timescale.
    pub composition_offset: i32,
    /// Whether this sample is a sync (key) sample.
    pub is_sync: bool,
    /// Chunk index this sample belongs to.
    pub chunk_index: u32,
}

// ─── Track State ────────────────────────────────────────────────────────────

/// Per-track state maintained during muxing.
#[derive(Debug, Clone)]
pub struct Mp4TrackState {
    /// Stream information.
    pub stream_info: StreamInfo,
    /// Track ID (1-based).
    pub track_id: u32,
    /// Timescale for this track.
    pub timescale: u32,
    /// Accumulated sample entries.
    pub samples: Vec<Mp4SampleEntry>,
    /// Raw mdat data for this track.
    pub mdat_data: Vec<u8>,
    /// Current chunk boundaries (sample indices).
    pub chunk_boundaries: Vec<u32>,
    /// Number of samples in the current chunk.
    pub current_chunk_samples: u32,
    /// Maximum samples per chunk before starting a new one.
    pub max_samples_per_chunk: u32,
    /// Total duration in timescale units.
    pub total_duration: u64,
    /// Handler type identifier.
    pub handler_type: [u8; 4],
    /// Handler name.
    pub handler_name: String,
}

impl Mp4TrackState {
    fn new(stream_info: StreamInfo, track_id: u32) -> Self {
        let (timescale, handler_type, handler_name) = match stream_info.media_type {
            MediaType::Video => {
                let ts = if stream_info.timebase.den != 0 {
                    // Use timebase denominator as timescale
                    stream_info.timebase.den as u32
                } else {
                    90000
                };
                (ts, *b"vide", "OxiMedia Video Handler".to_string())
            }
            MediaType::Audio => {
                let ts = stream_info.codec_params.sample_rate.unwrap_or(48000);
                (ts, *b"soun", "OxiMedia Audio Handler".to_string())
            }
            _ => (1000, *b"text", "OxiMedia Text Handler".to_string()),
        };

        Self {
            stream_info,
            track_id,
            timescale,
            samples: Vec::new(),
            mdat_data: Vec::new(),
            chunk_boundaries: vec![0],
            current_chunk_samples: 0,
            max_samples_per_chunk: 10,
            total_duration: 0,
            handler_type,
            handler_name,
        }
    }

    fn add_sample(&mut self, data: &[u8], duration: u32, comp_offset: i32, is_sync: bool) {
        let chunk_index = self.chunk_boundaries.len().saturating_sub(1) as u32;
        self.samples.push(Mp4SampleEntry {
            size: data.len() as u32,
            duration,
            composition_offset: comp_offset,
            is_sync,
            chunk_index,
        });
        self.mdat_data.extend_from_slice(data);
        self.total_duration += u64::from(duration);
        self.current_chunk_samples += 1;

        if self.current_chunk_samples >= self.max_samples_per_chunk {
            self.chunk_boundaries.push(self.samples.len() as u32);
            self.current_chunk_samples = 0;
        }
    }
}

// ─── MP4 Muxer ──────────────────────────────────────────────────────────────

/// MP4/ISOBMFF muxer.
///
/// Builds a valid ISOBMFF file in memory with proper box hierarchy.
/// Supports both progressive and fragmented modes.
#[derive(Debug)]
pub struct Mp4Muxer {
    /// Configuration.
    config: Mp4Config,
    /// Per-track states.
    tracks: Vec<Mp4TrackState>,
    /// Whether the header has been written.
    header_written: bool,
    /// Fragment sequence number (for fragmented mode).
    fragment_sequence: u32,
    /// Completed fragments (for fragmented mode).
    fragments: Vec<Vec<u8>>,
}

impl Mp4Muxer {
    /// Creates a new MP4 muxer.
    #[must_use]
    pub fn new(config: Mp4Config) -> Self {
        Self {
            config,
            tracks: Vec::new(),
            header_written: false,
            fragment_sequence: 1,
            fragments: Vec::new(),
        }
    }

    /// Adds a stream/track to the muxer. Returns the track index.
    ///
    /// # Errors
    ///
    /// Returns an error if the codec is not supported or the maximum number
    /// of tracks has been exceeded.
    pub fn add_stream(&mut self, info: StreamInfo) -> OxiResult<usize> {
        if self.header_written {
            return Err(OxiError::parse(
                0,
                "Cannot add streams after header is written",
            ));
        }
        if self.tracks.len() >= MAX_TRACKS {
            return Err(OxiError::parse(0, "Maximum number of tracks exceeded"));
        }

        // Validate codec is patent-free
        validate_codec(info.codec)?;

        let track_id = (self.tracks.len() + 1) as u32;
        self.tracks.push(Mp4TrackState::new(info, track_id));
        Ok(self.tracks.len() - 1)
    }

    /// Writes the file type box. Call after adding all streams.
    ///
    /// # Errors
    ///
    /// Returns an error if no streams have been added.
    pub fn write_header(&mut self) -> OxiResult<()> {
        if self.tracks.is_empty() {
            return Err(OxiError::parse(0, "No streams added"));
        }
        self.header_written = true;
        Ok(())
    }

    /// Writes a packet to the appropriate track.
    ///
    /// # Errors
    ///
    /// Returns an error if the stream index is invalid or the header
    /// has not been written.
    pub fn write_packet(&mut self, packet: &Packet) -> OxiResult<()> {
        if !self.header_written {
            return Err(OxiError::parse(0, "Header not written yet"));
        }

        let track_idx = packet.stream_index;
        if track_idx >= self.tracks.len() {
            return Err(OxiError::parse(
                0,
                format!("Invalid stream index {track_idx}"),
            ));
        }

        let track = &self.tracks[track_idx];
        let timescale = track.timescale;

        // Calculate sample duration from packet
        let duration = packet
            .duration()
            .map(|d| convert_duration(d, &track.stream_info.timebase, timescale))
            .unwrap_or(1);

        // Composition offset: PTS - DTS
        let comp_offset = if let Some(dts) = packet.dts() {
            let pts_ticks = convert_duration(packet.pts(), &track.stream_info.timebase, timescale);
            let dts_ticks = convert_duration(dts, &track.stream_info.timebase, timescale);
            (pts_ticks as i64 - dts_ticks as i64) as i32
        } else {
            0
        };

        let is_sync = packet.is_keyframe();

        let track = &mut self.tracks[track_idx];
        track.add_sample(&packet.data, duration, comp_offset, is_sync);

        Ok(())
    }

    /// Finalizes the MP4 file and returns the complete byte output.
    ///
    /// For progressive mode, this assembles ftyp + moov + mdat.
    /// For fragmented mode, this assembles ftyp + moov (with mvex) + fragments.
    ///
    /// # Errors
    ///
    /// Returns an error if finalization fails.
    pub fn finalize(&self) -> OxiResult<Vec<u8>> {
        if !self.header_written {
            return Err(OxiError::parse(0, "Header not written"));
        }

        let mut output = Vec::new();

        // 1. ftyp box
        output.extend(self.build_ftyp());

        match self.config.mode {
            Mp4FragmentMode::Progressive => {
                // 2. Collect all mdat data with offsets
                let (mdat_box, chunk_offsets) = self.build_mdat_progressive();

                // 3. moov box (needs chunk offsets, which depend on ftyp + moov size)
                // We need to do a two-pass: first estimate moov size, then recalc offsets
                let ftyp_size = output.len() as u64;
                let moov_estimate = self.build_moov_progressive(&chunk_offsets, ftyp_size);
                let moov_size = moov_estimate.len() as u64;

                // Recalculate offsets with correct moov size
                let mdat_start = ftyp_size + moov_size + 8; // +8 for mdat header
                let adjusted_offsets = adjust_chunk_offsets(&chunk_offsets, mdat_start);

                let moov_final = self.build_moov_progressive(&adjusted_offsets, ftyp_size);

                output.extend(moov_final);
                output.extend(mdat_box);
            }
            Mp4FragmentMode::Fragmented { .. } => {
                // moov with mvex
                let moov = self.build_moov_fragmented();
                output.extend(moov);

                // Append all cached fragments
                for frag in &self.fragments {
                    output.extend(frag);
                }

                // Build fragments from pending samples
                let frags = self.build_fragments_from_tracks();
                for frag in frags {
                    output.extend(frag);
                }
            }
        }

        Ok(output)
    }

    /// Returns the number of tracks.
    #[must_use]
    pub fn track_count(&self) -> usize {
        self.tracks.len()
    }

    /// Returns a reference to a track state by index.
    #[must_use]
    pub fn track(&self, index: usize) -> Option<&Mp4TrackState> {
        self.tracks.get(index)
    }

    /// Returns the total number of samples across all tracks.
    #[must_use]
    pub fn total_samples(&self) -> usize {
        self.tracks.iter().map(|t| t.samples.len()).sum()
    }

    // ─── Box builders ───────────────────────────────────────────────────

    fn build_ftyp(&self) -> Vec<u8> {
        let mut content = Vec::new();
        content.extend_from_slice(&self.config.major_brand);
        content.extend_from_slice(&write_u32_be(self.config.minor_version));
        for brand in &self.config.compatible_brands {
            content.extend_from_slice(brand);
        }
        write_box(b"ftyp", &content)
    }

    fn build_mdat_progressive(&self) -> (Vec<u8>, Vec<Vec<u64>>) {
        let mut mdat_payload = Vec::new();
        let mut all_chunk_offsets: Vec<Vec<u64>> = Vec::new();

        for track in &self.tracks {
            let mut track_offsets = Vec::new();
            let mut sample_idx: u32 = 0;
            let mut current_offset = mdat_payload.len() as u64;

            // Process each chunk
            for (chunk_idx, &chunk_start) in track.chunk_boundaries.iter().enumerate() {
                let chunk_end = track
                    .chunk_boundaries
                    .get(chunk_idx + 1)
                    .copied()
                    .unwrap_or(track.samples.len() as u32);

                track_offsets.push(current_offset);

                for si in chunk_start..chunk_end {
                    if let Some(sample) = track.samples.get(si as usize) {
                        let start = sample_data_offset(track, sample_idx);
                        let end = start + sample.size as usize;
                        if end <= track.mdat_data.len() {
                            mdat_payload.extend_from_slice(&track.mdat_data[start..end]);
                            current_offset += u64::from(sample.size);
                        }
                    }
                    sample_idx += 1;
                }
            }

            all_chunk_offsets.push(track_offsets);
        }

        let mdat = write_box(b"mdat", &mdat_payload);
        (mdat, all_chunk_offsets)
    }

    fn build_moov_progressive(&self, chunk_offsets: &[Vec<u64>], _ftyp_size: u64) -> Vec<u8> {
        let mut content = Vec::new();

        // mvhd
        content.extend(self.build_mvhd());

        // trak boxes
        for (i, track) in self.tracks.iter().enumerate() {
            let offsets = chunk_offsets.get(i).map_or(&[][..], |v| v.as_slice());
            content.extend(self.build_trak(track, offsets));
        }

        write_box(b"moov", &content)
    }

    fn build_moov_fragmented(&self) -> Vec<u8> {
        let mut content = Vec::new();
        // mvhd with duration=0 (indeterminate for fragmented)
        content.extend(self.build_mvhd_fragmented());

        // mvex with trex for each track
        let mut mvex_content = Vec::new();
        for track in &self.tracks {
            mvex_content.extend(build_trex(track.track_id));
        }
        content.extend(write_box(b"mvex", &mvex_content));

        // trak boxes with EMPTY stbl tables (no sample data in init segment)
        for track in &self.tracks {
            content.extend(self.build_trak_fragmented(track));
        }

        write_box(b"moov", &content)
    }

    /// Builds an `mvhd` with duration=0 (indeterminate, for fragmented mode).
    fn build_mvhd_fragmented(&self) -> Vec<u8> {
        let mut c = Vec::new();
        c.extend_from_slice(&write_u32_be(self.config.creation_time as u32));
        c.extend_from_slice(&write_u32_be(self.config.modification_time as u32));
        c.extend_from_slice(&write_u32_be(MOVIE_TIMESCALE));
        // duration = 0 for fragmented (unknown at init time)
        c.extend_from_slice(&write_u32_be(0));
        c.extend_from_slice(&write_u32_be(0x0001_0000)); // rate 1.0
        c.extend_from_slice(&[0x01, 0x00]); // volume 1.0
        c.extend_from_slice(&[0u8; 10]); // reserved
        c.extend_from_slice(&IDENTITY_MATRIX);
        c.extend_from_slice(&[0u8; 24]); // pre_defined
        c.extend_from_slice(&write_u32_be((self.tracks.len() + 1) as u32)); // next_track_id
        write_full_box(b"mvhd", 0, 0, &c)
    }

    /// Builds a `trak` box suitable for a fragmented init segment.
    ///
    /// The `stbl` contains only an `stsd` (sample description) plus the four
    /// mandatory empty sub-boxes — no sample timing, size, or chunk-offset data.
    fn build_trak_fragmented(&self, track: &Mp4TrackState) -> Vec<u8> {
        let mut content = Vec::new();
        content.extend(build_tkhd(track));
        content.extend(self.build_mdia_fragmented(track));
        write_box(b"trak", &content)
    }

    fn build_mdia_fragmented(&self, track: &Mp4TrackState) -> Vec<u8> {
        let mut content = Vec::new();
        content.extend(build_mdhd(track));
        content.extend(build_hdlr(track));
        content.extend(self.build_minf_fragmented(track));
        write_box(b"mdia", &content)
    }

    fn build_minf_fragmented(&self, track: &Mp4TrackState) -> Vec<u8> {
        let mut content = Vec::new();
        match track.stream_info.media_type {
            MediaType::Video => content.extend(build_vmhd()),
            MediaType::Audio => content.extend(build_smhd()),
            _ => content.extend(write_full_box(b"nmhd", 0, 0, &[])),
        }
        content.extend(build_dinf());
        content.extend(self.build_stbl_fragmented(track));
        write_box(b"minf", &content)
    }

    fn build_stbl_fragmented(&self, track: &Mp4TrackState) -> Vec<u8> {
        let mut content = Vec::new();
        // stsd must be present with proper sample entry
        content.extend(build_stsd(track));
        // Four mandatory empty tables (required by ISO 14496-12 §8.6.1 for fragmented)
        content.extend(build_empty_stts());
        content.extend(build_empty_stsc());
        content.extend(build_empty_stsz());
        content.extend(build_empty_stco());
        write_box(b"stbl", &content)
    }

    fn build_mvhd(&self) -> Vec<u8> {
        let mut c = Vec::new();
        // creation_time
        c.extend_from_slice(&write_u32_be(self.config.creation_time as u32));
        // modification_time
        c.extend_from_slice(&write_u32_be(self.config.modification_time as u32));
        // timescale
        c.extend_from_slice(&write_u32_be(MOVIE_TIMESCALE));
        // duration
        let max_dur = self
            .tracks
            .iter()
            .map(|t| {
                if t.timescale == 0 {
                    0
                } else {
                    t.total_duration * u64::from(MOVIE_TIMESCALE) / u64::from(t.timescale)
                }
            })
            .max()
            .unwrap_or(0);
        c.extend_from_slice(&write_u32_be(max_dur as u32));
        // rate (1.0 as fixed-point 16.16)
        c.extend_from_slice(&write_u32_be(0x0001_0000));
        // volume (1.0 as fixed-point 8.8)
        c.extend_from_slice(&[0x01, 0x00]);
        // reserved (10 bytes)
        c.extend_from_slice(&[0u8; 10]);
        // matrix (identity, 36 bytes)
        c.extend_from_slice(&IDENTITY_MATRIX);
        // pre_defined (24 bytes)
        c.extend_from_slice(&[0u8; 24]);
        // next_track_id
        c.extend_from_slice(&write_u32_be((self.tracks.len() + 1) as u32));

        write_full_box(b"mvhd", 0, 0, &c)
    }

    fn build_trak(&self, track: &Mp4TrackState, chunk_offsets: &[u64]) -> Vec<u8> {
        let mut content = Vec::new();
        content.extend(build_tkhd(track));
        content.extend(self.build_mdia(track, chunk_offsets));
        write_box(b"trak", &content)
    }

    fn build_mdia(&self, track: &Mp4TrackState, chunk_offsets: &[u64]) -> Vec<u8> {
        let mut content = Vec::new();
        content.extend(build_mdhd(track));
        content.extend(build_hdlr(track));
        content.extend(self.build_minf(track, chunk_offsets));
        write_box(b"mdia", &content)
    }

    fn build_minf(&self, track: &Mp4TrackState, chunk_offsets: &[u64]) -> Vec<u8> {
        let mut content = Vec::new();

        // vmhd or smhd
        match track.stream_info.media_type {
            MediaType::Video => {
                content.extend(build_vmhd());
            }
            MediaType::Audio => {
                content.extend(build_smhd());
            }
            _ => {
                // nmhd for other types
                content.extend(write_full_box(b"nmhd", 0, 0, &[]));
            }
        }

        // dinf + dref
        content.extend(build_dinf());

        // stbl
        content.extend(self.build_stbl(track, chunk_offsets));

        write_box(b"minf", &content)
    }

    fn build_stbl(&self, track: &Mp4TrackState, chunk_offsets: &[u64]) -> Vec<u8> {
        let mut content = Vec::new();

        // stsd (sample description)
        content.extend(build_stsd(track));

        // stts (time-to-sample)
        content.extend(build_stts(track));

        // ctts (composition time offsets, if needed)
        if track.samples.iter().any(|s| s.composition_offset != 0) {
            content.extend(build_ctts(track));
        }

        // stsc (sample-to-chunk)
        content.extend(build_stsc(track));

        // stsz (sample sizes)
        content.extend(build_stsz(track));

        // stco (chunk offsets)
        content.extend(build_stco(chunk_offsets));

        // stss (sync sample table, for video with non-all-sync)
        if track.stream_info.media_type == MediaType::Video
            && track.samples.iter().any(|s| !s.is_sync)
        {
            content.extend(build_stss(track));
        }

        write_box(b"stbl", &content)
    }

    /// Splits all track samples into timed fragments and builds `sidx`+`moof`+`mdat` for each.
    ///
    /// Fragment boundaries are determined by the `fragment_duration_ms` setting:
    /// a new fragment begins at each keyframe whose decode timestamp is at or beyond
    /// the target duration since the previous fragment start.
    fn build_fragments_from_tracks(&self) -> Vec<Vec<u8>> {
        // Collect fragments from the first non-empty track; use it as the reference timeline.
        let ref_track = match self.tracks.iter().find(|t| !t.samples.is_empty()) {
            Some(t) => t,
            None => return Vec::new(),
        };

        // Compute fragment boundaries (sample indices) based on the reference track.
        let config_frag_ms = match self.config.mode {
            Mp4FragmentMode::Fragmented {
                fragment_duration_ms,
            } => fragment_duration_ms,
            Mp4FragmentMode::Progressive => 0,
        };
        let frag_duration_ticks = if ref_track.timescale > 0 && config_frag_ms > 0 {
            u64::from(config_frag_ms) * u64::from(ref_track.timescale) / 1000
        } else {
            0
        };
        let boundaries = compute_fragment_boundaries(ref_track, frag_duration_ticks);

        let mut result = Vec::new();
        let mut seq = self.fragment_sequence;

        for &(frag_start, frag_end) in &boundaries {
            let fragment = self.build_one_fragment(seq, frag_start, frag_end);
            result.push(fragment);
            seq += 1;
        }

        result
    }

    /// Builds one `sidx`+`moof`+`mdat` fragment for samples `[frag_start, frag_end)` on the
    /// reference (video) track and matching time-range samples on every other track.
    fn build_one_fragment(&self, seq: u32, frag_start: usize, frag_end: usize) -> Vec<u8> {
        // ── Determine per-track sample slices ────────────────────────────────
        let ref_track = match self.tracks.first() {
            Some(t) => t,
            None => return Vec::new(),
        };

        // Compute the DTS range covered by this fragment on the reference track.
        let frag_base_dts = dts_at(ref_track, frag_start);
        let frag_end_dts = dts_at(ref_track, frag_end);

        // ── Collect mdat payload and traf descriptors ─────────────────────────
        let mut mdat_payload: Vec<u8> = Vec::new();
        let mut track_runs: Vec<FragTrackRun> = Vec::new();

        for track in &self.tracks {
            if track.samples.is_empty() {
                continue;
            }

            // For the reference track use explicit indices;
            // for other tracks select samples whose DTS falls in [frag_base_dts, frag_end_dts).
            let (slice_start, slice_end) = if track.track_id == ref_track.track_id {
                (frag_start, frag_end)
            } else {
                select_time_range(track, frag_base_dts, frag_end_dts)
            };

            if slice_start >= slice_end {
                continue;
            }

            let base_dts = dts_at(track, slice_start);
            let mdat_offset_start = mdat_payload.len() as u32;
            let mut entries: Vec<FragSampleEntry> = Vec::new();

            let mut byte_offset = 0usize;
            for s_idx in 0..slice_start {
                byte_offset += track.samples.get(s_idx).map_or(0, |s| s.size as usize);
            }

            for s_idx in slice_start..slice_end {
                let s = match track.samples.get(s_idx) {
                    Some(s) => s,
                    None => break,
                };
                let end = byte_offset + s.size as usize;
                if end <= track.mdat_data.len() {
                    mdat_payload.extend_from_slice(&track.mdat_data[byte_offset..end]);
                }
                byte_offset = end;

                let flags: u32 = if s.is_sync { 0x0200_0000 } else { 0x0101_0000 };
                #[allow(clippy::cast_possible_wrap)]
                let pts_offset = s.composition_offset;
                entries.push(FragSampleEntry {
                    duration: s.duration,
                    size: s.size,
                    flags,
                    pts_offset,
                });
            }

            track_runs.push(FragTrackRun {
                track_id: track.track_id,
                base_dts,
                mdat_offset_start,
                entries,
            });
        }

        // ── Two-pass moof build to fixup data_offset ─────────────────────────
        let build_moof = |data_offset_base: i32, runs: &[FragTrackRun]| -> Vec<u8> {
            let mut moof_content = Vec::new();
            moof_content.extend(build_mfhd(seq));
            for tr in runs {
                moof_content.extend(build_traf(
                    tr.track_id,
                    tr.base_dts,
                    data_offset_base + tr.mdat_offset_start as i32,
                    &tr.entries,
                ));
            }
            write_box(b"moof", &moof_content)
        };

        let moof_placeholder = build_moof(0, &track_runs);
        let moof_size = moof_placeholder.len() as i32;
        // data_offset is relative to start of moof; mdat header is 8 bytes.
        let moof = build_moof(moof_size + 8, &track_runs);
        let mdat = write_box(b"mdat", &mdat_payload);

        // ── sidx (segment index) ──────────────────────────────────────────────
        // Compute subsegment_duration in the reference track's timescale.
        let subseg_dur = track_runs
            .iter()
            .find(|tr| tr.track_id == ref_track.track_id)
            .map(|tr| {
                tr.entries
                    .iter()
                    .map(|e| u64::from(e.duration))
                    .sum::<u64>()
            })
            .unwrap_or(0);
        let is_sap = track_runs
            .iter()
            .find(|tr| tr.track_id == ref_track.track_id)
            .and_then(|tr| tr.entries.first())
            .is_some_and(|e| e.flags == 0x0200_0000);
        let referenced_size = (moof.len() + mdat.len()) as u32;
        let sidx = build_sidx(
            ref_track.track_id,
            ref_track.timescale,
            frag_base_dts,
            referenced_size,
            subseg_dur as u32,
            is_sap,
        );

        let mut out = sidx;
        out.extend(moof);
        out.extend(mdat);
        out
    }
}

// ─── Box builders (standalone functions) ────────────────────────────────────

/// Identity matrix for mvhd/tkhd (9 x u32 in fixed-point 16.16 / 2.30).
const IDENTITY_MATRIX: [u8; 36] = [
    0x00, 0x01, 0x00, 0x00, // 1.0
    0x00, 0x00, 0x00, 0x00, // 0.0
    0x00, 0x00, 0x00, 0x00, // 0.0
    0x00, 0x00, 0x00, 0x00, // 0.0
    0x00, 0x01, 0x00, 0x00, // 1.0
    0x00, 0x00, 0x00, 0x00, // 0.0
    0x00, 0x00, 0x00, 0x00, // 0.0
    0x00, 0x00, 0x00, 0x00, // 0.0
    0x40, 0x00, 0x00, 0x00, // 16384.0 (1.0 in 2.30)
];

fn build_tkhd(track: &Mp4TrackState) -> Vec<u8> {
    let mut c = Vec::new();
    // creation_time
    c.extend_from_slice(&write_u32_be(0));
    // modification_time
    c.extend_from_slice(&write_u32_be(0));
    // track_id
    c.extend_from_slice(&write_u32_be(track.track_id));
    // reserved
    c.extend_from_slice(&write_u32_be(0));
    // duration in movie timescale
    let dur_ms = if track.timescale == 0 {
        0
    } else {
        track.total_duration * u64::from(MOVIE_TIMESCALE) / u64::from(track.timescale)
    };
    c.extend_from_slice(&write_u32_be(dur_ms as u32));
    // reserved (8 bytes)
    c.extend_from_slice(&[0u8; 8]);
    // layer
    c.extend_from_slice(&[0u8; 2]);
    // alternate_group
    c.extend_from_slice(&[0u8; 2]);
    // volume (audio=0x0100, video=0)
    if track.stream_info.media_type == MediaType::Audio {
        c.extend_from_slice(&[0x01, 0x00]);
    } else {
        c.extend_from_slice(&[0x00, 0x00]);
    }
    // reserved
    c.extend_from_slice(&[0u8; 2]);
    // matrix
    c.extend_from_slice(&IDENTITY_MATRIX);
    // width (fixed-point 16.16)
    let w = track.stream_info.codec_params.width.unwrap_or(0);
    c.extend_from_slice(&write_u32_be(w << 16));
    // height (fixed-point 16.16)
    let h = track.stream_info.codec_params.height.unwrap_or(0);
    c.extend_from_slice(&write_u32_be(h << 16));

    // flags=3 (track_enabled | track_in_movie)
    write_full_box(b"tkhd", 0, 3, &c)
}

fn build_mdhd(track: &Mp4TrackState) -> Vec<u8> {
    let mut c = Vec::new();
    // creation_time
    c.extend_from_slice(&write_u32_be(0));
    // modification_time
    c.extend_from_slice(&write_u32_be(0));
    // timescale
    c.extend_from_slice(&write_u32_be(track.timescale));
    // duration
    c.extend_from_slice(&write_u32_be(track.total_duration as u32));
    // language (undetermined = 0x55C4)
    c.extend_from_slice(&[0x55, 0xC4]);
    // pre_defined
    c.extend_from_slice(&[0u8; 2]);

    write_full_box(b"mdhd", 0, 0, &c)
}

fn build_hdlr(track: &Mp4TrackState) -> Vec<u8> {
    let mut c = Vec::new();
    // pre_defined
    c.extend_from_slice(&write_u32_be(0));
    // handler_type
    c.extend_from_slice(&track.handler_type);
    // reserved (12 bytes)
    c.extend_from_slice(&[0u8; 12]);
    // name (null-terminated)
    c.extend_from_slice(track.handler_name.as_bytes());
    c.push(0);

    write_full_box(b"hdlr", 0, 0, &c)
}

fn build_vmhd() -> Vec<u8> {
    let mut c = Vec::new();
    // graphicsmode
    c.extend_from_slice(&[0u8; 2]);
    // opcolor (3 x u16)
    c.extend_from_slice(&[0u8; 6]);
    write_full_box(b"vmhd", 0, 1, &c)
}

fn build_smhd() -> Vec<u8> {
    let mut c = Vec::new();
    // balance
    c.extend_from_slice(&[0u8; 2]);
    // reserved
    c.extend_from_slice(&[0u8; 2]);
    write_full_box(b"smhd", 0, 0, &c)
}

fn build_dinf() -> Vec<u8> {
    // dref with one url entry (self-contained)
    let url_box = write_full_box(b"url ", 0, 1, &[]); // flag=1 means self-contained
    let mut dref_content = Vec::new();
    dref_content.extend_from_slice(&write_u32_be(1)); // entry_count
    dref_content.extend(url_box);
    let dref = write_full_box(b"dref", 0, 0, &dref_content);

    write_box(b"dinf", &dref)
}

fn build_stsd(track: &Mp4TrackState) -> Vec<u8> {
    let mut c = Vec::new();
    c.extend_from_slice(&write_u32_be(1)); // entry_count

    match track.stream_info.media_type {
        MediaType::Video => {
            c.extend(build_video_sample_entry(track));
        }
        MediaType::Audio => {
            c.extend(build_audio_sample_entry(track));
        }
        _ => {
            // Generic sample entry
            c.extend(build_generic_sample_entry(track));
        }
    }

    write_full_box(b"stsd", 0, 0, &c)
}

fn build_video_sample_entry(track: &Mp4TrackState) -> Vec<u8> {
    let fourcc = codec_to_fourcc(track.stream_info.codec);
    let mut c = Vec::new();

    // reserved (6 bytes)
    c.extend_from_slice(&[0u8; 6]);
    // data_reference_index
    c.extend_from_slice(&[0x00, 0x01]);
    // pre_defined + reserved (16 bytes)
    c.extend_from_slice(&[0u8; 16]);
    // width
    let w = track.stream_info.codec_params.width.unwrap_or(0) as u16;
    c.extend_from_slice(&w.to_be_bytes());
    // height
    let h = track.stream_info.codec_params.height.unwrap_or(0) as u16;
    c.extend_from_slice(&h.to_be_bytes());
    // horiz_resolution (72 dpi as fixed-point 16.16)
    c.extend_from_slice(&write_u32_be(0x0048_0000));
    // vert_resolution
    c.extend_from_slice(&write_u32_be(0x0048_0000));
    // reserved
    c.extend_from_slice(&write_u32_be(0));
    // frame_count
    c.extend_from_slice(&[0x00, 0x01]);
    // compressor_name (32 bytes, padded)
    let mut comp_name = [0u8; 32];
    let name = b"OxiMedia";
    let len = name.len().min(31);
    comp_name[0] = len as u8;
    comp_name[1..1 + len].copy_from_slice(&name[..len]);
    c.extend_from_slice(&comp_name);
    // depth
    c.extend_from_slice(&[0x00, 0x18]); // 24 bit
                                        // pre_defined
    c.extend_from_slice(&[0xFF, 0xFF]);

    // Codec-specific configuration box
    match track.stream_info.codec {
        CodecId::Av1 => {
            // For AV1, always emit an av1C box.  If the caller supplied a pre-built
            // AV1CodecConfigurationRecord in extradata we use that verbatim; otherwise
            // we attempt to derive one from the first OBU in extradata, falling back
            // to a safe default (Main profile, level 2.0, 4:2:0, 8-bit).
            let av1c_payload = track
                .stream_info
                .codec_params
                .extradata
                .as_deref()
                .and_then(|data| build_av1c_from_extradata(data))
                .unwrap_or_else(|| {
                    // Safe default: marker=1, version=1, Main profile, level 2.0,
                    // tier 0, 8-bit, 4:2:0, no initial presentation delay.
                    vec![0x81u8, 0x00, 0x04, 0x00]
                });
            c.extend(write_box(b"av1C", &av1c_payload));
        }
        _ => {
            if let Some(extradata) = &track.stream_info.codec_params.extradata {
                let config_fourcc = codec_config_fourcc(track.stream_info.codec);
                c.extend(write_box(&config_fourcc, extradata));
            }
        }
    }

    write_box(&fourcc, &c)
}

fn build_audio_sample_entry(track: &Mp4TrackState) -> Vec<u8> {
    let fourcc = codec_to_fourcc(track.stream_info.codec);
    let mut c = Vec::new();

    // reserved (6 bytes)
    c.extend_from_slice(&[0u8; 6]);
    // data_reference_index
    c.extend_from_slice(&[0x00, 0x01]);
    // reserved (8 bytes)
    c.extend_from_slice(&[0u8; 8]);
    // channel_count
    let channels = track.stream_info.codec_params.channels.unwrap_or(2) as u16;
    c.extend_from_slice(&channels.to_be_bytes());
    // sample_size (16 bits)
    c.extend_from_slice(&[0x00, 0x10]);
    // pre_defined + reserved
    c.extend_from_slice(&[0u8; 4]);
    // sample_rate (fixed-point 16.16)
    let sr = track.stream_info.codec_params.sample_rate.unwrap_or(48000);
    c.extend_from_slice(&write_u32_be(sr << 16));

    // Codec-specific configuration box
    if let Some(extradata) = &track.stream_info.codec_params.extradata {
        let config_fourcc = codec_config_fourcc(track.stream_info.codec);
        c.extend(write_box(&config_fourcc, extradata));
    }

    write_box(&fourcc, &c)
}

fn build_generic_sample_entry(track: &Mp4TrackState) -> Vec<u8> {
    let fourcc = codec_to_fourcc(track.stream_info.codec);
    let mut c = Vec::new();
    c.extend_from_slice(&[0u8; 6]); // reserved
    c.extend_from_slice(&[0x00, 0x01]); // data_reference_index
    write_box(&fourcc, &c)
}

fn build_stts(track: &Mp4TrackState) -> Vec<u8> {
    // Run-length encode sample durations
    let mut entries: Vec<(u32, u32)> = Vec::new();

    for sample in &track.samples {
        if let Some(last) = entries.last_mut() {
            if last.1 == sample.duration {
                last.0 += 1;
                continue;
            }
        }
        entries.push((1, sample.duration));
    }

    let mut c = Vec::new();
    c.extend_from_slice(&write_u32_be(entries.len() as u32));
    for (count, delta) in &entries {
        c.extend_from_slice(&write_u32_be(*count));
        c.extend_from_slice(&write_u32_be(*delta));
    }

    write_full_box(b"stts", 0, 0, &c)
}

fn build_ctts(track: &Mp4TrackState) -> Vec<u8> {
    // Run-length encode composition offsets
    let mut entries: Vec<(u32, i32)> = Vec::new();

    for sample in &track.samples {
        if let Some(last) = entries.last_mut() {
            if last.1 == sample.composition_offset {
                last.0 += 1;
                continue;
            }
        }
        entries.push((1, sample.composition_offset));
    }

    let mut c = Vec::new();
    c.extend_from_slice(&write_u32_be(entries.len() as u32));
    for (count, offset) in &entries {
        c.extend_from_slice(&write_u32_be(*count));
        c.extend_from_slice(&((*offset) as u32).to_be_bytes());
    }

    // version=1 for signed offsets
    write_full_box(b"ctts", 1, 0, &c)
}

fn build_stsc(track: &Mp4TrackState) -> Vec<u8> {
    // Build sample-to-chunk table
    let mut entries: Vec<(u32, u32, u32)> = Vec::new(); // (first_chunk, samples_per_chunk, sdi)

    for (chunk_idx, &chunk_start) in track.chunk_boundaries.iter().enumerate() {
        let chunk_end = track
            .chunk_boundaries
            .get(chunk_idx + 1)
            .copied()
            .unwrap_or(track.samples.len() as u32);
        let samples_in_chunk = chunk_end.saturating_sub(chunk_start);

        if samples_in_chunk == 0 {
            continue;
        }

        let first_chunk = (chunk_idx + 1) as u32; // 1-based
        if let Some(last) = entries.last() {
            if last.1 == samples_in_chunk {
                continue; // Same pattern as previous
            }
        }
        entries.push((first_chunk, samples_in_chunk, 1));
    }

    if entries.is_empty() {
        entries.push((1, 1, 1)); // Fallback
    }

    let mut c = Vec::new();
    c.extend_from_slice(&write_u32_be(entries.len() as u32));
    for (first_chunk, spc, sdi) in &entries {
        c.extend_from_slice(&write_u32_be(*first_chunk));
        c.extend_from_slice(&write_u32_be(*spc));
        c.extend_from_slice(&write_u32_be(*sdi));
    }

    write_full_box(b"stsc", 0, 0, &c)
}

fn build_stsz(track: &Mp4TrackState) -> Vec<u8> {
    let mut c = Vec::new();

    // Check if all samples have the same size
    let first_size = track.samples.first().map_or(0, |s| s.size);
    let all_same = track.samples.iter().all(|s| s.size == first_size);

    if all_same && !track.samples.is_empty() {
        // sample_size (common for all)
        c.extend_from_slice(&write_u32_be(first_size));
        // sample_count
        c.extend_from_slice(&write_u32_be(track.samples.len() as u32));
    } else {
        // sample_size = 0 (variable)
        c.extend_from_slice(&write_u32_be(0));
        // sample_count
        c.extend_from_slice(&write_u32_be(track.samples.len() as u32));
        // entry_size[]
        for sample in &track.samples {
            c.extend_from_slice(&write_u32_be(sample.size));
        }
    }

    write_full_box(b"stsz", 0, 0, &c)
}

fn build_stco(chunk_offsets: &[u64]) -> Vec<u8> {
    // Use stco (32-bit) if all offsets fit, otherwise co64
    let use_64bit = chunk_offsets.iter().any(|&o| o > u64::from(u32::MAX));

    if use_64bit {
        let mut c = Vec::new();
        c.extend_from_slice(&write_u32_be(chunk_offsets.len() as u32));
        for &offset in chunk_offsets {
            c.extend_from_slice(&write_u64_be(offset));
        }
        write_full_box(b"co64", 0, 0, &c)
    } else {
        let mut c = Vec::new();
        c.extend_from_slice(&write_u32_be(chunk_offsets.len() as u32));
        for &offset in chunk_offsets {
            c.extend_from_slice(&write_u32_be(offset as u32));
        }
        write_full_box(b"stco", 0, 0, &c)
    }
}

fn build_stss(track: &Mp4TrackState) -> Vec<u8> {
    let sync_samples: Vec<u32> = track
        .samples
        .iter()
        .enumerate()
        .filter(|(_, s)| s.is_sync)
        .map(|(i, _)| (i + 1) as u32) // 1-based
        .collect();

    let mut c = Vec::new();
    c.extend_from_slice(&write_u32_be(sync_samples.len() as u32));
    for &idx in &sync_samples {
        c.extend_from_slice(&write_u32_be(idx));
    }

    write_full_box(b"stss", 0, 0, &c)
}

fn build_trex(track_id: u32) -> Vec<u8> {
    let mut c = Vec::new();
    c.extend_from_slice(&write_u32_be(track_id));
    c.extend_from_slice(&write_u32_be(1)); // default_sample_description_index
    c.extend_from_slice(&write_u32_be(0)); // default_sample_duration
    c.extend_from_slice(&write_u32_be(0)); // default_sample_size
    c.extend_from_slice(&write_u32_be(0)); // default_sample_flags
    write_full_box(b"trex", 0, 0, &c)
}

// ─── Helpers ────────────────────────────────────────────────────────────────

fn validate_codec(codec: CodecId) -> OxiResult<()> {
    match codec {
        CodecId::Av1
        | CodecId::Vp9
        | CodecId::Vp8
        | CodecId::Opus
        | CodecId::Flac
        | CodecId::Vorbis
        | CodecId::Apv   // ISO/IEC 23009-13 — royalty-free
        | CodecId::Mjpeg // JPEG patents expired — royalty-free
        => Ok(()),
        _ => Err(OxiError::PatentViolation(format!(
            "Codec {:?} is not supported in MP4 muxer (patent-free codecs only)",
            codec
        ))),
    }
}

fn codec_to_fourcc(codec: CodecId) -> [u8; 4] {
    match codec {
        CodecId::Av1 => *b"av01",
        CodecId::Vp9 => *b"vp09",
        CodecId::Vp8 => *b"vp08",
        CodecId::Opus => *b"Opus",
        CodecId::Flac => *b"fLaC",
        CodecId::Vorbis => *b"vorb",
        CodecId::Apv => *b"apv1",   // ISO/IEC 23009-13 registered fourcc
        CodecId::Mjpeg => *b"jpeg", // ISOM-registered fourcc for Motion JPEG
        _ => *b"unkn",
    }
}

fn codec_config_fourcc(codec: CodecId) -> [u8; 4] {
    match codec {
        CodecId::Av1 => *b"av1C",
        CodecId::Vp9 => *b"vpcC",
        CodecId::Vp8 => *b"vpcC",
        CodecId::Opus => *b"dOps",
        CodecId::Flac => *b"dfLa",
        _ => *b"conf",
    }
}

#[allow(clippy::cast_precision_loss)]
fn convert_duration(ticks: i64, timebase: &Rational, target_timescale: u32) -> u32 {
    if timebase.den == 0 || target_timescale == 0 {
        return 1;
    }
    let seconds = (ticks as f64 * timebase.num as f64) / timebase.den as f64;
    let result = seconds * target_timescale as f64;
    if result <= 0.0 {
        1
    } else {
        result as u32
    }
}

fn sample_data_offset(track: &Mp4TrackState, sample_idx: u32) -> usize {
    let mut offset = 0usize;
    for i in 0..sample_idx as usize {
        if let Some(s) = track.samples.get(i) {
            offset += s.size as usize;
        }
    }
    offset
}

fn adjust_chunk_offsets(offsets: &[Vec<u64>], mdat_data_start: u64) -> Vec<Vec<u64>> {
    offsets
        .iter()
        .map(|track_offsets| track_offsets.iter().map(|&o| o + mdat_data_start).collect())
        .collect()
}

// ─── Fragment helpers ────────────────────────────────────────────────────────

/// Computes `(start_idx, end_idx)` fragment boundaries from a track's sample list.
///
/// A new fragment starts at each keyframe whose accumulated DTS since the previous
/// fragment start exceeds `frag_duration_ticks`.  The last fragment end is always
/// `track.samples.len()`.
fn compute_fragment_boundaries(
    track: &Mp4TrackState,
    frag_duration_ticks: u64,
) -> Vec<(usize, usize)> {
    if track.samples.is_empty() {
        return Vec::new();
    }

    let mut boundaries: Vec<(usize, usize)> = Vec::new();
    let mut frag_start = 0usize;
    let mut accumulated: u64 = 0;

    for (i, sample) in track.samples.iter().enumerate() {
        if i > frag_start && sample.is_sync {
            // Decide to cut here if we've accumulated enough or if no duration limit.
            if frag_duration_ticks == 0 || accumulated >= frag_duration_ticks {
                boundaries.push((frag_start, i));
                frag_start = i;
                accumulated = 0;
            }
        }
        accumulated += u64::from(sample.duration);
    }

    // Last fragment
    boundaries.push((frag_start, track.samples.len()));
    boundaries
}

/// Returns the cumulative DTS (in track timescale ticks) at sample index `idx`.
///
/// Returns 0 when `idx == 0` or the track has no samples.
fn dts_at(track: &Mp4TrackState, idx: usize) -> u64 {
    track
        .samples
        .iter()
        .take(idx)
        .map(|s| u64::from(s.duration))
        .sum()
}

/// Selects the slice `[start, end)` of samples from `track` whose accumulated DTS
/// falls within `[base_dts, end_dts)`.
///
/// Returns `(0, 0)` when the track has no samples in range.
fn select_time_range(track: &Mp4TrackState, base_dts: u64, end_dts: u64) -> (usize, usize) {
    let mut cursor: u64 = 0;
    let mut start: Option<usize> = None;
    let mut end = 0usize;

    for (i, s) in track.samples.iter().enumerate() {
        let sample_dts = cursor;
        cursor += u64::from(s.duration);

        if sample_dts >= end_dts {
            break;
        }
        if sample_dts >= base_dts {
            if start.is_none() {
                start = Some(i);
            }
            end = i + 1;
        }
    }

    match start {
        Some(s) => (s, end),
        None => (0, 0),
    }
}

/// Builds a minimal `sidx` (Segment Index Box, ISO 14496-12 §8.16.3).
///
/// Emits exactly one reference entry covering the subsequent `moof`+`mdat`.
fn build_sidx(
    reference_id: u32,
    timescale: u32,
    earliest_pts: u64,
    referenced_size: u32,
    subsegment_duration: u32,
    is_sap: bool,
) -> Vec<u8> {
    // version=1 → 64-bit earliest_presentation_time + first_offset
    let mut c = Vec::new();
    c.extend_from_slice(&write_u32_be(reference_id));
    c.extend_from_slice(&write_u32_be(timescale));
    c.extend_from_slice(&write_u64_be(earliest_pts));
    c.extend_from_slice(&write_u64_be(0)); // first_offset (0 = immediately follows)
    c.extend_from_slice(&[0u8; 2]); // reserved
    c.extend_from_slice(&1u16.to_be_bytes()); // reference_count = 1

    // reference entry:
    // bit(1) reference_type = 0 (movie fragment)
    // unsigned int(31) referenced_size
    let ref_type_size: u32 = referenced_size & 0x7FFF_FFFF; // reference_type=0
    c.extend_from_slice(&write_u32_be(ref_type_size));
    c.extend_from_slice(&write_u32_be(subsegment_duration));
    // SAP info: bit(1) starts_with_SAP + bit(3) SAP_type + bit(28) SAP_delta_time
    let sap_word: u32 = if is_sap { 0x9000_0000u32 } else { 0 };
    c.extend_from_slice(&write_u32_be(sap_word));

    write_full_box(b"sidx", 1, 0, &c)
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PacketFlags;
    use bytes::Bytes;
    use oximedia_core::{CodecId, Rational, Timestamp};

    fn make_video_stream() -> StreamInfo {
        let mut info = StreamInfo::new(0, CodecId::Av1, Rational::new(1, 90000));
        info.codec_params = crate::CodecParams::video(1920, 1080);
        info
    }

    fn make_audio_stream() -> StreamInfo {
        let mut info = StreamInfo::new(1, CodecId::Opus, Rational::new(1, 48000));
        info.codec_params = crate::CodecParams::audio(48000, 2);
        info
    }

    fn make_video_packet(stream_index: usize, pts: i64, keyframe: bool) -> Packet {
        let mut ts = Timestamp::new(pts, Rational::new(1, 90000));
        ts.duration = Some(3000); // 1 frame at 30fps in 90kHz
        Packet::new(
            stream_index,
            Bytes::from(vec![0xAA; 100]),
            ts,
            if keyframe {
                PacketFlags::KEYFRAME
            } else {
                PacketFlags::empty()
            },
        )
    }

    fn make_audio_packet(stream_index: usize, pts: i64) -> Packet {
        let mut ts = Timestamp::new(pts, Rational::new(1, 48000));
        ts.duration = Some(960); // 20ms of audio at 48kHz
        Packet::new(
            stream_index,
            Bytes::from(vec![0xBB; 50]),
            ts,
            PacketFlags::KEYFRAME,
        )
    }

    // ── Config tests ────────────────────────────────────────────────────

    #[test]
    fn test_mp4_config_default() {
        let config = Mp4Config::new();
        assert_eq!(config.mode, Mp4FragmentMode::Progressive);
        assert_eq!(config.major_brand, *b"isom");
    }

    #[test]
    fn test_mp4_config_builder() {
        let config = Mp4Config::new()
            .with_fragmented(4000)
            .with_major_brand(*b"av01")
            .with_minor_version(0x100)
            .with_compatible_brand(*b"dash");

        assert_eq!(
            config.mode,
            Mp4FragmentMode::Fragmented {
                fragment_duration_ms: 4000
            }
        );
        assert_eq!(config.major_brand, *b"av01");
        assert_eq!(config.minor_version, 0x100);
        assert!(config.compatible_brands.contains(b"dash"));
    }

    // ── Muxer lifecycle tests ───────────────────────────────────────────

    #[test]
    fn test_add_stream_returns_index() {
        let mut muxer = Mp4Muxer::new(Mp4Config::new());
        let idx0 = muxer
            .add_stream(make_video_stream())
            .expect("should succeed");
        let idx1 = muxer
            .add_stream(make_audio_stream())
            .expect("should succeed");
        assert_eq!(idx0, 0);
        assert_eq!(idx1, 1);
        assert_eq!(muxer.track_count(), 2);
    }

    #[test]
    fn test_add_stream_after_header_fails() {
        let mut muxer = Mp4Muxer::new(Mp4Config::new());
        muxer
            .add_stream(make_video_stream())
            .expect("should succeed");
        muxer.write_header().expect("should succeed");
        let result = muxer.add_stream(make_audio_stream());
        assert!(result.is_err());
    }

    #[test]
    fn test_write_header_no_streams_fails() {
        let mut muxer = Mp4Muxer::new(Mp4Config::new());
        let result = muxer.write_header();
        assert!(result.is_err());
    }

    #[test]
    fn test_write_packet_before_header_fails() {
        let mut muxer = Mp4Muxer::new(Mp4Config::new());
        muxer
            .add_stream(make_video_stream())
            .expect("should succeed");
        let pkt = make_video_packet(0, 0, true);
        let result = muxer.write_packet(&pkt);
        assert!(result.is_err());
    }

    #[test]
    fn test_write_packet_invalid_index_fails() {
        let mut muxer = Mp4Muxer::new(Mp4Config::new());
        muxer
            .add_stream(make_video_stream())
            .expect("should succeed");
        muxer.write_header().expect("should succeed");
        let pkt = make_video_packet(99, 0, true);
        let result = muxer.write_packet(&pkt);
        assert!(result.is_err());
    }

    #[test]
    fn test_patent_encumbered_codec_rejected() {
        let mut muxer = Mp4Muxer::new(Mp4Config::new());
        let info = StreamInfo::new(0, CodecId::Pcm, Rational::new(1, 44100));
        let result = muxer.add_stream(info);
        assert!(result.is_err());
    }

    // ── Progressive MP4 output tests ────────────────────────────────────

    #[test]
    fn test_progressive_ftyp_present() {
        let mut muxer = Mp4Muxer::new(Mp4Config::new());
        muxer
            .add_stream(make_video_stream())
            .expect("should succeed");
        muxer.write_header().expect("should succeed");
        let pkt = make_video_packet(0, 0, true);
        muxer.write_packet(&pkt).expect("should succeed");

        let output = muxer.finalize().expect("should succeed");
        // ftyp box should be first
        assert_eq!(&output[4..8], b"ftyp");
        // major brand should be "isom"
        assert_eq!(&output[8..12], b"isom");
    }

    #[test]
    fn test_progressive_contains_moov_and_mdat() {
        let mut muxer = Mp4Muxer::new(Mp4Config::new());
        muxer
            .add_stream(make_video_stream())
            .expect("should succeed");
        muxer.write_header().expect("should succeed");
        muxer
            .write_packet(&make_video_packet(0, 0, true))
            .expect("should succeed");
        muxer
            .write_packet(&make_video_packet(0, 3000, false))
            .expect("should succeed");

        let output = muxer.finalize().expect("should succeed");
        let has_moov = output.windows(4).any(|w| w == b"moov");
        let has_mdat = output.windows(4).any(|w| w == b"mdat");
        assert!(has_moov, "output must contain moov box");
        assert!(has_mdat, "output must contain mdat box");
    }

    #[test]
    fn test_progressive_contains_trak_boxes() {
        let mut muxer = Mp4Muxer::new(Mp4Config::new());
        muxer
            .add_stream(make_video_stream())
            .expect("should succeed");
        muxer
            .add_stream(make_audio_stream())
            .expect("should succeed");
        muxer.write_header().expect("should succeed");
        muxer
            .write_packet(&make_video_packet(0, 0, true))
            .expect("should succeed");
        muxer
            .write_packet(&make_audio_packet(1, 0))
            .expect("should succeed");

        let output = muxer.finalize().expect("should succeed");
        let trak_count = output.windows(4).filter(|w| *w == b"trak").count();
        assert_eq!(trak_count, 2, "output must contain 2 trak boxes");
    }

    #[test]
    fn test_progressive_contains_stbl_tables() {
        let mut muxer = Mp4Muxer::new(Mp4Config::new());
        muxer
            .add_stream(make_video_stream())
            .expect("should succeed");
        muxer.write_header().expect("should succeed");

        for i in 0..5 {
            muxer
                .write_packet(&make_video_packet(0, i * 3000, i == 0))
                .expect("should succeed");
        }

        let output = muxer.finalize().expect("should succeed");

        // Check for essential sample table boxes
        let has_stsd = output.windows(4).any(|w| w == b"stsd");
        let has_stts = output.windows(4).any(|w| w == b"stts");
        let has_stsz = output.windows(4).any(|w| w == b"stsz");
        let has_stco = output.windows(4).any(|w| w == b"stco");
        let has_stss = output.windows(4).any(|w| w == b"stss");

        assert!(has_stsd, "must have stsd");
        assert!(has_stts, "must have stts");
        assert!(has_stsz, "must have stsz");
        assert!(has_stco, "must have stco");
        assert!(has_stss, "must have stss (video with non-sync frames)");
    }

    #[test]
    fn test_progressive_mdat_contains_sample_data() {
        let mut muxer = Mp4Muxer::new(Mp4Config::new());
        muxer
            .add_stream(make_video_stream())
            .expect("should succeed");
        muxer.write_header().expect("should succeed");
        muxer
            .write_packet(&make_video_packet(0, 0, true))
            .expect("should succeed");

        let output = muxer.finalize().expect("should succeed");

        // Find mdat and verify it contains our 0xAA payload
        let mdat_pos = output
            .windows(4)
            .position(|w| w == b"mdat")
            .expect("mdat must exist");
        // mdat data starts 4 bytes after the fourcc
        let mdat_start = mdat_pos + 4;
        assert!(
            output[mdat_start..].windows(4).any(|w| w == [0xAA; 4]),
            "mdat must contain sample payload"
        );
    }

    #[test]
    fn test_total_samples_count() {
        let mut muxer = Mp4Muxer::new(Mp4Config::new());
        muxer
            .add_stream(make_video_stream())
            .expect("should succeed");
        muxer
            .add_stream(make_audio_stream())
            .expect("should succeed");
        muxer.write_header().expect("should succeed");

        muxer
            .write_packet(&make_video_packet(0, 0, true))
            .expect("should succeed");
        muxer
            .write_packet(&make_video_packet(0, 3000, false))
            .expect("should succeed");
        muxer
            .write_packet(&make_audio_packet(1, 0))
            .expect("should succeed");

        assert_eq!(muxer.total_samples(), 3);
    }

    // ── Fragmented MP4 tests ────────────────────────────────────────────

    #[test]
    fn test_fragmented_contains_mvex() {
        let config = Mp4Config::new().with_fragmented(2000);
        let mut muxer = Mp4Muxer::new(config);
        muxer
            .add_stream(make_video_stream())
            .expect("should succeed");
        muxer.write_header().expect("should succeed");
        muxer
            .write_packet(&make_video_packet(0, 0, true))
            .expect("should succeed");

        let output = muxer.finalize().expect("should succeed");
        let has_mvex = output.windows(4).any(|w| w == b"mvex");
        let has_trex = output.windows(4).any(|w| w == b"trex");
        assert!(has_mvex, "fragmented must contain mvex");
        assert!(has_trex, "fragmented must contain trex");
    }

    #[test]
    fn test_fragmented_contains_moof_mdat() {
        let config = Mp4Config::new().with_fragmented(2000);
        let mut muxer = Mp4Muxer::new(config);
        muxer
            .add_stream(make_video_stream())
            .expect("should succeed");
        muxer.write_header().expect("should succeed");
        muxer
            .write_packet(&make_video_packet(0, 0, true))
            .expect("should succeed");

        let output = muxer.finalize().expect("should succeed");
        let has_moof = output.windows(4).any(|w| w == b"moof");
        let has_mdat = output.windows(4).any(|w| w == b"mdat");
        assert!(has_moof, "fragmented must contain moof");
        assert!(has_mdat, "fragmented must contain mdat");
    }

    #[test]
    fn test_fragmented_contains_traf_trun() {
        let config = Mp4Config::new().with_fragmented(2000);
        let mut muxer = Mp4Muxer::new(config);
        muxer
            .add_stream(make_video_stream())
            .expect("should succeed");
        muxer.write_header().expect("should succeed");
        muxer
            .write_packet(&make_video_packet(0, 0, true))
            .expect("should succeed");

        let output = muxer.finalize().expect("should succeed");
        let has_traf = output.windows(4).any(|w| w == b"traf");
        let has_trun = output.windows(4).any(|w| w == b"trun");
        assert!(has_traf, "fragmented must contain traf");
        assert!(has_trun, "fragmented must contain trun");
    }

    // ── Helper function tests ───────────────────────────────────────────

    #[test]
    fn test_validate_codec_accepted() {
        assert!(validate_codec(CodecId::Av1).is_ok());
        assert!(validate_codec(CodecId::Vp9).is_ok());
        assert!(validate_codec(CodecId::Opus).is_ok());
        assert!(validate_codec(CodecId::Flac).is_ok());
    }

    #[test]
    fn test_validate_codec_rejected() {
        assert!(validate_codec(CodecId::Pcm).is_err());
    }

    #[test]
    fn test_codec_to_fourcc() {
        assert_eq!(codec_to_fourcc(CodecId::Av1), *b"av01");
        assert_eq!(codec_to_fourcc(CodecId::Vp9), *b"vp09");
        assert_eq!(codec_to_fourcc(CodecId::Opus), *b"Opus");
    }

    #[test]
    fn test_convert_duration_basic() {
        let tb = Rational::new(1, 90000);
        let result = convert_duration(90000, &tb, 1000);
        assert_eq!(result, 1000); // 1 second
    }

    #[test]
    fn test_convert_duration_zero_target() {
        let tb = Rational::new(1, 90000);
        let result = convert_duration(100, &tb, 0);
        assert_eq!(result, 1); // fallback
    }

    #[test]
    fn test_sample_data_offset() {
        let mut track = Mp4TrackState::new(make_video_stream(), 1);
        track.add_sample(&[0xAA; 100], 3000, 0, true);
        track.add_sample(&[0xBB; 200], 3000, 0, false);
        track.add_sample(&[0xCC; 150], 3000, 0, false);

        assert_eq!(sample_data_offset(&track, 0), 0);
        assert_eq!(sample_data_offset(&track, 1), 100);
        assert_eq!(sample_data_offset(&track, 2), 300);
    }

    #[test]
    fn test_adjust_chunk_offsets() {
        let offsets = vec![vec![0, 100, 200], vec![0, 50]];
        let adjusted = adjust_chunk_offsets(&offsets, 1000);
        assert_eq!(adjusted[0], vec![1000, 1100, 1200]);
        assert_eq!(adjusted[1], vec![1000, 1050]);
    }

    #[test]
    fn test_build_stts_run_length() {
        let mut track = Mp4TrackState::new(make_video_stream(), 1);
        // 5 samples all with duration 3000
        for _ in 0..5 {
            track.add_sample(&[0u8; 10], 3000, 0, true);
        }
        let stts = build_stts(&track);
        // Should have entry_count=1 (all same duration)
        // fullbox header (12) + entry_count (4) + one entry (8) = 24
        assert_eq!(stts.len(), 24);
    }

    #[test]
    fn test_build_stsz_uniform() {
        let mut track = Mp4TrackState::new(make_video_stream(), 1);
        for _ in 0..3 {
            track.add_sample(&[0u8; 42], 3000, 0, true);
        }
        let stsz = build_stsz(&track);
        // With uniform sizes: fullbox header (12) + sample_size (4) + sample_count (4) = 20
        assert_eq!(stsz.len(), 20);
    }

    #[test]
    fn test_build_stsz_variable() {
        let mut track = Mp4TrackState::new(make_video_stream(), 1);
        track.add_sample(&[0u8; 100], 3000, 0, true);
        track.add_sample(&[0u8; 200], 3000, 0, false);
        let stsz = build_stsz(&track);
        // Variable: fullbox header (12) + sample_size=0 (4) + sample_count (4) + 2*4 = 28
        assert_eq!(stsz.len(), 28);
    }

    #[test]
    fn test_build_stss_sync_samples() {
        let mut track = Mp4TrackState::new(make_video_stream(), 1);
        track.add_sample(&[0u8; 10], 3000, 0, true);
        track.add_sample(&[0u8; 10], 3000, 0, false);
        track.add_sample(&[0u8; 10], 3000, 0, false);
        track.add_sample(&[0u8; 10], 3000, 0, true);
        let stss = build_stss(&track);
        // 2 sync samples: fullbox header (12) + entry_count (4) + 2*4 = 24
        assert_eq!(stss.len(), 24);
    }

    #[test]
    fn test_track_state_handler_types() {
        let video_track = Mp4TrackState::new(make_video_stream(), 1);
        assert_eq!(video_track.handler_type, *b"vide");

        let audio_track = Mp4TrackState::new(make_audio_stream(), 2);
        assert_eq!(audio_track.handler_type, *b"soun");
    }

    #[test]
    fn test_progressive_multi_packet_output_valid() {
        let mut muxer = Mp4Muxer::new(Mp4Config::new());
        muxer
            .add_stream(make_video_stream())
            .expect("should succeed");
        muxer
            .add_stream(make_audio_stream())
            .expect("should succeed");
        muxer.write_header().expect("should succeed");

        // Interleaved packets
        for i in 0..10 {
            muxer
                .write_packet(&make_video_packet(0, i * 3000, i == 0 || i == 5))
                .expect("should succeed");
            muxer
                .write_packet(&make_audio_packet(1, i * 960))
                .expect("should succeed");
        }

        let output = muxer.finalize().expect("should succeed");
        assert!(!output.is_empty());

        // Verify box structure: ftyp must come first
        assert_eq!(&output[4..8], b"ftyp");

        // Must contain all essential boxes
        let has_mvhd = output.windows(4).any(|w| w == b"mvhd");
        let has_tkhd = output.windows(4).any(|w| w == b"tkhd");
        let has_mdhd = output.windows(4).any(|w| w == b"mdhd");
        let has_hdlr = output.windows(4).any(|w| w == b"hdlr");
        assert!(has_mvhd);
        assert!(has_tkhd);
        assert!(has_mdhd);
        assert!(has_hdlr);
    }

    #[test]
    fn test_finalize_before_header_fails() {
        let muxer = Mp4Muxer::new(Mp4Config::new());
        assert!(muxer.finalize().is_err());
    }

    #[test]
    fn test_empty_tracks_finalize() {
        let mut muxer = Mp4Muxer::new(Mp4Config::new());
        muxer
            .add_stream(make_video_stream())
            .expect("should succeed");
        muxer.write_header().expect("should succeed");
        // No packets written
        let output = muxer.finalize().expect("should succeed");
        assert!(!output.is_empty());
        assert_eq!(&output[4..8], b"ftyp");
    }
}
