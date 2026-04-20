//! CMAF (Common Media Application Format) muxer.
//!
//! Implements fragmented ISOBMFF suitable for DASH/HLS adaptive streaming.
//! Produces `ftyp`+`moov` init segments and `moof`+`mdat` media segments
//! in strict conformance with ISO/IEC 23000-19 (CMAF).
//!
//! # Box layout
//!
//! ```text
//! Init segment:   [ftyp][moov[mvhd][mvex[trex...]][trak[tkhd][mdia[mdhd][hdlr][minf[vmhd/smhd][dinf[dref]][stbl[stsd][stts][stsc][stsz][stco]]]]]]...]
//! Media segment:  [moof[mfhd][traf[tfhd][tfdt][trun]]...][mdat]
//! ```

#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

// ─── Public data types ────────────────────────────────────────────────────────

/// CMAF brand selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmafBrand {
    /// CMAF general (cmf2 / cmfc).
    CmafCm,
    /// CMAF HVC profile (HEVC track).
    CmafCmhv,
    /// CMAF HD profile (AVC HD track).
    CmafCmhd,
    /// CMAF AVC-compatible profile.
    CmafCmavc,
    /// CMAF AV1-compatible profile.
    CmafCmav1,
}

impl CmafBrand {
    /// Returns the 4-byte brand code for this CMAF brand.
    #[must_use]
    pub const fn as_bytes(self) -> &'static [u8; 4] {
        match self {
            Self::CmafCm => b"cmf2",
            Self::CmafCmhv => b"cmhv",
            Self::CmafCmhd => b"cmhd",
            Self::CmafCmavc => b"cavc",
            Self::CmafCmav1 => b"cav1",
        }
    }
}

/// Track type classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackType {
    /// Video elementary stream.
    Video,
    /// Audio elementary stream.
    Audio,
    /// Subtitle / timed-text stream.
    Subtitle,
}

/// Per-track descriptor added to the muxer before writing.
#[derive(Debug, Clone)]
pub struct CmafTrack {
    /// Caller-assigned track ID (must be unique and ≥ 1).
    pub track_id: u32,
    /// Track classification.
    pub track_type: TrackType,
    /// Codec four-character code (e.g. `b"av01"`, `b"Opus"`).
    pub codec_fourcc: [u8; 4],
    /// Timescale (ticks per second) for this track.
    pub timescale: u32,
    /// Pixel width for video tracks.
    pub width: Option<u32>,
    /// Pixel height for video tracks.
    pub height: Option<u32>,
    /// Sample rate for audio tracks.
    pub sample_rate: Option<u32>,
    /// Channel count for audio tracks.
    pub channels: Option<u8>,
    /// Codec-specific configuration record (e.g. `AV1CodecConfigurationRecord`).
    pub extradata: Vec<u8>,
}

/// Global configuration for the CMAF muxer.
#[derive(Debug, Clone)]
pub struct CmafConfig {
    /// Target fragment duration in milliseconds (e.g. `2000`).
    pub fragment_duration_ms: u32,
    /// Whether to add Common Encryption (CENC) signalling.
    pub use_encryption: bool,
    /// CMAF brand to declare in `ftyp`.
    pub brand: CmafBrand,
    /// Default timescale used when a track does not specify one.
    pub timescale: u32,
    /// Enables low-latency chunked fragment emission.
    pub low_latency_chunked: bool,
    /// Target chunk duration in milliseconds when low-latency mode is enabled.
    pub chunk_duration_ms: Option<u32>,
}

impl Default for CmafConfig {
    fn default() -> Self {
        Self {
            fragment_duration_ms: 2000,
            use_encryption: false,
            brand: CmafBrand::CmafCm,
            timescale: 90000,
            low_latency_chunked: false,
            chunk_duration_ms: None,
        }
    }
}

impl CmafConfig {
    /// Creates a new `CmafConfig` with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

/// One complete CMAF segment (either init or media).
#[derive(Debug, Clone)]
pub struct CmafSegment {
    /// Monotonically increasing sequence number (1-based for media segments).
    pub sequence_number: u32,
    /// Presentation start time in the track's timescale.
    pub start_pts: u64,
    /// Segment duration in the track's timescale.
    pub duration: u64,
    /// Raw segment bytes ready for delivery.
    pub data: Vec<u8>,
    /// `true` for `ftyp`+`moov`, `false` for `moof`+`mdat`.
    pub is_init: bool,
}

/// One compressed sample handed to the muxer.
#[derive(Debug, Clone)]
pub struct CmafSample {
    /// ID of the track this sample belongs to.
    pub track_id: u32,
    /// Presentation timestamp in the track's timescale.
    pub pts: u64,
    /// Decode timestamp in the track's timescale.
    pub dts: u64,
    /// Sample duration in the track's timescale.
    pub duration: u32,
    /// Compressed sample payload.
    pub data: Vec<u8>,
    /// Whether this sample is a sync (key) frame.
    pub keyframe: bool,
}

// ─── Internal pending segment state ──────────────────────────────────────────

#[derive(Debug, Default)]
struct PendingSegment {
    /// Samples grouped by track_id.
    samples: HashMap<u32, Vec<CmafSample>>,
    /// Earliest DTS seen across all tracks (used for tfdt).
    start_pts: Option<u64>,
}

impl PendingSegment {
    fn push(&mut self, sample: CmafSample) {
        if self.start_pts.is_none() || sample.dts < self.start_pts.unwrap_or(u64::MAX) {
            self.start_pts = Some(sample.dts);
        }
        self.samples
            .entry(sample.track_id)
            .or_default()
            .push(sample);
    }

    fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    #[allow(dead_code)]
    fn total_duration(&self, tracks: &HashMap<u32, CmafTrack>) -> u64 {
        // Return the sum of sample durations for the first track found (track
        // with the smallest track_id to be deterministic).
        let mut min_id = u32::MAX;
        for &id in self.samples.keys() {
            if id < min_id {
                min_id = id;
            }
        }
        if min_id == u32::MAX {
            return 0;
        }
        let timescale = tracks.get(&min_id).map_or(90000, |t| t.timescale);
        let samples = self.samples.get(&min_id).map_or(&[][..], |v| v.as_slice());
        let ticks: u64 = samples.iter().map(|s| u64::from(s.duration)).sum();
        // Convert ticks → ms
        if timescale == 0 {
            0
        } else {
            ticks * 1000 / u64::from(timescale)
        }
    }
}

// ─── Muxer ────────────────────────────────────────────────────────────────────

/// CMAF muxer.
///
/// Usage:
/// ```ignore
/// let config = CmafConfig::default();
/// let mut muxer = CmafMuxer::new(config);
/// muxer.add_track(video_track);
/// muxer.add_track(audio_track);
/// let init = muxer.write_init_segment();
/// // … feed samples …
/// muxer.write_media_segment(&samples);
/// if let Some(seg) = muxer.flush_segment() { /* deliver seg.data */ }
/// ```
#[derive(Debug)]
pub struct CmafMuxer {
    config: CmafConfig,
    /// Registered tracks keyed by `track_id`.
    tracks: HashMap<u32, CmafTrack>,
    /// Ordered list of track IDs (insertion order).
    track_order: Vec<u32>,
    /// Monotonically increasing media-segment sequence counter.
    sequence_number: u32,
    /// Samples accumulated for the segment currently being built.
    pending: PendingSegment,
}

impl CmafMuxer {
    /// Creates a new CMAF muxer with the given configuration.
    #[must_use]
    pub fn new(config: CmafConfig) -> Self {
        Self {
            config,
            tracks: HashMap::new(),
            track_order: Vec::new(),
            sequence_number: 1,
            pending: PendingSegment::default(),
        }
    }

    /// Registers a track.  Returns the `track_id` that was supplied in the
    /// `CmafTrack` struct (the caller controls IDs).
    pub fn add_track(&mut self, track: CmafTrack) -> u32 {
        let id = track.track_id;
        if !self.tracks.contains_key(&id) {
            self.track_order.push(id);
        }
        self.tracks.insert(id, track);
        id
    }

    // ─── Init segment ─────────────────────────────────────────────────────

    /// Writes the CMAF initialization segment (`ftyp` + `moov`).
    ///
    /// The returned `Vec<u8>` is the complete byte sequence ready for delivery.
    #[must_use]
    pub fn write_init_segment(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend(self.build_ftyp());
        out.extend(self.build_moov());
        out
    }

    fn build_ftyp(&self) -> Vec<u8> {
        let brand = self.config.brand.as_bytes();
        let mut content = Vec::new();
        // Major brand
        content.extend_from_slice(brand);
        // Minor version
        content.extend_from_slice(&write_u32_be(0));
        // Compatible brands
        content.extend_from_slice(b"iso6");
        content.extend_from_slice(b"iso5");
        content.extend_from_slice(b"cmf2");
        content.extend_from_slice(brand); // always include the declared brand
        write_box(b"ftyp", &content)
    }

    fn build_moov(&self) -> Vec<u8> {
        let mut content = Vec::new();
        content.extend(self.build_mvhd());
        content.extend(self.build_mvex());
        for &tid in &self.track_order {
            if let Some(track) = self.tracks.get(&tid) {
                content.extend(self.build_trak(track));
            }
        }
        write_box(b"moov", &content)
    }

    fn build_mvhd(&self) -> Vec<u8> {
        let mut c = Vec::new();
        // version=0, flags=0
        c.extend_from_slice(&write_u32_be(0));
        // creation_time
        c.extend_from_slice(&write_u32_be(0));
        // modification_time
        c.extend_from_slice(&write_u32_be(0));
        // timescale (ms)
        c.extend_from_slice(&write_u32_be(1000));
        // duration (0 = indeterminate for fragmented)
        c.extend_from_slice(&write_u32_be(0));
        // rate 1.0 (16.16 fixed point)
        c.extend_from_slice(&write_u32_be(0x0001_0000));
        // volume 1.0 (8.8 fixed point) + 10 reserved bytes
        c.extend_from_slice(&[0x01, 0x00]);
        c.extend_from_slice(&[0u8; 10]);
        // identity matrix
        for v in &[0x0001_0000_i32, 0, 0, 0, 0x0001_0000, 0, 0, 0, 0x4000_0000] {
            c.extend_from_slice(&v.to_be_bytes());
        }
        // pre-defined (6 × u32)
        c.extend_from_slice(&[0u8; 24]);
        // next_track_ID
        let next = self.track_order.iter().copied().max().map_or(1, |m| m + 1);
        c.extend_from_slice(&write_u32_be(next));
        write_full_box(b"mvhd", 0, 0, &c)
    }

    fn build_mvex(&self) -> Vec<u8> {
        let mut c = Vec::new();
        for &tid in &self.track_order {
            c.extend(self.build_trex(tid));
        }
        write_box(b"mvex", &c)
    }

    fn build_trex(&self, track_id: u32) -> Vec<u8> {
        let mut c = Vec::new();
        c.extend_from_slice(&write_u32_be(track_id));
        c.extend_from_slice(&write_u32_be(1)); // default_sample_description_index
        c.extend_from_slice(&write_u32_be(0)); // default_sample_duration
        c.extend_from_slice(&write_u32_be(0)); // default_sample_size
        c.extend_from_slice(&write_u32_be(0)); // default_sample_flags
        write_full_box(b"trex", 0, 0, &c)
    }

    fn build_trak(&self, track: &CmafTrack) -> Vec<u8> {
        let mut c = Vec::new();
        c.extend(self.build_tkhd(track));
        c.extend(self.build_mdia(track));
        write_box(b"trak", &c)
    }

    fn build_tkhd(&self, track: &CmafTrack) -> Vec<u8> {
        let mut c = Vec::new();
        // creation_time, modification_time
        c.extend_from_slice(&write_u32_be(0));
        c.extend_from_slice(&write_u32_be(0));
        // track_ID
        c.extend_from_slice(&write_u32_be(track.track_id));
        // reserved
        c.extend_from_slice(&write_u32_be(0));
        // duration (0 = indeterminate)
        c.extend_from_slice(&write_u32_be(0));
        // reserved (2 × u32)
        c.extend_from_slice(&[0u8; 8]);
        // layer, alternate_group
        c.extend_from_slice(&[0u8; 4]);
        // volume: 1.0 for audio, 0 for video/subtitle
        let volume: u16 = if track.track_type == TrackType::Audio {
            0x0100
        } else {
            0
        };
        c.extend_from_slice(&volume.to_be_bytes());
        // reserved
        c.extend_from_slice(&[0u8; 2]);
        // identity matrix
        for v in &[0x0001_0000_i32, 0, 0, 0, 0x0001_0000, 0, 0, 0, 0x4000_0000] {
            c.extend_from_slice(&v.to_be_bytes());
        }
        // width / height in 16.16 fixed point
        let w = track.width.unwrap_or(0);
        let h = track.height.unwrap_or(0);
        c.extend_from_slice(&write_u32_be(w << 16));
        c.extend_from_slice(&write_u32_be(h << 16));
        // flags: track_enabled(1) | track_in_movie(2) | track_in_preview(4)
        write_full_box(b"tkhd", 0, 0x00_00_00_07, &c)
    }

    fn build_mdia(&self, track: &CmafTrack) -> Vec<u8> {
        let mut c = Vec::new();
        c.extend(self.build_mdhd(track));
        c.extend(self.build_hdlr(track));
        c.extend(self.build_minf(track));
        write_box(b"mdia", &c)
    }

    fn build_mdhd(&self, track: &CmafTrack) -> Vec<u8> {
        let mut c = Vec::new();
        // creation_time, modification_time
        c.extend_from_slice(&write_u32_be(0));
        c.extend_from_slice(&write_u32_be(0));
        // timescale
        c.extend_from_slice(&write_u32_be(track.timescale));
        // duration (0 = indeterminate)
        c.extend_from_slice(&write_u32_be(0));
        // language: 'und' packed as ISO-639-2/T
        c.extend_from_slice(&0x55c4u16.to_be_bytes());
        // pre_defined
        c.extend_from_slice(&[0u8; 2]);
        write_full_box(b"mdhd", 0, 0, &c)
    }

    fn build_hdlr(&self, track: &CmafTrack) -> Vec<u8> {
        let mut c = Vec::new();
        // pre_defined
        c.extend_from_slice(&write_u32_be(0));
        // handler_type
        let handler: &[u8; 4] = match track.track_type {
            TrackType::Video => b"vide",
            TrackType::Audio => b"soun",
            TrackType::Subtitle => b"subt",
        };
        c.extend_from_slice(handler);
        // reserved (3 × u32)
        c.extend_from_slice(&[0u8; 12]);
        // name: null-terminated string
        c.push(0);
        write_full_box(b"hdlr", 0, 0, &c)
    }

    fn build_minf(&self, track: &CmafTrack) -> Vec<u8> {
        let mut c = Vec::new();
        // Media information header
        match track.track_type {
            TrackType::Video => c.extend(build_vmhd()),
            TrackType::Audio => c.extend(build_smhd()),
            TrackType::Subtitle => c.extend(build_nmhd()),
        }
        c.extend(build_dinf());
        c.extend(self.build_stbl(track));
        write_box(b"minf", &c)
    }

    fn build_stbl(&self, track: &CmafTrack) -> Vec<u8> {
        let mut c = Vec::new();
        c.extend(self.build_stsd(track));
        // Empty stts, stsc, stsz, stco (required by spec, but 0 entries for fragmented)
        c.extend(build_empty_stts());
        c.extend(build_empty_stsc());
        c.extend(build_empty_stsz());
        c.extend(build_empty_stco());
        write_box(b"stbl", &c)
    }

    fn build_stsd(&self, track: &CmafTrack) -> Vec<u8> {
        let mut entries = Vec::new();
        entries.extend(self.build_sample_entry(track));
        let mut c = Vec::new();
        // entry_count
        c.extend_from_slice(&write_u32_be(1));
        c.extend(entries);
        write_full_box(b"stsd", 0, 0, &c)
    }

    fn build_sample_entry(&self, track: &CmafTrack) -> Vec<u8> {
        let mut c = Vec::new();
        // 6-byte reserved
        c.extend_from_slice(&[0u8; 6]);
        // data_reference_index
        c.extend_from_slice(&1u16.to_be_bytes());

        match track.track_type {
            TrackType::Video => {
                // VisualSampleEntry fields
                c.extend_from_slice(&[0u8; 16]); // pre_defined + reserved
                let w = track.width.unwrap_or(0) as u16;
                let h = track.height.unwrap_or(0) as u16;
                c.extend_from_slice(&w.to_be_bytes());
                c.extend_from_slice(&h.to_be_bytes());
                c.extend_from_slice(&0x0048_0000u32.to_be_bytes()); // horizresolution 72 dpi
                c.extend_from_slice(&0x0048_0000u32.to_be_bytes()); // vertresolution
                c.extend_from_slice(&write_u32_be(0)); // reserved
                c.extend_from_slice(&1u16.to_be_bytes()); // frame_count
                c.extend_from_slice(&[0u8; 32]); // compressorname
                c.extend_from_slice(&0x0018u16.to_be_bytes()); // depth
                c.extend_from_slice(&(-1i16).to_be_bytes()); // pre_defined
                                                             // codec configuration box
                if !track.extradata.is_empty() {
                    // Wrap in a generic 'glbl' box (or codec-specific – use fourcc as box type)
                    let cfg_box = write_box(&track.codec_fourcc, &track.extradata);
                    c.extend(cfg_box);
                }
            }
            TrackType::Audio => {
                // AudioSampleEntry fields
                c.extend_from_slice(&[0u8; 8]); // reserved
                let ch = track.channels.unwrap_or(2);
                c.extend_from_slice(&(ch as u16).to_be_bytes());
                c.extend_from_slice(&16u16.to_be_bytes()); // samplesize
                c.extend_from_slice(&[0u8; 4]); // pre_defined + reserved
                let sr = track.sample_rate.unwrap_or(48000);
                c.extend_from_slice(&write_u32_be(sr << 16)); // samplerate 16.16
                if !track.extradata.is_empty() {
                    let cfg_box = write_box(&track.codec_fourcc, &track.extradata);
                    c.extend(cfg_box);
                }
            }
            TrackType::Subtitle => {
                // Minimal text sample entry
            }
        }

        write_box(&track.codec_fourcc, &c)
    }

    // ─── Media segment ────────────────────────────────────────────────────

    /// Writes a media segment (`moof` + `mdat`) from the provided samples.
    ///
    /// Samples for multiple tracks may be interleaved; the muxer groups them
    /// into one `traf` per track inside a single `moof`.
    pub fn write_media_segment(&mut self, samples: &[CmafSample]) -> Vec<u8> {
        for s in samples {
            self.pending.push(s.clone());
        }
        self.emit_segment()
    }

    /// Finalizes any pending samples into a `CmafSegment`.
    ///
    /// Returns `None` if there are no pending samples.
    pub fn flush_segment(&mut self) -> Option<CmafSegment> {
        if self.pending.is_empty() {
            return None;
        }
        let start_pts = self.pending.start_pts.unwrap_or(0);
        let data = self.emit_segment();
        let seg = CmafSegment {
            sequence_number: self.sequence_number - 1,
            start_pts,
            duration: 0, // computed externally from sample durations
            data,
            is_init: false,
        };
        Some(seg)
    }

    fn emit_segment(&mut self) -> Vec<u8> {
        if self.pending.is_empty() {
            return Vec::new();
        }

        if self.config.low_latency_chunked {
            return self.emit_chunked_segment();
        }

        let seq = self.sequence_number;
        self.sequence_number += 1;

        // Collect all raw sample bytes for mdat.
        // Use a two-pass approach: first build traf content, then fix up data offsets.
        let mut mdat_payload: Vec<u8> = Vec::new();
        let mut track_runs: Vec<FragTrackRun> = Vec::new();

        for &tid in &self.track_order {
            let samples_opt = self.pending.samples.get(&tid);
            let samples = match samples_opt {
                Some(v) if !v.is_empty() => v,
                _ => continue,
            };

            let mdat_offset_start = mdat_payload.len() as u32;
            let base_dts = samples[0].dts;
            let mut entries = Vec::new();

            for s in samples {
                let flags: u32 = if s.keyframe { 0x0200_0000 } else { 0x0101_0000 };
                #[allow(clippy::cast_possible_wrap)]
                let pts_offset = (s.pts as i64 - s.dts as i64) as i32;
                entries.push(FragSampleEntry {
                    duration: s.duration,
                    size: s.data.len() as u32,
                    flags,
                    pts_offset,
                });
                mdat_payload.extend_from_slice(&s.data);
            }

            track_runs.push(FragTrackRun {
                track_id: tid,
                base_dts,
                mdat_offset_start,
                entries,
            });
        }

        // Pass 2: build moof (we'll patch data_offset after computing moof size)
        // We build a placeholder moof with data_offset=0, then fix it.
        let build_moof = |data_offset_base: i32, track_runs: &[FragTrackRun]| -> Vec<u8> {
            let mut moof_content = Vec::new();
            // mfhd
            moof_content.extend(build_mfhd(seq));
            for tr in track_runs {
                moof_content.extend(build_traf(
                    tr.track_id,
                    tr.base_dts,
                    data_offset_base + tr.mdat_offset_start as i32,
                    &tr.entries,
                ));
            }
            write_box(b"moof", &moof_content)
        };

        // First pass: estimate moof size with dummy offset
        let moof_first = build_moof(0, &track_runs);
        let moof_size = moof_first.len() as i32;
        // data_offset in trun is relative to the start of the moof box.
        // The mdat box header is 8 bytes, so first sample byte is at moof_size + 8.
        let data_offset_base = moof_size + 8;
        let moof = build_moof(data_offset_base, &track_runs);

        // Build mdat
        let mdat = write_box(b"mdat", &mdat_payload);

        let mut out = moof;
        out.extend(mdat);

        // Clear pending
        self.pending = PendingSegment::default();

        out
    }

    fn emit_chunked_segment(&mut self) -> Vec<u8> {
        let chunk_duration_ms = self.config.chunk_duration_ms.unwrap_or(200);
        let chunks = self.partition_pending_chunks(chunk_duration_ms);
        let mut out = Vec::new();

        for chunk in chunks {
            if chunk.is_empty() {
                continue;
            }

            let reference_track_id = self
                .track_order
                .iter()
                .copied()
                .find(|track_id| chunk.samples.contains_key(track_id))
                .unwrap_or(1);
            let media_time = chunk.start_pts.unwrap_or(0);
            let seq = self.sequence_number;
            self.sequence_number += 1;

            let (moof, mdat) = build_fragment_boxes(&self.track_order, &chunk.samples, seq);
            out.extend(build_styp_cmfl());
            out.extend(build_prft(reference_track_id, media_time));
            out.extend(moof);
            out.extend(mdat);
        }

        self.pending = PendingSegment::default();
        out
    }

    fn partition_pending_chunks(&self, chunk_duration_ms: u32) -> Vec<PendingSegment> {
        let mut chunks: Vec<PendingSegment> = Vec::new();

        for &track_id in &self.track_order {
            let Some(track) = self.tracks.get(&track_id) else {
                continue;
            };
            let Some(samples) = self.pending.samples.get(&track_id) else {
                continue;
            };

            let first_dts = samples.first().map_or(0, |sample| sample.dts);
            for sample in samples {
                let elapsed_ticks = sample.dts.saturating_sub(first_dts);
                let elapsed_ms =
                    elapsed_ticks.saturating_mul(1000) / u64::from(track.timescale.max(1));
                let chunk_index = usize::try_from(elapsed_ms / u64::from(chunk_duration_ms.max(1)))
                    .unwrap_or(usize::MAX);
                while chunks.len() <= chunk_index {
                    chunks.push(PendingSegment::default());
                }
                chunks[chunk_index].push(sample.clone());
            }
        }

        chunks
    }
}

fn build_fragment_boxes(
    track_order: &[u32],
    samples_by_track: &HashMap<u32, Vec<CmafSample>>,
    sequence_number: u32,
) -> (Vec<u8>, Vec<u8>) {
    let mut mdat_payload: Vec<u8> = Vec::new();
    let mut track_runs: Vec<FragTrackRun> = Vec::new();

    for &track_id in track_order {
        let Some(samples) = samples_by_track.get(&track_id) else {
            continue;
        };
        if samples.is_empty() {
            continue;
        }

        let mdat_offset_start = mdat_payload.len() as u32;
        let base_dts = samples[0].dts;
        let mut entries = Vec::new();

        for sample in samples {
            let flags: u32 = if sample.keyframe {
                0x0200_0000
            } else {
                0x0101_0000
            };
            let pts_offset = i64::try_from(sample.pts)
                .and_then(|pts| i64::try_from(sample.dts).map(|dts| pts - dts))
                .ok()
                .and_then(|offset| i32::try_from(offset).ok())
                .unwrap_or(0);
            entries.push(FragSampleEntry {
                duration: sample.duration,
                size: sample.data.len() as u32,
                flags,
                pts_offset,
            });
            mdat_payload.extend_from_slice(&sample.data);
        }

        track_runs.push(FragTrackRun {
            track_id,
            base_dts,
            mdat_offset_start,
            entries,
        });
    }

    let build_moof = |data_offset_base: i32| -> Vec<u8> {
        let mut moof_content = Vec::new();
        moof_content.extend(build_mfhd(sequence_number));
        for track_run in &track_runs {
            moof_content.extend(build_traf(
                track_run.track_id,
                track_run.base_dts,
                data_offset_base + track_run.mdat_offset_start as i32,
                &track_run.entries,
            ));
        }
        write_box(b"moof", &moof_content)
    };

    let moof_first = build_moof(0);
    let moof_size = moof_first.len() as i32;
    let moof = build_moof(moof_size + 8);
    let mdat = write_box(b"mdat", &mdat_payload);
    (moof, mdat)
}

fn build_styp_cmfl() -> Vec<u8> {
    let mut content = Vec::new();
    content.extend_from_slice(b"cmfl");
    content.extend_from_slice(&write_u32_be(0));
    content.extend_from_slice(b"cmfl");
    content.extend_from_slice(b"cmf2");
    write_box(b"styp", &content)
}

fn build_prft(reference_track_id: u32, media_time: u64) -> Vec<u8> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let ntp_seconds = now.as_secs().saturating_add(2_208_988_800);
    let fractional = ((u128::from(now.subsec_nanos())) << 32) / 1_000_000_000u128;
    let ntp_timestamp = (ntp_seconds << 32) | u64::try_from(fractional).unwrap_or(u64::MAX);

    let mut content = Vec::new();
    content.extend_from_slice(&write_u32_be(reference_track_id));
    content.extend_from_slice(&write_u64_be(ntp_timestamp));
    content.extend_from_slice(&write_u64_be(media_time));
    write_full_box(b"prft", 1, 0, &content)
}

// ─── traf helpers ─────────────────────────────────────────────────────────────

/// A single sample entry within a track run (used by both CMAF and fMP4 muxers).
pub(crate) struct FragSampleEntry {
    pub(crate) duration: u32,
    pub(crate) size: u32,
    pub(crate) flags: u32,
    pub(crate) pts_offset: i32,
}

/// Describes one track's samples within a fragment, with mdat offset info.
pub(crate) struct FragTrackRun {
    pub(crate) track_id: u32,
    pub(crate) base_dts: u64,
    pub(crate) mdat_offset_start: u32,
    pub(crate) entries: Vec<FragSampleEntry>,
}

pub(crate) fn build_mfhd(sequence_number: u32) -> Vec<u8> {
    let mut c = Vec::new();
    c.extend_from_slice(&write_u32_be(sequence_number));
    write_full_box(b"mfhd", 0, 0, &c)
}

pub(crate) fn build_traf(
    track_id: u32,
    base_dts: u64,
    data_offset: i32,
    entries: &[FragSampleEntry],
) -> Vec<u8> {
    let mut c = Vec::new();
    c.extend(build_tfhd(track_id));
    c.extend(build_tfdt(base_dts));
    c.extend(build_trun(data_offset, entries));
    write_box(b"traf", &c)
}

pub(crate) fn build_tfhd(track_id: u32) -> Vec<u8> {
    let mut c = Vec::new();
    c.extend_from_slice(&write_u32_be(track_id));
    // flags: default-base-is-moof (0x020000)
    write_full_box(b"tfhd", 0, 0x02_00_00, &c)
}

pub(crate) fn build_tfdt(base_dts: u64) -> Vec<u8> {
    // version=1 → 64-bit baseMediaDecodeTime
    write_full_box(b"tfdt", 1, 0, &write_u64_be(base_dts))
}

pub(crate) fn build_trun(data_offset: i32, entries: &[FragSampleEntry]) -> Vec<u8> {
    // flags:
    //   0x000001 = data-offset-present
    //   0x000004 = first-sample-flags-present (not used here)
    //   0x000100 = sample-duration-present
    //   0x000200 = sample-size-present
    //   0x000400 = sample-flags-present
    //   0x000800 = sample-composition-time-offsets-present
    let flags: u32 = 0x0000_0B01; // data-offset + duration + size + flags + CTO
    let mut c = Vec::new();
    // sample_count
    c.extend_from_slice(&write_u32_be(entries.len() as u32));
    // data_offset (signed)
    c.extend_from_slice(&data_offset.to_be_bytes());
    for e in entries {
        c.extend_from_slice(&write_u32_be(e.duration));
        c.extend_from_slice(&write_u32_be(e.size));
        c.extend_from_slice(&write_u32_be(e.flags));
        c.extend_from_slice(&e.pts_offset.to_be_bytes());
    }
    // version=1 for signed CTO
    write_full_box(b"trun", 1, flags, &c)
}

// ─── stbl empty boxes ──────────────────────────────────────────────────────────

pub(crate) fn build_empty_stts() -> Vec<u8> {
    write_full_box(b"stts", 0, 0, &write_u32_be(0))
}

pub(crate) fn build_empty_stsc() -> Vec<u8> {
    write_full_box(b"stsc", 0, 0, &write_u32_be(0))
}

pub(crate) fn build_empty_stsz() -> Vec<u8> {
    let mut c = Vec::new();
    c.extend_from_slice(&write_u32_be(0)); // sample_size
    c.extend_from_slice(&write_u32_be(0)); // sample_count
    write_full_box(b"stsz", 0, 0, &c)
}

pub(crate) fn build_empty_stco() -> Vec<u8> {
    write_full_box(b"stco", 0, 0, &write_u32_be(0))
}

// ─── minf sub-boxes ──────────────────────────────────────────────────────────

fn build_vmhd() -> Vec<u8> {
    let mut c = Vec::new();
    c.extend_from_slice(&[0u8; 4]); // graphicsMode + opcolor
    write_full_box(b"vmhd", 0, 1, &c)
}

fn build_smhd() -> Vec<u8> {
    write_full_box(b"smhd", 0, 0, &[0u8; 4]) // balance + reserved
}

fn build_nmhd() -> Vec<u8> {
    write_full_box(b"nmhd", 0, 0, &[])
}

fn build_dinf() -> Vec<u8> {
    // dref with one self-referencing entry
    let mut dref_c = Vec::new();
    dref_c.extend_from_slice(&write_u32_be(1)); // entry_count
                                                // url entry with self-contained flag
    let url = write_full_box(b"url ", 0, 1, &[]);
    dref_c.extend(url);
    let dref = write_full_box(b"dref", 0, 0, &dref_c);
    write_box(b"dinf", &dref)
}

// ─── Box writing primitives ──────────────────────────────────────────────────

/// Writes a standard box: `[size:u32 BE][fourcc:4][content]`.
///
/// If `content.len() + 8 > u32::MAX` the box size overflows — callers must
/// ensure reasonable content sizes.
#[must_use]
pub fn write_box(fourcc: &[u8; 4], content: &[u8]) -> Vec<u8> {
    let size = (content.len() + 8) as u32;
    let mut out = Vec::with_capacity(size as usize);
    out.extend_from_slice(&size.to_be_bytes());
    out.extend_from_slice(fourcc);
    out.extend_from_slice(content);
    out
}

/// Writes a FullBox: `[size:u32 BE][fourcc:4][version:u8][flags:u24][content]`.
#[must_use]
pub fn write_full_box(fourcc: &[u8; 4], version: u8, flags: u32, content: &[u8]) -> Vec<u8> {
    let mut header = Vec::with_capacity(4);
    header.push(version);
    header.extend_from_slice(&(flags & 0x00FF_FFFF).to_be_bytes()[1..]);
    let mut full_content = header;
    full_content.extend_from_slice(content);
    write_box(fourcc, &full_content)
}

/// Returns the big-endian encoding of `v`.
#[must_use]
pub const fn write_u32_be(v: u32) -> [u8; 4] {
    v.to_be_bytes()
}

/// Returns the big-endian encoding of `v`.
#[must_use]
pub const fn write_u64_be(v: u64) -> [u8; 8] {
    v.to_be_bytes()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_video_track(id: u32) -> CmafTrack {
        CmafTrack {
            track_id: id,
            track_type: TrackType::Video,
            codec_fourcc: *b"av01",
            timescale: 90000,
            width: Some(1920),
            height: Some(1080),
            sample_rate: None,
            channels: None,
            extradata: vec![0x81, 0x00, 0x0C, 0x00],
        }
    }

    fn make_audio_track(id: u32) -> CmafTrack {
        CmafTrack {
            track_id: id,
            track_type: TrackType::Audio,
            codec_fourcc: *b"Opus",
            timescale: 48000,
            width: None,
            height: None,
            sample_rate: Some(48000),
            channels: Some(2),
            extradata: vec![0x4F, 0x70, 0x75, 0x73],
        }
    }

    // 1. write_u32_be round-trip
    #[test]
    fn test_write_u32_be() {
        assert_eq!(write_u32_be(0xDEAD_BEEF), [0xDE, 0xAD, 0xBE, 0xEF]);
        assert_eq!(write_u32_be(0), [0, 0, 0, 0]);
    }

    // 2. write_u64_be round-trip
    #[test]
    fn test_write_u64_be() {
        let v: u64 = 0x0102_0304_0506_0708;
        let b = write_u64_be(v);
        assert_eq!(b, [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]);
    }

    // 3. write_box size prefix
    #[test]
    fn test_write_box_size() {
        let content = b"hello";
        let b = write_box(b"test", content);
        // size = 5 + 8 = 13
        assert_eq!(&b[0..4], &13u32.to_be_bytes());
        assert_eq!(&b[4..8], b"test");
        assert_eq!(&b[8..], b"hello");
    }

    // 4. write_full_box structure
    #[test]
    fn test_write_full_box() {
        let b = write_full_box(b"mvhd", 0, 0, &[]);
        // size=12 (8 header + 4 version/flags), fourcc, version=0, flags=0
        assert_eq!(&b[0..4], &12u32.to_be_bytes());
        assert_eq!(&b[4..8], b"mvhd");
        assert_eq!(b[8], 0); // version
        assert_eq!(&b[9..12], &[0u8; 3]); // flags
    }

    // 5. ftyp box starts with correct major brand
    #[test]
    fn test_init_ftyp_brand() {
        let config = CmafConfig {
            brand: CmafBrand::CmafCmav1,
            ..CmafConfig::default()
        };
        let mut muxer = CmafMuxer::new(config);
        muxer.add_track(make_video_track(1));
        let init = muxer.write_init_segment();
        // ftyp is the first box
        assert_eq!(&init[4..8], b"ftyp");
        // major brand at bytes 8..12
        assert_eq!(&init[8..12], b"cav1");
    }

    // 6. init segment contains moov box
    #[test]
    fn test_init_contains_moov() {
        let mut muxer = CmafMuxer::new(CmafConfig::default());
        muxer.add_track(make_video_track(1));
        let init = muxer.write_init_segment();
        // Find moov fourcc somewhere after ftyp
        let found = init.windows(4).any(|w| w == b"moov");
        assert!(found, "moov box must be present in init segment");
    }

    // 7. add_track returns correct track_id
    #[test]
    fn test_add_track_returns_id() {
        let mut muxer = CmafMuxer::new(CmafConfig::default());
        let id = muxer.add_track(make_video_track(42));
        assert_eq!(id, 42);
    }

    // 8. media segment starts with moof
    #[test]
    fn test_media_segment_starts_with_moof() {
        let mut muxer = CmafMuxer::new(CmafConfig::default());
        muxer.add_track(make_video_track(1));
        let samples = vec![CmafSample {
            track_id: 1,
            pts: 0,
            dts: 0,
            duration: 3000,
            data: vec![0u8; 100],
            keyframe: true,
        }];
        let seg = muxer.write_media_segment(&samples);
        assert_eq!(&seg[4..8], b"moof", "media segment must start with moof");
    }

    // 9. media segment contains mdat
    #[test]
    fn test_media_segment_contains_mdat() {
        let mut muxer = CmafMuxer::new(CmafConfig::default());
        muxer.add_track(make_video_track(1));
        let samples = vec![CmafSample {
            track_id: 1,
            pts: 0,
            dts: 0,
            duration: 3000,
            data: vec![0xAA; 50],
            keyframe: true,
        }];
        let seg = muxer.write_media_segment(&samples);
        let has_mdat = seg.windows(4).any(|w| w == b"mdat");
        assert!(has_mdat, "media segment must contain mdat");
    }

    // 10. sequence number increments
    #[test]
    fn test_sequence_number_increments() {
        let mut muxer = CmafMuxer::new(CmafConfig::default());
        muxer.add_track(make_video_track(1));
        let sample = CmafSample {
            track_id: 1,
            pts: 0,
            dts: 0,
            duration: 3000,
            data: vec![0u8; 10],
            keyframe: true,
        };
        muxer.write_media_segment(std::slice::from_ref(&sample));
        let sample2 = CmafSample {
            pts: 3000,
            dts: 3000,
            ..sample
        };
        muxer.write_media_segment(&[sample2]);
        // After two segments, sequence_number should be 3
        assert_eq!(muxer.sequence_number, 3);
    }

    // 11. flush_segment returns None when empty
    #[test]
    fn test_flush_empty() {
        let mut muxer = CmafMuxer::new(CmafConfig::default());
        muxer.add_track(make_video_track(1));
        assert!(muxer.flush_segment().is_none());
    }

    // 12. multi-track init segment contains both trak boxes
    #[test]
    fn test_init_segment_two_tracks() {
        let mut muxer = CmafMuxer::new(CmafConfig::default());
        muxer.add_track(make_video_track(1));
        muxer.add_track(make_audio_track(2));
        let init = muxer.write_init_segment();
        // Count occurrences of "trak"
        let count = init.windows(4).filter(|w| *w == b"trak").count();
        assert_eq!(count, 2, "init segment must contain two trak boxes");
    }
}
