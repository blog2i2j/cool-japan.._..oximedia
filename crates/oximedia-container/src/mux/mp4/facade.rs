//! High-level MP4 muxer facade.
//!
//! Provides [`Mp4MuxerConfig`], [`Mp4Track`], [`FacadeMp4Sample`] and the
//! convenience constructor [`FacadeMp4Muxer`] that writes a complete
//! `ftyp + moov + mdat` structure to an in-memory `Vec<u8>`.
//!
//! This façade intentionally mirrors the minimal API described in the
//! `oximedia-container` TODO:
//!
//! - `Mp4MuxerConfig { timescale, creation_time }`
//! - `Mp4Track { id, codec, width, height, timescale, samples }`
//! - `FacadeMp4Sample { pts, dts, duration, is_sync, data }`
//! - `FacadeMp4Muxer::new(config) -> Self`
//! - `add_track(&mut self, track: Mp4Track) -> u32`
//! - `write_to_vec(&self) -> Vec<u8>`
//! - `is_mp4(data: &[u8]) -> bool`

#![forbid(unsafe_code)]
#![allow(clippy::cast_possible_truncation)]

// ─── Helper: write big-endian u32 ───────────────────────────────────────────

#[inline]
fn u32_be(v: u32) -> [u8; 4] {
    v.to_be_bytes()
}

#[inline]
fn u16_be(v: u16) -> [u8; 2] {
    v.to_be_bytes()
}

/// Write a 4-byte size + 4-byte fourcc + content box.
fn write_box(fourcc: &[u8; 4], content: &[u8]) -> Vec<u8> {
    let size = 8u32 + content.len() as u32;
    let mut out = Vec::with_capacity(size as usize);
    out.extend_from_slice(&u32_be(size));
    out.extend_from_slice(fourcc);
    out.extend_from_slice(content);
    out
}

/// Write a full-box (version + flags before content).
fn write_full_box(fourcc: &[u8; 4], version: u8, flags: u32, content: &[u8]) -> Vec<u8> {
    let mut body = Vec::with_capacity(4 + content.len());
    body.push(version);
    body.extend_from_slice(&[(flags >> 16) as u8, (flags >> 8) as u8, flags as u8]);
    body.extend_from_slice(content);
    write_box(fourcc, &body)
}

// ─── Public types ────────────────────────────────────────────────────────────

/// Configuration for [`FacadeMp4Muxer`].
#[derive(Debug, Clone)]
pub struct Mp4MuxerConfig {
    /// Movie timescale (ticks per second). Typical: 1000.
    pub timescale: u32,
    /// Creation time in seconds since midnight, January 1, 1904 (UTC).
    /// Use 0 for unspecified.
    pub creation_time: u64,
}

impl Default for Mp4MuxerConfig {
    fn default() -> Self {
        Self {
            timescale: 1000,
            creation_time: 0,
        }
    }
}

impl Mp4MuxerConfig {
    /// Creates a new `Mp4MuxerConfig` with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

/// A single compressed media sample for use with [`Mp4Track`].
#[derive(Debug, Clone)]
pub struct FacadeMp4Sample {
    /// Presentation timestamp in the track's timescale.
    pub pts: u64,
    /// Decode timestamp in the track's timescale.
    pub dts: u64,
    /// Sample duration in the track's timescale.
    pub duration: u32,
    /// Whether this is a sync (key) sample.
    pub is_sync: bool,
    /// Raw compressed bitstream.
    pub data: Vec<u8>,
}

impl FacadeMp4Sample {
    /// Creates a new [`FacadeMp4Sample`].
    #[must_use]
    pub fn new(pts: u64, dts: u64, duration: u32, is_sync: bool, data: Vec<u8>) -> Self {
        Self {
            pts,
            dts,
            duration,
            is_sync,
            data,
        }
    }
}

/// A single media track for [`FacadeMp4Muxer`].
///
/// Carries codec metadata and all samples for one stream (video, audio, or
/// subtitle) that will be written into an MP4 container.
#[derive(Debug, Clone)]
pub struct Mp4Track {
    /// 1-based track identifier.
    pub id: u32,
    /// Codec string (e.g. `"av01"`, `"vp09"`, `"Opus"`, `"mp4a"`).
    pub codec: String,
    /// Display width in pixels (video only; use 0 for audio).
    pub width: u32,
    /// Display height in pixels (video only; use 0 for audio).
    pub height: u32,
    /// Track timescale (ticks per second).
    pub timescale: u32,
    /// All samples in decode order.
    pub samples: Vec<FacadeMp4Sample>,
}

impl Mp4Track {
    /// Creates a new video track.
    #[must_use]
    pub fn video(
        id: u32,
        codec: impl Into<String>,
        width: u32,
        height: u32,
        timescale: u32,
    ) -> Self {
        Self {
            id,
            codec: codec.into(),
            width,
            height,
            timescale,
            samples: Vec::new(),
        }
    }

    /// Creates a new audio track.
    #[must_use]
    pub fn audio(id: u32, codec: impl Into<String>, timescale: u32) -> Self {
        Self {
            id,
            codec: codec.into(),
            width: 0,
            height: 0,
            timescale,
            samples: Vec::new(),
        }
    }

    /// Returns `true` if this is a video track (non-zero dimensions).
    #[must_use]
    pub fn is_video(&self) -> bool {
        self.width > 0 && self.height > 0
    }

    /// Total duration in track timescale ticks.
    fn total_duration(&self) -> u64 {
        self.samples.iter().map(|s| u64::from(s.duration)).sum()
    }

    /// Returns the four-character code from the codec string (first 4 bytes).
    fn fourcc(&self) -> [u8; 4] {
        let bytes = self.codec.as_bytes();
        [
            bytes.first().copied().unwrap_or(b' '),
            bytes.get(1).copied().unwrap_or(b' '),
            bytes.get(2).copied().unwrap_or(b' '),
            bytes.get(3).copied().unwrap_or(b' '),
        ]
    }
}

// ─── FacadeMp4Muxer ──────────────────────────────────────────────────────────

/// High-level MP4 muxer.
///
/// Accumulates tracks and their samples, then serialises the complete
/// `ftyp + moov + mdat` structure on demand via [`write_to_vec`].
///
/// # Example
///
/// ```ignore
/// use oximedia_container::mux::mp4::{
///     Mp4MuxerConfig, Mp4Track, FacadeMp4Sample, FacadeMp4Muxer,
/// };
///
/// let mut muxer = FacadeMp4Muxer::new(Mp4MuxerConfig::default());
///
/// let mut track = Mp4Track::video(1, "av01", 1920, 1080, 90_000);
/// track.samples.push(FacadeMp4Sample::new(0, 0, 3000, true, vec![0u8; 256]));
///
/// muxer.add_track(track);
/// let mp4 = muxer.write_to_vec();
/// assert!(oximedia_container::mux::mp4::is_mp4(&mp4));
/// ```
///
/// [`write_to_vec`]: FacadeMp4Muxer::write_to_vec
#[derive(Debug)]
pub struct FacadeMp4Muxer {
    config: Mp4MuxerConfig,
    tracks: Vec<Mp4Track>,
    next_id: u32,
}

impl FacadeMp4Muxer {
    /// Creates a new muxer with the given configuration.
    #[must_use]
    pub fn new(config: Mp4MuxerConfig) -> Self {
        Self {
            config,
            tracks: Vec::new(),
            next_id: 1,
        }
    }

    /// Adds a track to the muxer.
    ///
    /// Returns the 1-based track ID assigned to the track.  If `track.id` is
    /// non-zero it is used as-is; otherwise a monotonically increasing ID is
    /// assigned.
    pub fn add_track(&mut self, mut track: Mp4Track) -> u32 {
        if track.id == 0 {
            track.id = self.next_id;
        }
        let id = track.id;
        if id >= self.next_id {
            self.next_id = id + 1;
        }
        self.tracks.push(track);
        id
    }

    /// Writes the complete MP4 file to a `Vec<u8>`.
    ///
    /// Layout: `ftyp + mdat + moov`
    ///
    /// The `mdat` box is placed before `moov` so that the chunk offsets stored
    /// in the `stco` box can reference the absolute byte offset of each
    /// sample's first byte.  The muxer performs two passes:
    /// 1. Compute mdat content and its byte offset.
    /// 2. Build moov referencing those offsets.
    #[must_use]
    pub fn write_to_vec(&self) -> Vec<u8> {
        // ── ftyp ─────────────────────────────────────────────────────────────
        let ftyp = self.build_ftyp();

        // ── mdat payload ─────────────────────────────────────────────────────
        // Concatenate all sample data in track order.
        let mut mdat_payload: Vec<u8> = Vec::new();
        // Per-track: list of (absolute byte offset within mdat_payload)
        let mut track_offsets: Vec<Vec<u64>> = vec![Vec::new(); self.tracks.len()];

        for (ti, track) in self.tracks.iter().enumerate() {
            for sample in &track.samples {
                track_offsets[ti].push(mdat_payload.len() as u64);
                mdat_payload.extend_from_slice(&sample.data);
            }
        }

        let mdat_header_size = 8u64; // size(4) + fourcc(4)
        let mdat_start = ftyp.len() as u64 + mdat_header_size;

        // Adjust offsets: add ftyp.len() + mdat_header so they point to the
        // absolute file position of each sample.
        for offsets in &mut track_offsets {
            for off in offsets.iter_mut() {
                *off += mdat_start;
            }
        }

        let mdat = write_box(b"mdat", &mdat_payload);

        // ── moov ─────────────────────────────────────────────────────────────
        let moov = self.build_moov(&track_offsets);

        // ── combine ──────────────────────────────────────────────────────────
        let mut out = Vec::with_capacity(ftyp.len() + mdat.len() + moov.len());
        out.extend(ftyp);
        out.extend(mdat);
        out.extend(moov);
        out
    }

    // ── Private builders ─────────────────────────────────────────────────────

    fn build_ftyp(&self) -> Vec<u8> {
        // major_brand = mp42, minor_version = 0, compatible = [mp42, mp41, isom]
        let mut content = Vec::with_capacity(20);
        content.extend_from_slice(b"mp42"); // major brand
        content.extend_from_slice(&u32_be(0)); // minor version
        content.extend_from_slice(b"mp42");
        content.extend_from_slice(b"mp41");
        content.extend_from_slice(b"isom");
        write_box(b"ftyp", &content)
    }

    fn build_moov(&self, track_offsets: &[Vec<u64>]) -> Vec<u8> {
        let mut content = Vec::new();

        // ── mvhd ─────────────────────────────────────────────────────────────
        content.extend(self.build_mvhd());

        // ── trak per track ───────────────────────────────────────────────────
        for (ti, track) in self.tracks.iter().enumerate() {
            let offsets = track_offsets.get(ti).map(Vec::as_slice).unwrap_or(&[]);
            content.extend(self.build_trak(track, offsets));
        }

        write_box(b"moov", &content)
    }

    fn build_mvhd(&self) -> Vec<u8> {
        // version 1 (64-bit timestamps)
        let ct = self.config.creation_time;
        let ts = u64::from(self.config.timescale);

        // Compute overall duration: max of all tracks in movie timescale
        let duration = self.tracks.iter().fold(0u64, |acc, track| {
            let track_dur_movie = if track.timescale > 0 {
                track.total_duration() * ts / u64::from(track.timescale)
            } else {
                0
            };
            acc.max(track_dur_movie)
        });

        let next_track_id = self
            .tracks
            .iter()
            .map(|t| t.id)
            .max()
            .unwrap_or(0)
            + 1;

        let mut c = Vec::with_capacity(112);
        c.extend_from_slice(&ct.to_be_bytes()); // creation_time
        c.extend_from_slice(&ct.to_be_bytes()); // modification_time
        c.extend_from_slice(&(self.config.timescale).to_be_bytes()); // timescale (u32)
        c.extend_from_slice(&duration.to_be_bytes()); // duration
        // rate (1.0 = 0x00010000), volume (1.0 = 0x0100)
        c.extend_from_slice(&[0x00, 0x01, 0x00, 0x00]); // rate
        c.extend_from_slice(&[0x01, 0x00]); // volume
        c.extend_from_slice(&[0u8; 10]); // reserved
        // Unity matrix
        c.extend_from_slice(&[
            0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // row1
            0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // row2
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, // row3
        ]);
        c.extend_from_slice(&[0u8; 24]); // pre_defined
        c.extend_from_slice(&u32_be(next_track_id));
        write_full_box(b"mvhd", 1, 0, &c)
    }

    fn build_trak(&self, track: &Mp4Track, chunk_offsets: &[u64]) -> Vec<u8> {
        let mut content = Vec::new();
        content.extend(self.build_tkhd(track));
        content.extend(self.build_mdia(track, chunk_offsets));
        write_box(b"trak", &content)
    }

    fn build_tkhd(&self, track: &Mp4Track) -> Vec<u8> {
        let ct = self.config.creation_time;
        let duration_in_movie_ts = if track.timescale > 0 {
            track.total_duration() * u64::from(self.config.timescale)
                / u64::from(track.timescale)
        } else {
            0
        };

        // flags: 0x3 = track enabled + in movie
        let mut c = Vec::with_capacity(92);
        c.extend_from_slice(&ct.to_be_bytes()); // creation_time
        c.extend_from_slice(&ct.to_be_bytes()); // modification_time
        c.extend_from_slice(&u32_be(track.id));
        c.extend_from_slice(&u32_be(0)); // reserved
        c.extend_from_slice(&duration_in_movie_ts.to_be_bytes());
        c.extend_from_slice(&[0u8; 8]); // reserved
        c.extend_from_slice(&[0u8; 2]); // layer
        c.extend_from_slice(&[0u8; 2]); // alternate_group
        // volume: 0x0100 for audio, 0 for video
        if track.is_video() {
            c.extend_from_slice(&[0x00, 0x00]);
        } else {
            c.extend_from_slice(&[0x01, 0x00]);
        }
        c.extend_from_slice(&[0u8; 2]); // reserved
        // Unity matrix
        c.extend_from_slice(&[
            0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00,
        ]);
        // width/height as 16.16 fixed point
        c.extend_from_slice(&u32_be(track.width << 16));
        c.extend_from_slice(&u32_be(track.height << 16));
        write_full_box(b"tkhd", 1, 0x0003, &c)
    }

    fn build_mdia(&self, track: &Mp4Track, chunk_offsets: &[u64]) -> Vec<u8> {
        let mut content = Vec::new();
        content.extend(self.build_mdhd(track));
        content.extend(self.build_hdlr(track));
        content.extend(self.build_minf(track, chunk_offsets));
        write_box(b"mdia", &content)
    }

    fn build_mdhd(&self, track: &Mp4Track) -> Vec<u8> {
        let ct = self.config.creation_time;
        let duration = track.total_duration();
        let mut c = Vec::with_capacity(36);
        c.extend_from_slice(&ct.to_be_bytes());
        c.extend_from_slice(&ct.to_be_bytes());
        c.extend_from_slice(&u32_be(track.timescale));
        c.extend_from_slice(&duration.to_be_bytes());
        // language = 'und' (0x55c4)
        c.extend_from_slice(&[0x55, 0xc4]);
        c.extend_from_slice(&[0u8; 2]); // pre_defined
        write_full_box(b"mdhd", 1, 0, &c)
    }

    fn build_hdlr(&self, track: &Mp4Track) -> Vec<u8> {
        let handler = if track.is_video() {
            b"vide"
        } else {
            b"soun"
        };
        let name = if track.is_video() {
            b"VideoHandler\0" as &[u8]
        } else {
            b"SoundHandler\0" as &[u8]
        };
        let mut c = Vec::with_capacity(25 + name.len());
        c.extend_from_slice(&u32_be(0)); // pre_defined
        c.extend_from_slice(handler);
        c.extend_from_slice(&[0u8; 12]); // reserved
        c.extend_from_slice(name);
        write_full_box(b"hdlr", 0, 0, &c)
    }

    fn build_minf(&self, track: &Mp4Track, chunk_offsets: &[u64]) -> Vec<u8> {
        let mut content = Vec::new();
        // Media information header
        if track.is_video() {
            // vmhd
            let mut vmhd_c = Vec::with_capacity(8);
            vmhd_c.extend_from_slice(&[0u8; 8]); // graphicsMode + opcolor
            content.extend(write_full_box(b"vmhd", 0, 0x0001, &vmhd_c));
        } else {
            // smhd
            let mut smhd_c = Vec::with_capacity(4);
            smhd_c.extend_from_slice(&[0u8; 4]); // balance + reserved
            content.extend(write_full_box(b"smhd", 0, 0, &smhd_c));
        }
        // dinf + dref
        let dref_entry = write_full_box(b"url ", 0, 0x0001, &[]);
        let mut dref_c = Vec::new();
        dref_c.extend_from_slice(&u32_be(1)); // entry count
        dref_c.extend(dref_entry);
        let dref = write_full_box(b"dref", 0, 0, &dref_c);
        let dinf = write_box(b"dinf", &dref);
        content.extend(dinf);
        // stbl
        content.extend(self.build_stbl(track, chunk_offsets));
        write_box(b"minf", &content)
    }

    fn build_stbl(&self, track: &Mp4Track, chunk_offsets: &[u64]) -> Vec<u8> {
        let mut content = Vec::new();
        content.extend(self.build_stsd(track));
        content.extend(build_stts(track));
        content.extend(build_stsc(track));
        content.extend(build_stsz(track));
        content.extend(build_stco(chunk_offsets));
        if track.samples.iter().any(|s| s.is_sync) {
            content.extend(build_stss(track));
        }
        write_box(b"stbl", &content)
    }

    fn build_stsd(&self, track: &Mp4Track) -> Vec<u8> {
        let fourcc = track.fourcc();
        let entry = if track.is_video() {
            build_video_entry(track, &fourcc)
        } else {
            build_audio_entry(track, &fourcc)
        };
        let mut c = Vec::new();
        c.extend_from_slice(&u32_be(1)); // entry count
        c.extend(entry);
        write_full_box(b"stsd", 0, 0, &c)
    }
}

// ─── Sample table helpers ────────────────────────────────────────────────────

fn build_video_entry(track: &Mp4Track, fourcc: &[u8; 4]) -> Vec<u8> {
    let mut c = Vec::new();
    c.extend_from_slice(&[0u8; 6]); // reserved
    c.extend_from_slice(&u16_be(1)); // data_reference_index
    c.extend_from_slice(&[0u8; 16]); // pre_defined + reserved
    c.extend_from_slice(&u16_be(track.width as u16));
    c.extend_from_slice(&u16_be(track.height as u16));
    c.extend_from_slice(&[0x00, 0x48, 0x00, 0x00]); // horiz_res 72dpi
    c.extend_from_slice(&[0x00, 0x48, 0x00, 0x00]); // vert_res 72dpi
    c.extend_from_slice(&[0u8; 4]); // reserved
    c.extend_from_slice(&u16_be(1)); // frame count
    c.extend_from_slice(&[0u8; 32]); // compressor name
    c.extend_from_slice(&u16_be(0x0018)); // depth
    c.extend_from_slice(&[0xFF, 0xFF]); // pre_defined
    write_box(fourcc, &c)
}

fn build_audio_entry(track: &Mp4Track, fourcc: &[u8; 4]) -> Vec<u8> {
    let _ = track;
    let mut c = Vec::new();
    c.extend_from_slice(&[0u8; 6]); // reserved
    c.extend_from_slice(&u16_be(1)); // data_reference_index
    c.extend_from_slice(&[0u8; 8]); // reserved
    c.extend_from_slice(&u16_be(2)); // channel_count
    c.extend_from_slice(&u16_be(16)); // sample_size
    c.extend_from_slice(&[0u8; 4]); // pre_defined + reserved
    c.extend_from_slice(&u32_be(track.timescale << 16)); // sample_rate (16.16)
    write_box(fourcc, &c)
}

/// stts: time-to-sample
fn build_stts(track: &Mp4Track) -> Vec<u8> {
    // Run-length encode (duration, count) pairs
    let mut entries: Vec<(u32, u32)> = Vec::new();
    for sample in &track.samples {
        if let Some(last) = entries.last_mut() {
            if last.0 == sample.duration {
                last.1 += 1;
                continue;
            }
        }
        entries.push((sample.duration, 1));
    }
    let mut c = Vec::new();
    c.extend_from_slice(&u32_be(entries.len() as u32));
    for (dur, count) in &entries {
        c.extend_from_slice(&u32_be(*count));
        c.extend_from_slice(&u32_be(*dur));
    }
    write_full_box(b"stts", 0, 0, &c)
}

/// stsc: sample-to-chunk (all samples in one chunk each → 1-sample-per-chunk layout)
fn build_stsc(track: &Mp4Track) -> Vec<u8> {
    // One entry: first_chunk=1, samples_per_chunk=1, sample_description_index=1
    let _ = track;
    let mut c = Vec::new();
    c.extend_from_slice(&u32_be(1)); // entry_count
    c.extend_from_slice(&u32_be(1)); // first_chunk
    c.extend_from_slice(&u32_be(1)); // samples_per_chunk
    c.extend_from_slice(&u32_be(1)); // sample_description_index
    write_full_box(b"stsc", 0, 0, &c)
}

/// stsz: sample sizes
fn build_stsz(track: &Mp4Track) -> Vec<u8> {
    let mut c = Vec::new();
    c.extend_from_slice(&u32_be(0)); // sample_size (0 = variable)
    c.extend_from_slice(&u32_be(track.samples.len() as u32));
    for sample in &track.samples {
        c.extend_from_slice(&u32_be(sample.data.len() as u32));
    }
    write_full_box(b"stsz", 0, 0, &c)
}

/// stco: chunk offsets (one offset per sample in 1-sample-per-chunk layout)
fn build_stco(chunk_offsets: &[u64]) -> Vec<u8> {
    let mut c = Vec::new();
    c.extend_from_slice(&u32_be(chunk_offsets.len() as u32));
    for &off in chunk_offsets {
        // Use 32-bit stco; values are assumed to fit (file < 4 GiB)
        c.extend_from_slice(&u32_be(off as u32));
    }
    write_full_box(b"stco", 0, 0, &c)
}

/// stss: sync sample table (keyframes)
fn build_stss(track: &Mp4Track) -> Vec<u8> {
    let sync_indices: Vec<u32> = track
        .samples
        .iter()
        .enumerate()
        .filter_map(|(i, s)| if s.is_sync { Some(i as u32 + 1) } else { None })
        .collect();
    let mut c = Vec::new();
    c.extend_from_slice(&u32_be(sync_indices.len() as u32));
    for idx in &sync_indices {
        c.extend_from_slice(&u32_be(*idx));
    }
    write_full_box(b"stss", 0, 0, &c)
}

// ─── Probe ───────────────────────────────────────────────────────────────────

/// Returns `true` if `data` appears to be a valid MP4/ISOBMFF file.
///
/// Checks for a valid `ftyp` box at the start, or a `moov` box at any of
/// the first 64 bytes (permissive detection).
#[must_use]
pub fn is_mp4(data: &[u8]) -> bool {
    if data.len() < 8 {
        return false;
    }
    // Check for ftyp box in first 12 bytes
    if data.len() >= 12 && &data[4..8] == b"ftyp" {
        return true;
    }
    // Check for moov box at the very start (used by some encoders that omit ftyp)
    if &data[4..8] == b"moov" {
        return true;
    }
    // Scan the first 64 bytes for a moov/ftyp indicator
    let scan_len = data.len().min(64);
    let mut pos = 0;
    while pos + 8 <= scan_len {
        let size = u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
        let tag = &data[pos + 4..pos + 8];
        if tag == b"ftyp" || tag == b"moov" {
            return true;
        }
        if size < 8 {
            break;
        }
        pos += size as usize;
    }
    false
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_simple_muxer() -> FacadeMp4Muxer {
        FacadeMp4Muxer::new(Mp4MuxerConfig {
            timescale: 1000,
            creation_time: 0,
        })
    }

    #[test]
    fn test_is_mp4_ftyp() {
        // An ftyp box starts at offset 0
        let mut data = vec![0u8; 32];
        // size = 20
        data[0..4].copy_from_slice(&20u32.to_be_bytes());
        data[4..8].copy_from_slice(b"ftyp");
        data[8..12].copy_from_slice(b"mp42");
        assert!(is_mp4(&data));
    }

    #[test]
    fn test_is_mp4_moov() {
        let mut data = vec![0u8; 32];
        data[0..4].copy_from_slice(&20u32.to_be_bytes());
        data[4..8].copy_from_slice(b"moov");
        assert!(is_mp4(&data));
    }

    #[test]
    fn test_is_mp4_rejects_garbage() {
        let data = b"RIFF____WAVEfmt ";
        assert!(!is_mp4(data));
    }

    #[test]
    fn test_write_to_vec_empty_tracks() {
        let muxer = make_simple_muxer();
        let data = muxer.write_to_vec();
        // Must start with a valid ftyp
        assert!(data.len() >= 8);
        assert_eq!(&data[4..8], b"ftyp");
        assert!(is_mp4(&data));
    }

    #[test]
    fn test_add_track_returns_id() {
        let mut muxer = make_simple_muxer();
        let track = Mp4Track::video(0, "av01", 1920, 1080, 90_000);
        let id = muxer.add_track(track);
        assert_eq!(id, 1);

        let track2 = Mp4Track::audio(0, "Opus", 48_000);
        let id2 = muxer.add_track(track2);
        assert_eq!(id2, 2);
    }

    #[test]
    fn test_write_to_vec_with_video_track() {
        let mut muxer = make_simple_muxer();
        let mut track = Mp4Track::video(1, "av01", 1920, 1080, 90_000);
        track.samples.push(FacadeMp4Sample::new(0, 0, 3000, true, vec![0xAB; 128]));
        track.samples.push(FacadeMp4Sample::new(3000, 3000, 3000, false, vec![0xCD; 64]));
        muxer.add_track(track);

        let data = muxer.write_to_vec();
        assert!(is_mp4(&data));
        // Must contain moov box somewhere
        let moov_pos = data.windows(4).position(|w| w == b"moov");
        assert!(moov_pos.is_some(), "no moov box found");
        // Must contain mdat box
        let mdat_pos = data.windows(4).position(|w| w == b"mdat");
        assert!(mdat_pos.is_some(), "no mdat box found");
    }

    #[test]
    fn test_write_to_vec_audio_track() {
        let mut muxer = make_simple_muxer();
        let mut track = Mp4Track::audio(1, "Opus", 48_000);
        track.samples.push(FacadeMp4Sample::new(0, 0, 960, true, vec![0x01; 64]));
        muxer.add_track(track);

        let data = muxer.write_to_vec();
        assert!(is_mp4(&data));
        // smhd box must be present
        let smhd_pos = data.windows(4).position(|w| w == b"smhd");
        assert!(smhd_pos.is_some(), "no smhd box for audio track");
    }

    #[test]
    fn test_ftyp_brands() {
        let muxer = make_simple_muxer();
        let data = muxer.write_to_vec();
        // ftyp major brand at offset 8
        assert_eq!(&data[8..12], b"mp42");
        // Compatible brands: mp42, mp41, isom at offsets 16, 20, 24
        assert_eq!(&data[16..20], b"mp42");
        assert_eq!(&data[20..24], b"mp41");
        assert_eq!(&data[24..28], b"isom");
    }

    #[test]
    fn test_mp4_track_is_video() {
        let v = Mp4Track::video(1, "vp09", 640, 480, 90_000);
        assert!(v.is_video());
        let a = Mp4Track::audio(2, "Opus", 48_000);
        assert!(!a.is_video());
    }
}
