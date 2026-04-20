//! MP4/ISOBMFF container demuxer.
//!
//! Implements ISO Base Media File Format (ISO/IEC 14496-12) parsing.
//!
//! **Important**: Only royalty-free codecs are supported:
//! - Video: AV1, VP9
//! - Audio: Opus, FLAC, Vorbis
//!
//! Patent-encumbered codecs (H.264, H.265, AAC, AC-3, E-AC-3, DTS) are
//! rejected with a [`PatentViolation`](oximedia_core::OxiError::PatentViolation) error.
//!
//! # Example
//!
//! ```ignore
//! use oximedia_container::demux::Mp4Demuxer;
//! use oximedia_io::FileSource;
//!
//! let source = FileSource::open("video.mp4").await?;
//! let mut demuxer = Mp4Demuxer::new(source);
//!
//! let probe = demuxer.probe().await?;
//! println!("Detected: {:?}", probe.format);
//!
//! for stream in demuxer.streams() {
//!     println!("Stream {}: {:?}", stream.index, stream.codec);
//! }
//!
//! while let Ok(packet) = demuxer.read_packet().await {
//!     println!("Packet: stream={}, size={}",
//!              packet.stream_index, packet.size());
//! }
//! ```

mod atom;
mod boxes;

pub use atom::Mp4Atom;
pub use boxes::{
    BoxHeader, BoxType, CttsEntry, FtypBox, MoovBox, MvhdBox, StscEntry, SttsEntry, TkhdBox,
    TrakBox,
};

use std::io::SeekFrom;

use async_trait::async_trait;
use bytes::Bytes;
use oximedia_core::{CodecId, OxiError, OxiResult, Rational, Timestamp};
use oximedia_io::MediaSource;

use crate::demux::Demuxer;
use crate::DecodeSkipCursor;
use crate::{CodecParams, ContainerFormat, Metadata, Packet, PacketFlags, ProbeResult, StreamInfo};

/// MP4/ISOBMFF demuxer supporting AV1 and VP9 only.
///
/// This demuxer parses MP4 containers but only allows playback of
/// royalty-free codecs. Attempting to demux files containing H.264,
/// H.265, AAC, or other patent-encumbered codecs will result in an error.
///
/// # Supported Codecs
///
/// | Type  | Codecs           |
/// |-------|------------------|
/// | Video | AV1, VP9         |
/// | Audio | Opus, FLAC, Vorbis |
///
/// # Patent Protection
///
/// When a patent-encumbered codec is detected during probing, the demuxer
/// returns [`OxiError::PatentViolation`] with the codec name. This ensures
/// users are immediately aware that the file cannot be processed.
#[allow(dead_code)]
pub struct Mp4Demuxer<R> {
    /// The underlying media source.
    source: R,

    /// Internal read buffer.
    buffer: Vec<u8>,

    /// Parsed `ftyp` box.
    ftyp: Option<FtypBox>,

    /// Parsed `moov` box.
    moov: Option<MoovBox>,

    /// Stream information extracted from tracks.
    streams: Vec<StreamInfo>,

    /// Per-track demuxing state.
    tracks: Vec<TrackState>,

    /// Current file position.
    position: u64,

    /// Start of `mdat` box.
    mdat_start: u64,

    /// Size of `mdat` box.
    mdat_size: u64,

    /// Whether headers have been parsed.
    header_parsed: bool,
}

/// Per-track demuxing state.
#[derive(Clone, Debug, Default)]
pub struct TrackState {
    /// Track ID from the container.
    pub track_id: u32,
    /// Index in the streams array.
    pub stream_index: usize,
    /// Current sample index (0-based).
    pub sample_index: u32,
    /// Total number of samples.
    pub sample_count: u32,
    /// Precomputed sample information for random access.
    pub samples: Vec<SampleInfo>,
}

/// Information about a single sample in a track.
#[derive(Clone, Debug)]
pub struct SampleInfo {
    /// Absolute offset in the file.
    pub offset: u64,
    /// Size in bytes.
    pub size: u32,
    /// Duration in timescale units.
    pub duration: u32,
    /// Composition time offset (PTS - DTS).
    pub cts_offset: i32,
    /// Whether this sample is a sync point (keyframe).
    pub is_sync: bool,
}

impl<R> Mp4Demuxer<R> {
    /// Creates a new MP4 demuxer with the given source.
    ///
    /// After creation, call [`probe`](Demuxer::probe) to parse headers
    /// and detect streams.
    #[must_use]
    pub fn new(source: R) -> Self {
        Self {
            source,
            buffer: Vec::with_capacity(65536),
            ftyp: None,
            moov: None,
            streams: Vec::new(),
            tracks: Vec::new(),
            position: 0,
            mdat_start: 0,
            mdat_size: 0,
            header_parsed: false,
        }
    }

    /// Returns a reference to the underlying source.
    #[must_use]
    pub const fn source(&self) -> &R {
        &self.source
    }

    /// Returns a mutable reference to the underlying source.
    pub fn source_mut(&mut self) -> &mut R {
        &mut self.source
    }

    /// Consumes the demuxer and returns the underlying source.
    #[must_use]
    #[allow(dead_code)]
    pub fn into_source(self) -> R {
        self.source
    }

    /// Returns the parsed `ftyp` box, if available.
    #[must_use]
    pub const fn ftyp(&self) -> Option<&FtypBox> {
        self.ftyp.as_ref()
    }

    /// Returns the parsed `moov` box, if available.
    #[must_use]
    pub const fn moov(&self) -> Option<&MoovBox> {
        self.moov.as_ref()
    }

    /// Returns a slice of parsed `TrakBox` entries from the `moov` box.
    ///
    /// Returns an empty slice if headers have not been parsed yet.
    #[must_use]
    pub fn traks(&self) -> &[TrakBox] {
        self.moov.as_ref().map_or(&[], |m| m.traks.as_slice())
    }
}

impl<R: MediaSource> Mp4Demuxer<R> {
    /// Reads exactly `n` bytes from source into a new `Vec<u8>`.
    async fn read_n(&mut self, n: usize) -> OxiResult<Vec<u8>> {
        let mut buf = vec![0u8; n];
        let mut filled = 0usize;
        while filled < n {
            let count = self.source.read(&mut buf[filled..]).await?;
            if count == 0 {
                return Err(OxiError::UnexpectedEof);
            }
            filled += count;
        }
        self.position += n as u64;
        Ok(buf)
    }

    /// Seeks source to `pos` (absolute) and updates internal position tracker.
    async fn seek_to(&mut self, pos: u64) -> OxiResult<()> {
        self.source.seek(SeekFrom::Start(pos)).await?;
        self.position = pos;
        Ok(())
    }

    /// Parses the container headers by scanning top-level boxes.
    ///
    /// This performs a two-pass strategy:
    /// 1. Stream through boxes looking for `ftyp` and `moov` (and noting `mdat` position).
    /// 2. Parse `ftyp` and `moov` data in-memory.
    ///
    /// Because `moov` may appear before or after `mdat`, we buffer any `moov` data
    /// encountered before `mdat`, or seek back after finding `mdat`.
    ///
    /// # Errors
    ///
    /// Returns an error if the file is malformed or a patent-encumbered codec is detected.
    async fn parse_headers(&mut self) -> OxiResult<()> {
        // Seek to beginning
        self.seek_to(0).await?;

        let mut ftyp_data: Option<Vec<u8>> = None;
        let mut moov_data: Option<Vec<u8>> = None;

        // Scan top-level boxes
        loop {
            // Read 8-byte box header
            let mut header_buf = [0u8; 8];
            let mut filled = 0usize;
            loop {
                let n = self.source.read(&mut header_buf[filled..]).await?;
                if n == 0 {
                    // EOF - stop scanning
                    break;
                }
                filled += n;
                if filled == 8 {
                    break;
                }
            }
            if filled < 8 {
                // End of file or truncated - stop scanning
                break;
            }

            let size32 =
                u32::from_be_bytes([header_buf[0], header_buf[1], header_buf[2], header_buf[3]]);
            let box_type =
                BoxType::new([header_buf[4], header_buf[5], header_buf[6], header_buf[7]]);

            // Handle extended size (size32 == 1)
            let (box_size, header_size): (u64, u64) = if size32 == 1 {
                // Read 8 more bytes for extended size
                let mut ext = [0u8; 8];
                let mut ef = 0usize;
                while ef < 8 {
                    let n = self.source.read(&mut ext[ef..]).await?;
                    if n == 0 {
                        return Err(OxiError::UnexpectedEof);
                    }
                    ef += n;
                }
                let ext_size = u64::from_be_bytes(ext);
                self.position += 16;
                (ext_size, 16)
            } else if size32 == 0 {
                // Box extends to EOF
                self.position += 8;
                (0, 8)
            } else {
                self.position += 8;
                (u64::from(size32), 8)
            };

            let content_size = if box_size == 0 {
                // Unknown - we'll read to EOF for moov if needed, but for simplicity skip
                0u64
            } else {
                box_size.saturating_sub(header_size)
            };

            match box_type {
                BoxType::FTYP => {
                    if content_size > 0 && content_size <= 4096 {
                        let data = self.read_n(content_size as usize).await?;
                        ftyp_data = Some(data);
                    } else if content_size > 0 {
                        // Skip oversized ftyp
                        self.source
                            .seek(SeekFrom::Current(content_size as i64))
                            .await?;
                        self.position += content_size;
                    }
                }
                BoxType::MOOV => {
                    if content_size > 0 {
                        let data = self.read_n(content_size as usize).await?;
                        moov_data = Some(data);
                    }
                }
                BoxType::MDAT => {
                    // Record mdat position
                    self.mdat_start = self.position;
                    self.mdat_size = content_size;
                    if content_size > 0 {
                        // Skip mdat content — packets are read lazily via sample table offsets
                        self.source
                            .seek(SeekFrom::Current(content_size as i64))
                            .await?;
                        self.position += content_size;
                    }
                }
                BoxType::FREE | BoxType::SKIP | BoxType::UDTA | BoxType::META => {
                    // Skip ignorable boxes
                    if content_size > 0 {
                        self.source
                            .seek(SeekFrom::Current(content_size as i64))
                            .await?;
                        self.position += content_size;
                    }
                }
                _ => {
                    // Skip unknown top-level boxes
                    if content_size > 0 {
                        self.source
                            .seek(SeekFrom::Current(content_size as i64))
                            .await?;
                        self.position += content_size;
                    } else if box_size == 0 {
                        // Zero-size means to EOF - stop scanning
                        break;
                    }
                }
            }

            // Stop if we have both ftyp and moov (no need to scan further)
            if ftyp_data.is_some() && moov_data.is_some() {
                break;
            }
        }

        // Parse ftyp
        if let Some(ref data) = ftyp_data {
            let ftyp = FtypBox::parse(data)?;
            if !ftyp.is_mp4() {
                return Err(OxiError::InvalidData(
                    "Not a valid MP4/ISOBMFF file (ftyp brand not recognized)".into(),
                ));
            }
            self.ftyp = Some(ftyp);
        }

        // Parse moov and build streams
        if let Some(ref data) = moov_data {
            let moov = MoovBox::parse(data)?;
            self.build_streams_and_tracks(&moov)?;
            self.moov = Some(moov);
        }

        self.header_parsed = true;
        Ok(())
    }

    /// Builds stream info and track state from a parsed `MoovBox`.
    ///
    /// # Errors
    ///
    /// Returns an error if codec mapping fails (patent violation or unsupported).
    fn build_streams_and_tracks(&mut self, moov: &MoovBox) -> OxiResult<()> {
        let movie_timescale = moov.mvhd.as_ref().map_or(1000, |mvhd| mvhd.timescale);

        let mut stream_index = 0usize;
        for trak in &moov.traks {
            // Skip unsupported handler types (text, metadata, etc.)
            if !matches!(trak.handler_type.as_str(), "vide" | "soun") {
                continue;
            }

            // Attempt codec mapping — skip tracks with unsupported/patent-encumbered codecs
            let stream_info = match build_stream_info(stream_index, trak, movie_timescale) {
                Ok(info) => info,
                Err(OxiError::PatentViolation(_)) => {
                    return Err(build_stream_info(stream_index, trak, movie_timescale).unwrap_err());
                }
                Err(_) => {
                    // Unknown codec: skip silently (don't add to streams)
                    continue;
                }
            };

            // Build sample table for this track
            let samples = build_sample_table(trak);
            #[allow(clippy::cast_possible_truncation)]
            let sample_count = samples.len() as u32;

            let track_id = trak.tkhd.as_ref().map_or(0, |tkhd| tkhd.track_id);

            let track_state = TrackState {
                track_id,
                stream_index,
                sample_index: 0,
                sample_count,
                samples,
            };

            self.streams.push(stream_info);
            self.tracks.push(track_state);

            stream_index += 1;
        }

        Ok(())
    }

    /// Reads the raw bytes for a single sample from the source.
    ///
    /// Seeks to the sample's absolute offset and reads `size` bytes.
    ///
    /// # Errors
    ///
    /// Returns an error on I/O failure or unexpected EOF.
    async fn read_sample_data(&mut self, offset: u64, size: u32) -> OxiResult<Bytes> {
        if size == 0 {
            return Ok(Bytes::new());
        }
        self.source.seek(SeekFrom::Start(offset)).await?;
        self.position = offset;
        let data = self.read_n(size as usize).await?;
        Ok(Bytes::from(data))
    }

    /// Selects the track with the lowest current DTS (decode timestamp) to emit next.
    ///
    /// Returns the index into `self.tracks`, or `None` if all tracks are exhausted.
    fn next_track_index(&self) -> Option<usize> {
        let mut best: Option<(usize, u64)> = None;

        for (i, track) in self.tracks.iter().enumerate() {
            if track.sample_index >= track.sample_count {
                continue;
            }
            let idx = track.sample_index as usize;
            // Compute cumulative DTS as sum of durations up to this sample
            let dts: u64 = track.samples[..idx]
                .iter()
                .map(|s| u64::from(s.duration))
                .sum();

            match best {
                Some((_, best_dts)) if dts < best_dts => {
                    best = Some((i, dts));
                }
                None => {
                    best = Some((i, dts));
                }
                _ => {}
            }
        }

        best.map(|(i, _)| i)
    }

    fn sample_dts(track: &TrackState, sample_index: usize) -> u64 {
        track.samples[..sample_index]
            .iter()
            .map(|s| u64::from(s.duration))
            .sum()
    }

    fn sample_pts(track: &TrackState, sample_index: usize) -> Option<i64> {
        let sample = track.samples.get(sample_index)?;
        let dts = Self::sample_dts(track, sample_index);
        let dts_i64 = i64::try_from(dts).ok()?;
        Some(dts_i64 + i64::from(sample.cts_offset))
    }

    fn sample_accurate_cursor_for_track(
        &self,
        track: &TrackState,
        target_pts: u64,
    ) -> OxiResult<DecodeSkipCursor> {
        if track.samples.is_empty() {
            return Err(OxiError::InvalidData("Track has no samples".into()));
        }

        let target_pts_i64 = i64::try_from(target_pts)
            .map_err(|_| OxiError::InvalidData("Target PTS is out of range".into()))?;

        let target_index = track
            .samples
            .iter()
            .enumerate()
            .find_map(|(index, _)| {
                Self::sample_pts(track, index)
                    .filter(|&pts| pts >= target_pts_i64)
                    .map(|_| index)
            })
            .unwrap_or(track.samples.len().saturating_sub(1));

        let keyframe_index = (0..=target_index)
            .rev()
            .find(|&index| track.samples[index].is_sync)
            .unwrap_or(0);

        let byte_offset = track.samples[keyframe_index].offset;
        let skip_samples = u32::try_from(target_index.saturating_sub(keyframe_index))
            .map_err(|_| OxiError::InvalidData("Sample skip count is out of range".into()))?;

        Ok(DecodeSkipCursor {
            byte_offset,
            sample_index: keyframe_index,
            skip_samples,
            target_pts: target_pts_i64,
        })
    }

    /// Plans a sample-accurate seek for the default video track, or the first track.
    ///
    /// Returns a cursor describing the keyframe offset and the number of samples
    /// that must be decoded and discarded to reach `target_pts`.
    ///
    /// # Errors
    ///
    /// Returns an error if headers have not been parsed or no decodable track exists.
    pub async fn seek_sample_accurate(&mut self, target_pts: u64) -> OxiResult<DecodeSkipCursor> {
        if !self.header_parsed {
            self.parse_headers().await?;
        }

        let track_index = self
            .streams
            .iter()
            .position(StreamInfo::is_video)
            .unwrap_or(0);
        let track = self
            .tracks
            .get(track_index)
            .ok_or_else(|| OxiError::InvalidData("No MP4 tracks available".into()))?;

        self.sample_accurate_cursor_for_track(track, target_pts)
    }
}

#[async_trait]
impl<R: MediaSource> Demuxer for Mp4Demuxer<R> {
    /// Parses the container headers and detects streams.
    ///
    /// After a successful probe:
    /// - `self.ftyp` contains the parsed file-type box.
    /// - `self.moov` contains the parsed movie box.
    /// - `self.streams` is populated with stream info.
    /// - `self.tracks` is populated with per-track sample tables.
    ///
    /// # Errors
    ///
    /// - `OxiError::InvalidData` if the file is not a valid MP4.
    /// - `OxiError::PatentViolation` if any track uses a patent-encumbered codec.
    async fn probe(&mut self) -> OxiResult<ProbeResult> {
        if !self.header_parsed {
            self.parse_headers().await?;
        }

        // Determine confidence based on what we managed to parse
        let confidence = if self.ftyp.is_some() && !self.streams.is_empty() {
            0.99
        } else if self.ftyp.is_some() || self.moov.is_some() {
            0.95
        } else {
            0.90
        };

        Ok(ProbeResult::new(ContainerFormat::Mp4, confidence))
    }

    /// Reads the next packet from the container in interleaved DTS order.
    ///
    /// Selects the track whose next sample has the lowest cumulative DTS,
    /// seeks to that sample's file offset, reads the data, and returns a packet.
    ///
    /// # Errors
    ///
    /// - `OxiError::Eof` when all tracks are fully consumed.
    /// - I/O errors from the underlying source.
    async fn read_packet(&mut self) -> OxiResult<Packet> {
        // Ensure headers are parsed
        if !self.header_parsed {
            self.parse_headers().await?;
        }

        // Select the track to emit next
        let track_idx = self.next_track_index().ok_or(OxiError::Eof)?;

        let sample_idx = self.tracks[track_idx].sample_index as usize;
        let sample = self.tracks[track_idx].samples[sample_idx].clone();
        let stream_index = self.tracks[track_idx].stream_index;

        // Read raw bytes
        let data = self.read_sample_data(sample.offset, sample.size).await?;

        // Advance sample index
        self.tracks[track_idx].sample_index += 1;

        // Build timestamps — DTS is the cumulative duration up to this sample
        let stream = &self.streams[stream_index];
        let timebase = stream.timebase;

        let dts: i64 = self.tracks[track_idx].samples[..sample_idx]
            .iter()
            .map(|s| i64::from(s.duration))
            .sum();

        let pts = dts + i64::from(sample.cts_offset);

        let mut timestamp = Timestamp::new(pts, timebase);
        timestamp.dts = Some(dts);
        timestamp.duration = Some(i64::from(sample.duration));

        // Determine flags
        let mut flags = PacketFlags::empty();
        if sample.is_sync {
            flags |= PacketFlags::KEYFRAME;
        }

        Ok(Packet::new(stream_index, data, timestamp, flags))
    }

    fn streams(&self) -> &[StreamInfo] {
        &self.streams
    }
}

/// Maps an MP4 codec tag to an `OxiMedia` codec identifier.
///
/// This function enforces the patent-free policy by returning
/// [`PatentViolation`](OxiError::PatentViolation) for known
/// patent-encumbered codecs.
///
/// # Arguments
///
/// * `handler` - Handler type from `hdlr` box ("vide", "soun", etc.)
/// * `codec_tag` - 4CC codec tag from sample description
///
/// # Errors
///
/// - Returns [`PatentViolation`](OxiError::PatentViolation) for H.264, H.265, AAC, etc.
/// - Returns [`Unsupported`](OxiError::Unsupported) for unknown codecs.
///
/// # Supported Mappings
///
/// | Handler | Tag      | Codec  |
/// |---------|----------|--------|
/// | `vide`  | `av01`   | AV1    |
/// | `vide`  | `vp09`   | VP9    |
/// | `soun`  | `Opus`   | Opus   |
/// | `soun`  | `fLaC`   | FLAC   |
///
/// # Patent-Encumbered (Rejected)
///
/// | Handler | Tag      | Codec     |
/// |---------|----------|-----------|
/// | `vide`  | `avc1`   | H.264     |
/// | `vide`  | `avc3`   | H.264     |
/// | `vide`  | `hvc1`   | H.265     |
/// | `vide`  | `hev1`   | H.265     |
/// | `soun`  | `mp4a`   | AAC       |
/// | `soun`  | `ac-3`   | AC-3      |
/// | `soun`  | `ec-3`   | E-AC-3    |
pub fn map_codec(handler: &str, codec_tag: u32) -> OxiResult<CodecId> {
    match (handler, codec_tag) {
        // =====================
        // Royalty-Free Video
        // =====================

        // AV1 - "av01"
        ("vide", 0x6176_3031) => Ok(CodecId::Av1),

        // VP9 - "vp09"
        ("vide", 0x7670_3039) => Ok(CodecId::Vp9),

        // VP8 - "vp08"
        ("vide", 0x7670_3038) => Ok(CodecId::Vp8),

        // =====================
        // Royalty-Free Audio
        // =====================

        // Opus - "Opus"
        ("soun", 0x4F70_7573) => Ok(CodecId::Opus),

        // FLAC - "fLaC"
        ("soun", 0x664C_6143) => Ok(CodecId::Flac),

        // Vorbis - "vorb" (rare in MP4)
        ("soun", 0x766F_7262) => Ok(CodecId::Vorbis),

        // =====================
        // Patent-Encumbered - REJECT
        // =====================

        // H.264/AVC - "avc1", "avc2", "avc3", "avc4"
        ("vide", 0x6176_6331..=0x6176_6334) => Err(OxiError::PatentViolation("H.264/AVC".into())),

        // H.265/HEVC - "hvc1", "hev1", "hvc2", "hev2"
        ("vide", 0x6876_6331 | 0x6865_7631 | 0x6876_6332 | 0x6865_7632) => {
            Err(OxiError::PatentViolation("H.265/HEVC".into()))
        }

        // H.266/VVC - "vvc1", "vvi1"
        ("vide", 0x7676_6331 | 0x7676_6931) => Err(OxiError::PatentViolation("H.266/VVC".into())),

        // AAC - "mp4a"
        ("soun", 0x6D70_3461) => Err(OxiError::PatentViolation("AAC".into())),

        // AC-3 - "ac-3"
        ("soun", 0x6163_2D33) => Err(OxiError::PatentViolation("AC-3".into())),

        // E-AC-3 - "ec-3"
        ("soun", 0x6563_2D33) => Err(OxiError::PatentViolation("E-AC-3".into())),

        // DTS - "dtsc", "dtsh", "dtsl", "dtse"
        ("soun", 0x6474_7363 | 0x6474_7368 | 0x6474_736C | 0x6474_7365) => {
            Err(OxiError::PatentViolation("DTS".into()))
        }

        // MPEG-1/2 Audio - "mp3 ", ".mp3"
        ("soun", 0x6D70_3320 | 0x2E6D_7033) => Err(OxiError::PatentViolation("MP3".into())),

        // =====================
        // Unknown
        // =====================
        _ => Err(OxiError::Unsupported(format!(
            "Unknown MP4 codec: handler={handler}, tag=0x{codec_tag:08X} ({})",
            tag_to_string(codec_tag)
        ))),
    }
}

/// Converts a 4CC codec tag to a readable string.
fn tag_to_string(tag: u32) -> String {
    let bytes = tag.to_be_bytes();
    String::from_utf8_lossy(&bytes).into_owned()
}

/// Builds a [`StreamInfo`] from a parsed track.
///
/// # Arguments
///
/// * `index` - Stream index
/// * `track` - Parsed track information
/// * `movie_timescale` - Timescale from the movie header
///
/// # Errors
///
/// Returns an error if the codec is unsupported or patent-encumbered.
#[allow(dead_code)]
fn build_stream_info(index: usize, track: &TrakBox, movie_timescale: u32) -> OxiResult<StreamInfo> {
    let codec = map_codec(&track.handler_type, track.codec_tag)?;

    // Use track timescale, falling back to movie timescale
    let timescale = if track.timescale > 0 {
        track.timescale
    } else {
        movie_timescale
    };

    let mut stream = StreamInfo::new(index, codec, Rational::new(1, i64::from(timescale)));

    // Set codec-specific parameters
    match track.handler_type.as_str() {
        "vide" => {
            if let (Some(w), Some(h)) = (track.width, track.height) {
                stream.codec_params = CodecParams::video(w, h);
            }
        }
        "soun" => {
            if let (Some(rate), Some(ch)) = (track.sample_rate, track.channels) {
                stream.codec_params = CodecParams::audio(rate, u8::try_from(ch).unwrap_or(2));
            }
        }
        _ => {}
    }

    // Set extradata if available
    if let Some(ref extra) = track.extradata {
        stream.codec_params.extradata = Some(Bytes::copy_from_slice(extra));
    }

    // Calculate duration
    if let Some(tkhd) = &track.tkhd {
        if movie_timescale > 0 {
            #[allow(clippy::cast_possible_wrap)]
            let duration_in_stream_tb =
                (tkhd.duration as i64 * i64::from(timescale)) / i64::from(movie_timescale);
            stream.duration = Some(duration_in_stream_tb);
        }
    }

    stream.metadata = Metadata::new();

    Ok(stream)
}

/// Builds a sample table from track box information.
///
/// This function computes the offset, size, duration, and sync status
/// for each sample in the track.
#[allow(dead_code)]
fn build_sample_table(track: &TrakBox) -> Vec<SampleInfo> {
    let mut samples = Vec::new();

    // Calculate total sample count
    let sample_count = if track.sample_sizes.is_empty() {
        // All samples have the same size - count from stts
        track
            .stts_entries
            .iter()
            .map(|e| e.sample_count as usize)
            .sum()
    } else {
        track.sample_sizes.len()
    };

    if sample_count == 0 {
        return samples;
    }

    // Build sync sample lookup (if not all samples are sync)
    let sync_set: Option<std::collections::HashSet<u32>> = track
        .sync_samples
        .as_ref()
        .map(|ss| ss.iter().copied().collect());

    // Build sample-to-chunk lookup
    let mut chunk_sample_map: Vec<(u32, u32, u32)> = Vec::new(); // (first_sample, chunk, samples_per_chunk)
    let mut sample_num = 1u32;
    for i in 0..track.stsc_entries.len() {
        let entry = &track.stsc_entries[i];
        let next_first_chunk = if i + 1 < track.stsc_entries.len() {
            track.stsc_entries[i + 1].first_chunk
        } else {
            #[allow(clippy::cast_possible_truncation)]
            let chunk_count = track.chunk_offsets.len() as u32 + 1;
            chunk_count
        };

        for chunk in entry.first_chunk..next_first_chunk {
            chunk_sample_map.push((sample_num, chunk, entry.samples_per_chunk));
            sample_num += entry.samples_per_chunk;
        }
    }

    // Build duration lookup from stts
    let mut durations: Vec<u32> = Vec::with_capacity(sample_count);
    for entry in &track.stts_entries {
        for _ in 0..entry.sample_count {
            durations.push(entry.sample_delta);
        }
    }

    // Build CTS offset lookup from ctts
    let mut cts_offsets: Vec<i32> = Vec::with_capacity(sample_count);
    for entry in &track.ctts_entries {
        for _ in 0..entry.sample_count {
            cts_offsets.push(entry.sample_offset);
        }
    }

    // Now build the sample table
    let mut current_chunk_idx = 0usize;
    let mut sample_in_chunk = 0u32;

    for sample_idx in 0..sample_count {
        #[allow(clippy::cast_possible_truncation)]
        let sample_num_1based = sample_idx as u32 + 1;

        // Get chunk offset
        let chunk_offset = if current_chunk_idx < track.chunk_offsets.len() {
            track.chunk_offsets[current_chunk_idx]
        } else {
            0
        };

        // Calculate offset within chunk
        let samples_per_chunk = if current_chunk_idx < chunk_sample_map.len() {
            chunk_sample_map[current_chunk_idx].2
        } else {
            1
        };

        let mut offset_in_chunk = 0u64;
        let first_sample_in_chunk = sample_idx - sample_in_chunk as usize;
        for i in first_sample_in_chunk..sample_idx {
            let size = if track.default_sample_size > 0 {
                track.default_sample_size
            } else if i < track.sample_sizes.len() {
                track.sample_sizes[i]
            } else {
                0
            };
            offset_in_chunk += u64::from(size);
        }

        let offset = chunk_offset + offset_in_chunk;

        // Get sample size
        let size = if track.default_sample_size > 0 {
            track.default_sample_size
        } else if sample_idx < track.sample_sizes.len() {
            track.sample_sizes[sample_idx]
        } else {
            0
        };

        // Get duration
        let duration = durations.get(sample_idx).copied().unwrap_or(0);

        // Get CTS offset
        let cts_offset = cts_offsets.get(sample_idx).copied().unwrap_or(0);

        // Check if sync
        let is_sync = sync_set
            .as_ref()
            .map_or(true, |set| set.contains(&sample_num_1based));

        samples.push(SampleInfo {
            offset,
            size,
            duration,
            cts_offset,
            is_sync,
        });

        // Advance to next chunk if needed
        sample_in_chunk += 1;
        if sample_in_chunk >= samples_per_chunk {
            sample_in_chunk = 0;
            current_chunk_idx += 1;
        }
    }

    samples
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_io::MemorySource;

    /// Builds a minimal valid MP4 binary in memory with one AV1 video track.
    ///
    /// Structure:
    ///   ftyp (isom)
    ///   moov
    ///     mvhd (v0, timescale=1000, duration=100)
    ///     trak
    ///       tkhd (v0, track_id=1, duration=100, 320x240)
    ///       mdia
    ///         mdhd (v0, timescale=1000)
    ///         hdlr (vide)
    ///         minf
    ///           stbl
    ///             stsd (1 entry: av01, 320x240, no extradata child)
    ///             stts (1 entry: 1 sample, delta=100)
    ///             stsc (1 entry: first_chunk=1, samples_per_chunk=1)
    ///             stsz (uniform_size=4, count=1)
    ///             stco (1 offset pointing to mdat content)
    ///   mdat (4 bytes: 0xDE 0xAD 0xBE 0xEF)
    fn build_test_mp4() -> Vec<u8> {
        // ── helpers ──────────────────────────────────────────────────────
        fn u32be(v: u32) -> [u8; 4] {
            v.to_be_bytes()
        }
        fn u16be(v: u16) -> [u8; 2] {
            v.to_be_bytes()
        }
        fn box_with_content(tag: &[u8; 4], content: &[u8]) -> Vec<u8> {
            let size = 8u32 + content.len() as u32;
            let mut b = Vec::with_capacity(size as usize);
            b.extend_from_slice(&u32be(size));
            b.extend_from_slice(tag);
            b.extend_from_slice(content);
            b
        }

        // ── leaf boxes ────────────────────────────────────────────────────

        // mdhd v0: version(1) + flags(3) + ctime(4) + mtime(4) + timescale(4)=1000
        //          + duration(4)=100 + language(2) + pre_defined(2)
        let mut mdhd = Vec::new();
        mdhd.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // version + flags
        mdhd.extend_from_slice(&u32be(0)); // creation_time
        mdhd.extend_from_slice(&u32be(0)); // modification_time
        mdhd.extend_from_slice(&u32be(1000)); // timescale
        mdhd.extend_from_slice(&u32be(100)); // duration
        mdhd.extend_from_slice(&[0x55, 0xC4]); // language (und)
        mdhd.extend_from_slice(&[0x00, 0x00]); // pre_defined
        let mdhd_box = box_with_content(b"mdhd", &mdhd);

        // hdlr: version(1)+flags(3)+pre_defined(4)+handler_type(4)+"vide"
        //       + reserved(12) + name_null(1)
        let mut hdlr = Vec::new();
        hdlr.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // version + flags
        hdlr.extend_from_slice(&u32be(0)); // pre_defined
        hdlr.extend_from_slice(b"vide"); // handler_type
        hdlr.extend_from_slice(&[0u8; 12]); // reserved
        hdlr.push(0x00); // name (null-terminated)
        let hdlr_box = box_with_content(b"hdlr", &hdlr);

        // stsd entry for av01 (visual sample entry):
        //   4 size, 4 codec-tag ("av01"),
        //   6 reserved, 2 data_ref_idx=1,
        //   2 pre_defined, 2 reserved, 12 pre_defined,
        //   2 width=320, 2 height=240,
        //   4 horiz_res, 4 vert_res, 4 reserved, 2 frame_count=1,
        //   32 compressorname, 2 depth=0x0018, 2 pre_defined=-1
        // (no child config box — extradata will be None)
        let mut entry_body = Vec::new();
        entry_body.extend_from_slice(&[0u8; 6]); // reserved
        entry_body.extend_from_slice(&u16be(1)); // data_reference_index
        entry_body.extend_from_slice(&[0u8; 2]); // pre_defined
        entry_body.extend_from_slice(&[0u8; 2]); // reserved
        entry_body.extend_from_slice(&[0u8; 12]); // pre_defined
        entry_body.extend_from_slice(&u16be(320)); // width
        entry_body.extend_from_slice(&u16be(240)); // height
        entry_body.extend_from_slice(&u32be(0x00480000)); // horiz_res 72dpi
        entry_body.extend_from_slice(&u32be(0x00480000)); // vert_res
        entry_body.extend_from_slice(&u32be(0)); // reserved
        entry_body.extend_from_slice(&u16be(1)); // frame_count
        entry_body.extend_from_slice(&[0u8; 32]); // compressorname
        entry_body.extend_from_slice(&u16be(0x0018)); // depth
        entry_body.extend_from_slice(&[0xFF, 0xFF]); // pre_defined = -1

        // Build STSD entry: 4-byte size + "av01" + body
        let entry_size = 8u32 + entry_body.len() as u32;
        let mut stsd_content = Vec::new();
        stsd_content.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // version + flags
        stsd_content.extend_from_slice(&u32be(1)); // entry_count = 1
        stsd_content.extend_from_slice(&u32be(entry_size));
        stsd_content.extend_from_slice(b"av01");
        stsd_content.extend_from_slice(&entry_body);
        let stsd_box = box_with_content(b"stsd", &stsd_content);

        // stts: 1 entry, 1 sample, delta=100
        let mut stts = Vec::new();
        stts.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // version + flags
        stts.extend_from_slice(&u32be(1)); // entry_count
        stts.extend_from_slice(&u32be(1)); // sample_count
        stts.extend_from_slice(&u32be(100)); // sample_delta
        let stts_box = box_with_content(b"stts", &stts);

        // stsc: 1 entry, first_chunk=1, samples_per_chunk=1, desc_idx=1
        let mut stsc = Vec::new();
        stsc.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // version + flags
        stsc.extend_from_slice(&u32be(1)); // entry_count
        stsc.extend_from_slice(&u32be(1)); // first_chunk
        stsc.extend_from_slice(&u32be(1)); // samples_per_chunk
        stsc.extend_from_slice(&u32be(1)); // sample_description_index
        let stsc_box = box_with_content(b"stsc", &stsc);

        // stsz: default_sample_size=4, count=1 (all samples 4 bytes)
        let mut stsz = Vec::new();
        stsz.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // version + flags
        stsz.extend_from_slice(&u32be(4)); // default sample size
        stsz.extend_from_slice(&u32be(1)); // sample_count
        let stsz_box = box_with_content(b"stsz", &stsz);

        // We'll compute the mdat offset after assembling everything.
        // For now use a placeholder; we'll patch it later.
        let stco_placeholder_offset = 0u32;
        let mut stco = Vec::new();
        stco.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // version + flags
        stco.extend_from_slice(&u32be(1)); // entry_count
        stco.extend_from_slice(&u32be(stco_placeholder_offset)); // chunk_offset placeholder
        let stco_box = box_with_content(b"stco", &stco);

        // stbl container
        let mut stbl_content = Vec::new();
        stbl_content.extend_from_slice(&stsd_box);
        stbl_content.extend_from_slice(&stts_box);
        stbl_content.extend_from_slice(&stsc_box);
        stbl_content.extend_from_slice(&stsz_box);
        stbl_content.extend_from_slice(&stco_box);
        let stbl_box = box_with_content(b"stbl", &stbl_content);

        // minf (video media information header omitted for brevity)
        let minf_box = box_with_content(b"minf", &stbl_box);

        // mdia
        let mut mdia_content = Vec::new();
        mdia_content.extend_from_slice(&mdhd_box);
        mdia_content.extend_from_slice(&hdlr_box);
        mdia_content.extend_from_slice(&minf_box);
        let mdia_box = box_with_content(b"mdia", &mdia_content);

        // tkhd v0: version(1)+flags(3)+ctime(4)+mtime(4)+track_id(4)=1
        //          +reserved(4)+duration(4)=100+reserved(8)
        //          +layer(2)+alt_group(2)+volume(2)+reserved(2)+matrix(36)
        //          +width(4)=320<<16 +height(4)=240<<16
        let mut tkhd = Vec::new();
        tkhd.extend_from_slice(&[0x00, 0x00, 0x00, 0x03]); // version=0, flags=enabled|in_movie
        tkhd.extend_from_slice(&u32be(0)); // creation_time
        tkhd.extend_from_slice(&u32be(0)); // modification_time
        tkhd.extend_from_slice(&u32be(1)); // track_id
        tkhd.extend_from_slice(&u32be(0)); // reserved
        tkhd.extend_from_slice(&u32be(100)); // duration
        tkhd.extend_from_slice(&[0u8; 8]); // reserved
        tkhd.extend_from_slice(&[0u8; 4]); // layer + alternate_group
        tkhd.extend_from_slice(&[0u8; 4]); // volume + reserved
                                           // identity matrix
        tkhd.extend_from_slice(&u32be(0x00010000)); // a
        tkhd.extend_from_slice(&u32be(0));
        tkhd.extend_from_slice(&u32be(0));
        tkhd.extend_from_slice(&u32be(0));
        tkhd.extend_from_slice(&u32be(0x00010000)); // d
        tkhd.extend_from_slice(&u32be(0));
        tkhd.extend_from_slice(&u32be(0));
        tkhd.extend_from_slice(&u32be(0));
        tkhd.extend_from_slice(&u32be(0x40000000)); // w
        tkhd.extend_from_slice(&u32be(320u32 << 16)); // display width 320.0
        tkhd.extend_from_slice(&u32be(240u32 << 16)); // display height 240.0
        let tkhd_box = box_with_content(b"tkhd", &tkhd);

        // trak
        let mut trak_content = Vec::new();
        trak_content.extend_from_slice(&tkhd_box);
        trak_content.extend_from_slice(&mdia_box);
        let trak_box = box_with_content(b"trak", &trak_content);

        // mvhd v0: version(1)+flags(3)+ctime(4)+mtime(4)+timescale(4)=1000
        //          +duration(4)=100 +rate(4)=0x00010000 +volume(2)=0x0100
        //          +reserved(2)+reserved(8)+matrix(36)+pre_defined(24)
        //          +next_track_id(4)=2
        let mut mvhd = Vec::new();
        mvhd.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // version + flags
        mvhd.extend_from_slice(&u32be(0)); // creation_time
        mvhd.extend_from_slice(&u32be(0)); // modification_time
        mvhd.extend_from_slice(&u32be(1000)); // timescale
        mvhd.extend_from_slice(&u32be(100)); // duration
        mvhd.extend_from_slice(&u32be(0x00010000)); // rate = 1.0
        mvhd.extend_from_slice(&u16be(0x0100)); // volume = 1.0
        mvhd.extend_from_slice(&[0u8; 2]); // reserved
        mvhd.extend_from_slice(&[0u8; 8]); // reserved
                                           // identity matrix (36 bytes)
        mvhd.extend_from_slice(&u32be(0x00010000));
        mvhd.extend_from_slice(&u32be(0));
        mvhd.extend_from_slice(&u32be(0));
        mvhd.extend_from_slice(&u32be(0));
        mvhd.extend_from_slice(&u32be(0x00010000));
        mvhd.extend_from_slice(&u32be(0));
        mvhd.extend_from_slice(&u32be(0));
        mvhd.extend_from_slice(&u32be(0));
        mvhd.extend_from_slice(&u32be(0x40000000));
        // pre_defined (24 bytes)
        mvhd.extend_from_slice(&[0u8; 24]);
        mvhd.extend_from_slice(&u32be(2)); // next_track_id
        let mvhd_box = box_with_content(b"mvhd", &mvhd);

        // ftyp: major_brand=isom, minor_version=0, compatible=isom
        let mut ftyp_content = Vec::new();
        ftyp_content.extend_from_slice(b"isom"); // major_brand
        ftyp_content.extend_from_slice(&u32be(0)); // minor_version
        ftyp_content.extend_from_slice(b"isom"); // compatible brand
        let ftyp_box = box_with_content(b"ftyp", &ftyp_content);

        // moov
        let mut moov_content = Vec::new();
        moov_content.extend_from_slice(&mvhd_box);
        moov_content.extend_from_slice(&trak_box);
        let moov_box = box_with_content(b"moov", &moov_content);

        // mdat payload
        let mdat_payload = &[0xDE_u8, 0xAD, 0xBE, 0xEF];
        let mdat_box = box_with_content(b"mdat", mdat_payload);

        // Assemble: ftyp + moov + mdat
        let mut file = Vec::new();
        file.extend_from_slice(&ftyp_box);
        file.extend_from_slice(&moov_box);
        // mdat starts after ftyp_box + moov_box
        let mdat_offset = file.len() as u32 + 8; // +8 for mdat box header
        file.extend_from_slice(&mdat_box);

        // Patch stco offset — find it in the assembled bytes and update
        // The stco chunk_offset is the 4-byte field after:
        //   stco box header (8) + version+flags (4) + entry_count (4) = 16 bytes into stco_box
        // We need to find stco in the assembled moov region.
        // Strategy: search for "stco" magic in our file bytes and patch the offset there.
        for i in 0..file.len().saturating_sub(16) {
            if &file[i..i + 4] == b"stco" {
                // i+4 = version+flags, i+8 = entry_count, i+12 = first chunk offset
                let offset_pos = i + 12;
                if offset_pos + 4 <= file.len() {
                    let bytes = mdat_offset.to_be_bytes();
                    file[offset_pos..offset_pos + 4].copy_from_slice(&bytes);
                }
                break;
            }
        }

        file
    }

    #[test]
    fn test_mp4_demuxer_new() {
        let source = MemorySource::new(bytes::Bytes::new());
        let demuxer = Mp4Demuxer::new(source);
        assert!(!demuxer.header_parsed);
        assert!(demuxer.streams().is_empty());
        assert!(demuxer.ftyp().is_none());
        assert!(demuxer.moov().is_none());
    }

    #[tokio::test]
    async fn test_mp4_demuxer_probe() {
        let data = build_test_mp4();
        let source = MemorySource::from_vec(data);
        let mut demuxer = Mp4Demuxer::new(source);
        let result = demuxer.probe().await.expect("probe should succeed");

        assert_eq!(result.format, ContainerFormat::Mp4);
        assert!(result.confidence > 0.8);
        assert!(demuxer.header_parsed);
        assert!(demuxer.ftyp().is_some());
        assert!(demuxer.moov().is_some());
    }

    #[tokio::test]
    async fn test_mp4_demuxer_streams() {
        let data = build_test_mp4();
        let source = MemorySource::from_vec(data);
        let mut demuxer = Mp4Demuxer::new(source);
        demuxer.probe().await.expect("probe should succeed");

        let streams = demuxer.streams();
        assert_eq!(streams.len(), 1);
        assert_eq!(streams[0].codec, CodecId::Av1);
        assert!(streams[0].is_video());
        assert_eq!(streams[0].codec_params.width, Some(320));
        assert_eq!(streams[0].codec_params.height, Some(240));
    }

    #[tokio::test]
    async fn test_mp4_demuxer_read_packet() {
        let data = build_test_mp4();
        let source = MemorySource::from_vec(data);
        let mut demuxer = Mp4Demuxer::new(source);
        demuxer.probe().await.expect("probe should succeed");

        let packet = demuxer.read_packet().await.expect("should read one packet");
        assert_eq!(packet.stream_index, 0);
        assert_eq!(packet.size(), 4);
        assert_eq!(&packet.data[..], &[0xDE, 0xAD, 0xBE, 0xEF]);
        // First sample should be a sync point (no stss = all sync)
        assert!(packet.is_keyframe());
    }

    #[tokio::test]
    async fn test_mp4_demuxer_eof() {
        let data = build_test_mp4();
        let source = MemorySource::from_vec(data);
        let mut demuxer = Mp4Demuxer::new(source);
        demuxer.probe().await.expect("probe should succeed");

        // Read the single packet
        let _ = demuxer.read_packet().await.expect("first packet");

        // Next read must return EOF
        let result = demuxer.read_packet().await;
        assert!(matches!(result, Err(OxiError::Eof)));
    }

    #[test]
    fn test_map_codec_av1() {
        // "av01" = 0x61763031
        let codec = map_codec("vide", 0x6176_3031).expect("operation should succeed");
        assert_eq!(codec, CodecId::Av1);
    }

    #[test]
    fn test_map_codec_vp9() {
        // "vp09" = 0x76703039
        let codec = map_codec("vide", 0x7670_3039).expect("operation should succeed");
        assert_eq!(codec, CodecId::Vp9);
    }

    #[test]
    fn test_map_codec_opus() {
        // "Opus" = 0x4F707573
        let codec = map_codec("soun", 0x4F70_7573).expect("operation should succeed");
        assert_eq!(codec, CodecId::Opus);
    }

    #[test]
    fn test_map_codec_flac() {
        // "fLaC" = 0x664C6143
        let codec = map_codec("soun", 0x664C_6143).expect("operation should succeed");
        assert_eq!(codec, CodecId::Flac);
    }

    #[test]
    fn test_map_codec_h264_rejected() {
        // "avc1" = 0x61766331
        let result = map_codec("vide", 0x6176_6331);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.is_patent_violation());
        assert!(format!("{err}").contains("H.264"));
    }

    #[test]
    fn test_map_codec_h265_rejected() {
        // "hvc1" = 0x68766331
        let result = map_codec("vide", 0x6876_6331);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.is_patent_violation());
        assert!(format!("{err}").contains("H.265"));
    }

    #[test]
    fn test_map_codec_aac_rejected() {
        // "mp4a" = 0x6D703461
        let result = map_codec("soun", 0x6D70_3461);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.is_patent_violation());
        assert!(format!("{err}").contains("AAC"));
    }

    #[test]
    fn test_map_codec_ac3_rejected() {
        // "ac-3" = 0x61632D33
        let result = map_codec("soun", 0x6163_2D33);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.is_patent_violation());
        assert!(format!("{err}").contains("AC-3"));
    }

    #[test]
    fn test_map_codec_eac3_rejected() {
        // "ec-3" = 0x65632D33
        let result = map_codec("soun", 0x6563_2D33);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.is_patent_violation());
        assert!(format!("{err}").contains("E-AC-3"));
    }

    #[test]
    fn test_map_codec_dts_rejected() {
        // "dtsc" = 0x64747363
        let result = map_codec("soun", 0x6474_7363);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.is_patent_violation());
        assert!(format!("{err}").contains("DTS"));
    }

    #[test]
    fn test_map_codec_unknown() {
        // Unknown codec
        let result = map_codec("vide", 0x1234_5678);
        assert!(result.is_err());
        match result.unwrap_err() {
            OxiError::Unsupported(msg) => {
                assert!(msg.contains("Unknown MP4 codec"));
            }
            other => panic!("Expected Unsupported error, got: {other:?}"),
        }
    }

    #[test]
    fn test_tag_to_string() {
        assert_eq!(tag_to_string(0x6176_3031), "av01");
        assert_eq!(tag_to_string(0x6D70_3461), "mp4a");
    }

    #[test]
    fn test_build_stream_info_av1() {
        let mut track = TrakBox::default();
        track.handler_type = "vide".into();
        track.codec_tag = 0x6176_3031; // av01
        track.timescale = 24000;
        track.width = Some(1920);
        track.height = Some(1080);
        track.tkhd = Some(TkhdBox {
            track_id: 1,
            duration: 240_000,
            width: 1920.0,
            height: 1080.0,
        });

        let stream = build_stream_info(0, &track, 1000).expect("operation should succeed");
        assert_eq!(stream.codec, CodecId::Av1);
        assert_eq!(stream.codec_params.width, Some(1920));
        assert_eq!(stream.codec_params.height, Some(1080));
        assert!(stream.is_video());
    }

    #[test]
    fn test_build_stream_info_opus() {
        let mut track = TrakBox::default();
        track.handler_type = "soun".into();
        track.codec_tag = 0x4F70_7573; // Opus
        track.timescale = 48000;
        track.sample_rate = Some(48000);
        track.channels = Some(2);

        let stream = build_stream_info(0, &track, 1000).expect("operation should succeed");
        assert_eq!(stream.codec, CodecId::Opus);
        assert_eq!(stream.codec_params.sample_rate, Some(48000));
        assert_eq!(stream.codec_params.channels, Some(2));
        assert!(stream.is_audio());
    }

    #[test]
    fn test_build_stream_info_rejected() {
        let mut track = TrakBox::default();
        track.handler_type = "vide".into();
        track.codec_tag = 0x6176_6331; // avc1 (H.264)

        let result = build_stream_info(0, &track, 1000);
        assert!(result.is_err());
        assert!(result.unwrap_err().is_patent_violation());
    }

    #[test]
    fn test_build_sample_table_basic() {
        let mut track = TrakBox::default();

        // 10 samples, each 1000 bytes
        track.default_sample_size = 1000;
        track.stts_entries = vec![SttsEntry {
            sample_count: 10,
            sample_delta: 100,
        }];

        // 2 chunks, 5 samples each
        track.stsc_entries = vec![StscEntry {
            first_chunk: 1,
            samples_per_chunk: 5,
            sample_description_index: 1,
        }];

        // Chunk offsets
        track.chunk_offsets = vec![0, 5000];

        // All samples are sync
        track.sync_samples = None;

        let samples = build_sample_table(&track);
        assert_eq!(samples.len(), 10);

        // First sample
        assert_eq!(samples[0].offset, 0);
        assert_eq!(samples[0].size, 1000);
        assert_eq!(samples[0].duration, 100);
        assert!(samples[0].is_sync);

        // Fifth sample (last in first chunk)
        assert_eq!(samples[4].offset, 4000);

        // Sixth sample (first in second chunk)
        assert_eq!(samples[5].offset, 5000);
    }
}
