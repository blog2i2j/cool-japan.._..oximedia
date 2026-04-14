//! Common Media Application Format (CMAF) output support.
//!
//! CMAF (ISO 23000-19) is a constrained form of fragmented MP4 used as the
//! common media format for both HLS (fMP4 segments) and MPEG-DASH.  This
//! module provides a minimal, pure-Rust implementation of the box writing
//! routines required to produce valid CMAF init and media segments.
//!
//! # Box layout
//!
//! Each box is encoded as `[size: u32 big-endian][fourcc: 4 bytes][payload]`.
//! A zero-payload box has size = 8.
//!
//! ## Init segment
//! ```text
//! ftyp (brand=cmf2, compatible=[cmf2, iso6, mp41])
//! moov (minimal header — track info is provider-supplied downstream)
//! ```
//!
//! ## Media segment
//! ```text
//! moof (movie fragment — sequence number + track run)
//! mdat (media data)
//! ```

/// Configuration for a CMAF track.
#[derive(Debug, Clone)]
pub struct CmafConfig {
    /// ISO Base Media File Format track ID (typically 1).
    pub track_id: u32,
    /// Number of time units per second (e.g. 90000 for video, 44100 for audio).
    pub timescale: u32,
    /// Nominal segment duration in milliseconds.
    pub segment_duration_ms: u32,
}

impl CmafConfig {
    /// Creates a video CMAF config with a 90 kHz timescale and 2 s segments.
    #[must_use]
    pub fn video_default() -> Self {
        Self {
            track_id: 1,
            timescale: 90_000,
            segment_duration_ms: 2_000,
        }
    }

    /// Creates an audio CMAF config with a 44.1 kHz timescale and 2 s segments.
    #[must_use]
    pub fn audio_default() -> Self {
        Self {
            track_id: 1,
            timescale: 44_100,
            segment_duration_ms: 2_000,
        }
    }
}

/// A single CMAF media segment.
#[derive(Debug, Clone)]
pub struct CmafSegment {
    /// 1-based sequence number (incremented automatically by [`CmafTrack::push_segment`]).
    pub sequence: u64,
    /// Decode start time of this segment in timescale units.
    pub start_time: u64,
    /// Duration of this segment in timescale units.
    pub duration: u32,
    /// Raw encoded media data (elementary stream payload for `mdat`).
    pub data: Vec<u8>,
}

/// A CMAF track comprising an ordered list of media segments.
#[derive(Debug, Clone)]
pub struct CmafTrack {
    /// Track configuration.
    pub config: CmafConfig,
    /// Ordered list of appended segments.
    pub segments: Vec<CmafSegment>,
    /// Running decode time cursor in timescale units.
    next_start_time: u64,
    /// Next sequence number to assign.
    next_sequence: u64,
}

impl CmafTrack {
    /// Creates a new empty [`CmafTrack`].
    #[must_use]
    pub fn new(config: CmafConfig) -> Self {
        Self {
            config,
            segments: Vec::new(),
            next_start_time: 0,
            next_sequence: 1,
        }
    }

    /// Appends a new segment with the supplied raw media data.
    ///
    /// `duration_samples` is the segment duration expressed in timescale units.
    /// The sequence number is assigned automatically (starting at 1 and
    /// incrementing by 1 for each call).
    pub fn push_segment(&mut self, data: Vec<u8>, duration_samples: u32) {
        let seg = CmafSegment {
            sequence: self.next_sequence,
            start_time: self.next_start_time,
            duration: duration_samples,
            data,
        };
        self.next_start_time += u64::from(duration_samples);
        self.next_sequence += 1;
        self.segments.push(seg);
    }

    /// Writes the CMAF init segment (`ftyp` + `moov`).
    ///
    /// The `ftyp` box uses brand `cmf2` with compatible brands `[cmf2, iso6,
    /// mp41]`.  The `moov` box is a minimal stub — real encoders would embed
    /// `trak`/`mvex` sub-boxes, but for the purposes of this implementation
    /// we write only the box header so that downstream tooling can identify
    /// the moov boundary.
    #[must_use]
    pub fn write_init_segment(&self) -> Vec<u8> {
        let mut out = Vec::new();

        // ── ftyp ─────────────────────────────────────────────────────────────
        // Layout: size(4) + "ftyp"(4) + major_brand(4) + minor_version(4) +
        //         compatible_brands(n × 4)
        let compatible_brands: &[&[u8; 4]] = &[b"cmf2", b"iso6", b"mp41"];
        // 8 bytes header + 4 major + 4 minor + brands
        let ftyp_size: u32 = 8 + 4 + 4 + (compatible_brands.len() as u32 * 4);
        write_u32(&mut out, ftyp_size);
        out.extend_from_slice(b"ftyp");
        out.extend_from_slice(b"cmf2"); // major brand
        write_u32(&mut out, 0); // minor version
        for brand in compatible_brands {
            out.extend_from_slice(*brand);
        }

        // ── moov ─────────────────────────────────────────────────────────────
        // Minimal moov: just the box header (size + fourcc).  A full CMAF moov
        // would contain trak/mvex/udta, which is out of scope for this stub.
        let moov_size: u32 = 8; // header only
        write_u32(&mut out, moov_size);
        out.extend_from_slice(b"moov");

        out
    }

    /// Writes the `moof` + `mdat` boxes for the segment at index `idx`.
    ///
    /// Returns `None` if `idx` is out of range.
    ///
    /// # Box layout
    ///
    /// ```text
    /// moof (8 bytes header + mfhd sub-box + traf sub-box)
    ///   mfhd: sequence_number (u32)
    ///   traf:
    ///     tfhd: track_id (u32) + flags (u32)
    ///     trun: sample_count=1, duration (u32), data_offset (u32)
    /// mdat (8 bytes header + media data)
    /// ```
    #[must_use]
    pub fn write_segment(&self, idx: usize) -> Option<Vec<u8>> {
        let seg = self.segments.get(idx)?;

        let mut out = Vec::new();

        // ── mfhd sub-box ──────────────────────────────────────────────────────
        // size(4) + "mfhd"(4) + version(1) + flags(3) + sequence_number(4) = 16
        let mut mfhd = Vec::new();
        write_u32(&mut mfhd, 16); // size
        mfhd.extend_from_slice(b"mfhd");
        write_u32(&mut mfhd, 0); // version(1) + flags(3) = 0
        write_u32(&mut mfhd, seg.sequence as u32);

        // ── tfhd sub-box ─────────────────────────────────────────────────────
        // size(4) + "tfhd"(4) + version(1) + flags(3) + track_id(4) = 16
        let mut tfhd = Vec::new();
        write_u32(&mut tfhd, 16); // size
        tfhd.extend_from_slice(b"tfhd");
        write_u32(&mut tfhd, 0); // version + flags
        write_u32(&mut tfhd, self.config.track_id);

        // ── trun sub-box ─────────────────────────────────────────────────────
        // Flags = 0x000_b01 (data-offset present | sample-duration present)
        // size(4) + "trun"(4) + version(1) + flags(3) + sample_count(4) +
        // data_offset(4) + sample_duration(4) = 24
        let mdat_payload_size = seg.data.len() as u32;
        // data_offset is relative to the start of `moof` — we compute it after
        // we know moof's total size.  Use a placeholder first.
        let mut trun = Vec::new();
        write_u32(&mut trun, 24); // size
        trun.extend_from_slice(b"trun");
        write_u32(&mut trun, 0x0000_0b01); // version=0, flags: data-offset + duration
        write_u32(&mut trun, 1); // sample_count
        write_u32(&mut trun, 0); // data_offset placeholder
        write_u32(&mut trun, seg.duration);

        // ── traf sub-box ─────────────────────────────────────────────────────
        let traf_payload_len = tfhd.len() + trun.len();
        let traf_size = 8 + traf_payload_len as u32;
        let mut traf = Vec::new();
        write_u32(&mut traf, traf_size);
        traf.extend_from_slice(b"traf");
        traf.extend_from_slice(&tfhd);
        traf.extend_from_slice(&trun);

        // ── moof box ─────────────────────────────────────────────────────────
        let moof_payload = mfhd.len() + traf.len();
        let moof_size = 8 + moof_payload as u32;

        // data_offset in trun = moof_size + 8 (mdat header)
        let data_offset: u32 = moof_size + 8;
        // Patch traf's trun data_offset field.
        // trun starts at offset 8 (traf header) + 16 (tfhd) within traf.
        // Within trun: 8 header + 4 flags/version + 4 sample_count = offset 16 = data_offset field.
        let traf_trun_offset = 8 + 16; // within traf bytes
        let data_offset_offset = traf_trun_offset + 12; // 8 (trun hdr) + 4 (version/flags)
        if data_offset_offset + 4 <= traf.len() {
            let bytes = data_offset.to_be_bytes();
            traf[data_offset_offset] = bytes[0];
            traf[data_offset_offset + 1] = bytes[1];
            traf[data_offset_offset + 2] = bytes[2];
            traf[data_offset_offset + 3] = bytes[3];
        }

        write_u32(&mut out, moof_size);
        out.extend_from_slice(b"moof");
        out.extend_from_slice(&mfhd);
        out.extend_from_slice(&traf);

        // ── mdat box ─────────────────────────────────────────────────────────
        let mdat_size: u32 = 8 + mdat_payload_size;
        write_u32(&mut out, mdat_size);
        out.extend_from_slice(b"mdat");
        out.extend_from_slice(&seg.data);

        Some(out)
    }
}

// ─── helpers ─────────────────────────────────────────────────────────────────

/// Appends a big-endian `u32` to `buf`.
fn write_u32(buf: &mut Vec<u8>, value: u32) {
    buf.extend_from_slice(&value.to_be_bytes());
}

// read_u32 and read_fourcc are only used in tests; they live in the test module.

#[cfg(test)]
mod tests {
    use super::*;

    fn read_u32(buf: &[u8], offset: usize) -> Option<u32> {
        let slice = buf.get(offset..offset + 4)?;
        Some(u32::from_be_bytes([slice[0], slice[1], slice[2], slice[3]]))
    }

    fn read_fourcc(buf: &[u8], offset: usize) -> Option<[u8; 4]> {
        let slice = buf.get(offset..offset + 4)?;
        Some([slice[0], slice[1], slice[2], slice[3]])
    }

    #[test]
    fn test_init_segment_starts_with_ftyp() {
        let track = CmafTrack::new(CmafConfig::video_default());
        let init = track.write_init_segment();
        // First 8 bytes: size + "ftyp"
        assert!(init.len() >= 8, "init segment too short");
        let fourcc = &init[4..8];
        assert_eq!(fourcc, b"ftyp", "expected ftyp, got {:?}", fourcc);
    }

    #[test]
    fn test_init_segment_ftyp_brand_cmf2() {
        let track = CmafTrack::new(CmafConfig::video_default());
        let init = track.write_init_segment();
        // major brand is at bytes 8..12
        let major = &init[8..12];
        assert_eq!(major, b"cmf2");
    }

    #[test]
    fn test_init_segment_contains_moov() {
        let track = CmafTrack::new(CmafConfig::video_default());
        let init = track.write_init_segment();
        // Locate moov by scanning after ftyp
        let ftyp_size = read_u32(&init, 0).expect("ftyp size") as usize;
        assert!(init.len() >= ftyp_size + 8, "moov header missing");
        let moov_fourcc = read_fourcc(&init, ftyp_size + 4).expect("moov fourcc");
        assert_eq!(&moov_fourcc, b"moov");
    }

    #[test]
    fn test_push_segment_increments_sequence() {
        let mut track = CmafTrack::new(CmafConfig::video_default());
        track.push_segment(vec![0u8; 512], 90_000);
        track.push_segment(vec![0u8; 256], 90_000);
        assert_eq!(track.segments[0].sequence, 1);
        assert_eq!(track.segments[1].sequence, 2);
    }

    #[test]
    fn test_push_segment_advances_start_time() {
        let mut track = CmafTrack::new(CmafConfig::video_default());
        track.push_segment(vec![1u8; 100], 90_000);
        track.push_segment(vec![2u8; 100], 45_000);
        assert_eq!(track.segments[0].start_time, 0);
        assert_eq!(track.segments[1].start_time, 90_000);
    }

    #[test]
    fn test_write_segment_none_for_invalid_index() {
        let track = CmafTrack::new(CmafConfig::video_default());
        assert!(track.write_segment(0).is_none());
    }

    #[test]
    fn test_write_segment_contains_moof_and_mdat() {
        let mut track = CmafTrack::new(CmafConfig::video_default());
        let payload = vec![0xAB_u8; 128];
        track.push_segment(payload.clone(), 90_000);

        let seg_bytes = track.write_segment(0).expect("segment bytes");

        // First box must be moof
        assert!(seg_bytes.len() >= 8, "segment too short");
        let moof_fourcc = read_fourcc(&seg_bytes, 4).expect("moof fourcc");
        assert_eq!(&moof_fourcc, b"moof", "expected moof");

        // Locate mdat after moof
        let moof_size = read_u32(&seg_bytes, 0).expect("moof size") as usize;
        assert!(seg_bytes.len() >= moof_size + 8, "mdat header missing");
        let mdat_fourcc = read_fourcc(&seg_bytes, moof_size + 4).expect("mdat fourcc");
        assert_eq!(&mdat_fourcc, b"mdat", "expected mdat");
    }

    #[test]
    fn test_write_segment_mdat_contains_payload() {
        let mut track = CmafTrack::new(CmafConfig::video_default());
        let payload: Vec<u8> = (0..64).collect();
        track.push_segment(payload.clone(), 45_000);

        let seg_bytes = track.write_segment(0).expect("segment bytes");
        let moof_size = read_u32(&seg_bytes, 0).expect("moof size") as usize;
        // mdat header is 8 bytes; payload starts at moof_size + 8
        let mdat_payload_start = moof_size + 8;
        let mdat_payload = &seg_bytes[mdat_payload_start..];
        assert_eq!(mdat_payload, payload.as_slice());
    }

    #[test]
    fn test_segment_count() {
        let mut track = CmafTrack::new(CmafConfig::audio_default());
        for _ in 0..5 {
            track.push_segment(vec![0u8; 32], 44_100);
        }
        assert_eq!(track.segments.len(), 5);
    }

    #[test]
    fn test_config_defaults() {
        let v = CmafConfig::video_default();
        assert_eq!(v.timescale, 90_000);
        assert_eq!(v.segment_duration_ms, 2_000);

        let a = CmafConfig::audio_default();
        assert_eq!(a.timescale, 44_100);
    }
}
