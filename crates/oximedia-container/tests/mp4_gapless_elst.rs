//! Integration test for gapless audio via the MP4 Edit List box (`elst`).
//!
//! Builds a minimal, hand-crafted MP4 byte stream that includes an `elst`
//! box encoding **2112 pre-roll (encoder delay) samples**, then parses it
//! with `Mp4Demuxer` and asserts that `preroll_samples == 2112`.
//!
//! The file structure is:
//!   `ftyp` → `moov` → `mvhd` + `trak` → `tkhd` + `edts`/`elst` +
//!             `mdia` → `mdhd` + `hdlr` + `minf` → `stbl` → `stsd`+`stts`+
//!             `stsc`+`stsz`+`stco`
//!   `mdat` (1 byte)

use oximedia_container::demux::Mp4Demuxer;
use oximedia_container::Demuxer;
use oximedia_io::MemorySource;

// ---------------------------------------------------------------------------
// Helper: big-endian 4-byte box builder
// ---------------------------------------------------------------------------

/// Prepend an 8-byte box header (size, 4CC) to `content`.
fn box_with_header(four_cc: &[u8; 4], content: &[u8]) -> Vec<u8> {
    let total = 8u32 + content.len() as u32;
    let mut out = Vec::with_capacity(total as usize);
    out.extend_from_slice(&total.to_be_bytes());
    out.extend_from_slice(four_cc);
    out.extend_from_slice(content);
    out
}

fn u32be(v: u32) -> [u8; 4] {
    v.to_be_bytes()
}

fn u64be(v: u64) -> [u8; 8] {
    v.to_be_bytes()
}

fn u16be(v: u16) -> [u8; 2] {
    v.to_be_bytes()
}

fn i32be(v: i32) -> [u8; 4] {
    v.to_be_bytes()
}

// ---------------------------------------------------------------------------
// Build the elst box
// ---------------------------------------------------------------------------

/// Builds a version-0 `elst` box with one entry:
///   segment_duration = total_duration (u32)
///   media_time       = preroll       (i32)
///   media_rate       = 0x00010000    (u32, = 1.0)
fn build_elst_v0(total_duration: u32, preroll: i32) -> Vec<u8> {
    let mut content = Vec::new();
    // version=0, flags=0
    content.push(0u8);
    content.extend_from_slice(&[0u8; 3]);
    // entry_count = 1
    content.extend_from_slice(&u32be(1));
    // entry: segment_duration
    content.extend_from_slice(&u32be(total_duration));
    // entry: media_time (pre-roll)
    content.extend_from_slice(&i32be(preroll));
    // entry: media_rate (1.0 in 16.16)
    content.extend_from_slice(&u32be(0x0001_0000));
    box_with_header(b"elst", &content)
}

/// Builds a version-1 `elst` box with one entry (64-bit fields):
///   segment_duration = total_duration (u64)
///   media_time       = preroll       (i64)
///   media_rate       = 0x00010000    (u32)
fn build_elst_v1(total_duration: u64, preroll: i64) -> Vec<u8> {
    let mut content = Vec::new();
    // version=1, flags=0
    content.push(1u8);
    content.extend_from_slice(&[0u8; 3]);
    // entry_count = 1
    content.extend_from_slice(&u32be(1));
    // entry: segment_duration (u64)
    content.extend_from_slice(&u64be(total_duration));
    // entry: media_time (i64)
    content.extend_from_slice(&(preroll as u64).to_be_bytes());
    // entry: media_rate (u32)
    content.extend_from_slice(&u32be(0x0001_0000));
    box_with_header(b"elst", &content)
}

/// Builds an `edts` box wrapping the given `elst` bytes.
fn build_edts(elst: &[u8]) -> Vec<u8> {
    box_with_header(b"edts", elst)
}

// ---------------------------------------------------------------------------
// Build the rest of the trak
// ---------------------------------------------------------------------------

fn build_tkhd(track_id: u32, duration: u32, is_audio: bool) -> Vec<u8> {
    let mut content = Vec::new();
    content.push(0u8); // version=0
    content.extend_from_slice(&u32be(0x00000003u32)[1..]); // flags (lower 3 bytes)
    content.extend_from_slice(&u32be(0)); // creation_time
    content.extend_from_slice(&u32be(0)); // modification_time
    content.extend_from_slice(&u32be(track_id));
    content.extend_from_slice(&u32be(0)); // reserved
    content.extend_from_slice(&u32be(duration));
    content.extend_from_slice(&[0u8; 8]); // reserved
    content.extend_from_slice(&[0u8; 4]); // layer + alternate_group
    content.extend_from_slice(&[0u8; 4]); // volume + reserved
    content.extend_from_slice(&[0u8; 36]); // matrix
    if is_audio {
        content.extend_from_slice(&u32be(0)); // width = 0
        content.extend_from_slice(&u32be(0)); // height = 0
    } else {
        content.extend_from_slice(&u32be(0x0500_0000)); // width = 1280 fixed
        content.extend_from_slice(&u32be(0x02D0_0000)); // height = 720 fixed
    }
    box_with_header(b"tkhd", &content)
}

fn build_mdhd(timescale: u32, duration: u32) -> Vec<u8> {
    let mut content = Vec::new();
    content.push(0u8); // version=0
    content.extend_from_slice(&[0u8; 3]); // flags
    content.extend_from_slice(&u32be(0)); // creation_time
    content.extend_from_slice(&u32be(0)); // modification_time
    content.extend_from_slice(&u32be(timescale));
    content.extend_from_slice(&u32be(duration));
    content.extend_from_slice(&u16be(0x55C4)); // language = "und"
    content.extend_from_slice(&u16be(0)); // pre_defined
    box_with_header(b"mdhd", &content)
}

fn build_hdlr(handler: &[u8; 4]) -> Vec<u8> {
    let mut content = Vec::new();
    content.push(0u8); // version
    content.extend_from_slice(&[0u8; 3]); // flags
    content.extend_from_slice(&u32be(0)); // pre_defined
    content.extend_from_slice(handler); // handler_type e.g. "soun"
    content.extend_from_slice(&[0u8; 12]); // reserved
    content.push(0u8); // name (empty string, null-terminated)
    box_with_header(b"hdlr", &content)
}

/// Build a minimal Opus audio sample entry (codec tag = "Opus").
fn build_stsd_opus(sample_rate: u32, channels: u16) -> Vec<u8> {
    let mut entry = Vec::new();
    entry.extend_from_slice(b"Opus"); // codec_tag
    entry.extend_from_slice(&[0u8; 6]); // reserved
    entry.extend_from_slice(&u16be(1)); // data_reference_index

    // AudioSampleEntry fields
    entry.extend_from_slice(&[0u8; 8]); // reserved
    entry.extend_from_slice(&u16be(channels));
    entry.extend_from_slice(&u16be(16)); // samplesize
    entry.extend_from_slice(&u16be(0)); // pre_defined
    entry.extend_from_slice(&u16be(0)); // reserved

    // samplerate as 16.16 fixed-point
    let sr_fp = (sample_rate as u64) << 16;
    entry.extend_from_slice(&u32be(sr_fp as u32));

    // dOps box (minimal: version=0, channel_count=2, pre_skip=312, sample_rate=48000)
    let dops_content: &[u8] = &[
        0u8, // version
        2u8, // OutputChannelCount
        0x01, 0x38, // pre_skip = 312
        0x00, 0x00, 0xBB, 0x80, // InputSampleRate = 48000
        0x00, 0x00, // output_gain
        0u8,  // ChannelMappingFamily = 0 (stereo)
    ];
    let dops_box = box_with_header(b"dOps", dops_content);
    entry.extend_from_slice(&dops_box);

    // Wrap in stsd
    let entry_size = 8u32 + entry.len() as u32;
    let mut stsd_content = Vec::new();
    stsd_content.push(0u8); // version
    stsd_content.extend_from_slice(&[0u8; 3]); // flags
    stsd_content.extend_from_slice(&u32be(1)); // entry_count
    stsd_content.extend_from_slice(&entry_size.to_be_bytes());
    stsd_content.extend_from_slice(&entry);
    box_with_header(b"stsd", &stsd_content)
}

/// Build a minimal `stts` box (1 entry: N samples, each delta 1).
fn build_stts(sample_count: u32) -> Vec<u8> {
    let mut content = Vec::new();
    content.push(0u8); // version
    content.extend_from_slice(&[0u8; 3]); // flags
    content.extend_from_slice(&u32be(1)); // entry_count
    content.extend_from_slice(&u32be(sample_count)); // sample_count
    content.extend_from_slice(&u32be(1)); // sample_delta
    box_with_header(b"stts", &content)
}

/// Build a minimal `stsc` box.
fn build_stsc_minimal() -> Vec<u8> {
    let mut content = Vec::new();
    content.push(0u8);
    content.extend_from_slice(&[0u8; 3]); // flags
    content.extend_from_slice(&u32be(1)); // entry_count
    content.extend_from_slice(&u32be(1)); // first_chunk
    content.extend_from_slice(&u32be(1)); // samples_per_chunk
    content.extend_from_slice(&u32be(1)); // sample_description_index
    box_with_header(b"stsc", &content)
}

/// Build a minimal `stsz` box (all same size = 1 byte).
fn build_stsz(sample_count: u32) -> Vec<u8> {
    let mut content = Vec::new();
    content.push(0u8);
    content.extend_from_slice(&[0u8; 3]); // flags
    content.extend_from_slice(&u32be(1)); // sample_size = 1 (constant)
    content.extend_from_slice(&u32be(sample_count));
    box_with_header(b"stsz", &content)
}

/// Build a minimal `stco` box pointing to offset 0.
fn build_stco() -> Vec<u8> {
    let mut content = Vec::new();
    content.push(0u8);
    content.extend_from_slice(&[0u8; 3]); // flags
    content.extend_from_slice(&u32be(1)); // entry_count
    content.extend_from_slice(&u32be(0)); // chunk_offset
    box_with_header(b"stco", &content)
}

// ---------------------------------------------------------------------------
// Build the complete MP4 for a gapless audio track
// ---------------------------------------------------------------------------

/// Build a self-contained MP4 with a single Opus audio track that has an
/// `elst` box recording `preroll_samples` encoder delay samples.
///
/// The file layout is:
///   `ftyp` | `moov` { `mvhd` | `trak` { `tkhd` | `edts`{`elst`} |
///     `mdia`{`mdhd`|`hdlr`|`minf`{`smhd`|`dinf`|`stbl`{stsd|stts|stsc|stsz|stco}}} } }
///   | `mdat` { 0x00 }
fn build_gapless_mp4(preroll_samples: i32, use_version1: bool) -> Vec<u8> {
    let sample_count: u32 = 4800; // ~0.1 s at 48 kHz
    let timescale: u32 = 48000;
    let duration: u32 = sample_count;

    // edts / elst
    let elst = if use_version1 {
        build_elst_v1(u64::from(duration), i64::from(preroll_samples))
    } else {
        build_elst_v0(duration, preroll_samples)
    };
    let edts = build_edts(&elst);

    // tkhd
    let tkhd = build_tkhd(1, duration, true);

    // mdia children
    let mdhd = build_mdhd(timescale, duration);
    let hdlr = build_hdlr(b"soun");
    let stsd = build_stsd_opus(timescale, 2);
    let stts = build_stts(sample_count);
    let stsc = build_stsc_minimal();
    let stsz = build_stsz(sample_count);
    let stco = build_stco();

    // stbl
    let mut stbl_content = Vec::new();
    stbl_content.extend_from_slice(&stsd);
    stbl_content.extend_from_slice(&stts);
    stbl_content.extend_from_slice(&stsc);
    stbl_content.extend_from_slice(&stsz);
    stbl_content.extend_from_slice(&stco);
    let stbl = box_with_header(b"stbl", &stbl_content);

    // smhd (sound media header)
    let smhd = box_with_header(b"smhd", &[0u8; 8]);

    // dinf / dref (minimal)
    let dref_content: Vec<u8> = {
        let mut c = Vec::new();
        c.push(0u8);
        c.extend_from_slice(&[0u8; 3]); // version/flags
        c.extend_from_slice(&u32be(1)); // entry_count
                                        // url entry: size=12, type='url ', version=0, flags=0x000001
        c.extend_from_slice(&u32be(12));
        c.extend_from_slice(b"url ");
        c.push(0);
        c.extend_from_slice(&[0, 0, 1]); // flags = self-contained
        c
    };
    let dref = box_with_header(b"dref", &dref_content);
    let dinf = box_with_header(b"dinf", &dref);

    // minf
    let mut minf_content = Vec::new();
    minf_content.extend_from_slice(&smhd);
    minf_content.extend_from_slice(&dinf);
    minf_content.extend_from_slice(&stbl);
    let minf = box_with_header(b"minf", &minf_content);

    // mdia
    let mut mdia_content = Vec::new();
    mdia_content.extend_from_slice(&mdhd);
    mdia_content.extend_from_slice(&hdlr);
    mdia_content.extend_from_slice(&minf);
    let mdia = box_with_header(b"mdia", &mdia_content);

    // trak
    let mut trak_content = Vec::new();
    trak_content.extend_from_slice(&tkhd);
    trak_content.extend_from_slice(&edts);
    trak_content.extend_from_slice(&mdia);
    let trak = box_with_header(b"trak", &trak_content);

    // mvhd (version 0)
    let mvhd = {
        let mut content = Vec::new();
        content.push(0u8); // version
        content.extend_from_slice(&[0u8; 3]); // flags
        content.extend_from_slice(&u32be(0)); // creation_time
        content.extend_from_slice(&u32be(0)); // modification_time
        content.extend_from_slice(&u32be(timescale));
        content.extend_from_slice(&u32be(duration));
        content.extend_from_slice(&u32be(0x0001_0000)); // rate = 1.0
        content.extend_from_slice(&u16be(0x0100)); // volume = 1.0
        content.extend_from_slice(&[0u8; 2]); // reserved
        content.extend_from_slice(&[0u8; 8]); // reserved
        content.extend_from_slice(&[0u8; 36]); // matrix
        content.extend_from_slice(&[0u8; 24]); // pre_defined
        content.extend_from_slice(&u32be(2)); // next_track_id
        box_with_header(b"mvhd", &content)
    };

    // moov
    let mut moov_content = Vec::new();
    moov_content.extend_from_slice(&mvhd);
    moov_content.extend_from_slice(&trak);
    let moov = box_with_header(b"moov", &moov_content);

    // ftyp
    let ftyp = {
        let mut content = Vec::new();
        content.extend_from_slice(b"isom"); // major_brand
        content.extend_from_slice(&u32be(0)); // minor_version
        content.extend_from_slice(b"isom"); // compatible_brand
        content.extend_from_slice(b"iso2"); // compatible_brand
        box_with_header(b"ftyp", &content)
    };

    // mdat (1 dummy byte)
    let mdat = box_with_header(b"mdat", &[0u8]);

    let mut out = Vec::new();
    out.extend_from_slice(&ftyp);
    out.extend_from_slice(&moov);
    out.extend_from_slice(&mdat);
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_elst_v0_preroll_2112() {
    let data = build_gapless_mp4(2112, false);
    let source = MemorySource::new(bytes::Bytes::from(data));
    let mut demuxer = Mp4Demuxer::new(source);
    demuxer.probe().await.expect("probe should succeed");

    let streams = demuxer.streams();
    assert!(!streams.is_empty(), "Should have at least one stream");

    // Access the raw trak boxes to check preroll_samples
    let traks = demuxer.traks();
    assert!(!traks.is_empty(), "Should have at least one trak");

    let audio_trak = traks
        .iter()
        .find(|t| t.handler_type == "soun")
        .expect("Should find audio trak");

    assert_eq!(
        audio_trak.preroll_samples, 2112,
        "preroll_samples should be 2112 from elst v0, got {}",
        audio_trak.preroll_samples
    );
}

#[tokio::test]
async fn test_elst_v1_preroll_2112() {
    let data = build_gapless_mp4(2112, true);
    let source = MemorySource::new(bytes::Bytes::from(data));
    let mut demuxer = Mp4Demuxer::new(source);
    demuxer.probe().await.expect("probe should succeed");

    let traks = demuxer.traks();
    let audio_trak = traks
        .iter()
        .find(|t| t.handler_type == "soun")
        .expect("Should find audio trak");

    assert_eq!(
        audio_trak.preroll_samples, 2112,
        "preroll_samples should be 2112 from elst v1, got {}",
        audio_trak.preroll_samples
    );
}

#[tokio::test]
async fn test_elst_zero_preroll() {
    // When media_time = 0 there is no pre-roll to report.
    let data = build_gapless_mp4(0, false);
    let source = MemorySource::new(bytes::Bytes::from(data));
    let mut demuxer = Mp4Demuxer::new(source);
    demuxer.probe().await.expect("probe should succeed");

    let traks = demuxer.traks();
    let audio_trak = traks
        .iter()
        .find(|t| t.handler_type == "soun")
        .expect("Should find audio trak");

    assert_eq!(
        audio_trak.preroll_samples, 0,
        "preroll_samples should be 0 when media_time=0, got {}",
        audio_trak.preroll_samples
    );
}

#[tokio::test]
async fn test_elst_empty_edit_prefix() {
    // Pattern: Entry 0 = empty edit (media_time = -1), Entry 1 = preroll
    // Build the elst manually for this pattern.
    let sample_count: u32 = 4800;
    let timescale: u32 = 48000;
    let duration: u32 = sample_count;
    let preroll: i32 = 576; // common Opus pre-roll

    // Two-entry elst: empty edit + actual edit
    let elst = {
        let mut content = Vec::new();
        content.push(0u8); // version=0
        content.extend_from_slice(&[0u8; 3]); // flags
        content.extend_from_slice(&u32be(2)); // entry_count = 2
                                              // Entry 0: empty edit (silence)
        content.extend_from_slice(&u32be(576)); // segment_duration = preroll length
        content.extend_from_slice(&i32be(-1)); // media_time = -1 (empty edit)
        content.extend_from_slice(&u32be(0x0001_0000)); // media_rate
                                                        // Entry 1: actual media
        content.extend_from_slice(&u32be(duration));
        content.extend_from_slice(&i32be(preroll)); // media_time = pre-roll
        content.extend_from_slice(&u32be(0x0001_0000)); // media_rate
        box_with_header(b"elst", &content)
    };
    let edts = build_edts(&elst);

    let tkhd = build_tkhd(1, duration, true);
    let mdhd = build_mdhd(timescale, duration);
    let hdlr = build_hdlr(b"soun");
    let stsd = build_stsd_opus(timescale, 2);
    let stts = build_stts(sample_count);
    let stsc = build_stsc_minimal();
    let stsz = build_stsz(sample_count);
    let stco = build_stco();

    let mut stbl_content = Vec::new();
    stbl_content.extend_from_slice(&stsd);
    stbl_content.extend_from_slice(&stts);
    stbl_content.extend_from_slice(&stsc);
    stbl_content.extend_from_slice(&stsz);
    stbl_content.extend_from_slice(&stco);
    let stbl = box_with_header(b"stbl", &stbl_content);

    let smhd = box_with_header(b"smhd", &[0u8; 8]);
    let dref_content: Vec<u8> = {
        let mut c = Vec::new();
        c.push(0u8);
        c.extend_from_slice(&[0u8; 3]);
        c.extend_from_slice(&u32be(1));
        c.extend_from_slice(&u32be(12));
        c.extend_from_slice(b"url ");
        c.push(0);
        c.extend_from_slice(&[0, 0, 1]);
        c
    };
    let dref = box_with_header(b"dref", &dref_content);
    let dinf = box_with_header(b"dinf", &dref);

    let mut minf_content = Vec::new();
    minf_content.extend_from_slice(&smhd);
    minf_content.extend_from_slice(&dinf);
    minf_content.extend_from_slice(&stbl);
    let minf = box_with_header(b"minf", &minf_content);

    let mut mdia_content = Vec::new();
    mdia_content.extend_from_slice(&mdhd);
    mdia_content.extend_from_slice(&hdlr);
    mdia_content.extend_from_slice(&minf);
    let mdia = box_with_header(b"mdia", &mdia_content);

    let mut trak_content = Vec::new();
    trak_content.extend_from_slice(&tkhd);
    trak_content.extend_from_slice(&edts);
    trak_content.extend_from_slice(&mdia);
    let trak = box_with_header(b"trak", &trak_content);

    let mvhd = {
        let mut content = Vec::new();
        content.push(0u8);
        content.extend_from_slice(&[0u8; 3]);
        content.extend_from_slice(&u32be(0));
        content.extend_from_slice(&u32be(0));
        content.extend_from_slice(&u32be(timescale));
        content.extend_from_slice(&u32be(duration));
        content.extend_from_slice(&u32be(0x0001_0000));
        content.extend_from_slice(&u16be(0x0100));
        content.extend_from_slice(&[0u8; 2]);
        content.extend_from_slice(&[0u8; 8]);
        content.extend_from_slice(&[0u8; 36]);
        content.extend_from_slice(&[0u8; 24]);
        content.extend_from_slice(&u32be(2));
        box_with_header(b"mvhd", &content)
    };

    let mut moov_content = Vec::new();
    moov_content.extend_from_slice(&mvhd);
    moov_content.extend_from_slice(&trak);
    let moov = box_with_header(b"moov", &moov_content);

    let ftyp = {
        let mut content = Vec::new();
        content.extend_from_slice(b"isom");
        content.extend_from_slice(&u32be(0));
        content.extend_from_slice(b"isom");
        box_with_header(b"ftyp", &content)
    };

    let mdat = box_with_header(b"mdat", &[0u8]);

    let mut file_data = Vec::new();
    file_data.extend_from_slice(&ftyp);
    file_data.extend_from_slice(&moov);
    file_data.extend_from_slice(&mdat);

    let source = MemorySource::new(bytes::Bytes::from(file_data));
    let mut demuxer = Mp4Demuxer::new(source);
    demuxer.probe().await.expect("probe should succeed");

    let traks = demuxer.traks();
    let audio_trak = traks
        .iter()
        .find(|t| t.handler_type == "soun")
        .expect("Should find audio trak");

    assert_eq!(
        audio_trak.preroll_samples, 576,
        "preroll_samples from empty-edit pattern should be 576, got {}",
        audio_trak.preroll_samples
    );
}
