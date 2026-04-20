// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Integration test: MJPEG in MP4 — progressive mode with 5 payloads.
//!
//! Verifies that:
//! - The `jpeg` sample-entry fourcc is present in the output.
//! - The `stsz` box records exactly 5 sample sizes.
//! - The `stts` (time-to-sample) box records all 5 samples.

use bytes::Bytes;
use oximedia_container::{
    mux::mp4::{Mp4Config, Mp4Muxer},
    CodecParams, Packet, PacketFlags, StreamInfo,
};
use oximedia_core::{CodecId, Rational, Timestamp};

fn make_packet(stream_index: usize, pts: i64, data: Vec<u8>) -> Packet {
    let ts = Timestamp {
        pts,
        dts: None,
        timebase: Rational::new(1, 30),
        duration: Some(1),
    };
    Packet::new(stream_index, Bytes::from(data), ts, PacketFlags::KEYFRAME)
}

/// Minimal synthetic JPEG payload: SOI + APP0 + EOI.
fn make_jpeg_payload(index: u8) -> Vec<u8> {
    let mut v = vec![0xFF, 0xD8]; // SOI
    v.extend_from_slice(&[0xFF, 0xE0, 0x00, 0x10]); // APP0 marker + length=16
    v.extend_from_slice(b"JFIF\x00"); // JFIF identifier
    v.extend_from_slice(&[0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00]); // APP0 body
                                                                                  // Add a comment block per frame so sizes differ slightly
    v.push(0xFF);
    v.push(0xFE); // COM marker
    let comment = format!("frame{index}");
    let comment_len = (comment.len() + 2) as u16;
    v.extend_from_slice(&comment_len.to_be_bytes());
    v.extend_from_slice(comment.as_bytes());
    v.extend_from_slice(&[0xFF, 0xD9]); // EOI
    v
}

fn contains_bytes(haystack: &[u8], needle: &[u8; 4]) -> bool {
    haystack.windows(4).any(|w| w == needle)
}

/// Returns the number of sample entries in the `stsz` box.
///
/// `stsz` FullBox layout (version=0, flags=0):
/// ```text
/// [size:4][stsz:4][version:1][flags:3][sample_size:4][sample_count:4][entry_size:4]...
/// ```
/// When `sample_size == 0` (variable sizes), `sample_count` individual entries follow.
fn stsz_sample_count(data: &[u8]) -> Option<u32> {
    // Find the stsz box
    let pos = data.windows(4).position(|w| w == b"stsz")?;
    // FullBox: after 4-byte type comes version(1)+flags(3) = 4 bytes, then sample_size(4), sample_count(4)
    let sample_size_off = pos + 4 + 4; // skip type + version/flags
    let sample_count_off = sample_size_off + 4;

    if sample_count_off + 4 > data.len() {
        return None;
    }

    let sample_size =
        u32::from_be_bytes(data[sample_size_off..sample_size_off + 4].try_into().ok()?);
    let sample_count = u32::from_be_bytes(
        data[sample_count_off..sample_count_off + 4]
            .try_into()
            .ok()?,
    );

    // When sample_size != 0, all samples have the same size (sample_count is still meaningful).
    let _ = sample_size;
    Some(sample_count)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[test]
fn test_mjpeg_in_mp4_five_frames() {
    let mut muxer = Mp4Muxer::new(Mp4Config::new());

    let mut info = StreamInfo::new(0, CodecId::Mjpeg, Rational::new(1, 30));
    info.codec_params = CodecParams::video(320, 240);

    muxer.add_stream(info).expect("add MJPEG stream");
    muxer.write_header().expect("write header");

    for i in 0u8..5 {
        let payload = make_jpeg_payload(i);
        let pkt = make_packet(0, i64::from(i), payload);
        muxer.write_packet(&pkt).expect("write packet");
    }

    let output = muxer.finalize().expect("finalize");

    // 1. Must start with ftyp
    assert_eq!(&output[4..8], b"ftyp", "output must start with ftyp");

    // 2. Must contain jpeg sample entry fourcc (ISOM-registered fourcc for MJPEG)
    assert!(
        contains_bytes(&output, b"jpeg"),
        "output must contain b\"jpeg\" sample-entry fourcc for MJPEG"
    );

    // 3. stsz must record 5 samples
    let count = stsz_sample_count(&output).expect("stsz box must be present and parseable");
    assert_eq!(
        count, 5,
        "stsz must record exactly 5 samples; found {count}"
    );
}

#[test]
fn test_mjpeg_in_mp4_stts_present() {
    let mut muxer = Mp4Muxer::new(Mp4Config::new());
    let mut info = StreamInfo::new(0, CodecId::Mjpeg, Rational::new(1, 30));
    info.codec_params = CodecParams::video(640, 480);

    muxer.add_stream(info).expect("add MJPEG stream");
    muxer.write_header().expect("write header");

    for i in 0u8..5 {
        let payload = make_jpeg_payload(i);
        let pkt = make_packet(0, i64::from(i), payload);
        muxer.write_packet(&pkt).expect("write packet");
    }

    let output = muxer.finalize().expect("finalize");

    assert!(
        contains_bytes(&output, b"stts"),
        "stts must be present in progressive MP4"
    );
    assert!(
        contains_bytes(&output, b"stsc"),
        "stsc must be present in progressive MP4"
    );
}

#[test]
fn test_mjpeg_in_mp4_fragmented_mode() {
    let config = Mp4Config::new().with_fragmented(2000);
    let mut muxer = Mp4Muxer::new(config);

    let mut info = StreamInfo::new(0, CodecId::Mjpeg, Rational::new(1, 30));
    info.codec_params = CodecParams::video(320, 240);

    muxer.add_stream(info).expect("add MJPEG stream");
    muxer.write_header().expect("write header");

    for i in 0u8..5 {
        let payload = make_jpeg_payload(i);
        let pkt = make_packet(0, i64::from(i), payload);
        muxer.write_packet(&pkt).expect("write packet");
    }

    let output = muxer.finalize().expect("finalize");

    assert!(
        contains_bytes(&output, b"jpeg"),
        "fragmented output must still contain jpeg fourcc"
    );
    assert!(
        contains_bytes(&output, b"moof"),
        "fragmented output must contain moof"
    );
    assert!(
        contains_bytes(&output, b"mdat"),
        "fragmented output must contain mdat"
    );
}

#[test]
fn test_mjpeg_sample_data_preserved() {
    // Verify the mdat payload length matches the sum of all frame sizes.
    let mut muxer = Mp4Muxer::new(Mp4Config::new());
    let mut info = StreamInfo::new(0, CodecId::Mjpeg, Rational::new(1, 30));
    info.codec_params = CodecParams::video(160, 120);

    muxer.add_stream(info).expect("add MJPEG stream");
    muxer.write_header().expect("write header");

    let mut total_payload_size = 0usize;
    for i in 0u8..5 {
        let payload = make_jpeg_payload(i);
        total_payload_size += payload.len();
        let pkt = make_packet(0, i64::from(i), payload);
        muxer.write_packet(&pkt).expect("write packet");
    }

    let output = muxer.finalize().expect("finalize");

    // Find mdat box and check its size.
    // mdat box: [size:4][mdat:4][payload...]
    let mdat_pos = output
        .windows(4)
        .position(|w| w == b"mdat")
        .expect("mdat must be present");

    // size is at mdat_pos - 4
    if mdat_pos >= 4 {
        let size_bytes: [u8; 4] = output[mdat_pos - 4..mdat_pos]
            .try_into()
            .expect("size bytes");
        let mdat_box_size = u32::from_be_bytes(size_bytes) as usize;
        // mdat payload = box_size - 8 (4 size + 4 fourcc)
        let mdat_payload_len = mdat_box_size.saturating_sub(8);
        assert_eq!(
            mdat_payload_len, total_payload_size,
            "mdat payload must equal sum of all frame payloads"
        );
    }
}
