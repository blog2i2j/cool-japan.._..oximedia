// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Integration test: MP4 muxer writes a valid MJPEG Visual Sample Entry.
//!
//! Verifies that:
//! - `Mp4Muxer::add_stream(CodecId::Mjpeg)` is accepted (no PatentViolation error).
//! - After finalize, the resulting bytes contain the fourcc `b"jpeg"` somewhere
//!   in the stsd box (the ISOM-registered fourcc for MJPEG).

use bytes::Bytes;
use oximedia_container::{
    mux::mp4::{Mp4Config, Mp4Muxer},
    CodecParams, Packet, PacketFlags, StreamInfo,
};
use oximedia_core::{CodecId, Rational, Timestamp};

fn make_packet(stream_index: usize, data: Vec<u8>, timebase: Rational) -> Packet {
    let ts = Timestamp {
        pts: 0,
        dts: None,
        timebase,
        duration: Some(3000),
    };
    Packet::new(stream_index, Bytes::from(data), ts, PacketFlags::KEYFRAME)
}

/// Search a byte slice for a 4-byte sequence, returning true if found.
fn contains_bytes(haystack: &[u8], needle: &[u8; 4]) -> bool {
    haystack.windows(4).any(|w| w == needle)
}

#[test]
fn test_mjpeg_add_stream_accepted() {
    let mut muxer = Mp4Muxer::new(Mp4Config::new());

    let mut info = StreamInfo::new(0, CodecId::Mjpeg, Rational::new(1, 90000));
    info.codec_params = CodecParams::video(1920, 1080);

    let result = muxer.add_stream(info);
    assert!(
        result.is_ok(),
        "add_stream with CodecId::Mjpeg should not return a PatentViolation: {result:?}"
    );
}

#[test]
fn test_mjpeg_stsd_contains_jpeg_fourcc() {
    let mut muxer = Mp4Muxer::new(Mp4Config::new());

    let mut info = StreamInfo::new(0, CodecId::Mjpeg, Rational::new(1, 90000));
    info.codec_params = CodecParams::video(320, 240);

    muxer.add_stream(info).expect("add MJPEG stream");
    muxer.write_header().expect("write header");

    // Write one synthetic MJPEG packet (JPEG SOI + EOI = minimal valid JPEG).
    let jpeg_bytes: Vec<u8> = {
        let mut v = vec![0xFF, 0xD8]; // SOI
        v.extend_from_slice(&[0xFF, 0xE0, 0x00, 0x10]); // APP0 marker + length
        v.extend_from_slice(&[b'J', b'F', b'I', b'F', 0x00]); // JFIF identifier
        v.extend_from_slice(&[0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00]); // rest of APP0
        v.push(0xFF);
        v.push(0xD9); // EOI
        v
    };

    let pkt = make_packet(0, jpeg_bytes, Rational::new(1, 90000));
    muxer.write_packet(&pkt).expect("write packet");

    let output = muxer.finalize().expect("finalize");

    // The output must contain the MJPEG fourcc 'jpeg' (0x6A706567).
    assert!(
        contains_bytes(&output, b"jpeg"),
        "finalized MP4 must contain b\"jpeg\" fourcc for MJPEG visual sample entry"
    );
}

#[test]
fn test_mjpeg_stsd_not_rejected_as_patent_violation() {
    // Regression: before this fix, add_stream would return PatentViolation.
    let mut muxer = Mp4Muxer::new(Mp4Config::new());
    let mut info = StreamInfo::new(0, CodecId::Mjpeg, Rational::new(1, 30));
    info.codec_params = CodecParams::video(640, 480);

    let result = muxer.add_stream(info);
    assert!(
        result.is_ok(),
        "MJPEG must not be rejected as a patent violation; result: {result:?}"
    );
}

#[test]
fn test_mjpeg_track_count() {
    let mut muxer = Mp4Muxer::new(Mp4Config::new());

    let mut v_info = StreamInfo::new(0, CodecId::Mjpeg, Rational::new(1, 90000));
    v_info.codec_params = CodecParams::video(1920, 1080);

    muxer.add_stream(v_info).expect("MJPEG stream");
    assert_eq!(muxer.track_count(), 1);
}

#[test]
fn test_mjpeg_fragmented_mode_add_stream() {
    let config = Mp4Config::new().with_fragmented(2000);
    let mut muxer = Mp4Muxer::new(config);

    let mut info = StreamInfo::new(0, CodecId::Mjpeg, Rational::new(1, 90000));
    info.codec_params = CodecParams::video(1280, 720);

    let result = muxer.add_stream(info);
    assert!(
        result.is_ok(),
        "MJPEG accepted in fragmented mode: {result:?}"
    );
}
