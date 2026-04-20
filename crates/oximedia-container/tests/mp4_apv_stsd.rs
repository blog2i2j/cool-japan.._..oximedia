// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Integration test: MP4 muxer writes a valid APV Visual Sample Entry.
//!
//! Verifies that:
//! - `Mp4Muxer::add_stream(CodecId::Apv)` is accepted (no PatentViolation error).
//! - After finalize, the resulting bytes contain the fourcc `b"apv1"` somewhere
//!   in the stsd box (the ISO/IEC 23009-13 registered fourcc for APV).

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

/// Minimal synthetic APV access unit: magic + 12 bytes of header.
fn make_apv_au(width: u16, height: u16) -> Vec<u8> {
    let mut v = Vec::with_capacity(16);
    // Magic
    v.extend_from_slice(b"APV1");
    // profile (1 = Simple)
    v.push(1u8);
    // width (u16 LE)
    v.extend_from_slice(&width.to_le_bytes());
    // height (u16 LE)
    v.extend_from_slice(&height.to_le_bytes());
    // bit_depth (1 = 8-bit)
    v.push(1u8);
    // chroma_format (1 = 4:2:0)
    v.push(1u8);
    // qp
    v.push(22u8);
    // tile_cols (u16 LE)
    v.extend_from_slice(&1u16.to_le_bytes());
    // tile_rows (u16 LE)
    v.extend_from_slice(&1u16.to_le_bytes());
    // Minimal payload (1 tile with 8×8 DCT block)
    v.extend_from_slice(&[0u8; 32]);
    v
}

#[test]
fn test_apv_add_stream_accepted() {
    let mut muxer = Mp4Muxer::new(Mp4Config::new());

    let mut info = StreamInfo::new(0, CodecId::Apv, Rational::new(1, 90000));
    info.codec_params = CodecParams::video(1920, 1080);

    let result = muxer.add_stream(info);
    assert!(
        result.is_ok(),
        "add_stream with CodecId::Apv should not return a PatentViolation: {result:?}"
    );
}

#[test]
fn test_apv_stsd_contains_apv1_fourcc() {
    let mut muxer = Mp4Muxer::new(Mp4Config::new());

    let mut info = StreamInfo::new(0, CodecId::Apv, Rational::new(1, 90000));
    info.codec_params = CodecParams::video(320, 240);

    muxer.add_stream(info).expect("add APV stream");
    muxer.write_header().expect("write header");

    let apv_data = make_apv_au(320, 240);

    let pkt = make_packet(0, apv_data, Rational::new(1, 90000));
    muxer.write_packet(&pkt).expect("write packet");

    let output = muxer.finalize().expect("finalize");

    // The output must contain the APV fourcc 'apv1'.
    assert!(
        contains_bytes(&output, b"apv1"),
        "finalized MP4 must contain b\"apv1\" fourcc for APV visual sample entry"
    );
}

#[test]
fn test_apv_stsd_not_rejected_as_patent_violation() {
    // Regression: before this fix, add_stream would return PatentViolation.
    let mut muxer = Mp4Muxer::new(Mp4Config::new());
    let mut info = StreamInfo::new(0, CodecId::Apv, Rational::new(1, 30));
    info.codec_params = CodecParams::video(640, 480);

    let result = muxer.add_stream(info);
    assert!(
        result.is_ok(),
        "APV must not be rejected as a patent violation; result: {result:?}"
    );
}

#[test]
fn test_apv_track_count() {
    let mut muxer = Mp4Muxer::new(Mp4Config::new());

    let mut v_info = StreamInfo::new(0, CodecId::Apv, Rational::new(1, 90000));
    v_info.codec_params = CodecParams::video(1920, 1080);

    muxer.add_stream(v_info).expect("APV stream");
    assert_eq!(muxer.track_count(), 1);
}

#[test]
fn test_apv_fragmented_mode_add_stream() {
    let config = Mp4Config::new().with_fragmented(2000);
    let mut muxer = Mp4Muxer::new(config);

    let mut info = StreamInfo::new(0, CodecId::Apv, Rational::new(1, 90000));
    info.codec_params = CodecParams::video(1280, 720);

    let result = muxer.add_stream(info);
    assert!(
        result.is_ok(),
        "APV accepted in fragmented mode: {result:?}"
    );
}

#[test]
fn test_apv_mjpeg_both_added() {
    // Both APV and MJPEG tracks should be addable to the same muxer.
    let mut muxer = Mp4Muxer::new(Mp4Config::new());

    let mut v1 = StreamInfo::new(0, CodecId::Apv, Rational::new(1, 90000));
    v1.codec_params = CodecParams::video(1920, 1080);

    let mut v2 = StreamInfo::new(1, CodecId::Mjpeg, Rational::new(1, 90000));
    v2.codec_params = CodecParams::video(480, 270);

    muxer.add_stream(v1).expect("APV stream");
    muxer.add_stream(v2).expect("MJPEG stream");
    assert_eq!(muxer.track_count(), 2);
}
