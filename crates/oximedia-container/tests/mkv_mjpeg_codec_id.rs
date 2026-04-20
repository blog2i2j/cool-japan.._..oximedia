// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Integration test: Matroska muxer emits `V_MJPEG` as the CodecID for MJPEG tracks.
//!
//! Writes a minimal MKV file with one MJPEG video stream, then scans the raw
//! output bytes for the ASCII string `"V_MJPEG"`.  This is a smoke test that
//! does not require a Matroska demuxer round-trip.

use bytes::Bytes;
use oximedia_container::{
    mux::{MatroskaMuxer, Muxer, MuxerConfig},
    CodecParams, Packet, PacketFlags, StreamInfo,
};
use oximedia_core::{CodecId, Rational, Timestamp};
use oximedia_io::MemorySource;

/// Returns `true` if `needle` (ASCII) is present anywhere in `haystack`.
fn contains_str(haystack: &[u8], needle: &[u8]) -> bool {
    haystack.windows(needle.len()).any(|w| w == needle)
}

fn make_mjpeg_stream() -> StreamInfo {
    let mut info = StreamInfo::new(0, CodecId::Mjpeg, Rational::new(1, 25));
    info.codec_params = CodecParams::video(320, 240);
    info
}

fn make_fake_packet(stream_index: usize) -> Packet {
    // Minimal synthetic JPEG: SOI + EOI only.
    let data = Bytes::from_static(&[0xFF, 0xD8, 0xFF, 0xD9]);
    let ts = Timestamp {
        pts: 0,
        dts: None,
        timebase: Rational::new(1, 25),
        duration: Some(1),
    };
    Packet::new(stream_index, data, ts, PacketFlags::KEYFRAME)
}

#[tokio::test]
async fn test_mkv_mjpeg_codec_id_is_v_mjpeg() {
    let sink = MemorySource::new_writable(65536);
    let config = MuxerConfig::new();
    let mut muxer = MatroskaMuxer::new(sink, config);

    muxer
        .add_stream(make_mjpeg_stream())
        .expect("add MJPEG stream");
    muxer.write_header().await.expect("write header");

    let pkt = make_fake_packet(0);
    muxer.write_packet(&pkt).await.expect("write packet");
    muxer.write_trailer().await.expect("write trailer");

    let output = muxer.sink().written_data().to_vec();

    assert!(
        contains_str(&output, b"V_MJPEG"),
        "MKV output must contain the codec ID string 'V_MJPEG'"
    );
}

#[tokio::test]
async fn test_mkv_mjpeg_format_is_matroska() {
    // MJPEG is not a WebM-compatible codec, so the muxer must produce a
    // Matroska (not WebM) container — verify via 'matroska' doctype string.
    let sink = MemorySource::new_writable(65536);
    let config = MuxerConfig::new();
    let mut muxer = MatroskaMuxer::new(sink, config);

    muxer
        .add_stream(make_mjpeg_stream())
        .expect("add MJPEG stream");
    muxer.write_header().await.expect("write header");

    let output = muxer.sink().written_data().to_vec();

    assert!(
        contains_str(&output, b"matroska"),
        "DocType must be 'matroska' for an MJPEG stream (not 'webm')"
    );
}

#[tokio::test]
async fn test_mkv_mjpeg_no_codec_private() {
    // V_MJPEG does not require CodecPrivate; we verify that the raw bytes do NOT
    // contain a spurious BITMAPINFOHEADER by checking that the 'apv1' fourcc is
    // absent (i.e. the APV private data was not accidentally injected).
    let sink = MemorySource::new_writable(65536);
    let config = MuxerConfig::new();
    let mut muxer = MatroskaMuxer::new(sink, config);

    muxer
        .add_stream(make_mjpeg_stream())
        .expect("add MJPEG stream");
    muxer.write_header().await.expect("write header");

    let output = muxer.sink().written_data().to_vec();

    assert!(
        !contains_str(&output, b"apv1"),
        "MJPEG MKV header must not contain the 'apv1' fourcc"
    );
}
