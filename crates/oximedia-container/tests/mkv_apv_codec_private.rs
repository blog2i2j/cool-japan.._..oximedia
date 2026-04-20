// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Integration test: Matroska muxer emits `V_MS/VFW/FOURCC` and a
//! `BITMAPINFOHEADER` CodecPrivate blob containing `apv1` for APV tracks.
//!
//! Writes a minimal MKV file with one APV video stream, then scans the raw
//! output bytes for:
//!  - the codec ID string `"V_MS/VFW/FOURCC"`
//!  - the fourcc `b"apv1"` inside the CodecPrivate BITMAPINFOHEADER

use bytes::Bytes;
use oximedia_container::{
    mux::{MatroskaMuxer, Muxer, MuxerConfig},
    CodecParams, Packet, PacketFlags, StreamInfo,
};
use oximedia_core::{CodecId, Rational, Timestamp};
use oximedia_io::MemorySource;

/// Returns `true` if `needle` bytes are present anywhere in `haystack`.
fn contains_bytes(haystack: &[u8], needle: &[u8]) -> bool {
    haystack.windows(needle.len()).any(|w| w == needle)
}

fn make_apv_stream() -> StreamInfo {
    let mut info = StreamInfo::new(0, CodecId::Apv, Rational::new(1, 30));
    info.codec_params = CodecParams::video(640, 480);
    info
}

fn make_fake_apv_packet(stream_index: usize) -> Packet {
    // Minimal synthetic APV payload (128 zero bytes).
    let data = Bytes::from(vec![0u8; 128]);
    let ts = Timestamp {
        pts: 0,
        dts: None,
        timebase: Rational::new(1, 30),
        duration: Some(1),
    };
    Packet::new(stream_index, data, ts, PacketFlags::KEYFRAME)
}

#[tokio::test]
async fn test_mkv_apv_codec_id_is_vfw_fourcc() {
    let sink = MemorySource::new_writable(65536);
    let config = MuxerConfig::new();
    let mut muxer = MatroskaMuxer::new(sink, config);

    muxer.add_stream(make_apv_stream()).expect("add APV stream");
    muxer.write_header().await.expect("write header");

    let pkt = make_fake_apv_packet(0);
    muxer.write_packet(&pkt).await.expect("write packet");
    muxer.write_trailer().await.expect("write trailer");

    let output = muxer.sink().written_data().to_vec();

    assert!(
        contains_bytes(&output, b"V_MS/VFW/FOURCC"),
        "MKV output must contain the codec ID string 'V_MS/VFW/FOURCC' for APV"
    );
}

#[tokio::test]
async fn test_mkv_apv_codec_private_contains_apv1_fourcc() {
    let sink = MemorySource::new_writable(65536);
    let config = MuxerConfig::new();
    let mut muxer = MatroskaMuxer::new(sink, config);

    muxer.add_stream(make_apv_stream()).expect("add APV stream");
    muxer.write_header().await.expect("write header");

    let output = muxer.sink().written_data().to_vec();

    // The BITMAPINFOHEADER's biCompression field must be the 'apv1' fourcc.
    assert!(
        contains_bytes(&output, b"apv1"),
        "MKV output must contain the 'apv1' fourcc inside the CodecPrivate BITMAPINFOHEADER"
    );
}

#[tokio::test]
async fn test_mkv_apv_format_is_matroska() {
    // APV is not a WebM-compatible codec.
    let sink = MemorySource::new_writable(65536);
    let config = MuxerConfig::new();
    let mut muxer = MatroskaMuxer::new(sink, config);

    muxer.add_stream(make_apv_stream()).expect("add APV stream");
    muxer.write_header().await.expect("write header");

    let output = muxer.sink().written_data().to_vec();

    assert!(
        contains_bytes(&output, b"matroska"),
        "DocType must be 'matroska' for an APV stream (not 'webm')"
    );
}

#[tokio::test]
async fn test_mkv_apv_explicit_extradata_takes_precedence() {
    // If the caller provides explicit extradata, it must be used as CodecPrivate
    // instead of the synthesized BITMAPINFOHEADER.
    let sentinel = bytes::Bytes::from_static(b"SENTINEL_PRIVATE_DATA_XYZ");

    let sink = MemorySource::new_writable(65536);
    let config = MuxerConfig::new();
    let mut muxer = MatroskaMuxer::new(sink, config);

    let mut info = StreamInfo::new(0, CodecId::Apv, Rational::new(1, 30));
    info.codec_params = CodecParams::video(320, 240);
    info.codec_params.extradata = Some(sentinel);

    muxer
        .add_stream(info)
        .expect("add APV stream with extradata");
    muxer.write_header().await.expect("write header");

    let output = muxer.sink().written_data().to_vec();

    // The sentinel bytes must appear in the output (as CodecPrivate content).
    assert!(
        contains_bytes(&output, b"SENTINEL_PRIVATE_DATA_XYZ"),
        "Explicit extradata must be used as CodecPrivate when provided"
    );

    // The synthesized BITMAPINFOHEADER 'apv1' fourcc must NOT appear
    // (since extradata took precedence).
    assert!(
        !contains_bytes(&output, b"apv1"),
        "Synthesized BITMAPINFOHEADER must not appear when explicit extradata is supplied"
    );
}
