// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Integration test: AVI MJPEG muxer/demuxer roundtrip.
//!
//! Verifies that:
//! - `AviMjpegWriter` produces a valid RIFF AVI file.
//! - `AviMjpegReader` extracts the exact frames that were written.

use oximedia_container::demux::avi::AviMjpegReader;
use oximedia_container::mux::avi::AviMjpegWriter;

/// Construct a minimal fake JPEG: SOI + one arbitrary byte + EOI.
fn fake_jpeg(tag: u8) -> Vec<u8> {
    vec![0xFF, 0xD8, tag, 0xFF, 0xD9]
}

#[test]
fn avi_mjpeg_roundtrip_10_frames() {
    let mut writer = AviMjpegWriter::new(64, 48, 30, 1);

    for i in 0..10u8 {
        writer
            .write_frame(fake_jpeg(i))
            .expect("write_frame must not fail");
    }

    let avi_bytes = writer.finish().expect("finish must not fail");

    // Validate outer RIFF AVI signature
    assert!(avi_bytes.len() > 12, "output must be more than 12 bytes");
    assert_eq!(&avi_bytes[0..4], b"RIFF", "must start with RIFF");
    assert_eq!(&avi_bytes[8..12], b"AVI ", "must have AVI  fourcc");

    // Validate that idx1 and 00dc are present
    let has_idx1 = avi_bytes.windows(4).any(|w| w == b"idx1");
    let has_00dc = avi_bytes.windows(4).any(|w| w == b"00dc");
    assert!(has_idx1, "output must contain idx1 chunk");
    assert!(has_00dc, "output must contain 00dc chunks");

    // Validate that movi LIST is present
    let has_movi = avi_bytes.windows(4).any(|w| w == b"movi");
    assert!(has_movi, "output must contain movi list");

    // Roundtrip: demux must return the exact same frames
    let reader = AviMjpegReader::new(avi_bytes).expect("reader construction must succeed");
    let frames = reader.frames().expect("frames() must succeed");

    assert_eq!(frames.len(), 10, "must recover exactly 10 frames");

    for (i, frame) in frames.iter().enumerate() {
        let expected = fake_jpeg(i as u8);
        assert_eq!(
            frame, &expected,
            "frame {i} content mismatch: got {frame:?}, want {expected:?}"
        );
    }
}

#[test]
fn avi_mjpeg_single_frame() {
    let mut writer = AviMjpegWriter::new(320, 240, 25, 1);
    writer.write_frame(fake_jpeg(0xBE)).expect("write_frame");
    let avi_bytes = writer.finish().expect("finish");

    let reader = AviMjpegReader::new(avi_bytes).expect("reader");
    let frames = reader.frames().expect("frames");
    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0], fake_jpeg(0xBE));
}

#[test]
fn avi_mjpeg_zero_frames() {
    let writer = AviMjpegWriter::new(320, 240, 30, 1);
    let avi_bytes = writer.finish().expect("finish");

    // Must still be a valid RIFF AVI
    assert_eq!(&avi_bytes[0..4], b"RIFF");
    assert_eq!(&avi_bytes[8..12], b"AVI ");

    let reader = AviMjpegReader::new(avi_bytes).expect("reader");
    let frames = reader.frames().expect("frames");
    assert_eq!(frames.len(), 0, "zero frames written → zero frames read");
}

#[test]
fn avi_riff_size_field_is_correct() {
    let mut writer = AviMjpegWriter::new(8, 8, 10, 1);
    writer
        .write_frame(vec![0xFF, 0xD8, 0xFF, 0xD9])
        .expect("frame");
    let avi_bytes = writer.finish().expect("finish");

    // RIFF size field at bytes 4..8 should equal total_file_len - 8
    let riff_size =
        u32::from_le_bytes([avi_bytes[4], avi_bytes[5], avi_bytes[6], avi_bytes[7]]) as usize;
    assert_eq!(
        riff_size,
        avi_bytes.len() - 8,
        "RIFF size field must equal file length - 8"
    );
}

#[test]
fn avi_frame_order_preserved() {
    let mut writer = AviMjpegWriter::new(16, 16, 30, 1);
    let data: Vec<Vec<u8>> = (0u8..5)
        .map(|i| vec![0xFF, 0xD8, i * 11, 0xFF, 0xD9])
        .collect();
    for frame in &data {
        writer.write_frame(frame.clone()).expect("write_frame");
    }
    let avi_bytes = writer.finish().expect("finish");

    let reader = AviMjpegReader::new(avi_bytes).expect("reader");
    let recovered = reader.frames().expect("frames");
    assert_eq!(recovered.len(), data.len());
    for (i, (got, want)) in recovered.iter().zip(data.iter()).enumerate() {
        assert_eq!(got, want, "frame {i} ordering mismatch");
    }
}
