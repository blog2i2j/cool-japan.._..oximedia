// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Integration test: APV encoder dispatch via `codec_dispatch`.
//!
//! Verifies that `make_video_encoder(CodecId::Apv, &params)` creates a
//! working encoder and that feeding it a synthetic YUV 4:2:0 frame produces
//! a packet that begins with the APV magic bytes (`b"APV1"`).

use oximedia_codec::frame::{Plane, VideoFrame};
use oximedia_core::{CodecId, PixelFormat, Rational, Timestamp};
use oximedia_transcode::{make_video_encoder, VideoEncoderParams};

/// Create a minimal YUV 4:2:0 VideoFrame with all planes set to mid-grey.
fn make_grey_frame(width: u32, height: u32) -> VideoFrame {
    let w = width as usize;
    let h = height as usize;
    let cw = w / 2;
    let ch = h / 2;

    let y_plane = Plane {
        data: vec![128u8; w * h],
        stride: w,
        width,
        height,
    };
    let cb_plane = Plane {
        data: vec![128u8; cw * ch],
        stride: cw,
        width: width / 2,
        height: height / 2,
    };
    let cr_plane = Plane {
        data: vec![128u8; cw * ch],
        stride: cw,
        width: width / 2,
        height: height / 2,
    };

    let mut frame = VideoFrame::new(PixelFormat::Yuv420p, width, height);
    frame.planes = vec![y_plane, cb_plane, cr_plane];
    frame.timestamp = Timestamp::new(0, Rational::new(1, 30));
    frame
}

#[test]
fn test_apv_dispatch_encoder_creates_ok() {
    let params = VideoEncoderParams::new(32, 32, 22).expect("valid params");
    let enc = make_video_encoder(CodecId::Apv, &params);
    assert!(enc.is_ok(), "make_video_encoder(Apv) should succeed");

    let enc = enc.expect("encoder");
    assert_eq!(enc.codec(), CodecId::Apv);
}

#[test]
fn test_apv_dispatch_encode_produces_apv_packet() {
    let params = VideoEncoderParams::new(32, 32, 22).expect("valid params");
    let mut enc = make_video_encoder(CodecId::Apv, &params).expect("encoder");

    let frame = make_grey_frame(32, 32);
    enc.send_frame(&frame).expect("send_frame");

    let pkt = enc
        .receive_packet()
        .expect("receive_packet call succeeded")
        .expect("should have a packet");

    // APV access unit starts with the 4-byte ASCII magic "APV1".
    assert!(
        pkt.data.len() >= 4,
        "packet too small to be an APV AU: {} bytes",
        pkt.data.len()
    );
    assert_eq!(
        &pkt.data[..4],
        b"APV1",
        "APV access unit must start with b\"APV1\" magic"
    );

    assert!(
        pkt.keyframe,
        "all APV frames must be keyframes (intra-only codec)"
    );
}

#[test]
fn test_apv_dispatch_multiple_frames() {
    let params = VideoEncoderParams::new(16, 16, 22).expect("valid params");
    let mut enc = make_video_encoder(CodecId::Apv, &params).expect("encoder");

    let mut packet_count = 0usize;
    for _i in 0..5 {
        let frame = make_grey_frame(16, 16);
        enc.send_frame(&frame).expect("send_frame");
        while let Ok(Some(_pkt)) = enc.receive_packet() {
            packet_count += 1;
        }
    }

    // APV is intra-only: one packet per frame.
    assert_eq!(packet_count, 5, "expected 5 packets for 5 frames");
}

#[test]
fn test_apv_dispatch_qp_boundary() {
    // QP = 0 (best) and QP = 63 (worst) should both succeed.
    for qp in [0u8, 63u8] {
        let params = VideoEncoderParams::new(16, 16, qp).expect("valid params");
        let mut enc = make_video_encoder(CodecId::Apv, &params)
            .unwrap_or_else(|e| panic!("encoder for qp={qp} failed: {e}"));
        let frame = make_grey_frame(16, 16);
        enc.send_frame(&frame).expect("send_frame");
        let pkt = enc
            .receive_packet()
            .expect("receive ok")
            .expect("has packet");
        assert!(!pkt.data.is_empty(), "qp={qp}: empty packet");
        assert_eq!(&pkt.data[..4], b"APV1", "qp={qp}: missing APV1 magic");
    }
}

#[test]
fn test_apv_dispatch_flush() {
    let params = VideoEncoderParams::new(16, 16, 22).expect("valid params");
    let mut enc = make_video_encoder(CodecId::Apv, &params).expect("encoder");

    let frame = make_grey_frame(16, 16);
    enc.send_frame(&frame).expect("send_frame");
    // Do NOT drain — let flush do it.
    enc.flush().expect("flush");
    enc.flush().expect("second flush is idempotent");
}

#[test]
fn test_apv_packet_header_fields() {
    // The APV AU header is 16 bytes after the 4-byte magic:
    //   [0..4]  = b"APV1"
    //   [4]     = profile
    //   [5..7]  = width (u16 LE)
    //   [7..9]  = height (u16 LE)
    //   ...
    let params = VideoEncoderParams::new(32, 16, 22).expect("valid params");
    let mut enc = make_video_encoder(CodecId::Apv, &params).expect("encoder");

    let frame = make_grey_frame(32, 16);
    enc.send_frame(&frame).expect("send_frame");
    let pkt = enc
        .receive_packet()
        .expect("receive ok")
        .expect("has packet");

    // At minimum the 16-byte header must be present.
    assert!(
        pkt.data.len() >= 16,
        "APV AU too short to contain full header"
    );
    assert_eq!(&pkt.data[..4], b"APV1");

    // Width and height in the header (bytes 5-8, big-endian u16 per APV spec).
    let w = u16::from_be_bytes([pkt.data[5], pkt.data[6]]);
    let h = u16::from_be_bytes([pkt.data[7], pkt.data[8]]);
    assert_eq!(w, 32, "APV header width should be 32");
    assert_eq!(h, 16, "APV header height should be 16");
}
