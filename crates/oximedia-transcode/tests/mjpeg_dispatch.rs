// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Integration test: MJPEG encoder dispatch via `codec_dispatch`.
//!
//! Verifies that `make_video_encoder(CodecId::Mjpeg, &params)` creates a
//! working encoder and that feeding it a synthetic YUV 4:2:0 frame produces
//! a valid JPEG-SOI packet.

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
fn test_mjpeg_dispatch_encoder_creates_ok() {
    let params = VideoEncoderParams::new(32, 32, 85).expect("valid params");
    let enc = make_video_encoder(CodecId::Mjpeg, &params);
    assert!(enc.is_ok(), "make_video_encoder(Mjpeg) should succeed");

    let enc = enc.expect("encoder");
    assert_eq!(enc.codec(), CodecId::Mjpeg);
}

#[test]
fn test_mjpeg_dispatch_encode_produces_jpeg_packet() {
    let params = VideoEncoderParams::new(32, 32, 85).expect("valid params");
    let mut enc = make_video_encoder(CodecId::Mjpeg, &params).expect("encoder");

    let frame = make_grey_frame(32, 32);
    enc.send_frame(&frame).expect("send_frame");

    let pkt = enc
        .receive_packet()
        .expect("receive_packet call succeeded")
        .expect("should have a packet");

    // JPEG SOI (Start Of Image) marker
    assert!(
        pkt.data.len() >= 4,
        "packet too small to be a JPEG: {} bytes",
        pkt.data.len()
    );
    assert_eq!(pkt.data[0], 0xFF, "byte[0] must be 0xFF (JPEG SOI)");
    assert_eq!(pkt.data[1], 0xD8, "byte[1] must be 0xD8 (JPEG SOI)");

    // JPEG EOI (End Of Image) marker
    let n = pkt.data.len();
    assert_eq!(
        pkt.data[n - 2],
        0xFF,
        "penultimate byte must be 0xFF (JPEG EOI)"
    );
    assert_eq!(pkt.data[n - 1], 0xD9, "last byte must be 0xD9 (JPEG EOI)");

    assert!(pkt.keyframe, "all MJPEG frames must be keyframes");
}

#[test]
fn test_mjpeg_dispatch_multiple_frames() {
    let params = VideoEncoderParams::new(16, 16, 70).expect("valid params");
    let mut enc = make_video_encoder(CodecId::Mjpeg, &params).expect("encoder");

    let mut packet_count = 0usize;
    for _i in 0..5 {
        let frame = make_grey_frame(16, 16);
        enc.send_frame(&frame).expect("send_frame");
        while let Ok(Some(_pkt)) = enc.receive_packet() {
            packet_count += 1;
        }
    }

    // MJPEG is intra-only: one packet per frame.
    assert_eq!(packet_count, 5, "expected 5 packets for 5 frames");
}

#[test]
fn test_mjpeg_dispatch_quality_boundary() {
    // Quality = 1 (lowest) and quality = 100 (highest) should both succeed.
    for quality in [1u8, 100u8] {
        let params = VideoEncoderParams::new(16, 16, quality).expect("valid params");
        let mut enc = make_video_encoder(CodecId::Mjpeg, &params)
            .unwrap_or_else(|e| panic!("encoder for quality={quality} failed: {e}"));
        let frame = make_grey_frame(16, 16);
        enc.send_frame(&frame).expect("send_frame");
        let pkt = enc
            .receive_packet()
            .expect("receive ok")
            .expect("has packet");
        assert!(!pkt.data.is_empty(), "quality={quality}: empty packet");
    }
}

#[test]
fn test_mjpeg_dispatch_flush() {
    let params = VideoEncoderParams::new(16, 16, 85).expect("valid params");
    let mut enc = make_video_encoder(CodecId::Mjpeg, &params).expect("encoder");

    let frame = make_grey_frame(16, 16);
    enc.send_frame(&frame).expect("send_frame");
    // Do NOT call receive_packet — let flush drain everything.
    enc.flush().expect("flush");
    // Second flush should also be idempotent.
    enc.flush().expect("second flush");
}
