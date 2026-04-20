// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Integration test: MJPEG ProxyEncoder round-trip.
//!
//! Verifies that encoding a synthetic YUV 4:2:0 frame through
//! `ProxyEncoder::from_spec` (MJPEG variant) produces at least one
//! `EncodedPacket` whose bytes begin with the JPEG SOI marker (`0xFF 0xD8`)
//! and end with the EOI marker (`0xFF 0xD9`).

use oximedia_codec::frame::{Plane, VideoFrame};
use oximedia_core::{PixelFormat, Rational, Timestamp};
use oximedia_multicam::proxy::{
    ProxyCodec, ProxyConfig, ProxyEncoder, ProxyGenerator, ProxyQuality,
};

/// Build a minimal 8×8 YUV 4:2:0 `VideoFrame` filled with grey values.
fn grey_yuv_frame(width: u32, height: u32) -> VideoFrame {
    let w = width as usize;
    let h = height as usize;

    // Luma plane: Y = 128 (mid-grey)
    let y_data = vec![128u8; w * h];
    let y_plane = Plane {
        data: y_data,
        stride: w,
        width,
        height,
    };

    // Chroma planes: Cb and Cr = 128 (neutral)
    let cw = w / 2;
    let ch = h / 2;
    let cb_data = vec![128u8; cw * ch];
    let cr_data = vec![128u8; cw * ch];
    let cb_plane = Plane {
        data: cb_data,
        stride: cw,
        width: width / 2,
        height: height / 2,
    };
    let cr_plane = Plane {
        data: cr_data,
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
fn test_mjpeg_proxy_spec_generation() {
    let config = ProxyConfig {
        resolution_scale: 0.5,
        codec: ProxyCodec::Mjpeg,
        quality: ProxyQuality::Preview,
        frame_rate_divisor: 1,
        include_audio: false,
    };
    let mut gen = ProxyGenerator::new(config).expect("generator");
    gen.add_angle(0, 320, 240, "Cam-MJPEG").expect("add angle");

    let spec = gen.generate_spec(0).expect("spec");
    assert_eq!(spec.codec, ProxyCodec::Mjpeg);
    assert_eq!(spec.proxy_width, 160);
    assert_eq!(spec.proxy_height, 120);
}

#[test]
fn test_mjpeg_proxy_encoder_creates_ok() {
    let config = ProxyConfig {
        resolution_scale: 1.0,
        codec: ProxyCodec::Mjpeg,
        quality: ProxyQuality::HighQuality,
        frame_rate_divisor: 1,
        include_audio: false,
    };
    let mut gen = ProxyGenerator::new(config).expect("generator");
    gen.add_angle(0, 16, 16, "Cam-Mini").expect("add angle");

    let spec = gen.generate_spec(0).expect("spec");
    let enc = ProxyEncoder::from_spec(&spec);
    assert!(
        enc.is_ok(),
        "ProxyEncoder::from_spec should succeed for MJPEG"
    );

    let enc = enc.expect("encoder");
    assert_eq!(enc.codec_id(), oximedia_core::CodecId::Mjpeg);
}

#[test]
fn test_mjpeg_proxy_encode_roundtrip_produces_jpeg() {
    // Use a small but valid resolution for speed.
    let config = ProxyConfig {
        resolution_scale: 1.0,
        codec: ProxyCodec::Mjpeg,
        quality: ProxyQuality::Preview,
        frame_rate_divisor: 1,
        include_audio: false,
    };
    let mut gen = ProxyGenerator::new(config).expect("generator");
    gen.add_angle(0, 32, 32, "TestCam").expect("add angle");

    let spec = gen.generate_spec(0).expect("spec");
    let mut enc = ProxyEncoder::from_spec(&spec).expect("encoder");

    let frame = grey_yuv_frame(32, 32);
    enc.encode_frame(&frame).expect("encode_frame");

    let packets = enc.drain_packets().expect("drain_packets");
    assert!(!packets.is_empty(), "should produce at least one packet");

    let pkt = &packets[0];
    assert!(pkt.keyframe, "MJPEG frames must always be keyframes");
    assert!(pkt.data.len() >= 4, "packet data too short to be a JPEG");
    // JPEG SOI marker
    assert_eq!(pkt.data[0], 0xFF);
    assert_eq!(pkt.data[1], 0xD8);
    // JPEG EOI marker at end
    let n = pkt.data.len();
    assert_eq!(pkt.data[n - 2], 0xFF);
    assert_eq!(pkt.data[n - 1], 0xD9);
}

#[test]
fn test_mjpeg_proxy_encode_multiple_frames() {
    let config = ProxyConfig {
        resolution_scale: 1.0,
        codec: ProxyCodec::Mjpeg,
        quality: ProxyQuality::Draft,
        frame_rate_divisor: 1,
        include_audio: false,
    };
    let mut gen = ProxyGenerator::new(config).expect("generator");
    gen.add_angle(0, 16, 16, "TestCam").expect("add angle");
    let spec = gen.generate_spec(0).expect("spec");
    let mut enc = ProxyEncoder::from_spec(&spec).expect("encoder");

    let mut total_packets = 0usize;
    for _i in 0..3 {
        let frame = grey_yuv_frame(16, 16);
        enc.encode_frame(&frame).expect("encode_frame");
        let pkts = enc.drain_packets().expect("drain");
        total_packets += pkts.len();
    }

    // All 3 frames should produce packets (MJPEG is intra-only).
    assert_eq!(total_packets, 3, "expected one packet per frame");
}

#[test]
fn test_mjpeg_proxy_flush() {
    let config = ProxyConfig {
        resolution_scale: 1.0,
        codec: ProxyCodec::Mjpeg,
        quality: ProxyQuality::Preview,
        frame_rate_divisor: 1,
        include_audio: false,
    };
    let mut gen = ProxyGenerator::new(config).expect("generator");
    gen.add_angle(0, 16, 16, "TestCam").expect("add angle");
    let spec = gen.generate_spec(0).expect("spec");
    let mut enc = ProxyEncoder::from_spec(&spec).expect("encoder");

    let frame = grey_yuv_frame(16, 16);
    enc.encode_frame(&frame).expect("encode_frame");

    // Flush should not error even if the queue was already drained.
    let packets = enc.flush().expect("flush");
    // Flush may return 0 or 1 packet depending on queue state.
    // Any returned packets must be valid JPEG.
    for pkt in &packets {
        assert_eq!(pkt.data[0], 0xFF, "SOI byte 0");
        assert_eq!(pkt.data[1], 0xD8, "SOI byte 1");
    }
    // We just verify no panic/error; packet count may be 0 or 1.
    let _ = packets;
}

#[test]
fn test_unsupported_codec_returns_error() {
    let config = ProxyConfig {
        resolution_scale: 1.0,
        codec: ProxyCodec::Raw,
        quality: ProxyQuality::Preview,
        frame_rate_divisor: 1,
        include_audio: false,
    };
    let mut gen = ProxyGenerator::new(config).expect("generator");
    gen.add_angle(0, 16, 16, "RawCam").expect("add angle");
    let spec = gen.generate_spec(0).expect("spec");

    let result = ProxyEncoder::from_spec(&spec);
    assert!(
        result.is_err(),
        "Raw codec should not be encodable via ProxyEncoder"
    );
}
