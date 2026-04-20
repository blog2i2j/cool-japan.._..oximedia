//! Integration tests for `JxlStreamingDecoder` — iterator semantics.
//!
//! These tests exercise the streaming API specifically: lazy `next()` calls,
//! frame-count fidelity, and frame pixel-data round-tripping.  Format
//! detection is also verified by feeding both ISOBMFF and native-format
//! streams to the same decoder.

use oximedia_codec::jpegxl::{AnimatedJxlEncoder, JxlAnimation, JxlStreamingDecoder};
use std::io::Cursor;

/// Generate a deterministic test frame matching the pattern in the encoder tests.
fn make_test_frame(width: u32, height: u32, channels: u8, seed: u8) -> Vec<u8> {
    let pixel_count = (width * height) as usize;
    let mut data = Vec::with_capacity(pixel_count * channels as usize);
    for i in 0..pixel_count {
        for ch in 0..channels as usize {
            let val = ((i.wrapping_mul(3 + ch) + seed as usize * 37 + ch * 50) % 256) as u8;
            data.push(val);
        }
    }
    data
}

// ── ISOBMFF format tests ───────────────────────────────────────────────────────

#[test]
fn streaming_decode_isobmff_three_frames_yield_count() {
    let anim = JxlAnimation::millisecond();
    let width = 4u32;
    let height = 4u32;
    let channels = 3u8;

    let mut encoder =
        AnimatedJxlEncoder::new(anim, width, height, channels, 8).expect("encoder ok");
    encoder
        .add_frame(&make_test_frame(width, height, channels, 0), 100)
        .expect("f0");
    encoder
        .add_frame(&make_test_frame(width, height, channels, 1), 200)
        .expect("f1");
    encoder
        .add_frame(&make_test_frame(width, height, channels, 2), 150)
        .expect("f2");

    let bytes = encoder.finish_isobmff().expect("finish_isobmff");

    let decoder = JxlStreamingDecoder::new(Cursor::new(bytes)).expect("new ok");
    let frames: Vec<_> = decoder.collect::<Result<Vec<_>, _>>().expect("collect ok");

    assert_eq!(frames.len(), 3, "must yield exactly 3 frames");
}

#[test]
fn streaming_decode_isobmff_pixel_data_roundtrip() {
    let anim = JxlAnimation::millisecond();
    let width = 4u32;
    let height = 4u32;
    let channels = 3u8;

    let f0 = make_test_frame(width, height, channels, 10);
    let f1 = make_test_frame(width, height, channels, 20);
    let f2 = make_test_frame(width, height, channels, 30);

    let mut encoder =
        AnimatedJxlEncoder::new(anim, width, height, channels, 8).expect("encoder ok");
    encoder.add_frame(&f0, 100).expect("f0");
    encoder.add_frame(&f1, 200).expect("f1");
    encoder.add_frame(&f2, 150).expect("f2");

    let bytes = encoder.finish_isobmff().expect("finish_isobmff");

    let decoder = JxlStreamingDecoder::new(Cursor::new(bytes)).expect("new ok");
    let frames: Vec<_> = decoder.collect::<Result<Vec<_>, _>>().expect("collect ok");

    assert_eq!(frames.len(), 3);
    assert_eq!(frames[0].data, f0, "frame 0 pixel mismatch");
    assert_eq!(frames[1].data, f1, "frame 1 pixel mismatch");
    assert_eq!(frames[2].data, f2, "frame 2 pixel mismatch");
}

#[test]
fn streaming_decode_isobmff_single_frame() {
    let anim = JxlAnimation::millisecond().with_num_loops(1);
    let width = 2u32;
    let height = 2u32;
    let channels = 3u8;

    let frame_data = make_test_frame(width, height, channels, 42);

    let mut encoder =
        AnimatedJxlEncoder::new(anim, width, height, channels, 8).expect("encoder ok");
    encoder.add_frame(&frame_data, 500).expect("frame ok");

    let bytes = encoder.finish_isobmff().expect("finish_isobmff");

    let decoder = JxlStreamingDecoder::new(Cursor::new(bytes)).expect("new ok");
    let frames: Vec<_> = decoder.collect::<Result<Vec<_>, _>>().expect("collect ok");

    assert_eq!(
        frames.len(),
        1,
        "single-frame ISOBMFF must yield exactly 1 frame"
    );
    assert_eq!(frames[0].data, frame_data, "pixel data roundtrip failed");
    assert_eq!(frames[0].width, width);
    assert_eq!(frames[0].height, height);
    assert_eq!(frames[0].channels, channels);
}

#[test]
fn streaming_decode_isobmff_iterator_terminates_with_none() {
    let anim = JxlAnimation::millisecond();
    let width = 2u32;
    let height = 2u32;
    let channels = 3u8;

    let mut encoder =
        AnimatedJxlEncoder::new(anim, width, height, channels, 8).expect("encoder ok");
    encoder
        .add_frame(&make_test_frame(width, height, channels, 1), 100)
        .expect("frame ok");

    let bytes = encoder.finish_isobmff().expect("finish_isobmff");

    let mut decoder = JxlStreamingDecoder::new(Cursor::new(bytes)).expect("new ok");

    // Exhaust all frames.
    while decoder.next().is_some() {}

    // Subsequent calls must always return None.
    assert!(
        decoder.next().is_none(),
        "must return None after exhaustion"
    );
    assert!(
        decoder.next().is_none(),
        "must return None on repeated calls"
    );
}

// ── Native format tests ────────────────────────────────────────────────────────

#[test]
fn streaming_decode_native_format_three_frames() {
    let anim = JxlAnimation::millisecond();
    let width = 4u32;
    let height = 4u32;
    let channels = 3u8;

    let f0 = make_test_frame(width, height, channels, 5);
    let f1 = make_test_frame(width, height, channels, 6);
    let f2 = make_test_frame(width, height, channels, 7);

    let mut encoder =
        AnimatedJxlEncoder::new(anim, width, height, channels, 8).expect("encoder ok");
    encoder.add_frame(&f0, 100).expect("f0");
    encoder.add_frame(&f1, 200).expect("f1");
    encoder.add_frame(&f2, 150).expect("f2");

    // Use finish() (native format) instead of finish_isobmff().
    let bytes = encoder.finish().expect("finish");

    let decoder = JxlStreamingDecoder::new(Cursor::new(bytes)).expect("new ok");
    let frames: Vec<_> = decoder.collect::<Result<Vec<_>, _>>().expect("collect ok");

    assert_eq!(frames.len(), 3, "native format must yield 3 frames");
    assert_eq!(frames[0].data, f0, "frame 0 mismatch");
    assert_eq!(frames[1].data, f1, "frame 1 mismatch");
    assert_eq!(frames[2].data, f2, "frame 2 mismatch");
}

#[test]
fn streaming_decode_native_format_single_frame() {
    let anim = JxlAnimation::millisecond();
    let width = 2u32;
    let height = 2u32;
    let channels = 1u8;

    let frame_data = vec![10u8, 20, 30, 40];

    let mut encoder =
        AnimatedJxlEncoder::new(anim, width, height, channels, 8).expect("encoder ok");
    encoder.add_frame(&frame_data, 200).expect("frame ok");

    let bytes = encoder.finish().expect("finish");

    let decoder = JxlStreamingDecoder::new(Cursor::new(bytes)).expect("new ok");
    let frames: Vec<_> = decoder.collect::<Result<Vec<_>, _>>().expect("collect ok");

    assert!(!frames.is_empty(), "must yield at least one frame");
    assert_eq!(frames[0].data, frame_data, "pixel data mismatch");
}

#[test]
fn streaming_decode_native_iterator_terminates_with_none() {
    let anim = JxlAnimation::millisecond();
    let width = 2u32;
    let height = 2u32;
    let channels = 3u8;

    let mut encoder =
        AnimatedJxlEncoder::new(anim, width, height, channels, 8).expect("encoder ok");
    encoder
        .add_frame(&make_test_frame(width, height, channels, 3), 100)
        .expect("frame ok");

    let bytes = encoder.finish().expect("finish");

    let mut decoder = JxlStreamingDecoder::new(Cursor::new(bytes)).expect("new ok");

    while decoder.next().is_some() {}

    assert!(
        decoder.next().is_none(),
        "must return None after exhaustion"
    );
}

// ── Frame metadata tests ───────────────────────────────────────────────────────

#[test]
fn streaming_decode_isobmff_duration_ticks_preserved() {
    let anim = JxlAnimation::millisecond();
    let width = 2u32;
    let height = 2u32;
    let channels = 3u8;

    let mut encoder =
        AnimatedJxlEncoder::new(anim, width, height, channels, 8).expect("encoder ok");
    encoder
        .add_frame(&make_test_frame(width, height, channels, 0), 333)
        .expect("f0");
    encoder
        .add_frame(&make_test_frame(width, height, channels, 1), 666)
        .expect("f1");

    let bytes = encoder.finish_isobmff().expect("finish_isobmff");

    let decoder = JxlStreamingDecoder::new(Cursor::new(bytes)).expect("new ok");
    let frames: Vec<_> = decoder.collect::<Result<Vec<_>, _>>().expect("collect ok");

    assert_eq!(frames.len(), 2);
    assert_eq!(frames[0].duration_ticks, 333, "frame 0 duration mismatch");
    assert_eq!(frames[1].duration_ticks, 666, "frame 1 duration mismatch");
    // Last frame must have is_last = true.
    assert!(frames[1].is_last, "last frame must have is_last=true");
}
