//! Integration tests for `AnimatedJxlEncoder::finish_isobmff()`.

use oximedia_codec::container::isobmff::BoxIter;
use oximedia_codec::jpegxl::{AnimatedJxlEncoder, JxlAnimation};
use std::io::Cursor;

/// Generate a simple deterministic test frame.
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

#[test]
fn jxl_isobmff_box_structure_two_frames() {
    let anim = JxlAnimation::millisecond();
    let width = 4u32;
    let height = 4u32;
    let channels = 3u8;

    let mut encoder =
        AnimatedJxlEncoder::new(anim, width, height, channels, 8).expect("encoder ok");

    encoder
        .add_frame(&make_test_frame(width, height, channels, 0), 100)
        .expect("frame 0 ok");
    encoder
        .add_frame(&make_test_frame(width, height, channels, 1), 200)
        .expect("frame 1 ok");

    let bytes = encoder.finish_isobmff().expect("finish_isobmff failed");

    // Parse all top-level boxes.
    let boxes: Vec<([u8; 4], Vec<u8>)> = BoxIter::new(Cursor::new(&bytes))
        .collect::<Result<Vec<_>, _>>()
        .expect("box parse failed");

    // Must have at least ftyp + jxll + jxlp.
    assert!(
        boxes.len() >= 3,
        "expected at least 3 boxes, got {}",
        boxes.len()
    );

    // First box must be ftyp.
    assert_eq!(&boxes[0].0, b"ftyp", "first box must be ftyp");

    // ftyp payload: major_brand + minor_version + compatible_brands.
    let ftyp_payload = &boxes[0].1;
    assert!(
        ftyp_payload.len() >= 16,
        "ftyp payload too short: {}",
        ftyp_payload.len()
    );
    assert_eq!(&ftyp_payload[0..4], b"jxl ", "major brand must be 'jxl '");
    // minor_version = 0
    assert_eq!(
        &ftyp_payload[4..8],
        &[0, 0, 0, 0],
        "minor version must be 0"
    );
    // compatible brand 1
    assert_eq!(&ftyp_payload[8..12], b"jxl ", "first compat brand");
    // compatible brand 2
    assert_eq!(&ftyp_payload[12..16], b"isom", "second compat brand");

    // Must contain a jxll box.
    let jxll_box = boxes.iter().find(|(cc, _)| cc == b"jxll");
    assert!(jxll_box.is_some(), "must have a jxll box");
    let jxll_payload = &jxll_box.expect("just checked").1;
    assert!(!jxll_payload.is_empty(), "jxll payload must be non-empty");
    assert_eq!(jxll_payload[0], 5u8, "jxll level must be 5 (animated)");

    // Must contain at least one jxlp box.
    let jxlp_boxes: Vec<_> = boxes.iter().filter(|(cc, _)| cc == b"jxlp").collect();
    assert!(!jxlp_boxes.is_empty(), "must have at least one jxlp box");

    // The last jxlp must have the high bit set in its 4-byte index field.
    let last_jxlp = jxlp_boxes.last().expect("at least one jxlp");
    assert!(
        last_jxlp.1.len() >= 4,
        "jxlp payload must be at least 4 bytes"
    );
    let mut idx_buf = [0u8; 4];
    idx_buf.copy_from_slice(&last_jxlp.1[0..4]);
    let idx = u32::from_be_bytes(idx_buf);
    assert!(
        idx & 0x8000_0000 != 0,
        "last jxlp index must have the last-box flag (bit 31) set, got 0x{idx:08X}"
    );
}

#[test]
fn jxl_isobmff_single_frame() {
    let anim = JxlAnimation::millisecond().with_num_loops(1);
    let width = 2u32;
    let height = 2u32;
    let channels = 3u8;

    let mut encoder =
        AnimatedJxlEncoder::new(anim, width, height, channels, 8).expect("encoder ok");
    encoder
        .add_frame(&make_test_frame(width, height, channels, 42), 500)
        .expect("frame ok");

    let bytes = encoder.finish_isobmff().expect("finish_isobmff failed");

    // Sanity: output is non-empty and starts with the ftyp size field.
    assert!(bytes.len() > 16, "output too short");

    let boxes: Vec<_> = BoxIter::new(Cursor::new(&bytes))
        .collect::<Result<Vec<_>, _>>()
        .expect("parse ok");

    assert_eq!(&boxes[0].0, b"ftyp");

    let jxlp_count = boxes.iter().filter(|(cc, _)| cc == b"jxlp").count();
    assert!(jxlp_count >= 1, "must have at least one jxlp");
}

#[test]
fn jxl_isobmff_jxlp_contains_jxl_signature() {
    // The bare codestream wrapped in jxlp must begin with the JXL signature
    // 0xFF 0x0A (after the 4-byte index field).
    let anim = JxlAnimation::millisecond();
    let width = 4u32;
    let height = 4u32;
    let channels = 3u8;

    let mut encoder =
        AnimatedJxlEncoder::new(anim, width, height, channels, 8).expect("encoder ok");
    encoder
        .add_frame(&make_test_frame(width, height, channels, 7), 100)
        .expect("frame ok");

    let bytes = encoder.finish_isobmff().expect("finish ok");

    let boxes: Vec<_> = BoxIter::new(Cursor::new(&bytes))
        .collect::<Result<Vec<_>, _>>()
        .expect("parse ok");

    let jxlp = boxes
        .iter()
        .find(|(cc, _)| cc == b"jxlp")
        .expect("jxlp missing");

    // Payload layout: [4 bytes index] [codestream...]
    assert!(jxlp.1.len() >= 6, "jxlp payload too short");
    // JXL bare codestream signature: 0xFF 0x0A
    assert_eq!(
        jxlp.1[4], 0xFF,
        "codestream byte 0 must be 0xFF (JXL signature)"
    );
    assert_eq!(
        jxlp.1[5], 0x0A,
        "codestream byte 1 must be 0x0A (JXL signature)"
    );
}
