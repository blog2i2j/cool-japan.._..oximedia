//! Integration tests for `oximedia_codec::container::isobmff` primitives.

use oximedia_codec::container::isobmff::{make_box, make_full_box, BoxIter};
use std::io::Cursor;

#[test]
fn make_box_32bit() {
    let payload = b"hello";
    let boxed = make_box(*b"test", payload);
    assert_eq!(boxed.len(), 8 + 5);
    let mut size_buf = [0u8; 4];
    size_buf.copy_from_slice(&boxed[0..4]);
    let size = u32::from_be_bytes(size_buf);
    assert_eq!(size, 13);
    assert_eq!(&boxed[4..8], b"test");
    assert_eq!(&boxed[8..], b"hello");
}

#[test]
fn make_box_empty_payload() {
    let boxed = make_box(*b"free", b"");
    // Total = 8 (just header)
    assert_eq!(boxed.len(), 8);
    let mut size_buf = [0u8; 4];
    size_buf.copy_from_slice(&boxed[0..4]);
    assert_eq!(u32::from_be_bytes(size_buf), 8);
}

#[test]
fn make_full_box_version_and_flags() {
    let boxed = make_full_box(*b"mdhd", 1, 0x00_00_FF, b"payload");
    // 8 (box header) + 1 (version) + 3 (flags) + 7 (payload) = 19
    assert_eq!(boxed.len(), 19);
    // version byte is at offset 8
    assert_eq!(boxed[8], 1u8);
    // 24-bit flags at offsets 9..12
    assert_eq!(&boxed[9..12], &[0x00, 0x00, 0xFF]);
    // payload follows
    assert_eq!(&boxed[12..], b"payload");
}

#[test]
fn box_iter_roundtrip() {
    let box1 = make_box(*b"foo1", b"aaaa");
    let box2 = make_box(*b"foo2", b"bbbb");
    let mut combined = box1;
    combined.extend(box2);

    let parsed: Vec<_> = BoxIter::new(Cursor::new(combined))
        .collect::<Result<Vec<_>, _>>()
        .expect("box parse failed");

    assert_eq!(parsed.len(), 2);
    assert_eq!(&parsed[0].0, b"foo1");
    assert_eq!(&parsed[0].1, b"aaaa");
    assert_eq!(&parsed[1].0, b"foo2");
    assert_eq!(&parsed[1].1, b"bbbb");
}

#[test]
fn box_iter_three_boxes() {
    let mut data = Vec::new();
    data.extend(make_box(*b"box1", b"AAAA"));
    data.extend(make_box(*b"box2", b"BBBB"));
    data.extend(make_box(*b"box3", b"CCCC"));

    let parsed: Vec<_> = BoxIter::new(Cursor::new(data))
        .collect::<Result<Vec<_>, _>>()
        .expect("parse failed");

    assert_eq!(parsed.len(), 3);
    assert_eq!(&parsed[2].0, b"box3");
    assert_eq!(&parsed[2].1, b"CCCC");
}

#[test]
fn box_iter_empty_stream() {
    let parsed: Vec<_> = BoxIter::new(Cursor::new(Vec::<u8>::new()))
        .collect::<Result<Vec<_>, _>>()
        .expect("empty stream ok");
    assert!(parsed.is_empty());
}
