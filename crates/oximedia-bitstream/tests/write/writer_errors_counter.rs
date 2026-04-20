// Copyright 2017 Brian Langenberger
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::limited_writer::LimitedWriter;
use std::io;

#[test]
fn test_writer_io_errors_le() {
    use bitstream_io::{BitWrite, BitWriter, LittleEndian};
    use io::ErrorKind;
    let mut w = BitWriter::endian(LimitedWriter::new(1), LittleEndian);
    assert!(w.write_bit(true).is_ok());
    assert!(w.write_bit(false).is_ok());
    assert!(w.write_bit(false).is_ok());
    assert!(w.write_bit(false).is_ok());
    assert!(w.write_bit(true).is_ok());
    assert!(w.write_bit(true).is_ok());
    assert!(w.write_bit(false).is_ok());
    assert!(w.write_bit(true).is_ok());
    assert!(w.write_bit(true).is_ok());
    assert!(w.write_bit(false).is_ok());
    assert!(w.write_bit(true).is_ok());
    assert!(w.write_bit(true).is_ok());
    assert!(w.write_bit(false).is_ok());
    assert!(w.write_bit(true).is_ok());
    assert!(w.write_bit(true).is_ok());
    assert_eq!(w.write_bit(true).unwrap_err().kind(), ErrorKind::WriteZero);
    let mut w = BitWriter::endian(LimitedWriter::new(1), LittleEndian);
    assert!(w.write_var(2, 1u32).is_ok());
    assert!(w.write_var(3, 4u32).is_ok());
    assert!(w.write_var(5, 13u32).is_ok());
    assert!(w.write_var(3, 3u32).is_ok());
    assert_eq!(
        w.write_var(19, 0x609DFu32).unwrap_err().kind(),
        ErrorKind::WriteZero
    );
    let mut w = BitWriter::endian(LimitedWriter::new(1), LittleEndian);
    assert!(w.write_signed_var(2, 1).is_ok());
    assert!(w.write_signed_var(3, -4).is_ok());
    assert!(w.write_signed_var(5, 13).is_ok());
    assert!(w.write_signed_var(3, 3).is_ok());
    assert_eq!(
        w.write_signed_var(19, -128545).unwrap_err().kind(),
        ErrorKind::WriteZero
    );
    let mut w = BitWriter::endian(LimitedWriter::new(1), LittleEndian);
    assert!(w.write_unary::<0>(1).is_ok());
    assert!(w.write_unary::<0>(0).is_ok());
    assert!(w.write_unary::<0>(0).is_ok());
    assert!(w.write_unary::<0>(2).is_ok());
    assert!(w.write_unary::<0>(2).is_ok());
    assert!(w.write_unary::<0>(2).is_ok());
    assert_eq!(
        w.write_unary::<0>(5).unwrap_err().kind(),
        ErrorKind::WriteZero
    );
    let mut w = BitWriter::endian(LimitedWriter::new(1), LittleEndian);
    assert!(w.write_unary::<1>(0).is_ok());
    assert!(w.write_unary::<1>(3).is_ok());
    assert!(w.write_unary::<1>(0).is_ok());
    assert!(w.write_unary::<1>(1).is_ok());
    assert!(w.write_unary::<1>(0).is_ok());
    assert!(w.write_unary::<1>(1).is_ok());
    assert!(w.write_unary::<1>(0).is_ok());
    assert!(w.write_unary::<1>(1).is_ok());
    assert!(w.write_unary::<1>(0).is_ok());
    assert_eq!(
        w.write_unary::<1>(1).unwrap_err().kind(),
        ErrorKind::WriteZero
    );
    let mut w = BitWriter::endian(LimitedWriter::new(1), LittleEndian);
    assert!(w.write_var::<u16>(9, 0b111111111).is_ok());
    assert_eq!(w.byte_align().unwrap_err().kind(), ErrorKind::WriteZero);
    let mut w = BitWriter::endian(LimitedWriter::new(1), LittleEndian);
    assert_eq!(
        w.write_bytes(b"\xB1\xED").unwrap_err().kind(),
        ErrorKind::WriteZero
    );
    let mut w = BitWriter::endian(LimitedWriter::new(1), LittleEndian);
    assert!(w.write_var(4, 11u8).is_ok());
    assert_eq!(
        w.write_bytes(b"\xB1\xED").unwrap_err().kind(),
        ErrorKind::WriteZero
    );
}

#[test]
fn test_writer_bits_errors() {
    use bitstream_io::{BigEndian, BitWrite, BitWriter, LittleEndian};
    use io::{sink, ErrorKind};
    let mut w = BitWriter::endian(sink(), BigEndian);
    assert_eq!(
        w.write_var(9, 0u8).unwrap_err().kind(),
        ErrorKind::InvalidInput
    );
    assert_eq!(
        w.write_var(17, 0u16).unwrap_err().kind(),
        ErrorKind::InvalidInput
    );
    assert_eq!(
        w.write_var(33, 0u32).unwrap_err().kind(),
        ErrorKind::InvalidInput
    );
    assert_eq!(
        w.write_var(65, 0u64).unwrap_err().kind(),
        ErrorKind::InvalidInput
    );
    assert_eq!(
        w.write_var(1, 0b10u8).unwrap_err().kind(),
        ErrorKind::InvalidInput
    );
    assert_eq!(
        w.write_var(2, 0b100u8).unwrap_err().kind(),
        ErrorKind::InvalidInput
    );
    assert_eq!(
        w.write_var(3, 0b1000u8).unwrap_err().kind(),
        ErrorKind::InvalidInput
    );
    for bits in 1..8 {
        let val = 1u8 << bits;
        assert_eq!(
            w.write_var(bits, val).unwrap_err().kind(),
            ErrorKind::InvalidInput
        );
    }
    for bits in 8..16 {
        let val = 1u16 << bits;
        assert_eq!(
            w.write_var(bits, val).unwrap_err().kind(),
            ErrorKind::InvalidInput
        );
    }
    for bits in 16..32 {
        let val = 1u32 << bits;
        assert_eq!(
            w.write_var(bits, val).unwrap_err().kind(),
            ErrorKind::InvalidInput
        );
    }
    for bits in 32..64 {
        let val = 1u64 << bits;
        assert_eq!(
            w.write_var(bits, val).unwrap_err().kind(),
            ErrorKind::InvalidInput
        );
    }
    assert_eq!(
        w.write_signed_var(9, 0i8).unwrap_err().kind(),
        ErrorKind::InvalidInput
    );
    assert_eq!(
        w.write_signed_var(17, 0i16).unwrap_err().kind(),
        ErrorKind::InvalidInput
    );
    assert_eq!(
        w.write_signed_var(33, 0i32).unwrap_err().kind(),
        ErrorKind::InvalidInput
    );
    assert_eq!(
        w.write_signed_var(65, 0i64).unwrap_err().kind(),
        ErrorKind::InvalidInput
    );
    let mut w = BitWriter::endian(sink(), LittleEndian);
    assert_eq!(
        w.write_var(9, 0u8).unwrap_err().kind(),
        ErrorKind::InvalidInput
    );
    assert_eq!(
        w.write_var(17, 0u16).unwrap_err().kind(),
        ErrorKind::InvalidInput
    );
    assert_eq!(
        w.write_var(33, 0u32).unwrap_err().kind(),
        ErrorKind::InvalidInput
    );
    assert_eq!(
        w.write_var(65, 0u64).unwrap_err().kind(),
        ErrorKind::InvalidInput
    );
    assert_eq!(
        w.write_var(1, 0b10u8).unwrap_err().kind(),
        ErrorKind::InvalidInput
    );
    assert_eq!(
        w.write_var(2, 0b100u8).unwrap_err().kind(),
        ErrorKind::InvalidInput
    );
    assert_eq!(
        w.write_var(3, 0b1000u8).unwrap_err().kind(),
        ErrorKind::InvalidInput
    );
    for bits in 1..8 {
        let val = 1u8 << bits;
        assert_eq!(
            w.write_var(bits, val).unwrap_err().kind(),
            ErrorKind::InvalidInput
        );
    }
    for bits in 8..16 {
        let val = 1u16 << bits;
        assert_eq!(
            w.write_var(bits, val).unwrap_err().kind(),
            ErrorKind::InvalidInput
        );
    }
    for bits in 16..32 {
        let val = 1u32 << bits;
        assert_eq!(
            w.write_var(bits, val).unwrap_err().kind(),
            ErrorKind::InvalidInput
        );
    }
    for bits in 32..64 {
        let val = 1u64 << bits;
        assert_eq!(
            w.write_var(bits, val).unwrap_err().kind(),
            ErrorKind::InvalidInput
        );
    }
    assert_eq!(
        w.write_signed_var(9, 0i8).unwrap_err().kind(),
        ErrorKind::InvalidInput
    );
    assert_eq!(
        w.write_signed_var(17, 0i16).unwrap_err().kind(),
        ErrorKind::InvalidInput
    );
    assert_eq!(
        w.write_signed_var(33, 0i32).unwrap_err().kind(),
        ErrorKind::InvalidInput
    );
    assert_eq!(
        w.write_signed_var(65, 0i64).unwrap_err().kind(),
        ErrorKind::InvalidInput
    );
}

#[test]
fn test_counter_be() {
    use bitstream_io::{BitWrite, BitsWritten};
    let mut w: BitsWritten<u32> = BitsWritten::new();
    w.write_bit(true).unwrap();
    w.write_bit(false).unwrap();
    w.write_bit(true).unwrap();
    w.write_bit(true).unwrap();
    w.write_bit(false).unwrap();
    w.write_bit(false).unwrap();
    w.write_bit(false).unwrap();
    w.write_bit(true).unwrap();
    w.write_bit(true).unwrap();
    w.write_bit(true).unwrap();
    w.write_bit(true).unwrap();
    w.write_bit(false).unwrap();
    w.write_bit(true).unwrap();
    w.write_bit(true).unwrap();
    w.write_bit(false).unwrap();
    w.write_bit(true).unwrap();
    assert_eq!(w.written(), 16);
    let mut w: BitsWritten<u32> = BitsWritten::new();
    assert!(w.byte_aligned());
    w.write_var(2, 2u32).unwrap();
    assert!(!w.byte_aligned());
    w.write_var(3, 6u32).unwrap();
    assert!(!w.byte_aligned());
    w.write_var(5, 7u32).unwrap();
    assert!(!w.byte_aligned());
    w.write_var(3, 5u32).unwrap();
    assert!(!w.byte_aligned());
    w.write_var(19, 0x53BC1u32).unwrap();
    assert!(w.byte_aligned());
    assert_eq!(w.written(), 32);
    let mut w: BitsWritten<u32> = BitsWritten::new();
    w.write_signed_var(2, -2).unwrap();
    w.write_signed_var(3, -2).unwrap();
    w.write_signed_var(5, 7).unwrap();
    w.write_signed_var(3, -3).unwrap();
    w.write_signed_var(19, -181311).unwrap();
    assert_eq!(w.written(), 32);
    let mut w: BitsWritten<u32> = BitsWritten::new();
    w.write_unary::<0>(1).unwrap();
    w.write_unary::<0>(2).unwrap();
    w.write_unary::<0>(0).unwrap();
    w.write_unary::<0>(0).unwrap();
    w.write_unary::<0>(4).unwrap();
    w.write_unary::<0>(2).unwrap();
    w.write_unary::<0>(1).unwrap();
    w.write_unary::<0>(0).unwrap();
    w.write_unary::<0>(3).unwrap();
    w.write_unary::<0>(4).unwrap();
    w.write_unary::<0>(0).unwrap();
    w.write_unary::<0>(0).unwrap();
    w.write_unary::<0>(0).unwrap();
    w.write_unary::<0>(0).unwrap();
    w.write_var(1, 1u32).unwrap();
    assert_eq!(w.written(), 32);
    let mut w: BitsWritten<u32> = BitsWritten::new();
    w.write_unary::<1>(0).unwrap();
    w.write_unary::<1>(1).unwrap();
    w.write_unary::<1>(0).unwrap();
    w.write_unary::<1>(3).unwrap();
    w.write_unary::<1>(0).unwrap();
    w.write_unary::<1>(0).unwrap();
    w.write_unary::<1>(0).unwrap();
    w.write_unary::<1>(1).unwrap();
    w.write_unary::<1>(0).unwrap();
    w.write_unary::<1>(1).unwrap();
    w.write_unary::<1>(2).unwrap();
    w.write_unary::<1>(0).unwrap();
    w.write_unary::<1>(0).unwrap();
    w.write_unary::<1>(1).unwrap();
    w.write_unary::<1>(0).unwrap();
    w.write_unary::<1>(0).unwrap();
    w.write_unary::<1>(0).unwrap();
    w.write_unary::<1>(5).unwrap();
    assert_eq!(w.written(), 32);
    let mut w: BitsWritten<u32> = BitsWritten::new();
    w.write_var(3, 5u32).unwrap();
    w.byte_align().unwrap();
    w.write_var(3, 7u32).unwrap();
    w.byte_align().unwrap();
    w.byte_align().unwrap();
    w.write_var(8, 59u32).unwrap();
    w.byte_align().unwrap();
    w.write_var(4, 12u32).unwrap();
    w.byte_align().unwrap();
    assert_eq!(w.written(), 32);
    let mut w: BitsWritten<u32> = BitsWritten::new();
    w.write_bytes(b"\xB1\xED").unwrap();
    assert_eq!(w.written(), 16);
    let mut w: BitsWritten<u32> = BitsWritten::new();
    w.write_var(4, 11u32).unwrap();
    w.write_bytes(b"\xB1\xED").unwrap();
    w.byte_align().unwrap();
    assert_eq!(w.written(), 24);
}

#[test]
fn test_counter_huffman_be() {
    use bitstream_io::define_huffman_tree;
    use bitstream_io::{BitWrite, BitsWritten};
    define_huffman_tree!(TreeName : i32 = [[[4, 3], 2], [1, 0]]);
    let mut w: BitsWritten<u32> = BitsWritten::new();
    w.write_huffman::<TreeName>(1).unwrap();
    w.write_huffman::<TreeName>(0).unwrap();
    w.write_huffman::<TreeName>(4).unwrap();
    w.write_huffman::<TreeName>(0).unwrap();
    w.write_huffman::<TreeName>(0).unwrap();
    w.write_huffman::<TreeName>(2).unwrap();
    w.write_huffman::<TreeName>(1).unwrap();
    w.write_huffman::<TreeName>(1).unwrap();
    w.write_huffman::<TreeName>(2).unwrap();
    w.write_huffman::<TreeName>(0).unwrap();
    w.write_huffman::<TreeName>(2).unwrap();
    w.write_huffman::<TreeName>(0).unwrap();
    w.write_huffman::<TreeName>(1).unwrap();
    w.write_huffman::<TreeName>(4).unwrap();
    w.write_huffman::<TreeName>(2).unwrap();
    w.byte_align().unwrap();
    assert_eq!(w.written(), 32);
}

#[test]
fn test_counter_le() {
    use bitstream_io::{BitWrite, BitsWritten};
    let mut w: BitsWritten<u32> = BitsWritten::new();
    w.write_bit(true).unwrap();
    w.write_bit(false).unwrap();
    w.write_bit(false).unwrap();
    w.write_bit(false).unwrap();
    w.write_bit(true).unwrap();
    w.write_bit(true).unwrap();
    w.write_bit(false).unwrap();
    w.write_bit(true).unwrap();
    w.write_bit(true).unwrap();
    w.write_bit(false).unwrap();
    w.write_bit(true).unwrap();
    w.write_bit(true).unwrap();
    w.write_bit(false).unwrap();
    w.write_bit(true).unwrap();
    w.write_bit(true).unwrap();
    w.write_bit(true).unwrap();
    assert_eq!(w.written(), 16);
    let mut w: BitsWritten<u32> = BitsWritten::new();
    assert!(w.byte_aligned());
    w.write_var(2, 1u32).unwrap();
    assert!(!w.byte_aligned());
    w.write_var(3, 4u32).unwrap();
    assert!(!w.byte_aligned());
    w.write_var(5, 13u32).unwrap();
    assert!(!w.byte_aligned());
    w.write_var(3, 3u32).unwrap();
    assert!(!w.byte_aligned());
    w.write_var(19, 0x609DFu32).unwrap();
    assert!(w.byte_aligned());
    assert_eq!(w.written(), 32);
    let mut w: BitsWritten<u32> = BitsWritten::new();
    w.write_signed_var(2, 1).unwrap();
    w.write_signed_var(3, -4).unwrap();
    w.write_signed_var(5, 13).unwrap();
    w.write_signed_var(3, 3).unwrap();
    w.write_signed_var(19, -128545).unwrap();
    assert_eq!(w.written(), 32);
    let mut w: BitsWritten<u32> = BitsWritten::new();
    w.write_unary::<0>(1).unwrap();
    w.write_unary::<0>(0).unwrap();
    w.write_unary::<0>(0).unwrap();
    w.write_unary::<0>(2).unwrap();
    w.write_unary::<0>(2).unwrap();
    w.write_unary::<0>(2).unwrap();
    w.write_unary::<0>(5).unwrap();
    w.write_unary::<0>(3).unwrap();
    w.write_unary::<0>(0).unwrap();
    w.write_unary::<0>(1).unwrap();
    w.write_unary::<0>(0).unwrap();
    w.write_unary::<0>(0).unwrap();
    w.write_unary::<0>(0).unwrap();
    w.write_unary::<0>(0).unwrap();
    w.write_var(2, 3u32).unwrap();
    assert_eq!(w.written(), 32);
    let mut w: BitsWritten<u32> = BitsWritten::new();
    w.write_unary::<1>(0).unwrap();
    w.write_unary::<1>(3).unwrap();
    w.write_unary::<1>(0).unwrap();
    w.write_unary::<1>(1).unwrap();
    w.write_unary::<1>(0).unwrap();
    w.write_unary::<1>(1).unwrap();
    w.write_unary::<1>(0).unwrap();
    w.write_unary::<1>(1).unwrap();
    w.write_unary::<1>(0).unwrap();
    w.write_unary::<1>(0).unwrap();
    w.write_unary::<1>(0).unwrap();
    w.write_unary::<1>(0).unwrap();
    w.write_unary::<1>(1).unwrap();
    w.write_unary::<1>(0).unwrap();
    w.write_unary::<1>(0).unwrap();
    w.write_unary::<1>(2).unwrap();
    w.write_unary::<1>(5).unwrap();
    w.write_unary::<1>(0).unwrap();
    assert_eq!(w.written(), 32);
    let mut w: BitsWritten<u32> = BitsWritten::new();
    w.write_var(3, 5u32).unwrap();
    w.byte_align().unwrap();
    w.write_var(3, 7u32).unwrap();
    w.byte_align().unwrap();
    w.byte_align().unwrap();
    w.write_var(8, 59u32).unwrap();
    w.byte_align().unwrap();
    w.write_var(4, 12u32).unwrap();
    w.byte_align().unwrap();
    assert_eq!(w.written(), 32);
    let mut w: BitsWritten<u32> = BitsWritten::new();
    w.write_bytes(b"\xB1\xED").unwrap();
    assert_eq!(w.written(), 16);
    let mut w: BitsWritten<u32> = BitsWritten::new();
    w.write_var(4, 11u32).unwrap();
    w.write_bytes(b"\xB1\xED").unwrap();
    w.byte_align().unwrap();
    assert_eq!(w.written(), 24);
}

#[test]
fn test_counter_huffman_le() {
    use bitstream_io::define_huffman_tree;
    use bitstream_io::{BitWrite, BitsWritten};
    define_huffman_tree!(TreeName : i32 = [[[4, 3], 2], [1, 0]]);
    let mut w: BitsWritten<u32> = BitsWritten::new();
    w.write_huffman::<TreeName>(1).unwrap();
    w.write_huffman::<TreeName>(3).unwrap();
    w.write_huffman::<TreeName>(1).unwrap();
    w.write_huffman::<TreeName>(0).unwrap();
    w.write_huffman::<TreeName>(2).unwrap();
    w.write_huffman::<TreeName>(1).unwrap();
    w.write_huffman::<TreeName>(0).unwrap();
    w.write_huffman::<TreeName>(0).unwrap();
    w.write_huffman::<TreeName>(1).unwrap();
    w.write_huffman::<TreeName>(0).unwrap();
    w.write_huffman::<TreeName>(1).unwrap();
    w.write_huffman::<TreeName>(2).unwrap();
    w.write_huffman::<TreeName>(4).unwrap();
    w.write_huffman::<TreeName>(3).unwrap();
    w.write::<1, u8>(1).unwrap();
    assert_eq!(w.written(), 32);
}

#[test]
fn test_recorder_be() {
    use bitstream_io::{BigEndian, BitRecorder, BitWrite, BitWriter};
    let mut w: BitRecorder<u32, BigEndian> = BitRecorder::new();
    w.write_bit(true).unwrap();
    let mut w2 = BitWriter::endian(vec![], BigEndian);
    w.playback(&mut w2).unwrap();
    w2.byte_align().unwrap();
    let mut w3 = BitWriter::endian(vec![], BigEndian);
    w3.write_bit(true).unwrap();
    w3.byte_align().unwrap();
    assert_eq!(w2.into_writer(), w3.into_writer());
    let final_data: [u8; 4] = [0xB1, 0xED, 0x3B, 0xC1];
    let mut w: BitRecorder<u32, BigEndian> = BitRecorder::new();
    w.write_bit(true).unwrap();
    w.write_bit(false).unwrap();
    w.write_bit(true).unwrap();
    w.write_bit(true).unwrap();
    w.write_bit(false).unwrap();
    w.write_bit(false).unwrap();
    w.write_bit(false).unwrap();
    w.write_bit(true).unwrap();
    w.write_bit(true).unwrap();
    w.write_bit(true).unwrap();
    w.write_bit(true).unwrap();
    w.write_bit(false).unwrap();
    w.write_bit(true).unwrap();
    w.write_bit(true).unwrap();
    w.write_bit(false).unwrap();
    w.write_bit(true).unwrap();
    assert_eq!(w.written(), 16);
    let mut w2 = BitWriter::endian(Vec::with_capacity(2), BigEndian);
    w.playback(&mut w2).unwrap();
    assert_eq!(w2.into_writer().as_slice(), &final_data[0..2]);
    let mut w: BitRecorder<u32, BigEndian> = BitRecorder::new();
    assert!(w.byte_aligned());
    w.write_var(2, 2u32).unwrap();
    assert!(!w.byte_aligned());
    w.write_var(3, 6u32).unwrap();
    assert!(!w.byte_aligned());
    w.write_var(5, 7u32).unwrap();
    assert!(!w.byte_aligned());
    w.write_var(3, 5u32).unwrap();
    assert!(!w.byte_aligned());
    w.write_var(19, 0x53BC1u32).unwrap();
    assert!(w.byte_aligned());
    assert_eq!(w.written(), 32);
    let mut w2 = BitWriter::endian(Vec::with_capacity(4), BigEndian);
    w.playback(&mut w2).unwrap();
    assert_eq!(w2.into_writer().as_slice(), &final_data);
    let mut w: BitRecorder<u32, BigEndian> = BitRecorder::new();
    w.write_signed_var(2, -2).unwrap();
    w.write_signed_var(3, -2).unwrap();
    w.write_signed_var(5, 7).unwrap();
    w.write_signed_var(3, -3).unwrap();
    w.write_signed_var(19, -181311).unwrap();
    assert_eq!(w.written(), 32);
    let mut w2 = BitWriter::endian(Vec::with_capacity(4), BigEndian);
    w.playback(&mut w2).unwrap();
    assert_eq!(w2.into_writer().as_slice(), &final_data);
    let mut w: BitRecorder<u32, BigEndian> = BitRecorder::new();
    w.write_unary::<0>(1).unwrap();
    w.write_unary::<0>(2).unwrap();
    w.write_unary::<0>(0).unwrap();
    w.write_unary::<0>(0).unwrap();
    w.write_unary::<0>(4).unwrap();
    w.write_unary::<0>(2).unwrap();
    w.write_unary::<0>(1).unwrap();
    w.write_unary::<0>(0).unwrap();
    w.write_unary::<0>(3).unwrap();
    w.write_unary::<0>(4).unwrap();
    w.write_unary::<0>(0).unwrap();
    w.write_unary::<0>(0).unwrap();
    w.write_unary::<0>(0).unwrap();
    w.write_unary::<0>(0).unwrap();
    w.write_var(1, 1u32).unwrap();
    assert_eq!(w.written(), 32);
    let mut w2 = BitWriter::endian(Vec::with_capacity(4), BigEndian);
    w.playback(&mut w2).unwrap();
    assert_eq!(w2.into_writer().as_slice(), &final_data);
    let mut w: BitRecorder<u32, BigEndian> = BitRecorder::new();
    w.write_unary::<1>(0).unwrap();
    w.write_unary::<1>(1).unwrap();
    w.write_unary::<1>(0).unwrap();
    w.write_unary::<1>(3).unwrap();
    w.write_unary::<1>(0).unwrap();
    w.write_unary::<1>(0).unwrap();
    w.write_unary::<1>(0).unwrap();
    w.write_unary::<1>(1).unwrap();
    w.write_unary::<1>(0).unwrap();
    w.write_unary::<1>(1).unwrap();
    w.write_unary::<1>(2).unwrap();
    w.write_unary::<1>(0).unwrap();
    w.write_unary::<1>(0).unwrap();
    w.write_unary::<1>(1).unwrap();
    w.write_unary::<1>(0).unwrap();
    w.write_unary::<1>(0).unwrap();
    w.write_unary::<1>(0).unwrap();
    w.write_unary::<1>(5).unwrap();
    assert_eq!(w.written(), 32);
    let mut w2 = BitWriter::endian(Vec::with_capacity(4), BigEndian);
    w.playback(&mut w2).unwrap();
    assert_eq!(w2.into_writer().as_slice(), &final_data);
    let aligned_data = [0xA0, 0xE0, 0x3B, 0xC0];
    let mut w: BitRecorder<u32, BigEndian> = BitRecorder::new();
    w.write_var(3, 5u32).unwrap();
    w.byte_align().unwrap();
    w.write_var(3, 7u32).unwrap();
    w.byte_align().unwrap();
    w.byte_align().unwrap();
    w.write_var(8, 59u32).unwrap();
    w.byte_align().unwrap();
    w.write_var(4, 12u32).unwrap();
    w.byte_align().unwrap();
    assert_eq!(w.written(), 32);
    let mut w2 = BitWriter::endian(Vec::with_capacity(4), BigEndian);
    w.playback(&mut w2).unwrap();
    assert_eq!(w2.into_writer().as_slice(), &aligned_data);
    let final_data = [0xB1, 0xED];
    let mut w: BitRecorder<u32, BigEndian> = BitRecorder::new();
    w.write_bytes(b"\xB1\xED").unwrap();
    assert_eq!(w.written(), 16);
    let mut w2 = BitWriter::endian(Vec::with_capacity(2), BigEndian);
    w.playback(&mut w2).unwrap();
    assert_eq!(w2.into_writer().as_slice(), &final_data);
    let final_data = [0xBB, 0x1E, 0xD0];
    let mut w: BitRecorder<u32, BigEndian> = BitRecorder::new();
    w.write_var(4, 11u32).unwrap();
    w.write_bytes(b"\xB1\xED").unwrap();
    w.byte_align().unwrap();
    let mut w2 = BitWriter::endian(Vec::with_capacity(3), BigEndian);
    w.playback(&mut w2).unwrap();
    assert_eq!(w2.into_writer().as_slice(), &final_data);
}

#[test]
fn test_recorder_huffman_be() {
    use bitstream_io::define_huffman_tree;
    use bitstream_io::{BigEndian, BitRecorder, BitWrite, BitWriter};
    let final_data: [u8; 4] = [0xB1, 0xED, 0x3B, 0xC1];
    define_huffman_tree!(TreeName : i32 = [[[4, 3], 2], [1, 0]]);
    let mut w: BitRecorder<u32, BigEndian> = BitRecorder::new();
    w.write_huffman::<TreeName>(1).unwrap();
    w.write_huffman::<TreeName>(0).unwrap();
    w.write_huffman::<TreeName>(4).unwrap();
    w.write_huffman::<TreeName>(0).unwrap();
    w.write_huffman::<TreeName>(0).unwrap();
    w.write_huffman::<TreeName>(2).unwrap();
    w.write_huffman::<TreeName>(1).unwrap();
    w.write_huffman::<TreeName>(1).unwrap();
    w.write_huffman::<TreeName>(2).unwrap();
    w.write_huffman::<TreeName>(0).unwrap();
    w.write_huffman::<TreeName>(2).unwrap();
    w.write_huffman::<TreeName>(0).unwrap();
    w.write_huffman::<TreeName>(1).unwrap();
    w.write_huffman::<TreeName>(4).unwrap();
    w.write_huffman::<TreeName>(2).unwrap();
    w.byte_align().unwrap();
    let mut w2 = BitWriter::endian(Vec::with_capacity(4), BigEndian);
    w.playback(&mut w2).unwrap();
    assert_eq!(w2.into_writer().as_slice(), &final_data);
}
