// Copyright 2017 Brian Langenberger
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use bitstream_io::{BigEndian, BitWrite, BitWriter};
use std::io;

#[test]
fn test_recorder_huffman_le() {
    use bitstream_io::define_huffman_tree;
    use bitstream_io::{BitRecorder, BitWrite, BitWriter, LittleEndian};
    let final_data: [u8; 4] = [0xB1, 0xED, 0x3B, 0xC1];
    define_huffman_tree!(TreeName : i32 = [[[4, 3], 2], [1, 0]]);
    let mut w: BitRecorder<u32, LittleEndian> = BitRecorder::new();
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
    let mut w2 = BitWriter::endian(Vec::with_capacity(4), LittleEndian);
    w.playback(&mut w2).unwrap();
    assert_eq!(w2.into_writer().as_slice(), &final_data);
}

#[test]
fn test_recorder_le() {
    use bitstream_io::{BitRecorder, BitWrite, BitWriter, LittleEndian};
    let mut w: BitRecorder<u32, LittleEndian> = BitRecorder::new();
    w.write_bit(true).unwrap();
    let mut w2 = BitWriter::endian(vec![], LittleEndian);
    w.playback(&mut w2).unwrap();
    w2.byte_align().unwrap();
    let mut w3 = BitWriter::endian(vec![], LittleEndian);
    w3.write_bit(true).unwrap();
    w3.byte_align().unwrap();
    assert_eq!(w2.into_writer(), w3.into_writer());
    let final_data: [u8; 4] = [0xB1, 0xED, 0x3B, 0xC1];
    let mut w: BitRecorder<u32, LittleEndian> = BitRecorder::new();
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
    let mut w2 = BitWriter::endian(Vec::with_capacity(2), LittleEndian);
    w.playback(&mut w2).unwrap();
    assert_eq!(w2.into_writer().as_slice(), &final_data[0..2]);
    let mut w: BitRecorder<u32, LittleEndian> = BitRecorder::new();
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
    let mut w2 = BitWriter::endian(Vec::with_capacity(4), LittleEndian);
    w.playback(&mut w2).unwrap();
    assert_eq!(w2.into_writer().as_slice(), &final_data);
    let mut w: BitRecorder<u32, LittleEndian> = BitRecorder::new();
    w.write_signed_var(2, 1).unwrap();
    w.write_signed_var(3, -4).unwrap();
    w.write_signed_var(5, 13).unwrap();
    w.write_signed_var(3, 3).unwrap();
    w.write_signed_var(19, -128545).unwrap();
    assert_eq!(w.written(), 32);
    let mut w2 = BitWriter::endian(Vec::with_capacity(4), LittleEndian);
    w.playback(&mut w2).unwrap();
    assert_eq!(w2.into_writer().as_slice(), &final_data);
    let mut w: BitRecorder<u32, LittleEndian> = BitRecorder::new();
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
    let mut w2 = BitWriter::endian(Vec::with_capacity(4), LittleEndian);
    w.playback(&mut w2).unwrap();
    assert_eq!(w2.into_writer().as_slice(), &final_data);
    let mut w: BitRecorder<u32, LittleEndian> = BitRecorder::new();
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
    let mut w2 = BitWriter::endian(Vec::with_capacity(4), LittleEndian);
    w.playback(&mut w2).unwrap();
    assert_eq!(w2.into_writer().as_slice(), &final_data);
    let aligned_data = [0x05, 0x07, 0x3B, 0x0C];
    let mut w: BitRecorder<u32, LittleEndian> = BitRecorder::new();
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
    let mut w2 = BitWriter::endian(Vec::with_capacity(4), LittleEndian);
    w.playback(&mut w2).unwrap();
    assert_eq!(w2.into_writer().as_slice(), &aligned_data);
    let final_data = [0xB1, 0xED];
    let mut w: BitRecorder<u32, LittleEndian> = BitRecorder::new();
    w.write_bytes(b"\xB1\xED").unwrap();
    assert_eq!(w.written(), 16);
    let mut w2 = BitWriter::endian(Vec::with_capacity(2), LittleEndian);
    w.playback(&mut w2).unwrap();
    assert_eq!(w2.into_writer().as_slice(), &final_data);
    let final_data = [0x1B, 0xDB, 0x0E];
    let mut w: BitRecorder<u32, LittleEndian> = BitRecorder::new();
    w.write_var(4, 11u32).unwrap();
    w.write_bytes(b"\xB1\xED").unwrap();
    w.byte_align().unwrap();
    assert_eq!(w.written(), 24);
    let mut w2 = BitWriter::endian(Vec::with_capacity(3), LittleEndian);
    w.playback(&mut w2).unwrap();
    assert_eq!(w2.into_writer().as_slice(), &final_data);
}

#[test]
fn test_pad() {
    use bitstream_io::{BigEndian, Endianness, LittleEndian};
    fn test_pad_endian<E: Endianness>() {
        use bitstream_io::BitWriter;
        let mut plain: BitWriter<_, E> = BitWriter::new(Vec::new());
        let mut padded: BitWriter<_, E> = BitWriter::new(Vec::new());
        for bits in 1..64 {
            plain.write_bit(true).unwrap();
            plain.write_var(bits, 0u64).unwrap();
            padded.write_bit(true).unwrap();
            padded.pad(bits).unwrap();
        }
        plain.byte_align().unwrap();
        padded.byte_align().unwrap();
        assert_eq!(plain.into_writer(), padded.into_writer());
    }
    test_pad_endian::<BigEndian>();
    test_pad_endian::<LittleEndian>();
}

#[test]
fn test_counter_overflow() {
    use bitstream_io::BitsWritten;
    let mut counter: BitsWritten<u8> = BitsWritten::new();
    for _ in 0..255 {
        assert!(counter.write_bit(false).is_ok());
    }
    assert!(counter.write_bit(false).is_err());
    let mut counter: BitsWritten<u8> = BitsWritten::new();
    assert!(counter.write_from([0u8; 31]).is_ok());
    let mut counter: BitsWritten<u8> = BitsWritten::new();
    assert!(counter.write_from([0u8; 32]).is_err());
}

#[test]
fn test_negative_write() {
    let mut bit_writer = BitWriter::endian(Vec::new(), BigEndian);
    assert!(bit_writer.write_bit(false).is_ok());
    assert!(bit_writer.write_var(8, -1i8).is_ok());
    assert!(bit_writer.write_var(7, 0u8).is_ok());
    if let Some(writer) = bit_writer.writer() {
        assert_eq!(writer[0] >> 7, 0);
    } else {
        panic!("writer() returned None");
    }
}

#[test]
fn test_bitcount_write() {
    use bitstream_io::{BigEndian, BitCount, BitWrite, BitWriter};
    let bytes = vec![];
    let mut writer = BitWriter::endian(bytes, BigEndian);
    let count = BitCount::<0b1>::new::<1>();
    writer.write_count(count).unwrap();
    writer.write_counted::<1, u8>(count, 0b1).unwrap();
    writer.byte_align().unwrap();
    assert_eq!(writer.into_writer(), &[0b1_1_000000]);
    let bytes = vec![];
    let mut writer = BitWriter::endian(bytes, BigEndian);
    let count = BitCount::<0b11>::new::<0b11>();
    writer.write_count(count).unwrap();
    writer.write_counted::<0b11, u8>(count, 0b111).unwrap();
    writer.byte_align().unwrap();
    assert_eq!(writer.into_writer(), &[0b11_111_000]);
    let bytes = vec![];
    let mut writer = BitWriter::endian(bytes, BigEndian);
    let count = BitCount::<0b111>::new::<0b111>();
    writer.write_count(count).unwrap();
    writer
        .write_counted::<0b111, u8>(count, 0b11111_11)
        .unwrap();
    writer.byte_align().unwrap();
    assert_eq!(writer.into_writer(), &[0b111_11111, 0b11_000000]);
    let bytes = vec![];
    let mut writer = BitWriter::endian(bytes, BigEndian);
    let count = BitCount::<0b1111>::new::<0b1111>();
    writer.write_count(count).unwrap();
    writer
        .write_counted::<0b1111, u16>(count, 0b1111_11111111_111)
        .unwrap();
    writer.byte_align().unwrap();
    assert_eq!(
        writer.into_writer(),
        &[0b1111_1111, 0b11111111, 0b111_00000]
    );
    let bytes = vec![];
    let mut writer = BitWriter::endian(bytes, BigEndian);
    let count = BitCount::<0b11111>::new::<0b11111>();
    writer.write_count(count).unwrap();
    writer
        .write_counted::<0b11111, u32>(count, 0b111_11111111_11111111_11111111_1111)
        .unwrap();
    writer.byte_align().unwrap();
    assert_eq!(
        writer.into_writer(),
        &[0b11111_111, 0b11111111, 0b11111111, 0b11111111, 0b1111_0000]
    );
    let bytes = vec![];
    let mut writer = BitWriter::endian(bytes, BigEndian);
    let count = BitCount::<0b111111>::new::<0b111111>();
    writer.write_count(count).unwrap();
    writer
        .write_counted::<0b111111, u64>(
            count,
            0b11_11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111,
        )
        .unwrap();
    writer.byte_align().unwrap();
    assert_eq!(
        writer.into_writer(),
        &[
            0b111111_11,
            0b11111111,
            0b11111111,
            0b11111111,
            0b11111111,
            0b11111111,
            0b11111111,
            0b11111111,
            0b11111_000
        ]
    );
    let bytes = vec![];
    let mut writer = BitWriter::endian(bytes, BigEndian);
    let count = BitCount::<0b1111111>::new::<0b1111111>();
    writer.write_count(count).unwrap();
    writer
        .write_counted::<0b1111111, u128>(count, 0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF)
        .unwrap();
    writer.byte_align().unwrap();
    assert_eq!(
        writer.into_writer(),
        &[
            0b1111111_1,
            0b11111111,
            0b11111111,
            0b11111111,
            0b11111111,
            0b11111111,
            0b11111111,
            0b11111111,
            0b11111111,
            0b11111111,
            0b11111111,
            0b11111111,
            0b11111111,
            0b11111111,
            0b11111111,
            0b11111111,
            0b111111_00,
        ]
    );
}

#[test]
fn test_nonzero_writes() {
    use bitstream_io::{BigEndian, BitWrite, BitWriter, LittleEndian};
    use core::num::NonZero;
    let mut w = BitWriter::endian(vec![], BigEndian);
    w.write::<3, u8>(1).unwrap();
    w.byte_align().unwrap();
    assert_eq!(w.into_writer(), &[0b001_00000]);
    let mut w = BitWriter::endian(vec![], BigEndian);
    w.write::<3, NonZero<u8>>(NonZero::new(2).unwrap()).unwrap();
    w.byte_align().unwrap();
    assert_eq!(w.into_writer(), &[0b001_00000]);
    let mut w = BitWriter::endian(vec![], BigEndian);
    w.write_var::<u8>(3, 1).unwrap();
    w.byte_align().unwrap();
    assert_eq!(w.into_writer(), &[0b001_00000]);
    let mut w = BitWriter::endian(vec![], BigEndian);
    w.write_var::<NonZero<u8>>(3, NonZero::new(2).unwrap())
        .unwrap();
    w.byte_align().unwrap();
    assert_eq!(w.into_writer(), &[0b001_00000]);
    let mut w = BitWriter::endian(vec![], LittleEndian);
    w.write::<3, u8>(1).unwrap();
    w.byte_align().unwrap();
    assert_eq!(w.into_writer(), &[0b00000_001]);
    let mut w = BitWriter::endian(vec![], LittleEndian);
    w.write::<3, NonZero<u8>>(NonZero::new(2).unwrap()).unwrap();
    w.byte_align().unwrap();
    assert_eq!(w.into_writer(), &[0b00000_001]);
    let mut w = BitWriter::endian(vec![], LittleEndian);
    w.write_var::<u8>(3, 1).unwrap();
    w.byte_align().unwrap();
    assert_eq!(w.into_writer(), &[0b00000_001]);
    let mut w = BitWriter::endian(vec![], LittleEndian);
    w.write_var::<NonZero<u8>>(3, NonZero::new(2).unwrap())
        .unwrap();
    w.byte_align().unwrap();
    assert_eq!(w.into_writer(), &[0b00000_001]);
}

#[test]
fn test_const_writes() {
    use bitstream_io::{BigEndian, BitWrite, BitWriter, LittleEndian};
    let mut w = BitWriter::endian(vec![], BigEndian);
    w.write_const::<0, 0b0>().unwrap();
    w.write_const::<1, 0b1>().unwrap();
    w.write_const::<2, 0b10>().unwrap();
    w.write_const::<3, 0b100>().unwrap();
    w.write_const::<4, 0b1000>().unwrap();
    w.write_const::<5, 0b10000>().unwrap();
    w.write_const::<6, 0b100000>().unwrap();
    w.write_const::<7, 0b1000000>().unwrap();
    w.write_const::<8, 0b10000000>().unwrap();
    w.write_const::<9, 0b100000000>().unwrap();
    w.write_const::<10, 0b1000000000>().unwrap();
    w.byte_align().unwrap();
    assert_eq!(
        w.into_writer(),
        &[
            0b1_10_100_10,
            0b00_100001,
            0b00000_100,
            0b0000_1000,
            0b0000_1000,
            0b00000_100,
            0b00000000
        ]
    );
    let mut w = BitWriter::endian(vec![], LittleEndian);
    w.write_const::<0, 0b0>().unwrap();
    w.write_const::<1, 0b1>().unwrap();
    w.write_const::<2, 0b10>().unwrap();
    w.write_const::<3, 0b100>().unwrap();
    w.write_const::<4, 0b1000>().unwrap();
    w.write_const::<5, 0b10000>().unwrap();
    w.write_const::<6, 0b100000>().unwrap();
    w.write_const::<7, 0b1000000>().unwrap();
    w.write_const::<8, 0b10000000>().unwrap();
    w.write_const::<9, 0b100000000>().unwrap();
    w.write_const::<10, 0b1000000000>().unwrap();
    w.byte_align().unwrap();
    assert_eq!(
        w.into_writer(),
        &[
            0b00_100_10_1,
            0b0_10000_10,
            0b000_10000,
            0b0000_1000,
            0b0000_1000,
            0b000_10000,
            0b0_1000000
        ]
    );
}

#[test]
fn test_byte_count() {
    use bitstream_io::{ByteWrite, ToByteStream};
    #[derive(Default)]
    struct Builder {
        a: u8,
        b: u16,
        c: u32,
        d: u64,
        e: u128,
    }
    impl ToByteStream for Builder {
        type Error = io::Error;
        fn to_writer<W: ByteWrite + ?Sized>(&self, w: &mut W) -> io::Result<()> {
            w.write(self.a)?;
            w.write(self.b)?;
            w.write(self.c)?;
            w.write(self.d)?;
            w.write(self.e)?;
            Ok(())
        }
    }
    assert_eq!(
        Builder::default().bytes::<u32>().unwrap(),
        1 + 2 + 4 + 8 + 16
    );
}
