// Copyright 2024 The OxiMedia Project Developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Theora bitstream parsing.
//!
//! This module provides bitstream reading and writing capabilities for
//! Theora video encoding and decoding, following RFC 7845.

use crate::error::{CodecError, CodecResult};
use std::io::{Read, Write};

/// Bitstream reader for Theora decoding.
///
/// Reads bits from a byte buffer in big-endian order.
pub struct BitstreamReader<'a> {
    /// Input data buffer.
    data: &'a [u8],
    /// Current byte position.
    pos: usize,
    /// Current bit position within the byte (0-7).
    bit_pos: u8,
}

impl<'a> BitstreamReader<'a> {
    /// Create a new bitstream reader.
    #[must_use]
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            bit_pos: 0,
        }
    }

    /// Get remaining bits in the stream.
    #[must_use]
    pub fn remaining_bits(&self) -> usize {
        let remaining_bytes = self.data.len().saturating_sub(self.pos);
        remaining_bytes * 8 - usize::from(self.bit_pos)
    }

    /// Check if the reader is byte-aligned.
    #[must_use]
    pub fn is_byte_aligned(&self) -> bool {
        self.bit_pos == 0
    }

    /// Align to the next byte boundary.
    pub fn byte_align(&mut self) {
        if self.bit_pos != 0 {
            self.bit_pos = 0;
            self.pos += 1;
        }
    }

    /// Read a single bit.
    pub fn read_bit(&mut self) -> CodecResult<bool> {
        if self.pos >= self.data.len() {
            return Err(CodecError::InvalidBitstream(
                "Unexpected end of bitstream".to_string(),
            ));
        }

        let byte = self.data[self.pos];
        let bit = (byte >> (7 - self.bit_pos)) & 1;

        self.bit_pos += 1;
        if self.bit_pos == 8 {
            self.bit_pos = 0;
            self.pos += 1;
        }

        Ok(bit != 0)
    }

    /// Read multiple bits as a u32.
    pub fn read_bits(&mut self, n: u8) -> CodecResult<u32> {
        if n > 32 {
            return Err(CodecError::InvalidBitstream(format!(
                "Cannot read more than 32 bits at once, requested {n}"
            )));
        }

        let mut value = 0u32;
        for _ in 0..n {
            value = (value << 1) | u32::from(self.read_bit()?);
        }
        Ok(value)
    }

    /// Read bits as a signed integer.
    pub fn read_signed_bits(&mut self, n: u8) -> CodecResult<i32> {
        let value = self.read_bits(n)?;
        let sign_bit = 1u32 << (n - 1);
        if value & sign_bit != 0 {
            // Negative number in two's complement
            Ok((value | (!0u32 << n)) as i32)
        } else {
            Ok(value as i32)
        }
    }

    /// Read a byte (8 bits).
    pub fn read_byte(&mut self) -> CodecResult<u8> {
        self.read_bits(8).map(|v| v as u8)
    }

    /// Read a 16-bit value.
    pub fn read_u16(&mut self) -> CodecResult<u16> {
        self.read_bits(16).map(|v| v as u16)
    }

    /// Read a 32-bit value.
    pub fn read_u32(&mut self) -> CodecResult<u32> {
        let high = self.read_bits(16)?;
        let low = self.read_bits(16)?;
        Ok((high << 16) | low)
    }

    /// Read bytes directly (must be byte-aligned).
    pub fn read_bytes(&mut self, buf: &mut [u8]) -> CodecResult<()> {
        if !self.is_byte_aligned() {
            return Err(CodecError::InvalidBitstream(
                "Cannot read bytes when not byte-aligned".to_string(),
            ));
        }

        let end = self.pos + buf.len();
        if end > self.data.len() {
            return Err(CodecError::InvalidBitstream(
                "Not enough data for read_bytes".to_string(),
            ));
        }

        buf.copy_from_slice(&self.data[self.pos..end]);
        self.pos = end;
        Ok(())
    }

    /// Skip bits.
    pub fn skip_bits(&mut self, n: usize) -> CodecResult<()> {
        for _ in 0..n {
            self.read_bit()?;
        }
        Ok(())
    }

    /// Peek at the next bits without advancing.
    pub fn peek_bits(&mut self, n: u8) -> CodecResult<u32> {
        let saved_pos = self.pos;
        let saved_bit_pos = self.bit_pos;
        let value = self.read_bits(n)?;
        self.pos = saved_pos;
        self.bit_pos = saved_bit_pos;
        Ok(value)
    }

    /// Get current byte position.
    #[must_use]
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Get current bit position within byte.
    #[must_use]
    pub fn bit_position(&self) -> u8 {
        self.bit_pos
    }
}

/// Bitstream writer for Theora encoding.
///
/// Writes bits to a byte buffer in big-endian order.
pub struct BitstreamWriter {
    /// Output data buffer.
    data: Vec<u8>,
    /// Current bit position within the last byte (0-7).
    bit_pos: u8,
}

impl BitstreamWriter {
    /// Create a new bitstream writer.
    #[must_use]
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            bit_pos: 0,
        }
    }

    /// Create a new bitstream writer with capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
            bit_pos: 0,
        }
    }

    /// Write a single bit.
    pub fn write_bit(&mut self, bit: bool) {
        if self.bit_pos == 0 {
            self.data.push(0);
        }

        if bit {
            let last_idx = self.data.len() - 1;
            self.data[last_idx] |= 1 << (7 - self.bit_pos);
        }

        self.bit_pos += 1;
        if self.bit_pos == 8 {
            self.bit_pos = 0;
        }
    }

    /// Write multiple bits from a u32.
    pub fn write_bits(&mut self, value: u32, n: u8) {
        for i in (0..n).rev() {
            let bit = (value >> i) & 1;
            self.write_bit(bit != 0);
        }
    }

    /// Write signed bits.
    pub fn write_signed_bits(&mut self, value: i32, n: u8) {
        let unsigned = if value < 0 {
            let mask = (1u32 << n) - 1;
            (value as u32) & mask
        } else {
            value as u32
        };
        self.write_bits(unsigned, n);
    }

    /// Write a byte.
    pub fn write_byte(&mut self, byte: u8) {
        self.write_bits(u32::from(byte), 8);
    }

    /// Write a 16-bit value.
    pub fn write_u16(&mut self, value: u16) {
        self.write_bits(u32::from(value), 16);
    }

    /// Write a 32-bit value.
    pub fn write_u32(&mut self, value: u32) {
        self.write_bits(value >> 16, 16);
        self.write_bits(value & 0xFFFF, 16);
    }

    /// Byte-align the stream by padding with zeros.
    pub fn byte_align(&mut self) {
        if self.bit_pos != 0 {
            self.bit_pos = 0;
        }
    }

    /// Get the written data.
    #[must_use]
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Convert to a vector of bytes.
    #[must_use]
    pub fn into_vec(self) -> Vec<u8> {
        self.data
    }

    /// Get the current length in bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if the writer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Check if the writer is byte-aligned.
    #[must_use]
    pub fn is_byte_aligned(&self) -> bool {
        self.bit_pos == 0
    }

    /// Get current bit position.
    #[must_use]
    pub fn bit_position(&self) -> u8 {
        self.bit_pos
    }
}

impl Default for BitstreamWriter {
    fn default() -> Self {
        Self::new()
    }
}

/// Exponential-Golomb coding reader.
pub struct ExpGolombReader<'a> {
    reader: BitstreamReader<'a>,
}

impl<'a> ExpGolombReader<'a> {
    /// Create a new Exp-Golomb reader.
    #[must_use]
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            reader: BitstreamReader::new(data),
        }
    }

    /// Read an unsigned Exp-Golomb code.
    pub fn read_ue(&mut self) -> CodecResult<u32> {
        let mut leading_zeros = 0;
        while !self.reader.read_bit()? {
            leading_zeros += 1;
            if leading_zeros > 31 {
                return Err(CodecError::InvalidBitstream(
                    "Too many leading zeros in Exp-Golomb code".to_string(),
                ));
            }
        }

        if leading_zeros == 0 {
            return Ok(0);
        }

        let value = self.reader.read_bits(leading_zeros)?;
        Ok((1u32 << leading_zeros) - 1 + value)
    }

    /// Read a signed Exp-Golomb code.
    pub fn read_se(&mut self) -> CodecResult<i32> {
        let code = self.read_ue()?;
        if code == 0 {
            return Ok(0);
        }

        let sign = if code & 1 != 0 { 1 } else { -1 };
        Ok(sign * ((code + 1) / 2) as i32)
    }

    /// Get the underlying bitstream reader.
    #[must_use]
    pub fn reader(&mut self) -> &mut BitstreamReader<'a> {
        &mut self.reader
    }
}

/// Exponential-Golomb coding writer.
pub struct ExpGolombWriter {
    writer: BitstreamWriter,
}

impl ExpGolombWriter {
    /// Create a new Exp-Golomb writer.
    #[must_use]
    pub fn new() -> Self {
        Self {
            writer: BitstreamWriter::new(),
        }
    }

    /// Write an unsigned Exp-Golomb code.
    pub fn write_ue(&mut self, value: u32) {
        let code = value + 1;
        let bits = 32 - code.leading_zeros();
        let leading_zeros = bits - 1;

        for _ in 0..leading_zeros {
            self.writer.write_bit(false);
        }
        self.writer.write_bits(code, bits as u8);
    }

    /// Write a signed Exp-Golomb code.
    pub fn write_se(&mut self, value: i32) {
        let code = if value <= 0 {
            ((-value) * 2) as u32
        } else {
            (value * 2 - 1) as u32
        };
        self.write_ue(code);
    }

    /// Get the underlying bitstream writer.
    #[must_use]
    pub fn writer(&mut self) -> &mut BitstreamWriter {
        &mut self.writer
    }

    /// Convert to a vector of bytes.
    #[must_use]
    pub fn into_vec(self) -> Vec<u8> {
        self.writer.into_vec()
    }
}

impl Default for ExpGolombWriter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitstream_reader() {
        let data = [0b1010_1100, 0b1111_0000];
        let mut reader = BitstreamReader::new(&data);

        assert_eq!(reader.read_bit().expect("should succeed"), true);
        assert_eq!(reader.read_bit().expect("should succeed"), false);
        assert_eq!(reader.read_bits(3).expect("should succeed"), 0b101);
        assert_eq!(reader.read_bits(4).expect("should succeed"), 0b1001);
    }

    #[test]
    fn test_bitstream_writer() {
        let mut writer = BitstreamWriter::new();

        writer.write_bit(true);
        writer.write_bit(false);
        writer.write_bits(0b101, 3);
        writer.write_bits(0b1001, 4);
        writer.byte_align();

        let data = writer.data();
        assert_eq!(data[0], 0b1010_1100);
        assert_eq!(data[1], 0b1000_0000);
    }

    #[test]
    fn test_exp_golomb() {
        let mut writer = ExpGolombWriter::new();
        writer.write_ue(0);
        writer.write_ue(1);
        writer.write_ue(5);
        writer.writer().byte_align();

        let data = writer.into_vec();
        let mut reader = ExpGolombReader::new(&data);

        assert_eq!(reader.read_ue().expect("should succeed"), 0);
        assert_eq!(reader.read_ue().expect("should succeed"), 1);
        assert_eq!(reader.read_ue().expect("should succeed"), 5);
    }
}
