//! Bit-level reader and writer for JPEG-XL bitstream processing.
//!
//! JPEG-XL uses variable-length bit fields extensively. This module provides
//! efficient bit-granularity access to byte buffers in both reading and writing
//! directions.

use crate::error::{CodecError, CodecResult};

/// Bit-level reader for parsing JPEG-XL bitstreams.
///
/// Reads bits from a byte buffer in LSB-first order, which is the convention
/// used by JPEG-XL codestreams (and ANS entropy coding).
pub struct BitReader<'a> {
    data: &'a [u8],
    /// Current byte position in the data buffer.
    byte_pos: usize,
    /// Current bit position within the current byte (0-7, 0 = LSB).
    bit_pos: u8,
}

impl<'a> BitReader<'a> {
    /// Create a new bit reader over a byte slice.
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_pos: 0,
            bit_pos: 0,
        }
    }

    /// Read up to 32 bits from the stream (LSB first).
    ///
    /// # Errors
    ///
    /// Returns `CodecError::InvalidBitstream` if not enough bits remain.
    pub fn read_bits(&mut self, n: u8) -> CodecResult<u32> {
        if n == 0 {
            return Ok(0);
        }
        if n > 32 {
            return Err(CodecError::InvalidBitstream(
                "Cannot read more than 32 bits at once".into(),
            ));
        }
        if self.remaining_bits() < n as usize {
            return Err(CodecError::InvalidBitstream(
                "Not enough bits remaining in stream".into(),
            ));
        }

        let mut result: u32 = 0;
        let mut bits_read: u8 = 0;

        while bits_read < n {
            let bits_available_in_byte = 8 - self.bit_pos;
            let bits_needed = n - bits_read;
            let bits_to_read = bits_available_in_byte.min(bits_needed);

            let byte_val = self.data[self.byte_pos] as u32;
            let mask = (1u32 << bits_to_read) - 1;
            let extracted = (byte_val >> self.bit_pos) & mask;

            result |= extracted << bits_read;
            bits_read += bits_to_read;

            self.bit_pos += bits_to_read;
            if self.bit_pos >= 8 {
                self.bit_pos = 0;
                self.byte_pos += 1;
            }
        }

        Ok(result)
    }

    /// Read a single boolean bit.
    pub fn read_bool(&mut self) -> CodecResult<bool> {
        Ok(self.read_bits(1)? != 0)
    }

    /// Read up to 8 bits as a u8.
    pub fn read_u8(&mut self, n: u8) -> CodecResult<u8> {
        if n > 8 {
            return Err(CodecError::InvalidBitstream(
                "Cannot read more than 8 bits into u8".into(),
            ));
        }
        self.read_bits(n).map(|v| v as u8)
    }

    /// Read up to 16 bits as a u16.
    pub fn read_u16(&mut self, n: u8) -> CodecResult<u16> {
        if n > 16 {
            return Err(CodecError::InvalidBitstream(
                "Cannot read more than 16 bits into u16".into(),
            ));
        }
        self.read_bits(n).map(|v| v as u16)
    }

    /// Read up to 32 bits as a u32.
    pub fn read_u32(&mut self, n: u8) -> CodecResult<u32> {
        self.read_bits(n)
    }

    /// Read a u64 value using JPEG-XL's U64 encoding.
    ///
    /// The JXL U64 encoding uses a selector to determine how many bits follow:
    /// - 0: value = 0
    /// - 1: value = 1 + read(4)
    /// - 2: value = 17 + read(8)
    /// - 3: value = read(12) + (read variable chunks until done)
    pub fn read_u64(&mut self) -> CodecResult<u64> {
        let selector = self.read_bits(2)?;
        match selector {
            0 => Ok(0),
            1 => {
                let extra = self.read_bits(4)? as u64;
                Ok(1 + extra)
            }
            2 => {
                let extra = self.read_bits(8)? as u64;
                Ok(17 + extra)
            }
            3 => {
                // Variable-length: read 12 bits, then optionally more in 8-bit chunks
                let mut value = self.read_bits(12)? as u64;
                let mut shift = 12u32;
                while shift < 60 {
                    let more = self.read_bool()?;
                    if more {
                        let chunk = self.read_bits(8)? as u64;
                        value |= chunk << shift;
                        shift += 8;
                    } else {
                        break;
                    }
                }
                // Final chunk if we reached 60 bits
                if shift >= 60 {
                    let chunk = self.read_bits(4)? as u64;
                    value |= chunk << shift;
                }
                Ok(273 + value)
            }
            _ => Err(CodecError::InvalidBitstream("Invalid U64 selector".into())),
        }
    }

    /// Number of bits remaining in the stream.
    pub fn remaining_bits(&self) -> usize {
        if self.byte_pos >= self.data.len() {
            return 0;
        }
        (self.data.len() - self.byte_pos) * 8 - self.bit_pos as usize
    }

    /// Align the reader to the next byte boundary.
    ///
    /// If already at a byte boundary, this is a no-op.
    pub fn align_to_byte(&mut self) {
        if self.bit_pos != 0 {
            self.bit_pos = 0;
            self.byte_pos += 1;
        }
    }

    /// Get current byte position.
    pub fn byte_position(&self) -> usize {
        self.byte_pos
    }

    /// Check if the reader has been exhausted.
    pub fn is_empty(&self) -> bool {
        self.remaining_bits() == 0
    }
}

/// Bit-level writer for constructing JPEG-XL bitstreams.
///
/// Writes bits in LSB-first order, accumulating a byte buffer.
pub struct BitWriter {
    data: Vec<u8>,
    current_byte: u8,
    bit_pos: u8,
}

impl BitWriter {
    /// Create a new empty bit writer.
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            current_byte: 0,
            bit_pos: 0,
        }
    }

    /// Create a new bit writer with pre-allocated capacity (in bytes).
    pub fn with_capacity(bytes: usize) -> Self {
        Self {
            data: Vec::with_capacity(bytes),
            current_byte: 0,
            bit_pos: 0,
        }
    }

    /// Write up to 32 bits (LSB first).
    pub fn write_bits(&mut self, value: u32, n: u8) {
        if n == 0 {
            return;
        }
        let mut remaining = n;
        let mut val = value;
        let mut written: u8 = 0;

        while written < n {
            let space_in_byte = 8 - self.bit_pos;
            let bits_to_write = space_in_byte.min(remaining);
            let mask = (1u32 << bits_to_write) - 1;
            let bits = (val & mask) as u8;

            self.current_byte |= bits << self.bit_pos;
            self.bit_pos += bits_to_write;
            val >>= bits_to_write;
            written += bits_to_write;
            remaining -= bits_to_write;

            if self.bit_pos >= 8 {
                self.data.push(self.current_byte);
                self.current_byte = 0;
                self.bit_pos = 0;
            }
        }
    }

    /// Write a single boolean bit.
    pub fn write_bool(&mut self, v: bool) {
        self.write_bits(u32::from(v), 1);
    }

    /// Write a u64 value using JPEG-XL's U64 encoding.
    pub fn write_u64(&mut self, value: u64) {
        if value == 0 {
            self.write_bits(0, 2); // selector 0
        } else if value <= 16 {
            self.write_bits(1, 2); // selector 1
            self.write_bits((value - 1) as u32, 4);
        } else if value <= 272 {
            self.write_bits(2, 2); // selector 2
            self.write_bits((value - 17) as u32, 8);
        } else {
            self.write_bits(3, 2); // selector 3
            let mut remaining = value - 273;
            // Write first 12 bits
            self.write_bits((remaining & 0xFFF) as u32, 12);
            remaining >>= 12;
            let mut shift = 12u32;
            while shift < 60 && remaining > 0 {
                self.write_bool(true); // more chunks
                self.write_bits((remaining & 0xFF) as u32, 8);
                remaining >>= 8;
                shift += 8;
            }
            if shift < 60 {
                self.write_bool(false); // no more chunks
            } else if shift >= 60 {
                // Final 4 bits
                self.write_bits((remaining & 0xF) as u32, 4);
            }
        }
    }

    /// Align to the next byte boundary by writing zero bits.
    pub fn align_to_byte(&mut self) {
        if self.bit_pos != 0 {
            self.data.push(self.current_byte);
            self.current_byte = 0;
            self.bit_pos = 0;
        }
    }

    /// Consume the writer and return the accumulated byte buffer.
    ///
    /// Any partial byte is flushed with zero-padding.
    pub fn finish(mut self) -> Vec<u8> {
        self.align_to_byte();
        self.data
    }

    /// Current number of bytes written (excluding partial byte).
    pub fn bytes_written(&self) -> usize {
        self.data.len()
    }

    /// Current total bits written.
    pub fn bits_written(&self) -> usize {
        self.data.len() * 8 + self.bit_pos as usize
    }
}

impl Default for BitWriter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn test_bitreader_basic() {
        let data = [0b1010_0110u8, 0b1100_0011];
        let mut reader = BitReader::new(&data);

        // Read 4 bits from first byte (LSB first): 0110 -> 6
        assert_eq!(reader.read_bits(4).expect("ok"), 0b0110);
        // Read 4 more bits: 1010 -> 10
        assert_eq!(reader.read_bits(4).expect("ok"), 0b1010);
        // Read 8 bits from second byte: 0b1100_0011 -> 195
        assert_eq!(reader.read_bits(8).expect("ok"), 0b1100_0011);
    }

    #[test]
    #[ignore]
    fn test_bitreader_cross_byte() {
        let data = [0xFF, 0x00];
        let mut reader = BitReader::new(&data);

        // Read 4 bits: 1111 = 15
        assert_eq!(reader.read_bits(4).expect("ok"), 0xF);
        // Read 8 bits crossing byte boundary: 0000_1111 in LSB order
        assert_eq!(reader.read_bits(8).expect("ok"), 0x0F);
    }

    #[test]
    #[ignore]
    fn test_bitreader_bool() {
        let data = [0b0000_0101];
        let mut reader = BitReader::new(&data);

        assert!(reader.read_bool().expect("ok")); // bit 0 = 1
        assert!(!reader.read_bool().expect("ok")); // bit 1 = 0
        assert!(reader.read_bool().expect("ok")); // bit 2 = 1
    }

    #[test]
    #[ignore]
    fn test_bitreader_eof() {
        let data = [0xFF];
        let mut reader = BitReader::new(&data);
        let _ = reader.read_bits(8).expect("ok");
        assert!(reader.read_bits(1).is_err());
    }

    #[test]
    #[ignore]
    fn test_bitreader_remaining() {
        let data = [0xFF, 0xFF];
        let mut reader = BitReader::new(&data);
        assert_eq!(reader.remaining_bits(), 16);
        let _ = reader.read_bits(3).expect("ok");
        assert_eq!(reader.remaining_bits(), 13);
    }

    #[test]
    #[ignore]
    fn test_bitreader_align() {
        let data = [0xFF, 0xAA];
        let mut reader = BitReader::new(&data);
        let _ = reader.read_bits(3).expect("ok");
        reader.align_to_byte();
        // Should now be at byte 1
        assert_eq!(reader.read_bits(8).expect("ok"), 0xAA);
    }

    #[test]
    #[ignore]
    fn test_bitwriter_basic() {
        let mut writer = BitWriter::new();
        writer.write_bits(0b0110, 4);
        writer.write_bits(0b1010, 4);
        let data = writer.finish();
        assert_eq!(data, vec![0b1010_0110]);
    }

    #[test]
    #[ignore]
    fn test_bitwriter_cross_byte() {
        let mut writer = BitWriter::new();
        writer.write_bits(0xF, 4);
        writer.write_bits(0x0F, 8);
        let data = writer.finish();
        assert_eq!(data, vec![0xFF, 0x00]);
    }

    #[test]
    #[ignore]
    fn test_bitwriter_bool() {
        let mut writer = BitWriter::new();
        writer.write_bool(true);
        writer.write_bool(false);
        writer.write_bool(true);
        writer.write_bool(false);
        writer.write_bool(false);
        writer.write_bool(false);
        writer.write_bool(false);
        writer.write_bool(false);
        let data = writer.finish();
        assert_eq!(data, vec![0b0000_0101]);
    }

    #[test]
    #[ignore]
    fn test_roundtrip_bits() {
        let mut writer = BitWriter::new();
        writer.write_bits(42, 7);
        writer.write_bits(1023, 10);
        writer.write_bits(0, 3);
        writer.write_bits(255, 8);
        let data = writer.finish();

        let mut reader = BitReader::new(&data);
        assert_eq!(reader.read_bits(7).expect("ok"), 42);
        assert_eq!(reader.read_bits(10).expect("ok"), 1023);
        assert_eq!(reader.read_bits(3).expect("ok"), 0);
        assert_eq!(reader.read_bits(8).expect("ok"), 255);
    }

    #[test]
    #[ignore]
    fn test_roundtrip_u64() {
        for value in [0u64, 1, 5, 16, 17, 100, 272, 273, 1000, 65535, 1_000_000] {
            let mut writer = BitWriter::new();
            writer.write_u64(value);
            let data = writer.finish();

            let mut reader = BitReader::new(&data);
            let decoded = reader.read_u64().expect("ok");
            assert_eq!(decoded, value, "U64 roundtrip failed for {value}");
        }
    }

    #[test]
    #[ignore]
    fn test_bitwriter_align() {
        let mut writer = BitWriter::new();
        writer.write_bits(0b101, 3);
        writer.align_to_byte();
        writer.write_bits(0xAA, 8);
        let data = writer.finish();
        assert_eq!(data.len(), 2);
        assert_eq!(data[0], 0b0000_0101);
        assert_eq!(data[1], 0xAA);
    }
}
