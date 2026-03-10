//! Bit writer for FLAC encoding.
//!
//! Provides bit-level write operations for encoding FLAC frames.

#![forbid(unsafe_code)]

use crate::AudioError;

/// Bit writer for encoding.
#[derive(Debug, Clone)]
pub struct BitWriter {
    /// Output buffer.
    buffer: Vec<u8>,
    /// Current byte being written.
    current_byte: u8,
    /// Number of bits written to current byte (0-7).
    bit_count: u8,
}

impl BitWriter {
    /// Create new bit writer.
    #[must_use]
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            current_byte: 0,
            bit_count: 0,
        }
    }

    /// Create new bit writer with capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            current_byte: 0,
            bit_count: 0,
        }
    }

    /// Write a single bit.
    pub fn write_bit(&mut self, bit: bool) {
        if bit {
            self.current_byte |= 1 << (7 - self.bit_count);
        }
        self.bit_count += 1;

        if self.bit_count == 8 {
            self.flush_byte();
        }
    }

    /// Write multiple bits from unsigned value.
    pub fn write_bits(&mut self, value: u32, count: u8) {
        if count == 0 {
            return;
        }

        for i in (0..count).rev() {
            let bit = (value >> i) & 1;
            self.write_bit(bit != 0);
        }
    }

    /// Write signed value with given bit width.
    pub fn write_signed(&mut self, value: i32, bits: u8) {
        if bits == 0 {
            return;
        }

        // Sign extend if needed
        let mask = (1u32 << bits) - 1;
        let unsigned = (value as u32) & mask;
        self.write_bits(unsigned, bits);
    }

    /// Write unary coded value (n ones followed by a zero).
    pub fn write_unary(&mut self, value: u32) {
        for _ in 0..value {
            self.write_bit(true);
        }
        self.write_bit(false);
    }

    /// Write Rice-coded value.
    pub fn write_rice(&mut self, value: i32, parameter: u8) {
        // Zig-zag encode
        let unsigned = super::rice::zigzag_encode(value);

        // Split into quotient and remainder
        let quotient = unsigned >> parameter;
        let remainder = unsigned & ((1 << parameter) - 1);

        // Write quotient in unary
        self.write_unary(quotient);

        // Write remainder in binary
        self.write_bits(remainder, parameter);
    }

    /// Write UTF-8 encoded u32 (for frame number).
    ///
    /// # Errors
    ///
    /// Returns error if value is too large.
    pub fn write_utf8_u32(&mut self, value: u32) -> Result<(), AudioError> {
        if value < 0x80 {
            // 1 byte: 0xxxxxxx
            self.write_bits(value, 8);
        } else if value < 0x800 {
            // 2 bytes: 110xxxxx 10xxxxxx
            self.write_bits(0xC0 | (value >> 6), 8);
            self.write_bits(0x80 | (value & 0x3F), 8);
        } else if value < 0x1_0000 {
            // 3 bytes: 1110xxxx 10xxxxxx 10xxxxxx
            self.write_bits(0xE0 | (value >> 12), 8);
            self.write_bits(0x80 | ((value >> 6) & 0x3F), 8);
            self.write_bits(0x80 | (value & 0x3F), 8);
        } else if value < 0x20_0000 {
            // 4 bytes: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
            self.write_bits(0xF0 | (value >> 18), 8);
            self.write_bits(0x80 | ((value >> 12) & 0x3F), 8);
            self.write_bits(0x80 | ((value >> 6) & 0x3F), 8);
            self.write_bits(0x80 | (value & 0x3F), 8);
        } else {
            return Err(AudioError::InvalidData(
                "Frame number too large for UTF-8".into(),
            ));
        }
        Ok(())
    }

    /// Write UTF-8 encoded u64 (for sample number).
    ///
    /// # Errors
    ///
    /// Returns error if value is too large.
    pub fn write_utf8_u64(&mut self, value: u64) -> Result<(), AudioError> {
        if value < 0x80 {
            self.write_bits(value as u32, 8);
        } else if value < 0x800 {
            self.write_bits(0xC0 | ((value >> 6) as u32), 8);
            self.write_bits(0x80 | ((value & 0x3F) as u32), 8);
        } else if value < 0x1_0000 {
            self.write_bits(0xE0 | ((value >> 12) as u32), 8);
            self.write_bits(0x80 | (((value >> 6) & 0x3F) as u32), 8);
            self.write_bits(0x80 | ((value & 0x3F) as u32), 8);
        } else if value < 0x20_0000 {
            self.write_bits(0xF0 | ((value >> 18) as u32), 8);
            self.write_bits(0x80 | (((value >> 12) & 0x3F) as u32), 8);
            self.write_bits(0x80 | (((value >> 6) & 0x3F) as u32), 8);
            self.write_bits(0x80 | ((value & 0x3F) as u32), 8);
        } else if value < 0x400_0000 {
            self.write_bits(0xF8 | ((value >> 24) as u32), 8);
            self.write_bits(0x80 | (((value >> 18) & 0x3F) as u32), 8);
            self.write_bits(0x80 | (((value >> 12) & 0x3F) as u32), 8);
            self.write_bits(0x80 | (((value >> 6) & 0x3F) as u32), 8);
            self.write_bits(0x80 | ((value & 0x3F) as u32), 8);
        } else if value < 0x8000_0000 {
            self.write_bits(0xFC | ((value >> 30) as u32), 8);
            self.write_bits(0x80 | (((value >> 24) & 0x3F) as u32), 8);
            self.write_bits(0x80 | (((value >> 18) & 0x3F) as u32), 8);
            self.write_bits(0x80 | (((value >> 12) & 0x3F) as u32), 8);
            self.write_bits(0x80 | (((value >> 6) & 0x3F) as u32), 8);
            self.write_bits(0x80 | ((value & 0x3F) as u32), 8);
        } else if value < 0x1_0000_0000 {
            self.write_bits(0xFE, 8);
            self.write_bits(0x80 | (((value >> 30) & 0x3F) as u32), 8);
            self.write_bits(0x80 | (((value >> 24) & 0x3F) as u32), 8);
            self.write_bits(0x80 | (((value >> 18) & 0x3F) as u32), 8);
            self.write_bits(0x80 | (((value >> 12) & 0x3F) as u32), 8);
            self.write_bits(0x80 | (((value >> 6) & 0x3F) as u32), 8);
            self.write_bits(0x80 | ((value & 0x3F) as u32), 8);
        } else {
            return Err(AudioError::InvalidData(
                "Sample number too large for UTF-8".into(),
            ));
        }
        Ok(())
    }

    /// Flush current byte to buffer.
    fn flush_byte(&mut self) {
        self.buffer.push(self.current_byte);
        self.current_byte = 0;
        self.bit_count = 0;
    }

    /// Align to byte boundary (pad with zeros).
    pub fn byte_align(&mut self) {
        if self.bit_count > 0 {
            self.flush_byte();
        }
    }

    /// Get number of bytes written (including partial byte).
    #[must_use]
    pub fn len_bytes(&self) -> usize {
        if self.bit_count > 0 {
            self.buffer.len() + 1
        } else {
            self.buffer.len()
        }
    }

    /// Get number of bits written.
    #[must_use]
    pub fn len_bits(&self) -> usize {
        self.buffer.len() * 8 + usize::from(self.bit_count)
    }

    /// Check if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty() && self.bit_count == 0
    }

    /// Get the output buffer (finishes writing).
    #[must_use]
    pub fn finish(mut self) -> Vec<u8> {
        self.byte_align();
        self.buffer
    }

    /// Get reference to buffer without consuming.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.buffer
    }

    /// Clear the writer.
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.current_byte = 0;
        self.bit_count = 0;
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
    fn test_write_bit() {
        let mut writer = BitWriter::new();
        writer.write_bit(true);
        writer.write_bit(false);
        writer.write_bit(true);
        writer.write_bit(true);
        writer.write_bit(false);
        writer.write_bit(false);
        writer.write_bit(false);
        writer.write_bit(true);

        let data = writer.finish();
        assert_eq!(data, vec![0b10110001]);
    }

    #[test]
    fn test_write_bits() {
        let mut writer = BitWriter::new();
        writer.write_bits(0b1011, 4);
        writer.write_bits(0b0001, 4);

        let data = writer.finish();
        assert_eq!(data, vec![0b10110001]);
    }

    #[test]
    fn test_write_unary() {
        let mut writer = BitWriter::new();
        writer.write_unary(3);
        writer.write_unary(0);

        let data = writer.finish();
        assert_eq!(data[0] >> 4, 0b1110);
    }

    #[test]
    fn test_byte_align() {
        let mut writer = BitWriter::new();
        writer.write_bits(0b1011, 4);
        writer.byte_align();

        let data = writer.finish();
        assert_eq!(data, vec![0b10110000]);
    }

    #[test]
    fn test_len() {
        let mut writer = BitWriter::new();
        assert_eq!(writer.len_bits(), 0);
        assert_eq!(writer.len_bytes(), 0);

        writer.write_bits(0xFF, 8);
        assert_eq!(writer.len_bits(), 8);
        assert_eq!(writer.len_bytes(), 1);

        writer.write_bit(true);
        assert_eq!(writer.len_bits(), 9);
        assert_eq!(writer.len_bytes(), 2);
    }

    #[test]
    fn test_write_signed() {
        let mut writer = BitWriter::new();
        writer.write_signed(-1, 8);
        let data = writer.finish();
        assert_eq!(data, vec![0xFF]);
    }

    #[test]
    fn test_write_utf8_u32_small() {
        let mut writer = BitWriter::new();
        writer.write_utf8_u32(65).expect("should succeed");
        let data = writer.finish();
        assert_eq!(data, vec![65]);
    }

    #[test]
    fn test_write_utf8_u32_two_bytes() {
        let mut writer = BitWriter::new();
        writer.write_utf8_u32(0x80).expect("should succeed");
        let data = writer.finish();
        assert_eq!(data.len(), 2);
        assert_eq!(data[0] & 0xE0, 0xC0);
        assert_eq!(data[1] & 0xC0, 0x80);
    }

    #[test]
    fn test_clear() {
        let mut writer = BitWriter::new();
        writer.write_bits(0xFF, 8);
        writer.clear();
        assert!(writer.is_empty());
    }
}
