//! Bitstream reading and writing utilities.
//!
//! Provides bit-level access to byte buffers, supporting arbitrary-width
//! field reads and writes as used in video codec bitstream parsing.

// -------------------------------------------------------------------------
// BitstreamWriter
// -------------------------------------------------------------------------

/// Writes individual bits or multi-bit fields into a byte buffer.
///
/// Bits are packed MSB-first within each byte.  Call [`flush`](Self::flush)
/// to finalise and retrieve the completed buffer.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct BitstreamWriter {
    /// Completed bytes.
    buffer: Vec<u8>,
    /// Position of the next bit to write within `current_byte` (0 = MSB, 7 = LSB).
    bit_pos: u8,
    /// Byte being assembled; pushed to `buffer` when full.
    current_byte: u8,
}

impl BitstreamWriter {
    /// Creates an empty `BitstreamWriter`.
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            bit_pos: 0,
            current_byte: 0,
        }
    }

    /// Writes a single bit.  `true` → 1, `false` → 0.
    #[allow(dead_code)]
    pub fn write_bit(&mut self, bit: bool) {
        if bit {
            self.current_byte |= 0x80 >> self.bit_pos;
        }
        self.bit_pos += 1;
        if self.bit_pos == 8 {
            self.buffer.push(self.current_byte);
            self.current_byte = 0;
            self.bit_pos = 0;
        }
    }

    /// Writes the lowest `count` bits of `value`, most-significant bit first.
    ///
    /// `count` must be in `1..=64`.
    #[allow(dead_code)]
    pub fn write_bits(&mut self, value: u64, count: u8) {
        assert!(count > 0 && count <= 64, "count must be in 1..=64");
        for i in (0..count).rev() {
            let bit = (value >> i) & 1 == 1;
            self.write_bit(bit);
        }
    }

    /// Writes all 8 bits of `byte`, MSB first.
    #[allow(dead_code)]
    pub fn write_byte(&mut self, byte: u8) {
        self.write_bits(u64::from(byte), 8);
    }

    /// Flushes any partially-assembled byte (zero-padded on the LSB side)
    /// and returns the completed buffer.
    #[allow(dead_code)]
    pub fn flush(mut self) -> Vec<u8> {
        if self.bit_pos > 0 {
            self.buffer.push(self.current_byte);
        }
        self.buffer
    }

    /// Returns the total number of bits written so far (including bits in the
    /// current partial byte).
    #[allow(dead_code)]
    pub fn bits_written(&self) -> u64 {
        self.buffer.len() as u64 * 8 + u64::from(self.bit_pos)
    }

    /// Returns the number of complete bytes in the buffer (excluding any
    /// partial byte currently being assembled).
    #[allow(dead_code)]
    pub fn bytes_used(&self) -> usize {
        self.buffer.len()
    }
}

// -------------------------------------------------------------------------
// BitstreamReader
// -------------------------------------------------------------------------

/// Reads individual bits or multi-bit fields from a byte slice.
///
/// Bits are extracted MSB-first from each byte, matching the write order of
/// [`BitstreamWriter`].
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct BitstreamReader {
    /// The underlying data.
    data: Vec<u8>,
    /// Index of the byte currently being read.
    byte_pos: usize,
    /// Bit position within the current byte (0 = MSB).
    bit_pos: u8,
}

impl BitstreamReader {
    /// Creates a `BitstreamReader` over the given data.
    #[allow(dead_code)]
    pub fn new(data: Vec<u8>) -> Self {
        Self {
            data,
            byte_pos: 0,
            bit_pos: 0,
        }
    }

    /// Reads a single bit, returning `None` when the buffer is exhausted.
    #[allow(dead_code)]
    pub fn read_bit(&mut self) -> Option<bool> {
        if self.byte_pos >= self.data.len() {
            return None;
        }
        let byte = self.data[self.byte_pos];
        let bit = (byte >> (7 - self.bit_pos)) & 1 == 1;
        self.bit_pos += 1;
        if self.bit_pos == 8 {
            self.byte_pos += 1;
            self.bit_pos = 0;
        }
        Some(bit)
    }

    /// Reads `count` bits (MSB first) and assembles them into a `u64`.
    ///
    /// Returns `None` if there are fewer than `count` bits remaining.
    /// `count` must be in `1..=64`.
    #[allow(dead_code)]
    pub fn read_bits(&mut self, count: u8) -> Option<u64> {
        assert!(count > 0 && count <= 64, "count must be in 1..=64");
        let mut value: u64 = 0;
        for _ in 0..count {
            let bit = self.read_bit()?;
            value = (value << 1) | u64::from(bit);
        }
        Some(value)
    }

    /// Reads 8 bits and returns them as a `u8`.
    ///
    /// Returns `None` when fewer than 8 bits remain.
    #[allow(dead_code)]
    pub fn read_byte(&mut self) -> Option<u8> {
        self.read_bits(8).map(|v| v as u8)
    }

    /// Returns the number of bits remaining (bits not yet read).
    #[allow(dead_code)]
    pub fn bits_remaining(&self) -> usize {
        let total_bits = self.data.len() * 8;
        let consumed = self.byte_pos * 8 + usize::from(self.bit_pos);
        total_bits.saturating_sub(consumed)
    }

    /// Returns `true` when all bits have been consumed.
    #[allow(dead_code)]
    pub fn is_exhausted(&self) -> bool {
        self.bits_remaining() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- BitstreamWriter tests ---

    #[test]
    fn writer_new_is_empty() {
        let w = BitstreamWriter::new();
        assert_eq!(w.bits_written(), 0);
        assert_eq!(w.bytes_used(), 0);
    }

    #[test]
    fn writer_flush_empty() {
        let w = BitstreamWriter::new();
        assert!(w.flush().is_empty());
    }

    #[test]
    fn writer_write_single_bit_one() {
        let mut w = BitstreamWriter::new();
        w.write_bit(true);
        assert_eq!(w.bits_written(), 1);
        let buf = w.flush();
        assert_eq!(buf, vec![0x80]); // 1000_0000
    }

    #[test]
    fn writer_write_single_bit_zero() {
        let mut w = BitstreamWriter::new();
        w.write_bit(false);
        assert_eq!(w.bits_written(), 1);
        let buf = w.flush();
        assert_eq!(buf, vec![0x00]);
    }

    #[test]
    fn writer_write_byte_roundtrip() {
        for byte in [0x00u8, 0xFF, 0xAA, 0x55, 0x12, 0xAB] {
            let mut w = BitstreamWriter::new();
            w.write_byte(byte);
            let buf = w.flush();
            assert_eq!(buf, vec![byte], "failed for byte 0x{byte:02X}");
        }
    }

    #[test]
    fn writer_write_bits_4() {
        // Write 0b1011 as 4 bits, then flush → 0b1011_0000 = 0xB0
        let mut w = BitstreamWriter::new();
        w.write_bits(0b1011, 4);
        assert_eq!(w.bits_written(), 4);
        let buf = w.flush();
        assert_eq!(buf, vec![0xB0]);
    }

    #[test]
    fn writer_bits_written_and_bytes_used() {
        let mut w = BitstreamWriter::new();
        w.write_bits(0xFF, 8);
        assert_eq!(w.bytes_used(), 1);
        assert_eq!(w.bits_written(), 8);
        w.write_bit(true);
        // 9 bits: 1 complete byte + 1 bit pending
        assert_eq!(w.bytes_used(), 1);
        assert_eq!(w.bits_written(), 9);
    }

    #[test]
    fn writer_multiple_bytes() {
        let mut w = BitstreamWriter::new();
        w.write_byte(0xDE);
        w.write_byte(0xAD);
        let buf = w.flush();
        assert_eq!(buf, vec![0xDE, 0xAD]);
    }

    // --- BitstreamReader tests ---

    #[test]
    fn reader_empty_is_exhausted() {
        let r = BitstreamReader::new(vec![]);
        assert!(r.is_exhausted());
        assert_eq!(r.bits_remaining(), 0);
    }

    #[test]
    fn reader_read_bit_from_single_byte() {
        // 0xA5 = 1010_0101
        let mut r = BitstreamReader::new(vec![0xA5]);
        assert_eq!(r.read_bit(), Some(true));
        assert_eq!(r.read_bit(), Some(false));
        assert_eq!(r.read_bit(), Some(true));
        assert_eq!(r.read_bit(), Some(false));
        assert_eq!(r.read_bit(), Some(false));
        assert_eq!(r.read_bit(), Some(true));
        assert_eq!(r.read_bit(), Some(false));
        assert_eq!(r.read_bit(), Some(true));
        assert_eq!(r.read_bit(), None); // exhausted
    }

    #[test]
    fn reader_bits_remaining_decrements() {
        let mut r = BitstreamReader::new(vec![0xFF]);
        assert_eq!(r.bits_remaining(), 8);
        r.read_bit();
        assert_eq!(r.bits_remaining(), 7);
    }

    #[test]
    fn reader_read_byte_full() {
        let mut r = BitstreamReader::new(vec![0x42]);
        assert_eq!(r.read_byte(), Some(0x42));
        assert!(r.is_exhausted());
    }

    #[test]
    fn reader_read_byte_insufficient() {
        let mut r = BitstreamReader::new(vec![]);
        assert_eq!(r.read_byte(), None);
    }

    // --- Round-trip tests ---

    #[test]
    fn roundtrip_byte_sequence() {
        let original = vec![0xDE_u8, 0xAD, 0xBE, 0xEF];
        let mut w = BitstreamWriter::new();
        for &b in &original {
            w.write_byte(b);
        }
        let buf = w.flush();
        let mut r = BitstreamReader::new(buf);
        for &expected in &original {
            assert_eq!(r.read_byte(), Some(expected));
        }
        assert!(r.is_exhausted());
    }

    #[test]
    fn roundtrip_mixed_bit_widths() {
        // Write: 3-bit field (0b101=5), 5-bit field (0b10110=22), 8-bit byte (0xFF).
        let mut w = BitstreamWriter::new();
        w.write_bits(5, 3);
        w.write_bits(22, 5);
        w.write_byte(0xFF);
        let buf = w.flush();

        let mut r = BitstreamReader::new(buf);
        assert_eq!(r.read_bits(3), Some(5));
        assert_eq!(r.read_bits(5), Some(22));
        assert_eq!(r.read_byte(), Some(0xFF));
        assert!(r.is_exhausted());
    }

    #[test]
    fn roundtrip_single_bits() {
        let bits = [true, false, true, true, false, false, true, false];
        let mut w = BitstreamWriter::new();
        for &b in &bits {
            w.write_bit(b);
        }
        let buf = w.flush();

        let mut r = BitstreamReader::new(buf);
        for &expected in &bits {
            assert_eq!(r.read_bit(), Some(expected));
        }
        assert!(r.is_exhausted());
    }

    #[test]
    fn roundtrip_64bit_value() {
        let value: u64 = 0xDEAD_BEEF_CAFE_1234;
        let mut w = BitstreamWriter::new();
        w.write_bits(value, 64);
        let buf = w.flush();
        assert_eq!(buf.len(), 8);
        let mut r = BitstreamReader::new(buf);
        let read_back = r.read_bits(64).expect("should succeed");
        assert_eq!(read_back, value);
        assert!(r.is_exhausted());
    }
}
