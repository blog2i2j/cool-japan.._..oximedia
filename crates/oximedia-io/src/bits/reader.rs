//! Bit-level reader for parsing binary formats.

use oximedia_core::{OxiError, OxiResult};

/// Bit-level reader for parsing binary formats.
///
/// `BitReader` allows reading individual bits and multi-bit values from
/// a byte slice. It tracks both byte position and bit position within
/// the current byte.
///
/// # Bit Ordering
///
/// Bits are read from MSB (most significant bit) to LSB (least significant bit)
/// within each byte. This is the standard ordering used by most video codecs
/// including H.264, HEVC, AV1, and VP9.
///
/// # Example
///
/// ```
/// use oximedia_io::bits::BitReader;
///
/// let data = [0b10110100, 0b11001010];
/// let mut reader = BitReader::new(&data);
///
/// // Read individual bits (from MSB to LSB)
/// assert_eq!(reader.read_bit()?, 1);
/// assert_eq!(reader.read_bit()?, 0);
/// assert_eq!(reader.read_bit()?, 1);
/// assert_eq!(reader.read_bit()?, 1);
///
/// // Read multiple bits as a value
/// assert_eq!(reader.read_bits(4)?, 0b0100);
/// # Ok::<(), oximedia_core::OxiError>(())
/// ```
#[derive(Debug, Clone)]
pub struct BitReader<'a> {
    /// The underlying byte slice
    data: &'a [u8],
    /// Current byte position
    byte_pos: usize,
    /// Current bit position within the byte (0-7, where 0 is MSB)
    bit_pos: u8,
}

#[allow(clippy::elidable_lifetime_names)]
impl<'a> BitReader<'a> {
    /// Creates a new `BitReader` from a byte slice.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_io::bits::BitReader;
    ///
    /// let data = [0xFF, 0x00];
    /// let reader = BitReader::new(&data);
    /// assert!(reader.has_more_data());
    /// ```
    #[must_use]
    pub const fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_pos: 0,
            bit_pos: 0,
        }
    }

    /// Reads a single bit from the stream.
    ///
    /// Returns `0` or `1`.
    ///
    /// # Errors
    ///
    /// Returns [`OxiError::UnexpectedEof`] if there are no more bits to read.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_io::bits::BitReader;
    ///
    /// let data = [0b10000000];
    /// let mut reader = BitReader::new(&data);
    ///
    /// assert_eq!(reader.read_bit()?, 1);
    /// assert_eq!(reader.read_bit()?, 0);
    /// # Ok::<(), oximedia_core::OxiError>(())
    /// ```
    pub fn read_bit(&mut self) -> OxiResult<u8> {
        if self.byte_pos >= self.data.len() {
            return Err(OxiError::UnexpectedEof);
        }

        // Read bit from MSB to LSB (bit_pos 0 = MSB)
        let bit = (self.data[self.byte_pos] >> (7 - self.bit_pos)) & 1;
        self.bit_pos += 1;

        if self.bit_pos == 8 {
            self.bit_pos = 0;
            self.byte_pos += 1;
        }

        Ok(bit)
    }

    /// Reads up to 64 bits from the stream.
    ///
    /// # Arguments
    ///
    /// * `n` - Number of bits to read (0-64)
    ///
    /// # Errors
    ///
    /// Returns [`OxiError::InvalidData`] if `n > 64`.
    /// Returns [`OxiError::UnexpectedEof`] if there are not enough bits.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_io::bits::BitReader;
    ///
    /// let data = [0b10110100, 0b11001010];
    /// let mut reader = BitReader::new(&data);
    ///
    /// assert_eq!(reader.read_bits(4)?, 0b1011);
    /// assert_eq!(reader.read_bits(4)?, 0b0100);
    /// assert_eq!(reader.read_bits(8)?, 0b11001010);
    /// # Ok::<(), oximedia_core::OxiError>(())
    /// ```
    pub fn read_bits(&mut self, n: u8) -> OxiResult<u64> {
        if n > 64 {
            return Err(OxiError::InvalidData(
                "Cannot read more than 64 bits at once".to_string(),
            ));
        }

        if n == 0 {
            return Ok(0);
        }

        let mut value = 0u64;
        for _ in 0..n {
            value = (value << 1) | u64::from(self.read_bit()?);
        }

        Ok(value)
    }

    /// Reads an 8-bit unsigned integer.
    ///
    /// # Errors
    ///
    /// Returns [`OxiError::UnexpectedEof`] if there are not enough bits.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_io::bits::BitReader;
    ///
    /// let data = [0x12, 0x34];
    /// let mut reader = BitReader::new(&data);
    ///
    /// assert_eq!(reader.read_u8()?, 0x12);
    /// assert_eq!(reader.read_u8()?, 0x34);
    /// # Ok::<(), oximedia_core::OxiError>(())
    /// ```
    #[allow(clippy::cast_possible_truncation)]
    pub fn read_u8(&mut self) -> OxiResult<u8> {
        Ok(self.read_bits(8)? as u8)
    }

    /// Reads a 16-bit unsigned integer in big-endian order.
    ///
    /// # Errors
    ///
    /// Returns [`OxiError::UnexpectedEof`] if there are not enough bits.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_io::bits::BitReader;
    ///
    /// let data = [0x12, 0x34];
    /// let mut reader = BitReader::new(&data);
    ///
    /// assert_eq!(reader.read_u16()?, 0x1234);
    /// # Ok::<(), oximedia_core::OxiError>(())
    /// ```
    #[allow(clippy::cast_possible_truncation)]
    pub fn read_u16(&mut self) -> OxiResult<u16> {
        Ok(self.read_bits(16)? as u16)
    }

    /// Reads a 32-bit unsigned integer in big-endian order.
    ///
    /// # Errors
    ///
    /// Returns [`OxiError::UnexpectedEof`] if there are not enough bits.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_io::bits::BitReader;
    ///
    /// let data = [0x12, 0x34, 0x56, 0x78];
    /// let mut reader = BitReader::new(&data);
    ///
    /// assert_eq!(reader.read_u32()?, 0x12345678);
    /// # Ok::<(), oximedia_core::OxiError>(())
    /// ```
    #[allow(clippy::cast_possible_truncation)]
    pub fn read_u32(&mut self) -> OxiResult<u32> {
        Ok(self.read_bits(32)? as u32)
    }

    /// Reads a 64-bit unsigned integer in big-endian order.
    ///
    /// # Errors
    ///
    /// Returns [`OxiError::UnexpectedEof`] if there are not enough bits.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_io::bits::BitReader;
    ///
    /// let data = [0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0];
    /// let mut reader = BitReader::new(&data);
    ///
    /// assert_eq!(reader.read_u64()?, 0x123456789ABCDEF0);
    /// # Ok::<(), oximedia_core::OxiError>(())
    /// ```
    pub fn read_u64(&mut self) -> OxiResult<u64> {
        self.read_bits(64)
    }

    /// Reads a boolean flag (single bit).
    ///
    /// Returns `true` if the bit is 1, `false` if 0.
    ///
    /// # Errors
    ///
    /// Returns [`OxiError::UnexpectedEof`] if there are no more bits.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_io::bits::BitReader;
    ///
    /// let data = [0b10000000];
    /// let mut reader = BitReader::new(&data);
    ///
    /// assert!(reader.read_flag()?);
    /// assert!(!reader.read_flag()?);
    /// # Ok::<(), oximedia_core::OxiError>(())
    /// ```
    pub fn read_flag(&mut self) -> OxiResult<bool> {
        Ok(self.read_bit()? != 0)
    }

    /// Skips the specified number of bits.
    ///
    /// This method silently ignores attempts to skip past the end of data.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_io::bits::BitReader;
    ///
    /// let data = [0xFF, 0x00];
    /// let mut reader = BitReader::new(&data);
    ///
    /// reader.skip_bits(4);
    /// assert_eq!(reader.bits_read(), 4);
    /// ```
    pub fn skip_bits(&mut self, n: usize) {
        for _ in 0..n {
            if self.read_bit().is_err() {
                break;
            }
        }
    }

    /// Aligns the reader to the next byte boundary.
    ///
    /// If the reader is already at a byte boundary, this is a no-op.
    /// Otherwise, the remaining bits in the current byte are skipped.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_io::bits::BitReader;
    ///
    /// let data = [0xFF, 0x00];
    /// let mut reader = BitReader::new(&data);
    ///
    /// reader.read_bits(3)?;  // Read 3 bits
    /// reader.byte_align();           // Skip remaining 5 bits
    /// assert_eq!(reader.bits_read(), 8);
    /// # Ok::<(), oximedia_core::OxiError>(())
    /// ```
    pub fn byte_align(&mut self) {
        if self.bit_pos != 0 {
            self.bit_pos = 0;
            self.byte_pos += 1;
        }
    }

    /// Returns `true` if there is more data available to read.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_io::bits::BitReader;
    ///
    /// let data = [0xFF];
    /// let mut reader = BitReader::new(&data);
    ///
    /// assert!(reader.has_more_data());
    /// reader.read_bits(8)?;
    /// assert!(!reader.has_more_data());
    /// # Ok::<(), oximedia_core::OxiError>(())
    /// ```
    #[must_use]
    pub fn has_more_data(&self) -> bool {
        self.byte_pos < self.data.len()
    }

    /// Returns the number of complete bytes remaining.
    ///
    /// This does not count partial bytes at the current position.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_io::bits::BitReader;
    ///
    /// let data = [0xFF, 0x00, 0xFF];
    /// let mut reader = BitReader::new(&data);
    ///
    /// assert_eq!(reader.remaining_bytes(), 3);
    /// reader.read_bits(4)?;
    /// assert_eq!(reader.remaining_bytes(), 2);  // Partial byte not counted
    /// # Ok::<(), oximedia_core::OxiError>(())
    /// ```
    #[must_use]
    pub fn remaining_bytes(&self) -> usize {
        if self.byte_pos >= self.data.len() {
            0
        } else {
            self.data.len() - self.byte_pos - usize::from(self.bit_pos > 0)
        }
    }

    /// Returns the total number of bits read so far.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_io::bits::BitReader;
    ///
    /// let data = [0xFF, 0x00];
    /// let mut reader = BitReader::new(&data);
    ///
    /// assert_eq!(reader.bits_read(), 0);
    /// reader.read_bits(5)?;
    /// assert_eq!(reader.bits_read(), 5);
    /// reader.read_bits(3)?;
    /// assert_eq!(reader.bits_read(), 8);
    /// # Ok::<(), oximedia_core::OxiError>(())
    /// ```
    #[must_use]
    pub fn bits_read(&self) -> usize {
        self.byte_pos * 8 + self.bit_pos as usize
    }

    /// Returns the total number of remaining bits.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_io::bits::BitReader;
    ///
    /// let data = [0xFF, 0x00];
    /// let mut reader = BitReader::new(&data);
    ///
    /// assert_eq!(reader.remaining_bits(), 16);
    /// reader.read_bits(5)?;
    /// assert_eq!(reader.remaining_bits(), 11);
    /// # Ok::<(), oximedia_core::OxiError>(())
    /// ```
    #[must_use]
    pub fn remaining_bits(&self) -> usize {
        if self.byte_pos >= self.data.len() {
            0
        } else {
            (self.data.len() - self.byte_pos) * 8 - self.bit_pos as usize
        }
    }

    /// Peeks at the next bit without consuming it.
    ///
    /// # Errors
    ///
    /// Returns [`OxiError::UnexpectedEof`] if there are no more bits.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_io::bits::BitReader;
    ///
    /// let data = [0b10000000];
    /// let mut reader = BitReader::new(&data);
    ///
    /// assert_eq!(reader.peek_bit()?, 1);
    /// assert_eq!(reader.peek_bit()?, 1);  // Still 1, not consumed
    /// assert_eq!(reader.read_bit()?, 1);  // Now consumed
    /// assert_eq!(reader.peek_bit()?, 0);  // Next bit
    /// # Ok::<(), oximedia_core::OxiError>(())
    /// ```
    pub fn peek_bit(&self) -> OxiResult<u8> {
        if self.byte_pos >= self.data.len() {
            return Err(OxiError::UnexpectedEof);
        }

        Ok((self.data[self.byte_pos] >> (7 - self.bit_pos)) & 1)
    }

    /// Returns the underlying byte slice.
    #[must_use]
    pub const fn data(&self) -> &'a [u8] {
        self.data
    }

    /// Returns the current byte position.
    #[must_use]
    pub const fn byte_position(&self) -> usize {
        self.byte_pos
    }

    /// Returns the current bit position within the current byte (0-7).
    #[must_use]
    pub const fn bit_position(&self) -> u8 {
        self.bit_pos
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let data = [0xFF, 0x00];
        let reader = BitReader::new(&data);
        assert_eq!(reader.byte_position(), 0);
        assert_eq!(reader.bit_position(), 0);
        assert!(reader.has_more_data());
    }

    #[test]
    fn test_read_bit() {
        let data = [0b10110100];
        let mut reader = BitReader::new(&data);

        assert_eq!(reader.read_bit().expect("read_bit should succeed"), 1);
        assert_eq!(reader.read_bit().expect("read_bit should succeed"), 0);
        assert_eq!(reader.read_bit().expect("read_bit should succeed"), 1);
        assert_eq!(reader.read_bit().expect("read_bit should succeed"), 1);
        assert_eq!(reader.read_bit().expect("read_bit should succeed"), 0);
        assert_eq!(reader.read_bit().expect("read_bit should succeed"), 1);
        assert_eq!(reader.read_bit().expect("read_bit should succeed"), 0);
        assert_eq!(reader.read_bit().expect("read_bit should succeed"), 0);
    }

    #[test]
    fn test_read_bits() {
        let data = [0b10110100, 0b11001010];
        let mut reader = BitReader::new(&data);

        assert_eq!(
            reader.read_bits(4).expect("read_bits should succeed"),
            0b1011
        );
        assert_eq!(
            reader.read_bits(4).expect("read_bits should succeed"),
            0b0100
        );
        assert_eq!(
            reader.read_bits(8).expect("read_bits should succeed"),
            0b11001010
        );
    }

    #[test]
    fn test_read_bits_across_bytes() {
        let data = [0b10110100, 0b11001010];
        let mut reader = BitReader::new(&data);

        assert_eq!(
            reader.read_bits(12).expect("read_bits should succeed"),
            0b101101001100
        );
        assert_eq!(
            reader.read_bits(4).expect("read_bits should succeed"),
            0b1010
        );
    }

    #[test]
    fn test_read_bits_zero() {
        let data = [0xFF];
        let mut reader = BitReader::new(&data);
        assert_eq!(reader.read_bits(0).expect("read_bits should succeed"), 0);
        assert_eq!(reader.bits_read(), 0);
    }

    #[test]
    fn test_read_bits_too_many() {
        let data = [0xFF];
        let mut reader = BitReader::new(&data);
        let result = reader.read_bits(65);
        assert!(result.is_err());
    }

    #[test]
    fn test_read_u8() {
        let data = [0x12, 0x34];
        let mut reader = BitReader::new(&data);
        assert_eq!(reader.read_u8().expect("read_u8 should succeed"), 0x12);
        assert_eq!(reader.read_u8().expect("read_u8 should succeed"), 0x34);
    }

    #[test]
    fn test_read_u16() {
        let data = [0x12, 0x34];
        let mut reader = BitReader::new(&data);
        assert_eq!(reader.read_u16().expect("read_u16 should succeed"), 0x1234);
    }

    #[test]
    fn test_read_u32() {
        let data = [0x12, 0x34, 0x56, 0x78];
        let mut reader = BitReader::new(&data);
        assert_eq!(
            reader.read_u32().expect("read_u32 should succeed"),
            0x1234_5678
        );
    }

    #[test]
    fn test_read_u64() {
        let data = [0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0];
        let mut reader = BitReader::new(&data);
        assert_eq!(
            reader.read_u64().expect("read_u64 should succeed"),
            0x1234_5678_9ABC_DEF0
        );
    }

    #[test]
    fn test_read_flag() {
        let data = [0b10100000];
        let mut reader = BitReader::new(&data);
        assert!(reader.read_flag().expect("read_flag should succeed"));
        assert!(!reader.read_flag().expect("read_flag should succeed"));
        assert!(reader.read_flag().expect("read_flag should succeed"));
    }

    #[test]
    fn test_skip_bits() {
        let data = [0xFF, 0x12];
        let mut reader = BitReader::new(&data);
        reader.skip_bits(8);
        assert_eq!(reader.read_u8().expect("read_u8 should succeed"), 0x12);
    }

    #[test]
    fn test_byte_align() {
        let data = [0xFF, 0x12];
        let mut reader = BitReader::new(&data);
        reader.read_bits(3).expect("read_bits should succeed");
        reader.byte_align();
        assert_eq!(reader.bits_read(), 8);
        assert_eq!(reader.read_u8().expect("read_u8 should succeed"), 0x12);
    }

    #[test]
    fn test_byte_align_already_aligned() {
        let data = [0xFF, 0x12];
        let mut reader = BitReader::new(&data);
        reader.byte_align();
        assert_eq!(reader.bits_read(), 0);
    }

    #[test]
    fn test_has_more_data() {
        let data = [0xFF];
        let mut reader = BitReader::new(&data);
        assert!(reader.has_more_data());
        reader.read_bits(8).expect("read_bits should succeed");
        assert!(!reader.has_more_data());
    }

    #[test]
    fn test_remaining_bytes() {
        let data = [0xFF, 0x00, 0xFF];
        let mut reader = BitReader::new(&data);
        assert_eq!(reader.remaining_bytes(), 3);
        reader.read_bits(4).expect("read_bits should succeed");
        assert_eq!(reader.remaining_bytes(), 2);
        reader.read_bits(4).expect("read_bits should succeed");
        assert_eq!(reader.remaining_bytes(), 2);
    }

    #[test]
    fn test_bits_read() {
        let data = [0xFF, 0x00];
        let mut reader = BitReader::new(&data);
        assert_eq!(reader.bits_read(), 0);
        reader.read_bits(5).expect("read_bits should succeed");
        assert_eq!(reader.bits_read(), 5);
        reader.read_bits(3).expect("read_bits should succeed");
        assert_eq!(reader.bits_read(), 8);
    }

    #[test]
    fn test_remaining_bits() {
        let data = [0xFF, 0x00];
        let mut reader = BitReader::new(&data);
        assert_eq!(reader.remaining_bits(), 16);
        reader.read_bits(5).expect("read_bits should succeed");
        assert_eq!(reader.remaining_bits(), 11);
    }

    #[test]
    fn test_peek_bit() {
        let data = [0b10000000];
        let mut reader = BitReader::new(&data);
        assert_eq!(reader.peek_bit().expect("peek_bit should succeed"), 1);
        assert_eq!(reader.peek_bit().expect("peek_bit should succeed"), 1);
        assert_eq!(reader.bits_read(), 0);
        reader.read_bit().expect("read_bit should succeed");
        assert_eq!(reader.peek_bit().expect("peek_bit should succeed"), 0);
    }

    #[test]
    fn test_eof() {
        let data = [0xFF];
        let mut reader = BitReader::new(&data);
        reader.read_bits(8).expect("read_bits should succeed");
        assert!(reader.read_bit().is_err());
        assert!(reader.read_bits(1).is_err());
        assert!(reader.peek_bit().is_err());
    }

    #[test]
    fn test_empty_data() {
        let data: [u8; 0] = [];
        let reader = BitReader::new(&data);
        assert!(!reader.has_more_data());
        assert_eq!(reader.remaining_bytes(), 0);
        assert_eq!(reader.remaining_bits(), 0);
    }

    // Additional comprehensive tests

    #[test]
    fn test_read_64_bits_max() {
        let data = [0xFF; 8];
        let mut reader = BitReader::new(&data);
        let value = reader.read_bits(64).expect("read_bits should succeed");
        assert_eq!(value, u64::MAX);
    }

    #[test]
    fn test_read_across_multiple_bytes() {
        // Test reading across 3 byte boundaries
        let data = [0b10101010, 0b11001100, 0b11110000];
        let mut reader = BitReader::new(&data);

        assert_eq!(
            reader.read_bits(4).expect("read_bits should succeed"),
            0b1010
        );
        assert_eq!(
            reader.read_bits(8).expect("read_bits should succeed"),
            0b1010_1100
        );
        assert_eq!(
            reader.read_bits(12).expect("read_bits should succeed"),
            0b1100_1111_0000
        );
    }

    #[test]
    fn test_mixed_read_operations() {
        let data = [0b11010010, 0b10110100];
        let mut reader = BitReader::new(&data);

        assert!(reader.read_flag().expect("read_flag should succeed")); // 1
        assert!(reader.read_flag().expect("read_flag should succeed")); // 1
        assert!(!reader.read_flag().expect("read_flag should succeed")); // 0
        assert_eq!(
            reader.read_bits(5).expect("read_bits should succeed"),
            0b10010
        ); // 10010
        assert_eq!(
            reader.read_u8().expect("read_u8 should succeed"),
            0b10110100
        );
    }

    #[test]
    fn test_byte_align_at_boundary() {
        let data = [0xFF, 0x12, 0x34];
        let mut reader = BitReader::new(&data);

        reader.byte_align(); // Should do nothing
        assert_eq!(reader.bits_read(), 0);

        reader.read_bits(8).expect("read_bits should succeed");
        reader.byte_align(); // Should still do nothing
        assert_eq!(reader.bits_read(), 8);
    }

    #[test]
    fn test_skip_bits_partial_byte() {
        let data = [0xFF, 0x12];
        let mut reader = BitReader::new(&data);

        reader.skip_bits(3);
        assert_eq!(
            reader.read_bits(5).expect("read_bits should succeed"),
            0b11111
        );
        assert_eq!(reader.read_u8().expect("read_u8 should succeed"), 0x12);
    }

    #[test]
    fn test_skip_bits_beyond_end() {
        let data = [0xFF];
        let mut reader = BitReader::new(&data);

        reader.skip_bits(100); // Should stop at end
        assert!(!reader.has_more_data());
    }

    #[test]
    fn test_remaining_methods_consistency() {
        let data = [0xFF, 0x00, 0xFF, 0x00];
        let mut reader = BitReader::new(&data);

        assert_eq!(reader.remaining_bits(), 32);
        assert_eq!(reader.remaining_bytes(), 4);

        reader.read_bits(5).expect("read_bits should succeed");
        assert_eq!(reader.remaining_bits(), 27);
        assert_eq!(reader.remaining_bytes(), 3);

        reader.read_bits(11).expect("read_bits should succeed"); // Total 16 bits = 2 bytes
        assert_eq!(reader.remaining_bits(), 16);
        assert_eq!(reader.remaining_bytes(), 2);
    }

    #[test]
    fn test_peek_doesnt_consume() {
        let data = [0b10110100];
        let mut reader = BitReader::new(&data);

        for _ in 0..10 {
            assert_eq!(reader.peek_bit().expect("peek_bit should succeed"), 1);
        }
        assert_eq!(reader.bits_read(), 0);

        reader.read_bit().expect("read_bit should succeed");
        for _ in 0..10 {
            assert_eq!(reader.peek_bit().expect("peek_bit should succeed"), 0);
        }
        assert_eq!(reader.bits_read(), 1);
    }

    #[test]
    fn test_read_all_integer_types() {
        // Test reading all integer types in sequence
        let data = [
            0x12, // u8
            0x34, 0x56, // u16
            0x78, 0x9A, 0xBC, 0xDE, // u32
            0xF0, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, // u64
        ];
        let mut reader = BitReader::new(&data);

        assert_eq!(reader.read_u8().expect("read_u8 should succeed"), 0x12);
        assert_eq!(reader.read_u16().expect("read_u16 should succeed"), 0x3456);
        assert_eq!(
            reader.read_u32().expect("read_u32 should succeed"),
            0x789A_BCDE
        );
        assert_eq!(
            reader.read_u64().expect("read_u64 should succeed"),
            0xF011_2233_4455_6677
        );
        assert!(!reader.has_more_data());
    }

    #[test]
    fn test_unaligned_integer_reads() {
        let data = [0xFF, 0x12, 0x34, 0x56, 0x78];
        let mut reader = BitReader::new(&data);

        reader.read_bits(4).expect("read_bits should succeed"); // Unalign by 4 bits

        // Now all integer reads should work across byte boundaries
        assert_eq!(reader.read_bits(8).expect("read_bits should succeed"), 0xF1);
        assert_eq!(
            reader.read_bits(16).expect("read_bits should succeed"),
            0x2345
        );
        assert_eq!(reader.read_bits(8).expect("read_bits should succeed"), 0x67);
    }

    #[test]
    fn test_position_tracking() {
        let data = [0xFF, 0x00, 0xFF];
        let mut reader = BitReader::new(&data);

        assert_eq!(reader.byte_position(), 0);
        assert_eq!(reader.bit_position(), 0);

        reader.read_bits(10).expect("read_bits should succeed");
        assert_eq!(reader.byte_position(), 1);
        assert_eq!(reader.bit_position(), 2);

        reader.byte_align();
        assert_eq!(reader.byte_position(), 2);
        assert_eq!(reader.bit_position(), 0);
    }

    #[test]
    fn test_data_accessor() {
        let data = [0xFF, 0x00, 0xFF];
        let reader = BitReader::new(&data);

        assert_eq!(reader.data(), &data);
        assert_eq!(reader.data().len(), 3);
    }

    #[test]
    fn test_single_bit_pattern() {
        // Test reading alternating bit pattern
        let data = [0b10101010];
        let mut reader = BitReader::new(&data);

        for i in 0..8 {
            let expected = if i % 2 == 0 { 1 } else { 0 };
            assert_eq!(
                reader.read_bit().expect("read_bit should succeed"),
                expected
            );
        }
    }

    #[test]
    fn test_eof_on_exact_boundary() {
        let data = [0xFF];
        let mut reader = BitReader::new(&data);

        reader.read_bits(8).expect("read_bits should succeed");
        assert!(!reader.has_more_data());
        assert_eq!(reader.remaining_bits(), 0);

        let result = reader.read_bit();
        assert!(result.is_err());
    }
}
