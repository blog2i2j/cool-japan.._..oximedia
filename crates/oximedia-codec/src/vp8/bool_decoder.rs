//! VP8 Boolean arithmetic decoder.
//!
//! This module implements the boolean arithmetic coder used in VP8.
//! VP8 uses a range coder that encodes binary symbols with given probabilities.
//! See RFC 6386 Section 7 for the specification.

use crate::error::{CodecError, CodecResult};

/// Boolean arithmetic decoder for VP8.
///
/// The VP8 boolean decoder is a range coder that decodes binary symbols
/// with specified probabilities. It maintains a `range` and `value` that
/// are updated as symbols are decoded.
///
/// # Examples
///
/// ```
/// use oximedia_codec::vp8::BoolDecoder;
///
/// let data = [0x00, 0x01, 0x02, 0x03];
/// let mut decoder = BoolDecoder::new(&data)?;
///
/// // Read a boolean with 50% probability
/// let bit = decoder.read_bit()?;
/// # Ok::<(), oximedia_codec::error::CodecError>(())
/// ```
pub struct BoolDecoder<'a> {
    /// Input data buffer.
    data: &'a [u8],
    /// Current read position in bytes.
    pos: usize,
    /// Current range (0-255).
    range: u32,
    /// Current value being decoded.
    value: u32,
    /// Number of bits left in the current value.
    bits_left: i32,
}

impl<'a> BoolDecoder<'a> {
    /// Creates a new boolean decoder.
    ///
    /// # Arguments
    ///
    /// * `data` - The compressed data to decode
    ///
    /// # Errors
    ///
    /// Returns an error if the data is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_codec::vp8::BoolDecoder;
    ///
    /// let data = [0x00, 0x01, 0x02];
    /// let decoder = BoolDecoder::new(&data)?;
    /// assert!(decoder.bytes_consumed() > 0);
    /// # Ok::<(), oximedia_codec::error::CodecError>(())
    /// ```
    pub fn new(data: &'a [u8]) -> CodecResult<Self> {
        if data.is_empty() {
            return Err(CodecError::InvalidBitstream(
                "Empty data for boolean decoder".to_string(),
            ));
        }

        let mut decoder = Self {
            data,
            pos: 0,
            range: 255,
            value: 0,
            bits_left: 0,
        };

        // Initialize with first bytes
        decoder.fill()?;

        Ok(decoder)
    }

    /// Fills the value buffer with more bits from the input.
    #[allow(clippy::unnecessary_wraps)]
    fn fill(&mut self) -> CodecResult<()> {
        while self.bits_left < 8 {
            if self.pos < self.data.len() {
                self.value = (self.value << 8) | u32::from(self.data[self.pos]);
                self.pos += 1;
            } else {
                self.value <<= 8;
            }
            self.bits_left += 8;
        }
        Ok(())
    }

    /// Reads a single boolean with the given probability.
    ///
    /// The probability is specified as a value from 0-255, where
    /// 0 means the bit is almost certainly 0, and 255 means it is
    /// almost certainly 1.
    ///
    /// # Arguments
    ///
    /// * `prob` - Probability that the bit is 0 (0-255)
    ///
    /// # Errors
    ///
    /// Returns an error if decoding fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_codec::vp8::BoolDecoder;
    ///
    /// let data = [0x00, 0x01, 0x02, 0x03];
    /// let mut decoder = BoolDecoder::new(&data)?;
    ///
    /// // Read with 128 (50%) probability
    /// let bit = decoder.read_bool(128)?;
    /// # Ok::<(), oximedia_codec::error::CodecError>(())
    /// ```
    pub fn read_bool(&mut self, prob: u8) -> CodecResult<bool> {
        let split = 1 + (((self.range - 1) * u32::from(prob)) >> 8);
        let bigsplit = split << 8;

        let result = if self.value >= bigsplit {
            self.range -= split;
            self.value -= bigsplit;
            true
        } else {
            self.range = split;
            false
        };

        // Normalize the range
        while self.range < 128 {
            self.range <<= 1;
            self.value <<= 1;
            self.bits_left -= 1;

            if self.bits_left < 0 {
                self.fill()?;
            }
        }

        Ok(result)
    }

    /// Reads a boolean with 50% probability.
    ///
    /// This is equivalent to `read_bool(128)`.
    ///
    /// # Errors
    ///
    /// Returns an error if decoding fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_codec::vp8::BoolDecoder;
    ///
    /// let data = [0x00, 0x01, 0x02, 0x03];
    /// let mut decoder = BoolDecoder::new(&data)?;
    /// let bit = decoder.read_bit()?;
    /// # Ok::<(), oximedia_codec::error::CodecError>(())
    /// ```
    #[inline]
    pub fn read_bit(&mut self) -> CodecResult<bool> {
        self.read_bool(128)
    }

    /// Reads an unsigned integer of n bits.
    ///
    /// Reads n bits from the stream and returns them as an unsigned integer.
    /// Bits are read most-significant-bit first.
    ///
    /// # Arguments
    ///
    /// * `n` - Number of bits to read (1-32)
    ///
    /// # Errors
    ///
    /// Returns an error if decoding fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_codec::vp8::BoolDecoder;
    ///
    /// let data = [0xFF, 0x00, 0xFF, 0x00];
    /// let mut decoder = BoolDecoder::new(&data)?;
    ///
    /// // Read an 8-bit value
    /// let value = decoder.read_literal(8)?;
    /// # Ok::<(), oximedia_codec::error::CodecError>(())
    /// ```
    pub fn read_literal(&mut self, n: u8) -> CodecResult<u32> {
        let mut value = 0u32;
        for _ in 0..n {
            value = (value << 1) | u32::from(self.read_bit()?);
        }
        Ok(value)
    }

    /// Reads a signed integer of n bits.
    ///
    /// Reads n bits as an unsigned value, then reads a sign bit.
    /// If the sign bit is 1, the value is negated.
    ///
    /// # Arguments
    ///
    /// * `n` - Number of magnitude bits to read (1-31)
    ///
    /// # Errors
    ///
    /// Returns an error if decoding fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_codec::vp8::BoolDecoder;
    ///
    /// let data = [0xFF, 0xFF, 0xFF, 0xFF];
    /// let mut decoder = BoolDecoder::new(&data)?;
    ///
    /// // Read a signed 4-bit value
    /// let value = decoder.read_signed_literal(4)?;
    /// # Ok::<(), oximedia_codec::error::CodecError>(())
    /// ```
    #[allow(clippy::cast_possible_wrap)]
    pub fn read_signed_literal(&mut self, n: u8) -> CodecResult<i32> {
        let value = self.read_literal(n)? as i32;
        let sign = self.read_bit()?;
        if sign {
            Ok(-value)
        } else {
            Ok(value)
        }
    }

    /// Returns the number of bytes consumed from the input.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_codec::vp8::BoolDecoder;
    ///
    /// let data = [0x00, 0x01, 0x02];
    /// let decoder = BoolDecoder::new(&data)?;
    /// assert!(decoder.bytes_consumed() > 0);
    /// # Ok::<(), oximedia_codec::error::CodecError>(())
    /// ```
    #[must_use]
    pub const fn bytes_consumed(&self) -> usize {
        self.pos
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bool_decoder_new() {
        let data = [0x00, 0x01, 0x02];
        let decoder = BoolDecoder::new(&data).expect("should succeed");
        // fill() consumes bytes until bits_left >= 8
        assert!(decoder.bytes_consumed() >= 1);
    }

    #[test]
    fn test_empty_data() {
        let data: [u8; 0] = [];
        assert!(BoolDecoder::new(&data).is_err());
    }

    #[test]
    fn test_read_bit() {
        let data = [0x00, 0x00, 0x00, 0x00];
        let mut decoder = BoolDecoder::new(&data).expect("should succeed");

        // Reading bits should not panic
        for _ in 0..16 {
            let _ = decoder.read_bit();
        }
    }

    #[test]
    fn test_read_literal() {
        let data = [0xFF, 0xFF, 0xFF, 0xFF];
        let mut decoder = BoolDecoder::new(&data).expect("should succeed");

        // Read some bits
        let _ = decoder.read_literal(8);
    }

    #[test]
    fn test_read_signed_literal() {
        let data = [0xFF, 0xFF, 0xFF, 0xFF];
        let mut decoder = BoolDecoder::new(&data).expect("should succeed");

        // Read a signed value
        let _ = decoder.read_signed_literal(4);
    }
}
