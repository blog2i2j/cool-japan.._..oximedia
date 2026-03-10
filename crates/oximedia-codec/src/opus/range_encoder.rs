//! Range encoder for Opus entropy coding.
//!
//! Opus uses a range coder for entropy coding, which is similar to arithmetic
//! coding but uses simpler operations. This implementation follows RFC 6716.
//!
//! The range encoder is the inverse of the range decoder, maintaining a range
//! [low, low+range) and encoding symbols by subdividing the range based on
//! their probability distributions.

use crate::{CodecError, CodecResult};

/// Range encoder state for Opus entropy coding.
///
/// The range encoder maintains a current range [low, low+range) and writes
/// bits to the output as symbols are encoded.
#[derive(Debug)]
pub struct RangeEncoder {
    /// Output buffer
    buffer: Vec<u8>,
    /// Range value (23 bits + 1 guard bit)
    rng: u32,
    /// Low value (bottom of the range)
    low: u32,
    /// Number of bits written
    bits_written: usize,
    /// Temporary bit buffer
    #[allow(dead_code)]
    bit_buffer: u32,
    /// Number of bits in bit_buffer
    #[allow(dead_code)]
    bit_count: u32,
}

impl RangeEncoder {
    /// Creates a new range encoder.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Initial capacity for output buffer
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            rng: 0x0080_0000, // Initialize to 2^23
            low: 0,
            bits_written: 0,
            bit_buffer: 0,
            bit_count: 0,
        }
    }

    /// Encodes a symbol with uniform probability distribution.
    ///
    /// # Arguments
    ///
    /// * `symbol` - Symbol to encode (0..n-1)
    /// * `n` - Number of symbols (1..n-1)
    pub fn encode_uniform(&mut self, symbol: u32, n: u32) -> CodecResult<()> {
        if n <= 1 {
            return Ok(());
        }

        if symbol >= n {
            return Err(CodecError::InvalidData(format!(
                "Symbol {symbol} out of range 0..{n}"
            )));
        }

        let ft = n;
        let s = self.rng / ft;

        self.low += s * (ft - symbol - 1);
        self.rng = if symbol < ft - 1 {
            s
        } else {
            self.rng - s * (ft - 1)
        };

        self.normalize()?;

        Ok(())
    }

    /// Encodes a symbol with given cumulative distribution.
    ///
    /// # Arguments
    ///
    /// * `symbol` - Symbol index to encode
    /// * `cdf` - Cumulative distribution function (must be sorted)
    /// * `total` - Total probability mass
    pub fn encode_cdf(&mut self, symbol: usize, cdf: &[u16], total: u32) -> CodecResult<()> {
        if cdf.is_empty() {
            return Err(CodecError::InvalidData("Empty CDF".to_string()));
        }

        if symbol >= cdf.len() {
            return Err(CodecError::InvalidData(format!(
                "Symbol {symbol} out of range for CDF length {}",
                cdf.len()
            )));
        }

        let ft = total;
        let s = self.rng / ft;

        let fl_curr = if symbol > 0 {
            u32::from(cdf[symbol - 1])
        } else {
            0
        };
        let fl_next = u32::from(cdf[symbol]);

        self.low += s * fl_curr;
        self.rng = s * (fl_next - fl_curr);

        self.normalize()?;

        Ok(())
    }

    /// Encodes a single bit with given probability.
    ///
    /// # Arguments
    ///
    /// * `bit` - Bit value to encode (false=0, true=1)
    /// * `prob` - Probability of bit being 0 (0..32768)
    pub fn encode_bit(&mut self, bit: bool, prob: u32) -> CodecResult<()> {
        let split = 1 + ((u64::from(self.rng - 1) * u64::from(prob)) >> 15) as u32;

        if !bit {
            self.rng = split;
        } else {
            self.low += split;
            self.rng -= split;
        }

        self.normalize()?;

        Ok(())
    }

    /// Encodes a logarithmic value.
    ///
    /// # Arguments
    ///
    /// * `value` - Value to encode
    /// * `max_value` - Maximum value
    pub fn encode_log(&mut self, value: u32, max_value: u32) -> CodecResult<()> {
        if max_value <= 1 {
            return Ok(());
        }

        let val = value.min(max_value - 1);
        let bits = 32 - max_value.leading_zeros() - 1;

        for i in (0..bits).rev() {
            let bit = (val >> i) & 1 != 0;
            self.encode_bit(bit, 16384)?;
        }

        Ok(())
    }

    /// Encodes unsigned integer value.
    ///
    /// # Arguments
    ///
    /// * `value` - Value to encode
    /// * `bits` - Number of bits to encode
    pub fn encode_uint(&mut self, value: u32, bits: u32) -> CodecResult<()> {
        for i in (0..bits).rev() {
            let bit = (value >> i) & 1 != 0;
            self.encode_bit(bit, 16384)?;
        }
        Ok(())
    }

    /// Encodes signed integer value.
    ///
    /// # Arguments
    ///
    /// * `value` - Value to encode
    /// * `bits` - Number of bits to encode (including sign bit)
    pub fn encode_int(&mut self, value: i32, bits: u32) -> CodecResult<()> {
        if bits == 0 {
            return Ok(());
        }

        let magnitude = value.unsigned_abs();
        let sign = value < 0;

        self.encode_uint(magnitude, bits - 1)?;
        self.encode_bit(sign, 16384)?;

        Ok(())
    }

    /// Normalizes the range encoder state.
    ///
    /// Ensures the range is at least 2^8 by writing bits to the output.
    fn normalize(&mut self) -> CodecResult<()> {
        while self.rng <= 0x0080_0000 {
            let byte = ((self.low >> 23) & 0xFF) as u8;
            self.buffer.push(byte);
            self.bits_written += 8;
            self.low = (self.low << 8) & 0x7FFF_FFFF;
            self.rng <<= 8;
        }
        Ok(())
    }

    /// Finalizes encoding and returns the compressed data.
    ///
    /// This must be called after all symbols have been encoded.
    pub fn finalize(mut self) -> CodecResult<Vec<u8>> {
        // Flush remaining bits
        let nbits = (32 - self.rng.leading_zeros()).saturating_sub(9);
        let val = (self.low >> (23 - nbits)) + (self.rng >> (23 - nbits + 1));

        // Write final bytes
        for i in (0..((nbits + 7) / 8)).rev() {
            let byte = ((val >> (i * 8)) & 0xFF) as u8;
            self.buffer.push(byte);
        }

        Ok(self.buffer)
    }

    /// Returns the number of bytes written so far.
    #[must_use]
    pub fn bytes_written(&self) -> usize {
        self.buffer.len()
    }

    /// Returns the number of bits written.
    #[must_use]
    pub const fn bits_written(&self) -> usize {
        self.bits_written
    }

    /// Returns a reference to the current buffer (incomplete encoding).
    #[must_use]
    pub fn buffer(&self) -> &[u8] {
        &self.buffer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_range_encoder_creation() {
        let encoder = RangeEncoder::new(1024);
        assert_eq!(encoder.bytes_written(), 0);
    }

    #[test]
    fn test_encode_uniform() {
        let mut encoder = RangeEncoder::new(1024);
        let result = encoder.encode_uniform(2, 4);
        assert!(result.is_ok());
    }

    #[test]
    fn test_encode_bit() {
        let mut encoder = RangeEncoder::new(1024);
        let result = encoder.encode_bit(true, 16384);
        assert!(result.is_ok());
    }

    #[test]
    fn test_encode_uint() {
        let mut encoder = RangeEncoder::new(1024);
        let result = encoder.encode_uint(42, 8);
        assert!(result.is_ok());
    }

    #[test]
    fn test_encode_int() {
        let mut encoder = RangeEncoder::new(1024);
        let result = encoder.encode_int(-5, 4);
        assert!(result.is_ok());
    }

    #[test]
    fn test_finalize() {
        let mut encoder = RangeEncoder::new(1024);
        encoder.encode_uniform(1, 4).expect("should succeed");
        let data = encoder.finalize();
        assert!(data.is_ok());
        assert!(!data.expect("should succeed").is_empty());
    }

    #[test]
    fn test_invalid_symbol() {
        let mut encoder = RangeEncoder::new(1024);
        let result = encoder.encode_uniform(5, 4);
        assert!(result.is_err());
    }
}
