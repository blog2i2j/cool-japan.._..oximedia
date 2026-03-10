//! Range decoder for Opus entropy coding.
//!
//! Opus uses a range coder for entropy coding, which is similar to arithmetic
//! coding but uses simpler operations. This implementation follows RFC 6716.

use crate::{CodecError, CodecResult};

/// Range decoder state for Opus entropy coding.
///
/// The range decoder maintains a current range [low, low+range) and reads
/// bits from the input to determine which part of the range the coded value
/// falls into.
#[derive(Debug)]
pub struct RangeDecoder<'a> {
    /// Input data buffer
    data: &'a [u8],
    /// Current position in the buffer
    pos: usize,
    /// Range value (23 bits + 1 guard bit)
    rng: u32,
    /// Value within the range (23 bits + 1 guard bit)
    val: u32,
    /// Number of buffered bits
    #[allow(dead_code)]
    bits_left: u32,
}

impl<'a> RangeDecoder<'a> {
    /// Creates a new range decoder from input data.
    ///
    /// # Arguments
    ///
    /// * `data` - Input bitstream data
    pub fn new(data: &'a [u8]) -> CodecResult<Self> {
        if data.is_empty() {
            return Err(CodecError::InvalidData(
                "Range decoder requires non-empty input".to_string(),
            ));
        }

        let mut decoder = Self {
            data,
            pos: 0,
            rng: 128,
            val: 127 - (data[0] >> 1) as u32,
            bits_left: 0,
        };

        // Normalize initial state
        decoder.normalize()?;

        Ok(decoder)
    }

    /// Decodes a symbol with uniform probability distribution.
    ///
    /// # Arguments
    ///
    /// * `n` - Number of symbols (1..n-1)
    pub fn decode_uniform(&mut self, n: u32) -> CodecResult<u32> {
        if n <= 1 {
            return Ok(0);
        }

        let ft = n;
        let s = self.rng / ft;
        if s == 0 {
            return Ok(0);
        }

        let fl = self.val / s;
        let symbol = ft
            .saturating_sub(1)
            .saturating_sub(fl.min(ft.saturating_sub(1)));

        if symbol < ft {
            self.val = self
                .val
                .saturating_sub(s.saturating_mul(ft.saturating_sub(symbol)));
        }

        self.rng = if symbol < ft.saturating_sub(1) {
            s
        } else {
            self.rng
                .saturating_sub(s.saturating_mul(ft.saturating_sub(1)))
        };

        self.normalize()?;

        Ok(symbol)
    }

    /// Decodes a symbol with given cumulative distribution.
    ///
    /// # Arguments
    ///
    /// * `cdf` - Cumulative distribution function (must be sorted)
    /// * `total` - Total probability mass
    pub fn decode_cdf(&mut self, cdf: &[u16], total: u32) -> CodecResult<u32> {
        if cdf.is_empty() {
            return Err(CodecError::InvalidData("Empty CDF".to_string()));
        }

        let ft = total;
        let s = self.rng / ft;
        let fl = self.val / s;

        // Binary search in CDF
        let mut symbol = 0;
        for (i, &prob) in cdf.iter().enumerate() {
            if fl < u32::from(prob) {
                symbol = i;
                break;
            }
        }

        let fl_curr = if symbol > 0 {
            u32::from(cdf[symbol - 1])
        } else {
            0
        };
        let fl_next = u32::from(cdf[symbol]);

        self.val -= s * fl_curr;
        self.rng = s * (fl_next - fl_curr);

        self.normalize()?;

        Ok(symbol as u32)
    }

    /// Decodes a single bit with given probability.
    ///
    /// # Arguments
    ///
    /// * `prob` - Probability of bit being 0 (0..32768)
    pub fn decode_bit(&mut self, prob: u32) -> CodecResult<bool> {
        let split = 1 + (self.rng.saturating_sub(1).saturating_mul(prob) >> 15);

        let bit = if self.val < split {
            self.rng = split;
            false
        } else {
            self.val = self.val.saturating_sub(split);
            self.rng = self.rng.saturating_sub(split);
            true
        };

        self.normalize()?;

        Ok(bit)
    }

    /// Decodes a logarithmic value.
    ///
    /// # Arguments
    ///
    /// * `max_value` - Maximum value to decode
    pub fn decode_log(&mut self, max_value: u32) -> CodecResult<u32> {
        if max_value <= 1 {
            return Ok(0);
        }

        let bits = 32 - max_value.leading_zeros() - 1;
        let mut value = 0;

        for _ in 0..bits {
            let bit = self.decode_bit(16384)?;
            value = (value << 1) | u32::from(bit);
        }

        if value >= max_value {
            value = max_value - 1;
        }

        Ok(value)
    }

    /// Decodes unsigned integer value.
    ///
    /// # Arguments
    ///
    /// * `bits` - Number of bits to decode
    pub fn decode_uint(&mut self, bits: u32) -> CodecResult<u32> {
        let mut value = 0;
        for _ in 0..bits {
            let bit = self.decode_bit(16384)?;
            value = (value << 1) | u32::from(bit);
        }
        Ok(value)
    }

    /// Decodes signed integer value.
    ///
    /// # Arguments
    ///
    /// * `bits` - Number of bits to decode (including sign bit)
    pub fn decode_int(&mut self, bits: u32) -> CodecResult<i32> {
        if bits == 0 {
            return Ok(0);
        }

        let magnitude = self.decode_uint(bits - 1)?;
        let sign = self.decode_bit(16384)?;

        Ok(if sign {
            -(magnitude as i32)
        } else {
            magnitude as i32
        })
    }

    /// Normalizes the range decoder state.
    ///
    /// Ensures the range is at least 2^8 by reading new bits from the input.
    fn normalize(&mut self) -> CodecResult<()> {
        while self.rng <= 0x0080_0000 {
            self.val = (self.val << 8) | u32::from(self.read_byte()?);
            self.rng <<= 8;
        }
        Ok(())
    }

    /// Reads a single byte from the input.
    fn read_byte(&mut self) -> CodecResult<u8> {
        if self.pos >= self.data.len() {
            return Ok(0); // Return 0 for padding
        }

        let byte = self.data[self.pos];
        self.pos += 1;

        Ok(byte)
    }

    /// Returns the number of bytes consumed so far.
    #[must_use]
    pub const fn bytes_consumed(&self) -> usize {
        self.pos
    }

    /// Returns the remaining range value.
    #[must_use]
    pub const fn remaining_range(&self) -> u32 {
        self.rng
    }

    /// Checks if decoder has more data available.
    #[must_use]
    pub const fn has_more_data(&self) -> bool {
        self.pos < self.data.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_range_decoder_creation() {
        let data = vec![0x80, 0x00, 0x00, 0x00];
        let decoder = RangeDecoder::new(&data);
        assert!(decoder.is_ok());
    }

    #[test]
    fn test_decode_uniform() {
        let data = vec![0x80, 0x00, 0x00, 0x00];
        let mut decoder = RangeDecoder::new(&data).expect("should succeed");
        let symbol = decoder.decode_uniform(4);
        assert!(symbol.is_ok());
        assert!(symbol.expect("should succeed") < 4);
    }

    #[test]
    fn test_decode_bit() {
        let data = vec![0xFF, 0xFF, 0xFF, 0xFF];
        let mut decoder = RangeDecoder::new(&data).expect("should succeed");
        let bit = decoder.decode_bit(16384);
        assert!(bit.is_ok());
    }

    #[test]
    fn test_empty_data() {
        let data: Vec<u8> = vec![];
        let decoder = RangeDecoder::new(&data);
        assert!(decoder.is_err());
    }
}
