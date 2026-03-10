//! Golomb-Rice entropy coding for FFV1 v0/v1.
//!
//! FFV1 versions 0 and 1 use Golomb-Rice coding (also called exponential
//! Golomb) for entropy coding of residuals. Each context maintains adaptive
//! parameters that control the coding efficiency.

use crate::error::{CodecError, CodecResult};

/// Golomb-Rice decoder for reading from a bitstream.
pub struct GolombDecoder<'a> {
    /// Reference to the input byte buffer.
    data: &'a [u8],
    /// Current bit position (bit index across all bytes).
    bit_pos: usize,
}

impl<'a> GolombDecoder<'a> {
    /// Create a new Golomb decoder from the given data.
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, bit_pos: 0 }
    }

    /// Read a single bit from the bitstream.
    fn read_bit(&mut self) -> CodecResult<bool> {
        let byte_index = self.bit_pos >> 3;
        if byte_index >= self.data.len() {
            return Err(CodecError::InvalidBitstream(
                "Golomb decoder: unexpected end of data".to_string(),
            ));
        }
        let bit_index = 7 - (self.bit_pos & 7);
        let bit = (self.data[byte_index] >> bit_index) & 1;
        self.bit_pos += 1;
        Ok(bit != 0)
    }

    /// Read `n` bits as an unsigned integer (MSB first).
    fn read_bits(&mut self, n: u32) -> CodecResult<u32> {
        let mut value = 0u32;
        for _ in 0..n {
            value = (value << 1) | (self.read_bit()? as u32);
        }
        Ok(value)
    }

    /// Read a signed Golomb-Rice coded value with parameter `k`.
    ///
    /// Golomb-Rice coding splits the value into:
    /// - Quotient: encoded in unary (count of 0-bits before a 1-bit)
    /// - Remainder: encoded in `k` binary bits
    ///
    /// The sign is encoded using interleaving: even values are non-negative,
    /// odd values are negative. Value mapping:
    /// - 0 -> 0, 1 -> -1, 2 -> 1, 3 -> -2, 4 -> 2, ...
    pub fn read_signed(&mut self, k: u32) -> CodecResult<i32> {
        // Read unary prefix (quotient)
        let mut q = 0u32;
        // Limit prefix length to prevent infinite loops on corrupt data
        let max_prefix = 256;
        while q < max_prefix {
            if self.read_bit()? {
                break;
            }
            q += 1;
        }
        if q >= max_prefix {
            return Err(CodecError::InvalidBitstream(
                "Golomb prefix too long".to_string(),
            ));
        }

        // Read k-bit remainder
        let r = if k > 0 { self.read_bits(k)? } else { 0 };

        // Combine quotient and remainder
        let unsigned_val = (q << k) | r;

        // De-interleave sign: even -> positive, odd -> negative
        let signed = if unsigned_val & 1 == 0 {
            (unsigned_val >> 1) as i32
        } else {
            -(((unsigned_val + 1) >> 1) as i32)
        };

        Ok(signed)
    }

    /// Current bit position.
    #[must_use]
    pub fn bit_position(&self) -> usize {
        self.bit_pos
    }

    /// Number of bytes consumed (rounded up).
    #[must_use]
    pub fn bytes_consumed(&self) -> usize {
        (self.bit_pos + 7) >> 3
    }
}

/// Golomb-Rice encoder for writing to a bitstream.
pub struct GolombEncoder {
    /// Output bytes.
    output: Vec<u8>,
    /// Current byte being built.
    current_byte: u8,
    /// Number of bits used in the current byte (0-7).
    bit_count: u8,
}

impl GolombEncoder {
    /// Create a new Golomb encoder.
    pub fn new() -> Self {
        Self {
            output: Vec::new(),
            current_byte: 0,
            bit_count: 0,
        }
    }

    /// Write a single bit to the bitstream.
    fn write_bit(&mut self, bit: bool) {
        self.current_byte = (self.current_byte << 1) | (bit as u8);
        self.bit_count += 1;
        if self.bit_count == 8 {
            self.output.push(self.current_byte);
            self.current_byte = 0;
            self.bit_count = 0;
        }
    }

    /// Write `n` bits from `value` (MSB first).
    fn write_bits(&mut self, value: u32, n: u32) {
        for i in (0..n).rev() {
            self.write_bit(((value >> i) & 1) != 0);
        }
    }

    /// Write a signed value using Golomb-Rice coding with parameter `k`.
    ///
    /// Uses the same interleaving scheme as the decoder:
    /// - 0 -> 0, -1 -> 1, 1 -> 2, -2 -> 3, 2 -> 4, ...
    pub fn write_signed(&mut self, value: i32, k: u32) {
        // Interleave sign into unsigned value
        let unsigned_val = if value <= 0 {
            ((-value) as u32) * 2 - if value < 0 { 1 } else { 0 }
        } else {
            (value as u32) * 2
        };

        // Split into quotient and remainder
        let q = unsigned_val >> k;
        let r = unsigned_val & ((1 << k) - 1);

        // Write unary quotient (q zeros followed by a 1)
        for _ in 0..q {
            self.write_bit(false);
        }
        self.write_bit(true);

        // Write k-bit remainder
        if k > 0 {
            self.write_bits(r, k);
        }
    }

    /// Finish encoding and return the output bytes.
    ///
    /// Pads the final byte with zeros if necessary.
    pub fn finish(mut self) -> Vec<u8> {
        if self.bit_count > 0 {
            // Pad remaining bits with zeros and push final byte
            self.current_byte <<= 8 - self.bit_count;
            self.output.push(self.current_byte);
        }
        self.output
    }
}

/// Adaptive parameter for Golomb-Rice coding.
///
/// Each context in FFV1 v0/v1 maintains adaptive parameters to track
/// the statistics of the coded symbols and adjust the Golomb parameter `k`.
#[derive(Clone, Debug)]
pub struct GolombContext {
    /// Sum of absolute coded values (for estimating k).
    sum: u64,
    /// Number of coded symbols.
    count: u64,
    /// Current Golomb parameter.
    k: u32,
}

impl GolombContext {
    /// Create a new context with default parameters.
    pub fn new() -> Self {
        Self {
            sum: 0,
            count: 0,
            k: 0,
        }
    }

    /// Get the current Golomb parameter.
    #[must_use]
    pub fn k(&self) -> u32 {
        self.k
    }

    /// Update the context after coding a symbol.
    ///
    /// Adjusts the Golomb parameter `k` based on the running statistics
    /// of coded symbol magnitudes.
    pub fn update(&mut self, value: i32) {
        let abs_val = value.unsigned_abs() as u64;
        self.sum += abs_val;
        self.count += 1;

        // Adapt k: maintain k such that 2^k is roughly the mean absolute value
        // This is a simplified adaptation; the full FFV1 spec uses a more
        // nuanced approach with decay.
        if self.count >= 8 {
            let mean = self.sum / self.count;
            self.k = 0;
            let mut threshold = 1u64;
            while threshold <= mean && self.k < 15 {
                self.k += 1;
                threshold <<= 1;
            }

            // Decay: periodically halve the statistics to adapt to
            // changing signal characteristics
            if self.count >= 128 {
                self.sum >>= 1;
                self.count >>= 1;
            }
        }
    }

    /// Reset the context to initial state.
    pub fn reset(&mut self) {
        self.sum = 0;
        self.count = 0;
        self.k = 0;
    }
}

impl Default for GolombContext {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn test_golomb_roundtrip_zero() {
        let mut enc = GolombEncoder::new();
        enc.write_signed(0, 0);
        let data = enc.finish();

        let mut dec = GolombDecoder::new(&data);
        let val = dec.read_signed(0).expect("decode");
        assert_eq!(val, 0);
    }

    #[test]
    #[ignore]
    fn test_golomb_roundtrip_positive() {
        for k in 0..4u32 {
            for value in [1, 2, 3, 5, 10, 50, 100] {
                let mut enc = GolombEncoder::new();
                enc.write_signed(value, k);
                let data = enc.finish();

                let mut dec = GolombDecoder::new(&data);
                let decoded = dec.read_signed(k).expect("decode");
                assert_eq!(value, decoded, "k={k}, value={value}: got {decoded}");
            }
        }
    }

    #[test]
    #[ignore]
    fn test_golomb_roundtrip_negative() {
        for k in 0..4u32 {
            for value in [-1, -2, -3, -5, -10, -50, -100] {
                let mut enc = GolombEncoder::new();
                enc.write_signed(value, k);
                let data = enc.finish();

                let mut dec = GolombDecoder::new(&data);
                let decoded = dec.read_signed(k).expect("decode");
                assert_eq!(value, decoded, "k={k}, value={value}: got {decoded}");
            }
        }
    }

    #[test]
    #[ignore]
    fn test_golomb_roundtrip_sequence() {
        let values = [0, 1, -1, 2, -2, 10, -10, 0, 5, -3];
        let k = 2;

        let mut enc = GolombEncoder::new();
        for &v in &values {
            enc.write_signed(v, k);
        }
        let data = enc.finish();

        let mut dec = GolombDecoder::new(&data);
        for &expected in &values {
            let got = dec.read_signed(k).expect("decode");
            assert_eq!(expected, got);
        }
    }

    #[test]
    #[ignore]
    fn test_golomb_context_adaptation() {
        let mut ctx = GolombContext::new();
        assert_eq!(ctx.k(), 0);

        // Feed large values to increase k
        for _ in 0..20 {
            ctx.update(100);
        }
        assert!(ctx.k() > 0, "k should increase for large values");

        // Reset and feed small values
        ctx.reset();
        for _ in 0..20 {
            ctx.update(0);
        }
        assert_eq!(ctx.k(), 0, "k should be 0 for zero values");
    }

    #[test]
    #[ignore]
    fn test_golomb_context_decay() {
        let mut ctx = GolombContext::new();
        // Feed enough values to trigger decay
        for _ in 0..200 {
            ctx.update(10);
        }
        // Count should have been halved at least once
        assert!(ctx.count < 200);
    }

    #[test]
    #[ignore]
    fn test_golomb_decoder_eof() {
        let data = [0u8; 1];
        let mut dec = GolombDecoder::new(&data);
        // Try to read more bits than available
        let mut result = Ok(0i32);
        for _ in 0..20 {
            result = dec.read_signed(4);
            if result.is_err() {
                break;
            }
        }
        // Should eventually fail with end-of-data
        assert!(result.is_err());
    }

    #[test]
    #[ignore]
    fn test_golomb_encoder_bits() {
        let mut enc = GolombEncoder::new();
        // Value 0 with k=0: interleaved unsigned = 0, quotient=0, so "1" (1 bit)
        enc.write_signed(0, 0);
        let data = enc.finish();
        assert!(!data.is_empty());
    }
}
