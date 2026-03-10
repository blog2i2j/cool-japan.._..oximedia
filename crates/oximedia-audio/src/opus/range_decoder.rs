//! Range/arithmetic decoder for Opus.
//!
//! Opus uses range coding for entropy coding. This is a form of arithmetic
//! coding that works with integer arithmetic and produces byte-aligned output.
//!
//! # Range Coding Basics
//!
//! The decoder maintains a "range" that is progressively narrowed based on
//! symbol probabilities. When the range becomes too small, more bits are
//! read from the input to expand it.

#![forbid(unsafe_code)]

use crate::AudioError;

/// Minimum range value before normalization.
const RANGE_MIN: u32 = 1 << 23;

/// Symbol probability distribution.
#[derive(Debug, Clone)]
pub struct Symbol {
    /// Cumulative frequency of previous symbols.
    pub fl: u32,
    /// Frequency of this symbol.
    pub fh: u32,
    /// Total frequency (must be power of 2 for Opus).
    pub ft: u32,
}

impl Symbol {
    /// Create a new symbol with given probabilities.
    #[must_use]
    pub fn new(fl: u32, fh: u32, ft: u32) -> Self {
        Self { fl, fh, ft }
    }

    /// Create a binary symbol (for single-bit decisions).
    #[must_use]
    pub fn binary(probability: u32, total: u32) -> Self {
        Self {
            fl: 0,
            fh: probability,
            ft: total,
        }
    }

    /// Create a uniform symbol distribution.
    #[must_use]
    pub fn uniform(index: u32, count: u32) -> Self {
        Self {
            fl: index,
            fh: index + 1,
            ft: count,
        }
    }

    /// Get probability of this symbol.
    #[must_use]
    pub fn probability(&self) -> f64 {
        if self.ft == 0 {
            0.0
        } else {
            f64::from(self.fh - self.fl) / f64::from(self.ft)
        }
    }
}

/// Range decoder state.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct RangeDecoder {
    /// Compressed data buffer.
    data: Vec<u8>,
    /// Current byte position.
    byte_pos: usize,
    /// Current bit position within byte.
    bit_pos: u8,
    /// Current range.
    range: u32,
    /// Current value.
    value: u32,
    /// Bits consumed.
    bits_consumed: u32,
    /// Total bits available.
    total_bits: u32,
    /// End of stream flag.
    eos: bool,
}

impl RangeDecoder {
    /// Create a new range decoder from compressed data.
    ///
    /// # Errors
    ///
    /// Returns error if data is empty.
    #[allow(clippy::cast_possible_truncation)]
    pub fn new(data: &[u8]) -> Result<Self, AudioError> {
        if data.is_empty() {
            return Err(AudioError::InvalidData("Empty range coder data".into()));
        }

        let mut decoder = Self {
            data: data.to_vec(),
            byte_pos: 0,
            bit_pos: 0,
            range: 128,
            value: 0,
            bits_consumed: 0,
            total_bits: (data.len() * 8) as u32,
            eos: false,
        };

        // Initialize value from first byte
        decoder.value = 127u32.saturating_sub(u32::from(data[0] >> 1));
        decoder.byte_pos = 1;
        decoder.bits_consumed = 9;

        Ok(decoder)
    }

    /// Get bits remaining in the buffer.
    #[must_use]
    pub fn bits_remaining(&self) -> u32 {
        self.total_bits.saturating_sub(self.bits_consumed)
    }

    /// Check if we've reached end of stream.
    #[must_use]
    pub fn is_eos(&self) -> bool {
        self.eos
    }

    /// Read a single bit.
    fn read_bit(&mut self) -> u8 {
        if self.byte_pos >= self.data.len() {
            self.eos = true;
            return 0;
        }

        let bit = (self.data[self.byte_pos] >> (7 - self.bit_pos)) & 1;
        self.bit_pos += 1;
        if self.bit_pos >= 8 {
            self.bit_pos = 0;
            self.byte_pos += 1;
        }
        self.bits_consumed += 1;
        bit
    }

    /// Normalize the range by reading more bits.
    fn normalize(&mut self) {
        while self.range < RANGE_MIN {
            self.range <<= 8;
            let byte = if self.byte_pos < self.data.len() {
                self.data[self.byte_pos]
            } else {
                0
            };
            self.byte_pos += 1;
            self.value = (self.value << 8) | u32::from(byte);
            self.bits_consumed += 8;
        }
    }

    /// Decode a symbol with given probability distribution.
    ///
    /// # Errors
    ///
    /// Returns error if decoding fails.
    pub fn decode_symbol(&mut self, ft: u32) -> Result<u32, AudioError> {
        if ft == 0 {
            return Err(AudioError::InvalidData(
                "Zero total frequency in range decoder".into(),
            ));
        }

        self.normalize();

        // Calculate scaled frequency
        let fs = self.range / ft;
        if fs == 0 {
            return Err(AudioError::InvalidData(
                "Range too small for frequency".into(),
            ));
        }

        // Find symbol index
        let k = self.value.min((fs * ft).saturating_sub(1)) / fs;
        Ok(k.min(ft - 1))
    }

    /// Update decoder state after decoding a symbol.
    ///
    /// # Errors
    ///
    /// Returns error if update fails.
    pub fn decode_update(&mut self, sym: &Symbol) -> Result<(), AudioError> {
        if sym.ft == 0 {
            return Err(AudioError::InvalidData("Zero total frequency".into()));
        }

        let fs = self.range / sym.ft;
        self.value = self.value.saturating_sub(fs * sym.fl);
        self.range = if sym.fl + (sym.fh - sym.fl) == sym.ft {
            self.range.saturating_sub(fs * sym.fl)
        } else {
            fs * (sym.fh - sym.fl)
        };

        Ok(())
    }

    /// Decode a symbol and update state.
    ///
    /// # Errors
    ///
    /// Returns error if decoding fails.
    #[allow(dead_code)]
    pub fn decode(&mut self, sym: &Symbol) -> Result<u32, AudioError> {
        let k = self.decode_symbol(sym.ft)?;
        self.decode_update(sym)?;
        Ok(k)
    }

    /// Decode a uniform symbol.
    ///
    /// # Errors
    ///
    /// Returns error if decoding fails.
    pub fn decode_uniform(&mut self, count: u32) -> Result<u32, AudioError> {
        if count == 0 {
            return Err(AudioError::InvalidData(
                "Zero count in uniform decode".into(),
            ));
        }
        if count == 1 {
            return Ok(0);
        }

        let k = self.decode_symbol(count)?;
        let sym = Symbol::uniform(k, count);
        self.decode_update(&sym)?;
        Ok(k)
    }

    /// Decode raw bits (bypass mode).
    ///
    /// # Errors
    ///
    /// Returns error if insufficient bits.
    pub fn decode_bits(&mut self, bits: u32) -> Result<u32, AudioError> {
        if bits == 0 {
            return Ok(0);
        }
        if bits > 32 {
            return Err(AudioError::InvalidData("Too many bits requested".into()));
        }

        let mut value = 0u32;
        for _ in 0..bits {
            value = (value << 1) | u32::from(self.read_bit());
        }
        Ok(value)
    }

    /// Decode unsigned integer with given bits.
    ///
    /// # Errors
    ///
    /// Returns error if decoding fails.
    #[allow(dead_code)]
    pub fn decode_uint(&mut self, bits: u32) -> Result<u32, AudioError> {
        self.decode_bits(bits)
    }

    /// Get tell (bits consumed, for rate control).
    #[must_use]
    pub fn tell(&self) -> u32 {
        self.bits_consumed
    }

    /// Get tell in fractional bits (1/8 bit precision).
    #[must_use]
    pub fn tell_frac(&self) -> u32 {
        // Simplified: actual implementation uses range to compute fractional bits
        self.bits_consumed * 8
    }
}

/// Laplace distribution decoder for SILK.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct LaplaceDecoder {
    /// Mean value.
    mean: i32,
    /// Decay factor.
    decay: u32,
}

impl LaplaceDecoder {
    /// Create new Laplace decoder.
    #[must_use]
    pub fn new(mean: i32, decay: u32) -> Self {
        Self { mean, decay }
    }

    /// Decode a Laplace-distributed value.
    ///
    /// # Errors
    ///
    /// Returns error if decoding fails.
    #[allow(dead_code, clippy::cast_possible_wrap)]
    pub fn decode(&self, range_decoder: &mut RangeDecoder) -> Result<i32, AudioError> {
        // Simplified Laplace decoding
        let sign = range_decoder.decode_bits(1)? != 0;
        let magnitude = range_decoder.decode_uniform(self.decay)?;
        let value = self.mean + magnitude as i32;
        Ok(if sign { -value } else { value })
    }
}

/// ICDF (Inverse Cumulative Distribution Function) table.
#[derive(Debug, Clone)]
pub struct IcdfTable {
    /// ICDF values.
    values: Vec<u16>,
    /// Total probability (ft).
    total: u32,
}

impl IcdfTable {
    /// Create ICDF table from probabilities.
    #[must_use]
    pub fn from_probabilities(probs: &[u16]) -> Self {
        let mut values = Vec::with_capacity(probs.len() + 1);
        let mut cumsum = 0u16;
        values.push(0);
        for &p in probs {
            cumsum = cumsum.saturating_add(p);
            values.push(cumsum);
        }
        let total = u32::from(cumsum);
        Self { values, total }
    }

    /// Get number of symbols.
    #[must_use]
    pub fn symbol_count(&self) -> usize {
        self.values.len().saturating_sub(1)
    }

    /// Get symbol probability range.
    #[must_use]
    pub fn symbol(&self, index: usize) -> Option<Symbol> {
        if index + 1 < self.values.len() {
            Some(Symbol {
                fl: u32::from(self.values[index]),
                fh: u32::from(self.values[index + 1]),
                ft: self.total,
            })
        } else {
            None
        }
    }

    /// Decode a symbol using this ICDF.
    ///
    /// # Errors
    ///
    /// Returns error if decoding fails.
    #[allow(dead_code)]
    pub fn decode(&self, range_decoder: &mut RangeDecoder) -> Result<usize, AudioError> {
        let k = range_decoder.decode_symbol(self.total)?;

        // Find symbol by binary search
        let mut lo = 0;
        let mut hi = self.values.len() - 1;
        while lo < hi {
            let mid = (lo + hi) / 2;
            if u32::from(self.values[mid + 1]) <= k {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        let sym = self
            .symbol(lo)
            .ok_or_else(|| AudioError::InvalidData("Invalid symbol index".into()))?;
        range_decoder.decode_update(&sym)?;

        Ok(lo)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbol_new() {
        let sym = Symbol::new(10, 20, 100);
        assert_eq!(sym.fl, 10);
        assert_eq!(sym.fh, 20);
        assert_eq!(sym.ft, 100);
    }

    #[test]
    fn test_symbol_binary() {
        let sym = Symbol::binary(50, 100);
        assert_eq!(sym.fl, 0);
        assert_eq!(sym.fh, 50);
        assert_eq!(sym.ft, 100);
    }

    #[test]
    fn test_symbol_uniform() {
        let sym = Symbol::uniform(5, 10);
        assert_eq!(sym.fl, 5);
        assert_eq!(sym.fh, 6);
        assert_eq!(sym.ft, 10);
    }

    #[test]
    fn test_symbol_probability() {
        let sym = Symbol::new(0, 50, 100);
        assert!((sym.probability() - 0.5).abs() < f64::EPSILON);

        let zero = Symbol::new(0, 0, 0);
        assert!((zero.probability() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_range_decoder_creation() {
        let data = vec![0x80, 0x00, 0x00, 0x00];
        let decoder = RangeDecoder::new(&data).expect("should succeed");
        assert!(!decoder.is_eos());
    }

    #[test]
    fn test_range_decoder_empty() {
        let result = RangeDecoder::new(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_range_decoder_bits_remaining() {
        let data = vec![0x80, 0x00];
        let decoder = RangeDecoder::new(&data).expect("should succeed");
        assert!(decoder.bits_remaining() > 0);
    }

    #[test]
    fn test_decode_bits() {
        let data = vec![0xFF, 0x00, 0xFF, 0x00];
        let mut decoder = RangeDecoder::new(&data).expect("should succeed");
        // Just verify it doesn't panic
        let _ = decoder.decode_bits(4);
    }

    #[test]
    fn test_decode_uniform_one() {
        let data = vec![0x80, 0x00];
        let mut decoder = RangeDecoder::new(&data).expect("should succeed");
        let result = decoder.decode_uniform(1).expect("should succeed");
        assert_eq!(result, 0);
    }

    #[test]
    fn test_tell() {
        let data = vec![0x80, 0x00, 0x00, 0x00];
        let decoder = RangeDecoder::new(&data).expect("should succeed");
        assert!(decoder.tell() > 0);
    }

    #[test]
    fn test_icdf_table() {
        let probs = vec![10, 20, 30, 40];
        let icdf = IcdfTable::from_probabilities(&probs);
        assert_eq!(icdf.symbol_count(), 4);
        assert_eq!(icdf.total, 100);
    }

    #[test]
    fn test_icdf_symbol() {
        let probs = vec![10, 20, 30, 40];
        let icdf = IcdfTable::from_probabilities(&probs);

        let sym0 = icdf.symbol(0).expect("should succeed");
        assert_eq!(sym0.fl, 0);
        assert_eq!(sym0.fh, 10);

        let sym1 = icdf.symbol(1).expect("should succeed");
        assert_eq!(sym1.fl, 10);
        assert_eq!(sym1.fh, 30);
    }

    #[test]
    fn test_laplace_decoder() {
        let decoder = LaplaceDecoder::new(0, 100);
        assert_eq!(decoder.mean, 0);
        assert_eq!(decoder.decay, 100);
    }
}
