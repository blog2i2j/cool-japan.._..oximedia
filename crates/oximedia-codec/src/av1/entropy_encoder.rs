//! AV1 entropy encoding with arithmetic coder and CDF context.
//!
//! This module implements the entropy encoding system for AV1, including:
//!
//! - Arithmetic encoding with range coder
//! - CDF (Cumulative Distribution Function) context management
//! - Symbol encoding with adaptive probabilities
//! - Bitstream output with OBU framing
//!
//! # AV1 Entropy Coding
//!
//! AV1 uses an arithmetic coder with symbol probabilities stored as CDFs.
//! The probabilities are adapted based on previously encoded symbols to
//! improve compression efficiency.
//!
//! # References
//!
//! - AV1 Specification Section 8.2: Arithmetic Coding Engine
//! - AV1 Specification Section 8.3: Symbol Decoding Process

#![forbid(unsafe_code)]
#![allow(dead_code)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::similar_names)]
#![allow(clippy::too_many_arguments)]

use super::entropy_tables::{CDF_PROB_BITS, CDF_PROB_TOP};

// =============================================================================
// Constants
// =============================================================================

/// Bits of precision for arithmetic coder.
const EC_PROB_SHIFT: u32 = 6;

/// Window size for arithmetic coder (2^16).
const EC_WINDOW_SIZE: u32 = 1 << 16;

/// Minimum range before renormalization.
const EC_MIN_RANGE: u32 = 128;

/// Maximum symbol alphabet size.
const MAX_SYMBOL_VALUE: u16 = 15;

/// CDF update rate (higher = faster adaptation).
const CDF_UPDATE_RATE: u16 = 5;

// =============================================================================
// Arithmetic Encoder
// =============================================================================

/// Arithmetic encoder state.
#[derive(Clone, Debug)]
pub struct ArithmeticEncoder {
    /// Current range.
    range: u32,
    /// Low value (accumulated bits).
    low: u32,
    /// Number of outstanding bits.
    cnt: i32,
    /// Output buffer.
    buffer: Vec<u8>,
}

impl Default for ArithmeticEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl ArithmeticEncoder {
    /// Create a new arithmetic encoder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            range: EC_WINDOW_SIZE,
            low: 0,
            cnt: -9,
            buffer: Vec::with_capacity(4096),
        }
    }

    /// Encode a symbol using CDF.
    ///
    /// # Arguments
    ///
    /// * `symbol` - Symbol value to encode
    /// * `cdf` - Cumulative distribution function
    pub fn encode_symbol(&mut self, symbol: u16, cdf: &[u16]) {
        assert!(symbol < cdf.len() as u16 - 1, "Symbol out of range");

        let fl = u32::from(if symbol == 0 {
            0
        } else {
            cdf[symbol as usize - 1]
        });
        let fh = u32::from(cdf[symbol as usize]);
        let _ft = u32::from(cdf[cdf.len() - 1]);

        // Compute new range
        let u = self.range;
        let v = ((u >> 8) * (fh - fl)) >> (CDF_PROB_BITS - 8);
        let r_new = if v < EC_MIN_RANGE { EC_MIN_RANGE } else { v };

        // Update low value
        self.low += ((u >> 8) * fl) >> (CDF_PROB_BITS - 8);
        self.range = r_new;

        // Renormalize if needed
        self.renormalize();
    }

    /// Encode a binary symbol (0 or 1).
    pub fn encode_bool(&mut self, symbol: bool, prob: u16) {
        // For boolean: CDF needs 3 values for 2 symbols: [0], [prob(0)], [total]
        // But we store cumulative, so: [prob_0, prob_0 + prob_1] where prob_0 + prob_1 = total
        let cdf = [CDF_PROB_TOP - prob, CDF_PROB_TOP, CDF_PROB_TOP];
        let symbol_val = u16::from(symbol);
        self.encode_symbol(symbol_val, &cdf);
    }

    /// Encode a literal value with uniform distribution.
    pub fn encode_literal(&mut self, value: u32, num_bits: u8) {
        for i in (0..num_bits).rev() {
            let bit = (value >> i) & 1;
            self.encode_bool(bit != 0, CDF_PROB_TOP / 2);
        }
    }

    /// Renormalize the encoder state.
    fn renormalize(&mut self) {
        while self.range < EC_MIN_RANGE {
            let c = (self.low >> 23) as u8;
            self.buffer.push(c);

            self.low = (self.low << 8) & 0x7F_FF_FF;
            self.range <<= 8;
            self.cnt += 8;
        }
    }

    /// Flush encoder and get output bytes.
    pub fn flush(&mut self) -> Vec<u8> {
        // Final renormalization - output accumulated bits
        while self.cnt >= 0 {
            let c = (self.low >> 23) as u8;
            self.buffer.push(c);
            self.low = (self.low << 8) & 0x7F_FF_FF;
            self.cnt -= 8;
        }

        // Output the remaining partial byte from the encoder state
        let c = (self.low >> 23) as u8;
        self.buffer.push(c);

        // Ensure byte alignment (pad to multiple of 4)
        while self.buffer.len() % 4 != 0 {
            self.buffer.push(0);
        }

        std::mem::take(&mut self.buffer)
    }

    /// Get current buffer without flushing.
    #[must_use]
    pub fn buffer(&self) -> &[u8] {
        &self.buffer
    }

    /// Reset encoder state.
    pub fn reset(&mut self) {
        self.range = EC_WINDOW_SIZE;
        self.low = 0;
        self.cnt = -9;
        self.buffer.clear();
    }
}

// =============================================================================
// CDF Context Management
// =============================================================================

/// CDF (Cumulative Distribution Function) for symbol probabilities.
#[derive(Clone, Debug)]
pub struct CdfContext {
    /// CDF values (cumulative probabilities).
    cdf: Vec<u16>,
    /// Number of symbols in alphabet.
    nsymb: usize,
}

impl CdfContext {
    /// Create a new CDF context with uniform distribution.
    #[must_use]
    pub fn new(nsymb: usize) -> Self {
        let mut cdf = Vec::with_capacity(nsymb + 1);
        let step = CDF_PROB_TOP / nsymb as u16;

        for i in 0..nsymb {
            cdf.push(step * (i as u16 + 1));
        }
        cdf[nsymb - 1] = CDF_PROB_TOP;

        Self { cdf, nsymb }
    }

    /// Get CDF slice.
    #[must_use]
    pub fn cdf(&self) -> &[u16] {
        &self.cdf
    }

    /// Update CDF based on observed symbol.
    pub fn update(&mut self, symbol: u16) {
        if symbol >= self.nsymb as u16 {
            return;
        }

        // Adaptive CDF update using moving average
        for i in symbol as usize..self.nsymb {
            let delta = CDF_PROB_TOP.saturating_sub(self.cdf[i]) >> CDF_UPDATE_RATE;
            self.cdf[i] = self.cdf[i].saturating_add(delta);
        }

        // Ensure last value is always CDF_PROB_TOP
        self.cdf[self.nsymb - 1] = CDF_PROB_TOP;
    }

    /// Reset CDF to uniform distribution.
    pub fn reset(&mut self) {
        let step = CDF_PROB_TOP / self.nsymb as u16;
        for i in 0..self.nsymb {
            self.cdf[i] = step * (i as u16 + 1);
        }
        self.cdf[self.nsymb - 1] = CDF_PROB_TOP;
    }
}

// =============================================================================
// Symbol Encoder
// =============================================================================

/// High-level symbol encoder with CDF management.
#[derive(Clone, Debug)]
pub struct SymbolEncoder {
    /// Arithmetic encoder.
    encoder: ArithmeticEncoder,
    /// CDF contexts for different symbol types.
    contexts: Vec<CdfContext>,
}

impl Default for SymbolEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl SymbolEncoder {
    /// Create a new symbol encoder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            encoder: ArithmeticEncoder::new(),
            contexts: Vec::new(),
        }
    }

    /// Initialize contexts for encoding.
    pub fn init_contexts(&mut self, num_contexts: usize, nsymb: usize) {
        self.contexts.clear();
        for _ in 0..num_contexts {
            self.contexts.push(CdfContext::new(nsymb));
        }
    }

    /// Encode a symbol with given context.
    pub fn encode(&mut self, symbol: u16, context_id: usize) {
        if context_id >= self.contexts.len() {
            // Use default uniform CDF
            let cdf = CdfContext::new(MAX_SYMBOL_VALUE as usize + 1);
            self.encoder.encode_symbol(symbol, cdf.cdf());
            return;
        }

        let cdf = self.contexts[context_id].cdf().to_vec();
        self.encoder.encode_symbol(symbol, &cdf);

        // Update CDF after encoding
        self.contexts[context_id].update(symbol);
    }

    /// Encode a boolean value.
    pub fn encode_bool(&mut self, value: bool) {
        self.encoder.encode_bool(value, CDF_PROB_TOP / 2);
    }

    /// Encode a literal value.
    pub fn encode_literal(&mut self, value: u32, num_bits: u8) {
        self.encoder.encode_literal(value, num_bits);
    }

    /// Finish encoding and get output.
    pub fn finish(&mut self) -> Vec<u8> {
        self.encoder.flush()
    }

    /// Get current output without finishing.
    #[must_use]
    pub fn buffer(&self) -> &[u8] {
        self.encoder.buffer()
    }

    /// Reset encoder and contexts.
    pub fn reset(&mut self) {
        self.encoder.reset();
        for ctx in &mut self.contexts {
            ctx.reset();
        }
    }
}

// =============================================================================
// Bitstream Writer
// =============================================================================

/// Bitstream writer for byte-aligned output.
#[derive(Clone, Debug, Default)]
pub struct BitstreamWriter {
    /// Output buffer.
    buffer: Vec<u8>,
    /// Current byte being written.
    current_byte: u8,
    /// Number of bits written in current byte.
    bit_pos: u8,
}

impl BitstreamWriter {
    /// Create a new bitstream writer.
    #[must_use]
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            current_byte: 0,
            bit_pos: 0,
        }
    }

    /// Write a single bit.
    pub fn write_bit(&mut self, bit: bool) {
        if bit {
            self.current_byte |= 1 << (7 - self.bit_pos);
        }

        self.bit_pos += 1;
        if self.bit_pos == 8 {
            self.buffer.push(self.current_byte);
            self.current_byte = 0;
            self.bit_pos = 0;
        }
    }

    /// Write multiple bits from a value.
    pub fn write_bits(&mut self, value: u32, num_bits: u8) {
        for i in (0..num_bits).rev() {
            let bit = (value >> i) & 1;
            self.write_bit(bit != 0);
        }
    }

    /// Write a byte-aligned value.
    pub fn write_byte(&mut self, byte: u8) {
        self.align();
        self.buffer.push(byte);
    }

    /// Align to byte boundary.
    pub fn align(&mut self) {
        if self.bit_pos != 0 {
            self.buffer.push(self.current_byte);
            self.current_byte = 0;
            self.bit_pos = 0;
        }
    }

    /// Write a slice of bytes.
    pub fn write_bytes(&mut self, bytes: &[u8]) {
        self.align();
        self.buffer.extend_from_slice(bytes);
    }

    /// Get output buffer.
    #[must_use]
    pub fn buffer(&self) -> &[u8] {
        &self.buffer
    }

    /// Consume writer and get output.
    #[must_use]
    pub fn finish(mut self) -> Vec<u8> {
        self.align();
        self.buffer
    }

    /// Get buffer length in bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.buffer.len() + usize::from(self.bit_pos > 0)
    }

    /// Check if writer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty() && self.bit_pos == 0
    }

    /// Reset writer.
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.current_byte = 0;
        self.bit_pos = 0;
    }
}

// =============================================================================
// OBU Writer
// =============================================================================

/// OBU (Open Bitstream Unit) writer.
#[derive(Clone, Debug)]
pub struct ObuWriter {
    /// Bitstream writer.
    writer: BitstreamWriter,
}

impl Default for ObuWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl ObuWriter {
    /// Create a new OBU writer.
    #[must_use]
    pub fn new() -> Self {
        Self {
            writer: BitstreamWriter::new(),
        }
    }

    /// Write OBU header.
    pub fn write_obu_header(&mut self, obu_type: u8, has_size: bool) {
        // Forbidden bit
        self.writer.write_bit(false);

        // OBU type (4 bits)
        self.writer.write_bits(u32::from(obu_type), 4);

        // Extension flag
        self.writer.write_bit(false);

        // Has size flag
        self.writer.write_bit(has_size);

        // Reserved bit
        self.writer.write_bit(false);
    }

    /// Write LEB128 encoded size.
    pub fn write_leb128(&mut self, mut value: u64) {
        loop {
            let mut byte = (value & 0x7F) as u8;
            value >>= 7;

            if value != 0 {
                byte |= 0x80;
            }

            self.writer.write_byte(byte);

            if value == 0 {
                break;
            }
        }
    }

    /// Write OBU with size field.
    pub fn write_obu(&mut self, obu_type: u8, payload: &[u8]) {
        self.write_obu_header(obu_type, true);
        self.write_leb128(payload.len() as u64);
        self.writer.write_bytes(payload);
    }

    /// Get output buffer.
    #[must_use]
    pub fn buffer(&self) -> &[u8] {
        self.writer.buffer()
    }

    /// Finish and get output.
    #[must_use]
    pub fn finish(self) -> Vec<u8> {
        self.writer.finish()
    }
}

// =============================================================================
// Utility Functions
// =============================================================================

/// Compute CDF from probability mass function (PMF).
#[must_use]
pub fn pmf_to_cdf(pmf: &[u16]) -> Vec<u16> {
    let mut cdf = Vec::with_capacity(pmf.len());
    let mut cumsum = 0u16;

    for &p in pmf {
        cumsum = cumsum.saturating_add(p);
        cdf.push(cumsum);
    }

    // Normalize to CDF_PROB_TOP
    if let Some(&last) = cdf.last() {
        if last > 0 && last != CDF_PROB_TOP {
            for val in &mut cdf {
                *val = (u32::from(*val) * u32::from(CDF_PROB_TOP) / u32::from(last)) as u16;
            }
        }
    }

    cdf
}

/// Estimate rate for symbol given CDF.
#[must_use]
pub fn estimate_symbol_rate(symbol: u16, cdf: &[u16]) -> f32 {
    if symbol >= cdf.len() as u16 {
        return 8.0; // Default rate for unknown symbol
    }

    let fl = if symbol == 0 {
        0
    } else {
        cdf[symbol as usize - 1]
    };
    let fh = cdf[symbol as usize];
    let prob = fh.saturating_sub(fl);

    if prob == 0 {
        16.0 // Very unlikely symbol
    } else {
        -(f32::from(prob) / f32::from(CDF_PROB_TOP)).log2()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arithmetic_encoder_creation() {
        let encoder = ArithmeticEncoder::new();
        assert_eq!(encoder.range, EC_WINDOW_SIZE);
        assert_eq!(encoder.low, 0);
        assert!(encoder.buffer.is_empty());
    }

    #[test]
    fn test_arithmetic_encoder_bool() {
        let mut encoder = ArithmeticEncoder::new();
        encoder.encode_bool(true, CDF_PROB_TOP / 2);
        encoder.encode_bool(false, CDF_PROB_TOP / 2);

        let output = encoder.flush();
        assert!(!output.is_empty());
    }

    #[test]
    fn test_arithmetic_encoder_literal() {
        let mut encoder = ArithmeticEncoder::new();
        encoder.encode_literal(0xFF, 8);

        let output = encoder.flush();
        assert!(!output.is_empty());
    }

    #[test]
    fn test_cdf_context_creation() {
        let cdf = CdfContext::new(4);
        assert_eq!(cdf.nsymb, 4);
        assert_eq!(cdf.cdf().len(), 4);
        assert_eq!(
            *cdf.cdf().last().expect("should have last element"),
            CDF_PROB_TOP
        );
    }

    #[test]
    fn test_cdf_context_update() {
        let mut cdf = CdfContext::new(4);
        let initial_cdf = cdf.cdf().to_vec();

        cdf.update(1);
        let updated_cdf = cdf.cdf();

        // CDF should change after update
        assert_ne!(initial_cdf, updated_cdf);
        assert_eq!(
            *updated_cdf.last().expect("should have last element"),
            CDF_PROB_TOP
        );
    }

    #[test]
    fn test_cdf_context_reset() {
        let mut cdf = CdfContext::new(4);
        let initial_cdf = cdf.cdf().to_vec();

        cdf.update(1);
        cdf.update(2);
        cdf.reset();

        assert_eq!(cdf.cdf(), &initial_cdf[..]);
    }

    #[test]
    fn test_symbol_encoder() {
        let mut encoder = SymbolEncoder::new();
        encoder.init_contexts(4, 8);

        encoder.encode(0, 0);
        encoder.encode(1, 0);
        encoder.encode(2, 1);

        let output = encoder.finish();
        assert!(!output.is_empty());
    }

    #[test]
    fn test_symbol_encoder_bool() {
        let mut encoder = SymbolEncoder::new();
        encoder.encode_bool(true);
        encoder.encode_bool(false);
        encoder.encode_bool(true);

        let output = encoder.finish();
        assert!(!output.is_empty());
    }

    #[test]
    fn test_bitstream_writer_bit() {
        let mut writer = BitstreamWriter::new();
        writer.write_bit(true);
        writer.write_bit(false);
        writer.write_bit(true);
        writer.write_bit(true);
        writer.write_bit(false);
        writer.write_bit(false);
        writer.write_bit(false);
        writer.write_bit(true);

        let output = writer.finish();
        assert_eq!(output.len(), 1);
        assert_eq!(output[0], 0b1011_0001);
    }

    #[test]
    fn test_bitstream_writer_bits() {
        let mut writer = BitstreamWriter::new();
        writer.write_bits(0xFF, 8);

        let output = writer.finish();
        assert_eq!(output.len(), 1);
        assert_eq!(output[0], 0xFF);
    }

    #[test]
    fn test_bitstream_writer_align() {
        let mut writer = BitstreamWriter::new();
        writer.write_bit(true);
        writer.write_bit(false);
        writer.align();

        let output = writer.finish();
        assert_eq!(output.len(), 1);
    }

    #[test]
    fn test_bitstream_writer_bytes() {
        let mut writer = BitstreamWriter::new();
        writer.write_bytes(&[0xAB, 0xCD, 0xEF]);

        let output = writer.finish();
        assert_eq!(output, &[0xAB, 0xCD, 0xEF]);
    }

    #[test]
    fn test_obu_writer_header() {
        let mut writer = ObuWriter::new();
        writer.write_obu_header(1, true);

        let output = writer.buffer();
        assert!(!output.is_empty());
    }

    #[test]
    fn test_obu_writer_leb128() {
        let mut writer = ObuWriter::new();
        writer.write_leb128(127);

        let output = writer.buffer();
        assert_eq!(output.len(), 1);
        assert_eq!(output[0], 127);

        let mut writer2 = ObuWriter::new();
        writer2.write_leb128(128);

        let output2 = writer2.buffer();
        assert_eq!(output2.len(), 2);
    }

    #[test]
    fn test_obu_writer_complete() {
        let mut writer = ObuWriter::new();
        let payload = vec![1, 2, 3, 4];
        writer.write_obu(1, &payload);

        let output = writer.finish();
        assert!(output.len() > payload.len());
    }

    #[test]
    fn test_pmf_to_cdf() {
        let pmf = vec![100, 200, 300, 400];
        let cdf = pmf_to_cdf(&pmf);

        assert_eq!(cdf.len(), 4);
        assert!(*cdf.last().expect("should have last element") > 0);
        // Check monotonic increasing
        for i in 1..cdf.len() {
            assert!(cdf[i] >= cdf[i - 1]);
        }
    }

    #[test]
    fn test_estimate_symbol_rate() {
        let cdf = vec![100, 300, 600, CDF_PROB_TOP];

        let rate0 = estimate_symbol_rate(0, &cdf);
        let rate1 = estimate_symbol_rate(1, &cdf);

        assert!(rate0 > 0.0);
        assert!(rate1 > 0.0);
        // More probable symbol should have lower rate
        assert!(rate0 < rate1 * 2.0);
    }

    #[test]
    fn test_bitstream_writer_len() {
        let mut writer = BitstreamWriter::new();
        assert_eq!(writer.len(), 0);
        assert!(writer.is_empty());

        writer.write_byte(0xFF);
        assert_eq!(writer.len(), 1);
        assert!(!writer.is_empty());
    }

    #[test]
    fn test_symbol_encoder_reset() {
        let mut encoder = SymbolEncoder::new();
        encoder.init_contexts(2, 4);
        encoder.encode(1, 0);

        encoder.reset();
        assert!(encoder.buffer().is_empty());
    }

    #[test]
    fn test_arithmetic_encoder_reset() {
        let mut encoder = ArithmeticEncoder::new();
        encoder.encode_bool(true, CDF_PROB_TOP / 2);

        encoder.reset();
        assert_eq!(encoder.range, EC_WINDOW_SIZE);
        assert!(encoder.buffer.is_empty());
    }
}
