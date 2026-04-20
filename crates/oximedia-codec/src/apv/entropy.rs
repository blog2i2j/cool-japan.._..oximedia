//! APV entropy coding.
//!
//! Implements exponential-Golomb variable-length coding and run-length encoding
//! for zero coefficients. This is the entropy coding stage of the APV codec,
//! operating on quantized DCT coefficients after zigzag scanning.
//!
//! # Coding scheme
//!
//! - **Unsigned exp-Golomb**: encodes non-negative integers with prefix + suffix
//! - **Signed exp-Golomb**: maps signed integers to unsigned via zigzag mapping
//!   (0 → 0, -1 → 1, 1 → 2, -2 → 3, 2 → 4, ...)
//! - **Run-length**: encodes consecutive zero coefficients as (run_length, next_value)
//!   pairs, with a special end-of-block marker

use super::types::ApvError;

// ── BitWriter ───────────────────────────────────────────────────────────────

/// Bit-level writer that accumulates bits into a byte buffer.
///
/// Bits are written MSB-first within each byte.
#[derive(Clone, Debug)]
pub struct BitWriter {
    /// Accumulated bytes.
    buf: Vec<u8>,
    /// Current byte being assembled.
    current_byte: u8,
    /// Number of bits written into `current_byte` (0..8).
    bit_pos: u8,
}

impl BitWriter {
    /// Create a new `BitWriter` with the given initial capacity in bytes.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            buf: Vec::with_capacity(capacity),
            current_byte: 0,
            bit_pos: 0,
        }
    }

    /// Write a single bit (0 or 1).
    pub fn write_bit(&mut self, bit: u8) {
        self.current_byte = (self.current_byte << 1) | (bit & 1);
        self.bit_pos += 1;
        if self.bit_pos == 8 {
            self.buf.push(self.current_byte);
            self.current_byte = 0;
            self.bit_pos = 0;
        }
    }

    /// Write `n` bits from the least-significant end of `value`.
    ///
    /// Bits are written MSB-first (bit `n-1` first, bit 0 last).
    pub fn write_bits(&mut self, value: u32, n: u8) {
        for i in (0..n).rev() {
            self.write_bit(((value >> i) & 1) as u8);
        }
    }

    /// Flush any remaining bits, padding with zeros to byte boundary.
    /// Returns the accumulated byte buffer.
    #[must_use]
    pub fn finish(mut self) -> Vec<u8> {
        if self.bit_pos > 0 {
            // Pad remaining bits with zeros on the right
            self.current_byte <<= 8 - self.bit_pos;
            self.buf.push(self.current_byte);
        }
        self.buf
    }

    /// Number of bits written so far (including unflushed bits).
    #[must_use]
    pub fn bit_count(&self) -> usize {
        self.buf.len() * 8 + self.bit_pos as usize
    }
}

// ── BitReader ───────────────────────────────────────────────────────────────

/// Bit-level reader for consuming bits from a byte buffer.
///
/// Bits are read MSB-first within each byte.
#[derive(Clone, Debug)]
pub struct BitReader<'a> {
    /// Source data.
    data: &'a [u8],
    /// Current byte index.
    byte_pos: usize,
    /// Current bit position within the byte (0 = MSB, 7 = LSB).
    bit_pos: u8,
}

impl<'a> BitReader<'a> {
    /// Create a new `BitReader` over the given data.
    #[must_use]
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_pos: 0,
            bit_pos: 0,
        }
    }

    /// Read a single bit, returning 0 or 1.
    ///
    /// # Errors
    ///
    /// Returns `ApvError::InvalidBitstream` if there are no more bits.
    pub fn read_bit(&mut self) -> Result<u8, ApvError> {
        if self.byte_pos >= self.data.len() {
            return Err(ApvError::InvalidBitstream(
                "unexpected end of bitstream".to_string(),
            ));
        }
        let bit = (self.data[self.byte_pos] >> (7 - self.bit_pos)) & 1;
        self.bit_pos += 1;
        if self.bit_pos == 8 {
            self.bit_pos = 0;
            self.byte_pos += 1;
        }
        Ok(bit)
    }

    /// Read `n` bits and return them as a `u32` (MSB-first).
    ///
    /// # Errors
    ///
    /// Returns `ApvError::InvalidBitstream` if there are not enough bits.
    pub fn read_bits(&mut self, n: u8) -> Result<u32, ApvError> {
        let mut value = 0u32;
        for _ in 0..n {
            value = (value << 1) | self.read_bit()? as u32;
        }
        Ok(value)
    }

    /// Total number of bits consumed so far.
    #[must_use]
    pub fn bits_read(&self) -> usize {
        self.byte_pos * 8 + self.bit_pos as usize
    }

    /// Whether there are any remaining bits.
    #[must_use]
    pub fn has_more(&self) -> bool {
        self.byte_pos < self.data.len()
    }
}

// ── Exponential-Golomb coding ───────────────────────────────────────────────

/// Encode an unsigned integer using order-0 exponential-Golomb coding.
///
/// The codeword for value `x` is:
///   1. Compute `x + 1`
///   2. Write `floor(log2(x+1))` zero bits
///   3. Write the binary representation of `x + 1` (including leading 1)
///
/// Examples:
/// - 0 → `1`
/// - 1 → `010`
/// - 2 → `011`
/// - 3 → `00100`
/// - 4 → `00101`
pub fn exp_golomb_encode(writer: &mut BitWriter, value: u32) {
    let xp1 = value + 1;
    let n_bits = 32 - xp1.leading_zeros(); // floor(log2(x+1)) + 1
                                           // Write (n_bits - 1) leading zeros
    for _ in 0..(n_bits - 1) {
        writer.write_bit(0);
    }
    // Write the n_bits representation of x+1
    writer.write_bits(xp1, n_bits as u8);
}

/// Decode an unsigned exponential-Golomb coded value.
///
/// # Errors
///
/// Returns `ApvError::InvalidBitstream` if the bitstream is truncated.
pub fn exp_golomb_decode(reader: &mut BitReader<'_>) -> Result<u32, ApvError> {
    // Count leading zeros
    let mut leading_zeros: u32 = 0;
    loop {
        let bit = reader.read_bit()?;
        if bit == 0 {
            leading_zeros += 1;
            // Sanity: prevent absurdly long codes (>31 bits)
            if leading_zeros > 31 {
                return Err(ApvError::InvalidBitstream(
                    "exp-Golomb code too long".to_string(),
                ));
            }
        } else {
            break;
        }
    }
    // Read the remaining `leading_zeros` bits
    let remaining = reader.read_bits(leading_zeros as u8)?;
    Ok((1 << leading_zeros) - 1 + remaining)
}

/// Map a signed integer to an unsigned integer via zigzag mapping.
///
/// The mapping is: 0 → 0, -1 → 1, 1 → 2, -2 → 3, 2 → 4, ...
#[must_use]
pub fn zigzag_encode_signed(value: i32) -> u32 {
    ((value << 1) ^ (value >> 31)) as u32
}

/// Inverse zigzag mapping: unsigned → signed.
#[must_use]
pub fn zigzag_decode_signed(value: u32) -> i32 {
    ((value >> 1) as i32) ^ (-((value & 1) as i32))
}

/// Encode a signed integer using signed exponential-Golomb coding.
///
/// Maps the signed value to unsigned via zigzag, then exp-Golomb encodes it.
pub fn signed_exp_golomb_encode(writer: &mut BitWriter, value: i32) {
    let mapped = zigzag_encode_signed(value);
    exp_golomb_encode(writer, mapped);
}

/// Decode a signed exponential-Golomb coded value.
///
/// # Errors
///
/// Returns `ApvError::InvalidBitstream` on truncation.
pub fn signed_exp_golomb_decode(reader: &mut BitReader<'_>) -> Result<i32, ApvError> {
    let mapped = exp_golomb_decode(reader)?;
    Ok(zigzag_decode_signed(mapped))
}

// ── Run-length encoding for zero coefficients ──────────────────────────────

/// Sentinel value indicating end-of-block in the run-length stream.
///
/// Encoded as a run of 0 zeros followed by value 0 (which is distinct from
/// a normal (0, value) pair because the run-length for non-zero values is
/// the count of *preceding* zeros).
const EOB_RUN: u32 = 0;
const EOB_VALUE: i32 = 0;

/// Encode a zigzag-scanned block of 64 quantized coefficients using
/// run-length encoding for zero runs followed by signed exp-Golomb for
/// non-zero values.
///
/// The format for each non-zero coefficient is:
///   1. exp-Golomb(run_of_preceding_zeros)
///   2. signed-exp-Golomb(coefficient_value)
///
/// An end-of-block marker (EOB) is written when the remaining coefficients
/// are all zero: exp-Golomb(0) + signed-exp-Golomb(0).
pub fn encode_block_coefficients(writer: &mut BitWriter, coeffs: &[i32; 64]) {
    let mut i = 0;

    while i < 64 {
        // Find the next non-zero coefficient
        let mut zero_run: u32 = 0;
        while i < 64 && coeffs[i] == 0 {
            zero_run += 1;
            i += 1;
        }

        if i >= 64 {
            // All remaining coefficients are zero — write EOB
            exp_golomb_encode(writer, EOB_RUN);
            signed_exp_golomb_encode(writer, EOB_VALUE);
            return;
        }

        // Write (run_of_zeros, non_zero_value)
        exp_golomb_encode(writer, zero_run);
        signed_exp_golomb_encode(writer, coeffs[i]);
        i += 1;
    }

    // If we consumed all 64 coefficients with no trailing zeros, write EOB
    exp_golomb_encode(writer, EOB_RUN);
    signed_exp_golomb_encode(writer, EOB_VALUE);
}

/// Decode a block of 64 run-length + exp-Golomb coded coefficients.
///
/// # Errors
///
/// Returns `ApvError::InvalidBitstream` if the stream is truncated or malformed.
pub fn decode_block_coefficients(reader: &mut BitReader<'_>) -> Result<[i32; 64], ApvError> {
    let mut coeffs = [0i32; 64];
    let mut i: usize = 0;

    loop {
        if i >= 64 {
            break;
        }

        let zero_run = exp_golomb_decode(reader)?;
        let value = signed_exp_golomb_decode(reader)?;

        // Check for EOB marker
        if zero_run == EOB_RUN && value == EOB_VALUE {
            // Remaining coefficients are implicitly zero
            break;
        }

        // Advance past the zero run
        let new_pos = i + zero_run as usize;
        if new_pos >= 64 {
            return Err(ApvError::InvalidBitstream(format!(
                "zero run {zero_run} at position {i} overflows block"
            )));
        }
        i = new_pos;

        if i >= 64 {
            return Err(ApvError::InvalidBitstream(
                "coefficient position overflows block".to_string(),
            ));
        }

        coeffs[i] = value;
        i += 1;
    }

    Ok(coeffs)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── BitWriter / BitReader tests ─────────────────────────────────────────

    #[test]
    fn test_bitwriter_single_bits() {
        let mut w = BitWriter::new(16);
        w.write_bit(1);
        w.write_bit(0);
        w.write_bit(1);
        w.write_bit(1);
        w.write_bit(0);
        w.write_bit(0);
        w.write_bit(1);
        w.write_bit(0);
        let data = w.finish();
        assert_eq!(data, vec![0b1011_0010]);
    }

    #[test]
    fn test_bitwriter_partial_byte() {
        let mut w = BitWriter::new(16);
        w.write_bit(1);
        w.write_bit(1);
        w.write_bit(0);
        let data = w.finish();
        // 110 padded to 11000000
        assert_eq!(data, vec![0b1100_0000]);
    }

    #[test]
    fn test_bitwriter_write_bits() {
        let mut w = BitWriter::new(16);
        w.write_bits(0b1010, 4);
        w.write_bits(0b1100, 4);
        let data = w.finish();
        assert_eq!(data, vec![0b1010_1100]);
    }

    #[test]
    fn test_bitwriter_bit_count() {
        let mut w = BitWriter::new(16);
        assert_eq!(w.bit_count(), 0);
        w.write_bits(0xFF, 8);
        assert_eq!(w.bit_count(), 8);
        w.write_bit(1);
        assert_eq!(w.bit_count(), 9);
    }

    #[test]
    fn test_bitreader_single_bits() {
        let data = vec![0b1011_0010];
        let mut r = BitReader::new(&data);
        assert_eq!(r.read_bit().expect("bit"), 1);
        assert_eq!(r.read_bit().expect("bit"), 0);
        assert_eq!(r.read_bit().expect("bit"), 1);
        assert_eq!(r.read_bit().expect("bit"), 1);
        assert_eq!(r.read_bit().expect("bit"), 0);
        assert_eq!(r.read_bit().expect("bit"), 0);
        assert_eq!(r.read_bit().expect("bit"), 1);
        assert_eq!(r.read_bit().expect("bit"), 0);
    }

    #[test]
    fn test_bitreader_read_bits() {
        let data = vec![0b1010_1100];
        let mut r = BitReader::new(&data);
        assert_eq!(r.read_bits(4).expect("bits"), 0b1010);
        assert_eq!(r.read_bits(4).expect("bits"), 0b1100);
    }

    #[test]
    fn test_bitreader_exhausted() {
        let data = vec![0xFF];
        let mut r = BitReader::new(&data);
        for _ in 0..8 {
            r.read_bit().expect("bit");
        }
        assert!(r.read_bit().is_err());
    }

    #[test]
    fn test_bitwriter_reader_roundtrip() {
        let mut w = BitWriter::new(64);
        w.write_bits(0xDEAD, 16);
        w.write_bits(0b101, 3);
        w.write_bits(0b11, 2);
        let data = w.finish();

        let mut r = BitReader::new(&data);
        assert_eq!(r.read_bits(16).expect("bits"), 0xDEAD);
        assert_eq!(r.read_bits(3).expect("bits"), 0b101);
        assert_eq!(r.read_bits(2).expect("bits"), 0b11);
    }

    // ── Exp-Golomb tests ────────────────────────────────────────────────────

    #[test]
    fn test_exp_golomb_encode_decode_small_values() {
        for value in 0..=20u32 {
            let mut w = BitWriter::new(16);
            exp_golomb_encode(&mut w, value);
            let data = w.finish();

            let mut r = BitReader::new(&data);
            let decoded = exp_golomb_decode(&mut r).expect("decode");
            assert_eq!(decoded, value, "exp-Golomb roundtrip failed for {value}");
        }
    }

    #[test]
    fn test_exp_golomb_encode_decode_large_values() {
        for &value in &[100u32, 255, 1000, 10000, 65535] {
            let mut w = BitWriter::new(16);
            exp_golomb_encode(&mut w, value);
            let data = w.finish();

            let mut r = BitReader::new(&data);
            let decoded = exp_golomb_decode(&mut r).expect("decode");
            assert_eq!(decoded, value, "exp-Golomb roundtrip failed for {value}");
        }
    }

    #[test]
    fn test_exp_golomb_zero() {
        let mut w = BitWriter::new(16);
        exp_golomb_encode(&mut w, 0);
        let data = w.finish();
        // Code for 0 is just "1" (1 bit)
        assert_eq!(data.len(), 1);
        assert_eq!(data[0] & 0x80, 0x80); // MSB is 1
    }

    #[test]
    fn test_exp_golomb_one() {
        let mut w = BitWriter::new(16);
        exp_golomb_encode(&mut w, 1);
        let data = w.finish();
        // Code for 1 is "010" (3 bits)
        let mut r = BitReader::new(&data);
        assert_eq!(r.read_bit().expect("bit"), 0); // leading zero
        assert_eq!(r.read_bit().expect("bit"), 1); // 1
        assert_eq!(r.read_bit().expect("bit"), 0); // suffix
    }

    #[test]
    fn test_exp_golomb_multiple_values() {
        let values = [0u32, 5, 100, 3, 0, 255, 1];
        let mut w = BitWriter::new(128);
        for &v in &values {
            exp_golomb_encode(&mut w, v);
        }
        let data = w.finish();

        let mut r = BitReader::new(&data);
        for &expected in &values {
            let decoded = exp_golomb_decode(&mut r).expect("decode");
            assert_eq!(decoded, expected);
        }
    }

    // ── Signed exp-Golomb tests ─────────────────────────────────────────────

    #[test]
    fn test_zigzag_mapping() {
        assert_eq!(zigzag_encode_signed(0), 0);
        assert_eq!(zigzag_encode_signed(-1), 1);
        assert_eq!(zigzag_encode_signed(1), 2);
        assert_eq!(zigzag_encode_signed(-2), 3);
        assert_eq!(zigzag_encode_signed(2), 4);
    }

    #[test]
    fn test_zigzag_roundtrip() {
        for v in -1000..=1000 {
            let encoded = zigzag_encode_signed(v);
            let decoded = zigzag_decode_signed(encoded);
            assert_eq!(decoded, v, "zigzag roundtrip failed for {v}");
        }
    }

    #[test]
    fn test_signed_exp_golomb_roundtrip() {
        for value in -50..=50 {
            let mut w = BitWriter::new(16);
            signed_exp_golomb_encode(&mut w, value);
            let data = w.finish();

            let mut r = BitReader::new(&data);
            let decoded = signed_exp_golomb_decode(&mut r).expect("decode");
            assert_eq!(
                decoded, value,
                "signed exp-Golomb roundtrip failed for {value}"
            );
        }
    }

    #[test]
    fn test_signed_exp_golomb_negative() {
        let mut w = BitWriter::new(16);
        signed_exp_golomb_encode(&mut w, -42);
        let data = w.finish();

        let mut r = BitReader::new(&data);
        let decoded = signed_exp_golomb_decode(&mut r).expect("decode");
        assert_eq!(decoded, -42);
    }

    // ── Run-length coding tests ─────────────────────────────────────────────

    #[test]
    fn test_encode_decode_all_zeros() {
        let coeffs = [0i32; 64];
        let mut w = BitWriter::new(64);
        encode_block_coefficients(&mut w, &coeffs);
        let data = w.finish();

        let mut r = BitReader::new(&data);
        let decoded = decode_block_coefficients(&mut r).expect("decode");
        assert_eq!(decoded, coeffs);
    }

    #[test]
    fn test_encode_decode_dc_only() {
        let mut coeffs = [0i32; 64];
        coeffs[0] = 42;
        let mut w = BitWriter::new(64);
        encode_block_coefficients(&mut w, &coeffs);
        let data = w.finish();

        let mut r = BitReader::new(&data);
        let decoded = decode_block_coefficients(&mut r).expect("decode");
        assert_eq!(decoded, coeffs);
    }

    #[test]
    fn test_encode_decode_sparse_block() {
        let mut coeffs = [0i32; 64];
        coeffs[0] = 100;
        coeffs[5] = -20;
        coeffs[10] = 3;
        coeffs[63] = -1;

        let mut w = BitWriter::new(128);
        encode_block_coefficients(&mut w, &coeffs);
        let data = w.finish();

        let mut r = BitReader::new(&data);
        let decoded = decode_block_coefficients(&mut r).expect("decode");
        assert_eq!(decoded, coeffs);
    }

    #[test]
    fn test_encode_decode_dense_block() {
        let mut coeffs = [0i32; 64];
        for i in 0..64 {
            coeffs[i] = (i as i32 - 32) * 2;
        }

        let mut w = BitWriter::new(256);
        encode_block_coefficients(&mut w, &coeffs);
        let data = w.finish();

        let mut r = BitReader::new(&data);
        let decoded = decode_block_coefficients(&mut r).expect("decode");
        assert_eq!(decoded, coeffs);
    }

    #[test]
    fn test_encode_decode_negative_values() {
        let mut coeffs = [0i32; 64];
        coeffs[0] = -500;
        coeffs[1] = 300;
        coeffs[2] = -1;
        coeffs[3] = 1;

        let mut w = BitWriter::new(128);
        encode_block_coefficients(&mut w, &coeffs);
        let data = w.finish();

        let mut r = BitReader::new(&data);
        let decoded = decode_block_coefficients(&mut r).expect("decode");
        assert_eq!(decoded, coeffs);
    }

    #[test]
    fn test_encode_decode_all_nonzero() {
        let mut coeffs = [0i32; 64];
        for i in 0..64 {
            coeffs[i] = if i % 2 == 0 {
                (i as i32) + 1
            } else {
                -(i as i32) - 1
            };
        }

        let mut w = BitWriter::new(512);
        encode_block_coefficients(&mut w, &coeffs);
        let data = w.finish();

        let mut r = BitReader::new(&data);
        let decoded = decode_block_coefficients(&mut r).expect("decode");
        assert_eq!(decoded, coeffs);
    }

    #[test]
    fn test_encode_decode_multiple_blocks() {
        let block1 = {
            let mut c = [0i32; 64];
            c[0] = 100;
            c[1] = -50;
            c
        };
        let block2 = {
            let mut c = [0i32; 64];
            c[0] = -200;
            c[10] = 30;
            c[63] = 1;
            c
        };

        let mut w = BitWriter::new(256);
        encode_block_coefficients(&mut w, &block1);
        encode_block_coefficients(&mut w, &block2);
        let data = w.finish();

        let mut r = BitReader::new(&data);
        let dec1 = decode_block_coefficients(&mut r).expect("decode block 1");
        let dec2 = decode_block_coefficients(&mut r).expect("decode block 2");
        assert_eq!(dec1, block1);
        assert_eq!(dec2, block2);
    }

    #[test]
    fn test_bitreader_bits_read() {
        let data = vec![0xFF, 0x00];
        let mut r = BitReader::new(&data);
        assert_eq!(r.bits_read(), 0);
        r.read_bits(4).expect("bits");
        assert_eq!(r.bits_read(), 4);
        r.read_bits(8).expect("bits");
        assert_eq!(r.bits_read(), 12);
    }

    #[test]
    fn test_bitreader_has_more() {
        let data = vec![0xFF];
        let mut r = BitReader::new(&data);
        assert!(r.has_more());
        r.read_bits(8).expect("bits");
        assert!(!r.has_more());
    }

    #[test]
    fn test_encode_efficiency() {
        // All zeros should be very compact (just EOB marker)
        let zeros = [0i32; 64];
        let mut w = BitWriter::new(64);
        encode_block_coefficients(&mut w, &zeros);
        let zero_data = w.finish();

        // Dense block should be larger
        let mut dense = [0i32; 64];
        for i in 0..64 {
            dense[i] = (i as i32) * 10;
        }
        let mut w2 = BitWriter::new(256);
        encode_block_coefficients(&mut w2, &dense);
        let dense_data = w2.finish();

        assert!(
            dense_data.len() > zero_data.len(),
            "Dense block should produce more data than all-zeros"
        );
    }

    #[test]
    fn test_encode_large_coefficient() {
        let mut coeffs = [0i32; 64];
        coeffs[0] = 10000;
        coeffs[1] = -10000;

        let mut w = BitWriter::new(128);
        encode_block_coefficients(&mut w, &coeffs);
        let data = w.finish();

        let mut r = BitReader::new(&data);
        let decoded = decode_block_coefficients(&mut r).expect("decode");
        assert_eq!(decoded, coeffs);
    }
}
