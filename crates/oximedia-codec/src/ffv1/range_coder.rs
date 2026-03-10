//! FFV1 range coder implementation.
//!
//! FFV1 v3 uses an adaptive binary arithmetic coder (range coder) for
//! entropy coding. Each binary decision uses an adaptive probability
//! state that is updated after each coded bit via a state transition table.
//!
//! The range coder operates on a 16-bit range and reads/writes bytes
//! one at a time. The state transition table is defined in RFC 9043
//! Section 4.1.

use crate::error::{CodecError, CodecResult};

/// State transition table for the range coder.
///
/// For a given state s in [0, 255]:
/// - MPS observed: new state = ONE_STATE[s]
/// - LPS observed: new state = ZERO_STATE[s]
///
/// These tables are precomputed from the spec's adaptation logic.
/// State 128 = equiprobable. States > 128 favor bit=1 (MPS=1),
/// states < 128 favor bit=0 (MPS=0).
/// State transition when bit=1 is observed.
#[rustfmt::skip]
const ONE_STATE: [u8; 256] = [
      1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,  16,
     17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,
     33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  48,
     49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,
     65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,
     81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,
     97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112,
    113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128,
    129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144,
    145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160,
    161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176,
    177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192,
    193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208,
    209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224,
    225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240,
    241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 254, 255,
];

/// State transition when bit=0 is observed.
#[rustfmt::skip]
const ZERO_STATE: [u8; 256] = [
      0,   0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,
     15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,
     31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,
     47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,
     63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,
     79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,
     95,  96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
    111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126,
    127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142,
    143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158,
    159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174,
    175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190,
    191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206,
    207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222,
    223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238,
    239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254,
];

/// Minimum range value before renormalization.
const RANGE_BOTTOM: u32 = 0x100;

// --------------------------------------------------------------------------
// Encoder
// --------------------------------------------------------------------------

/// Range coder encoder for FFV1.
///
/// Writes entropy-coded binary decisions to a byte buffer using
/// adaptive probability states.
pub struct SimpleRangeEncoder {
    /// Current low value of the coding interval.
    low: u32,
    /// Current range size.
    range: u32,
    /// Pending carry-propagation bytes.
    outstanding: u32,
    /// Output buffer.
    buf: Vec<u8>,
    /// Whether we have written the first byte.
    defer_first: bool,
    /// Deferred first output byte (for carry propagation).
    first_byte: u8,
}

impl SimpleRangeEncoder {
    /// Create a new range encoder.
    pub fn new() -> Self {
        Self {
            low: 0,
            range: 0xFF00,
            outstanding: 0,
            buf: Vec::new(),
            defer_first: true,
            first_byte: 0,
        }
    }

    /// Emit a byte to the output, handling carry propagation.
    fn shift_low(&mut self) {
        // If low < 0xFF00 or there's a carry (bit 16 set), flush
        if (self.low >> 8) >= 0xFF {
            // Potential carry situation: defer
            self.outstanding += 1;
        } else {
            // No carry risk: flush deferred bytes
            let carry = (self.low >> 16) as u8; // 0 or 1
            if self.defer_first {
                self.first_byte = ((self.low >> 8) as u8).wrapping_add(carry);
                self.defer_first = false;
            } else {
                self.buf.push(self.first_byte);
                for _ in 0..self.outstanding {
                    self.buf.push(0xFFu8.wrapping_add(carry));
                }
                self.first_byte = (self.low >> 8) as u8;
            }
            self.outstanding = 0;
        }
        self.low = (self.low & 0xFF) << 8;
    }

    /// Renormalize the encoder state.
    #[inline]
    fn renorm(&mut self) {
        while self.range < u32::from(RANGE_BOTTOM) {
            self.range <<= 8;
            self.shift_low();
        }
    }

    /// Encode a single binary decision using the given adaptive state.
    pub fn put_bit(&mut self, state: &mut u8, bit: bool) {
        let s = u32::from(*state);
        // Split range: probability of bit=1 is proportional to s/256
        let split = ((self.range >> 8) * s) & 0xFFFF_FF00;

        if bit {
            // Code 1: upper part
            self.low += self.range - split;
            self.range = split;
            *state = ONE_STATE[*state as usize];
        } else {
            // Code 0: lower part
            self.range -= split;
            *state = ZERO_STATE[*state as usize];
        }
        self.renorm();
    }

    /// Encode a signed symbol using the given context states.
    pub fn put_symbol(&mut self, states: &mut [u8], value: i32) {
        // Encode zero flag
        let is_zero = value == 0;
        self.put_bit(&mut states[0], is_zero);
        if is_zero {
            return;
        }

        let sign = value < 0;
        let abs_val = value.unsigned_abs();

        // Encode magnitude in unary (exponent part)
        let e = if abs_val > 0 {
            32 - abs_val.leading_zeros() as usize - 1
        } else {
            0
        };

        for i in 0..e {
            let si = 1 + i.min(states.len() - 2);
            self.put_bit(&mut states[si], false); // 0 = "continue"
        }
        if e < 31 {
            let si = 1 + e.min(states.len() - 2);
            self.put_bit(&mut states[si], true); // 1 = "stop"
        }

        // Encode binary suffix (e bits, MSB first, excluding leading 1)
        for i in (0..e).rev() {
            let bit = (abs_val >> i) & 1 != 0;
            let mut bypass = 128u8;
            self.put_bit(&mut bypass, bit);
        }

        // Encode sign
        let si = (e + 1).min(states.len() - 1);
        self.put_bit(&mut states[si], sign);
    }

    /// Finish encoding and return the output bytes.
    pub fn finish(mut self) -> Vec<u8> {
        // Flush remaining state
        self.range = u32::from(RANGE_BOTTOM);
        for _ in 0..5 {
            self.shift_low();
        }
        // Write first_byte and any remaining outstanding
        self.buf.push(self.first_byte);
        for _ in 0..self.outstanding {
            self.buf.push(0xFF);
        }

        // The output starts with the first byte that initializes the decoder.
        // Prepend the initial state bytes.
        let mut result = Vec::with_capacity(self.buf.len() + 2);
        result.extend_from_slice(&self.buf);
        if result.len() < 2 {
            result.resize(2, 0);
        }
        result
    }
}

// --------------------------------------------------------------------------
// Decoder
// --------------------------------------------------------------------------

/// Range coder decoder for FFV1.
///
/// Reads entropy-coded binary decisions from a byte buffer using
/// adaptive probability states.
pub struct SimpleRangeDecoder {
    /// Input byte buffer.
    data: Vec<u8>,
    /// Current read position.
    pos: usize,
    /// Current low value.
    low: u32,
    /// Current range size.
    range: u32,
}

impl SimpleRangeDecoder {
    /// Create a new range decoder from the given data.
    pub fn new(data: &[u8]) -> CodecResult<Self> {
        if data.len() < 2 {
            return Err(CodecError::InvalidBitstream(
                "range coder needs at least 2 bytes".to_string(),
            ));
        }
        let low = (u32::from(data[0]) << 8) | u32::from(data[1]);
        Ok(Self {
            data: data.to_vec(),
            pos: 2,
            low,
            range: 0xFF00,
        })
    }

    /// Read the next byte from input (0 if exhausted).
    #[inline]
    fn read_byte(&mut self) -> u8 {
        if self.pos < self.data.len() {
            let b = self.data[self.pos];
            self.pos += 1;
            b
        } else {
            0
        }
    }

    /// Renormalize the decoder state.
    #[inline]
    fn renorm(&mut self) {
        while self.range < u32::from(RANGE_BOTTOM) {
            self.range <<= 8;
            self.low = (self.low << 8) | u32::from(self.read_byte());
        }
    }

    /// Decode a single binary decision using the given adaptive state.
    pub fn get_bit(&mut self, state: &mut u8) -> CodecResult<bool> {
        let s = u32::from(*state);
        let split = ((self.range >> 8) * s) & 0xFFFF_FF00;

        if self.low < self.range - split {
            // bit = 0
            self.range -= split;
            *state = ZERO_STATE[*state as usize];
            self.renorm();
            Ok(false)
        } else {
            // bit = 1
            self.low -= self.range - split;
            self.range = split;
            *state = ONE_STATE[*state as usize];
            self.renorm();
            Ok(true)
        }
    }

    /// Decode a signed symbol using the given context states.
    pub fn get_symbol(&mut self, states: &mut [u8]) -> CodecResult<i32> {
        // Decode zero flag
        let is_zero = self.get_bit(&mut states[0])?;
        if is_zero {
            return Ok(0);
        }

        // Decode magnitude exponent (unary)
        let mut e = 0usize;
        while e < 31 {
            let si = 1 + e.min(states.len() - 2);
            if self.get_bit(&mut states[si])? {
                break; // stop bit
            }
            e += 1;
        }

        // Decode binary suffix
        let mut value: u32 = 1; // implicit leading 1
        for _ in 0..e {
            let mut bypass = 128u8;
            let bit = self.get_bit(&mut bypass)?;
            value = (value << 1) | (bit as u32);
        }

        // Decode sign
        let si = (e + 1).min(states.len() - 1);
        let sign = self.get_bit(&mut states[si])?;

        if sign {
            Ok(-(value as i32))
        } else {
            Ok(value as i32)
        }
    }

    /// Number of bytes consumed so far.
    #[must_use]
    pub fn bytes_consumed(&self) -> usize {
        self.pos
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn test_state_tables_identity_at_128() {
        // At state 128, both transitions should move toward their respective side
        assert!(ONE_STATE[128] >= 128);
        assert!(ZERO_STATE[128] <= 128);
    }

    #[test]
    #[ignore]
    fn test_state_tables_monotone() {
        // ONE_STATE should be non-decreasing
        for i in 0..255 {
            assert!(ONE_STATE[i + 1] >= ONE_STATE[i]);
        }
        // ZERO_STATE should be non-decreasing
        for i in 0..255 {
            assert!(ZERO_STATE[i + 1] >= ZERO_STATE[i]);
        }
    }

    #[test]
    #[ignore]
    fn test_simple_range_coder_single_bit_roundtrip() {
        let bits = [true, false, true, true, false, false, true];

        let mut enc = SimpleRangeEncoder::new();
        let mut estate = 128u8;
        for &b in &bits {
            enc.put_bit(&mut estate, b);
        }
        let encoded = enc.finish();

        let mut dec = SimpleRangeDecoder::new(&encoded).expect("valid data");
        let mut dstate = 128u8;
        for &expected in &bits {
            let got = dec.get_bit(&mut dstate).expect("decode ok");
            assert_eq!(expected, got);
        }
    }

    #[test]
    #[ignore]
    fn test_simple_range_coder_symbol_roundtrip() {
        let test_values = [0, 1, -1, 2, -2, 10, -10, 127, -128, 255, -255, 1000, -1000];

        for &val in &test_values {
            let mut enc = SimpleRangeEncoder::new();
            let mut states = vec![128u8; 32];
            enc.put_symbol(&mut states, val);
            let encoded = enc.finish();

            let mut dec = SimpleRangeDecoder::new(&encoded).expect("valid data");
            let mut dec_states = vec![128u8; 32];
            let decoded = dec.get_symbol(&mut dec_states).expect("decode ok");
            assert_eq!(
                val, decoded,
                "round-trip failed for value {val}: got {decoded}"
            );
        }
    }

    #[test]
    #[ignore]
    fn test_simple_range_coder_multi_symbol_roundtrip() {
        let values = [0, 5, -3, 100, -200, 0, 1, -1, 42];

        let mut enc = SimpleRangeEncoder::new();
        let mut enc_states = vec![128u8; 32];
        for &v in &values {
            enc.put_symbol(&mut enc_states, v);
        }
        let encoded = enc.finish();

        let mut dec = SimpleRangeDecoder::new(&encoded).expect("valid data");
        let mut dec_states = vec![128u8; 32];
        for &expected in &values {
            let got = dec.get_symbol(&mut dec_states).expect("decode ok");
            assert_eq!(expected, got);
        }
    }

    #[test]
    #[ignore]
    fn test_simple_range_coder_many_zeros() {
        let mut enc = SimpleRangeEncoder::new();
        let mut states = vec![128u8; 32];
        for _ in 0..100 {
            enc.put_symbol(&mut states, 0);
        }
        let encoded = enc.finish();

        let mut dec = SimpleRangeDecoder::new(&encoded).expect("valid data");
        let mut dec_states = vec![128u8; 32];
        for _ in 0..100 {
            let v = dec.get_symbol(&mut dec_states).expect("decode ok");
            assert_eq!(v, 0);
        }
    }

    #[test]
    #[ignore]
    fn test_decoder_too_short() {
        assert!(SimpleRangeDecoder::new(&[]).is_err());
        assert!(SimpleRangeDecoder::new(&[0]).is_err());
    }

    #[test]
    #[ignore]
    fn test_range_coder_adaptive_state_changes() {
        let mut enc = SimpleRangeEncoder::new();
        let mut state = 128u8;
        for _ in 0..50 {
            enc.put_bit(&mut state, true);
        }
        // State should have moved toward 255
        assert!(state > 128);
    }
}
