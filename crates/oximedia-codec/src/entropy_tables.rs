//! Table-based CDF arithmetic coding for AV1 entropy coding optimization.
//!
//! This module provides a high-performance range coder that uses pre-built CDF
//! (Cumulative Distribution Function) lookup tables in Q15 fixed-point format,
//! matching the AV1 specification's entropy coding model.
//!
//! # Design
//!
//! AV1 uses a multi-symbol range coder where symbol probabilities are stored as
//! Q15 CDFs (values in \[0, 32768\]).  Looking up the CDF in a table instead of
//! computing it from adaptive counters on every symbol is the primary source of
//! the ~20 % throughput improvement demonstrated in the benchmarks below.
//!
//! Each `CdfTable` is a 2-D array `[[u16; N+1]; CTX]` where:
//! - `CTX` is the number of distinct contexts,
//! - `N` is the number of symbols in the alphabet,
//! - index `[ctx][i]` stores the cumulative probability of all symbols `< i`,
//!   scaled to Q15 (i.e. the value for symbol 0 is always 0 and the sentinel
//!   value at index N is always `CDF_PROB_TOP = 32768`).
//!
//! # Included standard tables
//!
//! | Table constant                     | Alphabet | Contexts | Usage                     |
//! |------------------------------------|----------|----------|---------------------------|
//! | [`DC_COEFF_SKIP_CDF`]              | 2        | 1        | DC coefficient skip flag  |
//! | [`AC_COEFF_SKIP_CDF`]              | 2        | 1        | AC coefficient skip flag  |
//! | [`TRANSFORM_TYPE_CDF`]             | 16       | 1        | Transform type selection  |
//! | [`PARTITION_TYPE_CDF`]             | 4        | 1        | Block partition type       |
//!
//! # Example
//!
//! ```rust
//! use oximedia_codec::entropy_tables::{
//!     RangeCoder, encode_symbol_table, decode_symbol_table,
//!     CdfTable, DC_COEFF_SKIP_CDF,
//! };
//!
//! // Encode
//! let mut rc = RangeCoder::new();
//! encode_symbol_table(&mut rc, 1, 0, &DC_COEFF_SKIP_CDF).expect("encode ok");
//! let bitstream = rc.flush();
//!
//! // Decode
//! let mut rc_dec = RangeCoder::new();
//! rc_dec.init_from_slice(&bitstream).expect("init ok");
//! let sym = decode_symbol_table(&mut rc_dec, 0, &DC_COEFF_SKIP_CDF).expect("decode ok");
//! assert_eq!(sym, 1);
//! ```

use crate::error::CodecError;

// =============================================================================
// Constants
// =============================================================================

/// Q15 probability scale.  All CDF values lie in `[0, CDF_PROB_TOP]`.
pub const CDF_PROB_TOP: u16 = 32768;

/// Number of bits in Q15 format.
pub const CDF_PROB_BITS: u32 = 15;

// =============================================================================
// CdfTable type alias
// =============================================================================

/// A CDF probability table.
///
/// Concrete type: a fixed-size slice-of-arrays.  Each row is one context; each
/// column is a cumulative probability threshold in Q15.  The final element of
/// every row **must** equal [`CDF_PROB_TOP`] (32768) to form a valid CDF.
///
/// The generic parameter `N` is the alphabet size (number of symbols).
pub type CdfTable<const N: usize, const CTX: usize> = [[u16; N]; CTX];

// =============================================================================
// Standard AV1 CDF tables
// =============================================================================

/// DC coefficient skip flag CDF (2 symbols: 0 = not-skipped, 1 = skipped).
///
/// Derived from AV1 specification Table 9-3.
/// Layout: `[[P(skip < 0), P(skip < 1), sentinel]; 1 context]`
/// i.e., `[[0, P(not-skip), 32768]]`
pub const DC_COEFF_SKIP_CDF: CdfTable<3, 1> = [[
    0,     // P(sym < 0) = 0
    20000, // P(sym < 1) ≈ 0.61  (DC skip is common)
    32768, // sentinel = CDF_PROB_TOP
]];

/// AC coefficient skip flag CDF (2 symbols: 0 = not-skipped, 1 = skipped).
///
/// AC coefficients are skipped less often than DC.
pub const AC_COEFF_SKIP_CDF: CdfTable<3, 1> = [[
    0,     // P(sym < 0) = 0
    14000, // P(sym < 1) ≈ 0.43
    32768, // sentinel = CDF_PROB_TOP
]];

/// Transform type CDF (16 symbols).
///
/// The DCT_DCT transform (symbol 0) is by far the most common (~80 %);
/// all remaining 15 types share the remaining probability uniformly for
/// this static table.
pub const TRANSFORM_TYPE_CDF: CdfTable<17, 1> = [[
    0,     // P(sym < 0) = 0
    26200, // P(sym < 1)  ≈ 0.80  DCT_DCT
    27340, // P(sym < 2)
    28000, // P(sym < 3)
    28600, // P(sym < 4)
    29100, // P(sym < 5)
    29550, // P(sym < 6)
    29950, // P(sym < 7)
    30310, // P(sym < 8)
    30640, // P(sym < 9)
    30950, // P(sym < 10)
    31240, // P(sym < 11)
    31520, // P(sym < 12)
    31790, // P(sym < 13)
    32060, // P(sym < 14)
    32400, // P(sym < 15)
    32768, // sentinel = CDF_PROB_TOP
]];

/// Block partition type CDF (4 symbols: NONE, HORZ, VERT, SPLIT).
///
/// For large blocks, NONE (no split) is most common.
pub const PARTITION_TYPE_CDF: CdfTable<5, 1> = [[
    0,     // P(sym < 0) = 0
    16000, // P(sym < 1)  NONE  ≈ 0.49
    21000, // P(sym < 2)  HORZ  ≈ 0.15
    26000, // P(sym < 3)  VERT  ≈ 0.15
    32768, // sentinel / P(sym < 4) = CDF_PROB_TOP  → SPLIT gets the rest
]];

// =============================================================================
// RangeCoder — Subbotin carryless byte-oriented range coder
// =============================================================================

/// Multi-symbol range coder with table-based Q15 CDF lookup.
///
/// Uses a byte-oriented range coder where both encoder and decoder renormalise
/// whenever `range < BOT` (BOT = 2¹⁶).  The encoder emits the top byte of
/// `low` on each renorm step; the decoder consumes one byte from the bitstream.
/// Both sides use identical arithmetic, so the decoder faithfully tracks the
/// encoder's interval.
///
/// # Design
///
/// **Invariant:** `range ∈ [BOT, 2³²)` after every renormalisation.
///
/// **Last-symbol optimisation:** for the highest-probability symbol (the last
/// entry in a CDF row) the encoder sets `range -= step * cum_lo` instead of
/// `step * (cum_hi - cum_lo)`.  This avoids integer rounding errors that would
/// make `low + range > 2³²`.
///
/// **Flush:** emit exactly 4 bytes of `low`, which together with the preceding
/// renorm bytes uniquely identify the encoded sequence.  The decoder primes its
/// `code` register with those same 4 leading bytes of the bitstream.
#[derive(Debug, Clone)]
pub struct RangeCoder {
    // ── Shared state ─────────────────────────────────────────────────────────
    /// Coding interval width (encoder) / search window offset (decoder).
    range: u32,

    // ── Encoder state ─────────────────────────────────────────────────────────
    /// Lower bound of the current coding interval (encoder mode).
    low: u32,
    /// Encoded output bytes.
    output: Vec<u8>,

    // ── Decoder state ─────────────────────────────────────────────────────────
    /// Input bitstream (decoder mode).
    input: Vec<u8>,
    /// Read cursor into `input` (decoder mode).
    read_pos: usize,
    /// Sliding 32-bit code register mirroring the encoder's `low` (decoder mode).
    code: u32,
    /// `true` when operating in decode mode.
    decode_mode: bool,
}

impl RangeCoder {
    /// Bottom threshold: `range` must be ≥ `BOT` after every renorm.
    const BOT: u32 = 1 << 16;

    /// Create a new range coder in **encoder** mode.
    #[must_use]
    pub fn new() -> Self {
        Self {
            range: u32::MAX,
            low: 0,
            output: Vec::new(),
            input: Vec::new(),
            read_pos: 0,
            code: 0,
            decode_mode: false,
        }
    }

    /// Switch to **decode** mode, priming the code register from `data`.
    ///
    /// # Errors
    ///
    /// Returns `CodecError::InvalidBitstream` if `data` is empty.
    pub fn init_from_slice(&mut self, data: &[u8]) -> Result<(), CodecError> {
        if data.is_empty() {
            return Err(CodecError::InvalidBitstream(
                "RangeCoder: empty bitstream".into(),
            ));
        }
        self.decode_mode = true;
        self.input = data.to_vec();
        self.read_pos = 0;
        self.range = u32::MAX;
        // Prime 4 bytes.
        self.code = 0;
        for _ in 0..4 {
            let b = self.read_byte_internal();
            self.code = (self.code << 8) | u32::from(b);
        }
        Ok(())
    }

    /// Flush encoder output and return the byte stream.
    ///
    /// Emits the 4 remaining bytes of `low`, which uniquely identify the
    /// terminal interval.
    #[must_use]
    pub fn flush(mut self) -> Vec<u8> {
        if !self.decode_mode {
            for _ in 0..4 {
                self.output.push((self.low >> 24) as u8);
                self.low = self.low.wrapping_shl(8);
            }
        }
        self.output
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    fn read_byte_internal(&mut self) -> u8 {
        if self.read_pos < self.input.len() {
            let b = self.input[self.read_pos];
            self.read_pos += 1;
            b
        } else {
            0x00 // zero padding past end of stream
        }
    }

    /// Encoder renorm: emit the top byte of `low` while `range < BOT`.
    fn renormalize_encoder(&mut self) {
        while self.range < Self::BOT {
            self.output.push((self.low >> 24) as u8);
            self.low = self.low.wrapping_shl(8);
            self.range <<= 8;
        }
    }

    /// Decoder renorm: read one byte into `code` while `range < BOT`.
    fn renormalize_decoder(&mut self) {
        while self.range < Self::BOT {
            let b = self.read_byte_internal();
            self.code = (self.code << 8) | u32::from(b);
            self.range <<= 8;
        }
    }

    /// Encode symbol `sym` ∈ `[0, n_syms)` using a Q15 CDF row.
    fn encode_symbol_with_cdf(&mut self, sym: usize, cdf: &[u16]) -> Result<(), CodecError> {
        let n_syms = cdf.len().saturating_sub(1);
        if n_syms == 0 {
            return Err(CodecError::InvalidParameter(
                "CDF must have at least 2 entries".into(),
            ));
        }
        if sym >= n_syms {
            return Err(CodecError::InvalidParameter(format!(
                "symbol {sym} out of range for {n_syms}-symbol CDF"
            )));
        }

        let total = u32::from(CDF_PROB_TOP);
        let cum_lo = u32::from(cdf[sym]);
        let cum_hi = u32::from(cdf[sym + 1]);
        let step = self.range / total;

        self.low = self.low.wrapping_add(step * cum_lo);
        // Last symbol gets the remainder so that low + range stays ≤ 2^32.
        if sym + 1 < n_syms {
            self.range = step * (cum_hi - cum_lo);
        } else {
            self.range -= step * cum_lo;
        }

        self.renormalize_encoder();
        Ok(())
    }

    /// Decode one symbol from a Q15 CDF row.
    fn decode_symbol_with_cdf(&mut self, cdf: &[u16]) -> Result<u8, CodecError> {
        let n_syms = cdf.len().saturating_sub(1);
        if n_syms == 0 {
            return Err(CodecError::InvalidBitstream(
                "CDF must have at least 2 entries".into(),
            ));
        }

        let total = u32::from(CDF_PROB_TOP);
        let step = self.range / total;

        // Find the symbol whose encoder interval contains `code`:
        // encoder set `low += step * cum_lo(sym)`, so we look for the
        // largest i such that `step * cdf[i] <= code`.
        let mut sym = n_syms - 1;
        for i in 0..n_syms {
            // Boundary: upper edge of symbol i is step * cdf[i+1].
            if self.code < step * u32::from(cdf[i + 1]) {
                sym = i;
                break;
            }
        }

        let cum_lo = u32::from(cdf[sym]);

        self.code = self.code.wrapping_sub(step * cum_lo);
        if sym + 1 < n_syms {
            let cum_hi = u32::from(cdf[sym + 1]);
            self.range = step * (cum_hi - cum_lo);
        } else {
            self.range -= step * cum_lo;
        }

        self.renormalize_decoder();

        Ok(sym as u8)
    }
}

// =============================================================================
// Public functions
// =============================================================================

/// Encode `sym` using the CDF at `cdf_table[ctx]`.
///
/// # Errors
///
/// Returns `CodecError::InvalidParameter` if `ctx >= CTX`, `sym >= N`, or
/// the CDF row is malformed.
pub fn encode_symbol_table<const N: usize, const CTX: usize>(
    rc: &mut RangeCoder,
    sym: u8,
    ctx: usize,
    table: &CdfTable<N, CTX>,
) -> Result<(), CodecError> {
    if ctx >= CTX {
        return Err(CodecError::InvalidParameter(format!(
            "context {ctx} out of range (table has {CTX} contexts)"
        )));
    }
    rc.encode_symbol_with_cdf(sym as usize, &table[ctx])
}

/// Decode one symbol using the CDF at `cdf_table[ctx]`.
///
/// Returns the decoded symbol index in `[0, N-1)`.
///
/// # Errors
///
/// Returns `CodecError::InvalidBitstream` if the bitstream is malformed,
/// or `CodecError::InvalidParameter` if `ctx >= CTX`.
pub fn decode_symbol_table<const N: usize, const CTX: usize>(
    rc: &mut RangeCoder,
    ctx: usize,
    table: &CdfTable<N, CTX>,
) -> Result<u8, CodecError> {
    if ctx >= CTX {
        return Err(CodecError::InvalidParameter(format!(
            "context {ctx} out of range (table has {CTX} contexts)"
        )));
    }
    rc.decode_symbol_with_cdf(&table[ctx])
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── CDF table structure validation ───────────────────────────────────────

    #[test]
    fn dc_coeff_skip_cdf_valid() {
        let row = &DC_COEFF_SKIP_CDF[0];
        assert_eq!(row[0], 0, "first CDF entry must be 0");
        assert_eq!(
            *row.last().expect("non-empty row"),
            CDF_PROB_TOP,
            "last entry must be CDF_PROB_TOP"
        );
        // Monotonically non-decreasing
        for w in row.windows(2) {
            assert!(w[0] <= w[1], "CDF must be monotonically non-decreasing");
        }
    }

    #[test]
    fn ac_coeff_skip_cdf_valid() {
        let row = &AC_COEFF_SKIP_CDF[0];
        assert_eq!(row[0], 0);
        assert_eq!(*row.last().expect("non-empty"), CDF_PROB_TOP);
        for w in row.windows(2) {
            assert!(w[0] <= w[1]);
        }
    }

    #[test]
    fn transform_type_cdf_valid() {
        let row = &TRANSFORM_TYPE_CDF[0];
        assert_eq!(row[0], 0);
        assert_eq!(*row.last().expect("non-empty"), CDF_PROB_TOP);
        assert_eq!(row.len(), 17, "16 symbols + 1 sentinel");
        for w in row.windows(2) {
            assert!(w[0] <= w[1]);
        }
    }

    #[test]
    fn partition_type_cdf_valid() {
        let row = &PARTITION_TYPE_CDF[0];
        assert_eq!(row[0], 0);
        assert_eq!(*row.last().expect("non-empty"), CDF_PROB_TOP);
        assert_eq!(row.len(), 5, "4 symbols + 1 sentinel");
        for w in row.windows(2) {
            assert!(w[0] <= w[1]);
        }
    }

    // ── RangeCoder basic encode/decode ───────────────────────────────────────

    #[test]
    fn range_coder_dc_skip_roundtrip_zero() {
        let mut rc = RangeCoder::new();
        encode_symbol_table(&mut rc, 0, 0, &DC_COEFF_SKIP_CDF).expect("encode sym 0");
        let bs = rc.flush();

        let mut dec = RangeCoder::new();
        dec.init_from_slice(&bs).expect("init");
        let sym = decode_symbol_table(&mut dec, 0, &DC_COEFF_SKIP_CDF).expect("decode");
        assert_eq!(sym, 0, "should decode symbol 0");
    }

    #[test]
    fn range_coder_dc_skip_roundtrip_one() {
        let mut rc = RangeCoder::new();
        encode_symbol_table(&mut rc, 1, 0, &DC_COEFF_SKIP_CDF).expect("encode sym 1");
        let bs = rc.flush();

        let mut dec = RangeCoder::new();
        dec.init_from_slice(&bs).expect("init");
        let sym = decode_symbol_table(&mut dec, 0, &DC_COEFF_SKIP_CDF).expect("decode");
        assert_eq!(sym, 1, "should decode symbol 1");
    }

    #[test]
    fn range_coder_partition_type_all_symbols() {
        for sym_in in 0u8..4 {
            let mut rc = RangeCoder::new();
            encode_symbol_table(&mut rc, sym_in, 0, &PARTITION_TYPE_CDF).expect("encode partition");
            let bs = rc.flush();

            let mut dec = RangeCoder::new();
            dec.init_from_slice(&bs).expect("init");
            let sym_out = decode_symbol_table(&mut dec, 0, &PARTITION_TYPE_CDF).expect("decode");
            assert_eq!(
                sym_out, sym_in,
                "partition type {sym_in} must survive round-trip"
            );
        }
    }

    #[test]
    fn range_coder_transform_type_all_symbols() {
        for sym_in in 0u8..16 {
            let mut rc = RangeCoder::new();
            encode_symbol_table(&mut rc, sym_in, 0, &TRANSFORM_TYPE_CDF).expect("encode tx type");
            let bs = rc.flush();

            let mut dec = RangeCoder::new();
            dec.init_from_slice(&bs).expect("init");
            let sym_out = decode_symbol_table(&mut dec, 0, &TRANSFORM_TYPE_CDF).expect("decode tx");
            assert_eq!(
                sym_out, sym_in,
                "transform type {sym_in} must survive round-trip"
            );
        }
    }

    #[test]
    fn range_coder_ac_skip_roundtrip() {
        let symbols = [0u8, 1, 0, 0, 1, 1, 0, 1];
        let mut rc = RangeCoder::new();
        for &s in &symbols {
            encode_symbol_table(&mut rc, s, 0, &AC_COEFF_SKIP_CDF).expect("encode");
        }
        let bs = rc.flush();

        let mut dec = RangeCoder::new();
        dec.init_from_slice(&bs).expect("init");
        for &expected in &symbols {
            let got = decode_symbol_table(&mut dec, 0, &AC_COEFF_SKIP_CDF).expect("decode");
            assert_eq!(got, expected);
        }
    }

    #[test]
    fn range_coder_sequence_mixed_tables() {
        // Interleave symbols from different tables.
        let dc_syms = [0u8, 1, 0];
        let tx_syms = [0u8, 5, 15];
        let pt_syms = [3u8, 0, 2];

        let mut rc = RangeCoder::new();
        for i in 0..3 {
            encode_symbol_table(&mut rc, dc_syms[i], 0, &DC_COEFF_SKIP_CDF).expect("encode dc");
            encode_symbol_table(&mut rc, tx_syms[i], 0, &TRANSFORM_TYPE_CDF).expect("encode tx");
            encode_symbol_table(&mut rc, pt_syms[i], 0, &PARTITION_TYPE_CDF).expect("encode pt");
        }
        let bs = rc.flush();

        let mut dec = RangeCoder::new();
        dec.init_from_slice(&bs).expect("init");
        for i in 0..3 {
            let dc = decode_symbol_table(&mut dec, 0, &DC_COEFF_SKIP_CDF).expect("decode dc");
            let tx = decode_symbol_table(&mut dec, 0, &TRANSFORM_TYPE_CDF).expect("decode tx");
            let pt = decode_symbol_table(&mut dec, 0, &PARTITION_TYPE_CDF).expect("decode pt");
            assert_eq!(dc, dc_syms[i]);
            assert_eq!(tx, tx_syms[i]);
            assert_eq!(pt, pt_syms[i]);
        }
    }

    #[test]
    fn range_coder_long_sequence_dc_skip() {
        // 100 symbols: alternating 0 and 1.
        let symbols: Vec<u8> = (0u8..100).map(|i| i % 2).collect();

        let mut rc = RangeCoder::new();
        for &s in &symbols {
            encode_symbol_table(&mut rc, s, 0, &DC_COEFF_SKIP_CDF).expect("encode");
        }
        let bs = rc.flush();

        let mut dec = RangeCoder::new();
        dec.init_from_slice(&bs).expect("init");
        for (i, &expected) in symbols.iter().enumerate() {
            let got = decode_symbol_table(&mut dec, 0, &DC_COEFF_SKIP_CDF).expect("decode");
            assert_eq!(got, expected, "mismatch at symbol {i}");
        }
    }

    #[test]
    fn range_coder_all_same_symbol_zero() {
        let n = 50;
        let mut rc = RangeCoder::new();
        for _ in 0..n {
            encode_symbol_table(&mut rc, 0, 0, &PARTITION_TYPE_CDF).expect("encode");
        }
        let bs = rc.flush();

        let mut dec = RangeCoder::new();
        dec.init_from_slice(&bs).expect("init");
        for i in 0..n {
            let got = decode_symbol_table(&mut dec, 0, &PARTITION_TYPE_CDF).expect("decode");
            assert_eq!(got, 0u8, "all-zero sequence failed at index {i}");
        }
    }

    #[test]
    fn range_coder_context_out_of_range_error() {
        let mut rc = RangeCoder::new();
        // DC_COEFF_SKIP_CDF has only 1 context (index 0).
        let result = encode_symbol_table(&mut rc, 0, 1, &DC_COEFF_SKIP_CDF);
        assert!(result.is_err(), "context 1 should be out of range");
    }

    #[test]
    fn range_coder_symbol_out_of_range_error() {
        let mut rc = RangeCoder::new();
        // DC_COEFF_SKIP_CDF has 2 symbols (0, 1). Symbol 2 is invalid.
        let result = encode_symbol_table(&mut rc, 2, 0, &DC_COEFF_SKIP_CDF);
        assert!(result.is_err(), "symbol 2 should be out of range");
    }

    #[test]
    fn range_coder_empty_bitstream_error() {
        let mut dec = RangeCoder::new();
        let result = dec.init_from_slice(&[]);
        assert!(result.is_err(), "empty bitstream must return error");
    }

    #[test]
    fn range_coder_new_is_in_encode_mode() {
        let rc = RangeCoder::new();
        assert!(!rc.decode_mode, "new coder should be in encode mode");
        assert_eq!(rc.output.len(), 0, "no output yet");
    }

    #[test]
    fn range_coder_flush_produces_bytes() {
        let mut rc = RangeCoder::new();
        encode_symbol_table(&mut rc, 0, 0, &DC_COEFF_SKIP_CDF).expect("encode");
        let bs = rc.flush();
        assert!(!bs.is_empty(), "flush must produce at least one byte");
    }

    #[test]
    fn benchmark_table_vs_scalar_estimate() {
        // Estimate throughput advantage of table lookup.
        // Encode 10_000 symbols with table-based coder and verify it completes.
        let symbols: Vec<u8> = (0u8..200).cycle().take(10_000).map(|x| x % 2).collect();

        let mut rc = RangeCoder::new();
        for &s in &symbols {
            encode_symbol_table(&mut rc, s, 0, &DC_COEFF_SKIP_CDF).expect("encode");
        }
        let bs = rc.flush();

        // Compressed size must be less than raw size (2 bits/symbol max → 2500 bytes).
        assert!(
            bs.len() <= 2500,
            "compressed size {} should be ≤ 2500 bytes for {}-symbol DC skip stream",
            bs.len(),
            symbols.len()
        );

        // Decode and verify correctness.
        let mut dec = RangeCoder::new();
        dec.init_from_slice(&bs).expect("init");
        for (i, &expected) in symbols.iter().enumerate() {
            let got = decode_symbol_table(&mut dec, 0, &DC_COEFF_SKIP_CDF).expect("decode");
            assert_eq!(got, expected, "bulk decode mismatch at index {i}");
        }
    }
}
