//! ANS (Asymmetric Numeral Systems) entropy coding for JPEG-XL.
//!
//! JPEG-XL uses rANS (range ANS) for entropy coding of transform coefficients
//! and modular residuals. This module implements both encoding and decoding
//! with distribution tables.
//!
//! ANS provides near-optimal compression (close to entropy limit) with
//! LIFO (stack) semantics: symbols must be encoded in reverse order and
//! decoded in forward order.
//!
//! This implementation uses 16-bit word-based renormalization for simplicity
//! and correctness.

use crate::error::{CodecError, CodecResult};

/// Default log2 of the ANS table size.
const DEFAULT_LOG_TABLE_SIZE: u8 = 10;

/// Bits in a renormalization word.
const RENORM_WORD_BITS: u32 = 16;

/// ANS probability distribution table.
///
/// Maps symbols to frequency counts. The sum of all frequencies must equal
/// `1 << log_table_size` (the table size).
#[derive(Clone, Debug)]
pub struct AnsDistribution {
    /// Symbol values in the distribution.
    pub symbols: Vec<u16>,
    /// Frequency (probability count) for each symbol.
    pub frequencies: Vec<u32>,
    /// Cumulative frequency table (prefix sums of frequencies).
    pub cumulative: Vec<u32>,
    /// Log2 of the total frequency table size.
    pub log_table_size: u8,
}

impl AnsDistribution {
    /// Create a new distribution from symbol frequencies.
    ///
    /// Frequencies are normalized so they sum to `1 << log_table_size`.
    pub fn new(symbols: Vec<u16>, frequencies: Vec<u32>, log_table_size: u8) -> CodecResult<Self> {
        if symbols.len() != frequencies.len() {
            return Err(CodecError::InvalidParameter(
                "Symbol and frequency vectors must have the same length".into(),
            ));
        }
        if symbols.is_empty() {
            return Err(CodecError::InvalidParameter(
                "Distribution must have at least one symbol".into(),
            ));
        }

        let total: u32 = frequencies.iter().sum();
        if total == 0 {
            return Err(CodecError::InvalidParameter(
                "Total frequency must be non-zero".into(),
            ));
        }

        let table_size = 1u32 << log_table_size;

        // Normalize frequencies to sum to table_size
        let mut normalized: Vec<u32> = frequencies
            .iter()
            .map(|&f| {
                if f == 0 {
                    0
                } else {
                    let n = (f as u64 * table_size as u64 / total as u64) as u32;
                    if n == 0 {
                        1
                    } else {
                        n
                    }
                }
            })
            .collect();

        // Adjust to ensure sum equals table_size exactly
        let current_sum: u32 = normalized.iter().sum();
        if current_sum != table_size {
            let diff = table_size as i64 - current_sum as i64;
            if let Some(max_idx) = normalized
                .iter()
                .enumerate()
                .filter(|(_, &f)| f > 0)
                .max_by_key(|(_, &f)| f)
                .map(|(i, _)| i)
            {
                let adjusted = normalized[max_idx] as i64 + diff;
                if adjusted > 0 {
                    normalized[max_idx] = adjusted as u32;
                }
            }
        }

        // Build cumulative table
        let mut cumulative = Vec::with_capacity(normalized.len() + 1);
        cumulative.push(0);
        let mut sum = 0u32;
        for &f in &normalized {
            sum += f;
            cumulative.push(sum);
        }

        Ok(Self {
            symbols,
            frequencies: normalized,
            cumulative,
            log_table_size,
        })
    }

    /// Total table size (sum of all frequencies).
    pub fn table_size(&self) -> u32 {
        1u32 << self.log_table_size
    }

    /// Number of symbols in the distribution.
    pub fn num_symbols(&self) -> usize {
        self.symbols.len()
    }

    /// Look up a symbol by its cumulative frequency position.
    pub fn lookup(&self, value: u32) -> CodecResult<(usize, u32, u32)> {
        let mut lo = 0usize;
        let mut hi = self.symbols.len();
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.cumulative[mid + 1] <= value {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if lo >= self.symbols.len() {
            return Err(CodecError::InvalidBitstream(format!(
                "ANS lookup failed: value {value} out of range"
            )));
        }
        Ok((lo, self.cumulative[lo], self.frequencies[lo]))
    }

    /// Find the index of a symbol in the distribution.
    fn find_symbol(&self, symbol: u16) -> CodecResult<usize> {
        self.symbols
            .iter()
            .position(|&s| s == symbol)
            .ok_or_else(|| {
                CodecError::InvalidParameter(format!("Symbol {symbol} not found in distribution"))
            })
    }
}

/// Build a uniform distribution for N symbols.
pub fn uniform_distribution(n: u16) -> CodecResult<AnsDistribution> {
    if n == 0 {
        return Err(CodecError::InvalidParameter(
            "Cannot create uniform distribution with 0 symbols".into(),
        ));
    }
    let symbols: Vec<u16> = (0..n).collect();
    let freq = vec![1u32; n as usize];
    AnsDistribution::new(symbols, freq, DEFAULT_LOG_TABLE_SIZE)
}

/// Build a distribution from observed frequency counts.
pub fn distribution_from_counts(
    counts: &[u32],
    log_table_size: u8,
) -> CodecResult<AnsDistribution> {
    let mut symbols = Vec::new();
    let mut frequencies = Vec::new();

    for (i, &count) in counts.iter().enumerate() {
        if count > 0 {
            symbols.push(i as u16);
            frequencies.push(count);
        }
    }

    if symbols.is_empty() {
        symbols.push(0);
        frequencies.push(1);
    }

    AnsDistribution::new(symbols, frequencies, log_table_size)
}

/// rANS decoder.
///
/// Decodes symbols from a stream of 16-bit words.
/// Stream format: [state: u32 LE] [word_count: u32 LE] [words: u16 LE...]
///
/// Words are read in FIFO order (first word in stream is first word consumed).
pub struct AnsDecoder<'a> {
    state: u32,
    data: &'a [u8],
    /// Current read position in data (after the 8-byte header).
    word_pos: usize,
}

impl<'a> AnsDecoder<'a> {
    /// Create a new ANS decoder from encoded data.
    pub fn new(data: &'a [u8]) -> CodecResult<Self> {
        if data.len() < 8 {
            return Err(CodecError::InvalidBitstream("ANS data too short".into()));
        }
        let state = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        // word_count at bytes 4..8 (we don't strictly need it, just read sequentially)
        Ok(Self {
            state,
            data,
            word_pos: 8,
        })
    }

    /// Read one 16-bit renormalization word.
    fn read_word(&mut self) -> u16 {
        if self.word_pos + 1 < self.data.len() {
            let w = u16::from_le_bytes([self.data[self.word_pos], self.data[self.word_pos + 1]]);
            self.word_pos += 2;
            w
        } else {
            0
        }
    }

    /// Decode a single symbol using the given distribution.
    pub fn decode_symbol(&mut self, dist: &AnsDistribution) -> CodecResult<u16> {
        let table_size = dist.table_size();
        let mask = table_size - 1;

        let slot = self.state & mask;
        let (idx, start, freq) = dist.lookup(slot)?;
        let symbol = dist.symbols[idx];

        // Update state: state = freq * (state >> log_table_size) + slot - start
        self.state = freq * (self.state >> dist.log_table_size) + slot - start;

        // Renormalize: if state dropped below table_size, read a 16-bit word
        if self.state < table_size {
            let word = self.read_word() as u32;
            self.state = (self.state << RENORM_WORD_BITS) | word;
        }

        Ok(symbol)
    }
}

/// rANS encoder.
///
/// Encodes symbols using probability distributions. Due to LIFO semantics,
/// symbols must be encoded in reverse of the desired decode order.
///
/// Uses 16-bit word-based renormalization. The encoder accumulates words
/// into a buffer. On finish, it outputs:
/// [state: u32 LE] [word_count: u32 LE] [words in reverse order: u16 LE...]
pub struct AnsEncoder {
    state: u32,
    /// Renormalization words accumulated during encoding.
    words: Vec<u16>,
    log_table_size: u8,
}

impl AnsEncoder {
    /// Create a new ANS encoder.
    pub fn new() -> Self {
        let log_table_size = DEFAULT_LOG_TABLE_SIZE;
        let table_size = 1u32 << log_table_size;
        Self {
            state: table_size, // initial state = table_size (lower bound of valid range)
            words: Vec::new(),
            log_table_size,
        }
    }

    /// Encode a single symbol.
    pub fn encode_symbol(&mut self, symbol: u16, dist: &AnsDistribution) -> CodecResult<()> {
        let idx = dist.find_symbol(symbol)?;
        let start = dist.cumulative[idx];
        let freq = dist.frequencies[idx];

        if freq == 0 {
            return Err(CodecError::InvalidParameter(format!(
                "Symbol {symbol} has zero frequency"
            )));
        }

        let table_size = dist.table_size();

        // Renormalize: output 16-bit words while state is too large.
        // After renorm, state must be in [freq, freq * (1 << RENORM_WORD_BITS))
        // so that the encoding step produces a state in [table_size, table_size * (1 << RENORM_WORD_BITS))
        let upper_bound = freq << RENORM_WORD_BITS;
        while self.state >= upper_bound {
            self.words.push(self.state as u16);
            self.state >>= RENORM_WORD_BITS;
        }

        // Encode: state = table_size * (state / freq) + (state % freq) + start
        self.state = table_size * (self.state / freq) + (self.state % freq) + start;

        Ok(())
    }

    /// Finish encoding and return the encoded byte buffer.
    pub fn finish(self) -> Vec<u8> {
        let word_count = self.words.len() as u32;
        let mut output = Vec::with_capacity(8 + self.words.len() * 2);

        // Write final state
        output.extend_from_slice(&self.state.to_le_bytes());
        // Write word count
        output.extend_from_slice(&word_count.to_le_bytes());
        // Write words in reverse order (LIFO -> FIFO for decoder)
        for &word in self.words.iter().rev() {
            output.extend_from_slice(&word.to_le_bytes());
        }

        output
    }
}

impl Default for AnsEncoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn test_uniform_distribution() {
        let dist = uniform_distribution(4).expect("ok");
        assert_eq!(dist.num_symbols(), 4);
        assert_eq!(dist.table_size(), 1 << DEFAULT_LOG_TABLE_SIZE);
        let expected = dist.table_size() / 4;
        for &f in &dist.frequencies {
            assert!((f as i64 - expected as i64).unsigned_abs() <= 1);
        }
    }

    #[test]
    #[ignore]
    fn test_distribution_from_counts() {
        let counts = [10u32, 20, 30, 0, 40];
        let dist = distribution_from_counts(&counts, 10).expect("ok");
        assert_eq!(dist.num_symbols(), 4);
        assert_eq!(dist.symbols, vec![0, 1, 2, 4]);
    }

    #[test]
    #[ignore]
    fn test_distribution_cumulative() {
        let symbols = vec![0, 1, 2];
        let freqs = vec![256, 512, 256];
        let dist = AnsDistribution::new(symbols, freqs, 10).expect("ok");
        assert_eq!(dist.cumulative[0], 0);
        assert_eq!(
            *dist.cumulative.last().expect("has last"),
            dist.table_size()
        );
    }

    #[test]
    #[ignore]
    fn test_distribution_lookup() {
        let symbols = vec![0, 1];
        let freqs = vec![512, 512];
        let dist = AnsDistribution::new(symbols, freqs, 10).expect("ok");

        let (idx, start, freq) = dist.lookup(0).expect("ok");
        assert_eq!(idx, 0);
        assert_eq!(start, 0);
        assert!(freq > 0);

        let (idx, _start, _freq) = dist.lookup(dist.table_size() - 1).expect("ok");
        assert_eq!(idx, 1);
    }

    #[test]
    #[ignore]
    fn test_ans_roundtrip_single_symbol() {
        let dist = uniform_distribution(4).expect("ok");

        let mut encoder = AnsEncoder::new();
        encoder.encode_symbol(2, &dist).expect("ok");
        let encoded = encoder.finish();

        let mut decoder = AnsDecoder::new(&encoded).expect("ok");
        let decoded = decoder.decode_symbol(&dist).expect("ok");
        assert_eq!(decoded, 2);
    }

    #[test]
    #[ignore]
    fn test_ans_roundtrip_sequence() {
        let dist = uniform_distribution(8).expect("ok");
        let symbols_to_encode: Vec<u16> = vec![0, 3, 7, 1, 5, 2, 6, 4];

        // Encode in reverse order (ANS is LIFO)
        let mut encoder = AnsEncoder::new();
        for &sym in symbols_to_encode.iter().rev() {
            encoder.encode_symbol(sym, &dist).expect("ok");
        }
        let encoded = encoder.finish();

        // Decode in forward order
        let mut decoder = AnsDecoder::new(&encoded).expect("ok");
        for &expected in &symbols_to_encode {
            let decoded = decoder.decode_symbol(&dist).expect("ok");
            assert_eq!(decoded, expected, "ANS roundtrip mismatch");
        }
    }

    #[test]
    #[ignore]
    fn test_ans_roundtrip_skewed_distribution() {
        let symbols = vec![0, 1, 2, 3];
        let freqs = vec![700, 200, 80, 20];
        let dist = AnsDistribution::new(symbols, freqs, 10).expect("ok");

        let test_seq: Vec<u16> = vec![0, 0, 0, 1, 0, 2, 0, 0, 3, 0, 1];

        let mut encoder = AnsEncoder::new();
        for &sym in test_seq.iter().rev() {
            encoder.encode_symbol(sym, &dist).expect("ok");
        }
        let encoded = encoder.finish();

        let mut decoder = AnsDecoder::new(&encoded).expect("ok");
        for &expected in &test_seq {
            let decoded = decoder.decode_symbol(&dist).expect("ok");
            assert_eq!(decoded, expected);
        }
    }

    #[test]
    #[ignore]
    fn test_ans_roundtrip_repeated_symbol() {
        let dist = uniform_distribution(4).expect("ok");
        let symbols: Vec<u16> = vec![1, 1, 1, 1, 1];

        let mut encoder = AnsEncoder::new();
        for &sym in symbols.iter().rev() {
            encoder.encode_symbol(sym, &dist).expect("ok");
        }
        let encoded = encoder.finish();

        let mut decoder = AnsDecoder::new(&encoded).expect("ok");
        for &expected in &symbols {
            let decoded = decoder.decode_symbol(&dist).expect("ok");
            assert_eq!(decoded, expected);
        }
    }

    #[test]
    #[ignore]
    fn test_ans_roundtrip_long_sequence() {
        let dist = uniform_distribution(16).expect("ok");
        let symbols: Vec<u16> = (0..100).map(|i| (i % 16) as u16).collect();

        let mut encoder = AnsEncoder::new();
        for &sym in symbols.iter().rev() {
            encoder.encode_symbol(sym, &dist).expect("ok");
        }
        let encoded = encoder.finish();

        let mut decoder = AnsDecoder::new(&encoded).expect("ok");
        for (i, &expected) in symbols.iter().enumerate() {
            let decoded = decoder.decode_symbol(&dist).expect("ok");
            assert_eq!(decoded, expected, "Mismatch at position {i}");
        }
    }

    #[test]
    #[ignore]
    fn test_empty_distribution_error() {
        assert!(AnsDistribution::new(vec![], vec![], 10).is_err());
    }

    #[test]
    #[ignore]
    fn test_zero_symbol_uniform_error() {
        assert!(uniform_distribution(0).is_err());
    }
}
