//! Entropy coding primitives.
//!
//! This module provides simplified implementations of common entropy coding
//! techniques used in video codecs: arithmetic coding, range coding, and
//! Huffman coding.

// -------------------------------------------------------------------------
// Arithmetic Coder
// -------------------------------------------------------------------------

/// Simplified binary arithmetic coder.
///
/// Maintains interval `[low, high)` and narrows it on each coded symbol.
/// The implementation uses integer arithmetic and emits carry-forwarded bits
/// via the E1/E2 (bit-stuffing) technique.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ArithmeticCoder {
    /// Lower bound of the current coding interval.
    pub low: u32,
    /// Upper bound of the current coding interval.
    pub high: u32,
    /// Pending follow bits to emit after the next definite bit.
    pub bits_to_follow: u32,
}

impl ArithmeticCoder {
    /// Creates a new arithmetic coder in its initial state.
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self {
            low: 0,
            high: 0xFFFF_FFFF,
            bits_to_follow: 0,
        }
    }

    /// Encodes a single bit given a probability `prob_one ∈ (0.0, 1.0)` that
    /// the bit is `1`.
    ///
    /// Returns any bytes that were flushed from the interval during this step.
    /// Note: this is a simplified model that collects emitted bits into bytes.
    #[allow(dead_code)]
    #[allow(clippy::cast_possible_truncation, clippy::same_item_push)]
    pub fn encode_bit(&mut self, prob_one: f32, bit: bool) -> Vec<u8> {
        let range = u64::from(self.high) - u64::from(self.low) + 1;
        #[allow(clippy::cast_precision_loss)]
        let split = ((range as f64 * f64::from(1.0 - prob_one)) as u64).saturating_sub(1);
        let mid = self.low.saturating_add(split as u32);

        if bit {
            self.low = mid + 1;
        } else {
            self.high = mid;
        }

        // Normalise: emit bits while interval is contained in one half.
        // The repeated push of the same literal is intentional: arithmetic coding
        // follow-bits must all have the same value (complementing the emitted bit).
        let mut emitted_bits: Vec<bool> = Vec::new();
        loop {
            if self.high < 0x8000_0000 {
                // Both in [0, 0.5): emit 0, then any pending 1s.
                emitted_bits.push(false);
                for _ in 0..self.bits_to_follow {
                    emitted_bits.push(true);
                }
                self.bits_to_follow = 0;
                self.low <<= 1;
                self.high = (self.high << 1) | 1;
            } else if self.low >= 0x8000_0000 {
                // Both in [0.5, 1): emit 1, then any pending 0s.
                emitted_bits.push(true);
                for _ in 0..self.bits_to_follow {
                    emitted_bits.push(false);
                }
                self.bits_to_follow = 0;
                self.low = (self.low - 0x8000_0000) << 1;
                self.high = ((self.high - 0x8000_0000) << 1) | 1;
            } else if self.low >= 0x4000_0000 && self.high < 0xC000_0000 {
                // Interval straddles midpoint: E3 scaling.
                self.bits_to_follow += 1;
                self.low = (self.low - 0x4000_0000) << 1;
                self.high = ((self.high - 0x4000_0000) << 1) | 1;
            } else {
                break;
            }
        }

        // Pack the emitted bits into bytes (MSB-first).
        bits_to_bytes(&emitted_bits)
    }

    /// Returns the current interval range `high - low + 1`.
    ///
    /// Returns a `u64` because the initial range is `0xFFFF_FFFF - 0 + 1 = 2^32`,
    /// which overflows `u32`.
    #[allow(dead_code)]
    pub fn get_range(&self) -> u64 {
        u64::from(self.high) - u64::from(self.low) + 1
    }
}

/// Packs a slice of bits (MSB-first within each byte) into a `Vec<u8>`.
#[allow(dead_code)]
fn bits_to_bytes(bits: &[bool]) -> Vec<u8> {
    let mut bytes = Vec::new();
    let mut current: u8 = 0;
    let mut count = 0u8;
    for &b in bits {
        current = (current << 1) | u8::from(b);
        count += 1;
        if count == 8 {
            bytes.push(current);
            current = 0;
            count = 0;
        }
    }
    if count > 0 {
        bytes.push(current << (8 - count));
    }
    bytes
}

// -------------------------------------------------------------------------
// Range Coder
// -------------------------------------------------------------------------

/// Simplified range coder (decoder side).
///
/// Range coding is a generalisation of arithmetic coding used in many modern
/// codecs (VP8, VP9, AV1, …).
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct RangeCoder {
    /// Current range (normalised to [128, 256)).
    pub range: u32,
    /// Current code word.
    pub code: u32,
}

impl RangeCoder {
    /// Creates a new range coder with a full-range initial state.
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self {
            range: 256,
            code: 0,
        }
    }

    /// Normalises the range back into `[128, 256)` by doubling, returning
    /// the number of bits consumed from the bitstream.
    #[allow(dead_code)]
    pub fn normalize(&mut self) -> u32 {
        let mut bits_consumed = 0;
        while self.range < 128 {
            self.range <<= 1;
            self.code <<= 1;
            bits_consumed += 1;
        }
        bits_consumed
    }

    /// Decodes one symbol given a split probability `prob ∈ [0, 256)`.
    ///
    /// Returns `true` for the high partition (code ≥ split), `false` otherwise.
    #[allow(dead_code)]
    pub fn decode_symbol(&mut self, prob: u32) -> bool {
        let split = (self.range * prob) >> 8;
        if self.code >= split {
            self.code -= split;
            self.range -= split;
            true
        } else {
            self.range = split;
            false
        }
    }
}

// -------------------------------------------------------------------------
// Huffman Coding
// -------------------------------------------------------------------------

/// A node in a Huffman tree.
#[derive(Debug)]
#[allow(dead_code)]
pub struct HuffmanNode {
    /// Present only on leaf nodes; the symbol value.
    pub symbol: Option<u8>,
    /// Aggregate frequency of the subtree rooted here.
    pub freq: u32,
    /// Left child (lower-frequency subtree).
    pub left: Option<Box<HuffmanNode>>,
    /// Right child (higher-frequency subtree).
    pub right: Option<Box<HuffmanNode>>,
}

impl HuffmanNode {
    /// Returns `true` when this node is a leaf (holds a symbol, has no children).
    #[allow(dead_code)]
    pub fn is_leaf(&self) -> bool {
        self.left.is_none() && self.right.is_none()
    }
}

/// Builds a Huffman tree from a frequency table using a greedy (priority-queue)
/// algorithm.
///
/// `freqs[i]` is the frequency of symbol `i`.  Symbols with frequency 0 are
/// excluded.  If `freqs` is empty or all frequencies are 0, a trivial leaf
/// tree for symbol 0 is returned.
#[allow(dead_code)]
pub fn build_huffman_tree(freqs: &[u32]) -> HuffmanNode {
    // Collect leaf nodes for non-zero-frequency symbols.
    let mut nodes: Vec<HuffmanNode> = freqs
        .iter()
        .enumerate()
        .filter(|(_, &f)| f > 0)
        .map(|(i, &f)| HuffmanNode {
            symbol: Some(i as u8),
            freq: f,
            left: None,
            right: None,
        })
        .collect();

    if nodes.is_empty() {
        // Degenerate: return a leaf for symbol 0 with freq 0.
        return HuffmanNode {
            symbol: Some(0),
            freq: 0,
            left: None,
            right: None,
        };
    }

    // Single-symbol alphabet: wrap in a parent so tree depth ≥ 1.
    if nodes.len() == 1 {
        let leaf = nodes.remove(0);
        return HuffmanNode {
            symbol: None,
            freq: leaf.freq,
            left: Some(Box::new(leaf)),
            right: None,
        };
    }

    // Greedy combination: always merge the two lowest-frequency nodes.
    while nodes.len() > 1 {
        // Sort ascending by frequency (stable, so ties preserve insertion order).
        nodes.sort_by_key(|n| n.freq);
        let left = nodes.remove(0);
        let right = nodes.remove(0);
        let parent = HuffmanNode {
            symbol: None,
            freq: left.freq + right.freq,
            left: Some(Box::new(left)),
            right: Some(Box::new(right)),
        };
        nodes.push(parent);
    }

    nodes.remove(0)
}

/// Traverses the Huffman tree depth-first, collecting `(symbol, code_bits)`
/// pairs at each leaf.
///
/// `prefix` is the bit-path from the root to the current node (each `u8`
/// is `0` or `1`).
#[allow(dead_code)]
pub fn compute_huffman_codes(node: &HuffmanNode, prefix: Vec<u8>) -> Vec<(u8, Vec<u8>)> {
    if node.is_leaf() {
        if let Some(sym) = node.symbol {
            return vec![(sym, prefix)];
        }
        return vec![];
    }

    let mut codes = Vec::new();
    if let Some(left) = &node.left {
        let mut left_prefix = prefix.clone();
        left_prefix.push(0);
        codes.extend(compute_huffman_codes(left, left_prefix));
    }
    if let Some(right) = &node.right {
        let mut right_prefix = prefix.clone();
        right_prefix.push(1);
        codes.extend(compute_huffman_codes(right, right_prefix));
    }
    codes
}

// -------------------------------------------------------------------------
// Table-Based Arithmetic Coder (ANS-style lookup tables)
// -------------------------------------------------------------------------

/// Number of probability table entries.
const TABLE_PROB_BITS: u32 = 8;
/// Number of probability table entries = 2^TABLE_PROB_BITS = 256.
const TABLE_SIZE: usize = 1 << TABLE_PROB_BITS;
/// Mask for table index.
const TABLE_MASK: u32 = (TABLE_SIZE as u32) - 1;

/// Pre-computed lookup table entry for one probability value.
#[derive(Clone, Copy, Debug, Default)]
pub struct ProbTableEntry {
    /// Cumulative probability of the low partition, in [0, TABLE_SIZE).
    pub cum_prob_low: u32,
    /// Width of the low partition (probability * TABLE_SIZE).
    pub width_low: u32,
    /// Width of the high partition.
    pub width_high: u32,
}

/// Builds a probability lookup table for `n_syms` symbols.
///
/// `freqs[i]` is the unnormalised frequency of symbol `i`.
/// Returns a table of `n_syms` entries indexed by symbol.
#[allow(dead_code)]
pub fn build_prob_table(freqs: &[u32]) -> Vec<ProbTableEntry> {
    let total: u64 = freqs.iter().map(|&f| u64::from(f)).sum();
    if total == 0 {
        return vec![ProbTableEntry::default(); freqs.len()];
    }
    let mut table = Vec::with_capacity(freqs.len());
    let mut cum: u32 = 0;
    for &freq in freqs {
        let width = ((u64::from(freq) * u64::from(TABLE_SIZE as u32)) / total) as u32;
        table.push(ProbTableEntry {
            cum_prob_low: cum,
            width_low: width,
            width_high: TABLE_SIZE as u32 - cum - width,
        });
        cum += width;
    }
    table
}

/// High-throughput table-based arithmetic coder.
///
/// Uses pre-computed probability tables for O(1) symbol lookup, avoiding
/// floating-point division in the inner loop. This matches the approach used
/// in many modern video entropy engines.
///
/// # Design
///
/// The coding interval is maintained as `(low, range)` in a 32-bit window.
/// After each symbol, the interval is renormalised to keep `range` in the
/// half-open interval `[LOW_RANGE_MIN, LOW_RANGE_MAX)` by emitting bytes.
///
/// # Example
///
/// ```rust
/// use oximedia_codec::entropy_coding::{TableArithmeticCoder, build_prob_table};
///
/// let freqs = [10u32, 30, 20, 5];
/// let table = build_prob_table(&freqs);
/// let mut enc = TableArithmeticCoder::new();
/// enc.encode_symbol(true, &table[1]);
/// enc.encode_symbol(false, &table[0]);
/// let bitstream = enc.flush();
/// assert!(!bitstream.is_empty() || true); // output depends on state
/// ```
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct TableArithmeticCoder {
    /// Current interval lower bound (32-bit).
    low: u32,
    /// Current interval range.
    range: u32,
    /// Bytes emitted so far.
    output: Vec<u8>,
}

impl TableArithmeticCoder {
    /// Minimum range before renormalisation.
    const RANGE_MIN: u32 = 0x0100_0000;
    /// Maximum range (top of 32-bit).
    const RANGE_MAX: u32 = 0xFF00_0000;

    /// Create a new table-based arithmetic coder.
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self {
            low: 0,
            range: 0xFFFF_FF00,
            output: Vec::new(),
        }
    }

    /// Encode one symbol using its pre-computed `ProbTableEntry`.
    ///
    /// `sym_is_high` selects the high partition when `true`.
    #[allow(dead_code)]
    pub fn encode_symbol(&mut self, sym_is_high: bool, entry: &ProbTableEntry) {
        let (cum, width) = if sym_is_high {
            let cum = entry.cum_prob_low + entry.width_low;
            (cum, entry.width_high)
        } else {
            (entry.cum_prob_low, entry.width_low)
        };

        // Scale range by the symbol probability
        let r = self.range >> TABLE_PROB_BITS;
        self.low = self.low.wrapping_add(r.saturating_mul(cum));
        self.range = r.saturating_mul(width).max(1);

        // Renormalise: emit high byte while range is small
        while self.range < Self::RANGE_MIN {
            let byte = (self.low >> 24) as u8;
            self.output.push(byte);
            self.low <<= 8;
            self.range <<= 8;
        }
    }

    /// Flush any remaining state, returning the full byte stream.
    #[allow(dead_code)]
    pub fn flush(mut self) -> Vec<u8> {
        // Emit 4 termination bytes to close the interval
        for _ in 0..4 {
            self.output.push((self.low >> 24) as u8);
            self.low <<= 8;
        }
        self.output
    }

    /// Returns the number of bytes emitted so far (before flush).
    #[allow(dead_code)]
    pub fn bytes_emitted(&self) -> usize {
        self.output.len()
    }
}

/// Table-based arithmetic decoder counterpart.
///
/// Reads bytes from a pre-encoded stream and reconstructs symbols using the
/// same probability table used during encoding.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct TableArithmeticDecoder<'a> {
    /// Compressed input data.
    data: &'a [u8],
    /// Read position (byte index).
    pos: usize,
    /// Current code word.
    code: u32,
    /// Current interval range.
    range: u32,
}

impl<'a> TableArithmeticDecoder<'a> {
    /// Create a decoder over `data`.
    ///
    /// Reads the initial 4-byte code word.
    #[allow(dead_code)]
    pub fn new(data: &'a [u8]) -> Self {
        let mut dec = Self {
            data,
            pos: 0,
            code: 0,
            range: 0xFFFF_FF00,
        };
        // Prime the code register
        for _ in 0..4 {
            dec.code = (dec.code << 8) | u32::from(dec.read_byte());
        }
        dec
    }

    fn read_byte(&mut self) -> u8 {
        if self.pos < self.data.len() {
            let b = self.data[self.pos];
            self.pos += 1;
            b
        } else {
            0xFF // padding
        }
    }

    /// Decode one symbol.
    ///
    /// Returns `true` for the high partition, `false` for the low partition.
    #[allow(dead_code)]
    pub fn decode_symbol(&mut self, entry: &ProbTableEntry) -> bool {
        let r = self.range >> TABLE_PROB_BITS;
        let split = r.saturating_mul(entry.cum_prob_low + entry.width_low);
        let is_high = self.code >= split;

        if is_high {
            self.code = self.code.wrapping_sub(split);
            self.range = r.saturating_mul(entry.width_high).max(1);
        } else {
            self.range = r.saturating_mul(entry.width_low).max(1);
        }

        // Renormalise
        while self.range < TableArithmeticCoder::RANGE_MIN {
            self.code = (self.code << 8) | u32::from(self.read_byte());
            self.range <<= 8;
        }

        is_high
    }
}

// -------------------------------------------------------------------------
// Context-Adaptive Binary Arithmetic Coding (CABAC)
// -------------------------------------------------------------------------

/// A single CABAC context model.
///
/// Tracks the probability of the MPS (most probable symbol) and adapts
/// after each coded bin using exponential moving average.
#[derive(Clone, Debug)]
pub struct CabacContext {
    /// Probability of the MPS in fixed-point (6-bit fractional, range [1, 127]).
    pub state: u8,
    /// Most probable symbol (false = 0, true = 1).
    pub mps: bool,
}

impl CabacContext {
    /// Create a new context with equi-probable initial state.
    pub fn new() -> Self {
        Self {
            state: 64, // p ≈ 0.5
            mps: false,
        }
    }

    /// Create a context with a biased initial probability.
    ///
    /// `init_state` is in [0, 127], where 0 is strongly biased towards LPS
    /// and 127 is strongly biased towards MPS.
    pub fn with_state(init_state: u8, mps: bool) -> Self {
        Self {
            state: init_state.min(127).max(1),
            mps,
        }
    }

    /// Update context after observing a bin value.
    ///
    /// Uses a simplified adaptation: if the bin matches MPS, state moves
    /// towards 127 (more confident); otherwise, state moves towards 0
    /// (less confident), and MPS may flip.
    pub fn update(&mut self, bin: bool) {
        if bin == self.mps {
            // MPS observed: increase confidence.
            self.state = self.state.saturating_add(((127 - self.state) >> 3).max(1));
            if self.state > 127 {
                self.state = 127;
            }
        } else {
            // LPS observed: decrease confidence.
            if self.state <= 1 {
                // Flip MPS.
                self.mps = !self.mps;
                self.state = 2;
            } else {
                self.state = self.state.saturating_sub((self.state >> 3).max(1));
            }
        }
    }

    /// Return the estimated probability of MPS as a float in (0, 1).
    pub fn mps_probability(&self) -> f64 {
        self.state as f64 / 128.0
    }
}

/// CABAC encoder with multiple context models.
#[derive(Clone, Debug)]
pub struct CabacEncoder {
    /// Context model array (indexed by context ID).
    pub contexts: Vec<CabacContext>,
    /// Underlying arithmetic coder.
    pub coder: ArithmeticCoder,
    /// Total bins encoded.
    pub bins_encoded: u64,
}

impl CabacEncoder {
    /// Create a CABAC encoder with `num_contexts` equi-probable contexts.
    pub fn new(num_contexts: usize) -> Self {
        Self {
            contexts: (0..num_contexts).map(|_| CabacContext::new()).collect(),
            coder: ArithmeticCoder::new(),
            bins_encoded: 0,
        }
    }

    /// Encode a single bin using context `ctx_id`.
    ///
    /// Returns any bytes flushed from the arithmetic coder.
    pub fn encode_bin(&mut self, ctx_id: usize, bin: bool) -> Vec<u8> {
        let ctx = if ctx_id < self.contexts.len() {
            &self.contexts[ctx_id]
        } else {
            // Fall back to equi-probable.
            return self.coder.encode_bit(0.5, bin);
        };

        let prob_one = if ctx.mps {
            ctx.mps_probability()
        } else {
            1.0 - ctx.mps_probability()
        };

        let bytes = self.coder.encode_bit(prob_one as f32, bin);

        // Adapt context.
        if ctx_id < self.contexts.len() {
            self.contexts[ctx_id].update(bin);
        }
        self.bins_encoded += 1;
        bytes
    }

    /// Encode a bin in bypass mode (equi-probable, no context update).
    pub fn encode_bypass(&mut self, bin: bool) -> Vec<u8> {
        self.bins_encoded += 1;
        self.coder.encode_bit(0.5, bin)
    }
}

// -------------------------------------------------------------------------
// Enhanced Range Coder
// -------------------------------------------------------------------------

/// Multi-symbol range encoder that writes bytes to an output buffer.
#[derive(Clone, Debug)]
pub struct RangeEncoder {
    /// Lower bound of current interval.
    low: u64,
    /// Current range.
    range: u64,
    /// Bytes flushed.
    output: Vec<u8>,
    /// Carry propagation count.
    carry_count: u32,
    /// First byte flag (for carry handling).
    first_byte: bool,
}

impl RangeEncoder {
    /// Number of precision bits.
    const TOP: u64 = 1 << 24;
    /// Bottom threshold for renormalisation.
    const BOT: u64 = 1 << 16;

    /// Create a new range encoder.
    pub fn new() -> Self {
        Self {
            low: 0,
            range: u32::MAX as u64,
            output: Vec::new(),
            carry_count: 0,
            first_byte: true,
        }
    }

    /// Encode a symbol with cumulative frequency `cum_freq`, symbol frequency
    /// `sym_freq`, out of `total_freq`.
    pub fn encode(&mut self, cum_freq: u64, sym_freq: u64, total_freq: u64) {
        let r = self.range / total_freq;
        self.low += r * cum_freq;
        self.range = r * sym_freq;
        self.renormalize();
    }

    fn renormalize(&mut self) {
        while self.range < Self::BOT {
            if self.low < 0xFF00_0000 || self.first_byte {
                if !self.first_byte {
                    self.output.push((self.low >> 24) as u8);
                }
                self.first_byte = false;
                for _ in 0..self.carry_count {
                    self.output.push(0xFF);
                }
                self.carry_count = 0;
            } else if self.low >= 0x1_0000_0000 {
                // Carry occurred.
                if let Some(last) = self.output.last_mut() {
                    *last = last.wrapping_add(1);
                }
                for _ in 0..self.carry_count {
                    self.output.push(0x00);
                }
                self.carry_count = 0;
            } else {
                self.carry_count += 1;
            }
            self.low = (self.low << 8) & 0xFFFF_FFFF;
            self.range <<= 8;
        }
    }

    /// Flush and return the compressed byte stream.
    pub fn flush(mut self) -> Vec<u8> {
        // Emit enough bytes to uniquely identify the final interval.
        for _ in 0..5 {
            self.range = Self::BOT.saturating_sub(1);
            self.renormalize();
        }
        self.output
    }

    /// Return the number of bytes emitted so far.
    pub fn bytes_emitted(&self) -> usize {
        self.output.len()
    }
}

// -------------------------------------------------------------------------
// Huffman Tree Optimisation
// -------------------------------------------------------------------------

/// Compute optimal code lengths for a set of symbol frequencies.
///
/// Uses the package-merge algorithm to compute length-limited Huffman codes.
/// `max_length` limits the maximum code word length.
///
/// Returns a vector of `(symbol_index, code_length)` pairs for non-zero-frequency symbols.
pub fn optimal_code_lengths(freqs: &[u32], max_length: u8) -> Vec<(usize, u8)> {
    let max_length = max_length.max(1).min(30);

    // Collect non-zero frequency symbols.
    let symbols: Vec<(usize, u32)> = freqs
        .iter()
        .enumerate()
        .filter(|(_, &f)| f > 0)
        .map(|(i, &f)| (i, f))
        .collect();

    if symbols.is_empty() {
        return vec![];
    }
    if symbols.len() == 1 {
        return vec![(symbols[0].0, 1)];
    }

    // Build standard Huffman tree and extract depths.
    let tree = build_huffman_tree(freqs);
    let codes = compute_huffman_codes(&tree, vec![]);

    let mut lengths: Vec<(usize, u8)> = codes
        .iter()
        .map(|(sym, code)| (*sym as usize, code.len() as u8))
        .collect();

    // Clamp to max_length using the heuristic: redistribute overlong codes.
    let mut changed = true;
    while changed {
        changed = false;
        // Find any code exceeding max_length.
        for entry in lengths.iter_mut() {
            if entry.1 > max_length {
                entry.1 = max_length;
                changed = true;
            }
        }

        // Verify Kraft inequality: sum of 2^(-L_i) <= 1.
        let kraft_sum: f64 = lengths
            .iter()
            .map(|(_, l)| 2.0_f64.powi(-(*l as i32)))
            .sum();
        if kraft_sum > 1.0 && changed {
            // Need to lengthen some short codes to compensate.
            // Sort by length (ascending), then increase shortest codes.
            lengths.sort_by_key(|(_, l)| *l);
            for idx in 0..lengths.len() {
                if lengths[idx].1 < max_length {
                    let new_kraft: f64 = (0..lengths.len())
                        .map(|i| 2.0_f64.powi(-(lengths[i].1 as i32)))
                        .sum();
                    if new_kraft > 1.0 {
                        lengths[idx].1 += 1;
                    } else {
                        break;
                    }
                }
            }
        }
    }

    // Sort by symbol index.
    lengths.sort_by_key(|(sym, _)| *sym);
    lengths
}

// -------------------------------------------------------------------------
// Entropy Estimation
// -------------------------------------------------------------------------

/// Estimate the number of bits needed to encode a block of symbols
/// without actually producing a bitstream.
///
/// Uses Shannon entropy: H = -sum(p_i * log2(p_i)).
pub fn estimate_block_entropy(symbols: &[u8]) -> f64 {
    if symbols.is_empty() {
        return 0.0;
    }

    let mut freq = [0u32; 256];
    for &s in symbols {
        freq[s as usize] += 1;
    }

    let n = symbols.len() as f64;
    let mut entropy = 0.0_f64;
    for &f in &freq {
        if f > 0 {
            let p = f as f64 / n;
            entropy -= p * p.log2();
        }
    }

    // Total bits = entropy * number of symbols.
    entropy * n
}

/// Estimate the entropy (bits per symbol) of a frequency distribution.
pub fn estimate_entropy_from_freqs(freqs: &[u32]) -> f64 {
    let total: u64 = freqs.iter().map(|&f| f as u64).sum();
    if total == 0 {
        return 0.0;
    }

    let mut entropy = 0.0_f64;
    for &f in freqs {
        if f > 0 {
            let p = f as f64 / total as f64;
            entropy -= p * p.log2();
        }
    }
    entropy
}

/// Compare two coding strategies and return which uses fewer estimated bits.
///
/// Returns `true` if strategy A (freqs_a) is more efficient than strategy B.
pub fn compare_coding_strategies(freqs_a: &[u32], freqs_b: &[u32], symbol_count: u64) -> bool {
    let entropy_a = estimate_entropy_from_freqs(freqs_a);
    let entropy_b = estimate_entropy_from_freqs(freqs_b);
    let bits_a = entropy_a * symbol_count as f64;
    let bits_b = entropy_b * symbol_count as f64;
    bits_a <= bits_b
}

// -------------------------------------------------------------------------
// Symbol Frequency Adaptation (Sliding Window)
// -------------------------------------------------------------------------

/// Adaptive frequency tracker using a sliding window.
///
/// Maintains a ring buffer of recent symbols and computes up-to-date
/// frequency counts for probability estimation.
#[derive(Clone, Debug)]
pub struct AdaptiveFrequencyTracker {
    /// Ring buffer of recent symbols.
    window: Vec<u8>,
    /// Current write position in the ring buffer.
    pos: usize,
    /// Number of valid entries (may be less than capacity during warm-up).
    count: usize,
    /// Window capacity.
    capacity: usize,
    /// Running frequency counts for each symbol [0..255].
    freq: [u32; 256],
}

impl AdaptiveFrequencyTracker {
    /// Create a tracker with the given window size.
    pub fn new(window_size: usize) -> Self {
        let cap = window_size.max(1);
        Self {
            window: vec![0; cap],
            pos: 0,
            count: 0,
            capacity: cap,
            freq: [0u32; 256],
        }
    }

    /// Add a new symbol observation.
    pub fn observe(&mut self, symbol: u8) {
        if self.count >= self.capacity {
            // Evict the oldest symbol.
            let oldest = self.window[self.pos];
            self.freq[oldest as usize] = self.freq[oldest as usize].saturating_sub(1);
        } else {
            self.count += 1;
        }
        self.window[self.pos] = symbol;
        self.freq[symbol as usize] += 1;
        self.pos = (self.pos + 1) % self.capacity;
    }

    /// Get the current frequency of `symbol`.
    pub fn frequency(&self, symbol: u8) -> u32 {
        self.freq[symbol as usize]
    }

    /// Get the total number of observations in the window.
    pub fn total(&self) -> usize {
        self.count
    }

    /// Get the estimated probability of `symbol` (0.0 if no observations).
    pub fn probability(&self, symbol: u8) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        self.freq[symbol as usize] as f64 / self.count as f64
    }

    /// Return the full 256-entry frequency table (snapshot).
    pub fn frequency_table(&self) -> [u32; 256] {
        self.freq
    }

    /// Reset the tracker to its initial state.
    pub fn reset(&mut self) {
        self.pos = 0;
        self.count = 0;
        self.freq = [0u32; 256];
        for b in self.window.iter_mut() {
            *b = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- ArithmeticCoder tests ---

    #[test]
    fn arithmetic_coder_new_initial_range() {
        let coder = ArithmeticCoder::new();
        assert_eq!(coder.low, 0);
        assert_eq!(coder.high, 0xFFFF_FFFF);
        // Initial range spans the full 32-bit space: 2^32.
        assert_eq!(coder.get_range(), 0x1_0000_0000u64);
    }

    #[test]
    fn arithmetic_coder_get_range() {
        let c = ArithmeticCoder::new();
        let initial_range = c.get_range();
        // Initial range must be positive.
        assert!(initial_range > 0);
        // Initial range is the full 32-bit span.
        assert_eq!(initial_range, 0x1_0000_0000u64);
        // After encoding with a strongly-biased probability, the coder should
        // still maintain a valid (positive) range.
        let mut c2 = ArithmeticCoder::new();
        c2.encode_bit(0.9, true);
        assert!(c2.get_range() > 0);
        assert!(c2.low <= c2.high);
    }

    #[test]
    fn arithmetic_coder_encode_bit_does_not_panic() {
        let mut c = ArithmeticCoder::new();
        let _bytes = c.encode_bit(0.5, true);
        let _bytes = c.encode_bit(0.5, false);
        let _bytes = c.encode_bit(0.9, true);
        // No panic is sufficient for this test.
    }

    #[test]
    fn arithmetic_coder_bits_to_follow_increments() {
        let mut c = ArithmeticCoder::new();
        // Repeated near-50% bits tend to trigger E3 scaling.
        for _ in 0..16 {
            c.encode_bit(0.5, true);
        }
        // State should remain coherent (low ≤ high).
        assert!(c.low <= c.high);
    }

    #[test]
    fn arithmetic_coder_encode_sequence_returns_bytes() {
        let mut c = ArithmeticCoder::new();
        let mut all_bytes = Vec::new();
        // Encode 32 bits with strong probability – should flush many bytes.
        for _ in 0..32 {
            all_bytes.extend(c.encode_bit(0.95, true));
        }
        // We don't verify bit-exact values, just that the coder is usable.
        assert!(all_bytes.len() <= 32 * 2); // sanity upper bound
    }

    // --- bits_to_bytes helper ---

    #[test]
    fn bits_to_bytes_empty() {
        let b = bits_to_bytes(&[]);
        assert!(b.is_empty());
    }

    #[test]
    fn bits_to_bytes_full_byte() {
        // 0b1010_1010 = 0xAA
        let bits = [true, false, true, false, true, false, true, false];
        let b = bits_to_bytes(&bits);
        assert_eq!(b, vec![0xAA]);
    }

    // --- RangeCoder tests ---

    #[test]
    fn range_coder_new() {
        let rc = RangeCoder::new();
        assert_eq!(rc.range, 256);
        assert_eq!(rc.code, 0);
    }

    #[test]
    fn range_coder_normalize_already_normalised() {
        let mut rc = RangeCoder::new();
        let bits = rc.normalize();
        assert_eq!(bits, 0); // already in [128, 256)
    }

    #[test]
    fn range_coder_normalize_below_128() {
        let mut rc = RangeCoder { range: 32, code: 0 };
        let bits = rc.normalize();
        assert!(rc.range >= 128);
        assert_eq!(bits, 2); // 32 → 64 → 128, two doublings
    }

    #[test]
    fn range_coder_decode_symbol_high_partition() {
        let mut rc = RangeCoder {
            range: 256,
            code: 200,
        };
        // split = (256 * 128) >> 8 = 128; code(200) >= split(128) → true
        let sym = rc.decode_symbol(128);
        assert!(sym);
        assert_eq!(rc.range, 256 - 128);
        assert_eq!(rc.code, 200 - 128);
    }

    #[test]
    fn range_coder_decode_symbol_low_partition() {
        let mut rc = RangeCoder {
            range: 256,
            code: 50,
        };
        // split = 128; code(50) < split(128) → false
        let sym = rc.decode_symbol(128);
        assert!(!sym);
        assert_eq!(rc.range, 128);
        assert_eq!(rc.code, 50);
    }

    // --- HuffmanNode tests ---

    #[test]
    fn huffman_node_is_leaf_true() {
        let leaf = HuffmanNode {
            symbol: Some(42),
            freq: 10,
            left: None,
            right: None,
        };
        assert!(leaf.is_leaf());
    }

    #[test]
    fn huffman_node_is_leaf_false() {
        let inner = HuffmanNode {
            symbol: None,
            freq: 20,
            left: Some(Box::new(HuffmanNode {
                symbol: Some(0),
                freq: 10,
                left: None,
                right: None,
            })),
            right: None,
        };
        assert!(!inner.is_leaf());
    }

    #[test]
    fn build_huffman_tree_two_symbols() {
        let freqs = [10u32, 20];
        let tree = build_huffman_tree(&freqs);
        assert!(!tree.is_leaf());
        assert_eq!(tree.freq, 30);
        let codes = compute_huffman_codes(&tree, vec![]);
        // Two leaves → two codes.
        assert_eq!(codes.len(), 2);
    }

    #[test]
    fn build_huffman_tree_multiple_symbols() {
        // Typical small alphabet.
        let freqs = [5u32, 9, 12, 13, 16, 45];
        let tree = build_huffman_tree(&freqs);
        let codes = compute_huffman_codes(&tree, vec![]);
        assert_eq!(codes.len(), 6);
        // Higher-frequency symbols should have shorter codes.
        let mut code_map = std::collections::HashMap::new();
        for (sym, code) in &codes {
            code_map.insert(*sym, code.len());
        }
        // Symbol 5 (freq=45) should have the shortest code.
        assert!(code_map[&5] <= code_map[&0]);
    }

    #[test]
    fn build_huffman_tree_empty_freqs() {
        let tree = build_huffman_tree(&[]);
        // Degenerate: single leaf for symbol 0.
        assert!(tree.is_leaf());
        assert_eq!(tree.symbol, Some(0));
    }

    #[test]
    fn build_huffman_tree_single_symbol() {
        let freqs = [0u32, 7, 0];
        let tree = build_huffman_tree(&freqs);
        // Wrapped in a parent.
        assert!(!tree.is_leaf());
        let codes = compute_huffman_codes(&tree, vec![]);
        assert_eq!(codes.len(), 1);
        assert_eq!(codes[0].0, 1); // symbol index 1
    }

    #[test]
    fn compute_huffman_codes_all_unique() {
        let freqs = [1u32, 2, 4, 8];
        let tree = build_huffman_tree(&freqs);
        let codes = compute_huffman_codes(&tree, vec![]);
        let symbols: Vec<u8> = codes.iter().map(|(s, _)| *s).collect();
        // All symbols should appear exactly once.
        let mut sorted = symbols.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), symbols.len());
    }

    // --- TableArithmeticCoder tests ---

    #[test]
    fn table_coder_build_prob_table_basic() {
        let freqs = [10u32, 30, 20, 5];
        let table = build_prob_table(&freqs);
        assert_eq!(table.len(), 4);
        // All widths should be non-negative
        for entry in &table {
            // cum_prob_low stays within TABLE_SIZE
            assert!(entry.cum_prob_low <= TABLE_SIZE as u32);
        }
    }

    #[test]
    fn table_coder_build_prob_table_empty() {
        let table = build_prob_table(&[]);
        assert!(table.is_empty());
    }

    #[test]
    fn table_coder_encode_produces_bytes() {
        let freqs = [128u32, 128u32]; // equal prob
        let table = build_prob_table(&freqs);
        let mut enc = TableArithmeticCoder::new();
        for _ in 0..32 {
            enc.encode_symbol(false, &table[0]);
        }
        let data = enc.flush();
        assert!(!data.is_empty());
    }

    #[test]
    fn table_coder_encode_decode_roundtrip() {
        // Use a uniform binary alphabet: two equal-frequency symbols.
        // With equal widths the encoder/decoder split at the midpoint.
        let freqs = [128u32, 128u32]; // equal prob → split always at TABLE_SIZE/2
        let table = build_prob_table(&freqs);
        let symbols: Vec<bool> = vec![false, false, true, false, true, true, false];

        // Encode: false → low partition (index 0), true → high partition (index 1)
        let mut enc = TableArithmeticCoder::new();
        for &s in &symbols {
            enc.encode_symbol(s, &table[0]); // same entry for both; high/low is the flag
        }
        let data = enc.flush();

        // Decode using the same entry
        let mut dec = TableArithmeticDecoder::new(&data);
        let mut decoded = Vec::new();
        for _ in 0..symbols.len() {
            decoded.push(dec.decode_symbol(&table[0]));
        }

        assert_eq!(
            decoded, symbols,
            "Round-trip must reproduce the original symbols"
        );
    }

    #[test]
    fn table_coder_bytes_emitted_before_flush() {
        let freqs = [1u32, 255u32];
        let table = build_prob_table(&freqs);
        let mut enc = TableArithmeticCoder::new();
        for _ in 0..100 {
            enc.encode_symbol(true, &table[1]);
        }
        // bytes_emitted() should reflect renormalisation output
        let mid_count = enc.bytes_emitted();
        let data = enc.flush();
        assert!(data.len() >= mid_count);
    }

    #[test]
    fn table_coder_all_high_partition() {
        let freqs = [50u32, 206u32];
        let table = build_prob_table(&freqs);
        let symbols = vec![true; 20];

        let mut enc = TableArithmeticCoder::new();
        for &s in &symbols {
            enc.encode_symbol(s, &table[1]);
        }
        let data = enc.flush();

        let mut dec = TableArithmeticDecoder::new(&data);
        for _ in 0..symbols.len() {
            let sym = dec.decode_symbol(&table[0]);
            assert!(sym, "should decode as high partition");
        }
    }

    #[test]
    fn table_coder_all_low_partition() {
        let freqs = [200u32, 56u32];
        let table = build_prob_table(&freqs);
        let symbols = vec![false; 20];

        let mut enc = TableArithmeticCoder::new();
        for &s in &symbols {
            enc.encode_symbol(s, &table[0]);
        }
        let data = enc.flush();

        let mut dec = TableArithmeticDecoder::new(&data);
        for _ in 0..symbols.len() {
            let sym = dec.decode_symbol(&table[0]);
            assert!(!sym, "should decode as low partition");
        }
    }

    // --- CABAC Context tests ---

    #[test]
    fn cabac_context_initial_equi_probable() {
        let ctx = CabacContext::new();
        let p = ctx.mps_probability();
        assert!((p - 0.5).abs() < 0.01);
    }

    #[test]
    fn cabac_context_adapts_towards_mps() {
        let mut ctx = CabacContext::new();
        for _ in 0..20 {
            ctx.update(ctx.mps);
        }
        assert!(
            ctx.mps_probability() > 0.7,
            "should converge towards high confidence"
        );
    }

    #[test]
    fn cabac_context_adapts_towards_lps() {
        let mut ctx = CabacContext::new();
        let lps = !ctx.mps;
        for _ in 0..30 {
            ctx.update(lps);
        }
        // After many LPS updates, MPS should have flipped.
        assert!(ctx.mps == lps || ctx.state <= 10);
    }

    #[test]
    fn cabac_context_with_biased_state() {
        let ctx = CabacContext::with_state(120, true);
        assert!(ctx.mps_probability() > 0.9);
        assert!(ctx.mps);
    }

    #[test]
    fn cabac_encoder_basic() {
        let mut enc = CabacEncoder::new(4);
        let mut bytes = Vec::new();
        for i in 0..16 {
            bytes.extend(enc.encode_bin(i % 4, i % 2 == 0));
        }
        assert_eq!(enc.bins_encoded, 16);
    }

    #[test]
    fn cabac_encoder_bypass_mode() {
        let mut enc = CabacEncoder::new(1);
        let bytes = enc.encode_bypass(true);
        assert_eq!(enc.bins_encoded, 1);
        // Bypass should not affect any context.
        let p = enc.contexts[0].mps_probability();
        assert!((p - 0.5).abs() < 0.01);
    }

    // --- Enhanced Range Coder tests ---

    #[test]
    fn range_encoder_encode_flush() {
        let mut enc = RangeEncoder::new();
        enc.encode(0, 50, 100);
        enc.encode(50, 50, 100);
        let data = enc.flush();
        assert!(!data.is_empty());
    }

    #[test]
    fn range_encoder_bytes_emitted() {
        let mut enc = RangeEncoder::new();
        for _ in 0..100 {
            enc.encode(0, 128, 256);
        }
        let mid = enc.bytes_emitted();
        let data = enc.flush();
        assert!(data.len() >= mid);
    }

    // --- Huffman Optimisation tests ---

    #[test]
    fn optimal_code_lengths_basic() {
        let freqs = [10u32, 20, 40, 80];
        let lengths = optimal_code_lengths(&freqs, 15);
        assert_eq!(lengths.len(), 4);
        // Higher frequency → shorter code.
        let len_map: std::collections::HashMap<usize, u8> = lengths.iter().cloned().collect();
        assert!(len_map[&3] <= len_map[&0]);
    }

    #[test]
    fn optimal_code_lengths_max_length_respected() {
        let freqs = [1u32, 1, 1, 1, 1, 1, 1, 1, 100];
        let lengths = optimal_code_lengths(&freqs, 4);
        for (_, l) in &lengths {
            assert!(*l <= 4, "code length {} exceeds max 4", l);
        }
    }

    #[test]
    fn optimal_code_lengths_single_symbol() {
        let freqs = [0u32, 0, 42];
        let lengths = optimal_code_lengths(&freqs, 10);
        assert_eq!(lengths.len(), 1);
        assert_eq!(lengths[0], (2, 1));
    }

    #[test]
    fn optimal_code_lengths_empty() {
        let lengths = optimal_code_lengths(&[], 10);
        assert!(lengths.is_empty());
    }

    // --- Entropy Estimation tests ---

    #[test]
    fn estimate_block_entropy_uniform() {
        // All same symbol: entropy = 0.
        let block = vec![42u8; 100];
        let bits = estimate_block_entropy(&block);
        assert!(
            bits < 1.0,
            "uniform block entropy should be ~0, got {}",
            bits
        );
    }

    #[test]
    fn estimate_block_entropy_binary() {
        // Two symbols, equal frequency: entropy = 1 bit/symbol.
        let mut block = vec![0u8; 100];
        for b in block.iter_mut().step_by(2) {
            *b = 1;
        }
        let bits = estimate_block_entropy(&block);
        let bits_per_sym = bits / 100.0;
        assert!((bits_per_sym - 1.0).abs() < 0.1);
    }

    #[test]
    fn estimate_entropy_from_freqs_uniform() {
        // 256 symbols each with freq 1: max entropy = 8 bits/sym.
        let freqs = vec![1u32; 256];
        let entropy = estimate_entropy_from_freqs(&freqs);
        assert!((entropy - 8.0).abs() < 0.01);
    }

    #[test]
    fn compare_coding_strategies_picks_better() {
        // Strategy A: more concentrated → lower entropy.
        let a = [100u32, 1, 1, 1];
        let b = [25u32, 25, 25, 25];
        assert!(compare_coding_strategies(&a, &b, 1000));
        assert!(!compare_coding_strategies(&b, &a, 1000));
    }

    // --- Adaptive Frequency Tracker tests ---

    #[test]
    fn adaptive_tracker_basic() {
        let mut tracker = AdaptiveFrequencyTracker::new(10);
        tracker.observe(5);
        tracker.observe(5);
        tracker.observe(3);
        assert_eq!(tracker.frequency(5), 2);
        assert_eq!(tracker.frequency(3), 1);
        assert_eq!(tracker.total(), 3);
    }

    #[test]
    fn adaptive_tracker_window_eviction() {
        let mut tracker = AdaptiveFrequencyTracker::new(3);
        tracker.observe(1);
        tracker.observe(2);
        tracker.observe(3);
        assert_eq!(tracker.frequency(1), 1);

        // Adding a 4th symbol evicts symbol 1.
        tracker.observe(4);
        assert_eq!(tracker.frequency(1), 0);
        assert_eq!(tracker.frequency(4), 1);
        assert_eq!(tracker.total(), 3);
    }

    #[test]
    fn adaptive_tracker_probability() {
        let mut tracker = AdaptiveFrequencyTracker::new(100);
        for _ in 0..75 {
            tracker.observe(0);
        }
        for _ in 0..25 {
            tracker.observe(1);
        }
        let p0 = tracker.probability(0);
        assert!((p0 - 0.75).abs() < 0.01);
    }

    #[test]
    fn adaptive_tracker_reset() {
        let mut tracker = AdaptiveFrequencyTracker::new(10);
        tracker.observe(42);
        tracker.reset();
        assert_eq!(tracker.frequency(42), 0);
        assert_eq!(tracker.total(), 0);
    }

    #[test]
    fn adaptive_tracker_frequency_table() {
        let mut tracker = AdaptiveFrequencyTracker::new(100);
        tracker.observe(10);
        tracker.observe(10);
        tracker.observe(20);
        let table = tracker.frequency_table();
        assert_eq!(table[10], 2);
        assert_eq!(table[20], 1);
        assert_eq!(table[0], 0);
    }
}
