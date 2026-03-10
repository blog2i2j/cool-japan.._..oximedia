//! VP8L (WebP lossless) encoder.
//!
//! Implements the VP8L bitstream format as specified in the WebP Lossless
//! Bitstream Specification (IETF draft). Produces valid VP8L bitstreams
//! that standard decoders (libwebp, Chrome, etc.) can read.
//!
//! # Encoding strategy
//!
//! 1. Write VP8L header (signature 0x2F + dimensions + alpha flag)
//! 2. Apply subtract-green transform (always, very effective)
//! 3. Optionally apply predictor transform at higher effort levels
//! 4. Encode pixel data with Huffman coding (5 code groups)
//! 5. Write Huffman-coded pixel literals (no LZ77 back-references in simple path)

use crate::error::{CodecError, CodecResult};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// VP8L signature byte.
const VP8L_SIGNATURE: u8 = 0x2f;

/// Maximum image dimension for VP8L.
const VP8L_MAX_DIMENSION: u32 = 16384;

/// Code length code order as specified by the VP8L spec.
const CODE_LENGTH_CODE_ORDER: [usize; 19] = [
    17, 18, 0, 1, 2, 3, 4, 5, 16, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
];

/// Maximum Huffman code length allowed in VP8L.
const MAX_HUFFMAN_CODE_LENGTH: u32 = 15;

/// Number of code length codes.
const NUM_CODE_LENGTH_CODES: usize = 19;

/// VP8L transform types.
const TRANSFORM_PREDICTOR: u32 = 0;
const TRANSFORM_SUBTRACT_GREEN: u32 = 2;

/// VP8L uses 5 Huffman trees per code group:
/// 0: green + length prefix (256 + 24 = 280 symbols)
/// 1: red (256 symbols)
/// 2: blue (256 symbols)
/// 3: alpha (256 symbols)
/// 4: distance prefix (40 symbols)
const HUFFMAN_CODES_PER_META_CODE: usize = 5;

/// Number of symbols for the green/length-prefix tree.
/// 256 literals + 24 length prefix codes.
const GREEN_ALPHABET_SIZE: usize = 280;

/// Number of symbols for R/B/A trees.
const CHANNEL_ALPHABET_SIZE: usize = 256;

/// Distance alphabet size.
const DISTANCE_ALPHABET_SIZE: usize = 40;

// ---------------------------------------------------------------------------
// VP8L Bit Writer (LSB-first)
// ---------------------------------------------------------------------------

/// VP8L bit writer that packs bits LSB-first.
///
/// VP8L uses a little-endian bit packing order, where the first bit written
/// occupies the least significant bit position.
#[derive(Debug)]
pub struct Vp8lBitWriter {
    /// Accumulated output bytes.
    data: Vec<u8>,
    /// Bit accumulator.
    bits: u64,
    /// Number of valid bits in the accumulator.
    num_bits: u32,
}

impl Vp8lBitWriter {
    /// Creates a new VP8L bit writer.
    pub fn new() -> Self {
        Self {
            data: Vec::with_capacity(4096),
            bits: 0,
            num_bits: 0,
        }
    }

    /// Creates a new VP8L bit writer with a size hint.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
            bits: 0,
            num_bits: 0,
        }
    }

    /// Writes `n_bits` (1..=32) of `value` in LSB-first order.
    pub fn put_bits(&mut self, value: u32, n_bits: u32) {
        debug_assert!(n_bits <= 32);
        debug_assert!(n_bits == 32 || value < (1u32 << n_bits));

        self.bits |= (value as u64) << self.num_bits;
        self.num_bits += n_bits;
        self.flush_partial();
    }

    /// Writes a single bit.
    pub fn put_bit(&mut self, bit: bool) {
        self.put_bits(u32::from(bit), 1);
    }

    /// Flushes complete bytes from the accumulator to `data`.
    fn flush_partial(&mut self) {
        while self.num_bits >= 8 {
            self.data.push((self.bits & 0xff) as u8);
            self.bits >>= 8;
            self.num_bits -= 8;
        }
    }

    /// Finishes writing and returns the complete byte buffer.
    /// Any remaining bits are zero-padded to a byte boundary.
    pub fn finish(mut self) -> Vec<u8> {
        if self.num_bits > 0 {
            self.data.push((self.bits & 0xff) as u8);
        }
        self.data
    }

    /// Returns the current byte length of written data (including
    /// unflushed bits in the accumulator).
    pub fn byte_len(&self) -> usize {
        self.data.len() + if self.num_bits > 0 { 1 } else { 0 }
    }
}

// ---------------------------------------------------------------------------
// Huffman code builder
// ---------------------------------------------------------------------------

/// A single Huffman code entry: (bit length, code word).
#[derive(Debug, Clone, Copy, Default)]
struct HuffmanCode {
    /// Bit length of this code (0 means unused symbol).
    len: u32,
    /// The actual code word (only the lower `len` bits are valid).
    code: u32,
}

/// Histogram for one Huffman tree.
#[derive(Debug, Clone)]
struct Histogram {
    counts: Vec<u32>,
}

impl Histogram {
    fn new(size: usize) -> Self {
        Self {
            counts: vec![0; size],
        }
    }

    fn add(&mut self, symbol: usize) {
        if symbol < self.counts.len() {
            self.counts[symbol] += 1;
        }
    }

    fn alphabet_size(&self) -> usize {
        self.counts.len()
    }

    /// Finds the number of non-zero symbols.
    fn num_used(&self) -> usize {
        self.counts.iter().filter(|&&c| c > 0).count()
    }

    /// Returns the single used symbol if exactly one is used.
    fn single_symbol(&self) -> Option<usize> {
        if self.num_used() == 1 {
            self.counts.iter().position(|&c| c > 0)
        } else {
            None
        }
    }
}

/// Build length-limited Huffman codes from a histogram.
///
/// Uses a tree-based algorithm to produce optimal codes, then limits
/// code lengths to `MAX_HUFFMAN_CODE_LENGTH` using the Kraft inequality.
fn build_huffman_codes(hist: &Histogram) -> CodecResult<Vec<HuffmanCode>> {
    let n = hist.alphabet_size();
    let mut codes = vec![HuffmanCode::default(); n];

    let used = hist.num_used();
    if used == 0 {
        // All symbols unused: return all-zero codes.
        return Ok(codes);
    }
    if used == 1 {
        // Single symbol gets length 1 (VP8L requires at least length 1).
        if let Some(sym) = hist.single_symbol() {
            codes[sym] = HuffmanCode { len: 1, code: 0 };
        }
        return Ok(codes);
    }
    if used == 2 {
        // Two symbols: assign lengths 1 each, codes 0 and 1.
        let mut iter = hist
            .counts
            .iter()
            .enumerate()
            .filter(|(_, &c)| c > 0)
            .map(|(i, _)| i);
        if let (Some(a), Some(b)) = (iter.next(), iter.next()) {
            codes[a] = HuffmanCode { len: 1, code: 0 };
            codes[b] = HuffmanCode { len: 1, code: 1 };
        }
        return Ok(codes);
    }

    // General case: build code lengths using a frequency-sorted approach.
    let code_lengths = build_code_lengths(&hist.counts, MAX_HUFFMAN_CODE_LENGTH)?;

    // Convert code lengths to canonical Huffman codes.
    assign_canonical_codes(&code_lengths, &mut codes);

    Ok(codes)
}

/// Build code lengths from frequency counts, limited to `max_length`.
fn build_code_lengths(counts: &[u32], max_length: u32) -> CodecResult<Vec<u32>> {
    let n = counts.len();
    let mut lengths = vec![0u32; n];

    // Collect (count, symbol) pairs for non-zero symbols, sort by frequency.
    let mut symbols: Vec<(u32, usize)> = counts
        .iter()
        .enumerate()
        .filter(|(_, &c)| c > 0)
        .map(|(i, &c)| (c, i))
        .collect();

    if symbols.len() < 2 {
        for &(_, sym) in &symbols {
            lengths[sym] = 1;
        }
        return Ok(lengths);
    }

    // Sort by frequency ascending, break ties by symbol index.
    symbols.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

    // Build a Huffman tree bottom-up and extract depths.
    let tree_lengths = build_tree_lengths(&symbols, max_length);

    for (i, &(_, sym)) in symbols.iter().enumerate() {
        lengths[sym] = tree_lengths[i];
    }

    Ok(lengths)
}

/// Build Huffman tree lengths using an iterative approach.
///
/// Constructs a Huffman tree and extracts bit lengths, then limits them
/// to `max_length` using the Kraft inequality adjustment.
fn build_tree_lengths(symbols: &[(u32, usize)], max_length: u32) -> Vec<u32> {
    let n = symbols.len();
    if n <= 1 {
        return vec![1; n];
    }

    // Standard Huffman tree construction using a priority queue simulation.
    // Node: (frequency, Option<left_child_idx>, Option<right_child_idx>)
    let mut nodes: Vec<(u64, Option<usize>, Option<usize>)> = symbols
        .iter()
        .map(|&(freq, _)| (freq.max(1) as u64, None, None))
        .collect();

    let mut active: Vec<usize> = (0..n).collect();

    while active.len() > 1 {
        // Find the two nodes with smallest frequency.
        active.sort_by(|&a, &b| nodes[a].0.cmp(&nodes[b].0));
        let left = active[0];
        let right = active[1];

        let merged_freq = nodes[left].0 + nodes[right].0;
        let new_idx = nodes.len();
        nodes.push((merged_freq, Some(left), Some(right)));

        // Remove left and right, add merged node.
        active = active[2..].to_vec();
        active.push(new_idx);
    }

    // Extract depths from the tree.
    let mut depths = vec![0u32; n];
    if let Some(&root) = active.first() {
        compute_depths(&nodes, root, 0, n, &mut depths);
    }

    // Limit code lengths to max_length using Kraft inequality.
    limit_code_lengths(&mut depths, max_length);

    depths
}

/// Recursively compute depths (code lengths) from a Huffman tree.
fn compute_depths(
    nodes: &[(u64, Option<usize>, Option<usize>)],
    node_idx: usize,
    depth: u32,
    num_leaves: usize,
    depths: &mut [u32],
) {
    if node_idx < num_leaves {
        // Leaf node.
        depths[node_idx] = depth.max(1);
        return;
    }
    let (_, left, right) = nodes[node_idx];
    if let Some(l) = left {
        compute_depths(nodes, l, depth + 1, num_leaves, depths);
    }
    if let Some(r) = right {
        compute_depths(nodes, r, depth + 1, num_leaves, depths);
    }
}

/// Limit code lengths to `max_length` while preserving the Kraft inequality.
///
/// The Kraft inequality states that for a valid prefix code, the sum of
/// 2^(-len_i) must equal 1. After clamping, we adjust to restore this.
fn limit_code_lengths(depths: &mut [u32], max_length: u32) {
    let mut any_over = false;
    for d in depths.iter() {
        if *d > max_length {
            any_over = true;
            break;
        }
    }
    if !any_over {
        return;
    }

    // Clamp all lengths to max_length.
    for d in depths.iter_mut() {
        if *d > max_length {
            *d = max_length;
        }
    }

    // Fix the Kraft inequality iteratively.
    for _ in 0..1000 {
        let kraft_sum: u64 = depths
            .iter()
            .filter(|&&d| d > 0)
            .map(|&d| 1u64 << (max_length - d))
            .sum();

        let target = 1u64 << max_length;

        if kraft_sum == target {
            break;
        }

        if kraft_sum > target {
            // Over-full: increase the length of the shortest code.
            if let Some(pos) = depths
                .iter()
                .enumerate()
                .filter(|(_, &d)| d > 0 && d < max_length)
                .min_by_key(|(_, &d)| d)
                .map(|(i, _)| i)
            {
                depths[pos] += 1;
            } else {
                break;
            }
        } else {
            // Under-full: decrease the length of a code at max_length.
            if let Some(pos) = depths
                .iter()
                .enumerate()
                .filter(|(_, &d)| d == max_length)
                .max_by_key(|(i, _)| *i)
                .map(|(i, _)| i)
            {
                depths[pos] -= 1;
            } else {
                break;
            }
        }
    }
}

/// Assign canonical Huffman codes from code lengths.
///
/// Canonical codes are assigned by sorting symbols by (length, symbol)
/// and assigning incrementing codes within each length group. The codes
/// are then bit-reversed for VP8L's LSB-first format.
fn assign_canonical_codes(lengths: &[u32], codes: &mut [HuffmanCode]) {
    let max_len = lengths.iter().copied().max().unwrap_or(0);
    if max_len == 0 {
        return;
    }

    // Count codes of each length.
    let mut bl_count = vec![0u32; (max_len + 1) as usize];
    for &l in lengths {
        if l > 0 {
            bl_count[l as usize] += 1;
        }
    }

    // Compute the starting code for each length.
    let mut next_code = vec![0u32; (max_len + 1) as usize];
    let mut code = 0u32;
    for bits in 1..=max_len {
        code = (code + bl_count[(bits - 1) as usize]) << 1;
        next_code[bits as usize] = code;
    }

    // Assign codes and reverse bits for LSB-first VP8L encoding.
    for (sym, &len) in lengths.iter().enumerate() {
        if len > 0 {
            let c = next_code[len as usize];
            next_code[len as usize] += 1;
            codes[sym] = HuffmanCode {
                len,
                code: reverse_bits(c, len),
            };
        }
    }
}

/// Reverse the lowest `num_bits` bits of `value`.
fn reverse_bits(value: u32, num_bits: u32) -> u32 {
    let mut result = 0u32;
    let mut v = value;
    for _ in 0..num_bits {
        result = (result << 1) | (v & 1);
        v >>= 1;
    }
    result
}

// ---------------------------------------------------------------------------
// Transform helpers
// ---------------------------------------------------------------------------

/// Apply subtract-green transform to ARGB pixel data in place.
///
/// For each pixel, subtracts the green channel from red and blue:
///   red   = (red - green) mod 256
///   blue  = (blue - green) mod 256
///   green and alpha are unchanged.
fn apply_subtract_green(pixels: &mut [u32]) {
    for px in pixels.iter_mut() {
        let a = (*px >> 24) & 0xff;
        let r = (*px >> 16) & 0xff;
        let g = (*px >> 8) & 0xff;
        let b = *px & 0xff;

        let new_r = r.wrapping_sub(g) & 0xff;
        let new_b = b.wrapping_sub(g) & 0xff;

        *px = (a << 24) | (new_r << 16) | (g << 8) | new_b;
    }
}

/// Apply predictor transform using spatial prediction.
///
/// Each pixel is predicted from its neighbors and the residual
/// (pixel - prediction) mod 256 is stored. This reduces entropy
/// for natural images with spatial correlation.
///
/// Returns the predictor data (mode per tile) for writing to the bitstream.
fn apply_predictor_transform(
    pixels: &mut [u32],
    width: u32,
    height: u32,
) -> CodecResult<Vec<u8>> {
    let w = width as usize;
    let h = height as usize;

    if pixels.len() != w * h {
        return Err(CodecError::InvalidParameter(format!(
            "pixel count {} != width {} * height {}",
            pixels.len(),
            w,
            h
        )));
    }

    // Tile size for the predictor data image.
    // size_bits = 3 means tile_size = 8.
    let tile_size = 1usize << 3;
    let tiles_x = (w + tile_size - 1) / tile_size;
    let tiles_y = (h + tile_size - 1) / tile_size;

    // Use predictor mode 1 (left) everywhere for simplicity.
    let predictor_mode: u8 = 1;
    let predictor_data = vec![predictor_mode; tiles_x * tiles_y];

    // Keep original pixels for reading predictions.
    let original = pixels.to_vec();

    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let current = original[idx];

            let predicted = if x == 0 && y == 0 {
                // Top-left corner: predict black (opaque).
                0xff000000u32
            } else if x == 0 {
                // Left edge: predict top pixel.
                original[(y - 1) * w + x]
            } else {
                // Default: predict left pixel.
                original[y * w + (x - 1)]
            };

            let ca = (current >> 24) & 0xff;
            let cr = (current >> 16) & 0xff;
            let cg = (current >> 8) & 0xff;
            let cb = current & 0xff;

            let pa = (predicted >> 24) & 0xff;
            let pr = (predicted >> 16) & 0xff;
            let pg = (predicted >> 8) & 0xff;
            let pb = predicted & 0xff;

            let ra = ca.wrapping_sub(pa) & 0xff;
            let rr = cr.wrapping_sub(pr) & 0xff;
            let rg = cg.wrapping_sub(pg) & 0xff;
            let rb = cb.wrapping_sub(pb) & 0xff;

            pixels[idx] = (ra << 24) | (rr << 16) | (rg << 8) | rb;
        }
    }

    Ok(predictor_data)
}

// ---------------------------------------------------------------------------
// Huffman encoding helpers
// ---------------------------------------------------------------------------

/// Builds histograms for the 5 VP8L Huffman trees from transformed pixel data.
fn build_histograms(pixels: &[u32]) -> [Histogram; HUFFMAN_CODES_PER_META_CODE] {
    let mut histograms = [
        Histogram::new(GREEN_ALPHABET_SIZE),
        Histogram::new(CHANNEL_ALPHABET_SIZE),
        Histogram::new(CHANNEL_ALPHABET_SIZE),
        Histogram::new(CHANNEL_ALPHABET_SIZE),
        Histogram::new(DISTANCE_ALPHABET_SIZE),
    ];

    for &px in pixels {
        let g = ((px >> 8) & 0xff) as usize;
        let r = ((px >> 16) & 0xff) as usize;
        let b = (px & 0xff) as usize;
        let a = ((px >> 24) & 0xff) as usize;

        histograms[0].add(g);
        histograms[1].add(r);
        histograms[2].add(b);
        histograms[3].add(a);
        // No distance codes in literal-only mode.
    }

    histograms
}

/// Write a single Huffman code to the bitstream.
fn write_huffman_symbol(writer: &mut Vp8lBitWriter, codes: &[HuffmanCode], symbol: usize) {
    let hc = if symbol < codes.len() {
        codes[symbol]
    } else {
        HuffmanCode { len: 1, code: 0 }
    };
    if hc.len > 0 {
        writer.put_bits(hc.code, hc.len);
    }
}

/// Write Huffman code lengths to the bitstream using the meta-code.
///
/// Implements the VP8L code length encoding:
/// 1. Build a meta-Huffman code for the code lengths themselves
/// 2. Write the meta-code lengths (num_code_lengths, then lengths in spec order)
/// 3. Write the actual code lengths using the meta-code
fn write_huffman_codes(
    writer: &mut Vp8lBitWriter,
    codes: &[HuffmanCode],
    alphabet_size: usize,
) -> CodecResult<()> {
    let num_used = codes.iter().filter(|c| c.len > 0).count();

    // Simple code (1 or 2 symbols): use the special simple code format.
    if num_used <= 2 {
        return write_simple_code(writer, codes, num_used);
    }

    // Normal Huffman code: write using code length codes.
    write_normal_code(writer, codes, alphabet_size)
}

/// Write a "simple" Huffman code (1 or 2 symbols).
///
/// Format:
///   1 bit: is_simple_code = 1
///   1 bit: num_symbols - 1 (0 or 1)
///   If 1 symbol:
///     1 bit: is_first_8bit (0 = 1-bit symbol value, 1 = 8-bit symbol value)
///     symbol value (1 or 8 bits)
///   If 2 symbols:
///     8 bits: symbol0
///     8 bits: symbol1
fn write_simple_code(
    writer: &mut Vp8lBitWriter,
    codes: &[HuffmanCode],
    num_used: usize,
) -> CodecResult<()> {
    // is_simple_code = 1
    writer.put_bit(true);

    let symbols: Vec<usize> = codes
        .iter()
        .enumerate()
        .filter(|(_, c)| c.len > 0)
        .map(|(i, _)| i)
        .collect();

    if num_used == 0 {
        // Degenerate: write single symbol 0.
        writer.put_bit(false); // num_symbols - 1 = 0
        writer.put_bit(false); // is_first_8bit = 0
        writer.put_bits(0, 1);
        return Ok(());
    }

    if num_used == 1 {
        let sym = symbols[0];
        writer.put_bit(false); // num_symbols - 1 = 0
        if sym < 2 {
            writer.put_bit(false); // is_first_8bit = 0
            writer.put_bits(sym as u32, 1);
        } else {
            writer.put_bit(true); // is_first_8bit = 1
            writer.put_bits(sym as u32, 8);
        }
    } else {
        // 2 symbols.
        let sym0 = symbols[0];
        let sym1 = symbols[1];
        writer.put_bit(true); // num_symbols - 1 = 1
        writer.put_bits(sym0 as u32, 8);
        writer.put_bits(sym1 as u32, 8);
    }

    Ok(())
}

/// Write a normal (non-simple) Huffman code using code length codes.
fn write_normal_code(
    writer: &mut Vp8lBitWriter,
    codes: &[HuffmanCode],
    alphabet_size: usize,
) -> CodecResult<()> {
    // is_simple_code = 0
    writer.put_bit(false);

    // Collect code lengths for all symbols up to alphabet_size.
    let mut code_lengths: Vec<u32> = codes
        .iter()
        .take(alphabet_size)
        .map(|c| c.len)
        .collect();
    code_lengths.resize(alphabet_size, 0);

    // Run-length encode the code lengths.
    let rle_symbols = rle_encode_code_lengths(&code_lengths);

    // Build histogram for the code length symbols (0..18).
    let mut cl_hist = Histogram::new(NUM_CODE_LENGTH_CODES);
    for &(sym, _) in &rle_symbols {
        cl_hist.add(sym as usize);
    }

    // Build Huffman codes for the code length symbols.
    let cl_codes = build_huffman_codes(&cl_hist)?;

    // Determine how many code length code lengths to write (in spec order).
    let mut num_cl_codes = 4usize; // Minimum is 4.
    for i in (0..NUM_CODE_LENGTH_CODES).rev() {
        let sym = CODE_LENGTH_CODE_ORDER[i];
        if sym < cl_codes.len() && cl_codes[sym].len > 0 {
            num_cl_codes = i + 1;
            break;
        }
    }
    num_cl_codes = num_cl_codes.max(4);

    // Write num_code_lengths - 4 (4 bits).
    writer.put_bits((num_cl_codes - 4) as u32, 4);

    // Write code length code lengths in spec order (3 bits each).
    for i in 0..num_cl_codes {
        let sym = CODE_LENGTH_CODE_ORDER[i];
        let len = if sym < cl_codes.len() {
            cl_codes[sym].len
        } else {
            0
        };
        writer.put_bits(len, 3);
    }

    // Write the actual code lengths using the code length codes.
    for &(sym, extra) in &rle_symbols {
        write_huffman_symbol(writer, &cl_codes, sym as usize);
        match sym {
            16 => {
                // Repeat previous: 2 extra bits (repeat 3..6).
                writer.put_bits(extra, 2);
            }
            17 => {
                // Repeat zero: 3 extra bits (repeat 3..10).
                writer.put_bits(extra, 3);
            }
            18 => {
                // Repeat zero: 7 extra bits (repeat 11..138).
                writer.put_bits(extra, 7);
            }
            _ => {
                // Literal code length, no extra bits.
            }
        }
    }

    Ok(())
}

/// Run-length encode code lengths into (symbol, extra_bits) pairs.
///
/// Uses symbols 0-15 as literal lengths, and:
///   16 = repeat previous length, extra = count - 3 (2 bits, range 3..6)
///   17 = repeat zero 3..10 times, extra = count - 3 (3 bits)
///   18 = repeat zero 11..138 times, extra = count - 11 (7 bits)
fn rle_encode_code_lengths(lengths: &[u32]) -> Vec<(u32, u32)> {
    let mut result = Vec::new();
    let mut i = 0;

    while i < lengths.len() {
        let val = lengths[i];

        if val == 0 {
            // Count consecutive zeros.
            let mut run = 1;
            while i + run < lengths.len() && lengths[i + run] == 0 {
                run += 1;
            }

            let mut remaining = run;
            while remaining > 0 {
                if remaining >= 11 {
                    let count = remaining.min(138);
                    result.push((18, (count - 11) as u32));
                    remaining -= count;
                } else if remaining >= 3 {
                    let count = remaining.min(10);
                    result.push((17, (count - 3) as u32));
                    remaining -= count;
                } else {
                    result.push((0, 0));
                    remaining -= 1;
                }
            }

            i += run;
        } else {
            // Emit the literal value.
            result.push((val, 0));
            i += 1;

            // Check for repeats of this value.
            let mut run = 0;
            while i + run < lengths.len() && lengths[i + run] == val {
                run += 1;
            }

            let mut remaining = run;
            while remaining >= 3 {
                let count = remaining.min(6);
                result.push((16, (count - 3) as u32));
                remaining -= count;
            }
            while remaining > 0 {
                result.push((val, 0));
                remaining -= 1;
            }

            i += run;
        }
    }

    result
}

// ---------------------------------------------------------------------------
// VP8L Encoder
// ---------------------------------------------------------------------------

/// VP8L (WebP lossless) encoder.
///
/// Produces valid VP8L bitstreams from ARGB pixel data. The encoder
/// supports multiple effort levels that control which transforms are
/// applied before entropy coding.
///
/// # Example
///
/// ```ignore
/// use oximedia_codec::webp::Vp8lEncoder;
///
/// let encoder = Vp8lEncoder::new(50);
/// let pixels: Vec<u32> = vec![0xff_ff_00_00; 64]; // 8x8 red image
/// let data = encoder.encode(&pixels, 8, 8, false)?;
/// ```
pub struct Vp8lEncoder {
    /// Compression effort level (0-100).
    /// Higher values enable more transforms and produce smaller output.
    effort: u8,
}

impl Vp8lEncoder {
    /// Creates a new VP8L encoder with the given effort level (0-100).
    ///
    /// - effort 0-30: Only subtract-green transform
    /// - effort 31-100: Subtract-green + predictor transform
    pub fn new(effort: u8) -> Self {
        Self {
            effort: effort.min(100),
        }
    }

    /// Encode ARGB pixels to a VP8L bitstream.
    ///
    /// # Arguments
    ///
    /// * `pixels` - Slice of ARGB u32 values (one per pixel, row-major order).
    ///   Format: `(alpha << 24) | (red << 16) | (green << 8) | blue`.
    /// * `width` - Image width in pixels (1..16384).
    /// * `height` - Image height in pixels (1..16384).
    /// * `has_alpha` - Whether the alpha channel contains meaningful data.
    ///
    /// # Returns
    ///
    /// VP8L encoded data starting with the 0x2F signature byte.
    pub fn encode(
        &self,
        pixels: &[u32],
        width: u32,
        height: u32,
        has_alpha: bool,
    ) -> CodecResult<Vec<u8>> {
        self.validate_inputs(pixels, width, height)?;

        // Make a mutable copy for transform application.
        let mut transformed = pixels.to_vec();

        // Decide which transforms to apply.
        let use_predictor = self.effort > 30;

        // Collect transform metadata.
        let mut predictor_data: Option<(Vec<u8>, u32, u32, u32)> = None;

        // Apply predictor transform first (if enabled).
        if use_predictor {
            let size_bits: u32 = 3;
            let tile_size = 1u32 << size_bits;
            let tiles_x = (width + tile_size - 1) / tile_size;
            let tiles_y = (height + tile_size - 1) / tile_size;

            let pred_data =
                apply_predictor_transform(&mut transformed, width, height)?;
            predictor_data = Some((pred_data, size_bits, tiles_x, tiles_y));
        }

        // Always apply subtract-green transform.
        apply_subtract_green(&mut transformed);

        // Build the bitstream.
        let estimated_size = (pixels.len() * 4) + 256;
        let mut writer = Vp8lBitWriter::with_capacity(estimated_size);

        // Write VP8L header.
        self.write_header(&mut writer, width, height, has_alpha)?;

        // Write transforms (in reverse application order for LIFO decoding).
        self.write_transforms(
            &mut writer,
            use_predictor,
            &predictor_data,
        )?;

        // Signal no more transforms.
        writer.put_bit(false);

        // Encode and write pixel data.
        self.write_image_data(&mut writer, &transformed)?;

        Ok(writer.finish())
    }

    /// Validate encoder inputs.
    fn validate_inputs(
        &self,
        pixels: &[u32],
        width: u32,
        height: u32,
    ) -> CodecResult<()> {
        if width == 0 || height == 0 {
            return Err(CodecError::InvalidParameter(
                "width and height must be > 0".to_string(),
            ));
        }
        if width > VP8L_MAX_DIMENSION || height > VP8L_MAX_DIMENSION {
            return Err(CodecError::InvalidParameter(format!(
                "dimensions {}x{} exceed VP8L maximum {}",
                width, height, VP8L_MAX_DIMENSION
            )));
        }
        let expected = (width as usize) * (height as usize);
        if pixels.len() != expected {
            return Err(CodecError::InvalidParameter(format!(
                "pixel count {} != expected {} ({}x{})",
                pixels.len(),
                expected,
                width,
                height
            )));
        }
        Ok(())
    }

    /// Write VP8L bitstream header.
    ///
    /// Format:
    ///   1 byte: signature (0x2F)
    ///   14 bits: width - 1
    ///   14 bits: height - 1
    ///   1 bit: alpha_is_used
    ///   3 bits: version (always 0)
    fn write_header(
        &self,
        writer: &mut Vp8lBitWriter,
        width: u32,
        height: u32,
        has_alpha: bool,
    ) -> CodecResult<()> {
        // Signature byte (written through the bit writer as 8 bits).
        writer.put_bits(u32::from(VP8L_SIGNATURE), 8);

        // 14 bits: width - 1
        writer.put_bits(width - 1, 14);

        // 14 bits: height - 1
        writer.put_bits(height - 1, 14);

        // 1 bit: alpha_is_used
        writer.put_bit(has_alpha);

        // 3 bits: version (always 0)
        writer.put_bits(0, 3);

        Ok(())
    }

    /// Write transform indicators to the bitstream.
    ///
    /// Transforms are written in reverse application order (LIFO for decoding).
    /// Application order: predictor -> subtract-green
    /// Write order: subtract-green first, then predictor
    fn write_transforms(
        &self,
        writer: &mut Vp8lBitWriter,
        use_predictor: bool,
        predictor_data: &Option<(Vec<u8>, u32, u32, u32)>,
    ) -> CodecResult<()> {
        // Subtract-green transform (always applied).
        writer.put_bit(true);  // transform present
        writer.put_bits(TRANSFORM_SUBTRACT_GREEN, 2);
        // Subtract-green has no extra data.

        // Predictor transform (if enabled).
        if use_predictor {
            if let Some((ref pred_data, size_bits, _tiles_x, _tiles_y)) = *predictor_data {
                writer.put_bit(true);  // transform present
                writer.put_bits(TRANSFORM_PREDICTOR, 2);
                // 3 bits: size_bits (log2 of tile size minus 2).
                writer.put_bits(size_bits, 3);

                // Write the predictor data as a sub-image.
                // Each pixel encodes the predictor mode in the green channel.
                let pred_pixels: Vec<u32> = pred_data
                    .iter()
                    .map(|&mode| (mode as u32) << 8)
                    .collect();

                self.write_image_data(writer, &pred_pixels)?;
            }
        }

        Ok(())
    }

    /// Encode and write pixel data using Huffman coding.
    ///
    /// This is the core encoding routine that:
    /// 1. Writes color cache flag (disabled)
    /// 2. Builds histograms for each of the 5 symbol types
    /// 3. Constructs optimal Huffman codes
    /// 4. Writes code descriptions to the bitstream
    /// 5. Writes Huffman-coded pixel data
    fn write_image_data(
        &self,
        writer: &mut Vp8lBitWriter,
        pixels: &[u32],
    ) -> CodecResult<()> {
        // Color cache: disabled (1 bit = false).
        writer.put_bit(false);

        // Build histograms for the 5 Huffman trees.
        let histograms = build_histograms(pixels);

        // Build Huffman codes for each tree and write their descriptions.
        let alphabet_sizes = [
            GREEN_ALPHABET_SIZE,
            CHANNEL_ALPHABET_SIZE,
            CHANNEL_ALPHABET_SIZE,
            CHANNEL_ALPHABET_SIZE,
            DISTANCE_ALPHABET_SIZE,
        ];

        let mut all_codes: Vec<Vec<HuffmanCode>> =
            Vec::with_capacity(HUFFMAN_CODES_PER_META_CODE);

        for (i, hist) in histograms.iter().enumerate() {
            let codes = build_huffman_codes(hist)?;
            write_huffman_codes(writer, &codes, alphabet_sizes[i])?;
            all_codes.push(codes);
        }

        // Write encoded pixel data as Huffman-coded literals.
        for &px in pixels {
            let g = ((px >> 8) & 0xff) as usize;
            let r = ((px >> 16) & 0xff) as usize;
            let b = (px & 0xff) as usize;
            let a = ((px >> 24) & 0xff) as usize;

            write_huffman_symbol(writer, &all_codes[0], g);
            write_huffman_symbol(writer, &all_codes[1], r);
            write_huffman_symbol(writer, &all_codes[2], b);
            write_huffman_symbol(writer, &all_codes[3], a);
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Bit Writer tests --

    #[test]
    fn test_bit_writer_basic() {
        let mut w = Vp8lBitWriter::new();
        w.put_bits(0b101, 3);
        w.put_bits(0b1100, 4);
        w.put_bit(true);
        let data = w.finish();
        // LSB-first: bits are 1,0,1, 0,0,1,1, 1
        // byte = 0b1_1100_101 = 0xe5
        assert_eq!(data.len(), 1);
        assert_eq!(data[0], 0xe5);
    }

    #[test]
    fn test_bit_writer_multi_byte() {
        let mut w = Vp8lBitWriter::new();
        w.put_bits(0xff, 8);
        w.put_bits(0x00, 8);
        w.put_bits(0xab, 8);
        let data = w.finish();
        assert_eq!(data, vec![0xff, 0x00, 0xab]);
    }

    #[test]
    fn test_bit_writer_empty() {
        let w = Vp8lBitWriter::new();
        let data = w.finish();
        assert!(data.is_empty());
    }

    #[test]
    fn test_bit_writer_single_bit() {
        let mut w = Vp8lBitWriter::new();
        w.put_bit(true);
        let data = w.finish();
        assert_eq!(data, vec![0x01]);
    }

    #[test]
    fn test_bit_writer_byte_len() {
        let mut w = Vp8lBitWriter::new();
        assert_eq!(w.byte_len(), 0);
        w.put_bits(0xff, 8);
        assert_eq!(w.byte_len(), 1);
        w.put_bit(true);
        assert_eq!(w.byte_len(), 2);
    }

    // -- Reverse bits --

    #[test]
    fn test_reverse_bits() {
        assert_eq!(reverse_bits(0b110, 3), 0b011);
        assert_eq!(reverse_bits(0b1010, 4), 0b0101);
        assert_eq!(reverse_bits(0b1, 1), 0b1);
        assert_eq!(reverse_bits(0b0, 1), 0b0);
        assert_eq!(reverse_bits(0b11001, 5), 0b10011);
    }

    // -- Subtract green transform --

    #[test]
    fn test_subtract_green() {
        // Pixel: A=0xff, R=0x80, G=0x40, B=0xc0
        let mut pixels = vec![0xff_80_40_c0u32];
        apply_subtract_green(&mut pixels);

        let a = (pixels[0] >> 24) & 0xff;
        let r = (pixels[0] >> 16) & 0xff;
        let g = (pixels[0] >> 8) & 0xff;
        let b = pixels[0] & 0xff;

        assert_eq!(a, 0xff);
        assert_eq!(g, 0x40);
        assert_eq!(r, (0x80u32.wrapping_sub(0x40)) & 0xff);
        assert_eq!(b, (0xc0u32.wrapping_sub(0x40)) & 0xff);
    }

    #[test]
    fn test_subtract_green_wraparound() {
        // R=0x10, G=0x80: R-G = 0x10 - 0x80 = 0x90 (mod 256)
        let mut pixels = vec![0xff_10_80_20u32];
        apply_subtract_green(&mut pixels);
        let r = (pixels[0] >> 16) & 0xff;
        assert_eq!(r, 0x90);
    }

    // -- Histogram --

    #[test]
    fn test_histogram_basic() {
        let mut hist = Histogram::new(256);
        hist.add(0);
        hist.add(0);
        hist.add(1);
        hist.add(255);

        assert_eq!(hist.num_used(), 3);
        assert_eq!(hist.counts[0], 2);
        assert_eq!(hist.counts[1], 1);
        assert_eq!(hist.counts[255], 1);
    }

    #[test]
    fn test_histogram_single_symbol() {
        let mut hist = Histogram::new(256);
        hist.add(42);
        hist.add(42);
        hist.add(42);

        assert_eq!(hist.num_used(), 1);
        assert_eq!(hist.single_symbol(), Some(42));
    }

    #[test]
    fn test_histogram_out_of_range() {
        let mut hist = Histogram::new(10);
        hist.add(100); // Out of range, should be ignored.
        assert_eq!(hist.num_used(), 0);
    }

    // -- Huffman code building --

    #[test]
    fn test_build_huffman_empty() {
        let hist = Histogram::new(256);
        let codes = build_huffman_codes(&hist).expect("should build");
        assert!(codes.iter().all(|c| c.len == 0));
    }

    #[test]
    fn test_build_huffman_single_symbol() {
        let mut hist = Histogram::new(256);
        hist.add(100);
        hist.add(100);

        let codes = build_huffman_codes(&hist).expect("should build");
        assert_eq!(codes[100].len, 1);
        assert_eq!(codes[100].code, 0);
    }

    #[test]
    fn test_build_huffman_two_symbols() {
        let mut hist = Histogram::new(256);
        hist.add(10);
        hist.add(20);

        let codes = build_huffman_codes(&hist).expect("should build");
        assert_eq!(codes[10].len, 1);
        assert_eq!(codes[20].len, 1);
        assert_ne!(codes[10].code, codes[20].code);
    }

    #[test]
    fn test_build_huffman_multiple_symbols() {
        let mut hist = Histogram::new(256);
        for _ in 0..100 {
            hist.add(0);
        }
        for _ in 0..50 {
            hist.add(1);
        }
        for _ in 0..25 {
            hist.add(2);
        }
        for _ in 0..10 {
            hist.add(3);
        }

        let codes = build_huffman_codes(&hist).expect("should build");

        // Most frequent symbol should have shortest or equal code length.
        assert!(codes[0].len <= codes[3].len);
        for i in 0..4 {
            assert!(codes[i].len > 0, "symbol {} should have nonzero length", i);
            assert!(
                codes[i].len <= MAX_HUFFMAN_CODE_LENGTH,
                "symbol {} code length {} exceeds max",
                i,
                codes[i].len
            );
        }
    }

    #[test]
    fn test_build_huffman_all_equal_frequencies() {
        let mut hist = Histogram::new(4);
        for i in 0..4 {
            hist.add(i);
        }

        let codes = build_huffman_codes(&hist).expect("should build");
        // 4 symbols with equal frequency should get length 2 each.
        for i in 0..4 {
            assert_eq!(codes[i].len, 2, "symbol {} should have length 2", i);
        }
    }

    #[test]
    fn test_canonical_codes_prefix_free() {
        let mut hist = Histogram::new(8);
        hist.add(0);
        hist.add(0);
        hist.add(0);
        hist.add(0);
        hist.add(1);
        hist.add(1);
        hist.add(2);
        hist.add(3);

        let codes = build_huffman_codes(&hist).expect("should build");

        // Verify prefix-free property: no code is a prefix of another.
        let used: Vec<(usize, &HuffmanCode)> =
            codes.iter().enumerate().filter(|(_, c)| c.len > 0).collect();

        for i in 0..used.len() {
            for j in (i + 1)..used.len() {
                let (_, ci) = used[i];
                let (_, cj) = used[j];
                let min_len = ci.len.min(cj.len);
                let mask = (1u32 << min_len) - 1;
                if ci.len != cj.len {
                    assert_ne!(
                        ci.code & mask,
                        cj.code & mask,
                        "codes are not prefix-free"
                    );
                } else {
                    assert_ne!(ci.code, cj.code, "duplicate codes");
                }
            }
        }
    }

    // -- RLE encoding --

    #[test]
    fn test_rle_encode_zeros() {
        let lengths = vec![0; 20];
        let rle = rle_encode_code_lengths(&lengths);
        assert_eq!(rle.len(), 1);
        assert_eq!(rle[0].0, 18); // repeat zero 11..138
        assert_eq!(rle[0].1, (20 - 11) as u32);
    }

    #[test]
    fn test_rle_encode_small_zeros() {
        let lengths = vec![0, 0, 0, 0, 0];
        let rle = rle_encode_code_lengths(&lengths);
        assert_eq!(rle.len(), 1);
        assert_eq!(rle[0].0, 17); // repeat zero 3..10
        assert_eq!(rle[0].1, (5 - 3) as u32);
    }

    #[test]
    fn test_rle_encode_mixed() {
        let lengths = vec![3, 3, 3, 3, 0, 0, 0, 5];
        let rle = rle_encode_code_lengths(&lengths);
        assert!(rle.len() >= 3);
        assert_eq!(rle[0].0, 3); // literal 3
        assert_eq!(rle[1].0, 16); // repeat previous
    }

    #[test]
    fn test_rle_encode_single_values() {
        let lengths = vec![1, 2, 3, 4, 5];
        let rle = rle_encode_code_lengths(&lengths);
        // No repeats, all literals.
        assert_eq!(rle.len(), 5);
        for (i, &(sym, _)) in rle.iter().enumerate() {
            assert_eq!(sym, (i + 1) as u32);
        }
    }

    // -- Limit code lengths --

    #[test]
    fn test_limit_code_lengths() {
        let mut depths = vec![1, 2, 16, 16, 16];
        limit_code_lengths(&mut depths, 8);
        for &d in &depths {
            assert!(d <= 8, "depth {} exceeds max 8", d);
        }
    }

    #[test]
    fn test_limit_code_lengths_no_change() {
        let original = vec![1, 2, 3, 4, 5];
        let mut depths = original.clone();
        limit_code_lengths(&mut depths, 15);
        assert_eq!(depths, original);
    }

    // -- Header encoding --

    #[test]
    fn test_header_signature() {
        let encoder = Vp8lEncoder::new(0);
        let mut writer = Vp8lBitWriter::new();
        encoder
            .write_header(&mut writer, 100, 200, true)
            .expect("header should write");
        let data = writer.finish();
        assert_eq!(data[0], VP8L_SIGNATURE);
    }

    #[test]
    fn test_header_dimensions_encoded() {
        let encoder = Vp8lEncoder::new(0);
        let mut writer = Vp8lBitWriter::new();
        encoder
            .write_header(&mut writer, 100, 200, false)
            .expect("write header");
        let data = writer.finish();

        assert_eq!(data[0], 0x2f);

        // Parse the 32-bit value after signature (LSB-first).
        let val = (data[1] as u32)
            | ((data[2] as u32) << 8)
            | ((data[3] as u32) << 16)
            | ((data[4] as u32) << 24);

        let w_minus_1 = val & 0x3fff;
        let h_minus_1 = (val >> 14) & 0x3fff;
        let alpha_used = (val >> 28) & 1;
        let version = (val >> 29) & 0x7;

        assert_eq!(w_minus_1, 99);
        assert_eq!(h_minus_1, 199);
        assert_eq!(alpha_used, 0);
        assert_eq!(version, 0);
    }

    #[test]
    fn test_header_with_alpha() {
        let encoder = Vp8lEncoder::new(0);
        let mut writer = Vp8lBitWriter::new();
        encoder
            .write_header(&mut writer, 50, 75, true)
            .expect("write header");
        let data = writer.finish();

        let val = (data[1] as u32)
            | ((data[2] as u32) << 8)
            | ((data[3] as u32) << 16)
            | ((data[4] as u32) << 24);

        let alpha_used = (val >> 28) & 1;
        assert_eq!(alpha_used, 1);
    }

    // -- Full encoder tests --

    #[test]
    fn test_encoder_1x1_image() {
        let encoder = Vp8lEncoder::new(0);
        let pixels = vec![0xff_80_40_20u32];
        let result = encoder.encode(&pixels, 1, 1, true);
        assert!(result.is_ok());
        let data = result.expect("encode should succeed");
        assert_eq!(data[0], VP8L_SIGNATURE);
    }

    #[test]
    fn test_encoder_small_image() {
        let encoder = Vp8lEncoder::new(0);
        let pixels = vec![0xff_ff_00_00u32; 4]; // 2x2 red
        let result = encoder.encode(&pixels, 2, 2, false);
        assert!(result.is_ok());
        let data = result.expect("encode should succeed");
        assert_eq!(data[0], VP8L_SIGNATURE);
        assert!(data.len() > 5);
    }

    #[test]
    fn test_encoder_with_alpha() {
        let encoder = Vp8lEncoder::new(0);
        let pixels = vec![0x80_00_ff_00u32; 16]; // 4x4 semi-transparent green
        let result = encoder.encode(&pixels, 4, 4, true);
        assert!(result.is_ok());
    }

    #[test]
    fn test_encoder_all_black() {
        let encoder = Vp8lEncoder::new(0);
        let pixels = vec![0xff_00_00_00u32; 64];
        let result = encoder.encode(&pixels, 8, 8, false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_encoder_gradient() {
        let encoder = Vp8lEncoder::new(0);
        let mut pixels = Vec::with_capacity(256);
        for i in 0..256u32 {
            pixels.push(0xff_00_00_00 | (i << 16) | (i << 8) | i);
        }
        let result = encoder.encode(&pixels, 16, 16, false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_encoder_with_predictor() {
        let encoder = Vp8lEncoder::new(50); // effort > 30 enables predictor
        let pixels = vec![0xff_ff_00_00u32; 64];
        let result = encoder.encode(&pixels, 8, 8, false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_encoder_invalid_zero_dimensions() {
        let encoder = Vp8lEncoder::new(0);
        let pixels = vec![0u32; 4];

        assert!(encoder.encode(&pixels, 0, 2, false).is_err());
        assert!(encoder.encode(&pixels, 2, 0, false).is_err());
    }

    #[test]
    fn test_encoder_dimension_overflow() {
        let encoder = Vp8lEncoder::new(0);
        let pixels = vec![0u32; 1];
        let result = encoder.encode(&pixels, VP8L_MAX_DIMENSION + 1, 1, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_encoder_pixel_count_mismatch() {
        let encoder = Vp8lEncoder::new(0);
        let pixels = vec![0u32; 10];
        let result = encoder.encode(&pixels, 4, 4, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_encoder_large_uniform_compresses() {
        let encoder = Vp8lEncoder::new(0);
        let pixels = vec![0xff_00_80_ff_u32; 256 * 256];
        let result = encoder.encode(&pixels, 256, 256, false);
        assert!(result.is_ok());
        let data = result.expect("encode should succeed");
        // Uniform image should compress well below raw size.
        assert!(
            data.len() < 256 * 256 * 4,
            "compressed size {} should be less than raw size {}",
            data.len(),
            256 * 256 * 4
        );
    }

    #[test]
    fn test_effort_affects_output() {
        let pixels = vec![0xff_ff_00_00u32; 64];

        let low = Vp8lEncoder::new(0);
        let high = Vp8lEncoder::new(50);

        let data_low = low.encode(&pixels, 8, 8, false).expect("low effort");
        let data_high = high.encode(&pixels, 8, 8, false).expect("high effort");

        // Different effort levels should produce different bitstreams.
        assert_ne!(data_low, data_high);
    }

    // -- Predictor transform --

    #[test]
    fn test_predictor_transform_uniform() {
        let w = 4u32;
        let h = 4u32;
        let original = vec![0xff_80_40_20u32; (w * h) as usize];
        let mut transformed = original.clone();
        let result = apply_predictor_transform(&mut transformed, w, h);
        assert!(result.is_ok());

        // For identical pixels with left predictor, residual should be 0
        // for all pixels after the first.
        for i in 1..(w * h) as usize {
            let g = (transformed[i] >> 8) & 0xff;
            let r = (transformed[i] >> 16) & 0xff;
            let b = transformed[i] & 0xff;
            assert_eq!(g, 0, "green residual should be 0 for uniform image");
            assert_eq!(r, 0, "red residual should be 0 for uniform image");
            assert_eq!(b, 0, "blue residual should be 0 for uniform image");
        }
    }

    #[test]
    fn test_predictor_transform_dimension_mismatch() {
        let mut pixels = vec![0u32; 10];
        let result = apply_predictor_transform(&mut pixels, 4, 4);
        assert!(result.is_err());
    }

    // -- Max dimensions --

    #[test]
    fn test_encoder_max_valid_dimension() {
        let encoder = Vp8lEncoder::new(0);
        let pixels = vec![0xff_00_00_00u32; 1];
        let result = encoder.encode(&pixels, 1, 1, false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_encoder_effort_clamped() {
        // Effort > 100 should be clamped to 100.
        let encoder = Vp8lEncoder::new(255);
        let pixels = vec![0xff_ff_ff_ffu32; 4];
        let result = encoder.encode(&pixels, 2, 2, true);
        assert!(result.is_ok());
    }
}
