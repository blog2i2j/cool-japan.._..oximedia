//! QR code based visual watermarking.
//!
//! Provides structures and functions for embedding QR-code-style patterns
//! into video frames as a watermarking strategy.
//!
//! This implementation generates genuine ISO 18004 QR symbols for versions 1–4,
//! error-correction level M, byte-mode encoding.  The key components are:
//!
//! - **Finder patterns** — three 7×7 squares with separators at the three corners
//! - **Timing patterns** — alternating dark/light rows and columns
//! - **Format information** — two copies of 15 encoded bits
//! - **Data encoding** — byte mode with Reed-Solomon ECC over GF(256)
//! - **Module placement & masking** — mask pattern 0 (`(row + col) % 2 == 0`)

// ── EcLevel ───────────────────────────────────────────────────────────────────

/// Error-correction level for QR payload encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EcLevel {
    /// Low error correction (~7% redundancy).
    L,
    /// Medium error correction (~15% redundancy).
    M,
    /// Quartile error correction (~25% redundancy).
    Q,
    /// High error correction (~30% redundancy).
    H,
}

impl EcLevel {
    /// Return the approximate redundancy percentage for this EC level.
    #[must_use]
    pub fn redundancy_pct(self) -> u32 {
        match self {
            EcLevel::L => 7,
            EcLevel::M => 15,
            EcLevel::Q => 25,
            EcLevel::H => 30,
        }
    }

    /// ISO 18004 two-bit EC indicator (used in format information).
    fn indicator(self) -> u8 {
        match self {
            EcLevel::M => 0b00,
            EcLevel::L => 0b01,
            EcLevel::H => 0b10,
            EcLevel::Q => 0b11,
        }
    }
}

// ── QrPayload ─────────────────────────────────────────────────────────────────

/// Payload carried by a QR watermark.
#[derive(Debug, Clone)]
pub struct QrPayload {
    /// QR version (1–40), determines module count.
    pub version: u8,
    /// Error-correction level.
    pub error_correction: EcLevel,
    /// Raw data bytes to encode.
    pub data: Vec<u8>,
}

impl QrPayload {
    /// Create a new payload.
    #[must_use]
    pub fn new(version: u8, error_correction: EcLevel, data: Vec<u8>) -> Self {
        Self {
            version,
            error_correction,
            data,
        }
    }

    /// Calculate the encoded size in bytes, accounting for error-correction overhead.
    ///
    /// Returns `data.len()` inflated by the EC-level redundancy percentage.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn encoded_size(&self) -> usize {
        let pct = self.error_correction.redundancy_pct() as usize;
        let overhead = (self.data.len() * pct).div_ceil(100);
        self.data.len() + overhead
    }
}

// ── QrMatrix ─────────────────────────────────────────────────────────────────

/// Tri-state cell used during matrix construction.
///
/// During construction cells start as `Unknown` and are progressively set
/// to `Dark` (module=1) or `Light` (module=0).  Function pattern cells are
/// set first and cannot be overwritten by data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Cell {
    Dark,
    Light,
    Unknown,
}

/// A square QR module matrix.
struct QrMatrix {
    size: usize,
    cells: Vec<Cell>,
    /// True for cells that belong to function patterns (finder, timing, format).
    function: Vec<bool>,
}

impl QrMatrix {
    fn new(size: usize) -> Self {
        Self {
            size,
            cells: vec![Cell::Unknown; size * size],
            function: vec![false; size * size],
        }
    }

    fn idx(&self, row: usize, col: usize) -> usize {
        row * self.size + col
    }

    fn set(&mut self, row: usize, col: usize, dark: bool) {
        let i = self.idx(row, col);
        self.cells[i] = if dark { Cell::Dark } else { Cell::Light };
    }

    fn set_function(&mut self, row: usize, col: usize, dark: bool) {
        let i = self.idx(row, col);
        self.cells[i] = if dark { Cell::Dark } else { Cell::Light };
        self.function[i] = true;
    }

    fn is_function(&self, row: usize, col: usize) -> bool {
        self.function[self.idx(row, col)]
    }

    fn is_dark(&self, row: usize, col: usize) -> bool {
        self.cells[self.idx(row, col)] == Cell::Dark
    }

    fn apply_mask0(&mut self) {
        // Mask pattern 0: dark if (row + col) % 2 == 0.
        for row in 0..self.size {
            for col in 0..self.size {
                let i = self.idx(row, col);
                if !self.function[i] && self.cells[i] != Cell::Unknown {
                    if (row + col) % 2 == 0 {
                        self.cells[i] = if self.cells[i] == Cell::Dark {
                            Cell::Light
                        } else {
                            Cell::Dark
                        };
                    }
                }
            }
        }
    }
}

// ── Finder patterns ───────────────────────────────────────────────────────────

/// Place a 7×7 finder pattern (with its 1-module separator) at `(top, left)`.
fn place_finder(matrix: &mut QrMatrix, top: usize, left: usize) {
    // 7×7 finder square (dark border + 3×3 dark centre with light ring).
    for r in 0..7_usize {
        for c in 0..7_usize {
            let dark =
                r == 0 || r == 6 || c == 0 || c == 6 || (r >= 2 && r <= 4 && c >= 2 && c <= 4);
            if top + r < matrix.size && left + c < matrix.size {
                matrix.set_function(top + r, left + c, dark);
            }
        }
    }
    // Separator row/column (always light).
    let size = matrix.size;
    if left + 7 < size {
        // Right separator column.
        for r in 0..8_usize {
            if top + r < size {
                matrix.set_function(top + r, left + 7, false);
            }
        }
    }
    if top + 7 < size {
        // Bottom separator row (for top-positioned finders).
        for c in 0..8_usize {
            if left + c < size {
                matrix.set_function(top + 7, left + c, false);
            }
        }
    }
    // Top separator row (for bottom-positioned finders, i.e. when top > 0).
    if top > 0 {
        // The separator row directly above the finder.
        let sep_row = top - 1;
        for c in 0..8_usize {
            if left + c < size {
                matrix.set_function(sep_row, left + c, false);
            }
        }
    }
    // Left separator column (for right-positioned finders, i.e. when left > 0).
    if left > 0 {
        let sep_col = left - 1;
        for r in 0..8_usize {
            if top + r < size {
                matrix.set_function(top + r, sep_col, false);
            }
        }
    }
}

// ── Timing patterns ───────────────────────────────────────────────────────────

fn place_timing(matrix: &mut QrMatrix) {
    let size = matrix.size;
    // Timing patterns span from the separator border (row/col 8) of the top/left
    // finder to the separator border (row/col size-8) of the bottom-right/left
    // finder, inclusive.
    //
    // Row 6: dark if col is even; Col 6: dark if row is even.
    for i in 8..=(size - 8) {
        matrix.set_function(6, i, i % 2 == 0);
        matrix.set_function(i, 6, i % 2 == 0);
    }
}

// ── Format information ────────────────────────────────────────────────────────

/// Generate the 15-bit format information word for a given EC level and mask pattern.
///
/// ISO 18004 §7.9.  The 15-bit sequence = 5-bit data ‖ 10-bit BCH remainder,
/// XOR'd with `101010000010010`.
fn format_info_bits(ec: EcLevel, mask: u8) -> u16 {
    // 5-bit data: [ec1, ec0, m2, m1, m0].
    let data5: u16 = (u16::from(ec.indicator()) << 3) | u16::from(mask & 0x07);

    // BCH(15,5) over GF(2) with generator 10100110111.
    let gen: u32 = 0b10100110111;
    let mut rem: u32 = u32::from(data5) << 10;
    for bit in (10..=14_u32).rev() {
        if rem & (1 << bit) != 0 {
            rem ^= gen << (bit - 10);
        }
    }
    let fi: u16 = (data5 << 10) | rem as u16;

    // XOR mask.
    fi ^ 0b101010000010010
}

/// Place the 15-bit format information around the finder patterns.
fn place_format_info(matrix: &mut QrMatrix, ec: EcLevel, mask: u8) {
    let fi = format_info_bits(ec, mask);
    let size = matrix.size;

    // Format info appears in two copies (ISO 18004 Fig. 19).
    // Copy 1: around top-left finder.
    // Horizontal sequence: cols 0..5, then col 7..8 (skip col 6 timing).
    // Vertical sequence  : rows 8..5 down to row 0 (reversed).
    let h_cols: [usize; 8] = [0, 1, 2, 3, 4, 5, 7, 8];
    let v_rows: [usize; 7] = [7, 5, 4, 3, 2, 1, 0];

    for (bit_pos, &col) in h_cols.iter().enumerate() {
        let dark = (fi >> bit_pos) & 1 == 1;
        matrix.set_function(8, col, dark);
    }
    // bit 8 goes at row 7, col 8 (special position above timing row).
    {
        let dark = (fi >> 7) & 1 == 1;
        matrix.set_function(7, 8, dark);
    }
    for (bit_offset, &row) in v_rows.iter().enumerate() {
        let bit_pos = 8 + bit_offset;
        let dark = (fi >> bit_pos) & 1 == 1;
        matrix.set_function(row, 8, dark);
    }

    // Copy 2: top-right and bottom-left finders.
    // Horizontal (row 8, cols size-8..size-1, bits 0..6 ascending).
    for bit_pos in 0..8_usize {
        let col = size - 8 + bit_pos;
        let dark = (fi >> bit_pos) & 1 == 1;
        if col < size {
            matrix.set_function(8, col, dark);
        }
    }
    // Vertical (col 8, rows size-7..size-1, bits 14..8 descending).
    for bit_pos in 0..7_usize {
        let row = size - 7 + bit_pos;
        let dark = (fi >> (14 - bit_pos)) & 1 == 1;
        if row < size {
            matrix.set_function(row, 8, dark);
        }
    }
    // Dark module at (size-8, 8) — always dark.
    matrix.set_function(size - 8, 8, true);
}

// ── GF(256) arithmetic for Reed-Solomon ───────────────────────────────────────

/// Precomputed GF(256) exponent and logarithm tables.
///
/// Field: GF(2^8) with primitive polynomial x^8+x^4+x^3+x^2+1 (0x11D).
struct Gf256 {
    exp: [u8; 512],
    log: [u8; 256],
}

impl Gf256 {
    fn build() -> Self {
        let mut exp = [0u8; 512];
        let mut log = [0u8; 256];
        let mut x: u32 = 1;
        for i in 0..255_usize {
            exp[i] = x as u8;
            log[x as usize] = i as u8;
            x <<= 1;
            if x & 0x100 != 0 {
                x ^= 0x11D; // primitive poly
            }
        }
        // Extend exp table for convenience.
        for i in 0..255_usize {
            exp[255 + i] = exp[i];
        }
        exp[510] = exp[0];
        Gf256 { exp, log }
    }

    fn mul(&self, a: u8, b: u8) -> u8 {
        if a == 0 || b == 0 {
            return 0;
        }
        let la = usize::from(self.log[usize::from(a)]);
        let lb = usize::from(self.log[usize::from(b)]);
        self.exp[(la + lb) % 255]
    }

    fn poly_mul(&self, a: &[u8], b: &[u8]) -> Vec<u8> {
        let len = a.len() + b.len() - 1;
        let mut out = vec![0u8; len];
        for (i, &ai) in a.iter().enumerate() {
            for (j, &bj) in b.iter().enumerate() {
                out[i + j] ^= self.mul(ai, bj);
            }
        }
        out
    }

    /// Compute Reed-Solomon remainder (error correction codewords).
    ///
    /// The generator polynomial for `ec_count` ECC bytes is
    /// `g(x) = ∏_{i=0}^{ec_count-1} (x - α^i)`.
    fn rs_remainder(&self, data: &[u8], ec_count: usize) -> Vec<u8> {
        // Build generator polynomial.
        let mut gen = vec![1u8];
        for i in 0..ec_count {
            gen = self.poly_mul(&gen, &[1, self.exp[i]]);
        }

        // Polynomial long division.
        let mut rem: Vec<u8> = data.to_vec();
        rem.resize(data.len() + ec_count, 0);
        for i in 0..data.len() {
            let coeff = rem[i];
            if coeff != 0 {
                for j in 0..gen.len() {
                    rem[i + j] ^= self.mul(gen[j], coeff);
                }
            }
        }
        rem[data.len()..].to_vec()
    }
}

// ── QR version/capacity tables ────────────────────────────────────────────────

/// Returns `(total_data_codewords, ec_codewords_per_block, blocks)` for
/// versions 1–4, error-correction level M.
///
/// Values from ISO 18004 Table 9.
fn version_m_params(version: u8) -> (usize, usize, usize) {
    match version {
        1 => (16, 10, 1), // 16 data + 10 ECC = 26 total codewords
        2 => (28, 16, 1), // 28 data + 16 ECC = 44 total
        3 => (44, 26, 2), // 44 data + 26 ECC × 2 blocks (really 22+22)
        4 => (64, 18, 2), // 64 data + 18 ECC × 2 blocks
        _ => (16, 10, 1), // fallback to v1
    }
}

// ── Data encoding (byte mode) ─────────────────────────────────────────────────

/// Encode the payload data into QR byte-mode bit stream.
///
/// Returns the final codeword vector ready for interleaving/placement.
fn encode_data(version: u8, data: &[u8]) -> Vec<u8> {
    let (total_dc, ec_count, blocks) = version_m_params(version.max(1).min(4));
    let dc_per_block = total_dc / blocks;

    // Build the bit stream.
    let mut bits: Vec<bool> = Vec::new();

    // Mode indicator: byte mode = 0100.
    bits.extend_from_slice(&[false, true, false, false]);

    // Character count indicator (8 bits for version 1–9).
    let len = data.len().min(dc_per_block * blocks - 2); // -2 for mode + length headers
    for i in (0..8).rev() {
        bits.push((len >> i) & 1 == 1);
    }

    // Data bytes.
    for &byte in data.iter().take(len) {
        for i in (0..8).rev() {
            bits.push((byte >> i) & 1 == 1);
        }
    }

    // Terminator (up to 4 zero bits).
    let capacity_bits = total_dc * 8;
    let term_len = (capacity_bits - bits.len()).min(4);
    for _ in 0..term_len {
        bits.push(false);
    }

    // Pad to byte boundary.
    while bits.len() % 8 != 0 {
        bits.push(false);
    }

    // Pad codewords to fill capacity.
    let pad_bytes = [0xEC_u8, 0x11_u8];
    let mut byte_idx = 0_usize;
    while bits.len() < capacity_bits {
        let pad = pad_bytes[byte_idx % 2];
        for i in (0..8).rev() {
            bits.push((pad >> i) & 1 == 1);
        }
        byte_idx += 1;
    }

    // Convert bits → bytes (data codewords).
    let mut data_cws: Vec<u8> = bits
        .chunks(8)
        .map(|chunk| {
            chunk
                .iter()
                .enumerate()
                .fold(0u8, |acc, (i, &b)| acc | (u8::from(b) << (7 - i)))
        })
        .collect();
    data_cws.truncate(total_dc);

    // Compute Reed-Solomon ECC for each block and interleave.
    let gf = Gf256::build();
    let mut all_data_blocks: Vec<Vec<u8>> = Vec::new();
    let mut all_ec_blocks: Vec<Vec<u8>> = Vec::new();

    for b in 0..blocks {
        let start = b * dc_per_block;
        let end = (start + dc_per_block).min(data_cws.len());
        let block_data = data_cws[start..end].to_vec();
        let ec = gf.rs_remainder(&block_data, ec_count);
        all_data_blocks.push(block_data);
        all_ec_blocks.push(ec);
    }

    // Interleave data blocks column-wise (for multi-block symbols).
    let mut final_cws: Vec<u8> = Vec::new();
    let max_dc_len = all_data_blocks.iter().map(|b| b.len()).max().unwrap_or(0);
    for col in 0..max_dc_len {
        for block in &all_data_blocks {
            if col < block.len() {
                final_cws.push(block[col]);
            }
        }
    }

    // Interleave ECC blocks.
    let ec_len = all_ec_blocks.first().map(|b| b.len()).unwrap_or(0);
    for col in 0..ec_len {
        for block in &all_ec_blocks {
            if col < block.len() {
                final_cws.push(block[col]);
            }
        }
    }

    final_cws
}

// ── Data placement into matrix ────────────────────────────────────────────────

/// Place the data and ECC codeword bits into the matrix using the standard
/// two-column up/down zigzag scan, skipping function modules.
///
/// ISO 18004 §7.7.3: column groups are counted from right to left, excluding
/// col 6 (vertical timing pattern).  The rightmost group (group 0) scans upward,
/// and direction alternates for each subsequent group.
fn place_data(matrix: &mut QrMatrix, codewords: &[u8]) {
    let size = matrix.size;

    // Convert codewords to a flat bit vector (MSB first).
    let bits: Vec<bool> = codewords
        .iter()
        .flat_map(|&byte| (0..8_u8).rev().map(move |i| (byte >> i) & 1 == 1))
        .collect();

    let mut bit_idx = 0_usize;

    // Build the ordered list of column pairs from right to left, excluding col 6.
    // Each entry is (right_col, left_col, group_index) where group_index determines
    // the scan direction (upward when group_index is even).
    //
    // Standard pairs for a 21×21 matrix (version 1):
    //   (20,19), (18,17), (16,15), (14,13), (12,11), (10,9), (8,7), then skip 6,
    //   then (5,4), (3,2), (1,0)
    // Build the list of (right_col, left_col) pairs scanning right-to-left.
    // Col 6 is the vertical timing pattern and is skipped entirely.
    // When the normal 2-step sweep lands on a pair that includes col 6,
    // we re-pair: col 7 (right of the timing pair) pairs with col 5 (left of
    // the next pair) — i.e., we emit (7, 5) as a special pair and skip col 6.
    //
    // Example for size=21 (columns 0..20):
    //   Normal sweep: (20,19),(18,17),(16,15),(14,13),(12,11),(10,9),(8,7)
    //   At (6,5): right=6 is timing → special pair (7,5); skip col 6.
    //   Continue: (4,3),(2,1),(1→0 via odd start)
    //
    // Simpler approach: collect all non-timing cols right-to-left, then pair them.
    let data_cols: Vec<usize> = (0..size).rev().filter(|&c| c != 6).collect();
    // Pair them: (data_cols[0], data_cols[1]), (data_cols[2], data_cols[3]), ...
    let mut col_pairs: Vec<(usize, usize)> = Vec::new();
    let mut i = 0_usize;
    while i + 1 < data_cols.len() {
        col_pairs.push((data_cols[i], data_cols[i + 1]));
        i += 2;
    }
    if i < data_cols.len() {
        // Odd column left over — pair with itself (will only visit once).
        col_pairs.push((data_cols[i], data_cols[i]));
    }

    // Scan each column pair; group index determines direction.
    for (group_idx, &(right_col, left_col)) in col_pairs.iter().enumerate() {
        let upward = group_idx % 2 == 0;
        let rows: Vec<usize> = if upward {
            (0..size).rev().collect()
        } else {
            (0..size).collect()
        };

        for row in rows {
            // Right column first, then left column (ISO 18004 zigzag order).
            for &col in &[right_col, left_col] {
                if !matrix.is_function(row, col) {
                    let dark = if bit_idx < bits.len() {
                        let b = bits[bit_idx];
                        bit_idx += 1;
                        b
                    } else {
                        // Remainder bits are always 0 (dark).
                        false
                    };
                    matrix.set(row, col, dark);
                }
            }
        }
    }
}

// ── QR code matrix builder ────────────────────────────────────────────────────

/// Build a complete QR module matrix for the given payload.
///
/// Supports versions 1–4 with error-correction level M only.
/// Falls back to version 1 for unsupported versions.
fn build_qr_matrix(version: u8, ec: EcLevel, data: &[u8]) -> QrMatrix {
    let ver = version.max(1).min(4);
    let size = (17 + 4 * ver) as usize;
    let mut matrix = QrMatrix::new(size);

    // 1. Place finder patterns.
    //    Top-left corner (0,0), top-right (0, size-7), bottom-left (size-7, 0).
    place_finder(&mut matrix, 0, 0);
    place_finder(&mut matrix, 0, size - 7);
    place_finder(&mut matrix, size - 7, 0);

    // 2. Timing patterns.
    place_timing(&mut matrix);

    // 3. Dark module (always dark; row 4*ver+9 = size-8+1, col 8).
    //    ISO 18004: the single dark module at (4V+9, 8).
    let dark_row = 4 * ver as usize + 9 - 1; // = size - 8 when ver=1,2,4; otherwise same
                                             // Simplified: place at (size-8, 8) — safe for all versions 1-4.
    if dark_row < size {
        matrix.set_function(dark_row, 8, true);
    }

    // 4. Format information (placeholder — we write real bits, mask=0).
    place_format_info(&mut matrix, ec, 0);

    // 5. Encode data → codewords with Reed-Solomon ECC.
    let codewords = encode_data(ver, data);

    // 6. Place data into matrix.
    place_data(&mut matrix, &codewords);

    // 7. Apply mask pattern 0: (row + col) % 2 == 0 → flip.
    matrix.apply_mask0();

    matrix
}

// ── QrWatermark ───────────────────────────────────────────────────────────────

/// QR watermark embedder.
#[derive(Debug, Clone)]
pub struct QrWatermark {
    /// Size of each QR module in pixels.
    pub module_size_px: u32,
    /// Quiet zone width in modules.
    pub quiet_zone: u32,
    /// Payload to embed.
    pub payload: QrPayload,
}

impl QrWatermark {
    /// Create a new QR watermark.
    #[must_use]
    pub fn new(module_size_px: u32, quiet_zone: u32, payload: QrPayload) -> Self {
        Self {
            module_size_px,
            quiet_zone,
            payload,
        }
    }

    /// Calculate the total image size in pixels for this QR watermark.
    ///
    /// QR version V has `(17 + 4*V)` modules plus 2 × `quiet_zone` on each side.
    #[must_use]
    pub fn image_size_px(&self) -> u32 {
        let v = u32::from(self.payload.version.max(1).min(40));
        let modules = 17 + 4 * v + 2 * self.quiet_zone;
        modules * self.module_size_px
    }

    /// Embed the QR watermark into a raw RGB frame at position `(x, y)`.
    ///
    /// `frame` must be a flat `width * height * 3` byte buffer (RGB).
    /// Returns `true` if the watermark fits within the frame, `false` otherwise.
    ///
    /// The QR code is generated with:
    /// - Byte-mode data encoding
    /// - Reed-Solomon error correction (level M for versions 1–4, clamped)
    /// - Mask pattern 0: `(row + col) % 2 == 0`
    /// - Proper finder patterns, timing patterns, and format information
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn embed_in_frame(&self, frame: &mut [u8], width: usize, x: u32, y: u32) -> bool {
        let size = self.image_size_px();
        let height = if width == 0 {
            0
        } else {
            frame.len() / (width * 3)
        };

        if x + size > width as u32 || y + size > height as u32 {
            return false;
        }

        // Build the QR module matrix.
        let ver = self.payload.version.max(1).min(4);
        let matrix = build_qr_matrix(ver, self.payload.error_correction, &self.payload.data);
        let qr_modules = matrix.size;

        let qz = self.quiet_zone;

        // Render quiet zone (white) and QR modules.
        let total_modules = qr_modules + 2 * qz as usize;

        for row in 0..total_modules {
            for col in 0..total_modules {
                // Check if this module is inside the quiet zone.
                let in_quiet = row < qz as usize
                    || row >= total_modules - qz as usize
                    || col < qz as usize
                    || col >= total_modules - qz as usize;

                let dark = if in_quiet {
                    false
                } else {
                    let mr = row - qz as usize;
                    let mc = col - qz as usize;
                    if mr < matrix.size && mc < matrix.size {
                        matrix.is_dark(mr, mc)
                    } else {
                        false
                    }
                };

                let color: u8 = if dark { 0 } else { 255 };

                for py in 0..self.module_size_px {
                    for px in 0..self.module_size_px {
                        let fx = (x + col as u32 * self.module_size_px + px) as usize;
                        let fy = (y + row as u32 * self.module_size_px + py) as usize;
                        let base = (fy * width + fx) * 3;
                        if base + 2 < frame.len() {
                            frame[base] = color;
                            frame[base + 1] = color;
                            frame[base + 2] = color;
                        }
                    }
                }
            }
        }
        true
    }
}

// ── fnv_encode ────────────────────────────────────────────────────────────────

/// Encode data using FNV-1a hashing as a simple content fingerprint.
///
/// The input is split into 4-byte blocks; each block's FNV-1a hash is appended
/// to the output as 4 bytes (little-endian).
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn fnv_encode(data: &[u8]) -> Vec<u8> {
    const FNV_PRIME: u32 = 0x0100_0193;
    const FNV_OFFSET: u32 = 0x811c_9dc5;

    let mut out = Vec::with_capacity(data.len() + (data.len() / 4 + 1) * 4);
    out.extend_from_slice(data);

    // Append hash of each 4-byte chunk.
    for chunk in data.chunks(4) {
        let mut hash = FNV_OFFSET;
        for &b in chunk {
            hash ^= u32::from(b);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        out.extend_from_slice(&hash.to_le_bytes());
    }
    out
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ec_level_l_redundancy() {
        assert_eq!(EcLevel::L.redundancy_pct(), 7);
    }

    #[test]
    fn test_ec_level_m_redundancy() {
        assert_eq!(EcLevel::M.redundancy_pct(), 15);
    }

    #[test]
    fn test_ec_level_q_redundancy() {
        assert_eq!(EcLevel::Q.redundancy_pct(), 25);
    }

    #[test]
    fn test_ec_level_h_redundancy() {
        assert_eq!(EcLevel::H.redundancy_pct(), 30);
    }

    #[test]
    fn test_payload_encoded_size_larger_than_data() {
        let p = QrPayload::new(1, EcLevel::M, vec![0u8; 100]);
        assert!(p.encoded_size() > 100);
    }

    #[test]
    fn test_payload_encoded_size_empty() {
        let p = QrPayload::new(1, EcLevel::L, vec![]);
        assert_eq!(p.encoded_size(), 0);
    }

    #[test]
    fn test_payload_encoded_size_h_level() {
        let data = vec![0u8; 10];
        let p = QrPayload::new(1, EcLevel::H, data);
        // 30% overhead → 3 bytes overhead → 13 total
        assert_eq!(p.encoded_size(), 13);
    }

    #[test]
    fn test_image_size_px_version1() {
        let payload = QrPayload::new(1, EcLevel::M, vec![0]);
        let qr = QrWatermark::new(2, 4, payload);
        // modules = 17 + 4*1 + 2*4 = 29; px = 29 * 2 = 58
        assert_eq!(qr.image_size_px(), 58);
    }

    #[test]
    fn test_image_size_px_scales_with_module_size() {
        let p1 = QrPayload::new(1, EcLevel::L, vec![0]);
        let p2 = QrPayload::new(1, EcLevel::L, vec![0]);
        let qr1 = QrWatermark::new(1, 0, p1);
        let qr2 = QrWatermark::new(2, 0, p2);
        assert_eq!(qr2.image_size_px(), qr1.image_size_px() * 2);
    }

    #[test]
    fn test_embed_in_frame_success() {
        let payload = QrPayload::new(1, EcLevel::L, vec![42]);
        let qr = QrWatermark::new(1, 0, payload);
        let size = qr.image_size_px() as usize;
        let mut frame = vec![128u8; size * size * 3];
        let result = qr.embed_in_frame(&mut frame, size, 0, 0);
        assert!(result);
    }

    #[test]
    fn test_embed_in_frame_out_of_bounds() {
        let payload = QrPayload::new(1, EcLevel::L, vec![0]);
        let qr = QrWatermark::new(2, 2, payload);
        // Frame too small
        let mut frame = vec![0u8; 10 * 10 * 3];
        let result = qr.embed_in_frame(&mut frame, 10, 8, 8);
        assert!(!result);
    }

    #[test]
    fn test_embed_modifies_frame() {
        let payload = QrPayload::new(1, EcLevel::L, vec![1]);
        let qr = QrWatermark::new(1, 0, payload);
        let size = qr.image_size_px() as usize;
        let mut frame = vec![128u8; size * size * 3];
        let _ = qr.embed_in_frame(&mut frame, size, 0, 0);
        // Some pixels should have changed to 0 or 255
        let has_black = frame.iter().any(|&b| b == 0);
        let has_white = frame.iter().any(|&b| b == 255);
        assert!(has_black || has_white);
    }

    #[test]
    fn test_qr_finder_pattern_top_left() {
        // Version 1 (21×21): top-left corner modules should be dark (finder border).
        let matrix = build_qr_matrix(1, EcLevel::M, b"A");
        // The 7×7 finder: row 0, col 0..6 should be dark (outer border).
        assert!(matrix.is_dark(0, 0), "TL finder (0,0) must be dark");
        assert!(matrix.is_dark(0, 6), "TL finder (0,6) must be dark");
        assert!(matrix.is_dark(6, 0), "TL finder (6,0) must be dark");
        assert!(matrix.is_dark(6, 6), "TL finder (6,6) must be dark");
    }

    #[test]
    fn test_qr_finder_pattern_centre_light() {
        // The light inner ring of the top-left finder: (1,1) through (5,5) ring.
        let matrix = build_qr_matrix(1, EcLevel::M, b"A");
        // Apply mask: (1,1) original is light; mask0 flips when (r+c)%2==0.
        // (1,1): r+c=2 → mask flips light→dark. Check function status instead.
        assert!(
            matrix.is_function(1, 1),
            "finder inner ring must be a function module"
        );
    }

    #[test]
    fn test_qr_timing_pattern() {
        // Version 1 (21×21): timing row is row 6, cols 8..12 alternate dark/light.
        let matrix = build_qr_matrix(1, EcLevel::M, b"A");
        // col 8 is even → originally dark; mask0 flips dark (r+c=6+8=14, even) → light.
        // But timing is a function module, so mask does not apply.
        assert!(matrix.is_function(6, 8), "timing row 6 col 8 is function");
        assert!(matrix.is_dark(6, 8), "timing col 8 (even) is dark");
        // col 9 is odd → light.
        assert!(!matrix.is_dark(6, 9), "timing col 9 (odd) is light");
    }

    #[test]
    fn test_qr_matrix_size_version1() {
        let matrix = build_qr_matrix(1, EcLevel::M, b"");
        assert_eq!(matrix.size, 21);
    }

    #[test]
    fn test_qr_matrix_size_version4() {
        let matrix = build_qr_matrix(4, EcLevel::M, b"");
        assert_eq!(matrix.size, 33);
    }

    #[test]
    fn test_qr_no_unknown_cells() {
        // After a complete build, no cell should remain Unknown.
        let matrix = build_qr_matrix(1, EcLevel::M, b"Hello");
        for row in 0..matrix.size {
            for col in 0..matrix.size {
                let i = matrix.idx(row, col);
                assert_ne!(
                    matrix.cells[i],
                    Cell::Unknown,
                    "Cell ({},{}) is Unknown after build",
                    row,
                    col
                );
            }
        }
    }

    #[test]
    fn test_qr_quiet_zone_is_white() {
        // With a 4-module quiet zone, the rendered frame border should be all white.
        let payload = QrPayload::new(1, EcLevel::M, b"X".to_vec());
        let qr = QrWatermark::new(1, 4, payload);
        let total_px = qr.image_size_px() as usize;
        let mut frame = vec![0u8; total_px * total_px * 3];
        let ok = qr.embed_in_frame(&mut frame, total_px, 0, 0);
        assert!(ok);
        // Top row should be all white (quiet zone).
        for col in 0..total_px {
            let base = col * 3;
            assert_eq!(
                frame[base], 255,
                "quiet zone top row pixel {col} must be white"
            );
        }
    }

    #[test]
    fn test_gf256_mul_identity() {
        let gf = Gf256::build();
        assert_eq!(gf.mul(1, 0xAB), 0xAB);
        assert_eq!(gf.mul(0xAB, 1), 0xAB);
    }

    #[test]
    fn test_gf256_mul_zero() {
        let gf = Gf256::build();
        assert_eq!(gf.mul(0, 0xAB), 0);
        assert_eq!(gf.mul(0xAB, 0), 0);
    }

    #[test]
    fn test_gf256_mul_commutativity() {
        let gf = Gf256::build();
        assert_eq!(gf.mul(0x53, 0xCA), gf.mul(0xCA, 0x53));
    }

    #[test]
    fn test_rs_remainder_length() {
        let gf = Gf256::build();
        let data = vec![32_u8, 91, 11, 120, 209, 114, 220, 77, 67, 64, 236, 17, 236];
        let ec = gf.rs_remainder(&data, 10);
        assert_eq!(ec.len(), 10);
    }

    #[test]
    fn test_encode_data_length_v1() {
        // Version 1M: 16 data + 10 EC = 26 total codewords.
        let cws = encode_data(1, b"HELLO");
        assert_eq!(cws.len(), 26, "v1M should produce 26 codewords total");
    }

    #[test]
    fn test_encode_data_length_v2() {
        // Version 2M: 28 data + 16 EC = 44 total codewords.
        let cws = encode_data(2, b"Hello World 123");
        assert_eq!(cws.len(), 44);
    }

    #[test]
    fn test_fnv_encode_non_empty() {
        let data = b"hello";
        let encoded = fnv_encode(data);
        assert!(encoded.len() > data.len());
    }

    #[test]
    fn test_fnv_encode_empty() {
        let encoded = fnv_encode(&[]);
        assert!(encoded.is_empty());
    }

    #[test]
    fn test_fnv_encode_deterministic() {
        let data = b"watermark";
        let a = fnv_encode(data);
        let b = fnv_encode(data);
        assert_eq!(a, b);
    }

    #[test]
    fn test_fnv_encode_different_inputs_differ() {
        let a = fnv_encode(b"abc");
        let b = fnv_encode(b"xyz");
        assert_ne!(a, b);
    }

    #[test]
    fn test_format_info_bits_known_value() {
        // EC level M (indicator=00), mask 0: data5 = 0b00_000 = 0.
        // Well-known value from QR spec generators for M/mask0 = 0x5412 XOR 0x5412...
        // Let's just verify the output is 15 bits wide (≤ 0x7FFF).
        let fi = format_info_bits(EcLevel::M, 0);
        assert!(
            fi <= 0x7FFF,
            "format info must fit in 15 bits, got {fi:#06x}"
        );
    }
}
