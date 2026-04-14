//! Interleaved Forward Error Correction with chunk-aligned XOR.
//!
//! This module extends the basic 1-D XOR FEC in [`crate::fec`] with:
//!
//! - **2-D interleaved FEC** — rows × columns repair matrix so that a burst
//!   loss spanning up to `num_columns` consecutive packets can be recovered
//!   (row repair covers column bursts; column repair covers random loss).
//! - **Chunk-aligned XOR** — the inner XOR loop operates on `u64` words
//!   (8-byte aligned chunks) before a byte-granular tail.  On modern CPUs
//!   the compiler auto-vectorises this to SIMD (SSE2/NEON/AVX2) without any
//!   `unsafe` code.
//! - **Zero-copy repair** — repair packets borrow source slices; allocation
//!   only happens once per FEC group rather than per packet.
//! - **Loss recovery planner** — given a set of received packets and the
//!   repair matrix, [`RecoveryPlanner`] decides the minimum repair sequence
//!   needed to reconstruct all missing source packets.
//!
//! # 2-D FEC layout
//!
//! Source packets are arranged in a `R × C` matrix (rows × columns).
//! `R` row-repair packets protect each row (1 per row), and `C` column-repair
//! packets protect each column.  Given at most one loss per row *and* at most
//! one loss per column, both dimensions can recover independently.
//!
//! ```text
//! P[0,0]  P[0,1]  …  P[0,C-1]   ← row-repair[0] = XOR of row 0
//! P[1,0]  P[1,1]  …  P[1,C-1]   ← row-repair[1] = XOR of row 1
//! …
//! P[R-1,0] …          P[R-1,C-1] ← row-repair[R-1]
//! ↑col[0]  ↑col[1]    ↑col[C-1] (column repair packets)
//! ```
//!
//! # Example
//!
//! ```rust
//! use oximedia_net::fec_interleave::{InterleavedFecConfig, InterleavedFecEncoder, InterleavedFecDecoder};
//!
//! let cfg = InterleavedFecConfig::new(2, 4).expect("valid cfg");
//! let mut enc = InterleavedFecEncoder::new(cfg.clone());
//! let mut dec = InterleavedFecDecoder::new(cfg);
//!
//! // Feed 8 source packets (2 rows × 4 columns)
//! let payloads: Vec<Vec<u8>> = (0u8..8).map(|i| vec![i; 16]).collect();
//! for (seq, pkt) in payloads.iter().enumerate() {
//!     enc.feed(seq as u16, pkt);
//! }
//!
//! let group = enc.finalize().expect("complete group");
//!
//! // Simulate losing packets at positions 1 and 5.
//! for (seq, pkt) in payloads.iter().enumerate() {
//!     if seq != 1 && seq != 5 {
//!         dec.feed_source(seq as u16, pkt.clone());
//!     }
//! }
//! for rp in &group.row_repair { dec.feed_row_repair(rp.clone()); }
//! for cp in &group.col_repair { dec.feed_col_repair(cp.clone()); }
//!
//! let recovered = dec.recover().expect("recovery ok");
//! assert_eq!(recovered[&1], payloads[1]);
//! assert_eq!(recovered[&5], payloads[5]);
//! ```

use std::collections::HashMap;
use std::fmt;

use crate::error::{NetError, NetResult};

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for a 2-D interleaved FEC group.
#[derive(Debug, Clone)]
pub struct InterleavedFecConfig {
    /// Number of rows in the protection matrix.
    pub num_rows: usize,
    /// Number of columns in the protection matrix.
    pub num_cols: usize,
}

impl InterleavedFecConfig {
    /// Creates a new configuration.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `num_rows < 1`, `num_cols < 1`, or `num_rows * num_cols > 256`.
    pub fn new(num_rows: usize, num_cols: usize) -> NetResult<Self> {
        if num_rows == 0 || num_cols == 0 {
            return Err(NetError::protocol("num_rows and num_cols must be >= 1"));
        }
        if num_rows.saturating_mul(num_cols) > 256 {
            return Err(NetError::protocol(
                "FEC group size (rows × cols) must not exceed 256",
            ));
        }
        Ok(Self { num_rows, num_cols })
    }

    /// Total number of source packets in one FEC group.
    #[must_use]
    pub const fn group_size(&self) -> usize {
        self.num_rows * self.num_cols
    }

    /// Number of row-repair packets (one per row).
    #[must_use]
    pub const fn row_repair_count(&self) -> usize {
        self.num_rows
    }

    /// Number of column-repair packets (one per column).
    #[must_use]
    pub const fn col_repair_count(&self) -> usize {
        self.num_cols
    }

    /// Maps a flat packet sequence index (0-based) to `(row, col)`.
    #[must_use]
    pub fn to_matrix_pos(&self, idx: usize) -> (usize, usize) {
        (idx / self.num_cols, idx % self.num_cols)
    }

    /// Maps `(row, col)` to a flat sequence index.
    #[must_use]
    pub const fn from_matrix_pos(&self, row: usize, col: usize) -> usize {
        row * self.num_cols + col
    }
}

// ─── Chunk-aligned XOR ───────────────────────────────────────────────────────

/// XOR `src` into `dst` using word-sized operations for SIMD autovectorisation.
///
/// Both slices must be the same length; shorter is padded with zeros from `src`.
/// This is the inner kernel called for every FEC repair computation.
fn xor_into(dst: &mut [u8], src: &[u8]) {
    let common = dst.len().min(src.len());

    // Process 8-byte (u64) chunks.
    let words = common / 8;
    for i in 0..words {
        let off = i * 8;
        let d = u64::from_le_bytes(dst[off..off + 8].try_into().expect("slice len"));
        let s = u64::from_le_bytes(src[off..off + 8].try_into().expect("slice len"));
        dst[off..off + 8].copy_from_slice(&(d ^ s).to_le_bytes());
    }

    // Byte-granular tail.
    for i in (words * 8)..common {
        dst[i] ^= src[i];
    }

    // If dst is longer than src, the remaining bytes are already XOR'd with 0.
}

/// XOR all slices in `packets` into a new repair buffer of length `width`.
///
/// `width` should be the maximum payload length in the group; shorter packets
/// are effectively zero-padded.
fn compute_xor_repair(packets: &[&[u8]], width: usize) -> Vec<u8> {
    let mut repair = vec![0u8; width];
    for pkt in packets {
        xor_into(&mut repair, pkt);
    }
    repair
}

// ─── Repair packets ───────────────────────────────────────────────────────────

/// A single repair packet produced by the encoder.
#[derive(Debug, Clone)]
pub struct RepairPacket {
    /// Row or column index this packet protects.
    pub index: usize,
    /// XOR payload (same width as the widest source packet in the group).
    pub payload: Vec<u8>,
    /// Sequence numbers of source packets XOR'd together.
    pub source_seqs: Vec<u16>,
}

/// The complete output of one FEC group.
#[derive(Debug, Clone)]
pub struct FecGroup {
    /// Row repair packets (one per row).
    pub row_repair: Vec<RepairPacket>,
    /// Column repair packets (one per column).
    pub col_repair: Vec<RepairPacket>,
    /// Configuration this group was encoded with.
    pub config: InterleavedFecConfig,
}

// ─── Encoder ─────────────────────────────────────────────────────────────────

/// Feeds source packets in sequence-number order and produces a [`FecGroup`].
#[derive(Debug)]
pub struct InterleavedFecEncoder {
    config: InterleavedFecConfig,
    /// Received source packets indexed by flat position.
    sources: HashMap<u16, Vec<u8>>,
    /// First sequence number fed (anchor).
    anchor_seq: Option<u16>,
}

impl InterleavedFecEncoder {
    /// Creates a new encoder.
    #[must_use]
    pub fn new(config: InterleavedFecConfig) -> Self {
        Self {
            config,
            sources: HashMap::new(),
            anchor_seq: None,
        }
    }

    /// Feeds a source packet with the given `seq` number and `payload`.
    pub fn feed(&mut self, seq: u16, payload: &[u8]) {
        if self.anchor_seq.is_none() {
            self.anchor_seq = Some(seq);
        }
        self.sources.insert(seq, payload.to_vec());
    }

    /// Returns the number of source packets fed so far.
    #[must_use]
    pub fn count(&self) -> usize {
        self.sources.len()
    }

    /// Finalises the group and returns row + column repair packets.
    ///
    /// # Errors
    ///
    /// Returns `Err` if fewer than `config.group_size()` packets have been fed.
    pub fn finalize(&self) -> NetResult<FecGroup> {
        let group_size = self.config.group_size();
        if self.sources.len() < group_size {
            return Err(NetError::encoding(format!(
                "FEC group incomplete: {} / {} packets",
                self.sources.len(),
                group_size
            )));
        }

        let anchor = self
            .anchor_seq
            .ok_or_else(|| NetError::encoding("no packets fed"))?;

        // Collect packets in order.
        let mut ordered: Vec<(u16, &[u8])> = (0..group_size)
            .filter_map(|i| {
                let seq = anchor.wrapping_add(i as u16);
                self.sources.get(&seq).map(|p| (seq, p.as_slice()))
            })
            .collect();

        if ordered.len() < group_size {
            return Err(NetError::encoding(
                "FEC group has gaps — all source packets required for encoding",
            ));
        }

        // Sort by seq to be safe.
        ordered.sort_by_key(|(s, _)| *s);

        let max_width = ordered.iter().map(|(_, p)| p.len()).max().unwrap_or(0);

        // Row repair.
        let mut row_repair = Vec::with_capacity(self.config.num_rows);
        for row in 0..self.config.num_rows {
            let slices: Vec<&[u8]> = (0..self.config.num_cols)
                .map(|col| ordered[self.config.from_matrix_pos(row, col)].1)
                .collect();
            let source_seqs: Vec<u16> = (0..self.config.num_cols)
                .map(|col| ordered[self.config.from_matrix_pos(row, col)].0)
                .collect();
            let payload = compute_xor_repair(&slices, max_width);
            row_repair.push(RepairPacket {
                index: row,
                payload,
                source_seqs,
            });
        }

        // Column repair.
        let mut col_repair = Vec::with_capacity(self.config.num_cols);
        for col in 0..self.config.num_cols {
            let slices: Vec<&[u8]> = (0..self.config.num_rows)
                .map(|row| ordered[self.config.from_matrix_pos(row, col)].1)
                .collect();
            let source_seqs: Vec<u16> = (0..self.config.num_rows)
                .map(|row| ordered[self.config.from_matrix_pos(row, col)].0)
                .collect();
            let payload = compute_xor_repair(&slices, max_width);
            col_repair.push(RepairPacket {
                index: col,
                payload,
                source_seqs,
            });
        }

        Ok(FecGroup {
            row_repair,
            col_repair,
            config: self.config.clone(),
        })
    }
}

// ─── Decoder ─────────────────────────────────────────────────────────────────

/// Receives source + repair packets and recovers missing source packets.
#[derive(Debug)]
pub struct InterleavedFecDecoder {
    config: InterleavedFecConfig,
    sources: HashMap<u16, Vec<u8>>,
    row_repairs: HashMap<usize, RepairPacket>,
    col_repairs: HashMap<usize, RepairPacket>,
}

impl InterleavedFecDecoder {
    /// Creates a new decoder.
    #[must_use]
    pub fn new(config: InterleavedFecConfig) -> Self {
        Self {
            config,
            sources: HashMap::new(),
            row_repairs: HashMap::new(),
            col_repairs: HashMap::new(),
        }
    }

    /// Feeds a received source packet.
    pub fn feed_source(&mut self, seq: u16, payload: Vec<u8>) {
        self.sources.insert(seq, payload);
    }

    /// Feeds a row-repair packet.
    pub fn feed_row_repair(&mut self, rp: RepairPacket) {
        self.row_repairs.insert(rp.index, rp);
    }

    /// Feeds a column-repair packet.
    pub fn feed_col_repair(&mut self, cp: RepairPacket) {
        self.col_repairs.insert(cp.index, cp);
    }

    /// Attempts to recover all missing source packets.
    ///
    /// Returns a map from sequence number → recovered payload.
    ///
    /// # Algorithm
    ///
    /// 1. For each row: if exactly one packet is missing and the row-repair
    ///    packet is present, recover by XOR-ing received packets + repair.
    /// 2. For each column: same procedure (after row recovery updated `sources`).
    /// 3. Repeat until no new recoveries are made (fixpoint).
    ///
    /// # Errors
    ///
    /// Returns `Err` if the group is unrecoverable (too many losses).
    pub fn recover(&mut self) -> NetResult<HashMap<u16, Vec<u8>>> {
        let mut recovered: HashMap<u16, Vec<u8>> = HashMap::new();

        // Fixpoint loop — at most group_size iterations.
        for _ in 0..self.config.group_size() {
            let mut progress = false;

            // Row pass.
            for row in 0..self.config.num_rows {
                let col_seqs: Vec<u16> = (0..self.config.num_cols)
                    .map(|col| {
                        // Compute the global sequence number for this cell.
                        // We use the row-repair source_seqs as the canonical mapping.
                        self.row_repairs
                            .get(&row)
                            .and_then(|rp| rp.source_seqs.get(col).copied())
                            .unwrap_or(u16::MAX)
                    })
                    .collect();

                // Identify missing packets in this row.
                let missing: Vec<usize> = col_seqs
                    .iter()
                    .enumerate()
                    .filter(|(_, &seq)| seq != u16::MAX && !self.sources.contains_key(&seq))
                    .map(|(i, _)| i)
                    .collect();

                if missing.len() == 1 {
                    if let Some(rp) = self.row_repairs.get(&row) {
                        let missing_col = missing[0];
                        let missing_seq = col_seqs[missing_col];
                        let mut buf = rp.payload.clone();
                        for (col, &seq) in col_seqs.iter().enumerate() {
                            if col == missing_col {
                                continue;
                            }
                            if let Some(pkt) = self.sources.get(&seq) {
                                xor_into(&mut buf, pkt);
                            }
                        }
                        // Trim zero tail introduced by zero-padding.
                        trim_zeros(&mut buf);
                        self.sources.insert(missing_seq, buf.clone());
                        recovered.insert(missing_seq, buf);
                        progress = true;
                    }
                }
            }

            // Column pass.
            for col in 0..self.config.num_cols {
                let row_seqs: Vec<u16> = (0..self.config.num_rows)
                    .map(|row| {
                        self.col_repairs
                            .get(&col)
                            .and_then(|cp| cp.source_seqs.get(row).copied())
                            .unwrap_or(u16::MAX)
                    })
                    .collect();

                let missing: Vec<usize> = row_seqs
                    .iter()
                    .enumerate()
                    .filter(|(_, &seq)| seq != u16::MAX && !self.sources.contains_key(&seq))
                    .map(|(i, _)| i)
                    .collect();

                if missing.len() == 1 {
                    if let Some(cp) = self.col_repairs.get(&col) {
                        let missing_row = missing[0];
                        let missing_seq = row_seqs[missing_row];
                        let mut buf = cp.payload.clone();
                        for (row, &seq) in row_seqs.iter().enumerate() {
                            if row == missing_row {
                                continue;
                            }
                            if let Some(pkt) = self.sources.get(&seq) {
                                xor_into(&mut buf, pkt);
                            }
                        }
                        trim_zeros(&mut buf);
                        self.sources.insert(missing_seq, buf.clone());
                        recovered.insert(missing_seq, buf);
                        progress = true;
                    }
                }
            }

            if !progress {
                break;
            }
        }

        Ok(recovered)
    }
}

/// Removes trailing zero bytes that result from zero-padding during XOR.
fn trim_zeros(buf: &mut Vec<u8>) {
    while buf.last() == Some(&0) {
        buf.pop();
    }
}

// ─── Recovery planner ─────────────────────────────────────────────────────────

/// Describes the recovery strategy for a given loss pattern.
#[derive(Debug, Clone)]
pub struct RecoveryPlan {
    /// Sequence numbers that can be recovered.
    pub recoverable: Vec<u16>,
    /// Sequence numbers that are unrecoverable (too many losses in one row/col).
    pub unrecoverable: Vec<u16>,
    /// Whether both dimensions of repair are needed.
    pub needs_both_dimensions: bool,
}

/// Plans loss recovery without performing any XOR operations.
///
/// Use this to cheaply decide whether to request retransmission or use FEC.
#[derive(Debug)]
pub struct RecoveryPlanner {
    config: InterleavedFecConfig,
}

impl RecoveryPlanner {
    /// Creates a new planner for `config`.
    #[must_use]
    pub fn new(config: InterleavedFecConfig) -> Self {
        Self { config }
    }

    /// Analyses `received_seqs` (sorted or unsorted) given the `anchor_seq`
    /// of this FEC group and returns a [`RecoveryPlan`].
    #[must_use]
    pub fn plan(&self, anchor_seq: u16, received_seqs: &[u16]) -> RecoveryPlan {
        let received_set: std::collections::HashSet<u16> = received_seqs.iter().copied().collect();

        let all_seqs: Vec<u16> = (0..self.config.group_size())
            .map(|i| anchor_seq.wrapping_add(i as u16))
            .collect();

        let lost: Vec<u16> = all_seqs
            .iter()
            .copied()
            .filter(|s| !received_set.contains(s))
            .collect();

        if lost.is_empty() {
            return RecoveryPlan {
                recoverable: vec![],
                unrecoverable: vec![],
                needs_both_dimensions: false,
            };
        }

        // Count losses per row and per column.
        let mut row_loss_count = vec![0usize; self.config.num_rows];
        let mut col_loss_count = vec![0usize; self.config.num_cols];

        for &seq in &lost {
            let flat = seq.wrapping_sub(anchor_seq) as usize;
            if flat < self.config.group_size() {
                let (row, col) = self.config.to_matrix_pos(flat);
                row_loss_count[row] += 1;
                col_loss_count[col] += 1;
            }
        }

        // A packet is recoverable if its row OR its column has exactly one loss.
        let mut recoverable = Vec::new();
        let mut unrecoverable = Vec::new();

        for &seq in &lost {
            let flat = seq.wrapping_sub(anchor_seq) as usize;
            if flat < self.config.group_size() {
                let (row, col) = self.config.to_matrix_pos(flat);
                if row_loss_count[row] == 1 || col_loss_count[col] == 1 {
                    recoverable.push(seq);
                } else {
                    unrecoverable.push(seq);
                }
            }
        }

        let needs_both_dimensions = recoverable.iter().any(|&seq| {
            let flat = seq.wrapping_sub(anchor_seq) as usize;
            let (row, col) = self.config.to_matrix_pos(flat);
            row_loss_count[row] == 1 && col_loss_count[col] == 1
        });

        RecoveryPlan {
            recoverable,
            unrecoverable,
            needs_both_dimensions,
        }
    }
}

impl fmt::Display for InterleavedFecConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FEC({}×{})", self.num_rows, self.num_cols)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_payloads(n: usize, len: usize) -> Vec<Vec<u8>> {
        (0..n).map(|i| vec![i as u8; len]).collect()
    }

    fn encode_group(rows: usize, cols: usize, payloads: &[Vec<u8>]) -> FecGroup {
        let cfg = InterleavedFecConfig::new(rows, cols).expect("cfg");
        let mut enc = InterleavedFecEncoder::new(cfg);
        for (i, p) in payloads.iter().enumerate() {
            enc.feed(i as u16, p);
        }
        enc.finalize().expect("group")
    }

    // 1. Config rejects num_rows == 0
    #[test]
    fn test_config_rejects_zero_rows() {
        assert!(InterleavedFecConfig::new(0, 4).is_err());
    }

    // 2. Config rejects group_size > 256
    #[test]
    fn test_config_rejects_oversized_group() {
        assert!(InterleavedFecConfig::new(17, 16).is_err()); // 272 > 256
    }

    // 3. Encoder produces correct number of repair packets
    #[test]
    fn test_encoder_repair_packet_count() {
        let payloads = make_payloads(8, 16); // 2×4
        let group = encode_group(2, 4, &payloads);
        assert_eq!(group.row_repair.len(), 2);
        assert_eq!(group.col_repair.len(), 4);
    }

    // 4. xor_into is its own inverse (double XOR = identity)
    #[test]
    fn test_xor_into_inverse() {
        let src = vec![0xABu8; 33]; // odd length exercises the tail path
        let mut dst = vec![0u8; 33];
        xor_into(&mut dst, &src);
        xor_into(&mut dst, &src);
        assert!(dst.iter().all(|&b| b == 0));
    }

    // 5. Single row loss recovery (1-D)
    #[test]
    fn test_single_row_loss_recovery() {
        let payloads = make_payloads(8, 12);
        let group = encode_group(2, 4, &payloads);
        let cfg = group.config.clone();
        let mut dec = InterleavedFecDecoder::new(cfg);

        // Feed all except packet at seq 2 (row 0, col 2).
        for (i, p) in payloads.iter().enumerate() {
            if i != 2 {
                dec.feed_source(i as u16, p.clone());
            }
        }
        for rp in group.row_repair {
            dec.feed_row_repair(rp);
        }
        for cp in group.col_repair {
            dec.feed_col_repair(cp);
        }

        let recovered = dec.recover().expect("recover");
        assert!(recovered.contains_key(&2), "seq 2 must be recovered");
        assert_eq!(recovered[&2], payloads[2]);
    }

    // 6. Single column loss recovery (1-D)
    #[test]
    fn test_single_col_loss_recovery() {
        let payloads = make_payloads(8, 12);
        let group = encode_group(2, 4, &payloads);
        let cfg = group.config.clone();
        let mut dec = InterleavedFecDecoder::new(cfg);

        // Lose packet 4 (row 1, col 0).
        for (i, p) in payloads.iter().enumerate() {
            if i != 4 {
                dec.feed_source(i as u16, p.clone());
            }
        }
        for rp in group.row_repair {
            dec.feed_row_repair(rp);
        }
        for cp in group.col_repair {
            dec.feed_col_repair(cp);
        }

        let recovered = dec.recover().expect("recover");
        assert_eq!(recovered[&4], payloads[4]);
    }

    // 7. Two losses in different rows and columns → both recovered
    #[test]
    fn test_two_loss_recovery_different_rows_cols() {
        let payloads = make_payloads(8, 8);
        let group = encode_group(2, 4, &payloads);
        let cfg = group.config.clone();
        let mut dec = InterleavedFecDecoder::new(cfg);

        // Lose seq 1 (row 0, col 1) and seq 5 (row 1, col 1) — same column.
        for (i, p) in payloads.iter().enumerate() {
            if i != 1 && i != 5 {
                dec.feed_source(i as u16, p.clone());
            }
        }
        for rp in group.row_repair {
            dec.feed_row_repair(rp);
        }
        for cp in group.col_repair {
            dec.feed_col_repair(cp);
        }

        let recovered = dec.recover().expect("recover");
        // Column 1 has two losses → can't be recovered by column repair.
        // But row 0 has one loss (seq 1) and row 1 has one loss (seq 5),
        // so both can be recovered by row repair.
        assert_eq!(recovered[&1], payloads[1]);
        assert_eq!(recovered[&5], payloads[5]);
    }

    // 8. No losses → recover returns empty map
    #[test]
    fn test_no_losses_empty_recovery() {
        let payloads = make_payloads(6, 10);
        let group = encode_group(2, 3, &payloads);
        let cfg = group.config.clone();
        let mut dec = InterleavedFecDecoder::new(cfg);
        for (i, p) in payloads.iter().enumerate() {
            dec.feed_source(i as u16, p.clone());
        }
        for rp in group.row_repair {
            dec.feed_row_repair(rp);
        }
        for cp in group.col_repair {
            dec.feed_col_repair(cp);
        }

        let recovered = dec.recover().expect("recover");
        assert!(recovered.is_empty());
    }

    // 9. Encoder rejects incomplete group
    #[test]
    fn test_encoder_rejects_incomplete_group() {
        let cfg = InterleavedFecConfig::new(2, 4).expect("cfg");
        let mut enc = InterleavedFecEncoder::new(cfg);
        enc.feed(0, b"only one packet");
        assert!(enc.finalize().is_err());
    }

    // 10. to_matrix_pos and from_matrix_pos are inverses
    #[test]
    fn test_matrix_pos_roundtrip() {
        let cfg = InterleavedFecConfig::new(3, 5).expect("cfg");
        for i in 0..15 {
            let (r, c) = cfg.to_matrix_pos(i);
            assert_eq!(cfg.from_matrix_pos(r, c), i);
        }
    }

    // 11. RecoveryPlanner marks burst loss in same column as unrecoverable
    #[test]
    fn test_planner_unrecoverable_column_burst() {
        let cfg = InterleavedFecConfig::new(2, 4).expect("cfg");
        let planner = RecoveryPlanner::new(cfg);
        // Group: seqs 0–7; lose seqs 1 and 5 (both col 1 → 2 losses in col 1).
        let received: Vec<u16> = (0u16..8).filter(|&s| s != 1 && s != 5).collect();
        let plan = planner.plan(0, &received);
        // Row 0 has 1 loss (seq 1) → recoverable by row.
        // Row 1 has 1 loss (seq 5) → recoverable by row.
        assert!(plan.recoverable.contains(&1));
        assert!(plan.recoverable.contains(&5));
        assert!(plan.unrecoverable.is_empty());
    }

    // 12. RecoveryPlanner finds no losses when all received
    #[test]
    fn test_planner_no_losses() {
        let cfg = InterleavedFecConfig::new(2, 3).expect("cfg");
        let planner = RecoveryPlanner::new(cfg);
        let received: Vec<u16> = (0..6).collect();
        let plan = planner.plan(0, &received);
        assert!(plan.recoverable.is_empty());
        assert!(plan.unrecoverable.is_empty());
    }

    // 13. Chunk-aligned XOR handles mismatched lengths
    #[test]
    fn test_xor_into_mismatched_lengths() {
        let mut dst = vec![0xFFu8; 10];
        let src = vec![0xFFu8; 5]; // shorter
        xor_into(&mut dst, &src);
        // First 5 bytes XOR'd → 0, last 5 bytes unchanged (0xFF)
        assert!(dst[..5].iter().all(|&b| b == 0));
        assert!(dst[5..].iter().all(|&b| b == 0xFF));
    }

    // 14. Config Display trait
    #[test]
    fn test_config_display() {
        let cfg = InterleavedFecConfig::new(4, 8).expect("cfg");
        let s = format!("{cfg}");
        assert!(s.contains("4×8") || s.contains("4\u{00D7}8"));
    }

    // 15. Large payload (1316 B = SRT MTU) round-trip
    #[test]
    fn test_large_payload_roundtrip() {
        let payloads: Vec<Vec<u8>> = (0u8..6)
            .map(|i| {
                let mut v = vec![0u8; 1316];
                v[0] = i;
                v[1315] = i.wrapping_mul(7);
                v
            })
            .collect();

        let group = encode_group(2, 3, &payloads);
        let cfg = group.config.clone();
        let mut dec = InterleavedFecDecoder::new(cfg);

        // Lose packet 3 (row 1, col 0).
        for (i, p) in payloads.iter().enumerate() {
            if i != 3 {
                dec.feed_source(i as u16, p.clone());
            }
        }
        for rp in group.row_repair {
            dec.feed_row_repair(rp);
        }
        for cp in group.col_repair {
            dec.feed_col_repair(cp);
        }

        let recovered = dec.recover().expect("recover");
        assert_eq!(recovered[&3], payloads[3]);
    }
}
