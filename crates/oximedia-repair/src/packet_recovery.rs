//! RTP/UDP packet-loss detection and recovery using Forward Error Correction.
//!
//! Implements a simple sliding-window recovery buffer and a Reed-Solomon-like
//! FEC scheme (XOR-based for simplicity) that lets a receiver recover single
//! lost packets in a protection group.

#![allow(dead_code)]

/// Describes a packet-loss event in the receive stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PacketLoss {
    /// Sequence numbers of packets that were never received.
    pub missing_seq: Vec<u32>,
    /// Total number of packets expected in the observation window.
    pub expected: usize,
    /// Window start sequence number.
    pub window_start: u32,
}

impl PacketLoss {
    /// Create a new packet-loss record.
    pub fn new(window_start: u32, expected: usize, missing_seq: Vec<u32>) -> Self {
        Self {
            missing_seq,
            expected,
            window_start,
        }
    }

    /// Loss rate in [0.0, 1.0].
    #[allow(clippy::cast_precision_loss)]
    pub fn loss_rate(&self) -> f64 {
        if self.expected == 0 {
            return 0.0;
        }
        self.missing_seq.len() as f64 / self.expected as f64
    }

    /// Returns `true` if there were no missing packets.
    pub fn is_perfect(&self) -> bool {
        self.missing_seq.is_empty()
    }

    /// Number of missing packets.
    pub fn missing_count(&self) -> usize {
        self.missing_seq.len()
    }
}

/// A single buffered packet entry.
#[derive(Debug, Clone)]
struct BufferEntry {
    seq: u32,
    payload: Vec<u8>,
}

/// Sliding-window receive buffer that tracks which packets arrived and which
/// are missing, and provides best-effort recovery via FEC data.
#[derive(Debug, Clone)]
pub struct RecoveryBuffer {
    window_size: usize,
    entries: Vec<Option<BufferEntry>>,
    base_seq: u32,
    fec_data: Option<Vec<u8>>,
    /// Sequence numbers covered by the current FEC block.
    fec_seqs: Vec<u32>,
}

impl RecoveryBuffer {
    /// Create a new recovery buffer with the given window size.
    pub fn new(window_size: usize) -> Self {
        assert!(window_size > 0, "window_size must be > 0");
        Self {
            window_size,
            entries: vec![None; window_size],
            base_seq: 0,
            fec_data: None,
            fec_seqs: Vec::new(),
        }
    }

    /// Set the base sequence number for the current window.
    pub fn set_base_seq(&mut self, seq: u32) {
        self.base_seq = seq;
    }

    /// Add a received packet to the buffer. Returns `false` if the sequence
    /// number falls outside the current window.
    pub fn add_packet(&mut self, seq: u32, payload: Vec<u8>) -> bool {
        let offset = seq.wrapping_sub(self.base_seq) as usize;
        if offset >= self.window_size {
            return false;
        }
        self.entries[offset] = Some(BufferEntry { seq, payload });
        true
    }

    /// Store FEC (XOR) redundancy packet for the current window.
    pub fn store_fec(&mut self, fec_payload: Vec<u8>, covered_seqs: Vec<u32>) {
        self.fec_data = Some(fec_payload);
        self.fec_seqs = covered_seqs;
    }

    /// Attempt to recover missing packets in the current window.
    /// Returns the sequence numbers that were successfully recovered.
    pub fn recover_missing(&mut self) -> Vec<u32> {
        let missing_indices: Vec<usize> = (0..self.window_size)
            .filter(|&i| self.entries[i].is_none())
            .collect();

        if missing_indices.is_empty() {
            return Vec::new();
        }

        // XOR-based FEC can recover at most 1 missing packet per group.
        if missing_indices.len() > 1 {
            return Vec::new();
        }

        let fec_payload = match &self.fec_data {
            Some(d) => d.clone(),
            None => return Vec::new(),
        };

        let missing_idx = missing_indices[0];
        let missing_seq = self.base_seq.wrapping_add(missing_idx as u32);

        // Only attempt recovery if the missing seq is covered by the FEC group.
        if !self.fec_seqs.contains(&missing_seq) {
            return Vec::new();
        }

        // XOR all present packets with the FEC packet to recover the missing one.
        let mut recovered = fec_payload.clone();
        for (i, entry) in self.entries.iter().enumerate() {
            if i == missing_idx {
                continue;
            }
            if let Some(e) = entry {
                let payload = &e.payload;
                for (r, &b) in recovered.iter_mut().zip(payload.iter()) {
                    *r ^= b;
                }
            }
        }

        self.entries[missing_idx] = Some(BufferEntry {
            seq: missing_seq,
            payload: recovered,
        });

        vec![missing_seq]
    }

    /// Return a snapshot of the packet-loss statistics for the current window.
    pub fn loss_stats(&self) -> PacketLoss {
        let missing_seq: Vec<u32> = (0..self.window_size)
            .filter(|&i| self.entries[i].is_none())
            .map(|i| self.base_seq.wrapping_add(i as u32))
            .collect();

        PacketLoss::new(self.base_seq, self.window_size, missing_seq)
    }

    /// Clear the buffer and advance the base sequence by `window_size`.
    pub fn advance_window(&mut self) {
        self.base_seq = self.base_seq.wrapping_add(self.window_size as u32);
        for e in &mut self.entries {
            *e = None;
        }
        self.fec_data = None;
        self.fec_seqs.clear();
    }
}

impl Default for RecoveryBuffer {
    fn default() -> Self {
        Self::new(64)
    }
}

/// FEC encoder / decoder using simple XOR protection groups.
///
/// A protection group consists of N media packets and 1 FEC packet that is
/// the XOR of all N payloads (all padded to the same length).
#[derive(Debug, Clone)]
pub struct FecRecovery {
    group_size: usize,
}

impl FecRecovery {
    /// Create a new FEC engine with the given group size.
    pub fn new(group_size: usize) -> Self {
        assert!(group_size > 1, "group_size must be > 1");
        Self { group_size }
    }

    /// Encode redundancy: XOR all `packets` together (zero-padded to the
    /// longest packet's length) and return the FEC packet payload.
    pub fn encode_redundancy(&self, packets: &[Vec<u8>]) -> Vec<u8> {
        if packets.is_empty() {
            return Vec::new();
        }
        let max_len = packets.iter().map(|p| p.len()).max().unwrap_or(0);
        let mut fec = vec![0u8; max_len];
        for pkt in packets {
            for (f, &b) in fec.iter_mut().zip(pkt.iter()) {
                *f ^= b;
            }
        }
        fec
    }

    /// Attempt to recover a single missing packet given the remaining packets
    /// in the group and the FEC packet.
    ///
    /// Returns `Some(recovered_payload)` when exactly one packet is missing.
    pub fn attempt_recovery(&self, received: &[Vec<u8>], fec_packet: &[u8]) -> Option<Vec<u8>> {
        let expected_media = self.group_size - 1; // FEC occupies 1 slot.
        if received.len() != expected_media.saturating_sub(1) {
            // We need exactly (group_size - 2) received packets to recover 1.
            // For simplicity: if we have all but 1 media packet, recover it.
            if received.len() + 1 != self.group_size - 1 {
                return None;
            }
        }

        let max_len = received
            .iter()
            .map(|p| p.len())
            .max()
            .unwrap_or(0)
            .max(fec_packet.len());

        let mut recovered = fec_packet.to_vec();
        recovered.resize(max_len, 0);

        for pkt in received {
            for (r, &b) in recovered.iter_mut().zip(pkt.iter()) {
                *r ^= b;
            }
        }

        Some(recovered)
    }

    /// Return the group size.
    pub fn group_size(&self) -> usize {
        self.group_size
    }
}

impl Default for FecRecovery {
    fn default() -> Self {
        Self::new(5) // 4 media + 1 FEC
    }
}

// ── unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── PacketLoss tests ──────────────────────────────────────────────────────

    #[test]
    fn test_loss_rate_zero_expected() {
        let loss = PacketLoss::new(0, 0, vec![]);
        assert!((loss.loss_rate() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_loss_rate_25_percent() {
        let loss = PacketLoss::new(0, 8, vec![2, 6]);
        assert!((loss.loss_rate() - 0.25).abs() < 1e-9);
    }

    #[test]
    fn test_is_perfect() {
        let ok = PacketLoss::new(0, 4, vec![]);
        assert!(ok.is_perfect());
        let bad = PacketLoss::new(0, 4, vec![1]);
        assert!(!bad.is_perfect());
    }

    #[test]
    fn test_missing_count() {
        let loss = PacketLoss::new(100, 10, vec![101, 103, 107]);
        assert_eq!(loss.missing_count(), 3);
    }

    // ── RecoveryBuffer tests ──────────────────────────────────────────────────

    #[test]
    fn test_add_packet_in_window() {
        let mut buf = RecoveryBuffer::new(8);
        buf.set_base_seq(0);
        assert!(buf.add_packet(0, vec![1, 2, 3]));
        assert!(buf.add_packet(7, vec![4, 5, 6]));
    }

    #[test]
    fn test_add_packet_out_of_window() {
        let mut buf = RecoveryBuffer::new(4);
        buf.set_base_seq(10);
        assert!(!buf.add_packet(14, vec![0])); // offset 4 >= window_size
    }

    #[test]
    fn test_recover_missing_no_fec() {
        let mut buf = RecoveryBuffer::new(4);
        buf.set_base_seq(0);
        buf.add_packet(0, vec![1]);
        buf.add_packet(2, vec![3]);
        buf.add_packet(3, vec![4]);
        // No FEC stored → cannot recover.
        let recovered = buf.recover_missing();
        assert!(recovered.is_empty());
    }

    #[test]
    fn test_recover_missing_with_fec() {
        // Group: packets 0,1,2 with FEC = XOR of all three.
        let p0 = vec![0xAA, 0xBB];
        let p1 = vec![0xCC, 0xDD];
        let p2 = vec![0x11, 0x22];
        // fec = p0 ^ p1 ^ p2
        let fec: Vec<u8> = p0
            .iter()
            .zip(p1.iter())
            .map(|(&a, &b)| a ^ b)
            .zip(p2.iter())
            .map(|(ab, &c)| ab ^ c)
            .collect();

        let mut buf = RecoveryBuffer::new(3);
        buf.set_base_seq(0);
        buf.add_packet(0, p0.clone());
        // p1 (seq=1) is lost.
        buf.add_packet(2, p2.clone());
        buf.store_fec(fec.clone(), vec![0, 1, 2]);

        let recovered = buf.recover_missing();
        assert_eq!(recovered, vec![1]);

        // Verify the recovered payload equals p1.
        let entry = buf.entries[1].as_ref().expect("unexpected None/Err");
        assert_eq!(entry.payload, p1);
    }

    #[test]
    fn test_recover_missing_two_losses_fails() {
        let mut buf = RecoveryBuffer::new(4);
        buf.set_base_seq(0);
        buf.add_packet(0, vec![1]);
        buf.add_packet(3, vec![4]);
        buf.store_fec(vec![0xFF], vec![0, 1, 2, 3]);
        // Two packets missing → XOR can't handle it.
        let recovered = buf.recover_missing();
        assert!(recovered.is_empty());
    }

    #[test]
    fn test_loss_stats() {
        let mut buf = RecoveryBuffer::new(4);
        buf.set_base_seq(10);
        buf.add_packet(10, vec![]);
        buf.add_packet(12, vec![]);
        let stats = buf.loss_stats();
        assert_eq!(stats.expected, 4);
        assert_eq!(stats.missing_seq.len(), 2);
        assert!(stats.missing_seq.contains(&11));
        assert!(stats.missing_seq.contains(&13));
    }

    #[test]
    fn test_advance_window() {
        let mut buf = RecoveryBuffer::new(4);
        buf.set_base_seq(0);
        buf.add_packet(0, vec![1]);
        buf.advance_window();
        assert_eq!(buf.base_seq, 4);
        assert!(buf.entries.iter().all(|e| e.is_none()));
    }

    // ── FecRecovery tests ─────────────────────────────────────────────────────

    #[test]
    fn test_encode_redundancy_empty() {
        let fec = FecRecovery::new(3);
        assert!(fec.encode_redundancy(&[]).is_empty());
    }

    #[test]
    fn test_encode_redundancy_xor() {
        let fec = FecRecovery::new(3);
        let packets = vec![vec![0x01, 0x02], vec![0x03, 0x04]];
        let redundancy = fec.encode_redundancy(&packets);
        // 0x01 ^ 0x03 = 0x02, 0x02 ^ 0x04 = 0x06
        assert_eq!(redundancy, vec![0x02, 0x06]);
    }

    #[test]
    fn test_attempt_recovery_success() {
        // group_size = 3 means 2 media + 1 FEC.
        let fec = FecRecovery::new(3);
        let p0 = vec![0xAA, 0xBB];
        let p1 = vec![0x11, 0x22];
        let redundancy = fec.encode_redundancy(&[p0.clone(), p1.clone()]);
        // p1 is lost; we have p0 and the FEC.
        let recovered = fec.attempt_recovery(std::slice::from_ref(&p0), &redundancy);
        assert_eq!(recovered, Some(p1));
    }

    #[test]
    fn test_fec_group_size() {
        let fec = FecRecovery::new(8);
        assert_eq!(fec.group_size(), 8);
    }

    #[test]
    fn test_fec_default_group_size() {
        let fec = FecRecovery::default();
        assert_eq!(fec.group_size(), 5);
    }
}
