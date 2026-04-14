//! Forward Error Correction (FEC) using Reed-Solomon codes.
//!
//! This module includes an `AdaptiveFecController` that measures packet loss
//! over a sliding window and automatically adjusts the parity-to-data ratio,
//! keeping overhead low during good network conditions and ramping up
//! protection during congestion.

use crate::error::{VideoIpError, VideoIpResult};
use crate::packet::{Packet, PacketBuilder, PacketFlags};
use crate::types::StreamType;
use bytes::Bytes;
use reed_solomon_erasure::galois_8::ReedSolomon;
use std::collections::HashMap;

/// FEC encoder for creating parity packets.
#[allow(dead_code)]
pub struct FecEncoder {
    /// Reed-Solomon encoder.
    encoder: ReedSolomon,
    /// Number of data packets in each FEC group.
    data_shards: usize,
    /// Number of parity packets in each FEC group.
    parity_shards: usize,
    /// Maximum packet size.
    max_packet_size: usize,
}

impl FecEncoder {
    /// Creates a new FEC encoder.
    ///
    /// # Arguments
    ///
    /// * `data_shards` - Number of data packets in each FEC group (typically 20-50)
    /// * `parity_shards` - Number of parity packets in each FEC group (typically 1-10)
    ///
    /// # Errors
    ///
    /// Returns an error if the shard configuration is invalid.
    pub fn new(data_shards: usize, parity_shards: usize) -> VideoIpResult<Self> {
        let encoder = ReedSolomon::new(data_shards, parity_shards)
            .map_err(|e| VideoIpError::Fec(format!("failed to create encoder: {e}")))?;

        Ok(Self {
            encoder,
            data_shards,
            parity_shards,
            max_packet_size: 8192,
        })
    }

    /// Creates an FEC encoder with a specific FEC ratio.
    ///
    /// # Arguments
    ///
    /// * `ratio` - FEC ratio (0.05 = 5%, 0.10 = 10%, etc.)
    ///
    /// # Errors
    ///
    /// Returns an error if the ratio is invalid.
    pub fn with_ratio(ratio: f32) -> VideoIpResult<Self> {
        if !(0.01..=0.5).contains(&ratio) {
            return Err(VideoIpError::Fec(format!(
                "invalid FEC ratio: {ratio} (must be between 0.01 and 0.5)"
            )));
        }

        let data_shards = 20;
        let parity_shards = ((data_shards as f32) * ratio).ceil() as usize;
        Self::new(data_shards, parity_shards)
    }

    /// Encodes a group of data packets and generates parity packets.
    ///
    /// # Errors
    ///
    /// Returns an error if encoding fails.
    pub fn encode(
        &self,
        packets: &[Packet],
        base_sequence: u16,
        timestamp: u64,
        stream_type: StreamType,
    ) -> VideoIpResult<Vec<Packet>> {
        if packets.is_empty() || packets.len() > self.data_shards {
            return Err(VideoIpError::Fec(format!(
                "invalid packet count: {} (expected 1-{})",
                packets.len(),
                self.data_shards
            )));
        }

        // Find the maximum packet size in this group
        let max_size = packets.iter().map(|p| p.payload.len()).max().unwrap_or(0);

        // Pad all packets to the same size
        let mut shards: Vec<Vec<u8>> = packets
            .iter()
            .map(|p| {
                let mut data = p.payload.to_vec();
                data.resize(max_size, 0);
                data
            })
            .collect();

        // Add empty shards up to data_shards count
        while shards.len() < self.data_shards {
            shards.push(vec![0u8; max_size]);
        }

        // Add parity shards
        for _ in 0..self.parity_shards {
            shards.push(vec![0u8; max_size]);
        }

        // Encode
        self.encoder
            .encode(&mut shards)
            .map_err(|e| VideoIpError::Fec(format!("encoding failed: {e}")))?;

        // Create parity packets from the parity shards
        let mut parity_packets = Vec::with_capacity(self.parity_shards);
        for (i, shard) in shards[self.data_shards..].iter().enumerate() {
            let sequence = base_sequence.wrapping_add(i as u16);
            let payload = Bytes::from(shard.clone());

            let packet = PacketBuilder::new(sequence)
                .fec()
                .with_timestamp(timestamp)
                .with_stream_type(stream_type)
                .build(payload)?;

            parity_packets.push(packet);
        }

        Ok(parity_packets)
    }

    /// Returns the number of data shards.
    #[must_use]
    pub const fn data_shards(&self) -> usize {
        self.data_shards
    }

    /// Returns the number of parity shards.
    #[must_use]
    pub const fn parity_shards(&self) -> usize {
        self.parity_shards
    }
}

/// FEC decoder for recovering lost packets.
pub struct FecDecoder {
    /// Reed-Solomon decoder.
    decoder: ReedSolomon,
    /// Number of data packets in each FEC group.
    data_shards: usize,
    /// Number of parity packets in each FEC group.
    parity_shards: usize,
    /// Pending FEC groups waiting for completion.
    pending_groups: HashMap<u16, FecGroup>,
}

/// A group of packets for FEC decoding.
struct FecGroup {
    /// Data packets (Some if received, None if missing).
    data_packets: Vec<Option<Packet>>,
    /// Parity packets (Some if received, None if missing).
    parity_packets: Vec<Option<Packet>>,
    /// Maximum packet size in this group.
    max_packet_size: usize,
    /// Timestamp of the group.
    timestamp: u64,
}

impl FecDecoder {
    /// Creates a new FEC decoder.
    ///
    /// # Errors
    ///
    /// Returns an error if the shard configuration is invalid.
    pub fn new(data_shards: usize, parity_shards: usize) -> VideoIpResult<Self> {
        let decoder = ReedSolomon::new(data_shards, parity_shards)
            .map_err(|e| VideoIpError::Fec(format!("failed to create decoder: {e}")))?;

        Ok(Self {
            decoder,
            data_shards,
            parity_shards,
            pending_groups: HashMap::new(),
        })
    }

    /// Adds a packet to the decoder.
    ///
    /// Returns recovered packets if FEC decoding was successful.
    ///
    /// # Errors
    ///
    /// Returns an error if decoding fails.
    pub fn add_packet(&mut self, packet: Packet) -> VideoIpResult<Vec<Packet>> {
        let group_id = self.get_group_id(packet.header.sequence);

        // Calculate indices before acquiring mutable borrow
        let is_parity = packet.header.flags.contains(PacketFlags::FEC);
        let parity_idx = if is_parity {
            Some(self.get_parity_index(packet.header.sequence))
        } else {
            None
        };
        let data_idx = if is_parity {
            None
        } else {
            Some(self.get_data_index(packet.header.sequence))
        };

        let group = self
            .pending_groups
            .entry(group_id)
            .or_insert_with(|| FecGroup {
                data_packets: vec![None; self.data_shards],
                parity_packets: vec![None; self.parity_shards],
                max_packet_size: 0,
                timestamp: packet.header.timestamp,
            });

        group.max_packet_size = group.max_packet_size.max(packet.payload.len());

        if is_parity {
            // This is a parity packet
            if let Some(idx) = parity_idx {
                if idx < self.parity_shards {
                    group.parity_packets[idx] = Some(packet);
                }
            }
        } else {
            // This is a data packet
            if let Some(idx) = data_idx {
                if idx < self.data_shards {
                    group.data_packets[idx] = Some(packet);
                }
            }
        }

        // Try to recover if we have enough packets
        self.try_recover(group_id)
    }
    /// Attempts to recover missing packets in a group.
    fn try_recover(&mut self, group_id: u16) -> VideoIpResult<Vec<Packet>> {
        let group = match self.pending_groups.get_mut(&group_id) {
            Some(g) => g,
            None => return Ok(Vec::new()),
        };

        let data_count = group.data_packets.iter().filter(|p| p.is_some()).count();
        let parity_count = group.parity_packets.iter().filter(|p| p.is_some()).count();
        let total_count = data_count + parity_count;

        // We need at least data_shards packets to recover
        if total_count < self.data_shards {
            return Ok(Vec::new());
        }

        // Build shards for decoding
        let mut shards: Vec<Option<Vec<u8>>> = Vec::new();
        for packet in &group.data_packets {
            shards.push(packet.as_ref().map(|p| {
                let mut data = p.payload.to_vec();
                data.resize(group.max_packet_size, 0);
                data
            }));
        }
        for packet in &group.parity_packets {
            shards.push(packet.as_ref().map(|p| {
                let mut data = p.payload.to_vec();
                data.resize(group.max_packet_size, 0);
                data
            }));
        }

        // Decode
        self.decoder
            .reconstruct(&mut shards)
            .map_err(|e| VideoIpError::Fec(format!("reconstruction failed: {e}")))?;

        // Extract recovered packets
        let mut recovered = Vec::new();
        for (i, shard) in shards[..self.data_shards].iter().enumerate() {
            if group.data_packets[i].is_none() {
                if let Some(data) = shard {
                    let sequence = group_id.wrapping_add(i as u16);
                    let payload = Bytes::from(data.clone());

                    let packet = PacketBuilder::new(sequence)
                        .video() // Assume video for now
                        .with_timestamp(group.timestamp)
                        .build(payload)?;

                    recovered.push(packet);
                }
            }
        }

        // Clean up the group
        self.pending_groups.remove(&group_id);

        Ok(recovered)
    }

    /// Gets the group ID for a sequence number.
    fn get_group_id(&self, sequence: u16) -> u16 {
        let total_shards = (self.data_shards + self.parity_shards) as u16;
        (sequence / total_shards) * total_shards
    }

    /// Gets the data packet index within a group.
    fn get_data_index(&self, sequence: u16) -> usize {
        let group_id = self.get_group_id(sequence);
        (sequence.wrapping_sub(group_id)) as usize
    }

    /// Gets the parity packet index within a group.
    fn get_parity_index(&self, sequence: u16) -> usize {
        let group_id = self.get_group_id(sequence);
        let offset = sequence.wrapping_sub(group_id) as usize;
        offset.saturating_sub(self.data_shards)
    }

    /// Cleans up old pending groups.
    pub fn cleanup_old_groups(&mut self, max_age_ms: u64) {
        let now = crate::packet::current_timestamp_micros();
        self.pending_groups
            .retain(|_, group| now - group.timestamp < max_age_ms * 1000);
    }
}

// ── AdaptiveFecController ─────────────────────────────────────────────────────

/// Configuration for the adaptive FEC rate controller.
#[derive(Debug, Clone)]
pub struct AdaptiveFecConfig {
    /// Minimum parity ratio (parity / data shards), e.g. 0.05 = 5 %.
    pub min_ratio: f32,
    /// Maximum parity ratio, e.g. 0.40 = 40 %.
    pub max_ratio: f32,
    /// Number of data shards to use (fixed; only parity count adapts).
    pub data_shards: usize,
    /// Sliding window size (in packet reports) for loss-rate measurement.
    pub window_size: usize,
    /// Packet-loss rate threshold above which the FEC ratio steps up.
    pub loss_step_up_threshold: f64,
    /// Packet-loss rate threshold below which the FEC ratio steps down.
    pub loss_step_down_threshold: f64,
    /// Amount to increase the parity ratio per adaptation cycle.
    pub ratio_step_up: f32,
    /// Amount to decrease the parity ratio per adaptation cycle.
    pub ratio_step_down: f32,
    /// Number of consecutive stable cycles before stepping down.
    pub stable_cycles_before_step_down: u32,
}

impl Default for AdaptiveFecConfig {
    fn default() -> Self {
        Self {
            min_ratio: 0.05,
            max_ratio: 0.40,
            data_shards: 20,
            window_size: 200,
            loss_step_up_threshold: 0.01,    // 1 % loss → step up
            loss_step_down_threshold: 0.001, // < 0.1 % loss → step down
            ratio_step_up: 0.05,
            ratio_step_down: 0.02,
            stable_cycles_before_step_down: 5,
        }
    }
}

/// Statistics reported by the [`AdaptiveFecController`].
#[derive(Debug, Clone, Default)]
pub struct AdaptiveFecStats {
    /// Current parity ratio in use.
    pub current_ratio: f32,
    /// Current parity shard count derived from the ratio.
    pub current_parity_shards: usize,
    /// Measured sliding-window packet-loss rate (0.0–1.0).
    pub loss_rate: f64,
    /// Total adaptation up-steps taken.
    pub steps_up: u64,
    /// Total adaptation down-steps taken.
    pub steps_down: u64,
    /// Consecutive stable adaptation cycles.
    pub stable_cycles: u32,
}

/// Adaptive FEC rate controller.
///
/// Observes a sliding window of packet delivery outcomes and adjusts the
/// parity-to-data shard ratio automatically:
///
/// - When measured loss rate exceeds `loss_step_up_threshold`, the ratio is
///   increased by `ratio_step_up` (up to `max_ratio`).
/// - When loss is consistently below `loss_step_down_threshold` for
///   `stable_cycles_before_step_down` cycles, the ratio is decreased by
///   `ratio_step_down` (down to `min_ratio`).
///
/// Call [`record_outcome`](Self::record_outcome) for each packet (received =
/// `true`, lost = `false`), then call [`adapt`](Self::adapt) periodically
/// (e.g. once per FEC group) to obtain a fresh [`FecEncoder`].
pub struct AdaptiveFecController {
    /// Configuration (public so callers can read data_shards etc.).
    pub config: AdaptiveFecConfig,
    /// Current parity ratio.
    current_ratio: f32,
    /// Sliding window: `true` = packet received, `false` = packet lost.
    window: std::collections::VecDeque<bool>,
    /// Consecutive cycles with loss below the step-down threshold.
    stable_cycles: u32,
    /// Statistics.
    stats: AdaptiveFecStats,
}

impl AdaptiveFecController {
    /// Creates a new adaptive FEC controller with the given configuration.
    ///
    /// The controller starts at the midpoint between `min_ratio` and
    /// `max_ratio` so that it can react quickly in both directions.
    #[must_use]
    pub fn new(config: AdaptiveFecConfig) -> Self {
        let initial_ratio = (config.min_ratio + config.max_ratio) / 2.0;
        let parity = Self::ratio_to_parity(initial_ratio, config.data_shards);
        Self {
            current_ratio: initial_ratio,
            window: std::collections::VecDeque::with_capacity(config.window_size),
            stable_cycles: 0,
            stats: AdaptiveFecStats {
                current_ratio: initial_ratio,
                current_parity_shards: parity,
                ..Default::default()
            },
            config,
        }
    }

    /// Creates a controller with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(AdaptiveFecConfig::default())
    }

    /// Records the reception outcome for a single packet.
    ///
    /// `received` should be `true` when the packet arrived intact and `false`
    /// when a loss was detected (gap in sequence numbers).
    pub fn record_outcome(&mut self, received: bool) {
        self.window.push_back(received);
        if self.window.len() > self.config.window_size {
            self.window.pop_front();
        }
    }

    /// Records multiple outcomes at once (convenience wrapper).
    ///
    /// `total` is the number of packets expected; `lost` is the number not
    /// received.  The remaining `total - lost` are recorded as received.
    pub fn record_batch(&mut self, total: usize, lost: usize) {
        let received = total.saturating_sub(lost);
        for _ in 0..received {
            self.record_outcome(true);
        }
        for _ in 0..lost {
            self.record_outcome(false);
        }
    }

    /// Runs one adaptation cycle and returns the current FEC ratio.
    ///
    /// The returned ratio can be used to construct a fresh [`FecEncoder`] via
    /// [`FecEncoder::with_ratio`].  Typically called once per FEC group
    /// (every `data_shards` packets).
    pub fn adapt(&mut self) -> f32 {
        let loss_rate = self.measured_loss_rate();
        self.stats.loss_rate = loss_rate;

        if loss_rate > self.config.loss_step_up_threshold {
            self.current_ratio =
                (self.current_ratio + self.config.ratio_step_up).min(self.config.max_ratio);
            self.stable_cycles = 0;
            self.stats.steps_up += 1;
        } else if loss_rate < self.config.loss_step_down_threshold {
            self.stable_cycles += 1;
            if self.stable_cycles >= self.config.stable_cycles_before_step_down {
                self.current_ratio =
                    (self.current_ratio - self.config.ratio_step_down).max(self.config.min_ratio);
                self.stable_cycles = 0;
                self.stats.steps_down += 1;
            }
        } else {
            self.stable_cycles = 0;
        }

        let parity = Self::ratio_to_parity(self.current_ratio, self.config.data_shards);
        self.stats.current_ratio = self.current_ratio;
        self.stats.current_parity_shards = parity;
        self.stats.stable_cycles = self.stable_cycles;

        self.current_ratio
    }

    /// Builds a [`FecEncoder`] with the *current* adaptive ratio.
    ///
    /// # Errors
    ///
    /// Returns an error if the Reed-Solomon encoder cannot be constructed.
    pub fn build_encoder(&self) -> VideoIpResult<FecEncoder> {
        let parity = Self::ratio_to_parity(self.current_ratio, self.config.data_shards).max(1);
        FecEncoder::new(self.config.data_shards, parity)
    }

    /// Returns the current parity ratio (0.0–1.0).
    #[must_use]
    pub fn current_ratio(&self) -> f32 {
        self.current_ratio
    }

    /// Returns the current parity shard count (always ≥ 1).
    #[must_use]
    pub fn current_parity_shards(&self) -> usize {
        Self::ratio_to_parity(self.current_ratio, self.config.data_shards).max(1)
    }

    /// Returns the measured sliding-window packet-loss rate (0.0–1.0).
    #[must_use]
    pub fn measured_loss_rate(&self) -> f64 {
        if self.window.is_empty() {
            return 0.0;
        }
        let lost = self.window.iter().filter(|&&r| !r).count();
        lost as f64 / self.window.len() as f64
    }

    /// Returns a snapshot of the controller statistics.
    #[must_use]
    pub fn stats(&self) -> &AdaptiveFecStats {
        &self.stats
    }

    /// Resets the sliding window and statistics without changing configuration
    /// or the current ratio.
    pub fn reset(&mut self) {
        self.window.clear();
        self.stable_cycles = 0;
        let parity = Self::ratio_to_parity(self.current_ratio, self.config.data_shards);
        self.stats = AdaptiveFecStats {
            current_ratio: self.current_ratio,
            current_parity_shards: parity,
            ..Default::default()
        };
    }

    /// Converts a ratio to a parity shard count (rounded up, clamped to ≥ 1).
    fn ratio_to_parity(ratio: f32, data_shards: usize) -> usize {
        ((data_shards as f32 * ratio).ceil() as usize).max(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fec_encoder_creation() {
        let encoder = FecEncoder::new(20, 2).expect("should succeed in test");
        assert_eq!(encoder.data_shards(), 20);
        assert_eq!(encoder.parity_shards(), 2);
    }

    #[test]
    fn test_fec_encoder_with_ratio() {
        let encoder = FecEncoder::with_ratio(0.1).expect("should succeed in test");
        assert_eq!(encoder.data_shards(), 20);
        assert_eq!(encoder.parity_shards(), 2);
    }

    #[test]
    fn test_fec_invalid_ratio() {
        assert!(FecEncoder::with_ratio(0.0).is_err());
        assert!(FecEncoder::with_ratio(0.6).is_err());
    }

    #[test]
    fn test_fec_encode() {
        let encoder = FecEncoder::new(10, 2).expect("should succeed in test");

        let packets: Vec<Packet> = (0..10)
            .map(|i| {
                PacketBuilder::new(i)
                    .video()
                    .with_timestamp(12345)
                    .build(Bytes::from(vec![i as u8; 100]))
                    .expect("should succeed in test")
            })
            .collect();

        let parity = encoder
            .encode(&packets, 100, 12345, StreamType::Program)
            .expect("should succeed in test");

        assert_eq!(parity.len(), 2);
        assert!(parity[0].header.flags.contains(PacketFlags::FEC));
    }

    #[test]
    fn test_fec_decoder_creation() {
        let decoder = FecDecoder::new(20, 2).expect("should succeed in test");
        assert_eq!(decoder.data_shards, 20);
        assert_eq!(decoder.parity_shards, 2);
    }

    #[test]
    fn test_fec_recovery() {
        let encoder = FecEncoder::new(5, 2).expect("should succeed in test");

        // Create 5 data packets
        let packets: Vec<Packet> = (0..5)
            .map(|i| {
                PacketBuilder::new(i)
                    .video()
                    .with_timestamp(12345)
                    .build(Bytes::from(vec![i as u8; 50]))
                    .expect("should succeed in test")
            })
            .collect();

        // Generate parity packets
        let parity = encoder
            .encode(&packets, 5, 12345, StreamType::Program)
            .expect("should succeed in test");

        // Create decoder and add 4 data packets + 2 parity packets (missing packet 2)
        let mut decoder = FecDecoder::new(5, 2).expect("should succeed in test");
        let mut all_recovered = Vec::new();
        all_recovered.extend(
            decoder
                .add_packet(packets[0].clone())
                .expect("should succeed in test"),
        );
        all_recovered.extend(
            decoder
                .add_packet(packets[1].clone())
                .expect("should succeed in test"),
        );
        // Skip packet 2
        all_recovered.extend(
            decoder
                .add_packet(packets[3].clone())
                .expect("should succeed in test"),
        );
        all_recovered.extend(
            decoder
                .add_packet(packets[4].clone())
                .expect("should succeed in test"),
        );
        all_recovered.extend(
            decoder
                .add_packet(parity[0].clone())
                .expect("should succeed in test"),
        );
        all_recovered.extend(
            decoder
                .add_packet(parity[1].clone())
                .expect("should succeed in test"),
        );

        // Should recover the missing packet
        assert!(!all_recovered.is_empty());
    }

    #[test]
    fn test_group_id_calculation() {
        let decoder = FecDecoder::new(20, 2).expect("should succeed in test");
        assert_eq!(decoder.get_group_id(0), 0);
        assert_eq!(decoder.get_group_id(21), 0);
        assert_eq!(decoder.get_group_id(22), 22);
        assert_eq!(decoder.get_group_id(43), 22);
    }

    #[test]
    fn test_cleanup_old_groups() {
        let mut decoder = FecDecoder::new(10, 2).expect("should succeed in test");

        let packet = PacketBuilder::new(0)
            .video()
            .with_timestamp(0)
            .build(Bytes::from_static(b"test"))
            .expect("should succeed in test");

        decoder.add_packet(packet).expect("should succeed in test");
        assert_eq!(decoder.pending_groups.len(), 1);

        decoder.cleanup_old_groups(0);
        assert_eq!(decoder.pending_groups.len(), 0);
    }

    // ── AdaptiveFecController tests ───────────────────────────────────────────

    #[test]
    fn test_adaptive_fec_initial_ratio_is_midpoint() {
        let config = AdaptiveFecConfig {
            min_ratio: 0.05,
            max_ratio: 0.25,
            ..Default::default()
        };
        let ctrl = AdaptiveFecController::new(config);
        let expected = 0.15_f32;
        assert!(
            (ctrl.current_ratio() - expected).abs() < 1e-4,
            "initial ratio {:.4} ≠ {expected:.4}",
            ctrl.current_ratio()
        );
    }

    #[test]
    fn test_adaptive_fec_steps_up_on_high_loss() {
        let config = AdaptiveFecConfig {
            min_ratio: 0.05,
            max_ratio: 0.40,
            loss_step_up_threshold: 0.01,
            ratio_step_up: 0.05,
            window_size: 100,
            ..Default::default()
        };
        let mut ctrl = AdaptiveFecController::new(config);
        let initial = ctrl.current_ratio();
        // Inject 10 % loss — well above the 1 % threshold.
        ctrl.record_batch(100, 10);
        let new_ratio = ctrl.adapt();
        assert!(
            new_ratio > initial,
            "ratio should increase on high loss: {new_ratio:.4} > {initial:.4}"
        );
        assert_eq!(ctrl.stats().steps_up, 1);
    }

    #[test]
    fn test_adaptive_fec_steps_down_after_stable_cycles() {
        // Use a tiny window so the old loss data ages out quickly.
        let config = AdaptiveFecConfig {
            min_ratio: 0.05,
            max_ratio: 0.40,
            loss_step_down_threshold: 0.001,
            ratio_step_down: 0.02,
            stable_cycles_before_step_down: 3,
            window_size: 10, // small: 10 new packets flush old loss data
            ..Default::default()
        };
        let mut ctrl = AdaptiveFecController::new(config);
        // Force ratio up: push some high-loss data and adapt.
        ctrl.record_batch(10, 5);
        ctrl.adapt();
        let after_up = ctrl.current_ratio();
        // Now flush the window with zero-loss data (more than window_size packets).
        // Then run 3 stable adapt cycles.
        for _ in 0..3 {
            ctrl.record_batch(10, 0); // overwrites the old loss data
            ctrl.adapt();
        }
        let after_down = ctrl.current_ratio();
        assert!(
            after_down < after_up,
            "ratio should decrease after stable cycles: {after_down:.4} < {after_up:.4}"
        );
        assert!(ctrl.stats().steps_down >= 1);
    }

    #[test]
    fn test_adaptive_fec_clamped_at_max() {
        let config = AdaptiveFecConfig {
            min_ratio: 0.05,
            max_ratio: 0.20,
            loss_step_up_threshold: 0.001,
            ratio_step_up: 0.10,
            window_size: 50,
            ..Default::default()
        };
        let mut ctrl = AdaptiveFecController::new(config);
        for _ in 0..20 {
            ctrl.record_batch(100, 50);
            ctrl.adapt();
        }
        assert!(
            ctrl.current_ratio() <= 0.20 + f32::EPSILON,
            "ratio must not exceed max_ratio, got {}",
            ctrl.current_ratio()
        );
    }

    #[test]
    fn test_adaptive_fec_clamped_at_min() {
        let config = AdaptiveFecConfig {
            min_ratio: 0.05,
            max_ratio: 0.40,
            loss_step_down_threshold: 1.0, // always stable
            ratio_step_down: 0.10,
            stable_cycles_before_step_down: 1,
            window_size: 50,
            ..Default::default()
        };
        let mut ctrl = AdaptiveFecController::new(config);
        for _ in 0..50 {
            ctrl.record_batch(100, 0);
            ctrl.adapt();
        }
        assert!(
            ctrl.current_ratio() >= 0.05 - f32::EPSILON,
            "ratio must not go below min_ratio, got {}",
            ctrl.current_ratio()
        );
    }

    #[test]
    fn test_adaptive_fec_build_encoder_succeeds() {
        let ctrl = AdaptiveFecController::with_defaults();
        let encoder = ctrl.build_encoder().expect("build_encoder should succeed");
        assert!(encoder.parity_shards() >= 1);
        assert_eq!(encoder.data_shards(), ctrl.config.data_shards);
    }

    #[test]
    fn test_adaptive_fec_measured_loss_rate_zero_on_empty() {
        let ctrl = AdaptiveFecController::with_defaults();
        assert!(
            ctrl.measured_loss_rate().abs() < f64::EPSILON,
            "empty window should report 0 % loss"
        );
    }

    #[test]
    fn test_adaptive_fec_measured_loss_rate_correct() {
        let mut ctrl = AdaptiveFecController::with_defaults();
        ctrl.record_batch(20, 10);
        let rate = ctrl.measured_loss_rate();
        assert!(
            (rate - 0.5).abs() < 0.01,
            "expected ~50 % loss, got {:.2} %",
            rate * 100.0
        );
    }

    #[test]
    fn test_adaptive_fec_reset_clears_window() {
        let mut ctrl = AdaptiveFecController::with_defaults();
        ctrl.record_batch(100, 10);
        ctrl.adapt();
        ctrl.reset();
        assert!(ctrl.measured_loss_rate().abs() < f64::EPSILON);
        assert_eq!(ctrl.stats().steps_up, 0);
    }

    #[test]
    fn test_adaptive_fec_stats_reflects_parity_shards() {
        let mut ctrl = AdaptiveFecController::with_defaults();
        ctrl.record_batch(100, 20);
        ctrl.adapt();
        let stats = ctrl.stats();
        let expected =
            ((stats.current_ratio * ctrl.config.data_shards as f32).ceil() as usize).max(1);
        assert_eq!(stats.current_parity_shards, expected);
    }
}
