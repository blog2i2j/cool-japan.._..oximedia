//! NDI protocol enhancements: SpeedHQ intra-prediction helpers, extended tally
//! serialisation, VISCA-like PTZ binary encoding, audio channel metadata,
//! discovery cache with TTL, metadata frame injection, bandwidth estimation,
//! and multi-sender management.

#![allow(dead_code)]

use crate::{NdiError, Result};
use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// SpeedHQ intra-prediction helpers
// ─────────────────────────────────────────────────────────────────────────────

/// DCT-domain intra-prediction encoder helper.
///
/// Subtracts a flat `pred` value from every coefficient in `block`, which models
/// the simplest possible "DC prediction" used in intra-frame coding.
pub struct SpeedHqEncoder;

impl SpeedHqEncoder {
    /// Return a new block where `pred` has been subtracted from every coefficient.
    ///
    /// This implements the residual computation step of intra-prediction:
    /// `residual[i] = block[i] - pred`.
    #[must_use]
    pub fn encode_intra_prediction(block: &[i32; 64], pred: i32) -> [i32; 64] {
        let mut residual = [0i32; 64];
        for (i, &v) in block.iter().enumerate() {
            residual[i] = v.saturating_sub(pred);
        }
        residual
    }
}

/// DCT-domain intra-prediction decoder helper.
pub struct SpeedHqDecoder;

impl SpeedHqDecoder {
    /// Reconstruct a block by adding `pred` back to every residual coefficient.
    ///
    /// Inverse of [`SpeedHqEncoder::encode_intra_prediction`]:
    /// `block[i] = residual[i] + pred`.
    #[must_use]
    pub fn decode_intra_prediction(residual: &[i32; 64], pred: i32) -> [i32; 64] {
        let mut block = [0i32; 64];
        for (i, &r) in residual.iter().enumerate() {
            block[i] = r.saturating_add(pred);
        }
        block
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Extended tally serialisation (v2)
// ─────────────────────────────────────────────────────────────────────────────

/// Flag bits for the v2 tally wire format.
pub mod tally_flags {
    /// Source is on program output (red tally).
    pub const PROGRAM: u16 = 0x0001;
    /// Source is on preview output (green tally).
    pub const PREVIEW: u16 = 0x0002;
    /// Source is currently being recorded.
    pub const RECORD: u16 = 0x0004;
}

/// Extended tally message with program, preview, and record flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TallyV2Message {
    /// True when the source is on the program bus.
    pub program: bool,
    /// True when the source is on the preview bus.
    pub preview: bool,
    /// True when recording is active.
    pub record: bool,
}

impl TallyV2Message {
    /// Create a new v2 tally message.
    #[must_use]
    pub fn new(program: bool, preview: bool, record: bool) -> Self {
        Self { program, preview, record }
    }

    /// Serialise to 2 bytes representing the packed flag word (little-endian u16).
    ///
    /// Bit layout:
    /// - bit 0 = program
    /// - bit 1 = preview
    /// - bit 2 = record
    #[must_use]
    pub fn serialize_v2(&self) -> Vec<u8> {
        let mut flags: u16 = 0;
        if self.program {
            flags |= tally_flags::PROGRAM;
        }
        if self.preview {
            flags |= tally_flags::PREVIEW;
        }
        if self.record {
            flags |= tally_flags::RECORD;
        }
        flags.to_le_bytes().to_vec()
    }

    /// Deserialise from 2 bytes (little-endian u16 flag word).
    ///
    /// # Errors
    ///
    /// Returns an error when `buf` is shorter than 2 bytes.
    pub fn deserialize_v2(buf: &[u8]) -> Result<Self> {
        if buf.len() < 2 {
            return Err(NdiError::Protocol(
                "TallyV2Message::deserialize_v2 requires at least 2 bytes".to_string(),
            ));
        }
        let flags = u16::from_le_bytes([buf[0], buf[1]]);
        Ok(Self {
            program: (flags & tally_flags::PROGRAM) != 0,
            preview: (flags & tally_flags::PREVIEW) != 0,
            record: (flags & tally_flags::RECORD) != 0,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PTZ VISCA-like binary serialisation
// ─────────────────────────────────────────────────────────────────────────────

/// Helper for encoding PTZ camera commands to a VISCA-inspired binary wire format.
///
/// This is a simplified implementation; a production implementation would
/// follow VISCA over IP (RFC-style) framing exactly.
pub struct PtzCommandEncoder;

/// Convert a normalised floating-point value in `[-1.0, 1.0]` to a u8 in `[0, 255]`.
/// 0.0 maps to 128 (neutral/stop).
fn f32_to_speed_byte(val: f32) -> u8 {
    let clamped = val.clamp(-1.0, 1.0);
    // Map [-1, 1] → [0, 255] with 128 = neutral
    let scaled = (clamped + 1.0) * 127.5;
    scaled.round().clamp(0.0, 255.0) as u8
}

/// Convert a normalised level in `[0.0, 1.0]` to a u8 in `[0, 255]`.
fn f32_to_level_byte(val: f32) -> u8 {
    (val.clamp(0.0, 1.0) * 255.0).round() as u8
}

impl PtzCommandEncoder {
    /// Encode a combined pan/tilt move command as a VISCA-like binary sequence.
    ///
    /// Wire format (8 bytes):
    /// ```text
    /// [0x81, 0x01, 0x06, 0x01, <pan_speed>, <tilt_speed>, <pan_dir>, <tilt_dir>]
    /// ```
    /// where `pan_speed` and `tilt_speed` are 0-24 (VISCA nominal range, mapped
    /// from the 0.0–1.0 magnitude), and `pan_dir`/`tilt_dir` are 1=right/up,
    /// 2=left/down, 3=stop.
    #[must_use]
    pub fn pan_tilt(pan: f32, tilt: f32, speed: f32) -> Vec<u8> {
        let speed_byte = (speed.clamp(0.0, 1.0) * 24.0).round() as u8;
        // Direction: positive pan = right (0x02), negative = left (0x01), zero = stop (0x03)
        let pan_dir: u8 = if pan > 0.0 {
            0x02
        } else if pan < 0.0 {
            0x01
        } else {
            0x03
        };
        let tilt_dir: u8 = if tilt > 0.0 {
            0x01 // up
        } else if tilt < 0.0 {
            0x02 // down
        } else {
            0x03 // stop
        };
        vec![0x81, 0x01, 0x06, 0x01, speed_byte, speed_byte, pan_dir, tilt_dir]
    }

    /// Encode a zoom command as a VISCA-like binary sequence.
    ///
    /// Wire format (6 bytes):
    /// ```text
    /// [0x81, 0x01, 0x04, 0x07, <zoom_level_byte>, 0xFF]
    /// ```
    /// where `zoom_level_byte` is `level * 255` mapped to `[0, 255]`.
    #[must_use]
    pub fn zoom(level: f32) -> Vec<u8> {
        let level_byte = f32_to_level_byte(level);
        vec![0x81, 0x01, 0x04, 0x07, level_byte, 0xFF]
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Audio channel metadata
// ─────────────────────────────────────────────────────────────────────────────

/// Wire-serialisable audio channel metadata for an NDI audio stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NdiAudioMetadata {
    /// Number of audio channels.
    pub channels: u16,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Bits per sample (e.g. 16, 24, 32).
    pub bits_per_sample: u16,
    /// Timestamp in 100-ns ticks (NDI convention).
    pub timestamp: u64,
}

impl NdiAudioMetadata {
    /// Create a new audio metadata record.
    #[must_use]
    pub fn new(channels: u16, sample_rate: u32, bits_per_sample: u16, timestamp: u64) -> Self {
        Self { channels, sample_rate, bits_per_sample, timestamp }
    }

    /// Serialise to bytes.
    ///
    /// Wire format (16 bytes, all little-endian):
    /// ```text
    /// [channels u16][sample_rate u32][bits_per_sample u16][timestamp u64]
    /// ```
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(16);
        buf.extend_from_slice(&self.channels.to_le_bytes());
        buf.extend_from_slice(&self.sample_rate.to_le_bytes());
        buf.extend_from_slice(&self.bits_per_sample.to_le_bytes());
        buf.extend_from_slice(&self.timestamp.to_le_bytes());
        buf
    }

    /// Deserialise from bytes.
    ///
    /// # Errors
    ///
    /// Returns an error when `buf` is shorter than 16 bytes.
    pub fn from_bytes(buf: &[u8]) -> Result<Self> {
        if buf.len() < 16 {
            return Err(NdiError::Protocol(
                "NdiAudioMetadata::from_bytes requires at least 16 bytes".to_string(),
            ));
        }
        let channels = u16::from_le_bytes([buf[0], buf[1]]);
        let sample_rate = u32::from_le_bytes([buf[2], buf[3], buf[4], buf[5]]);
        let bits_per_sample = u16::from_le_bytes([buf[6], buf[7]]);
        // Read all 8 bytes for timestamp starting at offset 8
        let timestamp = u64::from_le_bytes([
            buf[8], buf[9], buf[10], buf[11], buf[12], buf[13], buf[14], buf[15],
        ]);
        Ok(Self { channels, sample_rate, bits_per_sample, timestamp })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Discovery cache with TTL
// ─────────────────────────────────────────────────────────────────────────────

/// A minimal NDI source descriptor for caching purposes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NdiSource {
    /// Unique source name.
    pub name: String,
    /// Network address string (e.g. `"192.168.1.10:5960"`).
    pub address: String,
}

impl NdiSource {
    /// Create a new NDI source.
    #[must_use]
    pub fn new(name: impl Into<String>, address: impl Into<String>) -> Self {
        Self { name: name.into(), address: address.into() }
    }
}

/// Cache entry pairing a source with its expiry time.
#[derive(Debug, Clone)]
struct CacheEntry {
    source: NdiSource,
    expires_at: u64,
}

/// Time-to-live cache for NDI source discovery results.
///
/// Uses monotonic unix timestamps (seconds since epoch or any consistent
/// monotonic counter) provided by the caller so the cache is
/// deterministic in tests.
#[derive(Debug, Clone)]
pub struct DiscoveryCache {
    ttl_secs: u64,
    entries: HashMap<String, CacheEntry>,
}

impl DiscoveryCache {
    /// Create a new discovery cache with `ttl_secs` time-to-live per entry.
    #[must_use]
    pub fn new(ttl_secs: u64) -> Self {
        Self {
            ttl_secs,
            entries: HashMap::new(),
        }
    }

    /// Insert or refresh `source` in the cache with expiry at `now + ttl`.
    pub fn insert(&mut self, source: NdiSource, now: u64) {
        let expires_at = now.saturating_add(self.ttl_secs);
        self.entries.insert(
            source.name.clone(),
            CacheEntry { source, expires_at },
        );
    }

    /// Return all sources that have not yet expired as of `now`.
    #[must_use]
    pub fn get_active(&self, now: u64) -> Vec<NdiSource> {
        self.entries
            .values()
            .filter(|e| e.expires_at > now)
            .map(|e| e.source.clone())
            .collect()
    }

    /// Remove all expired entries (prune the cache).
    pub fn prune(&mut self, now: u64) {
        self.entries.retain(|_, e| e.expires_at > now);
    }

    /// Number of entries currently in the cache (including expired).
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` when the cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Metadata frame injection
// ─────────────────────────────────────────────────────────────────────────────

/// An NDI metadata frame wrapping arbitrary XML data.
#[derive(Debug, Clone)]
pub struct NdiMetadataFrame {
    /// The raw XML payload to inject into the NDI stream.
    pub xml_data: String,
}

impl NdiMetadataFrame {
    /// Create a new metadata frame.
    #[must_use]
    pub fn new(xml_data: impl Into<String>) -> Self {
        Self { xml_data: xml_data.into() }
    }

    /// Return the byte length of the XML payload.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.xml_data.len()
    }
}

/// Extension methods for injecting metadata frames into a sender.
pub struct NdiSenderMetadata;

impl NdiSenderMetadata {
    /// Validate and enqueue a metadata frame for transmission.
    ///
    /// In this stub implementation, we verify the XML is non-empty and returns
    /// `Ok(())` to signal readiness.  A production implementation would write
    /// `frame.xml_data` into the sender's outgoing metadata channel.
    ///
    /// # Errors
    ///
    /// Returns [`NdiError::Protocol`] when `frame.xml_data` is empty.
    pub fn send_metadata(frame: &NdiMetadataFrame) -> Result<()> {
        if frame.xml_data.is_empty() {
            return Err(NdiError::Protocol(
                "Metadata frame xml_data must not be empty".to_string(),
            ));
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Bandwidth estimation
// ─────────────────────────────────────────────────────────────────────────────

/// Exponential moving average bandwidth estimator.
#[derive(Debug, Clone)]
pub struct BandwidthEstimator {
    /// EMA smoothing factor α ∈ (0, 1].
    alpha: f32,
    /// Current EMA estimate in kbps.
    estimate_kbps: f32,
    /// Whether at least one sample has been provided.
    initialized: bool,
}

impl BandwidthEstimator {
    /// Create a new estimator with the given EMA alpha (0 < alpha ≤ 1).
    ///
    /// Use a smaller alpha (e.g. 0.1) for a more stable, slowly-reacting estimate,
    /// or a larger alpha (e.g. 0.5) for faster reaction to changes.
    #[must_use]
    pub fn new(alpha: f32) -> Self {
        Self {
            alpha: alpha.clamp(0.01, 1.0),
            estimate_kbps: 0.0,
            initialized: false,
        }
    }

    /// Update the estimate with a new measurement.
    ///
    /// `bytes_sent` is the number of bytes transmitted in the measurement window;
    /// `elapsed_ms` is the duration of the window in milliseconds.
    pub fn update(&mut self, bytes_sent: u64, elapsed_ms: u64) {
        if elapsed_ms == 0 {
            return;
        }
        // Convert bytes → kbps: (bytes * 8) / (elapsed_ms / 1000) / 1000
        let measured_kbps = (bytes_sent as f64 * 8.0 / elapsed_ms as f64) as f32;

        if self.initialized {
            self.estimate_kbps = self.alpha * measured_kbps + (1.0 - self.alpha) * self.estimate_kbps;
        } else {
            self.estimate_kbps = measured_kbps;
            self.initialized = true;
        }
    }

    /// Return the current estimated bandwidth in kbps.
    #[must_use]
    pub fn estimate_kbps(&self) -> f32 {
        self.estimate_kbps
    }
}

impl Default for BandwidthEstimator {
    fn default() -> Self {
        Self::new(0.1)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Multi-sender
// ─────────────────────────────────────────────────────────────────────────────


/// A stub video frame type for multi-sender demonstrations.
#[derive(Debug, Clone)]
pub struct VideoFrame {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Raw pixel data (RGBA).
    pub data: Vec<u8>,
}

impl VideoFrame {
    /// Create a new video frame.
    #[must_use]
    pub fn new(width: u32, height: u32, data: Vec<u8>) -> Self {
        Self { width, height, data }
    }
}

/// Manages multiple independent NDI senders identified by index.
///
/// Each sender is associated with a name; frames can be directed to any sender
/// by index.  This is a purely in-memory stub — a production implementation
/// would wire each sender to its own [`crate::sender::NdiSender`] instance.
#[derive(Debug, Default)]
pub struct NdiMultiSender {
    /// Sender names in insertion order (index == position).
    sender_names: Vec<String>,
    /// Frames queued per sender index (for testing/verification).
    queued_frames: Vec<Vec<VideoFrame>>,
}

impl NdiMultiSender {
    /// Create a new, empty multi-sender.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new named sender and return its index.
    ///
    /// # Errors
    ///
    /// Returns [`NdiError::Protocol`] when `name` is empty or already registered.
    pub fn add_sender(&mut self, name: String) -> Result<usize> {
        if name.is_empty() {
            return Err(NdiError::Protocol("Sender name must not be empty".to_string()));
        }
        if self.sender_names.contains(&name) {
            return Err(NdiError::Protocol(format!("Sender '{name}' already registered")));
        }
        let idx = self.sender_names.len();
        self.sender_names.push(name);
        self.queued_frames.push(Vec::new());
        Ok(idx)
    }

    /// Send `frame` to the sender at `index`.
    ///
    /// # Errors
    ///
    /// Returns [`NdiError::Protocol`] when `index` is out of range.
    pub fn send_to(&mut self, index: usize, frame: &VideoFrame) -> Result<()> {
        if index >= self.sender_names.len() {
            return Err(NdiError::Protocol(format!(
                "Sender index {index} out of range (have {})",
                self.sender_names.len()
            )));
        }
        self.queued_frames[index].push(frame.clone());
        Ok(())
    }

    /// Number of registered senders.
    #[must_use]
    pub fn sender_count(&self) -> usize {
        self.sender_names.len()
    }

    /// Name of the sender at `index`, or `None` when out of range.
    #[must_use]
    pub fn sender_name(&self, index: usize) -> Option<&str> {
        self.sender_names.get(index).map(String::as_str)
    }

    /// Frames queued for sender at `index` (for inspection in tests).
    #[must_use]
    pub fn frames_for(&self, index: usize) -> &[VideoFrame] {
        self.queued_frames
            .get(index)
            .map(Vec::as_slice)
            .unwrap_or_default()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── SpeedHQ intra-prediction ─────────────────────────────────────────────

    #[test]
    fn test_encode_intra_prediction_round_trip() {
        let mut block = [0i32; 64];
        for (i, v) in block.iter_mut().enumerate() {
            *v = i as i32 * 3 - 96;
        }
        let pred = 10;
        let residual = SpeedHqEncoder::encode_intra_prediction(&block, pred);
        let reconstructed = SpeedHqDecoder::decode_intra_prediction(&residual, pred);
        assert_eq!(block, reconstructed);
    }

    #[test]
    fn test_encode_intra_prediction_subtracts_pred() {
        let block = [5i32; 64];
        let pred = 3;
        let residual = SpeedHqEncoder::encode_intra_prediction(&block, pred);
        assert!(residual.iter().all(|&r| r == 2));
    }

    #[test]
    fn test_decode_intra_prediction_adds_pred() {
        let residual = [0i32; 64];
        let pred = 42;
        let block = SpeedHqDecoder::decode_intra_prediction(&residual, pred);
        assert!(block.iter().all(|&v| v == 42));
    }

    // ── TallyV2Message ───────────────────────────────────────────────────────

    #[test]
    fn test_tally_v2_serialize_length_is_2() {
        let msg = TallyV2Message::new(true, true, false);
        let bytes = msg.serialize_v2();
        assert_eq!(bytes.len(), 2, "v2 tally flags must be exactly 2 bytes");
    }

    #[test]
    fn test_tally_v2_round_trip() {
        for (p, pr, r) in [(true, false, false), (false, true, true), (true, true, true)] {
            let msg = TallyV2Message::new(p, pr, r);
            let bytes = msg.serialize_v2();
            let decoded = TallyV2Message::deserialize_v2(&bytes).expect("valid bytes");
            assert_eq!(decoded, msg, "round-trip failed for {:?}", msg);
        }
    }

    #[test]
    fn test_tally_v2_program_only() {
        let msg = TallyV2Message::new(true, false, false);
        let bytes = msg.serialize_v2();
        let flags = u16::from_le_bytes([bytes[0], bytes[1]]);
        assert_eq!(flags & tally_flags::PROGRAM, tally_flags::PROGRAM);
        assert_eq!(flags & tally_flags::PREVIEW, 0);
        assert_eq!(flags & tally_flags::RECORD, 0);
    }

    #[test]
    fn test_tally_v2_deserialize_too_short() {
        assert!(TallyV2Message::deserialize_v2(&[0x01]).is_err());
    }

    // ── PTZ encoding ─────────────────────────────────────────────────────────

    #[test]
    fn test_ptz_pan_tilt_non_empty() {
        let bytes = PtzCommandEncoder::pan_tilt(0.5, -0.3, 0.8);
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x81); // VISCA header
        assert_eq!(bytes[2], 0x06); // pan/tilt category
    }

    #[test]
    fn test_ptz_zoom_non_empty() {
        let bytes = PtzCommandEncoder::zoom(0.5);
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x81);
        assert_eq!(bytes[3], 0x07); // zoom command
    }

    #[test]
    fn test_ptz_pan_tilt_stop_direction() {
        let bytes = PtzCommandEncoder::pan_tilt(0.0, 0.0, 0.5);
        // pan_dir = 0x03 (stop), tilt_dir = 0x03 (stop)
        assert_eq!(bytes[6], 0x03);
        assert_eq!(bytes[7], 0x03);
    }

    // ── NdiAudioMetadata ─────────────────────────────────────────────────────

    #[test]
    fn test_audio_metadata_round_trip() {
        let meta = NdiAudioMetadata::new(2, 48_000, 24, 1_000_000_000);
        let bytes = meta.to_bytes();
        assert_eq!(bytes.len(), 16);
        let decoded = NdiAudioMetadata::from_bytes(&bytes).expect("valid bytes");
        assert_eq!(decoded.channels, 2);
        assert_eq!(decoded.sample_rate, 48_000);
        assert_eq!(decoded.bits_per_sample, 24);
        assert_eq!(decoded.timestamp, 1_000_000_000);
    }

    #[test]
    fn test_audio_metadata_too_short() {
        assert!(NdiAudioMetadata::from_bytes(&[0u8; 10]).is_err());
    }

    // ── DiscoveryCache ───────────────────────────────────────────────────────

    #[test]
    fn test_discovery_cache_insert_and_get_active() {
        let mut cache = DiscoveryCache::new(60);
        let src = NdiSource::new("Camera1", "192.168.1.10:5960");
        cache.insert(src.clone(), 1000); // inserted at t=1000, expires at t=1060

        let active = cache.get_active(1059);
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].name, "Camera1");
    }

    #[test]
    fn test_discovery_cache_expires_old_entries() {
        let mut cache = DiscoveryCache::new(30);
        cache.insert(NdiSource::new("OldCam", "10.0.0.1:5960"), 100); // expires at 130

        // At t=131, entry should be gone
        let active = cache.get_active(131);
        assert!(active.is_empty());
    }

    #[test]
    fn test_discovery_cache_refresh_extends_ttl() {
        let mut cache = DiscoveryCache::new(10);
        let src = NdiSource::new("Cam", "1.2.3.4:5960");
        cache.insert(src.clone(), 0);
        // Refresh at t=5
        cache.insert(src, 5); // new expiry = 15
        // At t=14, should still be active
        assert_eq!(cache.get_active(14).len(), 1);
    }

    // ── NdiMetadataFrame ─────────────────────────────────────────────────────

    #[test]
    fn test_send_metadata_valid() {
        let frame = NdiMetadataFrame::new("<ndi_product vendor=\"test\"/>");
        assert!(NdiSenderMetadata::send_metadata(&frame).is_ok());
    }

    #[test]
    fn test_send_metadata_empty_fails() {
        let frame = NdiMetadataFrame::new("");
        assert!(NdiSenderMetadata::send_metadata(&frame).is_err());
    }

    // ── BandwidthEstimator ───────────────────────────────────────────────────

    #[test]
    fn test_bandwidth_estimator_1mbps() {
        let mut est = BandwidthEstimator::new(1.0); // α=1 (instant)
        // 125_000 bytes in 1000ms = 1 Mbps = 1000 kbps
        est.update(125_000, 1_000);
        let kbps = est.estimate_kbps();
        assert!((kbps - 1000.0).abs() < 1.0, "expected ~1000 kbps, got {kbps}");
    }

    #[test]
    fn test_bandwidth_estimator_ema() {
        let mut est = BandwidthEstimator::new(0.5);
        est.update(125_000, 1_000); // 1000 kbps
        est.update(250_000, 1_000); // 2000 kbps
        // 0.5 * 2000 + 0.5 * 1000 = 1500 kbps
        let kbps = est.estimate_kbps();
        assert!((kbps - 1500.0).abs() < 1.0, "expected ~1500 kbps, got {kbps}");
    }

    #[test]
    fn test_bandwidth_estimator_zero_elapsed_ignored() {
        let mut est = BandwidthEstimator::new(1.0);
        est.update(100_000, 0); // Should not crash or update
        assert_eq!(est.estimate_kbps(), 0.0);
    }

    // ── NdiMultiSender ───────────────────────────────────────────────────────

    #[test]
    fn test_multi_sender_add_and_send() {
        let mut ms = NdiMultiSender::new();
        let idx = ms.add_sender("Camera1".to_string()).expect("add sender");
        assert_eq!(idx, 0);
        let frame = VideoFrame::new(1920, 1080, vec![0u8; 1920 * 1080 * 4]);
        ms.send_to(idx, &frame).expect("send frame");
        assert_eq!(ms.frames_for(0).len(), 1);
    }

    #[test]
    fn test_multi_sender_duplicate_name_fails() {
        let mut ms = NdiMultiSender::new();
        ms.add_sender("Cam".to_string()).expect("first add");
        assert!(ms.add_sender("Cam".to_string()).is_err());
    }

    #[test]
    fn test_multi_sender_out_of_range_fails() {
        let mut ms = NdiMultiSender::new();
        let frame = VideoFrame::new(640, 480, vec![]);
        assert!(ms.send_to(5, &frame).is_err());
    }

    #[test]
    fn test_multi_sender_two_senders() {
        let mut ms = NdiMultiSender::new();
        let i0 = ms.add_sender("Alpha".to_string()).expect("add alpha");
        let i1 = ms.add_sender("Beta".to_string()).expect("add beta");
        let f = VideoFrame::new(1, 1, vec![255u8; 4]);
        ms.send_to(i0, &f).expect("send to alpha");
        ms.send_to(i0, &f).expect("send to alpha 2");
        ms.send_to(i1, &f).expect("send to beta");
        assert_eq!(ms.frames_for(i0).len(), 2);
        assert_eq!(ms.frames_for(i1).len(), 1);
    }
}
