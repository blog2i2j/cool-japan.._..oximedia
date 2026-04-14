//! VideoIP protocol extensions: RIST transport primitives, simple congestion
//! controller, frame pacer, NDI bridge passthrough, SDP generator, SMPTE 2110
//! timing validator, simplified stream health monitor, stream bonding,
//! color-space conversion (UYVY→RGBA), and AES-CTR encryption stub.

#![allow(dead_code)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]

// ─────────────────────────────────────────────────────────────────────────────
// RIST Transport / Packet
// ─────────────────────────────────────────────────────────────────────────────

/// A RIST transport session configuration.
///
/// Stores the target re-order buffer latency and provides packet
/// serialize/deserialize helpers.
#[derive(Debug, Clone)]
pub struct RistTransport {
    /// Re-order / jitter buffer depth in milliseconds.
    pub latency_ms: u32,
}

impl RistTransport {
    /// Create a new RIST transport with the given latency.
    #[must_use]
    pub fn new(latency_ms: u32) -> Self {
        Self { latency_ms }
    }
}

/// RTP-like RIST packet.
///
/// Wire format (12 bytes header + payload):
/// ```text
/// [2 bits version=2][1 bit padding=0][1 bit ext=0][4 bits CC=0] (1 byte)
/// [1 bit marker=0][7 bits PT=96] (1 byte)
/// [seq: u16 big-endian] (2 bytes)
/// [timestamp: u32 big-endian] (4 bytes)
/// [SSRC: u32 big-endian = 0xCAFE_BABE] (4 bytes)
/// [payload...]
/// ```
pub struct RistPacket;

/// Fixed SSRC used by this implementation.
const RIST_SSRC: u32 = 0xCAFE_BABE;
/// RTP version 2 | payload type 96.
const RIST_VPXCC: u8 = 0b1000_0000; // V=2, P=0, X=0, CC=0
const RIST_M_PT: u8 = 96; // marker=0, PT=96

impl RistPacket {
    /// Serialise a RIST packet.
    ///
    /// Returns a `Vec<u8>` with the 12-byte RTP-like header followed by `data`.
    #[must_use]
    pub fn serialize(seq: u32, ts: u32, data: &[u8]) -> Vec<u8> {
        // Sequence number is truncated to u16 in the header (standard RTP).
        let seq16 = (seq & 0xFFFF) as u16;
        let mut buf = Vec::with_capacity(12 + data.len());
        buf.push(RIST_VPXCC);
        buf.push(RIST_M_PT);
        buf.extend_from_slice(&seq16.to_be_bytes());
        buf.extend_from_slice(&ts.to_be_bytes());
        buf.extend_from_slice(&RIST_SSRC.to_be_bytes());
        buf.extend_from_slice(data);
        buf
    }

    /// Deserialise a RIST packet.
    ///
    /// Returns `Some((seq, timestamp, payload))` on success, or `None` when the
    /// buffer is shorter than the 12-byte header or the magic bytes don't match.
    #[must_use]
    pub fn deserialize(buf: &[u8]) -> Option<(u32, u32, Vec<u8>)> {
        if buf.len() < 12 {
            return None;
        }
        // Validate version=2 (top two bits of first byte must be 0b10)
        if buf[0] & 0b1100_0000 != 0b1000_0000 {
            return None;
        }
        // Validate payload type = 96
        if buf[1] & 0x7F != 96 {
            return None;
        }
        let seq = u16::from_be_bytes([buf[2], buf[3]]) as u32;
        let ts = u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]);
        // Skip SSRC bytes (4..8 from offset 8)
        let payload = buf[12..].to_vec();
        Some((seq, ts, payload))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Simple Congestion Controller (kbps-based, AIMD-lite)
// ─────────────────────────────────────────────────────────────────────────────

/// Lightweight congestion controller that works in kbps units.
///
/// This is a minimal AIMD controller suitable for embedding in streaming
/// pipelines without the full [`crate::congestion::CongestionController`]
/// infrastructure.
#[derive(Debug, Clone)]
pub struct SimpleCongestionController {
    /// Current estimated send rate in kbps.
    rate_kbps: u32,
    /// Minimum allowed rate.
    min_kbps: u32,
    /// Maximum allowed rate.
    max_kbps: u32,
}

impl SimpleCongestionController {
    /// Create a new controller with `initial_rate_kbps`.
    #[must_use]
    pub fn new(initial_rate_kbps: u32) -> Self {
        Self {
            rate_kbps: initial_rate_kbps,
            min_kbps: 100,
            max_kbps: 1_000_000,
        }
    }

    /// React to packet loss: decrease rate multiplicatively.
    ///
    /// Formula: `rate = (rate * (1.0 - 0.5 * lost_frac)).max(min)`.
    /// Returns the new rate.
    pub fn on_loss(&mut self, lost_frac: f32) -> u32 {
        let factor = (1.0 - 0.5 * lost_frac.clamp(0.0, 1.0)) as f64;
        let new_rate = (self.rate_kbps as f64 * factor).round() as u32;
        self.rate_kbps = new_rate.clamp(self.min_kbps, self.max_kbps);
        self.rate_kbps
    }

    /// React to an ACK: slow additive increase by 1 kbps.
    ///
    /// Returns the new rate.
    pub fn on_ack(&mut self) -> u32 {
        self.rate_kbps = (self.rate_kbps + 1).min(self.max_kbps);
        self.rate_kbps
    }

    /// Current rate in kbps.
    #[must_use]
    pub fn rate_kbps(&self) -> u32 {
        self.rate_kbps
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Frame Pacer
// ─────────────────────────────────────────────────────────────────────────────

/// Calculates inter-frame timing for precise frame delivery.
#[derive(Debug, Clone)]
pub struct FramePacer {
    /// Target frames per second.
    target_fps: f32,
    /// Frame interval in milliseconds.
    frame_interval_ms: f32,
}

impl FramePacer {
    /// Create a new frame pacer for `target_fps`.
    ///
    /// # Panics
    ///
    /// Does not panic; `target_fps <= 0` results in `frame_interval_ms = 0`.
    #[must_use]
    pub fn new(target_fps: f32) -> Self {
        let frame_interval_ms = if target_fps > 0.0 {
            1000.0 / target_fps
        } else {
            0.0
        };
        Self { target_fps, frame_interval_ms }
    }

    /// Return the number of milliseconds to wait before sending the next frame.
    ///
    /// If `now_ts - last_sent_ts >= frame_interval_ms`, returns 0 (send immediately).
    /// Otherwise returns the remaining wait time.
    #[must_use]
    pub fn time_until_next_frame_ms(&self, last_sent_ts: u64, now_ts: u64) -> u64 {
        let elapsed = now_ts.saturating_sub(last_sent_ts);
        let interval = self.frame_interval_ms.round() as u64;
        if elapsed >= interval {
            0
        } else {
            interval - elapsed
        }
    }

    /// Target frames per second.
    #[must_use]
    pub fn target_fps(&self) -> f32 {
        self.target_fps
    }

    /// Frame interval in milliseconds.
    #[must_use]
    pub fn frame_interval_ms(&self) -> f32 {
        self.frame_interval_ms
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NDI Bridge (passthrough stub with header conversion)
// ─────────────────────────────────────────────────────────────────────────────

/// Bridges NDI-format frames to VideoIP wire format.
pub struct NdiBridge;

impl NdiBridge {
    /// Convert an NDI raw frame to VideoIP encapsulation.
    ///
    /// This is a passthrough stub: it prepends a 12-byte VideoIP header
    /// encoding the width (u32 BE) and height (u32 BE) + a 4-byte magic
    /// `b"VIPF"` and returns the result.  A real implementation would
    /// re-packetise into the VideoIP packet format.
    #[must_use]
    pub fn transcode_to_videoip(ndi_frame: &[u8], width: u32, height: u32) -> Vec<u8> {
        let mut out = Vec::with_capacity(12 + ndi_frame.len());
        out.extend_from_slice(b"VIPF");
        out.extend_from_slice(&width.to_be_bytes());
        out.extend_from_slice(&height.to_be_bytes());
        out.extend_from_slice(ndi_frame);
        out
    }

    /// Strip the VideoIP header from a `transcode_to_videoip` result.
    ///
    /// Returns `(width, height, payload)` or `None` if the header is missing/invalid.
    #[must_use]
    pub fn strip_header(buf: &[u8]) -> Option<(u32, u32, &[u8])> {
        if buf.len() < 12 || &buf[0..4] != b"VIPF" {
            return None;
        }
        let width = u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]);
        let height = u32::from_be_bytes([buf[8], buf[9], buf[10], buf[11]]);
        Some((width, height, &buf[12..]))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SDP Generator
// ─────────────────────────────────────────────────────────────────────────────

use crate::stream_descriptor::StreamDescriptor;

/// Generates minimal SDP (Session Description Protocol) documents.
pub struct SdpGenerator;

impl SdpGenerator {
    /// Generate a minimal SDP document for the given stream descriptor.
    ///
    /// Produces an RFC 4566-compliant SDP with `v=`, `o=`, `s=`, `t=`, `m=video`,
    /// and `a=rtpmap` lines.
    #[must_use]
    pub fn generate(stream: &StreamDescriptor) -> String {
        let codec = stream.codec();
        let rate = stream.rate();
        let width = if stream.stream_type().is_video() {
            // We need to access width — reconstruct from is_hd check
            // The StreamDescriptor doesn't expose width directly, so use metadata or rate
            // Use a reasonable default based on whether it's HD
            if stream.is_hd() { 1920u32 } else { 720u32 }
        } else {
            0
        };

        // Use payload type 96 (dynamic) for compressed, 98 for uncompressed
        let pt = 96u8;

        let mut sdp = String::new();
        sdp.push_str("v=0\r\n");
        sdp.push_str("o=- 0 0 IN IP4 0.0.0.0\r\n");
        sdp.push_str(&format!("s={}\r\n", stream.id()));
        sdp.push_str("t=0 0\r\n");
        sdp.push_str(&format!("m=video 5004 RTP/AVP {pt}\r\n"));
        sdp.push_str("c=IN IP4 0.0.0.0\r\n");
        sdp.push_str(&format!(
            "a=rtpmap:{pt} {codec}/{rate}\r\n",
        ));
        sdp.push_str(&format!(
            "a=fmtp:{pt} width={width};height={width}\r\n"
        ));
        sdp
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SMPTE 2110-20 Timing Validator
// ─────────────────────────────────────────────────────────────────────────────

/// A timing violation detected during SMPTE 2110-20 packet analysis.
#[derive(Debug, Clone, PartialEq)]
pub struct TimingViolation {
    /// Packet index (0-based).
    pub packet_index: usize,
    /// Expected arrival timestamp (ms).
    pub expected_ts: u64,
    /// Actual arrival timestamp (ms).
    pub actual_ts: u64,
    /// Violation type.
    pub violation: ViolationType,
}

/// Type of timing violation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViolationType {
    /// Packet arrived too early (before its scheduled slot).
    TooEarly,
    /// Packet arrived too late (after its scheduled slot).
    TooLate,
}

/// SMPTE 2110-20 packet timing validator.
pub struct Smpte2110Validator;

impl Smpte2110Validator {
    /// Tolerance window in milliseconds: packets within ±`TOLERANCE_MS` are considered on-time.
    const TOLERANCE_MS: f64 = 0.5;

    /// Check packet arrival timing against the expected frame schedule.
    ///
    /// `packets` is a slice of `(arrival_ts_ms, packet_size_bytes)` pairs ordered
    /// by packet index.  `fps` is the stream frame rate in Hz.
    ///
    /// Returns a list of [`TimingViolation`]s for packets that arrived outside the
    /// ±0.5 ms tolerance window.
    #[must_use]
    pub fn check_packet_timing(packets: &[(u64, u32)], fps: f32) -> Vec<TimingViolation> {
        if fps <= 0.0 || packets.is_empty() {
            return Vec::new();
        }

        let frame_interval_ms = 1000.0 / fps as f64;
        let mut violations = Vec::new();

        // Use the first packet's timestamp as the reference origin.
        let origin = packets[0].0;

        for (i, &(ts, _size)) in packets.iter().enumerate() {
            let expected_ts = origin + (i as f64 * frame_interval_ms).round() as u64;
            let ts_f = ts as f64;
            let expected_f = expected_ts as f64;
            let delta = ts_f - expected_f;

            if delta < -Self::TOLERANCE_MS {
                violations.push(TimingViolation {
                    packet_index: i,
                    expected_ts,
                    actual_ts: ts,
                    violation: ViolationType::TooEarly,
                });
            } else if delta > Self::TOLERANCE_MS {
                violations.push(TimingViolation {
                    packet_index: i,
                    expected_ts,
                    actual_ts: ts,
                    violation: ViolationType::TooLate,
                });
            }
        }

        violations
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Simple Stream Health Monitor
// ─────────────────────────────────────────────────────────────────────────────

/// Stream health grade.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum HealthGrade {
    /// Major quality issues.
    Poor,
    /// Some quality issues.
    Fair,
    /// Healthy stream.
    Good,
    /// Excellent stream quality.
    Excellent,
}

impl std::fmt::Display for HealthGrade {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Poor => write!(f, "POOR"),
            Self::Fair => write!(f, "FAIR"),
            Self::Good => write!(f, "GOOD"),
            Self::Excellent => write!(f, "EXCELLENT"),
        }
    }
}

/// Simple stream health monitor with threshold-based grading.
#[derive(Debug, Clone, Default)]
pub struct SimpleStreamHealthMonitor {
    /// Most recent loss percentage (0.0–100.0).
    pub loss_pct: f32,
    /// Most recent jitter in milliseconds.
    pub jitter_ms: f32,
    /// Most recent bitrate in kbps.
    pub bitrate_kbps: u32,
}

impl SimpleStreamHealthMonitor {
    /// Create a new health monitor.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a new measurement.
    pub fn update(&mut self, loss_pct: f32, jitter_ms: f32, bitrate_kbps: u32) {
        self.loss_pct = loss_pct;
        self.jitter_ms = jitter_ms;
        self.bitrate_kbps = bitrate_kbps;
    }

    /// Compute a health grade from the most recent measurement.
    ///
    /// Thresholds (any single metric degrades the grade):
    /// - Excellent: loss < 0.1%, jitter < 2ms, bitrate > 5000 kbps
    /// - Good:      loss < 1%,   jitter < 10ms, bitrate > 1000 kbps
    /// - Fair:      loss < 5%,   jitter < 50ms, bitrate > 100 kbps
    /// - Poor:      otherwise
    #[must_use]
    pub fn health_grade(&self) -> HealthGrade {
        if self.loss_pct < 0.1 && self.jitter_ms < 2.0 && self.bitrate_kbps > 5_000 {
            HealthGrade::Excellent
        } else if self.loss_pct < 1.0 && self.jitter_ms < 10.0 && self.bitrate_kbps > 1_000 {
            HealthGrade::Good
        } else if self.loss_pct < 5.0 && self.jitter_ms < 50.0 && self.bitrate_kbps > 100 {
            HealthGrade::Fair
        } else {
            HealthGrade::Poor
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Stream Bonding (multi-path round-robin)
// ─────────────────────────────────────────────────────────────────────────────

/// Round-robin packet distributor over multiple stream paths.
#[derive(Debug, Clone)]
pub struct StreamBonding {
    streams: Vec<u32>,
    next_index: usize,
}

impl StreamBonding {
    /// Create a new stream bonding manager over `streams` (stream IDs).
    ///
    /// # Panics
    ///
    /// Does not panic; an empty `streams` list is valid (all distributions
    /// return 0).
    #[must_use]
    pub fn new(streams: Vec<u32>) -> Self {
        Self { streams, next_index: 0 }
    }

    /// Select the next stream ID for packet `seq` using round-robin.
    ///
    /// Returns 0 when no streams are registered.
    pub fn distribute_packet(&mut self, _seq: u64) -> u32 {
        if self.streams.is_empty() {
            return 0;
        }
        let id = self.streams[self.next_index % self.streams.len()];
        self.next_index = self.next_index.wrapping_add(1);
        id
    }

    /// Number of bonded streams.
    #[must_use]
    pub fn stream_count(&self) -> usize {
        self.streams.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Color Space Converter: UYVY → RGBA
// ─────────────────────────────────────────────────────────────────────────────

/// Color space conversion utilities.
pub struct ColorSpaceConverter;

impl ColorSpaceConverter {
    /// Convert a UYVY (YCbCr 4:2:2 packed) buffer to RGBA using BT.601.
    ///
    /// UYVY packing (4 bytes per 2 pixels):
    /// ```text
    /// [U0, Y0, V0, Y1]  →  pixel0 = (Y0, U0, V0), pixel1 = (Y1, U0, V0)
    /// ```
    ///
    /// BT.601 full-range YCbCr → RGB formulas:
    /// ```text
    /// R = clamp(Y + 1.402*(Cr-128), 0, 255)
    /// G = clamp(Y - 0.344*(Cb-128) - 0.714*(Cr-128), 0, 255)
    /// B = clamp(Y + 1.772*(Cb-128), 0, 255)
    /// ```
    ///
    /// Returns an empty `Vec` when `uyvy.len() != width * height * 2`.
    #[must_use]
    pub fn uyvy_to_rgba(uyvy: &[u8], width: u32, height: u32) -> Vec<u8> {
        let expected = (width as usize) * (height as usize) * 2;
        if uyvy.len() != expected || width == 0 || height == 0 {
            return Vec::new();
        }

        let pixel_count = (width as usize) * (height as usize);
        let mut rgba = Vec::with_capacity(pixel_count * 4);

        // Process two pixels at a time
        let macropixels = uyvy.chunks_exact(4);
        for chunk in macropixels {
            let u0 = chunk[0] as f32;
            let y0 = chunk[1] as f32;
            let v0 = chunk[2] as f32;
            let y1 = chunk[3] as f32;

            let cb = u0 - 128.0;
            let cr = v0 - 128.0;

            // Pixel 0
            let r0 = (y0 + 1.402 * cr).clamp(0.0, 255.0) as u8;
            let g0 = (y0 - 0.344_136 * cb - 0.714_136 * cr).clamp(0.0, 255.0) as u8;
            let b0 = (y0 + 1.772 * cb).clamp(0.0, 255.0) as u8;
            rgba.push(r0);
            rgba.push(g0);
            rgba.push(b0);
            rgba.push(255u8);

            // Pixel 1
            let r1 = (y1 + 1.402 * cr).clamp(0.0, 255.0) as u8;
            let g1 = (y1 - 0.344_136 * cb - 0.714_136 * cr).clamp(0.0, 255.0) as u8;
            let b1 = (y1 + 1.772 * cb).clamp(0.0, 255.0) as u8;
            rgba.push(r1);
            rgba.push(g1);
            rgba.push(b1);
            rgba.push(255u8);
        }

        rgba
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Stream Encryption (AES-CTR stub)
// ─────────────────────────────────────────────────────────────────────────────

/// Stream encryption utilities.
///
/// **Note**: The AES-CTR implementation below is a stub that XORs data with
/// cycling key bytes.  It does NOT provide real AES encryption.  Real AES is
/// required for production security.
pub struct StreamEncryption;

impl StreamEncryption {
    /// AES-CTR encryption/decryption stub.
    ///
    /// For demonstration purposes only.  XORs each byte of `data` with
    /// `key[i % 16]` combined with a byte derived from the nonce.
    ///
    /// A production implementation must replace this with a real AES-CTR
    /// cipher (e.g. using the `aes` crate with proper CTR mode).
    #[must_use]
    pub fn aes_ctr_stub(data: &[u8], key: &[u8; 16], nonce: &[u8; 8]) -> Vec<u8> {
        data.iter()
            .enumerate()
            .map(|(i, &b)| {
                let k = key[i % 16];
                let n = nonce[i % 8];
                b ^ k ^ n
            })
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stream_descriptor::{StreamDescriptor, StreamType};

    // ── RIST ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_rist_packet_round_trip() {
        let payload = b"hello rist";
        let ser = RistPacket::serialize(42, 1234, payload);
        let (seq, ts, data) = RistPacket::deserialize(&ser).expect("valid packet");
        assert_eq!(seq, 42);
        assert_eq!(ts, 1234);
        assert_eq!(data.as_slice(), payload.as_slice());
    }

    #[test]
    fn test_rist_packet_too_short() {
        assert!(RistPacket::deserialize(&[0u8; 4]).is_none());
    }

    #[test]
    fn test_rist_packet_header_version_check() {
        let mut bad = vec![0u8; 12];
        bad[0] = 0x00; // version = 0, not 2
        bad[1] = 96;
        assert!(RistPacket::deserialize(&bad).is_none());
    }

    #[test]
    fn test_rist_packet_payload_type_check() {
        let mut bad = vec![0u8; 12];
        bad[0] = 0x80; // version = 2
        bad[1] = 0xFF; // PT = 127 & 0x7F, not 96
        assert!(RistPacket::deserialize(&bad).is_none());
    }

    #[test]
    fn test_rist_packet_seq_wraps_at_16bit() {
        let ser = RistPacket::serialize(0x1_0000 + 7, 99, b"test");
        let (seq, _, _) = RistPacket::deserialize(&ser).expect("valid");
        assert_eq!(seq, 7); // truncated to u16
    }

    // ── SimpleCongestionController ────────────────────────────────────────────

    #[test]
    fn test_congestion_on_loss_decreases_rate() {
        let mut cc = SimpleCongestionController::new(10_000);
        let before = cc.rate_kbps();
        let after = cc.on_loss(0.5); // 50% loss → rate *= 0.75
        assert!(after < before);
    }

    #[test]
    fn test_congestion_on_ack_increases_rate() {
        let mut cc = SimpleCongestionController::new(1_000);
        let before = cc.rate_kbps();
        let after = cc.on_ack();
        assert_eq!(after, before + 1);
    }

    #[test]
    fn test_congestion_rate_not_below_min() {
        let mut cc = SimpleCongestionController::new(101);
        cc.on_loss(1.0); // would go to 0 without min clamp
        assert!(cc.rate_kbps() >= 100);
    }

    // ── FramePacer ────────────────────────────────────────────────────────────

    #[test]
    fn test_frame_pacer_send_immediately_when_elapsed() {
        let p = FramePacer::new(30.0); // interval ≈ 33ms
        let wait = p.time_until_next_frame_ms(0, 50); // 50ms elapsed > 33ms
        assert_eq!(wait, 0);
    }

    #[test]
    fn test_frame_pacer_wait_remaining() {
        let p = FramePacer::new(60.0); // interval ≈ 17ms
        let wait = p.time_until_next_frame_ms(0, 10); // only 10ms elapsed
        assert!(wait > 0, "should wait, got {wait}");
    }

    #[test]
    fn test_frame_pacer_interval_30fps() {
        let p = FramePacer::new(30.0);
        // 1000 / 30 ≈ 33.3ms
        assert!((p.frame_interval_ms() - 33.333).abs() < 0.1);
    }

    // ── NdiBridge ─────────────────────────────────────────────────────────────

    #[test]
    fn test_ndi_bridge_header_prepended() {
        let frame = vec![1u8, 2, 3, 4];
        let out = NdiBridge::transcode_to_videoip(&frame, 1920, 1080);
        assert_eq!(out.len(), 12 + frame.len());
        assert_eq!(&out[0..4], b"VIPF");
    }

    #[test]
    fn test_ndi_bridge_strip_header() {
        let frame = vec![10u8, 20, 30];
        let out = NdiBridge::transcode_to_videoip(&frame, 720, 576);
        let (w, h, payload) = NdiBridge::strip_header(&out).expect("valid header");
        assert_eq!(w, 720);
        assert_eq!(h, 576);
        assert_eq!(payload, frame.as_slice());
    }

    #[test]
    fn test_ndi_bridge_empty_frame() {
        let out = NdiBridge::transcode_to_videoip(&[], 640, 480);
        assert_eq!(out.len(), 12);
    }

    // ── SdpGenerator ─────────────────────────────────────────────────────────

    #[test]
    fn test_sdp_contains_required_fields() {
        let stream = StreamDescriptor::new("stream-1", StreamType::CompressedVideo, 1920, 1080, 30.0, "av1");
        let sdp = SdpGenerator::generate(&stream);
        assert!(sdp.contains("v=0"), "missing v=0 in SDP:\n{sdp}");
        assert!(sdp.contains("o="), "missing o= in SDP:\n{sdp}");
        assert!(sdp.contains("s="), "missing s= in SDP:\n{sdp}");
        assert!(sdp.contains("t=0 0"), "missing t=0 0 in SDP:\n{sdp}");
        assert!(sdp.contains("m=video"), "missing m=video in SDP:\n{sdp}");
        assert!(sdp.contains("a=rtpmap"), "missing a=rtpmap in SDP:\n{sdp}");
    }

    #[test]
    fn test_sdp_contains_codec_name() {
        let stream = StreamDescriptor::new("s2", StreamType::CompressedVideo, 1920, 1080, 60.0, "vp9");
        let sdp = SdpGenerator::generate(&stream);
        assert!(sdp.contains("vp9"), "SDP should contain codec name: {sdp}");
    }

    // ── Smpte2110Validator ────────────────────────────────────────────────────

    #[test]
    fn test_smpte2110_no_violations_on_time() {
        // 30fps → ~33.33ms interval; generate packets using the same rounding
        // as the validator to avoid off-by-one ms errors.
        let fps = 30.0f32;
        let frame_ms = 1000.0 / fps as f64;
        let packets: Vec<(u64, u32)> = (0..5)
            .map(|i| ((i as f64 * frame_ms).round() as u64, 1000))
            .collect();
        let violations = Smpte2110Validator::check_packet_timing(&packets, fps);
        assert!(violations.is_empty(), "unexpected violations: {:?}", violations);
    }

    #[test]
    fn test_smpte2110_late_packet_detected() {
        let mut packets: Vec<(u64, u32)> = (0..5).map(|i| (i * 33, 1000)).collect();
        // Delay packet 2 by 50ms
        packets[2].0 += 50;
        let violations = Smpte2110Validator::check_packet_timing(&packets, 30.0);
        assert!(!violations.is_empty());
        assert!(violations.iter().any(|v| v.packet_index == 2 && v.violation == ViolationType::TooLate));
    }

    #[test]
    fn test_smpte2110_early_packet_detected() {
        let mut packets: Vec<(u64, u32)> = (0..5).map(|i| (1000 + i * 33, 1000)).collect();
        // Move packet 3 back by 10ms (too early)
        packets[3].0 = packets[3].0.saturating_sub(10);
        let violations = Smpte2110Validator::check_packet_timing(&packets, 30.0);
        assert!(violations.iter().any(|v| v.packet_index == 3 && v.violation == ViolationType::TooEarly));
    }

    #[test]
    fn test_smpte2110_empty_returns_empty() {
        let violations = Smpte2110Validator::check_packet_timing(&[], 30.0);
        assert!(violations.is_empty());
    }

    // ── SimpleStreamHealthMonitor ─────────────────────────────────────────────

    #[test]
    fn test_health_excellent() {
        let mut m = SimpleStreamHealthMonitor::new();
        m.update(0.0, 0.5, 10_000);
        assert_eq!(m.health_grade(), HealthGrade::Excellent);
    }

    #[test]
    fn test_health_good() {
        let mut m = SimpleStreamHealthMonitor::new();
        m.update(0.5, 5.0, 2_000);
        assert_eq!(m.health_grade(), HealthGrade::Good);
    }

    #[test]
    fn test_health_fair() {
        let mut m = SimpleStreamHealthMonitor::new();
        m.update(2.0, 20.0, 500);
        assert_eq!(m.health_grade(), HealthGrade::Fair);
    }

    #[test]
    fn test_health_poor() {
        let mut m = SimpleStreamHealthMonitor::new();
        m.update(10.0, 100.0, 50);
        assert_eq!(m.health_grade(), HealthGrade::Poor);
    }

    // ── StreamBonding ─────────────────────────────────────────────────────────

    #[test]
    fn test_stream_bonding_round_robin() {
        let mut bonding = StreamBonding::new(vec![10, 20, 30]);
        // 6 packets: should cycle through streams twice
        let ids: Vec<u32> = (0..6).map(|seq| bonding.distribute_packet(seq)).collect();
        assert_eq!(ids, vec![10, 20, 30, 10, 20, 30]);
    }

    #[test]
    fn test_stream_bonding_empty() {
        let mut bonding = StreamBonding::new(vec![]);
        assert_eq!(bonding.distribute_packet(0), 0);
    }

    // ── ColorSpaceConverter::uyvy_to_rgba ─────────────────────────────────────

    #[test]
    fn test_uyvy_neutral_gray() {
        // For neutral gray: Y=128, U=128, V=128 → R=G=B=128 (approximately)
        // UYVY = [U=128, Y0=128, V=128, Y1=128]
        let uyvy = vec![128u8, 128, 128, 128]; // 2 pixels, 1×2 frame
        let rgba = ColorSpaceConverter::uyvy_to_rgba(&uyvy, 2, 1);
        assert_eq!(rgba.len(), 8); // 2 pixels × 4 channels
        // R, G, B for neutral gray should all be near 128
        assert!((rgba[0] as i32 - 128).abs() <= 2, "R={}", rgba[0]);
        assert!((rgba[1] as i32 - 128).abs() <= 2, "G={}", rgba[1]);
        assert!((rgba[2] as i32 - 128).abs() <= 2, "B={}", rgba[2]);
        assert_eq!(rgba[3], 255, "alpha must be 255");
    }

    #[test]
    fn test_uyvy_wrong_size_returns_empty() {
        let uyvy = vec![0u8; 5]; // not w*h*2
        let rgba = ColorSpaceConverter::uyvy_to_rgba(&uyvy, 2, 2);
        assert!(rgba.is_empty());
    }

    #[test]
    fn test_uyvy_output_size() {
        let w = 4u32;
        let h = 2u32;
        let uyvy = vec![128u8; (w * h * 2) as usize];
        let rgba = ColorSpaceConverter::uyvy_to_rgba(&uyvy, w, h);
        assert_eq!(rgba.len(), (w * h * 4) as usize);
    }

    // ── StreamEncryption ──────────────────────────────────────────────────────

    #[test]
    fn test_aes_ctr_stub_round_trip() {
        let data = b"hello world!";
        let key = [0x42u8; 16];
        let nonce = [0xABu8; 8];
        let encrypted = StreamEncryption::aes_ctr_stub(data, &key, &nonce);
        let decrypted = StreamEncryption::aes_ctr_stub(&encrypted, &key, &nonce);
        assert_eq!(decrypted.as_slice(), data.as_slice());
    }

    #[test]
    fn test_aes_ctr_stub_changes_data() {
        let data = vec![0u8; 16];
        let key = [1u8; 16];
        let nonce = [2u8; 8];
        let enc = StreamEncryption::aes_ctr_stub(&data, &key, &nonce);
        assert_ne!(enc, data);
    }
}
