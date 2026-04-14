//! SMPTE ST 2110-20 RTP packetizer.
//!
//! Splits an uncompressed video frame into a sequence of ST 2110-20 RTP
//! packets sized to fit inside a standard Ethernet MTU (1500 bytes minus
//! IP/UDP/RTP overhead ≈ 1428 bytes of usable payload per packet).
//!
//! # Packet Format
//!
//! Each packet carries a minimal RTP-style header followed by one or more
//! ST 2110-20 row-extension words:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │  Byte  0..2  │  seq_num (u16 big-endian)                        │
//! │  Byte  2..6  │  timestamp (u32 big-endian, 90 kHz clock)        │
//! │  Byte  6..8  │  line_num (u16 big-endian)                       │
//! │  Byte  8..10 │  pixel_offset (u16 big-endian)                   │
//! │  Byte  10    │  flags: bit7 = continuation                      │
//! │  Byte  11    │  reserved                                        │
//! │  Byte  12..  │  payload (raw YCbCr 4:2:2 8-bit)                 │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

#![allow(dead_code)]

use crate::error::{VideoIpError, VideoIpResult};

/// Maximum payload bytes per ST 2110-20 RTP packet.
///
/// Leaves headroom for IP (20) + UDP (8) + RTP (12) + row extension (6)
/// headers inside a 1500-byte Ethernet MTU.
pub const MAX_PAYLOAD_BYTES: usize = 1428;

/// Bytes per 2 pixels of YCbCr 4:2:2 8-bit.
const YUV422_8_BYTES_PER_2PIX: usize = 4;

// ─── RtpPacket ───────────────────────────────────────────────────────────────

/// A single ST 2110-20 RTP packet produced by the packetizer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RtpPacket {
    /// RTP sequence number (wraps at 65535).
    pub seq_num: u16,
    /// RTP timestamp (90 kHz clock).
    pub timestamp: u32,
    /// Video line number (0-based).
    pub line_num: u16,
    /// Pixel offset within the line where this payload begins.
    pub pixel_offset: u16,
    /// `true` when more packets for the same line follow.
    pub continuation: bool,
    /// Raw YCbCr payload bytes for this packet.
    pub payload: Vec<u8>,
}

impl RtpPacket {
    /// Serialises the packet to a byte vector.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(12 + self.payload.len());
        out.extend_from_slice(&self.seq_num.to_be_bytes());
        out.extend_from_slice(&self.timestamp.to_be_bytes());
        out.extend_from_slice(&self.line_num.to_be_bytes());
        out.extend_from_slice(&self.pixel_offset.to_be_bytes());
        let flags: u8 = if self.continuation { 0x80 } else { 0x00 };
        out.push(flags);
        out.push(0x00); // reserved
        out.extend_from_slice(&self.payload);
        out
    }

    /// Deserialises a packet from bytes.
    ///
    /// Returns `None` when `data` is shorter than the 12-byte header.
    #[must_use]
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        if data.len() < 12 {
            return None;
        }
        let seq_num = u16::from_be_bytes([data[0], data[1]]);
        let timestamp = u32::from_be_bytes([data[2], data[3], data[4], data[5]]);
        let line_num = u16::from_be_bytes([data[6], data[7]]);
        let pixel_offset = u16::from_be_bytes([data[8], data[9]]);
        let continuation = (data[10] & 0x80) != 0;
        let payload = data[12..].to_vec();
        Some(Self {
            seq_num,
            timestamp,
            line_num,
            pixel_offset,
            continuation,
            payload,
        })
    }
}

// ─── Rtp2110Packetizer ───────────────────────────────────────────────────────

/// ST 2110-20 RTP packetizer.
///
/// Splits an uncompressed YCbCr 4:2:2 8-bit video frame into a series of
/// [`RtpPacket`]s whose payload fits within the Ethernet MTU.
///
/// # Usage
///
/// ```rust
/// use oximedia_videoip::rtp_2110::Rtp2110Packetizer;
///
/// let mut pkt = Rtp2110Packetizer::new(1920, 1080).expect("valid dims");
/// let frame = vec![0u8; 1920 * 1080 * 2]; // YCbCr 4:2:2 = 2 bytes/px
/// let packets = pkt.packetize(&frame).expect("ok");
/// assert!(!packets.is_empty());
/// ```
#[derive(Debug)]
pub struct Rtp2110Packetizer {
    /// Frame width in pixels.
    line_w: u32,
    /// Frame height in lines.
    line_h: u32,
    /// Bytes per pixel row (YCbCr 4:2:2 → 2 bytes/px).
    bytes_per_line: usize,
    /// Rolling RTP sequence number.
    seq: u16,
    /// Rolling RTP timestamp (90 kHz).
    timestamp: u32,
}

impl Rtp2110Packetizer {
    /// Creates a new packetizer for a frame of `line_w × line_h` pixels.
    ///
    /// # Errors
    ///
    /// Returns [`VideoIpError::InvalidConfig`] when either dimension is zero.
    pub fn new(line_w: u32, line_h: u32) -> VideoIpResult<Self> {
        if line_w == 0 || line_h == 0 {
            return Err(VideoIpError::InvalidVideoConfig(
                "frame dimensions must be > 0".into(),
            ));
        }
        // YCbCr 4:2:2 8-bit: 2 bytes per pixel.
        let bytes_per_line = line_w as usize * 2;
        Ok(Self {
            line_w,
            line_h,
            bytes_per_line,
            seq: 0,
            timestamp: 0,
        })
    }

    /// Packetizes `frame` into a `Vec<RtpPacket>`.
    ///
    /// `frame` must be exactly `line_w × line_h × 2` bytes.  Each call
    /// advances the internal sequence counter and timestamp by one frame
    /// period (3003 ticks at 90 kHz ≈ 29.97 fps).
    ///
    /// # Errors
    ///
    /// Returns [`VideoIpError::InvalidPacket`] when `frame.len()` does not
    /// match the expected frame size.
    pub fn packetize(&mut self, frame: &[u8]) -> VideoIpResult<Vec<RtpPacket>> {
        let expected = self.line_h as usize * self.bytes_per_line;
        if frame.len() != expected {
            return Err(VideoIpError::InvalidPacket(format!(
                "frame size mismatch: got {} bytes, expected {}",
                frame.len(),
                expected
            )));
        }

        let ts = self.timestamp;
        let mut packets = Vec::new();

        for line in 0..self.line_h {
            let line_start = line as usize * self.bytes_per_line;
            let line_data = &frame[line_start..line_start + self.bytes_per_line];

            // Each pixel consumes 2 bytes; chunk by pixel pairs.
            let mut pixel_offset: u16 = 0;
            let mut byte_offset = 0usize;

            while byte_offset < line_data.len() {
                let remaining = line_data.len() - byte_offset;
                let chunk_bytes = remaining.min(MAX_PAYLOAD_BYTES);
                // Align chunk to 4-byte (2-pixel) boundary for YCbCr 4:2:2.
                let chunk_bytes = (chunk_bytes / YUV422_8_BYTES_PER_2PIX) * YUV422_8_BYTES_PER_2PIX;
                let chunk_bytes = if chunk_bytes == 0 { remaining } else { chunk_bytes };

                let is_last_chunk = byte_offset + chunk_bytes >= line_data.len();
                let continuation = !is_last_chunk;

                let payload = line_data[byte_offset..byte_offset + chunk_bytes].to_vec();

                packets.push(RtpPacket {
                    seq_num: self.seq,
                    timestamp: ts,
                    line_num: line as u16,
                    pixel_offset,
                    continuation,
                    payload,
                });

                self.seq = self.seq.wrapping_add(1);
                pixel_offset = pixel_offset.wrapping_add((chunk_bytes / 2) as u16);
                byte_offset += chunk_bytes;
            }
        }

        // Advance timestamp by one 29.97 fps frame period (3003 ticks @ 90 kHz).
        self.timestamp = self.timestamp.wrapping_add(3003);
        Ok(packets)
    }

    /// Returns the current RTP timestamp (before the next call to [`packetize`]).
    #[must_use]
    pub fn current_timestamp(&self) -> u32 {
        self.timestamp
    }

    /// Returns the current sequence number.
    #[must_use]
    pub fn current_seq(&self) -> u16 {
        self.seq
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_dimensions_rejected() {
        assert!(Rtp2110Packetizer::new(0, 1080).is_err());
        assert!(Rtp2110Packetizer::new(1920, 0).is_err());
    }

    #[test]
    fn wrong_frame_size_rejected() {
        let mut p = Rtp2110Packetizer::new(4, 2).expect("valid");
        let wrong = vec![0u8; 1]; // 4×2×2=16 expected
        assert!(p.packetize(&wrong).is_err());
    }

    #[test]
    fn minimal_frame_produces_packets() {
        let mut p = Rtp2110Packetizer::new(4, 2).expect("valid"); // 4px wide, 2 lines
        let frame = vec![0u8; 4 * 2 * 2]; // 16 bytes
        let pkts = p.packetize(&frame).expect("ok");
        // 2 lines → at least 2 packets
        assert!(pkts.len() >= 2);
    }

    #[test]
    fn packet_payload_reassembles_frame() {
        let w = 8u32;
        let h = 2u32;
        let frame: Vec<u8> = (0..(w * h * 2)).map(|i| (i % 256) as u8).collect();
        let mut p = Rtp2110Packetizer::new(w, h).expect("valid");
        let pkts = p.packetize(&frame).expect("ok");

        let mut reassembled = vec![0u8; frame.len()];
        for pkt in &pkts {
            let line_start = pkt.line_num as usize * w as usize * 2;
            let byte_off = pkt.pixel_offset as usize * 2;
            let dst_start = line_start + byte_off;
            reassembled[dst_start..dst_start + pkt.payload.len()].copy_from_slice(&pkt.payload);
        }
        assert_eq!(reassembled, frame);
    }

    #[test]
    fn sequence_numbers_increment() {
        let mut p = Rtp2110Packetizer::new(4, 4).expect("valid");
        let frame = vec![0u8; 4 * 4 * 2];
        let pkts = p.packetize(&frame).expect("ok");
        for (i, pkt) in pkts.iter().enumerate() {
            assert_eq!(pkt.seq_num, i as u16);
        }
    }

    #[test]
    fn timestamp_advances_between_frames() {
        let mut p = Rtp2110Packetizer::new(4, 2).expect("valid");
        let frame = vec![0u8; 4 * 2 * 2];
        let ts_before = p.current_timestamp();
        let _ = p.packetize(&frame).expect("ok");
        let ts_after = p.current_timestamp();
        assert_eq!(ts_after, ts_before.wrapping_add(3003));
    }

    #[test]
    fn serialise_deserialise_roundtrip() {
        let pkt = RtpPacket {
            seq_num: 42,
            timestamp: 123456,
            line_num: 7,
            pixel_offset: 16,
            continuation: true,
            payload: vec![0xAA, 0xBB, 0xCC, 0xDD],
        };
        let bytes = pkt.to_bytes();
        let back = RtpPacket::from_bytes(&bytes).expect("valid");
        assert_eq!(back, pkt);
    }

    #[test]
    fn from_bytes_too_short_returns_none() {
        assert!(RtpPacket::from_bytes(&[0u8; 11]).is_none());
    }
}
