//! RTCP Sender Report and Receiver Report packet parsing and building.
//!
//! Implements RFC 3550 RTCP SR (Sender Report) and RR (Receiver Report) packets,
//! including NTP/RTP timestamp mapping and fraction-lost calculation.

use crate::error::{VideoIpError, VideoIpResult};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// NTP timestamp (RFC 3550 §4): 64-bit fixed-point number in seconds since
/// 1 Jan 1900.  Upper 32 bits = whole seconds, lower 32 bits = fractional part.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct NtpTimestamp {
    /// Seconds since NTP epoch (1 Jan 1900).
    pub seconds: u32,
    /// Fractional seconds (units of 2^-32 s).
    pub fraction: u32,
}

/// Difference between Unix epoch (1970-01-01) and NTP epoch (1900-01-01) in
/// seconds.
const NTP_UNIX_OFFSET_SECS: u64 = 70 * 365 * 24 * 3600 + 17 * 24 * 3600; // 2_208_988_800

impl NtpTimestamp {
    /// Constructs an `NtpTimestamp` from raw 64-bit NTP value (big-endian).
    #[must_use]
    pub const fn from_u64(raw: u64) -> Self {
        Self {
            seconds: (raw >> 32) as u32,
            fraction: raw as u32,
        }
    }

    /// Returns the raw 64-bit NTP value.
    #[must_use]
    pub const fn to_u64(self) -> u64 {
        ((self.seconds as u64) << 32) | (self.fraction as u64)
    }

    /// Computes the NTP timestamp corresponding to the current wall-clock time.
    ///
    /// # Errors
    ///
    /// Returns [`VideoIpError::InvalidState`] when the system clock is before
    /// the Unix epoch.
    pub fn now() -> VideoIpResult<Self> {
        let since_epoch = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| VideoIpError::InvalidState(format!("system clock error: {e}")))?;
        Ok(Self::from_unix_duration(since_epoch))
    }

    /// Builds an `NtpTimestamp` from a `Duration` measured from the Unix epoch.
    #[must_use]
    pub fn from_unix_duration(d: Duration) -> Self {
        let ntp_secs = d.as_secs() + NTP_UNIX_OFFSET_SECS;
        // Scale sub-second part: fraction = nanos * 2^32 / 1_000_000_000
        let fraction = ((d.subsec_nanos() as u64 * (1u64 << 32)) / 1_000_000_000) as u32;
        Self {
            seconds: ntp_secs as u32,
            fraction,
        }
    }

    /// Converts this NTP timestamp back to a `Duration` from the Unix epoch.
    ///
    /// Returns `None` when the timestamp pre-dates the Unix epoch.
    #[must_use]
    pub fn to_unix_duration(self) -> Option<Duration> {
        let secs = (self.seconds as u64).checked_sub(NTP_UNIX_OFFSET_SECS)?;
        let nanos = (self.fraction as u64 * 1_000_000_000) >> 32;
        Some(Duration::new(secs, nanos as u32))
    }

    /// Serialises the timestamp into a big-endian byte buffer (8 bytes).
    pub fn write_to(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&self.seconds.to_be_bytes());
        buf.extend_from_slice(&self.fraction.to_be_bytes());
    }

    /// Deserialises from an 8-byte big-endian slice.
    ///
    /// # Errors
    ///
    /// Returns [`VideoIpError::InvalidPacket`] when `data` is shorter than 8
    /// bytes.
    pub fn read_from(data: &[u8]) -> VideoIpResult<Self> {
        if data.len() < 8 {
            return Err(VideoIpError::InvalidPacket(format!(
                "NTP timestamp needs 8 bytes, got {}",
                data.len()
            )));
        }
        let seconds = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
        let fraction = u32::from_be_bytes([data[4], data[5], data[6], data[7]]);
        Ok(Self { seconds, fraction })
    }
}

/// Maps an RTP timestamp to an NTP timestamp using a known anchor point.
///
/// Given an anchor pair `(rtp_anchor, ntp_anchor)` and a clock rate in Hz,
/// computes the NTP timestamp for any `rtp_timestamp`.
///
/// # Errors
///
/// Returns [`VideoIpError::InvalidState`] when `clock_rate` is zero.
pub fn rtp_to_ntp(
    rtp_timestamp: u32,
    rtp_anchor: u32,
    ntp_anchor: NtpTimestamp,
    clock_rate: u32,
) -> VideoIpResult<NtpTimestamp> {
    if clock_rate == 0 {
        return Err(VideoIpError::InvalidState(
            "clock rate must be non-zero".to_string(),
        ));
    }
    // Signed RTP delta (handles wrap-around correctly for 32-bit counters).
    let delta_rtp = rtp_timestamp.wrapping_sub(rtp_anchor) as i32;
    let delta_ntp_frac: i64 = (delta_rtp as i64 * (1i64 << 32)) / clock_rate as i64;
    let raw_ntp = (ntp_anchor.to_u64() as i64).wrapping_add(delta_ntp_frac);
    Ok(NtpTimestamp::from_u64(raw_ntp as u64))
}

/// Maps an NTP timestamp to an RTP timestamp using a known anchor point.
///
/// # Errors
///
/// Returns [`VideoIpError::InvalidState`] when `clock_rate` is zero.
pub fn ntp_to_rtp(
    ntp: NtpTimestamp,
    rtp_anchor: u32,
    ntp_anchor: NtpTimestamp,
    clock_rate: u32,
) -> VideoIpResult<u32> {
    if clock_rate == 0 {
        return Err(VideoIpError::InvalidState(
            "clock rate must be non-zero".to_string(),
        ));
    }
    let delta_ntp = (ntp.to_u64() as i64).wrapping_sub(ntp_anchor.to_u64() as i64);
    let delta_rtp = (delta_ntp * clock_rate as i64) >> 32;
    Ok(rtp_anchor.wrapping_add(delta_rtp as u32))
}

// ─────────────────────────────────────────────────────────────────────────────
// Report Block (RFC 3550 §6.4)
// ─────────────────────────────────────────────────────────────────────────────

/// Single reception-report block as carried in SR and RR packets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReportBlock {
    /// SSRC of the source this block applies to.
    pub ssrc: u32,
    /// Fraction of packets lost since last SR/RR (8-bit fixed-point 0..=255).
    pub fraction_lost: u8,
    /// Cumulative number of packets lost (signed 24-bit, saturating).
    pub cumulative_lost: i32,
    /// Extended highest sequence number received.
    pub extended_highest_seq: u32,
    /// Inter-arrival jitter estimate (in RTP timestamp units).
    pub jitter: u32,
    /// Last SR timestamp (compact NTP: middle 32 bits of last received SR's
    /// NTP timestamp).
    pub last_sr: u32,
    /// Delay since last SR in units of 1/65536 seconds.
    pub delay_since_last_sr: u32,
}

impl ReportBlock {
    /// Wire size of a single report block (bytes).
    pub const WIRE_SIZE: usize = 24;

    /// Calculates the `fraction_lost` field from packet counters.
    ///
    /// - `expected` — packets expected in this interval.
    /// - `received` — packets actually received in this interval.
    ///
    /// Returns 0 when `expected` is zero (nothing was expected).
    #[must_use]
    pub fn compute_fraction_lost(expected: u32, received: u32) -> u8 {
        if expected == 0 {
            return 0;
        }
        let lost = expected.saturating_sub(received);
        // fraction_lost = lost/expected * 256, clamped to [0, 255]
        let fraction = (lost as u64 * 256) / expected as u64;
        fraction.min(255) as u8
    }

    /// Serialises this block into `buf` (big-endian, 24 bytes).
    pub fn write_to(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&self.ssrc.to_be_bytes());
        // Byte 4: fraction_lost; bytes 5-7: cumulative_lost (24-bit signed)
        let cl_clamped = self.cumulative_lost.clamp(-(1 << 23), (1 << 23) - 1);
        let cl_bits = (cl_clamped as u32) & 0x00FF_FFFF;
        buf.push(self.fraction_lost);
        buf.push((cl_bits >> 16) as u8);
        buf.push((cl_bits >> 8) as u8);
        buf.push(cl_bits as u8);
        buf.extend_from_slice(&self.extended_highest_seq.to_be_bytes());
        buf.extend_from_slice(&self.jitter.to_be_bytes());
        buf.extend_from_slice(&self.last_sr.to_be_bytes());
        buf.extend_from_slice(&self.delay_since_last_sr.to_be_bytes());
    }

    /// Deserialises a report block from a 24-byte slice.
    ///
    /// # Errors
    ///
    /// Returns [`VideoIpError::InvalidPacket`] when `data` is shorter than 24
    /// bytes.
    pub fn read_from(data: &[u8]) -> VideoIpResult<Self> {
        if data.len() < Self::WIRE_SIZE {
            return Err(VideoIpError::InvalidPacket(format!(
                "report block needs {} bytes, got {}",
                Self::WIRE_SIZE,
                data.len()
            )));
        }
        let ssrc = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
        let fraction_lost = data[4];
        // 24-bit signed cumulative lost (sign-extend from bit 23)
        let raw_cl = ((data[5] as u32) << 16) | ((data[6] as u32) << 8) | (data[7] as u32);
        let cumulative_lost = if raw_cl & 0x0080_0000 != 0 {
            (raw_cl | 0xFF00_0000) as i32
        } else {
            raw_cl as i32
        };
        let extended_highest_seq = u32::from_be_bytes([data[8], data[9], data[10], data[11]]);
        let jitter = u32::from_be_bytes([data[12], data[13], data[14], data[15]]);
        let last_sr = u32::from_be_bytes([data[16], data[17], data[18], data[19]]);
        let delay_since_last_sr = u32::from_be_bytes([data[20], data[21], data[22], data[23]]);
        Ok(Self {
            ssrc,
            fraction_lost,
            cumulative_lost,
            extended_highest_seq,
            jitter,
            last_sr,
            delay_since_last_sr,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RTCP Common Header
// ─────────────────────────────────────────────────────────────────────────────

/// RTCP packet type codes (RFC 3550 §6.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum RtcpPacketType {
    /// Sender Report (SR).
    SenderReport = 200,
    /// Receiver Report (RR).
    ReceiverReport = 201,
    /// Source Description (SDES).
    SourceDescription = 202,
    /// Goodbye (BYE).
    Goodbye = 203,
    /// Application-defined.
    ApplicationDefined = 204,
}

impl RtcpPacketType {
    /// Converts a raw byte to a known packet type, if recognised.
    #[must_use]
    pub const fn from_u8(v: u8) -> Option<Self> {
        match v {
            200 => Some(Self::SenderReport),
            201 => Some(Self::ReceiverReport),
            202 => Some(Self::SourceDescription),
            203 => Some(Self::Goodbye),
            204 => Some(Self::ApplicationDefined),
            _ => None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RTCP Sender Report (RFC 3550 §6.4.1)
// ─────────────────────────────────────────────────────────────────────────────

/// RTCP Sender Report packet.
///
/// ```text
///  0                   1                   2                   3
///  0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
/// +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
/// |V=2|P|    RC   |   PT=SR=200   |             length            |
/// +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
/// |                         SSRC of sender                        |
/// +=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+
/// |              NTP timestamp, most significant word             |
/// +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
/// |             NTP timestamp, least significant word             |
/// +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
/// |                         RTP timestamp                         |
/// +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
/// |                     sender's packet count                     |
/// +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
/// |                      sender's octet count                     |
/// +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
/// |                 report block(s) (RC entries)                  |
/// +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SenderReport {
    /// SSRC of the sender.
    pub ssrc: u32,
    /// NTP timestamp at the moment this SR was sent.
    pub ntp_timestamp: NtpTimestamp,
    /// RTP timestamp corresponding to the NTP timestamp.
    pub rtp_timestamp: u32,
    /// Total RTP data packets sent by this sender since session start.
    pub sender_packet_count: u32,
    /// Total payload octets sent by this sender since session start (excludes
    /// headers and padding).
    pub sender_octet_count: u32,
    /// Up to 31 reception report blocks.
    pub report_blocks: Vec<ReportBlock>,
    /// Whether the padding bit is set.
    pub padding: bool,
}

impl SenderReport {
    /// Fixed sender-info part of the payload (bytes, excluding common header).
    const SENDER_INFO_SIZE: usize = 4 + 8 + 4 + 4 + 4; // ssrc + ntp + rtp + pkt_cnt + oct_cnt

    /// Maximum number of report blocks per RFC 3550 (RC field is 5 bits).
    pub const MAX_REPORT_BLOCKS: usize = 31;

    /// Builds a new `SenderReport`.
    ///
    /// # Errors
    ///
    /// Returns [`VideoIpError::InvalidPacket`] when more than 31 report blocks
    /// are supplied.
    pub fn new(
        ssrc: u32,
        ntp_timestamp: NtpTimestamp,
        rtp_timestamp: u32,
        sender_packet_count: u32,
        sender_octet_count: u32,
        report_blocks: Vec<ReportBlock>,
    ) -> VideoIpResult<Self> {
        if report_blocks.len() > Self::MAX_REPORT_BLOCKS {
            return Err(VideoIpError::InvalidPacket(format!(
                "too many report blocks: {} (max {})",
                report_blocks.len(),
                Self::MAX_REPORT_BLOCKS,
            )));
        }
        Ok(Self {
            ssrc,
            ntp_timestamp,
            rtp_timestamp,
            sender_packet_count,
            sender_octet_count,
            report_blocks,
            padding: false,
        })
    }

    /// Serialises this SR into a `Vec<u8>` (big-endian, RFC 3550 wire format).
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let rc = self.report_blocks.len() as u8;
        // payload = ssrc(4) + ntp(8) + rtp_ts(4) + pkt_cnt(4) + oct_cnt(4) + rc*24
        let payload_words =
            (Self::SENDER_INFO_SIZE + rc as usize * ReportBlock::WIRE_SIZE) / 4 + 1; // +1 for SSRC word before ntp
        // Actually the length field counts 32-bit words MINUS ONE.
        // Payload after common header: ssrc(1w) + sender info(5w) + rc*6w
        let length_field = (1 + 5 + rc as usize * 6) as u16;

        let mut buf = Vec::with_capacity(4 + payload_words * 4);
        // Common header byte 0: V=2, P, RC
        let v_p_rc: u8 = (2u8 << 6) | (u8::from(self.padding) << 5) | rc;
        buf.push(v_p_rc);
        buf.push(200u8); // PT = SR
        buf.extend_from_slice(&length_field.to_be_bytes());
        buf.extend_from_slice(&self.ssrc.to_be_bytes());
        self.ntp_timestamp.write_to(&mut buf);
        buf.extend_from_slice(&self.rtp_timestamp.to_be_bytes());
        buf.extend_from_slice(&self.sender_packet_count.to_be_bytes());
        buf.extend_from_slice(&self.sender_octet_count.to_be_bytes());
        for block in &self.report_blocks {
            block.write_to(&mut buf);
        }
        buf
    }

    /// Parses a `SenderReport` from a raw byte slice (including the 4-byte
    /// common header).
    ///
    /// # Errors
    ///
    /// Returns [`VideoIpError::InvalidPacket`] on any format violation.
    pub fn decode(data: &[u8]) -> VideoIpResult<Self> {
        // Minimum: 4 (common header) + 4 (ssrc) + 8 (ntp) + 4 (rtp_ts) + 4 (pkt) + 4 (oct) = 28
        if data.len() < 28 {
            return Err(VideoIpError::InvalidPacket(format!(
                "SR packet too short: {} bytes",
                data.len()
            )));
        }
        let v = (data[0] >> 6) & 0x03;
        if v != 2 {
            return Err(VideoIpError::InvalidPacket(format!(
                "unsupported RTP version: {v}"
            )));
        }
        let padding = (data[0] & 0x20) != 0;
        let rc = (data[0] & 0x1F) as usize;
        let pt = data[1];
        if pt != 200 {
            return Err(VideoIpError::InvalidPacket(format!(
                "expected PT=200 (SR), got {pt}"
            )));
        }
        let length = u16::from_be_bytes([data[2], data[3]]) as usize;
        let expected_bytes = (length + 1) * 4;
        if data.len() < expected_bytes {
            return Err(VideoIpError::InvalidPacket(format!(
                "declared length {length} words requires {expected_bytes} bytes, have {}",
                data.len()
            )));
        }
        let ssrc = u32::from_be_bytes([data[4], data[5], data[6], data[7]]);
        let ntp_timestamp = NtpTimestamp::read_from(&data[8..])?;
        let rtp_timestamp = u32::from_be_bytes([data[16], data[17], data[18], data[19]]);
        let sender_packet_count = u32::from_be_bytes([data[20], data[21], data[22], data[23]]);
        let sender_octet_count = u32::from_be_bytes([data[24], data[25], data[26], data[27]]);
        // Parse report blocks starting at offset 28.
        let mut offset = 28usize;
        let mut report_blocks = Vec::with_capacity(rc);
        for _ in 0..rc {
            if offset + ReportBlock::WIRE_SIZE > data.len() {
                return Err(VideoIpError::InvalidPacket(
                    "truncated report block".to_string(),
                ));
            }
            let block = ReportBlock::read_from(&data[offset..])?;
            report_blocks.push(block);
            offset += ReportBlock::WIRE_SIZE;
        }
        Ok(Self {
            ssrc,
            ntp_timestamp,
            rtp_timestamp,
            sender_packet_count,
            sender_octet_count,
            report_blocks,
            padding,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RTCP Receiver Report (RFC 3550 §6.4.2)
// ─────────────────────────────────────────────────────────────────────────────

/// RTCP Receiver Report packet.
///
/// Sent by participants that are not active senders. Has the same structure as
/// SR but without the 5-word sender info block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReceiverReport {
    /// SSRC of the receiver sending this report.
    pub ssrc: u32,
    /// Reception report blocks (up to 31).
    pub report_blocks: Vec<ReportBlock>,
    /// Whether the padding bit is set.
    pub padding: bool,
}

impl ReceiverReport {
    /// Maximum number of report blocks per RFC 3550.
    pub const MAX_REPORT_BLOCKS: usize = 31;

    /// Builds a new `ReceiverReport`.
    ///
    /// # Errors
    ///
    /// Returns [`VideoIpError::InvalidPacket`] when more than 31 report blocks
    /// are supplied.
    pub fn new(ssrc: u32, report_blocks: Vec<ReportBlock>) -> VideoIpResult<Self> {
        if report_blocks.len() > Self::MAX_REPORT_BLOCKS {
            return Err(VideoIpError::InvalidPacket(format!(
                "too many report blocks: {} (max {})",
                report_blocks.len(),
                Self::MAX_REPORT_BLOCKS,
            )));
        }
        Ok(Self {
            ssrc,
            report_blocks,
            padding: false,
        })
    }

    /// Serialises this RR into a `Vec<u8>`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let rc = self.report_blocks.len() as u8;
        let length_field = (1 + rc as usize * 6) as u16; // ssrc(1w) + rc*6w

        let mut buf = Vec::with_capacity(4 + 4 + rc as usize * ReportBlock::WIRE_SIZE);
        let v_p_rc: u8 = (2u8 << 6) | (u8::from(self.padding) << 5) | rc;
        buf.push(v_p_rc);
        buf.push(201u8); // PT = RR
        buf.extend_from_slice(&length_field.to_be_bytes());
        buf.extend_from_slice(&self.ssrc.to_be_bytes());
        for block in &self.report_blocks {
            block.write_to(&mut buf);
        }
        buf
    }

    /// Parses a `ReceiverReport` from a raw byte slice.
    ///
    /// # Errors
    ///
    /// Returns [`VideoIpError::InvalidPacket`] on any format violation.
    pub fn decode(data: &[u8]) -> VideoIpResult<Self> {
        // Minimum: 4 (header) + 4 (ssrc) = 8
        if data.len() < 8 {
            return Err(VideoIpError::InvalidPacket(format!(
                "RR packet too short: {} bytes",
                data.len()
            )));
        }
        let v = (data[0] >> 6) & 0x03;
        if v != 2 {
            return Err(VideoIpError::InvalidPacket(format!(
                "unsupported RTP version: {v}"
            )));
        }
        let padding = (data[0] & 0x20) != 0;
        let rc = (data[0] & 0x1F) as usize;
        let pt = data[1];
        if pt != 201 {
            return Err(VideoIpError::InvalidPacket(format!(
                "expected PT=201 (RR), got {pt}"
            )));
        }
        let length = u16::from_be_bytes([data[2], data[3]]) as usize;
        let expected_bytes = (length + 1) * 4;
        if data.len() < expected_bytes {
            return Err(VideoIpError::InvalidPacket(format!(
                "declared length {length} words requires {expected_bytes} bytes, have {}",
                data.len()
            )));
        }
        let ssrc = u32::from_be_bytes([data[4], data[5], data[6], data[7]]);
        let mut offset = 8usize;
        let mut report_blocks = Vec::with_capacity(rc);
        for _ in 0..rc {
            if offset + ReportBlock::WIRE_SIZE > data.len() {
                return Err(VideoIpError::InvalidPacket(
                    "truncated RR report block".to_string(),
                ));
            }
            let block = ReportBlock::read_from(&data[offset..])?;
            report_blocks.push(block);
            offset += ReportBlock::WIRE_SIZE;
        }
        Ok(Self {
            ssrc,
            report_blocks,
            padding,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Compound RTCP Packet (common envelope for multiple RTCP sub-packets)
// ─────────────────────────────────────────────────────────────────────────────

/// Variant holding a single parsed RTCP sub-packet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RtcpPacket {
    /// Sender Report.
    Sr(SenderReport),
    /// Receiver Report.
    Rr(ReceiverReport),
    /// Unknown / not yet decoded packet type.
    Unknown {
        /// Raw packet type byte.
        pt: u8,
        /// Raw bytes of this sub-packet (including its 4-byte header).
        data: Vec<u8>,
    },
}

/// Parses a potentially compound RTCP packet into its constituent sub-packets.
///
/// RFC 3550 §6.1 requires that RTCP packets be sent as compound packets
/// containing at least an SR or RR, optionally followed by other RTCP
/// packets.  This function handles such compounds.
///
/// # Errors
///
/// Returns [`VideoIpError::InvalidPacket`] when the bytes cannot be parsed.
pub fn parse_compound_rtcp(mut data: &[u8]) -> VideoIpResult<Vec<RtcpPacket>> {
    let mut packets = Vec::new();
    while data.len() >= 4 {
        let length = u16::from_be_bytes([data[2], data[3]]) as usize;
        let pkt_size = (length + 1) * 4;
        if pkt_size > data.len() {
            return Err(VideoIpError::InvalidPacket(format!(
                "sub-packet length {pkt_size} exceeds remaining buffer {}",
                data.len()
            )));
        }
        let pkt_data = &data[..pkt_size];
        let pt = pkt_data[1];
        let parsed = match pt {
            200 => RtcpPacket::Sr(SenderReport::decode(pkt_data)?),
            201 => RtcpPacket::Rr(ReceiverReport::decode(pkt_data)?),
            _ => RtcpPacket::Unknown {
                pt,
                data: pkt_data.to_vec(),
            },
        };
        packets.push(parsed);
        data = &data[pkt_size..];
    }
    Ok(packets)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── NTP timestamp ────────────────────────────────────────────────────────

    #[test]
    fn test_ntp_roundtrip_u64() {
        let ts = NtpTimestamp {
            seconds: 3_913_056_000,
            fraction: 0x8000_0000,
        };
        let raw = ts.to_u64();
        let ts2 = NtpTimestamp::from_u64(raw);
        assert_eq!(ts, ts2);
    }

    #[test]
    fn test_ntp_from_unix_duration() {
        // Unix epoch maps to NTP epoch + offset.
        let ts = NtpTimestamp::from_unix_duration(Duration::ZERO);
        assert_eq!(ts.seconds, NTP_UNIX_OFFSET_SECS as u32);
        assert_eq!(ts.fraction, 0);
    }

    #[test]
    fn test_ntp_write_read() {
        let ts = NtpTimestamp {
            seconds: 3_000_000_000,
            fraction: 0x1234_5678,
        };
        let mut buf = Vec::new();
        ts.write_to(&mut buf);
        assert_eq!(buf.len(), 8);
        let ts2 = NtpTimestamp::read_from(&buf)
            .expect("buffer is 8 bytes, exactly NTP size");
        assert_eq!(ts, ts2);
    }

    #[test]
    fn test_ntp_read_too_short() {
        let err = NtpTimestamp::read_from(&[0u8; 4])
            .expect_err("4 bytes is shorter than NTP 8 bytes");
        assert!(matches!(err, VideoIpError::InvalidPacket(_)));
    }

    // ── RTP ↔ NTP mapping ────────────────────────────────────────────────────

    #[test]
    fn test_rtp_to_ntp_identity() {
        let anchor_ntp = NtpTimestamp {
            seconds: 3_900_000_000,
            fraction: 0,
        };
        // delta = 0 ⟹ same timestamp
        let result = rtp_to_ntp(1000, 1000, anchor_ntp, 90_000)
            .expect("valid clock rate and timestamps");
        assert_eq!(result, anchor_ntp);
    }

    #[test]
    fn test_rtp_to_ntp_one_second() {
        let anchor_ntp = NtpTimestamp {
            seconds: 3_900_000_000,
            fraction: 0,
        };
        // advance by exactly one second (90000 ticks at 90 kHz)
        let result = rtp_to_ntp(90_000 + 1000, 1000, anchor_ntp, 90_000)
            .expect("valid clock rate and timestamps");
        // NTP seconds should increase by 1
        assert_eq!(result.seconds, anchor_ntp.seconds + 1);
    }

    #[test]
    fn test_rtp_to_ntp_zero_clock_rate_error() {
        let anchor = NtpTimestamp::default();
        let err = rtp_to_ntp(0, 0, anchor, 0)
            .expect_err("zero clock rate must fail");
        assert!(matches!(err, VideoIpError::InvalidState(_)));
    }

    // ── ReportBlock ──────────────────────────────────────────────────────────

    #[test]
    fn test_fraction_lost_zero_expected() {
        assert_eq!(ReportBlock::compute_fraction_lost(0, 0), 0);
    }

    #[test]
    fn test_fraction_lost_half() {
        // 50 % lost ≈ 128/256
        let f = ReportBlock::compute_fraction_lost(100, 50);
        assert_eq!(f, 128);
    }

    #[test]
    fn test_fraction_lost_all_received() {
        assert_eq!(ReportBlock::compute_fraction_lost(100, 100), 0);
    }

    #[test]
    fn test_report_block_roundtrip() {
        let block = ReportBlock {
            ssrc: 0xDEAD_BEEF,
            fraction_lost: 42,
            cumulative_lost: -5,
            extended_highest_seq: 12345,
            jitter: 300,
            last_sr: 0x0000_FFFF,
            delay_since_last_sr: 65536,
        };
        let mut buf = Vec::new();
        block.write_to(&mut buf);
        assert_eq!(buf.len(), ReportBlock::WIRE_SIZE);
        let block2 = ReportBlock::read_from(&buf)
            .expect("buffer holds a valid ReportBlock");
        assert_eq!(block, block2);
    }

    // ── SenderReport ─────────────────────────────────────────────────────────

    #[test]
    fn test_sender_report_roundtrip_no_blocks() {
        let ntp = NtpTimestamp {
            seconds: 3_900_000_000,
            fraction: 0xAAAA_BBBB,
        };
        let sr = SenderReport::new(0x1234_5678, ntp, 90000, 600, 1_200_000, vec![])
            .expect("valid SR with no report blocks");
        let encoded = sr.encode();
        let decoded = SenderReport::decode(&encoded)
            .expect("encoded SR is valid");
        assert_eq!(decoded.ssrc, sr.ssrc);
        assert_eq!(decoded.ntp_timestamp, sr.ntp_timestamp);
        assert_eq!(decoded.rtp_timestamp, sr.rtp_timestamp);
        assert_eq!(decoded.sender_packet_count, sr.sender_packet_count);
        assert_eq!(decoded.sender_octet_count, sr.sender_octet_count);
        assert!(decoded.report_blocks.is_empty());
    }

    #[test]
    fn test_sender_report_roundtrip_with_block() {
        let ntp = NtpTimestamp {
            seconds: 3_900_000_001,
            fraction: 0,
        };
        let block = ReportBlock {
            ssrc: 0xCAFE_BABE,
            fraction_lost: 10,
            cumulative_lost: 3,
            extended_highest_seq: 999,
            jitter: 50,
            last_sr: 0x0000_1234,
            delay_since_last_sr: 1024,
        };
        let sr =
            SenderReport::new(0xABCD_EF01, ntp, 180000, 1200, 2_400_000, vec![block.clone()])
                .expect("valid SR with one report block");
        let decoded = SenderReport::decode(&sr.encode())
            .expect("encoded SR is valid");
        assert_eq!(decoded.report_blocks.len(), 1);
        assert_eq!(decoded.report_blocks[0], block);
    }

    #[test]
    fn test_sender_report_too_many_blocks() {
        let ntp = NtpTimestamp::default();
        let blocks: Vec<ReportBlock> = (0..32)
            .map(|i| ReportBlock {
                ssrc: i,
                fraction_lost: 0,
                cumulative_lost: 0,
                extended_highest_seq: 0,
                jitter: 0,
                last_sr: 0,
                delay_since_last_sr: 0,
            })
            .collect();
        assert!(SenderReport::new(1, ntp, 0, 0, 0, blocks).is_err());
    }

    // ── ReceiverReport ───────────────────────────────────────────────────────

    #[test]
    fn test_receiver_report_roundtrip() {
        let block = ReportBlock {
            ssrc: 0x1111_2222,
            fraction_lost: 5,
            cumulative_lost: 2,
            extended_highest_seq: 500,
            jitter: 10,
            last_sr: 0x0000_ABCD,
            delay_since_last_sr: 512,
        };
        let rr = ReceiverReport::new(0x5555_6666, vec![block.clone()])
            .expect("valid RR with one report block");
        let encoded = rr.encode();
        let decoded = ReceiverReport::decode(&encoded)
            .expect("encoded RR is valid");
        assert_eq!(decoded.ssrc, rr.ssrc);
        assert_eq!(decoded.report_blocks.len(), 1);
        assert_eq!(decoded.report_blocks[0], block);
    }

    // ── Compound parsing ─────────────────────────────────────────────────────

    #[test]
    fn test_compound_sr_rr() {
        let ntp = NtpTimestamp {
            seconds: 3_900_000_002,
            fraction: 0,
        };
        let sr = SenderReport::new(0xAAAA_AAAA, ntp, 0, 0, 0, vec![])
            .expect("valid SR with no blocks");
        let rr = ReceiverReport::new(0xBBBB_BBBB, vec![])
            .expect("valid RR with no blocks");
        let mut compound = sr.encode();
        compound.extend_from_slice(&rr.encode());
        let packets = parse_compound_rtcp(&compound)
            .expect("compound RTCP is valid");
        assert_eq!(packets.len(), 2);
        assert!(matches!(packets[0], RtcpPacket::Sr(_)));
        assert!(matches!(packets[1], RtcpPacket::Rr(_)));
    }

    #[test]
    fn test_decode_bad_version() {
        // Craft a buffer with V=1 (invalid)
        let mut buf = vec![0u8; 28];
        buf[0] = (1u8 << 6) | 0; // V=1, P=0, RC=0
        buf[1] = 200; // PT=SR
        // length = 6 (28 bytes total = 7 words, length field = 7 - 1 = 6)
        buf[2] = 0;
        buf[3] = 6;
        assert!(SenderReport::decode(&buf).is_err());
    }
}
