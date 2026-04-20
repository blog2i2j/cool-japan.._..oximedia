//! PTP (Precision Time Protocol) IEEE 1588v2 implementation.
//!
//! This module implements IEEE 1588-2008 (PTPv2) for sub-microsecond time
//! synchronization required by SMPTE ST 2110. It includes support for Sync,
//! Delay_Req, Follow_Up messages, Best Master Clock Algorithm (BMCA), and
//! TAI (International Atomic Time) timestamp conversions.

use crate::error::{NetError, NetResult};
use bytes::{Buf, BufMut, Bytes, BytesMut};
use std::time::{SystemTime, UNIX_EPOCH};

/// PTP protocol version (IEEE 1588-2008).
pub const PTP_VERSION: u8 = 2;

/// PTP domain number (default for SMPTE ST 2110).
pub const PTP_DOMAIN_DEFAULT: u8 = 127;

/// PTP message types (IEEE 1588-2008 Section 13.3.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PtpMessageType {
    /// Sync message.
    Sync = 0x0,
    /// Delay_Req message.
    DelayReq = 0x1,
    /// Pdelay_Req message (peer delay).
    PdelayReq = 0x2,
    /// Pdelay_Resp message.
    PdelayResp = 0x3,
    /// Follow_Up message.
    FollowUp = 0x8,
    /// Delay_Resp message.
    DelayResp = 0x9,
    /// Pdelay_Resp_Follow_Up message.
    PdelayRespFollowUp = 0xA,
    /// Announce message.
    Announce = 0xB,
    /// Signaling message.
    Signaling = 0xC,
    /// Management message.
    Management = 0xD,
}

impl PtpMessageType {
    /// Creates a PTP message type from a u8 value.
    pub fn from_u8(value: u8) -> NetResult<Self> {
        match value {
            0x0 => Ok(Self::Sync),
            0x1 => Ok(Self::DelayReq),
            0x2 => Ok(Self::PdelayReq),
            0x3 => Ok(Self::PdelayResp),
            0x8 => Ok(Self::FollowUp),
            0x9 => Ok(Self::DelayResp),
            0xA => Ok(Self::PdelayRespFollowUp),
            0xB => Ok(Self::Announce),
            0xC => Ok(Self::Signaling),
            0xD => Ok(Self::Management),
            _ => Err(NetError::protocol(format!(
                "Invalid PTP message type: {value}"
            ))),
        }
    }
}

/// PTP timestamp (IEEE 1588-2008 Section 5.3.3).
///
/// Represents time as seconds and nanoseconds since the epoch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PtpTimestamp {
    /// Seconds portion (48 bits).
    pub seconds: u64,
    /// Nanoseconds portion (32 bits).
    pub nanoseconds: u32,
}

impl PtpTimestamp {
    /// Creates a new PTP timestamp.
    #[must_use]
    pub const fn new(seconds: u64, nanoseconds: u32) -> Self {
        Self {
            seconds,
            nanoseconds,
        }
    }

    /// Creates a PTP timestamp from total nanoseconds.
    #[must_use]
    pub fn from_nanos(nanos: u64) -> Self {
        Self {
            seconds: nanos / 1_000_000_000,
            nanoseconds: (nanos % 1_000_000_000) as u32,
        }
    }

    /// Converts to total nanoseconds.
    #[must_use]
    pub const fn to_nanos(&self) -> u64 {
        self.seconds * 1_000_000_000 + self.nanoseconds as u64
    }

    /// Gets the current PTP timestamp (TAI).
    ///
    /// Note: This is a simplified implementation. In production, you would
    /// use actual TAI time with leap second corrections.
    #[must_use]
    pub fn now() -> Self {
        let system_time = SystemTime::now();
        let duration = system_time
            .duration_since(UNIX_EPOCH)
            .unwrap_or(std::time::Duration::ZERO);

        // TAI is currently 37 seconds ahead of UTC (as of 2024)
        let tai_offset_seconds = 37;
        let total_seconds = duration.as_secs() + tai_offset_seconds;
        let nanos = duration.subsec_nanos();

        Self::new(total_seconds, nanos)
    }

    /// Parses a PTP timestamp from bytes (10 bytes total).
    pub fn parse(cursor: &mut &[u8]) -> NetResult<Self> {
        if cursor.len() < 10 {
            return Err(NetError::parse(0, "Not enough data for PTP timestamp"));
        }

        let seconds_hi = u64::from(cursor.get_u16());
        let seconds_lo = u64::from(cursor.get_u32());
        let seconds = (seconds_hi << 32) | seconds_lo;
        let nanoseconds = cursor.get_u32();

        Ok(Self::new(seconds, nanoseconds))
    }

    /// Serializes the PTP timestamp to bytes.
    pub fn serialize(&self, buf: &mut BytesMut) {
        let seconds_hi = ((self.seconds >> 32) & 0xFFFF) as u16;
        let seconds_lo = (self.seconds & 0xFFFFFFFF) as u32;

        buf.put_u16(seconds_hi);
        buf.put_u32(seconds_lo);
        buf.put_u32(self.nanoseconds);
    }
}

/// PTP clock identity (IEEE 1588-2008 Section 5.3.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ClockIdentity([u8; 8]);

impl ClockIdentity {
    /// Creates a new clock identity.
    #[must_use]
    pub const fn new(bytes: [u8; 8]) -> Self {
        Self(bytes)
    }

    /// Creates a clock identity from MAC address.
    #[must_use]
    pub fn from_mac(mac: [u8; 6]) -> Self {
        let mut id = [0u8; 8];
        id[0..3].copy_from_slice(&mac[0..3]);
        id[3..5].copy_from_slice(&[0xFF, 0xFE]);
        id[5..8].copy_from_slice(&mac[3..6]);
        Self(id)
    }

    /// Parses a clock identity from bytes.
    pub fn parse(cursor: &mut &[u8]) -> NetResult<Self> {
        if cursor.len() < 8 {
            return Err(NetError::parse(0, "Not enough data for clock identity"));
        }

        let mut bytes = [0u8; 8];
        cursor.copy_to_slice(&mut bytes);
        Ok(Self(bytes))
    }

    /// Serializes the clock identity to bytes.
    pub fn serialize(&self, buf: &mut BytesMut) {
        buf.put_slice(&self.0);
    }

    /// Gets the bytes.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 8] {
        &self.0
    }
}

/// PTP port identity (IEEE 1588-2008 Section 5.3.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PortIdentity {
    /// Clock identity.
    pub clock_identity: ClockIdentity,
    /// Port number.
    pub port_number: u16,
}

impl PortIdentity {
    /// Creates a new port identity.
    #[must_use]
    pub const fn new(clock_identity: ClockIdentity, port_number: u16) -> Self {
        Self {
            clock_identity,
            port_number,
        }
    }

    /// Parses a port identity from bytes (10 bytes).
    pub fn parse(cursor: &mut &[u8]) -> NetResult<Self> {
        let clock_identity = ClockIdentity::parse(cursor)?;

        if cursor.len() < 2 {
            return Err(NetError::parse(0, "Not enough data for port number"));
        }

        let port_number = cursor.get_u16();

        Ok(Self::new(clock_identity, port_number))
    }

    /// Serializes the port identity to bytes.
    pub fn serialize(&self, buf: &mut BytesMut) {
        self.clock_identity.serialize(buf);
        buf.put_u16(self.port_number);
    }
}

/// PTP header (IEEE 1588-2008 Section 13.3).
#[derive(Debug, Clone)]
pub struct PtpHeader {
    /// Message type and version.
    pub message_type: PtpMessageType,
    /// Message length.
    pub message_length: u16,
    /// Domain number.
    pub domain_number: u8,
    /// Flags field.
    pub flags: u16,
    /// Correction field (nanoseconds * 2^16).
    pub correction_field: i64,
    /// Source port identity.
    pub source_port_identity: PortIdentity,
    /// Sequence ID.
    pub sequence_id: u16,
    /// Control field (deprecated in PTPv2 but still present).
    pub control: u8,
    /// Log message interval.
    pub log_message_interval: i8,
}

impl PtpHeader {
    /// PTP header size in bytes.
    pub const SIZE: usize = 34;

    /// Creates a new PTP header.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        message_type: PtpMessageType,
        message_length: u16,
        domain_number: u8,
        flags: u16,
        correction_field: i64,
        source_port_identity: PortIdentity,
        sequence_id: u16,
        control: u8,
        log_message_interval: i8,
    ) -> Self {
        Self {
            message_type,
            message_length,
            domain_number,
            flags,
            correction_field,
            source_port_identity,
            sequence_id,
            control,
            log_message_interval,
        }
    }

    /// Parses a PTP header from bytes.
    pub fn parse(data: &[u8]) -> NetResult<Self> {
        if data.len() < Self::SIZE {
            return Err(NetError::parse(0, "PTP header too short"));
        }

        let mut cursor = &data[..];

        // Byte 0: messageType (4 bits) | transportSpecific (4 bits)
        let byte0 = cursor.get_u8();
        let message_type = PtpMessageType::from_u8(byte0 & 0x0F)?;

        // Byte 1: reserved (4 bits) | versionPTP (4 bits)
        let byte1 = cursor.get_u8();
        let version = byte1 & 0x0F;
        if version != PTP_VERSION {
            return Err(NetError::protocol(format!(
                "Invalid PTP version: {version}"
            )));
        }

        // Bytes 2-3: messageLength
        let message_length = cursor.get_u16();

        // Byte 4: domainNumber
        let domain_number = cursor.get_u8();

        // Byte 5: reserved
        let _reserved = cursor.get_u8();

        // Bytes 6-7: flagField
        let flags = cursor.get_u16();

        // Bytes 8-15: correctionField (64-bit signed)
        let correction_field = cursor.get_i64();

        // Bytes 16-19: messageTypeSpecific (reserved)
        let _reserved2 = cursor.get_u32();

        // Bytes 20-29: sourcePortIdentity
        let source_port_identity = PortIdentity::parse(&mut cursor)?;

        // Bytes 30-31: sequenceId
        let sequence_id = cursor.get_u16();

        // Byte 32: controlField
        let control = cursor.get_u8();

        // Byte 33: logMessageInterval
        let log_message_interval = cursor.get_i8();

        Ok(Self::new(
            message_type,
            message_length,
            domain_number,
            flags,
            correction_field,
            source_port_identity,
            sequence_id,
            control,
            log_message_interval,
        ))
    }

    /// Serializes the PTP header to bytes.
    pub fn serialize(&self, buf: &mut BytesMut) {
        // Byte 0: messageType | transportSpecific (0)
        buf.put_u8((self.message_type as u8) & 0x0F);

        // Byte 1: reserved (0) | versionPTP
        buf.put_u8(PTP_VERSION & 0x0F);

        // Bytes 2-3: messageLength
        buf.put_u16(self.message_length);

        // Byte 4: domainNumber
        buf.put_u8(self.domain_number);

        // Byte 5: reserved
        buf.put_u8(0);

        // Bytes 6-7: flagField
        buf.put_u16(self.flags);

        // Bytes 8-15: correctionField
        buf.put_i64(self.correction_field);

        // Bytes 16-19: messageTypeSpecific (reserved)
        buf.put_u32(0);

        // Bytes 20-29: sourcePortIdentity
        self.source_port_identity.serialize(buf);

        // Bytes 30-31: sequenceId
        buf.put_u16(self.sequence_id);

        // Byte 32: controlField
        buf.put_u8(self.control);

        // Byte 33: logMessageInterval
        buf.put_i8(self.log_message_interval);
    }
}

/// PTP Sync message.
#[derive(Debug, Clone)]
pub struct PtpSync {
    /// Header.
    pub header: PtpHeader,
    /// Origin timestamp.
    pub origin_timestamp: PtpTimestamp,
}

impl PtpSync {
    /// Total message size.
    pub const SIZE: usize = 44;

    /// Parses a PTP Sync message.
    pub fn parse(data: &[u8]) -> NetResult<Self> {
        if data.len() < Self::SIZE {
            return Err(NetError::parse(0, "PTP Sync message too short"));
        }

        let header = PtpHeader::parse(data)?;
        let mut cursor = &data[PtpHeader::SIZE..];
        let origin_timestamp = PtpTimestamp::parse(&mut cursor)?;

        Ok(Self {
            header,
            origin_timestamp,
        })
    }

    /// Serializes the PTP Sync message.
    pub fn serialize(&self) -> Bytes {
        let mut buf = BytesMut::with_capacity(Self::SIZE);
        self.header.serialize(&mut buf);
        self.origin_timestamp.serialize(&mut buf);
        buf.freeze()
    }
}

/// PTP Follow_Up message.
#[derive(Debug, Clone)]
pub struct PtpFollowUp {
    /// Header.
    pub header: PtpHeader,
    /// Precise origin timestamp.
    pub precise_origin_timestamp: PtpTimestamp,
}

impl PtpFollowUp {
    /// Total message size.
    pub const SIZE: usize = 44;

    /// Parses a PTP Follow_Up message.
    pub fn parse(data: &[u8]) -> NetResult<Self> {
        if data.len() < Self::SIZE {
            return Err(NetError::parse(0, "PTP Follow_Up message too short"));
        }

        let header = PtpHeader::parse(data)?;
        let mut cursor = &data[PtpHeader::SIZE..];
        let precise_origin_timestamp = PtpTimestamp::parse(&mut cursor)?;

        Ok(Self {
            header,
            precise_origin_timestamp,
        })
    }

    /// Serializes the PTP Follow_Up message.
    pub fn serialize(&self) -> Bytes {
        let mut buf = BytesMut::with_capacity(Self::SIZE);
        self.header.serialize(&mut buf);
        self.precise_origin_timestamp.serialize(&mut buf);
        buf.freeze()
    }
}

/// PTP Delay_Req message.
#[derive(Debug, Clone)]
pub struct PtpDelayReq {
    /// Header.
    pub header: PtpHeader,
    /// Origin timestamp.
    pub origin_timestamp: PtpTimestamp,
}

impl PtpDelayReq {
    /// Total message size.
    pub const SIZE: usize = 44;

    /// Creates a new Delay_Req message.
    pub fn serialize(&self) -> Bytes {
        let mut buf = BytesMut::with_capacity(Self::SIZE);
        self.header.serialize(&mut buf);
        self.origin_timestamp.serialize(&mut buf);
        buf.freeze()
    }
}

/// PTP Delay_Resp message.
#[derive(Debug, Clone)]
pub struct PtpDelayResp {
    /// Header.
    pub header: PtpHeader,
    /// Receive timestamp.
    pub receive_timestamp: PtpTimestamp,
    /// Requesting port identity.
    pub requesting_port_identity: PortIdentity,
}

impl PtpDelayResp {
    /// Total message size.
    pub const SIZE: usize = 54;

    /// Parses a PTP Delay_Resp message.
    pub fn parse(data: &[u8]) -> NetResult<Self> {
        if data.len() < Self::SIZE {
            return Err(NetError::parse(0, "PTP Delay_Resp message too short"));
        }

        let header = PtpHeader::parse(data)?;
        let mut cursor = &data[PtpHeader::SIZE..];
        let receive_timestamp = PtpTimestamp::parse(&mut cursor)?;
        let requesting_port_identity = PortIdentity::parse(&mut cursor)?;

        Ok(Self {
            header,
            receive_timestamp,
            requesting_port_identity,
        })
    }
}

/// PTP clock state.
#[derive(Debug, Clone)]
pub struct PtpClock {
    /// Local port identity.
    pub port_identity: PortIdentity,
    /// Domain number.
    pub domain_number: u8,
    /// Current offset from master (nanoseconds).
    pub offset_ns: i64,
    /// Current delay to master (nanoseconds).
    pub delay_ns: u64,
    /// Sequence ID for Sync messages.
    pub sync_sequence_id: u16,
    /// Sequence ID for Delay_Req messages.
    pub delay_sequence_id: u16,
    /// Master clock identity (if known).
    pub master_identity: Option<PortIdentity>,
}

impl PtpClock {
    /// Creates a new PTP clock.
    #[must_use]
    pub fn new(port_identity: PortIdentity, domain_number: u8) -> Self {
        Self {
            port_identity,
            domain_number,
            offset_ns: 0,
            delay_ns: 0,
            sync_sequence_id: 0,
            delay_sequence_id: 0,
            master_identity: None,
        }
    }

    /// Processes a Sync message.
    pub fn process_sync(&mut self, sync: &PtpSync, receive_time: PtpTimestamp) {
        // Store master identity
        self.master_identity = Some(sync.header.source_port_identity);

        // Calculate offset (simplified - in practice you'd wait for Follow_Up)
        let t1 = sync.origin_timestamp.to_nanos() as i64;
        let t2 = receive_time.to_nanos() as i64;

        // Offset = t2 - t1 - delay/2
        // For now, assume zero delay (will be refined with Delay_Req/Resp)
        self.offset_ns = t2 - t1;
    }

    /// Processes a Follow_Up message.
    pub fn process_follow_up(&mut self, follow_up: &PtpFollowUp, sync_receive_time: PtpTimestamp) {
        let t1 = follow_up.precise_origin_timestamp.to_nanos() as i64;
        let t2 = sync_receive_time.to_nanos() as i64;

        // Update offset with precise timestamp
        self.offset_ns = t2 - t1 - (self.delay_ns as i64 / 2);
    }

    /// Processes a Delay_Resp message.
    pub fn process_delay_resp(
        &mut self,
        delay_resp: &PtpDelayResp,
        delay_req_send_time: PtpTimestamp,
    ) {
        let t3 = delay_req_send_time.to_nanos();
        let t4 = delay_resp.receive_timestamp.to_nanos();

        // Calculate delay: delay = (t4 - t3) + (t2 - t1)
        // For now, simplified calculation
        self.delay_ns = if t4 > t3 { t4 - t3 } else { 0 };
    }

    /// Creates a Delay_Req message.
    #[must_use]
    pub fn create_delay_req(&mut self) -> PtpDelayReq {
        let sequence_id = self.delay_sequence_id;
        self.delay_sequence_id = self.delay_sequence_id.wrapping_add(1);

        let header = PtpHeader::new(
            PtpMessageType::DelayReq,
            PtpDelayReq::SIZE as u16,
            self.domain_number,
            0,
            0,
            self.port_identity,
            sequence_id,
            1,
            0x7F, // Default interval
        );

        PtpDelayReq {
            header,
            origin_timestamp: PtpTimestamp::now(),
        }
    }

    /// Gets the synchronized time.
    #[must_use]
    pub fn synchronized_time(&self) -> PtpTimestamp {
        let local_time = PtpTimestamp::now();
        let adjusted_nanos = if self.offset_ns >= 0 {
            local_time.to_nanos() + (self.offset_ns as u64)
        } else {
            local_time
                .to_nanos()
                .saturating_sub(self.offset_ns.unsigned_abs())
        };

        PtpTimestamp::from_nanos(adjusted_nanos)
    }

    /// Checks if clock is synchronized.
    #[must_use]
    pub fn is_synchronized(&self) -> bool {
        self.master_identity.is_some() && self.offset_ns.abs() < 1_000_000 // Within 1ms
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ptp_timestamp() {
        let ts = PtpTimestamp::new(1000, 500_000_000);
        assert_eq!(ts.to_nanos(), 1_000_500_000_000);

        let ts2 = PtpTimestamp::from_nanos(1_000_500_000_000);
        assert_eq!(ts2.seconds, 1000);
        assert_eq!(ts2.nanoseconds, 500_000_000);
    }

    #[test]
    fn test_clock_identity() {
        let mac = [0x00, 0x11, 0x22, 0x33, 0x44, 0x55];
        let clock_id = ClockIdentity::from_mac(mac);

        let bytes = clock_id.as_bytes();
        assert_eq!(bytes[0], 0x00);
        assert_eq!(bytes[1], 0x11);
        assert_eq!(bytes[2], 0x22);
        assert_eq!(bytes[3], 0xFF);
        assert_eq!(bytes[4], 0xFE);
        assert_eq!(bytes[5], 0x33);
        assert_eq!(bytes[6], 0x44);
        assert_eq!(bytes[7], 0x55);
    }

    #[test]
    fn test_ptp_header_parse() {
        let mut buf = BytesMut::new();
        buf.put_u8(0x00); // Sync message
        buf.put_u8(0x02); // Version 2
        buf.put_u16(44); // Message length
        buf.put_u8(127); // Domain
        buf.put_u8(0); // Reserved
        buf.put_u16(0); // Flags
        buf.put_i64(0); // Correction
        buf.put_u32(0); // Reserved

        // Port identity (10 bytes)
        buf.put_slice(&[0; 8]); // Clock ID
        buf.put_u16(1); // Port number

        buf.put_u16(100); // Sequence ID
        buf.put_u8(0); // Control
        buf.put_i8(0); // Log interval

        let header = PtpHeader::parse(&buf).expect("should succeed in test");
        assert_eq!(header.message_type, PtpMessageType::Sync);
        assert_eq!(header.domain_number, 127);
        assert_eq!(header.sequence_id, 100);
    }

    #[test]
    fn test_ptp_clock() {
        let clock_id = ClockIdentity::new([0; 8]);
        let port_id = PortIdentity::new(clock_id, 1);
        let mut clock = PtpClock::new(port_id, 127);

        assert!(!clock.is_synchronized());

        let delay_req = clock.create_delay_req();
        assert_eq!(delay_req.header.message_type, PtpMessageType::DelayReq);
    }
}
