//! MPEG-TS packet parsing.
//!
//! MPEG Transport Stream packets are fixed 188-byte units that carry
//! Program Specific Information (PSI) and Packetized Elementary Stream (PES) data.

use oximedia_core::{OxiError, OxiResult};

/// MPEG-TS packet size in bytes.
pub const TS_PACKET_SIZE: usize = 188;

/// MPEG-TS sync byte (0x47).
pub const SYNC_BYTE: u8 = 0x47;

/// Null PID (used for stuffing).
pub const NULL_PID: u16 = 0x1FFF;

/// Program Association Table (PAT) PID.
pub const PAT_PID: u16 = 0x0000;

/// Conditional Access Table (CAT) PID.
#[allow(dead_code)]
pub const CAT_PID: u16 = 0x0001;

/// Transport Stream Description Table (TSDT) PID.
#[allow(dead_code)]
pub const TSDT_PID: u16 = 0x0002;

/// Adaptation field control values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdaptationFieldControl {
    /// Reserved (invalid).
    Reserved,
    /// Payload only.
    PayloadOnly,
    /// Adaptation field only.
    AdaptationFieldOnly,
    /// Adaptation field followed by payload.
    AdaptationFieldAndPayload,
}

impl AdaptationFieldControl {
    /// Creates from 2-bit value.
    #[must_use]
    pub const fn from_bits(bits: u8) -> Self {
        match bits & 0x03 {
            0b00 => Self::Reserved,
            0b01 => Self::PayloadOnly,
            0b10 => Self::AdaptationFieldOnly,
            0b11 => Self::AdaptationFieldAndPayload,
            _ => unreachable!(),
        }
    }

    /// Returns true if this packet has a payload.
    #[must_use]
    pub const fn has_payload(self) -> bool {
        matches!(self, Self::PayloadOnly | Self::AdaptationFieldAndPayload)
    }

    /// Returns true if this packet has an adaptation field.
    #[must_use]
    pub const fn has_adaptation_field(self) -> bool {
        matches!(
            self,
            Self::AdaptationFieldOnly | Self::AdaptationFieldAndPayload
        )
    }
}

/// Adaptation field containing timing and optional data.
#[derive(Debug, Clone)]
pub struct AdaptationField {
    /// Program Clock Reference (PCR) value, if present.
    pub pcr: Option<u64>,
    /// Original Program Clock Reference (OPCR) value, if present.
    #[allow(dead_code)]
    pub opcr: Option<u64>,
    /// Discontinuity indicator.
    #[allow(dead_code)]
    pub discontinuity: bool,
    /// Random access indicator (signals keyframe).
    pub random_access: bool,
    /// Elementary stream priority indicator.
    #[allow(dead_code)]
    pub es_priority: bool,
    /// Splice countdown value, if present.
    #[allow(dead_code)]
    pub splice_countdown: Option<i8>,
}

impl AdaptationField {
    /// Parses an adaptation field from raw bytes.
    ///
    /// # Arguments
    ///
    /// * `data` - Raw adaptation field data (starting with length byte)
    ///
    /// # Errors
    ///
    /// Returns an error if the adaptation field is malformed.
    pub fn parse(data: &[u8]) -> OxiResult<(Self, usize)> {
        if data.is_empty() {
            return Err(OxiError::InvalidData("Empty adaptation field".to_string()));
        }

        let length = data[0] as usize;
        if length == 0 {
            // Empty adaptation field
            return Ok((
                Self {
                    pcr: None,
                    opcr: None,
                    discontinuity: false,
                    random_access: false,
                    es_priority: false,
                    splice_countdown: None,
                },
                1,
            ));
        }

        if data.len() < length + 1 {
            return Err(OxiError::InvalidData(format!(
                "Adaptation field too short: expected {}, got {}",
                length + 1,
                data.len()
            )));
        }

        let flags = data[1];
        let discontinuity = (flags & 0x80) != 0;
        let random_access = (flags & 0x40) != 0;
        let es_priority = (flags & 0x20) != 0;
        let pcr_flag = (flags & 0x10) != 0;
        let opcr_flag = (flags & 0x08) != 0;
        let splicing_point_flag = (flags & 0x04) != 0;

        let mut offset = 2;
        let mut pcr = None;
        let mut opcr = None;
        let mut splice_countdown = None;

        // Parse PCR if present
        if pcr_flag && offset + 6 <= length + 1 {
            pcr = Some(Self::parse_pcr(&data[offset..offset + 6]));
            offset += 6;
        }

        // Parse OPCR if present
        if opcr_flag && offset + 6 <= length + 1 {
            opcr = Some(Self::parse_pcr(&data[offset..offset + 6]));
            offset += 6;
        }

        // Parse splice countdown if present
        if splicing_point_flag && offset < length + 1 {
            #[allow(clippy::cast_possible_wrap)]
            {
                splice_countdown = Some(data[offset] as i8);
            }
        }

        Ok((
            Self {
                pcr,
                opcr,
                discontinuity,
                random_access,
                es_priority,
                splice_countdown,
            },
            length + 1,
        ))
    }

    /// Parses a 42-bit PCR value from 6 bytes.
    ///
    /// PCR = `PCR_base` * 300 + `PCR_extension`
    /// `PCR_base` is 33 bits, `PCR_extension` is 9 bits.
    fn parse_pcr(data: &[u8]) -> u64 {
        let pcr_base = ((u64::from(data[0])) << 25)
            | ((u64::from(data[1])) << 17)
            | ((u64::from(data[2])) << 9)
            | ((u64::from(data[3])) << 1)
            | ((u64::from(data[4])) >> 7);

        let pcr_ext = (((u16::from(data[4])) & 0x01) << 8) | u16::from(data[5]);

        pcr_base * 300 + u64::from(pcr_ext)
    }
}

/// MPEG-TS transport packet.
#[derive(Debug, Clone)]
pub struct TsPacket {
    /// Transport Error Indicator (TEI).
    #[allow(dead_code)]
    pub transport_error: bool,
    /// Payload Unit Start Indicator (PUSI).
    pub payload_unit_start: bool,
    /// Transport priority.
    #[allow(dead_code)]
    pub priority: bool,
    /// Packet Identifier (PID).
    pub pid: u16,
    /// Transport Scrambling Control (TSC).
    #[allow(dead_code)]
    pub scrambling_control: u8,
    /// Adaptation field control.
    pub adaptation_field_control: AdaptationFieldControl,
    /// Continuity counter (4-bit).
    pub continuity_counter: u8,
    /// Adaptation field, if present.
    pub adaptation_field: Option<AdaptationField>,
    /// Payload data.
    pub payload: Vec<u8>,
}

impl TsPacket {
    /// Parses a transport stream packet from raw bytes.
    ///
    /// # Arguments
    ///
    /// * `data` - Raw 188-byte packet data
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The data is not exactly 188 bytes
    /// - The sync byte is missing or incorrect
    /// - The packet is malformed
    pub fn parse(data: &[u8]) -> OxiResult<Self> {
        if data.len() != TS_PACKET_SIZE {
            return Err(OxiError::InvalidData(format!(
                "Invalid TS packet size: expected {}, got {}",
                TS_PACKET_SIZE,
                data.len()
            )));
        }

        if data[0] != SYNC_BYTE {
            return Err(OxiError::InvalidData(format!(
                "Invalid sync byte: expected 0x{:02X}, got 0x{:02X}",
                SYNC_BYTE, data[0]
            )));
        }

        // Parse header (4 bytes)
        let transport_error = (data[1] & 0x80) != 0;
        let payload_unit_start = (data[1] & 0x40) != 0;
        let priority = (data[1] & 0x20) != 0;
        let pid = (u16::from(data[1] & 0x1F) << 8) | u16::from(data[2]);

        let scrambling_control = (data[3] >> 6) & 0x03;
        let adaptation_field_control = AdaptationFieldControl::from_bits((data[3] >> 4) & 0x03);
        let continuity_counter = data[3] & 0x0F;

        let mut offset = 4;
        let mut adaptation_field = None;

        // Parse adaptation field if present
        if adaptation_field_control.has_adaptation_field() {
            let (af, af_size) = AdaptationField::parse(&data[offset..])?;
            adaptation_field = Some(af);
            offset += af_size;
        }

        // Extract payload
        let payload = if adaptation_field_control.has_payload() && offset < TS_PACKET_SIZE {
            data[offset..].to_vec()
        } else {
            Vec::new()
        };

        Ok(Self {
            transport_error,
            payload_unit_start,
            priority,
            pid,
            scrambling_control,
            adaptation_field_control,
            continuity_counter,
            adaptation_field,
            payload,
        })
    }

    /// Returns true if this is a PAT packet.
    #[must_use]
    pub const fn is_pat(&self) -> bool {
        self.pid == PAT_PID
    }

    /// Returns true if this is a null packet (stuffing).
    #[must_use]
    pub const fn is_null(&self) -> bool {
        self.pid == NULL_PID
    }

    /// Returns the PCR value if present in the adaptation field.
    #[must_use]
    pub fn pcr(&self) -> Option<u64> {
        self.adaptation_field.as_ref().and_then(|af| af.pcr)
    }

    /// Returns true if this packet indicates a random access point (keyframe).
    #[must_use]
    pub fn is_random_access(&self) -> bool {
        self.adaptation_field
            .as_ref()
            .is_some_and(|af| af.random_access)
    }

    /// Returns true if this packet has a discontinuity.
    #[must_use]
    #[allow(dead_code)]
    pub fn has_discontinuity(&self) -> bool {
        self.adaptation_field
            .as_ref()
            .is_some_and(|af| af.discontinuity)
    }
}

/// Continuity counter tracker for detecting packet loss.
#[derive(Debug, Clone)]
pub struct ContinuityTracker {
    /// Last continuity counter value per PID.
    counters: std::collections::HashMap<u16, u8>,
}

impl Default for ContinuityTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(dead_code)]
impl ContinuityTracker {
    /// Creates a new continuity tracker.
    #[must_use]
    pub fn new() -> Self {
        Self {
            counters: std::collections::HashMap::new(),
        }
    }

    /// Checks and updates the continuity counter for a PID.
    ///
    /// Returns true if the counter is valid (no packets lost).
    ///
    /// # Arguments
    ///
    /// * `pid` - Packet Identifier
    /// * `counter` - Current continuity counter value
    /// * `has_payload` - Whether the packet has a payload
    pub fn check(&mut self, pid: u16, counter: u8, has_payload: bool) -> bool {
        // Null packets and packets without payload are not checked
        if pid == NULL_PID || !has_payload {
            return true;
        }

        if let Some(&last_counter) = self.counters.get(&pid) {
            let expected = (last_counter + 1) & 0x0F;
            if counter != expected {
                // Update to current counter and report discontinuity
                self.counters.insert(pid, counter);
                return false;
            }
        }

        self.counters.insert(pid, counter);
        true
    }

    /// Resets the counter for a specific PID.
    pub fn reset_pid(&mut self, pid: u16) {
        self.counters.remove(&pid);
    }

    /// Clears all counters.
    pub fn clear(&mut self) {
        self.counters.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptation_field_control() {
        assert_eq!(
            AdaptationFieldControl::from_bits(0b00),
            AdaptationFieldControl::Reserved
        );
        assert_eq!(
            AdaptationFieldControl::from_bits(0b01),
            AdaptationFieldControl::PayloadOnly
        );
        assert_eq!(
            AdaptationFieldControl::from_bits(0b10),
            AdaptationFieldControl::AdaptationFieldOnly
        );
        assert_eq!(
            AdaptationFieldControl::from_bits(0b11),
            AdaptationFieldControl::AdaptationFieldAndPayload
        );

        assert!(AdaptationFieldControl::PayloadOnly.has_payload());
        assert!(!AdaptationFieldControl::AdaptationFieldOnly.has_payload());
    }

    #[test]
    fn test_parse_ts_packet_minimal() {
        let mut data = [0u8; TS_PACKET_SIZE];
        data[0] = SYNC_BYTE;
        data[1] = 0x40; // PUSI=1, PID high bits
        data[2] = 0x00; // PID=0x0000 (PAT)
        data[3] = 0x10; // AFC=01 (payload only), CC=0

        let packet = TsPacket::parse(&data).expect("operation should succeed");
        assert_eq!(packet.pid, PAT_PID);
        assert!(packet.payload_unit_start);
        assert!(!packet.transport_error);
        assert_eq!(packet.continuity_counter, 0);
        assert_eq!(
            packet.adaptation_field_control,
            AdaptationFieldControl::PayloadOnly
        );
    }

    #[test]
    fn test_parse_ts_packet_with_adaptation_field() {
        let mut data = [0u8; TS_PACKET_SIZE];
        data[0] = SYNC_BYTE;
        data[1] = 0x01; // PID=0x0100
        data[2] = 0x00;
        data[3] = 0x30; // AFC=11 (adaptation + payload), CC=0
        data[4] = 7; // Adaptation field length
        data[5] = 0x50; // Flags: random_access=1, PCR=1

        // PCR (6 bytes) - simplified example
        data[6..12].copy_from_slice(&[0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC]);

        let packet = TsPacket::parse(&data).expect("operation should succeed");
        assert_eq!(packet.pid, 0x0100);
        assert!(packet.adaptation_field.is_some());

        let af = packet.adaptation_field.expect("operation should succeed");
        assert!(af.random_access);
        assert!(af.pcr.is_some());
    }

    #[test]
    fn test_continuity_tracker() {
        let mut tracker = ContinuityTracker::new();

        // First packet
        assert!(tracker.check(0x100, 0, true));

        // Sequential packets
        assert!(tracker.check(0x100, 1, true));
        assert!(tracker.check(0x100, 2, true));

        // Skip a packet (discontinuity)
        assert!(!tracker.check(0x100, 4, true));

        // Resume from new counter
        assert!(tracker.check(0x100, 5, true));

        // Counter wraps around at 15
        tracker.check(0x100, 14, true);
        tracker.check(0x100, 15, true);
        assert!(tracker.check(0x100, 0, true));
    }

    #[test]
    fn test_null_packet() {
        let mut data = [0u8; TS_PACKET_SIZE];
        data[0] = SYNC_BYTE;
        data[1] = 0x1F; // PID high bits
        data[2] = 0xFF; // PID=0x1FFF (null)
        data[3] = 0x10;

        let packet = TsPacket::parse(&data).expect("operation should succeed");
        assert!(packet.is_null());
        assert_eq!(packet.pid, NULL_PID);
    }
}
