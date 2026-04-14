//! VDCP (Video Disk Control Protocol) implementation.
//!
//! ## Packet format
//!
//! ```text
//! [STX=0x02][LEN][CMD][DATA…][CHK][ETX=0x03]
//! ```
//!
//! - **STX** / **ETX**: framing bytes (0x02 / 0x03).
//! - **LEN**: byte count of `CMD + DATA`.
//! - **CMD**: one-byte command code ([`VdcpCommand`]).
//! - **DATA**: zero or more payload bytes (e.g. timecode for `Cue`).
//! - **CHK**: simple wrapping-add checksum over `LEN + CMD + DATA`.
//!
//! ## Timecode encoding
//!
//! Timecode is encoded as four binary bytes `[HH, MM, SS, FF]` —
//! **not** BCD.  To convert to the industry-standard LTC BCD format each byte
//! must be split into tens and units nibbles:
//!
//! ```text
//! BCD byte = ((value / 10) << 4) | (value % 10)
//! ```

use crate::protocol::serial::SerialPort;
use crate::{AutomationError, Result};
use bytes::{BufMut, BytesMut};
use tracing::{debug, info};

/// VDCP command codes.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VdcpCommand {
    /// Play command
    Play = 0x01,
    /// Stop command
    Stop = 0x02,
    /// Cue command
    Cue = 0x03,
    /// Status request
    Status = 0x10,
}

impl VdcpCommand {
    /// Parse a raw command byte into a [`VdcpCommand`], if recognised.
    pub fn from_byte(b: u8) -> Option<Self> {
        match b {
            0x01 => Some(Self::Play),
            0x02 => Some(Self::Stop),
            0x03 => Some(Self::Cue),
            0x10 => Some(Self::Status),
            _ => None,
        }
    }
}

/// A parsed VDCP packet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VdcpPacket {
    /// The command.
    pub command: VdcpCommand,
    /// Optional payload bytes (e.g. timecode for [`VdcpCommand::Cue`]).
    pub data: Vec<u8>,
}

/// Serialise a [`VdcpPacket`] into a complete, framed byte buffer.
///
/// The resulting buffer can be written verbatim to a serial port.
pub fn serialize_vdcp_packet(packet: &VdcpPacket) -> Vec<u8> {
    let data_len = packet.data.len();
    let mut buf = BytesMut::with_capacity(5 + data_len);

    buf.put_u8(0x02); // STX
    buf.put_u8((data_len + 1) as u8); // LEN = len(CMD + DATA)
    buf.put_u8(packet.command as u8); // CMD

    for &b in &packet.data {
        buf.put_u8(b);
    }

    // CHK over LEN + CMD + DATA
    let checksum: u8 = buf[1..].iter().fold(0u8, |acc, &b| acc.wrapping_add(b));
    buf.put_u8(checksum);
    buf.put_u8(0x03); // ETX

    buf.to_vec()
}

/// Parse a raw byte buffer into a [`VdcpPacket`].
///
/// # Errors
///
/// Returns [`AutomationError::Protocol`] if:
/// - The buffer is too short (< 5 bytes).
/// - STX or ETX framing bytes are wrong.
/// - The checksum does not match.
/// - The command byte is not recognised.
pub fn parse_vdcp_packet(bytes: &[u8]) -> Result<VdcpPacket> {
    if bytes.len() < 5 {
        return Err(AutomationError::Protocol(
            "VDCP packet too short".to_string(),
        ));
    }

    let stx = bytes[0];
    let len = bytes[1] as usize;
    let etx = bytes[bytes.len() - 1];

    if stx != 0x02 {
        return Err(AutomationError::Protocol(format!(
            "invalid STX byte: {stx:#04x}"
        )));
    }
    if etx != 0x03 {
        return Err(AutomationError::Protocol(format!(
            "invalid ETX byte: {etx:#04x}"
        )));
    }

    // Expected total length: STX(1) + LEN(1) + CMD+DATA(len) + CHK(1) + ETX(1)
    let expected_total = 1 + 1 + len + 1 + 1;
    if bytes.len() < expected_total {
        return Err(AutomationError::Protocol(format!(
            "VDCP packet too short: expected {expected_total} bytes, got {}",
            bytes.len()
        )));
    }

    let cmd_byte = bytes[2];
    let data = bytes[3..2 + len].to_vec();
    let received_chk = bytes[2 + len];

    // Verify checksum: wrapping sum of bytes[1..2+len]
    let expected_chk: u8 = bytes[1..2 + len]
        .iter()
        .fold(0u8, |acc, &b| acc.wrapping_add(b));

    if received_chk != expected_chk {
        return Err(AutomationError::Protocol(format!(
            "VDCP checksum mismatch: expected {expected_chk:#04x}, got {received_chk:#04x}"
        )));
    }

    let command = VdcpCommand::from_byte(cmd_byte).ok_or_else(|| {
        AutomationError::Protocol(format!("unknown VDCP command byte: {cmd_byte:#04x}"))
    })?;

    Ok(VdcpPacket { command, data })
}

/// Encode a timecode string `HH:MM:SS:FF` into 4-byte LTC BCD format.
///
/// Each field is encoded as a BCD byte: `((value / 10) << 4) | (value % 10)`.
pub fn encode_timecode_ltc_bcd(timecode: &str) -> Result<[u8; 4]> {
    let parts: Vec<&str> = timecode.split(':').collect();
    if parts.len() != 4 {
        return Err(AutomationError::Protocol(format!(
            "invalid timecode: {timecode}"
        )));
    }

    let parse = |s: &str, field: &str| -> Result<u8> {
        s.parse::<u8>()
            .map_err(|_| AutomationError::Protocol(format!("invalid {field}: {s}")))
    };

    let h = parse(parts[0], "hours")?;
    let m = parse(parts[1], "minutes")?;
    let s = parse(parts[2], "seconds")?;
    let f = parse(parts[3], "frames")?;

    Ok([
        ((h / 10) << 4) | (h % 10),
        ((m / 10) << 4) | (m % 10),
        ((s / 10) << 4) | (s % 10),
        ((f / 10) << 4) | (f % 10),
    ])
}

/// Decode 4-byte LTC BCD timecode bytes into a `HH:MM:SS:FF` string.
pub fn decode_timecode_ltc_bcd(bcd: &[u8; 4]) -> String {
    let decode = |b: u8| -> u8 { ((b >> 4) * 10) + (b & 0x0F) };
    format!(
        "{:02}:{:02}:{:02}:{:02}",
        decode(bcd[0]),
        decode(bcd[1]),
        decode(bcd[2]),
        decode(bcd[3])
    )
}

/// VDCP status response flags.
///
/// In a real STATUS_SENSE response the device returns a bitmap where individual
/// bits indicate transport state.  This struct exposes the most common flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct VdcpStatusFlags(pub u8);

impl VdcpStatusFlags {
    /// `PLAY` bit — set when the transport is in play mode.
    pub const PLAY_BIT: u8 = 0x01;
    /// `STOP` bit — set when the transport is stopped.
    pub const STOP_BIT: u8 = 0x02;

    /// Return `true` if the PLAY bit is set.
    pub fn is_playing(self) -> bool {
        self.0 & Self::PLAY_BIT != 0
    }

    /// Return `true` if the STOP bit is set.
    pub fn is_stopped(self) -> bool {
        self.0 & Self::STOP_BIT != 0
    }
}

/// VDCP protocol handler.
pub struct VdcpProtocol {
    serial: SerialPort,
}

impl VdcpProtocol {
    /// Create a new VDCP protocol handler.
    pub async fn new(port: &str) -> Result<Self> {
        info!("Creating VDCP protocol on port: {}", port);

        let serial = SerialPort::new(port, 38400)?;

        Ok(Self { serial })
    }

    /// Close the connection.
    pub async fn close(&mut self) -> Result<()> {
        self.serial.close()
    }

    /// Send play command.
    pub async fn send_play(&mut self) -> Result<()> {
        debug!("Sending VDCP play command");
        self.send_command(VdcpCommand::Play, &[]).await
    }

    /// Send stop command.
    pub async fn send_stop(&mut self) -> Result<()> {
        debug!("Sending VDCP stop command");
        self.send_command(VdcpCommand::Stop, &[]).await
    }

    /// Send cue command.
    pub async fn send_cue(&mut self, timecode: &str) -> Result<()> {
        debug!("Sending VDCP cue command to: {}", timecode);

        // Parse timecode and encode as VDCP format
        let tc_bytes = self.encode_timecode(timecode)?;
        self.send_command(VdcpCommand::Cue, &tc_bytes).await
    }

    /// Get device status.
    pub async fn get_status(&mut self) -> Result<String> {
        debug!("Requesting VDCP status");
        self.send_command(VdcpCommand::Status, &[]).await?;

        // In a real implementation, this would read and parse the response
        Ok("OK".to_string())
    }

    /// Send a VDCP command.
    async fn send_command(&mut self, command: VdcpCommand, data: &[u8]) -> Result<()> {
        let mut buffer = BytesMut::with_capacity(256);

        // VDCP packet format: [STX][LEN][CMD][DATA...][CHK][ETX]
        buffer.put_u8(0x02); // STX
        buffer.put_u8((data.len() + 1) as u8); // Length
        buffer.put_u8(command as u8); // Command
        buffer.put_slice(data); // Data

        // Calculate checksum
        let checksum = self.calculate_checksum(&buffer[1..]);
        buffer.put_u8(checksum);
        buffer.put_u8(0x03); // ETX

        self.serial.write(&buffer)?;

        Ok(())
    }

    /// Encode timecode to VDCP format.
    fn encode_timecode(&self, timecode: &str) -> Result<Vec<u8>> {
        // Parse timecode format: HH:MM:SS:FF
        let parts: Vec<&str> = timecode.split(':').collect();
        if parts.len() != 4 {
            return Err(AutomationError::Protocol(format!(
                "Invalid timecode format: {timecode}"
            )));
        }

        let hours: u8 = parts[0]
            .parse()
            .map_err(|_| AutomationError::Protocol("Invalid hours".to_string()))?;
        let minutes: u8 = parts[1]
            .parse()
            .map_err(|_| AutomationError::Protocol("Invalid minutes".to_string()))?;
        let seconds: u8 = parts[2]
            .parse()
            .map_err(|_| AutomationError::Protocol("Invalid seconds".to_string()))?;
        let frames: u8 = parts[3]
            .parse()
            .map_err(|_| AutomationError::Protocol("Invalid frames".to_string()))?;

        Ok(vec![hours, minutes, seconds, frames])
    }

    /// Calculate VDCP checksum.
    fn calculate_checksum(&self, data: &[u8]) -> u8 {
        data.iter().fold(0u8, |acc, &byte| acc.wrapping_add(byte))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_timecode() {
        let protocol = VdcpProtocol {
            serial: SerialPort::mock(),
        };

        let result = protocol.encode_timecode("01:23:45:12");
        assert!(result.is_ok());

        let tc = result.expect("tc should be valid");
        assert_eq!(tc, vec![1, 23, 45, 12]);
    }

    #[test]
    fn test_invalid_timecode() {
        let protocol = VdcpProtocol {
            serial: SerialPort::mock(),
        };

        let result = protocol.encode_timecode("invalid");
        assert!(result.is_err());
    }

    #[test]
    fn test_calculate_checksum() {
        let protocol = VdcpProtocol {
            serial: SerialPort::mock(),
        };

        let data = vec![0x01, 0x02, 0x03];
        let checksum = protocol.calculate_checksum(&data);
        assert_eq!(checksum, 0x06);
    }

    // ── Serialization round-trip tests ────────────────────────────────────────

    #[test]
    fn test_vdcp_play_serialize_parse_roundtrip() {
        let packet = VdcpPacket {
            command: VdcpCommand::Play,
            data: vec![],
        };
        let bytes = serialize_vdcp_packet(&packet);
        let parsed = parse_vdcp_packet(&bytes).expect("should parse Play packet");
        assert_eq!(parsed.command, VdcpCommand::Play);
        assert!(parsed.data.is_empty());
    }

    #[test]
    fn test_vdcp_stop_serialize_parse_roundtrip() {
        let packet = VdcpPacket {
            command: VdcpCommand::Stop,
            data: vec![],
        };
        let bytes = serialize_vdcp_packet(&packet);
        let parsed = parse_vdcp_packet(&bytes).expect("should parse Stop packet");
        assert_eq!(parsed.command, VdcpCommand::Stop);
        assert!(parsed.data.is_empty());
    }

    #[test]
    fn test_vdcp_cue_with_timecode_roundtrip() {
        // Cue to timecode 01:02:03:04
        let tc_data = vec![1u8, 2, 3, 4];
        let packet = VdcpPacket {
            command: VdcpCommand::Cue,
            data: tc_data.clone(),
        };
        let bytes = serialize_vdcp_packet(&packet);
        let parsed = parse_vdcp_packet(&bytes).expect("should parse Cue packet");
        assert_eq!(parsed.command, VdcpCommand::Cue);
        assert_eq!(parsed.data, tc_data);
    }

    #[test]
    fn test_vdcp_status_serialize_parse_roundtrip() {
        let packet = VdcpPacket {
            command: VdcpCommand::Status,
            data: vec![],
        };
        let bytes = serialize_vdcp_packet(&packet);
        let parsed = parse_vdcp_packet(&bytes).expect("should parse Status packet");
        assert_eq!(parsed.command, VdcpCommand::Status);
    }

    #[test]
    fn test_vdcp_framing_bytes_correct() {
        let packet = VdcpPacket {
            command: VdcpCommand::Play,
            data: vec![],
        };
        let bytes = serialize_vdcp_packet(&packet);
        assert_eq!(bytes[0], 0x02, "STX must be 0x02");
        assert_eq!(*bytes.last().expect("ETX"), 0x03, "ETX must be 0x03");
    }

    #[test]
    fn test_vdcp_checksum_verified_on_parse() {
        let packet = VdcpPacket {
            command: VdcpCommand::Play,
            data: vec![],
        };
        let mut bytes = serialize_vdcp_packet(&packet);
        // Corrupt the checksum byte (second-to-last).
        let chk_idx = bytes.len() - 2;
        bytes[chk_idx] ^= 0xFF;
        let result = parse_vdcp_packet(&bytes);
        assert!(result.is_err(), "corrupted checksum must produce error");
    }

    #[test]
    fn test_vdcp_parse_too_short_returns_error() {
        let result = parse_vdcp_packet(&[0x02, 0x01]);
        assert!(result.is_err(), "short packet must produce error");
    }

    // ── LTC BCD timecode round-trip tests ─────────────────────────────────────

    #[test]
    fn test_ltc_bcd_roundtrip_midnight() {
        let tc = "00:00:00:00";
        let bcd = encode_timecode_ltc_bcd(tc).expect("encode");
        let decoded = decode_timecode_ltc_bcd(&bcd);
        assert_eq!(decoded, tc);
    }

    #[test]
    fn test_ltc_bcd_roundtrip_one_hour_mark() {
        let tc = "01:00:00:00";
        let bcd = encode_timecode_ltc_bcd(tc).expect("encode");
        let decoded = decode_timecode_ltc_bcd(&bcd);
        assert_eq!(decoded, tc);
    }

    #[test]
    fn test_ltc_bcd_roundtrip_arbitrary() {
        let tc = "12:34:56:29";
        let bcd = encode_timecode_ltc_bcd(tc).expect("encode");
        let decoded = decode_timecode_ltc_bcd(&bcd);
        assert_eq!(decoded, tc);
    }

    #[test]
    fn test_ltc_bcd_encoding_known_value() {
        // "01:23:45:12" → [0x01, 0x23, 0x45, 0x12]
        let bcd = encode_timecode_ltc_bcd("01:23:45:12").expect("encode");
        assert_eq!(bcd, [0x01, 0x23, 0x45, 0x12]);
    }

    #[test]
    fn test_ltc_bcd_invalid_format_returns_error() {
        let result = encode_timecode_ltc_bcd("not-a-timecode");
        assert!(result.is_err());
    }

    // ── Status flag tests ──────────────────────────────────────────────────────

    #[test]
    fn test_vdcp_status_play_bit() {
        let flags = VdcpStatusFlags(VdcpStatusFlags::PLAY_BIT);
        assert!(flags.is_playing());
        assert!(!flags.is_stopped());
    }

    #[test]
    fn test_vdcp_status_stop_bit() {
        let flags = VdcpStatusFlags(VdcpStatusFlags::STOP_BIT);
        assert!(!flags.is_playing());
        assert!(flags.is_stopped());
    }

    #[test]
    fn test_vdcp_status_neither_bit() {
        let flags = VdcpStatusFlags(0x00);
        assert!(!flags.is_playing());
        assert!(!flags.is_stopped());
    }
}
