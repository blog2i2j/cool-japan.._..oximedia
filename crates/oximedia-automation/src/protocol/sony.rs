//! Sony 9-pin (BVW-75) RS-422 protocol implementation.
//!
//! ## Packet format
//!
//! Sony 9-pin uses a 7-byte packet:
//!
//! ```text
//! [CMD1][CMD2][DATA1][DATA2][DATA3][DATA4][CHK]
//! ```
//!
//! - **CMD1**: transport-control group (`0x20` for all motion commands).
//! - **CMD2**: specific command within the group.
//! - **DATA1–DATA4**: optional payload (usually zeros for simple commands).
//! - **CHK**: wrapping-add checksum of CMD1..DATA4.
//!
//! Common CMD1/CMD2 pairs (BVW-75 / BVH-2000):
//!
//! | Command      | CMD1   | CMD2   |
//! |--------------|--------|--------|
//! | STOP         | `0x20` | `0x00` |
//! | PLAY         | `0x20` | `0x01` |
//! | RECORD       | `0x20` | `0x02` |
//! | FAST FORWARD | `0x20` | `0x10` |
//! | REWIND       | `0x20` | `0x20` |

use crate::protocol::serial::SerialPort;
use crate::{AutomationError, Result};
use bytes::{BufMut, BytesMut};
use tracing::{debug, info};

/// Sony 9-pin command group byte for transport control.
///
/// All standard motion commands share the same CMD1 group byte (`0x20`).
pub const SONY_CMD1_TRANSPORT: u8 = 0x20;

/// Sony 9-pin command codes (CMD2 bytes for the transport group).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SonyCommand {
    /// Stop command — `[0x20, 0x00]`
    Stop = 0x00,
    /// Play command — `[0x20, 0x01]`
    Play = 0x01,
    /// Record command — `[0x20, 0x02]`
    Record = 0x02,
    /// Fast forward command — `[0x20, 0x10]`
    FastForward = 0x10,
    /// Rewind command — `[0x20, 0x20]`
    Rewind = 0x20,
}

impl SonyCommand {
    /// Parse a CMD2 byte into a [`SonyCommand`], if recognised.
    pub fn from_byte(cmd2: u8) -> Option<Self> {
        match cmd2 {
            0x00 => Some(Self::Stop),
            0x01 => Some(Self::Play),
            0x02 => Some(Self::Record),
            0x10 => Some(Self::FastForward),
            0x20 => Some(Self::Rewind),
            _ => None,
        }
    }
}

/// A serialised Sony 9-pin packet (7 bytes: CMD1, CMD2, DATA×4, CHK).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SonyPacket {
    /// Transport-control group byte (always [`SONY_CMD1_TRANSPORT`] = `0x20`
    /// for standard motion commands).
    pub cmd1: u8,
    /// Command-specific byte.
    pub command: SonyCommand,
    /// Optional data payload (bytes DATA1–DATA4).
    pub data: [u8; 4],
}

impl SonyPacket {
    /// Create a simple command packet with all-zero data.
    pub fn simple(command: SonyCommand) -> Self {
        Self {
            cmd1: SONY_CMD1_TRANSPORT,
            command,
            data: [0u8; 4],
        }
    }
}

/// Serialise a [`SonyPacket`] into a 7-byte buffer.
pub fn serialize_sony_packet(packet: &SonyPacket) -> [u8; 7] {
    let mut buf = [0u8; 7];
    buf[0] = packet.cmd1;
    buf[1] = packet.command as u8;
    buf[2] = packet.data[0];
    buf[3] = packet.data[1];
    buf[4] = packet.data[2];
    buf[5] = packet.data[3];
    // Checksum: wrapping sum of bytes 0..6
    buf[6] = buf[0..6].iter().fold(0u8, |acc, &b| acc.wrapping_add(b));
    buf
}

/// Parse a 7-byte buffer into a [`SonyPacket`].
///
/// # Errors
///
/// Returns [`AutomationError::Protocol`] if:
/// - The buffer is not exactly 7 bytes.
/// - The checksum does not match.
/// - The CMD2 byte is not a recognised [`SonyCommand`].
pub fn parse_sony_packet(bytes: &[u8]) -> Result<SonyPacket> {
    if bytes.len() != 7 {
        return Err(AutomationError::Protocol(format!(
            "Sony 9-pin packet must be exactly 7 bytes, got {}",
            bytes.len()
        )));
    }

    let expected_chk: u8 = bytes[0..6].iter().fold(0u8, |acc, &b| acc.wrapping_add(b));
    let received_chk = bytes[6];

    if received_chk != expected_chk {
        return Err(AutomationError::Protocol(format!(
            "Sony 9-pin checksum mismatch: expected {expected_chk:#04x}, got {received_chk:#04x}"
        )));
    }

    let command = SonyCommand::from_byte(bytes[1]).ok_or_else(|| {
        AutomationError::Protocol(format!("unknown Sony 9-pin CMD2 byte: {:#04x}", bytes[1]))
    })?;

    Ok(SonyPacket {
        cmd1: bytes[0],
        command,
        data: [bytes[2], bytes[3], bytes[4], bytes[5]],
    })
}

/// Sony 9-pin STATUS_SENSE response flags.
///
/// The device returns an 8-byte status block.  Byte 0 contains transport-state
/// flags; byte 1 contains error flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SonyStatusFlags {
    /// Transport state byte (byte 0 of STATUS_SENSE response).
    pub transport: u8,
    /// Error flags byte (byte 1 of STATUS_SENSE response).
    pub errors: u8,
}

impl SonyStatusFlags {
    /// `PLAY` bit in the transport byte — set when transport is in play mode.
    pub const PLAY_BIT: u8 = 0x01;
    /// `STOP` bit in the transport byte.
    pub const STOP_BIT: u8 = 0x02;
    /// `RECORD` bit in the transport byte.
    pub const RECORD_BIT: u8 = 0x04;

    /// Parse from the first two bytes of a STATUS_SENSE response.
    pub fn from_status_bytes(transport: u8, errors: u8) -> Self {
        Self { transport, errors }
    }

    /// Return `true` if the PLAY bit is asserted.
    pub fn is_playing(self) -> bool {
        self.transport & Self::PLAY_BIT != 0
    }

    /// Return `true` if the STOP bit is asserted.
    pub fn is_stopped(self) -> bool {
        self.transport & Self::STOP_BIT != 0
    }

    /// Return `true` if the RECORD bit is asserted.
    pub fn is_recording(self) -> bool {
        self.transport & Self::RECORD_BIT != 0
    }

    /// Return `true` if any error bits are set.
    pub fn has_errors(self) -> bool {
        self.errors != 0
    }
}

/// Sony protocol handler.
pub struct SonyProtocol {
    serial: SerialPort,
}

impl SonyProtocol {
    /// Create a new Sony protocol handler.
    pub async fn new(port: &str) -> Result<Self> {
        info!("Creating Sony 9-pin protocol on port: {}", port);

        let serial = SerialPort::new(port, 38400)?;

        Ok(Self { serial })
    }

    /// Close the connection.
    pub async fn close(&mut self) -> Result<()> {
        self.serial.close()
    }

    /// Send play command.
    pub async fn send_play(&mut self) -> Result<()> {
        debug!("Sending Sony play command");
        self.send_command(SonyCommand::Play).await
    }

    /// Send stop command.
    pub async fn send_stop(&mut self) -> Result<()> {
        debug!("Sending Sony stop command");
        self.send_command(SonyCommand::Stop).await
    }

    /// Send record command.
    pub async fn send_record(&mut self) -> Result<()> {
        debug!("Sending Sony record command");
        self.send_command(SonyCommand::Record).await
    }

    /// Send fast forward command.
    pub async fn send_fast_forward(&mut self) -> Result<()> {
        debug!("Sending Sony fast forward command");
        self.send_command(SonyCommand::FastForward).await
    }

    /// Send rewind command.
    pub async fn send_rewind(&mut self) -> Result<()> {
        debug!("Sending Sony rewind command");
        self.send_command(SonyCommand::Rewind).await
    }

    /// Send a Sony 9-pin command.
    async fn send_command(&mut self, command: SonyCommand) -> Result<()> {
        let mut buffer = BytesMut::with_capacity(16);

        // Sony 9-pin packet format: [CMD1][CMD2][DATA1][DATA2][DATA3][DATA4][CHK]
        buffer.put_u8(command as u8);
        buffer.put_u8(0x00); // CMD2
        buffer.put_u8(0x00); // DATA1
        buffer.put_u8(0x00); // DATA2
        buffer.put_u8(0x00); // DATA3
        buffer.put_u8(0x00); // DATA4

        // Calculate checksum
        let checksum = self.calculate_checksum(&buffer);
        buffer.put_u8(checksum);

        self.serial.write(&buffer)?;

        Ok(())
    }

    /// Calculate Sony protocol checksum.
    fn calculate_checksum(&self, data: &[u8]) -> u8 {
        data.iter().fold(0u8, |acc, &byte| acc.wrapping_add(byte))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_checksum() {
        let protocol = SonyProtocol {
            serial: SerialPort::mock(),
        };

        let data = vec![0x01, 0x00, 0x00, 0x00, 0x00, 0x00];
        let checksum = protocol.calculate_checksum(&data);
        assert_eq!(checksum, 0x01);
    }

    // ── BVW-75 byte-level command encoding ────────────────────────────────────

    #[test]
    fn test_sony_stop_bytes_are_0x20_0x00() {
        let bytes = serialize_sony_packet(&SonyPacket::simple(SonyCommand::Stop));
        assert_eq!(bytes[0], 0x20, "STOP CMD1 must be 0x20");
        assert_eq!(bytes[1], 0x00, "STOP CMD2 must be 0x00");
    }

    #[test]
    fn test_sony_play_bytes_are_0x20_0x01() {
        let bytes = serialize_sony_packet(&SonyPacket::simple(SonyCommand::Play));
        assert_eq!(bytes[0], 0x20, "PLAY CMD1 must be 0x20");
        assert_eq!(bytes[1], 0x01, "PLAY CMD2 must be 0x01");
    }

    #[test]
    fn test_sony_record_bytes_are_0x20_0x02() {
        let bytes = serialize_sony_packet(&SonyPacket::simple(SonyCommand::Record));
        assert_eq!(bytes[0], 0x20);
        assert_eq!(bytes[1], 0x02);
    }

    #[test]
    fn test_sony_fast_forward_bytes_are_0x20_0x10() {
        let bytes = serialize_sony_packet(&SonyPacket::simple(SonyCommand::FastForward));
        assert_eq!(bytes[0], 0x20);
        assert_eq!(bytes[1], 0x10);
    }

    #[test]
    fn test_sony_rewind_bytes_are_0x20_0x20() {
        let bytes = serialize_sony_packet(&SonyPacket::simple(SonyCommand::Rewind));
        assert_eq!(bytes[0], 0x20);
        assert_eq!(bytes[1], 0x20);
    }

    // ── Serialization round-trip tests ────────────────────────────────────────

    #[test]
    fn test_sony_stop_roundtrip() {
        let original = SonyPacket::simple(SonyCommand::Stop);
        let bytes = serialize_sony_packet(&original);
        let parsed = parse_sony_packet(&bytes).expect("should parse STOP packet");
        assert_eq!(parsed.command, SonyCommand::Stop);
    }

    #[test]
    fn test_sony_play_roundtrip() {
        let original = SonyPacket::simple(SonyCommand::Play);
        let bytes = serialize_sony_packet(&original);
        let parsed = parse_sony_packet(&bytes).expect("should parse PLAY packet");
        assert_eq!(parsed.command, SonyCommand::Play);
    }

    #[test]
    fn test_sony_packet_length_always_7() {
        for cmd in [
            SonyCommand::Stop,
            SonyCommand::Play,
            SonyCommand::Record,
            SonyCommand::FastForward,
            SonyCommand::Rewind,
        ] {
            let bytes = serialize_sony_packet(&SonyPacket::simple(cmd));
            assert_eq!(bytes.len(), 7, "{cmd:?} packet must be exactly 7 bytes");
        }
    }

    #[test]
    fn test_sony_checksum_verified_on_parse() {
        let bytes = serialize_sony_packet(&SonyPacket::simple(SonyCommand::Play));
        let mut corrupt = bytes;
        corrupt[6] ^= 0xFF; // corrupt checksum
        let result = parse_sony_packet(&corrupt);
        assert!(result.is_err(), "corrupted checksum must produce error");
    }

    #[test]
    fn test_sony_parse_wrong_length_returns_error() {
        let result = parse_sony_packet(&[0x20, 0x01, 0x00]);
        assert!(result.is_err(), "wrong-length buffer must produce error");
    }

    // ── STATUS_SENSE response parsing ─────────────────────────────────────────

    #[test]
    fn test_sony_status_play_bit() {
        let flags = SonyStatusFlags::from_status_bytes(SonyStatusFlags::PLAY_BIT, 0x00);
        assert!(flags.is_playing());
        assert!(!flags.is_stopped());
        assert!(!flags.is_recording());
        assert!(!flags.has_errors());
    }

    #[test]
    fn test_sony_status_stop_bit() {
        let flags = SonyStatusFlags::from_status_bytes(SonyStatusFlags::STOP_BIT, 0x00);
        assert!(!flags.is_playing());
        assert!(flags.is_stopped());
    }

    #[test]
    fn test_sony_status_error_flags() {
        let flags = SonyStatusFlags::from_status_bytes(0x00, 0x01);
        assert!(flags.has_errors(), "error byte 0x01 should set has_errors");
    }

    #[test]
    fn test_sony_status_no_flags() {
        let flags = SonyStatusFlags::default();
        assert!(!flags.is_playing());
        assert!(!flags.is_stopped());
        assert!(!flags.has_errors());
    }
}
