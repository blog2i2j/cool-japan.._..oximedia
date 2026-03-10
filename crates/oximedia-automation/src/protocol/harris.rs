//! Harris Automation (Nexio) binary protocol simulation.
//!
//! Binary frame format:
//! ```text
//! [0x48][0x41][TYPE:u8][SLOT:u32 BE][PARAMS:N bytes]
//! ```
//! Header bytes are ASCII 'H' and 'A' (Harris Automation).

#![allow(dead_code)]

/// Harris command type codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HarrisCommandType {
    /// Play a clip
    Play,
    /// Stop playback
    Stop,
    /// Start recording
    Record,
    /// Cue to a position
    Cue,
    /// Load a clip
    Load,
    /// Eject media
    Eject,
    /// Query device status
    Status,
}

impl HarrisCommandType {
    /// Return the single-byte command code for this type.
    #[must_use]
    pub fn code(&self) -> u8 {
        match self {
            HarrisCommandType::Play => 0x01,
            HarrisCommandType::Stop => 0x02,
            HarrisCommandType::Record => 0x03,
            HarrisCommandType::Cue => 0x04,
            HarrisCommandType::Load => 0x05,
            HarrisCommandType::Eject => 0x06,
            HarrisCommandType::Status => 0x07,
        }
    }

    /// Parse a command type from its byte code.
    pub fn from_code(code: u8) -> Option<Self> {
        match code {
            0x01 => Some(HarrisCommandType::Play),
            0x02 => Some(HarrisCommandType::Stop),
            0x03 => Some(HarrisCommandType::Record),
            0x04 => Some(HarrisCommandType::Cue),
            0x05 => Some(HarrisCommandType::Load),
            0x06 => Some(HarrisCommandType::Eject),
            0x07 => Some(HarrisCommandType::Status),
            _ => None,
        }
    }
}

/// A Harris automation command.
#[derive(Debug, Clone)]
pub struct HarrisCommand {
    /// Command type
    pub command_type: HarrisCommandType,
    /// Target slot / deck number
    pub slot: u32,
    /// Optional string parameters (e.g. clip ID, timecode)
    pub params: Vec<String>,
}

impl HarrisCommand {
    /// Create a new `HarrisCommand`.
    pub fn new(command_type: HarrisCommandType, slot: u32) -> Self {
        Self {
            command_type,
            slot,
            params: Vec::new(),
        }
    }

    /// Create a command with parameters.
    pub fn with_params(command_type: HarrisCommandType, slot: u32, params: Vec<String>) -> Self {
        Self {
            command_type,
            slot,
            params,
        }
    }
}

/// Harris response status codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HarrisStatus {
    /// Command accepted and executed
    Ok,
    /// Generic error
    Error,
    /// Device is busy
    Busy,
    /// Requested resource not found
    NotFound,
    /// Slot number is out of range
    InvalidSlot,
}

impl HarrisStatus {
    /// Parse a status from a byte value.
    pub fn from_byte(b: u8) -> Option<Self> {
        match b {
            0x00 => Some(HarrisStatus::Ok),
            0x01 => Some(HarrisStatus::Error),
            0x02 => Some(HarrisStatus::Busy),
            0x03 => Some(HarrisStatus::NotFound),
            0x04 => Some(HarrisStatus::InvalidSlot),
            _ => None,
        }
    }

    /// Return the byte representation of this status.
    #[must_use]
    pub fn as_byte(&self) -> u8 {
        match self {
            HarrisStatus::Ok => 0x00,
            HarrisStatus::Error => 0x01,
            HarrisStatus::Busy => 0x02,
            HarrisStatus::NotFound => 0x03,
            HarrisStatus::InvalidSlot => 0x04,
        }
    }
}

/// A response received from a Harris device.
#[derive(Debug, Clone)]
pub struct HarrisResponse {
    /// Response status
    pub status: HarrisStatus,
    /// Raw response payload
    pub data: Vec<u8>,
}

impl HarrisResponse {
    /// Create a new `HarrisResponse`.
    pub fn new(status: HarrisStatus, data: Vec<u8>) -> Self {
        Self { status, data }
    }

    /// Convenience constructor for a simple OK response.
    pub fn ok() -> Self {
        Self::new(HarrisStatus::Ok, Vec::new())
    }
}

/// Harris binary protocol encoder/decoder.
pub struct HarrisProtocol;

impl HarrisProtocol {
    /// Encode a [`HarrisCommand`] into the binary wire format.
    ///
    /// Layout:
    /// ```text
    /// [0x48][0x41][TYPE:1][SLOT:4 BE][PARAM_LEN:1][PARAMS…]
    /// ```
    #[must_use]
    pub fn encode_command(cmd: &HarrisCommand) -> Vec<u8> {
        let params_bytes: Vec<u8> = cmd
            .params
            .iter()
            .flat_map(|p| {
                let mut v = p.as_bytes().to_vec();
                v.push(0x00); // null-terminate each param
                v
            })
            .collect();

        let mut buf = Vec::with_capacity(7 + params_bytes.len());
        buf.push(0x48); // 'H'
        buf.push(0x41); // 'A'
        buf.push(cmd.command_type.code());
        buf.extend_from_slice(&cmd.slot.to_be_bytes());
        buf.extend_from_slice(&params_bytes);
        buf
    }

    /// Decode a [`HarrisResponse`] from raw bytes.
    ///
    /// Expected layout: `[STATUS:1][DATA…]`
    ///
    /// # Errors
    /// Returns `Err` if the data slice is empty or the status byte is unknown.
    pub fn decode_response(data: &[u8]) -> Result<HarrisResponse, String> {
        if data.is_empty() {
            return Err("Empty response data".to_string());
        }
        let status = HarrisStatus::from_byte(data[0])
            .ok_or_else(|| format!("Unknown status byte: 0x{:02X}", data[0]))?;

        Ok(HarrisResponse {
            status,
            data: data[1..].to_vec(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_command_type_codes() {
        assert_eq!(HarrisCommandType::Play.code(), 0x01);
        assert_eq!(HarrisCommandType::Stop.code(), 0x02);
        assert_eq!(HarrisCommandType::Record.code(), 0x03);
        assert_eq!(HarrisCommandType::Cue.code(), 0x04);
        assert_eq!(HarrisCommandType::Load.code(), 0x05);
        assert_eq!(HarrisCommandType::Eject.code(), 0x06);
        assert_eq!(HarrisCommandType::Status.code(), 0x07);
    }

    #[test]
    fn test_command_type_roundtrip() {
        for code in 0x01u8..=0x07 {
            let t = HarrisCommandType::from_code(code).expect("valid code");
            assert_eq!(t.code(), code);
        }
    }

    #[test]
    fn test_command_type_unknown_code() {
        assert!(HarrisCommandType::from_code(0xFF).is_none());
    }

    #[test]
    fn test_encode_command_header() {
        let cmd = HarrisCommand::new(HarrisCommandType::Play, 1);
        let encoded = HarrisProtocol::encode_command(&cmd);
        assert_eq!(encoded[0], 0x48);
        assert_eq!(encoded[1], 0x41);
        assert_eq!(encoded[2], 0x01); // Play
    }

    #[test]
    fn test_encode_command_slot() {
        let cmd = HarrisCommand::new(HarrisCommandType::Stop, 42);
        let encoded = HarrisProtocol::encode_command(&cmd);
        let slot = u32::from_be_bytes([encoded[3], encoded[4], encoded[5], encoded[6]]);
        assert_eq!(slot, 42);
    }

    #[test]
    fn test_encode_command_with_params() {
        let cmd =
            HarrisCommand::with_params(HarrisCommandType::Load, 0, vec!["clip001".to_string()]);
        let encoded = HarrisProtocol::encode_command(&cmd);
        // Should contain clip name bytes after slot
        assert!(encoded.len() > 7);
    }

    #[test]
    fn test_decode_response_ok() {
        let data = vec![0x00, 0xDE, 0xAD];
        let resp = HarrisProtocol::decode_response(&data).expect("decode_response should succeed");
        assert_eq!(resp.status, HarrisStatus::Ok);
        assert_eq!(resp.data, vec![0xDE, 0xAD]);
    }

    #[test]
    fn test_decode_response_busy() {
        let data = vec![HarrisStatus::Busy.as_byte()];
        let resp = HarrisProtocol::decode_response(&data).expect("decode_response should succeed");
        assert_eq!(resp.status, HarrisStatus::Busy);
    }

    #[test]
    fn test_decode_response_empty_error() {
        let result = HarrisProtocol::decode_response(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_harris_status_as_byte_roundtrip() {
        let statuses = [
            HarrisStatus::Ok,
            HarrisStatus::Error,
            HarrisStatus::Busy,
            HarrisStatus::NotFound,
            HarrisStatus::InvalidSlot,
        ];
        for s in statuses {
            let b = s.as_byte();
            let parsed = HarrisStatus::from_byte(b).expect("from_byte should succeed");
            assert_eq!(parsed, s);
        }
    }
}
