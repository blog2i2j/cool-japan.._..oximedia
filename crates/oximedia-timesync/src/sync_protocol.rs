//! Synchronization protocol versioning, message framing, and encode/decode.
//!
//! This module defines a lightweight wire format for exchanging synchronization
//! state between OxiMedia nodes, independent of PTP or NTP specifics.

#![allow(dead_code)]

use std::fmt;

/// Protocol version negotiated between peers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ProtocolVersion {
    /// Initial version (basic offset exchange).
    V1 = 1,
    /// Version 2 adds frequency error and path delay fields.
    V2 = 2,
    /// Version 3 adds extended quality metrics.
    V3 = 3,
}

impl ProtocolVersion {
    /// Returns the wire integer for this version.
    pub fn wire_value(self) -> u8 {
        self as u8
    }

    /// Attempt to parse a wire value into a `ProtocolVersion`.
    ///
    /// Returns `None` if the value is unrecognised.
    pub fn from_wire(value: u8) -> Option<Self> {
        match value {
            1 => Some(Self::V1),
            2 => Some(Self::V2),
            3 => Some(Self::V3),
            _ => None,
        }
    }

    /// Returns the highest version supported by this build.
    pub fn latest() -> Self {
        Self::V3
    }

    /// Negotiate the highest mutually-supported version.
    pub fn negotiate(local: Self, remote: Self) -> Self {
        local.min(remote)
    }
}

impl fmt::Display for ProtocolVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "v{}", self.wire_value())
    }
}

/// Type tag for a `SyncMessage`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MessageType {
    /// Announce: a peer declares its presence and version.
    Announce = 0x01,
    /// Request: a peer requests an offset measurement.
    Request = 0x02,
    /// Response: carries the measured offset back to the requester.
    Response = 0x03,
    /// Follow-up: correction factor for two-step protocols.
    FollowUp = 0x04,
    /// Goodbye: orderly shutdown notification.
    Goodbye = 0xFF,
}

impl MessageType {
    /// Parse from a raw byte.
    pub fn from_byte(b: u8) -> Option<Self> {
        match b {
            0x01 => Some(Self::Announce),
            0x02 => Some(Self::Request),
            0x03 => Some(Self::Response),
            0x04 => Some(Self::FollowUp),
            0xFF => Some(Self::Goodbye),
            _ => None,
        }
    }
}

/// A synchronization protocol message.
///
/// The serialised format (little-endian) is:
/// ```text
/// [version: u8][type: u8][seq: u16][origin_ns: i64][receive_ns: i64][correction_ns: i64]
/// ```
/// Total: 28 bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyncMessage {
    /// Protocol version used for this message.
    pub version: ProtocolVersion,
    /// Message type tag.
    pub msg_type: MessageType,
    /// Sequence number for matching requests to responses.
    pub sequence: u16,
    /// Timestamp at origin in nanoseconds since epoch.
    pub origin_ns: i64,
    /// Timestamp at receive in nanoseconds since epoch (0 for requests).
    pub receive_ns: i64,
    /// Correction field in nanoseconds (for follow-up messages).
    pub correction_ns: i64,
}

/// Wire size of a serialised `SyncMessage`.
pub const SYNC_MESSAGE_SIZE: usize = 28;

impl SyncMessage {
    /// Create a new request message.
    pub fn new_request(version: ProtocolVersion, sequence: u16, origin_ns: i64) -> Self {
        Self {
            version,
            msg_type: MessageType::Request,
            sequence,
            origin_ns,
            receive_ns: 0,
            correction_ns: 0,
        }
    }

    /// Create a response message mirroring a request.
    pub fn new_response(request: &SyncMessage, receive_ns: i64) -> Self {
        Self {
            version: request.version,
            msg_type: MessageType::Response,
            sequence: request.sequence,
            origin_ns: request.origin_ns,
            receive_ns,
            correction_ns: 0,
        }
    }

    /// Create an announce message.
    pub fn new_announce(version: ProtocolVersion, sequence: u16, origin_ns: i64) -> Self {
        Self {
            version,
            msg_type: MessageType::Announce,
            sequence,
            origin_ns,
            receive_ns: 0,
            correction_ns: 0,
        }
    }

    /// Compute the one-way offset estimate given the round-trip timestamps.
    ///
    /// `send_ns` is when the request left the master; `receive_back_ns` is when
    /// the response was received by the master.
    ///
    /// Returns `None` if `receive_ns` is zero (not a response).
    pub fn estimate_offset(&self, send_ns: i64, receive_back_ns: i64) -> Option<i64> {
        if self.receive_ns == 0 {
            return None;
        }
        // offset = ((t2 - t1) - (t4 - t3)) / 2
        // t1 = origin_ns (send_ns), t2 = receive_ns, t3 = receive_ns (reuse), t4 = receive_back_ns
        let delay = (self.receive_ns - send_ns) + (receive_back_ns - self.receive_ns);
        Some((self.receive_ns - send_ns) - delay / 2)
    }
}

/// Errors that can occur during protocol encode/decode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProtocolError {
    /// Buffer too small for a complete message.
    BufferTooSmall {
        /// Number of bytes needed.
        needed: usize,
        /// Number of bytes available.
        got: usize,
    },
    /// Unknown version byte.
    UnknownVersion(u8),
    /// Unknown message type byte.
    UnknownMessageType(u8),
}

impl fmt::Display for ProtocolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BufferTooSmall { needed, got } => {
                write!(f, "buffer too small: need {needed}, got {got}")
            }
            Self::UnknownVersion(v) => write!(f, "unknown protocol version: {v}"),
            Self::UnknownMessageType(t) => write!(f, "unknown message type: {t:#04x}"),
        }
    }
}

/// Handles encoding and decoding of `SyncMessage` frames.
#[derive(Debug, Clone)]
pub struct SyncProtocol {
    /// The local protocol version.
    pub local_version: ProtocolVersion,
    /// Sequence counter for outgoing messages.
    sequence_counter: u16,
}

impl SyncProtocol {
    /// Create a new protocol handler with the latest version.
    pub fn new() -> Self {
        Self {
            local_version: ProtocolVersion::latest(),
            sequence_counter: 0,
        }
    }

    /// Create a protocol handler with a specific version.
    pub fn with_version(version: ProtocolVersion) -> Self {
        Self {
            local_version: version,
            sequence_counter: 0,
        }
    }

    /// Return and increment the sequence counter.
    pub fn next_sequence(&mut self) -> u16 {
        let seq = self.sequence_counter;
        self.sequence_counter = self.sequence_counter.wrapping_add(1);
        seq
    }

    /// Encode a `SyncMessage` into a fixed-size byte array.
    pub fn encode(msg: &SyncMessage) -> [u8; SYNC_MESSAGE_SIZE] {
        let mut buf = [0u8; SYNC_MESSAGE_SIZE];
        buf[0] = msg.version.wire_value();
        buf[1] = msg.msg_type as u8;
        buf[2..4].copy_from_slice(&msg.sequence.to_le_bytes());
        buf[4..12].copy_from_slice(&msg.origin_ns.to_le_bytes());
        buf[12..20].copy_from_slice(&msg.receive_ns.to_le_bytes());
        buf[20..28].copy_from_slice(&msg.correction_ns.to_le_bytes());
        buf
    }

    /// Decode a `SyncMessage` from a byte slice.
    ///
    /// # Errors
    ///
    /// Returns a `ProtocolError` if the buffer is too small, or contains an
    /// unrecognised version or message type.
    pub fn decode(buf: &[u8]) -> Result<SyncMessage, ProtocolError> {
        if buf.len() < SYNC_MESSAGE_SIZE {
            return Err(ProtocolError::BufferTooSmall {
                needed: SYNC_MESSAGE_SIZE,
                got: buf.len(),
            });
        }
        let version =
            ProtocolVersion::from_wire(buf[0]).ok_or(ProtocolError::UnknownVersion(buf[0]))?;
        let msg_type =
            MessageType::from_byte(buf[1]).ok_or(ProtocolError::UnknownMessageType(buf[1]))?;
        let sequence = u16::from_le_bytes([buf[2], buf[3]]);
        let origin_ns = i64::from_le_bytes(<[u8; 8]>::try_from(&buf[4..12]).map_err(|_| {
            ProtocolError::BufferTooSmall {
                needed: 12,
                got: buf.len(),
            }
        })?);
        let receive_ns = i64::from_le_bytes(<[u8; 8]>::try_from(&buf[12..20]).map_err(|_| {
            ProtocolError::BufferTooSmall {
                needed: 20,
                got: buf.len(),
            }
        })?);
        let correction_ns =
            i64::from_le_bytes(<[u8; 8]>::try_from(&buf[20..28]).map_err(|_| {
                ProtocolError::BufferTooSmall {
                    needed: 28,
                    got: buf.len(),
                }
            })?);
        Ok(SyncMessage {
            version,
            msg_type,
            sequence,
            origin_ns,
            receive_ns,
            correction_ns,
        })
    }
}

impl Default for SyncProtocol {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_protocol_version_wire() {
        assert_eq!(ProtocolVersion::V1.wire_value(), 1);
        assert_eq!(ProtocolVersion::V2.wire_value(), 2);
        assert_eq!(ProtocolVersion::V3.wire_value(), 3);
    }

    #[test]
    fn test_protocol_version_from_wire() {
        assert_eq!(ProtocolVersion::from_wire(1), Some(ProtocolVersion::V1));
        assert_eq!(ProtocolVersion::from_wire(3), Some(ProtocolVersion::V3));
        assert_eq!(ProtocolVersion::from_wire(99), None);
    }

    #[test]
    fn test_protocol_version_negotiate() {
        let negotiated = ProtocolVersion::negotiate(ProtocolVersion::V3, ProtocolVersion::V2);
        assert_eq!(negotiated, ProtocolVersion::V2);
    }

    #[test]
    fn test_protocol_version_display() {
        assert_eq!(format!("{}", ProtocolVersion::V2), "v2");
    }

    #[test]
    fn test_message_type_roundtrip() {
        for (byte, expected) in [
            (0x01, MessageType::Announce),
            (0x02, MessageType::Request),
            (0x03, MessageType::Response),
            (0x04, MessageType::FollowUp),
            (0xFF, MessageType::Goodbye),
        ] {
            assert_eq!(MessageType::from_byte(byte), Some(expected));
        }
        assert_eq!(MessageType::from_byte(0x42), None);
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let msg = SyncMessage {
            version: ProtocolVersion::V3,
            msg_type: MessageType::Request,
            sequence: 42,
            origin_ns: 1_000_000_000,
            receive_ns: 0,
            correction_ns: 0,
        };
        let encoded = SyncProtocol::encode(&msg);
        let decoded = SyncProtocol::decode(&encoded).expect("should succeed in test");
        assert_eq!(decoded, msg);
    }

    #[test]
    fn test_decode_buffer_too_small() {
        let short = [0u8; 10];
        let err = SyncProtocol::decode(&short).unwrap_err();
        assert!(matches!(
            err,
            ProtocolError::BufferTooSmall {
                needed: 28,
                got: 10
            }
        ));
    }

    #[test]
    fn test_decode_unknown_version() {
        let mut buf = [0u8; SYNC_MESSAGE_SIZE];
        buf[0] = 99; // bad version
        buf[1] = 0x02; // Request
        let err = SyncProtocol::decode(&buf).unwrap_err();
        assert!(matches!(err, ProtocolError::UnknownVersion(99)));
    }

    #[test]
    fn test_decode_unknown_message_type() {
        let mut buf = [0u8; SYNC_MESSAGE_SIZE];
        buf[0] = 1; // V1
        buf[1] = 0x42; // unknown type
        let err = SyncProtocol::decode(&buf).unwrap_err();
        assert!(matches!(err, ProtocolError::UnknownMessageType(0x42)));
    }

    #[test]
    fn test_new_request_fields() {
        let msg = SyncMessage::new_request(ProtocolVersion::V2, 7, 12345678);
        assert_eq!(msg.msg_type, MessageType::Request);
        assert_eq!(msg.sequence, 7);
        assert_eq!(msg.origin_ns, 12345678);
        assert_eq!(msg.receive_ns, 0);
    }

    #[test]
    fn test_new_response_mirrors_request() {
        let req = SyncMessage::new_request(ProtocolVersion::V2, 3, 100_000);
        let resp = SyncMessage::new_response(&req, 200_000);
        assert_eq!(resp.msg_type, MessageType::Response);
        assert_eq!(resp.sequence, 3);
        assert_eq!(resp.origin_ns, 100_000);
        assert_eq!(resp.receive_ns, 200_000);
    }

    #[test]
    fn test_sequence_counter_wraps() {
        let mut proto = SyncProtocol::new();
        proto.sequence_counter = u16::MAX;
        let _ = proto.next_sequence();
        assert_eq!(proto.sequence_counter, 0);
    }

    #[test]
    fn test_next_sequence_increments() {
        let mut proto = SyncProtocol::new();
        assert_eq!(proto.next_sequence(), 0);
        assert_eq!(proto.next_sequence(), 1);
        assert_eq!(proto.next_sequence(), 2);
    }

    #[test]
    fn test_encode_decode_response() {
        let msg = SyncMessage {
            version: ProtocolVersion::V1,
            msg_type: MessageType::Response,
            sequence: 255,
            origin_ns: -999,
            receive_ns: 12345,
            correction_ns: 7,
        };
        let encoded = SyncProtocol::encode(&msg);
        assert_eq!(encoded.len(), SYNC_MESSAGE_SIZE);
        let decoded = SyncProtocol::decode(&encoded).expect("should succeed in test");
        assert_eq!(decoded.correction_ns, 7);
    }

    #[test]
    fn test_protocol_version_ordering() {
        assert!(ProtocolVersion::V1 < ProtocolVersion::V2);
        assert!(ProtocolVersion::V2 < ProtocolVersion::V3);
    }
}
