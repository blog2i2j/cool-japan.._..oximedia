//! Blackmagic ATEM network protocol implementation.
//!
//! This module provides a pure-Rust, no-unsafe implementation of the
//! Blackmagic ATEM network control protocol used by ATEM hardware switchers
//! (Mini, Mini Pro, Television Studio, 1 M/E, 2 M/E, etc.).
//!
//! # Protocol Overview
//!
//! The ATEM protocol runs over UDP port 9910.  Each UDP datagram contains an
//! `AtemPacket` header followed by zero or more `AtemCommand` payloads.
//!
//! Packet header layout (12 bytes):
//! ```text
//! Byte  0-1: flags (high 3 bits) + payload length (low 13 bits)
//! Byte  2-3: session ID
//! Byte  4-5: remote acknowledgement packet ID
//! Byte  6-7: local acknowledgement packet ID  (unused in simple impls)
//! Byte  8-9: resend request number
//! Byte 10-11: local packet ID
//! ```
//!
//! Each command block inside the payload is prefixed with a 8-byte header:
//! ```text
//! Byte 0-1: block length (including the 8-byte header)
//! Byte 2-3: padding / unknown
//! Byte 4-7: 4-character ASCII command name
//! ```
//!
//! # Usage
//!
//! ```rust
//! use oximedia_switcher::atem_protocol::{AtemCommand, AtemCommandEncoder, AtemPacket, AtemPacketFlags};
//!
//! // Build a Cut command and encode it into a packet.
//! let cmd = AtemCommand::Cut { me: 0 };
//! let mut encoder = AtemCommandEncoder::new(0xABCD);
//! let packet = encoder.encode_commands(&[cmd]).expect("encode");
//! assert!(!packet.payload().is_empty());
//! ```

use thiserror::Error;

/// Errors that can occur during ATEM protocol operations.
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum AtemError {
    /// The supplied byte slice is too short to contain a valid packet header.
    #[error("Packet too short: expected at least {expected} bytes, got {actual}")]
    PacketTooShort { expected: usize, actual: usize },

    /// A command block header claims a length that exceeds the remaining data.
    #[error("Command block length {block_len} exceeds remaining payload {remaining}")]
    CommandBlockOverflow { block_len: usize, remaining: usize },

    /// The decoded length field in the packet header does not match the actual
    /// datagram size.
    #[error("Packet length mismatch: header says {header_len}, actual {actual}")]
    LengthMismatch { header_len: usize, actual: usize },

    /// An unknown command tag was encountered during decoding.
    #[error("Unknown command tag: {:?}", tag)]
    UnknownTag { tag: [u8; 4] },

    /// A command's payload is too short for the declared command type.
    #[error("Command payload too short for '{tag}': need {need}, got {got}")]
    CommandPayloadTooShort {
        tag: &'static str,
        need: usize,
        got: usize,
    },

    /// The supplied M/E index exceeds the hardware maximum.
    #[error("M/E index {0} is out of range (max 3)")]
    MeOutOfRange(u8),

    /// The supplied input number exceeds the hardware maximum.
    #[error("Input number {0} is out of range")]
    InputOutOfRange(u16),

    /// The supplied aux bus index is out of range.
    #[error("Aux bus index {0} is out of range (max 23)")]
    AuxOutOfRange(u8),

    /// The payload would overflow the maximum UDP datagram for ATEM packets.
    #[error("Packet payload too large: {0} bytes exceeds maximum {1}")]
    PayloadTooLarge(usize, usize),
}

/// Maximum ATEM UDP payload size (16-bit length field, top 3 bits are flags).
pub const ATEM_MAX_PACKET_SIZE: usize = 0x1FFF;
/// Minimum ATEM header length in bytes.
pub const ATEM_HEADER_LEN: usize = 12;
/// ATEM command block header length in bytes.
pub const ATEM_CMD_HEADER_LEN: usize = 8;
/// ATEM default UDP port.
pub const ATEM_UDP_PORT: u16 = 9910;
/// Maximum M/E row index (0-based, ATEM supports up to 4 M/E rows).
pub const ATEM_MAX_ME: u8 = 3;
/// Maximum aux bus index (0-based, ATEM supports up to 24 aux outputs).
pub const ATEM_MAX_AUX: u8 = 23;
/// Maximum valid input number on a large ATEM switcher.
pub const ATEM_MAX_INPUT: u16 = 10_010;

// ── Packet flags ──────────────────────────────────────────────────────────────

/// Bit-flags carried in the high 3 bits of the ATEM packet word 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AtemPacketFlags(u8);

impl AtemPacketFlags {
    /// No flags set (normal command packet).
    pub const NONE: Self = Self(0x00);
    /// Reliable delivery flag (0x01).  Server expects an ACK.
    pub const RELIABLE: Self = Self(0x01);
    /// SYN flag (0x02).  Used during session handshake.
    pub const SYN: Self = Self(0x02);
    /// Retransmit flag (0x04).  Packet is a retransmission.
    pub const RETRANSMIT: Self = Self(0x04);
    /// Request retransmit flag (0x08).
    pub const REQUEST_RETRANSMIT: Self = Self(0x08);
    /// ACK flag (0x10).  Standalone acknowledgement packet.
    pub const ACK: Self = Self(0x10);

    /// Construct from a raw flag nibble (high 3 bits of the first 16-bit word).
    pub fn from_raw(raw: u8) -> Self {
        Self(raw & 0x1F)
    }

    /// Return the raw byte value.
    pub fn raw(self) -> u8 {
        self.0
    }

    /// Check whether `flag` is set.
    pub fn has(self, flag: Self) -> bool {
        self.0 & flag.0 != 0
    }

    /// Set an additional flag.
    pub fn with(self, flag: Self) -> Self {
        Self(self.0 | flag.0)
    }
}

impl Default for AtemPacketFlags {
    fn default() -> Self {
        Self::RELIABLE
    }
}

// ── Packet ───────────────────────────────────────────────────────────────────

/// A parsed or constructed ATEM network packet.
///
/// Encapsulates the 12-byte header fields and the raw payload bytes.  The
/// payload contains zero or more back-to-back `AtemCommand` blocks; use
/// [`AtemCommandDecoder::decode_payload`] to parse them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AtemPacket {
    /// Protocol flags.
    pub flags: AtemPacketFlags,
    /// Session ID negotiated during handshake.
    pub session_id: u16,
    /// Remote's last-seen packet ID (acknowledgement from the remote side).
    pub remote_ack_id: u16,
    /// Local acknowledgement ID.
    pub local_ack_id: u16,
    /// Resend request number.
    pub resend_request: u16,
    /// Local packet ID (incremented per sent reliable packet).
    pub local_packet_id: u16,
    /// Raw payload bytes following the 12-byte header.
    payload: Vec<u8>,
}

impl AtemPacket {
    /// Construct a new packet with the given header fields and no payload.
    pub fn new(flags: AtemPacketFlags, session_id: u16, local_packet_id: u16) -> Self {
        Self {
            flags,
            session_id,
            remote_ack_id: 0,
            local_ack_id: 0,
            resend_request: 0,
            local_packet_id,
            payload: Vec::new(),
        }
    }

    /// Construct a simple ACK packet for the given session and remote packet ID.
    pub fn ack(session_id: u16, remote_packet_id: u16) -> Self {
        Self {
            flags: AtemPacketFlags::ACK,
            session_id,
            remote_ack_id: remote_packet_id,
            local_ack_id: 0,
            resend_request: 0,
            local_packet_id: 0,
            payload: Vec::new(),
        }
    }

    /// Access the raw payload bytes.
    pub fn payload(&self) -> &[u8] {
        &self.payload
    }

    /// Total on-wire length (header + payload).
    pub fn wire_length(&self) -> usize {
        ATEM_HEADER_LEN + self.payload.len()
    }

    /// Parse an ATEM packet from a raw UDP datagram.
    ///
    /// The slice must be at least [`ATEM_HEADER_LEN`] bytes long.
    pub fn from_bytes(data: &[u8]) -> Result<Self, AtemError> {
        if data.len() < ATEM_HEADER_LEN {
            return Err(AtemError::PacketTooShort {
                expected: ATEM_HEADER_LEN,
                actual: data.len(),
            });
        }

        // Word 0: upper 5 bits = flags, lower 11 bits = total packet length (BE).
        let word0 = u16::from_be_bytes([data[0], data[1]]);
        let flags_raw = (word0 >> 11) as u8;
        let _pkt_len = (word0 & 0x07FF) as usize;

        let flags = AtemPacketFlags::from_raw(flags_raw);
        // Note: some ATEM firmware populates _pkt_len differently; we trust the
        // actual datagram length for payload extraction rather than this field.

        let session_id = u16::from_be_bytes([data[2], data[3]]);
        let remote_ack_id = u16::from_be_bytes([data[4], data[5]]);
        let local_ack_id = u16::from_be_bytes([data[6], data[7]]);
        let resend_request = u16::from_be_bytes([data[8], data[9]]);
        let local_packet_id = u16::from_be_bytes([data[10], data[11]]);

        let payload = data[ATEM_HEADER_LEN..].to_vec();

        Ok(Self {
            flags,
            session_id,
            remote_ack_id,
            local_ack_id,
            resend_request,
            local_packet_id,
            payload,
        })
    }

    /// Serialize this packet to a `Vec<u8>` suitable for transmission.
    pub fn to_bytes(&self) -> Vec<u8> {
        let total_len = self.wire_length();
        let mut out = Vec::with_capacity(total_len);

        // Word 0: flags (5 bits, MSB) | length (11 bits, LSB)
        let len_field = (total_len & 0x07FF) as u16;
        let flags_field = (self.flags.raw() as u16) << 11;
        let word0 = flags_field | len_field;
        out.extend_from_slice(&word0.to_be_bytes());

        out.extend_from_slice(&self.session_id.to_be_bytes());
        out.extend_from_slice(&self.remote_ack_id.to_be_bytes());
        out.extend_from_slice(&self.local_ack_id.to_be_bytes());
        out.extend_from_slice(&self.resend_request.to_be_bytes());
        out.extend_from_slice(&self.local_packet_id.to_be_bytes());
        out.extend_from_slice(&self.payload);
        out
    }
}

// ── Command definitions ───────────────────────────────────────────────────────

/// ATEM command tags (4-byte ASCII identifiers).
mod tags {
    pub const CUT: &[u8; 4] = b"DCut";
    pub const AUTO: &[u8; 4] = b"DAut";
    pub const PGM_INPUT: &[u8; 4] = b"CPgI";
    pub const PVW_INPUT: &[u8; 4] = b"CPvI";
    pub const AUX_SOURCE: &[u8; 4] = b"CAuS";
    pub const DSK_ON_AIR: &[u8; 4] = b"CDsL";
    pub const DSK_TIE: &[u8; 4] = b"CDsT";
    pub const DSK_AUTO: &[u8; 4] = b"DDsA";
    pub const FTB_AUTO: &[u8; 4] = b"DFtB";
    pub const TRANSITION_TYPE: &[u8; 4] = b"CTTp";
    pub const TRANSITION_MIX: &[u8; 4] = b"CTMx";
    pub const TRANSITION_DIP: &[u8; 4] = b"CTDp";
    pub const TRANSITION_WIPE: &[u8; 4] = b"CTWp";
    pub const KEYER_ON_AIR: &[u8; 4] = b"CKeL";
    pub const MACRO_RUN: &[u8; 4] = b"MAct";
    pub const MACRO_STOP: &[u8; 4] = b"MRcS";
}

/// ATEM transition style selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AtemTransitionStyle {
    /// Mix/dissolve transition.
    Mix = 0,
    /// Dip transition.
    Dip = 1,
    /// Wipe transition.
    Wipe = 2,
    /// DVE transition.
    Dve = 3,
    /// Stinger transition.
    Sting = 4,
}

impl AtemTransitionStyle {
    /// Decode from a raw byte. Returns `None` for unknown values.
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Mix),
            1 => Some(Self::Dip),
            2 => Some(Self::Wipe),
            3 => Some(Self::Dve),
            4 => Some(Self::Sting),
            _ => None,
        }
    }
}

/// High-level representation of a single ATEM network command.
///
/// Each variant corresponds to a specific 4-character ASCII tag in the
/// Blackmagic ATEM protocol wire format.  Payload fields use Rust-friendly
/// types; encode/decode converts to/from the raw little-endian byte layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AtemCommand {
    /// Perform an immediate cut on the specified M/E row.
    /// Tag: `DCut`  Payload: 4 bytes (M/E index + 3 pad bytes).
    Cut {
        /// M/E row index (0-based).
        me: u8,
    },

    /// Perform an auto transition on the specified M/E row.
    /// Tag: `DAut`  Payload: 4 bytes (M/E index + 3 pad bytes).
    Auto {
        /// M/E row index (0-based).
        me: u8,
    },

    /// Set the program bus input for an M/E row.
    /// Tag: `CPgI`  Payload: 4 bytes (M/E u8, pad, input u16 BE).
    SetProgramInput {
        /// M/E row index (0-based).
        me: u8,
        /// Input number (1-based; 0 = black).
        input: u16,
    },

    /// Set the preview bus input for an M/E row.
    /// Tag: `CPvI`  Payload: 4 bytes (M/E u8, pad, input u16 BE).
    SetPreviewInput {
        /// M/E row index (0-based).
        me: u8,
        /// Input number (1-based; 0 = black).
        input: u16,
    },

    /// Assign a source to an aux output bus.
    /// Tag: `CAuS`  Payload: 4 bytes (mask u8, aux u8, input u16 BE).
    SetAuxSource {
        /// Aux bus index (0-based).
        aux: u8,
        /// Source input number.
        input: u16,
    },

    /// Set a downstream keyer on-air state.
    /// Tag: `CDsL`  Payload: 4 bytes (DSK index u8, on-air bool u8, 2 pad).
    SetDskOnAir {
        /// DSK index (0 or 1).
        dsk: u8,
        /// `true` to put the DSK on-air.
        on_air: bool,
    },

    /// Set a downstream keyer tie state.
    /// Tag: `CDsT`  Payload: 4 bytes (DSK index u8, tie bool u8, 2 pad).
    SetDskTie {
        /// DSK index (0 or 1).
        dsk: u8,
        /// `true` to enable tie.
        tie: bool,
    },

    /// Trigger an auto for a downstream keyer.
    /// Tag: `DDsA`  Payload: 4 bytes (DSK index u8, 3 pad).
    DskAuto {
        /// DSK index (0 or 1).
        dsk: u8,
    },

    /// Trigger a Fade-to-Black auto.
    /// Tag: `DFtB`  Payload: 4 bytes (M/E u8, 3 pad).
    FadeToBlack {
        /// M/E row index.
        me: u8,
    },

    /// Set the transition style for an M/E row.
    /// Tag: `CTTp`  Payload: 4 bytes (mask u8, M/E u8, style u8, pad).
    SetTransitionStyle {
        /// M/E row index.
        me: u8,
        /// Selected transition style.
        style: AtemTransitionStyle,
    },

    /// Set mix transition rate for an M/E row (frames).
    /// Tag: `CTMx`  Payload: 4 bytes (M/E u8, rate u8, 2 pad).
    SetMixRate {
        /// M/E row index.
        me: u8,
        /// Transition rate in frames (1–250).
        rate: u8,
    },

    /// Set dip transition rate/source for an M/E row.
    /// Tag: `CTDp`  Payload: 4 bytes (M/E u8, rate u8, input u16 BE).
    SetDipParams {
        /// M/E row index.
        me: u8,
        /// Transition rate in frames.
        rate: u8,
        /// Dip source input number.
        input: u16,
    },

    /// Set wipe transition rate for an M/E row.
    /// Tag: `CTWp`  Payload: 4 bytes (M/E u8, rate u8, 2 pad).
    SetWipeRate {
        /// M/E row index.
        me: u8,
        /// Transition rate in frames (1–250).
        rate: u8,
    },

    /// Set upstream keyer on-air state.
    /// Tag: `CKeL`  Payload: 4 bytes (M/E u8, keyer u8, on-air bool u8, pad).
    SetKeyerOnAir {
        /// M/E row index.
        me: u8,
        /// Keyer index (0-based).
        keyer: u8,
        /// `true` to put the keyer on-air.
        on_air: bool,
    },

    /// Run a macro by slot index.
    /// Tag: `MAct`  Payload: 4 bytes (macro index u16 BE, action u8, pad).
    RunMacro {
        /// Macro slot index (0-based).
        index: u16,
    },

    /// Stop macro playback.
    /// Tag: `MRcS`  Payload: 4 bytes (all pad).
    StopMacro,

    /// An unrecognised command preserved as raw bytes for pass-through.
    Unknown {
        /// The 4-byte ASCII tag.
        tag: [u8; 4],
        /// Raw payload bytes.
        payload: Vec<u8>,
    },
}

impl AtemCommand {
    /// Return the 4-byte ASCII tag for this command.
    pub fn tag(&self) -> [u8; 4] {
        match self {
            Self::Cut { .. } => *tags::CUT,
            Self::Auto { .. } => *tags::AUTO,
            Self::SetProgramInput { .. } => *tags::PGM_INPUT,
            Self::SetPreviewInput { .. } => *tags::PVW_INPUT,
            Self::SetAuxSource { .. } => *tags::AUX_SOURCE,
            Self::SetDskOnAir { .. } => *tags::DSK_ON_AIR,
            Self::SetDskTie { .. } => *tags::DSK_TIE,
            Self::DskAuto { .. } => *tags::DSK_AUTO,
            Self::FadeToBlack { .. } => *tags::FTB_AUTO,
            Self::SetTransitionStyle { .. } => *tags::TRANSITION_TYPE,
            Self::SetMixRate { .. } => *tags::TRANSITION_MIX,
            Self::SetDipParams { .. } => *tags::TRANSITION_DIP,
            Self::SetWipeRate { .. } => *tags::TRANSITION_WIPE,
            Self::SetKeyerOnAir { .. } => *tags::KEYER_ON_AIR,
            Self::RunMacro { .. } => *tags::MACRO_RUN,
            Self::StopMacro => *tags::MACRO_STOP,
            Self::Unknown { tag, .. } => *tag,
        }
    }

    /// Encode the command payload (bytes after the 8-byte block header).
    ///
    /// Returns the payload bytes; the caller must prepend the block header.
    pub fn encode_payload(&self) -> Vec<u8> {
        match self {
            Self::Cut { me } => vec![*me, 0, 0, 0],
            Self::Auto { me } => vec![*me, 0, 0, 0],
            Self::SetProgramInput { me, input } => {
                let [hi, lo] = input.to_be_bytes();
                vec![*me, 0, hi, lo]
            }
            Self::SetPreviewInput { me, input } => {
                let [hi, lo] = input.to_be_bytes();
                vec![*me, 0, hi, lo]
            }
            Self::SetAuxSource { aux, input } => {
                let [hi, lo] = input.to_be_bytes();
                vec![0x01, *aux, hi, lo]
            }
            Self::SetDskOnAir { dsk, on_air } => {
                vec![*dsk, *on_air as u8, 0, 0]
            }
            Self::SetDskTie { dsk, tie } => {
                vec![*dsk, *tie as u8, 0, 0]
            }
            Self::DskAuto { dsk } => vec![*dsk, 0, 0, 0],
            Self::FadeToBlack { me } => vec![*me, 0, 0, 0],
            Self::SetTransitionStyle { me, style } => {
                vec![0x04, *me, *style as u8, 0]
            }
            Self::SetMixRate { me, rate } => vec![*me, *rate, 0, 0],
            Self::SetDipParams { me, rate, input } => {
                let [hi, lo] = input.to_be_bytes();
                vec![*me, *rate, hi, lo]
            }
            Self::SetWipeRate { me, rate } => vec![*me, *rate, 0, 0],
            Self::SetKeyerOnAir { me, keyer, on_air } => {
                vec![*me, *keyer, *on_air as u8, 0]
            }
            Self::RunMacro { index } => {
                let [hi, lo] = index.to_be_bytes();
                vec![hi, lo, 0, 0]
            }
            Self::StopMacro => vec![0, 0, 0, 0],
            Self::Unknown { payload, .. } => payload.clone(),
        }
    }

    /// Decode a single command from a tag + payload slice.
    ///
    /// `payload` must be exactly the bytes following the 8-byte block header.
    pub fn decode(tag: &[u8; 4], payload: &[u8]) -> Result<Self, AtemError> {
        macro_rules! need {
            ($n:expr, $name:expr) => {
                if payload.len() < $n {
                    return Err(AtemError::CommandPayloadTooShort {
                        tag: $name,
                        need: $n,
                        got: payload.len(),
                    });
                }
            };
        }

        match tag {
            t if t == tags::CUT => {
                need!(1, "DCut");
                Ok(Self::Cut { me: payload[0] })
            }
            t if t == tags::AUTO => {
                need!(1, "DAut");
                Ok(Self::Auto { me: payload[0] })
            }
            t if t == tags::PGM_INPUT => {
                need!(4, "CPgI");
                let input = u16::from_be_bytes([payload[2], payload[3]]);
                Ok(Self::SetProgramInput {
                    me: payload[0],
                    input,
                })
            }
            t if t == tags::PVW_INPUT => {
                need!(4, "CPvI");
                let input = u16::from_be_bytes([payload[2], payload[3]]);
                Ok(Self::SetPreviewInput {
                    me: payload[0],
                    input,
                })
            }
            t if t == tags::AUX_SOURCE => {
                need!(4, "CAuS");
                let input = u16::from_be_bytes([payload[2], payload[3]]);
                Ok(Self::SetAuxSource {
                    aux: payload[1],
                    input,
                })
            }
            t if t == tags::DSK_ON_AIR => {
                need!(2, "CDsL");
                Ok(Self::SetDskOnAir {
                    dsk: payload[0],
                    on_air: payload[1] != 0,
                })
            }
            t if t == tags::DSK_TIE => {
                need!(2, "CDsT");
                Ok(Self::SetDskTie {
                    dsk: payload[0],
                    tie: payload[1] != 0,
                })
            }
            t if t == tags::DSK_AUTO => {
                need!(1, "DDsA");
                Ok(Self::DskAuto { dsk: payload[0] })
            }
            t if t == tags::FTB_AUTO => {
                need!(1, "DFtB");
                Ok(Self::FadeToBlack { me: payload[0] })
            }
            t if t == tags::TRANSITION_TYPE => {
                need!(3, "CTTp");
                let style =
                    AtemTransitionStyle::from_u8(payload[2]).unwrap_or(AtemTransitionStyle::Mix);
                Ok(Self::SetTransitionStyle {
                    me: payload[1],
                    style,
                })
            }
            t if t == tags::TRANSITION_MIX => {
                need!(2, "CTMx");
                Ok(Self::SetMixRate {
                    me: payload[0],
                    rate: payload[1],
                })
            }
            t if t == tags::TRANSITION_DIP => {
                need!(4, "CTDp");
                let input = u16::from_be_bytes([payload[2], payload[3]]);
                Ok(Self::SetDipParams {
                    me: payload[0],
                    rate: payload[1],
                    input,
                })
            }
            t if t == tags::TRANSITION_WIPE => {
                need!(2, "CTWp");
                Ok(Self::SetWipeRate {
                    me: payload[0],
                    rate: payload[1],
                })
            }
            t if t == tags::KEYER_ON_AIR => {
                need!(3, "CKeL");
                Ok(Self::SetKeyerOnAir {
                    me: payload[0],
                    keyer: payload[1],
                    on_air: payload[2] != 0,
                })
            }
            t if t == tags::MACRO_RUN => {
                need!(2, "MAct");
                let index = u16::from_be_bytes([payload[0], payload[1]]);
                Ok(Self::RunMacro { index })
            }
            t if t == tags::MACRO_STOP => Ok(Self::StopMacro),
            _ => Ok(Self::Unknown {
                tag: *tag,
                payload: payload.to_vec(),
            }),
        }
    }
}

// ── Encoder ───────────────────────────────────────────────────────────────────

/// Encodes `AtemCommand` values into `AtemPacket` wire format.
///
/// Maintains a rolling `local_packet_id` counter that increments on every
/// reliable packet produced.
pub struct AtemCommandEncoder {
    session_id: u16,
    local_packet_id: u16,
}

impl AtemCommandEncoder {
    /// Create a new encoder bound to `session_id`.
    pub fn new(session_id: u16) -> Self {
        Self {
            session_id,
            local_packet_id: 0,
        }
    }

    /// Set the session ID (e.g. after completing the handshake).
    pub fn set_session_id(&mut self, id: u16) {
        self.session_id = id;
    }

    /// Current session ID.
    pub fn session_id(&self) -> u16 {
        self.session_id
    }

    /// Last local packet ID used.
    pub fn local_packet_id(&self) -> u16 {
        self.local_packet_id
    }

    /// Encode a slice of commands into a single `AtemPacket`.
    ///
    /// Returns an error if the total payload would exceed `ATEM_MAX_PACKET_SIZE`.
    pub fn encode_commands(&mut self, commands: &[AtemCommand]) -> Result<AtemPacket, AtemError> {
        let payload = Self::build_payload(commands)?;

        self.local_packet_id = self.local_packet_id.wrapping_add(1);

        let mut pkt = AtemPacket::new(
            AtemPacketFlags::RELIABLE,
            self.session_id,
            self.local_packet_id,
        );
        pkt.payload = payload;

        Ok(pkt)
    }

    /// Encode a single command into its own packet.
    pub fn encode_command(&mut self, command: AtemCommand) -> Result<AtemPacket, AtemError> {
        self.encode_commands(&[command])
    }

    /// Build raw payload bytes for a slice of commands.
    ///
    /// Each command is wrapped in an 8-byte block header:
    /// `[len_hi, len_lo, 0, 0, tag[0], tag[1], tag[2], tag[3]]`
    fn build_payload(commands: &[AtemCommand]) -> Result<Vec<u8>, AtemError> {
        let mut payload = Vec::new();

        for cmd in commands {
            let tag = cmd.tag();
            let cmd_payload = cmd.encode_payload();
            let block_len = ATEM_CMD_HEADER_LEN + cmd_payload.len();

            if ATEM_HEADER_LEN + payload.len() + block_len > ATEM_MAX_PACKET_SIZE {
                return Err(AtemError::PayloadTooLarge(
                    payload.len() + block_len,
                    ATEM_MAX_PACKET_SIZE - ATEM_HEADER_LEN,
                ));
            }

            // Block header: length (2 bytes BE) + 2 pad + 4 tag bytes.
            let len_be = (block_len as u16).to_be_bytes();
            payload.push(len_be[0]);
            payload.push(len_be[1]);
            payload.push(0);
            payload.push(0);
            payload.extend_from_slice(&tag);
            payload.extend_from_slice(&cmd_payload);
        }

        Ok(payload)
    }
}

// ── Decoder ───────────────────────────────────────────────────────────────────

/// Decodes `AtemCommand` values from raw `AtemPacket` payload bytes.
pub struct AtemCommandDecoder;

impl AtemCommandDecoder {
    /// Decode all commands from the payload of an `AtemPacket`.
    ///
    /// Blocks with unknown tags are represented as `AtemCommand::Unknown`.
    pub fn decode_payload(payload: &[u8]) -> Result<Vec<AtemCommand>, AtemError> {
        let mut commands = Vec::new();
        let mut pos = 0;

        while pos < payload.len() {
            let remaining = payload.len() - pos;

            if remaining < ATEM_CMD_HEADER_LEN {
                break; // Trailing padding — not an error.
            }

            let block_len = u16::from_be_bytes([payload[pos], payload[pos + 1]]) as usize;

            if block_len < ATEM_CMD_HEADER_LEN {
                break; // Zero-length block signals end of commands.
            }

            if block_len > remaining {
                return Err(AtemError::CommandBlockOverflow {
                    block_len,
                    remaining,
                });
            }

            // Skip 2-byte pad at offset +2/+3.
            let tag: [u8; 4] = [
                payload[pos + 4],
                payload[pos + 5],
                payload[pos + 6],
                payload[pos + 7],
            ];
            let cmd_payload = &payload[pos + ATEM_CMD_HEADER_LEN..pos + block_len];

            let cmd = AtemCommand::decode(&tag, cmd_payload)?;
            commands.push(cmd);

            pos += block_len;
        }

        Ok(commands)
    }

    /// Decode commands from a full UDP datagram (parses packet header first).
    pub fn decode_datagram(data: &[u8]) -> Result<(AtemPacket, Vec<AtemCommand>), AtemError> {
        let pkt = AtemPacket::from_bytes(data)?;
        let commands = Self::decode_payload(pkt.payload())?;
        Ok((pkt, commands))
    }
}

// ── Session handshake helpers ─────────────────────────────────────────────────

/// Minimal ATEM session state machine for the initial handshake.
///
/// The ATEM handshake consists of:
/// 1. Client sends SYN packet (`flags = 0x02`, `session_id = 0x0000`).
/// 2. Server responds with SYN+ACK (`flags = 0x02`, `session_id = <assigned>`).
/// 3. Client sends ACK (`flags = 0x10`, `session_id = <assigned>`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AtemSessionState {
    /// Not yet connected; initial state.
    Disconnected,
    /// SYN sent, waiting for SYN+ACK from server.
    SynSent,
    /// Handshake complete; session ID negotiated.
    Connected { session_id: u16 },
}

impl AtemSessionState {
    /// Build the initial SYN packet to send to the ATEM device.
    pub fn build_syn() -> AtemPacket {
        AtemPacket::new(AtemPacketFlags::SYN, 0x0000, 0x0000)
    }

    /// Process a received packet and advance state.
    ///
    /// Returns the ACK packet to send (if any) and the new state.
    pub fn process_packet(&self, pkt: &AtemPacket) -> (Option<AtemPacket>, AtemSessionState) {
        match self {
            Self::SynSent => {
                if pkt.flags.has(AtemPacketFlags::SYN) {
                    let session_id = pkt.session_id;
                    let ack = AtemPacket::ack(session_id, pkt.local_packet_id);
                    (Some(ack), Self::Connected { session_id })
                } else {
                    (None, self.clone())
                }
            }
            Self::Connected { .. } => {
                // Acknowledge any reliable packet.
                if pkt.flags.has(AtemPacketFlags::RELIABLE) {
                    let ack = AtemPacket::ack(pkt.session_id, pkt.local_packet_id);
                    (Some(ack), self.clone())
                } else {
                    (None, self.clone())
                }
            }
            Self::Disconnected => (None, self.clone()),
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── AtemPacketFlags tests ─────────────────────────────────────────────────

    #[test]
    fn test_flags_has_and_with() {
        let flags = AtemPacketFlags::RELIABLE;
        assert!(flags.has(AtemPacketFlags::RELIABLE));
        assert!(!flags.has(AtemPacketFlags::SYN));

        let flags2 = flags.with(AtemPacketFlags::SYN);
        assert!(flags2.has(AtemPacketFlags::RELIABLE));
        assert!(flags2.has(AtemPacketFlags::SYN));
    }

    #[test]
    fn test_flags_default_is_reliable() {
        let flags = AtemPacketFlags::default();
        assert!(flags.has(AtemPacketFlags::RELIABLE));
    }

    // ── AtemPacket round-trip ─────────────────────────────────────────────────

    #[test]
    fn test_packet_round_trip_empty() {
        let pkt = AtemPacket::new(AtemPacketFlags::RELIABLE, 0x1234, 7);
        let bytes = pkt.to_bytes();
        let decoded = AtemPacket::from_bytes(&bytes).expect("should parse");
        assert_eq!(decoded.session_id, pkt.session_id);
        assert_eq!(decoded.local_packet_id, pkt.local_packet_id);
        assert!(decoded.payload().is_empty());
    }

    #[test]
    fn test_packet_too_short_error() {
        let short = [0u8; 8]; // less than ATEM_HEADER_LEN (12)
        let err = AtemPacket::from_bytes(&short).unwrap_err();
        assert!(matches!(err, AtemError::PacketTooShort { .. }));
    }

    #[test]
    fn test_ack_packet_has_ack_flag() {
        let ack = AtemPacket::ack(0xBEEF, 42);
        assert!(ack.flags.has(AtemPacketFlags::ACK));
        assert_eq!(ack.session_id, 0xBEEF);
        assert_eq!(ack.remote_ack_id, 42);
    }

    // ── AtemCommand encode / decode ───────────────────────────────────────────

    #[test]
    fn test_cut_command_encode_decode() {
        let cmd = AtemCommand::Cut { me: 1 };
        let payload = cmd.encode_payload();
        assert_eq!(payload[0], 1); // M/E = 1

        let decoded = AtemCommand::decode(&cmd.tag(), &payload).expect("decode");
        assert_eq!(decoded, cmd);
    }

    #[test]
    fn test_auto_command_encode_decode() {
        let cmd = AtemCommand::Auto { me: 0 };
        let payload = cmd.encode_payload();
        let decoded = AtemCommand::decode(&cmd.tag(), &payload).expect("decode");
        assert_eq!(decoded, cmd);
    }

    #[test]
    fn test_set_program_input_encode_decode() {
        let cmd = AtemCommand::SetProgramInput { me: 0, input: 3 };
        let payload = cmd.encode_payload();
        let decoded = AtemCommand::decode(&cmd.tag(), &payload).expect("decode");
        assert_eq!(decoded, cmd);
    }

    #[test]
    fn test_set_preview_input_encode_decode() {
        let cmd = AtemCommand::SetPreviewInput { me: 1, input: 5 };
        let payload = cmd.encode_payload();
        let decoded = AtemCommand::decode(&cmd.tag(), &payload).expect("decode");
        assert_eq!(decoded, cmd);
    }

    #[test]
    fn test_set_aux_source_encode_decode() {
        let cmd = AtemCommand::SetAuxSource { aux: 2, input: 7 };
        let payload = cmd.encode_payload();
        let decoded = AtemCommand::decode(&cmd.tag(), &payload).expect("decode");
        assert_eq!(decoded, cmd);
    }

    #[test]
    fn test_dsk_on_air_encode_decode() {
        let cmd = AtemCommand::SetDskOnAir {
            dsk: 0,
            on_air: true,
        };
        let payload = cmd.encode_payload();
        let decoded = AtemCommand::decode(&cmd.tag(), &payload).expect("decode");
        assert_eq!(decoded, cmd);

        let cmd_off = AtemCommand::SetDskOnAir {
            dsk: 1,
            on_air: false,
        };
        let payload_off = cmd_off.encode_payload();
        let decoded_off = AtemCommand::decode(&cmd_off.tag(), &payload_off).expect("decode");
        assert_eq!(decoded_off, cmd_off);
    }

    #[test]
    fn test_dsk_tie_encode_decode() {
        let cmd = AtemCommand::SetDskTie { dsk: 0, tie: true };
        let payload = cmd.encode_payload();
        let decoded = AtemCommand::decode(&cmd.tag(), &payload).expect("decode");
        assert_eq!(decoded, cmd);
    }

    #[test]
    fn test_dsk_auto_encode_decode() {
        let cmd = AtemCommand::DskAuto { dsk: 1 };
        let payload = cmd.encode_payload();
        let decoded = AtemCommand::decode(&cmd.tag(), &payload).expect("decode");
        assert_eq!(decoded, cmd);
    }

    #[test]
    fn test_fade_to_black_encode_decode() {
        let cmd = AtemCommand::FadeToBlack { me: 0 };
        let payload = cmd.encode_payload();
        let decoded = AtemCommand::decode(&cmd.tag(), &payload).expect("decode");
        assert_eq!(decoded, cmd);
    }

    #[test]
    fn test_transition_style_encode_decode() {
        let cmd = AtemCommand::SetTransitionStyle {
            me: 0,
            style: AtemTransitionStyle::Mix,
        };
        let payload = cmd.encode_payload();
        let decoded = AtemCommand::decode(&cmd.tag(), &payload).expect("decode");
        assert_eq!(decoded, cmd);

        let cmd_wipe = AtemCommand::SetTransitionStyle {
            me: 1,
            style: AtemTransitionStyle::Wipe,
        };
        let payload_wipe = cmd_wipe.encode_payload();
        let decoded_wipe = AtemCommand::decode(&cmd_wipe.tag(), &payload_wipe).expect("decode");
        assert_eq!(decoded_wipe, cmd_wipe);
    }

    #[test]
    fn test_mix_rate_encode_decode() {
        let cmd = AtemCommand::SetMixRate { me: 0, rate: 30 };
        let payload = cmd.encode_payload();
        let decoded = AtemCommand::decode(&cmd.tag(), &payload).expect("decode");
        assert_eq!(decoded, cmd);
    }

    #[test]
    fn test_dip_params_encode_decode() {
        let cmd = AtemCommand::SetDipParams {
            me: 0,
            rate: 20,
            input: 1000,
        };
        let payload = cmd.encode_payload();
        let decoded = AtemCommand::decode(&cmd.tag(), &payload).expect("decode");
        assert_eq!(decoded, cmd);
    }

    #[test]
    fn test_wipe_rate_encode_decode() {
        let cmd = AtemCommand::SetWipeRate { me: 0, rate: 25 };
        let payload = cmd.encode_payload();
        let decoded = AtemCommand::decode(&cmd.tag(), &payload).expect("decode");
        assert_eq!(decoded, cmd);
    }

    #[test]
    fn test_keyer_on_air_encode_decode() {
        let cmd = AtemCommand::SetKeyerOnAir {
            me: 0,
            keyer: 2,
            on_air: true,
        };
        let payload = cmd.encode_payload();
        let decoded = AtemCommand::decode(&cmd.tag(), &payload).expect("decode");
        assert_eq!(decoded, cmd);
    }

    #[test]
    fn test_run_macro_encode_decode() {
        let cmd = AtemCommand::RunMacro { index: 255 };
        let payload = cmd.encode_payload();
        let decoded = AtemCommand::decode(&cmd.tag(), &payload).expect("decode");
        assert_eq!(decoded, cmd);
    }

    #[test]
    fn test_stop_macro_encode_decode() {
        let cmd = AtemCommand::StopMacro;
        let payload = cmd.encode_payload();
        let decoded = AtemCommand::decode(&cmd.tag(), &payload).expect("decode");
        assert_eq!(decoded, cmd);
    }

    #[test]
    fn test_unknown_command_preserved() {
        let tag = [b'X', b'X', b'X', b'X'];
        let raw_payload = vec![1, 2, 3, 4, 5];
        let decoded = AtemCommand::decode(&tag, &raw_payload).expect("decode");
        assert!(matches!(decoded, AtemCommand::Unknown { .. }));
    }

    // ── AtemCommandEncoder tests ──────────────────────────────────────────────

    #[test]
    fn test_encoder_increments_packet_id() {
        let mut encoder = AtemCommandEncoder::new(0xABCD);
        let pkt1 = encoder
            .encode_command(AtemCommand::Cut { me: 0 })
            .expect("ok");
        let pkt2 = encoder
            .encode_command(AtemCommand::Cut { me: 0 })
            .expect("ok");
        assert_eq!(pkt1.local_packet_id, 1);
        assert_eq!(pkt2.local_packet_id, 2);
    }

    #[test]
    fn test_encoder_packet_contains_command() {
        let mut encoder = AtemCommandEncoder::new(0x1234);
        let cmd = AtemCommand::Cut { me: 0 };
        let pkt = encoder.encode_command(cmd).expect("ok");
        assert!(!pkt.payload().is_empty());

        // The block length in the payload header should be 12 (8 header + 4 payload).
        let block_len = u16::from_be_bytes([pkt.payload()[0], pkt.payload()[1]]) as usize;
        assert_eq!(block_len, ATEM_CMD_HEADER_LEN + 4);
    }

    #[test]
    fn test_encoder_multiple_commands_in_one_packet() {
        let mut encoder = AtemCommandEncoder::new(0x0001);
        let cmds = vec![
            AtemCommand::SetProgramInput { me: 0, input: 1 },
            AtemCommand::SetPreviewInput { me: 0, input: 2 },
            AtemCommand::Auto { me: 0 },
        ];
        let pkt = encoder.encode_commands(&cmds).expect("ok");
        // Decode and verify.
        let decoded = AtemCommandDecoder::decode_payload(pkt.payload()).expect("decode");
        assert_eq!(decoded.len(), 3);
        assert_eq!(decoded[0], AtemCommand::SetProgramInput { me: 0, input: 1 });
        assert_eq!(decoded[1], AtemCommand::SetPreviewInput { me: 0, input: 2 });
        assert_eq!(decoded[2], AtemCommand::Auto { me: 0 });
    }

    // ── AtemCommandDecoder tests ──────────────────────────────────────────────

    #[test]
    fn test_decoder_empty_payload() {
        let cmds = AtemCommandDecoder::decode_payload(&[]).expect("ok");
        assert!(cmds.is_empty());
    }

    #[test]
    fn test_decoder_full_datagram_round_trip() {
        let mut encoder = AtemCommandEncoder::new(0x5678);
        let original = vec![
            AtemCommand::Cut { me: 0 },
            AtemCommand::SetDskOnAir {
                dsk: 0,
                on_air: true,
            },
        ];
        let pkt = encoder.encode_commands(&original).expect("ok");
        let bytes = pkt.to_bytes();
        let (decoded_pkt, decoded_cmds) = AtemCommandDecoder::decode_datagram(&bytes).expect("ok");
        assert_eq!(decoded_pkt.session_id, 0x5678);
        assert_eq!(decoded_cmds.len(), 2);
        assert_eq!(decoded_cmds[0], AtemCommand::Cut { me: 0 });
        assert_eq!(
            decoded_cmds[1],
            AtemCommand::SetDskOnAir {
                dsk: 0,
                on_air: true
            }
        );
    }

    // ── AtemSessionState tests ────────────────────────────────────────────────

    #[test]
    fn test_session_handshake_flow() {
        // Step 1: build SYN.
        let syn = AtemSessionState::build_syn();
        assert!(syn.flags.has(AtemPacketFlags::SYN));

        // Step 2: simulate SYN+ACK from server.
        let mut server_syn_ack = AtemPacket::new(AtemPacketFlags::SYN, 0xCAFE, 1);
        server_syn_ack.remote_ack_id = 0;

        let state = AtemSessionState::SynSent;
        let (ack_opt, new_state) = state.process_packet(&server_syn_ack);

        assert!(ack_opt.is_some(), "should produce ACK");
        let ack = ack_opt.expect("ack packet");
        assert!(ack.flags.has(AtemPacketFlags::ACK));

        assert_eq!(
            new_state,
            AtemSessionState::Connected { session_id: 0xCAFE }
        );
    }

    #[test]
    fn test_session_connected_acks_reliable_packets() {
        let state = AtemSessionState::Connected { session_id: 0x1111 };
        let reliable_pkt = AtemPacket::new(AtemPacketFlags::RELIABLE, 0x1111, 5);
        let (ack_opt, new_state) = state.process_packet(&reliable_pkt);
        assert!(ack_opt.is_some());
        assert_eq!(
            new_state,
            AtemSessionState::Connected { session_id: 0x1111 }
        );
    }

    #[test]
    fn test_transition_style_from_u8() {
        assert_eq!(
            AtemTransitionStyle::from_u8(0),
            Some(AtemTransitionStyle::Mix)
        );
        assert_eq!(
            AtemTransitionStyle::from_u8(2),
            Some(AtemTransitionStyle::Wipe)
        );
        assert_eq!(
            AtemTransitionStyle::from_u8(4),
            Some(AtemTransitionStyle::Sting)
        );
        assert_eq!(AtemTransitionStyle::from_u8(99), None);
    }

    #[test]
    fn test_command_payload_too_short_error() {
        // Simulate a malformed DCut with zero bytes.
        let err = AtemCommand::decode(tags::CUT, &[]).unwrap_err();
        assert!(matches!(err, AtemError::CommandPayloadTooShort { .. }));
    }
}
