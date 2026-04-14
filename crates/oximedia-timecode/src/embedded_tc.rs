//! Embedded timecode (ATC - Ancillary Timecode) in SDI ancillary data packets.
//!
//! This module implements reading and writing of ATC (Ancillary Timecode)
//! as defined in SMPTE ST 12-3 and SMPTE 309M for embedding timecode in
//! HD-SDI ancillary data (HANC/VANC).
//!
//! # ATC Packet Format
//!
//! ATC packets are embedded in the Horizontal Ancillary (HANC) or Vertical
//! Ancillary (VANC) data space of SDI streams. The packet consists of:
//! - DID (Data Identifier): 0x60 for ATC
//! - SDID (Secondary Data ID): 0x60 for LTC-ATC or 0x61 for VITC-ATC
//! - DC (Data Count): number of data words
//! - User data: encoded timecode
//! - Checksum

#![allow(dead_code)]
#![allow(clippy::cast_possible_truncation)]

use crate::{FrameRate, Timecode, TimecodeError};

/// DID for ATC (Ancillary Timecode) per SMPTE ST 12-3.
pub const ATC_DID: u16 = 0x60;

/// SDID for LTC-based ATC.
pub const ATC_SDID_LTC: u16 = 0x60;

/// SDID for VITC-based ATC.
pub const ATC_SDID_VITC: u16 = 0x61;

/// ATC timecode type embedded in SDI ancillary data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AtcType {
    /// LTC-based ancillary timecode.
    Ltc,
    /// VITC-based ancillary timecode.
    Vitc,
}

/// Ancillary Timecode (ATC) packet for SDI embedding.
///
/// This struct holds a timecode value packaged for insertion into
/// the ancillary data space of an SDI signal.
#[derive(Debug, Clone)]
pub struct AtcPacket {
    /// Whether this is LTC-ATC or VITC-ATC.
    pub atc_type: AtcType,
    /// The timecode value.
    pub timecode: Timecode,
    /// User bits (32 bits of user-defined data).
    pub user_bits: u32,
    /// Whether continuity counter is valid.
    pub continuity_count_valid: bool,
    /// Continuity counter (0-127).
    pub continuity_count: u8,
    /// Whether this packet contains valid data.
    pub valid: bool,
}

impl AtcPacket {
    /// Create a new ATC packet with the given timecode.
    #[must_use]
    pub fn new(atc_type: AtcType, timecode: Timecode) -> Self {
        Self {
            atc_type,
            timecode,
            user_bits: 0,
            continuity_count_valid: false,
            continuity_count: 0,
            valid: true,
        }
    }

    /// Create an LTC-ATC packet.
    #[must_use]
    pub fn ltc(timecode: Timecode) -> Self {
        Self::new(AtcType::Ltc, timecode)
    }

    /// Create a VITC-ATC packet.
    #[must_use]
    pub fn vitc(timecode: Timecode) -> Self {
        Self::new(AtcType::Vitc, timecode)
    }

    /// Set user bits.
    #[must_use]
    pub fn with_user_bits(mut self, user_bits: u32) -> Self {
        self.user_bits = user_bits;
        self
    }

    /// Set continuity counter.
    #[must_use]
    pub fn with_continuity(mut self, count: u8) -> Self {
        self.continuity_count = count & 0x7F;
        self.continuity_count_valid = true;
        self
    }

    /// Serialize the ATC packet to 10-bit SDI words (as u16).
    ///
    /// Returns the ancillary data packet words including ADF, DID, SDID, DC,
    /// user data, and checksum per SMPTE ST 291.
    #[must_use]
    pub fn to_sdi_words(&self) -> Vec<u16> {
        let mut words = Vec::with_capacity(16);

        // Ancillary Data Flag (ADF): 0x000, 0x3FF, 0x3FF
        words.push(0x000);
        words.push(0x3FF);
        words.push(0x3FF);

        // DID (with 9-bit parity)
        let did = add_parity_9bit(ATC_DID as u8);
        words.push(did);

        // SDID
        let sdid = match self.atc_type {
            AtcType::Ltc => add_parity_9bit(ATC_SDID_LTC as u8),
            AtcType::Vitc => add_parity_9bit(ATC_SDID_VITC as u8),
        };
        words.push(sdid);

        // DC (Data Count): 9 data words
        let dc = add_parity_9bit(9);
        words.push(dc);

        // Encode timecode into 8 bytes (SMPTE 12M format)
        let tc_bytes = encode_timecode_bytes(&self.timecode, self.user_bits);
        for &byte in &tc_bytes {
            words.push(add_parity_9bit(byte));
        }

        // Continuity count word
        let cont = if self.continuity_count_valid {
            0x80 | (self.continuity_count & 0x7F)
        } else {
            0x00
        };
        words.push(add_parity_9bit(cont));

        // Checksum
        let checksum = compute_checksum(&words[3..]);
        words.push(checksum);

        words
    }

    /// Parse an ATC packet from 10-bit SDI words.
    ///
    /// # Errors
    ///
    /// Returns error if the packet is malformed, checksum fails,
    /// or the DID/SDID is not recognized as ATC.
    pub fn from_sdi_words(words: &[u16]) -> Result<Self, TimecodeError> {
        // Minimum size: ADF(3) + DID + SDID + DC + 9 data + checksum = 16
        if words.len() < 16 {
            return Err(TimecodeError::InvalidConfiguration);
        }

        // Validate ADF
        if words[0] != 0x000 || words[1] != 0x3FF || words[2] != 0x3FF {
            return Err(TimecodeError::InvalidConfiguration);
        }

        let did = (words[3] & 0xFF) as u8;
        if did != ATC_DID as u8 {
            return Err(TimecodeError::InvalidConfiguration);
        }

        let sdid = (words[4] & 0xFF) as u8;
        let atc_type = match sdid {
            s if s == ATC_SDID_LTC as u8 => AtcType::Ltc,
            s if s == ATC_SDID_VITC as u8 => AtcType::Vitc,
            _ => return Err(TimecodeError::InvalidConfiguration),
        };

        let dc = (words[5] & 0xFF) as usize;
        if dc < 9 || words.len() < 6 + dc + 1 {
            return Err(TimecodeError::InvalidConfiguration);
        }

        // Extract 8 timecode bytes + 1 continuity byte
        let mut tc_bytes = [0u8; 8];
        for (i, b) in tc_bytes.iter_mut().enumerate() {
            *b = (words[6 + i] & 0xFF) as u8;
        }
        let cont_byte = (words[14] & 0xFF) as u8;

        let (timecode, user_bits) = decode_timecode_bytes(&tc_bytes)?;

        let continuity_count_valid = (cont_byte & 0x80) != 0;
        let continuity_count = cont_byte & 0x7F;

        Ok(Self {
            atc_type,
            timecode,
            user_bits,
            continuity_count_valid,
            continuity_count,
            valid: true,
        })
    }
}

/// Encode a timecode into 8 SMPTE 12M bytes plus user bits.
fn encode_timecode_bytes(tc: &Timecode, user_bits: u32) -> [u8; 8] {
    let drop_flag: u8 = if tc.frame_rate.drop_frame { 0x40 } else { 0x00 };
    let frame_units = tc.frames % 10;
    let frame_tens = tc.frames / 10;

    let sec_units = tc.seconds % 10;
    let sec_tens = tc.seconds / 10;

    let min_units = tc.minutes % 10;
    let min_tens = tc.minutes / 10;

    let hour_units = tc.hours % 10;
    let hour_tens = tc.hours / 10;

    // User bits nibbles
    let ub = [
        ((user_bits >> 28) & 0xF) as u8,
        ((user_bits >> 24) & 0xF) as u8,
        ((user_bits >> 20) & 0xF) as u8,
        ((user_bits >> 16) & 0xF) as u8,
        ((user_bits >> 12) & 0xF) as u8,
        ((user_bits >> 8) & 0xF) as u8,
        ((user_bits >> 4) & 0xF) as u8,
        (user_bits & 0xF) as u8,
    ];

    [
        (frame_units & 0x0F) | (ub[0] << 4),
        (frame_tens & 0x03) | drop_flag | (ub[1] << 4),
        (sec_units & 0x0F) | (ub[2] << 4),
        (sec_tens & 0x07) | (ub[3] << 4),
        (min_units & 0x0F) | (ub[4] << 4),
        (min_tens & 0x07) | (ub[5] << 4),
        (hour_units & 0x0F) | (ub[6] << 4),
        (hour_tens & 0x03) | (ub[7] << 4),
    ]
}

/// Decode SMPTE 12M timecode bytes back to a Timecode and user bits.
fn decode_timecode_bytes(bytes: &[u8; 8]) -> Result<(Timecode, u32), TimecodeError> {
    let frame_units = bytes[0] & 0x0F;
    let frame_tens = bytes[1] & 0x03;
    let drop_frame = (bytes[1] & 0x40) != 0;
    let sec_units = bytes[2] & 0x0F;
    let sec_tens = bytes[3] & 0x07;
    let min_units = bytes[4] & 0x0F;
    let min_tens = bytes[5] & 0x07;
    let hour_units = bytes[6] & 0x0F;
    let hour_tens = bytes[7] & 0x03;

    let hours = hour_tens * 10 + hour_units;
    let minutes = min_tens * 10 + min_units;
    let seconds = sec_tens * 10 + sec_units;
    let frames = frame_tens * 10 + frame_units;

    let frame_rate = if drop_frame {
        FrameRate::Fps2997DF
    } else {
        FrameRate::Fps25
    };

    // User bits from upper nibbles
    let ub: [u8; 8] = [
        (bytes[0] >> 4) & 0x0F,
        (bytes[1] >> 4) & 0x0F,
        (bytes[2] >> 4) & 0x0F,
        (bytes[3] >> 4) & 0x0F,
        (bytes[4] >> 4) & 0x0F,
        (bytes[5] >> 4) & 0x0F,
        (bytes[6] >> 4) & 0x0F,
        (bytes[7] >> 4) & 0x0F,
    ];

    let decoded_user_bits = ((ub[0] as u32) << 28)
        | ((ub[1] as u32) << 24)
        | ((ub[2] as u32) << 20)
        | ((ub[3] as u32) << 16)
        | ((ub[4] as u32) << 12)
        | ((ub[5] as u32) << 8)
        | ((ub[6] as u32) << 4)
        | (ub[7] as u32);

    let timecode = Timecode::new(hours, minutes, seconds, frames, frame_rate)?;

    Ok((timecode, decoded_user_bits))
}

/// Add 9-bit odd parity to an 8-bit SDI word value.
///
/// SDI 10-bit words use bit 9 (NOT) and bit 8 (parity) for error detection.
#[must_use]
fn add_parity_9bit(byte: u8) -> u16 {
    let value = byte as u16;
    let ones = value.count_ones();
    let parity_bit: u16 = if ones % 2 == 0 { 0x100 } else { 0x000 };
    let not_bit: u16 = if parity_bit != 0 { 0x000 } else { 0x200 };
    value | parity_bit | not_bit
}

/// Compute SMPTE 291 checksum for the given SDI words.
#[must_use]
fn compute_checksum(words: &[u16]) -> u16 {
    let sum: u32 = words.iter().map(|&w| (w & 0x1FF) as u32).sum();
    let checksum = (sum & 0x1FF) as u16;
    // Bit 8 = NOT bit 8 of sum, bit 9 = NOT of bit 8
    let b8 = (checksum >> 8) & 1;
    checksum | ((1 - b8) << 9)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FrameRate;

    fn make_tc() -> Timecode {
        Timecode::new(1, 30, 0, 12, FrameRate::Fps25).expect("valid timecode")
    }

    #[test]
    fn test_atc_packet_creation() {
        let tc = make_tc();
        let pkt = AtcPacket::ltc(tc);
        assert_eq!(pkt.atc_type, AtcType::Ltc);
        assert!(pkt.valid);
    }

    #[test]
    fn test_atc_to_sdi_words_length() {
        let tc = make_tc();
        let pkt = AtcPacket::ltc(tc);
        let words = pkt.to_sdi_words();
        // ADF(3) + DID + SDID + DC + 9 data + 1 cont + checksum = 16
        assert_eq!(words.len(), 16);
    }

    #[test]
    fn test_atc_adf_header() {
        let tc = make_tc();
        let pkt = AtcPacket::vitc(tc);
        let words = pkt.to_sdi_words();
        assert_eq!(words[0], 0x000);
        assert_eq!(words[1], 0x3FF);
        assert_eq!(words[2], 0x3FF);
    }

    #[test]
    fn test_parity_9bit_even_input() {
        // 0x00 has 0 ones — even — parity bit set, NOT bit clear
        let w = add_parity_9bit(0x00);
        assert_eq!(w & 0x100, 0x100); // parity bit set
    }

    #[test]
    fn test_encode_decode_timecode_bytes() {
        let tc = make_tc();
        let bytes = encode_timecode_bytes(&tc, 0xDEAD_BEEF);
        let (decoded_tc, decoded_ub) = decode_timecode_bytes(&bytes).expect("decode ok");
        assert_eq!(decoded_tc.hours, tc.hours);
        assert_eq!(decoded_tc.minutes, tc.minutes);
        assert_eq!(decoded_tc.seconds, tc.seconds);
        assert_eq!(decoded_tc.frames, tc.frames);
        assert_eq!(decoded_ub, 0xDEAD_BEEF);
    }

    #[test]
    fn test_atc_round_trip_ltc() {
        let tc = make_tc();
        let pkt = AtcPacket::ltc(tc).with_user_bits(0x1234_5678);
        let words = pkt.to_sdi_words();
        let decoded = AtcPacket::from_sdi_words(&words).expect("decode ok");
        assert_eq!(decoded.atc_type, AtcType::Ltc);
        assert_eq!(decoded.timecode.hours, 1);
        assert_eq!(decoded.timecode.minutes, 30);
        assert_eq!(decoded.timecode.seconds, 0);
        assert_eq!(decoded.timecode.frames, 12);
        assert_eq!(decoded.user_bits, 0x1234_5678);
    }

    #[test]
    fn test_atc_round_trip_vitc() {
        let tc = Timecode::new(23, 59, 59, 24, FrameRate::Fps25).expect("valid tc");
        let pkt = AtcPacket::vitc(tc);
        let words = pkt.to_sdi_words();
        let decoded = AtcPacket::from_sdi_words(&words).expect("decode ok");
        assert_eq!(decoded.atc_type, AtcType::Vitc);
        assert_eq!(decoded.timecode.hours, 23);
        assert_eq!(decoded.timecode.seconds, 59);
    }

    #[test]
    fn test_atc_continuity_counter() {
        let tc = make_tc();
        let pkt = AtcPacket::ltc(tc).with_continuity(42);
        assert!(pkt.continuity_count_valid);
        assert_eq!(pkt.continuity_count, 42);
        let words = pkt.to_sdi_words();
        let decoded = AtcPacket::from_sdi_words(&words).expect("decode ok");
        assert!(decoded.continuity_count_valid);
        assert_eq!(decoded.continuity_count, 42);
    }

    #[test]
    fn test_atc_from_sdi_words_too_short() {
        let words = vec![0u16; 5];
        assert!(AtcPacket::from_sdi_words(&words).is_err());
    }

    #[test]
    fn test_atc_from_sdi_words_bad_adf() {
        let mut words = vec![0u16; 16];
        words[0] = 0x123; // invalid ADF
        words[1] = 0x3FF;
        words[2] = 0x3FF;
        assert!(AtcPacket::from_sdi_words(&words).is_err());
    }

    #[test]
    fn test_atc_zero_user_bits() {
        let tc = make_tc();
        let pkt = AtcPacket::ltc(tc).with_user_bits(0);
        let words = pkt.to_sdi_words();
        let decoded = AtcPacket::from_sdi_words(&words).expect("decode ok");
        assert_eq!(decoded.user_bits, 0);
    }

    #[test]
    fn test_atc_max_user_bits() {
        let tc = make_tc();
        let bytes = encode_timecode_bytes(&tc, 0x0FFF_FFFF);
        let (_, decoded_ub) = decode_timecode_bytes(&bytes).expect("decode ok");
        // Only lower 4 bits of each nibble are preserved (8 nibbles = 32 bits total,
        // but upper nibble bits interleave with TC data). With 0x0FFF_FFFF the top
        // nibble is 0x0 so everything fits.
        assert_eq!(decoded_ub, 0x0FFF_FFFF);
    }
}
