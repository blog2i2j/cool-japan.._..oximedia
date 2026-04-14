//! PSSH (Protection System Specific Header) box parser for DASH/HLS DRM systems.
//!
//! Parses ISO BMFF PSSH boxes as defined in ISO/IEC 23001-7 Common Encryption.
//!
//! Box layout (big-endian):
//! - 4 bytes: total size
//! - 4 bytes: box type `'pssh'`
//! - 1 byte:  version
//! - 3 bytes: flags
//! - 16 bytes: system_id
//! - [version >= 1] 4 bytes: key_id_count, then N×16 bytes: key_ids
//! - 4 bytes: data_size
//! - N bytes: data

use thiserror::Error;

// ---------------------------------------------------------------------------
// Well-known DRM system IDs
// ---------------------------------------------------------------------------

/// Widevine DRM system ID: `edef8ba9-79d6-4ace-a3c8-27dcd51d21ed`
pub const WIDEVINE_SYSTEM_ID: [u8; 16] = [
    0xed, 0xef, 0x8b, 0xa9, 0x79, 0xd6, 0x4a, 0xce, 0xa3, 0xc8, 0x27, 0xdc, 0xd5, 0x1d, 0x21,
    0xed,
];

/// PlayReady DRM system ID: `9a04f079-9840-4286-ab92-e65be0885f95`
pub const PLAYREADY_SYSTEM_ID: [u8; 16] = [
    0x9a, 0x04, 0xf0, 0x79, 0x98, 0x40, 0x42, 0x86, 0xab, 0x92, 0xe6, 0x5b, 0xe0, 0x88, 0x5f,
    0x95,
];

/// FairPlay Streaming DRM system ID: `94ce86fb-07ff-4f43-adb8-93d2fa968ca2`
pub const FAIRPLAY_SYSTEM_ID: [u8; 16] = [
    0x94, 0xce, 0x86, 0xfb, 0x07, 0xff, 0x4f, 0x43, 0xad, 0xb8, 0x93, 0xd2, 0xfa, 0x96, 0x8c,
    0xa2,
];

/// W3C Common Encryption system ID: `1077efec-c0b2-4d02-ace3-3c1e52e2fb4b`
pub const COMMON_ENCRYPTION_ID: [u8; 16] = [
    0x10, 0x77, 0xef, 0xec, 0xc0, 0xb2, 0x4d, 0x02, 0xac, 0xe3, 0x3c, 0x1e, 0x52, 0xe2, 0xfb,
    0x4b,
];

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors returned when parsing a PSSH box.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum PsshError {
    /// The four-byte box-type field was not `'pssh'`.
    #[error("Invalid PSSH magic: expected 'pssh'")]
    InvalidMagic,
    /// The byte slice is too short to contain a valid PSSH header.
    #[error("Input too short to be a valid PSSH box")]
    TooShort,
    /// The version field contains an unsupported value.
    #[error("Unsupported PSSH version: {0}")]
    InvalidVersion(u8),
    /// The data_size or key_id_count field would read past the end of the box.
    #[error("PSSH data overflow: field extends beyond box boundary")]
    DataOverflow,
}

// ---------------------------------------------------------------------------
// PsshBox
// ---------------------------------------------------------------------------

/// A parsed PSSH (Protection System Specific Header) box.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PsshBox {
    /// 16-byte DRM system identifier.
    pub system_id: [u8; 16],
    /// PSSH box version (0 or 1).
    pub version: u8,
    /// List of content key IDs (populated only for version ≥ 1 boxes).
    /// Each entry is exactly 16 bytes.
    pub key_ids: Vec<[u8; 16]>,
    /// System-specific DRM payload.
    pub data: Vec<u8>,
}

impl PsshBox {
    /// Return the human-readable name of the DRM system, if known.
    pub fn system_name(&self) -> Option<&'static str> {
        if self.system_id == WIDEVINE_SYSTEM_ID {
            Some("Widevine")
        } else if self.system_id == PLAYREADY_SYSTEM_ID {
            Some("PlayReady")
        } else if self.system_id == FAIRPLAY_SYSTEM_ID {
            Some("FairPlay")
        } else if self.system_id == COMMON_ENCRYPTION_ID {
            Some("Common")
        } else {
            None
        }
    }

    /// Serialize this `PsshBox` back to raw bytes.
    ///
    /// Emits a version-0 box when `self.key_ids` is empty, otherwise version-1.
    pub fn serialize(&self) -> Vec<u8> {
        let version: u8 = if self.key_ids.is_empty() { 0 } else { 1 };
        // Calculate size:
        //   4 (size) + 4 ('pssh') + 4 (version+flags) + 16 (system_id)
        //   + [v1: 4 + N*16] + 4 (data_size) + data
        let mut size: usize = 4 + 4 + 4 + 16 + 4 + self.data.len();
        if version >= 1 {
            size += 4 + self.key_ids.len() * 16;
        }

        let mut out = Vec::with_capacity(size);
        out.extend_from_slice(&(size as u32).to_be_bytes());
        out.extend_from_slice(b"pssh");
        out.push(version);
        out.push(0); // flags[0]
        out.push(0); // flags[1]
        out.push(0); // flags[2]
        out.extend_from_slice(&self.system_id);

        if version >= 1 {
            out.extend_from_slice(&(self.key_ids.len() as u32).to_be_bytes());
            for kid in &self.key_ids {
                out.extend_from_slice(kid);
            }
        }

        out.extend_from_slice(&(self.data.len() as u32).to_be_bytes());
        out.extend_from_slice(&self.data);
        out
    }
}

// ---------------------------------------------------------------------------
// PsshParser
// ---------------------------------------------------------------------------

/// Stateless PSSH box parser.
pub struct PsshParser;

impl PsshParser {
    /// Parse a single PSSH box from the beginning of `bytes`.
    ///
    /// # Errors
    ///
    /// - [`PsshError::TooShort`] — `bytes` is shorter than 32 bytes (minimum valid box).
    /// - [`PsshError::InvalidMagic`] — bytes[4..8] != b"pssh".
    /// - [`PsshError::InvalidVersion`] — version > 1 (future versions unsupported).
    /// - [`PsshError::DataOverflow`] — a length field would read past the box boundary.
    pub fn parse(bytes: &[u8]) -> Result<PsshBox, PsshError> {
        // Minimum PSSH box: 4(size)+4('pssh')+4(ver/flags)+16(system_id)+4(data_size) = 32 bytes
        if bytes.len() < 32 {
            return Err(PsshError::TooShort);
        }

        // Read declared box size
        let box_size = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
        if box_size < 32 || box_size > bytes.len() {
            return Err(PsshError::TooShort);
        }

        let box_data = &bytes[..box_size];

        // Validate magic 'pssh'
        if &box_data[4..8] != b"pssh" {
            return Err(PsshError::InvalidMagic);
        }

        let version = box_data[8];
        // Reject versions we don't understand (currently 0 and 1 are defined)
        if version > 1 {
            return Err(PsshError::InvalidVersion(version));
        }
        // flags: bytes 9-11 (consumed but not stored)

        // system_id: bytes 12-27
        let mut system_id = [0u8; 16];
        system_id.copy_from_slice(&box_data[12..28]);

        let mut offset = 28usize;

        // key_ids: only present in version 1
        let mut key_ids: Vec<[u8; 16]> = Vec::new();
        if version >= 1 {
            if offset + 4 > box_data.len() {
                return Err(PsshError::DataOverflow);
            }
            let count = u32::from_be_bytes([
                box_data[offset],
                box_data[offset + 1],
                box_data[offset + 2],
                box_data[offset + 3],
            ]) as usize;
            offset += 4;

            // Guard against extremely large counts before allocating
            if count > 65535 {
                return Err(PsshError::DataOverflow);
            }

            if offset + count * 16 > box_data.len() {
                return Err(PsshError::DataOverflow);
            }
            key_ids.reserve(count);
            for _ in 0..count {
                let mut kid = [0u8; 16];
                kid.copy_from_slice(&box_data[offset..offset + 16]);
                key_ids.push(kid);
                offset += 16;
            }
        }

        // data_size + data
        if offset + 4 > box_data.len() {
            return Err(PsshError::DataOverflow);
        }
        let data_size = u32::from_be_bytes([
            box_data[offset],
            box_data[offset + 1],
            box_data[offset + 2],
            box_data[offset + 3],
        ]) as usize;
        offset += 4;

        if offset + data_size > box_data.len() {
            return Err(PsshError::DataOverflow);
        }
        let data = box_data[offset..offset + data_size].to_vec();

        Ok(PsshBox {
            system_id,
            version,
            key_ids,
            data,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Build a minimal v0 PSSH box for the given system_id and data payload.
    fn build_v0(system_id: [u8; 16], data: &[u8]) -> Vec<u8> {
        let size: u32 = (4 + 4 + 4 + 16 + 4 + data.len()) as u32;
        let mut out = Vec::with_capacity(size as usize);
        out.extend_from_slice(&size.to_be_bytes());
        out.extend_from_slice(b"pssh");
        out.extend_from_slice(&[0, 0, 0, 0]); // version=0, flags=0
        out.extend_from_slice(&system_id);
        out.extend_from_slice(&(data.len() as u32).to_be_bytes());
        out.extend_from_slice(data);
        out
    }

    // Build a v1 PSSH box with the given key_ids and data.
    fn build_v1(system_id: [u8; 16], key_ids: &[[u8; 16]], data: &[u8]) -> Vec<u8> {
        let size: u32 =
            (4 + 4 + 4 + 16 + 4 + key_ids.len() * 16 + 4 + data.len()) as u32;
        let mut out = Vec::with_capacity(size as usize);
        out.extend_from_slice(&size.to_be_bytes());
        out.extend_from_slice(b"pssh");
        out.extend_from_slice(&[1, 0, 0, 0]); // version=1, flags=0
        out.extend_from_slice(&system_id);
        out.extend_from_slice(&(key_ids.len() as u32).to_be_bytes());
        for kid in key_ids {
            out.extend_from_slice(kid);
        }
        out.extend_from_slice(&(data.len() as u32).to_be_bytes());
        out.extend_from_slice(data);
        out
    }

    #[test]
    fn test_parse_widevine_v0() {
        let raw = build_v0(WIDEVINE_SYSTEM_ID, b"widevine-init-data");
        let pssh = PsshParser::parse(&raw).expect("parse should succeed");
        assert_eq!(pssh.system_id, WIDEVINE_SYSTEM_ID);
        assert_eq!(pssh.version, 0);
        assert!(pssh.key_ids.is_empty());
        assert_eq!(pssh.data, b"widevine-init-data");
    }

    #[test]
    fn test_parse_playready_v0() {
        let raw = build_v0(PLAYREADY_SYSTEM_ID, b"playready-data");
        let pssh = PsshParser::parse(&raw).expect("parse should succeed");
        assert_eq!(pssh.system_id, PLAYREADY_SYSTEM_ID);
        assert_eq!(pssh.version, 0);
        assert_eq!(pssh.data, b"playready-data");
    }

    #[test]
    fn test_parse_v1_with_key_ids() {
        let kid1: [u8; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let kid2: [u8; 16] = [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1];
        let raw = build_v1(WIDEVINE_SYSTEM_ID, &[kid1, kid2], b"v1-data");
        let pssh = PsshParser::parse(&raw).expect("parse should succeed");
        assert_eq!(pssh.version, 1);
        assert_eq!(pssh.key_ids.len(), 2);
        assert_eq!(pssh.key_ids[0], kid1);
        assert_eq!(pssh.key_ids[1], kid2);
        assert_eq!(pssh.data, b"v1-data");
    }

    #[test]
    fn test_parse_version_0_no_key_ids() {
        let raw = build_v0(FAIRPLAY_SYSTEM_ID, b"fp");
        let pssh = PsshParser::parse(&raw).expect("parse should succeed");
        assert_eq!(pssh.version, 0);
        assert!(pssh.key_ids.is_empty());
    }

    #[test]
    fn test_invalid_magic() {
        let raw = build_v0(WIDEVINE_SYSTEM_ID, b"data");
        let mut bad = raw.clone();
        bad[4] = b'x'; // corrupt the 'pssh' magic
        assert_eq!(PsshParser::parse(&bad).unwrap_err(), PsshError::InvalidMagic);
    }

    #[test]
    fn test_too_short() {
        let raw = vec![0u8; 10];
        assert_eq!(PsshParser::parse(&raw).unwrap_err(), PsshError::TooShort);
    }

    #[test]
    fn test_data_overflow() {
        let mut raw = build_v0(WIDEVINE_SYSTEM_ID, b"hello");
        // Layout: [0..4]=size, [4..8]='pssh', [8..12]=ver+flags,
        //         [12..28]=system_id, [28..32]=data_size, [32..37]=data
        // Overwrite data_size at offset 28 with a value larger than the box.
        let bad_size: u32 = 9999;
        raw[28..32].copy_from_slice(&bad_size.to_be_bytes());
        assert_eq!(
            PsshParser::parse(&raw).unwrap_err(),
            PsshError::DataOverflow
        );
    }

    #[test]
    fn test_system_name_widevine() {
        let raw = build_v0(WIDEVINE_SYSTEM_ID, b"");
        let pssh = PsshParser::parse(&raw).expect("parse");
        assert_eq!(pssh.system_name(), Some("Widevine"));
    }

    #[test]
    fn test_system_name_playready() {
        let raw = build_v0(PLAYREADY_SYSTEM_ID, b"");
        let pssh = PsshParser::parse(&raw).expect("parse");
        assert_eq!(pssh.system_name(), Some("PlayReady"));
    }

    #[test]
    fn test_system_name_fairplay() {
        let raw = build_v0(FAIRPLAY_SYSTEM_ID, b"");
        let pssh = PsshParser::parse(&raw).expect("parse");
        assert_eq!(pssh.system_name(), Some("FairPlay"));
    }

    #[test]
    fn test_system_name_common() {
        let raw = build_v0(COMMON_ENCRYPTION_ID, b"");
        let pssh = PsshParser::parse(&raw).expect("parse");
        assert_eq!(pssh.system_name(), Some("Common"));
    }

    #[test]
    fn test_system_name_unknown() {
        let unknown_id = [0xAAu8; 16];
        let raw = build_v0(unknown_id, b"");
        let pssh = PsshParser::parse(&raw).expect("parse");
        assert_eq!(pssh.system_name(), None);
    }

    #[test]
    fn test_round_trip_v0() {
        let pssh = PsshBox {
            system_id: WIDEVINE_SYSTEM_ID,
            version: 0,
            key_ids: Vec::new(),
            data: b"round-trip-v0".to_vec(),
        };
        let raw = pssh.serialize();
        let parsed = PsshParser::parse(&raw).expect("round-trip parse");
        assert_eq!(parsed.system_id, pssh.system_id);
        assert_eq!(parsed.data, pssh.data);
        assert!(parsed.key_ids.is_empty());
    }

    #[test]
    fn test_round_trip_v1() {
        let kid: [u8; 16] = [0xAA; 16];
        let pssh = PsshBox {
            system_id: PLAYREADY_SYSTEM_ID,
            version: 1,
            key_ids: vec![kid],
            data: b"round-trip-v1".to_vec(),
        };
        let raw = pssh.serialize();
        let parsed = PsshParser::parse(&raw).expect("round-trip parse");
        assert_eq!(parsed.version, 1);
        assert_eq!(parsed.key_ids, vec![kid]);
        assert_eq!(parsed.data, pssh.data);
    }
}
