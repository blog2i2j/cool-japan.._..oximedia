//! PSSH (Protection System Specific Header) box parsing and serialization.
//!
//! Implements the ISO BMFF PSSH box format as defined in ISO/IEC 23001-7.
//!
//! Box layout:
//! - 4 bytes: total size (big-endian)
//! - 4 bytes: box type ('pssh')
//! - 1 byte:  version
//! - 3 bytes: flags
//! - 16 bytes: system_id
//! - [version >= 1] 4 bytes key_id_count, N×16 bytes key_ids
//! - 4 bytes: data_size
//! - N bytes: data

/// Well-known DRM system IDs
pub const WIDEVINE_SYSTEM_ID: [u8; 16] = [
    0xed, 0xef, 0x8b, 0xa9, 0x79, 0xd6, 0x4a, 0xce, 0xa3, 0xc8, 0x27, 0xdc, 0xd5, 0x1d, 0x21, 0xed,
];

pub const PLAYREADY_SYSTEM_ID: [u8; 16] = [
    0x9a, 0x04, 0xf0, 0x79, 0x98, 0x40, 0x42, 0x86, 0xab, 0x92, 0xe6, 0x5b, 0xe0, 0x88, 0x5f, 0x95,
];

pub const FAIRPLAY_SYSTEM_ID: [u8; 16] = [
    0x94, 0xce, 0x86, 0xfb, 0x07, 0xff, 0x4f, 0x43, 0xad, 0xb8, 0x93, 0xd2, 0xfa, 0x96, 0x8c, 0xa2,
];

pub const CLEARKEY_SYSTEM_ID: [u8; 16] = [
    0x10, 0x77, 0xef, 0xec, 0xc0, 0xb2, 0x4d, 0x02, 0xac, 0xe3, 0x3c, 0x1e, 0x52, 0xe2, 0xfb, 0x4b,
];

/// A single PSSH box
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub struct PsshBox {
    /// 16-byte DRM system identifier
    pub system_id: [u8; 16],
    /// Key IDs (populated for version 1 boxes)
    pub key_ids: Vec<Vec<u8>>,
    /// System-specific DRM data
    pub data: Vec<u8>,
}

impl PsshBox {
    /// Parse one or more PSSH boxes from a byte slice.
    ///
    /// The slice may contain multiple concatenated PSSH boxes; all are returned.
    pub fn parse(mut data: &[u8]) -> Result<Vec<PsshBox>, String> {
        let mut boxes = Vec::new();

        while !data.is_empty() {
            if data.len() < 8 {
                return Err(format!(
                    "Not enough bytes for PSSH header: need 8, have {}",
                    data.len()
                ));
            }

            let size = u32::from_be_bytes([data[0], data[1], data[2], data[3]]) as usize;
            if size < 32 || size > data.len() {
                return Err(format!(
                    "Invalid PSSH box size: {} (available: {})",
                    size,
                    data.len()
                ));
            }

            let box_data = &data[..size];

            // Check box type 'pssh'
            if &box_data[4..8] != b"pssh" {
                return Err(format!(
                    "Expected box type 'pssh', got '{}'",
                    String::from_utf8_lossy(&box_data[4..8])
                ));
            }

            let version = box_data[8];
            // flags: bytes 9-11 (ignored but consumed)
            let _flags = [box_data[9], box_data[10], box_data[11]];

            // system_id: bytes 12-27
            let mut system_id = [0u8; 16];
            system_id.copy_from_slice(&box_data[12..28]);

            let mut offset = 28usize;

            // key_ids present in version >= 1
            let mut key_ids = Vec::new();
            if version >= 1 {
                if offset + 4 > box_data.len() {
                    return Err("Truncated PSSH: missing key_id_count".to_string());
                }
                let key_id_count = u32::from_be_bytes([
                    box_data[offset],
                    box_data[offset + 1],
                    box_data[offset + 2],
                    box_data[offset + 3],
                ]) as usize;
                offset += 4;

                for _ in 0..key_id_count {
                    if offset + 16 > box_data.len() {
                        return Err("Truncated PSSH: not enough bytes for key_id".to_string());
                    }
                    key_ids.push(box_data[offset..offset + 16].to_vec());
                    offset += 16;
                }
            }

            // data_size + data
            if offset + 4 > box_data.len() {
                return Err("Truncated PSSH: missing data_size".to_string());
            }
            let data_size = u32::from_be_bytes([
                box_data[offset],
                box_data[offset + 1],
                box_data[offset + 2],
                box_data[offset + 3],
            ]) as usize;
            offset += 4;

            if offset + data_size > box_data.len() {
                return Err(format!(
                    "Truncated PSSH: data_size {} exceeds box bounds",
                    data_size
                ));
            }
            let pssh_data = box_data[offset..offset + data_size].to_vec();

            boxes.push(PsshBox {
                system_id,
                key_ids,
                data: pssh_data,
            });

            data = &data[size..];
        }

        Ok(boxes)
    }

    /// Serialize this PSSH box to bytes.
    ///
    /// Produces a version-0 box when there are no key_ids, otherwise version-1.
    pub fn serialize(&self) -> Vec<u8> {
        let version: u8 = if self.key_ids.is_empty() { 0 } else { 1 };

        // Calculate total size
        let mut size: usize = 4 + 4 + 1 + 3 + 16 + 4 + self.data.len();
        if version >= 1 {
            size += 4 + self.key_ids.len() * 16;
        }

        let mut out = Vec::with_capacity(size);

        // size
        out.extend_from_slice(&(size as u32).to_be_bytes());
        // box type
        out.extend_from_slice(b"pssh");
        // version + flags (3 bytes)
        out.push(version);
        out.push(0);
        out.push(0);
        out.push(0);
        // system_id
        out.extend_from_slice(&self.system_id);

        // key_ids (version 1 only)
        if version >= 1 {
            out.extend_from_slice(&(self.key_ids.len() as u32).to_be_bytes());
            for kid in &self.key_ids {
                // Pad or truncate to 16 bytes
                let mut kid16 = [0u8; 16];
                let copy_len = kid.len().min(16);
                kid16[..copy_len].copy_from_slice(&kid[..copy_len]);
                out.extend_from_slice(&kid16);
            }
        }

        // data
        out.extend_from_slice(&(self.data.len() as u32).to_be_bytes());
        out.extend_from_slice(&self.data);

        out
    }

    /// Return the name of the DRM system for this box, if known
    pub fn drm_system_name(&self) -> Option<&'static str> {
        if self.system_id == WIDEVINE_SYSTEM_ID {
            Some("Widevine")
        } else if self.system_id == PLAYREADY_SYSTEM_ID {
            Some("PlayReady")
        } else if self.system_id == FAIRPLAY_SYSTEM_ID {
            Some("FairPlay")
        } else if self.system_id == CLEARKEY_SYSTEM_ID {
            Some("ClearKey")
        } else {
            None
        }
    }
}

/// Builder for constructing PSSH boxes
#[derive(Default)]
pub struct PsshBuilder {
    system_id: [u8; 16],
    key_ids: Vec<Vec<u8>>,
    data: Vec<u8>,
}

impl PsshBuilder {
    /// Create a new builder with a zeroed system_id
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the system_id
    pub fn set_system_id(mut self, system_id: [u8; 16]) -> Self {
        self.system_id = system_id;
        self
    }

    /// Add a key_id (will be included in a version-1 box)
    pub fn add_key_id(mut self, key_id: Vec<u8>) -> Self {
        self.key_ids.push(key_id);
        self
    }

    /// Set the system-specific data payload
    pub fn set_data(mut self, data: Vec<u8>) -> Self {
        self.data = data;
        self
    }

    /// Build the PsshBox
    pub fn build(self) -> PsshBox {
        PsshBox {
            system_id: self.system_id,
            key_ids: self.key_ids,
            data: self.data,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn build_v0_box(system_id: [u8; 16], data: &[u8]) -> Vec<u8> {
        let size = 4 + 4 + 1 + 3 + 16 + 4 + data.len();
        let mut out = Vec::with_capacity(size);
        out.extend_from_slice(&(size as u32).to_be_bytes());
        out.extend_from_slice(b"pssh");
        out.push(0); // version
        out.push(0);
        out.push(0);
        out.push(0); // flags
        out.extend_from_slice(&system_id);
        out.extend_from_slice(&(data.len() as u32).to_be_bytes());
        out.extend_from_slice(data);
        out
    }

    fn build_v1_box(system_id: [u8; 16], key_ids: &[Vec<u8>], data: &[u8]) -> Vec<u8> {
        let size = 4 + 4 + 1 + 3 + 16 + 4 + key_ids.len() * 16 + 4 + data.len();
        let mut out = Vec::with_capacity(size);
        out.extend_from_slice(&(size as u32).to_be_bytes());
        out.extend_from_slice(b"pssh");
        out.push(1); // version
        out.push(0);
        out.push(0);
        out.push(0); // flags
        out.extend_from_slice(&system_id);
        out.extend_from_slice(&(key_ids.len() as u32).to_be_bytes());
        for kid in key_ids {
            let mut buf = [0u8; 16];
            buf[..kid.len().min(16)].copy_from_slice(&kid[..kid.len().min(16)]);
            out.extend_from_slice(&buf);
        }
        out.extend_from_slice(&(data.len() as u32).to_be_bytes());
        out.extend_from_slice(data);
        out
    }

    #[test]
    fn test_parse_v0_box() {
        let raw = build_v0_box(WIDEVINE_SYSTEM_ID, b"hello");
        let boxes = PsshBox::parse(&raw).expect("operation should succeed");
        assert_eq!(boxes.len(), 1);
        assert_eq!(boxes[0].system_id, WIDEVINE_SYSTEM_ID);
        assert!(boxes[0].key_ids.is_empty());
        assert_eq!(boxes[0].data, b"hello");
    }

    #[test]
    fn test_parse_v1_box() {
        let kids = vec![vec![1u8; 16], vec![2u8; 16]];
        let raw = build_v1_box(PLAYREADY_SYSTEM_ID, &kids, b"world");
        let boxes = PsshBox::parse(&raw).expect("operation should succeed");
        assert_eq!(boxes.len(), 1);
        assert_eq!(boxes[0].system_id, PLAYREADY_SYSTEM_ID);
        assert_eq!(boxes[0].key_ids.len(), 2);
        assert_eq!(boxes[0].data, b"world");
    }

    #[test]
    fn test_parse_multiple_boxes() {
        let mut raw = build_v0_box(WIDEVINE_SYSTEM_ID, b"wv");
        raw.extend(build_v0_box(PLAYREADY_SYSTEM_ID, b"pr"));
        let boxes = PsshBox::parse(&raw).expect("operation should succeed");
        assert_eq!(boxes.len(), 2);
        assert_eq!(boxes[0].system_id, WIDEVINE_SYSTEM_ID);
        assert_eq!(boxes[1].system_id, PLAYREADY_SYSTEM_ID);
    }

    #[test]
    fn test_serialize_roundtrip_v0() {
        let pssh = PsshBox {
            system_id: CLEARKEY_SYSTEM_ID,
            key_ids: vec![],
            data: b"clear".to_vec(),
        };
        let bytes = pssh.serialize();
        let parsed = PsshBox::parse(&bytes).expect("operation should succeed");
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0], pssh);
    }

    #[test]
    fn test_serialize_roundtrip_v1() {
        let pssh = PsshBox {
            system_id: WIDEVINE_SYSTEM_ID,
            key_ids: vec![vec![0xABu8; 16], vec![0xCDu8; 16]],
            data: b"drm-data".to_vec(),
        };
        let bytes = pssh.serialize();
        let parsed = PsshBox::parse(&bytes).expect("operation should succeed");
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0], pssh);
    }

    #[test]
    fn test_drm_system_name() {
        let wv = PsshBox {
            system_id: WIDEVINE_SYSTEM_ID,
            key_ids: vec![],
            data: vec![],
        };
        assert_eq!(wv.drm_system_name(), Some("Widevine"));

        let pr = PsshBox {
            system_id: PLAYREADY_SYSTEM_ID,
            key_ids: vec![],
            data: vec![],
        };
        assert_eq!(pr.drm_system_name(), Some("PlayReady"));

        let fp = PsshBox {
            system_id: FAIRPLAY_SYSTEM_ID,
            key_ids: vec![],
            data: vec![],
        };
        assert_eq!(fp.drm_system_name(), Some("FairPlay"));

        let ck = PsshBox {
            system_id: CLEARKEY_SYSTEM_ID,
            key_ids: vec![],
            data: vec![],
        };
        assert_eq!(ck.drm_system_name(), Some("ClearKey"));

        let unknown = PsshBox {
            system_id: [0u8; 16],
            key_ids: vec![],
            data: vec![],
        };
        assert_eq!(unknown.drm_system_name(), None);
    }

    #[test]
    fn test_builder_basic() {
        let pssh = PsshBuilder::new()
            .set_system_id(WIDEVINE_SYSTEM_ID)
            .set_data(b"payload".to_vec())
            .build();
        assert_eq!(pssh.system_id, WIDEVINE_SYSTEM_ID);
        assert_eq!(pssh.data, b"payload");
        assert!(pssh.key_ids.is_empty());
    }

    #[test]
    fn test_builder_with_key_ids() {
        let pssh = PsshBuilder::new()
            .set_system_id(PLAYREADY_SYSTEM_ID)
            .add_key_id(vec![1u8; 16])
            .add_key_id(vec![2u8; 16])
            .set_data(b"data".to_vec())
            .build();
        assert_eq!(pssh.key_ids.len(), 2);
    }

    #[test]
    fn test_parse_empty_data_field() {
        let raw = build_v0_box(WIDEVINE_SYSTEM_ID, b"");
        let boxes = PsshBox::parse(&raw).expect("operation should succeed");
        assert_eq!(boxes[0].data, b"");
    }

    #[test]
    fn test_parse_invalid_box_type() {
        let mut raw = build_v0_box(WIDEVINE_SYSTEM_ID, b"x");
        raw[4] = b'X'; // corrupt box type
        let result = PsshBox::parse(&raw);
        assert!(result.is_err());
    }

    #[test]
    fn test_system_id_constants_length() {
        assert_eq!(WIDEVINE_SYSTEM_ID.len(), 16);
        assert_eq!(PLAYREADY_SYSTEM_ID.len(), 16);
        assert_eq!(FAIRPLAY_SYSTEM_ID.len(), 16);
        assert_eq!(CLEARKEY_SYSTEM_ID.len(), 16);
    }
}
