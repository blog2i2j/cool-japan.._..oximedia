//! RPU (Reference Processing Unit) binary validation.
//!
//! [`RpuValidator`] performs structural checks on raw RPU byte slices before
//! full parsing.  The checks are intentionally conservative: a failed check
//! adds a human-readable message to the error list, but validation continues
//! so the caller receives a complete picture of all problems.
//!
//! # Checks performed
//!
//! 1. **Magic bytes** — the first two bytes must be `0x19 0x01` (the HEVC NAL
//!    unit type for RPU data).
//! 2. **Minimum length** — an RPU with fewer than 8 bytes cannot contain a
//!    valid header.
//! 3. **CRC-32** — the last four bytes are interpreted as a little-endian
//!    CRC-32/ISO-HDLC checksum of all preceding bytes.  A mismatch is reported
//!    but does not prevent other checks from running.
//! 4. **Profile field** — byte index 2 must encode a recognised Dolby Vision
//!    profile number (5, 7, 8, 81, or 84 in the mapping table).

/// Magic byte prefix expected at the start of every RPU NAL unit.
///
/// These correspond to the HEVC NAL unit type 0x19 (25 decimal) with a
/// one-byte NUH layer / temporal ID byte of 0x01.
pub const RPU_MAGIC: [u8; 2] = [0x19, 0x01];

/// Minimum byte length required to contain a structurally valid RPU header.
pub const RPU_MIN_LEN: usize = 8;

/// Stateless RPU byte-slice validator.
///
/// All validation methods return a `Vec<String>` containing every problem
/// found.  An empty vector means the slice passed all checks.
pub struct RpuValidator;

impl RpuValidator {
    /// Validate a raw RPU byte slice.
    ///
    /// Returns a list of validation error messages.  An empty list indicates
    /// that the slice passes all structural checks.
    ///
    /// # Checks
    ///
    /// - Magic bytes `[0x19, 0x01]` at the start of the slice.
    /// - Minimum length of [`RPU_MIN_LEN`] bytes.
    /// - CRC-32/ISO-HDLC of all bytes except the final 4 must match the
    ///   little-endian u32 stored in bytes `[len-4 .. len]`.
    /// - Profile byte (index 2) must be a recognised Dolby Vision profile.
    #[must_use]
    pub fn validate(rpu: &[u8]) -> Vec<String> {
        let mut errors = Vec::new();

        // ── 1. Minimum length ──────────────────────────────────────────────
        if rpu.len() < RPU_MIN_LEN {
            errors.push(format!(
                "RPU too short: {} bytes (minimum {})",
                rpu.len(),
                RPU_MIN_LEN
            ));
            // Without enough bytes further checks are meaningless.
            return errors;
        }

        // ── 2. Magic bytes ─────────────────────────────────────────────────
        if rpu[0] != RPU_MAGIC[0] || rpu[1] != RPU_MAGIC[1] {
            errors.push(format!(
                "invalid magic bytes: expected [{:#04x}, {:#04x}], got [{:#04x}, {:#04x}]",
                RPU_MAGIC[0], RPU_MAGIC[1], rpu[0], rpu[1]
            ));
        }

        // ── 3. Profile field ───────────────────────────────────────────────
        let profile_byte = rpu[2];
        if !Self::is_known_profile(profile_byte) {
            errors.push(format!(
                "unknown profile byte {profile_byte:#04x} at offset 2"
            ));
        }

        // ── 4. CRC-32 ──────────────────────────────────────────────────────
        if rpu.len() >= 4 {
            let payload = &rpu[..rpu.len() - 4];
            let stored_crc = u32::from_le_bytes([
                rpu[rpu.len() - 4],
                rpu[rpu.len() - 3],
                rpu[rpu.len() - 2],
                rpu[rpu.len() - 1],
            ]);
            let computed_crc = crc32_iso_hdlc(payload);
            if computed_crc != stored_crc {
                errors.push(format!(
                    "CRC mismatch: stored {stored_crc:#010x}, computed {computed_crc:#010x}"
                ));
            }
        }

        errors
    }

    /// Returns `true` when `profile_byte` corresponds to a known Dolby Vision
    /// profile identifier.
    #[must_use]
    fn is_known_profile(profile_byte: u8) -> bool {
        // Profile 5=0x05, 7=0x07, 8=0x08, 8.1 encoded as 0x51 (81 dec),
        // 8.4 encoded as 0x54 (84 dec) in some implementations.  We also
        // accept the raw decimal values for flexibility.
        matches!(profile_byte, 5 | 7 | 8 | 81 | 84)
    }
}

/// CRC-32/ISO-HDLC (the standard "CRC-32" used in Ethernet, PNG, etc.).
///
/// Polynomial: `0xEDB88320` (reflected representation of `0x04C11DB7`).
#[must_use]
pub fn crc32_iso_hdlc(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        let idx = ((crc ^ u32::from(byte)) & 0xFF) as usize;
        crc = (crc >> 8) ^ CRC32_TABLE[idx];
    }
    crc ^ 0xFFFF_FFFF
}

/// Pre-computed CRC-32/ISO-HDLC lookup table (256 entries).
const CRC32_TABLE: [u32; 256] = {
    let poly: u32 = 0xEDB8_8320;
    let mut table = [0u32; 256];
    let mut i = 0usize;
    while i < 256 {
        let mut crc = i as u32;
        let mut j = 0usize;
        while j < 8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ poly;
            } else {
                crc >>= 1;
            }
            j += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
};

/// Build a minimal valid RPU byte slice for testing purposes.
///
/// The returned slice has the correct magic bytes, a given profile byte, and a
/// valid CRC-32 appended.
#[must_use]
pub fn build_test_rpu(profile_byte: u8, extra_payload: &[u8]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(3 + extra_payload.len() + 4);
    buf.extend_from_slice(&RPU_MAGIC);
    buf.push(profile_byte);
    buf.extend_from_slice(extra_payload);
    // Pad to minimum length if necessary (at least 4 payload bytes before CRC)
    while buf.len() < RPU_MIN_LEN - 4 {
        buf.push(0x00);
    }
    let crc = crc32_iso_hdlc(&buf);
    buf.extend_from_slice(&crc.to_le_bytes());
    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_rpu() {
        let rpu = build_test_rpu(8, &[0xAB, 0xCD, 0xEF, 0x00]);
        let errors = RpuValidator::validate(&rpu);
        assert!(
            errors.is_empty(),
            "valid RPU should pass: {:?}",
            errors
        );
    }

    #[test]
    fn test_too_short() {
        let errors = RpuValidator::validate(&[0x19, 0x01]);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("too short"));
    }

    #[test]
    fn test_bad_magic() {
        let mut rpu = build_test_rpu(8, &[0x00, 0x00, 0x00, 0x00]);
        rpu[0] = 0xFF; // corrupt magic
        // Re-compute CRC so only magic is wrong
        let payload_len = rpu.len() - 4;
        let crc = crc32_iso_hdlc(&rpu[..payload_len]);
        let crc_bytes = crc.to_le_bytes();
        let len = rpu.len();
        rpu[len - 4..].copy_from_slice(&crc_bytes);

        let errors = RpuValidator::validate(&rpu);
        let has_magic_err = errors.iter().any(|e| e.contains("magic bytes"));
        assert!(has_magic_err, "should report magic bytes error: {:?}", errors);
    }

    #[test]
    fn test_bad_crc() {
        let mut rpu = build_test_rpu(8, &[0x00, 0x00, 0x00, 0x00]);
        let len = rpu.len();
        rpu[len - 1] ^= 0xFF; // corrupt CRC
        let errors = RpuValidator::validate(&rpu);
        let has_crc_err = errors.iter().any(|e| e.contains("CRC mismatch"));
        assert!(has_crc_err, "should report CRC mismatch: {:?}", errors);
    }

    #[test]
    fn test_unknown_profile() {
        let rpu = build_test_rpu(99, &[0x00, 0x00, 0x00, 0x00]);
        let errors = RpuValidator::validate(&rpu);
        let has_profile_err = errors.iter().any(|e| e.contains("unknown profile"));
        assert!(has_profile_err, "should report unknown profile: {:?}", errors);
    }

    #[test]
    fn test_known_profiles() {
        for profile in [5u8, 7, 8, 81, 84] {
            let rpu = build_test_rpu(profile, &[0x00, 0x00, 0x00, 0x00]);
            let errors = RpuValidator::validate(&rpu);
            // Should have no profile error (CRC might or might not be wrong
            // depending on build_test_rpu, but no profile error expected)
            let profile_errors: Vec<_> = errors
                .iter()
                .filter(|e| e.contains("unknown profile"))
                .collect();
            assert!(
                profile_errors.is_empty(),
                "profile {profile} should be recognised: {:?}",
                errors
            );
        }
    }

    #[test]
    fn test_crc32_empty() {
        // CRC of empty slice is a well-known value: 0x00000000 after XOR out.
        let crc = crc32_iso_hdlc(&[]);
        assert_eq!(crc, 0x0000_0000);
    }

    #[test]
    fn test_crc32_known_value() {
        // CRC-32 of b"123456789" is 0xCBF43926.
        let crc = crc32_iso_hdlc(b"123456789");
        assert_eq!(crc, 0xCBF4_3926);
    }
}
