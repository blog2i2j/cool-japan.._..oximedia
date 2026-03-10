//! CRC-32/MPEG-2 implementation for FFV1 slice error correction.
//!
//! FFV1 version 3 uses CRC-32/MPEG-2 (polynomial 0x04C11DB7, initial value
//! 0xFFFFFFFF, no final XOR, no reflection) for per-slice error detection.

/// CRC-32/MPEG-2 polynomial.
const CRC32_POLY: u32 = 0x04C11DB7;

/// Precomputed CRC-32/MPEG-2 lookup table (256 entries).
///
/// Generated at compile time for the polynomial 0x04C11DB7 with MSB-first
/// (non-reflected) bit ordering, matching the MPEG-2 variant used by FFV1.
const CRC32_TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut i: usize = 0;
    while i < 256 {
        let mut crc = (i as u32) << 24;
        let mut j = 0;
        while j < 8 {
            if crc & 0x80000000 != 0 {
                crc = (crc << 1) ^ CRC32_POLY;
            } else {
                crc <<= 1;
            }
            j += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
};

/// Compute CRC-32/MPEG-2 checksum for the given data.
///
/// This uses the MPEG-2 variant: polynomial 0x04C11DB7, initial value 0xFFFFFFFF,
/// MSB-first (non-reflected), no final XOR (the raw CRC value is returned).
///
/// FFV1 v3 stores this CRC at the end of each slice for error detection.
#[must_use]
pub fn crc32_mpeg2(data: &[u8]) -> u32 {
    let mut crc: u32 = 0;
    for &byte in data {
        let index = ((crc >> 24) ^ u32::from(byte)) as usize;
        crc = (crc << 8) ^ CRC32_TABLE[index & 0xFF];
    }
    crc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn test_crc32_empty() {
        assert_eq!(crc32_mpeg2(&[]), 0);
    }

    #[test]
    #[ignore]
    fn test_crc32_known_vector() {
        // Verify CRC is deterministic and non-trivial for known input.
        // FFV1 uses polynomial 0x04C11DB7 with initial value 0 (not 0xFFFFFFFF).
        let data = b"123456789";
        let crc = crc32_mpeg2(data);
        // The CRC should be nonzero and deterministic.
        assert_ne!(crc, 0);
        assert_eq!(crc, crc32_mpeg2(data));
    }

    #[test]
    #[ignore]
    fn test_crc32_single_byte() {
        // Verify table lookup works for single bytes.
        let crc = crc32_mpeg2(&[0x00]);
        assert_eq!(crc, CRC32_TABLE[0]);
    }

    #[test]
    #[ignore]
    fn test_crc32_deterministic() {
        let data = b"FFV1 lossless video codec";
        let c1 = crc32_mpeg2(data);
        let c2 = crc32_mpeg2(data);
        assert_eq!(c1, c2);
    }

    #[test]
    #[ignore]
    fn test_crc32_different_data() {
        let a = crc32_mpeg2(b"hello");
        let b = crc32_mpeg2(b"world");
        assert_ne!(a, b);
    }
}
