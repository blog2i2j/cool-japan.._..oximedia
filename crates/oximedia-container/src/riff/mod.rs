//! RIFF chunk write helpers shared across RIFF-based muxers (WAV, AVI).
//!
//! These utilities deal with building RIFF byte buffers in memory.  They are
//! intentionally kept low-level (operating on `Vec<u8>`) so that they can be
//! used from both synchronous and async contexts without any I/O dependency.

#![forbid(unsafe_code)]
#![allow(clippy::cast_possible_truncation)]

/// Write a RIFF leaf chunk: fourcc(4) + size(4 LE) + payload, padded to even boundary.
///
/// The padding byte (if needed) is `0x00` per the RIFF specification.
pub fn write_chunk(out: &mut Vec<u8>, fourcc: &[u8; 4], payload: &[u8]) {
    out.extend_from_slice(fourcc);
    let size = payload.len() as u32;
    out.extend_from_slice(&size.to_le_bytes());
    out.extend_from_slice(payload);
    if payload.len() % 2 == 1 {
        out.push(0); // RIFF padding byte
    }
}

/// Write a RIFF LIST chunk: `LIST`(4) + total_size(4 LE) + list_type(4) + payload.
///
/// `total_size` is the length of `list_type || payload`, i.e. `4 + payload.len()`.
///
/// Unlike leaf chunks, LIST chunks are not automatically padded here because
/// their payloads are composed of self-padded child chunks and therefore are
/// always even-length in practice.
pub fn write_list(out: &mut Vec<u8>, list_type: &[u8; 4], payload: &[u8]) {
    let total = 4u32 + payload.len() as u32; // list_type(4) + payload
    out.extend_from_slice(b"LIST");
    out.extend_from_slice(&total.to_le_bytes());
    out.extend_from_slice(list_type);
    out.extend_from_slice(payload);
}

/// Patch a `u32` LE value at a given byte offset in `buf`.
///
/// # Panics
///
/// Panics in debug mode if `offset + 4 > buf.len()`.
pub fn patch_u32_le(buf: &mut [u8], offset: usize, value: u32) {
    buf[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_chunk_even_payload() {
        let mut out = Vec::new();
        write_chunk(&mut out, b"data", &[1u8, 2, 3, 4]);
        // fourcc(4) + size(4) + payload(4) = 12 bytes, no padding
        assert_eq!(out.len(), 12);
        assert_eq!(&out[0..4], b"data");
        assert_eq!(u32::from_le_bytes([out[4], out[5], out[6], out[7]]), 4);
        assert_eq!(&out[8..12], &[1, 2, 3, 4]);
    }

    #[test]
    fn write_chunk_odd_payload() {
        let mut out = Vec::new();
        write_chunk(&mut out, b"test", &[0xABu8]);
        // fourcc(4) + size(4) + payload(1) + pad(1) = 10 bytes
        assert_eq!(out.len(), 10);
        assert_eq!(u32::from_le_bytes([out[4], out[5], out[6], out[7]]), 1);
        assert_eq!(out[8], 0xAB);
        assert_eq!(out[9], 0x00); // padding
    }

    #[test]
    fn write_list_structure() {
        let payload = b"abcd";
        let mut out = Vec::new();
        write_list(&mut out, b"hdrl", payload);
        // LIST(4) + total_size(4) + list_type(4) + payload(4) = 16
        assert_eq!(out.len(), 16);
        assert_eq!(&out[0..4], b"LIST");
        // total = 4 (list_type) + 4 (payload) = 8
        assert_eq!(u32::from_le_bytes([out[4], out[5], out[6], out[7]]), 8);
        assert_eq!(&out[8..12], b"hdrl");
        assert_eq!(&out[12..16], b"abcd");
    }

    #[test]
    fn patch_u32_le_works() {
        let mut buf = vec![0u8; 8];
        patch_u32_le(&mut buf, 2, 0xDEAD_BEEFu32);
        assert_eq!(&buf[2..6], &[0xEF, 0xBE, 0xAD, 0xDE]);
    }
}
