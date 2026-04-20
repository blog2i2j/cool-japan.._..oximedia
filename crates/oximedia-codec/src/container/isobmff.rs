//! ISOBMFF (ISO Base Media File Format / ISO 14496-12) box primitives.
//!
//! Provides low-level utilities for reading and writing ISOBMFF boxes used by
//! container formats such as JXL (ISO/IEC 18181-2), HEIF, AVIF, and MP4.

use std::io::{self, Read};

/// Write an ISOBMFF box with a 4-byte type code.
///
/// Uses a 32-bit size field when the total box fits in `u32::MAX` bytes, and
/// falls back to the extended 64-bit `largesize` form for very large payloads.
///
/// # Box layout (standard)
/// ```text
/// [4 bytes: size] [4 bytes: fourcc] [payload]
/// ```
///
/// # Box layout (extended)
/// ```text
/// [4 bytes: 0x00000001] [4 bytes: fourcc] [8 bytes: largesize] [payload]
/// ```
pub fn make_box(fourcc: [u8; 4], payload: &[u8]) -> Vec<u8> {
    let payload_len = payload.len();
    if let Ok(size) = u32::try_from(payload_len + 8) {
        let mut out = Vec::with_capacity(8 + payload_len);
        out.extend_from_slice(&size.to_be_bytes());
        out.extend_from_slice(&fourcc);
        out.extend_from_slice(payload);
        out
    } else {
        // Extended size box: size=1 signals that a 64-bit largesize follows.
        let size64 = (payload_len as u64) + 16; // 8 header + 8 largesize
        let mut out = Vec::with_capacity(16 + payload_len);
        out.extend_from_slice(&1u32.to_be_bytes()); // size = 1 → largesize
        out.extend_from_slice(&fourcc);
        out.extend_from_slice(&size64.to_be_bytes());
        out.extend_from_slice(payload);
        out
    }
}

/// Write an ISOBMFF FullBox (version + 24-bit flags precede the payload).
///
/// # Box layout
/// ```text
/// [make_box header] [1 byte: version] [3 bytes: flags] [payload]
/// ```
pub fn make_full_box(fourcc: [u8; 4], version: u8, flags: u32, payload: &[u8]) -> Vec<u8> {
    let mut full_payload = Vec::with_capacity(4 + payload.len());
    full_payload.push(version);
    let flags_be = flags.to_be_bytes();
    full_payload.extend_from_slice(&flags_be[1..4]); // 24-bit flags (big-endian, skip MSB)
    full_payload.extend_from_slice(payload);
    make_box(fourcc, &full_payload)
}

/// An iterator over ISOBMFF boxes in a `Read` stream.
///
/// Each successful item is `([u8; 4], Vec<u8>)` where the first element is the
/// 4-byte box type code and the second is the box payload (excluding the
/// 8-byte / 16-byte header).
pub struct BoxIter<R: Read> {
    reader: R,
    done: bool,
}

impl<R: Read> BoxIter<R> {
    /// Create a new `BoxIter` wrapping `reader`.
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            done: false,
        }
    }
}

impl<R: Read> Iterator for BoxIter<R> {
    type Item = io::Result<([u8; 4], Vec<u8>)>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        // Read 8-byte box header (size + fourcc).
        let mut header = [0u8; 8];
        match self.reader.read_exact(&mut header) {
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {
                self.done = true;
                return None;
            }
            Err(e) => {
                self.done = true;
                return Some(Err(e));
            }
            Ok(()) => {}
        }

        // Parse size field and fourcc — use copy_from_slice to avoid unwrap().
        let mut size_bytes = [0u8; 4];
        size_bytes.copy_from_slice(&header[0..4]);
        let size_field = u32::from_be_bytes(size_bytes);

        let mut fourcc = [0u8; 4];
        fourcc.copy_from_slice(&header[4..8]);

        let payload_len: usize = match size_field {
            0 => {
                // Box extends to EOF — read the remainder.
                let mut payload = Vec::new();
                if let Err(e) = self.reader.read_to_end(&mut payload) {
                    self.done = true;
                    return Some(Err(e));
                }
                self.done = true;
                return Some(Ok((fourcc, payload)));
            }
            1 => {
                // Extended size: next 8 bytes hold the full 64-bit box size.
                let mut ls = [0u8; 8];
                if let Err(e) = self.reader.read_exact(&mut ls) {
                    self.done = true;
                    return Some(Err(e));
                }
                let total = u64::from_be_bytes(ls);
                // Payload = total_size − 8 (header) − 8 (largesize field)
                (total as usize).saturating_sub(16)
            }
            n => {
                // Standard 32-bit size: payload is size − 8 (header).
                (n as usize).saturating_sub(8)
            }
        };

        let mut payload = vec![0u8; payload_len];
        if let Err(e) = self.reader.read_exact(&mut payload) {
            self.done = true;
            return Some(Err(e));
        }
        Some(Ok((fourcc, payload)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn make_box_32bit_layout() {
        let payload = b"hello";
        let boxed = make_box(*b"test", payload);
        // Total = 8 header + 5 payload = 13
        assert_eq!(boxed.len(), 13);
        let mut size_buf = [0u8; 4];
        size_buf.copy_from_slice(&boxed[0..4]);
        assert_eq!(u32::from_be_bytes(size_buf), 13);
        assert_eq!(&boxed[4..8], b"test");
        assert_eq!(&boxed[8..], b"hello");
    }

    #[test]
    fn make_full_box_layout() {
        let payload = b"data";
        let boxed = make_full_box(*b"mdhd", 0, 0x000001, payload);
        // Outer size = 8 + 1 (version) + 3 (flags) + 4 (payload) = 16
        assert_eq!(boxed.len(), 16);
        // Version byte
        assert_eq!(boxed[8], 0);
        // 24-bit flags = 0x000001
        assert_eq!(&boxed[9..12], &[0x00, 0x00, 0x01]);
    }

    #[test]
    fn box_iter_single_box() {
        let boxed = make_box(*b"foo1", b"abcd");
        let parsed: Vec<_> = BoxIter::new(Cursor::new(boxed))
            .collect::<Result<Vec<_>, _>>()
            .expect("parse ok");
        assert_eq!(parsed.len(), 1);
        assert_eq!(&parsed[0].0, b"foo1");
        assert_eq!(&parsed[0].1, b"abcd");
    }

    #[test]
    fn box_iter_multiple_boxes() {
        let box1 = make_box(*b"foo1", b"aaaa");
        let box2 = make_box(*b"foo2", b"bbbb");
        let mut combined = box1;
        combined.extend(box2);
        let parsed: Vec<_> = BoxIter::new(Cursor::new(combined))
            .collect::<Result<Vec<_>, _>>()
            .expect("parse ok");
        assert_eq!(parsed.len(), 2);
        assert_eq!(&parsed[0].0, b"foo1");
        assert_eq!(&parsed[1].0, b"foo2");
        assert_eq!(&parsed[0].1, b"aaaa");
        assert_eq!(&parsed[1].1, b"bbbb");
    }

    #[test]
    fn box_iter_empty_payload() {
        let boxed = make_box(*b"free", b"");
        let parsed: Vec<_> = BoxIter::new(Cursor::new(boxed))
            .collect::<Result<Vec<_>, _>>()
            .expect("parse ok");
        assert_eq!(parsed.len(), 1);
        assert!(parsed[0].1.is_empty());
    }
}
