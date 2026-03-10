//! Container atom/box parsing.
//!
//! Implements parsing of ISO Base Media File Format (ISOBMFF) atoms (also
//! called "boxes"), which are the fundamental building blocks of MP4, MOV,
//! and related container formats.
//!
//! Each atom has:
//! - 4-byte size field (includes the header itself)
//! - 4-byte `FourCC` type tag
//! - Optional 8-byte extended size (when `size == 1`)
//! - Payload data

#![allow(dead_code)]

/// A four-character code identifying an atom type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FourCC(pub [u8; 4]);

impl FourCC {
    /// Create a `FourCC` from a byte slice (must be exactly 4 bytes).
    ///
    /// # Panics
    ///
    /// Panics if `bytes.len() != 4`.
    #[must_use]
    pub fn from_bytes(bytes: &[u8]) -> Self {
        assert_eq!(bytes.len(), 4, "FourCC must be 4 bytes");
        Self([bytes[0], bytes[1], bytes[2], bytes[3]])
    }

    /// Create a `FourCC` from a 4-byte ASCII literal.
    #[must_use]
    pub const fn from_ascii(s: &[u8; 4]) -> Self {
        Self(*s)
    }

    /// Returns the ASCII string representation, replacing non-printable bytes
    /// with `'?'`.
    #[must_use]
    pub fn as_str(&self) -> String {
        self.0
            .iter()
            .map(|&b| {
                if b.is_ascii_graphic() || b == b' ' {
                    b as char
                } else {
                    '?'
                }
            })
            .collect()
    }
}

impl std::fmt::Display for FourCC {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Parsed atom header.
#[derive(Debug, Clone, PartialEq)]
pub struct AtomHeader {
    /// Total size of the atom in bytes (includes header).
    pub size: u64,
    /// `FourCC` atom type.
    pub atom_type: FourCC,
    /// Number of bytes consumed by the header itself.
    pub header_size: usize,
}

impl AtomHeader {
    /// Parse an atom header from a byte buffer.
    ///
    /// Returns `Err` if the buffer is too short or the size is invalid.
    ///
    /// # Errors
    ///
    /// Returns `AtomError::BufferTooShort` or `AtomError::InvalidSize`.
    pub fn parse(buf: &[u8]) -> Result<Self, AtomError> {
        if buf.len() < 8 {
            return Err(AtomError::BufferTooShort {
                needed: 8,
                available: buf.len(),
            });
        }

        let raw_size = u64::from(u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]));
        let atom_type = FourCC::from_bytes(&buf[4..8]);

        let (size, header_size) = if raw_size == 1 {
            // Extended 64-bit size
            if buf.len() < 16 {
                return Err(AtomError::BufferTooShort {
                    needed: 16,
                    available: buf.len(),
                });
            }
            let ext = u64::from_be_bytes([
                buf[8], buf[9], buf[10], buf[11], buf[12], buf[13], buf[14], buf[15],
            ]);
            if ext < 16 {
                return Err(AtomError::InvalidSize(ext));
            }
            (ext, 16)
        } else if raw_size == 0 {
            // "Extends to end of file" - we return u64::MAX as sentinel
            (u64::MAX, 8)
        } else {
            if raw_size < 8 {
                return Err(AtomError::InvalidSize(raw_size));
            }
            (raw_size, 8)
        };

        Ok(Self {
            size,
            atom_type,
            header_size,
        })
    }

    /// Returns the payload size (size - `header_size`), or `None` if the atom
    /// uses the "end-of-file" sentinel.
    #[must_use]
    pub fn payload_size(&self) -> Option<u64> {
        if self.size == u64::MAX {
            None
        } else {
            Some(self.size - self.header_size as u64)
        }
    }
}

/// Errors that can occur while parsing atoms.
#[derive(Debug, Clone, PartialEq)]
pub enum AtomError {
    /// The input buffer is too short to contain a valid atom header.
    BufferTooShort { needed: usize, available: usize },
    /// The atom's size field is inconsistent.
    InvalidSize(u64),
    /// Nested atom would exceed the parent atom's boundary.
    NestingOverflow,
    /// The atom type is not recognised in the current context.
    UnknownAtomType(FourCC),
}

impl std::fmt::Display for AtomError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BufferTooShort { needed, available } => {
                write!(f, "buffer too short: need {needed}, have {available}")
            }
            Self::InvalidSize(s) => write!(f, "invalid atom size: {s}"),
            Self::NestingOverflow => write!(f, "nested atom overflows parent boundary"),
            Self::UnknownAtomType(cc) => write!(f, "unknown atom type: {cc}"),
        }
    }
}

/// A parsed atom with its header and raw payload.
#[derive(Debug, Clone)]
pub struct Atom {
    /// Parsed header.
    pub header: AtomHeader,
    /// Raw payload bytes (without the header).
    pub payload: Vec<u8>,
}

impl Atom {
    /// Parse one atom from the start of `buf`.
    ///
    /// # Errors
    ///
    /// Propagates [`AtomError`] from header parsing or if the buffer is too
    /// short to hold the declared payload.
    pub fn parse(buf: &[u8]) -> Result<Self, AtomError> {
        let header = AtomHeader::parse(buf)?;
        #[allow(clippy::cast_possible_truncation)]
        let payload = if let Some(payload_size) = header.payload_size() {
            let end = header.header_size + payload_size as usize;
            if buf.len() < end {
                return Err(AtomError::BufferTooShort {
                    needed: end,
                    available: buf.len(),
                });
            }
            buf[header.header_size..end].to_vec()
        } else {
            // Extend-to-EOF: take everything after the header
            buf[header.header_size..].to_vec()
        };
        Ok(Self { header, payload })
    }

    /// Parse a sequence of atoms from a buffer (until the buffer is exhausted).
    ///
    /// Stops on the first error but returns all successfully parsed atoms.
    #[must_use]
    pub fn parse_all(mut buf: &[u8]) -> Vec<Atom> {
        let mut atoms = Vec::new();
        while !buf.is_empty() {
            match Atom::parse(buf) {
                Ok(atom) => {
                    #[allow(clippy::cast_possible_truncation)]
                    let consumed = if atom.header.size == u64::MAX {
                        buf.len() // EOF atom consumes everything
                    } else {
                        atom.header.size as usize
                    };
                    atoms.push(atom);
                    if consumed >= buf.len() {
                        break;
                    }
                    buf = &buf[consumed..];
                }
                Err(_) => break,
            }
        }
        atoms
    }

    /// Validate that the atom's declared size matches the provided buffer
    /// length for non-EOF atoms.
    #[must_use]
    pub fn is_size_valid(&self, buf_len: usize) -> bool {
        if self.header.size == u64::MAX {
            return true; // EOF atom is always considered valid
        }
        #[allow(clippy::cast_possible_truncation)]
        {
            self.header.size as usize <= buf_len
        }
    }
}

/// Utility: encode a 4-byte big-endian size + 4-byte type into a header.
#[must_use]
pub fn encode_atom_header(size: u32, atom_type: &[u8; 4]) -> [u8; 8] {
    let sb = size.to_be_bytes();
    [
        sb[0],
        sb[1],
        sb[2],
        sb[3],
        atom_type[0],
        atom_type[1],
        atom_type[2],
        atom_type[3],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_atom(size: u32, fourcc: &[u8; 4], payload: &[u8]) -> Vec<u8> {
        let mut v = encode_atom_header(size, fourcc).to_vec();
        v.extend_from_slice(payload);
        v
    }

    #[test]
    fn test_fourcc_from_bytes() {
        let cc = FourCC::from_bytes(b"ftyp");
        assert_eq!(cc.0, *b"ftyp");
    }

    #[test]
    fn test_fourcc_from_ascii() {
        let cc = FourCC::from_ascii(b"moov");
        assert_eq!(cc.as_str(), "moov");
    }

    #[test]
    fn test_fourcc_display() {
        let cc = FourCC::from_ascii(b"mdat");
        assert_eq!(format!("{cc}"), "mdat");
    }

    #[test]
    fn test_atom_header_parse_basic() {
        let buf = build_atom(16, b"ftyp", &[0u8; 8]);
        let hdr = AtomHeader::parse(&buf).expect("operation should succeed");
        assert_eq!(hdr.size, 16);
        assert_eq!(hdr.atom_type, FourCC::from_ascii(b"ftyp"));
        assert_eq!(hdr.header_size, 8);
        assert_eq!(hdr.payload_size(), Some(8));
    }

    #[test]
    fn test_atom_header_parse_too_short() {
        let err = AtomHeader::parse(&[0u8; 4]).unwrap_err();
        assert!(matches!(err, AtomError::BufferTooShort { .. }));
    }

    #[test]
    fn test_atom_header_invalid_size() {
        // size=4 is less than the minimum of 8
        let buf = build_atom(4, b"mdat", &[]);
        let err = AtomHeader::parse(&buf).unwrap_err();
        assert!(matches!(err, AtomError::InvalidSize(4)));
    }

    #[test]
    fn test_atom_header_eof_sentinel() {
        let buf = build_atom(0, b"mdat", &[1, 2, 3]);
        let hdr = AtomHeader::parse(&buf).expect("operation should succeed");
        assert_eq!(hdr.size, u64::MAX);
        assert!(hdr.payload_size().is_none());
    }

    #[test]
    fn test_atom_parse_ok() {
        let buf = build_atom(12, b"moov", &[0xDE, 0xAD, 0xBE, 0xEF]);
        let atom = Atom::parse(&buf).expect("operation should succeed");
        assert_eq!(atom.payload.len(), 4);
        assert_eq!(atom.payload[0], 0xDE);
    }

    #[test]
    fn test_atom_parse_payload_too_short() {
        // Declare size=20 but only provide 12 bytes total
        let mut buf = encode_atom_header(20, b"mdat").to_vec();
        buf.extend_from_slice(&[0u8; 4]); // only 4 payload bytes, but 12 needed
        let err = Atom::parse(&buf).unwrap_err();
        assert!(matches!(err, AtomError::BufferTooShort { .. }));
    }

    #[test]
    fn test_atom_parse_all() {
        let mut buf = build_atom(12, b"ftyp", &[1, 2, 3, 4]);
        buf.extend(build_atom(10, b"mdat", &[5, 6]));
        let atoms = Atom::parse_all(&buf);
        assert_eq!(atoms.len(), 2);
        assert_eq!(atoms[0].header.atom_type, FourCC::from_ascii(b"ftyp"));
        assert_eq!(atoms[1].header.atom_type, FourCC::from_ascii(b"mdat"));
    }

    #[test]
    fn test_atom_is_size_valid() {
        let buf = build_atom(12, b"free", &[0u8; 4]);
        let atom = Atom::parse(&buf).expect("operation should succeed");
        assert!(atom.is_size_valid(buf.len()));
        assert!(!atom.is_size_valid(4)); // smaller than declared size
    }

    #[test]
    fn test_encode_atom_header() {
        let hdr = encode_atom_header(24, b"moof");
        assert_eq!(&hdr[0..4], &24u32.to_be_bytes());
        assert_eq!(&hdr[4..8], b"moof");
    }

    #[test]
    fn test_atom_error_display() {
        let e = AtomError::BufferTooShort {
            needed: 8,
            available: 4,
        };
        let s = format!("{e}");
        assert!(s.contains("8"));
        assert!(s.contains("4"));

        let e2 = AtomError::InvalidSize(2);
        let s2 = format!("{e2}");
        assert!(s2.contains("2"));
    }

    #[test]
    fn test_parse_all_empty() {
        let atoms = Atom::parse_all(&[]);
        assert!(atoms.is_empty());
    }

    #[test]
    fn test_fourcc_non_printable() {
        let cc = FourCC([0x00, 0x01, 0x41, 0x42]);
        let s = cc.as_str();
        // First two are non-printable -> '?', then 'AB'
        assert_eq!(&s[2..], "AB");
    }
}
