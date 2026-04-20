//! MP4 atom (box) primitives.
//!
//! This module provides low-level reading utilities for parsing MP4 atoms.
//! All multi-byte integers in MP4 are big-endian.

use oximedia_core::{OxiError, OxiResult};

/// MP4 atom reader for parsing binary box data.
///
/// Provides methods for reading various data types from a byte slice,
/// including integers, fixed-point numbers, and type codes.
///
/// # Example
///
/// ```
/// use oximedia_container::demux::mp4::Mp4Atom;
///
/// let data = [0x00, 0x00, 0x00, 0x14, b'f', b't', b'y', b'p'];
/// let mut atom = Mp4Atom::new(&data);
/// assert_eq!(atom.read_u32().expect("valid u32"), 20);
/// assert_eq!(atom.read_type().expect("valid type"), "ftyp");
/// ```
pub struct Mp4Atom<'a> {
    /// The underlying data buffer.
    data: &'a [u8],
    /// Current read position.
    position: usize,
}

impl<'a> Mp4Atom<'a> {
    /// Creates a new atom reader from a byte slice.
    ///
    /// # Arguments
    ///
    /// * `data` - The byte slice to read from
    #[must_use]
    pub const fn new(data: &'a [u8]) -> Self {
        Self { data, position: 0 }
    }

    /// Reads a single unsigned byte.
    ///
    /// # Errors
    ///
    /// Returns `UnexpectedEof` if there are no bytes remaining.
    pub fn read_u8(&mut self) -> OxiResult<u8> {
        if self.position >= self.data.len() {
            return Err(OxiError::UnexpectedEof);
        }
        let value = self.data[self.position];
        self.position += 1;
        Ok(value)
    }

    /// Reads an unsigned 16-bit integer (big-endian).
    ///
    /// # Errors
    ///
    /// Returns `UnexpectedEof` if fewer than 2 bytes remain.
    pub fn read_u16(&mut self) -> OxiResult<u16> {
        if self.position + 2 > self.data.len() {
            return Err(OxiError::UnexpectedEof);
        }
        let value = u16::from_be_bytes([self.data[self.position], self.data[self.position + 1]]);
        self.position += 2;
        Ok(value)
    }

    /// Reads an unsigned 32-bit integer (big-endian).
    ///
    /// # Errors
    ///
    /// Returns `UnexpectedEof` if fewer than 4 bytes remain.
    pub fn read_u32(&mut self) -> OxiResult<u32> {
        if self.position + 4 > self.data.len() {
            return Err(OxiError::UnexpectedEof);
        }
        let value = u32::from_be_bytes([
            self.data[self.position],
            self.data[self.position + 1],
            self.data[self.position + 2],
            self.data[self.position + 3],
        ]);
        self.position += 4;
        Ok(value)
    }

    /// Reads an unsigned 64-bit integer (big-endian).
    ///
    /// # Errors
    ///
    /// Returns `UnexpectedEof` if fewer than 8 bytes remain.
    pub fn read_u64(&mut self) -> OxiResult<u64> {
        if self.position + 8 > self.data.len() {
            return Err(OxiError::UnexpectedEof);
        }
        let value = u64::from_be_bytes([
            self.data[self.position],
            self.data[self.position + 1],
            self.data[self.position + 2],
            self.data[self.position + 3],
            self.data[self.position + 4],
            self.data[self.position + 5],
            self.data[self.position + 6],
            self.data[self.position + 7],
        ]);
        self.position += 8;
        Ok(value)
    }

    /// Reads a signed 32-bit integer (big-endian).
    ///
    /// # Errors
    ///
    /// Returns `UnexpectedEof` if fewer than 4 bytes remain.
    #[allow(clippy::cast_possible_wrap)]
    pub fn read_i32(&mut self) -> OxiResult<i32> {
        Ok(self.read_u32()? as i32)
    }

    /// Reads a signed 64-bit integer (big-endian).
    ///
    /// # Errors
    ///
    /// Returns `UnexpectedEof` if fewer than 8 bytes remain.
    #[allow(clippy::cast_possible_wrap)]
    pub fn read_i64(&mut self) -> OxiResult<i64> {
        Ok(self.read_u64()? as i64)
    }

    /// Reads a 16.16 fixed-point number as `f64`.
    ///
    /// The value is interpreted as a signed 16-bit integer part
    /// and an unsigned 16-bit fractional part.
    ///
    /// # Errors
    ///
    /// Returns `UnexpectedEof` if fewer than 4 bytes remain.
    pub fn read_fixed_16_16(&mut self) -> OxiResult<f64> {
        let raw = self.read_u32()?;
        Ok(f64::from(raw) / 65536.0)
    }

    /// Reads an 8.8 fixed-point number as `f64`.
    ///
    /// The value is interpreted as an unsigned 8-bit integer part
    /// and an unsigned 8-bit fractional part.
    ///
    /// # Errors
    ///
    /// Returns `UnexpectedEof` if fewer than 2 bytes remain.
    pub fn read_fixed_8_8(&mut self) -> OxiResult<f64> {
        let raw = self.read_u16()?;
        Ok(f64::from(raw) / 256.0)
    }

    /// Reads a 4-byte type code as a string.
    ///
    /// Non-ASCII bytes are replaced with the Unicode replacement character.
    ///
    /// # Errors
    ///
    /// Returns `UnexpectedEof` if fewer than 4 bytes remain.
    pub fn read_type(&mut self) -> OxiResult<String> {
        if self.position + 4 > self.data.len() {
            return Err(OxiError::UnexpectedEof);
        }
        let bytes = &self.data[self.position..self.position + 4];
        self.position += 4;
        Ok(String::from_utf8_lossy(bytes).into_owned())
    }

    /// Reads a specified number of bytes as a slice.
    ///
    /// # Arguments
    ///
    /// * `len` - The number of bytes to read
    ///
    /// # Errors
    ///
    /// Returns `UnexpectedEof` if fewer than `len` bytes remain.
    pub fn read_bytes(&mut self, len: usize) -> OxiResult<&'a [u8]> {
        if self.position + len > self.data.len() {
            return Err(OxiError::UnexpectedEof);
        }
        let bytes = &self.data[self.position..self.position + len];
        self.position += len;
        Ok(bytes)
    }

    /// Skips a specified number of bytes.
    ///
    /// # Arguments
    ///
    /// * `len` - The number of bytes to skip
    ///
    /// # Errors
    ///
    /// Returns `UnexpectedEof` if fewer than `len` bytes remain.
    pub fn skip(&mut self, len: usize) -> OxiResult<()> {
        if self.position + len > self.data.len() {
            return Err(OxiError::UnexpectedEof);
        }
        self.position += len;
        Ok(())
    }

    /// Returns the remaining unread bytes as a slice.
    #[must_use]
    pub fn remaining(&self) -> &'a [u8] {
        &self.data[self.position..]
    }

    /// Returns the current read position.
    #[must_use]
    pub const fn position(&self) -> usize {
        self.position
    }

    /// Returns `true` if all bytes have been read.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.position >= self.data.len()
    }

    /// Returns the total length of the underlying data.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_u8() {
        let data = [0x42];
        let mut atom = Mp4Atom::new(&data);
        assert_eq!(atom.read_u8().expect("read_u8 should succeed"), 0x42);
        assert!(atom.is_empty());
    }

    #[test]
    fn test_read_u16() {
        let data = [0x01, 0x02];
        let mut atom = Mp4Atom::new(&data);
        assert_eq!(atom.read_u16().expect("read_u16 should succeed"), 0x0102);
    }

    #[test]
    fn test_read_u32() {
        let data = [0x00, 0x00, 0x00, 0x14];
        let mut atom = Mp4Atom::new(&data);
        assert_eq!(atom.read_u32().expect("read_u32 should succeed"), 20);
    }

    #[test]
    fn test_read_u64() {
        let data = [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x64];
        let mut atom = Mp4Atom::new(&data);
        assert_eq!(atom.read_u64().expect("read_u64 should succeed"), 100);
    }

    #[test]
    fn test_read_i32() {
        let data = [0xFF, 0xFF, 0xFF, 0xFF];
        let mut atom = Mp4Atom::new(&data);
        assert_eq!(atom.read_i32().expect("operation should succeed"), -1);
    }

    #[test]
    fn test_read_type() {
        let data = b"ftyp";
        let mut atom = Mp4Atom::new(data);
        assert_eq!(atom.read_type().expect("operation should succeed"), "ftyp");
    }

    #[test]
    fn test_read_fixed_16_16() {
        // 1.0 in 16.16 fixed-point
        let data = [0x00, 0x01, 0x00, 0x00];
        let mut atom = Mp4Atom::new(&data);
        let value = atom.read_fixed_16_16().expect("operation should succeed");
        assert!((value - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_read_fixed_8_8() {
        // 1.0 in 8.8 fixed-point
        let data = [0x01, 0x00];
        let mut atom = Mp4Atom::new(&data);
        let value = atom.read_fixed_8_8().expect("operation should succeed");
        assert!((value - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_read_bytes() {
        let data = [0x01, 0x02, 0x03, 0x04];
        let mut atom = Mp4Atom::new(&data);
        let bytes = atom.read_bytes(2).expect("read_bytes should succeed");
        assert_eq!(bytes, &[0x01, 0x02]);
        assert_eq!(atom.position(), 2);
    }

    #[test]
    fn test_skip() {
        let data = [0x01, 0x02, 0x03, 0x04];
        let mut atom = Mp4Atom::new(&data);
        atom.skip(2).expect("operation should succeed");
        assert_eq!(atom.position(), 2);
        assert_eq!(atom.remaining(), &[0x03, 0x04]);
    }

    #[test]
    fn test_eof_error() {
        let data = [0x01];
        let mut atom = Mp4Atom::new(&data);
        assert!(atom.read_u32().is_err());
    }

    #[test]
    fn test_len_and_is_empty() {
        let data = [0x01, 0x02];
        let mut atom = Mp4Atom::new(&data);
        assert_eq!(atom.len(), 2);
        assert!(!atom.is_empty());
        atom.skip(2).expect("operation should succeed");
        assert!(atom.is_empty());
    }
}
