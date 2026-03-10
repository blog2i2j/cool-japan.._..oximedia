#![allow(dead_code)]
//! Seekable buffer abstraction for random-access I/O operations.

/// Represents a seek position within a buffer or file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeekPosition {
    /// Seek to an absolute byte offset from the start.
    FromStart(u64),
    /// Seek to a byte offset relative to the current position (may be negative).
    FromCurrent(i64),
    /// Seek to a byte offset relative to the end (non-positive values seek back from end).
    FromEnd(i64),
}

impl SeekPosition {
    /// Convert this seek position to an absolute byte offset given the file size and
    /// the current read position.
    ///
    /// Returns `None` if the resulting offset would be out of bounds.
    #[must_use]
    #[allow(
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation
    )]
    pub fn to_offset(self, current: u64, file_size: u64) -> Option<u64> {
        match self {
            SeekPosition::FromStart(off) => {
                if off <= file_size {
                    Some(off)
                } else {
                    None
                }
            }
            SeekPosition::FromCurrent(delta) => {
                let result = current as i64 + delta;
                if result >= 0 && result as u64 <= file_size {
                    Some(result as u64)
                } else {
                    None
                }
            }
            SeekPosition::FromEnd(delta) => {
                let result = file_size as i64 + delta;
                if result >= 0 && result as u64 <= file_size {
                    Some(result as u64)
                } else {
                    None
                }
            }
        }
    }
}

/// An in-memory buffer that supports random-access seeking.
#[derive(Debug, Clone)]
pub struct SeekableBuffer {
    data: Vec<u8>,
    position: u64,
}

impl SeekableBuffer {
    /// Create a new `SeekableBuffer` wrapping the given byte vector.
    #[must_use]
    pub fn new(data: Vec<u8>) -> Self {
        Self { data, position: 0 }
    }

    /// Create an empty `SeekableBuffer`.
    #[must_use]
    pub fn empty() -> Self {
        Self::new(Vec::new())
    }

    /// Seek to the given position.
    ///
    /// Returns `Ok(new_position)` on success, or an error string if out of bounds.
    ///
    /// # Errors
    ///
    /// Returns `Err(String)` if the seek position is out of bounds.
    pub fn seek(&mut self, pos: SeekPosition) -> Result<u64, String> {
        let file_size = self.data.len() as u64;
        match pos.to_offset(self.position, file_size) {
            Some(off) => {
                self.position = off;
                Ok(off)
            }
            None => Err(format!(
                "seek position {pos:?} is out of bounds (file_size={file_size})"
            )),
        }
    }

    /// Return the current read position (byte offset from start).
    #[must_use]
    pub fn position(&self) -> u64 {
        self.position
    }

    /// Return the number of bytes remaining from the current position to the end.
    #[must_use]
    pub fn bytes_remaining(&self) -> u64 {
        let len = self.data.len() as u64;
        len.saturating_sub(self.position)
    }

    /// Read up to `buf.len()` bytes from the current position, advancing the position.
    ///
    /// Returns the number of bytes actually read.
    #[allow(clippy::cast_possible_truncation)]
    pub fn read_bytes(&mut self, buf: &mut [u8]) -> usize {
        let available = self.bytes_remaining() as usize;
        let to_read = buf.len().min(available);
        let start = self.position as usize;
        buf[..to_read].copy_from_slice(&self.data[start..start + to_read]);
        self.position += to_read as u64;
        to_read
    }

    /// Return the total size of the underlying buffer.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Return `true` if the underlying buffer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_buf() -> SeekableBuffer {
        SeekableBuffer::new((0u8..=9u8).collect())
    }

    #[test]
    fn test_initial_position() {
        let buf = make_buf();
        assert_eq!(buf.position(), 0);
    }

    #[test]
    fn test_bytes_remaining_at_start() {
        let buf = make_buf();
        assert_eq!(buf.bytes_remaining(), 10);
    }

    #[test]
    fn test_seek_from_start() {
        let mut buf = make_buf();
        let new_pos = buf
            .seek(SeekPosition::FromStart(5))
            .expect("seek should succeed");
        assert_eq!(new_pos, 5);
        assert_eq!(buf.position(), 5);
        assert_eq!(buf.bytes_remaining(), 5);
    }

    #[test]
    fn test_seek_from_current_forward() {
        let mut buf = make_buf();
        buf.seek(SeekPosition::FromStart(3))
            .expect("seek should succeed");
        buf.seek(SeekPosition::FromCurrent(4))
            .expect("seek should succeed");
        assert_eq!(buf.position(), 7);
    }

    #[test]
    fn test_seek_from_current_backward() {
        let mut buf = make_buf();
        buf.seek(SeekPosition::FromStart(6))
            .expect("seek should succeed");
        buf.seek(SeekPosition::FromCurrent(-3))
            .expect("seek should succeed");
        assert_eq!(buf.position(), 3);
    }

    #[test]
    fn test_seek_from_end() {
        let mut buf = make_buf();
        buf.seek(SeekPosition::FromEnd(-2))
            .expect("seek should succeed");
        assert_eq!(buf.position(), 8);
    }

    #[test]
    fn test_seek_to_end_boundary() {
        let mut buf = make_buf();
        let pos = buf
            .seek(SeekPosition::FromStart(10))
            .expect("seek should succeed");
        assert_eq!(pos, 10);
        assert_eq!(buf.bytes_remaining(), 0);
    }

    #[test]
    fn test_seek_out_of_bounds_returns_err() {
        let mut buf = make_buf();
        assert!(buf.seek(SeekPosition::FromStart(99)).is_err());
    }

    #[test]
    fn test_seek_from_end_out_of_bounds() {
        let mut buf = make_buf();
        assert!(buf.seek(SeekPosition::FromEnd(-99)).is_err());
    }

    #[test]
    fn test_read_bytes_full() {
        let mut buf = make_buf();
        let mut out = vec![0u8; 10];
        let n = buf.read_bytes(&mut out);
        assert_eq!(n, 10);
        assert_eq!(out, (0u8..=9u8).collect::<Vec<_>>());
    }

    #[test]
    fn test_read_bytes_partial() {
        let mut buf = make_buf();
        let mut out = vec![0u8; 3];
        buf.read_bytes(&mut out);
        assert_eq!(buf.position(), 3);
        let mut out2 = vec![0u8; 3];
        let n = buf.read_bytes(&mut out2);
        assert_eq!(n, 3);
        assert_eq!(out2, vec![3, 4, 5]);
    }

    #[test]
    fn test_read_bytes_past_end() {
        let mut buf = make_buf();
        buf.seek(SeekPosition::FromStart(8))
            .expect("seek should succeed");
        let mut out = vec![0u8; 5];
        let n = buf.read_bytes(&mut out);
        assert_eq!(n, 2);
        assert_eq!(&out[..2], &[8, 9]);
    }

    #[test]
    fn test_empty_buffer() {
        let buf = SeekableBuffer::empty();
        assert!(buf.is_empty());
        assert_eq!(buf.bytes_remaining(), 0);
        assert_eq!(buf.len(), 0);
    }

    #[test]
    fn test_seek_position_from_start_to_offset() {
        let pos = SeekPosition::FromStart(5);
        assert_eq!(pos.to_offset(0, 10), Some(5));
        assert_eq!(pos.to_offset(0, 4), None);
    }

    #[test]
    fn test_seek_position_from_end_to_offset() {
        let pos = SeekPosition::FromEnd(0);
        assert_eq!(pos.to_offset(0, 10), Some(10));
        let pos2 = SeekPosition::FromEnd(-3);
        assert_eq!(pos2.to_offset(0, 10), Some(7));
    }
}
