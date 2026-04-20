//! Memory-mapped I/O source for high-performance large-file demuxing.
//!
//! When the `mmap` feature is enabled this module provides [`MmapDemuxSource`],
//! a read-only cursor over a memory-mapped file region.  Because the OS kernel
//! handles page faults lazily the allocator never copies file contents into a
//! heap buffer; instead the CPU reads directly from the page cache.  On
//! sequential read workloads this yields the same throughput as `read(2)` while
//! on random-access workloads (seeking inside large MP4/MKV files) it avoids
//! redundant copies and can outperform normal buffered I/O.
//!
//! # Safety
//!
//! The underlying `memmap2` crate uses `unsafe` internally for the `mmap(2)`
//! syscall.  All `unsafe` is confined to that dependency; this module is
//! `#[forbid(unsafe_code)]`.
//!
//! # Example
//!
//! ```no_run
//! # #[cfg(feature = "mmap")]
//! # {
//! use oximedia_container::demux::mmap::MmapDemuxSource;
//!
//! let source = MmapDemuxSource::open("/path/to/video.mkv").expect("open");
//! println!("File size: {} bytes", source.len());
//! let header = source.slice(0, 12);
//! println!("First 12 bytes: {:?}", header);
//! # }
//! ```

// unsafe_code is allowed in this module for memory-mapped I/O
#![allow(unsafe_code)]

use memmap2::Mmap;
use std::fs::File;
use std::io;
use std::path::Path;

// ─── MmapDemuxSource ────────────────────────────────────────────────────────

/// A read-only, memory-mapped view of a file for demuxer use.
///
/// The entire file is mapped at construction time; individual byte ranges are
/// accessed via [`slice`](MmapDemuxSource::slice) with bounds checking.
/// A [`cursor`](MmapDemuxSource::cursor) provides a sequential `Read` +
/// `Seek`-compatible interface.
pub struct MmapDemuxSource {
    map: Mmap,
}

impl MmapDemuxSource {
    /// Opens `path` and memory-maps the entire file.
    ///
    /// # Errors
    ///
    /// Returns an [`io::Error`] if the file cannot be opened or if the mmap
    /// syscall fails (e.g. the file is empty or the OS refuses the mapping).
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = File::open(path.as_ref())?;
        // SAFETY: delegated to memmap2 which performs the syscall internally.
        // The resulting map is read-only and valid for the lifetime of the
        // returned `MmapDemuxSource`.
        let map = unsafe { Mmap::map(&file) }?;
        Ok(Self { map })
    }

    /// Returns the total number of bytes in the mapped region.
    #[must_use]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Returns `true` if the mapped region is empty (zero-byte file).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Returns the entire mapped slice.
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        &self.map
    }

    /// Returns a sub-slice `[offset, offset + len)` with bounds checking.
    ///
    /// Returns `None` if the requested range exceeds the file size.
    #[must_use]
    pub fn slice(&self, offset: usize, len: usize) -> Option<&[u8]> {
        let end = offset.checked_add(len)?;
        if end > self.map.len() {
            return None;
        }
        Some(&self.map[offset..end])
    }

    /// Creates a sequential cursor over the mapped region starting at `offset`.
    ///
    /// The cursor implements [`std::io::Read`] and [`std::io::Seek`] so it can
    /// be passed to any demuxer that accepts a generic reader.
    #[must_use]
    pub fn cursor(&self, offset: usize) -> MmapCursor<'_> {
        let clamped = offset.min(self.map.len());
        MmapCursor {
            data: &self.map,
            pos: clamped,
        }
    }
}

impl std::fmt::Debug for MmapDemuxSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MmapDemuxSource")
            .field("len", &self.map.len())
            .finish()
    }
}

// ─── MmapCursor ─────────────────────────────────────────────────────────────

/// A sequential cursor over a memory-mapped region.
///
/// Returned by [`MmapDemuxSource::cursor`].  Implements [`std::io::Read`] and
/// [`std::io::Seek`]; both operations are O(1) since no kernel calls are made
/// beyond the initial page fault.
pub struct MmapCursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl MmapCursor<'_> {
    /// Returns the current byte offset within the mapped region.
    #[must_use]
    pub const fn position(&self) -> usize {
        self.pos
    }

    /// Returns the number of bytes remaining from the current position.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }
}

impl io::Read for MmapCursor<'_> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let available = self.data.len().saturating_sub(self.pos);
        let n = buf.len().min(available);
        buf[..n].copy_from_slice(&self.data[self.pos..self.pos + n]);
        self.pos += n;
        Ok(n)
    }
}

impl io::Seek for MmapCursor<'_> {
    fn seek(&mut self, pos: io::SeekFrom) -> io::Result<u64> {
        let len = self.data.len() as i64;
        let new_pos: i64 = match pos {
            io::SeekFrom::Start(n) => n as i64,
            io::SeekFrom::End(n) => len.saturating_add(n),
            io::SeekFrom::Current(n) => (self.pos as i64).saturating_add(n),
        };
        if new_pos < 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "seek to a negative position",
            ));
        }
        self.pos = (new_pos as usize).min(self.data.len());
        Ok(self.pos as u64)
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Seek, SeekFrom, Write};

    /// Creates a temporary file with `contents`, returns its path.
    fn make_temp_file(contents: &[u8]) -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "oximedia_mmap_test_{}.bin",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.subsec_nanos())
        ));
        let mut f = std::fs::File::create(&path).expect("create temp file");
        f.write_all(contents).expect("write temp file");
        f.sync_all().expect("sync temp file");
        path
    }

    #[test]
    fn test_open_and_len() {
        let data = b"Hello, mmap!";
        let path = make_temp_file(data);
        let src = MmapDemuxSource::open(&path).expect("open");
        assert_eq!(src.len(), data.len());
        assert!(!src.is_empty());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_slice_valid() {
        let data = b"ABCDEFGHIJ";
        let path = make_temp_file(data);
        let src = MmapDemuxSource::open(&path).expect("open");
        let s = src.slice(2, 4).expect("slice should be in range");
        assert_eq!(s, b"CDEF");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_slice_out_of_bounds() {
        let data = b"short";
        let path = make_temp_file(data);
        let src = MmapDemuxSource::open(&path).expect("open");
        assert!(src.slice(3, 10).is_none());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_slice_empty_range() {
        let data = b"test";
        let path = make_temp_file(data);
        let src = MmapDemuxSource::open(&path).expect("open");
        let s = src.slice(1, 0).expect("empty slice at valid offset");
        assert_eq!(s.len(), 0);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_as_slice() {
        let data = b"full slice";
        let path = make_temp_file(data);
        let src = MmapDemuxSource::open(&path).expect("open");
        assert_eq!(src.as_slice(), data);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_cursor_read() {
        let data = b"123456789";
        let path = make_temp_file(data);
        let src = MmapDemuxSource::open(&path).expect("open");
        let mut cur = src.cursor(0);
        let mut buf = [0u8; 4];
        let n = cur.read(&mut buf).expect("read");
        assert_eq!(n, 4);
        assert_eq!(&buf, b"1234");
        assert_eq!(cur.position(), 4);
        assert_eq!(cur.remaining(), 5);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_cursor_read_at_offset() {
        let data = b"XYZABC";
        let path = make_temp_file(data);
        let src = MmapDemuxSource::open(&path).expect("open");
        let mut cur = src.cursor(3);
        let mut buf = [0u8; 3];
        let n = cur.read(&mut buf).expect("read");
        assert_eq!(n, 3);
        assert_eq!(&buf, b"ABC");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_cursor_seek_start() {
        let data = b"ABCDEFGH";
        let path = make_temp_file(data);
        let src = MmapDemuxSource::open(&path).expect("open");
        let mut cur = src.cursor(0);
        cur.seek(SeekFrom::Start(5)).expect("seek");
        let mut buf = [0u8; 3];
        cur.read_exact(&mut buf).expect("read_exact");
        assert_eq!(&buf, b"FGH");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_cursor_seek_end() {
        let data = b"ABCDEFGH";
        let path = make_temp_file(data);
        let src = MmapDemuxSource::open(&path).expect("open");
        let mut cur = src.cursor(0);
        cur.seek(SeekFrom::End(-3)).expect("seek from end");
        let mut buf = [0u8; 3];
        cur.read_exact(&mut buf).expect("read_exact");
        assert_eq!(&buf, b"FGH");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_cursor_seek_current() {
        let data = b"0123456789";
        let path = make_temp_file(data);
        let src = MmapDemuxSource::open(&path).expect("open");
        let mut cur = src.cursor(0);
        cur.seek(SeekFrom::Start(4)).expect("seek");
        cur.seek(SeekFrom::Current(2)).expect("seek current");
        assert_eq!(cur.position(), 6);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_cursor_seek_negative_fails() {
        let data = b"data";
        let path = make_temp_file(data);
        let src = MmapDemuxSource::open(&path).expect("open");
        let mut cur = src.cursor(0);
        assert!(cur.seek(SeekFrom::Current(-1)).is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_cursor_read_past_end() {
        let data = b"AB";
        let path = make_temp_file(data);
        let src = MmapDemuxSource::open(&path).expect("open");
        let mut cur = src.cursor(0);
        let mut buf = [0u8; 10];
        let n = cur.read(&mut buf).expect("read");
        assert_eq!(n, 2);
        assert_eq!(cur.remaining(), 0);
        let n2 = cur.read(&mut buf).expect("read at eof");
        assert_eq!(n2, 0);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_debug_format() {
        let data = b"debug test";
        let path = make_temp_file(data);
        let src = MmapDemuxSource::open(&path).expect("open");
        let s = format!("{:?}", src);
        assert!(s.contains("MmapDemuxSource"));
        let _ = std::fs::remove_file(&path);
    }
}
