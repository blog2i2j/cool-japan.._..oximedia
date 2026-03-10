#![allow(dead_code)]

//! Aligned I/O for direct and DMA-style transfers.
//!
//! This module provides memory-aligned buffer allocation and I/O operations
//! that ensure data is aligned to sector or page boundaries, which is
//! critical for direct I/O (`O_DIRECT` on Linux) and high-performance
//! DMA transfers in media pipelines.
//!
//! # Features
//!
//! - [`AlignedBuffer`] - Heap buffer with configurable alignment
//! - [`AlignmentSpec`] - Alignment specification (512, 4096, etc.)
//! - [`AlignedReader`] - Read adapter that produces aligned reads
//! - [`AlignedWriter`] - Write adapter that flushes in aligned blocks

use std::fmt;

/// Common alignment values for I/O operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AlignmentSpec {
    /// 512-byte alignment (traditional disk sector)
    Sector512,
    /// 4096-byte alignment (Advanced Format / page size)
    Page4k,
    /// 8192-byte alignment
    Block8k,
    /// 65536-byte alignment (64 KiB, common for large block devices)
    Large64k,
    /// Custom alignment (must be a power of two)
    Custom(usize),
}

impl AlignmentSpec {
    /// Return the alignment value in bytes.
    #[must_use]
    pub fn bytes(self) -> usize {
        match self {
            Self::Sector512 => 512,
            Self::Page4k => 4096,
            Self::Block8k => 8192,
            Self::Large64k => 65536,
            Self::Custom(v) => v,
        }
    }

    /// Check whether a given value is a valid power-of-two alignment.
    #[must_use]
    pub fn is_valid_alignment(value: usize) -> bool {
        value > 0 && value.is_power_of_two()
    }

    /// Round `size` up to the nearest multiple of this alignment.
    #[must_use]
    pub fn round_up(self, size: usize) -> usize {
        let align = self.bytes();
        if align == 0 {
            return size;
        }
        let mask = align - 1;
        (size + mask) & !mask
    }

    /// Round `size` down to the nearest multiple of this alignment.
    #[must_use]
    pub fn round_down(self, size: usize) -> usize {
        let align = self.bytes();
        if align == 0 {
            return size;
        }
        size & !(align - 1)
    }

    /// Check whether `addr` is aligned to this specification.
    #[must_use]
    pub fn is_aligned(self, addr: usize) -> bool {
        let align = self.bytes();
        if align == 0 {
            return true;
        }
        addr % align == 0
    }
}

impl fmt::Display for AlignmentSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sector512 => write!(f, "512B-sector"),
            Self::Page4k => write!(f, "4KiB-page"),
            Self::Block8k => write!(f, "8KiB-block"),
            Self::Large64k => write!(f, "64KiB-large"),
            Self::Custom(v) => write!(f, "custom-{v}B"),
        }
    }
}

/// A heap-allocated buffer whose start address is aligned.
///
/// The buffer owns its memory and ensures the data pointer satisfies the
/// requested alignment. Useful for direct I/O or DMA operations.
#[derive(Clone)]
pub struct AlignedBuffer {
    /// The raw data storage (over-allocated to allow alignment).
    storage: Vec<u8>,
    /// Offset into `storage` where aligned data begins.
    offset: usize,
    /// Usable (aligned) capacity.
    capacity: usize,
    /// Current logical length of valid data.
    len: usize,
    /// The alignment specification.
    alignment: AlignmentSpec,
}

impl AlignedBuffer {
    /// Allocate a new aligned buffer with at least `capacity` usable bytes.
    ///
    /// # Panics
    ///
    /// Panics if the alignment is not a power of two.
    #[must_use]
    pub fn new(capacity: usize, alignment: AlignmentSpec) -> Self {
        let align = alignment.bytes();
        assert!(
            AlignmentSpec::is_valid_alignment(align),
            "alignment must be a power of two, got {align}"
        );

        // Over-allocate by (align - 1) so we can find an aligned start.
        let alloc_size = capacity + align - 1;
        let storage = vec![0u8; alloc_size];
        let base = storage.as_ptr() as usize;
        let offset = alignment.round_up(base) - base;

        Self {
            storage,
            offset,
            capacity,
            len: 0,
            alignment,
        }
    }

    /// Return a slice of the valid (written) data.
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        &self.storage[self.offset..self.offset + self.len]
    }

    /// Return a mutable slice of the valid data.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        let start = self.offset;
        let end = start + self.len;
        &mut self.storage[start..end]
    }

    /// Return the full aligned capacity (number of usable bytes).
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Return the current logical length.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the buffer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Return the alignment specification.
    #[must_use]
    pub fn alignment(&self) -> AlignmentSpec {
        self.alignment
    }

    /// Write data into the buffer, returning how many bytes were written.
    ///
    /// Will not exceed the buffer capacity.
    pub fn write(&mut self, data: &[u8]) -> usize {
        let available = self.capacity - self.len;
        let to_copy = data.len().min(available);
        let start = self.offset + self.len;
        self.storage[start..start + to_copy].copy_from_slice(&data[..to_copy]);
        self.len += to_copy;
        to_copy
    }

    /// Clear the buffer (sets logical length to 0).
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Check that the internal pointer is actually aligned.
    #[must_use]
    pub fn is_properly_aligned(&self) -> bool {
        let ptr = self.storage.as_ptr() as usize + self.offset;
        self.alignment.is_aligned(ptr)
    }
}

impl fmt::Debug for AlignedBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AlignedBuffer")
            .field("alignment", &self.alignment)
            .field("capacity", &self.capacity)
            .field("len", &self.len)
            .field("offset", &self.offset)
            .field("properly_aligned", &self.is_properly_aligned())
            .finish_non_exhaustive()
    }
}

/// Reads from an inner reader in aligned blocks.
///
/// Wraps any `std::io::Read` and ensures that each underlying read request
/// is aligned in both offset and size.
pub struct AlignedReader<R> {
    /// The inner reader.
    inner: R,
    /// Internal aligned buffer.
    buffer: AlignedBuffer,
    /// Current read position within the buffer.
    buf_pos: usize,
    /// How many valid bytes are in the buffer.
    buf_len: usize,
}

impl<R: std::io::Read> AlignedReader<R> {
    /// Create a new aligned reader with a given block size and alignment.
    #[must_use]
    pub fn new(inner: R, block_size: usize, alignment: AlignmentSpec) -> Self {
        let aligned_block = alignment.round_up(block_size);
        Self {
            inner,
            buffer: AlignedBuffer::new(aligned_block, alignment),
            buf_pos: 0,
            buf_len: 0,
        }
    }

    /// Consume the reader and return the inner reader.
    #[must_use]
    pub fn into_inner(self) -> R {
        self.inner
    }

    /// Fill the internal buffer from the inner reader.
    fn fill_buffer(&mut self) -> std::io::Result<usize> {
        self.buf_pos = 0;
        self.buffer.clear();
        let cap = self.buffer.capacity();
        // Read up to capacity into a temporary slice
        let start = self.buffer.offset;
        let end = start + cap;
        let n = self.inner.read(&mut self.buffer.storage[start..end])?;
        self.buffer.len = n;
        self.buf_len = n;
        Ok(n)
    }
}

impl<R: std::io::Read> std::io::Read for AlignedReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.buf_pos >= self.buf_len {
            let filled = self.fill_buffer()?;
            if filled == 0 {
                return Ok(0);
            }
        }

        let available = self.buf_len - self.buf_pos;
        let to_copy = buf.len().min(available);
        let src_start = self.buffer.offset + self.buf_pos;
        buf[..to_copy].copy_from_slice(&self.buffer.storage[src_start..src_start + to_copy]);
        self.buf_pos += to_copy;
        Ok(to_copy)
    }
}

/// Writes data in aligned blocks to an inner writer.
///
/// Accumulates data in an internal aligned buffer and flushes full blocks
/// to the inner writer.
pub struct AlignedWriter<W> {
    /// The inner writer.
    inner: W,
    /// Internal aligned buffer.
    buffer: AlignedBuffer,
    /// The block size for flushing.
    block_size: usize,
    /// Total bytes written through this writer.
    total_written: u64,
}

impl<W: std::io::Write> AlignedWriter<W> {
    /// Create a new aligned writer.
    #[must_use]
    pub fn new(inner: W, block_size: usize, alignment: AlignmentSpec) -> Self {
        let aligned_block = alignment.round_up(block_size);
        Self {
            inner,
            buffer: AlignedBuffer::new(aligned_block, alignment),
            block_size: aligned_block,
            total_written: 0,
        }
    }

    /// Return total bytes written.
    #[must_use]
    pub fn total_written(&self) -> u64 {
        self.total_written
    }

    /// Flush internal buffer to the inner writer.
    fn flush_buffer(&mut self) -> std::io::Result<()> {
        if !self.buffer.is_empty() {
            let data = self.buffer.as_slice();
            self.inner.write_all(data)?;
            self.total_written += data.len() as u64;
            self.buffer.clear();
        }
        Ok(())
    }

    /// Consume the writer, flushing any remaining data, and return the inner writer.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if flushing fails.
    pub fn into_inner(mut self) -> std::io::Result<W> {
        self.flush_buffer()?;
        Ok(self.inner)
    }
}

impl<W: std::io::Write> std::io::Write for AlignedWriter<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let written = self.buffer.write(buf);
        if self.buffer.len() >= self.block_size {
            self.flush_buffer()?;
        }
        Ok(written)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.flush_buffer()?;
        self.inner.flush()
    }
}

/// Statistics for aligned I/O operations.
#[derive(Debug, Clone, Default)]
pub struct AlignedIoStats {
    /// Number of aligned reads performed.
    pub aligned_reads: u64,
    /// Number of aligned writes performed.
    pub aligned_writes: u64,
    /// Number of unaligned (fallback) reads.
    pub unaligned_reads: u64,
    /// Number of unaligned (fallback) writes.
    pub unaligned_writes: u64,
    /// Total bytes read.
    pub bytes_read: u64,
    /// Total bytes written.
    pub bytes_written: u64,
}

impl AlignedIoStats {
    /// Create new zeroed stats.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Return the alignment hit rate for reads (0.0 to 1.0).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn read_alignment_rate(&self) -> f64 {
        let total = self.aligned_reads + self.unaligned_reads;
        if total == 0 {
            return 1.0;
        }
        self.aligned_reads as f64 / total as f64
    }

    /// Return the alignment hit rate for writes (0.0 to 1.0).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn write_alignment_rate(&self) -> f64 {
        let total = self.aligned_writes + self.unaligned_writes;
        if total == 0 {
            return 1.0;
        }
        self.aligned_writes as f64 / total as f64
    }

    /// Record an aligned read.
    pub fn record_aligned_read(&mut self, bytes: u64) {
        self.aligned_reads += 1;
        self.bytes_read += bytes;
    }

    /// Record an unaligned read.
    pub fn record_unaligned_read(&mut self, bytes: u64) {
        self.unaligned_reads += 1;
        self.bytes_read += bytes;
    }

    /// Record an aligned write.
    pub fn record_aligned_write(&mut self, bytes: u64) {
        self.aligned_writes += 1;
        self.bytes_written += bytes;
    }

    /// Record an unaligned write.
    pub fn record_unaligned_write(&mut self, bytes: u64) {
        self.unaligned_writes += 1;
        self.bytes_written += bytes;
    }

    /// Merge another stats instance into this one.
    pub fn merge(&mut self, other: &Self) {
        self.aligned_reads += other.aligned_reads;
        self.aligned_writes += other.aligned_writes;
        self.unaligned_reads += other.unaligned_reads;
        self.unaligned_writes += other.unaligned_writes;
        self.bytes_read += other.bytes_read;
        self.bytes_written += other.bytes_written;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Cursor, Read, Write};

    #[test]
    fn test_alignment_spec_bytes() {
        assert_eq!(AlignmentSpec::Sector512.bytes(), 512);
        assert_eq!(AlignmentSpec::Page4k.bytes(), 4096);
        assert_eq!(AlignmentSpec::Block8k.bytes(), 8192);
        assert_eq!(AlignmentSpec::Large64k.bytes(), 65536);
        assert_eq!(AlignmentSpec::Custom(2048).bytes(), 2048);
    }

    #[test]
    fn test_is_valid_alignment() {
        assert!(AlignmentSpec::is_valid_alignment(1));
        assert!(AlignmentSpec::is_valid_alignment(512));
        assert!(AlignmentSpec::is_valid_alignment(4096));
        assert!(!AlignmentSpec::is_valid_alignment(0));
        assert!(!AlignmentSpec::is_valid_alignment(3));
        assert!(!AlignmentSpec::is_valid_alignment(1000));
    }

    #[test]
    fn test_round_up() {
        let spec = AlignmentSpec::Page4k;
        assert_eq!(spec.round_up(0), 0);
        assert_eq!(spec.round_up(1), 4096);
        assert_eq!(spec.round_up(4096), 4096);
        assert_eq!(spec.round_up(4097), 8192);
        assert_eq!(spec.round_up(8000), 8192);
    }

    #[test]
    fn test_round_down() {
        let spec = AlignmentSpec::Sector512;
        assert_eq!(spec.round_down(0), 0);
        assert_eq!(spec.round_down(511), 0);
        assert_eq!(spec.round_down(512), 512);
        assert_eq!(spec.round_down(1000), 512);
        assert_eq!(spec.round_down(1024), 1024);
    }

    #[test]
    fn test_is_aligned() {
        let spec = AlignmentSpec::Page4k;
        assert!(spec.is_aligned(0));
        assert!(spec.is_aligned(4096));
        assert!(spec.is_aligned(8192));
        assert!(!spec.is_aligned(1));
        assert!(!spec.is_aligned(4097));
    }

    #[test]
    fn test_alignment_spec_display() {
        assert_eq!(AlignmentSpec::Sector512.to_string(), "512B-sector");
        assert_eq!(AlignmentSpec::Page4k.to_string(), "4KiB-page");
        assert_eq!(AlignmentSpec::Custom(2048).to_string(), "custom-2048B");
    }

    #[test]
    fn test_aligned_buffer_new() {
        let buf = AlignedBuffer::new(1024, AlignmentSpec::Sector512);
        assert_eq!(buf.capacity(), 1024);
        assert_eq!(buf.len(), 0);
        assert!(buf.is_empty());
        assert!(buf.is_properly_aligned());
    }

    #[test]
    fn test_aligned_buffer_write_and_read() {
        let mut buf = AlignedBuffer::new(256, AlignmentSpec::Sector512);
        let data = b"Hello, aligned world!";
        let written = buf.write(data);
        assert_eq!(written, data.len());
        assert_eq!(buf.len(), data.len());
        assert!(!buf.is_empty());
        assert_eq!(buf.as_slice(), data);
    }

    #[test]
    fn test_aligned_buffer_write_overflow() {
        let mut buf = AlignedBuffer::new(8, AlignmentSpec::Sector512);
        let data = b"0123456789ABCDEF";
        let written = buf.write(data);
        assert_eq!(written, 8);
        assert_eq!(buf.len(), 8);
    }

    #[test]
    fn test_aligned_buffer_clear() {
        let mut buf = AlignedBuffer::new(64, AlignmentSpec::Page4k);
        buf.write(b"data");
        assert_eq!(buf.len(), 4);
        buf.clear();
        assert_eq!(buf.len(), 0);
        assert!(buf.is_empty());
    }

    #[test]
    fn test_aligned_reader() {
        let data: Vec<u8> = (0..256).map(|i| (i & 0xFF) as u8).collect();
        let cursor = Cursor::new(data.clone());
        let mut reader = AlignedReader::new(cursor, 64, AlignmentSpec::Sector512);

        let mut out = vec![0u8; 256];
        let mut total = 0;
        loop {
            let n = reader.read(&mut out[total..]).expect("failed to read");
            if n == 0 {
                break;
            }
            total += n;
        }
        assert_eq!(total, 256);
        assert_eq!(&out[..total], &data[..]);
    }

    #[test]
    fn test_aligned_writer() {
        let inner = Vec::new();
        let mut writer = AlignedWriter::new(inner, 64, AlignmentSpec::Sector512);
        let data = b"test data for aligned writer";
        writer.write_all(data).expect("failed to write");
        writer.flush().expect("failed to flush");
        let total = writer.total_written();
        assert!(total > 0);
    }

    #[test]
    fn test_aligned_io_stats_default() {
        let stats = AlignedIoStats::new();
        assert_eq!(stats.aligned_reads, 0);
        assert_eq!(stats.aligned_writes, 0);
        assert_eq!(stats.bytes_read, 0);
        assert_eq!(stats.bytes_written, 0);
        // No reads at all => rate is 1.0 by convention
        assert!((stats.read_alignment_rate() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_aligned_io_stats_recording() {
        let mut stats = AlignedIoStats::new();
        stats.record_aligned_read(4096);
        stats.record_aligned_read(4096);
        stats.record_unaligned_read(512);
        assert_eq!(stats.aligned_reads, 2);
        assert_eq!(stats.unaligned_reads, 1);
        assert_eq!(stats.bytes_read, 4096 + 4096 + 512);

        let rate = stats.read_alignment_rate();
        let expected = 2.0 / 3.0;
        assert!((rate - expected).abs() < 1e-9);
    }

    #[test]
    fn test_aligned_io_stats_merge() {
        let mut a = AlignedIoStats::new();
        a.record_aligned_write(1000);
        a.record_unaligned_write(500);

        let mut b = AlignedIoStats::new();
        b.record_aligned_write(2000);

        a.merge(&b);
        assert_eq!(a.aligned_writes, 2);
        assert_eq!(a.unaligned_writes, 1);
        assert_eq!(a.bytes_written, 3500);
    }
}
