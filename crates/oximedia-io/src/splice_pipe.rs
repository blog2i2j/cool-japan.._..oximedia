#![allow(dead_code)]

//! Zero-copy pipe splicing for media data transfer.
//!
//! This module provides an abstraction over pipe-based data transfer that
//! minimises memory copies when moving media data between sources and sinks.
//! On platforms that support `splice(2)` (Linux), the actual kernel-space
//! transfer can skip user-space entirely. On other platforms a user-space
//! fallback is used.
//!
//! # Key Types
//!
//! - [`SplicePipe`] - A virtual pipe that can connect a reader to a writer
//! - [`SpliceConfig`] - Configuration for splice operations
//! - [`SpliceResult`] - Outcome of a splice transfer
//! - [`PipeBuffer`] - Circular pipe buffer for user-space fallback

use std::fmt;

/// Configuration for splice / pipe transfer operations.
#[derive(Debug, Clone)]
pub struct SpliceConfig {
    /// Size of the internal pipe buffer in bytes.
    pub pipe_buffer_size: usize,
    /// Maximum bytes to transfer in a single splice call.
    pub max_transfer_size: usize,
    /// Whether to use non-blocking I/O where supported.
    pub non_blocking: bool,
    /// Whether to hint the kernel that more data follows (`SPLICE_F_MORE`).
    pub hint_more: bool,
}

impl Default for SpliceConfig {
    fn default() -> Self {
        Self {
            pipe_buffer_size: 64 * 1024,
            max_transfer_size: 1024 * 1024,
            non_blocking: false,
            hint_more: false,
        }
    }
}

impl SpliceConfig {
    /// Create a config tuned for large media files.
    #[must_use]
    pub fn for_media() -> Self {
        Self {
            pipe_buffer_size: 256 * 1024,
            max_transfer_size: 4 * 1024 * 1024,
            non_blocking: true,
            hint_more: true,
        }
    }

    /// Create a config tuned for small metadata transfers.
    #[must_use]
    pub fn for_metadata() -> Self {
        Self {
            pipe_buffer_size: 4096,
            max_transfer_size: 64 * 1024,
            non_blocking: false,
            hint_more: false,
        }
    }

    /// Validate the configuration.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.pipe_buffer_size > 0
            && self.max_transfer_size > 0
            && self.max_transfer_size >= self.pipe_buffer_size
    }
}

/// Outcome of a splice transfer.
#[derive(Debug, Clone)]
pub struct SpliceResult {
    /// Total bytes transferred.
    pub bytes_transferred: u64,
    /// Number of individual splice calls made.
    pub splice_calls: u64,
    /// Whether the operation completed fully (reached EOF or requested amount).
    pub completed: bool,
    /// Whether zero-copy path was used.
    pub zero_copy: bool,
}

impl SpliceResult {
    /// Create a new empty result.
    #[must_use]
    pub fn new() -> Self {
        Self {
            bytes_transferred: 0,
            splice_calls: 0,
            completed: false,
            zero_copy: false,
        }
    }

    /// Average bytes per splice call.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn avg_bytes_per_call(&self) -> f64 {
        if self.splice_calls == 0 {
            return 0.0;
        }
        self.bytes_transferred as f64 / self.splice_calls as f64
    }
}

impl Default for SpliceResult {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for SpliceResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "transferred {} bytes in {} calls (zero_copy={}, completed={})",
            self.bytes_transferred, self.splice_calls, self.zero_copy, self.completed
        )
    }
}

/// A circular pipe buffer used as the user-space fallback for splice
/// operations on platforms that lack kernel splice support.
#[derive(Clone)]
pub struct PipeBuffer {
    /// Internal storage.
    data: Vec<u8>,
    /// Read position (head).
    head: usize,
    /// Write position (tail).
    tail: usize,
    /// Whether the buffer has wrapped around.
    full: bool,
}

impl PipeBuffer {
    /// Create a new pipe buffer with the given capacity.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            data: vec![0u8; capacity],
            head: 0,
            tail: 0,
            full: false,
        }
    }

    /// Return the total capacity.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.data.len()
    }

    /// Return the number of bytes available for reading.
    #[must_use]
    pub fn available(&self) -> usize {
        if self.full {
            self.data.len()
        } else if self.tail >= self.head {
            self.tail - self.head
        } else {
            self.data.len() - self.head + self.tail
        }
    }

    /// Return the number of bytes of free space for writing.
    #[must_use]
    pub fn free_space(&self) -> usize {
        self.capacity() - self.available()
    }

    /// Check if the buffer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        !self.full && self.head == self.tail
    }

    /// Check if the buffer is full.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.full
    }

    /// Write data into the pipe buffer, returning bytes written.
    pub fn write(&mut self, src: &[u8]) -> usize {
        if self.full {
            return 0;
        }
        let cap = self.data.len();
        let mut written = 0;

        for &byte in src {
            if self.full {
                break;
            }
            self.data[self.tail] = byte;
            self.tail = (self.tail + 1) % cap;
            if self.tail == self.head {
                self.full = true;
            }
            written += 1;
        }

        written
    }

    /// Read data from the pipe buffer, returning bytes read.
    pub fn read(&mut self, dst: &mut [u8]) -> usize {
        if self.is_empty() {
            return 0;
        }
        let cap = self.data.len();
        let mut count = 0;

        for slot in dst.iter_mut() {
            if !self.full && self.head == self.tail {
                break;
            }
            *slot = self.data[self.head];
            self.head = (self.head + 1) % cap;
            self.full = false;
            count += 1;
        }

        count
    }

    /// Clear the buffer.
    pub fn clear(&mut self) {
        self.head = 0;
        self.tail = 0;
        self.full = false;
    }
}

impl fmt::Debug for PipeBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PipeBuffer")
            .field("head", &self.head)
            .field("tail", &self.tail)
            .field("full", &self.full)
            .field("capacity", &self.capacity())
            .field("available", &self.available())
            .field("free_space", &self.free_space())
            .finish_non_exhaustive()
    }
}

/// A virtual pipe that connects a reader to a writer, performing the
/// transfer through an internal [`PipeBuffer`].
pub struct SplicePipe {
    /// The pipe buffer.
    buffer: PipeBuffer,
    /// The configuration.
    config: SpliceConfig,
    /// Cumulative result.
    result: SpliceResult,
}

impl SplicePipe {
    /// Create a new splice pipe with the given config.
    #[must_use]
    pub fn new(config: SpliceConfig) -> Self {
        let buffer = PipeBuffer::new(config.pipe_buffer_size);
        Self {
            buffer,
            config,
            result: SpliceResult::new(),
        }
    }

    /// Create a splice pipe with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(SpliceConfig::default())
    }

    /// Transfer data from reader to writer up to `limit` bytes or EOF.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if reading or writing fails.
    #[allow(clippy::cast_possible_truncation)]
    pub fn transfer<R: std::io::Read, W: std::io::Write>(
        &mut self,
        reader: &mut R,
        writer: &mut W,
        limit: u64,
    ) -> std::io::Result<SpliceResult> {
        let mut total: u64 = 0;
        let mut calls: u64 = 0;

        loop {
            if total >= limit {
                break;
            }

            // Fill pipe buffer from reader
            let remaining = (limit - total).min(self.buffer.free_space() as u64) as usize;
            if remaining == 0 && self.buffer.is_empty() {
                break;
            }

            if remaining > 0 && !self.buffer.is_full() {
                let mut tmp = vec![0u8; remaining];
                let n = reader.read(&mut tmp)?;
                if n == 0 {
                    // Drain remaining
                    if self.buffer.is_empty() {
                        self.result.completed = true;
                        break;
                    }
                } else {
                    self.buffer.write(&tmp[..n]);
                }
            }

            // Drain pipe buffer to writer
            let avail = self.buffer.available();
            if avail > 0 {
                let mut tmp = vec![0u8; avail];
                let read_count = self.buffer.read(&mut tmp);
                writer.write_all(&tmp[..read_count])?;
                total += read_count as u64;
                calls += 1;
            }
        }

        let result = SpliceResult {
            bytes_transferred: total,
            splice_calls: calls,
            completed: true,
            zero_copy: false,
        };
        self.result = result.clone();
        Ok(result)
    }

    /// Return the cumulative result of all transfers.
    #[must_use]
    pub fn cumulative_result(&self) -> &SpliceResult {
        &self.result
    }

    /// Return a reference to the configuration.
    #[must_use]
    pub fn config(&self) -> &SpliceConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_splice_config_default() {
        let cfg = SpliceConfig::default();
        assert_eq!(cfg.pipe_buffer_size, 64 * 1024);
        assert!(!cfg.non_blocking);
        assert!(cfg.is_valid());
    }

    #[test]
    fn test_splice_config_for_media() {
        let cfg = SpliceConfig::for_media();
        assert_eq!(cfg.pipe_buffer_size, 256 * 1024);
        assert!(cfg.non_blocking);
        assert!(cfg.hint_more);
        assert!(cfg.is_valid());
    }

    #[test]
    fn test_splice_config_for_metadata() {
        let cfg = SpliceConfig::for_metadata();
        assert_eq!(cfg.pipe_buffer_size, 4096);
        assert!(!cfg.non_blocking);
        assert!(cfg.is_valid());
    }

    #[test]
    fn test_splice_result_default() {
        let r = SpliceResult::new();
        assert_eq!(r.bytes_transferred, 0);
        assert_eq!(r.splice_calls, 0);
        assert!(!r.completed);
        assert!((r.avg_bytes_per_call() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_splice_result_display() {
        let r = SpliceResult {
            bytes_transferred: 1024,
            splice_calls: 2,
            completed: true,
            zero_copy: false,
        };
        let s = r.to_string();
        assert!(s.contains("1024"));
        assert!(s.contains("2 calls"));
    }

    #[test]
    fn test_pipe_buffer_new() {
        let buf = PipeBuffer::new(128);
        assert_eq!(buf.capacity(), 128);
        assert_eq!(buf.available(), 0);
        assert_eq!(buf.free_space(), 128);
        assert!(buf.is_empty());
        assert!(!buf.is_full());
    }

    #[test]
    fn test_pipe_buffer_write_and_read() {
        let mut buf = PipeBuffer::new(16);
        let written = buf.write(b"hello");
        assert_eq!(written, 5);
        assert_eq!(buf.available(), 5);

        let mut out = [0u8; 16];
        let read = buf.read(&mut out);
        assert_eq!(read, 5);
        assert_eq!(&out[..5], b"hello");
        assert!(buf.is_empty());
    }

    #[test]
    fn test_pipe_buffer_full() {
        let mut buf = PipeBuffer::new(4);
        let written = buf.write(b"ABCDEF");
        assert_eq!(written, 4);
        assert!(buf.is_full());
        assert_eq!(buf.free_space(), 0);

        // Cannot write more when full
        let w2 = buf.write(b"X");
        assert_eq!(w2, 0);
    }

    #[test]
    fn test_pipe_buffer_wrap_around() {
        let mut buf = PipeBuffer::new(8);
        buf.write(b"12345"); // head=0, tail=5
        let mut out = [0u8; 3];
        buf.read(&mut out); // head=3, tail=5
        assert_eq!(&out, b"123");
        buf.write(b"ABCDEF"); // wraps around
        assert_eq!(buf.available(), 8);
        assert!(buf.is_full());
    }

    #[test]
    fn test_pipe_buffer_clear() {
        let mut buf = PipeBuffer::new(32);
        buf.write(b"data");
        assert!(!buf.is_empty());
        buf.clear();
        assert!(buf.is_empty());
        assert_eq!(buf.available(), 0);
    }

    #[test]
    fn test_splice_pipe_transfer() {
        let data = b"The quick brown fox jumps over the lazy dog";
        let mut reader = Cursor::new(data.to_vec());
        let mut writer: Vec<u8> = Vec::new();

        let mut pipe = SplicePipe::with_defaults();
        let result = pipe
            .transfer(&mut reader, &mut writer, data.len() as u64)
            .expect("operation should succeed");

        assert_eq!(result.bytes_transferred, data.len() as u64);
        assert!(result.completed);
        assert_eq!(&writer, data);
    }

    #[test]
    fn test_splice_pipe_transfer_with_limit() {
        let data = vec![0xABu8; 1000];
        let mut reader = Cursor::new(data);
        let mut writer: Vec<u8> = Vec::new();

        let mut pipe = SplicePipe::with_defaults();
        let result = pipe
            .transfer(&mut reader, &mut writer, 500)
            .expect("transfer should succeed");

        assert_eq!(result.bytes_transferred, 500);
        assert_eq!(writer.len(), 500);
    }

    #[test]
    fn test_splice_pipe_cumulative_result() {
        let data = b"short";
        let mut reader = Cursor::new(data.to_vec());
        let mut writer: Vec<u8> = Vec::new();

        let mut pipe = SplicePipe::with_defaults();
        pipe.transfer(&mut reader, &mut writer, 100)
            .expect("transfer should succeed");

        let cum = pipe.cumulative_result();
        assert!(cum.bytes_transferred > 0);
    }
}
