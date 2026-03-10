#![allow(dead_code)]
//! Streaming media reader utilities for Python bindings.
//!
//! Provides chunk-based reading, buffering, and seek support for
//! streaming media data across the Python/Rust boundary.

use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// ReadMode
// ---------------------------------------------------------------------------

/// Mode controlling how data is read from a stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadMode {
    /// Read the entire stream into memory before returning.
    Eager,
    /// Return data in fixed-size chunks as they become available.
    Chunked,
    /// Return data line-by-line (for text-based formats like SRT/VTT).
    LineByLine,
}

impl std::fmt::Display for ReadMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Eager => write!(f, "eager"),
            Self::Chunked => write!(f, "chunked"),
            Self::LineByLine => write!(f, "line-by-line"),
        }
    }
}

// ---------------------------------------------------------------------------
// StreamStats
// ---------------------------------------------------------------------------

/// Statistics collected during a streaming read session.
#[derive(Debug, Clone)]
pub struct StreamStats {
    /// Total bytes read so far.
    pub bytes_read: u64,
    /// Number of chunks delivered so far.
    pub chunks_delivered: u64,
    /// Average chunk size in bytes (0 when no chunks delivered).
    pub avg_chunk_size: f64,
    /// Peak memory usage in bytes across all internal buffers.
    pub peak_buffer_bytes: u64,
}

impl Default for StreamStats {
    fn default() -> Self {
        Self {
            bytes_read: 0,
            chunks_delivered: 0,
            avg_chunk_size: 0.0,
            peak_buffer_bytes: 0,
        }
    }
}

impl StreamStats {
    /// Create a new empty stats instance.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record that `n` bytes were read.
    #[allow(clippy::cast_precision_loss)]
    pub fn record_read(&mut self, n: u64) {
        self.bytes_read += n;
        self.chunks_delivered += 1;
        self.avg_chunk_size = self.bytes_read as f64 / self.chunks_delivered as f64;
    }

    /// Update peak buffer usage if `current` exceeds the stored peak.
    pub fn update_peak(&mut self, current: u64) {
        if current > self.peak_buffer_bytes {
            self.peak_buffer_bytes = current;
        }
    }
}

// ---------------------------------------------------------------------------
// ChunkBuffer
// ---------------------------------------------------------------------------

/// A simple FIFO buffer that accumulates bytes and emits fixed-size chunks.
#[derive(Debug)]
pub struct ChunkBuffer {
    /// Target chunk size in bytes.
    pub chunk_size: usize,
    /// Internal accumulation buffer.
    buf: Vec<u8>,
    /// Ready chunks waiting to be consumed.
    ready: VecDeque<Vec<u8>>,
}

impl ChunkBuffer {
    /// Create a buffer that emits chunks of `chunk_size` bytes.
    ///
    /// # Panics
    /// Panics if `chunk_size` is 0.
    pub fn new(chunk_size: usize) -> Self {
        assert!(chunk_size > 0, "chunk_size must be > 0");
        Self {
            chunk_size,
            buf: Vec::with_capacity(chunk_size),
            ready: VecDeque::new(),
        }
    }

    /// Push raw bytes into the buffer, potentially producing one or more chunks.
    pub fn push(&mut self, data: &[u8]) {
        self.buf.extend_from_slice(data);
        while self.buf.len() >= self.chunk_size {
            let rest = self.buf.split_off(self.chunk_size);
            let chunk = std::mem::replace(&mut self.buf, rest);
            self.ready.push_back(chunk);
        }
    }

    /// Take the next ready chunk, if any.
    pub fn pop_chunk(&mut self) -> Option<Vec<u8>> {
        self.ready.pop_front()
    }

    /// Flush remaining bytes as a final (possibly undersized) chunk.
    pub fn flush(&mut self) -> Option<Vec<u8>> {
        if self.buf.is_empty() {
            None
        } else {
            Some(std::mem::take(&mut self.buf))
        }
    }

    /// Number of ready chunks available for consumption.
    pub fn ready_count(&self) -> usize {
        self.ready.len()
    }

    /// Total bytes currently held (buffered + ready).
    pub fn buffered_bytes(&self) -> usize {
        self.buf.len() + self.ready.iter().map(|c| c.len()).sum::<usize>()
    }
}

// ---------------------------------------------------------------------------
// SeekHint
// ---------------------------------------------------------------------------

/// A hint that can be sent to a remote source to request a byte-range seek.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SeekHint {
    /// Byte offset to seek to.
    pub offset: u64,
    /// Optional length of the range (None = to end).
    pub length: Option<u64>,
}

impl SeekHint {
    /// Create a seek hint for a given offset with optional length.
    pub fn new(offset: u64, length: Option<u64>) -> Self {
        Self { offset, length }
    }

    /// Return the inclusive end byte, or `None` if unbounded.
    pub fn end_byte(&self) -> Option<u64> {
        self.length.map(|l| self.offset + l - 1)
    }

    /// Format as an HTTP `Range` header value.
    pub fn to_range_header(&self) -> String {
        match self.length {
            Some(l) => format!("bytes={}-{}", self.offset, self.offset + l - 1),
            None => format!("bytes={}-", self.offset),
        }
    }
}

// ---------------------------------------------------------------------------
// StreamReader
// ---------------------------------------------------------------------------

/// High-level streaming reader that wraps a `ChunkBuffer` with stats tracking.
#[derive(Debug)]
pub struct StreamReader {
    /// The underlying chunk buffer.
    buffer: ChunkBuffer,
    /// Current read mode.
    mode: ReadMode,
    /// Accumulated statistics.
    stats: StreamStats,
    /// Whether end-of-stream has been signalled.
    eos: bool,
}

impl StreamReader {
    /// Create a new stream reader with the given mode and chunk size.
    pub fn new(mode: ReadMode, chunk_size: usize) -> Self {
        Self {
            buffer: ChunkBuffer::new(chunk_size),
            mode,
            stats: StreamStats::new(),
            eos: false,
        }
    }

    /// Feed raw bytes into the reader.
    pub fn feed(&mut self, data: &[u8]) {
        self.buffer.push(data);
        let buffered = self.buffer.buffered_bytes() as u64;
        self.stats.update_peak(buffered);
    }

    /// Signal end-of-stream.
    pub fn signal_eos(&mut self) {
        self.eos = true;
    }

    /// Retrieve the next chunk from the reader.
    pub fn next_chunk(&mut self) -> Option<Vec<u8>> {
        if let Some(chunk) = self.buffer.pop_chunk() {
            self.stats.record_read(chunk.len() as u64);
            return Some(chunk);
        }
        if self.eos {
            if let Some(chunk) = self.buffer.flush() {
                self.stats.record_read(chunk.len() as u64);
                return Some(chunk);
            }
        }
        None
    }

    /// Current read mode.
    pub fn mode(&self) -> ReadMode {
        self.mode
    }

    /// Current statistics snapshot.
    pub fn stats(&self) -> &StreamStats {
        &self.stats
    }

    /// Whether end-of-stream has been reached and all chunks consumed.
    pub fn is_done(&self) -> bool {
        self.eos && self.buffer.ready_count() == 0 && self.buffer.buffered_bytes() == 0
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_mode_display() {
        assert_eq!(ReadMode::Eager.to_string(), "eager");
        assert_eq!(ReadMode::Chunked.to_string(), "chunked");
        assert_eq!(ReadMode::LineByLine.to_string(), "line-by-line");
    }

    #[test]
    fn test_read_mode_equality() {
        assert_eq!(ReadMode::Eager, ReadMode::Eager);
        assert_ne!(ReadMode::Eager, ReadMode::Chunked);
    }

    #[test]
    fn test_stream_stats_default() {
        let s = StreamStats::new();
        assert_eq!(s.bytes_read, 0);
        assert_eq!(s.chunks_delivered, 0);
        assert_eq!(s.peak_buffer_bytes, 0);
    }

    #[test]
    fn test_stream_stats_record_read() {
        let mut s = StreamStats::new();
        s.record_read(100);
        assert_eq!(s.bytes_read, 100);
        assert_eq!(s.chunks_delivered, 1);
        assert!((s.avg_chunk_size - 100.0).abs() < f64::EPSILON);
        s.record_read(200);
        assert_eq!(s.bytes_read, 300);
        assert_eq!(s.chunks_delivered, 2);
        assert!((s.avg_chunk_size - 150.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_stream_stats_peak() {
        let mut s = StreamStats::new();
        s.update_peak(500);
        assert_eq!(s.peak_buffer_bytes, 500);
        s.update_peak(300);
        assert_eq!(s.peak_buffer_bytes, 500);
        s.update_peak(700);
        assert_eq!(s.peak_buffer_bytes, 700);
    }

    #[test]
    fn test_chunk_buffer_basic() {
        let mut buf = ChunkBuffer::new(4);
        buf.push(b"abcdefgh");
        assert_eq!(buf.ready_count(), 2);
        assert_eq!(buf.pop_chunk().expect("pop_chunk should succeed"), b"abcd");
        assert_eq!(buf.pop_chunk().expect("pop_chunk should succeed"), b"efgh");
        assert!(buf.pop_chunk().is_none());
    }

    #[test]
    fn test_chunk_buffer_partial() {
        let mut buf = ChunkBuffer::new(4);
        buf.push(b"ab");
        assert_eq!(buf.ready_count(), 0);
        buf.push(b"cd");
        assert_eq!(buf.ready_count(), 1);
        assert_eq!(buf.pop_chunk().expect("pop_chunk should succeed"), b"abcd");
    }

    #[test]
    fn test_chunk_buffer_flush() {
        let mut buf = ChunkBuffer::new(10);
        buf.push(b"hello");
        assert!(buf.pop_chunk().is_none());
        let flushed = buf.flush().expect("flushed should be valid");
        assert_eq!(flushed, b"hello");
        assert!(buf.flush().is_none());
    }

    #[test]
    fn test_chunk_buffer_buffered_bytes() {
        let mut buf = ChunkBuffer::new(4);
        buf.push(b"abcde");
        // 1 ready chunk (4 bytes) + 1 byte in partial buffer
        assert_eq!(buf.buffered_bytes(), 5);
    }

    #[test]
    fn test_seek_hint_no_length() {
        let hint = SeekHint::new(100, None);
        assert_eq!(hint.offset, 100);
        assert!(hint.end_byte().is_none());
        assert_eq!(hint.to_range_header(), "bytes=100-");
    }

    #[test]
    fn test_seek_hint_with_length() {
        let hint = SeekHint::new(50, Some(200));
        assert_eq!(hint.end_byte(), Some(249));
        assert_eq!(hint.to_range_header(), "bytes=50-249");
    }

    #[test]
    fn test_stream_reader_basic() {
        let mut reader = StreamReader::new(ReadMode::Chunked, 4);
        reader.feed(b"abcdefgh");
        let c1 = reader.next_chunk().expect("c1 should be valid");
        assert_eq!(c1, b"abcd");
        let c2 = reader.next_chunk().expect("c2 should be valid");
        assert_eq!(c2, b"efgh");
        assert!(reader.next_chunk().is_none());
        assert!(!reader.is_done());
        reader.signal_eos();
        assert!(reader.is_done());
    }

    #[test]
    fn test_stream_reader_eos_flush() {
        let mut reader = StreamReader::new(ReadMode::Chunked, 10);
        reader.feed(b"short");
        assert!(reader.next_chunk().is_none());
        reader.signal_eos();
        let c = reader.next_chunk().expect("c should be valid");
        assert_eq!(c, b"short");
        assert!(reader.is_done());
    }

    #[test]
    fn test_stream_reader_stats_update() {
        let mut reader = StreamReader::new(ReadMode::Eager, 4);
        reader.feed(b"12345678");
        let _ = reader.next_chunk();
        let _ = reader.next_chunk();
        assert_eq!(reader.stats().bytes_read, 8);
        assert_eq!(reader.stats().chunks_delivered, 2);
    }

    #[test]
    fn test_stream_reader_mode() {
        let reader = StreamReader::new(ReadMode::LineByLine, 16);
        assert_eq!(reader.mode(), ReadMode::LineByLine);
    }
}
