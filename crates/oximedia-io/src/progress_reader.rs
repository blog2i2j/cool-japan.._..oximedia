//! Progress-tracking reader wrapper for I/O streams.
//!
//! Wraps any `Read` implementation and reports progress via a callback as
//! bytes are consumed. Useful for displaying progress bars during large file
//! reads, checksum computation, or transcoding pipelines.

#![allow(dead_code)]

use std::io::{self, Read};

// ---------------------------------------------------------------------------
// Progress callback
// ---------------------------------------------------------------------------

/// Information passed to the progress callback on each read.
#[derive(Debug, Clone, Copy)]
pub struct ReadProgress {
    /// Total bytes read so far.
    pub bytes_read: u64,
    /// Total expected size, if known.
    pub total_bytes: Option<u64>,
    /// Instantaneous throughput in bytes/sec (smoothed).
    pub throughput_bps: f64,
}

impl ReadProgress {
    /// Return progress as a fraction in `[0.0, 1.0]`, or `None` if total is unknown.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn fraction(&self) -> Option<f64> {
        self.total_bytes.map(|t| {
            if t == 0 {
                1.0
            } else {
                (self.bytes_read as f64) / (t as f64)
            }
        })
    }

    /// Return progress as a percentage (0-100), or `None`.
    #[must_use]
    pub fn percent(&self) -> Option<f64> {
        self.fraction().map(|f| f * 100.0)
    }
}

// ---------------------------------------------------------------------------
// ProgressReader
// ---------------------------------------------------------------------------

/// A reader that reports progress through a callback.
///
/// Wraps any `R: Read` and invokes the supplied closure after each `read()`
/// call, passing a [`ReadProgress`] snapshot.
pub struct ProgressReader<R, F> {
    /// The inner reader.
    inner: R,
    /// Callback invoked on each read.
    callback: F,
    /// Total bytes read so far.
    bytes_read: u64,
    /// Optional total size.
    total_bytes: Option<u64>,
    /// Timestamp (monotonic ns) of last throughput calculation.
    last_ts: u64,
    /// Bytes at last throughput calculation.
    last_bytes: u64,
    /// Smoothed throughput.
    throughput: f64,
    /// Minimum bytes between callback invocations.
    report_interval: u64,
    /// Bytes since last report.
    since_last_report: u64,
}

impl<R: Read, F: FnMut(&ReadProgress)> ProgressReader<R, F> {
    /// Create a new `ProgressReader` wrapping `inner`.
    pub fn new(inner: R, callback: F) -> Self {
        Self {
            inner,
            callback,
            bytes_read: 0,
            total_bytes: None,
            last_ts: Self::now_ns(),
            last_bytes: 0,
            throughput: 0.0,
            report_interval: 0,
            since_last_report: 0,
        }
    }

    /// Set the expected total size so progress fraction can be computed.
    #[must_use]
    pub fn with_total(mut self, total: u64) -> Self {
        self.total_bytes = Some(total);
        self
    }

    /// Set a minimum number of bytes between callback invocations.
    ///
    /// Default is 0 (report on every `read` call).
    #[must_use]
    pub fn with_report_interval(mut self, interval: u64) -> Self {
        self.report_interval = interval;
        self
    }

    /// Return total bytes read so far.
    #[must_use]
    pub fn bytes_read(&self) -> u64 {
        self.bytes_read
    }

    /// Consume this reader, returning the inner reader.
    #[must_use]
    pub fn into_inner(self) -> R {
        self.inner
    }

    /// Monotonic clock in nanoseconds (approximation via `Instant`).
    #[allow(clippy::cast_possible_truncation)]
    fn now_ns() -> u64 {
        use std::time::SystemTime;
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64
    }

    /// Update throughput estimate and invoke callback.
    #[allow(clippy::cast_precision_loss)]
    fn report(&mut self) {
        let now = Self::now_ns();
        let elapsed_ns = now.saturating_sub(self.last_ts);
        if elapsed_ns > 0 {
            let bytes_delta = self.bytes_read.saturating_sub(self.last_bytes);
            let bps = (bytes_delta as f64) / (elapsed_ns as f64 / 1_000_000_000.0);
            // exponential moving average
            self.throughput = self.throughput * 0.7 + bps * 0.3;
        }
        self.last_ts = now;
        self.last_bytes = self.bytes_read;

        let progress = ReadProgress {
            bytes_read: self.bytes_read,
            total_bytes: self.total_bytes,
            throughput_bps: self.throughput,
        };
        (self.callback)(&progress);
    }
}

impl<R: Read, F: FnMut(&ReadProgress)> Read for ProgressReader<R, F> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.bytes_read += n as u64;
        self.since_last_report += n as u64;

        if self.since_last_report >= self.report_interval {
            self.since_last_report = 0;
            self.report();
        }
        Ok(n)
    }
}

// ---------------------------------------------------------------------------
// Throttled counter (no callback)
// ---------------------------------------------------------------------------

/// A simple byte counter that wraps a `Read` without a callback.
///
/// Lighter alternative when you only need the final count.
pub struct ByteCounter<R> {
    /// The inner reader.
    inner: R,
    /// Total bytes read.
    count: u64,
}

impl<R: Read> ByteCounter<R> {
    /// Wrap `inner`.
    pub fn new(inner: R) -> Self {
        Self { inner, count: 0 }
    }

    /// Return the number of bytes read so far.
    #[must_use]
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Consume the counter and return the inner reader.
    #[must_use]
    pub fn into_inner(self) -> R {
        self.inner
    }
}

impl<R: Read> Read for ByteCounter<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.count += n as u64;
        Ok(n)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;

    #[test]
    fn test_read_progress_fraction_known() {
        let p = ReadProgress {
            bytes_read: 50,
            total_bytes: Some(100),
            throughput_bps: 0.0,
        };
        let f = p.fraction().expect("fraction should be known");
        assert!((f - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_read_progress_fraction_unknown() {
        let p = ReadProgress {
            bytes_read: 50,
            total_bytes: None,
            throughput_bps: 0.0,
        };
        assert!(p.fraction().is_none());
    }

    #[test]
    fn test_read_progress_percent() {
        let p = ReadProgress {
            bytes_read: 75,
            total_bytes: Some(100),
            throughput_bps: 0.0,
        };
        let pct = p.percent().expect("percent should be known");
        assert!((pct - 75.0).abs() < 1e-9);
    }

    #[test]
    fn test_read_progress_zero_total() {
        let p = ReadProgress {
            bytes_read: 0,
            total_bytes: Some(0),
            throughput_bps: 0.0,
        };
        assert!((p.fraction().expect("fraction should be known") - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_progress_reader_basic() {
        let data = b"hello world, this is progress reader test data!";
        let cursor = Cursor::new(data.to_vec());
        let call_count = Arc::new(AtomicU64::new(0));
        let cc = Arc::clone(&call_count);

        let mut reader = ProgressReader::new(cursor, move |_p| {
            cc.fetch_add(1, Ordering::Relaxed);
        });

        let mut buf = [0u8; 10];
        let mut total = 0usize;
        loop {
            let n = reader.read(&mut buf).expect("failed to read");
            if n == 0 {
                break;
            }
            total += n;
        }
        assert_eq!(total, data.len());
        assert!(call_count.load(Ordering::Relaxed) > 0);
    }

    #[test]
    fn test_progress_reader_with_total() {
        let data = vec![0u8; 200];
        let cursor = Cursor::new(data.clone());

        let mut reader = ProgressReader::new(cursor, |p| {
            let _ = p.fraction();
        })
        .with_total(200);

        let mut buf = [0u8; 50];
        while reader.read(&mut buf).expect("failed to read") > 0 {}
        assert_eq!(reader.bytes_read(), 200);
    }

    #[test]
    fn test_progress_reader_report_interval() {
        let data = vec![42u8; 1000];
        let cursor = Cursor::new(data);
        let call_count = Arc::new(AtomicU64::new(0));
        let cc = Arc::clone(&call_count);

        let mut reader = ProgressReader::new(cursor, move |_| {
            cc.fetch_add(1, Ordering::Relaxed);
        })
        .with_report_interval(500);

        let mut buf = [0u8; 100];
        while reader.read(&mut buf).expect("failed to read") > 0 {}
        // with 1000 bytes and 500 interval, expect ~2 reports
        let count = call_count.load(Ordering::Relaxed);
        assert!(count >= 2, "expected at least 2 reports, got {count}");
    }

    #[test]
    fn test_progress_reader_into_inner() {
        let data = vec![1, 2, 3];
        let cursor = Cursor::new(data.clone());
        let reader = ProgressReader::new(cursor, |_| {});
        let inner = reader.into_inner();
        assert_eq!(inner.into_inner(), data);
    }

    #[test]
    fn test_byte_counter_basic() {
        let data = b"count these bytes";
        let cursor = Cursor::new(data.to_vec());
        let mut counter = ByteCounter::new(cursor);

        let mut buf = [0u8; 5];
        let mut total = 0usize;
        loop {
            let n = counter.read(&mut buf).expect("failed to read");
            if n == 0 {
                break;
            }
            total += n;
        }
        assert_eq!(total, data.len());
        assert_eq!(counter.count(), data.len() as u64);
    }

    #[test]
    fn test_byte_counter_empty() {
        let cursor = Cursor::new(Vec::<u8>::new());
        let mut counter = ByteCounter::new(cursor);
        let mut buf = [0u8; 16];
        let n = counter.read(&mut buf).expect("failed to read");
        assert_eq!(n, 0);
        assert_eq!(counter.count(), 0);
    }

    #[test]
    fn test_byte_counter_into_inner() {
        let data = vec![10, 20, 30];
        let cursor = Cursor::new(data.clone());
        let counter = ByteCounter::new(cursor);
        let inner = counter.into_inner();
        assert_eq!(inner.into_inner(), data);
    }

    #[test]
    fn test_progress_throughput_positive() {
        let data = vec![0u8; 10_000];
        let cursor = Cursor::new(data);

        let mut reader = ProgressReader::new(cursor, |p| {
            let _ = p.throughput_bps;
        });

        let mut buf = [0u8; 500];
        while reader.read(&mut buf).expect("failed to read") > 0 {}
        assert_eq!(reader.bytes_read(), 10_000);
    }
}
