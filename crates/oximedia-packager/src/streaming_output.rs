//! Streaming segment output — write segments as they are produced rather than
//! buffering entire files in memory.
//!
//! [`SegmentStream`] provides a channel-based sink that accepts produced segments
//! asynchronously and writes them to disk (or hands them to a custom sink
//! callback) as soon as they arrive.  This allows the packaging pipeline to
//! start delivering segments to downstream consumers before the entire input
//! has been processed.
//!
//! # Design
//!
//! ```text
//!  Packager ──► SegmentSender ──(channel)──► SegmentStream ──► Disk / CDN
//!                                                         └───► Manifest updater
//! ```
//!
//! # Example
//!
//! ```rust
//! use oximedia_packager::streaming_output::{SegmentStream, SegmentStreamConfig, ProducedSegment};
//! use std::path::PathBuf;
//!
//! # tokio_test::block_on(async {
//! let config = SegmentStreamConfig::default();
//! let (mut stream, tx) = SegmentStream::new(config);
//!
//! // Simulate sending two segments
//! let seg1 = ProducedSegment {
//!     sequence: 0,
//!     data: vec![0u8; 512],
//!     duration_secs: 6.0,
//!     is_init: false,
//!     path_hint: Some(PathBuf::from("/tmp/seg0.ts")),
//! };
//! tx.send(seg1).await.expect("send ok");
//! drop(tx); // close the channel
//!
//! // Drain produced segments (no-op writer mode for testing)
//! while let Some(seg) = stream.next().await {
//!     assert_eq!(seg.sequence, 0);
//! }
//! # });
//! ```

use std::path::PathBuf;
use tokio::sync::mpsc;

// ---------------------------------------------------------------------------
// ProducedSegment
// ---------------------------------------------------------------------------

/// A fully encoded segment ready for output.
#[derive(Debug, Clone)]
pub struct ProducedSegment {
    /// Monotonically increasing sequence number (0-based).
    pub sequence: u64,
    /// Encoded byte payload (TS, fMP4, …).
    pub data: Vec<u8>,
    /// Duration of this segment in seconds.
    pub duration_secs: f64,
    /// Whether this is an initialisation segment (fMP4 init, CMAF init).
    pub is_init: bool,
    /// Optional filesystem path hint for writing to disk.
    pub path_hint: Option<PathBuf>,
}

impl ProducedSegment {
    /// Create a media segment (not init).
    #[must_use]
    pub fn media(sequence: u64, data: Vec<u8>, duration_secs: f64) -> Self {
        Self {
            sequence,
            data,
            duration_secs,
            is_init: false,
            path_hint: None,
        }
    }

    /// Create an initialisation segment.
    #[must_use]
    pub fn init(data: Vec<u8>) -> Self {
        Self {
            sequence: 0,
            data,
            duration_secs: 0.0,
            is_init: true,
            path_hint: None,
        }
    }

    /// Attach a filesystem path hint.
    #[must_use]
    pub fn with_path(mut self, path: PathBuf) -> Self {
        self.path_hint = Some(path);
        self
    }

    /// Return the byte length of the segment payload.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.data.len()
    }
}

// ---------------------------------------------------------------------------
// SegmentStreamConfig
// ---------------------------------------------------------------------------

/// Configuration for a [`SegmentStream`].
#[derive(Debug, Clone)]
pub struct SegmentStreamConfig {
    /// Channel buffer depth (number of segments that can be queued before
    /// the producer is back-pressured).  Default: 8.
    pub channel_depth: usize,
    /// Write segments to disk automatically when a `path_hint` is present.
    /// Default: `true`.
    pub auto_write: bool,
    /// Whether to pre-allocate the output file using `set_len` before writing.
    /// This reduces filesystem fragmentation at the cost of one extra syscall
    /// per segment.  Default: `false`.
    pub pre_allocate: bool,
}

impl Default for SegmentStreamConfig {
    fn default() -> Self {
        Self {
            channel_depth: 8,
            auto_write: true,
            pre_allocate: false,
        }
    }
}

// ---------------------------------------------------------------------------
// SegmentSender / SegmentStream
// ---------------------------------------------------------------------------

/// Sending half of a [`SegmentStream`] channel.
///
/// Obtained from [`SegmentStream::new`].  Clone freely — the stream closes
/// when the last sender is dropped.
#[derive(Clone, Debug)]
pub struct SegmentSender {
    tx: mpsc::Sender<ProducedSegment>,
}

impl SegmentSender {
    /// Send a segment.  Back-pressures when the channel buffer is full.
    ///
    /// # Errors
    ///
    /// Returns an error if the receiving [`SegmentStream`] has been dropped.
    pub async fn send(&self, segment: ProducedSegment) -> Result<(), String> {
        self.tx
            .send(segment)
            .await
            .map_err(|e| format!("SegmentSender: channel closed: {e}"))
    }

    /// Non-blocking try-send (drops if the channel is full).
    pub fn try_send(&self, segment: ProducedSegment) -> Result<(), String> {
        self.tx
            .try_send(segment)
            .map_err(|e| format!("SegmentSender::try_send: {e}"))
    }
}

/// Receiving / writing half of a segment output stream.
///
/// Call [`SegmentStream::next`] in a loop (or `await` on each produced
/// segment), optionally writing to disk via the `auto_write` path.
pub struct SegmentStream {
    rx: mpsc::Receiver<ProducedSegment>,
    config: SegmentStreamConfig,
    /// Running total of bytes written.
    bytes_written: u64,
    /// Number of segments received.
    segments_received: u64,
}

impl SegmentStream {
    /// Create a new streaming segment output.
    ///
    /// Returns `(stream, sender)`.  The `sender` is given to the packaging
    /// pipeline; the `stream` is owned by the output / manifest updater.
    #[must_use]
    pub fn new(config: SegmentStreamConfig) -> (Self, SegmentSender) {
        let (tx, rx) = mpsc::channel(config.channel_depth);
        let stream = Self {
            rx,
            config,
            bytes_written: 0,
            segments_received: 0,
        };
        let sender = SegmentSender { tx };
        (stream, sender)
    }

    /// Wait for the next produced segment.
    ///
    /// Returns `None` when all senders have been dropped (end of stream).
    /// If `auto_write` is enabled and the segment has a `path_hint`, the
    /// payload is written to disk (with optional pre-allocation) before
    /// returning.
    pub async fn next(&mut self) -> Option<ProducedSegment> {
        let segment = self.rx.recv().await?;

        self.segments_received += 1;
        self.bytes_written += segment.byte_len() as u64;

        if self.config.auto_write {
            if let Some(path) = &segment.path_hint {
                let path = path.clone();
                let data = segment.data.clone();
                let pre_alloc = self.config.pre_allocate;

                // Non-fatal: log and continue if the write fails.
                if let Err(e) = write_segment_to_disk(&path, &data, pre_alloc).await {
                    tracing::warn!("SegmentStream: failed to write segment to {:?}: {}", path, e);
                }
            }
        }

        Some(segment)
    }

    /// Drain all remaining segments without writing to disk, discarding them.
    ///
    /// Useful for graceful shutdown.
    pub async fn drain(&mut self) {
        while self.rx.recv().await.is_some() {
            // discard
        }
    }

    /// Total bytes received (written) since the stream was created.
    #[must_use]
    pub fn bytes_written(&self) -> u64 {
        self.bytes_written
    }

    /// Number of segments received since the stream was created.
    #[must_use]
    pub fn segments_received(&self) -> u64 {
        self.segments_received
    }
}

// ---------------------------------------------------------------------------
// Disk writer
// ---------------------------------------------------------------------------

/// Write segment data to disk, optionally pre-allocating the file.
async fn write_segment_to_disk(
    path: &std::path::Path,
    data: &[u8],
    pre_allocate: bool,
) -> std::io::Result<()> {
    use tokio::io::AsyncWriteExt;

    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    let file = tokio::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(path)
        .await?;

    if pre_allocate && !data.is_empty() {
        // Use `set_len` to pre-allocate the file extent, reducing fragmentation.
        file.set_len(data.len() as u64).await?;
    }

    let mut writer = tokio::io::BufWriter::new(file);
    writer.write_all(data).await?;
    writer.flush().await?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_produced_segment_media() {
        let seg = ProducedSegment::media(1, vec![0xAB; 100], 6.0);
        assert_eq!(seg.sequence, 1);
        assert_eq!(seg.byte_len(), 100);
        assert!(!seg.is_init);
        assert_eq!(seg.duration_secs, 6.0);
    }

    #[test]
    fn test_produced_segment_init() {
        let init = ProducedSegment::init(vec![0xFF; 256]);
        assert!(init.is_init);
        assert_eq!(init.sequence, 0);
        assert_eq!(init.byte_len(), 256);
    }

    #[test]
    fn test_produced_segment_with_path() {
        let seg = ProducedSegment::media(0, vec![], 6.0)
            .with_path(PathBuf::from("/tmp/seg0.ts"));
        assert!(seg.path_hint.is_some());
    }

    #[test]
    fn test_segment_stream_config_default() {
        let cfg = SegmentStreamConfig::default();
        assert_eq!(cfg.channel_depth, 8);
        assert!(cfg.auto_write);
        assert!(!cfg.pre_allocate);
    }

    #[tokio::test]
    async fn test_segment_stream_send_recv() {
        let config = SegmentStreamConfig {
            auto_write: false, // disable disk writes in tests
            ..Default::default()
        };
        let (mut stream, tx) = SegmentStream::new(config);

        let seg = ProducedSegment::media(0, vec![1, 2, 3], 6.0);
        tx.send(seg).await.expect("send ok");
        drop(tx);

        let received = stream.next().await.expect("should receive segment");
        assert_eq!(received.sequence, 0);
        assert_eq!(received.data, vec![1, 2, 3]);

        assert!(stream.next().await.is_none());
        assert_eq!(stream.segments_received(), 1);
        assert_eq!(stream.bytes_written(), 3);
    }

    #[tokio::test]
    async fn test_segment_stream_multiple_segments() {
        let config = SegmentStreamConfig {
            auto_write: false,
            ..Default::default()
        };
        let (mut stream, tx) = SegmentStream::new(config);

        for i in 0..5u64 {
            let seg = ProducedSegment::media(i, vec![i as u8; 100], 6.0);
            tx.send(seg).await.expect("send ok");
        }
        drop(tx);

        let mut count = 0u64;
        while let Some(seg) = stream.next().await {
            assert_eq!(seg.sequence, count);
            count += 1;
        }
        assert_eq!(count, 5);
        assert_eq!(stream.segments_received(), 5);
    }

    #[tokio::test]
    async fn test_segment_stream_drain() {
        let config = SegmentStreamConfig {
            auto_write: false,
            ..Default::default()
        };
        let (mut stream, tx) = SegmentStream::new(config);

        for i in 0..3u64 {
            tx.send(ProducedSegment::media(i, vec![0; 10], 6.0))
                .await
                .expect("send ok");
        }
        drop(tx);

        stream.drain().await;
        // After drain the channel is exhausted; next() should return None immediately.
        assert!(stream.next().await.is_none());
    }

    #[tokio::test]
    async fn test_sender_try_send() {
        let config = SegmentStreamConfig {
            channel_depth: 2,
            auto_write: false,
            ..Default::default()
        };
        let (_stream, tx) = SegmentStream::new(config);

        // With depth 2, both sends should succeed.
        assert!(tx
            .try_send(ProducedSegment::media(0, vec![], 0.0))
            .is_ok());
        assert!(tx
            .try_send(ProducedSegment::media(1, vec![], 0.0))
            .is_ok());
        // Third send should fail (full).
        assert!(tx
            .try_send(ProducedSegment::media(2, vec![], 0.0))
            .is_err());
    }

    #[tokio::test]
    async fn test_auto_write_with_path() {
        let tmp_dir = std::env::temp_dir().join("oximedia_stream_test");
        tokio::fs::create_dir_all(&tmp_dir)
            .await
            .expect("create dir");

        let path = tmp_dir.join("seg0.ts");
        let data = vec![0xDE, 0xAD, 0xBE, 0xEF];

        let config = SegmentStreamConfig {
            auto_write: true,
            pre_allocate: false,
            ..Default::default()
        };
        let (mut stream, tx) = SegmentStream::new(config);

        let seg = ProducedSegment::media(0, data.clone(), 6.0).with_path(path.clone());
        tx.send(seg).await.expect("send ok");
        drop(tx);

        stream.next().await.expect("segment");

        // Verify the file was written.
        let written = tokio::fs::read(&path).await.expect("read written file");
        assert_eq!(written, data);

        // Cleanup.
        let _ = tokio::fs::remove_file(&path).await;
    }
}
