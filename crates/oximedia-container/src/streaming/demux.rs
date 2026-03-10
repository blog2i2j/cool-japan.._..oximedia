//! Streaming demuxer for network sources.
//!
//! Provides progressive demuxing without requiring seek support,
//! optimized for live streaming and network sources.

#![forbid(unsafe_code)]

use async_trait::async_trait;
use bytes::Bytes;
use oximedia_core::{OxiError, OxiResult};
use std::collections::VecDeque;
use tokio::sync::mpsc;

use crate::{Demuxer, Packet, ProbeResult, StreamInfo};

/// Configuration for streaming demuxer.
#[derive(Clone, Debug)]
pub struct StreamingDemuxerConfig {
    /// Initial buffer size in bytes before starting demux.
    pub initial_buffer_size: usize,
    /// Maximum buffer size in bytes.
    pub max_buffer_size: usize,
    /// Enable low-latency mode (minimal buffering).
    pub low_latency: bool,
    /// Timeout for network reads in milliseconds.
    pub read_timeout_ms: u64,
}

impl Default for StreamingDemuxerConfig {
    fn default() -> Self {
        Self {
            initial_buffer_size: 64 * 1024,    // 64 KB
            max_buffer_size: 10 * 1024 * 1024, // 10 MB
            low_latency: false,
            read_timeout_ms: 5000,
        }
    }
}

impl StreamingDemuxerConfig {
    /// Creates a new configuration with default values.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            initial_buffer_size: 64 * 1024,
            max_buffer_size: 10 * 1024 * 1024,
            low_latency: false,
            read_timeout_ms: 5000,
        }
    }

    /// Enables low-latency mode.
    #[must_use]
    pub const fn with_low_latency(mut self, enabled: bool) -> Self {
        self.low_latency = enabled;
        self
    }

    /// Sets the initial buffer size.
    #[must_use]
    pub const fn with_initial_buffer(mut self, size: usize) -> Self {
        self.initial_buffer_size = size;
        self
    }

    /// Sets the maximum buffer size.
    #[must_use]
    pub const fn with_max_buffer(mut self, size: usize) -> Self {
        self.max_buffer_size = size;
        self
    }

    /// Sets the read timeout.
    #[must_use]
    pub const fn with_timeout(mut self, timeout_ms: u64) -> Self {
        self.read_timeout_ms = timeout_ms;
        self
    }
}

/// State of the streaming demuxer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamingState {
    /// Initial state, waiting for data.
    Initializing,
    /// Buffering data.
    Buffering,
    /// Actively demuxing.
    Active,
    /// Underrun, waiting for more data.
    Underrun,
    /// End of stream reached.
    Eof,
}

/// Wrapper that adds streaming capabilities to any demuxer.
pub struct StreamingDemuxer<D: Demuxer> {
    inner: D,
    config: StreamingDemuxerConfig,
    #[allow(dead_code)]
    buffer: VecDeque<u8>,
    state: StreamingState,
    bytes_buffered: usize,
    packets_read: u64,
}

impl<D: Demuxer> StreamingDemuxer<D> {
    /// Creates a new streaming demuxer with default configuration.
    pub fn new(inner: D) -> Self {
        Self::with_config(inner, StreamingDemuxerConfig::default())
    }

    /// Creates a new streaming demuxer with custom configuration.
    pub fn with_config(inner: D, config: StreamingDemuxerConfig) -> Self {
        let buffer_size = config.initial_buffer_size;
        Self {
            inner,
            config,
            buffer: VecDeque::with_capacity(buffer_size),
            state: StreamingState::Initializing,
            bytes_buffered: 0,
            packets_read: 0,
        }
    }

    /// Returns the current state.
    #[must_use]
    pub const fn state(&self) -> StreamingState {
        self.state
    }

    /// Returns the number of bytes currently buffered.
    #[must_use]
    pub const fn bytes_buffered(&self) -> usize {
        self.bytes_buffered
    }

    /// Returns the number of packets read so far.
    #[must_use]
    pub const fn packets_read(&self) -> u64 {
        self.packets_read
    }

    /// Returns a reference to the inner demuxer.
    #[must_use]
    pub const fn inner(&self) -> &D {
        &self.inner
    }

    /// Returns a mutable reference to the inner demuxer.
    pub fn inner_mut(&mut self) -> &mut D {
        &mut self.inner
    }

    /// Unwraps and returns the inner demuxer.
    #[must_use]
    pub fn into_inner(self) -> D {
        self.inner
    }

    /// Checks if buffering is needed.
    fn needs_buffering(&self) -> bool {
        if self.config.low_latency {
            return false;
        }
        self.bytes_buffered < self.config.initial_buffer_size
    }

    /// Updates the state based on buffer level.
    fn update_state(&mut self) {
        if self.bytes_buffered == 0 {
            if self.state == StreamingState::Eof {
                return;
            }
            self.state = StreamingState::Underrun;
        } else if self.needs_buffering() {
            self.state = StreamingState::Buffering;
        } else {
            self.state = StreamingState::Active;
        }
    }
}

#[async_trait]
impl<D: Demuxer> Demuxer for StreamingDemuxer<D> {
    async fn probe(&mut self) -> OxiResult<ProbeResult> {
        self.state = StreamingState::Initializing;
        let result = self.inner.probe().await?;
        self.update_state();
        Ok(result)
    }

    async fn read_packet(&mut self) -> OxiResult<Packet> {
        // Check if we need to buffer more data
        if self.needs_buffering() && self.state != StreamingState::Eof {
            self.state = StreamingState::Buffering;
            // In a real implementation, we would read from the source here
            // For now, we just delegate to the inner demuxer
        }

        match self.inner.read_packet().await {
            Ok(packet) => {
                self.packets_read += 1;
                self.state = StreamingState::Active;
                Ok(packet)
            }
            Err(OxiError::Eof) => {
                self.state = StreamingState::Eof;
                Err(OxiError::Eof)
            }
            Err(e) => {
                self.state = StreamingState::Underrun;
                Err(e)
            }
        }
    }

    fn streams(&self) -> &[StreamInfo] {
        self.inner.streams()
    }

    fn is_seekable(&self) -> bool {
        // Streaming demuxers are not seekable
        false
    }
}

/// Async packet receiver for background demuxing.
pub struct PacketReceiver {
    rx: mpsc::UnboundedReceiver<OxiResult<Packet>>,
    streams: Vec<StreamInfo>,
}

impl PacketReceiver {
    /// Creates a new packet receiver.
    fn new(rx: mpsc::UnboundedReceiver<OxiResult<Packet>>, streams: Vec<StreamInfo>) -> Self {
        Self { rx, streams }
    }

    /// Receives the next packet.
    pub async fn recv(&mut self) -> Option<OxiResult<Packet>> {
        self.rx.recv().await
    }

    /// Returns stream information.
    #[must_use]
    pub fn streams(&self) -> &[StreamInfo] {
        &self.streams
    }

    /// Tries to receive a packet without blocking.
    ///
    /// # Errors
    ///
    /// Returns `Err(TryRecvError)` if no packet is available or the channel is closed.
    pub fn try_recv(&mut self) -> Result<OxiResult<Packet>, mpsc::error::TryRecvError> {
        self.rx.try_recv()
    }
}

/// Spawns a background task for demuxing.
///
/// This function creates a background task that continuously reads packets
/// from the demuxer and sends them through a channel. This is useful for
/// streaming scenarios where you want to decouple demuxing from processing.
///
/// # Arguments
///
/// * `demuxer` - The demuxer to run in the background
///
/// # Returns
///
/// A `PacketReceiver` that can be used to receive packets from the background task.
///
/// # Errors
///
/// Returns `Err` if the demuxer fails during probing.
///
/// # Examples
///
/// ```ignore
/// let demuxer = MatroskaDemuxer::new(source);
/// let mut receiver = spawn_demuxer(demuxer).await?;
///
/// while let Some(result) = receiver.recv().await {
///     match result {
///         Ok(packet) => process_packet(packet),
///         Err(e) => handle_error(e),
///     }
/// }
/// ```
pub async fn spawn_demuxer<D: Demuxer + Send + 'static>(
    mut demuxer: D,
) -> OxiResult<PacketReceiver> {
    // Probe the demuxer first
    demuxer.probe().await?;
    let streams = demuxer.streams().to_vec();

    let (tx, rx) = mpsc::unbounded_channel();

    tokio::spawn(async move {
        loop {
            match demuxer.read_packet().await {
                Ok(packet) => {
                    if tx.send(Ok(packet)).is_err() {
                        // Receiver dropped, exit
                        break;
                    }
                }
                Err(OxiError::Eof) => {
                    let _ = tx.send(Err(OxiError::Eof));
                    break;
                }
                Err(e) => {
                    let _ = tx.send(Err(e));
                    break;
                }
            }
        }
    });

    Ok(PacketReceiver::new(rx, streams))
}

/// Buffer for progressive data accumulation.
#[derive(Debug)]
pub struct ProgressiveBuffer {
    data: VecDeque<u8>,
    max_size: usize,
    total_received: u64,
}

impl ProgressiveBuffer {
    /// Creates a new progressive buffer.
    #[must_use]
    pub fn new(max_size: usize) -> Self {
        Self {
            data: VecDeque::with_capacity(max_size.min(64 * 1024)),
            max_size,
            total_received: 0,
        }
    }

    /// Appends data to the buffer.
    ///
    /// # Errors
    ///
    /// Returns `Err` if adding the data would exceed the maximum buffer size.
    pub fn append(&mut self, data: &[u8]) -> OxiResult<()> {
        if self.data.len() + data.len() > self.max_size {
            return Err(OxiError::BufferTooSmall {
                needed: self.data.len() + data.len(),
                have: self.max_size,
            });
        }
        self.data.extend(data);
        self.total_received += data.len() as u64;
        Ok(())
    }

    /// Consumes bytes from the front of the buffer.
    pub fn consume(&mut self, count: usize) -> Option<Bytes> {
        if count > self.data.len() {
            return None;
        }
        let bytes: Vec<u8> = self.data.drain(..count).collect();
        Some(Bytes::from(bytes))
    }

    /// Peeks at the front of the buffer without consuming.
    #[must_use]
    pub fn peek(&self, count: usize) -> Option<&[u8]> {
        if count > self.data.len() {
            return None;
        }
        // Convert VecDeque slices to a single slice if possible
        let (first, _second) = self.data.as_slices();
        if count <= first.len() {
            Some(&first[..count])
        } else {
            None // Data is fragmented across VecDeque halves
        }
    }

    /// Returns the number of bytes currently in the buffer.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns true if the buffer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns the total number of bytes received.
    #[must_use]
    pub const fn total_received(&self) -> u64 {
        self.total_received
    }

    /// Clears the buffer.
    pub fn clear(&mut self) {
        self.data.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = StreamingDemuxerConfig::default();
        assert_eq!(config.initial_buffer_size, 64 * 1024);
        assert_eq!(config.max_buffer_size, 10 * 1024 * 1024);
        assert!(!config.low_latency);
    }

    #[test]
    fn test_config_builder() {
        let config = StreamingDemuxerConfig::new()
            .with_low_latency(true)
            .with_initial_buffer(128 * 1024)
            .with_max_buffer(20 * 1024 * 1024)
            .with_timeout(10000);

        assert!(config.low_latency);
        assert_eq!(config.initial_buffer_size, 128 * 1024);
        assert_eq!(config.max_buffer_size, 20 * 1024 * 1024);
        assert_eq!(config.read_timeout_ms, 10000);
    }

    #[test]
    fn test_progressive_buffer() {
        let mut buffer = ProgressiveBuffer::new(1024);
        assert!(buffer.is_empty());
        assert_eq!(buffer.len(), 0);

        // Append data
        buffer
            .append(&[1, 2, 3, 4])
            .expect("operation should succeed");
        assert_eq!(buffer.len(), 4);
        assert_eq!(buffer.total_received(), 4);

        // Peek
        let peeked = buffer.peek(2).expect("operation should succeed");
        assert_eq!(peeked, &[1, 2]);
        assert_eq!(buffer.len(), 4); // Still has all data

        // Consume
        let consumed = buffer.consume(2).expect("operation should succeed");
        assert_eq!(consumed.as_ref(), &[1, 2]);
        assert_eq!(buffer.len(), 2);

        // Clear
        buffer.clear();
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_progressive_buffer_overflow() {
        let mut buffer = ProgressiveBuffer::new(10);
        assert!(buffer.append(&[1, 2, 3, 4, 5]).is_ok());
        assert!(buffer.append(&[6, 7, 8, 9, 10, 11]).is_err());
    }
}
