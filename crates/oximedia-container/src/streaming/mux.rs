//! Streaming muxer for live output.
//!
//! Provides progressive muxing without pre-buffering,
//! optimized for live streaming scenarios.

#![forbid(unsafe_code)]

use async_trait::async_trait;
use oximedia_core::OxiResult;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::time::Instant;

use crate::{Muxer, MuxerConfig, Packet, StreamInfo};

/// Configuration for streaming muxer.
#[derive(Clone, Debug)]
pub struct StreamingMuxerConfig {
    /// Target latency in milliseconds.
    pub target_latency_ms: u64,
    /// Enable low-latency mode (no buffering).
    pub low_latency: bool,
    /// Fragment duration in milliseconds (for fragmented formats).
    pub fragment_duration_ms: Option<u64>,
    /// Enable real-time mode (enforce timing).
    pub realtime: bool,
}

impl Default for StreamingMuxerConfig {
    fn default() -> Self {
        Self {
            target_latency_ms: 1000,
            low_latency: false,
            fragment_duration_ms: None,
            realtime: false,
        }
    }
}

impl StreamingMuxerConfig {
    /// Creates a new configuration with default values.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            target_latency_ms: 1000,
            low_latency: false,
            fragment_duration_ms: None,
            realtime: false,
        }
    }

    /// Enables low-latency mode.
    #[must_use]
    pub const fn with_low_latency(mut self, enabled: bool) -> Self {
        self.low_latency = enabled;
        self
    }

    /// Sets the target latency.
    #[must_use]
    pub const fn with_target_latency(mut self, latency_ms: u64) -> Self {
        self.target_latency_ms = latency_ms;
        self
    }

    /// Sets the fragment duration.
    #[must_use]
    pub const fn with_fragment_duration(mut self, duration_ms: u64) -> Self {
        self.fragment_duration_ms = Some(duration_ms);
        self
    }

    /// Enables real-time mode.
    #[must_use]
    pub const fn with_realtime(mut self, enabled: bool) -> Self {
        self.realtime = enabled;
        self
    }
}

/// Wrapper that adds streaming capabilities to any muxer.
pub struct StreamingMuxer<M: Muxer> {
    inner: M,
    #[allow(dead_code)]
    streaming_config: StreamingMuxerConfig,
    packets_written: u64,
    bytes_written: u64,
    start_time: Option<Instant>,
    last_packet_time: Option<Instant>,
}

impl<M: Muxer> StreamingMuxer<M> {
    /// Creates a new streaming muxer with default configuration.
    pub const fn new(inner: M) -> Self {
        Self::with_config(inner, StreamingMuxerConfig::new())
    }

    /// Creates a new streaming muxer with custom configuration.
    pub const fn with_config(inner: M, streaming_config: StreamingMuxerConfig) -> Self {
        Self {
            inner,
            streaming_config,
            packets_written: 0,
            bytes_written: 0,
            start_time: None,
            last_packet_time: None,
        }
    }

    /// Returns the number of packets written.
    #[must_use]
    pub const fn packets_written(&self) -> u64 {
        self.packets_written
    }

    /// Returns the number of bytes written.
    #[must_use]
    pub const fn bytes_written(&self) -> u64 {
        self.bytes_written
    }

    /// Returns the elapsed time since muxing started.
    #[must_use]
    pub fn elapsed(&self) -> Option<Duration> {
        self.start_time.map(|start| start.elapsed())
    }

    /// Returns a reference to the inner muxer.
    #[must_use]
    pub const fn inner(&self) -> &M {
        &self.inner
    }

    /// Returns a mutable reference to the inner muxer.
    pub fn inner_mut(&mut self) -> &mut M {
        &mut self.inner
    }

    /// Unwraps and returns the inner muxer.
    #[must_use]
    pub fn into_inner(self) -> M {
        self.inner
    }
}

#[async_trait]
impl<M: Muxer> Muxer for StreamingMuxer<M> {
    fn add_stream(&mut self, info: StreamInfo) -> OxiResult<usize> {
        self.inner.add_stream(info)
    }

    async fn write_header(&mut self) -> OxiResult<()> {
        self.start_time = Some(Instant::now());
        self.inner.write_header().await
    }

    async fn write_packet(&mut self, packet: &Packet) -> OxiResult<()> {
        let now = Instant::now();
        self.last_packet_time = Some(now);
        self.packets_written += 1;
        self.bytes_written += packet.size() as u64;
        self.inner.write_packet(packet).await
    }

    async fn write_trailer(&mut self) -> OxiResult<()> {
        self.inner.write_trailer().await
    }

    fn streams(&self) -> &[StreamInfo] {
        self.inner.streams()
    }

    fn config(&self) -> &MuxerConfig {
        self.inner.config()
    }
}

/// Packet sender for background muxing.
pub struct PacketSender {
    tx: mpsc::UnboundedSender<Packet>,
}

impl PacketSender {
    /// Creates a new packet sender.
    const fn new(tx: mpsc::UnboundedSender<Packet>) -> Self {
        Self { tx }
    }

    /// Sends a packet to the background muxer.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the background muxer task has terminated.
    pub fn send(&self, packet: Packet) -> Result<(), mpsc::error::SendError<Packet>> {
        self.tx.send(packet)
    }

    /// Tries to send a packet without blocking.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the background muxer task has terminated.
    pub fn try_send(&self, packet: Packet) -> Result<(), mpsc::error::SendError<Packet>> {
        self.tx.send(packet)
    }
}

/// Spawns a background task for muxing.
///
/// This function creates a background task that continuously receives packets
/// from a channel and writes them to the muxer. This is useful for streaming
/// scenarios where you want to decouple packet production from muxing.
///
/// # Arguments
///
/// * `muxer` - The muxer to run in the background
///
/// # Returns
///
/// A `PacketSender` that can be used to send packets to the background task.
///
/// # Errors
///
/// Returns `Err` if writing the container header fails.
///
/// # Examples
///
/// ```ignore
/// let muxer = MatroskaMuxer::new(sink, config);
/// let sender = spawn_muxer(muxer).await?;
///
/// for packet in packets {
///     sender.send(packet)?;
/// }
/// ```
pub async fn spawn_muxer<M: Muxer + Send + 'static>(mut muxer: M) -> OxiResult<PacketSender> {
    // Write header first
    muxer.write_header().await?;

    let (tx, mut rx) = mpsc::unbounded_channel();

    tokio::spawn(async move {
        while let Some(packet) = rx.recv().await {
            if muxer.write_packet(&packet).await.is_err() {
                break;
            }
        }
        let _ = muxer.write_trailer().await;
    });

    Ok(PacketSender::new(tx))
}

/// Statistics for streaming muxing.
#[derive(Debug, Clone, Copy, Default)]
pub struct MuxingStats {
    /// Total packets written.
    pub packets_written: u64,
    /// Total bytes written.
    pub bytes_written: u64,
    /// Average bitrate in bits per second.
    pub avg_bitrate: f64,
    /// Current bitrate in bits per second.
    pub current_bitrate: f64,
    /// Total duration in seconds.
    pub duration_secs: f64,
}

impl MuxingStats {
    /// Creates new statistics.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            packets_written: 0,
            bytes_written: 0,
            avg_bitrate: 0.0,
            current_bitrate: 0.0,
            duration_secs: 0.0,
        }
    }

    /// Updates statistics with a new packet.
    pub fn update(&mut self, packet_size: usize, duration_secs: f64) {
        self.packets_written += 1;
        self.bytes_written += packet_size as u64;
        self.duration_secs = duration_secs;

        if duration_secs > 0.0 {
            #[allow(clippy::cast_precision_loss)]
            {
                self.avg_bitrate = (self.bytes_written as f64 * 8.0) / duration_secs;
            }
        }
    }

    /// Sets the current bitrate.
    pub fn set_current_bitrate(&mut self, bitrate: f64) {
        self.current_bitrate = bitrate;
    }
}

/// Latency monitor for streaming.
#[derive(Debug)]
pub struct LatencyMonitor {
    target_latency: Duration,
    measurements: Vec<Duration>,
    max_measurements: usize,
}

impl LatencyMonitor {
    /// Creates a new latency monitor.
    #[must_use]
    pub fn new(target_latency: Duration) -> Self {
        Self {
            target_latency,
            measurements: Vec::with_capacity(100),
            max_measurements: 100,
        }
    }

    /// Records a latency measurement.
    pub fn record(&mut self, latency: Duration) {
        if self.measurements.len() >= self.max_measurements {
            self.measurements.remove(0);
        }
        self.measurements.push(latency);
    }

    /// Returns the average latency.
    #[must_use]
    pub fn average_latency(&self) -> Option<Duration> {
        if self.measurements.is_empty() {
            return None;
        }

        let sum: Duration = self.measurements.iter().sum();
        #[allow(clippy::cast_possible_truncation)]
        let count = self.measurements.len() as u32;
        Some(sum / count)
    }

    /// Returns true if latency is within target.
    #[must_use]
    pub fn is_within_target(&self) -> bool {
        self.average_latency()
            .map_or(true, |avg| avg <= self.target_latency)
    }

    /// Returns the target latency.
    #[must_use]
    pub const fn target_latency(&self) -> Duration {
        self.target_latency
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_config_default() {
        let config = StreamingMuxerConfig::default();
        assert_eq!(config.target_latency_ms, 1000);
        assert!(!config.low_latency);
        assert!(config.fragment_duration_ms.is_none());
        assert!(!config.realtime);
    }

    #[test]
    fn test_streaming_config_builder() {
        let config = StreamingMuxerConfig::new()
            .with_low_latency(true)
            .with_target_latency(500)
            .with_fragment_duration(2000)
            .with_realtime(true);

        assert!(config.low_latency);
        assert_eq!(config.target_latency_ms, 500);
        assert_eq!(config.fragment_duration_ms, Some(2000));
        assert!(config.realtime);
    }

    #[test]
    fn test_muxing_stats() {
        let mut stats = MuxingStats::new();
        assert_eq!(stats.packets_written, 0);
        assert_eq!(stats.bytes_written, 0);

        stats.update(1000, 1.0);
        assert_eq!(stats.packets_written, 1);
        assert_eq!(stats.bytes_written, 1000);
        assert!(stats.avg_bitrate > 0.0);

        stats.update(2000, 2.0);
        assert_eq!(stats.packets_written, 2);
        assert_eq!(stats.bytes_written, 3000);
    }

    #[test]
    fn test_latency_monitor() {
        let mut monitor = LatencyMonitor::new(Duration::from_millis(100));

        monitor.record(Duration::from_millis(50));
        monitor.record(Duration::from_millis(60));
        monitor.record(Duration::from_millis(70));

        let avg = monitor.average_latency().expect("operation should succeed");
        assert!(avg >= Duration::from_millis(59) && avg <= Duration::from_millis(61));
        assert!(monitor.is_within_target());
    }
}
