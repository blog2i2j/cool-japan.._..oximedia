//! Lock-free ring-queue stream buffer for audio frame pipelines.
#![allow(dead_code)]

use std::collections::VecDeque;

/// Configuration for a [`StreamBuffer`].
#[derive(Debug, Clone, Copy)]
pub struct StreamBufferConfig {
    /// Maximum number of frames the buffer may hold.
    pub max_frames: usize,
    /// Sample rate in Hz (used for duration calculations).
    pub sample_rate: u32,
    /// Number of channels per frame.
    pub channels: u16,
}

impl StreamBufferConfig {
    /// Create a new configuration.
    #[must_use]
    pub fn new(max_frames: usize, sample_rate: u32, channels: u16) -> Self {
        Self {
            max_frames,
            sample_rate,
            channels,
        }
    }

    /// Maximum queue depth expressed in frames.
    #[must_use]
    pub fn max_frames(&self) -> usize {
        self.max_frames
    }
}

impl Default for StreamBufferConfig {
    fn default() -> Self {
        Self {
            max_frames: 64,
            sample_rate: 48_000,
            channels: 2,
        }
    }
}

/// A single audio frame inside the stream buffer.
#[derive(Debug, Clone)]
pub struct StreamFrame {
    /// Interleaved PCM samples (f32).
    pub samples: Vec<f32>,
    /// Presentation timestamp in samples since stream start.
    pub pts_samples: u64,
    /// Number of channels in this frame.
    pub channels: u16,
    /// Sample rate of the frame (Hz).
    pub sample_rate: u32,
}

impl StreamFrame {
    /// Create a new frame.
    #[must_use]
    pub fn new(samples: Vec<f32>, pts_samples: u64, channels: u16, sample_rate: u32) -> Self {
        Self {
            samples,
            pts_samples,
            channels,
            sample_rate,
        }
    }

    /// Number of multi-channel audio samples (frames) in this buffer.
    ///
    /// That is, the length of `samples` divided by the channel count.
    #[must_use]
    pub fn sample_count(&self) -> usize {
        if self.channels == 0 {
            return 0;
        }
        self.samples.len() / self.channels as usize
    }

    /// Duration of this frame in milliseconds.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn duration_ms(&self) -> f64 {
        if self.sample_rate == 0 {
            return 0.0;
        }
        self.sample_count() as f64 / self.sample_rate as f64 * 1_000.0
    }
}

/// FIFO queue of [`StreamFrame`]s with a configurable capacity.
#[derive(Debug)]
pub struct StreamBuffer {
    queue: VecDeque<StreamFrame>,
    config: StreamBufferConfig,
    /// Total number of frames ever pushed (monotonically increasing).
    total_pushed: u64,
    /// Total number of frames ever popped.
    total_popped: u64,
}

impl StreamBuffer {
    /// Create a new stream buffer with the given configuration.
    #[must_use]
    pub fn new(config: StreamBufferConfig) -> Self {
        Self {
            queue: VecDeque::with_capacity(config.max_frames),
            config,
            total_pushed: 0,
            total_popped: 0,
        }
    }

    /// Push a frame into the buffer.
    ///
    /// Returns `false` (and discards the frame) when the buffer is full.
    pub fn push_frame(&mut self, frame: StreamFrame) -> bool {
        if self.queue.len() >= self.config.max_frames {
            return false;
        }
        self.queue.push_back(frame);
        self.total_pushed += 1;
        true
    }

    /// Pop the oldest frame from the buffer, or `None` if empty.
    pub fn pop_frame(&mut self) -> Option<StreamFrame> {
        let frame = self.queue.pop_front();
        if frame.is_some() {
            self.total_popped += 1;
        }
        frame
    }

    /// Approximate total buffered audio in milliseconds.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn duration_ms(&self) -> f64 {
        self.queue.iter().map(|f| f.duration_ms()).sum()
    }

    /// Number of frames currently queued.
    #[must_use]
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Returns `true` when no frames are queued.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Returns `true` when the queue has reached its configured maximum.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.queue.len() >= self.config.max_frames
    }

    /// Total frames pushed since creation.
    #[must_use]
    pub fn total_pushed(&self) -> u64 {
        self.total_pushed
    }

    /// Total frames popped since creation.
    #[must_use]
    pub fn total_popped(&self) -> u64 {
        self.total_popped
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(n_samples: usize, pts: u64) -> StreamFrame {
        StreamFrame::new(
            vec![0.0_f32; n_samples * 2], // stereo
            pts,
            2,
            48_000,
        )
    }

    #[test]
    fn test_config_max_frames() {
        let cfg = StreamBufferConfig::new(32, 48_000, 2);
        assert_eq!(cfg.max_frames(), 32);
    }

    #[test]
    fn test_config_default() {
        let cfg = StreamBufferConfig::default();
        assert_eq!(cfg.max_frames, 64);
        assert_eq!(cfg.sample_rate, 48_000);
    }

    #[test]
    fn test_frame_sample_count() {
        let frame = make_frame(480, 0);
        assert_eq!(frame.sample_count(), 480);
    }

    #[test]
    fn test_frame_duration_ms() {
        let frame = make_frame(480, 0); // 480/48000 = 10ms
        let dur = frame.duration_ms();
        assert!((dur - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_frame_duration_zero_rate() {
        let frame = StreamFrame::new(vec![0.0; 4], 0, 2, 0);
        assert_eq!(frame.duration_ms(), 0.0);
    }

    #[test]
    fn test_buffer_push_and_pop() {
        let cfg = StreamBufferConfig::default();
        let mut buf = StreamBuffer::new(cfg);
        let f = make_frame(480, 0);
        assert!(buf.push_frame(f));
        assert_eq!(buf.len(), 1);
        let popped = buf.pop_frame();
        assert!(popped.is_some());
        assert!(buf.is_empty());
    }

    #[test]
    fn test_buffer_fifo_order() {
        let cfg = StreamBufferConfig::default();
        let mut buf = StreamBuffer::new(cfg);
        buf.push_frame(make_frame(480, 0));
        buf.push_frame(make_frame(480, 480));
        let first = buf.pop_frame().expect("should succeed");
        assert_eq!(first.pts_samples, 0);
        let second = buf.pop_frame().expect("should succeed");
        assert_eq!(second.pts_samples, 480);
    }

    #[test]
    fn test_buffer_full_rejects_push() {
        let cfg = StreamBufferConfig::new(2, 48_000, 2);
        let mut buf = StreamBuffer::new(cfg);
        assert!(buf.push_frame(make_frame(480, 0)));
        assert!(buf.push_frame(make_frame(480, 480)));
        assert!(buf.is_full());
        assert!(!buf.push_frame(make_frame(480, 960)));
    }

    #[test]
    fn test_buffer_is_empty_initially() {
        let buf = StreamBuffer::new(StreamBufferConfig::default());
        assert!(buf.is_empty());
    }

    #[test]
    fn test_buffer_pop_empty_returns_none() {
        let mut buf = StreamBuffer::new(StreamBufferConfig::default());
        assert!(buf.pop_frame().is_none());
    }

    #[test]
    fn test_buffer_duration_ms() {
        let cfg = StreamBufferConfig::default();
        let mut buf = StreamBuffer::new(cfg);
        buf.push_frame(make_frame(480, 0)); // 10ms
        buf.push_frame(make_frame(480, 480)); // 10ms
        let dur = buf.duration_ms();
        assert!((dur - 20.0).abs() < 0.01);
    }

    #[test]
    fn test_buffer_total_counters() {
        let cfg = StreamBufferConfig::default();
        let mut buf = StreamBuffer::new(cfg);
        buf.push_frame(make_frame(480, 0));
        buf.push_frame(make_frame(480, 480));
        buf.pop_frame();
        assert_eq!(buf.total_pushed(), 2);
        assert_eq!(buf.total_popped(), 1);
    }

    #[test]
    fn test_frame_zero_channels() {
        let frame = StreamFrame::new(vec![0.0; 10], 0, 0, 48_000);
        assert_eq!(frame.sample_count(), 0);
    }
}
