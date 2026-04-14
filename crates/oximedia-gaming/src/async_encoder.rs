#![allow(dead_code)]

//! Async encoder output with double-buffered frame submission.
//!
//! Implements a double-buffering scheme where the capture stage writes into
//! one buffer while the encoder processes the other. This decouples the
//! capture and encoding rates and allows the encoder to work asynchronously
//! without blocking the capture thread.
//!
//! # Architecture
//!
//! ```text
//!  Capture ──► [Front Buffer] ◄──swap──► [Back Buffer] ──► Encoder
//! ```
//!
//! When the capture stage fills the front buffer it signals a swap. The
//! encoder reads from the back buffer at its own pace. If the encoder is
//! still working when a swap is requested, the frame in the front buffer
//! is either held (blocking) or dropped depending on the configured policy.
//!
//! # Features
//!
//! - **Double buffering**: Two pre-allocated frame buffers with atomic swap
//! - **Encoding sessions**: Track per-frame encoding results and timing
//! - **Rate control integration**: VBR/CBR/CRF mode selection
//! - **Bitrate estimation**: Running average bitrate from encoded output
//! - **GOP management**: Keyframe interval enforcement
//! - **Encoder statistics**: Frames encoded, average/peak timing, output size

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use crate::{GamingError, GamingResult};

// ---------------------------------------------------------------------------
// Rate control mode
// ---------------------------------------------------------------------------

/// Rate control mode for the encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RateControlMode {
    /// Constant Bitrate — maintains a target bitrate.
    Cbr {
        /// Target bitrate in kbps.
        target_kbps: u32,
    },
    /// Variable Bitrate — fluctuates around a target.
    Vbr {
        /// Target bitrate in kbps.
        target_kbps: u32,
        /// Maximum allowed bitrate in kbps.
        max_kbps: u32,
    },
    /// Constant Rate Factor — quality-based.
    Crf {
        /// Quality factor (0 = lossless, 63 = worst).
        crf: u8,
    },
}

impl Default for RateControlMode {
    fn default() -> Self {
        Self::Vbr {
            target_kbps: 6000,
            max_kbps: 8000,
        }
    }
}

// ---------------------------------------------------------------------------
// Frame type
// ---------------------------------------------------------------------------

/// Encoded frame type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FrameType {
    /// Keyframe (IDR / intra).
    Key,
    /// Inter-predicted frame.
    Inter,
}

// ---------------------------------------------------------------------------
// Encoder configuration
// ---------------------------------------------------------------------------

/// Configuration for the async encoder.
#[derive(Debug, Clone)]
pub struct AsyncEncoderConfig {
    /// Frame resolution (width, height).
    pub resolution: (u32, u32),
    /// Target framerate.
    pub framerate: u32,
    /// Rate control mode.
    pub rate_control: RateControlMode,
    /// Keyframe interval in frames (GOP size).
    pub gop_size: u32,
    /// Maximum number of encoded packets to buffer before back-pressure.
    pub output_queue_capacity: usize,
    /// Drop policy when front buffer is full and encoder is still working.
    pub drop_on_backpressure: bool,
}

impl Default for AsyncEncoderConfig {
    fn default() -> Self {
        Self {
            resolution: (1920, 1080),
            framerate: 60,
            rate_control: RateControlMode::default(),
            gop_size: 120,
            output_queue_capacity: 16,
            drop_on_backpressure: true,
        }
    }
}

/// Builder for [`AsyncEncoderConfig`].
#[derive(Debug, Clone)]
pub struct AsyncEncoderConfigBuilder {
    inner: AsyncEncoderConfig,
}

impl Default for AsyncEncoderConfigBuilder {
    fn default() -> Self {
        Self {
            inner: AsyncEncoderConfig::default(),
        }
    }
}

impl AsyncEncoderConfigBuilder {
    /// Create a new builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set resolution.
    #[must_use]
    pub fn resolution(mut self, width: u32, height: u32) -> Self {
        self.inner.resolution = (width, height);
        self
    }

    /// Set framerate.
    #[must_use]
    pub fn framerate(mut self, fps: u32) -> Self {
        self.inner.framerate = fps;
        self
    }

    /// Set rate control mode.
    #[must_use]
    pub fn rate_control(mut self, mode: RateControlMode) -> Self {
        self.inner.rate_control = mode;
        self
    }

    /// Set GOP size.
    #[must_use]
    pub fn gop_size(mut self, gop: u32) -> Self {
        self.inner.gop_size = gop;
        self
    }

    /// Set output queue capacity.
    #[must_use]
    pub fn output_queue_capacity(mut self, cap: usize) -> Self {
        self.inner.output_queue_capacity = cap;
        self
    }

    /// Set drop-on-backpressure policy.
    #[must_use]
    pub fn drop_on_backpressure(mut self, drop: bool) -> Self {
        self.inner.drop_on_backpressure = drop;
        self
    }

    /// Build the configuration.
    ///
    /// # Errors
    ///
    /// Returns error for invalid settings.
    pub fn build(self) -> GamingResult<AsyncEncoderConfig> {
        let c = &self.inner;
        if c.resolution.0 == 0 || c.resolution.1 == 0 {
            return Err(GamingError::InvalidConfig(
                "Resolution must be non-zero".into(),
            ));
        }
        if c.framerate == 0 || c.framerate > 240 {
            return Err(GamingError::InvalidConfig("Framerate must be 1–240".into()));
        }
        if c.gop_size == 0 {
            return Err(GamingError::InvalidConfig("GOP size must be >= 1".into()));
        }
        if c.output_queue_capacity == 0 {
            return Err(GamingError::InvalidConfig(
                "Output queue capacity must be >= 1".into(),
            ));
        }
        Ok(self.inner)
    }
}

// ---------------------------------------------------------------------------
// Double buffer
// ---------------------------------------------------------------------------

/// A double-buffered frame holder. Two pre-allocated buffers alternate
/// between capture (front) and encode (back) roles.
#[derive(Debug)]
struct DoubleBuffer {
    /// The two buffers.
    buffers: [Vec<u8>; 2],
    /// Metadata for each buffer.
    meta: [BufferMeta; 2],
    /// Index of the current front (capture) buffer: 0 or 1.
    front: usize,
}

/// Metadata associated with a buffer.
#[derive(Debug, Clone)]
struct BufferMeta {
    /// Whether this buffer contains a valid frame.
    filled: bool,
    /// Frame width.
    width: u32,
    /// Frame height.
    height: u32,
    /// Presentation timestamp in microseconds.
    pts_us: u64,
    /// Monotonic frame number.
    frame_number: u64,
    /// When the frame was captured.
    capture_time: Option<Instant>,
}

impl Default for BufferMeta {
    fn default() -> Self {
        Self {
            filled: false,
            width: 0,
            height: 0,
            pts_us: 0,
            frame_number: 0,
            capture_time: None,
        }
    }
}

impl DoubleBuffer {
    /// Allocate a double buffer with the given per-buffer byte size.
    fn new(buf_size: usize) -> Self {
        Self {
            buffers: [vec![0u8; buf_size], vec![0u8; buf_size]],
            meta: [BufferMeta::default(), BufferMeta::default()],
            front: 0,
        }
    }

    /// Write frame data into the front buffer.
    fn write_front(
        &mut self,
        data: &[u8],
        width: u32,
        height: u32,
        pts_us: u64,
        frame_number: u64,
    ) {
        let f = self.front;
        if data.len() > self.buffers[f].len() {
            self.buffers[f].resize(data.len(), 0);
        }
        self.buffers[f][..data.len()].copy_from_slice(data);
        self.meta[f] = BufferMeta {
            filled: true,
            width,
            height,
            pts_us,
            frame_number,
            capture_time: Some(Instant::now()),
        };
    }

    /// Swap front and back buffers.
    fn swap(&mut self) {
        self.front = 1 - self.front;
    }

    /// Index of the back buffer.
    fn back_index(&self) -> usize {
        1 - self.front
    }

    /// Check if the back buffer has a filled frame.
    fn back_is_filled(&self) -> bool {
        self.meta[self.back_index()].filled
    }

    /// Read the back buffer data (for encoding).
    fn read_back(&self) -> Option<(&[u8], &BufferMeta)> {
        let b = self.back_index();
        if self.meta[b].filled {
            let len = (self.meta[b].width as usize) * (self.meta[b].height as usize) * 4;
            let end = len.min(self.buffers[b].len());
            Some((&self.buffers[b][..end], &self.meta[b]))
        } else {
            None
        }
    }

    /// Mark the back buffer as consumed.
    fn consume_back(&mut self) {
        let b = self.back_index();
        self.meta[b].filled = false;
    }
}

// ---------------------------------------------------------------------------
// Encoded packet
// ---------------------------------------------------------------------------

/// An encoded output packet.
#[derive(Debug, Clone)]
pub struct EncodedPacket {
    /// Encoded data bytes.
    pub data: Vec<u8>,
    /// Presentation timestamp in microseconds.
    pub pts_us: u64,
    /// Frame type (key or inter).
    pub frame_type: FrameType,
    /// Frame number.
    pub frame_number: u64,
    /// Time spent encoding this frame.
    pub encode_time: Duration,
    /// Size in bytes.
    pub size_bytes: usize,
}

// ---------------------------------------------------------------------------
// Encoder statistics
// ---------------------------------------------------------------------------

/// Statistics collected by the async encoder.
#[derive(Debug, Clone)]
pub struct EncoderStats {
    /// Total frames submitted to the encoder.
    pub frames_submitted: u64,
    /// Total frames encoded.
    pub frames_encoded: u64,
    /// Total frames dropped due to back-pressure.
    pub frames_dropped: u64,
    /// Total bytes output.
    pub total_bytes: u64,
    /// Average encoding time per frame.
    pub avg_encode_time: Duration,
    /// Peak encoding time for a single frame.
    pub peak_encode_time: Duration,
    /// Current estimated bitrate in kbps.
    pub estimated_bitrate_kbps: u32,
    /// Number of keyframes emitted.
    pub keyframes_emitted: u64,
}

// ---------------------------------------------------------------------------
// AsyncEncoder
// ---------------------------------------------------------------------------

/// Async encoder with double-buffered frame submission.
///
/// The encoder simulates encoding by generating placeholder output packets
/// sized according to the configured rate control mode. A real integration
/// would call into `oximedia-codec` here.
#[derive(Debug)]
pub struct AsyncEncoder {
    config: AsyncEncoderConfig,
    double_buf: DoubleBuffer,
    /// Output packet queue.
    output_queue: VecDeque<EncodedPacket>,
    /// Running frame counter.
    frame_counter: u64,
    /// Frames since last keyframe.
    frames_since_keyframe: u32,
    // -- stats --
    frames_submitted: u64,
    frames_encoded: u64,
    frames_dropped: u64,
    total_bytes: u64,
    total_encode_time: Duration,
    peak_encode_time: Duration,
    keyframes_emitted: u64,
    /// Window of recent encoded sizes for bitrate estimation.
    recent_sizes: VecDeque<(Instant, u64)>,
}

impl AsyncEncoder {
    /// Create a new async encoder.
    ///
    /// # Errors
    ///
    /// Returns error if configuration is invalid.
    pub fn new(config: AsyncEncoderConfig) -> GamingResult<Self> {
        if config.resolution.0 == 0 || config.resolution.1 == 0 {
            return Err(GamingError::InvalidConfig(
                "Resolution must be non-zero".into(),
            ));
        }
        let buf_size = (config.resolution.0 as usize) * (config.resolution.1 as usize) * 4;
        Ok(Self {
            double_buf: DoubleBuffer::new(buf_size),
            config,
            output_queue: VecDeque::new(),
            frame_counter: 0,
            frames_since_keyframe: 0,
            frames_submitted: 0,
            frames_encoded: 0,
            frames_dropped: 0,
            total_bytes: 0,
            total_encode_time: Duration::ZERO,
            peak_encode_time: Duration::ZERO,
            keyframes_emitted: 0,
            recent_sizes: VecDeque::new(),
        })
    }

    /// Submit a raw RGBA frame for encoding.
    ///
    /// The frame is written into the front buffer. If the back buffer still
    /// contains an un-consumed frame, a swap is attempted. If that is not
    /// possible and `drop_on_backpressure` is enabled, the oldest frame is
    /// dropped.
    ///
    /// # Errors
    ///
    /// Returns error if frame dimensions are wrong.
    pub fn submit_frame(
        &mut self,
        data: &[u8],
        width: u32,
        height: u32,
        pts_us: u64,
    ) -> GamingResult<()> {
        let expected = (width as usize) * (height as usize) * 4;
        if data.len() != expected {
            return Err(GamingError::InvalidConfig(format!(
                "Frame data {} != expected {} ({}x{}x4)",
                data.len(),
                expected,
                width,
                height
            )));
        }

        self.frames_submitted += 1;
        let frame_num = self.frame_counter;
        self.frame_counter += 1;

        // Write into front buffer.
        self.double_buf
            .write_front(data, width, height, pts_us, frame_num);

        // Attempt swap.
        if !self.double_buf.back_is_filled() {
            self.double_buf.swap();
        } else if self.config.drop_on_backpressure {
            // Drop the back buffer's frame and swap.
            self.double_buf.consume_back();
            self.frames_dropped += 1;
            self.double_buf.swap();
        }
        // else: front buffer holds the frame until `encode_pending` is called.

        Ok(())
    }

    /// Encode the frame currently in the back buffer (if any).
    ///
    /// This is the "async" step — in a real system this would be dispatched
    /// to a thread pool or GPU queue. Here we perform a synchronous
    /// simulation that produces a sized packet.
    ///
    /// Returns `true` if a frame was encoded.
    pub fn encode_pending(&mut self) -> bool {
        let (back_data_len, meta) = match self.double_buf.read_back() {
            Some((data, meta)) => (data.len(), meta.clone()),
            None => return false,
        };

        let start = Instant::now();

        // Determine frame type.
        let frame_type =
            if self.frames_since_keyframe >= self.config.gop_size || self.frames_encoded == 0 {
                self.frames_since_keyframe = 0;
                self.keyframes_emitted += 1;
                FrameType::Key
            } else {
                FrameType::Inter
            };
        self.frames_since_keyframe += 1;

        // Estimate output size based on rate control.
        let output_size = self.estimate_output_size(frame_type, back_data_len);

        // Simulate encoded data.
        let encoded_data = vec![0u8; output_size];
        let encode_time = start.elapsed();

        let packet = EncodedPacket {
            data: encoded_data,
            pts_us: meta.pts_us,
            frame_type,
            frame_number: meta.frame_number,
            encode_time,
            size_bytes: output_size,
        };

        // Enqueue output, dropping oldest if at capacity.
        if self.output_queue.len() >= self.config.output_queue_capacity {
            self.output_queue.pop_front();
        }
        self.output_queue.push_back(packet);

        // Update stats.
        self.frames_encoded += 1;
        self.total_bytes += output_size as u64;
        self.total_encode_time += encode_time;
        if encode_time > self.peak_encode_time {
            self.peak_encode_time = encode_time;
        }

        // Track for bitrate estimation.
        self.recent_sizes
            .push_back((Instant::now(), output_size as u64));
        // Keep only last 2 seconds of data.
        let cutoff = Instant::now() - Duration::from_secs(2);
        while self.recent_sizes.front().is_some_and(|(t, _)| *t < cutoff) {
            self.recent_sizes.pop_front();
        }

        self.double_buf.consume_back();
        true
    }

    /// Dequeue the next encoded packet.
    pub fn poll_output(&mut self) -> Option<EncodedPacket> {
        self.output_queue.pop_front()
    }

    /// Peek at the next encoded packet without removing it.
    #[must_use]
    pub fn peek_output(&self) -> Option<&EncodedPacket> {
        self.output_queue.front()
    }

    /// Number of encoded packets waiting in the output queue.
    #[must_use]
    pub fn output_queue_len(&self) -> usize {
        self.output_queue.len()
    }

    /// Force a keyframe on the next encode call.
    pub fn force_keyframe(&mut self) {
        self.frames_since_keyframe = self.config.gop_size;
    }

    /// Get encoder statistics.
    #[must_use]
    pub fn stats(&self) -> EncoderStats {
        let avg = if self.frames_encoded > 0 {
            self.total_encode_time / self.frames_encoded as u32
        } else {
            Duration::ZERO
        };

        EncoderStats {
            frames_submitted: self.frames_submitted,
            frames_encoded: self.frames_encoded,
            frames_dropped: self.frames_dropped,
            total_bytes: self.total_bytes,
            avg_encode_time: avg,
            peak_encode_time: self.peak_encode_time,
            estimated_bitrate_kbps: self.estimate_bitrate_kbps(),
            keyframes_emitted: self.keyframes_emitted,
        }
    }

    /// Reset the encoder state.
    pub fn reset(&mut self) {
        let buf_size =
            (self.config.resolution.0 as usize) * (self.config.resolution.1 as usize) * 4;
        self.double_buf = DoubleBuffer::new(buf_size);
        self.output_queue.clear();
        self.frame_counter = 0;
        self.frames_since_keyframe = 0;
        self.frames_submitted = 0;
        self.frames_encoded = 0;
        self.frames_dropped = 0;
        self.total_bytes = 0;
        self.total_encode_time = Duration::ZERO;
        self.peak_encode_time = Duration::ZERO;
        self.keyframes_emitted = 0;
        self.recent_sizes.clear();
    }

    // -- private helpers --

    /// Estimate encoded output size for a frame given rate control settings.
    fn estimate_output_size(&self, frame_type: FrameType, raw_size: usize) -> usize {
        let base_size = match &self.config.rate_control {
            RateControlMode::Cbr { target_kbps } => {
                // target_kbps * 1000 / 8 / framerate = bytes per frame
                let bpf = (*target_kbps as u64 * 1000) / 8 / self.config.framerate.max(1) as u64;
                bpf as usize
            }
            RateControlMode::Vbr {
                target_kbps,
                max_kbps: _,
            } => {
                let bpf = (*target_kbps as u64 * 1000) / 8 / self.config.framerate.max(1) as u64;
                bpf as usize
            }
            RateControlMode::Crf { crf } => {
                // Higher CRF = smaller output. Rough model.
                let ratio = 1.0 - (*crf as f64 / 63.0) * 0.95;
                ((raw_size as f64) * ratio * 0.05) as usize
            }
        };

        // Keyframes are typically 3-5x larger.
        let multiplier = match frame_type {
            FrameType::Key => 4,
            FrameType::Inter => 1,
        };

        (base_size * multiplier).max(64)
    }

    /// Estimate current bitrate from recent packet sizes.
    fn estimate_bitrate_kbps(&self) -> u32 {
        if self.recent_sizes.len() < 2 {
            return 0;
        }
        let total_bits: u64 = self.recent_sizes.iter().map(|(_, s)| s * 8).sum();
        let first = self.recent_sizes.front().map(|(t, _)| *t);
        let last = self.recent_sizes.back().map(|(t, _)| *t);
        match (first, last) {
            (Some(f), Some(l)) => {
                let dur = l.duration_since(f).as_secs_f64();
                if dur > 0.0 {
                    (total_bits as f64 / dur / 1000.0) as u32
                } else {
                    0
                }
            }
            _ => 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn small_frame(val: u8) -> Vec<u8> {
        vec![val; 4 * 4 * 4] // 4x4 RGBA
    }

    fn small_config() -> AsyncEncoderConfig {
        AsyncEncoderConfigBuilder::new()
            .resolution(4, 4)
            .framerate(30)
            .gop_size(5)
            .output_queue_capacity(8)
            .build()
            .expect("config")
    }

    #[test]
    fn test_config_defaults() {
        let cfg = AsyncEncoderConfig::default();
        assert_eq!(cfg.resolution, (1920, 1080));
        assert_eq!(cfg.framerate, 60);
        assert_eq!(cfg.gop_size, 120);
        assert_eq!(cfg.output_queue_capacity, 16);
        assert!(cfg.drop_on_backpressure);
    }

    #[test]
    fn test_config_builder_validation() {
        assert!(AsyncEncoderConfigBuilder::new()
            .resolution(0, 100)
            .build()
            .is_err());
        assert!(AsyncEncoderConfigBuilder::new()
            .framerate(0)
            .build()
            .is_err());
        assert!(AsyncEncoderConfigBuilder::new()
            .framerate(241)
            .build()
            .is_err());
        assert!(AsyncEncoderConfigBuilder::new()
            .gop_size(0)
            .build()
            .is_err());
        assert!(AsyncEncoderConfigBuilder::new()
            .output_queue_capacity(0)
            .build()
            .is_err());
    }

    #[test]
    fn test_submit_and_encode_one_frame() {
        let cfg = small_config();
        let mut enc = AsyncEncoder::new(cfg).expect("encoder");

        enc.submit_frame(&small_frame(42), 4, 4, 0).expect("submit");
        assert!(enc.encode_pending());

        let pkt = enc.poll_output().expect("packet");
        assert_eq!(pkt.frame_number, 0);
        assert_eq!(pkt.frame_type, FrameType::Key); // First frame is keyframe
        assert!(pkt.size_bytes > 0);
    }

    #[test]
    fn test_gop_keyframe_interval() {
        let cfg = AsyncEncoderConfigBuilder::new()
            .resolution(4, 4)
            .framerate(30)
            .gop_size(3)
            .build()
            .expect("cfg");
        let mut enc = AsyncEncoder::new(cfg).expect("encoder");

        let mut types = Vec::new();
        for i in 0..9u64 {
            enc.submit_frame(&small_frame(0), 4, 4, i * 33_333)
                .expect("s");
            enc.encode_pending();
            let pkt = enc.poll_output().expect("pkt");
            types.push(pkt.frame_type);
        }

        // Frame 0: Key, 1: Inter, 2: Inter, 3: Key (after gop_size=3), ...
        assert_eq!(types[0], FrameType::Key);
        assert_eq!(types[1], FrameType::Inter);
        assert_eq!(types[2], FrameType::Inter);
        assert_eq!(types[3], FrameType::Key);
    }

    #[test]
    fn test_force_keyframe() {
        let cfg = small_config();
        let mut enc = AsyncEncoder::new(cfg).expect("encoder");

        // First frame is always key.
        enc.submit_frame(&small_frame(0), 4, 4, 0).expect("s");
        enc.encode_pending();
        enc.poll_output();

        // Second frame would normally be inter.
        enc.force_keyframe();
        enc.submit_frame(&small_frame(0), 4, 4, 33_333).expect("s");
        enc.encode_pending();
        let pkt = enc.poll_output().expect("pkt");
        assert_eq!(pkt.frame_type, FrameType::Key);
    }

    #[test]
    fn test_wrong_frame_size_rejected() {
        let cfg = small_config();
        let mut enc = AsyncEncoder::new(cfg).expect("encoder");
        let res = enc.submit_frame(&[0u8; 10], 4, 4, 0);
        assert!(res.is_err());
    }

    #[test]
    fn test_encode_pending_when_empty() {
        let cfg = small_config();
        let mut enc = AsyncEncoder::new(cfg).expect("encoder");
        assert!(!enc.encode_pending());
    }

    #[test]
    fn test_output_queue_capacity() {
        let cfg = AsyncEncoderConfigBuilder::new()
            .resolution(4, 4)
            .framerate(30)
            .gop_size(100)
            .output_queue_capacity(3)
            .build()
            .expect("cfg");
        let mut enc = AsyncEncoder::new(cfg).expect("encoder");

        for i in 0..5u64 {
            enc.submit_frame(&small_frame(0), 4, 4, i * 33_333)
                .expect("s");
            enc.encode_pending();
        }

        // Queue capacity is 3, so oldest packets were dropped.
        assert!(enc.output_queue_len() <= 3);
    }

    #[test]
    fn test_encoder_stats() {
        let cfg = small_config();
        let mut enc = AsyncEncoder::new(cfg).expect("encoder");

        for i in 0..5u64 {
            enc.submit_frame(&small_frame(0), 4, 4, i * 33_333)
                .expect("s");
            enc.encode_pending();
        }

        let stats = enc.stats();
        assert_eq!(stats.frames_submitted, 5);
        assert_eq!(stats.frames_encoded, 5);
        assert!(stats.total_bytes > 0);
        assert!(stats.keyframes_emitted >= 1);
    }

    #[test]
    fn test_encoder_reset() {
        let cfg = small_config();
        let mut enc = AsyncEncoder::new(cfg).expect("encoder");

        enc.submit_frame(&small_frame(0), 4, 4, 0).expect("s");
        enc.encode_pending();
        enc.reset();

        let stats = enc.stats();
        assert_eq!(stats.frames_submitted, 0);
        assert_eq!(stats.frames_encoded, 0);
        assert_eq!(stats.total_bytes, 0);
        assert_eq!(enc.output_queue_len(), 0);
    }

    #[test]
    fn test_cbr_rate_control() {
        let cfg = AsyncEncoderConfigBuilder::new()
            .resolution(4, 4)
            .framerate(30)
            .rate_control(RateControlMode::Cbr { target_kbps: 3000 })
            .gop_size(30)
            .build()
            .expect("cfg");
        let mut enc = AsyncEncoder::new(cfg).expect("encoder");

        enc.submit_frame(&small_frame(0), 4, 4, 0).expect("s");
        enc.encode_pending();
        let pkt = enc.poll_output().expect("pkt");
        // CBR: 3000 * 1000 / 8 / 30 = 12500 bytes per frame.
        // Keyframe multiplier 4x = 50000.
        assert!(pkt.size_bytes > 0);
    }

    #[test]
    fn test_crf_rate_control() {
        let cfg = AsyncEncoderConfigBuilder::new()
            .resolution(4, 4)
            .framerate(30)
            .rate_control(RateControlMode::Crf { crf: 23 })
            .gop_size(30)
            .build()
            .expect("cfg");
        let mut enc = AsyncEncoder::new(cfg).expect("encoder");

        enc.submit_frame(&small_frame(0), 4, 4, 0).expect("s");
        enc.encode_pending();
        let pkt = enc.poll_output().expect("pkt");
        assert!(pkt.size_bytes > 0);
    }

    #[test]
    fn test_peek_output() {
        let cfg = small_config();
        let mut enc = AsyncEncoder::new(cfg).expect("encoder");

        enc.submit_frame(&small_frame(0), 4, 4, 0).expect("s");
        enc.encode_pending();

        let peeked = enc.peek_output().expect("peek");
        assert_eq!(peeked.frame_number, 0);
        // Should still be there after peek.
        assert_eq!(enc.output_queue_len(), 1);
    }

    #[test]
    fn test_full_submit_encode_poll_cycle() {
        let cfg = small_config();
        let mut enc = AsyncEncoder::new(cfg).expect("encoder");

        let mut total_bytes = 0u64;
        for i in 0..20u64 {
            enc.submit_frame(&small_frame((i % 256) as u8), 4, 4, i * 33_333)
                .expect("s");
            if enc.encode_pending() {
                if let Some(pkt) = enc.poll_output() {
                    total_bytes += pkt.size_bytes as u64;
                }
            }
        }

        assert!(total_bytes > 0);
        let stats = enc.stats();
        assert_eq!(stats.frames_submitted, 20);
        assert!(stats.frames_encoded > 0);
    }

    #[test]
    fn test_rate_control_default() {
        let rc = RateControlMode::default();
        match rc {
            RateControlMode::Vbr {
                target_kbps,
                max_kbps,
            } => {
                assert_eq!(target_kbps, 6000);
                assert_eq!(max_kbps, 8000);
            }
            _ => panic!("Expected VBR default"),
        }
    }

    #[test]
    fn test_frame_type_eq() {
        assert_eq!(FrameType::Key, FrameType::Key);
        assert_ne!(FrameType::Key, FrameType::Inter);
    }
}
