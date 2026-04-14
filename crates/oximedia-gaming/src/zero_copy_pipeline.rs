#![allow(dead_code)]

//! Zero-copy frame pipeline from capture to encoder.
//!
//! Implements a lock-free ring buffer based pipeline that passes frame
//! references between the capture and encoder stages without copying pixel
//! data. Uses a slot-based approach where the capture stage writes into the
//! next available slot and the encoder reads from the oldest unconsumed slot.
//!
//! # Architecture
//!
//! ```text
//!  Capture ──► [Slot 0][Slot 1]...[Slot N-1] ──► Encoder
//!                ▲ write_idx          read_idx ▲
//! ```
//!
//! Each slot holds a pre-allocated buffer that is reused across frames,
//! eliminating per-frame allocation overhead. The pipeline tracks which
//! slots are available (empty), filled (ready to encode), and in-flight
//! (being encoded).
//!
//! # Features
//!
//! - **Pre-allocated ring buffer**: Fixed memory footprint with configurable
//!   slot count
//! - **Slot lifecycle tracking**: Empty → Captured → Encoding → Empty
//! - **Frame dropping policy**: When the encoder falls behind, the oldest
//!   un-consumed frame is dropped and counted
//! - **Pipeline statistics**: Frames captured, encoded, dropped, and queue depth
//! - **Configurable capacity**: 2–64 slot ring buffer

use std::time::{Duration, Instant};

use crate::{GamingError, GamingResult};

// ---------------------------------------------------------------------------
// Slot state machine
// ---------------------------------------------------------------------------

/// Lifecycle state of a single frame slot in the ring buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotState {
    /// Slot is empty and ready to receive a new capture.
    Empty,
    /// Slot contains a captured frame waiting for the encoder.
    Captured,
    /// Slot is currently being read by the encoder.
    Encoding,
}

// ---------------------------------------------------------------------------
// FrameSlot
// ---------------------------------------------------------------------------

/// A single slot in the zero-copy ring buffer.
#[derive(Debug, Clone)]
pub struct FrameSlot {
    /// Pre-allocated pixel buffer (reused across frames).
    buffer: Vec<u8>,
    /// Current state of this slot.
    state: SlotState,
    /// Width of the frame stored (0 when empty).
    width: u32,
    /// Height of the frame stored (0 when empty).
    height: u32,
    /// Timestamp when this frame was captured.
    capture_time: Option<Instant>,
    /// Monotonic frame number assigned at capture time.
    frame_number: u64,
    /// Presentation timestamp in microseconds.
    pts_us: u64,
}

impl FrameSlot {
    /// Create a new empty slot with a pre-allocated buffer.
    fn new(capacity: usize) -> Self {
        Self {
            buffer: vec![0u8; capacity],
            state: SlotState::Empty,
            width: 0,
            height: 0,
            capture_time: None,
            frame_number: 0,
            pts_us: 0,
        }
    }

    /// Reset the slot to empty without deallocating the buffer.
    fn reset(&mut self) {
        self.state = SlotState::Empty;
        self.width = 0;
        self.height = 0;
        self.capture_time = None;
        self.frame_number = 0;
        self.pts_us = 0;
    }

    /// Write frame data into this slot. The buffer is reused, expanding only
    /// if the new data is larger than the pre-allocated capacity.
    fn fill(&mut self, data: &[u8], width: u32, height: u32, frame_number: u64, pts_us: u64) {
        if data.len() > self.buffer.len() {
            self.buffer.resize(data.len(), 0);
        }
        self.buffer[..data.len()].copy_from_slice(data);
        self.width = width;
        self.height = height;
        self.state = SlotState::Captured;
        self.capture_time = Some(Instant::now());
        self.frame_number = frame_number;
        self.pts_us = pts_us;
    }

    /// Get a read-only reference to the pixel data.
    ///
    /// # Errors
    ///
    /// Returns error if the slot is not in `Captured` state.
    fn data(&self) -> GamingResult<&[u8]> {
        if self.state != SlotState::Captured && self.state != SlotState::Encoding {
            return Err(GamingError::ReplayBufferError(
                "Slot is not captured".into(),
            ));
        }
        let len = (self.width as usize) * (self.height as usize) * 4;
        if len > self.buffer.len() {
            return Err(GamingError::ReplayBufferError(
                "Buffer size mismatch".into(),
            ));
        }
        Ok(&self.buffer[..len])
    }
}

// ---------------------------------------------------------------------------
// Pipeline configuration
// ---------------------------------------------------------------------------

/// Configuration for the zero-copy pipeline.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Number of slots in the ring buffer (2–64).
    pub slot_count: usize,
    /// Frame resolution (width, height) for pre-allocation.
    pub resolution: (u32, u32),
    /// Bytes per pixel (typically 4 for RGBA).
    pub bytes_per_pixel: usize,
    /// Whether to allow overwriting the oldest un-consumed frame when the
    /// ring buffer is full (drop policy).
    pub allow_overwrite: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            slot_count: 4,
            resolution: (1920, 1080),
            bytes_per_pixel: 4,
            allow_overwrite: true,
        }
    }
}

impl PipelineConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// Returns error for invalid slot count or zero resolution.
    pub fn validate(&self) -> GamingResult<()> {
        if self.slot_count < 2 || self.slot_count > 64 {
            return Err(GamingError::InvalidConfig(
                "Slot count must be between 2 and 64".into(),
            ));
        }
        if self.resolution.0 == 0 || self.resolution.1 == 0 {
            return Err(GamingError::InvalidConfig(
                "Resolution must be non-zero".into(),
            ));
        }
        if self.bytes_per_pixel == 0 {
            return Err(GamingError::InvalidConfig(
                "Bytes per pixel must be non-zero".into(),
            ));
        }
        Ok(())
    }

    /// Calculate the byte size of a single slot's buffer.
    #[must_use]
    pub fn slot_byte_size(&self) -> usize {
        (self.resolution.0 as usize) * (self.resolution.1 as usize) * self.bytes_per_pixel
    }

    /// Calculate total memory footprint for the ring buffer.
    #[must_use]
    pub fn total_memory(&self) -> usize {
        self.slot_byte_size() * self.slot_count
    }
}

/// Builder for [`PipelineConfig`].
#[derive(Debug, Clone)]
pub struct PipelineConfigBuilder {
    inner: PipelineConfig,
}

impl Default for PipelineConfigBuilder {
    fn default() -> Self {
        Self {
            inner: PipelineConfig::default(),
        }
    }
}

impl PipelineConfigBuilder {
    /// Create a new builder with defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set slot count.
    #[must_use]
    pub fn slot_count(mut self, count: usize) -> Self {
        self.inner.slot_count = count;
        self
    }

    /// Set frame resolution.
    #[must_use]
    pub fn resolution(mut self, width: u32, height: u32) -> Self {
        self.inner.resolution = (width, height);
        self
    }

    /// Set bytes per pixel.
    #[must_use]
    pub fn bytes_per_pixel(mut self, bpp: usize) -> Self {
        self.inner.bytes_per_pixel = bpp;
        self
    }

    /// Set overwrite policy.
    #[must_use]
    pub fn allow_overwrite(mut self, allow: bool) -> Self {
        self.inner.allow_overwrite = allow;
        self
    }

    /// Build the configuration.
    ///
    /// # Errors
    ///
    /// Returns error for invalid settings.
    pub fn build(self) -> GamingResult<PipelineConfig> {
        self.inner.validate()?;
        Ok(self.inner)
    }
}

// ---------------------------------------------------------------------------
// Pipeline statistics
// ---------------------------------------------------------------------------

/// Runtime statistics for the zero-copy pipeline.
#[derive(Debug, Clone)]
pub struct PipelineStats {
    /// Total frames submitted by the capture stage.
    pub frames_submitted: u64,
    /// Total frames consumed by the encoder stage.
    pub frames_consumed: u64,
    /// Total frames dropped due to ring buffer overflow.
    pub frames_dropped: u64,
    /// Current number of captured frames waiting for the encoder.
    pub queue_depth: usize,
    /// Total memory allocated for slots.
    pub total_memory_bytes: usize,
    /// Average time a frame spent waiting in the queue.
    pub avg_queue_wait: Duration,
}

// ---------------------------------------------------------------------------
// FrameRef — lightweight handle returned to the encoder
// ---------------------------------------------------------------------------

/// A lightweight reference to a captured frame in the ring buffer.
///
/// The encoder receives this handle to read frame data without copying.
/// When processing is complete, the encoder calls
/// [`ZeroCopyPipeline::finish_encoding`] to release the slot.
#[derive(Debug, Clone)]
pub struct FrameRef {
    /// Index of the slot in the ring buffer.
    pub slot_index: usize,
    /// Frame number.
    pub frame_number: u64,
    /// Frame width.
    pub width: u32,
    /// Frame height.
    pub height: u32,
    /// Presentation timestamp in microseconds.
    pub pts_us: u64,
    /// When the frame was captured.
    pub capture_time: Instant,
}

// ---------------------------------------------------------------------------
// ZeroCopyPipeline
// ---------------------------------------------------------------------------

/// A zero-copy frame pipeline connecting capture and encoder stages.
///
/// The pipeline maintains a fixed-size ring buffer of pre-allocated frame
/// slots. The capture stage writes into the next available slot, and the
/// encoder reads from the oldest unconsumed slot.
#[derive(Debug)]
pub struct ZeroCopyPipeline {
    slots: Vec<FrameSlot>,
    /// Index of the next slot to write (capture stage).
    write_idx: usize,
    /// Index of the next slot to read (encoder stage).
    read_idx: usize,
    /// Configuration.
    config: PipelineConfig,
    /// Running frame counter.
    frame_counter: u64,
    /// Frames dropped due to overflow.
    frames_dropped: u64,
    /// Frames submitted.
    frames_submitted: u64,
    /// Frames consumed by encoder.
    frames_consumed: u64,
    /// Accumulated queue wait time.
    total_queue_wait: Duration,
}

impl ZeroCopyPipeline {
    /// Create a new zero-copy pipeline with the given configuration.
    ///
    /// # Errors
    ///
    /// Returns error if the configuration is invalid.
    pub fn new(config: PipelineConfig) -> GamingResult<Self> {
        config.validate()?;
        let slot_size = config.slot_byte_size();
        let slots: Vec<FrameSlot> = (0..config.slot_count)
            .map(|_| FrameSlot::new(slot_size))
            .collect();

        Ok(Self {
            slots,
            write_idx: 0,
            read_idx: 0,
            config,
            frame_counter: 0,
            frames_dropped: 0,
            frames_submitted: 0,
            frames_consumed: 0,
            total_queue_wait: Duration::ZERO,
        })
    }

    /// Submit a captured frame into the next available slot.
    ///
    /// If the ring buffer is full and `allow_overwrite` is enabled, the
    /// oldest un-consumed frame is dropped. If overwrite is disabled, an
    /// error is returned.
    ///
    /// # Errors
    ///
    /// Returns error if the buffer is full and overwrite is disabled, or if
    /// data size does not match the expected frame size.
    pub fn submit_frame(
        &mut self,
        data: &[u8],
        width: u32,
        height: u32,
        pts_us: u64,
    ) -> GamingResult<u64> {
        let expected = (width as usize) * (height as usize) * self.config.bytes_per_pixel;
        if data.len() != expected {
            return Err(GamingError::InvalidConfig(format!(
                "Frame data length {} does not match expected {} ({}x{}x{})",
                data.len(),
                expected,
                width,
                height,
                self.config.bytes_per_pixel
            )));
        }

        // Check if write slot is occupied.
        let slot = &self.slots[self.write_idx];
        if slot.state == SlotState::Captured {
            if self.config.allow_overwrite {
                // Drop the oldest frame.
                self.frames_dropped += 1;
                self.slots[self.write_idx].reset();
            } else {
                return Err(GamingError::ReplayBufferError(
                    "Pipeline ring buffer is full".into(),
                ));
            }
        } else if slot.state == SlotState::Encoding {
            // Cannot overwrite a slot the encoder is currently reading.
            if self.config.allow_overwrite {
                // Skip this slot and try the next one.
                // In a real implementation this would be more sophisticated.
                self.frames_dropped += 1;
                self.frames_submitted += 1;
                self.frame_counter += 1;
                return Ok(self.frame_counter - 1);
            }
            return Err(GamingError::ReplayBufferError(
                "Pipeline slot is being encoded".into(),
            ));
        }

        let frame_num = self.frame_counter;
        self.frame_counter += 1;

        self.slots[self.write_idx].fill(data, width, height, frame_num, pts_us);
        self.write_idx = (self.write_idx + 1) % self.config.slot_count;
        self.frames_submitted += 1;

        Ok(frame_num)
    }

    /// Acquire the next frame for encoding.
    ///
    /// Returns a [`FrameRef`] handle that provides read access to the frame
    /// data. Call [`finish_encoding`](Self::finish_encoding) when done.
    ///
    /// Returns `None` if no frames are available.
    pub fn acquire_frame(&mut self) -> Option<FrameRef> {
        // Scan from read_idx forward to find a Captured slot.
        for _ in 0..self.config.slot_count {
            let idx = self.read_idx;
            let slot = &mut self.slots[idx];
            if slot.state == SlotState::Captured {
                slot.state = SlotState::Encoding;
                let capture_time = slot.capture_time.unwrap_or_else(Instant::now);
                self.read_idx = (self.read_idx + 1) % self.config.slot_count;
                return Some(FrameRef {
                    slot_index: idx,
                    frame_number: slot.frame_number,
                    width: slot.width,
                    height: slot.height,
                    pts_us: slot.pts_us,
                    capture_time,
                });
            }
            self.read_idx = (self.read_idx + 1) % self.config.slot_count;
        }
        None
    }

    /// Read the pixel data for a frame that was previously acquired.
    ///
    /// # Errors
    ///
    /// Returns error if the slot index is out of range or the slot is not
    /// in the `Encoding` state.
    pub fn read_frame_data(&self, frame_ref: &FrameRef) -> GamingResult<&[u8]> {
        if frame_ref.slot_index >= self.slots.len() {
            return Err(GamingError::InvalidConfig("Invalid slot index".into()));
        }
        let slot = &self.slots[frame_ref.slot_index];
        if slot.state != SlotState::Encoding {
            return Err(GamingError::ReplayBufferError(
                "Slot is not in encoding state".into(),
            ));
        }
        slot.data()
    }

    /// Release a slot after encoding is complete.
    ///
    /// # Errors
    ///
    /// Returns error if the slot is not in `Encoding` state.
    pub fn finish_encoding(&mut self, frame_ref: &FrameRef) -> GamingResult<()> {
        if frame_ref.slot_index >= self.slots.len() {
            return Err(GamingError::InvalidConfig("Invalid slot index".into()));
        }
        let slot = &mut self.slots[frame_ref.slot_index];
        if slot.state != SlotState::Encoding {
            return Err(GamingError::ReplayBufferError(
                "Slot not in encoding state".into(),
            ));
        }

        // Record queue wait time.
        if let Some(ct) = slot.capture_time {
            self.total_queue_wait += ct.elapsed();
        }

        slot.reset();
        self.frames_consumed += 1;
        Ok(())
    }

    /// Get current pipeline statistics.
    #[must_use]
    pub fn stats(&self) -> PipelineStats {
        let queue_depth = self
            .slots
            .iter()
            .filter(|s| s.state == SlotState::Captured)
            .count();

        let avg_wait = if self.frames_consumed > 0 {
            self.total_queue_wait / self.frames_consumed as u32
        } else {
            Duration::ZERO
        };

        PipelineStats {
            frames_submitted: self.frames_submitted,
            frames_consumed: self.frames_consumed,
            frames_dropped: self.frames_dropped,
            queue_depth,
            total_memory_bytes: self.config.total_memory(),
            avg_queue_wait: avg_wait,
        }
    }

    /// Number of slots currently available for writing.
    #[must_use]
    pub fn available_slots(&self) -> usize {
        self.slots
            .iter()
            .filter(|s| s.state == SlotState::Empty)
            .count()
    }

    /// Total slot count.
    #[must_use]
    pub fn slot_count(&self) -> usize {
        self.config.slot_count
    }

    /// Check whether the pipeline has frames ready for encoding.
    #[must_use]
    pub fn has_pending_frames(&self) -> bool {
        self.slots.iter().any(|s| s.state == SlotState::Captured)
    }

    /// Reset the pipeline, clearing all slots and counters.
    pub fn reset(&mut self) {
        for slot in &mut self.slots {
            slot.reset();
        }
        self.write_idx = 0;
        self.read_idx = 0;
        self.frame_counter = 0;
        self.frames_dropped = 0;
        self.frames_submitted = 0;
        self.frames_consumed = 0;
        self.total_queue_wait = Duration::ZERO;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config(slots: usize) -> PipelineConfig {
        PipelineConfigBuilder::new()
            .slot_count(slots)
            .resolution(2, 2)
            .bytes_per_pixel(4)
            .allow_overwrite(true)
            .build()
            .expect("valid config")
    }

    fn frame_data(width: u32, height: u32, val: u8) -> Vec<u8> {
        vec![val; (width as usize) * (height as usize) * 4]
    }

    #[test]
    fn test_pipeline_config_defaults() {
        let cfg = PipelineConfig::default();
        assert_eq!(cfg.slot_count, 4);
        assert_eq!(cfg.resolution, (1920, 1080));
        assert_eq!(cfg.bytes_per_pixel, 4);
        assert!(cfg.allow_overwrite);
    }

    #[test]
    fn test_pipeline_config_validation() {
        assert!(PipelineConfigBuilder::new().slot_count(1).build().is_err());
        assert!(PipelineConfigBuilder::new().slot_count(65).build().is_err());
        assert!(PipelineConfigBuilder::new()
            .resolution(0, 100)
            .build()
            .is_err());
        assert!(PipelineConfigBuilder::new()
            .bytes_per_pixel(0)
            .build()
            .is_err());
    }

    #[test]
    fn test_submit_and_acquire() {
        let cfg = small_config(4);
        let mut pipe = ZeroCopyPipeline::new(cfg).expect("pipeline");

        let data = frame_data(2, 2, 42);
        let fnum = pipe.submit_frame(&data, 2, 2, 1000).expect("submit");
        assert_eq!(fnum, 0);

        let fr = pipe.acquire_frame().expect("frame available");
        assert_eq!(fr.frame_number, 0);
        assert_eq!(fr.width, 2);
        assert_eq!(fr.height, 2);
        assert_eq!(fr.pts_us, 1000);

        let read_data = pipe.read_frame_data(&fr).expect("read");
        assert_eq!(read_data.len(), 16);
        assert_eq!(read_data[0], 42);

        pipe.finish_encoding(&fr).expect("finish");
    }

    #[test]
    fn test_multiple_frames_fifo() {
        let cfg = small_config(4);
        let mut pipe = ZeroCopyPipeline::new(cfg).expect("pipeline");

        for i in 0..3u8 {
            let data = frame_data(2, 2, i);
            pipe.submit_frame(&data, 2, 2, u64::from(i) * 1000)
                .expect("submit");
        }

        // Frames should come out in order.
        for i in 0..3u64 {
            let fr = pipe.acquire_frame().expect("frame");
            assert_eq!(fr.frame_number, i);
            pipe.finish_encoding(&fr).expect("finish");
        }
    }

    #[test]
    fn test_ring_buffer_overflow_with_overwrite() {
        let cfg = small_config(2);
        let mut pipe = ZeroCopyPipeline::new(cfg).expect("pipeline");

        // Fill both slots.
        pipe.submit_frame(&frame_data(2, 2, 1), 2, 2, 0)
            .expect("s1");
        pipe.submit_frame(&frame_data(2, 2, 2), 2, 2, 1000)
            .expect("s2");
        // Third frame overwrites the oldest.
        pipe.submit_frame(&frame_data(2, 2, 3), 2, 2, 2000)
            .expect("s3");

        let stats = pipe.stats();
        assert!(stats.frames_dropped > 0, "should have dropped a frame");
    }

    #[test]
    fn test_ring_buffer_no_overwrite_fails() {
        let cfg = PipelineConfigBuilder::new()
            .slot_count(2)
            .resolution(2, 2)
            .allow_overwrite(false)
            .build()
            .expect("cfg");
        let mut pipe = ZeroCopyPipeline::new(cfg).expect("pipeline");

        pipe.submit_frame(&frame_data(2, 2, 1), 2, 2, 0)
            .expect("s1");
        pipe.submit_frame(&frame_data(2, 2, 2), 2, 2, 1000)
            .expect("s2");
        let res = pipe.submit_frame(&frame_data(2, 2, 3), 2, 2, 2000);
        assert!(res.is_err());
    }

    #[test]
    fn test_no_pending_frames() {
        let cfg = small_config(2);
        let pipe = ZeroCopyPipeline::new(cfg).expect("pipeline");
        assert!(!pipe.has_pending_frames());
        assert_eq!(pipe.available_slots(), 2);
    }

    #[test]
    fn test_acquire_when_empty_returns_none() {
        let cfg = small_config(2);
        let mut pipe = ZeroCopyPipeline::new(cfg).expect("pipeline");
        assert!(pipe.acquire_frame().is_none());
    }

    #[test]
    fn test_wrong_data_size_rejected() {
        let cfg = small_config(2);
        let mut pipe = ZeroCopyPipeline::new(cfg).expect("pipeline");
        // 2x2x4 = 16 bytes expected, send 10
        let res = pipe.submit_frame(&[0u8; 10], 2, 2, 0);
        assert!(res.is_err());
    }

    #[test]
    fn test_pipeline_stats() {
        let cfg = small_config(4);
        let mut pipe = ZeroCopyPipeline::new(cfg).expect("pipeline");

        pipe.submit_frame(&frame_data(2, 2, 1), 2, 2, 0).expect("s");
        pipe.submit_frame(&frame_data(2, 2, 2), 2, 2, 1000)
            .expect("s");

        let fr = pipe.acquire_frame().expect("fr");
        pipe.finish_encoding(&fr).expect("fin");

        let stats = pipe.stats();
        assert_eq!(stats.frames_submitted, 2);
        assert_eq!(stats.frames_consumed, 1);
        assert_eq!(stats.queue_depth, 1);
    }

    #[test]
    fn test_pipeline_reset() {
        let cfg = small_config(4);
        let mut pipe = ZeroCopyPipeline::new(cfg).expect("pipeline");

        pipe.submit_frame(&frame_data(2, 2, 1), 2, 2, 0).expect("s");
        pipe.reset();

        assert_eq!(pipe.stats().frames_submitted, 0);
        assert!(!pipe.has_pending_frames());
        assert_eq!(pipe.available_slots(), 4);
    }

    #[test]
    fn test_total_memory_calculation() {
        let cfg = PipelineConfigBuilder::new()
            .slot_count(4)
            .resolution(1920, 1080)
            .bytes_per_pixel(4)
            .build()
            .expect("cfg");
        // 1920 * 1080 * 4 * 4 = 33_177_600
        assert_eq!(cfg.total_memory(), 1920 * 1080 * 4 * 4);
    }

    #[test]
    fn test_slot_byte_size() {
        let cfg = PipelineConfig {
            slot_count: 2,
            resolution: (100, 100),
            bytes_per_pixel: 4,
            allow_overwrite: true,
        };
        assert_eq!(cfg.slot_byte_size(), 40_000);
    }

    #[test]
    fn test_finish_encoding_invalid_slot() {
        let cfg = small_config(2);
        let mut pipe = ZeroCopyPipeline::new(cfg).expect("pipeline");
        let fake_ref = FrameRef {
            slot_index: 99,
            frame_number: 0,
            width: 2,
            height: 2,
            pts_us: 0,
            capture_time: Instant::now(),
        };
        assert!(pipe.finish_encoding(&fake_ref).is_err());
    }

    #[test]
    fn test_read_frame_data_wrong_state() {
        let cfg = small_config(2);
        let pipe = ZeroCopyPipeline::new(cfg).expect("pipeline");
        let fake_ref = FrameRef {
            slot_index: 0,
            frame_number: 0,
            width: 2,
            height: 2,
            pts_us: 0,
            capture_time: Instant::now(),
        };
        // Slot is Empty, not Encoding.
        assert!(pipe.read_frame_data(&fake_ref).is_err());
    }

    #[test]
    fn test_full_pipeline_cycle() {
        let cfg = small_config(3);
        let mut pipe = ZeroCopyPipeline::new(cfg).expect("pipeline");

        // Simulate 10 capture-encode cycles.
        for i in 0..10u64 {
            let data = frame_data(2, 2, (i % 256) as u8);
            pipe.submit_frame(&data, 2, 2, i * 16_667).expect("submit");
            if let Some(fr) = pipe.acquire_frame() {
                let _ = pipe.read_frame_data(&fr).expect("read");
                pipe.finish_encoding(&fr).expect("finish");
            }
        }

        let stats = pipe.stats();
        assert_eq!(stats.frames_submitted, 10);
        assert_eq!(stats.frames_consumed, 10);
        assert_eq!(stats.frames_dropped, 0);
    }
}
