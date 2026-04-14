//! Streaming / pipeline denoising mode for integration with oximedia-graph.
//!
//! This module provides a `StreamingDenoiser` that accepts frames one at a
//! time, buffers them internally, and emits denoised frames once enough
//! context has been accumulated.  This is suitable for use inside a media
//! processing graph where frames arrive sequentially and low-latency output
//! is desired.

use std::collections::VecDeque;

/// Configuration for the streaming denoiser pipeline.
#[derive(Clone, Debug)]
pub struct DenoiserPipelineConfig {
    /// Maximum number of frames kept in the look-ahead buffer.
    pub frame_buffer_size: usize,
    /// Number of frames that must be buffered before the first output is
    /// produced.  Must be ≤ `frame_buffer_size`.
    pub latency_frames: usize,
}

impl Default for DenoiserPipelineConfig {
    fn default() -> Self {
        Self {
            frame_buffer_size: 5,
            latency_frames: 2,
        }
    }
}

impl DenoiserPipelineConfig {
    /// Validate configuration consistency.
    ///
    /// # Errors
    /// Returns an error string if latency exceeds buffer size or if buffer
    /// size is zero.
    pub fn validate(&self) -> Result<(), String> {
        if self.frame_buffer_size == 0 {
            return Err("frame_buffer_size must be at least 1".to_string());
        }
        if self.latency_frames > self.frame_buffer_size {
            return Err(format!(
                "latency_frames ({}) must not exceed frame_buffer_size ({})",
                self.latency_frames, self.frame_buffer_size
            ));
        }
        Ok(())
    }
}

/// A streaming denoiser that operates in a pipeline / graph context.
///
/// Frames are pushed one at a time via [`Self::push_frame`].  Once the
/// internal buffer holds at least `latency_frames + 1` frames, the oldest
/// frame is denoised (temporal average) and returned.  Before that threshold
/// is reached `None` is returned so that the caller knows output is not yet
/// available.
///
/// Use [`Self::flush`] to drain all remaining buffered frames at the end of a
/// sequence (e.g. at stream end).
pub struct StreamingDenoiser {
    config: DenoiserPipelineConfig,
    /// Buffered input frames (f32 luma, row-major).
    buffer: VecDeque<Vec<f32>>,
}

impl StreamingDenoiser {
    /// Create a new streaming denoiser.
    ///
    /// # Panics
    /// Panics if the configuration is invalid (use [`DenoiserPipelineConfig::validate`]
    /// before constructing if untrusted input).
    #[must_use]
    pub fn new(config: DenoiserPipelineConfig) -> Self {
        debug_assert!(
            config.validate().is_ok(),
            "invalid DenoiserPipelineConfig: {:?}",
            config.validate()
        );
        let capacity = config.frame_buffer_size.max(1);
        Self {
            config,
            buffer: VecDeque::with_capacity(capacity),
        }
    }

    /// Push a new frame into the buffer.
    ///
    /// Returns `Some(denoised_frame)` once the buffer has filled past the
    /// configured latency threshold, otherwise `None`.  The denoised frame is
    /// produced by temporally averaging all buffered frames.
    ///
    /// # Arguments
    /// * `frame` — Flat f32 luma samples (row-major).  All frames in a
    ///   session must have the same length.
    pub fn push_frame(&mut self, frame: Vec<f32>) -> Option<Vec<f32>> {
        // Add the new frame to the buffer
        self.buffer.push_back(frame);

        // Trim buffer to the configured maximum size, discarding oldest entries
        while self.buffer.len() > self.config.frame_buffer_size {
            self.buffer.pop_front();
        }

        // We only emit output once we have accumulated enough context
        let needed = self.config.latency_frames + 1;
        if self.buffer.len() < needed {
            return None;
        }

        // Produce a temporally averaged frame from the current buffer contents
        Some(self.average_buffer())
    }

    /// Drain all remaining frames from the buffer as denoised outputs.
    ///
    /// Each buffered frame is denoised using the progressively shrinking
    /// window of remaining frames.  After this call the buffer is empty.
    pub fn flush(&mut self) -> Vec<Vec<f32>> {
        let mut outputs = Vec::with_capacity(self.buffer.len());

        while !self.buffer.is_empty() {
            outputs.push(self.average_buffer());
            self.buffer.pop_front();
        }

        outputs
    }

    /// Compute the temporal average across all frames currently in the buffer.
    fn average_buffer(&self) -> Vec<f32> {
        let n = self.buffer.len();
        if n == 0 {
            return Vec::new();
        }

        let len = self.buffer.front().map_or(0, |f| f.len());
        if len == 0 {
            return Vec::new();
        }

        let mut output = vec![0.0f32; len];
        let weight = 1.0 / n as f32;

        for hist_frame in &self.buffer {
            for (o, &s) in output.iter_mut().zip(hist_frame.iter()) {
                *o += s * weight;
            }
        }

        output
    }

    /// Return the number of frames currently held in the buffer.
    #[must_use]
    pub fn buffered_count(&self) -> usize {
        self.buffer.len()
    }

    /// Return a reference to the configuration.
    #[must_use]
    pub fn config(&self) -> &DenoiserPipelineConfig {
        &self.config
    }

    /// Reset internal state, discarding all buffered frames.
    pub fn reset(&mut self) {
        self.buffer.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_config_default() {
        let config = DenoiserPipelineConfig::default();
        assert!(config.validate().is_ok());
        assert_eq!(config.frame_buffer_size, 5);
        assert_eq!(config.latency_frames, 2);
    }

    #[test]
    fn test_pipeline_config_invalid_latency() {
        let config = DenoiserPipelineConfig {
            frame_buffer_size: 3,
            latency_frames: 5,
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_pipeline_config_zero_buffer() {
        let config = DenoiserPipelineConfig {
            frame_buffer_size: 0,
            latency_frames: 0,
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_streaming_denoiser_latency() {
        // With latency_frames = 2, the first output should arrive on frame 3
        let config = DenoiserPipelineConfig {
            frame_buffer_size: 5,
            latency_frames: 2,
        };
        let mut denoiser = StreamingDenoiser::new(config);
        let frame = vec![10.0f32; 16];

        // Frames 1 and 2 should return None
        assert!(
            denoiser.push_frame(frame.clone()).is_none(),
            "frame 1 should return None"
        );
        assert!(
            denoiser.push_frame(frame.clone()).is_none(),
            "frame 2 should return None"
        );
        // Frame 3 should produce output
        let out = denoiser.push_frame(frame.clone());
        assert!(out.is_some(), "frame 3 should produce output");
    }

    #[test]
    fn test_streaming_denoiser_uniform_average() {
        // All frames identical → output should match
        let config = DenoiserPipelineConfig {
            frame_buffer_size: 3,
            latency_frames: 1,
        };
        let mut denoiser = StreamingDenoiser::new(config);
        let frame = vec![42.0f32; 8];
        denoiser.push_frame(frame.clone()); // None
        let output = denoiser.push_frame(frame.clone()).expect("should produce output");
        for &v in &output {
            assert!((v - 42.0).abs() < 1e-4, "uniform average: {v}");
        }
    }

    #[test]
    fn test_streaming_denoiser_flush_drains_buffer() {
        let config = DenoiserPipelineConfig {
            frame_buffer_size: 4,
            latency_frames: 1,
        };
        let mut denoiser = StreamingDenoiser::new(config);
        let frame = vec![1.0f32; 4];
        // Push 2 frames (only 1 output emitted via push_frame)
        denoiser.push_frame(frame.clone());
        denoiser.push_frame(frame.clone());
        // Flush the remainder
        let flushed = denoiser.flush();
        assert!(!flushed.is_empty(), "flush should produce outputs");
        assert_eq!(
            denoiser.buffered_count(),
            0,
            "buffer should be empty after flush"
        );
    }

    #[test]
    fn test_streaming_denoiser_flush_empty_buffer() {
        let config = DenoiserPipelineConfig::default();
        let mut denoiser = StreamingDenoiser::new(config);
        let flushed = denoiser.flush();
        assert!(flushed.is_empty(), "flush on empty buffer should yield []");
    }

    #[test]
    fn test_streaming_denoiser_reset() {
        let config = DenoiserPipelineConfig::default();
        let mut denoiser = StreamingDenoiser::new(config);
        let frame = vec![5.0f32; 8];
        denoiser.push_frame(frame.clone());
        denoiser.push_frame(frame.clone());
        assert!(denoiser.buffered_count() > 0);
        denoiser.reset();
        assert_eq!(denoiser.buffered_count(), 0);
    }

    #[test]
    fn test_streaming_denoiser_buffer_size_capped() {
        // Buffer should not grow beyond frame_buffer_size
        let config = DenoiserPipelineConfig {
            frame_buffer_size: 3,
            latency_frames: 1,
        };
        let mut denoiser = StreamingDenoiser::new(config);
        let frame = vec![0.0f32; 4];
        for _ in 0..10 {
            denoiser.push_frame(frame.clone());
        }
        assert!(
            denoiser.buffered_count() <= 3,
            "buffer must not exceed frame_buffer_size: {}",
            denoiser.buffered_count()
        );
    }

    #[test]
    fn test_streaming_denoiser_latency_zero() {
        // latency_frames = 0 → every frame produces output immediately
        let config = DenoiserPipelineConfig {
            frame_buffer_size: 3,
            latency_frames: 0,
        };
        let mut denoiser = StreamingDenoiser::new(config);
        let frame = vec![7.0f32; 4];
        let out = denoiser.push_frame(frame);
        assert!(out.is_some(), "latency=0 should always produce output");
    }
}
