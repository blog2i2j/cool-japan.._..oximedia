//! Multi-pass encoding with look-ahead for OxiMedia codecs.
//!
//! This module provides a comprehensive multi-pass encoding system that includes:
//! - Two-pass encoding with statistics collection
//! - Look-ahead buffer for analyzing future frames
//! - Adaptive quantization based on frame complexity
//! - VBV-compliant bitrate allocation
//! - Scene change detection
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                   MultiPassEncoder                          │
//! │  ┌──────────────────────────────────────────────────────┐   │
//! │  │               First Pass                             │   │
//! │  │  ┌──────────┐  ┌────────────┐  ┌─────────────────┐  │   │
//! │  │  │ Analyze  │→ │ Collect    │→ │ Save Statistics │  │   │
//! │  │  │ Frames   │  │ Statistics │  │ to File         │  │   │
//! │  │  └──────────┘  └────────────┘  └─────────────────┘  │   │
//! │  └──────────────────────────────────────────────────────┘   │
//! │  ┌──────────────────────────────────────────────────────┐   │
//! │  │               Second Pass                            │   │
//! │  │  ┌──────────┐  ┌────────────┐  ┌─────────────────┐  │   │
//! │  │  │ Load     │→ │ Allocate   │→ │ Encode with     │  │   │
//! │  │  │ Stats    │  │ Bitrate    │  │ Optimal QP      │  │   │
//! │  │  └──────────┘  └────────────┘  └─────────────────┘  │   │
//! │  └──────────────────────────────────────────────────────┘   │
//! │  ┌──────────────────────────────────────────────────────┐   │
//! │  │          Single-Pass with Look-ahead                 │   │
//! │  │  ┌──────────┐  ┌────────────┐  ┌─────────────────┐  │   │
//! │  │  │ Analyze  │→ │ Allocate   │→ │ Encode          │  │   │
//! │  │  │ Window   │  │ Adaptive   │  │                 │  │   │
//! │  │  └──────────┘  └────────────┘  └─────────────────┘  │   │
//! │  └──────────────────────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Example: Two-Pass Encoding
//!
//! ```ignore
//! use oximedia_codec::multipass::{MultiPassEncoder, PassType, EncoderConfig};
//!
//! // First pass
//! let config = EncoderConfig::new(1920, 1080)
//!     .with_pass(PassType::FirstPass)
//!     .with_stats_file("pass1.stats");
//!
//! let mut encoder = MultiPassEncoder::new(config)?;
//!
//! for frame in frames {
//!     encoder.encode_frame(&frame)?;
//! }
//!
//! encoder.save_stats("pass1.stats")?;
//!
//! // Second pass
//! let config = EncoderConfig::new(1920, 1080)
//!     .with_pass(PassType::SecondPass)
//!     .with_stats_file("pass1.stats")
//!     .with_target_bitrate(5_000_000);
//!
//! let mut encoder = MultiPassEncoder::new(config)?;
//! encoder.load_stats("pass1.stats")?;
//!
//! for frame in frames {
//!     let packet = encoder.encode_frame(&frame)?;
//!     // Write packet...
//! }
//! ```
//!
//! # Example: Single-Pass with Look-ahead
//!
//! ```ignore
//! use oximedia_codec::multipass::{MultiPassEncoder, PassType, EncoderConfig};
//!
//! let config = EncoderConfig::new(1920, 1080)
//!     .with_pass(PassType::SinglePassLookahead)
//!     .with_lookahead_frames(40);
//!
//! let mut encoder = MultiPassEncoder::new(config)?;
//!
//! for frame in frames {
//!     let packet = encoder.encode_frame(&frame)?;
//!     // Write packet...
//! }
//! ```

#![forbid(unsafe_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::module_name_repetitions)]

pub mod allocation;
pub mod complexity;
pub mod examples;
pub mod lookahead;
pub mod stats;
pub mod vbv;

use crate::frame::{FrameType, VideoFrame};
use crate::traits::EncodedPacket;
use allocation::{AllocationConfig, AllocationStrategy, BitrateAllocator, VbvAwareAllocator};
use complexity::ComplexityAnalyzer;
use lookahead::{LookaheadBuffer, LookaheadConfig};
use stats::{FrameStatistics, PassStatistics};
use vbv::{VbvBuffer, VbvConfig};

// Re-export main types
pub use allocation::{FrameAllocation, VbvAwareAllocator as Allocator};
pub use complexity::{ComplexityAnalyzer as Analyzer, FrameComplexity};
pub use lookahead::{LookaheadAnalysis, LookaheadFrame, SceneChangeDetector};
pub use stats::{ComplexityStats, FrameStatistics as Stats, PassStatistics as PassStats};
pub use vbv::{VbvBuffer as Buffer, VbvConfig as BufferConfig, VbvStatistics};

/// Pass type for multi-pass encoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PassType {
    /// First pass: collect statistics only.
    FirstPass,
    /// Second pass: encode with optimal bitrate allocation.
    SecondPass,
    /// Single pass with look-ahead (no statistics file).
    SinglePassLookahead,
}

/// Multi-pass encoder configuration.
#[derive(Clone, Debug)]
pub struct EncoderConfig {
    /// Video width in pixels.
    pub width: u32,
    /// Video height in pixels.
    pub height: u32,
    /// Pass type.
    pub pass: PassType,
    /// Number of lookahead frames (10-250).
    pub lookahead_frames: usize,
    /// Target bitrate in bits per second.
    pub target_bitrate: u64,
    /// Maximum bitrate (for VBV).
    pub max_bitrate: Option<u64>,
    /// VBV buffer size in bits.
    pub vbv_buffer_size: Option<u64>,
    /// Frame rate numerator.
    pub framerate_num: u32,
    /// Frame rate denominator.
    pub framerate_den: u32,
    /// Minimum keyframe interval.
    pub min_keyint: u32,
    /// Maximum keyframe interval.
    pub max_keyint: u32,
    /// Scene change threshold (0.0-1.0).
    pub scene_change_threshold: f64,
    /// Enable adaptive quantization.
    pub enable_aq: bool,
    /// Statistics file path (for two-pass).
    pub stats_file: Option<String>,
    /// Bitrate allocation strategy.
    pub allocation_strategy: AllocationStrategy,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
            pass: PassType::SinglePassLookahead,
            lookahead_frames: 40,
            target_bitrate: 5_000_000,
            max_bitrate: None,
            vbv_buffer_size: None,
            framerate_num: 30,
            framerate_den: 1,
            min_keyint: 10,
            max_keyint: 250,
            scene_change_threshold: 0.4,
            enable_aq: true,
            stats_file: None,
            allocation_strategy: AllocationStrategy::Complexity,
        }
    }
}

impl EncoderConfig {
    /// Create a new encoder configuration.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            ..Default::default()
        }
    }

    /// Set pass type.
    #[must_use]
    pub fn with_pass(mut self, pass: PassType) -> Self {
        self.pass = pass;
        self
    }

    /// Set lookahead frames.
    #[must_use]
    pub fn with_lookahead_frames(mut self, frames: usize) -> Self {
        self.lookahead_frames = frames.clamp(10, 250);
        self
    }

    /// Set target bitrate.
    #[must_use]
    pub fn with_target_bitrate(mut self, bitrate: u64) -> Self {
        self.target_bitrate = bitrate;
        self
    }

    /// Set VBV parameters.
    #[must_use]
    pub fn with_vbv(mut self, buffer_size: u64, max_bitrate: u64) -> Self {
        self.vbv_buffer_size = Some(buffer_size);
        self.max_bitrate = Some(max_bitrate);
        self
    }

    /// Set frame rate.
    #[must_use]
    pub fn with_framerate(mut self, num: u32, den: u32) -> Self {
        self.framerate_num = num;
        self.framerate_den = den;
        self
    }

    /// Set keyframe interval range.
    #[must_use]
    pub fn with_keyint_range(mut self, min: u32, max: u32) -> Self {
        self.min_keyint = min;
        self.max_keyint = max;
        self
    }

    /// Set statistics file path.
    #[must_use]
    pub fn with_stats_file(mut self, path: impl Into<String>) -> Self {
        self.stats_file = Some(path.into());
        self
    }

    /// Set allocation strategy.
    #[must_use]
    pub fn with_allocation_strategy(mut self, strategy: AllocationStrategy) -> Self {
        self.allocation_strategy = strategy;
        self
    }
}

/// Multi-pass encoder state.
pub struct MultiPassEncoder {
    config: EncoderConfig,
    pass: PassType,
    lookahead_buffer: Option<LookaheadBuffer>,
    complexity_analyzer: ComplexityAnalyzer,
    bitrate_allocator: BitrateAllocator,
    vbv_buffer: Option<VbvBuffer>,
    pass_statistics: PassStatistics,
    frame_count: u64,
}

impl MultiPassEncoder {
    /// Create a new multi-pass encoder.
    #[must_use]
    pub fn new(config: EncoderConfig) -> Self {
        let complexity_analyzer = ComplexityAnalyzer::new(config.width, config.height);

        let lookahead_buffer = if config.pass == PassType::SinglePassLookahead
            || config.pass == PassType::SecondPass
        {
            let lookahead_config = LookaheadConfig::new(config.lookahead_frames)
                .with_keyint_range(config.min_keyint, config.max_keyint)
                .with_scene_threshold(config.scene_change_threshold);

            Some(LookaheadBuffer::new(
                lookahead_config,
                config.width,
                config.height,
            ))
        } else {
            None
        };

        let allocation_config =
            AllocationConfig::new(config.allocation_strategy, config.target_bitrate)
                .with_framerate(config.framerate_num, config.framerate_den);

        let bitrate_allocator = BitrateAllocator::new(allocation_config);

        let vbv_buffer = if let (Some(buffer_size), Some(max_bitrate)) =
            (config.vbv_buffer_size, config.max_bitrate)
        {
            let vbv_config = VbvConfig::new(
                buffer_size,
                max_bitrate,
                config.framerate_num,
                config.framerate_den,
            );
            Some(VbvBuffer::new(vbv_config))
        } else {
            None
        };

        let pass_statistics = PassStatistics::new(
            config.width,
            config.height,
            config.framerate_num,
            config.framerate_den,
        );

        Self {
            pass: config.pass,
            config,
            lookahead_buffer,
            complexity_analyzer,
            bitrate_allocator,
            vbv_buffer,
            pass_statistics,
            frame_count: 0,
        }
    }

    /// Encode a frame (behavior depends on pass type).
    pub fn encode_frame(
        &mut self,
        frame: &VideoFrame,
    ) -> Result<Option<EncodingResult>, EncoderError> {
        match self.pass {
            PassType::FirstPass => self.encode_first_pass(frame),
            PassType::SecondPass => self.encode_second_pass(frame),
            PassType::SinglePassLookahead => self.encode_single_pass(frame),
        }
    }

    /// First pass: analyze and collect statistics.
    fn encode_first_pass(
        &mut self,
        frame: &VideoFrame,
    ) -> Result<Option<EncodingResult>, EncoderError> {
        // Analyze frame complexity
        let complexity = self.complexity_analyzer.analyze(frame, self.frame_count);

        // Allocate bits (basic allocation for first pass)
        let allocation = self.bitrate_allocator.allocate(
            self.frame_count,
            frame.frame_type,
            complexity.combined_complexity,
        );

        // Store statistics
        let frame_stats = FrameStatistics::new(
            self.frame_count,
            frame.frame_type,
            28.0, // Dummy QP for first pass
            allocation.target_bits,
            complexity,
        );

        self.pass_statistics.add_frame(frame_stats);
        self.frame_count += 1;

        // First pass doesn't produce packets
        Ok(None)
    }

    /// Second pass: encode with optimal bitrate allocation.
    fn encode_second_pass(
        &mut self,
        frame: &VideoFrame,
    ) -> Result<Option<EncodingResult>, EncoderError> {
        // Use lookahead if available
        if let Some(ref mut lookahead) = self.lookahead_buffer {
            lookahead.add_frame(frame.clone());

            // Wait until lookahead is full (not flushing in this implementation)
            if !lookahead.is_full() {
                return Ok(None);
            }

            // Get next frame from lookahead
            if let Some(lookahead_frame) = lookahead.get_next_frame() {
                let complexity = lookahead_frame.complexity.combined_complexity;
                let allocation = self.bitrate_allocator.allocate(
                    self.frame_count,
                    lookahead_frame.assigned_type,
                    complexity,
                );

                let result = self.create_encoding_result(
                    &lookahead_frame.frame,
                    lookahead_frame.assigned_type,
                    allocation,
                    lookahead_frame.qp_offset,
                );

                self.frame_count += 1;
                return Ok(Some(result));
            }
        } else {
            // No lookahead, encode directly
            let complexity = self.complexity_analyzer.analyze(frame, self.frame_count);
            let allocation = self.bitrate_allocator.allocate(
                self.frame_count,
                frame.frame_type,
                complexity.combined_complexity,
            );

            let result = self.create_encoding_result(frame, frame.frame_type, allocation, 0);
            self.frame_count += 1;
            return Ok(Some(result));
        }

        Ok(None)
    }

    /// Single pass with lookahead.
    fn encode_single_pass(
        &mut self,
        frame: &VideoFrame,
    ) -> Result<Option<EncodingResult>, EncoderError> {
        if let Some(ref mut lookahead) = self.lookahead_buffer {
            lookahead.add_frame(frame.clone());

            // Wait until lookahead is full (not flushing in this implementation)
            if !lookahead.is_full() {
                return Ok(None);
            }

            // Get next frame from lookahead
            if let Some(lookahead_frame) = lookahead.get_next_frame() {
                let complexity = lookahead_frame.complexity.combined_complexity;
                let allocation = self.bitrate_allocator.allocate(
                    self.frame_count,
                    lookahead_frame.assigned_type,
                    complexity,
                );

                // Apply VBV constraints if enabled
                let target_bits = if let Some(ref vbv) = self.vbv_buffer {
                    vbv.target_frame_size(
                        lookahead_frame.assigned_type,
                        allocation.target_bits as f64,
                    )
                } else {
                    allocation.target_bits
                };

                let mut result = self.create_encoding_result(
                    &lookahead_frame.frame,
                    lookahead_frame.assigned_type,
                    allocation,
                    lookahead_frame.qp_offset,
                );

                result.target_bits = target_bits;

                // Update VBV buffer
                if let Some(ref mut vbv) = self.vbv_buffer {
                    vbv.update(target_bits);
                }

                self.frame_count += 1;
                return Ok(Some(result));
            }
        }

        Ok(None)
    }

    /// Create encoding result from allocation.
    fn create_encoding_result(
        &self,
        frame: &VideoFrame,
        frame_type: FrameType,
        allocation: allocation::FrameAllocation,
        qp_offset: i32,
    ) -> EncodingResult {
        // Calculate QP from target bits (simplified model)
        let base_qp = self.bits_to_qp(allocation.target_bits);
        let adjusted_qp = (base_qp + allocation.qp_adjustment + qp_offset as f64).clamp(1.0, 63.0);

        EncodingResult {
            frame_index: self.frame_count,
            frame_type,
            target_bits: allocation.target_bits,
            min_bits: allocation.min_bits,
            max_bits: allocation.max_bits,
            qp: adjusted_qp,
            complexity: 0.5, // Placeholder
        }
    }

    /// Convert target bits to QP (simplified logarithmic model).
    fn bits_to_qp(&self, target_bits: u64) -> f64 {
        let pixels = (self.config.width as u64) * (self.config.height as u64);
        let bpp = target_bits as f64 / pixels as f64;

        // Logarithmic model: QP ≈ 69.0 - 12.0 * log2(bpp)
        if bpp > 0.0 {
            (69.0 - 12.0 * bpp.log2()).clamp(1.0, 63.0)
        } else {
            51.0 // Default QP
        }
    }

    /// Save first-pass statistics to a file.
    pub fn save_stats(&self, path: &str) -> Result<(), EncoderError> {
        self.pass_statistics
            .save_to_file(path)
            .map_err(|e| EncoderError::IoError(e.to_string()))
    }

    /// Load first-pass statistics from a file.
    pub fn load_stats(&mut self, path: &str) -> Result<(), EncoderError> {
        let stats = PassStatistics::load_from_file(path)
            .map_err(|e| EncoderError::IoError(e.to_string()))?;

        self.bitrate_allocator.set_first_pass_stats(stats);
        Ok(())
    }

    /// Get encoding statistics.
    #[must_use]
    pub fn statistics(&self) -> &PassStatistics {
        &self.pass_statistics
    }

    /// Get VBV statistics if available.
    #[must_use]
    pub fn vbv_statistics(&self) -> Option<VbvStatistics> {
        self.vbv_buffer.as_ref().map(|vbv| vbv.statistics())
    }

    /// Get current frame count.
    #[must_use]
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Reset encoder state.
    pub fn reset(&mut self) {
        self.frame_count = 0;
        self.complexity_analyzer.reset();
        self.bitrate_allocator.reset();

        if let Some(ref mut lookahead) = self.lookahead_buffer {
            lookahead.reset();
        }

        if let Some(ref mut vbv) = self.vbv_buffer {
            vbv.reset();
        }

        self.pass_statistics = PassStatistics::new(
            self.config.width,
            self.config.height,
            self.config.framerate_num,
            self.config.framerate_den,
        );
    }
}

/// Encoding result with bitrate allocation.
#[derive(Clone, Debug)]
pub struct EncodingResult {
    /// Frame index.
    pub frame_index: u64,
    /// Frame type.
    pub frame_type: FrameType,
    /// Target bits for this frame.
    pub target_bits: u64,
    /// Minimum acceptable bits.
    pub min_bits: u64,
    /// Maximum allowed bits.
    pub max_bits: u64,
    /// Quantization parameter.
    pub qp: f64,
    /// Frame complexity.
    pub complexity: f64,
}

/// Encoder error types.
#[derive(Debug)]
pub enum EncoderError {
    /// I/O error.
    IoError(String),
    /// Configuration error.
    ConfigError(String),
    /// Encoding error.
    EncodingError(String),
}

impl std::fmt::Display for EncoderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IoError(msg) => write!(f, "I/O error: {}", msg),
            Self::ConfigError(msg) => write!(f, "Configuration error: {}", msg),
            Self::EncodingError(msg) => write!(f, "Encoding error: {}", msg),
        }
    }
}

impl std::error::Error for EncoderError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::Plane;
    use oximedia_core::{PixelFormat, Rational, Timestamp};

    fn create_test_frame(width: u32, height: u32) -> VideoFrame {
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, width, height);
        let size = (width * height) as usize;
        let data = vec![128u8; size];
        frame.planes.push(Plane::new(data, width as usize));
        frame.timestamp = Timestamp::new(0, Rational::new(1, 30));
        frame
    }

    #[test]
    fn test_encoder_config_new() {
        let config = EncoderConfig::new(1920, 1080);
        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
        assert_eq!(config.pass, PassType::SinglePassLookahead);
    }

    #[test]
    fn test_encoder_config_builder() {
        let config = EncoderConfig::new(1920, 1080)
            .with_pass(PassType::FirstPass)
            .with_lookahead_frames(50)
            .with_target_bitrate(10_000_000);

        assert_eq!(config.pass, PassType::FirstPass);
        assert_eq!(config.lookahead_frames, 50);
        assert_eq!(config.target_bitrate, 10_000_000);
    }

    #[test]
    fn test_multipass_encoder_new() {
        let config = EncoderConfig::new(1920, 1080);
        let encoder = MultiPassEncoder::new(config);
        assert_eq!(encoder.frame_count(), 0);
    }

    #[test]
    fn test_first_pass_encoding() {
        let config = EncoderConfig::new(320, 240).with_pass(PassType::FirstPass);
        let mut encoder = MultiPassEncoder::new(config);

        let frame = create_test_frame(320, 240);
        let result = encoder.encode_frame(&frame);

        assert!(result.is_ok());
        assert!(result.expect("should succeed").is_none()); // First pass returns no packets
        assert_eq!(encoder.frame_count(), 1);
    }

    #[test]
    fn test_single_pass_lookahead() {
        let config = EncoderConfig::new(320, 240)
            .with_pass(PassType::SinglePassLookahead)
            .with_lookahead_frames(10);

        let mut encoder = MultiPassEncoder::new(config);

        // Add frames to fill lookahead buffer
        for _ in 0..15 {
            let frame = create_test_frame(320, 240);
            let result = encoder.encode_frame(&frame);
            assert!(result.is_ok());
        }

        assert!(encoder.frame_count() > 0);
    }
}
