//! Examples and usage documentation for multi-pass encoding.
//!
//! This module provides comprehensive examples of how to use the multi-pass
//! encoding system in various scenarios.

#![forbid(unsafe_code)]
#![allow(dead_code)]

use crate::frame::{FrameType, VideoFrame};
use crate::multipass::{
    allocation::AllocationStrategy, EncoderConfig, EncodingResult, MultiPassEncoder, PassType,
};

/// Example: Two-pass encoding for optimal quality.
///
/// This example demonstrates how to perform two-pass encoding to achieve
/// optimal bitrate allocation across frames.
pub struct TwoPassExample {
    width: u32,
    height: u32,
    target_bitrate: u64,
    stats_file: String,
}

impl TwoPassExample {
    /// Create a new two-pass encoding example.
    #[must_use]
    pub fn new(width: u32, height: u32, target_bitrate: u64) -> Self {
        Self {
            width,
            height,
            target_bitrate,
            stats_file: std::env::temp_dir()
                .join("oximedia-multipass-stats.txt")
                .to_string_lossy()
                .into_owned(),
        }
    }

    /// Run the first pass to collect statistics.
    pub fn run_first_pass(&self, frames: &[VideoFrame]) -> Result<(), String> {
        // Configure first pass
        let config = EncoderConfig::new(self.width, self.height)
            .with_pass(PassType::FirstPass)
            .with_target_bitrate(self.target_bitrate)
            .with_stats_file(&self.stats_file);

        let mut encoder = MultiPassEncoder::new(config);

        // Process all frames
        for frame in frames {
            encoder
                .encode_frame(frame)
                .map_err(|e| format!("First pass encoding failed: {}", e))?;
        }

        // Save statistics
        encoder
            .save_stats(&self.stats_file)
            .map_err(|e| format!("Failed to save stats: {}", e))?;

        Ok(())
    }

    /// Run the second pass with optimal bitrate allocation.
    pub fn run_second_pass(&self, frames: &[VideoFrame]) -> Result<Vec<EncodingResult>, String> {
        // Configure second pass
        let config = EncoderConfig::new(self.width, self.height)
            .with_pass(PassType::SecondPass)
            .with_target_bitrate(self.target_bitrate)
            .with_stats_file(&self.stats_file)
            .with_lookahead_frames(40)
            .with_allocation_strategy(AllocationStrategy::TwoPass);

        let mut encoder = MultiPassEncoder::new(config);

        // Load first-pass statistics
        encoder
            .load_stats(&self.stats_file)
            .map_err(|e| format!("Failed to load stats: {}", e))?;

        let mut results = Vec::new();

        // Process all frames with optimal allocation
        for frame in frames {
            if let Some(result) = encoder
                .encode_frame(frame)
                .map_err(|e| format!("Second pass encoding failed: {}", e))?
            {
                results.push(result);
            }
        }

        Ok(results)
    }
}

/// Example: Single-pass encoding with look-ahead for live streaming.
///
/// This example shows how to use look-ahead in single-pass mode for
/// applications where two-pass encoding is not feasible.
pub struct SinglePassLookaheadExample {
    width: u32,
    height: u32,
    target_bitrate: u64,
    lookahead_frames: usize,
}

impl SinglePassLookaheadExample {
    /// Create a new single-pass look-ahead example.
    #[must_use]
    pub fn new(width: u32, height: u32, target_bitrate: u64, lookahead_frames: usize) -> Self {
        Self {
            width,
            height,
            target_bitrate,
            lookahead_frames,
        }
    }

    /// Run single-pass encoding with look-ahead.
    pub fn run(&self, frames: &[VideoFrame]) -> Result<Vec<EncodingResult>, String> {
        // Configure single-pass with look-ahead
        let config = EncoderConfig::new(self.width, self.height)
            .with_pass(PassType::SinglePassLookahead)
            .with_target_bitrate(self.target_bitrate)
            .with_lookahead_frames(self.lookahead_frames)
            .with_allocation_strategy(AllocationStrategy::Perceptual);

        let mut encoder = MultiPassEncoder::new(config);
        let mut results = Vec::new();

        // Process frames with look-ahead
        for frame in frames {
            if let Some(result) = encoder
                .encode_frame(frame)
                .map_err(|e| format!("Encoding failed: {}", e))?
            {
                results.push(result);
            }
        }

        Ok(results)
    }
}

/// Example: VBV-constrained encoding for streaming.
///
/// This example demonstrates how to use VBV buffer constraints for
/// streaming applications that require strict bitrate compliance.
pub struct VbvConstrainedExample {
    width: u32,
    height: u32,
    target_bitrate: u64,
    max_bitrate: u64,
    buffer_size: u64,
}

impl VbvConstrainedExample {
    /// Create a new VBV-constrained encoding example.
    #[must_use]
    pub fn new(
        width: u32,
        height: u32,
        target_bitrate: u64,
        max_bitrate: u64,
        buffer_size: u64,
    ) -> Self {
        Self {
            width,
            height,
            target_bitrate,
            max_bitrate,
            buffer_size,
        }
    }

    /// Run VBV-constrained encoding.
    pub fn run(&self, frames: &[VideoFrame]) -> Result<(Vec<EncodingResult>, VbvReport), String> {
        // Configure with VBV constraints
        let config = EncoderConfig::new(self.width, self.height)
            .with_pass(PassType::SinglePassLookahead)
            .with_target_bitrate(self.target_bitrate)
            .with_vbv(self.buffer_size, self.max_bitrate)
            .with_lookahead_frames(40);

        let mut encoder = MultiPassEncoder::new(config);
        let mut results = Vec::new();

        // Process frames with VBV constraints
        for frame in frames {
            if let Some(result) = encoder
                .encode_frame(frame)
                .map_err(|e| format!("VBV encoding failed: {}", e))?
            {
                results.push(result);
            }
        }

        // Get VBV statistics
        let vbv_stats = encoder
            .vbv_statistics()
            .ok_or("VBV statistics not available")?;

        let report = VbvReport {
            is_compliant: vbv_stats.is_compliant(),
            underflow_count: vbv_stats.underflow_count,
            overflow_count: vbv_stats.overflow_count,
            utilization: vbv_stats.utilization(),
        };

        Ok((results, report))
    }
}

/// VBV compliance report.
#[derive(Clone, Debug)]
pub struct VbvReport {
    /// Whether encoding was VBV compliant.
    pub is_compliant: bool,
    /// Number of buffer underflows.
    pub underflow_count: u64,
    /// Number of buffer overflows.
    pub overflow_count: u64,
    /// Buffer utilization.
    pub utilization: crate::multipass::vbv::BufferUtilization,
}

/// Example: Scene change detection with adaptive keyframes.
///
/// This example shows how the encoder automatically detects scene changes
/// and inserts keyframes at appropriate locations.
pub struct SceneChangeExample {
    width: u32,
    height: u32,
    scene_threshold: f64,
    min_keyint: u32,
    max_keyint: u32,
}

impl SceneChangeExample {
    /// Create a new scene change detection example.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            scene_threshold: 0.4,
            min_keyint: 10,
            max_keyint: 250,
        }
    }

    /// Set scene change detection parameters.
    #[must_use]
    pub fn with_params(mut self, threshold: f64, min_keyint: u32, max_keyint: u32) -> Self {
        self.scene_threshold = threshold;
        self.min_keyint = min_keyint;
        self.max_keyint = max_keyint;
        self
    }

    /// Run encoding with scene change detection.
    pub fn run(&self, frames: &[VideoFrame]) -> Result<SceneChangeReport, String> {
        // Configure with scene change detection
        let config = EncoderConfig::new(self.width, self.height)
            .with_pass(PassType::SinglePassLookahead)
            .with_lookahead_frames(40)
            .with_keyint_range(self.min_keyint, self.max_keyint);

        let mut encoder = MultiPassEncoder::new(config);
        let mut keyframe_positions = Vec::new();
        let mut total_frames = 0;

        // Process frames and track keyframes
        for frame in frames {
            if let Some(result) = encoder
                .encode_frame(frame)
                .map_err(|e| format!("Encoding failed: {}", e))?
            {
                if result.frame_type == FrameType::Key {
                    keyframe_positions.push(result.frame_index);
                }
                total_frames += 1;
            }
        }

        let keyframe_count = keyframe_positions.len();
        let avg_gop_length = if keyframe_count > 1 {
            total_frames as f64 / keyframe_count as f64
        } else {
            total_frames as f64
        };

        let report = SceneChangeReport {
            total_frames,
            keyframe_count,
            keyframe_positions,
            avg_gop_length,
        };

        Ok(report)
    }
}

/// Scene change detection report.
#[derive(Clone, Debug)]
pub struct SceneChangeReport {
    /// Total frames processed.
    pub total_frames: u64,
    /// Number of keyframes inserted.
    pub keyframe_count: usize,
    /// Positions of keyframes in the stream.
    pub keyframe_positions: Vec<u64>,
    /// Average GOP (Group of Pictures) length.
    pub avg_gop_length: f64,
}

/// Example: Adaptive quantization based on frame complexity.
///
/// This example demonstrates how the encoder adjusts QP based on frame
/// complexity to maintain consistent perceptual quality.
pub struct AdaptiveQuantizationExample {
    width: u32,
    height: u32,
    target_bitrate: u64,
}

impl AdaptiveQuantizationExample {
    /// Create a new adaptive quantization example.
    #[must_use]
    pub fn new(width: u32, height: u32, target_bitrate: u64) -> Self {
        Self {
            width,
            height,
            target_bitrate,
        }
    }

    /// Run encoding with adaptive quantization.
    pub fn run(&self, frames: &[VideoFrame]) -> Result<AqReport, String> {
        // Configure with perceptual allocation (uses AQ).
        // Use a lookahead that fits within the number of available frames.
        let lookahead = if frames.len() > 10 {
            frames.len() / 3
        } else {
            10
        };
        let config = EncoderConfig::new(self.width, self.height)
            .with_pass(PassType::SinglePassLookahead)
            .with_target_bitrate(self.target_bitrate)
            .with_lookahead_frames(lookahead)
            .with_allocation_strategy(AllocationStrategy::Perceptual);

        let mut encoder = MultiPassEncoder::new(config);
        let mut qp_values = Vec::new();
        let mut complexity_values = Vec::new();

        // Process frames and track QP and complexity
        for frame in frames {
            if let Some(result) = encoder
                .encode_frame(frame)
                .map_err(|e| format!("AQ encoding failed: {}", e))?
            {
                qp_values.push(result.qp);
                complexity_values.push(result.complexity);
            }
        }

        // Calculate statistics
        let avg_qp = qp_values.iter().sum::<f64>() / qp_values.len() as f64;
        let avg_complexity = complexity_values.iter().sum::<f64>() / complexity_values.len() as f64;

        let min_qp = qp_values
            .iter()
            .copied()
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);

        let max_qp = qp_values
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);

        let report = AqReport {
            frame_count: qp_values.len(),
            avg_qp,
            min_qp,
            max_qp,
            qp_range: max_qp - min_qp,
            avg_complexity,
        };

        Ok(report)
    }
}

/// Adaptive quantization report.
#[derive(Clone, Debug)]
pub struct AqReport {
    /// Total frames processed.
    pub frame_count: usize,
    /// Average QP across all frames.
    pub avg_qp: f64,
    /// Minimum QP used.
    pub min_qp: f64,
    /// Maximum QP used.
    pub max_qp: f64,
    /// QP range (max - min).
    pub qp_range: f64,
    /// Average frame complexity.
    pub avg_complexity: f64,
}

/// Example: Comparing different allocation strategies.
///
/// This example compares the performance of different bitrate allocation
/// strategies on the same content.
pub struct AllocationComparisonExample {
    width: u32,
    height: u32,
    target_bitrate: u64,
}

impl AllocationComparisonExample {
    /// Create a new allocation comparison example.
    #[must_use]
    pub fn new(width: u32, height: u32, target_bitrate: u64) -> Self {
        Self {
            width,
            height,
            target_bitrate,
        }
    }

    /// Run comparison across all allocation strategies.
    pub fn run(&self, frames: &[VideoFrame]) -> Result<ComparisonReport, String> {
        let strategies = [
            AllocationStrategy::Uniform,
            AllocationStrategy::Complexity,
            AllocationStrategy::Perceptual,
        ];

        let mut results = Vec::new();

        for strategy in &strategies {
            let config = EncoderConfig::new(self.width, self.height)
                .with_pass(PassType::SinglePassLookahead)
                .with_target_bitrate(self.target_bitrate)
                .with_lookahead_frames(40)
                .with_allocation_strategy(*strategy);

            let mut encoder = MultiPassEncoder::new(config);
            let mut total_bits = 0u64;
            let mut qp_values = Vec::new();

            for frame in frames {
                if let Some(result) = encoder
                    .encode_frame(frame)
                    .map_err(|e| format!("Comparison encoding failed: {}", e))?
                {
                    total_bits += result.target_bits;
                    qp_values.push(result.qp);
                }
            }

            let avg_qp = qp_values.iter().sum::<f64>() / qp_values.len() as f64;
            let qp_variance = qp_values
                .iter()
                .map(|qp| (qp - avg_qp).powi(2))
                .sum::<f64>()
                / qp_values.len() as f64;

            results.push(StrategyResult {
                strategy: *strategy,
                total_bits,
                avg_qp,
                qp_variance,
                frame_count: qp_values.len(),
            });
        }

        Ok(ComparisonReport { results })
    }
}

/// Comparison report across strategies.
#[derive(Clone, Debug)]
pub struct ComparisonReport {
    /// Results for each strategy.
    pub results: Vec<StrategyResult>,
}

/// Result for a single allocation strategy.
#[derive(Clone, Debug)]
pub struct StrategyResult {
    /// Allocation strategy used.
    pub strategy: AllocationStrategy,
    /// Total bits used.
    pub total_bits: u64,
    /// Average QP.
    pub avg_qp: f64,
    /// QP variance (measure of consistency).
    pub qp_variance: f64,
    /// Number of frames encoded.
    pub frame_count: usize,
}

impl ComparisonReport {
    /// Find the strategy with lowest QP variance (most consistent quality).
    #[must_use]
    pub fn most_consistent(&self) -> Option<&StrategyResult> {
        self.results.iter().min_by(|a, b| {
            a.qp_variance
                .partial_cmp(&b.qp_variance)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Find the strategy that used fewest bits.
    #[must_use]
    pub fn most_efficient(&self) -> Option<&StrategyResult> {
        self.results.iter().min_by_key(|r| r.total_bits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::Plane;
    use oximedia_core::{PixelFormat, Rational, Timestamp};

    fn create_test_frames(count: usize) -> Vec<VideoFrame> {
        (0..count)
            .map(|i| {
                let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 320, 240);
                let size = 320 * 240;
                let data = vec![(i % 256) as u8; size];
                frame.planes.push(Plane::new(data, 320));
                frame.timestamp = Timestamp::new(i as i64, Rational::new(1, 30));
                frame
            })
            .collect()
    }

    #[test]
    fn test_single_pass_lookahead_example() {
        let example = SinglePassLookaheadExample::new(320, 240, 1_000_000, 20);
        let frames = create_test_frames(30);
        let result = example.run(&frames);
        assert!(result.is_ok());
    }

    #[test]
    fn test_scene_change_example() {
        let example = SceneChangeExample::new(320, 240);
        let frames = create_test_frames(50);
        let result = example.run(&frames);
        assert!(result.is_ok());
    }

    #[test]
    fn test_adaptive_quantization_example() {
        let example = AdaptiveQuantizationExample::new(320, 240, 1_000_000);
        let frames = create_test_frames(30);
        let result = example.run(&frames);
        assert!(result.is_ok());

        if let Ok(report) = result {
            assert!(report.frame_count > 0);
            assert!(report.avg_qp > 0.0);
        }
    }

    #[test]
    fn test_allocation_comparison_example() {
        let example = AllocationComparisonExample::new(320, 240, 1_000_000);
        let frames = create_test_frames(20);
        let result = example.run(&frames);
        assert!(result.is_ok());

        if let Ok(report) = result {
            assert_eq!(report.results.len(), 3);
            assert!(report.most_consistent().is_some());
            assert!(report.most_efficient().is_some());
        }
    }
}
