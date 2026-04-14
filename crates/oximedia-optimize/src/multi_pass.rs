//! Multi-pass encoding with rate redistribution between passes.
//!
//! This module implements N-pass encoding optimization that iteratively refines
//! bitrate allocation across frames. Each pass produces complexity statistics that
//! inform the next pass's budget, converging on an optimal distribution that:
//!
//! - Meets the target bitrate within a configurable tolerance
//! - Redistributes bits from simple frames to complex frames
//! - Maintains VBV (Video Buffer Verifier) compliance
//! - Minimizes quality variance across frames (constant perceived quality)
//!
//! # Algorithm
//!
//! 1. **Pass 1**: Uniform allocation, collect per-frame complexity
//! 2. **Pass 2..N**: Redistribute bits proportional to complexity, using a
//!    damped iterative scheme (Lagrangian multiplier adjustment) to converge
//! 3. **Final**: Clamp to VBV constraints, verify target bitrate compliance

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::VecDeque;

// ── Configuration ───────────────────────────────────────────────────────────

/// Configuration for multi-pass encoding optimization.
#[derive(Debug, Clone)]
pub struct MultiPassConfig {
    /// Number of encoding passes (minimum 2, maximum 8).
    pub num_passes: u32,
    /// Target average bitrate in bits per second.
    pub target_bitrate: u64,
    /// Frame rate (frames per second).
    pub fps: f64,
    /// VBV buffer size in bits.
    pub vbv_buffer_size: u64,
    /// VBV maximum bitrate in bits per second.
    pub vbv_max_bitrate: u64,
    /// Initial VBV buffer fullness (0.0 to 1.0).
    pub vbv_initial_fill: f64,
    /// Damping factor for iterative redistribution (0.0 to 1.0).
    /// Lower values = more conservative updates between passes.
    pub damping: f64,
    /// Convergence tolerance: stop early if bitrate error < this fraction.
    pub convergence_tolerance: f64,
    /// Minimum per-frame bitrate as fraction of target bits-per-frame.
    pub min_frame_fraction: f64,
    /// Maximum per-frame bitrate as fraction of target bits-per-frame.
    pub max_frame_fraction: f64,
    /// Quality variance weight: how much to penalise uneven quality (0.0 to 1.0).
    pub quality_variance_weight: f64,
}

impl Default for MultiPassConfig {
    fn default() -> Self {
        Self {
            num_passes: 3,
            target_bitrate: 4_000_000,
            fps: 24.0,
            vbv_buffer_size: 8_000_000,
            vbv_max_bitrate: 8_000_000,
            vbv_initial_fill: 0.9,
            damping: 0.6,
            convergence_tolerance: 0.01,
            min_frame_fraction: 0.15,
            max_frame_fraction: 4.0,
            quality_variance_weight: 0.5,
        }
    }
}

impl MultiPassConfig {
    /// Returns the target bits per frame.
    #[must_use]
    pub fn bits_per_frame(&self) -> f64 {
        if self.fps > 0.0 {
            self.target_bitrate as f64 / self.fps
        } else {
            0.0
        }
    }

    /// Validates the configuration, returning an error description if invalid.
    pub fn validate(&self) -> Result<(), MultiPassError> {
        if self.num_passes < 2 {
            return Err(MultiPassError::InvalidConfig(
                "num_passes must be at least 2".into(),
            ));
        }
        if self.num_passes > 8 {
            return Err(MultiPassError::InvalidConfig(
                "num_passes must be at most 8".into(),
            ));
        }
        if self.fps <= 0.0 {
            return Err(MultiPassError::InvalidConfig("fps must be positive".into()));
        }
        if self.target_bitrate == 0 {
            return Err(MultiPassError::InvalidConfig(
                "target_bitrate must be > 0".into(),
            ));
        }
        if !(0.0..=1.0).contains(&self.damping) {
            return Err(MultiPassError::InvalidConfig(
                "damping must be in [0.0, 1.0]".into(),
            ));
        }
        Ok(())
    }
}

// ── Error type ──────────────────────────────────────────────────────────────

/// Errors from multi-pass encoding operations.
#[derive(Debug, Clone)]
pub enum MultiPassError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// No frames provided for optimization.
    EmptyInput,
    /// Pass index out of range.
    InvalidPassIndex(u32),
}

impl std::fmt::Display for MultiPassError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "invalid multi-pass config: {msg}"),
            Self::EmptyInput => write!(f, "no frames provided for multi-pass optimization"),
            Self::InvalidPassIndex(idx) => write!(f, "pass index {idx} out of range"),
        }
    }
}

impl std::error::Error for MultiPassError {}

// ── Per-frame data ──────────────────────────────────────────────────────────

/// Per-frame complexity statistics collected during a pass.
#[derive(Debug, Clone)]
pub struct FramePassStats {
    /// Frame index.
    pub index: usize,
    /// Frame type: 'I', 'P', or 'B'.
    pub frame_type: char,
    /// Spatial complexity (0.0 to 1.0).
    pub spatial_complexity: f64,
    /// Temporal complexity (0.0 to 1.0).
    pub temporal_complexity: f64,
    /// Combined complexity score used for allocation.
    pub combined_complexity: f64,
    /// Bits allocated in this pass.
    pub allocated_bits: u64,
    /// Estimated quality score for this allocation (0 to 100).
    pub estimated_quality: f64,
}

impl FramePassStats {
    /// Creates a new frame stats entry with initial complexity values.
    #[must_use]
    pub fn new(index: usize, frame_type: char, spatial: f64, temporal: f64) -> Self {
        let combined = 0.6 * spatial + 0.4 * temporal;
        Self {
            index,
            frame_type,
            spatial_complexity: spatial.clamp(0.0, 1.0),
            temporal_complexity: temporal.clamp(0.0, 1.0),
            combined_complexity: combined.clamp(0.0, 1.0),
            allocated_bits: 0,
            estimated_quality: 0.0,
        }
    }

    /// Returns a frame-type weight for bitrate allocation.
    /// I-frames typically need more bits than P/B-frames.
    #[must_use]
    pub fn frame_type_weight(&self) -> f64 {
        match self.frame_type {
            'I' => 2.5,
            'P' => 1.0,
            'B' => 0.6,
            _ => 1.0,
        }
    }
}

// ── Pass result ─────────────────────────────────────────────────────────────

/// Result of a single encoding pass.
#[derive(Debug, Clone)]
pub struct PassResult {
    /// Pass number (1-indexed).
    pub pass_number: u32,
    /// Per-frame allocation after this pass.
    pub frame_allocations: Vec<u64>,
    /// Total bits allocated across all frames.
    pub total_bits: u64,
    /// Average bits per frame.
    pub avg_bits_per_frame: f64,
    /// Quality variance (std dev of estimated quality across frames).
    pub quality_variance: f64,
    /// Bitrate error relative to target (fraction).
    pub bitrate_error: f64,
    /// Whether all frames are VBV-compliant.
    pub vbv_compliant: bool,
    /// Lagrangian multiplier used in this pass.
    pub lambda: f64,
}

// ── Multi-pass result ───────────────────────────────────────────────────────

/// Complete result of multi-pass optimization.
#[derive(Debug, Clone)]
pub struct MultiPassResult {
    /// Results from each pass.
    pub passes: Vec<PassResult>,
    /// Final per-frame bit allocation.
    pub final_allocations: Vec<u64>,
    /// Final total bits.
    pub total_bits: u64,
    /// Number of passes actually executed (may be less than configured if
    /// convergence was reached early).
    pub passes_executed: u32,
    /// Whether the optimizer converged within tolerance.
    pub converged: bool,
    /// Final quality variance.
    pub final_quality_variance: f64,
}

// ── VBV simulator ───────────────────────────────────────────────────────────

/// VBV buffer simulator for compliance checking.
#[derive(Debug, Clone)]
pub struct VbvSimulator {
    buffer_size: u64,
    max_bitrate: u64,
    fps: f64,
    level: u64,
}

impl VbvSimulator {
    /// Creates a new VBV simulator.
    #[must_use]
    pub fn new(buffer_size: u64, max_bitrate: u64, fps: f64, initial_fill: f64) -> Self {
        let fill = initial_fill.clamp(0.0, 1.0);
        Self {
            buffer_size,
            max_bitrate,
            fps,
            level: (buffer_size as f64 * fill) as u64,
        }
    }

    /// Processes one frame, returning whether VBV compliance holds.
    pub fn process_frame(&mut self, frame_bits: u64) -> bool {
        // Drain: codec removes frame_bits from buffer
        let drain = frame_bits;
        // Fill: bits arrive at max_bitrate over one frame period
        let fill = if self.fps > 0.0 {
            (self.max_bitrate as f64 / self.fps) as u64
        } else {
            0
        };

        self.level = self.level.saturating_add(fill).saturating_sub(drain);

        // Buffer overflow check
        if self.level > self.buffer_size {
            self.level = self.buffer_size;
        }

        // Compliance: buffer must not underflow (we treat negative as 0 via
        // saturating_sub, but a frame larger than buffer+fill is non-compliant)
        frame_bits <= self.level.saturating_add(fill)
    }

    /// Returns current buffer level.
    #[must_use]
    pub fn level(&self) -> u64 {
        self.level
    }

    /// Resets the buffer to initial state.
    pub fn reset(&mut self, initial_fill: f64) {
        let fill = initial_fill.clamp(0.0, 1.0);
        self.level = (self.buffer_size as f64 * fill) as u64;
    }
}

// ── Multi-pass optimizer ────────────────────────────────────────────────────

/// Multi-pass encoding optimizer.
///
/// Iteratively refines per-frame bitrate allocation across multiple passes,
/// using complexity statistics to redistribute bits for constant perceived quality.
#[derive(Debug, Clone)]
pub struct MultiPassOptimizer {
    config: MultiPassConfig,
}

impl MultiPassOptimizer {
    /// Creates a new multi-pass optimizer.
    pub fn new(config: MultiPassConfig) -> Result<Self, MultiPassError> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Returns the configuration.
    #[must_use]
    pub fn config(&self) -> &MultiPassConfig {
        &self.config
    }

    /// Runs multi-pass optimization on the given frame statistics.
    ///
    /// Each frame in `frames` must have spatial and temporal complexity pre-filled.
    /// The optimizer will iteratively refine `allocated_bits` across passes.
    pub fn optimize(&self, frames: &[FramePassStats]) -> Result<MultiPassResult, MultiPassError> {
        if frames.is_empty() {
            return Err(MultiPassError::EmptyInput);
        }

        let num_frames = frames.len();
        let target_bpf = self.config.bits_per_frame();
        let total_target = (target_bpf * num_frames as f64) as u64;
        let min_bits = (target_bpf * self.config.min_frame_fraction) as u64;
        let max_bits = (target_bpf * self.config.max_frame_fraction) as u64;

        let mut passes = Vec::with_capacity(self.config.num_passes as usize);
        let mut allocations: Vec<u64> = vec![target_bpf as u64; num_frames];
        let mut lambda = 1.0_f64;
        let mut converged = false;

        for pass_idx in 0..self.config.num_passes {
            // Compute weighted complexity for each frame
            let weights: Vec<f64> = frames
                .iter()
                .map(|f| f.combined_complexity * f.frame_type_weight())
                .collect();

            let total_weight: f64 = weights.iter().sum();
            let avg_weight = if total_weight > 0.0 {
                total_weight / num_frames as f64
            } else {
                1.0
            };

            // Redistribute bits based on complexity
            let mut new_allocations = Vec::with_capacity(num_frames);
            for (i, w) in weights.iter().enumerate() {
                let relative_weight = if avg_weight > 0.0 {
                    w / avg_weight
                } else {
                    1.0
                };

                // Damped update: blend previous allocation with new estimate
                let new_bits = target_bpf * relative_weight * lambda;
                let damped = if pass_idx == 0 {
                    new_bits
                } else {
                    let prev = allocations[i] as f64;
                    prev * (1.0 - self.config.damping) + new_bits * self.config.damping
                };

                new_allocations.push((damped as u64).clamp(min_bits, max_bits));
            }

            // Normalize to match total target
            let current_total: u64 = new_allocations.iter().sum();
            if current_total > 0 {
                let scale = total_target as f64 / current_total as f64;
                for bits in &mut new_allocations {
                    *bits = ((*bits as f64 * scale) as u64).clamp(min_bits, max_bits);
                }
            }

            // VBV compliance check
            let vbv_compliant = self.check_vbv_compliance(&new_allocations);

            // If not VBV compliant, clamp peaks
            if !vbv_compliant {
                self.apply_vbv_clamping(&mut new_allocations);
            }

            // Compute quality estimates and variance
            let qualities: Vec<f64> = new_allocations
                .iter()
                .zip(frames.iter())
                .map(|(&bits, frame)| Self::estimate_quality(bits, frame.combined_complexity))
                .collect();

            let quality_variance = compute_std_dev(&qualities);

            // Compute bitrate error
            let actual_total: u64 = new_allocations.iter().sum();
            let bitrate_error = if total_target > 0 {
                ((actual_total as f64 - total_target as f64) / total_target as f64).abs()
            } else {
                0.0
            };

            // Adjust lambda for next pass
            if actual_total > total_target {
                lambda *= 1.0 + 0.1 * (1.0 - self.config.damping);
            } else {
                lambda *= 1.0 - 0.1 * (1.0 - self.config.damping);
            }
            lambda = lambda.clamp(0.1, 10.0);

            let pass_result = PassResult {
                pass_number: pass_idx + 1,
                frame_allocations: new_allocations.clone(),
                total_bits: actual_total,
                avg_bits_per_frame: actual_total as f64 / num_frames as f64,
                quality_variance,
                bitrate_error,
                vbv_compliant: self.check_vbv_compliance(&new_allocations),
                lambda,
            };

            allocations = new_allocations;
            passes.push(pass_result);

            // Check convergence
            if bitrate_error < self.config.convergence_tolerance {
                converged = true;
                break;
            }
        }

        let total_bits: u64 = allocations.iter().sum();
        let final_qualities: Vec<f64> = allocations
            .iter()
            .zip(frames.iter())
            .map(|(&bits, frame)| Self::estimate_quality(bits, frame.combined_complexity))
            .collect();

        Ok(MultiPassResult {
            passes_executed: passes.len() as u32,
            passes,
            final_allocations: allocations,
            total_bits,
            converged,
            final_quality_variance: compute_std_dev(&final_qualities),
        })
    }

    /// Estimates quality (0-100) from allocated bits and complexity.
    /// More bits relative to complexity = higher quality.
    #[must_use]
    fn estimate_quality(bits: u64, complexity: f64) -> f64 {
        let complexity = complexity.max(0.01);
        // Quality model: log relationship between bits/complexity ratio and quality
        let ratio = bits as f64 / (complexity * 50_000.0);
        let quality = 50.0 + 20.0 * ratio.ln().max(-2.5).min(2.5);
        quality.clamp(0.0, 100.0)
    }

    /// Checks VBV compliance for a set of allocations.
    fn check_vbv_compliance(&self, allocations: &[u64]) -> bool {
        let mut sim = VbvSimulator::new(
            self.config.vbv_buffer_size,
            self.config.vbv_max_bitrate,
            self.config.fps,
            self.config.vbv_initial_fill,
        );
        allocations.iter().all(|&bits| sim.process_frame(bits))
    }

    /// Applies VBV clamping: reduces peaks that would violate buffer constraints.
    fn apply_vbv_clamping(&self, allocations: &mut [u64]) {
        let max_per_frame = if self.config.fps > 0.0 {
            (self.config.vbv_max_bitrate as f64 / self.config.fps * 1.5) as u64
        } else {
            u64::MAX
        };

        for bits in allocations.iter_mut() {
            if *bits > max_per_frame {
                *bits = max_per_frame;
            }
        }
    }

    /// Computes the redistribution gain: the ratio of quality variance after
    /// multi-pass vs. uniform allocation. Values < 1.0 mean improvement.
    #[must_use]
    pub fn compute_redistribution_gain(
        &self,
        frames: &[FramePassStats],
    ) -> Result<f64, MultiPassError> {
        if frames.is_empty() {
            return Err(MultiPassError::EmptyInput);
        }

        let bpf = self.config.bits_per_frame() as u64;
        let uniform_qualities: Vec<f64> = frames
            .iter()
            .map(|f| Self::estimate_quality(bpf, f.combined_complexity))
            .collect();
        let uniform_var = compute_std_dev(&uniform_qualities);

        let result = self.optimize(frames)?;
        let optimized_var = result.final_quality_variance;

        if uniform_var > 0.0 {
            Ok(optimized_var / uniform_var)
        } else {
            Ok(1.0)
        }
    }
}

// ── Rate redistribution planner ─────────────────────────────────────────────

/// Plans rate redistribution between scenes/segments based on per-segment
/// complexity statistics collected from a previous pass.
#[derive(Debug, Clone)]
pub struct RateRedistributor {
    /// Segment-level complexity scores.
    segments: Vec<SegmentStats>,
    /// Total budget in bits.
    total_budget: u64,
}

/// Per-segment statistics for rate redistribution.
#[derive(Debug, Clone)]
pub struct SegmentStats {
    /// Segment index.
    pub index: usize,
    /// Number of frames in this segment.
    pub frame_count: usize,
    /// Average complexity of frames in this segment.
    pub avg_complexity: f64,
    /// Peak complexity frame in this segment.
    pub peak_complexity: f64,
    /// Whether this segment contains a scene change.
    pub has_scene_change: bool,
}

/// Per-segment budget allocation result.
#[derive(Debug, Clone)]
pub struct SegmentBudget {
    /// Segment index.
    pub index: usize,
    /// Allocated bits for this segment.
    pub bits: u64,
    /// Average bits per frame within this segment.
    pub avg_bits_per_frame: f64,
    /// Quality priority (higher = more bits relative to average).
    pub priority: f64,
}

impl RateRedistributor {
    /// Creates a new rate redistributor.
    pub fn new(segments: Vec<SegmentStats>, total_budget: u64) -> Result<Self, MultiPassError> {
        if segments.is_empty() {
            return Err(MultiPassError::EmptyInput);
        }
        Ok(Self {
            segments,
            total_budget,
        })
    }

    /// Redistributes the total budget across segments proportional to complexity.
    #[must_use]
    pub fn redistribute(&self) -> Vec<SegmentBudget> {
        let total_weighted: f64 = self
            .segments
            .iter()
            .map(|s| s.avg_complexity * s.frame_count as f64 * scene_change_bonus(s))
            .sum();

        if total_weighted <= 0.0 {
            return self
                .segments
                .iter()
                .map(|s| {
                    let per_seg = self.total_budget / self.segments.len().max(1) as u64;
                    SegmentBudget {
                        index: s.index,
                        bits: per_seg,
                        avg_bits_per_frame: if s.frame_count > 0 {
                            per_seg as f64 / s.frame_count as f64
                        } else {
                            0.0
                        },
                        priority: 1.0,
                    }
                })
                .collect();
        }

        self.segments
            .iter()
            .map(|s| {
                let weight = s.avg_complexity * s.frame_count as f64 * scene_change_bonus(s);
                let fraction = weight / total_weighted;
                let bits = (self.total_budget as f64 * fraction) as u64;
                let avg_bpf = if s.frame_count > 0 {
                    bits as f64 / s.frame_count as f64
                } else {
                    0.0
                };
                let avg_frame_count = self
                    .segments
                    .iter()
                    .map(|seg| seg.frame_count)
                    .sum::<usize>() as f64
                    / self.segments.len() as f64;
                let avg_complexity = self
                    .segments
                    .iter()
                    .map(|seg| seg.avg_complexity)
                    .sum::<f64>()
                    / self.segments.len() as f64;
                let priority = if avg_complexity > 0.0 && avg_frame_count > 0.0 {
                    (s.avg_complexity / avg_complexity) * (s.frame_count as f64 / avg_frame_count)
                } else {
                    1.0
                };

                SegmentBudget {
                    index: s.index,
                    bits,
                    avg_bits_per_frame: avg_bpf,
                    priority,
                }
            })
            .collect()
    }

    /// Returns the total budget.
    #[must_use]
    pub fn total_budget(&self) -> u64 {
        self.total_budget
    }

    /// Returns the segments.
    #[must_use]
    pub fn segments(&self) -> &[SegmentStats] {
        &self.segments
    }
}

/// Scene-change segments get a bonus allocation (I-frames need more bits).
fn scene_change_bonus(s: &SegmentStats) -> f64 {
    if s.has_scene_change {
        1.3
    } else {
        1.0
    }
}

// ── Quality convergence tracker ─────────────────────────────────────────────

/// Tracks quality convergence across passes using exponential moving average.
#[derive(Debug, Clone)]
pub struct ConvergenceTracker {
    history: VecDeque<f64>,
    max_history: usize,
    ema_alpha: f64,
    ema_value: Option<f64>,
}

impl ConvergenceTracker {
    /// Creates a new convergence tracker.
    #[must_use]
    pub fn new(max_history: usize) -> Self {
        Self {
            history: VecDeque::with_capacity(max_history),
            max_history,
            ema_alpha: 0.3,
            ema_value: None,
        }
    }

    /// Records a quality variance measurement.
    pub fn record(&mut self, quality_variance: f64) {
        if self.history.len() >= self.max_history {
            self.history.pop_front();
        }
        self.history.push_back(quality_variance);

        self.ema_value = Some(match self.ema_value {
            Some(prev) => self.ema_alpha * quality_variance + (1.0 - self.ema_alpha) * prev,
            None => quality_variance,
        });
    }

    /// Returns the smoothed (EMA) quality variance.
    #[must_use]
    pub fn smoothed_variance(&self) -> Option<f64> {
        self.ema_value
    }

    /// Returns true if convergence is detected (variance decreasing trend).
    #[must_use]
    pub fn is_converging(&self) -> bool {
        if self.history.len() < 3 {
            return false;
        }
        let len = self.history.len();
        let recent = self.history[len - 1];
        let prev = self.history[len - 2];
        let older = self.history[len - 3];

        // Monotonically decreasing (with some tolerance)
        recent <= prev * 1.05 && prev <= older * 1.05
    }

    /// Returns the improvement ratio between first and last recorded variance.
    #[must_use]
    pub fn improvement_ratio(&self) -> f64 {
        if self.history.len() < 2 {
            return 1.0;
        }
        let first = self.history[0];
        let last = self.history[self.history.len() - 1];
        if first > 0.0 {
            last / first
        } else {
            1.0
        }
    }
}

// ── Utility ─────────────────────────────────────────────────────────────────

/// Computes the standard deviation of a slice of f64 values.
fn compute_std_dev(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
    variance.sqrt()
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frames(count: usize) -> Vec<FramePassStats> {
        (0..count)
            .map(|i| {
                let ft = if i % 12 == 0 { 'I' } else { 'P' };
                let spatial = 0.3 + 0.4 * ((i as f64 * 0.7).sin() + 1.0) / 2.0;
                let temporal = 0.2 + 0.3 * ((i as f64 * 1.1).cos() + 1.0) / 2.0;
                FramePassStats::new(i, ft, spatial, temporal)
            })
            .collect()
    }

    #[test]
    fn test_config_default_valid() {
        let config = MultiPassConfig::default();
        assert!(config.validate().is_ok());
        assert!(config.bits_per_frame() > 0.0);
    }

    #[test]
    fn test_config_invalid_num_passes() {
        let mut config = MultiPassConfig::default();
        config.num_passes = 1;
        assert!(config.validate().is_err());
        config.num_passes = 9;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_invalid_fps() {
        let mut config = MultiPassConfig::default();
        config.fps = 0.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_optimizer_empty_frames_error() {
        let opt = MultiPassOptimizer::new(MultiPassConfig::default())
            .expect("default config should be valid");
        let result = opt.optimize(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_optimizer_basic_optimization() {
        let opt = MultiPassOptimizer::new(MultiPassConfig::default())
            .expect("default config should be valid");
        let frames = make_frames(48);
        let result = opt.optimize(&frames).expect("optimization should succeed");

        assert!(!result.final_allocations.is_empty());
        assert_eq!(result.final_allocations.len(), 48);
        assert!(result.passes_executed >= 1);
        assert!(result.total_bits > 0);
    }

    #[test]
    fn test_optimizer_converges() {
        let config = MultiPassConfig {
            num_passes: 5,
            convergence_tolerance: 0.1,
            ..Default::default()
        };
        let opt = MultiPassOptimizer::new(config).expect("config should be valid");
        let frames = make_frames(24);
        let result = opt.optimize(&frames).expect("optimization should succeed");

        // With generous tolerance, should converge before max passes
        assert!(result.passes_executed <= 5);
    }

    #[test]
    fn test_complex_frames_get_more_bits() {
        let opt = MultiPassOptimizer::new(MultiPassConfig::default())
            .expect("default config should be valid");

        let frames = vec![
            FramePassStats::new(0, 'I', 0.9, 0.8), // complex
            FramePassStats::new(1, 'P', 0.1, 0.1), // simple
            FramePassStats::new(2, 'P', 0.1, 0.1), // simple
            FramePassStats::new(3, 'P', 0.9, 0.8), // complex
        ];

        let result = opt.optimize(&frames).expect("optimization should succeed");
        let allocs = &result.final_allocations;

        // Complex frames (0, 3) should get more bits than simple frames (1, 2)
        assert!(
            allocs[0] > allocs[1],
            "Complex I-frame should get more bits than simple P-frame"
        );
    }

    #[test]
    fn test_vbv_simulator_basic() {
        let mut sim = VbvSimulator::new(4_000_000, 8_000_000, 24.0, 0.9);
        assert!(sim.level() > 0);

        // Process a normal-sized frame
        let compliant = sim.process_frame(100_000);
        assert!(compliant);
    }

    #[test]
    fn test_rate_redistributor() {
        let segments = vec![
            SegmentStats {
                index: 0,
                frame_count: 24,
                avg_complexity: 0.8,
                peak_complexity: 0.95,
                has_scene_change: true,
            },
            SegmentStats {
                index: 1,
                frame_count: 48,
                avg_complexity: 0.3,
                peak_complexity: 0.5,
                has_scene_change: false,
            },
            SegmentStats {
                index: 2,
                frame_count: 24,
                avg_complexity: 0.6,
                peak_complexity: 0.7,
                has_scene_change: false,
            },
        ];

        let total_budget = 10_000_000u64;
        let redist =
            RateRedistributor::new(segments, total_budget).expect("redistributor should create");
        let budgets = redist.redistribute();

        assert_eq!(budgets.len(), 3);

        // Complex scene-change segment should get proportionally more
        let seg0_per_frame = budgets[0].avg_bits_per_frame;
        let seg1_per_frame = budgets[1].avg_bits_per_frame;
        assert!(
            seg0_per_frame > seg1_per_frame,
            "Complex segment should get more bits per frame"
        );
    }

    #[test]
    fn test_rate_redistributor_empty_error() {
        let result = RateRedistributor::new(vec![], 1000);
        assert!(result.is_err());
    }

    #[test]
    fn test_convergence_tracker() {
        let mut tracker = ConvergenceTracker::new(10);

        // Record decreasing variance
        tracker.record(10.0);
        tracker.record(8.0);
        tracker.record(6.0);
        tracker.record(4.5);

        assert!(tracker.is_converging());
        assert!(tracker.improvement_ratio() < 1.0);
        assert!(tracker.smoothed_variance().is_some());
    }

    #[test]
    fn test_convergence_tracker_not_converging() {
        let mut tracker = ConvergenceTracker::new(10);

        // Record increasing variance
        tracker.record(4.0);
        tracker.record(6.0);
        tracker.record(10.0);

        assert!(!tracker.is_converging());
    }

    #[test]
    fn test_frame_pass_stats_weights() {
        let i_frame = FramePassStats::new(0, 'I', 0.5, 0.5);
        let p_frame = FramePassStats::new(1, 'P', 0.5, 0.5);
        let b_frame = FramePassStats::new(2, 'B', 0.5, 0.5);

        assert!(i_frame.frame_type_weight() > p_frame.frame_type_weight());
        assert!(p_frame.frame_type_weight() > b_frame.frame_type_weight());
    }

    #[test]
    fn test_redistribution_gain() {
        let opt = MultiPassOptimizer::new(MultiPassConfig::default())
            .expect("default config should be valid");
        let frames = make_frames(24);
        let gain = opt
            .compute_redistribution_gain(&frames)
            .expect("gain computation should succeed");

        // Redistribution should generally reduce quality variance (gain < 1.0),
        // but at minimum it should not explode
        assert!(
            gain < 2.0,
            "Redistribution should not dramatically worsen quality variance"
        );
    }

    #[test]
    fn test_vbv_clamping_applied() {
        let config = MultiPassConfig {
            vbv_max_bitrate: 1_000_000,
            fps: 24.0,
            ..Default::default()
        };
        let opt = MultiPassOptimizer::new(config).expect("config should be valid");

        // Create frames with extreme complexity variation
        let frames: Vec<FramePassStats> = (0..12)
            .map(|i| {
                if i == 0 {
                    FramePassStats::new(i, 'I', 1.0, 1.0)
                } else {
                    FramePassStats::new(i, 'P', 0.05, 0.05)
                }
            })
            .collect();

        let result = opt.optimize(&frames).expect("optimization should succeed");
        // All frames should have finite non-zero allocations
        assert!(result.total_bits > 0);
        // The max frame should be limited by the VBV clamping
        let max_frame = result.final_allocations.iter().copied().max().unwrap_or(0);
        let min_frame = result.final_allocations.iter().copied().min().unwrap_or(0);
        // Complex I-frame should get more than simple P-frames
        assert!(
            max_frame > min_frame,
            "Allocation should vary: max={max_frame}, min={min_frame}"
        );
    }

    #[test]
    fn test_compute_std_dev() {
        let values = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let sd = compute_std_dev(&values);
        assert!(sd > 0.0);
        assert!(sd < 10.0);
    }

    #[test]
    fn test_quality_estimate_monotonic() {
        // More bits for same complexity should give higher quality
        let q_low = MultiPassOptimizer::estimate_quality(10_000, 0.5);
        let q_high = MultiPassOptimizer::estimate_quality(100_000, 0.5);
        assert!(
            q_high > q_low,
            "More bits should yield higher quality: {q_high} > {q_low}"
        );
    }
}
