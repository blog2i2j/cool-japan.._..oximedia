//! Average Bitrate (ABR) rate control with multi-pass encoding.
//!
//! ABR mode targets an average bitrate over the entire encode, allowing
//! more flexibility than CBR while ensuring predictable file sizes.
//!
//! # Features
//!
//! - Multi-pass encoding (1-pass and 2-pass modes)
//! - Sophisticated lookahead analysis for optimal bit distribution
//! - Rate control models (linear, quadratic, power)
//! - Curve compression for bitrate management
//! - Adaptive reference frame selection
//! - Target file size achievement
//!
//! # Two-Pass Encoding Workflow
//!
//! ```text
//! Pass 1: Analysis
//!   ┌─────────────┐
//!   │ Read Frames │
//!   └──────┬──────┘
//!          │
//!   ┌──────▼──────────┐
//!   │ Complexity      │
//!   │ Analysis        │
//!   └──────┬──────────┘
//!          │
//!   ┌──────▼──────────┐
//!   │ Statistics      │
//!   │ Collection      │
//!   └──────┬──────────┘
//!          │
//!   ┌──────▼──────────┐
//!   │ Write Stats     │
//!   │ File            │
//!   └─────────────────┘
//!
//! Pass 2: Encoding
//!   ┌─────────────┐
//!   │ Read Stats  │
//!   │ File        │
//!   └──────┬──────┘
//!          │
//!   ┌──────▼──────────┐
//!   │ Bit Allocation  │
//!   │ Planning        │
//!   └──────┬──────────┘
//!          │
//!   ┌──────▼──────────┐
//!   │ Optimal         │
//!   │ Encoding        │
//!   └─────────────────┘
//! ```

#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::struct_excessive_bools)]
#![allow(dead_code)]
#![forbid(unsafe_code)]

use crate::frame::FrameType;

use super::types::{FrameStats, GopStats, RcConfig, RcOutput};

/// ABR (Average Bitrate) rate controller.
#[derive(Clone, Debug)]
pub struct AbrController {
    /// Target bitrate in bits per second.
    target_bitrate: u64,
    /// Target file size in bytes (if specified).
    target_file_size: Option<u64>,
    /// Total frames in the sequence (if known).
    total_frames: Option<u64>,
    /// Current encoding pass (1 or 2).
    pass: u8,
    /// Frame rate.
    framerate: f64,
    /// Current QP.
    current_qp: f32,
    /// Min/max QP bounds.
    min_qp: u8,
    max_qp: u8,
    /// Frame type QP offsets.
    i_qp_offset: i8,
    b_qp_offset: i8,
    /// Frame counter.
    frame_count: u64,
    /// Total bits encoded so far.
    total_bits: u64,
    /// First pass statistics.
    first_pass_stats: Option<FirstPassStats>,
    /// Multi-pass allocator.
    allocator: MultiPassAllocator,
    /// Lookahead analyzer.
    lookahead: LookaheadAnalyzer,
    /// Rate control model.
    rc_model: RateControlModel,
    /// Curve compression state.
    curve_state: CurveCompressionState,
    /// GOP tracking.
    current_gop: GopStats,
    gop_history: Vec<GopStats>,
}

/// First pass statistics collection.
#[derive(Clone, Debug, Default)]
pub struct FirstPassStats {
    /// Per-frame statistics.
    pub frames: Vec<FramePassStats>,
    /// Total complexity.
    pub total_complexity: f64,
    /// Total frames analyzed.
    pub frame_count: u64,
    /// Per-GOP complexity.
    pub gop_complexities: Vec<f64>,
    /// Scene change indices.
    pub scene_changes: Vec<u64>,
    /// Average frame complexity.
    pub avg_complexity: f64,
    /// Complexity variance.
    pub complexity_variance: f64,
}

/// Per-frame statistics from first pass.
#[derive(Clone, Debug, Default)]
pub struct FramePassStats {
    /// Frame index.
    pub frame_index: u64,
    /// Frame type.
    pub frame_type: FrameType,
    /// Spatial complexity.
    pub spatial_complexity: f32,
    /// Temporal complexity.
    pub temporal_complexity: f32,
    /// Combined complexity.
    pub combined_complexity: f32,
    /// Is scene change.
    pub is_scene_change: bool,
    /// Number of intra-coded blocks.
    pub intra_blocks: u32,
    /// Number of inter-coded blocks.
    pub inter_blocks: u32,
    /// Average motion vector magnitude.
    pub avg_motion: f32,
    /// Predicted bits needed.
    pub predicted_bits: u64,
    /// Allocated bits (filled in second pass).
    pub allocated_bits: u64,
}

impl FirstPassStats {
    /// Add frame statistics.
    pub fn add_frame(&mut self, stats: FramePassStats) {
        self.total_complexity += stats.combined_complexity as f64;
        self.frame_count += 1;
        if stats.is_scene_change {
            self.scene_changes.push(stats.frame_index);
        }
        self.frames.push(stats);
    }

    /// Finalize statistics after first pass.
    pub fn finalize(&mut self) {
        if self.frame_count == 0 {
            return;
        }

        self.avg_complexity = self.total_complexity / self.frame_count as f64;

        // Calculate variance
        let mut variance_sum = 0.0;
        for frame in &self.frames {
            let diff = frame.combined_complexity as f64 - self.avg_complexity;
            variance_sum += diff * diff;
        }
        self.complexity_variance = variance_sum / self.frame_count as f64;
    }

    /// Get frame statistics by index.
    pub fn get_frame(&self, index: u64) -> Option<&FramePassStats> {
        self.frames.get(index as usize)
    }
}

/// Multi-pass bit allocator.
#[derive(Clone, Debug)]
struct MultiPassAllocator {
    /// Total bits available for entire encode.
    total_bits_budget: u64,
    /// Bits used so far.
    bits_used: u64,
    /// Frames remaining.
    frames_remaining: u64,
    /// Complexity-based allocation enabled.
    complexity_based: bool,
    /// Frame-type allocation ratios.
    i_frame_ratio: f64,
    p_frame_ratio: f64,
    b_frame_ratio: f64,
    /// Adaptive allocation strength.
    adaptation_strength: f32,
}

impl Default for MultiPassAllocator {
    fn default() -> Self {
        Self {
            total_bits_budget: 0,
            bits_used: 0,
            frames_remaining: 0,
            complexity_based: true,
            i_frame_ratio: 3.0,
            p_frame_ratio: 1.0,
            b_frame_ratio: 0.5,
            adaptation_strength: 1.0,
        }
    }
}

impl MultiPassAllocator {
    /// Initialize allocator with total budget.
    fn initialize(&mut self, total_bits: u64, total_frames: u64) {
        self.total_bits_budget = total_bits;
        self.frames_remaining = total_frames;
        self.bits_used = 0;
    }

    /// Allocate bits for a frame based on complexity and frame type.
    fn allocate_frame_bits(
        &mut self,
        frame_type: FrameType,
        complexity: f32,
        avg_complexity: f32,
        first_pass: &FirstPassStats,
    ) -> u64 {
        if self.frames_remaining == 0 {
            return 0;
        }

        let bits_remaining = self.total_bits_budget.saturating_sub(self.bits_used);
        let base_allocation = bits_remaining / self.frames_remaining;

        // Frame type multiplier
        let type_mult = match frame_type {
            FrameType::Key => self.i_frame_ratio,
            FrameType::Inter => self.p_frame_ratio,
            FrameType::BiDir => self.b_frame_ratio,
            FrameType::Switch => 1.5,
        };

        // Complexity-based adjustment
        let complexity_mult = if self.complexity_based && avg_complexity > 0.0 {
            (complexity / avg_complexity).clamp(0.5, 2.0)
        } else {
            1.0
        };

        // Calculate allocation with lookahead consideration
        let lookahead_mult = self.calculate_lookahead_multiplier(first_pass);

        let allocated =
            (base_allocation as f64 * type_mult * complexity_mult as f64 * lookahead_mult)
                .max(base_allocation as f64 * 0.25) as u64;

        self.bits_used += allocated;
        self.frames_remaining = self.frames_remaining.saturating_sub(1);

        allocated
    }

    /// Calculate multiplier based on future complexity.
    fn calculate_lookahead_multiplier(&self, first_pass: &FirstPassStats) -> f64 {
        // Simplified lookahead - check remaining complexity distribution
        let skip_count =
            (first_pass.frame_count as usize).saturating_sub(self.frames_remaining as usize);
        let remaining_complexity: f64 = first_pass
            .frames
            .iter()
            .skip(skip_count)
            .map(|f| f.combined_complexity as f64)
            .sum();

        if remaining_complexity > 0.0 && self.frames_remaining > 0 {
            let avg_remaining = remaining_complexity / self.frames_remaining as f64;
            let ratio = avg_remaining / first_pass.avg_complexity;

            // Adjust based on future complexity
            if ratio > 1.2 {
                0.95 // Save bits for complex future
            } else if ratio < 0.8 {
                1.05 // Spend more now
            } else {
                1.0
            }
        } else {
            1.0
        }
    }

    /// Update after frame encoding.
    fn update(&mut self, actual_bits: u64) {
        // In practice, might adjust budget based on actual vs allocated
        let _ = actual_bits;
    }
}

/// Lookahead analyzer for optimal encoding decisions.
#[derive(Clone, Debug)]
struct LookaheadAnalyzer {
    /// Lookahead depth in frames.
    depth: usize,
    /// Buffered frame complexities.
    complexity_buffer: Vec<f32>,
    /// Enable motion estimation lookahead.
    enable_me_lookahead: bool,
    /// Enable scene cut lookahead.
    enable_scene_lookahead: bool,
    /// Scene cut threshold.
    scene_threshold: f32,
}

impl Default for LookaheadAnalyzer {
    fn default() -> Self {
        Self {
            depth: 40,
            complexity_buffer: Vec::new(),
            enable_me_lookahead: true,
            enable_scene_lookahead: true,
            scene_threshold: 0.4,
        }
    }
}

impl LookaheadAnalyzer {
    /// Push frame complexity to lookahead buffer.
    fn push_complexity(&mut self, complexity: f32) {
        self.complexity_buffer.push(complexity);
        if self.complexity_buffer.len() > self.depth {
            self.complexity_buffer.remove(0);
        }
    }

    /// Analyze lookahead window for bit allocation hints.
    fn analyze(&self) -> LookaheadAnalysis {
        if self.complexity_buffer.is_empty() {
            return LookaheadAnalysis::default();
        }

        let avg_complexity: f32 =
            self.complexity_buffer.iter().sum::<f32>() / self.complexity_buffer.len() as f32;

        let current_complexity = self.complexity_buffer[0];

        // Detect scene cuts
        let has_scene_cut = if self.enable_scene_lookahead && self.complexity_buffer.len() > 1 {
            self.detect_scene_cut()
        } else {
            false
        };

        // Calculate complexity trend
        let trend = if self.complexity_buffer.len() >= 10 {
            let first_half: f32 = self.complexity_buffer[..5].iter().sum::<f32>() / 5.0;
            let second_half: f32 = self.complexity_buffer[5..10].iter().sum::<f32>() / 5.0;
            (second_half - first_half) / first_half
        } else {
            0.0
        };

        LookaheadAnalysis {
            avg_complexity,
            current_complexity,
            has_scene_cut,
            complexity_trend: trend,
            recommend_bits_multiplier: self.calculate_multiplier(
                current_complexity,
                avg_complexity,
                trend,
            ),
        }
    }

    /// Detect scene cut in lookahead window.
    fn detect_scene_cut(&self) -> bool {
        if self.complexity_buffer.len() < 2 {
            return false;
        }

        let window = self.complexity_buffer.len().min(self.depth);
        for i in 1..window {
            let prev = self.complexity_buffer[i - 1];
            let curr = self.complexity_buffer[i];
            if prev > 0.0 && (curr / prev) > (1.0 + self.scene_threshold) {
                return true;
            }
        }
        false
    }

    /// Calculate bit allocation multiplier based on lookahead.
    fn calculate_multiplier(&self, current: f32, avg: f32, trend: f32) -> f64 {
        let base_mult = if avg > 0.0 {
            (current / avg).clamp(0.7, 1.5)
        } else {
            1.0
        };

        // Adjust for trend
        let trend_adj = if trend > 0.2 {
            0.95 // Complexity increasing, save bits
        } else if trend < -0.2 {
            1.05 // Complexity decreasing, spend more
        } else {
            1.0
        };

        base_mult as f64 * trend_adj
    }
}

/// Lookahead analysis result.
#[derive(Clone, Debug, Default)]
struct LookaheadAnalysis {
    /// Average complexity in lookahead window.
    avg_complexity: f32,
    /// Current frame complexity.
    current_complexity: f32,
    /// Scene cut detected ahead.
    has_scene_cut: bool,
    /// Complexity trend (-1.0 to 1.0).
    complexity_trend: f32,
    /// Recommended bits multiplier.
    recommend_bits_multiplier: f64,
}

/// Rate control model for bit prediction.
#[derive(Clone, Debug)]
struct RateControlModel {
    /// Model type (0=linear, 1=quadratic, 2=power).
    model_type: u8,
    /// Linear model parameters.
    linear_params: LinearModel,
    /// Quadratic model parameters.
    quad_params: QuadraticModel,
    /// Power model parameters.
    power_params: PowerModel,
    /// Model selection based on error.
    auto_select: bool,
}

impl Default for RateControlModel {
    fn default() -> Self {
        Self {
            model_type: 0,
            linear_params: LinearModel::default(),
            quad_params: QuadraticModel::default(),
            power_params: PowerModel::default(),
            auto_select: true,
        }
    }
}

impl RateControlModel {
    /// Predict bits for given complexity and QP.
    fn predict(&self, complexity: f32, qp: f32) -> u64 {
        let prediction = match self.model_type {
            0 => self.linear_params.predict(complexity, qp),
            1 => self.quad_params.predict(complexity, qp),
            2 => self.power_params.predict(complexity, qp),
            _ => self.linear_params.predict(complexity, qp),
        };

        prediction.max(100.0) as u64
    }

    /// Fit models using first pass data.
    fn fit(&mut self, first_pass: &FirstPassStats) {
        if first_pass.frames.len() < 10 {
            return;
        }

        self.linear_params.fit(first_pass);
        self.quad_params.fit(first_pass);
        self.power_params.fit(first_pass);

        if self.auto_select {
            self.select_best_model(first_pass);
        }
    }

    /// Select best model based on prediction error.
    fn select_best_model(&mut self, first_pass: &FirstPassStats) {
        let mut errors = [0.0_f64; 3];

        for frame in &first_pass.frames {
            if frame.predicted_bits == 0 {
                continue;
            }

            let actual = frame.predicted_bits as f64;
            let qp = 28.0; // Assume default QP for model selection

            errors[0] += (actual - self.linear_params.predict(frame.combined_complexity, qp)).abs();
            errors[1] += (actual - self.quad_params.predict(frame.combined_complexity, qp)).abs();
            errors[2] += (actual - self.power_params.predict(frame.combined_complexity, qp)).abs();
        }

        // Select model with minimum error
        self.model_type = if errors[0] <= errors[1] && errors[0] <= errors[2] {
            0
        } else if errors[1] <= errors[2] {
            1
        } else {
            2
        };
    }
}

/// Linear rate control model: bits = a * complexity + b * qp + c
#[derive(Clone, Debug)]
struct LinearModel {
    a: f64,
    b: f64,
    c: f64,
}

impl Default for LinearModel {
    fn default() -> Self {
        Self {
            a: 100_000.0,
            b: -5_000.0,
            c: 50_000.0,
        }
    }
}

impl LinearModel {
    fn predict(&self, complexity: f32, qp: f32) -> f64 {
        self.a * complexity as f64 + self.b * qp as f64 + self.c
    }

    fn fit(&mut self, first_pass: &FirstPassStats) {
        // Simplified least squares fitting
        let n = first_pass.frames.len() as f64;
        if n < 3.0 {
            return;
        }

        let mut sum_c = 0.0;
        let mut sum_bits = 0.0;

        for frame in &first_pass.frames {
            if frame.predicted_bits > 0 {
                sum_c += frame.combined_complexity as f64;
                sum_bits += frame.predicted_bits as f64;
            }
        }

        let avg_c = sum_c / n;
        let avg_bits = sum_bits / n;

        // Simple linear fit
        if avg_c > 0.0 {
            self.a = avg_bits / avg_c;
            self.c = avg_bits * 0.2;
        }
    }
}

/// Quadratic model: bits = a * complexity^2 + b * complexity + c * qp + d
#[derive(Clone, Debug)]
struct QuadraticModel {
    a: f64,
    b: f64,
    c: f64,
    d: f64,
}

impl Default for QuadraticModel {
    fn default() -> Self {
        Self {
            a: 10_000.0,
            b: 80_000.0,
            c: -5_000.0,
            d: 50_000.0,
        }
    }
}

impl QuadraticModel {
    fn predict(&self, complexity: f32, qp: f32) -> f64 {
        let c = complexity as f64;
        self.a * c * c + self.b * c + self.c * qp as f64 + self.d
    }

    fn fit(&mut self, first_pass: &FirstPassStats) {
        // Simplified quadratic fitting
        let _ = first_pass;
        // In production, would use proper polynomial regression
    }
}

/// Power model: bits = a * complexity^b / qp^c
#[derive(Clone, Debug)]
struct PowerModel {
    a: f64,
    b: f64,
    c: f64,
}

impl Default for PowerModel {
    fn default() -> Self {
        Self {
            a: 150_000.0,
            b: 1.1,
            c: 0.8,
        }
    }
}

impl PowerModel {
    fn predict(&self, complexity: f32, qp: f32) -> f64 {
        if complexity <= 0.0 || qp <= 0.0 {
            return 50_000.0;
        }
        self.a * (complexity as f64).powf(self.b) / (qp as f64).powf(self.c)
    }

    fn fit(&mut self, first_pass: &FirstPassStats) {
        // Simplified power model fitting using log-space regression
        let _ = first_pass;
        // In production, would use proper log-linear regression
    }
}

/// Curve compression for rate smoothing.
#[derive(Clone, Debug)]
struct CurveCompressionState {
    /// Enable curve compression.
    enabled: bool,
    /// Compression ratio (0.0-1.0).
    ratio: f32,
    /// Complexity adjustment factor.
    adjustment_factor: f32,
}

impl Default for CurveCompressionState {
    fn default() -> Self {
        Self {
            enabled: true,
            ratio: 0.6,
            adjustment_factor: 1.0,
        }
    }
}

impl CurveCompressionState {
    /// Apply curve compression to complexity.
    fn compress(&self, complexity: f32, avg_complexity: f32) -> f32 {
        if !self.enabled || avg_complexity <= 0.0 {
            return complexity;
        }

        let ratio = complexity / avg_complexity;
        let compressed_ratio = if ratio > 1.0 {
            // Compress high complexity
            1.0 + (ratio - 1.0) * self.ratio
        } else {
            // Expand low complexity
            1.0 - (1.0 - ratio) * self.ratio
        };

        avg_complexity * compressed_ratio * self.adjustment_factor
    }

    /// Update adjustment factor based on encoding results.
    fn update(&mut self, target_bitrate: f64, actual_bitrate: f64) {
        if target_bitrate <= 0.0 {
            return;
        }

        let error = (actual_bitrate - target_bitrate) / target_bitrate;

        // Gradually adjust
        let adjustment = 1.0 + (error * 0.05) as f32;
        self.adjustment_factor = (self.adjustment_factor * adjustment).clamp(0.5, 2.0);
    }
}

impl AbrController {
    /// Create new ABR controller from configuration.
    #[must_use]
    pub fn new(config: &RcConfig) -> Self {
        Self {
            target_bitrate: config.target_bitrate,
            target_file_size: None,
            total_frames: None,
            pass: 1,
            framerate: config.framerate(),
            current_qp: config.initial_qp as f32,
            min_qp: config.min_qp,
            max_qp: config.max_qp,
            i_qp_offset: config.i_qp_offset,
            b_qp_offset: config.b_qp_offset,
            frame_count: 0,
            total_bits: 0,
            first_pass_stats: None,
            allocator: MultiPassAllocator::default(),
            lookahead: LookaheadAnalyzer::default(),
            rc_model: RateControlModel::default(),
            curve_state: CurveCompressionState::default(),
            current_gop: GopStats::new(0, 0),
            gop_history: Vec::new(),
        }
    }

    /// Set target file size in bytes.
    pub fn set_target_file_size(&mut self, size_bytes: u64, duration_seconds: f64) {
        self.target_file_size = Some(size_bytes);
        if duration_seconds > 0.0 {
            // Calculate target bitrate from file size
            self.target_bitrate = ((size_bytes * 8) as f64 / duration_seconds) as u64;
            self.total_frames = Some((duration_seconds * self.framerate) as u64);
        }
    }

    /// Set encoding pass (1 or 2).
    pub fn set_pass(&mut self, pass: u8) {
        self.pass = pass.clamp(1, 2);
        if pass == 1 {
            self.first_pass_stats = Some(FirstPassStats::default());
        }
    }

    /// Set total frames for the encode.
    pub fn set_total_frames(&mut self, count: u64) {
        self.total_frames = Some(count);
    }

    /// Import first pass statistics for second pass.
    pub fn set_first_pass_stats(&mut self, stats: FirstPassStats) {
        // Fit rate control models
        self.rc_model.fit(&stats);

        // Initialize allocator
        if let Some(total_frames) = self.total_frames {
            let duration = total_frames as f64 / self.framerate;
            let total_bits = (self.target_bitrate as f64 * duration) as u64;
            self.allocator.initialize(total_bits, total_frames);
        }

        self.first_pass_stats = Some(stats);
    }

    /// Add frame to lookahead (for real-time first pass).
    pub fn push_lookahead(&mut self, complexity: f32) {
        self.lookahead.push_complexity(complexity);
    }

    /// Get rate control output for current frame.
    #[must_use]
    pub fn get_rc(&mut self, frame_type: FrameType, complexity: f32) -> RcOutput {
        let target_bits = if self.pass == 2 && self.first_pass_stats.is_some() {
            // Second pass: use allocated bits from first pass
            self.calculate_second_pass_bits(frame_type, complexity)
        } else {
            // First pass: estimate bits
            self.calculate_first_pass_bits(frame_type, complexity)
        };

        // Calculate QP
        let qp = self.calculate_qp(frame_type, complexity, target_bits);

        // Calculate lambda
        let lambda = self.calculate_lambda(qp, frame_type);
        let lambda_me = lambda.sqrt();

        RcOutput {
            qp,
            qp_f: qp as f32,
            target_bits,
            min_bits: target_bits / 4,
            max_bits: target_bits * 4,
            lambda,
            lambda_me,
            ..Default::default()
        }
    }

    /// Calculate bits for first pass.
    fn calculate_first_pass_bits(&self, frame_type: FrameType, complexity: f32) -> u64 {
        let base_bits = if self.framerate > 0.0 {
            (self.target_bitrate as f64 / self.framerate) as u64
        } else {
            100_000
        };

        let type_mult = match frame_type {
            FrameType::Key => 3.0,
            FrameType::Inter => 1.0,
            FrameType::BiDir => 0.5,
            FrameType::Switch => 1.5,
        };

        let complexity_mult = if complexity > 0.5 {
            complexity.clamp(0.5, 2.0)
        } else {
            1.0
        };

        (base_bits as f64 * type_mult * complexity_mult as f64) as u64
    }

    /// Calculate bits for second pass using first pass statistics.
    fn calculate_second_pass_bits(&mut self, frame_type: FrameType, complexity: f32) -> u64 {
        if let Some(ref stats) = self.first_pass_stats {
            // Get frame statistics
            if let Some(frame_stats) = stats.get_frame(self.frame_count) {
                // Use pre-allocated bits
                if frame_stats.allocated_bits > 0 {
                    return frame_stats.allocated_bits;
                }
            }

            // Allocate bits based on complexity
            let avg_complexity = stats.avg_complexity as f32;
            let compressed_complexity = self.curve_state.compress(complexity, avg_complexity);

            self.allocator.allocate_frame_bits(
                frame_type,
                compressed_complexity,
                avg_complexity,
                stats,
            )
        } else {
            self.calculate_first_pass_bits(frame_type, complexity)
        }
    }

    /// Calculate QP based on target bits and complexity.
    fn calculate_qp(&self, frame_type: FrameType, complexity: f32, target_bits: u64) -> u8 {
        // Use rate model to estimate QP
        let base_qp = if target_bits > 0 {
            // Inverse model: estimate QP from bits and complexity
            let model_prediction = self.rc_model.predict(complexity, 28.0_f32) as f64;
            let ratio = target_bits as f64 / model_prediction;

            // Adjust QP based on ratio
            if ratio > 1.0 {
                // More bits available, decrease QP
                28.0_f64 - (ratio.ln() * 3.0)
            } else {
                // Fewer bits available, increase QP
                28.0_f64 + ((1.0_f64 / ratio).ln() * 3.0)
            }
        } else {
            28.0_f64
        };

        // Apply frame type offset
        let offset = match frame_type {
            FrameType::Key => self.i_qp_offset,
            FrameType::BiDir => self.b_qp_offset,
            _ => 0,
        };

        let final_qp = (base_qp + offset as f64).clamp(self.min_qp as f64, self.max_qp as f64);
        final_qp as u8
    }

    /// Calculate lambda for RDO.
    fn calculate_lambda(&self, qp: u8, frame_type: FrameType) -> f64 {
        let base = 0.85 * 2.0_f64.powf((f64::from(qp) - 12.0) / 3.0);

        let multiplier = match frame_type {
            FrameType::Key => 0.6,
            FrameType::BiDir => 1.4,
            _ => 1.0,
        };

        base * multiplier
    }

    /// Update controller with frame encoding results.
    pub fn update(&mut self, stats: &FrameStats) {
        self.frame_count += 1;
        self.total_bits += stats.bits;

        // Update allocator
        if self.pass == 2 {
            self.allocator.update(stats.bits);
        }

        // Update curve compression
        if self.frame_count > 10 {
            let elapsed = self.frame_count as f64 / self.framerate;
            let actual_bitrate = self.total_bits as f64 / elapsed;
            self.curve_state
                .update(self.target_bitrate as f64, actual_bitrate);
        }

        // Update GOP tracking
        self.current_gop.add_frame(stats.clone());
        if stats.frame_type == FrameType::Key && self.current_gop.frame_count > 1 {
            self.gop_history.push(self.current_gop.clone());
            let next_idx = self.current_gop.gop_index + 1;
            self.current_gop = GopStats::new(next_idx, self.frame_count);
        }

        // Collect first pass stats
        if self.pass == 1 {
            if let Some(ref mut pass_stats) = self.first_pass_stats {
                let frame_stats = FramePassStats {
                    frame_index: stats.frame_num,
                    frame_type: stats.frame_type,
                    spatial_complexity: stats.spatial_complexity,
                    temporal_complexity: stats.temporal_complexity,
                    combined_complexity: stats.complexity,
                    is_scene_change: stats.scene_cut,
                    intra_blocks: 0,
                    inter_blocks: 0,
                    avg_motion: 0.0,
                    predicted_bits: stats.bits,
                    allocated_bits: 0,
                };
                pass_stats.add_frame(frame_stats);
            }
        }
    }

    /// Finalize first pass and return statistics.
    #[must_use]
    pub fn finalize_first_pass(&mut self) -> Option<FirstPassStats> {
        if self.pass != 1 {
            return None;
        }

        if let Some(ref mut stats) = self.first_pass_stats {
            stats.finalize();

            // Calculate bit allocation for second pass
            if let Some(total_frames) = self.total_frames {
                let duration = total_frames as f64 / self.framerate;
                let total_bits = (self.target_bitrate as f64 * duration) as u64;

                for frame in &mut stats.frames {
                    // Simple allocation based on complexity
                    let complexity_ratio = if stats.total_complexity > 0.0 {
                        frame.combined_complexity as f64 / stats.avg_complexity
                    } else {
                        1.0
                    };

                    frame.allocated_bits =
                        (total_bits as f64 / total_frames as f64 * complexity_ratio) as u64;
                }
            }
        }

        self.first_pass_stats.take()
    }

    /// Get current average bitrate.
    #[must_use]
    pub fn current_bitrate(&self) -> f64 {
        if self.frame_count == 0 || self.framerate <= 0.0 {
            return 0.0;
        }
        let elapsed = self.frame_count as f64 / self.framerate;
        self.total_bits as f64 / elapsed
    }

    /// Get target bitrate.
    #[must_use]
    pub fn target_bitrate(&self) -> u64 {
        self.target_bitrate
    }

    /// Get frame count.
    #[must_use]
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Get total bits encoded.
    #[must_use]
    pub fn total_bits(&self) -> u64 {
        self.total_bits
    }

    /// Get current QP.
    #[must_use]
    pub fn current_qp(&self) -> f32 {
        self.current_qp
    }

    /// Calculate projected file size based on current progress.
    #[must_use]
    pub fn projected_file_size(&self) -> Option<u64> {
        if let Some(total_frames) = self.total_frames {
            if self.frame_count > 0 && self.frame_count < total_frames {
                let bits_per_frame = self.total_bits as f64 / self.frame_count as f64;
                let projected_bits = bits_per_frame * total_frames as f64;
                Some((projected_bits / 8.0) as u64)
            } else {
                Some(self.total_bits / 8)
            }
        } else {
            None
        }
    }

    /// Get bitrate deviation from target.
    #[must_use]
    pub fn bitrate_deviation(&self) -> f64 {
        let current = self.current_bitrate();
        if self.target_bitrate > 0 {
            (current - self.target_bitrate as f64) / self.target_bitrate as f64
        } else {
            0.0
        }
    }

    /// Reset controller state.
    pub fn reset(&mut self) {
        self.frame_count = 0;
        self.total_bits = 0;
        self.current_gop = GopStats::new(0, 0);
        self.gop_history.clear();
        self.lookahead.complexity_buffer.clear();
    }
}

impl Default for AbrController {
    fn default() -> Self {
        Self {
            target_bitrate: 5_000_000,
            target_file_size: None,
            total_frames: None,
            pass: 1,
            framerate: 30.0,
            current_qp: 28.0,
            min_qp: 1,
            max_qp: 63,
            i_qp_offset: -2,
            b_qp_offset: 2,
            frame_count: 0,
            total_bits: 0,
            first_pass_stats: None,
            allocator: MultiPassAllocator::default(),
            lookahead: LookaheadAnalyzer::default(),
            rc_model: RateControlModel::default(),
            curve_state: CurveCompressionState::default(),
            current_gop: GopStats::new(0, 0),
            gop_history: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_abr_creation() {
        let config = RcConfig::default();
        let controller = AbrController::new(&config);
        assert_eq!(controller.target_bitrate(), 5_000_000);
        assert_eq!(controller.frame_count(), 0);
    }

    #[test]
    fn test_first_pass_collection() {
        let config = RcConfig::default();
        let mut controller = AbrController::new(&config);
        controller.set_pass(1);

        for i in 0..30 {
            let frame_type = if i % 10 == 0 {
                FrameType::Key
            } else {
                FrameType::Inter
            };

            let output = controller.get_rc(frame_type, 1.0);

            let mut stats = FrameStats::new(i, frame_type);
            stats.bits = output.target_bits;
            stats.complexity = 1.0;
            controller.update(&stats);
        }

        let first_pass = controller.finalize_first_pass().expect("should succeed");
        assert_eq!(first_pass.frame_count, 30);
    }

    #[test]
    fn test_two_pass_encoding() {
        let config = RcConfig::default();
        let mut controller = AbrController::new(&config);
        controller.set_total_frames(60);
        controller.set_pass(1);

        // First pass
        for i in 0..30 {
            let frame_type = if i % 10 == 0 {
                FrameType::Key
            } else {
                FrameType::Inter
            };
            let _output = controller.get_rc(frame_type, 1.0 + (i as f32 % 3.0) * 0.1);

            let mut stats = FrameStats::new(i, frame_type);
            stats.bits = 100_000;
            stats.complexity = 1.0;
            controller.update(&stats);
        }

        let first_pass_stats = controller.finalize_first_pass().expect("should succeed");

        // Second pass
        let mut controller2 = AbrController::new(&config);
        controller2.set_total_frames(60);
        controller2.set_pass(2);
        controller2.set_first_pass_stats(first_pass_stats);

        let output = controller2.get_rc(FrameType::Key, 1.0);
        assert!(output.target_bits > 0);
    }

    #[test]
    fn test_target_file_size() {
        let config = RcConfig::default();
        let mut controller = AbrController::new(&config);

        let target_size = 10_000_000; // 10 MB
        let duration = 60.0; // 60 seconds
        controller.set_target_file_size(target_size, duration);

        // Target bitrate should be calculated from file size
        assert!(controller.target_bitrate() > 0);
    }

    #[test]
    fn test_bit_allocation() {
        let mut allocator = MultiPassAllocator::default();
        allocator.initialize(10_000_000, 100);

        let mut stats = FirstPassStats::default();
        for i in 0..10 {
            let frame_stats = FramePassStats {
                frame_index: i,
                combined_complexity: 1.0,
                ..Default::default()
            };
            stats.add_frame(frame_stats);
        }
        stats.finalize();

        let bits = allocator.allocate_frame_bits(FrameType::Key, 1.5, 1.0, &stats);
        assert!(bits > 0);
    }

    #[test]
    fn test_lookahead_analysis() {
        let mut lookahead = LookaheadAnalyzer::default();

        for i in 0..20 {
            lookahead.push_complexity(1.0 + (i as f32 % 5.0) * 0.1);
        }

        let analysis = lookahead.analyze();
        assert!(analysis.avg_complexity > 0.0);
        assert!(analysis.recommend_bits_multiplier > 0.0);
    }

    #[test]
    fn test_scene_cut_detection() {
        let mut lookahead = LookaheadAnalyzer::default();

        // Normal complexity
        for _ in 0..5 {
            lookahead.push_complexity(1.0);
        }

        // Sudden spike (scene cut)
        lookahead.push_complexity(3.0);

        let analysis = lookahead.analyze();
        assert!(analysis.has_scene_cut);
    }

    #[test]
    fn test_curve_compression() {
        let curve = CurveCompressionState::default();

        let high_complexity = 2.0;
        let low_complexity = 0.5;
        let avg = 1.0;

        let compressed_high = curve.compress(high_complexity, avg);
        let compressed_low = curve.compress(low_complexity, avg);

        // High complexity should be compressed (closer to avg)
        assert!(compressed_high < high_complexity * avg);
        // Low complexity should be expanded (closer to avg)
        assert!(compressed_low > low_complexity * avg);
    }

    #[test]
    fn test_rate_model_prediction() {
        let model = RateControlModel::default();

        let bits = model.predict(1.5, 28.0);
        assert!(bits > 0);

        // Higher complexity should predict more bits
        let bits_high = model.predict(2.0, 28.0);
        assert!(bits_high > bits);
    }

    #[test]
    fn test_linear_model() {
        let model = LinearModel::default();

        let bits1 = model.predict(1.0, 28.0);
        let bits2 = model.predict(2.0, 28.0);

        // Higher complexity should predict more bits
        assert!(bits2 > bits1);
    }

    #[test]
    fn test_quadratic_model() {
        let model = QuadraticModel::default();

        let bits1 = model.predict(1.0, 28.0);
        let bits2 = model.predict(2.0, 28.0);

        assert!(bits2 > bits1);
    }

    #[test]
    fn test_power_model() {
        let model = PowerModel::default();

        let bits1 = model.predict(1.0, 28.0);
        let bits2 = model.predict(2.0, 28.0);

        assert!(bits2 > bits1);
    }

    #[test]
    fn test_projected_file_size() {
        let config = RcConfig::default();
        let mut controller = AbrController::new(&config);
        controller.set_total_frames(100);

        // Encode some frames
        for i in 0..25 {
            let mut stats = FrameStats::new(i, FrameType::Inter);
            stats.bits = 100_000;
            controller.update(&stats);
        }

        let projected = controller.projected_file_size();
        assert!(projected.is_some());
        assert!(projected.expect("should succeed") > 0);
    }

    #[test]
    fn test_bitrate_deviation() {
        let config = RcConfig::default();
        let mut controller = AbrController::new(&config);

        for i in 0..30 {
            let mut stats = FrameStats::new(i, FrameType::Inter);
            stats.bits = 200_000; // Higher than target
            controller.update(&stats);
        }

        let deviation = controller.bitrate_deviation();
        // Should be positive (over target)
        assert!(deviation > 0.0);
    }

    #[test]
    fn test_reset() {
        let config = RcConfig::default();
        let mut controller = AbrController::new(&config);

        for i in 0..10 {
            let mut stats = FrameStats::new(i, FrameType::Inter);
            stats.bits = 100_000;
            controller.update(&stats);
        }

        controller.reset();
        assert_eq!(controller.frame_count(), 0);
        assert_eq!(controller.total_bits(), 0);
    }
}
