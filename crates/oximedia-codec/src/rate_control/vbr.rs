//! Variable Bitrate rate control with closed-loop feedback.
//!
//! VBR (Variable Bitrate) allows the bitrate to fluctuate within defined
//! bounds while targeting an average bitrate. This implementation provides:
//!
//! - Closed-loop feedback control for accurate bitrate targeting
//! - VBV (Video Buffering Verifier) and HRD compliance
//! - Advanced frame complexity analysis
//! - Lookahead buffer for optimal bit allocation
//! - Adaptive quantization integration
//! - Multi-pass encoding support
//!
//! # Architecture
//!
//! The VBR controller operates in a closed-loop fashion:
//!
//! ```text
//! Frame → Complexity → Bit → QP → Encode → Update → Next
//!         Analysis     Alloc   Calc          Stats    Frame
//!            ↓           ↓       ↓              ↓       ↓
//!         Lookahead   Model   Lambda        Feedback  Loop
//! ```

#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::struct_excessive_bools)]
#![forbid(unsafe_code)]

use crate::frame::FrameType;

use super::buffer::BufferModel;
use super::types::{FrameStats, GopStats, RcConfig, RcOutput};

/// Variable Bitrate rate controller with closed-loop feedback.
#[derive(Clone, Debug)]
pub struct VbrController {
    /// Target bitrate in bits per second.
    target_bitrate: u64,
    /// Maximum bitrate in bits per second.
    max_bitrate: u64,
    /// Minimum bitrate in bits per second.
    min_bitrate: u64,
    /// Current QP (floating point for smoother adjustments).
    current_qp: f32,
    /// Minimum QP.
    min_qp: u8,
    /// Maximum QP.
    max_qp: u8,
    /// I-frame QP offset.
    i_qp_offset: i8,
    /// B-frame QP offset.
    b_qp_offset: i8,
    /// Frame rate.
    framerate: f64,
    /// GOP length.
    gop_length: u32,
    /// Frame counter.
    frame_count: u64,
    /// Total bits encoded.
    total_bits: u64,
    /// Current GOP statistics.
    current_gop: GopStats,
    /// Historical GOP statistics for analysis.
    gop_history: Vec<GopStats>,
    /// Maximum GOP history size.
    max_gop_history: usize,
    /// Encoding pass (0 = single pass, 1 = first pass, 2 = second pass).
    pass: u8,
    /// First pass complexity data.
    first_pass_data: Option<FirstPassData>,
    /// Quality vs bitrate stability factor (0.0-1.0).
    quality_stability: f32,
    /// Bit reservoir (accumulated bits for averaging).
    bit_reservoir: i64,
    /// Maximum reservoir size.
    max_reservoir: i64,
    /// VBV buffer model for HRD compliance.
    vbv_buffer: Option<BufferModel>,
    /// Enable VBV/HRD compliance.
    enable_vbv: bool,
    /// Lookahead buffer size.
    lookahead_size: usize,
    /// Lookahead frame data.
    lookahead_frames: Vec<LookaheadFrameData>,
    /// Rate prediction model.
    rate_model: RatePredictionModel,
    /// Adaptive GOP sizing enabled.
    adaptive_gop: bool,
    /// Scene change detection threshold.
    scene_change_threshold: f32,
    /// Enable adaptive quantization.
    enable_aq: bool,
    /// AQ strength.
    #[allow(dead_code)]
    aq_strength: f32,
    /// PID controller state.
    pid_state: PidControllerState,
    /// Rate-distortion optimization parameters.
    rdo_params: RdoParameters,
    /// Frame type decision state.
    frame_type_state: FrameTypeDecisionState,
}

/// First pass analysis data.
#[derive(Clone, Debug, Default)]
pub struct FirstPassData {
    /// Per-frame complexity values.
    pub frame_complexity: Vec<f32>,
    /// Per-frame spatial complexity.
    pub spatial_complexity: Vec<f32>,
    /// Per-frame temporal complexity.
    pub temporal_complexity: Vec<f32>,
    /// Per-GOP total complexity.
    pub gop_complexity: Vec<f32>,
    /// Total complexity for the entire sequence.
    pub total_complexity: f32,
    /// Frame count.
    pub frame_count: u64,
    /// Suggested bits per frame based on complexity.
    pub suggested_bits: Vec<u64>,
    /// Scene change frame indices.
    pub scene_changes: Vec<u64>,
    /// Optimal GOP boundaries.
    pub gop_boundaries: Vec<u64>,
    /// Per-frame QP recommendations.
    pub recommended_qp: Vec<f32>,
}

impl FirstPassData {
    /// Add a frame's complexity data.
    pub fn add_frame(&mut self, spatial: f32, temporal: f32, combined: f32) {
        self.frame_complexity.push(combined);
        self.spatial_complexity.push(spatial);
        self.temporal_complexity.push(temporal);
        self.total_complexity += combined;
        self.frame_count += 1;
    }

    /// Mark a frame as scene change.
    pub fn mark_scene_change(&mut self, frame_num: u64) {
        self.scene_changes.push(frame_num);
    }

    /// Add GOP boundary.
    pub fn add_gop_boundary(&mut self, frame_num: u64) {
        self.gop_boundaries.push(frame_num);
    }

    /// Finalize a GOP's complexity.
    pub fn finalize_gop(&mut self) {
        let gop_start = self
            .gop_complexity
            .last()
            .map(|_| self.gop_boundaries.last().copied().unwrap_or(0))
            .unwrap_or(0) as usize;

        let gop_sum: f32 = self
            .frame_complexity
            .get(gop_start..)
            .map(|slice| slice.iter().sum())
            .unwrap_or(0.0);

        self.gop_complexity.push(gop_sum);
    }

    /// Calculate suggested bit allocation for second pass.
    pub fn calculate_bit_allocation(&mut self, total_bits: u64) {
        if self.total_complexity <= 0.0 || self.frame_complexity.is_empty() {
            return;
        }

        let bits_per_complexity = total_bits as f64 / self.total_complexity as f64;

        self.suggested_bits = self
            .frame_complexity
            .iter()
            .map(|c| ((*c as f64) * bits_per_complexity) as u64)
            .collect();

        // Calculate recommended QP based on complexity
        let avg_complexity = self.total_complexity / self.frame_count as f32;

        for complexity in &self.frame_complexity {
            let complexity_ratio = complexity / avg_complexity;
            // Higher complexity → higher QP to save bits
            let qp_adjustment = (complexity_ratio - 1.0) * 4.0;
            let base_qp = 28.0;
            let recommended = (base_qp + qp_adjustment).clamp(18.0, 51.0);
            self.recommended_qp.push(recommended);
        }
    }

    /// Get suggested bits for a frame.
    #[must_use]
    pub fn get_suggested_bits(&self, frame_num: u64) -> Option<u64> {
        self.suggested_bits.get(frame_num as usize).copied()
    }

    /// Get recommended QP for a frame.
    #[must_use]
    pub fn get_recommended_qp(&self, frame_num: u64) -> Option<f32> {
        self.recommended_qp.get(frame_num as usize).copied()
    }

    /// Check if frame is a detected scene change.
    #[must_use]
    pub fn is_scene_change(&self, frame_num: u64) -> bool {
        self.scene_changes.contains(&frame_num)
    }
}

/// Lookahead frame data for rate control.
#[derive(Clone, Debug, Default)]
struct LookaheadFrameData {
    /// Frame index.
    #[allow(dead_code)]
    frame_index: u64,
    /// Spatial complexity.
    #[allow(dead_code)]
    spatial_complexity: f32,
    /// Temporal complexity.
    #[allow(dead_code)]
    temporal_complexity: f32,
    /// Combined complexity.
    combined_complexity: f32,
    /// Is scene change.
    is_scene_change: bool,
    /// Predicted frame type.
    #[allow(dead_code)]
    predicted_type: FrameType,
    /// Predicted bits needed.
    #[allow(dead_code)]
    predicted_bits: u64,
}

/// Rate prediction model using multiple regression approaches.
#[derive(Clone, Debug)]
struct RatePredictionModel {
    /// Linear model coefficients: bits = a * complexity + b
    linear_a: f64,
    linear_b: f64,
    /// Quadratic model coefficients: bits = a * complexity^2 + b * complexity + c
    quad_a: f64,
    quad_b: f64,
    quad_c: f64,
    /// Power model coefficients: bits = a * complexity^b
    power_a: f64,
    power_b: f64,
    /// Model selection (0 = linear, 1 = quadratic, 2 = power)
    active_model: u8,
    /// Historical data for model fitting.
    history_complexity: Vec<f32>,
    history_bits: Vec<u64>,
    /// Maximum history size.
    max_history: usize,
    /// Model update counter.
    update_count: u32,
}

impl Default for RatePredictionModel {
    fn default() -> Self {
        Self {
            linear_a: 100_000.0,
            linear_b: 50_000.0,
            quad_a: 10_000.0,
            quad_b: 50_000.0,
            quad_c: 20_000.0,
            power_a: 100_000.0,
            power_b: 1.2,
            active_model: 0,
            history_complexity: Vec::new(),
            history_bits: Vec::new(),
            max_history: 100,
            update_count: 0,
        }
    }
}

impl RatePredictionModel {
    /// Predict bits needed for given complexity.
    fn predict(&self, complexity: f32) -> u64 {
        if complexity <= 0.0 {
            return 50_000;
        }

        let prediction = match self.active_model {
            0 => self.predict_linear(complexity),
            1 => self.predict_quadratic(complexity),
            2 => self.predict_power(complexity),
            _ => self.predict_linear(complexity),
        };

        prediction.max(1000.0) as u64
    }

    /// Linear prediction.
    fn predict_linear(&self, complexity: f32) -> f64 {
        self.linear_a * complexity as f64 + self.linear_b
    }

    /// Quadratic prediction.
    fn predict_quadratic(&self, complexity: f32) -> f64 {
        let c = complexity as f64;
        self.quad_a * c * c + self.quad_b * c + self.quad_c
    }

    /// Power prediction.
    fn predict_power(&self, complexity: f32) -> f64 {
        self.power_a * (complexity as f64).powf(self.power_b)
    }

    /// Update model with new observation.
    fn update(&mut self, complexity: f32, bits: u64) {
        self.history_complexity.push(complexity);
        self.history_bits.push(bits);

        if self.history_complexity.len() > self.max_history {
            self.history_complexity.remove(0);
            self.history_bits.remove(0);
        }

        // Update models periodically
        self.update_count += 1;
        if self.update_count >= 10 {
            self.fit_models();
            self.update_count = 0;
        }
    }

    /// Fit all models using historical data.
    fn fit_models(&mut self) {
        if self.history_complexity.len() < 5 {
            return;
        }

        self.fit_linear_model();
        self.fit_quadratic_model();
        self.fit_power_model();
        self.select_best_model();
    }

    /// Fit linear model using least squares.
    fn fit_linear_model(&mut self) {
        let n = self.history_complexity.len();
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_xx = 0.0;

        for i in 0..n {
            let x = self.history_complexity[i] as f64;
            let y = self.history_bits[i] as f64;
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_xx += x * x;
        }

        let n_f = n as f64;
        let denominator = n_f * sum_xx - sum_x * sum_x;

        if denominator.abs() > 1e-6 {
            self.linear_a = (n_f * sum_xy - sum_x * sum_y) / denominator;
            self.linear_b = (sum_y - self.linear_a * sum_x) / n_f;
        }
    }

    /// Fit quadratic model (simplified).
    fn fit_quadratic_model(&mut self) {
        // Simplified quadratic fitting
        // In practice, would use proper polynomial regression
        let n = self.history_complexity.len();
        if n < 3 {
            return;
        }

        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_x2 = 0.0;
        let mut sum_xy = 0.0;

        for i in 0..n {
            let x = self.history_complexity[i] as f64;
            let y = self.history_bits[i] as f64;
            sum_x += x;
            sum_y += y;
            sum_x2 += x * x;
            sum_xy += x * y;
        }

        let n_f = n as f64;
        let avg_x = sum_x / n_f;
        let avg_y = sum_y / n_f;

        // Simplified quadratic coefficients
        self.quad_b = self.linear_a;
        self.quad_a = (sum_xy - n_f * avg_x * avg_y) / (sum_x2 - n_f * avg_x * avg_x) * 0.1;
        self.quad_c = avg_y - self.quad_b * avg_x - self.quad_a * avg_x * avg_x;
    }

    /// Fit power model using log transformation.
    fn fit_power_model(&mut self) {
        let n = self.history_complexity.len();
        let mut sum_log_x = 0.0;
        let mut sum_log_y = 0.0;
        let mut sum_log_xy = 0.0;
        let mut sum_log_xx = 0.0;
        let mut count = 0;

        for i in 0..n {
            let x = self.history_complexity[i] as f64;
            let y = self.history_bits[i] as f64;

            if x > 0.0 && y > 0.0 {
                let log_x = x.ln();
                let log_y = y.ln();
                sum_log_x += log_x;
                sum_log_y += log_y;
                sum_log_xy += log_x * log_y;
                sum_log_xx += log_x * log_x;
                count += 1;
            }
        }

        if count < 3 {
            return;
        }

        let n_f = count as f64;
        let denominator = n_f * sum_log_xx - sum_log_x * sum_log_x;

        if denominator.abs() > 1e-6 {
            self.power_b = (n_f * sum_log_xy - sum_log_x * sum_log_y) / denominator;
            let log_a = (sum_log_y - self.power_b * sum_log_x) / n_f;
            self.power_a = log_a.exp();
        }
    }

    /// Select the best model based on prediction error.
    fn select_best_model(&mut self) {
        if self.history_complexity.len() < 5 {
            return;
        }

        let mut error_linear = 0.0;
        let mut error_quadratic = 0.0;
        let mut error_power = 0.0;

        for i in 0..self.history_complexity.len() {
            let complexity = self.history_complexity[i];
            let actual_bits = self.history_bits[i] as f64;

            let pred_linear = self.predict_linear(complexity);
            let pred_quadratic = self.predict_quadratic(complexity);
            let pred_power = self.predict_power(complexity);

            error_linear += (actual_bits - pred_linear).abs();
            error_quadratic += (actual_bits - pred_quadratic).abs();
            error_power += (actual_bits - pred_power).abs();
        }

        // Select model with lowest error
        if error_linear <= error_quadratic && error_linear <= error_power {
            self.active_model = 0;
        } else if error_quadratic <= error_power {
            self.active_model = 1;
        } else {
            self.active_model = 2;
        }
    }
}

/// PID controller state for closed-loop rate control.
#[derive(Clone, Debug, Default)]
struct PidControllerState {
    /// Proportional gain.
    kp: f32,
    /// Integral gain.
    ki: f32,
    /// Derivative gain.
    kd: f32,
    /// Previous error.
    prev_error: f32,
    /// Accumulated error (integral term).
    integral: f32,
    /// Maximum integral windup.
    max_integral: f32,
}

impl PidControllerState {
    /// Create new PID controller with default gains.
    fn new() -> Self {
        Self {
            kp: 0.5,
            ki: 0.1,
            kd: 0.05,
            prev_error: 0.0,
            integral: 0.0,
            max_integral: 10.0,
        }
    }

    /// Calculate PID output.
    fn calculate(&mut self, error: f32) -> f32 {
        // Proportional term
        let p_term = self.kp * error;

        // Integral term with anti-windup
        self.integral += error;
        self.integral = self.integral.clamp(-self.max_integral, self.max_integral);
        let i_term = self.ki * self.integral;

        // Derivative term
        let d_term = self.kd * (error - self.prev_error);
        self.prev_error = error;

        p_term + i_term + d_term
    }

    /// Reset controller state.
    fn reset(&mut self) {
        self.prev_error = 0.0;
        self.integral = 0.0;
    }
}

/// Rate-distortion optimization parameters.
#[derive(Clone, Debug)]
struct RdoParameters {
    /// Base lambda value.
    base_lambda: f64,
    /// Lambda multiplier for I-frames.
    i_lambda_mult: f64,
    /// Lambda multiplier for B-frames.
    b_lambda_mult: f64,
    /// Enable psychovisual RDO.
    #[allow(dead_code)]
    psy_rd: bool,
    /// Psychovisual strength.
    #[allow(dead_code)]
    psy_strength: f64,
}

impl Default for RdoParameters {
    fn default() -> Self {
        Self {
            base_lambda: 1.0,
            i_lambda_mult: 0.6,
            b_lambda_mult: 1.4,
            psy_rd: true,
            psy_strength: 1.0,
        }
    }
}

impl RdoParameters {
    /// Calculate lambda for given QP and frame type.
    fn calculate_lambda(&self, qp: f32, frame_type: FrameType) -> f64 {
        let base = 0.85 * 2.0_f64.powf((f64::from(qp) - 12.0) / 3.0);

        let multiplier = match frame_type {
            FrameType::Key => self.i_lambda_mult,
            FrameType::BiDir => self.b_lambda_mult,
            _ => 1.0,
        };

        base * multiplier * self.base_lambda
    }

    /// Calculate motion estimation lambda.
    fn calculate_lambda_me(&self, lambda: f64) -> f64 {
        lambda.sqrt()
    }
}

/// Frame type decision state.
#[derive(Clone, Debug, Default)]
struct FrameTypeDecisionState {
    /// Frames since last keyframe.
    frames_since_keyframe: u32,
    /// Consecutive B-frames count.
    consecutive_b_frames: u32,
    /// Maximum consecutive B-frames.
    max_b_frames: u32,
    /// Force next keyframe.
    force_keyframe: bool,
}

impl FrameTypeDecisionState {
    /// Create new frame type decision state.
    fn new(max_b_frames: u32) -> Self {
        Self {
            frames_since_keyframe: 0,
            consecutive_b_frames: 0,
            max_b_frames,
            force_keyframe: false,
        }
    }

    /// Decide frame type based on GOP structure and scene changes.
    fn decide_type(
        &mut self,
        gop_length: u32,
        is_scene_change: bool,
        adaptive_gop: bool,
    ) -> FrameType {
        // Check for forced keyframe or GOP boundary
        if self.force_keyframe
            || self.frames_since_keyframe == 0
            || (!adaptive_gop && self.frames_since_keyframe >= gop_length)
            || is_scene_change
        {
            self.frames_since_keyframe = 1;
            self.consecutive_b_frames = 0;
            self.force_keyframe = false;
            return FrameType::Key;
        }

        self.frames_since_keyframe += 1;

        // Adaptive B-frame decision
        if self.max_b_frames > 0 && self.consecutive_b_frames < self.max_b_frames {
            // Use B-frames for middle frames in mini-GOP
            let mini_gop_pos = self.frames_since_keyframe % (self.max_b_frames + 1);
            if mini_gop_pos > 0 && mini_gop_pos <= self.max_b_frames {
                self.consecutive_b_frames += 1;
                return FrameType::BiDir;
            }
        }

        self.consecutive_b_frames = 0;
        FrameType::Inter
    }

    /// Force next frame to be keyframe.
    fn force_next_keyframe(&mut self) {
        self.force_keyframe = true;
    }
}

impl VbrController {
    /// Create a new VBR controller from configuration.
    #[must_use]
    pub fn new(config: &RcConfig) -> Self {
        let max_bitrate = config.max_bitrate.unwrap_or(config.target_bitrate * 2);
        let min_bitrate = config.min_bitrate.unwrap_or(config.target_bitrate / 4);
        let framerate = config.framerate();

        let vbv_buffer = if config.buffer_size > 0 {
            Some(BufferModel::new(
                config.buffer_size,
                config.target_bitrate,
                framerate,
                config.initial_buffer_fullness as f64,
            ))
        } else {
            None
        };

        Self {
            target_bitrate: config.target_bitrate,
            max_bitrate,
            min_bitrate,
            current_qp: config.initial_qp as f32,
            min_qp: config.min_qp,
            max_qp: config.max_qp,
            i_qp_offset: config.i_qp_offset,
            b_qp_offset: config.b_qp_offset,
            framerate,
            gop_length: config.gop_length,
            frame_count: 0,
            total_bits: 0,
            current_gop: GopStats::new(0, 0),
            gop_history: Vec::new(),
            max_gop_history: 10,
            pass: 0,
            first_pass_data: None,
            quality_stability: 0.5,
            bit_reservoir: 0,
            max_reservoir: (config.target_bitrate as i64) * 2,
            vbv_buffer,
            enable_vbv: config.buffer_size > 0,
            lookahead_size: config.lookahead_depth.min(250),
            lookahead_frames: Vec::new(),
            rate_model: RatePredictionModel::default(),
            adaptive_gop: true,
            scene_change_threshold: config.scene_cut_threshold,
            enable_aq: config.enable_aq,
            aq_strength: config.aq_strength,
            pid_state: PidControllerState::new(),
            rdo_params: RdoParameters::default(),
            frame_type_state: FrameTypeDecisionState::new(3),
        }
    }

    /// Set the encoding pass (0 = single, 1 = first, 2 = second).
    pub fn set_pass(&mut self, pass: u8) {
        self.pass = pass.min(2);
        if pass == 1 {
            self.first_pass_data = Some(FirstPassData::default());
        }
    }

    /// Set quality stability factor (0.0-1.0).
    pub fn set_quality_stability(&mut self, stability: f32) {
        self.quality_stability = stability.clamp(0.0, 1.0);
    }

    /// Enable or disable VBV/HRD compliance.
    pub fn set_vbv_enabled(&mut self, enabled: bool) {
        self.enable_vbv = enabled;
    }

    /// Set lookahead buffer size (10-250 frames).
    pub fn set_lookahead_size(&mut self, size: usize) {
        self.lookahead_size = size.clamp(10, 250);
        self.lookahead_frames.reserve(self.lookahead_size);
    }

    /// Enable or disable adaptive GOP sizing.
    pub fn set_adaptive_gop(&mut self, enabled: bool) {
        self.adaptive_gop = enabled;
    }

    /// Set scene change detection threshold.
    pub fn set_scene_change_threshold(&mut self, threshold: f32) {
        self.scene_change_threshold = threshold.clamp(0.0, 1.0);
    }

    /// Import first pass data for second pass encoding.
    pub fn set_first_pass_data(&mut self, data: FirstPassData) {
        self.first_pass_data = Some(data);
    }

    /// Add frame to lookahead buffer.
    pub fn push_lookahead_frame(
        &mut self,
        spatial: f32,
        temporal: f32,
        combined: f32,
        is_scene_change: bool,
    ) {
        let frame_data = LookaheadFrameData {
            frame_index: self.frame_count + self.lookahead_frames.len() as u64,
            spatial_complexity: spatial,
            temporal_complexity: temporal,
            combined_complexity: combined,
            is_scene_change,
            predicted_type: FrameType::Inter,
            predicted_bits: self.rate_model.predict(combined),
        };

        self.lookahead_frames.push(frame_data);

        // Trim to size
        if self.lookahead_frames.len() > self.lookahead_size {
            self.lookahead_frames.remove(0);
        }
    }

    /// Get rate control output for a frame with closed-loop feedback.
    #[must_use]
    pub fn get_rc(&mut self, frame_type: FrameType, complexity: f32) -> RcOutput {
        // Determine frame type using lookahead and adaptive GOP
        let is_scene_change = self.detect_scene_change_from_lookahead();
        let actual_frame_type =
            self.frame_type_state
                .decide_type(self.gop_length, is_scene_change, self.adaptive_gop);

        // Use provided frame_type if it's more restrictive (Key)
        let final_frame_type = if frame_type == FrameType::Key {
            FrameType::Key
        } else {
            actual_frame_type
        };

        // Calculate target bits using multiple strategies
        let target_bits = self.calculate_target_bits(final_frame_type, complexity);

        // Closed-loop QP adjustment using PID controller
        let qp_adjustment = self.calculate_closed_loop_qp_adjustment();
        let adjusted_qp = self.current_qp + qp_adjustment;

        // Apply frame type offset
        let offset = match final_frame_type {
            FrameType::Key => self.i_qp_offset,
            FrameType::BiDir => self.b_qp_offset,
            FrameType::Inter | FrameType::Switch => 0,
        };

        // Apply complexity-based adjustment
        let complexity_adjustment = self.calculate_complexity_adjustment(complexity);

        // Apply VBV buffer constraint if enabled
        let vbv_adjustment = if self.enable_vbv {
            self.calculate_vbv_qp_adjustment()
        } else {
            0.0
        };

        let final_qp = (adjusted_qp + offset as f32 + complexity_adjustment + vbv_adjustment)
            .clamp(self.min_qp as f32, self.max_qp as f32);
        let qp = final_qp.round() as u8;

        // Calculate bit limits with VBV constraints
        let (min_bits, max_bits) = self.calculate_bit_limits(final_frame_type, target_bits);

        // Calculate lambda for RDO
        let lambda = self.rdo_params.calculate_lambda(final_qp, final_frame_type);
        let lambda_me = self.rdo_params.calculate_lambda_me(lambda);

        let mut output = RcOutput {
            qp,
            qp_f: final_qp,
            target_bits,
            min_bits,
            max_bits,
            lambda,
            lambda_me,
            force_keyframe: final_frame_type == FrameType::Key && is_scene_change,
            ..Default::default()
        };

        // Apply adaptive quantization if enabled
        if self.enable_aq {
            output.qp_offsets = Some(self.calculate_aq_offsets(final_qp));
        }

        output
    }

    /// Detect scene change from lookahead buffer.
    fn detect_scene_change_from_lookahead(&self) -> bool {
        if self.lookahead_frames.is_empty() {
            return false;
        }

        // Check current frame in lookahead
        self.lookahead_frames
            .first()
            .map(|f| f.is_scene_change)
            .unwrap_or(false)
    }

    /// Calculate target bits for a frame with multiple strategies.
    fn calculate_target_bits(&self, frame_type: FrameType, complexity: f32) -> u64 {
        let base_target = self.bits_per_frame_at_bitrate(self.target_bitrate);

        // Check if we have second pass data
        if self.pass == 2 {
            if let Some(ref data) = self.first_pass_data {
                if let Some(suggested) = data.get_suggested_bits(self.frame_count) {
                    return suggested;
                }
            }
        }

        // Use rate prediction model
        let model_prediction = self.rate_model.predict(complexity);

        // Frame type multiplier
        let type_multiplier = match frame_type {
            FrameType::Key => 3.0,
            FrameType::Inter => 1.0,
            FrameType::BiDir => 0.5,
            FrameType::Switch => 1.5,
        };

        // Lookahead-based adjustment
        let lookahead_mult = self.calculate_lookahead_multiplier();

        // Complexity-based adjustment
        let complexity_multiplier: f64 = if complexity > 0.0 {
            (f64::from(complexity) / f64::from(self.average_complexity())).clamp(0.5, 2.0)
        } else {
            1.0
        };

        // Reservoir adjustment
        let reservoir_adjustment = self.calculate_reservoir_adjustment();

        // Combine predictions
        let target =
            (base_target as f64 * type_multiplier * complexity_multiplier * lookahead_mult)
                .max(model_prediction as f64)
                + reservoir_adjustment;

        let adjusted_target = target.max(base_target as f64 / 4.0);

        adjusted_target as u64
    }

    /// Calculate lookahead-based bitrate multiplier.
    fn calculate_lookahead_multiplier(&self) -> f64 {
        if self.lookahead_frames.is_empty() {
            return 1.0;
        }

        // Calculate average complexity in lookahead window
        let avg_future_complexity: f32 = self
            .lookahead_frames
            .iter()
            .map(|f| f.combined_complexity)
            .sum::<f32>()
            / self.lookahead_frames.len() as f32;

        let current_complexity = self
            .lookahead_frames
            .first()
            .map(|f| f.combined_complexity)
            .unwrap_or(1.0);

        // Adjust based on future complexity trend
        if avg_future_complexity > current_complexity * 1.3 {
            // Future is more complex, save bits now
            0.9
        } else if avg_future_complexity < current_complexity * 0.7 {
            // Future is simpler, spend more bits now
            1.1
        } else {
            1.0
        }
    }

    /// Calculate closed-loop QP adjustment using PID controller.
    fn calculate_closed_loop_qp_adjustment(&mut self) -> f32 {
        if self.frame_count == 0 {
            return 0.0;
        }

        let elapsed_time = self.frame_count as f64 / self.framerate;
        if elapsed_time <= 0.0 {
            return 0.0;
        }

        let actual_bitrate = self.total_bits as f64 / elapsed_time;
        let error = (actual_bitrate - self.target_bitrate as f64) / self.target_bitrate as f64;

        // Use PID controller for smooth adjustment
        let pid_output = self.pid_state.calculate(error as f32);

        // Scale by quality stability
        pid_output * (1.0 - self.quality_stability)
    }

    /// Calculate QP adjustment based on frame complexity.
    fn calculate_complexity_adjustment(&self, complexity: f32) -> f32 {
        let avg = self.average_complexity();
        if avg <= 0.0 {
            return 0.0;
        }

        let ratio = complexity / avg;

        // Higher complexity frames get slightly higher QP to save bits
        // Lower complexity frames get slightly lower QP for better quality
        let adjustment = (ratio - 1.0) * 2.0 * self.quality_stability;
        adjustment.clamp(-2.0, 2.0)
    }

    /// Calculate VBV buffer-based QP adjustment.
    fn calculate_vbv_qp_adjustment(&self) -> f32 {
        if let Some(ref buffer) = self.vbv_buffer {
            let fullness = buffer.fullness();

            // If buffer is too full, increase QP to reduce bitrate
            // If buffer is too empty, decrease QP to increase bitrate
            let target_fullness = 0.5;
            let error = fullness - target_fullness;

            // Aggressive adjustment when near overflow/underflow
            if error > 0.3 {
                // Buffer nearly full, strongly increase QP
                (error - 0.3) * 20.0
            } else if error < -0.3 {
                // Buffer nearly empty, strongly decrease QP
                (error + 0.3) * 20.0
            } else {
                // Normal range, gentle adjustment
                error * 5.0
            }
        } else {
            0.0
        }
    }

    /// Calculate bit limits with VBV constraints.
    fn calculate_bit_limits(&self, frame_type: FrameType, target_bits: u64) -> (u64, u64) {
        let base_min = self.bits_per_frame_at_bitrate(self.min_bitrate) / 4;
        let base_max = self.bits_per_frame_at_bitrate(self.max_bitrate) * 4;

        let mut min_bits = base_min;
        let mut max_bits = base_max;

        // Apply VBV constraints
        if self.enable_vbv {
            if let Some(ref buffer) = self.vbv_buffer {
                let available = buffer.max_frame_bits();
                max_bits = max_bits.min(available);

                // Ensure we don't underflow
                let min_to_prevent_underflow = target_bits / 2;
                min_bits = min_bits.max(min_to_prevent_underflow);
            }
        }

        // Frame type specific limits
        match frame_type {
            FrameType::Key => {
                // I-frames can be much larger
                max_bits = max_bits.max(target_bits * 5);
            }
            FrameType::BiDir => {
                // B-frames should be smaller
                max_bits = max_bits.min(target_bits * 2);
            }
            _ => {}
        }

        (min_bits, max_bits)
    }

    /// Calculate AQ offsets (simplified placeholder).
    fn calculate_aq_offsets(&self, _base_qp: f32) -> Vec<f32> {
        // In a full implementation, this would analyze the frame
        // and return per-block QP offsets
        Vec::new()
    }

    /// Calculate reservoir adjustment (bits to borrow or save).
    fn calculate_reservoir_adjustment(&self) -> f64 {
        let target_per_frame = self.bits_per_frame_at_bitrate(self.target_bitrate);

        // Slowly use or build reservoir
        let reservoir_factor = self.bit_reservoir as f64 / self.max_reservoir as f64;
        reservoir_factor * (target_per_frame as f64 * 0.1)
    }

    /// Calculate target bits per frame at a given bitrate.
    fn bits_per_frame_at_bitrate(&self, bitrate: u64) -> u64 {
        if self.framerate <= 0.0 {
            return 0;
        }
        (bitrate as f64 / self.framerate) as u64
    }

    /// Get average complexity from history.
    fn average_complexity(&self) -> f32 {
        if self.gop_history.is_empty() {
            return 1.0;
        }

        let total: f32 = self.gop_history.iter().map(|g| g.average_complexity).sum();
        (total / self.gop_history.len() as f32).max(0.01)
    }

    /// Update controller with frame statistics (closed-loop feedback).
    pub fn update(&mut self, stats: &FrameStats) {
        self.frame_count += 1;
        self.total_bits += stats.bits;

        // Update bit reservoir
        let target = self.bits_per_frame_at_bitrate(self.target_bitrate);
        self.bit_reservoir += target as i64 - stats.bits as i64;
        self.bit_reservoir = self
            .bit_reservoir
            .clamp(-self.max_reservoir, self.max_reservoir);

        // Update VBV buffer
        if self.enable_vbv {
            if let Some(ref mut buffer) = self.vbv_buffer {
                buffer.fill_for_frame();
                buffer.remove_frame_bits(stats.bits);
            }
        }

        // Update rate prediction model
        self.rate_model.update(stats.complexity, stats.bits);

        // Update current GOP
        self.current_gop.add_frame(stats.clone());

        // Check for GOP boundary
        if stats.frame_type == FrameType::Key && self.current_gop.frame_count > 1 {
            self.finalize_gop();
        }

        // First pass data collection
        if self.pass == 1 {
            if let Some(ref mut data) = self.first_pass_data {
                data.add_frame(
                    stats.spatial_complexity,
                    stats.temporal_complexity,
                    stats.complexity,
                );

                if stats.scene_cut {
                    data.mark_scene_change(stats.frame_num);
                }

                if stats.frame_type == FrameType::Key && data.frame_count > 1 {
                    data.finalize_gop();
                    data.add_gop_boundary(stats.frame_num);
                }
            }
        }

        // Adjust base QP based on results
        self.adjust_base_qp(stats);

        // Remove oldest lookahead frame as it's now encoded
        if !self.lookahead_frames.is_empty() {
            self.lookahead_frames.remove(0);
        }
    }

    /// Finalize current GOP and start a new one.
    fn finalize_gop(&mut self) {
        self.gop_history.push(self.current_gop.clone());
        if self.gop_history.len() > self.max_gop_history {
            self.gop_history.remove(0);
        }

        let next_gop_index = self.current_gop.gop_index + 1;
        self.current_gop = GopStats::new(next_gop_index, self.frame_count);
    }

    /// Adjust base QP based on encoding results.
    fn adjust_base_qp(&mut self, stats: &FrameStats) {
        if stats.target_bits == 0 {
            return;
        }

        let accuracy = stats.bits as f32 / stats.target_bits as f32;

        // Gradual QP adjustment
        let adjustment = if accuracy > 1.3 {
            0.15
        } else if accuracy > 1.1 {
            0.05
        } else if accuracy < 0.7 {
            -0.15
        } else if accuracy < 0.9 {
            -0.05
        } else {
            0.0
        };

        self.current_qp =
            (self.current_qp + adjustment).clamp(self.min_qp as f32, self.max_qp as f32);
    }

    /// Finalize first pass and get data.
    #[must_use]
    pub fn finalize_first_pass(&mut self) -> Option<FirstPassData> {
        if self.pass != 1 {
            return None;
        }

        if let Some(ref mut data) = self.first_pass_data {
            data.finalize_gop();

            // Calculate total duration
            let duration = self.frame_count as f64 / self.framerate;
            let total_bits = (self.target_bitrate as f64 * duration) as u64;
            data.calculate_bit_allocation(total_bits);
        }

        self.first_pass_data.take()
    }

    /// Get target bitrate.
    #[must_use]
    pub fn target_bitrate(&self) -> u64 {
        self.target_bitrate
    }

    /// Get maximum bitrate.
    #[must_use]
    pub fn max_bitrate(&self) -> u64 {
        self.max_bitrate
    }

    /// Get minimum bitrate.
    #[must_use]
    pub fn min_bitrate(&self) -> u64 {
        self.min_bitrate
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

    /// Get current QP.
    #[must_use]
    pub fn current_qp(&self) -> f32 {
        self.current_qp
    }

    /// Get frame count.
    #[must_use]
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Get bit reservoir level.
    #[must_use]
    pub fn bit_reservoir(&self) -> i64 {
        self.bit_reservoir
    }

    /// Get VBV buffer fullness (0.0-1.0).
    #[must_use]
    pub fn vbv_fullness(&self) -> f32 {
        self.vbv_buffer
            .as_ref()
            .map(|b| b.fullness())
            .unwrap_or(0.5)
    }

    /// Force next frame to be a keyframe.
    pub fn force_keyframe(&mut self) {
        self.frame_type_state.force_next_keyframe();
    }

    /// Reset the controller state.
    pub fn reset(&mut self) {
        self.frame_count = 0;
        self.total_bits = 0;
        self.bit_reservoir = 0;
        self.current_gop = GopStats::new(0, 0);
        self.gop_history.clear();
        self.lookahead_frames.clear();
        self.pid_state.reset();
        self.frame_type_state = FrameTypeDecisionState::new(self.frame_type_state.max_b_frames);

        if let Some(ref mut buffer) = self.vbv_buffer {
            buffer.reset();
        }
    }
}

impl Default for VbrController {
    fn default() -> Self {
        Self {
            target_bitrate: 5_000_000,
            max_bitrate: 10_000_000,
            min_bitrate: 1_000_000,
            current_qp: 28.0,
            min_qp: 1,
            max_qp: 63,
            i_qp_offset: -2,
            b_qp_offset: 2,
            framerate: 30.0,
            gop_length: 250,
            frame_count: 0,
            total_bits: 0,
            current_gop: GopStats::new(0, 0),
            gop_history: Vec::new(),
            max_gop_history: 10,
            pass: 0,
            first_pass_data: None,
            quality_stability: 0.5,
            bit_reservoir: 0,
            max_reservoir: 10_000_000,
            vbv_buffer: None,
            enable_vbv: false,
            lookahead_size: 40,
            lookahead_frames: Vec::new(),
            rate_model: RatePredictionModel::default(),
            adaptive_gop: true,
            scene_change_threshold: 0.4,
            enable_aq: true,
            aq_strength: 1.0,
            pid_state: PidControllerState::new(),
            rdo_params: RdoParameters::default(),
            frame_type_state: FrameTypeDecisionState::new(3),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_controller() -> VbrController {
        let config = RcConfig::vbr(5_000_000, 10_000_000);
        VbrController::new(&config)
    }

    #[test]
    fn test_vbr_creation() {
        let controller = create_test_controller();
        assert_eq!(controller.target_bitrate(), 5_000_000);
        assert_eq!(controller.max_bitrate(), 10_000_000);
    }

    #[test]
    fn test_get_rc() {
        let mut controller = create_test_controller();
        let output = controller.get_rc(FrameType::Key, 1.0);

        assert!(!output.drop_frame);
        assert!(output.target_bits > 0);
        assert!(output.qp > 0);
        assert!(output.lambda > 0.0);
    }

    #[test]
    fn test_closed_loop_feedback() {
        let mut controller = create_test_controller();

        // Simulate encoding frames
        for i in 0..30 {
            let frame_type = if i % 10 == 0 {
                FrameType::Key
            } else {
                FrameType::Inter
            };

            let output = controller.get_rc(frame_type, 1.0);

            let mut stats = FrameStats::new(i, frame_type);
            stats.bits = output.target_bits;
            stats.target_bits = output.target_bits;
            stats.qp = output.qp;
            stats.qp_f = output.qp_f;
            stats.complexity = 1.0;

            controller.update(&stats);
        }

        // Check that controller adapted
        assert_eq!(controller.frame_count(), 30);
        assert!(controller.current_bitrate() > 0.0);
    }

    #[test]
    fn test_lookahead_buffer() {
        let mut controller = create_test_controller();
        controller.set_lookahead_size(20);

        // Fill lookahead buffer
        for _ in 0..25 {
            controller.push_lookahead_frame(1.0, 1.0, 1.0, false);
        }

        assert!(controller.lookahead_frames.len() <= 20);
    }

    #[test]
    fn test_vbv_compliance() {
        let mut config = RcConfig::vbr(5_000_000, 10_000_000);
        config.buffer_size = 5_000_000;
        let mut controller = VbrController::new(&config);
        controller.set_vbv_enabled(true);

        let output = controller.get_rc(FrameType::Key, 1.0);
        assert!(output.max_bits > 0);

        let mut stats = FrameStats::new(0, FrameType::Key);
        stats.bits = output.target_bits;
        stats.target_bits = output.target_bits;
        controller.update(&stats);

        let fullness = controller.vbv_fullness();
        assert!(fullness >= 0.0 && fullness <= 1.0);
    }

    #[test]
    fn test_rate_prediction_model() {
        let mut model = RatePredictionModel::default();

        // Add some observations
        for i in 1..20 {
            let complexity = i as f32 * 0.1;
            let bits = 50_000 + i * 5_000;
            model.update(complexity, bits);
        }

        let prediction = model.predict(1.5);
        assert!(prediction > 0);
    }

    #[test]
    fn test_pid_controller() {
        let mut pid = PidControllerState::new();

        // Simulate error correction
        let mut error = 1.0;
        for _ in 0..10 {
            let output = pid.calculate(error);
            error -= output * 0.1; // Simulate system response
        }

        // Error should decrease
        assert!(error.abs() < 1.0);
    }

    #[test]
    fn test_adaptive_gop() {
        let mut controller = create_test_controller();
        controller.set_adaptive_gop(true);

        // Simulate scene change
        controller.push_lookahead_frame(1.0, 1.0, 1.0, false);
        controller.push_lookahead_frame(5.0, 5.0, 5.0, true); // Scene change

        let output = controller.get_rc(FrameType::Inter, 1.0);
        // Controller should handle scene changes
        assert!(output.qp > 0);
    }

    #[test]
    fn test_two_pass_encoding() {
        let mut controller = create_test_controller();
        controller.set_pass(1);

        // First pass
        for i in 0..30 {
            let frame_type = if i % 10 == 0 {
                FrameType::Key
            } else {
                FrameType::Inter
            };

            let _output = controller.get_rc(frame_type, 1.0 + (i as f32 % 3.0) * 0.2);

            let mut stats = FrameStats::new(i, frame_type);
            stats.bits = 100_000;
            stats.target_bits = 100_000;
            stats.spatial_complexity = 1.0;
            stats.temporal_complexity = 1.0;
            stats.complexity = 1.0;
            controller.update(&stats);
        }

        let first_pass_data = controller.finalize_first_pass().expect("should succeed");
        assert_eq!(first_pass_data.frame_count, 30);
        assert!(!first_pass_data.suggested_bits.is_empty());

        // Second pass
        let mut controller2 = create_test_controller();
        controller2.set_pass(2);
        controller2.set_first_pass_data(first_pass_data);

        let output = controller2.get_rc(FrameType::Key, 1.0);
        assert!(output.target_bits > 0);
    }

    #[test]
    fn test_frame_type_decision() {
        let mut state = FrameTypeDecisionState::new(3);

        // First frame should be keyframe
        let ft = state.decide_type(250, false, false);
        assert_eq!(ft, FrameType::Key);

        // Next frames should follow B-frame pattern
        for _ in 0..3 {
            let ft = state.decide_type(250, false, false);
            assert!(matches!(ft, FrameType::BiDir | FrameType::Inter));
        }

        // Scene change should force keyframe
        let ft = state.decide_type(250, true, true);
        assert_eq!(ft, FrameType::Key);
    }

    #[test]
    fn test_complexity_adjustment() {
        let mut controller = create_test_controller();

        // Add GOP history with known complexity
        let mut gop = GopStats::new(0, 0);
        gop.average_complexity = 1.0;
        controller.gop_history.push(gop);

        let low = controller.calculate_complexity_adjustment(0.5);
        let high = controller.calculate_complexity_adjustment(2.0);

        // High complexity should increase QP more
        assert!(high > low);
    }

    #[test]
    fn test_bit_reservoir() {
        let mut controller = create_test_controller();

        // Under-spend bits
        for i in 0..10 {
            let output = controller.get_rc(FrameType::Inter, 1.0);
            let mut stats = FrameStats::new(i, FrameType::Inter);
            stats.bits = output.target_bits / 2;
            stats.target_bits = output.target_bits;
            controller.update(&stats);
        }

        assert!(controller.bit_reservoir() > 0);
    }

    #[test]
    fn test_reset() {
        let mut controller = create_test_controller();

        for i in 0..10 {
            let mut stats = FrameStats::new(i, FrameType::Inter);
            stats.bits = 100_000;
            controller.update(&stats);
        }

        controller.reset();
        assert_eq!(controller.frame_count(), 0);
        assert_eq!(controller.bit_reservoir(), 0);
    }
}
