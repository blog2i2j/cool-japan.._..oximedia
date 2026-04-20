//! Unified Adaptive Bitrate (ABR) streaming controller.
//!
//! This module provides a comprehensive ABR implementation that works with
//! both HLS and DASH protocols. It includes sophisticated bandwidth estimation,
//! buffer-based adaptation, and configurable quality selection strategies.
//!
//! # Key Components
//!
//! - [`AdaptiveBitrateController`] - Main ABR controller trait
//! - [`AbrConfig`] - Configuration for ABR behavior
//! - [`AbrMode`] - Aggressive/conservative mode selection
//! - [`HybridAbrController`] - Advanced hybrid controller combining multiple algorithms
//! - [`BandwidthEstimator`] - Sophisticated bandwidth estimation
//! - `QualitySelector` - Quality selection with smooth transitions
//!
//! ## BBA-1: Buffer-Based Rate Adaptation
//!
//! The BBA-1 algorithm (Huang et al., 2014) selects the video bitrate variant
//! based on the client's current buffer fill level rather than measured throughput.
//! This avoids the rebuffering oscillation that bandwidth-based ABR exhibits on
//! live streams with variable throughput.  The implementation lives in
//! [`bba1::select_variant`].
//!
//! ### Regions
//!
//! ```text
//! Buffer level
//! ┌──────────────────────────────────────────────────────────────┐ ← capacity (B = 30s)
//! │                                                              │
//! │                    UPPER REGION                              │ → always highest variant
//! │                                                              │
//! ├────────────────────────────────────────────────────── r+c ───┤ (reservoir + cushion = 30s)
//! │                                                              │
//! │                    CUSHION REGION (c = 20s)                  │ → linear interpolation
//! │                                                              │
//! ├───────────────────────────────────────────────────── r ──────┤ (reservoir = 10s)
//! │                                                              │
//! │                    RESERVOIR REGION                          │ → always lowest variant
//! │                                                              │
//! └──────────────────────────────────────────────────────────────┘ ← 0 (empty)
//! ```
//!
//! Default parameters: `B = 30s`, `r = 10s`, `c = 20s`.
//! These can be customised via [`bba1::BbaParams`].
//!
//! # Example
//!
//! ```ignore
//! use oximedia_net::abr::{AbrConfig, AbrMode, HybridAbrController};
//!
//! // Create an aggressive ABR controller for fast quality switching
//! let config = AbrConfig::default().with_mode(AbrMode::Aggressive);
//! let mut abr = HybridAbrController::new(config);
//!
//! // Report download metrics
//! abr.report_segment_download(1_000_000, Duration::from_secs(1));
//! abr.report_buffer_level(Duration::from_secs(15));
//!
//! // Select quality
//! let decision = abr.select_quality(&quality_levels, current_index);
//! ```

#![allow(dead_code)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::similar_names)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::comparison_chain)]
#![allow(clippy::trivially_copy_pass_by_ref)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::should_implement_trait)]
#![allow(clippy::items_after_statements)]
#![allow(clippy::if_not_else)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Represents a quality level with bandwidth and optional metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QualityLevel {
    /// Unique index for this quality level.
    pub index: usize,
    /// Bandwidth requirement in bits per second.
    pub bandwidth: u64,
    /// Video resolution (width, height) if applicable.
    pub resolution: Option<(u32, u32)>,
    /// Codec string.
    pub codecs: Option<String>,
    /// Average bandwidth (may differ from peak).
    pub average_bandwidth: Option<u64>,
}

impl QualityLevel {
    /// Creates a new quality level.
    #[must_use]
    pub const fn new(index: usize, bandwidth: u64) -> Self {
        Self {
            index,
            bandwidth,
            resolution: None,
            codecs: None,
            average_bandwidth: None,
        }
    }

    /// Sets the resolution.
    #[must_use]
    pub const fn with_resolution(mut self, width: u32, height: u32) -> Self {
        self.resolution = Some((width, height));
        self
    }

    /// Sets the codecs.
    #[must_use]
    pub fn with_codecs(mut self, codecs: impl Into<String>) -> Self {
        self.codecs = Some(codecs.into());
        self
    }

    /// Sets the average bandwidth.
    #[must_use]
    pub const fn with_average_bandwidth(mut self, avg_bw: u64) -> Self {
        self.average_bandwidth = Some(avg_bw);
        self
    }

    /// Returns the effective bandwidth (average if available, otherwise peak).
    #[must_use]
    pub const fn effective_bandwidth(&self) -> u64 {
        if let Some(avg) = self.average_bandwidth {
            avg
        } else {
            self.bandwidth
        }
    }

    /// Returns the height in pixels if available.
    #[must_use]
    pub const fn height(&self) -> Option<u32> {
        if let Some((_, h)) = self.resolution {
            Some(h)
        } else {
            None
        }
    }

    /// Returns the width in pixels if available.
    #[must_use]
    pub const fn width(&self) -> Option<u32> {
        if let Some((w, _)) = self.resolution {
            Some(w)
        } else {
            None
        }
    }
}

/// ABR decision indicating which quality level to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbrDecision {
    /// Maintain current quality level.
    Maintain,
    /// Switch to a different quality level.
    SwitchTo(usize),
}

impl AbrDecision {
    /// Returns true if this is a switch decision.
    #[must_use]
    pub const fn is_switch(&self) -> bool {
        matches!(self, Self::SwitchTo(_))
    }

    /// Returns the target quality level index.
    #[must_use]
    pub const fn target_index(&self, current: usize) -> usize {
        match self {
            Self::Maintain => current,
            Self::SwitchTo(idx) => *idx,
        }
    }

    /// Returns the target level if switching.
    #[must_use]
    pub const fn switch_target(&self) -> Option<usize> {
        match self {
            Self::Maintain => None,
            Self::SwitchTo(idx) => Some(*idx),
        }
    }
}

/// ABR mode controlling aggressiveness of quality switching.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbrMode {
    /// Conservative mode: slow to switch up, fast to switch down.
    /// Prioritizes stability and avoids rebuffering.
    Conservative,
    /// Balanced mode: moderate switching behavior.
    Balanced,
    /// Aggressive mode: fast to switch up, slow to switch down.
    /// Prioritizes quality over stability.
    Aggressive,
}

impl Default for AbrMode {
    fn default() -> Self {
        Self::Balanced
    }
}

impl AbrMode {
    /// Returns the bandwidth safety factor for this mode.
    /// Conservative mode uses more headroom.
    #[must_use]
    pub const fn safety_factor(&self) -> f64 {
        match self {
            Self::Conservative => 0.75,
            Self::Balanced => 0.85,
            Self::Aggressive => 0.90,
        }
    }

    /// Returns the minimum buffer level required to switch up.
    #[must_use]
    pub const fn min_buffer_for_upswitch(&self) -> Duration {
        match self {
            Self::Conservative => Duration::from_secs(15),
            Self::Balanced => Duration::from_secs(10),
            Self::Aggressive => Duration::from_secs(6),
        }
    }

    /// Returns the critical buffer threshold for emergency downswitch.
    #[must_use]
    pub const fn critical_buffer(&self) -> Duration {
        match self {
            Self::Conservative => Duration::from_secs(8),
            Self::Balanced => Duration::from_secs(5),
            Self::Aggressive => Duration::from_secs(3),
        }
    }

    /// Returns the minimum interval between quality switches.
    #[must_use]
    pub const fn min_switch_interval(&self) -> Duration {
        match self {
            Self::Conservative => Duration::from_secs(12),
            Self::Balanced => Duration::from_secs(8),
            Self::Aggressive => Duration::from_secs(4),
        }
    }

    /// Returns the EMA alpha for bandwidth estimation.
    #[must_use]
    pub const fn ema_alpha(&self) -> f64 {
        match self {
            Self::Conservative => 0.5,
            Self::Balanced => 0.7,
            Self::Aggressive => 0.8,
        }
    }
}

/// Configuration for ABR controller.
#[derive(Debug, Clone)]
pub struct AbrConfig {
    /// ABR mode (conservative, balanced, aggressive).
    pub mode: AbrMode,
    /// Minimum quality level index (never go below this).
    pub min_quality: Option<usize>,
    /// Maximum quality level index (never go above this).
    pub max_quality: Option<usize>,
    /// Initial quality level index.
    pub initial_quality: Option<usize>,
    /// Maximum bandwidth in bits per second (0 = unlimited).
    pub max_bandwidth: u64,
    /// Enable fast startup (start low, ramp up quickly).
    pub fast_startup: bool,
    /// Target buffer level in seconds.
    pub target_buffer: Duration,
    /// Maximum buffer level in seconds.
    pub max_buffer: Duration,
    /// Bandwidth estimation window size (number of samples).
    pub estimation_window: usize,
    /// Sample time-to-live for bandwidth estimation.
    pub sample_ttl: Duration,
}

impl Default for AbrConfig {
    fn default() -> Self {
        Self {
            mode: AbrMode::Balanced,
            min_quality: None,
            max_quality: None,
            initial_quality: None,
            max_bandwidth: 0,
            fast_startup: true,
            target_buffer: Duration::from_secs(20),
            max_buffer: Duration::from_secs(40),
            estimation_window: 25,
            sample_ttl: Duration::from_secs(60),
        }
    }
}

impl AbrConfig {
    /// Creates a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the ABR mode.
    #[must_use]
    pub const fn with_mode(mut self, mode: AbrMode) -> Self {
        self.mode = mode;
        self
    }

    /// Sets the minimum quality constraint.
    #[must_use]
    pub const fn with_min_quality(mut self, min: usize) -> Self {
        self.min_quality = Some(min);
        self
    }

    /// Sets the maximum quality constraint.
    #[must_use]
    pub const fn with_max_quality(mut self, max: usize) -> Self {
        self.max_quality = Some(max);
        self
    }

    /// Sets the initial quality level.
    #[must_use]
    pub const fn with_initial_quality(mut self, initial: usize) -> Self {
        self.initial_quality = Some(initial);
        self
    }

    /// Sets the maximum bandwidth constraint.
    #[must_use]
    pub const fn with_max_bandwidth(mut self, max_bw: u64) -> Self {
        self.max_bandwidth = max_bw;
        self
    }

    /// Enables or disables fast startup.
    #[must_use]
    pub const fn with_fast_startup(mut self, enabled: bool) -> Self {
        self.fast_startup = enabled;
        self
    }

    /// Sets the target buffer level.
    #[must_use]
    pub const fn with_target_buffer(mut self, target: Duration) -> Self {
        self.target_buffer = target;
        self
    }

    /// Sets the maximum buffer level.
    #[must_use]
    pub const fn with_max_buffer(mut self, max: Duration) -> Self {
        self.max_buffer = max;
        self
    }
}

/// Adaptive bitrate controller trait.
pub trait AdaptiveBitrateController: Send + Sync {
    /// Selects the best quality level based on current network and buffer conditions.
    fn select_quality(&self, levels: &[QualityLevel], current_index: usize) -> AbrDecision;

    /// Reports a completed segment download.
    fn report_segment_download(&mut self, bytes: usize, duration: Duration);

    /// Reports current buffer level.
    fn report_buffer_level(&mut self, buffer_duration: Duration);

    /// Returns estimated throughput in bits per second.
    fn estimated_throughput(&self) -> f64;

    /// Returns current buffer level.
    fn current_buffer(&self) -> Duration;

    /// Resets the controller state.
    fn reset(&mut self);

    /// Returns the configuration.
    fn config(&self) -> &AbrConfig;
}

/// Bandwidth sample for estimation.
#[derive(Debug, Clone, Copy)]
struct BandwidthSample {
    /// Bytes downloaded.
    bytes: usize,
    /// Time taken to download.
    duration: Duration,
    /// Timestamp when sample was recorded.
    timestamp: Instant,
    /// Calculated throughput in bytes per second.
    throughput: f64,
}

impl BandwidthSample {
    fn new(bytes: usize, duration: Duration) -> Self {
        let throughput = if duration.as_secs_f64() > 0.0 {
            bytes as f64 / duration.as_secs_f64()
        } else {
            0.0
        };
        Self {
            bytes,
            duration,
            timestamp: Instant::now(),
            throughput,
        }
    }
}

/// Advanced bandwidth estimator using multiple techniques.
#[derive(Debug)]
pub struct BandwidthEstimator {
    /// Recent bandwidth samples.
    samples: VecDeque<BandwidthSample>,
    /// Maximum number of samples to keep.
    max_samples: usize,
    /// Sample time-to-live.
    sample_ttl: Duration,
    /// Exponential moving average estimate (bytes/sec).
    ema_estimate: f64,
    /// EMA alpha factor.
    alpha: f64,
    /// Harmonic mean estimate (bytes/sec).
    harmonic_mean: f64,
    /// Last update time.
    last_update: Option<Instant>,
}

impl BandwidthEstimator {
    /// Creates a new bandwidth estimator.
    #[must_use]
    pub fn new(max_samples: usize, sample_ttl: Duration, alpha: f64) -> Self {
        Self {
            samples: VecDeque::with_capacity(max_samples),
            max_samples,
            sample_ttl,
            ema_estimate: 0.0,
            alpha: alpha.clamp(0.0, 1.0),
            harmonic_mean: 0.0,
            last_update: None,
        }
    }

    /// Adds a new bandwidth sample.
    pub fn add_sample(&mut self, bytes: usize, duration: Duration) {
        let sample = BandwidthSample::new(bytes, duration);
        let now = Instant::now();

        // Remove expired samples
        while let Some(front) = self.samples.front() {
            if now.duration_since(front.timestamp) > self.sample_ttl {
                self.samples.pop_front();
            } else {
                break;
            }
        }

        // Remove oldest if at capacity
        if self.samples.len() >= self.max_samples {
            self.samples.pop_front();
        }

        // Update EMA
        if self.ema_estimate <= 0.0 {
            self.ema_estimate = sample.throughput;
        } else {
            self.ema_estimate =
                self.alpha * sample.throughput + (1.0 - self.alpha) * self.ema_estimate;
        }

        self.samples.push_back(sample);
        self.last_update = Some(now);
        self.update_harmonic_mean();
    }

    /// Updates the harmonic mean estimate.
    fn update_harmonic_mean(&mut self) {
        if self.samples.is_empty() {
            self.harmonic_mean = 0.0;
            return;
        }

        let sum_reciprocals: f64 = self
            .samples
            .iter()
            .map(|s| {
                if s.throughput > 0.0 {
                    1.0 / s.throughput
                } else {
                    0.0
                }
            })
            .sum();

        if sum_reciprocals > 0.0 {
            self.harmonic_mean = self.samples.len() as f64 / sum_reciprocals;
        } else {
            self.harmonic_mean = 0.0;
        }
    }

    /// Returns the estimated throughput using EMA (bytes/sec).
    #[must_use]
    pub fn estimate_ema(&self) -> f64 {
        self.ema_estimate
    }

    /// Returns the estimated throughput using harmonic mean (bytes/sec).
    #[must_use]
    pub fn estimate_harmonic(&self) -> f64 {
        self.harmonic_mean
    }

    /// Returns the minimum throughput from recent samples (bytes/sec).
    #[must_use]
    pub fn estimate_min(&self) -> f64 {
        self.samples
            .iter()
            .map(|s| s.throughput)
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0)
    }

    /// Returns the median throughput from recent samples (bytes/sec).
    #[must_use]
    pub fn estimate_median(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }

        let mut throughputs: Vec<f64> = self.samples.iter().map(|s| s.throughput).collect();
        throughputs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mid = throughputs.len() / 2;
        if throughputs.len() % 2 == 0 {
            (throughputs[mid - 1] + throughputs[mid]) / 2.0
        } else {
            throughputs[mid]
        }
    }

    /// Returns a conservative estimate combining multiple methods (bytes/sec).
    #[must_use]
    pub fn estimate_conservative(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }

        // Use minimum of EMA and harmonic mean for conservative estimate
        let ema = self.estimate_ema();
        let harmonic = self.estimate_harmonic();
        ema.min(harmonic)
    }

    /// Returns an aggressive estimate combining multiple methods (bytes/sec).
    #[must_use]
    pub fn estimate_aggressive(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }

        // Use EMA for aggressive estimate
        self.estimate_ema()
    }

    /// Returns a balanced estimate (bytes/sec).
    #[must_use]
    pub fn estimate_balanced(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }

        // Weighted average of EMA (70%) and harmonic mean (30%)
        let ema = self.estimate_ema();
        let harmonic = self.estimate_harmonic();
        0.7 * ema + 0.3 * harmonic
    }

    /// Returns the number of samples.
    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Returns true if we have enough samples for reliable estimation.
    #[must_use]
    pub fn is_reliable(&self) -> bool {
        self.samples.len() >= (self.max_samples / 4).max(3)
    }

    /// Resets the estimator.
    pub fn reset(&mut self) {
        self.samples.clear();
        self.ema_estimate = 0.0;
        self.harmonic_mean = 0.0;
        self.last_update = None;
    }
}

/// Quality selector with smooth transition logic.
#[derive(Debug)]
struct QualitySelector {
    /// Last switch time.
    last_switch: Option<Instant>,
    /// Switch history (recent quality indices).
    switch_history: VecDeque<(Instant, usize)>,
    /// Maximum history size.
    max_history: usize,
}

impl QualitySelector {
    fn new() -> Self {
        Self {
            last_switch: None,
            switch_history: VecDeque::new(),
            max_history: 10,
        }
    }

    /// Records a quality switch.
    fn record_switch(&mut self, quality_index: usize) {
        let now = Instant::now();
        self.last_switch = Some(now);
        self.switch_history.push_back((now, quality_index));

        // Keep history bounded
        if self.switch_history.len() > self.max_history {
            self.switch_history.pop_front();
        }
    }

    /// Returns true if enough time has passed since last switch.
    fn can_switch(&self, min_interval: Duration) -> bool {
        match self.last_switch {
            Some(last) => last.elapsed() >= min_interval,
            None => true,
        }
    }

    /// Returns true if switching too frequently (oscillating).
    fn is_oscillating(&self, window: Duration) -> bool {
        if self.switch_history.len() < 3 {
            return false;
        }

        let now = Instant::now();
        let recent_switches: Vec<_> = self
            .switch_history
            .iter()
            .filter(|(time, _)| now.duration_since(*time) < window)
            .collect();

        // Oscillating if we have 3+ switches in the window
        recent_switches.len() >= 3
    }

    /// Resets the selector.
    fn reset(&mut self) {
        self.last_switch = None;
        self.switch_history.clear();
    }
}

/// Hybrid ABR controller combining throughput and buffer-based algorithms.
#[derive(Debug)]
pub struct HybridAbrController {
    /// Configuration.
    config: AbrConfig,
    /// Bandwidth estimator.
    bandwidth_estimator: BandwidthEstimator,
    /// Current buffer level.
    buffer_level: Duration,
    /// Quality selector.
    quality_selector: QualitySelector,
    /// Startup phase active.
    in_startup: bool,
    /// Number of segments downloaded in startup.
    startup_segments: usize,
}

impl HybridAbrController {
    /// Creates a new hybrid ABR controller.
    #[must_use]
    pub fn new(config: AbrConfig) -> Self {
        let alpha = config.mode.ema_alpha();
        let bandwidth_estimator =
            BandwidthEstimator::new(config.estimation_window, config.sample_ttl, alpha);

        Self {
            config,
            bandwidth_estimator,
            buffer_level: Duration::ZERO,
            quality_selector: QualitySelector::new(),
            in_startup: true,
            startup_segments: 0,
        }
    }

    /// Selects quality during startup phase.
    fn select_quality_startup(&self, levels: &[QualityLevel], current_index: usize) -> AbrDecision {
        if levels.is_empty() {
            return AbrDecision::Maintain;
        }

        // In startup, be conservative unless we have good buffer
        let estimated_bps = self.bandwidth_estimator.estimate_conservative() * 8.0;
        if estimated_bps <= 0.0 {
            return AbrDecision::Maintain;
        }

        // Use extra conservative safety factor during startup
        let available_bw = estimated_bps * 0.7;

        // Find best quality that fits bandwidth
        let target = self.find_best_quality(levels, available_bw);

        // Constrain by config limits
        let target = self.apply_quality_constraints(target, levels.len());

        if target != current_index {
            AbrDecision::SwitchTo(target)
        } else {
            AbrDecision::Maintain
        }
    }

    /// Selects quality during steady state.
    fn select_quality_steady(&self, levels: &[QualityLevel], current_index: usize) -> AbrDecision {
        if levels.is_empty() {
            return AbrDecision::Maintain;
        }

        // Critical buffer situation - emergency downshift
        let critical_buffer = self.config.mode.critical_buffer();
        if self.buffer_level < critical_buffer && current_index > 0 {
            let min_quality = self.config.min_quality.unwrap_or(0);
            return AbrDecision::SwitchTo(min_quality);
        }

        // Get estimated bandwidth based on mode
        let estimated_bps = match self.config.mode {
            AbrMode::Conservative => self.bandwidth_estimator.estimate_conservative() * 8.0,
            AbrMode::Balanced => self.bandwidth_estimator.estimate_balanced() * 8.0,
            AbrMode::Aggressive => self.bandwidth_estimator.estimate_aggressive() * 8.0,
        };

        if estimated_bps <= 0.0 {
            return AbrDecision::Maintain;
        }

        // Apply safety factor
        let safety_factor = self.config.mode.safety_factor();
        let available_bw = estimated_bps * safety_factor;

        // Apply max bandwidth constraint
        let available_bw = if self.config.max_bandwidth > 0 {
            available_bw.min(self.config.max_bandwidth as f64)
        } else {
            available_bw
        };

        // Buffer-based adjustment
        let buffer_ratio =
            self.buffer_level.as_secs_f64() / self.config.target_buffer.as_secs_f64();
        let buffer_multiplier = if buffer_ratio > 1.5 {
            1.1 // High buffer, be more aggressive
        } else if buffer_ratio > 1.0 {
            1.0
        } else if buffer_ratio > 0.5 {
            0.95
        } else {
            0.85 // Low buffer, be conservative
        };

        let effective_bw = available_bw * buffer_multiplier;

        // Find best quality
        let mut target = self.find_best_quality(levels, effective_bw);

        // Apply quality constraints
        target = self.apply_quality_constraints(target, levels.len());

        // Check if we can switch
        let min_interval = self.config.mode.min_switch_interval();
        if !self.quality_selector.can_switch(min_interval) {
            return AbrDecision::Maintain;
        }

        // Check for oscillation
        if self
            .quality_selector
            .is_oscillating(Duration::from_secs(30))
        {
            return AbrDecision::Maintain;
        }

        // Prevent upswitch if buffer is too low
        if target > current_index {
            let min_buffer = self.config.mode.min_buffer_for_upswitch();
            if self.buffer_level < min_buffer {
                return AbrDecision::Maintain;
            }
        }

        if target != current_index {
            AbrDecision::SwitchTo(target)
        } else {
            AbrDecision::Maintain
        }
    }

    /// Finds the best quality level for the given available bandwidth.
    fn find_best_quality(&self, levels: &[QualityLevel], available_bw: f64) -> usize {
        let mut best_idx = 0;
        let mut best_bandwidth = 0u64;

        for (idx, level) in levels.iter().enumerate() {
            let level_bw = level.effective_bandwidth();
            if (level_bw as f64) <= available_bw && level_bw > best_bandwidth {
                best_idx = idx;
                best_bandwidth = level_bw;
            }
        }

        best_idx
    }

    /// Applies quality constraints from config.
    fn apply_quality_constraints(&self, quality: usize, num_levels: usize) -> usize {
        let mut constrained = quality;

        if let Some(min) = self.config.min_quality {
            constrained = constrained.max(min);
        }

        if let Some(max) = self.config.max_quality {
            constrained = constrained.min(max);
        }

        constrained.min(num_levels.saturating_sub(1))
    }

    /// Updates startup state.
    fn update_startup_state(&mut self) {
        if !self.in_startup {
            return;
        }

        self.startup_segments += 1;

        // Exit startup after 5 segments or if buffer is healthy
        if self.startup_segments >= 5 || self.buffer_level >= Duration::from_secs(10) {
            self.in_startup = false;
        }
    }
}

impl AdaptiveBitrateController for HybridAbrController {
    fn select_quality(&self, levels: &[QualityLevel], current_index: usize) -> AbrDecision {
        if levels.is_empty() {
            return AbrDecision::Maintain;
        }

        // Handle initial quality selection
        if self.bandwidth_estimator.sample_count() == 0 {
            if let Some(initial) = self.config.initial_quality {
                let initial = self.apply_quality_constraints(initial, levels.len());
                return AbrDecision::SwitchTo(initial);
            }
            // Default to lowest quality for first segment
            let min_quality = self.config.min_quality.unwrap_or(0);
            return AbrDecision::SwitchTo(min_quality);
        }

        // Not enough samples yet for reliable decision
        if !self.bandwidth_estimator.is_reliable() {
            return AbrDecision::Maintain;
        }

        // Use startup or steady-state logic
        if self.in_startup && self.config.fast_startup {
            self.select_quality_startup(levels, current_index)
        } else {
            self.select_quality_steady(levels, current_index)
        }
    }

    fn report_segment_download(&mut self, bytes: usize, duration: Duration) {
        self.bandwidth_estimator.add_sample(bytes, duration);
        self.update_startup_state();
    }

    fn report_buffer_level(&mut self, buffer_duration: Duration) {
        self.buffer_level = buffer_duration;
    }

    fn estimated_throughput(&self) -> f64 {
        match self.config.mode {
            AbrMode::Conservative => self.bandwidth_estimator.estimate_conservative() * 8.0,
            AbrMode::Balanced => self.bandwidth_estimator.estimate_balanced() * 8.0,
            AbrMode::Aggressive => self.bandwidth_estimator.estimate_aggressive() * 8.0,
        }
    }

    fn current_buffer(&self) -> Duration {
        self.buffer_level
    }

    fn reset(&mut self) {
        self.bandwidth_estimator.reset();
        self.buffer_level = Duration::ZERO;
        self.quality_selector.reset();
        self.in_startup = true;
        self.startup_segments = 0;
    }

    fn config(&self) -> &AbrConfig {
        &self.config
    }
}

/// Simple throughput-only ABR controller.
#[derive(Debug)]
pub struct SimpleThroughputAbr {
    /// Configuration.
    config: AbrConfig,
    /// Bandwidth estimator.
    bandwidth_estimator: BandwidthEstimator,
    /// Buffer level (not used but tracked).
    buffer_level: Duration,
    /// Last switch time.
    last_switch: Option<Instant>,
}

impl SimpleThroughputAbr {
    /// Creates a new simple throughput-based ABR controller.
    #[must_use]
    pub fn new(config: AbrConfig) -> Self {
        let alpha = config.mode.ema_alpha();
        let bandwidth_estimator =
            BandwidthEstimator::new(config.estimation_window, config.sample_ttl, alpha);

        Self {
            config,
            bandwidth_estimator,
            buffer_level: Duration::ZERO,
            last_switch: None,
        }
    }

    fn can_switch(&self) -> bool {
        match self.last_switch {
            Some(last) => last.elapsed() >= self.config.mode.min_switch_interval(),
            None => true,
        }
    }

    fn find_best_quality(&self, levels: &[QualityLevel], available_bw: f64) -> usize {
        let mut best_idx = 0;
        let mut best_bandwidth = 0u64;

        for (idx, level) in levels.iter().enumerate() {
            let level_bw = level.effective_bandwidth();
            if (level_bw as f64) <= available_bw && level_bw > best_bandwidth {
                best_idx = idx;
                best_bandwidth = level_bw;
            }
        }

        best_idx
    }

    fn apply_quality_constraints(&self, quality: usize, num_levels: usize) -> usize {
        let mut constrained = quality;

        if let Some(min) = self.config.min_quality {
            constrained = constrained.max(min);
        }

        if let Some(max) = self.config.max_quality {
            constrained = constrained.min(max);
        }

        constrained.min(num_levels.saturating_sub(1))
    }
}

impl AdaptiveBitrateController for SimpleThroughputAbr {
    fn select_quality(&self, levels: &[QualityLevel], current_index: usize) -> AbrDecision {
        if levels.is_empty() {
            return AbrDecision::Maintain;
        }

        if !self.bandwidth_estimator.is_reliable() {
            if let Some(initial) = self.config.initial_quality {
                let initial = self.apply_quality_constraints(initial, levels.len());
                return AbrDecision::SwitchTo(initial);
            }
            return AbrDecision::Maintain;
        }

        let estimated_bps = self.bandwidth_estimator.estimate_ema() * 8.0;
        if estimated_bps <= 0.0 {
            return AbrDecision::Maintain;
        }

        let available_bw = estimated_bps * self.config.mode.safety_factor();
        let target = self.find_best_quality(levels, available_bw);
        let target = self.apply_quality_constraints(target, levels.len());

        if target == current_index || !self.can_switch() {
            AbrDecision::Maintain
        } else {
            AbrDecision::SwitchTo(target)
        }
    }

    fn report_segment_download(&mut self, bytes: usize, duration: Duration) {
        self.bandwidth_estimator.add_sample(bytes, duration);
    }

    fn report_buffer_level(&mut self, buffer_duration: Duration) {
        self.buffer_level = buffer_duration;
    }

    fn estimated_throughput(&self) -> f64 {
        self.bandwidth_estimator.estimate_ema() * 8.0
    }

    fn current_buffer(&self) -> Duration {
        self.buffer_level
    }

    fn reset(&mut self) {
        self.bandwidth_estimator.reset();
        self.buffer_level = Duration::ZERO;
        self.last_switch = None;
    }

    fn config(&self) -> &AbrConfig {
        &self.config
    }
}

pub mod bba1;
pub mod bola;
pub mod dash_ctrl;
pub mod history;
pub mod mpc;
pub mod streaming;

pub use bba1::{select_variant as bba1_select_variant, BbaParams};
pub use bola::BolaBbrController;
pub use dash_ctrl::{DashAbrController, DashSegmentAvailability};
pub use history::{DownloadWindowStats, SegmentDownloadHistory, SegmentDownloadRecord};
pub use mpc::{MpcWeights, RobustMpcController};

/// Conversion helpers for HLS types.
pub mod hls {
    use super::QualityLevel;
    use crate::hls::VariantStream;

    /// Converts HLS variant streams to quality levels.
    #[must_use]
    pub fn variants_to_quality_levels(variants: &[VariantStream]) -> Vec<QualityLevel> {
        variants
            .iter()
            .enumerate()
            .map(|(idx, variant)| {
                let mut level = QualityLevel::new(idx, variant.stream_inf.bandwidth);
                if let Some((w, h)) = variant.stream_inf.resolution {
                    level = level.with_resolution(w, h);
                }
                if let Some(ref codecs) = variant.stream_inf.codecs {
                    level = level.with_codecs(codecs.clone());
                }
                if let Some(avg_bw) = variant.stream_inf.average_bandwidth {
                    level = level.with_average_bandwidth(avg_bw);
                }
                level
            })
            .collect()
    }
}

/// Conversion helpers for DASH types.
pub mod dash {
    use super::QualityLevel;
    use crate::dash::Representation;

    /// Converts DASH representations to quality levels.
    #[must_use]
    pub fn representations_to_quality_levels(
        representations: &[Representation],
    ) -> Vec<QualityLevel> {
        representations
            .iter()
            .enumerate()
            .map(|(idx, repr)| {
                let mut level = QualityLevel::new(idx, repr.bandwidth);
                if let Some((w, h)) = repr.resolution() {
                    level = level.with_resolution(w, h);
                }
                if let Some(ref codecs) = repr.codecs {
                    level = level.with_codecs(codecs.clone());
                }
                level
            })
            .collect()
    }
}
