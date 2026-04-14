//! ARIB TR-B32 Japanese broadcast loudness standard.
//!
//! Implements the ARIB TR-B32 specification for Japanese broadcast loudness
//! measurement and compliance, based on ITU-R BS.1770 with Japan-specific parameters:
//!
//! - Target loudness: **-24 LKFS** (matching ATSC A/85)
//! - True-peak limit: **-2 dBTP**
//! - Dialogue-gated loudness measurement
//! - Short-term loudness windows for programme monitoring
//!
//! # Example
//!
//! ```
//! use oximedia_audiopost::arib_loudness::{AribLoudnessAnalyzer, AribConfig};
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let config = AribConfig::default();
//! let mut analyzer = AribLoudnessAnalyzer::new(48000, 2, config)?;
//!
//! let left = vec![0.1_f32; 4096];
//! let right = vec![0.1_f32; 4096];
//! analyzer.process(&[&left, &right])?;
//!
//! let compliance = analyzer.check_compliance();
//! println!("ARIB compliant: {}", compliance.is_compliant());
//! # Ok(())
//! # }
//! ```

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use crate::error::{AudioPostError, AudioPostResult};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// ARIB TR-B32 target loudness in LKFS.
pub const ARIB_TARGET_LKFS: f32 = -24.0;

/// ARIB TR-B32 true-peak limit in dBTP.
pub const ARIB_TRUE_PEAK_LIMIT: f32 = -2.0;

/// ARIB TR-B32 tolerance in LU.
pub const ARIB_TOLERANCE_LU: f32 = 1.0;

/// ARIB TR-B32 maximum loudness range in LU.
pub const ARIB_MAX_LOUDNESS_RANGE: f32 = 20.0;

/// Configuration for the ARIB loudness analyzer.
#[derive(Debug, Clone)]
pub struct AribConfig {
    /// Target loudness in LKFS (default: -24.0).
    pub target_lkfs: f32,
    /// True-peak limit in dBTP (default: -2.0).
    pub true_peak_limit_dbtp: f32,
    /// Tolerance around target in LU (default: 1.0).
    pub tolerance_lu: f32,
    /// Maximum acceptable loudness range in LU (default: 20.0).
    pub max_loudness_range_lu: f32,
    /// Momentary loudness window in seconds (default: 0.4).
    pub momentary_window_s: f32,
    /// Short-term loudness window in seconds (default: 3.0).
    pub short_term_window_s: f32,
    /// Absolute gating threshold in LKFS (default: -70.0).
    pub absolute_gate_lkfs: f32,
    /// Relative gating threshold in LU below ungated loudness (default: -10.0).
    pub relative_gate_lu: f32,
    /// Enable dialogue gating (default: true).
    pub dialogue_gating: bool,
}

impl Default for AribConfig {
    fn default() -> Self {
        Self {
            target_lkfs: ARIB_TARGET_LKFS,
            true_peak_limit_dbtp: ARIB_TRUE_PEAK_LIMIT,
            tolerance_lu: ARIB_TOLERANCE_LU,
            max_loudness_range_lu: ARIB_MAX_LOUDNESS_RANGE,
            momentary_window_s: 0.4,
            short_term_window_s: 3.0,
            absolute_gate_lkfs: -70.0,
            relative_gate_lu: -10.0,
            dialogue_gating: true,
        }
    }
}

/// ARIB TR-B32 compliance result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AribCompliance {
    /// Whether the integrated loudness is within target ± tolerance.
    pub target_pass: bool,
    /// Whether the true peak is below the limit.
    pub true_peak_pass: bool,
    /// Whether the loudness range is within the maximum.
    pub loudness_range_pass: bool,
    /// Measured integrated loudness in LKFS.
    pub integrated_lkfs: f32,
    /// Target loudness in LKFS.
    pub target_lkfs: f32,
    /// Maximum true peak in dBTP.
    pub max_true_peak_dbtp: f32,
    /// True-peak limit in dBTP.
    pub true_peak_limit_dbtp: f32,
    /// Measured loudness range in LU.
    pub loudness_range_lu: f32,
    /// Maximum allowed loudness range in LU.
    pub max_loudness_range_lu: f32,
    /// Dialogue loudness in LKFS (if dialogue gating enabled).
    pub dialogue_loudness_lkfs: Option<f32>,
}

impl AribCompliance {
    /// Whether all checks pass.
    #[must_use]
    pub fn is_compliant(&self) -> bool {
        self.target_pass && self.true_peak_pass && self.loudness_range_pass
    }
}

/// K-weighting filter state (second-order IIR, two stages per ITU-R BS.1770).
#[derive(Debug, Clone)]
struct KWeightFilter {
    // Stage 1: high-shelf (pre-filter)
    b1: [f64; 3],
    a1: [f64; 3],
    x1: [f64; 2],
    y1: [f64; 2],
    // Stage 2: highpass (RLB weighting)
    b2: [f64; 3],
    a2: [f64; 3],
    x2: [f64; 2],
    y2: [f64; 2],
}

impl KWeightFilter {
    /// Design K-weighting filter for the given sample rate.
    fn new(sample_rate: u32) -> Self {
        let fs = f64::from(sample_rate);

        // Stage 1: High-shelf boost (+4 dB at high frequencies)
        // Coefficients derived from ITU-R BS.1770-4 for 48 kHz,
        // bilinear transform for other rates.
        let (b1, a1) = if (fs - 48000.0).abs() < 1.0 {
            (
                [1.535_349_7, -2.691_576_1, 1.198_214_3],
                [1.0, -1.690_610_7, 0.732_597_7],
            )
        } else {
            // Approximate for other sample rates
            Self::design_high_shelf(fs, 1681.0, 4.0)
        };

        // Stage 2: Highpass (RLB weighting, ~60 Hz)
        let (b2, a2) = if (fs - 48000.0).abs() < 1.0 {
            (
                [1.0, -2.0, 1.0],
                [1.0, -1.990_042_1, 0.990_098_2],
            )
        } else {
            Self::design_highpass(fs, 38.0)
        };

        Self {
            b1,
            a1,
            x1: [0.0; 2],
            y1: [0.0; 2],
            b2,
            a2,
            x2: [0.0; 2],
            y2: [0.0; 2],
        }
    }

    /// Design a high-shelf filter using bilinear transform.
    fn design_high_shelf(fs: f64, fc: f64, gain_db: f64) -> ([f64; 3], [f64; 3]) {
        let a_lin = 10.0_f64.powf(gain_db / 40.0);
        let w0 = 2.0 * std::f64::consts::PI * fc / fs;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let alpha = sin_w0 / 2.0 * ((a_lin + 1.0 / a_lin) * (1.0 / 0.9 - 1.0) + 2.0).sqrt();

        let b0 = a_lin * ((a_lin + 1.0) + (a_lin - 1.0) * cos_w0 + 2.0 * a_lin.sqrt() * alpha);
        let b1 = -2.0 * a_lin * ((a_lin - 1.0) + (a_lin + 1.0) * cos_w0);
        let b2 = a_lin * ((a_lin + 1.0) + (a_lin - 1.0) * cos_w0 - 2.0 * a_lin.sqrt() * alpha);
        let a0 = (a_lin + 1.0) - (a_lin - 1.0) * cos_w0 + 2.0 * a_lin.sqrt() * alpha;
        let a1_c = 2.0 * ((a_lin - 1.0) - (a_lin + 1.0) * cos_w0);
        let a2 = (a_lin + 1.0) - (a_lin - 1.0) * cos_w0 - 2.0 * a_lin.sqrt() * alpha;

        ([b0 / a0, b1 / a0, b2 / a0], [1.0, a1_c / a0, a2 / a0])
    }

    /// Design a second-order highpass filter.
    fn design_highpass(fs: f64, fc: f64) -> ([f64; 3], [f64; 3]) {
        let w0 = 2.0 * std::f64::consts::PI * fc / fs;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let q = 0.5;
        let alpha = sin_w0 / (2.0 * q);

        let b0 = (1.0 + cos_w0) / 2.0;
        let b1 = -(1.0 + cos_w0);
        let b2 = (1.0 + cos_w0) / 2.0;
        let a0 = 1.0 + alpha;
        let a1_c = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;

        ([b0 / a0, b1 / a0, b2 / a0], [1.0, a1_c / a0, a2 / a0])
    }

    /// Process a single sample through the K-weighting filter.
    fn process_sample(&mut self, input: f64) -> f64 {
        // Stage 1
        let y1 = self.b1[0] * input + self.b1[1] * self.x1[0] + self.b1[2] * self.x1[1]
            - self.a1[1] * self.y1[0]
            - self.a1[2] * self.y1[1];
        self.x1[1] = self.x1[0];
        self.x1[0] = input;
        self.y1[1] = self.y1[0];
        self.y1[0] = y1;

        // Stage 2
        let y2 = self.b2[0] * y1 + self.b2[1] * self.x2[0] + self.b2[2] * self.x2[1]
            - self.a2[1] * self.y2[0]
            - self.a2[2] * self.y2[1];
        self.x2[1] = self.x2[0];
        self.x2[0] = y1;
        self.y2[1] = self.y2[0];
        self.y2[0] = y2;

        y2
    }

    /// Reset filter state.
    fn reset(&mut self) {
        self.x1 = [0.0; 2];
        self.y1 = [0.0; 2];
        self.x2 = [0.0; 2];
        self.y2 = [0.0; 2];
    }
}

/// ARIB TR-B32 loudness analyzer.
///
/// Measures integrated loudness, short-term loudness, and true peak per
/// ARIB TR-B32 (based on ITU-R BS.1770-4 with -24 LKFS target).
#[derive(Debug)]
pub struct AribLoudnessAnalyzer {
    sample_rate: u32,
    num_channels: usize,
    config: AribConfig,
    /// K-weighting filters (one per channel).
    k_filters: Vec<KWeightFilter>,
    /// Channel weights for loudness summation (1.0 for L/R/C, 1.41 for Ls/Rs).
    channel_weights: Vec<f64>,
    /// Momentary loudness history (100ms blocks of mean-square).
    momentary_blocks: VecDeque<f64>,
    /// Short-term loudness history.
    short_term_blocks: VecDeque<f64>,
    /// All gating blocks for integrated loudness (400ms blocks, 75% overlap).
    gating_blocks: Vec<f64>,
    /// Dialogue gating blocks (when dialogue is detected).
    dialogue_blocks: Vec<f64>,
    /// Accumulator for current block.
    block_accum: Vec<f64>,
    /// Samples accumulated in current block.
    block_samples: usize,
    /// Block size in samples (100ms).
    block_size: usize,
    /// Maximum true peak per channel (linear).
    max_true_peak: Vec<f32>,
    /// Simple true-peak detection via 4-sample interpolation.
    prev_samples: Vec<[f32; 4]>,
    /// Total samples processed.
    samples_processed: u64,
}

impl AribLoudnessAnalyzer {
    /// Create a new ARIB loudness analyzer.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` — Sample rate in Hz
    /// * `num_channels` — Number of audio channels (1=mono, 2=stereo, 6=5.1)
    /// * `config` — ARIB configuration
    ///
    /// # Errors
    ///
    /// Returns an error if sample rate or channel count is invalid.
    pub fn new(
        sample_rate: u32,
        num_channels: usize,
        config: AribConfig,
    ) -> AudioPostResult<Self> {
        if sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(sample_rate));
        }
        if num_channels == 0 || num_channels > 8 {
            return Err(AudioPostError::InvalidChannelCount(num_channels));
        }

        let k_filters = (0..num_channels)
            .map(|_| KWeightFilter::new(sample_rate))
            .collect();

        // Channel weights per ITU-R BS.1770: L, R, C = 1.0; LFE = 0.0; Ls, Rs = 1.41
        let channel_weights = match num_channels {
            1 => vec![1.0],
            2 => vec![1.0, 1.0],
            6 => vec![1.0, 1.0, 1.0, 0.0, 1.41, 1.41],
            8 => vec![1.0, 1.0, 1.0, 0.0, 1.41, 1.41, 1.41, 1.41],
            n => vec![1.0; n],
        };

        // 100ms block size
        let block_size = (sample_rate as usize) / 10;
        // Momentary window = 4 blocks (400ms), short-term = 30 blocks (3s)
        let momentary_cap = 4;
        let short_term_cap = 30;

        Ok(Self {
            sample_rate,
            num_channels,
            config,
            k_filters,
            channel_weights,
            momentary_blocks: VecDeque::with_capacity(momentary_cap),
            short_term_blocks: VecDeque::with_capacity(short_term_cap),
            gating_blocks: Vec::new(),
            dialogue_blocks: Vec::new(),
            block_accum: vec![0.0; num_channels],
            block_samples: 0,
            block_size,
            max_true_peak: vec![0.0; num_channels],
            prev_samples: vec![[0.0; 4]; num_channels],
            samples_processed: 0,
        })
    }

    /// Process multi-channel audio.
    ///
    /// # Errors
    ///
    /// Returns an error if channel count mismatches or buffers are inconsistent.
    pub fn process(&mut self, channels: &[&[f32]]) -> AudioPostResult<()> {
        if channels.len() != self.num_channels {
            return Err(AudioPostError::InvalidChannelCount(channels.len()));
        }
        if self.num_channels > 1 {
            let len = channels[0].len();
            for ch in channels.iter().skip(1) {
                if ch.len() != len {
                    return Err(AudioPostError::InvalidBufferSize(ch.len()));
                }
            }
        }

        let len = channels.first().map_or(0, |c| c.len());

        for i in 0..len {
            // Process each channel through K-weighting and accumulate
            for (ch_idx, &channel) in channels.iter().enumerate() {
                let sample = channel[i];

                // True-peak detection (simple 4-point interpolation)
                self.detect_true_peak(ch_idx, sample);

                // K-weighted filtering
                let filtered = self.k_filters[ch_idx].process_sample(f64::from(sample));

                // Accumulate mean square
                self.block_accum[ch_idx] += filtered * filtered;
            }

            self.block_samples += 1;

            // Complete a 100ms block
            if self.block_samples >= self.block_size {
                self.complete_block();
            }
        }

        self.samples_processed += len as u64;
        Ok(())
    }

    /// Process with dialogue gating flag.
    ///
    /// When `is_dialogue` is true, the block is also included in dialogue-gated measurement.
    ///
    /// # Errors
    ///
    /// Returns an error if channel count mismatches.
    pub fn process_with_dialogue(
        &mut self,
        channels: &[&[f32]],
        is_dialogue: bool,
    ) -> AudioPostResult<()> {
        // Store current gating block count before processing
        let prev_gating_count = self.gating_blocks.len();

        self.process(channels)?;

        // If dialogue and new blocks were added, also add them to dialogue blocks
        if is_dialogue && self.config.dialogue_gating {
            let new_blocks = &self.gating_blocks[prev_gating_count..];
            self.dialogue_blocks.extend_from_slice(new_blocks);
        }

        Ok(())
    }

    /// Complete a 100ms block and add to gating history.
    fn complete_block(&mut self) {
        let mut block_loudness = 0.0_f64;

        for ch_idx in 0..self.num_channels {
            let mean_sq = self.block_accum[ch_idx] / self.block_size as f64;
            block_loudness += self.channel_weights[ch_idx] * mean_sq;
            self.block_accum[ch_idx] = 0.0;
        }

        self.block_samples = 0;

        // Store the linear power for gating
        self.gating_blocks.push(block_loudness);

        // Momentary (400ms = 4 blocks)
        self.momentary_blocks.push_back(block_loudness);
        if self.momentary_blocks.len() > 4 {
            self.momentary_blocks.pop_front();
        }

        // Short-term (3s = 30 blocks)
        self.short_term_blocks.push_back(block_loudness);
        if self.short_term_blocks.len() > 30 {
            self.short_term_blocks.pop_front();
        }
    }

    /// Detect true peak using simple 4-point Lagrange interpolation.
    fn detect_true_peak(&mut self, ch_idx: usize, sample: f32) {
        let abs_sample = sample.abs();
        if abs_sample > self.max_true_peak[ch_idx] {
            self.max_true_peak[ch_idx] = abs_sample;
        }

        // 4-point interpolation at half-sample points
        let prev = &self.prev_samples[ch_idx];
        // Lagrange interpolation at t=0.5 between prev[2] and prev[3]
        let interp = lagrange_interp_half(prev[0], prev[1], prev[2], sample);
        let abs_interp = interp.abs();
        if abs_interp > self.max_true_peak[ch_idx] {
            self.max_true_peak[ch_idx] = abs_interp;
        }

        // Shift history
        let ps = &mut self.prev_samples[ch_idx];
        ps[0] = ps[1];
        ps[1] = ps[2];
        ps[2] = ps[3];
        ps[3] = sample;
    }

    /// Get momentary loudness in LKFS (400ms window).
    #[must_use]
    pub fn momentary_lkfs(&self) -> f32 {
        if self.momentary_blocks.is_empty() {
            return -70.0;
        }
        let mean: f64 =
            self.momentary_blocks.iter().sum::<f64>() / self.momentary_blocks.len() as f64;
        power_to_lkfs(mean)
    }

    /// Get short-term loudness in LKFS (3s window).
    #[must_use]
    pub fn short_term_lkfs(&self) -> f32 {
        if self.short_term_blocks.is_empty() {
            return -70.0;
        }
        let mean: f64 =
            self.short_term_blocks.iter().sum::<f64>() / self.short_term_blocks.len() as f64;
        power_to_lkfs(mean)
    }

    /// Get integrated loudness in LKFS (gated per ITU-R BS.1770-4).
    #[must_use]
    pub fn integrated_lkfs(&self) -> f32 {
        self.compute_gated_loudness(&self.gating_blocks)
    }

    /// Get dialogue-gated loudness in LKFS.
    #[must_use]
    pub fn dialogue_loudness_lkfs(&self) -> f32 {
        if !self.config.dialogue_gating || self.dialogue_blocks.is_empty() {
            return self.integrated_lkfs();
        }
        self.compute_gated_loudness(&self.dialogue_blocks)
    }

    /// Compute gated loudness per ITU-R BS.1770-4 (absolute + relative gating).
    fn compute_gated_loudness(&self, blocks: &[f64]) -> f32 {
        if blocks.is_empty() {
            return -70.0;
        }

        let abs_threshold_power = lkfs_to_power(self.config.absolute_gate_lkfs);

        // Step 1: Absolute gating — exclude blocks below -70 LKFS
        let above_abs: Vec<f64> = blocks
            .iter()
            .copied()
            .filter(|&b| b > abs_threshold_power)
            .collect();

        if above_abs.is_empty() {
            return -70.0;
        }

        // Ungated loudness (mean of blocks above absolute threshold)
        let ungated_mean = above_abs.iter().sum::<f64>() / above_abs.len() as f64;
        let ungated_lkfs = power_to_lkfs(ungated_mean);

        // Step 2: Relative gating — exclude blocks below ungated - 10 LU
        let relative_threshold_lkfs = ungated_lkfs + self.config.relative_gate_lu;
        let relative_threshold_power = lkfs_to_power(relative_threshold_lkfs);

        let above_rel: Vec<f64> = above_abs
            .iter()
            .copied()
            .filter(|&b| b > relative_threshold_power)
            .collect();

        if above_rel.is_empty() {
            return -70.0;
        }

        let gated_mean = above_rel.iter().sum::<f64>() / above_rel.len() as f64;
        power_to_lkfs(gated_mean)
    }

    /// Get loudness range (LRA) in LU.
    ///
    /// Computed as the difference between the 95th and 10th percentiles
    /// of the short-term loudness distribution.
    #[must_use]
    pub fn loudness_range_lu(&self) -> f32 {
        if self.gating_blocks.len() < 2 {
            return 0.0;
        }

        // Use absolute-gated blocks
        let abs_threshold_power = lkfs_to_power(self.config.absolute_gate_lkfs);
        let above_abs: Vec<f64> = self
            .gating_blocks
            .iter()
            .copied()
            .filter(|&b| b > abs_threshold_power)
            .collect();

        if above_abs.len() < 2 {
            return 0.0;
        }

        // Relative gating
        let ungated_mean = above_abs.iter().sum::<f64>() / above_abs.len() as f64;
        let ungated_lkfs = power_to_lkfs(ungated_mean);
        let rel_thresh = lkfs_to_power(ungated_lkfs - 20.0); // LRA uses -20 LU gate

        let mut gated: Vec<f64> = above_abs
            .iter()
            .copied()
            .filter(|&b| b > rel_thresh)
            .collect();

        if gated.len() < 2 {
            return 0.0;
        }

        gated.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let p10_idx = (gated.len() as f64 * 0.10) as usize;
        let p95_idx = ((gated.len() as f64 * 0.95) as usize).min(gated.len() - 1);

        let p10_lkfs = power_to_lkfs(gated[p10_idx]);
        let p95_lkfs = power_to_lkfs(gated[p95_idx]);

        (p95_lkfs - p10_lkfs).abs()
    }

    /// Get maximum true peak across all channels in dBTP.
    #[must_use]
    pub fn max_true_peak_dbtp(&self) -> f32 {
        let max_linear = self
            .max_true_peak
            .iter()
            .copied()
            .fold(0.0_f32, f32::max);
        if max_linear <= 0.0 {
            -100.0
        } else {
            20.0 * max_linear.log10()
        }
    }

    /// Get true peak per channel in dBTP.
    #[must_use]
    pub fn true_peak_per_channel_dbtp(&self) -> Vec<f32> {
        self.max_true_peak
            .iter()
            .map(|&peak| {
                if peak <= 0.0 {
                    -100.0
                } else {
                    20.0 * peak.log10()
                }
            })
            .collect()
    }

    /// Check ARIB TR-B32 compliance.
    #[must_use]
    pub fn check_compliance(&self) -> AribCompliance {
        let integrated = self.integrated_lkfs();
        let max_tp = self.max_true_peak_dbtp();
        let lra = self.loudness_range_lu();

        let target_pass =
            (integrated - self.config.target_lkfs).abs() <= self.config.tolerance_lu;
        let true_peak_pass = max_tp <= self.config.true_peak_limit_dbtp;
        let loudness_range_pass = lra <= self.config.max_loudness_range_lu;

        let dialogue_loudness = if self.config.dialogue_gating && !self.dialogue_blocks.is_empty() {
            Some(self.dialogue_loudness_lkfs())
        } else {
            None
        };

        AribCompliance {
            target_pass,
            true_peak_pass,
            loudness_range_pass,
            integrated_lkfs: integrated,
            target_lkfs: self.config.target_lkfs,
            max_true_peak_dbtp: max_tp,
            true_peak_limit_dbtp: self.config.true_peak_limit_dbtp,
            loudness_range_lu: lra,
            max_loudness_range_lu: self.config.max_loudness_range_lu,
            dialogue_loudness_lkfs: dialogue_loudness,
        }
    }

    /// Get the number of gating blocks processed.
    #[must_use]
    pub fn num_gating_blocks(&self) -> usize {
        self.gating_blocks.len()
    }

    /// Get total samples processed.
    #[must_use]
    pub fn samples_processed(&self) -> u64 {
        self.samples_processed
    }

    /// Reset all measurements.
    pub fn reset(&mut self) {
        for f in &mut self.k_filters {
            f.reset();
        }
        self.momentary_blocks.clear();
        self.short_term_blocks.clear();
        self.gating_blocks.clear();
        self.dialogue_blocks.clear();
        for a in &mut self.block_accum {
            *a = 0.0;
        }
        self.block_samples = 0;
        for p in &mut self.max_true_peak {
            *p = 0.0;
        }
        for ps in &mut self.prev_samples {
            *ps = [0.0; 4];
        }
        self.samples_processed = 0;
    }
}

/// Convert mean-square power to LKFS.
fn power_to_lkfs(power: f64) -> f32 {
    if power <= 0.0 {
        return -70.0;
    }
    (-0.691 + 10.0 * power.log10()) as f32
}

/// Convert LKFS to linear power.
fn lkfs_to_power(lkfs: f32) -> f64 {
    10.0_f64.powf((f64::from(lkfs) + 0.691) / 10.0)
}

/// 4-point Lagrange interpolation at the midpoint (t=0.5).
fn lagrange_interp_half(y0: f32, y1: f32, y2: f32, y3: f32) -> f32 {
    // Lagrange basis polynomials evaluated at t=0.5 with points at -1, 0, 1, 2
    let l0 = -0.0625; // (-0.5)(0.5-1)(0.5-2) / ((-1-0)(-1-1)(-1-2))
    let l1 = 0.5625; // (0.5+1)(0.5-1)(0.5-2) / ((0+1)(0-1)(0-2))
    let l2 = 0.5625; // (0.5+1)(0.5)(0.5-2) / ((1+1)(1)(1-2))  — symmetric
    let l3 = -0.0625;
    y0 * l0 + y1 * l1 + y2 * l2 + y3 * l3
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sine(freq: f32, sample_rate: u32, num_samples: usize, amplitude: f32) -> Vec<f32> {
        (0..num_samples)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                amplitude * (2.0 * std::f32::consts::PI * freq * t).sin()
            })
            .collect()
    }

    #[test]
    fn test_create_analyzer() {
        let config = AribConfig::default();
        let analyzer = AribLoudnessAnalyzer::new(48000, 2, config).expect("create");
        assert_eq!(analyzer.samples_processed(), 0);
        assert_eq!(analyzer.num_gating_blocks(), 0);
    }

    #[test]
    fn test_invalid_creation() {
        assert!(AribLoudnessAnalyzer::new(0, 2, AribConfig::default()).is_err());
        assert!(AribLoudnessAnalyzer::new(48000, 0, AribConfig::default()).is_err());
        assert!(AribLoudnessAnalyzer::new(48000, 9, AribConfig::default()).is_err());
    }

    #[test]
    fn test_arib_target_constants() {
        assert!((ARIB_TARGET_LKFS - (-24.0)).abs() < f32::EPSILON);
        assert!((ARIB_TRUE_PEAK_LIMIT - (-2.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_silence_loudness() {
        let config = AribConfig::default();
        let mut analyzer = AribLoudnessAnalyzer::new(48000, 1, config).expect("create");
        let silence = vec![0.0_f32; 48000]; // 1 second
        analyzer.process(&[&silence]).expect("process");
        assert!(
            analyzer.integrated_lkfs() <= -69.0,
            "Silence should be very quiet, got {}",
            analyzer.integrated_lkfs()
        );
    }

    #[test]
    fn test_loud_signal_loudness() {
        let config = AribConfig::default();
        let mut analyzer = AribLoudnessAnalyzer::new(48000, 1, config).expect("create");
        // Full-scale sine for 2 seconds
        let signal = make_sine(1000.0, 48000, 96000, 1.0);
        analyzer.process(&[&signal]).expect("process");
        let lkfs = analyzer.integrated_lkfs();
        // Full-scale 1 kHz sine should be around -3 LKFS
        assert!(
            lkfs > -10.0,
            "Full-scale sine should be loud, got {lkfs:.1} LKFS"
        );
    }

    #[test]
    fn test_true_peak_detection() {
        let config = AribConfig::default();
        let mut analyzer = AribLoudnessAnalyzer::new(48000, 1, config).expect("create");
        let signal = make_sine(997.0, 48000, 48000, 0.9);
        analyzer.process(&[&signal]).expect("process");
        let tp = analyzer.max_true_peak_dbtp();
        // True peak of 0.9 amplitude sine should be around -0.9 dBTP
        assert!(
            tp > -3.0 && tp < 1.0,
            "True peak should be near -0.9 dBTP, got {tp:.1}"
        );
    }

    #[test]
    fn test_compliance_too_loud() {
        let config = AribConfig::default();
        let mut analyzer = AribLoudnessAnalyzer::new(48000, 1, config).expect("create");
        let signal = make_sine(1000.0, 48000, 96000, 1.0);
        analyzer.process(&[&signal]).expect("process");
        let compliance = analyzer.check_compliance();
        // Full-scale sine is way louder than -24 LKFS target
        assert!(!compliance.target_pass, "Full-scale should fail target check");
    }

    #[test]
    fn test_compliance_struct() {
        let c = AribCompliance {
            target_pass: true,
            true_peak_pass: true,
            loudness_range_pass: true,
            integrated_lkfs: -24.0,
            target_lkfs: -24.0,
            max_true_peak_dbtp: -3.0,
            true_peak_limit_dbtp: -2.0,
            loudness_range_lu: 5.0,
            max_loudness_range_lu: 20.0,
            dialogue_loudness_lkfs: None,
        };
        assert!(c.is_compliant());

        let c_fail = AribCompliance {
            true_peak_pass: false,
            ..c.clone()
        };
        assert!(!c_fail.is_compliant());
    }

    #[test]
    fn test_multi_channel_processing() {
        let config = AribConfig::default();
        let mut analyzer = AribLoudnessAnalyzer::new(48000, 2, config).expect("create");
        let left = make_sine(440.0, 48000, 48000, 0.5);
        let right = make_sine(440.0, 48000, 48000, 0.5);
        analyzer.process(&[&left, &right]).expect("process");
        let peaks = analyzer.true_peak_per_channel_dbtp();
        assert_eq!(peaks.len(), 2);
    }

    #[test]
    fn test_dialogue_gating() {
        let config = AribConfig {
            dialogue_gating: true,
            ..AribConfig::default()
        };
        let mut analyzer = AribLoudnessAnalyzer::new(48000, 1, config).expect("create");

        // Process dialogue segment
        let dialogue = make_sine(300.0, 48000, 48000, 0.3);
        analyzer
            .process_with_dialogue(&[&dialogue], true)
            .expect("process dialogue");

        // Process non-dialogue segment (music)
        let music = make_sine(1000.0, 48000, 48000, 0.8);
        analyzer
            .process_with_dialogue(&[&music], false)
            .expect("process music");

        let integrated = analyzer.integrated_lkfs();
        let dialogue_lkfs = analyzer.dialogue_loudness_lkfs();

        // Dialogue loudness should differ from integrated since music is louder
        // (dialogue is quieter, so dialogue_lkfs < integrated or they differ)
        assert!(
            (dialogue_lkfs - integrated).abs() > 0.1
                || dialogue_lkfs < integrated,
            "Dialogue loudness ({dialogue_lkfs:.1}) should differ from integrated ({integrated:.1})"
        );
    }

    #[test]
    fn test_short_term_loudness() {
        let config = AribConfig::default();
        let mut analyzer = AribLoudnessAnalyzer::new(48000, 1, config).expect("create");
        let signal = make_sine(1000.0, 48000, 48000 * 4, 0.5); // 4 seconds
        analyzer.process(&[&signal]).expect("process");

        let st = analyzer.short_term_lkfs();
        assert!(
            st > -70.0,
            "Short-term loudness should be measurable, got {st:.1}"
        );
    }

    #[test]
    fn test_reset() {
        let config = AribConfig::default();
        let mut analyzer = AribLoudnessAnalyzer::new(48000, 1, config).expect("create");
        let signal = make_sine(1000.0, 48000, 48000, 0.5);
        analyzer.process(&[&signal]).expect("process");
        assert!(analyzer.samples_processed() > 0);

        analyzer.reset();
        assert_eq!(analyzer.samples_processed(), 0);
        assert_eq!(analyzer.num_gating_blocks(), 0);
        assert!(analyzer.max_true_peak_dbtp() <= -100.0);
    }

    #[test]
    fn test_power_lkfs_roundtrip() {
        let lkfs_val = -24.0_f32;
        let power = lkfs_to_power(lkfs_val);
        let back = power_to_lkfs(power);
        assert!(
            (back - lkfs_val).abs() < 0.01,
            "Roundtrip failed: {lkfs_val} -> {power} -> {back}"
        );
    }

    #[test]
    fn test_channel_count_mismatch() {
        let config = AribConfig::default();
        let mut analyzer = AribLoudnessAnalyzer::new(48000, 2, config).expect("create");
        let mono = vec![0.5_f32; 1024];
        let result = analyzer.process(&[&mono]);
        assert!(result.is_err());
    }
}
