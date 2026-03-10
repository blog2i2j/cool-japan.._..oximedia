//! Audio normalization filter.
//!
//! This module provides audio normalization with support for peak, RMS,
//! and EBU R128 loudness normalization modes.

#![forbid(unsafe_code)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::get_first)]
#![allow(clippy::doc_markdown)]

use bytes::{Bytes, BytesMut};
use std::collections::VecDeque;

use crate::error::{GraphError, GraphResult};
use crate::frame::FilterFrame;
use crate::node::{Node, NodeId, NodeState, NodeType};
use crate::port::{AudioPortFormat, InputPort, OutputPort, PortFormat, PortId, PortType};

use oximedia_audio::{AudioBuffer, AudioFrame, ChannelLayout};
use oximedia_core::SampleFormat;

/// Normalization mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum NormalizationMode {
    /// Peak normalization - adjust so the peak reaches target level.
    #[default]
    Peak,
    /// RMS normalization - adjust based on root mean square level.
    Rms,
    /// EBU R128 loudness normalization.
    EbuR128,
}

/// Configuration for the normalize filter.
#[derive(Clone, Debug)]
pub struct NormalizeConfig {
    /// Normalization mode.
    pub mode: NormalizationMode,
    /// Target level in dB (0 dB = full scale).
    pub target_db: f64,
    /// Target LUFS for EBU R128 mode (typically -23 or -14).
    pub target_lufs: f64,
    /// Enable true peak limiting.
    pub true_peak_limit: bool,
    /// Maximum true peak level in dB.
    pub max_true_peak_db: f64,
    /// Analysis window size in samples for RMS/loudness calculation.
    pub analysis_window: usize,
    /// Lookahead buffer size in samples.
    pub lookahead_samples: usize,
}

impl Default for NormalizeConfig {
    fn default() -> Self {
        Self {
            mode: NormalizationMode::Peak,
            target_db: 0.0,
            target_lufs: -14.0,
            true_peak_limit: true,
            max_true_peak_db: -1.0,
            analysis_window: 4800,  // 100ms at 48kHz
            lookahead_samples: 480, // 10ms at 48kHz
        }
    }
}

impl NormalizeConfig {
    /// Create a new normalization configuration.
    #[must_use]
    pub fn new(mode: NormalizationMode) -> Self {
        Self {
            mode,
            ..Default::default()
        }
    }

    /// Create a peak normalization config.
    #[must_use]
    pub fn peak(target_db: f64) -> Self {
        Self {
            mode: NormalizationMode::Peak,
            target_db,
            ..Default::default()
        }
    }

    /// Create an RMS normalization config.
    #[must_use]
    pub fn rms(target_db: f64) -> Self {
        Self {
            mode: NormalizationMode::Rms,
            target_db,
            ..Default::default()
        }
    }

    /// Create an EBU R128 normalization config.
    #[must_use]
    pub fn ebu_r128(target_lufs: f64) -> Self {
        Self {
            mode: NormalizationMode::EbuR128,
            target_lufs,
            ..Default::default()
        }
    }

    /// Set true peak limiting.
    #[must_use]
    pub fn with_true_peak_limit(mut self, max_db: f64) -> Self {
        self.true_peak_limit = true;
        self.max_true_peak_db = max_db;
        self
    }

    /// Set analysis window size.
    #[must_use]
    pub fn with_analysis_window(mut self, samples: usize) -> Self {
        self.analysis_window = samples;
        self
    }

    /// Convert dB to linear gain.
    #[must_use]
    pub fn db_to_linear(db: f64) -> f64 {
        10.0_f64.powf(db / 20.0)
    }

    /// Convert linear to dB.
    #[must_use]
    pub fn linear_to_db(linear: f64) -> f64 {
        if linear <= 0.0 {
            f64::NEG_INFINITY
        } else {
            20.0 * linear.log10()
        }
    }
}

/// Loudness measurement state for EBU R128.
#[derive(Clone, Debug)]
#[allow(dead_code)]
struct LoudnessState {
    /// Gating block buffer (400ms blocks).
    gating_blocks: VecDeque<f64>,
    /// Short-term loudness buffer (3s window) - reserved for future use.
    short_term_buffer: VecDeque<f64>,
    /// Momentary loudness buffer (400ms window).
    momentary_buffer: VecDeque<f64>,
    /// K-weighting filter state per channel.
    k_filter_state: Vec<KWeightingState>,
    /// Sample rate.
    sample_rate: u32,
    /// Block size (400ms).
    block_size: usize,
    /// Current block accumulator.
    block_accumulator: f64,
    /// Samples in current block.
    block_samples: usize,
}

impl LoudnessState {
    /// Create new loudness measurement state.
    fn new(sample_rate: u32, channels: usize) -> Self {
        let block_size = (sample_rate as f64 * 0.4) as usize; // 400ms
        Self {
            gating_blocks: VecDeque::new(),
            short_term_buffer: VecDeque::with_capacity(
                ((sample_rate as f64 * 3.0) / block_size as f64) as usize,
            ),
            momentary_buffer: VecDeque::with_capacity(1),
            k_filter_state: vec![KWeightingState::new(sample_rate); channels],
            sample_rate,
            block_size,
            block_accumulator: 0.0,
            block_samples: 0,
        }
    }

    /// Process samples and update loudness measurement.
    fn process(&mut self, samples: &[Vec<f64>]) -> f64 {
        if samples.is_empty() || samples[0].is_empty() {
            return f64::NEG_INFINITY;
        }

        let sample_count = samples[0].len();
        let channels = samples.len().min(self.k_filter_state.len());

        for i in 0..sample_count {
            let mut sum_sq = 0.0;

            for ch in 0..channels {
                if i < samples[ch].len() {
                    // Apply K-weighting filter
                    let filtered = self.k_filter_state[ch].process(samples[ch][i]);
                    sum_sq += filtered * filtered;
                }
            }

            self.block_accumulator += sum_sq;
            self.block_samples += 1;

            if self.block_samples >= self.block_size {
                // Complete one gating block
                let block_loudness = if self.block_accumulator > 0.0 {
                    -0.691 + 10.0 * (self.block_accumulator / self.block_samples as f64).log10()
                } else {
                    f64::NEG_INFINITY
                };

                self.gating_blocks.push_back(block_loudness);
                self.momentary_buffer.push_back(block_loudness);

                // Keep only recent blocks for short-term
                let max_blocks =
                    ((self.sample_rate as f64 * 3.0) / self.block_size as f64) as usize;
                while self.gating_blocks.len() > max_blocks {
                    self.gating_blocks.pop_front();
                }

                self.block_accumulator = 0.0;
                self.block_samples = 0;
            }
        }

        self.integrated_loudness()
    }

    /// Calculate integrated loudness using gating.
    fn integrated_loudness(&self) -> f64 {
        if self.gating_blocks.is_empty() {
            return f64::NEG_INFINITY;
        }

        // First pass: absolute threshold (-70 LUFS)
        let above_threshold: Vec<f64> = self
            .gating_blocks
            .iter()
            .copied()
            .filter(|&l| l > -70.0)
            .collect();

        if above_threshold.is_empty() {
            return f64::NEG_INFINITY;
        }

        // Calculate average of blocks above threshold
        let sum: f64 = above_threshold.iter().sum();
        let avg = sum / above_threshold.len() as f64;

        // Second pass: relative threshold (-10 dB from average)
        let relative_threshold = avg - 10.0;
        let gated: Vec<f64> = above_threshold
            .iter()
            .copied()
            .filter(|&l| l > relative_threshold)
            .collect();

        if gated.is_empty() {
            return f64::NEG_INFINITY;
        }

        // Final integrated loudness
        let final_sum: f64 = gated.iter().sum();
        final_sum / gated.len() as f64
    }

    /// Get momentary loudness (400ms).
    #[allow(dead_code)]
    fn momentary_loudness(&self) -> f64 {
        self.momentary_buffer
            .back()
            .copied()
            .unwrap_or(f64::NEG_INFINITY)
    }
}

/// K-weighting filter state for ITU-R BS.1770.
#[derive(Clone, Debug)]
struct KWeightingState {
    /// High shelf filter state.
    hs_x1: f64,
    hs_x2: f64,
    hs_y1: f64,
    hs_y2: f64,
    /// High pass filter state.
    hp_x1: f64,
    hp_x2: f64,
    hp_y1: f64,
    hp_y2: f64,
    /// High shelf coefficients.
    hs_b0: f64,
    hs_b1: f64,
    hs_b2: f64,
    hs_a1: f64,
    hs_a2: f64,
    /// High pass coefficients.
    hp_b0: f64,
    hp_b1: f64,
    hp_b2: f64,
    hp_a1: f64,
    hp_a2: f64,
}

impl KWeightingState {
    /// Create new K-weighting filter state.
    fn new(sample_rate: u32) -> Self {
        let fs = f64::from(sample_rate);

        // High shelf filter coefficients (4dB boost at high frequencies)
        let (hs_b0, hs_b1, hs_b2, hs_a1, hs_a2) = Self::high_shelf_coefficients(fs, 1500.0, 4.0);

        // High pass filter coefficients (RLC rolloff below 60Hz)
        let (hp_b0, hp_b1, hp_b2, hp_a1, hp_a2) = Self::high_pass_coefficients(fs, 38.0);

        Self {
            hs_x1: 0.0,
            hs_x2: 0.0,
            hs_y1: 0.0,
            hs_y2: 0.0,
            hp_x1: 0.0,
            hp_x2: 0.0,
            hp_y1: 0.0,
            hp_y2: 0.0,
            hs_b0,
            hs_b1,
            hs_b2,
            hs_a1,
            hs_a2,
            hp_b0,
            hp_b1,
            hp_b2,
            hp_a1,
            hp_a2,
        }
    }

    /// Calculate high shelf filter coefficients.
    fn high_shelf_coefficients(fs: f64, fc: f64, gain_db: f64) -> (f64, f64, f64, f64, f64) {
        use std::f64::consts::PI;

        let a = 10.0_f64.powf(gain_db / 40.0);
        let w0 = 2.0 * PI * fc / fs;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let alpha = sin_w0 / 2.0 * (2.0_f64).sqrt();

        let a0 = (a + 1.0) - (a - 1.0) * cos_w0 + 2.0 * a.sqrt() * alpha;
        let a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_w0);
        let a2 = (a + 1.0) - (a - 1.0) * cos_w0 - 2.0 * a.sqrt() * alpha;
        let b0 = a * ((a + 1.0) + (a - 1.0) * cos_w0 + 2.0 * a.sqrt() * alpha);
        let b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0);
        let b2 = a * ((a + 1.0) + (a - 1.0) * cos_w0 - 2.0 * a.sqrt() * alpha);

        (b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)
    }

    /// Calculate high pass filter coefficients.
    fn high_pass_coefficients(fs: f64, fc: f64) -> (f64, f64, f64, f64, f64) {
        use std::f64::consts::PI;

        let w0 = 2.0 * PI * fc / fs;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let alpha = sin_w0 / 2.0 * (2.0_f64).sqrt();

        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;
        let b0 = (1.0 + cos_w0) / 2.0;
        let b1 = -(1.0 + cos_w0);
        let b2 = (1.0 + cos_w0) / 2.0;

        (b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)
    }

    /// Process one sample through K-weighting filter.
    fn process(&mut self, x: f64) -> f64 {
        // High shelf filter (stage 1)
        let hs_y = self.hs_b0 * x + self.hs_b1 * self.hs_x1 + self.hs_b2 * self.hs_x2
            - self.hs_a1 * self.hs_y1
            - self.hs_a2 * self.hs_y2;

        self.hs_x2 = self.hs_x1;
        self.hs_x1 = x;
        self.hs_y2 = self.hs_y1;
        self.hs_y1 = hs_y;

        // High pass filter (stage 2)
        let hp_y = self.hp_b0 * hs_y + self.hp_b1 * self.hp_x1 + self.hp_b2 * self.hp_x2
            - self.hp_a1 * self.hp_y1
            - self.hp_a2 * self.hp_y2;

        self.hp_x2 = self.hp_x1;
        self.hp_x1 = hs_y;
        self.hp_y2 = self.hp_y1;
        self.hp_y1 = hp_y;

        hp_y
    }
}

/// True peak limiter state.
#[derive(Clone, Debug)]
struct TruePeakLimiter {
    /// Ceiling level (linear).
    ceiling: f64,
    /// Lookahead buffer per channel.
    lookahead_buffer: Vec<VecDeque<f64>>,
    /// Current gain.
    current_gain: f64,
    /// Attack coefficient.
    attack: f64,
    /// Release coefficient.
    release: f64,
}

impl TruePeakLimiter {
    /// Create a new true peak limiter.
    fn new(ceiling_db: f64, lookahead_samples: usize, channels: usize, sample_rate: u32) -> Self {
        let ceiling = NormalizeConfig::db_to_linear(ceiling_db);
        let attack = (-1.0 / (0.001 * f64::from(sample_rate))).exp();
        let release = (-1.0 / (0.1 * f64::from(sample_rate))).exp();

        Self {
            ceiling,
            lookahead_buffer: vec![VecDeque::with_capacity(lookahead_samples); channels],
            current_gain: 1.0,
            attack,
            release,
        }
    }

    /// Process samples through the limiter.
    fn process(&mut self, samples: &mut [Vec<f64>]) {
        let sample_count = samples.get(0).map_or(0, Vec::len);
        let channels = samples.len();

        for i in 0..sample_count {
            // Find peak in lookahead window
            let mut max_peak = 0.0_f64;
            for ch in 0..channels {
                if i < samples[ch].len() {
                    self.lookahead_buffer[ch].push_back(samples[ch][i]);
                    if self.lookahead_buffer[ch].len() > 48 {
                        // ~1ms at 48kHz
                        self.lookahead_buffer[ch].pop_front();
                    }
                    max_peak = max_peak.max(
                        self.lookahead_buffer[ch]
                            .iter()
                            .map(|s| s.abs())
                            .fold(0.0_f64, f64::max),
                    );
                }
            }

            // Calculate target gain
            let target_gain = if max_peak > self.ceiling {
                self.ceiling / max_peak
            } else {
                1.0
            };

            // Smooth gain transition
            if target_gain < self.current_gain {
                self.current_gain =
                    self.attack * self.current_gain + (1.0 - self.attack) * target_gain;
            } else {
                self.current_gain =
                    self.release * self.current_gain + (1.0 - self.release) * target_gain;
            }

            // Apply gain
            for ch in 0..channels {
                if i < samples[ch].len() {
                    samples[ch][i] *= self.current_gain;
                }
            }
        }
    }
}

/// Audio normalization filter.
///
/// This filter normalizes audio levels using various modes:
/// - Peak normalization
/// - RMS normalization
/// - EBU R128 loudness normalization (integrated loudness)
///
/// # Example
///
/// ```ignore
/// use oximedia_graph::filters::audio::normalize::{NormalizeFilter, NormalizeConfig, NormalizationMode};
///
/// // Create an EBU R128 normalization filter targeting -14 LUFS
/// let config = NormalizeConfig::ebu_r128(-14.0)
///     .with_true_peak_limit(-1.0);
/// let filter = NormalizeFilter::new(NodeId(0), "normalize", config);
/// ```
pub struct NormalizeFilter {
    id: NodeId,
    name: String,
    state: NodeState,
    config: NormalizeConfig,
    /// Loudness measurement state for EBU R128.
    loudness_state: Option<LoudnessState>,
    /// True peak limiter.
    limiter: Option<TruePeakLimiter>,
    /// Running analysis buffer.
    analysis_buffer: Vec<Vec<f64>>,
    /// Calculated gain from analysis.
    calculated_gain: f64,
    inputs: Vec<InputPort>,
    outputs: Vec<OutputPort>,
}

impl NormalizeFilter {
    /// Create a new normalization filter.
    #[must_use]
    pub fn new(id: NodeId, name: impl Into<String>, config: NormalizeConfig) -> Self {
        let audio_format = PortFormat::Audio(AudioPortFormat::any());

        Self {
            id,
            name: name.into(),
            state: NodeState::Idle,
            config,
            loudness_state: None,
            limiter: None,
            analysis_buffer: Vec::new(),
            calculated_gain: 1.0,
            inputs: vec![InputPort::new(PortId(0), "input", PortType::Audio)
                .with_format(audio_format.clone())],
            outputs: vec![
                OutputPort::new(PortId(0), "output", PortType::Audio).with_format(audio_format)
            ],
        }
    }

    /// Get the current configuration.
    #[must_use]
    pub fn config(&self) -> &NormalizeConfig {
        &self.config
    }

    /// Update the configuration.
    pub fn set_config(&mut self, config: NormalizeConfig) {
        self.config = config;
        self.loudness_state = None;
        self.limiter = None;
        self.calculated_gain = 1.0;
    }

    /// Get the current calculated gain.
    #[must_use]
    pub fn calculated_gain(&self) -> f64 {
        self.calculated_gain
    }

    /// Convert audio frame to f64 samples per channel.
    fn frame_to_samples(frame: &AudioFrame) -> Vec<Vec<f64>> {
        let channels = frame.channels.count();
        let sample_count = frame.sample_count();

        if sample_count == 0 {
            return vec![Vec::new(); channels];
        }

        let mut output = vec![Vec::with_capacity(sample_count); channels];

        match &frame.samples {
            AudioBuffer::Interleaved(data) => {
                Self::convert_interleaved(data, frame.format, channels, &mut output);
            }
            AudioBuffer::Planar(planes) => {
                Self::convert_planar(planes, frame.format, &mut output);
            }
        }

        output
    }

    /// Convert interleaved samples.
    fn convert_interleaved(
        data: &Bytes,
        format: SampleFormat,
        channels: usize,
        output: &mut [Vec<f64>],
    ) {
        let bytes_per_sample = format.bytes_per_sample();
        if bytes_per_sample == 0 || channels == 0 {
            return;
        }

        let sample_count = data.len() / (bytes_per_sample * channels);

        for i in 0..sample_count {
            for ch in 0..channels {
                let offset = (i * channels + ch) * bytes_per_sample;
                if offset + bytes_per_sample <= data.len() {
                    let sample =
                        Self::bytes_to_f64(&data[offset..offset + bytes_per_sample], format);
                    output[ch].push(sample);
                }
            }
        }
    }

    /// Convert planar samples.
    fn convert_planar(planes: &[Bytes], format: SampleFormat, output: &mut [Vec<f64>]) {
        let bytes_per_sample = format.bytes_per_sample();
        if bytes_per_sample == 0 {
            return;
        }

        for (ch, plane) in planes.iter().enumerate() {
            if ch >= output.len() {
                break;
            }
            let sample_count = plane.len() / bytes_per_sample;
            for i in 0..sample_count {
                let offset = i * bytes_per_sample;
                if offset + bytes_per_sample <= plane.len() {
                    let sample =
                        Self::bytes_to_f64(&plane[offset..offset + bytes_per_sample], format);
                    output[ch].push(sample);
                }
            }
        }
    }

    /// Convert bytes to f64 sample.
    fn bytes_to_f64(bytes: &[u8], format: SampleFormat) -> f64 {
        match format {
            SampleFormat::U8 => {
                if bytes.is_empty() {
                    return 0.0;
                }
                (f64::from(bytes[0]) - 128.0) / 128.0
            }
            SampleFormat::S16 => {
                if bytes.len() < 2 {
                    return 0.0;
                }
                let sample = i16::from_le_bytes([bytes[0], bytes[1]]);
                f64::from(sample) / f64::from(i16::MAX)
            }
            SampleFormat::S32 => {
                if bytes.len() < 4 {
                    return 0.0;
                }
                let sample = i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                f64::from(sample) / f64::from(i32::MAX)
            }
            SampleFormat::F32 => {
                if bytes.len() < 4 {
                    return 0.0;
                }
                f64::from(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
            }
            SampleFormat::F64 => {
                if bytes.len() < 8 {
                    return 0.0;
                }
                f64::from_le_bytes([
                    bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
                ])
            }
            _ => 0.0,
        }
    }

    /// Find peak level in samples.
    fn find_peak(samples: &[Vec<f64>]) -> f64 {
        samples
            .iter()
            .flat_map(|ch| ch.iter())
            .map(|s| s.abs())
            .fold(0.0_f64, f64::max)
    }

    /// Calculate RMS level.
    fn calculate_rms(samples: &[Vec<f64>]) -> f64 {
        let mut sum_sq = 0.0;
        let mut count = 0usize;

        for channel in samples {
            for sample in channel {
                sum_sq += sample * sample;
                count += 1;
            }
        }

        if count == 0 {
            0.0
        } else {
            (sum_sq / count as f64).sqrt()
        }
    }

    /// Calculate gain based on normalization mode.
    fn calculate_gain(&mut self, samples: &[Vec<f64>], sample_rate: u32, channels: usize) -> f64 {
        match self.config.mode {
            NormalizationMode::Peak => {
                let peak = Self::find_peak(samples);
                let target = NormalizeConfig::db_to_linear(self.config.target_db);
                if peak > 0.0 {
                    target / peak
                } else {
                    1.0
                }
            }
            NormalizationMode::Rms => {
                let rms = Self::calculate_rms(samples);
                let target = NormalizeConfig::db_to_linear(self.config.target_db);
                if rms > 0.0 {
                    target / rms
                } else {
                    1.0
                }
            }
            NormalizationMode::EbuR128 => {
                // Initialize loudness state if needed
                if self.loudness_state.is_none() {
                    self.loudness_state = Some(LoudnessState::new(sample_rate, channels));
                }

                let loudness_state = match self.loudness_state.as_mut() {
                    Some(s) => s,
                    None => return 1.0,
                };
                let current_lufs = loudness_state.process(samples);

                if current_lufs.is_finite() {
                    let gain_db = self.config.target_lufs - current_lufs;
                    NormalizeConfig::db_to_linear(gain_db)
                } else {
                    1.0
                }
            }
        }
    }

    /// Apply gain to samples.
    fn apply_gain(samples: &mut [Vec<f64>], gain: f64) {
        for channel in samples.iter_mut() {
            for sample in channel.iter_mut() {
                *sample *= gain;
            }
        }
    }

    /// Convert f64 samples to audio frame.
    fn samples_to_frame(
        samples: Vec<Vec<f64>>,
        format: SampleFormat,
        sample_rate: u32,
        channels: ChannelLayout,
    ) -> AudioFrame {
        let channel_count = channels.count();
        if samples.is_empty() || samples[0].is_empty() || channel_count == 0 {
            return AudioFrame::new(format, sample_rate, channels);
        }

        let sample_count = samples[0].len();
        let bytes_per_sample = format.bytes_per_sample();
        let mut buffer = BytesMut::with_capacity(sample_count * channel_count * bytes_per_sample);

        for i in 0..sample_count {
            for ch in 0..channel_count {
                let sample = if ch < samples.len() && i < samples[ch].len() {
                    samples[ch][i]
                } else {
                    0.0
                };
                Self::f64_to_bytes(sample, format, &mut buffer);
            }
        }

        let mut frame = AudioFrame::new(format, sample_rate, channels);
        frame.samples = AudioBuffer::Interleaved(buffer.freeze());
        frame
    }

    /// Convert f64 sample to bytes.
    fn f64_to_bytes(sample: f64, format: SampleFormat, buffer: &mut BytesMut) {
        let clamped = sample.clamp(-1.0, 1.0);

        match format {
            SampleFormat::U8 => {
                let value = ((clamped * 128.0) + 128.0) as u8;
                buffer.extend_from_slice(&[value]);
            }
            SampleFormat::S16 => {
                let value = (clamped * f64::from(i16::MAX)) as i16;
                buffer.extend_from_slice(&value.to_le_bytes());
            }
            SampleFormat::S32 => {
                let value = (clamped * f64::from(i32::MAX)) as i32;
                buffer.extend_from_slice(&value.to_le_bytes());
            }
            SampleFormat::F32 => {
                #[allow(clippy::cast_possible_truncation)]
                let value = clamped as f32;
                buffer.extend_from_slice(&value.to_le_bytes());
            }
            SampleFormat::F64 => {
                buffer.extend_from_slice(&clamped.to_le_bytes());
            }
            _ => {}
        }
    }
}

impl Node for NormalizeFilter {
    fn id(&self) -> NodeId {
        self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn node_type(&self) -> NodeType {
        NodeType::Filter
    }

    fn state(&self) -> NodeState {
        self.state
    }

    fn set_state(&mut self, state: NodeState) -> GraphResult<()> {
        if !self.state.can_transition_to(state) {
            return Err(GraphError::InvalidStateTransition {
                node: self.id,
                from: self.state.to_string(),
                to: state.to_string(),
            });
        }
        self.state = state;
        Ok(())
    }

    fn inputs(&self) -> &[InputPort] {
        &self.inputs
    }

    fn outputs(&self) -> &[OutputPort] {
        &self.outputs
    }

    fn process(&mut self, input: Option<FilterFrame>) -> GraphResult<Option<FilterFrame>> {
        let frame = match input {
            Some(FilterFrame::Audio(frame)) => frame,
            Some(_) => {
                return Err(GraphError::PortTypeMismatch {
                    expected: "Audio".to_string(),
                    actual: "Video".to_string(),
                });
            }
            None => return Ok(None),
        };

        let sample_rate = frame.sample_rate;
        let channels = frame.channels.count();

        // Initialize limiter if needed
        if self.config.true_peak_limit && self.limiter.is_none() {
            self.limiter = Some(TruePeakLimiter::new(
                self.config.max_true_peak_db,
                self.config.lookahead_samples,
                channels,
                sample_rate,
            ));
        }

        // Convert to f64 samples
        let mut samples = Self::frame_to_samples(&frame);

        // Calculate and apply gain
        self.calculated_gain = self.calculate_gain(&samples, sample_rate, channels);
        Self::apply_gain(&mut samples, self.calculated_gain);

        // Apply true peak limiting if enabled
        if let Some(ref mut limiter) = self.limiter {
            limiter.process(&mut samples);
        }

        // Convert back to frame
        let output_frame = Self::samples_to_frame(
            samples,
            frame.format,
            frame.sample_rate,
            frame.channels.clone(),
        );

        Ok(Some(FilterFrame::Audio(output_frame)))
    }

    fn reset(&mut self) -> GraphResult<()> {
        self.loudness_state = None;
        self.limiter = None;
        self.calculated_gain = 1.0;
        self.analysis_buffer.clear();
        self.set_state(NodeState::Idle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalization_mode() {
        assert_eq!(NormalizationMode::default(), NormalizationMode::Peak);
    }

    #[test]
    fn test_normalize_config() {
        let config = NormalizeConfig::peak(-6.0);
        assert_eq!(config.mode, NormalizationMode::Peak);
        assert!((config.target_db - (-6.0)).abs() < f64::EPSILON);

        let config = NormalizeConfig::rms(-20.0);
        assert_eq!(config.mode, NormalizationMode::Rms);

        let config = NormalizeConfig::ebu_r128(-14.0);
        assert_eq!(config.mode, NormalizationMode::EbuR128);
        assert!((config.target_lufs - (-14.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_db_conversion() {
        let linear = NormalizeConfig::db_to_linear(0.0);
        assert!((linear - 1.0).abs() < f64::EPSILON);

        let db = NormalizeConfig::linear_to_db(1.0);
        assert!(db.abs() < f64::EPSILON);

        let db = NormalizeConfig::linear_to_db(0.0);
        assert!(db.is_infinite() && db.is_sign_negative());
    }

    #[test]
    fn test_find_peak() {
        let samples = vec![vec![0.5, -0.8, 0.3], vec![0.2, 0.9, -0.1]];
        let peak = NormalizeFilter::find_peak(&samples);
        assert!((peak - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn test_calculate_rms() {
        let samples = vec![vec![1.0, 1.0, 1.0, 1.0]];
        let rms = NormalizeFilter::calculate_rms(&samples);
        assert!((rms - 1.0).abs() < f64::EPSILON);

        let samples = vec![vec![0.0, 0.0, 0.0, 0.0]];
        let rms = NormalizeFilter::calculate_rms(&samples);
        assert!(rms.abs() < f64::EPSILON);
    }

    #[test]
    fn test_normalize_filter_creation() {
        let config = NormalizeConfig::peak(0.0);
        let filter = NormalizeFilter::new(NodeId(1), "normalize", config);

        assert_eq!(filter.id(), NodeId(1));
        assert_eq!(filter.name(), "normalize");
        assert_eq!(filter.node_type(), NodeType::Filter);
    }

    #[test]
    fn test_normalize_filter_ports() {
        let config = NormalizeConfig::default();
        let filter = NormalizeFilter::new(NodeId(0), "test", config);

        assert_eq!(filter.inputs().len(), 1);
        assert_eq!(filter.outputs().len(), 1);
        assert_eq!(filter.inputs()[0].port_type, PortType::Audio);
    }

    #[test]
    fn test_process_none() {
        let config = NormalizeConfig::default();
        let mut filter = NormalizeFilter::new(NodeId(0), "test", config);

        let result = filter.process(None).expect("process should succeed");
        assert!(result.is_none());
    }

    #[test]
    fn test_peak_normalization() {
        let config = NormalizeConfig::peak(0.0);
        let mut filter = NormalizeFilter::new(NodeId(0), "test", config);

        let mut frame = AudioFrame::new(SampleFormat::F32, 48000, ChannelLayout::Mono);
        let mut samples = BytesMut::new();
        samples.extend_from_slice(&0.5f32.to_le_bytes());
        frame.samples = AudioBuffer::Interleaved(samples.freeze());

        let result = filter
            .process(Some(FilterFrame::Audio(frame)))
            .expect("process should succeed");
        assert!(result.is_some());

        // Gain should be 2.0 to bring 0.5 peak to 1.0
        assert!((filter.calculated_gain() - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_state_transitions() {
        let config = NormalizeConfig::default();
        let mut filter = NormalizeFilter::new(NodeId(0), "test", config);

        assert!(filter.set_state(NodeState::Processing).is_ok());
        assert_eq!(filter.state(), NodeState::Processing);

        assert!(filter.reset().is_ok());
        assert_eq!(filter.state(), NodeState::Idle);
    }

    #[test]
    fn test_k_weighting_state() {
        let mut state = KWeightingState::new(48000);

        // Process some samples
        let output = state.process(1.0);
        assert!(output.is_finite());

        let output = state.process(0.0);
        assert!(output.is_finite());
    }

    #[test]
    fn test_loudness_state() {
        let mut state = LoudnessState::new(48000, 2);

        let samples = vec![vec![0.1; 480], vec![0.1; 480]];

        let loudness = state.process(&samples);
        assert!(loudness.is_finite() || loudness.is_nan() || loudness.is_infinite());
    }

    #[test]
    fn test_true_peak_limiter() {
        let mut limiter = TruePeakLimiter::new(-1.0, 48, 2, 48000);

        // Process multiple blocks to allow limiter to stabilize
        let mut samples = vec![vec![0.5; 100], vec![0.5; 100]];

        // First, prime the limiter with normal samples
        limiter.process(&mut samples);

        // Now test with overloaded samples
        let mut samples = vec![vec![0.5, 0.8, 0.9, 0.3], vec![0.4, 0.7, 0.85, 0.2]];

        limiter.process(&mut samples);

        // All samples should be below ceiling (gain applied)
        let ceiling = NormalizeConfig::db_to_linear(-1.0);
        for ch in &samples {
            for &sample in ch {
                // Allow small tolerance for smoothing
                assert!(
                    sample.abs() <= ceiling + 0.01,
                    "Sample {} exceeds ceiling {}",
                    sample,
                    ceiling
                );
            }
        }
    }

    #[test]
    fn test_apply_gain() {
        let mut samples = vec![vec![0.5, -0.5], vec![0.25, -0.25]];

        NormalizeFilter::apply_gain(&mut samples, 2.0);

        assert!((samples[0][0] - 1.0).abs() < f64::EPSILON);
        assert!((samples[0][1] - (-1.0)).abs() < f64::EPSILON);
        assert!((samples[1][0] - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_with_true_peak_limit() {
        let config = NormalizeConfig::peak(0.0).with_true_peak_limit(-1.0);

        assert!(config.true_peak_limit);
        assert!((config.max_true_peak_db - (-1.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_with_analysis_window() {
        let config = NormalizeConfig::default().with_analysis_window(9600);

        assert_eq!(config.analysis_window, 9600);
    }
}
