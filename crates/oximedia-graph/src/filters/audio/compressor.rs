//! Audio dynamics compressor filter.
//!
//! This module provides a dynamics compressor with configurable threshold,
//! ratio, attack, release, and knee settings.

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

/// Knee type for compression curve.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum KneeType {
    /// Hard knee - abrupt transition at threshold.
    #[default]
    Hard,
    /// Soft knee - gradual transition around threshold.
    Soft,
}

/// Configuration for the compressor.
#[derive(Clone, Debug)]
pub struct CompressorConfig {
    /// Threshold in dB.
    pub threshold_db: f64,
    /// Compression ratio (e.g., 4.0 for 4:1).
    pub ratio: f64,
    /// Attack time in milliseconds.
    pub attack_ms: f64,
    /// Release time in milliseconds.
    pub release_ms: f64,
    /// Knee type.
    pub knee_type: KneeType,
    /// Soft knee width in dB.
    pub knee_width_db: f64,
    /// Makeup gain in dB.
    pub makeup_gain_db: f64,
    /// Auto makeup gain enabled.
    pub auto_makeup: bool,
    /// Lookahead time in milliseconds.
    pub lookahead_ms: f64,
    /// Sidechain filter cutoff (Hz) - 0 disables filter.
    pub sidechain_hpf_hz: f64,
}

impl Default for CompressorConfig {
    fn default() -> Self {
        Self {
            threshold_db: -20.0,
            ratio: 4.0,
            attack_ms: 10.0,
            release_ms: 100.0,
            knee_type: KneeType::Hard,
            knee_width_db: 6.0,
            makeup_gain_db: 0.0,
            auto_makeup: false,
            lookahead_ms: 0.0,
            sidechain_hpf_hz: 0.0,
        }
    }
}

impl CompressorConfig {
    /// Create a new compressor configuration.
    #[must_use]
    pub fn new(threshold_db: f64, ratio: f64) -> Self {
        Self {
            threshold_db,
            ratio,
            ..Default::default()
        }
    }

    /// Set attack and release times.
    #[must_use]
    pub fn with_timing(mut self, attack_ms: f64, release_ms: f64) -> Self {
        self.attack_ms = attack_ms;
        self.release_ms = release_ms;
        self
    }

    /// Set soft knee.
    #[must_use]
    pub fn with_soft_knee(mut self, width_db: f64) -> Self {
        self.knee_type = KneeType::Soft;
        self.knee_width_db = width_db;
        self
    }

    /// Set makeup gain.
    #[must_use]
    pub fn with_makeup_gain(mut self, gain_db: f64) -> Self {
        self.makeup_gain_db = gain_db;
        self.auto_makeup = false;
        self
    }

    /// Enable auto makeup gain.
    #[must_use]
    pub fn with_auto_makeup(mut self) -> Self {
        self.auto_makeup = true;
        self
    }

    /// Set lookahead time.
    #[must_use]
    pub fn with_lookahead(mut self, lookahead_ms: f64) -> Self {
        self.lookahead_ms = lookahead_ms;
        self
    }

    /// Set sidechain high-pass filter.
    #[must_use]
    pub fn with_sidechain_hpf(mut self, cutoff_hz: f64) -> Self {
        self.sidechain_hpf_hz = cutoff_hz;
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

    /// Calculate auto makeup gain based on threshold and ratio.
    #[must_use]
    pub fn calculate_auto_makeup(&self) -> f64 {
        // Estimate average gain reduction and compensate
        if self.ratio <= 1.0 {
            return 0.0;
        }

        let gain_at_threshold = self.threshold_db - (self.threshold_db / self.ratio);
        -gain_at_threshold * 0.5 // Apply half the theoretical reduction
    }
}

/// Sidechain high-pass filter state.
#[derive(Clone, Debug, Default)]
struct SidechainHpf {
    x1: f64,
    y1: f64,
    alpha: f64,
    enabled: bool,
}

impl SidechainHpf {
    /// Create a new sidechain HPF.
    fn new(cutoff_hz: f64, sample_rate: f64) -> Self {
        if cutoff_hz <= 0.0 {
            return Self::default();
        }

        let rc = 1.0 / (2.0 * std::f64::consts::PI * cutoff_hz);
        let dt = 1.0 / sample_rate;
        let alpha = rc / (rc + dt);

        Self {
            x1: 0.0,
            y1: 0.0,
            alpha,
            enabled: true,
        }
    }

    /// Process a sample through the HPF.
    fn process(&mut self, x: f64) -> f64 {
        if !self.enabled {
            return x;
        }

        let y = self.alpha * (self.y1 + x - self.x1);
        self.x1 = x;
        self.y1 = y;
        y
    }

    /// Reset filter state.
    fn reset(&mut self) {
        self.x1 = 0.0;
        self.y1 = 0.0;
    }
}

/// Compressor envelope follower state.
#[derive(Clone, Debug)]
struct EnvelopeState {
    /// Current envelope level.
    envelope: f64,
    /// Attack coefficient.
    attack_coeff: f64,
    /// Release coefficient.
    release_coeff: f64,
}

impl EnvelopeState {
    /// Create a new envelope state.
    fn new(attack_ms: f64, release_ms: f64, sample_rate: f64) -> Self {
        let attack_coeff = if attack_ms > 0.0 {
            (-1.0 / (attack_ms * 0.001 * sample_rate)).exp()
        } else {
            0.0
        };

        let release_coeff = if release_ms > 0.0 {
            (-1.0 / (release_ms * 0.001 * sample_rate)).exp()
        } else {
            0.0
        };

        Self {
            envelope: 0.0,
            attack_coeff,
            release_coeff,
        }
    }

    /// Update envelope with new input level.
    fn update(&mut self, input_level: f64) {
        if input_level > self.envelope {
            self.envelope =
                self.attack_coeff * self.envelope + (1.0 - self.attack_coeff) * input_level;
        } else {
            self.envelope =
                self.release_coeff * self.envelope + (1.0 - self.release_coeff) * input_level;
        }
    }

    /// Reset envelope.
    fn reset(&mut self) {
        self.envelope = 0.0;
    }
}

/// Lookahead buffer state.
#[derive(Clone, Debug)]
struct LookaheadBuffer {
    /// Buffer per channel.
    buffers: Vec<VecDeque<f64>>,
    /// Delay in samples.
    delay_samples: usize,
}

impl LookaheadBuffer {
    /// Create a new lookahead buffer.
    fn new(lookahead_ms: f64, sample_rate: f64, channels: usize) -> Self {
        let delay_samples = (lookahead_ms * 0.001 * sample_rate) as usize;

        Self {
            buffers: vec![VecDeque::with_capacity(delay_samples + 1); channels],
            delay_samples,
        }
    }

    /// Push sample and get delayed sample.
    fn process(&mut self, channel: usize, sample: f64) -> f64 {
        if channel >= self.buffers.len() || self.delay_samples == 0 {
            return sample;
        }

        self.buffers[channel].push_back(sample);

        if self.buffers[channel].len() > self.delay_samples {
            self.buffers[channel].pop_front().unwrap_or(sample)
        } else {
            0.0 // Output silence until buffer is full
        }
    }

    /// Reset buffer.
    fn reset(&mut self) {
        for buffer in &mut self.buffers {
            buffer.clear();
        }
    }
}

/// Compressor internal state.
struct CompressorState {
    /// Envelope follower.
    envelope: EnvelopeState,
    /// Sidechain HPF per channel.
    sidechain_hpf: Vec<SidechainHpf>,
    /// Lookahead buffer.
    lookahead: LookaheadBuffer,
    /// Current gain reduction in dB (for metering).
    gain_reduction_db: f64,
    /// Effective makeup gain.
    effective_makeup: f64,
    /// Sample rate.
    sample_rate: f64,
}

impl CompressorState {
    /// Create new compressor state.
    fn new(config: &CompressorConfig, sample_rate: f64, channels: usize) -> Self {
        let envelope = EnvelopeState::new(config.attack_ms, config.release_ms, sample_rate);

        let sidechain_hpf = (0..channels)
            .map(|_| SidechainHpf::new(config.sidechain_hpf_hz, sample_rate))
            .collect();

        let lookahead = LookaheadBuffer::new(config.lookahead_ms, sample_rate, channels);

        let effective_makeup = if config.auto_makeup {
            CompressorConfig::db_to_linear(config.calculate_auto_makeup())
        } else {
            CompressorConfig::db_to_linear(config.makeup_gain_db)
        };

        Self {
            envelope,
            sidechain_hpf,
            lookahead,
            gain_reduction_db: 0.0,
            effective_makeup,
            sample_rate,
        }
    }

    /// Update timing parameters.
    fn update_timing(&mut self, config: &CompressorConfig) {
        self.envelope = EnvelopeState::new(config.attack_ms, config.release_ms, self.sample_rate);
    }

    /// Calculate gain reduction for a given input level.
    fn calculate_gain_reduction(config: &CompressorConfig, input_db: f64) -> f64 {
        if input_db < config.threshold_db {
            return 0.0;
        }

        match config.knee_type {
            KneeType::Hard => {
                let excess = input_db - config.threshold_db;
                let compressed_excess = excess / config.ratio;
                -(excess - compressed_excess)
            }
            KneeType::Soft => {
                let half_knee = config.knee_width_db / 2.0;
                let knee_start = config.threshold_db - half_knee;
                let knee_end = config.threshold_db + half_knee;

                if input_db < knee_start {
                    0.0
                } else if input_db > knee_end {
                    let excess = input_db - config.threshold_db;
                    let compressed_excess = excess / config.ratio;
                    -(excess - compressed_excess)
                } else {
                    // Soft knee region
                    let x = input_db - knee_start;
                    let knee_factor = x / config.knee_width_db;
                    let ratio_blend = 1.0 + (config.ratio - 1.0) * knee_factor;
                    let excess = x;
                    let compressed = excess / ratio_blend;
                    -(excess - compressed) * knee_factor
                }
            }
        }
    }

    /// Process samples.
    fn process(&mut self, samples: &mut [Vec<f64>], config: &CompressorConfig) {
        let sample_count = samples.get(0).map_or(0, Vec::len);
        let channels = samples.len();

        for i in 0..sample_count {
            // Detect level from sidechain (all channels summed)
            let mut sidechain_level = 0.0_f64;
            for ch in 0..channels {
                if i < samples[ch].len() && ch < self.sidechain_hpf.len() {
                    let filtered = self.sidechain_hpf[ch].process(samples[ch][i]);
                    sidechain_level = sidechain_level.max(filtered.abs());
                }
            }

            // Convert to dB (for potential metering/debugging)
            let _input_db = CompressorConfig::linear_to_db(sidechain_level);

            // Update envelope
            self.envelope.update(sidechain_level);

            // Calculate gain reduction
            let envelope_db = CompressorConfig::linear_to_db(self.envelope.envelope);
            self.gain_reduction_db = Self::calculate_gain_reduction(config, envelope_db);

            // Convert to linear gain
            let gain =
                CompressorConfig::db_to_linear(self.gain_reduction_db) * self.effective_makeup;

            // Apply to all channels with lookahead
            for ch in 0..channels {
                if i < samples[ch].len() {
                    let delayed = self.lookahead.process(ch, samples[ch][i]);
                    samples[ch][i] = delayed * gain;
                }
            }
        }
    }

    /// Reset state.
    fn reset(&mut self) {
        self.envelope.reset();
        for hpf in &mut self.sidechain_hpf {
            hpf.reset();
        }
        self.lookahead.reset();
        self.gain_reduction_db = 0.0;
    }
}

/// Audio dynamics compressor filter.
///
/// This filter provides dynamic range compression with configurable
/// threshold, ratio, attack, release, and knee settings.
///
/// # Example
///
/// ```ignore
/// use oximedia_graph::filters::audio::compressor::{CompressorFilter, CompressorConfig};
///
/// // Create a gentle compressor
/// let config = CompressorConfig::new(-20.0, 4.0)
///     .with_timing(10.0, 100.0)
///     .with_soft_knee(6.0)
///     .with_auto_makeup();
/// let filter = CompressorFilter::new(NodeId(0), "compressor", config);
/// ```
pub struct CompressorFilter {
    id: NodeId,
    name: String,
    state: NodeState,
    config: CompressorConfig,
    compressor_state: Option<CompressorState>,
    inputs: Vec<InputPort>,
    outputs: Vec<OutputPort>,
}

impl CompressorFilter {
    /// Create a new compressor filter.
    #[must_use]
    pub fn new(id: NodeId, name: impl Into<String>, config: CompressorConfig) -> Self {
        let audio_format = PortFormat::Audio(AudioPortFormat::any());

        Self {
            id,
            name: name.into(),
            state: NodeState::Idle,
            config,
            compressor_state: None,
            inputs: vec![InputPort::new(PortId(0), "input", PortType::Audio)
                .with_format(audio_format.clone())],
            outputs: vec![
                OutputPort::new(PortId(0), "output", PortType::Audio).with_format(audio_format)
            ],
        }
    }

    /// Get the current configuration.
    #[must_use]
    pub fn config(&self) -> &CompressorConfig {
        &self.config
    }

    /// Update the configuration.
    pub fn set_config(&mut self, config: CompressorConfig) {
        self.config = config;
        if let Some(ref mut state) = self.compressor_state {
            state.update_timing(&self.config);
        }
    }

    /// Get current gain reduction in dB.
    #[must_use]
    pub fn gain_reduction_db(&self) -> f64 {
        self.compressor_state
            .as_ref()
            .map_or(0.0, |s| s.gain_reduction_db)
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

impl Node for CompressorFilter {
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

        // Initialize compressor state if needed
        if self.compressor_state.is_none() {
            let channels = frame.channels.count();
            self.compressor_state = Some(CompressorState::new(
                &self.config,
                f64::from(frame.sample_rate),
                channels,
            ));
        }

        // Convert to f64 samples
        let mut samples = Self::frame_to_samples(&frame);

        // Apply compression
        if let Some(ref mut comp_state) = self.compressor_state {
            comp_state.process(&mut samples, &self.config);
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
        if let Some(ref mut state) = self.compressor_state {
            state.reset();
        }
        self.set_state(NodeState::Idle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_knee_type_default() {
        assert_eq!(KneeType::default(), KneeType::Hard);
    }

    #[test]
    fn test_compressor_config() {
        let config = CompressorConfig::new(-20.0, 4.0)
            .with_timing(10.0, 100.0)
            .with_soft_knee(6.0)
            .with_makeup_gain(6.0);

        assert!((config.threshold_db - (-20.0)).abs() < f64::EPSILON);
        assert!((config.ratio - 4.0).abs() < f64::EPSILON);
        assert!((config.attack_ms - 10.0).abs() < f64::EPSILON);
        assert!((config.release_ms - 100.0).abs() < f64::EPSILON);
        assert_eq!(config.knee_type, KneeType::Soft);
        assert!((config.makeup_gain_db - 6.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_db_conversion() {
        let linear = CompressorConfig::db_to_linear(0.0);
        assert!((linear - 1.0).abs() < f64::EPSILON);

        let db = CompressorConfig::linear_to_db(1.0);
        assert!(db.abs() < f64::EPSILON);

        let db = CompressorConfig::linear_to_db(0.0);
        assert!(db.is_infinite() && db.is_sign_negative());
    }

    #[test]
    fn test_auto_makeup() {
        let config = CompressorConfig::new(-20.0, 4.0).with_auto_makeup();
        assert!(config.auto_makeup);

        let makeup = config.calculate_auto_makeup();
        assert!(makeup > 0.0); // Should add positive gain
    }

    #[test]
    fn test_lookahead() {
        let config = CompressorConfig::new(-20.0, 4.0).with_lookahead(5.0);
        assert!((config.lookahead_ms - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_sidechain_hpf() {
        let config = CompressorConfig::new(-20.0, 4.0).with_sidechain_hpf(80.0);
        assert!((config.sidechain_hpf_hz - 80.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_envelope_state() {
        let mut envelope = EnvelopeState::new(10.0, 100.0, 48000.0);

        // Attack phase
        envelope.update(1.0);
        assert!(envelope.envelope > 0.0);
        assert!(envelope.envelope < 1.0);

        // Release phase
        envelope.update(0.0);
        let level = envelope.envelope;
        envelope.update(0.0);
        assert!(envelope.envelope < level);

        // Reset
        envelope.reset();
        assert!(envelope.envelope.abs() < f64::EPSILON);
    }

    #[test]
    fn test_lookahead_buffer() {
        let mut buffer = LookaheadBuffer::new(1.0, 48000.0, 2);

        // First samples should be delayed
        let delayed = buffer.process(0, 1.0);
        assert!(delayed.abs() < f64::EPSILON); // Still filling buffer

        // After buffer fills, should get delayed samples
        for _ in 0..100 {
            buffer.process(0, 0.5);
        }

        // Reset
        buffer.reset();
        assert!(buffer.buffers[0].is_empty());
    }

    #[test]
    fn test_gain_reduction_hard_knee() {
        let config = CompressorConfig::new(-20.0, 4.0);

        // Below threshold - no reduction
        let gr = CompressorState::calculate_gain_reduction(&config, -30.0);
        assert!(gr.abs() < f64::EPSILON);

        // At threshold - no reduction
        let gr = CompressorState::calculate_gain_reduction(&config, -20.0);
        assert!(gr.abs() < f64::EPSILON);

        // Above threshold - should have reduction
        let gr = CompressorState::calculate_gain_reduction(&config, -10.0);
        assert!(gr < 0.0);
    }

    #[test]
    fn test_gain_reduction_soft_knee() {
        let config = CompressorConfig::new(-20.0, 4.0).with_soft_knee(6.0);

        // Well below threshold
        let gr = CompressorState::calculate_gain_reduction(&config, -30.0);
        assert!(gr.abs() < f64::EPSILON);

        // In knee region
        let gr = CompressorState::calculate_gain_reduction(&config, -20.0);
        assert!(gr <= 0.0);
    }

    #[test]
    fn test_compressor_filter_creation() {
        let config = CompressorConfig::new(-20.0, 4.0);
        let filter = CompressorFilter::new(NodeId(1), "compressor", config);

        assert_eq!(filter.id(), NodeId(1));
        assert_eq!(filter.name(), "compressor");
        assert_eq!(filter.node_type(), NodeType::Filter);
    }

    #[test]
    fn test_compressor_filter_ports() {
        let config = CompressorConfig::default();
        let filter = CompressorFilter::new(NodeId(0), "test", config);

        assert_eq!(filter.inputs().len(), 1);
        assert_eq!(filter.outputs().len(), 1);
        assert_eq!(filter.inputs()[0].port_type, PortType::Audio);
    }

    #[test]
    fn test_process_none() {
        let config = CompressorConfig::default();
        let mut filter = CompressorFilter::new(NodeId(0), "test", config);

        let result = filter.process(None).expect("process should succeed");
        assert!(result.is_none());
    }

    #[test]
    fn test_process_audio() {
        let config = CompressorConfig::new(-20.0, 4.0);
        let mut filter = CompressorFilter::new(NodeId(0), "test", config);

        let mut frame = AudioFrame::new(SampleFormat::F32, 48000, ChannelLayout::Mono);
        let mut samples = BytesMut::new();
        for _ in 0..100 {
            samples.extend_from_slice(&0.5f32.to_le_bytes());
        }
        frame.samples = AudioBuffer::Interleaved(samples.freeze());

        let result = filter
            .process(Some(FilterFrame::Audio(frame)))
            .expect("process should succeed");
        assert!(result.is_some());
    }

    #[test]
    fn test_gain_reduction_metering() {
        let config = CompressorConfig::new(-20.0, 4.0);
        let filter = CompressorFilter::new(NodeId(0), "test", config);

        // Before processing, GR should be 0
        assert!(filter.gain_reduction_db().abs() < f64::EPSILON);
    }

    #[test]
    fn test_state_transitions() {
        let config = CompressorConfig::default();
        let mut filter = CompressorFilter::new(NodeId(0), "test", config);

        assert!(filter.set_state(NodeState::Processing).is_ok());
        assert_eq!(filter.state(), NodeState::Processing);

        assert!(filter.reset().is_ok());
        assert_eq!(filter.state(), NodeState::Idle);
    }

    #[test]
    fn test_sidechain_hpf_state() {
        let mut hpf = SidechainHpf::new(80.0, 48000.0);
        assert!(hpf.enabled);

        let output = hpf.process(1.0);
        assert!(output.is_finite());

        hpf.reset();
        assert!(hpf.x1.abs() < f64::EPSILON);
    }

    #[test]
    fn test_disabled_sidechain() {
        let hpf = SidechainHpf::new(0.0, 48000.0);
        assert!(!hpf.enabled);
    }
}
