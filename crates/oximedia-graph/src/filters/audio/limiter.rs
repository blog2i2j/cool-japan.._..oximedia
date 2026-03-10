//! Audio limiter filter.
//!
//! This module provides a brickwall limiter with true peak limiting
//! and soft limiting options.

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

/// Limiter mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum LimiterMode {
    /// Brickwall limiting - hard ceiling.
    #[default]
    Brickwall,
    /// Soft limiting with saturation curve.
    Soft,
}

/// Configuration for the limiter.
#[derive(Clone, Debug)]
pub struct LimiterConfig {
    /// Ceiling level in dB (maximum output level).
    pub ceiling_db: f64,
    /// Release time in milliseconds.
    pub release_ms: f64,
    /// Limiter mode.
    pub mode: LimiterMode,
    /// Enable true peak limiting (oversampled detection).
    pub true_peak: bool,
    /// Lookahead time in milliseconds.
    pub lookahead_ms: f64,
}

impl Default for LimiterConfig {
    fn default() -> Self {
        Self {
            ceiling_db: -0.3,
            release_ms: 100.0,
            mode: LimiterMode::Brickwall,
            true_peak: true,
            lookahead_ms: 5.0,
        }
    }
}

impl LimiterConfig {
    /// Create a new limiter configuration.
    #[must_use]
    pub fn new(ceiling_db: f64) -> Self {
        Self {
            ceiling_db,
            ..Default::default()
        }
    }

    /// Set release time.
    #[must_use]
    pub fn with_release(mut self, release_ms: f64) -> Self {
        self.release_ms = release_ms;
        self
    }

    /// Set limiter mode.
    #[must_use]
    pub fn with_mode(mut self, mode: LimiterMode) -> Self {
        self.mode = mode;
        self
    }

    /// Enable or disable true peak limiting.
    #[must_use]
    pub fn with_true_peak(mut self, enabled: bool) -> Self {
        self.true_peak = enabled;
        self
    }

    /// Set lookahead time.
    #[must_use]
    pub fn with_lookahead(mut self, lookahead_ms: f64) -> Self {
        self.lookahead_ms = lookahead_ms;
        self
    }

    /// Create a brickwall limiter.
    #[must_use]
    pub fn brickwall(ceiling_db: f64) -> Self {
        Self::new(ceiling_db).with_mode(LimiterMode::Brickwall)
    }

    /// Create a soft limiter.
    #[must_use]
    pub fn soft(ceiling_db: f64) -> Self {
        Self::new(ceiling_db).with_mode(LimiterMode::Soft)
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

/// True peak detector using oversampling.
#[derive(Clone, Debug)]
struct TruePeakDetector {
    /// Oversampling factor.
    oversample: usize,
    /// FIR filter coefficients for upsampling.
    filter_coeffs: Vec<f64>,
    /// History buffer for each channel.
    history: Vec<VecDeque<f64>>,
}

impl TruePeakDetector {
    /// Create a new true peak detector.
    fn new(channels: usize) -> Self {
        // 4x oversampling with simple linear interpolation
        let oversample = 4;
        let filter_coeffs = Self::create_filter_coeffs();
        let history = vec![VecDeque::with_capacity(filter_coeffs.len()); channels];

        Self {
            oversample,
            filter_coeffs,
            history,
        }
    }

    /// Create FIR filter coefficients for upsampling.
    fn create_filter_coeffs() -> Vec<f64> {
        // Simple windowed sinc filter for 4x oversampling
        vec![
            0.0, 0.0625, 0.0, 0.125, 0.0, 0.25, 0.0, 0.5, 1.0, 0.5, 0.0, 0.25, 0.0, 0.125, 0.0,
            0.0625,
        ]
    }

    /// Detect true peak for a sample.
    fn detect(&mut self, channel: usize, sample: f64) -> f64 {
        if channel >= self.history.len() {
            return sample.abs();
        }

        // Add sample to history
        self.history[channel].push_back(sample);
        if self.history[channel].len() > self.filter_coeffs.len() {
            self.history[channel].pop_front();
        }

        // Find peak across oversampled positions
        let mut peak = sample.abs();

        for phase in 0..self.oversample {
            let mut sum = 0.0;
            for (i, &coeff) in self.filter_coeffs.iter().enumerate() {
                let idx = (i * self.oversample + phase) / self.oversample;
                if idx < self.history[channel].len() {
                    let hist_sample = self.history[channel].get(idx).copied().unwrap_or(0.0);
                    sum += hist_sample * coeff;
                }
            }
            peak = peak.max(sum.abs());
        }

        peak
    }

    /// Reset detector state.
    fn reset(&mut self) {
        for hist in &mut self.history {
            hist.clear();
        }
    }
}

/// Lookahead buffer for the limiter.
#[derive(Clone, Debug)]
struct LookaheadBuffer {
    /// Buffer per channel.
    buffers: Vec<VecDeque<f64>>,
    /// Peak buffer for the lookahead window.
    peak_buffer: VecDeque<f64>,
    /// Delay in samples.
    delay_samples: usize,
}

impl LookaheadBuffer {
    /// Create a new lookahead buffer.
    fn new(lookahead_ms: f64, sample_rate: f64, channels: usize) -> Self {
        let delay_samples = ((lookahead_ms * 0.001 * sample_rate) as usize).max(1);

        Self {
            buffers: vec![VecDeque::with_capacity(delay_samples + 1); channels],
            peak_buffer: VecDeque::with_capacity(delay_samples + 1),
            delay_samples,
        }
    }

    /// Push sample and get delayed sample.
    fn process(&mut self, channel: usize, sample: f64, peak: f64) -> (f64, f64) {
        if channel >= self.buffers.len() {
            return (sample, peak);
        }

        self.buffers[channel].push_back(sample);
        if channel == 0 {
            self.peak_buffer.push_back(peak);
        }

        let delayed_sample = if self.buffers[channel].len() > self.delay_samples {
            self.buffers[channel].pop_front().unwrap_or(sample)
        } else {
            0.0
        };

        // Find maximum peak in lookahead window
        let lookahead_peak = if channel == 0 && self.peak_buffer.len() > self.delay_samples {
            self.peak_buffer.pop_front();
            self.peak_buffer.iter().copied().fold(0.0_f64, f64::max)
        } else {
            self.peak_buffer.iter().copied().fold(0.0_f64, f64::max)
        };

        (delayed_sample, lookahead_peak)
    }

    /// Reset buffer.
    fn reset(&mut self) {
        for buffer in &mut self.buffers {
            buffer.clear();
        }
        self.peak_buffer.clear();
    }
}

/// Gain smoothing state.
#[derive(Clone, Debug)]
struct GainSmoother {
    /// Current gain value.
    current_gain: f64,
    /// Release coefficient.
    release_coeff: f64,
}

impl GainSmoother {
    /// Create a new gain smoother.
    fn new(release_ms: f64, sample_rate: f64) -> Self {
        let release_coeff = if release_ms > 0.0 {
            (-1.0 / (release_ms * 0.001 * sample_rate)).exp()
        } else {
            0.0
        };

        Self {
            current_gain: 1.0,
            release_coeff,
        }
    }

    /// Update gain for a target gain value.
    fn update(&mut self, target_gain: f64) -> f64 {
        if target_gain < self.current_gain {
            // Instant attack
            self.current_gain = target_gain;
        } else {
            // Smooth release
            self.current_gain =
                self.release_coeff * self.current_gain + (1.0 - self.release_coeff) * target_gain;
        }
        self.current_gain
    }

    /// Reset state.
    fn reset(&mut self) {
        self.current_gain = 1.0;
    }
}

/// Limiter internal state.
struct LimiterState {
    /// Ceiling level (linear).
    ceiling: f64,
    /// True peak detector.
    true_peak_detector: TruePeakDetector,
    /// Lookahead buffer.
    lookahead: LookaheadBuffer,
    /// Gain smoother.
    gain_smoother: GainSmoother,
    /// Current gain reduction in dB (for metering).
    gain_reduction_db: f64,
    /// Limiter mode.
    mode: LimiterMode,
}

impl LimiterState {
    /// Create new limiter state.
    fn new(config: &LimiterConfig, sample_rate: f64, channels: usize) -> Self {
        let ceiling = LimiterConfig::db_to_linear(config.ceiling_db);

        Self {
            ceiling,
            true_peak_detector: TruePeakDetector::new(channels),
            lookahead: LookaheadBuffer::new(config.lookahead_ms, sample_rate, channels),
            gain_smoother: GainSmoother::new(config.release_ms, sample_rate),
            gain_reduction_db: 0.0,
            mode: config.mode,
        }
    }

    /// Apply soft limiting curve.
    fn soft_limit(sample: f64, ceiling: f64) -> f64 {
        if sample.abs() <= ceiling {
            sample
        } else {
            let sign = sample.signum();
            let excess = sample.abs() - ceiling;
            let range = 1.0 - ceiling;
            let saturated = ceiling + range * (excess / range).tanh();
            sign * saturated.min(1.0)
        }
    }

    /// Process samples.
    fn process(&mut self, samples: &mut [Vec<f64>], config: &LimiterConfig) {
        let sample_count = samples.get(0).map_or(0, Vec::len);
        let channels = samples.len();

        for i in 0..sample_count {
            // Detect peak across all channels
            let mut peak = 0.0_f64;
            for ch in 0..channels {
                if i < samples[ch].len() {
                    let sample_peak = if config.true_peak {
                        self.true_peak_detector.detect(ch, samples[ch][i])
                    } else {
                        samples[ch][i].abs()
                    };
                    peak = peak.max(sample_peak);
                }
            }

            // Calculate required gain
            let target_gain = if peak > self.ceiling {
                self.ceiling / peak
            } else {
                1.0
            };

            // Apply to all channels with lookahead
            for ch in 0..channels {
                if i < samples[ch].len() {
                    let (delayed_sample, lookahead_peak) =
                        self.lookahead.process(ch, samples[ch][i], peak);

                    // Calculate gain based on lookahead peak
                    let lookahead_gain = if lookahead_peak > self.ceiling {
                        self.ceiling / lookahead_peak
                    } else {
                        1.0
                    };

                    // Use the more aggressive gain
                    let final_target = target_gain.min(lookahead_gain);
                    let smoothed_gain = self.gain_smoother.update(final_target);

                    // Apply gain
                    let output = match self.mode {
                        LimiterMode::Brickwall => {
                            (delayed_sample * smoothed_gain).clamp(-self.ceiling, self.ceiling)
                        }
                        LimiterMode::Soft => {
                            Self::soft_limit(delayed_sample * smoothed_gain, self.ceiling)
                        }
                    };

                    samples[ch][i] = output;
                }
            }

            // Update metering
            if target_gain < 1.0 {
                self.gain_reduction_db = LimiterConfig::linear_to_db(target_gain);
            } else {
                // Slow release of meter
                self.gain_reduction_db *= 0.999;
            }
        }
    }

    /// Reset state.
    fn reset(&mut self) {
        self.true_peak_detector.reset();
        self.lookahead.reset();
        self.gain_smoother.reset();
        self.gain_reduction_db = 0.0;
    }
}

/// Audio limiter filter.
///
/// This filter provides brickwall or soft limiting with true peak detection
/// and lookahead for transparent gain reduction.
///
/// # Example
///
/// ```ignore
/// use oximedia_graph::filters::audio::limiter::{LimiterFilter, LimiterConfig};
///
/// // Create a brickwall limiter at -0.3 dBFS
/// let config = LimiterConfig::brickwall(-0.3)
///     .with_release(100.0)
///     .with_true_peak(true);
/// let filter = LimiterFilter::new(NodeId(0), "limiter", config);
/// ```
pub struct LimiterFilter {
    id: NodeId,
    name: String,
    state: NodeState,
    config: LimiterConfig,
    limiter_state: Option<LimiterState>,
    inputs: Vec<InputPort>,
    outputs: Vec<OutputPort>,
}

impl LimiterFilter {
    /// Create a new limiter filter.
    #[must_use]
    pub fn new(id: NodeId, name: impl Into<String>, config: LimiterConfig) -> Self {
        let audio_format = PortFormat::Audio(AudioPortFormat::any());

        Self {
            id,
            name: name.into(),
            state: NodeState::Idle,
            config,
            limiter_state: None,
            inputs: vec![InputPort::new(PortId(0), "input", PortType::Audio)
                .with_format(audio_format.clone())],
            outputs: vec![
                OutputPort::new(PortId(0), "output", PortType::Audio).with_format(audio_format)
            ],
        }
    }

    /// Get the current configuration.
    #[must_use]
    pub fn config(&self) -> &LimiterConfig {
        &self.config
    }

    /// Update the configuration.
    pub fn set_config(&mut self, config: LimiterConfig) {
        self.config = config;
        self.limiter_state = None; // Reset state
    }

    /// Get current gain reduction in dB.
    #[must_use]
    pub fn gain_reduction_db(&self) -> f64 {
        self.limiter_state
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

impl Node for LimiterFilter {
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

        // Initialize limiter state if needed
        if self.limiter_state.is_none() {
            let channels = frame.channels.count();
            self.limiter_state = Some(LimiterState::new(
                &self.config,
                f64::from(frame.sample_rate),
                channels,
            ));
        }

        // Convert to f64 samples
        let mut samples = Self::frame_to_samples(&frame);

        // Apply limiting
        if let Some(ref mut limiter_state) = self.limiter_state {
            limiter_state.process(&mut samples, &self.config);
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
        if let Some(ref mut state) = self.limiter_state {
            state.reset();
        }
        self.set_state(NodeState::Idle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_limiter_mode_default() {
        assert_eq!(LimiterMode::default(), LimiterMode::Brickwall);
    }

    #[test]
    fn test_limiter_config() {
        let config = LimiterConfig::new(-0.3)
            .with_release(100.0)
            .with_mode(LimiterMode::Soft)
            .with_true_peak(true)
            .with_lookahead(5.0);

        assert!((config.ceiling_db - (-0.3)).abs() < f64::EPSILON);
        assert!((config.release_ms - 100.0).abs() < f64::EPSILON);
        assert_eq!(config.mode, LimiterMode::Soft);
        assert!(config.true_peak);
        assert!((config.lookahead_ms - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_preset_configs() {
        let brickwall = LimiterConfig::brickwall(-0.5);
        assert_eq!(brickwall.mode, LimiterMode::Brickwall);
        assert!((brickwall.ceiling_db - (-0.5)).abs() < f64::EPSILON);

        let soft = LimiterConfig::soft(-1.0);
        assert_eq!(soft.mode, LimiterMode::Soft);
    }

    #[test]
    fn test_db_conversion() {
        let linear = LimiterConfig::db_to_linear(0.0);
        assert!((linear - 1.0).abs() < f64::EPSILON);

        let db = LimiterConfig::linear_to_db(1.0);
        assert!(db.abs() < f64::EPSILON);
    }

    #[test]
    fn test_soft_limit() {
        let ceiling = 0.9;

        // Below ceiling - no change
        let result = LimiterState::soft_limit(0.5, ceiling);
        assert!((result - 0.5).abs() < f64::EPSILON);

        // At ceiling - no change
        let result = LimiterState::soft_limit(0.9, ceiling);
        assert!((result - 0.9).abs() < f64::EPSILON);

        // Above ceiling - limited
        let result = LimiterState::soft_limit(1.5, ceiling);
        assert!(result > ceiling);
        assert!(result <= 1.0);

        // Negative values
        let result = LimiterState::soft_limit(-1.5, ceiling);
        assert!(result < -ceiling);
        assert!(result >= -1.0);
    }

    #[test]
    fn test_true_peak_detector() {
        let mut detector = TruePeakDetector::new(2);

        let peak = detector.detect(0, 0.5);
        assert!(peak >= 0.5);
        assert!(peak.is_finite());

        detector.reset();
        assert!(detector.history[0].is_empty());
    }

    #[test]
    fn test_lookahead_buffer() {
        let mut buffer = LookaheadBuffer::new(1.0, 48000.0, 2);

        // First samples should be delayed
        let (delayed, _peak) = buffer.process(0, 1.0, 1.0);
        assert!(delayed.abs() < f64::EPSILON); // Still filling buffer

        // Reset
        buffer.reset();
        assert!(buffer.buffers[0].is_empty());
    }

    #[test]
    fn test_gain_smoother() {
        let mut smoother = GainSmoother::new(100.0, 48000.0);

        // Attack should be instant
        let gain = smoother.update(0.5);
        assert!((gain - 0.5).abs() < f64::EPSILON);

        // Release should be gradual
        let gain1 = smoother.update(1.0);
        let gain2 = smoother.update(1.0);
        assert!(gain1 < gain2);
        assert!(gain2 < 1.0);

        smoother.reset();
        assert!((smoother.current_gain - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_limiter_filter_creation() {
        let config = LimiterConfig::brickwall(-0.3);
        let filter = LimiterFilter::new(NodeId(1), "limiter", config);

        assert_eq!(filter.id(), NodeId(1));
        assert_eq!(filter.name(), "limiter");
        assert_eq!(filter.node_type(), NodeType::Filter);
    }

    #[test]
    fn test_limiter_filter_ports() {
        let config = LimiterConfig::default();
        let filter = LimiterFilter::new(NodeId(0), "test", config);

        assert_eq!(filter.inputs().len(), 1);
        assert_eq!(filter.outputs().len(), 1);
        assert_eq!(filter.inputs()[0].port_type, PortType::Audio);
    }

    #[test]
    fn test_process_none() {
        let config = LimiterConfig::default();
        let mut filter = LimiterFilter::new(NodeId(0), "test", config);

        let result = filter.process(None).expect("process should succeed");
        assert!(result.is_none());
    }

    #[test]
    fn test_process_audio() {
        let config = LimiterConfig::brickwall(-6.0);
        let mut filter = LimiterFilter::new(NodeId(0), "test", config);

        let mut frame = AudioFrame::new(SampleFormat::F32, 48000, ChannelLayout::Mono);
        let mut samples = BytesMut::new();
        for _ in 0..100 {
            samples.extend_from_slice(&1.0f32.to_le_bytes()); // Full scale
        }
        frame.samples = AudioBuffer::Interleaved(samples.freeze());

        let result = filter
            .process(Some(FilterFrame::Audio(frame)))
            .expect("process should succeed");
        assert!(result.is_some());

        if let Some(FilterFrame::Audio(output)) = result {
            if let AudioBuffer::Interleaved(data) = &output.samples {
                // After lookahead delay, samples should be limited
                let last_offset = data.len() - 4;
                let sample = f32::from_le_bytes([
                    data[last_offset],
                    data[last_offset + 1],
                    data[last_offset + 2],
                    data[last_offset + 3],
                ]);
                let ceiling = LimiterConfig::db_to_linear(-6.0);
                assert!(sample.abs() <= ceiling as f32 + 0.01);
            }
        }
    }

    #[test]
    fn test_gain_reduction_metering() {
        let config = LimiterConfig::default();
        let filter = LimiterFilter::new(NodeId(0), "test", config);

        // Before processing, GR should be 0
        assert!(filter.gain_reduction_db().abs() < f64::EPSILON);
    }

    #[test]
    fn test_state_transitions() {
        let config = LimiterConfig::default();
        let mut filter = LimiterFilter::new(NodeId(0), "test", config);

        assert!(filter.set_state(NodeState::Processing).is_ok());
        assert_eq!(filter.state(), NodeState::Processing);

        assert!(filter.reset().is_ok());
        assert_eq!(filter.state(), NodeState::Idle);
    }
}
