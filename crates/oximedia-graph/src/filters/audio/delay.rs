//! Audio delay filter.
//!
//! This module provides a delay effect with feedback, ping-pong mode,
//! and dry/wet mix control.

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

/// Delay mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DelayMode {
    /// Normal delay - same delay for all channels.
    #[default]
    Normal,
    /// Ping-pong stereo delay - alternates between left and right.
    PingPong,
}

/// Configuration for the delay filter.
#[derive(Clone, Debug)]
pub struct DelayConfig {
    /// Delay time in milliseconds.
    pub delay_ms: f64,
    /// Feedback amount (0.0 to 1.0).
    pub feedback: f64,
    /// Dry/wet mix (0.0 = dry only, 1.0 = wet only).
    pub mix: f64,
    /// Delay mode.
    pub mode: DelayMode,
    /// High-frequency damping factor (0.0 = none, 1.0 = full).
    pub damping: f64,
}

impl Default for DelayConfig {
    fn default() -> Self {
        Self {
            delay_ms: 250.0,
            feedback: 0.5,
            mix: 0.5,
            mode: DelayMode::Normal,
            damping: 0.0,
        }
    }
}

impl DelayConfig {
    /// Create a new delay configuration.
    #[must_use]
    pub fn new(delay_ms: f64) -> Self {
        Self {
            delay_ms,
            ..Default::default()
        }
    }

    /// Set feedback amount.
    #[must_use]
    pub fn with_feedback(mut self, feedback: f64) -> Self {
        self.feedback = feedback.clamp(0.0, 0.99);
        self
    }

    /// Set dry/wet mix.
    #[must_use]
    pub fn with_mix(mut self, mix: f64) -> Self {
        self.mix = mix.clamp(0.0, 1.0);
        self
    }

    /// Set delay mode.
    #[must_use]
    pub fn with_mode(mut self, mode: DelayMode) -> Self {
        self.mode = mode;
        self
    }

    /// Enable ping-pong mode.
    #[must_use]
    pub fn ping_pong(mut self) -> Self {
        self.mode = DelayMode::PingPong;
        self
    }

    /// Set damping factor.
    #[must_use]
    pub fn with_damping(mut self, damping: f64) -> Self {
        self.damping = damping.clamp(0.0, 1.0);
        self
    }
}

/// Delay line for one channel.
#[derive(Clone, Debug)]
struct DelayLine {
    /// Circular buffer.
    buffer: VecDeque<f64>,
    /// Delay in samples.
    delay_samples: usize,
    /// Low-pass filter state for damping.
    lp_state: f64,
    /// Damping coefficient.
    damping: f64,
}

impl DelayLine {
    /// Create a new delay line.
    fn new(delay_ms: f64, sample_rate: f64, damping: f64) -> Self {
        let delay_samples = ((delay_ms * 0.001 * sample_rate) as usize).max(1);
        let mut buffer = VecDeque::with_capacity(delay_samples + 1);

        // Initialize with zeros
        for _ in 0..delay_samples {
            buffer.push_back(0.0);
        }

        Self {
            buffer,
            delay_samples,
            lp_state: 0.0,
            damping,
        }
    }

    /// Process one sample through the delay line.
    fn process(&mut self, input: f64, feedback: f64) -> f64 {
        // Get delayed output
        let output = self.buffer.pop_front().unwrap_or(0.0);

        // Apply damping (simple one-pole low-pass)
        let damped = if self.damping > 0.0 {
            self.lp_state = self.lp_state + self.damping * (output - self.lp_state);
            output - self.damping * (output - self.lp_state)
        } else {
            output
        };

        // Feed back into delay line
        self.buffer.push_back(input + damped * feedback);

        damped
    }

    /// Reset delay line.
    fn reset(&mut self) {
        self.buffer.clear();
        for _ in 0..self.delay_samples {
            self.buffer.push_back(0.0);
        }
        self.lp_state = 0.0;
    }

    /// Get current delay samples.
    #[allow(dead_code)]
    fn delay_samples(&self) -> usize {
        self.delay_samples
    }
}

/// Delay internal state.
struct DelayState {
    /// Delay lines per channel.
    delay_lines: Vec<DelayLine>,
    /// Ping-pong state (which channel to feed back to).
    ping_pong_state: bool,
}

impl DelayState {
    /// Create new delay state.
    fn new(config: &DelayConfig, sample_rate: f64, channels: usize) -> Self {
        let delay_lines = (0..channels)
            .map(|_| DelayLine::new(config.delay_ms, sample_rate, config.damping))
            .collect();

        Self {
            delay_lines,
            ping_pong_state: false,
        }
    }

    /// Process samples.
    fn process(&mut self, samples: &mut [Vec<f64>], config: &DelayConfig) {
        let sample_count = samples.get(0).map_or(0, Vec::len);
        let channels = samples.len();

        for i in 0..sample_count {
            match config.mode {
                DelayMode::Normal => {
                    // Normal delay - each channel delayed independently
                    for ch in 0..channels {
                        if i < samples[ch].len() && ch < self.delay_lines.len() {
                            let dry = samples[ch][i];
                            let wet = self.delay_lines[ch].process(dry, config.feedback);
                            samples[ch][i] = dry * (1.0 - config.mix) + wet * config.mix;
                        }
                    }
                }
                DelayMode::PingPong => {
                    // Ping-pong delay - alternates between channels
                    if channels >= 2 {
                        let (left_dry, right_dry) = if i < samples[0].len() && i < samples[1].len()
                        {
                            (samples[0][i], samples[1][i])
                        } else {
                            (0.0, 0.0)
                        };

                        // Process with cross-feedback
                        let (left_wet, right_wet) = if self.ping_pong_state {
                            (
                                self.delay_lines[0].process(right_dry, config.feedback),
                                self.delay_lines[1].process(left_dry, config.feedback),
                            )
                        } else {
                            (
                                self.delay_lines[0].process(left_dry, config.feedback),
                                self.delay_lines[1].process(right_dry, config.feedback),
                            )
                        };

                        if i < samples[0].len() {
                            samples[0][i] = left_dry * (1.0 - config.mix) + left_wet * config.mix;
                        }
                        if i < samples[1].len() {
                            samples[1][i] = right_dry * (1.0 - config.mix) + right_wet * config.mix;
                        }

                        // Toggle ping-pong state
                        self.ping_pong_state = !self.ping_pong_state;
                    } else {
                        // Mono - fall back to normal delay
                        for ch in 0..channels {
                            if i < samples[ch].len() && ch < self.delay_lines.len() {
                                let dry = samples[ch][i];
                                let wet = self.delay_lines[ch].process(dry, config.feedback);
                                samples[ch][i] = dry * (1.0 - config.mix) + wet * config.mix;
                            }
                        }
                    }
                }
            }
        }
    }

    /// Reset state.
    fn reset(&mut self) {
        for line in &mut self.delay_lines {
            line.reset();
        }
        self.ping_pong_state = false;
    }
}

/// Audio delay filter.
///
/// This filter provides a delay effect with feedback, ping-pong stereo mode,
/// and dry/wet mix control.
///
/// # Example
///
/// ```ignore
/// use oximedia_graph::filters::audio::delay::{DelayFilter, DelayConfig};
///
/// // Create a ping-pong delay
/// let config = DelayConfig::new(250.0)
///     .with_feedback(0.5)
///     .with_mix(0.5)
///     .ping_pong();
/// let filter = DelayFilter::new(NodeId(0), "delay", config);
/// ```
pub struct DelayFilter {
    id: NodeId,
    name: String,
    state: NodeState,
    config: DelayConfig,
    delay_state: Option<DelayState>,
    inputs: Vec<InputPort>,
    outputs: Vec<OutputPort>,
}

impl DelayFilter {
    /// Create a new delay filter.
    #[must_use]
    pub fn new(id: NodeId, name: impl Into<String>, config: DelayConfig) -> Self {
        let audio_format = PortFormat::Audio(AudioPortFormat::any());

        Self {
            id,
            name: name.into(),
            state: NodeState::Idle,
            config,
            delay_state: None,
            inputs: vec![InputPort::new(PortId(0), "input", PortType::Audio)
                .with_format(audio_format.clone())],
            outputs: vec![
                OutputPort::new(PortId(0), "output", PortType::Audio).with_format(audio_format)
            ],
        }
    }

    /// Get the current configuration.
    #[must_use]
    pub fn config(&self) -> &DelayConfig {
        &self.config
    }

    /// Update the configuration.
    pub fn set_config(&mut self, config: DelayConfig) {
        self.config = config;
        self.delay_state = None; // Reset state
    }

    /// Set delay time.
    pub fn set_delay_time(&mut self, delay_ms: f64) {
        self.config.delay_ms = delay_ms;
        self.delay_state = None; // Need to rebuild delay lines
    }

    /// Set feedback amount.
    pub fn set_feedback(&mut self, feedback: f64) {
        self.config.feedback = feedback.clamp(0.0, 0.99);
    }

    /// Set dry/wet mix.
    pub fn set_mix(&mut self, mix: f64) {
        self.config.mix = mix.clamp(0.0, 1.0);
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

impl Node for DelayFilter {
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

        // Initialize delay state if needed
        if self.delay_state.is_none() {
            let channels = frame.channels.count();
            self.delay_state = Some(DelayState::new(
                &self.config,
                f64::from(frame.sample_rate),
                channels,
            ));
        }

        // Convert to f64 samples
        let mut samples = Self::frame_to_samples(&frame);

        // Apply delay
        if let Some(ref mut delay_state) = self.delay_state {
            delay_state.process(&mut samples, &self.config);
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
        if let Some(ref mut state) = self.delay_state {
            state.reset();
        }
        self.set_state(NodeState::Idle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delay_mode_default() {
        assert_eq!(DelayMode::default(), DelayMode::Normal);
    }

    #[test]
    fn test_delay_config() {
        let config = DelayConfig::new(500.0)
            .with_feedback(0.7)
            .with_mix(0.3)
            .with_damping(0.2);

        assert!((config.delay_ms - 500.0).abs() < f64::EPSILON);
        assert!((config.feedback - 0.7).abs() < f64::EPSILON);
        assert!((config.mix - 0.3).abs() < f64::EPSILON);
        assert!((config.damping - 0.2).abs() < f64::EPSILON);
    }

    #[test]
    fn test_feedback_clamping() {
        let config = DelayConfig::new(250.0).with_feedback(1.5);
        assert!((config.feedback - 0.99).abs() < f64::EPSILON);

        let config = DelayConfig::new(250.0).with_feedback(-0.5);
        assert!(config.feedback.abs() < f64::EPSILON);
    }

    #[test]
    fn test_mix_clamping() {
        let config = DelayConfig::new(250.0).with_mix(1.5);
        assert!((config.mix - 1.0).abs() < f64::EPSILON);

        let config = DelayConfig::new(250.0).with_mix(-0.5);
        assert!(config.mix.abs() < f64::EPSILON);
    }

    #[test]
    fn test_ping_pong_mode() {
        let config = DelayConfig::new(250.0).ping_pong();
        assert_eq!(config.mode, DelayMode::PingPong);
    }

    #[test]
    fn test_delay_line() {
        let mut line = DelayLine::new(10.0, 48000.0, 0.0);

        // First output should be from initial buffer (silence)
        let output = line.process(1.0, 0.0);
        assert!(output.abs() < f64::EPSILON);

        // After delay time, should hear original input
        for _ in 0..500 {
            line.process(0.0, 0.0);
        }

        // Reset and verify
        line.reset();
        let output = line.process(0.5, 0.0);
        assert!(output.abs() < f64::EPSILON);
    }

    #[test]
    fn test_delay_line_with_damping() {
        let mut line = DelayLine::new(10.0, 48000.0, 0.5);

        // Process some samples
        for _ in 0..100 {
            let output = line.process(0.5, 0.5);
            assert!(output.is_finite());
        }
    }

    #[test]
    fn test_delay_filter_creation() {
        let config = DelayConfig::new(250.0);
        let filter = DelayFilter::new(NodeId(1), "delay", config);

        assert_eq!(filter.id(), NodeId(1));
        assert_eq!(filter.name(), "delay");
        assert_eq!(filter.node_type(), NodeType::Filter);
    }

    #[test]
    fn test_delay_filter_ports() {
        let config = DelayConfig::default();
        let filter = DelayFilter::new(NodeId(0), "test", config);

        assert_eq!(filter.inputs().len(), 1);
        assert_eq!(filter.outputs().len(), 1);
        assert_eq!(filter.inputs()[0].port_type, PortType::Audio);
    }

    #[test]
    fn test_set_parameters() {
        let config = DelayConfig::new(250.0);
        let mut filter = DelayFilter::new(NodeId(0), "test", config);

        filter.set_delay_time(500.0);
        assert!((filter.config().delay_ms - 500.0).abs() < f64::EPSILON);

        filter.set_feedback(0.8);
        assert!((filter.config().feedback - 0.8).abs() < f64::EPSILON);

        filter.set_mix(0.7);
        assert!((filter.config().mix - 0.7).abs() < f64::EPSILON);
    }

    #[test]
    fn test_process_none() {
        let config = DelayConfig::default();
        let mut filter = DelayFilter::new(NodeId(0), "test", config);

        let result = filter.process(None).expect("process should succeed");
        assert!(result.is_none());
    }

    #[test]
    fn test_process_audio() {
        let config = DelayConfig::new(10.0).with_feedback(0.5).with_mix(0.5);
        let mut filter = DelayFilter::new(NodeId(0), "test", config);

        let mut frame = AudioFrame::new(SampleFormat::F32, 48000, ChannelLayout::Stereo);
        let mut samples = BytesMut::new();
        for _ in 0..200 {
            samples.extend_from_slice(&0.5f32.to_le_bytes()); // L
            samples.extend_from_slice(&0.5f32.to_le_bytes()); // R
        }
        frame.samples = AudioBuffer::Interleaved(samples.freeze());

        let result = filter
            .process(Some(FilterFrame::Audio(frame)))
            .expect("process should succeed");
        assert!(result.is_some());
    }

    #[test]
    fn test_state_transitions() {
        let config = DelayConfig::default();
        let mut filter = DelayFilter::new(NodeId(0), "test", config);

        assert!(filter.set_state(NodeState::Processing).is_ok());
        assert_eq!(filter.state(), NodeState::Processing);

        assert!(filter.reset().is_ok());
        assert_eq!(filter.state(), NodeState::Idle);
    }

    #[test]
    fn test_delay_state_reset() {
        let config = DelayConfig::new(250.0);
        let mut state = DelayState::new(&config, 48000.0, 2);

        // Process some samples
        let mut samples = vec![vec![0.5; 100], vec![0.5; 100]];
        state.process(&mut samples, &config);

        // Reset
        state.reset();

        // After reset, ping_pong_state should be false
        assert!(!state.ping_pong_state);
    }
}
