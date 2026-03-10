//! Audio resampling filter.
//!
//! This module provides high-quality sample rate conversion using sinc interpolation
//! with windowed kernels.

#![forbid(unsafe_code)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::get_first)]
#![allow(clippy::doc_markdown)]

use bytes::{Bytes, BytesMut};
use std::f64::consts::PI;

use crate::error::{GraphError, GraphResult};
use crate::frame::FilterFrame;
use crate::node::{Node, NodeId, NodeState, NodeType};
use crate::port::{AudioPortFormat, InputPort, OutputPort, PortFormat, PortId, PortType};

use oximedia_audio::{AudioBuffer, AudioFrame, ChannelLayout};
use oximedia_core::SampleFormat;

/// Quality preset for resampling.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ResampleQuality {
    /// Fast resampling with lower quality (8 taps).
    Fast,
    /// Medium quality (16 taps).
    #[default]
    Medium,
    /// High quality (32 taps).
    High,
    /// Very high quality (64 taps).
    VeryHigh,
}

impl ResampleQuality {
    /// Get the number of filter taps for this quality level.
    #[must_use]
    pub fn taps(self) -> usize {
        match self {
            Self::Fast => 8,
            Self::Medium => 16,
            Self::High => 32,
            Self::VeryHigh => 64,
        }
    }

    /// Get Kaiser window beta parameter for this quality level.
    #[must_use]
    pub fn kaiser_beta(self) -> f64 {
        match self {
            Self::Fast => 5.0,
            Self::Medium => 6.0,
            Self::High => 8.0,
            Self::VeryHigh => 10.0,
        }
    }
}

/// Configuration for the resample filter.
#[derive(Clone, Debug)]
pub struct ResampleConfig {
    /// Input sample rate in Hz.
    pub input_rate: u32,
    /// Output sample rate in Hz.
    pub output_rate: u32,
    /// Quality preset.
    pub quality: ResampleQuality,
    /// Anti-aliasing filter enabled.
    pub anti_alias: bool,
}

impl Default for ResampleConfig {
    fn default() -> Self {
        Self {
            input_rate: 48000,
            output_rate: 44100,
            quality: ResampleQuality::Medium,
            anti_alias: true,
        }
    }
}

impl ResampleConfig {
    /// Create a new resample configuration.
    #[must_use]
    pub fn new(input_rate: u32, output_rate: u32) -> Self {
        Self {
            input_rate,
            output_rate,
            ..Default::default()
        }
    }

    /// Set the quality preset.
    #[must_use]
    pub fn with_quality(mut self, quality: ResampleQuality) -> Self {
        self.quality = quality;
        self
    }

    /// Enable or disable anti-aliasing.
    #[must_use]
    pub fn with_anti_alias(mut self, enabled: bool) -> Self {
        self.anti_alias = enabled;
        self
    }
}

/// Sinc interpolation kernel.
#[derive(Clone, Debug)]
struct SincKernel {
    /// Precomputed filter coefficients.
    coefficients: Vec<f64>,
    /// Number of taps.
    taps: usize,
    /// Kaiser beta parameter.
    beta: f64,
    /// Cutoff frequency (normalized).
    cutoff: f64,
}

impl SincKernel {
    /// Create a new sinc kernel.
    fn new(taps: usize, beta: f64, cutoff: f64) -> Self {
        let coefficients = Self::compute_coefficients(taps, beta, cutoff);
        Self {
            coefficients,
            taps,
            beta,
            cutoff,
        }
    }

    /// Compute filter coefficients.
    fn compute_coefficients(taps: usize, beta: f64, cutoff: f64) -> Vec<f64> {
        let half_taps = taps / 2;
        let mut coeffs = Vec::with_capacity(taps);

        for i in 0..taps {
            let x = (i as f64 - half_taps as f64) / half_taps as f64;
            let sinc = Self::sinc(x * cutoff * half_taps as f64);
            let window = Self::kaiser_window(x, beta);
            coeffs.push(sinc * window * cutoff);
        }

        // Normalize coefficients
        let sum: f64 = coeffs.iter().sum();
        if sum.abs() > f64::EPSILON {
            for coeff in &mut coeffs {
                *coeff /= sum;
            }
        }

        coeffs
    }

    /// Sinc function: sin(pi * x) / (pi * x).
    fn sinc(x: f64) -> f64 {
        if x.abs() < f64::EPSILON {
            1.0
        } else {
            let px = PI * x;
            px.sin() / px
        }
    }

    /// Kaiser window function.
    fn kaiser_window(x: f64, beta: f64) -> f64 {
        if x.abs() >= 1.0 {
            return 0.0;
        }
        let arg = 1.0 - x * x;
        if arg < 0.0 {
            return 0.0;
        }
        Self::bessel_i0(beta * arg.sqrt()) / Self::bessel_i0(beta)
    }

    /// Modified Bessel function of the first kind, order 0.
    fn bessel_i0(x: f64) -> f64 {
        let mut sum = 1.0;
        let mut term = 1.0;
        let x_half = x / 2.0;

        for k in 1..=25 {
            term *= x_half * x_half / (k * k) as f64;
            sum += term;
            if term < f64::EPSILON * sum {
                break;
            }
        }

        sum
    }

    /// Interpolate samples at a fractional position.
    fn interpolate(&self, samples: &[f64], position: f64) -> f64 {
        let base_index = position.floor() as isize;
        let frac = position - position.floor();
        let half_taps = (self.taps / 2) as isize;

        let mut result = 0.0;
        for (tap, coeff) in self.coefficients.iter().enumerate() {
            let tap_offset = tap as isize - half_taps;
            let sample_index = base_index + tap_offset;

            if sample_index >= 0 && (sample_index as usize) < samples.len() {
                // Apply phase-adjusted coefficient
                let phase_adjust = Self::sinc((tap_offset as f64 - frac) * self.cutoff);
                let window = Self::kaiser_window(
                    (tap as f64 - self.taps as f64 / 2.0) / (self.taps as f64 / 2.0),
                    self.beta,
                );
                result += samples[sample_index as usize] * coeff * phase_adjust * window;
            }
        }

        result
    }
}

/// Internal state for resampling.
#[derive(Clone, Debug)]
struct ResampleState {
    /// Input sample buffer for overlap.
    input_buffer: Vec<Vec<f64>>,
    /// Current input position (fractional).
    position: f64,
    /// Ratio of input to output rate.
    ratio: f64,
    /// Sinc interpolation kernel.
    kernel: SincKernel,
    /// Number of channels.
    channels: usize,
}

impl ResampleState {
    /// Create new resample state.
    fn new(config: &ResampleConfig, channels: usize) -> Self {
        let ratio = config.input_rate as f64 / config.output_rate as f64;
        let cutoff = if config.anti_alias && ratio > 1.0 {
            1.0 / ratio * 0.95 // Apply 5% guard band
        } else {
            0.95
        };

        let kernel = SincKernel::new(config.quality.taps(), config.quality.kaiser_beta(), cutoff);

        Self {
            input_buffer: vec![Vec::new(); channels],
            position: 0.0,
            ratio,
            kernel,
            channels,
        }
    }

    /// Process samples through the resampler.
    fn process(&mut self, input: &[Vec<f64>]) -> Vec<Vec<f64>> {
        if input.is_empty() || input[0].is_empty() {
            return vec![Vec::new(); self.channels];
        }

        // Append input to buffer
        for (ch, samples) in input.iter().enumerate() {
            if ch < self.channels {
                self.input_buffer[ch].extend_from_slice(samples);
            }
        }

        // Calculate output sample count
        let available_input = self.input_buffer[0].len() as f64 - self.position;
        let output_samples = (available_input / self.ratio).floor() as usize;

        if output_samples == 0 {
            return vec![Vec::new(); self.channels];
        }

        // Resample each channel
        let mut output = vec![Vec::with_capacity(output_samples); self.channels];

        for out_idx in 0..output_samples {
            let in_pos = self.position + out_idx as f64 * self.ratio;

            for ch in 0..self.channels {
                let sample = self.kernel.interpolate(&self.input_buffer[ch], in_pos);
                output[ch].push(sample);
            }
        }

        // Update position and trim consumed samples
        self.position += output_samples as f64 * self.ratio;
        let consumed = self.position.floor() as usize;

        // Keep overlap for next block
        let keep = self.kernel.taps;
        if consumed > keep {
            let trim = consumed - keep;
            for ch in 0..self.channels {
                if trim < self.input_buffer[ch].len() {
                    self.input_buffer[ch].drain(0..trim);
                }
            }
            self.position -= trim as f64;
        }

        output
    }

    /// Flush remaining samples.
    fn flush(&mut self) -> Vec<Vec<f64>> {
        // Pad with zeros for final samples
        let padding = self.kernel.taps;
        for ch in 0..self.channels {
            self.input_buffer[ch].extend(vec![0.0; padding]);
        }

        // Process remaining samples
        let input: Vec<Vec<f64>> = vec![Vec::new(); self.channels];
        let output = self.process(&input);

        // Reset state
        for ch in 0..self.channels {
            self.input_buffer[ch].clear();
        }
        self.position = 0.0;

        output
    }
}

/// Audio resampling filter using sinc interpolation.
///
/// This filter converts audio between different sample rates using
/// high-quality sinc interpolation with a Kaiser window.
///
/// # Example
///
/// ```ignore
/// use oximedia_graph::filters::audio::resample::{ResampleFilter, ResampleConfig, ResampleQuality};
///
/// let config = ResampleConfig::new(48000, 44100)
///     .with_quality(ResampleQuality::High);
/// let filter = ResampleFilter::new(NodeId(0), "resample", config);
/// ```
pub struct ResampleFilter {
    id: NodeId,
    name: String,
    state: NodeState,
    config: ResampleConfig,
    resample_state: Option<ResampleState>,
    inputs: Vec<InputPort>,
    outputs: Vec<OutputPort>,
}

impl ResampleFilter {
    /// Create a new resample filter.
    #[must_use]
    pub fn new(id: NodeId, name: impl Into<String>, config: ResampleConfig) -> Self {
        let input_format =
            PortFormat::Audio(AudioPortFormat::any().with_sample_rate(config.input_rate));
        let output_format =
            PortFormat::Audio(AudioPortFormat::any().with_sample_rate(config.output_rate));

        Self {
            id,
            name: name.into(),
            state: NodeState::Idle,
            config,
            resample_state: None,
            inputs: vec![
                InputPort::new(PortId(0), "input", PortType::Audio).with_format(input_format)
            ],
            outputs: vec![
                OutputPort::new(PortId(0), "output", PortType::Audio).with_format(output_format)
            ],
        }
    }

    /// Get the current configuration.
    #[must_use]
    pub fn config(&self) -> &ResampleConfig {
        &self.config
    }

    /// Update the configuration.
    pub fn set_config(&mut self, config: ResampleConfig) {
        self.config = config;
        self.resample_state = None; // Reset state
    }

    /// Convert audio frame samples to f64 vectors (one per channel).
    fn frame_to_samples(frame: &AudioFrame) -> Vec<Vec<f64>> {
        let channels = frame.channels.count();
        let sample_count = frame.sample_count();

        if sample_count == 0 {
            return vec![Vec::new(); channels];
        }

        let mut output = vec![Vec::with_capacity(sample_count); channels];

        match &frame.samples {
            AudioBuffer::Interleaved(data) => {
                Self::convert_interleaved_to_f64(data, frame.format, channels, &mut output);
            }
            AudioBuffer::Planar(planes) => {
                Self::convert_planar_to_f64(planes, frame.format, &mut output);
            }
        }

        output
    }

    /// Convert interleaved samples to f64 vectors.
    fn convert_interleaved_to_f64(
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

    /// Convert planar samples to f64 vectors.
    fn convert_planar_to_f64(planes: &[Bytes], format: SampleFormat, output: &mut [Vec<f64>]) {
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

    /// Convert f64 samples back to audio frame.
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

        // Convert to interleaved format
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
        // Clamp sample to valid range
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

impl Node for ResampleFilter {
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

    fn initialize(&mut self) -> GraphResult<()> {
        // State will be initialized on first frame when we know channel count
        Ok(())
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

        // Initialize resample state if needed
        if self.resample_state.is_none() {
            let channels = frame.channels.count();
            self.resample_state = Some(ResampleState::new(&self.config, channels));
        }

        let state = match self.resample_state.as_mut() {
            Some(s) => s,
            None => return Ok(None),
        };

        // Convert to f64 samples
        let input_samples = Self::frame_to_samples(&frame);

        // Resample
        let output_samples = state.process(&input_samples);

        // Convert back to frame
        let output_frame = Self::samples_to_frame(
            output_samples,
            frame.format,
            self.config.output_rate,
            frame.channels.clone(),
        );

        Ok(Some(FilterFrame::Audio(output_frame)))
    }

    fn flush(&mut self) -> GraphResult<Vec<FilterFrame>> {
        if let Some(state) = self.resample_state.as_mut() {
            let output_samples = state.flush();
            if !output_samples.is_empty() && !output_samples[0].is_empty() {
                let frame = Self::samples_to_frame(
                    output_samples,
                    SampleFormat::F32,
                    self.config.output_rate,
                    ChannelLayout::from_count(state.channels),
                );
                return Ok(vec![FilterFrame::Audio(frame)]);
            }
        }
        Ok(Vec::new())
    }

    fn reset(&mut self) -> GraphResult<()> {
        self.resample_state = None;
        self.set_state(NodeState::Idle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resample_quality() {
        assert_eq!(ResampleQuality::Fast.taps(), 8);
        assert_eq!(ResampleQuality::Medium.taps(), 16);
        assert_eq!(ResampleQuality::High.taps(), 32);
        assert_eq!(ResampleQuality::VeryHigh.taps(), 64);
    }

    #[test]
    fn test_resample_config() {
        let config = ResampleConfig::new(48000, 44100)
            .with_quality(ResampleQuality::High)
            .with_anti_alias(true);

        assert_eq!(config.input_rate, 48000);
        assert_eq!(config.output_rate, 44100);
        assert_eq!(config.quality, ResampleQuality::High);
        assert!(config.anti_alias);
    }

    #[test]
    fn test_sinc_function() {
        let sinc_0 = SincKernel::sinc(0.0);
        assert!((sinc_0 - 1.0).abs() < f64::EPSILON);

        let sinc_1 = SincKernel::sinc(1.0);
        assert!(sinc_1.abs() < 0.01);
    }

    #[test]
    fn test_kaiser_window() {
        let center = SincKernel::kaiser_window(0.0, 6.0);
        assert!((center - 1.0).abs() < 0.01);

        let edge = SincKernel::kaiser_window(1.0, 6.0);
        assert!(edge.abs() < f64::EPSILON);
    }

    #[test]
    fn test_bessel_i0() {
        let i0_0 = SincKernel::bessel_i0(0.0);
        assert!((i0_0 - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_resample_filter_creation() {
        let config = ResampleConfig::new(48000, 44100);
        let filter = ResampleFilter::new(NodeId(1), "test_resample", config);

        assert_eq!(filter.id(), NodeId(1));
        assert_eq!(filter.name(), "test_resample");
        assert_eq!(filter.node_type(), NodeType::Filter);
        assert_eq!(filter.state(), NodeState::Idle);
    }

    #[test]
    fn test_resample_filter_ports() {
        let config = ResampleConfig::new(48000, 44100);
        let filter = ResampleFilter::new(NodeId(0), "test", config);

        assert_eq!(filter.inputs().len(), 1);
        assert_eq!(filter.outputs().len(), 1);
        assert_eq!(filter.inputs()[0].port_type, PortType::Audio);
        assert_eq!(filter.outputs()[0].port_type, PortType::Audio);
    }

    #[test]
    fn test_bytes_to_f64_u8() {
        let sample = ResampleFilter::bytes_to_f64(&[128], SampleFormat::U8);
        assert!(sample.abs() < 0.01);

        let sample = ResampleFilter::bytes_to_f64(&[255], SampleFormat::U8);
        assert!((sample - 0.992).abs() < 0.01);
    }

    #[test]
    fn test_bytes_to_f64_s16() {
        let sample = ResampleFilter::bytes_to_f64(&[0, 0], SampleFormat::S16);
        assert!(sample.abs() < f64::EPSILON);

        let sample = ResampleFilter::bytes_to_f64(&[0xFF, 0x7F], SampleFormat::S16);
        assert!((sample - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_bytes_to_f64_f32() {
        let value: f32 = 0.5;
        let bytes = value.to_le_bytes();
        let sample = ResampleFilter::bytes_to_f64(&bytes, SampleFormat::F32);
        assert!((sample - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_resample_process_empty() {
        let config = ResampleConfig::new(48000, 44100);
        let mut filter = ResampleFilter::new(NodeId(0), "test", config);

        let result = filter.process(None).expect("process should succeed");
        assert!(result.is_none());
    }

    #[test]
    fn test_resample_process_audio() {
        let config = ResampleConfig::new(48000, 48000); // Same rate for simple test
        let mut filter = ResampleFilter::new(NodeId(0), "test", config);

        let mut frame = AudioFrame::new(SampleFormat::F32, 48000, ChannelLayout::Mono);
        // Create some test samples
        let mut samples = BytesMut::new();
        for i in 0..1024 {
            let sample = (i as f32 * 0.001).sin();
            samples.extend_from_slice(&sample.to_le_bytes());
        }
        frame.samples = AudioBuffer::Interleaved(samples.freeze());

        let result = filter
            .process(Some(FilterFrame::Audio(frame)))
            .expect("process should succeed");
        assert!(result.is_some());
    }

    #[test]
    fn test_state_transitions() {
        let config = ResampleConfig::new(48000, 44100);
        let mut filter = ResampleFilter::new(NodeId(0), "test", config);

        assert!(filter.set_state(NodeState::Processing).is_ok());
        assert_eq!(filter.state(), NodeState::Processing);

        assert!(filter.set_state(NodeState::Done).is_ok());
        assert_eq!(filter.state(), NodeState::Done);

        assert!(filter.reset().is_ok());
        assert_eq!(filter.state(), NodeState::Idle);
    }

    #[test]
    fn test_sinc_kernel_creation() {
        let kernel = SincKernel::new(16, 6.0, 0.95);
        assert_eq!(kernel.taps, 16);
        assert_eq!(kernel.coefficients.len(), 16);

        // Coefficients should be normalized
        let sum: f64 = kernel.coefficients.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_resample_state_creation() {
        let config = ResampleConfig::new(48000, 44100);
        let state = ResampleState::new(&config, 2);

        assert_eq!(state.channels, 2);
        assert!(state.ratio > 1.0); // Downsampling
    }

    #[test]
    fn test_f64_conversion_roundtrip() {
        let original: f64 = 0.5;
        let mut buffer = BytesMut::new();

        ResampleFilter::f64_to_bytes(original, SampleFormat::F32, &mut buffer);
        let converted = ResampleFilter::bytes_to_f64(&buffer, SampleFormat::F32);

        assert!((original - converted).abs() < 0.0001);
    }

    #[test]
    fn test_sample_clamping() {
        let mut buffer = BytesMut::new();

        // Test clamping of out-of-range values
        ResampleFilter::f64_to_bytes(2.0, SampleFormat::F32, &mut buffer);
        let clamped = ResampleFilter::bytes_to_f64(&buffer, SampleFormat::F32);
        assert!((clamped - 1.0).abs() < f64::EPSILON);

        buffer.clear();
        ResampleFilter::f64_to_bytes(-2.0, SampleFormat::F32, &mut buffer);
        let clamped = ResampleFilter::bytes_to_f64(&buffer, SampleFormat::F32);
        assert!((clamped + 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_resample_different_rates() {
        let config = ResampleConfig::new(48000, 24000); // 2:1 downsampling
        let mut filter = ResampleFilter::new(NodeId(0), "test", config);

        let mut frame = AudioFrame::new(SampleFormat::F32, 48000, ChannelLayout::Mono);
        let mut samples = BytesMut::new();
        for i in 0..4800 {
            let sample = (i as f64 * 0.01).sin() as f32;
            samples.extend_from_slice(&sample.to_le_bytes());
        }
        frame.samples = AudioBuffer::Interleaved(samples.freeze());

        let result = filter
            .process(Some(FilterFrame::Audio(frame)))
            .expect("process should succeed");
        assert!(result.is_some());

        if let Some(FilterFrame::Audio(output)) = result {
            // Output should have approximately half the samples
            let output_count = output.sample_count();
            assert!(output_count > 0);
            assert!(output_count < 4800);
        }
    }

    #[test]
    fn test_flush() {
        let config = ResampleConfig::new(48000, 44100);
        let mut filter = ResampleFilter::new(NodeId(0), "test", config);

        // Process some samples first
        let mut frame = AudioFrame::new(SampleFormat::F32, 48000, ChannelLayout::Mono);
        let mut samples = BytesMut::new();
        for _ in 0..1024 {
            samples.extend_from_slice(&0.5f32.to_le_bytes());
        }
        frame.samples = AudioBuffer::Interleaved(samples.freeze());

        let _ = filter.process(Some(FilterFrame::Audio(frame)));

        // Flush should work without error
        let flushed = filter.flush().expect("flush should succeed");
        // May or may not have remaining samples depending on state
        assert!(flushed.len() <= 1);
    }
}
