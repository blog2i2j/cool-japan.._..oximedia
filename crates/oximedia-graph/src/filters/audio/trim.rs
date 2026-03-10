//! Audio trim filter.
//!
//! This module provides trimming of audio streams with support for
//! start/end time specification and fade in/out at boundaries.

#![forbid(unsafe_code)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::get_first)]
#![allow(clippy::doc_markdown)]

use bytes::{Bytes, BytesMut};

use crate::error::{GraphError, GraphResult};
use crate::frame::FilterFrame;
use crate::node::{Node, NodeId, NodeState, NodeType};
use crate::port::{AudioPortFormat, InputPort, OutputPort, PortFormat, PortId, PortType};

use oximedia_audio::{AudioBuffer, AudioFrame, ChannelLayout};
use oximedia_core::SampleFormat;

/// Trim mode for specifying the trim region.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TrimMode {
    /// Trim by start and end time in seconds.
    TimeRange {
        /// Start time in seconds.
        start: f64,
        /// End time in seconds (None = end of stream).
        end: Option<f64>,
    },
    /// Trim by duration starting from a given time.
    Duration {
        /// Start time in seconds.
        start: f64,
        /// Duration in seconds.
        duration: f64,
    },
    /// Trim by sample count.
    SampleRange {
        /// Start sample.
        start_sample: u64,
        /// End sample (None = end of stream).
        end_sample: Option<u64>,
    },
}

impl Default for TrimMode {
    fn default() -> Self {
        Self::TimeRange {
            start: 0.0,
            end: None,
        }
    }
}

/// Configuration for the trim filter.
#[derive(Clone, Debug)]
pub struct TrimConfig {
    /// Trim mode.
    pub mode: TrimMode,
    /// Fade in duration in milliseconds.
    pub fade_in_ms: f64,
    /// Fade out duration in milliseconds.
    pub fade_out_ms: f64,
}

impl Default for TrimConfig {
    fn default() -> Self {
        Self {
            mode: TrimMode::default(),
            fade_in_ms: 0.0,
            fade_out_ms: 0.0,
        }
    }
}

impl TrimConfig {
    /// Create a new trim configuration with time range.
    #[must_use]
    pub fn time_range(start: f64, end: Option<f64>) -> Self {
        Self {
            mode: TrimMode::TimeRange { start, end },
            ..Default::default()
        }
    }

    /// Create a trim configuration with duration.
    #[must_use]
    pub fn duration(start: f64, duration: f64) -> Self {
        Self {
            mode: TrimMode::Duration { start, duration },
            ..Default::default()
        }
    }

    /// Create a trim configuration with sample range.
    #[must_use]
    pub fn sample_range(start_sample: u64, end_sample: Option<u64>) -> Self {
        Self {
            mode: TrimMode::SampleRange {
                start_sample,
                end_sample,
            },
            ..Default::default()
        }
    }

    /// Set fade in duration.
    #[must_use]
    pub fn with_fade_in(mut self, fade_in_ms: f64) -> Self {
        self.fade_in_ms = fade_in_ms.max(0.0);
        self
    }

    /// Set fade out duration.
    #[must_use]
    pub fn with_fade_out(mut self, fade_out_ms: f64) -> Self {
        self.fade_out_ms = fade_out_ms.max(0.0);
        self
    }

    /// Set fade in and out duration.
    #[must_use]
    pub fn with_fades(mut self, fade_in_ms: f64, fade_out_ms: f64) -> Self {
        self.fade_in_ms = fade_in_ms.max(0.0);
        self.fade_out_ms = fade_out_ms.max(0.0);
        self
    }

    /// Get the start sample for the given sample rate.
    #[must_use]
    pub fn start_sample(&self, sample_rate: u32) -> u64 {
        match self.mode {
            TrimMode::TimeRange { start, .. } | TrimMode::Duration { start, .. } => {
                (start * f64::from(sample_rate)) as u64
            }
            TrimMode::SampleRange { start_sample, .. } => start_sample,
        }
    }

    /// Get the end sample for the given sample rate (None = no limit).
    #[must_use]
    pub fn end_sample(&self, sample_rate: u32) -> Option<u64> {
        match self.mode {
            TrimMode::TimeRange { end, .. } => end.map(|e| (e * f64::from(sample_rate)) as u64),
            TrimMode::Duration { start, duration } => {
                Some(((start + duration) * f64::from(sample_rate)) as u64)
            }
            TrimMode::SampleRange { end_sample, .. } => end_sample,
        }
    }

    /// Get fade in duration in samples.
    #[must_use]
    pub fn fade_in_samples(&self, sample_rate: u32) -> u64 {
        (self.fade_in_ms * 0.001 * f64::from(sample_rate)) as u64
    }

    /// Get fade out duration in samples.
    #[must_use]
    pub fn fade_out_samples(&self, sample_rate: u32) -> u64 {
        (self.fade_out_ms * 0.001 * f64::from(sample_rate)) as u64
    }
}

/// Trim internal state.
struct TrimState {
    /// Current sample position (input stream).
    current_sample: u64,
    /// Start sample.
    start_sample: u64,
    /// End sample (None = no limit).
    end_sample: Option<u64>,
    /// Fade in duration in samples.
    fade_in_samples: u64,
    /// Fade out duration in samples.
    fade_out_samples: u64,
    /// Number of output samples produced.
    output_samples: u64,
    /// Whether end of trim region has been reached.
    done: bool,
}

impl TrimState {
    /// Create new trim state.
    fn new(config: &TrimConfig, sample_rate: u32) -> Self {
        Self {
            current_sample: 0,
            start_sample: config.start_sample(sample_rate),
            end_sample: config.end_sample(sample_rate),
            fade_in_samples: config.fade_in_samples(sample_rate),
            fade_out_samples: config.fade_out_samples(sample_rate),
            output_samples: 0,
            done: false,
        }
    }

    /// Check if we're before the trim region.
    fn before_start(&self) -> bool {
        self.current_sample < self.start_sample
    }

    /// Check if we're past the trim region.
    fn past_end(&self) -> bool {
        if let Some(end) = self.end_sample {
            self.current_sample >= end
        } else {
            false
        }
    }

    /// Calculate fade gain for the current output sample.
    fn fade_gain(&self, output_position: u64) -> f64 {
        // Fade in
        if self.fade_in_samples > 0 && output_position < self.fade_in_samples {
            return output_position as f64 / self.fade_in_samples as f64;
        }

        // Fade out
        if self.fade_out_samples > 0 {
            if let Some(end) = self.end_sample {
                let total_output = end.saturating_sub(self.start_sample);
                let fade_start = total_output.saturating_sub(self.fade_out_samples);
                if output_position >= fade_start {
                    let fade_position = output_position - fade_start;
                    return 1.0 - (fade_position as f64 / self.fade_out_samples as f64);
                }
            }
        }

        1.0
    }

    /// Process samples and return trimmed output.
    fn process(&mut self, samples: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let sample_count = samples.get(0).map_or(0, Vec::len);
        let channels = samples.len();

        if sample_count == 0 || self.done {
            return vec![Vec::new(); channels];
        }

        let mut output = vec![Vec::new(); channels];

        for i in 0..sample_count {
            // Check if before start
            if self.before_start() {
                self.current_sample += 1;
                continue;
            }

            // Check if past end
            if self.past_end() {
                self.done = true;
                break;
            }

            // Apply fade and output
            let fade = self.fade_gain(self.output_samples);

            for ch in 0..channels {
                if i < samples[ch].len() {
                    output[ch].push(samples[ch][i] * fade);
                }
            }

            self.current_sample += 1;
            self.output_samples += 1;
        }

        output
    }

    /// Check if trimming is complete.
    fn is_done(&self) -> bool {
        self.done
    }

    /// Reset state.
    fn reset(&mut self) {
        self.current_sample = 0;
        self.output_samples = 0;
        self.done = false;
    }
}

/// Audio trim filter.
///
/// This filter trims audio streams to a specified time or sample range,
/// with optional fade in/out at the boundaries.
///
/// # Example
///
/// ```ignore
/// use oximedia_graph::filters::audio::trim::{TrimFilter, TrimConfig};
///
/// // Trim to 10-30 seconds with 100ms fades
/// let config = TrimConfig::time_range(10.0, Some(30.0))
///     .with_fades(100.0, 100.0);
/// let filter = TrimFilter::new(NodeId(0), "trim", config);
/// ```
pub struct TrimFilter {
    id: NodeId,
    name: String,
    state: NodeState,
    config: TrimConfig,
    trim_state: Option<TrimState>,
    inputs: Vec<InputPort>,
    outputs: Vec<OutputPort>,
}

impl TrimFilter {
    /// Create a new trim filter.
    #[must_use]
    pub fn new(id: NodeId, name: impl Into<String>, config: TrimConfig) -> Self {
        let audio_format = PortFormat::Audio(AudioPortFormat::any());

        Self {
            id,
            name: name.into(),
            state: NodeState::Idle,
            config,
            trim_state: None,
            inputs: vec![InputPort::new(PortId(0), "input", PortType::Audio)
                .with_format(audio_format.clone())],
            outputs: vec![
                OutputPort::new(PortId(0), "output", PortType::Audio).with_format(audio_format)
            ],
        }
    }

    /// Get the current configuration.
    #[must_use]
    pub fn config(&self) -> &TrimConfig {
        &self.config
    }

    /// Update the configuration.
    pub fn set_config(&mut self, config: TrimConfig) {
        self.config = config;
        self.trim_state = None; // Reset state
    }

    /// Check if trimming is complete.
    #[must_use]
    pub fn is_done(&self) -> bool {
        self.trim_state.as_ref().is_some_and(TrimState::is_done)
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

impl Node for TrimFilter {
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

        // Initialize trim state if needed
        if self.trim_state.is_none() {
            self.trim_state = Some(TrimState::new(&self.config, frame.sample_rate));
        }

        let trim_state = match self.trim_state.as_mut() {
            Some(s) => s,
            None => return Ok(None),
        };

        // Check if we're done
        if trim_state.is_done() {
            return Ok(None);
        }

        // Convert to f64 samples
        let samples = Self::frame_to_samples(&frame);

        // Process through trim
        let output_samples = trim_state.process(&samples);

        // If no output samples, return None
        if output_samples.is_empty() || output_samples[0].is_empty() {
            return Ok(None);
        }

        // Convert back to frame
        let output_frame = Self::samples_to_frame(
            output_samples,
            frame.format,
            frame.sample_rate,
            frame.channels.clone(),
        );

        Ok(Some(FilterFrame::Audio(output_frame)))
    }

    fn reset(&mut self) -> GraphResult<()> {
        if let Some(ref mut state) = self.trim_state {
            state.reset();
        }
        self.set_state(NodeState::Idle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trim_mode_default() {
        let mode = TrimMode::default();
        if let TrimMode::TimeRange { start, end } = mode {
            assert!(start.abs() < f64::EPSILON);
            assert!(end.is_none());
        } else {
            panic!("Expected TimeRange mode");
        }
    }

    #[test]
    fn test_trim_config_time_range() {
        let config = TrimConfig::time_range(10.0, Some(30.0));

        assert_eq!(config.start_sample(48000), 480000);
        assert_eq!(config.end_sample(48000), Some(1440000));
    }

    #[test]
    fn test_trim_config_duration() {
        let config = TrimConfig::duration(5.0, 10.0);

        assert_eq!(config.start_sample(48000), 240000);
        assert_eq!(config.end_sample(48000), Some(720000)); // 5 + 10 = 15 seconds
    }

    #[test]
    fn test_trim_config_sample_range() {
        let config = TrimConfig::sample_range(1000, Some(5000));

        assert_eq!(config.start_sample(48000), 1000);
        assert_eq!(config.end_sample(48000), Some(5000));
    }

    #[test]
    fn test_fade_settings() {
        let config = TrimConfig::time_range(0.0, Some(10.0))
            .with_fade_in(100.0)
            .with_fade_out(200.0);

        assert!((config.fade_in_ms - 100.0).abs() < f64::EPSILON);
        assert!((config.fade_out_ms - 200.0).abs() < f64::EPSILON);

        assert_eq!(config.fade_in_samples(48000), 4800);
        assert_eq!(config.fade_out_samples(48000), 9600);
    }

    #[test]
    fn test_fade_settings_combined() {
        let config = TrimConfig::time_range(0.0, Some(10.0)).with_fades(50.0, 100.0);

        assert!((config.fade_in_ms - 50.0).abs() < f64::EPSILON);
        assert!((config.fade_out_ms - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_trim_state_before_start() {
        let config = TrimConfig::time_range(1.0, Some(2.0));
        let state = TrimState::new(&config, 48000);

        assert!(state.before_start());
        assert!(!state.past_end());
    }

    #[test]
    fn test_fade_gain_calculation() {
        let config = TrimConfig::time_range(0.0, Some(1.0)).with_fade_in(100.0);
        let state = TrimState::new(&config, 48000);

        // At start, fade should be 0
        let fade = state.fade_gain(0);
        assert!(fade.abs() < f64::EPSILON);

        // At half fade in, should be 0.5
        let fade = state.fade_gain(2400); // Half of 4800
        assert!((fade - 0.5).abs() < 0.01);

        // After fade in, should be 1.0
        let fade = state.fade_gain(5000);
        assert!((fade - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_trim_filter_creation() {
        let config = TrimConfig::time_range(0.0, Some(10.0));
        let filter = TrimFilter::new(NodeId(1), "trim", config);

        assert_eq!(filter.id(), NodeId(1));
        assert_eq!(filter.name(), "trim");
        assert_eq!(filter.node_type(), NodeType::Filter);
    }

    #[test]
    fn test_trim_filter_ports() {
        let config = TrimConfig::default();
        let filter = TrimFilter::new(NodeId(0), "test", config);

        assert_eq!(filter.inputs().len(), 1);
        assert_eq!(filter.outputs().len(), 1);
        assert_eq!(filter.inputs()[0].port_type, PortType::Audio);
    }

    #[test]
    fn test_process_none() {
        let config = TrimConfig::default();
        let mut filter = TrimFilter::new(NodeId(0), "test", config);

        let result = filter.process(None).expect("process should succeed");
        assert!(result.is_none());
    }

    #[test]
    fn test_process_audio() {
        // Trim starting from 0, no end limit
        let config = TrimConfig::time_range(0.0, None);
        let mut filter = TrimFilter::new(NodeId(0), "test", config);

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

        if let Some(FilterFrame::Audio(output)) = result {
            assert_eq!(output.sample_count(), 100);
        }
    }

    #[test]
    fn test_trim_before_start() {
        // Start at 1 second
        let config = TrimConfig::time_range(1.0, None);
        let mut filter = TrimFilter::new(NodeId(0), "test", config);

        // First frame is 100 samples at 48000 Hz (< 1 second)
        let mut frame = AudioFrame::new(SampleFormat::F32, 48000, ChannelLayout::Mono);
        let mut samples = BytesMut::new();
        for _ in 0..100 {
            samples.extend_from_slice(&0.5f32.to_le_bytes());
        }
        frame.samples = AudioBuffer::Interleaved(samples.freeze());

        let result = filter
            .process(Some(FilterFrame::Audio(frame)))
            .expect("process should succeed");
        // Should return None because we're before the start
        assert!(result.is_none());
    }

    #[test]
    fn test_is_done() {
        let config = TrimConfig::sample_range(0, Some(50));
        let mut filter = TrimFilter::new(NodeId(0), "test", config);

        assert!(!filter.is_done());

        // Process more samples than the trim region
        let mut frame = AudioFrame::new(SampleFormat::F32, 48000, ChannelLayout::Mono);
        let mut samples = BytesMut::new();
        for _ in 0..100 {
            samples.extend_from_slice(&0.5f32.to_le_bytes());
        }
        frame.samples = AudioBuffer::Interleaved(samples.freeze());

        let _ = filter.process(Some(FilterFrame::Audio(frame)));

        assert!(filter.is_done());
    }

    #[test]
    fn test_state_transitions() {
        let config = TrimConfig::default();
        let mut filter = TrimFilter::new(NodeId(0), "test", config);

        assert!(filter.set_state(NodeState::Processing).is_ok());
        assert_eq!(filter.state(), NodeState::Processing);

        assert!(filter.reset().is_ok());
        assert_eq!(filter.state(), NodeState::Idle);
    }

    #[test]
    fn test_trim_state_reset() {
        let config = TrimConfig::time_range(0.0, Some(1.0));
        let mut state = TrimState::new(&config, 48000);

        state.current_sample = 1000;
        state.output_samples = 500;
        state.done = true;

        state.reset();

        assert_eq!(state.current_sample, 0);
        assert_eq!(state.output_samples, 0);
        assert!(!state.done);
    }
}
