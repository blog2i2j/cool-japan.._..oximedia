//! Audio volume control filter.
//!
//! This module provides volume adjustment with support for linear and dB gain,
//! fade in/out, peak normalization, and soft clipping.

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

/// Fade direction for volume transitions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FadeDirection {
    /// Fading in (silence to full volume).
    In,
    /// Fading out (full volume to silence).
    Out,
}

/// Configuration for volume fade.
#[derive(Clone, Debug)]
pub struct FadeConfig {
    /// Fade direction.
    pub direction: FadeDirection,
    /// Duration in samples.
    pub duration_samples: usize,
    /// Current position in fade.
    pub position: usize,
    /// Whether fade is active.
    pub active: bool,
}

impl FadeConfig {
    /// Create a new fade configuration.
    #[must_use]
    pub fn new(direction: FadeDirection, duration_samples: usize) -> Self {
        Self {
            direction,
            duration_samples,
            position: 0,
            active: true,
        }
    }

    /// Create a fade in configuration.
    #[must_use]
    pub fn fade_in(duration_samples: usize) -> Self {
        Self::new(FadeDirection::In, duration_samples)
    }

    /// Create a fade out configuration.
    #[must_use]
    pub fn fade_out(duration_samples: usize) -> Self {
        Self::new(FadeDirection::Out, duration_samples)
    }

    /// Get the gain multiplier at the current position.
    #[must_use]
    pub fn gain_at_position(&self, sample_offset: usize) -> f64 {
        if !self.active || self.duration_samples == 0 {
            return 1.0;
        }

        let pos = (self.position + sample_offset).min(self.duration_samples);
        let t = pos as f64 / self.duration_samples as f64;

        match self.direction {
            FadeDirection::In => t,
            FadeDirection::Out => 1.0 - t,
        }
    }

    /// Check if the fade is complete.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.position >= self.duration_samples
    }

    /// Advance the fade position.
    pub fn advance(&mut self, samples: usize) {
        self.position = self.position.saturating_add(samples);
        if self.is_complete() {
            self.active = false;
        }
    }
}

/// Configuration for the volume filter.
#[derive(Clone, Debug)]
pub struct VolumeConfig {
    /// Gain in linear scale (1.0 = unity gain).
    pub gain: f64,
    /// Optional fade configuration.
    pub fade: Option<FadeConfig>,
    /// Enable peak normalization.
    pub normalize_peak: bool,
    /// Target peak level for normalization (linear, typically 1.0).
    pub target_peak: f64,
    /// Enable soft clipping for overflow protection.
    pub soft_clip: bool,
    /// Soft clip threshold (linear).
    pub soft_clip_threshold: f64,
}

impl Default for VolumeConfig {
    fn default() -> Self {
        Self {
            gain: 1.0,
            fade: None,
            normalize_peak: false,
            target_peak: 1.0,
            soft_clip: false,
            soft_clip_threshold: 0.9,
        }
    }
}

impl VolumeConfig {
    /// Create a new volume configuration with the specified gain.
    #[must_use]
    pub fn new(gain: f64) -> Self {
        Self {
            gain,
            ..Default::default()
        }
    }

    /// Create a configuration from dB gain.
    #[must_use]
    pub fn from_db(db: f64) -> Self {
        Self::new(Self::db_to_linear(db))
    }

    /// Set gain in dB.
    #[must_use]
    pub fn with_db_gain(mut self, db: f64) -> Self {
        self.gain = Self::db_to_linear(db);
        self
    }

    /// Set fade in.
    #[must_use]
    pub fn with_fade_in(mut self, duration_samples: usize) -> Self {
        self.fade = Some(FadeConfig::fade_in(duration_samples));
        self
    }

    /// Set fade out.
    #[must_use]
    pub fn with_fade_out(mut self, duration_samples: usize) -> Self {
        self.fade = Some(FadeConfig::fade_out(duration_samples));
        self
    }

    /// Enable peak normalization.
    #[must_use]
    pub fn with_peak_normalization(mut self, target_peak: f64) -> Self {
        self.normalize_peak = true;
        self.target_peak = target_peak;
        self
    }

    /// Enable soft clipping.
    #[must_use]
    pub fn with_soft_clip(mut self, threshold: f64) -> Self {
        self.soft_clip = true;
        self.soft_clip_threshold = threshold;
        self
    }

    /// Convert dB to linear gain.
    #[must_use]
    pub fn db_to_linear(db: f64) -> f64 {
        10.0_f64.powf(db / 20.0)
    }

    /// Convert linear gain to dB.
    #[must_use]
    pub fn linear_to_db(linear: f64) -> f64 {
        if linear <= 0.0 {
            f64::NEG_INFINITY
        } else {
            20.0 * linear.log10()
        }
    }
}

/// Audio volume control filter.
///
/// This filter adjusts audio volume with support for:
/// - Linear and dB gain control
/// - Fade in/out effects
/// - Peak normalization
/// - Soft clipping for overflow protection
///
/// # Example
///
/// ```ignore
/// use oximedia_graph::filters::audio::volume::{VolumeFilter, VolumeConfig};
///
/// // Create a filter with -6dB gain and fade in
/// let config = VolumeConfig::from_db(-6.0)
///     .with_fade_in(48000); // 1 second fade at 48kHz
/// let filter = VolumeFilter::new(NodeId(0), "volume", config);
/// ```
pub struct VolumeFilter {
    id: NodeId,
    name: String,
    state: NodeState,
    config: VolumeConfig,
    inputs: Vec<InputPort>,
    outputs: Vec<OutputPort>,
}

impl VolumeFilter {
    /// Create a new volume filter.
    #[must_use]
    pub fn new(id: NodeId, name: impl Into<String>, config: VolumeConfig) -> Self {
        let audio_format = PortFormat::Audio(AudioPortFormat::any());

        Self {
            id,
            name: name.into(),
            state: NodeState::Idle,
            config,
            inputs: vec![InputPort::new(PortId(0), "input", PortType::Audio)
                .with_format(audio_format.clone())],
            outputs: vec![
                OutputPort::new(PortId(0), "output", PortType::Audio).with_format(audio_format)
            ],
        }
    }

    /// Get the current configuration.
    #[must_use]
    pub fn config(&self) -> &VolumeConfig {
        &self.config
    }

    /// Set the gain (linear).
    pub fn set_gain(&mut self, gain: f64) {
        self.config.gain = gain;
    }

    /// Set the gain in dB.
    pub fn set_gain_db(&mut self, db: f64) {
        self.config.gain = VolumeConfig::db_to_linear(db);
    }

    /// Start a fade in.
    pub fn start_fade_in(&mut self, duration_samples: usize) {
        self.config.fade = Some(FadeConfig::fade_in(duration_samples));
    }

    /// Start a fade out.
    pub fn start_fade_out(&mut self, duration_samples: usize) {
        self.config.fade = Some(FadeConfig::fade_out(duration_samples));
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

    /// Apply soft clipping to a sample.
    fn soft_clip(sample: f64, threshold: f64) -> f64 {
        if sample.abs() <= threshold {
            sample
        } else {
            let sign = sample.signum();
            let excess = sample.abs() - threshold;
            let range = 1.0 - threshold;
            // Soft saturation using tanh
            let compressed = threshold + range * (excess / range).tanh();
            sign * compressed
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

    /// Apply volume processing to samples.
    fn process_samples(&mut self, samples: &mut [Vec<f64>]) {
        let sample_count = samples.get(0).map_or(0, Vec::len);

        // Calculate normalization gain if enabled
        let norm_gain = if self.config.normalize_peak {
            let peak = Self::find_peak(samples);
            if peak > 0.0 {
                self.config.target_peak / peak
            } else {
                1.0
            }
        } else {
            1.0
        };

        for sample_idx in 0..sample_count {
            // Calculate fade gain
            let fade_gain = if let Some(ref fade) = self.config.fade {
                fade.gain_at_position(sample_idx)
            } else {
                1.0
            };

            // Combined gain
            let total_gain = self.config.gain * norm_gain * fade_gain;

            // Apply to all channels
            for channel in samples.iter_mut() {
                if sample_idx < channel.len() {
                    let mut sample = channel[sample_idx] * total_gain;

                    // Apply soft clipping if enabled
                    if self.config.soft_clip {
                        sample = Self::soft_clip(sample, self.config.soft_clip_threshold);
                    }

                    channel[sample_idx] = sample;
                }
            }
        }

        // Advance fade position
        if let Some(ref mut fade) = self.config.fade {
            fade.advance(sample_count);
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

impl Node for VolumeFilter {
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

        // Convert to f64 samples
        let mut samples = Self::frame_to_samples(&frame);

        // Apply volume processing
        self.process_samples(&mut samples);

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
        self.config.fade = None;
        self.set_state(NodeState::Idle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_db_to_linear() {
        let linear = VolumeConfig::db_to_linear(0.0);
        assert!((linear - 1.0).abs() < f64::EPSILON);

        let linear = VolumeConfig::db_to_linear(-6.0);
        assert!((linear - 0.501).abs() < 0.01);

        let linear = VolumeConfig::db_to_linear(6.0);
        assert!((linear - 1.995).abs() < 0.01);
    }

    #[test]
    fn test_linear_to_db() {
        let db = VolumeConfig::linear_to_db(1.0);
        assert!(db.abs() < f64::EPSILON);

        let db = VolumeConfig::linear_to_db(0.5);
        assert!((db - (-6.02)).abs() < 0.1);

        let db = VolumeConfig::linear_to_db(0.0);
        assert!(db.is_infinite() && db.is_sign_negative());
    }

    #[test]
    fn test_fade_config() {
        let mut fade = FadeConfig::fade_in(1000);
        assert!(fade.active);
        assert!(!fade.is_complete());
        assert!(fade.gain_at_position(0).abs() < f64::EPSILON);
        assert!((fade.gain_at_position(500) - 0.5).abs() < f64::EPSILON);
        assert!((fade.gain_at_position(1000) - 1.0).abs() < f64::EPSILON);

        fade.advance(500);
        assert!(!fade.is_complete());

        fade.advance(500);
        assert!(fade.is_complete());
        assert!(!fade.active);
    }

    #[test]
    fn test_fade_out() {
        let fade = FadeConfig::fade_out(1000);
        assert!((fade.gain_at_position(0) - 1.0).abs() < f64::EPSILON);
        assert!((fade.gain_at_position(500) - 0.5).abs() < f64::EPSILON);
        assert!(fade.gain_at_position(1000).abs() < f64::EPSILON);
    }

    #[test]
    fn test_volume_config() {
        let config = VolumeConfig::from_db(-6.0)
            .with_fade_in(1000)
            .with_peak_normalization(0.9)
            .with_soft_clip(0.8);

        assert!((config.gain - 0.501).abs() < 0.01);
        assert!(config.fade.is_some());
        assert!(config.normalize_peak);
        assert!((config.target_peak - 0.9).abs() < f64::EPSILON);
        assert!(config.soft_clip);
        assert!((config.soft_clip_threshold - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn test_soft_clip() {
        let result = VolumeFilter::soft_clip(0.5, 0.9);
        assert!((result - 0.5).abs() < f64::EPSILON);

        let result = VolumeFilter::soft_clip(1.5, 0.9);
        assert!(result > 0.9);
        assert!(result < 1.0);

        let result = VolumeFilter::soft_clip(-1.5, 0.9);
        assert!(result < -0.9);
        assert!(result > -1.0);
    }

    #[test]
    fn test_find_peak() {
        let samples = vec![vec![0.5, -0.8, 0.3], vec![0.2, 0.9, -0.1]];
        let peak = VolumeFilter::find_peak(&samples);
        assert!((peak - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn test_volume_filter_creation() {
        let config = VolumeConfig::new(0.5);
        let filter = VolumeFilter::new(NodeId(1), "volume", config);

        assert_eq!(filter.id(), NodeId(1));
        assert_eq!(filter.name(), "volume");
        assert_eq!(filter.node_type(), NodeType::Filter);
    }

    #[test]
    fn test_volume_filter_ports() {
        let config = VolumeConfig::default();
        let filter = VolumeFilter::new(NodeId(0), "test", config);

        assert_eq!(filter.inputs().len(), 1);
        assert_eq!(filter.outputs().len(), 1);
        assert_eq!(filter.inputs()[0].port_type, PortType::Audio);
    }

    #[test]
    fn test_set_gain() {
        let config = VolumeConfig::default();
        let mut filter = VolumeFilter::new(NodeId(0), "test", config);

        filter.set_gain(0.5);
        assert!((filter.config().gain - 0.5).abs() < f64::EPSILON);

        filter.set_gain_db(-6.0);
        assert!((filter.config().gain - 0.501).abs() < 0.01);
    }

    #[test]
    fn test_start_fade() {
        let config = VolumeConfig::default();
        let mut filter = VolumeFilter::new(NodeId(0), "test", config);

        filter.start_fade_in(1000);
        assert!(filter.config().fade.is_some());
        assert_eq!(
            filter
                .config()
                .fade
                .as_ref()
                .expect("as_ref should succeed")
                .direction,
            FadeDirection::In
        );

        filter.start_fade_out(2000);
        assert_eq!(
            filter
                .config()
                .fade
                .as_ref()
                .expect("as_ref should succeed")
                .direction,
            FadeDirection::Out
        );
    }

    #[test]
    fn test_process_none() {
        let config = VolumeConfig::default();
        let mut filter = VolumeFilter::new(NodeId(0), "test", config);

        let result = filter.process(None).expect("process should succeed");
        assert!(result.is_none());
    }

    #[test]
    fn test_process_with_gain() {
        let config = VolumeConfig::new(0.5);
        let mut filter = VolumeFilter::new(NodeId(0), "test", config);

        let mut frame = AudioFrame::new(SampleFormat::F32, 48000, ChannelLayout::Mono);
        let mut samples = BytesMut::new();
        samples.extend_from_slice(&1.0f32.to_le_bytes());
        frame.samples = AudioBuffer::Interleaved(samples.freeze());

        let result = filter
            .process(Some(FilterFrame::Audio(frame)))
            .expect("process should succeed");
        assert!(result.is_some());

        if let Some(FilterFrame::Audio(output)) = result {
            if let AudioBuffer::Interleaved(data) = &output.samples {
                let sample = f32::from_le_bytes([data[0], data[1], data[2], data[3]]);
                assert!((sample - 0.5).abs() < 0.01);
            }
        }
    }

    #[test]
    fn test_state_transitions() {
        let config = VolumeConfig::default();
        let mut filter = VolumeFilter::new(NodeId(0), "test", config);

        assert!(filter.set_state(NodeState::Processing).is_ok());
        assert_eq!(filter.state(), NodeState::Processing);

        assert!(filter.reset().is_ok());
        assert_eq!(filter.state(), NodeState::Idle);
        assert!(filter.config().fade.is_none());
    }

    #[test]
    fn test_bytes_conversion_roundtrip() {
        let original = 0.5;
        let mut buffer = BytesMut::new();

        VolumeFilter::f64_to_bytes(original, SampleFormat::F32, &mut buffer);
        let converted = VolumeFilter::bytes_to_f64(&buffer, SampleFormat::F32);

        assert!((original - converted).abs() < 0.0001);
    }

    #[test]
    fn test_peak_normalization() {
        let config = VolumeConfig::new(1.0).with_peak_normalization(1.0);
        let mut filter = VolumeFilter::new(NodeId(0), "test", config);

        let mut frame = AudioFrame::new(SampleFormat::F32, 48000, ChannelLayout::Mono);
        let mut samples = BytesMut::new();
        // Sample with peak at 0.5
        samples.extend_from_slice(&0.5f32.to_le_bytes());
        frame.samples = AudioBuffer::Interleaved(samples.freeze());

        let result = filter
            .process(Some(FilterFrame::Audio(frame)))
            .expect("process should succeed");
        assert!(result.is_some());

        if let Some(FilterFrame::Audio(output)) = result {
            if let AudioBuffer::Interleaved(data) = &output.samples {
                let sample = f32::from_le_bytes([data[0], data[1], data[2], data[3]]);
                // Should be normalized to 1.0
                assert!((sample - 1.0).abs() < 0.01);
            }
        }
    }
}
