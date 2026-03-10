//! Audio channel mixing filter.
//!
//! This module provides channel mixing and routing capabilities for audio streams,
//! supporting common presets and custom mixing matrices.

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

/// Maximum number of channels supported.
pub const MAX_CHANNELS: usize = 16;

/// A mixing matrix that defines how input channels map to output channels.
///
/// Each row represents an output channel, and each column represents an input channel.
/// The value at `[out][in]` is the gain applied to input channel `in` when contributing
/// to output channel `out`.
#[derive(Clone, Debug)]
pub struct MixMatrix {
    /// The mixing coefficients [output_channel][input_channel].
    coefficients: Vec<Vec<f64>>,
    /// Number of input channels.
    input_channels: usize,
    /// Number of output channels.
    output_channels: usize,
}

impl MixMatrix {
    /// Create a new mixing matrix with the specified dimensions.
    #[must_use]
    pub fn new(input_channels: usize, output_channels: usize) -> Self {
        let coefficients = vec![vec![0.0; input_channels]; output_channels];
        Self {
            coefficients,
            input_channels,
            output_channels,
        }
    }

    /// Create an identity matrix (passthrough).
    #[must_use]
    pub fn identity(channels: usize) -> Self {
        let mut matrix = Self::new(channels, channels);
        for i in 0..channels {
            matrix.coefficients[i][i] = 1.0;
        }
        matrix
    }

    /// Create a mono to stereo matrix (duplicate mono to both channels).
    #[must_use]
    pub fn mono_to_stereo() -> Self {
        let mut matrix = Self::new(1, 2);
        matrix.coefficients[0][0] = 1.0; // Mono -> Left
        matrix.coefficients[1][0] = 1.0; // Mono -> Right
        matrix
    }

    /// Create a stereo to mono matrix (average L and R).
    #[must_use]
    pub fn stereo_to_mono() -> Self {
        let mut matrix = Self::new(2, 1);
        matrix.coefficients[0][0] = 0.5; // Left contribution
        matrix.coefficients[0][1] = 0.5; // Right contribution
        matrix
    }

    /// Create a 5.1 to stereo downmix matrix.
    ///
    /// Standard ITU-R BS.775 downmix coefficients:
    /// - L' = L + 0.707*C + 0.707*Ls
    /// - R' = R + 0.707*C + 0.707*Rs
    #[must_use]
    pub fn surround51_to_stereo() -> Self {
        let mut matrix = Self::new(6, 2);
        let center_gain = 0.707; // -3dB
        let surround_gain = 0.707;

        // Input order: L, R, C, LFE, Ls, Rs
        // Left output
        matrix.coefficients[0][0] = 1.0; // L -> L
        matrix.coefficients[0][2] = center_gain; // C -> L
        matrix.coefficients[0][4] = surround_gain; // Ls -> L

        // Right output
        matrix.coefficients[1][1] = 1.0; // R -> R
        matrix.coefficients[1][2] = center_gain; // C -> R
        matrix.coefficients[1][5] = surround_gain; // Rs -> R

        // LFE is typically discarded in simple downmix

        matrix
    }

    /// Create a 7.1 to stereo downmix matrix.
    #[must_use]
    pub fn surround71_to_stereo() -> Self {
        let mut matrix = Self::new(8, 2);
        let center_gain = 0.707;
        let surround_gain = 0.5;
        let back_gain = 0.5;

        // Input order: L, R, C, LFE, Ls, Rs, Lb, Rb
        // Left output
        matrix.coefficients[0][0] = 1.0; // L -> L
        matrix.coefficients[0][2] = center_gain; // C -> L
        matrix.coefficients[0][4] = surround_gain; // Ls -> L
        matrix.coefficients[0][6] = back_gain; // Lb -> L

        // Right output
        matrix.coefficients[1][1] = 1.0; // R -> R
        matrix.coefficients[1][2] = center_gain; // C -> R
        matrix.coefficients[1][5] = surround_gain; // Rs -> R
        matrix.coefficients[1][7] = back_gain; // Rb -> R

        matrix
    }

    /// Create a stereo to 5.1 upmix matrix.
    ///
    /// Basic upmix that places stereo content in front channels.
    #[must_use]
    pub fn stereo_to_surround51() -> Self {
        let mut matrix = Self::new(2, 6);

        // Output order: L, R, C, LFE, Ls, Rs
        matrix.coefficients[0][0] = 1.0; // L -> L
        matrix.coefficients[1][1] = 1.0; // R -> R
        matrix.coefficients[2][0] = 0.5; // L -> C
        matrix.coefficients[2][1] = 0.5; // R -> C
                                         // LFE and surround channels are silent
        matrix.coefficients[4][0] = 0.3; // L -> Ls (ambient)
        matrix.coefficients[5][1] = 0.3; // R -> Rs (ambient)

        matrix
    }

    /// Set a coefficient in the matrix.
    pub fn set_coefficient(&mut self, output_channel: usize, input_channel: usize, gain: f64) {
        if output_channel < self.output_channels && input_channel < self.input_channels {
            self.coefficients[output_channel][input_channel] = gain;
        }
    }

    /// Get a coefficient from the matrix.
    #[must_use]
    pub fn get_coefficient(&self, output_channel: usize, input_channel: usize) -> f64 {
        if output_channel < self.output_channels && input_channel < self.input_channels {
            self.coefficients[output_channel][input_channel]
        } else {
            0.0
        }
    }

    /// Get the number of input channels.
    #[must_use]
    pub fn input_channels(&self) -> usize {
        self.input_channels
    }

    /// Get the number of output channels.
    #[must_use]
    pub fn output_channels(&self) -> usize {
        self.output_channels
    }

    /// Apply the mixing matrix to input samples.
    fn apply(&self, input: &[Vec<f64>]) -> Vec<Vec<f64>> {
        if input.is_empty() {
            return vec![Vec::new(); self.output_channels];
        }

        let sample_count = input.get(0).map_or(0, Vec::len);
        let mut output = vec![vec![0.0; sample_count]; self.output_channels];

        for out_ch in 0..self.output_channels {
            for sample_idx in 0..sample_count {
                let mut sum = 0.0;
                for in_ch in 0..self.input_channels.min(input.len()) {
                    if sample_idx < input[in_ch].len() {
                        sum += input[in_ch][sample_idx] * self.coefficients[out_ch][in_ch];
                    }
                }
                output[out_ch][sample_idx] = sum;
            }
        }

        output
    }
}

impl Default for MixMatrix {
    fn default() -> Self {
        Self::identity(2)
    }
}

/// Configuration for crossfade transitions.
#[derive(Clone, Debug)]
pub struct CrossfadeConfig {
    /// Duration of crossfade in samples.
    pub duration_samples: usize,
    /// Current position in crossfade (0 = start, duration = end).
    pub position: usize,
    /// Source matrix (fading from).
    pub from_matrix: MixMatrix,
    /// Target matrix (fading to).
    pub to_matrix: MixMatrix,
}

impl CrossfadeConfig {
    /// Create a new crossfade configuration.
    #[must_use]
    pub fn new(from: MixMatrix, to: MixMatrix, duration_samples: usize) -> Self {
        Self {
            duration_samples,
            position: 0,
            from_matrix: from,
            to_matrix: to,
        }
    }

    /// Get the current interpolation factor (0.0 to 1.0).
    #[must_use]
    pub fn factor(&self) -> f64 {
        if self.duration_samples == 0 {
            return 1.0;
        }
        (self.position as f64 / self.duration_samples as f64).min(1.0)
    }

    /// Check if crossfade is complete.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.position >= self.duration_samples
    }

    /// Advance the crossfade position.
    pub fn advance(&mut self, samples: usize) {
        self.position = self.position.saturating_add(samples);
    }
}

/// Configuration for the channel mix filter.
#[derive(Clone, Debug)]
pub struct ChannelMixConfig {
    /// The mixing matrix.
    pub matrix: MixMatrix,
    /// Optional crossfade for smooth transitions.
    pub crossfade: Option<CrossfadeConfig>,
}

impl Default for ChannelMixConfig {
    fn default() -> Self {
        Self {
            matrix: MixMatrix::identity(2),
            crossfade: None,
        }
    }
}

impl ChannelMixConfig {
    /// Create a new configuration with the specified matrix.
    #[must_use]
    pub fn new(matrix: MixMatrix) -> Self {
        Self {
            matrix,
            crossfade: None,
        }
    }

    /// Create a mono to stereo configuration.
    #[must_use]
    pub fn mono_to_stereo() -> Self {
        Self::new(MixMatrix::mono_to_stereo())
    }

    /// Create a stereo to mono configuration.
    #[must_use]
    pub fn stereo_to_mono() -> Self {
        Self::new(MixMatrix::stereo_to_mono())
    }

    /// Create a 5.1 to stereo downmix configuration.
    #[must_use]
    pub fn surround51_to_stereo() -> Self {
        Self::new(MixMatrix::surround51_to_stereo())
    }

    /// Set a crossfade transition.
    #[must_use]
    pub fn with_crossfade(mut self, from: MixMatrix, duration_samples: usize) -> Self {
        self.crossfade = Some(CrossfadeConfig::new(
            from,
            self.matrix.clone(),
            duration_samples,
        ));
        self
    }
}

/// Audio channel mixing filter.
///
/// This filter remaps audio channels using a mixing matrix, supporting
/// common operations like stereo to mono conversion and 5.1 downmixing.
///
/// # Example
///
/// ```ignore
/// use oximedia_graph::filters::audio::mixer::{ChannelMixFilter, ChannelMixConfig};
///
/// // Create a stereo to mono downmix filter
/// let config = ChannelMixConfig::stereo_to_mono();
/// let filter = ChannelMixFilter::new(NodeId(0), "downmix", config);
/// ```
pub struct ChannelMixFilter {
    id: NodeId,
    name: String,
    state: NodeState,
    config: ChannelMixConfig,
    inputs: Vec<InputPort>,
    outputs: Vec<OutputPort>,
}

impl ChannelMixFilter {
    /// Create a new channel mix filter.
    #[must_use]
    pub fn new(id: NodeId, name: impl Into<String>, config: ChannelMixConfig) -> Self {
        let input_format = PortFormat::Audio(
            AudioPortFormat::any().with_channels(config.matrix.input_channels() as u32),
        );
        let output_format = PortFormat::Audio(
            AudioPortFormat::any().with_channels(config.matrix.output_channels() as u32),
        );

        Self {
            id,
            name: name.into(),
            state: NodeState::Idle,
            config,
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
    pub fn config(&self) -> &ChannelMixConfig {
        &self.config
    }

    /// Update the configuration.
    pub fn set_config(&mut self, config: ChannelMixConfig) {
        self.config = config;
    }

    /// Set a new target matrix with crossfade.
    pub fn crossfade_to(&mut self, target: MixMatrix, duration_samples: usize) {
        let from = self.config.matrix.clone();
        self.config.crossfade = Some(CrossfadeConfig::new(from, target.clone(), duration_samples));
        self.config.matrix = target;
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
        output_layout: ChannelLayout,
    ) -> AudioFrame {
        let channel_count = output_layout.count();
        if samples.is_empty() || samples[0].is_empty() || channel_count == 0 {
            return AudioFrame::new(format, sample_rate, output_layout);
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

        let mut frame = AudioFrame::new(format, sample_rate, output_layout);
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

    /// Apply crossfade mixing if active.
    fn apply_crossfade(&mut self, input: &[Vec<f64>]) -> Vec<Vec<f64>> {
        if let Some(ref mut crossfade) = self.config.crossfade {
            if crossfade.is_complete() {
                self.config.crossfade = None;
                return self.config.matrix.apply(input);
            }

            let from_output = crossfade.from_matrix.apply(input);
            let to_output = crossfade.to_matrix.apply(input);
            let _factor = crossfade.factor(); // Used for reference, local_factor is per-sample

            let sample_count = from_output.get(0).map_or(0, Vec::len);
            let output_channels = crossfade.to_matrix.output_channels();
            let mut output = vec![Vec::with_capacity(sample_count); output_channels];

            for sample_idx in 0..sample_count {
                let local_factor = ((crossfade.position + sample_idx) as f64
                    / crossfade.duration_samples as f64)
                    .min(1.0);

                for ch in 0..output_channels {
                    let from_sample = from_output
                        .get(ch)
                        .and_then(|v| v.get(sample_idx))
                        .unwrap_or(&0.0);
                    let to_sample = to_output
                        .get(ch)
                        .and_then(|v| v.get(sample_idx))
                        .unwrap_or(&0.0);
                    let mixed = from_sample * (1.0 - local_factor) + to_sample * local_factor;
                    output[ch].push(mixed);
                }
            }

            crossfade.advance(sample_count);
            output
        } else {
            self.config.matrix.apply(input)
        }
    }
}

impl Node for ChannelMixFilter {
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
        let input_samples = Self::frame_to_samples(&frame);

        // Apply mixing (with crossfade if active)
        let output_samples = self.apply_crossfade(&input_samples);

        // Determine output channel layout
        let output_layout = ChannelLayout::from_count(self.config.matrix.output_channels());

        // Convert back to frame
        let output_frame = Self::samples_to_frame(
            output_samples,
            frame.format,
            frame.sample_rate,
            output_layout,
        );

        Ok(Some(FilterFrame::Audio(output_frame)))
    }

    fn reset(&mut self) -> GraphResult<()> {
        self.config.crossfade = None;
        self.set_state(NodeState::Idle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mix_matrix_identity() {
        let matrix = MixMatrix::identity(2);
        assert_eq!(matrix.input_channels(), 2);
        assert_eq!(matrix.output_channels(), 2);
        assert_eq!(matrix.get_coefficient(0, 0), 1.0);
        assert_eq!(matrix.get_coefficient(1, 1), 1.0);
        assert_eq!(matrix.get_coefficient(0, 1), 0.0);
    }

    #[test]
    fn test_mix_matrix_mono_to_stereo() {
        let matrix = MixMatrix::mono_to_stereo();
        assert_eq!(matrix.input_channels(), 1);
        assert_eq!(matrix.output_channels(), 2);
        assert_eq!(matrix.get_coefficient(0, 0), 1.0);
        assert_eq!(matrix.get_coefficient(1, 0), 1.0);
    }

    #[test]
    fn test_mix_matrix_stereo_to_mono() {
        let matrix = MixMatrix::stereo_to_mono();
        assert_eq!(matrix.input_channels(), 2);
        assert_eq!(matrix.output_channels(), 1);
        assert_eq!(matrix.get_coefficient(0, 0), 0.5);
        assert_eq!(matrix.get_coefficient(0, 1), 0.5);
    }

    #[test]
    fn test_mix_matrix_apply() {
        let matrix = MixMatrix::stereo_to_mono();
        let input = vec![
            vec![1.0, 0.0, -1.0], // Left
            vec![1.0, 0.0, -1.0], // Right
        ];
        let output = matrix.apply(&input);

        assert_eq!(output.len(), 1);
        assert_eq!(output[0].len(), 3);
        assert!((output[0][0] - 1.0).abs() < f64::EPSILON);
        assert!(output[0][1].abs() < f64::EPSILON);
        assert!((output[0][2] + 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_mix_matrix_51_to_stereo() {
        let matrix = MixMatrix::surround51_to_stereo();
        assert_eq!(matrix.input_channels(), 6);
        assert_eq!(matrix.output_channels(), 2);

        // Check that main channels are preserved
        assert_eq!(matrix.get_coefficient(0, 0), 1.0); // L -> L
        assert_eq!(matrix.get_coefficient(1, 1), 1.0); // R -> R
    }

    #[test]
    fn test_crossfade_config() {
        let from = MixMatrix::identity(2);
        let to = MixMatrix::stereo_to_mono();
        let mut crossfade = CrossfadeConfig::new(from, to, 1000);

        assert!(!crossfade.is_complete());
        assert!(crossfade.factor() < f64::EPSILON);

        crossfade.advance(500);
        assert!(!crossfade.is_complete());
        assert!((crossfade.factor() - 0.5).abs() < f64::EPSILON);

        crossfade.advance(500);
        assert!(crossfade.is_complete());
        assert!((crossfade.factor() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_channel_mix_filter_creation() {
        let config = ChannelMixConfig::stereo_to_mono();
        let filter = ChannelMixFilter::new(NodeId(1), "downmix", config);

        assert_eq!(filter.id(), NodeId(1));
        assert_eq!(filter.name(), "downmix");
        assert_eq!(filter.node_type(), NodeType::Filter);
    }

    #[test]
    fn test_channel_mix_filter_ports() {
        let config = ChannelMixConfig::stereo_to_mono();
        let filter = ChannelMixFilter::new(NodeId(0), "test", config);

        assert_eq!(filter.inputs().len(), 1);
        assert_eq!(filter.outputs().len(), 1);
        assert_eq!(filter.inputs()[0].port_type, PortType::Audio);
    }

    #[test]
    fn test_bytes_to_f64() {
        let sample = ChannelMixFilter::bytes_to_f64(&[128], SampleFormat::U8);
        assert!(sample.abs() < 0.01);

        let sample = ChannelMixFilter::bytes_to_f64(&[0, 0], SampleFormat::S16);
        assert!(sample.abs() < f64::EPSILON);
    }

    #[test]
    fn test_f64_to_bytes_roundtrip() {
        let original = 0.5;
        let mut buffer = BytesMut::new();

        ChannelMixFilter::f64_to_bytes(original, SampleFormat::F32, &mut buffer);
        let converted = ChannelMixFilter::bytes_to_f64(&buffer, SampleFormat::F32);

        assert!((original - converted).abs() < 0.0001);
    }

    #[test]
    fn test_process_none() {
        let config = ChannelMixConfig::default();
        let mut filter = ChannelMixFilter::new(NodeId(0), "test", config);

        let result = filter.process(None).expect("process should succeed");
        assert!(result.is_none());
    }

    #[test]
    fn test_process_stereo_to_mono() {
        let config = ChannelMixConfig::stereo_to_mono();
        let mut filter = ChannelMixFilter::new(NodeId(0), "test", config);

        let mut frame = AudioFrame::new(SampleFormat::F32, 48000, ChannelLayout::Stereo);
        let mut samples = BytesMut::new();
        // Create stereo samples: L=0.5, R=0.5
        for _ in 0..100 {
            samples.extend_from_slice(&0.5f32.to_le_bytes()); // L
            samples.extend_from_slice(&0.5f32.to_le_bytes()); // R
        }
        frame.samples = AudioBuffer::Interleaved(samples.freeze());

        let result = filter
            .process(Some(FilterFrame::Audio(frame)))
            .expect("process should succeed");
        assert!(result.is_some());

        if let Some(FilterFrame::Audio(output)) = result {
            assert_eq!(output.channels.count(), 1);
        }
    }

    #[test]
    fn test_crossfade_to() {
        let config = ChannelMixConfig::default();
        let mut filter = ChannelMixFilter::new(NodeId(0), "test", config);

        let target = MixMatrix::stereo_to_mono();
        filter.crossfade_to(target, 1000);

        assert!(filter.config.crossfade.is_some());
    }

    #[test]
    fn test_state_transitions() {
        let config = ChannelMixConfig::default();
        let mut filter = ChannelMixFilter::new(NodeId(0), "test", config);

        assert!(filter.set_state(NodeState::Processing).is_ok());
        assert_eq!(filter.state(), NodeState::Processing);

        assert!(filter.reset().is_ok());
        assert_eq!(filter.state(), NodeState::Idle);
    }

    #[test]
    fn test_set_coefficient() {
        let mut matrix = MixMatrix::new(2, 2);
        matrix.set_coefficient(0, 0, 0.7);
        matrix.set_coefficient(0, 1, 0.3);

        assert!((matrix.get_coefficient(0, 0) - 0.7).abs() < f64::EPSILON);
        assert!((matrix.get_coefficient(0, 1) - 0.3).abs() < f64::EPSILON);
    }

    #[test]
    fn test_stereo_to_surround51() {
        let matrix = MixMatrix::stereo_to_surround51();
        assert_eq!(matrix.input_channels(), 2);
        assert_eq!(matrix.output_channels(), 6);

        // Front channels should have direct mapping
        assert_eq!(matrix.get_coefficient(0, 0), 1.0);
        assert_eq!(matrix.get_coefficient(1, 1), 1.0);
    }

    #[test]
    fn test_surround71_to_stereo() {
        let matrix = MixMatrix::surround71_to_stereo();
        assert_eq!(matrix.input_channels(), 8);
        assert_eq!(matrix.output_channels(), 2);
    }

    #[test]
    fn test_apply_empty_input() {
        let matrix = MixMatrix::identity(2);
        let output = matrix.apply(&[]);
        assert_eq!(output.len(), 2);
        assert!(output[0].is_empty());
    }
}
