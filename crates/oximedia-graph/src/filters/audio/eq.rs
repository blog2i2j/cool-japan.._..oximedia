//! Audio equalizer filter.
//!
//! This module provides a parametric equalizer with multiple band types
//! using biquad filter coefficients.

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

/// Maximum number of EQ bands supported.
pub const MAX_BANDS: usize = 10;

/// Type of EQ band filter.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum BandType {
    /// Low shelf filter - boost/cut below frequency.
    LowShelf,
    /// High shelf filter - boost/cut above frequency.
    HighShelf,
    /// Peaking filter - boost/cut around center frequency.
    #[default]
    Peaking,
    /// Low pass filter - attenuate above frequency.
    LowPass,
    /// High pass filter - attenuate below frequency.
    HighPass,
    /// Band pass filter - pass around center frequency.
    BandPass,
    /// Notch filter - attenuate around center frequency.
    Notch,
    /// All pass filter - phase shift only.
    AllPass,
}

/// A single EQ band configuration.
#[derive(Clone, Debug)]
pub struct EqBand {
    /// Type of filter.
    pub band_type: BandType,
    /// Center frequency in Hz.
    pub frequency: f64,
    /// Gain in dB (for peaking and shelf filters).
    pub gain_db: f64,
    /// Q factor (bandwidth control).
    pub q: f64,
    /// Whether this band is enabled.
    pub enabled: bool,
}

impl Default for EqBand {
    fn default() -> Self {
        Self {
            band_type: BandType::Peaking,
            frequency: 1000.0,
            gain_db: 0.0,
            q: 1.0,
            enabled: true,
        }
    }
}

impl EqBand {
    /// Create a new EQ band.
    #[must_use]
    pub fn new(band_type: BandType, frequency: f64, gain_db: f64, q: f64) -> Self {
        Self {
            band_type,
            frequency,
            gain_db,
            q,
            enabled: true,
        }
    }

    /// Create a low shelf band.
    #[must_use]
    pub fn low_shelf(frequency: f64, gain_db: f64) -> Self {
        Self::new(BandType::LowShelf, frequency, gain_db, 0.707)
    }

    /// Create a high shelf band.
    #[must_use]
    pub fn high_shelf(frequency: f64, gain_db: f64) -> Self {
        Self::new(BandType::HighShelf, frequency, gain_db, 0.707)
    }

    /// Create a peaking band.
    #[must_use]
    pub fn peaking(frequency: f64, gain_db: f64, q: f64) -> Self {
        Self::new(BandType::Peaking, frequency, gain_db, q)
    }

    /// Create a low pass band.
    #[must_use]
    pub fn low_pass(frequency: f64, q: f64) -> Self {
        Self::new(BandType::LowPass, frequency, 0.0, q)
    }

    /// Create a high pass band.
    #[must_use]
    pub fn high_pass(frequency: f64, q: f64) -> Self {
        Self::new(BandType::HighPass, frequency, 0.0, q)
    }

    /// Create a notch filter band.
    #[must_use]
    pub fn notch(frequency: f64, q: f64) -> Self {
        Self::new(BandType::Notch, frequency, 0.0, q)
    }

    /// Set the enabled state.
    #[must_use]
    pub fn enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }
}

/// Biquad filter coefficients.
#[derive(Clone, Debug, Default)]
struct BiquadCoefficients {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
}

impl BiquadCoefficients {
    /// Calculate coefficients for the given band type.
    fn calculate(band: &EqBand, sample_rate: f64) -> Self {
        let w0 = 2.0 * PI * band.frequency / sample_rate;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let alpha = sin_w0 / (2.0 * band.q);
        let a = 10.0_f64.powf(band.gain_db / 40.0);

        let (b0, b1, b2, a0, a1, a2) = match band.band_type {
            BandType::LowShelf => {
                let two_sqrt_a_alpha = 2.0 * a.sqrt() * alpha;
                let a_plus_1 = a + 1.0;
                let a_minus_1 = a - 1.0;

                let b0 = a * (a_plus_1 - a_minus_1 * cos_w0 + two_sqrt_a_alpha);
                let b1 = 2.0 * a * (a_minus_1 - a_plus_1 * cos_w0);
                let b2 = a * (a_plus_1 - a_minus_1 * cos_w0 - two_sqrt_a_alpha);
                let a0 = a_plus_1 + a_minus_1 * cos_w0 + two_sqrt_a_alpha;
                let a1 = -2.0 * (a_minus_1 + a_plus_1 * cos_w0);
                let a2 = a_plus_1 + a_minus_1 * cos_w0 - two_sqrt_a_alpha;

                (b0, b1, b2, a0, a1, a2)
            }
            BandType::HighShelf => {
                let two_sqrt_a_alpha = 2.0 * a.sqrt() * alpha;
                let a_plus_1 = a + 1.0;
                let a_minus_1 = a - 1.0;

                let b0 = a * (a_plus_1 + a_minus_1 * cos_w0 + two_sqrt_a_alpha);
                let b1 = -2.0 * a * (a_minus_1 + a_plus_1 * cos_w0);
                let b2 = a * (a_plus_1 + a_minus_1 * cos_w0 - two_sqrt_a_alpha);
                let a0 = a_plus_1 - a_minus_1 * cos_w0 + two_sqrt_a_alpha;
                let a1 = 2.0 * (a_minus_1 - a_plus_1 * cos_w0);
                let a2 = a_plus_1 - a_minus_1 * cos_w0 - two_sqrt_a_alpha;

                (b0, b1, b2, a0, a1, a2)
            }
            BandType::Peaking => {
                let alpha_a = alpha * a;
                let alpha_over_a = alpha / a;

                let b0 = 1.0 + alpha_a;
                let b1 = -2.0 * cos_w0;
                let b2 = 1.0 - alpha_a;
                let a0 = 1.0 + alpha_over_a;
                let a1 = -2.0 * cos_w0;
                let a2 = 1.0 - alpha_over_a;

                (b0, b1, b2, a0, a1, a2)
            }
            BandType::LowPass => {
                let b0 = (1.0 - cos_w0) / 2.0;
                let b1 = 1.0 - cos_w0;
                let b2 = (1.0 - cos_w0) / 2.0;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_w0;
                let a2 = 1.0 - alpha;

                (b0, b1, b2, a0, a1, a2)
            }
            BandType::HighPass => {
                let b0 = (1.0 + cos_w0) / 2.0;
                let b1 = -(1.0 + cos_w0);
                let b2 = (1.0 + cos_w0) / 2.0;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_w0;
                let a2 = 1.0 - alpha;

                (b0, b1, b2, a0, a1, a2)
            }
            BandType::BandPass => {
                let b0 = alpha;
                let b1 = 0.0;
                let b2 = -alpha;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_w0;
                let a2 = 1.0 - alpha;

                (b0, b1, b2, a0, a1, a2)
            }
            BandType::Notch => {
                let b0 = 1.0;
                let b1 = -2.0 * cos_w0;
                let b2 = 1.0;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_w0;
                let a2 = 1.0 - alpha;

                (b0, b1, b2, a0, a1, a2)
            }
            BandType::AllPass => {
                let b0 = 1.0 - alpha;
                let b1 = -2.0 * cos_w0;
                let b2 = 1.0 + alpha;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_w0;
                let a2 = 1.0 - alpha;

                (b0, b1, b2, a0, a1, a2)
            }
        };

        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        }
    }
}

/// Biquad filter state for one channel.
#[derive(Clone, Debug, Default)]
struct BiquadState {
    x1: f64,
    x2: f64,
    y1: f64,
    y2: f64,
}

impl BiquadState {
    /// Process one sample through the biquad filter.
    fn process(&mut self, x: f64, coeffs: &BiquadCoefficients) -> f64 {
        let y = coeffs.b0 * x + coeffs.b1 * self.x1 + coeffs.b2 * self.x2
            - coeffs.a1 * self.y1
            - coeffs.a2 * self.y2;

        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = y;

        y
    }

    /// Reset filter state.
    fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }
}

/// Configuration for the equalizer filter.
#[derive(Clone, Debug, Default)]
pub struct EqualizerConfig {
    /// EQ bands.
    pub bands: Vec<EqBand>,
}

impl EqualizerConfig {
    /// Create a new equalizer configuration.
    #[must_use]
    pub fn new() -> Self {
        Self { bands: Vec::new() }
    }

    /// Add a band.
    #[must_use]
    pub fn with_band(mut self, band: EqBand) -> Self {
        if self.bands.len() < MAX_BANDS {
            self.bands.push(band);
        }
        self
    }

    /// Create a 3-band EQ preset.
    #[must_use]
    pub fn three_band(low_gain: f64, mid_gain: f64, high_gain: f64) -> Self {
        Self::new()
            .with_band(EqBand::low_shelf(250.0, low_gain))
            .with_band(EqBand::peaking(1000.0, mid_gain, 1.0))
            .with_band(EqBand::high_shelf(4000.0, high_gain))
    }

    /// Create a 10-band graphic EQ preset.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn graphic_10_band(
        g31: f64,
        g62: f64,
        g125: f64,
        g250: f64,
        g500: f64,
        g1k: f64,
        g2k: f64,
        g4k: f64,
        g8k: f64,
        g16k: f64,
    ) -> Self {
        let q = 1.414; // ~1 octave bandwidth
        Self::new()
            .with_band(EqBand::peaking(31.0, g31, q))
            .with_band(EqBand::peaking(62.0, g62, q))
            .with_band(EqBand::peaking(125.0, g125, q))
            .with_band(EqBand::peaking(250.0, g250, q))
            .with_band(EqBand::peaking(500.0, g500, q))
            .with_band(EqBand::peaking(1000.0, g1k, q))
            .with_band(EqBand::peaking(2000.0, g2k, q))
            .with_band(EqBand::peaking(4000.0, g4k, q))
            .with_band(EqBand::peaking(8000.0, g8k, q))
            .with_band(EqBand::peaking(16000.0, g16k, q))
    }
}

/// Equalizer filter state.
struct EqualizerState {
    /// Biquad coefficients per band.
    coefficients: Vec<BiquadCoefficients>,
    /// Biquad state per band per channel.
    states: Vec<Vec<BiquadState>>,
    /// Sample rate used for coefficient calculation.
    sample_rate: f64,
}

impl EqualizerState {
    /// Create new equalizer state.
    fn new(config: &EqualizerConfig, sample_rate: f64, channels: usize) -> Self {
        let coefficients: Vec<_> = config
            .bands
            .iter()
            .map(|band| BiquadCoefficients::calculate(band, sample_rate))
            .collect();

        let states = vec![vec![BiquadState::default(); config.bands.len()]; channels];

        Self {
            coefficients,
            states,
            sample_rate,
        }
    }

    /// Update coefficients for changed bands.
    fn update_coefficients(&mut self, config: &EqualizerConfig) {
        self.coefficients.clear();
        for band in &config.bands {
            self.coefficients
                .push(BiquadCoefficients::calculate(band, self.sample_rate));
        }

        // Adjust state arrays
        let band_count = config.bands.len();
        for channel_states in &mut self.states {
            while channel_states.len() < band_count {
                channel_states.push(BiquadState::default());
            }
            channel_states.truncate(band_count);
        }
    }

    /// Process samples through all bands.
    fn process(&mut self, samples: &mut [Vec<f64>], bands: &[EqBand]) {
        for (ch, channel) in samples.iter_mut().enumerate() {
            if ch >= self.states.len() {
                break;
            }

            for sample in channel.iter_mut() {
                let mut value = *sample;

                for (band_idx, band) in bands.iter().enumerate() {
                    if !band.enabled {
                        continue;
                    }

                    if band_idx < self.coefficients.len() && band_idx < self.states[ch].len() {
                        value =
                            self.states[ch][band_idx].process(value, &self.coefficients[band_idx]);
                    }
                }

                *sample = value;
            }
        }
    }

    /// Reset all filter states.
    fn reset(&mut self) {
        for channel_states in &mut self.states {
            for state in channel_states {
                state.reset();
            }
        }
    }
}

/// Audio parametric equalizer filter.
///
/// This filter provides a multi-band parametric equalizer with various
/// filter types including peaking, shelf, and pass filters.
///
/// # Example
///
/// ```ignore
/// use oximedia_graph::filters::audio::eq::{EqualizerFilter, EqualizerConfig, EqBand, BandType};
///
/// // Create a 3-band EQ with bass boost and treble cut
/// let config = EqualizerConfig::three_band(6.0, 0.0, -3.0);
/// let filter = EqualizerFilter::new(NodeId(0), "eq", config);
/// ```
pub struct EqualizerFilter {
    id: NodeId,
    name: String,
    state: NodeState,
    config: EqualizerConfig,
    eq_state: Option<EqualizerState>,
    inputs: Vec<InputPort>,
    outputs: Vec<OutputPort>,
}

impl EqualizerFilter {
    /// Create a new equalizer filter.
    #[must_use]
    pub fn new(id: NodeId, name: impl Into<String>, config: EqualizerConfig) -> Self {
        let audio_format = PortFormat::Audio(AudioPortFormat::any());

        Self {
            id,
            name: name.into(),
            state: NodeState::Idle,
            config,
            eq_state: None,
            inputs: vec![InputPort::new(PortId(0), "input", PortType::Audio)
                .with_format(audio_format.clone())],
            outputs: vec![
                OutputPort::new(PortId(0), "output", PortType::Audio).with_format(audio_format)
            ],
        }
    }

    /// Get the current configuration.
    #[must_use]
    pub fn config(&self) -> &EqualizerConfig {
        &self.config
    }

    /// Update the configuration.
    pub fn set_config(&mut self, config: EqualizerConfig) {
        self.config = config;
        if let Some(ref mut eq_state) = self.eq_state {
            eq_state.update_coefficients(&self.config);
        }
    }

    /// Add a band.
    pub fn add_band(&mut self, band: EqBand) {
        if self.config.bands.len() < MAX_BANDS {
            self.config.bands.push(band);
            if let Some(ref mut eq_state) = self.eq_state {
                eq_state.update_coefficients(&self.config);
            }
        }
    }

    /// Remove a band by index.
    pub fn remove_band(&mut self, index: usize) {
        if index < self.config.bands.len() {
            self.config.bands.remove(index);
            if let Some(ref mut eq_state) = self.eq_state {
                eq_state.update_coefficients(&self.config);
            }
        }
    }

    /// Set band parameters.
    pub fn set_band(&mut self, index: usize, band: EqBand) {
        if index < self.config.bands.len() {
            self.config.bands[index] = band;
            if let Some(ref mut eq_state) = self.eq_state {
                eq_state.update_coefficients(&self.config);
            }
        }
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

impl Node for EqualizerFilter {
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

        // Initialize EQ state if needed
        if self.eq_state.is_none() {
            let channels = frame.channels.count();
            self.eq_state = Some(EqualizerState::new(
                &self.config,
                f64::from(frame.sample_rate),
                channels,
            ));
        }

        // Convert to f64 samples
        let mut samples = Self::frame_to_samples(&frame);

        // Apply EQ processing
        if let Some(ref mut eq_state) = self.eq_state {
            eq_state.process(&mut samples, &self.config.bands);
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
        if let Some(ref mut eq_state) = self.eq_state {
            eq_state.reset();
        }
        self.set_state(NodeState::Idle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_band_type_default() {
        assert_eq!(BandType::default(), BandType::Peaking);
    }

    #[test]
    fn test_eq_band_creation() {
        let band = EqBand::new(BandType::Peaking, 1000.0, 6.0, 1.0);
        assert_eq!(band.band_type, BandType::Peaking);
        assert!((band.frequency - 1000.0).abs() < f64::EPSILON);
        assert!((band.gain_db - 6.0).abs() < f64::EPSILON);
        assert!(band.enabled);
    }

    #[test]
    fn test_eq_band_presets() {
        let low_shelf = EqBand::low_shelf(250.0, 3.0);
        assert_eq!(low_shelf.band_type, BandType::LowShelf);

        let high_shelf = EqBand::high_shelf(4000.0, -3.0);
        assert_eq!(high_shelf.band_type, BandType::HighShelf);

        let peaking = EqBand::peaking(1000.0, 6.0, 2.0);
        assert_eq!(peaking.band_type, BandType::Peaking);

        let low_pass = EqBand::low_pass(8000.0, 0.707);
        assert_eq!(low_pass.band_type, BandType::LowPass);

        let high_pass = EqBand::high_pass(80.0, 0.707);
        assert_eq!(high_pass.band_type, BandType::HighPass);

        let notch = EqBand::notch(60.0, 10.0);
        assert_eq!(notch.band_type, BandType::Notch);
    }

    #[test]
    fn test_eq_band_enabled() {
        let band = EqBand::peaking(1000.0, 6.0, 1.0).enabled(false);
        assert!(!band.enabled);
    }

    #[test]
    fn test_equalizer_config() {
        let config = EqualizerConfig::new()
            .with_band(EqBand::low_shelf(250.0, 3.0))
            .with_band(EqBand::peaking(1000.0, 0.0, 1.0))
            .with_band(EqBand::high_shelf(4000.0, -3.0));

        assert_eq!(config.bands.len(), 3);
    }

    #[test]
    fn test_three_band_preset() {
        let config = EqualizerConfig::three_band(3.0, 0.0, -3.0);
        assert_eq!(config.bands.len(), 3);
        assert_eq!(config.bands[0].band_type, BandType::LowShelf);
        assert_eq!(config.bands[1].band_type, BandType::Peaking);
        assert_eq!(config.bands[2].band_type, BandType::HighShelf);
    }

    #[test]
    fn test_graphic_10_band() {
        let config =
            EqualizerConfig::graphic_10_band(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert_eq!(config.bands.len(), 10);
    }

    #[test]
    fn test_biquad_coefficients() {
        let band = EqBand::peaking(1000.0, 6.0, 1.0);
        let coeffs = BiquadCoefficients::calculate(&band, 48000.0);

        // Coefficients should be finite
        assert!(coeffs.b0.is_finite());
        assert!(coeffs.b1.is_finite());
        assert!(coeffs.b2.is_finite());
        assert!(coeffs.a1.is_finite());
        assert!(coeffs.a2.is_finite());
    }

    #[test]
    fn test_biquad_state() {
        let band = EqBand::peaking(1000.0, 0.0, 1.0); // Unity gain
        let coeffs = BiquadCoefficients::calculate(&band, 48000.0);
        let mut state = BiquadState::default();

        // Process some samples
        for _ in 0..100 {
            let output = state.process(0.5, &coeffs);
            assert!(output.is_finite());
        }

        // Reset should clear state
        state.reset();
        assert!(state.x1.abs() < f64::EPSILON);
        assert!(state.y1.abs() < f64::EPSILON);
    }

    #[test]
    fn test_equalizer_filter_creation() {
        let config = EqualizerConfig::three_band(0.0, 0.0, 0.0);
        let filter = EqualizerFilter::new(NodeId(1), "eq", config);

        assert_eq!(filter.id(), NodeId(1));
        assert_eq!(filter.name(), "eq");
        assert_eq!(filter.node_type(), NodeType::Filter);
    }

    #[test]
    fn test_equalizer_filter_ports() {
        let config = EqualizerConfig::default();
        let filter = EqualizerFilter::new(NodeId(0), "test", config);

        assert_eq!(filter.inputs().len(), 1);
        assert_eq!(filter.outputs().len(), 1);
        assert_eq!(filter.inputs()[0].port_type, PortType::Audio);
    }

    #[test]
    fn test_add_remove_band() {
        let config = EqualizerConfig::new();
        let mut filter = EqualizerFilter::new(NodeId(0), "test", config);

        filter.add_band(EqBand::peaking(1000.0, 6.0, 1.0));
        assert_eq!(filter.config().bands.len(), 1);

        filter.remove_band(0);
        assert!(filter.config().bands.is_empty());
    }

    #[test]
    fn test_set_band() {
        let config = EqualizerConfig::new().with_band(EqBand::peaking(1000.0, 0.0, 1.0));
        let mut filter = EqualizerFilter::new(NodeId(0), "test", config);

        filter.set_band(0, EqBand::peaking(2000.0, 6.0, 2.0));

        assert!((filter.config().bands[0].frequency - 2000.0).abs() < f64::EPSILON);
        assert!((filter.config().bands[0].gain_db - 6.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_process_none() {
        let config = EqualizerConfig::default();
        let mut filter = EqualizerFilter::new(NodeId(0), "test", config);

        let result = filter.process(None).expect("process should succeed");
        assert!(result.is_none());
    }

    #[test]
    fn test_process_audio() {
        let config = EqualizerConfig::three_band(0.0, 0.0, 0.0); // Flat EQ
        let mut filter = EqualizerFilter::new(NodeId(0), "test", config);

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
    fn test_state_transitions() {
        let config = EqualizerConfig::default();
        let mut filter = EqualizerFilter::new(NodeId(0), "test", config);

        assert!(filter.set_state(NodeState::Processing).is_ok());
        assert_eq!(filter.state(), NodeState::Processing);

        assert!(filter.reset().is_ok());
        assert_eq!(filter.state(), NodeState::Idle);
    }

    #[test]
    fn test_all_band_types_coefficients() {
        let sample_rate = 48000.0;
        let test_bands = vec![
            EqBand::new(BandType::LowShelf, 250.0, 6.0, 0.707),
            EqBand::new(BandType::HighShelf, 4000.0, 6.0, 0.707),
            EqBand::new(BandType::Peaking, 1000.0, 6.0, 1.0),
            EqBand::new(BandType::LowPass, 8000.0, 0.0, 0.707),
            EqBand::new(BandType::HighPass, 80.0, 0.0, 0.707),
            EqBand::new(BandType::BandPass, 1000.0, 0.0, 1.0),
            EqBand::new(BandType::Notch, 60.0, 0.0, 10.0),
            EqBand::new(BandType::AllPass, 1000.0, 0.0, 0.707),
        ];

        for band in &test_bands {
            let coeffs = BiquadCoefficients::calculate(band, sample_rate);
            assert!(
                coeffs.b0.is_finite(),
                "b0 not finite for {:?}",
                band.band_type
            );
            assert!(
                coeffs.b1.is_finite(),
                "b1 not finite for {:?}",
                band.band_type
            );
            assert!(
                coeffs.b2.is_finite(),
                "b2 not finite for {:?}",
                band.band_type
            );
            assert!(
                coeffs.a1.is_finite(),
                "a1 not finite for {:?}",
                band.band_type
            );
            assert!(
                coeffs.a2.is_finite(),
                "a2 not finite for {:?}",
                band.band_type
            );
        }
    }

    #[test]
    fn test_max_bands_limit() {
        let mut config = EqualizerConfig::new();
        for i in 0..MAX_BANDS + 5 {
            config = config.with_band(EqBand::peaking(100.0 * (i + 1) as f64, 0.0, 1.0));
        }

        assert_eq!(config.bands.len(), MAX_BANDS);
    }
}
