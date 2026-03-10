#![allow(dead_code)]
//! Advanced audio effects processing.

use crate::error::{AudioPostError, AudioPostResult};
use std::collections::VecDeque;

/// Multiband compressor
#[derive(Debug)]
pub struct MultibandCompressor {
    sample_rate: u32,
    bands: Vec<CompressorBand>,
}

impl MultibandCompressor {
    /// Create a new multiband compressor with specified number of bands
    ///
    /// # Errors
    ///
    /// Returns an error if sample rate or band count is invalid
    pub fn new(sample_rate: u32, num_bands: usize) -> AudioPostResult<Self> {
        if sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(sample_rate));
        }
        if num_bands < 2 || num_bands > 6 {
            return Err(AudioPostError::Generic(
                "Band count must be 2-6".to_string(),
            ));
        }

        let bands = (0..num_bands).map(|_| CompressorBand::new()).collect();

        Ok(Self { sample_rate, bands })
    }

    /// Get band count
    #[must_use]
    pub fn band_count(&self) -> usize {
        self.bands.len()
    }

    /// Get a band
    #[must_use]
    pub fn get_band(&self, index: usize) -> Option<&CompressorBand> {
        self.bands.get(index)
    }

    /// Get a mutable band
    pub fn get_band_mut(&mut self, index: usize) -> Option<&mut CompressorBand> {
        self.bands.get_mut(index)
    }
}

/// Compressor band
#[derive(Debug, Clone)]
pub struct CompressorBand {
    /// Threshold in dB
    pub threshold: f32,
    /// Ratio
    pub ratio: f32,
    /// Attack time in ms
    pub attack_ms: f32,
    /// Release time in ms
    pub release_ms: f32,
    /// Enabled flag
    pub enabled: bool,
}

impl CompressorBand {
    /// Create a new compressor band
    #[must_use]
    pub fn new() -> Self {
        Self {
            threshold: -20.0,
            ratio: 4.0,
            attack_ms: 5.0,
            release_ms: 50.0,
            enabled: true,
        }
    }
}

impl Default for CompressorBand {
    fn default() -> Self {
        Self::new()
    }
}

/// De-esser for reducing sibilance
#[derive(Debug)]
pub struct DeEsser {
    sample_rate: u32,
    threshold: f32,
    frequency: f32,
}

impl DeEsser {
    /// Create a new de-esser
    ///
    /// # Errors
    ///
    /// Returns an error if sample rate is invalid
    pub fn new(sample_rate: u32) -> AudioPostResult<Self> {
        if sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(sample_rate));
        }

        Ok(Self {
            sample_rate,
            threshold: -20.0,
            frequency: 6000.0,
        })
    }

    /// Set threshold
    ///
    /// # Errors
    ///
    /// Returns an error if threshold is invalid
    pub fn set_threshold(&mut self, threshold: f32) -> AudioPostResult<()> {
        if threshold > 0.0 {
            return Err(AudioPostError::InvalidThreshold(threshold));
        }
        self.threshold = threshold;
        Ok(())
    }

    /// Set frequency
    ///
    /// # Errors
    ///
    /// Returns an error if frequency is invalid
    pub fn set_frequency(&mut self, frequency: f32) -> AudioPostResult<()> {
        if frequency <= 0.0 || frequency >= self.sample_rate as f32 / 2.0 {
            return Err(AudioPostError::InvalidFrequency(frequency));
        }
        self.frequency = frequency;
        Ok(())
    }
}

/// Transient designer for shaping attack and sustain
#[derive(Debug)]
pub struct TransientDesigner {
    sample_rate: u32,
    attack_gain: f32,
    sustain_gain: f32,
}

impl TransientDesigner {
    /// Create a new transient designer
    ///
    /// # Errors
    ///
    /// Returns an error if sample rate is invalid
    pub fn new(sample_rate: u32) -> AudioPostResult<Self> {
        if sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(sample_rate));
        }

        Ok(Self {
            sample_rate,
            attack_gain: 1.0,
            sustain_gain: 1.0,
        })
    }

    /// Set attack gain (0.0 to 2.0)
    pub fn set_attack_gain(&mut self, gain: f32) {
        self.attack_gain = gain.clamp(0.0, 2.0);
    }

    /// Set sustain gain (0.0 to 2.0)
    pub fn set_sustain_gain(&mut self, gain: f32) {
        self.sustain_gain = gain.clamp(0.0, 2.0);
    }
}

/// Convolution reverb
#[derive(Debug)]
pub struct ConvolutionReverb {
    sample_rate: u32,
    impulse_response: Vec<f32>,
    wet_dry_mix: f32,
}

impl ConvolutionReverb {
    /// Create a new convolution reverb
    ///
    /// # Errors
    ///
    /// Returns an error if sample rate is invalid
    pub fn new(sample_rate: u32) -> AudioPostResult<Self> {
        if sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(sample_rate));
        }

        Ok(Self {
            sample_rate,
            impulse_response: Vec::new(),
            wet_dry_mix: 0.3,
        })
    }

    /// Load impulse response
    pub fn load_impulse_response(&mut self, ir: Vec<f32>) {
        self.impulse_response = ir;
    }

    /// Set wet/dry mix (0.0 to 1.0)
    pub fn set_mix(&mut self, mix: f32) {
        self.wet_dry_mix = mix.clamp(0.0, 1.0);
    }
}

/// Algorithmic reverb type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReverbType {
    /// Plate reverb
    Plate,
    /// Hall reverb
    Hall,
    /// Room reverb
    Room,
    /// Chamber reverb
    Chamber,
}

/// Algorithmic reverb
#[derive(Debug)]
pub struct AlgorithmicReverb {
    sample_rate: u32,
    reverb_type: ReverbType,
    size: f32,
    damping: f32,
    wet_dry_mix: f32,
}

impl AlgorithmicReverb {
    /// Create a new algorithmic reverb
    ///
    /// # Errors
    ///
    /// Returns an error if sample rate is invalid
    pub fn new(sample_rate: u32, reverb_type: ReverbType) -> AudioPostResult<Self> {
        if sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(sample_rate));
        }

        Ok(Self {
            sample_rate,
            reverb_type,
            size: 0.5,
            damping: 0.5,
            wet_dry_mix: 0.3,
        })
    }

    /// Set size (0.0 to 1.0)
    pub fn set_size(&mut self, size: f32) {
        self.size = size.clamp(0.0, 1.0);
    }

    /// Set damping (0.0 to 1.0)
    pub fn set_damping(&mut self, damping: f32) {
        self.damping = damping.clamp(0.0, 1.0);
    }

    /// Set wet/dry mix (0.0 to 1.0)
    pub fn set_mix(&mut self, mix: f32) {
        self.wet_dry_mix = mix.clamp(0.0, 1.0);
    }
}

/// Delay effect
#[derive(Debug)]
pub struct Delay {
    sample_rate: u32,
    delay_buffer: VecDeque<f32>,
    delay_time_ms: f32,
    feedback: f32,
    wet_dry_mix: f32,
}

impl Delay {
    /// Create a new delay effect
    ///
    /// # Errors
    ///
    /// Returns an error if sample rate is invalid
    pub fn new(sample_rate: u32, max_delay_ms: f32) -> AudioPostResult<Self> {
        if sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(sample_rate));
        }

        let buffer_size = (sample_rate as f32 * max_delay_ms / 1000.0) as usize;
        let delay_buffer = VecDeque::with_capacity(buffer_size);

        Ok(Self {
            sample_rate,
            delay_buffer,
            delay_time_ms: 250.0,
            feedback: 0.5,
            wet_dry_mix: 0.3,
        })
    }

    /// Set delay time in milliseconds
    pub fn set_delay_time(&mut self, ms: f32) {
        self.delay_time_ms = ms.max(0.0);
    }

    /// Set feedback (0.0 to 1.0)
    pub fn set_feedback(&mut self, feedback: f32) {
        self.feedback = feedback.clamp(0.0, 0.99);
    }

    /// Set wet/dry mix (0.0 to 1.0)
    pub fn set_mix(&mut self, mix: f32) {
        self.wet_dry_mix = mix.clamp(0.0, 1.0);
    }

    /// Process audio
    pub fn process(&mut self, input: f32) -> f32 {
        let delay_samples = (self.sample_rate as f32 * self.delay_time_ms / 1000.0) as usize;

        // Get delayed sample
        let delayed = if self.delay_buffer.len() >= delay_samples {
            *self
                .delay_buffer
                .get(self.delay_buffer.len() - delay_samples)
                .unwrap_or(&0.0)
        } else {
            0.0
        };

        // Add input with feedback
        let output = delayed;
        self.delay_buffer.push_back(input + delayed * self.feedback);

        // Remove old samples to maintain buffer size
        if self.delay_buffer.len() > delay_samples * 2 {
            self.delay_buffer.pop_front();
        }

        // Mix wet and dry
        input * (1.0 - self.wet_dry_mix) + output * self.wet_dry_mix
    }
}

/// Chorus effect
#[derive(Debug)]
pub struct Chorus {
    sample_rate: u32,
    rate_hz: f32,
    depth: f32,
    wet_dry_mix: f32,
}

impl Chorus {
    /// Create a new chorus effect
    ///
    /// # Errors
    ///
    /// Returns an error if sample rate is invalid
    pub fn new(sample_rate: u32) -> AudioPostResult<Self> {
        if sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(sample_rate));
        }

        Ok(Self {
            sample_rate,
            rate_hz: 0.5,
            depth: 0.5,
            wet_dry_mix: 0.5,
        })
    }

    /// Set modulation rate in Hz
    pub fn set_rate(&mut self, rate_hz: f32) {
        self.rate_hz = rate_hz.max(0.0);
    }

    /// Set depth (0.0 to 1.0)
    pub fn set_depth(&mut self, depth: f32) {
        self.depth = depth.clamp(0.0, 1.0);
    }

    /// Set wet/dry mix (0.0 to 1.0)
    pub fn set_mix(&mut self, mix: f32) {
        self.wet_dry_mix = mix.clamp(0.0, 1.0);
    }
}

/// Flanger effect
#[derive(Debug)]
pub struct Flanger {
    sample_rate: u32,
    rate_hz: f32,
    depth: f32,
    feedback: f32,
    wet_dry_mix: f32,
}

impl Flanger {
    /// Create a new flanger effect
    ///
    /// # Errors
    ///
    /// Returns an error if sample rate is invalid
    pub fn new(sample_rate: u32) -> AudioPostResult<Self> {
        if sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(sample_rate));
        }

        Ok(Self {
            sample_rate,
            rate_hz: 0.5,
            depth: 0.5,
            feedback: 0.5,
            wet_dry_mix: 0.5,
        })
    }

    /// Set modulation rate in Hz
    pub fn set_rate(&mut self, rate_hz: f32) {
        self.rate_hz = rate_hz.max(0.0);
    }

    /// Set depth (0.0 to 1.0)
    pub fn set_depth(&mut self, depth: f32) {
        self.depth = depth.clamp(0.0, 1.0);
    }

    /// Set feedback (0.0 to 1.0)
    pub fn set_feedback(&mut self, feedback: f32) {
        self.feedback = feedback.clamp(0.0, 0.99);
    }

    /// Set wet/dry mix (0.0 to 1.0)
    pub fn set_mix(&mut self, mix: f32) {
        self.wet_dry_mix = mix.clamp(0.0, 1.0);
    }
}

/// Phaser effect
#[derive(Debug)]
pub struct Phaser {
    sample_rate: u32,
    num_stages: usize,
    rate_hz: f32,
    depth: f32,
    feedback: f32,
    wet_dry_mix: f32,
}

impl Phaser {
    /// Create a new phaser effect
    ///
    /// # Errors
    ///
    /// Returns an error if sample rate or stage count is invalid
    pub fn new(sample_rate: u32, num_stages: usize) -> AudioPostResult<Self> {
        if sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(sample_rate));
        }
        if !(2..=12).contains(&num_stages) || num_stages % 2 != 0 {
            return Err(AudioPostError::Generic(
                "Stage count must be even and between 2 and 12".to_string(),
            ));
        }

        Ok(Self {
            sample_rate,
            num_stages,
            rate_hz: 0.5,
            depth: 0.5,
            feedback: 0.5,
            wet_dry_mix: 0.5,
        })
    }

    /// Set modulation rate in Hz
    pub fn set_rate(&mut self, rate_hz: f32) {
        self.rate_hz = rate_hz.max(0.0);
    }

    /// Set depth (0.0 to 1.0)
    pub fn set_depth(&mut self, depth: f32) {
        self.depth = depth.clamp(0.0, 1.0);
    }

    /// Set feedback (0.0 to 1.0)
    pub fn set_feedback(&mut self, feedback: f32) {
        self.feedback = feedback.clamp(0.0, 0.99);
    }

    /// Set wet/dry mix (0.0 to 1.0)
    pub fn set_mix(&mut self, mix: f32) {
        self.wet_dry_mix = mix.clamp(0.0, 1.0);
    }
}

/// Tremolo effect
#[derive(Debug)]
pub struct Tremolo {
    sample_rate: u32,
    rate_hz: f32,
    depth: f32,
    phase: f32,
}

impl Tremolo {
    /// Create a new tremolo effect
    ///
    /// # Errors
    ///
    /// Returns an error if sample rate is invalid
    pub fn new(sample_rate: u32) -> AudioPostResult<Self> {
        if sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(sample_rate));
        }

        Ok(Self {
            sample_rate,
            rate_hz: 5.0,
            depth: 0.5,
            phase: 0.0,
        })
    }

    /// Set modulation rate in Hz
    pub fn set_rate(&mut self, rate_hz: f32) {
        self.rate_hz = rate_hz.max(0.0);
    }

    /// Set depth (0.0 to 1.0)
    pub fn set_depth(&mut self, depth: f32) {
        self.depth = depth.clamp(0.0, 1.0);
    }

    /// Process audio
    pub fn process(&mut self, input: f32) -> f32 {
        let modulation = 1.0 - self.depth * (1.0 - self.phase.sin()) / 2.0;
        self.phase += 2.0 * std::f32::consts::PI * self.rate_hz / self.sample_rate as f32;
        if self.phase > 2.0 * std::f32::consts::PI {
            self.phase -= 2.0 * std::f32::consts::PI;
        }
        input * modulation
    }
}

/// Vocoder
#[derive(Debug)]
pub struct Vocoder {
    sample_rate: u32,
    num_bands: usize,
}

impl Vocoder {
    /// Create a new vocoder
    ///
    /// # Errors
    ///
    /// Returns an error if sample rate or band count is invalid
    pub fn new(sample_rate: u32, num_bands: usize) -> AudioPostResult<Self> {
        if sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(sample_rate));
        }
        if num_bands < 4 || num_bands > 32 {
            return Err(AudioPostError::Generic(
                "Band count must be 4-32".to_string(),
            ));
        }

        Ok(Self {
            sample_rate,
            num_bands,
        })
    }

    /// Get band count
    #[must_use]
    pub fn band_count(&self) -> usize {
        self.num_bands
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multiband_compressor() {
        let comp = MultibandCompressor::new(48000, 4).expect("failed to create");
        assert_eq!(comp.band_count(), 4);
    }

    #[test]
    fn test_invalid_band_count() {
        assert!(MultibandCompressor::new(48000, 1).is_err());
        assert!(MultibandCompressor::new(48000, 7).is_err());
    }

    #[test]
    fn test_de_esser() {
        let mut de_esser = DeEsser::new(48000).expect("failed to create");
        assert!(de_esser.set_threshold(-15.0).is_ok());
        assert!(de_esser.set_frequency(7000.0).is_ok());
    }

    #[test]
    fn test_transient_designer() {
        let mut td = TransientDesigner::new(48000).expect("failed to create");
        td.set_attack_gain(1.5);
        td.set_sustain_gain(0.8);
        assert_eq!(td.attack_gain, 1.5);
        assert_eq!(td.sustain_gain, 0.8);
    }

    #[test]
    fn test_convolution_reverb() {
        let mut reverb = ConvolutionReverb::new(48000).expect("failed to create");
        reverb.load_impulse_response(vec![1.0, 0.5, 0.25]);
        reverb.set_mix(0.4);
        assert_eq!(reverb.wet_dry_mix, 0.4);
    }

    #[test]
    fn test_algorithmic_reverb() {
        let mut reverb = AlgorithmicReverb::new(48000, ReverbType::Hall).expect("failed to create");
        reverb.set_size(0.7);
        reverb.set_damping(0.6);
        assert_eq!(reverb.size, 0.7);
    }

    #[test]
    fn test_delay() {
        let mut delay = Delay::new(48000, 1000.0).expect("failed to create");
        delay.set_delay_time(500.0);
        delay.set_feedback(0.6);
        assert_eq!(delay.delay_time_ms, 500.0);
        assert_eq!(delay.feedback, 0.6);
    }

    #[test]
    fn test_delay_process() {
        let mut delay = Delay::new(48000, 100.0).expect("failed to create");
        delay.set_delay_time(10.0);
        let output = delay.process(1.0);
        assert!(output.is_finite());
    }

    #[test]
    fn test_chorus() {
        let mut chorus = Chorus::new(48000).expect("failed to create");
        chorus.set_rate(0.8);
        chorus.set_depth(0.6);
        assert_eq!(chorus.rate_hz, 0.8);
        assert_eq!(chorus.depth, 0.6);
    }

    #[test]
    fn test_flanger() {
        let mut flanger = Flanger::new(48000).expect("failed to create");
        flanger.set_rate(0.5);
        flanger.set_depth(0.7);
        flanger.set_feedback(0.6);
        assert_eq!(flanger.feedback, 0.6);
    }

    #[test]
    fn test_phaser() {
        let mut phaser = Phaser::new(48000, 4).expect("failed to create");
        phaser.set_rate(0.5);
        phaser.set_depth(0.7);
        assert_eq!(phaser.num_stages, 4);
    }

    #[test]
    fn test_phaser_invalid_stages() {
        assert!(Phaser::new(48000, 3).is_err()); // Odd number
        assert!(Phaser::new(48000, 14).is_err()); // Too many
    }

    #[test]
    fn test_tremolo() {
        let mut tremolo = Tremolo::new(48000).expect("failed to create");
        tremolo.set_rate(6.0);
        tremolo.set_depth(0.8);
        let output = tremolo.process(1.0);
        assert!(output.is_finite());
    }

    #[test]
    fn test_vocoder() {
        let vocoder = Vocoder::new(48000, 16).expect("failed to create");
        assert_eq!(vocoder.band_count(), 16);
    }

    #[test]
    fn test_vocoder_invalid_bands() {
        assert!(Vocoder::new(48000, 2).is_err());
        assert!(Vocoder::new(48000, 64).is_err());
    }
}
