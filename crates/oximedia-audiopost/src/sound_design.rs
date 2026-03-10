#![allow(dead_code)]
//! Sound design tools including synthesizers and spatial audio.

use crate::error::{AudioPostError, AudioPostResult};
use std::f32::consts::PI;

/// Additive synthesizer
#[derive(Debug, Clone)]
pub struct AdditiveSynth {
    sample_rate: u32,
    phase: Vec<f32>,
    amplitudes: Vec<f32>,
}

impl AdditiveSynth {
    /// Create a new additive synthesizer
    ///
    /// # Errors
    ///
    /// Returns an error if sample rate is invalid
    pub fn new(sample_rate: u32, num_harmonics: usize) -> AudioPostResult<Self> {
        if sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(sample_rate));
        }

        Ok(Self {
            sample_rate,
            phase: vec![0.0; num_harmonics],
            amplitudes: vec![1.0 / num_harmonics as f32; num_harmonics],
        })
    }

    /// Set harmonic amplitude
    ///
    /// # Errors
    ///
    /// Returns an error if harmonic index is out of range
    pub fn set_harmonic_amplitude(
        &mut self,
        harmonic: usize,
        amplitude: f32,
    ) -> AudioPostResult<()> {
        if harmonic >= self.amplitudes.len() {
            return Err(AudioPostError::Generic(format!(
                "Harmonic index {harmonic} out of range"
            )));
        }
        self.amplitudes[harmonic] = amplitude;
        Ok(())
    }

    /// Process a buffer with the given fundamental frequency
    pub fn process(&mut self, output: &mut [f32], frequency: f32) {
        for sample in output.iter_mut() {
            let mut value = 0.0;

            for (harmonic, (phase, amplitude)) in self
                .phase
                .iter_mut()
                .zip(self.amplitudes.iter())
                .enumerate()
            {
                let harmonic_freq = frequency * (harmonic + 1) as f32;
                value += amplitude * phase.sin();
                *phase += 2.0 * PI * harmonic_freq / self.sample_rate as f32;
                if *phase > 2.0 * PI {
                    *phase -= 2.0 * PI;
                }
            }

            *sample = value;
        }
    }
}

/// Subtractive synthesizer with filter
#[derive(Debug, Clone)]
pub struct SubtractiveSynth {
    sample_rate: u32,
    phase: f32,
    filter: StateVariableFilter,
}

impl SubtractiveSynth {
    /// Create a new subtractive synthesizer
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
            phase: 0.0,
            filter: StateVariableFilter::new(sample_rate),
        })
    }

    /// Set filter parameters
    ///
    /// # Errors
    ///
    /// Returns an error if parameters are invalid
    pub fn set_filter(&mut self, cutoff: f32, resonance: f32) -> AudioPostResult<()> {
        self.filter.set_parameters(cutoff, resonance)
    }

    /// Process a buffer
    pub fn process(&mut self, output: &mut [f32], frequency: f32) {
        for sample in output.iter_mut() {
            // Generate sawtooth wave
            let value = 2.0 * (self.phase / (2.0 * PI)) - 1.0;
            self.phase += 2.0 * PI * frequency / self.sample_rate as f32;
            if self.phase > 2.0 * PI {
                self.phase -= 2.0 * PI;
            }

            *sample = self.filter.process_lowpass(value);
        }
    }
}

/// State variable filter
#[derive(Debug, Clone)]
pub struct StateVariableFilter {
    sample_rate: u32,
    lowpass: f32,
    bandpass: f32,
    frequency: f32,
    q: f32,
}

impl StateVariableFilter {
    /// Create a new state variable filter
    #[must_use]
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            lowpass: 0.0,
            bandpass: 0.0,
            frequency: 1000.0,
            q: 0.707,
        }
    }

    /// Set filter parameters
    ///
    /// # Errors
    ///
    /// Returns an error if parameters are invalid
    pub fn set_parameters(&mut self, frequency: f32, q: f32) -> AudioPostResult<()> {
        if frequency <= 0.0 || frequency >= self.sample_rate as f32 / 2.0 {
            return Err(AudioPostError::InvalidFrequency(frequency));
        }
        if q <= 0.0 {
            return Err(AudioPostError::InvalidQ(q));
        }

        self.frequency = frequency;
        self.q = q;
        Ok(())
    }

    /// Process a sample through the lowpass filter
    pub fn process_lowpass(&mut self, input: f32) -> f32 {
        let f = 2.0 * (PI * self.frequency / self.sample_rate as f32).sin();
        self.lowpass += f * self.bandpass;
        let highpass = input - self.lowpass - self.q * self.bandpass;
        self.bandpass += f * highpass;
        self.lowpass
    }

    /// Process a sample through the bandpass filter
    pub fn process_bandpass(&mut self, input: f32) -> f32 {
        let f = 2.0 * (PI * self.frequency / self.sample_rate as f32).sin();
        self.lowpass += f * self.bandpass;
        let highpass = input - self.lowpass - self.q * self.bandpass;
        self.bandpass += f * highpass;
        self.bandpass
    }

    /// Process a sample through the highpass filter
    pub fn process_highpass(&mut self, input: f32) -> f32 {
        let f = 2.0 * (PI * self.frequency / self.sample_rate as f32).sin();
        self.lowpass += f * self.bandpass;
        let highpass = input - self.lowpass - self.q * self.bandpass;
        self.bandpass += f * highpass;
        highpass
    }
}

/// FM synthesizer
#[derive(Debug, Clone)]
pub struct FmSynth {
    sample_rate: u32,
    carrier_phase: f32,
    modulator_phase: f32,
    modulation_index: f32,
}

impl FmSynth {
    /// Create a new FM synthesizer
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
            carrier_phase: 0.0,
            modulator_phase: 0.0,
            modulation_index: 1.0,
        })
    }

    /// Set modulation index
    pub fn set_modulation_index(&mut self, index: f32) {
        self.modulation_index = index;
    }

    /// Process a buffer
    pub fn process(&mut self, output: &mut [f32], carrier_freq: f32, modulator_freq: f32) {
        for sample in output.iter_mut() {
            let modulator = self.modulator_phase.sin();
            let modulated_freq = carrier_freq + modulator * self.modulation_index * modulator_freq;

            *sample = self.carrier_phase.sin();

            self.carrier_phase += 2.0 * PI * modulated_freq / self.sample_rate as f32;
            self.modulator_phase += 2.0 * PI * modulator_freq / self.sample_rate as f32;

            if self.carrier_phase > 2.0 * PI {
                self.carrier_phase -= 2.0 * PI;
            }
            if self.modulator_phase > 2.0 * PI {
                self.modulator_phase -= 2.0 * PI;
            }
        }
    }
}

/// Granular synthesizer
#[derive(Debug, Clone)]
pub struct GranularSynth {
    sample_rate: u32,
    grain_size: usize,
    grain_overlap: f32,
}

impl GranularSynth {
    /// Create a new granular synthesizer
    ///
    /// # Errors
    ///
    /// Returns an error if sample rate is invalid
    pub fn new(sample_rate: u32, grain_size: usize) -> AudioPostResult<Self> {
        if sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(sample_rate));
        }

        Ok(Self {
            sample_rate,
            grain_size,
            grain_overlap: 0.5,
        })
    }

    /// Set grain overlap (0.0 to 1.0)
    pub fn set_grain_overlap(&mut self, overlap: f32) {
        self.grain_overlap = overlap.clamp(0.0, 1.0);
    }

    /// Process granular synthesis
    pub fn process(&self, _source: &[f32], output: &mut [f32], _pitch_shift: f32) {
        // Placeholder implementation
        for sample in output.iter_mut() {
            *sample = 0.0;
        }
    }
}

/// Wavetable synthesizer
#[derive(Debug, Clone)]
pub struct WavetableSynth {
    sample_rate: u32,
    wavetable: Vec<f32>,
    phase: f32,
}

impl WavetableSynth {
    /// Create a new wavetable synthesizer
    ///
    /// # Errors
    ///
    /// Returns an error if sample rate is invalid or wavetable is empty
    pub fn new(sample_rate: u32, wavetable: Vec<f32>) -> AudioPostResult<Self> {
        if sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(sample_rate));
        }
        if wavetable.is_empty() {
            return Err(AudioPostError::Generic(
                "Wavetable cannot be empty".to_string(),
            ));
        }

        Ok(Self {
            sample_rate,
            wavetable,
            phase: 0.0,
        })
    }

    /// Create a sine wavetable
    ///
    /// # Errors
    ///
    /// Returns an error if sample rate is invalid
    pub fn sine(sample_rate: u32, table_size: usize) -> AudioPostResult<Self> {
        let mut wavetable = vec![0.0; table_size];
        for (i, sample) in wavetable.iter_mut().enumerate() {
            *sample = (2.0 * PI * i as f32 / table_size as f32).sin();
        }
        Self::new(sample_rate, wavetable)
    }

    /// Create a square wavetable
    ///
    /// # Errors
    ///
    /// Returns an error if sample rate is invalid
    pub fn square(sample_rate: u32, table_size: usize) -> AudioPostResult<Self> {
        let mut wavetable = vec![0.0; table_size];
        for (i, sample) in wavetable.iter_mut().enumerate() {
            *sample = if i < table_size / 2 { 1.0 } else { -1.0 };
        }
        Self::new(sample_rate, wavetable)
    }

    /// Create a sawtooth wavetable
    ///
    /// # Errors
    ///
    /// Returns an error if sample rate is invalid
    pub fn sawtooth(sample_rate: u32, table_size: usize) -> AudioPostResult<Self> {
        let mut wavetable = vec![0.0; table_size];
        for (i, sample) in wavetable.iter_mut().enumerate() {
            *sample = 2.0 * (i as f32 / table_size as f32) - 1.0;
        }
        Self::new(sample_rate, wavetable)
    }

    /// Create a triangle wavetable
    ///
    /// # Errors
    ///
    /// Returns an error if sample rate is invalid
    pub fn triangle(sample_rate: u32, table_size: usize) -> AudioPostResult<Self> {
        let mut wavetable = vec![0.0; table_size];
        for (i, sample) in wavetable.iter_mut().enumerate() {
            let t = i as f32 / table_size as f32;
            *sample = if t < 0.5 {
                4.0 * t - 1.0
            } else {
                3.0 - 4.0 * t
            };
        }
        Self::new(sample_rate, wavetable)
    }

    /// Process a buffer with linear interpolation
    pub fn process(&mut self, output: &mut [f32], frequency: f32) {
        let table_len = self.wavetable.len();
        let phase_inc = frequency * table_len as f32 / self.sample_rate as f32;

        for sample in output.iter_mut() {
            // Linear interpolation
            let index = self.phase as usize % table_len;
            let next_index = (index + 1) % table_len;
            let frac = self.phase - self.phase.floor();

            *sample = self.wavetable[index] * (1.0 - frac) + self.wavetable[next_index] * frac;

            self.phase += phase_inc;
            if self.phase >= table_len as f32 {
                self.phase -= table_len as f32;
            }
        }
    }
}

/// Spatial audio panning
#[derive(Debug, Clone, Copy)]
pub enum PanningMode {
    /// Stereo panning
    Stereo,
    /// 5.1 surround
    Surround51,
    /// 7.1 surround
    Surround71,
    /// 7.1.4 Atmos
    Atmos714,
}

/// 3D position for spatial audio
#[derive(Debug, Clone, Copy)]
pub struct Position3D {
    /// X coordinate
    pub x: f32,
    /// Y coordinate
    pub y: f32,
    /// Z coordinate
    pub z: f32,
}

impl Position3D {
    /// Create a new 3D position
    #[must_use]
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Calculate distance from origin
    #[must_use]
    pub fn distance(&self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Calculate distance to another position
    #[must_use]
    pub fn distance_to(&self, other: &Self) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Normalize to unit vector
    #[must_use]
    pub fn normalize(&self) -> Self {
        let len = self.distance();
        if len > 0.0 {
            Self {
                x: self.x / len,
                y: self.y / len,
                z: self.z / len,
            }
        } else {
            *self
        }
    }
}

/// Spatial audio processor
#[derive(Debug)]
pub struct SpatialAudio {
    mode: PanningMode,
    sample_rate: u32,
}

impl SpatialAudio {
    /// Create a new spatial audio processor
    ///
    /// # Errors
    ///
    /// Returns an error if sample rate is invalid
    pub fn new(mode: PanningMode, sample_rate: u32) -> AudioPostResult<Self> {
        if sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(sample_rate));
        }

        Ok(Self { mode, sample_rate })
    }

    /// Calculate stereo pan gains
    #[must_use]
    pub fn calculate_stereo_gains(&self, pan: f32) -> (f32, f32) {
        let pan = pan.clamp(-1.0, 1.0);
        let left = ((1.0 - pan) / 2.0).sqrt();
        let right = ((1.0 + pan) / 2.0).sqrt();
        (left, right)
    }

    /// Calculate 5.1 surround gains for a 3D position
    #[must_use]
    pub fn calculate_51_gains(&self, pos: &Position3D) -> [f32; 6] {
        let normalized = pos.normalize();
        let angle = normalized.x.atan2(normalized.z);
        let distance = pos.distance();
        let attenuation = self.calculate_distance_attenuation(distance);

        // Simple panning based on angle
        let mut gains = [0.0_f32; 6]; // L, R, C, LFE, LS, RS

        // Front speakers
        if angle.abs() < PI / 4.0 {
            gains[2] = attenuation; // Center
        } else if angle > 0.0 {
            gains[1] = attenuation; // Right
        } else {
            gains[0] = attenuation; // Left
        }

        gains
    }

    /// Calculate distance attenuation
    #[must_use]
    pub fn calculate_distance_attenuation(&self, distance: f32) -> f32 {
        (1.0 / (1.0 + distance)).max(0.0)
    }

    /// Calculate Doppler shift
    #[must_use]
    pub fn calculate_doppler_shift(&self, velocity: f32, sound_speed: f32) -> f32 {
        sound_speed / (sound_speed + velocity)
    }

    /// Get channel count for panning mode
    #[must_use]
    pub fn channel_count(&self) -> usize {
        match self.mode {
            PanningMode::Stereo => 2,
            PanningMode::Surround51 => 6,
            PanningMode::Surround71 => 8,
            PanningMode::Atmos714 => 12,
        }
    }
}

/// Pitch shifter using time-domain method
#[derive(Debug)]
pub struct PitchShifter {
    sample_rate: u32,
}

impl PitchShifter {
    /// Create a new pitch shifter
    ///
    /// # Errors
    ///
    /// Returns an error if sample rate is invalid
    pub fn new(sample_rate: u32) -> AudioPostResult<Self> {
        if sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(sample_rate));
        }

        Ok(Self { sample_rate })
    }

    /// Shift pitch of audio buffer
    pub fn process(&self, _input: &[f32], output: &mut [f32], _pitch_ratio: f32) {
        // Placeholder implementation
        for sample in output.iter_mut() {
            *sample = 0.0;
        }
    }
}

/// Time stretcher using WSOLA algorithm
#[derive(Debug)]
pub struct TimeStretcher {
    sample_rate: u32,
    window_size: usize,
}

impl TimeStretcher {
    /// Create a new time stretcher
    ///
    /// # Errors
    ///
    /// Returns an error if sample rate is invalid
    pub fn new(sample_rate: u32, window_size: usize) -> AudioPostResult<Self> {
        if sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(sample_rate));
        }

        Ok(Self {
            sample_rate,
            window_size,
        })
    }

    /// Stretch time of audio buffer
    pub fn process(&self, _input: &[f32], output: &mut [f32], _time_ratio: f32) {
        // Placeholder implementation
        for sample in output.iter_mut() {
            *sample = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_additive_synth_creation() {
        let synth = AdditiveSynth::new(48000, 8).expect("failed to create");
        assert_eq!(synth.sample_rate, 48000);
        assert_eq!(synth.phase.len(), 8);
    }

    #[test]
    fn test_additive_synth_process() {
        let mut synth = AdditiveSynth::new(48000, 8).expect("failed to create");
        let mut buffer = vec![0.0_f32; 1024];
        synth.process(&mut buffer, 440.0);
        assert!(buffer.iter().any(|&x| x != 0.0));
    }

    #[test]
    fn test_subtractive_synth() {
        let mut synth = SubtractiveSynth::new(48000).expect("failed to create");
        synth
            .set_filter(1000.0, 0.707)
            .expect("set_filter should succeed");
        let mut buffer = vec![0.0_f32; 1024];
        synth.process(&mut buffer, 440.0);
        assert!(buffer.iter().any(|&x| x != 0.0));
    }

    #[test]
    fn test_fm_synth() {
        let mut synth = FmSynth::new(48000).expect("failed to create");
        synth.set_modulation_index(2.0);
        let mut buffer = vec![0.0_f32; 1024];
        synth.process(&mut buffer, 440.0, 220.0);
        assert!(buffer.iter().any(|&x| x != 0.0));
    }

    #[test]
    fn test_state_variable_filter() {
        let mut filter = StateVariableFilter::new(48000);
        assert!(filter.set_parameters(1000.0, 0.707).is_ok());
        let output = filter.process_lowpass(1.0);
        assert!(output.is_finite());
    }

    #[test]
    fn test_invalid_filter_frequency() {
        let mut filter = StateVariableFilter::new(48000);
        assert!(filter.set_parameters(0.0, 0.707).is_err());
        assert!(filter.set_parameters(50000.0, 0.707).is_err());
    }

    #[test]
    fn test_position_3d() {
        let pos = Position3D::new(3.0, 4.0, 0.0);
        assert!((pos.distance() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_position_distance_to() {
        let pos1 = Position3D::new(0.0, 0.0, 0.0);
        let pos2 = Position3D::new(3.0, 4.0, 0.0);
        assert!((pos1.distance_to(&pos2) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_position_normalize() {
        let pos = Position3D::new(3.0, 4.0, 0.0);
        let normalized = pos.normalize();
        assert!((normalized.distance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_spatial_audio_stereo_gains() {
        let spatial = SpatialAudio::new(PanningMode::Stereo, 48000).expect("failed to create");
        let (left, right) = spatial.calculate_stereo_gains(0.0);
        assert!((left - 0.707).abs() < 0.01);
        assert!((right - 0.707).abs() < 0.01);
    }

    #[test]
    fn test_spatial_audio_51_gains() {
        let spatial = SpatialAudio::new(PanningMode::Surround51, 48000).expect("failed to create");
        let pos = Position3D::new(0.0, 0.0, 1.0);
        let gains = spatial.calculate_51_gains(&pos);
        assert_eq!(gains.len(), 6);
    }

    #[test]
    fn test_spatial_audio_channel_count() {
        let spatial = SpatialAudio::new(PanningMode::Surround51, 48000).expect("failed to create");
        assert_eq!(spatial.channel_count(), 6);
    }

    #[test]
    fn test_distance_attenuation() {
        let spatial = SpatialAudio::new(PanningMode::Stereo, 48000).expect("failed to create");
        let attenuation = spatial.calculate_distance_attenuation(0.0);
        assert_eq!(attenuation, 1.0);
    }

    #[test]
    fn test_doppler_shift() {
        let spatial = SpatialAudio::new(PanningMode::Stereo, 48000).expect("failed to create");
        let shift = spatial.calculate_doppler_shift(0.0, 343.0);
        assert_eq!(shift, 1.0);
    }

    #[test]
    fn test_granular_synth() {
        let synth = GranularSynth::new(48000, 1024).expect("failed to create");
        assert_eq!(synth.grain_size, 1024);
    }

    #[test]
    fn test_pitch_shifter() {
        let shifter = PitchShifter::new(48000).expect("failed to create");
        let input = vec![0.0_f32; 1024];
        let mut output = vec![0.0_f32; 1024];
        shifter.process(&input, &mut output, 1.5);
        assert_eq!(output.len(), 1024);
    }

    #[test]
    fn test_wavetable_sine() {
        let synth = WavetableSynth::sine(48000, 1024).expect("operation should succeed");
        assert_eq!(synth.wavetable.len(), 1024);
    }

    #[test]
    fn test_wavetable_square() {
        let synth = WavetableSynth::square(48000, 1024).expect("operation should succeed");
        assert_eq!(synth.wavetable.len(), 1024);
    }

    #[test]
    fn test_wavetable_sawtooth() {
        let synth = WavetableSynth::sawtooth(48000, 1024).expect("operation should succeed");
        assert_eq!(synth.wavetable.len(), 1024);
    }

    #[test]
    fn test_wavetable_triangle() {
        let synth = WavetableSynth::triangle(48000, 1024).expect("operation should succeed");
        assert_eq!(synth.wavetable.len(), 1024);
    }

    #[test]
    fn test_wavetable_process() {
        let mut synth = WavetableSynth::sine(48000, 1024).expect("operation should succeed");
        let mut buffer = vec![0.0_f32; 100];
        synth.process(&mut buffer, 440.0);
        assert!(buffer.iter().any(|&x| x != 0.0));
    }

    #[test]
    fn test_empty_wavetable() {
        assert!(WavetableSynth::new(48000, vec![]).is_err());
    }

    #[test]
    fn test_time_stretcher() {
        let stretcher = TimeStretcher::new(48000, 1024).expect("failed to create");
        assert_eq!(stretcher.window_size, 1024);
    }
}
