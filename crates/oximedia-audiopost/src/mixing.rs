#![allow(dead_code)]
//! Professional mixing console with channel strips and master section.

use crate::error::{AudioPostError, AudioPostResult};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

/// Mixing console
#[derive(Debug)]
pub struct MixingConsole {
    sample_rate: u32,
    buffer_size: usize,
    channels: HashMap<usize, Arc<RwLock<ChannelStrip>>>,
    next_channel_id: usize,
    master: MasterSection,
    aux_busses: Vec<AuxBus>,
}

impl MixingConsole {
    /// Create a new mixing console
    ///
    /// # Errors
    ///
    /// Returns an error if sample rate or buffer size is invalid
    pub fn new(sample_rate: u32, buffer_size: usize) -> AudioPostResult<Self> {
        if sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(sample_rate));
        }
        if buffer_size == 0 {
            return Err(AudioPostError::InvalidBufferSize(buffer_size));
        }

        Ok(Self {
            sample_rate,
            buffer_size,
            channels: HashMap::new(),
            next_channel_id: 1,
            master: MasterSection::new(sample_rate)?,
            aux_busses: Vec::new(),
        })
    }

    /// Add a channel strip
    pub fn add_channel(&mut self, name: &str) -> AudioPostResult<usize> {
        let id = self.next_channel_id;
        let channel = ChannelStrip::new(name, self.sample_rate)?;
        self.channels.insert(id, Arc::new(RwLock::new(channel)));
        self.next_channel_id += 1;
        Ok(id)
    }

    /// Get a channel strip
    ///
    /// # Errors
    ///
    /// Returns an error if channel is not found
    pub fn get_channel(&self, id: usize) -> AudioPostResult<Arc<RwLock<ChannelStrip>>> {
        self.channels
            .get(&id)
            .cloned()
            .ok_or(AudioPostError::ChannelNotFound(id))
    }

    /// Set channel gain
    ///
    /// # Errors
    ///
    /// Returns an error if channel is not found or gain is invalid
    pub fn set_channel_gain(&mut self, id: usize, gain_db: f32) -> AudioPostResult<()> {
        let channel = self.get_channel(id)?;
        {
            let mut guard = channel.write();
            guard.set_gain(gain_db)
        }
    }

    /// Set channel pan
    ///
    /// # Errors
    ///
    /// Returns an error if channel is not found or pan is invalid
    pub fn set_channel_pan(&mut self, id: usize, pan: f32) -> AudioPostResult<()> {
        let channel = self.get_channel(id)?;
        {
            let mut guard = channel.write();
            guard.set_pan(pan)
        }
    }

    /// Add an aux bus
    pub fn add_aux_bus(&mut self, name: &str) -> usize {
        let id = self.aux_busses.len();
        self.aux_busses.push(AuxBus::new(name));
        id
    }

    /// Get master section
    #[must_use]
    pub fn get_master(&self) -> &MasterSection {
        &self.master
    }

    /// Get mutable master section
    pub fn get_master_mut(&mut self) -> &mut MasterSection {
        &mut self.master
    }

    /// Process a mix.
    ///
    /// `inputs` is a slice of per-channel mono buffers, one entry per channel in insertion
    /// order.  The output buffer is assumed to be stereo-interleaved (L, R, L, R, …).
    /// When the output length is not a multiple of 2, it is treated as mono.
    pub fn process(&mut self, inputs: &[Vec<f32>], output: &mut [f32]) {
        // Zero output buffer.
        output.fill(0.0);

        let stereo = output.len() % 2 == 0 && output.len() >= 2;
        let frame_count = if stereo {
            output.len() / 2
        } else {
            output.len()
        };

        // Collect channel parameters while holding no locks across iterations.
        let channel_params: Vec<(f32, f32, f32, bool)> = self
            .channels
            .values()
            .map(|arc| {
                let ch = arc.read();
                let gain_linear = 10.0_f32.powf(ch.input_gain / 20.0);
                let fader = ch.fader;
                let pan = ch.pan.clamp(-1.0, 1.0);
                let active = !ch.muted && fader > 0.0;
                (gain_linear, fader, pan, active)
            })
            .collect();

        // Accumulate each channel into the output buffer.
        for (ch_idx, (gain_linear, fader, pan, active)) in channel_params.iter().enumerate() {
            if !active {
                continue;
            }
            let input = match inputs.get(ch_idx) {
                Some(buf) => buf,
                None => continue,
            };
            if input.is_empty() {
                continue;
            }

            // Constant-power pan law:
            //   pan ∈ [-1, 1] → angle ∈ [0, π/2]
            // left  = cos(angle),  right = sin(angle)
            let angle = ((*pan + 1.0) / 2.0) * std::f32::consts::FRAC_PI_2;
            let pan_left = angle.cos();
            let pan_right = angle.sin();
            let channel_gain = gain_linear * fader;

            if stereo {
                for frame in 0..frame_count {
                    let sample = input.get(frame).copied().unwrap_or(0.0) * channel_gain;
                    output[frame * 2] += sample * pan_left;
                    output[frame * 2 + 1] += sample * pan_right;
                }
            } else {
                for frame in 0..frame_count {
                    let sample = input.get(frame).copied().unwrap_or(0.0) * channel_gain;
                    output[frame] += sample;
                }
            }
        }

        // Normalize to prevent inter-channel clipping before master processing.
        // Find peak absolute value; only normalize if it exceeds unity.
        let peak = output.iter().map(|s| s.abs()).fold(0.0_f32, f32::max);
        if peak > 1.0 {
            let inv_peak = 1.0 / peak;
            for s in output.iter_mut() {
                *s *= inv_peak;
            }
        }

        // Apply master processing (fader + limiter).
        self.master.process(output);
    }
}

/// Channel strip
#[derive(Debug, Clone)]
pub struct ChannelStrip {
    /// Channel name
    pub name: String,
    /// Sample rate
    sample_rate: u32,
    /// Input gain (dB)
    pub input_gain: f32,
    /// High-pass filter frequency
    pub hpf_freq: Option<f32>,
    /// Gate threshold (dB)
    pub gate_threshold: Option<f32>,
    /// Compressor settings
    pub compressor: Option<CompressorSettings>,
    /// EQ settings
    pub eq: ParametricEq,
    /// Pan (-1.0 to 1.0)
    pub pan: f32,
    /// Fader (0.0 to 1.0, representing -∞ to +12dB)
    pub fader: f32,
    /// Solo flag
    pub solo: bool,
    /// Mute flag
    pub muted: bool,
    /// Record enable flag
    pub record_enabled: bool,
}

impl ChannelStrip {
    /// Create a new channel strip
    ///
    /// # Errors
    ///
    /// Returns an error if sample rate is invalid
    pub fn new(name: &str, sample_rate: u32) -> AudioPostResult<Self> {
        if sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(sample_rate));
        }

        Ok(Self {
            name: name.to_string(),
            sample_rate,
            input_gain: 0.0,
            hpf_freq: None,
            gate_threshold: None,
            compressor: None,
            eq: ParametricEq::new(),
            pan: 0.0,
            fader: 0.75, // Unity gain
            solo: false,
            muted: false,
            record_enabled: false,
        })
    }

    /// Set input gain in dB
    ///
    /// # Errors
    ///
    /// Returns an error if gain is out of range
    pub fn set_gain(&mut self, gain_db: f32) -> AudioPostResult<()> {
        if !(-60.0..=60.0).contains(&gain_db) {
            return Err(AudioPostError::InvalidGain(gain_db));
        }
        self.input_gain = gain_db;
        Ok(())
    }

    /// Set pan (-1.0 to 1.0)
    ///
    /// # Errors
    ///
    /// Returns an error if pan is out of range
    pub fn set_pan(&mut self, pan: f32) -> AudioPostResult<()> {
        if !(-1.0..=1.0).contains(&pan) {
            return Err(AudioPostError::InvalidPan(pan));
        }
        self.pan = pan;
        Ok(())
    }

    /// Set fader level (0.0 to 1.0)
    pub fn set_fader(&mut self, level: f32) {
        self.fader = level.clamp(0.0, 1.0);
    }

    /// Enable high-pass filter
    ///
    /// # Errors
    ///
    /// Returns an error if frequency is invalid
    pub fn enable_hpf(&mut self, frequency: f32) -> AudioPostResult<()> {
        if frequency <= 0.0 || frequency >= self.sample_rate as f32 / 2.0 {
            return Err(AudioPostError::InvalidFrequency(frequency));
        }
        self.hpf_freq = Some(frequency);
        Ok(())
    }

    /// Disable high-pass filter
    pub fn disable_hpf(&mut self) {
        self.hpf_freq = None;
    }
}

/// 4-band parametric EQ
#[derive(Debug, Clone)]
pub struct ParametricEq {
    /// EQ bands
    pub bands: Vec<EqBand>,
}

impl ParametricEq {
    /// Create a new parametric EQ with 4 bands
    #[must_use]
    pub fn new() -> Self {
        Self {
            bands: vec![
                EqBand::new(100.0, 0.0, 1.0, EqFilterType::LowShelf),
                EqBand::new(500.0, 0.0, 1.0, EqFilterType::Bell),
                EqBand::new(2000.0, 0.0, 1.0, EqFilterType::Bell),
                EqBand::new(8000.0, 0.0, 1.0, EqFilterType::HighShelf),
            ],
        }
    }

    /// Get a band
    #[must_use]
    pub fn get_band(&self, index: usize) -> Option<&EqBand> {
        self.bands.get(index)
    }

    /// Get a mutable band
    pub fn get_band_mut(&mut self, index: usize) -> Option<&mut EqBand> {
        self.bands.get_mut(index)
    }
}

impl Default for ParametricEq {
    fn default() -> Self {
        Self::new()
    }
}

/// EQ band
#[derive(Debug, Clone)]
pub struct EqBand {
    /// Frequency in Hz
    pub frequency: f32,
    /// Gain in dB
    pub gain: f32,
    /// Q factor
    pub q: f32,
    /// Filter type
    pub filter_type: EqFilterType,
    /// Enabled flag
    pub enabled: bool,
}

impl EqBand {
    /// Create a new EQ band
    #[must_use]
    pub fn new(frequency: f32, gain: f32, q: f32, filter_type: EqFilterType) -> Self {
        Self {
            frequency,
            gain,
            q,
            filter_type,
            enabled: true,
        }
    }
}

/// EQ filter type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EqFilterType {
    /// Low shelf
    LowShelf,
    /// High shelf
    HighShelf,
    /// Bell/Peak
    Bell,
    /// Low pass
    LowPass,
    /// High pass
    HighPass,
}

/// Compressor settings
#[derive(Debug, Clone)]
pub struct CompressorSettings {
    /// Threshold in dB
    pub threshold: f32,
    /// Ratio (1:1 to 20:1)
    pub ratio: f32,
    /// Attack time in ms
    pub attack_ms: f32,
    /// Release time in ms
    pub release_ms: f32,
    /// Knee width in dB
    pub knee_db: f32,
    /// Makeup gain in dB
    pub makeup_gain: f32,
}

impl CompressorSettings {
    /// Create new compressor settings
    ///
    /// # Errors
    ///
    /// Returns an error if any parameter is invalid
    pub fn new(
        threshold: f32,
        ratio: f32,
        attack_ms: f32,
        release_ms: f32,
    ) -> AudioPostResult<Self> {
        if threshold > 0.0 {
            return Err(AudioPostError::InvalidThreshold(threshold));
        }
        if ratio < 1.0 {
            return Err(AudioPostError::InvalidRatio(ratio));
        }
        if attack_ms <= 0.0 {
            return Err(AudioPostError::InvalidAttack(attack_ms));
        }
        if release_ms <= 0.0 {
            return Err(AudioPostError::InvalidRelease(release_ms));
        }

        Ok(Self {
            threshold,
            ratio,
            attack_ms,
            release_ms,
            knee_db: 0.0,
            makeup_gain: 0.0,
        })
    }
}

/// Aux bus
#[derive(Debug, Clone)]
pub struct AuxBus {
    /// Bus name
    pub name: String,
    /// Pre-fader flag
    pub pre_fader: bool,
    /// Level
    pub level: f32,
    /// Pan
    pub pan: f32,
}

impl AuxBus {
    /// Create a new aux bus
    #[must_use]
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            pre_fader: false,
            level: 0.0,
            pan: 0.0,
        }
    }
}

/// Master section
#[derive(Debug)]
pub struct MasterSection {
    sample_rate: u32,
    /// Master fader (0.0 to 1.0)
    pub fader: f32,
    /// Master bus compressor
    pub compressor: Option<CompressorSettings>,
    /// Master limiter threshold
    pub limiter_threshold: f32,
}

impl MasterSection {
    /// Create a new master section
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
            fader: 0.75,
            compressor: None,
            limiter_threshold: 0.9,
        })
    }

    /// Set master fader level
    pub fn set_fader(&mut self, level: f32) {
        self.fader = level.clamp(0.0, 1.0);
    }

    /// Process master output
    pub fn process(&self, buffer: &mut [f32]) {
        // Apply fader
        let gain = self.fader;
        for sample in buffer.iter_mut() {
            *sample *= gain;

            // Apply limiting
            let thresh = self.limiter_threshold.abs();
            if *sample > thresh {
                *sample = thresh;
            } else if *sample < -thresh {
                *sample = -thresh;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mixing_console_creation() {
        let console = MixingConsole::new(48000, 512).expect("failed to create");
        assert_eq!(console.sample_rate, 48000);
    }

    #[test]
    fn test_add_channel() {
        let mut console = MixingConsole::new(48000, 512).expect("failed to create");
        let id = console
            .add_channel("Dialogue")
            .expect("add_channel should succeed");
        assert_eq!(id, 1);
    }

    #[test]
    fn test_set_channel_gain() {
        let mut console = MixingConsole::new(48000, 512).expect("failed to create");
        let id = console
            .add_channel("Dialogue")
            .expect("add_channel should succeed");
        assert!(console.set_channel_gain(id, 6.0).is_ok());
    }

    #[test]
    fn test_set_channel_pan() {
        let mut console = MixingConsole::new(48000, 512).expect("failed to create");
        let id = console
            .add_channel("Dialogue")
            .expect("add_channel should succeed");
        assert!(console.set_channel_pan(id, 0.5).is_ok());
    }

    #[test]
    fn test_channel_strip_creation() {
        let channel = ChannelStrip::new("Test", 48000).expect("failed to create");
        assert_eq!(channel.name, "Test");
        assert_eq!(channel.pan, 0.0);
    }

    #[test]
    fn test_channel_gain_validation() {
        let mut channel = ChannelStrip::new("Test", 48000).expect("failed to create");
        assert!(channel.set_gain(6.0).is_ok());
        assert!(channel.set_gain(100.0).is_err());
    }

    #[test]
    fn test_channel_pan_validation() {
        let mut channel = ChannelStrip::new("Test", 48000).expect("failed to create");
        assert!(channel.set_pan(0.0).is_ok());
        assert!(channel.set_pan(1.0).is_ok());
        assert!(channel.set_pan(-1.0).is_ok());
        assert!(channel.set_pan(2.0).is_err());
    }

    #[test]
    fn test_parametric_eq() {
        let eq = ParametricEq::new();
        assert_eq!(eq.bands.len(), 4);
    }

    #[test]
    fn test_eq_band() {
        let band = EqBand::new(1000.0, 3.0, 0.707, EqFilterType::Bell);
        assert_eq!(band.frequency, 1000.0);
        assert_eq!(band.gain, 3.0);
    }

    #[test]
    fn test_compressor_settings() {
        let comp = CompressorSettings::new(-10.0, 4.0, 5.0, 50.0).expect("failed to create");
        assert_eq!(comp.threshold, -10.0);
        assert_eq!(comp.ratio, 4.0);
    }

    #[test]
    fn test_invalid_compressor_settings() {
        assert!(CompressorSettings::new(10.0, 4.0, 5.0, 50.0).is_err()); // Positive threshold
        assert!(CompressorSettings::new(-10.0, 0.5, 5.0, 50.0).is_err()); // Ratio < 1
        assert!(CompressorSettings::new(-10.0, 4.0, 0.0, 50.0).is_err()); // Zero attack
    }

    #[test]
    fn test_aux_bus() {
        let bus = AuxBus::new("Reverb");
        assert_eq!(bus.name, "Reverb");
        assert!(!bus.pre_fader);
    }

    #[test]
    fn test_master_section() {
        let master = MasterSection::new(48000).expect("failed to create");
        assert_eq!(master.fader, 0.75);
    }

    #[test]
    fn test_master_process() {
        let master = MasterSection::new(48000).expect("failed to create");
        let mut buffer = vec![0.5_f32; 1024];
        master.process(&mut buffer);
        assert!((buffer[0] - 0.375).abs() < 1e-6);
    }

    #[test]
    fn test_hpf_enable() {
        let mut channel = ChannelStrip::new("Test", 48000).expect("failed to create");
        assert!(channel.enable_hpf(80.0).is_ok());
        assert_eq!(channel.hpf_freq, Some(80.0));
    }

    #[test]
    fn test_hpf_disable() {
        let mut channel = ChannelStrip::new("Test", 48000).expect("failed to create");
        channel.enable_hpf(80.0).expect("enable_hpf should succeed");
        channel.disable_hpf();
        assert_eq!(channel.hpf_freq, None);
    }
}
