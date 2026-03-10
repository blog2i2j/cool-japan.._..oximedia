//! Multi-track mixing for audio description.
//!
//! This module provides professional audio mixing capabilities including
//! level balancing, crossfades, dynamic range control, and peak limiting
//! for combining main audio with audio description tracks.

#![forbid(unsafe_code)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_precision_loss)]

use super::ducking::{AutomaticDucker, DuckingConfig};
use crate::dsp::{Compressor, CompressorConfig};
use crate::{AudioError, AudioResult};

/// Mixing mode for audio description.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum MixingMode {
    /// Replace main audio completely with AD.
    Replace,
    /// Mix AD with main audio at specified levels.
    #[default]
    Mix,
    /// Duck (lower) main audio when AD is active.
    Duck,
    /// Pause main audio during AD (extended AD).
    Pause,
}

/// Configuration for audio mixing.
#[derive(Clone, Debug)]
pub struct MixingConfig {
    /// Mixing mode.
    pub mode: MixingMode,
    /// Main audio level (0.0-1.0).
    pub main_level: f64,
    /// Audio description level (0.0-1.0).
    pub ad_level: f64,
    /// Crossfade duration in milliseconds.
    pub crossfade_ms: f64,
    /// Enable dynamic range compression.
    pub enable_compression: bool,
    /// Compressor configuration.
    pub compressor_config: CompressorConfig,
    /// Enable peak limiting.
    pub enable_limiting: bool,
    /// Peak limiter ceiling in dB.
    pub limiter_ceiling_db: f64,
    /// Ducking configuration (used when mode is Duck).
    pub ducking_config: DuckingConfig,
}

impl Default for MixingConfig {
    fn default() -> Self {
        Self {
            mode: MixingMode::Duck,
            main_level: 1.0,
            ad_level: 1.0,
            crossfade_ms: 20.0,
            enable_compression: false,
            compressor_config: CompressorConfig::default(),
            enable_limiting: true,
            limiter_ceiling_db: -0.5,
            ducking_config: DuckingConfig::default(),
        }
    }
}

impl MixingConfig {
    /// Create a new mixing configuration.
    #[must_use]
    pub fn new(mode: MixingMode) -> Self {
        Self {
            mode,
            ..Default::default()
        }
    }

    /// Set audio levels.
    #[must_use]
    pub fn with_levels(mut self, main_level: f64, ad_level: f64) -> Self {
        self.main_level = main_level.clamp(0.0, 2.0);
        self.ad_level = ad_level.clamp(0.0, 2.0);
        self
    }

    /// Set crossfade duration.
    #[must_use]
    pub fn with_crossfade(mut self, duration_ms: f64) -> Self {
        self.crossfade_ms = duration_ms.max(0.0);
        self
    }

    /// Enable compression with configuration.
    #[must_use]
    pub fn with_compression(mut self, config: CompressorConfig) -> Self {
        self.enable_compression = true;
        self.compressor_config = config;
        self
    }

    /// Enable peak limiting.
    #[must_use]
    pub fn with_limiting(mut self, ceiling_db: f64) -> Self {
        self.enable_limiting = true;
        self.limiter_ceiling_db = ceiling_db;
        self
    }

    /// Set ducking configuration.
    #[must_use]
    pub fn with_ducking(mut self, config: DuckingConfig) -> Self {
        self.ducking_config = config;
        self
    }

    /// Create broadcast-quality mixing preset.
    #[must_use]
    pub fn broadcast() -> Self {
        Self {
            mode: MixingMode::Duck,
            main_level: 1.0,
            ad_level: 1.0,
            crossfade_ms: 10.0,
            enable_compression: true,
            compressor_config: CompressorConfig::new(-15.0, 3.0)
                .with_timing(5.0, 80.0)
                .with_soft_knee(4.0),
            enable_limiting: true,
            limiter_ceiling_db: -1.0,
            ducking_config: DuckingConfig::broadcast(),
        }
    }

    /// Create gentle mixing preset.
    #[must_use]
    pub fn gentle() -> Self {
        Self {
            mode: MixingMode::Mix,
            main_level: 0.5,
            ad_level: 1.0,
            crossfade_ms: 50.0,
            enable_compression: false,
            compressor_config: CompressorConfig::default(),
            enable_limiting: false,
            limiter_ceiling_db: -0.5,
            ducking_config: DuckingConfig::gentle(),
        }
    }
}

/// Audio mixer for combining main and AD tracks.
pub struct AudioMixer {
    /// Mixing configuration.
    config: MixingConfig,
    /// Sample rate.
    sample_rate: f64,
    /// Number of channels.
    channels: usize,
    /// Automatic ducker (for Duck mode).
    ducker: Option<AutomaticDucker>,
    /// Compressor (for compression).
    compressor: Option<Compressor>,
    /// Crossfade state.
    crossfade_state: CrossfadeState,
    /// Output peak level (for metering).
    peak_level: f64,
}

impl AudioMixer {
    /// Create a new audio mixer.
    ///
    /// # Arguments
    ///
    /// * `config` - Mixing configuration
    /// * `sample_rate` - Sample rate in Hz
    /// * `channels` - Number of audio channels
    #[must_use]
    pub fn new(config: MixingConfig, sample_rate: f64, channels: usize) -> Self {
        let ducker = if config.mode == MixingMode::Duck {
            Some(AutomaticDucker::new(
                config.ducking_config.clone(),
                sample_rate,
                channels,
            ))
        } else {
            None
        };

        let compressor = if config.enable_compression {
            Some(Compressor::new(
                config.compressor_config.clone(),
                sample_rate,
                channels,
            ))
        } else {
            None
        };

        let crossfade_samples = (config.crossfade_ms * 0.001 * sample_rate) as usize;

        Self {
            config,
            sample_rate,
            channels,
            ducker,
            compressor,
            crossfade_state: CrossfadeState::new(crossfade_samples),
            peak_level: 0.0,
        }
    }

    /// Set mixing configuration.
    pub fn set_config(&mut self, config: MixingConfig) {
        if config.mode == MixingMode::Duck && self.ducker.is_none() {
            self.ducker = Some(AutomaticDucker::new(
                config.ducking_config.clone(),
                self.sample_rate,
                self.channels,
            ));
        } else if let Some(ref mut ducker) = self.ducker {
            ducker.set_config(config.ducking_config.clone());
        }

        if config.enable_compression && self.compressor.is_none() {
            self.compressor = Some(Compressor::new(
                config.compressor_config.clone(),
                self.sample_rate,
                self.channels,
            ));
        } else if let Some(ref mut compressor) = self.compressor {
            compressor.set_config(config.compressor_config.clone());
        }

        let crossfade_samples = (config.crossfade_ms * 0.001 * self.sample_rate) as usize;
        self.crossfade_state.set_duration(crossfade_samples);

        self.config = config;
    }

    /// Get current configuration.
    #[must_use]
    pub fn config(&self) -> &MixingConfig {
        &self.config
    }

    /// Get peak level (for metering).
    #[must_use]
    pub fn peak_level(&self) -> f64 {
        self.peak_level
    }

    /// Mix interleaved audio buffers.
    ///
    /// # Arguments
    ///
    /// * `main_audio` - Main audio buffer
    /// * `ad_audio` - Audio description buffer
    /// * `output` - Output buffer (must be same size as main_audio)
    /// * `num_samples` - Number of samples per channel
    /// * `ad_active` - Whether AD is currently active
    pub fn mix_interleaved(
        &mut self,
        main_audio: &[f64],
        ad_audio: &[f64],
        output: &mut [f64],
        num_samples: usize,
        ad_active: bool,
    ) -> AudioResult<()> {
        if output.len() < main_audio.len() {
            return Err(AudioError::BufferTooSmall {
                needed: main_audio.len(),
                have: output.len(),
            });
        }

        match self.config.mode {
            MixingMode::Replace => {
                self.mix_replace_interleaved(main_audio, ad_audio, output, num_samples, ad_active)
            }
            MixingMode::Mix => {
                self.mix_blend_interleaved(main_audio, ad_audio, output, num_samples)
            }
            MixingMode::Duck => {
                self.mix_duck_interleaved(main_audio, ad_audio, output, num_samples, ad_active)
            }
            MixingMode::Pause => {
                self.mix_pause_interleaved(main_audio, ad_audio, output, num_samples, ad_active)
            }
        }
    }

    /// Mix planar audio buffers.
    ///
    /// # Arguments
    ///
    /// * `main_channels` - Main audio channels
    /// * `ad_channels` - Audio description channels
    /// * `output_channels` - Output channels
    /// * `ad_active` - Whether AD is currently active
    pub fn mix_planar(
        &mut self,
        main_channels: &[Vec<f64>],
        ad_channels: &[Vec<f64>],
        output_channels: &mut [Vec<f64>],
        ad_active: bool,
    ) -> AudioResult<()> {
        if output_channels.len() < main_channels.len() {
            return Err(AudioError::InvalidParameter(
                "Output must have at least as many channels as main".to_string(),
            ));
        }

        match self.config.mode {
            MixingMode::Replace => {
                self.mix_replace_planar(main_channels, ad_channels, output_channels, ad_active)
            }
            MixingMode::Mix => self.mix_blend_planar(main_channels, ad_channels, output_channels),
            MixingMode::Duck => {
                self.mix_duck_planar(main_channels, ad_channels, output_channels, ad_active)
            }
            MixingMode::Pause => {
                self.mix_pause_planar(main_channels, ad_channels, output_channels, ad_active)
            }
        }
    }

    fn mix_replace_interleaved(
        &mut self,
        main_audio: &[f64],
        ad_audio: &[f64],
        output: &mut [f64],
        num_samples: usize,
        ad_active: bool,
    ) -> AudioResult<()> {
        self.crossfade_state.update(ad_active);
        self.peak_level = 0.0;

        for i in 0..num_samples {
            let (main_gain, ad_gain) = self.crossfade_state.process();

            for ch in 0..self.channels {
                let idx = i * self.channels + ch;
                if idx < output.len() {
                    let main_sample = if idx < main_audio.len() {
                        main_audio[idx] * main_gain * self.config.main_level
                    } else {
                        0.0
                    };

                    let ad_sample = if idx < ad_audio.len() {
                        ad_audio[idx] * ad_gain * self.config.ad_level
                    } else {
                        0.0
                    };

                    output[idx] = main_sample + ad_sample;
                    self.peak_level = self.peak_level.max(output[idx].abs());
                }
            }
        }

        self.apply_processing(output, num_samples)?;
        Ok(())
    }

    fn mix_blend_interleaved(
        &mut self,
        main_audio: &[f64],
        ad_audio: &[f64],
        output: &mut [f64],
        num_samples: usize,
    ) -> AudioResult<()> {
        self.peak_level = 0.0;

        for i in 0..num_samples {
            for ch in 0..self.channels {
                let idx = i * self.channels + ch;
                if idx < output.len() {
                    let main_sample = if idx < main_audio.len() {
                        main_audio[idx] * self.config.main_level
                    } else {
                        0.0
                    };

                    let ad_sample = if idx < ad_audio.len() {
                        ad_audio[idx] * self.config.ad_level
                    } else {
                        0.0
                    };

                    output[idx] = main_sample + ad_sample;
                    self.peak_level = self.peak_level.max(output[idx].abs());
                }
            }
        }

        self.apply_processing(output, num_samples)?;
        Ok(())
    }

    fn mix_duck_interleaved(
        &mut self,
        main_audio: &[f64],
        ad_audio: &[f64],
        output: &mut [f64],
        num_samples: usize,
        ad_active: bool,
    ) -> AudioResult<()> {
        output[..main_audio.len()].copy_from_slice(main_audio);

        if let Some(ref mut ducker) = self.ducker {
            ducker.set_ad_active(ad_active);
            ducker.process_interleaved(output, ad_audio, num_samples);
        }

        self.peak_level = 0.0;
        for i in 0..num_samples {
            for ch in 0..self.channels {
                let idx = i * self.channels + ch;
                if idx < output.len() {
                    let ad_sample = if idx < ad_audio.len() {
                        ad_audio[idx] * self.config.ad_level
                    } else {
                        0.0
                    };

                    output[idx] += ad_sample;
                    self.peak_level = self.peak_level.max(output[idx].abs());
                }
            }
        }

        self.apply_processing(output, num_samples)?;
        Ok(())
    }

    fn mix_pause_interleaved(
        &mut self,
        main_audio: &[f64],
        ad_audio: &[f64],
        output: &mut [f64],
        num_samples: usize,
        ad_active: bool,
    ) -> AudioResult<()> {
        self.peak_level = 0.0;

        for i in 0..num_samples {
            for ch in 0..self.channels {
                let idx = i * self.channels + ch;
                if idx < output.len() {
                    output[idx] = if ad_active {
                        if idx < ad_audio.len() {
                            ad_audio[idx] * self.config.ad_level
                        } else {
                            0.0
                        }
                    } else if idx < main_audio.len() {
                        main_audio[idx] * self.config.main_level
                    } else {
                        0.0
                    };

                    self.peak_level = self.peak_level.max(output[idx].abs());
                }
            }
        }

        self.apply_processing(output, num_samples)?;
        Ok(())
    }

    fn mix_replace_planar(
        &mut self,
        main_channels: &[Vec<f64>],
        ad_channels: &[Vec<f64>],
        output_channels: &mut [Vec<f64>],
        ad_active: bool,
    ) -> AudioResult<()> {
        self.crossfade_state.update(ad_active);
        self.peak_level = 0.0;

        let num_samples = main_channels.first().map_or(0, Vec::len);

        for i in 0..num_samples {
            let (main_gain, ad_gain) = self.crossfade_state.process();

            for (ch, output_channel) in output_channels.iter_mut().enumerate() {
                if i < output_channel.len() {
                    let main_sample = if ch < main_channels.len() && i < main_channels[ch].len() {
                        main_channels[ch][i] * main_gain * self.config.main_level
                    } else {
                        0.0
                    };

                    let ad_sample = if ch < ad_channels.len() && i < ad_channels[ch].len() {
                        ad_channels[ch][i] * ad_gain * self.config.ad_level
                    } else {
                        0.0
                    };

                    output_channel[i] = main_sample + ad_sample;
                    self.peak_level = self.peak_level.max(output_channel[i].abs());
                }
            }
        }

        self.apply_processing_planar(output_channels)?;
        Ok(())
    }

    fn mix_blend_planar(
        &mut self,
        main_channels: &[Vec<f64>],
        ad_channels: &[Vec<f64>],
        output_channels: &mut [Vec<f64>],
    ) -> AudioResult<()> {
        self.peak_level = 0.0;

        let num_samples = main_channels.first().map_or(0, Vec::len);

        for (ch, output_channel) in output_channels.iter_mut().enumerate() {
            for i in 0..num_samples.min(output_channel.len()) {
                let main_sample = if ch < main_channels.len() && i < main_channels[ch].len() {
                    main_channels[ch][i] * self.config.main_level
                } else {
                    0.0
                };

                let ad_sample = if ch < ad_channels.len() && i < ad_channels[ch].len() {
                    ad_channels[ch][i] * self.config.ad_level
                } else {
                    0.0
                };

                output_channel[i] = main_sample + ad_sample;
                self.peak_level = self.peak_level.max(output_channel[i].abs());
            }
        }

        self.apply_processing_planar(output_channels)?;
        Ok(())
    }

    fn mix_duck_planar(
        &mut self,
        main_channels: &[Vec<f64>],
        ad_channels: &[Vec<f64>],
        output_channels: &mut [Vec<f64>],
        ad_active: bool,
    ) -> AudioResult<()> {
        for (ch, output_channel) in output_channels.iter_mut().enumerate() {
            if ch < main_channels.len() {
                output_channel.copy_from_slice(&main_channels[ch]);
            }
        }

        if let Some(ref mut ducker) = self.ducker {
            ducker.set_ad_active(ad_active);
            let ad_first_channel = ad_channels.first().map(Vec::as_slice).unwrap_or(&[]);
            ducker.process_planar(output_channels, ad_first_channel);
        }

        self.peak_level = 0.0;
        for (ch, output_channel) in output_channels.iter_mut().enumerate() {
            for i in 0..output_channel.len() {
                let ad_sample = if ch < ad_channels.len() && i < ad_channels[ch].len() {
                    ad_channels[ch][i] * self.config.ad_level
                } else {
                    0.0
                };

                output_channel[i] += ad_sample;
                self.peak_level = self.peak_level.max(output_channel[i].abs());
            }
        }

        self.apply_processing_planar(output_channels)?;
        Ok(())
    }

    fn mix_pause_planar(
        &mut self,
        main_channels: &[Vec<f64>],
        ad_channels: &[Vec<f64>],
        output_channels: &mut [Vec<f64>],
        ad_active: bool,
    ) -> AudioResult<()> {
        self.peak_level = 0.0;

        for (ch, output_channel) in output_channels.iter_mut().enumerate() {
            for i in 0..output_channel.len() {
                output_channel[i] = if ad_active {
                    if ch < ad_channels.len() && i < ad_channels[ch].len() {
                        ad_channels[ch][i] * self.config.ad_level
                    } else {
                        0.0
                    }
                } else if ch < main_channels.len() && i < main_channels[ch].len() {
                    main_channels[ch][i] * self.config.main_level
                } else {
                    0.0
                };

                self.peak_level = self.peak_level.max(output_channel[i].abs());
            }
        }

        self.apply_processing_planar(output_channels)?;
        Ok(())
    }

    fn apply_processing(&mut self, output: &mut [f64], num_samples: usize) -> AudioResult<()> {
        if let Some(ref mut compressor) = self.compressor {
            compressor.process_interleaved(output, num_samples);
        }

        if self.config.enable_limiting {
            self.apply_limiter_interleaved(output, num_samples);
        }

        Ok(())
    }

    fn apply_processing_planar(&mut self, output_channels: &mut [Vec<f64>]) -> AudioResult<()> {
        if let Some(ref mut compressor) = self.compressor {
            compressor.process_planar(output_channels);
        }

        if self.config.enable_limiting {
            self.apply_limiter_planar(output_channels);
        }

        Ok(())
    }

    fn apply_limiter_interleaved(&self, output: &mut [f64], num_samples: usize) {
        let ceiling = DuckingConfig::db_to_linear(self.config.limiter_ceiling_db);

        for i in 0..num_samples {
            for ch in 0..self.channels {
                let idx = i * self.channels + ch;
                if idx < output.len() {
                    if output[idx].abs() > ceiling {
                        output[idx] = ceiling * output[idx].signum();
                    }
                }
            }
        }
    }

    fn apply_limiter_planar(&self, output_channels: &mut [Vec<f64>]) {
        let ceiling = DuckingConfig::db_to_linear(self.config.limiter_ceiling_db);

        for channel in output_channels {
            for sample in channel {
                if sample.abs() > ceiling {
                    *sample = ceiling * sample.signum();
                }
            }
        }
    }

    /// Reset all processing state.
    pub fn reset(&mut self) {
        if let Some(ref mut ducker) = self.ducker {
            ducker.reset();
        }
        if let Some(ref mut compressor) = self.compressor {
            compressor.reset();
        }
        self.crossfade_state.reset();
        self.peak_level = 0.0;
    }
}

/// Crossfade state manager.
struct CrossfadeState {
    /// Crossfade duration in samples.
    duration_samples: usize,
    /// Current position (0 = main only, duration_samples = AD only).
    position: usize,
    /// Target position.
    target_position: usize,
}

impl CrossfadeState {
    fn new(duration_samples: usize) -> Self {
        Self {
            duration_samples,
            position: 0,
            target_position: 0,
        }
    }

    fn set_duration(&mut self, duration_samples: usize) {
        self.duration_samples = duration_samples;
    }

    fn update(&mut self, ad_active: bool) {
        self.target_position = if ad_active { self.duration_samples } else { 0 };
    }

    fn process(&mut self) -> (f64, f64) {
        if self.position < self.target_position {
            self.position = self.position.saturating_add(1).min(self.target_position);
        } else if self.position > self.target_position {
            self.position = self.position.saturating_sub(1);
        }

        let progress = if self.duration_samples > 0 {
            self.position as f64 / self.duration_samples as f64
        } else if self.target_position > 0 {
            1.0
        } else {
            0.0
        };

        let ad_gain = progress;
        let main_gain = 1.0 - progress;

        (main_gain, ad_gain)
    }

    fn reset(&mut self) {
        self.position = 0;
        self.target_position = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mixing_config() {
        let config = MixingConfig::new(MixingMode::Duck);
        assert_eq!(config.mode, MixingMode::Duck);
    }

    #[test]
    fn test_crossfade_state() {
        let mut state = CrossfadeState::new(100);
        state.update(true);

        let (main, ad) = state.process();
        assert!(main > ad);

        for _ in 0..100 {
            state.process();
        }

        let (main, ad) = state.process();
        assert!((main - 0.0).abs() < 0.01);
        assert!((ad - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_audio_mixer_blend() {
        let config = MixingConfig::new(MixingMode::Mix);
        let mut mixer = AudioMixer::new(config, 48000.0, 2);

        let main_audio = vec![0.5; 200];
        let ad_audio = vec![0.3; 200];
        let mut output = vec![0.0; 200];

        mixer
            .mix_interleaved(&main_audio, &ad_audio, &mut output, 100, false)
            .expect("should succeed");

        assert!(output.iter().any(|&s| s > 0.0));
    }
}
