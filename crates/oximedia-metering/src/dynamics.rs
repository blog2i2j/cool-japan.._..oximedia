//! Dynamic range measurement.
//!
//! Measures the dynamic range of audio signals using various methods.

use crate::{MeteringError, MeteringResult};

/// Dynamic range meter.
///
/// Measures the dynamic range of an audio signal using the difference
/// between peak levels and RMS levels.
pub struct DynamicRangeMeter {
    sample_rate: f64,
    channels: usize,
    peak_values: Vec<f64>,
    rms_values: Vec<f64>,
    rms_buffer: Vec<Vec<f64>>,
    rms_buffer_size: usize,
    rms_write_pos: usize,
}

impl DynamicRangeMeter {
    /// Create a new dynamic range meter.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate in Hz
    /// * `channels` - Number of channels
    /// * `rms_integration_time` - RMS integration time in seconds
    ///
    /// # Errors
    ///
    /// Returns error if configuration is invalid.
    pub fn new(
        sample_rate: f64,
        channels: usize,
        rms_integration_time: f64,
    ) -> MeteringResult<Self> {
        if sample_rate <= 0.0 {
            return Err(MeteringError::InvalidConfig(
                "Sample rate must be positive".to_string(),
            ));
        }

        if channels == 0 {
            return Err(MeteringError::InvalidConfig(
                "Must have at least one channel".to_string(),
            ));
        }

        let rms_buffer_size = (sample_rate * rms_integration_time) as usize;
        let rms_buffer = vec![vec![0.0; rms_buffer_size]; channels];

        Ok(Self {
            sample_rate,
            channels,
            peak_values: vec![0.0; channels],
            rms_values: vec![0.0; channels],
            rms_buffer,
            rms_buffer_size,
            rms_write_pos: 0,
        })
    }

    /// Process interleaved audio samples.
    ///
    /// # Arguments
    ///
    /// * `samples` - Interleaved audio samples
    pub fn process_interleaved(&mut self, samples: &[f64]) {
        let num_frames = samples.len() / self.channels;

        for frame_idx in 0..num_frames {
            for ch in 0..self.channels {
                let sample_idx = frame_idx * self.channels + ch;
                let sample = samples[sample_idx];

                // Update peak
                let abs_sample = sample.abs();
                if abs_sample > self.peak_values[ch] {
                    self.peak_values[ch] = abs_sample;
                }

                // Update RMS buffer
                let squared = sample * sample;
                self.rms_buffer[ch][self.rms_write_pos] = squared;
            }

            // Advance RMS write position
            self.rms_write_pos = (self.rms_write_pos + 1) % self.rms_buffer_size;
        }

        // Calculate RMS for all channels
        for ch in 0..self.channels {
            let sum: f64 = self.rms_buffer[ch].iter().sum();
            self.rms_values[ch] = (sum / self.rms_buffer_size as f64).sqrt();
        }
    }

    /// Get the dynamic range in dB for each channel.
    ///
    /// Dynamic range = Peak level - RMS level (in dB)
    pub fn dynamic_range_db(&self) -> Vec<f64> {
        self.peak_values
            .iter()
            .zip(&self.rms_values)
            .map(|(&peak, &rms)| {
                let peak_db = if peak > 0.0 {
                    20.0 * peak.log10()
                } else {
                    f64::NEG_INFINITY
                };

                let rms_db = if rms > 0.0 {
                    20.0 * rms.log10()
                } else {
                    f64::NEG_INFINITY
                };

                if peak_db.is_finite() && rms_db.is_finite() {
                    peak_db - rms_db
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Get the crest factor for each channel.
    ///
    /// Crest factor = Peak / RMS (linear ratio)
    pub fn crest_factor(&self) -> Vec<f64> {
        self.peak_values
            .iter()
            .zip(&self.rms_values)
            .map(|(&peak, &rms)| if rms > 0.0 { peak / rms } else { 0.0 })
            .collect()
    }

    /// Get the crest factor in dB for each channel.
    pub fn crest_factor_db(&self) -> Vec<f64> {
        self.crest_factor()
            .iter()
            .map(|&cf| {
                if cf > 0.0 {
                    20.0 * cf.log10()
                } else {
                    f64::NEG_INFINITY
                }
            })
            .collect()
    }

    /// Get peak values in dBFS.
    pub fn peak_dbfs(&self) -> Vec<f64> {
        self.peak_values
            .iter()
            .map(|&peak| {
                if peak > 0.0 {
                    20.0 * peak.log10()
                } else {
                    f64::NEG_INFINITY
                }
            })
            .collect()
    }

    /// Get RMS values in dBFS.
    pub fn rms_dbfs(&self) -> Vec<f64> {
        self.rms_values
            .iter()
            .map(|&rms| {
                if rms > 0.0 {
                    20.0 * rms.log10()
                } else {
                    f64::NEG_INFINITY
                }
            })
            .collect()
    }

    /// Reset the meter.
    pub fn reset(&mut self) {
        self.peak_values.fill(0.0);
        self.rms_values.fill(0.0);
        self.rms_write_pos = 0;

        for buffer in &mut self.rms_buffer {
            buffer.fill(0.0);
        }
    }
}

/// PLR (Peak to Loudness Ratio) meter.
///
/// Measures the difference between true peak and integrated loudness,
/// useful for assessing headroom and dynamics in mastered content.
pub struct PlrMeter {
    true_peak_dbtp: f64,
    integrated_lufs: f64,
}

impl PlrMeter {
    /// Create a new PLR meter.
    pub fn new() -> Self {
        Self {
            true_peak_dbtp: f64::NEG_INFINITY,
            integrated_lufs: f64::NEG_INFINITY,
        }
    }

    /// Update with true peak and integrated loudness values.
    ///
    /// # Arguments
    ///
    /// * `true_peak_dbtp` - True peak in dBTP
    /// * `integrated_lufs` - Integrated loudness in LUFS
    pub fn update(&mut self, true_peak_dbtp: f64, integrated_lufs: f64) {
        self.true_peak_dbtp = true_peak_dbtp;
        self.integrated_lufs = integrated_lufs;
    }

    /// Get the PLR value in dB.
    ///
    /// PLR = True Peak (dBTP) - Integrated Loudness (LUFS)
    pub fn plr_db(&self) -> f64 {
        if self.true_peak_dbtp.is_finite() && self.integrated_lufs.is_finite() {
            self.true_peak_dbtp - self.integrated_lufs
        } else {
            0.0
        }
    }

    /// Get the true peak value.
    pub fn true_peak_dbtp(&self) -> f64 {
        self.true_peak_dbtp
    }

    /// Get the integrated loudness value.
    pub fn integrated_lufs(&self) -> f64 {
        self.integrated_lufs
    }

    /// Reset the meter.
    pub fn reset(&mut self) {
        self.true_peak_dbtp = f64::NEG_INFINITY;
        self.integrated_lufs = f64::NEG_INFINITY;
    }
}

impl Default for PlrMeter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dynamic_range_meter() {
        let mut meter = DynamicRangeMeter::new(48000.0, 2, 0.1).expect("test expectation failed");

        // Generate test signal with known dynamics
        let mut samples = Vec::new();
        for i in 0..4800 {
            let t = i as f64 / 48000.0;
            let signal = (2.0 * std::f64::consts::PI * 1000.0 * t).sin() * 0.5;
            samples.push(signal);
            samples.push(signal);
        }

        meter.process_interleaved(&samples);

        let dr = meter.dynamic_range_db();
        assert!(dr[0] > 0.0);
        assert!(dr[0] < 10.0); // Sine wave has moderate dynamic range
    }

    #[test]
    fn test_crest_factor() {
        let mut meter = DynamicRangeMeter::new(48000.0, 1, 0.1).expect("test expectation failed");

        // Generate sine wave
        let mut samples = Vec::new();
        for i in 0..4800 {
            let t = i as f64 / 48000.0;
            let signal = (2.0 * std::f64::consts::PI * 1000.0 * t).sin();
            samples.push(signal);
        }

        meter.process_interleaved(&samples);

        let cf = meter.crest_factor()[0];
        // Sine wave crest factor should be ~1.414 (sqrt(2))
        assert!((cf - 1.414).abs() < 0.1);
    }

    #[test]
    fn test_plr_meter() {
        let mut meter = PlrMeter::new();

        meter.update(-1.0, -23.0);

        let plr = meter.plr_db();
        assert_eq!(plr, 22.0); // -1.0 - (-23.0) = 22.0
    }

    #[test]
    fn test_dynamic_range_reset() {
        let mut meter = DynamicRangeMeter::new(48000.0, 2, 0.1).expect("test expectation failed");

        meter.process_interleaved(&[0.5, 0.5, 0.5, 0.5]);
        meter.reset();

        let dr = meter.dynamic_range_db();
        assert_eq!(dr[0], 0.0);
        assert_eq!(dr[1], 0.0);
    }
}
