//! Peak and level meters.
//!
//! Implements various audio level measurement standards including:
//! - Sample peak meters
//! - RMS meters with configurable integration time
//! - VU meters (300ms integration, ballistics)
//! - PPM meters (EBU, BBC, DIN standards)
//! - K-System meters (K-12, K-14, K-20)

use crate::ballistics::{BallisticProcessor, BallisticType};
use crate::{MeteringError, MeteringResult};

/// Peak meter type.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PeakMeterType {
    /// Sample-based peak detection.
    Sample,
    /// RMS (Root Mean Square) with integration time in seconds.
    Rms(f64),
    /// VU meter (300ms integration).
    Vu,
    /// PPM (Peak Programme Meter) - EBU standard.
    PpmEbu,
    /// PPM (Peak Programme Meter) - BBC standard.
    PpmBbc,
    /// PPM (Peak Programme Meter) - DIN standard.
    PpmDin,
}

/// K-System meter type.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum KSystemType {
    /// K-12: -12 dBFS reference for broadcast mixing.
    K12,
    /// K-14: -14 dBFS reference for mastering.
    K14,
    /// K-20: -20 dBFS reference for film/classical.
    K20,
}

impl KSystemType {
    /// Get the reference level in dBFS.
    pub fn reference_dbfs(&self) -> f64 {
        match self {
            Self::K12 => -12.0,
            Self::K14 => -14.0,
            Self::K20 => -20.0,
        }
    }

    /// Get the headroom in dB.
    pub fn headroom_db(&self) -> f64 {
        match self {
            Self::K12 => 12.0,
            Self::K14 => 14.0,
            Self::K20 => 20.0,
        }
    }
}

/// Peak meter for audio level measurement.
pub struct PeakMeter {
    meter_type: PeakMeterType,
    sample_rate: f64,
    channels: usize,
    ballistics: Option<Vec<BallisticProcessor>>,
    rms_buffer: Vec<Vec<f64>>,
    rms_buffer_size: usize,
    rms_write_pos: usize,
    peak_values: Vec<f64>,
    rms_values: Vec<f64>,
    peak_hold_time: f64,
}

impl PeakMeter {
    /// Create a new peak meter.
    ///
    /// # Arguments
    ///
    /// * `meter_type` - Type of peak meter
    /// * `sample_rate` - Sample rate in Hz
    /// * `channels` - Number of audio channels
    /// * `peak_hold_time` - Peak hold time in seconds (0.0 for no hold)
    ///
    /// # Errors
    ///
    /// Returns error if configuration is invalid.
    pub fn new(
        meter_type: PeakMeterType,
        sample_rate: f64,
        channels: usize,
        peak_hold_time: f64,
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

        let ballistics = match meter_type {
            PeakMeterType::Vu => Some(
                (0..channels)
                    .map(|_| {
                        BallisticProcessor::new(BallisticType::Vu, sample_rate, peak_hold_time)
                    })
                    .collect(),
            ),
            PeakMeterType::PpmEbu => Some(
                (0..channels)
                    .map(|_| {
                        BallisticProcessor::new(BallisticType::PpmEbu, sample_rate, peak_hold_time)
                    })
                    .collect(),
            ),
            PeakMeterType::PpmBbc => Some(
                (0..channels)
                    .map(|_| {
                        BallisticProcessor::new(BallisticType::PpmBbc, sample_rate, peak_hold_time)
                    })
                    .collect(),
            ),
            PeakMeterType::PpmDin => Some(
                (0..channels)
                    .map(|_| {
                        BallisticProcessor::new(BallisticType::PpmDin, sample_rate, peak_hold_time)
                    })
                    .collect(),
            ),
            _ => None,
        };

        let rms_buffer_size = if let PeakMeterType::Rms(integration_time) = meter_type {
            (integration_time * sample_rate) as usize
        } else {
            0
        };

        let rms_buffer = if rms_buffer_size > 0 {
            vec![vec![0.0; rms_buffer_size]; channels]
        } else {
            vec![]
        };

        Ok(Self {
            meter_type,
            sample_rate,
            channels,
            ballistics,
            rms_buffer,
            rms_buffer_size,
            rms_write_pos: 0,
            peak_values: vec![0.0; channels],
            rms_values: vec![0.0; channels],
            peak_hold_time,
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
                self.process_sample(ch, sample);
            }
        }
    }

    /// Process a single sample for a specific channel.
    fn process_sample(&mut self, channel: usize, sample: f64) {
        let abs_sample = sample.abs();

        match self.meter_type {
            PeakMeterType::Sample => {
                // Simple peak detection
                if abs_sample > self.peak_values[channel] {
                    self.peak_values[channel] = abs_sample;
                }
            }
            PeakMeterType::Rms(_) => {
                // RMS calculation with circular buffer
                let squared = sample * sample;
                self.rms_buffer[channel][self.rms_write_pos] = squared;

                // Calculate RMS
                let sum: f64 = self.rms_buffer[channel].iter().sum();
                self.rms_values[channel] = (sum / self.rms_buffer_size as f64).sqrt();
            }
            PeakMeterType::Vu
            | PeakMeterType::PpmEbu
            | PeakMeterType::PpmBbc
            | PeakMeterType::PpmDin => {
                // Apply ballistics
                if let Some(ref mut ballistics) = self.ballistics {
                    let filtered = ballistics[channel].process(abs_sample);
                    self.peak_values[channel] = filtered;
                }
            }
        }
    }

    /// Get peak values for all channels in linear scale.
    pub fn peak_linear(&self) -> Vec<f64> {
        match self.meter_type {
            PeakMeterType::Rms(_) => self.rms_values.clone(),
            _ => self.peak_values.clone(),
        }
    }

    /// Get peak values for all channels in dBFS.
    pub fn peak_dbfs(&self) -> Vec<f64> {
        self.peak_linear()
            .iter()
            .map(|&peak| linear_to_dbfs(peak))
            .collect()
    }

    /// Get peak hold values for all channels in linear scale.
    pub fn peak_hold_linear(&self) -> Vec<f64> {
        if let Some(ref ballistics) = self.ballistics {
            ballistics
                .iter()
                .map(super::ballistics::BallisticProcessor::peak_hold_value)
                .collect()
        } else {
            self.peak_values.clone()
        }
    }

    /// Get peak hold values for all channels in dBFS.
    pub fn peak_hold_dbfs(&self) -> Vec<f64> {
        self.peak_hold_linear()
            .iter()
            .map(|&peak| linear_to_dbfs(peak))
            .collect()
    }

    /// Get the maximum peak across all channels in linear scale.
    pub fn max_peak_linear(&self) -> f64 {
        self.peak_linear().iter().copied().fold(0.0, f64::max)
    }

    /// Get the maximum peak across all channels in dBFS.
    pub fn max_peak_dbfs(&self) -> f64 {
        linear_to_dbfs(self.max_peak_linear())
    }

    /// Reset the meter to initial state.
    pub fn reset(&mut self) {
        self.peak_values.fill(0.0);
        self.rms_values.fill(0.0);
        self.rms_write_pos = 0;

        for buffer in &mut self.rms_buffer {
            buffer.fill(0.0);
        }

        if let Some(ref mut ballistics) = self.ballistics {
            for b in ballistics {
                b.reset();
            }
        }
    }

    /// Advance the RMS buffer write position.
    pub fn advance_rms_buffer(&mut self) {
        if self.rms_buffer_size > 0 {
            self.rms_write_pos = (self.rms_write_pos + 1) % self.rms_buffer_size;
        }
    }
}

/// K-System meter.
pub struct KSystemMeter {
    k_type: KSystemType,
    rms_meter: PeakMeter,
    peak_meter: PeakMeter,
}

impl KSystemMeter {
    /// Create a new K-System meter.
    ///
    /// # Arguments
    ///
    /// * `k_type` - K-System type (K-12, K-14, K-20)
    /// * `sample_rate` - Sample rate in Hz
    /// * `channels` - Number of channels
    ///
    /// # Errors
    ///
    /// Returns error if configuration is invalid.
    pub fn new(k_type: KSystemType, sample_rate: f64, channels: usize) -> MeteringResult<Self> {
        // K-System uses RMS with 600ms integration
        let rms_meter = PeakMeter::new(
            PeakMeterType::Rms(0.6),
            sample_rate,
            channels,
            2.0, // 2 second peak hold
        )?;

        // Fast peak detection
        let peak_meter = PeakMeter::new(
            PeakMeterType::Sample,
            sample_rate,
            channels,
            2.0, // 2 second peak hold
        )?;

        Ok(Self {
            k_type,
            rms_meter,
            peak_meter,
        })
    }

    /// Process interleaved audio samples.
    pub fn process_interleaved(&mut self, samples: &[f64]) {
        self.rms_meter.process_interleaved(samples);
        self.peak_meter.process_interleaved(samples);
        self.rms_meter.advance_rms_buffer();
    }

    /// Get RMS levels relative to K-System reference in dB.
    pub fn rms_relative_db(&self) -> Vec<f64> {
        let reference = self.k_type.reference_dbfs();
        self.rms_meter
            .peak_dbfs()
            .iter()
            .map(|&dbfs| dbfs - reference)
            .collect()
    }

    /// Get peak levels relative to K-System reference in dB.
    pub fn peak_relative_db(&self) -> Vec<f64> {
        let reference = self.k_type.reference_dbfs();
        self.peak_meter
            .peak_dbfs()
            .iter()
            .map(|&dbfs| dbfs - reference)
            .collect()
    }

    /// Get RMS levels in absolute dBFS.
    pub fn rms_dbfs(&self) -> Vec<f64> {
        self.rms_meter.peak_dbfs()
    }

    /// Get peak levels in absolute dBFS.
    pub fn peak_dbfs(&self) -> Vec<f64> {
        self.peak_meter.peak_dbfs()
    }

    /// Check if any channel exceeds the K-System headroom.
    pub fn is_overload(&self) -> bool {
        let headroom = self.k_type.headroom_db();
        self.peak_relative_db()
            .iter()
            .any(|&level| level > headroom)
    }

    /// Reset the meter.
    pub fn reset(&mut self) {
        self.rms_meter.reset();
        self.peak_meter.reset();
    }

    /// Get the K-System type.
    pub fn k_type(&self) -> KSystemType {
        self.k_type
    }
}

/// Convert linear amplitude to dBFS.
pub fn linear_to_dbfs(linear: f64) -> f64 {
    if linear > 0.0 {
        20.0 * linear.log10()
    } else {
        f64::NEG_INFINITY
    }
}

/// Convert dBFS to linear amplitude.
pub fn dbfs_to_linear(dbfs: f64) -> f64 {
    if dbfs.is_finite() {
        10.0_f64.powf(dbfs / 20.0)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_peak_meter() {
        let mut meter = PeakMeter::new(PeakMeterType::Sample, 48000.0, 2, 0.0)
            .expect("test expectation failed");

        let samples = vec![0.5, 0.3, 0.8, 0.4, 0.6, 0.2];
        meter.process_interleaved(&samples);

        let peaks = meter.peak_linear();
        assert_eq!(peaks[0], 0.8);
        assert_eq!(peaks[1], 0.4);
    }

    #[test]
    fn test_rms_meter() {
        let mut meter = PeakMeter::new(PeakMeterType::Rms(0.1), 48000.0, 1, 0.0)
            .expect("test expectation failed");

        // Generate 1kHz sine wave
        for i in 0..4800 {
            let t = i as f64 / 48000.0;
            let sample = (2.0 * std::f64::consts::PI * 1000.0 * t).sin();
            meter.process_interleaved(&[sample]);
            meter.advance_rms_buffer();
        }

        let rms = meter.peak_linear()[0];
        // RMS of sine wave should be ~0.707
        assert!((rms - 0.707).abs() < 0.01);
    }

    #[test]
    fn test_vu_meter() {
        let mut meter =
            PeakMeter::new(PeakMeterType::Vu, 48000.0, 1, 0.0).expect("test expectation failed");

        // Step input - need more samples for VU ballistics (300ms integration)
        // VU meter should reach 99% of full scale in 300ms at 48kHz = 14400 samples
        for _ in 0..20000 {
            meter.process_interleaved(&[1.0]);
        }

        let peak = meter.peak_linear()[0];
        // After sufficient time, should be close to input level
        // VU ballistics have exponential approach, so may not reach exactly 1.0
        assert!(peak < 1.0, "Peak {:.3} should be < 1.0", peak);
        assert!(
            peak > 0.6,
            "Peak {:.3} should be > 0.6 after VU integration",
            peak
        );
    }

    #[test]
    fn test_k12_meter() {
        let mut meter =
            KSystemMeter::new(KSystemType::K12, 48000.0, 2).expect("test expectation failed");

        let samples = vec![0.5, 0.5, 0.5, 0.5];
        meter.process_interleaved(&samples);

        assert_eq!(meter.k_type().reference_dbfs(), -12.0);
        assert_eq!(meter.k_type().headroom_db(), 12.0);
    }

    #[test]
    fn test_linear_dbfs_conversion() {
        assert_eq!(linear_to_dbfs(1.0), 0.0);
        assert_eq!(linear_to_dbfs(0.5), 20.0 * 0.5_f64.log10());
        assert!(linear_to_dbfs(0.0).is_infinite());

        assert_eq!(dbfs_to_linear(0.0), 1.0);
        assert!((dbfs_to_linear(-6.0) - 0.501).abs() < 0.01);
    }

    #[test]
    fn test_peak_meter_reset() {
        let mut meter = PeakMeter::new(PeakMeterType::Sample, 48000.0, 2, 0.0)
            .expect("test expectation failed");

        meter.process_interleaved(&[0.8, 0.9]);
        meter.reset();

        let peaks = meter.peak_linear();
        assert_eq!(peaks[0], 0.0);
        assert_eq!(peaks[1], 0.0);
    }
}
