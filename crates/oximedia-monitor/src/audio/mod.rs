//! Audio metering and monitoring.
//!
//! This module provides comprehensive audio metering including VU, PPM, loudness,
//! phase correlation, and spectrum analysis.

pub mod vu;
pub mod ppm;
pub mod loudness;
pub mod phase;
pub mod spectrum;

use crate::{MonitorError, MonitorResult};
use oximedia_metering::{LoudnessMeter, MeterConfig, Standard, LoudnessMetrics};
use serde::{Deserialize, Serialize};

pub use vu::{VuMeter, VuMetrics};
pub use ppm::{PpmMeter, PpmMetrics, PpmStandard};
pub use loudness::{LoudnessMonitor, LoudnessMonitorMetrics};
pub use phase::{PhaseCorrelation, PhaseMetrics};
pub use spectrum::{SpectrumAnalyzer, SpectrumMetrics};

/// Audio meter type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AudioMeterType {
    /// VU meter (Volume Unit).
    Vu,

    /// PPM (Peak Programme Meter).
    Ppm,

    /// Loudness meter (LUFS).
    Loudness,

    /// Phase correlation meter.
    Phase,

    /// Spectrum analyzer.
    Spectrum,

    /// All meters.
    All,
}

/// Audio monitoring metrics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AudioMetrics {
    /// VU meter metrics.
    pub vu_metrics: VuMetrics,

    /// PPM metrics.
    pub ppm_metrics: PpmMetrics,

    /// Loudness metrics.
    pub loudness_metrics: LoudnessMonitorMetrics,

    /// Phase correlation metrics.
    pub phase_metrics: PhaseMetrics,

    /// Spectrum analyzer metrics.
    pub spectrum_metrics: SpectrumMetrics,
}

/// Comprehensive audio meter.
pub struct AudioMeter {
    sample_rate: f64,
    channels: usize,
    vu_meter: VuMeter,
    ppm_meter: PpmMeter,
    loudness_monitor: LoudnessMonitor,
    phase_correlation: PhaseCorrelation,
    spectrum_analyzer: SpectrumAnalyzer,
    metrics: AudioMetrics,
}

impl AudioMeter {
    /// Create a new audio meter.
    ///
    /// # Errors
    ///
    /// Returns an error if initialization fails.
    pub fn new(sample_rate: f64, channels: usize) -> MonitorResult<Self> {
        if channels == 0 || channels > 16 {
            return Err(MonitorError::InvalidConfig(
                format!("Invalid channel count: {}", channels),
            ));
        }

        Ok(Self {
            sample_rate,
            channels,
            vu_meter: VuMeter::new(sample_rate, channels),
            ppm_meter: PpmMeter::new(sample_rate, channels, PpmStandard::Ebu),
            loudness_monitor: LoudnessMonitor::new(sample_rate, channels)?,
            phase_correlation: PhaseCorrelation::new(),
            spectrum_analyzer: SpectrumAnalyzer::new(sample_rate, channels),
            metrics: AudioMetrics::default(),
        })
    }

    /// Process audio samples.
    ///
    /// # Errors
    ///
    /// Returns an error if processing fails.
    pub fn process_samples(&mut self, samples: &[f32]) -> MonitorResult<()> {
        if samples.len() % self.channels != 0 {
            return Err(MonitorError::ProcessingError(
                "Sample count must be multiple of channel count".to_string(),
            ));
        }

        // Process all meters
        self.vu_meter.process(samples);
        self.ppm_meter.process(samples);
        self.loudness_monitor.process(samples)?;

        // Phase correlation only for stereo
        if self.channels == 2 {
            self.phase_correlation.process(samples);
        }

        self.spectrum_analyzer.process(samples);

        // Update metrics
        self.update_metrics();

        Ok(())
    }

    /// Get current metrics.
    #[must_use]
    pub const fn metrics(&self) -> &AudioMetrics {
        &self.metrics
    }

    /// Reset all meters.
    pub fn reset(&mut self) {
        self.vu_meter.reset();
        self.ppm_meter.reset();
        self.loudness_monitor.reset();
        self.phase_correlation.reset();
        self.spectrum_analyzer.reset();
        self.metrics = AudioMetrics::default();
    }

    fn update_metrics(&mut self) {
        self.metrics.vu_metrics = self.vu_meter.metrics().clone();
        self.metrics.ppm_metrics = self.ppm_meter.metrics().clone();
        self.metrics.loudness_metrics = self.loudness_monitor.metrics().clone();
        self.metrics.phase_metrics = self.phase_correlation.metrics().clone();
        self.metrics.spectrum_metrics = self.spectrum_analyzer.metrics().clone();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_meter_creation() {
        let result = AudioMeter::new(48000.0, 2);
        assert!(result.is_ok());
    }

    #[test]
    fn test_audio_meter_invalid_channels() {
        let result = AudioMeter::new(48000.0, 0);
        assert!(result.is_err());

        let result = AudioMeter::new(48000.0, 20);
        assert!(result.is_err());
    }

    #[test]
    fn test_audio_meter_process() {
        let mut meter = AudioMeter::new(48000.0, 2).expect("failed to create");

        // Process some samples
        let samples = vec![0.5f32; 1000];
        assert!(meter.process_samples(&samples).is_ok());
    }
}
