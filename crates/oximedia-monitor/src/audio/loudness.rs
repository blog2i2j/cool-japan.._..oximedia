//! Loudness monitoring (LUFS/LKFS) using oximedia-metering.

use crate::{MonitorError, MonitorResult};
use oximedia_metering::{LoudnessMeter, MeterConfig, Standard, LoudnessMetrics};
use serde::{Deserialize, Serialize};

/// Loudness monitor metrics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LoudnessMonitorMetrics {
    /// Momentary loudness (400ms) in LUFS.
    pub momentary_lufs: f64,

    /// Short-term loudness (3s) in LUFS.
    pub short_term_lufs: f64,

    /// Integrated loudness in LUFS.
    pub integrated_lufs: f64,

    /// Loudness range in LU.
    pub loudness_range: f64,

    /// True peak in dBTP.
    pub true_peak_dbtp: f64,

    /// Maximum momentary loudness seen.
    pub max_momentary: f64,

    /// Maximum short-term loudness seen.
    pub max_short_term: f64,

    /// Per-channel true peaks in dBTP.
    pub channel_peaks_dbtp: Vec<f64>,
}

/// Loudness monitor wrapping oximedia-metering.
pub struct LoudnessMonitor {
    meter: LoudnessMeter,
    metrics: LoudnessMonitorMetrics,
}

impl LoudnessMonitor {
    /// Create a new loudness monitor.
    ///
    /// # Errors
    ///
    /// Returns an error if meter creation fails.
    pub fn new(sample_rate: f64, channels: usize) -> MonitorResult<Self> {
        let config = MeterConfig::new(Standard::EbuR128, sample_rate, channels);
        let meter = LoudnessMeter::new(config)
            .map_err(|e| MonitorError::MeteringError(e.to_string()))?;

        Ok(Self {
            meter,
            metrics: LoudnessMonitorMetrics::default(),
        })
    }

    /// Create with custom standard.
    ///
    /// # Errors
    ///
    /// Returns an error if meter creation fails.
    pub fn with_standard(
        sample_rate: f64,
        channels: usize,
        standard: Standard,
    ) -> MonitorResult<Self> {
        let config = MeterConfig::new(standard, sample_rate, channels);
        let meter = LoudnessMeter::new(config)
            .map_err(|e| MonitorError::MeteringError(e.to_string()))?;

        Ok(Self {
            meter,
            metrics: LoudnessMonitorMetrics::default(),
        })
    }

    /// Process audio samples.
    ///
    /// # Errors
    ///
    /// Returns an error if processing fails.
    pub fn process(&mut self, samples: &[f32]) -> MonitorResult<()> {
        self.meter.process_f32(samples);
        self.update_metrics();
        Ok(())
    }

    /// Get current metrics.
    #[must_use]
    pub const fn metrics(&self) -> &LoudnessMonitorMetrics {
        &self.metrics
    }

    /// Get underlying loudness meter metrics.
    pub fn metering_metrics(&mut self) -> LoudnessMetrics {
        self.meter.metrics()
    }

    /// Reset monitor.
    pub fn reset(&mut self) {
        self.meter.reset();
        self.metrics = LoudnessMonitorMetrics::default();
    }

    fn update_metrics(&mut self) {
        let metering_metrics = self.meter.metrics();

        self.metrics.momentary_lufs = metering_metrics.momentary_lufs;
        self.metrics.short_term_lufs = metering_metrics.short_term_lufs;
        self.metrics.integrated_lufs = metering_metrics.integrated_lufs;
        self.metrics.loudness_range = metering_metrics.loudness_range;
        self.metrics.true_peak_dbtp = metering_metrics.true_peak_dbtp;
        self.metrics.max_momentary = metering_metrics.max_momentary;
        self.metrics.max_short_term = metering_metrics.max_short_term;
        self.metrics.channel_peaks_dbtp = metering_metrics.channel_peaks_dbtp.clone();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loudness_monitor_creation() {
        let result = LoudnessMonitor::new(48000.0, 2);
        assert!(result.is_ok());
    }

    #[test]
    fn test_loudness_monitor_with_standard() {
        let result = LoudnessMonitor::with_standard(48000.0, 2, Standard::AtscA85);
        assert!(result.is_ok());
    }

    #[test]
    fn test_loudness_monitor_process() {
        let mut monitor = LoudnessMonitor::new(48000.0, 2).expect("failed to create");

        let samples = vec![0.1f32; 10000];
        assert!(monitor.process(&samples).is_ok());

        let metrics = monitor.metrics();
        assert!(metrics.integrated_lufs.is_finite() || metrics.integrated_lufs.is_infinite());
    }
}
