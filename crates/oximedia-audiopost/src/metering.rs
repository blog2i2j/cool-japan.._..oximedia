#![allow(dead_code)]
//! Professional audio metering and analysis tools.

use crate::error::{AudioPostError, AudioPostResult};
use std::collections::VecDeque;

/// Peak meter
#[derive(Debug)]
pub struct PeakMeter {
    sample_rate: u32,
    peak: f32,
    hold_time_ms: f32,
    hold_counter: usize,
    hold_value: f32,
    decay_rate: f32,
}

impl PeakMeter {
    /// Create a new peak meter
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
            peak: -std::f32::INFINITY,
            hold_time_ms: 1500.0,
            hold_counter: 0,
            hold_value: -std::f32::INFINITY,
            decay_rate: 0.99,
        })
    }

    /// Process audio samples
    pub fn process(&mut self, samples: &[f32]) {
        for &sample in samples {
            let abs_sample = sample.abs();

            if abs_sample > self.peak {
                self.peak = abs_sample;
            }

            if abs_sample > self.hold_value {
                self.hold_value = abs_sample;
                self.hold_counter = (self.hold_time_ms / 1000.0 * self.sample_rate as f32) as usize;
            }
        }

        // Apply decay
        self.peak *= self.decay_rate;

        // Update hold
        if self.hold_counter > 0 {
            self.hold_counter -= samples.len();
        } else {
            self.hold_value *= self.decay_rate;
        }
    }

    /// Get current peak in dB
    #[must_use]
    pub fn get_peak_db(&self) -> f32 {
        20.0 * self.peak.max(1e-10).log10()
    }

    /// Get hold peak in dB
    #[must_use]
    pub fn get_hold_peak_db(&self) -> f32 {
        20.0 * self.hold_value.max(1e-10).log10()
    }

    /// Reset meter
    pub fn reset(&mut self) {
        self.peak = -std::f32::INFINITY;
        self.hold_value = -std::f32::INFINITY;
        self.hold_counter = 0;
    }
}

/// RMS meter
#[derive(Debug)]
pub struct RmsMeter {
    sample_rate: u32,
    window_size: usize,
    buffer: VecDeque<f32>,
    sum_squares: f64,
}

impl RmsMeter {
    /// Create a new RMS meter
    ///
    /// # Errors
    ///
    /// Returns an error if sample rate is invalid
    pub fn new(sample_rate: u32, window_ms: f32) -> AudioPostResult<Self> {
        if sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(sample_rate));
        }

        let window_size = (sample_rate as f32 * window_ms / 1000.0) as usize;

        Ok(Self {
            sample_rate,
            window_size,
            buffer: VecDeque::with_capacity(window_size),
            sum_squares: 0.0,
        })
    }

    /// Process audio samples
    pub fn process(&mut self, samples: &[f32]) {
        for &sample in samples {
            let square = f64::from(sample * sample);

            // Add new sample
            self.buffer.push_back(sample);
            self.sum_squares += square;

            // Remove old sample if window is full
            if self.buffer.len() > self.window_size {
                if let Some(old) = self.buffer.pop_front() {
                    self.sum_squares -= f64::from(old * old);
                }
            }
        }
    }

    /// Get current RMS in dB
    #[must_use]
    pub fn get_rms_db(&self) -> f32 {
        if self.buffer.is_empty() {
            return -std::f32::INFINITY;
        }

        let rms = (self.sum_squares / self.buffer.len() as f64).sqrt() as f32;
        20.0 * rms.max(1e-10).log10()
    }

    /// Reset meter
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.sum_squares = 0.0;
    }
}

/// VU meter (Volume Unit)
#[derive(Debug)]
pub struct VuMeter {
    sample_rate: u32,
    ballistics_coefficient: f32,
    current_value: f32,
}

impl VuMeter {
    /// Create a new VU meter
    ///
    /// # Errors
    ///
    /// Returns an error if sample rate is invalid
    pub fn new(sample_rate: u32) -> AudioPostResult<Self> {
        if sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(sample_rate));
        }

        // VU meter has 300ms integration time
        let ballistics_coefficient = (-1.0 / (0.3 * sample_rate as f32)).exp();

        Ok(Self {
            sample_rate,
            ballistics_coefficient,
            current_value: 0.0,
        })
    }

    /// Process audio samples
    pub fn process(&mut self, samples: &[f32]) {
        for &sample in samples {
            let abs_sample = sample.abs();
            self.current_value = self.ballistics_coefficient * self.current_value
                + (1.0 - self.ballistics_coefficient) * abs_sample;
        }
    }

    /// Get current VU level in dB
    #[must_use]
    pub fn get_vu_db(&self) -> f32 {
        20.0 * self.current_value.max(1e-10).log10()
    }

    /// Reset meter
    pub fn reset(&mut self) {
        self.current_value = 0.0;
    }
}

/// True peak meter
#[derive(Debug)]
pub struct TruePeakMeter {
    sample_rate: u32,
    oversample_factor: usize,
    peak: f32,
}

impl TruePeakMeter {
    /// Create a new true peak meter
    ///
    /// # Errors
    ///
    /// Returns an error if sample rate is invalid
    pub fn new(sample_rate: u32, oversample_factor: usize) -> AudioPostResult<Self> {
        if sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(sample_rate));
        }

        Ok(Self {
            sample_rate,
            oversample_factor,
            peak: -std::f32::INFINITY,
        })
    }

    /// Process audio samples
    pub fn process(&mut self, samples: &[f32]) {
        // Simple implementation - real version would use proper oversampling
        for &sample in samples {
            let abs_sample = sample.abs();
            if abs_sample > self.peak {
                self.peak = abs_sample;
            }
        }
    }

    /// Get true peak in dBTP
    #[must_use]
    pub fn get_true_peak_dbtp(&self) -> f32 {
        20.0 * self.peak.max(1e-10).log10()
    }

    /// Reset meter
    pub fn reset(&mut self) {
        self.peak = -std::f32::INFINITY;
    }
}

/// Phase correlation meter
#[derive(Debug)]
pub struct PhaseCorrelationMeter {
    sample_rate: u32,
    correlation: f32,
    alpha: f32,
}

impl PhaseCorrelationMeter {
    /// Create a new phase correlation meter
    ///
    /// # Errors
    ///
    /// Returns an error if sample rate is invalid
    pub fn new(sample_rate: u32) -> AudioPostResult<Self> {
        if sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(sample_rate));
        }

        // Smoothing coefficient
        let alpha = (-1.0 / (0.3 * sample_rate as f32)).exp();

        Ok(Self {
            sample_rate,
            correlation: 0.0,
            alpha,
        })
    }

    /// Process stereo audio samples
    pub fn process(&mut self, left: &[f32], right: &[f32]) {
        for (&l, &r) in left.iter().zip(right.iter()) {
            let mid = (l + r) / 2.0;
            let side = (l - r) / 2.0;

            let mid_sq = mid * mid;
            let side_sq = side * side;

            let sum = mid_sq + side_sq;
            let correlation = if sum > 1e-10 {
                (mid_sq - side_sq) / sum
            } else {
                0.0
            };

            self.correlation = self.alpha * self.correlation + (1.0 - self.alpha) * correlation;
        }
    }

    /// Get phase correlation (-1.0 to 1.0)
    #[must_use]
    pub fn get_correlation(&self) -> f32 {
        self.correlation.clamp(-1.0, 1.0)
    }

    /// Reset meter
    pub fn reset(&mut self) {
        self.correlation = 0.0;
    }
}

/// Stereo width meter
#[derive(Debug)]
pub struct StereoWidthMeter {
    sample_rate: u32,
    width: f32,
    alpha: f32,
}

impl StereoWidthMeter {
    /// Create a new stereo width meter
    ///
    /// # Errors
    ///
    /// Returns an error if sample rate is invalid
    pub fn new(sample_rate: u32) -> AudioPostResult<Self> {
        if sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(sample_rate));
        }

        let alpha = (-1.0 / (0.3 * sample_rate as f32)).exp();

        Ok(Self {
            sample_rate,
            width: 0.0,
            alpha,
        })
    }

    /// Process stereo audio samples
    pub fn process(&mut self, left: &[f32], right: &[f32]) {
        for (&l, &r) in left.iter().zip(right.iter()) {
            let mid = (l + r).abs();
            let side = (l - r).abs();

            // When mid is zero (fully out-of-phase), width is maximum (2.0)
            let width = if mid > 1e-10 {
                side / mid
            } else if side > 1e-10 {
                2.0
            } else {
                0.0
            };

            self.width = self.alpha * self.width + (1.0 - self.alpha) * width;
        }
    }

    /// Get stereo width (0.0 = mono, 1.0 = normal, >1.0 = enhanced)
    #[must_use]
    pub fn get_width(&self) -> f32 {
        self.width
    }

    /// Reset meter
    pub fn reset(&mut self) {
        self.width = 0.0;
    }
}

/// Spectrum analyzer
#[derive(Debug)]
pub struct SpectrumAnalyzer {
    sample_rate: u32,
    fft_size: usize,
    bins: Vec<f32>,
}

impl SpectrumAnalyzer {
    /// Create a new spectrum analyzer
    ///
    /// # Errors
    ///
    /// Returns an error if sample rate or FFT size is invalid
    pub fn new(sample_rate: u32, fft_size: usize) -> AudioPostResult<Self> {
        if sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(sample_rate));
        }
        if !fft_size.is_power_of_two() {
            return Err(AudioPostError::InvalidBufferSize(fft_size));
        }

        Ok(Self {
            sample_rate,
            fft_size,
            bins: vec![0.0; fft_size / 2],
        })
    }

    /// Process audio samples
    pub fn process(&mut self, _samples: &[f32]) {
        // Placeholder - real implementation would use FFT
    }

    /// Get magnitude for frequency bin
    #[must_use]
    pub fn get_bin(&self, bin: usize) -> f32 {
        self.bins.get(bin).copied().unwrap_or(0.0)
    }

    /// Get frequency for bin
    #[must_use]
    pub fn bin_to_frequency(&self, bin: usize) -> f32 {
        bin as f32 * self.sample_rate as f32 / self.fft_size as f32
    }

    /// Get bin for frequency
    #[must_use]
    pub fn frequency_to_bin(&self, frequency: f32) -> usize {
        (frequency * self.fft_size as f32 / self.sample_rate as f32) as usize
    }
}

/// Goniometer for stereo imaging
#[derive(Debug)]
pub struct Goniometer {
    sample_rate: u32,
    history_size: usize,
    points: VecDeque<(f32, f32)>,
}

impl Goniometer {
    /// Create a new goniometer
    ///
    /// # Errors
    ///
    /// Returns an error if sample rate is invalid
    pub fn new(sample_rate: u32, history_ms: f32) -> AudioPostResult<Self> {
        if sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(sample_rate));
        }

        let history_size = (sample_rate as f32 * history_ms / 1000.0) as usize;

        Ok(Self {
            sample_rate,
            history_size,
            points: VecDeque::with_capacity(history_size),
        })
    }

    /// Process stereo samples
    pub fn process(&mut self, left: &[f32], right: &[f32]) {
        for (&l, &r) in left.iter().zip(right.iter()) {
            self.points.push_back((l, r));

            if self.points.len() > self.history_size {
                self.points.pop_front();
            }
        }
    }

    /// Get history points
    #[must_use]
    pub fn get_points(&self) -> Vec<(f32, f32)> {
        self.points.iter().copied().collect()
    }

    /// Clear history
    pub fn clear(&mut self) {
        self.points.clear();
    }
}

/// Loudness range meter (LRA)
#[derive(Debug)]
pub struct LoudnessRangeMeter {
    sample_rate: u32,
    measurements: Vec<f32>,
}

impl LoudnessRangeMeter {
    /// Create a new loudness range meter
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
            measurements: Vec::new(),
        })
    }

    /// Add a loudness measurement
    pub fn add_measurement(&mut self, loudness_lu: f32) {
        self.measurements.push(loudness_lu);
    }

    /// Calculate loudness range (LRA)
    #[must_use]
    pub fn calculate_lra(&self) -> f32 {
        if self.measurements.is_empty() {
            return 0.0;
        }

        let mut sorted = self.measurements.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Calculate 10th and 95th percentiles
        let len = sorted.len();
        let index_10 = (len as f32 * 0.1) as usize;
        let index_95 = (len as f32 * 0.95) as usize;

        sorted[index_95] - sorted[index_10]
    }

    /// Reset measurements
    pub fn reset(&mut self) {
        self.measurements.clear();
    }
}

/// Multi-channel meter
#[derive(Debug)]
pub struct MultiChannelMeter {
    sample_rate: u32,
    channel_count: usize,
    peak_meters: Vec<PeakMeter>,
    rms_meters: Vec<RmsMeter>,
}

impl MultiChannelMeter {
    /// Create a new multi-channel meter
    ///
    /// # Errors
    ///
    /// Returns an error if sample rate is invalid
    pub fn new(sample_rate: u32, channel_count: usize) -> AudioPostResult<Self> {
        if sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(sample_rate));
        }

        let peak_meters: Result<Vec<_>, _> = (0..channel_count)
            .map(|_| PeakMeter::new(sample_rate))
            .collect();

        let rms_meters: Result<Vec<_>, _> = (0..channel_count)
            .map(|_| RmsMeter::new(sample_rate, 300.0))
            .collect();

        Ok(Self {
            sample_rate,
            channel_count,
            peak_meters: peak_meters?,
            rms_meters: rms_meters?,
        })
    }

    /// Process multi-channel audio
    pub fn process(&mut self, channels: &[Vec<f32>]) {
        for (i, channel) in channels.iter().enumerate().take(self.channel_count) {
            if let Some(peak_meter) = self.peak_meters.get_mut(i) {
                peak_meter.process(channel);
            }
            if let Some(rms_meter) = self.rms_meters.get_mut(i) {
                rms_meter.process(channel);
            }
        }
    }

    /// Get peak for channel
    #[must_use]
    pub fn get_channel_peak_db(&self, channel: usize) -> Option<f32> {
        self.peak_meters.get(channel).map(PeakMeter::get_peak_db)
    }

    /// Get RMS for channel
    #[must_use]
    pub fn get_channel_rms_db(&self, channel: usize) -> Option<f32> {
        self.rms_meters.get(channel).map(RmsMeter::get_rms_db)
    }

    /// Reset all meters
    pub fn reset(&mut self) {
        for meter in &mut self.peak_meters {
            meter.reset();
        }
        for meter in &mut self.rms_meters {
            meter.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_peak_meter_creation() {
        let meter = PeakMeter::new(48000).expect("failed to create");
        assert!(meter.get_peak_db().is_finite());
    }

    #[test]
    fn test_peak_meter_process() {
        let mut meter = PeakMeter::new(48000).expect("failed to create");
        let samples = vec![0.5_f32; 100];
        meter.process(&samples);
        assert!(meter.get_peak_db() > -100.0);
    }

    #[test]
    fn test_peak_meter_reset() {
        let mut meter = PeakMeter::new(48000).expect("failed to create");
        let samples = vec![0.5_f32; 100];
        meter.process(&samples);
        meter.reset();
        assert!(meter.get_peak_db() < -60.0);
    }

    #[test]
    fn test_rms_meter_creation() {
        let meter = RmsMeter::new(48000, 300.0).expect("failed to create");
        assert!(meter.get_rms_db().is_infinite());
    }

    #[test]
    fn test_rms_meter_process() {
        let mut meter = RmsMeter::new(48000, 300.0).expect("failed to create");
        let samples = vec![0.5_f32; 1000];
        meter.process(&samples);
        assert!(meter.get_rms_db() > -100.0);
    }

    #[test]
    fn test_rms_meter_reset() {
        let mut meter = RmsMeter::new(48000, 300.0).expect("failed to create");
        let samples = vec![0.5_f32; 1000];
        meter.process(&samples);
        meter.reset();
        assert!(meter.get_rms_db().is_infinite());
    }

    #[test]
    fn test_vu_meter_creation() {
        let meter = VuMeter::new(48000).expect("failed to create");
        assert!(meter.get_vu_db() < -60.0);
    }

    #[test]
    fn test_vu_meter_process() {
        let mut meter = VuMeter::new(48000).expect("failed to create");
        let samples = vec![0.5_f32; 1000];
        meter.process(&samples);
        assert!(meter.get_vu_db() > -100.0);
    }

    #[test]
    fn test_vu_meter_reset() {
        let mut meter = VuMeter::new(48000).expect("failed to create");
        let samples = vec![0.5_f32; 1000];
        meter.process(&samples);
        meter.reset();
        assert_eq!(meter.current_value, 0.0);
    }

    #[test]
    fn test_true_peak_meter_creation() {
        let meter = TruePeakMeter::new(48000, 4).expect("failed to create");
        assert!(meter.get_true_peak_dbtp().is_finite());
    }

    #[test]
    fn test_true_peak_meter_process() {
        let mut meter = TruePeakMeter::new(48000, 4).expect("failed to create");
        let samples = vec![0.8_f32; 100];
        meter.process(&samples);
        assert!(meter.get_true_peak_dbtp() > -10.0);
    }

    #[test]
    fn test_phase_correlation_meter_creation() {
        let meter = PhaseCorrelationMeter::new(48000).expect("failed to create");
        assert_eq!(meter.get_correlation(), 0.0);
    }

    #[test]
    fn test_phase_correlation_meter_process() {
        let mut meter = PhaseCorrelationMeter::new(48000).expect("failed to create");
        let left = vec![1.0_f32; 100];
        let right = vec![1.0_f32; 100];
        meter.process(&left, &right);
        assert!(meter.get_correlation() > 0.0);
    }

    #[test]
    fn test_stereo_width_meter_creation() {
        let meter = StereoWidthMeter::new(48000).expect("failed to create");
        assert_eq!(meter.get_width(), 0.0);
    }

    #[test]
    fn test_stereo_width_meter_process() {
        let mut meter = StereoWidthMeter::new(48000).expect("failed to create");
        let left = vec![1.0_f32; 100];
        let right = vec![-1.0_f32; 100];
        meter.process(&left, &right);
        assert!(meter.get_width() > 0.0);
    }

    #[test]
    fn test_spectrum_analyzer_creation() {
        let analyzer = SpectrumAnalyzer::new(48000, 1024).expect("failed to create");
        assert_eq!(analyzer.bins.len(), 512);
    }

    #[test]
    fn test_spectrum_analyzer_bin_to_frequency() {
        let analyzer = SpectrumAnalyzer::new(48000, 1024).expect("failed to create");
        let freq = analyzer.bin_to_frequency(100);
        assert!((freq - 4687.5).abs() < 0.1);
    }

    #[test]
    fn test_spectrum_analyzer_frequency_to_bin() {
        let analyzer = SpectrumAnalyzer::new(48000, 1024).expect("failed to create");
        let bin = analyzer.frequency_to_bin(1000.0);
        assert_eq!(bin, 21);
    }

    #[test]
    fn test_goniometer_creation() {
        let goniometer = Goniometer::new(48000, 100.0).expect("failed to create");
        assert_eq!(goniometer.get_points().len(), 0);
    }

    #[test]
    fn test_goniometer_process() {
        let mut goniometer = Goniometer::new(48000, 100.0).expect("failed to create");
        let left = vec![1.0_f32; 100];
        let right = vec![0.5_f32; 100];
        goniometer.process(&left, &right);
        assert!(!goniometer.get_points().is_empty());
    }

    #[test]
    fn test_goniometer_clear() {
        let mut goniometer = Goniometer::new(48000, 100.0).expect("failed to create");
        let left = vec![1.0_f32; 100];
        let right = vec![0.5_f32; 100];
        goniometer.process(&left, &right);
        goniometer.clear();
        assert_eq!(goniometer.get_points().len(), 0);
    }

    #[test]
    fn test_loudness_range_meter_creation() {
        let meter = LoudnessRangeMeter::new(48000).expect("failed to create");
        assert_eq!(meter.calculate_lra(), 0.0);
    }

    #[test]
    fn test_loudness_range_meter_measurements() {
        let mut meter = LoudnessRangeMeter::new(48000).expect("failed to create");
        meter.add_measurement(-23.0);
        meter.add_measurement(-25.0);
        meter.add_measurement(-22.0);
        let lra = meter.calculate_lra();
        assert!(lra > 0.0);
    }

    #[test]
    fn test_loudness_range_meter_reset() {
        let mut meter = LoudnessRangeMeter::new(48000).expect("failed to create");
        meter.add_measurement(-23.0);
        meter.reset();
        assert_eq!(meter.calculate_lra(), 0.0);
    }

    #[test]
    fn test_multi_channel_meter_creation() {
        let meter = MultiChannelMeter::new(48000, 8).expect("failed to create");
        assert_eq!(meter.channel_count, 8);
    }

    #[test]
    fn test_multi_channel_meter_process() {
        let mut meter = MultiChannelMeter::new(48000, 2).expect("failed to create");
        let channels = vec![vec![0.5_f32; 100], vec![0.3_f32; 100]];
        meter.process(&channels);
        assert!(meter.get_channel_peak_db(0).is_some());
    }

    #[test]
    fn test_multi_channel_meter_reset() {
        let mut meter = MultiChannelMeter::new(48000, 2).expect("failed to create");
        let channels = vec![vec![0.5_f32; 100], vec![0.3_f32; 100]];
        meter.process(&channels);
        meter.reset();
        assert!(
            meter
                .get_channel_peak_db(0)
                .expect("get_channel_peak_db should succeed")
                < -60.0
        );
    }

    #[test]
    fn test_invalid_sample_rate() {
        assert!(PeakMeter::new(0).is_err());
        assert!(RmsMeter::new(0, 300.0).is_err());
        assert!(VuMeter::new(0).is_err());
    }

    #[test]
    fn test_invalid_fft_size() {
        assert!(SpectrumAnalyzer::new(48000, 1000).is_err());
    }
}
