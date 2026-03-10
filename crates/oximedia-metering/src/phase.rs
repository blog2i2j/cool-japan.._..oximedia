//! Phase and stereo analysis meters.
//!
//! Implements:
//! - Phase correlation meter (-1 to +1)
//! - Goniometer (Lissajous) display data
//! - Phase scope (L vs R)
//! - Stereo width analysis

use crate::{MeteringError, MeteringResult};
use std::collections::VecDeque;

/// Phase correlation meter.
///
/// Measures the phase correlation between two channels (typically L and R).
/// Correlation ranges from -1 (completely out of phase) to +1 (completely in phase).
pub struct PhaseCorrelationMeter {
    sample_rate: f64,
    integration_time: f64,
    buffer_size: usize,
    left_buffer: VecDeque<f64>,
    right_buffer: VecDeque<f64>,
    correlation: f64,
}

impl PhaseCorrelationMeter {
    /// Create a new phase correlation meter.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate in Hz
    /// * `integration_time` - Integration window in seconds (typically 0.4 for momentary)
    ///
    /// # Errors
    ///
    /// Returns error if configuration is invalid.
    pub fn new(sample_rate: f64, integration_time: f64) -> MeteringResult<Self> {
        if sample_rate <= 0.0 {
            return Err(MeteringError::InvalidConfig(
                "Sample rate must be positive".to_string(),
            ));
        }

        if integration_time <= 0.0 {
            return Err(MeteringError::InvalidConfig(
                "Integration time must be positive".to_string(),
            ));
        }

        let buffer_size = (sample_rate * integration_time) as usize;

        Ok(Self {
            sample_rate,
            integration_time,
            buffer_size,
            left_buffer: VecDeque::with_capacity(buffer_size),
            right_buffer: VecDeque::with_capacity(buffer_size),
            correlation: 0.0,
        })
    }

    /// Process a stereo sample pair.
    ///
    /// # Arguments
    ///
    /// * `left` - Left channel sample
    /// * `right` - Right channel sample
    pub fn process(&mut self, left: f64, right: f64) {
        // Add to buffers
        if self.left_buffer.len() >= self.buffer_size {
            self.left_buffer.pop_front();
        }
        if self.right_buffer.len() >= self.buffer_size {
            self.right_buffer.pop_front();
        }

        self.left_buffer.push_back(left);
        self.right_buffer.push_back(right);

        // Calculate correlation when buffer is full
        if self.left_buffer.len() == self.buffer_size {
            self.update_correlation();
        }
    }

    /// Process interleaved stereo samples.
    ///
    /// # Arguments
    ///
    /// * `samples` - Interleaved stereo samples [L, R, L, R, ...]
    pub fn process_interleaved(&mut self, samples: &[f64]) {
        for chunk in samples.chunks_exact(2) {
            self.process(chunk[0], chunk[1]);
        }
    }

    /// Update the correlation calculation.
    fn update_correlation(&mut self) {
        let n = self.left_buffer.len();
        if n == 0 {
            self.correlation = 0.0;
            return;
        }

        // Calculate means
        let left_mean: f64 = self.left_buffer.iter().sum::<f64>() / n as f64;
        let right_mean: f64 = self.right_buffer.iter().sum::<f64>() / n as f64;

        // Calculate correlation coefficient
        let mut numerator = 0.0;
        let mut left_variance = 0.0;
        let mut right_variance = 0.0;

        for i in 0..n {
            let left_diff = self.left_buffer[i] - left_mean;
            let right_diff = self.right_buffer[i] - right_mean;

            numerator += left_diff * right_diff;
            left_variance += left_diff * left_diff;
            right_variance += right_diff * right_diff;
        }

        let denominator = (left_variance * right_variance).sqrt();

        self.correlation = if denominator > 0.0 {
            (numerator / denominator).clamp(-1.0, 1.0)
        } else {
            0.0
        };
    }

    /// Get the current phase correlation value.
    ///
    /// Returns a value from -1 (completely out of phase) to +1 (completely in phase).
    pub fn correlation(&self) -> f64 {
        self.correlation
    }

    /// Check if the signal is mono (correlation near +1).
    pub fn is_mono(&self) -> bool {
        self.correlation > 0.95
    }

    /// Check if the signal has phase issues (correlation < 0).
    pub fn has_phase_issues(&self) -> bool {
        self.correlation < 0.0
    }

    /// Reset the meter.
    pub fn reset(&mut self) {
        self.left_buffer.clear();
        self.right_buffer.clear();
        self.correlation = 0.0;
    }
}

/// Goniometer point for Lissajous display.
#[derive(Clone, Debug)]
pub struct GoniometerPoint {
    /// Mid (M = L + R) component.
    pub mid: f64,
    /// Side (S = L - R) component.
    pub side: f64,
}

/// Goniometer for stereo field visualization.
///
/// Provides data for a Lissajous display showing the stereo image.
pub struct Goniometer {
    sample_rate: f64,
    history_size: usize,
    points: VecDeque<GoniometerPoint>,
}

impl Goniometer {
    /// Create a new goniometer.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate in Hz
    /// * `history_duration` - History duration in seconds
    pub fn new(sample_rate: f64, history_duration: f64) -> Self {
        let history_size = (sample_rate * history_duration) as usize;

        Self {
            sample_rate,
            history_size,
            points: VecDeque::with_capacity(history_size),
        }
    }

    /// Process a stereo sample pair.
    ///
    /// # Arguments
    ///
    /// * `left` - Left channel sample
    /// * `right` - Right channel sample
    pub fn process(&mut self, left: f64, right: f64) {
        // Convert to Mid/Side
        let mid = (left + right) / 2.0;
        let side = (left - right) / 2.0;

        if self.points.len() >= self.history_size {
            self.points.pop_front();
        }

        self.points.push_back(GoniometerPoint { mid, side });
    }

    /// Process interleaved stereo samples.
    pub fn process_interleaved(&mut self, samples: &[f64]) {
        for chunk in samples.chunks_exact(2) {
            self.process(chunk[0], chunk[1]);
        }
    }

    /// Get the goniometer points for display.
    pub fn points(&self) -> &VecDeque<GoniometerPoint> {
        &self.points
    }

    /// Reset the goniometer.
    pub fn reset(&mut self) {
        self.points.clear();
    }
}

/// Stereo width analyzer.
pub struct StereoWidthAnalyzer {
    sample_rate: f64,
    correlation_meter: PhaseCorrelationMeter,
    mid_energy: f64,
    side_energy: f64,
    smoothing_factor: f64,
}

impl StereoWidthAnalyzer {
    /// Create a new stereo width analyzer.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Errors
    ///
    /// Returns error if configuration is invalid.
    pub fn new(sample_rate: f64) -> MeteringResult<Self> {
        let correlation_meter = PhaseCorrelationMeter::new(sample_rate, 0.4)?;

        Ok(Self {
            sample_rate,
            correlation_meter,
            mid_energy: 0.0,
            side_energy: 0.0,
            smoothing_factor: 0.99, // Smoothing coefficient
        })
    }

    /// Process a stereo sample pair.
    pub fn process(&mut self, left: f64, right: f64) {
        // Update correlation
        self.correlation_meter.process(left, right);

        // Calculate Mid/Side
        let mid = (left + right) / 2.0;
        let side = (left - right) / 2.0;

        // Update energies with smoothing
        self.mid_energy =
            self.smoothing_factor * self.mid_energy + (1.0 - self.smoothing_factor) * (mid * mid);
        self.side_energy = self.smoothing_factor * self.side_energy
            + (1.0 - self.smoothing_factor) * (side * side);
    }

    /// Process interleaved stereo samples.
    pub fn process_interleaved(&mut self, samples: &[f64]) {
        for chunk in samples.chunks_exact(2) {
            self.process(chunk[0], chunk[1]);
        }
    }

    /// Get the stereo width percentage (0-200%).
    ///
    /// - 0%: mono signal
    /// - 100%: normal stereo
    /// - 200%: maximum stereo width
    pub fn width_percentage(&self) -> f64 {
        let total_energy = self.mid_energy + self.side_energy;

        if total_energy > 0.0 {
            let side_ratio = self.side_energy / total_energy;
            side_ratio * 200.0
        } else {
            0.0
        }
    }

    /// Get the phase correlation.
    pub fn correlation(&self) -> f64 {
        self.correlation_meter.correlation()
    }

    /// Get the Mid energy level in dB.
    pub fn mid_level_db(&self) -> f64 {
        if self.mid_energy > 0.0 {
            10.0 * self.mid_energy.log10()
        } else {
            f64::NEG_INFINITY
        }
    }

    /// Get the Side energy level in dB.
    pub fn side_level_db(&self) -> f64 {
        if self.side_energy > 0.0 {
            10.0 * self.side_energy.log10()
        } else {
            f64::NEG_INFINITY
        }
    }

    /// Reset the analyzer.
    pub fn reset(&mut self) {
        self.correlation_meter.reset();
        self.mid_energy = 0.0;
        self.side_energy = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase_correlation_mono() {
        // Use a short integration window (0.01 s at 1000 Hz) so buffer_size = 10
        // and we only need ~10 samples to fill it — keeping O(N*buffer_size) tiny.
        let mut meter = PhaseCorrelationMeter::new(1000.0, 0.01).expect("test expectation failed");

        // Feed identical varying signals (mono) - use sine wave; 200 samples is plenty.
        for i in 0..200 {
            let t = i as f64 / 1000.0;
            let signal = (2.0 * std::f64::consts::PI * 100.0 * t).sin() * 0.5;
            meter.process(signal, signal);
        }

        // Should be perfectly correlated
        let corr = meter.correlation();
        assert!(
            corr > 0.95,
            "Correlation {:.3} should be > 0.95 for mono signal",
            corr
        );
        assert!(meter.is_mono());
    }

    #[test]
    fn test_phase_correlation_out_of_phase() {
        // Use a short integration window (0.01 s at 1000 Hz) so buffer_size = 10.
        let mut meter = PhaseCorrelationMeter::new(1000.0, 0.01).expect("test expectation failed");

        // Feed inverted varying signals (out of phase) - use sine wave; 200 samples is plenty.
        for i in 0..200 {
            let t = i as f64 / 1000.0;
            let signal = (2.0 * std::f64::consts::PI * 100.0 * t).sin() * 0.5;
            meter.process(signal, -signal);
        }

        // Should be negatively correlated
        let corr = meter.correlation();
        assert!(
            corr < -0.95,
            "Correlation {:.3} should be < -0.95 for out-of-phase signal",
            corr
        );
        assert!(meter.has_phase_issues());
    }

    #[test]
    fn test_phase_correlation_stereo() {
        let mut meter = PhaseCorrelationMeter::new(48000.0, 0.1).expect("test expectation failed");

        // Feed uncorrelated signals
        for i in 0..5000 {
            let t = i as f64 / 48000.0;
            let left = (2.0 * std::f64::consts::PI * 1000.0 * t).sin();
            let right = (2.0 * std::f64::consts::PI * 1500.0 * t).sin();
            meter.process(left, right);
        }

        // Should have low correlation
        assert!(meter.correlation().abs() < 0.5);
    }

    #[test]
    fn test_goniometer() {
        let mut goniometer = Goniometer::new(48000.0, 0.1);

        // Process some samples
        for i in 0..100 {
            let t = i as f64 / 100.0;
            goniometer.process(t, -t);
        }

        assert_eq!(goniometer.points().len(), 100);
    }

    #[test]
    fn test_stereo_width_mono() {
        let mut analyzer = StereoWidthAnalyzer::new(48000.0).expect("test expectation failed");

        // Feed mono signal
        for _ in 0..5000 {
            analyzer.process(0.5, 0.5);
        }

        // Width should be near 0% for mono
        assert!(analyzer.width_percentage() < 10.0);
    }

    #[test]
    fn test_stereo_width_stereo() {
        let mut analyzer = StereoWidthAnalyzer::new(48000.0).expect("test expectation failed");

        // Feed pure side signal (maximum stereo)
        for _ in 0..5000 {
            analyzer.process(0.5, -0.5);
        }

        // Width should be high for wide stereo
        assert!(analyzer.width_percentage() > 150.0);
    }

    #[test]
    fn test_phase_meter_reset() {
        let mut meter = PhaseCorrelationMeter::new(48000.0, 0.1).expect("test expectation failed");

        for _ in 0..1000 {
            meter.process(0.5, 0.5);
        }

        meter.reset();

        assert_eq!(meter.correlation(), 0.0);
    }
}
