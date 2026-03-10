//! Echo and reverb detection.

use crate::{AnalysisConfig, Result};

/// Echo detector for detecting echoes and reverb.
pub struct EchoDetector {
    #[allow(dead_code)]
    config: AnalysisConfig,
}

impl EchoDetector {
    /// Create a new echo detector.
    #[must_use]
    pub fn new(config: AnalysisConfig) -> Self {
        Self { config }
    }

    /// Detect echo in audio samples.
    pub fn detect(&self, samples: &[f32], sample_rate: f32) -> Result<EchoResult> {
        // Compute autocorrelation to find repeating patterns
        let max_delay = (sample_rate * 2.0) as usize; // Check up to 2 seconds
        let autocorr = self.compute_autocorrelation(samples, max_delay);

        // Find peaks in autocorrelation (indicating echoes)
        let echo_delays = self.find_echo_delays(&autocorr, sample_rate);

        // Estimate reverb amount from decay
        let reverb_amount = self.estimate_reverb(&autocorr);

        Ok(EchoResult {
            has_echo: !echo_delays.is_empty(),
            echo_delays,
            reverb_amount,
        })
    }

    /// Compute autocorrelation.
    #[allow(clippy::unused_self)]
    fn compute_autocorrelation(&self, samples: &[f32], max_lag: usize) -> Vec<f32> {
        let max_lag = max_lag.min(samples.len());
        let mut autocorr = vec![0.0; max_lag];

        for lag in 0..max_lag {
            let mut sum = 0.0;
            let mut norm = 0.0;

            for i in 0..(samples.len() - lag) {
                sum += samples[i] * samples[i + lag];
                norm += samples[i] * samples[i];
            }

            autocorr[lag] = if norm > 0.0 { sum / norm } else { 0.0 };
        }

        autocorr
    }

    /// Find echo delays from autocorrelation peaks.
    #[allow(clippy::unused_self)]
    fn find_echo_delays(&self, autocorr: &[f32], sample_rate: f32) -> Vec<f32> {
        let min_delay_samples = (sample_rate * 0.02) as usize; // Minimum 20ms
        let threshold = 0.3;

        let mut delays = Vec::new();

        for i in min_delay_samples..(autocorr.len() - 1) {
            if autocorr[i] > threshold
                && autocorr[i] > autocorr[i - 1]
                && autocorr[i] > autocorr[i + 1]
            {
                let delay_seconds = i as f32 / sample_rate;
                delays.push(delay_seconds);

                if delays.len() >= 5 {
                    break;
                }
            }
        }

        delays
    }

    /// Estimate reverb amount from autocorrelation decay.
    #[allow(clippy::unused_self)]
    fn estimate_reverb(&self, autocorr: &[f32]) -> f32 {
        if autocorr.len() < 100 {
            return 0.0;
        }

        // Measure how slowly autocorrelation decays
        let early = autocorr[10..50].iter().sum::<f32>() / 40.0;
        let late = autocorr[100..(autocorr.len().min(500))].iter().sum::<f32>()
            / (autocorr.len().min(500) - 100) as f32;

        // More reverb = slower decay = higher late/early ratio
        (late / (early + 1e-6)).min(1.0)
    }
}

/// Echo detection result.
#[derive(Debug, Clone)]
pub struct EchoResult {
    /// Whether echo is detected
    pub has_echo: bool,
    /// Echo delay times in seconds
    pub echo_delays: Vec<f32>,
    /// Reverb amount (0-1)
    pub reverb_amount: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_echo_detector() {
        let config = AnalysisConfig::default();
        let detector = EchoDetector::new(config);

        // Use a low sample rate and short buffer to keep autocorrelation fast
        // (compute_autocorrelation is O(max_lag * n), so large buffers are very slow)
        let sample_rate = 4000.0;
        let mut samples = vec![0.0; (sample_rate * 0.5) as usize]; // 0.5s buffer

        // Original signal
        for i in 0..200 {
            samples[i] = (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate).sin();
        }

        // Echo at 0.1 seconds
        let echo_delay = (sample_rate * 0.1) as usize;
        for i in 0..200 {
            if i + echo_delay < samples.len() {
                samples[i + echo_delay] += 0.5 * samples[i];
            }
        }

        let result = detector
            .detect(&samples, sample_rate)
            .expect("detection should succeed");
        assert!(result.has_echo || result.reverb_amount > 0.0);
    }
}
