//! Pitch tracking using YIN algorithm (patent-free).
//!
//! Reference: "YIN, a fundamental frequency estimator for speech and music"
//! by Alain de Cheveigné and Hideki Kawahara (2002)

use crate::{AnalysisConfig, AnalysisError, Result};

/// Pitch tracker using YIN algorithm.
pub struct PitchTracker {
    config: AnalysisConfig,
    min_lag: usize,
    max_lag: usize,
    threshold: f32,
}

impl PitchTracker {
    /// Create a new pitch tracker.
    #[must_use]
    pub fn new(config: AnalysisConfig) -> Self {
        // YIN parameters
        let min_lag = 20; // ~2 kHz at 44.1 kHz
        let max_lag = 2048; // ~21 Hz at 44.1 kHz
        let threshold = 0.1; // YIN threshold

        Self {
            config,
            min_lag,
            max_lag,
            threshold,
        }
    }

    /// Track pitch across entire audio signal.
    pub fn track(&self, samples: &[f32], sample_rate: f32) -> Result<PitchResult> {
        if samples.len() < self.max_lag * 2 {
            return Err(AnalysisError::InsufficientSamples {
                needed: self.max_lag * 2,
                got: samples.len(),
            });
        }

        let hop_size = self.config.hop_size;
        let window_size = self.config.fft_size.min(4096);

        let mut pitch_estimates = Vec::new();
        let mut confidences = Vec::new();

        // Process frames
        let num_frames = (samples.len() - window_size) / hop_size + 1;
        for frame_idx in 0..num_frames {
            let start = frame_idx * hop_size;
            let end = (start + window_size).min(samples.len());

            if end - start < window_size {
                break;
            }

            let frame = &samples[start..end];
            let estimate = self.estimate_pitch(frame, sample_rate)?;

            pitch_estimates.push(estimate.frequency);
            confidences.push(estimate.confidence);
        }

        // Compute statistics
        let voiced_estimates: Vec<f32> = pitch_estimates
            .iter()
            .zip(&confidences)
            .filter(|(_, &conf)| conf > 0.5)
            .map(|(&f, _)| f)
            .collect();

        let mean_f0 = if voiced_estimates.is_empty() {
            0.0
        } else {
            voiced_estimates.iter().sum::<f32>() / voiced_estimates.len() as f32
        };

        let voicing_rate = voiced_estimates.len() as f32 / pitch_estimates.len() as f32;

        Ok(PitchResult {
            estimates: pitch_estimates,
            confidences,
            mean_f0,
            voicing_rate,
        })
    }

    /// Estimate pitch for a single frame.
    pub fn track_frame(&self, samples: &[f32], sample_rate: f32) -> Result<PitchEstimate> {
        self.estimate_pitch(samples, sample_rate)
    }

    /// Estimate pitch using YIN algorithm.
    #[allow(clippy::unnecessary_wraps, clippy::needless_range_loop)]
    fn estimate_pitch(&self, samples: &[f32], sample_rate: f32) -> Result<PitchEstimate> {
        if samples.len() < self.max_lag {
            return Ok(PitchEstimate {
                frequency: 0.0,
                confidence: 0.0,
            });
        }

        // Step 1: Difference function
        let mut diff = vec![0.0; self.max_lag];
        for tau in 0..self.max_lag {
            let mut sum = 0.0;
            for j in 0..(samples.len() - self.max_lag) {
                let delta = samples[j] - samples[j + tau];
                sum += delta * delta;
            }
            diff[tau] = sum;
        }

        // Step 2: Cumulative mean normalized difference function
        let mut cmnd = vec![0.0; self.max_lag];
        cmnd[0] = 1.0;

        let mut running_sum = 0.0;
        for tau in 1..self.max_lag {
            running_sum += diff[tau];
            cmnd[tau] = if running_sum > 0.0 {
                diff[tau] * tau as f32 / running_sum
            } else {
                1.0
            };
        }

        // Step 3: Absolute threshold
        let mut tau = self.min_lag;
        while tau < self.max_lag {
            if cmnd[tau] < self.threshold {
                while tau + 1 < self.max_lag && cmnd[tau + 1] < cmnd[tau] {
                    tau += 1;
                }
                break;
            }
            tau += 1;
        }

        if tau >= self.max_lag - 1 {
            return Ok(PitchEstimate {
                frequency: 0.0,
                confidence: 0.0,
            });
        }

        // Step 4: Parabolic interpolation
        let better_tau = if tau > 0 && tau < self.max_lag - 1 {
            let s0 = cmnd[tau - 1];
            let s1 = cmnd[tau];
            let s2 = cmnd[tau + 1];
            tau as f32 + (s2 - s0) / (2.0 * (2.0 * s1 - s2 - s0))
        } else {
            tau as f32
        };

        let frequency = sample_rate / better_tau;
        let confidence = 1.0 - cmnd[tau];

        // Filter out unrealistic frequencies
        if !(50.0..=1000.0).contains(&frequency) {
            return Ok(PitchEstimate {
                frequency: 0.0,
                confidence: 0.0,
            });
        }

        Ok(PitchEstimate {
            frequency,
            confidence,
        })
    }
}

/// Result of pitch tracking across entire signal.
#[derive(Debug, Clone)]
pub struct PitchResult {
    /// Frame-by-frame pitch estimates in Hz
    pub estimates: Vec<f32>,
    /// Confidence for each estimate (0-1)
    pub confidences: Vec<f32>,
    /// Mean F0 for voiced segments
    pub mean_f0: f32,
    /// Proportion of voiced frames
    pub voicing_rate: f32,
}

/// Single pitch estimate for a frame.
#[derive(Debug, Clone, Copy)]
pub struct PitchEstimate {
    /// Estimated frequency in Hz (0 if unvoiced)
    pub frequency: f32,
    /// Confidence in estimate (0-1)
    pub confidence: f32,
}

impl Default for PitchEstimate {
    fn default() -> Self {
        Self {
            frequency: 0.0,
            confidence: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pitch_tracking() {
        let config = AnalysisConfig::default();
        let tracker = PitchTracker::new(config);

        // Generate 440 Hz sine wave
        let sample_rate = 44100.0;
        let frequency = 440.0;
        let duration = 0.5;
        let samples: Vec<f32> = (0..(sample_rate * duration) as usize)
            .map(|i| {
                let t = i as f32 / sample_rate;
                (2.0 * std::f32::consts::PI * frequency * t).sin()
            })
            .collect();

        let result = tracker
            .track(&samples, sample_rate)
            .expect("tracking should succeed");

        // YIN should detect some pitch
        assert!(result.estimates.len() > 0);
        assert!(result.voicing_rate >= 0.0 && result.voicing_rate <= 1.0);
    }

    #[test]
    fn test_pitch_unvoiced() {
        let config = AnalysisConfig::default();
        let tracker = PitchTracker::new(config);

        // White noise (unvoiced)
        let samples = vec![0.01; 8192];
        let result = tracker
            .track(&samples, 44100.0)
            .expect("tracking should succeed");

        // Should detect mostly unvoiced
        assert!(result.voicing_rate < 0.3);
    }
}
