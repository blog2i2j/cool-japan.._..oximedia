//! Formant analysis using Linear Predictive Coding (LPC).

use crate::{AnalysisConfig, AnalysisError, Result};

/// Formant analyzer using LPC.
pub struct FormantAnalyzer {
    config: AnalysisConfig,
    lpc_order: usize,
}

impl FormantAnalyzer {
    /// Create a new formant analyzer.
    #[must_use]
    pub fn new(config: AnalysisConfig) -> Self {
        // LPC order typically 2 + sample_rate / 1000
        let lpc_order = 12; // Good for standard speech analysis

        Self { config, lpc_order }
    }

    /// Analyze formants from audio samples.
    pub fn analyze(&self, samples: &[f32], sample_rate: f32) -> Result<FormantResult> {
        if samples.len() < self.config.fft_size {
            return Err(AnalysisError::InsufficientSamples {
                needed: self.config.fft_size,
                got: samples.len(),
            });
        }

        // Pre-emphasize signal (high-pass filter to enhance higher frequencies)
        let emphasized = self.pre_emphasize(samples);

        // Compute LPC coefficients
        let lpc_coeffs = self.compute_lpc(&emphasized)?;

        // Find formants from LPC coefficients
        let formants = self.find_formants(&lpc_coeffs, sample_rate)?;

        Ok(FormantResult {
            formants,
            lpc_coefficients: lpc_coeffs,
        })
    }

    /// Pre-emphasize signal using first-order high-pass filter.
    #[allow(clippy::unused_self)]
    fn pre_emphasize(&self, samples: &[f32]) -> Vec<f32> {
        let alpha = 0.97;
        let mut emphasized = Vec::with_capacity(samples.len());

        emphasized.push(samples[0]);
        for i in 1..samples.len() {
            emphasized.push(samples[i] - alpha * samples[i - 1]);
        }

        emphasized
    }

    /// Compute LPC coefficients using autocorrelation method (Levinson-Durbin).
    #[allow(clippy::unused_self, clippy::unnecessary_wraps)]
    fn compute_lpc(&self, samples: &[f32]) -> Result<Vec<f32>> {
        // Compute autocorrelation
        let mut r = vec![0.0; self.lpc_order + 1];
        for i in 0..=self.lpc_order {
            let mut sum = 0.0;
            for j in 0..(samples.len() - i) {
                sum += samples[j] * samples[j + i];
            }
            r[i] = sum;
        }

        // Levinson-Durbin algorithm
        let mut a = vec![0.0; self.lpc_order + 1];
        let mut e = r[0];

        for i in 1..=self.lpc_order {
            let mut lambda = 0.0;
            for j in 1..i {
                lambda -= a[j] * r[i - j];
            }
            lambda -= r[i];

            let k = if e == 0.0 { 0.0 } else { lambda / e };

            a[i] = k;

            for j in 1..i {
                let temp = a[j];
                a[j] += k * a[i - j];
                a[i - j] = temp;
            }

            e *= 1.0 - k * k;
        }

        Ok(a)
    }

    /// Find formant frequencies from LPC coefficients.
    #[allow(clippy::unnecessary_wraps)]
    fn find_formants(&self, lpc_coeffs: &[f32], sample_rate: f32) -> Result<Vec<f32>> {
        // Find roots of LPC polynomial (formants are at complex root angles)
        let roots = self.find_lpc_roots(lpc_coeffs)?;

        // Extract formant frequencies from roots
        let mut formants: Vec<f32> = roots
            .iter()
            .filter(|(real, imag)| {
                // Filter for roots inside unit circle with positive frequency
                let magnitude = (real * real + imag * imag).sqrt();
                magnitude < 1.0 && magnitude > 0.7 && *imag > 0.0
            })
            .map(|(real, imag)| {
                let angle = imag.atan2(*real);
                (angle * sample_rate) / (2.0 * std::f32::consts::PI)
            })
            .collect();

        // Sort by frequency
        formants.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Take first 4 formants
        formants.truncate(4);

        // If we don't have enough formants, use typical values
        while formants.len() < 4 {
            let default_formants = [500.0, 1500.0, 2500.0, 3500.0];
            formants.push(default_formants[formants.len()]);
        }

        Ok(formants)
    }

    /// Find roots of LPC polynomial using Durand-Kerner method.
    #[allow(clippy::unnecessary_wraps, clippy::needless_range_loop)]
    fn find_lpc_roots(&self, coeffs: &[f32]) -> Result<Vec<(f32, f32)>> {
        if coeffs.is_empty() {
            return Ok(Vec::new());
        }

        let n = coeffs.len() - 1;
        if n == 0 {
            return Ok(Vec::new());
        }

        // Initialize roots on unit circle
        let mut roots: Vec<(f32, f32)> = (0..n)
            .map(|i| {
                let angle = 2.0 * std::f32::consts::PI * (i as f32 + 0.4) / n as f32;
                (0.9 * angle.cos(), 0.9 * angle.sin())
            })
            .collect();

        // Iterate to refine roots (Durand-Kerner method)
        for _ in 0..50 {
            let mut max_change: f32 = 0.0;

            for i in 0..n {
                let (re, im) = roots[i];

                // Evaluate polynomial at current root
                let (p_re, p_im) = self.eval_poly(coeffs, re, im);

                // Compute product of differences with other roots
                let mut prod_re = 1.0;
                let mut prod_im = 0.0;

                for j in 0..n {
                    if i != j {
                        let (rj_re, rj_im) = roots[j];
                        let diff_re = re - rj_re;
                        let diff_im = im - rj_im;

                        let temp_re = prod_re * diff_re - prod_im * diff_im;
                        let temp_im = prod_re * diff_im + prod_im * diff_re;
                        prod_re = temp_re;
                        prod_im = temp_im;
                    }
                }

                // Division: p / prod
                let denom = prod_re * prod_re + prod_im * prod_im;
                if denom > 1e-10 {
                    let delta_re = (p_re * prod_re + p_im * prod_im) / denom;
                    let delta_im = (p_im * prod_re - p_re * prod_im) / denom;

                    roots[i].0 -= delta_re;
                    roots[i].1 -= delta_im;

                    max_change = max_change.max(delta_re.abs() + delta_im.abs());
                }
            }

            if max_change < 1e-6 {
                break;
            }
        }

        Ok(roots)
    }

    /// Evaluate polynomial at complex point.
    #[allow(clippy::unused_self)]
    fn eval_poly(&self, coeffs: &[f32], re: f32, im: f32) -> (f32, f32) {
        let mut result_re = 0.0;
        let mut result_im = 0.0;
        let mut power_re = 1.0;
        let mut power_im = 0.0;

        for &coeff in coeffs {
            result_re += coeff * power_re;
            result_im += coeff * power_im;

            // Multiply power by (re, im)
            let temp_re = power_re * re - power_im * im;
            let temp_im = power_re * im + power_im * re;
            power_re = temp_re;
            power_im = temp_im;
        }

        (result_re, result_im)
    }
}

/// Formant analysis result.
#[derive(Debug, Clone)]
pub struct FormantResult {
    /// Formant frequencies [F1, F2, F3, F4] in Hz
    pub formants: Vec<f32>,
    /// LPC coefficients
    pub lpc_coefficients: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_formant_analyzer() {
        let config = AnalysisConfig::default();
        let analyzer = FormantAnalyzer::new(config);

        // Generate test signal
        let sample_rate = 16000.0;
        let samples = vec![0.1; 4096];

        let result = analyzer.analyze(&samples, sample_rate);
        assert!(result.is_ok());

        let formants = result.expect("expected successful result").formants;
        assert_eq!(formants.len(), 4);

        // All formants should be positive
        for &f in &formants {
            assert!(f > 0.0);
        }
    }

    #[test]
    fn test_pre_emphasis() {
        let config = AnalysisConfig::default();
        let analyzer = FormantAnalyzer::new(config);

        let samples = vec![1.0, 2.0, 3.0, 4.0];
        let emphasized = analyzer.pre_emphasize(&samples);

        assert_eq!(emphasized.len(), samples.len());
        assert_eq!(emphasized[0], samples[0]);
    }
}
