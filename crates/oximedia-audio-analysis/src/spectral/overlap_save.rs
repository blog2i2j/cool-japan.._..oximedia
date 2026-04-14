//! Overlap-save (overlap-discard) method for efficient long-duration convolution
//! and spectral analysis.
//!
//! # Algorithm
//!
//! The overlap-save method processes a long signal by splitting it into
//! overlapping blocks of length `N` (the FFT size).  For each block:
//!
//! 1. Fill a buffer of length `N` with `(N - step)` samples from the previous
//!    block (the "overlap") followed by `step` new samples.
//! 2. Apply a window function and compute the FFT.
//! 3. Retain only the `step` **valid** output samples (discarding the `N - step`
//!    samples that are contaminated by circular-convolution wrap-around).
//!
//! This is more cache-efficient than the overlap-add method because output
//! blocks are produced without an accumulation pass.
//!
//! # Reference
//!
//! Oppenheim & Schafer, *Discrete-Time Signal Processing*, 3rd ed., §8.7.

use crate::{generate_window, AnalysisConfig, AnalysisError, Result};
use oxifft::Complex;

/// Output of a single overlap-save analysis block.
#[derive(Debug, Clone)]
pub struct OlaBlock {
    /// Index of the first valid sample in the original signal.
    pub sample_offset: usize,
    /// Magnitude spectrum for this block (length = fft_size / 2 + 1).
    pub magnitude: Vec<f32>,
    /// Power spectrum (magnitude squared) for this block.
    pub power: Vec<f32>,
}

/// Overlap-save spectral analyser for efficient processing of long signals.
///
/// Pre-allocates all intermediate buffers at construction time.
pub struct OverlapSaveAnalyzer {
    /// FFT size `N`.
    fft_size: usize,
    /// Step size (hop) `S = N - overlap`.  Must be ≤ fft_size.
    step: usize,
    /// Window coefficients of length `N`.
    window: Vec<f32>,
    /// Overlap buffer containing the last `(fft_size - step)` input samples.
    overlap_buf: Vec<f32>,
}

impl OverlapSaveAnalyzer {
    /// Create a new overlap-save analyser.
    ///
    /// # Parameters
    /// * `config` – standard `AnalysisConfig` (uses `fft_size` and `window_type`).
    /// * `step`   – number of *new* samples consumed per block.  Must satisfy
    ///   `1 ≤ step ≤ fft_size`.  A value of `fft_size / 2` gives 50 % overlap.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError::InvalidConfig`] if `step == 0` or `step > fft_size`.
    pub fn new(config: &AnalysisConfig, step: usize) -> Result<Self> {
        if step == 0 || step > config.fft_size {
            return Err(AnalysisError::InvalidConfig(format!(
                "step ({step}) must be in 1..={fft}",
                fft = config.fft_size
            )));
        }
        let window = generate_window(config.window_type, config.fft_size);
        let overlap_len = config.fft_size - step;
        Ok(Self {
            fft_size: config.fft_size,
            step,
            window,
            overlap_buf: vec![0.0_f32; overlap_len],
        })
    }

    /// Create with the default Hann window and 50 % overlap.
    ///
    /// # Errors
    ///
    /// Propagates any `AnalysisConfig` validation errors.
    pub fn default_50pct(config: &AnalysisConfig) -> Result<Self> {
        Self::new(config, config.fft_size / 2)
    }

    /// Process `samples` and return one [`OlaBlock`] per valid output block.
    ///
    /// The analyser is stateful: the internal overlap buffer is updated on
    /// each call so that consecutive calls with successive audio segments
    /// produce a seamless result.
    pub fn process(&mut self, samples: &[f32]) -> Result<Vec<OlaBlock>> {
        let overlap_len = self.fft_size - self.step;
        let mut blocks = Vec::with_capacity(samples.len() / self.step + 1);
        let mut pos = 0usize;

        while pos + self.step <= samples.len() {
            // Build the analysis frame: `overlap_len` old samples + `step` new.
            let mut frame = Vec::with_capacity(self.fft_size);
            frame.extend_from_slice(&self.overlap_buf);
            frame.extend_from_slice(&samples[pos..pos + self.step]);

            debug_assert_eq!(frame.len(), self.fft_size);

            // Apply window.
            let windowed: Vec<Complex<f64>> = frame
                .iter()
                .zip(self.window.iter())
                .map(|(&s, &w)| Complex::new(f64::from(s * w), 0.0))
                .collect();

            // FFT.
            let fft_out = oxifft::fft(&windowed);
            let half = self.fft_size / 2 + 1;

            let magnitude: Vec<f32> = fft_out[..half]
                .iter()
                .map(|c| c.norm() as f32)
                .collect();

            let power: Vec<f32> = magnitude.iter().map(|&m| m * m).collect();

            blocks.push(OlaBlock {
                sample_offset: pos,
                magnitude,
                power,
            });

            // Slide the overlap buffer forward.
            // New overlap = last `overlap_len` samples of the current frame.
            let src_start = self.fft_size - overlap_len;
            self.overlap_buf.copy_from_slice(&frame[src_start..]);

            pos += self.step;
        }

        Ok(blocks)
    }

    /// Reset the internal overlap buffer (call at stream boundaries).
    pub fn reset(&mut self) {
        for s in &mut self.overlap_buf {
            *s = 0.0;
        }
    }
}

/// Convenience function: run overlap-save analysis on a complete signal and
/// return per-block average power.
///
/// Returns a `Vec<f32>` of length equal to the number of blocks processed.
///
/// # Errors
///
/// Returns an error if the configuration is invalid.
pub fn overlap_save_power_envelope(
    samples: &[f32],
    config: &AnalysisConfig,
) -> Result<Vec<f32>> {
    let mut ola = OverlapSaveAnalyzer::default_50pct(config)?;
    let blocks = ola.process(samples)?;
    Ok(blocks
        .iter()
        .map(|b| b.power.iter().copied().sum::<f32>() / b.power.len() as f32)
        .collect())
}

/// Convenience function: run overlap-save analysis and return the
/// per-block magnitude spectra.
///
/// # Errors
///
/// Returns an error if the configuration is invalid.
pub fn overlap_save_spectra(
    samples: &[f32],
    config: &AnalysisConfig,
) -> Result<Vec<Vec<f32>>> {
    let mut ola = OverlapSaveAnalyzer::default_50pct(config)?;
    let blocks = ola.process(samples)?;
    Ok(blocks.into_iter().map(|b| b.magnitude).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AnalysisConfig;

    fn make_config(fft_size: usize) -> AnalysisConfig {
        AnalysisConfig {
            fft_size,
            ..Default::default()
        }
    }

    #[test]
    fn test_overlap_save_step_zero_is_error() {
        let cfg = make_config(512);
        assert!(OverlapSaveAnalyzer::new(&cfg, 0).is_err());
    }

    #[test]
    fn test_overlap_save_step_too_large_is_error() {
        let cfg = make_config(512);
        assert!(OverlapSaveAnalyzer::new(&cfg, 513).is_err());
    }

    #[test]
    fn test_overlap_save_produces_blocks_for_long_signal() {
        let cfg = make_config(256);
        let step = 128;
        let mut ola =
            OverlapSaveAnalyzer::new(&cfg, step).expect("valid config");

        // 1024 samples → 8 complete steps → 8 blocks.
        let signal: Vec<f32> = (0..1024)
            .map(|i| (i as f32 * 0.01).sin())
            .collect();

        let blocks = ola.process(&signal).expect("process should succeed");
        assert_eq!(blocks.len(), 8, "expected 8 blocks for 1024 samples / step 128");

        // Each block must have the correct spectrum length.
        for block in &blocks {
            assert_eq!(block.magnitude.len(), cfg.fft_size / 2 + 1);
            assert_eq!(block.power.len(), cfg.fft_size / 2 + 1);
        }
    }

    #[test]
    fn test_overlap_save_power_envelope_non_empty() {
        let cfg = make_config(256);
        let signal: Vec<f32> = (0..2048).map(|_| 0.5_f32).collect();
        let env = overlap_save_power_envelope(&signal, &cfg).expect("power envelope");
        assert!(!env.is_empty(), "power envelope must not be empty");
        for &p in &env {
            assert!(p >= 0.0, "power must be non-negative");
        }
    }

    #[test]
    fn test_overlap_save_reset_clears_state() {
        let cfg = make_config(256);
        let mut ola = OverlapSaveAnalyzer::default_50pct(&cfg).expect("valid");
        let signal: Vec<f32> = (0..512).map(|i| i as f32).collect();
        let _ = ola.process(&signal).expect("first pass");

        ola.reset();
        // After reset all overlap samples must be zero.
        assert!(
            ola.overlap_buf.iter().all(|&s| s == 0.0),
            "overlap buffer should be zeroed after reset"
        );
    }

    #[test]
    fn test_overlap_save_spectra_has_correct_shape() {
        let cfg = make_config(256);
        let signal: Vec<f32> = (0..1024).map(|i| (i as f32).sin()).collect();
        let spectra = overlap_save_spectra(&signal, &cfg).expect("spectra");
        assert!(!spectra.is_empty());
        for spec in &spectra {
            assert_eq!(spec.len(), cfg.fft_size / 2 + 1);
        }
    }
}
