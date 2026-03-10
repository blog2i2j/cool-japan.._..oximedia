//! Spectral analysis utilities for audio restoration.

use crate::error::{RestoreError, RestoreResult};
use oxifft::Complex;

/// Window function type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowFunction {
    /// Rectangular (no window).
    Rectangular,
    /// Hann window.
    Hann,
    /// Hamming window.
    Hamming,
    /// Blackman window.
    Blackman,
    /// Blackman-Harris window.
    BlackmanHarris,
}

/// Apply window function to samples.
pub fn apply_window(samples: &mut [f32], window: WindowFunction) {
    let n = samples.len();
    if n == 0 {
        return;
    }

    for (i, sample) in samples.iter_mut().enumerate() {
        #[allow(clippy::cast_precision_loss)]
        let factor = window_factor(i, n, window);
        *sample *= factor;
    }
}

/// Compute window factor for a given index.
#[allow(clippy::cast_precision_loss)]
fn window_factor(i: usize, n: usize, window: WindowFunction) -> f32 {
    use std::f32::consts::PI;

    let i_f32 = i as f32;
    let n_f32 = n as f32;

    match window {
        WindowFunction::Rectangular => 1.0,
        WindowFunction::Hann => 0.5 * (1.0 - ((2.0 * PI * i_f32) / (n_f32 - 1.0)).cos()),
        WindowFunction::Hamming => 0.54 - 0.46 * ((2.0 * PI * i_f32) / (n_f32 - 1.0)).cos(),
        WindowFunction::Blackman => {
            0.42 - 0.5 * ((2.0 * PI * i_f32) / (n_f32 - 1.0)).cos()
                + 0.08 * ((4.0 * PI * i_f32) / (n_f32 - 1.0)).cos()
        }
        WindowFunction::BlackmanHarris => {
            0.355_68 - 0.487_4 * ((2.0 * PI * i_f32) / (n_f32 - 1.0)).cos()
                + 0.144_32 * ((4.0 * PI * i_f32) / (n_f32 - 1.0)).cos()
                - 0.012_04 * ((6.0 * PI * i_f32) / (n_f32 - 1.0)).cos()
        }
    }
}

/// FFT processor for spectral analysis.
///
/// Wraps OxiFFT to provide a simple forward/inverse FFT interface over
/// `Complex<f32>` buffers.
pub struct FftProcessor {
    size: usize,
}

impl FftProcessor {
    /// Create a new FFT processor.
    ///
    /// # Arguments
    ///
    /// * `size` - FFT size (should be power of 2)
    #[must_use]
    pub fn new(size: usize) -> Self {
        Self { size }
    }

    /// Get FFT size.
    #[must_use]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Perform forward FFT.
    ///
    /// # Arguments
    ///
    /// * `samples` - Input samples (real)
    ///
    /// # Returns
    ///
    /// Complex frequency domain data.
    pub fn forward(&self, samples: &[f32]) -> RestoreResult<Vec<Complex<f32>>> {
        if samples.len() != self.size {
            return Err(RestoreError::InvalidParameter(format!(
                "Sample count {} does not match FFT size {}",
                samples.len(),
                self.size
            )));
        }

        let input: Vec<Complex<f32>> = samples.iter().map(|&s| Complex::new(s, 0.0)).collect();
        let output = oxifft::fft(&input);

        Ok(output)
    }

    /// Perform inverse FFT.
    ///
    /// # Arguments
    ///
    /// * `spectrum` - Complex frequency domain data
    ///
    /// # Returns
    ///
    /// Real time domain samples.
    pub fn inverse(&self, spectrum: &[Complex<f32>]) -> RestoreResult<Vec<f32>> {
        if spectrum.len() != self.size {
            return Err(RestoreError::InvalidParameter(format!(
                "Spectrum size {} does not match FFT size {}",
                spectrum.len(),
                self.size
            )));
        }

        let output = oxifft::ifft(spectrum);

        // Extract real part (OxiFFT's ifft already normalizes by 1/N)
        Ok(output.iter().map(|c| c.re).collect())
    }

    /// Compute magnitude spectrum.
    ///
    /// # Arguments
    ///
    /// * `spectrum` - Complex frequency domain data
    ///
    /// # Returns
    ///
    /// Magnitude values.
    #[must_use]
    pub fn magnitude(&self, spectrum: &[Complex<f32>]) -> Vec<f32> {
        spectrum.iter().map(|c| c.norm()).collect()
    }

    /// Compute power spectrum.
    ///
    /// # Arguments
    ///
    /// * `spectrum` - Complex frequency domain data
    ///
    /// # Returns
    ///
    /// Power values (magnitude squared).
    #[must_use]
    pub fn power(&self, spectrum: &[Complex<f32>]) -> Vec<f32> {
        spectrum.iter().map(|c| c.norm_sqr()).collect()
    }

    /// Compute phase spectrum.
    ///
    /// # Arguments
    ///
    /// * `spectrum` - Complex frequency domain data
    ///
    /// # Returns
    ///
    /// Phase values in radians.
    #[must_use]
    pub fn phase(&self, spectrum: &[Complex<f32>]) -> Vec<f32> {
        spectrum.iter().map(|c| c.arg()).collect()
    }

    /// Convert magnitude and phase back to complex spectrum.
    ///
    /// # Arguments
    ///
    /// * `magnitude` - Magnitude values
    /// * `phase` - Phase values in radians
    ///
    /// # Returns
    ///
    /// Complex spectrum.
    pub fn from_polar(magnitude: &[f32], phase: &[f32]) -> RestoreResult<Vec<Complex<f32>>> {
        if magnitude.len() != phase.len() {
            return Err(RestoreError::InvalidParameter(
                "Magnitude and phase arrays must have same length".to_string(),
            ));
        }

        Ok(magnitude
            .iter()
            .zip(phase.iter())
            .map(|(&mag, &ph)| Complex::from_polar(mag, ph))
            .collect())
    }
}

/// Compute spectral centroid.
///
/// # Arguments
///
/// * `magnitude` - Magnitude spectrum
/// * `sample_rate` - Sample rate in Hz
///
/// # Returns
///
/// Spectral centroid in Hz.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn spectral_centroid(magnitude: &[f32], sample_rate: u32) -> f32 {
    if magnitude.is_empty() {
        return 0.0;
    }

    let mut weighted_sum = 0.0;
    let mut sum = 0.0;

    let bin_width = sample_rate as f32 / (2.0 * magnitude.len() as f32);

    for (i, &mag) in magnitude.iter().enumerate() {
        let freq = i as f32 * bin_width;
        weighted_sum += freq * mag;
        sum += mag;
    }

    if sum > f32::EPSILON {
        weighted_sum / sum
    } else {
        0.0
    }
}

/// Compute spectral flatness (Wiener entropy).
///
/// # Arguments
///
/// * `magnitude` - Magnitude spectrum
///
/// # Returns
///
/// Spectral flatness (0.0 = tonal, 1.0 = noisy).
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn spectral_flatness(magnitude: &[f32]) -> f32 {
    if magnitude.is_empty() {
        return 0.0;
    }

    let mut geometric_mean = 1.0;
    let mut arithmetic_mean = 0.0;
    let n = magnitude.len();

    for &mag in magnitude {
        let mag = mag.max(1e-10); // Avoid log(0)
        geometric_mean *= mag.powf(1.0 / n as f32);
        arithmetic_mean += mag;
    }

    arithmetic_mean /= n as f32;

    if arithmetic_mean > f32::EPSILON {
        geometric_mean / arithmetic_mean
    } else {
        0.0
    }
}

/// Compute spectral rolloff frequency.
///
/// # Arguments
///
/// * `magnitude` - Magnitude spectrum
/// * `sample_rate` - Sample rate in Hz
/// * `threshold` - Rolloff threshold (e.g., 0.85 for 85% of energy)
///
/// # Returns
///
/// Rolloff frequency in Hz.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn spectral_rolloff(magnitude: &[f32], sample_rate: u32, threshold: f32) -> f32 {
    if magnitude.is_empty() {
        return 0.0;
    }

    let total_energy: f32 = magnitude.iter().map(|&m| m * m).sum();
    let target_energy = total_energy * threshold;

    let mut accumulated = 0.0;
    let bin_width = sample_rate as f32 / (2.0 * magnitude.len() as f32);

    for (i, &mag) in magnitude.iter().enumerate() {
        accumulated += mag * mag;
        if accumulated >= target_energy {
            return i as f32 * bin_width;
        }
    }

    (magnitude.len() - 1) as f32 * bin_width
}

/// Find peaks in spectrum.
///
/// # Arguments
///
/// * `magnitude` - Magnitude spectrum
/// * `threshold` - Minimum peak magnitude
/// * `min_distance` - Minimum distance between peaks in bins
///
/// # Returns
///
/// Indices of peaks.
#[must_use]
pub fn find_peaks(magnitude: &[f32], threshold: f32, min_distance: usize) -> Vec<usize> {
    let mut peaks = Vec::new();

    if magnitude.len() < 3 {
        return peaks;
    }

    for i in 1..magnitude.len() - 1 {
        if magnitude[i] > threshold
            && magnitude[i] > magnitude[i - 1]
            && magnitude[i] > magnitude[i + 1]
        {
            // Check minimum distance from previous peaks
            if peaks.is_empty() || i - peaks[peaks.len() - 1] >= min_distance {
                peaks.push(i);
            } else if magnitude[i] > magnitude[peaks[peaks.len() - 1]] {
                // Replace previous peak if this one is higher
                peaks.pop();
                peaks.push(i);
            }
        }
    }

    peaks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_window_functions() {
        let mut samples = vec![1.0; 64];
        apply_window(&mut samples, WindowFunction::Hann);
        assert!(samples[0] < 0.1);
        assert!(samples[32] > 0.9);
        assert!(samples[63] < 0.1);
    }

    #[test]
    fn test_fft_processor() {
        let processor = FftProcessor::new(64);
        assert_eq!(processor.size(), 64);

        let samples = vec![1.0; 64];
        let spectrum = processor.forward(&samples).expect("should succeed in test");
        assert_eq!(spectrum.len(), 64);

        let reconstructed = processor
            .inverse(&spectrum)
            .expect("should succeed in test");
        assert_eq!(reconstructed.len(), 64);

        for (original, recon) in samples.iter().zip(reconstructed.iter()) {
            assert!((original - recon).abs() < 1e-4);
        }
    }

    #[test]
    fn test_magnitude_phase() {
        let processor = FftProcessor::new(64);
        let samples = vec![1.0; 64];
        let spectrum = processor.forward(&samples).expect("should succeed in test");

        let magnitude = processor.magnitude(&spectrum);
        let phase = processor.phase(&spectrum);

        assert_eq!(magnitude.len(), 64);
        assert_eq!(phase.len(), 64);

        let reconstructed =
            FftProcessor::from_polar(&magnitude, &phase).expect("should succeed in test");
        assert_eq!(reconstructed.len(), 64);
    }

    #[test]
    fn test_spectral_features() {
        let magnitude = vec![0.1, 0.5, 1.0, 0.5, 0.1];

        let centroid = spectral_centroid(&magnitude, 44100);
        assert!(centroid > 0.0);

        let flatness = spectral_flatness(&magnitude);
        assert!(flatness >= 0.0 && flatness <= 1.0);

        let rolloff = spectral_rolloff(&magnitude, 44100, 0.85);
        assert!(rolloff > 0.0);
    }

    #[test]
    fn test_find_peaks() {
        let magnitude = vec![0.1, 0.5, 0.2, 0.8, 0.3, 1.0, 0.4];
        let peaks = find_peaks(&magnitude, 0.4, 1);
        assert!(peaks.contains(&1));
        assert!(peaks.contains(&3));
        assert!(peaks.contains(&5));
    }
}
