//! Spectrum analyzers and frequency analysis.
//!
//! Implements:
//! - FFT-based spectrum analyzer
//! - 1/3 octave and 1/6 octave band analyzers
//! - Frequency weighting (A, B, C, K)
//! - Peak hold and decay

use crate::{MeteringError, MeteringResult};
use rustfft::{num_complex::Complex, FftPlanner};
use std::f64::consts::PI;

/// Frequency weighting curve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WeightingCurve {
    /// No weighting (flat).
    Flat,
    /// A-weighting (simulates human hearing at low levels).
    A,
    /// B-weighting (simulates human hearing at moderate levels).
    B,
    /// C-weighting (simulates human hearing at high levels).
    C,
    /// K-weighting (ITU-R BS.1770-4 for loudness measurement).
    K,
}

impl WeightingCurve {
    /// Calculate the weighting gain at a specific frequency.
    ///
    /// # Arguments
    ///
    /// * `frequency` - Frequency in Hz
    ///
    /// # Returns
    ///
    /// Gain factor (linear, not dB)
    pub fn gain_at_frequency(&self, frequency: f64) -> f64 {
        match self {
            Self::Flat => 1.0,
            Self::A => a_weighting_gain(frequency),
            Self::B => b_weighting_gain(frequency),
            Self::C => c_weighting_gain(frequency),
            Self::K => k_weighting_gain(frequency),
        }
    }
}

/// Window function for FFT.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WindowFunction {
    /// Rectangular window (no windowing).
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

impl WindowFunction {
    /// Calculate the window coefficient for a sample index.
    ///
    /// # Arguments
    ///
    /// * `n` - Sample index (0 to size-1)
    /// * `size` - Window size
    ///
    /// # Returns
    ///
    /// Window coefficient
    pub fn coefficient(&self, n: usize, size: usize) -> f64 {
        let n_f = n as f64;
        let size_f = size as f64;

        match self {
            Self::Rectangular => 1.0,
            Self::Hann => 0.5 * (1.0 - (2.0 * PI * n_f / (size_f - 1.0)).cos()),
            Self::Hamming => 0.54 - 0.46 * (2.0 * PI * n_f / (size_f - 1.0)).cos(),
            Self::Blackman => {
                0.42 - 0.5 * (2.0 * PI * n_f / (size_f - 1.0)).cos()
                    + 0.08 * (4.0 * PI * n_f / (size_f - 1.0)).cos()
            }
            Self::BlackmanHarris => {
                0.35875 - 0.48829 * (2.0 * PI * n_f / (size_f - 1.0)).cos()
                    + 0.14128 * (4.0 * PI * n_f / (size_f - 1.0)).cos()
                    - 0.01168 * (6.0 * PI * n_f / (size_f - 1.0)).cos()
            }
        }
    }
}

/// FFT-based spectrum analyzer.
pub struct SpectrumAnalyzer {
    sample_rate: f64,
    fft_size: usize,
    window_function: WindowFunction,
    weighting: WeightingCurve,
    fft_planner: FftPlanner<f64>,
    input_buffer: Vec<f64>,
    windowed_buffer: Vec<Complex<f64>>,
    spectrum: Vec<f64>,
    peak_hold: Vec<f64>,
    peak_hold_time: f64,
    peak_hold_counters: Vec<usize>,
    peak_hold_samples: usize,
}

impl SpectrumAnalyzer {
    /// Create a new spectrum analyzer.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate in Hz
    /// * `fft_size` - FFT size (must be power of 2)
    /// * `window_function` - Window function to apply
    /// * `weighting` - Frequency weighting curve
    /// * `peak_hold_time` - Peak hold time in seconds
    ///
    /// # Errors
    ///
    /// Returns error if configuration is invalid.
    pub fn new(
        sample_rate: f64,
        fft_size: usize,
        window_function: WindowFunction,
        weighting: WeightingCurve,
        peak_hold_time: f64,
    ) -> MeteringResult<Self> {
        if !fft_size.is_power_of_two() {
            return Err(MeteringError::InvalidConfig(
                "FFT size must be a power of 2".to_string(),
            ));
        }

        if sample_rate <= 0.0 {
            return Err(MeteringError::InvalidConfig(
                "Sample rate must be positive".to_string(),
            ));
        }

        let num_bins = fft_size / 2 + 1;
        let peak_hold_samples = (peak_hold_time * sample_rate) as usize / fft_size;

        Ok(Self {
            sample_rate,
            fft_size,
            window_function,
            weighting,
            fft_planner: FftPlanner::new(),
            input_buffer: vec![0.0; fft_size],
            windowed_buffer: vec![Complex::new(0.0, 0.0); fft_size],
            spectrum: vec![0.0; num_bins],
            peak_hold: vec![0.0; num_bins],
            peak_hold_time,
            peak_hold_counters: vec![0; num_bins],
            peak_hold_samples,
        })
    }

    /// Process audio samples and update spectrum.
    ///
    /// # Arguments
    ///
    /// * `samples` - Audio samples (mono)
    pub fn process(&mut self, samples: &[f64]) {
        if samples.len() < self.fft_size {
            return;
        }

        // Copy samples to input buffer
        self.input_buffer.copy_from_slice(&samples[..self.fft_size]);

        // Apply window function
        for (i, &sample) in self.input_buffer.iter().enumerate() {
            let window_coeff = self.window_function.coefficient(i, self.fft_size);
            self.windowed_buffer[i] = Complex::new(sample * window_coeff, 0.0);
        }

        // Perform FFT
        let fft = self.fft_planner.plan_fft_forward(self.fft_size);
        fft.process(&mut self.windowed_buffer);

        // Calculate magnitude spectrum
        for (i, &complex_value) in self
            .windowed_buffer
            .iter()
            .enumerate()
            .take(self.spectrum.len())
        {
            let magnitude = complex_value.norm();
            let frequency = i as f64 * self.sample_rate / self.fft_size as f64;

            // Apply frequency weighting
            let weighted_magnitude = magnitude * self.weighting.gain_at_frequency(frequency);

            self.spectrum[i] = weighted_magnitude;

            // Update peak hold
            if weighted_magnitude > self.peak_hold[i] {
                self.peak_hold[i] = weighted_magnitude;
                self.peak_hold_counters[i] = self.peak_hold_samples;
            } else if self.peak_hold_counters[i] > 0 {
                self.peak_hold_counters[i] -= 1;
            } else {
                // Decay peak hold
                self.peak_hold[i] = weighted_magnitude;
            }
        }
    }

    /// Get the current spectrum magnitude (linear scale).
    pub fn spectrum(&self) -> &[f64] {
        &self.spectrum
    }

    /// Get the spectrum in dB.
    pub fn spectrum_db(&self) -> Vec<f64> {
        self.spectrum
            .iter()
            .map(|&mag| {
                if mag > 0.0 {
                    20.0 * mag.log10()
                } else {
                    f64::NEG_INFINITY
                }
            })
            .collect()
    }

    /// Get the peak hold spectrum (linear scale).
    pub fn peak_hold_spectrum(&self) -> &[f64] {
        &self.peak_hold
    }

    /// Get the peak hold spectrum in dB.
    pub fn peak_hold_spectrum_db(&self) -> Vec<f64> {
        self.peak_hold
            .iter()
            .map(|&mag| {
                if mag > 0.0 {
                    20.0 * mag.log10()
                } else {
                    f64::NEG_INFINITY
                }
            })
            .collect()
    }

    /// Get the frequency for a bin index.
    pub fn bin_frequency(&self, bin: usize) -> f64 {
        bin as f64 * self.sample_rate / self.fft_size as f64
    }

    /// Get the number of frequency bins.
    pub fn num_bins(&self) -> usize {
        self.spectrum.len()
    }

    /// Reset the analyzer.
    pub fn reset(&mut self) {
        self.spectrum.fill(0.0);
        self.peak_hold.fill(0.0);
        self.peak_hold_counters.fill(0);
    }
}

/// Octave band analyzer for 1/3 octave or 1/6 octave analysis.
pub struct OctaveBandAnalyzer {
    sample_rate: f64,
    bands: Vec<OctaveBand>,
    band_levels: Vec<f64>,
    peak_hold: Vec<f64>,
    peak_hold_time: f64,
    peak_hold_counters: Vec<usize>,
}

/// Octave band definition.
#[derive(Clone, Debug)]
pub struct OctaveBand {
    /// Center frequency in Hz.
    pub center_freq: f64,
    /// Lower frequency in Hz.
    pub lower_freq: f64,
    /// Upper frequency in Hz.
    pub upper_freq: f64,
}

impl OctaveBandAnalyzer {
    /// Create a new 1/3 octave band analyzer (20 Hz - 20 kHz).
    pub fn new_third_octave(sample_rate: f64, peak_hold_time: f64) -> Self {
        let bands = generate_third_octave_bands();
        Self::new(sample_rate, bands, peak_hold_time)
    }

    /// Create a new 1/6 octave band analyzer (20 Hz - 20 kHz).
    pub fn new_sixth_octave(sample_rate: f64, peak_hold_time: f64) -> Self {
        let bands = generate_sixth_octave_bands();
        Self::new(sample_rate, bands, peak_hold_time)
    }

    /// Create a new octave band analyzer with custom bands.
    fn new(sample_rate: f64, bands: Vec<OctaveBand>, peak_hold_time: f64) -> Self {
        let num_bands = bands.len();

        Self {
            sample_rate,
            bands,
            band_levels: vec![0.0; num_bands],
            peak_hold: vec![0.0; num_bands],
            peak_hold_time,
            peak_hold_counters: vec![0; num_bands],
        }
    }

    /// Process FFT spectrum data.
    ///
    /// # Arguments
    ///
    /// * `spectrum` - FFT magnitude spectrum
    /// * `fft_size` - Size of the FFT
    pub fn process_spectrum(&mut self, spectrum: &[f64], fft_size: usize) {
        for (band_idx, band) in self.bands.iter().enumerate() {
            let mut band_energy = 0.0;
            let mut bin_count = 0;

            // Sum energy in this band
            for (bin_idx, &magnitude) in spectrum.iter().enumerate() {
                let freq = bin_idx as f64 * self.sample_rate / fft_size as f64;

                if freq >= band.lower_freq && freq <= band.upper_freq {
                    band_energy += magnitude * magnitude;
                    bin_count += 1;
                }
            }

            // Average and take RMS
            let band_level = if bin_count > 0 {
                (band_energy / f64::from(bin_count)).sqrt()
            } else {
                0.0
            };

            self.band_levels[band_idx] = band_level;

            // Update peak hold
            if band_level > self.peak_hold[band_idx] {
                self.peak_hold[band_idx] = band_level;
                self.peak_hold_counters[band_idx] =
                    (self.peak_hold_time * self.sample_rate) as usize;
            } else if self.peak_hold_counters[band_idx] > 0 {
                self.peak_hold_counters[band_idx] -= 1;
            } else {
                self.peak_hold[band_idx] = band_level;
            }
        }
    }

    /// Get band levels in linear scale.
    pub fn band_levels(&self) -> &[f64] {
        &self.band_levels
    }

    /// Get band levels in dB.
    pub fn band_levels_db(&self) -> Vec<f64> {
        self.band_levels
            .iter()
            .map(|&level| {
                if level > 0.0 {
                    20.0 * level.log10()
                } else {
                    f64::NEG_INFINITY
                }
            })
            .collect()
    }

    /// Get the bands.
    pub fn bands(&self) -> &[OctaveBand] {
        &self.bands
    }

    /// Reset the analyzer.
    pub fn reset(&mut self) {
        self.band_levels.fill(0.0);
        self.peak_hold.fill(0.0);
        self.peak_hold_counters.fill(0);
    }
}

/// Generate 1/3 octave band definitions.
fn generate_third_octave_bands() -> Vec<OctaveBand> {
    let mut bands = Vec::new();
    let base_freq = 1000.0; // 1 kHz reference
    let ratio = 2.0_f64.powf(1.0 / 3.0); // Third octave ratio

    // Generate bands from 20 Hz to 20 kHz
    for i in -18..13 {
        let center = base_freq * ratio.powi(i);
        let lower = center / ratio.sqrt();
        let upper = center * ratio.sqrt();

        if (20.0..=20_000.0).contains(&center) {
            bands.push(OctaveBand {
                center_freq: center,
                lower_freq: lower,
                upper_freq: upper,
            });
        }
    }

    bands
}

/// Generate 1/6 octave band definitions.
fn generate_sixth_octave_bands() -> Vec<OctaveBand> {
    let mut bands = Vec::new();
    let base_freq = 1000.0; // 1 kHz reference
    let ratio = 2.0_f64.powf(1.0 / 6.0); // Sixth octave ratio

    // Generate bands from 20 Hz to 20 kHz
    for i in -36..26 {
        let center = base_freq * ratio.powi(i);
        let lower = center / ratio.sqrt();
        let upper = center * ratio.sqrt();

        if (20.0..=20_000.0).contains(&center) {
            bands.push(OctaveBand {
                center_freq: center,
                lower_freq: lower,
                upper_freq: upper,
            });
        }
    }

    bands
}

/// A-weighting gain calculation.
fn a_weighting_gain(freq: f64) -> f64 {
    let f2 = freq * freq;
    let num = 12_194.0 * 12_194.0 * f2 * f2;
    let den1 = f2 + 20.6 * 20.6;
    let den2 = f2 + 12_194.0 * 12_194.0;
    let den3 = (f2 + 107.7 * 107.7).sqrt() * (f2 + 737.9 * 737.9).sqrt();

    let db = 20.0 * (num / (den1 * den2 * den3)).log10() + 2.0;
    10.0_f64.powf(db / 20.0)
}

/// B-weighting gain calculation.
fn b_weighting_gain(freq: f64) -> f64 {
    let f2 = freq * freq;
    let num = 12_194.0 * 12_194.0 * f2 * freq;
    let den1 = f2 + 20.6 * 20.6;
    let den2 = f2 + 12_194.0 * 12_194.0;
    let den3 = (f2 + 158.5 * 158.5).sqrt();

    let db = 20.0 * (num / (den1 * den2 * den3)).log10() + 0.17;
    10.0_f64.powf(db / 20.0)
}

/// C-weighting gain calculation.
fn c_weighting_gain(freq: f64) -> f64 {
    let f2 = freq * freq;
    let num = 12_194.0 * 12_194.0 * f2;
    let den1 = f2 + 20.6 * 20.6;
    let den2 = f2 + 12_194.0 * 12_194.0;

    let db = 20.0 * (num / (den1 * den2)).log10() + 0.06;
    10.0_f64.powf(db / 20.0)
}

/// K-weighting gain calculation (simplified approximation).
fn k_weighting_gain(_freq: f64) -> f64 {
    // K-weighting is applied as a filter, not a simple gain curve
    // This is a placeholder; actual K-weighting uses the filters module
    1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectrum_analyzer_creation() {
        let analyzer = SpectrumAnalyzer::new(
            48000.0,
            2048,
            WindowFunction::Hann,
            WeightingCurve::Flat,
            1.0,
        );

        assert!(analyzer.is_ok());
        let analyzer = analyzer.expect("analyzer should be valid");
        assert_eq!(analyzer.num_bins(), 1025);
    }

    #[test]
    fn test_window_functions() {
        assert_eq!(WindowFunction::Rectangular.coefficient(0, 100), 1.0);
        assert!(WindowFunction::Hann.coefficient(50, 100) > 0.9);
        assert!(WindowFunction::Hann.coefficient(0, 100) < 0.1);
    }

    #[test]
    fn test_third_octave_bands() {
        let bands = generate_third_octave_bands();
        assert!(!bands.is_empty());
        assert!(bands[0].center_freq >= 20.0);
    }

    #[test]
    fn test_sixth_octave_bands() {
        let bands = generate_sixth_octave_bands();
        assert!(!bands.is_empty());
        assert!(bands[0].center_freq >= 20.0);
    }

    #[test]
    fn test_a_weighting() {
        let gain_1k = a_weighting_gain(1000.0);
        let gain_100 = a_weighting_gain(100.0);

        // A-weighting should attenuate low frequencies
        assert!(gain_100 < gain_1k);
    }

    #[test]
    fn test_octave_band_analyzer() {
        let mut analyzer = OctaveBandAnalyzer::new_third_octave(48000.0, 1.0);

        let spectrum = vec![0.1; 1024];
        analyzer.process_spectrum(&spectrum, 2048);

        assert!(!analyzer.band_levels().is_empty());
    }
}
