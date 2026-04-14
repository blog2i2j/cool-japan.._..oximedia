//! Real-time spectrum analyzer for the `OxiMedia` mixer.
//!
//! Implements a rolling-window spectrum analyzer using a pure-Rust DFT/FFT
//! approach (Cooley-Tukey radix-2 iterative FFT).  The analyzer computes
//! magnitude spectrum bins, supports configurable window functions
//! (Hann, Blackman, Flat-Top, Rectangular), and provides logarithmic
//! frequency-bin grouping useful for mixing console display.
//!
//! # Design
//!
//! ```text
//! audio samples → window function → FFT → magnitude → log-bin grouping → SpectrumFrame
//! ```
//!
//! Processing is intentionally allocation-light: the analyzer pre-allocates
//! all internal buffers at construction time and reuses them across calls.

/// Window function applied to the analysis block before the DFT.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowFunction {
    /// Rectangular (no windowing) — maximum frequency resolution, highest spectral leakage.
    Rectangular,
    /// Hann window — good general-purpose window.
    Hann,
    /// Blackman window — lower leakage, slightly wider main lobe.
    Blackman,
    /// Flat-Top window — best amplitude accuracy, widest main lobe.
    FlatTop,
}

impl WindowFunction {
    /// Compute the window coefficient at sample index `n` in a block of `size` samples.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn coefficient(self, n: usize, size: usize) -> f32 {
        if size == 0 {
            return 0.0;
        }
        let n_f = n as f64;
        let n_m1 = (size - 1) as f64;
        match self {
            Self::Rectangular => 1.0,
            Self::Hann => {
                (0.5 * (1.0 - (2.0 * std::f64::consts::PI * n_f / n_m1).cos())) as f32
            }
            Self::Blackman => {
                let a0: f64 = 0.42;
                let a1: f64 = 0.5;
                let a2: f64 = 0.08;
                let x = 2.0 * std::f64::consts::PI * n_f / n_m1;
                (a0 - a1 * x.cos() + a2 * (2.0 * x).cos()) as f32
            }
            Self::FlatTop => {
                let x = 2.0 * std::f64::consts::PI * n_f / n_m1;
                (1.0 - 1.930_796 * x.cos() + 1.290_115 * (2.0 * x).cos()
                    - 0.388_168 * (3.0 * x).cos()
                    + 0.032_136 * (4.0 * x).cos()) as f32
            }
        }
    }

    /// Build a window coefficient table for `size` samples.
    #[must_use]
    pub fn build_table(self, size: usize) -> Vec<f32> {
        (0..size).map(|n| self.coefficient(n, size)).collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Cooley-Tukey radix-2 iterative in-place FFT (pure Rust, no dependencies)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the bit-reversal permutation index.
fn bit_reverse(mut x: usize, log2n: u32) -> usize {
    let mut result = 0usize;
    for _ in 0..log2n {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}

/// In-place radix-2 Cooley-Tukey FFT.
///
/// `re` and `im` must have the same power-of-two length.
/// After the call they contain the DFT output (complex).
fn fft_inplace(re: &mut [f32], im: &mut [f32]) {
    let n = re.len();
    if n <= 1 {
        return;
    }
    let log2n = n.trailing_zeros();

    // Bit-reversal permutation
    for i in 0..n {
        let j = bit_reverse(i, log2n);
        if j > i {
            re.swap(i, j);
            im.swap(i, j);
        }
    }

    // Butterfly stages
    let mut half = 1usize;
    while half < n {
        let full = half * 2;
        #[allow(clippy::cast_precision_loss)]
        let angle_step = -std::f64::consts::PI / half as f64;
        let mut k = 0usize;
        while k < n {
            for j in 0..half {
                let tw_re = (angle_step * j as f64).cos() as f32;
                let tw_im = (angle_step * j as f64).sin() as f32;
                let u_re = re[k + j];
                let u_im = im[k + j];
                let v_re = re[k + j + half] * tw_re - im[k + j + half] * tw_im;
                let v_im = re[k + j + half] * tw_im + im[k + j + half] * tw_re;
                re[k + j] = u_re + v_re;
                im[k + j] = u_im + v_im;
                re[k + j + half] = u_re - v_re;
                im[k + j + half] = u_im - v_im;
            }
            k += full;
        }
        half = full;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SpectrumBin
// ─────────────────────────────────────────────────────────────────────────────

/// A single bin in the spectrum output.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpectrumBin {
    /// Centre frequency of this bin in Hz.
    pub frequency_hz: f32,
    /// Magnitude in dBFS.
    pub magnitude_db: f32,
    /// Linear magnitude (0.0..=1.0+).
    pub magnitude_linear: f32,
}

// ─────────────────────────────────────────────────────────────────────────────
// SpectrumFrame
// ─────────────────────────────────────────────────────────────────────────────

/// A single spectrum analysis result.
#[derive(Debug, Clone)]
pub struct SpectrumFrame {
    /// Analysis bins, ordered by ascending frequency.
    pub bins: Vec<SpectrumBin>,
    /// Sample rate used for this frame.
    pub sample_rate: u32,
    /// FFT block size (number of samples analysed).
    pub fft_size: usize,
}

impl SpectrumFrame {
    /// Find the bin with the highest magnitude.
    #[must_use]
    pub fn peak_bin(&self) -> Option<&SpectrumBin> {
        self.bins
            .iter()
            .max_by(|a, b| a.magnitude_db.partial_cmp(&b.magnitude_db).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Return the magnitude in dBFS at the given frequency, interpolating between bins.
    ///
    /// Returns `None` if `bins` is empty.
    #[must_use]
    pub fn magnitude_at(&self, frequency_hz: f32) -> Option<f32> {
        if self.bins.is_empty() {
            return None;
        }
        // Find the two bins bracketing the target frequency
        let idx = self
            .bins
            .partition_point(|b| b.frequency_hz < frequency_hz);
        if idx == 0 {
            return Some(self.bins[0].magnitude_db);
        }
        if idx >= self.bins.len() {
            return Some(self.bins[self.bins.len() - 1].magnitude_db);
        }
        let lo = &self.bins[idx - 1];
        let hi = &self.bins[idx];
        let span = hi.frequency_hz - lo.frequency_hz;
        let t = if span > 0.0 {
            (frequency_hz - lo.frequency_hz) / span
        } else {
            0.0
        };
        Some(lo.magnitude_db + (hi.magnitude_db - lo.magnitude_db) * t)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SpectrumAnalyzerConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the spectrum analyzer.
#[derive(Debug, Clone)]
pub struct SpectrumAnalyzerConfig {
    /// FFT block size (must be a power of two, ≥ 64).
    pub fft_size: usize,
    /// Window function applied before the DFT.
    pub window: WindowFunction,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Minimum magnitude threshold in dBFS (below this, bins are reported at
    /// this floor rather than −∞).
    pub floor_db: f32,
    /// Number of logarithmically-spaced output bins (0 = use raw FFT bins).
    pub log_bins: usize,
    /// Lowest frequency for log-bin grouping in Hz.
    pub log_freq_min: f32,
    /// Highest frequency for log-bin grouping in Hz (0 = Nyquist).
    pub log_freq_max: f32,
}

impl Default for SpectrumAnalyzerConfig {
    fn default() -> Self {
        Self {
            fft_size: 2048,
            window: WindowFunction::Hann,
            sample_rate: 48000,
            floor_db: -90.0,
            log_bins: 128,
            log_freq_min: 20.0,
            log_freq_max: 0.0, // 0 = Nyquist
        }
    }
}

impl SpectrumAnalyzerConfig {
    /// Validate and sanitise the configuration.
    ///
    /// Returns an error string if the configuration is invalid.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `fft_size` is not a power of two or is less than 64.
    pub fn validate(&self) -> Result<(), String> {
        if self.fft_size < 64 || !self.fft_size.is_power_of_two() {
            return Err(format!(
                "fft_size must be a power of two ≥ 64, got {}",
                self.fft_size
            ));
        }
        if self.sample_rate == 0 {
            return Err("sample_rate must be > 0".into());
        }
        Ok(())
    }

    /// Nyquist frequency in Hz.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn nyquist(&self) -> f32 {
        self.sample_rate as f32 / 2.0
    }

    /// Effective maximum frequency for log bins.
    #[must_use]
    pub fn effective_log_freq_max(&self) -> f32 {
        if self.log_freq_max <= 0.0 {
            self.nyquist()
        } else {
            self.log_freq_max.min(self.nyquist())
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SpectrumAnalyzer
// ─────────────────────────────────────────────────────────────────────────────

/// Real-time mono spectrum analyzer.
///
/// Feed audio samples with [`SpectrumAnalyzer::push`], then call
/// [`SpectrumAnalyzer::analyze`] to obtain a [`SpectrumFrame`].
///
/// Internally maintains a ring-buffer of the most recent `fft_size` samples.
pub struct SpectrumAnalyzer {
    config: SpectrumAnalyzerConfig,
    /// Ring buffer holding the most recent samples.
    ring: Vec<f32>,
    /// Write index into the ring buffer.
    write_pos: usize,
    /// Pre-computed window coefficients.
    window_table: Vec<f32>,
    /// FFT working buffer (real part).
    fft_re: Vec<f32>,
    /// FFT working buffer (imaginary part).
    fft_im: Vec<f32>,
}

impl std::fmt::Debug for SpectrumAnalyzer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpectrumAnalyzer")
            .field("fft_size", &self.config.fft_size)
            .field("sample_rate", &self.config.sample_rate)
            .field("log_bins", &self.config.log_bins)
            .finish()
    }
}

impl SpectrumAnalyzer {
    /// Create a new spectrum analyzer from configuration.
    ///
    /// # Errors
    ///
    /// Returns an error string if the configuration is invalid.
    pub fn new(config: SpectrumAnalyzerConfig) -> Result<Self, String> {
        config.validate()?;
        let n = config.fft_size;
        let window_table = config.window.build_table(n);
        Ok(Self {
            ring: vec![0.0f32; n],
            write_pos: 0,
            window_table,
            fft_re: vec![0.0f32; n],
            fft_im: vec![0.0f32; n],
            config,
        })
    }

    /// Push a single sample into the analyzer's ring buffer.
    pub fn push(&mut self, sample: f32) {
        self.ring[self.write_pos] = sample;
        self.write_pos = (self.write_pos + 1) % self.config.fft_size;
    }

    /// Push a slice of samples.
    pub fn push_slice(&mut self, samples: &[f32]) {
        for &s in samples {
            self.push(s);
        }
    }

    /// Perform spectral analysis on the current ring-buffer contents and return a
    /// [`SpectrumFrame`].
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn analyze(&mut self) -> SpectrumFrame {
        let n = self.config.fft_size;
        // Copy ring buffer into FFT input, applying window.
        for i in 0..n {
            let ring_idx = (self.write_pos + i) % n;
            self.fft_re[i] = self.ring[ring_idx] * self.window_table[i];
            self.fft_im[i] = 0.0;
        }

        fft_inplace(&mut self.fft_re, &mut self.fft_im);

        // Compute magnitude spectrum for the positive half (bins 0..=N/2).
        let n_half = n / 2 + 1;
        let normalization = 1.0_f32 / n as f32;
        let floor = self.config.floor_db;

        let raw_magnitudes: Vec<(f32, f32)> = (0..n_half)
            .map(|k| {
                let mag_linear = (self.fft_re[k].powi(2) + self.fft_im[k].powi(2))
                    .sqrt()
                    * normalization;
                let mag_db = if mag_linear > 0.0 {
                    (20.0 * mag_linear.log10()).max(floor)
                } else {
                    floor
                };
                (mag_linear, mag_db)
            })
            .collect();

        let sr = self.config.sample_rate as f32;
        let bin_freq = |k: usize| k as f32 * sr / n as f32;

        if self.config.log_bins == 0 || self.config.log_bins >= n_half {
            // Return raw FFT bins
            let bins = (0..n_half)
                .map(|k| SpectrumBin {
                    frequency_hz: bin_freq(k),
                    magnitude_db: raw_magnitudes[k].1,
                    magnitude_linear: raw_magnitudes[k].0,
                })
                .collect();
            return SpectrumFrame {
                bins,
                sample_rate: self.config.sample_rate,
                fft_size: n,
            };
        }

        // Build log-spaced output bins
        let freq_min = self.config.log_freq_min.max(1.0);
        let freq_max = self.config.effective_log_freq_max();
        let num_bins = self.config.log_bins;

        let log_min = freq_min.log2();
        let log_max = freq_max.log2();
        let log_step = (log_max - log_min) / (num_bins - 1).max(1) as f32;

        let bins: Vec<SpectrumBin> = (0..num_bins)
            .map(|b| {
                let centre_freq = 2.0_f32.powf(log_min + b as f32 * log_step);
                // Average magnitude of all raw bins that fall within this log band.
                let f_lo = if b == 0 {
                    freq_min
                } else {
                    2.0_f32.powf(log_min + (b as f32 - 0.5) * log_step)
                };
                let f_hi = if b == num_bins - 1 {
                    freq_max
                } else {
                    2.0_f32.powf(log_min + (b as f32 + 0.5) * log_step)
                };
                // Map frequency range to FFT bin range.
                let k_lo = ((f_lo / sr * n as f32) as usize).max(0).min(n_half - 1);
                let k_hi = ((f_hi / sr * n as f32) as usize + 1)
                    .max(k_lo + 1)
                    .min(n_half);
                // Maximum magnitude in range for display.
                let (best_linear, best_db) = (k_lo..k_hi).fold((0.0f32, floor), |acc, k| {
                    let (ml, md) = raw_magnitudes[k];
                    (acc.0.max(ml), acc.1.max(md))
                });
                SpectrumBin {
                    frequency_hz: centre_freq,
                    magnitude_db: best_db,
                    magnitude_linear: best_linear,
                }
            })
            .collect();

        SpectrumFrame {
            bins,
            sample_rate: self.config.sample_rate,
            fft_size: n,
        }
    }

    /// Return the current configuration.
    #[must_use]
    pub fn config(&self) -> &SpectrumAnalyzerConfig {
        &self.config
    }

    /// Clear the internal ring buffer (silence).
    pub fn reset(&mut self) {
        self.ring.fill(0.0);
        self.write_pos = 0;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn make_sine(freq_hz: f32, sample_rate: u32, n: usize) -> Vec<f32> {
        #[allow(clippy::cast_precision_loss)]
        (0..n)
            .map(|i| (2.0 * PI * freq_hz * i as f32 / sample_rate as f32).sin())
            .collect()
    }

    #[test]
    fn test_window_rectangular_is_all_ones() {
        let table = WindowFunction::Rectangular.build_table(8);
        for w in &table {
            assert!((*w - 1.0).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_window_hann_endpoints_near_zero() {
        let n = 256;
        let table = WindowFunction::Hann.build_table(n);
        assert!(table[0].abs() < 1e-5, "hann[0] should be ~0");
        assert!(table[n - 1].abs() < 1e-3, "hann[n-1] should be ~0");
    }

    #[test]
    fn test_window_hann_midpoint_near_one() {
        // For even n, the Hann window peak is at (n-1)/2, not n/2.
        // Use n=257 (odd) so that n/2 = 128 is the exact midpoint.
        let n = 257;
        let table = WindowFunction::Hann.build_table(n);
        // For odd n the peak is at index n/2 = 128 (integer division).
        assert!((table[n / 2] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_window_blackman_range() {
        let n = 1024;
        let table = WindowFunction::Blackman.build_table(n);
        for &w in &table {
            assert!(w >= -0.01 && w <= 1.01, "blackman coefficient out of range: {w}");
        }
    }

    #[test]
    fn test_analyzer_creation_valid() {
        let config = SpectrumAnalyzerConfig::default();
        let analyzer = SpectrumAnalyzer::new(config);
        assert!(analyzer.is_ok());
    }

    #[test]
    fn test_analyzer_creation_invalid_fft_size() {
        let config = SpectrumAnalyzerConfig {
            fft_size: 100, // not a power of two
            ..Default::default()
        };
        let result = SpectrumAnalyzer::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_analyzer_creation_too_small() {
        let config = SpectrumAnalyzerConfig {
            fft_size: 32, // < 64
            ..Default::default()
        };
        let result = SpectrumAnalyzer::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_silence_produces_floor_db() {
        let config = SpectrumAnalyzerConfig {
            fft_size: 256,
            log_bins: 0, // raw bins
            floor_db: -90.0,
            ..Default::default()
        };
        let mut analyzer = SpectrumAnalyzer::new(config).unwrap();
        let frame = analyzer.analyze();
        for bin in &frame.bins {
            assert!(
                bin.magnitude_db <= -89.0,
                "silence should be near floor, got {}",
                bin.magnitude_db
            );
        }
    }

    #[test]
    fn test_sine_peak_near_frequency() {
        let sr = 48000u32;
        let fft_size = 4096usize;
        let test_freq = 1000.0_f32;
        let config = SpectrumAnalyzerConfig {
            fft_size,
            log_bins: 0, // raw bins
            window: WindowFunction::Hann,
            sample_rate: sr,
            floor_db: -120.0,
            ..Default::default()
        };
        let mut analyzer = SpectrumAnalyzer::new(config).unwrap();
        let signal = make_sine(test_freq, sr, fft_size * 2);
        analyzer.push_slice(&signal);
        let frame = analyzer.analyze();
        let peak = frame.peak_bin().unwrap();
        // Peak bin should be within 2 bins of the expected frequency.
        #[allow(clippy::cast_precision_loss)]
        let bin_width = sr as f32 / fft_size as f32;
        let diff = (peak.frequency_hz - test_freq).abs();
        assert!(
            diff < 2.0 * bin_width,
            "peak at {} Hz, expected ~{test_freq} Hz (bin_width={bin_width})",
            peak.frequency_hz
        );
    }

    #[test]
    fn test_log_bins_count() {
        let config = SpectrumAnalyzerConfig {
            fft_size: 1024,
            log_bins: 64,
            ..Default::default()
        };
        let mut analyzer = SpectrumAnalyzer::new(config).unwrap();
        let frame = analyzer.analyze();
        assert_eq!(frame.bins.len(), 64);
    }

    #[test]
    fn test_log_bins_frequencies_ascending() {
        let config = SpectrumAnalyzerConfig {
            fft_size: 1024,
            log_bins: 32,
            ..Default::default()
        };
        let mut analyzer = SpectrumAnalyzer::new(config).unwrap();
        let frame = analyzer.analyze();
        let freqs: Vec<f32> = frame.bins.iter().map(|b| b.frequency_hz).collect();
        for w in freqs.windows(2) {
            assert!(w[1] > w[0], "log bins should be in ascending order");
        }
    }

    #[test]
    fn test_reset_clears_buffer() {
        let config = SpectrumAnalyzerConfig {
            fft_size: 256,
            log_bins: 0,
            floor_db: -90.0,
            ..Default::default()
        };
        let mut analyzer = SpectrumAnalyzer::new(config).unwrap();
        // Push a loud tone.
        let signal = make_sine(1000.0, 48000, 256);
        analyzer.push_slice(&signal);
        // Reset and analyze — should be back to silence.
        analyzer.reset();
        let frame = analyzer.analyze();
        for bin in &frame.bins {
            assert!(
                bin.magnitude_db <= -89.0,
                "after reset should be floor, got {}",
                bin.magnitude_db
            );
        }
    }

    #[test]
    fn test_magnitude_at_interpolation() {
        let config = SpectrumAnalyzerConfig {
            fft_size: 256,
            log_bins: 0,
            floor_db: -90.0,
            ..Default::default()
        };
        let mut analyzer = SpectrumAnalyzer::new(config).unwrap();
        let frame = analyzer.analyze();
        // Should return Some for any frequency within range.
        assert!(frame.magnitude_at(100.0).is_some());
        assert!(frame.magnitude_at(10000.0).is_some());
    }

    #[test]
    fn test_spectrum_frame_peak_bin_on_silence() {
        let config = SpectrumAnalyzerConfig {
            fft_size: 256,
            log_bins: 0,
            floor_db: -90.0,
            ..Default::default()
        };
        let mut analyzer = SpectrumAnalyzer::new(config).unwrap();
        let frame = analyzer.analyze();
        // peak_bin should return Some (even on silence).
        assert!(frame.peak_bin().is_some());
    }

    #[test]
    fn test_push_slice_matches_push_individual() {
        let config = SpectrumAnalyzerConfig {
            fft_size: 128,
            log_bins: 0,
            floor_db: -90.0,
            window: WindowFunction::Rectangular,
            ..Default::default()
        };
        let signal = make_sine(440.0, 48000, 128);

        let mut a = SpectrumAnalyzer::new(config.clone()).unwrap();
        let mut b = SpectrumAnalyzer::new(config).unwrap();

        a.push_slice(&signal);
        for s in &signal {
            b.push(*s);
        }

        let fa = a.analyze();
        let fb = b.analyze();
        for (ba, bb) in fa.bins.iter().zip(fb.bins.iter()) {
            assert!(
                (ba.magnitude_db - bb.magnitude_db).abs() < 0.1,
                "push_slice and push should produce identical results"
            );
        }
    }

    #[test]
    fn test_fft_inplace_identity_impulse() {
        // DFT of a unit impulse at t=0 should be all-ones (magnitude).
        let n = 64;
        let mut re = vec![0.0f32; n];
        let mut im = vec![0.0f32; n];
        re[0] = 1.0;
        fft_inplace(&mut re, &mut im);
        for k in 0..n {
            let mag = (re[k].powi(2) + im[k].powi(2)).sqrt();
            assert!((mag - 1.0).abs() < 1e-4, "impulse DFT bin {k}: mag={mag}");
        }
    }

    #[test]
    fn test_config_validate_non_power_of_two() {
        let config = SpectrumAnalyzerConfig {
            fft_size: 300,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_nyquist() {
        let config = SpectrumAnalyzerConfig {
            sample_rate: 44100,
            ..Default::default()
        };
        assert!((config.nyquist() - 22050.0).abs() < 0.1);
    }
}
