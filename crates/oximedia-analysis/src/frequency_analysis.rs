#![allow(dead_code)]
//! Frequency-domain analysis for video frame luminance signals.
//!
//! This module performs frequency-domain analysis on the per-frame mean
//! luminance signal derived from a video clip. It computes:
//!
//! - **Power Spectral Density (PSD)** — Welch-method approximation using a
//!   sliding window of frame luminance values.
//! - **Dominant frequency** — The spectral bin with the highest power, reported
//!   in cycles-per-frame and in Hz (given a known frame rate).
//! - **Harmonic ratio** — Ratio of power contained in harmonic multiples of the
//!   dominant frequency, indicating periodic structure vs. noise.
//! - **Spectral entropy** — Normalized Shannon entropy of the PSD, a proxy for
//!   spectral flatness / complexity.
//! - **Low / mid / high band power fractions** — Breakdown of energy into
//!   three frequency bands.
//!
//! The module is purely computational and does **not** depend on external FFT
//! libraries; it uses a hand-rolled Radix-2 Cooley-Tukey DFT on power-of-two
//! windows so that the crate remains self-contained.

use std::collections::VecDeque;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for frequency analysis.
#[derive(Debug, Clone)]
pub struct FrequencyAnalysisConfig {
    /// Number of recent per-frame luminance values to keep in the analysis
    /// window.  Must be a power of two >= 4.  Defaults to 64.
    pub window_size: usize,
    /// Frame rate (frames per second) of the source video.  Used to convert
    /// cycle-per-frame frequencies to Hz.  Defaults to 25.0.
    pub frame_rate: f64,
    /// Number of harmonic multiples to include in the harmonic-ratio
    /// computation (beyond the fundamental).  Defaults to 3.
    pub harmonics: usize,
    /// Fraction of total power that defines the boundary between the *low*
    /// and *mid* bands (0.0..1.0).  Defaults to 0.25 (lowest 25 % of bins).
    pub low_band_fraction: f64,
    /// Fraction of total power that defines the boundary between the *mid*
    /// and *high* bands (0.0..1.0).  Defaults to 0.75 (lowest 75 % of bins).
    pub mid_band_fraction: f64,
}

impl Default for FrequencyAnalysisConfig {
    fn default() -> Self {
        Self {
            window_size: 64,
            frame_rate: 25.0,
            harmonics: 3,
            low_band_fraction: 0.25,
            mid_band_fraction: 0.75,
        }
    }
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Power spectral density for one analysis window.
#[derive(Debug, Clone)]
pub struct PowerSpectrum {
    /// Power values for each frequency bin (length = window_size / 2 + 1).
    /// Values are in units of (luminance^2 / bin).
    pub power: Vec<f64>,
    /// Frequency (in cycles per frame) of each bin.
    pub frequencies_cpf: Vec<f64>,
    /// Total power across all bins.
    pub total_power: f64,
}

/// Dominant frequency information.
#[derive(Debug, Clone, Copy)]
pub struct DominantFrequency {
    /// Bin index with the maximum power (DC excluded when dc_excluded = true).
    pub bin_index: usize,
    /// Frequency in cycles per frame.
    pub cpf: f64,
    /// Frequency in Hz (cpf × frame_rate).
    pub hz: f64,
    /// Power in the dominant bin as a fraction of total power.
    pub power_fraction: f64,
}

/// Harmonic analysis result.
#[derive(Debug, Clone)]
pub struct HarmonicAnalysis {
    /// Fundamental frequency bin index.
    pub fundamental_bin: usize,
    /// Power in the fundamental bin.
    pub fundamental_power: f64,
    /// Powers in the harmonic multiples (2×, 3×, … up to `harmonics`).
    pub harmonic_powers: Vec<f64>,
    /// Ratio: (fundamental + harmonics) / total power.
    pub harmonic_ratio: f64,
}

/// Complete result from one frequency analysis pass.
#[derive(Debug, Clone)]
pub struct FrequencyAnalysisResult {
    /// The computed power spectrum.
    pub spectrum: PowerSpectrum,
    /// Dominant frequency information.
    pub dominant: DominantFrequency,
    /// Harmonic analysis centred on the dominant frequency.
    pub harmonics: HarmonicAnalysis,
    /// Normalised spectral entropy (0 = single spike, 1 = perfectly flat).
    pub spectral_entropy: f64,
    /// Fraction of total power in the low-frequency band.
    pub low_band_power: f64,
    /// Fraction of total power in the mid-frequency band.
    pub mid_band_power: f64,
    /// Fraction of total power in the high-frequency band.
    pub high_band_power: f64,
    /// Number of luminance samples used (equals window_size).
    pub sample_count: usize,
}

// ---------------------------------------------------------------------------
// Analyzer
// ---------------------------------------------------------------------------

/// Stateful frequency analyzer fed one frame at a time.
///
/// Call [`push_frame`] with the Y-plane bytes of each video frame; the
/// analyzer maintains a rolling window of per-frame mean luminance values.
/// Call [`analyze`] at any point to obtain [`FrequencyAnalysisResult`] from
/// the current window.
#[derive(Debug)]
pub struct FrequencyAnalyzer {
    config: FrequencyAnalysisConfig,
    /// Rolling window of normalised (0..1) per-frame mean luminance.
    luma_window: VecDeque<f64>,
    /// Total frames pushed.
    frame_count: usize,
}

impl FrequencyAnalyzer {
    /// Create a new analyzer with default configuration.
    pub fn new() -> Self {
        Self::with_config(FrequencyAnalysisConfig::default())
    }

    /// Create a new analyzer with custom configuration.
    ///
    /// # Errors
    /// Returns `None` if `window_size` is not a power of two or is less than 4.
    pub fn with_config(config: FrequencyAnalysisConfig) -> Self {
        let ws = config.window_size.max(4).next_power_of_two();
        let effective_config = FrequencyAnalysisConfig {
            window_size: ws,
            ..config
        };
        Self {
            luma_window: VecDeque::with_capacity(ws),
            config: effective_config,
            frame_count: 0,
        }
    }

    /// Push a Y-plane frame into the analyzer.
    ///
    /// The frame mean luminance is normalised to [0.0, 1.0] before storage.
    pub fn push_frame(&mut self, y_plane: &[u8], width: usize, height: usize) {
        let n = width * height;
        if n == 0 {
            return;
        }
        let mean = y_plane.iter().map(|&v| f64::from(v)).sum::<f64>() / n as f64 / 255.0;

        if self.luma_window.len() >= self.config.window_size {
            self.luma_window.pop_front();
        }
        self.luma_window.push_back(mean);
        self.frame_count += 1;
    }

    /// Returns the number of frames pushed so far.
    pub fn frame_count(&self) -> usize {
        self.frame_count
    }

    /// Returns the number of luminance samples currently in the window.
    pub fn window_len(&self) -> usize {
        self.luma_window.len()
    }

    /// Perform frequency analysis on the current window.
    ///
    /// Returns `None` if fewer than 4 samples are available.
    pub fn analyze(&self) -> Option<FrequencyAnalysisResult> {
        let n = self.luma_window.len();
        if n < 4 {
            return None;
        }

        // Collect and zero-pad to the configured window size
        let pad_n = self.config.window_size;
        let mut signal: Vec<f64> = self.luma_window.iter().copied().collect();
        signal.resize(pad_n, 0.0);

        // Apply Hann window to reduce spectral leakage
        apply_hann_window(&mut signal);

        // Forward DFT
        let spectrum = compute_power_spectrum(&signal, self.config.frame_rate);

        let num_bins = spectrum.power.len();

        // Dominant frequency (skip DC bin 0)
        let dominant = find_dominant_frequency(&spectrum, self.config.frame_rate);

        // Harmonic analysis
        let harmonics = compute_harmonic_analysis(
            &spectrum,
            dominant.bin_index,
            self.config.harmonics,
        );

        // Spectral entropy
        let spectral_entropy = compute_spectral_entropy(&spectrum.power, spectrum.total_power);

        // Band powers
        let low_limit = ((num_bins as f64 * self.config.low_band_fraction) as usize)
            .clamp(1, num_bins);
        let mid_limit = ((num_bins as f64 * self.config.mid_band_fraction) as usize)
            .clamp(low_limit, num_bins);

        let (low_pwr, mid_pwr, high_pwr) =
            compute_band_powers(&spectrum.power, low_limit, mid_limit, spectrum.total_power);

        Some(FrequencyAnalysisResult {
            sample_count: n,
            spectrum,
            dominant,
            harmonics,
            spectral_entropy,
            low_band_power: low_pwr,
            mid_band_power: mid_pwr,
            high_band_power: high_pwr,
        })
    }

    /// Reset all state (window cleared, frame count zeroed).
    pub fn reset(&mut self) {
        self.luma_window.clear();
        self.frame_count = 0;
    }
}

impl Default for FrequencyAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// DSP helpers
// ---------------------------------------------------------------------------

/// Apply an in-place Hann window to a real-valued signal.
fn apply_hann_window(signal: &mut [f64]) {
    let n = signal.len();
    if n == 0 {
        return;
    }
    for (i, s) in signal.iter_mut().enumerate() {
        let w = 0.5 * (1.0 - (2.0 * PI * i as f64 / (n - 1) as f64).cos());
        *s *= w;
    }
}

/// In-place Radix-2 Cooley-Tukey FFT (Decimation in time).
///
/// `buf` must have power-of-two length.
/// Returns real/imag pairs interleaved: buf[2i] = Re, buf[2i+1] = Im.
fn fft_inplace(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    debug_assert!(n.is_power_of_two());
    debug_assert_eq!(im.len(), n);

    // Bit-reversal permutation
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }

    // Butterfly stages
    let mut len = 2usize;
    while len <= n {
        let half = len / 2;
        let angle = -2.0 * PI / len as f64;
        let wr0 = angle.cos();
        let wi0 = angle.sin();

        let mut i = 0;
        while i < n {
            let (mut wr, mut wi) = (1.0_f64, 0.0_f64);
            for k in 0..half {
                let u_re = re[i + k];
                let u_im = im[i + k];
                let v_re = re[i + k + half] * wr - im[i + k + half] * wi;
                let v_im = re[i + k + half] * wi + im[i + k + half] * wr;
                re[i + k] = u_re + v_re;
                im[i + k] = u_im + v_im;
                re[i + k + half] = u_re - v_re;
                im[i + k + half] = u_im - v_im;
                let new_wr = wr * wr0 - wi * wi0;
                let new_wi = wr * wi0 + wi * wr0;
                wr = new_wr;
                wi = new_wi;
            }
            i += len;
        }
        len <<= 1;
    }
}

/// Compute the one-sided power spectrum from a real signal.
fn compute_power_spectrum(signal: &[f64], frame_rate: f64) -> PowerSpectrum {
    let n = signal.len();
    let mut re = signal.to_vec();
    let mut im = vec![0.0_f64; n];

    fft_inplace(&mut re, &mut im);

    // One-sided spectrum: bins 0..=n/2
    let num_bins = n / 2 + 1;
    let mut power = Vec::with_capacity(num_bins);
    let mut frequencies_cpf = Vec::with_capacity(num_bins);

    let scale = 1.0 / n as f64;
    for k in 0..num_bins {
        let p = if k == 0 || k == n / 2 {
            (re[k] * re[k] + im[k] * im[k]) * scale * scale
        } else {
            2.0 * (re[k] * re[k] + im[k] * im[k]) * scale * scale
        };
        power.push(p);
        frequencies_cpf.push(k as f64 / n as f64 * frame_rate);
    }

    let total_power = power.iter().sum::<f64>();

    PowerSpectrum {
        power,
        frequencies_cpf,
        total_power,
    }
}

/// Find the dominant (peak-power) frequency, skipping the DC bin.
fn find_dominant_frequency(spectrum: &PowerSpectrum, frame_rate: f64) -> DominantFrequency {
    let bins = &spectrum.power;
    // Start from bin 1 to skip DC
    let start = if bins.len() > 1 { 1 } else { 0 };
    let (max_bin, max_power) = bins[start..]
        .iter()
        .enumerate()
        .fold((0usize, 0.0_f64), |(mi, mp), (i, &p)| {
            if p > mp {
                (i + start, p)
            } else {
                (mi, mp)
            }
        });

    let n_total = (bins.len() - 1) * 2; // original signal length
    let cpf = if n_total > 0 {
        max_bin as f64 / n_total as f64
    } else {
        0.0
    };
    let hz = cpf * frame_rate;
    let power_fraction = if spectrum.total_power > 0.0 {
        max_power / spectrum.total_power
    } else {
        0.0
    };

    DominantFrequency {
        bin_index: max_bin,
        cpf,
        hz,
        power_fraction,
    }
}

/// Compute harmonic analysis around a given fundamental bin.
fn compute_harmonic_analysis(
    spectrum: &PowerSpectrum,
    fundamental_bin: usize,
    num_harmonics: usize,
) -> HarmonicAnalysis {
    let bins = &spectrum.power;
    let n_bins = bins.len();

    let fundamental_power = bins.get(fundamental_bin).copied().unwrap_or(0.0);

    let harmonic_powers: Vec<f64> = (2..=num_harmonics + 1)
        .map(|h| {
            let hbin = fundamental_bin * h;
            if hbin < n_bins {
                bins[hbin]
            } else {
                0.0
            }
        })
        .collect();

    let harmonic_total: f64 = fundamental_power + harmonic_powers.iter().sum::<f64>();
    let harmonic_ratio = if spectrum.total_power > 0.0 {
        harmonic_total / spectrum.total_power
    } else {
        0.0
    };

    HarmonicAnalysis {
        fundamental_bin,
        fundamental_power,
        harmonic_powers,
        harmonic_ratio,
    }
}

/// Compute normalised spectral entropy.
fn compute_spectral_entropy(power: &[f64], total_power: f64) -> f64 {
    if total_power <= 0.0 || power.is_empty() {
        return 0.0;
    }
    let n_bins = power.len();
    let log_n = (n_bins as f64).ln();
    if log_n <= 0.0 {
        return 0.0;
    }
    let entropy = power
        .iter()
        .filter_map(|&p| {
            if p > 0.0 {
                let prob = p / total_power;
                Some(-prob * prob.ln())
            } else {
                None
            }
        })
        .sum::<f64>();
    (entropy / log_n).clamp(0.0, 1.0)
}

/// Compute fractional power in three frequency bands.
fn compute_band_powers(
    power: &[f64],
    low_limit: usize,
    mid_limit: usize,
    total_power: f64,
) -> (f64, f64, f64) {
    if total_power <= 0.0 {
        return (0.0, 0.0, 0.0);
    }
    let n = power.len();
    let low_pwr: f64 = power[..low_limit.min(n)].iter().sum();
    let mid_pwr: f64 = power[low_limit.min(n)..mid_limit.min(n)].iter().sum();
    let high_pwr: f64 = power[mid_limit.min(n)..].iter().sum();
    (
        low_pwr / total_power,
        mid_pwr / total_power,
        high_pwr / total_power,
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a constant-luma frame.
    fn flat_frame(w: usize, h: usize, val: u8) -> Vec<u8> {
        vec![val; w * h]
    }

    /// Push sinusoidal luminance directly by building frames whose pixels vary
    /// to produce the desired per-frame mean.  We use a row-gradient trick so
    /// that the mean of the frame exactly matches our target value without
    /// any saturation clipping.
    fn push_sinusoid(
        analyzer: &mut FrequencyAnalyzer,
        freq_cpf: f64,
        amplitude: f64,
        n_frames: usize,
    ) {
        let w = 16usize;
        let h = 16usize;
        for i in 0..n_frames {
            let mean = 128.0 + amplitude * (2.0 * PI * freq_cpf * i as f64).sin();
            // Build a uniform frame so the mean IS exactly `mean`
            let val = mean.clamp(0.0, 255.0).round() as u8;
            analyzer.push_frame(&flat_frame(w, h, val), w, h);
        }
    }

    #[test]
    fn test_default_config() {
        let cfg = FrequencyAnalysisConfig::default();
        assert_eq!(cfg.window_size, 64);
        assert!((cfg.frame_rate - 25.0).abs() < f64::EPSILON);
        assert_eq!(cfg.harmonics, 3);
    }

    #[test]
    fn test_empty_analyzer_returns_none() {
        let analyzer = FrequencyAnalyzer::new();
        assert!(analyzer.analyze().is_none());
    }

    #[test]
    fn test_insufficient_samples_returns_none() {
        let mut analyzer = FrequencyAnalyzer::new();
        analyzer.push_frame(&flat_frame(4, 4, 128), 4, 4);
        analyzer.push_frame(&flat_frame(4, 4, 130), 4, 4);
        // 2 samples < 4 minimum
        assert!(analyzer.analyze().is_none());
    }

    #[test]
    fn test_constant_signal_low_band_dominates() {
        let mut analyzer = FrequencyAnalyzer::new();
        // Constant luminance => energy concentrated at DC (bin 0)
        for _ in 0..64 {
            analyzer.push_frame(&flat_frame(8, 8, 200), 8, 8);
        }
        let result = analyzer.analyze().expect("should produce result");
        // Low band should dominate for a constant signal
        assert!(
            result.low_band_power >= result.high_band_power,
            "low_band={} high_band={}",
            result.low_band_power,
            result.high_band_power
        );
    }

    #[test]
    fn test_sinusoid_dominant_frequency() {
        let cfg = FrequencyAnalysisConfig {
            window_size: 64,
            frame_rate: 64.0, // use frame_rate == window_size so 1 Hz == 1 cpf exactly
            harmonics: 2,
            ..Default::default()
        };
        let mut analyzer = FrequencyAnalyzer::with_config(cfg);
        // Inject a sinusoid with large amplitude so quantization to u8 is not
        // too lossy.  Use 2 cycles over 64 frames (freq = 2/64 cpf = bin 2).
        // Large amplitude (100 counts) ensures the sinusoidal component is well
        // above any quantization artefact.
        push_sinusoid(&mut analyzer, 2.0 / 64.0, 100.0, 64);

        let result = analyzer.analyze().expect("should produce result");
        // Dominant bin should be near index 2 (allow ±2 for Hann-window spread
        // and rounding).
        let dom_bin = result.dominant.bin_index;
        assert!(
            dom_bin >= 1 && dom_bin <= 5,
            "dominant bin {} should be near 2",
            dom_bin
        );
    }

    #[test]
    fn test_band_powers_sum_to_one() {
        let mut analyzer = FrequencyAnalyzer::new();
        push_sinusoid(&mut analyzer, 0.1, 30.0, 64);
        let result = analyzer.analyze().expect("should produce result");
        let total = result.low_band_power + result.mid_band_power + result.high_band_power;
        assert!(
            (total - 1.0).abs() < 1e-6,
            "band powers should sum to 1.0, got {total}"
        );
    }

    #[test]
    fn test_spectral_entropy_range() {
        let mut analyzer = FrequencyAnalyzer::new();
        push_sinusoid(&mut analyzer, 0.05, 50.0, 64);
        let result = analyzer.analyze().expect("should produce result");
        assert!(
            result.spectral_entropy >= 0.0 && result.spectral_entropy <= 1.0,
            "entropy must be in [0,1], got {}",
            result.spectral_entropy
        );
    }

    #[test]
    fn test_harmonic_ratio_range() {
        let mut analyzer = FrequencyAnalyzer::new();
        push_sinusoid(&mut analyzer, 0.05, 40.0, 64);
        let result = analyzer.analyze().expect("should produce result");
        assert!(
            result.harmonics.harmonic_ratio >= 0.0 && result.harmonics.harmonic_ratio <= 1.0,
            "harmonic_ratio must be in [0,1], got {}",
            result.harmonics.harmonic_ratio
        );
    }

    #[test]
    fn test_zero_dimension_frame_ignored() {
        let mut analyzer = FrequencyAnalyzer::new();
        analyzer.push_frame(&[], 0, 0);
        assert_eq!(analyzer.frame_count(), 0);
        assert_eq!(analyzer.window_len(), 0);
    }

    #[test]
    fn test_reset_clears_state() {
        let mut analyzer = FrequencyAnalyzer::new();
        push_sinusoid(&mut analyzer, 0.1, 20.0, 64);
        assert!(analyzer.window_len() > 0);
        analyzer.reset();
        assert_eq!(analyzer.frame_count(), 0);
        assert_eq!(analyzer.window_len(), 0);
        assert!(analyzer.analyze().is_none());
    }

    #[test]
    fn test_power_spectrum_length() {
        let cfg = FrequencyAnalysisConfig {
            window_size: 32,
            ..Default::default()
        };
        let mut analyzer = FrequencyAnalyzer::with_config(cfg);
        push_sinusoid(&mut analyzer, 0.1, 20.0, 32);
        let result = analyzer.analyze().expect("should produce result");
        // One-sided spectrum: window_size/2 + 1 bins
        assert_eq!(result.spectrum.power.len(), 17); // 32/2+1
    }
}
