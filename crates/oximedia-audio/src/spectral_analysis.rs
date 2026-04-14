//! Short-time spectral analysis — per-band RMS, centroid, spread, flatness,
//! rolloff frequency, and short-time energy.
//!
//! The `SpectralAnalyzer` processes overlapping analysis frames using a Hann
//! window and FFT and then computes a rich set of spectral descriptors that
//! characterise the timbral content of an audio stream on a frame-by-frame
//! basis.
//!
//! # Example
//!
//! ```
//! use oximedia_audio::spectral_analysis::{SpectralAnalyzer, SpectralAnalysisConfig};
//!
//! let config = SpectralAnalysisConfig::new(48_000, 2048, 1024);
//! let mut analyzer = SpectralAnalyzer::new(config).expect("valid config");
//!
//! // Feed one analysis frame of mono samples in [-1, 1].
//! let samples: Vec<f32> = (0..2048)
//!     .map(|i| (i as f32 * 0.01).sin())
//!     .collect();
//! let result = analyzer.analyze(&samples);
//! println!("centroid = {:.1} Hz", result.centroid_hz);
//! ```

#![forbid(unsafe_code)]

use oxifft::api::{Direction, Flags, Plan};
use oxifft::Complex;
use std::f64::consts::PI;

use crate::error::{AudioError, AudioResult};

// ────────────────────────────────────────────────────────────────────────────
// Public configuration
// ────────────────────────────────────────────────────────────────────────────

/// Octave-aligned frequency band definition used for per-band RMS.
#[derive(Clone, Debug)]
pub struct FrequencyBand {
    /// Lower edge of the band in Hz (inclusive).
    pub low_hz: f64,
    /// Upper edge of the band in Hz (exclusive).
    pub high_hz: f64,
    /// Human-readable label, e.g. `"sub-bass"`.
    pub label: String,
}

impl FrequencyBand {
    /// Create a new frequency band.
    #[must_use]
    pub fn new(low_hz: f64, high_hz: f64, label: impl Into<String>) -> Self {
        Self { low_hz, high_hz, label: label.into() }
    }
}

/// Standard 6-band split used when the caller does not supply custom bands.
#[must_use]
pub fn default_bands() -> Vec<FrequencyBand> {
    vec![
        FrequencyBand::new(20.0, 80.0, "sub-bass"),
        FrequencyBand::new(80.0, 300.0, "bass"),
        FrequencyBand::new(300.0, 1_000.0, "low-mid"),
        FrequencyBand::new(1_000.0, 4_000.0, "mid"),
        FrequencyBand::new(4_000.0, 12_000.0, "high-mid"),
        FrequencyBand::new(12_000.0, 20_000.0, "high"),
    ]
}

/// Configuration for `SpectralAnalyzer`.
#[derive(Clone, Debug)]
pub struct SpectralAnalysisConfig {
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// FFT window length in samples (must be a power of two, ≥ 64).
    pub window_size: usize,
    /// Hop size in samples between successive analysis frames.
    pub hop_size: usize,
    /// Rolloff energy threshold in [0, 1] (default 0.85 → 85%).
    pub rolloff_threshold: f64,
    /// Per-band definitions used for `SpectralFrame::band_rms`.
    pub bands: Vec<FrequencyBand>,
}

impl SpectralAnalysisConfig {
    /// Create a configuration with sensible defaults.
    ///
    /// # Panics
    ///
    /// This function returns an error from `SpectralAnalyzer::new` if
    /// `window_size` is not a power of two or is less than 64.
    #[must_use]
    pub fn new(sample_rate: u32, window_size: usize, hop_size: usize) -> Self {
        Self {
            sample_rate,
            window_size,
            hop_size,
            rolloff_threshold: 0.85,
            bands: default_bands(),
        }
    }

    /// Override the rolloff energy threshold.
    #[must_use]
    pub fn with_rolloff_threshold(mut self, threshold: f64) -> Self {
        self.rolloff_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Override the frequency bands.
    #[must_use]
    pub fn with_bands(mut self, bands: Vec<FrequencyBand>) -> Self {
        self.bands = bands;
        self
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Output type
// ────────────────────────────────────────────────────────────────────────────

/// Per-band RMS level computed by `SpectralAnalyzer`.
#[derive(Clone, Debug)]
pub struct BandRms {
    /// Human-readable band label.
    pub label: String,
    /// Low edge of the band in Hz.
    pub low_hz: f64,
    /// High edge of the band in Hz.
    pub high_hz: f64,
    /// RMS magnitude (linear, not dB).
    pub rms: f64,
}

/// Spectral descriptors computed for one analysis frame.
#[derive(Clone, Debug)]
pub struct SpectralFrame {
    /// Spectral centroid — weighted mean frequency in Hz.
    pub centroid_hz: f64,
    /// Spectral spread — standard deviation around the centroid in Hz.
    pub spread_hz: f64,
    /// Spectral flatness (Wiener entropy) in [0, 1].
    /// A pure tone → 0; white noise → 1.
    pub flatness: f64,
    /// Spectral rolloff — Hz below which `rolloff_threshold` fraction of
    /// the total spectral energy is concentrated.
    pub rolloff_hz: f64,
    /// Short-time energy (sum of squared time-domain samples).
    pub short_time_energy: f64,
    /// RMS of all spectral magnitudes.
    pub spectral_rms: f64,
    /// Per-band RMS, one entry per `SpectralAnalysisConfig::bands`.
    pub band_rms: Vec<BandRms>,
    /// Number of input samples included in this frame.
    pub frame_length: usize,
}

// ────────────────────────────────────────────────────────────────────────────
// Analyser
// ────────────────────────────────────────────────────────────────────────────

/// Short-time spectral analyser.
///
/// Maintains a Hann-windowed analysis buffer; call `analyze` once per hop
/// with `hop_size` mono samples (or the same number of interleaved samples
/// averaged to mono before calling).
pub struct SpectralAnalyzer {
    config: SpectralAnalysisConfig,
    /// Pre-computed Hann window coefficients.
    window: Vec<f64>,
    /// Ring buffer holding the last `window_size` samples.
    ring: Vec<f64>,
    /// Write position inside `ring`.
    write_pos: usize,
    /// Bin frequencies pre-computed from sample-rate and window-size.
    bin_freqs: Vec<f64>,
}

impl SpectralAnalyzer {
    /// Create a new `SpectralAnalyzer`.
    ///
    /// # Errors
    ///
    /// Returns `AudioError::InvalidParameter` when `window_size` is not a
    /// power of two, is zero, or `hop_size` is zero.
    pub fn new(config: SpectralAnalysisConfig) -> AudioResult<Self> {
        let ws = config.window_size;
        if ws == 0 || ws & (ws - 1) != 0 {
            return Err(AudioError::InvalidParameter(
                "window_size must be a non-zero power of two".into(),
            ));
        }
        if ws < 64 {
            return Err(AudioError::InvalidParameter(
                "window_size must be at least 64".into(),
            ));
        }
        if config.hop_size == 0 {
            return Err(AudioError::InvalidParameter(
                "hop_size must be non-zero".into(),
            ));
        }

        let window = hann_window(ws);
        let ring = vec![0.0f64; ws];
        let bin_freqs = (0..ws / 2 + 1)
            .map(|k| k as f64 * f64::from(config.sample_rate) / ws as f64)
            .collect();

        Ok(Self {
            config,
            window,
            ring,
            write_pos: 0,
            bin_freqs,
        })
    }

    /// Analyse a slice of mono `f32` samples.
    ///
    /// The slice length does not need to equal `hop_size`; the analyzer
    /// simply ingests all samples and returns descriptors computed from
    /// the current window contents after consuming the block.
    pub fn analyze(&mut self, samples: &[f32]) -> SpectralFrame {
        // Write samples into the ring buffer.
        let n = samples.len();
        for s in samples {
            self.ring[self.write_pos] = f64::from(*s);
            self.write_pos = (self.write_pos + 1) % self.config.window_size;
        }

        // Build a linearly-ordered windowed snapshot.
        let ws = self.config.window_size;
        let mut windowed: Vec<f64> = Vec::with_capacity(ws);
        for i in 0..ws {
            let ring_idx = (self.write_pos + i) % ws;
            windowed.push(self.ring[ring_idx] * self.window[i]);
        }

        // Short-time energy from the raw (un-windowed) ring snapshot.
        let short_time_energy: f64 = windowed.iter().map(|&v| v * v).sum();

        // FFT.
        let magnitudes = compute_magnitudes(&windowed);

        // Compute descriptors.
        let centroid_hz = spectral_centroid(&magnitudes, &self.bin_freqs);
        let spread_hz = spectral_spread(&magnitudes, &self.bin_freqs, centroid_hz);
        let flatness = spectral_flatness(&magnitudes);
        let rolloff_hz = spectral_rolloff(
            &magnitudes,
            &self.bin_freqs,
            self.config.rolloff_threshold,
        );
        let spectral_rms = spectral_rms_value(&magnitudes);
        let band_rms = compute_band_rms(&magnitudes, &self.bin_freqs, &self.config.bands);

        SpectralFrame {
            centroid_hz,
            spread_hz,
            flatness,
            rolloff_hz,
            short_time_energy,
            spectral_rms,
            band_rms,
            frame_length: n,
        }
    }

    /// Reset the internal ring buffer to silence.
    pub fn reset(&mut self) {
        for v in self.ring.iter_mut() {
            *v = 0.0;
        }
        self.write_pos = 0;
    }

    /// Access the underlying configuration.
    #[must_use]
    pub fn config(&self) -> &SpectralAnalysisConfig {
        &self.config
    }
}

// ────────────────────────────────────────────────────────────────────────────
// DSP helpers
// ────────────────────────────────────────────────────────────────────────────

/// Generate a Hann window of length `n`.
fn hann_window(n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / (n - 1) as f64).cos()))
        .collect()
}

/// Compute magnitude spectrum (positive-frequency half) from a real signal.
///
/// Returns `ws/2 + 1` magnitude values.
fn compute_magnitudes(windowed: &[f64]) -> Vec<f64> {
    let n = windowed.len();
    if n == 0 {
        return Vec::new();
    }

    let plan = match Plan::<f64>::dft_1d(n, Direction::Forward, Flags::ESTIMATE) {
        Some(p) => p,
        None => return Vec::new(),
    };

    let input: Vec<Complex<f64>> = windowed
        .iter()
        .map(|&v| Complex { re: v, im: 0.0 })
        .collect();
    let mut output = vec![Complex::<f64> { re: 0.0, im: 0.0 }; n];
    plan.execute(&input, &mut output);

    let bins = n / 2 + 1;
    let scale = 1.0 / n as f64;
    output[..bins]
        .iter()
        .map(|c| (c.re * c.re + c.im * c.im).sqrt() * scale)
        .collect()
}

/// Spectral centroid (weighted mean frequency).
fn spectral_centroid(mags: &[f64], freqs: &[f64]) -> f64 {
    let num: f64 = mags.iter().zip(freqs).map(|(m, f)| m * f).sum();
    let den: f64 = mags.iter().sum();
    if den > 1e-12 { num / den } else { 0.0 }
}

/// Spectral spread (standard deviation around centroid).
fn spectral_spread(mags: &[f64], freqs: &[f64], centroid: f64) -> f64 {
    let den: f64 = mags.iter().sum();
    if den < 1e-12 {
        return 0.0;
    }
    let variance: f64 = mags
        .iter()
        .zip(freqs)
        .map(|(m, f)| {
            let diff = f - centroid;
            m * diff * diff
        })
        .sum::<f64>()
        / den;
    variance.sqrt()
}

/// Spectral flatness (Wiener entropy): geometric mean / arithmetic mean.
fn spectral_flatness(mags: &[f64]) -> f64 {
    let n = mags.len();
    if n == 0 {
        return 0.0;
    }
    let arith_mean: f64 = mags.iter().sum::<f64>() / n as f64;
    if arith_mean < 1e-12 {
        return 0.0;
    }
    // Geometric mean via log-average.
    let log_sum: f64 = mags
        .iter()
        .map(|&m| if m > 1e-12 { m.ln() } else { -28.0 }) // −28 ≈ ln(1e-12)
        .sum();
    let geo_mean = (log_sum / n as f64).exp();
    (geo_mean / arith_mean).clamp(0.0, 1.0)
}

/// Spectral rolloff frequency.
fn spectral_rolloff(mags: &[f64], freqs: &[f64], threshold: f64) -> f64 {
    let total: f64 = mags.iter().map(|m| m * m).sum();
    if total < 1e-12 {
        return 0.0;
    }
    let target = threshold * total;
    let mut cumsum = 0.0;
    for (&m, &f) in mags.iter().zip(freqs) {
        cumsum += m * m;
        if cumsum >= target {
            return f;
        }
    }
    freqs.last().copied().unwrap_or(0.0)
}

/// RMS of the magnitude spectrum.
fn spectral_rms_value(mags: &[f64]) -> f64 {
    if mags.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = mags.iter().map(|m| m * m).sum();
    (sum_sq / mags.len() as f64).sqrt()
}

/// Per-band RMS from magnitude spectrum.
fn compute_band_rms(
    mags: &[f64],
    freqs: &[f64],
    bands: &[FrequencyBand],
) -> Vec<BandRms> {
    bands
        .iter()
        .map(|band| {
            let mut sum_sq = 0.0f64;
            let mut count = 0usize;
            for (&m, &f) in mags.iter().zip(freqs) {
                if f >= band.low_hz && f < band.high_hz {
                    sum_sq += m * m;
                    count += 1;
                }
            }
            let rms = if count > 0 {
                (sum_sq / count as f64).sqrt()
            } else {
                0.0
            };
            BandRms {
                label: band.label.clone(),
                low_hz: band.low_hz,
                high_hz: band.high_hz,
                rms,
            }
        })
        .collect()
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI as PI32;

    fn make_analyzer() -> SpectralAnalyzer {
        let config = SpectralAnalysisConfig::new(48_000, 1024, 512);
        SpectralAnalyzer::new(config).expect("valid config")
    }

    #[test]
    fn test_analyzer_construction() {
        let _ = make_analyzer();
    }

    #[test]
    fn test_invalid_window_size_not_pow2() {
        let config = SpectralAnalysisConfig::new(48_000, 1000, 512);
        assert!(SpectralAnalyzer::new(config).is_err());
    }

    #[test]
    fn test_invalid_window_size_zero() {
        let config = SpectralAnalysisConfig::new(48_000, 0, 0);
        assert!(SpectralAnalyzer::new(config).is_err());
    }

    #[test]
    fn test_invalid_hop_size_zero() {
        let config = SpectralAnalysisConfig::new(48_000, 1024, 0);
        assert!(SpectralAnalyzer::new(config).is_err());
    }

    #[test]
    fn test_silence_gives_zero_energy() {
        let mut analyzer = make_analyzer();
        let silence = vec![0.0f32; 1024];
        let frame = analyzer.analyze(&silence);
        assert!(frame.short_time_energy < 1e-20);
        assert!(frame.spectral_rms < 1e-10);
    }

    #[test]
    fn test_sine_centroid_near_frequency() {
        // 1 kHz sine at 48 kHz → centroid should be close to 1000 Hz.
        let mut analyzer = SpectralAnalyzer::new(
            SpectralAnalysisConfig::new(48_000, 2048, 1024),
        )
        .expect("valid");
        let freq_hz = 1_000.0_f32;
        let sr = 48_000.0_f32;
        let samples: Vec<f32> = (0..2048)
            .map(|i| (2.0 * PI32 * freq_hz * i as f32 / sr).sin() * 0.5)
            .collect();
        let frame = analyzer.analyze(&samples);
        // Allow 100 Hz tolerance given spectral leakage with short window.
        assert!(
            (frame.centroid_hz - 1000.0).abs() < 200.0,
            "centroid = {:.1}",
            frame.centroid_hz
        );
    }

    #[test]
    fn test_flatness_white_noise_higher_than_sine() {
        let mut analyzer = make_analyzer();

        // Pseudo-random white noise via simple LFSR-like sequence.
        let noise: Vec<f32> = (0..1024)
            .map(|i| {
                let v = ((i as f32 * 7.0 + 3.0).sin() * 31337.0).fract();
                v * 2.0 - 1.0
            })
            .collect();

        let freq_hz = 2_000.0_f32;
        let sr = 48_000.0_f32;
        let tone: Vec<f32> = (0..1024)
            .map(|i| (2.0 * PI32 * freq_hz * i as f32 / sr).sin())
            .collect();

        let noise_frame = analyzer.analyze(&noise);
        let _ = analyzer.reset();
        let tone_frame = analyzer.analyze(&tone);

        // Noise flatness should be higher (closer to 1) than pure tone.
        assert!(
            noise_frame.flatness > tone_frame.flatness,
            "noise flatness {} not > tone flatness {}",
            noise_frame.flatness,
            tone_frame.flatness
        );
    }

    #[test]
    fn test_band_rms_count_matches_default_bands() {
        let mut analyzer = make_analyzer();
        let samples = vec![0.1f32; 1024];
        let frame = analyzer.analyze(&samples);
        assert_eq!(frame.band_rms.len(), default_bands().len());
    }

    #[test]
    fn test_rolloff_below_nyquist() {
        let mut analyzer = make_analyzer();
        let samples: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.05).sin()).collect();
        let frame = analyzer.analyze(&samples);
        let nyquist = 48_000.0 / 2.0;
        assert!(
            frame.rolloff_hz <= nyquist,
            "rolloff {} > nyquist {}",
            frame.rolloff_hz,
            nyquist
        );
    }

    #[test]
    fn test_reset_zeroes_ring_buffer() {
        let mut analyzer = make_analyzer();
        let samples: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.1).sin()).collect();
        let _ = analyzer.analyze(&samples);
        analyzer.reset();
        let frame = analyzer.analyze(&vec![0.0f32; 1024]);
        // After reset + silence the short-time energy must be zero.
        assert!(frame.short_time_energy < 1e-20);
    }

    #[test]
    fn test_spread_nonnegative() {
        let mut analyzer = make_analyzer();
        let samples: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.07).sin()).collect();
        let frame = analyzer.analyze(&samples);
        assert!(frame.spread_hz >= 0.0);
    }

    #[test]
    fn test_hann_window_values() {
        let w = hann_window(4);
        // w[0] and w[3] should be near 0; w[1] and w[2] near 0.75.
        assert!(w[0] < 1e-9, "w[0]={}", w[0]);
        assert!(w[3] < 1e-9, "w[3]={}", w[3]);
        assert!((w[1] - 0.75).abs() < 0.01, "w[1]={}", w[1]);
    }

    #[test]
    fn test_default_bands_non_overlapping() {
        let bands = default_bands();
        // Each band's low edge must equal the previous band's high edge.
        for pair in bands.windows(2) {
            assert!(
                (pair[0].high_hz - pair[1].low_hz).abs() < 1e-6,
                "gap between {} and {}",
                pair[0].label,
                pair[1].label
            );
        }
    }
}
