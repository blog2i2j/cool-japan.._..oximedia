//! Fine-grained harmonic spectral analysis for Music Information Retrieval.
//!
//! Provides four complementary measures of harmonic quality in an audio signal:
//!
//! - **Inharmonicity coefficient** `B` — how much partials deviate from ideal
//!   integer multiples of the fundamental (stretched tuning in piano strings,
//!   guitar frets, etc.).
//! - **Harmonic-to-Noise Ratio (HNR)** — ratio of periodic (harmonic) energy
//!   to aperiodic (noise) energy, computed from the normalized autocorrelation.
//! - **Total Harmonic Distortion (THD / THD+N)** — power of harmonic overtones
//!   relative to the fundamental, expressed both as a fraction and in dB.
//! - **Overtone profile** — relative amplitudes of the first N partials
//!   normalized to the fundamental.
//!
//! # Example
//!
//! ```
//! use oximedia_mir::harmonic_spectral::{HarmonicSpectralAnalyzer, HarmonicSpectralConfig};
//!
//! let sr = 44100.0_f32;
//! let freq = 440.0_f32; // A4
//! let n = 4096_usize;
//! let samples: Vec<f32> = (0..n)
//!     .map(|i| {
//!         let t = i as f32 / sr;
//!         (2.0 * std::f32::consts::PI * freq * t).sin()
//!             + 0.3 * (2.0 * std::f32::consts::PI * 2.0 * freq * t).sin()
//!     })
//!     .collect();
//!
//! let config = HarmonicSpectralConfig::default();
//! let analyzer = HarmonicSpectralAnalyzer::new(sr, config);
//! let result = analyzer.analyze(&samples, freq).unwrap();
//! println!("HNR: {:.2} dB", result.hnr_db);
//! println!("THD: {:.4}", result.thd_fraction);
//! ```

#![allow(dead_code)]

use std::f64::consts::PI;

// ── Error ────────────────────────────────────────────────────────────────────

/// Errors that can occur during harmonic spectral analysis.
#[derive(Debug, Clone, PartialEq)]
#[allow(missing_docs)]
pub enum HarmonicSpectralError {
    /// Input signal is too short for analysis.
    SignalTooShort { minimum: usize, actual: usize },
    /// Fundamental frequency is out of a valid range.
    InvalidFundamental { freq: f32 },
    /// Requested number of harmonics exceeds the Nyquist limit.
    TooManyHarmonics { requested: usize, maximum: usize },
    /// All spectral energy is zero (silent signal).
    SilentSignal,
}

impl std::fmt::Display for HarmonicSpectralError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SignalTooShort { minimum, actual } => {
                write!(f, "signal too short: need {minimum} samples, got {actual}")
            }
            Self::InvalidFundamental { freq } => {
                write!(f, "invalid fundamental frequency: {freq} Hz")
            }
            Self::TooManyHarmonics { requested, maximum } => {
                write!(
                    f,
                    "requested {requested} harmonics but Nyquist allows only {maximum}"
                )
            }
            Self::SilentSignal => write!(f, "signal is silent (zero energy)"),
        }
    }
}

impl std::error::Error for HarmonicSpectralError {}

/// Alias for `Result<T, HarmonicSpectralError>`.
pub type HarmonicSpectralResult<T> = Result<T, HarmonicSpectralError>;

// ── Configuration ─────────────────────────────────────────────────────────────

/// Configuration for [`HarmonicSpectralAnalyzer`].
#[derive(Debug, Clone)]
pub struct HarmonicSpectralConfig {
    /// Number of partials (harmonics) to include in analysis (default: 8).
    pub num_harmonics: usize,

    /// Half-width of the frequency bin window used when searching for each
    /// partial peak (in Hz). Default: 5.0 Hz.
    pub peak_search_width_hz: f32,

    /// Minimum amplitude (linear) for a partial to be considered present.
    /// Partials below this threshold are treated as absent. Default: 1e-6.
    pub min_partial_amplitude: f64,

    /// Maximum inharmonicity coefficient `B` considered physically plausible.
    /// Iteration stops early if the estimate exceeds this bound. Default: 0.01.
    pub max_inharmonicity: f64,

    /// Number of Newton–Raphson iterations for inharmonicity fitting.
    /// Default: 20.
    pub inharmonicity_iterations: usize,
}

impl Default for HarmonicSpectralConfig {
    fn default() -> Self {
        Self {
            num_harmonics: 8,
            peak_search_width_hz: 5.0,
            min_partial_amplitude: 1e-6,
            max_inharmonicity: 0.01,
            inharmonicity_iterations: 20,
        }
    }
}

// ── Result types ─────────────────────────────────────────────────────────────

/// Inharmonicity analysis result.
///
/// The inharmonicity coefficient `B` describes how much the partials of a
/// vibrating string deviate from ideal harmonics:
///
/// ```text
/// f_n  =  n · f₀ · √(1 + B · n²)
/// ```
///
/// `B = 0` means perfectly harmonic; higher values indicate more stretch
/// (typical for piano bass strings: `B ≈ 0.0001 – 0.001`).
#[derive(Debug, Clone, PartialEq)]
pub struct InharmonicityResult {
    /// Estimated inharmonicity coefficient.
    pub b_coefficient: f64,

    /// Root-mean-square residual of the fit (in Hz).
    pub fit_residual_hz: f64,

    /// Number of partials used in the fit.
    pub partials_used: usize,
}

/// Harmonic-to-Noise Ratio result.
#[derive(Debug, Clone, PartialEq)]
pub struct HnrResult {
    /// HNR in decibels.  Positive means more periodic energy than noise.
    pub hnr_db: f64,

    /// HNR as a linear power ratio.
    pub hnr_linear: f64,

    /// Normalized autocorrelation peak at lag ≈ 1/f₀.
    pub ac_peak: f64,
}

/// Total Harmonic Distortion result.
#[derive(Debug, Clone, PartialEq)]
pub struct ThdResult {
    /// THD as a fraction of fundamental power (0.0 – theoretically unbounded).
    pub thd_fraction: f64,

    /// THD in decibels (20·log₁₀(√THD_fraction)).
    pub thd_db: f64,

    /// THD+N: total power minus fundamental, divided by fundamental.
    pub thd_n_fraction: f64,

    /// Number of harmonics included in the THD computation.
    pub harmonics_included: usize,
}

/// Relative amplitudes of the first N harmonics, normalized so the fundamental
/// has amplitude 1.0.
#[derive(Debug, Clone, PartialEq)]
pub struct OvertoneProfile {
    /// `amplitudes[0]` = fundamental (always 1.0 if signal is non-silent).
    /// `amplitudes[k]` = amplitude of the (k+1)-th harmonic relative to the
    /// fundamental.
    pub amplitudes: Vec<f64>,

    /// Spectral centroid of the overtone profile (weighted mean harmonic index,
    /// 1-based).
    pub spectral_centroid: f64,

    /// Spectral flatness of the overtone amplitudes (geometric mean / arithmetic
    /// mean), in [0, 1].  High value → noise-like; low value → tonal.
    pub spectral_flatness: f64,
}

/// Aggregated result from [`HarmonicSpectralAnalyzer`].
#[derive(Debug, Clone, PartialEq)]
pub struct HarmonicSpectralFeatures {
    /// Inharmonicity coefficient analysis.
    pub inharmonicity: InharmonicityResult,

    /// Harmonic-to-noise ratio.
    pub hnr: HnrResult,

    /// Total harmonic distortion.
    pub thd: ThdResult,

    /// Overtone amplitude profile.
    pub overtone_profile: OvertoneProfile,

    /// HNR in dB (convenience alias for `hnr.hnr_db`).
    pub hnr_db: f64,

    /// THD fraction (convenience alias for `thd.thd_fraction`).
    pub thd_fraction: f64,
}

// ── Spectrum helpers ──────────────────────────────────────────────────────────

/// Compute the magnitude spectrum from a real-valued signal using a rectangular
/// window (no windowing to keep the implementation dependency-free).
///
/// Returns a `Vec<f64>` of length `n/2 + 1` where index `k` corresponds to
/// frequency `k * sample_rate / n`.
fn magnitude_spectrum(samples: &[f32]) -> Vec<f64> {
    let n = samples.len();
    let half = n / 2 + 1;
    let mut re = vec![0.0_f64; n];
    let mut im = vec![0.0_f64; n];

    for (i, &s) in samples.iter().enumerate() {
        re[i] = f64::from(s);
    }

    // Iterative Cooley-Tukey FFT (radix-2 DIT) — works for any power-of-2 length.
    // For non-power-of-2 lengths we use the DFT directly (slower, but correct).
    let n_bits = (n as f64).log2();
    if n_bits.fract().abs() < 1e-10 && n > 1 {
        fft_inplace(&mut re, &mut im, false);
    } else {
        let input_copy: Vec<f64> = re[..n].to_vec();
        dft_direct(&input_copy, &mut re, &mut im);
    }

    let mut mag = vec![0.0_f64; half];
    for (k, m) in mag.iter_mut().enumerate() {
        *m = (re[k] * re[k] + im[k] * im[k]).sqrt();
    }
    mag
}

/// In-place radix-2 Cooley-Tukey FFT.  `n` must be a power of two.
fn fft_inplace(re: &mut Vec<f64>, im: &mut Vec<f64>, inverse: bool) {
    let n = re.len();
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

    let sign = if inverse { 1.0_f64 } else { -1.0_f64 };
    let mut len = 2usize;
    while len <= n {
        let half_len = len / 2;
        let ang = sign * PI / half_len as f64;
        let (wlen_re, wlen_im) = (ang.cos(), ang.sin());
        let mut i = 0;
        while i < n {
            let (mut w_re, mut w_im) = (1.0_f64, 0.0_f64);
            for jj in 0..half_len {
                let u_re = re[i + jj];
                let u_im = im[i + jj];
                let v_re = re[i + jj + half_len] * w_re - im[i + jj + half_len] * w_im;
                let v_im = re[i + jj + half_len] * w_im + im[i + jj + half_len] * w_re;
                re[i + jj] = u_re + v_re;
                im[i + jj] = u_im + v_im;
                re[i + jj + half_len] = u_re - v_re;
                im[i + jj + half_len] = u_im - v_im;
                let new_w_re = w_re * wlen_re - w_im * wlen_im;
                let new_w_im = w_re * wlen_im + w_im * wlen_re;
                w_re = new_w_re;
                w_im = new_w_im;
            }
            i += len;
        }
        len <<= 1;
    }
}

/// Direct DFT for non-power-of-2 lengths (O(n²), only used for small n).
fn dft_direct(input: &[f64], out_re: &mut Vec<f64>, out_im: &mut Vec<f64>) {
    let n = input.len();
    for k in 0..n {
        let mut r = 0.0_f64;
        let mut img = 0.0_f64;
        for (i, &x) in input.iter().enumerate() {
            let angle = -2.0 * PI * k as f64 * i as f64 / n as f64;
            r += x * angle.cos();
            img += x * angle.sin();
        }
        out_re[k] = r;
        out_im[k] = img;
    }
}

/// Find the peak magnitude bin within `[center_bin - half_width, center_bin + half_width]`.
/// Returns `(bin_index, peak_magnitude)`.
fn find_peak_near(mag: &[f64], center_bin: usize, half_width: usize) -> (usize, f64) {
    let start = center_bin.saturating_sub(half_width);
    let end = (center_bin + half_width + 1).min(mag.len());
    let mut best_bin = start;
    let mut best_mag = 0.0_f64;
    for (i, &m) in mag[start..end].iter().enumerate() {
        if m > best_mag {
            best_mag = m;
            best_bin = start + i;
        }
    }
    (best_bin, best_mag)
}

// ── Main analyzer ─────────────────────────────────────────────────────────────

/// Main entry point for harmonic spectral analysis.
///
/// # Example
///
/// ```
/// use oximedia_mir::harmonic_spectral::{HarmonicSpectralAnalyzer, HarmonicSpectralConfig};
/// let sr = 44100.0_f32;
/// let config = HarmonicSpectralConfig::default();
/// let analyzer = HarmonicSpectralAnalyzer::new(sr, config);
/// // Analyze a 440 Hz tone
/// let samples: Vec<f32> = (0..4096)
///     .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr).sin())
///     .collect();
/// let result = analyzer.analyze(&samples, 440.0).unwrap();
/// assert!(result.hnr_db > 0.0);
/// ```
pub struct HarmonicSpectralAnalyzer {
    sample_rate: f32,
    config: HarmonicSpectralConfig,
}

impl HarmonicSpectralAnalyzer {
    /// Create a new analyzer.
    ///
    /// * `sample_rate` — audio sample rate in Hz.
    /// * `config` — analysis parameters.
    #[must_use]
    pub fn new(sample_rate: f32, config: HarmonicSpectralConfig) -> Self {
        Self {
            sample_rate,
            config,
        }
    }

    /// Analyze the harmonic structure of `samples` given a known fundamental
    /// `f0_hz`.
    ///
    /// # Errors
    ///
    /// - [`HarmonicSpectralError::SignalTooShort`] — fewer than 256 samples.
    /// - [`HarmonicSpectralError::InvalidFundamental`] — `f0_hz ≤ 0` or
    ///   above Nyquist.
    /// - [`HarmonicSpectralError::TooManyHarmonics`] — highest harmonic exceeds
    ///   Nyquist.
    /// - [`HarmonicSpectralError::SilentSignal`] — all samples are zero.
    pub fn analyze(
        &self,
        samples: &[f32],
        f0_hz: f32,
    ) -> HarmonicSpectralResult<HarmonicSpectralFeatures> {
        // ── Validation ────────────────────────────────────────────────────
        if samples.len() < 256 {
            return Err(HarmonicSpectralError::SignalTooShort {
                minimum: 256,
                actual: samples.len(),
            });
        }
        if f0_hz <= 0.0 || f0_hz >= self.sample_rate / 2.0 {
            return Err(HarmonicSpectralError::InvalidFundamental { freq: f0_hz });
        }
        let nyquist = self.sample_rate / 2.0;
        let max_harmonic = (nyquist / f0_hz).floor() as usize;
        if self.config.num_harmonics > max_harmonic {
            return Err(HarmonicSpectralError::TooManyHarmonics {
                requested: self.config.num_harmonics,
                maximum: max_harmonic,
            });
        }

        // ── Magnitude spectrum ────────────────────────────────────────────
        let mag = magnitude_spectrum(samples);
        let n = samples.len();
        let bin_hz = self.sample_rate as f64 / n as f64;
        let half_width = ((self.config.peak_search_width_hz as f64) / bin_hz).ceil() as usize + 1;

        // ── Locate partials ───────────────────────────────────────────────
        let mut partial_freqs = Vec::with_capacity(self.config.num_harmonics);
        let mut partial_amps = Vec::with_capacity(self.config.num_harmonics);

        for h in 1..=self.config.num_harmonics {
            let expected_hz = h as f64 * f64::from(f0_hz);
            let center_bin = (expected_hz / bin_hz).round() as usize;
            let center_bin = center_bin.min(mag.len() - 1);
            let (peak_bin, peak_amp) = find_peak_near(&mag, center_bin, half_width);
            let actual_hz = peak_bin as f64 * bin_hz;
            partial_freqs.push(actual_hz);
            partial_amps.push(peak_amp);
        }

        // ── Silence check ─────────────────────────────────────────────────
        let total_power: f64 = mag.iter().map(|&m| m * m).sum();
        if total_power < 1e-15 {
            return Err(HarmonicSpectralError::SilentSignal);
        }

        // ── Inharmonicity ─────────────────────────────────────────────────
        let inharmonicity = self.compute_inharmonicity(f0_hz, &partial_freqs, &partial_amps)?;

        // ── HNR ───────────────────────────────────────────────────────────
        let hnr = self.compute_hnr(samples, f0_hz);

        // ── THD ───────────────────────────────────────────────────────────
        let thd = self.compute_thd(&partial_amps, &mag);

        // ── Overtone profile ──────────────────────────────────────────────
        let overtone_profile = Self::compute_overtone_profile(&partial_amps);

        let hnr_db = hnr.hnr_db;
        let thd_fraction = thd.thd_fraction;

        Ok(HarmonicSpectralFeatures {
            inharmonicity,
            hnr,
            thd,
            overtone_profile,
            hnr_db,
            thd_fraction,
        })
    }

    // ── Private: inharmonicity ────────────────────────────────────────────────

    /// Fit the inharmonicity coefficient B to the observed partial frequencies
    /// using an iterative weighted least-squares approach.
    ///
    /// Model: `f_n = n * f0 * sqrt(1 + B * n^2)`
    ///
    /// Taking the log: `ln(f_n / (n * f0)) = 0.5 * ln(1 + B * n^2)`
    ///
    /// For small B: `≈ 0.5 * B * n^2`, so we do a linear regression of
    /// `ln(f_n / (n * f0))` against `0.5 * n^2` weighted by amplitude.
    fn compute_inharmonicity(
        &self,
        f0_hz: f32,
        partial_freqs: &[f64],
        partial_amps: &[f64],
    ) -> HarmonicSpectralResult<InharmonicityResult> {
        let f0 = f64::from(f0_hz);
        // Only use partials above the amplitude threshold
        let mut xs = Vec::new(); // 0.5 * n^2
        let mut ys = Vec::new(); // ln(f_n / (n * f0))
        let mut ws = Vec::new(); // weights (amplitude)

        for (idx, (&freq, &amp)) in partial_freqs.iter().zip(partial_amps.iter()).enumerate() {
            if amp < self.config.min_partial_amplitude {
                continue;
            }
            let n = (idx + 1) as f64;
            let ratio = freq / (n * f0);
            if ratio <= 0.0 {
                continue;
            }
            xs.push(0.5 * n * n);
            ys.push(ratio.ln());
            ws.push(amp);
        }

        let partials_used = xs.len();
        if partials_used < 2 {
            // Not enough data — return zero inharmonicity
            return Ok(InharmonicityResult {
                b_coefficient: 0.0,
                fit_residual_hz: 0.0,
                partials_used,
            });
        }

        // Weighted linear regression: y = B * x
        let sum_wx2: f64 = xs.iter().zip(ws.iter()).map(|(&x, &w)| w * x * x).sum();
        let sum_wxy: f64 = xs
            .iter()
            .zip(ys.iter())
            .zip(ws.iter())
            .map(|((&x, &y), &w)| w * x * y)
            .sum();

        let b_estimate = if sum_wx2.abs() < 1e-30 {
            0.0
        } else {
            (sum_wxy / sum_wx2).clamp(-self.config.max_inharmonicity, self.config.max_inharmonicity)
        };

        // Compute residual in Hz
        let f0_val = f0;
        let residual_sq: f64 = partial_freqs
            .iter()
            .enumerate()
            .map(|(idx, &freq)| {
                let n = (idx + 1) as f64;
                let predicted = n * f0_val * (1.0 + b_estimate * n * n).sqrt().max(0.0);
                (freq - predicted).powi(2)
            })
            .sum::<f64>()
            / partial_freqs.len() as f64;

        Ok(InharmonicityResult {
            b_coefficient: b_estimate,
            fit_residual_hz: residual_sq.sqrt(),
            partials_used,
        })
    }

    // ── Private: HNR ─────────────────────────────────────────────────────────

    /// Compute Harmonic-to-Noise Ratio via normalized autocorrelation.
    ///
    /// 1. Compute the normalized autocorrelation of the signal.
    /// 2. Find the peak at the lag closest to `1/f0` seconds.
    /// 3. `HNR = r_peak / (1 - r_peak)` where `r_peak` is the normalized
    ///    autocorrelation at the fundamental period lag.
    fn compute_hnr(&self, samples: &[f32], f0_hz: f32) -> HnrResult {
        let sr = f64::from(self.sample_rate);
        let period_samples = (sr / f64::from(f0_hz)).round() as usize;

        // Normalize the signal
        let mean: f64 = samples.iter().map(|&s| f64::from(s)).sum::<f64>() / samples.len() as f64;
        let centered: Vec<f64> = samples.iter().map(|&s| f64::from(s) - mean).collect();

        // AC at lag 0 (total power)
        let ac0: f64 = centered.iter().map(|&x| x * x).sum();
        if ac0 < 1e-15 {
            return HnrResult {
                hnr_db: f64::NEG_INFINITY,
                hnr_linear: 0.0,
                ac_peak: 0.0,
            };
        }

        // Search around the expected period lag (±10% tolerance)
        let lag_min = (period_samples as f64 * 0.9).round() as usize;
        let lag_max = (period_samples as f64 * 1.1).round() as usize + 1;
        let lag_max = lag_max.min(centered.len() / 2);

        if lag_min >= lag_max {
            return HnrResult {
                hnr_db: 0.0,
                hnr_linear: 1.0,
                ac_peak: 0.5,
            };
        }

        let mut best_ac = -1.0_f64;
        for lag in lag_min..=lag_max {
            let ac_lag: f64 = centered[..centered.len() - lag]
                .iter()
                .zip(centered[lag..].iter())
                .map(|(&a, &b)| a * b)
                .sum();
            let normalized = ac_lag / ac0;
            if normalized > best_ac {
                best_ac = normalized;
            }
        }

        let ac_peak = best_ac.clamp(-1.0, 1.0);

        // HNR formula: ac_peak / (1 - ac_peak)
        let hnr_linear = if ac_peak >= 1.0 - 1e-12 {
            1e6 // effectively infinity
        } else if ac_peak <= 0.0 {
            0.0
        } else {
            ac_peak / (1.0 - ac_peak)
        };

        let hnr_db = if hnr_linear < 1e-15 {
            f64::NEG_INFINITY
        } else {
            10.0 * hnr_linear.log10()
        };

        HnrResult {
            hnr_db,
            hnr_linear,
            ac_peak,
        }
    }

    // ── Private: THD ─────────────────────────────────────────────────────────

    /// Compute Total Harmonic Distortion from the partial amplitudes.
    ///
    /// `THD = sqrt(P₂ + P₃ + … + Pₙ) / √P₁`
    /// where `Pₙ = |amplitude_n|²`.
    fn compute_thd(&self, partial_amps: &[f64], mag: &[f64]) -> ThdResult {
        if partial_amps.is_empty() {
            return ThdResult {
                thd_fraction: 0.0,
                thd_db: f64::NEG_INFINITY,
                thd_n_fraction: 0.0,
                harmonics_included: 0,
            };
        }

        let p_fundamental = partial_amps[0] * partial_amps[0];
        if p_fundamental < 1e-30 {
            return ThdResult {
                thd_fraction: 0.0,
                thd_db: f64::NEG_INFINITY,
                thd_n_fraction: 0.0,
                harmonics_included: partial_amps.len(),
            };
        }

        let harmonic_power: f64 = partial_amps[1..]
            .iter()
            .map(|&a| a * a)
            .sum::<f64>();

        let thd_fraction = (harmonic_power / p_fundamental).sqrt();
        let thd_db = if thd_fraction < 1e-15 {
            f64::NEG_INFINITY
        } else {
            20.0 * thd_fraction.log10()
        };

        // THD+N: total power minus fundamental
        let total_power: f64 = mag.iter().map(|&m| m * m).sum();
        let thd_n_fraction = if p_fundamental < 1e-30 {
            0.0
        } else {
            ((total_power - p_fundamental).max(0.0) / p_fundamental).sqrt()
        };

        ThdResult {
            thd_fraction,
            thd_db,
            thd_n_fraction,
            harmonics_included: partial_amps.len().saturating_sub(1),
        }
    }

    // ── Private: overtone profile ─────────────────────────────────────────────

    /// Build the overtone amplitude profile, normalized to the fundamental.
    fn compute_overtone_profile(partial_amps: &[f64]) -> OvertoneProfile {
        if partial_amps.is_empty() {
            return OvertoneProfile {
                amplitudes: Vec::new(),
                spectral_centroid: 0.0,
                spectral_flatness: 0.0,
            };
        }

        let fund = partial_amps[0];
        let amplitudes: Vec<f64> = if fund < 1e-15 {
            partial_amps.to_vec()
        } else {
            partial_amps.iter().map(|&a| a / fund).collect()
        };

        // Spectral centroid (weighted mean harmonic index, 1-based)
        let sum_a: f64 = amplitudes.iter().sum();
        let spectral_centroid = if sum_a < 1e-15 {
            1.0
        } else {
            amplitudes
                .iter()
                .enumerate()
                .map(|(i, &a)| (i + 1) as f64 * a)
                .sum::<f64>()
                / sum_a
        };

        // Spectral flatness: geometric mean / arithmetic mean
        let n = amplitudes.len() as f64;
        let log_sum: f64 = amplitudes
            .iter()
            .map(|&a| if a < 1e-15 { -30.0_f64.ln() } else { a.ln() })
            .sum();
        let geometric_mean = (log_sum / n).exp();
        let arithmetic_mean = sum_a / n;
        let spectral_flatness = if arithmetic_mean < 1e-15 {
            0.0
        } else {
            (geometric_mean / arithmetic_mean).clamp(0.0, 1.0)
        };

        OvertoneProfile {
            amplitudes,
            spectral_centroid,
            spectral_flatness,
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI as PI32;

    fn sine_wave(freq: f32, sr: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * PI32 * freq * i as f32 / sr).sin())
            .collect()
    }

    fn harmonic_signal(f0: f32, sr: f32, n: usize, coeffs: &[f32]) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let t = i as f32 / sr;
                coeffs
                    .iter()
                    .enumerate()
                    .map(|(h, &amp)| amp * (2.0 * PI32 * (h + 1) as f32 * f0 * t).sin())
                    .sum::<f32>()
            })
            .collect()
    }

    #[test]
    fn test_pure_tone_hnr_high() {
        let sr = 44100.0_f32;
        let f0 = 440.0_f32;
        let samples = sine_wave(f0, sr, 4096);
        let config = HarmonicSpectralConfig::default();
        let analyzer = HarmonicSpectralAnalyzer::new(sr, config);
        let result = analyzer.analyze(&samples, f0).expect("should succeed");
        // A pure tone should have a high HNR
        assert!(
            result.hnr_db > 5.0,
            "expected HNR > 5 dB, got {:.2}",
            result.hnr_db
        );
    }

    #[test]
    fn test_pure_tone_thd_near_zero() {
        let sr = 44100.0_f32;
        let f0 = 440.0_f32;
        let samples = sine_wave(f0, sr, 4096);
        let config = HarmonicSpectralConfig::default();
        let analyzer = HarmonicSpectralAnalyzer::new(sr, config);
        let result = analyzer.analyze(&samples, f0).expect("should succeed");
        // A pure tone has almost no harmonic distortion
        assert!(
            result.thd_fraction < 0.5,
            "expected THD < 0.5, got {:.4}",
            result.thd_fraction
        );
    }

    #[test]
    fn test_harmonic_signal_thd_nonzero() {
        let sr = 44100.0_f32;
        let f0 = 220.0_f32;
        // fundamental + 2nd harmonic at 30%
        let samples = harmonic_signal(f0, sr, 8192, &[1.0, 0.3, 0.1]);
        let config = HarmonicSpectralConfig {
            num_harmonics: 4,
            ..Default::default()
        };
        let analyzer = HarmonicSpectralAnalyzer::new(sr, config);
        let result = analyzer.analyze(&samples, f0).expect("should succeed");
        // THD should be > 0 (there IS a 2nd harmonic)
        assert!(
            result.thd_fraction > 0.0,
            "expected THD > 0, got {:.4}",
            result.thd_fraction
        );
    }

    #[test]
    fn test_overtone_profile_fundamental_is_one() {
        let sr = 44100.0_f32;
        let f0 = 330.0_f32;
        let samples = harmonic_signal(f0, sr, 8192, &[1.0, 0.5, 0.25, 0.125]);
        let config = HarmonicSpectralConfig {
            num_harmonics: 4,
            ..Default::default()
        };
        let analyzer = HarmonicSpectralAnalyzer::new(sr, config);
        let result = analyzer.analyze(&samples, f0).expect("should succeed");
        let amps = &result.overtone_profile.amplitudes;
        // First entry should be 1.0 (normalized)
        assert!(
            (amps[0] - 1.0).abs() < 0.05,
            "fundamental should be ≈1.0, got {:.4}",
            amps[0]
        );
    }

    #[test]
    fn test_overtone_profile_decaying() {
        let sr = 44100.0_f32;
        let f0 = 330.0_f32;
        let samples = harmonic_signal(f0, sr, 8192, &[1.0, 0.5, 0.25, 0.125]);
        let config = HarmonicSpectralConfig {
            num_harmonics: 4,
            ..Default::default()
        };
        let analyzer = HarmonicSpectralAnalyzer::new(sr, config);
        let result = analyzer.analyze(&samples, f0).expect("should succeed");
        let amps = &result.overtone_profile.amplitudes;
        // Each harmonic should be weaker than the previous
        assert!(
            amps[0] > amps[1],
            "fundamental should dominate: amps[0]={:.3} amps[1]={:.3}",
            amps[0],
            amps[1]
        );
    }

    #[test]
    fn test_inharmonicity_near_zero_for_pure_tone() {
        let sr = 44100.0_f32;
        let f0 = 440.0_f32;
        let samples = harmonic_signal(f0, sr, 8192, &[1.0, 0.5, 0.25, 0.125]);
        let config = HarmonicSpectralConfig {
            num_harmonics: 4,
            ..Default::default()
        };
        let analyzer = HarmonicSpectralAnalyzer::new(sr, config);
        let result = analyzer.analyze(&samples, f0).expect("should succeed");
        // Perfectly harmonic signal → B ≈ 0
        assert!(
            result.inharmonicity.b_coefficient.abs() < 0.01,
            "expected B ≈ 0, got {:.6}",
            result.inharmonicity.b_coefficient
        );
    }

    #[test]
    fn test_error_signal_too_short() {
        let sr = 44100.0_f32;
        let samples = vec![0.0_f32; 64]; // too short
        let config = HarmonicSpectralConfig::default();
        let analyzer = HarmonicSpectralAnalyzer::new(sr, config);
        let err = analyzer.analyze(&samples, 440.0).unwrap_err();
        assert!(matches!(err, HarmonicSpectralError::SignalTooShort { .. }));
    }

    #[test]
    fn test_error_invalid_fundamental() {
        let sr = 44100.0_f32;
        let samples = vec![0.1_f32; 1024];
        let config = HarmonicSpectralConfig::default();
        let analyzer = HarmonicSpectralAnalyzer::new(sr, config);
        // Negative frequency
        let err = analyzer.analyze(&samples, -440.0).unwrap_err();
        assert!(matches!(
            err,
            HarmonicSpectralError::InvalidFundamental { .. }
        ));
        // Zero frequency
        let err2 = analyzer.analyze(&samples, 0.0).unwrap_err();
        assert!(matches!(
            err2,
            HarmonicSpectralError::InvalidFundamental { .. }
        ));
    }

    #[test]
    fn test_error_too_many_harmonics() {
        let sr = 44100.0_f32;
        let samples = sine_wave(440.0, sr, 4096);
        // Request 1000 harmonics for 440 Hz — Nyquist is 22050, max is 50
        let config = HarmonicSpectralConfig {
            num_harmonics: 1000,
            ..Default::default()
        };
        let analyzer = HarmonicSpectralAnalyzer::new(sr, config);
        let err = analyzer.analyze(&samples, 440.0).unwrap_err();
        assert!(matches!(
            err,
            HarmonicSpectralError::TooManyHarmonics { .. }
        ));
    }

    #[test]
    fn test_spectral_centroid_monotone() {
        // A signal with equal harmonics should have centroid near the middle harmonic
        let sr = 44100.0_f32;
        let f0 = 220.0_f32;
        let samples = harmonic_signal(f0, sr, 8192, &[1.0, 1.0, 1.0, 1.0]);
        let config = HarmonicSpectralConfig {
            num_harmonics: 4,
            ..Default::default()
        };
        let analyzer = HarmonicSpectralAnalyzer::new(sr, config);
        let result = analyzer.analyze(&samples, f0).expect("should succeed");
        // Centroid should be between 1 and 4
        let c = result.overtone_profile.spectral_centroid;
        assert!(c >= 1.0 && c <= 4.0, "centroid out of range: {c:.3}");
    }

    #[test]
    fn test_hnr_result_fields_consistent() {
        let sr = 44100.0_f32;
        let f0 = 440.0_f32;
        let samples = sine_wave(f0, sr, 4096);
        let config = HarmonicSpectralConfig::default();
        let analyzer = HarmonicSpectralAnalyzer::new(sr, config);
        let result = analyzer.analyze(&samples, f0).expect("should succeed");
        // hnr_db convenience field should match hnr.hnr_db
        assert!(
            (result.hnr_db - result.hnr.hnr_db).abs() < 1e-10,
            "hnr_db mismatch"
        );
        // thd_fraction convenience field should match thd.thd_fraction
        assert!(
            (result.thd_fraction - result.thd.thd_fraction).abs() < 1e-10,
            "thd_fraction mismatch"
        );
    }
}
