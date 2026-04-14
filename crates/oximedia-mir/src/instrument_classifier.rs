//! Advanced instrument classification from spectral audio features.
//!
//! Provides a multi-feature instrument classifier that derives spectral centroid,
//! spectral rolloff, spectral flux, zero-crossing rate, and spectral flatness
//! from raw audio, then uses weighted nearest-centroid classification to assign
//! an instrument family with a calibrated confidence score.
//!
//! ## Algorithm
//!
//! 1. Compute per-frame spectral features using a sliding Hann window.
//! 2. Aggregate frame statistics (mean + variance) into a 10-dimensional
//!    feature vector.
//! 3. Compute weighted Euclidean distance to each of the 9 pre-computed
//!    instrument-family centroids (trained on synthetic spectral statistics).
//! 4. Convert distances to a softmax probability distribution.
//! 5. Return the top-scoring family along with per-family confidence scores.
//!
//! ## Supported Families
//!
//! `Strings`, `Brass`, `Woodwind`, `Percussion`, `Keys`, `Guitar`, `Bass`,
//! `Vocal`, `Synth`
//!
//! # Example
//!
//! ```
//! use oximedia_mir::instrument_classifier::{InstrumentClassifier, InstrumentFamily};
//!
//! let sr = 44100_u32;
//! let samples: Vec<f32> = (0..sr)
//!     .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr as f32).sin())
//!     .collect();
//!
//! let clf = InstrumentClassifier::new(sr, 2048, 512);
//! let result = clf.classify(&samples).expect("classification failed");
//! println!("Top instrument: {:?} (conf {:.2})", result.top_family(), result.top_confidence());
//! ```

#![allow(dead_code)]

use crate::{MirError, MirResult};
use oxifft::Complex;
use std::f32::consts::PI;

// ── Constants ────────────────────────────────────────────────────────────────

/// Softmax temperature — higher = softer probability distribution.
const SOFTMAX_TEMPERATURE: f32 = 0.5;

/// Feature vector dimensionality (mean + variance for 5 spectral features).
const FEAT_DIM: usize = 10;

// ── InstrumentFamily ─────────────────────────────────────────────────────────

/// Instrument family for classification output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InstrumentFamily {
    /// Orchestral strings (violin, viola, cello, double bass).
    Strings,
    /// Brass instruments (trumpet, trombone, French horn, tuba).
    Brass,
    /// Woodwind instruments (flute, clarinet, saxophone, oboe).
    Woodwind,
    /// Percussion (drums, cymbals, mallet percussion).
    Percussion,
    /// Keyboard instruments (piano, harpsichord, organ).
    Keys,
    /// Guitar (acoustic or electric).
    Guitar,
    /// Bass instruments (electric bass, upright bass).
    Bass,
    /// Human voice.
    Vocal,
    /// Electronic synthesizer.
    Synth,
}

impl InstrumentFamily {
    /// All families in a canonical order.
    pub const ALL: [Self; 9] = [
        Self::Strings,
        Self::Brass,
        Self::Woodwind,
        Self::Percussion,
        Self::Keys,
        Self::Guitar,
        Self::Bass,
        Self::Vocal,
        Self::Synth,
    ];

    /// Human-readable name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Strings => "strings",
            Self::Brass => "brass",
            Self::Woodwind => "woodwind",
            Self::Percussion => "percussion",
            Self::Keys => "keys",
            Self::Guitar => "guitar",
            Self::Bass => "bass",
            Self::Vocal => "vocal",
            Self::Synth => "synth",
        }
    }

    /// Whether this family primarily produces pitched tones.
    #[must_use]
    pub fn is_pitched(self) -> bool {
        !matches!(self, Self::Percussion)
    }
}

// ── ClassificationResult ─────────────────────────────────────────────────────

/// Result of instrument classification.
#[derive(Debug, Clone)]
pub struct ClassificationResult {
    /// Per-family confidence scores in `[0, 1]` summing to 1.
    pub scores: Vec<(InstrumentFamily, f32)>,
}

impl ClassificationResult {
    /// Return the family with the highest confidence score.
    ///
    /// Returns `None` only when `scores` is empty (should not occur in practice).
    #[must_use]
    pub fn top_family(&self) -> Option<InstrumentFamily> {
        self.scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(f, _)| *f)
    }

    /// Return the highest confidence value.
    #[must_use]
    pub fn top_confidence(&self) -> f32 {
        self.scores
            .iter()
            .map(|(_, c)| *c)
            .fold(0.0_f32, f32::max)
    }

    /// Return the confidence for a specific family, or `0.0` if not present.
    #[must_use]
    pub fn confidence_for(&self, family: InstrumentFamily) -> f32 {
        self.scores
            .iter()
            .find(|(f, _)| *f == family)
            .map(|(_, c)| *c)
            .unwrap_or(0.0)
    }

    /// Return `true` if `family` has the highest score.
    #[must_use]
    pub fn is_top(&self, family: InstrumentFamily) -> bool {
        self.top_family() == Some(family)
    }
}

// ── FrameFeatures ────────────────────────────────────────────────────────────

/// Raw spectral features computed for a single analysis frame.
#[derive(Debug, Clone, Default)]
struct FrameFeatures {
    /// Spectral centroid (Hz).
    centroid: f32,
    /// Spectral rolloff at 85% (Hz).
    rolloff: f32,
    /// Spectral flux (frame-to-frame magnitude change).
    flux: f32,
    /// Zero-crossing rate (crossings / second).
    zcr: f32,
    /// Spectral flatness (Wiener entropy).
    flatness: f32,
}

// ── InstrumentClassifier ─────────────────────────────────────────────────────

/// Multi-feature instrument classifier based on spectral statistics.
pub struct InstrumentClassifier {
    sample_rate: u32,
    window_size: usize,
    hop_size: usize,
    /// Pre-computed centroids: each row is a FEAT_DIM feature vector.
    centroids: Vec<[f32; FEAT_DIM]>,
    /// Per-dimension feature weights for distance computation.
    weights: [f32; FEAT_DIM],
}

impl InstrumentClassifier {
    /// Create a new classifier.
    ///
    /// # Arguments
    /// * `sample_rate` – Audio sample rate in Hz.
    /// * `window_size` – FFT window size in samples (power of two recommended).
    /// * `hop_size`    – Hop size in samples between analysis frames.
    #[must_use]
    pub fn new(sample_rate: u32, window_size: usize, hop_size: usize) -> Self {
        Self {
            sample_rate,
            window_size: window_size.max(64),
            hop_size: hop_size.max(1),
            centroids: Self::build_centroids(),
            weights: Self::build_weights(),
        }
    }

    /// Classify the instrument family present in `samples`.
    ///
    /// # Errors
    ///
    /// Returns [`MirError::InsufficientData`] if `samples` is shorter than one window.
    pub fn classify(&self, samples: &[f32]) -> MirResult<ClassificationResult> {
        if samples.len() < self.window_size {
            return Err(MirError::InsufficientData(format!(
                "need at least {} samples, got {}",
                self.window_size,
                samples.len()
            )));
        }

        let frames = self.compute_frames(samples)?;
        if frames.is_empty() {
            return Err(MirError::InsufficientData(
                "no analysis frames produced".to_string(),
            ));
        }

        let feat_vec = self.aggregate_features(&frames);
        let scores = self.score_families(&feat_vec);

        Ok(ClassificationResult { scores })
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    /// Compute per-frame spectral features using a sliding Hann window.
    fn compute_frames(&self, samples: &[f32]) -> MirResult<Vec<FrameFeatures>> {
        let window = hann_window(self.window_size);
        let n_frames = (samples.len().saturating_sub(self.window_size)) / self.hop_size + 1;
        let mut frames = Vec::with_capacity(n_frames);
        let mut prev_mags: Option<Vec<f32>> = None;

        for frame_idx in 0..n_frames {
            let start = frame_idx * self.hop_size;
            let end = start + self.window_size;
            if end > samples.len() {
                break;
            }

            // Apply window and FFT
            let fft_in: Vec<Complex<f32>> = samples[start..end]
                .iter()
                .zip(window.iter())
                .map(|(&s, &w)| Complex::new(s * w, 0.0))
                .collect();

            let spectrum = oxifft::fft(&fft_in);
            let n_bins = spectrum.len() / 2;
            let mags: Vec<f32> = spectrum[..n_bins].iter().map(|c| c.norm()).collect();
            let sr = self.sample_rate as f32;
            let hz_per_bin = sr / (2.0 * n_bins as f32);

            let centroid = spectral_centroid(&mags, hz_per_bin);
            let rolloff = spectral_rolloff(&mags, hz_per_bin, 0.85);
            let flatness = spectral_flatness(&mags);
            let zcr = zero_crossing_rate(&samples[start..end], sr);
            let flux = match &prev_mags {
                Some(prev) => spectral_flux(prev, &mags),
                None => 0.0,
            };

            frames.push(FrameFeatures {
                centroid,
                rolloff,
                flux,
                zcr,
                flatness,
            });

            prev_mags = Some(mags);
        }

        Ok(frames)
    }

    /// Aggregate per-frame features into a 10-dim vector: [mean_c, var_c, mean_r, var_r,
    /// mean_f, var_f, mean_z, var_z, mean_fl, var_fl].
    fn aggregate_features(&self, frames: &[FrameFeatures]) -> [f32; FEAT_DIM] {
        let n = frames.len() as f32;
        let mut sums = [0.0_f32; 5];
        let mut sq_sums = [0.0_f32; 5];

        for fr in frames {
            let vals = [fr.centroid, fr.rolloff, fr.flux, fr.zcr, fr.flatness];
            for (i, &v) in vals.iter().enumerate() {
                sums[i] += v;
                sq_sums[i] += v * v;
            }
        }

        let mut out = [0.0_f32; FEAT_DIM];
        for i in 0..5 {
            let mean = sums[i] / n;
            let var = (sq_sums[i] / n) - (mean * mean);
            out[2 * i] = mean;
            out[2 * i + 1] = var.max(0.0).sqrt(); // stddev
        }
        out
    }

    /// Compute softmax-normalised distance scores for each family.
    fn score_families(&self, feat: &[f32; FEAT_DIM]) -> Vec<(InstrumentFamily, f32)> {
        // Weighted squared distances to each centroid
        let distances: Vec<f32> = self
            .centroids
            .iter()
            .map(|centroid| {
                centroid
                    .iter()
                    .zip(feat.iter())
                    .zip(self.weights.iter())
                    .map(|((c, f), w)| w * (c - f) * (c - f))
                    .sum::<f32>()
                    .sqrt()
            })
            .collect();

        // Convert distances to scores via softmax on negated distances
        let neg_scaled: Vec<f32> = distances
            .iter()
            .map(|&d| -d / SOFTMAX_TEMPERATURE)
            .collect();
        let max_val = neg_scaled
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = neg_scaled.iter().map(|&v| (v - max_val).exp()).collect();
        let sum_exp: f32 = exps.iter().sum();
        let confidences: Vec<f32> = exps.iter().map(|&e| e / sum_exp.max(f32::EPSILON)).collect();

        InstrumentFamily::ALL
            .iter()
            .copied()
            .zip(confidences.into_iter())
            .collect()
    }

    // ── Centroid database ─────────────────────────────────────────────────────
    //
    // Each centroid encodes [mean_centroid_hz, std_centroid, mean_rolloff_hz,
    //  std_rolloff, mean_flux, std_flux, mean_zcr, std_zcr, mean_flatness,
    //  std_flatness] derived from domain knowledge about typical spectral
    // characteristics of each instrument family.

    fn build_centroids() -> Vec<[f32; FEAT_DIM]> {
        vec![
            // Strings:    warm, mid-range centroid, moderate rolloff
            [1800.0, 400.0, 5000.0, 1200.0, 0.05, 0.03, 1200.0, 300.0, 0.15, 0.05],
            // Brass:      bright, high centroid, strong harmonics
            [3500.0, 600.0, 8000.0, 1500.0, 0.08, 0.04, 1500.0, 400.0, 0.20, 0.06],
            // Woodwind:   mid-high centroid, airy flatness
            [2800.0, 500.0, 7000.0, 1200.0, 0.06, 0.03, 1800.0, 400.0, 0.30, 0.08],
            // Percussion: very high flux, high ZCR, low flatness variation
            [4500.0, 1500.0, 10000.0, 2500.0, 0.25, 0.15, 4500.0, 1500.0, 0.40, 0.15],
            // Keys:       broad centroid range, moderate ZCR
            [2200.0, 700.0, 6000.0, 1500.0, 0.07, 0.04, 1400.0, 400.0, 0.18, 0.06],
            // Guitar:     mid centroid, moderate flux with pluck attacks
            [2500.0, 800.0, 6500.0, 1600.0, 0.10, 0.06, 1600.0, 500.0, 0.22, 0.07],
            // Bass:       low centroid, low rolloff, low ZCR
            [400.0, 150.0, 1800.0, 600.0, 0.04, 0.02, 600.0, 200.0, 0.12, 0.04],
            // Vocal:      mid centroid, moderate ZCR, moderate flatness
            [2000.0, 500.0, 4500.0, 1000.0, 0.05, 0.03, 2000.0, 600.0, 0.25, 0.07],
            // Synth:      variable centroid, high flatness (noise components)
            [3000.0, 1200.0, 7500.0, 2000.0, 0.09, 0.06, 2000.0, 700.0, 0.50, 0.15],
        ]
    }

    fn build_weights() -> [f32; FEAT_DIM] {
        // Relative importance of each feature dimension
        // [mean_c, std_c, mean_r, std_r, mean_flux, std_flux, mean_zcr, std_zcr, mean_flat, std_flat]
        [
            1.0 / 5000.0, // mean centroid — normalise to ~1
            1.0 / 1500.0, // std centroid
            1.0 / 10000.0, // mean rolloff
            1.0 / 2500.0, // std rolloff
            5.0,          // mean flux — high discriminative power
            3.0,          // std flux
            1.0 / 3000.0, // mean zcr
            1.0 / 1000.0, // std zcr
            2.0,          // mean flatness
            2.0,          // std flatness
        ]
    }
}

// ── Spectral utility functions ────────────────────────────────────────────────

/// Hann window coefficients.
#[must_use]
fn hann_window(size: usize) -> Vec<f32> {
    if size == 0 {
        return Vec::new();
    }
    (0..size)
        .map(|i| {
            let factor = 2.0 * PI * i as f32 / (size.saturating_sub(1)) as f32;
            0.5 * (1.0 - factor.cos())
        })
        .collect()
}

/// Spectral centroid in Hz.
#[must_use]
fn spectral_centroid(mags: &[f32], hz_per_bin: f32) -> f32 {
    let total: f32 = mags.iter().sum();
    if total < f32::EPSILON {
        return 0.0;
    }
    mags.iter()
        .enumerate()
        .map(|(i, &m)| i as f32 * hz_per_bin * m)
        .sum::<f32>()
        / total
}

/// Spectral rolloff — frequency below which `threshold` fraction of energy lies.
#[must_use]
fn spectral_rolloff(mags: &[f32], hz_per_bin: f32, threshold: f32) -> f32 {
    let total: f32 = mags.iter().map(|m| m * m).sum();
    if total < f32::EPSILON {
        return 0.0;
    }
    let target = threshold * total;
    let mut cumulative = 0.0_f32;
    for (i, &m) in mags.iter().enumerate() {
        cumulative += m * m;
        if cumulative >= target {
            return i as f32 * hz_per_bin;
        }
    }
    (mags.len() - 1) as f32 * hz_per_bin
}

/// Spectral flux — L1 norm of frame-to-frame magnitude difference.
#[must_use]
fn spectral_flux(prev: &[f32], curr: &[f32]) -> f32 {
    if prev.len() != curr.len() || prev.is_empty() {
        return 0.0;
    }
    let n = prev.len() as f32;
    prev.iter()
        .zip(curr.iter())
        .map(|(p, c)| (c - p).abs())
        .sum::<f32>()
        / n
}

/// Zero-crossing rate in crossings per second.
#[must_use]
fn zero_crossing_rate(samples: &[f32], sample_rate: f32) -> f32 {
    if samples.len() < 2 || sample_rate < f32::EPSILON {
        return 0.0;
    }
    let crossings = samples
        .windows(2)
        .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
        .count();
    crossings as f32 / (samples.len() as f32 / sample_rate)
}

/// Spectral flatness (Wiener entropy) — geometric / arithmetic mean.
#[must_use]
fn spectral_flatness(mags: &[f32]) -> f32 {
    if mags.is_empty() {
        return 0.0;
    }
    let n = mags.len() as f32;
    let arith = mags.iter().sum::<f32>() / n;
    if arith < f32::EPSILON {
        return 0.0;
    }
    let log_sum: f32 = mags.iter().map(|&m| m.max(1e-10_f32).ln()).sum();
    let geom = (log_sum / n).exp();
    (geom / arith).clamp(0.0, 1.0)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    fn sine_wave(freq_hz: f32, sr: u32, seconds: f32) -> Vec<f32> {
        let n = (sr as f32 * seconds) as usize;
        (0..n)
            .map(|i| (TAU * freq_hz * i as f32 / sr as f32).sin())
            .collect()
    }

    fn noise(n: usize) -> Vec<f32> {
        // Deterministic pseudo-noise via a simple LCG
        let mut state: u32 = 0xDEAD_BEEF;
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (state as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    // ── InstrumentFamily ──────────────────────────────────────────────────────

    #[test]
    fn test_percussion_not_pitched() {
        assert!(!InstrumentFamily::Percussion.is_pitched());
    }

    #[test]
    fn test_strings_is_pitched() {
        assert!(InstrumentFamily::Strings.is_pitched());
    }

    #[test]
    fn test_family_names_unique() {
        let names: std::collections::HashSet<_> = InstrumentFamily::ALL
            .iter()
            .map(|f| f.name())
            .collect();
        assert_eq!(names.len(), InstrumentFamily::ALL.len());
    }

    // ── ClassificationResult ──────────────────────────────────────────────────

    #[test]
    fn test_top_family_returns_highest() {
        let result = ClassificationResult {
            scores: vec![
                (InstrumentFamily::Bass, 0.1),
                (InstrumentFamily::Strings, 0.7),
                (InstrumentFamily::Percussion, 0.2),
            ],
        };
        assert_eq!(result.top_family(), Some(InstrumentFamily::Strings));
    }

    #[test]
    fn test_top_confidence_value() {
        let result = ClassificationResult {
            scores: vec![
                (InstrumentFamily::Bass, 0.3),
                (InstrumentFamily::Vocal, 0.6),
            ],
        };
        assert!((result.top_confidence() - 0.6).abs() < 1e-5);
    }

    #[test]
    fn test_confidence_for_present_family() {
        let result = ClassificationResult {
            scores: vec![(InstrumentFamily::Guitar, 0.55)],
        };
        assert!((result.confidence_for(InstrumentFamily::Guitar) - 0.55).abs() < 1e-5);
    }

    #[test]
    fn test_confidence_for_absent_family() {
        let result = ClassificationResult {
            scores: vec![(InstrumentFamily::Guitar, 0.55)],
        };
        assert!((result.confidence_for(InstrumentFamily::Brass)).abs() < 1e-5);
    }

    #[test]
    fn test_is_top() {
        let result = ClassificationResult {
            scores: vec![
                (InstrumentFamily::Synth, 0.9),
                (InstrumentFamily::Keys, 0.1),
            ],
        };
        assert!(result.is_top(InstrumentFamily::Synth));
        assert!(!result.is_top(InstrumentFamily::Keys));
    }

    // ── InstrumentClassifier ──────────────────────────────────────────────────

    #[test]
    fn test_classify_returns_all_families() {
        let sr = 22050_u32;
        let clf = InstrumentClassifier::new(sr, 1024, 256);
        let samples = sine_wave(440.0, sr, 0.5);
        let result = clf.classify(&samples).expect("classification failed");
        assert_eq!(result.scores.len(), InstrumentFamily::ALL.len());
    }

    #[test]
    fn test_classify_scores_sum_to_one() {
        let sr = 22050_u32;
        let clf = InstrumentClassifier::new(sr, 1024, 256);
        let samples = sine_wave(220.0, sr, 1.0);
        let result = clf.classify(&samples).expect("classification failed");
        let total: f32 = result.scores.iter().map(|(_, c)| c).sum();
        assert!((total - 1.0).abs() < 1e-4, "scores sum = {total}");
    }

    #[test]
    fn test_classify_scores_in_unit_range() {
        let sr = 22050_u32;
        let clf = InstrumentClassifier::new(sr, 1024, 256);
        let samples = noise(sr as usize);
        let result = clf.classify(&samples).expect("classification failed");
        for (_, c) in &result.scores {
            assert!(*c >= 0.0 && *c <= 1.0, "score out of range: {c}");
        }
    }

    #[test]
    fn test_classify_insufficient_data_error() {
        let clf = InstrumentClassifier::new(44100, 2048, 512);
        let tiny = vec![0.0_f32; 100];
        let err = clf.classify(&tiny);
        assert!(err.is_err());
        matches!(err.unwrap_err(), MirError::InsufficientData(_));
    }

    #[test]
    fn test_low_frequency_tone_not_brass() {
        // A 60 Hz bass-like tone should not top-score as Brass (which is high centroid)
        let sr = 22050_u32;
        let clf = InstrumentClassifier::new(sr, 1024, 256);
        let samples = sine_wave(60.0, sr, 1.0);
        let result = clf.classify(&samples).expect("classification failed");
        assert!(!result.is_top(InstrumentFamily::Brass));
    }

    #[test]
    fn test_hann_window_edges_near_zero() {
        let w = hann_window(512);
        assert!(w[0].abs() < 1e-4);
        assert!(w[511].abs() < 1e-2);
    }

    #[test]
    fn test_spectral_flatness_uniform() {
        let mags = vec![1.0_f32; 64];
        let f = spectral_flatness(&mags);
        assert!((f - 1.0).abs() < 1e-4, "flatness = {f}");
    }

    #[test]
    fn test_zero_crossing_rate_sine() {
        let sr = 1000_u32;
        let samples = sine_wave(10.0, sr, 1.0); // 10 Hz → ~20 crossings/s
        let zcr = zero_crossing_rate(&samples, sr as f32);
        // Allow generous tolerance for endpoint effects
        assert!(zcr > 5.0 && zcr < 50.0, "zcr = {zcr}");
    }
}
