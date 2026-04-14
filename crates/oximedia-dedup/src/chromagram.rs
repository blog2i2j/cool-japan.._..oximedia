//! Chromagram-based audio feature extraction for music deduplication.
//!
//! A chromagram (also called a chroma feature or pitch class profile) represents
//! the energy distribution across the 12 pitch classes of the Western chromatic
//! scale (C, C#, D, …, B). It is highly robust for music matching because it is
//! largely invariant to:
//! - Timbre / instrument changes (same melody, different instruments)
//! - Transposition (if the transposition amount is tracked separately)
//! - Recording conditions and mild equalization
//!
//! # Algorithm
//!
//! 1. Convert audio to mono and downsample to 22050 Hz.
//! 2. Frame the signal with a Hann window.
//! 3. Compute the magnitude spectrum of each frame using OxiFFT.
//! 4. Map each FFT bin to its corresponding chroma bin (0–11) using the
//!    MIDI pitch formula: `pitch = 12 * log2(freq / 440) + 69`.
//! 5. Accumulate energy per chroma class and L2-normalise.
//! 6. Aggregate chroma frames into a compact fingerprint by computing
//!    per-chroma-class statistics (mean, standard deviation) over all frames.
//!
//! # Usage
//!
//! ```
//! use oximedia_dedup::chromagram::{ChromagramExtractor, ChromaConfig};
//!
//! let config = ChromaConfig::default();
//! let extractor = ChromagramExtractor::new(config);
//!
//! // 1-second, 440 Hz sine wave at 22050 Hz sample rate
//! let samples: Vec<f32> = (0..22050)
//!     .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 22050.0).sin())
//!     .collect();
//!
//! let fingerprint = extractor.fingerprint(&samples, 22050).unwrap();
//! assert_eq!(fingerprint.chroma_means.len(), 12);
//! ```

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use crate::{DedupError, DedupResult};
use oxifft::Complex;

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the chromagram extractor.
#[derive(Debug, Clone)]
pub struct ChromaConfig {
    /// FFT window size (must be a power of two).
    pub fft_size: usize,
    /// Hop length between successive frames (samples).
    pub hop_length: usize,
    /// Target internal sample rate.  Audio is re-sampled to this rate before
    /// analysis.  22050 Hz is standard for chroma analysis.
    pub target_sample_rate: u32,
    /// Reference frequency for A4 (Hz).  Standard is 440.0.
    pub a4_hz: f64,
}

impl Default for ChromaConfig {
    fn default() -> Self {
        Self {
            fft_size: 2048,
            hop_length: 512,
            target_sample_rate: 22050,
            a4_hz: 440.0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Chroma fingerprint
// ─────────────────────────────────────────────────────────────────────────────

/// A compact chroma fingerprint derived from all frames of an audio signal.
///
/// The fingerprint stores per-chroma-class mean and standard deviation across
/// all frames.  Two fingerprints can be compared via [`ChromaFingerprint::similarity`].
#[derive(Debug, Clone, PartialEq)]
pub struct ChromaFingerprint {
    /// Mean energy per chroma class (length 12).
    pub chroma_means: Vec<f64>,
    /// Standard deviation of energy per chroma class (length 12).
    pub chroma_stds: Vec<f64>,
    /// Number of frames analysed.
    pub num_frames: usize,
    /// Sample rate of the source material (before downsampling).
    pub source_sample_rate: u32,
}

impl ChromaFingerprint {
    /// Compute the cosine similarity between two fingerprints in [0.0, 1.0].
    ///
    /// Uses the mean chroma vector for comparison.  Returns 0.0 when either
    /// vector has zero magnitude.
    #[must_use]
    pub fn similarity(&self, other: &Self) -> f64 {
        cosine_similarity(&self.chroma_means, &other.chroma_means)
    }

    /// Compute the combined mean+std similarity as a weighted average.
    ///
    /// `mean_weight` controls the importance of the mean chroma vector
    /// (0.0–1.0); the remainder is given to the standard deviation vector.
    #[must_use]
    pub fn combined_similarity(&self, other: &Self, mean_weight: f64) -> f64 {
        let std_weight = (1.0 - mean_weight).max(0.0);
        let mean_sim = cosine_similarity(&self.chroma_means, &other.chroma_means);
        let std_sim = cosine_similarity(&self.chroma_stds, &other.chroma_stds);
        mean_weight * mean_sim + std_weight * std_sim
    }

    /// Serialize the fingerprint to a compact byte representation.
    ///
    /// Layout: 12 × f32 means || 12 × f32 stds (little-endian), 96 bytes total.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(96);
        for &v in &self.chroma_means {
            out.extend_from_slice(&(v as f32).to_le_bytes());
        }
        for &v in &self.chroma_stds {
            out.extend_from_slice(&(v as f32).to_le_bytes());
        }
        out
    }

    /// Deserialize a fingerprint from the compact byte representation.
    ///
    /// # Errors
    ///
    /// Returns [`DedupError::Audio`] if the byte slice is not exactly 96 bytes.
    pub fn from_bytes(bytes: &[u8], source_sample_rate: u32) -> DedupResult<Self> {
        if bytes.len() != 96 {
            return Err(DedupError::Audio(format!(
                "ChromaFingerprint::from_bytes expected 96 bytes, got {}",
                bytes.len()
            )));
        }
        let read_f32 = |chunk: &[u8]| -> f64 {
            let arr: [u8; 4] = chunk.try_into().unwrap_or([0; 4]);
            f64::from(f32::from_le_bytes(arr))
        };
        let chroma_means = (0..12).map(|i| read_f32(&bytes[i * 4..(i + 1) * 4])).collect();
        let chroma_stds = (0..12)
            .map(|i| read_f32(&bytes[48 + i * 4..48 + (i + 1) * 4]))
            .collect();
        Ok(Self {
            chroma_means,
            chroma_stds,
            num_frames: 0,
            source_sample_rate,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Extractor
// ─────────────────────────────────────────────────────────────────────────────

/// Chromagram-based audio fingerprint extractor.
pub struct ChromagramExtractor {
    config: ChromaConfig,
    /// Pre-computed Hann window of length `fft_size`.
    hann_window: Vec<f64>,
}

impl ChromagramExtractor {
    /// Create a new extractor with the given configuration.
    #[must_use]
    pub fn new(config: ChromaConfig) -> Self {
        let hann_window = hann_window(config.fft_size);
        Self {
            config,
            hann_window,
        }
    }

    /// Compute a [`ChromaFingerprint`] from raw PCM samples.
    ///
    /// `samples` must be mono `f32` at `sample_rate` Hz.  Multi-channel audio
    /// should be downmixed to mono by the caller before passing here.
    ///
    /// # Errors
    ///
    /// Returns [`DedupError::Audio`] if `samples` is empty or `sample_rate` is
    /// zero, or if the FFT size is invalid.
    pub fn fingerprint(&self, samples: &[f32], sample_rate: u32) -> DedupResult<ChromaFingerprint> {
        if samples.is_empty() {
            return Err(DedupError::Audio(
                "chromagram: samples slice is empty".to_string(),
            ));
        }
        if sample_rate == 0 {
            return Err(DedupError::Audio(
                "chromagram: sample_rate must be > 0".to_string(),
            ));
        }
        if self.config.fft_size == 0 || !self.config.fft_size.is_power_of_two() {
            return Err(DedupError::Audio(format!(
                "chromagram: fft_size {} is not a non-zero power of two",
                self.config.fft_size
            )));
        }

        // Resample to target rate if needed (simple linear interpolation).
        let resampled: Vec<f32> = if sample_rate != self.config.target_sample_rate {
            resample_linear(samples, sample_rate, self.config.target_sample_rate)
        } else {
            samples.to_vec()
        };

        let effective_rate = self.config.target_sample_rate;
        let fft_size = self.config.fft_size;
        let hop = self.config.hop_length.max(1);

        // Accumulate per-chroma-class energy across frames.
        // chroma_accum[frame][bin]
        let mut chroma_frames: Vec<[f64; 12]> = Vec::new();

        let mut frame_start = 0usize;
        while frame_start + fft_size <= resampled.len() {
            let frame = &resampled[frame_start..frame_start + fft_size];
            let chroma = self.analyse_frame(frame, effective_rate)?;
            chroma_frames.push(chroma);
            frame_start += hop;
        }

        // Handle the case where the signal is shorter than one FFT frame by
        // zero-padding a single frame.
        if chroma_frames.is_empty() {
            let mut padded = resampled.clone();
            padded.resize(fft_size, 0.0);
            let chroma = self.analyse_frame(&padded, effective_rate)?;
            chroma_frames.push(chroma);
        }

        let num_frames = chroma_frames.len();

        // Compute mean and std per chroma class.
        let mut means = [0.0f64; 12];
        for frame in &chroma_frames {
            for (i, &v) in frame.iter().enumerate() {
                means[i] += v;
            }
        }
        for m in &mut means {
            *m /= num_frames as f64;
        }

        let mut stds = [0.0f64; 12];
        for frame in &chroma_frames {
            for (i, &v) in frame.iter().enumerate() {
                stds[i] += (v - means[i]).powi(2);
            }
        }
        for s in &mut stds {
            *s = (*s / num_frames as f64).sqrt();
        }

        Ok(ChromaFingerprint {
            chroma_means: means.to_vec(),
            chroma_stds: stds.to_vec(),
            num_frames,
            source_sample_rate: sample_rate,
        })
    }

    /// Analyse one windowed FFT frame and return L2-normalised chroma vector.
    fn analyse_frame(
        &self,
        frame: &[f32],
        sample_rate: u32,
    ) -> DedupResult<[f64; 12]> {
        let n = frame.len();

        // Apply Hann window and convert to Complex<f64>.
        let mut spectrum: Vec<Complex<f64>> = frame
            .iter()
            .zip(self.hann_window.iter().take(n))
            .map(|(&s, &w)| Complex::new(f64::from(s) * w, 0.0))
            .collect();

        // Zero-pad to FFT size if the frame is shorter (tail of signal).
        spectrum.resize(self.config.fft_size, Complex::new(0.0, 0.0));

        // Forward FFT via OxiFFT convenience function.
        let spectrum_out = oxifft::fft(&spectrum);

        // Compute magnitude for the positive-frequency half.
        let half = self.config.fft_size / 2 + 1;
        let magnitudes: Vec<f64> = spectrum_out[..half]
            .iter()
            .map(|c| (c.re * c.re + c.im * c.im).sqrt())
            .collect();

        // Map FFT bins to chroma classes.
        let mut chroma = [0.0f64; 12];
        let sr = f64::from(sample_rate);
        let a4 = self.config.a4_hz;

        for (bin, &mag) in magnitudes.iter().enumerate() {
            if mag < f64::EPSILON {
                continue;
            }
            let freq = bin as f64 * sr / self.config.fft_size as f64;
            if freq < 27.5 || freq > 14080.0 {
                // Outside the standard piano range (A0..A9); skip.
                continue;
            }
            // MIDI note: pitch = 12 * log2(freq / a4) + 69
            let midi = 12.0 * (freq / a4).log2() + 69.0;
            // Chroma class = MIDI mod 12, handling negative values.
            let chroma_bin = ((midi.round() as i64).rem_euclid(12)) as usize;
            chroma[chroma_bin] += mag;
        }

        // L2-normalise the chroma vector.
        let norm: f64 = chroma.iter().map(|&v| v * v).sum::<f64>().sqrt();
        if norm > f64::EPSILON {
            for v in &mut chroma {
                *v /= norm;
            }
        }

        Ok(chroma)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute a Hann window of length `n`.
fn hann_window(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (n as f64 - 1.0)).cos())
        })
        .collect()
}

/// Linearly interpolate `samples` from `src_rate` to `dst_rate`.
fn resample_linear(samples: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if src_rate == dst_rate || samples.is_empty() {
        return samples.to_vec();
    }
    let ratio = f64::from(src_rate) / f64::from(dst_rate);
    let new_len = ((samples.len() as f64) / ratio).ceil() as usize;
    let mut out = Vec::with_capacity(new_len);
    for i in 0..new_len {
        let src_pos = i as f64 * ratio;
        let lo = src_pos.floor() as usize;
        let hi = (lo + 1).min(samples.len() - 1);
        let frac = src_pos - lo as f64;
        let v = f64::from(samples[lo]) * (1.0 - frac) + f64::from(samples[hi]) * frac;
        out.push(v as f32);
    }
    out
}

/// Cosine similarity between two equal-length f64 slices.
///
/// Returns 0.0 when either vector has zero magnitude.
fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len(), "cosine_similarity: length mismatch");
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let mag_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if mag_a < f64::EPSILON || mag_b < f64::EPSILON {
        return 0.0;
    }
    (dot / (mag_a * mag_b)).clamp(0.0, 1.0)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    /// Helper: generate a mono sine wave at `freq_hz` for `duration_secs`.
    fn sine_wave(freq_hz: f32, sample_rate: u32, duration_secs: f32) -> Vec<f32> {
        let n = (sample_rate as f32 * duration_secs) as usize;
        (0..n)
            .map(|i| (2.0 * PI * freq_hz * i as f32 / sample_rate as f32).sin())
            .collect()
    }

    #[test]
    fn test_default_config() {
        let cfg = ChromaConfig::default();
        assert_eq!(cfg.fft_size, 2048);
        assert_eq!(cfg.hop_length, 512);
        assert_eq!(cfg.target_sample_rate, 22050);
        assert!((cfg.a4_hz - 440.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_hann_window_length() {
        let w = hann_window(2048);
        assert_eq!(w.len(), 2048);
    }

    #[test]
    fn test_hann_window_endpoints() {
        let w = hann_window(128);
        // First and last samples should be near 0.
        assert!(w[0].abs() < 1e-10);
        assert!(w[127].abs() < 1e-3);
    }

    #[test]
    fn test_fingerprint_returns_12_chroma_bins() {
        let extractor = ChromagramExtractor::new(ChromaConfig::default());
        let samples = sine_wave(440.0, 22050, 1.0);
        let fp = extractor.fingerprint(&samples, 22050).expect("should succeed");
        assert_eq!(fp.chroma_means.len(), 12);
        assert_eq!(fp.chroma_stds.len(), 12);
    }

    #[test]
    fn test_fingerprint_identical_signals_high_similarity() {
        let extractor = ChromagramExtractor::new(ChromaConfig::default());
        let samples = sine_wave(440.0, 22050, 1.0);
        let fp1 = extractor.fingerprint(&samples, 22050).expect("ok");
        let fp2 = extractor.fingerprint(&samples, 22050).expect("ok");
        let sim = fp1.similarity(&fp2);
        assert!(
            sim > 0.99,
            "identical signals should have similarity > 0.99, got {sim}"
        );
    }

    #[test]
    fn test_fingerprint_different_pitches_lower_similarity() {
        let extractor = ChromagramExtractor::new(ChromaConfig::default());
        // A4 (440 Hz) vs D4 (293 Hz) — different chroma class
        let a4 = sine_wave(440.0, 22050, 1.0);
        let d4 = sine_wave(293.66, 22050, 1.0);
        let fp_a = extractor.fingerprint(&a4, 22050).expect("ok");
        let fp_d = extractor.fingerprint(&d4, 22050).expect("ok");
        let sim = fp_a.similarity(&fp_d);
        // They should be less similar than identical signals.
        assert!(sim < 0.99, "different pitches should differ; got {sim}");
    }

    #[test]
    fn test_fingerprint_resampling() {
        let extractor = ChromagramExtractor::new(ChromaConfig::default());
        // Provide audio at 44100 Hz; extractor should downsample to 22050 Hz.
        let samples_44k = sine_wave(440.0, 44100, 1.0);
        let fp = extractor
            .fingerprint(&samples_44k, 44100)
            .expect("should handle resampling");
        assert_eq!(fp.chroma_means.len(), 12);
        assert_eq!(fp.source_sample_rate, 44100);
    }

    #[test]
    fn test_fingerprint_short_signal_no_panic() {
        let extractor = ChromagramExtractor::new(ChromaConfig::default());
        // Signal shorter than one FFT frame (2048 samples).
        let samples: Vec<f32> = vec![0.5; 512];
        let fp = extractor.fingerprint(&samples, 22050).expect("short signal ok");
        assert_eq!(fp.chroma_means.len(), 12);
    }

    #[test]
    fn test_fingerprint_error_empty_samples() {
        let extractor = ChromagramExtractor::new(ChromaConfig::default());
        let result = extractor.fingerprint(&[], 22050);
        assert!(result.is_err());
    }

    #[test]
    fn test_fingerprint_error_zero_sample_rate() {
        let extractor = ChromagramExtractor::new(ChromaConfig::default());
        let samples = vec![0.0f32; 4096];
        let result = extractor.fingerprint(&samples, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_to_bytes_from_bytes_roundtrip() {
        let extractor = ChromagramExtractor::new(ChromaConfig::default());
        let samples = sine_wave(440.0, 22050, 0.5);
        let fp = extractor.fingerprint(&samples, 22050).expect("ok");
        let bytes = fp.to_bytes();
        assert_eq!(bytes.len(), 96);
        let fp2 = ChromaFingerprint::from_bytes(&bytes, 22050).expect("from_bytes ok");
        // f32 round-trip: small error expected.
        for (a, b) in fp.chroma_means.iter().zip(fp2.chroma_means.iter()) {
            assert!((a - b).abs() < 1e-5, "mean mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_from_bytes_wrong_length() {
        let result = ChromaFingerprint::from_bytes(&[0u8; 50], 22050);
        assert!(result.is_err());
    }

    #[test]
    fn test_combined_similarity_identity() {
        let extractor = ChromagramExtractor::new(ChromaConfig::default());
        let samples = sine_wave(220.0, 22050, 1.0);
        let fp = extractor.fingerprint(&samples, 22050).expect("ok");
        let sim = fp.combined_similarity(&fp, 0.7);
        assert!(
            sim > 0.99,
            "combined_similarity of identical fp should be > 0.99, got {sim}"
        );
    }

    #[test]
    fn test_num_frames_reasonable() {
        let extractor = ChromagramExtractor::new(ChromaConfig::default());
        // 1 second at 22050 Hz → ~(22050 - 2048) / 512 ≈ 39 frames
        let samples = sine_wave(440.0, 22050, 1.0);
        let fp = extractor.fingerprint(&samples, 22050).expect("ok");
        assert!(
            fp.num_frames >= 30,
            "expected at least 30 frames, got {}",
            fp.num_frames
        );
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < f64::EPSILON, "orthogonal vectors sim should be 0");
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![0.0f64; 12];
        let b = vec![1.0f64; 12];
        let sim = cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0, "zero-vector sim should be 0");
    }
}
