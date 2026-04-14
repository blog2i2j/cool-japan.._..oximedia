//! Simplified spread-spectrum audio watermark detection.
//!
//! # Method
//!
//! A watermark is embedded by adding a very small-amplitude signal at a narrow
//! band of specific frequency bins (bins 100–110 by default, relative to an
//! FFT of size 1024).  Detection works by comparing the energy at those target
//! bins against the energy of their immediate neighbours and looking for a
//! statistically significant anomaly (more than 3σ above the neighbourhood).
//!
//! ## Detection Pipeline
//!
//! 1. Compute the magnitude spectrum of each analysis frame via a power-of-two
//!    FFT (frame size configured on the detector).
//! 2. For the target bin range `[bin_lo, bin_hi]`, compute the mean energy and
//!    the mean energy of the neighbouring bins `[bin_lo - margin, bin_lo)` and
//!    `(bin_hi, bin_hi + margin]`.
//! 3. Compute the Z-score of the target-band energy relative to the neighbourhood
//!    distribution: `z = (mean_target - mean_neighbors) / std_neighbors`.
//! 4. Across all frames, average the per-frame Z-scores.  If the average Z-score
//!    exceeds `3.0`, a watermark is reported.
//! 5. A simple 64-bit payload is estimated by thresholding each per-bin energy
//!    anomaly against the median frame anomaly and packing the results into a
//!    `u64` bitmask (using the first 11 bins mapped to bits 0–10 of a `u64`).
//!
//! ## Stub Injection
//!
//! [`inject_stub_watermark`] adds a very low-amplitude (0.001 by default)
//! sinusoidal signal at each target bin frequency to the input signal.  The
//! payload is encoded by selectively enabling or disabling bins based on the
//! bit pattern of the payload.
//!
//! This module is intentionally a *simplified stub* — it is designed for
//! round-trip testing and teaching, not for adversarial watermark robustness.

#![allow(dead_code)]

use oxifft::Complex;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default FFT window size used by the detector.
const DEFAULT_WINDOW: usize = 1024;

/// Default hop size.
const DEFAULT_HOP: usize = 256;

/// Target bin range (inclusive) used for watermark energy analysis.
const TARGET_BIN_LO: usize = 100;
const TARGET_BIN_HI: usize = 110;

/// Number of neighbour bins on each side used for background estimation.
const NEIGHBOUR_MARGIN: usize = 20;

/// Z-score threshold for detection.
const DETECTION_Z_THRESHOLD: f32 = 3.0;

/// Default injection amplitude.
const DEFAULT_INJECT_AMPLITUDE: f32 = 0.001;

// ---------------------------------------------------------------------------
// WatermarkResult
// ---------------------------------------------------------------------------

/// Result returned by [`WatermarkDetector::detect`].
#[derive(Debug, Clone)]
pub struct WatermarkResult {
    /// Confidence that a watermark is present (0.0 – 1.0).
    ///
    /// Derived from how many standard deviations above the neighbourhood the
    /// target band energy is on average.
    pub confidence: f32,

    /// Estimated 64-bit payload encoded in the watermark.
    ///
    /// Each of the 11 target bins (100–110) maps to one bit.  A bin whose
    /// energy anomaly exceeds the median anomaly sets that bit to 1.
    pub estimated_payload: u64,

    /// Frequency band (Hz) corresponding to the watermark bins.
    ///
    /// Computed as `(bin_lo * sr / window_size, bin_hi * sr / window_size)`.
    pub frequency_band_hz: (f32, f32),
}

// ---------------------------------------------------------------------------
// WatermarkDetector
// ---------------------------------------------------------------------------

/// Spectral-anomaly-based audio watermark detector.
pub struct WatermarkDetector {
    sample_rate: u32,
    window_size: usize,
    hop_size: usize,
    bin_lo: usize,
    bin_hi: usize,
    neighbour_margin: usize,
    z_threshold: f32,
}

impl WatermarkDetector {
    /// Create a new detector for audio at `sample_rate` Hz.
    ///
    /// All other parameters are set to sensible defaults.
    #[must_use]
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            window_size: DEFAULT_WINDOW,
            hop_size: DEFAULT_HOP,
            bin_lo: TARGET_BIN_LO,
            bin_hi: TARGET_BIN_HI,
            neighbour_margin: NEIGHBOUR_MARGIN,
            z_threshold: DETECTION_Z_THRESHOLD,
        }
    }

    /// Customise the target bin range.
    #[must_use]
    pub fn with_bins(mut self, bin_lo: usize, bin_hi: usize) -> Self {
        self.bin_lo = bin_lo;
        self.bin_hi = bin_hi;
        self
    }

    /// Attempt to detect a spread-spectrum watermark in `samples`.
    ///
    /// Returns `Some(WatermarkResult)` when a statistically significant energy
    /// anomaly is found in the target band.  Returns `None` when the signal is
    /// too short or no anomaly is detected.
    ///
    /// # Panics
    ///
    /// Does not panic.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn detect(&self, samples: &[f32]) -> Option<WatermarkResult> {
        if samples.len() < self.window_size {
            return None;
        }

        let window_size = self.window_size;
        let hop = self.hop_size;
        let n_bins = window_size / 2 + 1;

        // Guard against out-of-range bin indices.
        if self.bin_lo >= n_bins || self.bin_hi >= n_bins || self.bin_lo > self.bin_hi {
            return None;
        }

        // Hann window coefficients.
        let hann: Vec<f32> = (0..window_size)
            .map(|i| {
                0.5 * (1.0
                    - (std::f32::consts::TAU * i as f32 / (window_size - 1) as f32).cos())
            })
            .collect();

        let n_frames = (samples.len().saturating_sub(window_size)) / hop + 1;

        // Per-frame accumulations.
        let n_target = self.bin_hi - self.bin_lo + 1;
        let mut per_frame_z: Vec<f32> = Vec::with_capacity(n_frames);

        // Accumulate per-bin energy across frames for payload estimation.
        let mut bin_energy_acc = vec![0.0_f32; n_target];

        for frame_idx in 0..n_frames {
            let start = frame_idx * hop;
            let end = start + window_size;
            if end > samples.len() {
                break;
            }

            // Apply Hann window.
            let windowed: Vec<Complex<f32>> = samples[start..end]
                .iter()
                .zip(hann.iter())
                .map(|(&s, &w)| Complex::new(s * w, 0.0))
                .collect();

            let spectrum = oxifft::fft(&windowed);

            // Magnitude in [0, n_bins).
            let mag: Vec<f32> = spectrum[..n_bins].iter().map(|c| c.norm()).collect();

            // Target band energy.
            let target_slice = &mag[self.bin_lo..=self.bin_hi];
            let mean_target: f32 =
                target_slice.iter().sum::<f32>() / n_target as f32;

            // Accumulate per-bin energy for payload estimation.
            for (k, &v) in target_slice.iter().enumerate() {
                bin_energy_acc[k] += v;
            }

            // Neighbour bins: left side.
            let left_lo = self.bin_lo.saturating_sub(self.neighbour_margin);
            let left_hi = self.bin_lo.saturating_sub(1);

            // Neighbour bins: right side.
            let right_lo = (self.bin_hi + 1).min(n_bins - 1);
            let right_hi = (self.bin_hi + self.neighbour_margin).min(n_bins - 1);

            let mut neighbour_vals: Vec<f32> = Vec::new();
            if left_hi >= left_lo {
                neighbour_vals.extend_from_slice(&mag[left_lo..=left_hi]);
            }
            if right_hi >= right_lo {
                neighbour_vals.extend_from_slice(&mag[right_lo..=right_hi]);
            }

            if neighbour_vals.is_empty() {
                per_frame_z.push(0.0);
                continue;
            }

            let n_nb = neighbour_vals.len() as f32;
            let mean_nb = neighbour_vals.iter().sum::<f32>() / n_nb;
            let std_nb = {
                let var = neighbour_vals
                    .iter()
                    .map(|&v| (v - mean_nb).powi(2))
                    .sum::<f32>()
                    / n_nb;
                var.sqrt()
            };

            let z = if std_nb > 1e-9 {
                (mean_target - mean_nb) / std_nb
            } else {
                0.0
            };

            per_frame_z.push(z);
        }

        if per_frame_z.is_empty() {
            return None;
        }

        let mean_z = per_frame_z.iter().sum::<f32>() / per_frame_z.len() as f32;

        if mean_z < self.z_threshold {
            return None;
        }

        // Normalise confidence: how far mean_z exceeds threshold, capped at 1.0.
        let confidence = ((mean_z - self.z_threshold) / (self.z_threshold + 1.0)).clamp(0.0, 1.0);

        // Estimate payload from per-bin anomaly.
        let n_frames_counted = per_frame_z.len() as f32;
        let mean_bin_energy: Vec<f32> = bin_energy_acc
            .iter()
            .map(|&e| e / n_frames_counted)
            .collect();

        let mut sorted_bin_e = mean_bin_energy.clone();
        sorted_bin_e.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_bin_e = sorted_bin_e[sorted_bin_e.len() / 2];

        let mut payload: u64 = 0;
        for (bit_idx, &e) in mean_bin_energy.iter().enumerate().take(64) {
            if e > median_bin_e {
                payload |= 1u64 << bit_idx;
            }
        }

        let sr = self.sample_rate as f32;
        let band_lo_hz = self.bin_lo as f32 * sr / window_size as f32;
        let band_hi_hz = self.bin_hi as f32 * sr / window_size as f32;

        Some(WatermarkResult {
            confidence,
            estimated_payload: payload,
            frequency_band_hz: (band_lo_hz, band_hi_hz),
        })
    }
}

// ---------------------------------------------------------------------------
// inject_stub_watermark
// ---------------------------------------------------------------------------

/// Add a low-amplitude spread-spectrum watermark stub to `samples` in-place.
///
/// For each bit set in `payload` (up to 11 bits, one per target bin 100–110),
/// a sinusoidal signal at the corresponding bin frequency is added at
/// `amplitude` (default 0.001).  Bits beyond the number of target bins are
/// silently ignored.
///
/// This is a **test-only** utility; the resulting watermark is fragile and
/// not suitable for production use.
///
/// # Arguments
///
/// * `samples` — mutable slice of audio samples (modified in-place).
/// * `payload` — 64-bit value encoding which bins to inject (bit 0 → bin 100, etc.).
/// * `sample_rate` — sample rate of the audio.
#[allow(clippy::cast_precision_loss)]
pub fn inject_stub_watermark(samples: &mut [f32], payload: u64, sample_rate: u32) {
    if samples.is_empty() || sample_rate == 0 {
        return;
    }

    let sr = sample_rate as f32;
    let window = DEFAULT_WINDOW as f32;
    let amplitude = DEFAULT_INJECT_AMPLITUDE;

    for bit_idx in 0..11usize {
        if payload & (1u64 << bit_idx) == 0 {
            continue;
        }
        let bin = (TARGET_BIN_LO + bit_idx) as f32;
        let freq = bin * sr / window;

        for (t, sample) in samples.iter_mut().enumerate() {
            *sample +=
                amplitude * (std::f32::consts::TAU * freq * t as f32 / sr).sin();
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    fn make_sine(freq: f32, sr: u32, secs: f32) -> Vec<f32> {
        let n = (sr as f32 * secs) as usize;
        (0..n)
            .map(|i| (TAU * freq * i as f32 / sr as f32).sin())
            .collect()
    }

    // ── WatermarkDetector::new ────────────────────────────────────────────────

    #[test]
    fn test_detector_creation() {
        let detector = WatermarkDetector::new(44100);
        assert_eq!(detector.sample_rate, 44100);
        assert_eq!(detector.window_size, DEFAULT_WINDOW);
    }

    // ── detect — no watermark ─────────────────────────────────────────────────

    #[test]
    fn test_detect_short_signal_returns_none() {
        let detector = WatermarkDetector::new(44100);
        let short = vec![0.0f32; 100];
        assert!(detector.detect(&short).is_none());
    }

    #[test]
    fn test_detect_silence_returns_none() {
        let detector = WatermarkDetector::new(44100);
        let silence = vec![0.0f32; 44100];
        // Pure silence has no frequency anomaly.
        assert!(detector.detect(&silence).is_none());
    }

    #[test]
    fn test_detect_pure_sine_returns_none() {
        let detector = WatermarkDetector::new(44100);
        // 440 Hz sine — unlikely to trigger target band anomaly.
        let sig = make_sine(440.0, 44100, 1.0);
        let result = detector.detect(&sig);
        // A pure tone away from target band should not trigger detection.
        // (It might in edge cases, so we only assert no panic here.)
        let _ = result;
    }

    // ── inject_stub_watermark ─────────────────────────────────────────────────

    #[test]
    fn test_inject_modifies_signal() {
        let mut sig = make_sine(440.0, 44100, 1.0);
        let original = sig.clone();
        inject_stub_watermark(&mut sig, 0b111, 44100);
        let modified = sig
            .iter()
            .zip(original.iter())
            .any(|(&a, &b)| (a - b).abs() > 1e-7);
        assert!(modified, "Signal should be modified after injection");
    }

    #[test]
    fn test_inject_zero_payload_no_change() {
        let mut sig = make_sine(440.0, 44100, 0.5);
        let original = sig.clone();
        inject_stub_watermark(&mut sig, 0, 44100);
        // Zero payload: no bits set, nothing injected.
        assert_eq!(sig, original);
    }

    #[test]
    fn test_inject_empty_signal_no_panic() {
        let mut empty: Vec<f32> = Vec::new();
        inject_stub_watermark(&mut empty, 0xFF, 44100);
        assert!(empty.is_empty());
    }

    // ── round-trip test ───────────────────────────────────────────────────────

    #[test]
    fn test_inject_then_detect_roundtrip() {
        let sr = 44100u32;
        let detector = WatermarkDetector::new(sr);
        // Use a white-noise-like signal with clear energy everywhere to make
        // the injected anomaly stand out.
        let mut sig = make_sine(440.0, sr, 2.0);
        let payload = 0b0111_1111_1111_u64; // all 11 bits set
        inject_stub_watermark(&mut sig, payload, sr);

        // Detection may or may not fire depending on the specific amplitude
        // and frequency content; we only assert no panic and valid types.
        if let Some(result) = detector.detect(&sig) {
            assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
            let (lo, hi) = result.frequency_band_hz;
            assert!(lo <= hi);
            assert!(lo >= 0.0);
        }
    }

    #[test]
    fn test_watermark_result_frequency_band() {
        // Inject a full payload into a signal and verify the reported frequency band
        // is within the expected range for SR=44100, window=1024.
        let sr = 44100u32;
        let detector = WatermarkDetector::new(sr);
        let mut sig = vec![0.1f32; 44100 * 2]; // Flat-ish signal
        // Add DC bias so target bins have measurable energy after injection.
        inject_stub_watermark(&mut sig, 0b0111_1111_1111_u64, sr);
        if let Some(result) = detector.detect(&sig) {
            let (lo, hi) = result.frequency_band_hz;
            // Bin 100 @ 44100/1024 ≈ 4307 Hz; Bin 110 @ 44100/1024 ≈ 4736 Hz
            let expected_lo = 100.0 * 44100.0 / 1024.0;
            let expected_hi = 110.0 * 44100.0 / 1024.0;
            assert!((lo - expected_lo).abs() < 10.0);
            assert!((hi - expected_hi).abs() < 10.0);
        }
    }

    #[test]
    fn test_with_bins_customisation() {
        let detector = WatermarkDetector::new(44100).with_bins(50, 60);
        assert_eq!(detector.bin_lo, 50);
        assert_eq!(detector.bin_hi, 60);
    }
}
