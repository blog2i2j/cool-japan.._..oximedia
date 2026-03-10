#![allow(dead_code)]
//! High-level spectral feature computation: centroid, flux, rolloff.

/// Spectral centroid — the "centre of mass" of a spectrum.
pub struct SpectralCentroid {
    /// Minimum frequency bin to include (Hz).
    pub min_freq: f32,
    /// Maximum frequency bin to include (Hz).
    pub max_freq: f32,
}

impl SpectralCentroid {
    /// Create a new `SpectralCentroid` with default frequency range.
    #[must_use]
    pub fn new() -> Self {
        Self {
            min_freq: 20.0,
            max_freq: 20_000.0,
        }
    }

    /// Compute spectral centroid (Hz) from frequency bins and their magnitudes.
    ///
    /// `freqs` and `magnitudes` must be the same length.
    /// Returns 0.0 if all magnitudes are zero or slices are empty.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn compute(&self, freqs: &[f32], magnitudes: &[f32]) -> f32 {
        assert_eq!(
            freqs.len(),
            magnitudes.len(),
            "freqs and magnitudes must have the same length"
        );

        let mut weighted_sum = 0.0_f64;
        let mut total_mag = 0.0_f64;

        for (&f, &m) in freqs.iter().zip(magnitudes.iter()) {
            if f < self.min_freq || f > self.max_freq {
                continue;
            }
            let mag = f64::from(m);
            weighted_sum += f64::from(f) * mag;
            total_mag += mag;
        }

        if total_mag == 0.0 {
            return 0.0;
        }

        (weighted_sum / total_mag) as f32
    }
}

impl Default for SpectralCentroid {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------

/// Spectral flux — measures the rate of change of the power spectrum over time.
pub struct SpectralFlux {
    /// Previous frame magnitudes.
    prev_magnitudes: Vec<f32>,
    /// Whether to use half-wave rectification (only positive differences).
    pub half_wave: bool,
}

impl SpectralFlux {
    /// Create a new `SpectralFlux` processor.
    #[must_use]
    pub fn new(half_wave: bool) -> Self {
        Self {
            prev_magnitudes: Vec::new(),
            half_wave,
        }
    }

    /// Feed a new frame of magnitudes and return the flux value.
    ///
    /// Returns `None` on the first call (no previous frame to compare).
    #[allow(clippy::cast_precision_loss)]
    pub fn compute_flux(&mut self, magnitudes: &[f32]) -> Option<f32> {
        if self.prev_magnitudes.is_empty() {
            self.prev_magnitudes = magnitudes.to_vec();
            return None;
        }

        let n = self.prev_magnitudes.len().min(magnitudes.len());
        let mut flux = 0.0_f64;

        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            let diff = f64::from(magnitudes[i]) - f64::from(self.prev_magnitudes[i]);
            let val = if self.half_wave { diff.max(0.0) } else { diff };
            flux += val * val;
        }

        self.prev_magnitudes = magnitudes.to_vec();
        Some((flux / n as f64).sqrt() as f32)
    }

    /// Reset internal state (clears the previous frame buffer).
    pub fn reset(&mut self) {
        self.prev_magnitudes.clear();
    }
}

// ---------------------------------------------------------------------------

/// Spectral rolloff — the frequency below which `threshold` (e.g. 0.85) of
/// total spectral energy is contained.
pub struct SpectralRolloff {
    /// Default rolloff threshold (0.0–1.0).
    pub threshold: f32,
}

impl SpectralRolloff {
    /// Create a new `SpectralRolloff` with the given threshold.
    #[must_use]
    pub fn new(threshold: f32) -> Self {
        Self {
            threshold: threshold.clamp(0.0, 1.0),
        }
    }

    /// Compute the rolloff frequency given frequency bins and magnitudes.
    ///
    /// Returns the frequency (Hz) at which the cumulative energy reaches
    /// `threshold` of the total energy.  Returns 0.0 if the spectrum is empty.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn compute_rolloff(&self, freqs: &[f32], magnitudes: &[f32]) -> f32 {
        assert_eq!(freqs.len(), magnitudes.len());

        if freqs.is_empty() {
            return 0.0;
        }

        let total_energy: f64 = magnitudes
            .iter()
            .map(|&m| f64::from(m) * f64::from(m))
            .sum();
        if total_energy == 0.0 {
            return 0.0;
        }

        let target = total_energy * f64::from(self.threshold);
        let mut cumulative = 0.0_f64;

        for (&f, &m) in freqs.iter().zip(magnitudes.iter()) {
            cumulative += f64::from(m) * f64::from(m);
            if cumulative >= target {
                return f;
            }
        }

        // If we never hit the threshold, return the last frequency.
        *freqs.last().unwrap_or(&0.0)
    }
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_freqs(n: usize, max_hz: f32) -> Vec<f32> {
        (0..n).map(|i| i as f32 / (n - 1) as f32 * max_hz).collect()
    }

    // --- SpectralCentroid ---

    #[test]
    fn test_centroid_uniform_magnitudes() {
        let freqs = vec![100.0, 200.0, 300.0, 400.0];
        let mags = vec![1.0, 1.0, 1.0, 1.0];
        let centroid = SpectralCentroid::new();
        let c = centroid.compute(&freqs, &mags);
        // Centroid should be mean of freqs = 250 Hz
        assert!((c - 250.0).abs() < 1.0, "centroid={c}");
    }

    #[test]
    fn test_centroid_single_peak() {
        let freqs = vec![100.0, 500.0, 1000.0];
        let mags = vec![0.0, 1.0, 0.0]; // only 500 Hz has energy
        let centroid = SpectralCentroid::new();
        let c = centroid.compute(&freqs, &mags);
        assert!((c - 500.0).abs() < 0.1, "centroid={c}");
    }

    #[test]
    fn test_centroid_zero_magnitudes() {
        let freqs = vec![100.0, 200.0];
        let mags = vec![0.0, 0.0];
        let centroid = SpectralCentroid::new();
        assert_eq!(centroid.compute(&freqs, &mags), 0.0);
    }

    #[test]
    fn test_centroid_empty() {
        let centroid = SpectralCentroid::new();
        assert_eq!(centroid.compute(&[], &[]), 0.0);
    }

    #[test]
    fn test_centroid_frequency_range_filter() {
        // Only 500 Hz is inside the custom range.
        let freqs = vec![10.0, 500.0, 50_000.0];
        let mags = vec![5.0, 2.0, 5.0];
        let centroid = SpectralCentroid {
            min_freq: 100.0,
            max_freq: 10_000.0,
        };
        let c = centroid.compute(&freqs, &mags);
        assert!((c - 500.0).abs() < 0.1, "centroid={c}");
    }

    #[test]
    fn test_centroid_default() {
        let centroid = SpectralCentroid::default();
        assert!((centroid.min_freq - 20.0).abs() < f32::EPSILON);
    }

    // --- SpectralFlux ---

    #[test]
    fn test_flux_first_frame_returns_none() {
        let mut flux = SpectralFlux::new(false);
        assert!(flux.compute_flux(&[1.0, 2.0, 3.0]).is_none());
    }

    #[test]
    fn test_flux_identical_frames_zero() {
        let mut flux = SpectralFlux::new(false);
        let mags = vec![1.0, 1.0, 1.0];
        flux.compute_flux(&mags);
        let f = flux
            .compute_flux(&mags)
            .expect("flux computation should succeed");
        assert!(f.abs() < 1e-6, "flux={f}");
    }

    #[test]
    fn test_flux_increasing_energy() {
        let mut flux = SpectralFlux::new(false);
        flux.compute_flux(&[0.0, 0.0, 0.0]);
        let f = flux
            .compute_flux(&[1.0, 1.0, 1.0])
            .expect("flux computation should succeed");
        assert!(f > 0.0, "Expected positive flux, got {f}");
    }

    #[test]
    fn test_flux_half_wave_suppresses_negative() {
        let mut flux_full = SpectralFlux::new(false);
        let mut flux_half = SpectralFlux::new(true);

        let frame1 = vec![2.0, 2.0, 2.0];
        let frame2 = vec![1.0, 1.0, 1.0]; // energy drops

        flux_full.compute_flux(&frame1);
        flux_half.compute_flux(&frame1);

        let full = flux_full
            .compute_flux(&frame2)
            .expect("flux computation should succeed");
        let half = flux_half
            .compute_flux(&frame2)
            .expect("flux computation should succeed");

        // Half-wave should be 0 (energy drop suppressed)
        assert!(half.abs() < 1e-6, "half={half}");
        // Full should be non-zero
        assert!(full > 0.0, "full={full}");
    }

    #[test]
    fn test_flux_reset_clears_state() {
        let mut flux = SpectralFlux::new(false);
        flux.compute_flux(&[1.0, 2.0]);
        flux.reset();
        // After reset, next call should return None again
        assert!(flux.compute_flux(&[1.0, 2.0]).is_none());
    }

    // --- SpectralRolloff ---

    #[test]
    fn test_rolloff_finds_correct_bin() {
        let freqs: Vec<f32> = (1..=10).map(|i| i as f32 * 100.0).collect();
        // Uniform magnitudes: all 1.0, so total energy = 10.
        // 85% threshold → need 8.5 units → rolloff at bin 9 (900 Hz).
        let mags = vec![1.0_f32; 10];
        let rolloff = SpectralRolloff::new(0.85);
        let r = rolloff.compute_rolloff(&freqs, &mags);
        assert!(r > 0.0, "rolloff={r}");
        assert!(r <= 1000.0, "rolloff={r}");
    }

    #[test]
    fn test_rolloff_empty_spectrum() {
        let rolloff = SpectralRolloff::new(0.85);
        assert_eq!(rolloff.compute_rolloff(&[], &[]), 0.0);
    }

    #[test]
    fn test_rolloff_zero_energy() {
        let freqs = vec![100.0, 200.0];
        let mags = vec![0.0, 0.0];
        let rolloff = SpectralRolloff::new(0.85);
        assert_eq!(rolloff.compute_rolloff(&freqs, &mags), 0.0);
    }

    #[test]
    fn test_rolloff_full_threshold() {
        // Threshold = 1.0 → should return the last frequency.
        let freqs: Vec<f32> = make_freqs(5, 4000.0);
        let mags = vec![1.0_f32; 5];
        let rolloff = SpectralRolloff::new(1.0);
        let r = rolloff.compute_rolloff(&freqs, &mags);
        assert!((r - 4000.0).abs() < 1.0, "rolloff={r}");
    }

    #[test]
    fn test_rolloff_clamped_threshold() {
        let rolloff = SpectralRolloff::new(1.5);
        assert!((rolloff.threshold - 1.0).abs() < f32::EPSILON);
        let rolloff2 = SpectralRolloff::new(-0.1);
        assert!((rolloff2.threshold - 0.0).abs() < f32::EPSILON);
    }
}
