//! Spectral subtraction noise reduction.
//!
//! Spectral subtraction is a classic frequency-domain noise-reduction technique.
//! A noise estimate (power spectrum) is measured during a known-quiet segment
//! and then subtracted from each incoming spectral frame.  A half-wave
//! rectification (flooring at zero) prevents the result from going negative.
//!
//! This implementation operates directly on a complex-or-magnitude spectrum
//! slice rather than performing its own FFT, making it composable with any
//! FFT front-end.
//!
//! # References
//!
//! Boll, S. F. (1979). "Suppression of acoustic noise in speech using spectral
//! subtraction." *IEEE Transactions on Acoustics, Speech, and Signal
//! Processing*, 27(2), 113–120.
//!
//! # Example
//!
//! ```
//! use oximedia_restore::spectral_sub::SpectralSubtractor;
//!
//! let noise_est = vec![0.1f32; 8];
//! let mut sub = SpectralSubtractor::new(&noise_est);
//!
//! let mut spectrum = vec![0.5f32; 8];
//! sub.reduce(&mut spectrum);
//! // Each bin should be (0.5 - 0.1) = 0.4, floored at 0
//! for &s in &spectrum {
//!     assert!((s - 0.4).abs() < 1e-5);
//! }
//! ```

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Spectral subtractor that reduces noise using a fixed noise power estimate.
///
/// The noise estimate is set once at construction time.  For adaptive noise
/// floor tracking, create a new `SpectralSubtractor` whenever the estimated
/// noise changes.
#[derive(Debug, Clone)]
pub struct SpectralSubtractor {
    /// Per-bin noise magnitude estimate.
    noise_est: Vec<f32>,
    /// Over-subtraction factor α — values > 1.0 reduce musical noise at the
    /// cost of slight signal distortion.  Defaults to 1.0.
    alpha: f32,
    /// Spectral floor β — minimum fraction of the original magnitude retained
    /// even in strongly-noisy bins.  Prevents complete silence.  Defaults to 0.0.
    beta: f32,
}

impl SpectralSubtractor {
    /// Create a new `SpectralSubtractor` from a noise estimate.
    ///
    /// # Parameters
    ///
    /// * `noise_est` — per-bin noise magnitude estimate (non-negative).
    ///   Bins are zero-padded / truncated to match the spectrum length during
    ///   [`reduce`][Self::reduce].
    pub fn new(noise_est: &[f32]) -> Self {
        Self {
            noise_est: noise_est.to_vec(),
            alpha: 1.0,
            beta: 0.0,
        }
    }

    /// Set the over-subtraction factor α (default: 1.0).
    ///
    /// Values > 1.0 provide more aggressive noise suppression at the cost of
    /// slight distortion of the desired signal.
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha.max(0.0);
        self
    }

    /// Set the spectral floor β (default: 0.0).
    ///
    /// The output for bin *k* is floored at `β × |X[k]|` so that no bin is
    /// reduced to complete silence (avoids musical noise in some cases).
    pub fn with_beta(mut self, beta: f32) -> Self {
        self.beta = beta.clamp(0.0, 1.0);
        self
    }

    /// Update the noise estimate in-place.
    pub fn update_noise_estimate(&mut self, noise_est: &[f32]) {
        self.noise_est = noise_est.to_vec();
    }

    /// Apply spectral subtraction to `spectrum` in-place.
    ///
    /// For each bin *k*:
    ///
    /// ```text
    /// spectrum[k] = max(spectrum[k] − α · noise_est[k],  β · spectrum[k])
    /// ```
    ///
    /// Bins for which no noise estimate is available (i.e., beyond
    /// `noise_est.len()`) are left unchanged.
    pub fn reduce(&self, spectrum: &mut [f32]) {
        for (k, s) in spectrum.iter_mut().enumerate() {
            let noise = if k < self.noise_est.len() {
                self.noise_est[k]
            } else {
                // No estimate for this bin — leave unchanged.
                continue;
            };
            let floor = self.beta * *s;
            let subtracted = *s - self.alpha * noise;
            *s = subtracted.max(floor).max(0.0);
        }
    }

    /// Return the length of the stored noise estimate.
    pub fn noise_len(&self) -> usize {
        self.noise_est.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_subtraction() {
        let noise = vec![0.1f32; 8];
        let sub = SpectralSubtractor::new(&noise);
        let mut spec = vec![0.5f32; 8];
        sub.reduce(&mut spec);
        for &s in &spec {
            assert!((s - 0.4).abs() < 1e-5, "expected 0.4, got {s}");
        }
    }

    #[test]
    fn test_floor_at_zero() {
        // Noise estimate exceeds signal → output floored at zero.
        let noise = vec![1.0f32; 4];
        let sub = SpectralSubtractor::new(&noise);
        let mut spec = vec![0.5f32; 4];
        sub.reduce(&mut spec);
        for &s in &spec {
            assert!(s >= 0.0, "output must not go negative");
            assert!(s.abs() < 1e-5, "fully suppressed bin should be ~0");
        }
    }

    #[test]
    fn test_zero_noise_unchanged() {
        let noise = vec![0.0f32; 6];
        let sub = SpectralSubtractor::new(&noise);
        let mut spec = vec![0.8f32; 6];
        sub.reduce(&mut spec);
        for &s in &spec {
            assert!((s - 0.8).abs() < 1e-6);
        }
    }

    #[test]
    fn test_bins_beyond_noise_estimate_unchanged() {
        let noise = vec![0.1f32; 4];
        let sub = SpectralSubtractor::new(&noise);
        let mut spec = vec![0.5f32; 8]; // 8 bins but only 4 noise estimates
        sub.reduce(&mut spec);
        // First 4 bins should be subtracted.
        for &s in spec.iter().take(4) {
            assert!((s - 0.4).abs() < 1e-5);
        }
        // Last 4 bins should be unchanged at 0.5.
        for &s in spec.iter().skip(4) {
            assert!((s - 0.5).abs() < 1e-5);
        }
    }

    #[test]
    fn test_over_subtraction_alpha() {
        let noise = vec![0.1f32; 4];
        let sub = SpectralSubtractor::new(&noise).with_alpha(2.0);
        let mut spec = vec![0.5f32; 4];
        sub.reduce(&mut spec);
        // Expected: max(0.5 - 2*0.1, 0) = max(0.3, 0) = 0.3
        for &s in &spec {
            assert!((s - 0.3).abs() < 1e-5, "expected 0.3, got {s}");
        }
    }

    #[test]
    fn test_spectral_floor_beta() {
        let noise = vec![0.4f32; 4]; // large noise
        let sub = SpectralSubtractor::new(&noise).with_beta(0.2); // floor at 20 % of original
        let mut spec = vec![0.5f32; 4];
        sub.reduce(&mut spec);
        // subtracted = 0.5 - 0.4 = 0.1, floor = 0.2 * 0.5 = 0.1 → max = 0.1
        for &s in &spec {
            assert!(
                s >= 0.09 && s <= 0.11,
                "beta-floor should yield ~0.1, got {s}"
            );
        }
    }

    #[test]
    fn test_noise_len_accessor() {
        let noise = vec![0.0f32; 10];
        let sub = SpectralSubtractor::new(&noise);
        assert_eq!(sub.noise_len(), 10);
    }

    #[test]
    fn test_update_noise_estimate() {
        let mut sub = SpectralSubtractor::new(&[0.1f32; 4]);
        sub.update_noise_estimate(&[0.2f32; 4]);
        let mut spec = vec![0.5f32; 4];
        sub.reduce(&mut spec);
        for &s in &spec {
            assert!(
                (s - 0.3).abs() < 1e-5,
                "updated noise: expected 0.3, got {s}"
            );
        }
    }

    #[test]
    fn test_empty_spectrum_no_panic() {
        let noise = vec![0.1f32; 4];
        let sub = SpectralSubtractor::new(&noise);
        let mut spec: Vec<f32> = Vec::new();
        sub.reduce(&mut spec); // must not panic
    }
}
