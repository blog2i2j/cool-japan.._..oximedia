//! Occlusion filter for spatially-occluded sound sources.
//!
//! When a sound source is blocked by a wall, object, or other acoustic barrier,
//! the path to the listener is dominated by low-frequency transmission and
//! diffraction.  High frequencies are strongly attenuated while low frequencies
//! pass through more easily.
//!
//! This module models occlusion with a simple first-order IIR low-pass filter.
//! The cutoff frequency is configurable, allowing anything from mild filtering
//! (partial occlusion) to heavy filtering (thick walls).
//!
//! ## Example
//!
//! ```rust
//! use oximedia_spatial::occlusion::OcclusionFilter;
//!
//! let mut filter = OcclusionFilter::new(500.0);
//! let input: Vec<f32> = vec![1.0, -1.0, 0.5, -0.5];
//! let output = filter.apply(&input, 48000);
//! assert_eq!(output.len(), 4);
//! ```

/// Low-pass occlusion filter modelling acoustic shadowing.
///
/// State is maintained between calls so the filter can be applied to
/// consecutive audio buffers without discontinuities.
#[derive(Debug, Clone)]
pub struct OcclusionFilter {
    /// Cutoff frequency in Hz.
    pub cutoff_hz: f32,
    /// Internal IIR state (previous output sample).
    state: f32,
}

impl OcclusionFilter {
    /// Create a new occlusion filter with the given cutoff frequency.
    ///
    /// # Arguments
    ///
    /// * `cutoff_hz` — −3 dB point of the low-pass filter in Hz.
    ///   Clamped to the range [1 Hz, 24 000 Hz].
    #[must_use]
    pub fn new(cutoff_hz: f32) -> Self {
        Self {
            cutoff_hz: cutoff_hz.clamp(1.0, 24_000.0),
            state: 0.0,
        }
    }

    /// Reset the internal filter state.
    pub fn reset(&mut self) {
        self.state = 0.0;
    }

    /// Apply the occlusion low-pass filter to `samples`.
    ///
    /// The filter coefficient is computed from `cutoff_hz` and `sample_rate`
    /// using the bilinear transform approximation:
    ///
    /// ```text
    /// ω  = 2π · cutoff / sample_rate
    /// α  = ω / (ω + 1)          (bilinear, stable for ω < π)
    /// y[n] = α · x[n] + (1−α) · y[n−1]
    /// ```
    ///
    /// # Arguments
    ///
    /// * `samples`     — Input audio samples.
    /// * `sample_rate` — Sample rate in Hz (must be > 0; clamped to 1 internally).
    ///
    /// # Returns
    ///
    /// Low-pass filtered output samples of the same length as `samples`.
    pub fn apply(&mut self, samples: &[f32], sample_rate: u32) -> Vec<f32> {
        let fs = sample_rate.max(1) as f32;
        let omega = 2.0 * std::f32::consts::PI * self.cutoff_hz / fs;
        // Stable bilinear-approximation coefficient.
        let alpha = omega / (omega + 1.0);

        let mut out = Vec::with_capacity(samples.len());
        for &x in samples {
            self.state = alpha * x + (1.0 - alpha) * self.state;
            out.push(self.state);
        }
        out
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_clamps_cutoff() {
        let f = OcclusionFilter::new(-100.0);
        assert!((f.cutoff_hz - 1.0).abs() < 1e-6);
        let g = OcclusionFilter::new(99_999.0);
        assert!((g.cutoff_hz - 24_000.0).abs() < 1e-6);
    }

    #[test]
    fn test_output_length_matches_input() {
        let mut filter = OcclusionFilter::new(500.0);
        let input = vec![0.5_f32; 128];
        let output = filter.apply(&input, 48000);
        assert_eq!(output.len(), 128);
    }

    #[test]
    fn test_dc_passthrough() {
        // A DC signal (all 1.0) should converge to 1.0 after many samples.
        let mut filter = OcclusionFilter::new(100.0);
        let input = vec![1.0_f32; 2000];
        let output = filter.apply(&input, 48000);
        let last = *output.last().expect("non-empty output");
        assert!(
            (last - 1.0).abs() < 0.05,
            "DC should pass through low-pass, got {last}"
        );
    }

    #[test]
    fn test_high_freq_attenuated() {
        // A high-frequency signal should be attenuated by a low cutoff filter.
        let mut filter = OcclusionFilter::new(200.0);
        let sr = 48000_u32;
        // Generate a 10 kHz sine (well above cutoff).
        let input: Vec<f32> = (0..512)
            .map(|i| (2.0 * std::f32::consts::PI * 10_000.0 * i as f32 / sr as f32).sin())
            .collect();
        let output = filter.apply(&input, sr);
        let rms_in: f32 =
            (input.iter().map(|v| v * v).sum::<f32>() / input.len() as f32).sqrt();
        let rms_out: f32 =
            (output.iter().map(|v| v * v).sum::<f32>() / output.len() as f32).sqrt();
        assert!(
            rms_out < rms_in * 0.5,
            "High-freq should be attenuated: in={rms_in:.4} out={rms_out:.4}"
        );
    }

    #[test]
    fn test_reset_clears_state() {
        let mut filter = OcclusionFilter::new(500.0);
        let input = vec![1.0_f32; 100];
        let _ = filter.apply(&input, 48000);
        filter.reset();
        assert!((filter.state).abs() < 1e-10, "State should be zero after reset");
    }

    #[test]
    fn test_empty_input() {
        let mut filter = OcclusionFilter::new(500.0);
        let output = filter.apply(&[], 48000);
        assert!(output.is_empty());
    }

    #[test]
    fn test_state_continuity() {
        // Processing in two halves should equal processing the whole at once.
        let mut filter_a = OcclusionFilter::new(1000.0);
        let mut filter_b = OcclusionFilter::new(1000.0);
        let input: Vec<f32> = (0..128).map(|i| (i as f32) * 0.01).collect();
        let full_out = filter_a.apply(&input, 48000);
        let first_half = filter_b.apply(&input[..64], 48000);
        let second_half = filter_b.apply(&input[64..], 48000);
        let split_out: Vec<f32> = first_half.into_iter().chain(second_half).collect();
        for (a, b) in full_out.iter().zip(split_out.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "Stateful processing must be continuous: {a} vs {b}"
            );
        }
    }
}
