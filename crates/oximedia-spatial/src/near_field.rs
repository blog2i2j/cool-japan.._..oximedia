//! Near-field compensation for close-source spatial audio rendering.
//!
//! When a sound source is very close to the listener (within a few metres),
//! the standard inverse-square amplitude model and HRTF assumptions break down.
//! This module provides a proximity equalisation filter that boosts low
//! frequencies (the "proximity effect") and attenuates high frequencies for
//! very close distances.
//!
//! ## Algorithm
//!
//! A simple shelving model is used:
//!
//! 1. **Proximity boost** — Below the configured `threshold_m`, a low-frequency
//!    shelf gain proportional to `threshold_m / dist_m` is applied up to a
//!    maximum of +12 dB.
//! 2. **High-frequency damping** — At very close distances (`< 0.1 m`), a mild
//!    first-order IIR low-pass filter rolls off frequencies above ~8 kHz.
//!
//! ## Example
//!
//! ```rust
//! use oximedia_spatial::near_field::NearFieldProcessor;
//!
//! let proc = NearFieldProcessor::new(1.0);
//! let input: Vec<f32> = vec![0.5; 128];
//! let output = proc.apply_proximity_eq(0.3, &input);
//! assert_eq!(output.len(), 128);
//! ```

/// Near-field compensation processor.
#[derive(Debug, Clone)]
pub struct NearFieldProcessor {
    /// Distance threshold below which proximity EQ is applied (metres).
    pub threshold_m: f32,
}

impl NearFieldProcessor {
    /// Create a new near-field processor.
    ///
    /// # Arguments
    ///
    /// * `threshold_m` — Distance below which proximity EQ is applied (metres).
    ///   Must be positive; clamped to a minimum of 0.01 m internally.
    #[must_use]
    pub fn new(threshold_m: f32) -> Self {
        Self {
            threshold_m: threshold_m.max(0.01),
        }
    }

    /// Apply proximity EQ to a block of samples for a source at `dist_m` metres.
    ///
    /// When `dist_m >= threshold_m`, the samples are returned unmodified.
    /// When `dist_m < threshold_m`:
    /// - A gain proportional to the proximity ratio is computed (capped at
    ///   linear +4, i.e. approximately +12 dB).
    /// - If `dist_m < 0.1 m`, a first-order low-pass IIR (cutoff ≈ 8 kHz at
    ///   48 kHz sample rate) is applied to model near-field high-frequency
    ///   directional changes.
    ///
    /// # Arguments
    ///
    /// * `dist_m`  — Distance from source to listener in metres (clamped to ≥ 0.001).
    /// * `samples` — Input audio samples.
    ///
    /// # Returns
    ///
    /// Processed audio samples of the same length as `samples`.
    #[must_use]
    pub fn apply_proximity_eq(&self, dist_m: f32, samples: &[f32]) -> Vec<f32> {
        let dist = dist_m.max(0.001);

        if dist >= self.threshold_m {
            return samples.to_vec();
        }

        // Proximity ratio: how far into the near-field zone are we?
        // ratio == 1.0 at the threshold, grows as distance shrinks.
        let ratio = self.threshold_m / dist;

        // Low-frequency boost: cap at linear 4.0 (≈ +12 dB).
        let lf_gain = ratio.min(4.0);

        // Apply gain and optional high-frequency damping.
        if dist < 0.1 {
            // Very close: apply IIR low-pass to attenuate HF (approximate).
            // Coefficient for fc ≈ 8 kHz at 48 kHz: alpha = exp(-2π·fc/fs).
            // Using a fixed alpha suitable for a broad range of sample rates.
            const ALPHA: f32 = 0.35; // tuned for ~8 kHz at 48 kHz
            let mut out = Vec::with_capacity(samples.len());
            let mut z = 0.0_f32;
            for &s in samples {
                // First-order IIR low-pass: y[n] = (1-α)·x[n] + α·y[n-1]
                z = (1.0 - ALPHA) * s + ALPHA * z;
                out.push(z * lf_gain);
            }
            out
        } else {
            // Moderately close: gain only.
            samples.iter().map(|&s| s * lf_gain).collect()
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_clamps_threshold() {
        let proc = NearFieldProcessor::new(-5.0);
        assert!((proc.threshold_m - 0.01).abs() < 1e-6);
    }

    #[test]
    fn test_beyond_threshold_passthrough() {
        let proc = NearFieldProcessor::new(1.0);
        let input: Vec<f32> = (0..64).map(|i| i as f32 * 0.01).collect();
        let output = proc.apply_proximity_eq(2.0, &input);
        assert_eq!(output.len(), input.len());
        for (a, b) in input.iter().zip(output.iter()) {
            assert!((a - b).abs() < 1e-7, "Expected passthrough at dist > threshold");
        }
    }

    #[test]
    fn test_at_threshold_passthrough() {
        let proc = NearFieldProcessor::new(1.0);
        let input = vec![0.5_f32; 32];
        let output = proc.apply_proximity_eq(1.0, &input);
        for v in &output {
            assert!((v - 0.5).abs() < 1e-6, "At threshold should be unchanged");
        }
    }

    #[test]
    fn test_near_field_boosts_gain() {
        let proc = NearFieldProcessor::new(1.0);
        let input = vec![0.1_f32; 64];
        let output = proc.apply_proximity_eq(0.5, &input);
        // ratio = 1.0/0.5 = 2.0 → gain = 2.0
        assert_eq!(output.len(), 64);
        for v in &output {
            assert!(*v > 0.1, "Near-field should boost amplitude");
        }
    }

    #[test]
    fn test_very_close_applies_lowpass() {
        // At 0.05 m, HF damping kicks in; output should differ from pure gain.
        let proc = NearFieldProcessor::new(1.0);
        let impulse = {
            let mut v = vec![0.0_f32; 128];
            v[0] = 1.0;
            v
        };
        let output = proc.apply_proximity_eq(0.05, &impulse);
        assert_eq!(output.len(), 128);
        // Output should be non-zero after the impulse (IIR tail).
        assert!(output[1] > 0.0, "IIR tail should be present after impulse");
    }

    #[test]
    fn test_gain_capped_at_4x() {
        // ratio = 1.0 / 0.001 = 1000 → capped at 4.0.
        let proc = NearFieldProcessor::new(1.0);
        let input = vec![0.1_f32; 16];
        let output = proc.apply_proximity_eq(0.001, &input);
        for v in &output {
            // At 0.001 m the IIR low-pass reduces amplitude; cap is still in effect.
            assert!(*v <= 0.4 * 1.1, "Gain must be capped (≤ 4.0 × input + tolerance)");
        }
    }

    #[test]
    fn test_empty_input() {
        let proc = NearFieldProcessor::new(1.0);
        let output = proc.apply_proximity_eq(0.5, &[]);
        assert!(output.is_empty());
    }
}
