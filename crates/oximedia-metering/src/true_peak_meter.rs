//! True peak metering with 4x oversampling via cubic interpolation.
//!
//! `TruePeakMeter` provides a self-contained, stateless measurement of the
//! inter-sample peak level in a block of audio samples.
//!
//! ## Algorithm
//!
//! The signal is virtually upsampled 4× by inserting three interpolated
//! samples between every pair of original samples using **cubic (Catmull-Rom)**
//! interpolation.  The peak of the upsampled signal is the "true peak" — the
//! maximum amplitude that would appear after digital-to-analogue conversion.
//!
//! Catmull-Rom spline between `p1` and `p2` with parameter `t ∈ (0, 1)`:
//!
//! ```text
//! q(t) = 0.5 · [ (2p1)
//!              + (-p0 + p2)·t
//!              + (2p0 - 5p1 + 4p2 - p3)·t²
//!              + (-p0 + 3p1 - 3p2 + p3)·t³ ]
//! ```
//!
//! ## Example
//!
//! ```rust
//! use oximedia_metering::true_peak_meter::TruePeakMeter;
//!
//! let meter = TruePeakMeter::new();
//! let samples: Vec<f32> = (0..1024)
//!     .map(|i| (2.0 * std::f32::consts::PI * 997.0 * i as f32 / 48000.0).sin())
//!     .collect();
//! let peak = meter.measure(&samples);
//! assert!(peak >= 0.99 && peak <= 1.05, "997 Hz sine peak ≈ 1.0, got {peak}");
//! ```

/// True peak meter with 4× cubic oversampling.
#[derive(Debug, Clone, Default)]
pub struct TruePeakMeter;

impl TruePeakMeter {
    /// Create a new `TruePeakMeter`.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Measure the true peak of `samples` using 4× cubic interpolation.
    ///
    /// Returns the maximum absolute amplitude found in the oversampled signal.
    /// Returns `0.0` if `samples` is empty.
    #[must_use]
    pub fn measure(&self, samples: &[f32]) -> f32 {
        let n = samples.len();
        if n == 0 {
            return 0.0;
        }

        // Catmull-Rom helper: interpolate between p1 and p2 at t ∈ (0, 1).
        let catmull_rom = |p0: f32, p1: f32, p2: f32, p3: f32, t: f32| -> f32 {
            let t2 = t * t;
            let t3 = t2 * t;
            0.5 * ((2.0 * p1)
                + (-p0 + p2) * t
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)
        };

        // Helper to get a sample with reflect-boundary padding.
        let s = |i: isize| -> f32 {
            if i < 0 {
                samples[0]
            } else if i >= n as isize {
                samples[n - 1]
            } else {
                samples[i as usize]
            }
        };

        let mut peak = 0.0_f32;

        for i in 0..n {
            let ii = i as isize;
            let p0 = s(ii - 1);
            let p1 = s(ii);
            let p2 = s(ii + 1);
            let p3 = s(ii + 2);

            // Original sample.
            peak = peak.max(p1.abs());

            // Three interpolated sub-samples at t = 0.25, 0.50, 0.75.
            for &t in &[0.25_f32, 0.50, 0.75] {
                let v = catmull_rom(p0, p1, p2, p3, t);
                peak = peak.max(v.abs());
            }
        }

        peak
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_input_returns_zero() {
        let meter = TruePeakMeter::new();
        assert_eq!(meter.measure(&[]), 0.0);
    }

    #[test]
    fn test_dc_signal() {
        let meter = TruePeakMeter::new();
        let dc = vec![0.5_f32; 256];
        let peak = meter.measure(&dc);
        assert!(
            (peak - 0.5).abs() < 0.01,
            "DC signal peak should ≈ 0.5, got {peak}"
        );
    }

    #[test]
    fn test_sine_997hz_peak_near_one() {
        let meter = TruePeakMeter::new();
        let samples: Vec<f32> = (0..4096)
            .map(|i| (2.0 * std::f32::consts::PI * 997.0 * i as f32 / 48000.0).sin())
            .collect();
        let peak = meter.measure(&samples);
        assert!(
            peak >= 0.99 && peak <= 1.05,
            "997 Hz sine at 0 dBFS should have true peak ≈ 1.0, got {peak}"
        );
    }

    #[test]
    fn test_peak_at_least_as_large_as_sample_peak() {
        let meter = TruePeakMeter::new();
        let samples: Vec<f32> = (0..512)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48000.0).sin() * 0.8)
            .collect();
        let sample_peak = samples.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
        let true_peak = meter.measure(&samples);
        assert!(
            true_peak >= sample_peak,
            "True peak ({true_peak}) must be ≥ sample peak ({sample_peak})"
        );
    }

    #[test]
    fn test_single_impulse() {
        let meter = TruePeakMeter::new();
        let mut samples = vec![0.0_f32; 64];
        samples[32] = 0.9;
        let peak = meter.measure(&samples);
        assert!(
            peak >= 0.9,
            "Peak must capture the impulse, got {peak}"
        );
    }

    #[test]
    fn test_negative_impulse() {
        let meter = TruePeakMeter::new();
        let mut samples = vec![0.0_f32; 64];
        samples[16] = -0.7;
        let peak = meter.measure(&samples);
        assert!(
            peak >= 0.7,
            "Absolute peak of negative impulse should be ≥ 0.7, got {peak}"
        );
    }
}
