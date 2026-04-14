//! Click detection and repair for audio restoration.
//!
//! Clicks and pops in digitised vinyl or tape recordings appear as isolated
//! samples with anomalously large magnitude.  This module provides:
//!
//! * [`Declicker::detect_clicks`] — identifies click positions using an
//!   adaptive threshold based on the local RMS level.
//! * [`Declicker::repair_click`] — replaces a detected click with a smooth
//!   interpolated segment derived from the surrounding context.
//!
//! # Algorithm
//!
//! **Detection** — A click at position *p* is declared when
//! `|audio[p]| > threshold × local_rms(p)`.  The local RMS is computed over
//! a window of ±`context_radius` samples centred on *p*.
//!
//! **Repair** — Samples in `[pos − radius, pos + radius]` are replaced with a
//! cubic Hermite spline that interpolates between the two boundary samples and
//! their first derivatives (estimated via finite differences).  The result is
//! a smooth, artefact-free waveform.
//!
//! # Example
//!
//! ```
//! use oximedia_restore::declicker::Declicker;
//! use std::f32::consts::PI;
//!
//! let dc = Declicker::default();
//! // Build a 128-sample sine wave, then inject a loud spike at sample 64.
//! let mut audio: Vec<f32> = (0..128)
//!     .map(|i| (2.0 * PI * 440.0 * i as f32 / 44100.0).sin() * 0.5)
//!     .collect();
//! audio[64] = 10.0; // simulate a loud click against a non-silent background
//!
//! let clicks = dc.detect_clicks(&audio, 5.0);
//! assert!(clicks.contains(&64));
//!
//! dc.repair_click(&mut audio, 64, 8);
//! // The click should be smoothed out
//! assert!(audio[64].abs() < 1.0);
//! ```

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Click detector and repairer.
#[derive(Debug, Clone)]
pub struct Declicker {
    /// Number of samples on each side used to compute the local RMS.
    pub context_radius: usize,
}

impl Default for Declicker {
    fn default() -> Self {
        Self { context_radius: 32 }
    }
}

impl Declicker {
    /// Create a `Declicker` with the specified context radius.
    pub fn new(context_radius: usize) -> Self {
        Self {
            context_radius: context_radius.max(1),
        }
    }

    /// Detect click positions in `audio`.
    ///
    /// A sample at index *i* is classified as a click when:
    ///
    /// ```text
    /// |audio[i]| > threshold × local_rms(i)
    /// ```
    ///
    /// where `local_rms(i)` is the RMS of a window of ±`context_radius`
    /// samples (excluding the candidate sample itself to avoid self-bias).
    ///
    /// # Parameters
    ///
    /// * `audio`     — mono audio samples.
    /// * `threshold` — multiplier applied to the local RMS; values in 3–10
    ///                 are typical for vinyl restoration.
    ///
    /// # Returns
    ///
    /// A sorted `Vec<usize>` of click sample indices.
    pub fn detect_clicks(&self, audio: &[f32], threshold: f32) -> Vec<usize> {
        if audio.is_empty() || threshold <= 0.0 {
            return Vec::new();
        }

        let n = audio.len();
        let r = self.context_radius;
        let mut clicks = Vec::new();

        for i in 0..n {
            let lo = i.saturating_sub(r);
            let hi = (i + r + 1).min(n);

            // Compute RMS of context window, excluding sample i.
            let mut sum_sq = 0.0f64;
            let mut count = 0usize;
            for j in lo..hi {
                if j != i {
                    let v = audio[j] as f64;
                    sum_sq += v * v;
                    count += 1;
                }
            }

            if count == 0 {
                continue;
            }

            let rms = (sum_sq / count as f64).sqrt() as f32;
            // Avoid false positives in pure-silence regions.
            if rms < 1e-9 {
                continue;
            }

            if audio[i].abs() > threshold * rms {
                clicks.push(i);
            }
        }

        clicks
    }

    /// Repair a click at `pos` by replacing `[pos−radius, pos+radius]` with
    /// a cubic Hermite spline interpolated from the surrounding context.
    ///
    /// If `pos` is too close to either end of `audio` for a full repair
    /// window, the window is silently clamped.
    ///
    /// # Parameters
    ///
    /// * `audio`  — mutable audio sample buffer (modified in-place).
    /// * `pos`    — click position to repair.
    /// * `radius` — half-width of the repair window in samples.
    pub fn repair_click(&self, audio: &mut [f32], pos: usize, radius: usize) {
        let n = audio.len();
        if n < 2 || radius == 0 {
            return;
        }

        // Determine the repair region [start, end] (inclusive).
        let start = pos.saturating_sub(radius);
        let end = (pos + radius).min(n - 1);

        if start >= end {
            return;
        }

        // Boundary values and finite-difference tangents.
        let v0 = audio[start] as f64;
        let v1 = audio[end] as f64;

        // Finite-difference tangent at start.
        let t0 = if start > 0 {
            (audio[start + 1] as f64 - audio[start - 1] as f64) * 0.5
        } else {
            0.0
        };

        // Finite-difference tangent at end.
        let t1 = if end + 1 < n {
            (audio[end + 1] as f64 - audio[end - 1] as f64) * 0.5
        } else {
            0.0
        };

        let len = (end - start) as f64;

        // Interpolate using cubic Hermite spline.
        for i in start..=end {
            let t = (i - start) as f64 / len;
            let h00 = 2.0 * t * t * t - 3.0 * t * t + 1.0;
            let h10 = t * t * t - 2.0 * t * t + t;
            let h01 = -2.0 * t * t * t + 3.0 * t * t;
            let h11 = t * t * t - t * t;
            let interp = h00 * v0 + h10 * len * t0 + h01 * v1 + h11 * len * t1;
            audio[i] = interp.clamp(-1.0, 1.0) as f32;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a sine wave with a click injected at `click_pos`.
    fn sine_with_click(n: usize, click_pos: usize, click_amp: f32) -> Vec<f32> {
        let mut audio: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * i as f32 / 64.0).sin() * 0.1)
            .collect();
        if click_pos < n {
            audio[click_pos] = click_amp;
        }
        audio
    }

    #[test]
    fn test_detect_click_in_sine() {
        let dc = Declicker::new(32);
        let audio = sine_with_click(256, 128, 5.0); // loud click at 128
        let clicks = dc.detect_clicks(&audio, 4.0);
        assert!(
            clicks.contains(&128),
            "click at 128 should be detected: {clicks:?}"
        );
    }

    #[test]
    fn test_detect_no_click_in_silence() {
        let dc = Declicker::default();
        let audio = vec![0.0f32; 256];
        let clicks = dc.detect_clicks(&audio, 5.0);
        assert!(clicks.is_empty(), "silence should contain no clicks");
    }

    #[test]
    fn test_detect_empty_audio() {
        let dc = Declicker::default();
        let clicks = dc.detect_clicks(&[], 5.0);
        assert!(clicks.is_empty());
    }

    #[test]
    fn test_detect_zero_threshold_returns_empty() {
        let dc = Declicker::default();
        let audio = vec![0.5f32; 64];
        let clicks = dc.detect_clicks(&audio, 0.0);
        assert!(clicks.is_empty());
    }

    #[test]
    fn test_repair_reduces_click_amplitude() {
        let dc = Declicker::default();
        let mut audio = sine_with_click(256, 128, 5.0);
        let before = audio[128].abs();
        dc.repair_click(&mut audio, 128, 8);
        let after = audio[128].abs();
        assert!(after < before, "repair should reduce click amplitude");
        assert!(
            after <= 1.0,
            "repaired sample must be within [-1, 1]: {after}"
        );
    }

    #[test]
    fn test_repair_near_start_no_panic() {
        let dc = Declicker::default();
        let mut audio = vec![0.0f32; 64];
        audio[2] = 5.0;
        dc.repair_click(&mut audio, 2, 8); // window clips at start
    }

    #[test]
    fn test_repair_near_end_no_panic() {
        let dc = Declicker::default();
        let mut audio = vec![0.0f32; 64];
        audio[62] = 5.0;
        dc.repair_click(&mut audio, 62, 8); // window clips at end
    }

    #[test]
    fn test_repair_zero_radius_no_change() {
        let dc = Declicker::default();
        let mut audio = vec![1.0f32; 16];
        audio[8] = 5.0;
        dc.repair_click(&mut audio, 8, 0); // nothing should happen
        assert!(
            (audio[8] - 5.0).abs() < 1e-6,
            "radius=0 should leave audio unchanged"
        );
    }

    #[test]
    fn test_detect_returns_sorted() {
        let dc = Declicker::new(4);
        // Two clicks far apart
        let mut audio = vec![0.01f32; 200];
        audio[50] = 5.0;
        audio[150] = 5.0;
        let clicks = dc.detect_clicks(&audio, 3.0);
        let is_sorted = clicks.windows(2).all(|w| w[0] <= w[1]);
        assert!(is_sorted, "detected clicks should be in ascending order");
    }
}
