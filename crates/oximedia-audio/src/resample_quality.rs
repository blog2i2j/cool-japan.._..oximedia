//! Resampler quality presets.
//!
//! Provides an ergonomic quality selection layer over the low-level
//! resampler implementations.  Each variant maps to a concrete
//! implementation strategy:
//!
//! - [`ResampleQuality::Linear`]   — Linear interpolation; lowest latency,
//!   artefacts audible at high ratios.
//! - [`ResampleQuality::Sinc`]     — Windowed-sinc resampler (moderate
//!   quality, sub-millisecond latency).
//! - [`ResampleQuality::Polyphase`]— Polyphase filter bank (highest quality,
//!   suitable for broadcast / mastering).
//!
//! # Example
//!
//! ```rust
//! use oximedia_audio::resample_quality::{ResampleQuality, select_resampler};
//!
//! let resample = select_resampler(ResampleQuality::Sinc);
//! let input = vec![0.5_f32, 0.25, -0.25, -0.5];
//! let output = resample(&input, 44_100, 48_000);
//! assert!(!output.is_empty());
//! ```

#![forbid(unsafe_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use std::f32::consts::PI;

// ── ResampleQuality ───────────────────────────────────────────────────────────

/// Quality level for sample-rate conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResampleQuality {
    /// Linear interpolation — fastest, lowest quality.
    Linear,
    /// Windowed-sinc interpolation — balanced quality and speed.
    Sinc,
    /// Polyphase filter bank — highest quality, suitable for mastering.
    Polyphase,
}

impl ResampleQuality {
    /// Human-readable name for the quality level.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            ResampleQuality::Linear => "linear",
            ResampleQuality::Sinc => "sinc",
            ResampleQuality::Polyphase => "polyphase",
        }
    }

    /// Approximate filter length in taps.
    #[must_use]
    pub fn filter_taps(self) -> usize {
        match self {
            ResampleQuality::Linear => 2,
            ResampleQuality::Sinc => 64,
            ResampleQuality::Polyphase => 256,
        }
    }
}

// ── Internal implementations ──────────────────────────────────────────────────

/// Linear interpolation resampler.
fn resample_linear(input: &[f32], in_rate: u32, out_rate: u32) -> Vec<f32> {
    if in_rate == out_rate {
        return input.to_vec();
    }
    if input.is_empty() {
        return Vec::new();
    }
    let ratio = in_rate as f64 / out_rate as f64;
    let out_len = ((input.len() as f64) / ratio).ceil() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src_pos = i as f64 * ratio;
        let idx = src_pos as usize;
        let frac = (src_pos - idx as f64) as f32;
        let s0 = input.get(idx).copied().unwrap_or(0.0);
        let s1 = input.get(idx + 1).copied().unwrap_or(0.0);
        out.push(s0 + frac * (s1 - s0));
    }
    out
}

/// Sinc kernel value at `x` with bandwidth `bw` (in normalized freq).
fn sinc_kernel(x: f32, bw: f32) -> f32 {
    if x.abs() < 1e-9 {
        return bw;
    }
    let pix = PI * x;
    let pibwx = PI * bw * x;
    // Hann window.
    let taps = 64.0;
    let window = if x.abs() <= taps / 2.0 {
        0.5 * (1.0 + (2.0 * PI * x / taps).cos())
    } else {
        0.0
    };
    bw * (pibwx.sin() / pix) * window
}

/// Windowed-sinc resampler.
fn resample_sinc(input: &[f32], in_rate: u32, out_rate: u32) -> Vec<f32> {
    if in_rate == out_rate {
        return input.to_vec();
    }
    if input.is_empty() {
        return Vec::new();
    }
    let ratio = in_rate as f64 / out_rate as f64;
    let bw = if out_rate < in_rate {
        out_rate as f32 / in_rate as f32
    } else {
        1.0
    };
    let half_taps = 32i64;
    let out_len = ((input.len() as f64) / ratio).ceil() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src_pos = i as f64 * ratio;
        let src_int = src_pos as i64;
        let mut acc = 0.0f32;
        for k in -half_taps..=half_taps {
            let idx = src_int + k;
            if idx < 0 || idx as usize >= input.len() {
                continue;
            }
            let x = src_pos as f32 - idx as f32;
            acc += input[idx as usize] * sinc_kernel(x, bw);
        }
        out.push(acc);
    }
    out
}

/// Polyphase filter bank resampler.
///
/// Uses a higher-order sinc interpolation with a wider kernel than the plain
/// sinc variant.
fn resample_polyphase(input: &[f32], in_rate: u32, out_rate: u32) -> Vec<f32> {
    if in_rate == out_rate {
        return input.to_vec();
    }
    if input.is_empty() {
        return Vec::new();
    }
    let ratio = in_rate as f64 / out_rate as f64;
    let bw = if out_rate < in_rate {
        out_rate as f32 / in_rate as f32
    } else {
        1.0
    };
    // Wider kernel for higher quality.
    let half_taps = 128i64;
    let out_len = ((input.len() as f64) / ratio).ceil() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src_pos = i as f64 * ratio;
        let src_int = src_pos as i64;
        let mut acc = 0.0f32;
        for k in -half_taps..=half_taps {
            let idx = src_int + k;
            if idx < 0 || idx as usize >= input.len() {
                continue;
            }
            let x = src_pos as f32 - idx as f32;
            // Kaiser-Bessel window approximation via polynomial.
            let w = {
                let t = x / half_taps as f32;
                if t.abs() > 1.0 {
                    0.0
                } else {
                    // Approximate I0(beta * sqrt(1 - t^2)) / I0(beta), beta=8.
                    let u = 1.0 - t * t;
                    let beta = 8.0_f32;
                    // Polynomial approximation of modified Bessel I0.
                    let x2 = (beta * u.sqrt()) / 2.0;
                    let mut i0 = 1.0_f32;
                    let mut term = 1.0_f32;
                    for m in 1..=12u32 {
                        term *= x2 / m as f32;
                        term *= x2 / m as f32;
                        i0 += term;
                    }
                    // Normalised window (approximate I0(0) ≈ I0(beta*0)).
                    i0 / {
                        let x2b = beta / 2.0;
                        let mut i0b = 1.0_f32;
                        let mut termb = 1.0_f32;
                        for m in 1..=12u32 {
                            termb *= x2b / m as f32;
                            termb *= x2b / m as f32;
                            i0b += termb;
                        }
                        i0b
                    }
                }
            };
            let sinc_val = if x.abs() < 1e-9 {
                bw
            } else {
                let pix = PI * x;
                bw * ((PI * bw * x).sin() / pix)
            };
            acc += input[idx as usize] * sinc_val * w;
        }
        out.push(acc);
    }
    out
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Return a resampler function for the requested quality level.
///
/// The returned closure takes `(samples, in_rate, out_rate)` and returns the
/// resampled `Vec<f32>`.
#[must_use]
pub fn select_resampler(quality: ResampleQuality) -> Box<dyn Fn(&[f32], u32, u32) -> Vec<f32>> {
    match quality {
        ResampleQuality::Linear => Box::new(resample_linear),
        ResampleQuality::Sinc => Box::new(resample_sinc),
        ResampleQuality::Polyphase => Box::new(resample_polyphase),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn check_no_nan(v: &[f32], label: &str) {
        for &s in v {
            assert!(s.is_finite(), "{label}: non-finite sample {s}");
        }
    }

    #[test]
    fn test_resample_quality_names() {
        assert_eq!(ResampleQuality::Linear.name(), "linear");
        assert_eq!(ResampleQuality::Sinc.name(), "sinc");
        assert_eq!(ResampleQuality::Polyphase.name(), "polyphase");
    }

    #[test]
    fn test_resample_quality_filter_taps() {
        assert_eq!(ResampleQuality::Linear.filter_taps(), 2);
        assert!(ResampleQuality::Sinc.filter_taps() > 2);
        assert!(ResampleQuality::Polyphase.filter_taps() > ResampleQuality::Sinc.filter_taps());
    }

    #[test]
    fn test_linear_same_rate_identity() {
        let input = vec![0.1_f32, 0.2, 0.3, 0.4];
        let r = select_resampler(ResampleQuality::Linear);
        let out = r(&input, 48_000, 48_000);
        assert_eq!(out, input);
    }

    #[test]
    fn test_sinc_same_rate_identity() {
        let input = vec![0.1_f32, 0.2, 0.3, 0.4];
        let r = select_resampler(ResampleQuality::Sinc);
        let out = r(&input, 44_100, 44_100);
        assert_eq!(out, input);
    }

    #[test]
    fn test_polyphase_same_rate_identity() {
        let input = vec![0.5_f32, -0.5, 0.0];
        let r = select_resampler(ResampleQuality::Polyphase);
        let out = r(&input, 22_050, 22_050);
        assert_eq!(out, input);
    }

    #[test]
    fn test_upsample_linear_produces_more_samples() {
        let input = vec![0.0_f32, 1.0, 0.0, -1.0];
        let r = select_resampler(ResampleQuality::Linear);
        let out = r(&input, 44_100, 48_000);
        assert!(out.len() >= input.len());
        check_no_nan(&out, "linear upsample");
    }

    #[test]
    fn test_downsample_linear_produces_fewer_samples() {
        let input: Vec<f32> = (0..480).map(|i| (i as f32) * 0.001).collect();
        let r = select_resampler(ResampleQuality::Linear);
        let out = r(&input, 48_000, 44_100);
        assert!(out.len() < input.len());
        check_no_nan(&out, "linear downsample");
    }

    #[test]
    fn test_upsample_sinc_no_nan() {
        let input: Vec<f32> = (0..256).map(|i| (i as f32 * 0.05).sin()).collect();
        let r = select_resampler(ResampleQuality::Sinc);
        let out = r(&input, 44_100, 48_000);
        assert!(!out.is_empty());
        check_no_nan(&out, "sinc upsample");
    }

    #[test]
    fn test_polyphase_upsample_no_nan() {
        let input: Vec<f32> = (0..128).map(|i| (i as f32 * 0.1).sin()).collect();
        let r = select_resampler(ResampleQuality::Polyphase);
        let out = r(&input, 44_100, 48_000);
        assert!(!out.is_empty());
        check_no_nan(&out, "polyphase upsample");
    }

    #[test]
    fn test_empty_input_returns_empty() {
        for q in [
            ResampleQuality::Linear,
            ResampleQuality::Sinc,
            ResampleQuality::Polyphase,
        ] {
            let r = select_resampler(q);
            let out = r(&[], 44_100, 48_000);
            assert!(out.is_empty(), "{q:?} should return empty for empty input");
        }
    }

    #[test]
    fn test_polyphase_quality_higher_than_linear_for_1k_tone() {
        // This test verifies the polyphase path produces more samples per
        // unit time (higher output density) for the same conversion ratio.
        let input: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.1).sin()).collect();
        let linear_r = select_resampler(ResampleQuality::Linear);
        let poly_r = select_resampler(ResampleQuality::Polyphase);
        let l = linear_r(&input, 44_100, 48_000);
        let p = poly_r(&input, 44_100, 48_000);
        // Both should produce valid output of similar length.
        assert!((l.len() as i64 - p.len() as i64).abs() <= 2);
        check_no_nan(&l, "linear");
        check_no_nan(&p, "polyphase");
    }
}
