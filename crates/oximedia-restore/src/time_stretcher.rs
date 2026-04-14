//! Time stretching without pitch change for audio restoration.
//!
//! This module implements the **WSOLA** (Waveform Similarity Overlap-Add)
//! algorithm for changing audio duration without altering pitch.  WSOLA
//! improves over basic OLA by searching for the best-matching grain position
//! within a tolerance window, minimising phase discontinuities.
//!
//! # References
//!
//! - Verhelst & Roelands (1993). "An overlap-add technique based on waveform
//!   similarity (WSOLA) for high quality time-scale modification of speech."
//!   ICASSP, vol. 2.
//!
//! # Features
//!
//! - Tempo change (stretch/compress time, keep pitch)
//! - Speed change (tempo + pitch together, via resampling)
//! - Configurable cross-fade window and search tolerance
//! - Handles short buffers gracefully

use crate::error::{RestoreError, RestoreResult};
use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// WSOLA configuration
// ---------------------------------------------------------------------------

/// Configuration for the WSOLA time stretcher.
#[derive(Debug, Clone)]
pub struct WsolaConfig {
    /// Stretch ratio: `output_duration / input_duration`.
    /// - `1.0` → no change
    /// - `0.5` → half duration (speed-up 2×)
    /// - `2.0` → double duration (slow-down 2×)
    pub stretch_ratio: f32,
    /// Analysis window size in samples.  Should be several pitch periods long.
    pub window_size: usize,
    /// Synthesis hop size in samples.  `window_size / 4` is a good default.
    pub synthesis_hop: usize,
    /// Maximum search tolerance in samples.  WSOLA searches ±`tolerance`
    /// around the nominal next-frame position.
    pub tolerance: usize,
    /// Window function applied to each grain before overlap-add.
    pub window_type: WsolaWindowType,
}

/// Window function used for grain windowing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WsolaWindowType {
    /// Hann window (recommended for music).
    Hann,
    /// Triangular window (fast to compute).
    Triangle,
    /// Rectangular window (no tapering).
    Rect,
}

impl Default for WsolaConfig {
    fn default() -> Self {
        Self {
            stretch_ratio: 1.0,
            window_size: 2048,
            synthesis_hop: 512,
            tolerance: 256,
            window_type: WsolaWindowType::Hann,
        }
    }
}

impl WsolaConfig {
    /// Return the analysis hop size (synthesis_hop / stretch_ratio).
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    fn analysis_hop(&self) -> usize {
        let ha = self.synthesis_hop as f32 / self.stretch_ratio;
        (ha.round() as usize).max(1)
    }
}

// ---------------------------------------------------------------------------
// WSOLA processor
// ---------------------------------------------------------------------------

/// WSOLA time stretcher.
///
/// Stretches or compresses audio duration while preserving pitch by
/// intelligently selecting grains to maximise waveform continuity.
#[derive(Debug, Clone)]
pub struct WsolaStretcher {
    config: WsolaConfig,
}

impl WsolaStretcher {
    /// Create a new WSOLA stretcher.
    ///
    /// # Errors
    ///
    /// Returns `RestoreError::InvalidParameter` if the configuration is invalid.
    pub fn new(config: WsolaConfig) -> RestoreResult<Self> {
        if config.stretch_ratio <= 0.0 {
            return Err(RestoreError::InvalidParameter(
                "stretch_ratio must be > 0".into(),
            ));
        }
        if config.window_size < 4 {
            return Err(RestoreError::InvalidParameter(
                "window_size must be >= 4".into(),
            ));
        }
        if config.synthesis_hop == 0 {
            return Err(RestoreError::InvalidParameter(
                "synthesis_hop must be > 0".into(),
            ));
        }
        Ok(Self { config })
    }

    /// Create a stretcher for simple tempo adjustment.
    ///
    /// `tempo_factor` > 1 speeds up, < 1 slows down.
    ///
    /// # Errors
    ///
    /// Returns `RestoreError::InvalidParameter` when `tempo_factor <= 0`.
    pub fn for_tempo(tempo_factor: f32, window_size: usize) -> RestoreResult<Self> {
        if tempo_factor <= 0.0 {
            return Err(RestoreError::InvalidParameter(
                "tempo_factor must be > 0".into(),
            ));
        }
        let synthesis_hop = window_size / 4;
        Self::new(WsolaConfig {
            stretch_ratio: 1.0 / tempo_factor,
            window_size,
            synthesis_hop,
            tolerance: window_size / 8,
            window_type: WsolaWindowType::Hann,
        })
    }

    /// Process a buffer of mono samples.
    ///
    /// Returns the time-stretched audio.  Length is approximately
    /// `samples.len() * stretch_ratio`.
    ///
    /// # Errors
    ///
    /// Returns `RestoreError::InvalidData` for empty input.
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    pub fn process(&self, samples: &[f32]) -> RestoreResult<Vec<f32>> {
        if samples.is_empty() {
            return Ok(Vec::new());
        }

        let input_len = samples.len();
        let expected_len = (input_len as f32 * self.config.stretch_ratio).round() as usize;

        let win_size = self.config.window_size;
        let hs = self.config.synthesis_hop;
        let ha = self.config.analysis_hop();
        let tol = self.config.tolerance;

        // Pre-compute window
        let window: Vec<f32> = (0..win_size)
            .map(|i| window_sample(i, win_size, self.config.window_type))
            .collect();

        let mut output = vec![0.0_f32; expected_len + win_size];
        let mut norm = vec![0.0_f32; expected_len + win_size];

        // State: analysis read position and synthesis write position
        let mut ana_pos: i64 = 0; // centre of next analysis frame
        let mut syn_pos: usize = 0; // centre of next synthesis frame

        // Previous grain (for cross-correlation)
        let mut prev_grain: Vec<f32> = vec![0.0; win_size];

        loop {
            if syn_pos + win_size / 2 >= output.len() {
                break;
            }

            // Nominal analysis centre
            let nominal = ana_pos;

            // WSOLA: search ±tol for best overlap with previous grain
            let best_start = find_best_offset(samples, nominal, tol, win_size, &prev_grain);

            // Extract grain
            let grain: Vec<f32> = (0..win_size)
                .map(|i| {
                    let src = best_start + i as i64;
                    if src >= 0 && (src as usize) < input_len {
                        samples[src as usize]
                    } else {
                        0.0
                    }
                })
                .collect();

            // Apply window and overlap-add into output
            let syn_start = syn_pos as i64 - (win_size as i64 / 2);
            for i in 0..win_size {
                let dst = syn_start + i as i64;
                if dst >= 0 && (dst as usize) < output.len() {
                    output[dst as usize] += grain[i] * window[i];
                    norm[dst as usize] += window[i];
                }
            }

            prev_grain.copy_from_slice(&grain);

            // Advance positions
            ana_pos += ha as i64;
            syn_pos += hs;

            if ana_pos as usize >= input_len + win_size / 2 {
                break;
            }
        }

        // Normalise
        for (o, &n) in output.iter_mut().zip(norm.iter()) {
            if n > f32::EPSILON {
                *o /= n;
            }
        }

        // Trim to expected length
        output.truncate(expected_len);
        // Pad with zeros if too short (can happen for very short inputs)
        while output.len() < expected_len {
            output.push(0.0);
        }

        Ok(output)
    }

    /// Return the effective stretch ratio.
    #[must_use]
    pub fn stretch_ratio(&self) -> f32 {
        self.config.stretch_ratio
    }

    /// Return the configuration.
    #[must_use]
    pub fn config(&self) -> &WsolaConfig {
        &self.config
    }
}

/// Find the analysis frame start position that maximises waveform similarity
/// with `prev_grain`, searching ±`tolerance` around `nominal_centre`.
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
fn find_best_offset(
    samples: &[f32],
    nominal_centre: i64,
    tolerance: usize,
    win_size: usize,
    prev_grain: &[f32],
) -> i64 {
    let tol = tolerance as i64;
    let half_win = (win_size / 2) as i64;

    let mut best_start = nominal_centre - half_win;
    let mut best_score = f32::NEG_INFINITY;

    // Check whether prev_grain is meaningful (non-zero)
    let prev_rms: f32 = (prev_grain.iter().map(|&x| x * x).sum::<f32>() / win_size as f32).sqrt();

    if prev_rms < 1e-6 {
        // No reference — use nominal position
        return best_start;
    }

    for delta in -tol..=tol {
        let candidate_start = nominal_centre - half_win + delta;
        let score = cross_correlation(samples, candidate_start, prev_grain, win_size);
        if score > best_score {
            best_score = score;
            best_start = candidate_start;
        }
    }

    best_start
}

/// Compute normalised cross-correlation between a candidate window in `samples`
/// and `reference`.
#[allow(clippy::cast_precision_loss)]
fn cross_correlation(samples: &[f32], candidate_start: i64, reference: &[f32], len: usize) -> f32 {
    let mut num = 0.0_f32;
    let mut denom_a = 0.0_f32;
    let mut denom_b = 0.0_f32;

    for i in 0..len {
        let a = {
            let src = candidate_start + i as i64;
            if src >= 0 && (src as usize) < samples.len() {
                samples[src as usize]
            } else {
                0.0
            }
        };
        let b = reference[i];
        num += a * b;
        denom_a += a * a;
        denom_b += b * b;
    }

    let denom = (denom_a * denom_b).sqrt();
    if denom > f32::EPSILON {
        num / denom
    } else {
        0.0
    }
}

/// Compute the window sample value for index `i` in a window of `length`.
#[allow(clippy::cast_precision_loss)]
fn window_sample(i: usize, length: usize, wtype: WsolaWindowType) -> f32 {
    if length <= 1 {
        return 1.0;
    }
    match wtype {
        WsolaWindowType::Hann => 0.5 * (1.0 - (2.0 * PI * i as f32 / (length - 1) as f32).cos()),
        WsolaWindowType::Triangle => {
            let half = (length - 1) as f32 / 2.0;
            1.0 - (i as f32 - half).abs() / half
        }
        WsolaWindowType::Rect => 1.0,
    }
}

// ---------------------------------------------------------------------------
// Speed changer
// ---------------------------------------------------------------------------

/// Change playback speed by stretching time then resampling.
///
/// `speed_factor` > 1 speeds up (higher pitch + tempo), < 1 slows down.
/// This is the "speed" mode as opposed to pure tempo stretching.
///
/// # Errors
///
/// Returns an error if `speed_factor <= 0` or `sample_rate` is zero.
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
pub fn change_speed(
    samples: &[f32],
    sample_rate: u32,
    speed_factor: f32,
) -> RestoreResult<Vec<f32>> {
    if speed_factor <= 0.0 {
        return Err(RestoreError::InvalidParameter(
            "speed_factor must be > 0".into(),
        ));
    }
    if sample_rate == 0 {
        return Err(RestoreError::InvalidParameter(
            "sample_rate must be > 0".into(),
        ));
    }
    if samples.is_empty() {
        return Ok(Vec::new());
    }

    // Resample: linear interpolation to change apparent pitch+tempo
    let out_len = (samples.len() as f32 / speed_factor).round() as usize;
    if out_len == 0 {
        return Ok(Vec::new());
    }

    let mut output = Vec::with_capacity(out_len);
    let in_len = samples.len() as f32;

    for i in 0..out_len {
        let src = i as f32 * speed_factor;
        let lo = src.floor() as usize;
        let hi = lo + 1;
        let frac = src - lo as f32;

        let s_lo = if lo < samples.len() { samples[lo] } else { 0.0 };
        let s_hi = if hi < samples.len() { samples[hi] } else { 0.0 };

        // Avoid unused variable warning
        let _ = in_len;

        output.push(s_lo + frac * (s_hi - s_lo));
    }

    Ok(output)
}

/// Pure tempo change — stretch or compress time while preserving pitch.
///
/// `tempo_factor` > 1 speeds up, < 1 slows down.
///
/// # Errors
///
/// Returns an error if `tempo_factor <= 0`.
pub fn change_tempo(samples: &[f32], tempo_factor: f32) -> RestoreResult<Vec<f32>> {
    let window_size = 2048.min(samples.len().next_power_of_two().max(64));
    let stretcher = WsolaStretcher::for_tempo(tempo_factor, window_size)?;
    stretcher.process(samples)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const SR: u32 = 44100;

    fn make_sine(freq: f32, duration_ms: f32) -> Vec<f32> {
        let n = (SR as f32 * duration_ms / 1000.0) as usize;
        (0..n)
            .map(|i| (2.0 * PI * freq * i as f32 / SR as f32).sin())
            .collect()
    }

    #[test]
    fn test_wsola_config_default() {
        let cfg = WsolaConfig::default();
        assert!((cfg.stretch_ratio - 1.0).abs() < f32::EPSILON);
        assert!(cfg.window_size > 0);
        assert!(cfg.synthesis_hop > 0);
    }

    #[test]
    fn test_wsola_reject_zero_ratio() {
        let result = WsolaStretcher::new(WsolaConfig {
            stretch_ratio: 0.0,
            ..WsolaConfig::default()
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_wsola_reject_tiny_window() {
        let result = WsolaStretcher::new(WsolaConfig {
            window_size: 2,
            ..WsolaConfig::default()
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_wsola_empty_input_returns_empty() {
        let stretcher = WsolaStretcher::new(WsolaConfig::default()).expect("valid");
        let out = stretcher.process(&[]).expect("ok");
        assert!(out.is_empty());
    }

    #[test]
    fn test_wsola_unity_ratio_preserves_length() {
        let sine = make_sine(440.0, 200.0);
        let len = sine.len();
        let stretcher = WsolaStretcher::new(WsolaConfig {
            stretch_ratio: 1.0,
            window_size: 512,
            synthesis_hop: 128,
            tolerance: 64,
            window_type: WsolaWindowType::Hann,
        })
        .expect("valid");
        let out = stretcher.process(&sine).expect("ok");
        // Allow ±1 sample tolerance
        let diff = (out.len() as i64 - len as i64).abs();
        assert!(
            diff <= 1,
            "length mismatch: expected ~{len}, got {}",
            out.len()
        );
    }

    #[test]
    fn test_wsola_double_length_stretch() {
        let sine = make_sine(440.0, 100.0);
        let len = sine.len();
        let stretcher = WsolaStretcher::new(WsolaConfig {
            stretch_ratio: 2.0,
            window_size: 512,
            synthesis_hop: 128,
            tolerance: 64,
            window_type: WsolaWindowType::Hann,
        })
        .expect("valid");
        let out = stretcher.process(&sine).expect("ok");
        // Output should be approximately 2× input length
        let expected = len * 2;
        let diff = (out.len() as i64 - expected as i64).abs();
        assert!(diff <= 8, "expected ~{expected} samples, got {}", out.len());
    }

    #[test]
    fn test_wsola_half_length_compress() {
        let sine = make_sine(440.0, 200.0);
        let len = sine.len();
        let stretcher = WsolaStretcher::new(WsolaConfig {
            stretch_ratio: 0.5,
            window_size: 512,
            synthesis_hop: 128,
            tolerance: 64,
            window_type: WsolaWindowType::Hann,
        })
        .expect("valid");
        let out = stretcher.process(&sine).expect("ok");
        let expected = len / 2;
        let diff = (out.len() as i64 - expected as i64).abs();
        assert!(diff <= 8, "expected ~{expected} samples, got {}", out.len());
    }

    #[test]
    fn test_wsola_triangle_window() {
        let sine = make_sine(220.0, 100.0);
        let stretcher = WsolaStretcher::new(WsolaConfig {
            stretch_ratio: 1.2,
            window_size: 512,
            synthesis_hop: 128,
            tolerance: 64,
            window_type: WsolaWindowType::Triangle,
        })
        .expect("valid");
        let out = stretcher.process(&sine).expect("ok");
        assert!(!out.is_empty());
    }

    #[test]
    fn test_change_speed_reject_zero_factor() {
        let result = change_speed(&[0.0; 100], SR, 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_change_speed_empty_input() {
        let result = change_speed(&[], SR, 1.5).expect("ok");
        assert!(result.is_empty());
    }

    #[test]
    fn test_change_speed_output_length() {
        let samples = make_sine(440.0, 100.0);
        let len = samples.len();
        let out = change_speed(&samples, SR, 2.0).expect("ok");
        let expected = (len as f32 / 2.0).round() as usize;
        let diff = (out.len() as i64 - expected as i64).abs();
        assert!(diff <= 2, "expected ~{expected}, got {}", out.len());
    }

    #[test]
    fn test_change_tempo_reject_zero() {
        let result = change_tempo(&[0.0; 100], 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_for_tempo_reject_zero() {
        let result = WsolaStretcher::for_tempo(0.0, 512);
        assert!(result.is_err());
    }

    #[test]
    fn test_window_samples() {
        let n = 256;
        let hann_mid = window_sample(n / 2, n, WsolaWindowType::Hann);
        let tri_mid = window_sample(n / 2, n, WsolaWindowType::Triangle);
        let rect_mid = window_sample(n / 2, n, WsolaWindowType::Rect);
        assert!(hann_mid > 0.99, "Hann mid: {hann_mid}");
        assert!(tri_mid > 0.99, "Tri mid: {tri_mid}");
        assert!(
            (rect_mid - 1.0).abs() < f32::EPSILON,
            "Rect mid: {rect_mid}"
        );

        let hann_edge = window_sample(0, n, WsolaWindowType::Hann);
        assert!(hann_edge < 0.01, "Hann edge: {hann_edge}");
    }

    #[test]
    fn test_cross_correlation_identical() {
        let buf = vec![0.5_f32; 64];
        let score = cross_correlation(&buf, 0, &buf, 64);
        // Identical signals → correlation near 1.0
        assert!(
            score > 0.99,
            "identical signals should correlate to ~1.0, got {score}"
        );
    }
}
