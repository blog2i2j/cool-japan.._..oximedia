#![allow(dead_code)]
//! Tape dropout detection and repair for audio restoration.
//!
//! Magnetic tape recordings suffer from *dropouts* — sudden, short-duration
//! level losses caused by oxide shedding, debris, or head contact issues.
//! Unlike clicks (which are additive impulses), dropouts are *subtractive*:
//! the signal temporarily vanishes or drops dramatically in level.
//!
//! This module detects dropouts by monitoring the instantaneous RMS envelope
//! and identifying anomalous dips.  Detected regions are then repaired using
//! one of several interpolation strategies:
//!
//! - **Linear** — simple linear interpolation across the gap.
//! - **Cubic** — Hermite spline interpolation preserving slope continuity.
//! - **Waveform continuation** — extrapolates from the surrounding waveform
//!   using autocorrelation-based pitch detection and repeats the nearest
//!   clean period.
//!
//! # Typical usage
//!
//! ```ignore
//! use oximedia_restore::tape_dropout_repair::*;
//!
//! let config = DropoutDetectorConfig::default();
//! let mut detector = TapeDropoutDetector::new(config);
//! let dropouts = detector.detect(&samples, 44100)?;
//! let repairer = TapeDropoutRepairer::new(RepairMethod::Cubic);
//! let repaired = repairer.repair(&samples, &dropouts)?;
//! ```

use crate::error::{RestoreError, RestoreResult};

// ---------------------------------------------------------------------------
// Dropout descriptor
// ---------------------------------------------------------------------------

/// A detected tape dropout region.
#[derive(Debug, Clone, PartialEq)]
pub struct TapeDropout {
    /// Start sample index (inclusive).
    pub start: usize,
    /// End sample index (exclusive).
    pub end: usize,
    /// Severity: ratio of the dropout level to the surrounding level.
    /// 0.0 = total silence, 1.0 = no dropout.
    pub severity: f64,
    /// Confidence in the detection (0.0 – 1.0).
    pub confidence: f64,
}

impl TapeDropout {
    /// Length of the dropout in samples.
    #[must_use]
    pub fn len(&self) -> usize {
        if self.end > self.start {
            self.end - self.start
        } else {
            0
        }
    }

    /// Returns `true` when the dropout is zero-length.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Duration in seconds at the given sample rate.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn duration_s(&self, sample_rate: u32) -> f64 {
        self.len() as f64 / f64::from(sample_rate)
    }
}

// ---------------------------------------------------------------------------
// Detection
// ---------------------------------------------------------------------------

/// Configuration for the tape dropout detector.
#[derive(Debug, Clone)]
pub struct DropoutDetectorConfig {
    /// RMS envelope window size in samples.
    pub envelope_window: usize,
    /// Hop size between envelope measurements in samples.
    pub hop_size: usize,
    /// Threshold ratio: a segment whose RMS drops below
    /// `threshold_ratio * surrounding_rms` is flagged.
    pub threshold_ratio: f64,
    /// Minimum dropout duration in samples to report.
    pub min_length: usize,
    /// Maximum dropout duration in samples to report.
    pub max_length: usize,
    /// Number of surrounding windows used to estimate local level.
    pub context_windows: usize,
}

impl Default for DropoutDetectorConfig {
    fn default() -> Self {
        Self {
            envelope_window: 64,
            hop_size: 32,
            threshold_ratio: 0.15,
            min_length: 16,
            max_length: 4410, // 100 ms at 44.1 kHz
            context_windows: 8,
        }
    }
}

/// Tape dropout detector based on RMS envelope analysis.
#[derive(Debug, Clone)]
pub struct TapeDropoutDetector {
    config: DropoutDetectorConfig,
}

impl TapeDropoutDetector {
    /// Create a new detector.
    pub fn new(config: DropoutDetectorConfig) -> Self {
        Self { config }
    }

    /// Create a detector with default parameters.
    pub fn with_defaults() -> Self {
        Self::new(DropoutDetectorConfig::default())
    }

    /// Compute the RMS of a slice.
    #[allow(clippy::cast_precision_loss)]
    fn rms(samples: &[f32]) -> f64 {
        if samples.is_empty() {
            return 0.0;
        }
        let sum_sq: f64 = samples.iter().map(|&s| f64::from(s) * f64::from(s)).sum();
        (sum_sq / samples.len() as f64).sqrt()
    }

    /// Detect dropouts in a mono signal.
    ///
    /// # Errors
    ///
    /// Returns [`RestoreError::InvalidData`] when `samples` is too short for
    /// analysis.
    #[allow(clippy::cast_precision_loss)]
    pub fn detect(&self, samples: &[f32], sample_rate: u32) -> RestoreResult<Vec<TapeDropout>> {
        let _ = sample_rate; // kept for API consistency
        let win = self.config.envelope_window;
        let hop = self.config.hop_size.max(1);

        if samples.len() < win {
            return Ok(Vec::new());
        }

        // Step 1: compute RMS envelope
        let mut envelope: Vec<f64> = Vec::new();
        let mut pos = 0;
        while pos + win <= samples.len() {
            envelope.push(Self::rms(&samples[pos..pos + win]));
            pos += hop;
        }

        if envelope.is_empty() {
            return Ok(Vec::new());
        }

        // Step 2: compute local context level for each window
        let ctx = self.config.context_windows;
        let mut dropouts = Vec::new();
        let mut in_dropout = false;
        let mut dropout_start_win = 0_usize;

        for (i, &env_val) in envelope.iter().enumerate() {
            // Compute average of surrounding windows (excluding current)
            let ctx_start = if i >= ctx { i - ctx } else { 0 };
            let ctx_end = (i + ctx + 1).min(envelope.len());
            let mut ctx_sum = 0.0_f64;
            let mut ctx_count = 0_usize;
            for (j, &v) in envelope[ctx_start..ctx_end].iter().enumerate() {
                if ctx_start + j != i {
                    ctx_sum += v;
                    ctx_count += 1;
                }
            }
            let local_level = if ctx_count > 0 {
                ctx_sum / ctx_count as f64
            } else {
                env_val
            };

            let is_drop = local_level > 1e-8 && env_val < local_level * self.config.threshold_ratio;

            match (in_dropout, is_drop) {
                (false, true) => {
                    in_dropout = true;
                    dropout_start_win = i;
                }
                (true, false) => {
                    in_dropout = false;
                    let start_sample = dropout_start_win * hop;
                    let end_sample = (i * hop + win).min(samples.len());
                    let len = end_sample.saturating_sub(start_sample);
                    if len >= self.config.min_length && len <= self.config.max_length {
                        let severity = if local_level > 1e-8 {
                            (env_val / local_level).clamp(0.0, 1.0)
                        } else {
                            0.0
                        };
                        let confidence = (1.0 - severity).clamp(0.0, 1.0);
                        dropouts.push(TapeDropout {
                            start: start_sample,
                            end: end_sample,
                            severity,
                            confidence,
                        });
                    }
                }
                _ => {}
            }
        }

        // Close any open dropout at the end
        if in_dropout {
            let start_sample = dropout_start_win * hop;
            let end_sample = samples.len();
            let len = end_sample.saturating_sub(start_sample);
            if len >= self.config.min_length && len <= self.config.max_length {
                dropouts.push(TapeDropout {
                    start: start_sample,
                    end: end_sample,
                    severity: 0.0,
                    confidence: 1.0,
                });
            }
        }

        Ok(dropouts)
    }

    /// Return the current configuration.
    pub fn config(&self) -> &DropoutDetectorConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Repair
// ---------------------------------------------------------------------------

/// Interpolation strategy for filling dropout regions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepairMethod {
    /// Simple linear interpolation between boundary samples.
    Linear,
    /// Hermite cubic interpolation using four boundary samples.
    Cubic,
    /// Repeat the nearest clean waveform period from before the dropout.
    WaveformContinuation,
}

/// Tape dropout repairer.
#[derive(Debug, Clone)]
pub struct TapeDropoutRepairer {
    method: RepairMethod,
    /// Number of samples to use as context for waveform continuation.
    context_len: usize,
}

impl TapeDropoutRepairer {
    /// Create a repairer with the given method.
    pub fn new(method: RepairMethod) -> Self {
        Self {
            method,
            context_len: 2048,
        }
    }

    /// Set the context length for waveform continuation.
    pub fn set_context_len(&mut self, len: usize) {
        self.context_len = len.max(64);
    }

    /// Repair all detected dropouts in a mono signal.
    ///
    /// # Errors
    ///
    /// Returns [`RestoreError::InvalidData`] if any dropout region extends
    /// beyond the sample buffer.
    pub fn repair(&self, samples: &[f32], dropouts: &[TapeDropout]) -> RestoreResult<Vec<f32>> {
        let mut output = samples.to_vec();

        for dropout in dropouts {
            if dropout.end > output.len() {
                return Err(RestoreError::InvalidData(format!(
                    "dropout region {}..{} exceeds buffer length {}",
                    dropout.start,
                    dropout.end,
                    output.len()
                )));
            }
            if dropout.is_empty() {
                continue;
            }
            match self.method {
                RepairMethod::Linear => Self::repair_linear(&mut output, dropout),
                RepairMethod::Cubic => Self::repair_cubic(&mut output, dropout),
                RepairMethod::WaveformContinuation => {
                    self.repair_waveform(&mut output, dropout);
                }
            }
        }

        Ok(output)
    }

    /// Linear interpolation across the dropout gap.
    #[allow(clippy::cast_precision_loss)]
    fn repair_linear(output: &mut [f32], dropout: &TapeDropout) {
        let left = if dropout.start > 0 {
            output[dropout.start - 1]
        } else {
            0.0
        };
        let right = if dropout.end < output.len() {
            output[dropout.end]
        } else {
            0.0
        };
        let len = dropout.len() as f64;
        for (i, idx) in (dropout.start..dropout.end).enumerate() {
            let t = (i as f64 + 1.0) / (len + 1.0);
            output[idx] = left + (right - left) * t as f32;
        }
    }

    /// Hermite cubic interpolation across the dropout gap.
    #[allow(clippy::cast_precision_loss)]
    fn repair_cubic(output: &mut [f32], dropout: &TapeDropout) {
        // p0, p1 are left context; p2, p3 are right context
        let p0 = if dropout.start >= 2 {
            f64::from(output[dropout.start - 2])
        } else {
            0.0
        };
        let p1 = if dropout.start >= 1 {
            f64::from(output[dropout.start - 1])
        } else {
            0.0
        };
        let p2 = if dropout.end < output.len() {
            f64::from(output[dropout.end])
        } else {
            0.0
        };
        let p3 = if dropout.end + 1 < output.len() {
            f64::from(output[dropout.end + 1])
        } else {
            p2
        };

        let len = dropout.len() as f64;
        for (i, idx) in (dropout.start..dropout.end).enumerate() {
            let t = (i as f64 + 1.0) / (len + 1.0);
            let val = hermite(p0, p1, p2, p3, t);
            output[idx] = (val as f32).clamp(-1.0, 1.0);
        }
    }

    /// Waveform continuation: copy the nearest clean period before the dropout.
    fn repair_waveform(&self, output: &mut [f32], dropout: &TapeDropout) {
        let gap_len = dropout.len();
        if gap_len == 0 {
            return;
        }

        // Use the context preceding the dropout
        let ctx_start = dropout.start.saturating_sub(self.context_len);
        let ctx_end = dropout.start;
        let ctx_len = ctx_end - ctx_start;

        if ctx_len < gap_len {
            // Not enough context: fall back to linear
            Self::repair_linear(output, dropout);
            return;
        }

        // Simple period estimation via autocorrelation on context
        let period = estimate_period(&output[ctx_start..ctx_end]);
        let fill_period = if period > 0 && period <= ctx_len {
            period
        } else {
            ctx_len
        };

        // Copy cyclically from the context region just before the dropout
        let src_start = ctx_end.saturating_sub(fill_period);
        for (i, idx) in (dropout.start..dropout.end).enumerate() {
            let src_idx = src_start + (i % fill_period);
            output[idx] = output[src_idx];
        }

        // Apply a short cross-fade at the boundaries to avoid clicks
        let fade_len = (gap_len / 8).max(1).min(32);
        // Fade in at start
        if dropout.start > 0 {
            let boundary = output[dropout.start - 1];
            for i in 0..fade_len.min(gap_len) {
                let t = (i + 1) as f32 / (fade_len + 1) as f32;
                let idx = dropout.start + i;
                output[idx] = boundary * (1.0 - t) + output[idx] * t;
            }
        }
        // Fade out at end
        if dropout.end < output.len() {
            let boundary = output[dropout.end];
            for i in 0..fade_len.min(gap_len) {
                let t = (i + 1) as f32 / (fade_len + 1) as f32;
                let idx = dropout.end - 1 - i;
                if idx >= dropout.start {
                    output[idx] = boundary * (1.0 - t) + output[idx] * t;
                }
            }
        }
    }
}

/// Hermite interpolation between p1 and p2, using p0 and p3 as outer points.
fn hermite(p0: f64, p1: f64, p2: f64, p3: f64, t: f64) -> f64 {
    let c0 = p1;
    let c1 = 0.5 * (p2 - p0);
    let c2 = p0 - 2.5 * p1 + 2.0 * p2 - 0.5 * p3;
    let c3 = 0.5 * (p3 - p0) + 1.5 * (p1 - p2);
    ((c3 * t + c2) * t + c1) * t + c0
}

/// Estimate the dominant period in a signal segment using normalised
/// autocorrelation with first-dip detection (finds the fundamental, not a
/// harmonic).
#[allow(clippy::cast_precision_loss)]
fn estimate_period(samples: &[f32]) -> usize {
    let len = samples.len();
    if len < 8 {
        return 0;
    }

    let min_period = 4;
    let max_period = len / 2;

    // Compute energy at lag 0 for normalisation
    let energy: f64 = samples.iter().map(|&s| f64::from(s) * f64::from(s)).sum();
    if energy < 1e-12 {
        return 0;
    }

    // Find the first peak in normalised autocorrelation after the initial dip.
    let mut prev_corr = 1.0_f64;
    let mut found_dip = false;
    let mut best_lag = 0_usize;
    let mut best_corr = f64::NEG_INFINITY;

    for lag in min_period..max_period {
        let mut corr = 0.0_f64;
        let n = len - lag;
        for i in 0..n {
            corr += f64::from(samples[i]) * f64::from(samples[i + lag]);
        }
        let norm_corr = corr / energy;

        if !found_dip {
            if norm_corr < prev_corr {
                // Still descending
            } else {
                found_dip = true;
            }
        }

        if found_dip && norm_corr > best_corr {
            best_corr = norm_corr;
            best_lag = lag;
        }

        // Once we found the first peak after the dip, stop if we start descending
        if found_dip && norm_corr < best_corr - 0.05 {
            break;
        }

        prev_corr = norm_corr;
    }
    best_lag
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Generate a sine wave.
    #[allow(clippy::cast_precision_loss)]
    fn make_sine(freq: f64, sample_rate: u32, len: usize, amp: f32) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let t = i as f64 / f64::from(sample_rate);
                (amp as f64 * (2.0 * PI * freq * t).sin()) as f32
            })
            .collect()
    }

    /// Insert a dropout (zero out a region).
    fn insert_dropout(samples: &mut [f32], start: usize, end: usize) {
        for s in &mut samples[start..end] {
            *s = 0.0;
        }
    }

    #[test]
    fn test_dropout_len() {
        let d = TapeDropout {
            start: 100,
            end: 200,
            severity: 0.0,
            confidence: 1.0,
        };
        assert_eq!(d.len(), 100);
        assert!(!d.is_empty());
    }

    #[test]
    fn test_dropout_duration() {
        let d = TapeDropout {
            start: 0,
            end: 44100,
            severity: 0.0,
            confidence: 1.0,
        };
        assert!((d.duration_s(44100) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_detect_no_dropout_in_silence() {
        let detector = TapeDropoutDetector::with_defaults();
        let silence = vec![0.0_f32; 4096];
        let result = detector.detect(&silence, 44100).expect("ok");
        // Silence is uniform — no anomalous dip — should detect zero dropouts
        assert!(result.is_empty());
    }

    #[test]
    fn test_detect_dropout_in_sine() {
        let mut sine = make_sine(440.0, 44100, 8192, 0.8);
        insert_dropout(&mut sine, 3000, 3200);
        let detector = TapeDropoutDetector::new(DropoutDetectorConfig {
            envelope_window: 32,
            hop_size: 16,
            threshold_ratio: 0.2,
            min_length: 16,
            max_length: 4410,
            context_windows: 8,
        });
        let dropouts = detector.detect(&sine, 44100).expect("ok");
        // Should detect at least one dropout
        assert!(!dropouts.is_empty(), "expected at least one dropout");
        // The detected region should overlap with the injected dropout
        let found = dropouts.iter().any(|d| d.start <= 3200 && d.end >= 3000);
        assert!(found, "detected dropout should overlap injected region");
    }

    #[test]
    fn test_detect_short_signal() {
        let detector = TapeDropoutDetector::with_defaults();
        let short = vec![0.5_f32; 10]; // shorter than envelope_window
        let result = detector.detect(&short, 44100).expect("ok");
        assert!(result.is_empty());
    }

    #[test]
    fn test_repair_linear() {
        let mut signal = vec![1.0_f32; 100];
        insert_dropout(&mut signal, 40, 60);
        let dropout = TapeDropout {
            start: 40,
            end: 60,
            severity: 0.0,
            confidence: 1.0,
        };
        let repairer = TapeDropoutRepairer::new(RepairMethod::Linear);
        let repaired = repairer.repair(&signal, &[dropout]).expect("ok");
        // All samples in the repaired region should be close to 1.0
        for &s in &repaired[40..60] {
            assert!((s - 1.0).abs() < 0.1, "expected ~1.0, got {s}");
        }
    }

    #[test]
    fn test_repair_cubic() {
        let mut signal = vec![0.5_f32; 200];
        insert_dropout(&mut signal, 80, 120);
        let dropout = TapeDropout {
            start: 80,
            end: 120,
            severity: 0.0,
            confidence: 1.0,
        };
        let repairer = TapeDropoutRepairer::new(RepairMethod::Cubic);
        let repaired = repairer.repair(&signal, &[dropout]).expect("ok");
        for &s in &repaired[80..120] {
            assert!((s - 0.5).abs() < 0.2, "expected ~0.5, got {s}");
        }
    }

    #[test]
    fn test_repair_waveform_continuation() {
        let mut sine = make_sine(440.0, 44100, 4096, 0.7);
        insert_dropout(&mut sine, 2000, 2100);
        let dropout = TapeDropout {
            start: 2000,
            end: 2100,
            severity: 0.0,
            confidence: 1.0,
        };
        let repairer = TapeDropoutRepairer::new(RepairMethod::WaveformContinuation);
        let repaired = repairer.repair(&sine, &[dropout]).expect("ok");
        // Repaired region should have non-zero values (waveform continued)
        let energy: f32 = repaired[2000..2100].iter().map(|s| s * s).sum();
        assert!(energy > 0.01, "repaired region should have signal energy");
    }

    #[test]
    fn test_repair_out_of_bounds() {
        let signal = vec![0.5_f32; 100];
        let bad_dropout = TapeDropout {
            start: 50,
            end: 200, // beyond buffer
            severity: 0.0,
            confidence: 1.0,
        };
        let repairer = TapeDropoutRepairer::new(RepairMethod::Linear);
        let result = repairer.repair(&signal, &[bad_dropout]);
        assert!(result.is_err());
    }

    #[test]
    fn test_repair_empty_dropout() {
        let signal = vec![0.5_f32; 100];
        let empty = TapeDropout {
            start: 50,
            end: 50,
            severity: 0.0,
            confidence: 1.0,
        };
        let repairer = TapeDropoutRepairer::new(RepairMethod::Linear);
        let repaired = repairer.repair(&signal, &[empty]).expect("ok");
        assert_eq!(repaired, signal);
    }

    #[test]
    fn test_hermite_interpolation() {
        // At t=0 should return p1, at t=1 should return p2
        let v0 = hermite(0.0, 1.0, 2.0, 3.0, 0.0);
        assert!((v0 - 1.0).abs() < 1e-10);
        let v1 = hermite(0.0, 1.0, 2.0, 3.0, 1.0);
        assert!((v1 - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_estimate_period_sine() {
        let sine = make_sine(440.0, 44100, 2048, 0.8);
        let period = estimate_period(&sine);
        // Expected period ~100 samples (44100/440)
        let expected = 44100.0 / 440.0;
        assert!(
            (period as f64 - expected).abs() < 10.0,
            "estimated period {period}, expected ~{expected}"
        );
    }

    #[test]
    fn test_set_context_len() {
        let mut repairer = TapeDropoutRepairer::new(RepairMethod::WaveformContinuation);
        repairer.set_context_len(4096);
        assert_eq!(repairer.context_len, 4096);
        // Should clamp to minimum 64
        repairer.set_context_len(10);
        assert_eq!(repairer.context_len, 64);
    }
}
