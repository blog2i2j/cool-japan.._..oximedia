//! Pitch correction for audio restoration.
//!
//! This module provides monophonic pitch detection using the YIN algorithm and
//! pitch correction using formant-preserving shifting. It also includes vibrato
//! reduction by detecting and smoothing periodic pitch deviations.
//!
//! # Architecture
//!
//! 1. **Pitch Detection** — YIN-based F0 estimator with confidence gating.
//! 2. **Pitch Correction** — PSOLA (Pitch Synchronous Overlap-Add) for
//!    formant-preserving transposition.
//! 3. **Vibrato Reduction** — median-based pitch trajectory smoothing with
//!    configurable depth and rate limits.

use crate::error::{RestoreError, RestoreResult};
use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// YIN pitch detector
// ---------------------------------------------------------------------------

/// Configuration for the YIN pitch detector.
#[derive(Debug, Clone)]
pub struct YinConfig {
    /// The probability threshold for accepting a pitch estimate (0.0–1.0).
    /// Lower values are more permissive, higher values demand higher confidence.
    pub threshold: f32,
    /// Minimum fundamental frequency to consider (Hz).
    pub f0_min_hz: f32,
    /// Maximum fundamental frequency to consider (Hz).
    pub f0_max_hz: f32,
    /// Analysis frame size in samples.  Should be at least `sample_rate / f0_min`.
    pub frame_size: usize,
    /// Hop size between consecutive frames.
    pub hop_size: usize,
}

impl Default for YinConfig {
    fn default() -> Self {
        Self {
            threshold: 0.15,
            f0_min_hz: 60.0,
            f0_max_hz: 1000.0,
            frame_size: 2048,
            hop_size: 512,
        }
    }
}

/// A single pitch estimate for one analysis frame.
#[derive(Debug, Clone)]
pub struct PitchFrame {
    /// Estimated fundamental frequency in Hz, or `None` if unvoiced.
    pub frequency_hz: Option<f32>,
    /// YIN aperiodicity (0 = perfectly periodic, 1 = noise).
    pub aperiodicity: f32,
    /// Centre sample of the frame.
    pub centre_sample: usize,
}

/// YIN-based monophonic pitch detector.
///
/// Reference: de Cheveigné & Kawahara (2002). "YIN, a fundamental frequency
/// estimator for speech and music." JASA 111(4).
#[derive(Debug, Clone)]
pub struct YinDetector {
    config: YinConfig,
}

impl YinDetector {
    /// Create a new YIN detector with the given configuration.
    #[must_use]
    pub fn new(config: YinConfig) -> Self {
        Self { config }
    }

    /// Create a detector with default settings tuned for a given sample rate.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn for_sample_rate(sample_rate: u32) -> Self {
        let frame_size = (sample_rate as f32 / 60.0 * 2.0).ceil() as usize;
        let frame_size = frame_size.next_power_of_two().max(1024);
        Self::new(YinConfig {
            frame_size,
            hop_size: frame_size / 4,
            ..YinConfig::default()
        })
    }

    /// Detect pitch in a buffer of mono samples.
    ///
    /// # Errors
    ///
    /// Returns `RestoreError::InvalidParameter` if `sample_rate` is zero.
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    pub fn detect(&self, samples: &[f32], sample_rate: u32) -> RestoreResult<Vec<PitchFrame>> {
        if sample_rate == 0 {
            return Err(RestoreError::InvalidParameter(
                "sample_rate must be > 0".into(),
            ));
        }
        if samples.is_empty() {
            return Ok(Vec::new());
        }

        let sr = sample_rate as f32;
        let tau_min = (sr / self.config.f0_max_hz).floor() as usize;
        let tau_max = (sr / self.config.f0_min_hz).ceil() as usize;
        let tau_max = tau_max.min(self.config.frame_size / 2);

        if tau_min >= tau_max {
            return Ok(Vec::new());
        }

        let mut frames = Vec::new();
        let mut offset = 0;

        while offset + self.config.frame_size <= samples.len() {
            let frame = &samples[offset..offset + self.config.frame_size];
            let centre = offset + self.config.frame_size / 2;

            let (frequency_hz, aperiodicity) =
                self.estimate_pitch(frame, sample_rate, tau_min, tau_max);

            frames.push(PitchFrame {
                frequency_hz,
                aperiodicity,
                centre_sample: centre,
            });

            offset += self.config.hop_size;
        }

        Ok(frames)
    }

    /// Run YIN on a single frame and return `(Option<f0_hz>, aperiodicity)`.
    #[allow(clippy::cast_precision_loss)]
    fn estimate_pitch(
        &self,
        frame: &[f32],
        sample_rate: u32,
        tau_min: usize,
        tau_max: usize,
    ) -> (Option<f32>, f32) {
        let n = frame.len();

        // Step 1: Difference function
        let mut diff = vec![0.0_f32; tau_max + 1];
        for tau in 1..=tau_max {
            let mut sum = 0.0_f32;
            for j in 0..n.saturating_sub(tau) {
                let d = frame[j] - frame[j + tau];
                sum += d * d;
            }
            diff[tau] = sum;
        }

        // Step 2: Cumulative mean normalised difference
        let mut cmnd = vec![0.0_f32; tau_max + 1];
        cmnd[0] = 1.0;
        let mut running_sum = 0.0_f32;
        for tau in 1..=tau_max {
            running_sum += diff[tau];
            if running_sum.abs() < f32::EPSILON {
                cmnd[tau] = 1.0;
            } else {
                cmnd[tau] = diff[tau] * tau as f32 / running_sum;
            }
        }

        // Step 3: Absolute threshold — find first tau with cmnd < threshold,
        // then descend to the local minimum within that basin.
        let threshold = self.config.threshold;
        let mut best_tau: Option<usize> = None;
        let mut tau = tau_min;
        while tau <= tau_max {
            if cmnd[tau] < threshold {
                // Step 4: Descend to the bottom of the current basin.
                // Walk forward as long as CMND keeps decreasing.
                let basin_start = tau;
                while tau < tau_max && cmnd[tau + 1] < cmnd[tau] {
                    tau += 1;
                }
                // `tau` is now at the local minimum.  Refine with parabolic interp.
                let local_min_tau = tau;
                let refined = if local_min_tau > basin_start && local_min_tau < tau_max {
                    let better = parabolic_interp(local_min_tau, &cmnd);
                    let t = better.round() as usize;
                    t.clamp(tau_min, tau_max)
                } else {
                    local_min_tau
                };
                best_tau = Some(refined);
                break;
            }
            tau += 1;
        }

        // If no threshold crossing, take global minimum
        let best_tau = best_tau.unwrap_or_else(|| {
            let mut min_val = f32::MAX;
            let mut min_tau = tau_min;
            for t in tau_min..=tau_max {
                if cmnd[t] < min_val {
                    min_val = cmnd[t];
                    min_tau = t;
                }
            }
            min_tau
        });

        let aperiodicity = cmnd[best_tau].clamp(0.0, 1.0);
        let sr = sample_rate as f32;
        let frequency_hz = if aperiodicity < self.config.threshold && best_tau > 0 {
            Some(sr / best_tau as f32)
        } else {
            None
        };

        (frequency_hz, aperiodicity)
    }
}

/// Parabolic interpolation for refining peak location.
#[allow(clippy::cast_precision_loss)]
fn parabolic_interp(tau: usize, values: &[f32]) -> f32 {
    if tau == 0 || tau + 1 >= values.len() {
        return tau as f32;
    }
    let y0 = values[tau - 1];
    let y1 = values[tau];
    let y2 = values[tau + 1];
    let denom = 2.0 * (2.0 * y1 - y0 - y2);
    if denom.abs() < f32::EPSILON {
        return tau as f32;
    }
    tau as f32 + (y0 - y2) / denom
}

// ---------------------------------------------------------------------------
// Vibrato reducer
// ---------------------------------------------------------------------------

/// Configuration for vibrato reduction.
#[derive(Debug, Clone)]
pub struct VibratoReducerConfig {
    /// Minimum vibrato rate in Hz (below this, pitch variation is not vibrato).
    pub min_rate_hz: f32,
    /// Maximum vibrato rate in Hz (above this, variation is too fast for vibrato).
    pub max_rate_hz: f32,
    /// Maximum vibrato depth to correct in cents.  Larger deviations may be
    /// intentional expression and are left untouched.
    pub max_depth_cents: f32,
    /// Reduction amount: 0.0 = no change, 1.0 = fully flatten vibrato.
    pub reduction: f32,
    /// Smoothing window in frames used to estimate the vibrato centre pitch.
    pub smoothing_window: usize,
}

impl Default for VibratoReducerConfig {
    fn default() -> Self {
        Self {
            min_rate_hz: 4.0,
            max_rate_hz: 8.0,
            max_depth_cents: 100.0,
            reduction: 0.8,
            smoothing_window: 11,
        }
    }
}

/// Summary of vibrato reduction applied.
#[derive(Debug, Clone)]
pub struct VibratoReductionResult {
    /// Number of frames where vibrato was detected and reduced.
    pub frames_corrected: usize,
    /// Average vibrato depth in cents across corrected frames.
    pub mean_depth_cents: f32,
}

/// Vibrato reducer — smooths out periodic pitch deviations in a pitch track.
#[derive(Debug, Clone)]
pub struct VibratoReducer {
    /// Reducer configuration.
    pub config: VibratoReducerConfig,
}

impl Default for VibratoReducer {
    fn default() -> Self {
        Self {
            config: VibratoReducerConfig::default(),
        }
    }
}

impl VibratoReducer {
    /// Create a new vibrato reducer.
    #[must_use]
    pub fn new(config: VibratoReducerConfig) -> Self {
        Self { config }
    }

    /// Reduce vibrato in a pitch track.
    ///
    /// Modifies `pitch_frames` in-place, adjusting the `frequency_hz` of each
    /// frame toward the smoothed (median) trajectory.
    ///
    /// # Errors
    ///
    /// Returns `RestoreError::InvalidParameter` if `sample_rate` is zero.
    #[allow(clippy::cast_precision_loss)]
    pub fn reduce(
        &self,
        pitch_frames: &mut [PitchFrame],
        sample_rate: u32,
        hop_size: usize,
    ) -> RestoreResult<VibratoReductionResult> {
        if sample_rate == 0 {
            return Err(RestoreError::InvalidParameter(
                "sample_rate must be > 0".into(),
            ));
        }
        if pitch_frames.is_empty() || hop_size == 0 {
            return Ok(VibratoReductionResult {
                frames_corrected: 0,
                mean_depth_cents: 0.0,
            });
        }

        let n = pitch_frames.len();
        let half_win = self.config.smoothing_window / 2;

        // Compute median-smoothed pitch track (voiced only)
        let voiced_hz: Vec<Option<f32>> = pitch_frames.iter().map(|f| f.frequency_hz).collect();

        let smoothed: Vec<Option<f32>> = (0..n)
            .map(|i| {
                let lo = i.saturating_sub(half_win);
                let hi = (i + half_win + 1).min(n);
                let window_vals: Vec<f32> = voiced_hz[lo..hi].iter().filter_map(|&v| v).collect();
                if window_vals.is_empty() {
                    None
                } else {
                    Some(median_f32(&window_vals))
                }
            })
            .collect();

        // Frames per second
        let frames_per_sec = sample_rate as f32 / hop_size as f32;
        let mut frames_corrected = 0;
        let mut total_depth_cents = 0.0_f32;

        for i in 0..n {
            let (Some(current_hz), Some(centre_hz)) = (pitch_frames[i].frequency_hz, smoothed[i])
            else {
                continue;
            };

            if centre_hz < f32::EPSILON {
                continue;
            }

            let deviation_cents = hz_to_cents(current_hz, centre_hz);
            let deviation_abs = deviation_cents.abs();

            // Only correct if deviation is within vibrato depth range
            if deviation_abs > self.config.max_depth_cents {
                continue;
            }

            // Estimate vibrato rate from local pitch oscillation
            let vibrato_rate = estimate_vibrato_rate(&voiced_hz, i, frames_per_sec, half_win);

            let is_vibrato =
                vibrato_rate >= self.config.min_rate_hz && vibrato_rate <= self.config.max_rate_hz;

            if is_vibrato {
                let corrected_cents = deviation_cents * (1.0 - self.config.reduction);
                let corrected_hz = cents_to_hz(centre_hz, corrected_cents);
                pitch_frames[i].frequency_hz = Some(corrected_hz);
                frames_corrected += 1;
                total_depth_cents += deviation_abs;
            }
        }

        let mean_depth_cents = if frames_corrected > 0 {
            total_depth_cents / frames_corrected as f32
        } else {
            0.0
        };

        Ok(VibratoReductionResult {
            frames_corrected,
            mean_depth_cents,
        })
    }
}

/// Convert frequency ratio to cents.
fn hz_to_cents(f: f32, reference: f32) -> f32 {
    if reference < f32::EPSILON {
        return 0.0;
    }
    1200.0 * (f / reference).log2()
}

/// Convert cents offset to Hz.
fn cents_to_hz(reference_hz: f32, cents: f32) -> f32 {
    reference_hz * 2.0_f32.powf(cents / 1200.0)
}

/// Compute the median of a slice (modifies a temporary copy).
#[allow(clippy::cast_precision_loss)]
fn median_f32(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    }
}

/// Estimate vibrato rate from local pitch oscillations (zero-crossing rate).
#[allow(clippy::cast_precision_loss)]
fn estimate_vibrato_rate(
    voiced_hz: &[Option<f32>],
    centre: usize,
    frames_per_sec: f32,
    half_win: usize,
) -> f32 {
    let lo = centre.saturating_sub(half_win * 2);
    let hi = (centre + half_win * 2 + 1).min(voiced_hz.len());
    let window: Vec<f32> = voiced_hz[lo..hi].iter().filter_map(|&v| v).collect();

    if window.len() < 4 {
        return 0.0;
    }

    // Compute mean
    let mean = window.iter().sum::<f32>() / window.len() as f32;
    // Count zero crossings (relative to mean)
    let mut crossings = 0_usize;
    for i in 1..window.len() {
        let prev = window[i - 1] - mean;
        let curr = window[i] - mean;
        if prev * curr < 0.0 {
            crossings += 1;
        }
    }
    // Rate = (crossings / 2) oscillations over the window duration
    let window_dur_sec = window.len() as f32 / frames_per_sec;
    if window_dur_sec < f32::EPSILON {
        return 0.0;
    }
    (crossings as f32 / 2.0) / window_dur_sec
}

// ---------------------------------------------------------------------------
// PSOLA pitch corrector
// ---------------------------------------------------------------------------

/// Configuration for the PSOLA pitch corrector.
#[derive(Debug, Clone)]
pub struct PsolaConfig {
    /// Target pitch in Hz.  When `Some`, all voiced frames are transposed to
    /// this fixed pitch.  When `None`, per-frame correction is driven by
    /// `target_frames`.
    pub target_hz: Option<f32>,
    /// Maximum shift in semitones.  Shifts larger than this are clamped.
    pub max_shift_semitones: f32,
    /// Overlap-add synthesis window.
    pub window_size: usize,
    /// Hop size for overlap-add synthesis.
    pub hop_size: usize,
}

impl Default for PsolaConfig {
    fn default() -> Self {
        Self {
            target_hz: None,
            max_shift_semitones: 6.0,
            window_size: 2048,
            hop_size: 512,
        }
    }
}

/// Result of PSOLA pitch correction.
#[derive(Debug, Clone)]
pub struct PitchCorrectionResult {
    /// Output samples (same length as input).
    pub samples: Vec<f32>,
    /// Number of voiced frames corrected.
    pub frames_corrected: usize,
    /// Mean shift applied in cents across corrected frames.
    pub mean_shift_cents: f32,
}

/// Formant-preserving PSOLA pitch corrector.
///
/// Uses the detected pitch frames to resample each pitch period to the desired
/// F0 while preserving the spectral envelope (formants) via the
/// Overlap-Add synthesis.
///
/// Note: This is a simplified TD-PSOLA that works best on clean, monophonic
/// signals with well-detected pitch.
#[derive(Debug, Clone)]
pub struct PsolaPitchCorrector {
    /// Corrector configuration.
    pub config: PsolaConfig,
}

impl Default for PsolaPitchCorrector {
    fn default() -> Self {
        Self {
            config: PsolaConfig::default(),
        }
    }
}

impl PsolaPitchCorrector {
    /// Create a new PSOLA corrector.
    #[must_use]
    pub fn new(config: PsolaConfig) -> Self {
        Self { config }
    }

    /// Correct pitch of `samples` using the provided pitch frames.
    ///
    /// When `config.target_hz` is `Some(f)`, every voiced frame is shifted to
    /// `f`.  When `None`, `target_frames` must be the same length as
    /// `pitch_frames`; its `frequency_hz` values are the desired pitches.
    ///
    /// # Errors
    ///
    /// Returns `RestoreError::InvalidParameter` if `sample_rate` is zero.
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    pub fn correct(
        &self,
        samples: &[f32],
        pitch_frames: &[PitchFrame],
        sample_rate: u32,
    ) -> RestoreResult<PitchCorrectionResult> {
        if sample_rate == 0 {
            return Err(RestoreError::InvalidParameter(
                "sample_rate must be > 0".into(),
            ));
        }
        if samples.is_empty() {
            return Ok(PitchCorrectionResult {
                samples: Vec::new(),
                frames_corrected: 0,
                mean_shift_cents: 0.0,
            });
        }

        // Output accumulator and normalisation buffer
        let mut output = vec![0.0_f32; samples.len()];
        let mut norm = vec![0.0_f32; samples.len()];

        let win_size = self.config.window_size;
        let hop = self.config.hop_size;
        let sr = sample_rate as f32;

        let mut frames_corrected = 0;
        let mut total_shift_cents = 0.0_f32;

        // Process each analysis frame
        for (frame_idx, pframe) in pitch_frames.iter().enumerate() {
            let Some(source_hz) = pframe.frequency_hz else {
                // Unvoiced: copy through as-is using overlap-add
                let centre = pframe.centre_sample;
                let start = centre.saturating_sub(win_size / 2);
                let end = (start + win_size).min(samples.len());
                if end <= start {
                    continue;
                }
                let frame_len = end - start;
                for i in 0..frame_len {
                    if start + i < output.len() {
                        let env = hann_window_sample(i, frame_len);
                        output[start + i] += samples[start + i] * env;
                        norm[start + i] += env;
                    }
                }
                continue;
            };

            // Compute target frequency
            let target_hz = self.config.target_hz.unwrap_or(source_hz);
            let shift_ratio = target_hz / source_hz;

            // Clamp by max_shift_semitones
            let max_ratio = 2.0_f32.powf(self.config.max_shift_semitones / 12.0);
            let shift_ratio = shift_ratio.clamp(1.0 / max_ratio, max_ratio);

            let shift_cents = 1200.0 * shift_ratio.log2();

            // Source period in samples
            let source_period = (sr / source_hz).round() as usize;
            let source_period = source_period.max(1);

            // Centre of this analysis frame
            let centre = pframe.centre_sample;

            // Iterate over pitch-synchronous windows in this frame
            let frame_start = frame_idx * hop;
            let frame_end = ((frame_idx + 1) * hop).min(samples.len());

            let mut synth_pos = frame_start;
            while synth_pos < frame_end {
                // Analysis mark: nearest pitch mark in source
                let analysis_pos =
                    find_nearest_pitch_mark(synth_pos, source_period, centre, samples.len());

                let half_period = source_period;
                let src_start = analysis_pos.saturating_sub(half_period);
                let src_end = (analysis_pos + half_period).min(samples.len());

                if src_end <= src_start {
                    synth_pos += (source_period as f32 / shift_ratio).round() as usize;
                    synth_pos = synth_pos.max(synth_pos + 1);
                    continue;
                }

                let grain_len = src_end - src_start;
                let dst_start = synth_pos.saturating_sub(half_period);
                let dst_end = (dst_start + grain_len).min(output.len());
                let actual_len = dst_end.saturating_sub(dst_start);

                for i in 0..actual_len {
                    let env = hann_window_sample(i, grain_len);
                    if src_start + i < samples.len() && dst_start + i < output.len() {
                        output[dst_start + i] += samples[src_start + i] * env;
                        norm[dst_start + i] += env;
                    }
                }

                let next_hop = (source_period as f32 / shift_ratio).round() as usize;
                synth_pos += next_hop.max(1);
            }

            if (shift_cents).abs() > 1.0 {
                frames_corrected += 1;
                total_shift_cents += shift_cents.abs();
            }
        }

        // Normalise output
        for (out, &n) in output.iter_mut().zip(norm.iter()) {
            if n > f32::EPSILON {
                *out /= n;
            }
        }

        // Blend regions with zero normalisation using dry signal
        for i in 0..output.len() {
            if norm[i] < f32::EPSILON {
                output[i] = samples[i];
            }
        }

        let mean_shift_cents = if frames_corrected > 0 {
            total_shift_cents / frames_corrected as f32
        } else {
            0.0
        };

        Ok(PitchCorrectionResult {
            samples: output,
            frames_corrected,
            mean_shift_cents,
        })
    }
}

/// Hann window sample value for index `i` in a window of `length`.
#[allow(clippy::cast_precision_loss)]
fn hann_window_sample(i: usize, length: usize) -> f32 {
    if length <= 1 {
        return 1.0;
    }
    0.5 * (1.0 - (2.0 * PI * i as f32 / (length - 1) as f32).cos())
}

/// Find the nearest pitch mark (sample-level pitch epoch) to `pos`.
///
/// Marks are spaced `period` samples apart, aligned to `reference`.
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
fn find_nearest_pitch_mark(pos: usize, period: usize, reference: usize, len: usize) -> usize {
    if period == 0 {
        return pos.min(len.saturating_sub(1));
    }
    // Offset of pos from reference
    let diff = if pos >= reference {
        pos - reference
    } else {
        // pos before reference
        let d = reference - pos;
        return reference.saturating_sub(((d as f32 / period as f32).round() as usize) * period);
    };
    let n_periods = (diff as f32 / period as f32).round() as usize;
    (reference + n_periods * period).min(len.saturating_sub(1))
}

// ---------------------------------------------------------------------------
// High-level facade
// ---------------------------------------------------------------------------

/// Complete pitch correction pipeline: detect → reduce vibrato → correct pitch.
///
/// # Errors
///
/// Returns `RestoreError::InvalidParameter` if parameters are invalid.
pub fn correct_pitch(
    samples: &[f32],
    sample_rate: u32,
    target_hz: Option<f32>,
    vibrato_reduction: f32,
) -> RestoreResult<PitchCorrectionResult> {
    let detector = YinDetector::for_sample_rate(sample_rate);
    let mut frames = detector.detect(samples, sample_rate)?;

    if vibrato_reduction > 0.0 {
        let config = VibratoReducerConfig {
            reduction: vibrato_reduction.clamp(0.0, 1.0),
            ..VibratoReducerConfig::default()
        };
        let reducer = VibratoReducer::new(config);
        reducer.reduce(&mut frames, sample_rate, detector.config.hop_size)?;
    }

    let psola_config = PsolaConfig {
        target_hz,
        ..PsolaConfig::default()
    };
    let corrector = PsolaPitchCorrector::new(psola_config);
    corrector.correct(samples, &frames, sample_rate)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    const SR: u32 = 44100;

    fn make_sine(freq: f32, duration_ms: f32) -> Vec<f32> {
        let n = (SR as f32 * duration_ms / 1000.0) as usize;
        (0..n)
            .map(|i| (2.0 * PI * freq * i as f32 / SR as f32).sin())
            .collect()
    }

    #[test]
    fn test_yin_config_default() {
        let cfg = YinConfig::default();
        assert!(cfg.f0_min_hz < cfg.f0_max_hz);
        assert!(cfg.frame_size > 0);
        assert!(cfg.threshold > 0.0 && cfg.threshold < 1.0);
    }

    #[test]
    fn test_yin_detector_empty_input() {
        let det = YinDetector::for_sample_rate(SR);
        let frames = det.detect(&[], SR).expect("should succeed");
        assert!(frames.is_empty());
    }

    #[test]
    fn test_yin_detector_zero_sample_rate_errors() {
        let det = YinDetector::for_sample_rate(SR);
        let result = det.detect(&[0.0; 4096], 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_yin_detects_440hz_sine() {
        let samples = make_sine(440.0, 200.0);
        let det = YinDetector::new(YinConfig {
            threshold: 0.15,
            f0_min_hz: 60.0,
            f0_max_hz: 1000.0,
            frame_size: 2048,
            hop_size: 512,
        });
        let frames = det.detect(&samples, SR).expect("ok");
        // At least one voiced frame expected for a clean sine
        let voiced: Vec<_> = frames.iter().filter(|f| f.frequency_hz.is_some()).collect();
        assert!(
            !voiced.is_empty(),
            "should detect at least one voiced frame"
        );
        // Frequency should be near 440 Hz
        if let Some(first) = voiced.first() {
            let f = first.frequency_hz.expect("voiced");
            assert!((f - 440.0).abs() < 30.0, "expected ~440 Hz, got {f}");
        }
    }

    #[test]
    fn test_yin_silence_unvoiced() {
        let samples = vec![0.0_f32; 8192];
        let det = YinDetector::for_sample_rate(SR);
        let frames = det.detect(&samples, SR).expect("ok");
        for f in &frames {
            assert!(
                f.frequency_hz.is_none(),
                "silence should be unvoiced, got {:?}",
                f.frequency_hz
            );
        }
    }

    #[test]
    fn test_vibrato_reducer_empty_input() {
        let reducer = VibratoReducer::default();
        let mut frames: Vec<PitchFrame> = Vec::new();
        let result = reducer.reduce(&mut frames, SR, 512).expect("ok");
        assert_eq!(result.frames_corrected, 0);
    }

    #[test]
    fn test_vibrato_reducer_zero_sample_rate_errors() {
        let reducer = VibratoReducer::default();
        let mut frames = vec![PitchFrame {
            frequency_hz: Some(440.0),
            aperiodicity: 0.1,
            centre_sample: 1024,
        }];
        assert!(reducer.reduce(&mut frames, 0, 512).is_err());
    }

    #[test]
    fn test_psola_corrector_empty_input() {
        let corrector = PsolaPitchCorrector::default();
        let frames: Vec<PitchFrame> = Vec::new();
        let result = corrector.correct(&[], &frames, SR).expect("ok");
        assert!(result.samples.is_empty());
    }

    #[test]
    fn test_psola_output_same_length_as_input() {
        let samples = make_sine(440.0, 100.0);
        let len = samples.len();
        let det = YinDetector::for_sample_rate(SR);
        let frames = det.detect(&samples, SR).expect("ok");
        let corrector = PsolaPitchCorrector::new(PsolaConfig {
            target_hz: Some(440.0),
            ..PsolaConfig::default()
        });
        let result = corrector.correct(&samples, &frames, SR).expect("ok");
        assert_eq!(result.samples.len(), len);
    }

    #[test]
    fn test_psola_unison_shift_low_correction_count() {
        // Target == source → minimal or zero correction
        let samples = make_sine(440.0, 200.0);
        let det = YinDetector::for_sample_rate(SR);
        let frames = det.detect(&samples, SR).expect("ok");
        let corrector = PsolaPitchCorrector::new(PsolaConfig {
            target_hz: Some(440.0),
            ..PsolaConfig::default()
        });
        let result = corrector.correct(&samples, &frames, SR).expect("ok");
        // When target == detected, shift_cents ~ 0 → frames_corrected should be low
        assert!(result.mean_shift_cents < 50.0);
    }

    #[test]
    fn test_correct_pitch_facade_succeeds() {
        let samples = make_sine(440.0, 100.0);
        let result = correct_pitch(&samples, SR, Some(440.0), 0.5);
        assert!(result.is_ok());
        let r = result.expect("ok");
        assert_eq!(r.samples.len(), samples.len());
    }

    #[test]
    fn test_hann_window_edges() {
        // Hann window should be near 0 at edges and near 1 at centre
        let n = 1024;
        let start = hann_window_sample(0, n);
        let mid = hann_window_sample(n / 2, n);
        let end_val = hann_window_sample(n - 1, n);
        assert!(start < 0.01, "start should be ~0, got {start}");
        assert!(mid > 0.99, "mid should be ~1, got {mid}");
        assert!(end_val < 0.01, "end should be ~0, got {end_val}");
    }

    #[test]
    fn test_hz_to_cents_and_back() {
        let ref_hz = 440.0_f32;
        let cents = 100.0_f32;
        let shifted = cents_to_hz(ref_hz, cents);
        let recovered = hz_to_cents(shifted, ref_hz);
        assert!(
            (recovered - cents).abs() < 0.01,
            "round-trip failed: {recovered}"
        );
    }
}
