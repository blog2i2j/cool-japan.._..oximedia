//! Parallel per-channel gain processing for surround content (> 2 channels).
//!
//! For multichannel audio (5.1, 7.1, 7.1.4 Atmos, etc.) the per-channel loudness
//! measurements and gain applications are independent of each other. This module
//! exploits that independence via **rayon** data-parallel iterators, distributing
//! the work across all available CPU cores.
//!
//! ## Architecture
//!
//! Interleaved audio is first **de-interleaved** into separate per-channel buffers.
//! Each channel is then processed concurrently:
//!
//! 1. **RMS / peak measurement** — cheap loudness proxy for per-channel gain decisions.
//! 2. **Gain application** — scale each sample by the channel-specific linear factor.
//! 3. **Optional peak clamping** — prevent over-shoot on a per-channel basis.
//!
//! After parallel processing, the channels are **re-interleaved** back into the
//! output buffer.
//!
//! ## When to use
//!
//! Use this module whenever:
//! - The channel count is > 2 (stereo processing rarely benefits from parallelism
//!   given the per-thread setup overhead).
//! - You have per-channel gain values (e.g. from a [`SurroundNormalizer`] analysis).
//! - Throughput is the priority over real-time latency.
//!
//! ## Example
//!
//! ```rust
//! use oximedia_normalize::parallel_channels::{parallel_apply_gains, ChannelGains};
//!
//! // 5.1 audio: 6 channels, 4800 frames (interleaved)
//! let channels = 6;
//! let frames = 4800;
//! let input: Vec<f32> = (0..(frames * channels))
//!     .map(|i| (i as f32) / (frames * channels) as f32)
//!     .collect();
//! let mut output = vec![0.0_f32; input.len()];
//!
//! // Apply +6 dB to all channels
//! let gains = ChannelGains::uniform(channels, 6.0);
//! parallel_apply_gains(&input, &mut output, &gains).expect("parallel apply failed");
//! ```
//!
//! [`SurroundNormalizer`]: crate::surround_norm::SurroundNormalizer

use crate::{NormalizeError, NormalizeResult};
use rayon::prelude::*;

// ─── ChannelGains ────────────────────────────────────────────────────────────

/// Per-channel gain values to apply during parallel processing.
///
/// Stores one dB value per channel together with a corresponding linear value
/// that is pre-computed at construction time (so the hot inner loop avoids
/// repeated `pow10` calls).
#[derive(Clone, Debug)]
pub struct ChannelGains {
    /// Gain in dB for each channel.
    pub gains_db: Vec<f64>,
    /// Pre-computed linear gains (power of 10, amplitude ratio).
    pub gains_linear: Vec<f64>,
}

impl ChannelGains {
    /// Create per-channel gains from a slice of dB values (one per channel).
    ///
    /// # Errors
    ///
    /// Returns [`NormalizeError::InvalidConfig`] if `gains_db` is empty.
    pub fn from_db(gains_db: &[f64]) -> NormalizeResult<Self> {
        if gains_db.is_empty() {
            return Err(NormalizeError::InvalidConfig(
                "gains_db must contain at least one value".to_string(),
            ));
        }

        let gains_linear: Vec<f64> = gains_db.iter().map(|&g| db_to_linear(g)).collect();

        Ok(Self {
            gains_db: gains_db.to_vec(),
            gains_linear,
        })
    }

    /// Create identical `gain_db` applied to all `channel_count` channels.
    pub fn uniform(channel_count: usize, gain_db: f64) -> Self {
        let gains_db = vec![gain_db; channel_count.max(1)];
        let gains_linear = gains_db.iter().map(|&g| db_to_linear(g)).collect();
        Self {
            gains_db,
            gains_linear,
        }
    }

    /// Create a [`ChannelGains`] that applies 0 dB (unity gain) to every channel.
    pub fn unity(channel_count: usize) -> Self {
        Self::uniform(channel_count, 0.0)
    }

    /// Returns the number of channels.
    pub fn channel_count(&self) -> usize {
        self.gains_db.len()
    }

    /// Returns the dB gain for `channel`, or 0 dB if out of range.
    pub fn gain_db(&self, channel: usize) -> f64 {
        self.gains_db.get(channel).copied().unwrap_or(0.0)
    }

    /// Returns the linear gain for `channel`, or 1.0 if out of range.
    pub fn gain_linear(&self, channel: usize) -> f64 {
        self.gains_linear.get(channel).copied().unwrap_or(1.0)
    }
}

// ─── Per-channel measurements ────────────────────────────────────────────────

/// Per-channel measurement results computed during parallel analysis.
#[derive(Clone, Debug)]
pub struct ChannelMeasurement {
    /// Channel index (0-based).
    pub channel_idx: usize,

    /// RMS level of this channel (linear, 0–1).
    pub rms: f64,

    /// Peak (maximum absolute sample value).
    pub peak: f64,

    /// Crest factor in dB (peak / RMS ratio, 0 if silent).
    pub crest_factor_db: f64,
}

impl ChannelMeasurement {
    /// Returns the RMS in dBFS.
    pub fn rms_db(&self) -> f64 {
        if self.rms < 1e-15 {
            -144.0
        } else {
            20.0 * self.rms.log10()
        }
    }

    /// Returns the peak in dBTP.
    pub fn peak_db(&self) -> f64 {
        if self.peak < 1e-15 {
            -144.0
        } else {
            20.0 * self.peak.log10()
        }
    }
}

// ─── Primary API ─────────────────────────────────────────────────────────────

/// Apply per-channel gains to interleaved audio in parallel.
///
/// # Arguments
///
/// * `input`  — Interleaved f32 samples (length = `frames × gains.channel_count()`).
/// * `output` — Destination buffer of the same length.
/// * `gains`  — Per-channel gain set.
///
/// # Errors
///
/// - [`NormalizeError::ProcessingError`] — if output length ≠ input length.
/// - [`NormalizeError::InvalidConfig`]   — if channel count is 0.
/// - [`NormalizeError::ProcessingError`] — if input length is not a multiple of channel count.
pub fn parallel_apply_gains(
    input: &[f32],
    output: &mut [f32],
    gains: &ChannelGains,
) -> NormalizeResult<()> {
    let channels = gains.channel_count();

    validate_buffers(input, output, channels)?;

    // De-interleave: collect one Vec<f32> per channel.
    let frames = input.len() / channels;
    let mut channel_buffers: Vec<Vec<f32>> = (0..channels)
        .map(|c| {
            (0..frames)
                .map(|f| input[f * channels + c])
                .collect::<Vec<_>>()
        })
        .collect();

    // Apply gains in parallel (one channel per Rayon task).
    channel_buffers
        .par_iter_mut()
        .enumerate()
        .for_each(|(ch, buf)| {
            let gain = gains.gain_linear(ch) as f32;
            for s in buf.iter_mut() {
                *s *= gain;
            }
        });

    // Re-interleave into output.
    for (f, frame_out) in output.chunks_exact_mut(channels).enumerate() {
        for (c, out_sample) in frame_out.iter_mut().enumerate() {
            *out_sample = channel_buffers[c][f];
        }
    }

    Ok(())
}

/// Apply per-channel gains to interleaved f64 audio in parallel.
///
/// Equivalent to [`parallel_apply_gains`] but operates on `f64` samples.
///
/// # Errors
///
/// Same conditions as [`parallel_apply_gains`].
pub fn parallel_apply_gains_f64(
    input: &[f64],
    output: &mut [f64],
    gains: &ChannelGains,
) -> NormalizeResult<()> {
    let channels = gains.channel_count();

    if output.len() != input.len() {
        return Err(NormalizeError::ProcessingError(format!(
            "output ({}) and input ({}) length differ",
            output.len(),
            input.len()
        )));
    }

    if channels == 0 {
        return Err(NormalizeError::InvalidConfig(
            "channel count must be > 0".to_string(),
        ));
    }

    if input.len() % channels != 0 {
        return Err(NormalizeError::ProcessingError(format!(
            "input length {} is not a multiple of channel count {}",
            input.len(),
            channels
        )));
    }

    let frames = input.len() / channels;
    let mut channel_buffers: Vec<Vec<f64>> = (0..channels)
        .map(|c| (0..frames).map(|f| input[f * channels + c]).collect())
        .collect();

    channel_buffers
        .par_iter_mut()
        .enumerate()
        .for_each(|(ch, buf)| {
            let gain = gains.gain_linear(ch);
            for s in buf.iter_mut() {
                *s *= gain;
            }
        });

    for (f, frame_out) in output.chunks_exact_mut(channels).enumerate() {
        for (c, out_sample) in frame_out.iter_mut().enumerate() {
            *out_sample = channel_buffers[c][f];
        }
    }

    Ok(())
}

/// Measure each channel independently in parallel, returning per-channel statistics.
///
/// # Arguments
///
/// * `input`    — Interleaved f32 samples.
/// * `channels` — Number of channels.
///
/// # Errors
///
/// Returns an error if `channels == 0` or `input.len()` is not a multiple of `channels`.
pub fn parallel_measure_channels(
    input: &[f32],
    channels: usize,
) -> NormalizeResult<Vec<ChannelMeasurement>> {
    if channels == 0 {
        return Err(NormalizeError::InvalidConfig(
            "channel count must be > 0".to_string(),
        ));
    }

    if input.len() % channels != 0 {
        return Err(NormalizeError::ProcessingError(format!(
            "input length {} is not a multiple of channel count {}",
            input.len(),
            channels
        )));
    }

    let frames = input.len() / channels;

    // De-interleave
    let channel_data: Vec<Vec<f32>> = (0..channels)
        .map(|c| (0..frames).map(|f| input[f * channels + c]).collect())
        .collect();

    // Measure in parallel
    let measurements: Vec<ChannelMeasurement> = channel_data
        .par_iter()
        .enumerate()
        .map(|(ch_idx, buf)| {
            let peak = buf.iter().map(|&s| s.abs()).fold(0.0_f32, f32::max) as f64;
            let sum_sq: f64 = buf.iter().map(|&s| (s as f64) * (s as f64)).sum();
            let rms = if buf.is_empty() {
                0.0
            } else {
                (sum_sq / buf.len() as f64).sqrt()
            };
            let crest_factor_db = if rms < 1e-15 {
                0.0
            } else {
                20.0 * (peak / rms).log10()
            };

            ChannelMeasurement {
                channel_idx: ch_idx,
                rms,
                peak,
                crest_factor_db,
            }
        })
        .collect();

    Ok(measurements)
}

/// Apply per-channel gain with optional peak clamping in parallel.
///
/// After gain application, each sample is hard-clipped to `±clamp_linear`
/// (0.0 disables clamping, as does any value ≥ 1.0 which would never clip).
///
/// # Errors
///
/// Same conditions as [`parallel_apply_gains`].
pub fn parallel_apply_gains_clamped(
    input: &[f32],
    output: &mut [f32],
    gains: &ChannelGains,
    clamp_linear: f32,
) -> NormalizeResult<()> {
    let channels = gains.channel_count();
    validate_buffers(input, output, channels)?;

    let frames = input.len() / channels;
    let mut channel_buffers: Vec<Vec<f32>> = (0..channels)
        .map(|c| (0..frames).map(|f| input[f * channels + c]).collect())
        .collect();

    let do_clamp = clamp_linear > 0.0;

    channel_buffers
        .par_iter_mut()
        .enumerate()
        .for_each(|(ch, buf)| {
            let gain = gains.gain_linear(ch) as f32;
            for s in buf.iter_mut() {
                let v = *s * gain;
                *s = if do_clamp {
                    v.clamp(-clamp_linear, clamp_linear)
                } else {
                    v
                };
            }
        });

    for (f, frame_out) in output.chunks_exact_mut(channels).enumerate() {
        for (c, out_sample) in frame_out.iter_mut().enumerate() {
            *out_sample = channel_buffers[c][f];
        }
    }

    Ok(())
}

// ─── helpers ─────────────────────────────────────────────────────────────────

/// Convert dB to linear amplitude.
#[inline]
fn db_to_linear(db: f64) -> f64 {
    10.0_f64.powf(db / 20.0)
}

/// Validate that input and output buffers are compatible with the given channel count.
fn validate_buffers(input: &[f32], output: &[f32], channels: usize) -> NormalizeResult<()> {
    if channels == 0 {
        return Err(NormalizeError::InvalidConfig(
            "channel count must be > 0".to_string(),
        ));
    }

    if output.len() != input.len() {
        return Err(NormalizeError::ProcessingError(format!(
            "output ({}) and input ({}) length differ",
            output.len(),
            input.len()
        )));
    }

    if input.len() % channels != 0 {
        return Err(NormalizeError::ProcessingError(format!(
            "input length {} is not a multiple of channel count {}",
            input.len(),
            channels
        )));
    }

    Ok(())
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build interleaved audio: `channels` channels, each a sine wave at a different
    /// frequency, `frames` long at `sample_rate`.
    fn interleaved_sines(channels: usize, frames: usize, sample_rate: f32) -> Vec<f32> {
        let mut out = vec![0.0_f32; frames * channels];
        for c in 0..channels {
            let freq = 440.0 * (c + 1) as f32;
            for f in 0..frames {
                let t = f as f32 / sample_rate;
                out[f * channels + c] = 0.5 * (2.0 * std::f32::consts::PI * freq * t).sin();
            }
        }
        out
    }

    #[test]
    fn test_channel_gains_from_db() {
        let gains = ChannelGains::from_db(&[0.0, 6.0, -6.0]).expect("from_db failed");
        assert_eq!(gains.channel_count(), 3);
        assert!((gains.gain_linear(0) - 1.0).abs() < 1e-9, "0 dB → 1.0");
        // 10^(6/20) ≈ 1.99526, not exactly 2.0 — use 0.01 tolerance
        assert!((gains.gain_linear(1) - 2.0).abs() < 0.01, "+6 dB → ~2.0");
        // 10^(-6/20) ≈ 0.50119, not exactly 0.5
        assert!((gains.gain_linear(2) - 0.5).abs() < 0.01, "-6 dB → ~0.5");
    }

    #[test]
    fn test_channel_gains_uniform() {
        let gains = ChannelGains::uniform(6, 0.0);
        assert_eq!(gains.channel_count(), 6);
        for i in 0..6 {
            assert!((gains.gain_linear(i) - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn test_channel_gains_unity() {
        let gains = ChannelGains::unity(4);
        assert_eq!(gains.channel_count(), 4);
        for i in 0..4 {
            assert!((gains.gain_db(i) - 0.0).abs() < 1e-9);
        }
    }

    #[test]
    fn test_channel_gains_out_of_range_returns_defaults() {
        let gains = ChannelGains::unity(2);
        assert_eq!(gains.gain_db(999), 0.0);
        assert_eq!(gains.gain_linear(999), 1.0);
    }

    #[test]
    fn test_parallel_apply_gains_unity_preserves_signal() {
        let channels = 6;
        let frames = 4800;
        let input = interleaved_sines(channels, frames, 48000.0);
        let mut output = vec![0.0_f32; input.len()];
        let gains = ChannelGains::unity(channels);

        parallel_apply_gains(&input, &mut output, &gains).expect("apply failed");

        for (i, (&a, &b)) in input.iter().zip(output.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6, "sample {i} differed: {a} vs {b}");
        }
    }

    #[test]
    fn test_parallel_apply_gains_6db_boost() {
        let channels = 6;
        let frames = 480;
        let input = interleaved_sines(channels, frames, 48000.0);
        let mut output = vec![0.0_f32; input.len()];
        let gains = ChannelGains::uniform(channels, 6.0);

        parallel_apply_gains(&input, &mut output, &gains).expect("apply failed");

        // Use the actual linear gain (not the approximate 2×) to verify correctness.
        let gain_linear = gains.gain_linear(0) as f32;
        for (i, (&a, &b)) in input.iter().zip(output.iter()).enumerate() {
            let expected = a * gain_linear;
            assert!(
                (b - expected).abs() < 1e-5,
                "sample {i}: expected {expected}, got {b}"
            );
        }
    }

    #[test]
    fn test_parallel_apply_gains_mismatched_length_returns_error() {
        let input = vec![0.1_f32; 120];
        let mut output = vec![0.0_f32; 130];
        let gains = ChannelGains::unity(6);
        let result = parallel_apply_gains(&input, &mut output, &gains);
        assert!(result.is_err());
    }

    #[test]
    fn test_parallel_apply_gains_non_multiple_returns_error() {
        let input = vec![0.1_f32; 13]; // 13 is not divisible by 6
        let mut output = vec![0.0_f32; 13];
        let gains = ChannelGains::unity(6);
        let result = parallel_apply_gains(&input, &mut output, &gains);
        assert!(result.is_err());
    }

    #[test]
    fn test_parallel_measure_channels_basic() {
        let channels = 6;
        let frames = 4800;
        let input = interleaved_sines(channels, frames, 48000.0);

        let measurements = parallel_measure_channels(&input, channels).expect("measure failed");
        assert_eq!(measurements.len(), channels);

        for (ch, m) in measurements.iter().enumerate() {
            assert_eq!(m.channel_idx, ch);
            assert!(m.rms > 0.0, "channel {ch} RMS should be > 0");
            assert!(m.peak > 0.0, "channel {ch} peak should be > 0");
            assert!(m.crest_factor_db > 0.0, "channel {ch} crest factor > 0");
        }
    }

    #[test]
    fn test_parallel_measure_channels_zero_channel_count_returns_error() {
        let result = parallel_measure_channels(&[0.1_f32; 12], 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_parallel_apply_gains_clamped_prevents_clipping() {
        let channels = 4;
        let frames = 480;
        // Build audio that would exceed 0 dBFS after +20 dB gain
        let input: Vec<f32> = (0..(frames * channels)).map(|_| 0.5_f32).collect();
        let mut output = vec![0.0_f32; input.len()];
        let gains = ChannelGains::uniform(channels, 20.0); // ×10 → 5.0 → clips
        let clamp = 1.0_f32;

        parallel_apply_gains_clamped(&input, &mut output, &gains, clamp)
            .expect("clamped apply failed");

        for &s in &output {
            assert!(s <= clamp, "sample {s} exceeds clamp {clamp}");
            assert!(s >= -clamp, "sample {s} is below -{clamp}");
        }
    }

    #[test]
    fn test_parallel_apply_gains_f64_unity() {
        let channels = 8;
        let frames = 480;
        let input: Vec<f64> = (0..(frames * channels))
            .map(|i| (i as f64) * 0.0001)
            .collect();
        let mut output = vec![0.0_f64; input.len()];
        let gains = ChannelGains::unity(channels);

        parallel_apply_gains_f64(&input, &mut output, &gains).expect("f64 apply failed");

        for (i, (&a, &b)) in input.iter().zip(output.iter()).enumerate() {
            assert!((a - b).abs() < 1e-12, "f64 sample {i} differed: {a} vs {b}");
        }
    }

    #[test]
    fn test_channel_measurement_db_conversions() {
        let m = ChannelMeasurement {
            channel_idx: 0,
            rms: 0.1,
            peak: 0.5,
            crest_factor_db: 20.0 * (0.5_f64 / 0.1).log10(),
        };
        let rms_db = m.rms_db();
        assert!((rms_db - (-20.0)).abs() < 0.1, "rms_db: {rms_db}");
        let peak_db = m.peak_db();
        assert!((peak_db - (-6.02)).abs() < 0.1, "peak_db: {peak_db}");
    }
}
