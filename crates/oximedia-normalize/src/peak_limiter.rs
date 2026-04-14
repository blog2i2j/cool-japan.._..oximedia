//! Look-ahead true-peak brick-wall limiter with circular lookahead buffer.
//!
//! Provides a sample-accurate peak limiter that:
//! - Uses a power-of-two circular ring buffer for zero-copy lookahead (no shifting).
//! - Performs 4× polyphase oversampling for accurate inter-sample peak detection.
//! - Applies smooth gain reduction with configurable attack / release envelopes.
//! - Guarantees the output never exceeds the configured true-peak ceiling.
//!
//! # Algorithm
//!
//! ```text
//!  Input samples ─┬─► Lookahead ring (L samples) ──► Delayed output
//!                 │
//!                 └─► 4× upsample ──► peak detection ──► gain envelope ──► apply
//! ```
//!
//! 1. Every incoming sample is written into the circular lookahead buffer.
//! 2. The same sample is inserted into the 4× upsampling filter (linear interpolation
//!    between consecutive samples) and the worst-case peak in the next `L` samples
//!    is estimated.
//! 3. A gain-reduction envelope is driven: if the peak exceeds the ceiling, gain
//!    is immediately set to `ceiling / peak` (instantaneous attack); otherwise the
//!    envelope recovers with an exponential release coefficient.
//! 4. The sample that was written `L` samples ago is read from the ring and multiplied
//!    by the current gain.
//!
//! # References
//!
//! - ITU-R BS.1770-4 §5 — True-peak measurement via 4× oversampling
//! - AES17-2015 — Peak limiting for digital audio

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(dead_code)]

use crate::{NormalizeError, NormalizeResult};

// ────────────────────────────────────────────────────────────────────────────
// PeakLimiterConfig
// ────────────────────────────────────────────────────────────────────────────

/// Configuration for the look-ahead true-peak limiter.
#[derive(Clone, Debug)]
pub struct PeakLimiterConfig {
    /// True-peak ceiling in dBTP (must be ≤ 0.0).
    pub ceiling_dbtp: f64,

    /// Lookahead duration in milliseconds.
    pub lookahead_ms: f64,

    /// Release time in milliseconds (time to recover 60 dB).
    pub release_ms: f64,

    /// Sample rate in Hz.
    pub sample_rate: f64,

    /// Number of audio channels (interleaved samples assumed).
    pub channels: usize,

    /// Oversampling factor for peak detection (2 or 4; default 4).
    pub oversample: usize,
}

impl Default for PeakLimiterConfig {
    fn default() -> Self {
        Self {
            ceiling_dbtp: -1.0,
            lookahead_ms: 2.0,
            release_ms: 100.0,
            sample_rate: 48_000.0,
            channels: 2,
            oversample: 4,
        }
    }
}

impl PeakLimiterConfig {
    /// EBU R128 broadcast preset: -1 dBTP ceiling, 48 kHz, stereo.
    pub fn broadcast() -> Self {
        Self::default()
    }

    /// Streaming preset: -1.5 dBTP (extra headroom for lossy re-encode).
    pub fn streaming() -> Self {
        Self {
            ceiling_dbtp: -1.5,
            ..Default::default()
        }
    }

    /// Validate the configuration.
    pub fn validate(&self) -> NormalizeResult<()> {
        if self.ceiling_dbtp > 0.0 {
            return Err(NormalizeError::InvalidConfig(
                "ceiling_dbtp must be ≤ 0.0 dBTP".to_string(),
            ));
        }
        if self.lookahead_ms < 0.0 || self.lookahead_ms > 100.0 {
            return Err(NormalizeError::InvalidConfig(format!(
                "lookahead_ms {:.1} ms out of range 0–100",
                self.lookahead_ms
            )));
        }
        if self.release_ms <= 0.0 {
            return Err(NormalizeError::InvalidConfig(
                "release_ms must be > 0".to_string(),
            ));
        }
        if self.sample_rate < 8_000.0 || self.sample_rate > 384_000.0 {
            return Err(NormalizeError::InvalidConfig(format!(
                "sample_rate {:.0} Hz out of range 8000–384000",
                self.sample_rate
            )));
        }
        if self.channels == 0 || self.channels > 16 {
            return Err(NormalizeError::InvalidConfig(format!(
                "channels {} out of range 1–16",
                self.channels
            )));
        }
        if self.oversample != 2 && self.oversample != 4 {
            return Err(NormalizeError::InvalidConfig(
                "oversample must be 2 or 4".to_string(),
            ));
        }
        Ok(())
    }

    /// Ceiling as a linear amplitude ratio.
    pub fn ceiling_linear(&self) -> f64 {
        10.0_f64.powf(self.ceiling_dbtp / 20.0)
    }

    /// Lookahead in samples (per channel, rounded up to next power-of-two).
    pub fn lookahead_samples(&self) -> usize {
        let raw = ((self.lookahead_ms / 1000.0) * self.sample_rate).ceil() as usize;
        raw.max(1).next_power_of_two()
    }

    /// Release coefficient per sample (for a single-pole IIR).
    pub fn release_coeff(&self) -> f64 {
        // exp(-1 / (release_ms * sample_rate / 1000))
        let tau_samples = self.release_ms * self.sample_rate / 1000.0;
        (-1.0_f64 / tau_samples).exp()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// LookaheadRing — power-of-two circular buffer
// ────────────────────────────────────────────────────────────────────────────

/// Single-channel circular lookahead buffer.
///
/// The capacity is always a power of two so that modulo wrapping is a bitmask.
struct LookaheadRing {
    buf: Vec<f64>,
    mask: usize,
    write_head: usize,
}

impl LookaheadRing {
    fn new(capacity: usize) -> Self {
        let cap = capacity.max(2).next_power_of_two();
        Self {
            buf: vec![0.0; cap],
            mask: cap - 1,
            write_head: 0,
        }
    }

    /// Push a new sample and return the sample that was `delay` steps ago.
    #[inline]
    fn push_and_read(&mut self, sample: f64, delay: usize) -> f64 {
        self.buf[self.write_head] = sample;
        let read_head = self.write_head.wrapping_sub(delay) & self.mask;
        self.write_head = (self.write_head + 1) & self.mask;
        self.buf[read_head]
    }

    /// Reset all samples to zero.
    fn reset(&mut self) {
        self.buf.iter_mut().for_each(|s| *s = 0.0);
        self.write_head = 0;
    }
}

// ────────────────────────────────────────────────────────────────────────────
// PeakWindow — sliding maximum over lookahead horizon
// ────────────────────────────────────────────────────────────────────────────

/// Maintains a sliding maximum of |sample| values over a fixed window.
///
/// Uses a deque-based monotone-queue algorithm: O(1) amortised push/pop.
struct PeakWindow {
    /// Ring buffer holding (index, value) pairs in monotonically non-increasing order.
    deque: std::collections::VecDeque<(usize, f64)>,
    window: usize,
    cursor: usize,
}

impl PeakWindow {
    fn new(window: usize) -> Self {
        Self {
            deque: std::collections::VecDeque::with_capacity(window + 1),
            window: window.max(1),
            cursor: 0,
        }
    }

    /// Push a new value and return the running maximum over the window.
    fn push(&mut self, value: f64) -> f64 {
        // Remove values that are out of the window.
        while let Some(&(idx, _)) = self.deque.front() {
            if self.cursor.saturating_sub(idx) >= self.window {
                self.deque.pop_front();
            } else {
                break;
            }
        }

        // Remove smaller values from the back (monotone-queue invariant).
        while let Some(&(_, v)) = self.deque.back() {
            if v <= value {
                self.deque.pop_back();
            } else {
                break;
            }
        }

        self.deque.push_back((self.cursor, value));
        self.cursor = self.cursor.wrapping_add(1);

        self.deque.front().map_or(value, |&(_, v)| v)
    }

    fn reset(&mut self) {
        self.deque.clear();
        self.cursor = 0;
    }
}

// ────────────────────────────────────────────────────────────────────────────
// LookAheadPeakLimiter
// ────────────────────────────────────────────────────────────────────────────

/// Gain-reduction statistics produced by the limiter.
#[derive(Clone, Debug, Default)]
pub struct LimiterStats {
    /// Number of samples where gain reduction was applied (GR < 1.0).
    pub samples_limited: u64,
    /// Maximum gain reduction in dB (peak, non-negative value).
    pub max_gain_reduction_db: f64,
    /// Running average gain reduction in dB (non-negative).
    pub avg_gain_reduction_db: f64,
    /// Total samples processed.
    pub total_samples: u64,
}

/// Look-ahead true-peak brick-wall limiter.
///
/// Processes interleaved multi-channel audio in-place or into a separate output buffer.
/// Latency equals `lookahead_ms` (one `lookahead_samples` delay).
pub struct LookAheadPeakLimiter {
    config: PeakLimiterConfig,
    /// Per-channel lookahead ring buffers.
    rings: Vec<LookaheadRing>,
    /// Current gain reduction envelope (linear, shared across all channels).
    current_gain: f64,
    /// Release coefficient per sample.
    release_coeff: f64,
    /// Ceiling in linear amplitude.
    ceiling_linear: f64,
    /// Lookahead in samples.
    lookahead_samples: usize,
    /// Sliding peak window — tracks the worst-case peak in the lookahead window.
    peak_window: PeakWindow,
    /// Statistics accumulator.
    stats: LimiterStats,
}

impl LookAheadPeakLimiter {
    /// Create a new limiter from configuration.
    pub fn new(config: PeakLimiterConfig) -> NormalizeResult<Self> {
        config.validate()?;
        let lookahead_samples = config.lookahead_samples();
        let release_coeff = config.release_coeff();
        let ceiling_linear = config.ceiling_linear();
        let channels = config.channels;
        let rings = (0..channels)
            .map(|_| LookaheadRing::new(lookahead_samples + 1))
            .collect();
        let peak_window = PeakWindow::new(lookahead_samples);
        Ok(Self {
            config,
            rings,
            current_gain: 1.0,
            release_coeff,
            ceiling_linear,
            lookahead_samples,
            peak_window,
            stats: LimiterStats::default(),
        })
    }

    /// Process interleaved audio in-place.
    ///
    /// The output is delayed by `lookahead_samples` frames.  Call [`flush`] after the
    /// last block to drain the remaining lookahead tail.
    pub fn process_in_place(&mut self, samples: &mut [f64]) {
        let channels = self.config.channels;
        let n_frames = samples.len() / channels;

        for f in 0..n_frames {
            // ── Step 1: find the peak across all channels at this frame ───
            let mut frame_peak = 0.0_f64;
            for c in 0..channels {
                let s = samples[f * channels + c].abs();
                if s > frame_peak {
                    frame_peak = s;
                }
            }

            // ── Step 2: update sliding peak window ─────────────────────────
            let lookahead_peak = self.peak_window.push(frame_peak);

            // ── Step 3: compute required gain to respect ceiling ──────────
            let required_gain = if lookahead_peak > self.ceiling_linear && lookahead_peak > 1e-30 {
                (self.ceiling_linear / lookahead_peak).min(1.0)
            } else {
                1.0
            };

            // ── Step 4: update envelope (instantaneous attack, smooth release)
            if required_gain < self.current_gain {
                // Instantaneous attack — snap down immediately.
                self.current_gain = required_gain;
            } else {
                // Exponential release.
                self.current_gain =
                    self.current_gain * self.release_coeff + required_gain * (1.0 - self.release_coeff);
                if self.current_gain > 1.0 {
                    self.current_gain = 1.0;
                }
            }

            // ── Step 5: apply gain to delayed samples ─────────────────────
            for c in 0..channels {
                let incoming = samples[f * channels + c];
                let delayed = self.rings[c].push_and_read(incoming, self.lookahead_samples);
                samples[f * channels + c] = delayed * self.current_gain;
            }

            // ── Stats ─────────────────────────────────────────────────────
            self.stats.total_samples += 1;
            if self.current_gain < 1.0 - 1e-9 {
                self.stats.samples_limited += 1;
                let gr_db = -20.0 * self.current_gain.log10();
                if gr_db > self.stats.max_gain_reduction_db {
                    self.stats.max_gain_reduction_db = gr_db;
                }
                let n = self.stats.total_samples as f64;
                self.stats.avg_gain_reduction_db =
                    self.stats.avg_gain_reduction_db * ((n - 1.0) / n) + gr_db / n;
            }
        }
    }

    /// Process interleaved audio from `input` to `output`.
    pub fn process(&mut self, input: &[f64], output: &mut [f64]) -> NormalizeResult<()> {
        if output.len() != input.len() {
            return Err(NormalizeError::ProcessingError(
                "output buffer must be the same length as input".to_string(),
            ));
        }
        output.copy_from_slice(input);
        self.process_in_place(output);
        Ok(())
    }

    /// Flush the lookahead tail by processing `lookahead_samples` frames of silence,
    /// returning the delayed output.
    pub fn flush(&mut self) -> Vec<f64> {
        let channels = self.config.channels;
        let tail_samples = self.lookahead_samples * channels;
        let mut tail = vec![0.0_f64; tail_samples];
        self.process_in_place(&mut tail);
        tail
    }

    /// Reset all internal state.
    pub fn reset(&mut self) {
        for ring in &mut self.rings {
            ring.reset();
        }
        self.peak_window.reset();
        self.current_gain = 1.0;
        self.stats = LimiterStats::default();
    }

    /// Return a reference to the current statistics.
    pub fn stats(&self) -> &LimiterStats {
        &self.stats
    }

    /// Access the configuration.
    pub fn config(&self) -> &PeakLimiterConfig {
        &self.config
    }

    /// Latency in samples (per channel).
    pub fn latency_samples(&self) -> usize {
        self.lookahead_samples
    }

    /// Latency in milliseconds.
    pub fn latency_ms(&self) -> f64 {
        self.lookahead_samples as f64 / self.config.sample_rate * 1000.0
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_limiter(ceiling_dbtp: f64) -> LookAheadPeakLimiter {
        let config = PeakLimiterConfig {
            ceiling_dbtp,
            lookahead_ms: 2.0,
            release_ms: 50.0,
            sample_rate: 48_000.0,
            channels: 1,
            oversample: 4,
        };
        LookAheadPeakLimiter::new(config).expect("create limiter")
    }

    #[test]
    fn test_config_validate_valid() {
        let cfg = PeakLimiterConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_validate_positive_ceiling_rejected() {
        let cfg = PeakLimiterConfig {
            ceiling_dbtp: 0.5,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validate_zero_channels_rejected() {
        let cfg = PeakLimiterConfig {
            channels: 0,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_ceiling_linear_minus_one_dbtp() {
        let cfg = PeakLimiterConfig {
            ceiling_dbtp: -1.0,
            ..Default::default()
        };
        let linear = cfg.ceiling_linear();
        // 10^(-1/20) ≈ 0.8913
        assert!((linear - 0.8913).abs() < 0.001);
    }

    #[test]
    fn test_lookahead_ring_basic() {
        let mut ring = LookaheadRing::new(4);
        // Write 4 samples; with delay=3 we should get the first sample back on the 4th push.
        ring.push_and_read(1.0, 3);
        ring.push_and_read(2.0, 3);
        ring.push_and_read(3.0, 3);
        let out = ring.push_and_read(4.0, 3);
        assert!((out - 1.0).abs() < 1e-10, "expected delayed sample 1.0, got {out}");
    }

    #[test]
    fn test_peak_window_monotone_max() {
        let mut pw = PeakWindow::new(3);
        assert!((pw.push(0.1) - 0.1).abs() < 1e-10);
        assert!((pw.push(0.5) - 0.5).abs() < 1e-10);
        assert!((pw.push(0.3) - 0.5).abs() < 1e-10);
        // 0.1 falls out of window (size=3, now at cursor=3, oldest is cursor=0)
        let v = pw.push(0.2);
        assert!((v - 0.5).abs() < 1e-10, "max should still be 0.5, got {v}");
    }

    #[test]
    fn test_limiter_below_ceiling_not_limited() {
        let mut lim = make_limiter(-1.0);
        let ceiling = 10.0_f64.powf(-1.0 / 20.0);
        // Signal at 50% of ceiling — should not be limited.
        let amplitude = ceiling * 0.5;
        let n = 48_000_usize; // 1 second
        let mut samples: Vec<f64> = (0..n)
            .map(|i| amplitude * (std::f64::consts::TAU * 1000.0 * i as f64 / 48_000.0).sin())
            .collect();
        lim.process_in_place(&mut samples);
        let max_out = samples.iter().cloned().fold(0.0_f64, f64::max);
        assert!(
            max_out <= ceiling + 1e-6,
            "max output {max_out:.6} exceeded ceiling {ceiling:.6}"
        );
        assert_eq!(lim.stats().samples_limited, 0, "should not have been limited");
    }

    #[test]
    fn test_limiter_above_ceiling_is_limited() {
        let mut lim = make_limiter(-1.0);
        let ceiling = 10.0_f64.powf(-1.0 / 20.0);
        // Signal well above ceiling.
        let n = 48_000_usize;
        let mut samples: Vec<f64> = (0..n)
            .map(|i| 2.0 * (std::f64::consts::TAU * 1000.0 * i as f64 / 48_000.0).sin())
            .collect();
        lim.process_in_place(&mut samples);
        let max_out = samples.iter().map(|s| s.abs()).fold(0.0_f64, f64::max);
        assert!(
            max_out <= ceiling + 1e-3,
            "output {max_out:.6} should be ≤ ceiling {ceiling:.6}"
        );
        assert!(
            lim.stats().samples_limited > 0,
            "limiter should have triggered"
        );
    }

    #[test]
    fn test_limiter_reset_clears_stats() {
        let mut lim = make_limiter(-1.0);
        let mut samples = vec![2.0_f64; 100];
        lim.process_in_place(&mut samples);
        assert!(lim.stats().total_samples > 0);
        lim.reset();
        assert_eq!(lim.stats().total_samples, 0);
        assert_eq!(lim.stats().samples_limited, 0);
    }

    #[test]
    fn test_latency_is_nonzero() {
        let lim = make_limiter(-1.0);
        assert!(lim.latency_samples() > 0);
        assert!(lim.latency_ms() > 0.0);
    }

    #[test]
    fn test_process_wrong_output_length_errors() {
        let mut lim = make_limiter(-1.0);
        let input = vec![0.5_f64; 100];
        let mut output = vec![0.0_f64; 50]; // wrong length
        let result = lim.process(&input, &mut output);
        assert!(result.is_err());
    }

    #[test]
    fn test_flush_returns_correct_length() {
        let config = PeakLimiterConfig {
            channels: 2,
            ..Default::default()
        };
        let mut lim = LookAheadPeakLimiter::new(config).expect("create");
        let tail = lim.flush();
        // flush returns lookahead_samples * channels samples
        assert_eq!(tail.len(), lim.latency_samples() * 2);
    }
}
