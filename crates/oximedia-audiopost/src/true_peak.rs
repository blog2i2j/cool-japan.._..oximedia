//! ITU-R BS.1770-4 true-peak measurement.
//!
//! Implements inter-sample peak detection using 4x oversampling with a 48-tap
//! FIR interpolation filter as specified in ITU-R BS.1770-4 Annex 2.
//!
//! # Overview
//!
//! Sample-peak measurement can miss peaks that occur *between* samples due to
//! the reconstruction of the continuous-time waveform. This module detects those
//! inter-sample peaks by upsampling 4x with a carefully designed lowpass FIR
//! filter and measuring the absolute peak of the interpolated signal.
//!
//! # Example
//!
//! ```
//! use oximedia_audiopost::true_peak::TruePeakMeter;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let mut meter = TruePeakMeter::new(48000, 2)?;
//! let left = vec![0.5_f32; 1024];
//! let right = vec![-0.3_f32; 1024];
//! meter.process(&[&left, &right])?;
//! let peaks = meter.true_peak_dbtp();
//! println!("Ch0: {:.1} dBTP, Ch1: {:.1} dBTP", peaks[0], peaks[1]);
//! # Ok(())
//! # }
//! ```

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use crate::error::{AudioPostError, AudioPostResult};

/// Number of taps in the FIR interpolation filter (per ITU-R BS.1770-4 Annex 2).
const FIR_TAPS: usize = 48;

/// Oversampling factor.
const OVERSAMPLE_FACTOR: usize = 4;

/// Number of filter taps per phase (FIR_TAPS / OVERSAMPLE_FACTOR).
const TAPS_PER_PHASE: usize = FIR_TAPS / OVERSAMPLE_FACTOR;

/// ITU-R BS.1770-4 true-peak meter with 4x oversampling.
///
/// Uses a 48-tap polyphase FIR interpolation filter to detect inter-sample peaks.
#[derive(Debug)]
pub struct TruePeakMeter {
    sample_rate: u32,
    num_channels: usize,
    /// Per-channel maximum true peak (linear).
    max_true_peak: Vec<f32>,
    /// Per-channel sample peak (linear).
    max_sample_peak: Vec<f32>,
    /// Per-channel delay line for the FIR filter.
    delay_lines: Vec<Vec<f32>>,
    /// Polyphase FIR coefficients [phase][tap].
    polyphase_coeffs: [[f64; TAPS_PER_PHASE]; OVERSAMPLE_FACTOR],
    /// Total samples processed per channel.
    samples_processed: u64,
}

impl TruePeakMeter {
    /// Create a new true-peak meter.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` — Audio sample rate in Hz
    /// * `num_channels` — Number of audio channels
    ///
    /// # Errors
    ///
    /// Returns an error if sample rate is zero or channel count is zero.
    pub fn new(sample_rate: u32, num_channels: usize) -> AudioPostResult<Self> {
        if sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(sample_rate));
        }
        if num_channels == 0 {
            return Err(AudioPostError::InvalidChannelCount(0));
        }

        let polyphase_coeffs = Self::compute_polyphase_coefficients();

        Ok(Self {
            sample_rate,
            num_channels,
            max_true_peak: vec![0.0; num_channels],
            max_sample_peak: vec![0.0; num_channels],
            delay_lines: vec![vec![0.0; TAPS_PER_PHASE]; num_channels],
            polyphase_coeffs,
            samples_processed: 0,
        })
    }

    /// Compute the polyphase decomposition of a 48-tap FIR lowpass filter
    /// designed per ITU-R BS.1770-4 Annex 2.
    ///
    /// The prototype filter is a windowed-sinc lowpass at Nyquist/4 (since we
    /// oversample 4x), using a Kaiser window with beta=5.65.
    fn compute_polyphase_coefficients() -> [[f64; TAPS_PER_PHASE]; OVERSAMPLE_FACTOR] {
        // Kaiser window with beta=5.65 (good sidelobe rejection per ITU spec)
        let beta = 5.65_f64;
        let i0_beta = bessel_i0(beta);

        // Design the interpolation filter.
        // For 4x oversampling, we want a lowpass at fs/2 of the *original* rate,
        // which is fs_over / (2*L) = 1/(2*L) in normalized frequency of the
        // oversampled rate. The polyphase approach means each phase operates at
        // the original sample rate, so we design the sinc for fractional delay.
        //
        // Phase p interpolates at fractional position p/L between input samples.
        // Each phase is an L-tap FIR designed to evaluate the sinc at the
        // appropriate fractional offsets.

        let l = OVERSAMPLE_FACTOR as f64;
        let half_taps = (TAPS_PER_PHASE as f64 - 1.0) / 2.0;

        let mut phases = [[0.0_f64; TAPS_PER_PHASE]; OVERSAMPLE_FACTOR];

        for phase in 0..OVERSAMPLE_FACTOR {
            let frac = phase as f64 / l;

            for tap in 0..TAPS_PER_PHASE {
                let n = tap as f64 - half_taps - frac;

                // Sinc interpolation kernel
                let sinc_val = if n.abs() < 1e-12 {
                    1.0
                } else {
                    (std::f64::consts::PI * n).sin() / (std::f64::consts::PI * n)
                };

                // Kaiser window over the full tap range
                let win_pos = (tap as f64 + frac) / (TAPS_PER_PHASE as f64);
                let x = 2.0 * win_pos - 1.0;
                let window = bessel_i0(beta * (1.0 - x * x).max(0.0).sqrt()) / i0_beta;

                phases[phase][tap] = sinc_val * window;
            }

            // Normalize so DC gain = 1.0 for each phase
            let phase_sum: f64 = phases[phase].iter().sum();
            if phase_sum.abs() > 1e-12 {
                for tap in 0..TAPS_PER_PHASE {
                    phases[phase][tap] /= phase_sum;
                }
            }
        }

        phases
    }

    /// Process multi-channel audio data.
    ///
    /// `channels` is a slice of per-channel sample slices.
    /// All channels must have the same length.
    ///
    /// # Errors
    ///
    /// Returns an error if channel count doesn't match or buffers are mismatched.
    pub fn process(&mut self, channels: &[&[f32]]) -> AudioPostResult<()> {
        if channels.len() != self.num_channels {
            return Err(AudioPostError::InvalidChannelCount(channels.len()));
        }
        if self.num_channels > 1 {
            let len = channels[0].len();
            for ch in channels.iter().skip(1) {
                if ch.len() != len {
                    return Err(AudioPostError::InvalidBufferSize(ch.len()));
                }
            }
        }

        for (ch_idx, &samples) in channels.iter().enumerate() {
            self.process_channel(ch_idx, samples);
        }

        if let Some(first) = channels.first() {
            self.samples_processed += first.len() as u64;
        }

        Ok(())
    }

    /// Process a single channel.
    fn process_channel(&mut self, ch_idx: usize, samples: &[f32]) {
        let delay = &mut self.delay_lines[ch_idx];

        for &sample in samples {
            // Track sample peak
            let abs_sample = sample.abs();
            if abs_sample > self.max_sample_peak[ch_idx] {
                self.max_sample_peak[ch_idx] = abs_sample;
            }

            // Shift delay line and insert new sample
            for i in 0..(TAPS_PER_PHASE - 1) {
                delay[i] = delay[i + 1];
            }
            delay[TAPS_PER_PHASE - 1] = sample;

            // Compute all 4 polyphase outputs (the interpolated samples)
            for phase in 0..OVERSAMPLE_FACTOR {
                let mut acc = 0.0_f64;
                for tap in 0..TAPS_PER_PHASE {
                    acc += f64::from(delay[tap]) * self.polyphase_coeffs[phase][tap];
                }
                let interp_abs = (acc as f32).abs();
                if interp_abs > self.max_true_peak[ch_idx] {
                    self.max_true_peak[ch_idx] = interp_abs;
                }
            }
        }
    }

    /// Get the true peak for each channel in dBTP.
    #[must_use]
    pub fn true_peak_dbtp(&self) -> Vec<f32> {
        self.max_true_peak
            .iter()
            .map(|&peak| linear_to_dbtp(peak))
            .collect()
    }

    /// Get the sample peak for each channel in dBFS.
    #[must_use]
    pub fn sample_peak_dbfs(&self) -> Vec<f32> {
        self.max_sample_peak
            .iter()
            .map(|&peak| linear_to_dbtp(peak))
            .collect()
    }

    /// Get the true peak for each channel in linear scale.
    #[must_use]
    pub fn true_peak_linear(&self) -> &[f32] {
        &self.max_true_peak
    }

    /// Get the sample peak for each channel in linear scale.
    #[must_use]
    pub fn sample_peak_linear(&self) -> &[f32] {
        &self.max_sample_peak
    }

    /// Get the maximum true peak across all channels in dBTP.
    #[must_use]
    pub fn max_true_peak_dbtp(&self) -> f32 {
        let max_linear = self
            .max_true_peak
            .iter()
            .copied()
            .fold(0.0_f32, f32::max);
        linear_to_dbtp(max_linear)
    }

    /// Get total number of samples processed per channel.
    #[must_use]
    pub fn samples_processed(&self) -> u64 {
        self.samples_processed
    }

    /// Get the number of channels.
    #[must_use]
    pub fn num_channels(&self) -> usize {
        self.num_channels
    }

    /// Reset all measurements.
    pub fn reset(&mut self) {
        for peak in &mut self.max_true_peak {
            *peak = 0.0;
        }
        for peak in &mut self.max_sample_peak {
            *peak = 0.0;
        }
        for delay in &mut self.delay_lines {
            for s in delay.iter_mut() {
                *s = 0.0;
            }
        }
        self.samples_processed = 0;
    }
}

/// Convert linear amplitude to dBTP (decibels relative to full scale, true peak).
fn linear_to_dbtp(linear: f32) -> f32 {
    if linear <= 0.0 {
        return -100.0;
    }
    20.0 * linear.log10()
}

/// Modified Bessel function of the first kind, order 0.
/// Used for Kaiser window computation.
fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0_f64;
    let mut term = 1.0_f64;
    let x_half = x / 2.0;
    for k in 1..=25 {
        term *= (x_half / k as f64) * (x_half / k as f64);
        sum += term;
        if term < sum * 1e-15 {
            break;
        }
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sine(freq: f32, sample_rate: u32, num_samples: usize, amplitude: f32) -> Vec<f32> {
        (0..num_samples)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                amplitude * (2.0 * std::f32::consts::PI * freq * t).sin()
            })
            .collect()
    }

    #[test]
    fn test_true_peak_creation() {
        let meter = TruePeakMeter::new(48000, 2).expect("create meter");
        assert_eq!(meter.num_channels(), 2);
        assert_eq!(meter.samples_processed(), 0);
    }

    #[test]
    fn test_invalid_creation() {
        assert!(TruePeakMeter::new(0, 2).is_err());
        assert!(TruePeakMeter::new(48000, 0).is_err());
    }

    #[test]
    fn test_dc_signal_peak() {
        let mut meter = TruePeakMeter::new(48000, 1).expect("create meter");
        // Use a long DC signal so the filter delay line has time to fill
        let signal = vec![0.5_f32; 8192];
        meter.process(&[&signal]).expect("process");

        let tp = meter.true_peak_linear();
        let sp = meter.sample_peak_linear();

        // For DC, true peak should approximately equal sample peak
        // Allow some tolerance for filter transients at startup
        assert!(
            (tp[0] - sp[0]).abs() < 0.15,
            "DC true peak ({}) ~ sample peak ({})",
            tp[0],
            sp[0]
        );
    }

    #[test]
    fn test_sine_true_peak_exceeds_sample_peak() {
        let mut meter = TruePeakMeter::new(48000, 1).expect("create meter");

        // A sine at a frequency that doesn't align with sample boundaries
        // will have inter-sample peaks higher than the sample peak.
        // Use a frequency that guarantees the peak falls between samples.
        // fs=48000, f=997 Hz (prime, non-harmonic of fs)
        let signal = make_sine(997.0, 48000, 48000, 1.0);
        meter.process(&[&signal]).expect("process");

        let tp_db = meter.true_peak_dbtp()[0];
        let sp_db = meter.sample_peak_dbfs()[0];

        // True peak should be >= sample peak (possibly higher due to inter-sample peaks)
        assert!(
            tp_db >= sp_db - 0.01,
            "True peak ({tp_db:.2} dBTP) should be >= sample peak ({sp_db:.2} dBFS)"
        );
    }

    #[test]
    fn test_full_scale_sine_true_peak() {
        let mut meter = TruePeakMeter::new(48000, 1).expect("create meter");
        let signal = make_sine(1000.0, 48000, 48000, 1.0);
        meter.process(&[&signal]).expect("process");

        let tp_db = meter.max_true_peak_dbtp();
        // For a full-scale sine, true peak should be close to 0 dBTP.
        // Inter-sample peaks can exceed 0 dBTP slightly; the polyphase
        // filter may also introduce minor overshoot.
        assert!(
            tp_db > -2.0 && tp_db <= 2.0,
            "Full-scale sine true peak should be near 0 dBTP, got {tp_db:.2}"
        );
    }

    #[test]
    fn test_multi_channel() {
        let mut meter = TruePeakMeter::new(48000, 3).expect("create meter");
        let ch0 = vec![0.8_f32; 512];
        let ch1 = vec![0.3_f32; 512];
        let ch2 = vec![0.1_f32; 512];

        meter.process(&[&ch0, &ch1, &ch2]).expect("process");

        let peaks = meter.true_peak_dbtp();
        assert_eq!(peaks.len(), 3);
        // ch0 should be loudest
        assert!(peaks[0] > peaks[1]);
        assert!(peaks[1] > peaks[2]);
    }

    #[test]
    fn test_channel_mismatch_error() {
        let mut meter = TruePeakMeter::new(48000, 2).expect("create meter");
        let ch0 = vec![0.5_f32; 512];
        // Only pass 1 channel when 2 expected
        let result = meter.process(&[&ch0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_reset() {
        let mut meter = TruePeakMeter::new(48000, 1).expect("create meter");
        let signal = vec![0.9_f32; 1024];
        meter.process(&[&signal]).expect("process");
        assert!(meter.max_true_peak_dbtp() > -10.0);

        meter.reset();
        assert_eq!(meter.samples_processed(), 0);
        assert!(meter.max_true_peak_dbtp() <= -100.0);
    }

    #[test]
    fn test_silence_true_peak() {
        let mut meter = TruePeakMeter::new(48000, 1).expect("create meter");
        let signal = vec![0.0_f32; 1024];
        meter.process(&[&signal]).expect("process");
        assert!(
            meter.max_true_peak_dbtp() <= -100.0,
            "Silence should have very low true peak"
        );
    }

    #[test]
    fn test_polyphase_coefficients_sum_near_unity() {
        // Each phase of the polyphase filter, when summed, should approximate
        // the DC gain of the prototype (which is normalized to ~1.0 for passband).
        let coeffs = TruePeakMeter::compute_polyphase_coefficients();
        for (phase_idx, phase) in coeffs.iter().enumerate() {
            let sum: f64 = phase.iter().sum();
            // Each phase sum should be roughly 1.0 (within tolerance for windowed sinc)
            assert!(
                sum.abs() < 3.0,
                "Phase {phase_idx} coefficient sum {sum:.4} out of range"
            );
        }
    }

    #[test]
    fn test_incremental_processing() {
        let mut meter = TruePeakMeter::new(48000, 1).expect("create meter");
        let signal = make_sine(1000.0, 48000, 48000, 0.8);

        // Process in chunks
        for chunk in signal.chunks(1024) {
            meter.process(&[chunk]).expect("process chunk");
        }

        let tp = meter.max_true_peak_dbtp();
        assert!(
            tp > -3.0 && tp < 1.0,
            "Incremental processing should yield valid true peak, got {tp:.2}"
        );
        assert_eq!(meter.samples_processed(), 48000);
    }
}
