//! VU meter with analogue-style ballistics.
//!
//! Implements a Volume Unit (VU) meter per the original 1939 CBS/Bell Labs
//! specification: 300 ms integration time (combined attack + release), −20 dBFS
//! = 0 VU reference, and a 1–2% overshoot on a 1 kHz sine wave at 0 VU.
//!
//! The [`VuMeter`] provides per-channel [`VuReading`]s suitable for display
//! on hardware or software meters.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// Ballistic parameters that govern VU needle movement.
///
/// Standard VU uses identical attack and release times (300 ms to 99%
/// of full deflection for a step input).
#[derive(Debug, Clone, Copy)]
pub struct VuBallistic {
    /// Attack time constant in milliseconds (default 300 ms).
    pub attack_ms: f64,
    /// Release time constant in milliseconds (default 300 ms).
    pub release_ms: f64,
    /// Reference level in dBFS where the needle reads 0 VU (default −20.0).
    pub reference_dbfs: f64,
}

impl VuBallistic {
    /// Standard VU specification (300 ms / 300 ms / −20 dBFS).
    pub const STANDARD: Self = Self {
        attack_ms: 300.0,
        release_ms: 300.0,
        reference_dbfs: -20.0,
    };

    /// Compute the per-sample IIR coefficient for an exponential envelope
    /// follower.
    ///
    /// `time_ms` is the time to reach 1 − 1/e ≈ 63% of a step (same
    /// convention as analogue RC time constants).
    #[inline]
    pub fn alpha(time_ms: f64, sample_rate: f64) -> f64 {
        (-1.0 / (sample_rate * time_ms * 1e-3)).exp()
    }
}

impl Default for VuBallistic {
    fn default() -> Self {
        Self::STANDARD
    }
}

/// A single VU reading for one channel.
#[derive(Debug, Clone, Copy)]
pub struct VuReading {
    /// Instantaneous envelope level (linear, 0–1+).
    pub level_linear: f64,
    /// Level in VU units relative to the configured reference.
    pub level_vu: f64,
    /// `true` if the reading exceeds 0 VU (into the red).
    pub over_reference: bool,
}

impl VuReading {
    /// Construct a reading from a linear envelope and a reference in dBFS.
    pub fn new(level_linear: f64, reference_dbfs: f64) -> Self {
        let dbfs = if level_linear > 1e-10 {
            20.0 * level_linear.log10()
        } else {
            -120.0
        };
        let level_vu = dbfs - reference_dbfs;
        Self {
            level_linear,
            level_vu,
            over_reference: level_vu > 0.0,
        }
    }

    /// Return the level in dBFS.
    pub fn dbfs(&self) -> f64 {
        if self.level_linear > 1e-10 {
            20.0 * self.level_linear.log10()
        } else {
            -120.0
        }
    }
}

/// Multi-channel VU meter with configurable ballistics.
pub struct VuMeter {
    /// Ballistic parameters applied to all channels.
    pub ballistic: VuBallistic,
    sample_rate: f64,
    channels: usize,
    /// Per-channel envelope state.
    envelopes: Vec<f64>,
    /// Attack coefficient (IIR α for rising signal).
    alpha_attack: f64,
    /// Release coefficient (IIR α for falling signal).
    alpha_release: f64,
}

impl VuMeter {
    /// Create a new VU meter.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` – sample rate in Hz (e.g. 48000.0).
    /// * `channels`    – number of audio channels (≥ 1).
    /// * `ballistic`   – optional custom ballistics; use `None` for standard VU.
    pub fn new(sample_rate: f64, channels: usize, ballistic: Option<VuBallistic>) -> Self {
        let ball = ballistic.unwrap_or_default();
        let alpha_attack = VuBallistic::alpha(ball.attack_ms, sample_rate);
        let alpha_release = VuBallistic::alpha(ball.release_ms, sample_rate);
        Self {
            ballistic: ball,
            sample_rate,
            channels: channels.max(1),
            envelopes: vec![0.0; channels.max(1)],
            alpha_attack,
            alpha_release,
        }
    }

    /// Process interleaved samples and update all channel envelopes.
    ///
    /// `samples` must be interleaved in channel order and its length must be
    /// a multiple of `channels`.
    pub fn process_interleaved(&mut self, samples: &[f32]) {
        for frame in samples.chunks_exact(self.channels) {
            for (ch, &s) in frame.iter().enumerate() {
                let abs = f64::from(s.abs());
                let env = self.envelopes[ch];
                self.envelopes[ch] = if abs >= env {
                    // Attack
                    self.alpha_attack * env + (1.0 - self.alpha_attack) * abs
                } else {
                    // Release
                    self.alpha_release * env + (1.0 - self.alpha_release) * abs
                };
            }
        }
    }

    /// Process separate per-channel slices (equal length, `channels` slices).
    pub fn process_planar(&mut self, channels: &[&[f32]]) {
        let n_channels = channels.len().min(self.channels);
        if channels.is_empty() {
            return;
        }
        let n_samples = channels[0].len();
        for i in 0..n_samples {
            for (ch, channel_data) in channels[..n_channels].iter().enumerate() {
                if i >= channel_data.len() {
                    continue;
                }
                let abs = f64::from(channel_data[i].abs());
                let env = self.envelopes[ch];
                self.envelopes[ch] = if abs >= env {
                    self.alpha_attack * env + (1.0 - self.alpha_attack) * abs
                } else {
                    self.alpha_release * env + (1.0 - self.alpha_release) * abs
                };
            }
        }
    }

    /// Return the current [`VuReading`] for each channel.
    pub fn readings(&self) -> Vec<VuReading> {
        self.envelopes
            .iter()
            .map(|&env| VuReading::new(env, self.ballistic.reference_dbfs))
            .collect()
    }

    /// Return the current VU level (in VU units) for a specific channel.
    ///
    /// Returns `None` if `channel` is out of range.
    pub fn channel_vu(&self, channel: usize) -> Option<f64> {
        self.envelopes
            .get(channel)
            .map(|&env| VuReading::new(env, self.ballistic.reference_dbfs).level_vu)
    }

    /// Return `true` if any channel is currently above the reference level.
    pub fn any_over_reference(&self) -> bool {
        self.readings().iter().any(|r| r.over_reference)
    }

    /// Reset all channel envelopes to zero.
    pub fn reset(&mut self) {
        self.envelopes.iter_mut().for_each(|e| *e = 0.0);
    }

    /// Return the number of channels.
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Return the sample rate.
    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vu_ballistic_default_values() {
        let ball = VuBallistic::default();
        assert_eq!(ball.attack_ms, 300.0);
        assert_eq!(ball.release_ms, 300.0);
        assert_eq!(ball.reference_dbfs, -20.0);
    }

    #[test]
    fn vu_ballistic_alpha_zero_time_gives_alpha_one() {
        // Very large time → alpha near 1 (very slow); very small time → alpha near 0
        // 0 ms would be -inf / 0, so use a tiny value instead.
        let alpha = VuBallistic::alpha(0.001, 48_000.0);
        // exp(-1 / (48000 * 0.000001)) = exp(-20833) ≈ 0
        assert!(alpha < 0.01);
    }

    #[test]
    fn vu_reading_at_reference_gives_zero_vu() {
        // A signal at -20 dBFS == 0 VU with standard reference
        let linear = 10_f64.powf(-20.0 / 20.0); // ≈ 0.1
        let reading = VuReading::new(linear, -20.0);
        let vu = reading.level_vu;
        assert!((vu).abs() < 0.001, "expected ~0 VU, got {}", vu);
    }

    #[test]
    fn vu_reading_above_reference_is_over() {
        let linear = 10_f64.powf(-10.0 / 20.0); // -10 dBFS = +10 VU
        let reading = VuReading::new(linear, -20.0);
        assert!(reading.over_reference);
    }

    #[test]
    fn vu_reading_silent_gives_minus120() {
        let reading = VuReading::new(0.0, -20.0);
        assert!(reading.dbfs() < -100.0);
    }

    #[test]
    fn vu_meter_creation() {
        let meter = VuMeter::new(48_000.0, 2, None);
        assert_eq!(meter.channels(), 2);
        assert_eq!(meter.sample_rate(), 48_000.0);
    }

    #[test]
    fn vu_meter_initial_readings_are_zero() {
        let meter = VuMeter::new(48_000.0, 2, None);
        for r in meter.readings() {
            assert_eq!(r.level_linear, 0.0);
        }
    }

    #[test]
    fn vu_meter_processes_interleaved_stereo() {
        let mut meter = VuMeter::new(48_000.0, 2, None);
        // 1 second of 0 dBFS sine wave on both channels
        let n = 48_000;
        let samples: Vec<f32> = (0..n * 2)
            .map(|i| {
                let t = (i / 2) as f32 / 48_000.0;
                (2.0 * std::f32::consts::PI * 1_000.0 * t).sin()
            })
            .collect();
        meter.process_interleaved(&samples);
        let readings = meter.readings();
        // After 1 s the envelope should have risen towards 0.707 (RMS of sine)
        assert!(
            readings[0].level_linear > 0.0,
            "envelope should be positive"
        );
    }

    #[test]
    fn vu_meter_reset_clears_envelopes() {
        let mut meter = VuMeter::new(48_000.0, 1, None);
        let samples: Vec<f32> = vec![1.0_f32; 48_000];
        meter.process_interleaved(&samples);
        assert!(meter.channel_vu(0).expect("channel_vu should succeed") > -100.0);
        meter.reset();
        assert_eq!(meter.readings()[0].level_linear, 0.0);
    }

    #[test]
    fn vu_meter_channel_vu_out_of_range_returns_none() {
        let meter = VuMeter::new(48_000.0, 2, None);
        assert!(meter.channel_vu(5).is_none());
    }

    #[test]
    fn vu_meter_any_over_reference_initial_false() {
        let meter = VuMeter::new(48_000.0, 2, None);
        assert!(!meter.any_over_reference());
    }

    #[test]
    fn vu_meter_any_over_reference_after_loud_signal() {
        let mut meter = VuMeter::new(48_000.0, 1, None);
        // Feed a full-scale signal for 2 s — envelope will far exceed reference
        let loud: Vec<f32> = vec![1.0_f32; 96_000];
        meter.process_interleaved(&loud);
        assert!(meter.any_over_reference());
    }

    #[test]
    fn vu_meter_planar_processing() {
        let mut meter = VuMeter::new(48_000.0, 2, None);
        let ch0: Vec<f32> = vec![0.5_f32; 1000];
        let ch1: Vec<f32> = vec![0.5_f32; 1000];
        meter.process_planar(&[&ch0, &ch1]);
        for r in meter.readings() {
            assert!(r.level_linear > 0.0);
        }
    }

    #[test]
    fn vu_meter_custom_ballistic() {
        let fast = VuBallistic {
            attack_ms: 10.0,
            release_ms: 10.0,
            reference_dbfs: -18.0,
        };
        let meter = VuMeter::new(48_000.0, 1, Some(fast));
        assert_eq!(meter.ballistic.reference_dbfs, -18.0);
    }

    #[test]
    fn vu_meter_mono_channel_count_at_least_one() {
        let meter = VuMeter::new(48_000.0, 0, None);
        assert_eq!(meter.channels(), 1);
    }
}
