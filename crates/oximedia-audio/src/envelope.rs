//! ADSR envelope generator for audio synthesis and dynamics.
//!
//! This module provides a classic Attack-Decay-Sustain-Release envelope
//! generator ([`EnvelopeGenerator`]) along with supporting configuration
//! ([`EnvelopeConfig`]) and shape types ([`EnvelopeShape`]).
//!
//! # Example
//!
//! ```
//! use oximedia_audio::envelope::{EnvelopeConfig, EnvelopeGenerator, EnvelopeShape};
//!
//! let config = EnvelopeConfig {
//!     attack_ms: 10.0,
//!     decay_ms: 50.0,
//!     sustain_level: 0.7,
//!     release_ms: 200.0,
//!     shape: EnvelopeShape::Linear,
//! };
//! let mut env = EnvelopeGenerator::new(config, 48_000.0);
//! env.trigger();
//! let v = env.next_sample();
//! assert!(v >= 0.0);
//! ```

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Shape of envelope segments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EnvelopeShape {
    /// Straight-line ramp.
    Linear,
    /// Exponential curve (more natural for audio).
    Exponential,
}

/// Current phase of the envelope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EnvelopePhase {
    /// Envelope is idle (output 0).
    Idle,
    /// Attack phase (rising to 1.0).
    Attack,
    /// Decay phase (falling to sustain level).
    Decay,
    /// Sustain phase (holding).
    Sustain,
    /// Release phase (falling to 0).
    Release,
}

/// Configuration for an ADSR envelope.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnvelopeConfig {
    /// Attack time in milliseconds.
    pub attack_ms: f32,
    /// Decay time in milliseconds.
    pub decay_ms: f32,
    /// Sustain level (0.0 -- 1.0).
    pub sustain_level: f32,
    /// Release time in milliseconds.
    pub release_ms: f32,
    /// Curve shape for all segments.
    pub shape: EnvelopeShape,
}

impl Default for EnvelopeConfig {
    fn default() -> Self {
        Self {
            attack_ms: 10.0,
            decay_ms: 100.0,
            sustain_level: 0.7,
            release_ms: 300.0,
            shape: EnvelopeShape::Linear,
        }
    }
}

impl EnvelopeConfig {
    /// Create a percussive envelope with no sustain.
    pub fn percussive(attack_ms: f32, decay_ms: f32) -> Self {
        Self {
            attack_ms,
            decay_ms,
            sustain_level: 0.0,
            release_ms: decay_ms,
            shape: EnvelopeShape::Exponential,
        }
    }

    /// Create a pad-style envelope with long attack and release.
    pub fn pad() -> Self {
        Self {
            attack_ms: 500.0,
            decay_ms: 200.0,
            sustain_level: 0.8,
            release_ms: 1000.0,
            shape: EnvelopeShape::Linear,
        }
    }
}

// ---------------------------------------------------------------------------
// Generator
// ---------------------------------------------------------------------------

/// ADSR envelope generator.
///
/// Call [`trigger`](Self::trigger) to start the envelope, and
/// [`release`](Self::release) to begin the release phase. Each call to
/// [`next_sample`](Self::next_sample) advances by one sample.
#[derive(Debug, Clone)]
pub struct EnvelopeGenerator {
    config: EnvelopeConfig,
    sample_rate: f32,
    phase: EnvelopePhase,
    /// Current output level.
    level: f32,
    /// Sample counter within the current phase.
    counter: usize,
    /// Pre-computed segment lengths in samples.
    attack_samples: usize,
    decay_samples: usize,
    release_samples: usize,
    /// Level at the instant release was triggered.
    release_start_level: f32,
}

impl EnvelopeGenerator {
    /// Create a new envelope generator.
    pub fn new(config: EnvelopeConfig, sample_rate: f32) -> Self {
        let ms_to_samples = |ms: f32| -> usize {
            #[allow(clippy::cast_precision_loss)]
            let s = (ms * 0.001 * sample_rate).round() as usize;
            s.max(1)
        };
        Self {
            attack_samples: ms_to_samples(config.attack_ms),
            decay_samples: ms_to_samples(config.decay_ms),
            release_samples: ms_to_samples(config.release_ms),
            config,
            sample_rate,
            phase: EnvelopePhase::Idle,
            level: 0.0,
            counter: 0,
            release_start_level: 0.0,
        }
    }

    /// Return the current phase.
    pub fn phase(&self) -> EnvelopePhase {
        self.phase
    }

    /// Return the current output level.
    pub fn level(&self) -> f32 {
        self.level
    }

    /// Return the sample rate.
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    /// Begin the attack phase.
    pub fn trigger(&mut self) {
        self.phase = EnvelopePhase::Attack;
        self.counter = 0;
    }

    /// Begin the release phase from whatever level the envelope is currently at.
    pub fn release(&mut self) {
        if self.phase != EnvelopePhase::Idle && self.phase != EnvelopePhase::Release {
            self.release_start_level = self.level;
            self.phase = EnvelopePhase::Release;
            self.counter = 0;
        }
    }

    /// Force the envelope back to idle immediately.
    pub fn reset(&mut self) {
        self.phase = EnvelopePhase::Idle;
        self.level = 0.0;
        self.counter = 0;
    }

    /// Return `true` if the envelope is active (not idle).
    pub fn is_active(&self) -> bool {
        self.phase != EnvelopePhase::Idle
    }

    /// Advance by one sample and return the envelope value.
    pub fn next_sample(&mut self) -> f32 {
        match self.phase {
            EnvelopePhase::Idle => {
                self.level = 0.0;
            }
            EnvelopePhase::Attack => {
                #[allow(clippy::cast_precision_loss)]
                let t = self.counter as f32 / self.attack_samples as f32;
                self.level = self.apply_shape(t);
                self.counter += 1;
                if self.counter >= self.attack_samples {
                    self.level = 1.0;
                    self.phase = EnvelopePhase::Decay;
                    self.counter = 0;
                }
            }
            EnvelopePhase::Decay => {
                #[allow(clippy::cast_precision_loss)]
                let t = self.counter as f32 / self.decay_samples as f32;
                let shaped = self.apply_shape(t);
                self.level = 1.0 - shaped * (1.0 - self.config.sustain_level);
                self.counter += 1;
                if self.counter >= self.decay_samples {
                    self.level = self.config.sustain_level;
                    self.phase = EnvelopePhase::Sustain;
                    self.counter = 0;
                }
            }
            EnvelopePhase::Sustain => {
                self.level = self.config.sustain_level;
            }
            EnvelopePhase::Release => {
                #[allow(clippy::cast_precision_loss)]
                let t = self.counter as f32 / self.release_samples as f32;
                let shaped = self.apply_shape(t);
                self.level = self.release_start_level * (1.0 - shaped);
                self.counter += 1;
                if self.counter >= self.release_samples {
                    self.level = 0.0;
                    self.phase = EnvelopePhase::Idle;
                }
            }
        }
        self.level
    }

    /// Apply curve shape (0..1 input, 0..1 output).
    fn apply_shape(&self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        match self.config.shape {
            EnvelopeShape::Linear => t,
            EnvelopeShape::Exponential => {
                // Attempt a natural exponential ramp.
                // Use 1 - e^(-5t) normalised so f(1) ~ 1.
                let raw = 1.0 - (-5.0 * t).exp();
                let norm = 1.0 - (-5.0_f32).exp();
                raw / norm
            }
        }
    }

    /// Generate a full envelope cycle (trigger + sustain_samples + release)
    /// and return the sample buffer.
    pub fn generate_cycle(&mut self, sustain_samples: usize) -> Vec<f32> {
        let total =
            self.attack_samples + self.decay_samples + sustain_samples + self.release_samples;
        let mut buf = Vec::with_capacity(total);
        self.trigger();
        for _ in 0..(self.attack_samples + self.decay_samples + sustain_samples) {
            buf.push(self.next_sample());
        }
        self.release();
        for _ in 0..self.release_samples {
            buf.push(self.next_sample());
        }
        buf
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_gen() -> EnvelopeGenerator {
        EnvelopeGenerator::new(EnvelopeConfig::default(), 48000.0)
    }

    #[test]
    fn test_idle_output_zero() {
        let mut env = default_gen();
        assert!((env.next_sample()).abs() < 1e-6);
        assert_eq!(env.phase(), EnvelopePhase::Idle);
    }

    #[test]
    fn test_trigger_starts_attack() {
        let mut env = default_gen();
        env.trigger();
        assert_eq!(env.phase(), EnvelopePhase::Attack);
    }

    #[test]
    fn test_attack_rises_to_one() {
        let config = EnvelopeConfig {
            attack_ms: 10.0,
            decay_ms: 100.0,
            sustain_level: 0.5,
            release_ms: 10.0,
            shape: EnvelopeShape::Linear,
        };
        let mut env = EnvelopeGenerator::new(config, 48000.0);
        env.trigger();
        // Run enough samples to finish attack (10ms * 48 samples/ms = 480 samples)
        for _ in 0..500 {
            env.next_sample();
        }
        // Should have reached 1.0 during attack and now be in decay
        assert!(env.level() >= 0.99 || env.phase() == EnvelopePhase::Decay);
    }

    #[test]
    fn test_sustain_holds_level() {
        let config = EnvelopeConfig {
            attack_ms: 1.0,
            decay_ms: 1.0,
            sustain_level: 0.6,
            release_ms: 1.0,
            shape: EnvelopeShape::Linear,
        };
        let mut env = EnvelopeGenerator::new(config, 48000.0);
        env.trigger();
        // Run enough samples to get past attack+decay
        for _ in 0..500 {
            env.next_sample();
        }
        assert_eq!(env.phase(), EnvelopePhase::Sustain);
        assert!((env.level() - 0.6).abs() < 1e-4);
    }

    #[test]
    fn test_release_decays_to_zero() {
        let config = EnvelopeConfig {
            attack_ms: 1.0,
            decay_ms: 1.0,
            sustain_level: 0.7,
            release_ms: 10.0,
            shape: EnvelopeShape::Linear,
        };
        let mut env = EnvelopeGenerator::new(config, 48000.0);
        env.trigger();
        for _ in 0..1000 {
            env.next_sample();
        }
        env.release();
        for _ in 0..2000 {
            env.next_sample();
        }
        assert_eq!(env.phase(), EnvelopePhase::Idle);
        assert!(env.level().abs() < 1e-4);
    }

    #[test]
    fn test_reset_returns_to_idle() {
        let mut env = default_gen();
        env.trigger();
        env.next_sample();
        env.reset();
        assert_eq!(env.phase(), EnvelopePhase::Idle);
        assert!(env.level().abs() < 1e-6);
    }

    #[test]
    fn test_is_active() {
        let mut env = default_gen();
        assert!(!env.is_active());
        env.trigger();
        assert!(env.is_active());
        env.reset();
        assert!(!env.is_active());
    }

    #[test]
    fn test_exponential_shape() {
        let config = EnvelopeConfig {
            attack_ms: 10.0,
            decay_ms: 10.0,
            sustain_level: 0.5,
            release_ms: 10.0,
            shape: EnvelopeShape::Exponential,
        };
        let mut env = EnvelopeGenerator::new(config, 48000.0);
        env.trigger();
        let first = env.next_sample();
        assert!(first >= 0.0);
    }

    #[test]
    fn test_percussive_preset() {
        let config = EnvelopeConfig::percussive(5.0, 50.0);
        assert!((config.sustain_level).abs() < 1e-6);
        assert_eq!(config.shape, EnvelopeShape::Exponential);
    }

    #[test]
    fn test_pad_preset() {
        let config = EnvelopeConfig::pad();
        assert!(config.attack_ms > 100.0);
        assert!(config.release_ms > 500.0);
    }

    #[test]
    fn test_default_config() {
        let config = EnvelopeConfig::default();
        assert!((config.sustain_level - 0.7).abs() < 1e-6);
        assert!(config.attack_ms > 0.0);
    }

    #[test]
    fn test_generate_cycle() {
        let config = EnvelopeConfig {
            attack_ms: 1.0,
            decay_ms: 1.0,
            sustain_level: 0.5,
            release_ms: 1.0,
            shape: EnvelopeShape::Linear,
        };
        let mut env = EnvelopeGenerator::new(config, 1000.0);
        let buf = env.generate_cycle(10);
        assert!(!buf.is_empty());
        // last sample should be near zero
        assert!(buf.last().expect("should have last element").abs() < 0.1);
    }

    #[test]
    fn test_sample_rate_stored() {
        let env = EnvelopeGenerator::new(EnvelopeConfig::default(), 96000.0);
        assert!((env.sample_rate() - 96000.0).abs() < 1e-6);
    }

    #[test]
    fn test_release_from_idle_stays_idle() {
        let mut env = default_gen();
        env.release(); // should be no-op
        assert_eq!(env.phase(), EnvelopePhase::Idle);
    }

    #[test]
    fn test_level_never_exceeds_one() {
        let mut env = default_gen();
        env.trigger();
        for _ in 0..5000 {
            let v = env.next_sample();
            assert!(v <= 1.0 + 1e-6, "level exceeded 1.0: {v}");
            assert!(v >= -1e-6, "level went negative: {v}");
        }
    }

    #[test]
    fn test_double_trigger_restarts() {
        let mut env = default_gen();
        env.trigger();
        for _ in 0..100 {
            env.next_sample();
        }
        env.trigger(); // re-trigger
        assert_eq!(env.phase(), EnvelopePhase::Attack);
    }
}
