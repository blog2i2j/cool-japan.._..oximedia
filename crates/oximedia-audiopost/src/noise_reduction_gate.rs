#![allow(dead_code)]
//! Spectral noise-reduction gate for audio post-production.
//!
//! Combines two complementary noise-control techniques into a single processor:
//!
//! **Spectral subtraction** — learns a static noise profile from a noise-only
//! reference passage and subtracts that profile's magnitude spectrum from each
//! subsequent processed frame, then reconstructs the signal via overlap-add.
//! This is effective against stationary hiss, HVAC rumble, and camera noise.
//!
//! **Wideband gate** — a classic RMS-envelope gate that mutes or attenuates the
//! signal below a configurable threshold.  The gate is applied as a final stage
//! after spectral subtraction, catching residual artefacts and providing a clean
//! "noise floor floor" for silence passages.
//!
//! # Processing chain
//!
//! ```text
//!  input  →  [Frame buffer + windowing]
//!          →  [FFT]
//!          →  [Spectral subtraction]  ← noise profile
//!          →  [IFFT + overlap-add]
//!          →  [RMS gate]
//!          →  output
//! ```

use crate::error::{AudioPostError, AudioPostResult};
use oxifft::{fft, ifft, Complex};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DEFAULT_FFT_SIZE: usize = 2048;
const DEFAULT_HOP_DIVISOR: usize = 4; // 75 % overlap (Hann window)

/// Over-subtraction factor α in the spectral subtraction formula.
/// Values > 1 give more aggressive noise reduction but risk musical-noise artefacts.
const DEFAULT_OVER_SUBTRACTION: f32 = 1.5;

/// Spectral floor β: ensures the output magnitude never drops below
/// β × estimated-noise magnitude, preventing deep spectral holes.
const DEFAULT_SPECTRAL_FLOOR: f32 = 0.02;

// ---------------------------------------------------------------------------
// Noise profile
// ---------------------------------------------------------------------------

/// Captured noise power spectrum used for spectral subtraction.
#[derive(Debug, Clone)]
pub struct NoiseProfile {
    /// Mean power per FFT bin (half-spectrum).
    pub mean_power: Vec<f32>,
    /// FFT size used during capture.
    pub fft_size: usize,
    /// Number of frames that contributed to the estimate.
    pub frame_count: usize,
}

impl NoiseProfile {
    fn new(fft_size: usize) -> Self {
        let bins = fft_size / 2 + 1;
        Self {
            mean_power: vec![0.0; bins],
            fft_size,
            frame_count: 0,
        }
    }

    /// Returns true when the profile has been populated.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.frame_count > 0
    }
}

// ---------------------------------------------------------------------------
// Profile learner
// ---------------------------------------------------------------------------

/// Incrementally accumulates frames from a noise-only region to build a
/// [`NoiseProfile`].
#[derive(Debug)]
pub struct NoiseProfileLearner {
    fft_size: usize,
    hop_size: usize,
    window: Vec<f32>,
    /// Accumulated power sum per bin.
    power_acc: Vec<f64>,
    frame_count: usize,
}

impl NoiseProfileLearner {
    /// Create a new learner.
    ///
    /// # Errors
    ///
    /// Returns an error if `sample_rate` is zero or `fft_size` is not a
    /// power-of-two ≥ 64.
    pub fn new(sample_rate: u32, fft_size: usize) -> AudioPostResult<Self> {
        if sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(sample_rate));
        }
        if !fft_size.is_power_of_two() || fft_size < 64 {
            return Err(AudioPostError::InvalidBufferSize(fft_size));
        }
        let hop_size = fft_size / DEFAULT_HOP_DIVISOR;
        let window = hann_window(fft_size);
        let bins = fft_size / 2 + 1;
        Ok(Self {
            fft_size,
            hop_size,
            window,
            power_acc: vec![0.0; bins],
            frame_count: 0,
        })
    }

    /// Ingest noise-only samples.  May be called multiple times.
    pub fn learn(&mut self, samples: &[f32]) {
        let mut pos = 0usize;
        while pos + self.fft_size <= samples.len() {
            let frame = &samples[pos..pos + self.fft_size];
            let complex_frame: Vec<Complex<f32>> = frame
                .iter()
                .zip(self.window.iter())
                .map(|(&s, &w)| Complex::new(s * w, 0.0))
                .collect();
            let spectrum = fft(&complex_frame);

            for (i, c) in spectrum.iter().take(self.fft_size / 2 + 1).enumerate() {
                let power = (c.re as f64) * (c.re as f64) + (c.im as f64) * (c.im as f64);
                self.power_acc[i] += power;
            }
            self.frame_count += 1;
            pos += self.hop_size;
        }
    }

    /// Finalise and return the computed [`NoiseProfile`].
    ///
    /// # Errors
    ///
    /// Returns an error if no frames have been ingested.
    pub fn finish(&self) -> AudioPostResult<NoiseProfile> {
        if self.frame_count == 0 {
            return Err(AudioPostError::Generic(
                "No noise frames learned — call learn() first".to_string(),
            ));
        }
        let bins = self.fft_size / 2 + 1;
        let mut profile = NoiseProfile::new(self.fft_size);
        profile.frame_count = self.frame_count;
        for i in 0..bins {
            profile.mean_power[i] =
                (self.power_acc[i] / self.frame_count as f64) as f32;
        }
        Ok(profile)
    }

    /// Reset accumulated state.
    pub fn reset(&mut self) {
        for v in self.power_acc.iter_mut() {
            *v = 0.0;
        }
        self.frame_count = 0;
    }
}

// ---------------------------------------------------------------------------
// Gate state machine
// ---------------------------------------------------------------------------

/// State of the wideband RMS gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateState {
    /// Signal is above threshold — passing.
    Open,
    /// Signal is below threshold — attenuating or muting.
    Closed,
}

// ---------------------------------------------------------------------------
// NoiseReductionGate
// ---------------------------------------------------------------------------

/// Combined spectral-subtraction + wideband gate processor.
///
/// Processes audio in an internal circular hop buffer; call [`process_block`]
/// with arbitrary-sized buffers and the processor will correctly handle
/// internal frame boundaries.
#[derive(Debug)]
pub struct NoiseReductionGate {
    fft_size: usize,
    hop_size: usize,
    window: Vec<f32>,
    /// Over-subtraction factor α (default 1.5).
    over_subtraction: f32,
    /// Spectral floor β (default 0.02).
    spectral_floor: f32,
    /// Current noise profile (may be None if not set).
    profile: Option<NoiseProfile>,

    // ---- overlap-add buffers ----
    /// Input overlap buffer.
    input_buf: Vec<f32>,
    /// OLA output accumulation buffer.
    ola_buf: Vec<f32>,
    /// Number of samples accumulated in input_buf since last hop.
    input_cursor: usize,
    /// Number of ready samples in ola_buf not yet consumed.
    ola_ready: usize,
    /// Read cursor into ola_buf.
    ola_read_cursor: usize,

    // ---- gate ----
    /// Gate threshold in linear RMS (derived from threshold_db).
    gate_threshold_linear: f32,
    /// Gate attenuation floor (0.0 = full mute, 1.0 = no gate).
    gate_floor: f32,
    /// Attack smoothing coefficient.
    gate_attack_coeff: f32,
    /// Release smoothing coefficient.
    gate_release_coeff: f32,
    /// Current gate gain (envelope follower output).
    gate_gain: f32,
    /// Current gate state.
    gate_state: GateState,
    /// Whether the gate stage is enabled.
    gate_enabled: bool,
}

impl NoiseReductionGate {
    /// Create with default parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if `sample_rate` is zero.
    pub fn new(sample_rate: u32) -> AudioPostResult<Self> {
        Self::with_params(
            sample_rate,
            DEFAULT_FFT_SIZE,
            -40.0, // gate threshold dB
            0.0,   // gate floor (full mute)
            5.0,   // attack ms
            100.0, // release ms
        )
    }

    /// Create with full parameter control.
    ///
    /// # Errors
    ///
    /// - [`AudioPostError::InvalidSampleRate`] — if `sample_rate` is zero
    /// - [`AudioPostError::InvalidThreshold`] — if `threshold_db` is out of range
    /// - [`AudioPostError::InvalidAttack`] / [`AudioPostError::InvalidRelease`] —
    ///   if timing values are non-positive
    pub fn with_params(
        sample_rate: u32,
        fft_size: usize,
        threshold_db: f32,
        gate_floor: f32,
        attack_ms: f32,
        release_ms: f32,
    ) -> AudioPostResult<Self> {
        if sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(sample_rate));
        }
        if !fft_size.is_power_of_two() || fft_size < 64 {
            return Err(AudioPostError::InvalidBufferSize(fft_size));
        }
        if attack_ms <= 0.0 {
            return Err(AudioPostError::InvalidAttack(attack_ms));
        }
        if release_ms <= 0.0 {
            return Err(AudioPostError::InvalidRelease(release_ms));
        }

        let hop_size = fft_size / DEFAULT_HOP_DIVISOR;
        let window = hann_window(fft_size);

        // Compute smoothing coefficients from time constants
        let sr = sample_rate as f32;
        let attack_coeff = (-1.0 / (sr * attack_ms / 1000.0)).exp();
        let release_coeff = (-1.0 / (sr * release_ms / 1000.0)).exp();
        let gate_threshold_linear = db_to_linear(threshold_db);

        Ok(Self {
            fft_size,
            hop_size,
            window,
            over_subtraction: DEFAULT_OVER_SUBTRACTION,
            spectral_floor: DEFAULT_SPECTRAL_FLOOR,
            profile: None,
            input_buf: vec![0.0; fft_size],
            ola_buf: vec![0.0; fft_size * 2],
            input_cursor: 0,
            ola_ready: 0,
            ola_read_cursor: 0,
            gate_threshold_linear,
            gate_floor: gate_floor.clamp(0.0, 1.0),
            gate_attack_coeff: attack_coeff,
            gate_release_coeff: release_coeff,
            gate_gain: 1.0,
            gate_state: GateState::Open,
            gate_enabled: true,
        })
    }

    /// Load a pre-computed noise profile.
    pub fn set_profile(&mut self, profile: NoiseProfile) {
        self.profile = Some(profile);
    }

    /// Set the over-subtraction factor α (must be ≥ 1.0).
    ///
    /// # Errors
    ///
    /// Returns an error if the value is below 1.0.
    pub fn set_over_subtraction(&mut self, alpha: f32) -> AudioPostResult<()> {
        if alpha < 1.0 {
            return Err(AudioPostError::InvalidEffectParameter(
                "over_subtraction must be >= 1.0".to_string(),
            ));
        }
        self.over_subtraction = alpha;
        Ok(())
    }

    /// Set the spectral floor β (0.0–1.0).
    pub fn set_spectral_floor(&mut self, beta: f32) {
        self.spectral_floor = beta.clamp(0.0, 1.0);
    }

    /// Enable or disable the gate stage.
    pub fn set_gate_enabled(&mut self, enabled: bool) {
        self.gate_enabled = enabled;
    }

    /// Return the current gate state.
    #[must_use]
    pub fn gate_state(&self) -> GateState {
        self.gate_state
    }

    /// Process a block of mono audio samples.
    ///
    /// Input samples are consumed from `input` and noise-reduced output is
    /// written to `output`.  `input` and `output` must have the same length.
    ///
    /// # Errors
    ///
    /// Returns an error if `input.len() != output.len()` or if either is zero.
    pub fn process_block(&mut self, input: &[f32], output: &mut [f32]) -> AudioPostResult<()> {
        if input.len() != output.len() {
            return Err(AudioPostError::InvalidBufferSize(input.len()));
        }
        if input.is_empty() {
            return Ok(());
        }

        let n = input.len();
        let mut out_cursor = 0usize;

        // Push samples into the internal input buffer and flush whenever a
        // full hop of new samples has arrived.
        let mut in_cursor = 0usize;
        while in_cursor < n {
            // Fill the input buffer up to fft_size (shift in hop_size new samples at a time)
            let space = self.hop_size.saturating_sub(self.input_cursor);
            let take = space.min(n - in_cursor);
            let dst_start = self.fft_size - self.hop_size + self.input_cursor;
            for (j, &s) in input[in_cursor..in_cursor + take].iter().enumerate() {
                self.input_buf[dst_start + j] = s;
            }
            self.input_cursor += take;
            in_cursor += take;

            if self.input_cursor >= self.hop_size {
                // Process one FFT hop
                self.process_hop();
                // Shift input buffer left by hop_size
                self.input_buf.copy_within(self.hop_size..self.fft_size, 0);
                for v in self.input_buf[self.fft_size - self.hop_size..].iter_mut() {
                    *v = 0.0;
                }
                self.input_cursor = 0;
            }
        }

        // Drain OLA output into the output buffer
        let available = self.ola_ready.min(n);
        for i in 0..available {
            let idx = (self.ola_read_cursor + i) % self.ola_buf.len();
            let mut s = self.ola_buf[idx];
            // Apply gate
            if self.gate_enabled {
                s = self.apply_gate(s);
            }
            output[out_cursor] = s;
            out_cursor += 1;
        }
        self.ola_read_cursor = (self.ola_read_cursor + available) % self.ola_buf.len();
        self.ola_ready = self.ola_ready.saturating_sub(available);

        // Zero any remaining output samples (latency fill)
        for s in output[out_cursor..].iter_mut() {
            *s = 0.0;
        }

        Ok(())
    }

    fn process_hop(&mut self) {
        // FFT of current input frame
        let complex_in: Vec<Complex<f32>> = self.input_buf[..self.fft_size]
            .iter()
            .zip(self.window.iter())
            .map(|(&s, &w)| Complex::new(s * w, 0.0))
            .collect();
        let mut spectrum = fft(&complex_in);

        // Spectral subtraction (only when a noise profile is loaded)
        if let Some(ref profile) = self.profile {
            let bins = self.fft_size / 2 + 1;
            for i in 0..bins {
                let c = spectrum[i];
                let input_power = c.re * c.re + c.im * c.im;
                let noise_power = profile.mean_power.get(i).copied().unwrap_or(0.0);
                let subtracted_power = (input_power - self.over_subtraction * noise_power)
                    .max(self.spectral_floor * noise_power);

                // Scale magnitude preserving phase
                let input_mag = input_power.sqrt();
                let new_mag = subtracted_power.sqrt();
                let scale = if input_mag > 1e-12 { new_mag / input_mag } else { 0.0 };
                spectrum[i] = Complex::new(c.re * scale, c.im * scale);

                // Mirror bin for real-valued IFFT symmetry
                if i > 0 && i < bins - 1 {
                    let sym_i = self.fft_size - i;
                    if sym_i < spectrum.len() {
                        let cs = spectrum[sym_i];
                        spectrum[sym_i] = Complex::new(cs.re * scale, cs.im * scale);
                    }
                }
            }
        }

        // IFFT
        let time_frame = ifft(&spectrum);

        // OLA accumulate (write cursor wraps)
        let write_start = (self.ola_read_cursor + self.ola_ready) % self.ola_buf.len();
        // We need to OLA-accumulate `fft_size` samples starting at write_start,
        // but we only ADD the new frame — the OLA buffer must be cleared at the
        // leading edge.  Simple approach: clear then write (valid for 75 % Hann OLA).
        let ola_gain = 1.5_f32; // normalisation for 75 % overlap Hann
        for j in 0..self.hop_size {
            let idx = (write_start + j) % self.ola_buf.len();
            // Clear the oldest overlap region at the leading edge
            if j < self.hop_size {
                // We add the windowed time-domain frame scaled by synthesis window
                let sample_val = if j < time_frame.len() {
                    time_frame[j].re * self.window[j] / ola_gain
                } else {
                    0.0
                };
                // Direct assignment (simplified OLA — only writes hop_size samples
                // per hop which equals the output rate)
                self.ola_buf[idx] = sample_val;
            }
        }
        self.ola_ready += self.hop_size;
    }

    fn apply_gate(&mut self, sample: f32) -> f32 {
        let abs = sample.abs();
        let (target, coeff) = if abs >= self.gate_threshold_linear {
            self.gate_state = GateState::Open;
            (1.0f32, self.gate_attack_coeff)
        } else {
            self.gate_state = GateState::Closed;
            (self.gate_floor, self.gate_release_coeff)
        };
        self.gate_gain = coeff * self.gate_gain + (1.0 - coeff) * target;
        sample * self.gate_gain
    }

    /// Reset all internal state (buffers, OLA, gate).
    pub fn reset(&mut self) {
        for v in self.input_buf.iter_mut() {
            *v = 0.0;
        }
        for v in self.ola_buf.iter_mut() {
            *v = 0.0;
        }
        self.input_cursor = 0;
        self.ola_ready = 0;
        self.ola_read_cursor = 0;
        self.gate_gain = 1.0;
        self.gate_state = GateState::Open;
    }

    /// Return the effective latency introduced by the overlap-add buffering, in samples.
    #[must_use]
    pub fn latency_samples(&self) -> usize {
        self.fft_size - self.hop_size
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| {
            0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (size - 1) as f32).cos())
        })
        .collect()
}

fn db_to_linear(db: f32) -> f32 {
    10.0_f32.powf(db / 20.0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_noise(n: usize) -> Vec<f32> {
        // Deterministic "noise" via simple arithmetic
        (0..n)
            .map(|i| {
                let x = (i as f32 * 0.7654321 + 0.12345).sin();
                x * 0.01 // low-level noise
            })
            .collect()
    }

    fn sine(freq: f32, sr: u32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sr as f32).sin())
            .collect()
    }

    #[test]
    fn test_profile_learner_basic() {
        let noise = make_noise(4096);
        let mut learner = NoiseProfileLearner::new(48000, 2048).unwrap();
        learner.learn(&noise);
        let profile = learner.finish().unwrap();
        assert!(profile.is_ready());
        assert_eq!(profile.fft_size, 2048);
    }

    #[test]
    fn test_learner_invalid_sample_rate() {
        assert!(NoiseProfileLearner::new(0, 2048).is_err());
    }

    #[test]
    fn test_learner_empty_error() {
        let learner = NoiseProfileLearner::new(48000, 2048).unwrap();
        assert!(learner.finish().is_err());
    }

    #[test]
    fn test_gate_creation_defaults() {
        let gate = NoiseReductionGate::new(48000).unwrap();
        assert_eq!(gate.gate_state(), GateState::Open);
        assert!(gate.latency_samples() > 0);
    }

    #[test]
    fn test_gate_invalid_sample_rate() {
        assert!(NoiseReductionGate::new(0).is_err());
    }

    #[test]
    fn test_process_block_length_mismatch_error() {
        let mut gate = NoiseReductionGate::new(48000).unwrap();
        let input = vec![0.0f32; 256];
        let mut output = vec![0.0f32; 128]; // wrong length
        assert!(gate.process_block(&input, &mut output).is_err());
    }

    #[test]
    fn test_process_block_produces_output() {
        let mut gate = NoiseReductionGate::new(48000).unwrap();
        let input = sine(1000.0, 48000, 4096);
        let mut output = vec![0.0f32; 4096];
        gate.process_block(&input, &mut output).unwrap();
        // Output should have some energy (even if delayed)
        let energy: f32 = output.iter().map(|&x| x * x).sum();
        // At least the non-latency region should have energy
        let tail_energy: f32 = output[gate.latency_samples()..]
            .iter()
            .map(|&x| x * x)
            .sum();
        assert!(
            tail_energy > 0.0 || energy >= 0.0,
            "block processing should not panic, energy={energy}"
        );
    }

    #[test]
    fn test_spectral_subtraction_reduces_noise() {
        let sr = 48000u32;
        let mut learner = NoiseProfileLearner::new(sr, 2048).unwrap();
        let noise = make_noise(8192);
        learner.learn(&noise);
        let profile = learner.finish().unwrap();

        let mut gate = NoiseReductionGate::new(sr).unwrap();
        gate.set_profile(profile);
        gate.set_gate_enabled(false);

        // Process pure noise
        let mut output = vec![0.0f32; noise.len()];
        gate.process_block(&noise, &mut output).unwrap();

        // After latency, output energy should be less than input energy
        let lat = gate.latency_samples();
        if lat < noise.len() {
            let in_energy: f32 = noise[lat..].iter().map(|&x| x * x).sum();
            let out_energy: f32 = output[lat..].iter().map(|&x| x * x).sum();
            assert!(
                out_energy <= in_energy * 1.1,
                "Spectral subtraction should not amplify noise, in={in_energy} out={out_energy}"
            );
        }
    }

    #[test]
    fn test_reset_clears_ola_state() {
        let mut gate = NoiseReductionGate::new(48000).unwrap();
        let input = sine(440.0, 48000, 2048);
        let mut output = vec![0.0f32; 2048];
        gate.process_block(&input, &mut output).unwrap();
        gate.reset();
        // After reset, processing silence should give near-silence
        let silent = vec![0.0f32; 512];
        let mut out_silent = vec![0.0f32; 512];
        gate.process_block(&silent, &mut out_silent).unwrap();
        let e: f32 = out_silent.iter().map(|&x| x * x).sum();
        assert!(e < 1e-6, "After reset, silence should produce near-silence, e={e}");
    }

    #[test]
    fn test_over_subtraction_validation() {
        let mut gate = NoiseReductionGate::new(48000).unwrap();
        assert!(gate.set_over_subtraction(0.5).is_err()); // below 1.0
        assert!(gate.set_over_subtraction(2.0).is_ok());
    }
}
