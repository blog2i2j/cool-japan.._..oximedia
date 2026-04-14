//! Dialogue clarity enhancement: presence boost, de-essing, and noise gate.
//!
//! This module implements professional dialogue enhancement processing commonly
//! used in broadcast and post-production workflows. A typical chain is:
//!
//! 1. **Noise gate** — silences the signal when RMS energy is below a threshold,
//!    removing background hum and room noise between lines.
//! 2. **Presence boost** — applies a peak EQ in the 2–5 kHz range to improve
//!    intelligibility and "cut through" a mix.
//! 3. **De-esser** — detects excessive high-frequency (>6 kHz) sibilant energy
//!    and applies transparent gain reduction to tame harsh "s" and "sh" sounds.

use thiserror::Error;

// ─── Error ────────────────────────────────────────────────────────────────────

/// Errors that can occur during dialogue enhancement.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum DialogueError {
    /// Sample rate is zero or otherwise unusable.
    #[error("Invalid sample rate: {0}")]
    InvalidSampleRate(u32),

    /// The input slice contained no samples.
    #[error("Input samples slice is empty")]
    EmptyInput,

    /// One or more configuration values are out of their valid ranges.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
}

// ─── Config ───────────────────────────────────────────────────────────────────

/// Configuration for `DialogueEnhancer`.
///
/// All dB values use the convention where 0 dBFS = full-scale digital amplitude
/// of 1.0 (linear).
#[derive(Debug, Clone)]
pub struct DialogueEnhancerConfig {
    /// Gain added by the presence peak EQ (dB).  Typical range: 0–12 dB.
    pub presence_boost_db: f32,
    /// Centre frequency of the presence boost (Hz).  Typical: 2 000–5 000 Hz.
    pub presence_freq_hz: f32,
    /// Q (bandwidth) of the presence peak EQ.  Higher Q = narrower band.
    pub presence_q: f32,
    /// High-frequency RMS threshold above which the de-esser kicks in (dBFS).
    pub de_ess_threshold_db: f32,
    /// Crossover frequency separating "sibilant" HF band from the rest (Hz).
    pub de_ess_freq_hz: f32,
    /// Gain reduction ratio applied by the de-esser when over threshold.
    /// E.g. 4.0 means 4:1 compression in the HF band.
    pub de_ess_ratio: f32,
    /// Noise gate open threshold (dBFS).  Signal below this is attenuated by 60 dB.
    pub noise_gate_threshold_db: f32,
}

impl DialogueEnhancerConfig {
    /// Creates a sensible default configuration for broadcast dialogue.
    #[must_use]
    pub fn default_broadcast() -> Self {
        Self {
            presence_boost_db: 3.0,
            presence_freq_hz: 3_000.0,
            presence_q: 1.0,
            de_ess_threshold_db: -20.0,
            de_ess_freq_hz: 6_000.0,
            de_ess_ratio: 4.0,
            noise_gate_threshold_db: -40.0,
        }
    }

    /// Validate all fields and return an error string when something is wrong.
    pub fn validate(&self) -> Result<(), DialogueError> {
        if self.presence_boost_db < 0.0 || self.presence_boost_db > 24.0 {
            return Err(DialogueError::InvalidConfig(format!(
                "presence_boost_db {} out of range [0, 24]",
                self.presence_boost_db
            )));
        }
        if self.presence_freq_hz <= 0.0 || self.presence_freq_hz > 20_000.0 {
            return Err(DialogueError::InvalidConfig(format!(
                "presence_freq_hz {} out of range (0, 20000]",
                self.presence_freq_hz
            )));
        }
        if self.presence_q <= 0.0 {
            return Err(DialogueError::InvalidConfig(format!(
                "presence_q {} must be > 0",
                self.presence_q
            )));
        }
        if self.de_ess_freq_hz <= 0.0 || self.de_ess_freq_hz > 20_000.0 {
            return Err(DialogueError::InvalidConfig(format!(
                "de_ess_freq_hz {} out of range (0, 20000]",
                self.de_ess_freq_hz
            )));
        }
        if self.de_ess_ratio < 1.0 {
            return Err(DialogueError::InvalidConfig(format!(
                "de_ess_ratio {} must be >= 1.0",
                self.de_ess_ratio
            )));
        }
        Ok(())
    }
}

impl Default for DialogueEnhancerConfig {
    fn default() -> Self {
        Self::default_broadcast()
    }
}

// ─── Peak-EQ biquad helpers ───────────────────────────────────────────────────

/// Second-order biquad filter state (Direct Form I).
#[derive(Debug, Clone)]
struct Biquad {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    // delay lines
    x1: f64,
    x2: f64,
    y1: f64,
    y2: f64,
}

impl Biquad {
    /// Construct a peak (bell) EQ biquad.
    ///
    /// Uses the Audio-EQ-Cookbook formulation.
    fn peak_eq(sample_rate: f64, freq_hz: f64, q: f64, gain_db: f64) -> Self {
        let a = 10.0_f64.powf(gain_db / 40.0);
        let w0 = 2.0 * std::f64::consts::PI * freq_hz / sample_rate;
        let alpha = w0.sin() / (2.0 * q);

        let b0 = 1.0 + alpha * a;
        let b1 = -2.0 * w0.cos();
        let b2 = 1.0 - alpha * a;
        let a0 = 1.0 + alpha / a;
        let a1 = -2.0 * w0.cos();
        let a2 = 1.0 - alpha / a;

        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    /// Construct a first-order high-pass shelving approximation used for the
    /// de-esser detector band.
    fn high_shelf(sample_rate: f64, freq_hz: f64) -> Self {
        // Simple one-pole approximation: H(z) = (1 - k) / (1 - k·z⁻¹)
        // where k = exp(-2π·f/fs).  We fold it into the biquad structure with
        // b1=b2=a2=0 so it degenerates to a first-order filter.
        let k = (-2.0 * std::f64::consts::PI * freq_hz / sample_rate).exp();
        // Difference equation: y[n] = (1-k)·x[n] + k·y[n-1]
        // => b0=(1-k), b1=0, b2=0, a1=-k, a2=0
        Self {
            b0: 1.0 - k,
            b1: 0.0,
            b2: 0.0,
            a1: -k,
            a2: 0.0,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    /// Process one sample.
    #[inline]
    fn process(&mut self, x: f64) -> f64 {
        let y = self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }
}

// ─── DialogueEnhancer ─────────────────────────────────────────────────────────

/// Processes audio blocks to enhance dialogue clarity.
///
/// Each call to [`process_block`] runs three serial processors:
///
/// 1. **Noise gate** — 60 dB of attenuation when the block RMS is below
///    [`DialogueEnhancerConfig::noise_gate_threshold_db`].
/// 2. **Presence peak EQ** — a biquad bell filter adds presence boost.
/// 3. **De-esser** — a smoothed high-frequency level detector drives gentle
///    gain reduction when sibilant energy exceeds the threshold.
///
/// [`process_block`]: DialogueEnhancer::process_block
#[derive(Debug)]
pub struct DialogueEnhancer {
    config: DialogueEnhancerConfig,
    /// Presence boost biquad filter state.
    presence_eq: Biquad,
    /// High-pass detector filter used by the de-esser.
    hf_detector: Biquad,
    /// Smoothed envelope of the HF detector output (linear RMS).
    hf_env: f64,
    /// Gain reduction currently applied by the de-esser (linear, 0..1).
    de_ess_gain: f64,
}

impl DialogueEnhancer {
    /// Create a new `DialogueEnhancer` from a config.
    ///
    /// # Errors
    ///
    /// Returns [`DialogueError::InvalidSampleRate`] or [`DialogueError::InvalidConfig`]
    /// when the parameters are out of their valid ranges.
    pub fn new(config: DialogueEnhancerConfig, sample_rate: u32) -> Result<Self, DialogueError> {
        if sample_rate == 0 {
            return Err(DialogueError::InvalidSampleRate(sample_rate));
        }
        config.validate()?;

        let fs = f64::from(sample_rate);
        let presence_eq = Biquad::peak_eq(
            fs,
            f64::from(config.presence_freq_hz),
            f64::from(config.presence_q),
            f64::from(config.presence_boost_db),
        );
        let hf_detector = Biquad::high_shelf(fs, f64::from(config.de_ess_freq_hz));

        Ok(Self {
            config,
            presence_eq,
            hf_detector,
            hf_env: 0.0,
            de_ess_gain: 1.0,
        })
    }

    /// Process a block of mono samples in-place.
    ///
    /// # Errors
    ///
    /// Returns [`DialogueError::EmptyInput`] if `samples` is empty, or
    /// [`DialogueError::InvalidSampleRate`] when `sample_rate` is zero.
    pub fn process_block(
        &mut self,
        samples: &mut [f32],
        sample_rate: u32,
    ) -> Result<(), DialogueError> {
        if sample_rate == 0 {
            return Err(DialogueError::InvalidSampleRate(sample_rate));
        }
        if samples.is_empty() {
            return Err(DialogueError::EmptyInput);
        }

        // ── 1. Noise gate (block-level RMS decision) ─────────────────────────
        let rms = block_rms(samples);
        let rms_db = linear_to_db(rms);
        let gate_open = rms_db >= self.config.noise_gate_threshold_db;
        let gate_gain: f32 = if gate_open { 1.0 } else { db_to_linear(-60.0) };

        // ── 2 & 3. Per-sample: presence EQ + de-esser ────────────────────────
        let de_ess_threshold_linear = db_to_linear_f64(f64::from(self.config.de_ess_threshold_db));
        let ratio = f64::from(self.config.de_ess_ratio);

        // Envelope follower time constants (~5 ms attack, ~50 ms release at fs).
        let fs = f64::from(sample_rate);
        let attack_coeff = (-1.0 / (0.005 * fs)).exp();
        let release_coeff = (-1.0 / (0.050 * fs)).exp();

        for s in samples.iter_mut() {
            // Gate
            *s *= gate_gain;

            // Presence EQ
            let eq_out = self.presence_eq.process(f64::from(*s));
            *s = eq_out as f32;

            // De-esser: detect HF energy
            let hf = self.hf_detector.process(f64::from(*s)).abs();
            // Envelope follower
            let coeff = if hf > self.hf_env {
                attack_coeff
            } else {
                release_coeff
            };
            self.hf_env = coeff * self.hf_env + (1.0 - coeff) * hf;

            // Compute target de-ess gain
            let target_gain = if self.hf_env > de_ess_threshold_linear {
                // Over-threshold gain reduction
                let over_db = linear_to_db_f64(self.hf_env) - f64::from(self.config.de_ess_threshold_db);
                let reduction_db = over_db * (1.0 - 1.0 / ratio);
                db_to_linear_f64(-reduction_db).clamp(0.0, 1.0)
            } else {
                1.0
            };

            // Smooth the gain control to avoid zipper noise
            let g_coeff = if target_gain < self.de_ess_gain {
                attack_coeff
            } else {
                release_coeff
            };
            self.de_ess_gain = g_coeff * self.de_ess_gain + (1.0 - g_coeff) * target_gain;
            *s = (*s as f64 * self.de_ess_gain) as f32;
        }

        Ok(())
    }

    /// Analyse a read-only slice of samples and return diagnostic metrics.
    ///
    /// # Errors
    ///
    /// Returns [`DialogueError::EmptyInput`] or [`DialogueError::InvalidSampleRate`].
    pub fn analyze(
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<DialogueAnalysis, DialogueError> {
        if sample_rate == 0 {
            return Err(DialogueError::InvalidSampleRate(sample_rate));
        }
        if samples.is_empty() {
            return Err(DialogueError::EmptyInput);
        }

        // Average level
        let rms = block_rms(samples);
        let avg_level_dbfs = linear_to_db(rms);

        // Speech fraction: fraction of 10-ms windows where RMS > -40 dBFS.
        const SPEECH_GATE_DB: f32 = -40.0;
        let window_size = (f32::from(sample_rate as u16) * 0.010) as usize;
        let window_size = window_size.max(1);
        let num_windows = (samples.len() + window_size - 1) / window_size;
        let mut speech_windows = 0usize;

        for chunk in samples.chunks(window_size) {
            let w_rms = block_rms(chunk);
            if linear_to_db(w_rms) > SPEECH_GATE_DB {
                speech_windows += 1;
            }
        }
        let detected_speech_fraction = speech_windows as f32 / num_windows as f32;

        // De-essing fraction: fraction of samples where HF level exceeds -20 dBFS.
        let fs = f64::from(sample_rate);
        let mut hf_filter = Biquad::high_shelf(fs, 6_000.0);
        let de_ess_thresh = db_to_linear_f64(-20.0);
        let mut de_ess_count = 0usize;

        // Run HF envelope detection over the whole signal.
        let mut hf_env: f64 = 0.0;
        let attack_coeff = (-1.0 / (0.005 * fs)).exp();
        let release_coeff = (-1.0 / (0.050 * fs)).exp();
        for &s in samples {
            let hf = hf_filter.process(f64::from(s)).abs();
            let coeff = if hf > hf_env { attack_coeff } else { release_coeff };
            hf_env = coeff * hf_env + (1.0 - coeff) * hf;
            if hf_env > de_ess_thresh {
                de_ess_count += 1;
            }
        }
        let de_essing_applied_fraction = de_ess_count as f32 / samples.len() as f32;

        Ok(DialogueAnalysis {
            detected_speech_fraction,
            avg_level_dbfs,
            de_essing_applied_fraction,
        })
    }
}

// ─── DialogueAnalysis ─────────────────────────────────────────────────────────

/// Diagnostic metrics produced by [`DialogueEnhancer::analyze`].
#[derive(Debug, Clone)]
pub struct DialogueAnalysis {
    /// Fraction (0–1) of short windows detected as speech (RMS > −40 dBFS).
    pub detected_speech_fraction: f32,
    /// Average RMS level of the full block in dBFS.
    pub avg_level_dbfs: f32,
    /// Fraction (0–1) of samples where the de-esser would have been active.
    pub de_essing_applied_fraction: f32,
}

// ─── DSP utilities ────────────────────────────────────────────────────────────

#[inline]
fn block_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|&s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

#[inline]
fn linear_to_db(linear: f32) -> f32 {
    if linear <= 0.0 {
        return -144.0; // silence
    }
    20.0 * linear.log10()
}

#[inline]
fn linear_to_db_f64(linear: f64) -> f64 {
    if linear <= 0.0 {
        return -144.0;
    }
    20.0 * linear.log10()
}

#[inline]
fn db_to_linear(db: f32) -> f32 {
    10.0_f32.powf(db / 20.0)
}

#[inline]
fn db_to_linear_f64(db: f64) -> f64 {
    10.0_f64.powf(db / 20.0)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const FS: u32 = 48_000;

    fn make_silence(n: usize) -> Vec<f32> {
        vec![0.0f32; n]
    }

    fn make_sine(freq: f32, amplitude: f32, n: usize, fs: u32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                amplitude
                    * (2.0 * std::f32::consts::PI * freq * i as f32 / fs as f32).sin()
            })
            .collect()
    }

    // ── Config validation ───────────────────────────────────────────────────

    #[test]
    fn invalid_presence_boost_rejected() {
        let mut cfg = DialogueEnhancerConfig::default();
        cfg.presence_boost_db = -5.0;
        assert!(matches!(cfg.validate(), Err(DialogueError::InvalidConfig(_))));
    }

    #[test]
    fn invalid_presence_freq_rejected() {
        let mut cfg = DialogueEnhancerConfig::default();
        cfg.presence_freq_hz = 0.0;
        assert!(matches!(cfg.validate(), Err(DialogueError::InvalidConfig(_))));
    }

    #[test]
    fn invalid_q_rejected() {
        let mut cfg = DialogueEnhancerConfig::default();
        cfg.presence_q = 0.0;
        assert!(matches!(cfg.validate(), Err(DialogueError::InvalidConfig(_))));
    }

    #[test]
    fn invalid_de_ess_ratio_rejected() {
        let mut cfg = DialogueEnhancerConfig::default();
        cfg.de_ess_ratio = 0.5;
        assert!(matches!(cfg.validate(), Err(DialogueError::InvalidConfig(_))));
    }

    // ── Invalid sample rate ─────────────────────────────────────────────────

    #[test]
    fn zero_sample_rate_returns_error() {
        let cfg = DialogueEnhancerConfig::default();
        assert!(matches!(
            DialogueEnhancer::new(cfg, 0),
            Err(DialogueError::InvalidSampleRate(0))
        ));
    }

    // ── Noise gate: silence stays silent ───────────────────────────────────

    #[test]
    fn silence_passes_through_noise_gate_as_silence() {
        let cfg = DialogueEnhancerConfig {
            noise_gate_threshold_db: -40.0,
            presence_boost_db: 0.0,
            de_ess_threshold_db: 0.0, // gate only, no de-essing at 0 dBFS threshold
            ..DialogueEnhancerConfig::default()
        };
        let mut enhancer = DialogueEnhancer::new(cfg, FS).expect("valid config");
        let mut samples = make_silence(512);
        enhancer.process_block(&mut samples, FS).expect("process ok");
        // All samples should be tiny (noise gate attenuated 60 dB)
        for &s in &samples {
            assert!(s.abs() < 1e-3, "expected near-zero, got {s}");
        }
    }

    // ── Noise gate: loud signal passes ─────────────────────────────────────

    #[test]
    fn loud_signal_opens_noise_gate() {
        let cfg = DialogueEnhancerConfig {
            noise_gate_threshold_db: -40.0,
            presence_boost_db: 0.0,
            de_ess_threshold_db: 0.0, // gate only
            ..DialogueEnhancerConfig::default()
        };
        let mut enhancer = DialogueEnhancer::new(cfg, FS).expect("valid config");
        // 1 kHz sine at -6 dBFS → well above -40 dBFS gate
        let mut samples = make_sine(1_000.0, 0.5, 512, FS);
        let original: Vec<f32> = samples.clone();
        enhancer.process_block(&mut samples, FS).expect("process ok");
        // Output should be significantly larger than gate-attenuated version
        let out_rms = block_rms(&samples);
        let in_rms = block_rms(&original);
        assert!(
            out_rms > in_rms * 0.1,
            "gate should pass signal: in_rms={in_rms}, out_rms={out_rms}"
        );
    }

    // ── Empty input error ───────────────────────────────────────────────────

    #[test]
    fn empty_input_returns_error() {
        let mut enhancer =
            DialogueEnhancer::new(DialogueEnhancerConfig::default(), FS).expect("valid");
        assert!(matches!(
            enhancer.process_block(&mut [], FS),
            Err(DialogueError::EmptyInput)
        ));
    }

    // ── Analyze: silence gives near-zero speech fraction ───────────────────

    #[test]
    fn analyze_silence_low_speech_fraction() {
        let silence = make_silence(FS as usize); // 1 second
        let analysis = DialogueEnhancer::analyze(&silence, FS).expect("ok");
        assert!(
            analysis.detected_speech_fraction < 0.05,
            "silence should have low speech fraction, got {}",
            analysis.detected_speech_fraction
        );
    }

    // ── Analyze: loud tone gives high speech fraction ──────────────────────

    #[test]
    fn analyze_loud_signal_high_speech_fraction() {
        let samples = make_sine(1_000.0, 0.5, FS as usize, FS);
        let analysis = DialogueEnhancer::analyze(&samples, FS).expect("ok");
        assert!(
            analysis.detected_speech_fraction > 0.9,
            "loud signal should have high speech fraction, got {}",
            analysis.detected_speech_fraction
        );
    }

    // ── Analyze: avg level roughly matches known amplitude ─────────────────

    #[test]
    fn analyze_avg_level_roughly_correct() {
        // -3 dBFS sine → RMS ≈ -6 dBFS (because sine RMS = amplitude/√2)
        let samples = make_sine(440.0, 0.707_f32, FS as usize, FS);
        let analysis = DialogueEnhancer::analyze(&samples, FS).expect("ok");
        // Sine RMS = 0.707/√2 ≈ 0.5 → -6 dBFS.  Allow ±3 dB tolerance.
        assert!(
            analysis.avg_level_dbfs > -10.0 && analysis.avg_level_dbfs < -2.0,
            "expected ~-6 dBFS, got {}",
            analysis.avg_level_dbfs
        );
    }

    // ── Analyze: analyze rejects empty input ───────────────────────────────

    #[test]
    fn analyze_empty_input_error() {
        assert!(matches!(
            DialogueEnhancer::analyze(&[], FS),
            Err(DialogueError::EmptyInput)
        ));
    }

    // ── De-essing fraction with very hot HF signal ─────────────────────────

    #[test]
    fn analyze_hot_hf_signal_high_de_ess_fraction() {
        // 10 kHz tone at 0 dBFS → lots of HF energy → de-esser fraction > 0
        let samples = make_sine(10_000.0, 1.0, FS as usize, FS);
        let analysis = DialogueEnhancer::analyze(&samples, FS).expect("ok");
        assert!(
            analysis.de_essing_applied_fraction > 0.0,
            "expected de-essing to be active for hot HF signal"
        );
    }
}
