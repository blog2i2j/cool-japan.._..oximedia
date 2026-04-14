#![allow(dead_code)]
//! Loudness gate processing for normalization.
//!
//! Implements configurable loudness gating with threshold, hold time,
//! and hysteresis. Used to exclude silent or very quiet passages from
//! loudness measurements and gain calculations.

/// Gate state for the loudness gate processor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateState {
    /// Gate is open (signal above threshold).
    Open,
    /// Gate is in hold phase (signal dropped but hold timer active).
    Hold,
    /// Gate is closed (signal below threshold).
    Closed,
}

/// Configuration for the loudness gate.
#[derive(Debug, Clone)]
pub struct LoudnessGateConfig {
    /// Absolute gate threshold in LUFS (ITU-R BS.1770 uses -70 LUFS).
    pub absolute_threshold_lufs: f64,
    /// Relative gate threshold offset in LU (ITU-R BS.1770 uses -10 LU).
    pub relative_threshold_lu: f64,
    /// Hold time in milliseconds before closing gate after signal drops.
    pub hold_time_ms: f64,
    /// Hysteresis in dB to prevent rapid gate cycling.
    pub hysteresis_db: f64,
    /// Attack time in milliseconds for gate opening.
    pub attack_ms: f64,
    /// Release time in milliseconds for gate closing.
    pub release_ms: f64,
    /// Sample rate in Hz.
    pub sample_rate: f64,
    /// Number of audio channels.
    pub channels: usize,
}

impl LoudnessGateConfig {
    /// Create a new gate configuration with ITU-R BS.1770-4 defaults.
    pub fn new(sample_rate: f64, channels: usize) -> Self {
        Self {
            absolute_threshold_lufs: -70.0,
            relative_threshold_lu: -10.0,
            hold_time_ms: 200.0,
            hysteresis_db: 2.0,
            attack_ms: 0.1,
            release_ms: 50.0,
            sample_rate,
            channels,
        }
    }

    /// Create a strict broadcast gate configuration.
    pub fn broadcast(sample_rate: f64, channels: usize) -> Self {
        Self {
            absolute_threshold_lufs: -70.0,
            relative_threshold_lu: -10.0,
            hold_time_ms: 400.0,
            hysteresis_db: 3.0,
            attack_ms: 0.05,
            release_ms: 100.0,
            sample_rate,
            channels,
        }
    }

    /// Validate the configuration parameters.
    pub fn validate(&self) -> Result<(), String> {
        if self.sample_rate < 8000.0 || self.sample_rate > 192_000.0 {
            return Err(format!("Invalid sample rate: {}", self.sample_rate));
        }
        if self.channels == 0 || self.channels > 16 {
            return Err(format!("Invalid channel count: {}", self.channels));
        }
        if self.hold_time_ms < 0.0 {
            return Err("Hold time cannot be negative".to_string());
        }
        if self.hysteresis_db < 0.0 {
            return Err("Hysteresis cannot be negative".to_string());
        }
        if self.attack_ms < 0.0 {
            return Err("Attack time cannot be negative".to_string());
        }
        if self.release_ms < 0.0 {
            return Err("Release time cannot be negative".to_string());
        }
        Ok(())
    }
}

/// Loudness-gated measurement block result.
#[derive(Debug, Clone)]
pub struct GatedBlock {
    /// Block loudness in LUFS.
    pub loudness_lufs: f64,
    /// Whether this block passed the absolute gate.
    pub passed_absolute: bool,
    /// Whether this block passed the relative gate.
    pub passed_relative: bool,
    /// Block start sample index.
    pub start_sample: usize,
    /// Block duration in samples.
    pub duration_samples: usize,
}

/// Loudness gate processor implementing ITU-R BS.1770 gating.
///
/// Processes audio in 400ms blocks with 75% overlap as specified
/// by the standard. Applies absolute and relative gating stages.
#[derive(Debug)]
pub struct LoudnessGate {
    /// Gate configuration.
    config: LoudnessGateConfig,
    /// Current gate state.
    state: GateState,
    /// Hold timer remaining in samples.
    hold_remaining: usize,
    /// Envelope follower value for smooth gating.
    envelope: f64,
    /// Attack coefficient for envelope.
    attack_coeff: f64,
    /// Release coefficient for envelope.
    release_coeff: f64,
    /// Accumulated gated blocks.
    gated_blocks: Vec<GatedBlock>,
    /// Running sum for integrated loudness of gated blocks.
    gated_sum: f64,
    /// Count of blocks that passed gating.
    gated_count: usize,
    /// Total blocks processed.
    total_blocks: usize,
    /// Current block buffer.
    block_buffer: Vec<f64>,
    /// Block size in samples (400ms worth).
    block_size: usize,
    /// Hop size in samples (100ms for 75% overlap).
    hop_size: usize,
    /// Samples accumulated in current block.
    samples_in_block: usize,
    /// Total samples processed.
    total_samples_processed: usize,
}

impl LoudnessGate {
    /// Create a new loudness gate processor.
    pub fn new(config: LoudnessGateConfig) -> Self {
        let block_size = (config.sample_rate * 0.4) as usize * config.channels;
        let hop_size = (config.sample_rate * 0.1) as usize * config.channels;
        let hold_samples = (config.hold_time_ms * config.sample_rate / 1000.0) as usize;
        let attack_coeff = if config.attack_ms > 0.0 {
            1.0 - (-2.2 / (config.attack_ms * config.sample_rate / 1000.0)).exp()
        } else {
            1.0
        };
        let release_coeff = if config.release_ms > 0.0 {
            1.0 - (-2.2 / (config.release_ms * config.sample_rate / 1000.0)).exp()
        } else {
            1.0
        };

        Self {
            config,
            state: GateState::Closed,
            hold_remaining: hold_samples,
            envelope: 0.0,
            attack_coeff,
            release_coeff,
            gated_blocks: Vec::new(),
            gated_sum: 0.0,
            gated_count: 0,
            total_blocks: 0,
            block_buffer: Vec::with_capacity(block_size),
            block_size,
            hop_size,
            samples_in_block: 0,
            total_samples_processed: 0,
        }
    }

    /// Get the current gate state.
    pub fn state(&self) -> GateState {
        self.state
    }

    /// Get total blocks processed.
    pub fn total_blocks(&self) -> usize {
        self.total_blocks
    }

    /// Get count of blocks that passed gating.
    pub fn gated_count(&self) -> usize {
        self.gated_count
    }

    /// Process a buffer of audio samples through the gate.
    pub fn process(&mut self, samples: &[f32]) {
        for &sample in samples {
            self.block_buffer.push(f64::from(sample));
            self.samples_in_block += 1;
            self.total_samples_processed += 1;

            if self.samples_in_block >= self.block_size / self.config.channels {
                self.process_block();
                // Overlap: keep last 75% of samples
                let keep = self.block_buffer.len().saturating_sub(self.hop_size);
                if keep > 0 && keep < self.block_buffer.len() {
                    let drained: Vec<f64> = self.block_buffer.drain(..self.hop_size).collect();
                    drop(drained);
                }
                self.samples_in_block = self.block_buffer.len();
            }
        }
    }

    /// Process a complete block and determine gating.
    fn process_block(&mut self) {
        let block_loudness = self.compute_block_loudness();
        let passed_absolute = block_loudness > self.config.absolute_threshold_lufs;

        let block = GatedBlock {
            loudness_lufs: block_loudness,
            passed_absolute,
            passed_relative: false, // Updated in finalize
            start_sample: self.total_samples_processed.saturating_sub(self.block_size),
            duration_samples: self.block_size / self.config.channels,
        };

        if passed_absolute {
            self.gated_sum += 10.0_f64.powf(block_loudness / 10.0);
            self.gated_count += 1;
        }

        self.gated_blocks.push(block);
        self.total_blocks += 1;

        // Update gate state based on block loudness
        self.update_gate_state(block_loudness);
    }

    /// Compute the mean-square loudness of the current block.
    fn compute_block_loudness(&self) -> f64 {
        if self.block_buffer.is_empty() {
            return -70.0;
        }

        let mut sum_sq = 0.0;
        for &s in &self.block_buffer {
            sum_sq += s * s;
        }
        let mean_sq = sum_sq / self.block_buffer.len() as f64;

        if mean_sq <= 0.0 {
            -70.0
        } else {
            -0.691 + 10.0 * mean_sq.log10()
        }
    }

    /// Update the gate state based on current loudness.
    fn update_gate_state(&mut self, loudness_lufs: f64) {
        let threshold = self.config.absolute_threshold_lufs;
        let hysteresis = self.config.hysteresis_db;
        let hold_samples = (self.config.hold_time_ms * self.config.sample_rate / 1000.0) as usize;

        match self.state {
            GateState::Closed => {
                if loudness_lufs > threshold + hysteresis {
                    self.state = GateState::Open;
                    self.hold_remaining = hold_samples;
                }
            }
            GateState::Open => {
                if loudness_lufs < threshold {
                    self.state = GateState::Hold;
                    self.hold_remaining = hold_samples;
                } else {
                    self.hold_remaining = hold_samples;
                }
            }
            GateState::Hold => {
                if loudness_lufs > threshold + hysteresis {
                    self.state = GateState::Open;
                    self.hold_remaining = hold_samples;
                } else if self.hold_remaining == 0 {
                    self.state = GateState::Closed;
                } else {
                    self.hold_remaining = self.hold_remaining.saturating_sub(self.hop_size);
                }
            }
        }

        // Update envelope
        let target = if self.state == GateState::Open {
            1.0
        } else {
            0.0
        };
        let coeff = if target > self.envelope {
            self.attack_coeff
        } else {
            self.release_coeff
        };
        self.envelope += coeff * (target - self.envelope);
    }

    /// Get the envelope value (0.0 = fully closed, 1.0 = fully open).
    pub fn envelope(&self) -> f64 {
        self.envelope
    }

    /// Finalize gating and apply relative gate.
    ///
    /// Returns the integrated loudness after both absolute and relative gating.
    pub fn finalize(&mut self) -> f64 {
        if self.gated_count == 0 {
            return -70.0;
        }

        // Compute ungated mean loudness from absolute-gated blocks
        let absolute_gated_loudness =
            -0.691 + 10.0 * (self.gated_sum / self.gated_count as f64).log10();

        // Relative threshold
        let relative_threshold = absolute_gated_loudness + self.config.relative_threshold_lu;

        // Apply relative gate
        let mut final_sum = 0.0;
        let mut final_count = 0usize;

        for block in &mut self.gated_blocks {
            if block.passed_absolute && block.loudness_lufs > relative_threshold {
                block.passed_relative = true;
                final_sum += 10.0_f64.powf(block.loudness_lufs / 10.0);
                final_count += 1;
            }
        }

        if final_count == 0 {
            return -70.0;
        }

        -0.691 + 10.0 * (final_sum / final_count as f64).log10()
    }

    /// Get a reference to all gated blocks.
    pub fn blocks(&self) -> &[GatedBlock] {
        &self.gated_blocks
    }

    /// Get the gate activity ratio (fraction of time gate was open).
    pub fn activity_ratio(&self) -> f64 {
        if self.total_blocks == 0 {
            return 0.0;
        }
        self.gated_count as f64 / self.total_blocks as f64
    }

    /// Reset the gate to initial state.
    pub fn reset(&mut self) {
        self.state = GateState::Closed;
        self.envelope = 0.0;
        self.gated_blocks.clear();
        self.gated_sum = 0.0;
        self.gated_count = 0;
        self.total_blocks = 0;
        self.block_buffer.clear();
        self.samples_in_block = 0;
        self.total_samples_processed = 0;
    }
}

/// Compute a simple RMS level in dB from samples.
#[allow(clippy::cast_precision_loss)]
fn rms_db(samples: &[f32]) -> f64 {
    if samples.is_empty() {
        return -100.0;
    }
    let sum_sq: f64 = samples.iter().map(|&s| f64::from(s) * f64::from(s)).sum();
    let mean_sq = sum_sq / samples.len() as f64;
    if mean_sq <= 0.0 {
        -100.0
    } else {
        10.0 * mean_sq.log10()
    }
}

/// Apply soft gating to a buffer using the envelope value.
pub fn apply_soft_gate(samples: &mut [f32], envelope: f64) {
    let env = envelope as f32;
    for sample in samples.iter_mut() {
        *sample *= env;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gate_config_default() {
        let config = LoudnessGateConfig::new(48000.0, 2);
        assert!((config.absolute_threshold_lufs - (-70.0)).abs() < f64::EPSILON);
        assert!((config.relative_threshold_lu - (-10.0)).abs() < f64::EPSILON);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_gate_config_broadcast() {
        let config = LoudnessGateConfig::broadcast(48000.0, 2);
        assert!((config.hold_time_ms - 400.0).abs() < f64::EPSILON);
        assert!((config.hysteresis_db - 3.0).abs() < f64::EPSILON);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_gate_config_validation_bad_sample_rate() {
        let config = LoudnessGateConfig::new(100.0, 2);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_gate_config_validation_bad_channels() {
        let config = LoudnessGateConfig::new(48000.0, 0);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_gate_initial_state() {
        let config = LoudnessGateConfig::new(48000.0, 1);
        let gate = LoudnessGate::new(config);
        assert_eq!(gate.state(), GateState::Closed);
        assert_eq!(gate.total_blocks(), 0);
        assert_eq!(gate.gated_count(), 0);
    }

    #[test]
    fn test_gate_silence_stays_closed() {
        let config = LoudnessGateConfig::new(48000.0, 1);
        let mut gate = LoudnessGate::new(config);
        let silence = vec![0.0f32; 48000]; // 1 second of silence
        gate.process(&silence);
        assert_eq!(gate.state(), GateState::Closed);
    }

    #[test]
    fn test_gate_loud_signal_opens() {
        let config = LoudnessGateConfig::new(48000.0, 1);
        let mut gate = LoudnessGate::new(config);
        // Generate a loud signal (sine wave at -6 dBFS)
        let samples: Vec<f32> = (0..48000)
            .map(|i| {
                (0.5 * (2.0 * std::f64::consts::PI * 1000.0 * i as f64 / 48000.0).sin()) as f32
            })
            .collect();
        gate.process(&samples);
        // After processing loud signal, gate should have opened at some point
        assert!(gate.gated_count() > 0);
    }

    #[test]
    fn test_gate_finalize_silence() {
        let config = LoudnessGateConfig::new(48000.0, 1);
        let mut gate = LoudnessGate::new(config);
        let silence = vec![0.0f32; 48000];
        gate.process(&silence);
        let result = gate.finalize();
        assert!(result <= -70.0);
    }

    #[test]
    fn test_gate_activity_ratio_silence() {
        let config = LoudnessGateConfig::new(48000.0, 1);
        let mut gate = LoudnessGate::new(config);
        let silence = vec![0.0f32; 48000];
        gate.process(&silence);
        assert!((gate.activity_ratio() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_gate_reset() {
        let config = LoudnessGateConfig::new(48000.0, 1);
        let mut gate = LoudnessGate::new(config);
        let signal: Vec<f32> = (0..48000)
            .map(|i| (0.5 * (2.0 * std::f64::consts::PI * 440.0 * i as f64 / 48000.0).sin()) as f32)
            .collect();
        gate.process(&signal);
        gate.reset();
        assert_eq!(gate.state(), GateState::Closed);
        assert_eq!(gate.total_blocks(), 0);
        assert_eq!(gate.gated_count(), 0);
        assert!(gate.blocks().is_empty());
    }

    #[test]
    fn test_rms_db_silence() {
        let silence = vec![0.0f32; 100];
        assert!((rms_db(&silence) - (-100.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_rms_db_full_scale() {
        let full = vec![1.0f32; 100];
        let level = rms_db(&full);
        assert!((level - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_apply_soft_gate() {
        let mut samples = vec![1.0f32, -1.0, 0.5, -0.5];
        apply_soft_gate(&mut samples, 0.5);
        assert!((samples[0] - 0.5).abs() < 1e-6);
        assert!((samples[1] - (-0.5)).abs() < 1e-6);
    }

    #[test]
    fn test_gate_state_equality() {
        assert_eq!(GateState::Open, GateState::Open);
        assert_ne!(GateState::Open, GateState::Closed);
        assert_ne!(GateState::Hold, GateState::Open);
    }

    #[test]
    fn test_gate_envelope_initial() {
        let config = LoudnessGateConfig::new(48000.0, 1);
        let gate = LoudnessGate::new(config);
        assert!((gate.envelope() - 0.0).abs() < f64::EPSILON);
    }
}
