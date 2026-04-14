//! Dialogue-gated loudness measurement (ITU-R BS.1770-4 with dialogue intelligence gating).
//!
//! Cinema and broadcast delivery (e.g. Dolby Atmos, SMPTE ST 2095-1) requires that integrated
//! loudness be measured **only over segments containing dialogue**, not over music/effects beds.
//! This module layers a dialogue-intelligence gate on top of the standard ITU-R BS.1770-4
//! absolute/relative gating to produce a dialogue-gated integrated loudness figure.
//!
//! # Algorithm
//!
//! 1. Audio is chopped into 100 ms blocks (3 s with 100 ms step) as per BS.1770-4.
//! 2. Each block is also classified as "dialogue-active" by a two-stage criterion:
//!    - Short-time spectral centroid in the 250–3500 Hz speech band must exceed a threshold.
//!    - Speech-band energy ratio (SBER) must exceed a configurable minimum.
//! 3. Only dialogue-active blocks enter the absolute (-70 LUFS) and relative (-10 LU) gates.
//! 4. Integrated loudness is computed from the surviving block set.
//!
//! # References
//!
//! - ITU-R BS.1770-4 (2015)
//! - Dolby Atmos Delivery Specification 1.0 (-27 LUFS cinema target)
//! - SMPTE ST 2095-1:2015 (Speech-Gated Loudness)

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(dead_code)]

use crate::{NormalizeError, NormalizeResult};

// ────────────────────────────────────────────────────────────────────────────
// Constants
// ────────────────────────────────────────────────────────────────────────────

/// Absolute gate threshold per BS.1770-4 (-70 LUFS).
const ABSOLUTE_GATE_LUFS: f64 = -70.0;

/// Relative gate offset per BS.1770-4 (-10 LU below ungated mean).
const RELATIVE_GATE_OFFSET_LU: f64 = -10.0;

/// Block length in seconds (BS.1770-4 uses 3 s blocks but 100 ms step for integration;
/// we use a lighter 400 ms block with 100 ms step to keep latency manageable).
const BLOCK_LEN_S: f64 = 0.4;

/// Block step in seconds.
const BLOCK_STEP_S: f64 = 0.1;

/// Lower edge of speech frequency band for SBER (Hz).
const SPEECH_BAND_LOW_HZ: f64 = 250.0;

/// Upper edge of speech frequency band for SBER (Hz).
const SPEECH_BAND_HIGH_HZ: f64 = 3500.0;

// ────────────────────────────────────────────────────────────────────────────
// DialogueGateConfig
// ────────────────────────────────────────────────────────────────────────────

/// Configuration for dialogue-gated loudness measurement.
#[derive(Clone, Debug)]
pub struct DialogueGateConfig {
    /// Sample rate in Hz.
    pub sample_rate: f64,

    /// Number of audio channels.
    pub channels: usize,

    /// Minimum speech-band energy ratio (0.0–1.0) for a block to be classified as dialogue.
    /// A block whose speech-band energy is below `min_sber * total_energy` is rejected.
    pub min_sber: f64,

    /// Minimum spectral centroid frequency (Hz) to classify a block as dialogue.
    pub min_centroid_hz: f64,

    /// Maximum spectral centroid frequency (Hz) to classify a block as dialogue.
    pub max_centroid_hz: f64,

    /// Cinema dialogue target loudness in LUFS (Dolby Atmos: -27 LUFS).
    pub target_lufs: f64,

    /// True-peak ceiling in dBTP.
    pub true_peak_ceiling_dbtp: f64,

    /// Maximum number of blocks to retain in memory.
    /// Older blocks are dropped when this limit is reached.
    pub max_blocks: usize,
}

impl Default for DialogueGateConfig {
    fn default() -> Self {
        Self {
            sample_rate: 48_000.0,
            channels: 2,
            min_sber: 0.30,
            min_centroid_hz: 300.0,
            max_centroid_hz: 3200.0,
            target_lufs: -23.0,
            true_peak_ceiling_dbtp: -1.0,
            max_blocks: 10_000,
        }
    }
}

impl DialogueGateConfig {
    /// Cinema preset: Dolby Atmos -27 LUFS dialogue-gated measurement.
    pub fn cinema() -> Self {
        Self {
            target_lufs: -27.0,
            min_sber: 0.35,
            ..Default::default()
        }
    }

    /// Broadcast preset: EBU R128 dialogue-gated (-23 LUFS).
    pub fn broadcast() -> Self {
        Self {
            target_lufs: -23.0,
            min_sber: 0.25,
            ..Default::default()
        }
    }

    /// Validate the configuration.
    pub fn validate(&self) -> NormalizeResult<()> {
        if self.sample_rate < 8_000.0 || self.sample_rate > 192_000.0 {
            return Err(NormalizeError::InvalidConfig(format!(
                "sample_rate {:.0} Hz is out of range 8000–192000",
                self.sample_rate
            )));
        }
        if self.channels == 0 || self.channels > 16 {
            return Err(NormalizeError::InvalidConfig(format!(
                "channels {} is out of range 1–16",
                self.channels
            )));
        }
        if self.min_sber < 0.0 || self.min_sber > 1.0 {
            return Err(NormalizeError::InvalidConfig(
                "min_sber must be in 0.0–1.0".to_string(),
            ));
        }
        if self.min_centroid_hz >= self.max_centroid_hz {
            return Err(NormalizeError::InvalidConfig(
                "min_centroid_hz must be < max_centroid_hz".to_string(),
            ));
        }
        Ok(())
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ────────────────────────────────────────────────────────────────────────────

/// Classify a mono/downmixed block using SBER + spectral centroid.
///
/// Returns `true` if the block appears to contain speech/dialogue.
fn classify_dialogue_block(
    block: &[f32],
    sample_rate: f64,
    min_sber: f64,
    min_centroid_hz: f64,
    max_centroid_hz: f64,
) -> bool {
    let n = block.len();
    if n == 0 {
        return false;
    }

    // Compute per-bin spectral power via DFT magnitude squared.
    // We use a simplified single-pass real DFT (O(n²)) — blocks are short
    // (≤ ~20 000 samples) so this is acceptable.
    let n_half = n / 2 + 1;
    let freq_resolution = sample_rate / n as f64;

    let mut total_power = 0.0_f64;
    let mut speech_power = 0.0_f64;
    let mut centroid_num = 0.0_f64;
    let mut centroid_den = 0.0_f64;

    for k in 0..n_half {
        let freq_hz = k as f64 * freq_resolution;

        // DFT bin k real and imaginary parts
        let mut re = 0.0_f64;
        let mut im = 0.0_f64;
        let phase_step = std::f64::consts::TAU * k as f64 / n as f64;
        for (j, &s) in block.iter().enumerate() {
            let angle = phase_step * j as f64;
            re += f64::from(s) * angle.cos();
            im -= f64::from(s) * angle.sin();
        }
        let power = re * re + im * im;

        total_power += power;
        centroid_num += freq_hz * power;
        centroid_den += power;

        if freq_hz >= SPEECH_BAND_LOW_HZ && freq_hz <= SPEECH_BAND_HIGH_HZ {
            speech_power += power;
        }
    }

    if total_power < 1e-20 {
        return false; // silent block — not dialogue
    }

    let sber = speech_power / total_power;
    let centroid = if centroid_den > 0.0 {
        centroid_num / centroid_den
    } else {
        0.0
    };

    sber >= min_sber && centroid >= min_centroid_hz && centroid <= max_centroid_hz
}

/// Compute mean-square (K-weighted approximation: equal-power downmix) for a
/// multi-channel interleaved block.
fn mean_square_loudness(block: &[f32], channels: usize) -> f64 {
    if channels == 0 || block.is_empty() {
        return 0.0;
    }
    let n_frames = block.len() / channels;
    if n_frames == 0 {
        return 0.0;
    }

    // Simple equal-weight channel mean (good approximation for stereo/5.1 when
    // proper K-filter is applied upstream or approximated here).
    let sum_sq: f64 = block
        .iter()
        .map(|&s| {
            let v = f64::from(s);
            v * v
        })
        .sum();
    sum_sq / (n_frames * channels) as f64
}

/// Convert mean-square to LUFS.
#[inline]
fn mean_sq_to_lufs(mean_sq: f64) -> f64 {
    if mean_sq < 1e-25 {
        return -120.0;
    }
    -0.691 + 10.0 * mean_sq.log10()
}

// ────────────────────────────────────────────────────────────────────────────
// DialogueBlock
// ────────────────────────────────────────────────────────────────────────────

/// A single loudness analysis block with dialogue classification flag.
#[derive(Clone, Debug)]
struct DialogueBlock {
    /// Mean-square power of the block (linear).
    mean_sq: f64,
    /// Whether this block was classified as dialogue-active.
    is_dialogue: bool,
}

// ────────────────────────────────────────────────────────────────────────────
// DialogueGateMeasurement
// ────────────────────────────────────────────────────────────────────────────

/// Result produced by [`DialogueGateMeasurer::finish`].
#[derive(Clone, Debug)]
pub struct DialogueGateMeasurement {
    /// Dialogue-gated integrated loudness in LUFS.
    pub integrated_lufs: f64,

    /// Ungated integrated loudness (all blocks, no dialogue filter) in LUFS.
    pub ungated_lufs: f64,

    /// Fraction of blocks classified as dialogue (0.0–1.0).
    pub dialogue_ratio: f64,

    /// Total number of analysis blocks processed.
    pub total_blocks: usize,

    /// Number of dialogue blocks that passed absolute and relative gates.
    pub gated_dialogue_blocks: usize,

    /// Recommended gain in dB to reach the configured target loudness.
    pub recommended_gain_db: f64,
}

// ────────────────────────────────────────────────────────────────────────────
// DialogueGateMeasurer
// ────────────────────────────────────────────────────────────────────────────

/// Streaming dialogue-gated loudness measurer.
///
/// Feed audio via [`process_f32`] in arbitrary-size chunks, then call
/// [`finish`] to obtain the [`DialogueGateMeasurement`].
pub struct DialogueGateMeasurer {
    config: DialogueGateConfig,

    /// Partial block accumulation buffer (interleaved, mixed-down to mono).
    sample_buf: Vec<f32>,

    /// Accumulated blocks for gating.
    blocks: Vec<DialogueBlock>,

    /// Block length in samples (per channel).
    block_len_samples: usize,

    /// Block step in samples (overlap step).
    block_step_samples: usize,
}

impl DialogueGateMeasurer {
    /// Create a new measurer.
    pub fn new(config: DialogueGateConfig) -> NormalizeResult<Self> {
        config.validate()?;
        let block_len_samples =
            ((BLOCK_LEN_S * config.sample_rate).ceil() as usize).max(1);
        let block_step_samples =
            ((BLOCK_STEP_S * config.sample_rate).ceil() as usize).max(1);
        Ok(Self {
            config,
            sample_buf: Vec::new(),
            blocks: Vec::new(),
            block_len_samples,
            block_step_samples,
        })
    }

    /// Downmix interleaved multi-channel samples to mono (equal-weight).
    fn downmix_mono(interleaved: &[f32], channels: usize) -> Vec<f32> {
        if channels == 0 {
            return Vec::new();
        }
        let n_frames = interleaved.len() / channels;
        let mut mono = Vec::with_capacity(n_frames);
        let inv_ch = 1.0 / channels as f32;
        for f in 0..n_frames {
            let mut sum = 0.0_f32;
            for c in 0..channels {
                sum += interleaved[f * channels + c];
            }
            mono.push(sum * inv_ch);
        }
        mono
    }

    /// Feed interleaved audio samples for analysis.
    pub fn process_f32(&mut self, samples: &[f32]) {
        let channels = self.config.channels;
        let mono = Self::downmix_mono(samples, channels);
        self.sample_buf.extend_from_slice(&mono);

        let block_len = self.block_len_samples;
        let block_step = self.block_step_samples;

        while self.sample_buf.len() >= block_len {
            let block: Vec<f32> = self.sample_buf[..block_len].to_vec();
            self.process_block(&block);

            // Advance by step (overlapping blocks).
            if block_step >= self.sample_buf.len() {
                self.sample_buf.clear();
            } else {
                self.sample_buf.drain(..block_step);
            }

            // Safety valve: cap block retention to configured maximum.
            if self.blocks.len() >= self.config.max_blocks {
                break;
            }
        }
    }

    /// Analyse a single mono block and push a `DialogueBlock`.
    fn process_block(&mut self, block: &[f32]) {
        let cfg = &self.config;
        let is_dialogue = classify_dialogue_block(
            block,
            cfg.sample_rate,
            cfg.min_sber,
            cfg.min_centroid_hz,
            cfg.max_centroid_hz,
        );

        // Mean-square is computed on the interleaved multi-channel block at the
        // mono level (already downmixed).
        let mean_sq = mean_square_loudness(block, 1);

        self.blocks.push(DialogueBlock { mean_sq, is_dialogue });
    }

    /// Compute integrated loudness using BS.1770-4 absolute + relative gates,
    /// restricted to dialogue blocks.
    fn compute_gated_lufs(
        blocks: &[DialogueBlock],
        dialogue_only: bool,
    ) -> (f64, usize) {
        let candidate_blocks: Vec<&DialogueBlock> = if dialogue_only {
            blocks.iter().filter(|b| b.is_dialogue).collect()
        } else {
            blocks.iter().collect()
        };

        if candidate_blocks.is_empty() {
            return (-120.0, 0);
        }

        // ── Absolute gate ─────────────────────────────────────────────────
        let abs_threshold = 10.0_f64.powf((ABSOLUTE_GATE_LUFS + 0.691) / 10.0);
        let abs_passed: Vec<&DialogueBlock> = candidate_blocks
            .iter()
            .copied()
            .filter(|b| b.mean_sq >= abs_threshold)
            .collect();

        if abs_passed.is_empty() {
            return (-120.0, 0);
        }

        // ── Ungated mean for relative gate ────────────────────────────────
        let ungated_mean_sq: f64 =
            abs_passed.iter().map(|b| b.mean_sq).sum::<f64>() / abs_passed.len() as f64;
        let ungated_lufs = mean_sq_to_lufs(ungated_mean_sq);
        let rel_threshold_lufs = ungated_lufs + RELATIVE_GATE_OFFSET_LU;
        let rel_threshold = 10.0_f64.powf((rel_threshold_lufs + 0.691) / 10.0);

        // ── Relative gate ─────────────────────────────────────────────────
        let rel_passed: Vec<&DialogueBlock> = abs_passed
            .iter()
            .copied()
            .filter(|b| b.mean_sq >= rel_threshold)
            .collect();

        let count = rel_passed.len();
        if count == 0 {
            return (-120.0, 0);
        }

        let mean_sq: f64 = rel_passed.iter().map(|b| b.mean_sq).sum::<f64>() / count as f64;
        (mean_sq_to_lufs(mean_sq), count)
    }

    /// Finish analysis and return the measurement result.
    pub fn finish(&self) -> NormalizeResult<DialogueGateMeasurement> {
        let total_blocks = self.blocks.len();
        if total_blocks == 0 {
            return Err(NormalizeError::InsufficientData(
                "No blocks were analysed — call process_f32() with audio data first".to_string(),
            ));
        }

        let dialogue_count = self.blocks.iter().filter(|b| b.is_dialogue).count();
        let dialogue_ratio = dialogue_count as f64 / total_blocks as f64;

        let (integrated_lufs, gated_dialogue_blocks) =
            Self::compute_gated_lufs(&self.blocks, true);

        let (ungated_lufs, _) = Self::compute_gated_lufs(&self.blocks, false);

        let recommended_gain_db = if integrated_lufs > -119.0 {
            self.config.target_lufs - integrated_lufs
        } else {
            0.0
        };

        Ok(DialogueGateMeasurement {
            integrated_lufs,
            ungated_lufs,
            dialogue_ratio,
            total_blocks,
            gated_dialogue_blocks,
            recommended_gain_db,
        })
    }

    /// Reset internal state to allow reuse.
    pub fn reset(&mut self) {
        self.sample_buf.clear();
        self.blocks.clear();
    }

    /// Access the configuration.
    pub fn config(&self) -> &DialogueGateConfig {
        &self.config
    }

    /// Number of blocks accumulated so far.
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sine_wave(freq_hz: f32, sample_rate: f32, num_samples: usize, amplitude: f32) -> Vec<f32> {
        (0..num_samples)
            .map(|i| {
                amplitude * (std::f32::consts::TAU * freq_hz * i as f32 / sample_rate).sin()
            })
            .collect()
    }

    fn silence(num_samples: usize) -> Vec<f32> {
        vec![0.0_f32; num_samples]
    }

    #[test]
    fn test_config_default_validates() {
        let cfg = DialogueGateConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_cinema_preset() {
        let cfg = DialogueGateConfig::cinema();
        assert!((cfg.target_lufs - (-27.0)).abs() < 0.01);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_broadcast_preset() {
        let cfg = DialogueGateConfig::broadcast();
        assert!((cfg.target_lufs - (-23.0)).abs() < 0.01);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_invalid_sample_rate() {
        let cfg = DialogueGateConfig {
            sample_rate: 100.0,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_invalid_channels() {
        let cfg = DialogueGateConfig {
            channels: 0,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_finish_no_data_returns_error() {
        let cfg = DialogueGateConfig::default();
        let m = DialogueGateMeasurer::new(cfg).expect("create");
        assert!(m.finish().is_err());
    }

    #[test]
    fn test_silence_produces_low_lufs() {
        let cfg = DialogueGateConfig::default();
        let mut m = DialogueGateMeasurer::new(cfg).expect("create");
        // 2 seconds of silence at 48 kHz stereo
        let silent = silence(48_000 * 2 * 2);
        m.process_f32(&silent);
        // After silence, finish should either error (no blocks pass gate) or return very low LUFS
        let result = m.finish();
        if let Ok(meas) = result {
            assert!(meas.integrated_lufs < -60.0 || meas.gated_dialogue_blocks == 0);
        }
        // error path is also acceptable (no blocks pass gate)
    }

    #[test]
    fn test_speech_tone_classified_as_dialogue() {
        // 1 kHz tone should be classified as speech (within 250–3500 Hz band)
        let sample_rate = 48_000.0_f32;
        let n = (0.5 * sample_rate) as usize; // 500 ms block (mono)
        let block = sine_wave(1000.0, sample_rate, n, 0.5);
        let classified = classify_dialogue_block(
            &block,
            f64::from(sample_rate),
            0.30,
            300.0,
            3200.0,
        );
        assert!(classified, "1 kHz tone should be classified as dialogue");
    }

    #[test]
    fn test_low_frequency_not_classified_as_dialogue() {
        // 60 Hz tone should NOT be classified as speech
        let sample_rate = 48_000.0_f32;
        let n = (0.4 * sample_rate) as usize;
        let block = sine_wave(60.0, sample_rate, n, 0.5);
        let classified = classify_dialogue_block(
            &block,
            f64::from(sample_rate),
            0.30,
            300.0,
            3200.0,
        );
        assert!(!classified, "60 Hz tone should not be classified as dialogue");
    }

    #[test]
    fn test_reset_clears_state() {
        let cfg = DialogueGateConfig::default();
        let mut m = DialogueGateMeasurer::new(cfg).expect("create");
        let audio = sine_wave(1000.0, 48_000.0, 48_000 * 2, 0.5);
        m.process_f32(&audio);
        assert!(m.block_count() > 0);
        m.reset();
        assert_eq!(m.block_count(), 0);
        assert!(m.finish().is_err());
    }

    #[test]
    fn test_block_count_increases_with_audio() {
        let cfg = DialogueGateConfig::default();
        let mut m = DialogueGateMeasurer::new(cfg).expect("create");
        let audio = sine_wave(1000.0, 48_000.0, 48_000 * 3, 0.3);
        m.process_f32(&audio);
        assert!(m.block_count() > 0, "blocks should be accumulated");
    }

    #[test]
    fn test_recommended_gain_direction() {
        // Feed a loud signal — dialogue LUFS should be high → recommended gain negative
        let cfg = DialogueGateConfig {
            target_lufs: -23.0,
            min_sber: 0.10, // lower threshold so tone is accepted
            ..Default::default()
        };
        let mut m = DialogueGateMeasurer::new(cfg).expect("create");
        // 1 kHz sine at 0.5 amplitude, 3 seconds
        let audio = sine_wave(1000.0, 48_000.0, 48_000 * 3, 0.5);
        m.process_f32(&audio);
        if let Ok(meas) = m.finish() {
            if meas.gated_dialogue_blocks > 0 {
                // If measured loudness > target, gain should be negative (attenuate)
                // If measured loudness < target, gain should be positive (boost)
                let expected_sign = if meas.integrated_lufs > -23.0 {
                    -1.0_f64
                } else {
                    1.0_f64
                };
                assert_eq!(
                    meas.recommended_gain_db.signum(),
                    expected_sign,
                    "gain direction should match loudness offset"
                );
            }
        }
    }
}
