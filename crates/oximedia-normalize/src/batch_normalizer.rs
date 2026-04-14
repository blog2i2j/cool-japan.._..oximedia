//! Batch loudness normalization — two-pass per-file gain calculation and scheduling.
//!
//! This module provides a batch normalization engine that operates on collections of audio
//! buffers (or file metadata) using a two-pass approach:
//!
//! 1. **Measurement pass** — measure the integrated loudness of every item.
//! 2. **Gain scheduling pass** — compute per-item gain offsets to reach a shared target.
//!
//! The engine supports two gain modes:
//!
//! - **Independent** — each item is normalized to the absolute target loudness.
//! - **Album** — all items are shifted by the same gain (derived from the loudest item) so
//!   relative loudness relationships between tracks are preserved.
//!
//! # Example
//!
//! ```rust
//! use oximedia_normalize::batch_normalizer::{
//!     BatchNormalizer, BatchNormalizerConfig, GainMode,
//! };
//!
//! let config = BatchNormalizerConfig {
//!     target_lufs: -14.0,
//!     max_gain_db: 20.0,
//!     mode: GainMode::Independent,
//!     ..Default::default()
//! };
//!
//! let mut normalizer = BatchNormalizer::new(config).expect("create");
//!
//! // Measure phase
//! let audio1 = vec![0.1_f32; 48_000]; // 1 s of audio at low amplitude
//! let audio2 = vec![0.3_f32; 48_000];
//! let id1 = normalizer.measure("track1", &audio1, 48_000.0, 1).expect("measure");
//! let id2 = normalizer.measure("track2", &audio2, 48_000.0, 1).expect("measure");
//!
//! // Schedule gains
//! let schedule = normalizer.schedule_gains().expect("schedule");
//!
//! // Apply gains
//! let mut out1 = audio1.clone();
//! schedule.apply_to_item(id1, &audio1, &mut out1).expect("apply");
//! ```

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(dead_code)]

use crate::{NormalizeError, NormalizeResult};
use std::collections::HashMap;

// ────────────────────────────────────────────────────────────────────────────
// GainMode
// ────────────────────────────────────────────────────────────────────────────

/// Gain scheduling mode for batch normalization.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum GainMode {
    /// Each item is independently normalized to the target loudness.
    #[default]
    Independent,

    /// All items receive the same gain offset, preserving relative loudness relationships.
    /// The gain is derived from the loudest item so that it just reaches the target.
    Album,

    /// Each item receives the same gain offset, derived from the *average* measured loudness
    /// across the batch.
    AlbumAverage,
}

// ────────────────────────────────────────────────────────────────────────────
// BatchNormalizerConfig
// ────────────────────────────────────────────────────────────────────────────

/// Configuration for the batch loudness normalizer.
#[derive(Clone, Debug)]
pub struct BatchNormalizerConfig {
    /// Target integrated loudness in LUFS.
    pub target_lufs: f64,

    /// Maximum gain allowed in dB (safety cap).
    pub max_gain_db: f64,

    /// Minimum gain (maximum attenuation) allowed in dB.
    pub min_gain_db: f64,

    /// Gain mode: independent, album, or album-average.
    pub mode: GainMode,

    /// True-peak ceiling in dBTP.  If `None`, no peak limiting is applied.
    pub true_peak_ceiling_dbtp: Option<f64>,

    /// If `true`, an item that would require more gain than `max_gain_db` is skipped
    /// (gain is clamped to max but a warning flag is set).
    pub clamp_gain: bool,
}

impl Default for BatchNormalizerConfig {
    fn default() -> Self {
        Self {
            target_lufs: -14.0,
            max_gain_db: 20.0,
            min_gain_db: -20.0,
            mode: GainMode::Independent,
            true_peak_ceiling_dbtp: Some(-1.0),
            clamp_gain: true,
        }
    }
}

impl BatchNormalizerConfig {
    /// Validate the configuration.
    pub fn validate(&self) -> NormalizeResult<()> {
        if self.max_gain_db <= 0.0 {
            return Err(NormalizeError::InvalidConfig(
                "max_gain_db must be > 0".to_string(),
            ));
        }
        if self.min_gain_db > 0.0 {
            return Err(NormalizeError::InvalidConfig(
                "min_gain_db must be ≤ 0".to_string(),
            ));
        }
        if let Some(ceil) = self.true_peak_ceiling_dbtp {
            if ceil > 0.0 {
                return Err(NormalizeError::InvalidConfig(
                    "true_peak_ceiling_dbtp must be ≤ 0".to_string(),
                ));
            }
        }
        Ok(())
    }
}

// ────────────────────────────────────────────────────────────────────────────
// ItemMeasurement
// ────────────────────────────────────────────────────────────────────────────

/// Measurement result for a single batch item.
#[derive(Clone, Debug)]
pub struct ItemMeasurement {
    /// User-provided label (e.g. file name).
    pub label: String,

    /// Measured integrated loudness in LUFS.
    pub integrated_lufs: f64,

    /// Measured true peak in dBTP.
    pub true_peak_dbtp: f64,

    /// Sample rate at which the audio was measured.
    pub sample_rate: f64,

    /// Number of channels.
    pub channels: usize,
}

// ────────────────────────────────────────────────────────────────────────────
// GainEntry
// ────────────────────────────────────────────────────────────────────────────

/// Scheduled gain for a single batch item.
#[derive(Clone, Debug)]
pub struct GainEntry {
    /// Item ID within this batch.
    pub item_id: usize,

    /// User-provided label.
    pub label: String,

    /// Gain in dB to apply to reach the target loudness.
    pub gain_db: f64,

    /// Linear gain factor (10^(gain_db / 20)).
    pub gain_linear: f64,

    /// Measured loudness before normalization.
    pub measured_lufs: f64,

    /// `true` if the gain was clamped because it exceeded the configured limits.
    pub gain_clamped: bool,
}

impl GainEntry {
    fn new(item_id: usize, label: String, raw_gain_db: f64, measured_lufs: f64, config: &BatchNormalizerConfig) -> Self {
        let clamped = raw_gain_db > config.max_gain_db || raw_gain_db < config.min_gain_db;
        let gain_db = raw_gain_db.clamp(config.min_gain_db, config.max_gain_db);
        let gain_linear = 10.0_f64.powf(gain_db / 20.0);
        Self {
            item_id,
            label,
            gain_db,
            gain_linear,
            measured_lufs,
            gain_clamped: clamped,
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// GainSchedule
// ────────────────────────────────────────────────────────────────────────────

/// The result of the scheduling pass — a per-item gain table.
#[derive(Clone, Debug)]
pub struct GainSchedule {
    /// Ordered list of gain entries (indexed by item_id).
    pub entries: Vec<GainEntry>,

    /// Effective target loudness used (may differ from config in album modes).
    pub effective_target_lufs: f64,

    /// Number of items whose gain was clamped.
    pub clamped_count: usize,
}

impl GainSchedule {
    /// Return the gain entry for an item ID, if it exists.
    pub fn entry(&self, item_id: usize) -> Option<&GainEntry> {
        self.entries.iter().find(|e| e.item_id == item_id)
    }

    /// Apply the scheduled gain for `item_id` to a mono/interleaved f32 buffer.
    ///
    /// Returns `Err` if the item_id is not found in the schedule.
    pub fn apply_to_item(
        &self,
        item_id: usize,
        input: &[f32],
        output: &mut [f32],
    ) -> NormalizeResult<()> {
        if output.len() != input.len() {
            return Err(NormalizeError::ProcessingError(
                "output buffer must be the same length as input".to_string(),
            ));
        }
        let entry = self.entry(item_id).ok_or_else(|| {
            NormalizeError::ProcessingError(format!("item_id {item_id} not found in schedule"))
        })?;
        let gain = entry.gain_linear as f32;
        for (o, &s) in output.iter_mut().zip(input.iter()) {
            *o = s * gain;
        }
        Ok(())
    }

    /// Apply the scheduled gain for `item_id` in-place (f32 buffer).
    pub fn apply_in_place(&self, item_id: usize, samples: &mut [f32]) -> NormalizeResult<()> {
        let entry = self.entry(item_id).ok_or_else(|| {
            NormalizeError::ProcessingError(format!("item_id {item_id} not found in schedule"))
        })?;
        let gain = entry.gain_linear as f32;
        for s in samples.iter_mut() {
            *s *= gain;
        }
        Ok(())
    }

    /// Apply the scheduled gain for `item_id` in-place (f64 buffer).
    pub fn apply_in_place_f64(&self, item_id: usize, samples: &mut [f64]) -> NormalizeResult<()> {
        let entry = self.entry(item_id).ok_or_else(|| {
            NormalizeError::ProcessingError(format!("item_id {item_id} not found in schedule"))
        })?;
        let gain = entry.gain_linear;
        for s in samples.iter_mut() {
            *s *= gain;
        }
        Ok(())
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Loudness helpers
// ────────────────────────────────────────────────────────────────────────────

/// Compute the true peak in dBTP from a mono or interleaved buffer.
///
/// Uses the maximum absolute value (no oversampling — conservative estimate).
fn compute_true_peak_dbtp(samples: &[f32]) -> f64 {
    let peak = samples
        .iter()
        .map(|s| s.abs())
        .fold(0.0_f32, f32::max);
    if peak < 1e-15 {
        -120.0
    } else {
        20.0 * f64::from(peak).log10()
    }
}

/// Compute integrated loudness in LUFS using an ITU-R BS.1770-4 approximation.
///
/// This is a simplified (ungated) energy-based measurement.  For production use,
/// the full gated measurement in `oximedia-metering` should be preferred; this
/// implementation is intentionally self-contained to keep the module dependency-free.
fn compute_loudness_lufs(samples: &[f32], sample_rate: f64, channels: usize) -> f64 {
    if samples.is_empty() || channels == 0 {
        return -120.0;
    }
    // K-weighting approximation: apply a first-order high-shelf pre-filter (+4 dB at 4 kHz).
    // Here we compute a simplified version using the definition:
    //   L = -0.691 + 10 * log10(mean_sq)
    // with a roughK-weighting achieved by the shelf coefficients below.
    let b0_hs = 1.53512485958697;
    let b1_hs = -2.69169618940638;
    let b2_hs = 1.19839281085285;
    let a1_hs = -1.69065929318241;
    let a2_hs = 0.73248077421585;

    // High-pass pre-filter
    let b0_hp = 1.0_f64;
    let b1_hp = -2.0_f64;
    let b2_hp = 1.0_f64;
    let a1_hp = -1.99004745483398_f64;
    let a2_hp = 0.99007225036603_f64;

    // Scale filter coefficients to the sample rate if not 48 kHz.
    // For simplicity we apply the filters regardless (they are designed for 48 kHz).
    let _ = sample_rate; // accepted but not used in this approximation

    let n_frames = samples.len() / channels;
    let mut total_mean_sq = 0.0_f64;

    for c in 0..channels {
        let channel_samples: Vec<f64> = (0..n_frames)
            .map(|f| f64::from(samples[f * channels + c]))
            .collect();

        // Stage 1: High-shelf filter
        let mut stage1 = Vec::with_capacity(n_frames);
        let (mut x1, mut x2, mut y1, mut y2) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
        for &x in &channel_samples {
            let y = b0_hs * x + b1_hs * x1 + b2_hs * x2 - a1_hs * y1 - a2_hs * y2;
            stage1.push(y);
            x2 = x1;
            x1 = x;
            y2 = y1;
            y1 = y;
        }

        // Stage 2: High-pass filter
        let (mut x1, mut x2, mut y1, mut y2) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
        let mut ch_sum_sq = 0.0_f64;
        for x in stage1 {
            let y = b0_hp * x + b1_hp * x1 + b2_hp * x2 - a1_hp * y1 - a2_hp * y2;
            ch_sum_sq += y * y;
            x2 = x1;
            x1 = x;
            y2 = y1;
            y1 = y;
        }
        total_mean_sq += ch_sum_sq / n_frames as f64;
    }

    let mean_sq = total_mean_sq / channels as f64;
    if mean_sq < 1e-25 {
        -120.0
    } else {
        -0.691 + 10.0 * mean_sq.log10()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// BatchNormalizer
// ────────────────────────────────────────────────────────────────────────────

/// Batch loudness normalizer.
///
/// Collects audio items during the measurement pass, then produces a
/// [`GainSchedule`] for the second (application) pass.
pub struct BatchNormalizer {
    config: BatchNormalizerConfig,
    measurements: Vec<ItemMeasurement>,
    label_to_id: HashMap<String, usize>,
}

impl BatchNormalizer {
    /// Create a new batch normalizer.
    pub fn new(config: BatchNormalizerConfig) -> NormalizeResult<Self> {
        config.validate()?;
        Ok(Self {
            config,
            measurements: Vec::new(),
            label_to_id: HashMap::new(),
        })
    }

    /// Measure the loudness of an audio item and register it in the batch.
    ///
    /// Returns the item ID assigned to this entry.
    pub fn measure(
        &mut self,
        label: impl Into<String>,
        samples: &[f32],
        sample_rate: f64,
        channels: usize,
    ) -> NormalizeResult<usize> {
        if channels == 0 {
            return Err(NormalizeError::InvalidConfig(
                "channels must be > 0".to_string(),
            ));
        }
        if samples.is_empty() {
            return Err(NormalizeError::InsufficientData(
                "audio buffer is empty".to_string(),
            ));
        }
        let label: String = label.into();
        let item_id = self.measurements.len();
        let integrated_lufs = compute_loudness_lufs(samples, sample_rate, channels);
        let true_peak_dbtp = compute_true_peak_dbtp(samples);
        let m = ItemMeasurement {
            label: label.clone(),
            integrated_lufs,
            true_peak_dbtp,
            sample_rate,
            channels,
        };
        self.label_to_id.insert(label, item_id);
        self.measurements.push(m);
        Ok(item_id)
    }

    /// Measure loudness from pre-computed metrics (useful when audio is too large to keep in RAM).
    pub fn register_measurement(
        &mut self,
        label: impl Into<String>,
        integrated_lufs: f64,
        true_peak_dbtp: f64,
        sample_rate: f64,
        channels: usize,
    ) -> NormalizeResult<usize> {
        if channels == 0 {
            return Err(NormalizeError::InvalidConfig(
                "channels must be > 0".to_string(),
            ));
        }
        let label: String = label.into();
        let item_id = self.measurements.len();
        let m = ItemMeasurement {
            label: label.clone(),
            integrated_lufs,
            true_peak_dbtp,
            sample_rate,
            channels,
        };
        self.label_to_id.insert(label, item_id);
        self.measurements.push(m);
        Ok(item_id)
    }

    /// Schedule gains for all registered items.
    ///
    /// Must be called after all items have been measured.
    pub fn schedule_gains(&self) -> NormalizeResult<GainSchedule> {
        if self.measurements.is_empty() {
            return Err(NormalizeError::InsufficientData(
                "no items have been measured".to_string(),
            ));
        }

        let target = self.config.target_lufs;
        let effective_target = target;

        let album_gain_db: Option<f64> = match self.config.mode {
            GainMode::Independent => None,
            GainMode::Album => {
                // Use the item with the highest loudness as the reference.
                let max_lufs = self
                    .measurements
                    .iter()
                    .map(|m| m.integrated_lufs)
                    .fold(f64::NEG_INFINITY, f64::max);
                // Single gain that brings the loudest item to target.
                Some(target - max_lufs)
            }
            GainMode::AlbumAverage => {
                let avg_lufs = self
                    .measurements
                    .iter()
                    .map(|m| m.integrated_lufs)
                    .sum::<f64>()
                    / self.measurements.len() as f64;
                Some(target - avg_lufs)
            }
        };

        let mut entries = Vec::with_capacity(self.measurements.len());
        let mut clamped_count = 0usize;

        for (item_id, m) in self.measurements.iter().enumerate() {
            let raw_gain_db = match album_gain_db {
                Some(g) => g,
                None => target - m.integrated_lufs,
            };

            // Optionally reduce gain if it would push the true peak above the ceiling.
            let raw_gain_db = if let Some(ceil_dbtp) = self.config.true_peak_ceiling_dbtp {
                let peak_after = m.true_peak_dbtp + raw_gain_db;
                if peak_after > ceil_dbtp {
                    raw_gain_db - (peak_after - ceil_dbtp)
                } else {
                    raw_gain_db
                }
            } else {
                raw_gain_db
            };

            let entry = GainEntry::new(item_id, m.label.clone(), raw_gain_db, m.integrated_lufs, &self.config);
            if entry.gain_clamped {
                clamped_count += 1;
            }
            entries.push(entry);
        }

        Ok(GainSchedule {
            entries,
            effective_target_lufs: effective_target,
            clamped_count,
        })
    }

    /// Look up item ID by label.
    pub fn item_id(&self, label: &str) -> Option<usize> {
        self.label_to_id.get(label).copied()
    }

    /// Number of items registered.
    pub fn item_count(&self) -> usize {
        self.measurements.len()
    }

    /// Access all measurements.
    pub fn measurements(&self) -> &[ItemMeasurement] {
        &self.measurements
    }

    /// Reset the batch (clear all measurements).
    pub fn reset(&mut self) {
        self.measurements.clear();
        self.label_to_id.clear();
    }

    /// Access the configuration.
    pub fn config(&self) -> &BatchNormalizerConfig {
        &self.config
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_normalizer() -> BatchNormalizer {
        BatchNormalizer::new(BatchNormalizerConfig::default()).expect("create")
    }

    fn sine_samples(amplitude: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| amplitude * (std::f32::consts::TAU * 1000.0 * i as f32 / 48_000.0).sin())
            .collect()
    }

    #[test]
    fn test_config_validate_ok() {
        let cfg = BatchNormalizerConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_validate_bad_max_gain() {
        let cfg = BatchNormalizerConfig {
            max_gain_db: -1.0,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validate_bad_min_gain() {
        let cfg = BatchNormalizerConfig {
            min_gain_db: 5.0,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_measure_registers_item() {
        let mut bn = default_normalizer();
        let samples = sine_samples(0.1, 48_000);
        let id = bn.measure("track1", &samples, 48_000.0, 1).expect("measure");
        assert_eq!(id, 0);
        assert_eq!(bn.item_count(), 1);
        assert_eq!(bn.item_id("track1"), Some(0));
    }

    #[test]
    fn test_schedule_empty_batch_errors() {
        let bn = default_normalizer();
        assert!(bn.schedule_gains().is_err());
    }

    #[test]
    fn test_independent_gain_targets_lufs() {
        let cfg = BatchNormalizerConfig {
            target_lufs: -14.0,
            true_peak_ceiling_dbtp: None,
            mode: GainMode::Independent,
            ..Default::default()
        };
        let mut bn = BatchNormalizer::new(cfg).expect("create");
        // Register a pre-computed measurement at -20 LUFS.
        let id = bn
            .register_measurement("t1", -20.0, -3.0, 48_000.0, 2)
            .expect("register");
        let schedule = bn.schedule_gains().expect("schedule");
        let entry = schedule.entry(id).expect("entry");
        // Expected gain = -14 - (-20) = +6 dB
        assert!((entry.gain_db - 6.0).abs() < 0.01, "expected 6 dB, got {}", entry.gain_db);
    }

    #[test]
    fn test_album_mode_preserves_relative_loudness() {
        let cfg = BatchNormalizerConfig {
            target_lufs: -14.0,
            true_peak_ceiling_dbtp: None,
            mode: GainMode::Album,
            ..Default::default()
        };
        let mut bn = BatchNormalizer::new(cfg).expect("create");
        let id0 = bn.register_measurement("t0", -20.0, -6.0, 48_000.0, 2).expect("register");
        let id1 = bn.register_measurement("t1", -16.0, -3.0, 48_000.0, 2).expect("register");
        let schedule = bn.schedule_gains().expect("schedule");
        let g0 = schedule.entry(id0).expect("entry").gain_db;
        let g1 = schedule.entry(id1).expect("entry").gain_db;
        // Both should receive the same gain in album mode.
        assert!((g0 - g1).abs() < 0.01, "gains differ in album mode: {g0} vs {g1}");
        // The loudest item (-16 LUFS) should be brought to -14 → gain = +2 dB.
        assert!((g1 - 2.0).abs() < 0.01, "expected +2 dB for loudest item, got {g1}");
    }

    #[test]
    fn test_gain_clamped_when_exceeds_max() {
        let cfg = BatchNormalizerConfig {
            target_lufs: -14.0,
            max_gain_db: 5.0, // low cap
            true_peak_ceiling_dbtp: None,
            mode: GainMode::Independent,
            ..Default::default()
        };
        let mut bn = BatchNormalizer::new(cfg).expect("create");
        // At -40 LUFS, raw gain would be +26 dB — above max of 5.
        let id = bn.register_measurement("quiet", -40.0, -20.0, 48_000.0, 1).expect("register");
        let schedule = bn.schedule_gains().expect("schedule");
        let entry = schedule.entry(id).expect("entry");
        assert!(entry.gain_clamped, "gain should be flagged as clamped");
        assert!((entry.gain_db - 5.0).abs() < 0.01, "gain should be clamped to 5 dB");
        assert_eq!(schedule.clamped_count, 1);
    }

    #[test]
    fn test_apply_to_item_scales_samples() {
        let cfg = BatchNormalizerConfig {
            target_lufs: -14.0,
            true_peak_ceiling_dbtp: None,
            mode: GainMode::Independent,
            ..Default::default()
        };
        let mut bn = BatchNormalizer::new(cfg).expect("create");
        // Register a measurement manually at 0 dB gain (measured = target).
        let id = bn.register_measurement("unity", -14.0, -3.0, 48_000.0, 1).expect("register");
        let schedule = bn.schedule_gains().expect("schedule");
        let input = vec![0.5_f32; 10];
        let mut output = vec![0.0_f32; 10];
        schedule.apply_to_item(id, &input, &mut output).expect("apply");
        // gain = 0 dB → linear = 1.0 → output ≈ input
        for (&i, &o) in input.iter().zip(output.iter()) {
            assert!((o - i).abs() < 1e-5, "unity gain: out {o} != in {i}");
        }
    }

    #[test]
    fn test_apply_in_place_f64() {
        let cfg = BatchNormalizerConfig {
            target_lufs: -14.0,
            true_peak_ceiling_dbtp: None,
            mode: GainMode::Independent,
            ..Default::default()
        };
        let mut bn = BatchNormalizer::new(cfg).expect("create");
        // +6 dB gain: measured -20 LUFS, target -14 LUFS.
        let id = bn.register_measurement("t", -20.0, -6.0, 48_000.0, 1).expect("register");
        let schedule = bn.schedule_gains().expect("schedule");
        let mut samples = vec![1.0_f64; 4];
        schedule.apply_in_place_f64(id, &mut samples).expect("apply");
        let expected = 10.0_f64.powf(6.0 / 20.0);
        for &s in &samples {
            assert!((s - expected).abs() < 1e-9, "expected {expected}, got {s}");
        }
    }

    #[test]
    fn test_reset_clears_measurements() {
        let mut bn = default_normalizer();
        let samples = sine_samples(0.1, 1024);
        bn.measure("a", &samples, 48_000.0, 1).expect("measure");
        assert_eq!(bn.item_count(), 1);
        bn.reset();
        assert_eq!(bn.item_count(), 0);
        assert!(bn.item_id("a").is_none());
    }
}
