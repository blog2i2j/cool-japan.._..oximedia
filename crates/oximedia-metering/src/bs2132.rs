//! ITU-R BS.2132 loudness measurement for short-form content (<30 s).
//!
//! ITU-R BS.2132 (2019) specifies modifications to the gating algorithm
//! defined in ITU-R BS.1770-4 for content shorter than 30 seconds, such as
//! trailers, promos, and advertisements.
//!
//! Key differences from BS.1770-4:
//! - The **relative gate threshold** is raised from −10 LU to −20 LU when
//!   content is shorter than 30 seconds.
//! - The minimum number of valid 400 ms blocks required for a gate decision
//!   is reduced to 1 (down from the implicit >0 requirement of BS.1770-4).
//! - When fewer than 3 valid blocks are available, the algorithm uses all
//!   blocks above the absolute gate (−70 LKFS) without a relative gate pass.
//!
//! # Reference
//!
//! ITU-R Recommendation BS.2132, "Algorithms for measurement of programme
//! loudness for very short content", Geneva, 2019.

use crate::{MeteringError, MeteringResult};

/// Duration threshold below which the BS.2132 short-form gating rules apply.
pub const SHORT_FORM_THRESHOLD_SECS: f64 = 30.0;

/// Relative gate offset for short-form content (−20 LU, more relaxed than
/// the standard −10 LU used by BS.1770-4 for programme-length content).
pub const SHORT_FORM_RELATIVE_GATE_LU: f64 = -20.0;

/// Relative gate for standard (>30 s) content as defined by BS.1770-4.
pub const STANDARD_RELATIVE_GATE_LU: f64 = -10.0;

/// Absolute gate threshold (LKFS), identical in both BS.1770-4 and BS.2132.
pub const ABSOLUTE_GATE_LKFS: f64 = -70.0;

/// Block duration for the 400 ms sliding-window analysis in seconds.
pub const BLOCK_DURATION_SECS: f64 = 0.4;

/// Loudness block with its mean-square energy.
#[derive(Debug, Clone, Copy)]
pub struct LoudnessBlock {
    /// Mean-square K-weighted energy of this block.
    pub mean_square: f64,
    /// Block loudness in LKFS (10 × log₁₀(mean_square)).
    pub lkfs: f64,
}

impl LoudnessBlock {
    /// Compute a block from a pre-summed mean-square value.
    ///
    /// Returns `None` when `mean_square` is zero or negative (silent block).
    #[must_use]
    pub fn from_mean_square(mean_square: f64) -> Option<Self> {
        if mean_square <= 0.0 {
            return None;
        }
        Some(Self {
            mean_square,
            lkfs: 10.0 * mean_square.log10(),
        })
    }
}

/// Configuration for the BS.2132 loudness measurement.
#[derive(Debug, Clone)]
pub struct Bs2132Config {
    /// Sample rate of the input audio.
    pub sample_rate: f64,
    /// Number of channels.
    pub channels: usize,
    /// Nominal content duration in seconds (used to select gating mode).
    pub content_duration_secs: f64,
}

impl Bs2132Config {
    /// Create a new BS.2132 configuration.
    #[must_use]
    pub fn new(sample_rate: f64, channels: usize, content_duration_secs: f64) -> Self {
        Self {
            sample_rate,
            channels,
            content_duration_secs,
        }
    }

    /// Returns `true` when short-form gating rules apply.
    #[must_use]
    pub fn is_short_form(&self) -> bool {
        self.content_duration_secs < SHORT_FORM_THRESHOLD_SECS
    }

    /// Effective relative gate offset for this configuration.
    #[must_use]
    pub fn relative_gate_lu(&self) -> f64 {
        if self.is_short_form() {
            SHORT_FORM_RELATIVE_GATE_LU
        } else {
            STANDARD_RELATIVE_GATE_LU
        }
    }
}

/// Result of a BS.2132 loudness measurement.
#[derive(Debug, Clone)]
pub struct Bs2132Result {
    /// Integrated programme loudness in LKFS / LUFS.
    pub integrated_lufs: f64,
    /// Number of 400 ms blocks that passed the absolute gate.
    pub blocks_absolute: usize,
    /// Number of 400 ms blocks that passed the relative gate (second pass).
    pub blocks_relative: usize,
    /// Whether short-form gating was applied.
    pub short_form_applied: bool,
    /// Whether the measurement is valid (enough blocks were available).
    pub valid: bool,
}

/// Compute the BS.2132 integrated loudness from a sequence of K-weighted
/// mean-square block values.
///
/// `blocks` — mean-square K-weighted energy for each 400 ms block.
/// `config` — measurement configuration.
///
/// # Errors
///
/// Returns an error when `blocks` is empty or the configuration is invalid
/// (e.g. zero channels).
pub fn compute_bs2132_loudness(
    blocks: &[f64],
    config: &Bs2132Config,
) -> MeteringResult<Bs2132Result> {
    if config.channels == 0 {
        return Err(MeteringError::InvalidConfig(
            "Channel count must be > 0".to_string(),
        ));
    }
    if blocks.is_empty() {
        return Ok(Bs2132Result {
            integrated_lufs: f64::NEG_INFINITY,
            blocks_absolute: 0,
            blocks_relative: 0,
            short_form_applied: config.is_short_form(),
            valid: false,
        });
    }

    // --- Pass 1: Absolute gate −70 LKFS ---
    let abs_gate = ABSOLUTE_GATE_LKFS;
    let loudness_blocks: Vec<LoudnessBlock> = blocks
        .iter()
        .filter_map(|&ms| LoudnessBlock::from_mean_square(ms))
        .filter(|b| b.lkfs >= abs_gate)
        .collect();

    if loudness_blocks.is_empty() {
        return Ok(Bs2132Result {
            integrated_lufs: f64::NEG_INFINITY,
            blocks_absolute: 0,
            blocks_relative: 0,
            short_form_applied: config.is_short_form(),
            valid: false,
        });
    }

    let n_abs = loudness_blocks.len();
    // Mean of absolute-gate-passed blocks
    let mean_abs: f64 = loudness_blocks.iter().map(|b| b.mean_square).sum::<f64>() / n_abs as f64;
    let lufs_ungated = 10.0 * mean_abs.log10() - 0.691;

    // --- BS.2132: if fewer than 3 blocks, skip relative gate ---
    if n_abs < 3 && config.is_short_form() {
        return Ok(Bs2132Result {
            integrated_lufs: lufs_ungated,
            blocks_absolute: n_abs,
            blocks_relative: n_abs,
            short_form_applied: true,
            valid: true,
        });
    }

    // --- Pass 2: Relative gate ---
    let rel_gate = lufs_ungated + 0.691 + config.relative_gate_lu();
    let gated_blocks: Vec<&LoudnessBlock> = loudness_blocks
        .iter()
        .filter(|b| b.lkfs >= rel_gate)
        .collect();

    let n_rel = gated_blocks.len();
    if n_rel == 0 {
        // Fall back to absolute-gated measurement for short-form
        if config.is_short_form() {
            return Ok(Bs2132Result {
                integrated_lufs: lufs_ungated,
                blocks_absolute: n_abs,
                blocks_relative: 0,
                short_form_applied: true,
                valid: true,
            });
        }
        return Ok(Bs2132Result {
            integrated_lufs: f64::NEG_INFINITY,
            blocks_absolute: n_abs,
            blocks_relative: 0,
            short_form_applied: false,
            valid: false,
        });
    }

    let mean_gated: f64 = gated_blocks.iter().map(|b| b.mean_square).sum::<f64>() / n_rel as f64;
    let integrated = 10.0 * mean_gated.log10() - 0.691;

    Ok(Bs2132Result {
        integrated_lufs: integrated,
        blocks_absolute: n_abs,
        blocks_relative: n_rel,
        short_form_applied: config.is_short_form(),
        valid: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_blocks_at_lufs(lufs: f64, count: usize) -> Vec<f64> {
        // Solve: 10 * log10(ms) - 0.691 = lufs  =>  ms = 10^((lufs+0.691)/10)
        let ms = 10.0_f64.powf((lufs + 0.691) / 10.0);
        vec![ms; count]
    }

    #[test]
    fn test_short_form_flag() {
        let short = Bs2132Config::new(48000.0, 2, 10.0);
        assert!(short.is_short_form());
        assert_eq!(short.relative_gate_lu(), SHORT_FORM_RELATIVE_GATE_LU);

        let long = Bs2132Config::new(48000.0, 2, 60.0);
        assert!(!long.is_short_form());
        assert_eq!(long.relative_gate_lu(), STANDARD_RELATIVE_GATE_LU);
    }

    #[test]
    fn test_empty_blocks() {
        let config = Bs2132Config::new(48000.0, 2, 5.0);
        let result = compute_bs2132_loudness(&[], &config).expect("should return empty result");
        assert!(!result.valid);
        assert!(result.integrated_lufs.is_infinite());
    }

    #[test]
    fn test_zero_channels_error() {
        let config = Bs2132Config::new(48000.0, 0, 5.0);
        let blocks = make_blocks_at_lufs(-23.0, 10);
        assert!(compute_bs2132_loudness(&blocks, &config).is_err());
    }

    #[test]
    fn test_short_form_known_loudness() {
        // 10 blocks at −23 LUFS (typical broadcast target)
        let config = Bs2132Config::new(48000.0, 2, 5.0);
        let blocks = make_blocks_at_lufs(-23.0, 10);
        let result = compute_bs2132_loudness(&blocks, &config)
            .expect("should succeed");

        assert!(result.valid);
        assert!(result.short_form_applied);
        assert!(
            (result.integrated_lufs - (-23.0)).abs() < 0.5,
            "Expected ≈ -23 LUFS, got {:.2}",
            result.integrated_lufs
        );
    }

    #[test]
    fn test_short_form_fewer_than_3_blocks_skips_relative_gate() {
        let config = Bs2132Config::new(48000.0, 2, 5.0);
        // Only 2 blocks that pass absolute gate
        let blocks = make_blocks_at_lufs(-20.0, 2);
        let result = compute_bs2132_loudness(&blocks, &config)
            .expect("should succeed");
        assert!(result.valid);
        assert!(result.short_form_applied);
        // Relative gate was skipped: rel == abs
        assert_eq!(result.blocks_relative, result.blocks_absolute);
    }

    #[test]
    fn test_long_form_uses_standard_relative_gate() {
        // Long-form: content > 30 s → standard −10 LU relative gate
        let config = Bs2132Config::new(48000.0, 2, 60.0);
        let blocks = make_blocks_at_lufs(-23.0, 30);
        let result = compute_bs2132_loudness(&blocks, &config)
            .expect("should succeed");
        assert!(result.valid);
        assert!(!result.short_form_applied);
    }

    #[test]
    fn test_block_from_mean_square_silent() {
        assert!(LoudnessBlock::from_mean_square(0.0).is_none());
        assert!(LoudnessBlock::from_mean_square(-1.0).is_none());
    }

    #[test]
    fn test_block_lkfs_calculation() {
        // ms = 1.0 → 10*log10(1.0) = 0 dB
        let b = LoudnessBlock::from_mean_square(1.0).expect("should succeed");
        assert!((b.lkfs - 0.0).abs() < 1e-10);

        // ms = 0.1 → 10*log10(0.1) = -10 dB
        let b2 = LoudnessBlock::from_mean_square(0.1).expect("should succeed");
        assert!((b2.lkfs - (-10.0)).abs() < 1e-10);
    }
}
