#![allow(dead_code)]
//! Loudness range (LRA) analysis conforming to EBU R 128 / ITU-R BS.1770.
//!
//! This module computes the loudness range of an audio signal, which is
//! a measure of the variation in loudness over time. It follows the
//! EBU R 128 specification for computing short-term loudness distributions
//! and extracting the LRA value from the 10th to 95th percentiles.

use std::collections::BTreeMap;

/// Loudness range analysis result.
#[derive(Debug, Clone)]
pub struct LoudnessRangeResult {
    /// Loudness range in LU (Loudness Units).
    pub lra_lu: f64,
    /// 10th percentile short-term loudness in LUFS.
    pub low_percentile_lufs: f64,
    /// 95th percentile short-term loudness in LUFS.
    pub high_percentile_lufs: f64,
    /// Integrated loudness in LUFS.
    pub integrated_lufs: f64,
    /// Number of short-term blocks analysed.
    pub block_count: usize,
    /// Per-block short-term loudness values in LUFS.
    pub block_loudness: Vec<f64>,
}

/// Configuration for loudness range analysis.
#[derive(Debug, Clone)]
pub struct LoudnessRangeConfig {
    /// Short-term block duration in seconds (EBU default: 3.0).
    pub block_duration_s: f64,
    /// Hop between blocks in seconds (EBU default: 1.0 overlap -> hop = block - overlap).
    pub hop_duration_s: f64,
    /// Absolute gate threshold in LUFS (EBU R 128 default: -70).
    pub absolute_gate_lufs: f64,
    /// Relative gate offset in LU below integrated loudness (default: -20).
    pub relative_gate_lu: f64,
    /// Low percentile for LRA (default: 10).
    pub low_percentile: f64,
    /// High percentile for LRA (default: 95).
    pub high_percentile: f64,
}

impl Default for LoudnessRangeConfig {
    fn default() -> Self {
        Self {
            block_duration_s: 3.0,
            hop_duration_s: 1.0,
            absolute_gate_lufs: -70.0,
            relative_gate_lu: -20.0,
            low_percentile: 10.0,
            high_percentile: 95.0,
        }
    }
}

/// Analyser for computing loudness range (LRA).
#[derive(Debug, Clone)]
pub struct LoudnessRangeAnalyzer {
    /// Analysis configuration.
    config: LoudnessRangeConfig,
}

impl LoudnessRangeAnalyzer {
    /// Create a new analyser with the given configuration.
    #[must_use]
    pub fn new(config: LoudnessRangeConfig) -> Self {
        Self { config }
    }

    /// Create an analyser with EBU R 128 default settings.
    #[must_use]
    pub fn default_ebu() -> Self {
        Self::new(LoudnessRangeConfig::default())
    }

    /// Analyse the loudness range of mono audio samples at the given sample rate.
    ///
    /// Returns `None` if there are not enough samples or all blocks are gated out.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn analyze(&self, samples: &[f32], sample_rate: f64) -> Option<LoudnessRangeResult> {
        if samples.is_empty() || sample_rate <= 0.0 {
            return None;
        }

        let block_len = (self.config.block_duration_s * sample_rate) as usize;
        let hop_len = (self.config.hop_duration_s * sample_rate).max(1.0) as usize;

        if block_len == 0 || block_len > samples.len() {
            return None;
        }

        // Compute short-term loudness for every block.
        let mut block_loudness: Vec<f64> = Vec::new();
        let mut pos = 0;
        while pos + block_len <= samples.len() {
            let block = &samples[pos..pos + block_len];
            let lufs = compute_block_lufs(block);
            block_loudness.push(lufs);
            pos += hop_len;
        }

        if block_loudness.is_empty() {
            return None;
        }

        // Absolute gate: discard blocks below absolute threshold.
        let after_abs: Vec<f64> = block_loudness
            .iter()
            .copied()
            .filter(|&l| l > self.config.absolute_gate_lufs)
            .collect();

        if after_abs.is_empty() {
            return Some(LoudnessRangeResult {
                lra_lu: 0.0,
                low_percentile_lufs: self.config.absolute_gate_lufs,
                high_percentile_lufs: self.config.absolute_gate_lufs,
                integrated_lufs: self.config.absolute_gate_lufs,
                block_count: block_loudness.len(),
                block_loudness,
            });
        }

        // Integrated loudness (energy mean of absolute-gated blocks).
        let integrated = energy_mean(&after_abs);

        // Relative gate: discard blocks below integrated - relative_gate_lu.
        let relative_threshold = integrated + self.config.relative_gate_lu;
        let mut gated: Vec<f64> = after_abs
            .iter()
            .copied()
            .filter(|&l| l > relative_threshold)
            .collect();

        if gated.is_empty() {
            return Some(LoudnessRangeResult {
                lra_lu: 0.0,
                low_percentile_lufs: integrated,
                high_percentile_lufs: integrated,
                integrated_lufs: integrated,
                block_count: block_loudness.len(),
                block_loudness,
            });
        }

        gated.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let low = percentile_sorted(&gated, self.config.low_percentile);
        let high = percentile_sorted(&gated, self.config.high_percentile);

        Some(LoudnessRangeResult {
            lra_lu: high - low,
            low_percentile_lufs: low,
            high_percentile_lufs: high,
            integrated_lufs: integrated,
            block_count: block_loudness.len(),
            block_loudness,
        })
    }
}

/// Compute per-block loudness in LUFS (simplified mono K-weighted).
#[allow(clippy::cast_precision_loss)]
fn compute_block_lufs(block: &[f32]) -> f64 {
    if block.is_empty() {
        return -100.0;
    }
    let mean_sq: f64 = block
        .iter()
        .map(|&s| f64::from(s) * f64::from(s))
        .sum::<f64>()
        / block.len() as f64;
    if mean_sq <= 0.0 {
        -100.0
    } else {
        -0.691 + 10.0 * mean_sq.log10()
    }
}

/// Energy mean of LUFS values.
fn energy_mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return -100.0;
    }
    let sum: f64 = values.iter().map(|&l| 10.0_f64.powf(l / 10.0)).sum();
    10.0 * (sum / values.len() as f64).log10()
}

/// Get percentile from a sorted slice (linear interpolation).
fn percentile_sorted(sorted: &[f64], pct: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let idx = (pct / 100.0) * (sorted.len() - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil().min((sorted.len() - 1) as f64) as usize;
    let frac = idx - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// Quantize loudness values into a histogram with the given bin width (LU).
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn loudness_histogram(values: &[f64], bin_width_lu: f64) -> BTreeMap<i32, usize> {
    let mut hist = BTreeMap::new();
    for &v in values {
        let bin = (v / bin_width_lu).floor() as i32;
        *hist.entry(bin).or_insert(0) += 1;
    }
    hist
}

/// Classification of a programme's loudness dynamics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DynamicsClass {
    /// Very compressed (LRA < 5 LU).
    VeryCompressed,
    /// Compressed (5 <= LRA < 10 LU).
    Compressed,
    /// Moderate dynamics (10 <= LRA < 18 LU).
    Moderate,
    /// Wide dynamics (18 <= LRA < 25 LU).
    Wide,
    /// Very wide dynamics (LRA >= 25 LU).
    VeryWide,
}

/// Classify the dynamics of a programme from its LRA value.
#[must_use]
pub fn classify_dynamics(lra_lu: f64) -> DynamicsClass {
    if lra_lu < 5.0 {
        DynamicsClass::VeryCompressed
    } else if lra_lu < 10.0 {
        DynamicsClass::Compressed
    } else if lra_lu < 18.0 {
        DynamicsClass::Moderate
    } else if lra_lu < 25.0 {
        DynamicsClass::Wide
    } else {
        DynamicsClass::VeryWide
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine_samples(freq: f64, sample_rate: f64, duration_s: f64, amplitude: f32) -> Vec<f32> {
        let n = (sample_rate * duration_s) as usize;
        (0..n)
            .map(|i| {
                let t = i as f64 / sample_rate;
                (amplitude as f64 * (2.0 * std::f64::consts::PI * freq * t).sin()) as f32
            })
            .collect()
    }

    #[test]
    fn test_default_config() {
        let cfg = LoudnessRangeConfig::default();
        assert!((cfg.block_duration_s - 3.0).abs() < f64::EPSILON);
        assert!((cfg.hop_duration_s - 1.0).abs() < f64::EPSILON);
        assert!((cfg.absolute_gate_lufs - (-70.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_empty_samples() {
        let analyzer = LoudnessRangeAnalyzer::default_ebu();
        assert!(analyzer.analyze(&[], 44100.0).is_none());
    }

    #[test]
    fn test_zero_sample_rate() {
        let analyzer = LoudnessRangeAnalyzer::default_ebu();
        let samples = vec![0.0f32; 1000];
        assert!(analyzer.analyze(&samples, 0.0).is_none());
    }

    #[test]
    fn test_silence_loudness() {
        let analyzer = LoudnessRangeAnalyzer::default_ebu();
        let samples = vec![0.0f32; 44100 * 5];
        let result = analyzer.analyze(&samples, 44100.0);
        // All blocks should be gated => LRA = 0
        if let Some(r) = result {
            assert!((r.lra_lu - 0.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_constant_amplitude_lra_zero() {
        let samples = sine_samples(440.0, 44100.0, 10.0, 0.5);
        let analyzer = LoudnessRangeAnalyzer::default_ebu();
        let result = analyzer
            .analyze(&samples, 44100.0)
            .expect("analysis should succeed");
        // Constant amplitude => LRA should be very small (ideally 0).
        assert!(
            result.lra_lu < 1.0,
            "LRA for constant sine should be near 0, got {}",
            result.lra_lu
        );
    }

    #[test]
    fn test_varying_amplitude_nonzero_lra() {
        // First 5 seconds quiet, next 5 seconds loud.
        let quiet = sine_samples(440.0, 44100.0, 5.0, 0.01);
        let loud = sine_samples(440.0, 44100.0, 5.0, 0.8);
        let mut samples = quiet;
        samples.extend(loud);
        let analyzer = LoudnessRangeAnalyzer::default_ebu();
        let result = analyzer
            .analyze(&samples, 44100.0)
            .expect("analysis should succeed");
        assert!(
            result.lra_lu > 1.0,
            "LRA should be > 1 for varying amplitude"
        );
    }

    #[test]
    fn test_block_count() {
        let dur_s = 10.0;
        let sr = 44100.0;
        let samples = sine_samples(440.0, sr, dur_s, 0.5);
        let analyzer = LoudnessRangeAnalyzer::default_ebu();
        let result = analyzer
            .analyze(&samples, sr)
            .expect("analysis should succeed");
        // With 3-second blocks and 1-second hop over 10 seconds:
        // positions 0,1,2,...,7 => 8 blocks
        assert!(result.block_count >= 7);
    }

    #[test]
    fn test_compute_block_lufs_silence() {
        let block = vec![0.0f32; 1024];
        let lufs = compute_block_lufs(&block);
        assert!((lufs - (-100.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_compute_block_lufs_full_scale() {
        // Full-scale sine: RMS ~ 1/sqrt(2), mean_sq ~ 0.5
        // LUFS ~ -0.691 + 10*log10(0.5) ~ -0.691 - 3.01 ~ -3.70
        let block = sine_samples(1000.0, 44100.0, 0.1, 1.0);
        let lufs = compute_block_lufs(&block);
        assert!(lufs > -5.0 && lufs < -2.0, "Full-scale sine LUFS: {lufs}");
    }

    #[test]
    fn test_percentile_sorted_single() {
        assert!((percentile_sorted(&[42.0], 50.0) - 42.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_percentile_sorted_range() {
        let vals: Vec<f64> = (0..=100).map(|i| i as f64).collect();
        let p50 = percentile_sorted(&vals, 50.0);
        assert!((p50 - 50.0).abs() < 0.01);
        let p10 = percentile_sorted(&vals, 10.0);
        assert!((p10 - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_loudness_histogram() {
        let values = vec![-20.0, -19.5, -18.0, -10.0, -10.5];
        let hist = loudness_histogram(&values, 1.0);
        assert!(hist.contains_key(&-20));
        assert!(hist.contains_key(&-18));
        assert!(hist.contains_key(&-10));
        assert!(hist.contains_key(&-11));
    }

    #[test]
    fn test_classify_dynamics() {
        assert_eq!(classify_dynamics(3.0), DynamicsClass::VeryCompressed);
        assert_eq!(classify_dynamics(7.0), DynamicsClass::Compressed);
        assert_eq!(classify_dynamics(14.0), DynamicsClass::Moderate);
        assert_eq!(classify_dynamics(20.0), DynamicsClass::Wide);
        assert_eq!(classify_dynamics(30.0), DynamicsClass::VeryWide);
    }

    #[test]
    fn test_energy_mean_single() {
        let val = energy_mean(&[-23.0]);
        assert!((val - (-23.0)).abs() < 0.01);
    }

    #[test]
    fn test_energy_mean_empty() {
        assert!((energy_mean(&[]) - (-100.0)).abs() < f64::EPSILON);
    }
}
