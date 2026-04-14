#![allow(dead_code)]
//! Dynamic range quality assessment for luma planes.
//!
//! Evaluates whether the full luminance range is being utilized and detects
//! crushed blacks, clipped highlights, and poor contrast that degrade
//! perceived quality.

/// Result of a dynamic range quality analysis.
#[derive(Debug, Clone)]
pub struct DynamicRangeResult {
    /// Overall dynamic range quality score in \[0.0, 1.0\] (1.0 = ideal).
    pub score: f64,
    /// Minimum luma value in the frame.
    pub min_luma: u8,
    /// Maximum luma value in the frame.
    pub max_luma: u8,
    /// Luma range (max - min).
    pub range: u8,
    /// Mean luma value.
    pub mean_luma: f64,
    /// Standard deviation of luma.
    pub std_luma: f64,
    /// Fraction of pixels at or below the black threshold.
    pub crushed_black_ratio: f64,
    /// Fraction of pixels at or above the white threshold.
    pub clipped_white_ratio: f64,
    /// Contrast ratio (max / (min + 1)) in linear domain.
    pub contrast_ratio: f64,
}

/// Configuration for dynamic range quality analysis.
#[derive(Debug, Clone)]
pub struct DynamicRangeConfig {
    /// Luma value at or below which pixels are considered "crushed black".
    pub black_threshold: u8,
    /// Luma value at or above which pixels are considered "clipped white".
    pub white_threshold: u8,
    /// Maximum tolerable crushed-black ratio before penalizing.
    pub max_black_ratio: f64,
    /// Maximum tolerable clipped-white ratio before penalizing.
    pub max_white_ratio: f64,
    /// Target range (ideal max - min) used for scoring.
    pub target_range: u8,
}

impl Default for DynamicRangeConfig {
    fn default() -> Self {
        Self {
            black_threshold: 16,
            white_threshold: 235,
            max_black_ratio: 0.05,
            max_white_ratio: 0.05,
            target_range: 219, // broadcast range 16..235
        }
    }
}

/// Build a 256-bin histogram from a luma plane.
pub fn build_histogram(luma: &[u8]) -> [u64; 256] {
    let mut hist = [0u64; 256];
    for &v in luma {
        hist[v as usize] += 1;
    }
    hist
}

/// Compute basic statistics from a luma plane.
#[allow(clippy::cast_precision_loss)]
pub fn luma_statistics(luma: &[u8]) -> (u8, u8, f64, f64) {
    if luma.is_empty() {
        return (0, 0, 0.0, 0.0);
    }
    let mut min_v = 255u8;
    let mut max_v = 0u8;
    let mut sum = 0u64;
    for &v in luma {
        if v < min_v {
            min_v = v;
        }
        if v > max_v {
            max_v = v;
        }
        sum += v as u64;
    }
    let mean = sum as f64 / luma.len() as f64;
    let variance = luma
        .iter()
        .map(|&v| {
            let d = v as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / luma.len() as f64;
    (min_v, max_v, mean, variance.sqrt())
}

/// Compute the dynamic range quality score.
#[allow(clippy::cast_precision_loss)]
pub fn analyze_dynamic_range(luma: &[u8], config: &DynamicRangeConfig) -> DynamicRangeResult {
    let (min_v, max_v, mean_v, std_v) = luma_statistics(luma);
    let range = max_v.saturating_sub(min_v);

    let total = luma.len() as f64;
    let crushed = if total > 0.0 {
        luma.iter()
            .filter(|&&v| v <= config.black_threshold)
            .count() as f64
            / total
    } else {
        0.0
    };
    let clipped = if total > 0.0 {
        luma.iter()
            .filter(|&&v| v >= config.white_threshold)
            .count() as f64
            / total
    } else {
        0.0
    };

    let contrast_ratio = (max_v as f64 + 1.0) / (min_v as f64 + 1.0);

    // Scoring components
    let range_score = (range as f64 / config.target_range as f64).min(1.0);
    let black_penalty = if crushed > config.max_black_ratio {
        ((crushed - config.max_black_ratio) / config.max_black_ratio).min(1.0)
    } else {
        0.0
    };
    let white_penalty = if clipped > config.max_white_ratio {
        ((clipped - config.max_white_ratio) / config.max_white_ratio).min(1.0)
    } else {
        0.0
    };

    let score = (range_score - 0.3 * black_penalty - 0.3 * white_penalty)
        .max(0.0)
        .min(1.0);

    DynamicRangeResult {
        score,
        min_luma: min_v,
        max_luma: max_v,
        range,
        mean_luma: mean_v,
        std_luma: std_v,
        crushed_black_ratio: crushed,
        clipped_white_ratio: clipped,
        contrast_ratio,
    }
}

/// Quick check: does the frame have a narrow dynamic range?
pub fn is_low_contrast(luma: &[u8], min_range: u8) -> bool {
    let (min_v, max_v, _, _) = luma_statistics(luma);
    max_v.saturating_sub(min_v) < min_range
}

/// Compute the percentage of "legal range" pixels (16..=235).
#[allow(clippy::cast_precision_loss)]
pub fn legal_range_ratio(luma: &[u8]) -> f64 {
    if luma.is_empty() {
        return 0.0;
    }
    let count = luma.iter().filter(|&&v| (16..=235).contains(&v)).count();
    count as f64 / luma.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform(value: u8, len: usize) -> Vec<u8> {
        vec![value; len]
    }

    fn ramp(len: usize) -> Vec<u8> {
        (0..len).map(|i| (i % 256) as u8).collect()
    }

    #[test]
    fn test_histogram_uniform() {
        let data = uniform(128, 100);
        let hist = build_histogram(&data);
        assert_eq!(hist[128], 100);
        assert_eq!(hist[0], 0);
    }

    #[test]
    fn test_luma_statistics_uniform() {
        let data = uniform(100, 50);
        let (min_v, max_v, mean_v, std_v) = luma_statistics(&data);
        assert_eq!(min_v, 100);
        assert_eq!(max_v, 100);
        assert!((mean_v - 100.0).abs() < 1e-9);
        assert!(std_v < 1e-9);
    }

    #[test]
    fn test_luma_statistics_empty() {
        let (min_v, max_v, mean_v, std_v) = luma_statistics(&[]);
        assert_eq!(min_v, 0);
        assert_eq!(max_v, 0);
        assert!((mean_v - 0.0).abs() < 1e-9);
        assert!(std_v < 1e-9);
    }

    #[test]
    fn test_full_range_high_score() {
        let data = ramp(256);
        let cfg = DynamicRangeConfig::default();
        let result = analyze_dynamic_range(&data, &cfg);
        // A full 0-255 ramp has max range, but the default config penalizes pixels
        // below black_threshold=16 and above white_threshold=235. The ramp includes
        // ~6.6% black-crushed and ~8.2% white-clipped pixels, so score is ~0.7 rather
        // than close to 1.0.  Verify the structural fields are correct.
        assert!(
            result.score > 0.5,
            "full ramp should still score well: got {}",
            result.score
        );
        assert_eq!(result.min_luma, 0);
        assert_eq!(result.max_luma, 255);
        assert_eq!(result.range, 255);
    }

    #[test]
    fn test_uniform_low_score() {
        let data = uniform(128, 256);
        let cfg = DynamicRangeConfig::default();
        let result = analyze_dynamic_range(&data, &cfg);
        assert!(result.score < 0.1);
        assert_eq!(result.range, 0);
    }

    #[test]
    fn test_crushed_blacks() {
        // Most pixels at 0
        let mut data = vec![0u8; 100];
        data.extend_from_slice(&[128u8; 10]);
        let cfg = DynamicRangeConfig::default();
        let result = analyze_dynamic_range(&data, &cfg);
        assert!(result.crushed_black_ratio > 0.5);
    }

    #[test]
    fn test_clipped_whites() {
        let mut data = vec![128u8; 10];
        data.extend_from_slice(&[255u8; 100]);
        let cfg = DynamicRangeConfig::default();
        let result = analyze_dynamic_range(&data, &cfg);
        assert!(result.clipped_white_ratio > 0.5);
    }

    #[test]
    fn test_is_low_contrast_true() {
        let data = uniform(128, 100);
        assert!(is_low_contrast(&data, 10));
    }

    #[test]
    fn test_is_low_contrast_false() {
        let data = ramp(256);
        assert!(!is_low_contrast(&data, 10));
    }

    #[test]
    fn test_legal_range_ratio_all_legal() {
        let data: Vec<u8> = (16..=235).collect();
        let ratio = legal_range_ratio(&data);
        assert!((ratio - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_legal_range_ratio_none_legal() {
        let mut data = vec![0u8; 16]; // values 0..15
        data.extend(vec![255u8; 20]); // value 255
        let ratio = legal_range_ratio(&data);
        assert!(ratio < 0.01);
    }

    #[test]
    fn test_contrast_ratio() {
        let mut data = vec![0u8; 50];
        data.extend(vec![255u8; 50]);
        let cfg = DynamicRangeConfig::default();
        let result = analyze_dynamic_range(&data, &cfg);
        // (255+1)/(0+1) = 256
        assert!((result.contrast_ratio - 256.0).abs() < 1e-9);
    }

    #[test]
    fn test_legal_range_empty() {
        let ratio = legal_range_ratio(&[]);
        assert!((ratio - 0.0).abs() < 1e-12);
    }
}
