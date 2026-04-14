#![allow(dead_code)]
//! Temporal flicker detection and scoring.
//!
//! Measures frame-to-frame luminance fluctuations that indicate encoding
//! artifacts such as quantization flicker, rate-control oscillation, or
//! scene-cut induced brightness jumps.

/// Result of flicker analysis over a window of frames.
#[derive(Debug, Clone)]
pub struct FlickerResult {
    /// Overall flicker score (0.0 = no flicker, higher = worse).
    pub score: f64,
    /// Per-frame luminance values used in the analysis.
    pub luminance_values: Vec<f64>,
    /// Per-frame delta values (frame-to-frame difference).
    pub deltas: Vec<f64>,
    /// Maximum absolute delta in the window.
    pub max_delta: f64,
    /// Standard deviation of the deltas.
    pub delta_std: f64,
}

/// Configuration for flicker detection.
#[derive(Debug, Clone)]
pub struct FlickerConfig {
    /// Minimum delta to count as a flicker event.
    pub delta_threshold: f64,
    /// Weight for the standard deviation component in the composite score.
    pub std_weight: f64,
    /// Weight for the max-delta component in the composite score.
    pub max_weight: f64,
}

impl Default for FlickerConfig {
    fn default() -> Self {
        Self {
            delta_threshold: 1.5,
            std_weight: 0.6,
            max_weight: 0.4,
        }
    }
}

/// Compute mean luminance of a luma plane.
#[allow(clippy::cast_precision_loss)]
pub fn mean_luminance(luma: &[u8]) -> f64 {
    if luma.is_empty() {
        return 0.0;
    }
    let sum: u64 = luma.iter().map(|&v| v as u64).sum();
    sum as f64 / luma.len() as f64
}

/// Compute flicker metrics from a sequence of per-frame mean luminance values.
///
/// Returns `None` if fewer than 2 values are provided.
#[allow(clippy::cast_precision_loss)]
pub fn analyze_flicker(luminances: &[f64], config: &FlickerConfig) -> Option<FlickerResult> {
    if luminances.len() < 2 {
        return None;
    }

    let deltas: Vec<f64> = luminances.windows(2).map(|w| w[1] - w[0]).collect();

    let abs_deltas: Vec<f64> = deltas.iter().map(|d| d.abs()).collect();
    let max_delta = abs_deltas.iter().copied().fold(0.0f64, f64::max);

    let mean_delta = abs_deltas.iter().sum::<f64>() / abs_deltas.len() as f64;
    let variance = abs_deltas
        .iter()
        .map(|d| (d - mean_delta).powi(2))
        .sum::<f64>()
        / abs_deltas.len() as f64;
    let delta_std = variance.sqrt();

    // Composite score: weighted combination, clamped to [0, 100]
    let raw_score = config.std_weight * delta_std + config.max_weight * max_delta;
    let score = raw_score.min(100.0).max(0.0);

    Some(FlickerResult {
        score,
        luminance_values: luminances.to_vec(),
        deltas,
        max_delta,
        delta_std,
    })
}

/// Count the number of flicker events (deltas exceeding the threshold).
#[allow(clippy::cast_precision_loss)]
pub fn count_flicker_events(luminances: &[f64], threshold: f64) -> usize {
    if luminances.len() < 2 {
        return 0;
    }
    luminances
        .windows(2)
        .filter(|w| (w[1] - w[0]).abs() > threshold)
        .count()
}

/// Compute a rolling flicker score over a sliding window of `window_size` frames.
///
/// Returns one score per position, starting from position `window_size - 1`.
pub fn rolling_flicker(luminances: &[f64], window_size: usize, config: &FlickerConfig) -> Vec<f64> {
    if luminances.len() < window_size || window_size < 2 {
        return Vec::new();
    }
    luminances
        .windows(window_size)
        .filter_map(|w| analyze_flicker(w, config).map(|r| r.score))
        .collect()
}

/// Normalize luminance values to the range \[0, 1\] based on min/max.
pub fn normalize_luminance(luminances: &[f64]) -> Vec<f64> {
    if luminances.is_empty() {
        return Vec::new();
    }
    let min = luminances.iter().copied().fold(f64::INFINITY, f64::min);
    let max = luminances.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = max - min;
    if range < 1e-12 {
        return vec![0.5; luminances.len()];
    }
    luminances.iter().map(|&v| (v - min) / range).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean_luminance_uniform() {
        let plane = vec![100u8; 64];
        let m = mean_luminance(&plane);
        assert!((m - 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_mean_luminance_empty() {
        assert!((mean_luminance(&[]) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_constant_luminance_no_flicker() {
        let lum = vec![120.0; 10];
        let cfg = FlickerConfig::default();
        let result = analyze_flicker(&lum, &cfg).expect("should succeed in test");
        assert!((result.score - 0.0).abs() < 1e-9);
        assert!((result.max_delta - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_oscillating_luminance_high_flicker() {
        let lum: Vec<f64> = (0..20)
            .map(|i| if i % 2 == 0 { 80.0 } else { 120.0 })
            .collect();
        let cfg = FlickerConfig::default();
        let result = analyze_flicker(&lum, &cfg).expect("should succeed in test");
        assert!(result.score > 10.0);
        assert!((result.max_delta - 40.0).abs() < 1e-9);
    }

    #[test]
    fn test_single_frame_returns_none() {
        let lum = vec![100.0];
        let cfg = FlickerConfig::default();
        assert!(analyze_flicker(&lum, &cfg).is_none());
    }

    #[test]
    fn test_deltas_length() {
        let lum = vec![10.0, 20.0, 30.0, 40.0];
        let cfg = FlickerConfig::default();
        let result = analyze_flicker(&lum, &cfg).expect("should succeed in test");
        assert_eq!(result.deltas.len(), 3);
    }

    #[test]
    fn test_count_flicker_events_none() {
        let lum = vec![100.0, 100.0, 100.0];
        assert_eq!(count_flicker_events(&lum, 1.0), 0);
    }

    #[test]
    fn test_count_flicker_events_some() {
        let lum = vec![100.0, 110.0, 100.0, 110.0];
        assert_eq!(count_flicker_events(&lum, 5.0), 3);
    }

    #[test]
    fn test_rolling_flicker_length() {
        let lum: Vec<f64> = (0..10).map(|i| i as f64 * 5.0).collect();
        let cfg = FlickerConfig::default();
        let rolling = rolling_flicker(&lum, 4, &cfg);
        assert_eq!(rolling.len(), 7); // 10 - 4 + 1
    }

    #[test]
    fn test_rolling_flicker_short_input() {
        let lum = vec![1.0, 2.0];
        let cfg = FlickerConfig::default();
        let rolling = rolling_flicker(&lum, 5, &cfg);
        assert!(rolling.is_empty());
    }

    #[test]
    fn test_normalize_luminance() {
        let lum = vec![0.0, 50.0, 100.0];
        let norm = normalize_luminance(&lum);
        assert!((norm[0] - 0.0).abs() < 1e-9);
        assert!((norm[1] - 0.5).abs() < 1e-9);
        assert!((norm[2] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_normalize_constant() {
        let lum = vec![50.0, 50.0, 50.0];
        let norm = normalize_luminance(&lum);
        assert!((norm[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_score_clamped() {
        // Very large oscillation to trigger clamping
        let lum: Vec<f64> = (0..50)
            .map(|i| if i % 2 == 0 { 0.0 } else { 255.0 })
            .collect();
        let cfg = FlickerConfig::default();
        let result = analyze_flicker(&lum, &cfg).expect("should succeed in test");
        assert!(result.score <= 100.0);
        assert!(result.score >= 0.0);
    }
}
