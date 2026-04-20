#![allow(dead_code)]
//! Recommendation score calibration.
//!
//! Ensures that recommendation scores are well-calibrated, meaning that a score
//! of 0.8 should correspond to roughly 80% likelihood the user will engage.
//! Supports Platt scaling, isotonic regression approximation, and temperature
//! scaling methods.

use std::collections::BTreeMap;

/// Calibration method to apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalibrationMethod {
    /// Platt scaling (logistic regression on scores).
    Platt,
    /// Isotonic regression (monotonic piecewise-linear mapping).
    Isotonic,
    /// Temperature scaling (divide logits by temperature).
    Temperature,
    /// Identity (no calibration).
    Identity,
}

/// A data point for calibration: predicted score and actual outcome.
#[derive(Debug, Clone, Copy)]
pub struct CalibrationSample {
    /// Predicted score (0.0 to 1.0).
    pub predicted: f64,
    /// Actual outcome (1.0 = positive, 0.0 = negative).
    pub actual: f64,
}

impl CalibrationSample {
    /// Create a new calibration sample.
    #[must_use]
    pub fn new(predicted: f64, actual: f64) -> Self {
        Self { predicted, actual }
    }
}

/// Platt scaling parameters (logistic sigmoid: 1 / (1 + exp(a*x + b))).
#[derive(Debug, Clone, Copy)]
pub struct PlattParams {
    /// Slope parameter.
    pub a: f64,
    /// Intercept parameter.
    pub b: f64,
}

impl PlattParams {
    /// Create new Platt parameters.
    #[must_use]
    pub fn new(a: f64, b: f64) -> Self {
        Self { a, b }
    }

    /// Apply Platt scaling to a raw score.
    #[must_use]
    pub fn calibrate(&self, score: f64) -> f64 {
        let exponent = self.a * score + self.b;
        1.0 / (1.0 + exponent.exp())
    }
}

impl Default for PlattParams {
    fn default() -> Self {
        Self { a: -1.0, b: 0.0 }
    }
}

/// Isotonic calibration mapping using piecewise-linear interpolation.
#[derive(Debug, Clone)]
pub struct IsotonicMap {
    /// Sorted breakpoints: (`predicted_score`, `calibrated_score`).
    breakpoints: Vec<(f64, f64)>,
}

impl IsotonicMap {
    /// Create from sorted breakpoints.
    #[must_use]
    pub fn new(breakpoints: Vec<(f64, f64)>) -> Self {
        Self { breakpoints }
    }

    /// Build from calibration samples using pool adjacent violators.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn from_samples(samples: &[CalibrationSample]) -> Self {
        if samples.is_empty() {
            return Self {
                breakpoints: vec![(0.0, 0.0), (1.0, 1.0)],
            };
        }

        let mut sorted: Vec<CalibrationSample> = samples.to_vec();
        sorted.sort_by(|a, b| {
            a.predicted
                .partial_cmp(&b.predicted)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Pool adjacent violators algorithm
        let mut blocks: Vec<(f64, f64, usize)> =
            sorted.iter().map(|s| (s.predicted, s.actual, 1)).collect();

        let mut i = 0;
        while i < blocks.len().saturating_sub(1) {
            if blocks[i].1 > blocks[i + 1].1 {
                let total_count = blocks[i].2 + blocks[i + 1].2;
                let merged_actual = (blocks[i].1 * blocks[i].2 as f64
                    + blocks[i + 1].1 * blocks[i + 1].2 as f64)
                    / total_count as f64;
                let merged_pred = (blocks[i].0 * blocks[i].2 as f64
                    + blocks[i + 1].0 * blocks[i + 1].2 as f64)
                    / total_count as f64;
                blocks[i] = (merged_pred, merged_actual, total_count);
                blocks.remove(i + 1);
                i = i.saturating_sub(1);
            } else {
                i += 1;
            }
        }

        let breakpoints: Vec<(f64, f64)> = blocks.iter().map(|(p, a, _)| (*p, *a)).collect();
        Self { breakpoints }
    }

    /// Calibrate a score using linear interpolation between breakpoints.
    #[must_use]
    pub fn calibrate(&self, score: f64) -> f64 {
        if self.breakpoints.is_empty() {
            return score;
        }
        if score <= self.breakpoints[0].0 {
            return self.breakpoints[0].1;
        }
        if score >= self.breakpoints[self.breakpoints.len() - 1].0 {
            return self.breakpoints[self.breakpoints.len() - 1].1;
        }
        for window in self.breakpoints.windows(2) {
            let (x0, y0) = window[0];
            let (x1, y1) = window[1];
            if score >= x0 && score <= x1 {
                if (x1 - x0).abs() < 1e-12 {
                    return y0;
                }
                let t = (score - x0) / (x1 - x0);
                return y0 + t * (y1 - y0);
            }
        }
        score
    }

    /// Return the number of breakpoints.
    #[must_use]
    pub fn breakpoint_count(&self) -> usize {
        self.breakpoints.len()
    }
}

/// Temperature scaling parameters.
#[derive(Debug, Clone, Copy)]
pub struct TemperatureParams {
    /// Temperature value (> 0). Higher = softer predictions.
    pub temperature: f64,
}

impl TemperatureParams {
    /// Create new temperature parameters.
    #[must_use]
    pub fn new(temperature: f64) -> Self {
        Self {
            temperature: temperature.max(0.01),
        }
    }

    /// Calibrate a score using temperature scaling.
    /// Applies logit -> divide by temperature -> sigmoid.
    #[must_use]
    pub fn calibrate(&self, score: f64) -> f64 {
        let clamped = score.clamp(1e-7, 1.0 - 1e-7);
        let logit = (clamped / (1.0 - clamped)).ln();
        let scaled = logit / self.temperature;
        1.0 / (1.0 + (-scaled).exp())
    }
}

impl Default for TemperatureParams {
    fn default() -> Self {
        Self { temperature: 1.0 }
    }
}

/// Calibration reliability metrics.
#[derive(Debug, Clone)]
pub struct CalibrationMetrics {
    /// Expected calibration error (lower is better).
    pub ece: f64,
    /// Maximum calibration error.
    pub mce: f64,
    /// Number of bins used.
    pub num_bins: usize,
    /// Per-bin accuracy and confidence.
    pub bin_stats: Vec<BinStat>,
}

/// Statistics for a single calibration bin.
#[derive(Debug, Clone)]
pub struct BinStat {
    /// Bin index.
    pub bin: usize,
    /// Average predicted score in this bin.
    pub avg_predicted: f64,
    /// Actual positive rate in this bin.
    pub actual_rate: f64,
    /// Number of samples in this bin.
    pub count: usize,
}

/// Compute calibration metrics (ECE, MCE) for a set of samples.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn compute_calibration_metrics(
    samples: &[CalibrationSample],
    num_bins: usize,
) -> CalibrationMetrics {
    let num_bins = num_bins.max(1);
    let bin_width = 1.0 / num_bins as f64;
    let mut bins: BTreeMap<usize, Vec<&CalibrationSample>> = BTreeMap::new();

    for sample in samples {
        let bin_idx = ((sample.predicted / bin_width) as usize).min(num_bins - 1);
        bins.entry(bin_idx).or_default().push(sample);
    }

    let total = samples.len() as f64;
    let mut ece = 0.0;
    let mut mce = 0.0_f64;
    let mut bin_stats = Vec::new();

    for bin_idx in 0..num_bins {
        let entries = bins.get(&bin_idx);
        let count = entries.map_or(0, std::vec::Vec::len);
        if count == 0 {
            bin_stats.push(BinStat {
                bin: bin_idx,
                avg_predicted: 0.0,
                actual_rate: 0.0,
                count: 0,
            });
            continue;
        }
        // Safe: count > 0 was checked above, so entries is Some.
        let items = match entries {
            Some(v) => v,
            None => continue,
        };
        let avg_pred: f64 = items.iter().map(|s| s.predicted).sum::<f64>() / count as f64;
        let actual_rate: f64 = items.iter().map(|s| s.actual).sum::<f64>() / count as f64;
        let gap = (avg_pred - actual_rate).abs();
        ece += (count as f64 / total) * gap;
        mce = mce.max(gap);
        bin_stats.push(BinStat {
            bin: bin_idx,
            avg_predicted: avg_pred,
            actual_rate,
            count,
        });
    }

    CalibrationMetrics {
        ece,
        mce,
        num_bins,
        bin_stats,
    }
}

/// Unified calibrator that applies the selected method.
pub struct ScoreCalibrator {
    /// Calibration method.
    method: CalibrationMethod,
    /// Platt parameters (used if method is Platt).
    platt: PlattParams,
    /// Isotonic map (used if method is Isotonic).
    isotonic: IsotonicMap,
    /// Temperature parameters (used if method is Temperature).
    temperature: TemperatureParams,
}

impl ScoreCalibrator {
    /// Create a new calibrator with the given method.
    #[must_use]
    pub fn new(method: CalibrationMethod) -> Self {
        Self {
            method,
            platt: PlattParams::default(),
            isotonic: IsotonicMap::new(vec![(0.0, 0.0), (1.0, 1.0)]),
            temperature: TemperatureParams::default(),
        }
    }

    /// Set Platt parameters.
    #[must_use]
    pub fn with_platt(mut self, params: PlattParams) -> Self {
        self.platt = params;
        self
    }

    /// Set isotonic map.
    #[must_use]
    pub fn with_isotonic(mut self, map: IsotonicMap) -> Self {
        self.isotonic = map;
        self
    }

    /// Set temperature parameters.
    #[must_use]
    pub fn with_temperature(mut self, params: TemperatureParams) -> Self {
        self.temperature = params;
        self
    }

    /// Calibrate a single score.
    #[must_use]
    pub fn calibrate(&self, score: f64) -> f64 {
        match self.method {
            CalibrationMethod::Platt => self.platt.calibrate(score),
            CalibrationMethod::Isotonic => self.isotonic.calibrate(score),
            CalibrationMethod::Temperature => self.temperature.calibrate(score),
            CalibrationMethod::Identity => score,
        }
    }

    /// Calibrate a batch of scores.
    #[must_use]
    pub fn calibrate_batch(&self, scores: &[f64]) -> Vec<f64> {
        scores.iter().map(|&s| self.calibrate(s)).collect()
    }

    /// Return the active calibration method.
    #[must_use]
    pub fn method(&self) -> CalibrationMethod {
        self.method
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_platt_params_default() {
        let params = PlattParams::default();
        // With a=-1, b=0, calibrate(0) should be sigmoid(0) = 0.5
        let result = params.calibrate(0.0);
        assert!((result - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_platt_calibrate_monotonic() {
        let params = PlattParams::new(-2.0, 1.0);
        let low = params.calibrate(0.2);
        let high = params.calibrate(0.8);
        // With negative a, higher input => lower exponent => higher output
        assert!(high > low);
    }

    #[test]
    fn test_temperature_identity_at_one() {
        let params = TemperatureParams::new(1.0);
        let result = params.calibrate(0.7);
        assert!((result - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_temperature_higher_makes_softer() {
        let soft = TemperatureParams::new(2.0);
        let sharp = TemperatureParams::new(0.5);
        // High score: soft should pull it closer to 0.5, sharp should push it further
        let soft_val = soft.calibrate(0.9);
        let sharp_val = sharp.calibrate(0.9);
        assert!(soft_val < sharp_val);
    }

    #[test]
    fn test_isotonic_map_endpoints() {
        let map = IsotonicMap::new(vec![(0.0, 0.0), (1.0, 1.0)]);
        assert!((map.calibrate(0.0)).abs() < 1e-10);
        assert!((map.calibrate(1.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_isotonic_map_interpolation() {
        let map = IsotonicMap::new(vec![(0.0, 0.0), (0.5, 0.8), (1.0, 1.0)]);
        let result = map.calibrate(0.25);
        // Linear interpolation between (0.0, 0.0) and (0.5, 0.8) at t=0.5 => 0.4
        assert!((result - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_isotonic_from_samples() {
        let samples = vec![
            CalibrationSample::new(0.1, 0.0),
            CalibrationSample::new(0.3, 0.0),
            CalibrationSample::new(0.5, 1.0),
            CalibrationSample::new(0.7, 1.0),
            CalibrationSample::new(0.9, 1.0),
        ];
        let map = IsotonicMap::from_samples(&samples);
        assert!(map.breakpoint_count() > 0);
        // Low scores should map lower than high scores
        let low = map.calibrate(0.1);
        let high = map.calibrate(0.9);
        assert!(high >= low);
    }

    #[test]
    fn test_isotonic_from_empty() {
        let map = IsotonicMap::from_samples(&[]);
        assert_eq!(map.breakpoint_count(), 2);
    }

    #[test]
    fn test_calibration_metrics_perfect() {
        let samples = vec![
            CalibrationSample::new(0.1, 0.0),
            CalibrationSample::new(0.9, 1.0),
        ];
        let metrics = compute_calibration_metrics(&samples, 10);
        assert!(metrics.ece < 0.2);
    }

    #[test]
    fn test_calibration_metrics_bins() {
        let samples = vec![
            CalibrationSample::new(0.15, 0.0),
            CalibrationSample::new(0.85, 1.0),
        ];
        let metrics = compute_calibration_metrics(&samples, 5);
        assert_eq!(metrics.num_bins, 5);
        assert_eq!(metrics.bin_stats.len(), 5);
    }

    #[test]
    fn test_score_calibrator_identity() {
        let calibrator = ScoreCalibrator::new(CalibrationMethod::Identity);
        assert!((calibrator.calibrate(0.42) - 0.42).abs() < 1e-10);
    }

    #[test]
    fn test_score_calibrator_batch() {
        let calibrator = ScoreCalibrator::new(CalibrationMethod::Identity);
        let results = calibrator.calibrate_batch(&[0.1, 0.5, 0.9]);
        assert_eq!(results.len(), 3);
        assert!((results[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_score_calibrator_method() {
        let calibrator = ScoreCalibrator::new(CalibrationMethod::Platt);
        assert_eq!(calibrator.method(), CalibrationMethod::Platt);
    }

    #[test]
    fn test_score_calibrator_with_platt() {
        let calibrator =
            ScoreCalibrator::new(CalibrationMethod::Platt).with_platt(PlattParams::new(-1.0, 0.0));
        let result = calibrator.calibrate(0.0);
        assert!((result - 0.5).abs() < 1e-6);
    }
}
