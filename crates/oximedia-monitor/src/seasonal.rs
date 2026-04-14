//! Seasonal decomposition for periodic anomaly detection in media pipeline metrics.
//!
//! Implements additive decomposition: `Y = Trend + Seasonal + Residual`.
//!
//! Typical usage is detecting daily throughput cycles (period = 24 for hourly samples)
//! or other repeating patterns in time-series data.

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// SeasonalDecomposition
// ---------------------------------------------------------------------------

/// Additive seasonal decomposition: `Y = Trend + Seasonal + Residual`.
#[derive(Debug, Clone)]
pub struct SeasonalDecomposition {
    /// Seasonal period (e.g., 24 for hourly data with daily patterns).
    pub period: usize,
    /// Centered moving-average trend component (length = `original.len()`).
    ///
    /// Values near the series boundaries where the CMA window extends outside
    /// the data are set to the series mean as a fallback.
    pub trend: Vec<f32>,
    /// Seasonal pattern component (length = `period`, repeating).
    pub seasonal: Vec<f32>,
    /// Residual after removing trend and seasonal (length = `original.len()`).
    pub residual: Vec<f32>,
    /// The original input data.
    pub original: Vec<f32>,
}

impl SeasonalDecomposition {
    /// Decompose a time series with the given `period`.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `data.len() < 2 * period` (insufficient data).
    pub fn decompose(data: &[f32], period: usize) -> Result<Self, String> {
        if period < 2 {
            return Err(format!("period must be >= 2, got {period}"));
        }
        if data.len() < 2 * period {
            return Err(format!(
                "data length {} is less than 2 * period ({})",
                data.len(),
                2 * period
            ));
        }

        let n = data.len();
        let half = period / 2;

        // 1. Centered moving average as trend.
        let series_mean: f32 = data.iter().sum::<f32>() / n as f32;
        let mut trend = vec![series_mean; n];
        for i in 0..n {
            let lo = i.saturating_sub(half);
            let hi = (i + half + 1).min(n);
            // Only compute CMA where the full window is available.
            if lo == i.saturating_sub(half) && (i + half) < n {
                let window = &data[lo..hi];
                trend[i] = window.iter().sum::<f32>() / window.len() as f32;
            }
        }

        // 2. De-trend: dt[i] = data[i] - trend[i]
        let detrended: Vec<f32> = data.iter().zip(trend.iter()).map(|(y, t)| y - t).collect();

        // 3. Seasonal: for each phase p (0..period), average all de-trended
        //    values at positions p, p+period, p+2*period, …
        let mut seasonal = vec![0.0f32; period];
        for p in 0..period {
            let vals: Vec<f32> = (0..)
                .map(|k| p + k * period)
                .take_while(|&idx| idx < n)
                .map(|idx| detrended[idx])
                .collect();
            if !vals.is_empty() {
                seasonal[p] = vals.iter().sum::<f32>() / vals.len() as f32;
            }
        }

        // 4. Residual: residual[i] = data[i] - trend[i] - seasonal[i % period]
        let residual: Vec<f32> = (0..n)
            .map(|i| data[i] - trend[i] - seasonal[i % period])
            .collect();

        Ok(Self {
            period,
            trend,
            seasonal,
            residual,
            original: data.to_vec(),
        })
    }

    /// Return the seasonal pattern (length = `period`).
    #[must_use]
    pub fn seasonal_pattern(&self) -> &[f32] {
        &self.seasonal
    }

    /// Return the trend value at index `i`, or `None` if out of bounds.
    #[must_use]
    pub fn trend_at(&self, i: usize) -> Option<f32> {
        self.trend.get(i).copied()
    }

    /// Return the residual value at index `i`, or `None` if out of bounds.
    #[must_use]
    pub fn residual_at(&self, i: usize) -> Option<f32> {
        self.residual.get(i).copied()
    }

    /// Detect anomalies: indices where `|residual - mean| > threshold_sigma * std`.
    #[must_use]
    pub fn detect_anomalies(&self, threshold_sigma: f32) -> Vec<usize> {
        let std = running_std(&self.residual);
        if std < f32::EPSILON {
            return Vec::new();
        }
        let mean = self.residual.iter().sum::<f32>() / self.residual.len() as f32;
        self.residual
            .iter()
            .enumerate()
            .filter(|(_, &r)| (r - mean).abs() > threshold_sigma * std)
            .map(|(i, _)| i)
            .collect()
    }

    /// Forecast the next `n` values by continuing the seasonal pattern.
    ///
    /// The trend extension uses the mean of the last few valid trend values.
    #[must_use]
    pub fn forecast(&self, n: usize) -> Vec<f32> {
        if n == 0 {
            return Vec::new();
        }

        // Use the mean of the last `period` trend values as the "last trend".
        let tail_len = self.period.min(self.trend.len());
        let tail = &self.trend[self.trend.len() - tail_len..];
        let last_trend = tail.iter().sum::<f32>() / tail_len as f32;

        let data_len = self.original.len();
        (0..n)
            .map(|i| last_trend + self.seasonal[(data_len + i) % self.period])
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute the standard deviation of `data` (population std, mean-centred).
///
/// Returns `0.0` for empty or single-element slices.
#[must_use]
pub fn running_std(data: &[f32]) -> f32 {
    let n = data.len();
    if n < 2 {
        return 0.0;
    }
    let mean = data.iter().sum::<f32>() / n as f32;
    let variance = data.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / n as f32;
    variance.sqrt()
}

/// Decompose hourly data assuming a period of 24 (daily pattern).
///
/// # Errors
///
/// Returns `Err` if `data.len() < 48`.
pub fn detect_hourly_pattern(data: &[f32]) -> Result<SeasonalDecomposition, String> {
    SeasonalDecomposition::decompose(data, 24)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    /// Generate a pure sinusoidal series of `n` points with `period`.
    fn sinusoid(n: usize, period: usize, amplitude: f32) -> Vec<f32> {
        (0..n)
            .map(|i| amplitude * (TAU * i as f32 / period as f32).sin())
            .collect()
    }

    #[test]
    fn test_decompose_sinusoid_seasonal_length() {
        let data = sinusoid(96, 24, 10.0);
        let dec = SeasonalDecomposition::decompose(&data, 24).expect("ok");
        assert_eq!(dec.seasonal_pattern().len(), 24);
    }

    #[test]
    fn test_decompose_sinusoid_residual_small() {
        // A pure sinusoid should have very small residuals after decomposition.
        let data = sinusoid(96, 24, 10.0);
        let dec = SeasonalDecomposition::decompose(&data, 24).expect("ok");
        let max_residual = dec
            .residual
            .iter()
            .map(|r| r.abs())
            .fold(0.0f32, f32::max);
        // Residuals won't be perfectly zero because CMA boundary effects.
        assert!(
            max_residual < 15.0,
            "max residual {max_residual} too large for pure sinusoid"
        );
    }

    #[test]
    fn test_decompose_period_too_large_returns_err() {
        let data = vec![1.0f32; 10];
        let result = SeasonalDecomposition::decompose(&data, 8); // 2*8 = 16 > 10
        assert!(result.is_err(), "should fail when data < 2*period");
    }

    #[test]
    fn test_seasonal_pattern_length_equals_period() {
        let data: Vec<f32> = (0..48).map(|i| i as f32).collect();
        let dec = SeasonalDecomposition::decompose(&data, 12).expect("ok");
        assert_eq!(dec.seasonal_pattern().len(), 12);
    }

    #[test]
    fn test_trend_at_and_residual_at_valid_indices() {
        let data: Vec<f32> = (0..48).map(|i| i as f32).collect();
        let dec = SeasonalDecomposition::decompose(&data, 12).expect("ok");
        assert!(dec.trend_at(0).is_some());
        assert!(dec.trend_at(47).is_some());
        assert!(dec.trend_at(48).is_none());
        assert!(dec.residual_at(0).is_some());
    }

    #[test]
    fn test_detect_anomalies_finds_spike() {
        // Create a near-zero series and insert a large spike.
        let mut data = sinusoid(96, 24, 1.0);
        data[50] += 100.0; // huge spike
        let dec = SeasonalDecomposition::decompose(&data, 24).expect("ok");
        let anomalies = dec.detect_anomalies(2.0);
        assert!(
            anomalies.contains(&50),
            "spike at index 50 should be detected as anomaly"
        );
    }

    #[test]
    fn test_detect_anomalies_empty_for_flat_residuals() {
        // Completely flat data → zero residual std → no anomalies.
        let data = vec![5.0f32; 50];
        let dec = SeasonalDecomposition::decompose(&data, 10).expect("ok");
        let anomalies = dec.detect_anomalies(3.0);
        assert!(anomalies.is_empty());
    }

    #[test]
    fn test_forecast_returns_n_values() {
        let data = sinusoid(96, 24, 5.0);
        let dec = SeasonalDecomposition::decompose(&data, 24).expect("ok");
        let forecast = dec.forecast(12);
        assert_eq!(forecast.len(), 12);
    }

    #[test]
    fn test_running_std_known_values() {
        // Population std of [2, 4, 4, 4, 5, 5, 7, 9] = 2.0
        let data = [2.0f32, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let std = running_std(&data);
        assert!((std - 2.0).abs() < 0.001, "expected std ≈ 2.0, got {std}");
    }

    #[test]
    fn test_running_std_empty() {
        assert_eq!(running_std(&[]), 0.0);
    }

    #[test]
    fn test_detect_hourly_pattern_period_24() {
        let data = sinusoid(72, 24, 3.0);
        let dec = detect_hourly_pattern(&data).expect("ok");
        assert_eq!(dec.period, 24);
    }

    #[test]
    fn test_detect_hourly_pattern_too_short_fails() {
        let data = vec![1.0f32; 40]; // < 48
        assert!(detect_hourly_pattern(&data).is_err());
    }
}
