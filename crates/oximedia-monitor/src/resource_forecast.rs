#![allow(dead_code)]
//! Resource usage forecasting and trend analysis.
//!
//! Provides predictive analytics for CPU, memory, disk, and network resources
//! using linear regression and moving averages to project future utilization.

use std::collections::VecDeque;

/// A single time-stamped resource sample.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResourceSample {
    /// Unix timestamp in seconds.
    pub timestamp: f64,
    /// Resource utilization value (0.0 to 100.0 for percentages, or raw values).
    pub value: f64,
}

impl ResourceSample {
    /// Create a new resource sample.
    #[must_use]
    pub fn new(timestamp: f64, value: f64) -> Self {
        Self { timestamp, value }
    }
}

/// Kind of resource being tracked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResourceKind {
    /// CPU utilization percentage.
    Cpu,
    /// Memory utilization percentage.
    Memory,
    /// Disk utilization percentage.
    Disk,
    /// Network bandwidth utilization in Mbps.
    Network,
    /// GPU utilization percentage.
    Gpu,
}

impl std::fmt::Display for ResourceKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cpu => write!(f, "CPU"),
            Self::Memory => write!(f, "Memory"),
            Self::Disk => write!(f, "Disk"),
            Self::Network => write!(f, "Network"),
            Self::Gpu => write!(f, "GPU"),
        }
    }
}

/// Trend direction of a resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrendDirection {
    /// Resource usage is increasing.
    Rising,
    /// Resource usage is stable.
    Stable,
    /// Resource usage is decreasing.
    Falling,
}

/// Result of a linear regression fit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearFit {
    /// Slope of the fitted line.
    pub slope: f64,
    /// Y-intercept of the fitted line.
    pub intercept: f64,
    /// R-squared goodness of fit (0.0 to 1.0).
    pub r_squared: f64,
}

impl LinearFit {
    /// Predict value at a given timestamp.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn predict(&self, timestamp: f64) -> f64 {
        self.slope * timestamp + self.intercept
    }
}

/// Exponentially-weighted moving average calculator.
#[derive(Debug, Clone)]
pub struct Ewma {
    /// Smoothing factor (0.0 to 1.0).
    alpha: f64,
    /// Current EWMA value.
    current: Option<f64>,
}

impl Ewma {
    /// Create a new EWMA with the given smoothing factor.
    ///
    /// Alpha should be between 0.0 and 1.0. Higher values track recent
    /// data more closely; lower values produce smoother output.
    #[must_use]
    pub fn new(alpha: f64) -> Self {
        let alpha = alpha.clamp(0.0, 1.0);
        Self {
            alpha,
            current: None,
        }
    }

    /// Feed a new sample and return the updated EWMA value.
    #[allow(clippy::cast_precision_loss)]
    pub fn update(&mut self, value: f64) -> f64 {
        let result = match self.current {
            Some(prev) => self.alpha * value + (1.0 - self.alpha) * prev,
            None => value,
        };
        self.current = Some(result);
        result
    }

    /// Get the current EWMA value.
    #[must_use]
    pub fn value(&self) -> Option<f64> {
        self.current
    }

    /// Reset the EWMA state.
    pub fn reset(&mut self) {
        self.current = None;
    }
}

/// Forecast result for a resource.
#[derive(Debug, Clone)]
pub struct ForecastResult {
    /// Resource kind.
    pub kind: ResourceKind,
    /// Predicted value at the forecast horizon.
    pub predicted_value: f64,
    /// Trend direction.
    pub trend: TrendDirection,
    /// Confidence score (0.0 to 1.0) based on R-squared.
    pub confidence: f64,
    /// Estimated time until threshold breach (in seconds), or `None` if not applicable.
    pub time_to_threshold: Option<f64>,
}

/// Resource forecaster that accumulates samples and predicts future usage.
#[derive(Debug, Clone)]
pub struct ResourceForecaster {
    /// Kind of resource.
    kind: ResourceKind,
    /// Rolling sample window.
    samples: VecDeque<ResourceSample>,
    /// Maximum number of samples to retain.
    max_samples: usize,
    /// EWMA tracker.
    ewma: Ewma,
    /// Threshold at which to raise an alert.
    alert_threshold: f64,
    /// Minimum slope magnitude to classify as rising/falling.
    trend_sensitivity: f64,
}

impl ResourceForecaster {
    /// Create a new forecaster for the given resource kind.
    #[must_use]
    pub fn new(kind: ResourceKind) -> Self {
        Self {
            kind,
            samples: VecDeque::with_capacity(1024),
            max_samples: 1024,
            ewma: Ewma::new(0.3),
            alert_threshold: 90.0,
            trend_sensitivity: 0.001,
        }
    }

    /// Set the maximum number of retained samples.
    #[must_use]
    pub fn with_max_samples(mut self, n: usize) -> Self {
        self.max_samples = n.max(2);
        self
    }

    /// Set the alert threshold.
    #[must_use]
    pub fn with_alert_threshold(mut self, threshold: f64) -> Self {
        self.alert_threshold = threshold;
        self
    }

    /// Set the EWMA alpha.
    #[must_use]
    pub fn with_ewma_alpha(mut self, alpha: f64) -> Self {
        self.ewma = Ewma::new(alpha);
        self
    }

    /// Set the trend sensitivity.
    #[must_use]
    pub fn with_trend_sensitivity(mut self, sensitivity: f64) -> Self {
        self.trend_sensitivity = sensitivity.abs();
        self
    }

    /// Add a new sample.
    pub fn add_sample(&mut self, sample: ResourceSample) {
        self.ewma.update(sample.value);
        self.samples.push_back(sample);
        while self.samples.len() > self.max_samples {
            self.samples.pop_front();
        }
    }

    /// Return the number of accumulated samples.
    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Compute a linear regression over all current samples.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn fit_linear(&self) -> Option<LinearFit> {
        let n = self.samples.len();
        if n < 2 {
            return None;
        }
        let n_f = n as f64;

        let mut sum_x = 0.0_f64;
        let mut sum_y = 0.0_f64;
        let mut sum_xy = 0.0_f64;
        let mut sum_xx = 0.0_f64;

        for s in &self.samples {
            sum_x += s.timestamp;
            sum_y += s.value;
            sum_xy += s.timestamp * s.value;
            sum_xx += s.timestamp * s.timestamp;
        }

        let denom = n_f * sum_xx - sum_x * sum_x;
        if denom.abs() < 1e-15 {
            return None;
        }

        let slope = (n_f * sum_xy - sum_x * sum_y) / denom;
        let intercept = (sum_y - slope * sum_x) / n_f;

        // Compute R-squared
        let mean_y = sum_y / n_f;
        let mut ss_res = 0.0_f64;
        let mut ss_tot = 0.0_f64;
        for s in &self.samples {
            let predicted = slope * s.timestamp + intercept;
            ss_res += (s.value - predicted).powi(2);
            ss_tot += (s.value - mean_y).powi(2);
        }

        let r_squared = if ss_tot.abs() < 1e-15 {
            1.0
        } else {
            1.0 - ss_res / ss_tot
        };

        Some(LinearFit {
            slope,
            intercept,
            r_squared,
        })
    }

    /// Forecast resource usage at a future timestamp.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn forecast(&self, future_timestamp: f64) -> Option<ForecastResult> {
        let fit = self.fit_linear()?;
        let predicted = fit.predict(future_timestamp);

        let trend = if fit.slope > self.trend_sensitivity {
            TrendDirection::Rising
        } else if fit.slope < -self.trend_sensitivity {
            TrendDirection::Falling
        } else {
            TrendDirection::Stable
        };

        let time_to_threshold = if fit.slope > 0.0 {
            let last_ts = self.samples.back()?.timestamp;
            let last_val = fit.predict(last_ts);
            if last_val < self.alert_threshold {
                let remaining = (self.alert_threshold - last_val) / fit.slope;
                Some(remaining)
            } else {
                Some(0.0)
            }
        } else {
            None
        };

        Some(ForecastResult {
            kind: self.kind,
            predicted_value: predicted,
            trend,
            confidence: fit.r_squared.clamp(0.0, 1.0),
            time_to_threshold,
        })
    }

    /// Compute simple moving average over the last `window` samples.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn moving_average(&self, window: usize) -> Option<f64> {
        if self.samples.is_empty() || window == 0 {
            return None;
        }
        let w = window.min(self.samples.len());
        let start = self.samples.len() - w;
        let sum: f64 = self.samples.iter().skip(start).map(|s| s.value).sum();
        Some(sum / w as f64)
    }

    /// Get the current EWMA value.
    #[must_use]
    pub fn ewma_value(&self) -> Option<f64> {
        self.ewma.value()
    }

    /// Get the resource kind.
    #[must_use]
    pub fn kind(&self) -> ResourceKind {
        self.kind
    }

    /// Reset all accumulated data.
    pub fn reset(&mut self) {
        self.samples.clear();
        self.ewma.reset();
    }
}

/// Multi-resource forecast manager that tracks several resource types.
#[derive(Debug)]
pub struct ForecastManager {
    /// Individual forecasters keyed by resource kind.
    forecasters: Vec<ResourceForecaster>,
}

impl ForecastManager {
    /// Create a new manager with default forecasters for all resource kinds.
    #[must_use]
    pub fn new() -> Self {
        let kinds = [
            ResourceKind::Cpu,
            ResourceKind::Memory,
            ResourceKind::Disk,
            ResourceKind::Network,
            ResourceKind::Gpu,
        ];
        let forecasters = kinds.iter().map(|&k| ResourceForecaster::new(k)).collect();
        Self { forecasters }
    }

    /// Get a mutable reference to the forecaster for a given kind.
    pub fn get_mut(&mut self, kind: ResourceKind) -> Option<&mut ResourceForecaster> {
        self.forecasters.iter_mut().find(|f| f.kind() == kind)
    }

    /// Get a reference to the forecaster for a given kind.
    #[must_use]
    pub fn get(&self, kind: ResourceKind) -> Option<&ResourceForecaster> {
        self.forecasters.iter().find(|f| f.kind() == kind)
    }

    /// Add a sample to the appropriate forecaster.
    pub fn add_sample(&mut self, kind: ResourceKind, sample: ResourceSample) {
        if let Some(f) = self.get_mut(kind) {
            f.add_sample(sample);
        }
    }

    /// Run forecasts for all resource kinds at the given future timestamp.
    #[must_use]
    pub fn forecast_all(&self, future_timestamp: f64) -> Vec<ForecastResult> {
        self.forecasters
            .iter()
            .filter_map(|f| f.forecast(future_timestamp))
            .collect()
    }

    /// Return the number of tracked resource kinds.
    #[must_use]
    pub fn resource_count(&self) -> usize {
        self.forecasters.len()
    }
}

impl Default for ForecastManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_sample_creation() {
        let s = ResourceSample::new(1000.0, 45.0);
        assert!((s.timestamp - 1000.0).abs() < f64::EPSILON);
        assert!((s.value - 45.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_resource_kind_display() {
        assert_eq!(ResourceKind::Cpu.to_string(), "CPU");
        assert_eq!(ResourceKind::Memory.to_string(), "Memory");
        assert_eq!(ResourceKind::Disk.to_string(), "Disk");
        assert_eq!(ResourceKind::Network.to_string(), "Network");
        assert_eq!(ResourceKind::Gpu.to_string(), "GPU");
    }

    #[test]
    fn test_ewma_single_value() {
        let mut ewma = Ewma::new(0.5);
        let v = ewma.update(10.0);
        assert!((v - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_ewma_convergence() {
        let mut ewma = Ewma::new(0.3);
        for _ in 0..100 {
            ewma.update(50.0);
        }
        let v = ewma.value().expect("value should succeed");
        assert!((v - 50.0).abs() < 0.001);
    }

    #[test]
    fn test_ewma_reset() {
        let mut ewma = Ewma::new(0.5);
        ewma.update(100.0);
        assert!(ewma.value().is_some());
        ewma.reset();
        assert!(ewma.value().is_none());
    }

    #[test]
    fn test_linear_fit_constant() {
        let mut forecaster = ResourceForecaster::new(ResourceKind::Cpu);
        for i in 0..10 {
            forecaster.add_sample(ResourceSample::new(i as f64, 50.0));
        }
        let fit = forecaster.fit_linear().expect("fit_linear should succeed");
        assert!(fit.slope.abs() < 1e-10);
        assert!((fit.intercept - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_linear_fit_rising() {
        let mut forecaster = ResourceForecaster::new(ResourceKind::Memory);
        for i in 0..10 {
            forecaster.add_sample(ResourceSample::new(i as f64, i as f64 * 5.0));
        }
        let fit = forecaster.fit_linear().expect("fit_linear should succeed");
        assert!((fit.slope - 5.0).abs() < 1e-10);
        assert!(fit.r_squared > 0.99);
    }

    #[test]
    fn test_forecast_trend_rising() {
        let mut forecaster = ResourceForecaster::new(ResourceKind::Disk).with_alert_threshold(90.0);
        for i in 0..20 {
            forecaster.add_sample(ResourceSample::new(i as f64, 40.0 + i as f64 * 2.0));
        }
        let result = forecaster.forecast(30.0).expect("forecast should succeed");
        assert_eq!(result.trend, TrendDirection::Rising);
        assert!(result.time_to_threshold.is_some());
    }

    #[test]
    fn test_forecast_trend_falling() {
        let mut forecaster = ResourceForecaster::new(ResourceKind::Network);
        for i in 0..20 {
            forecaster.add_sample(ResourceSample::new(i as f64, 80.0 - i as f64 * 2.0));
        }
        let result = forecaster.forecast(30.0).expect("forecast should succeed");
        assert_eq!(result.trend, TrendDirection::Falling);
        assert!(result.time_to_threshold.is_none());
    }

    #[test]
    fn test_moving_average() {
        let mut forecaster = ResourceForecaster::new(ResourceKind::Cpu);
        for i in 1..=10 {
            forecaster.add_sample(ResourceSample::new(i as f64, i as f64));
        }
        // Last 5 values: 6,7,8,9,10 → avg = 8.0
        let avg = forecaster
            .moving_average(5)
            .expect("moving_average should succeed");
        assert!((avg - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_moving_average_empty() {
        let forecaster = ResourceForecaster::new(ResourceKind::Cpu);
        assert!(forecaster.moving_average(5).is_none());
    }

    #[test]
    fn test_forecast_manager_creation() {
        let manager = ForecastManager::new();
        assert_eq!(manager.resource_count(), 5);
        assert!(manager.get(ResourceKind::Cpu).is_some());
        assert!(manager.get(ResourceKind::Gpu).is_some());
    }

    #[test]
    fn test_forecast_manager_add_sample() {
        let mut manager = ForecastManager::new();
        for i in 0..10 {
            manager.add_sample(
                ResourceKind::Cpu,
                ResourceSample::new(i as f64, 30.0 + i as f64),
            );
        }
        let cpu = manager.get(ResourceKind::Cpu).expect("failed to get value");
        assert_eq!(cpu.sample_count(), 10);
    }

    #[test]
    fn test_forecast_manager_forecast_all() {
        let mut manager = ForecastManager::new();
        for i in 0..10 {
            let t = i as f64;
            manager.add_sample(ResourceKind::Cpu, ResourceSample::new(t, 30.0 + t));
            manager.add_sample(ResourceKind::Memory, ResourceSample::new(t, 50.0 + t * 0.5));
        }
        let results = manager.forecast_all(20.0);
        assert!(results.len() >= 2);
    }

    #[test]
    fn test_forecaster_reset() {
        let mut forecaster = ResourceForecaster::new(ResourceKind::Gpu);
        for i in 0..5 {
            forecaster.add_sample(ResourceSample::new(i as f64, 10.0));
        }
        assert_eq!(forecaster.sample_count(), 5);
        forecaster.reset();
        assert_eq!(forecaster.sample_count(), 0);
        assert!(forecaster.ewma_value().is_none());
    }

    #[test]
    fn test_max_samples_cap() {
        let mut forecaster = ResourceForecaster::new(ResourceKind::Cpu).with_max_samples(5);
        for i in 0..20 {
            forecaster.add_sample(ResourceSample::new(i as f64, i as f64));
        }
        assert_eq!(forecaster.sample_count(), 5);
    }

    #[test]
    fn test_linear_fit_predict() {
        let fit = LinearFit {
            slope: 2.0,
            intercept: 10.0,
            r_squared: 0.99,
        };
        assert!((fit.predict(5.0) - 20.0).abs() < f64::EPSILON);
        assert!((fit.predict(0.0) - 10.0).abs() < f64::EPSILON);
    }
}
