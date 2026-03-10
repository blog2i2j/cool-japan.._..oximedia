#![allow(dead_code)]

//! Capacity planning and resource forecasting for media pipelines.
//!
//! Provides trend analysis and simple linear projection of resource usage
//! so operators can anticipate when additional capacity will be needed.

use std::collections::VecDeque;
use std::time::{Duration, SystemTime};

// ---------------------------------------------------------------------------
// Data model
// ---------------------------------------------------------------------------

/// A single resource observation.
#[derive(Debug, Clone, Copy)]
pub struct ResourceObservation {
    /// Timestamp of the observation.
    pub timestamp: SystemTime,
    /// Observed utilisation (0.0 – 1.0 typically, but can exceed 1.0 for over-provisioned metrics).
    pub utilisation: f64,
}

impl ResourceObservation {
    /// Create an observation at the current time.
    #[must_use]
    pub fn now(utilisation: f64) -> Self {
        Self {
            timestamp: SystemTime::now(),
            utilisation,
        }
    }

    /// Create an observation with an explicit timestamp.
    #[must_use]
    pub fn at(timestamp: SystemTime, utilisation: f64) -> Self {
        Self {
            timestamp,
            utilisation,
        }
    }
}

/// Kind of resource being tracked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResourceKind {
    /// CPU utilisation.
    Cpu,
    /// Memory utilisation.
    Memory,
    /// Disk utilisation.
    Disk,
    /// Network bandwidth utilisation.
    Network,
    /// GPU utilisation.
    Gpu,
    /// Encoding pipeline throughput headroom.
    EncodeThroughput,
    /// Custom / user-defined.
    Custom,
}

impl ResourceKind {
    /// Return a human-readable label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            Self::Memory => "Memory",
            Self::Disk => "Disk",
            Self::Network => "Network",
            Self::Gpu => "GPU",
            Self::EncodeThroughput => "Encode Throughput",
            Self::Custom => "Custom",
        }
    }
}

/// Configuration for a capacity planner instance.
#[derive(Debug, Clone)]
pub struct PlannerConfig {
    /// Resource being planned.
    pub kind: ResourceKind,
    /// Maximum number of historical observations to keep.
    pub max_observations: usize,
    /// Utilisation threshold that is considered "full" (default 0.90).
    pub saturation_threshold: f64,
    /// Warning threshold (default 0.75).
    pub warning_threshold: f64,
}

impl PlannerConfig {
    /// Create a default config for the given resource kind.
    #[must_use]
    pub fn new(kind: ResourceKind) -> Self {
        Self {
            kind,
            max_observations: 1000,
            saturation_threshold: 0.90,
            warning_threshold: 0.75,
        }
    }

    /// Override the saturation threshold.
    #[must_use]
    pub fn with_saturation(mut self, v: f64) -> Self {
        self.saturation_threshold = v.clamp(0.0, 1.0);
        self
    }

    /// Override the warning threshold.
    #[must_use]
    pub fn with_warning(mut self, v: f64) -> Self {
        self.warning_threshold = v.clamp(0.0, 1.0);
        self
    }

    /// Override the history size.
    #[must_use]
    pub fn with_max_observations(mut self, n: usize) -> Self {
        self.max_observations = n.max(2);
        self
    }
}

/// Result of a linear regression fit.
#[derive(Debug, Clone, Copy)]
pub struct LinearFit {
    /// Slope (change in utilisation per second).
    pub slope: f64,
    /// Y-intercept (utilisation at t=0 of the observation window).
    pub intercept: f64,
    /// Coefficient of determination (R^2).
    pub r_squared: f64,
}

/// Status level derived from current utilisation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapacityStatus {
    /// Plenty of headroom.
    Healthy,
    /// Approaching the warning threshold.
    Warning,
    /// At or above the saturation threshold.
    Saturated,
}

/// Forecast result: when will saturation be reached?
#[derive(Debug, Clone)]
pub struct Forecast {
    /// Estimated duration until saturation, if trend is upward.
    pub time_to_saturation: Option<Duration>,
    /// Projected utilisation at the given horizon.
    pub projected_utilisation: f64,
    /// Current status.
    pub status: CapacityStatus,
    /// Linear fit used for projection.
    pub fit: LinearFit,
}

// ---------------------------------------------------------------------------
// Planner
// ---------------------------------------------------------------------------

/// Capacity planner for a single resource dimension.
#[derive(Debug)]
pub struct CapacityPlanner {
    /// Configuration.
    config: PlannerConfig,
    /// Ring buffer of observations (oldest first).
    observations: VecDeque<ResourceObservation>,
}

impl CapacityPlanner {
    /// Create a new planner from config.
    #[must_use]
    pub fn new(config: PlannerConfig) -> Self {
        Self {
            config,
            observations: VecDeque::new(),
        }
    }

    /// Record a new observation.
    pub fn observe(&mut self, obs: ResourceObservation) {
        self.observations.push_back(obs);
        while self.observations.len() > self.config.max_observations {
            self.observations.pop_front();
        }
    }

    /// Record utilisation at the current time.
    pub fn observe_now(&mut self, utilisation: f64) {
        self.observe(ResourceObservation::now(utilisation));
    }

    /// Number of stored observations.
    #[must_use]
    pub fn observation_count(&self) -> usize {
        self.observations.len()
    }

    /// Current utilisation (latest observation).
    #[must_use]
    pub fn current_utilisation(&self) -> Option<f64> {
        self.observations.back().map(|o| o.utilisation)
    }

    /// Determine current capacity status.
    #[must_use]
    pub fn status(&self) -> CapacityStatus {
        match self.current_utilisation() {
            Some(u) if u >= self.config.saturation_threshold => CapacityStatus::Saturated,
            Some(u) if u >= self.config.warning_threshold => CapacityStatus::Warning,
            _ => CapacityStatus::Healthy,
        }
    }

    /// Perform ordinary-least-squares linear regression on the observation window.
    ///
    /// Returns `None` if fewer than 2 observations exist.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn linear_fit(&self) -> Option<LinearFit> {
        if self.observations.len() < 2 {
            return None;
        }

        let base_time = self.observations.front()?.timestamp;
        let n = self.observations.len() as f64;

        let mut sum_x = 0.0_f64;
        let mut sum_y = 0.0_f64;
        let mut sum_xy = 0.0_f64;
        let mut sum_x2 = 0.0_f64;
        let mut sum_y2 = 0.0_f64;

        for obs in &self.observations {
            let x = obs
                .timestamp
                .duration_since(base_time)
                .unwrap_or_default()
                .as_secs_f64();
            let y = obs.utilisation;
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_x2 += x * x;
            sum_y2 += y * y;
        }

        let denom = n * sum_x2 - sum_x * sum_x;
        if denom.abs() < 1e-15 {
            return Some(LinearFit {
                slope: 0.0,
                intercept: sum_y / n,
                r_squared: 0.0,
            });
        }

        let slope = (n * sum_xy - sum_x * sum_y) / denom;
        let intercept = (sum_y - slope * sum_x) / n;

        // R^2
        let ss_tot = sum_y2 - (sum_y * sum_y) / n;
        let ss_res = sum_y2 - slope * sum_xy - intercept * sum_y;
        let r_squared = if ss_tot.abs() < 1e-15 {
            1.0
        } else {
            1.0 - ss_res / ss_tot
        };

        Some(LinearFit {
            slope,
            intercept,
            r_squared: r_squared.clamp(0.0, 1.0),
        })
    }

    /// Project utilisation at a future horizon and estimate time-to-saturation.
    #[must_use]
    pub fn forecast(&self, horizon: Duration) -> Option<Forecast> {
        let fit = self.linear_fit()?;
        let base_time = self.observations.front()?.timestamp;
        let latest_time = self.observations.back()?.timestamp;
        let current_secs = latest_time
            .duration_since(base_time)
            .unwrap_or_default()
            .as_secs_f64();

        let horizon_secs = current_secs + horizon.as_secs_f64();
        let projected = fit.slope * horizon_secs + fit.intercept;

        let time_to_sat = if fit.slope > 1e-12 {
            let current_util = fit.slope * current_secs + fit.intercept;
            let remaining = self.config.saturation_threshold - current_util;
            if remaining > 0.0 {
                let secs = remaining / fit.slope;
                Some(Duration::from_secs_f64(secs))
            } else {
                Some(Duration::ZERO)
            }
        } else {
            None // not increasing
        };

        let status = if projected >= self.config.saturation_threshold {
            CapacityStatus::Saturated
        } else if projected >= self.config.warning_threshold {
            CapacityStatus::Warning
        } else {
            CapacityStatus::Healthy
        };

        Some(Forecast {
            time_to_saturation: time_to_sat,
            projected_utilisation: projected,
            status,
            fit,
        })
    }

    /// Compute the moving average over the last `window` observations.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn moving_average(&self, window: usize) -> Option<f64> {
        if self.observations.is_empty() || window == 0 {
            return None;
        }
        let w = window.min(self.observations.len());
        let start = self.observations.len() - w;
        let sum: f64 = self
            .observations
            .iter()
            .skip(start)
            .map(|o| o.utilisation)
            .sum();
        Some(sum / w as f64)
    }

    /// Clear all observations.
    pub fn clear(&mut self) {
        self.observations.clear();
    }

    /// Reference to the planner config.
    #[must_use]
    pub fn config(&self) -> &PlannerConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_planner_with_linear_data(n: usize) -> CapacityPlanner {
        let mut planner = CapacityPlanner::new(
            PlannerConfig::new(ResourceKind::Cpu).with_max_observations(n + 10),
        );
        let base = SystemTime::now();
        for i in 0..n {
            let ts = base + Duration::from_secs(i as u64);
            // linear: 0.1 + 0.001 * i
            let util = 0.1 + 0.001 * i as f64;
            planner.observe(ResourceObservation::at(ts, util));
        }
        planner
    }

    #[test]
    fn test_resource_kind_label() {
        assert_eq!(ResourceKind::Cpu.label(), "CPU");
        assert_eq!(ResourceKind::Memory.label(), "Memory");
        assert_eq!(ResourceKind::Custom.label(), "Custom");
    }

    #[test]
    fn test_planner_config_defaults() {
        let cfg = PlannerConfig::new(ResourceKind::Disk);
        assert!((cfg.saturation_threshold - 0.90).abs() < 1e-9);
        assert!((cfg.warning_threshold - 0.75).abs() < 1e-9);
        assert_eq!(cfg.max_observations, 1000);
    }

    #[test]
    fn test_planner_observe_and_count() {
        let mut p = CapacityPlanner::new(PlannerConfig::new(ResourceKind::Cpu));
        p.observe_now(0.5);
        p.observe_now(0.6);
        assert_eq!(p.observation_count(), 2);
    }

    #[test]
    fn test_planner_ring_eviction() {
        let mut p =
            CapacityPlanner::new(PlannerConfig::new(ResourceKind::Memory).with_max_observations(3));
        for i in 0..5 {
            p.observe_now(i as f64 * 0.1);
        }
        assert_eq!(p.observation_count(), 3);
    }

    #[test]
    fn test_current_utilisation() {
        let mut p = CapacityPlanner::new(PlannerConfig::new(ResourceKind::Gpu));
        assert!(p.current_utilisation().is_none());
        p.observe_now(0.42);
        assert!(
            (p.current_utilisation()
                .expect("current_utilisation should succeed")
                - 0.42)
                .abs()
                < 1e-9
        );
    }

    #[test]
    fn test_status_healthy() {
        let mut p = CapacityPlanner::new(PlannerConfig::new(ResourceKind::Cpu));
        p.observe_now(0.30);
        assert_eq!(p.status(), CapacityStatus::Healthy);
    }

    #[test]
    fn test_status_warning() {
        let mut p = CapacityPlanner::new(PlannerConfig::new(ResourceKind::Cpu));
        p.observe_now(0.80);
        assert_eq!(p.status(), CapacityStatus::Warning);
    }

    #[test]
    fn test_status_saturated() {
        let mut p = CapacityPlanner::new(PlannerConfig::new(ResourceKind::Cpu));
        p.observe_now(0.95);
        assert_eq!(p.status(), CapacityStatus::Saturated);
    }

    #[test]
    fn test_linear_fit_slope_positive() {
        let p = make_planner_with_linear_data(100);
        let fit = p.linear_fit().expect("linear_fit should succeed");
        assert!(fit.slope > 0.0);
        assert!(
            fit.r_squared > 0.99,
            "R^2 should be near 1 for perfect linear data"
        );
    }

    #[test]
    fn test_linear_fit_insufficient_data() {
        let mut p = CapacityPlanner::new(PlannerConfig::new(ResourceKind::Cpu));
        p.observe_now(0.5);
        assert!(p.linear_fit().is_none());
    }

    #[test]
    fn test_forecast_projects_correctly() {
        let p = make_planner_with_linear_data(100);
        let forecast = p
            .forecast(Duration::from_secs(600))
            .expect("operation should succeed");
        // slope ~ 0.001 / sec, after 600 extra seconds from last point: extra ~ 0.6
        // last observation util ~ 0.1 + 0.001*99 = 0.199
        // projected ~ 0.199 + 0.6 = 0.799
        assert!(
            forecast.projected_utilisation > 0.5,
            "projected should be > 0.5, got {}",
            forecast.projected_utilisation
        );
    }

    #[test]
    fn test_forecast_time_to_saturation() {
        let p = make_planner_with_linear_data(100);
        let forecast = p
            .forecast(Duration::from_secs(1))
            .expect("operation should succeed");
        assert!(forecast.time_to_saturation.is_some());
        let tts = forecast
            .time_to_saturation
            .expect("time_to_saturation should be valid");
        assert!(tts.as_secs() > 0, "time-to-saturation should be positive");
    }

    #[test]
    fn test_moving_average() {
        let mut p = CapacityPlanner::new(PlannerConfig::new(ResourceKind::Network));
        p.observe_now(0.1);
        p.observe_now(0.2);
        p.observe_now(0.3);
        let ma = p.moving_average(3).expect("moving_average should succeed");
        assert!((ma - 0.2).abs() < 1e-9);
    }

    #[test]
    fn test_moving_average_window_larger_than_data() {
        let mut p = CapacityPlanner::new(PlannerConfig::new(ResourceKind::Disk));
        p.observe_now(0.5);
        let ma = p
            .moving_average(100)
            .expect("moving_average should succeed");
        assert!((ma - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_clear() {
        let mut p = CapacityPlanner::new(PlannerConfig::new(ResourceKind::Cpu));
        p.observe_now(0.5);
        p.clear();
        assert_eq!(p.observation_count(), 0);
        assert!(p.current_utilisation().is_none());
    }
}
