#![allow(dead_code)]
//! Display aging and drift modeling for calibration compensation.
//!
//! Displays (monitors, projectors, LED panels) change their color characteristics
//! over time due to component aging, thermal drift, and environmental factors.
//! This module provides models to predict display drift and compensate for it,
//! extending the useful lifetime of calibration profiles.
//!
//! # Features
//!
//! - **Luminance decay modeling**: Predict luminance loss over time
//! - **Color shift tracking**: Track chromaticity drift over measurement history
//! - **Thermal drift compensation**: Correct for warm-up related color shifts
//! - **Recalibration scheduling**: Predict when recalibration is needed

/// Display technology type, which affects aging characteristics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisplayTechnology {
    /// LCD (LED-backlit).
    Lcd,
    /// OLED (organic LED).
    Oled,
    /// Projector (DLP/LCD/LCoS).
    Projector,
    /// LED video wall.
    LedWall,
    /// CRT (cathode ray tube).
    Crt,
}

impl DisplayTechnology {
    /// Get the typical half-life in hours for this display technology.
    ///
    /// Half-life is the time for luminance to drop to 50% of original.
    #[must_use]
    pub const fn typical_half_life_hours(&self) -> u64 {
        match self {
            Self::Lcd => 60_000,
            Self::Oled => 30_000,
            Self::Projector => 10_000,
            Self::LedWall => 100_000,
            Self::Crt => 20_000,
        }
    }

    /// Get the typical warm-up time in minutes.
    #[must_use]
    pub const fn warmup_minutes(&self) -> u32 {
        match self {
            Self::Lcd => 30,
            Self::Oled => 5,
            Self::Projector => 45,
            Self::LedWall => 15,
            Self::Crt => 30,
        }
    }
}

/// A single calibration measurement point in time.
#[derive(Debug, Clone)]
pub struct CalibrationMeasurement {
    /// Hours of display usage at measurement time.
    pub usage_hours: f64,
    /// Measured peak luminance (cd/m^2).
    pub peak_luminance: f64,
    /// Measured white point x chromaticity.
    pub white_x: f64,
    /// Measured white point y chromaticity.
    pub white_y: f64,
    /// Measured gamma exponent.
    pub gamma: f64,
    /// Display temperature at measurement (Celsius).
    pub temperature: f64,
}

impl CalibrationMeasurement {
    /// Create a new calibration measurement.
    #[must_use]
    pub fn new(
        usage_hours: f64,
        peak_luminance: f64,
        white_x: f64,
        white_y: f64,
        gamma: f64,
        temperature: f64,
    ) -> Self {
        Self {
            usage_hours,
            peak_luminance,
            white_x,
            white_y,
            gamma,
            temperature,
        }
    }
}

/// Luminance decay model using exponential decay.
#[derive(Debug, Clone)]
pub struct LuminanceDecayModel {
    /// Initial luminance (cd/m^2).
    pub initial_luminance: f64,
    /// Decay constant (1/hours). Luminance = initial * exp(-decay_rate * hours).
    pub decay_rate: f64,
}

impl LuminanceDecayModel {
    /// Create a model from initial luminance and half-life.
    #[must_use]
    pub fn from_half_life(initial_luminance: f64, half_life_hours: f64) -> Self {
        let decay_rate = if half_life_hours > 0.0 {
            (2.0_f64).ln() / half_life_hours
        } else {
            0.0
        };
        Self {
            initial_luminance,
            decay_rate,
        }
    }

    /// Create a model for a specific display technology.
    #[must_use]
    pub fn for_technology(initial_luminance: f64, tech: DisplayTechnology) -> Self {
        Self::from_half_life(initial_luminance, tech.typical_half_life_hours() as f64)
    }

    /// Predict luminance at a given number of usage hours.
    #[must_use]
    pub fn predict(&self, hours: f64) -> f64 {
        self.initial_luminance * (-self.decay_rate * hours).exp()
    }

    /// Predict the percentage of original luminance remaining.
    #[must_use]
    pub fn percent_remaining(&self, hours: f64) -> f64 {
        ((-self.decay_rate * hours).exp()) * 100.0
    }

    /// Estimate hours until luminance drops below a threshold.
    #[must_use]
    pub fn hours_until(&self, target_luminance: f64) -> f64 {
        if self.decay_rate <= 0.0 || target_luminance >= self.initial_luminance {
            return f64::INFINITY;
        }
        if target_luminance <= 0.0 {
            return f64::INFINITY;
        }
        -(target_luminance / self.initial_luminance).ln() / self.decay_rate
    }

    /// Fit a model to a series of measurements using least-squares.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn fit(measurements: &[CalibrationMeasurement]) -> Option<Self> {
        if measurements.len() < 2 {
            return None;
        }
        // Use log-linear regression: ln(L) = ln(L0) - k*t
        let n = measurements.len() as f64;
        let mut sum_t = 0.0_f64;
        let mut sum_ln_l = 0.0_f64;
        let mut sum_t2 = 0.0_f64;
        let mut sum_t_ln_l = 0.0_f64;

        for m in measurements {
            if m.peak_luminance <= 0.0 {
                continue;
            }
            let t = m.usage_hours;
            let ln_l = m.peak_luminance.ln();
            sum_t += t;
            sum_ln_l += ln_l;
            sum_t2 += t * t;
            sum_t_ln_l += t * ln_l;
        }

        let denom = n * sum_t2 - sum_t * sum_t;
        if denom.abs() < 1e-15 {
            return None;
        }

        let slope = (n * sum_t_ln_l - sum_t * sum_ln_l) / denom;
        let intercept = (sum_ln_l - slope * sum_t) / n;

        let initial_luminance = intercept.exp();
        let decay_rate = -slope;

        if decay_rate < 0.0 {
            // Luminance increasing over time doesn't make sense for aging
            return Some(Self {
                initial_luminance,
                decay_rate: 0.0,
            });
        }

        Some(Self {
            initial_luminance,
            decay_rate,
        })
    }
}

/// Chromaticity drift tracker.
#[derive(Debug, Clone)]
pub struct ChromaticityDriftTracker {
    /// Reference white point x.
    pub ref_x: f64,
    /// Reference white point y.
    pub ref_y: f64,
    /// History of measurements.
    pub history: Vec<(f64, f64, f64)>, // (hours, x, y)
}

impl ChromaticityDriftTracker {
    /// Create a new drift tracker with a reference white point.
    #[must_use]
    pub fn new(ref_x: f64, ref_y: f64) -> Self {
        Self {
            ref_x,
            ref_y,
            history: Vec::new(),
        }
    }

    /// Add a measurement to the tracker.
    pub fn add_measurement(&mut self, hours: f64, x: f64, y: f64) {
        self.history.push((hours, x, y));
    }

    /// Compute the current drift magnitude (Euclidean distance in xy).
    #[must_use]
    pub fn current_drift(&self) -> f64 {
        if let Some(&(_, x, y)) = self.history.last() {
            let dx = x - self.ref_x;
            let dy = y - self.ref_y;
            (dx * dx + dy * dy).sqrt()
        } else {
            0.0
        }
    }

    /// Predict drift at a future number of hours using linear extrapolation.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn predict_drift(&self, target_hours: f64) -> (f64, f64) {
        if self.history.len() < 2 {
            return (self.ref_x, self.ref_y);
        }
        // Simple linear regression on x and y vs hours
        let n = self.history.len() as f64;
        let mut sum_t = 0.0_f64;
        let mut sum_x = 0.0_f64;
        let mut sum_y = 0.0_f64;
        let mut sum_t2 = 0.0_f64;
        let mut sum_tx = 0.0_f64;
        let mut sum_ty = 0.0_f64;
        for &(t, x, y) in &self.history {
            sum_t += t;
            sum_x += x;
            sum_y += y;
            sum_t2 += t * t;
            sum_tx += t * x;
            sum_ty += t * y;
        }
        let denom = n * sum_t2 - sum_t * sum_t;
        if denom.abs() < 1e-15 {
            return (self.ref_x, self.ref_y);
        }
        let slope_x = (n * sum_tx - sum_t * sum_x) / denom;
        let intercept_x = (sum_x - slope_x * sum_t) / n;
        let slope_y = (n * sum_ty - sum_t * sum_y) / denom;
        let intercept_y = (sum_y - slope_y * sum_t) / n;

        (
            intercept_x + slope_x * target_hours,
            intercept_y + slope_y * target_hours,
        )
    }

    /// Get the number of measurements in history.
    #[must_use]
    pub fn measurement_count(&self) -> usize {
        self.history.len()
    }
}

/// Thermal drift model for display warm-up compensation.
#[derive(Debug, Clone)]
pub struct ThermalDriftModel {
    /// Steady-state luminance (after warm-up).
    pub steady_luminance: f64,
    /// Cold-start luminance overshoot/undershoot factor.
    pub cold_start_factor: f64,
    /// Time constant for warm-up (minutes).
    pub time_constant_min: f64,
}

impl ThermalDriftModel {
    /// Create a thermal drift model.
    #[must_use]
    pub fn new(steady_luminance: f64, cold_start_factor: f64, time_constant_min: f64) -> Self {
        Self {
            steady_luminance,
            cold_start_factor,
            time_constant_min,
        }
    }

    /// Create a model for a specific display technology.
    #[must_use]
    pub fn for_technology(steady_luminance: f64, tech: DisplayTechnology) -> Self {
        let tc = tech.warmup_minutes() as f64 / 3.0; // ~3 time constants to settle
        let csf = match tech {
            DisplayTechnology::Lcd => 0.92,
            DisplayTechnology::Oled => 0.98,
            DisplayTechnology::Projector => 0.85,
            DisplayTechnology::LedWall => 0.95,
            DisplayTechnology::Crt => 0.90,
        };
        Self::new(steady_luminance, csf, tc)
    }

    /// Predict luminance at a given number of minutes after power-on.
    #[must_use]
    pub fn predict(&self, minutes: f64) -> f64 {
        if self.time_constant_min <= 0.0 {
            return self.steady_luminance;
        }
        let settled_fraction = 1.0 - (-minutes / self.time_constant_min).exp();
        let cold = self.steady_luminance * self.cold_start_factor;
        cold + (self.steady_luminance - cold) * settled_fraction
    }

    /// Check if the display has sufficiently warmed up.
    #[must_use]
    pub fn is_warmed_up(&self, minutes: f64, tolerance_percent: f64) -> bool {
        let current = self.predict(minutes);
        let diff_pct = ((current - self.steady_luminance) / self.steady_luminance).abs() * 100.0;
        diff_pct <= tolerance_percent
    }
}

/// Recalibration scheduler based on aging model predictions.
#[derive(Debug)]
pub struct RecalibrationScheduler {
    /// Maximum acceptable luminance drop percentage.
    pub max_luminance_drop_pct: f64,
    /// Maximum acceptable chromaticity drift (xy distance).
    pub max_chromaticity_drift: f64,
    /// Luminance decay model.
    pub decay_model: Option<LuminanceDecayModel>,
    /// Chromaticity drift tracker.
    pub drift_tracker: Option<ChromaticityDriftTracker>,
}

impl RecalibrationScheduler {
    /// Create a new recalibration scheduler.
    #[must_use]
    pub fn new(max_luminance_drop_pct: f64, max_chromaticity_drift: f64) -> Self {
        Self {
            max_luminance_drop_pct,
            max_chromaticity_drift,
            decay_model: None,
            drift_tracker: None,
        }
    }

    /// Set the luminance decay model.
    #[must_use]
    pub fn with_decay_model(mut self, model: LuminanceDecayModel) -> Self {
        self.decay_model = Some(model);
        self
    }

    /// Set the chromaticity drift tracker.
    #[must_use]
    pub fn with_drift_tracker(mut self, tracker: ChromaticityDriftTracker) -> Self {
        self.drift_tracker = Some(tracker);
        self
    }

    /// Check if recalibration is needed at the given usage hours.
    #[must_use]
    pub fn needs_recalibration(&self, hours: f64) -> bool {
        if let Some(model) = &self.decay_model {
            let pct = model.percent_remaining(hours);
            if 100.0 - pct > self.max_luminance_drop_pct {
                return true;
            }
        }
        if let Some(tracker) = &self.drift_tracker {
            if tracker.current_drift() > self.max_chromaticity_drift {
                return true;
            }
        }
        false
    }

    /// Estimate hours until next recalibration is needed.
    #[must_use]
    pub fn hours_until_recalibration(&self) -> f64 {
        if let Some(model) = &self.decay_model {
            let target = model.initial_luminance * (1.0 - self.max_luminance_drop_pct / 100.0);
            return model.hours_until(target);
        }
        f64::INFINITY
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_display_technology_half_life() {
        assert_eq!(DisplayTechnology::Lcd.typical_half_life_hours(), 60_000);
        assert_eq!(DisplayTechnology::Oled.typical_half_life_hours(), 30_000);
        assert_eq!(
            DisplayTechnology::Projector.typical_half_life_hours(),
            10_000
        );
    }

    #[test]
    fn test_display_technology_warmup() {
        assert_eq!(DisplayTechnology::Lcd.warmup_minutes(), 30);
        assert_eq!(DisplayTechnology::Oled.warmup_minutes(), 5);
    }

    #[test]
    fn test_luminance_decay_prediction() {
        let model = LuminanceDecayModel::from_half_life(300.0, 30000.0);
        let at_zero = model.predict(0.0);
        assert!((at_zero - 300.0).abs() < 1e-10);
        let at_half = model.predict(30000.0);
        assert!((at_half - 150.0).abs() < 1.0);
    }

    #[test]
    fn test_luminance_percent_remaining() {
        let model = LuminanceDecayModel::from_half_life(100.0, 10000.0);
        let pct = model.percent_remaining(0.0);
        assert!((pct - 100.0).abs() < 1e-10);
        let pct_half = model.percent_remaining(10000.0);
        assert!((pct_half - 50.0).abs() < 1.0);
    }

    #[test]
    fn test_hours_until_threshold() {
        let model = LuminanceDecayModel::from_half_life(300.0, 30000.0);
        let hours = model.hours_until(150.0);
        assert!((hours - 30000.0).abs() < 100.0);
    }

    #[test]
    fn test_hours_until_infinity() {
        let model = LuminanceDecayModel::from_half_life(300.0, 30000.0);
        let hours = model.hours_until(300.0);
        assert!(hours.is_infinite());
    }

    #[test]
    fn test_fit_model() {
        let measurements = vec![
            CalibrationMeasurement::new(0.0, 300.0, 0.3127, 0.3290, 2.2, 25.0),
            CalibrationMeasurement::new(10000.0, 240.0, 0.3130, 0.3292, 2.2, 25.0),
            CalibrationMeasurement::new(20000.0, 192.0, 0.3133, 0.3295, 2.2, 26.0),
        ];
        let model = LuminanceDecayModel::fit(&measurements);
        assert!(model.is_some());
        let model = model.expect("expected model to be Some/Ok");
        assert!(model.initial_luminance > 200.0);
        assert!(model.decay_rate > 0.0);
    }

    #[test]
    fn test_fit_insufficient_data() {
        let measurements = vec![CalibrationMeasurement::new(
            0.0, 300.0, 0.3127, 0.3290, 2.2, 25.0,
        )];
        assert!(LuminanceDecayModel::fit(&measurements).is_none());
    }

    #[test]
    fn test_chromaticity_drift_tracker() {
        let mut tracker = ChromaticityDriftTracker::new(0.3127, 0.3290);
        tracker.add_measurement(0.0, 0.3127, 0.3290);
        tracker.add_measurement(5000.0, 0.3130, 0.3295);
        assert!(tracker.current_drift() > 0.0);
        assert_eq!(tracker.measurement_count(), 2);
    }

    #[test]
    fn test_chromaticity_predict_drift() {
        let mut tracker = ChromaticityDriftTracker::new(0.3127, 0.3290);
        tracker.add_measurement(0.0, 0.3127, 0.3290);
        tracker.add_measurement(10000.0, 0.3137, 0.3300);
        let (px, py) = tracker.predict_drift(20000.0);
        // Linear extrapolation should give ~0.3147, 0.3310
        assert!((px - 0.3147).abs() < 0.001);
        assert!((py - 0.3310).abs() < 0.001);
    }

    #[test]
    fn test_thermal_drift_warmup() {
        let model = ThermalDriftModel::new(300.0, 0.9, 10.0);
        // After many time constants, should be near steady state
        let lum = model.predict(100.0);
        assert!((lum - 300.0).abs() < 1.0);
        // At t=0 should be cold_start_factor * steady
        let cold = model.predict(0.0);
        assert!((cold - 270.0).abs() < 1.0);
    }

    #[test]
    fn test_thermal_is_warmed_up() {
        let model = ThermalDriftModel::new(300.0, 0.9, 10.0);
        assert!(!model.is_warmed_up(0.0, 1.0));
        assert!(model.is_warmed_up(100.0, 1.0));
    }

    #[test]
    fn test_recalibration_scheduler() {
        let model = LuminanceDecayModel::from_half_life(300.0, 30000.0);
        let scheduler = RecalibrationScheduler::new(10.0, 0.005).with_decay_model(model);
        assert!(!scheduler.needs_recalibration(0.0));
        assert!(scheduler.needs_recalibration(50000.0));
    }

    #[test]
    fn test_recalibration_hours_estimate() {
        let model = LuminanceDecayModel::from_half_life(300.0, 30000.0);
        let scheduler = RecalibrationScheduler::new(10.0, 0.005).with_decay_model(model);
        let hours = scheduler.hours_until_recalibration();
        assert!(hours > 0.0 && hours < 30000.0);
    }
}
