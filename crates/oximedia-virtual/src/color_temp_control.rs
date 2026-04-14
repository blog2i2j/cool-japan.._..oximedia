//! LED volume color temperature control.
//!
//! Provides correlated color temperature (CCT) targeting, Duv (distance from
//! the Planckian locus) correction, and panel-uniformity compensation for
//! LED video walls used in virtual production.
//!
//! # Concepts
//!
//! - **CCT** — Correlated Color Temperature in Kelvin. Warm tones ≈ 2700 K,
//!   neutral daylight ≈ 5500–6500 K, cool/blue ≈ 8000+ K.
//! - **Duv** — Signed distance from the Planckian (blackbody) locus in the
//!   CIE 1960 (u, v) chromaticity diagram. Positive Duv = greenish, negative
//!   Duv = magenta/pink. Target for broadcast is |Duv| < 0.005.
//! - **CIE xy chromaticity** — Two-dimensional chromaticity coordinates
//!   derived from the XYZ tristimulus values.
//!
//! # Color temperature formulae
//!
//! Robertson's method is used to convert CCT→xy and the inverse McCamy
//! approximation for xy→CCT.  Both are standard broadcast engineering
//! references.
//!
//! # Example
//!
//! ```rust
//! use oximedia_virtual::color_temp_control::{
//!     ColorTempController, ColorTempConfig, PanelColorState,
//! };
//!
//! let config = ColorTempConfig::default();
//! let controller = ColorTempController::new(config);
//!
//! // Compute the xy chromaticity for 5600 K (film/video daylight standard).
//! let (x, y) = controller.cct_to_xy(5600.0).expect("valid CCT");
//! assert!((x - 0.328).abs() < 0.005, "x chromaticity near daylight");
//!
//! // Estimate CCT from those coordinates.
//! let cct_est = controller.xy_to_cct(x, y).expect("valid xy");
//! assert!((cct_est - 5600.0).abs() < 100.0, "round-trip CCT within 100 K");
//! ```

use std::collections::HashMap;
use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by the color temperature control system.
#[derive(Debug, Error)]
pub enum ColorTempError {
    /// The requested CCT is outside the supported range (1000–20000 K).
    #[error("CCT {0} K is outside the supported range [1000, 20000]")]
    CctOutOfRange(f64),

    /// The supplied xy chromaticity coordinates are invalid.
    #[error("invalid chromaticity coordinates ({x}, {y})")]
    InvalidChromaticity {
        /// x component.
        x: f64,
        /// y component.
        y: f64,
    },

    /// The panel ID was not found in the registry.
    #[error("panel not found: {0}")]
    PanelNotFound(String),

    /// A numerical result was non-finite.
    #[error("non-finite value in color temperature computation: {0}")]
    NonFinite(String),

    /// Configuration is invalid.
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
}

/// Convenience result alias.
pub type Result<T> = std::result::Result<T, ColorTempError>;

// ---------------------------------------------------------------------------
// Robertson's table (reciprocal megakelvin, u, v, t)
// From Wyszecki & Stiles "Color Science" Table 1(3.11)
// ---------------------------------------------------------------------------

/// One row in Robertson's table: `(r, u, v)` where r = 10^6 / T.
#[derive(Clone, Copy)]
struct RobRow {
    r: f64, // reciprocal megakelvin = 1e6 / T
    u: f64,
    v: f64,
}

/// Robertson's isotemperature table (truncated to 31 standard entries).
const ROB: &[RobRow] = &[
    RobRow { r:   0.0, u: 0.18006, v: 0.26352 },
    RobRow { r:  10.0, u: 0.18066, v: 0.26589 },
    RobRow { r:  20.0, u: 0.18133, v: 0.26846 },
    RobRow { r:  30.0, u: 0.18208, v: 0.27119 },
    RobRow { r:  40.0, u: 0.18293, v: 0.27407 },
    RobRow { r:  50.0, u: 0.18388, v: 0.27709 },
    RobRow { r:  60.0, u: 0.18494, v: 0.28021 },
    RobRow { r:  70.0, u: 0.18611, v: 0.28342 },
    RobRow { r:  80.0, u: 0.18740, v: 0.28668 },
    RobRow { r:  90.0, u: 0.18880, v: 0.28997 },
    RobRow { r: 100.0, u: 0.19032, v: 0.29326 },
    RobRow { r: 125.0, u: 0.19462, v: 0.30141 },
    RobRow { r: 150.0, u: 0.19962, v: 0.30921 },
    RobRow { r: 175.0, u: 0.20525, v: 0.31647 },
    RobRow { r: 200.0, u: 0.21142, v: 0.32312 },
    RobRow { r: 225.0, u: 0.21807, v: 0.32909 },
    RobRow { r: 250.0, u: 0.22511, v: 0.33439 },
    RobRow { r: 275.0, u: 0.23247, v: 0.33904 },
    RobRow { r: 300.0, u: 0.24010, v: 0.34308 },
    RobRow { r: 325.0, u: 0.24792, v: 0.34655 },
    RobRow { r: 350.0, u: 0.25591, v: 0.34951 },
    RobRow { r: 375.0, u: 0.26400, v: 0.35200 },
    RobRow { r: 400.0, u: 0.27218, v: 0.35407 },
    RobRow { r: 425.0, u: 0.28039, v: 0.35577 },
    RobRow { r: 450.0, u: 0.28863, v: 0.35714 },
    RobRow { r: 475.0, u: 0.29685, v: 0.35823 },
    RobRow { r: 500.0, u: 0.30505, v: 0.35907 },
    RobRow { r: 525.0, u: 0.31320, v: 0.35968 },
    RobRow { r: 550.0, u: 0.32129, v: 0.36011 },
    RobRow { r: 575.0, u: 0.32931, v: 0.36038 },
    RobRow { r: 600.0, u: 0.33724, v: 0.36051 },
];

// ---------------------------------------------------------------------------
// CIE 1960 (u,v) ↔ CIE (x,y) conversion helpers
// ---------------------------------------------------------------------------

/// Convert CIE xy to CIE 1960 uv.
fn xy_to_uv(x: f64, y: f64) -> Option<(f64, f64)> {
    let denom = -2.0 * x + 12.0 * y + 3.0;
    if denom.abs() < f64::EPSILON {
        return None;
    }
    let u = 4.0 * x / denom;
    let v = 6.0 * y / denom;
    Some((u, v))
}

/// Convert CIE 1960 uv to CIE xy.
fn uv_to_xy(u: f64, v: f64) -> Option<(f64, f64)> {
    let denom = 2.0 * u - 8.0 * v + 4.0;
    if denom.abs() < f64::EPSILON {
        return None;
    }
    let x = 3.0 * u / denom;
    let y = 2.0 * v / denom;
    Some((x, y))
}

// ---------------------------------------------------------------------------
// Panel color state
// ---------------------------------------------------------------------------

/// Measured or calibrated color state of a single LED panel.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PanelColorState {
    /// Panel identifier.
    pub id: String,
    /// Measured CCT in Kelvin.
    pub measured_cct: f64,
    /// Measured Duv (distance from Planckian locus; CIE 1960 uv).
    pub measured_duv: f64,
    /// Current RGB gain offsets `[r, g, b]` applied for uniformity correction.
    pub gain_rgb: [f64; 3],
    /// Whether this panel has been calibrated.
    pub calibrated: bool,
    /// Panel physical position `[x, y]` on the LED wall in metres.
    pub position: [f64; 2],
}

impl PanelColorState {
    /// Create a new uncalibrated panel state at a given position.
    #[must_use]
    pub fn new(id: impl Into<String>, position: [f64; 2]) -> Self {
        Self {
            id: id.into(),
            measured_cct: 6500.0,
            measured_duv: 0.0,
            gain_rgb: [1.0, 1.0, 1.0],
            calibrated: false,
            position,
        }
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the color temperature controller.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ColorTempConfig {
    /// Target CCT for the LED volume in Kelvin.
    pub target_cct: f64,
    /// Acceptable CCT deviation (±) in Kelvin.
    pub cct_tolerance_k: f64,
    /// Target Duv (usually 0.0 for Planckian).
    pub target_duv: f64,
    /// Maximum allowed |Duv| deviation.
    pub duv_tolerance: f64,
    /// Maximum RGB gain that can be applied for uniformity correction.
    pub max_gain: f64,
    /// Minimum RGB gain (must be > 0).
    pub min_gain: f64,
}

impl Default for ColorTempConfig {
    fn default() -> Self {
        Self {
            target_cct: 5600.0,   // film/video daylight standard
            cct_tolerance_k: 150.0,
            target_duv: 0.0,
            duv_tolerance: 0.005, // broadcast standard
            max_gain: 1.5,
            min_gain: 0.5,
        }
    }
}

impl ColorTempConfig {
    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        if self.target_cct < 1000.0 || self.target_cct > 20_000.0 {
            return Err(ColorTempError::CctOutOfRange(self.target_cct));
        }
        if self.cct_tolerance_k <= 0.0 {
            return Err(ColorTempError::InvalidConfig(
                "cct_tolerance_k must be > 0".into(),
            ));
        }
        if self.max_gain <= self.min_gain || self.min_gain <= 0.0 {
            return Err(ColorTempError::InvalidConfig(
                "gain bounds must satisfy 0 < min_gain < max_gain".into(),
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Uniformity correction result
// ---------------------------------------------------------------------------

/// Recommended gain adjustment for one panel to reach the target CCT/Duv.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PanelCorrectionResult {
    /// Panel ID.
    pub id: String,
    /// Recommended RGB gain `[r, g, b]`.
    pub gain_rgb: [f64; 3],
    /// Estimated residual CCT error after correction in Kelvin.
    pub residual_cct_k: f64,
    /// Estimated residual Duv after correction.
    pub residual_duv: f64,
    /// Whether this panel is within tolerance after correction.
    pub in_tolerance: bool,
}

// ---------------------------------------------------------------------------
// Thermal drift model
// ---------------------------------------------------------------------------

/// Simple thermal drift model: CCT shifts as a function of LED panel
/// temperature (in degrees Celsius).
///
/// Real LED panels typically drift warm as temperature rises.  The model
/// uses a linear coefficient (`cct_per_degree_c`) calibrated empirically.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ThermalDriftModel {
    /// Reference panel temperature at calibration time (°C).
    pub reference_temp_c: f64,
    /// Calibrated CCT at the reference temperature.
    pub reference_cct: f64,
    /// CCT change per °C panel temperature increase (typically negative,
    /// i.e. panels get warmer/lower-CCT as they heat up).
    pub cct_per_degree_c: f64,
}

impl ThermalDriftModel {
    /// Create a new thermal drift model.
    #[must_use]
    pub fn new(reference_temp_c: f64, reference_cct: f64, cct_per_degree_c: f64) -> Self {
        Self {
            reference_temp_c,
            reference_cct,
            cct_per_degree_c,
        }
    }

    /// Predict the CCT at the given panel temperature.
    #[must_use]
    pub fn predict_cct(&self, panel_temp_c: f64) -> f64 {
        let delta_t = panel_temp_c - self.reference_temp_c;
        self.reference_cct + delta_t * self.cct_per_degree_c
    }

    /// Compute the CCT correction needed to reach the target CCT at the
    /// given panel temperature.
    ///
    /// A positive return value means the panel is currently cooler (higher
    /// CCT) than target; a negative value means it is warmer (lower CCT).
    #[must_use]
    pub fn correction_needed(&self, panel_temp_c: f64, target_cct: f64) -> f64 {
        target_cct - self.predict_cct(panel_temp_c)
    }
}

impl Default for ThermalDriftModel {
    fn default() -> Self {
        // Empirical: panels calibrated at 25°C, drift −15 K/°C.
        Self::new(25.0, 6500.0, -15.0)
    }
}

// ---------------------------------------------------------------------------
// Main controller
// ---------------------------------------------------------------------------

/// LED volume color temperature controller.
///
/// Provides:
/// - CCT→xy and xy→CCT conversions (Robertson's method).
/// - Duv computation from xy or (u, v) chromaticity.
/// - Panel uniformity correction via RGB gain.
/// - Thermal drift compensation using the [`ThermalDriftModel`].
pub struct ColorTempController {
    config: ColorTempConfig,
    panels: HashMap<String, PanelColorState>,
    thermal_model: Option<ThermalDriftModel>,
}

impl ColorTempController {
    /// Create a new controller with the given configuration.
    pub fn new(config: ColorTempConfig) -> Self {
        Self {
            config,
            panels: HashMap::new(),
            thermal_model: None,
        }
    }

    /// Attach a thermal drift model for predictive CCT compensation.
    pub fn set_thermal_model(&mut self, model: ThermalDriftModel) {
        self.thermal_model = Some(model);
    }

    /// Return a reference to the current configuration.
    #[must_use]
    pub fn config(&self) -> &ColorTempConfig {
        &self.config
    }

    /// Register a panel.
    pub fn add_panel(&mut self, state: PanelColorState) {
        self.panels.insert(state.id.clone(), state);
    }

    /// Update the measured state of an existing panel.
    pub fn update_panel(&mut self, state: PanelColorState) -> Result<()> {
        if !self.panels.contains_key(&state.id) {
            return Err(ColorTempError::PanelNotFound(state.id));
        }
        self.panels.insert(state.id.clone(), state);
        Ok(())
    }

    /// Remove a panel.
    pub fn remove_panel(&mut self, id: &str) -> Result<PanelColorState> {
        self.panels
            .remove(id)
            .ok_or_else(|| ColorTempError::PanelNotFound(id.to_owned()))
    }

    /// Return the number of registered panels.
    #[must_use]
    pub fn panel_count(&self) -> usize {
        self.panels.len()
    }

    /// Return an iterator over all panel states.
    pub fn panels(&self) -> impl Iterator<Item = &PanelColorState> {
        self.panels.values()
    }

    // -----------------------------------------------------------------------
    // CCT / chromaticity conversions
    // -----------------------------------------------------------------------

    /// Convert a CCT value (Kelvin) to CIE xy chromaticity using Robertson's
    /// method.
    ///
    /// Returns an error if `cct` is outside [1000, 20000] K.
    pub fn cct_to_xy(&self, cct: f64) -> Result<(f64, f64)> {
        if !(1000.0..=20_000.0).contains(&cct) {
            return Err(ColorTempError::CctOutOfRange(cct));
        }
        let r = 1.0e6 / cct; // reciprocal megakelvin

        // Find bracketing rows in Robertson's table.
        let n = ROB.len();
        let idx = ROB.partition_point(|row| row.r < r).min(n - 1);
        let idx = idx.max(1); // ensure we have a left neighbour

        let row0 = &ROB[idx - 1];
        let row1 = &ROB[idx];

        // Linear interpolation in reciprocal megakelvin.
        let frac = if (row1.r - row0.r).abs() < f64::EPSILON {
            0.0
        } else {
            (r - row0.r) / (row1.r - row0.r)
        };

        let u = row0.u + frac * (row1.u - row0.u);
        let v = row0.v + frac * (row1.v - row0.v);

        let xy = uv_to_xy(u, v).ok_or_else(|| {
            ColorTempError::NonFinite("uv_to_xy produced degenerate denominator".into())
        })?;

        if !xy.0.is_finite() || !xy.1.is_finite() {
            return Err(ColorTempError::NonFinite(format!(
                "CCT {} K produced non-finite chromaticity",
                cct
            )));
        }
        Ok(xy)
    }

    /// Estimate the CCT from CIE xy chromaticity using the McCamy cubic
    /// approximation:
    ///
    /// ```text
    /// n = (x - 0.3320) / (y - 0.1858)
    /// CCT = −449 n³ + 3525 n² − 6823.3 n + 5520.33
    /// ```
    ///
    /// Valid for 2856–10000 K; extrapolates reasonably outside that range.
    pub fn xy_to_cct(&self, x: f64, y: f64) -> Result<f64> {
        if !x.is_finite() || !y.is_finite() || y <= 0.0 || x <= 0.0 {
            return Err(ColorTempError::InvalidChromaticity { x, y });
        }
        let denom = y - 0.1858;
        if denom.abs() < f64::EPSILON {
            return Err(ColorTempError::InvalidChromaticity { x, y });
        }
        let n = (x - 0.3320) / denom;
        let cct = -449.0 * n * n * n + 3525.0 * n * n - 6823.3 * n + 5520.33;
        if !cct.is_finite() || cct < 500.0 {
            return Err(ColorTempError::InvalidChromaticity { x, y });
        }
        Ok(cct)
    }

    /// Compute Duv — the signed distance from the Planckian (blackbody) locus
    /// — given CIE xy chromaticity.
    ///
    /// Duv is measured in CIE 1960 (u, v) space.  Positive Duv = above the
    /// locus (greenish), negative Duv = below (magenta/pink).
    pub fn xy_to_duv(&self, x: f64, y: f64) -> Result<f64> {
        let (u_test, v_test) = xy_to_uv(x, y).ok_or(ColorTempError::InvalidChromaticity { x, y })?;

        // Estimate CCT for finding the nearest Planckian point.
        let cct = self.xy_to_cct(x, y).unwrap_or(6500.0);

        // Get the Planckian point at that CCT.
        let (x_plan, y_plan) = self.cct_to_xy(cct.clamp(1000.0, 20_000.0))?;
        let (u_plan, v_plan) = xy_to_uv(x_plan, y_plan)
            .ok_or_else(|| ColorTempError::NonFinite("Planckian uv degenerate".into()))?;

        // Signed distance: positive = above locus in (u,v).
        let du = u_test - u_plan;
        let dv = v_test - v_plan;
        let dist = (du * du + dv * dv).sqrt();
        // Sign: v above Planckian = positive.
        let sign = if v_test > v_plan { 1.0 } else { -1.0 };
        Ok(sign * dist)
    }

    // -----------------------------------------------------------------------
    // Panel uniformity correction
    // -----------------------------------------------------------------------

    /// Compute RGB gain corrections for all panels to bring them in line with
    /// the target CCT/Duv specified in the configuration.
    ///
    /// The algorithm:
    /// 1. Compute the target xy chromaticity.
    /// 2. For each panel, compute the xy chromaticity from its measured CCT
    ///    and Duv.
    /// 3. Compute the delta in (u,v) space and map to RGB gain adjustments
    ///    using a simplified linear model.
    pub fn compute_uniformity_corrections(&self) -> Result<Vec<PanelCorrectionResult>> {
        let (target_x, target_y) = self.cct_to_xy(self.config.target_cct)?;
        let (target_u, target_v) = xy_to_uv(target_x, target_y)
            .ok_or_else(|| ColorTempError::NonFinite("target uv degenerate".into()))?;

        let max_g = self.config.max_gain;
        let min_g = self.config.min_gain;

        let mut results = Vec::with_capacity(self.panels.len());

        for panel in self.panels.values() {
            let (panel_x, panel_y) = self.cct_to_xy(panel.measured_cct.clamp(1000.0, 20_000.0))?;
            let (panel_u, panel_v) = xy_to_uv(panel_x, panel_y)
                .ok_or_else(|| ColorTempError::NonFinite("panel uv degenerate".into()))?;

            // Duv adjustment: shift v of the panel by target_duv - panel.measured_duv.
            let du = target_u - panel_u;
            let dv = (target_v - panel_v) + (self.config.target_duv - panel.measured_duv);

            // Map (du, dv) deltas to approximate RGB gain changes.
            // - du > 0 → need more blue (decrease red / increase blue)
            // - dv > 0 → need more green
            let gain_r = (panel.gain_rgb[0] - du * 2.0).clamp(min_g, max_g);
            let gain_g = (panel.gain_rgb[1] + dv * 3.0).clamp(min_g, max_g);
            let gain_b = (panel.gain_rgb[2] + du * 2.0).clamp(min_g, max_g);

            // Estimate residuals.
            let residual_cct_k = (panel.measured_cct - self.config.target_cct).abs() * 0.05;
            let residual_duv = (panel.measured_duv - self.config.target_duv).abs() * 0.1;

            let in_tolerance = residual_cct_k <= self.config.cct_tolerance_k
                && residual_duv <= self.config.duv_tolerance;

            results.push(PanelCorrectionResult {
                id: panel.id.clone(),
                gain_rgb: [gain_r, gain_g, gain_b],
                residual_cct_k,
                residual_duv,
                in_tolerance,
            });
        }

        Ok(results)
    }

    // -----------------------------------------------------------------------
    // Thermal drift compensation
    // -----------------------------------------------------------------------

    /// Return the thermally-compensated CCT for a panel at the given
    /// panel temperature.  Requires a thermal model to be configured.
    pub fn thermally_compensated_target(&self, panel_temp_c: f64) -> Result<f64> {
        let model = self.thermal_model.as_ref().ok_or_else(|| {
            ColorTempError::InvalidConfig("no thermal model configured".into())
        })?;
        let drift = model.cct_per_degree_c * (panel_temp_c - model.reference_temp_c);
        // Counter-act the drift: if panels get warmer (lower CCT), push target up.
        let compensated = self.config.target_cct - drift;
        if !(1000.0..=20_000.0).contains(&compensated) {
            return Err(ColorTempError::CctOutOfRange(compensated));
        }
        Ok(compensated)
    }

    // -----------------------------------------------------------------------
    // Aggregate statistics
    // -----------------------------------------------------------------------

    /// Compute the mean and standard deviation of CCT across all registered
    /// panels.
    ///
    /// Returns `(mean_cct, std_dev_cct)`.  Returns `(0.0, 0.0)` when no
    /// panels are registered.
    #[must_use]
    pub fn cct_statistics(&self) -> (f64, f64) {
        let n = self.panels.len();
        if n == 0 {
            return (0.0, 0.0);
        }
        let sum: f64 = self.panels.values().map(|p| p.measured_cct).sum();
        let mean = sum / n as f64;
        let variance = self.panels.values().map(|p| {
            let d = p.measured_cct - mean;
            d * d
        }).sum::<f64>() / n as f64;
        (mean, variance.sqrt())
    }

    /// Return the panels whose measured CCT deviates from the target by more
    /// than `tolerance_k` Kelvin.
    pub fn out_of_tolerance_panels(&self, tolerance_k: f64) -> Vec<&PanelColorState> {
        self.panels
            .values()
            .filter(|p| (p.measured_cct - self.config.target_cct).abs() > tolerance_k)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_controller() -> ColorTempController {
        ColorTempController::new(ColorTempConfig::default())
    }

    #[test]
    fn test_cct_to_xy_daylight_5600k() {
        let ctrl = default_controller();
        let (x, y) = ctrl.cct_to_xy(5600.0).expect("valid CCT");
        // CIE xy for ~5600 K should be in the rough daylight region.
        assert!(x > 0.28 && x < 0.36, "x={x} out of expected range");
        assert!(y > 0.28 && y < 0.36, "y={y} out of expected range");
    }

    #[test]
    fn test_cct_to_xy_round_trip() {
        let ctrl = default_controller();
        for &cct in &[2700.0_f64, 4000.0, 5600.0, 6500.0, 9000.0] {
            let (x, y) = ctrl.cct_to_xy(cct).expect("valid CCT");
            let cct_est = ctrl.xy_to_cct(x, y).expect("valid xy");
            assert!(
                (cct_est - cct).abs() < 300.0,
                "CCT round-trip error too large for {cct} K: estimated {cct_est}"
            );
        }
    }

    #[test]
    fn test_cct_out_of_range_rejected() {
        let ctrl = default_controller();
        assert!(ctrl.cct_to_xy(500.0).is_err(), "500 K must fail");
        assert!(ctrl.cct_to_xy(25_000.0).is_err(), "25000 K must fail");
    }

    #[test]
    fn test_xy_to_duv_near_zero_on_locus() {
        let ctrl = default_controller();
        // A point exactly on the Planckian locus should have |Duv| ≈ 0.
        let (x, y) = ctrl.cct_to_xy(6500.0).expect("valid");
        let duv = ctrl.xy_to_duv(x, y).expect("valid xy");
        assert!(duv.abs() < 0.01, "|Duv| on Planckian locus should be near 0; got {duv}");
    }

    #[test]
    fn test_panel_add_and_count() {
        let mut ctrl = default_controller();
        ctrl.add_panel(PanelColorState::new("p1", [0.0, 0.0]));
        ctrl.add_panel(PanelColorState::new("p2", [0.5, 0.0]));
        assert_eq!(ctrl.panel_count(), 2);
    }

    #[test]
    fn test_panel_remove() {
        let mut ctrl = default_controller();
        ctrl.add_panel(PanelColorState::new("p1", [0.0, 0.0]));
        let removed = ctrl.remove_panel("p1").expect("should remove");
        assert_eq!(removed.id, "p1");
        assert_eq!(ctrl.panel_count(), 0);
    }

    #[test]
    fn test_uniformity_correction_computes() {
        let mut ctrl = default_controller();
        ctrl.add_panel(PanelColorState {
            id: "p1".into(),
            measured_cct: 6200.0,
            measured_duv: 0.002,
            gain_rgb: [1.0, 1.0, 1.0],
            calibrated: false,
            position: [0.0, 0.0],
        });
        let results = ctrl.compute_uniformity_corrections().expect("should succeed");
        assert_eq!(results.len(), 1);
        // Gains must be within configured bounds.
        let g = results[0].gain_rgb;
        for gain in g {
            assert!(gain >= ctrl.config.min_gain && gain <= ctrl.config.max_gain,
                "gain {gain} out of bounds");
        }
    }

    #[test]
    fn test_cct_statistics_empty() {
        let ctrl = default_controller();
        let (mean, std) = ctrl.cct_statistics();
        assert_eq!(mean, 0.0);
        assert_eq!(std, 0.0);
    }

    #[test]
    fn test_cct_statistics_nonzero() {
        let mut ctrl = default_controller();
        ctrl.add_panel(PanelColorState {
            id: "a".into(),
            measured_cct: 6000.0,
            measured_duv: 0.0,
            gain_rgb: [1.0, 1.0, 1.0],
            calibrated: true,
            position: [0.0, 0.0],
        });
        ctrl.add_panel(PanelColorState {
            id: "b".into(),
            measured_cct: 7000.0,
            measured_duv: 0.0,
            gain_rgb: [1.0, 1.0, 1.0],
            calibrated: true,
            position: [1.0, 0.0],
        });
        let (mean, std) = ctrl.cct_statistics();
        assert!((mean - 6500.0).abs() < 1e-6, "mean should be 6500; got {mean}");
        assert!(std > 0.0, "std dev should be positive");
    }

    #[test]
    fn test_thermal_drift_model_prediction() {
        let model = ThermalDriftModel::new(25.0, 6500.0, -15.0);
        // At 35°C (10°C above reference) CCT should drop 150 K.
        let predicted = model.predict_cct(35.0);
        assert!((predicted - 6350.0).abs() < 1e-9);
    }

    #[test]
    fn test_thermal_compensation() {
        let mut ctrl = default_controller();
        ctrl.set_thermal_model(ThermalDriftModel::new(25.0, 5600.0, -10.0));
        // At 35°C the panel drifts -100 K; compensation should push target up.
        let compensated = ctrl.thermally_compensated_target(35.0).expect("valid");
        assert!((compensated - 5700.0).abs() < 1e-9,
            "expected 5700.0, got {compensated}");
    }

    #[test]
    fn test_out_of_tolerance_panels() {
        let mut ctrl = default_controller(); // target = 5600 K, tolerance = 150 K
        ctrl.add_panel(PanelColorState {
            id: "ok".into(),
            measured_cct: 5650.0,
            measured_duv: 0.0,
            gain_rgb: [1.0, 1.0, 1.0],
            calibrated: true,
            position: [0.0, 0.0],
        });
        ctrl.add_panel(PanelColorState {
            id: "bad".into(),
            measured_cct: 7000.0,
            measured_duv: 0.0,
            gain_rgb: [1.0, 1.0, 1.0],
            calibrated: false,
            position: [1.0, 0.0],
        });
        let bad = ctrl.out_of_tolerance_panels(150.0);
        assert_eq!(bad.len(), 1, "only 1 panel should be out of tolerance");
        assert_eq!(bad[0].id, "bad");
    }

    #[test]
    fn test_uv_xy_roundtrip() {
        let x0 = 0.3127;
        let y0 = 0.3290;
        let (u, v) = xy_to_uv(x0, y0).expect("valid");
        let (x1, y1) = uv_to_xy(u, v).expect("valid");
        assert!((x1 - x0).abs() < 1e-9, "x round-trip failed: {x1} vs {x0}");
        assert!((y1 - y0).abs() < 1e-9, "y round-trip failed: {y1} vs {y0}");
    }
}
