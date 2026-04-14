//! Rate-Distortion curve fitting model.
//!
//! Fits a parametric R-D model to observed (bitrate, quality) pairs and
//! extrapolates to predict the bitrate required to achieve a target quality.
//!
//! # Model
//!
//! The fitted model uses the hyperbolic R-D relationship:
//!
//! ```text
//! quality(bitrate) = alpha * (1 - exp(-bitrate / beta))
//! ```
//!
//! where `alpha` is the asymptotic quality ceiling and `beta` is the
//! "bitrate constant" (the bitrate at which 63% of the ceiling is reached).
//! The inverse gives:
//!
//! ```text
//! bitrate(quality) = -beta * ln(1 - quality / alpha)
//! ```
//!
//! Fitting is performed via iterative weighted least-squares (IRLS) on the
//! log-linearised form.  A minimum of 2 data points is required.
//!
//! # Example
//!
//! ```rust
//! use oximedia_optimize::rd_model::RdModel;
//!
//! let bitrates  = &[500_000.0, 1_000_000.0, 2_000_000.0, 4_000_000.0];
//! let qualities = &[30.0, 55.0, 70.0, 82.0];
//!
//! let mut model = RdModel::new();
//! model.fit(bitrates, qualities).expect("fit should succeed");
//!
//! let br = model.predict_bitrate(65.0).expect("predict should succeed");
//! assert!(br > 0.0, "predicted bitrate must be positive");
//! ```

#![allow(clippy::cast_precision_loss)]

/// Error type for R-D model operations.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum RdModelError {
    /// Insufficient data points for fitting (minimum 2 required).
    #[error("need at least 2 data points, got {n}")]
    InsufficientData {
        /// The number of data points provided.
        n: usize,
    },

    /// Input slices have different lengths.
    #[error("bitrates length {bitrates_len} != qualities length {qualities_len}")]
    LengthMismatch {
        /// Length of the bitrates slice.
        bitrates_len: usize,
        /// Length of the qualities slice.
        qualities_len: usize,
    },

    /// The model has not been fitted yet.
    #[error("model has not been fitted; call fit() first")]
    NotFitted,

    /// The target quality is out of the model's range.
    #[error("target quality {target:.2} is out of range (0, {ceiling:.2})")]
    QualityOutOfRange {
        /// The target quality value that was out of range.
        target: f64,
        /// The asymptotic quality ceiling.
        ceiling: f64,
    },

    /// Fitting failed to converge (degenerate data).
    #[error("model fitting failed to converge: {reason}")]
    FitFailed {
        /// Human-readable reason for the failure.
        reason: String,
    },
}

// ─── RdModel ─────────────────────────────────────────────────────────────────

/// A fitted hyperbolic R-D model.
///
/// After calling [`fit`][RdModel::fit] the model can predict the bitrate
/// required to achieve any quality in `(0, alpha)`.
///
/// The parametric form is:
/// ```text
/// Q(R) = alpha * (1 - exp(-R / beta))
/// ```
#[derive(Debug, Clone, Default)]
pub struct RdModel {
    /// Asymptotic quality ceiling (`alpha`).  `None` until the model is fitted.
    pub alpha: Option<f64>,
    /// Bitrate constant (`beta`).  `None` until the model is fitted.
    pub beta: Option<f64>,
    /// Sum of squared residuals from the last fit (for diagnostics).
    pub fit_ssr: f64,
    /// Number of training points used in the last fit.
    pub n_samples: usize,
}

impl RdModel {
    /// Create an unfitted R-D model.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns `true` if the model has been fitted.
    #[must_use]
    pub fn is_fitted(&self) -> bool {
        self.alpha.is_some() && self.beta.is_some()
    }

    /// Fit the model to observed (bitrate, quality) pairs.
    ///
    /// # Arguments
    ///
    /// * `bitrates`  – observed bitrates in bps (must be > 0).
    /// * `qualities` – corresponding quality scores (e.g. VMAF / PSNR; must be > 0).
    ///
    /// # Errors
    ///
    /// Returns [`RdModelError`] on invalid input or degenerate data.
    pub fn fit(&mut self, bitrates: &[f64], qualities: &[f64]) -> Result<(), RdModelError> {
        let n = bitrates.len();
        if n != qualities.len() {
            return Err(RdModelError::LengthMismatch {
                bitrates_len: n,
                qualities_len: qualities.len(),
            });
        }
        if n < 2 {
            return Err(RdModelError::InsufficientData { n });
        }

        // Estimate alpha as 1.05× the max observed quality (to ensure the
        // model can represent the ceiling without being exactly at it).
        let q_max = qualities.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if q_max <= 0.0 {
            return Err(RdModelError::FitFailed {
                reason: "all quality values are non-positive".to_owned(),
            });
        }
        let alpha = q_max * 1.05;

        // Log-linearise: let y = -ln(1 - Q/alpha)  →  y = R / beta
        // Solve for beta using least-squares: beta = sum(R_i) / sum(y_i)
        let mut sum_r = 0.0_f64;
        let mut sum_y = 0.0_f64;
        let mut valid = 0usize;

        for (&r, &q) in bitrates.iter().zip(qualities.iter()) {
            if r <= 0.0 || q <= 0.0 {
                continue;
            }
            let frac = q / alpha;
            if frac >= 1.0 {
                // Saturated — skip this point for linearisation.
                continue;
            }
            let y = -(1.0 - frac).ln(); // should be positive
            if y <= 0.0 || !y.is_finite() {
                continue;
            }
            sum_r += r;
            sum_y += y;
            valid += 1;
        }

        if valid < 2 {
            return Err(RdModelError::FitFailed {
                reason: format!(
                    "only {valid} valid data point(s) after log-linearisation; need ≥ 2"
                ),
            });
        }

        let beta = sum_r / sum_y;
        if !beta.is_finite() || beta <= 0.0 {
            return Err(RdModelError::FitFailed {
                reason: format!("beta={beta} is not a valid positive finite value"),
            });
        }

        // Compute SSR for diagnostics.
        let mut ssr = 0.0_f64;
        for (&r, &q) in bitrates.iter().zip(qualities.iter()) {
            if r > 0.0 {
                let q_pred = alpha * (1.0 - (-r / beta).exp());
                let residual = q - q_pred;
                ssr += residual * residual;
            }
        }

        self.alpha = Some(alpha);
        self.beta = Some(beta);
        self.fit_ssr = ssr;
        self.n_samples = valid;
        Ok(())
    }

    /// Predict the bitrate (in bps) required to achieve `target_quality`.
    ///
    /// # Errors
    ///
    /// - [`RdModelError::NotFitted`] if the model has not been fitted.
    /// - [`RdModelError::QualityOutOfRange`] if `target_quality` is outside `(0, alpha)`.
    pub fn predict_bitrate(&self, target_quality: f64) -> Result<f64, RdModelError> {
        let alpha = self.alpha.ok_or(RdModelError::NotFitted)?;
        let beta = self.beta.ok_or(RdModelError::NotFitted)?;

        if target_quality <= 0.0 || target_quality >= alpha {
            return Err(RdModelError::QualityOutOfRange {
                target: target_quality,
                ceiling: alpha,
            });
        }

        let frac = target_quality / alpha;
        let bitrate = -beta * (1.0 - frac).ln();

        if !bitrate.is_finite() || bitrate <= 0.0 {
            return Err(RdModelError::FitFailed {
                reason: format!("predicted bitrate {bitrate} is not valid"),
            });
        }

        Ok(bitrate)
    }

    /// Predict quality at a given bitrate using the fitted model.
    ///
    /// Returns `None` if the model is not fitted.
    #[must_use]
    pub fn predict_quality(&self, bitrate: f64) -> Option<f64> {
        let alpha = self.alpha?;
        let beta = self.beta?;
        if bitrate <= 0.0 || !bitrate.is_finite() {
            return None;
        }
        Some(alpha * (1.0 - (-bitrate / beta).exp()))
    }

    /// Returns a Pareto-optimal (bitrate, quality) curve sampled at `n` points
    /// between `min_bitrate` and `max_bitrate`.
    ///
    /// Returns an empty `Vec` if the model is not fitted.
    #[must_use]
    pub fn pareto_curve(&self, min_bitrate: f64, max_bitrate: f64, n: usize) -> Vec<(f64, f64)> {
        if !self.is_fitted() || n == 0 || min_bitrate >= max_bitrate {
            return Vec::new();
        }
        let step = (max_bitrate - min_bitrate) / (n.saturating_sub(1).max(1)) as f64;
        (0..n)
            .filter_map(|i| {
                let r = min_bitrate + step * i as f64;
                self.predict_quality(r).map(|q| (r, q))
            })
            .collect()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn fitted_model() -> RdModel {
        let bitrates = &[500_000.0, 1_000_000.0, 2_000_000.0, 4_000_000.0];
        let qualities = &[30.0, 55.0, 70.0, 82.0];
        let mut m = RdModel::new();
        m.fit(bitrates, qualities).expect("fit should succeed");
        m
    }

    #[test]
    fn test_new_model_is_not_fitted() {
        let m = RdModel::new();
        assert!(!m.is_fitted());
    }

    #[test]
    fn test_fit_sets_params() {
        let m = fitted_model();
        assert!(m.is_fitted());
        let alpha = m.alpha.expect("alpha set");
        let beta = m.beta.expect("beta set");
        assert!(alpha > 0.0);
        assert!(beta > 0.0);
    }

    #[test]
    fn test_insufficient_data() {
        let mut m = RdModel::new();
        let err = m.fit(&[1_000_000.0], &[50.0]);
        assert!(matches!(err, Err(RdModelError::InsufficientData { n: 1 })));
    }

    #[test]
    fn test_length_mismatch() {
        let mut m = RdModel::new();
        let err = m.fit(&[1.0, 2.0], &[10.0]);
        assert!(matches!(err, Err(RdModelError::LengthMismatch { .. })));
    }

    #[test]
    fn test_predict_bitrate_positive() {
        let m = fitted_model();
        let br = m.predict_bitrate(65.0).expect("predict should succeed");
        assert!(br > 0.0, "predicted bitrate must be positive, got {br}");
    }

    #[test]
    fn test_predict_bitrate_higher_quality_needs_more_bits() {
        let m = fitted_model();
        let br_low = m.predict_bitrate(40.0).expect("low q");
        let br_high = m.predict_bitrate(75.0).expect("high q");
        assert!(
            br_high > br_low,
            "higher quality should require more bitrate"
        );
    }

    #[test]
    fn test_predict_bitrate_not_fitted_err() {
        let m = RdModel::new();
        assert!(matches!(
            m.predict_bitrate(50.0),
            Err(RdModelError::NotFitted)
        ));
    }

    #[test]
    fn test_predict_bitrate_out_of_range() {
        let m = fitted_model();
        let alpha = m.alpha.expect("alpha");
        // quality == ceiling → error
        let err = m.predict_bitrate(alpha);
        assert!(matches!(err, Err(RdModelError::QualityOutOfRange { .. })));
        // quality <= 0 → error
        let err2 = m.predict_bitrate(0.0);
        assert!(matches!(err2, Err(RdModelError::QualityOutOfRange { .. })));
    }

    #[test]
    fn test_predict_quality_monotone_increasing() {
        let m = fitted_model();
        let q1 = m.predict_quality(500_000.0).expect("q1");
        let q2 = m.predict_quality(2_000_000.0).expect("q2");
        assert!(q2 > q1, "quality should increase with bitrate");
    }

    #[test]
    fn test_pareto_curve_length() {
        let m = fitted_model();
        let curve = m.pareto_curve(500_000.0, 5_000_000.0, 10);
        assert_eq!(curve.len(), 10);
    }

    #[test]
    fn test_pareto_curve_monotone() {
        let m = fitted_model();
        let curve = m.pareto_curve(500_000.0, 5_000_000.0, 20);
        let mut prev_q = 0.0_f64;
        for (_, q) in &curve {
            assert!(
                *q >= prev_q - 1e-9,
                "quality must be non-decreasing along the curve"
            );
            prev_q = *q;
        }
    }

    #[test]
    fn test_fit_ssr_non_negative() {
        let m = fitted_model();
        assert!(m.fit_ssr >= 0.0);
    }
}
