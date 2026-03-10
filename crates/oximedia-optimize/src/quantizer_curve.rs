#![allow(dead_code)]
//! QP-to-quality curve modeling and analysis.
//!
//! This module models the relationship between quantization parameter (QP) and
//! visual quality metrics (PSNR, SSIM). It supports building empirical curves
//! from encoding samples, fitting parametric models, and predicting quality at
//! arbitrary QP values. Useful for CRF-like targeting and ABR mode decisions.

use std::collections::BTreeMap;

/// A single data point mapping QP to measured quality.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QpQualitySample {
    /// Quantization parameter value (0-63 typically).
    pub qp: f64,
    /// PSNR measurement in dB.
    pub psnr: f64,
    /// SSIM measurement (0.0 to 1.0).
    pub ssim: f64,
    /// Bitrate at this QP in bits per second.
    pub bitrate_bps: f64,
}

impl QpQualitySample {
    /// Creates a new quality sample.
    #[must_use]
    pub fn new(qp: f64, psnr: f64, ssim: f64, bitrate_bps: f64) -> Self {
        Self {
            qp,
            psnr,
            ssim,
            bitrate_bps,
        }
    }
}

/// Fitted model type for the QP-quality relationship.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CurveModelType {
    /// Linear fit: quality = a * qp + b.
    Linear,
    /// Logarithmic fit: quality = a * ln(qp) + b.
    Logarithmic,
    /// Exponential decay fit: quality = a * exp(-b * qp) + c.
    Exponential,
}

/// Parameters of a fitted curve model.
#[derive(Debug, Clone)]
pub struct CurveModel {
    /// The model type.
    pub model_type: CurveModelType,
    /// Coefficient a.
    pub coeff_a: f64,
    /// Coefficient b.
    pub coeff_b: f64,
    /// Coefficient c (used for exponential model).
    pub coeff_c: f64,
    /// R-squared goodness of fit (0.0 to 1.0).
    pub r_squared: f64,
}

impl CurveModel {
    /// Creates a linear model with given coefficients.
    #[must_use]
    pub fn linear(a: f64, b: f64, r_squared: f64) -> Self {
        Self {
            model_type: CurveModelType::Linear,
            coeff_a: a,
            coeff_b: b,
            coeff_c: 0.0,
            r_squared,
        }
    }

    /// Predicts quality at a given QP.
    #[must_use]
    pub fn predict(&self, qp: f64) -> f64 {
        match self.model_type {
            CurveModelType::Linear => self.coeff_a * qp + self.coeff_b,
            CurveModelType::Logarithmic => {
                if qp <= 0.0 {
                    self.coeff_b
                } else {
                    self.coeff_a * qp.ln() + self.coeff_b
                }
            }
            CurveModelType::Exponential => self.coeff_a * (-self.coeff_b * qp).exp() + self.coeff_c,
        }
    }

    /// Returns the inverse: finds QP for a target quality value.
    /// Only reliable for monotonic models.
    #[must_use]
    pub fn inverse_predict(&self, target_quality: f64) -> Option<f64> {
        match self.model_type {
            CurveModelType::Linear => {
                if self.coeff_a.abs() < f64::EPSILON {
                    return None;
                }
                Some((target_quality - self.coeff_b) / self.coeff_a)
            }
            CurveModelType::Logarithmic => {
                if self.coeff_a.abs() < f64::EPSILON {
                    return None;
                }
                let ln_val = (target_quality - self.coeff_b) / self.coeff_a;
                Some(ln_val.exp())
            }
            CurveModelType::Exponential => {
                let diff = target_quality - self.coeff_c;
                if self.coeff_a.abs() < f64::EPSILON || self.coeff_b.abs() < f64::EPSILON {
                    return None;
                }
                let ratio = diff / self.coeff_a;
                if ratio <= 0.0 {
                    return None;
                }
                Some(-ratio.ln() / self.coeff_b)
            }
        }
    }
}

/// Builder for constructing QP-quality curves from samples.
#[derive(Debug)]
pub struct QuantizerCurveBuilder {
    samples: Vec<QpQualitySample>,
}

impl Default for QuantizerCurveBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl QuantizerCurveBuilder {
    /// Creates a new empty curve builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
        }
    }

    /// Adds a quality sample.
    pub fn add_sample(&mut self, sample: QpQualitySample) {
        self.samples.push(sample);
    }

    /// Returns the number of samples.
    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Fits a linear model (PSNR vs QP) using least squares.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn fit_linear_psnr(&self) -> Option<CurveModel> {
        if self.samples.len() < 2 {
            return None;
        }
        let n = self.samples.len() as f64;
        let sum_x: f64 = self.samples.iter().map(|s| s.qp).sum();
        let sum_y: f64 = self.samples.iter().map(|s| s.psnr).sum();
        let sum_xy: f64 = self.samples.iter().map(|s| s.qp * s.psnr).sum();
        let sum_xx: f64 = self.samples.iter().map(|s| s.qp * s.qp).sum();

        let denom = n * sum_xx - sum_x * sum_x;
        if denom.abs() < f64::EPSILON {
            return None;
        }

        let a = (n * sum_xy - sum_x * sum_y) / denom;
        let b = (sum_y - a * sum_x) / n;

        // Calculate R-squared
        let y_mean = sum_y / n;
        let ss_tot: f64 = self.samples.iter().map(|s| (s.psnr - y_mean).powi(2)).sum();
        let ss_res: f64 = self
            .samples
            .iter()
            .map(|s| {
                let predicted = a * s.qp + b;
                (s.psnr - predicted).powi(2)
            })
            .sum();

        let r_squared = if ss_tot.abs() < f64::EPSILON {
            1.0
        } else {
            1.0 - ss_res / ss_tot
        };

        Some(CurveModel::linear(a, b, r_squared))
    }

    /// Returns the QP range in the samples.
    #[must_use]
    pub fn qp_range(&self) -> Option<(f64, f64)> {
        if self.samples.is_empty() {
            return None;
        }
        let min = self
            .samples
            .iter()
            .map(|s| s.qp)
            .fold(f64::INFINITY, f64::min);
        let max = self
            .samples
            .iter()
            .map(|s| s.qp)
            .fold(f64::NEG_INFINITY, f64::max);
        Some((min, max))
    }

    /// Returns the PSNR range in the samples.
    #[must_use]
    pub fn psnr_range(&self) -> Option<(f64, f64)> {
        if self.samples.is_empty() {
            return None;
        }
        let min = self
            .samples
            .iter()
            .map(|s| s.psnr)
            .fold(f64::INFINITY, f64::min);
        let max = self
            .samples
            .iter()
            .map(|s| s.psnr)
            .fold(f64::NEG_INFINITY, f64::max);
        Some((min, max))
    }

    /// Returns a reference to all samples.
    #[must_use]
    pub fn samples(&self) -> &[QpQualitySample] {
        &self.samples
    }
}

/// Quantizer curve lookup table for fast QP-to-quality mapping.
#[derive(Debug, Clone)]
pub struct QuantizerLut {
    /// Map from integer QP to estimated PSNR.
    psnr_table: BTreeMap<u32, f64>,
    /// Map from integer QP to estimated SSIM.
    ssim_table: BTreeMap<u32, f64>,
    /// Map from integer QP to estimated bitrate.
    bitrate_table: BTreeMap<u32, f64>,
}

impl Default for QuantizerLut {
    fn default() -> Self {
        Self::new()
    }
}

impl QuantizerLut {
    /// Creates an empty LUT.
    #[must_use]
    pub fn new() -> Self {
        Self {
            psnr_table: BTreeMap::new(),
            ssim_table: BTreeMap::new(),
            bitrate_table: BTreeMap::new(),
        }
    }

    /// Inserts a QP entry into the LUT.
    pub fn insert(&mut self, qp: u32, psnr: f64, ssim: f64, bitrate_bps: f64) {
        self.psnr_table.insert(qp, psnr);
        self.ssim_table.insert(qp, ssim);
        self.bitrate_table.insert(qp, bitrate_bps);
    }

    /// Looks up the estimated PSNR for a given QP.
    #[must_use]
    pub fn lookup_psnr(&self, qp: u32) -> Option<f64> {
        self.psnr_table.get(&qp).copied()
    }

    /// Looks up the estimated SSIM for a given QP.
    #[must_use]
    pub fn lookup_ssim(&self, qp: u32) -> Option<f64> {
        self.ssim_table.get(&qp).copied()
    }

    /// Looks up the estimated bitrate for a given QP.
    #[must_use]
    pub fn lookup_bitrate(&self, qp: u32) -> Option<f64> {
        self.bitrate_table.get(&qp).copied()
    }

    /// Finds the QP that most closely achieves the target PSNR.
    #[must_use]
    pub fn find_qp_for_psnr(&self, target_psnr: f64) -> Option<u32> {
        self.psnr_table
            .iter()
            .min_by(|(_, a_psnr), (_, b_psnr)| {
                let da = (*a_psnr - target_psnr).abs();
                let db = (*b_psnr - target_psnr).abs();
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(qp, _)| *qp)
    }

    /// Finds the QP that most closely achieves the target bitrate.
    #[must_use]
    pub fn find_qp_for_bitrate(&self, target_bps: f64) -> Option<u32> {
        self.bitrate_table
            .iter()
            .min_by(|(_, a_br), (_, b_br)| {
                let da = (*a_br - target_bps).abs();
                let db = (*b_br - target_bps).abs();
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(qp, _)| *qp)
    }

    /// Returns the number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.psnr_table.len()
    }

    /// Returns true if the LUT is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.psnr_table.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qp_quality_sample_new() {
        let s = QpQualitySample::new(22.0, 42.0, 0.97, 5_000_000.0);
        assert!((s.qp - 22.0).abs() < f64::EPSILON);
        assert!((s.psnr - 42.0).abs() < f64::EPSILON);
        assert!((s.ssim - 0.97).abs() < f64::EPSILON);
    }

    #[test]
    fn test_curve_model_linear_predict() {
        // quality = -0.5 * qp + 50
        let model = CurveModel::linear(-0.5, 50.0, 0.99);
        let q = model.predict(20.0);
        assert!((q - 40.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_curve_model_linear_inverse() {
        let model = CurveModel::linear(-0.5, 50.0, 0.99);
        let qp = model
            .inverse_predict(40.0)
            .expect("inverse prediction should succeed");
        assert!((qp - 20.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_curve_model_logarithmic_predict() {
        let model = CurveModel {
            model_type: CurveModelType::Logarithmic,
            coeff_a: -5.0,
            coeff_b: 55.0,
            coeff_c: 0.0,
            r_squared: 0.95,
        };
        let q = model.predict(1.0); // ln(1) = 0
        assert!((q - 55.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_curve_model_logarithmic_at_zero() {
        let model = CurveModel {
            model_type: CurveModelType::Logarithmic,
            coeff_a: -5.0,
            coeff_b: 55.0,
            coeff_c: 0.0,
            r_squared: 0.95,
        };
        let q = model.predict(0.0);
        assert!((q - 55.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_curve_model_exponential_predict() {
        let model = CurveModel {
            model_type: CurveModelType::Exponential,
            coeff_a: 20.0,
            coeff_b: 0.1,
            coeff_c: 25.0,
            r_squared: 0.98,
        };
        let q = model.predict(0.0);
        // 20 * exp(0) + 25 = 45
        assert!((q - 45.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_builder_fit_linear_psnr() {
        let mut builder = QuantizerCurveBuilder::new();
        // Synthetic linear relationship: PSNR = -0.5 * QP + 50
        for qp in [18, 22, 26, 30, 34, 38] {
            #[allow(clippy::cast_precision_loss)]
            let psnr = -0.5 * qp as f64 + 50.0;
            builder.add_sample(QpQualitySample::new(qp as f64, psnr, 0.95, 1_000_000.0));
        }
        let model = builder
            .fit_linear_psnr()
            .expect("linear PSNR fit should succeed");
        assert!((model.coeff_a - (-0.5)).abs() < 0.01);
        assert!((model.coeff_b - 50.0).abs() < 0.1);
        assert!(model.r_squared > 0.99);
    }

    #[test]
    fn test_builder_qp_range() {
        let mut builder = QuantizerCurveBuilder::new();
        builder.add_sample(QpQualitySample::new(18.0, 44.0, 0.98, 8_000_000.0));
        builder.add_sample(QpQualitySample::new(38.0, 34.0, 0.90, 2_000_000.0));
        let (min, max) = builder.qp_range().expect("QP range should be available");
        assert!((min - 18.0).abs() < f64::EPSILON);
        assert!((max - 38.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_builder_psnr_range() {
        let mut builder = QuantizerCurveBuilder::new();
        builder.add_sample(QpQualitySample::new(18.0, 44.0, 0.98, 8_000_000.0));
        builder.add_sample(QpQualitySample::new(38.0, 34.0, 0.90, 2_000_000.0));
        let (min, max) = builder
            .psnr_range()
            .expect("PSNR range should be available");
        assert!((min - 34.0).abs() < f64::EPSILON);
        assert!((max - 44.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_builder_too_few_samples() {
        let mut builder = QuantizerCurveBuilder::new();
        builder.add_sample(QpQualitySample::new(22.0, 42.0, 0.97, 5_000_000.0));
        assert!(builder.fit_linear_psnr().is_none());
    }

    #[test]
    fn test_quantizer_lut_insert_and_lookup() {
        let mut lut = QuantizerLut::new();
        lut.insert(22, 42.0, 0.97, 5_000_000.0);
        lut.insert(28, 38.0, 0.94, 3_000_000.0);
        assert_eq!(lut.len(), 2);
        assert!(
            (lut.lookup_psnr(22).expect("PSNR lookup should succeed") - 42.0).abs() < f64::EPSILON
        );
        assert!(
            (lut.lookup_ssim(28).expect("SSIM lookup should succeed") - 0.94).abs() < f64::EPSILON
        );
        assert!(
            (lut.lookup_bitrate(22)
                .expect("bitrate lookup should succeed")
                - 5_000_000.0)
                .abs()
                < f64::EPSILON
        );
    }

    #[test]
    fn test_quantizer_lut_find_qp_for_psnr() {
        let mut lut = QuantizerLut::new();
        lut.insert(18, 46.0, 0.99, 10_000_000.0);
        lut.insert(22, 42.0, 0.97, 5_000_000.0);
        lut.insert(28, 38.0, 0.94, 3_000_000.0);
        lut.insert(34, 34.0, 0.90, 1_500_000.0);
        let qp = lut
            .find_qp_for_psnr(40.0)
            .expect("QP for target PSNR should be found");
        assert_eq!(qp, 22); // Closest to 42.0
    }

    #[test]
    fn test_quantizer_lut_find_qp_for_bitrate() {
        let mut lut = QuantizerLut::new();
        lut.insert(18, 46.0, 0.99, 10_000_000.0);
        lut.insert(28, 38.0, 0.94, 3_000_000.0);
        lut.insert(34, 34.0, 0.90, 1_500_000.0);
        let qp = lut
            .find_qp_for_bitrate(2_000_000.0)
            .expect("QP for target bitrate should be found");
        assert_eq!(qp, 34); // 1_500_000 is closest to 2_000_000
    }

    #[test]
    fn test_quantizer_lut_empty() {
        let lut = QuantizerLut::new();
        assert!(lut.is_empty());
        assert_eq!(lut.len(), 0);
        assert!(lut.lookup_psnr(22).is_none());
        assert!(lut.find_qp_for_psnr(40.0).is_none());
    }

    #[test]
    fn test_inverse_predict_zero_coeff() {
        let model = CurveModel::linear(0.0, 50.0, 1.0);
        assert!(model.inverse_predict(40.0).is_none());
    }
}
