#![allow(dead_code)]
//! Color-difference calculation and reporting utilities.

use std::f64::consts::PI;

/// A pair of Lab colors and their computed ΔE 2000 difference.
#[derive(Debug, Clone)]
pub struct ColorDifference {
    /// L* of color A.
    pub l1: f64,
    /// a* of color A.
    pub a1: f64,
    /// b* of color A.
    pub b1: f64,
    /// L* of color B.
    pub l2: f64,
    /// a* of color B.
    pub a2: f64,
    /// b* of color B.
    pub b2: f64,
}

impl ColorDifference {
    /// Create a new color difference from two Lab colors.
    pub fn new(l1: f64, a1: f64, b1: f64, l2: f64, a2: f64, b2: f64) -> Self {
        Self {
            l1,
            a1,
            b1,
            l2,
            a2,
            b2,
        }
    }

    /// Compute ΔE 2000 (CIEDE2000) for this pair.
    #[allow(clippy::too_many_lines)]
    pub fn delta_e_2000(&self) -> f64 {
        ciede2000(self.l1, self.a1, self.b1, self.l2, self.a2, self.b2)
    }

    /// Return `true` if the pair is perceptually indistinguishable (ΔE < threshold).
    pub fn perceptual_threshold(&self, threshold: f64) -> bool {
        self.delta_e_2000() < threshold
    }

    /// Compute ΔE 1976 (simple Euclidean Lab distance).
    pub fn delta_e_76(&self) -> f64 {
        let dl = self.l1 - self.l2;
        let da = self.a1 - self.a2;
        let db = self.b1 - self.b2;
        (dl * dl + da * da + db * db).sqrt()
    }
}

/// Statistic report over a batch of color difference measurements.
#[derive(Debug, Clone)]
pub struct ColorDiffReport {
    differences: Vec<ColorDifference>,
    /// Threshold applied when checking pass/fail.
    pub threshold: f64,
}

impl ColorDiffReport {
    /// Create a new report from a list of differences and a pass threshold.
    pub fn new(differences: Vec<ColorDifference>, threshold: f64) -> Self {
        Self {
            differences,
            threshold,
        }
    }

    /// Return the pair with the highest ΔE 2000, or `None` if empty.
    pub fn worst_pair(&self) -> Option<&ColorDifference> {
        self.differences.iter().max_by(|a, b| {
            a.delta_e_2000()
                .partial_cmp(&b.delta_e_2000())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Compute the average ΔE 2000 across all pairs.
    #[allow(clippy::cast_precision_loss)]
    pub fn avg_delta_e(&self) -> f64 {
        if self.differences.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.differences.iter().map(|d| d.delta_e_2000()).sum();
        sum / self.differences.len() as f64
    }

    /// Return `true` if all pairs pass the threshold.
    pub fn passes_threshold(&self) -> bool {
        self.differences
            .iter()
            .all(|d| d.perceptual_threshold(self.threshold))
    }

    /// Number of pairs that fail the threshold.
    pub fn fail_count(&self) -> usize {
        self.differences
            .iter()
            .filter(|d| !d.perceptual_threshold(self.threshold))
            .count()
    }

    /// Total number of pairs in the report.
    pub fn len(&self) -> usize {
        self.differences.len()
    }

    /// Return `true` if there are no pairs in this report.
    pub fn is_empty(&self) -> bool {
        self.differences.is_empty()
    }

    /// Maximum ΔE 2000 in the report.
    pub fn max_delta_e(&self) -> f64 {
        self.differences
            .iter()
            .map(|d| d.delta_e_2000())
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Minimum ΔE 2000 in the report.
    pub fn min_delta_e(&self) -> f64 {
        self.differences
            .iter()
            .map(|d| d.delta_e_2000())
            .fold(f64::INFINITY, f64::min)
    }
}

// ── Internal CIEDE2000 implementation ─────────────────────────────────────

fn ciede2000(l1: f64, a1: f64, b1: f64, l2: f64, a2: f64, b2: f64) -> f64 {
    let c1 = (a1 * a1 + b1 * b1).sqrt();
    let c2 = (a2 * a2 + b2 * b2).sqrt();
    let c_bar = (c1 + c2) / 2.0;
    let g = 0.5 * (1.0 - (c_bar.powi(7) / (c_bar.powi(7) + 25.0_f64.powi(7))).sqrt());

    let a1p = a1 * (1.0 + g);
    let a2p = a2 * (1.0 + g);
    let c1p = (a1p * a1p + b1 * b1).sqrt();
    let c2p = (a2p * a2p + b2 * b2).sqrt();

    let h_prime = |bp: f64, ap: f64| -> f64 {
        if bp == 0.0 && ap == 0.0 {
            0.0
        } else {
            let h = bp.atan2(ap).to_degrees();
            if h < 0.0 {
                h + 360.0
            } else {
                h
            }
        }
    };
    let h1p = h_prime(b1, a1p);
    let h2p = h_prime(b2, a2p);

    let delta_lp = l2 - l1;
    let delta_cp = c2p - c1p;
    let delta_hp = if c1p * c2p == 0.0 {
        0.0
    } else if (h2p - h1p).abs() <= 180.0 {
        h2p - h1p
    } else if h2p - h1p > 180.0 {
        h2p - h1p - 360.0
    } else {
        h2p - h1p + 360.0
    };
    let delta_bhp = 2.0 * (c1p * c2p).sqrt() * (delta_hp / 2.0 * PI / 180.0).sin();

    let l_bar = (l1 + l2) / 2.0;
    let c_bar_p = (c1p + c2p) / 2.0;
    let h_bar_p = if c1p * c2p == 0.0 {
        h1p + h2p
    } else if (h1p - h2p).abs() <= 180.0 {
        (h1p + h2p) / 2.0
    } else if h1p + h2p < 360.0 {
        (h1p + h2p + 360.0) / 2.0
    } else {
        (h1p + h2p - 360.0) / 2.0
    };

    let t = 1.0 - 0.17 * ((h_bar_p - 30.0) * PI / 180.0).cos()
        + 0.24 * (2.0 * h_bar_p * PI / 180.0).cos()
        + 0.32 * ((3.0 * h_bar_p + 6.0) * PI / 180.0).cos()
        - 0.20 * ((4.0 * h_bar_p - 63.0) * PI / 180.0).cos();

    let s_l = 1.0 + 0.015 * (l_bar - 50.0).powi(2) / (20.0 + (l_bar - 50.0).powi(2)).sqrt();
    let s_c = 1.0 + 0.045 * c_bar_p;
    let s_h = 1.0 + 0.015 * c_bar_p * t;

    let delta_theta = 30.0 * (-((h_bar_p - 275.0) / 25.0).powi(2)).exp();
    let r_c = 2.0 * (c_bar_p.powi(7) / (c_bar_p.powi(7) + 25.0_f64.powi(7))).sqrt();
    let r_t = -r_c * (2.0 * delta_theta * PI / 180.0).sin();

    ((delta_lp / s_l).powi(2)
        + (delta_cp / s_c).powi(2)
        + (delta_bhp / s_h).powi(2)
        + r_t * (delta_cp / s_c) * (delta_bhp / s_h))
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identical() -> ColorDifference {
        ColorDifference::new(50.0, 10.0, -5.0, 50.0, 10.0, -5.0)
    }

    #[test]
    fn test_delta_e_2000_identical() {
        assert!(identical().delta_e_2000() < 1e-6);
    }

    #[test]
    fn test_delta_e_2000_positive() {
        let d = ColorDifference::new(50.0, 0.0, 0.0, 60.0, 5.0, 5.0);
        assert!(d.delta_e_2000() > 0.0);
    }

    #[test]
    fn test_delta_e_76_identical() {
        assert!(identical().delta_e_76() < 1e-10);
    }

    #[test]
    fn test_delta_e_76_known() {
        let d = ColorDifference::new(50.0, 0.0, 0.0, 53.0, 4.0, 0.0);
        assert!((d.delta_e_76() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_perceptual_threshold_passes() {
        let d = ColorDifference::new(50.0, 0.0, 0.0, 50.0, 0.0, 0.0);
        assert!(d.perceptual_threshold(1.0));
    }

    #[test]
    fn test_perceptual_threshold_fails() {
        let d = ColorDifference::new(50.0, 0.0, 0.0, 70.0, 30.0, 20.0);
        assert!(!d.perceptual_threshold(1.0));
    }

    #[test]
    fn test_report_empty() {
        let report = ColorDiffReport::new(vec![], 2.3);
        assert!(report.is_empty());
        assert!(report.passes_threshold());
        assert!((report.avg_delta_e() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_report_worst_pair() {
        let small = ColorDifference::new(50.0, 0.0, 0.0, 50.5, 0.0, 0.0);
        let big = ColorDifference::new(0.0, 0.0, 0.0, 80.0, 50.0, -50.0);
        let report = ColorDiffReport::new(vec![small, big], 2.3);
        let worst = report.worst_pair().expect("worst pair should be found");
        assert!(worst.delta_e_2000() > 10.0);
    }

    #[test]
    fn test_report_avg_delta_e() {
        let d1 = ColorDifference::new(50.0, 0.0, 0.0, 50.0, 0.0, 0.0); // ~0
        let d2 = ColorDifference::new(50.0, 0.0, 0.0, 60.0, 0.0, 0.0); // ~10
        let report = ColorDiffReport::new(vec![d1, d2], 5.0);
        let avg = report.avg_delta_e();
        assert!(avg > 0.0 && avg < 15.0);
    }

    #[test]
    fn test_report_passes_threshold_true() {
        let d = ColorDifference::new(50.0, 0.0, 0.0, 50.0, 0.0, 0.0);
        let report = ColorDiffReport::new(vec![d], 1.0);
        assert!(report.passes_threshold());
    }

    #[test]
    fn test_report_passes_threshold_false() {
        let d = ColorDifference::new(0.0, 0.0, 0.0, 80.0, 50.0, -50.0);
        let report = ColorDiffReport::new(vec![d], 1.0);
        assert!(!report.passes_threshold());
    }

    #[test]
    fn test_report_fail_count() {
        let ok = ColorDifference::new(50.0, 0.0, 0.0, 50.0, 0.1, 0.0);
        let fail = ColorDifference::new(0.0, 0.0, 0.0, 80.0, 50.0, -50.0);
        let report = ColorDiffReport::new(vec![ok, fail], 1.0);
        assert_eq!(report.fail_count(), 1);
    }

    #[test]
    fn test_report_len() {
        let report = ColorDiffReport::new(vec![identical(), identical(), identical()], 2.3);
        assert_eq!(report.len(), 3);
    }

    #[test]
    fn test_report_max_min_identical() {
        let report = ColorDiffReport::new(vec![identical()], 1.0);
        assert!(report.max_delta_e() < 1e-6);
        assert!(report.min_delta_e() < 1e-6);
    }
}
