//! A/B testing framework for recommendation algorithms.
//!
//! This module provides the ability to run controlled experiments
//! comparing different recommendation variants (algorithms, parameters,
//! models) and determine statistically significant winners.

#![allow(dead_code)]

use std::collections::HashMap;
use uuid::Uuid;

/// Configuration for a single variant in an A/B test.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VariantConfig {
    /// Unique identifier for this variant
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Traffic allocation fraction (0.0-1.0)
    pub traffic_fraction: f64,
    /// Variant-specific parameters
    pub params: HashMap<String, String>,
    /// Whether this is the control group
    pub is_control: bool,
}

impl VariantConfig {
    /// Creates a new variant config.
    #[must_use]
    pub fn new(id: &str, name: &str, traffic_fraction: f64) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            traffic_fraction,
            params: HashMap::new(),
            is_control: false,
        }
    }

    /// Marks this variant as the control group.
    #[must_use]
    pub fn as_control(mut self) -> Self {
        self.is_control = true;
        self
    }

    /// Adds a parameter to the variant.
    #[must_use]
    pub fn with_param(mut self, key: &str, value: &str) -> Self {
        self.params.insert(key.to_string(), value.to_string());
        self
    }
}

/// Metric observation for a variant.
#[derive(Debug, Clone)]
struct Observation {
    /// Value observed (e.g., click-through rate, watch time)
    value: f64,
    /// Timestamp of the observation
    _timestamp: i64,
}

/// Tracks metrics and outcomes for an A/B test.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AbTestResult {
    /// Variant ID
    pub variant_id: String,
    /// Number of impressions (users exposed)
    pub impressions: u64,
    /// Number of conversions (clicks, watches, etc.)
    pub conversions: u64,
    /// Sum of metric values
    pub metric_sum: f64,
    /// Sum of squared metric values (for variance calculation)
    pub metric_sum_sq: f64,
}

impl AbTestResult {
    /// Creates a new empty result for a variant.
    #[must_use]
    pub fn new(variant_id: &str) -> Self {
        Self {
            variant_id: variant_id.to_string(),
            impressions: 0,
            conversions: 0,
            metric_sum: 0.0,
            metric_sum_sq: 0.0,
        }
    }

    /// Records an impression (user was shown this variant).
    pub fn record_impression(&mut self) {
        self.impressions += 1;
    }

    /// Records a conversion with a metric value.
    pub fn record_conversion(&mut self, value: f64) {
        self.conversions += 1;
        self.metric_sum += value;
        self.metric_sum_sq += value * value;
    }

    /// Computes the conversion rate.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn conversion_rate(&self) -> f64 {
        if self.impressions == 0 {
            return 0.0;
        }
        self.conversions as f64 / self.impressions as f64
    }

    /// Computes the mean metric value among conversions.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn mean_metric(&self) -> f64 {
        if self.conversions == 0 {
            return 0.0;
        }
        self.metric_sum / self.conversions as f64
    }

    /// Computes the variance of the metric values.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn metric_variance(&self) -> f64 {
        if self.conversions < 2 {
            return 0.0;
        }
        let n = self.conversions as f64;
        let mean = self.mean_metric();
        (self.metric_sum_sq / n) - (mean * mean)
    }

    /// Computes the standard error of the mean metric.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn standard_error(&self) -> f64 {
        if self.conversions < 2 {
            return 0.0;
        }
        (self.metric_variance() / self.conversions as f64).sqrt()
    }
}

/// Status of an A/B test.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum AbTestStatus {
    /// Test is configured but not yet running
    Draft,
    /// Test is actively collecting data
    Running,
    /// Test is paused
    Paused,
    /// Test has concluded
    Completed,
}

/// An A/B test comparing recommendation variants.
#[derive(Debug)]
pub struct AbTest {
    /// Unique test identifier
    pub id: Uuid,
    /// Human-readable name
    pub name: String,
    /// Test status
    pub status: AbTestStatus,
    /// Variants being tested
    variants: Vec<VariantConfig>,
    /// Results per variant
    results: HashMap<String, AbTestResult>,
    /// User-to-variant assignment (sticky)
    assignments: HashMap<Uuid, String>,
    /// Minimum observations per variant to declare a winner
    min_observations: u64,
    /// Significance level (e.g., 0.05 for 95% confidence)
    significance_level: f64,
}

impl AbTest {
    /// Creates a new A/B test.
    #[must_use]
    pub fn new(name: &str, variants: Vec<VariantConfig>) -> Self {
        let mut results = HashMap::new();
        for v in &variants {
            results.insert(v.id.clone(), AbTestResult::new(&v.id));
        }
        Self {
            id: Uuid::new_v4(),
            name: name.to_string(),
            status: AbTestStatus::Draft,
            variants,
            results,
            assignments: HashMap::new(),
            min_observations: 100,
            significance_level: 0.05,
        }
    }

    /// Sets the minimum observations needed per variant.
    #[must_use]
    pub fn with_min_observations(mut self, min: u64) -> Self {
        self.min_observations = min;
        self
    }

    /// Sets the significance level.
    #[must_use]
    pub fn with_significance_level(mut self, level: f64) -> Self {
        self.significance_level = level;
        self
    }

    /// Starts the test.
    pub fn start(&mut self) {
        self.status = AbTestStatus::Running;
    }

    /// Pauses the test.
    pub fn pause(&mut self) {
        self.status = AbTestStatus::Paused;
    }

    /// Completes the test.
    pub fn complete(&mut self) {
        self.status = AbTestStatus::Completed;
    }

    /// Assigns a user to a variant (sticky assignment).
    ///
    /// Uses deterministic hashing so the same user always gets
    /// the same variant for this test.
    #[must_use]
    pub fn assign_variant(&mut self, user_id: Uuid) -> &str {
        if let Some(variant_id) = self.assignments.get(&user_id) {
            // Find the variant and return a reference to variants vec
            for v in &self.variants {
                if v.id == *variant_id {
                    return &v.id;
                }
            }
        }

        // Deterministic assignment based on user_id hash
        let hash = Self::hash_user(user_id);
        let mut cumulative = 0.0_f64;
        let mut assigned_idx = 0;
        for (i, v) in self.variants.iter().enumerate() {
            cumulative += v.traffic_fraction;
            if hash < cumulative {
                assigned_idx = i;
                break;
            }
            if i == self.variants.len() - 1 {
                assigned_idx = i;
            }
        }

        let variant_id = self.variants[assigned_idx].id.clone();
        self.assignments.insert(user_id, variant_id);
        &self.variants[assigned_idx].id
    }

    /// Records an impression for a variant.
    pub fn record_impression(&mut self, variant_id: &str) {
        if let Some(result) = self.results.get_mut(variant_id) {
            result.record_impression();
        }
    }

    /// Records a conversion for a variant.
    pub fn record_conversion(&mut self, variant_id: &str, metric_value: f64) {
        if let Some(result) = self.results.get_mut(variant_id) {
            result.record_conversion(metric_value);
        }
    }

    /// Gets results for a specific variant.
    #[must_use]
    pub fn get_result(&self, variant_id: &str) -> Option<&AbTestResult> {
        self.results.get(variant_id)
    }

    /// Returns all variant results.
    #[must_use]
    pub fn all_results(&self) -> &HashMap<String, AbTestResult> {
        &self.results
    }

    /// Determines the winner among variants, if any.
    ///
    /// Returns `None` if not enough data or no statistically significant winner.
    /// Returns `Some(variant_id)` of the best-performing variant.
    #[must_use]
    pub fn winner(&self) -> Option<String> {
        // Check minimum observations
        for result in self.results.values() {
            if result.impressions < self.min_observations {
                return None;
            }
        }

        // Find the control variant
        let control = self.variants.iter().find(|v| v.is_control)?;
        let control_result = self.results.get(&control.id)?;
        let control_rate = control_result.conversion_rate();

        let mut best_variant: Option<String> = None;
        let mut best_lift = 0.0_f64;

        for v in &self.variants {
            if v.is_control {
                continue;
            }
            let result = self.results.get(&v.id)?;
            let variant_rate = result.conversion_rate();
            let lift = variant_rate - control_rate;

            // Simple z-test for proportions
            if self.is_significant(control_result, result) && lift > best_lift {
                best_lift = lift;
                best_variant = Some(v.id.clone());
            }
        }

        // If no treatment beats control significantly, control wins if it has data
        if best_variant.is_none() && control_result.impressions >= self.min_observations {
            return Some(control.id.clone());
        }

        best_variant
    }

    /// Performs a z-test for two proportions.
    #[allow(clippy::cast_precision_loss)]
    fn is_significant(&self, control: &AbTestResult, treatment: &AbTestResult) -> bool {
        let n1 = control.impressions as f64;
        let n2 = treatment.impressions as f64;
        if n1 == 0.0 || n2 == 0.0 {
            return false;
        }
        let p1 = control.conversion_rate();
        let p2 = treatment.conversion_rate();
        let p_pool = (control.conversions as f64 + treatment.conversions as f64) / (n1 + n2);

        if p_pool <= 0.0 || p_pool >= 1.0 {
            return false;
        }

        let se = (p_pool * (1.0 - p_pool) * (1.0 / n1 + 1.0 / n2)).sqrt();
        if se == 0.0 {
            return false;
        }

        let z = (p2 - p1).abs() / se;

        // z > 1.96 for ~95% confidence (two-tailed)
        let z_threshold = match () {
            () if self.significance_level <= 0.01 => 2.576,
            () if self.significance_level <= 0.05 => 1.960,
            () if self.significance_level <= 0.10 => 1.645,
            () => 1.282,
        };

        z > z_threshold
    }

    /// Simple hash of user ID to a value in [0, 1).
    #[allow(clippy::cast_precision_loss)]
    fn hash_user(user_id: Uuid) -> f64 {
        let bytes = user_id.as_bytes();
        let mut hash: u64 = 0;
        for &b in bytes {
            hash = hash.wrapping_mul(31).wrapping_add(u64::from(b));
        }
        (hash % 10000) as f64 / 10000.0
    }

    /// Returns the number of variants.
    #[must_use]
    pub fn variant_count(&self) -> usize {
        self.variants.len()
    }

    /// Returns total impressions across all variants.
    #[must_use]
    pub fn total_impressions(&self) -> u64 {
        self.results.values().map(|r| r.impressions).sum()
    }

    /// Perform a Pearson chi-squared test comparing click/no-click contingency
    /// tables for `control` and `treatment`.
    ///
    /// Returns `(chi2, significant)` where `chi2` is the test statistic and
    /// `significant` indicates whether it exceeds the critical value at the
    /// experiment's significance level with 1 degree of freedom.
    ///
    /// The chi-squared critical values used are:
    /// - α = 0.01 → χ² = 6.635
    /// - α = 0.05 → χ² = 3.841
    /// - α = 0.10 → χ² = 2.706
    /// - otherwise → χ² = 2.072
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn chi_squared_test(
        &self,
        control_id: &str,
        treatment_id: &str,
    ) -> Option<ChiSquaredResult> {
        let ctrl = self.results.get(control_id)?;
        let treat = self.results.get(treatment_id)?;

        let n1 = ctrl.impressions as f64;
        let n2 = treat.impressions as f64;
        if n1 == 0.0 || n2 == 0.0 {
            return None;
        }

        // Observed: [[ctrl_click, ctrl_no_click], [treat_click, treat_no_click]]
        let a = ctrl.conversions as f64; // ctrl clicks
        let b = n1 - a; // ctrl no-click
        let c = treat.conversions as f64; // treat clicks
        let d = n2 - c; // treat no-click

        if b < 0.0 || d < 0.0 {
            return None;
        }

        let n = n1 + n2;
        let row1 = a + b; // = n1
        let row2 = c + d; // = n2
        let col1 = a + c;
        let col2 = b + d;

        // Expected frequencies
        let e_a = row1 * col1 / n;
        let e_b = row1 * col2 / n;
        let e_c = row2 * col1 / n;
        let e_d = row2 * col2 / n;

        // Guard against zero expected frequencies
        if e_a < 1e-10 || e_b < 1e-10 || e_c < 1e-10 || e_d < 1e-10 {
            return None;
        }

        let chi2 = (a - e_a).powi(2) / e_a
            + (b - e_b).powi(2) / e_b
            + (c - e_c).powi(2) / e_c
            + (d - e_d).powi(2) / e_d;

        let critical = match () {
            () if self.significance_level <= 0.01 => 6.635,
            () if self.significance_level <= 0.05 => 3.841,
            () if self.significance_level <= 0.10 => 2.706,
            () => 2.072,
        };

        Some(ChiSquaredResult {
            chi2,
            degrees_of_freedom: 1,
            significant: chi2 > critical,
            critical_value: critical,
            control_rate: ctrl.conversion_rate(),
            treatment_rate: treat.conversion_rate(),
        })
    }

    /// Perform a Welch's t-test comparing the continuous metric distributions
    /// (e.g., watch time) of `control` and `treatment`.
    ///
    /// Returns `None` if either variant has fewer than 2 conversions.
    ///
    /// The t-statistic is:
    ///
    /// ```text
    /// t = (μ₁ − μ₂) / sqrt(s₁²/n₁ + s₂²/n₂)
    /// ```
    ///
    /// Degrees of freedom are approximated via the Welch–Satterthwaite equation.
    /// Critical values are taken from the t-distribution at large df (z-approx):
    /// - α = 0.01 → t = 2.576
    /// - α = 0.05 → t = 1.960
    /// - α = 0.10 → t = 1.645
    /// - otherwise → t = 1.282
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn welch_t_test(&self, control_id: &str, treatment_id: &str) -> Option<WelchTTestResult> {
        let ctrl = self.results.get(control_id)?;
        let treat = self.results.get(treatment_id)?;

        if ctrl.conversions < 2 || treat.conversions < 2 {
            return None;
        }

        let n1 = ctrl.conversions as f64;
        let n2 = treat.conversions as f64;
        let mu1 = ctrl.mean_metric();
        let mu2 = treat.mean_metric();
        let var1 = ctrl.metric_variance();
        let var2 = treat.metric_variance();

        let se_sq = var1 / n1 + var2 / n2;
        if se_sq < f64::EPSILON {
            return None;
        }

        let t = (mu1 - mu2).abs() / se_sq.sqrt();

        // Welch–Satterthwaite degrees of freedom
        let df_num = se_sq.powi(2);
        let df_den = (var1 / n1).powi(2) / (n1 - 1.0) + (var2 / n2).powi(2) / (n2 - 1.0);
        let df = if df_den > f64::EPSILON {
            df_num / df_den
        } else {
            (n1 + n2 - 2.0).max(1.0)
        };

        // Use large-sample z-approximation for critical value
        let critical = match () {
            () if self.significance_level <= 0.01 => 2.576,
            () if self.significance_level <= 0.05 => 1.960,
            () if self.significance_level <= 0.10 => 1.645,
            () => 1.282,
        };

        Some(WelchTTestResult {
            t_statistic: t,
            degrees_of_freedom: df,
            significant: t > critical,
            critical_value: critical,
            control_mean: mu1,
            treatment_mean: mu2,
            effect_size: (mu2 - mu1) / ((var1 + var2) / 2.0).sqrt().max(f64::EPSILON),
        })
    }
}

// ---------------------------------------------------------------------------
// Statistical test result types
// ---------------------------------------------------------------------------

/// Result of a Pearson chi-squared test for independence.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ChiSquaredResult {
    /// Computed chi-squared statistic.
    pub chi2: f64,
    /// Degrees of freedom (1 for a 2×2 contingency table).
    pub degrees_of_freedom: u32,
    /// Whether the result is statistically significant at the experiment's α level.
    pub significant: bool,
    /// Critical value used for the significance decision.
    pub critical_value: f64,
    /// Conversion rate of the control group.
    pub control_rate: f64,
    /// Conversion rate of the treatment group.
    pub treatment_rate: f64,
}

impl ChiSquaredResult {
    /// Absolute lift: treatment rate − control rate.
    #[must_use]
    pub fn absolute_lift(&self) -> f64 {
        self.treatment_rate - self.control_rate
    }

    /// Relative lift: (treatment − control) / control (undefined if control = 0).
    #[must_use]
    pub fn relative_lift(&self) -> Option<f64> {
        if self.control_rate.abs() < f64::EPSILON {
            return None;
        }
        Some(self.absolute_lift() / self.control_rate)
    }
}

/// Result of a Welch's t-test for two independent means.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WelchTTestResult {
    /// Computed t-statistic.
    pub t_statistic: f64,
    /// Welch–Satterthwaite degrees of freedom.
    pub degrees_of_freedom: f64,
    /// Whether the result is statistically significant at the experiment's α level.
    pub significant: bool,
    /// Critical value used for the significance decision.
    pub critical_value: f64,
    /// Mean metric value for the control group.
    pub control_mean: f64,
    /// Mean metric value for the treatment group.
    pub treatment_mean: f64,
    /// Cohen's d-like effect size: (μ₂ − μ₁) / pooled_std.
    pub effect_size: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_variants() -> Vec<VariantConfig> {
        vec![
            VariantConfig::new("control", "Control", 0.5).as_control(),
            VariantConfig::new("treatment", "Treatment A", 0.5),
        ]
    }

    #[test]
    fn test_variant_config_creation() {
        let v = VariantConfig::new("v1", "Variant 1", 0.5);
        assert_eq!(v.id, "v1");
        assert!((v.traffic_fraction - 0.5).abs() < f64::EPSILON);
        assert!(!v.is_control);
    }

    #[test]
    fn test_variant_config_as_control() {
        let v = VariantConfig::new("ctrl", "Control", 0.5).as_control();
        assert!(v.is_control);
    }

    #[test]
    fn test_variant_config_with_param() {
        let v = VariantConfig::new("v1", "V1", 0.5)
            .with_param("model", "v2")
            .with_param("threshold", "0.7");
        assert_eq!(v.params.len(), 2);
        assert_eq!(v.params.get("model").expect("should succeed in test"), "v2");
    }

    #[test]
    fn test_ab_test_result_empty() {
        let r = AbTestResult::new("v1");
        assert_eq!(r.impressions, 0);
        assert!((r.conversion_rate() - 0.0).abs() < f64::EPSILON);
        assert!((r.mean_metric() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_ab_test_result_conversion_rate() {
        let mut r = AbTestResult::new("v1");
        for _ in 0..100 {
            r.record_impression();
        }
        for _ in 0..25 {
            r.record_conversion(1.0);
        }
        assert!((r.conversion_rate() - 0.25).abs() < f64::EPSILON);
    }

    #[test]
    fn test_ab_test_result_mean_metric() {
        let mut r = AbTestResult::new("v1");
        r.record_conversion(2.0);
        r.record_conversion(4.0);
        r.record_conversion(6.0);
        assert!((r.mean_metric() - 4.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_ab_test_result_variance() {
        let mut r = AbTestResult::new("v1");
        r.record_conversion(10.0);
        r.record_conversion(10.0);
        r.record_conversion(10.0);
        assert!(r.metric_variance().abs() < f64::EPSILON);
    }

    #[test]
    fn test_ab_test_creation() {
        let test = AbTest::new("Test 1", make_variants());
        assert_eq!(test.variant_count(), 2);
        assert_eq!(test.status, AbTestStatus::Draft);
    }

    #[test]
    fn test_ab_test_lifecycle() {
        let mut test = AbTest::new("Test 1", make_variants());
        assert_eq!(test.status, AbTestStatus::Draft);
        test.start();
        assert_eq!(test.status, AbTestStatus::Running);
        test.pause();
        assert_eq!(test.status, AbTestStatus::Paused);
        test.complete();
        assert_eq!(test.status, AbTestStatus::Completed);
    }

    #[test]
    fn test_ab_test_assign_variant_sticky() {
        let mut test = AbTest::new("Test 1", make_variants());
        let u = Uuid::new_v4();
        let v1 = test.assign_variant(u).to_string();
        let v2 = test.assign_variant(u).to_string();
        assert_eq!(v1, v2, "same user should get same variant");
    }

    #[test]
    fn test_ab_test_record_and_get_result() {
        let mut test = AbTest::new("Test 1", make_variants());
        test.record_impression("control");
        test.record_impression("control");
        test.record_conversion("control", 1.0);
        let r = test.get_result("control").expect("should succeed in test");
        assert_eq!(r.impressions, 2);
        assert_eq!(r.conversions, 1);
    }

    #[test]
    fn test_ab_test_winner_insufficient_data() {
        let test = AbTest::new("Test 1", make_variants()).with_min_observations(100);
        assert!(test.winner().is_none());
    }

    #[test]
    fn test_ab_test_winner_significant_treatment() {
        let mut test = AbTest::new("Test 1", make_variants()).with_min_observations(10);
        // Control: 10% conversion
        for _ in 0..200 {
            test.record_impression("control");
        }
        for _ in 0..20 {
            test.record_conversion("control", 1.0);
        }
        // Treatment: 30% conversion (clearly better)
        for _ in 0..200 {
            test.record_impression("treatment");
        }
        for _ in 0..60 {
            test.record_conversion("treatment", 1.0);
        }
        let w = test.winner();
        assert_eq!(w, Some("treatment".to_string()));
    }

    #[test]
    fn test_total_impressions() {
        let mut test = AbTest::new("Test 1", make_variants());
        test.record_impression("control");
        test.record_impression("control");
        test.record_impression("treatment");
        assert_eq!(test.total_impressions(), 3);
    }

    #[test]
    fn test_standard_error() {
        let mut r = AbTestResult::new("v1");
        r.record_conversion(10.0);
        r.record_conversion(20.0);
        r.record_conversion(30.0);
        assert!(r.standard_error() > 0.0);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Chi-squared test
    // ─────────────────────────────────────────────────────────────────────────

    fn make_test_with_data(
        ctrl_imp: u64,
        ctrl_conv: u64,
        treat_imp: u64,
        treat_conv: u64,
        alpha: f64,
    ) -> AbTest {
        let variants = make_variants();
        let mut test = AbTest::new("chi2_test", variants)
            .with_min_observations(1)
            .with_significance_level(alpha);
        for _ in 0..ctrl_imp {
            test.record_impression("control");
        }
        for _ in 0..ctrl_conv {
            test.record_conversion("control", 1.0);
        }
        for _ in 0..treat_imp {
            test.record_impression("treatment");
        }
        for _ in 0..treat_conv {
            test.record_conversion("treatment", 1.0);
        }
        test
    }

    #[test]
    fn test_chi_squared_significant_difference() {
        // Control: 10% CTR (20/200), Treatment: 30% CTR (60/200)
        let test = make_test_with_data(200, 20, 200, 60, 0.05);
        let result = test
            .chi_squared_test("control", "treatment")
            .expect("should compute chi2");
        assert!(
            result.chi2 > 3.841,
            "chi2={} should exceed 3.841",
            result.chi2
        );
        assert!(result.significant, "should be significant");
    }

    #[test]
    fn test_chi_squared_no_difference() {
        // Control and treatment both 50% CTR
        let test = make_test_with_data(100, 50, 100, 50, 0.05);
        let result = test
            .chi_squared_test("control", "treatment")
            .expect("should compute chi2");
        assert!(
            result.chi2 < 3.841,
            "chi2={} should be below critical",
            result.chi2
        );
        assert!(!result.significant);
    }

    #[test]
    fn test_chi_squared_none_when_no_impressions() {
        let variants = make_variants();
        let test = AbTest::new("empty", variants);
        let result = test.chi_squared_test("control", "treatment");
        assert!(result.is_none());
    }

    #[test]
    fn test_chi_squared_result_absolute_lift() {
        let test = make_test_with_data(100, 10, 100, 30, 0.05);
        let result = test
            .chi_squared_test("control", "treatment")
            .expect("should compute chi2");
        let lift = result.absolute_lift();
        assert!((lift - 0.20).abs() < 1e-9, "lift={lift}");
    }

    #[test]
    fn test_chi_squared_result_relative_lift() {
        let test = make_test_with_data(100, 10, 100, 20, 0.05);
        let result = test
            .chi_squared_test("control", "treatment")
            .expect("should compute chi2");
        let rel = result.relative_lift().expect("control rate > 0");
        assert!(
            (rel - 1.0).abs() < 1e-9,
            "expected +100% rel lift, got {rel}"
        );
    }

    #[test]
    fn test_chi_squared_relative_lift_none_if_zero_control() {
        // control has zero conversions so control_rate = 0
        let test = make_test_with_data(100, 0, 100, 50, 0.05);
        let result = test
            .chi_squared_test("control", "treatment")
            .expect("should compute chi2");
        assert!(result.relative_lift().is_none());
    }

    #[test]
    fn test_chi_squared_unknown_variant_returns_none() {
        let test = make_test_with_data(100, 20, 100, 30, 0.05);
        assert!(test.chi_squared_test("control", "nonexistent").is_none());
        assert!(test.chi_squared_test("nonexistent", "treatment").is_none());
    }

    #[test]
    fn test_chi_squared_degrees_of_freedom() {
        let test = make_test_with_data(200, 40, 200, 80, 0.05);
        let result = test
            .chi_squared_test("control", "treatment")
            .expect("should compute chi2");
        assert_eq!(result.degrees_of_freedom, 1);
    }

    #[test]
    fn test_chi_squared_critical_value_for_alpha_001() {
        let test = make_test_with_data(200, 40, 200, 80, 0.01);
        let result = test
            .chi_squared_test("control", "treatment")
            .expect("should compute chi2");
        assert!((result.critical_value - 6.635).abs() < 1e-9);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Welch's t-test
    // ─────────────────────────────────────────────────────────────────────────

    fn make_test_with_metric_data(ctrl_values: &[f64], treat_values: &[f64], alpha: f64) -> AbTest {
        let variants = make_variants();
        let mut test = AbTest::new("t_test", variants).with_significance_level(alpha);
        for &v in ctrl_values {
            test.record_impression("control");
            test.record_conversion("control", v);
        }
        for &v in treat_values {
            test.record_impression("treatment");
            test.record_conversion("treatment", v);
        }
        test
    }

    #[test]
    fn test_welch_t_test_significant_means_different() {
        // Control: watch times ~10s, Treatment: watch times ~30s
        let ctrl: Vec<f64> = (0..50).map(|i| 10.0 + (i % 3) as f64 * 0.1).collect();
        let treat: Vec<f64> = (0..50).map(|i| 30.0 + (i % 3) as f64 * 0.1).collect();
        let test = make_test_with_metric_data(&ctrl, &treat, 0.05);
        let result = test
            .welch_t_test("control", "treatment")
            .expect("should compute t-test");
        assert!(
            result.significant,
            "t={} should be significant",
            result.t_statistic
        );
        assert!(result.t_statistic > 1.96);
    }

    #[test]
    fn test_welch_t_test_no_difference() {
        // Both groups identical metric values
        let values: Vec<f64> = (0..30).map(|i| 20.0 + (i % 5) as f64).collect();
        let test = make_test_with_metric_data(&values, &values, 0.05);
        let result = test
            .welch_t_test("control", "treatment")
            .expect("should compute t-test");
        // t should be 0 or near 0 since the distributions are identical
        assert!(result.t_statistic < f64::EPSILON);
        assert!(!result.significant);
    }

    #[test]
    fn test_welch_t_test_none_when_insufficient_conversions() {
        let variants = make_variants();
        let mut test = AbTest::new("small", variants);
        test.record_impression("control");
        test.record_conversion("control", 5.0);
        // treatment has only 1 conversion → not enough
        test.record_impression("treatment");
        test.record_conversion("treatment", 10.0);
        let result = test.welch_t_test("control", "treatment");
        assert!(result.is_none(), "should be None for n < 2");
    }

    #[test]
    fn test_welch_t_test_effect_size_direction() {
        // Treatment has higher mean → positive effect size.
        // Use varied data so variance > 0 (required for Welch t-test).
        let ctrl: Vec<f64> = (0..20).map(|i| 10.0 + (i % 4) as f64 * 0.5).collect();
        let treat: Vec<f64> = (0..20).map(|i| 15.0 + (i % 4) as f64 * 0.5).collect();
        let test = make_test_with_metric_data(&ctrl, &treat, 0.05);
        let result = test
            .welch_t_test("control", "treatment")
            .expect("should compute t-test");
        assert!(
            result.effect_size > 0.0,
            "treatment is better, effect_size should be positive"
        );
    }

    #[test]
    fn test_welch_t_test_degrees_of_freedom_positive() {
        let ctrl: Vec<f64> = (0..20).map(|i| 5.0 + i as f64 * 0.3).collect();
        let treat: Vec<f64> = (0..20).map(|i| 8.0 + i as f64 * 0.3).collect();
        let test = make_test_with_metric_data(&ctrl, &treat, 0.05);
        let result = test
            .welch_t_test("control", "treatment")
            .expect("should compute t-test");
        assert!(result.degrees_of_freedom > 0.0);
    }

    #[test]
    fn test_welch_t_test_critical_value_for_alpha_001() {
        // Use varied data so variance > 0
        let ctrl: Vec<f64> = (0..10).map(|i| 10.0 + i as f64 * 0.3).collect();
        let treat: Vec<f64> = (0..10).map(|i| 20.0 + i as f64 * 0.3).collect();
        let test = make_test_with_metric_data(&ctrl, &treat, 0.01);
        let result = test
            .welch_t_test("control", "treatment")
            .expect("should compute t-test");
        assert!((result.critical_value - 2.576).abs() < 1e-9);
    }

    #[test]
    fn test_welch_t_test_unknown_variant_returns_none() {
        let ctrl: Vec<f64> = vec![10.0; 10];
        let treat: Vec<f64> = vec![20.0; 10];
        let test = make_test_with_metric_data(&ctrl, &treat, 0.05);
        assert!(test.welch_t_test("control", "ghost").is_none());
        assert!(test.welch_t_test("ghost", "treatment").is_none());
    }
}
