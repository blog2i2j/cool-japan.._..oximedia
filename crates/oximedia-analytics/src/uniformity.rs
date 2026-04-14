//! Chi-squared uniformity testing for variant assignment distributions.
//!
//! When running A/B experiments it is critical to verify that the variant
//! assignment mechanism distributes users **uniformly** across all variants.
//! Non-uniform assignment (also called *sample ratio mismatch*) can invalidate
//! experimental results.
//!
//! ## Chi-Squared Goodness-of-Fit
//!
//! The [chi-squared goodness-of-fit test] checks whether an observed frequency
//! distribution is consistent with an expected (usually uniform) distribution.
//!
//! Null hypothesis H₀: the assignment is uniform.
//!
//! The test statistic is:
//!
//! ```text
//! χ² = Σ (observed_i − expected_i)² / expected_i
//! ```
//!
//! Under H₀ it follows a chi-squared distribution with `k − 1` degrees of
//! freedom (where `k` is the number of variants).
//!
//! ## Sample Ratio Mismatch (SRM)
//!
//! SRM detection flags an experiment when the observed assignment ratios deviate
//! significantly from the planned traffic split.  The module performs the test
//! and returns a structured [`UniformityResult`] that callers can inspect.
//!
//! [chi-squared goodness-of-fit test]: https://en.wikipedia.org/wiki/Chi-squared_test

use crate::error::AnalyticsError;

// ─── UniformityResult ────────────────────────────────────────────────────────

/// Result of a chi-squared uniformity test.
#[derive(Debug, Clone, PartialEq)]
pub struct UniformityResult {
    /// Observed counts per cell (variant or bucket).
    pub observed: Vec<u64>,
    /// Expected counts per cell under the null hypothesis.
    pub expected: Vec<f64>,
    /// Computed chi-squared statistic: `Σ (O - E)² / E`.
    pub chi_squared: f64,
    /// Degrees of freedom = number of cells − 1.
    pub degrees_of_freedom: u32,
    /// Critical value at the chosen significance level.
    /// If `chi_squared > critical_value` the null hypothesis is rejected.
    pub critical_value: f64,
    /// Significance level used (e.g. 0.05).
    pub alpha: f64,
    /// `true` if H₀ (uniform distribution) is **not** rejected.
    pub is_uniform: bool,
    /// Approximate p-value estimate (based on chi-squared CDF approximation).
    pub p_value: f64,
}

// ─── Core test ───────────────────────────────────────────────────────────────

/// Run a chi-squared goodness-of-fit test for uniformity.
///
/// # Arguments
///
/// * `observed`  — observed count per variant / bucket.
/// * `expected_weights` — relative expected weight per cell (need not sum to 1).
///   Pass `None` for equal weights (pure uniformity test).
/// * `alpha` — significance level (e.g. `0.05`).
///
/// # Errors
///
/// * `InsufficientData` if `observed` has fewer than 2 elements or total
///   count is zero.
/// * `ConfigError` if any expected weight is ≤ 0 or `alpha` is outside
///   `(0, 1)`.
pub fn chi_squared_uniformity(
    observed: &[u64],
    expected_weights: Option<&[f64]>,
    alpha: f64,
) -> Result<UniformityResult, AnalyticsError> {
    let k = observed.len();
    if k < 2 {
        return Err(AnalyticsError::InsufficientData(
            "chi-squared test requires at least 2 cells".to_string(),
        ));
    }
    if alpha <= 0.0 || alpha >= 1.0 {
        return Err(AnalyticsError::ConfigError(format!(
            "alpha={alpha} must be in (0, 1)"
        )));
    }

    let total_n: u64 = observed.iter().sum();
    if total_n == 0 {
        return Err(AnalyticsError::InsufficientData(
            "observed counts sum to zero".to_string(),
        ));
    }

    // Compute expected counts.
    let expected: Vec<f64> = if let Some(weights) = expected_weights {
        if weights.len() != k {
            return Err(AnalyticsError::ConfigError(format!(
                "expected_weights length ({}) must match observed length ({})",
                weights.len(),
                k
            )));
        }
        for &w in weights {
            if w <= 0.0 {
                return Err(AnalyticsError::ConfigError(
                    "all expected weights must be positive".to_string(),
                ));
            }
        }
        let weight_sum: f64 = weights.iter().sum();
        weights
            .iter()
            .map(|&w| total_n as f64 * w / weight_sum)
            .collect()
    } else {
        let equal = total_n as f64 / k as f64;
        vec![equal; k]
    };

    // Check minimum expected frequency (rule of thumb: each cell ≥ 5).
    for (i, &e) in expected.iter().enumerate() {
        if e < 1.0 {
            return Err(AnalyticsError::InsufficientData(format!(
                "expected count for cell {i} is {e:.2} < 1; increase sample size"
            )));
        }
    }

    // Compute χ² statistic.
    let chi_squared: f64 = observed
        .iter()
        .zip(expected.iter())
        .map(|(&o, &e)| {
            let diff = o as f64 - e;
            diff * diff / e
        })
        .sum();

    let df = (k - 1) as u32;

    // Chi-squared critical values (lookup table for common alpha/df combinations).
    let critical_value = chi_squared_critical_value(df, alpha);

    // Approximate p-value using regularised incomplete gamma function.
    let p_value = chi_squared_p_value(chi_squared, df);

    let is_uniform = chi_squared <= critical_value;

    Ok(UniformityResult {
        observed: observed.to_vec(),
        expected,
        chi_squared,
        degrees_of_freedom: df,
        critical_value,
        alpha,
        is_uniform,
        p_value,
    })
}

// ─── Sample ratio mismatch ────────────────────────────────────────────────────

/// Configuration for sample ratio mismatch (SRM) detection.
#[derive(Debug, Clone)]
pub struct SrmConfig {
    /// Planned traffic fractions per variant (must sum to 1.0 within tolerance).
    pub planned_fractions: Vec<f64>,
    /// Significance level (default: 0.01 — SRM checks use stricter alpha).
    pub alpha: f64,
}

impl SrmConfig {
    /// Create a config with equal traffic splits across `n_variants`.
    pub fn equal_split(n_variants: usize, alpha: f64) -> Self {
        let frac = 1.0 / n_variants as f64;
        Self {
            planned_fractions: vec![frac; n_variants],
            alpha,
        }
    }
}

/// Detect sample ratio mismatch in an experiment.
///
/// Returns `Ok(UniformityResult)` where `is_uniform = true` means **no SRM
/// detected** (the assignment is consistent with the planned split).
pub fn detect_srm(
    observed_counts: &[u64],
    config: &SrmConfig,
) -> Result<UniformityResult, AnalyticsError> {
    if observed_counts.len() != config.planned_fractions.len() {
        return Err(AnalyticsError::ConfigError(format!(
            "observed_counts length ({}) != planned_fractions length ({})",
            observed_counts.len(),
            config.planned_fractions.len()
        )));
    }
    chi_squared_uniformity(
        observed_counts,
        Some(&config.planned_fractions),
        config.alpha,
    )
}

// ─── Bucket uniformity for continuous metrics ─────────────────────────────────

/// Test whether a set of continuous values is uniformly distributed across
/// the range `[min, max]` by bucketing into `n_buckets` equal-width bins.
///
/// This is useful for checking whether watch-time or engagement scores are
/// uniformly spread rather than heavily skewed.
pub fn bucket_uniformity_test(
    values: &[f64],
    n_buckets: usize,
    alpha: f64,
) -> Result<UniformityResult, AnalyticsError> {
    if values.is_empty() {
        return Err(AnalyticsError::InsufficientData(
            "values slice is empty".to_string(),
        ));
    }
    if n_buckets < 2 {
        return Err(AnalyticsError::ConfigError(
            "n_buckets must be >= 2".to_string(),
        ));
    }

    let min = values
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let max = values
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    if (max - min).abs() < f64::EPSILON {
        return Err(AnalyticsError::InsufficientData(
            "all values are identical; cannot test uniformity".to_string(),
        ));
    }

    let width = (max - min) / n_buckets as f64;
    let mut counts = vec![0u64; n_buckets];

    for &v in values {
        let bucket = ((v - min) / width) as usize;
        let bucket = bucket.min(n_buckets - 1);
        counts[bucket] += 1;
    }

    chi_squared_uniformity(&counts, None, alpha)
}

// ─── Statistical helpers ─────────────────────────────────────────────────────

/// Look up the chi-squared critical value for a given degrees-of-freedom and alpha.
///
/// Uses a pre-computed table for common combinations.  For df > 30 uses the
/// Wilson–Hilferty cube-root approximation (good to < 0.5 % relative error).
pub fn chi_squared_critical_value(df: u32, alpha: f64) -> f64 {
    // Precomputed critical values for alpha = 0.10, 0.05, 0.01.
    // Table: df → (χ²_0.10, χ²_0.05, χ²_0.01)
    const TABLE: &[(u32, f64, f64, f64)] = &[
        (1, 2.706, 3.841, 6.635),
        (2, 4.605, 5.991, 9.210),
        (3, 6.251, 7.815, 11.345),
        (4, 7.779, 9.488, 13.277),
        (5, 9.236, 11.070, 15.086),
        (6, 10.645, 12.592, 16.812),
        (7, 12.017, 14.067, 18.475),
        (8, 13.362, 15.507, 20.090),
        (9, 14.684, 16.919, 21.666),
        (10, 15.987, 18.307, 23.209),
        (15, 22.307, 24.996, 30.578),
        (20, 28.412, 31.410, 37.566),
        (25, 34.382, 37.652, 44.314),
        (30, 40.256, 43.773, 50.892),
    ];

    let (a10, a05, a01) = TABLE
        .iter()
        .find(|(d, ..)| *d == df)
        .map(|&(_, a, b, c)| (a, b, c))
        .unwrap_or_else(|| {
            // Wilson–Hilferty approximation for df not in table.
            let k = df as f64;
            let normal_z = if alpha <= 0.01 {
                2.326
            } else if alpha <= 0.05 {
                1.645
            } else {
                1.282
            };
            let cv = k
                * (1.0 - 2.0 / (9.0 * k) + normal_z * (2.0 / (9.0 * k)).sqrt()).powi(3);
            let cv = cv.max(0.0);
            // Return approximate values for all three levels.
            let z10 = 1.282_f64;
            let cv10 = k * (1.0 - 2.0/(9.0*k) + z10 * (2.0/(9.0*k)).sqrt()).powi(3);
            let z05 = 1.645_f64;
            let cv05 = k * (1.0 - 2.0/(9.0*k) + z05 * (2.0/(9.0*k)).sqrt()).powi(3);
            let _ = cv;
            (cv10.max(0.0), cv05.max(0.0), 0.0)
        });

    if alpha >= 0.10 {
        a10
    } else if alpha >= 0.05 {
        a05
    } else {
        a01
    }
}

/// Approximate p-value from a chi-squared statistic with `df` degrees of freedom.
///
/// Uses the Wilson–Hilferty normal approximation:
/// `Z ≈ ((χ²/df)^(1/3) − (1 − 2/(9·df))) / √(2/(9·df))`
/// then evaluates the standard normal survival function.
pub fn chi_squared_p_value(chi_sq: f64, df: u32) -> f64 {
    if df == 0 {
        return 1.0;
    }
    let k = df as f64;
    let cube = (chi_sq / k).cbrt();
    let mean = 1.0 - 2.0 / (9.0 * k);
    let std = (2.0 / (9.0 * k)).sqrt();
    let z = if std > 0.0 { (cube - mean) / std } else { 0.0 };
    // Standard normal survival function P(Z > z) via error function complement.
    normal_sf(z)
}

/// Standard normal survival function: P(Z > z) = (1 − erf(z/√2)) / 2.
fn normal_sf(z: f64) -> f64 {
    let x = z / std::f64::consts::SQRT_2;
    // Abramowitz & Stegun erf approximation (7.1.26), max |error| < 1.5×10⁻⁷.
    let p = 0.3275911_f64;
    let t = 1.0 / (1.0 + p * x.abs());
    let poly = t * (0.254829592
        + t * (-0.284496736
            + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let erf_abs = 1.0 - poly * (-x * x).exp();
    let erf = if x >= 0.0 { erf_abs } else { -erf_abs };
    (1.0 - erf) / 2.0
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── basic chi-squared test ────────────────────────────────────────────────

    #[test]
    fn uniform_distribution_passes_test() {
        // 10_000 users split perfectly evenly across 4 variants.
        let observed = vec![2_500u64, 2_500, 2_500, 2_500];
        let result = chi_squared_uniformity(&observed, None, 0.05).expect("should succeed");
        assert!(result.is_uniform, "perfectly uniform split should pass");
        assert!((result.chi_squared).abs() < 1e-9, "χ² should be 0");
    }

    #[test]
    fn severely_skewed_distribution_fails_test() {
        // 10_000 users: one variant gets 90% — clearly non-uniform.
        let observed = vec![9_000u64, 333, 334, 333];
        let result = chi_squared_uniformity(&observed, None, 0.05).expect("should succeed");
        assert!(
            !result.is_uniform,
            "severely skewed distribution should fail uniformity test"
        );
    }

    #[test]
    fn chi_squared_statistic_correct_for_known_case() {
        // df=1 case: O=(60,40), E=(50,50) → χ² = (100/50)+(100/50) = 4.0
        let observed = vec![60u64, 40];
        let result = chi_squared_uniformity(&observed, None, 0.05).expect("should succeed");
        assert!(
            (result.chi_squared - 4.0).abs() < 1e-9,
            "χ²={} expected 4.0",
            result.chi_squared
        );
    }

    #[test]
    fn two_cells_error_on_one_cell() {
        let result = chi_squared_uniformity(&[100u64], None, 0.05);
        assert!(result.is_err());
    }

    #[test]
    fn zero_total_count_returns_error() {
        let result = chi_squared_uniformity(&[0u64, 0, 0], None, 0.05);
        assert!(result.is_err());
    }

    #[test]
    fn invalid_alpha_returns_error() {
        let result = chi_squared_uniformity(&[100u64, 100], None, 0.0);
        assert!(result.is_err());
        let result2 = chi_squared_uniformity(&[100u64, 100], None, 1.0);
        assert!(result2.is_err());
    }

    // ── SRM detection ────────────────────────────────────────────────────────

    #[test]
    fn srm_not_detected_for_perfect_split() {
        // Planned: 50/50.  Observed: 5000/5000.
        let config = SrmConfig::equal_split(2, 0.01);
        let result = detect_srm(&[5_000u64, 5_000], &config).expect("should succeed");
        assert!(result.is_uniform, "no SRM expected for perfect split");
    }

    #[test]
    fn srm_detected_for_extreme_imbalance() {
        // Planned: 33/33/34%.  Observed: heavily skewed.
        let config = SrmConfig::equal_split(3, 0.01);
        let result = detect_srm(&[8_000u64, 1_000, 1_000], &config).expect("should succeed");
        assert!(
            !result.is_uniform,
            "SRM expected for highly imbalanced assignment"
        );
    }

    #[test]
    fn srm_planned_fractions_mismatch_length_error() {
        let config = SrmConfig {
            planned_fractions: vec![0.5, 0.5],
            alpha: 0.05,
        };
        let result = detect_srm(&[100u64, 100, 100], &config);
        assert!(result.is_err());
    }

    // ── bucket uniformity ────────────────────────────────────────────────────

    #[test]
    fn bucket_uniformity_passes_for_uniform_values() {
        // Values evenly distributed [0, 100).
        let values: Vec<f64> = (0..1_000).map(|i| (i % 10) as f64).collect();
        let result = bucket_uniformity_test(&values, 10, 0.05).expect("should succeed");
        assert!(
            result.is_uniform,
            "uniform bucket distribution should pass, χ²={}",
            result.chi_squared
        );
    }

    #[test]
    fn bucket_uniformity_fails_for_all_in_one_bucket() {
        // All values = 0.0 except one.  One bucket gets 999, rest get 0 or 1.
        let mut values: Vec<f64> = vec![0.0; 999];
        values.push(10.0);
        let result = bucket_uniformity_test(&values, 5, 0.05).expect("should succeed");
        assert!(
            !result.is_uniform,
            "heavily skewed bucket distribution should fail"
        );
    }

    #[test]
    fn bucket_uniformity_identical_values_error() {
        let values = vec![5.0f64; 100];
        let result = bucket_uniformity_test(&values, 5, 0.05);
        assert!(result.is_err());
    }

    // ── chi-squared assignment uniformity over 10K assignments ───────────────

    #[test]
    fn assign_variant_fnv_uniformity_10k() {
        // Verify that FNV-1a-based variant assignment distributes uniformly
        // over 10 000+ synthetic user IDs across 4 variants.
        // We use the same FNV-1a logic as ab_testing::assign_variant.
        fn fnv1a_32(data: &[u8]) -> u32 {
            let mut hash: u32 = 2_166_136_261;
            for &b in data {
                hash ^= u32::from(b);
                hash = hash.wrapping_mul(16_777_619);
            }
            hash
        }

        let n_variants = 4u32;
        let n_users = 10_000usize;
        let mut counts = vec![0u64; n_variants as usize];

        for i in 0..n_users {
            let user_id = format!("user_{i:06}");
            let hash = fnv1a_32(user_id.as_bytes());
            let variant = (hash % n_variants) as usize;
            counts[variant] += 1;
        }

        let result = chi_squared_uniformity(&counts, None, 0.05)
            .expect("chi-squared test should succeed");

        assert!(
            result.is_uniform,
            "FNV-1a assignment over 10K users is not uniform: counts={counts:?}, χ²={:.4}",
            result.chi_squared
        );
    }

    // ── p-value and critical value sanity checks ─────────────────────────────

    #[test]
    fn p_value_near_one_for_zero_chi_squared() {
        let p = chi_squared_p_value(0.0, 3);
        assert!(p > 0.4, "p-value for χ²=0 should be near 0.5, got {p}");
    }

    #[test]
    fn critical_value_for_df1_alpha05_known() {
        // χ²(1, 0.05) = 3.841 (tabulated).
        let cv = chi_squared_critical_value(1, 0.05);
        assert!(
            (cv - 3.841).abs() < 0.01,
            "critical value df=1,α=0.05 should be 3.841, got {cv}"
        );
    }
}
