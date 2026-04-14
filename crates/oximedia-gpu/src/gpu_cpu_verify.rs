//! GPU vs CPU output comparison and verification utilities.
//!
//! Provides tools for verifying that GPU-accelerated operations produce
//! results within an acceptable tolerance of their CPU reference
//! implementations.
//!
//! # Tolerance Metrics
//!
//! | Metric | Description |
//! |--------|-------------|
//! | [`ToleranceMetric::MaxAbsDiff`] | Maximum absolute difference across all elements. |
//! | [`ToleranceMetric::MeanAbsDiff`] | Mean absolute difference. |
//! | [`ToleranceMetric::Psnr`] | Peak Signal-to-Noise Ratio (dB). |
//! | [`ToleranceMetric::RmsError`] | Root Mean Square error. |
//!
//! # Example
//!
//! ```rust
//! use oximedia_gpu::gpu_cpu_verify::{ComparisonResult, VerificationSuite, ToleranceMetric};
//!
//! let gpu_output = vec![128u8, 129, 127, 255];
//! let cpu_output = vec![128u8, 128, 128, 255];
//! let result = ComparisonResult::compare_u8(&gpu_output, &cpu_output);
//! assert!(result.max_abs_diff <= 1);
//! ```

// ─── Tolerance metric ──────────────────────────────────────────────────────

/// Which metric to use for pass/fail decisions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ToleranceMetric {
    /// Maximum absolute difference must be ≤ threshold.
    MaxAbsDiff(u32),
    /// Mean absolute difference must be ≤ threshold.
    MeanAbsDiff(f64),
    /// PSNR must be ≥ threshold (in dB).  Higher is better.
    Psnr(f64),
    /// RMS error must be ≤ threshold.
    RmsError(f64),
}

impl std::fmt::Display for ToleranceMetric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MaxAbsDiff(t) => write!(f, "max_abs_diff <= {t}"),
            Self::MeanAbsDiff(t) => write!(f, "mean_abs_diff <= {t:.4}"),
            Self::Psnr(t) => write!(f, "psnr >= {t:.2} dB"),
            Self::RmsError(t) => write!(f, "rms_error <= {t:.4}"),
        }
    }
}

// ─── ComparisonResult ───────────────────────────────────────────────────────

/// Detailed result of comparing two buffers element-by-element.
#[derive(Debug, Clone)]
pub struct ComparisonResult {
    /// Maximum absolute difference between any two corresponding elements.
    pub max_abs_diff: u32,
    /// Mean absolute difference.
    pub mean_abs_diff: f64,
    /// Peak Signal-to-Noise Ratio (dB).  `f64::INFINITY` if buffers are identical.
    pub psnr_db: f64,
    /// Root Mean Square error.
    pub rms_error: f64,
    /// Index of the element with the largest absolute difference.
    pub worst_index: usize,
    /// Total number of elements compared.
    pub element_count: usize,
    /// Number of elements that differ by at least 1.
    pub differing_elements: usize,
    /// Histogram of absolute differences (bin `i` = count of elements with diff == `i`).
    /// Limited to the first 256 bins.
    pub diff_histogram: [u64; 256],
}

impl ComparisonResult {
    /// Compare two u8 buffers element-by-element.
    ///
    /// If the buffers have different lengths, the comparison is performed
    /// over the shorter length.
    #[must_use]
    pub fn compare_u8(a: &[u8], b: &[u8]) -> Self {
        let len = a.len().min(b.len());
        if len == 0 {
            return Self::empty();
        }

        let mut max_diff: u32 = 0;
        let mut sum_diff: f64 = 0.0;
        let mut sum_sq_diff: f64 = 0.0;
        let mut worst_idx: usize = 0;
        let mut differing: usize = 0;
        let mut histogram = [0u64; 256];

        for i in 0..len {
            let diff = (a[i] as i32 - b[i] as i32).unsigned_abs();
            if diff > max_diff {
                max_diff = diff;
                worst_idx = i;
            }
            sum_diff += diff as f64;
            sum_sq_diff += (diff as f64) * (diff as f64);
            if diff > 0 {
                differing += 1;
            }
            let bin = (diff as usize).min(255);
            histogram[bin] += 1;
        }

        let n = len as f64;
        let mean_diff = sum_diff / n;
        let mse = sum_sq_diff / n;
        let rms = mse.sqrt();
        let psnr = if mse < 1e-12 {
            f64::INFINITY
        } else {
            10.0 * (255.0 * 255.0 / mse).log10()
        };

        Self {
            max_abs_diff: max_diff,
            mean_abs_diff: mean_diff,
            psnr_db: psnr,
            rms_error: rms,
            worst_index: worst_idx,
            element_count: len,
            differing_elements: differing,
            diff_histogram: histogram,
        }
    }

    /// Compare two f32 buffers element-by-element.
    ///
    /// Values are assumed to be in the range \[0.0, 1.0\].
    #[must_use]
    pub fn compare_f32(a: &[f32], b: &[f32]) -> Self {
        let len = a.len().min(b.len());
        if len == 0 {
            return Self::empty();
        }

        let mut max_diff: u32 = 0;
        let mut sum_diff: f64 = 0.0;
        let mut sum_sq_diff: f64 = 0.0;
        let mut worst_idx: usize = 0;
        let mut differing: usize = 0;
        let mut histogram = [0u64; 256];

        for i in 0..len {
            let diff_f = ((a[i] - b[i]) as f64).abs();
            let diff_u = (diff_f * 255.0).round() as u32;
            if diff_u > max_diff {
                max_diff = diff_u;
                worst_idx = i;
            }
            sum_diff += diff_f * 255.0;
            sum_sq_diff += diff_f * diff_f * 255.0 * 255.0;
            if diff_u > 0 {
                differing += 1;
            }
            let bin = (diff_u as usize).min(255);
            histogram[bin] += 1;
        }

        let n = len as f64;
        let mean_diff = sum_diff / n;
        let mse = sum_sq_diff / n;
        let rms = mse.sqrt();
        let psnr = if mse < 1e-12 {
            f64::INFINITY
        } else {
            10.0 * (255.0 * 255.0 / mse).log10()
        };

        Self {
            max_abs_diff: max_diff,
            mean_abs_diff: mean_diff,
            psnr_db: psnr,
            rms_error: rms,
            worst_index: worst_idx,
            element_count: len,
            differing_elements: differing,
            diff_histogram: histogram,
        }
    }

    /// Check whether this result passes a given tolerance metric.
    #[must_use]
    pub fn passes(&self, metric: &ToleranceMetric) -> bool {
        match metric {
            ToleranceMetric::MaxAbsDiff(t) => self.max_abs_diff <= *t,
            ToleranceMetric::MeanAbsDiff(t) => self.mean_abs_diff <= *t,
            ToleranceMetric::Psnr(t) => self.psnr_db >= *t,
            ToleranceMetric::RmsError(t) => self.rms_error <= *t,
        }
    }

    /// Percentage of elements that differ.
    #[must_use]
    pub fn diff_percentage(&self) -> f64 {
        if self.element_count == 0 {
            return 0.0;
        }
        (self.differing_elements as f64 / self.element_count as f64) * 100.0
    }

    /// Whether the two buffers are exactly identical.
    #[must_use]
    pub fn is_exact_match(&self) -> bool {
        self.max_abs_diff == 0
    }

    fn empty() -> Self {
        Self {
            max_abs_diff: 0,
            mean_abs_diff: 0.0,
            psnr_db: f64::INFINITY,
            rms_error: 0.0,
            worst_index: 0,
            element_count: 0,
            differing_elements: 0,
            diff_histogram: [0u64; 256],
        }
    }
}

// ─── VerificationReport ─────────────────────────────────────────────────────

/// A single test case result within a verification suite.
#[derive(Debug, Clone)]
pub struct VerificationCase {
    /// Human-readable name of the test case.
    pub name: String,
    /// The comparison result.
    pub result: ComparisonResult,
    /// The tolerance metric used.
    pub metric: ToleranceMetric,
    /// Whether this case passed.
    pub passed: bool,
}

/// A suite of GPU vs CPU verification tests.
#[derive(Debug, Clone)]
pub struct VerificationSuite {
    /// Name of the suite.
    pub name: String,
    /// Individual test cases.
    pub cases: Vec<VerificationCase>,
}

impl VerificationSuite {
    /// Create a new empty suite.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            cases: Vec::new(),
        }
    }

    /// Add a test case comparing two u8 buffers.
    pub fn add_u8_case(
        &mut self,
        name: impl Into<String>,
        gpu_output: &[u8],
        cpu_output: &[u8],
        metric: ToleranceMetric,
    ) {
        let result = ComparisonResult::compare_u8(gpu_output, cpu_output);
        let passed = result.passes(&metric);
        self.cases.push(VerificationCase {
            name: name.into(),
            result,
            metric,
            passed,
        });
    }

    /// Add a test case comparing two f32 buffers.
    pub fn add_f32_case(
        &mut self,
        name: impl Into<String>,
        gpu_output: &[f32],
        cpu_output: &[f32],
        metric: ToleranceMetric,
    ) {
        let result = ComparisonResult::compare_f32(gpu_output, cpu_output);
        let passed = result.passes(&metric);
        self.cases.push(VerificationCase {
            name: name.into(),
            result,
            metric,
            passed,
        });
    }

    /// Number of test cases.
    #[must_use]
    pub fn case_count(&self) -> usize {
        self.cases.len()
    }

    /// Number of passing test cases.
    #[must_use]
    pub fn passed_count(&self) -> usize {
        self.cases.iter().filter(|c| c.passed).count()
    }

    /// Number of failing test cases.
    #[must_use]
    pub fn failed_count(&self) -> usize {
        self.cases.iter().filter(|c| !c.passed).count()
    }

    /// Whether all test cases passed.
    #[must_use]
    pub fn all_passed(&self) -> bool {
        self.cases.iter().all(|c| c.passed)
    }

    /// Get all failing cases.
    #[must_use]
    pub fn failures(&self) -> Vec<&VerificationCase> {
        self.cases.iter().filter(|c| !c.passed).collect()
    }

    /// Generate a summary report as a string.
    #[must_use]
    pub fn summary(&self) -> String {
        let mut report = format!(
            "Verification Suite: {}\n  Cases: {} total, {} passed, {} failed\n",
            self.name,
            self.case_count(),
            self.passed_count(),
            self.failed_count(),
        );

        for case in &self.cases {
            let status = if case.passed { "PASS" } else { "FAIL" };
            report.push_str(&format!(
                "  [{status}] {} — max_diff={}, mean_diff={:.4}, psnr={:.2}dB, rms={:.4}\n",
                case.name,
                case.result.max_abs_diff,
                case.result.mean_abs_diff,
                case.result.psnr_db,
                case.result.rms_error,
            ));
        }

        report
    }
}

// ─── Convenience functions ──────────────────────────────────────────────────

/// Quick check: are two u8 buffers within `max_diff` of each other at every element?
#[must_use]
pub fn buffers_within_tolerance(a: &[u8], b: &[u8], max_diff: u32) -> bool {
    ComparisonResult::compare_u8(a, b).max_abs_diff <= max_diff
}

/// Quick check: compute PSNR between two u8 buffers.
#[must_use]
pub fn compute_buffer_psnr(a: &[u8], b: &[u8]) -> f64 {
    ComparisonResult::compare_u8(a, b).psnr_db
}

/// Quick check: compute RMS error between two u8 buffers.
#[must_use]
pub fn compute_buffer_rms(a: &[u8], b: &[u8]) -> f64 {
    ComparisonResult::compare_u8(a, b).rms_error
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical_buffers_exact_match() {
        let a = vec![0u8, 128, 255, 64, 192];
        let result = ComparisonResult::compare_u8(&a, &a);
        assert!(result.is_exact_match());
        assert_eq!(result.max_abs_diff, 0);
        assert_eq!(result.mean_abs_diff, 0.0);
        assert_eq!(result.psnr_db, f64::INFINITY);
        assert_eq!(result.rms_error, 0.0);
        assert_eq!(result.differing_elements, 0);
    }

    #[test]
    fn test_single_element_difference() {
        let a = vec![100u8];
        let b = vec![105u8];
        let result = ComparisonResult::compare_u8(&a, &b);
        assert_eq!(result.max_abs_diff, 5);
        assert_eq!(result.mean_abs_diff, 5.0);
        assert_eq!(result.differing_elements, 1);
        assert_eq!(result.worst_index, 0);
    }

    #[test]
    fn test_max_diff_is_worst_case() {
        let a = vec![0u8, 0, 0, 0];
        let b = vec![1u8, 2, 10, 3];
        let result = ComparisonResult::compare_u8(&a, &b);
        assert_eq!(result.max_abs_diff, 10);
        assert_eq!(result.worst_index, 2);
    }

    #[test]
    fn test_psnr_high_for_small_differences() {
        let a: Vec<u8> = (0..=255).collect();
        let b: Vec<u8> = (0..=255).map(|v: u8| v.saturating_add(1)).collect();
        let result = ComparisonResult::compare_u8(&a, &b);
        assert!(
            result.psnr_db > 40.0,
            "PSNR should be high for +-1 diff, got {}",
            result.psnr_db
        );
    }

    #[test]
    fn test_tolerance_max_abs_diff_pass() {
        let result = ComparisonResult::compare_u8(&[100], &[103]);
        assert!(result.passes(&ToleranceMetric::MaxAbsDiff(5)));
        assert!(!result.passes(&ToleranceMetric::MaxAbsDiff(2)));
    }

    #[test]
    fn test_tolerance_mean_abs_diff_pass() {
        let a = vec![100u8, 100, 100, 100];
        let b = vec![102u8, 98, 101, 99];
        let result = ComparisonResult::compare_u8(&a, &b);
        assert!(result.passes(&ToleranceMetric::MeanAbsDiff(2.0)));
    }

    #[test]
    fn test_tolerance_psnr_pass() {
        let a: Vec<u8> = vec![128; 1000];
        let b: Vec<u8> = vec![129; 1000];
        let result = ComparisonResult::compare_u8(&a, &b);
        assert!(result.passes(&ToleranceMetric::Psnr(40.0)));
    }

    #[test]
    fn test_tolerance_rms_error_pass() {
        let a = vec![100u8, 100, 100, 100];
        let b = vec![101u8, 99, 100, 100];
        let result = ComparisonResult::compare_u8(&a, &b);
        assert!(result.passes(&ToleranceMetric::RmsError(1.0)));
    }

    #[test]
    fn test_diff_percentage() {
        let a = vec![0u8, 0, 0, 0, 0, 0, 0, 0, 0, 0]; // 10 elements
        let b = vec![0u8, 1, 0, 1, 0, 0, 0, 0, 0, 0]; // 2 differ
        let result = ComparisonResult::compare_u8(&a, &b);
        assert!((result.diff_percentage() - 20.0).abs() < 1e-6);
    }

    #[test]
    fn test_empty_buffers() {
        let result = ComparisonResult::compare_u8(&[], &[]);
        assert!(result.is_exact_match());
        assert_eq!(result.element_count, 0);
        assert_eq!(result.diff_percentage(), 0.0);
    }

    #[test]
    fn test_compare_f32_identical() {
        let a = vec![0.0f32, 0.5, 1.0];
        let result = ComparisonResult::compare_f32(&a, &a);
        assert!(result.is_exact_match());
        assert_eq!(result.psnr_db, f64::INFINITY);
    }

    #[test]
    fn test_compare_f32_small_diff() {
        let a = vec![0.5f32, 0.5, 0.5];
        let b = vec![0.502f32, 0.498, 0.5];
        let result = ComparisonResult::compare_f32(&a, &b);
        assert!(
            result.max_abs_diff <= 2,
            "max_abs_diff={}",
            result.max_abs_diff
        );
    }

    #[test]
    fn test_verification_suite_all_pass() {
        let mut suite = VerificationSuite::new("test suite");
        let a = vec![100u8; 16];
        let b = vec![101u8; 16];
        suite.add_u8_case("close match", &a, &b, ToleranceMetric::MaxAbsDiff(2));
        assert!(suite.all_passed());
        assert_eq!(suite.passed_count(), 1);
        assert_eq!(suite.failed_count(), 0);
    }

    #[test]
    fn test_verification_suite_with_failure() {
        let mut suite = VerificationSuite::new("mixed");
        let a = vec![100u8; 16];
        let b = vec![110u8; 16];
        suite.add_u8_case("too different", &a, &b, ToleranceMetric::MaxAbsDiff(5));
        assert!(!suite.all_passed());
        assert_eq!(suite.failed_count(), 1);
        let failures = suite.failures();
        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0].name, "too different");
    }

    #[test]
    fn test_verification_suite_summary_format() {
        let mut suite = VerificationSuite::new("blur verification");
        suite.add_u8_case(
            "uniform image",
            &[128u8; 4],
            &[128u8; 4],
            ToleranceMetric::MaxAbsDiff(0),
        );
        let summary = suite.summary();
        assert!(summary.contains("blur verification"));
        assert!(summary.contains("PASS"));
        assert!(summary.contains("1 total"));
    }

    #[test]
    fn test_buffers_within_tolerance_convenience() {
        assert!(buffers_within_tolerance(&[100, 200], &[101, 199], 1));
        assert!(!buffers_within_tolerance(&[100, 200], &[110, 190], 5));
    }

    #[test]
    fn test_compute_buffer_psnr_convenience() {
        let psnr = compute_buffer_psnr(&[128; 100], &[128; 100]);
        assert_eq!(psnr, f64::INFINITY);
    }

    #[test]
    fn test_compute_buffer_rms_convenience() {
        let rms = compute_buffer_rms(&[128; 100], &[128; 100]);
        assert_eq!(rms, 0.0);
    }

    #[test]
    fn test_diff_histogram_correct() {
        let a = vec![10u8, 20, 30, 40, 50];
        let b = vec![10u8, 21, 32, 40, 53];
        let result = ComparisonResult::compare_u8(&a, &b);
        assert_eq!(result.diff_histogram[0], 2); // indices 0, 3
        assert_eq!(result.diff_histogram[1], 1); // index 1 (diff=1)
        assert_eq!(result.diff_histogram[2], 1); // index 2 (diff=2)
        assert_eq!(result.diff_histogram[3], 1); // index 4 (diff=3)
    }

    #[test]
    fn test_tolerance_metric_display() {
        let m = ToleranceMetric::MaxAbsDiff(5);
        assert!(format!("{m}").contains("max_abs_diff"));
        let m2 = ToleranceMetric::Psnr(40.0);
        assert!(format!("{m2}").contains("psnr"));
    }

    #[test]
    fn test_suite_f32_case() {
        let mut suite = VerificationSuite::new("f32 test");
        let a = vec![0.5f32, 0.5, 0.5];
        let b = vec![0.5f32, 0.5, 0.5];
        suite.add_f32_case("exact", &a, &b, ToleranceMetric::MaxAbsDiff(0));
        assert!(suite.all_passed());
    }

    #[test]
    fn test_different_length_buffers_uses_shorter() {
        let a = vec![100u8, 200, 150];
        let b = vec![100u8, 200];
        let result = ComparisonResult::compare_u8(&a, &b);
        assert_eq!(result.element_count, 2);
        assert!(result.is_exact_match());
    }
}
