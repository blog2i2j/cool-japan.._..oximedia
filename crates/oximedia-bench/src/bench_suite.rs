#![allow(dead_code)]
//! High-level benchmark suite orchestration.
//!
//! Provides `BenchSuite` for collecting named benchmark cases, running them in
//! mock mode (pure computation, no I/O), and summarising throughput results.

use std::time::{Duration, Instant};

/// A single benchmark case with a name, a closure producing a mock timing, and
/// an optional baseline throughput (items/s) to compare against.
pub struct BenchCase {
    /// Human-readable name.
    pub name: String,
    /// Simulated throughput in items/second.
    pub simulated_throughput: f64,
    /// Optional baseline throughput for ratio calculation.
    pub baseline_throughput: Option<f64>,
}

impl BenchCase {
    /// Create a new benchmark case.
    pub fn new(name: impl Into<String>, simulated_throughput: f64) -> Self {
        Self {
            name: name.into(),
            simulated_throughput,
            baseline_throughput: None,
        }
    }

    /// Attach a baseline throughput so ratios can be computed.
    pub fn with_baseline(mut self, baseline: f64) -> Self {
        self.baseline_throughput = Some(baseline);
        self
    }

    /// Returns current/baseline throughput ratio, or `None` if no baseline set.
    ///
    /// Values > 1.0 mean the current run is faster than baseline.
    #[allow(clippy::cast_precision_loss)]
    pub fn throughput_ratio(&self) -> Option<f64> {
        self.baseline_throughput.map(|b| {
            if b == 0.0 {
                0.0
            } else {
                self.simulated_throughput / b
            }
        })
    }
}

/// Result produced after running a `BenchCase`.
#[derive(Debug, Clone)]
pub struct CaseResult {
    /// Name of the case.
    pub name: String,
    /// Measured (or simulated) throughput in items/second.
    pub throughput: f64,
    /// Wall-clock duration the mock run took.
    pub duration: Duration,
    /// Throughput ratio vs baseline (if available).
    pub throughput_ratio: Option<f64>,
}

/// Aggregated results for an entire suite run.
#[derive(Debug)]
pub struct BenchSuiteResult {
    /// Individual case results.
    pub cases: Vec<CaseResult>,
    /// Total wall-clock time for the suite.
    pub total_duration: Duration,
}

impl BenchSuiteResult {
    /// Return the result for the case with the highest throughput, or `None`.
    pub fn fastest(&self) -> Option<&CaseResult> {
        self.cases.iter().max_by(|a, b| {
            a.throughput
                .partial_cmp(&b.throughput)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Return the result for the case with the lowest throughput, or `None`.
    pub fn slowest(&self) -> Option<&CaseResult> {
        self.cases.iter().min_by(|a, b| {
            a.throughput
                .partial_cmp(&b.throughput)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Total number of cases.
    pub fn case_count(&self) -> usize {
        self.cases.len()
    }

    /// Mean throughput across all cases.
    #[allow(clippy::cast_precision_loss)]
    pub fn mean_throughput(&self) -> f64 {
        if self.cases.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.cases.iter().map(|c| c.throughput).sum();
        sum / self.cases.len() as f64
    }
}

/// A suite of benchmark cases that can be run collectively.
#[derive(Default)]
pub struct BenchSuite {
    name: String,
    cases: Vec<BenchCase>,
    /// Number of mock iterations to simulate per case (affects simulated duration).
    iterations: u32,
}

impl BenchSuite {
    /// Create a new, empty benchmark suite.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            cases: Vec::new(),
            iterations: 1,
        }
    }

    /// Set how many iterations each case simulates.
    pub fn with_iterations(mut self, n: u32) -> Self {
        self.iterations = n.max(1);
        self
    }

    /// Add a benchmark case to this suite.
    pub fn add_case(&mut self, case: BenchCase) {
        self.cases.push(case);
    }

    /// Run all cases in mock mode (no real I/O).
    ///
    /// Each case is timed using a real `Instant` around a trivial computation
    /// so that elapsed durations are non-zero.
    pub fn run_all_mock(&self) -> BenchSuiteResult {
        let suite_start = Instant::now();
        let mut results = Vec::with_capacity(self.cases.len());

        for case in &self.cases {
            let start = Instant::now();
            // Simulate work proportional to iterations.
            let mut _acc: u64 = 0;
            for i in 0..self.iterations {
                _acc = _acc.wrapping_add(u64::from(i));
            }
            let duration = start.elapsed();

            results.push(CaseResult {
                name: case.name.clone(),
                throughput: case.simulated_throughput,
                duration,
                throughput_ratio: case.throughput_ratio(),
            });
        }

        BenchSuiteResult {
            cases: results,
            total_duration: suite_start.elapsed(),
        }
    }

    /// Return a human-readable summary of the suite configuration.
    pub fn summary(&self) -> String {
        format!(
            "BenchSuite '{}': {} cases, {} iteration(s) each",
            self.name,
            self.cases.len(),
            self.iterations
        )
    }

    /// Number of registered cases.
    pub fn len(&self) -> usize {
        self.cases.len()
    }

    /// Whether the suite has no cases.
    pub fn is_empty(&self) -> bool {
        self.cases.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_suite() -> BenchSuite {
        let mut suite = BenchSuite::new("test_suite");
        suite.add_case(BenchCase::new("fast_codec", 120.0).with_baseline(100.0));
        suite.add_case(BenchCase::new("slow_codec", 40.0).with_baseline(100.0));
        suite.add_case(BenchCase::new("medium_codec", 80.0));
        suite
    }

    #[test]
    fn test_bench_case_throughput_ratio_with_baseline() {
        let c = BenchCase::new("enc", 150.0).with_baseline(100.0);
        let ratio = c.throughput_ratio();
        assert!(ratio.is_some());
        assert!((ratio.expect("test expectation failed") - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_bench_case_throughput_ratio_no_baseline() {
        let c = BenchCase::new("enc", 100.0);
        assert!(c.throughput_ratio().is_none());
    }

    #[test]
    fn test_bench_case_zero_baseline() {
        let c = BenchCase::new("enc", 100.0).with_baseline(0.0);
        assert_eq!(
            c.throughput_ratio()
                .expect("throughput_ratio should succeed"),
            0.0
        );
    }

    #[test]
    fn test_suite_add_case_and_len() {
        let suite = make_suite();
        assert_eq!(suite.len(), 3);
        assert!(!suite.is_empty());
    }

    #[test]
    fn test_suite_summary_contains_name() {
        let suite = make_suite();
        let s = suite.summary();
        assert!(s.contains("test_suite"));
        assert!(s.contains("3 cases"));
    }

    #[test]
    fn test_run_all_mock_case_count() {
        let suite = make_suite();
        let result = suite.run_all_mock();
        assert_eq!(result.cases.len(), 3);
    }

    #[test]
    fn test_run_all_mock_fastest() {
        let suite = make_suite();
        let result = suite.run_all_mock();
        let fastest = result.fastest().expect("should have fastest");
        assert_eq!(fastest.name, "fast_codec");
    }

    #[test]
    fn test_run_all_mock_slowest() {
        let suite = make_suite();
        let result = suite.run_all_mock();
        let slowest = result.slowest().expect("should have slowest");
        assert_eq!(slowest.name, "slow_codec");
    }

    #[test]
    fn test_run_all_mock_throughput_preserved() {
        let suite = make_suite();
        let result = suite.run_all_mock();
        let fast = result
            .cases
            .iter()
            .find(|c| c.name == "fast_codec")
            .expect("test expectation failed");
        assert!((fast.throughput - 120.0).abs() < 1e-9);
    }

    #[test]
    fn test_run_all_mock_ratio_present() {
        let suite = make_suite();
        let result = suite.run_all_mock();
        let fast = result
            .cases
            .iter()
            .find(|c| c.name == "fast_codec")
            .expect("test expectation failed");
        assert!(fast.throughput_ratio.is_some());
        assert!((fast.throughput_ratio.expect("test expectation failed") - 1.2).abs() < 1e-9);
    }

    #[test]
    fn test_run_all_mock_no_ratio_without_baseline() {
        let suite = make_suite();
        let result = suite.run_all_mock();
        let medium = result
            .cases
            .iter()
            .find(|c| c.name == "medium_codec")
            .expect("test expectation failed");
        assert!(medium.throughput_ratio.is_none());
    }

    #[test]
    fn test_mean_throughput() {
        let suite = make_suite();
        let result = suite.run_all_mock();
        let mean = result.mean_throughput();
        // (120 + 40 + 80) / 3 = 80
        assert!((mean - 80.0).abs() < 1e-9);
    }

    #[test]
    fn test_mean_throughput_empty() {
        let result = BenchSuiteResult {
            cases: vec![],
            total_duration: Duration::ZERO,
        };
        assert_eq!(result.mean_throughput(), 0.0);
    }

    #[test]
    fn test_fastest_slowest_none_when_empty() {
        let result = BenchSuiteResult {
            cases: vec![],
            total_duration: Duration::ZERO,
        };
        assert!(result.fastest().is_none());
        assert!(result.slowest().is_none());
    }

    #[test]
    fn test_suite_iterations() {
        let mut suite = BenchSuite::new("iter_test").with_iterations(10);
        suite.add_case(BenchCase::new("x", 50.0));
        let result = suite.run_all_mock();
        assert_eq!(result.case_count(), 1);
    }
}
