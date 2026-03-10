//! Benchmark regression tracking system for `OxiMedia`.
//!
//! This module provides tools for detecting performance regressions across benchmark runs,
//! storing historical results, computing statistical baselines, and performing trend analysis.
//!
//! # Overview
//!
//! - [`BenchmarkRecord`]: A single benchmark result snapshot
//! - [`BenchmarkHistory`]: Persistent store of all records with baseline computation
//! - [`RegressionDetector`]: Statistically-sound regression detection via z-score
//! - [`TrendAnalysis`]: Linear regression over time to detect slow degradation
//!
//! # Submodules
//!
//! - [`types`]: All data structs and enums
//! - [`detector`]: [`RegressionDetector`] and [`BenchmarkHistory`]
//! - [`trend`]: Mann-Kendall trend test and [`TrendAnalyzer`]
//! - [`outliers`]: IQR-based [`OutlierDetector`]
//!
//! # Example
//!
//! ```
//! use oximedia_bench::regression::{
//!     BenchmarkRecord, BenchmarkHistory, RegressionDetector, RegressionKind,
//! };
//!
//! // Build some historical data
//! let mut history = BenchmarkHistory::new(100);
//! for i in 0..10u64 {
//!     history.add(BenchmarkRecord {
//!         name: "encode_av1".to_string(),
//!         timestamp: 1_700_000_000 + i * 3600,
//!         throughput_fps: 30.0 + (i as f64) * 0.1,
//!         latency_ms: 33.0,
//!         memory_bytes: 512_000_000,
//!         quality_score: 38.5,
//!         metadata: std::collections::HashMap::new(),
//!     });
//! }
//!
//! // Check current result for regression
//! let current = BenchmarkRecord {
//!     name: "encode_av1".to_string(),
//!     timestamp: 1_700_040_000,
//!     throughput_fps: 20.0,  // big drop!
//!     latency_ms: 55.0,
//!     memory_bytes: 512_000_000,
//!     quality_score: 38.4,
//!     metadata: std::collections::HashMap::new(),
//! };
//!
//! let detector = RegressionDetector::default();
//! // Pass the full history slice — detect filters by name internally.
//! let analysis = detector.detect(&current, &history.records);
//! println!("Regression kind: {:?}", analysis.fps_kind);
//! ```

#![allow(dead_code)]

pub mod detector;
pub mod outliers;
pub mod trend;
pub mod types;

// ─────────────────────────────────────────────────────────────────────────────
// Flat re-exports — keep the same public surface as the original flat file
// ─────────────────────────────────────────────────────────────────────────────

pub use detector::{BenchmarkHistory, RegressionDetector};
pub use outliers::OutlierDetector;
pub use trend::{MannKendallResult, MannKendallTrend, TrendAnalyzer};
pub use types::{
    BenchmarkBaseline, BenchmarkRecord, ConfidenceRegressionAnalysis, DetectorConfig,
    IqrOutlierResult, OutlierInfo, RegressionAnalysis, RegressionKind, Severity, TrendAnalysis,
};

// ─────────────────────────────────────────────────────────────────────────────
// Tests (split by section, kept in mod.rs as in the original file)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn make_record(name: &str, ts: u64, fps: f64, latency: f64, quality: f64) -> BenchmarkRecord {
        BenchmarkRecord {
            name: name.to_string(),
            timestamp: ts,
            throughput_fps: fps,
            latency_ms: latency,
            memory_bytes: 512_000_000,
            quality_score: quality,
            metadata: HashMap::new(),
        }
    }

    fn stable_history(n: usize) -> Vec<BenchmarkRecord> {
        (0..n)
            .map(|i| make_record("bench_a", 1_700_000_000 + i as u64 * 3600, 30.0, 33.0, 38.5))
            .collect()
    }

    // ── mean / std_dev ───────────────────────────────────────────────────────

    #[test]
    fn test_mean_basic() {
        assert_eq!(detector::mean(&[1.0, 2.0, 3.0, 4.0, 5.0]), 3.0);
    }

    #[test]
    fn test_mean_empty() {
        assert_eq!(detector::mean(&[]), 0.0);
    }

    #[test]
    fn test_sample_std_dev_known() {
        // Sample std-dev of [2,4,4,4,5,5,7,9] ≈ 2.138 (divides by n-1 = 7)
        // Population std-dev would be ~2.0 (divides by n = 8)
        let values = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let sd = detector::sample_std_dev(&values);
        assert!((sd - 2.138).abs() < 0.001, "expected ~2.138, got {sd}");
    }

    #[test]
    fn test_sample_std_dev_single() {
        assert_eq!(detector::sample_std_dev(&[42.0]), 0.0);
    }

    #[test]
    fn test_sample_std_dev_empty() {
        assert_eq!(detector::sample_std_dev(&[]), 0.0);
    }

    // ── z-score ──────────────────────────────────────────────────────────────

    #[test]
    fn test_z_score_zero_std() {
        assert_eq!(detector::z_score(10.0, 10.0, 0.0), 0.0);
    }

    #[test]
    fn test_z_score_two_sigma_below() {
        let z = detector::z_score(6.0, 10.0, 2.0);
        assert!((z - (-2.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_z_score_positive() {
        let z = detector::z_score(14.0, 10.0, 2.0);
        assert!((z - 2.0).abs() < f64::EPSILON);
    }

    // ── linear regression ────────────────────────────────────────────────────

    #[test]
    fn test_linear_regression_perfect() {
        let xs = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let ys = vec![1.0, 3.0, 5.0, 7.0, 9.0]; // y = 2x + 1
        let (slope, intercept, r2) = detector::linear_regression(&xs, &ys);
        assert!((slope - 2.0).abs() < 1e-9, "slope: {slope}");
        assert!((intercept - 1.0).abs() < 1e-9, "intercept: {intercept}");
        assert!((r2 - 1.0).abs() < 1e-9, "r²: {r2}");
    }

    #[test]
    fn test_linear_regression_flat() {
        let xs = vec![0.0, 1.0, 2.0, 3.0];
        let ys = vec![5.0, 5.0, 5.0, 5.0];
        let (slope, _intercept, r2) = detector::linear_regression(&xs, &ys);
        assert!(slope.abs() < f64::EPSILON);
        assert!((r2 - 1.0).abs() < f64::EPSILON); // perfect horizontal line
    }

    #[test]
    fn test_linear_regression_negative_slope() {
        let xs = vec![0.0, 1.0, 2.0, 3.0];
        let ys = vec![10.0, 8.0, 6.0, 4.0]; // y = -2x + 10
        let (slope, intercept, r2) = detector::linear_regression(&xs, &ys);
        assert!((slope - (-2.0)).abs() < 1e-9);
        assert!((intercept - 10.0).abs() < 1e-9);
        assert!((r2 - 1.0).abs() < 1e-9);
    }

    // ── confidence interval ───────────────────────────────────────────────────

    #[test]
    fn test_confidence_interval_95() {
        let samples = vec![10.0, 12.0, 11.0, 13.0, 10.5, 11.5, 12.5, 11.0, 10.0, 13.5];
        let (lo, hi) = RegressionDetector::confidence_interval(&samples, 0.95);
        assert!(lo < hi);
        let m = detector::mean(&samples);
        assert!(lo < m && m < hi);
    }

    #[test]
    fn test_confidence_interval_single() {
        let (lo, hi) = RegressionDetector::confidence_interval(&[42.0], 0.95);
        assert_eq!(lo, 42.0);
        assert_eq!(hi, 42.0);
    }

    #[test]
    fn test_confidence_interval_empty() {
        let (lo, hi) = RegressionDetector::confidence_interval(&[], 0.95);
        assert_eq!(lo, 0.0);
        assert_eq!(hi, 0.0);
    }

    // ── severity ─────────────────────────────────────────────────────────────

    #[test]
    fn test_severity_thresholds() {
        assert_eq!(Severity::from_percent(3.0), Severity::Minor);
        assert_eq!(Severity::from_percent(10.0), Severity::Minor);
        assert_eq!(Severity::from_percent(20.0), Severity::Moderate);
        assert_eq!(Severity::from_percent(35.0), Severity::Major);
        assert_eq!(Severity::from_percent(55.0), Severity::Critical);
    }

    #[test]
    fn test_severity_labels() {
        assert_eq!(Severity::Minor.label(), "minor");
        assert_eq!(Severity::Moderate.label(), "moderate");
        assert_eq!(Severity::Major.label(), "major");
        assert_eq!(Severity::Critical.label(), "critical");
    }

    // ── RegressionKind ───────────────────────────────────────────────────────

    #[test]
    fn test_regression_kind_predicates() {
        let reg = RegressionKind::Regression {
            percent: 20.0,
            severity: Severity::Moderate,
        };
        assert!(reg.is_regression());
        assert!(!reg.is_improvement());

        let imp = RegressionKind::Improvement { percent: 10.0 };
        assert!(imp.is_improvement());
        assert!(!imp.is_regression());

        assert!(!RegressionKind::Stable.is_regression());
        assert!(!RegressionKind::Stable.is_improvement());
    }

    // ── RegressionDetector – no history ──────────────────────────────────────

    #[test]
    fn test_detect_no_history() {
        let detector = RegressionDetector::default();
        let current = make_record("bench_a", 1_700_000_000, 30.0, 33.0, 38.5);
        let analysis = detector.detect(&current, &[]);
        assert_eq!(analysis.fps_kind, RegressionKind::Stable);
        assert!(!analysis.has_regression);
        assert_eq!(analysis.baseline.sample_count, 0);
    }

    // ── RegressionDetector – stable history ──────────────────────────────────

    #[test]
    fn test_detect_stable_within_noise() {
        let det = RegressionDetector::default();
        let history = stable_history(10);
        // Tiny deviation — within noise
        let current = make_record("bench_a", 1_700_040_000, 30.1, 33.0, 38.5);
        let analysis = det.detect(&current, &history);
        assert_eq!(analysis.fps_kind, RegressionKind::Stable);
        assert!(!analysis.has_regression);
    }

    // ── RegressionDetector – clear FPS regression ────────────────────────────

    #[test]
    fn test_detect_fps_regression() {
        let det = RegressionDetector::default();
        // History of stable 30 FPS with very small variance.
        let mut history: Vec<BenchmarkRecord> = (0..20)
            .map(|i| {
                make_record(
                    "bench_a",
                    1_700_000_000 + i as u64 * 3600,
                    30.0 + (i % 2) as f64 * 0.1, // ±0.1 fps jitter
                    33.0,
                    38.5,
                )
            })
            .collect();

        // Simulate a 40 % throughput drop.
        let current = make_record("bench_a", 1_700_080_000, 18.0, 33.0, 38.5);
        let analysis = det.detect(&current, &history);

        // Consume the borrow of `history` after calling detect.
        history.clear();
        let _ = history; // suppress unused warning

        assert!(
            analysis.fps_kind.is_regression(),
            "expected FPS regression, got {:?}",
            analysis.fps_kind
        );
        assert!(analysis.has_regression);

        if let RegressionKind::Regression { severity, .. } = analysis.fps_kind {
            assert!(
                matches!(severity, Severity::Major | Severity::Critical),
                "expected major/critical severity, got {severity:?}"
            );
        }
    }

    // ── RegressionDetector – FPS improvement ─────────────────────────────────

    #[test]
    fn test_detect_fps_improvement() {
        let det = RegressionDetector::default();
        let history: Vec<BenchmarkRecord> = (0..15)
            .map(|i| make_record("bench_b", 1_700_000_000 + i as u64 * 3600, 30.0, 33.0, 38.5))
            .collect();

        // 40 % throughput gain.
        let current = make_record("bench_b", 1_700_060_000, 42.0, 33.0, 38.5);
        let analysis = det.detect(&current, &history);
        assert!(
            analysis.fps_kind.is_improvement(),
            "expected improvement, got {:?}",
            analysis.fps_kind
        );
        assert!(!analysis.has_regression);
    }

    // ── RegressionDetector – latency regression ───────────────────────────────

    #[test]
    fn test_detect_latency_regression() {
        let det = RegressionDetector::default();
        let history: Vec<BenchmarkRecord> = (0..15)
            .map(|i| make_record("bench_c", 1_700_000_000 + i as u64 * 3600, 30.0, 33.0, 38.5))
            .collect();

        // Latency doubles — clear regression.
        let current = make_record("bench_c", 1_700_060_000, 30.0, 70.0, 38.5);
        let analysis = det.detect(&current, &history);
        assert!(
            analysis.latency_kind.is_regression(),
            "expected latency regression, got {:?}",
            analysis.latency_kind
        );
    }

    // ── RegressionDetector – quality regression ───────────────────────────────

    #[test]
    fn test_detect_quality_regression() {
        let det = RegressionDetector::default();
        let history: Vec<BenchmarkRecord> = (0..15)
            .map(|i| make_record("bench_d", 1_700_000_000 + i as u64 * 3600, 30.0, 33.0, 40.0))
            .collect();

        // Quality drops from 40 dB to 20 dB.
        let current = make_record("bench_d", 1_700_060_000, 30.0, 33.0, 20.0);
        let analysis = det.detect(&current, &history);
        assert!(
            analysis.quality_kind.is_regression(),
            "expected quality regression, got {:?}",
            analysis.quality_kind
        );
    }

    // ── worst_severity ────────────────────────────────────────────────────────

    #[test]
    fn test_worst_severity_none() {
        let analysis = RegressionAnalysis {
            name: "x".to_string(),
            fps_kind: RegressionKind::Stable,
            latency_kind: RegressionKind::Stable,
            quality_kind: RegressionKind::Stable,
            fps_z_score: 0.0,
            latency_z_score: 0.0,
            baseline: BenchmarkBaseline {
                mean_fps: 30.0,
                std_fps: 1.0,
                mean_latency_ms: 33.0,
                std_latency_ms: 1.0,
                mean_quality: 38.5,
                std_quality: 0.5,
                sample_count: 10,
            },
            has_regression: false,
        };
        assert!(analysis.worst_severity().is_none());
    }

    #[test]
    fn test_worst_severity_picks_highest() {
        let analysis = RegressionAnalysis {
            name: "x".to_string(),
            fps_kind: RegressionKind::Regression {
                percent: 10.0,
                severity: Severity::Minor,
            },
            latency_kind: RegressionKind::Regression {
                percent: 35.0,
                severity: Severity::Major,
            },
            quality_kind: RegressionKind::Stable,
            fps_z_score: -3.0,
            latency_z_score: 4.0,
            baseline: BenchmarkBaseline {
                mean_fps: 30.0,
                std_fps: 1.0,
                mean_latency_ms: 33.0,
                std_latency_ms: 1.0,
                mean_quality: 38.5,
                std_quality: 0.5,
                sample_count: 10,
            },
            has_regression: true,
        };
        assert_eq!(analysis.worst_severity(), Some(Severity::Major));
    }

    // ── BenchmarkHistory ──────────────────────────────────────────────────────

    #[test]
    fn test_history_add_and_len() {
        let mut h = BenchmarkHistory::new(5);
        assert!(h.is_empty());
        h.add(make_record("a", 1, 30.0, 33.0, 38.5));
        h.add(make_record("b", 2, 25.0, 40.0, 35.0));
        assert_eq!(h.len(), 2);
    }

    #[test]
    fn test_history_eviction() {
        let mut h = BenchmarkHistory::new(3);
        for i in 0..6u64 {
            h.add(make_record("a", i, 30.0 + i as f64, 33.0, 38.5));
        }
        assert_eq!(h.len(), 3);
        // Oldest records should have been dropped; remaining FPS are 33, 34, 35.
        let fps: Vec<f64> = h.records.iter().map(|r| r.throughput_fps).collect();
        assert_eq!(fps, vec![33.0, 34.0, 35.0]);
    }

    #[test]
    fn test_history_baseline() {
        let mut h = BenchmarkHistory::new(100);
        for i in 0..10u64 {
            h.add(make_record("x", i, 30.0, 33.0, 38.5));
        }
        let b = h.baseline("x", 5).expect("b should be valid");
        assert_eq!(b.sample_count, 5);
        assert!((b.mean_fps - 30.0).abs() < f64::EPSILON);
        assert!((b.mean_latency_ms - 33.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_history_baseline_missing_name() {
        let h = BenchmarkHistory::new(100);
        assert!(h.baseline("nonexistent", 10).is_none());
    }

    #[test]
    fn test_history_trend_increasing() {
        let mut h = BenchmarkHistory::new(100);
        for i in 0..10u64 {
            h.add(make_record("y", i, 20.0 + i as f64 * 2.0, 33.0, 38.5));
        }
        let t = h.trend("y").expect("t should be valid");
        assert!(t.slope_fps_per_run > 0.0, "slope should be positive");
        assert!(!t.is_trending_down);
        assert!(t.r_squared > 0.99);
    }

    #[test]
    fn test_history_trend_decreasing() {
        let mut h = BenchmarkHistory::new(100);
        for i in 0..8u64 {
            h.add(make_record("z", i, 40.0 - i as f64 * 1.5, 33.0, 38.5));
        }
        let t = h.trend("z").expect("t should be valid");
        assert!(t.slope_fps_per_run < 0.0, "slope should be negative");
        assert!(t.is_trending_down);
    }

    #[test]
    fn test_history_trend_too_few_records() {
        let mut h = BenchmarkHistory::new(100);
        h.add(make_record("w", 1, 30.0, 33.0, 38.5));
        assert!(h.trend("w").is_none());
    }

    #[test]
    fn test_history_benchmark_names() {
        let mut h = BenchmarkHistory::new(100);
        h.add(make_record("encode_av1", 1, 30.0, 33.0, 38.5));
        h.add(make_record("encode_vp9", 2, 25.0, 40.0, 35.0));
        h.add(make_record("encode_av1", 3, 31.0, 33.0, 38.6));
        let names = h.benchmark_names();
        assert_eq!(names, vec!["encode_av1", "encode_vp9"]);
    }

    // ── JSON round-trip ───────────────────────────────────────────────────────

    #[test]
    fn test_history_json_roundtrip() {
        let mut h = BenchmarkHistory::new(50);
        let mut meta = HashMap::new();
        meta.insert("git_sha".to_string(), "abc123".to_string());
        h.add(BenchmarkRecord {
            name: "bench_json".to_string(),
            timestamp: 1_700_000_000,
            throughput_fps: 29.97,
            latency_ms: 33.37,
            memory_bytes: 1_024_000,
            quality_score: 42.1,
            metadata: meta,
        });

        let json = h.to_json();
        let restored = BenchmarkHistory::from_json(&json).expect("restored should be valid");
        assert_eq!(restored.len(), 1);
        let r = &restored.records[0];
        assert_eq!(r.name, "bench_json");
        assert!((r.throughput_fps - 29.97).abs() < 1e-9);
        assert_eq!(r.metadata.get("git_sha"), Some(&"abc123".to_string()));
    }

    #[test]
    fn test_history_from_json_invalid() {
        assert!(BenchmarkHistory::from_json("not valid json {{").is_err());
    }

    // ── BenchmarkRecord::simple ───────────────────────────────────────────────

    #[test]
    fn test_record_simple_constructor() {
        let r = BenchmarkRecord::simple("my_bench", 12345, 60.0, 16.7);
        assert_eq!(r.name, "my_bench");
        assert_eq!(r.timestamp, 12345);
        assert!((r.throughput_fps - 60.0).abs() < f64::EPSILON);
        assert!((r.latency_ms - 16.7).abs() < f64::EPSILON);
        assert!(r.metadata.is_empty());
    }

    // ── DetectorConfig ────────────────────────────────────────────────────────

    #[test]
    fn test_detector_with_strict_config() {
        let config = DetectorConfig {
            z_score_threshold: 1.0, // very sensitive
            min_regression_percent: 1.0,
            confidence_level: 0.90,
        };
        let det = RegressionDetector::with_config(config);

        let history: Vec<BenchmarkRecord> = (0..20)
            .map(|i| make_record("strict", 1_700_000_000 + i as u64 * 3600, 30.0, 33.0, 38.5))
            .collect();

        // Even a small dip triggers a regression with the strict config.
        let current = make_record("strict", 1_700_080_000, 27.0, 33.0, 38.5);
        let analysis = det.detect(&current, &history);
        assert!(
            analysis.fps_kind.is_regression(),
            "strict detector should flag 10% drop: {:?}",
            analysis.fps_kind
        );
    }

    #[test]
    fn test_detector_with_lenient_config() {
        let config = DetectorConfig {
            z_score_threshold: 4.0, // very lenient
            min_regression_percent: 40.0,
            confidence_level: 0.99,
        };
        let det = RegressionDetector::with_config(config);

        let history: Vec<BenchmarkRecord> = (0..20)
            .map(|i| make_record("lenient", 1_700_000_000 + i as u64 * 3600, 30.0, 33.0, 38.5))
            .collect();

        // A moderate 15% drop should not trigger with very lenient thresholds.
        let current = make_record("lenient", 1_700_080_000, 25.5, 33.0, 38.5);
        let analysis = det.detect(&current, &history);
        assert!(
            !analysis.fps_kind.is_regression(),
            "lenient detector should not flag 15% drop: {:?}",
            analysis.fps_kind
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Enhanced tests (confidence, Mann-Kendall, IQR outliers)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod enhanced_tests {
    use super::*;
    use std::collections::HashMap;

    fn make_record(name: &str, ts: u64, fps: f64, latency: f64, quality: f64) -> BenchmarkRecord {
        BenchmarkRecord {
            name: name.to_string(),
            timestamp: ts,
            throughput_fps: fps,
            latency_ms: latency,
            memory_bytes: 512_000_000,
            quality_score: quality,
            metadata: HashMap::new(),
        }
    }

    // ── detect_with_confidence ────────────────────────────────────────────────

    #[test]
    fn test_detect_with_confidence_stable() {
        let det = RegressionDetector::default();
        // Use slightly varying data so std_dev > 0, which produces a non-degenerate CI.
        let history: Vec<BenchmarkRecord> = (0..20)
            .map(|i| {
                make_record(
                    "bench_ci",
                    1_700_000_000 + i as u64 * 3600,
                    29.5 + (i % 3) as f64 * 0.5,
                    33.0,
                    38.5,
                )
            })
            .collect();
        let current = make_record("bench_ci", 1_700_080_000, 29.8, 33.1, 38.4);
        let result = det.detect_with_confidence(&current, &history);
        // Small deviation — within CI
        assert_eq!(result.confidence_level, 0.95);
        let (lo, hi) = result.fps_ci;
        assert!(lo < hi, "CI lower should be < upper (got lo={lo}, hi={hi})");
    }

    #[test]
    fn test_detect_with_confidence_regression_outside_ci() {
        let det = RegressionDetector::default();
        // Tight distribution: all exactly 30.0 FPS
        let history: Vec<BenchmarkRecord> = (0..20)
            .map(|i| {
                make_record(
                    "bench_ci2",
                    1_700_000_000 + i as u64 * 3600,
                    30.0,
                    33.0,
                    38.5,
                )
            })
            .collect();
        // Big drop — should be outside CI
        let current = make_record("bench_ci2", 1_700_080_000, 10.0, 33.0, 38.5);
        let result = det.detect_with_confidence(&current, &history);
        assert!(
            result.fps_outside_ci || result.core.fps_kind.is_regression(),
            "Expected CI breach or regression flag"
        );
    }

    #[test]
    fn test_detect_with_confidence_no_history() {
        let det = RegressionDetector::default();
        let current = make_record("new_bench", 1_700_000_000, 30.0, 33.0, 38.5);
        let result = det.detect_with_confidence(&current, &[]);
        assert_eq!(result.fps_ci.0, 30.0);
        assert_eq!(result.fps_ci.1, 30.0);
        assert!(!result.fps_outside_ci);
    }

    #[test]
    fn test_detect_with_confidence_uses_config_level() {
        let config = DetectorConfig {
            z_score_threshold: 2.0,
            min_regression_percent: 5.0,
            confidence_level: 0.99,
        };
        let det = RegressionDetector::with_config(config);
        let history: Vec<BenchmarkRecord> = (0..20)
            .map(|i| {
                make_record(
                    "bench_99",
                    1_700_000_000 + i as u64 * 3600,
                    30.0 + i as f64 * 0.1,
                    33.0,
                    38.5,
                )
            })
            .collect();
        let current = make_record("bench_99", 1_700_080_000, 32.0, 33.0, 38.5);
        let result = det.detect_with_confidence(&current, &history);
        assert_eq!(result.confidence_level, 0.99);
        // CI for 0.99 is wider than for 0.95
        let (lo, hi) = result.fps_ci;
        assert!(hi - lo > 0.0);
    }

    // ── Mann-Kendall test ──────────────────────────────────────────────────────

    #[test]
    fn test_mann_kendall_increasing_trend() {
        let analyzer = TrendAnalyzer::new();
        let history: Vec<BenchmarkRecord> = (0..20)
            .map(|i| {
                make_record(
                    "mk_inc",
                    1_700_000_000 + i as u64 * 3600,
                    10.0 + i as f64,
                    33.0,
                    38.5,
                )
            })
            .collect();
        let result = analyzer
            .mann_kendall_test("mk_inc", &history)
            .expect("result should be valid");
        assert_eq!(result.trend, MannKendallTrend::Increasing);
        assert!(result.significant);
        assert!(result.s_statistic > 0.0);
        assert!(result.sens_slope > 0.0);
    }

    #[test]
    fn test_mann_kendall_decreasing_trend() {
        let analyzer = TrendAnalyzer::new();
        let history: Vec<BenchmarkRecord> = (0..20)
            .map(|i| {
                make_record(
                    "mk_dec",
                    1_700_000_000 + i as u64 * 3600,
                    50.0 - i as f64 * 1.5,
                    33.0,
                    38.5,
                )
            })
            .collect();
        let result = analyzer
            .mann_kendall_test("mk_dec", &history)
            .expect("result should be valid");
        assert_eq!(result.trend, MannKendallTrend::Decreasing);
        assert!(result.significant);
        assert!(result.s_statistic < 0.0);
        assert!(result.sens_slope < 0.0);
    }

    #[test]
    fn test_mann_kendall_no_trend_constant() {
        let analyzer = TrendAnalyzer::new();
        let history: Vec<BenchmarkRecord> = (0..15)
            .map(|i| make_record("mk_flat", 1_700_000_000 + i as u64 * 3600, 30.0, 33.0, 38.5))
            .collect();
        let result = analyzer
            .mann_kendall_test("mk_flat", &history)
            .expect("result should be valid");
        assert_eq!(result.trend, MannKendallTrend::NoTrend);
        assert!(!result.significant);
    }

    #[test]
    fn test_mann_kendall_not_enough_data() {
        let analyzer = TrendAnalyzer::new();
        let history: Vec<BenchmarkRecord> = (0..3)
            .map(|i| make_record("mk_few", 1_700_000_000 + i as u64, 30.0, 33.0, 38.5))
            .collect();
        assert!(analyzer.mann_kendall_test("mk_few", &history).is_none());
    }

    #[test]
    fn test_mann_kendall_values_directly() {
        let analyzer = TrendAnalyzer::new();
        let values: Vec<f64> = (0..15).map(|i| i as f64 * 2.0).collect();
        let result = analyzer
            .mann_kendall_values(&values)
            .expect("result should be valid");
        assert_eq!(result.trend, MannKendallTrend::Increasing);
        assert!(result.n == 15);
    }

    #[test]
    fn test_mann_kendall_p_value_range() {
        let analyzer = TrendAnalyzer::new();
        let values: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let result = analyzer
            .mann_kendall_values(&values)
            .expect("result should be valid");
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn test_mann_kendall_sens_slope_constant() {
        let analyzer = TrendAnalyzer::new();
        let values = vec![5.0; 10];
        let result = analyzer
            .mann_kendall_values(&values)
            .expect("result should be valid");
        assert!(result.sens_slope.abs() < f64::EPSILON);
    }

    // ── IQR outlier detection ─────────────────────────────────────────────────

    #[test]
    fn test_iqr_no_outliers() {
        let det = OutlierDetector::new();
        let values: Vec<f64> = (0..20).map(|i| 30.0 + i as f64 * 0.1).collect();
        let result = det.iqr_method(&values).expect("result should be valid");
        assert!(
            !result.has_outliers(),
            "Uniformly distributed data should have no outliers"
        );
        assert!(result.iqr > 0.0);
    }

    #[test]
    fn test_iqr_detects_high_outlier() {
        let det = OutlierDetector::new();
        let mut values: Vec<f64> = vec![30.0; 20];
        values[5] = 1000.0; // Massive outlier
        let result = det.iqr_method(&values).expect("result should be valid");
        assert!(result.has_outliers(), "Should detect the 1000.0 outlier");
        assert!(result
            .outliers
            .iter()
            .any(|o| (o.value - 1000.0).abs() < 1.0));
    }

    #[test]
    fn test_iqr_detects_low_outlier() {
        let det = OutlierDetector::new();
        let mut values: Vec<f64> = vec![30.0; 20];
        values[10] = -500.0; // Very low outlier
        let result = det.iqr_method(&values).expect("result should be valid");
        assert!(result.has_outliers(), "Should detect the -500.0 outlier");
        assert!(result
            .outliers
            .iter()
            .any(|o| (o.value - (-500.0)).abs() < 1.0));
    }

    #[test]
    fn test_iqr_extreme_vs_mild() {
        let det = OutlierDetector::new();
        let mut values: Vec<f64> = (0..20).map(|_| 30.0).collect();
        values[0] = 200.0; // Extreme outlier (well beyond 3*IQR)
        let result = det.iqr_method(&values).expect("result should be valid");
        let extreme: Vec<_> = result.outliers.iter().filter(|o| o.is_extreme).collect();
        assert!(!extreme.is_empty(), "Expected at least one extreme outlier");
    }

    #[test]
    fn test_iqr_too_few_values() {
        let det = OutlierDetector::new();
        let values = vec![1.0, 2.0, 3.0]; // Only 3 values
        assert!(det.iqr_method(&values).is_none());
    }

    #[test]
    fn test_iqr_cleaned_data_excludes_outliers() {
        let det = OutlierDetector::new();
        let mut values: Vec<f64> = vec![10.0; 20];
        values[7] = 9999.0;
        let result = det.iqr_method(&values).expect("result should be valid");
        assert!(
            !result.cleaned.contains(&9999.0),
            "Outlier should be removed from cleaned data"
        );
    }

    #[test]
    fn test_iqr_quartiles_ordered() {
        let det = OutlierDetector::new();
        let values: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let result = det.iqr_method(&values).expect("result should be valid");
        assert!(result.q1 <= result.median, "Q1 ≤ median");
        assert!(result.median <= result.q3, "median ≤ Q3");
        assert!(result.iqr >= 0.0, "IQR ≥ 0");
    }

    #[test]
    fn test_iqr_fps_outliers_on_records() {
        let det = OutlierDetector::new();
        let mut history: Vec<BenchmarkRecord> = (0..20)
            .map(|i| make_record("fps_out", 1_700_000_000 + i as u64 * 3600, 30.0, 33.0, 38.5))
            .collect();
        history[3].throughput_fps = 999.9; // Outlier record
        let (iqr_result, outlier_records) = det
            .detect_fps_outliers("fps_out", &history)
            .expect("detect_fps_outliers should succeed");
        assert!(iqr_result.has_outliers());
        assert!(!outlier_records.is_empty());
        assert!((outlier_records[0].throughput_fps - 999.9).abs() < 1.0);
    }

    #[test]
    fn test_percentile_basic() {
        let sorted = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((outliers::percentile(&sorted, 0.0) - 1.0).abs() < 1e-9);
        assert!((outliers::percentile(&sorted, 1.0) - 5.0).abs() < 1e-9);
        assert!((outliers::percentile(&sorted, 0.5) - 3.0).abs() < 1e-9);
    }
}
