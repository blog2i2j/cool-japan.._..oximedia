//! Identify bottlenecks in media processing pipelines.
//!
//! [`PipelineAnalyzer`] collects per-stage timing samples (in microseconds)
//! and produces a [`BottleneckReport`] that names the slowest stage, its P95
//! latency, the fraction of the frame budget it consumes, and actionable
//! optimisation suggestions.
//!
//! # Example
//!
//! ```
//! use oximedia_profiler::pipeline_bottleneck::PipelineAnalyzer;
//!
//! let mut analyzer = PipelineAnalyzer::new();
//! // 60 fps → ~16 667 µs frame budget
//! let frame_period_us = 16_667_u64;
//! for _ in 0..100 {
//!     analyzer.add_timing("decode", 5_000);
//!     analyzer.add_timing("denoise", 14_000);
//!     analyzer.add_timing("encode", 2_000);
//! }
//! let report = analyzer.analyze(frame_period_us);
//! assert_eq!(report.slowest_stage, "denoise");
//! ```

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// StageTimings
// ---------------------------------------------------------------------------

/// Duration samples (in microseconds) collected for a single pipeline stage.
#[derive(Debug, Clone)]
pub struct StageTimings {
    /// Name of the pipeline stage.
    pub stage_name: String,
    /// Per-invocation durations in microseconds.
    pub samples: Vec<u64>,
}

impl StageTimings {
    /// Create a new, empty `StageTimings` for `stage_name`.
    pub fn new(stage_name: impl Into<String>) -> Self {
        Self {
            stage_name: stage_name.into(),
            samples: Vec::new(),
        }
    }

    /// Arithmetic mean of all samples in microseconds.
    ///
    /// Returns `0.0` when there are no samples.
    #[allow(clippy::cast_precision_loss)]
    pub fn mean_us(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        self.samples.iter().sum::<u64>() as f64 / self.samples.len() as f64
    }

    /// 95th-percentile latency in microseconds.
    ///
    /// Returns `0` when there are no samples.
    pub fn p95_us(&self) -> u64 {
        percentile(&self.samples, 95)
    }

    /// 99th-percentile latency in microseconds.
    ///
    /// Returns `0` when there are no samples.
    pub fn p99_us(&self) -> u64 {
        percentile(&self.samples, 99)
    }

    /// Theoretical throughput in frames-per-second given a fixed
    /// `frame_period_us` frame budget.
    ///
    /// `throughput = frame_period_us / mean_us`
    ///
    /// Returns `0.0` if there are no samples or `mean_us` is zero.
    #[allow(clippy::cast_precision_loss)]
    pub fn throughput_fps(&self, frame_period_us: u64) -> f64 {
        let mean = self.mean_us();
        if mean <= 0.0 || frame_period_us == 0 {
            return 0.0;
        }
        frame_period_us as f64 / mean
    }
}

/// Compute the `pct`th percentile of `samples` without modifying the original.
fn percentile(samples: &[u64], pct: u8) -> u64 {
    if samples.is_empty() {
        return 0;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();

    // Nearest-rank method (1-indexed, rounded up)
    #[allow(clippy::cast_precision_loss)]
    let idx = {
        let rank = (pct as f64 / 100.0 * sorted.len() as f64).ceil() as usize;
        rank.saturating_sub(1).min(sorted.len() - 1)
    };
    sorted[idx]
}

// ---------------------------------------------------------------------------
// BottleneckReport
// ---------------------------------------------------------------------------

/// Result of a pipeline bottleneck analysis.
#[derive(Debug, Clone)]
pub struct BottleneckReport {
    /// Name of the stage with the highest P95 latency.
    pub slowest_stage: String,
    /// P95 latency of the slowest stage in microseconds.
    pub slowest_p95_us: u64,
    /// Fraction of the frame budget consumed by the slowest stage
    /// (`slowest_p95_us / frame_period_us`), clamped to `[0.0, 1.0]`.
    pub bottleneck_fraction: f32,
    /// Human-readable optimisation suggestions.
    pub suggestions: Vec<String>,
}

// ---------------------------------------------------------------------------
// PipelineAnalyzer
// ---------------------------------------------------------------------------

/// Accumulates per-stage timing samples and analyses pipeline bottlenecks.
#[derive(Debug, Default)]
pub struct PipelineAnalyzer {
    /// Stage insertion order for deterministic output.
    order: Vec<String>,
    /// Timing data per stage.
    stages: HashMap<String, StageTimings>,
}

impl PipelineAnalyzer {
    /// Create a new, empty analyzer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a single timing sample for the named stage.
    pub fn add_timing(&mut self, stage: &str, duration_us: u64) {
        if !self.stages.contains_key(stage) {
            self.order.push(stage.to_owned());
        }
        self.stages
            .entry(stage.to_owned())
            .or_insert_with(|| StageTimings::new(stage))
            .samples
            .push(duration_us);
    }

    /// Return a reference to all `StageTimings`, in insertion order.
    pub fn stage_timings(&self) -> Vec<&StageTimings> {
        self.order
            .iter()
            .filter_map(|name| self.stages.get(name))
            .collect()
    }

    /// Identify the pipeline bottleneck given a target `frame_period_us`.
    ///
    /// The bottleneck is the stage with the highest P95 latency.
    /// `bottleneck_fraction` is that P95 divided by `frame_period_us`,
    /// clamped to `1.0`.
    ///
    /// Returns a default `BottleneckReport` with empty `slowest_stage` when
    /// no stages have been registered.
    #[allow(clippy::cast_precision_loss)]
    pub fn analyze(&self, frame_period_us: u64) -> BottleneckReport {
        let slowest = self
            .order
            .iter()
            .filter_map(|name| self.stages.get(name))
            .max_by_key(|s| s.p95_us());

        let Some(stage) = slowest else {
            return BottleneckReport {
                slowest_stage: String::new(),
                slowest_p95_us: 0,
                bottleneck_fraction: 0.0,
                suggestions: Vec::new(),
            };
        };

        let p95 = stage.p95_us();
        let fraction = if frame_period_us == 0 {
            1.0_f32
        } else {
            (p95 as f64 / frame_period_us as f64).min(1.0) as f32
        };

        let pct_used = (fraction * 100.0).round() as u32;
        let suggestions = vec![format!(
            "Stage '{}' uses {}% of frame budget (p95={} µs). Consider parallelization.",
            stage.stage_name, pct_used, p95
        )];

        BottleneckReport {
            slowest_stage: stage.stage_name.clone(),
            slowest_p95_us: p95,
            bottleneck_fraction: fraction,
            suggestions,
        }
    }

    /// Returns per-stage utilisation fractions sorted by utilisation
    /// descending.
    ///
    /// Utilisation = `p95_us / frame_period_us`, clamped to `1.0`.
    #[allow(clippy::cast_precision_loss)]
    pub fn stage_utilization(&self, frame_period_us: u64) -> Vec<(String, f32)> {
        let mut result: Vec<(String, f32)> = self
            .order
            .iter()
            .filter_map(|name| self.stages.get(name))
            .map(|s| {
                let frac = if frame_period_us == 0 {
                    1.0_f32
                } else {
                    (s.p95_us() as f64 / frame_period_us as f64).min(1.0) as f32
                };
                (s.stage_name.clone(), frac)
            })
            .collect();

        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        result
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: populate `n` identical samples for a stage.
    fn fill(analyzer: &mut PipelineAnalyzer, stage: &str, duration_us: u64, n: usize) {
        for _ in 0..n {
            analyzer.add_timing(stage, duration_us);
        }
    }

    // ------------------------------------------------------------------
    // StageTimings statistics
    // ------------------------------------------------------------------

    #[test]
    fn test_mean_empty_is_zero() {
        let st = StageTimings::new("x");
        assert_eq!(st.mean_us(), 0.0);
    }

    #[test]
    fn test_mean_computed_correctly() {
        let mut st = StageTimings::new("x");
        st.samples = vec![100, 200, 300];
        assert!((st.mean_us() - 200.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_p95_single_sample() {
        let mut st = StageTimings::new("x");
        st.samples = vec![1000];
        assert_eq!(st.p95_us(), 1000);
    }

    #[test]
    fn test_p95_large_set() {
        // 100 samples: 99 × 100, 1 × 5000.  P95 of 100 samples at idx 94 = 100.
        let mut st = StageTimings::new("x");
        st.samples = vec![100u64; 99];
        st.samples.push(5000);
        // sorted: [100,100,...(99 times),5000]
        // idx = ceil(0.95 * 100) - 1 = 95 - 1 = 94 → 100
        assert_eq!(st.p95_us(), 100);
    }

    #[test]
    fn test_p99_large_set() {
        let mut st = StageTimings::new("x");
        st.samples = vec![100u64; 99];
        st.samples.push(5000);
        // idx = ceil(0.99 * 100) - 1 = 99 - 1 = 98 → 100
        assert_eq!(st.p99_us(), 100);
    }

    #[test]
    fn test_throughput_fps() {
        // mean = 16_667 µs → fps ≈ 1.0 for same period
        let mut st = StageTimings::new("x");
        st.samples = vec![16_667; 10];
        let fps = st.throughput_fps(16_667);
        assert!((fps - 1.0).abs() < 0.01, "fps was {fps}");
    }

    #[test]
    fn test_throughput_fps_zero_mean_is_zero() {
        let st = StageTimings::new("x");
        assert_eq!(st.throughput_fps(16_667), 0.0);
    }

    // ------------------------------------------------------------------
    // PipelineAnalyzer::analyze
    // ------------------------------------------------------------------

    #[test]
    fn test_single_stage_is_100_percent_utilization() {
        let mut a = PipelineAnalyzer::new();
        fill(&mut a, "decode", 16_667, 100);
        let report = a.analyze(16_667);
        assert_eq!(report.slowest_stage, "decode");
        assert!((report.bottleneck_fraction - 1.0).abs() < 0.01,
            "fraction was {}", report.bottleneck_fraction);
    }

    #[test]
    fn test_multiple_stages_ranked_by_p95() {
        let mut a = PipelineAnalyzer::new();
        fill(&mut a, "fast", 1_000, 100);
        fill(&mut a, "slow", 10_000, 100);
        fill(&mut a, "medium", 5_000, 100);
        let report = a.analyze(16_667);
        assert_eq!(report.slowest_stage, "slow");
    }

    #[test]
    fn test_bottleneck_fraction_clamped_to_one() {
        let mut a = PipelineAnalyzer::new();
        // p95 = 20_000 which is > 16_667
        fill(&mut a, "decode", 20_000, 100);
        let report = a.analyze(16_667);
        assert!(report.bottleneck_fraction <= 1.0,
            "fraction should be clamped; was {}", report.bottleneck_fraction);
    }

    #[test]
    fn test_suggestions_generated() {
        let mut a = PipelineAnalyzer::new();
        fill(&mut a, "denoise", 8_000, 100);
        let report = a.analyze(16_000);
        assert!(!report.suggestions.is_empty());
        assert!(report.suggestions[0].contains("denoise"),
            "suggestion was: {}", report.suggestions[0]);
    }

    #[test]
    fn test_stage_utilization_sorted_desc() {
        let mut a = PipelineAnalyzer::new();
        fill(&mut a, "fast", 1_000, 50);
        fill(&mut a, "slow", 12_000, 50);
        fill(&mut a, "medium", 5_000, 50);
        let util = a.stage_utilization(16_000);
        assert_eq!(util[0].0, "slow");
        assert_eq!(util[1].0, "medium");
        assert_eq!(util[2].0, "fast");
    }

    #[test]
    fn test_analyze_no_stages_returns_empty_report() {
        let a = PipelineAnalyzer::new();
        let report = a.analyze(16_667);
        assert!(report.slowest_stage.is_empty());
        assert_eq!(report.slowest_p95_us, 0);
    }
}
