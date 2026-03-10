//! Media pipeline benchmarking for `OxiMedia`.
//!
//! Provides per-stage timing, bottleneck detection, overall throughput,
//! and pipeline efficiency analysis.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};

/// Statistics for a single pipeline stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStage {
    /// Stage name (e.g. "decode", "filter", "encode").
    pub name: String,
    /// Running average latency in milliseconds.
    pub avg_ms: f64,
    /// Minimum observed latency in milliseconds.
    pub min_ms: f64,
    /// Maximum observed latency in milliseconds.
    pub max_ms: f64,
    /// Number of samples recorded.
    pub samples: u64,
    /// Sum of all samples (for incremental mean computation).
    sum_ms: f64,
    /// Sum of squared samples (for incremental variance computation).
    sum_sq_ms: f64,
}

impl PipelineStage {
    /// Create a new stage with the given name.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            avg_ms: 0.0,
            min_ms: f64::MAX,
            max_ms: 0.0,
            samples: 0,
            sum_ms: 0.0,
            sum_sq_ms: 0.0,
        }
    }

    /// Record a new latency sample and update running statistics.
    pub fn update(&mut self, sample_ms: f64) {
        self.samples += 1;
        self.sum_ms += sample_ms;
        self.sum_sq_ms += sample_ms * sample_ms;
        self.avg_ms = self.sum_ms / self.samples as f64;
        if sample_ms < self.min_ms {
            self.min_ms = sample_ms;
        }
        if sample_ms > self.max_ms {
            self.max_ms = sample_ms;
        }
    }

    /// Population standard deviation of recorded samples.
    #[must_use]
    pub fn std_dev(&self) -> f64 {
        if self.samples < 2 {
            return 0.0;
        }
        let n = self.samples as f64;
        let variance = (self.sum_sq_ms / n) - (self.avg_ms * self.avg_ms);
        variance.max(0.0).sqrt()
    }

    /// Throughput in frames per second based on average latency.
    #[must_use]
    pub fn throughput_fps(&self) -> f64 {
        if self.avg_ms == 0.0 {
            return 0.0;
        }
        1000.0 / self.avg_ms
    }

    /// Effective minimum throughput (worst-case).
    #[must_use]
    pub fn min_throughput_fps(&self) -> f64 {
        if self.max_ms == 0.0 {
            return 0.0;
        }
        1000.0 / self.max_ms
    }
}

/// Complete pipeline benchmark with all stages and aggregate metrics.
#[derive(Debug, Clone, Default)]
pub struct PipelineBench {
    /// Ordered list of pipeline stages.
    pub stages: Vec<PipelineStage>,
    /// Total number of frames that have completed the full pipeline.
    pub total_frames: u64,
    /// Elapsed wall-clock time for all frames in milliseconds.
    pub wall_time_ms: f64,
}

impl PipelineBench {
    /// Create an empty pipeline benchmark.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a new named stage to the pipeline (appended at the end).
    pub fn add_stage(&mut self, name: &str) {
        self.stages.push(PipelineStage::new(name));
    }

    /// Record a latency sample for the stage with the given name.
    ///
    /// No-op if the stage does not exist.
    pub fn record_stage(&mut self, name: &str, ms: f64) {
        if let Some(stage) = self.stages.iter_mut().find(|s| s.name == name) {
            stage.update(ms);
        }
    }

    /// Return the stage with the highest average latency (the bottleneck).
    #[must_use]
    pub fn bottleneck(&self) -> Option<&PipelineStage> {
        self.stages.iter().filter(|s| s.samples > 0).max_by(|a, b| {
            a.avg_ms
                .partial_cmp(&b.avg_ms)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Overall pipeline throughput in frames per second.
    #[must_use]
    pub fn overall_fps(&self) -> f64 {
        if self.wall_time_ms == 0.0 {
            return 0.0;
        }
        self.total_frames as f64 / (self.wall_time_ms / 1000.0)
    }

    /// Pipeline efficiency: ratio of the bottleneck throughput to the sum of all stage throughputs.
    ///
    /// A value of 1.0 means perfectly balanced; lower values indicate wasted capacity.
    #[must_use]
    pub fn pipeline_efficiency(&self) -> f64 {
        let active_stages: Vec<&PipelineStage> =
            self.stages.iter().filter(|s| s.samples > 0).collect();
        if active_stages.is_empty() {
            return 0.0;
        }
        // Bottleneck throughput
        let bottleneck_fps = self.bottleneck().map_or(0.0, PipelineStage::throughput_fps);
        if bottleneck_fps == 0.0 {
            return 0.0;
        }
        // Ideal throughput (limited by slowest stage) vs average of all stages
        let avg_fps: f64 = active_stages
            .iter()
            .map(|s| s.throughput_fps())
            .sum::<f64>()
            / active_stages.len() as f64;
        if avg_fps == 0.0 {
            return 0.0;
        }
        (bottleneck_fps / avg_fps).min(1.0)
    }

    /// Get a stage by name.
    #[must_use]
    pub fn get_stage(&self, name: &str) -> Option<&PipelineStage> {
        self.stages.iter().find(|s| s.name == name)
    }
}

/// Format a human-readable benchmark report for the given pipeline.
#[must_use]
pub fn format_bench_report(bench: &PipelineBench) -> String {
    let mut out = String::from("=== Pipeline Benchmark Report ===\n");
    out.push_str(&format!("Total frames : {}\n", bench.total_frames));
    out.push_str(&format!("Wall time    : {:.1} ms\n", bench.wall_time_ms));
    out.push_str(&format!("Overall FPS  : {:.2}\n", bench.overall_fps()));
    out.push_str(&format!(
        "Efficiency   : {:.1}%\n",
        bench.pipeline_efficiency() * 100.0
    ));

    out.push_str("\n--- Stage breakdown ---\n");
    for stage in &bench.stages {
        if stage.samples == 0 {
            out.push_str(&format!("  {:20}  (no data)\n", stage.name));
        } else {
            out.push_str(&format!(
                "  {:20}  avg={:.2}ms  min={:.2}ms  max={:.2}ms  \
                 std={:.2}ms  fps={:.1}  n={}\n",
                stage.name,
                stage.avg_ms,
                if stage.min_ms == f64::MAX {
                    0.0
                } else {
                    stage.min_ms
                },
                stage.max_ms,
                stage.std_dev(),
                stage.throughput_fps(),
                stage.samples,
            ));
        }
    }

    if let Some(bn) = bench.bottleneck() {
        out.push_str(&format!(
            "\nBottleneck: {} ({:.2} ms avg)\n",
            bn.name, bn.avg_ms
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_bench_with_stages() -> PipelineBench {
        let mut bench = PipelineBench::new();
        bench.add_stage("decode");
        bench.add_stage("filter");
        bench.add_stage("encode");
        bench
    }

    #[test]
    fn test_stage_update_avg() {
        let mut stage = PipelineStage::new("decode");
        stage.update(10.0);
        stage.update(20.0);
        assert!((stage.avg_ms - 15.0).abs() < 1e-9);
    }

    #[test]
    fn test_stage_min_max() {
        let mut stage = PipelineStage::new("decode");
        stage.update(5.0);
        stage.update(15.0);
        stage.update(10.0);
        assert_eq!(stage.min_ms, 5.0);
        assert_eq!(stage.max_ms, 15.0);
    }

    #[test]
    fn test_stage_std_dev() {
        let mut stage = PipelineStage::new("encode");
        // Two equal samples -> std_dev = 0
        stage.update(10.0);
        stage.update(10.0);
        assert!(stage.std_dev() < 1e-9);
    }

    #[test]
    fn test_stage_std_dev_single_sample() {
        let mut stage = PipelineStage::new("encode");
        stage.update(10.0);
        assert_eq!(stage.std_dev(), 0.0);
    }

    #[test]
    fn test_stage_throughput_fps() {
        let mut stage = PipelineStage::new("encode");
        stage.update(40.0); // 40 ms per frame -> 25 fps
        assert!((stage.throughput_fps() - 25.0).abs() < 1e-6);
    }

    #[test]
    fn test_stage_throughput_zero_avg() {
        let stage = PipelineStage::new("encode");
        assert_eq!(stage.throughput_fps(), 0.0);
    }

    #[test]
    fn test_add_and_record_stage() {
        let mut bench = make_bench_with_stages();
        bench.record_stage("decode", 10.0);
        bench.record_stage("decode", 20.0);
        let stage = bench.get_stage("decode").expect("stage should be valid");
        assert!((stage.avg_ms - 15.0).abs() < 1e-9);
    }

    #[test]
    fn test_record_nonexistent_stage_noop() {
        let mut bench = make_bench_with_stages();
        // Should not panic
        bench.record_stage("nonexistent", 5.0);
        assert_eq!(bench.stages.len(), 3);
    }

    #[test]
    fn test_bottleneck_detection() {
        let mut bench = make_bench_with_stages();
        bench.record_stage("decode", 5.0);
        bench.record_stage("filter", 50.0); // bottleneck
        bench.record_stage("encode", 10.0);
        let bn = bench.bottleneck().expect("bn should be valid");
        assert_eq!(bn.name, "filter");
    }

    #[test]
    fn test_bottleneck_empty_returns_none() {
        let bench = PipelineBench::new();
        assert!(bench.bottleneck().is_none());
    }

    #[test]
    fn test_overall_fps() {
        let mut bench = make_bench_with_stages();
        bench.total_frames = 100;
        bench.wall_time_ms = 5000.0; // 100 frames in 5 s = 20 fps
        assert!((bench.overall_fps() - 20.0).abs() < 1e-6);
    }

    #[test]
    fn test_overall_fps_zero_wall_time() {
        let bench = PipelineBench::new();
        assert_eq!(bench.overall_fps(), 0.0);
    }

    #[test]
    fn test_pipeline_efficiency_balanced() {
        let mut bench = PipelineBench::new();
        bench.add_stage("a");
        bench.add_stage("b");
        // Both stages have identical avg
        bench.record_stage("a", 10.0);
        bench.record_stage("b", 10.0);
        let eff = bench.pipeline_efficiency();
        // perfectly balanced -> efficiency = 1.0
        assert!((eff - 1.0).abs() < 1e-9, "eff={eff}");
    }

    #[test]
    fn test_format_bench_report_contains_stages() {
        let mut bench = make_bench_with_stages();
        bench.record_stage("decode", 10.0);
        bench.record_stage("encode", 30.0);
        bench.total_frames = 50;
        bench.wall_time_ms = 2000.0;
        let report = format_bench_report(&bench);
        assert!(report.contains("decode"));
        assert!(report.contains("encode"));
        assert!(report.contains("Bottleneck"));
    }
}
