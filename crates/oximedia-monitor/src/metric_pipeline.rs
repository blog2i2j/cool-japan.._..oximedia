#![allow(dead_code)]
//! Metric processing pipeline with transformations and aggregations.
//!
//! Provides a composable pipeline for ingesting raw metric data points,
//! applying transform stages (smoothing, thresholding, rate computation),
//! and emitting processed results.

use std::collections::VecDeque;

/// A tagged metric data point.
#[derive(Debug, Clone, PartialEq)]
pub struct MetricPoint {
    /// Metric name / tag.
    pub name: String,
    /// Unix timestamp in seconds.
    pub timestamp: f64,
    /// Metric value.
    pub value: f64,
}

impl MetricPoint {
    /// Create a new metric data point.
    pub fn new(name: impl Into<String>, timestamp: f64, value: f64) -> Self {
        Self {
            name: name.into(),
            timestamp,
            value,
        }
    }
}

/// Aggregation mode for a window of data points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregationMode {
    /// Arithmetic mean.
    Mean,
    /// Minimum value.
    Min,
    /// Maximum value.
    Max,
    /// Sum of values.
    Sum,
    /// Most recent value.
    Last,
    /// Number of data points.
    Count,
}

/// A stage in the metric processing pipeline.
#[derive(Debug, Clone)]
pub enum PipelineStage {
    /// Multiply every value by a constant factor.
    Scale(f64),
    /// Add a constant offset to every value.
    Offset(f64),
    /// Clamp values to a range.
    Clamp {
        /// Minimum allowed value.
        min: f64,
        /// Maximum allowed value.
        max: f64,
    },
    /// Replace values below the threshold with zero.
    Threshold(f64),
    /// Compute the per-second rate of change between consecutive points.
    Rate,
    /// Compute a windowed aggregation.
    WindowAggregate {
        /// Number of points in the window.
        window_size: usize,
        /// Aggregation mode.
        mode: AggregationMode,
    },
    /// Apply absolute value.
    Abs,
    /// Apply a power function.
    Power(f64),
}

/// Processes a vector of metric points through a single stage.
#[allow(clippy::cast_precision_loss)]
fn apply_stage(points: Vec<MetricPoint>, stage: &PipelineStage) -> Vec<MetricPoint> {
    match stage {
        PipelineStage::Scale(factor) => points
            .into_iter()
            .map(|mut p| {
                p.value *= factor;
                p
            })
            .collect(),

        PipelineStage::Offset(offset) => points
            .into_iter()
            .map(|mut p| {
                p.value += offset;
                p
            })
            .collect(),

        PipelineStage::Clamp { min, max } => points
            .into_iter()
            .map(|mut p| {
                p.value = p.value.clamp(*min, *max);
                p
            })
            .collect(),

        PipelineStage::Threshold(thresh) => points
            .into_iter()
            .map(|mut p| {
                if p.value < *thresh {
                    p.value = 0.0;
                }
                p
            })
            .collect(),

        PipelineStage::Rate => {
            if points.len() < 2 {
                return Vec::new();
            }
            points
                .windows(2)
                .map(|w| {
                    let dt = w[1].timestamp - w[0].timestamp;
                    let rate = if dt.abs() > 1e-15 {
                        (w[1].value - w[0].value) / dt
                    } else {
                        0.0
                    };
                    MetricPoint::new(w[1].name.clone(), w[1].timestamp, rate)
                })
                .collect()
        }

        PipelineStage::WindowAggregate { window_size, mode } => {
            let ws = (*window_size).max(1);
            let mut result = Vec::with_capacity(points.len());
            let mut window: VecDeque<f64> = VecDeque::with_capacity(ws);

            for p in &points {
                window.push_back(p.value);
                if window.len() > ws {
                    window.pop_front();
                }

                let agg = aggregate_window(&window, *mode);
                result.push(MetricPoint::new(p.name.clone(), p.timestamp, agg));
            }
            result
        }

        PipelineStage::Abs => points
            .into_iter()
            .map(|mut p| {
                p.value = p.value.abs();
                p
            })
            .collect(),

        PipelineStage::Power(exp) => points
            .into_iter()
            .map(|mut p| {
                p.value = p.value.powf(*exp);
                p
            })
            .collect(),
    }
}

/// Compute aggregation over a window of values.
#[allow(clippy::cast_precision_loss)]
fn aggregate_window(window: &VecDeque<f64>, mode: AggregationMode) -> f64 {
    if window.is_empty() {
        return 0.0;
    }
    match mode {
        AggregationMode::Mean => {
            let sum: f64 = window.iter().sum();
            sum / window.len() as f64
        }
        AggregationMode::Min => window.iter().copied().fold(f64::INFINITY, f64::min),
        AggregationMode::Max => window.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        AggregationMode::Sum => window.iter().sum(),
        AggregationMode::Last => *window.back().unwrap_or(&0.0),
        AggregationMode::Count => window.len() as f64,
    }
}

/// A composable metric processing pipeline.
#[derive(Debug, Clone)]
pub struct MetricPipeline {
    /// Ordered list of processing stages.
    stages: Vec<PipelineStage>,
    /// Human-readable name for this pipeline.
    name: String,
}

impl MetricPipeline {
    /// Create a new empty pipeline.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            stages: Vec::new(),
            name: name.into(),
        }
    }

    /// Append a processing stage.
    #[must_use]
    pub fn add_stage(mut self, stage: PipelineStage) -> Self {
        self.stages.push(stage);
        self
    }

    /// Process a batch of metric points through all stages.
    #[must_use]
    pub fn process(&self, points: Vec<MetricPoint>) -> Vec<MetricPoint> {
        let mut current = points;
        for stage in &self.stages {
            current = apply_stage(current, stage);
        }
        current
    }

    /// Get the pipeline name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the number of stages.
    #[must_use]
    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }
}

/// Pipeline registry that holds named pipelines.
#[derive(Debug)]
pub struct PipelineRegistry {
    /// Named pipelines.
    pipelines: Vec<MetricPipeline>,
}

impl PipelineRegistry {
    /// Create a new empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            pipelines: Vec::new(),
        }
    }

    /// Register a pipeline.
    pub fn register(&mut self, pipeline: MetricPipeline) {
        self.pipelines.push(pipeline);
    }

    /// Find a pipeline by name.
    #[must_use]
    pub fn find(&self, name: &str) -> Option<&MetricPipeline> {
        self.pipelines.iter().find(|p| p.name() == name)
    }

    /// Get the number of registered pipelines.
    #[must_use]
    pub fn count(&self) -> usize {
        self.pipelines.len()
    }

    /// Process points through a named pipeline.
    #[must_use]
    pub fn process(&self, name: &str, points: Vec<MetricPoint>) -> Option<Vec<MetricPoint>> {
        self.find(name).map(|p| p.process(points))
    }
}

impl Default for PipelineRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_points(n: usize) -> Vec<MetricPoint> {
        (0..n)
            .map(|i| MetricPoint::new("test", i as f64, (i + 1) as f64 * 10.0))
            .collect()
    }

    #[test]
    fn test_metric_point_creation() {
        let p = MetricPoint::new("cpu", 100.0, 42.5);
        assert_eq!(p.name, "cpu");
        assert!((p.timestamp - 100.0).abs() < f64::EPSILON);
        assert!((p.value - 42.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_scale_stage() {
        let points = sample_points(3); // 10, 20, 30
        let pipeline = MetricPipeline::new("scale_test").add_stage(PipelineStage::Scale(2.0));
        let result = pipeline.process(points);
        assert_eq!(result.len(), 3);
        assert!((result[0].value - 20.0).abs() < f64::EPSILON);
        assert!((result[2].value - 60.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_offset_stage() {
        let points = sample_points(2); // 10, 20
        let pipeline = MetricPipeline::new("offset_test").add_stage(PipelineStage::Offset(5.0));
        let result = pipeline.process(points);
        assert!((result[0].value - 15.0).abs() < f64::EPSILON);
        assert!((result[1].value - 25.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_clamp_stage() {
        let points = vec![
            MetricPoint::new("t", 0.0, -5.0),
            MetricPoint::new("t", 1.0, 50.0),
            MetricPoint::new("t", 2.0, 150.0),
        ];
        let pipeline = MetricPipeline::new("clamp_test").add_stage(PipelineStage::Clamp {
            min: 0.0,
            max: 100.0,
        });
        let result = pipeline.process(points);
        assert!((result[0].value).abs() < f64::EPSILON);
        assert!((result[1].value - 50.0).abs() < f64::EPSILON);
        assert!((result[2].value - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_threshold_stage() {
        let points = vec![
            MetricPoint::new("t", 0.0, 3.0),
            MetricPoint::new("t", 1.0, 7.0),
            MetricPoint::new("t", 2.0, 15.0),
        ];
        let pipeline = MetricPipeline::new("thresh_test").add_stage(PipelineStage::Threshold(5.0));
        let result = pipeline.process(points);
        assert!((result[0].value).abs() < f64::EPSILON); // 3 < 5 → 0
        assert!((result[1].value - 7.0).abs() < f64::EPSILON);
        assert!((result[2].value - 15.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_rate_stage() {
        let points = vec![
            MetricPoint::new("t", 0.0, 100.0),
            MetricPoint::new("t", 1.0, 110.0),
            MetricPoint::new("t", 2.0, 130.0),
        ];
        let pipeline = MetricPipeline::new("rate_test").add_stage(PipelineStage::Rate);
        let result = pipeline.process(points);
        assert_eq!(result.len(), 2);
        assert!((result[0].value - 10.0).abs() < f64::EPSILON);
        assert!((result[1].value - 20.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_window_aggregate_mean() {
        let points = vec![
            MetricPoint::new("t", 0.0, 10.0),
            MetricPoint::new("t", 1.0, 20.0),
            MetricPoint::new("t", 2.0, 30.0),
            MetricPoint::new("t", 3.0, 40.0),
        ];
        let pipeline = MetricPipeline::new("win_mean").add_stage(PipelineStage::WindowAggregate {
            window_size: 2,
            mode: AggregationMode::Mean,
        });
        let result = pipeline.process(points);
        // window at index 3: [30,40] → mean 35
        assert!((result[3].value - 35.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_window_aggregate_max() {
        let points = vec![
            MetricPoint::new("t", 0.0, 5.0),
            MetricPoint::new("t", 1.0, 15.0),
            MetricPoint::new("t", 2.0, 10.0),
        ];
        let pipeline = MetricPipeline::new("win_max").add_stage(PipelineStage::WindowAggregate {
            window_size: 3,
            mode: AggregationMode::Max,
        });
        let result = pipeline.process(points);
        assert!((result[2].value - 15.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_abs_stage() {
        let points = vec![
            MetricPoint::new("t", 0.0, -10.0),
            MetricPoint::new("t", 1.0, 5.0),
        ];
        let pipeline = MetricPipeline::new("abs_test").add_stage(PipelineStage::Abs);
        let result = pipeline.process(points);
        assert!((result[0].value - 10.0).abs() < f64::EPSILON);
        assert!((result[1].value - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_power_stage() {
        let points = vec![MetricPoint::new("t", 0.0, 3.0)];
        let pipeline = MetricPipeline::new("pow_test").add_stage(PipelineStage::Power(2.0));
        let result = pipeline.process(points);
        assert!((result[0].value - 9.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_chained_stages() {
        let points = sample_points(3); // 10, 20, 30
        let pipeline = MetricPipeline::new("chain")
            .add_stage(PipelineStage::Scale(0.5)) // 5, 10, 15
            .add_stage(PipelineStage::Offset(1.0)); // 6, 11, 16
        let result = pipeline.process(points);
        assert!((result[0].value - 6.0).abs() < f64::EPSILON);
        assert!((result[1].value - 11.0).abs() < f64::EPSILON);
        assert!((result[2].value - 16.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_pipeline_metadata() {
        let pipeline = MetricPipeline::new("my_pipe")
            .add_stage(PipelineStage::Scale(1.0))
            .add_stage(PipelineStage::Abs);
        assert_eq!(pipeline.name(), "my_pipe");
        assert_eq!(pipeline.stage_count(), 2);
    }

    #[test]
    fn test_pipeline_registry() {
        let mut registry = PipelineRegistry::new();
        registry.register(
            MetricPipeline::new("normalize").add_stage(PipelineStage::Clamp {
                min: 0.0,
                max: 100.0,
            }),
        );
        registry.register(MetricPipeline::new("double").add_stage(PipelineStage::Scale(2.0)));

        assert_eq!(registry.count(), 2);
        assert!(registry.find("normalize").is_some());
        assert!(registry.find("missing").is_none());

        let points = vec![MetricPoint::new("t", 0.0, 7.0)];
        let result = registry
            .process("double", points)
            .expect("process should succeed");
        assert!((result[0].value - 14.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_empty_pipeline() {
        let pipeline = MetricPipeline::new("noop");
        let points = sample_points(5);
        let result = pipeline.process(points.clone());
        assert_eq!(result.len(), points.len());
        for (a, b) in result.iter().zip(points.iter()) {
            assert!((a.value - b.value).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn test_rate_with_single_point() {
        let points = vec![MetricPoint::new("t", 0.0, 100.0)];
        let pipeline = MetricPipeline::new("rate_single").add_stage(PipelineStage::Rate);
        let result = pipeline.process(points);
        assert!(result.is_empty());
    }
}
