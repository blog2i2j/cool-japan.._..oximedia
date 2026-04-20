//! Metric aggregation and reporting.
//!
//! Provides utilities for aggregating quality scores across multiple frames
//! and generating comparison reports.

use crate::{MetricType, PoolingMethod, QualityScore};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Aggregated quality metrics for a video sequence.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AggregatedMetrics {
    /// Metric type
    pub metric: MetricType,
    /// Number of frames assessed
    pub frame_count: usize,
    /// Mean score
    pub mean: f64,
    /// Harmonic mean score
    pub harmonic_mean: f64,
    /// Minimum score
    pub min: f64,
    /// Maximum score
    pub max: f64,
    /// Standard deviation
    pub stddev: f64,
    /// Percentile scores (5th, 10th, 25th, 50th, 75th, 90th, 95th)
    pub percentiles: HashMap<u8, f64>,
    /// Per-frame scores
    pub per_frame: Vec<f64>,
}

impl AggregatedMetrics {
    /// Creates aggregated metrics from a sequence of quality scores.
    #[must_use]
    pub fn from_scores(scores: &[QualityScore]) -> Self {
        if scores.is_empty() {
            return Self::empty(MetricType::Psnr);
        }

        let metric = scores[0].metric;
        let values: Vec<f64> = scores.iter().map(|s| s.score).collect();

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let harmonic_mean = PoolingMethod::HarmonicMean.apply(&values);
        let min = values
            .iter()
            .copied()
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);
        let max = values
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);

        let variance = values
            .iter()
            .map(|v| {
                let diff = v - mean;
                diff * diff
            })
            .sum::<f64>()
            / values.len() as f64;
        let stddev = variance.sqrt();

        let mut percentiles = HashMap::new();
        for p in [5, 10, 25, 50, 75, 90, 95] {
            percentiles.insert(p, PoolingMethod::Percentile(p).apply(&values));
        }

        Self {
            metric,
            frame_count: values.len(),
            mean,
            harmonic_mean,
            min,
            max,
            stddev,
            percentiles,
            per_frame: values,
        }
    }

    /// Creates an empty aggregated metrics structure.
    #[must_use]
    pub fn empty(metric: MetricType) -> Self {
        Self {
            metric,
            frame_count: 0,
            mean: 0.0,
            harmonic_mean: 0.0,
            min: 0.0,
            max: 0.0,
            stddev: 0.0,
            percentiles: HashMap::new(),
            per_frame: Vec::new(),
        }
    }

    /// Returns the score using the specified pooling method.
    #[must_use]
    pub fn pooled(&self, method: PoolingMethod) -> f64 {
        match method {
            PoolingMethod::Mean => self.mean,
            PoolingMethod::HarmonicMean => self.harmonic_mean,
            PoolingMethod::Min => self.min,
            PoolingMethod::Percentile(p) => *self.percentiles.get(&p).unwrap_or(&0.0),
        }
    }
}

/// Quality comparison report for two videos.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ComparisonReport {
    /// Reference video information
    pub reference_info: VideoInfo,
    /// Distorted video information
    pub distorted_info: VideoInfo,
    /// Metrics computed
    pub metrics: Vec<AggregatedMetrics>,
    /// Overall quality rating (0-100)
    pub overall_quality: f64,
}

impl ComparisonReport {
    /// Creates a new comparison report.
    #[must_use]
    pub fn new(reference_info: VideoInfo, distorted_info: VideoInfo) -> Self {
        Self {
            reference_info,
            distorted_info,
            metrics: Vec::new(),
            overall_quality: 0.0,
        }
    }

    /// Adds aggregated metrics to the report.
    pub fn add_metrics(&mut self, metrics: AggregatedMetrics) {
        self.metrics.push(metrics);
    }

    /// Computes overall quality score based on multiple metrics.
    ///
    /// Uses a weighted combination of PSNR, SSIM, and VMAF if available.
    pub fn compute_overall_quality(&mut self) {
        let mut total_weight = 0.0;
        let mut weighted_sum = 0.0;

        for metric in &self.metrics {
            let (weight, normalized_score) = match metric.metric {
                MetricType::Vmaf => (0.5, metric.mean), // VMAF is already 0-100
                MetricType::Ssim | MetricType::MsSsim => (0.3, metric.mean * 100.0), // 0-1 to 0-100
                MetricType::Psnr => {
                    // PSNR typically 20-50, normalize to 0-100
                    let normalized = ((metric.mean - 20.0) / 30.0 * 100.0).clamp(0.0, 100.0);
                    (0.2, normalized)
                }
                _ => continue,
            };

            weighted_sum += weight * normalized_score;
            total_weight += weight;
        }

        self.overall_quality = if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            0.0
        };
    }

    /// Exports report to JSON.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Exports report to CSV.
    #[must_use]
    pub fn to_csv(&self) -> String {
        let mut csv = String::new();
        csv.push_str("Metric,Mean,HarmonicMean,Min,Max,StdDev,P5,P10,P25,P50,P75,P90,P95\n");

        for metric in &self.metrics {
            csv.push_str(&format!(
                "{:?},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}\n",
                metric.metric,
                metric.mean,
                metric.harmonic_mean,
                metric.min,
                metric.max,
                metric.stddev,
                metric.percentiles.get(&5).unwrap_or(&0.0),
                metric.percentiles.get(&10).unwrap_or(&0.0),
                metric.percentiles.get(&25).unwrap_or(&0.0),
                metric.percentiles.get(&50).unwrap_or(&0.0),
                metric.percentiles.get(&75).unwrap_or(&0.0),
                metric.percentiles.get(&90).unwrap_or(&0.0),
                metric.percentiles.get(&95).unwrap_or(&0.0),
            ));
        }

        csv
    }
}

/// Video information for comparison reports.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VideoInfo {
    /// Video file path
    pub path: String,
    /// Width in pixels
    pub width: usize,
    /// Height in pixels
    pub height: usize,
    /// Number of frames
    pub frame_count: usize,
    /// Frame rate
    pub fps: f64,
    /// Pixel format
    pub format: String,
}

impl VideoInfo {
    /// Creates new video information.
    #[must_use]
    pub fn new(
        path: String,
        width: usize,
        height: usize,
        frame_count: usize,
        fps: f64,
        format: String,
    ) -> Self {
        Self {
            path,
            width,
            height,
            frame_count,
            fps,
            format,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aggregated_metrics() {
        let scores = vec![
            QualityScore::new(MetricType::Psnr, 30.0),
            QualityScore::new(MetricType::Psnr, 35.0),
            QualityScore::new(MetricType::Psnr, 32.0),
            QualityScore::new(MetricType::Psnr, 33.0),
        ];

        let agg = AggregatedMetrics::from_scores(&scores);
        assert_eq!(agg.frame_count, 4);
        assert!((agg.mean - 32.5).abs() < 0.01);
        assert_eq!(agg.min, 30.0);
        assert_eq!(agg.max, 35.0);
    }

    #[test]
    fn test_pooling_methods() {
        let scores = vec![10.0, 20.0, 30.0, 40.0, 50.0];

        assert!((PoolingMethod::Mean.apply(&scores) - 30.0).abs() < 0.01);
        assert_eq!(PoolingMethod::Min.apply(&scores), 10.0);

        let harmonic = PoolingMethod::HarmonicMean.apply(&scores);
        assert!(harmonic < 30.0); // Harmonic mean is less than arithmetic mean
    }

    #[test]
    fn test_comparison_report() {
        let ref_info = VideoInfo::new(
            "reference.mp4".to_string(),
            1920,
            1080,
            100,
            30.0,
            "yuv420p".to_string(),
        );
        let dist_info = VideoInfo::new(
            "distorted.mp4".to_string(),
            1920,
            1080,
            100,
            30.0,
            "yuv420p".to_string(),
        );

        let mut report = ComparisonReport::new(ref_info, dist_info);

        let scores = vec![
            QualityScore::new(MetricType::Ssim, 0.95),
            QualityScore::new(MetricType::Ssim, 0.92),
            QualityScore::new(MetricType::Ssim, 0.93),
        ];
        report.add_metrics(AggregatedMetrics::from_scores(&scores));

        report.compute_overall_quality();
        assert!(report.overall_quality > 0.0);
    }
}
