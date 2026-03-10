//! Aggregation functions for time series data.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Aggregate function types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AggregateFunction {
    /// Minimum value.
    Min,
    /// Maximum value.
    Max,
    /// Average value.
    Avg,
    /// Sum of values.
    Sum,
    /// Count of values.
    Count,
    /// Percentile (value between 0.0 and 1.0).
    Percentile(u8), // 0-100
    /// Rate of change.
    Rate,
}

/// Aggregated metric result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedMetric {
    /// Timestamp (start of aggregation window).
    pub timestamp: DateTime<Utc>,
    /// Minimum value in window.
    pub min: f64,
    /// Maximum value in window.
    pub max: f64,
    /// Average value in window.
    pub avg: f64,
    /// Sum of values in window.
    pub sum: f64,
    /// Count of values in window.
    pub count: usize,
}

/// Aggregator for time series data.
pub struct Aggregator;

impl Aggregator {
    /// Aggregate values using the specified function.
    #[must_use]
    pub fn aggregate(values: &[f64], function: AggregateFunction) -> Option<f64> {
        if values.is_empty() {
            return None;
        }

        match function {
            AggregateFunction::Min => values
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .copied(),
            AggregateFunction::Max => values
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .copied(),
            AggregateFunction::Avg => Some(values.iter().sum::<f64>() / values.len() as f64),
            AggregateFunction::Sum => Some(values.iter().sum()),
            AggregateFunction::Count => Some(values.len() as f64),
            AggregateFunction::Percentile(p) => {
                let mut sorted = values.to_vec();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let idx = ((sorted.len() as f64 - 1.0) * (f64::from(p) / 100.0)) as usize;
                Some(sorted[idx])
            }
            AggregateFunction::Rate => {
                if values.len() < 2 {
                    None
                } else {
                    let first = values.first()?;
                    let last = values.last()?;
                    Some(last - first)
                }
            }
        }
    }

    /// Create a full aggregation of values.
    #[must_use]
    pub fn aggregate_full(timestamp: DateTime<Utc>, values: &[f64]) -> Option<AggregatedMetric> {
        if values.is_empty() {
            return None;
        }

        Some(AggregatedMetric {
            timestamp,
            min: Self::aggregate(values, AggregateFunction::Min)?,
            max: Self::aggregate(values, AggregateFunction::Max)?,
            avg: Self::aggregate(values, AggregateFunction::Avg)?,
            sum: Self::aggregate(values, AggregateFunction::Sum)?,
            count: values.len(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_min_aggregation() {
        let values = vec![1.0, 5.0, 3.0, 9.0, 2.0];
        let result = Aggregator::aggregate(&values, AggregateFunction::Min);
        assert_eq!(result, Some(1.0));
    }

    #[test]
    fn test_max_aggregation() {
        let values = vec![1.0, 5.0, 3.0, 9.0, 2.0];
        let result = Aggregator::aggregate(&values, AggregateFunction::Max);
        assert_eq!(result, Some(9.0));
    }

    #[test]
    fn test_avg_aggregation() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = Aggregator::aggregate(&values, AggregateFunction::Avg);
        assert_eq!(result, Some(3.0));
    }

    #[test]
    fn test_sum_aggregation() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = Aggregator::aggregate(&values, AggregateFunction::Sum);
        assert_eq!(result, Some(15.0));
    }

    #[test]
    fn test_count_aggregation() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = Aggregator::aggregate(&values, AggregateFunction::Count);
        assert_eq!(result, Some(5.0));
    }

    #[test]
    fn test_percentile_aggregation() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let p50 = Aggregator::aggregate(&values, AggregateFunction::Percentile(50));
        assert_eq!(p50, Some(5.0));

        let p90 = Aggregator::aggregate(&values, AggregateFunction::Percentile(90));
        assert_eq!(p90, Some(9.0));
    }

    #[test]
    fn test_rate_aggregation() {
        let values = vec![10.0, 15.0, 25.0, 40.0];
        let result = Aggregator::aggregate(&values, AggregateFunction::Rate);
        assert_eq!(result, Some(30.0)); // 40 - 10
    }

    #[test]
    fn test_empty_values() {
        let values: Vec<f64> = vec![];
        let result = Aggregator::aggregate(&values, AggregateFunction::Avg);
        assert_eq!(result, None);
    }

    #[test]
    fn test_aggregate_full() {
        let timestamp = Utc::now();
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let aggregated =
            Aggregator::aggregate_full(timestamp, &values).expect("operation should succeed");

        assert_eq!(aggregated.min, 1.0);
        assert_eq!(aggregated.max, 5.0);
        assert_eq!(aggregated.avg, 3.0);
        assert_eq!(aggregated.sum, 15.0);
        assert_eq!(aggregated.count, 5);
    }
}
