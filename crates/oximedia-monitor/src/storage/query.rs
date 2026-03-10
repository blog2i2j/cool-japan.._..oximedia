//! Query engine for time series data.

use crate::error::MonitorResult;
use crate::storage::{AggregateFunction, Aggregator, SqliteStorage};
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};

/// Time range for queries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeRange {
    /// Start time.
    pub start: DateTime<Utc>,
    /// End time.
    pub end: DateTime<Utc>,
}

impl TimeRange {
    /// Create a new time range.
    #[must_use]
    pub fn new(start: DateTime<Utc>, end: DateTime<Utc>) -> Self {
        Self { start, end }
    }

    /// Create a time range for the last N seconds.
    #[must_use]
    pub fn last_seconds(seconds: i64) -> Self {
        let end = Utc::now();
        let start = end - Duration::seconds(seconds);
        Self { start, end }
    }

    /// Create a time range for the last N minutes.
    #[must_use]
    pub fn last_minutes(minutes: i64) -> Self {
        Self::last_seconds(minutes * 60)
    }

    /// Create a time range for the last N hours.
    #[must_use]
    pub fn last_hours(hours: i64) -> Self {
        Self::last_minutes(hours * 60)
    }

    /// Create a time range for the last N days.
    #[must_use]
    pub fn last_days(days: i64) -> Self {
        Self::last_hours(days * 24)
    }

    /// Get the duration of this time range.
    #[must_use]
    pub fn duration(&self) -> Duration {
        self.end.signed_duration_since(self.start)
    }
}

/// Time series query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesQuery {
    /// Metric name.
    pub metric_name: String,
    /// Time range.
    pub time_range: TimeRange,
    /// Aggregation function (optional).
    pub aggregation: Option<AggregateFunction>,
    /// Labels filter (optional).
    pub labels: Option<String>,
}

/// Time series query result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesResult {
    /// Metric name.
    pub metric_name: String,
    /// Data points.
    pub points: Vec<DataPoint>,
    /// Query metadata.
    pub metadata: QueryMetadata,
}

/// Data point in query result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPoint {
    /// Timestamp.
    pub timestamp: DateTime<Utc>,
    /// Value.
    pub value: f64,
}

/// Query metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryMetadata {
    /// Number of points returned.
    pub count: usize,
    /// Time range queried.
    pub time_range: TimeRange,
    /// Whether data was aggregated.
    pub aggregated: bool,
}

/// Query engine for time series data.
pub struct QueryEngine {
    storage: SqliteStorage,
}

impl QueryEngine {
    /// Create a new query engine.
    #[must_use]
    pub fn new(storage: SqliteStorage) -> Self {
        Self { storage }
    }

    /// Execute a time series query.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn query(&self, query: TimeSeriesQuery) -> MonitorResult<TimeSeriesResult> {
        let points = self.storage.query(
            &query.metric_name,
            query.time_range.start,
            query.time_range.end,
        )?;

        let mut data_points: Vec<DataPoint> = points
            .into_iter()
            .map(|p| DataPoint {
                timestamp: p.timestamp,
                value: p.value,
            })
            .collect();

        let aggregated = if let Some(agg_func) = query.aggregation {
            let values: Vec<f64> = data_points.iter().map(|p| p.value).collect();
            if let Some(aggregated_value) = Aggregator::aggregate(&values, agg_func) {
                data_points = vec![DataPoint {
                    timestamp: query.time_range.start,
                    value: aggregated_value,
                }];
                true
            } else {
                false
            }
        } else {
            false
        };

        Ok(TimeSeriesResult {
            metric_name: query.metric_name,
            metadata: QueryMetadata {
                count: data_points.len(),
                time_range: query.time_range,
                aggregated,
            },
            points: data_points,
        })
    }

    /// Query multiple metrics at once.
    ///
    /// # Errors
    ///
    /// Returns an error if any query fails.
    pub fn query_multi(
        &self,
        queries: Vec<TimeSeriesQuery>,
    ) -> MonitorResult<Vec<TimeSeriesResult>> {
        queries.into_iter().map(|q| self.query(q)).collect()
    }

    /// Query with automatic aggregation based on time range.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn query_smart(&self, query: TimeSeriesQuery) -> MonitorResult<TimeSeriesResult> {
        let duration = query.time_range.duration();

        // Choose appropriate aggregation table based on time range
        if duration > Duration::days(7) {
            // Use 1-day aggregates
            self.query_aggregated(&query, &AggregationLevel::Day)
        } else if duration > Duration::hours(24) {
            // Use 1-hour aggregates
            self.query_aggregated(&query, &AggregationLevel::Hour)
        } else if duration > Duration::hours(2) {
            // Use 1-minute aggregates
            self.query_aggregated(&query, &AggregationLevel::Minute)
        } else {
            // Use raw data
            self.query(query)
        }
    }

    fn query_aggregated(
        &self,
        query: &TimeSeriesQuery,
        level: &AggregationLevel,
    ) -> MonitorResult<TimeSeriesResult> {
        let aggregates = match level {
            AggregationLevel::Minute => self.storage.query_1min_aggregates(
                &query.metric_name,
                query.time_range.start,
                query.time_range.end,
            )?,
            AggregationLevel::Hour => self.storage.query_1hour_aggregates(
                &query.metric_name,
                query.time_range.start,
                query.time_range.end,
            )?,
            AggregationLevel::Day => self.storage.query_1day_aggregates(
                &query.metric_name,
                query.time_range.start,
                query.time_range.end,
            )?,
        };

        let data_points: Vec<DataPoint> = aggregates
            .into_iter()
            .map(|agg| DataPoint {
                timestamp: agg.timestamp,
                value: agg.avg_value, // Use average for aggregated data
            })
            .collect();

        Ok(TimeSeriesResult {
            metric_name: query.metric_name.clone(),
            metadata: QueryMetadata {
                count: data_points.len(),
                time_range: query.time_range.clone(),
                aggregated: true,
            },
            points: data_points,
        })
    }
}

enum AggregationLevel {
    Minute,
    Hour,
    Day,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::TimeSeriesPoint;
    use chrono::Duration;
    use tempfile::tempdir;

    #[test]
    fn test_time_range_last_hours() {
        let range = TimeRange::last_hours(1);
        let duration = range.duration();
        assert_eq!(duration.num_hours(), 1);
    }

    #[test]
    fn test_time_range_last_days() {
        let range = TimeRange::last_days(7);
        let duration = range.duration();
        assert_eq!(duration.num_days(), 7);
    }

    #[test]
    fn test_query_engine() {
        let dir = tempdir().expect("failed to create temp dir");
        let db_path = dir.path().join("test.db");
        let storage = SqliteStorage::new(&db_path).expect("failed to create");
        let engine = QueryEngine::new(storage);

        // Insert some test data
        let now = Utc::now();
        let points: Vec<TimeSeriesPoint> = (0..10)
            .map(|i| TimeSeriesPoint {
                metric_name: "cpu.usage".to_string(),
                timestamp: now + Duration::seconds(i),
                value: i as f64,
                labels: None,
            })
            .collect();

        engine
            .storage
            .insert_batch(&points)
            .expect("insert_batch should succeed");

        // Query the data
        let query = TimeSeriesQuery {
            metric_name: "cpu.usage".to_string(),
            time_range: TimeRange::new(now - Duration::seconds(10), now + Duration::seconds(20)),
            aggregation: None,
            labels: None,
        };

        let result = engine.query(query).expect("failed to query");
        assert_eq!(result.points.len(), 10);
    }

    #[test]
    fn test_query_with_aggregation() {
        let dir = tempdir().expect("failed to create temp dir");
        let db_path = dir.path().join("test.db");
        let storage = SqliteStorage::new(&db_path).expect("failed to create");
        let engine = QueryEngine::new(storage);

        let now = Utc::now();
        let points: Vec<TimeSeriesPoint> = (0..10)
            .map(|i| TimeSeriesPoint {
                metric_name: "cpu.usage".to_string(),
                timestamp: now + Duration::seconds(i),
                value: i as f64,
                labels: None,
            })
            .collect();

        engine
            .storage
            .insert_batch(&points)
            .expect("insert_batch should succeed");

        let query = TimeSeriesQuery {
            metric_name: "cpu.usage".to_string(),
            time_range: TimeRange::new(now - Duration::seconds(10), now + Duration::seconds(20)),
            aggregation: Some(AggregateFunction::Avg),
            labels: None,
        };

        let result = engine.query(query).expect("failed to query");
        assert_eq!(result.points.len(), 1);
        assert_eq!(result.points[0].value, 4.5); // Average of 0-9
        assert!(result.metadata.aggregated);
    }
}
