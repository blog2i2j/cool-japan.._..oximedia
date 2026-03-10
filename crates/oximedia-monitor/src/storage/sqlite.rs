//! `SQLite` storage for historical time series data.

use crate::error::MonitorResult;
use chrono::{DateTime, Utc};
use parking_lot::Mutex;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;

/// A time series point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesPoint {
    /// Metric name.
    pub metric_name: String,
    /// Timestamp.
    pub timestamp: DateTime<Utc>,
    /// Value.
    pub value: f64,
    /// Labels (JSON encoded).
    pub labels: Option<String>,
}

/// `SQLite` storage for time series data.
#[derive(Clone)]
pub struct SqliteStorage {
    conn: Arc<Mutex<Connection>>,
}

impl SqliteStorage {
    /// Create a new `SQLite` storage.
    ///
    /// # Errors
    ///
    /// Returns an error if database initialization fails.
    pub fn new(path: impl AsRef<Path>) -> MonitorResult<Self> {
        // Create parent directory if it doesn't exist
        if let Some(parent) = path.as_ref().parent() {
            std::fs::create_dir_all(parent)?;
        }

        let conn = Connection::open(path)?;

        // Create tables
        conn.execute(
            "CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                value REAL NOT NULL,
                labels TEXT,
                UNIQUE(metric_name, timestamp, labels)
            )",
            [],
        )?;

        // Create indices for faster queries
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_metrics_name_time
             ON metrics(metric_name, timestamp)",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_metrics_time
             ON metrics(timestamp)",
            [],
        )?;

        // Create aggregated tables
        conn.execute(
            "CREATE TABLE IF NOT EXISTS metrics_1min (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                min_value REAL NOT NULL,
                max_value REAL NOT NULL,
                avg_value REAL NOT NULL,
                sum_value REAL NOT NULL,
                count INTEGER NOT NULL,
                labels TEXT,
                UNIQUE(metric_name, timestamp, labels)
            )",
            [],
        )?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS metrics_1hour (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                min_value REAL NOT NULL,
                max_value REAL NOT NULL,
                avg_value REAL NOT NULL,
                sum_value REAL NOT NULL,
                count INTEGER NOT NULL,
                labels TEXT,
                UNIQUE(metric_name, timestamp, labels)
            )",
            [],
        )?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS metrics_1day (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                min_value REAL NOT NULL,
                max_value REAL NOT NULL,
                avg_value REAL NOT NULL,
                sum_value REAL NOT NULL,
                count INTEGER NOT NULL,
                labels TEXT,
                UNIQUE(metric_name, timestamp, labels)
            )",
            [],
        )?;

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    /// Insert a time series point.
    ///
    /// # Errors
    ///
    /// Returns an error if insertion fails.
    pub fn insert(&self, point: &TimeSeriesPoint) -> MonitorResult<()> {
        let conn = self.conn.lock();

        conn.execute(
            "INSERT OR REPLACE INTO metrics (metric_name, timestamp, value, labels)
             VALUES (?1, ?2, ?3, ?4)",
            params![
                point.metric_name,
                point.timestamp.timestamp(),
                point.value,
                point.labels,
            ],
        )?;

        Ok(())
    }

    /// Insert multiple time series points in a transaction.
    ///
    /// # Errors
    ///
    /// Returns an error if insertion fails.
    pub fn insert_batch(&self, points: &[TimeSeriesPoint]) -> MonitorResult<()> {
        let conn = self.conn.lock();

        let tx = conn.unchecked_transaction()?;

        for point in points {
            tx.execute(
                "INSERT OR REPLACE INTO metrics (metric_name, timestamp, value, labels)
                 VALUES (?1, ?2, ?3, ?4)",
                params![
                    point.metric_name,
                    point.timestamp.timestamp(),
                    point.value,
                    point.labels,
                ],
            )?;
        }

        tx.commit()?;

        Ok(())
    }

    /// Query time series points.
    ///
    /// # Errors
    ///
    /// Returns an error if query fails.
    pub fn query(
        &self,
        metric_name: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> MonitorResult<Vec<TimeSeriesPoint>> {
        let conn = self.conn.lock();

        let mut stmt = conn.prepare(
            "SELECT metric_name, timestamp, value, labels
             FROM metrics
             WHERE metric_name = ?1 AND timestamp >= ?2 AND timestamp <= ?3
             ORDER BY timestamp ASC",
        )?;

        let rows = stmt.query_map(
            params![metric_name, start.timestamp(), end.timestamp()],
            |row| {
                let ts_secs: i64 = row.get(1)?;
                let timestamp = DateTime::from_timestamp(ts_secs, 0).ok_or_else(|| {
                    rusqlite::Error::FromSqlConversionFailure(
                        1,
                        rusqlite::types::Type::Integer,
                        Box::new(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            format!("timestamp value {ts_secs} is out of valid DateTime range"),
                        )),
                    )
                })?;
                Ok(TimeSeriesPoint {
                    metric_name: row.get(0)?,
                    timestamp,
                    value: row.get(2)?,
                    labels: row.get(3)?,
                })
            },
        )?;

        let mut points = Vec::new();
        for row in rows {
            points.push(row?);
        }

        Ok(points)
    }

    /// Query aggregated data from 1-minute table.
    ///
    /// # Errors
    ///
    /// Returns an error if query fails.
    pub fn query_1min_aggregates(
        &self,
        metric_name: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> MonitorResult<Vec<AggregateRow>> {
        self.query_aggregate_table("metrics_1min", metric_name, start, end)
    }

    /// Query aggregated data from 1-hour table.
    ///
    /// # Errors
    ///
    /// Returns an error if query fails.
    pub fn query_1hour_aggregates(
        &self,
        metric_name: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> MonitorResult<Vec<AggregateRow>> {
        self.query_aggregate_table("metrics_1hour", metric_name, start, end)
    }

    /// Query aggregated data from 1-day table.
    ///
    /// # Errors
    ///
    /// Returns an error if query fails.
    pub fn query_1day_aggregates(
        &self,
        metric_name: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> MonitorResult<Vec<AggregateRow>> {
        self.query_aggregate_table("metrics_1day", metric_name, start, end)
    }

    fn query_aggregate_table(
        &self,
        table: &str,
        metric_name: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> MonitorResult<Vec<AggregateRow>> {
        let conn = self.conn.lock();

        let query = format!(
            "SELECT metric_name, timestamp, min_value, max_value, avg_value, sum_value, count, labels
             FROM {table}
             WHERE metric_name = ?1 AND timestamp >= ?2 AND timestamp <= ?3
             ORDER BY timestamp ASC"
        );

        let mut stmt = conn.prepare(&query)?;

        let rows = stmt.query_map(
            params![metric_name, start.timestamp(), end.timestamp()],
            |row| {
                let ts_secs: i64 = row.get(1)?;
                let timestamp = DateTime::from_timestamp(ts_secs, 0).ok_or_else(|| {
                    rusqlite::Error::FromSqlConversionFailure(
                        1,
                        rusqlite::types::Type::Integer,
                        Box::new(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            format!("timestamp value {ts_secs} is out of valid DateTime range"),
                        )),
                    )
                })?;
                Ok(AggregateRow {
                    metric_name: row.get(0)?,
                    timestamp,
                    min_value: row.get(2)?,
                    max_value: row.get(3)?,
                    avg_value: row.get(4)?,
                    sum_value: row.get(5)?,
                    count: row.get(6)?,
                    labels: row.get(7)?,
                })
            },
        )?;

        let mut aggregates = Vec::new();
        for row in rows {
            aggregates.push(row?);
        }

        Ok(aggregates)
    }

    /// Delete old data points before the given timestamp.
    ///
    /// # Errors
    ///
    /// Returns an error if deletion fails.
    pub fn delete_before(&self, timestamp: DateTime<Utc>) -> MonitorResult<usize> {
        let conn = self.conn.lock();

        let deleted = conn.execute(
            "DELETE FROM metrics WHERE timestamp < ?1",
            params![timestamp.timestamp()],
        )?;

        Ok(deleted)
    }

    /// Delete old rows from the 1-minute aggregate table.
    ///
    /// # Errors
    ///
    /// Returns an error if deletion fails.
    pub fn delete_1min_before(&self, timestamp: DateTime<Utc>) -> MonitorResult<usize> {
        self.delete_aggregate_before("metrics_1min", timestamp)
    }

    /// Delete old rows from the 1-hour aggregate table.
    ///
    /// # Errors
    ///
    /// Returns an error if deletion fails.
    pub fn delete_1hour_before(&self, timestamp: DateTime<Utc>) -> MonitorResult<usize> {
        self.delete_aggregate_before("metrics_1hour", timestamp)
    }

    /// Delete old rows from the 1-day aggregate table.
    ///
    /// # Errors
    ///
    /// Returns an error if deletion fails.
    pub fn delete_1day_before(&self, timestamp: DateTime<Utc>) -> MonitorResult<usize> {
        self.delete_aggregate_before("metrics_1day", timestamp)
    }

    fn delete_aggregate_before(
        &self,
        table: &str,
        timestamp: DateTime<Utc>,
    ) -> MonitorResult<usize> {
        let conn = self.conn.lock();
        let sql = format!("DELETE FROM {table} WHERE timestamp < ?1");
        let deleted = conn.execute(&sql, params![timestamp.timestamp()])?;
        Ok(deleted)
    }

    /// Get the database size in bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if query fails.
    pub fn size(&self) -> MonitorResult<u64> {
        let conn = self.conn.lock();

        let size: i64 = conn.query_row(
            "SELECT page_count * page_size FROM pragma_page_count(), pragma_page_size()",
            [],
            |row| row.get(0),
        )?;

        Ok(size as u64)
    }

    /// Vacuum the database to reclaim space.
    ///
    /// # Errors
    ///
    /// Returns an error if vacuum fails.
    pub fn vacuum(&self) -> MonitorResult<()> {
        let conn = self.conn.lock();
        conn.execute("VACUUM", [])?;
        Ok(())
    }

    /// Get the count of metrics.
    ///
    /// # Errors
    ///
    /// Returns an error if query fails.
    pub fn count(&self) -> MonitorResult<usize> {
        let conn = self.conn.lock();

        let count: i64 = conn.query_row("SELECT COUNT(*) FROM metrics", [], |row| row.get(0))?;

        Ok(count as usize)
    }
}

/// Aggregated row from database.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregateRow {
    /// Metric name.
    pub metric_name: String,
    /// Timestamp.
    pub timestamp: DateTime<Utc>,
    /// Minimum value.
    pub min_value: f64,
    /// Maximum value.
    pub max_value: f64,
    /// Average value.
    pub avg_value: f64,
    /// Sum of values.
    pub sum_value: f64,
    /// Count of values.
    pub count: i64,
    /// Labels (JSON encoded).
    pub labels: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_sqlite_storage_creation() {
        let dir = tempdir().expect("failed to create temp dir");
        let db_path = dir.path().join("test.db");

        let _storage = SqliteStorage::new(&db_path).expect("failed to create");
        assert!(db_path.exists());
    }

    #[test]
    fn test_insert_and_query() {
        let dir = tempdir().expect("failed to create temp dir");
        let db_path = dir.path().join("test.db");
        let storage = SqliteStorage::new(&db_path).expect("failed to create");

        let now = Utc::now();

        let point = TimeSeriesPoint {
            metric_name: "cpu.usage".to_string(),
            timestamp: now,
            value: 42.5,
            labels: None,
        };

        storage.insert(&point).expect("failed to insert");

        let points = storage
            .query(
                "cpu.usage",
                now - chrono::Duration::seconds(10),
                now + chrono::Duration::seconds(10),
            )
            .expect("operation should succeed");

        assert_eq!(points.len(), 1);
        assert_eq!(points[0].value, 42.5);
    }

    #[test]
    fn test_insert_batch() {
        let dir = tempdir().expect("failed to create temp dir");
        let db_path = dir.path().join("test.db");
        let storage = SqliteStorage::new(&db_path).expect("failed to create");

        let now = Utc::now();

        let points: Vec<TimeSeriesPoint> = (0..10)
            .map(|i| TimeSeriesPoint {
                metric_name: "cpu.usage".to_string(),
                timestamp: now + chrono::Duration::seconds(i),
                value: i as f64,
                labels: None,
            })
            .collect();

        storage
            .insert_batch(&points)
            .expect("insert_batch should succeed");

        let queried = storage
            .query(
                "cpu.usage",
                now - chrono::Duration::seconds(10),
                now + chrono::Duration::seconds(20),
            )
            .expect("operation should succeed");

        assert_eq!(queried.len(), 10);
    }

    #[test]
    fn test_delete_before() {
        let dir = tempdir().expect("failed to create temp dir");
        let db_path = dir.path().join("test.db");
        let storage = SqliteStorage::new(&db_path).expect("failed to create");

        let now = Utc::now();

        let points: Vec<TimeSeriesPoint> = (0..10)
            .map(|i| TimeSeriesPoint {
                metric_name: "cpu.usage".to_string(),
                timestamp: now + chrono::Duration::seconds(i),
                value: i as f64,
                labels: None,
            })
            .collect();

        storage
            .insert_batch(&points)
            .expect("insert_batch should succeed");

        // Delete points before now + 5 seconds
        let deleted = storage
            .delete_before(now + chrono::Duration::seconds(5))
            .expect("operation should succeed");

        assert_eq!(deleted, 5);

        let remaining = storage.count().expect("count should succeed");
        assert_eq!(remaining, 5);
    }

    #[test]
    fn test_database_size() {
        let dir = tempdir().expect("failed to create temp dir");
        let db_path = dir.path().join("test.db");
        let storage = SqliteStorage::new(&db_path).expect("failed to create");

        let size = storage.size().expect("size should succeed");
        assert!(size > 0);
    }

    #[test]
    fn test_vacuum() {
        let dir = tempdir().expect("failed to create temp dir");
        let db_path = dir.path().join("test.db");
        let storage = SqliteStorage::new(&db_path).expect("failed to create");

        storage.vacuum().expect("vacuum should succeed");
    }
}
