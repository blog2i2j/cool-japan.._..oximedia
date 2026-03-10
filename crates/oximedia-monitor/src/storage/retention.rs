//! Retention manager for time series data.

use crate::config::RetentionConfig;
use crate::error::{MonitorError, MonitorResult};
use crate::storage::SqliteStorage;
use chrono::Utc;
use std::sync::Arc;
use tokio::time::interval;

/// Retention manager for cleaning up old data.
pub struct RetentionManager {
    config: RetentionConfig,
    storage: Arc<SqliteStorage>,
    running: Arc<tokio::sync::RwLock<bool>>,
}

impl RetentionManager {
    /// Create a new retention manager.
    #[must_use]
    pub fn new(config: RetentionConfig, storage: Arc<SqliteStorage>) -> Self {
        Self {
            config,
            storage,
            running: Arc::new(tokio::sync::RwLock::new(false)),
        }
    }

    /// Start the retention manager background task.
    pub async fn start(&self) -> MonitorResult<()> {
        let mut running = self.running.write().await;
        if *running {
            return Err(MonitorError::Storage(
                "Retention manager already running".to_string(),
            ));
        }

        *running = true;
        drop(running);

        let storage = self.storage.clone();
        let config = self.config.clone();
        let running_flag = self.running.clone();

        tokio::spawn(async move {
            // Run every hour
            let mut ticker = interval(std::time::Duration::from_secs(3600));

            loop {
                ticker.tick().await;

                let is_running = *running_flag.read().await;
                if !is_running {
                    break;
                }

                if let Err(e) = Self::cleanup(&storage, &config).await {
                    tracing::error!("Failed to cleanup old data: {}", e);
                }
            }
        });

        Ok(())
    }

    /// Stop the retention manager.
    pub async fn stop(&self) {
        let mut running = self.running.write().await;
        *running = false;
    }

    /// Perform cleanup of old data.
    async fn cleanup(storage: &SqliteStorage, config: &RetentionConfig) -> MonitorResult<()> {
        let now = Utc::now();

        // Delete raw data older than retention period
        let raw_cutoff = now - config.raw_data;
        let deleted = storage.delete_before(raw_cutoff)?;
        tracing::info!("Deleted {deleted} old raw data points");

        // Delete hourly aggregates older than the configured retention period
        let hourly_cutoff = now
            - chrono::Duration::from_std(config.hour_aggregates)
                .unwrap_or_else(|_| chrono::Duration::days(30));
        let deleted_hourly = storage.delete_1hour_before(hourly_cutoff)?;
        tracing::debug!("Deleted {deleted_hourly} old hourly aggregate points");

        // Delete daily aggregates older than the configured retention period
        let daily_cutoff = now
            - chrono::Duration::from_std(config.day_aggregates)
                .unwrap_or_else(|_| chrono::Duration::days(365));
        let deleted_daily = storage.delete_1day_before(daily_cutoff)?;
        tracing::debug!("Deleted {deleted_daily} old daily aggregate points");

        // Vacuum database to reclaim space
        storage.vacuum()?;

        Ok(())
    }

    /// Run cleanup once immediately.
    ///
    /// # Errors
    ///
    /// Returns an error if cleanup fails.
    pub async fn cleanup_now(&self) -> MonitorResult<()> {
        Self::cleanup(&self.storage, &self.config).await
    }

    /// Check if retention manager is running.
    pub async fn is_running(&self) -> bool {
        *self.running.read().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::TimeSeriesPoint;
    use chrono::Duration;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_retention_manager() {
        let dir = tempdir().expect("failed to create temp dir");
        let db_path = dir.path().join("test.db");
        let storage = Arc::new(SqliteStorage::new(&db_path).expect("failed to create"));

        let config = RetentionConfig {
            raw_data: std::time::Duration::from_secs(10),
            minute_aggregates: std::time::Duration::from_secs(7 * 24 * 3600),
            hour_aggregates: std::time::Duration::from_secs(30 * 24 * 3600),
            day_aggregates: std::time::Duration::from_secs(365 * 24 * 3600),
        };

        let manager = RetentionManager::new(config, storage.clone());

        // Insert some old data
        let old_time = Utc::now() - Duration::seconds(20);
        let point = TimeSeriesPoint {
            metric_name: "test".to_string(),
            timestamp: old_time,
            value: 42.0,
            labels: None,
        };

        storage.insert(&point).expect("failed to insert");

        // Insert some recent data
        let recent_time = Utc::now();
        let point = TimeSeriesPoint {
            metric_name: "test".to_string(),
            timestamp: recent_time,
            value: 100.0,
            labels: None,
        };

        storage.insert(&point).expect("failed to insert");

        assert_eq!(storage.count().expect("count should succeed"), 2);

        // Run cleanup
        manager.cleanup_now().await.expect("await should be valid");

        // Old data should be deleted
        assert_eq!(storage.count().expect("count should succeed"), 1);
    }

    #[tokio::test]
    async fn test_retention_manager_start_stop() {
        let dir = tempdir().expect("failed to create temp dir");
        let db_path = dir.path().join("test.db");
        let storage = Arc::new(SqliteStorage::new(&db_path).expect("failed to create"));

        let config = RetentionConfig::default();
        let manager = RetentionManager::new(config, storage);

        assert!(!manager.is_running().await);

        manager.start().await.expect("await should be valid");
        assert!(manager.is_running().await);

        manager.stop().await;
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        assert!(!manager.is_running().await);
    }
}
