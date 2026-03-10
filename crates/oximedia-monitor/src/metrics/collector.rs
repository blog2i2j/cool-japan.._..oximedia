//! Metrics collector that orchestrates system, application, and quality metrics collection.

use crate::config::MetricsConfig;
use crate::error::{MonitorError, MonitorResult};
use crate::metrics::{
    ApplicationMetrics, ApplicationMetricsTracker, QualityMetrics, QualityMetricsTracker,
    SystemMetrics, SystemMetricsCollector,
};
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::time::interval;

/// Metrics collector.
pub struct MetricsCollector {
    config: MetricsConfig,
    system_collector: Arc<RwLock<Option<SystemMetricsCollector>>>,
    application_tracker: Arc<ApplicationMetricsTracker>,
    quality_tracker: Arc<QualityMetricsTracker>,
    running: Arc<RwLock<bool>>,
}

impl MetricsCollector {
    /// Create a new metrics collector.
    ///
    /// # Errors
    ///
    /// Returns an error if initialization fails.
    pub fn new(config: MetricsConfig) -> MonitorResult<Self> {
        let system_collector = if config.enable_system_metrics {
            Some(SystemMetricsCollector::new_with_options(
                config.enable_disk_metrics,
            )?)
        } else {
            None
        };

        Ok(Self {
            config,
            system_collector: Arc::new(RwLock::new(system_collector)),
            application_tracker: Arc::new(ApplicationMetricsTracker::new()),
            quality_tracker: Arc::new(QualityMetricsTracker::new()),
            running: Arc::new(RwLock::new(false)),
        })
    }

    /// Start collecting metrics in the background.
    pub async fn start(&self) -> MonitorResult<()> {
        let mut running = self.running.write().await;
        if *running {
            return Err(MonitorError::MetricsCollection(
                "Metrics collector already running".to_string(),
            ));
        }

        *running = true;
        drop(running);

        // Spawn background task
        let system_collector = self.system_collector.clone();
        let running = self.running.clone();
        let collection_interval = self.config.collection_interval;
        let enable_system_metrics = self.config.enable_system_metrics;

        tokio::spawn(async move {
            let mut ticker = interval(collection_interval);

            loop {
                ticker.tick().await;

                let is_running = *running.read().await;
                if !is_running {
                    break;
                }

                if enable_system_metrics {
                    if let Some(ref mut collector) = *system_collector.write().await {
                        if let Err(e) = collector.collect() {
                            tracing::error!("Failed to collect system metrics: {}", e);
                        }
                    }
                }
            }
        });

        Ok(())
    }

    /// Stop collecting metrics.
    pub async fn stop(&self) {
        let mut running = self.running.write().await;
        *running = false;
    }

    /// Collect system metrics once.
    ///
    /// # Errors
    ///
    /// Returns an error if collection fails.
    pub async fn collect_system_metrics(&self) -> MonitorResult<Option<SystemMetrics>> {
        if !self.config.enable_system_metrics {
            return Ok(None);
        }

        let mut collector = self.system_collector.write().await;
        if let Some(ref mut c) = *collector {
            Ok(Some(c.collect()?))
        } else {
            Ok(None)
        }
    }

    /// Get application metrics snapshot.
    #[must_use]
    pub fn application_metrics(&self) -> ApplicationMetrics {
        self.application_tracker.snapshot()
    }

    /// Get quality metrics snapshot.
    #[must_use]
    pub fn quality_metrics(&self) -> QualityMetrics {
        self.quality_tracker.snapshot()
    }

    /// Get application metrics tracker.
    #[must_use]
    pub fn application_tracker(&self) -> Arc<ApplicationMetricsTracker> {
        self.application_tracker.clone()
    }

    /// Get quality metrics tracker.
    #[must_use]
    pub fn quality_tracker(&self) -> Arc<QualityMetricsTracker> {
        self.quality_tracker.clone()
    }

    /// Check if collector is running.
    pub async fn is_running(&self) -> bool {
        *self.running.read().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// Build a config suitable for fast unit tests: system-metrics collection is
    /// disabled so no expensive sysinfo I/O occurs during construction.
    fn fast_test_config() -> MetricsConfig {
        MetricsConfig {
            enable_system_metrics: false,
            collection_interval: Duration::from_millis(100),
            ..MetricsConfig::default()
        }
    }

    #[tokio::test]
    async fn test_collector_creation() {
        let collector = MetricsCollector::new(fast_test_config()).expect("failed to create");
        assert!(!collector.is_running().await);
    }

    #[tokio::test]
    async fn test_collector_start_stop() {
        let collector = MetricsCollector::new(fast_test_config()).expect("failed to create");

        collector.start().await.expect("await should be valid");
        assert!(collector.is_running().await);

        collector.stop().await;
        // Give the background task a moment to observe the flag change.
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        assert!(!collector.is_running().await);
    }

    #[tokio::test]
    async fn test_collect_system_metrics() {
        // Enable system metrics but disable disk I/O to keep the test fast.
        let config = MetricsConfig {
            enable_system_metrics: true,
            enable_disk_metrics: false,
            collection_interval: Duration::from_millis(100),
            ..MetricsConfig::default()
        };
        let collector = MetricsCollector::new(config).expect("failed to create");

        let metrics = collector
            .collect_system_metrics()
            .await
            .expect("await should be valid");
        assert!(metrics.is_some());

        let metrics = metrics.expect("metrics should be valid");
        assert!(metrics.cpu.cpu_count > 0);
        assert!(metrics.memory.total > 0);
    }

    #[tokio::test]
    async fn test_application_metrics() {
        let collector = MetricsCollector::new(fast_test_config()).expect("failed to create");

        let tracker = collector.application_tracker();
        tracker.record_frame_encoded(16.67);
        tracker.record_job_completed(120.0);

        let metrics = collector.application_metrics();
        assert_eq!(metrics.encoding.total_frames, 1);
        assert_eq!(metrics.jobs.completed, 1);
    }

    #[tokio::test]
    async fn test_quality_metrics() {
        let collector = MetricsCollector::new(fast_test_config()).expect("failed to create");

        let tracker = collector.quality_tracker();
        tracker.update_bitrate(5_000_000, 128_000);
        tracker.update_scores(Some(35.0), Some(0.98), Some(85.0));

        let metrics = collector.quality_metrics();
        assert_eq!(metrics.bitrate.video_bitrate_bps, 5_000_000);
        assert_eq!(metrics.scores.psnr, Some(35.0));
    }
}
