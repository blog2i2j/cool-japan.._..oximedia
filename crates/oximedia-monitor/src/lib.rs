//! Comprehensive system monitoring and alerting for `OxiMedia`.
//!
//! This crate provides professional-grade system and application monitoring with:
//!
//! - **System Metrics**: CPU, memory, disk, network, GPU, temperature
//! - **Application Metrics**: Encoding throughput, job statistics, worker status
//! - **Quality Metrics**: Bitrate, quality scores (PSNR, SSIM, VMAF)
//! - **Time Series Storage**: In-memory ring buffer + `SQLite` historical storage
//! - **Alerting**: Multiple channels (email, Slack, Discord, webhook, SMS, file)
//! - **REST API**: Query metrics, manage alerts, health checks
//! - **WebSocket**: Real-time metric streaming
//! - **Health Checks**: Component health monitoring
//! - **Log Aggregation**: Structured logging with search
//! - **Dashboards**: Data provider for external visualization tools
//! - **Prometheus**: Compatible exposition format
//!
//! # Example
//!
//! ```no_run
//! use oximedia_monitor::{MonitorConfig, OximediaMonitor};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = MonitorConfig::default();
//!     let monitor = OximediaMonitor::new(config).await?;
//!
//!     // Start monitoring
//!     monitor.start().await?;
//!
//!     // Get system metrics
//!     if let Some(system_metrics) = monitor.system_metrics().await? {
//!         println!("CPU Usage: {:.2}%", system_metrics.cpu.total_usage);
//!     }
//!
//!     Ok(())
//! }
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::struct_excessive_bools)]
#![allow(dead_code)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::unused_async,
    clippy::unused_self
)]

pub mod alert;
pub mod alert_rule;
pub mod alerting_pipeline;
pub mod anomaly;
pub mod api;
/// Capacity planning and resource utilisation forecasting.
pub mod capacity_planner;
pub mod config;
pub mod correlation;
/// Monotonic counter metrics for discrete event tracking.
pub mod counter_metrics;
pub mod dashboard;
pub mod dashboard_metric;
pub mod dashboard_widget;
pub mod error;
pub mod event_bus;
pub mod health;
pub mod health_check;
/// Incident tracking and lifecycle management.
pub mod incident_tracker;
pub mod integration;
pub mod log_aggregator;
pub mod logs;
/// Metric export in Prometheus, JSON, CSV, and StatsD formats.
pub mod metric_export;
/// Metric processing pipeline with transformations and aggregations.
pub mod metric_pipeline;
pub mod metric_store;
pub mod metrics;
pub mod panel_view;
pub mod reporting;
/// Resource usage forecasting and trend analysis.
pub mod resource_forecast;
pub mod retention;
pub mod simple;
pub mod sla;
pub mod slo_tracker;
pub mod storage;
/// System-level metrics: CPU, memory, disk, and network.
pub mod system_metrics;
/// Distributed trace-span tracking for latency measurement.
pub mod trace_span;
pub mod uptime_tracker;

pub use alert::{Alert, AlertManager, AlertRule, AlertSeverity};
pub use config::{AlertConfig, ApiConfig, MetricsConfig, MonitorConfig, StorageConfig};
pub use error::{MonitorError, MonitorResult};
pub use metrics::{
    ApplicationMetrics, EncodingMetrics, JobMetrics, MetricsCollector, QualityMetrics,
    SystemMetrics, WorkerMetrics, WorkerStatus,
};
pub use simple::{
    CodecMetrics, Comparison, FiredAlert, HealthCheck, HealthCheckAggregator, HealthStatus,
    NotificationAction, SimpleAlertManager, SimpleAlertRule, SimpleMetricsCollector,
    SimpleMetricsSnapshot,
};
pub use storage::{QueryEngine, RingBuffer, SqliteStorage, TimeRange, TimeSeriesQuery};

use std::sync::Arc;

/// Main monitoring system.
pub struct OximediaMonitor {
    config: MonitorConfig,
    metrics_collector: Arc<MetricsCollector>,
    storage: Arc<SqliteStorage>,
    query_engine: Arc<QueryEngine>,
    alert_manager: Option<Arc<AlertManager>>,
}

impl OximediaMonitor {
    /// Create a new monitoring system.
    ///
    /// # Errors
    ///
    /// Returns an error if initialization fails.
    pub async fn new(config: MonitorConfig) -> MonitorResult<Self> {
        config.validate()?;

        let storage = Arc::new(SqliteStorage::new(&config.storage.db_path)?);
        let query_engine = Arc::new(QueryEngine::new((*storage).clone()));

        let metrics_collector = Arc::new(MetricsCollector::new(config.metrics.clone())?);

        let alert_manager = if config.alerts.enabled {
            Some(Arc::new(AlertManager::new(config.alerts.clone())))
        } else {
            None
        };

        Ok(Self {
            config,
            metrics_collector,
            storage,
            query_engine,
            alert_manager,
        })
    }

    /// Start the monitoring system.
    ///
    /// # Errors
    ///
    /// Returns an error if start fails.
    pub async fn start(&self) -> MonitorResult<()> {
        self.metrics_collector.start().await?;

        if let Some(ref alert_manager) = self.alert_manager {
            alert_manager.start().await?;
        }

        Ok(())
    }

    /// Stop the monitoring system.
    pub async fn stop(&self) {
        self.metrics_collector.stop().await;

        if let Some(ref alert_manager) = self.alert_manager {
            alert_manager.stop().await;
        }
    }

    /// Get current system metrics.
    ///
    /// # Errors
    ///
    /// Returns an error if collection fails.
    pub async fn system_metrics(&self) -> MonitorResult<Option<SystemMetrics>> {
        self.metrics_collector.collect_system_metrics().await
    }

    /// Get application metrics.
    #[must_use]
    pub fn application_metrics(&self) -> ApplicationMetrics {
        self.metrics_collector.application_metrics()
    }

    /// Get quality metrics.
    #[must_use]
    pub fn quality_metrics(&self) -> QualityMetrics {
        self.metrics_collector.quality_metrics()
    }

    /// Get the query engine.
    #[must_use]
    pub fn query_engine(&self) -> Arc<QueryEngine> {
        self.query_engine.clone()
    }

    /// Get the metrics collector.
    #[must_use]
    pub fn metrics_collector(&self) -> Arc<MetricsCollector> {
        self.metrics_collector.clone()
    }

    /// Get the storage.
    #[must_use]
    pub fn storage(&self) -> Arc<SqliteStorage> {
        self.storage.clone()
    }

    /// Get the alert manager.
    #[must_use]
    pub fn alert_manager(&self) -> Option<Arc<AlertManager>> {
        self.alert_manager.clone()
    }

    /// Get the configuration.
    #[must_use]
    pub const fn config(&self) -> &MonitorConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tempfile::tempdir;

    /// Build a [`MonitorConfig`] suitable for fast unit tests.
    ///
    /// System-metrics collection is disabled so no expensive sysinfo I/O
    /// occurs during construction or during the start/stop lifecycle.  Each
    /// test provides its own temp-dir backed database to avoid conflicts.
    fn fast_monitor_config(dir: &tempfile::TempDir) -> MonitorConfig {
        let mut config = MonitorConfig::default();
        config.storage.db_path = dir.path().join("monitor.db");
        config.metrics.enable_system_metrics = false;
        config.metrics.collection_interval = Duration::from_millis(100);
        config
    }

    #[tokio::test]
    async fn test_monitor_creation() {
        let dir = tempdir().expect("failed to create temp dir");
        let monitor = OximediaMonitor::new(fast_monitor_config(&dir))
            .await
            .expect("operation should succeed");
        assert!(monitor.alert_manager().is_some());
    }

    #[tokio::test]
    async fn test_monitor_start_stop() {
        let dir = tempdir().expect("failed to create temp dir");
        let monitor = OximediaMonitor::new(fast_monitor_config(&dir))
            .await
            .expect("operation should succeed");

        monitor.start().await.expect("await should be valid");
        assert!(monitor.metrics_collector().is_running().await);

        monitor.stop().await;
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        assert!(!monitor.metrics_collector().is_running().await);
    }

    #[tokio::test]
    async fn test_collect_metrics() {
        let dir = tempdir().expect("failed to create temp dir");
        // Enable system metrics (CPU + memory) but disable disk I/O so the
        // test completes quickly on macOS with many mount points.
        let mut config = MonitorConfig::default();
        config.storage.db_path = dir.path().join("monitor.db");
        config.metrics.enable_disk_metrics = false;

        let monitor = OximediaMonitor::new(config)
            .await
            .expect("failed to create");

        let system_metrics = monitor
            .system_metrics()
            .await
            .expect("await should be valid");
        assert!(system_metrics.is_some());

        let app_metrics = monitor.application_metrics();
        assert_eq!(app_metrics.encoding.total_frames, 0);

        let quality_metrics = monitor.quality_metrics();
        assert_eq!(quality_metrics.bitrate.video_bitrate_bps, 0);
    }
}
