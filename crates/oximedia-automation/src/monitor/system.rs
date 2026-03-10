//! System monitoring and health checks.

use crate::monitor::metrics::{MetricsCollector, SystemMetrics};
use crate::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{debug, info};

/// Monitoring configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitorConfig {
    /// Metrics collection interval in seconds
    pub collection_interval: u64,
    /// Enable CPU monitoring
    pub monitor_cpu: bool,
    /// Enable memory monitoring
    pub monitor_memory: bool,
    /// Enable disk monitoring
    pub monitor_disk: bool,
    /// Enable network monitoring
    pub monitor_network: bool,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            collection_interval: 5,
            monitor_cpu: true,
            monitor_memory: true,
            monitor_disk: true,
            monitor_network: true,
        }
    }
}

/// System monitor.
pub struct SystemMonitor {
    config: MonitorConfig,
    metrics_collector: MetricsCollector,
    running: Arc<RwLock<bool>>,
    current_metrics: Arc<RwLock<SystemMetrics>>,
}

impl SystemMonitor {
    /// Create a new system monitor.
    pub async fn new(config: MonitorConfig) -> Result<Self> {
        info!("Creating system monitor");

        Ok(Self {
            config: config.clone(),
            metrics_collector: MetricsCollector::new(config),
            running: Arc::new(RwLock::new(false)),
            current_metrics: Arc::new(RwLock::new(SystemMetrics::default())),
        })
    }

    /// Start system monitoring.
    pub async fn start(&mut self) -> Result<()> {
        info!("Starting system monitor");

        {
            let mut running = self.running.write().await;
            *running = true;
        }

        let metrics_collector = self.metrics_collector.clone();
        let running = Arc::clone(&self.running);
        let current_metrics = Arc::clone(&self.current_metrics);
        let collection_interval = self.config.collection_interval;

        tokio::spawn(async move {
            while *running.read().await {
                debug!("Collecting system metrics");

                let metrics = metrics_collector.collect().await;

                let mut current = current_metrics.write().await;
                *current = metrics;

                tokio::time::sleep(Duration::from_secs(collection_interval)).await;
            }
        });

        Ok(())
    }

    /// Stop system monitoring.
    pub async fn stop(&mut self) -> Result<()> {
        info!("Stopping system monitor");

        let mut running = self.running.write().await;
        *running = false;

        Ok(())
    }

    /// Get current system metrics.
    pub async fn get_metrics(&self) -> SystemMetrics {
        self.current_metrics.read().await.clone()
    }

    /// Get metrics as a map.
    pub async fn metrics(&self) -> Result<HashMap<String, f64>> {
        let metrics = self.current_metrics.read().await;

        let mut map = HashMap::new();
        map.insert("cpu_usage".to_string(), metrics.cpu_usage);
        map.insert("memory_usage".to_string(), metrics.memory_usage);
        map.insert("disk_usage".to_string(), metrics.disk_usage);
        map.insert("network_rx_mbps".to_string(), metrics.network_rx_mbps);
        map.insert("network_tx_mbps".to_string(), metrics.network_tx_mbps);

        Ok(map)
    }

    /// Check if system is healthy.
    pub async fn is_healthy(&self) -> bool {
        let metrics = self.current_metrics.read().await;

        // System is healthy if:
        // - CPU usage < 90%
        // - Memory usage < 90%
        // - Disk usage < 90%

        metrics.cpu_usage < 90.0 && metrics.memory_usage < 90.0 && metrics.disk_usage < 90.0
    }

    /// Get system health status.
    pub async fn health_status(&self) -> String {
        if self.is_healthy().await {
            "Healthy".to_string()
        } else {
            "Degraded".to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_system_monitor_creation() {
        let config = MonitorConfig::default();
        let monitor = SystemMonitor::new(config).await;
        assert!(monitor.is_ok());
    }

    #[tokio::test]
    async fn test_monitor_lifecycle() {
        let config = MonitorConfig::default();
        let mut monitor = SystemMonitor::new(config)
            .await
            .expect("new should succeed");

        assert!(monitor.start().await.is_ok());
        assert!(monitor.stop().await.is_ok());
    }

    #[tokio::test]
    async fn test_get_metrics() {
        let config = MonitorConfig::default();
        let monitor = SystemMonitor::new(config)
            .await
            .expect("new should succeed");

        let metrics = monitor.get_metrics().await;
        assert!(metrics.cpu_usage >= 0.0);
    }
}
