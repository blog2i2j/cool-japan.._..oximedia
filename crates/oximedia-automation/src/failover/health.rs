//! Health monitoring for failover.

use crate::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;
use tracing::{debug, info};

/// Health status for a monitored system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    /// Is the system healthy
    pub healthy: bool,
    /// Last check time
    pub last_check: SystemTime,
    /// Consecutive failures
    pub consecutive_failures: u32,
    /// Consecutive successes
    pub consecutive_successes: u32,
    /// Last error message
    pub last_error: Option<String>,
}

impl Default for HealthStatus {
    fn default() -> Self {
        Self {
            healthy: true,
            last_check: SystemTime::now(),
            consecutive_failures: 0,
            consecutive_successes: 0,
            last_error: None,
        }
    }
}

/// Health monitor.
#[derive(Clone)]
pub struct HealthMonitor {
    check_interval: u64,
    statuses: Arc<RwLock<HashMap<usize, HealthStatus>>>,
    running: Arc<RwLock<bool>>,
}

impl HealthMonitor {
    /// Create a new health monitor.
    pub fn new(check_interval: u64) -> Self {
        info!("Creating health monitor with {}s interval", check_interval);

        Self {
            check_interval,
            statuses: Arc::new(RwLock::new(HashMap::new())),
            running: Arc::new(RwLock::new(false)),
        }
    }

    /// Start health monitoring.
    pub async fn start(&self) -> Result<()> {
        info!("Starting health monitor");

        {
            let mut running = self.running.write().await;
            *running = true;
        }

        let statuses = Arc::clone(&self.statuses);
        let running = Arc::clone(&self.running);
        let check_interval = self.check_interval;

        tokio::spawn(async move {
            while *running.read().await {
                debug!("Performing health checks");

                // In a real implementation, this would perform actual health checks
                // For now, we just update the last check time

                let mut statuses = statuses.write().await;
                for status in statuses.values_mut() {
                    status.last_check = SystemTime::now();
                }

                tokio::time::sleep(Duration::from_secs(check_interval)).await;
            }
        });

        Ok(())
    }

    /// Stop health monitoring.
    pub async fn stop(&self) -> Result<()> {
        info!("Stopping health monitor");

        let mut running = self.running.write().await;
        *running = false;

        Ok(())
    }

    /// Register a channel for monitoring.
    pub async fn register_channel(&self, channel_id: usize) {
        debug!("Registering channel {} for health monitoring", channel_id);

        let mut statuses = self.statuses.write().await;
        statuses.insert(channel_id, HealthStatus::default());
    }

    /// Unregister a channel.
    pub async fn unregister_channel(&self, channel_id: usize) {
        debug!(
            "Unregistering channel {} from health monitoring",
            channel_id
        );

        let mut statuses = self.statuses.write().await;
        statuses.remove(&channel_id);
    }

    /// Report health check result.
    pub async fn report_health(&self, channel_id: usize, healthy: bool, error: Option<String>) {
        let mut statuses = self.statuses.write().await;

        let status = statuses
            .entry(channel_id)
            .or_insert_with(HealthStatus::default);

        status.healthy = healthy;
        status.last_check = SystemTime::now();

        if healthy {
            status.consecutive_successes += 1;
            status.consecutive_failures = 0;
            status.last_error = None;
        } else {
            status.consecutive_failures += 1;
            status.consecutive_successes = 0;
            status.last_error = error;
        }

        debug!(
            "Channel {} health: {} (failures: {}, successes: {})",
            channel_id, healthy, status.consecutive_failures, status.consecutive_successes
        );
    }

    /// Get health status for a channel.
    pub async fn get_status(&self, channel_id: usize) -> Option<HealthStatus> {
        let statuses = self.statuses.read().await;
        statuses.get(&channel_id).cloned()
    }

    /// Get all health statuses.
    pub async fn get_all_statuses(&self) -> HashMap<usize, HealthStatus> {
        self.statuses.read().await.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_health_monitor_creation() {
        let monitor = HealthMonitor::new(5);
        assert_eq!(monitor.check_interval, 5);
    }

    #[tokio::test]
    async fn test_register_channel() {
        let monitor = HealthMonitor::new(5);
        monitor.register_channel(0).await;

        let status = monitor.get_status(0).await;
        assert!(status.is_some());
    }

    #[tokio::test]
    async fn test_report_health() {
        let monitor = HealthMonitor::new(5);
        monitor.register_channel(0).await;

        monitor
            .report_health(0, false, Some("Test error".to_string()))
            .await;

        let status = monitor
            .get_status(0)
            .await
            .expect("get_status should succeed");
        assert!(!status.healthy);
        assert_eq!(status.consecutive_failures, 1);
        assert_eq!(status.last_error, Some("Test error".to_string()));

        monitor.report_health(0, true, None).await;

        let status = monitor
            .get_status(0)
            .await
            .expect("get_status should succeed");
        assert!(status.healthy);
        assert_eq!(status.consecutive_successes, 1);
        assert_eq!(status.consecutive_failures, 0);
    }
}
