//! Failover management and orchestration.

use crate::failover::health::{HealthMonitor, HealthStatus};
use crate::failover::switch::FailoverSwitch;
use crate::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{error, info, warn};

/// Failover configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailoverConfig {
    /// Enable automatic failover
    pub auto_failover: bool,
    /// Health check interval in seconds
    pub health_check_interval: u64,
    /// Number of failed checks before failover
    pub failure_threshold: u32,
    /// Failover switch delay in milliseconds
    pub switch_delay_ms: u64,
}

impl Default for FailoverConfig {
    fn default() -> Self {
        Self {
            auto_failover: true,
            health_check_interval: 5,
            failure_threshold: 3,
            switch_delay_ms: 100,
        }
    }
}

/// Failover state for a channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FailoverState {
    /// Primary is active
    Primary,
    /// Secondary is active (failover)
    Secondary,
}

/// Failover manager.
pub struct FailoverManager {
    config: FailoverConfig,
    health_monitor: HealthMonitor,
    failover_switch: FailoverSwitch,
    channel_states: Arc<RwLock<HashMap<usize, FailoverState>>>,
    running: Arc<RwLock<bool>>,
}

impl FailoverManager {
    /// Create a new failover manager.
    pub async fn new(config: FailoverConfig) -> Result<Self> {
        info!("Creating failover manager");

        Ok(Self {
            config: config.clone(),
            health_monitor: HealthMonitor::new(config.health_check_interval),
            failover_switch: FailoverSwitch::new(config.switch_delay_ms),
            channel_states: Arc::new(RwLock::new(HashMap::new())),
            running: Arc::new(RwLock::new(false)),
        })
    }

    /// Start the failover manager.
    pub async fn start(&mut self) -> Result<()> {
        info!("Starting failover manager");

        {
            let mut running = self.running.write().await;
            *running = true;
        }

        self.health_monitor.start().await?;

        // Spawn monitoring task if auto-failover is enabled
        if self.config.auto_failover {
            let health_monitor = self.health_monitor.clone();
            let failover_switch = self.failover_switch.clone();
            let channel_states = Arc::clone(&self.channel_states);
            let running = Arc::clone(&self.running);
            let failure_threshold = self.config.failure_threshold;

            tokio::spawn(async move {
                while *running.read().await {
                    // Check health and trigger failover if needed
                    let channels = health_monitor.get_all_statuses().await;

                    for (channel_id, status) in channels {
                        if status.consecutive_failures >= failure_threshold {
                            warn!(
                                "Channel {} health check failed {} times, triggering failover",
                                channel_id, status.consecutive_failures
                            );

                            if let Err(e) = failover_switch.trigger(channel_id).await {
                                error!("Failover switch failed for channel {}: {}", channel_id, e);
                            } else {
                                let mut states = channel_states.write().await;
                                states.insert(channel_id, FailoverState::Secondary);
                            }
                        }
                    }

                    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
                }
            });
        }

        Ok(())
    }

    /// Stop the failover manager.
    pub async fn stop(&mut self) -> Result<()> {
        info!("Stopping failover manager");

        {
            let mut running = self.running.write().await;
            *running = false;
        }

        self.health_monitor.stop().await?;

        Ok(())
    }

    /// Manually trigger failover for a channel.
    pub async fn trigger_failover(&mut self, channel_id: usize) -> Result<()> {
        info!("Manually triggering failover for channel {}", channel_id);

        self.failover_switch.trigger(channel_id).await?;

        let mut states = self.channel_states.write().await;
        states.insert(channel_id, FailoverState::Secondary);

        Ok(())
    }

    /// Restore to primary for a channel.
    pub async fn restore_primary(&mut self, channel_id: usize) -> Result<()> {
        info!("Restoring channel {} to primary", channel_id);

        self.failover_switch.restore(channel_id).await?;

        let mut states = self.channel_states.write().await;
        states.insert(channel_id, FailoverState::Primary);

        Ok(())
    }

    /// Get failover state for a channel.
    pub async fn get_state(&self, channel_id: usize) -> FailoverState {
        let states = self.channel_states.read().await;
        states
            .get(&channel_id)
            .copied()
            .unwrap_or(FailoverState::Primary)
    }

    /// Get health status for a channel.
    pub async fn get_health(&self, channel_id: usize) -> Option<HealthStatus> {
        self.health_monitor.get_status(channel_id).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_failover_manager_creation() {
        let config = FailoverConfig::default();
        let manager = FailoverManager::new(config).await;
        assert!(manager.is_ok());
    }

    #[tokio::test]
    async fn test_failover_state() {
        let config = FailoverConfig::default();
        let mut manager = FailoverManager::new(config)
            .await
            .expect("new should succeed");

        assert_eq!(manager.get_state(0).await, FailoverState::Primary);

        manager
            .trigger_failover(0)
            .await
            .expect("operation should succeed");
        assert_eq!(manager.get_state(0).await, FailoverState::Secondary);

        manager
            .restore_primary(0)
            .await
            .expect("operation should succeed");
        assert_eq!(manager.get_state(0).await, FailoverState::Primary);
    }
}
