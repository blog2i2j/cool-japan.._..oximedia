//! Master control orchestration for broadcast automation.

use crate::channel::automation::{ChannelAutomation, ChannelConfig};
use crate::eas::alert::{EasAlert, EasManager};
use crate::failover::manager::{FailoverConfig, FailoverManager};
use crate::logging::asrun::AsRunLogger;
use crate::master::state::{SystemState, SystemStatus};
use crate::monitor::system::{MonitorConfig, SystemMonitor};
use crate::{AutomationError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{error, info, warn};

/// Master control configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MasterControlConfig {
    /// Number of channels to manage
    pub num_channels: usize,
    /// Failover configuration
    pub failover: FailoverConfig,
    /// Monitoring configuration
    pub monitoring: MonitorConfig,
    /// Enable EAS integration
    pub eas_enabled: bool,
    /// Enable remote control
    pub remote_enabled: bool,
}

impl Default for MasterControlConfig {
    fn default() -> Self {
        Self {
            num_channels: 1,
            failover: FailoverConfig::default(),
            monitoring: MonitorConfig::default(),
            eas_enabled: true,
            remote_enabled: true,
        }
    }
}

/// Master control system for broadcast automation.
///
/// Coordinates all automation subsystems including channel automation,
/// failover, EAS, logging, and monitoring.
#[allow(dead_code)]
pub struct MasterControl {
    config: MasterControlConfig,
    state: Arc<RwLock<SystemState>>,
    channels: HashMap<usize, ChannelAutomation>,
    failover: FailoverManager,
    eas: Option<EasManager>,
    logger: AsRunLogger,
    monitor: SystemMonitor,
}

impl MasterControl {
    /// Create a new master control system.
    pub async fn new(config: MasterControlConfig) -> Result<Self> {
        info!(
            "Initializing master control with {} channels",
            config.num_channels
        );

        let state = Arc::new(RwLock::new(SystemState::default()));
        let failover = FailoverManager::new(config.failover.clone()).await?;
        let eas = if config.eas_enabled {
            Some(EasManager::new().await?)
        } else {
            None
        };
        let logger = AsRunLogger::new()?;
        let monitor = SystemMonitor::new(config.monitoring.clone()).await?;

        let mut channels = HashMap::new();
        for channel_id in 0..config.num_channels {
            let channel_config = ChannelConfig {
                id: channel_id,
                name: format!("Channel {}", channel_id + 1),
                ..Default::default()
            };
            let channel = ChannelAutomation::new(channel_config).await?;
            channels.insert(channel_id, channel);
        }

        Ok(Self {
            config,
            state,
            channels,
            failover,
            eas,
            logger,
            monitor,
        })
    }

    /// Start the master control system.
    pub async fn start(&mut self) -> Result<()> {
        info!("Starting master control system");

        {
            let mut state = self.state.write().await;
            state.status = SystemStatus::Starting;
        }

        // Start monitoring
        self.monitor.start().await?;

        // Start failover manager
        self.failover.start().await?;

        // Start EAS if enabled
        if let Some(ref mut eas) = self.eas {
            eas.start().await?;
        }

        // Start all channels
        for (id, channel) in &mut self.channels {
            info!("Starting channel {}", id);
            channel.start().await?;
        }

        {
            let mut state = self.state.write().await;
            state.status = SystemStatus::Running;
        }

        info!("Master control system started successfully");
        Ok(())
    }

    /// Stop the master control system.
    pub async fn stop(&mut self) -> Result<()> {
        info!("Stopping master control system");

        {
            let mut state = self.state.write().await;
            state.status = SystemStatus::Stopping;
        }

        // Stop all channels
        for (id, channel) in &mut self.channels {
            info!("Stopping channel {}", id);
            if let Err(e) = channel.stop().await {
                error!("Error stopping channel {}: {}", id, e);
            }
        }

        // Stop EAS
        if let Some(ref mut eas) = self.eas {
            eas.stop().await?;
        }

        // Stop failover
        self.failover.stop().await?;

        // Stop monitoring
        self.monitor.stop().await?;

        {
            let mut state = self.state.write().await;
            state.status = SystemStatus::Stopped;
        }

        info!("Master control system stopped");
        Ok(())
    }

    /// Get current system status.
    pub async fn status(&self) -> Result<SystemStatus> {
        let state = self.state.read().await;
        Ok(state.status)
    }

    /// Get a channel by ID.
    pub fn get_channel(&self, channel_id: usize) -> Option<&ChannelAutomation> {
        self.channels.get(&channel_id)
    }

    /// Get a mutable channel by ID.
    pub fn get_channel_mut(&mut self, channel_id: usize) -> Option<&mut ChannelAutomation> {
        self.channels.get_mut(&channel_id)
    }

    /// Handle emergency alert.
    pub async fn handle_alert(&mut self, alert: EasAlert) -> Result<()> {
        info!("Handling EAS alert: {:?}", alert);

        if let Some(ref mut eas) = self.eas {
            eas.handle_alert(alert).await?;
        } else {
            warn!("EAS not enabled, ignoring alert");
        }

        Ok(())
    }

    /// Trigger failover for a channel.
    pub async fn trigger_failover(&mut self, channel_id: usize) -> Result<()> {
        info!("Triggering failover for channel {}", channel_id);

        if !self.channels.contains_key(&channel_id) {
            return Err(AutomationError::NotFound(format!("Channel {channel_id}")));
        }

        self.failover.trigger_failover(channel_id).await?;

        Ok(())
    }

    /// Get system metrics.
    pub async fn metrics(&self) -> Result<HashMap<String, f64>> {
        self.monitor.metrics().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_master_control_creation() {
        let config = MasterControlConfig::default();
        let master = MasterControl::new(config).await;
        assert!(master.is_ok());
    }

    #[tokio::test]
    async fn test_master_control_lifecycle() {
        let config = MasterControlConfig {
            num_channels: 1,
            eas_enabled: false,
            ..Default::default()
        };
        let mut master = MasterControl::new(config)
            .await
            .expect("new should succeed");

        assert!(master.start().await.is_ok());
        assert_eq!(
            master.status().await.expect("value should be valid"),
            SystemStatus::Running
        );
        assert!(master.stop().await.is_ok());
    }

    #[tokio::test]
    async fn test_get_channel() {
        let config = MasterControlConfig {
            num_channels: 2,
            eas_enabled: false,
            ..Default::default()
        };
        let master = MasterControl::new(config)
            .await
            .expect("new should succeed");

        assert!(master.get_channel(0).is_some());
        assert!(master.get_channel(1).is_some());
        assert!(master.get_channel(2).is_none());
    }
}
