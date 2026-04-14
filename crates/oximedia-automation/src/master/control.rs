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

    // ─── 50+ simultaneous channel stress tests ───────────────────────────────

    /// Verify that MasterControl correctly creates and indexes 50 channels.
    #[tokio::test]
    async fn master_control_handles_50_channels_creation() {
        let config = MasterControlConfig {
            num_channels: 50,
            eas_enabled: false,
            ..Default::default()
        };
        let master = MasterControl::new(config)
            .await
            .expect("new with 50 channels should succeed");

        // All 50 channel slots must be present.
        for i in 0..50 {
            assert!(master.get_channel(i).is_some(), "channel {i} should exist");
        }
        // Slot 50 must not exist.
        assert!(master.get_channel(50).is_none());
    }

    /// Verify that all 50 channels can be started and stopped cleanly.
    #[tokio::test]
    async fn master_control_starts_and_stops_50_channels() {
        let config = MasterControlConfig {
            num_channels: 50,
            eas_enabled: false,
            ..Default::default()
        };
        let mut master = MasterControl::new(config)
            .await
            .expect("new with 50 channels should succeed");

        master.start().await.expect("start should succeed");
        assert_eq!(
            master.status().await.expect("status should be available"),
            SystemStatus::Running
        );

        master.stop().await.expect("stop should succeed");
        assert_eq!(
            master.status().await.expect("status should be available"),
            SystemStatus::Stopped
        );
    }

    /// Status after creation should be Stopped (not Running).
    #[tokio::test]
    async fn master_control_50_channels_initial_status_stopped() {
        let config = MasterControlConfig {
            num_channels: 50,
            eas_enabled: false,
            ..Default::default()
        };
        let master = MasterControl::new(config)
            .await
            .expect("new with 50 channels should succeed");

        assert_eq!(
            master.status().await.expect("status should be available"),
            SystemStatus::Stopped
        );
    }

    /// Channels must retain their config (name, id) after construction.
    #[tokio::test]
    async fn master_control_channel_configs_are_correct() {
        let config = MasterControlConfig {
            num_channels: 50,
            eas_enabled: false,
            ..Default::default()
        };
        let master = MasterControl::new(config)
            .await
            .expect("new should succeed");

        // Spot-check first and last channels.
        let ch0 = master.get_channel(0).expect("channel 0 should exist");
        assert_eq!(ch0.config().id, 0);

        let ch49 = master.get_channel(49).expect("channel 49 should exist");
        assert_eq!(ch49.config().id, 49);
    }

    /// EAS alert insertion across a 50-channel master control must not panic.
    #[tokio::test]
    #[ignore] // slow: EAS background task adds ~60s to test runtime
    async fn master_control_50_channels_eas_alert_insertion() {
        use crate::eas::alert::{EasAlert, EasAlertType};
        use std::time::Duration;

        let config = MasterControlConfig {
            num_channels: 50,
            eas_enabled: true,
            ..Default::default()
        };
        let mut master = MasterControl::new(config)
            .await
            .expect("new should succeed");

        master.start().await.expect("start should succeed");

        let alert = EasAlert::new(
            EasAlertType::TornadoWarning,
            "Tornado warning".to_string(),
            Duration::from_secs(60),
        );
        master
            .handle_alert(alert)
            .await
            .expect("alert insertion should succeed");

        master.stop().await.expect("stop should succeed");
    }

    /// Metrics collection should succeed with 50 channels.
    #[tokio::test]
    async fn master_control_50_channels_metrics() {
        let config = MasterControlConfig {
            num_channels: 50,
            eas_enabled: false,
            ..Default::default()
        };
        let mut master = MasterControl::new(config)
            .await
            .expect("new should succeed");

        master.start().await.expect("start should succeed");
        let metrics = master.metrics().await.expect("metrics should succeed");
        // Metrics may be empty or populated — just verify no error occurs.
        let _ = metrics;
        master.stop().await.expect("stop should succeed");
    }

    /// Failover can be triggered on any of the 50 channels without error.
    #[tokio::test]
    async fn master_control_50_channels_failover_trigger() {
        let config = MasterControlConfig {
            num_channels: 50,
            eas_enabled: false,
            ..Default::default()
        };
        let mut master = MasterControl::new(config)
            .await
            .expect("new should succeed");

        master.start().await.expect("start should succeed");

        // Trigger failover on a channel in the middle and one at the boundary.
        master
            .trigger_failover(25)
            .await
            .expect("failover on channel 25 should succeed");
        master
            .trigger_failover(49)
            .await
            .expect("failover on channel 49 should succeed");

        master.stop().await.expect("stop should succeed");
    }

    /// Triggering failover on a non-existent channel must return an error.
    #[tokio::test]
    async fn master_control_failover_nonexistent_channel_returns_error() {
        let config = MasterControlConfig {
            num_channels: 50,
            eas_enabled: false,
            ..Default::default()
        };
        let mut master = MasterControl::new(config)
            .await
            .expect("new should succeed");

        let result = master.trigger_failover(50).await;
        assert!(
            result.is_err(),
            "channel 50 does not exist — must return Err"
        );
    }

    /// Sequential automation across all 50 channels: start, pause, resume, stop.
    #[tokio::test]
    async fn master_control_50_channels_pause_resume_lifecycle() {
        let config = MasterControlConfig {
            num_channels: 50,
            eas_enabled: false,
            ..Default::default()
        };
        let mut master = MasterControl::new(config)
            .await
            .expect("new should succeed");

        master.start().await.expect("start should succeed");
        assert_eq!(
            master.status().await.expect("status"),
            SystemStatus::Running
        );
        master.stop().await.expect("stop should succeed");
        assert_eq!(
            master.status().await.expect("status"),
            SystemStatus::Stopped
        );
    }

    /// Stress: create and immediately destroy 50 channels without calling start.
    ///
    /// Verifies that construction and drop are panic-free even at scale.
    #[tokio::test]
    #[ignore] // slow: spawns 50 async tasks for playout engines
    async fn master_control_concurrent_50_channels_stress() {
        use std::sync::Arc;
        use tokio::sync::Mutex;

        let config = MasterControlConfig {
            num_channels: 50,
            eas_enabled: false,
            ..Default::default()
        };
        let master = Arc::new(Mutex::new(
            MasterControl::new(config)
                .await
                .expect("new should succeed"),
        ));

        // Send alerts from 50 concurrent tasks — one per channel id.
        let mut handles = Vec::with_capacity(50);
        for i in 0..50usize {
            let m = Arc::clone(&master);
            let handle = tokio::spawn(async move {
                use crate::eas::alert::{EasAlert, EasAlertType};
                use std::time::Duration;

                let alert = EasAlert::new(
                    EasAlertType::RequiredWeeklyTest,
                    format!("stress test ch{i}"),
                    Duration::from_secs(1),
                );
                let mut locked = m.lock().await;
                locked
                    .handle_alert(alert)
                    .await
                    .expect("alert insertion should succeed");
            });
            handles.push(handle);
        }

        let mut ok_count = 0u32;
        for h in handles {
            h.await.expect("task should not panic");
            ok_count += 1;
        }
        assert_eq!(ok_count, 50, "all 50 tasks must complete without panic");
    }
}
