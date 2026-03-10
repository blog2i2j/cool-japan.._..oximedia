//! Per-channel automation engine.

use crate::channel::playout::{PlayoutEngine, PlayoutState};
use crate::channel::switcher::AutomatedSwitcher;
use crate::device::control::{DeviceController, DeviceType};
use crate::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{error, info};

/// Channel automation configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelConfig {
    /// Channel ID
    pub id: usize,
    /// Channel name
    pub name: String,
    /// Enable live switching
    pub live_switching_enabled: bool,
    /// Enable device control
    pub device_control_enabled: bool,
    /// Devices to control
    pub devices: Vec<DeviceType>,
}

impl Default for ChannelConfig {
    fn default() -> Self {
        Self {
            id: 0,
            name: "Channel 1".to_string(),
            live_switching_enabled: false,
            device_control_enabled: false,
            devices: Vec::new(),
        }
    }
}

/// Channel automation state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChannelState {
    /// Channel is stopped
    Stopped,
    /// Channel is starting
    Starting,
    /// Channel is running
    Running,
    /// Channel is paused
    Paused,
    /// Channel has an error
    Error,
}

/// Per-channel automation engine.
///
/// Manages playout, live switching, and device control for a single channel.
pub struct ChannelAutomation {
    config: ChannelConfig,
    state: Arc<RwLock<ChannelState>>,
    playout: PlayoutEngine,
    switcher: Option<AutomatedSwitcher>,
    devices: HashMap<String, DeviceController>,
}

impl ChannelAutomation {
    /// Create a new channel automation engine.
    pub async fn new(config: ChannelConfig) -> Result<Self> {
        info!("Creating channel automation for: {}", config.name);

        let playout = PlayoutEngine::new(config.id).await?;

        let switcher = if config.live_switching_enabled {
            Some(AutomatedSwitcher::new(config.id).await?)
        } else {
            None
        };

        let mut devices = HashMap::new();
        if config.device_control_enabled {
            for device_type in &config.devices {
                let device_id = format!("{}_{}", config.id, device_type.name());
                let controller = DeviceController::new(device_type.clone()).await?;
                devices.insert(device_id, controller);
            }
        }

        Ok(Self {
            config,
            state: Arc::new(RwLock::new(ChannelState::Stopped)),
            playout,
            switcher,
            devices,
        })
    }

    /// Start channel automation.
    pub async fn start(&mut self) -> Result<()> {
        info!("Starting channel automation: {}", self.config.name);

        {
            let mut state = self.state.write().await;
            *state = ChannelState::Starting;
        }

        // Start playout engine
        self.playout.start().await?;

        // Start switcher if enabled
        if let Some(ref mut switcher) = self.switcher {
            switcher.start().await?;
        }

        // Initialize devices
        for (id, device) in &mut self.devices {
            info!("Initializing device: {}", id);
            if let Err(e) = device.initialize().await {
                error!("Failed to initialize device {}: {}", id, e);
            }
        }

        {
            let mut state = self.state.write().await;
            *state = ChannelState::Running;
        }

        info!("Channel automation started: {}", self.config.name);
        Ok(())
    }

    /// Stop channel automation.
    pub async fn stop(&mut self) -> Result<()> {
        info!("Stopping channel automation: {}", self.config.name);

        // Stop playout
        self.playout.stop().await?;

        // Stop switcher
        if let Some(ref mut switcher) = self.switcher {
            switcher.stop().await?;
        }

        // Release devices
        for (id, device) in &mut self.devices {
            info!("Releasing device: {}", id);
            if let Err(e) = device.release().await {
                error!("Failed to release device {}: {}", id, e);
            }
        }

        {
            let mut state = self.state.write().await;
            *state = ChannelState::Stopped;
        }

        Ok(())
    }

    /// Pause channel automation.
    pub async fn pause(&mut self) -> Result<()> {
        self.playout.pause().await?;

        let mut state = self.state.write().await;
        *state = ChannelState::Paused;

        Ok(())
    }

    /// Resume channel automation.
    pub async fn resume(&mut self) -> Result<()> {
        self.playout.resume().await?;

        let mut state = self.state.write().await;
        *state = ChannelState::Running;

        Ok(())
    }

    /// Get current channel state.
    pub async fn get_state(&self) -> ChannelState {
        *self.state.read().await
    }

    /// Get playout state.
    pub async fn playout_state(&self) -> Result<PlayoutState> {
        self.playout.get_state().await
    }

    /// Get channel configuration.
    pub fn config(&self) -> &ChannelConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_channel_automation_creation() {
        let config = ChannelConfig::default();
        let channel = ChannelAutomation::new(config).await;
        assert!(channel.is_ok());
    }

    #[tokio::test]
    async fn test_channel_lifecycle() {
        let config = ChannelConfig::default();
        let mut channel = ChannelAutomation::new(config)
            .await
            .expect("new should succeed");

        assert_eq!(channel.get_state().await, ChannelState::Stopped);
        assert!(channel.start().await.is_ok());
        assert_eq!(channel.get_state().await, ChannelState::Running);
        assert!(channel.pause().await.is_ok());
        assert_eq!(channel.get_state().await, ChannelState::Paused);
        assert!(channel.resume().await.is_ok());
        assert_eq!(channel.get_state().await, ChannelState::Running);
        assert!(channel.stop().await.is_ok());
        assert_eq!(channel.get_state().await, ChannelState::Stopped);
    }
}
