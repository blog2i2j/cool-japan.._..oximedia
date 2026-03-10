//! Configuration management for broadcast automation.

use crate::{
    channel::automation::ChannelConfig, device::control::DeviceType,
    failover::manager::FailoverConfig, master::control::MasterControlConfig,
    monitor::system::MonitorConfig, remote::server::RemoteConfig, AutomationError, Result,
};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

/// Complete automation system configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutomationConfig {
    /// Master control configuration
    pub master: MasterControlConfig,
    /// Channel configurations
    pub channels: Vec<ChannelConfig>,
    /// Remote control configuration
    pub remote: Option<RemoteConfig>,
    /// Global settings
    pub global: GlobalSettings,
}

/// Global automation settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalSettings {
    /// System name/identifier
    pub system_name: String,
    /// Facility ID
    pub facility_id: String,
    /// Time zone
    pub timezone: String,
    /// Frame rate (fps)
    pub frame_rate: f64,
    /// Enable debug logging
    pub debug_logging: bool,
    /// Log directory
    pub log_directory: String,
    /// Content root directory
    pub content_root: String,
}

impl Default for GlobalSettings {
    fn default() -> Self {
        Self {
            system_name: "OxiMedia Automation".to_string(),
            facility_id: "FAC001".to_string(),
            timezone: "America/New_York".to_string(),
            frame_rate: 29.97,
            debug_logging: false,
            log_directory: "/var/log/oximedia".to_string(),
            content_root: "/content".to_string(),
        }
    }
}

impl Default for AutomationConfig {
    fn default() -> Self {
        Self {
            master: MasterControlConfig::default(),
            channels: vec![],
            remote: Some(RemoteConfig::default()),
            global: GlobalSettings::default(),
        }
    }
}

impl AutomationConfig {
    /// Create a new automation configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Load configuration from JSON file.
    pub fn from_json_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut file = File::open(path).map_err(AutomationError::Io)?;

        let mut contents = String::new();
        file.read_to_string(&mut contents)
            .map_err(AutomationError::Io)?;

        let config: AutomationConfig = serde_json::from_str(&contents)?;
        Ok(config)
    }

    /// Save configuration to JSON file.
    pub fn to_json_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;

        let mut file = File::create(path).map_err(AutomationError::Io)?;

        file.write_all(json.as_bytes())
            .map_err(AutomationError::Io)?;

        Ok(())
    }

    /// Load configuration from TOML file.
    pub fn from_toml_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut file = File::open(path).map_err(AutomationError::Io)?;

        let mut contents = String::new();
        file.read_to_string(&mut contents)
            .map_err(AutomationError::Io)?;

        let config: AutomationConfig =
            toml::from_str(&contents).map_err(|e| AutomationError::Configuration(e.to_string()))?;

        Ok(config)
    }

    /// Save configuration to TOML file.
    pub fn to_toml_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let toml = toml::to_string_pretty(self)
            .map_err(|e| AutomationError::Configuration(e.to_string()))?;

        let mut file = File::create(path).map_err(AutomationError::Io)?;

        file.write_all(toml.as_bytes())
            .map_err(AutomationError::Io)?;

        Ok(())
    }

    /// Add a channel configuration.
    pub fn add_channel(&mut self, channel: ChannelConfig) {
        self.channels.push(channel);
    }

    /// Remove a channel configuration by ID.
    pub fn remove_channel(&mut self, channel_id: usize) {
        self.channels.retain(|ch| ch.id != channel_id);
    }

    /// Get channel configuration by ID.
    pub fn get_channel(&self, channel_id: usize) -> Option<&ChannelConfig> {
        self.channels.iter().find(|ch| ch.id == channel_id)
    }

    /// Validate configuration.
    pub fn validate(&self) -> Result<()> {
        // Check for duplicate channel IDs
        let mut ids = std::collections::HashSet::new();
        for channel in &self.channels {
            if !ids.insert(channel.id) {
                return Err(AutomationError::InvalidState(format!(
                    "Duplicate channel ID: {}",
                    channel.id
                )));
            }
        }

        // Validate frame rate
        if self.global.frame_rate <= 0.0 {
            return Err(AutomationError::InvalidState(
                "Invalid frame rate".to_string(),
            ));
        }

        // Validate directories exist or can be created
        // (In production, would check actual filesystem)

        Ok(())
    }
}

/// Configuration builder for fluent API.
pub struct ConfigBuilder {
    config: AutomationConfig,
}

impl ConfigBuilder {
    /// Create a new configuration builder.
    pub fn new() -> Self {
        Self {
            config: AutomationConfig::default(),
        }
    }

    /// Set system name.
    pub fn system_name(mut self, name: String) -> Self {
        self.config.global.system_name = name;
        self
    }

    /// Set facility ID.
    pub fn facility_id(mut self, id: String) -> Self {
        self.config.global.facility_id = id;
        self
    }

    /// Set time zone.
    pub fn timezone(mut self, tz: String) -> Self {
        self.config.global.timezone = tz;
        self
    }

    /// Set frame rate.
    pub fn frame_rate(mut self, fps: f64) -> Self {
        self.config.global.frame_rate = fps;
        self
    }

    /// Enable debug logging.
    pub fn debug_logging(mut self, enabled: bool) -> Self {
        self.config.global.debug_logging = enabled;
        self
    }

    /// Set log directory.
    pub fn log_directory(mut self, dir: String) -> Self {
        self.config.global.log_directory = dir;
        self
    }

    /// Set content root directory.
    pub fn content_root(mut self, root: String) -> Self {
        self.config.global.content_root = root;
        self
    }

    /// Set number of channels.
    pub fn num_channels(mut self, count: usize) -> Self {
        self.config.master.num_channels = count;
        self
    }

    /// Enable EAS.
    pub fn enable_eas(mut self, enabled: bool) -> Self {
        self.config.master.eas_enabled = enabled;
        self
    }

    /// Enable remote control.
    pub fn enable_remote(mut self, enabled: bool) -> Self {
        self.config.master.remote_enabled = enabled;
        self
    }

    /// Set failover configuration.
    pub fn failover_config(mut self, config: FailoverConfig) -> Self {
        self.config.master.failover = config;
        self
    }

    /// Set monitoring configuration.
    pub fn monitoring_config(mut self, config: MonitorConfig) -> Self {
        self.config.master.monitoring = config;
        self
    }

    /// Set remote control configuration.
    pub fn remote_config(mut self, config: RemoteConfig) -> Self {
        self.config.remote = Some(config);
        self
    }

    /// Add a channel.
    pub fn add_channel(mut self, channel: ChannelConfig) -> Self {
        self.config.channels.push(channel);
        self
    }

    /// Build the configuration.
    pub fn build(self) -> Result<AutomationConfig> {
        self.config.validate()?;
        Ok(self.config)
    }
}

impl Default for ConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Preset configurations for common broadcast scenarios.
pub struct ConfigPresets;

impl ConfigPresets {
    /// Single-channel basic configuration.
    pub fn single_channel_basic() -> AutomationConfig {
        let mut config = AutomationConfig::default();
        config.master.num_channels = 1;
        config.master.eas_enabled = false;
        config.master.remote_enabled = true;

        let channel = ChannelConfig {
            id: 0,
            name: "Main Channel".to_string(),
            live_switching_enabled: false,
            device_control_enabled: false,
            devices: vec![],
        };

        config.channels.push(channel);
        config
    }

    /// Multi-channel professional configuration.
    pub fn multi_channel_professional() -> AutomationConfig {
        let mut config = AutomationConfig::default();
        config.master.num_channels = 4;
        config.master.eas_enabled = true;
        config.master.remote_enabled = true;

        config.master.failover = FailoverConfig {
            auto_failover: true,
            health_check_interval: 5,
            failure_threshold: 3,
            switch_delay_ms: 100,
        };

        config.master.monitoring = MonitorConfig {
            collection_interval: 10,
            monitor_cpu: true,
            monitor_memory: true,
            monitor_disk: true,
            monitor_network: true,
        };

        for i in 0..4 {
            let channel = ChannelConfig {
                id: i,
                name: format!("Channel {}", i + 1),
                live_switching_enabled: i == 0, // Main channel has live switching
                device_control_enabled: true,
                devices: vec![DeviceType::Sony9Pin {
                    port: format!("/dev/ttyS{i}"),
                }],
            };

            config.channels.push(channel);
        }

        config
    }

    /// News station configuration.
    pub fn news_station() -> AutomationConfig {
        let mut config = AutomationConfig::default();
        config.master.num_channels = 2;
        config.master.eas_enabled = true;
        config.master.remote_enabled = true;

        config.global.system_name = "News Station Automation".to_string();

        // Main news channel
        let main_channel = ChannelConfig {
            id: 0,
            name: "Main News Channel".to_string(),
            live_switching_enabled: true,
            device_control_enabled: true,
            devices: vec![
                DeviceType::Sony9Pin {
                    port: "/dev/ttyS0".to_string(),
                },
                DeviceType::Vdcp {
                    port: "/dev/ttyS1".to_string(),
                },
                DeviceType::Gpo {
                    port: "/dev/gpio0".to_string(),
                },
            ],
        };

        // Weather/alternate channel
        let weather_channel = ChannelConfig {
            id: 1,
            name: "Weather Channel".to_string(),
            live_switching_enabled: false,
            device_control_enabled: true,
            devices: vec![DeviceType::Sony9Pin {
                port: "/dev/ttyS2".to_string(),
            }],
        };

        config.channels.push(main_channel);
        config.channels.push(weather_channel);

        config
    }

    /// 24/7 entertainment channel configuration.
    pub fn entertainment_247() -> AutomationConfig {
        let mut config = AutomationConfig::default();
        config.master.num_channels = 1;
        config.master.eas_enabled = true;
        config.master.remote_enabled = true;

        config.global.system_name = "24/7 Entertainment".to_string();

        config.master.failover = FailoverConfig {
            auto_failover: true,
            health_check_interval: 10,
            failure_threshold: 3,
            switch_delay_ms: 200,
        };

        let channel = ChannelConfig {
            id: 0,
            name: "Entertainment Channel".to_string(),
            live_switching_enabled: false,
            device_control_enabled: true,
            devices: vec![DeviceType::Vdcp {
                port: "/dev/ttyS0".to_string(),
            }],
        };

        config.channels.push(channel);

        config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_creation() {
        let config = AutomationConfig::new();
        assert_eq!(config.channels.len(), 0);
    }

    #[test]
    fn test_config_builder() {
        let config = ConfigBuilder::new()
            .system_name("Test System".to_string())
            .num_channels(2)
            .enable_eas(true)
            .build();

        assert!(config.is_ok());
        let config = config.expect("config should be valid");
        assert_eq!(config.global.system_name, "Test System");
        assert_eq!(config.master.num_channels, 2);
        assert!(config.master.eas_enabled);
    }

    #[test]
    fn test_add_remove_channel() {
        let mut config = AutomationConfig::new();

        let channel = ChannelConfig {
            id: 0,
            name: "Test Channel".to_string(),
            live_switching_enabled: false,
            device_control_enabled: false,
            devices: vec![],
        };

        config.add_channel(channel);
        assert_eq!(config.channels.len(), 1);

        config.remove_channel(0);
        assert_eq!(config.channels.len(), 0);
    }

    #[test]
    fn test_preset_single_channel() {
        let config = ConfigPresets::single_channel_basic();
        assert_eq!(config.master.num_channels, 1);
        assert_eq!(config.channels.len(), 1);
        assert!(!config.master.eas_enabled);
    }

    #[test]
    fn test_preset_professional() {
        let config = ConfigPresets::multi_channel_professional();
        assert_eq!(config.master.num_channels, 4);
        assert_eq!(config.channels.len(), 4);
        assert!(config.master.eas_enabled);
        assert!(config.master.failover.auto_failover);
    }

    #[test]
    fn test_preset_news_station() {
        let config = ConfigPresets::news_station();
        assert_eq!(config.master.num_channels, 2);
        assert_eq!(config.channels.len(), 2);
        assert!(config.channels[0].live_switching_enabled);
        assert!(!config.channels[1].live_switching_enabled);
    }

    #[test]
    fn test_validation() {
        let config = AutomationConfig::new();
        assert!(config.validate().is_ok());

        // Test duplicate channel IDs
        let mut bad_config = AutomationConfig::new();
        bad_config.add_channel(ChannelConfig {
            id: 0,
            name: "Channel 1".to_string(),
            live_switching_enabled: false,
            device_control_enabled: false,
            devices: vec![],
        });
        bad_config.add_channel(ChannelConfig {
            id: 0, // Duplicate!
            name: "Channel 2".to_string(),
            live_switching_enabled: false,
            device_control_enabled: false,
            devices: vec![],
        });

        assert!(bad_config.validate().is_err());
    }
}
