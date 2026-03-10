//! Configuration for the monitoring system.

use crate::error::{MonitorError, MonitorResult};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;

/// Main configuration for the monitoring system.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MonitorConfig {
    /// Metrics collection configuration.
    pub metrics: MetricsConfig,

    /// Storage configuration.
    pub storage: StorageConfig,

    /// Alert configuration.
    pub alerts: AlertConfig,

    /// API configuration.
    pub api: ApiConfig,

    /// Health check configuration.
    pub health: HealthConfig,

    /// Log configuration.
    pub logs: LogConfig,
}

impl MonitorConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if any configuration values are invalid.
    pub fn validate(&self) -> MonitorResult<()> {
        self.metrics.validate()?;
        self.storage.validate()?;
        self.alerts.validate()?;
        self.api.validate()?;
        self.health.validate()?;
        self.logs.validate()?;
        Ok(())
    }
}

/// Configuration for metrics collection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    /// Enable system metrics collection.
    pub enable_system_metrics: bool,

    /// Enable application metrics collection.
    pub enable_application_metrics: bool,

    /// Enable quality metrics collection.
    pub enable_quality_metrics: bool,

    /// Metrics collection interval.
    #[serde(with = "serde_duration")]
    pub collection_interval: Duration,

    /// Enable CPU per-core metrics.
    pub cpu_per_core: bool,

    /// Enable GPU metrics (requires gpu feature).
    #[cfg(feature = "gpu")]
    pub enable_gpu_metrics: bool,

    /// Enable temperature monitoring.
    pub enable_temperature: bool,

    /// Enable disk I/O metrics.
    ///
    /// On systems with many mount points (e.g. macOS with app-wrapper volumes)
    /// disk enumeration can be very slow.  Set this to `false` to skip disk
    /// collection entirely.  Defaults to `true`.
    pub enable_disk_metrics: bool,

    /// Maximum CPU overhead (0.0 to 1.0).
    pub max_cpu_overhead: f64,

    /// Maximum memory overhead in MB.
    pub max_memory_overhead_mb: usize,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            enable_system_metrics: true,
            enable_application_metrics: true,
            enable_quality_metrics: true,
            collection_interval: Duration::from_secs(1),
            cpu_per_core: true,
            #[cfg(feature = "gpu")]
            enable_gpu_metrics: true,
            enable_temperature: true,
            enable_disk_metrics: true,
            max_cpu_overhead: 0.05, // 5%
            max_memory_overhead_mb: 100,
        }
    }
}

impl MetricsConfig {
    /// Validate metrics configuration.
    pub fn validate(&self) -> MonitorResult<()> {
        if self.collection_interval < Duration::from_millis(100) {
            return Err(MonitorError::Config(
                "Collection interval must be at least 100ms".to_string(),
            ));
        }

        if self.max_cpu_overhead < 0.0 || self.max_cpu_overhead > 1.0 {
            return Err(MonitorError::Config(
                "Max CPU overhead must be between 0.0 and 1.0".to_string(),
            ));
        }

        Ok(())
    }
}

/// Configuration for time series storage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Database file path.
    pub db_path: PathBuf,

    /// Ring buffer capacity (number of data points per metric).
    pub ring_buffer_capacity: usize,

    /// Retention configuration.
    pub retention: RetentionConfig,

    /// Enable compression.
    pub enable_compression: bool,

    /// Maximum database size in MB (0 = unlimited).
    pub max_db_size_mb: usize,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            db_path: PathBuf::from("./data/monitor.db"),
            ring_buffer_capacity: 86400, // 24 hours at 1 sample/second
            retention: RetentionConfig::default(),
            enable_compression: true,
            max_db_size_mb: 1024, // 1GB
        }
    }
}

impl StorageConfig {
    /// Validate storage configuration.
    pub fn validate(&self) -> MonitorResult<()> {
        if self.ring_buffer_capacity == 0 {
            return Err(MonitorError::Config(
                "Ring buffer capacity must be greater than 0".to_string(),
            ));
        }

        self.retention.validate()?;
        Ok(())
    }
}

/// Data retention configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionConfig {
    /// Raw data retention duration.
    #[serde(with = "serde_duration")]
    pub raw_data: Duration,

    /// 1-minute aggregates retention.
    #[serde(with = "serde_duration")]
    pub minute_aggregates: Duration,

    /// 1-hour aggregates retention.
    #[serde(with = "serde_duration")]
    pub hour_aggregates: Duration,

    /// 1-day aggregates retention.
    #[serde(with = "serde_duration")]
    pub day_aggregates: Duration,
}

impl Default for RetentionConfig {
    fn default() -> Self {
        Self {
            raw_data: Duration::from_secs(24 * 3600), // 24 hours
            minute_aggregates: Duration::from_secs(7 * 24 * 3600), // 7 days
            hour_aggregates: Duration::from_secs(30 * 24 * 3600), // 30 days
            day_aggregates: Duration::from_secs(365 * 24 * 3600), // 1 year
        }
    }
}

impl RetentionConfig {
    /// Validate retention configuration.
    pub fn validate(&self) -> MonitorResult<()> {
        if self.raw_data < Duration::from_secs(3600) {
            return Err(MonitorError::Config(
                "Raw data retention must be at least 1 hour".to_string(),
            ));
        }
        Ok(())
    }
}

/// Alert system configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertConfig {
    /// Enable alerting.
    pub enabled: bool,

    /// Alert deduplication window.
    #[serde(with = "serde_duration")]
    pub dedup_window: Duration,

    /// Maximum alerts per hour.
    pub max_alerts_per_hour: usize,

    /// Enable email alerts.
    pub enable_email: bool,

    /// Email configuration.
    pub email: Option<EmailConfig>,

    /// Enable Slack alerts.
    pub enable_slack: bool,

    /// Slack webhook URL.
    pub slack_webhook_url: Option<String>,

    /// Enable Discord alerts.
    pub enable_discord: bool,

    /// Discord webhook URL.
    pub discord_webhook_url: Option<String>,

    /// Enable generic webhook.
    pub enable_webhook: bool,

    /// Generic webhook URL.
    pub webhook_url: Option<String>,

    /// Enable SMS alerts.
    pub enable_sms: bool,

    /// SMS configuration.
    pub sms: Option<SmsConfig>,

    /// Enable file logging.
    pub enable_file_logging: bool,

    /// Alert log file path.
    pub alert_log_path: Option<PathBuf>,
}

impl Default for AlertConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            dedup_window: Duration::from_secs(300), // 5 minutes
            max_alerts_per_hour: 100,
            enable_email: false,
            email: None,
            enable_slack: false,
            slack_webhook_url: None,
            enable_discord: false,
            discord_webhook_url: None,
            enable_webhook: false,
            webhook_url: None,
            enable_sms: false,
            sms: None,
            enable_file_logging: true,
            alert_log_path: Some(PathBuf::from("./data/alerts.log")),
        }
    }
}

impl AlertConfig {
    /// Validate alert configuration.
    pub fn validate(&self) -> MonitorResult<()> {
        if self.enabled {
            if self.enable_email && self.email.is_none() {
                return Err(MonitorError::Config(
                    "Email enabled but no email configuration provided".to_string(),
                ));
            }

            if self.enable_slack && self.slack_webhook_url.is_none() {
                return Err(MonitorError::Config(
                    "Slack enabled but no webhook URL provided".to_string(),
                ));
            }

            if self.enable_discord && self.discord_webhook_url.is_none() {
                return Err(MonitorError::Config(
                    "Discord enabled but no webhook URL provided".to_string(),
                ));
            }

            if self.enable_webhook && self.webhook_url.is_none() {
                return Err(MonitorError::Config(
                    "Webhook enabled but no URL provided".to_string(),
                ));
            }

            if self.enable_sms && self.sms.is_none() {
                return Err(MonitorError::Config(
                    "SMS enabled but no SMS configuration provided".to_string(),
                ));
            }
        }

        Ok(())
    }
}

/// Email configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailConfig {
    /// SMTP server.
    pub smtp_server: String,

    /// SMTP port.
    pub smtp_port: u16,

    /// SMTP username.
    pub smtp_username: String,

    /// SMTP password.
    pub smtp_password: String,

    /// From email address.
    pub from_address: String,

    /// To email addresses.
    pub to_addresses: Vec<String>,

    /// Use TLS.
    pub use_tls: bool,
}

/// SMS configuration (Twilio).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmsConfig {
    /// Twilio account SID.
    pub account_sid: String,

    /// Twilio auth token.
    pub auth_token: String,

    /// From phone number.
    pub from_number: String,

    /// To phone numbers.
    pub to_numbers: Vec<String>,
}

/// API configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiConfig {
    /// Enable REST API.
    pub enabled: bool,

    /// API bind address.
    pub bind_address: String,

    /// API port.
    pub port: u16,

    /// Enable WebSocket streaming.
    pub enable_websocket: bool,

    /// Enable CORS.
    pub enable_cors: bool,

    /// Enable Prometheus exposition.
    pub enable_prometheus: bool,

    /// Prometheus path.
    pub prometheus_path: String,

    /// Enable authentication.
    pub enable_auth: bool,

    /// API token (if auth enabled).
    pub api_token: Option<String>,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            bind_address: "127.0.0.1".to_string(),
            port: 8080,
            enable_websocket: true,
            enable_cors: true,
            enable_prometheus: true,
            prometheus_path: "/metrics".to_string(),
            enable_auth: false,
            api_token: None,
        }
    }
}

impl ApiConfig {
    /// Validate API configuration.
    pub fn validate(&self) -> MonitorResult<()> {
        if self.enabled {
            if self.port == 0 {
                return Err(MonitorError::Config("Invalid API port".to_string()));
            }

            if self.enable_auth && self.api_token.is_none() {
                return Err(MonitorError::Config(
                    "Authentication enabled but no API token provided".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Get the full bind address.
    #[must_use]
    pub fn bind_addr(&self) -> String {
        format!("{}:{}", self.bind_address, self.port)
    }
}

/// Health check configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthConfig {
    /// Enable health checks.
    pub enabled: bool,

    /// Health check interval.
    #[serde(with = "serde_duration")]
    pub check_interval: Duration,

    /// Disk space warning threshold (percentage).
    pub disk_warning_threshold: f64,

    /// Disk space critical threshold (percentage).
    pub disk_critical_threshold: f64,

    /// Database connection timeout.
    #[serde(with = "serde_duration")]
    pub db_timeout: Duration,
}

impl Default for HealthConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            check_interval: Duration::from_secs(30),
            disk_warning_threshold: 0.8,   // 80%
            disk_critical_threshold: 0.95, // 95%
            db_timeout: Duration::from_secs(5),
        }
    }
}

impl HealthConfig {
    /// Validate health configuration.
    pub fn validate(&self) -> MonitorResult<()> {
        if self.disk_warning_threshold >= self.disk_critical_threshold {
            return Err(MonitorError::Config(
                "Disk warning threshold must be less than critical threshold".to_string(),
            ));
        }

        Ok(())
    }
}

/// Log configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogConfig {
    /// Enable log aggregation.
    pub enabled: bool,

    /// Log database path.
    pub db_path: PathBuf,

    /// Log retention duration.
    #[serde(with = "serde_duration")]
    pub retention: Duration,

    /// Enable full-text search.
    pub enable_fts: bool,

    /// Maximum log entries per table.
    pub max_entries: usize,
}

impl Default for LogConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            db_path: PathBuf::from("./data/logs.db"),
            retention: Duration::from_secs(7 * 24 * 3600), // 7 days
            enable_fts: true,
            max_entries: 1_000_000,
        }
    }
}

impl LogConfig {
    /// Validate log configuration.
    pub fn validate(&self) -> MonitorResult<()> {
        if self.enabled && self.max_entries == 0 {
            return Err(MonitorError::Config(
                "Max log entries must be greater than 0".to_string(),
            ));
        }

        Ok(())
    }
}

/// Helper module for serializing/deserializing Duration.
mod serde_duration {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u64(duration.as_secs())
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs = u64::deserialize(deserializer)?;
        Ok(Duration::from_secs(secs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = MonitorConfig::default();
        assert!(config.metrics.enable_system_metrics);
        assert!(config.storage.enable_compression);
        assert!(config.api.enabled);
    }

    #[test]
    fn test_config_validation() {
        let config = MonitorConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_invalid_collection_interval() {
        let mut config = MetricsConfig::default();
        config.collection_interval = Duration::from_millis(50);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_invalid_cpu_overhead() {
        let mut config = MetricsConfig::default();
        config.max_cpu_overhead = 1.5;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_api_bind_addr() {
        let config = ApiConfig::default();
        assert_eq!(config.bind_addr(), "127.0.0.1:8080");
    }

    #[test]
    fn test_storage_validation() {
        let mut config = StorageConfig::default();
        config.ring_buffer_capacity = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_retention_validation() {
        let mut config = RetentionConfig::default();
        config.raw_data = Duration::from_secs(1800); // 30 minutes
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_alert_config_validation_slack() {
        let mut config = AlertConfig::default();
        config.enable_slack = true;
        config.slack_webhook_url = None;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_health_config_validation() {
        let mut config = HealthConfig::default();
        config.disk_warning_threshold = 0.95;
        config.disk_critical_threshold = 0.80;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_serde_config() {
        let config = MonitorConfig::default();
        let json = serde_json::to_string(&config).expect("failed to serialize to JSON");
        let deserialized: MonitorConfig =
            serde_json::from_str(&json).expect("failed to deserialize from JSON");
        assert_eq!(
            config.metrics.collection_interval,
            deserialized.metrics.collection_interval
        );
    }
}
