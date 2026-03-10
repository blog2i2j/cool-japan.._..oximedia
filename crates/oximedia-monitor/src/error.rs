//! Error types for the oximedia-monitor crate.

use thiserror::Error;

/// Result type for monitor operations.
pub type MonitorResult<T> = Result<T, MonitorError>;

/// Error types for the monitoring system.
#[derive(Error, Debug)]
pub enum MonitorError {
    /// Configuration error.
    #[error("Configuration error: {0}")]
    Config(String),

    /// Metrics collection error.
    #[error("Metrics collection error: {0}")]
    MetricsCollection(String),

    /// Storage error.
    #[error("Storage error: {0}")]
    Storage(String),

    /// Database error.
    #[error("Database error: {0}")]
    Database(#[from] rusqlite::Error),

    /// Alert error.
    #[error("Alert error: {0}")]
    Alert(String),

    /// API error.
    #[error("API error: {0}")]
    Api(String),

    /// Health check error.
    #[error("Health check error: {0}")]
    HealthCheck(String),

    /// Log error.
    #[error("Log error: {0}")]
    Log(String),

    /// Channel error.
    #[error("Channel error: {0}")]
    Channel(String),

    /// Serialization error.
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// HTTP error.
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    /// Email error.
    #[error("Email error: {0}")]
    Email(String),

    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Invalid metric name.
    #[error("Invalid metric name: {0}")]
    InvalidMetricName(String),

    /// Invalid time range.
    #[error("Invalid time range: {0}")]
    InvalidTimeRange(String),

    /// Query error.
    #[error("Query error: {0}")]
    Query(String),

    /// Aggregation error.
    #[error("Aggregation error: {0}")]
    Aggregation(String),

    /// Rule error.
    #[error("Rule error: {0}")]
    Rule(String),

    /// System error.
    #[error("System error: {0}")]
    System(String),

    /// GPU error.
    #[cfg(feature = "gpu")]
    #[error("GPU error: {0}")]
    Gpu(String),

    /// Integration error.
    #[cfg(feature = "integrations")]
    #[error("Integration error: {0}")]
    Integration(String),

    /// Other error.
    #[error("{0}")]
    Other(String),
}

impl From<String> for MonitorError {
    fn from(s: String) -> Self {
        Self::Other(s)
    }
}

impl From<&str> for MonitorError {
    fn from(s: &str) -> Self {
        Self::Other(s.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = MonitorError::Config("test error".to_string());
        assert_eq!(err.to_string(), "Configuration error: test error");
    }

    #[test]
    fn test_error_from_string() {
        let err: MonitorError = "test error".into();
        assert_eq!(err.to_string(), "test error");
    }

    #[test]
    fn test_result_type() {
        let result: MonitorResult<i32> = Ok(42);
        assert!(result.is_ok());
        assert_eq!(result.expect("result should be valid"), 42);
    }

    #[test]
    fn test_error_conversion() {
        let json_err = serde_json::from_str::<i32>("invalid").unwrap_err();
        let monitor_err: MonitorError = json_err.into();
        assert!(matches!(monitor_err, MonitorError::Serialization(_)));
    }
}
