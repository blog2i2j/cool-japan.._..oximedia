//! Error types for batch processing

use thiserror::Error;

/// Result type for batch operations
pub type Result<T> = std::result::Result<T, BatchError>;

/// Batch processing errors
#[derive(Error, Debug)]
pub enum BatchError {
    /// Job not found
    #[error("Job not found: {0}")]
    JobNotFound(String),

    /// Invalid job configuration
    #[error("Invalid job configuration: {0}")]
    InvalidJobConfig(String),

    /// Database error
    #[error("Database error: {0}")]
    DatabaseError(#[from] rusqlite::Error),

    /// I/O error
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    /// Template error
    #[error("Template error: {0}")]
    TemplateError(String),

    /// Execution error
    #[error("Execution error: {0}")]
    ExecutionError(String),

    /// Worker error
    #[error("Worker error: {0}")]
    WorkerError(String),

    /// Queue error
    #[error("Queue error: {0}")]
    QueueError(String),

    /// Script error
    #[error("Script error: {0}")]
    ScriptError(String),

    /// Watch folder error
    #[error("Watch folder error: {0}")]
    WatchError(String),

    /// API error
    #[error("API error: {0}")]
    ApiError(String),

    /// Validation error
    #[error("Validation error: {0}")]
    ValidationError(String),

    /// Dependency error
    #[error("Job dependency error: {0}")]
    DependencyError(String),

    /// Resource allocation error
    #[error("Resource allocation error: {0}")]
    ResourceError(String),

    /// Retry exhausted
    #[error("Retry attempts exhausted for job: {0}")]
    RetryExhausted(String),

    /// Cancelled
    #[error("Job cancelled: {0}")]
    Cancelled(String),

    /// Timeout
    #[error("Operation timeout: {0}")]
    Timeout(String),

    /// Pattern matching error
    #[error("Pattern matching error: {0}")]
    PatternError(String),

    /// File operation error
    #[error("File operation error: {0}")]
    FileOperationError(String),

    /// Media operation error
    #[error("Media operation error: {0}")]
    MediaOperationError(String),

    /// Integration error
    #[error("Integration error: {0}")]
    IntegrationError(String),
}

impl From<mlua::Error> for BatchError {
    fn from(err: mlua::Error) -> Self {
        Self::ScriptError(err.to_string())
    }
}

impl From<notify::Error> for BatchError {
    fn from(err: notify::Error) -> Self {
        Self::WatchError(err.to_string())
    }
}

impl From<regex::Error> for BatchError {
    fn from(err: regex::Error) -> Self {
        Self::PatternError(err.to_string())
    }
}

impl From<glob::PatternError> for BatchError {
    fn from(err: glob::PatternError) -> Self {
        Self::PatternError(err.to_string())
    }
}

impl From<r2d2::Error> for BatchError {
    fn from(err: r2d2::Error) -> Self {
        Self::FileOperationError(format!("Connection pool error: {err}"))
    }
}

impl From<walkdir::Error> for BatchError {
    fn from(err: walkdir::Error) -> Self {
        Self::FileOperationError(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = BatchError::JobNotFound("test-job".to_string());
        assert_eq!(err.to_string(), "Job not found: test-job");
    }

    #[test]
    fn test_invalid_config_error() {
        let err = BatchError::InvalidJobConfig("missing field".to_string());
        assert!(err.to_string().contains("Invalid job configuration"));
    }

    #[test]
    fn test_template_error() {
        let err = BatchError::TemplateError("invalid syntax".to_string());
        assert!(err.to_string().contains("Template error"));
    }

    #[test]
    fn test_execution_error() {
        let err = BatchError::ExecutionError("worker failed".to_string());
        assert!(err.to_string().contains("Execution error"));
    }

    #[test]
    fn test_retry_exhausted() {
        let err = BatchError::RetryExhausted("job-123".to_string());
        assert!(err.to_string().contains("Retry attempts exhausted"));
    }

    #[test]
    fn test_cancelled() {
        let err = BatchError::Cancelled("job-456".to_string());
        assert!(err.to_string().contains("Job cancelled"));
    }

    #[test]
    fn test_timeout() {
        let err = BatchError::Timeout("operation took too long".to_string());
        assert!(err.to_string().contains("Operation timeout"));
    }
}
