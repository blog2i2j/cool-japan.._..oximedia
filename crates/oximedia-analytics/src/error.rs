//! Error types for the analytics crate.

/// Errors produced by media engagement analytics operations.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum AnalyticsError {
    /// The viewer session is invalid or malformed.
    #[error("invalid session")]
    InvalidSession,

    /// An input value is invalid (bad range, empty slice, etc.).
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// Not enough data to perform the requested analysis.
    #[error("insufficient data: {0}")]
    InsufficientData(String),

    /// A statistical computation failed.
    #[error("statistical error: {0}")]
    StatisticalError(String),

    /// Configuration is invalid or missing.
    #[error("config error: {0}")]
    ConfigError(String),

    /// The experiment has no variants defined.
    #[error("experiment '{0}' has no variants")]
    NoVariants(String),

    /// All variant allocation weights are zero or negative.
    #[error("experiment '{0}' has invalid (zero/negative) weights")]
    InvalidWeights(String),
}
