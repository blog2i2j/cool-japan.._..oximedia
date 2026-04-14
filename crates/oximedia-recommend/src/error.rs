//! Error types for recommendation engine.

use thiserror::Error;

/// Result type for recommendation operations
pub type RecommendResult<T> = Result<T, RecommendError>;

/// Errors that can occur during recommendation operations
#[derive(Debug, Error)]
pub enum RecommendError {
    /// User not found
    #[error("User not found: {0}")]
    UserNotFound(uuid::Uuid),

    /// Content not found
    #[error("Content not found: {0}")]
    ContentNotFound(uuid::Uuid),

    /// Insufficient data for recommendation
    #[error("Insufficient data: {0}")]
    InsufficientData(String),

    /// Invalid rating value
    #[error("Invalid rating: {0}")]
    InvalidRating(f32),

    /// Invalid similarity score
    #[error("Invalid similarity score: {0}")]
    InvalidSimilarity(f32),

    /// Matrix computation error
    #[error("Matrix computation error: {0}")]
    MatrixError(String),

    /// Profile error
    #[error("Profile error: {0}")]
    ProfileError(String),

    /// History tracking error
    #[error("History tracking error: {0}")]
    HistoryError(String),

    /// Trending detection error
    #[error("Trending detection error: {0}")]
    TrendingError(String),

    /// Personalization error
    #[error("Personalization error: {0}")]
    PersonalizationError(String),

    /// Diversity enforcement error
    #[error("Diversity enforcement error: {0}")]
    DiversityError(String),

    /// Ranking error
    #[error("Ranking error: {0}")]
    RankingError(String),

    /// Explanation generation error
    #[error("Explanation generation error: {0}")]
    ExplanationError(String),

    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    /// Request was rate-limited
    #[error("Rate limited: {0}")]
    RateLimited(String),

    /// Generic error
    #[error("{0}")]
    Other(String),
}

impl RecommendError {
    /// Create an insufficient data error
    #[must_use]
    pub fn insufficient_data(msg: impl Into<String>) -> Self {
        Self::InsufficientData(msg.into())
    }

    /// Create a matrix error
    #[must_use]
    pub fn matrix_error(msg: impl Into<String>) -> Self {
        Self::MatrixError(msg.into())
    }

    /// Create a profile error
    #[must_use]
    pub fn profile_error(msg: impl Into<String>) -> Self {
        Self::ProfileError(msg.into())
    }

    /// Create a history error
    #[must_use]
    pub fn history_error(msg: impl Into<String>) -> Self {
        Self::HistoryError(msg.into())
    }

    /// Create a trending error
    #[must_use]
    pub fn trending_error(msg: impl Into<String>) -> Self {
        Self::TrendingError(msg.into())
    }

    /// Create a personalization error
    #[must_use]
    pub fn personalization_error(msg: impl Into<String>) -> Self {
        Self::PersonalizationError(msg.into())
    }

    /// Create a diversity error
    #[must_use]
    pub fn diversity_error(msg: impl Into<String>) -> Self {
        Self::DiversityError(msg.into())
    }

    /// Create a ranking error
    #[must_use]
    pub fn ranking_error(msg: impl Into<String>) -> Self {
        Self::RankingError(msg.into())
    }

    /// Create an explanation error
    #[must_use]
    pub fn explanation_error(msg: impl Into<String>) -> Self {
        Self::ExplanationError(msg.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn test_user_not_found_error() {
        let id = Uuid::new_v4();
        let error = RecommendError::UserNotFound(id);
        assert!(error.to_string().contains("User not found"));
    }

    #[test]
    fn test_insufficient_data_error() {
        let error = RecommendError::insufficient_data("Not enough ratings");
        assert!(error.to_string().contains("Insufficient data"));
    }

    #[test]
    fn test_invalid_rating_error() {
        let error = RecommendError::InvalidRating(-1.0);
        assert!(error.to_string().contains("Invalid rating"));
    }
}
