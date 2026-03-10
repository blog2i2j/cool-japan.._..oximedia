//! Content Rights and Licensing Management for OxiMedia
//!
//! This crate provides comprehensive rights management capabilities including:
//! - Rights tracking and ownership management
//! - License management with various types (royalty-free, rights-managed, etc.)
//! - Expiration tracking and renewal management
//! - Geographic territory restrictions
//! - Usage tracking and reporting
//! - Clearance tracking (music, footage, talent)
//! - Royalty calculation and payment tracking
//! - Watermarking integration
//! - DRM metadata management
//! - Audit trail and compliance reporting

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod audit;
pub mod clearance;

// Rights holder registry, usage reporting, and distribution rights
pub mod clearance_workflow;
pub mod compliance;
pub mod contract;
pub mod database;
pub mod distribution_rights;
pub mod distribution_window;
pub mod drm;
pub mod embargo;
pub mod embargo_policy;
pub mod embargo_window;
pub mod expiration;
pub mod license;
pub mod license_template;
pub mod licensing_model;
pub mod licensing_terms;
pub mod registry;
pub mod rights;
pub mod rights_audit_trail;
pub mod rights_bundle;
pub mod rights_check;
pub mod rights_conflict;
pub mod rights_database;
pub mod rights_holder;
pub mod rights_negotiation;
pub mod rights_timeline;
pub mod royalty;
pub mod royalty_calc;
pub mod royalty_schedule;
pub mod sync_rights;
pub mod syndication;
pub mod territory;
pub mod usage;
pub mod usage_report;
pub mod usage_rights;
pub mod watermark;

use thiserror::Error;

/// Result type for rights management operations
pub type Result<T> = std::result::Result<T, RightsError>;

/// Errors that can occur in rights management operations
#[derive(Error, Debug)]
pub enum RightsError {
    /// Database error
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),

    /// Rights not found
    #[error("Rights not found: {0}")]
    NotFound(String),

    /// Rights expired
    #[error("Rights expired: {0}")]
    Expired(String),

    /// Territory restriction violation
    #[error("Territory restriction violation: {0}")]
    TerritoryViolation(String),

    /// Usage restriction violation
    #[error("Usage restriction violation: {0}")]
    UsageViolation(String),

    /// License not valid
    #[error("License not valid: {0}")]
    InvalidLicense(String),

    /// Clearance not obtained
    #[error("Clearance not obtained: {0}")]
    ClearanceRequired(String),

    /// Royalty calculation error
    #[error("Royalty calculation error: {0}")]
    RoyaltyError(String),

    /// Watermark error
    #[error("Watermark error: {0}")]
    WatermarkError(String),

    /// DRM error
    #[error("DRM error: {0}")]
    DrmError(String),

    /// Compliance violation
    #[error("Compliance violation: {0}")]
    ComplianceViolation(String),

    /// Invalid operation
    #[error("Invalid operation: {0}")]
    InvalidOperation(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Main rights management system
pub struct RightsManager {
    db: database::RightsDatabase,
}

impl RightsManager {
    /// Create a new rights manager with the specified database path
    pub async fn new(database_path: &str) -> Result<Self> {
        let db = database::RightsDatabase::new(database_path).await?;
        Ok(Self { db })
    }

    /// Get reference to the database
    pub fn database(&self) -> &database::RightsDatabase {
        &self.db
    }

    /// Get mutable reference to the database
    pub fn database_mut(&mut self) -> &mut database::RightsDatabase {
        &mut self.db
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_rights_manager_creation() {
        let temp_dir = tempfile::tempdir().expect("rights test operation should succeed");
        let db_path = temp_dir.path().join("rights.db");
        let manager = RightsManager::new(
            db_path
                .to_str()
                .expect("rights test operation should succeed"),
        )
        .await;
        assert!(manager.is_ok());
    }
}
