//! OxiMedia Archive Verification System
//!
//! This crate provides comprehensive media archive verification and long-term preservation
//! capabilities, including checksumming, fixity checking, validation, and OAIS compliance.

pub mod archive_verify;
pub mod asset_manifest;
pub mod audit_trail;
pub mod bagit_profile;
pub mod batch_archive;
pub mod catalog;
pub mod catalog_export;
#[cfg(feature = "sqlite")]
pub mod checksum;
pub mod cloud_backend;
pub mod compliance_check;
pub mod dedup_archive;
pub mod dedup_report;
#[cfg(feature = "sqlite")]
pub mod fixity;
pub mod format_registry;
pub mod health_dashboard;
pub mod incremental_checksum;
pub mod indexing;
pub mod ingest_log;
pub mod integrity;
pub mod integrity_scan;
pub mod ltfs;
pub mod media_fingerprint;
pub mod media_validate;
pub mod migration;
pub mod migration_dryrun;
pub mod notification;
pub mod parallel_checksum;
pub mod partial_restore;
pub mod preservation;
pub mod preservation_policy;
#[cfg(feature = "sqlite")]
pub mod quarantine;
pub mod quarantine_policy;
#[cfg(feature = "sqlite")]
pub mod report;
pub mod restore_plan;
pub mod retention_enforce;
pub mod retention_schedule;
pub mod search_index;
pub mod sidecar;
pub mod split_archive;
pub mod stats;
pub mod storage_tier;
pub mod streaming_compress;
pub mod tape;
pub mod validate;
pub mod version_history;
pub mod versioning;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
#[cfg(feature = "sqlite")]
use std::path::Path;
use std::path::PathBuf;
use thiserror::Error;

/// Archive verification error types
#[derive(Error, Debug)]
pub enum ArchiveError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[cfg(feature = "sqlite")]
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),

    #[cfg(not(feature = "sqlite"))]
    #[error("Database error: {0}")]
    Database(String),

    #[error("Checksum mismatch: expected {expected}, got {actual}")]
    ChecksumMismatch { expected: String, actual: String },

    #[error("Validation error: {0}")]
    Validation(String),

    #[error("Corruption detected: {0}")]
    Corruption(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Quarantine error: {0}")]
    Quarantine(String),

    #[error("Report generation error: {0}")]
    Report(String),
}

/// Result type for archive operations
pub type ArchiveResult<T> = Result<T, ArchiveError>;

/// Verification configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationConfig {
    /// Enable BLAKE3 checksumming (recommended)
    pub enable_blake3: bool,

    /// Enable MD5 checksumming (legacy compatibility)
    pub enable_md5: bool,

    /// Enable SHA-256 checksumming
    pub enable_sha256: bool,

    /// Enable CRC32 checksumming (fast)
    pub enable_crc32: bool,

    /// Generate sidecar checksum files
    pub generate_sidecars: bool,

    /// Verify file structure and metadata
    pub validate_containers: bool,

    /// Enable fixity checking (scheduled verification)
    pub enable_fixity_checks: bool,

    /// Fixity check interval in days
    pub fixity_check_interval_days: u32,

    /// Quarantine corrupted files automatically
    pub auto_quarantine: bool,

    /// Number of parallel verification threads
    pub parallel_threads: usize,

    /// Database path for verification history
    pub database_path: PathBuf,

    /// Quarantine directory
    pub quarantine_dir: PathBuf,

    /// Generate PREMIS events
    pub enable_premis_logging: bool,

    /// Enable BagIt support
    pub enable_bagit: bool,
}

impl Default for VerificationConfig {
    fn default() -> Self {
        Self {
            enable_blake3: true,
            enable_md5: false,
            enable_sha256: true,
            enable_crc32: true,
            generate_sidecars: true,
            validate_containers: true,
            enable_fixity_checks: true,
            fixity_check_interval_days: 90,
            auto_quarantine: false,
            parallel_threads: num_cpus::get(),
            database_path: PathBuf::from("archive_verification.db"),
            quarantine_dir: PathBuf::from("quarantine"),
            enable_premis_logging: true,
            enable_bagit: false,
        }
    }
}

/// Archive verifier - main entry point
#[cfg(feature = "sqlite")]
pub struct ArchiveVerifier {
    config: VerificationConfig,
    db_pool: Option<sqlx::SqlitePool>,
}

#[cfg(feature = "sqlite")]
impl ArchiveVerifier {
    /// Create a new archive verifier with default configuration
    pub fn new() -> Self {
        Self {
            config: VerificationConfig::default(),
            db_pool: None,
        }
    }

    /// Create a new archive verifier with custom configuration
    pub fn with_config(config: VerificationConfig) -> Self {
        Self {
            config,
            db_pool: None,
        }
    }

    /// Initialize the verifier (connects to database)
    pub async fn initialize(&mut self) -> ArchiveResult<()> {
        let db_url = format!("sqlite:{}", self.config.database_path.display());
        let pool = sqlx::SqlitePool::connect(&db_url).await?;

        // Create tables
        self.create_tables(&pool).await?;

        self.db_pool = Some(pool);
        Ok(())
    }

    /// Create database tables
    async fn create_tables(&self, pool: &sqlx::SqlitePool) -> ArchiveResult<()> {
        sqlx::query(
            r"
            CREATE TABLE IF NOT EXISTS checksums (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                blake3 TEXT,
                md5 TEXT,
                sha256 TEXT,
                crc32 TEXT,
                created_at TEXT NOT NULL,
                last_verified_at TEXT
            )
            ",
        )
        .execute(pool)
        .await?;

        sqlx::query(
            r"
            CREATE TABLE IF NOT EXISTS fixity_checks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                check_time TEXT NOT NULL,
                status TEXT NOT NULL,
                error_message TEXT,
                blake3_match BOOLEAN,
                md5_match BOOLEAN,
                sha256_match BOOLEAN,
                crc32_match BOOLEAN
            )
            ",
        )
        .execute(pool)
        .await?;

        sqlx::query(
            r"
            CREATE TABLE IF NOT EXISTS premis_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL UNIQUE,
                event_type TEXT NOT NULL,
                event_date_time TEXT NOT NULL,
                event_detail TEXT,
                event_outcome TEXT NOT NULL,
                event_outcome_detail TEXT,
                linking_object_id TEXT NOT NULL
            )
            ",
        )
        .execute(pool)
        .await?;

        sqlx::query(
            r"
            CREATE TABLE IF NOT EXISTS quarantine_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_path TEXT NOT NULL,
                quarantine_path TEXT NOT NULL,
                quarantine_date TEXT NOT NULL,
                reason TEXT NOT NULL,
                checksum_before TEXT,
                auto_quarantine BOOLEAN NOT NULL,
                restored BOOLEAN DEFAULT 0,
                restore_date TEXT
            )
            ",
        )
        .execute(pool)
        .await?;

        Ok(())
    }

    /// Get the configuration
    pub fn config(&self) -> &VerificationConfig {
        &self.config
    }

    /// Get mutable reference to configuration
    pub fn config_mut(&mut self) -> &mut VerificationConfig {
        &mut self.config
    }

    /// Get database pool
    pub fn db_pool(&self) -> Option<&sqlx::SqlitePool> {
        self.db_pool.as_ref()
    }

    /// Verify a single file
    pub async fn verify_file(&self, path: &Path) -> ArchiveResult<VerificationResult> {
        let mut result = VerificationResult {
            file_path: path.to_path_buf(),
            verified_at: Utc::now(),
            status: VerificationStatus::Success,
            checksums: ChecksumSet::default(),
            validation_errors: Vec::new(),
            fixity_status: None,
        };

        // Compute checksums
        if self.config.enable_blake3
            || self.config.enable_md5
            || self.config.enable_sha256
            || self.config.enable_crc32
        {
            result.checksums = checksum::compute_checksums(path, &self.config).await?;
        }

        // Validate container if enabled
        if self.config.validate_containers {
            if let Err(e) = validate::validate_file(path).await {
                result
                    .validation_errors
                    .push(format!("Validation error: {e}"));
                result.status = VerificationStatus::ValidationFailed;
            }
        }

        // Check fixity if we have a database
        if self.config.enable_fixity_checks {
            if let Some(pool) = self.db_pool.as_ref() {
                let fixity_result = fixity::check_fixity(path, &result.checksums, pool).await?;
                result.fixity_status = Some(fixity_result);
            }
        }

        Ok(result)
    }

    /// Verify multiple files in parallel
    pub async fn verify_files(&self, paths: &[PathBuf]) -> ArchiveResult<Vec<VerificationResult>> {
        let mut results = Vec::new();

        for path in paths {
            let result = self.verify_file(path).await?;
            results.push(result);
        }

        Ok(results)
    }

    /// Run scheduled fixity checks
    pub async fn run_fixity_checks(&self) -> ArchiveResult<fixity::FixityReport> {
        if let Some(pool) = &self.db_pool {
            fixity::run_scheduled_checks(pool, &self.config).await
        } else {
            Err(ArchiveError::Config("Database not initialized".to_string()))
        }
    }

    /// Generate verification report
    pub async fn generate_report(
        &self,
        format: report::ReportFormat,
        output_path: &Path,
    ) -> ArchiveResult<()> {
        if let Some(pool) = &self.db_pool {
            report::generate_report(pool, format, output_path).await
        } else {
            Err(ArchiveError::Config("Database not initialized".to_string()))
        }
    }
}

#[cfg(feature = "sqlite")]
impl Default for ArchiveVerifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Checksum set for a file
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChecksumSet {
    pub blake3: Option<String>,
    pub md5: Option<String>,
    pub sha256: Option<String>,
    pub crc32: Option<String>,
}

/// Verification result for a single file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    pub file_path: PathBuf,
    pub verified_at: DateTime<Utc>,
    pub status: VerificationStatus,
    pub checksums: ChecksumSet,
    pub validation_errors: Vec<String>,
    #[cfg(feature = "sqlite")]
    pub fixity_status: Option<fixity::FixityStatus>,
}

/// Verification status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerificationStatus {
    Success,
    ChecksumMismatch,
    ValidationFailed,
    Corrupted,
    Quarantined,
}

// Conditional compilation for num_cpus
#[cfg(not(target_env = "msvc"))]
mod num_cpus {
    pub fn get() -> usize {
        std::thread::available_parallelism()
            .map(std::num::NonZero::get)
            .unwrap_or(4)
    }
}

#[cfg(target_env = "msvc")]
mod num_cpus {
    pub fn get() -> usize {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
    }
}
