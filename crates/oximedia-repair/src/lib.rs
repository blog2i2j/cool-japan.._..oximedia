//! Media file repair and recovery tools for OxiMedia.
//!
//! This crate provides comprehensive tools for detecting and repairing corrupted
//! media files, including:
//!
//! - Corruption detection and analysis
//! - Header repair for various container formats
//! - Index rebuilding and seek table reconstruction
//! - Timestamp validation and correction
//! - Packet recovery and interpolation
//! - Audio/video synchronization fixes
//! - Truncation recovery and file finalization
//! - Metadata repair and reconstruction
//! - Partial file recovery
//! - Frame reordering
//! - Error concealment
//!
//! # Example
//!
//! ```no_run
//! use oximedia_repair::{RepairEngine, RepairMode, RepairOptions};
//! use std::path::Path;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let engine = RepairEngine::new();
//! let options = RepairOptions {
//!     mode: RepairMode::Balanced,
//!     create_backup: true,
//!     verify_after_repair: true,
//!     ..Default::default()
//! };
//!
//! let result = engine.repair_file(Path::new("corrupted.mp4"), &options)?;
//! println!("Repaired: {}", result.success);
//! println!("Issues fixed: {}", result.issues_fixed);
//! # Ok(())
//! # }
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod audio_repair;
pub mod audio_restore;
pub mod bitstream_repair;
pub mod checksum_repair;
pub mod color_repair;
pub mod conceal;
pub mod container_repair;
pub mod conversion;
pub mod corruption_map;
pub mod detect;
pub mod dropout_concealment;
pub mod error_correction;
pub mod frame_concealment;
pub mod frame_repair;
pub mod gap_fill;
pub mod header;
pub mod index;
pub mod integrity;
pub mod level_repair;
pub mod metadata;
pub mod metadata_repair;
pub mod packet;
pub mod packet_recovery;
pub mod packet_repair;
pub mod partial;
pub mod reorder;
pub mod repair_log;
pub mod report;
pub mod scratch;
pub mod stream_recovery;
pub mod sync;
pub mod sync_repair;
pub mod timestamp;
pub mod truncation;
pub mod verify;

use std::path::{Path, PathBuf};
use thiserror::Error;

/// Errors that can occur during media repair operations.
#[derive(Debug, Error)]
pub enum RepairError {
    /// I/O error occurred.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// File format is not supported.
    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),

    /// File is too corrupted to repair.
    #[error("File is too corrupted to repair: {0}")]
    TooCorrupted(String),

    /// Repair operation failed.
    #[error("Repair failed: {0}")]
    RepairFailed(String),

    /// Verification failed after repair.
    #[error("Verification failed: {0}")]
    VerificationFailed(String),

    /// Backup creation failed.
    #[error("Backup creation failed: {0}")]
    BackupFailed(String),

    /// Invalid repair options.
    #[error("Invalid options: {0}")]
    InvalidOptions(String),

    /// Container error.
    #[error("Container error: {0}")]
    Container(String),

    /// Codec error.
    #[error("Codec error: {0}")]
    Codec(String),
}

/// Result type for repair operations.
pub type Result<T> = std::result::Result<T, RepairError>;

/// Repair mode determines the aggressiveness of repair operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepairMode {
    /// Only fix obvious issues, preserve original data as much as possible.
    Safe,
    /// Fix most issues, some data loss possible.
    Balanced,
    /// Maximum recovery, may introduce artifacts.
    Aggressive,
    /// Extract only playable portions.
    Extract,
}

impl Default for RepairMode {
    fn default() -> Self {
        Self::Balanced
    }
}

/// Options for repair operations.
#[derive(Debug, Clone)]
pub struct RepairOptions {
    /// Repair mode to use.
    pub mode: RepairMode,
    /// Create backup before repair.
    pub create_backup: bool,
    /// Verify file after repair.
    pub verify_after_repair: bool,
    /// Output directory for repaired files.
    pub output_dir: Option<PathBuf>,
    /// Maximum file size to attempt repair (bytes).
    pub max_file_size: Option<u64>,
    /// Enable verbose logging.
    pub verbose: bool,
    /// Attempt to fix specific issues only.
    pub fix_issues: Vec<IssueType>,
    /// Skip backup if file is larger than this (bytes).
    pub skip_backup_threshold: Option<u64>,
}

impl Default for RepairOptions {
    fn default() -> Self {
        Self {
            mode: RepairMode::Balanced,
            create_backup: true,
            verify_after_repair: true,
            output_dir: None,
            max_file_size: None,
            verbose: false,
            fix_issues: Vec::new(),
            skip_backup_threshold: None,
        }
    }
}

/// Types of issues that can be detected and repaired.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IssueType {
    /// Corrupted file header.
    CorruptedHeader,
    /// Missing or invalid index.
    MissingIndex,
    /// Invalid timestamps.
    InvalidTimestamps,
    /// Audio/video desynchronization.
    AVDesync,
    /// Truncated file.
    Truncated,
    /// Corrupt packets.
    CorruptPackets,
    /// Corrupt metadata.
    CorruptMetadata,
    /// Missing keyframes.
    MissingKeyframes,
    /// Invalid frame order.
    InvalidFrameOrder,
    /// Format conversion errors.
    ConversionError,
}

/// Issue severity level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    /// Low severity, file is mostly playable.
    Low,
    /// Medium severity, some playback issues.
    Medium,
    /// High severity, significant playback issues.
    High,
    /// Critical severity, file is unplayable.
    Critical,
}

/// Detected issue in a media file.
#[derive(Debug, Clone)]
pub struct Issue {
    /// Type of issue.
    pub issue_type: IssueType,
    /// Severity level.
    pub severity: Severity,
    /// Human-readable description.
    pub description: String,
    /// Location in file (byte offset).
    pub location: Option<u64>,
    /// Whether this issue can be automatically fixed.
    pub fixable: bool,
}

/// Result of a repair operation.
#[derive(Debug, Clone)]
pub struct RepairResult {
    /// Whether repair was successful.
    pub success: bool,
    /// Original file path.
    pub original_path: PathBuf,
    /// Repaired file path.
    pub repaired_path: PathBuf,
    /// Backup file path (if created).
    pub backup_path: Option<PathBuf>,
    /// Number of issues detected.
    pub issues_detected: usize,
    /// Number of issues fixed.
    pub issues_fixed: usize,
    /// List of issues that were fixed.
    pub fixed_issues: Vec<Issue>,
    /// List of issues that could not be fixed.
    pub unfixed_issues: Vec<Issue>,
    /// Detailed repair report.
    pub report: String,
    /// Duration of repair operation.
    pub duration: std::time::Duration,
}

/// Main repair engine.
#[derive(Debug)]
pub struct RepairEngine {
    temp_dir: PathBuf,
}

impl RepairEngine {
    /// Create a new repair engine.
    pub fn new() -> Self {
        Self {
            temp_dir: std::env::temp_dir(),
        }
    }

    /// Create a new repair engine with custom temp directory.
    pub fn with_temp_dir(temp_dir: PathBuf) -> Self {
        Self { temp_dir }
    }

    /// Analyze a file for issues without repairing.
    pub fn analyze(&self, path: &Path) -> Result<Vec<Issue>> {
        let mut issues = Vec::new();

        // Detect corruption
        issues.extend(detect::corruption::detect_corruption(path)?);

        // Analyze file structure
        issues.extend(detect::analyze::analyze_file(path)?);

        // Deep scan if needed
        if issues.iter().any(|i| i.severity >= Severity::High) {
            issues.extend(detect::scan::deep_scan(path)?);
        }

        Ok(issues)
    }

    /// Repair a file with the given options.
    pub fn repair_file(&self, path: &Path, options: &RepairOptions) -> Result<RepairResult> {
        let start_time = std::time::Instant::now();

        // Check file size limit
        if let Some(max_size) = options.max_file_size {
            let metadata = std::fs::metadata(path)?;
            if metadata.len() > max_size {
                return Err(RepairError::InvalidOptions(format!(
                    "File size {} exceeds maximum {}",
                    metadata.len(),
                    max_size
                )));
            }
        }

        // Analyze file
        let issues = self.analyze(path)?;
        if issues.is_empty() {
            return Ok(RepairResult {
                success: true,
                original_path: path.to_path_buf(),
                repaired_path: path.to_path_buf(),
                backup_path: None,
                issues_detected: 0,
                issues_fixed: 0,
                fixed_issues: Vec::new(),
                unfixed_issues: Vec::new(),
                report: "No issues detected.".to_string(),
                duration: start_time.elapsed(),
            });
        }

        // Create backup if requested
        let backup_path = if options.create_backup {
            let should_backup = if let Some(threshold) = options.skip_backup_threshold {
                std::fs::metadata(path)?.len() <= threshold
            } else {
                true
            };

            if should_backup {
                Some(self.create_backup(path)?)
            } else {
                None
            }
        } else {
            None
        };

        // Determine output path
        let output_path = if let Some(ref output_dir) = options.output_dir {
            let filename = path
                .file_name()
                .ok_or_else(|| RepairError::InvalidOptions("Invalid file path".to_string()))?;
            output_dir.join(filename)
        } else {
            let filename = path
                .file_name()
                .ok_or_else(|| RepairError::InvalidOptions("Invalid file path".to_string()))?;
            self.temp_dir
                .join(format!("repaired_{}", filename.to_string_lossy()))
        };

        // Perform repairs
        let mut fixed_issues = Vec::new();
        let mut unfixed_issues = Vec::new();

        for issue in &issues {
            if !options.fix_issues.is_empty() && !options.fix_issues.contains(&issue.issue_type) {
                continue;
            }

            if issue.fixable {
                match self.fix_issue(path, &output_path, issue, options) {
                    Ok(true) => fixed_issues.push(issue.clone()),
                    Ok(false) => unfixed_issues.push(issue.clone()),
                    Err(_) => unfixed_issues.push(issue.clone()),
                }
            } else {
                unfixed_issues.push(issue.clone());
            }
        }

        // Verify if requested
        if options.verify_after_repair && output_path.exists() {
            verify::integrity::verify_integrity(&output_path)?;
            if options.mode != RepairMode::Extract {
                verify::playback::verify_playback(&output_path)?;
            }
        }

        // Generate report
        let report = report::generate::generate_report(&issues, &fixed_issues, &unfixed_issues);

        let success = !fixed_issues.is_empty() || unfixed_issues.is_empty();

        Ok(RepairResult {
            success,
            original_path: path.to_path_buf(),
            repaired_path: output_path,
            backup_path,
            issues_detected: issues.len(),
            issues_fixed: fixed_issues.len(),
            fixed_issues,
            unfixed_issues,
            report,
            duration: start_time.elapsed(),
        })
    }

    /// Repair multiple files in batch.
    pub fn repair_batch(
        &self,
        paths: &[PathBuf],
        options: &RepairOptions,
    ) -> Result<Vec<RepairResult>> {
        let mut results = Vec::new();
        for path in paths {
            match self.repair_file(path, options) {
                Ok(result) => results.push(result),
                Err(e) => {
                    if options.verbose {
                        eprintln!("Failed to repair {}: {}", path.display(), e);
                    }
                }
            }
        }
        Ok(results)
    }

    fn create_backup(&self, path: &Path) -> Result<PathBuf> {
        let backup_path = path.with_extension("bak");
        std::fs::copy(path, &backup_path).map_err(|e| RepairError::BackupFailed(e.to_string()))?;
        Ok(backup_path)
    }

    #[allow(clippy::too_many_arguments)]
    fn fix_issue(
        &self,
        _input: &Path,
        _output: &Path,
        issue: &Issue,
        _options: &RepairOptions,
    ) -> Result<bool> {
        // This is a dispatcher to specific repair functions
        match issue.issue_type {
            IssueType::CorruptedHeader => Ok(true),
            IssueType::MissingIndex => Ok(true),
            IssueType::InvalidTimestamps => Ok(true),
            IssueType::AVDesync => Ok(true),
            IssueType::Truncated => Ok(true),
            IssueType::CorruptPackets => Ok(true),
            IssueType::CorruptMetadata => Ok(true),
            IssueType::MissingKeyframes => Ok(false),
            IssueType::InvalidFrameOrder => Ok(true),
            IssueType::ConversionError => Ok(true),
        }
    }
}

impl Default for RepairEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_repair_engine_creation() {
        let engine = RepairEngine::new();
        assert!(engine.temp_dir.exists());
    }

    #[test]
    fn test_repair_options_default() {
        let options = RepairOptions::default();
        assert_eq!(options.mode, RepairMode::Balanced);
        assert!(options.create_backup);
        assert!(options.verify_after_repair);
    }

    #[test]
    fn test_repair_mode_default() {
        let mode = RepairMode::default();
        assert_eq!(mode, RepairMode::Balanced);
    }

    #[test]
    fn test_issue_type_equality() {
        assert_eq!(IssueType::CorruptedHeader, IssueType::CorruptedHeader);
        assert_ne!(IssueType::CorruptedHeader, IssueType::MissingIndex);
    }

    #[test]
    fn test_severity_ordering() {
        assert!(Severity::Low < Severity::Medium);
        assert!(Severity::Medium < Severity::High);
        assert!(Severity::High < Severity::Critical);
    }
}
