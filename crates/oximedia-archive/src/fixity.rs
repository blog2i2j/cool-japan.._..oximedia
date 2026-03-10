//! Fixity checking - data integrity verification over time
//!
//! This module provides long-term preservation features including:
//! - Scheduled verification
//! - Bit rot detection
//! - Corruption detection
//! - PREMIS event logging
//! - BagIt support

use crate::{ArchiveError, ArchiveResult, ChecksumSet, VerificationConfig};
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use sqlx::Row;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokio::fs;
use tracing::{debug, error, info, warn};

/// Fixity check status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixityStatus {
    pub file_path: PathBuf,
    pub last_check: DateTime<Utc>,
    pub status: FixityCheckResult,
    pub checksums_match: HashMap<String, bool>,
    pub days_since_last_check: i64,
    pub total_checks: u32,
    pub failed_checks: u32,
}

/// Fixity check result
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FixityCheckResult {
    Pass,
    Fail,
    NoBaseline,
    FileNotFound,
}

/// Check fixity for a file
pub async fn check_fixity(
    path: &Path,
    current_checksums: &ChecksumSet,
    pool: &sqlx::SqlitePool,
) -> ArchiveResult<FixityStatus> {
    let path_str = path.to_string_lossy().to_string();

    // Load baseline checksums from database
    let baseline = crate::checksum::ChecksumRecord::load(pool, &path_str).await?;

    if let Some(baseline) = baseline {
        let mut checksums_match = HashMap::new();
        let mut all_match = true;

        // Compare BLAKE3
        if let (Some(ref current), Some(ref stored)) = (&current_checksums.blake3, &baseline.blake3)
        {
            let matches = current == stored;
            checksums_match.insert("blake3".to_string(), matches);
            all_match = all_match && matches;
        }

        // Compare MD5
        if let (Some(ref current), Some(ref stored)) = (&current_checksums.md5, &baseline.md5) {
            let matches = current == stored;
            checksums_match.insert("md5".to_string(), matches);
            all_match = all_match && matches;
        }

        // Compare SHA-256
        if let (Some(ref current), Some(ref stored)) = (&current_checksums.sha256, &baseline.sha256)
        {
            let matches = current == stored;
            checksums_match.insert("sha256".to_string(), matches);
            all_match = all_match && matches;
        }

        // Compare CRC32
        if let (Some(ref current), Some(ref stored)) = (&current_checksums.crc32, &baseline.crc32) {
            let matches = current == stored;
            checksums_match.insert("crc32".to_string(), matches);
            all_match = all_match && matches;
        }

        let days_since_last = if let Some(last_verified) = baseline.last_verified_at {
            (Utc::now() - last_verified).num_days()
        } else {
            (Utc::now() - baseline.created_at).num_days()
        };

        // Get check statistics
        let (total_checks, failed_checks) = get_check_statistics(pool, &path_str).await?;

        // Record this check
        record_fixity_check(pool, &path_str, all_match, &checksums_match).await?;

        let status = if all_match {
            FixityCheckResult::Pass
        } else {
            FixityCheckResult::Fail
        };

        Ok(FixityStatus {
            file_path: path.to_path_buf(),
            last_check: Utc::now(),
            status,
            checksums_match,
            days_since_last_check: days_since_last,
            total_checks: total_checks + 1,
            failed_checks: if all_match {
                failed_checks
            } else {
                failed_checks + 1
            },
        })
    } else {
        // No baseline - create one
        let file_size = fs::metadata(path).await?.len() as i64;
        let record =
            crate::checksum::ChecksumRecord::new(path, file_size, current_checksums.clone());
        record.save(pool).await?;

        info!("Created baseline checksums for {}", path.display());

        Ok(FixityStatus {
            file_path: path.to_path_buf(),
            last_check: Utc::now(),
            status: FixityCheckResult::NoBaseline,
            checksums_match: HashMap::new(),
            days_since_last_check: 0,
            total_checks: 0,
            failed_checks: 0,
        })
    }
}

/// Get check statistics for a file
async fn get_check_statistics(
    pool: &sqlx::SqlitePool,
    file_path: &str,
) -> ArchiveResult<(u32, u32)> {
    let row = sqlx::query(
        r"
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN status = 'fail' THEN 1 ELSE 0 END) as failed
        FROM fixity_checks
        WHERE file_path = ?
        ",
    )
    .bind(file_path)
    .fetch_one(pool)
    .await?;

    let total: i64 = row.get("total");
    let failed: i64 = row.get("failed");

    Ok((total as u32, failed as u32))
}

/// Record a fixity check in the database
async fn record_fixity_check(
    pool: &sqlx::SqlitePool,
    file_path: &str,
    passed: bool,
    checksums_match: &HashMap<String, bool>,
) -> ArchiveResult<()> {
    let check_time = Utc::now().to_rfc3339();
    let status = if passed { "pass" } else { "fail" };

    let blake3_match = checksums_match.get("blake3").copied();
    let md5_match = checksums_match.get("md5").copied();
    let sha256_match = checksums_match.get("sha256").copied();
    let crc32_match = checksums_match.get("crc32").copied();

    let error_message = if passed {
        None
    } else {
        Some(format!("Checksum mismatch detected: {checksums_match:?}"))
    };

    sqlx::query(
        r"
        INSERT INTO fixity_checks (file_path, check_time, status, error_message, blake3_match, md5_match, sha256_match, crc32_match)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ",
    )
    .bind(file_path)
    .bind(&check_time)
    .bind(status)
    .bind(&error_message)
    .bind(blake3_match)
    .bind(md5_match)
    .bind(sha256_match)
    .bind(crc32_match)
    .execute(pool)
    .await?;

    Ok(())
}

/// Run scheduled fixity checks
pub async fn run_scheduled_checks(
    pool: &sqlx::SqlitePool,
    config: &VerificationConfig,
) -> ArchiveResult<FixityReport> {
    info!(
        "Running scheduled fixity checks (interval: {} days)",
        config.fixity_check_interval_days
    );

    let cutoff_date = Utc::now() - Duration::days(config.fixity_check_interval_days as i64);
    let cutoff_str = cutoff_date.to_rfc3339();

    // Find files that need checking
    let rows = sqlx::query(
        r"
        SELECT file_path, last_verified_at
        FROM checksums
        WHERE last_verified_at IS NULL OR last_verified_at < ?
        ORDER BY last_verified_at ASC NULLS FIRST
        ",
    )
    .bind(&cutoff_str)
    .fetch_all(pool)
    .await?;

    let mut report = FixityReport {
        check_time: Utc::now(),
        total_files: rows.len(),
        passed: 0,
        failed: 0,
        no_baseline: 0,
        not_found: 0,
        errors: Vec::new(),
        failed_files: Vec::new(),
    };

    for row in rows {
        let file_path: String = row.get("file_path");
        let path = PathBuf::from(&file_path);

        if !path.exists() {
            report.not_found += 1;
            report
                .errors
                .push((path.clone(), "File not found".to_string()));
            continue;
        }

        match check_file_fixity(&path, pool, config).await {
            Ok(status) => match status.status {
                FixityCheckResult::Pass => report.passed += 1,
                FixityCheckResult::Fail => {
                    report.failed += 1;
                    report.failed_files.push(path.clone());
                    warn!("Fixity check failed for {}", path.display());
                }
                FixityCheckResult::NoBaseline => report.no_baseline += 1,
                FixityCheckResult::FileNotFound => {
                    report.not_found += 1;
                    report
                        .errors
                        .push((path.clone(), "File not found".to_string()));
                }
            },
            Err(e) => {
                report.errors.push((path.clone(), e.to_string()));
                error!("Error checking fixity for {}: {}", path.display(), e);
            }
        }
    }

    info!(
        "Fixity check complete: {} passed, {} failed, {} no baseline, {} not found",
        report.passed, report.failed, report.no_baseline, report.not_found
    );

    Ok(report)
}

/// Check fixity for a single file (complete workflow)
async fn check_file_fixity(
    path: &Path,
    pool: &sqlx::SqlitePool,
    config: &VerificationConfig,
) -> ArchiveResult<FixityStatus> {
    if !path.exists() {
        return Ok(FixityStatus {
            file_path: path.to_path_buf(),
            last_check: Utc::now(),
            status: FixityCheckResult::FileNotFound,
            checksums_match: HashMap::new(),
            days_since_last_check: 0,
            total_checks: 0,
            failed_checks: 0,
        });
    }

    // Compute current checksums
    let checksums = crate::checksum::compute_checksums(path, config).await?;

    // Check fixity
    let mut status = check_fixity(path, &checksums, pool).await?;

    // Update last verified timestamp
    let path_str = path.to_string_lossy().to_string();
    if let Some(mut record) = crate::checksum::ChecksumRecord::load(pool, &path_str).await? {
        record.update_verified(pool).await?;
    }

    // Log PREMIS event if enabled
    if config.enable_premis_logging {
        log_premis_event(
            pool,
            &path_str,
            "fixity check",
            if status.status == FixityCheckResult::Pass {
                "success"
            } else {
                "failure"
            },
        )
        .await?;
    }

    // Quarantine if auto-quarantine is enabled and check failed
    if config.auto_quarantine && status.status == FixityCheckResult::Fail {
        if let Err(e) =
            crate::quarantine::quarantine_file(path, pool, config, "Fixity check failed").await
        {
            error!("Failed to quarantine {}: {}", path.display(), e);
        } else {
            status.status = FixityCheckResult::Fail; // Keep as fail but note it's quarantined
        }
    }

    Ok(status)
}

/// Fixity report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixityReport {
    pub check_time: DateTime<Utc>,
    pub total_files: usize,
    pub passed: usize,
    pub failed: usize,
    pub no_baseline: usize,
    pub not_found: usize,
    pub errors: Vec<(PathBuf, String)>,
    pub failed_files: Vec<PathBuf>,
}

impl FixityReport {
    /// Check if all checks passed
    pub fn all_passed(&self) -> bool {
        self.failed == 0 && self.errors.is_empty()
    }

    /// Get success rate
    pub fn success_rate(&self) -> f64 {
        if self.total_files == 0 {
            return 0.0;
        }
        (self.passed as f64) / (self.total_files as f64)
    }
}

/// PREMIS event types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PremisEvent {
    pub event_id: String,
    pub event_type: String,
    pub event_date_time: DateTime<Utc>,
    pub event_detail: Option<String>,
    pub event_outcome: String,
    pub event_outcome_detail: Option<String>,
    pub linking_object_id: String,
}

impl PremisEvent {
    /// Create a new PREMIS event
    pub fn new(event_type: &str, linking_object_id: &str, outcome: &str) -> Self {
        Self {
            event_id: format!("premis-{}-{}", Utc::now().timestamp(), uuid_simple()),
            event_type: event_type.to_string(),
            event_date_time: Utc::now(),
            event_detail: None,
            event_outcome: outcome.to_string(),
            event_outcome_detail: None,
            linking_object_id: linking_object_id.to_string(),
        }
    }

    /// Save to database
    pub async fn save(&self, pool: &sqlx::SqlitePool) -> ArchiveResult<()> {
        let event_date_time_str = self.event_date_time.to_rfc3339();

        sqlx::query(
            r"
            INSERT INTO premis_events (event_id, event_type, event_date_time, event_detail, event_outcome, event_outcome_detail, linking_object_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ",
        )
        .bind(&self.event_id)
        .bind(&self.event_type)
        .bind(&event_date_time_str)
        .bind(&self.event_detail)
        .bind(&self.event_outcome)
        .bind(&self.event_outcome_detail)
        .bind(&self.linking_object_id)
        .execute(pool)
        .await?;

        Ok(())
    }

    /// Load events for a file
    pub async fn load_for_file(
        pool: &sqlx::SqlitePool,
        linking_object_id: &str,
    ) -> ArchiveResult<Vec<Self>> {
        let rows = sqlx::query(
            r"
            SELECT event_id, event_type, event_date_time, event_detail, event_outcome, event_outcome_detail, linking_object_id
            FROM premis_events
            WHERE linking_object_id = ?
            ORDER BY event_date_time DESC
            ",
        )
        .bind(linking_object_id)
        .fetch_all(pool)
        .await?;

        let mut events = Vec::new();
        for row in rows {
            let event_date_time_str: String = row.get("event_date_time");
            let event_date_time = DateTime::parse_from_rfc3339(&event_date_time_str)
                .map_err(|e| ArchiveError::Database(sqlx::Error::Decode(Box::new(e))))?
                .with_timezone(&Utc);

            events.push(Self {
                event_id: row.get("event_id"),
                event_type: row.get("event_type"),
                event_date_time,
                event_detail: row.get("event_detail"),
                event_outcome: row.get("event_outcome"),
                event_outcome_detail: row.get("event_outcome_detail"),
                linking_object_id: row.get("linking_object_id"),
            });
        }

        Ok(events)
    }
}

/// Log a PREMIS event
pub async fn log_premis_event(
    pool: &sqlx::SqlitePool,
    linking_object_id: &str,
    event_type: &str,
    outcome: &str,
) -> ArchiveResult<()> {
    let event = PremisEvent::new(event_type, linking_object_id, outcome);
    event.save(pool).await?;
    debug!(
        "Logged PREMIS event: {} for {}",
        event_type, linking_object_id
    );
    Ok(())
}

/// BagIt manifest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BagItManifest {
    pub bag_dir: PathBuf,
    pub checksums: HashMap<String, String>,
    pub algorithm: String,
}

impl BagItManifest {
    /// Create a new BagIt manifest
    pub fn new(bag_dir: PathBuf, algorithm: &str) -> Self {
        Self {
            bag_dir,
            checksums: HashMap::new(),
            algorithm: algorithm.to_string(),
        }
    }

    /// Add a file to the manifest
    pub fn add_file(&mut self, relative_path: &str, checksum: &str) {
        self.checksums
            .insert(relative_path.to_string(), checksum.to_string());
    }

    /// Write manifest to file
    pub async fn write(&self) -> ArchiveResult<()> {
        let manifest_name = format!("manifest-{}.txt", self.algorithm);
        let manifest_path = self.bag_dir.join(&manifest_name);

        let mut lines = Vec::new();
        for (path, checksum) in &self.checksums {
            lines.push(format!("{checksum}  {path}"));
        }
        lines.sort();

        let content = lines.join("\n") + "\n";
        fs::write(&manifest_path, content).await?;

        info!("Wrote BagIt manifest: {}", manifest_path.display());
        Ok(())
    }

    /// Read manifest from file
    pub async fn read(bag_dir: &Path, algorithm: &str) -> ArchiveResult<Self> {
        let manifest_name = format!("manifest-{algorithm}.txt");
        let manifest_path = bag_dir.join(&manifest_name);

        let content = fs::read_to_string(&manifest_path).await?;
        let mut manifest = Self::new(bag_dir.to_path_buf(), algorithm);

        for line in content.lines() {
            let parts: Vec<&str> = line.splitn(2, "  ").collect();
            if parts.len() == 2 {
                manifest.add_file(parts[1], parts[0]);
            }
        }

        Ok(manifest)
    }

    /// Verify the bag
    pub async fn verify(
        &self,
        _config: &VerificationConfig,
    ) -> ArchiveResult<BagVerificationResult> {
        let mut result = BagVerificationResult {
            bag_dir: self.bag_dir.clone(),
            total_files: self.checksums.len(),
            verified_files: 0,
            failed_files: 0,
            missing_files: Vec::new(),
            mismatched_files: Vec::new(),
        };

        for (relative_path, expected_checksum) in &self.checksums {
            let file_path = self.bag_dir.join(relative_path);

            if !file_path.exists() {
                result.missing_files.push(file_path.clone());
                result.failed_files += 1;
                continue;
            }

            let computed_checksum = match self.algorithm.as_str() {
                "md5" => crate::checksum::compute_md5(&file_path).await?,
                "sha256" => crate::checksum::compute_sha256(&file_path).await?,
                "blake3" => crate::checksum::compute_blake3(&file_path).await?,
                _ => {
                    return Err(ArchiveError::Validation(format!(
                        "Unsupported algorithm: {}",
                        self.algorithm
                    )))
                }
            };

            if &computed_checksum == expected_checksum {
                result.verified_files += 1;
            } else {
                result.mismatched_files.push((
                    file_path,
                    expected_checksum.clone(),
                    computed_checksum,
                ));
                result.failed_files += 1;
            }
        }

        Ok(result)
    }
}

/// BagIt verification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BagVerificationResult {
    pub bag_dir: PathBuf,
    pub total_files: usize,
    pub verified_files: usize,
    pub failed_files: usize,
    pub missing_files: Vec<PathBuf>,
    pub mismatched_files: Vec<(PathBuf, String, String)>,
}

impl BagVerificationResult {
    /// Check if bag is valid
    pub fn is_valid(&self) -> bool {
        self.failed_files == 0
    }

    /// Get success rate
    pub fn success_rate(&self) -> f64 {
        if self.total_files == 0 {
            return 0.0;
        }
        (self.verified_files as f64) / (self.total_files as f64)
    }
}

/// Create a BagIt bag from a directory
pub async fn create_bagit_bag(
    source_dir: &Path,
    bag_dir: &Path,
    algorithm: &str,
    _config: &VerificationConfig,
) -> ArchiveResult<BagItManifest> {
    info!(
        "Creating BagIt bag from {} to {}",
        source_dir.display(),
        bag_dir.display()
    );

    // Create bag directory structure
    let data_dir = bag_dir.join("data");
    fs::create_dir_all(&data_dir).await?;

    // Copy files to data directory
    copy_dir_recursive(source_dir, &data_dir).await?;

    // Create manifest
    let mut manifest = BagItManifest::new(bag_dir.to_path_buf(), algorithm);

    // Compute checksums for all files in data directory
    let files = collect_files(&data_dir).await?;

    for file in files {
        let relative_path = file
            .strip_prefix(bag_dir)
            .map_err(|e| ArchiveError::Validation(format!("Path error: {e}")))?
            .to_string_lossy()
            .to_string();

        let checksum = match algorithm {
            "md5" => crate::checksum::compute_md5(&file).await?,
            "sha256" => crate::checksum::compute_sha256(&file).await?,
            "blake3" => crate::checksum::compute_blake3(&file).await?,
            _ => {
                return Err(ArchiveError::Validation(format!(
                    "Unsupported algorithm: {algorithm}"
                )))
            }
        };

        manifest.add_file(&relative_path, &checksum);
    }

    // Write manifest
    manifest.write().await?;

    // Write bagit.txt
    let bagit_txt = bag_dir.join("bagit.txt");
    fs::write(
        &bagit_txt,
        "BagIt-Version: 1.0\nTag-File-Character-Encoding: UTF-8\n",
    )
    .await?;

    info!("Created BagIt bag at {}", bag_dir.display());

    Ok(manifest)
}

/// Copy directory recursively
fn copy_dir_recursive<'a>(
    src: &'a Path,
    dst: &'a Path,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = ArchiveResult<()>> + 'a>> {
    Box::pin(async move {
        fs::create_dir_all(dst).await?;

        let mut entries = fs::read_dir(src).await?;
        while let Some(entry) = entries.next_entry().await? {
            let file_type = entry.file_type().await?;
            let src_path = entry.path();
            let dst_path = dst.join(entry.file_name());

            if file_type.is_dir() {
                copy_dir_recursive(&src_path, &dst_path).await?;
            } else {
                fs::copy(&src_path, &dst_path).await?;
            }
        }

        Ok(())
    })
}

/// Collect all files in a directory recursively
fn collect_files(
    dir: &Path,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = ArchiveResult<Vec<PathBuf>>> + '_>> {
    Box::pin(async move {
        let mut files = Vec::new();
        let mut entries = fs::read_dir(dir).await?;

        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            let file_type = entry.file_type().await?;

            if file_type.is_dir() {
                files.extend(collect_files(&path).await?);
            } else if file_type.is_file() {
                files.push(path);
            }
        }

        Ok(files)
    })
}

/// Simple UUID generator (for PREMIS event IDs)
fn uuid_simple() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{now:x}")
}

/// Detect bit rot by comparing file sizes and modification times
pub async fn detect_bit_rot(path: &Path, pool: &sqlx::SqlitePool) -> ArchiveResult<BitRotStatus> {
    let path_str = path.to_string_lossy().to_string();

    // Get stored record
    let record = crate::checksum::ChecksumRecord::load(pool, &path_str).await?;

    if let Some(record) = record {
        let current_metadata = fs::metadata(path).await?;
        let current_size = current_metadata.len() as i64;

        if current_size != record.file_size {
            warn!(
                "Potential bit rot detected: file size changed for {}",
                path.display()
            );
            return Ok(BitRotStatus {
                detected: true,
                reason: format!(
                    "File size changed from {} to {} bytes",
                    record.file_size, current_size
                ),
            });
        }

        // Check modification time
        if let Ok(modified) = current_metadata.modified() {
            let modified_chrono = DateTime::<Utc>::from(modified);
            if modified_chrono > record.created_at {
                // File was modified after baseline was created
                return Ok(BitRotStatus {
                    detected: true,
                    reason: format!(
                        "File was modified after baseline (baseline: {}, modified: {})",
                        record.created_at, modified_chrono
                    ),
                });
            }
        }

        Ok(BitRotStatus {
            detected: false,
            reason: String::new(),
        })
    } else {
        Ok(BitRotStatus {
            detected: false,
            reason: "No baseline available".to_string(),
        })
    }
}

/// Bit rot detection status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BitRotStatus {
    pub detected: bool,
    pub reason: String,
}

/// Repair suggestions for corrupted files
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepairSuggestion {
    pub file_path: PathBuf,
    pub corruption_type: CorruptionType,
    pub suggestions: Vec<String>,
    pub can_auto_repair: bool,
}

/// Types of corruption
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CorruptionType {
    ChecksumMismatch,
    BitRot,
    StructuralDamage,
    MetadataCorruption,
    Unknown,
}

/// Generate repair suggestions for a corrupted file
#[allow(dead_code)]
pub async fn generate_repair_suggestions(
    path: &Path,
    corruption_type: CorruptionType,
) -> ArchiveResult<RepairSuggestion> {
    let mut suggestions = Vec::new();

    let can_auto_repair = match corruption_type {
        CorruptionType::ChecksumMismatch => {
            suggestions.push("Restore from backup if available".to_string());
            suggestions.push("Check for recent file system errors".to_string());
            suggestions.push("Verify storage media health".to_string());
            false
        }
        CorruptionType::BitRot => {
            suggestions.push("Restore from backup immediately".to_string());
            suggestions.push("Check storage media for hardware issues".to_string());
            suggestions.push("Consider migrating to new storage".to_string());
            false
        }
        CorruptionType::StructuralDamage => {
            suggestions.push(
                "Attempt repair with media-specific tools (e.g., ffmpeg, MP4Box)".to_string(),
            );
            suggestions.push("Restore from backup if repair fails".to_string());
            false
        }
        CorruptionType::MetadataCorruption => {
            suggestions.push("Rebuild metadata from content analysis".to_string());
            suggestions.push("Restore metadata from backup".to_string());
            true
        }
        CorruptionType::Unknown => {
            suggestions.push("Perform detailed file analysis".to_string());
            suggestions.push("Consult with digital preservation experts".to_string());
            false
        }
    };

    Ok(RepairSuggestion {
        file_path: path.to_path_buf(),
        corruption_type,
        suggestions,
        can_auto_repair,
    })
}
