//! File quarantine and corruption handling
//!
//! This module provides:
//! - Suspicious file quarantine
//! - Corruption isolation
//! - Repair workflows
//! - Backup restoration
//! - Notification system

use crate::{ArchiveError, ArchiveResult, VerificationConfig};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::Row;
use std::path::{Path, PathBuf};
use tokio::fs;
use tracing::{error, info, warn};

/// Quarantine record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuarantineRecord {
    pub id: Option<i64>,
    pub original_path: PathBuf,
    pub quarantine_path: PathBuf,
    pub quarantine_date: DateTime<Utc>,
    pub reason: String,
    pub checksum_before: Option<String>,
    pub auto_quarantine: bool,
    pub restored: bool,
    pub restore_date: Option<DateTime<Utc>>,
}

impl QuarantineRecord {
    /// Create a new quarantine record
    pub fn new(
        original_path: PathBuf,
        quarantine_path: PathBuf,
        reason: String,
        checksum_before: Option<String>,
        auto_quarantine: bool,
    ) -> Self {
        Self {
            id: None,
            original_path,
            quarantine_path,
            quarantine_date: Utc::now(),
            reason,
            checksum_before,
            auto_quarantine,
            restored: false,
            restore_date: None,
        }
    }

    /// Save to database
    pub async fn save(&self, pool: &sqlx::SqlitePool) -> ArchiveResult<i64> {
        let quarantine_date_str = self.quarantine_date.to_rfc3339();
        let restore_date_str = self.restore_date.map(|dt| dt.to_rfc3339());

        let result = sqlx::query(
            r"
            INSERT INTO quarantine_records (original_path, quarantine_path, quarantine_date, reason, checksum_before, auto_quarantine, restored, restore_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ",
        )
        .bind(self.original_path.to_string_lossy().as_ref())
        .bind(self.quarantine_path.to_string_lossy().as_ref())
        .bind(&quarantine_date_str)
        .bind(&self.reason)
        .bind(&self.checksum_before)
        .bind(self.auto_quarantine)
        .bind(self.restored)
        .bind(&restore_date_str)
        .execute(pool)
        .await?;

        Ok(result.last_insert_rowid())
    }

    /// Load from database by ID
    pub async fn load(pool: &sqlx::SqlitePool, id: i64) -> ArchiveResult<Option<Self>> {
        let row = sqlx::query(
            r"
            SELECT id, original_path, quarantine_path, quarantine_date, reason, checksum_before, auto_quarantine, restored, restore_date
            FROM quarantine_records
            WHERE id = ?
            ",
        )
        .bind(id)
        .fetch_optional(pool)
        .await?;

        if let Some(row) = row {
            let quarantine_date_str: String = row.get("quarantine_date");
            let restore_date_str: Option<String> = row.get("restore_date");

            Ok(Some(Self {
                id: Some(row.get("id")),
                original_path: PathBuf::from(row.get::<String, _>("original_path")),
                quarantine_path: PathBuf::from(row.get::<String, _>("quarantine_path")),
                quarantine_date: DateTime::parse_from_rfc3339(&quarantine_date_str)
                    .map_err(|e| ArchiveError::Database(sqlx::Error::Decode(Box::new(e))))?
                    .with_timezone(&Utc),
                reason: row.get("reason"),
                checksum_before: row.get("checksum_before"),
                auto_quarantine: row.get("auto_quarantine"),
                restored: row.get("restored"),
                restore_date: restore_date_str
                    .map(|s| DateTime::parse_from_rfc3339(&s).map(|dt| dt.with_timezone(&Utc)))
                    .transpose()
                    .map_err(|e| ArchiveError::Database(sqlx::Error::Decode(Box::new(e))))?,
            }))
        } else {
            Ok(None)
        }
    }

    /// Load all quarantine records
    pub async fn load_all(pool: &sqlx::SqlitePool) -> ArchiveResult<Vec<Self>> {
        let rows = sqlx::query(
            r"
            SELECT id, original_path, quarantine_path, quarantine_date, reason, checksum_before, auto_quarantine, restored, restore_date
            FROM quarantine_records
            ORDER BY quarantine_date DESC
            ",
        )
        .fetch_all(pool)
        .await?;

        let mut records = Vec::new();
        for row in rows {
            let quarantine_date_str: String = row.get("quarantine_date");
            let restore_date_str: Option<String> = row.get("restore_date");

            records.push(Self {
                id: Some(row.get("id")),
                original_path: PathBuf::from(row.get::<String, _>("original_path")),
                quarantine_path: PathBuf::from(row.get::<String, _>("quarantine_path")),
                quarantine_date: DateTime::parse_from_rfc3339(&quarantine_date_str)
                    .map_err(|e| ArchiveError::Database(sqlx::Error::Decode(Box::new(e))))?
                    .with_timezone(&Utc),
                reason: row.get("reason"),
                checksum_before: row.get("checksum_before"),
                auto_quarantine: row.get("auto_quarantine"),
                restored: row.get("restored"),
                restore_date: restore_date_str
                    .map(|s| DateTime::parse_from_rfc3339(&s).map(|dt| dt.with_timezone(&Utc)))
                    .transpose()
                    .map_err(|e| ArchiveError::Database(sqlx::Error::Decode(Box::new(e))))?,
            });
        }

        Ok(records)
    }

    /// Mark as restored
    pub async fn mark_restored(&mut self, pool: &sqlx::SqlitePool) -> ArchiveResult<()> {
        self.restored = true;
        let now = Utc::now();
        self.restore_date = Some(now);
        let restore_date_str = now.to_rfc3339();

        sqlx::query(
            r"
            UPDATE quarantine_records
            SET restored = ?, restore_date = ?
            WHERE id = ?
            ",
        )
        .bind(self.restored)
        .bind(&restore_date_str)
        .bind(self.id)
        .execute(pool)
        .await?;

        Ok(())
    }
}

/// Quarantine a file
pub async fn quarantine_file(
    path: &Path,
    pool: &sqlx::SqlitePool,
    config: &VerificationConfig,
    reason: &str,
) -> ArchiveResult<QuarantineRecord> {
    info!("Quarantining file: {} (reason: {})", path.display(), reason);

    if !path.exists() {
        return Err(ArchiveError::Quarantine("File does not exist".to_string()));
    }

    // Ensure quarantine directory exists
    fs::create_dir_all(&config.quarantine_dir).await?;

    // Generate quarantine path
    let filename = path
        .file_name()
        .ok_or_else(|| ArchiveError::Quarantine("Invalid filename".to_string()))?;
    let timestamp = Utc::now().timestamp();
    let quarantine_filename = format!("{}_{}", timestamp, filename.to_string_lossy());
    let quarantine_path = config.quarantine_dir.join(quarantine_filename);

    // Compute checksum before moving
    let checksum_before = if config.enable_blake3 {
        Some(crate::checksum::compute_blake3(path).await?)
    } else {
        None
    };

    // Move file to quarantine
    fs::rename(path, &quarantine_path)
        .await
        .map_err(|e| ArchiveError::Quarantine(format!("Failed to move file: {e}")))?;

    info!(
        "Moved {} to quarantine: {}",
        path.display(),
        quarantine_path.display()
    );

    // Create quarantine record
    let mut record = QuarantineRecord::new(
        path.to_path_buf(),
        quarantine_path,
        reason.to_string(),
        checksum_before,
        config.auto_quarantine,
    );

    // Save to database
    let id = record.save(pool).await?;
    record.id = Some(id);

    // Log PREMIS event if enabled
    if config.enable_premis_logging {
        crate::fixity::log_premis_event(pool, &path.to_string_lossy(), "quarantine", "success")
            .await?;
    }

    Ok(record)
}

/// Restore a quarantined file
pub async fn restore_file(
    record_id: i64,
    pool: &sqlx::SqlitePool,
    config: &VerificationConfig,
) -> ArchiveResult<()> {
    let mut record = QuarantineRecord::load(pool, record_id)
        .await?
        .ok_or_else(|| ArchiveError::Quarantine("Quarantine record not found".to_string()))?;

    if record.restored {
        return Err(ArchiveError::Quarantine(
            "File already restored".to_string(),
        ));
    }

    if !record.quarantine_path.exists() {
        return Err(ArchiveError::Quarantine(
            "Quarantined file not found".to_string(),
        ));
    }

    // Check if original path exists (don't overwrite)
    if record.original_path.exists() {
        return Err(ArchiveError::Quarantine(
            "Original path already exists, cannot restore".to_string(),
        ));
    }

    // Ensure parent directory exists
    if let Some(parent) = record.original_path.parent() {
        fs::create_dir_all(parent).await?;
    }

    // Move file back
    fs::rename(&record.quarantine_path, &record.original_path)
        .await
        .map_err(|e| ArchiveError::Quarantine(format!("Failed to restore file: {e}")))?;

    info!(
        "Restored {} from quarantine",
        record.original_path.display()
    );

    // Update record
    record.mark_restored(pool).await?;

    // Log PREMIS event if enabled
    if config.enable_premis_logging {
        crate::fixity::log_premis_event(
            pool,
            &record.original_path.to_string_lossy(),
            "restore from quarantine",
            "success",
        )
        .await?;
    }

    Ok(())
}

/// Delete a quarantined file permanently
pub async fn delete_quarantined_file(record_id: i64, pool: &sqlx::SqlitePool) -> ArchiveResult<()> {
    let record = QuarantineRecord::load(pool, record_id)
        .await?
        .ok_or_else(|| ArchiveError::Quarantine("Quarantine record not found".to_string()))?;

    if record.quarantine_path.exists() {
        fs::remove_file(&record.quarantine_path).await?;
        info!(
            "Deleted quarantined file: {}",
            record.quarantine_path.display()
        );
    }

    // Remove record from database
    sqlx::query("DELETE FROM quarantine_records WHERE id = ?")
        .bind(record_id)
        .execute(pool)
        .await?;

    Ok(())
}

/// Quarantine status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuarantineStatus {
    pub total_quarantined: usize,
    pub active_quarantined: usize,
    pub restored: usize,
    pub auto_quarantined: usize,
    pub manual_quarantined: usize,
}

/// Get quarantine status
pub async fn get_quarantine_status(pool: &sqlx::SqlitePool) -> ArchiveResult<QuarantineStatus> {
    let records = QuarantineRecord::load_all(pool).await?;

    let total_quarantined = records.len();
    let active_quarantined = records.iter().filter(|r| !r.restored).count();
    let restored = records.iter().filter(|r| r.restored).count();
    let auto_quarantined = records.iter().filter(|r| r.auto_quarantine).count();
    let manual_quarantined = records.iter().filter(|r| !r.auto_quarantine).count();

    Ok(QuarantineStatus {
        total_quarantined,
        active_quarantined,
        restored,
        auto_quarantined,
        manual_quarantined,
    })
}

/// Repair workflow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepairWorkflow {
    pub file_path: PathBuf,
    pub corruption_detected: bool,
    pub repair_attempted: bool,
    pub repair_successful: bool,
    pub backup_available: bool,
    pub backup_path: Option<PathBuf>,
    pub steps_taken: Vec<String>,
    pub recommendations: Vec<String>,
}

impl RepairWorkflow {
    /// Create a new repair workflow
    pub fn new(file_path: PathBuf) -> Self {
        Self {
            file_path,
            corruption_detected: false,
            repair_attempted: false,
            repair_successful: false,
            backup_available: false,
            backup_path: None,
            steps_taken: Vec::new(),
            recommendations: Vec::new(),
        }
    }

    /// Add a step
    pub fn add_step(&mut self, step: String) {
        self.steps_taken.push(step);
    }

    /// Add a recommendation
    pub fn add_recommendation(&mut self, recommendation: String) {
        self.recommendations.push(recommendation);
    }
}

/// Attempt to repair a corrupted file
pub async fn attempt_repair(
    path: &Path,
    pool: &sqlx::SqlitePool,
    config: &VerificationConfig,
) -> ArchiveResult<RepairWorkflow> {
    let mut workflow = RepairWorkflow::new(path.to_path_buf());
    workflow.corruption_detected = true;
    workflow.add_step("Corruption detected".to_string());

    // Check for backup
    let backup_path = find_backup(path).await?;
    if let Some(ref backup) = backup_path {
        workflow.backup_available = true;
        workflow.backup_path = Some(backup.clone());
        workflow.add_step(format!("Found backup: {}", backup.display()));
        workflow.add_recommendation("Restore from backup".to_string());
    }

    // Try format-specific repair
    let container_format = crate::validate::detect_container_format(path).await?;
    match container_format.as_str() {
        "mp4" | "mov" => {
            workflow.add_step("Attempting MP4 repair".to_string());
            if let Ok(success) = attempt_mp4_repair(path).await {
                workflow.repair_attempted = true;
                workflow.repair_successful = success;
                if success {
                    workflow.add_step("MP4 repair successful".to_string());
                } else {
                    workflow.add_step("MP4 repair failed".to_string());
                    workflow
                        .add_recommendation("Use MP4Box or FFmpeg for manual repair".to_string());
                }
            }
        }
        "matroska" | "mkv" => {
            workflow.add_recommendation("Use mkvtoolnix for manual repair".to_string());
        }
        _ => {
            workflow
                .add_recommendation("No automatic repair available for this format".to_string());
        }
    }

    // If repair failed and backup is available, suggest restoration
    if workflow.repair_attempted && !workflow.repair_successful && workflow.backup_available {
        workflow.add_recommendation(
            "Automatic repair failed, restore from backup immediately".to_string(),
        );
    }

    // Log PREMIS event
    if config.enable_premis_logging {
        crate::fixity::log_premis_event(
            pool,
            &path.to_string_lossy(),
            "repair attempt",
            if workflow.repair_successful {
                "success"
            } else {
                "failure"
            },
        )
        .await?;
    }

    Ok(workflow)
}

/// Find backup for a file
async fn find_backup(path: &Path) -> ArchiveResult<Option<PathBuf>> {
    // Common backup locations
    let backup_extensions = [".bak", ".backup", ".orig"];
    let backup_dirs = ["backup", "backups", ".backup"];

    // Check for backup with extension
    for ext in &backup_extensions {
        let backup_path = PathBuf::from(format!("{}{}", path.display(), ext));
        if backup_path.exists() {
            return Ok(Some(backup_path));
        }
    }

    // Check in backup directories
    if let Some(parent) = path.parent() {
        if let Some(filename) = path.file_name() {
            for backup_dir in &backup_dirs {
                let backup_path = parent.join(backup_dir).join(filename);
                if backup_path.exists() {
                    return Ok(Some(backup_path));
                }
            }
        }
    }

    Ok(None)
}

/// Attempt MP4 repair using ffmpeg
async fn attempt_mp4_repair(path: &Path) -> ArchiveResult<bool> {
    let repaired_path = path.with_extension("repaired.mp4");

    let output = std::process::Command::new("ffmpeg")
        .args([
            "-y",
            "-i",
            path.to_str()
                .ok_or_else(|| ArchiveError::Quarantine("Invalid path".to_string()))?,
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            repaired_path
                .to_str()
                .ok_or_else(|| ArchiveError::Quarantine("Invalid path".to_string()))?,
        ])
        .output()
        .map_err(|e| ArchiveError::Quarantine(format!("ffmpeg not available: {e}")))?;

    if output.status.success() && repaired_path.exists() {
        // Replace original with repaired version
        fs::rename(&repaired_path, path).await?;
        info!("Successfully repaired MP4: {}", path.display());
        Ok(true)
    } else {
        // Clean up failed repair
        if repaired_path.exists() {
            let _ = fs::remove_file(&repaired_path).await;
        }
        warn!("Failed to repair MP4: {}", path.display());
        Ok(false)
    }
}

/// Restore from backup
pub async fn restore_from_backup(
    corrupted_path: &Path,
    backup_path: &Path,
    pool: &sqlx::SqlitePool,
    config: &VerificationConfig,
) -> ArchiveResult<()> {
    info!(
        "Restoring {} from backup {}",
        corrupted_path.display(),
        backup_path.display()
    );

    if !backup_path.exists() {
        return Err(ArchiveError::Quarantine(
            "Backup file does not exist".to_string(),
        ));
    }

    // Quarantine the corrupted file first
    quarantine_file(
        corrupted_path,
        pool,
        config,
        "Corrupted, restoring from backup",
    )
    .await?;

    // Copy backup to original location
    fs::copy(backup_path, corrupted_path).await?;

    info!("Restored {} from backup", corrupted_path.display());

    // Verify the restored file
    let checksums = crate::checksum::compute_checksums(corrupted_path, config).await?;
    info!("Restored file checksum (BLAKE3): {:?}", checksums.blake3);

    // Log PREMIS event
    if config.enable_premis_logging {
        crate::fixity::log_premis_event(
            pool,
            &corrupted_path.to_string_lossy(),
            "restore from backup",
            "success",
        )
        .await?;
    }

    Ok(())
}

/// Notification system for quarantine events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuarantineNotification {
    pub notification_type: NotificationType,
    pub file_path: PathBuf,
    pub timestamp: DateTime<Utc>,
    pub message: String,
    pub severity: NotificationSeverity,
}

/// Notification type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NotificationType {
    FileQuarantined,
    FileRestored,
    RepairAttempted,
    RepairSucceeded,
    RepairFailed,
    BackupRestored,
}

/// Notification severity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NotificationSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

impl QuarantineNotification {
    /// Create a new notification
    pub fn new(
        notification_type: NotificationType,
        file_path: PathBuf,
        message: String,
        severity: NotificationSeverity,
    ) -> Self {
        Self {
            notification_type,
            file_path,
            timestamp: Utc::now(),
            message,
            severity,
        }
    }

    /// Send notification (placeholder - implement actual notification logic)
    pub async fn send(&self) -> ArchiveResult<()> {
        // In a real implementation, this would send emails, log to monitoring systems, etc.
        match self.severity {
            NotificationSeverity::Critical | NotificationSeverity::Error => {
                error!(
                    "[NOTIFICATION] {}: {}",
                    self.file_path.display(),
                    self.message
                );
            }
            NotificationSeverity::Warning => {
                warn!(
                    "[NOTIFICATION] {}: {}",
                    self.file_path.display(),
                    self.message
                );
            }
            NotificationSeverity::Info => {
                info!(
                    "[NOTIFICATION] {}: {}",
                    self.file_path.display(),
                    self.message
                );
            }
        }
        Ok(())
    }
}

/// Send quarantine notification
pub async fn send_quarantine_notification(
    record: &QuarantineRecord,
    notification_type: NotificationType,
) -> ArchiveResult<()> {
    let (message, severity) = match notification_type {
        NotificationType::FileQuarantined => (
            format!("File quarantined: {}", record.reason),
            NotificationSeverity::Warning,
        ),
        NotificationType::FileRestored => (
            "File restored from quarantine".to_string(),
            NotificationSeverity::Info,
        ),
        _ => return Ok(()),
    };

    let notification = QuarantineNotification::new(
        notification_type,
        record.original_path.clone(),
        message,
        severity,
    );

    notification.send().await
}
