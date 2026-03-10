//! As-run log generation for broadcast compliance.

use crate::{AutomationError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::SystemTime;
use tokio::sync::RwLock;
use tracing::{debug, info};

/// As-run log entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsRunEntry {
    /// Entry ID
    pub id: String,
    /// Channel ID
    pub channel_id: usize,
    /// Item title
    pub title: String,
    /// Media file path
    pub file_path: Option<String>,
    /// Scheduled start time
    pub scheduled_start: SystemTime,
    /// Actual start time
    pub actual_start: SystemTime,
    /// Scheduled duration in seconds
    pub scheduled_duration: f64,
    /// Actual duration in seconds
    pub actual_duration: f64,
    /// Item type (program, commercial, promotion, etc.)
    pub item_type: String,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

impl AsRunEntry {
    /// Create a new as-run log entry.
    pub fn new(channel_id: usize, title: String) -> Self {
        let now = SystemTime::now();

        Self {
            id: format!("{now:?}"),
            channel_id,
            title,
            file_path: None,
            scheduled_start: now,
            actual_start: now,
            scheduled_duration: 0.0,
            actual_duration: 0.0,
            item_type: "unknown".to_string(),
            metadata: HashMap::new(),
        }
    }

    /// Calculate timing accuracy (difference between scheduled and actual).
    pub fn timing_accuracy(&self) -> f64 {
        if let Ok(duration) = self.actual_start.duration_since(self.scheduled_start) {
            duration.as_secs_f64()
        } else {
            0.0
        }
    }

    /// Check if timing is within tolerance.
    pub fn is_timing_accurate(&self, tolerance_secs: f64) -> bool {
        self.timing_accuracy().abs() <= tolerance_secs
    }
}

/// As-run log format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AsRunFormat {
    /// JSON format
    Json,
    /// CSV format
    Csv,
    /// XML format
    Xml,
}

/// As-run logger.
pub struct AsRunLogger {
    entries: Arc<RwLock<Vec<AsRunEntry>>>,
    log_path: Option<PathBuf>,
}

impl AsRunLogger {
    /// Create a new as-run logger.
    pub fn new() -> Result<Self> {
        info!("Creating as-run logger");

        Ok(Self {
            entries: Arc::new(RwLock::new(Vec::new())),
            log_path: None,
        })
    }

    /// Create with log file path.
    pub fn with_path(log_path: PathBuf) -> Result<Self> {
        info!("Creating as-run logger with path: {:?}", log_path);

        Ok(Self {
            entries: Arc::new(RwLock::new(Vec::new())),
            log_path: Some(log_path),
        })
    }

    /// Log an as-run entry.
    pub async fn log(&self, entry: AsRunEntry) -> Result<()> {
        debug!("Logging as-run entry: {}", entry.title);

        let mut entries = self.entries.write().await;
        entries.push(entry);

        Ok(())
    }

    /// Get all entries.
    pub async fn get_entries(&self) -> Vec<AsRunEntry> {
        self.entries.read().await.clone()
    }

    /// Get entries for a specific channel.
    pub async fn get_channel_entries(&self, channel_id: usize) -> Vec<AsRunEntry> {
        let entries = self.entries.read().await;
        entries
            .iter()
            .filter(|e| e.channel_id == channel_id)
            .cloned()
            .collect()
    }

    /// Clear all entries.
    pub async fn clear(&self) {
        info!("Clearing as-run log");

        let mut entries = self.entries.write().await;
        entries.clear();
    }

    /// Export log to file.
    pub async fn export(&self, format: AsRunFormat) -> Result<()> {
        if let Some(ref log_path) = self.log_path {
            info!("Exporting as-run log to: {:?}", log_path);

            let entries = self.entries.read().await;

            match format {
                AsRunFormat::Json => self.export_json(&entries, log_path)?,
                AsRunFormat::Csv => self.export_csv(&entries, log_path)?,
                AsRunFormat::Xml => self.export_xml(&entries, log_path)?,
            }

            Ok(())
        } else {
            Err(AutomationError::Logging(
                "No log path configured".to_string(),
            ))
        }
    }

    /// Export to JSON format.
    fn export_json(&self, entries: &[AsRunEntry], path: &PathBuf) -> Result<()> {
        let json = serde_json::to_string_pretty(entries)?;
        let mut file = File::create(path)?;
        file.write_all(json.as_bytes())?;
        Ok(())
    }

    /// Export to CSV format.
    fn export_csv(&self, entries: &[AsRunEntry], path: &PathBuf) -> Result<()> {
        let mut file = File::create(path)?;

        // Write header
        writeln!(file, "ID,Channel,Title,Item Type,Scheduled Start,Actual Start,Scheduled Duration,Actual Duration")?;

        // Write entries
        for entry in entries {
            writeln!(
                file,
                "{},{},{},{},{:?},{:?},{},{}",
                entry.id,
                entry.channel_id,
                entry.title,
                entry.item_type,
                entry.scheduled_start,
                entry.actual_start,
                entry.scheduled_duration,
                entry.actual_duration
            )?;
        }

        Ok(())
    }

    /// Export to XML format.
    fn export_xml(&self, entries: &[AsRunEntry], path: &PathBuf) -> Result<()> {
        let mut file = File::create(path)?;

        writeln!(file, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>")?;
        writeln!(file, "<asrun>")?;

        for entry in entries {
            writeln!(file, "  <entry>")?;
            writeln!(file, "    <id>{}</id>", entry.id)?;
            writeln!(file, "    <channel>{}</channel>", entry.channel_id)?;
            writeln!(file, "    <title>{}</title>", entry.title)?;
            writeln!(file, "    <type>{}</type>", entry.item_type)?;
            writeln!(file, "  </entry>")?;
        }

        writeln!(file, "</asrun>")?;

        Ok(())
    }
}

/// As-run log type alias.
pub type AsRunLog = AsRunLogger;

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_asrun_logger() {
        let logger = AsRunLogger::new().expect("new should succeed");

        let entry = AsRunEntry::new(0, "Test Program".to_string());
        logger.log(entry).await.expect("operation should succeed");

        let entries = logger.get_entries().await;
        assert_eq!(entries.len(), 1);
    }

    #[tokio::test]
    async fn test_channel_entries() {
        let logger = AsRunLogger::new().expect("new should succeed");

        let entry1 = AsRunEntry::new(0, "Program 1".to_string());
        let entry2 = AsRunEntry::new(1, "Program 2".to_string());

        logger.log(entry1).await.expect("operation should succeed");
        logger.log(entry2).await.expect("operation should succeed");

        let channel0_entries = logger.get_channel_entries(0).await;
        assert_eq!(channel0_entries.len(), 1);

        let channel1_entries = logger.get_channel_entries(1).await;
        assert_eq!(channel1_entries.len(), 1);
    }

    #[test]
    fn test_timing_accuracy() {
        let mut entry = AsRunEntry::new(0, "Test".to_string());
        entry.scheduled_start = SystemTime::now();
        entry.actual_start = SystemTime::now();

        assert!(entry.is_timing_accurate(1.0));
    }
}
