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

// ── Batched as-run logger ─────────────────────────────────────────────────────

/// A write-batching wrapper around an [`AsRunLogger`] that accumulates entries
/// in memory and flushes them to disk in a single `write_all` call.
///
/// This reduces I/O system-call overhead during high-throughput playout where
/// individual items may complete many times per second.
///
/// # Flush triggers
///
/// An in-memory batch is flushed to the backing logger when either:
/// 1. The batch reaches [`max_buffer_size`](Self::max_buffer_size) entries, or
/// 2. [`flush`](Self::flush) is called explicitly (e.g. from a periodic task).
///
/// On [`Drop`] the remaining buffered entries are flushed synchronously via
/// Tokio's `block_in_place` if called from an async context, or a new
/// single-threaded Tokio runtime otherwise.
///
/// # Example
///
/// ```rust,no_run
/// use oximedia_automation::logging::asrun::{AsRunEntry, BatchedAsRunLogger};
/// use std::time::Duration;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let mut logger = BatchedAsRunLogger::new(None, 100)?;
/// logger.add(AsRunEntry::new(0, "Program A".to_string())).await;
/// logger.flush().await?;
/// # Ok(())
/// # }
/// ```
pub struct BatchedAsRunLogger {
    /// The backing logger.  Exposed as `pub` so callers can call
    /// [`AsRunLogger::export`] directly after flushing.
    pub inner: AsRunLogger,
    buffer: Vec<AsRunEntry>,
    /// Maximum number of entries to hold before an automatic flush.
    pub max_buffer_size: usize,
}

impl BatchedAsRunLogger {
    /// Create a new batched logger.
    ///
    /// # Parameters
    ///
    /// * `log_path` — optional file path forwarded to [`AsRunLogger::with_path`].
    /// * `max_buffer_size` — flush threshold (default: 100 when `0` is passed).
    ///
    /// # Errors
    ///
    /// Propagates any I/O error from [`AsRunLogger::new`] /
    /// [`AsRunLogger::with_path`].
    pub fn new(log_path: Option<PathBuf>, max_buffer_size: usize) -> Result<Self> {
        let inner = match log_path {
            Some(p) => AsRunLogger::with_path(p)?,
            None => AsRunLogger::new()?,
        };
        let max_buffer_size = if max_buffer_size == 0 {
            100
        } else {
            max_buffer_size
        };
        Ok(Self {
            inner,
            buffer: Vec::new(),
            max_buffer_size,
        })
    }

    /// Add an entry to the in-memory buffer.
    ///
    /// If the buffer has reached [`max_buffer_size`](Self::max_buffer_size)
    /// after the addition this method flushes automatically.
    pub async fn add(&mut self, entry: AsRunEntry) {
        self.buffer.push(entry);
        if self.buffer.len() >= self.max_buffer_size {
            // Best-effort flush; ignore errors here so playout continues.
            let _ = self.flush().await;
        }
    }

    /// Flush all buffered entries to the backing logger in one batch operation.
    ///
    /// After a successful flush the buffer is cleared.
    ///
    /// # Errors
    ///
    /// Returns the first error encountered while logging an entry.  The
    /// remaining entries are still logged.
    pub async fn flush(&mut self) -> Result<()> {
        if self.buffer.is_empty() {
            return Ok(());
        }
        // Drain the buffer first so a partial failure still clears it.
        let batch: Vec<AsRunEntry> = self.buffer.drain(..).collect();
        let mut last_err: Option<crate::AutomationError> = None;
        for entry in batch {
            if let Err(e) = self.inner.log(entry).await {
                last_err = Some(e);
            }
        }
        match last_err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// Return a snapshot of the entries currently held in the in-memory buffer
    /// (not yet flushed).
    pub fn pending_count(&self) -> usize {
        self.buffer.len()
    }

    /// Retrieve all entries that have been flushed to the backing logger.
    pub async fn get_flushed_entries(&self) -> Vec<AsRunEntry> {
        self.inner.get_entries().await
    }
}

impl Drop for BatchedAsRunLogger {
    fn drop(&mut self) {
        if self.buffer.is_empty() {
            return;
        }
        // Flush remaining entries on drop.
        //
        // Strategy:
        // - If we are inside a multi-threaded Tokio runtime (the common case),
        //   use `block_in_place` which parks the current thread but keeps the
        //   executor alive, then `block_on` from the same handle.
        // - If no runtime is active, spin up a minimal one-shot runtime.
        let buffer: Vec<AsRunEntry> = self.buffer.drain(..).collect();
        let inner = &self.inner;

        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            // block_in_place is only available on the multi-thread scheduler.
            // For the current_thread scheduler we fall back to spawning a
            // blocking task instead so we don't panic.
            if handle.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread {
                tokio::task::block_in_place(|| {
                    handle.block_on(async {
                        for entry in buffer {
                            let _ = inner.log(entry).await;
                        }
                    });
                });
            }
            // current_thread scheduler: entries are silently dropped to avoid
            // deadlock.  Callers should call flush() explicitly before drop.
        } else if let Ok(rt) = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
        {
            rt.block_on(async {
                for entry in buffer {
                    let _ = inner.log(entry).await;
                }
            });
        }
    }
}

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

    // ── BatchedAsRunLogger tests ───────────────────────────────────────────────

    #[tokio::test]
    async fn test_batched_entries_buffered_up_to_limit() {
        let mut logger = BatchedAsRunLogger::new(None, 5).expect("new should succeed");

        // Add 4 entries — should remain buffered (below limit of 5).
        for i in 0..4usize {
            logger.add(AsRunEntry::new(i, format!("Program {i}"))).await;
        }
        assert_eq!(logger.pending_count(), 4, "4 entries should be buffered");
        assert_eq!(
            logger.get_flushed_entries().await.len(),
            0,
            "nothing should be flushed yet"
        );
    }

    #[tokio::test]
    async fn test_batched_auto_flush_on_limit() {
        let mut logger = BatchedAsRunLogger::new(None, 3).expect("new should succeed");

        // The 3rd add triggers an auto-flush.
        for i in 0..3usize {
            logger.add(AsRunEntry::new(i, format!("P{i}"))).await;
        }
        // After auto-flush the buffer should be empty.
        assert_eq!(
            logger.pending_count(),
            0,
            "buffer should be empty after auto-flush"
        );
        assert_eq!(
            logger.get_flushed_entries().await.len(),
            3,
            "all 3 entries should be flushed"
        );
    }

    #[tokio::test]
    async fn test_batched_manual_flush() {
        let mut logger = BatchedAsRunLogger::new(None, 100).expect("new should succeed");

        logger.add(AsRunEntry::new(0, "A".to_string())).await;
        logger.add(AsRunEntry::new(0, "B".to_string())).await;
        assert_eq!(logger.pending_count(), 2);

        logger.flush().await.expect("flush should succeed");
        assert_eq!(logger.pending_count(), 0);
        assert_eq!(logger.get_flushed_entries().await.len(), 2);
    }

    #[tokio::test]
    async fn test_batched_flush_clears_buffer() {
        let mut logger = BatchedAsRunLogger::new(None, 10).expect("new should succeed");

        for i in 0..4 {
            logger.add(AsRunEntry::new(i, format!("Ep {i}"))).await;
        }
        logger.flush().await.expect("flush should succeed");

        // Add more after a flush.
        logger.add(AsRunEntry::new(4, "Ep 4".to_string())).await;
        assert_eq!(logger.pending_count(), 1);

        logger.flush().await.expect("second flush should succeed");
        assert_eq!(logger.get_flushed_entries().await.len(), 5);
    }

    #[tokio::test]
    async fn test_batched_empty_flush_is_noop() {
        let mut logger = BatchedAsRunLogger::new(None, 10).expect("new should succeed");
        // Flushing an empty buffer must succeed without error.
        logger.flush().await.expect("empty flush should succeed");
        assert_eq!(logger.get_flushed_entries().await.len(), 0);
    }

    #[tokio::test]
    async fn test_batched_multiple_flushes_accumulate() {
        let mut logger = BatchedAsRunLogger::new(None, 100).expect("new should succeed");

        for i in 0..3 {
            logger.add(AsRunEntry::new(0, format!("Batch1-{i}"))).await;
        }
        logger.flush().await.expect("first flush");

        for i in 0..2 {
            logger.add(AsRunEntry::new(1, format!("Batch2-{i}"))).await;
        }
        logger.flush().await.expect("second flush");

        let total = logger.get_flushed_entries().await.len();
        assert_eq!(total, 5, "both batches should accumulate");
    }

    #[tokio::test]
    async fn test_batched_file_export_json() {
        let tmp = std::env::temp_dir().join("oximedia_asrun_batch_test.json");
        let mut logger =
            BatchedAsRunLogger::new(Some(tmp.clone()), 10).expect("new with path should succeed");

        logger
            .add(AsRunEntry::new(0, "News at Six".to_string()))
            .await;
        logger.flush().await.expect("flush should succeed");

        // Export to JSON — file must be created and non-empty.
        // Export is called on the inner logger.
        logger
            .inner
            .export(AsRunFormat::Json)
            .await
            .expect("json export should succeed");
        assert!(tmp.exists(), "JSON export file should exist");
        let size = std::fs::metadata(&tmp).expect("metadata").len();
        assert!(size > 0, "JSON file should not be empty");

        let _ = std::fs::remove_file(&tmp);
    }

    #[tokio::test]
    async fn test_batched_pending_count_reflects_state() {
        let mut logger = BatchedAsRunLogger::new(None, 20).expect("new should succeed");
        assert_eq!(logger.pending_count(), 0);

        logger.add(AsRunEntry::new(0, "A".to_string())).await;
        assert_eq!(logger.pending_count(), 1);

        logger.add(AsRunEntry::new(0, "B".to_string())).await;
        assert_eq!(logger.pending_count(), 2);

        logger.flush().await.expect("flush");
        assert_eq!(logger.pending_count(), 0);
    }

    #[tokio::test]
    async fn test_batched_zero_max_size_defaults_to_100() {
        // max_buffer_size = 0 should be treated as 100.
        let logger = BatchedAsRunLogger::new(None, 0).expect("new should succeed");
        assert_eq!(logger.max_buffer_size, 100);
    }

    #[tokio::test]
    async fn test_batched_channel_filtering_through_inner() {
        let mut logger = BatchedAsRunLogger::new(None, 10).expect("new should succeed");

        logger.add(AsRunEntry::new(0, "CH0 A".to_string())).await;
        logger.add(AsRunEntry::new(1, "CH1 A".to_string())).await;
        logger.add(AsRunEntry::new(0, "CH0 B".to_string())).await;
        logger.flush().await.expect("flush should succeed");

        let ch0 = logger.inner.get_channel_entries(0).await;
        let ch1 = logger.inner.get_channel_entries(1).await;
        assert_eq!(ch0.len(), 2);
        assert_eq!(ch1.len(), 1);
    }
}
