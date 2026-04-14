#![allow(dead_code)]
//! Conversion logging and audit trail for media conversions.
//!
//! This module provides structured logging of every conversion operation
//! including input/output metadata, timing, quality metrics, and error
//! details. Logs can be queried, filtered, and exported for compliance
//! reporting and debugging.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, SystemTime};

/// Severity level for log entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LogLevel {
    /// Debug information for development.
    Debug,
    /// Informational messages about normal operation.
    Info,
    /// Warnings about non-critical issues.
    Warning,
    /// Errors that caused partial or full failure.
    Error,
}

impl std::fmt::Display for LogLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Debug => write!(f, "DEBUG"),
            Self::Info => write!(f, "INFO"),
            Self::Warning => write!(f, "WARN"),
            Self::Error => write!(f, "ERROR"),
        }
    }
}

/// Outcome of a conversion operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConversionOutcome {
    /// Conversion completed successfully.
    Success,
    /// Conversion completed with warnings.
    SuccessWithWarnings,
    /// Conversion failed.
    Failed,
    /// Conversion was cancelled by the user.
    Cancelled,
    /// Conversion was skipped (e.g., output already exists).
    Skipped,
}

impl std::fmt::Display for ConversionOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Success => write!(f, "SUCCESS"),
            Self::SuccessWithWarnings => write!(f, "SUCCESS_WITH_WARNINGS"),
            Self::Failed => write!(f, "FAILED"),
            Self::Cancelled => write!(f, "CANCELLED"),
            Self::Skipped => write!(f, "SKIPPED"),
        }
    }
}

/// A single log entry recording a conversion event.
#[derive(Debug, Clone)]
pub struct ConvertLogEntry {
    /// Unique identifier for this conversion job.
    pub job_id: String,
    /// Timestamp of this log entry.
    pub timestamp: SystemTime,
    /// Log severity level.
    pub level: LogLevel,
    /// Phase of conversion (e.g., "detect", "decode", "encode", "mux").
    pub phase: String,
    /// Human-readable message.
    pub message: String,
    /// Optional key-value metadata.
    pub metadata: HashMap<String, String>,
}

impl ConvertLogEntry {
    /// Create a new log entry.
    #[must_use]
    pub fn new(job_id: &str, level: LogLevel, phase: &str, message: &str) -> Self {
        Self {
            job_id: job_id.to_string(),
            timestamp: SystemTime::now(),
            level,
            phase: phase.to_string(),
            message: message.to_string(),
            metadata: HashMap::new(),
        }
    }

    /// Add a metadata key-value pair to this entry.
    #[must_use]
    pub fn with_meta(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }

    /// Format this entry as a single log line.
    #[must_use]
    pub fn format_line(&self) -> String {
        let elapsed = self
            .timestamp
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or(Duration::ZERO);
        format!(
            "[{:.3}] [{}] [{}] [{}] {}",
            elapsed.as_secs_f64(),
            self.level,
            self.job_id,
            self.phase,
            self.message,
        )
    }
}

/// Summary of a completed conversion job.
#[derive(Debug, Clone)]
pub struct ConversionJobSummary {
    /// Unique job identifier.
    pub job_id: String,
    /// Input file path.
    pub input_path: PathBuf,
    /// Output file path.
    pub output_path: PathBuf,
    /// Input file size in bytes.
    pub input_size_bytes: u64,
    /// Output file size in bytes.
    pub output_size_bytes: u64,
    /// Wall-clock duration of the conversion.
    pub wall_duration: Duration,
    /// Processing speed ratio (e.g., 2.5x means 2.5 times faster than real-time).
    pub speed_ratio: f64,
    /// Outcome of the conversion.
    pub outcome: ConversionOutcome,
    /// Number of warnings encountered.
    pub warning_count: usize,
    /// Number of errors encountered.
    pub error_count: usize,
    /// Codec used for video output.
    pub video_codec: Option<String>,
    /// Codec used for audio output.
    pub audio_codec: Option<String>,
}

impl ConversionJobSummary {
    /// Compute the compression ratio (`input_size` / `output_size`).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn compression_ratio(&self) -> f64 {
        if self.output_size_bytes == 0 {
            0.0
        } else {
            self.input_size_bytes as f64 / self.output_size_bytes as f64
        }
    }

    /// Compute the size reduction percentage.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn size_reduction_pct(&self) -> f64 {
        if self.input_size_bytes == 0 {
            0.0
        } else {
            let diff = self.input_size_bytes.saturating_sub(self.output_size_bytes);
            (diff as f64 / self.input_size_bytes as f64) * 100.0
        }
    }

    /// Whether the conversion was successful (with or without warnings).
    #[must_use]
    pub fn is_success(&self) -> bool {
        matches!(
            self.outcome,
            ConversionOutcome::Success | ConversionOutcome::SuccessWithWarnings
        )
    }
}

/// In-memory conversion log that accumulates entries.
#[derive(Debug, Clone)]
pub struct ConvertLog {
    /// All log entries.
    entries: Vec<ConvertLogEntry>,
    /// Maximum number of entries to retain (0 = unlimited).
    max_entries: usize,
}

impl ConvertLog {
    /// Create a new empty log.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            max_entries: 0,
        }
    }

    /// Create a log with a maximum entry limit.
    #[must_use]
    pub fn with_capacity(max_entries: usize) -> Self {
        Self {
            entries: Vec::with_capacity(max_entries.min(10000)),
            max_entries,
        }
    }

    /// Append a log entry.
    pub fn push(&mut self, entry: ConvertLogEntry) {
        if self.max_entries > 0 && self.entries.len() >= self.max_entries {
            self.entries.remove(0);
        }
        self.entries.push(entry);
    }

    /// Append a simple message at the given level.
    pub fn log(&mut self, job_id: &str, level: LogLevel, phase: &str, message: &str) {
        self.push(ConvertLogEntry::new(job_id, level, phase, message));
    }

    /// Number of entries in the log.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the log is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get all entries.
    #[must_use]
    pub fn entries(&self) -> &[ConvertLogEntry] {
        &self.entries
    }

    /// Filter entries by job ID.
    #[must_use]
    pub fn filter_by_job(&self, job_id: &str) -> Vec<&ConvertLogEntry> {
        self.entries.iter().filter(|e| e.job_id == job_id).collect()
    }

    /// Filter entries by minimum log level.
    #[must_use]
    pub fn filter_by_level(&self, min_level: LogLevel) -> Vec<&ConvertLogEntry> {
        self.entries
            .iter()
            .filter(|e| e.level >= min_level)
            .collect()
    }

    /// Count entries at or above the given level.
    #[must_use]
    pub fn count_at_level(&self, min_level: LogLevel) -> usize {
        self.entries.iter().filter(|e| e.level >= min_level).count()
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Export all entries as formatted lines.
    #[must_use]
    pub fn export_lines(&self) -> Vec<String> {
        self.entries
            .iter()
            .map(ConvertLogEntry::format_line)
            .collect()
    }
}

impl Default for ConvertLog {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_level_ordering() {
        assert!(LogLevel::Debug < LogLevel::Info);
        assert!(LogLevel::Info < LogLevel::Warning);
        assert!(LogLevel::Warning < LogLevel::Error);
    }

    #[test]
    fn test_log_level_display() {
        assert_eq!(format!("{}", LogLevel::Debug), "DEBUG");
        assert_eq!(format!("{}", LogLevel::Info), "INFO");
        assert_eq!(format!("{}", LogLevel::Warning), "WARN");
        assert_eq!(format!("{}", LogLevel::Error), "ERROR");
    }

    #[test]
    fn test_outcome_display() {
        assert_eq!(format!("{}", ConversionOutcome::Success), "SUCCESS");
        assert_eq!(format!("{}", ConversionOutcome::Failed), "FAILED");
        assert_eq!(format!("{}", ConversionOutcome::Cancelled), "CANCELLED");
        assert_eq!(format!("{}", ConversionOutcome::Skipped), "SKIPPED");
    }

    #[test]
    fn test_log_entry_creation() {
        let entry = ConvertLogEntry::new("job-1", LogLevel::Info, "encode", "Starting encode");
        assert_eq!(entry.job_id, "job-1");
        assert_eq!(entry.level, LogLevel::Info);
        assert_eq!(entry.phase, "encode");
        assert_eq!(entry.message, "Starting encode");
    }

    #[test]
    fn test_log_entry_with_meta() {
        let entry = ConvertLogEntry::new("j1", LogLevel::Debug, "init", "ok")
            .with_meta("codec", "h264")
            .with_meta("bitrate", "5000000");
        assert_eq!(entry.metadata.get("codec").unwrap(), "h264");
        assert_eq!(entry.metadata.get("bitrate").unwrap(), "5000000");
    }

    #[test]
    fn test_log_entry_format_line() {
        let entry = ConvertLogEntry::new("j2", LogLevel::Warning, "mux", "container mismatch");
        let line = entry.format_line();
        assert!(line.contains("WARN"));
        assert!(line.contains("j2"));
        assert!(line.contains("mux"));
        assert!(line.contains("container mismatch"));
    }

    #[test]
    fn test_convert_log_push_and_len() {
        let mut log = ConvertLog::new();
        assert!(log.is_empty());
        log.log("j1", LogLevel::Info, "detect", "Detected MP4");
        assert_eq!(log.len(), 1);
        log.log("j1", LogLevel::Info, "decode", "Decoding started");
        assert_eq!(log.len(), 2);
    }

    #[test]
    fn test_convert_log_capacity_limit() {
        let mut log = ConvertLog::with_capacity(3);
        for i in 0..5 {
            log.log(&format!("j{i}"), LogLevel::Info, "test", "msg");
        }
        assert_eq!(log.len(), 3);
        // Oldest entries should have been evicted
        assert_eq!(log.entries()[0].job_id, "j2");
    }

    #[test]
    fn test_filter_by_job() {
        let mut log = ConvertLog::new();
        log.log("j1", LogLevel::Info, "a", "m1");
        log.log("j2", LogLevel::Info, "b", "m2");
        log.log("j1", LogLevel::Warning, "c", "m3");
        let filtered = log.filter_by_job("j1");
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_filter_by_level() {
        let mut log = ConvertLog::new();
        log.log("j1", LogLevel::Debug, "a", "d");
        log.log("j1", LogLevel::Info, "b", "i");
        log.log("j1", LogLevel::Warning, "c", "w");
        log.log("j1", LogLevel::Error, "d", "e");
        let warnings_and_above = log.filter_by_level(LogLevel::Warning);
        assert_eq!(warnings_and_above.len(), 2);
    }

    #[test]
    fn test_count_at_level() {
        let mut log = ConvertLog::new();
        log.log("j1", LogLevel::Debug, "a", "d");
        log.log("j1", LogLevel::Error, "b", "e");
        assert_eq!(log.count_at_level(LogLevel::Error), 1);
        assert_eq!(log.count_at_level(LogLevel::Debug), 2);
    }

    #[test]
    fn test_clear_log() {
        let mut log = ConvertLog::new();
        log.log("j1", LogLevel::Info, "a", "m");
        log.clear();
        assert!(log.is_empty());
    }

    #[test]
    fn test_export_lines() {
        let mut log = ConvertLog::new();
        log.log("j1", LogLevel::Info, "a", "hello");
        log.log("j1", LogLevel::Error, "b", "world");
        let lines = log.export_lines();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("hello"));
        assert!(lines[1].contains("world"));
    }

    #[test]
    fn test_job_summary_compression_ratio() {
        let summary = ConversionJobSummary {
            job_id: "j1".into(),
            input_path: PathBuf::from("in.mov"),
            output_path: PathBuf::from("out.mp4"),
            input_size_bytes: 1_000_000,
            output_size_bytes: 500_000,
            wall_duration: Duration::from_secs(10),
            speed_ratio: 5.0,
            outcome: ConversionOutcome::Success,
            warning_count: 0,
            error_count: 0,
            video_codec: Some("h264".into()),
            audio_codec: Some("aac".into()),
        };
        assert!((summary.compression_ratio() - 2.0).abs() < 1e-6);
        assert!((summary.size_reduction_pct() - 50.0).abs() < 1e-6);
        assert!(summary.is_success());
    }

    #[test]
    fn test_job_summary_zero_output() {
        let summary = ConversionJobSummary {
            job_id: "j2".into(),
            input_path: PathBuf::from("in.mov"),
            output_path: PathBuf::from("out.mp4"),
            input_size_bytes: 100,
            output_size_bytes: 0,
            wall_duration: Duration::from_secs(1),
            speed_ratio: 0.0,
            outcome: ConversionOutcome::Failed,
            warning_count: 0,
            error_count: 1,
            video_codec: None,
            audio_codec: None,
        };
        assert!((summary.compression_ratio() - 0.0).abs() < f64::EPSILON);
        assert!(!summary.is_success());
    }

    #[test]
    fn test_outcome_success_with_warnings() {
        let summary = ConversionJobSummary {
            job_id: "j3".into(),
            input_path: PathBuf::from("in.mov"),
            output_path: PathBuf::from("out.mp4"),
            input_size_bytes: 100,
            output_size_bytes: 80,
            wall_duration: Duration::from_secs(1),
            speed_ratio: 1.0,
            outcome: ConversionOutcome::SuccessWithWarnings,
            warning_count: 2,
            error_count: 0,
            video_codec: None,
            audio_codec: None,
        };
        assert!(summary.is_success());
    }
}
