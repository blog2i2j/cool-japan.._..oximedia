//! Timecode log module for recording timecode-stamped production notes
//! and metadata events.
//!
//! This module provides a production log that associates textual notes,
//! event markers, and metadata with specific timecode positions. It is
//! useful for:
//!
//! - Logging editorial decisions with precise timecode reference
//! - Recording QC (Quality Control) findings at specific points
//! - Storing production cue sheets and event lists
//! - Exporting EDL-compatible event descriptions
//!
//! # Example
//!
//! ```rust
//! use oximedia_timecode::{Timecode, FrameRate};
//! use oximedia_timecode::timecode_log::{TimecodeLog, LogEntry, LogLevel};
//!
//! let mut log = TimecodeLog::new("Production A");
//! let tc = Timecode::new(1, 0, 0, 0, FrameRate::Fps25).expect("valid tc");
//! log.record(tc, LogLevel::Info, "Scene 1 start");
//! ```

#![allow(dead_code)]

use crate::{FrameRate, Timecode, TimecodeError};
use std::fmt;

/// Severity level for log entries.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub enum LogLevel {
    /// Debug-level event (verbose).
    Debug = 0,
    /// Informational event (normal production notes).
    Info = 1,
    /// Warning event (potential issue that was resolved or noted).
    Warning = 2,
    /// Error event (problem that affected the production).
    Error = 3,
    /// Critical event (scene/take retake required).
    Critical = 4,
}

impl fmt::Display for LogLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LogLevel::Debug => write!(f, "DEBUG"),
            LogLevel::Info => write!(f, "INFO"),
            LogLevel::Warning => write!(f, "WARN"),
            LogLevel::Error => write!(f, "ERROR"),
            LogLevel::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// A single entry in the timecode log.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LogEntry {
    /// The timecode position of this event.
    pub timecode: Timecode,
    /// Severity level.
    pub level: LogLevel,
    /// Human-readable message.
    pub message: String,
    /// Optional category tag (e.g., "QC", "EDITORIAL", "AUDIO").
    pub category: Option<String>,
    /// Optional metadata key-value pairs.
    pub metadata: std::collections::HashMap<String, String>,
    /// Wall-clock timestamp when this entry was recorded (Unix seconds).
    pub wall_clock_secs: Option<i64>,
}

impl LogEntry {
    /// Create a new log entry.
    #[must_use]
    pub fn new(timecode: Timecode, level: LogLevel, message: impl Into<String>) -> Self {
        Self {
            timecode,
            level,
            message: message.into(),
            category: None,
            metadata: std::collections::HashMap::new(),
            wall_clock_secs: None,
        }
    }

    /// Set the category tag.
    #[must_use]
    pub fn with_category(mut self, category: impl Into<String>) -> Self {
        self.category = Some(category.into());
        self
    }

    /// Add a metadata key-value pair.
    #[must_use]
    pub fn with_meta(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Set the wall-clock timestamp.
    #[must_use]
    pub fn with_wall_clock(mut self, secs: i64) -> Self {
        self.wall_clock_secs = Some(secs);
        self
    }

    /// Format the entry as a single log line.
    #[must_use]
    pub fn format_line(&self) -> String {
        let cat = self
            .category
            .as_deref()
            .map(|c| format!("[{c}] "))
            .unwrap_or_default();
        format!("{} {}{} {}", self.timecode, cat, self.level, self.message)
    }
}

impl fmt::Display for LogEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.format_line())
    }
}

/// Filter criteria for querying log entries.
#[derive(Debug, Default)]
pub struct LogFilter {
    /// Minimum log level (inclusive).
    pub min_level: Option<LogLevel>,
    /// Required category (exact match).
    pub category: Option<String>,
    /// Substring match on message.
    pub message_contains: Option<String>,
    /// Start timecode of the range (inclusive).
    pub range_start: Option<Timecode>,
    /// End timecode of the range (inclusive).
    pub range_end: Option<Timecode>,
}

impl LogFilter {
    /// Create a filter that matches all entries.
    #[must_use]
    pub fn all() -> Self {
        Self::default()
    }

    /// Filter by minimum level.
    #[must_use]
    pub fn with_min_level(mut self, level: LogLevel) -> Self {
        self.min_level = Some(level);
        self
    }

    /// Filter by category.
    #[must_use]
    pub fn with_category(mut self, cat: impl Into<String>) -> Self {
        self.category = Some(cat.into());
        self
    }

    /// Filter by message substring.
    #[must_use]
    pub fn with_message(mut self, msg: impl Into<String>) -> Self {
        self.message_contains = Some(msg.into());
        self
    }

    /// Filter by timecode range.
    #[must_use]
    pub fn with_range(mut self, start: Timecode, end: Timecode) -> Self {
        self.range_start = Some(start);
        self.range_end = Some(end);
        self
    }

    /// Test whether an entry matches this filter.
    #[must_use]
    pub fn matches(&self, entry: &LogEntry) -> bool {
        if let Some(min) = self.min_level {
            if entry.level < min {
                return false;
            }
        }
        if let Some(ref cat) = self.category {
            if entry.category.as_deref() != Some(cat.as_str()) {
                return false;
            }
        }
        if let Some(ref needle) = self.message_contains {
            if !entry.message.contains(needle.as_str()) {
                return false;
            }
        }
        if let Some(ref start) = self.range_start {
            if entry.timecode < *start {
                return false;
            }
        }
        if let Some(ref end) = self.range_end {
            if entry.timecode > *end {
                return false;
            }
        }
        true
    }
}

/// Production timecode log.
///
/// A sorted list of [`LogEntry`] values that can be recorded, queried,
/// and exported in various formats.
#[derive(Debug)]
pub struct TimecodeLog {
    /// Log name / production title.
    pub name: String,
    /// All log entries, kept sorted by timecode.
    entries: Vec<LogEntry>,
}

impl TimecodeLog {
    /// Create a new empty log.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            entries: Vec::new(),
        }
    }

    /// Record a new entry.
    pub fn record(&mut self, timecode: Timecode, level: LogLevel, message: impl Into<String>) {
        let entry = LogEntry::new(timecode, level, message);
        self.insert_sorted(entry);
    }

    /// Insert a pre-constructed entry in sorted order.
    pub fn insert(&mut self, entry: LogEntry) {
        self.insert_sorted(entry);
    }

    fn insert_sorted(&mut self, entry: LogEntry) {
        let pos = self
            .entries
            .partition_point(|e| e.timecode <= entry.timecode);
        self.entries.insert(pos, entry);
    }

    /// Return all entries matching the given filter.
    #[must_use]
    pub fn query(&self, filter: &LogFilter) -> Vec<&LogEntry> {
        self.entries.iter().filter(|e| filter.matches(e)).collect()
    }

    /// Return all entries.
    #[must_use]
    pub fn all_entries(&self) -> &[LogEntry] {
        &self.entries
    }

    /// Number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the log is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Remove all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Export as plain-text log.
    #[must_use]
    pub fn to_text(&self) -> String {
        let mut out = format!("# Timecode Log: {}\n", self.name);
        out.push_str(&format!("# Entries: {}\n\n", self.entries.len()));
        for entry in &self.entries {
            out.push_str(&format!("{}\n", entry.format_line()));
        }
        out
    }

    /// Export as CSV (timecode, level, category, message).
    #[must_use]
    pub fn to_csv(&self) -> String {
        let mut out = String::from("timecode,level,category,message\n");
        for entry in &self.entries {
            let cat = entry.category.as_deref().unwrap_or("");
            // Escape double quotes in message
            let msg = entry.message.replace('"', "\"\"");
            out.push_str(&format!(
                "{},{},{},\"{}\"\n",
                entry.timecode, entry.level, cat, msg
            ));
        }
        out
    }

    /// Find the first entry at or after the given timecode.
    #[must_use]
    pub fn first_at_or_after(&self, tc: &Timecode) -> Option<&LogEntry> {
        let pos = self.entries.partition_point(|e| &e.timecode < tc);
        self.entries.get(pos)
    }

    /// Find all entries within a timecode range (inclusive).
    #[must_use]
    pub fn entries_in_range(&self, start: &Timecode, end: &Timecode) -> Vec<&LogEntry> {
        self.entries
            .iter()
            .filter(|e| &e.timecode >= start && &e.timecode <= end)
            .collect()
    }

    /// Parse from CSV text (inverse of `to_csv`).
    ///
    /// # Errors
    ///
    /// Returns error if a line has invalid timecode format.
    pub fn from_csv(
        name: impl Into<String>,
        csv: &str,
        frame_rate: FrameRate,
    ) -> Result<Self, TimecodeError> {
        let mut log = Self::new(name);
        for (line_num, line) in csv.lines().enumerate() {
            if line_num == 0 || line.trim().is_empty() {
                continue; // skip header
            }
            let parts: Vec<&str> = line.splitn(4, ',').collect();
            if parts.len() < 4 {
                continue;
            }
            let tc_str = parts[0].trim();
            let level_str = parts[1].trim();
            let cat_str = parts[2].trim();
            let msg = parts[3].trim().trim_matches('"').replace("\"\"", "\"");

            let tc = Timecode::from_string(tc_str, frame_rate)?;
            let level = match level_str {
                "DEBUG" => LogLevel::Debug,
                "INFO" => LogLevel::Info,
                "WARN" => LogLevel::Warning,
                "ERROR" => LogLevel::Error,
                "CRITICAL" => LogLevel::Critical,
                _ => LogLevel::Info,
            };
            let mut entry = LogEntry::new(tc, level, msg);
            if !cat_str.is_empty() {
                entry.category = Some(cat_str.to_string());
            }
            log.insert_sorted(entry);
        }
        Ok(log)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FrameRate;

    fn tc(h: u8, m: u8, s: u8, f: u8) -> Timecode {
        Timecode::new(h, m, s, f, FrameRate::Fps25).expect("valid tc")
    }

    #[test]
    fn test_record_and_query() {
        let mut log = TimecodeLog::new("Test");
        log.record(tc(1, 0, 0, 0), LogLevel::Info, "A");
        log.record(tc(0, 0, 0, 0), LogLevel::Warning, "B");
        assert_eq!(log.len(), 2);
        // Entries should be sorted
        assert_eq!(log.all_entries()[0].timecode, tc(0, 0, 0, 0));
    }

    #[test]
    fn test_filter_by_level() {
        let mut log = TimecodeLog::new("Test");
        log.record(tc(0, 0, 1, 0), LogLevel::Debug, "debug");
        log.record(tc(0, 0, 2, 0), LogLevel::Warning, "warn");
        let filter = LogFilter::all().with_min_level(LogLevel::Warning);
        let results = log.query(&filter);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].level, LogLevel::Warning);
    }

    #[test]
    fn test_csv_round_trip() {
        let mut log = TimecodeLog::new("Trip");
        log.record(tc(1, 2, 3, 4), LogLevel::Info, "hello");
        let csv = log.to_csv();
        let log2 = TimecodeLog::from_csv("Trip", &csv, FrameRate::Fps25).expect("csv parse ok");
        assert_eq!(log2.len(), 1);
        assert_eq!(log2.all_entries()[0].message, "hello");
    }

    #[test]
    fn test_to_text_contains_header() {
        let log = TimecodeLog::new("MyProd");
        let text = log.to_text();
        assert!(text.contains("MyProd"));
    }

    #[test]
    fn test_entries_in_range() {
        let mut log = TimecodeLog::new("Range");
        log.record(tc(0, 0, 1, 0), LogLevel::Info, "in");
        log.record(tc(0, 0, 2, 0), LogLevel::Info, "also-in");
        log.record(tc(0, 0, 5, 0), LogLevel::Info, "out");
        let results = log.entries_in_range(&tc(0, 0, 0, 0), &tc(0, 0, 3, 0));
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_filter_by_category() {
        let mut log = TimecodeLog::new("Cat");
        let e1 = LogEntry::new(tc(0, 0, 0, 0), LogLevel::Info, "a").with_category("QC");
        let e2 = LogEntry::new(tc(0, 0, 1, 0), LogLevel::Info, "b").with_category("EDITORIAL");
        log.insert(e1);
        log.insert(e2);
        let filter = LogFilter::all().with_category("QC");
        let results = log.query(&filter);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].message, "a");
    }

    #[test]
    fn test_filter_by_message() {
        let mut log = TimecodeLog::new("Msg");
        log.record(tc(0, 0, 0, 0), LogLevel::Info, "scene start");
        log.record(tc(0, 0, 1, 0), LogLevel::Info, "cut here");
        let filter = LogFilter::all().with_message("scene");
        let results = log.query(&filter);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_first_at_or_after() {
        let mut log = TimecodeLog::new("First");
        log.record(tc(0, 0, 1, 0), LogLevel::Info, "first");
        log.record(tc(0, 0, 5, 0), LogLevel::Info, "second");
        let found = log.first_at_or_after(&tc(0, 0, 3, 0));
        assert!(found.is_some());
        assert_eq!(found.map(|e| e.message.as_str()), Some("second"));
    }

    #[test]
    fn test_log_entry_format_line() {
        let e = LogEntry::new(tc(1, 0, 0, 0), LogLevel::Error, "bad frame").with_category("QC");
        let line = e.format_line();
        assert!(line.contains("01:00:00:00"));
        assert!(line.contains("[QC]"));
        assert!(line.contains("ERROR"));
        assert!(line.contains("bad frame"));
    }

    #[test]
    fn test_log_clear() {
        let mut log = TimecodeLog::new("Clear");
        log.record(tc(0, 0, 0, 0), LogLevel::Info, "hello");
        assert!(!log.is_empty());
        log.clear();
        assert!(log.is_empty());
        assert_eq!(log.len(), 0);
    }

    #[test]
    fn test_log_entry_with_metadata() {
        let e = LogEntry::new(tc(0, 0, 0, 0), LogLevel::Info, "take 1")
            .with_meta("camera", "A")
            .with_meta("lens", "50mm");
        assert_eq!(e.metadata.len(), 2);
        assert_eq!(e.metadata.get("camera").map(String::as_str), Some("A"));
    }

    #[test]
    fn test_log_entry_wall_clock() {
        let e = LogEntry::new(tc(0, 0, 0, 0), LogLevel::Info, "x").with_wall_clock(1_700_000_000);
        assert_eq!(e.wall_clock_secs, Some(1_700_000_000));
    }
}
