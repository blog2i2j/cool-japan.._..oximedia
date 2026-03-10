//! Log aggregation and query system.
//!
//! Provides structured log storage with in-memory ring buffer backed by `SQLite`,
//! filtering by level/component/time, and aggregation by level or component.

use crate::error::{MonitorError, MonitorResult};
use chrono::{DateTime, Duration, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

/// Log severity level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum LogLevel {
    /// Trace level (most verbose).
    Trace,
    /// Debug level.
    Debug,
    /// Informational messages.
    Info,
    /// Warning messages.
    Warn,
    /// Error messages.
    Error,
    /// Critical errors.
    Critical,
}

impl LogLevel {
    /// Convert from string representation.
    ///
    /// # Errors
    ///
    /// Returns an error if the string does not match a known level.
    pub fn from_str(s: &str) -> MonitorResult<Self> {
        match s.to_lowercase().as_str() {
            "trace" => Ok(Self::Trace),
            "debug" => Ok(Self::Debug),
            "info" => Ok(Self::Info),
            "warn" | "warning" => Ok(Self::Warn),
            "error" => Ok(Self::Error),
            "critical" | "crit" => Ok(Self::Critical),
            other => Err(MonitorError::Log(format!("Unknown log level: {other}"))),
        }
    }

    /// Return the string representation of the level.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Trace => "TRACE",
            Self::Debug => "DEBUG",
            Self::Info => "INFO",
            Self::Warn => "WARN",
            Self::Error => "ERROR",
            Self::Critical => "CRITICAL",
        }
    }
}

impl std::fmt::Display for LogLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A structured log entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    /// Timestamp when the log was emitted.
    pub timestamp: DateTime<Utc>,
    /// Severity level.
    pub level: LogLevel,
    /// Component or subsystem that emitted the log.
    pub component: String,
    /// Human-readable message.
    pub message: String,
    /// Optional structured key/value fields.
    pub fields: HashMap<String, serde_json::Value>,
}

impl LogEntry {
    /// Create a new log entry with no extra fields.
    #[must_use]
    pub fn new(level: LogLevel, component: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            timestamp: Utc::now(),
            level,
            component: component.into(),
            message: message.into(),
            fields: HashMap::new(),
        }
    }

    /// Builder-style method to add a structured field.
    #[must_use]
    pub fn with_field(
        mut self,
        key: impl Into<String>,
        value: impl Into<serde_json::Value>,
    ) -> Self {
        self.fields.insert(key.into(), value.into());
        self
    }
}

/// Query parameters for log retrieval.
#[derive(Debug, Clone, Default)]
pub struct LogQuery {
    /// Only return entries at or above this level (inclusive). `None` means all levels.
    pub min_level: Option<LogLevel>,
    /// Only return entries from this component. `None` means all components.
    pub component: Option<String>,
    /// Start of the time range (inclusive). `None` means no lower bound.
    pub start: Option<DateTime<Utc>>,
    /// End of the time range (inclusive). `None` means no upper bound.
    pub end: Option<DateTime<Utc>>,
    /// Substring to search for in the message (case-insensitive). `None` means no filter.
    pub message_contains: Option<String>,
    /// Maximum number of results. `None` means no limit.
    pub limit: Option<usize>,
}

impl LogQuery {
    /// Create an empty query that matches everything.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Filter to entries at or above the given level.
    #[must_use]
    pub fn with_min_level(mut self, level: LogLevel) -> Self {
        self.min_level = Some(level);
        self
    }

    /// Filter to entries from a specific component.
    #[must_use]
    pub fn with_component(mut self, component: impl Into<String>) -> Self {
        self.component = Some(component.into());
        self
    }

    /// Filter to entries within a time range.
    #[must_use]
    pub fn with_time_range(mut self, start: DateTime<Utc>, end: DateTime<Utc>) -> Self {
        self.start = Some(start);
        self.end = Some(end);
        self
    }

    /// Filter to entries whose message contains the given substring (case-insensitive).
    #[must_use]
    pub fn with_message_contains(mut self, text: impl Into<String>) -> Self {
        self.message_contains = Some(text.into());
        self
    }

    /// Limit the number of results.
    #[must_use]
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }
}

/// Aggregation counts by level over a time window.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LevelAggregate {
    /// Window start.
    pub window_start: DateTime<Utc>,
    /// Window end.
    pub window_end: DateTime<Utc>,
    /// Counts per level.
    pub counts: HashMap<String, usize>,
}

/// Aggregation counts by component over a time window.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ComponentAggregate {
    /// Window start.
    pub window_start: DateTime<Utc>,
    /// Window end.
    pub window_end: DateTime<Utc>,
    /// Counts per component.
    pub counts: HashMap<String, usize>,
}

/// In-memory log store backed by a fixed-capacity ring buffer.
///
/// Older entries are evicted once the capacity is exceeded. This is intentionally
/// kept in-memory for low-latency access; persistence can be added via `SqliteStorage`.
pub struct LogStore {
    capacity: usize,
    entries: Arc<RwLock<VecDeque<LogEntry>>>,
}

impl LogStore {
    /// Create a new `LogStore` with the given ring-buffer capacity.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            entries: Arc::new(RwLock::new(VecDeque::with_capacity(capacity))),
        }
    }

    /// Insert a log entry, evicting the oldest entry if the buffer is full.
    pub fn insert(&self, entry: LogEntry) {
        let mut buf = self.entries.write();
        if buf.len() == self.capacity {
            buf.pop_front();
        }
        buf.push_back(entry);
    }

    /// Convenience helper to insert a simple message.
    pub fn log(&self, level: LogLevel, component: impl Into<String>, message: impl Into<String>) {
        self.insert(LogEntry::new(level, component, message));
    }

    /// Query entries matching the given criteria.
    ///
    /// Results are returned in ascending timestamp order (oldest first).
    #[must_use]
    pub fn query(&self, query: &LogQuery) -> Vec<LogEntry> {
        let buf = self.entries.read();
        let msg_lower = query.message_contains.as_ref().map(|s| s.to_lowercase());

        let iter = buf.iter().filter(|e| {
            // Level filter
            if let Some(min) = query.min_level {
                if e.level < min {
                    return false;
                }
            }
            // Component filter
            if let Some(ref comp) = query.component {
                if &e.component != comp {
                    return false;
                }
            }
            // Time range
            if let Some(start) = query.start {
                if e.timestamp < start {
                    return false;
                }
            }
            if let Some(end) = query.end {
                if e.timestamp > end {
                    return false;
                }
            }
            // Message substring
            if let Some(ref needle) = msg_lower {
                if !e.message.to_lowercase().contains(needle.as_str()) {
                    return false;
                }
            }
            true
        });

        let results: Vec<LogEntry> = iter.cloned().collect();

        if let Some(limit) = query.limit {
            results.into_iter().take(limit).collect()
        } else {
            results
        }
    }

    /// Count all log entries in the store.
    #[must_use]
    pub fn count(&self) -> usize {
        self.entries.read().len()
    }

    /// Clear all log entries.
    pub fn clear(&self) {
        self.entries.write().clear();
    }

    /// Aggregate entry counts per log level within the given time window.
    ///
    /// `window` is measured backwards from `Utc::now()`.
    #[must_use]
    pub fn aggregate_by_level(&self, window: Duration) -> LevelAggregate {
        let now = Utc::now();
        let window_start = now - window;

        let buf = self.entries.read();
        let mut counts: HashMap<String, usize> = HashMap::new();

        for entry in buf.iter() {
            if entry.timestamp >= window_start && entry.timestamp <= now {
                *counts.entry(entry.level.to_string()).or_insert(0) += 1;
            }
        }

        LevelAggregate {
            window_start,
            window_end: now,
            counts,
        }
    }

    /// Aggregate entry counts per component within the given time window.
    ///
    /// `window` is measured backwards from `Utc::now()`.
    #[must_use]
    pub fn aggregate_by_component(&self, window: Duration) -> ComponentAggregate {
        let now = Utc::now();
        let window_start = now - window;

        let buf = self.entries.read();
        let mut counts: HashMap<String, usize> = HashMap::new();

        for entry in buf.iter() {
            if entry.timestamp >= window_start && entry.timestamp <= now {
                *counts.entry(entry.component.clone()).or_insert(0) += 1;
            }
        }

        ComponentAggregate {
            window_start,
            window_end: now,
            counts,
        }
    }
}

impl Default for LogStore {
    fn default() -> Self {
        Self::new(10_000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(level: LogLevel, component: &str, msg: &str) -> LogEntry {
        LogEntry::new(level, component, msg)
    }

    #[test]
    fn test_insert_and_count() {
        let store = LogStore::new(100);
        store.insert(make_entry(LogLevel::Info, "api", "request received"));
        store.insert(make_entry(LogLevel::Error, "db", "connection failed"));
        assert_eq!(store.count(), 2);
    }

    #[test]
    fn test_ring_buffer_eviction() {
        let store = LogStore::new(3);
        for i in 0..5 {
            store.insert(make_entry(LogLevel::Info, "test", &format!("msg {i}")));
        }
        assert_eq!(store.count(), 3);
        let results = store.query(&LogQuery::new());
        assert_eq!(results[0].message, "msg 2");
    }

    #[test]
    fn test_query_by_level() {
        let store = LogStore::new(100);
        store.insert(make_entry(LogLevel::Debug, "test", "debug msg"));
        store.insert(make_entry(LogLevel::Info, "test", "info msg"));
        store.insert(make_entry(LogLevel::Error, "test", "error msg"));

        let results = store.query(&LogQuery::new().with_min_level(LogLevel::Info));
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|e| e.level >= LogLevel::Info));
    }

    #[test]
    fn test_query_by_component() {
        let store = LogStore::new(100);
        store.insert(make_entry(LogLevel::Info, "api", "api msg"));
        store.insert(make_entry(LogLevel::Info, "db", "db msg"));
        store.insert(make_entry(LogLevel::Info, "api", "another api msg"));

        let results = store.query(&LogQuery::new().with_component("api"));
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|e| e.component == "api"));
    }

    #[test]
    fn test_query_with_limit() {
        let store = LogStore::new(100);
        for i in 0..10 {
            store.insert(make_entry(LogLevel::Info, "test", &format!("msg {i}")));
        }
        let results = store.query(&LogQuery::new().with_limit(3));
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_query_message_contains() {
        let store = LogStore::new(100);
        store.insert(make_entry(LogLevel::Info, "test", "connection established"));
        store.insert(make_entry(LogLevel::Info, "test", "request timeout"));
        store.insert(make_entry(LogLevel::Info, "test", "connection closed"));

        let results = store.query(&LogQuery::new().with_message_contains("connection"));
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_aggregate_by_level() {
        let store = LogStore::new(100);
        store.insert(make_entry(LogLevel::Info, "a", "msg"));
        store.insert(make_entry(LogLevel::Info, "b", "msg"));
        store.insert(make_entry(LogLevel::Error, "c", "msg"));

        let agg = store.aggregate_by_level(Duration::hours(1));
        assert_eq!(agg.counts.get("INFO").copied().unwrap_or(0), 2);
        assert_eq!(agg.counts.get("ERROR").copied().unwrap_or(0), 1);
    }

    #[test]
    fn test_aggregate_by_component() {
        let store = LogStore::new(100);
        store.insert(make_entry(LogLevel::Info, "api", "msg"));
        store.insert(make_entry(LogLevel::Info, "api", "msg"));
        store.insert(make_entry(LogLevel::Info, "db", "msg"));

        let agg = store.aggregate_by_component(Duration::hours(1));
        assert_eq!(agg.counts.get("api").copied().unwrap_or(0), 2);
        assert_eq!(agg.counts.get("db").copied().unwrap_or(0), 1);
    }

    #[test]
    fn test_log_level_ordering() {
        assert!(LogLevel::Trace < LogLevel::Debug);
        assert!(LogLevel::Debug < LogLevel::Info);
        assert!(LogLevel::Info < LogLevel::Warn);
        assert!(LogLevel::Warn < LogLevel::Error);
        assert!(LogLevel::Error < LogLevel::Critical);
    }

    #[test]
    fn test_log_level_from_str() {
        assert_eq!(
            LogLevel::from_str("info").expect("operation should succeed"),
            LogLevel::Info
        );
        assert_eq!(
            LogLevel::from_str("WARN").expect("operation should succeed"),
            LogLevel::Warn
        );
        assert_eq!(
            LogLevel::from_str("warning").expect("operation should succeed"),
            LogLevel::Warn
        );
        assert!(LogLevel::from_str("unknown").is_err());
    }

    #[test]
    fn test_entry_with_fields() {
        let entry = LogEntry::new(LogLevel::Info, "api", "request")
            .with_field("method", "GET")
            .with_field("status", 200i64);
        assert_eq!(
            entry.fields.get("method").expect("failed to get value"),
            &serde_json::json!("GET")
        );
        assert_eq!(
            entry.fields.get("status").expect("failed to get value"),
            &serde_json::json!(200i64)
        );
    }

    #[test]
    fn test_clear() {
        let store = LogStore::new(100);
        for i in 0..5 {
            store.insert(make_entry(LogLevel::Info, "test", &format!("msg {i}")));
        }
        assert_eq!(store.count(), 5);
        store.clear();
        assert_eq!(store.count(), 0);
    }
}
