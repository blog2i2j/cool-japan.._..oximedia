//! Structured HTTP request logger.
//!
//! Formats HTTP request metadata as a JSON-Lines record suitable for
//! ingestion into log aggregators (Elasticsearch, Loki, Datadog, etc.).
//! Each call to [`RequestLogger::log`](crate::req_log::RequestLogger::log) returns a single-line JSON string and
//! optionally stores it in an in-memory ring buffer for inspection.
//!
//! # Example
//!
//! ```rust
//! use oximedia_server::req_log::RequestLogger;
//!
//! let mut logger = RequestLogger::new(1000);
//! let line = logger.log("GET", "/api/media", 200, 42);
//! assert!(line.contains("\"method\":\"GET\""));
//! assert!(line.contains("\"status\":200"));
//! ```

#![allow(dead_code)]

use std::collections::VecDeque;

// ── RequestLogger ─────────────────────────────────────────────────────────────

/// Single HTTP request log entry.
#[derive(Debug, Clone)]
pub struct LogEntry {
    /// HTTP method (e.g. `"GET"`).
    pub method: String,
    /// Request path (e.g. `"/api/media/123"`).
    pub path: String,
    /// HTTP status code (e.g. `200`).
    pub status: u16,
    /// Response duration in milliseconds.
    pub dur_ms: u64,
    /// JSON-serialized representation.
    pub json: String,
}

/// In-memory request logger with JSON-Lines output.
///
/// Keeps the last `capacity` log entries in a bounded ring buffer.
pub struct RequestLogger {
    /// Ring buffer of recent entries.
    entries: VecDeque<LogEntry>,
    /// Maximum number of entries to retain.
    capacity: usize,
    /// Total number of requests logged (monotonic counter).
    pub total_logged: u64,
    /// Count of requests that resulted in HTTP 4xx or 5xx.
    pub error_count: u64,
}

impl RequestLogger {
    /// Create a new logger with `capacity` entries in the ring buffer.
    ///
    /// When the buffer is full, the oldest entry is discarded on each new log.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: VecDeque::with_capacity(capacity.min(65_536)),
            capacity: capacity.max(1),
            total_logged: 0,
            error_count: 0,
        }
    }

    /// Log a single HTTP request and return the JSON-Lines string.
    ///
    /// # Arguments
    ///
    /// * `method` – HTTP verb (e.g. `"GET"`, `"POST"`).
    /// * `path`   – Request URI path.
    /// * `status` – HTTP response status code.
    /// * `dur_ms` – Round-trip duration in milliseconds.
    #[must_use]
    pub fn log(&mut self, method: &str, path: &str, status: u16, dur_ms: u64) -> String {
        let json = Self::format_json(method, path, status, dur_ms);
        let entry = LogEntry {
            method: method.to_string(),
            path: path.to_string(),
            status,
            dur_ms,
            json: json.clone(),
        };

        self.total_logged += 1;
        if status >= 400 {
            self.error_count += 1;
        }

        if self.entries.len() >= self.capacity {
            self.entries.pop_front();
        }
        self.entries.push_back(entry);
        json
    }

    /// Format a JSON-Lines record without storing it.
    #[must_use]
    pub fn format_json(method: &str, path: &str, status: u16, dur_ms: u64) -> String {
        // Hand-rolled minimal JSON to avoid pulling in serde.
        // Field escaping: replace `"` with `\"` in path/method.
        let esc_method = method.replace('"', "\\\"");
        let esc_path = path.replace('"', "\\\"");
        format!(
            r#"{{"method":"{esc_method}","path":"{esc_path}","status":{status},"dur_ms":{dur_ms}}}"#
        )
    }

    /// Return the most recent `n` log entries (oldest first).
    #[must_use]
    pub fn recent(&self, n: usize) -> Vec<&LogEntry> {
        let skip = self.entries.len().saturating_sub(n);
        self.entries.iter().skip(skip).collect()
    }

    /// Return all entries where `status >= 400`.
    #[must_use]
    pub fn errors(&self) -> Vec<&LogEntry> {
        self.entries.iter().filter(|e| e.status >= 400).collect()
    }

    /// Return all entries where `dur_ms > threshold`.
    #[must_use]
    pub fn slow_requests(&self, threshold_ms: u64) -> Vec<&LogEntry> {
        self.entries
            .iter()
            .filter(|e| e.dur_ms > threshold_ms)
            .collect()
    }

    /// Number of entries currently in the ring buffer.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// `true` if the ring buffer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Clear all buffered entries (counters are preserved).
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_returns_json_line() {
        let mut logger = RequestLogger::new(100);
        let line = logger.log("GET", "/api/media", 200, 42);
        assert!(line.contains("\"method\":\"GET\""), "line={line}");
        assert!(line.contains("\"path\":\"/api/media\""), "line={line}");
        assert!(line.contains("\"status\":200"), "line={line}");
        assert!(line.contains("\"dur_ms\":42"), "line={line}");
    }

    #[test]
    fn test_log_stored_in_buffer() {
        let mut logger = RequestLogger::new(100);
        logger.log("POST", "/upload", 201, 100);
        assert_eq!(logger.len(), 1);
    }

    #[test]
    fn test_ring_buffer_evicts_oldest() {
        let mut logger = RequestLogger::new(3);
        logger.log("GET", "/a", 200, 1);
        logger.log("GET", "/b", 200, 2);
        logger.log("GET", "/c", 200, 3);
        logger.log("GET", "/d", 200, 4); // evicts /a
        assert_eq!(logger.len(), 3);
        // /a should be gone
        let paths: Vec<&str> = logger.entries.iter().map(|e| e.path.as_str()).collect();
        assert!(
            !paths.contains(&"/a"),
            "oldest should be evicted: {:?}",
            paths
        );
        assert!(paths.contains(&"/d"));
    }

    #[test]
    fn test_total_logged_counter() {
        let mut logger = RequestLogger::new(10);
        for i in 0..5 {
            logger.log("GET", &format!("/{i}"), 200, 10);
        }
        assert_eq!(logger.total_logged, 5);
    }

    #[test]
    fn test_error_count_tracks_4xx_5xx() {
        let mut logger = RequestLogger::new(100);
        logger.log("GET", "/", 200, 1);
        logger.log("GET", "/missing", 404, 1);
        logger.log("POST", "/crash", 500, 1);
        assert_eq!(logger.error_count, 2);
    }

    #[test]
    fn test_errors_returns_only_error_entries() {
        let mut logger = RequestLogger::new(100);
        logger.log("GET", "/ok", 200, 1);
        logger.log("GET", "/bad", 400, 1);
        let errors = logger.errors();
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].status, 400);
    }

    #[test]
    fn test_slow_requests_filter() {
        let mut logger = RequestLogger::new(100);
        logger.log("GET", "/fast", 200, 10);
        logger.log("GET", "/slow", 200, 1000);
        let slow = logger.slow_requests(500);
        assert_eq!(slow.len(), 1);
        assert_eq!(slow[0].path, "/slow");
    }

    #[test]
    fn test_recent_returns_last_n() {
        let mut logger = RequestLogger::new(100);
        for i in 0..10 {
            logger.log("GET", &format!("/{i}"), 200, i as u64);
        }
        let recent = logger.recent(3);
        assert_eq!(recent.len(), 3);
        assert_eq!(recent[0].path, "/7");
        assert_eq!(recent[2].path, "/9");
    }

    #[test]
    fn test_clear_removes_entries() {
        let mut logger = RequestLogger::new(100);
        logger.log("DELETE", "/all", 200, 5);
        logger.clear();
        assert!(logger.is_empty());
    }

    #[test]
    fn test_json_escapes_quotes_in_path() {
        let json = RequestLogger::format_json("GET", "/path/\"quoted\"", 200, 0);
        assert!(json.contains("\\\"quoted\\\""), "json={json}");
    }
}
