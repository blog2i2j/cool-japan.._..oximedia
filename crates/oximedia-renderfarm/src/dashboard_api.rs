// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Lightweight web-dashboard API endpoint handler.
//!
//! Provides the data types and request router needed to power a farm
//! monitoring dashboard without pulling in a full HTTP framework at this
//! layer.  Callers forward the raw path string to [`handle_api_request`],
//! which returns an `(HTTP status code, JSON body)` tuple.
//!
//! # Example
//!
//! ```rust
//! use oximedia_renderfarm::dashboard_api::{DashboardStats, handle_api_request};
//!
//! let stats = DashboardStats {
//!     total_jobs: 100,
//!     running_jobs: 5,
//!     queued_jobs: 20,
//!     failed_jobs: 3,
//!     workers_online: 8,
//!     avg_throughput_jobs_per_hour: 12.5,
//! };
//!
//! let (status, body) = handle_api_request("/api/stats", &stats);
//! assert_eq!(status, 200);
//! assert!(body.contains("total_jobs"));
//! ```

// ---------------------------------------------------------------------------
// DashboardStats
// ---------------------------------------------------------------------------

/// Aggregate statistics exposed by the `/api/stats` endpoint.
#[derive(Debug, Clone)]
pub struct DashboardStats {
    /// Total jobs ever submitted (all states).
    pub total_jobs: usize,
    /// Jobs currently being rendered.
    pub running_jobs: usize,
    /// Jobs waiting in the queue.
    pub queued_jobs: usize,
    /// Jobs that terminated with an error.
    pub failed_jobs: usize,
    /// Workers currently reachable and accepting work.
    pub workers_online: usize,
    /// Exponentially-smoothed throughput in jobs completed per hour.
    pub avg_throughput_jobs_per_hour: f32,
}

impl DashboardStats {
    /// Serialize to a compact JSON object string.
    ///
    /// The output is hand-written to avoid a `serde_json` dependency at this
    /// level (though `serde_json` is available in the workspace if needed).
    #[must_use]
    pub fn to_json(&self) -> String {
        format!(
            "{{\
\"total_jobs\":{total},\
\"running_jobs\":{running},\
\"queued_jobs\":{queued},\
\"failed_jobs\":{failed},\
\"workers_online\":{workers},\
\"avg_throughput_jobs_per_hour\":{throughput:.4}\
}}",
            total = self.total_jobs,
            running = self.running_jobs,
            queued = self.queued_jobs,
            failed = self.failed_jobs,
            workers = self.workers_online,
            throughput = self.avg_throughput_jobs_per_hour,
        )
    }

    /// Compute the number of jobs that have completed successfully.
    #[must_use]
    pub fn completed_jobs(&self) -> usize {
        self.total_jobs
            .saturating_sub(self.running_jobs)
            .saturating_sub(self.queued_jobs)
            .saturating_sub(self.failed_jobs)
    }
}

// ---------------------------------------------------------------------------
// WorkerStatus
// ---------------------------------------------------------------------------

/// Per-worker status record exposed by the `/api/workers` endpoint.
#[derive(Debug, Clone)]
pub struct WorkerStatus {
    /// Unique worker identifier.
    pub id: String,
    /// Hostname or IP of the worker node.
    pub hostname: String,
    /// Human-readable status string (e.g. `"idle"`, `"busy"`, `"offline"`).
    pub status: String,
    /// ID of the job the worker is currently processing, if any.
    pub current_job: Option<String>,
    /// Number of jobs this worker has completed since startup.
    pub jobs_completed: u64,
}

impl WorkerStatus {
    /// Serialize a single worker record to a JSON object string.
    #[must_use]
    pub fn to_json(&self) -> String {
        let current_job_json = match &self.current_job {
            Some(job) => format!("\"{}\"", escape_json_string(job)),
            None => "null".to_owned(),
        };
        format!(
            "{{\
\"id\":\"{id}\",\
\"hostname\":\"{hostname}\",\
\"status\":\"{status}\",\
\"current_job\":{current_job},\
\"jobs_completed\":{completed}\
}}",
            id = escape_json_string(&self.id),
            hostname = escape_json_string(&self.hostname),
            status = escape_json_string(&self.status),
            current_job = current_job_json,
            completed = self.jobs_completed,
        )
    }
}

// ---------------------------------------------------------------------------
// format_worker_list
// ---------------------------------------------------------------------------

/// Serialize a slice of [`WorkerStatus`] records to a JSON array string.
///
/// Returns `"[]"` for an empty slice.
#[must_use]
pub fn format_worker_list(workers: &[WorkerStatus]) -> String {
    if workers.is_empty() {
        return "[]".to_owned();
    }
    let items: Vec<String> = workers.iter().map(WorkerStatus::to_json).collect();
    format!("[{}]", items.join(","))
}

// ---------------------------------------------------------------------------
// handle_api_request
// ---------------------------------------------------------------------------

/// Route an incoming HTTP-like request and return `(status_code, body)`.
///
/// Recognised paths:
/// - `/api/stats`   → 200 + [`DashboardStats::to_json`]
/// - `/api/workers` → 200 + `"[]"` (empty worker array — callers should
///   integrate live worker data as needed)
/// - anything else → 404 + a JSON error body
pub fn handle_api_request(path: &str, stats: &DashboardStats) -> (u16, String) {
    match path {
        "/api/stats" => (200, stats.to_json()),
        "/api/workers" => (200, "[]".to_owned()),
        _ => (
            404,
            format!(
                "{{\"error\":\"not found\",\"path\":\"{}\"}}",
                escape_json_string(path)
            ),
        ),
    }
}

/// Variant that also accepts a live worker list for the `/api/workers` path.
pub fn handle_api_request_with_workers(
    path: &str,
    stats: &DashboardStats,
    workers: &[WorkerStatus],
) -> (u16, String) {
    match path {
        "/api/stats" => (200, stats.to_json()),
        "/api/workers" => (200, format_worker_list(workers)),
        _ => (
            404,
            format!(
                "{{\"error\":\"not found\",\"path\":\"{}\"}}",
                escape_json_string(path)
            ),
        ),
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Escape a string for safe embedding inside a JSON string literal.
///
/// Only escapes the characters required by RFC 8259: `"` → `\"`,
/// `\` → `\\`, and control characters → `\uXXXX`.
fn escape_json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_stats() -> DashboardStats {
        DashboardStats {
            total_jobs: 50,
            running_jobs: 5,
            queued_jobs: 10,
            failed_jobs: 2,
            workers_online: 4,
            avg_throughput_jobs_per_hour: 6.25,
        }
    }

    // --- DashboardStats::to_json ---

    #[test]
    fn stats_to_json_contains_all_fields() {
        let json = sample_stats().to_json();
        assert!(json.contains("\"total_jobs\":50"));
        assert!(json.contains("\"running_jobs\":5"));
        assert!(json.contains("\"queued_jobs\":10"));
        assert!(json.contains("\"failed_jobs\":2"));
        assert!(json.contains("\"workers_online\":4"));
        assert!(json.contains("avg_throughput_jobs_per_hour"));
    }

    #[test]
    fn stats_to_json_is_valid_json_structure() {
        let json = sample_stats().to_json();
        assert!(json.starts_with('{'));
        assert!(json.ends_with('}'));
    }

    #[test]
    fn stats_completed_jobs_calculation() {
        let s = sample_stats();
        // 50 - 5 (running) - 10 (queued) - 2 (failed) = 33
        assert_eq!(s.completed_jobs(), 33);
    }

    #[test]
    fn stats_completed_does_not_underflow() {
        let s = DashboardStats {
            total_jobs: 0,
            running_jobs: 5,
            queued_jobs: 0,
            failed_jobs: 0,
            workers_online: 1,
            avg_throughput_jobs_per_hour: 0.0,
        };
        // saturating_sub prevents underflow
        assert_eq!(s.completed_jobs(), 0);
    }

    // --- WorkerStatus::to_json ---

    #[test]
    fn worker_to_json_with_current_job() {
        let w = WorkerStatus {
            id: "w-1".to_owned(),
            hostname: "node01.local".to_owned(),
            status: "busy".to_owned(),
            current_job: Some("job-42".to_owned()),
            jobs_completed: 17,
        };
        let json = w.to_json();
        assert!(json.contains("\"id\":\"w-1\""));
        assert!(json.contains("\"current_job\":\"job-42\""));
        assert!(json.contains("\"jobs_completed\":17"));
    }

    #[test]
    fn worker_to_json_without_current_job() {
        let w = WorkerStatus {
            id: "w-2".to_owned(),
            hostname: "node02.local".to_owned(),
            status: "idle".to_owned(),
            current_job: None,
            jobs_completed: 0,
        };
        let json = w.to_json();
        assert!(json.contains("\"current_job\":null"));
    }

    #[test]
    fn worker_to_json_escapes_special_chars() {
        let w = WorkerStatus {
            id: "w-\"quote\"".to_owned(),
            hostname: "host".to_owned(),
            status: "idle".to_owned(),
            current_job: None,
            jobs_completed: 0,
        };
        let json = w.to_json();
        // The double-quotes inside the id should be escaped.
        assert!(json.contains("\\\"quote\\\""));
    }

    // --- format_worker_list ---

    #[test]
    fn format_empty_worker_list() {
        assert_eq!(format_worker_list(&[]), "[]");
    }

    #[test]
    fn format_single_worker() {
        let workers = vec![WorkerStatus {
            id: "w-1".to_owned(),
            hostname: "h1".to_owned(),
            status: "idle".to_owned(),
            current_job: None,
            jobs_completed: 0,
        }];
        let json = format_worker_list(&workers);
        assert!(json.starts_with('['));
        assert!(json.ends_with(']'));
        assert!(json.contains("\"w-1\""));
    }

    #[test]
    fn format_multiple_workers_comma_separated() {
        let workers = vec![
            WorkerStatus {
                id: "w-1".to_owned(),
                hostname: "h1".to_owned(),
                status: "idle".to_owned(),
                current_job: None,
                jobs_completed: 0,
            },
            WorkerStatus {
                id: "w-2".to_owned(),
                hostname: "h2".to_owned(),
                status: "busy".to_owned(),
                current_job: Some("j-7".to_owned()),
                jobs_completed: 5,
            },
        ];
        let json = format_worker_list(&workers);
        assert!(json.contains("\"w-1\""));
        assert!(json.contains("\"w-2\""));
    }

    // --- handle_api_request ---

    #[test]
    fn route_api_stats_returns_200() {
        let stats = sample_stats();
        let (status, body) = handle_api_request("/api/stats", &stats);
        assert_eq!(status, 200);
        assert!(body.contains("total_jobs"));
    }

    #[test]
    fn route_api_workers_returns_200_empty_array() {
        let stats = sample_stats();
        let (status, body) = handle_api_request("/api/workers", &stats);
        assert_eq!(status, 200);
        assert_eq!(body, "[]");
    }

    #[test]
    fn route_unknown_path_returns_404() {
        let stats = sample_stats();
        let (status, body) = handle_api_request("/api/unknown", &stats);
        assert_eq!(status, 404);
        assert!(body.contains("not found"));
    }

    #[test]
    fn route_root_returns_404() {
        let stats = sample_stats();
        let (status, _) = handle_api_request("/", &stats);
        assert_eq!(status, 404);
    }

    // --- handle_api_request_with_workers ---

    #[test]
    fn route_with_workers_returns_worker_list() {
        let stats = sample_stats();
        let workers = vec![WorkerStatus {
            id: "w-3".to_owned(),
            hostname: "h3".to_owned(),
            status: "busy".to_owned(),
            current_job: Some("j-1".to_owned()),
            jobs_completed: 2,
        }];
        let (status, body) = handle_api_request_with_workers("/api/workers", &stats, &workers);
        assert_eq!(status, 200);
        assert!(body.contains("\"w-3\""));
    }

    // --- escape_json_string ---

    #[test]
    fn escape_plain_string() {
        assert_eq!(escape_json_string("hello"), "hello");
    }

    #[test]
    fn escape_double_quote() {
        assert_eq!(escape_json_string("say \"hi\""), "say \\\"hi\\\"");
    }

    #[test]
    fn escape_newline() {
        assert_eq!(escape_json_string("line\nnewline"), "line\\nnewline");
    }
}
