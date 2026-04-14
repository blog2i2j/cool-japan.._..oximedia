//! Web dashboard API endpoints for monitoring farm status.
//!
//! Provides a lightweight HTTP/1.1 request handler that parses incoming
//! GET requests and returns JSON responses for the following endpoints:
//!
//! | Endpoint        | Description                          |
//! |-----------------|--------------------------------------|
//! | `GET /api/workers`  | Worker registry snapshot         |
//! | `GET /api/jobs`     | Current job queue state          |
//! | `GET /api/metrics`  | Aggregate farm performance metrics |
//!
//! The implementation uses no web framework — it hand-parses the HTTP/1.1
//! request line and writes a minimal HTTP/1.1 response. This keeps the
//! dependency surface minimal while remaining spec-compliant enough to be
//! consumed by any standard HTTP client.

#![allow(dead_code)]

use std::collections::HashMap;

use crate::{JobState, Priority, WorkerState};

// ---------------------------------------------------------------------------
// Request / Response types
// ---------------------------------------------------------------------------

/// A parsed HTTP/1.1 request (method + path only).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpRequest {
    /// HTTP method (e.g. `"GET"`).
    pub method: String,
    /// Request path (e.g. `"/api/workers"`).
    pub path: String,
    /// HTTP version string (e.g. `"HTTP/1.1"`).
    pub version: String,
    /// Request headers (lowercase key → value).
    pub headers: HashMap<String, String>,
}

impl HttpRequest {
    /// Parse the first line (and optional headers) of an HTTP/1.1 request.
    ///
    /// Returns `None` if the request line is malformed.
    #[must_use]
    pub fn parse(raw: &str) -> Option<Self> {
        let mut lines = raw.lines();
        let request_line = lines.next()?;
        let mut parts = request_line.splitn(3, ' ');
        let method_raw = parts.next()?;
        let path_raw = parts.next()?;
        // Validate that both method and path are non-empty.
        if method_raw.trim().is_empty() || path_raw.trim().is_empty() {
            return None;
        }
        let method = method_raw.to_string();
        let path = path_raw.to_string();
        let version = parts.next().unwrap_or("HTTP/1.1").to_string();

        let mut headers = HashMap::new();
        for line in lines {
            if line.is_empty() {
                break;
            }
            if let Some((k, v)) = line.split_once(':') {
                headers.insert(k.trim().to_lowercase(), v.trim().to_string());
            }
        }

        Some(Self {
            method,
            path,
            version,
            headers,
        })
    }
}

/// A minimal HTTP/1.1 response.
#[derive(Debug, Clone)]
pub struct HttpResponse {
    /// HTTP status code (e.g. `200`).
    pub status_code: u16,
    /// HTTP reason phrase (e.g. `"OK"`).
    pub reason: String,
    /// Response body.
    pub body: String,
    /// Content-Type header value.
    pub content_type: String,
}

impl HttpResponse {
    /// Create a `200 OK` JSON response.
    #[must_use]
    pub fn ok_json(body: impl Into<String>) -> Self {
        Self {
            status_code: 200,
            reason: "OK".to_string(),
            body: body.into(),
            content_type: "application/json".to_string(),
        }
    }

    /// Create a `404 Not Found` response.
    #[must_use]
    pub fn not_found(path: &str) -> Self {
        Self {
            status_code: 404,
            reason: "Not Found".to_string(),
            body: format!("{{\"error\":\"not found\",\"path\":\"{path}\"}}"),
            content_type: "application/json".to_string(),
        }
    }

    /// Create a `405 Method Not Allowed` response.
    #[must_use]
    pub fn method_not_allowed(method: &str) -> Self {
        Self {
            status_code: 405,
            reason: "Method Not Allowed".to_string(),
            body: format!("{{\"error\":\"method not allowed\",\"method\":\"{method}\"}}"),
            content_type: "application/json".to_string(),
        }
    }

    /// Serialize this response to a raw HTTP/1.1 byte string.
    #[must_use]
    pub fn to_http_bytes(&self) -> Vec<u8> {
        let content_length = self.body.len();
        let headers = format!(
            "HTTP/1.1 {} {}\r\nContent-Type: {}; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            self.status_code, self.reason, self.content_type, content_length
        );
        let mut out = headers.into_bytes();
        out.extend_from_slice(self.body.as_bytes());
        out
    }

    /// Return the raw HTTP response as a UTF-8 string (for testing / logging).
    ///
    /// # Errors
    ///
    /// Returns an error if the byte representation is not valid UTF-8 (which
    /// should not happen for JSON bodies, but the API is fallible for safety).
    pub fn to_http_string(&self) -> Result<String, std::string::FromUtf8Error> {
        String::from_utf8(self.to_http_bytes())
    }
}

// ---------------------------------------------------------------------------
// Data snapshots passed to the handler
// ---------------------------------------------------------------------------

/// A lightweight snapshot of a single worker's state.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WorkerSnapshot {
    /// Worker identifier.
    pub id: String,
    /// Current state.
    pub state: WorkerStateRepr,
    /// Number of tasks currently assigned.
    pub active_tasks: u32,
    /// Maximum concurrent tasks this worker can handle.
    pub capacity: u32,
    /// Worker address (host:port).
    pub address: String,
}

/// Serialisable representation of [`WorkerState`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WorkerStateRepr {
    Idle,
    Busy,
    Overloaded,
    Draining,
    Offline,
}

impl From<WorkerState> for WorkerStateRepr {
    fn from(s: WorkerState) -> Self {
        match s {
            WorkerState::Idle => Self::Idle,
            WorkerState::Busy => Self::Busy,
            WorkerState::Overloaded => Self::Overloaded,
            WorkerState::Draining => Self::Draining,
            WorkerState::Offline => Self::Offline,
        }
    }
}

/// A lightweight snapshot of a single job's state.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct JobSnapshot {
    /// Job identifier.
    pub id: String,
    /// Job name or description.
    pub name: String,
    /// Current state.
    pub state: JobStateRepr,
    /// Job priority.
    pub priority: PriorityRepr,
    /// Submission timestamp (unix seconds).
    pub submitted_at: u64,
    /// Completion percentage (0–100).
    pub progress_pct: f32,
}

/// Serialisable representation of [`JobState`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum JobStateRepr {
    Pending,
    Queued,
    Running,
    Completed,
    CompletedWithWarnings,
    Failed,
    Cancelled,
    Paused,
}

impl From<JobState> for JobStateRepr {
    fn from(s: JobState) -> Self {
        match s {
            JobState::Pending => Self::Pending,
            JobState::Queued => Self::Queued,
            JobState::Running => Self::Running,
            JobState::Completed => Self::Completed,
            JobState::CompletedWithWarnings => Self::CompletedWithWarnings,
            JobState::Failed => Self::Failed,
            JobState::Cancelled => Self::Cancelled,
            JobState::Paused => Self::Paused,
        }
    }
}

/// Serialisable representation of [`Priority`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PriorityRepr {
    Low,
    Normal,
    High,
    Critical,
}

impl From<Priority> for PriorityRepr {
    fn from(p: Priority) -> Self {
        match p {
            Priority::Low => Self::Low,
            Priority::Normal => Self::Normal,
            Priority::High => Self::High,
            Priority::Critical => Self::Critical,
        }
    }
}

/// Aggregate performance metrics for the farm.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FarmMetricsSnapshot {
    /// Total workers registered.
    pub total_workers: u32,
    /// Workers currently idle.
    pub idle_workers: u32,
    /// Workers currently busy.
    pub busy_workers: u32,
    /// Workers offline or draining.
    pub unavailable_workers: u32,
    /// Total jobs in the queue (all states).
    pub total_jobs: u32,
    /// Jobs currently pending or queued.
    pub pending_jobs: u32,
    /// Jobs currently running.
    pub running_jobs: u32,
    /// Jobs completed since last reset.
    pub completed_jobs: u32,
    /// Jobs failed since last reset.
    pub failed_jobs: u32,
    /// Average job progress percent across running jobs.
    pub avg_progress_pct: f32,
    /// Queue utilisation ratio (0.0–1.0).
    pub queue_utilisation: f32,
}

impl FarmMetricsSnapshot {
    /// Compute a metrics snapshot from worker and job snapshots.
    #[must_use]
    pub fn compute(workers: &[WorkerSnapshot], jobs: &[JobSnapshot]) -> Self {
        let total_workers = workers.len() as u32;
        let idle_workers = workers
            .iter()
            .filter(|w| matches!(w.state, WorkerStateRepr::Idle))
            .count() as u32;
        let busy_workers = workers
            .iter()
            .filter(|w| matches!(w.state, WorkerStateRepr::Busy))
            .count() as u32;
        let unavailable_workers = workers
            .iter()
            .filter(|w| {
                matches!(
                    w.state,
                    WorkerStateRepr::Offline | WorkerStateRepr::Draining
                )
            })
            .count() as u32;

        let total_jobs = jobs.len() as u32;
        let pending_jobs = jobs
            .iter()
            .filter(|j| matches!(j.state, JobStateRepr::Pending | JobStateRepr::Queued))
            .count() as u32;
        let running_jobs = jobs
            .iter()
            .filter(|j| matches!(j.state, JobStateRepr::Running))
            .count() as u32;
        let completed_jobs = jobs
            .iter()
            .filter(|j| matches!(j.state, JobStateRepr::Completed))
            .count() as u32;
        let failed_jobs = jobs
            .iter()
            .filter(|j| matches!(j.state, JobStateRepr::Failed))
            .count() as u32;

        let running_job_list: Vec<&JobSnapshot> = jobs
            .iter()
            .filter(|j| matches!(j.state, JobStateRepr::Running))
            .collect();
        let avg_progress_pct = if running_job_list.is_empty() {
            0.0
        } else {
            running_job_list.iter().map(|j| j.progress_pct).sum::<f32>()
                / running_job_list.len() as f32
        };

        let total_capacity: u32 = workers.iter().map(|w| w.capacity).sum();
        let queue_utilisation = if total_capacity == 0 {
            0.0
        } else {
            let active_tasks: u32 = workers.iter().map(|w| w.active_tasks).sum();
            (active_tasks as f32 / total_capacity as f32).min(1.0)
        };

        Self {
            total_workers,
            idle_workers,
            busy_workers,
            unavailable_workers,
            total_jobs,
            pending_jobs,
            running_jobs,
            completed_jobs,
            failed_jobs,
            avg_progress_pct,
            queue_utilisation,
        }
    }
}

// ---------------------------------------------------------------------------
// DashboardApiHandler
// ---------------------------------------------------------------------------

/// State snapshot fed into the dashboard handler on each request.
///
/// Clone is cheap because all owned data is snapshot-level.
#[derive(Debug, Clone, Default)]
pub struct DashboardState {
    /// Current workers.
    pub workers: Vec<WorkerSnapshot>,
    /// Current jobs.
    pub jobs: Vec<JobSnapshot>,
}

/// Stateless HTTP/1.1 dashboard API handler.
///
/// Route table:
/// ```text
/// GET /api/workers  → WorkerSnapshot[]  (JSON array)
/// GET /api/jobs     → JobSnapshot[]     (JSON array)
/// GET /api/metrics  → FarmMetricsSnapshot (JSON object)
/// ```
///
/// All other methods return `405 Method Not Allowed`.
/// All other paths return `404 Not Found`.
#[derive(Debug, Clone, Default)]
pub struct DashboardApiHandler {
    state: DashboardState,
}

impl DashboardApiHandler {
    /// Create a new handler with an empty state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: DashboardState::default(),
        }
    }

    /// Create a handler with pre-populated state.
    #[must_use]
    pub fn with_state(state: DashboardState) -> Self {
        Self { state }
    }

    /// Update the handler's state snapshot.
    pub fn update_state(&mut self, state: DashboardState) {
        self.state = state;
    }

    /// Handle an [`HttpRequest`] and return an [`HttpResponse`].
    ///
    /// This method is the core dispatch table and is always synchronous so
    /// it can be called from any context (sync or async).
    ///
    /// # Errors
    ///
    /// Returns an [`HttpResponse`] with an appropriate error status code
    /// if the method is not allowed or the path is not found. JSON
    /// serialisation errors are promoted to `500 Internal Server Error`.
    pub fn handle(&self, req: &HttpRequest) -> HttpResponse {
        if req.method != "GET" {
            return HttpResponse::method_not_allowed(&req.method);
        }

        // Strip query string for routing.
        let path = req.path.split('?').next().unwrap_or(&req.path);

        match path {
            "/api/workers" => self.handle_workers(),
            "/api/jobs" => self.handle_jobs(),
            "/api/metrics" => self.handle_metrics(),
            _ => HttpResponse::not_found(path),
        }
    }

    /// Handle a raw HTTP/1.1 request string.
    ///
    /// Parses the request and delegates to [`Self::handle`]. Returns a
    /// `400 Bad Request` response if the request line cannot be parsed.
    #[must_use]
    pub fn handle_raw(&self, raw: &str) -> HttpResponse {
        match HttpRequest::parse(raw) {
            Some(req) => self.handle(&req),
            None => HttpResponse {
                status_code: 400,
                reason: "Bad Request".to_string(),
                body: "{\"error\":\"bad request\"}".to_string(),
                content_type: "application/json".to_string(),
            },
        }
    }

    // -- Route handlers --

    fn handle_workers(&self) -> HttpResponse {
        match serde_json::to_string(&self.state.workers) {
            Ok(json) => HttpResponse::ok_json(json),
            Err(e) => HttpResponse {
                status_code: 500,
                reason: "Internal Server Error".to_string(),
                body: format!("{{\"error\":\"serialization failed: {e}\"}}"),
                content_type: "application/json".to_string(),
            },
        }
    }

    fn handle_jobs(&self) -> HttpResponse {
        match serde_json::to_string(&self.state.jobs) {
            Ok(json) => HttpResponse::ok_json(json),
            Err(e) => HttpResponse {
                status_code: 500,
                reason: "Internal Server Error".to_string(),
                body: format!("{{\"error\":\"serialization failed: {e}\"}}"),
                content_type: "application/json".to_string(),
            },
        }
    }

    fn handle_metrics(&self) -> HttpResponse {
        let metrics = FarmMetricsSnapshot::compute(&self.state.workers, &self.state.jobs);
        match serde_json::to_string(&metrics) {
            Ok(json) => HttpResponse::ok_json(json),
            Err(e) => HttpResponse {
                status_code: 500,
                reason: "Internal Server Error".to_string(),
                body: format!("{{\"error\":\"serialization failed: {e}\"}}"),
                content_type: "application/json".to_string(),
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- HttpRequest parsing --

    #[test]
    fn test_parse_get_request() {
        let raw = "GET /api/workers HTTP/1.1\r\nHost: localhost\r\n\r\n";
        let req = HttpRequest::parse(raw).expect("should parse");
        assert_eq!(req.method, "GET");
        assert_eq!(req.path, "/api/workers");
        assert_eq!(req.version, "HTTP/1.1");
    }

    #[test]
    fn test_parse_request_with_headers() {
        let raw = "GET /api/jobs HTTP/1.1\r\nHost: localhost\r\nAccept: application/json\r\n\r\n";
        let req = HttpRequest::parse(raw).expect("should parse");
        assert_eq!(
            req.headers.get("host").map(String::as_str),
            Some("localhost")
        );
        assert_eq!(
            req.headers.get("accept").map(String::as_str),
            Some("application/json")
        );
    }

    #[test]
    fn test_parse_malformed_returns_none() {
        assert!(HttpRequest::parse("").is_none());
        assert!(HttpRequest::parse("   ").is_none());
    }

    #[test]
    fn test_parse_post_request() {
        let raw = "POST /api/jobs HTTP/1.1\r\n\r\n";
        let req = HttpRequest::parse(raw).expect("should parse");
        assert_eq!(req.method, "POST");
    }

    // -- HttpResponse serialization --

    #[test]
    fn test_ok_json_response_bytes() {
        let resp = HttpResponse::ok_json("{\"key\":\"value\"}");
        let bytes = resp.to_http_bytes();
        let text = String::from_utf8(bytes).expect("utf8");
        assert!(text.starts_with("HTTP/1.1 200 OK"));
        assert!(text.contains("application/json"));
        assert!(text.contains("{\"key\":\"value\"}"));
    }

    #[test]
    fn test_not_found_response() {
        let resp = HttpResponse::not_found("/unknown");
        assert_eq!(resp.status_code, 404);
        assert!(resp.body.contains("not found"));
    }

    #[test]
    fn test_method_not_allowed_response() {
        let resp = HttpResponse::method_not_allowed("POST");
        assert_eq!(resp.status_code, 405);
        assert!(resp.body.contains("POST"));
    }

    #[test]
    fn test_http_string_contains_content_length() {
        let resp = HttpResponse::ok_json("{\"a\":1}");
        let s = resp.to_http_string().expect("utf8");
        assert!(s.contains("Content-Length: 7"));
    }

    // -- WorkerStateRepr conversion --

    #[test]
    fn test_worker_state_repr_from() {
        assert!(matches!(
            WorkerStateRepr::from(WorkerState::Idle),
            WorkerStateRepr::Idle
        ));
        assert!(matches!(
            WorkerStateRepr::from(WorkerState::Offline),
            WorkerStateRepr::Offline
        ));
    }

    // -- JobStateRepr conversion --

    #[test]
    fn test_job_state_repr_from() {
        assert!(matches!(
            JobStateRepr::from(JobState::Running),
            JobStateRepr::Running
        ));
        assert!(matches!(
            JobStateRepr::from(JobState::Failed),
            JobStateRepr::Failed
        ));
    }

    // -- PriorityRepr conversion --

    #[test]
    fn test_priority_repr_from() {
        assert!(matches!(
            PriorityRepr::from(Priority::Critical),
            PriorityRepr::Critical
        ));
        assert!(matches!(
            PriorityRepr::from(Priority::Low),
            PriorityRepr::Low
        ));
    }

    // -- FarmMetricsSnapshot computation --

    fn make_worker(id: &str, state: WorkerStateRepr, active: u32, capacity: u32) -> WorkerSnapshot {
        WorkerSnapshot {
            id: id.to_string(),
            state,
            active_tasks: active,
            capacity,
            address: format!("127.0.0.1:800{}", id),
        }
    }

    fn make_job(
        id: &str,
        state: JobStateRepr,
        priority: PriorityRepr,
        progress: f32,
    ) -> JobSnapshot {
        JobSnapshot {
            id: id.to_string(),
            name: format!("job-{id}"),
            state,
            priority,
            submitted_at: 1_700_000_000,
            progress_pct: progress,
        }
    }

    #[test]
    fn test_metrics_compute_empty() {
        let m = FarmMetricsSnapshot::compute(&[], &[]);
        assert_eq!(m.total_workers, 0);
        assert_eq!(m.total_jobs, 0);
        assert!((m.queue_utilisation - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_metrics_compute_workers() {
        let workers = vec![
            make_worker("1", WorkerStateRepr::Idle, 0, 4),
            make_worker("2", WorkerStateRepr::Busy, 3, 4),
            make_worker("3", WorkerStateRepr::Offline, 0, 4),
        ];
        let m = FarmMetricsSnapshot::compute(&workers, &[]);
        assert_eq!(m.total_workers, 3);
        assert_eq!(m.idle_workers, 1);
        assert_eq!(m.busy_workers, 1);
        assert_eq!(m.unavailable_workers, 1);
        // queue_utilisation = 3 / 12 = 0.25
        assert!((m.queue_utilisation - 0.25).abs() < 1e-5);
    }

    #[test]
    fn test_metrics_compute_jobs() {
        let jobs = vec![
            make_job("1", JobStateRepr::Pending, PriorityRepr::Normal, 0.0),
            make_job("2", JobStateRepr::Running, PriorityRepr::High, 50.0),
            make_job("3", JobStateRepr::Running, PriorityRepr::Critical, 80.0),
            make_job("4", JobStateRepr::Completed, PriorityRepr::Low, 100.0),
            make_job("5", JobStateRepr::Failed, PriorityRepr::Normal, 30.0),
        ];
        let m = FarmMetricsSnapshot::compute(&[], &jobs);
        assert_eq!(m.total_jobs, 5);
        assert_eq!(m.pending_jobs, 1);
        assert_eq!(m.running_jobs, 2);
        assert_eq!(m.completed_jobs, 1);
        assert_eq!(m.failed_jobs, 1);
        // avg_progress_pct = (50 + 80) / 2 = 65
        assert!((m.avg_progress_pct - 65.0).abs() < 1e-4);
    }

    // -- DashboardApiHandler routing --

    fn make_handler() -> DashboardApiHandler {
        let workers = vec![
            make_worker("w1", WorkerStateRepr::Idle, 0, 8),
            make_worker("w2", WorkerStateRepr::Busy, 4, 8),
        ];
        let jobs = vec![
            make_job("j1", JobStateRepr::Running, PriorityRepr::High, 60.0),
            make_job("j2", JobStateRepr::Pending, PriorityRepr::Normal, 0.0),
        ];
        DashboardApiHandler::with_state(DashboardState { workers, jobs })
    }

    #[test]
    fn test_handle_get_workers_returns_200() {
        let handler = make_handler();
        let req = HttpRequest::parse("GET /api/workers HTTP/1.1\r\n\r\n").expect("should parse");
        let resp = handler.handle(&req);
        assert_eq!(resp.status_code, 200);
        assert!(resp.body.contains("w1"));
        assert!(resp.body.contains("w2"));
    }

    #[test]
    fn test_handle_get_jobs_returns_200() {
        let handler = make_handler();
        let req = HttpRequest::parse("GET /api/jobs HTTP/1.1\r\n\r\n").expect("should parse");
        let resp = handler.handle(&req);
        assert_eq!(resp.status_code, 200);
        assert!(resp.body.contains("j1"));
        assert!(resp.body.contains("j2"));
    }

    #[test]
    fn test_handle_get_metrics_returns_200() {
        let handler = make_handler();
        let req = HttpRequest::parse("GET /api/metrics HTTP/1.1\r\n\r\n").expect("should parse");
        let resp = handler.handle(&req);
        assert_eq!(resp.status_code, 200);
        assert!(resp.body.contains("total_workers"));
        assert!(resp.body.contains("total_jobs"));
    }

    #[test]
    fn test_handle_unknown_path_returns_404() {
        let handler = make_handler();
        let req = HttpRequest::parse("GET /unknown HTTP/1.1\r\n\r\n").expect("should parse");
        let resp = handler.handle(&req);
        assert_eq!(resp.status_code, 404);
    }

    #[test]
    fn test_handle_post_returns_405() {
        let handler = make_handler();
        let req = HttpRequest::parse("POST /api/workers HTTP/1.1\r\n\r\n").expect("should parse");
        let resp = handler.handle(&req);
        assert_eq!(resp.status_code, 405);
    }

    #[test]
    fn test_handle_raw_valid() {
        let handler = make_handler();
        let resp = handler.handle_raw("GET /api/metrics HTTP/1.1\r\n\r\n");
        assert_eq!(resp.status_code, 200);
    }

    #[test]
    fn test_handle_raw_malformed_returns_400() {
        let handler = make_handler();
        let resp = handler.handle_raw("");
        assert_eq!(resp.status_code, 400);
    }

    #[test]
    fn test_path_with_query_string_routes_correctly() {
        let handler = make_handler();
        let req =
            HttpRequest::parse("GET /api/workers?limit=10 HTTP/1.1\r\n\r\n").expect("should parse");
        let resp = handler.handle(&req);
        assert_eq!(resp.status_code, 200);
    }

    #[test]
    fn test_update_state() {
        let mut handler = DashboardApiHandler::new();
        let req = HttpRequest::parse("GET /api/workers HTTP/1.1\r\n\r\n").expect("should parse");

        // Initially empty.
        let resp = handler.handle(&req);
        assert_eq!(resp.status_code, 200);
        assert_eq!(resp.body, "[]");

        // After update.
        handler.update_state(DashboardState {
            workers: vec![make_worker("w99", WorkerStateRepr::Idle, 0, 4)],
            jobs: vec![],
        });
        let resp = handler.handle(&req);
        assert_eq!(resp.status_code, 200);
        assert!(resp.body.contains("w99"));
    }

    #[test]
    fn test_response_content_type_is_json() {
        let handler = make_handler();
        let req = HttpRequest::parse("GET /api/metrics HTTP/1.1\r\n\r\n").expect("should parse");
        let resp = handler.handle(&req);
        assert!(resp.content_type.contains("application/json"));
    }

    #[test]
    fn test_workers_json_contains_state_field() {
        let handler = make_handler();
        let req = HttpRequest::parse("GET /api/workers HTTP/1.1\r\n\r\n").expect("should parse");
        let resp = handler.handle(&req);
        assert!(resp.body.contains("\"state\""));
        assert!(resp.body.contains("\"idle\""));
    }

    #[test]
    fn test_jobs_json_contains_priority_field() {
        let handler = make_handler();
        let req = HttpRequest::parse("GET /api/jobs HTTP/1.1\r\n\r\n").expect("should parse");
        let resp = handler.handle(&req);
        assert!(resp.body.contains("\"priority\""));
    }
}
