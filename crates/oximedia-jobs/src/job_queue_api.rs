#![allow(dead_code)]
//! REST-style API data structures for remote job queue management.
//!
//! This module provides request/response types, routing logic, and an in-process
//! HTTP handler abstraction for submitting and monitoring jobs over the network.
//! It is intentionally decoupled from any specific HTTP framework so that it
//! can be wired into axum, actix-web, or a test harness without changes.
//!
//! # Design
//! The `ApiHandler` owns a reference-counted queue and exposes `handle_request`
//! which accepts an `ApiRequest` enum and returns an `ApiResponse`.  The actual
//! HTTP framing (routing, status codes, JSON encoding) is handled by callers.

use crate::job::{Job, JobStatus, Priority};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Request types
// ---------------------------------------------------------------------------

/// Top-level API request discriminant.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum ApiRequest {
    /// Submit a new job.
    SubmitJob(SubmitJobRequest),
    /// Cancel an existing job.
    CancelJob { job_id: Uuid },
    /// Get the status of a job.
    GetJob { job_id: Uuid },
    /// List all jobs, optionally filtered by status.
    ListJobs(ListJobsRequest),
    /// Update the progress of an in-progress job.
    UpdateProgress { job_id: Uuid, progress: u8 },
    /// Get queue-wide statistics.
    GetStats,
}

/// Parameters for submitting a new job via the API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubmitJobRequest {
    /// Human-readable name.
    pub name: String,
    /// Priority level string: "high", "normal", or "low".
    #[serde(default = "default_priority_str")]
    pub priority: String,
    /// Arbitrary JSON payload string (interpreted by the executor).
    pub payload_json: String,
    /// Optional comma-separated tags.
    #[serde(default)]
    pub tags: Vec<String>,
    /// Optional deadline (RFC 3339 date-time string).
    #[serde(default)]
    pub deadline: Option<String>,
}

fn default_priority_str() -> String {
    "normal".to_string()
}

impl SubmitJobRequest {
    /// Parse the priority field into a `Priority` enum.
    #[must_use]
    pub fn parsed_priority(&self) -> Priority {
        match self.priority.to_ascii_lowercase().as_str() {
            "high" => Priority::High,
            "low" => Priority::Low,
            _ => Priority::Normal,
        }
    }
}

/// Parameters for listing jobs.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ListJobsRequest {
    /// If set, only return jobs with this status.
    pub status_filter: Option<String>,
    /// Maximum number of results (default 100).
    #[serde(default = "default_page_size")]
    pub limit: usize,
    /// Offset for pagination.
    #[serde(default)]
    pub offset: usize,
}

fn default_page_size() -> usize {
    100
}

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

/// Standardised API response envelope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiResponse {
    /// Whether the request succeeded.
    pub ok: bool,
    /// HTTP-like status code (200, 400, 404, 500, …).
    pub status: u16,
    /// Human-readable message.
    pub message: String,
    /// Optional JSON body.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

impl ApiResponse {
    /// Successful response with optional data payload.
    #[must_use]
    pub fn ok(message: impl Into<String>, data: Option<serde_json::Value>) -> Self {
        Self {
            ok: true,
            status: 200,
            message: message.into(),
            data,
        }
    }

    /// Created response (201).
    #[must_use]
    pub fn created(message: impl Into<String>, data: Option<serde_json::Value>) -> Self {
        Self {
            ok: true,
            status: 201,
            message: message.into(),
            data,
        }
    }

    /// Bad request error (400).
    #[must_use]
    pub fn bad_request(message: impl Into<String>) -> Self {
        Self {
            ok: false,
            status: 400,
            message: message.into(),
            data: None,
        }
    }

    /// Not found error (404).
    #[must_use]
    pub fn not_found(message: impl Into<String>) -> Self {
        Self {
            ok: false,
            status: 404,
            message: message.into(),
            data: None,
        }
    }

    /// Internal error (500).
    #[must_use]
    pub fn internal_error(message: impl Into<String>) -> Self {
        Self {
            ok: false,
            status: 500,
            message: message.into(),
            data: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Lightweight job summary (safe to serialise over the wire)
// ---------------------------------------------------------------------------

/// A compact representation of a job suitable for API responses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobSummary {
    /// Job identifier.
    pub id: Uuid,
    /// Human-readable name.
    pub name: String,
    /// Priority label.
    pub priority: String,
    /// Current status label.
    pub status: String,
    /// Progress (0-100).
    pub progress: u8,
    /// Tags attached to the job.
    pub tags: Vec<String>,
    /// Optional error message.
    pub error: Option<String>,
}

impl JobSummary {
    /// Build a summary from a `Job`.
    #[must_use]
    pub fn from_job(job: &Job) -> Self {
        Self {
            id: job.id,
            name: job.name.clone(),
            priority: format!("{:?}", job.priority),
            status: format!("{:?}", job.status),
            progress: job.progress,
            tags: job.tags.clone(),
            error: job.error.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// In-process API handler
// ---------------------------------------------------------------------------

/// Queue statistics summary returned by the API.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QueueStatsSummary {
    /// Total jobs known to the queue.
    pub total: usize,
    /// Pending jobs waiting for execution.
    pub pending: usize,
    /// Currently running jobs.
    pub running: usize,
    /// Successfully completed jobs.
    pub completed: usize,
    /// Failed jobs.
    pub failed: usize,
    /// Cancelled jobs.
    pub cancelled: usize,
}

/// In-memory job store for the API handler (lightweight, no persistence).
#[derive(Debug, Default)]
pub struct InMemoryJobStore {
    jobs: std::collections::HashMap<Uuid, Job>,
}

impl InMemoryJobStore {
    /// Create an empty job store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or update a job.
    pub fn upsert(&mut self, job: Job) {
        self.jobs.insert(job.id, job);
    }

    /// Get a job by id.
    pub fn get(&self, id: Uuid) -> Option<&Job> {
        self.jobs.get(&id)
    }

    /// Get a mutable job by id.
    pub fn get_mut(&mut self, id: Uuid) -> Option<&mut Job> {
        self.jobs.get_mut(&id)
    }

    /// All jobs, cloned.
    pub fn all(&self) -> Vec<Job> {
        self.jobs.values().cloned().collect()
    }

    /// Jobs filtered by status.
    pub fn by_status(&self, status: &JobStatus) -> Vec<&Job> {
        self.jobs
            .values()
            .filter(|j| &j.status == status)
            .collect()
    }

    /// Queue statistics.
    pub fn stats(&self) -> QueueStatsSummary {
        let mut stats = QueueStatsSummary::default();
        stats.total = self.jobs.len();
        for job in self.jobs.values() {
            match job.status {
                JobStatus::Pending | JobStatus::Waiting | JobStatus::Scheduled => {
                    stats.pending += 1;
                }
                JobStatus::Running => stats.running += 1,
                JobStatus::Completed => stats.completed += 1,
                JobStatus::Failed => stats.failed += 1,
                JobStatus::Cancelled => stats.cancelled += 1,
            }
        }
        stats
    }
}

/// In-process REST-style API handler for the job queue.
///
/// `handle_request` is synchronous for simplicity; wrap with async if needed.
pub struct ApiHandler {
    store: std::sync::Mutex<InMemoryJobStore>,
}

impl ApiHandler {
    /// Create a new API handler with an empty job store.
    #[must_use]
    pub fn new() -> Self {
        Self {
            store: std::sync::Mutex::new(InMemoryJobStore::new()),
        }
    }

    /// Handle a single API request and return a response.
    pub fn handle_request(&self, request: ApiRequest) -> ApiResponse {
        match request {
            ApiRequest::SubmitJob(req) => self.handle_submit(req),
            ApiRequest::CancelJob { job_id } => self.handle_cancel(job_id),
            ApiRequest::GetJob { job_id } => self.handle_get_job(job_id),
            ApiRequest::ListJobs(req) => self.handle_list_jobs(req),
            ApiRequest::UpdateProgress { job_id, progress } => {
                self.handle_update_progress(job_id, progress)
            }
            ApiRequest::GetStats => self.handle_get_stats(),
        }
    }

    fn handle_submit(&self, req: SubmitJobRequest) -> ApiResponse {
        use crate::job::JobPayload;
        let priority = req.parsed_priority();
        // Parse the payload JSON into a generic Analysis payload for now
        let payload = match serde_json::from_str::<JobPayload>(&req.payload_json) {
            Ok(p) => p,
            Err(e) => {
                return ApiResponse::bad_request(format!("Invalid payload JSON: {e}"));
            }
        };
        let mut job = Job::new(req.name, priority, payload);
        job.tags = req.tags;
        let id = job.id;
        let mut store = match self.store.lock() {
            Ok(g) => g,
            Err(e) => return ApiResponse::internal_error(format!("Lock error: {e}")),
        };
        store.upsert(job);
        ApiResponse::created(
            "Job submitted",
            Some(serde_json::json!({ "job_id": id })),
        )
    }

    fn handle_cancel(&self, job_id: Uuid) -> ApiResponse {
        let mut store = match self.store.lock() {
            Ok(g) => g,
            Err(e) => return ApiResponse::internal_error(format!("Lock error: {e}")),
        };
        match store.get_mut(job_id) {
            None => ApiResponse::not_found(format!("Job {job_id} not found")),
            Some(job) => {
                if matches!(job.status, JobStatus::Completed | JobStatus::Failed | JobStatus::Cancelled) {
                    ApiResponse::bad_request(format!(
                        "Job {job_id} is already in terminal state"
                    ))
                } else {
                    job.status = JobStatus::Cancelled;
                    ApiResponse::ok("Job cancelled", Some(serde_json::json!({ "job_id": job_id })))
                }
            }
        }
    }

    fn handle_get_job(&self, job_id: Uuid) -> ApiResponse {
        let store = match self.store.lock() {
            Ok(g) => g,
            Err(e) => return ApiResponse::internal_error(format!("Lock error: {e}")),
        };
        match store.get(job_id) {
            None => ApiResponse::not_found(format!("Job {job_id} not found")),
            Some(job) => {
                let summary = JobSummary::from_job(job);
                let data = match serde_json::to_value(&summary) {
                    Ok(v) => v,
                    Err(e) => return ApiResponse::internal_error(format!("Serialise error: {e}")),
                };
                ApiResponse::ok("Job found", Some(data))
            }
        }
    }

    fn handle_list_jobs(&self, req: ListJobsRequest) -> ApiResponse {
        let store = match self.store.lock() {
            Ok(g) => g,
            Err(e) => return ApiResponse::internal_error(format!("Lock error: {e}")),
        };
        let mut jobs: Vec<Job> = store.all();
        // Optional status filter
        if let Some(status_str) = &req.status_filter {
            let wanted = status_str.to_ascii_lowercase();
            jobs.retain(|j| format!("{:?}", j.status).to_ascii_lowercase() == wanted);
        }
        // Pagination
        let total = jobs.len();
        let page: Vec<JobSummary> = jobs
            .iter()
            .skip(req.offset)
            .take(req.limit)
            .map(JobSummary::from_job)
            .collect();
        let data = match serde_json::to_value(&page) {
            Ok(v) => v,
            Err(e) => return ApiResponse::internal_error(format!("Serialise error: {e}")),
        };
        ApiResponse::ok(
            format!("{} of {} jobs returned", page.len(), total),
            Some(serde_json::json!({ "jobs": data, "total": total, "offset": req.offset })),
        )
    }

    fn handle_update_progress(&self, job_id: Uuid, progress: u8) -> ApiResponse {
        let mut store = match self.store.lock() {
            Ok(g) => g,
            Err(e) => return ApiResponse::internal_error(format!("Lock error: {e}")),
        };
        match store.get_mut(job_id) {
            None => ApiResponse::not_found(format!("Job {job_id} not found")),
            Some(job) => {
                job.progress = progress.min(100);
                ApiResponse::ok(
                    "Progress updated",
                    Some(serde_json::json!({ "job_id": job_id, "progress": job.progress })),
                )
            }
        }
    }

    fn handle_get_stats(&self) -> ApiResponse {
        let store = match self.store.lock() {
            Ok(g) => g,
            Err(e) => return ApiResponse::internal_error(format!("Lock error: {e}")),
        };
        let stats = store.stats();
        let data = match serde_json::to_value(&stats) {
            Ok(v) => v,
            Err(e) => return ApiResponse::internal_error(format!("Serialise error: {e}")),
        };
        ApiResponse::ok("Queue statistics", Some(data))
    }
}

impl Default for ApiHandler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::job::{AnalysisType, JobPayload};

    fn make_submit_request(name: &str) -> SubmitJobRequest {
        let payload = JobPayload::Analysis(crate::job::AnalysisParams {
            input: "file.mp4".to_string(),
            analysis_type: AnalysisType::Quality,
            output: None,
        });
        SubmitJobRequest {
            name: name.to_string(),
            priority: "normal".to_string(),
            payload_json: serde_json::to_string(&payload).expect("test expectation"),
            tags: vec!["test".to_string()],
            deadline: None,
        }
    }

    #[test]
    fn test_submit_job_returns_created() {
        let handler = ApiHandler::new();
        let resp = handler.handle_request(ApiRequest::SubmitJob(make_submit_request("job1")));
        assert!(resp.ok);
        assert_eq!(resp.status, 201);
        assert!(resp.data.is_some());
    }

    #[test]
    fn test_get_job_after_submit() {
        let handler = ApiHandler::new();
        let resp = handler.handle_request(ApiRequest::SubmitJob(make_submit_request("job2")));
        let job_id: Uuid = serde_json::from_value(
            resp.data
                .expect("data")
                .get("job_id")
                .expect("job_id key")
                .clone(),
        )
        .expect("uuid");
        let get_resp = handler.handle_request(ApiRequest::GetJob { job_id });
        assert!(get_resp.ok);
        assert_eq!(get_resp.status, 200);
    }

    #[test]
    fn test_get_nonexistent_job_returns_404() {
        let handler = ApiHandler::new();
        let resp = handler.handle_request(ApiRequest::GetJob {
            job_id: Uuid::new_v4(),
        });
        assert!(!resp.ok);
        assert_eq!(resp.status, 404);
    }

    #[test]
    fn test_cancel_job() {
        let handler = ApiHandler::new();
        let resp = handler.handle_request(ApiRequest::SubmitJob(make_submit_request("job3")));
        let job_id: Uuid = serde_json::from_value(
            resp.data
                .expect("data")
                .get("job_id")
                .expect("job_id key")
                .clone(),
        )
        .expect("uuid");
        let cancel = handler.handle_request(ApiRequest::CancelJob { job_id });
        assert!(cancel.ok);
    }

    #[test]
    fn test_cancel_nonexistent_returns_404() {
        let handler = ApiHandler::new();
        let resp = handler.handle_request(ApiRequest::CancelJob {
            job_id: Uuid::new_v4(),
        });
        assert!(!resp.ok);
        assert_eq!(resp.status, 404);
    }

    #[test]
    fn test_update_progress() {
        let handler = ApiHandler::new();
        let resp = handler.handle_request(ApiRequest::SubmitJob(make_submit_request("job4")));
        let job_id: Uuid = serde_json::from_value(
            resp.data
                .expect("data")
                .get("job_id")
                .expect("job_id key")
                .clone(),
        )
        .expect("uuid");
        let upd = handler.handle_request(ApiRequest::UpdateProgress {
            job_id,
            progress: 75,
        });
        assert!(upd.ok);
    }

    #[test]
    fn test_list_jobs_returns_all() {
        let handler = ApiHandler::new();
        handler.handle_request(ApiRequest::SubmitJob(make_submit_request("a")));
        handler.handle_request(ApiRequest::SubmitJob(make_submit_request("b")));
        let list = handler.handle_request(ApiRequest::ListJobs(ListJobsRequest::default()));
        assert!(list.ok);
        let data = list.data.expect("data");
        let total = data.get("total").and_then(|v| v.as_u64()).expect("total");
        assert_eq!(total, 2);
    }

    #[test]
    fn test_get_stats() {
        let handler = ApiHandler::new();
        handler.handle_request(ApiRequest::SubmitJob(make_submit_request("s1")));
        let resp = handler.handle_request(ApiRequest::GetStats);
        assert!(resp.ok);
        let data = resp.data.expect("data");
        assert!(data.get("total").is_some());
    }

    #[test]
    fn test_submit_invalid_payload_returns_400() {
        let handler = ApiHandler::new();
        let req = SubmitJobRequest {
            name: "bad".to_string(),
            priority: "normal".to_string(),
            payload_json: "not json at all".to_string(),
            tags: vec![],
            deadline: None,
        };
        let resp = handler.handle_request(ApiRequest::SubmitJob(req));
        assert!(!resp.ok);
        assert_eq!(resp.status, 400);
    }

    #[test]
    fn test_priority_parsing() {
        use crate::job::Priority as JobPriority;
        let mut req = make_submit_request("p");
        req.priority = "high".to_string();
        assert_eq!(req.parsed_priority(), JobPriority::High);
        req.priority = "low".to_string();
        assert_eq!(req.parsed_priority(), JobPriority::Low);
        req.priority = "unknown".to_string();
        assert_eq!(req.parsed_priority(), JobPriority::Normal);
    }
}
