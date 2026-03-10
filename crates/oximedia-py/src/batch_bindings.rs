//! Batch processing Python binding data structures.
//!
//! Defines plain Rust representations for batch job submission, result
//! collection, and status queries that bridge Python and the OxiMedia
//! batch processor.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;

/// Priority level for a batch job.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum JobPriority {
    /// Lowest priority.
    Low = 0,
    /// Normal priority (default).
    Normal = 1,
    /// High priority.
    High = 2,
    /// Urgent – process before all others.
    Urgent = 3,
}

impl JobPriority {
    /// Return the priority as a human-readable label.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Normal => "normal",
            Self::High => "high",
            Self::Urgent => "urgent",
        }
    }
}

/// Status of a batch job.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum JobStatus {
    /// Waiting in the queue.
    Queued,
    /// Currently being processed.
    Running,
    /// Completed successfully.
    Completed,
    /// Failed with an error message.
    Failed(String),
    /// Cancelled by the user.
    Cancelled,
}

impl JobStatus {
    /// Returns `true` if the job has reached a terminal state.
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed | Self::Failed(_) | Self::Cancelled)
    }

    /// Returns `true` if the job succeeded.
    #[must_use]
    pub fn is_success(&self) -> bool {
        matches!(self, Self::Completed)
    }

    /// Short label for the status.
    #[must_use]
    pub fn label(&self) -> &str {
        match self {
            Self::Queued => "queued",
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Failed(_) => "failed",
            Self::Cancelled => "cancelled",
        }
    }
}

/// A batch job submission request.
#[derive(Clone, Debug, PartialEq)]
pub struct JobSubmission {
    /// Client-assigned job identifier (e.g. UUID string).
    pub job_id: String,
    /// Input file or URL.
    pub input: String,
    /// Output file or URL.
    pub output: String,
    /// Named operation to perform (e.g. `"transcode"`, `"thumbnail"`).
    pub operation: String,
    /// Additional key-value parameters for the operation.
    pub params: HashMap<String, String>,
    /// Priority of this job.
    pub priority: JobPriority,
}

impl JobSubmission {
    /// Create a new `JobSubmission` with `Normal` priority and no extra params.
    #[must_use]
    pub fn new(
        job_id: impl Into<String>,
        input: impl Into<String>,
        output: impl Into<String>,
        operation: impl Into<String>,
    ) -> Self {
        Self {
            job_id: job_id.into(),
            input: input.into(),
            output: output.into(),
            operation: operation.into(),
            params: HashMap::new(),
            priority: JobPriority::Normal,
        }
    }

    /// Set job priority.
    #[must_use]
    pub fn with_priority(mut self, priority: JobPriority) -> Self {
        self.priority = priority;
        self
    }

    /// Add an operation parameter.
    #[must_use]
    pub fn with_param(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.params.insert(key.into(), value.into());
        self
    }
}

/// The result of a completed (or failed) batch job.
#[derive(Clone, Debug, PartialEq)]
pub struct JobResult {
    /// Job identifier matching `JobSubmission.job_id`.
    pub job_id: String,
    /// Final status.
    pub status: JobStatus,
    /// Wall-clock processing time in seconds.
    pub elapsed_secs: f64,
    /// Output file size in bytes (if applicable).
    pub output_bytes: Option<u64>,
    /// Optional diagnostic or informational message.
    pub message: Option<String>,
}

impl JobResult {
    /// Create a successful `JobResult`.
    #[must_use]
    pub fn success(job_id: impl Into<String>, elapsed_secs: f64) -> Self {
        Self {
            job_id: job_id.into(),
            status: JobStatus::Completed,
            elapsed_secs,
            output_bytes: None,
            message: None,
        }
    }

    /// Create a failed `JobResult`.
    #[must_use]
    pub fn failure(job_id: impl Into<String>, error: impl Into<String>) -> Self {
        Self {
            job_id: job_id.into(),
            status: JobStatus::Failed(error.into()),
            elapsed_secs: 0.0,
            output_bytes: None,
            message: None,
        }
    }

    /// Returns `true` if the job completed successfully.
    #[must_use]
    pub fn is_success(&self) -> bool {
        self.status.is_success()
    }

    /// Processing throughput in MB/s (output_bytes / elapsed), or 0 if unknown.
    #[must_use]
    pub fn throughput_mb_per_s(&self) -> f64 {
        if self.elapsed_secs <= 0.0 {
            return 0.0;
        }
        self.output_bytes
            .map(|b| b as f64 / (1024.0 * 1024.0) / self.elapsed_secs)
            .unwrap_or(0.0)
    }
}

/// A lightweight status query response.
#[derive(Clone, Debug, PartialEq)]
pub struct StatusQuery {
    /// Job identifier.
    pub job_id: String,
    /// Current status.
    pub status: JobStatus,
    /// Progress in `[0.0, 1.0]`, or `None` if not yet known.
    pub progress: Option<f64>,
    /// Queue position (0-based); `None` if already running or done.
    pub queue_position: Option<usize>,
}

impl StatusQuery {
    /// Returns the progress percentage (0–100), or `None`.
    #[must_use]
    pub fn progress_pct(&self) -> Option<f64> {
        self.progress.map(|p| p * 100.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_job_priority_ordering() {
        assert!(JobPriority::Urgent > JobPriority::High);
        assert!(JobPriority::High > JobPriority::Normal);
        assert!(JobPriority::Normal > JobPriority::Low);
    }

    #[test]
    fn test_job_priority_as_str() {
        assert_eq!(JobPriority::Low.as_str(), "low");
        assert_eq!(JobPriority::Urgent.as_str(), "urgent");
    }

    #[test]
    fn test_job_status_terminal_completed() {
        assert!(JobStatus::Completed.is_terminal());
    }

    #[test]
    fn test_job_status_terminal_failed() {
        assert!(JobStatus::Failed("oops".to_string()).is_terminal());
    }

    #[test]
    fn test_job_status_not_terminal_running() {
        assert!(!JobStatus::Running.is_terminal());
    }

    #[test]
    fn test_job_status_not_terminal_queued() {
        assert!(!JobStatus::Queued.is_terminal());
    }

    #[test]
    fn test_job_status_is_success() {
        assert!(JobStatus::Completed.is_success());
        assert!(!JobStatus::Failed("x".to_string()).is_success());
    }

    #[test]
    fn test_job_status_label() {
        assert_eq!(JobStatus::Queued.label(), "queued");
        assert_eq!(JobStatus::Running.label(), "running");
        assert_eq!(JobStatus::Completed.label(), "completed");
        assert_eq!(JobStatus::Failed("e".to_string()).label(), "failed");
        assert_eq!(JobStatus::Cancelled.label(), "cancelled");
    }

    #[test]
    fn test_job_submission_new_defaults() {
        let j = JobSubmission::new("id1", "/in.mkv", "/out.mp4", "transcode");
        assert_eq!(j.priority, JobPriority::Normal);
        assert!(j.params.is_empty());
    }

    #[test]
    fn test_job_submission_with_priority() {
        let j = JobSubmission::new("id1", "/in.mkv", "/out.mp4", "transcode")
            .with_priority(JobPriority::High);
        assert_eq!(j.priority, JobPriority::High);
    }

    #[test]
    fn test_job_submission_with_param() {
        let j =
            JobSubmission::new("id1", "/in.mkv", "/out.mp4", "transcode").with_param("crf", "28");
        assert_eq!(j.params.get("crf").map(String::as_str), Some("28"));
    }

    #[test]
    fn test_job_result_success() {
        let r = JobResult::success("id1", 5.0);
        assert!(r.is_success());
        assert_eq!(r.elapsed_secs, 5.0);
    }

    #[test]
    fn test_job_result_failure() {
        let r = JobResult::failure("id1", "disk full");
        assert!(!r.is_success());
        assert!(matches!(r.status, JobStatus::Failed(_)));
    }

    #[test]
    fn test_job_result_throughput() {
        let mut r = JobResult::success("id1", 2.0);
        r.output_bytes = Some(2 * 1024 * 1024); // 2 MB
        assert!((r.throughput_mb_per_s() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_job_result_throughput_no_output() {
        let r = JobResult::success("id1", 2.0);
        assert_eq!(r.throughput_mb_per_s(), 0.0);
    }

    #[test]
    fn test_status_query_progress_pct() {
        let q = StatusQuery {
            job_id: "x".to_string(),
            status: JobStatus::Running,
            progress: Some(0.75),
            queue_position: None,
        };
        assert!(
            (q.progress_pct().expect("progress_pct should succeed") - 75.0).abs() < f64::EPSILON
        );
    }

    #[test]
    fn test_status_query_no_progress() {
        let q = StatusQuery {
            job_id: "x".to_string(),
            status: JobStatus::Queued,
            progress: None,
            queue_position: Some(3),
        };
        assert!(q.progress_pct().is_none());
    }
}
