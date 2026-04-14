// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Per-job structured event log with state transitions, user actions, system events,
//! and structured JSON export.
//!
//! `JobEventLog` maintains a chronological, append-only log of every significant
//! occurrence in a job's lifetime: state transitions (e.g. `Pending → Running`),
//! user actions (cancel, pause, reprioritize), and system-generated events
//! (retry scheduled, worker assigned, deadline exceeded).
//!
//! # Design
//! - Events are keyed by job UUID for O(1) lookup.
//! - Each event carries: timestamp, actor (System/User/Worker), kind, and an
//!   optional freeform message.
//! - The log can be exported as structured JSON for audit trails and debugging.
//! - Log entries can be filtered by kind, actor, or time range.
//!
//! # Example
//! ```rust
//! use oximedia_jobs::job_event_log::{JobEventLog, EventKind, EventActor};
//! use uuid::Uuid;
//!
//! let job_id = Uuid::new_v4();
//! let mut log = JobEventLog::new();
//!
//! log.record(job_id, EventActor::System, EventKind::StateTransition {
//!     from: "Pending".into(),
//!     to: "Running".into(),
//! }, None);
//!
//! let events = log.events_for(job_id);
//! assert_eq!(events.len(), 1);
//! ```

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// EventActor
// ---------------------------------------------------------------------------

/// Who or what generated a log event.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventActor {
    /// Generated automatically by the job scheduling system.
    System,
    /// Triggered by an authenticated user action.
    User(String),
    /// Emitted by a specific worker node.
    Worker(String),
}

impl EventActor {
    /// Returns a human-readable label for display.
    #[must_use]
    pub fn label(&self) -> String {
        match self {
            Self::System => "system".into(),
            Self::User(id) => format!("user:{id}"),
            Self::Worker(id) => format!("worker:{id}"),
        }
    }
}

// ---------------------------------------------------------------------------
// EventKind
// ---------------------------------------------------------------------------

/// The semantic kind of a log event.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventKind {
    /// A job moved from one state to another.
    StateTransition {
        /// Previous state label.
        from: String,
        /// New state label.
        to: String,
    },
    /// A user submitted the job.
    Submitted,
    /// A worker was assigned to execute the job.
    WorkerAssigned {
        /// Identifier of the assigned worker.
        worker_id: String,
    },
    /// Job execution progress was updated.
    ProgressUpdate {
        /// Progress percentage (0–100).
        percent: u8,
    },
    /// A retry has been scheduled after a failure.
    RetryScheduled {
        /// Attempt number (1-indexed).
        attempt: u32,
        /// Delay in seconds before the retry.
        delay_secs: u64,
    },
    /// The job was cancelled by an actor.
    Cancelled {
        /// Reason for cancellation, if provided.
        reason: Option<String>,
    },
    /// The job deadline was exceeded without completion.
    DeadlineExceeded,
    /// The job was paused.
    Paused,
    /// The job was resumed after a pause.
    Resumed,
    /// A custom application-defined event.
    Custom {
        /// Short identifier for the custom event type.
        event_type: String,
    },
    /// The job result was stored.
    ResultStored {
        /// Brief summary of the stored result.
        summary: String,
    },
    /// The job's priority changed.
    PriorityChanged {
        /// Previous priority label.
        from: String,
        /// New priority label.
        to: String,
    },
}

impl EventKind {
    /// Returns a short category string for filtering.
    #[must_use]
    pub fn category(&self) -> &'static str {
        match self {
            Self::StateTransition { .. } => "state",
            Self::Submitted => "lifecycle",
            Self::WorkerAssigned { .. } => "assignment",
            Self::ProgressUpdate { .. } => "progress",
            Self::RetryScheduled { .. } => "retry",
            Self::Cancelled { .. } => "lifecycle",
            Self::DeadlineExceeded => "deadline",
            Self::Paused => "lifecycle",
            Self::Resumed => "lifecycle",
            Self::Custom { .. } => "custom",
            Self::ResultStored { .. } => "result",
            Self::PriorityChanged { .. } => "priority",
        }
    }
}

// ---------------------------------------------------------------------------
// LogEntry
// ---------------------------------------------------------------------------

/// A single structured log entry for a job event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    /// Sequential index within the job's event log (0-based).
    pub index: usize,
    /// The job this entry belongs to.
    pub job_id: Uuid,
    /// UTC timestamp when the event was recorded.
    pub timestamp: DateTime<Utc>,
    /// Who or what generated the event.
    pub actor: EventActor,
    /// The semantic kind of the event.
    pub kind: EventKind,
    /// Optional freeform message for additional context.
    pub message: Option<String>,
}

impl LogEntry {
    /// Returns `true` if this is a terminal event (the job will not continue).
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        matches!(
            &self.kind,
            EventKind::StateTransition { to, .. } if matches!(
                to.as_str(),
                "completed" | "failed" | "cancelled"
            )
        ) || matches!(&self.kind, EventKind::Cancelled { .. })
    }
}

// ---------------------------------------------------------------------------
// EventLogError
// ---------------------------------------------------------------------------

/// Errors from the job event log.
#[derive(Debug, thiserror::Error)]
pub enum EventLogError {
    /// No log entries exist for the given job.
    #[error("No event log found for job {0}")]
    UnknownJob(Uuid),
    /// JSON serialisation failed.
    #[error("Serialisation error: {0}")]
    Serialisation(#[from] serde_json::Error),
}

// ---------------------------------------------------------------------------
// TimeRange — optional filtering helper
// ---------------------------------------------------------------------------

/// An optional half-open time window `[from, to)` for event filtering.
#[derive(Debug, Clone, Default)]
pub struct TimeRange {
    /// Start of the window (inclusive). `None` means no lower bound.
    pub from: Option<DateTime<Utc>>,
    /// End of the window (exclusive). `None` means no upper bound.
    pub to: Option<DateTime<Utc>>,
}

impl TimeRange {
    /// Create an unbounded time range (matches everything).
    #[must_use]
    pub fn unbounded() -> Self {
        Self::default()
    }

    /// Create a range starting at `from` with no upper bound.
    #[must_use]
    pub fn since(from: DateTime<Utc>) -> Self {
        Self { from: Some(from), to: None }
    }

    /// Create a range ending before `to` with no lower bound.
    #[must_use]
    pub fn before(to: DateTime<Utc>) -> Self {
        Self { from: None, to: Some(to) }
    }

    /// Create a bounded range `[from, to)`.
    #[must_use]
    pub fn between(from: DateTime<Utc>, to: DateTime<Utc>) -> Self {
        Self { from: Some(from), to: Some(to) }
    }

    /// Returns `true` if `ts` falls within this range.
    #[must_use]
    pub fn contains(&self, ts: &DateTime<Utc>) -> bool {
        if let Some(from) = &self.from {
            if ts < from {
                return false;
            }
        }
        if let Some(to) = &self.to {
            if ts >= to {
                return false;
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// JobEventLog
// ---------------------------------------------------------------------------

/// Per-job structured event log.
///
/// Stores events in insertion order per job.  The log is in-memory; for
/// durable storage callers should periodically call [`JobEventLog::export_json`]
/// and persist the result.
#[derive(Debug, Default)]
pub struct JobEventLog {
    /// Map from job ID to its ordered list of log entries.
    entries: HashMap<Uuid, Vec<LogEntry>>,
}

impl JobEventLog {
    /// Create a new empty event log.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    // --- Recording ----------------------------------------------------------

    /// Record an event for `job_id`.
    ///
    /// Returns the index assigned to this entry.
    pub fn record(
        &mut self,
        job_id: Uuid,
        actor: EventActor,
        kind: EventKind,
        message: Option<String>,
    ) -> usize {
        let bucket = self.entries.entry(job_id).or_default();
        let index = bucket.len();
        bucket.push(LogEntry {
            index,
            job_id,
            timestamp: Utc::now(),
            actor,
            kind,
            message,
        });
        index
    }

    /// Record a state transition event.
    pub fn record_transition(
        &mut self,
        job_id: Uuid,
        from: impl Into<String>,
        to: impl Into<String>,
    ) -> usize {
        self.record(
            job_id,
            EventActor::System,
            EventKind::StateTransition {
                from: from.into(),
                to: to.into(),
            },
            None,
        )
    }

    /// Record that a worker was assigned to the job.
    pub fn record_worker_assigned(
        &mut self,
        job_id: Uuid,
        worker_id: impl Into<String>,
    ) -> usize {
        self.record(
            job_id,
            EventActor::System,
            EventKind::WorkerAssigned {
                worker_id: worker_id.into(),
            },
            None,
        )
    }

    /// Record a progress update.
    pub fn record_progress(&mut self, job_id: Uuid, percent: u8) -> usize {
        self.record(
            job_id,
            EventActor::System,
            EventKind::ProgressUpdate { percent },
            None,
        )
    }

    /// Record a user-initiated cancellation.
    pub fn record_cancellation(
        &mut self,
        job_id: Uuid,
        user_id: impl Into<String>,
        reason: Option<String>,
    ) -> usize {
        self.record(
            job_id,
            EventActor::User(user_id.into()),
            EventKind::Cancelled { reason },
            None,
        )
    }

    // --- Querying -----------------------------------------------------------

    /// Return all events for a job in insertion order.
    #[must_use]
    pub fn events_for(&self, job_id: Uuid) -> &[LogEntry] {
        self.entries.get(&job_id).map_or(&[], Vec::as_slice)
    }

    /// Return events for a job filtered by category.
    #[must_use]
    pub fn events_by_category(&self, job_id: Uuid, category: &str) -> Vec<&LogEntry> {
        self.events_for(job_id)
            .iter()
            .filter(|e| e.kind.category() == category)
            .collect()
    }

    /// Return events for a job filtered by actor type (label prefix match).
    #[must_use]
    pub fn events_by_actor_label(&self, job_id: Uuid, label_prefix: &str) -> Vec<&LogEntry> {
        self.events_for(job_id)
            .iter()
            .filter(|e| e.actor.label().starts_with(label_prefix))
            .collect()
    }

    /// Return events for a job within a time range.
    #[must_use]
    pub fn events_in_range(&self, job_id: Uuid, range: &TimeRange) -> Vec<&LogEntry> {
        self.events_for(job_id)
            .iter()
            .filter(|e| range.contains(&e.timestamp))
            .collect()
    }

    /// Count total events recorded across all jobs.
    #[must_use]
    pub fn total_event_count(&self) -> usize {
        self.entries.values().map(Vec::len).sum()
    }

    /// Number of distinct jobs that have at least one event.
    #[must_use]
    pub fn tracked_job_count(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if any recorded event for `job_id` is terminal.
    #[must_use]
    pub fn is_terminal(&self, job_id: Uuid) -> bool {
        self.events_for(job_id).iter().any(LogEntry::is_terminal)
    }

    // --- Export -------------------------------------------------------------

    /// Serialise all events for a job to a JSON string (one object per line,
    /// NDJSON-compatible).
    ///
    /// # Errors
    ///
    /// Returns `EventLogError::UnknownJob` if no events exist for `job_id`.
    /// Returns `EventLogError::Serialisation` on JSON encoding failures.
    pub fn export_json(&self, job_id: Uuid) -> Result<String, EventLogError> {
        let events = self
            .entries
            .get(&job_id)
            .ok_or(EventLogError::UnknownJob(job_id))?;
        let mut lines = Vec::with_capacity(events.len());
        for entry in events {
            lines.push(serde_json::to_string(entry)?);
        }
        Ok(lines.join("\n"))
    }

    /// Serialise all events for all jobs to a JSON array string.
    ///
    /// # Errors
    ///
    /// Returns `EventLogError::Serialisation` on JSON encoding failures.
    pub fn export_all_json(&self) -> Result<String, EventLogError> {
        let all: Vec<&LogEntry> = self.entries.values().flatten().collect();
        Ok(serde_json::to_string(&all)?)
    }

    /// Clear all events for a specific job.
    pub fn clear_job(&mut self, job_id: Uuid) {
        self.entries.remove(&job_id);
    }

    /// Clear the entire log.
    pub fn clear_all(&mut self) {
        self.entries.clear();
    }

    /// Retain only events for jobs whose IDs are in `keep_ids`.
    pub fn retain_jobs(&mut self, keep_ids: &[Uuid]) {
        self.entries.retain(|id, _| keep_ids.contains(id));
    }

    /// Return a summary of event counts per category for a given job.
    #[must_use]
    pub fn category_summary(&self, job_id: Uuid) -> HashMap<&'static str, usize> {
        let mut map: HashMap<&'static str, usize> = HashMap::new();
        for entry in self.events_for(job_id) {
            *map.entry(entry.kind.category()).or_insert(0) += 1;
        }
        map
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn new_job_id() -> Uuid {
        Uuid::new_v4()
    }

    #[test]
    fn test_record_and_retrieve() {
        let mut log = JobEventLog::new();
        let job_id = new_job_id();
        log.record(
            job_id,
            EventActor::System,
            EventKind::Submitted,
            Some("job queued".into()),
        );
        let events = log.events_for(job_id);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].kind, EventKind::Submitted);
        assert_eq!(events[0].index, 0);
        assert_eq!(events[0].message.as_deref(), Some("job queued"));
    }

    #[test]
    fn test_multiple_events_indexed() {
        let mut log = JobEventLog::new();
        let job_id = new_job_id();
        log.record_transition(job_id, "pending", "running");
        log.record_progress(job_id, 50);
        log.record_transition(job_id, "running", "completed");

        let events = log.events_for(job_id);
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].index, 0);
        assert_eq!(events[1].index, 1);
        assert_eq!(events[2].index, 2);
    }

    #[test]
    fn test_events_by_category() {
        let mut log = JobEventLog::new();
        let job_id = new_job_id();
        log.record_transition(job_id, "pending", "running");
        log.record_progress(job_id, 30);
        log.record_progress(job_id, 70);
        log.record_worker_assigned(job_id, "worker-42");

        let state_events = log.events_by_category(job_id, "state");
        assert_eq!(state_events.len(), 1);

        let progress_events = log.events_by_category(job_id, "progress");
        assert_eq!(progress_events.len(), 2);

        let assignment_events = log.events_by_category(job_id, "assignment");
        assert_eq!(assignment_events.len(), 1);
    }

    #[test]
    fn test_unknown_job_returns_empty_slice() {
        let log = JobEventLog::new();
        let events = log.events_for(Uuid::new_v4());
        assert!(events.is_empty());
    }

    #[test]
    fn test_is_terminal_via_cancelled_event() {
        let mut log = JobEventLog::new();
        let job_id = new_job_id();
        log.record_cancellation(job_id, "admin", Some("test run".into()));
        assert!(log.is_terminal(job_id));
    }

    #[test]
    fn test_is_not_terminal_for_running() {
        let mut log = JobEventLog::new();
        let job_id = new_job_id();
        log.record_transition(job_id, "pending", "running");
        assert!(!log.is_terminal(job_id));
    }

    #[test]
    fn test_export_json_success() {
        let mut log = JobEventLog::new();
        let job_id = new_job_id();
        log.record(job_id, EventActor::System, EventKind::Submitted, None);
        log.record_transition(job_id, "pending", "running");

        let json = log.export_json(job_id).expect("export should succeed");
        assert!(json.contains("Submitted"));
        // Should be NDJSON (two lines for two events)
        assert_eq!(json.lines().count(), 2);
    }

    #[test]
    fn test_export_json_unknown_job_error() {
        let log = JobEventLog::new();
        let result = log.export_json(Uuid::new_v4());
        assert!(matches!(result, Err(EventLogError::UnknownJob(_))));
    }

    #[test]
    fn test_category_summary() {
        let mut log = JobEventLog::new();
        let job_id = new_job_id();
        log.record_transition(job_id, "pending", "running");
        log.record_progress(job_id, 50);
        log.record_progress(job_id, 100);

        let summary = log.category_summary(job_id);
        assert_eq!(*summary.get("state").unwrap_or(&0), 1);
        assert_eq!(*summary.get("progress").unwrap_or(&0), 2);
    }

    #[test]
    fn test_clear_job() {
        let mut log = JobEventLog::new();
        let job_id = new_job_id();
        log.record(job_id, EventActor::System, EventKind::Submitted, None);
        assert_eq!(log.events_for(job_id).len(), 1);
        log.clear_job(job_id);
        assert!(log.events_for(job_id).is_empty());
        assert_eq!(log.tracked_job_count(), 0);
    }

    #[test]
    fn test_retain_jobs() {
        let mut log = JobEventLog::new();
        let id_a = new_job_id();
        let id_b = new_job_id();
        let id_c = new_job_id();
        log.record(id_a, EventActor::System, EventKind::Submitted, None);
        log.record(id_b, EventActor::System, EventKind::Submitted, None);
        log.record(id_c, EventActor::System, EventKind::Submitted, None);

        assert_eq!(log.tracked_job_count(), 3);
        log.retain_jobs(&[id_a, id_b]);
        assert_eq!(log.tracked_job_count(), 2);
        assert!(log.events_for(id_c).is_empty());
    }

    #[test]
    fn test_event_actor_labels() {
        assert_eq!(EventActor::System.label(), "system");
        assert_eq!(EventActor::User("alice".into()).label(), "user:alice");
        assert_eq!(EventActor::Worker("node-1".into()).label(), "worker:node-1");
    }

    #[test]
    fn test_time_range_filter() {
        let mut log = JobEventLog::new();
        let job_id = new_job_id();

        // Record an event then set a range that must contain it
        log.record(job_id, EventActor::System, EventKind::Submitted, None);
        let range = TimeRange::unbounded();
        let filtered = log.events_in_range(job_id, &range);
        assert_eq!(filtered.len(), 1);

        // A range in the far future should exclude the event
        let future = Utc::now() + chrono::Duration::hours(1);
        let range_future = TimeRange::since(future);
        let filtered_future = log.events_in_range(job_id, &range_future);
        assert_eq!(filtered_future.len(), 0);
    }

    #[test]
    fn test_priority_changed_event() {
        let mut log = JobEventLog::new();
        let job_id = new_job_id();
        log.record(
            job_id,
            EventActor::User("ops".into()),
            EventKind::PriorityChanged {
                from: "low".into(),
                to: "high".into(),
            },
            Some("SLA breach risk".into()),
        );
        let events = log.events_for(job_id);
        assert_eq!(events[0].kind.category(), "priority");
        assert_eq!(
            events[0].message.as_deref(),
            Some("SLA breach risk")
        );
    }
}
