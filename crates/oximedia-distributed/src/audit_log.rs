#![allow(dead_code)]
//! Audit logging for coordinator state changes.
//!
//! Records every significant state mutation in the distributed coordinator:
//! job submissions, cancellations, worker joins/leaves, leader elections,
//! configuration changes, and snapshot operations.
//!
//! The audit log is append-only, indexed by monotonic sequence number, and
//! supports querying by time range, event type, and actor.

use std::collections::VecDeque;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use uuid::Uuid;

// ---------------------------------------------------------------------------
// AuditEventKind
// ---------------------------------------------------------------------------

/// Categories of auditable events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AuditEventKind {
    /// A new job was submitted.
    JobSubmitted,
    /// A job was cancelled.
    JobCancelled,
    /// A job completed successfully.
    JobCompleted,
    /// A job failed.
    JobFailed,
    /// A job was preempted by a higher-priority job.
    JobPreempted,
    /// A job was reassigned to a different worker.
    JobReassigned,
    /// A worker joined the cluster.
    WorkerJoined,
    /// A worker left (or was removed from) the cluster.
    WorkerLeft,
    /// A leader election completed.
    LeaderElected,
    /// A snapshot was created.
    SnapshotCreated,
    /// A snapshot was restored.
    SnapshotRestored,
    /// Configuration was changed.
    ConfigChanged,
    /// A segment merge was completed.
    SegmentMerged,
    /// A custom/application-defined event.
    Custom,
}

impl fmt::Display for AuditEventKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::JobSubmitted => write!(f, "JobSubmitted"),
            Self::JobCancelled => write!(f, "JobCancelled"),
            Self::JobCompleted => write!(f, "JobCompleted"),
            Self::JobFailed => write!(f, "JobFailed"),
            Self::JobPreempted => write!(f, "JobPreempted"),
            Self::JobReassigned => write!(f, "JobReassigned"),
            Self::WorkerJoined => write!(f, "WorkerJoined"),
            Self::WorkerLeft => write!(f, "WorkerLeft"),
            Self::LeaderElected => write!(f, "LeaderElected"),
            Self::SnapshotCreated => write!(f, "SnapshotCreated"),
            Self::SnapshotRestored => write!(f, "SnapshotRestored"),
            Self::ConfigChanged => write!(f, "ConfigChanged"),
            Self::SegmentMerged => write!(f, "SegmentMerged"),
            Self::Custom => write!(f, "Custom"),
        }
    }
}

// ---------------------------------------------------------------------------
// AuditSeverity
// ---------------------------------------------------------------------------

/// Severity level for audit events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AuditSeverity {
    /// Informational — normal operations.
    Info,
    /// Warning — unexpected but non-critical.
    Warning,
    /// Error — something failed.
    Error,
    /// Critical — system integrity may be at risk.
    Critical,
}

impl fmt::Display for AuditSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Info => write!(f, "INFO"),
            Self::Warning => write!(f, "WARN"),
            Self::Error => write!(f, "ERROR"),
            Self::Critical => write!(f, "CRITICAL"),
        }
    }
}

// ---------------------------------------------------------------------------
// AuditEntry
// ---------------------------------------------------------------------------

/// A single entry in the audit log.
#[derive(Debug, Clone)]
pub struct AuditEntry {
    /// Monotonically increasing sequence number.
    pub sequence: u64,
    /// Wall-clock timestamp (microseconds since Unix epoch).
    pub timestamp_us: u64,
    /// Kind of event.
    pub kind: AuditEventKind,
    /// Severity level.
    pub severity: AuditSeverity,
    /// Actor that caused the event (e.g., worker ID, coordinator ID, user).
    pub actor: String,
    /// Optional target entity (e.g., job ID, worker ID).
    pub target: Option<String>,
    /// Human-readable description of the event.
    pub message: String,
    /// Optional structured metadata as key-value pairs.
    pub metadata: Vec<(String, String)>,
}

impl fmt::Display for AuditEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[#{} {} {} {}] actor={} {}",
            self.sequence, self.timestamp_us, self.severity, self.kind, self.actor, self.message
        )
    }
}

// ---------------------------------------------------------------------------
// AuditEntryBuilder
// ---------------------------------------------------------------------------

/// Builder for constructing [`AuditEntry`] instances.
pub struct AuditEntryBuilder {
    kind: AuditEventKind,
    severity: AuditSeverity,
    actor: String,
    target: Option<String>,
    message: String,
    metadata: Vec<(String, String)>,
}

impl AuditEntryBuilder {
    /// Create a new builder with the given event kind and actor.
    pub fn new(kind: AuditEventKind, actor: impl Into<String>) -> Self {
        Self {
            kind,
            severity: AuditSeverity::Info,
            actor: actor.into(),
            target: None,
            message: String::new(),
            metadata: Vec::new(),
        }
    }

    /// Set the severity level.
    pub fn severity(mut self, severity: AuditSeverity) -> Self {
        self.severity = severity;
        self
    }

    /// Set the target entity.
    pub fn target(mut self, target: impl Into<String>) -> Self {
        self.target = Some(target.into());
        self
    }

    /// Set the message.
    pub fn message(mut self, message: impl Into<String>) -> Self {
        self.message = message.into();
        self
    }

    /// Add a metadata key-value pair.
    pub fn meta(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.push((key.into(), value.into()));
        self
    }

    /// Build the entry. The sequence number and timestamp will be set by the
    /// [`AuditLog`] when the entry is appended.
    fn build(self, sequence: u64, timestamp_us: u64) -> AuditEntry {
        AuditEntry {
            sequence,
            timestamp_us,
            kind: self.kind,
            severity: self.severity,
            actor: self.actor,
            target: self.target,
            message: self.message,
            metadata: self.metadata,
        }
    }
}

// ---------------------------------------------------------------------------
// AuditLogConfig
// ---------------------------------------------------------------------------

/// Configuration for the audit log.
#[derive(Debug, Clone)]
pub struct AuditLogConfig {
    /// Maximum number of entries to retain in memory. When exceeded, the
    /// oldest entries are discarded (ring buffer behaviour).
    pub max_entries: usize,
    /// Minimum severity level to record. Events below this level are dropped.
    pub min_severity: AuditSeverity,
}

impl Default for AuditLogConfig {
    fn default() -> Self {
        Self {
            max_entries: 100_000,
            min_severity: AuditSeverity::Info,
        }
    }
}

// ---------------------------------------------------------------------------
// AuditLog
// ---------------------------------------------------------------------------

/// An append-only, bounded audit log for coordinator state changes.
///
/// Thread-safety note: this struct is *not* internally synchronised. In a
/// multi-threaded context, wrap it in `Arc<Mutex<AuditLog>>` or similar.
pub struct AuditLog {
    config: AuditLogConfig,
    /// Monotonic sequence counter.
    next_sequence: AtomicU64,
    /// Ring buffer of entries.
    entries: VecDeque<AuditEntry>,
    /// Total number of entries ever appended (including evicted ones).
    total_appended: u64,
}

impl AuditLog {
    /// Create a new audit log with the given configuration.
    pub fn new(config: AuditLogConfig) -> Self {
        Self {
            next_sequence: AtomicU64::new(1),
            entries: VecDeque::with_capacity(config.max_entries.min(4096)),
            total_appended: 0,
            config,
        }
    }

    /// Create a new audit log with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(AuditLogConfig::default())
    }

    /// Append an entry built from a builder.
    ///
    /// Returns the assigned sequence number, or `None` if the event was
    /// filtered out by the minimum severity setting.
    pub fn append(&mut self, builder: AuditEntryBuilder) -> Option<u64> {
        if builder.severity < self.config.min_severity {
            return None;
        }

        let seq = self.next_sequence.fetch_add(1, Ordering::Relaxed);
        let timestamp_us = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_micros() as u64;

        let entry = builder.build(seq, timestamp_us);

        // Evict oldest if at capacity.
        if self.entries.len() >= self.config.max_entries {
            self.entries.pop_front();
        }

        self.entries.push_back(entry);
        self.total_appended += 1;
        Some(seq)
    }

    /// Convenience: log a job submission.
    pub fn log_job_submitted(&mut self, job_id: Uuid, actor: &str, codec: &str) -> Option<u64> {
        self.append(
            AuditEntryBuilder::new(AuditEventKind::JobSubmitted, actor)
                .target(job_id.to_string())
                .message(format!("Job {job_id} submitted"))
                .meta("codec", codec),
        )
    }

    /// Convenience: log a job cancellation.
    pub fn log_job_cancelled(&mut self, job_id: Uuid, actor: &str, reason: &str) -> Option<u64> {
        self.append(
            AuditEntryBuilder::new(AuditEventKind::JobCancelled, actor)
                .severity(AuditSeverity::Warning)
                .target(job_id.to_string())
                .message(format!("Job {job_id} cancelled: {reason}")),
        )
    }

    /// Convenience: log a job failure.
    pub fn log_job_failed(&mut self, job_id: Uuid, actor: &str, error: &str) -> Option<u64> {
        self.append(
            AuditEntryBuilder::new(AuditEventKind::JobFailed, actor)
                .severity(AuditSeverity::Error)
                .target(job_id.to_string())
                .message(format!("Job {job_id} failed: {error}")),
        )
    }

    /// Convenience: log a worker joining the cluster.
    pub fn log_worker_joined(&mut self, worker_id: &str, addr: &str) -> Option<u64> {
        self.append(
            AuditEntryBuilder::new(AuditEventKind::WorkerJoined, "coordinator")
                .target(worker_id)
                .message(format!("Worker {worker_id} joined from {addr}"))
                .meta("address", addr),
        )
    }

    /// Convenience: log a worker leaving the cluster.
    pub fn log_worker_left(&mut self, worker_id: &str, reason: &str) -> Option<u64> {
        self.append(
            AuditEntryBuilder::new(AuditEventKind::WorkerLeft, "coordinator")
                .severity(AuditSeverity::Warning)
                .target(worker_id)
                .message(format!("Worker {worker_id} left: {reason}")),
        )
    }

    /// Convenience: log a leader election.
    pub fn log_leader_elected(&mut self, leader_id: &str, term: u64) -> Option<u64> {
        self.append(
            AuditEntryBuilder::new(AuditEventKind::LeaderElected, leader_id)
                .message(format!("Node {leader_id} elected leader for term {term}"))
                .meta("term", term.to_string()),
        )
    }

    /// Number of entries currently in the log.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the log is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Total entries ever appended (including evicted).
    pub fn total_appended(&self) -> u64 {
        self.total_appended
    }

    /// Get an entry by sequence number.
    pub fn get_by_sequence(&self, seq: u64) -> Option<&AuditEntry> {
        self.entries.iter().find(|e| e.sequence == seq)
    }

    /// Return the most recent `n` entries (newest last).
    pub fn recent(&self, n: usize) -> Vec<&AuditEntry> {
        let start = self.entries.len().saturating_sub(n);
        self.entries.iter().skip(start).collect()
    }

    /// Query entries by event kind.
    pub fn query_by_kind(&self, kind: AuditEventKind) -> Vec<&AuditEntry> {
        self.entries.iter().filter(|e| e.kind == kind).collect()
    }

    /// Query entries by actor.
    pub fn query_by_actor(&self, actor: &str) -> Vec<&AuditEntry> {
        self.entries.iter().filter(|e| e.actor == actor).collect()
    }

    /// Query entries by target.
    pub fn query_by_target(&self, target: &str) -> Vec<&AuditEntry> {
        self.entries
            .iter()
            .filter(|e| e.target.as_deref() == Some(target))
            .collect()
    }

    /// Query entries by severity at or above the given level.
    pub fn query_by_min_severity(&self, min: AuditSeverity) -> Vec<&AuditEntry> {
        self.entries.iter().filter(|e| e.severity >= min).collect()
    }

    /// Query entries within a time range (microseconds since Unix epoch).
    pub fn query_by_time_range(&self, start_us: u64, end_us: u64) -> Vec<&AuditEntry> {
        self.entries
            .iter()
            .filter(|e| e.timestamp_us >= start_us && e.timestamp_us <= end_us)
            .collect()
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Export all entries as a vector (for serialization or transfer).
    pub fn export_all(&self) -> Vec<AuditEntry> {
        self.entries.iter().cloned().collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_append_and_len() {
        let mut log = AuditLog::with_defaults();
        assert!(log.is_empty());

        let seq = log.append(
            AuditEntryBuilder::new(AuditEventKind::JobSubmitted, "user1").message("test job"),
        );
        assert!(seq.is_some());
        assert_eq!(log.len(), 1);
        assert!(!log.is_empty());
    }

    #[test]
    fn test_sequence_numbers_are_monotonic() {
        let mut log = AuditLog::with_defaults();
        let s1 = log
            .append(AuditEntryBuilder::new(AuditEventKind::JobSubmitted, "a").message("1"))
            .expect("append");
        let s2 = log
            .append(AuditEntryBuilder::new(AuditEventKind::JobCancelled, "a").message("2"))
            .expect("append");
        let s3 = log
            .append(AuditEntryBuilder::new(AuditEventKind::JobCompleted, "a").message("3"))
            .expect("append");
        assert!(s1 < s2);
        assert!(s2 < s3);
    }

    #[test]
    fn test_max_entries_eviction() {
        let mut log = AuditLog::new(AuditLogConfig {
            max_entries: 3,
            min_severity: AuditSeverity::Info,
        });

        for i in 0..5 {
            log.append(
                AuditEntryBuilder::new(AuditEventKind::Custom, "actor")
                    .message(format!("event {i}")),
            );
        }

        assert_eq!(log.len(), 3);
        assert_eq!(log.total_appended(), 5);
        // Oldest two should have been evicted
        let entries = log.export_all();
        assert!(entries[0].message.contains("event 2"));
    }

    #[test]
    fn test_min_severity_filter() {
        let mut log = AuditLog::new(AuditLogConfig {
            max_entries: 100,
            min_severity: AuditSeverity::Warning,
        });

        // Info event should be filtered out
        let seq = log.append(
            AuditEntryBuilder::new(AuditEventKind::JobSubmitted, "user")
                .severity(AuditSeverity::Info)
                .message("should be dropped"),
        );
        assert!(seq.is_none());
        assert_eq!(log.len(), 0);

        // Warning event should be accepted
        let seq = log.append(
            AuditEntryBuilder::new(AuditEventKind::JobCancelled, "user")
                .severity(AuditSeverity::Warning)
                .message("should be kept"),
        );
        assert!(seq.is_some());
        assert_eq!(log.len(), 1);
    }

    #[test]
    fn test_query_by_kind() {
        let mut log = AuditLog::with_defaults();
        log.log_job_submitted(Uuid::new_v4(), "user", "av1");
        log.log_job_cancelled(Uuid::new_v4(), "user", "timeout");
        log.log_job_submitted(Uuid::new_v4(), "user", "vp9");

        let submitted = log.query_by_kind(AuditEventKind::JobSubmitted);
        assert_eq!(submitted.len(), 2);

        let cancelled = log.query_by_kind(AuditEventKind::JobCancelled);
        assert_eq!(cancelled.len(), 1);
    }

    #[test]
    fn test_query_by_actor() {
        let mut log = AuditLog::with_defaults();
        log.log_job_submitted(Uuid::new_v4(), "alice", "av1");
        log.log_job_submitted(Uuid::new_v4(), "bob", "vp9");
        log.log_job_submitted(Uuid::new_v4(), "alice", "opus");

        let alice_events = log.query_by_actor("alice");
        assert_eq!(alice_events.len(), 2);
    }

    #[test]
    fn test_query_by_target() {
        let mut log = AuditLog::with_defaults();
        let job_id = Uuid::new_v4();
        log.log_job_submitted(job_id, "user", "av1");
        log.log_job_cancelled(job_id, "user", "user request");
        log.log_job_submitted(Uuid::new_v4(), "user", "vp9");

        let target_events = log.query_by_target(&job_id.to_string());
        assert_eq!(target_events.len(), 2);
    }

    #[test]
    fn test_query_by_min_severity() {
        let mut log = AuditLog::with_defaults();
        log.log_job_submitted(Uuid::new_v4(), "user", "av1"); // Info
        log.log_job_cancelled(Uuid::new_v4(), "user", "timeout"); // Warning
        log.log_job_failed(Uuid::new_v4(), "user", "crash"); // Error

        let warnings_plus = log.query_by_min_severity(AuditSeverity::Warning);
        assert_eq!(warnings_plus.len(), 2);

        let errors_only = log.query_by_min_severity(AuditSeverity::Error);
        assert_eq!(errors_only.len(), 1);
    }

    #[test]
    fn test_recent_entries() {
        let mut log = AuditLog::with_defaults();
        for i in 0..10 {
            log.append(
                AuditEntryBuilder::new(AuditEventKind::Custom, "actor")
                    .message(format!("event {i}")),
            );
        }

        let recent = log.recent(3);
        assert_eq!(recent.len(), 3);
        assert!(recent[0].message.contains("event 7"));
        assert!(recent[2].message.contains("event 9"));
    }

    #[test]
    fn test_get_by_sequence() {
        let mut log = AuditLog::with_defaults();
        let seq = log
            .append(
                AuditEntryBuilder::new(AuditEventKind::LeaderElected, "node-1").message("elected"),
            )
            .expect("append");

        let entry = log.get_by_sequence(seq).expect("found");
        assert_eq!(entry.kind, AuditEventKind::LeaderElected);
        assert!(log.get_by_sequence(99999).is_none());
    }

    #[test]
    fn test_clear() {
        let mut log = AuditLog::with_defaults();
        log.log_job_submitted(Uuid::new_v4(), "user", "av1");
        log.log_job_submitted(Uuid::new_v4(), "user", "vp9");
        assert_eq!(log.len(), 2);

        log.clear();
        assert!(log.is_empty());
        assert_eq!(log.total_appended(), 2); // total is preserved
    }

    #[test]
    fn test_convenience_worker_and_leader_logs() {
        let mut log = AuditLog::with_defaults();
        log.log_worker_joined("worker-1", "192.168.1.10:50052");
        log.log_worker_left("worker-1", "heartbeat timeout");
        log.log_leader_elected("node-3", 42);

        assert_eq!(log.len(), 3);

        let worker_events = log.query_by_kind(AuditEventKind::WorkerJoined);
        assert_eq!(worker_events.len(), 1);

        let leader_events = log.query_by_kind(AuditEventKind::LeaderElected);
        assert_eq!(leader_events.len(), 1);
        assert!(leader_events[0].message.contains("term 42"));
    }

    #[test]
    fn test_builder_with_metadata() {
        let mut log = AuditLog::with_defaults();
        let seq = log
            .append(
                AuditEntryBuilder::new(AuditEventKind::ConfigChanged, "admin")
                    .severity(AuditSeverity::Warning)
                    .target("cluster-config")
                    .message("max_retries changed")
                    .meta("old_value", "3")
                    .meta("new_value", "5"),
            )
            .expect("append");

        let entry = log.get_by_sequence(seq).expect("found");
        assert_eq!(entry.metadata.len(), 2);
        assert_eq!(
            entry.metadata[0],
            ("old_value".to_string(), "3".to_string())
        );
        assert_eq!(entry.severity, AuditSeverity::Warning);
    }
}
