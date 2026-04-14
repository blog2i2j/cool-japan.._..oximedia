// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Job result storage with configurable retention.
//!
//! `JobResultStore` persists the outcome of completed, failed, and cancelled
//! jobs — including output artefact paths, error details, and execution
//! statistics — and enforces configurable retention policies so stale results
//! are automatically evicted.
//!
//! # Design
//! - `JobResult` captures the final state of a job: status, duration, output
//!   paths, error message, and custom metadata.
//! - `RetentionPolicy` controls how long results are kept (by age and/or count).
//! - `JobResultStore` provides O(1) lookup by job ID plus time-ordered
//!   iteration for retention enforcement.
//! - Results can be queried by status, tag intersection, and time range.
//!
//! # Example
//! ```rust
//! use oximedia_jobs::job_result_store::{JobResult, JobResultStore, RetentionPolicy};
//! use oximedia_jobs::job::JobStatus;
//! use uuid::Uuid;
//! use chrono::Duration;
//!
//! let policy = RetentionPolicy {
//!     max_age: Some(Duration::days(7)),
//!     max_results: Some(50_000),
//! };
//! let mut store = JobResultStore::new(policy);
//!
//! let result = JobResult::success(
//!     Uuid::new_v4(),
//!     "encode".into(),
//!     Duration::seconds(42),
//!     vec!["output.mp4".into()],
//! );
//! let id = result.job_id;
//! store.insert(result);
//!
//! assert!(store.get(id).is_some());
//! ```

use crate::job::JobStatus;
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors produced by `JobResultStore`.
#[derive(Debug, thiserror::Error)]
pub enum ResultStoreError {
    /// A result with the given job ID does not exist.
    #[error("Result not found for job {0}")]
    NotFound(Uuid),
    /// A result for the given job ID already exists.
    #[error("Duplicate result for job {0}")]
    Duplicate(Uuid),
}

// ---------------------------------------------------------------------------
// ExecutionStats
// ---------------------------------------------------------------------------

/// Fine-grained execution statistics associated with a job result.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExecutionStats {
    /// Peak CPU usage percentage (0–100).
    pub peak_cpu_pct: Option<f32>,
    /// Peak memory usage in bytes.
    pub peak_memory_bytes: Option<u64>,
    /// Number of retry attempts before the final outcome.
    pub retry_count: u32,
    /// Worker ID that executed the job.
    pub worker_id: Option<String>,
    /// Custom counters (e.g. frames processed, bytes written).
    pub counters: HashMap<String, u64>,
}

impl ExecutionStats {
    /// Create a minimal stats struct.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a custom counter.
    #[must_use]
    pub fn with_counter(mut self, key: impl Into<String>, value: u64) -> Self {
        self.counters.insert(key.into(), value);
        self
    }
}

// ---------------------------------------------------------------------------
// JobResult
// ---------------------------------------------------------------------------

/// The outcome of a single job execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobResult {
    /// ID of the job this result belongs to.
    pub job_id: Uuid,
    /// Human-readable job name.
    pub job_name: String,
    /// Final status of the job.
    pub status: JobStatus,
    /// Wall-clock duration of the job.
    pub duration: Duration,
    /// Paths of output artefacts produced (if any).
    pub output_paths: Vec<String>,
    /// Error description if the job failed.
    pub error_message: Option<String>,
    /// Timestamp when the result was recorded.
    pub recorded_at: DateTime<Utc>,
    /// Job tags at completion time.
    pub tags: Vec<String>,
    /// Execution statistics.
    pub stats: ExecutionStats,
    /// Arbitrary string metadata.
    pub metadata: HashMap<String, String>,
}

impl JobResult {
    /// Create a success result.
    #[must_use]
    pub fn success(
        job_id: Uuid,
        job_name: String,
        duration: Duration,
        output_paths: Vec<String>,
    ) -> Self {
        Self {
            job_id,
            job_name,
            status: JobStatus::Completed,
            duration,
            output_paths,
            error_message: None,
            recorded_at: Utc::now(),
            tags: Vec::new(),
            stats: ExecutionStats::default(),
            metadata: HashMap::new(),
        }
    }

    /// Create a failure result.
    #[must_use]
    pub fn failure(
        job_id: Uuid,
        job_name: String,
        duration: Duration,
        error: impl Into<String>,
    ) -> Self {
        Self {
            job_id,
            job_name,
            status: JobStatus::Failed,
            duration,
            output_paths: Vec::new(),
            error_message: Some(error.into()),
            recorded_at: Utc::now(),
            tags: Vec::new(),
            stats: ExecutionStats::default(),
            metadata: HashMap::new(),
        }
    }

    /// Create a cancellation result.
    #[must_use]
    pub fn cancelled(job_id: Uuid, job_name: String, duration: Duration) -> Self {
        Self {
            job_id,
            job_name,
            status: JobStatus::Cancelled,
            duration,
            output_paths: Vec::new(),
            error_message: None,
            recorded_at: Utc::now(),
            tags: Vec::new(),
            stats: ExecutionStats::default(),
            metadata: HashMap::new(),
        }
    }

    /// Attach tags to this result.
    #[must_use]
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    /// Attach execution stats.
    #[must_use]
    pub fn with_stats(mut self, stats: ExecutionStats) -> Self {
        self.stats = stats;
        self
    }

    /// Add a metadata key/value pair.
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Duration in seconds as an `f64`.
    #[must_use]
    pub fn duration_secs(&self) -> f64 {
        self.duration.num_milliseconds() as f64 / 1000.0
    }
}

// ---------------------------------------------------------------------------
// RetentionPolicy
// ---------------------------------------------------------------------------

/// Controls how long job results are kept in the store.
#[derive(Debug, Clone)]
pub struct RetentionPolicy {
    /// Remove results older than this duration.  `None` disables age-based eviction.
    pub max_age: Option<Duration>,
    /// Keep at most this many results.  `None` disables count-based eviction.
    pub max_results: Option<usize>,
}

impl Default for RetentionPolicy {
    fn default() -> Self {
        Self {
            max_age: Some(Duration::days(90)),
            max_results: Some(100_000),
        }
    }
}

// ---------------------------------------------------------------------------
// ResultStoreStats
// ---------------------------------------------------------------------------

/// Aggregate statistics for a `JobResultStore`.
#[derive(Debug, Clone, Default)]
pub struct ResultStoreStats {
    /// Total results ever inserted.
    pub total_inserted: u64,
    /// Total results removed by the retention policy.
    pub total_evicted: u64,
    /// Current number of results in the store.
    pub current_count: usize,
    /// Total successful results (status = Completed).
    pub success_count: u64,
    /// Total failed results (status = Failed).
    pub failure_count: u64,
    /// Total cancelled results.
    pub cancelled_count: u64,
    /// Sum of durations for computing average (in milliseconds).
    pub total_duration_ms: i64,
}

impl ResultStoreStats {
    /// Average execution duration across all stored results.
    #[must_use]
    pub fn average_duration(&self) -> Option<Duration> {
        if self.total_inserted == 0 {
            None
        } else {
            Some(Duration::milliseconds(
                self.total_duration_ms / self.total_inserted as i64,
            ))
        }
    }

    /// Success rate (0.0–1.0) over all results ever inserted.
    #[must_use]
    pub fn success_rate(&self) -> f64 {
        if self.total_inserted == 0 {
            0.0
        } else {
            self.success_count as f64 / self.total_inserted as f64
        }
    }
}

// ---------------------------------------------------------------------------
// JobResultStore
// ---------------------------------------------------------------------------

/// Durable in-memory store for job execution results with retention enforcement.
pub struct JobResultStore {
    /// Retention policy applied on insert and explicit prune.
    policy: RetentionPolicy,
    /// Primary store: job_id → result.
    store: HashMap<Uuid, JobResult>,
    /// Insertion-order index for LRU/age eviction.
    order: Vec<(Uuid, DateTime<Utc>)>,
    /// Aggregate statistics.
    stats: ResultStoreStats,
}

impl JobResultStore {
    /// Create a new `JobResultStore` with the given retention policy.
    #[must_use]
    pub fn new(policy: RetentionPolicy) -> Self {
        Self {
            policy,
            store: HashMap::new(),
            order: Vec::new(),
            stats: ResultStoreStats::default(),
        }
    }

    /// Insert a result.
    ///
    /// If a result for the same job ID already exists it is **replaced**.
    /// After insertion, the retention policy is enforced.
    pub fn insert(&mut self, result: JobResult) {
        let id = result.job_id;
        let recorded_at = result.recorded_at;
        let duration_ms = result.duration.num_milliseconds();

        // Track stats before possible eviction.
        match result.status {
            JobStatus::Completed => self.stats.success_count += 1,
            JobStatus::Failed => self.stats.failure_count += 1,
            JobStatus::Cancelled => self.stats.cancelled_count += 1,
            _ => {}
        }
        self.stats.total_inserted += 1;
        self.stats.total_duration_ms += duration_ms;

        // Replace existing entry if present.
        if self.store.contains_key(&id) {
            self.order.retain(|(oid, _)| *oid != id);
        }

        self.store.insert(id, result);
        self.order.push((id, recorded_at));

        // Enforce retention.
        self.enforce_retention();
        self.stats.current_count = self.store.len();
    }

    /// Retrieve a result by job ID.
    #[must_use]
    pub fn get(&self, job_id: Uuid) -> Option<&JobResult> {
        self.store.get(&job_id)
    }

    /// Remove a result by job ID.
    ///
    /// # Errors
    ///
    /// Returns [`ResultStoreError::NotFound`] if no result exists.
    pub fn remove(&mut self, job_id: Uuid) -> Result<JobResult, ResultStoreError> {
        let result = self
            .store
            .remove(&job_id)
            .ok_or(ResultStoreError::NotFound(job_id))?;
        self.order.retain(|(id, _)| *id != job_id);
        self.stats.current_count = self.store.len();
        Ok(result)
    }

    /// Number of results currently in the store.
    #[must_use]
    pub fn len(&self) -> usize {
        self.store.len()
    }

    /// Whether the store is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }

    /// Query results by status.
    #[must_use]
    pub fn by_status(&self, status: JobStatus) -> Vec<&JobResult> {
        self.store
            .values()
            .filter(|r| r.status == status)
            .collect()
    }

    /// Query results for jobs that carry at least one of the given tags.
    #[must_use]
    pub fn by_any_tag(&self, tags: &[&str]) -> Vec<&JobResult> {
        self.store
            .values()
            .filter(|r| tags.iter().any(|t| r.tags.iter().any(|rt| rt == t)))
            .collect()
    }

    /// Query results recorded within the given time range (inclusive).
    #[must_use]
    pub fn by_time_range(
        &self,
        from: DateTime<Utc>,
        to: DateTime<Utc>,
    ) -> Vec<&JobResult> {
        self.store
            .values()
            .filter(|r| r.recorded_at >= from && r.recorded_at <= to)
            .collect()
    }

    /// Return results sorted by duration descending (slowest first).
    #[must_use]
    pub fn slowest(&self, limit: usize) -> Vec<&JobResult> {
        let mut sorted: Vec<&JobResult> = self.store.values().collect();
        sorted.sort_by(|a, b| b.duration.cmp(&a.duration));
        sorted.truncate(limit);
        sorted
    }

    /// Explicitly enforce the retention policy, evicting stale entries.
    /// Returns the number of evicted entries.
    pub fn prune(&mut self) -> usize {
        let before = self.store.len();
        self.enforce_retention();
        let evicted = before - self.store.len();
        self.stats.current_count = self.store.len();
        evicted
    }

    /// Aggregate statistics for this store.
    #[must_use]
    pub fn stats(&self) -> &ResultStoreStats {
        &self.stats
    }

    /// Iterate over all results.
    pub fn iter(&self) -> impl Iterator<Item = &JobResult> {
        self.store.values()
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    fn enforce_retention(&mut self) {
        // Age-based: remove entries older than max_age.
        if let Some(max_age) = self.policy.max_age {
            let cutoff = Utc::now() - max_age;
            let stale: Vec<Uuid> = self
                .order
                .iter()
                .filter(|(_, ts)| *ts < cutoff)
                .map(|(id, _)| *id)
                .collect();
            for id in stale {
                self.store.remove(&id);
                self.order.retain(|(oid, _)| *oid != id);
                self.stats.total_evicted += 1;
            }
        }

        // Count-based: remove oldest entries beyond max_results.
        if let Some(max_results) = self.policy.max_results {
            while self.order.len() > max_results {
                if let Some((oldest_id, _)) = self.order.first().copied() {
                    self.store.remove(&oldest_id);
                    self.order.remove(0);
                    self.stats.total_evicted += 1;
                } else {
                    break;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_success(name: &str) -> JobResult {
        JobResult::success(
            Uuid::new_v4(),
            name.into(),
            Duration::seconds(10),
            vec!["out.mp4".into()],
        )
    }

    fn make_failure(name: &str) -> JobResult {
        JobResult::failure(
            Uuid::new_v4(),
            name.into(),
            Duration::seconds(5),
            "codec error",
        )
    }

    #[test]
    fn test_insert_and_get() {
        let mut store = JobResultStore::new(RetentionPolicy::default());
        let result = make_success("encode");
        let id = result.job_id;
        store.insert(result);
        assert!(store.get(id).is_some());
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_remove_result() {
        let mut store = JobResultStore::new(RetentionPolicy::default());
        let result = make_failure("bad-job");
        let id = result.job_id;
        store.insert(result);
        let removed = store.remove(id).expect("should remove");
        assert_eq!(removed.job_name, "bad-job");
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn test_remove_not_found() {
        let mut store = JobResultStore::new(RetentionPolicy::default());
        let res = store.remove(Uuid::new_v4());
        assert!(matches!(res, Err(ResultStoreError::NotFound(_))));
    }

    #[test]
    fn test_by_status() {
        let mut store = JobResultStore::new(RetentionPolicy::default());
        store.insert(make_success("s1"));
        store.insert(make_success("s2"));
        store.insert(make_failure("f1"));

        let completed = store.by_status(JobStatus::Completed);
        let failed = store.by_status(JobStatus::Failed);
        assert_eq!(completed.len(), 2);
        assert_eq!(failed.len(), 1);
    }

    #[test]
    fn test_by_tag() {
        let mut store = JobResultStore::new(RetentionPolicy::default());
        let r1 = make_success("tagged").with_tags(vec!["video".into()]);
        let r2 = make_success("untagged");
        store.insert(r1);
        store.insert(r2);
        let found = store.by_any_tag(&["video"]);
        assert_eq!(found.len(), 1);
    }

    #[test]
    fn test_count_retention() {
        let policy = RetentionPolicy {
            max_age: None,
            max_results: Some(3),
        };
        let mut store = JobResultStore::new(policy);
        for i in 0..5 {
            store.insert(make_success(&format!("job-{i}")));
        }
        assert_eq!(store.len(), 3);
        assert_eq!(store.stats().total_evicted, 2);
    }

    #[test]
    fn test_age_retention() {
        let policy = RetentionPolicy {
            max_age: Some(Duration::days(1)),
            max_results: None,
        };
        let mut store = JobResultStore::new(policy);

        // Insert a recent result first.
        store.insert(make_success("new-job"));

        // Insert a result that appears old — it will be evicted immediately on insert.
        let mut old = make_success("old-job");
        old.recorded_at = Utc::now() - Duration::days(5);
        store.insert(old);

        // The old result should have been evicted during insert; only new-job remains.
        assert_eq!(store.len(), 1);
        assert!(store.stats().total_evicted >= 1);

        // A subsequent explicit prune finds nothing further to remove.
        let pruned = store.prune();
        assert_eq!(pruned, 0);
    }

    #[test]
    fn test_slowest_returns_in_order() {
        let mut store = JobResultStore::new(RetentionPolicy::default());
        let fast = JobResult::success(
            Uuid::new_v4(),
            "fast".into(),
            Duration::seconds(1),
            vec![],
        );
        let slow = JobResult::success(
            Uuid::new_v4(),
            "slow".into(),
            Duration::seconds(100),
            vec![],
        );
        store.insert(fast);
        store.insert(slow);
        let top = store.slowest(1);
        assert_eq!(top[0].job_name, "slow");
    }

    #[test]
    fn test_success_rate_stats() {
        let mut store = JobResultStore::new(RetentionPolicy::default());
        store.insert(make_success("s1"));
        store.insert(make_success("s2"));
        store.insert(make_failure("f1"));

        let stats = store.stats();
        let rate = stats.success_rate();
        // 2 out of 3 successful
        assert!((rate - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_execution_stats_counters() {
        let stats = ExecutionStats::new()
            .with_counter("frames", 1080)
            .with_counter("bytes", 1_048_576);
        assert_eq!(stats.counters.get("frames"), Some(&1080));
        assert_eq!(stats.counters.get("bytes"), Some(&1_048_576));
    }

    #[test]
    fn test_by_time_range() {
        let mut store = JobResultStore::new(RetentionPolicy::default());
        let now = Utc::now();
        let mut r1 = make_success("r1");
        r1.recorded_at = now - Duration::hours(2);
        let mut r2 = make_success("r2");
        r2.recorded_at = now - Duration::hours(1);
        let mut r3 = make_success("r3");
        r3.recorded_at = now;
        store.insert(r1);
        store.insert(r2);
        store.insert(r3);

        let from = now - Duration::minutes(90);
        let to = now + Duration::minutes(1);
        let found = store.by_time_range(from, to);
        assert_eq!(found.len(), 2);
    }
}
