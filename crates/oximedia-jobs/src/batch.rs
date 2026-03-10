// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Batch job management for oximedia-jobs.
//!
//! Provides grouping of multiple jobs into a single batch with status
//! tracking, parallel execution limits, and aggregate result reporting.

// ---------------------------------------------------------------------------
// BatchJob
// ---------------------------------------------------------------------------

/// A named batch of jobs that are executed together.
#[derive(Debug, Clone)]
pub struct BatchJob {
    /// Unique batch identifier.
    pub id: u64,
    /// Human-readable batch name.
    pub name: String,
    /// IDs of the jobs contained in this batch.
    pub job_ids: Vec<u64>,
    /// Creation timestamp (milliseconds since epoch).
    pub created_at: u64,
    /// Maximum number of jobs that may run in parallel.
    pub max_parallel: usize,
}

// ---------------------------------------------------------------------------
// BatchStatus
// ---------------------------------------------------------------------------

/// Current execution state of a batch.
#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub enum BatchStatus {
    /// Batch has been created but no jobs have started yet.
    Pending,
    /// Batch is actively running.
    Running {
        /// Number of jobs currently active.
        active: usize,
        /// Number of jobs that have finished (success or failure).
        done: usize,
    },
    /// All jobs in the batch have finished successfully.
    Completed {
        /// Wall-clock time from start to completion in milliseconds.
        elapsed_ms: u64,
    },
    /// One or more jobs failed and the batch is considered failed.
    Failed {
        /// Human-readable failure reason.
        reason: String,
    },
}

// ---------------------------------------------------------------------------
// BatchResult
// ---------------------------------------------------------------------------

/// Aggregate outcome of a completed batch.
#[derive(Debug, Clone)]
pub struct BatchResult {
    /// ID of the batch this result belongs to.
    pub batch_id: u64,
    /// IDs of jobs that completed successfully.
    pub succeeded: Vec<u64>,
    /// Pairs of (job_id, error_message) for failed jobs.
    pub failed: Vec<(u64, String)>,
    /// Total elapsed time for the batch in milliseconds.
    pub total_ms: u64,
}

impl BatchResult {
    /// Fraction of jobs that succeeded, in `[0.0, 1.0]`.
    ///
    /// Returns `1.0` when the batch contains no jobs.
    #[must_use]
    pub fn success_rate(&self) -> f64 {
        let total = self.succeeded.len() + self.failed.len();
        if total == 0 {
            return 1.0;
        }
        self.succeeded.len() as f64 / total as f64
    }

    /// Returns `true` when every job in the batch succeeded.
    #[must_use]
    pub fn is_fully_successful(&self) -> bool {
        self.failed.is_empty()
    }
}

// ---------------------------------------------------------------------------
// BatchManager
// ---------------------------------------------------------------------------

/// Manages the lifecycle of job batches.
pub struct BatchManager {
    batches: Vec<BatchJob>,
    results: Vec<BatchResult>,
    next_batch_id: u64,
    /// Tracks which batch_id each job_id belongs to (job_id → batch_id).
    job_batch_map: std::collections::HashMap<u64, u64>,
    /// Start timestamp per batch (batch_id → start_ms).
    start_times: std::collections::HashMap<u64, u64>,
    /// Running/done counters (batch_id → (active, done)).
    progress: std::collections::HashMap<u64, (usize, usize)>,
}

impl BatchManager {
    /// Create a new, empty batch manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            batches: Vec::new(),
            results: Vec::new(),
            next_batch_id: 1,
            job_batch_map: std::collections::HashMap::new(),
            start_times: std::collections::HashMap::new(),
            progress: std::collections::HashMap::new(),
        }
    }

    /// Create a new batch and return its assigned ID.
    ///
    /// * `name` – human-readable label for the batch.
    /// * `jobs` – IDs of the jobs belonging to this batch.
    /// * `max_parallel` – upper bound on simultaneously active jobs (0 = unlimited).
    pub fn create_batch(&mut self, name: &str, jobs: Vec<u64>, max_parallel: usize) -> u64 {
        let id = self.next_batch_id;
        self.next_batch_id += 1;

        let now_ms = current_ms();
        for &jid in &jobs {
            self.job_batch_map.insert(jid, id);
        }

        let batch = BatchJob {
            id,
            name: name.to_string(),
            job_ids: jobs,
            created_at: now_ms,
            max_parallel,
        };

        self.batches.push(batch);
        id
    }

    /// Record that a job within a batch has completed.
    ///
    /// * `batch_id` – the owning batch.
    /// * `job_id`   – the job that finished.
    /// * `success`  – whether the job succeeded.
    /// * `error`    – optional error message (used when `success == false`).
    pub fn complete_job(&mut self, batch_id: u64, job_id: u64, success: bool, error: Option<&str>) {
        // Lazily record start time on first completion call
        let now_ms = current_ms();
        self.start_times.entry(batch_id).or_insert(now_ms);

        // Find or create result entry
        let result = self.results.iter_mut().find(|r| r.batch_id == batch_id);

        if let Some(r) = result {
            if success {
                r.succeeded.push(job_id);
            } else {
                r.failed
                    .push((job_id, error.unwrap_or("unknown error").to_string()));
            }
            r.total_ms = now_ms.saturating_sub(*self.start_times.get(&batch_id).unwrap_or(&now_ms));
        } else {
            let start = *self.start_times.get(&batch_id).unwrap_or(&now_ms);
            let mut new_result = BatchResult {
                batch_id,
                succeeded: Vec::new(),
                failed: Vec::new(),
                total_ms: now_ms.saturating_sub(start),
            };
            if success {
                new_result.succeeded.push(job_id);
            } else {
                new_result
                    .failed
                    .push((job_id, error.unwrap_or("unknown error").to_string()));
            }
            self.results.push(new_result);
        }

        // Update progress counters
        let prog = self.progress.entry(batch_id).or_insert((0, 0));
        prog.1 += 1; // done count
    }

    /// Query the current status of a batch.
    #[must_use]
    pub fn batch_status(&self, batch_id: u64) -> Option<BatchStatus> {
        let batch = self.batches.iter().find(|b| b.id == batch_id)?;
        let total_jobs = batch.job_ids.len();

        let result = self.results.iter().find(|r| r.batch_id == batch_id);
        let done = result
            .map(|r| r.succeeded.len() + r.failed.len())
            .unwrap_or(0);
        let has_failure = result.map(|r| !r.failed.is_empty()).unwrap_or(false);

        if done == 0 && !self.start_times.contains_key(&batch_id) {
            return Some(BatchStatus::Pending);
        }

        if has_failure && done >= total_jobs {
            let reason = result
                .and_then(|r| r.failed.first())
                .map(|(_, msg)| msg.clone())
                .unwrap_or_else(|| "job failure".to_string());
            return Some(BatchStatus::Failed { reason });
        }

        if done >= total_jobs {
            let elapsed_ms = result.map(|r| r.total_ms).unwrap_or(0);
            return Some(BatchStatus::Completed { elapsed_ms });
        }

        let active = self.progress.get(&batch_id).map(|(a, _)| *a).unwrap_or(0);
        Some(BatchStatus::Running { active, done })
    }

    /// Returns all batches registered with this manager.
    #[must_use]
    pub fn batches(&self) -> &[BatchJob] {
        &self.batches
    }

    /// Returns all results accumulated so far.
    #[must_use]
    pub fn results(&self) -> &[BatchResult] {
        &self.results
    }
}

impl Default for BatchManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Returns a coarse millisecond timestamp (not monotonic; suitable for
/// elapsed-time calculations in tests and approximate bookkeeping).
fn current_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_result_success_rate_empty() {
        let r = BatchResult {
            batch_id: 1,
            succeeded: vec![],
            failed: vec![],
            total_ms: 0,
        };
        assert!((r.success_rate() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_batch_result_success_rate_all_success() {
        let r = BatchResult {
            batch_id: 1,
            succeeded: vec![1, 2, 3],
            failed: vec![],
            total_ms: 100,
        };
        assert!((r.success_rate() - 1.0).abs() < f64::EPSILON);
        assert!(r.is_fully_successful());
    }

    #[test]
    fn test_batch_result_success_rate_partial() {
        let r = BatchResult {
            batch_id: 1,
            succeeded: vec![1, 2],
            failed: vec![(3, "err".to_string())],
            total_ms: 50,
        };
        assert!((r.success_rate() - 2.0 / 3.0).abs() < 1e-9);
        assert!(!r.is_fully_successful());
    }

    #[test]
    fn test_batch_result_all_failed() {
        let r = BatchResult {
            batch_id: 2,
            succeeded: vec![],
            failed: vec![(1, "fail".to_string()), (2, "fail".to_string())],
            total_ms: 10,
        };
        assert!((r.success_rate() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_create_batch_assigns_unique_ids() {
        let mut mgr = BatchManager::new();
        let id1 = mgr.create_batch("batch-a", vec![1, 2], 2);
        let id2 = mgr.create_batch("batch-b", vec![3, 4], 2);
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_batch_status_pending() {
        let mut mgr = BatchManager::new();
        let bid = mgr.create_batch("test", vec![10, 20], 2);
        assert_eq!(mgr.batch_status(bid), Some(BatchStatus::Pending));
    }

    #[test]
    fn test_batch_status_running() {
        let mut mgr = BatchManager::new();
        let bid = mgr.create_batch("test", vec![1, 2, 3], 2);
        // Complete one out of three
        mgr.complete_job(bid, 1, true, None);
        let status = mgr.batch_status(bid).expect("status should be valid");
        match status {
            BatchStatus::Running { done, .. } => assert_eq!(done, 1),
            _ => panic!("Expected Running, got {status:?}"),
        }
    }

    #[test]
    fn test_batch_status_completed() {
        let mut mgr = BatchManager::new();
        let bid = mgr.create_batch("test", vec![1, 2], 2);
        mgr.complete_job(bid, 1, true, None);
        mgr.complete_job(bid, 2, true, None);
        let status = mgr.batch_status(bid).expect("status should be valid");
        assert!(matches!(status, BatchStatus::Completed { .. }));
    }

    #[test]
    fn test_batch_status_failed() {
        let mut mgr = BatchManager::new();
        let bid = mgr.create_batch("test", vec![1, 2], 2);
        mgr.complete_job(bid, 1, false, Some("disk full"));
        mgr.complete_job(bid, 2, true, None);
        let status = mgr.batch_status(bid).expect("status should be valid");
        assert!(matches!(status, BatchStatus::Failed { .. }));
    }

    #[test]
    fn test_batch_status_unknown_id() {
        let mgr = BatchManager::new();
        assert!(mgr.batch_status(9999).is_none());
    }

    #[test]
    fn test_complete_job_records_error_message() {
        let mut mgr = BatchManager::new();
        let bid = mgr.create_batch("test", vec![1], 1);
        mgr.complete_job(bid, 1, false, Some("codec not found"));
        let results = mgr.results();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].failed[0].1, "codec not found");
    }

    #[test]
    fn test_batches_accessor() {
        let mut mgr = BatchManager::new();
        mgr.create_batch("alpha", vec![1], 1);
        mgr.create_batch("beta", vec![2], 1);
        assert_eq!(mgr.batches().len(), 2);
    }

    #[test]
    fn test_default_batch_manager() {
        let mgr = BatchManager::default();
        assert!(mgr.batches().is_empty());
        assert!(mgr.results().is_empty());
    }

    #[test]
    fn test_max_parallel_stored() {
        let mut mgr = BatchManager::new();
        let bid = mgr.create_batch("par-test", vec![1, 2, 3], 5);
        let batch = mgr
            .batches()
            .iter()
            .find(|b| b.id == bid)
            .expect("batch should be valid");
        assert_eq!(batch.max_parallel, 5);
    }
}
