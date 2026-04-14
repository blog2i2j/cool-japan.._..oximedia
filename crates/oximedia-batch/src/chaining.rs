//! Job chaining — dependency-aware sequential job dispatch.
//!
//! [`BatchJobChain`] maintains an ordered list of jobs with optional `depends_on`
//! constraints.  Only jobs whose dependency (if any) has been completed are
//! considered *ready*.  Calling `next_ready()` pops and returns the first job
//! that is ready to run.
//!
//! # Example
//!
//! ```no_run
//! use oximedia_batch::chaining::BatchJobChain;
//! use oximedia_batch::job::{BatchJob, BatchOperation};
//!
//! let job_a = BatchJob::new("clip-transcode".into(), BatchOperation::Transcode { preset: "h264".into() });
//! let id_a = job_a.id.as_str().to_string();
//! let job_b = BatchJob::new("qc-check".into(), BatchOperation::QualityCheck { profile: "broadcast".into() });
//!
//! let mut chain = BatchJobChain::new();
//! chain.add(job_a, None);
//! chain.add(job_b, Some(id_a));
//!
//! let first = chain.next_ready().unwrap();
//! chain.mark_complete(first.id.as_str());
//! let second = chain.next_ready().unwrap();
//! assert_eq!(second.name, "qc-check");
//! ```

#![allow(dead_code)]

use crate::job::BatchJob;
use std::collections::HashSet;

// ---------------------------------------------------------------------------
// Internal chain entry
// ---------------------------------------------------------------------------

struct ChainEntry {
    job: BatchJob,
    /// String ID of the job that must complete before this one can run.
    depends_on: Option<String>,
    dispatched: bool,
}

// ---------------------------------------------------------------------------
// BatchJobChain
// ---------------------------------------------------------------------------

/// Dependency-aware job chain.
///
/// Jobs are stored in insertion order.  Each job may optionally declare a
/// predecessor by its string ID.  A job is *ready* when:
/// - it has not been dispatched yet, **and**
/// - it has no dependency, **or** its dependency ID is in the `completed` set.
pub struct BatchJobChain {
    entries: Vec<ChainEntry>,
    completed: HashSet<String>,
}

impl BatchJobChain {
    /// Create a new, empty chain.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            completed: HashSet::new(),
        }
    }

    /// Append a job to the chain.
    ///
    /// `depends_on` is the string ID of the job that must have been marked
    /// complete (via `mark_complete`) before this job can be dispatched.
    pub fn add(&mut self, job: BatchJob, depends_on: Option<String>) {
        self.entries.push(ChainEntry {
            job,
            depends_on,
            dispatched: false,
        });
    }

    /// Pop the first job that is ready to run.
    ///
    /// A job is ready when it has not been dispatched and its dependency
    /// (if any) has been marked complete.
    ///
    /// Returns `None` when no ready jobs remain.
    pub fn next_ready(&mut self) -> Option<BatchJob> {
        let completed = &self.completed;
        let pos = self.entries.iter().position(|e| {
            !e.dispatched
                && e.depends_on
                    .as_deref()
                    .map(|dep| completed.contains(dep))
                    .unwrap_or(true)
        })?;

        let entry = &mut self.entries[pos];
        entry.dispatched = true;
        Some(entry.job.clone())
    }

    /// Mark a job as complete so that jobs depending on it become ready.
    pub fn mark_complete(&mut self, job_id: &str) {
        self.completed.insert(job_id.to_string());
    }

    /// Number of jobs added to the chain (including dispatched ones).
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` when the chain is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Number of jobs that have been dispatched.
    #[must_use]
    pub fn dispatched_count(&self) -> usize {
        self.entries.iter().filter(|e| e.dispatched).count()
    }

    /// Number of jobs that are still waiting (not yet dispatched).
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.entries.iter().filter(|e| !e.dispatched).count()
    }
}

impl Default for BatchJobChain {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::job::{BatchJob, BatchOperation};

    fn job(name: &str) -> BatchJob {
        BatchJob::new(
            name.to_string(),
            BatchOperation::Transcode { preset: "default".to_string() },
        )
    }

    // ── new / add ────────────────────────────────────────────────────────────

    #[test]
    fn test_new_chain_is_empty() {
        let chain = BatchJobChain::new();
        assert!(chain.is_empty());
        assert_eq!(chain.len(), 0);
    }

    #[test]
    fn test_add_increments_len() {
        let mut chain = BatchJobChain::new();
        chain.add(job("a"), None);
        chain.add(job("b"), None);
        assert_eq!(chain.len(), 2);
    }

    // ── next_ready ────────────────────────────────────────────────────────────

    #[test]
    fn test_next_ready_no_dependency() {
        let mut chain = BatchJobChain::new();
        chain.add(job("solo"), None);
        let next = chain.next_ready();
        assert!(next.is_some());
        assert_eq!(next.expect("job ready").name, "solo");
    }

    #[test]
    fn test_next_ready_returns_none_when_empty() {
        let mut chain = BatchJobChain::new();
        assert!(chain.next_ready().is_none());
    }

    #[test]
    fn test_next_ready_blocked_by_dependency() {
        let a = job("a");
        let a_id = a.id.as_str().to_string();

        let mut chain = BatchJobChain::new();
        chain.add(a, None);
        chain.next_ready(); // dispatch a

        let b = job("b");
        chain.add(b, Some(a_id));

        // b is blocked (a not yet marked complete)
        assert!(chain.next_ready().is_none());
    }

    #[test]
    fn test_next_ready_unblocked_after_mark_complete() {
        let a = job("a");
        let a_id = a.id.as_str().to_string();
        let b = job("b");

        let mut chain = BatchJobChain::new();
        chain.add(a, None);
        chain.add(b, Some(a_id));

        let first = chain.next_ready().expect("a should be ready");
        chain.mark_complete(first.id.as_str());

        let second = chain.next_ready().expect("b should now be ready");
        assert_eq!(second.name, "b");
    }

    #[test]
    fn test_next_ready_does_not_return_same_job_twice() {
        let mut chain = BatchJobChain::new();
        chain.add(job("once"), None);
        chain.next_ready();
        assert!(chain.next_ready().is_none());
    }

    // ── pending / dispatched counts ───────────────────────────────────────────

    #[test]
    fn test_dispatched_count_increases() {
        let mut chain = BatchJobChain::new();
        chain.add(job("x"), None);
        assert_eq!(chain.dispatched_count(), 0);
        chain.next_ready();
        assert_eq!(chain.dispatched_count(), 1);
    }

    #[test]
    fn test_pending_count_decreases() {
        let mut chain = BatchJobChain::new();
        chain.add(job("x"), None);
        chain.add(job("y"), None);
        assert_eq!(chain.pending_count(), 2);
        chain.next_ready();
        assert_eq!(chain.pending_count(), 1);
    }

    // ── three-job chain ───────────────────────────────────────────────────────

    #[test]
    fn test_three_job_linear_chain() {
        let a = job("a");
        let a_id = a.id.as_str().to_string();
        let b = job("b");
        let b_dep = a_id.clone();

        let mut chain = BatchJobChain::new();
        chain.add(a, None);
        chain.add(b, Some(b_dep));

        let j_a = chain.next_ready().expect("a ready");
        // c depends on a too
        let c = job("c");
        chain.add(c, Some(a_id));
        chain.mark_complete(j_a.id.as_str());

        // Both b and c are now ready; insertion order → b first
        let j_b = chain.next_ready().expect("b ready");
        assert_eq!(j_b.name, "b");
        let j_c = chain.next_ready().expect("c ready");
        assert_eq!(j_c.name, "c");
        assert!(chain.next_ready().is_none());
    }
}
