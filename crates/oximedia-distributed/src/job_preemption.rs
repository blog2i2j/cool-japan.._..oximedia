#![allow(dead_code)]
//! Job preemption for higher-priority jobs.
//!
//! Provides a [`PreemptionScheduler`] that can preempt (suspend) running jobs
//! when a higher-priority job arrives and the system has no free capacity.
//! Preempted jobs are placed back into a waiting queue and resumed when
//! resources become available.

use std::collections::{BinaryHeap, HashMap};
use std::fmt;
use std::time::Instant;

use uuid::Uuid;

use crate::{DistributedError, JobPriority, Result};

// ---------------------------------------------------------------------------
// PreemptionPolicy
// ---------------------------------------------------------------------------

/// Controls when preemption is allowed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreemptionPolicy {
    /// Never preempt running jobs (FIFO behaviour).
    Never,
    /// Preempt only when the incoming job has strictly higher priority.
    HigherPriority,
    /// Preempt when incoming priority is higher *and* the running job has been
    /// active for at least the grace period.
    HigherPriorityWithGrace {
        /// Minimum time (ms) a job must have been running before it can be
        /// preempted.
        grace_period_ms: u64,
    },
    /// Always preempt the lowest-priority running job when capacity is full.
    Always,
}

impl Default for PreemptionPolicy {
    fn default() -> Self {
        Self::HigherPriority
    }
}

// ---------------------------------------------------------------------------
// PreemptibleJob
// ---------------------------------------------------------------------------

/// The state a job can be in within the preemption scheduler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreemptionState {
    /// Waiting in the queue for a slot.
    Queued,
    /// Currently running on a worker.
    Running,
    /// Was running but has been preempted; waiting for a new slot.
    Suspended,
    /// Finished (either completed or cancelled).
    Finished,
}

impl fmt::Display for PreemptionState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Queued => write!(f, "Queued"),
            Self::Running => write!(f, "Running"),
            Self::Suspended => write!(f, "Suspended"),
            Self::Finished => write!(f, "Finished"),
        }
    }
}

/// A job tracked by the preemption scheduler.
#[derive(Debug, Clone)]
pub struct PreemptibleJob {
    /// Unique job ID.
    pub id: Uuid,
    /// Job priority.
    pub priority: JobPriority,
    /// Current state.
    pub state: PreemptionState,
    /// When the job was submitted.
    pub submitted_at: Instant,
    /// When the job most recently started running (if ever).
    pub started_at: Option<Instant>,
    /// How many times this job has been preempted.
    pub preemption_count: u32,
    /// Optional worker ID the job is assigned to.
    pub worker_id: Option<String>,
}

// We need an Ord impl so the BinaryHeap pops the *highest* priority first.
// Tie-break: earlier submission first.
impl PartialEq for PreemptibleJob {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for PreemptibleJob {}

impl PartialOrd for PreemptibleJob {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PreemptibleJob {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Higher priority first
        let pri_cmp = self.priority.cmp(&other.priority);
        if pri_cmp != std::cmp::Ordering::Equal {
            return pri_cmp;
        }
        // Earlier submission first (reverse: earlier instant is "greater" in heap)
        other.submitted_at.cmp(&self.submitted_at)
    }
}

// ---------------------------------------------------------------------------
// PreemptionEvent
// ---------------------------------------------------------------------------

/// Records a preemption event for auditing.
#[derive(Debug, Clone)]
pub struct PreemptionEvent {
    /// ID of the preempted (victim) job.
    pub victim_job_id: Uuid,
    /// Priority of the victim.
    pub victim_priority: JobPriority,
    /// ID of the incoming job that caused the preemption.
    pub preemptor_job_id: Uuid,
    /// Priority of the preemptor.
    pub preemptor_priority: JobPriority,
    /// When the preemption occurred.
    pub timestamp: Instant,
}

// ---------------------------------------------------------------------------
// PreemptionScheduler
// ---------------------------------------------------------------------------

/// Manages job scheduling with preemption support.
///
/// The scheduler maintains a set of *running* slots (limited by `capacity`)
/// and a priority-ordered waiting queue. When a new job arrives and no slot is
/// free, the scheduler may preempt the lowest-priority running job if the
/// [`PreemptionPolicy`] allows it.
pub struct PreemptionScheduler {
    /// Maximum number of concurrently running jobs.
    capacity: usize,
    /// Preemption policy.
    policy: PreemptionPolicy,
    /// All tracked jobs keyed by ID (authoritative state).
    jobs: HashMap<Uuid, PreemptibleJob>,
    /// Waiting queue (Queued + Suspended jobs), ordered by priority.
    wait_queue: BinaryHeap<WaitEntry>,
    /// IDs of currently running jobs.
    running: Vec<Uuid>,
    /// History of preemption events.
    events: Vec<PreemptionEvent>,
}

/// Lightweight entry in the wait queue (to avoid cloning the full job).
#[derive(Debug, Clone)]
struct WaitEntry {
    id: Uuid,
    priority: JobPriority,
    submitted_at: Instant,
}

impl PartialEq for WaitEntry {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
impl Eq for WaitEntry {}

impl PartialOrd for WaitEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for WaitEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let pri = self.priority.cmp(&other.priority);
        if pri != std::cmp::Ordering::Equal {
            return pri;
        }
        other.submitted_at.cmp(&self.submitted_at)
    }
}

impl PreemptionScheduler {
    /// Create a new scheduler with the given capacity and policy.
    pub fn new(capacity: usize, policy: PreemptionPolicy) -> Self {
        Self {
            capacity: capacity.max(1),
            policy,
            jobs: HashMap::new(),
            wait_queue: BinaryHeap::new(),
            running: Vec::new(),
            events: Vec::new(),
        }
    }

    /// Number of currently running jobs.
    pub fn running_count(&self) -> usize {
        self.running.len()
    }

    /// Number of jobs waiting in the queue.
    pub fn waiting_count(&self) -> usize {
        self.wait_queue.len()
    }

    /// Total tracked jobs (all states except Finished).
    pub fn active_count(&self) -> usize {
        self.jobs
            .values()
            .filter(|j| j.state != PreemptionState::Finished)
            .count()
    }

    /// Returns the list of preemption events.
    pub fn preemption_events(&self) -> &[PreemptionEvent] {
        &self.events
    }

    /// Submit a new job. Returns `Ok(true)` if the job started running
    /// immediately, `Ok(false)` if it was queued, and possibly preempts a
    /// lower-priority running job.
    pub fn submit(&mut self, id: Uuid, priority: JobPriority) -> Result<bool> {
        if self.jobs.contains_key(&id) {
            return Err(DistributedError::Job(format!(
                "Job {id} already tracked by preemption scheduler"
            )));
        }

        let now = Instant::now();
        let job = PreemptibleJob {
            id,
            priority,
            state: PreemptionState::Queued,
            submitted_at: now,
            started_at: None,
            preemption_count: 0,
            worker_id: None,
        };

        self.jobs.insert(id, job);

        // Try to run immediately if capacity available.
        if self.running.len() < self.capacity {
            self.start_job(id, now);
            return Ok(true);
        }

        // Capacity full — attempt preemption.
        if let Some(victim_id) = self.find_preemption_victim(priority, now) {
            // Record event.
            let victim_priority = self
                .jobs
                .get(&victim_id)
                .map(|j| j.priority)
                .unwrap_or(JobPriority::Low);
            self.events.push(PreemptionEvent {
                victim_job_id: victim_id,
                victim_priority,
                preemptor_job_id: id,
                preemptor_priority: priority,
                timestamp: now,
            });

            // Suspend the victim.
            self.suspend_job(victim_id, now);

            // Start the new job.
            self.start_job(id, now);
            return Ok(true);
        }

        // No preemption possible — just queue.
        self.wait_queue.push(WaitEntry {
            id,
            priority,
            submitted_at: now,
        });

        Ok(false)
    }

    /// Mark a running job as finished and schedule the next waiting job.
    pub fn finish(&mut self, id: Uuid) -> Result<Option<Uuid>> {
        let job = self
            .jobs
            .get_mut(&id)
            .ok_or_else(|| DistributedError::Job(format!("Job {id} not found")))?;

        if job.state != PreemptionState::Running {
            return Err(DistributedError::Job(format!(
                "Job {id} is not running (state: {})",
                job.state
            )));
        }

        job.state = PreemptionState::Finished;
        self.running.retain(|rid| *rid != id);

        // Try to schedule next from wait queue.
        let scheduled = self.schedule_next();
        Ok(scheduled)
    }

    /// Cancel a job in any non-finished state.
    pub fn cancel(&mut self, id: Uuid) -> Result<Option<Uuid>> {
        let job = self
            .jobs
            .get_mut(&id)
            .ok_or_else(|| DistributedError::Job(format!("Job {id} not found")))?;

        if job.state == PreemptionState::Finished {
            return Err(DistributedError::Job(format!(
                "Job {id} is already finished"
            )));
        }

        let was_running = job.state == PreemptionState::Running;
        job.state = PreemptionState::Finished;

        if was_running {
            self.running.retain(|rid| *rid != id);
            let scheduled = self.schedule_next();
            return Ok(scheduled);
        }

        // Remove from wait queue if present.
        self.rebuild_wait_queue();
        Ok(None)
    }

    /// Get the current state of a job.
    pub fn job_state(&self, id: Uuid) -> Option<PreemptionState> {
        self.jobs.get(&id).map(|j| j.state)
    }

    /// Get the preemption count for a job.
    pub fn preemption_count(&self, id: Uuid) -> Option<u32> {
        self.jobs.get(&id).map(|j| j.preemption_count)
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    fn start_job(&mut self, id: Uuid, now: Instant) {
        if let Some(job) = self.jobs.get_mut(&id) {
            job.state = PreemptionState::Running;
            job.started_at = Some(now);
            self.running.push(id);
        }
    }

    fn suspend_job(&mut self, id: Uuid, now: Instant) {
        if let Some(job) = self.jobs.get_mut(&id) {
            job.state = PreemptionState::Suspended;
            job.preemption_count += 1;
            job.started_at = None;
            self.running.retain(|rid| *rid != id);
            self.wait_queue.push(WaitEntry {
                id,
                priority: job.priority,
                submitted_at: job.submitted_at,
            });
            let _ = now; // used for timestamp context
        }
    }

    fn find_preemption_victim(&self, incoming_priority: JobPriority, now: Instant) -> Option<Uuid> {
        match self.policy {
            PreemptionPolicy::Never => None,
            PreemptionPolicy::Always => self.lowest_priority_running(),
            PreemptionPolicy::HigherPriority => {
                let victim_id = self.lowest_priority_running()?;
                let victim = self.jobs.get(&victim_id)?;
                if incoming_priority > victim.priority {
                    Some(victim_id)
                } else {
                    None
                }
            }
            PreemptionPolicy::HigherPriorityWithGrace { grace_period_ms } => {
                let victim_id = self.lowest_priority_running()?;
                let victim = self.jobs.get(&victim_id)?;
                if incoming_priority <= victim.priority {
                    return None;
                }
                let grace = std::time::Duration::from_millis(grace_period_ms);
                if let Some(started) = victim.started_at {
                    if now.duration_since(started) >= grace {
                        Some(victim_id)
                    } else {
                        None
                    }
                } else {
                    Some(victim_id)
                }
            }
        }
    }

    fn lowest_priority_running(&self) -> Option<Uuid> {
        self.running
            .iter()
            .filter_map(|id| self.jobs.get(id).map(|j| (id, j)))
            .min_by(|(_, a), (_, b)| a.priority.cmp(&b.priority))
            .map(|(id, _)| *id)
    }

    fn schedule_next(&mut self) -> Option<Uuid> {
        while self.running.len() < self.capacity {
            // Pop highest-priority waiting job.
            let entry = self.wait_queue.pop()?;
            let job = match self.jobs.get(&entry.id) {
                Some(j)
                    if j.state == PreemptionState::Queued
                        || j.state == PreemptionState::Suspended =>
                {
                    entry.id
                }
                _ => continue, // stale entry
            };
            self.start_job(job, Instant::now());
            return Some(job);
        }
        None
    }

    /// Rebuild the wait queue removing entries whose jobs are no longer waiting.
    fn rebuild_wait_queue(&mut self) {
        let jobs = &self.jobs;
        let entries: Vec<WaitEntry> = self
            .wait_queue
            .drain()
            .filter(|e| {
                jobs.get(&e.id)
                    .map(|j| {
                        j.state == PreemptionState::Queued || j.state == PreemptionState::Suspended
                    })
                    .unwrap_or(false)
            })
            .collect();
        self.wait_queue = BinaryHeap::from(entries);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_submit_runs_immediately_when_capacity_available() {
        let mut sched = PreemptionScheduler::new(2, PreemptionPolicy::HigherPriority);
        let id = Uuid::new_v4();
        let started = sched.submit(id, JobPriority::Normal).expect("submit ok");
        assert!(started);
        assert_eq!(sched.running_count(), 1);
        assert_eq!(sched.job_state(id), Some(PreemptionState::Running));
    }

    #[test]
    fn test_submit_queues_when_capacity_full_no_preemption() {
        let mut sched = PreemptionScheduler::new(1, PreemptionPolicy::HigherPriority);
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();

        sched.submit(id1, JobPriority::High).expect("submit ok");
        // id2 has same priority — no preemption
        let started = sched.submit(id2, JobPriority::High).expect("submit ok");
        assert!(!started);
        assert_eq!(sched.running_count(), 1);
        assert_eq!(sched.waiting_count(), 1);
        assert_eq!(sched.job_state(id2), Some(PreemptionState::Queued));
    }

    #[test]
    fn test_preemption_higher_priority() {
        let mut sched = PreemptionScheduler::new(1, PreemptionPolicy::HigherPriority);
        let low_id = Uuid::new_v4();
        let high_id = Uuid::new_v4();

        sched.submit(low_id, JobPriority::Low).expect("submit ok");
        let started = sched
            .submit(high_id, JobPriority::Critical)
            .expect("submit ok");

        assert!(started);
        assert_eq!(sched.job_state(low_id), Some(PreemptionState::Suspended));
        assert_eq!(sched.job_state(high_id), Some(PreemptionState::Running));
        assert_eq!(sched.preemption_count(low_id), Some(1));
        assert_eq!(sched.preemption_events().len(), 1);
    }

    #[test]
    fn test_no_preemption_with_never_policy() {
        let mut sched = PreemptionScheduler::new(1, PreemptionPolicy::Never);
        let low_id = Uuid::new_v4();
        let high_id = Uuid::new_v4();

        sched.submit(low_id, JobPriority::Low).expect("submit ok");
        let started = sched
            .submit(high_id, JobPriority::Critical)
            .expect("submit ok");

        assert!(!started);
        assert_eq!(sched.job_state(low_id), Some(PreemptionState::Running));
        assert_eq!(sched.job_state(high_id), Some(PreemptionState::Queued));
    }

    #[test]
    fn test_finish_schedules_next_from_queue() {
        let mut sched = PreemptionScheduler::new(1, PreemptionPolicy::HigherPriority);
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();

        sched.submit(id1, JobPriority::Normal).expect("submit ok");
        sched.submit(id2, JobPriority::Normal).expect("submit ok");

        let next = sched.finish(id1).expect("finish ok");
        assert_eq!(next, Some(id2));
        assert_eq!(sched.job_state(id1), Some(PreemptionState::Finished));
        assert_eq!(sched.job_state(id2), Some(PreemptionState::Running));
    }

    #[test]
    fn test_cancel_running_schedules_next() {
        let mut sched = PreemptionScheduler::new(1, PreemptionPolicy::HigherPriority);
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();

        sched.submit(id1, JobPriority::Normal).expect("submit ok");
        sched.submit(id2, JobPriority::Normal).expect("submit ok");

        let next = sched.cancel(id1).expect("cancel ok");
        assert_eq!(next, Some(id2));
        assert_eq!(sched.job_state(id1), Some(PreemptionState::Finished));
    }

    #[test]
    fn test_cancel_queued_job() {
        let mut sched = PreemptionScheduler::new(1, PreemptionPolicy::HigherPriority);
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();

        sched.submit(id1, JobPriority::Normal).expect("submit ok");
        sched.submit(id2, JobPriority::Normal).expect("submit ok");

        let next = sched.cancel(id2).expect("cancel ok");
        assert!(next.is_none());
        assert_eq!(sched.waiting_count(), 0);
    }

    #[test]
    fn test_duplicate_submit_fails() {
        let mut sched = PreemptionScheduler::new(2, PreemptionPolicy::HigherPriority);
        let id = Uuid::new_v4();
        sched.submit(id, JobPriority::Normal).expect("submit ok");
        let result = sched.submit(id, JobPriority::High);
        assert!(result.is_err());
    }

    #[test]
    fn test_finish_non_running_fails() {
        let mut sched = PreemptionScheduler::new(1, PreemptionPolicy::HigherPriority);
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();

        sched.submit(id1, JobPriority::Normal).expect("submit ok");
        sched.submit(id2, JobPriority::Normal).expect("submit ok");

        // id2 is queued, not running
        let result = sched.finish(id2);
        assert!(result.is_err());
    }

    #[test]
    fn test_cancel_finished_fails() {
        let mut sched = PreemptionScheduler::new(1, PreemptionPolicy::HigherPriority);
        let id = Uuid::new_v4();
        sched.submit(id, JobPriority::Normal).expect("submit ok");
        sched.finish(id).expect("finish ok");
        let result = sched.cancel(id);
        assert!(result.is_err());
    }

    #[test]
    fn test_preempted_job_resumes_after_preemptor_finishes() {
        let mut sched = PreemptionScheduler::new(1, PreemptionPolicy::HigherPriority);
        let low_id = Uuid::new_v4();
        let high_id = Uuid::new_v4();

        sched.submit(low_id, JobPriority::Low).expect("submit ok");
        sched
            .submit(high_id, JobPriority::Critical)
            .expect("submit ok");

        assert_eq!(sched.job_state(low_id), Some(PreemptionState::Suspended));

        let next = sched.finish(high_id).expect("finish ok");
        assert_eq!(next, Some(low_id));
        assert_eq!(sched.job_state(low_id), Some(PreemptionState::Running));
    }

    #[test]
    fn test_always_preemption_policy() {
        let mut sched = PreemptionScheduler::new(1, PreemptionPolicy::Always);
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();

        sched.submit(id1, JobPriority::Critical).expect("submit ok");
        // Same priority but Always policy still preempts
        let started = sched.submit(id2, JobPriority::Critical).expect("submit ok");
        assert!(started);
        assert_eq!(sched.job_state(id1), Some(PreemptionState::Suspended));
        assert_eq!(sched.job_state(id2), Some(PreemptionState::Running));
    }

    #[test]
    fn test_active_count() {
        let mut sched = PreemptionScheduler::new(2, PreemptionPolicy::HigherPriority);
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();

        assert_eq!(sched.active_count(), 0);
        sched.submit(id1, JobPriority::Normal).expect("submit ok");
        sched.submit(id2, JobPriority::Normal).expect("submit ok");
        assert_eq!(sched.active_count(), 2);

        sched.finish(id1).expect("finish ok");
        assert_eq!(sched.active_count(), 1);
    }
}
