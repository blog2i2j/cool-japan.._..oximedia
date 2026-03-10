//! Distributed job tracking.
//!
//! Provides a lifecycle-aware store for distributed encoding jobs.

/// State machine for a distributed job.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum JobState {
    Queued,
    Assigned { worker_id: u64 },
    Running { progress_pct: f32 },
    Completed { output: String },
    Failed { error: String },
    Cancelled,
}

impl JobState {
    fn is_completed(&self) -> bool {
        matches!(self, JobState::Completed { .. })
    }

    fn is_failed(&self) -> bool {
        matches!(self, JobState::Failed { .. })
    }

    fn is_queued(&self) -> bool {
        matches!(self, JobState::Queued)
    }

    fn is_running(&self) -> bool {
        matches!(self, JobState::Running { .. })
    }
}

/// A single distributed encoding job.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct DistributedJob {
    pub id: u64,
    pub name: String,
    pub state: JobState,
    pub created_at: u64,
    pub updated_at: u64,
    pub priority: i32,
}

impl DistributedJob {
    /// Create a new job in the `Queued` state.
    #[must_use]
    pub fn new(id: u64, name: &str, priority: i32, now: u64) -> Self {
        Self {
            id,
            name: name.to_string(),
            state: JobState::Queued,
            created_at: now,
            updated_at: now,
            priority,
        }
    }

    /// Transition to `Assigned`.
    pub fn assign(&mut self, worker_id: u64, now: u64) {
        self.state = JobState::Assigned { worker_id };
        self.updated_at = now;
    }

    /// Transition to `Running` with the given progress percentage.
    pub fn update_progress(&mut self, pct: f32, now: u64) {
        self.state = JobState::Running {
            progress_pct: pct.clamp(0.0, 100.0),
        };
        self.updated_at = now;
    }

    /// Transition to `Completed`.
    pub fn complete(&mut self, output: &str, now: u64) {
        self.state = JobState::Completed {
            output: output.to_string(),
        };
        self.updated_at = now;
    }

    /// Transition to `Failed`.
    pub fn fail(&mut self, error: &str, now: u64) {
        self.state = JobState::Failed {
            error: error.to_string(),
        };
        self.updated_at = now;
    }

    /// Transition to `Cancelled`.
    pub fn cancel(&mut self, now: u64) {
        self.state = JobState::Cancelled;
        self.updated_at = now;
    }
}

/// Stores and queries distributed jobs.
#[allow(dead_code)]
#[derive(Debug, Default)]
pub struct JobTracker {
    jobs: Vec<DistributedJob>,
}

impl JobTracker {
    /// Create an empty tracker.
    #[must_use]
    pub fn new() -> Self {
        Self { jobs: Vec::new() }
    }

    /// Add a job to the tracker.
    pub fn submit(&mut self, job: DistributedJob) {
        self.jobs.push(job);
    }

    /// Look up a job by ID (immutable).
    #[must_use]
    pub fn get(&self, id: u64) -> Option<&DistributedJob> {
        self.jobs.iter().find(|j| j.id == id)
    }

    /// Look up a job by ID (mutable).
    pub fn get_mut(&mut self, id: u64) -> Option<&mut DistributedJob> {
        self.jobs.iter_mut().find(|j| j.id == id)
    }

    /// Return all jobs currently in the `Queued` state.
    #[must_use]
    pub fn queued_jobs(&self) -> Vec<&DistributedJob> {
        self.jobs.iter().filter(|j| j.state.is_queued()).collect()
    }

    /// Return all jobs currently in the `Running` state.
    #[must_use]
    pub fn running_jobs(&self) -> Vec<&DistributedJob> {
        self.jobs.iter().filter(|j| j.state.is_running()).collect()
    }

    /// Return all jobs currently in the `Failed` state.
    #[must_use]
    pub fn failed_jobs(&self) -> Vec<&DistributedJob> {
        self.jobs.iter().filter(|j| j.state.is_failed()).collect()
    }

    /// Fraction of jobs that completed successfully.
    ///
    /// Returns `0.0` when no jobs have been submitted.
    #[must_use]
    pub fn completion_rate(&self) -> f64 {
        if self.jobs.is_empty() {
            return 0.0;
        }
        let completed = self.jobs.iter().filter(|j| j.state.is_completed()).count();
        completed as f64 / self.jobs.len() as f64
    }

    /// Return the total number of tracked jobs.
    #[must_use]
    pub fn total_jobs(&self) -> usize {
        self.jobs.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn job(id: u64) -> DistributedJob {
        DistributedJob::new(id, &format!("job-{}", id), 0, 1000)
    }

    #[test]
    fn test_new_job_queued() {
        let j = job(1);
        assert!(j.state.is_queued());
        assert_eq!(j.id, 1);
        assert_eq!(j.name, "job-1");
    }

    #[test]
    fn test_assign_job() {
        let mut j = job(1);
        j.assign(42, 2000);
        assert!(matches!(j.state, JobState::Assigned { worker_id: 42 }));
        assert_eq!(j.updated_at, 2000);
    }

    #[test]
    fn test_update_progress() {
        let mut j = job(1);
        j.assign(1, 1001);
        j.update_progress(55.5, 1002);
        assert!(matches!(
            j.state,
            JobState::Running { progress_pct } if (progress_pct - 55.5).abs() < 1e-4
        ));
    }

    #[test]
    fn test_progress_clamps() {
        let mut j = job(1);
        j.update_progress(200.0, 1001);
        assert!(matches!(
            j.state,
            JobState::Running { progress_pct } if (progress_pct - 100.0).abs() < 1e-4
        ));
    }

    #[test]
    fn test_complete_job() {
        let mut j = job(1);
        j.complete("s3://bucket/output.mp4", 3000);
        assert!(j.state.is_completed());
        assert!(matches!(j.state, JobState::Completed { ref output } if output.contains("mp4")));
    }

    #[test]
    fn test_fail_job() {
        let mut j = job(1);
        j.fail("out of memory", 4000);
        assert!(j.state.is_failed());
    }

    #[test]
    fn test_cancel_job() {
        let mut j = job(1);
        j.cancel(5000);
        assert!(matches!(j.state, JobState::Cancelled));
    }

    #[test]
    fn test_tracker_submit_and_get() {
        let mut t = JobTracker::new();
        t.submit(job(10));
        let j = t.get(10).expect("get should return a value");
        assert_eq!(j.id, 10);
    }

    #[test]
    fn test_tracker_get_mut() {
        let mut t = JobTracker::new();
        t.submit(job(1));
        t.get_mut(1)
            .expect("get_mut should return a value")
            .assign(99, 2000);
        assert!(matches!(
            t.get(1).expect("get should return a value").state,
            JobState::Assigned { .. }
        ));
    }

    #[test]
    fn test_tracker_queued_jobs() {
        let mut t = JobTracker::new();
        t.submit(job(1));
        t.submit(job(2));
        t.get_mut(1)
            .expect("get_mut should return a value")
            .assign(7, 1001);
        assert_eq!(t.queued_jobs().len(), 1);
        assert_eq!(t.queued_jobs()[0].id, 2);
    }

    #[test]
    fn test_tracker_running_jobs() {
        let mut t = JobTracker::new();
        t.submit(job(1));
        t.submit(job(2));
        t.get_mut(1)
            .expect("get_mut should return a value")
            .update_progress(50.0, 1001);
        assert_eq!(t.running_jobs().len(), 1);
    }

    #[test]
    fn test_tracker_failed_jobs() {
        let mut t = JobTracker::new();
        t.submit(job(1));
        t.submit(job(2));
        t.get_mut(2)
            .expect("get_mut should return a value")
            .fail("error", 1001);
        assert_eq!(t.failed_jobs().len(), 1);
        assert_eq!(t.failed_jobs()[0].id, 2);
    }

    #[test]
    fn test_completion_rate_empty() {
        let t = JobTracker::new();
        assert_eq!(t.completion_rate(), 0.0);
    }

    #[test]
    fn test_completion_rate_all_complete() {
        let mut t = JobTracker::new();
        for i in 1..=3 {
            t.submit(job(i));
            t.get_mut(i)
                .expect("get_mut should return a value")
                .complete("out", 2000);
        }
        assert!((t.completion_rate() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_completion_rate_partial() {
        let mut t = JobTracker::new();
        t.submit(job(1));
        t.submit(job(2));
        t.get_mut(1)
            .expect("get_mut should return a value")
            .complete("out", 2000);
        assert!((t.completion_rate() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_total_jobs() {
        let mut t = JobTracker::new();
        assert_eq!(t.total_jobs(), 0);
        t.submit(job(1));
        t.submit(job(2));
        assert_eq!(t.total_jobs(), 2);
    }
}
