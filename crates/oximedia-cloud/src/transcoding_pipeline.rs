//! Cloud transcoding pipeline types.
//!
//! Provides a lightweight, network-free model for submitting, tracking, and
//! querying cloud media transcoding jobs.

#![allow(dead_code)]

/// Status of a cloud transcoding job.
#[derive(Debug, Clone, PartialEq)]
pub enum JobStatus {
    /// Waiting to be picked up by a worker.
    Pending,
    /// Currently being transcoded; value is the completion percentage (0–100).
    Running(u8),
    /// Successfully completed; value is the output URL.
    Completed(String),
    /// Failed; value is the error message.
    Failed(String),
}

impl JobStatus {
    /// Returns the progress percentage.
    ///
    /// - `Running(n)` → `n`
    /// - `Completed` → 100
    /// - `Pending` / `Failed` → 0
    #[must_use]
    pub fn progress_pct(&self) -> u8 {
        match self {
            JobStatus::Running(pct) => *pct,
            JobStatus::Completed(_) => 100,
            _ => 0,
        }
    }

    /// Returns the output URL when the job has completed, otherwise `None`.
    #[must_use]
    pub fn output_url(&self) -> Option<&str> {
        match self {
            JobStatus::Completed(url) => Some(url.as_str()),
            _ => None,
        }
    }

    /// Returns `true` when the job has reached a terminal state.
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        matches!(self, JobStatus::Completed(_) | JobStatus::Failed(_))
    }
}

/// A single cloud transcoding job.
#[derive(Debug, Clone)]
pub struct CloudTranscodeJob {
    /// Unique job identifier.
    pub id: String,
    /// URL or cloud path of the source media.
    pub input_url: String,
    /// Cloud storage prefix where output files are written.
    pub output_prefix: String,
    /// Name of the transcoding profile to apply.
    pub profile_name: String,
    /// Current status of the job.
    pub status: JobStatus,
}

impl CloudTranscodeJob {
    /// Creates a new `CloudTranscodeJob` in the `Pending` state.
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        input_url: impl Into<String>,
        output_prefix: impl Into<String>,
        profile_name: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            input_url: input_url.into(),
            output_prefix: output_prefix.into(),
            profile_name: profile_name.into(),
            status: JobStatus::Pending,
        }
    }

    /// Returns `true` when the job has completed or failed.
    #[must_use]
    pub fn is_done(&self) -> bool {
        self.status.is_terminal()
    }
}

/// A named cloud transcoding profile.
#[derive(Debug, Clone)]
pub struct CloudProfile {
    /// Profile name (used to look up profiles by `CloudTranscodingQueue`).
    pub name: String,
    /// Video codec (e.g. `"av1"`, `"vp9"`).
    pub codec: String,
    /// Target bitrate in kilobits per second.
    pub bitrate_kbps: u32,
    /// Output resolution as `(width, height)`.
    pub resolution: (u32, u32),
    /// Target frame rate.
    pub fps: f32,
}

impl CloudProfile {
    /// Creates a new `CloudProfile`.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        codec: impl Into<String>,
        bitrate_kbps: u32,
        resolution: (u32, u32),
        fps: f32,
    ) -> Self {
        Self {
            name: name.into(),
            codec: codec.into(),
            bitrate_kbps,
            resolution,
            fps,
        }
    }

    /// Estimates the output file size in megabytes for a source of `duration_s` seconds.
    ///
    /// Formula: `(bitrate_kbps * duration_s) / 8_000`
    #[must_use]
    pub fn estimated_output_size_mb(&self, duration_s: f64) -> f64 {
        (self.bitrate_kbps as f64 * duration_s) / 8_000.0
    }
}

/// An in-memory queue of [`CloudTranscodeJob`]s with an associated profile registry.
#[derive(Debug, Default)]
pub struct CloudTranscodingQueue {
    jobs: Vec<CloudTranscodeJob>,
    profiles: Vec<CloudProfile>,
    next_id: u64,
}

impl CloudTranscodingQueue {
    /// Creates an empty queue.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a transcoding profile.
    pub fn add_profile(&mut self, profile: CloudProfile) {
        self.profiles.push(profile);
    }

    /// Submits a new transcoding job and returns its generated ID.
    ///
    /// The job is created in the `Pending` state.
    pub fn submit(&mut self, input: &str, output_prefix: &str, profile_name: &str) -> String {
        self.next_id += 1;
        let id = format!("job-{}", self.next_id);
        let job = CloudTranscodeJob::new(id.clone(), input, output_prefix, profile_name);
        self.jobs.push(job);
        id
    }

    /// Finds a job by its ID.
    #[must_use]
    pub fn find_job(&self, id: &str) -> Option<&CloudTranscodeJob> {
        self.jobs.iter().find(|j| j.id == id)
    }

    /// Finds a mutable reference to a job by its ID.
    #[must_use]
    pub fn find_job_mut(&mut self, id: &str) -> Option<&mut CloudTranscodeJob> {
        self.jobs.iter_mut().find(|j| j.id == id)
    }

    /// Returns all jobs in the `Pending` state.
    #[must_use]
    pub fn pending_jobs(&self) -> Vec<&CloudTranscodeJob> {
        self.jobs
            .iter()
            .filter(|j| j.status == JobStatus::Pending)
            .collect()
    }

    /// Returns all jobs in a `Completed` state.
    #[must_use]
    pub fn completed_jobs(&self) -> Vec<&CloudTranscodeJob> {
        self.jobs
            .iter()
            .filter(|j| matches!(j.status, JobStatus::Completed(_)))
            .collect()
    }

    /// Total number of jobs in the queue (all states).
    #[must_use]
    pub fn total_jobs(&self) -> usize {
        self.jobs.len()
    }

    /// Looks up a profile by name.
    #[must_use]
    pub fn find_profile(&self, name: &str) -> Option<&CloudProfile> {
        self.profiles.iter().find(|p| p.name == name)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn av1_profile() -> CloudProfile {
        CloudProfile::new("av1-1080p", "av1", 4_000, (1920, 1080), 30.0)
    }

    // 1. JobStatus::progress_pct
    #[test]
    fn test_job_status_progress() {
        assert_eq!(JobStatus::Pending.progress_pct(), 0);
        assert_eq!(JobStatus::Running(55).progress_pct(), 55);
        assert_eq!(
            JobStatus::Completed("s3://b/out.mp4".to_string()).progress_pct(),
            100
        );
        assert_eq!(JobStatus::Failed("err".to_string()).progress_pct(), 0);
    }

    // 2. JobStatus::output_url
    #[test]
    fn test_job_status_output_url() {
        let url = "s3://bucket/output.mp4";
        assert_eq!(
            JobStatus::Completed(url.to_string()).output_url(),
            Some(url)
        );
        assert_eq!(JobStatus::Pending.output_url(), None);
        assert_eq!(JobStatus::Running(50).output_url(), None);
    }

    // 3. JobStatus::is_terminal
    #[test]
    fn test_job_status_is_terminal() {
        assert!(!JobStatus::Pending.is_terminal());
        assert!(!JobStatus::Running(10).is_terminal());
        assert!(JobStatus::Completed("url".to_string()).is_terminal());
        assert!(JobStatus::Failed("err".to_string()).is_terminal());
    }

    // 4. CloudTranscodeJob::new
    #[test]
    fn test_job_new() {
        let job = CloudTranscodeJob::new("j1", "s3://in/a.mp4", "s3://out/", "av1-1080p");
        assert_eq!(job.id, "j1");
        assert_eq!(job.status, JobStatus::Pending);
        assert!(!job.is_done());
    }

    // 5. CloudTranscodeJob::is_done – completed
    #[test]
    fn test_job_is_done_completed() {
        let mut job = CloudTranscodeJob::new("j2", "s3://in/b.mp4", "s3://out/", "av1-1080p");
        job.status = JobStatus::Completed("s3://out/b.mp4".to_string());
        assert!(job.is_done());
    }

    // 6. CloudTranscodeJob::is_done – failed
    #[test]
    fn test_job_is_done_failed() {
        let mut job = CloudTranscodeJob::new("j3", "s3://in/c.mp4", "s3://out/", "av1-1080p");
        job.status = JobStatus::Failed("codec error".to_string());
        assert!(job.is_done());
    }

    // 7. CloudProfile::estimated_output_size_mb
    #[test]
    fn test_profile_estimated_size() {
        let profile = av1_profile();
        // 4000 kbps * 60 s / 8000 = 30 MB
        let size = profile.estimated_output_size_mb(60.0);
        assert!((size - 30.0).abs() < f64::EPSILON);
    }

    // 8. CloudTranscodingQueue::submit generates unique IDs
    #[test]
    fn test_queue_submit_unique_ids() {
        let mut q = CloudTranscodingQueue::new();
        let id1 = q.submit("s3://in/a.mp4", "s3://out/", "prof1");
        let id2 = q.submit("s3://in/b.mp4", "s3://out/", "prof1");
        assert_ne!(id1, id2);
        assert_eq!(q.total_jobs(), 2);
    }

    // 9. CloudTranscodingQueue::find_job
    #[test]
    fn test_queue_find_job() {
        let mut q = CloudTranscodingQueue::new();
        let id = q.submit("s3://in/a.mp4", "s3://out/", "prof");
        let job = q.find_job(&id).expect("job should be valid");
        assert_eq!(job.id, id);
        assert!(q.find_job("nonexistent").is_none());
    }

    // 10. CloudTranscodingQueue::pending_jobs
    #[test]
    fn test_queue_pending_jobs() {
        let mut q = CloudTranscodingQueue::new();
        let id1 = q.submit("s3://in/a.mp4", "s3://out/", "prof");
        let _id2 = q.submit("s3://in/b.mp4", "s3://out/", "prof");
        // Mark id1 as completed
        if let Some(j) = q.find_job_mut(&id1) {
            j.status = JobStatus::Completed("s3://out/a.mp4".to_string());
        }
        assert_eq!(q.pending_jobs().len(), 1);
    }

    // 11. CloudTranscodingQueue::completed_jobs
    #[test]
    fn test_queue_completed_jobs() {
        let mut q = CloudTranscodingQueue::new();
        let id1 = q.submit("s3://in/a.mp4", "s3://out/", "prof");
        let id2 = q.submit("s3://in/b.mp4", "s3://out/", "prof");
        if let Some(j) = q.find_job_mut(&id1) {
            j.status = JobStatus::Completed("s3://out/a.mp4".to_string());
        }
        if let Some(j) = q.find_job_mut(&id2) {
            j.status = JobStatus::Completed("s3://out/b.mp4".to_string());
        }
        assert_eq!(q.completed_jobs().len(), 2);
    }

    // 12. CloudTranscodingQueue::add_profile and find_profile
    #[test]
    fn test_queue_add_and_find_profile() {
        let mut q = CloudTranscodingQueue::new();
        q.add_profile(av1_profile());
        let p = q.find_profile("av1-1080p").expect("p should be valid");
        assert_eq!(p.codec, "av1");
        assert!(q.find_profile("nonexistent").is_none());
    }

    // 13. CloudTranscodingQueue – total_jobs
    #[test]
    fn test_queue_total_jobs() {
        let mut q = CloudTranscodingQueue::new();
        assert_eq!(q.total_jobs(), 0);
        q.submit("s3://in/a.mp4", "s3://out/", "prof");
        q.submit("s3://in/b.mp4", "s3://out/", "prof");
        assert_eq!(q.total_jobs(), 2);
    }

    // 14. CloudProfile fields
    #[test]
    fn test_profile_fields() {
        let p = CloudProfile::new("vp9-720p", "vp9", 2_500, (1280, 720), 60.0);
        assert_eq!(p.name, "vp9-720p");
        assert_eq!(p.resolution, (1280, 720));
        assert!((p.fps - 60.0).abs() < f32::EPSILON);
    }

    // 15. JobStatus Running progress boundary
    #[test]
    fn test_running_progress_boundary() {
        assert_eq!(JobStatus::Running(0).progress_pct(), 0);
        assert_eq!(JobStatus::Running(100).progress_pct(), 100);
    }
}
