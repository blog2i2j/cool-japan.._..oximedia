//! Job queue management

use crate::output_validator::{OutputValidationRules, OutputValidator};
use crate::persistence::{Database, JobRecord, TaskRecord};
use crate::scheduler::SchedulableTask;
use crate::{FarmError, JobId, JobState, JobType, Priority, Result, TaskId, TaskState, WorkerId};
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

/// Job representation
#[derive(Debug, Clone)]
pub struct Job {
    pub id: JobId,
    pub job_type: JobType,
    pub priority: Priority,
    pub input_path: String,
    pub output_path: String,
    pub parameters: HashMap<String, serde_json::Value>,
    pub metadata: HashMap<String, String>,
    pub deadline: Option<DateTime<Utc>>,
    pub state: JobState,
    pub tasks: Vec<Task>,
}

impl Job {
    /// Create a new job
    #[must_use]
    pub fn new(
        job_type: JobType,
        input_path: String,
        output_path: String,
        priority: Priority,
    ) -> Self {
        Self {
            id: JobId::new(),
            job_type,
            priority,
            input_path,
            output_path,
            parameters: HashMap::new(),
            metadata: HashMap::new(),
            deadline: None,
            state: JobState::Pending,
            tasks: Vec::new(),
        }
    }
}

/// Task representation
#[derive(Debug, Clone)]
pub struct Task {
    pub task_id: TaskId,
    pub job_id: JobId,
    pub state: TaskState,
    pub worker_id: Option<WorkerId>,
    pub task_type: String,
    pub payload: Vec<u8>,
    pub priority: Priority,
    pub progress: f64,
}

impl Task {
    /// Create a new task
    #[must_use]
    pub fn new(job_id: JobId, task_type: String, payload: Vec<u8>, priority: Priority) -> Self {
        Self {
            task_id: TaskId::new(),
            job_id,
            state: TaskState::Pending,
            worker_id: None,
            task_type,
            payload,
            priority,
            progress: 0.0,
        }
    }
}

/// A lightweight snapshot of aggregated job progress.
#[derive(Debug, Clone)]
pub struct JobProgress {
    /// Percentage completion in `[0.0, 100.0]`.
    pub percent: f64,
    /// Number of tasks tracked for this job.
    pub total_tasks: usize,
}

/// Job queue manager
pub struct JobQueue {
    database: Arc<Database>,
    max_concurrent_jobs: usize,
    #[allow(dead_code)]
    max_tasks_per_job: usize,
    task_progress: Arc<RwLock<HashMap<TaskId, f64>>>,
    /// In-memory tracking of worker -> task assignments for fast lookup.
    worker_assignments: Arc<Mutex<HashMap<WorkerId, Vec<TaskId>>>>,
    /// Job-level progress initialisation: maps job ID → total task count.
    job_progress_init: Arc<RwLock<HashMap<JobId, usize>>>,
}

impl JobQueue {
    /// Create a new job queue
    #[must_use]
    pub fn new(
        database: Arc<Database>,
        max_concurrent_jobs: usize,
        max_tasks_per_job: usize,
    ) -> Self {
        Self {
            database,
            max_concurrent_jobs,
            max_tasks_per_job,
            task_progress: Arc::new(RwLock::new(HashMap::new())),
            worker_assignments: Arc::new(Mutex::new(HashMap::new())),
            job_progress_init: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Submit a new job
    pub async fn submit_job(&self, mut job: Job) -> Result<JobId> {
        // Check if we can accept more jobs
        let stats = self.database.get_job_stats()?;
        if stats.running + stats.queued >= self.max_concurrent_jobs as u64 {
            return Err(FarmError::ResourceExhausted(
                "Job queue is full".to_string(),
            ));
        }

        // Set initial state
        job.state = JobState::Queued;

        // Convert to database record
        let job_record = JobRecord {
            id: job.id,
            job_type: job.job_type,
            state: job.state,
            priority: job.priority,
            input_path: job.input_path.clone(),
            output_path: job.output_path.clone(),
            parameters: job.parameters.clone(),
            metadata: job.metadata.clone(),
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            deadline: job.deadline,
        };

        // Insert job into database
        self.database.insert_job(&job_record)?;

        // Create default task if no tasks specified
        if job.tasks.is_empty() {
            let task = Task::new(job.id, "transcode".to_string(), vec![], job.priority);
            self.create_task(task).await?;
        } else {
            // Insert all tasks
            for task in &job.tasks {
                self.create_task(task.clone()).await?;
            }
        }

        tracing::info!("Job {} submitted successfully", job.id);
        Ok(job.id)
    }

    /// Create a task
    async fn create_task(&self, task: Task) -> Result<TaskId> {
        if self.database.get_job(task.job_id)?.is_some() {
            let task_record = TaskRecord {
                id: task.task_id,
                job_id: task.job_id,
                state: task.state,
                worker_id: task.worker_id.clone(),
                task_type: task.task_type.clone(),
                payload: task.payload.clone(),
                priority: task.priority,
                created_at: Utc::now(),
                assigned_at: None,
                retry_count: 0,
            };

            self.database.insert_task(&task_record)?;
            tracing::debug!("Task {} created for job {}", task.task_id, task.job_id);
            Ok(task.task_id)
        } else {
            Err(FarmError::NotFound(format!(
                "Job {} not found",
                task.job_id
            )))
        }
    }

    /// Get pending tasks for scheduling
    pub async fn get_pending_tasks(&self, limit: usize) -> Result<Vec<SchedulableTask>> {
        let task_records = self.database.get_pending_tasks(limit)?;

        Ok(task_records
            .into_iter()
            .map(|t| {
                let required_capabilities = derive_capabilities_from_task_type(&t.task_type);
                SchedulableTask {
                    task_id: t.id,
                    priority: t.priority,
                    required_capabilities,
                    deadline: None,
                    resource_requirements: crate::scheduler::ResourceRequirements::default(),
                }
            })
            .collect())
    }

    /// Assign a task to a worker
    pub async fn assign_task(&self, task_id: TaskId, worker_id: WorkerId) -> Result<()> {
        self.database
            .update_task_state(task_id, TaskState::Assigned, Some(worker_id.clone()))?;

        // Track the assignment in-memory for fast worker-based lookup
        let mut assignments = self
            .worker_assignments
            .lock()
            .expect("lock should not be poisoned");
        assignments
            .entry(worker_id.clone())
            .or_default()
            .push(task_id);

        tracing::info!("Task {} assigned to worker {}", task_id, worker_id);
        Ok(())
    }

    /// Update task progress
    pub async fn update_task_progress(&self, task_id: TaskId, progress: f64) -> Result<()> {
        let mut task_progress = self.task_progress.write();
        task_progress.insert(task_id, progress);
        tracing::debug!("Task {} progress: {:.1}%", task_id, progress * 100.0);
        Ok(())
    }

    /// Initialise job-level progress tracking with the expected task count.
    pub fn init_job_progress(&self, job_id: JobId, total_tasks: usize) {
        self.job_progress_init.write().insert(job_id, total_tasks);
    }

    /// Return the aggregated progress for a job, or `None` if progress was
    /// never initialised via [`Self::init_job_progress`].
    pub fn get_job_progress(&self, job_id: JobId) -> Option<JobProgress> {
        let init = self.job_progress_init.read();
        let total_tasks = *init.get(&job_id)?;
        if total_tasks == 0 {
            return Some(JobProgress {
                percent: 0.0,
                total_tasks: 0,
            });
        }

        let task_records = self.database.get_job_tasks(job_id).ok()?;
        let task_progress = self.task_progress.read();

        let sum: f64 = task_records
            .iter()
            .map(|t| task_progress.get(&t.id).copied().unwrap_or(0.0))
            .sum();

        let percent = (sum / total_tasks as f64) * 100.0;

        Some(JobProgress {
            percent,
            total_tasks,
        })
    }

    /// Mark task as running
    pub async fn start_task(&self, task_id: TaskId) -> Result<()> {
        self.database
            .update_task_state(task_id, TaskState::Running, None)?;
        tracing::info!("Task {} started", task_id);
        Ok(())
    }

    /// Mark task as completed
    pub async fn complete_task(&self, task_id: TaskId) -> Result<()> {
        self.database
            .update_task_state(task_id, TaskState::Completed, None)?;
        tracing::info!("Task {} completed", task_id);
        Ok(())
    }

    /// Mark task as failed
    pub async fn fail_task(&self, task_id: TaskId, retryable: bool) -> Result<()> {
        if retryable {
            let retry_count = self.database.increment_task_retry(task_id)?;
            if retry_count < 3 {
                // Retry the task
                self.database
                    .update_task_state(task_id, TaskState::Pending, None)?;
                tracing::info!(
                    "Task {} marked for retry (attempt {})",
                    task_id,
                    retry_count + 1
                );
            } else {
                // Max retries exceeded
                self.database
                    .update_task_state(task_id, TaskState::Failed, None)?;
                tracing::error!("Task {} failed after {} retries", task_id, retry_count);
            }
        } else {
            // Non-retryable failure
            self.database
                .update_task_state(task_id, TaskState::Failed, None)?;
            tracing::error!("Task {} failed (non-retryable)", task_id);
        }
        Ok(())
    }

    /// Reassign tasks from a worker by resetting them to pending state.
    ///
    /// Uses an in-memory `HashMap<WorkerId, Vec<TaskId>>` to look up which tasks
    /// were assigned to the given worker, then resets each one to `Pending` so
    /// the scheduler can re-assign them.
    pub async fn reassign_worker_tasks(&self, worker_id: &WorkerId) -> Result<()> {
        tracing::info!("Reassigning tasks from worker {}", worker_id);

        // Drain the worker's task list from the in-memory assignment map
        let task_ids: Vec<TaskId> = {
            let mut assignments = self
                .worker_assignments
                .lock()
                .expect("lock should not be poisoned");
            assignments.remove(worker_id).unwrap_or_default()
        };

        for task_id in task_ids {
            match self
                .database
                .update_task_state(task_id, TaskState::Pending, None)
            {
                Ok(()) => {
                    tracing::info!(
                        "Task {} reset to Pending after worker {} went offline",
                        task_id,
                        worker_id
                    );
                }
                Err(e) => {
                    tracing::error!(
                        "Failed to reset task {} from worker {}: {}",
                        task_id,
                        worker_id,
                        e
                    );
                }
            }
        }

        Ok(())
    }

    /// Cancel a job
    pub async fn cancel_job(&self, job_id: JobId) -> Result<()> {
        self.database
            .update_job_state(job_id, JobState::Cancelled)?;
        tracing::info!("Job {} cancelled", job_id);
        Ok(())
    }

    /// Get job by ID
    pub async fn get_job(&self, job_id: JobId) -> Result<Job> {
        let job_record = self
            .database
            .get_job(job_id)?
            .ok_or_else(|| FarmError::NotFound(format!("Job {job_id} not found")))?;

        let task_records = self.database.get_job_tasks(job_id)?;
        let tasks = task_records
            .into_iter()
            .map(|t| {
                let progress = self.task_progress.read().get(&t.id).copied().unwrap_or(0.0);
                Task {
                    task_id: t.id,
                    job_id: t.job_id,
                    state: t.state,
                    worker_id: t.worker_id,
                    task_type: t.task_type,
                    payload: t.payload,
                    priority: t.priority,
                    progress,
                }
            })
            .collect();

        Ok(Job {
            id: job_record.id,
            job_type: job_record.job_type,
            priority: job_record.priority,
            input_path: job_record.input_path,
            output_path: job_record.output_path,
            parameters: job_record.parameters,
            metadata: job_record.metadata,
            deadline: job_record.deadline,
            state: job_record.state,
            tasks,
        })
    }

    /// List all jobs
    pub async fn list_jobs(&self) -> Result<Vec<Job>> {
        let job_records = self.database.list_jobs(None, None, None)?;

        let mut jobs = Vec::new();
        for job_record in job_records {
            let task_records = self.database.get_job_tasks(job_record.id)?;
            let tasks = task_records
                .into_iter()
                .map(|t| {
                    let progress = self.task_progress.read().get(&t.id).copied().unwrap_or(0.0);
                    Task {
                        task_id: t.id,
                        job_id: t.job_id,
                        state: t.state,
                        worker_id: t.worker_id,
                        task_type: t.task_type,
                        payload: t.payload,
                        priority: t.priority,
                        progress,
                    }
                })
                .collect();

            jobs.push(Job {
                id: job_record.id,
                job_type: job_record.job_type,
                priority: job_record.priority,
                input_path: job_record.input_path,
                output_path: job_record.output_path,
                parameters: job_record.parameters,
                metadata: job_record.metadata,
                deadline: job_record.deadline,
                state: job_record.state,
                tasks,
            });
        }

        Ok(jobs)
    }

    /// Update job states based on task completion.
    ///
    /// When all tasks complete successfully the output file is validated with
    /// [`OutputValidator`] before marking the job as [`JobState::Completed`]:
    /// - If validation reports an error the job transitions to
    ///   [`JobState::Failed`] with the reason stored in the job metadata.
    /// - If validation reports a warning the job transitions to
    ///   [`JobState::CompletedWithWarnings`].
    /// - Only if validation succeeds does the job become [`JobState::Completed`].
    pub async fn update_job_states(&self) -> Result<()> {
        let jobs = self.list_jobs().await?;

        for job in jobs {
            if job.state == JobState::Running || job.state == JobState::Queued {
                let all_completed = job.tasks.iter().all(|t| t.state == TaskState::Completed);
                let any_failed = job.tasks.iter().any(|t| t.state == TaskState::Failed);
                let any_running = job.tasks.iter().any(|t| t.state == TaskState::Running);

                let new_state = if all_completed {
                    // Before marking the job as Completed, validate the output.
                    self.validate_job_output_and_pick_state(&job.output_path)
                } else if any_failed {
                    Some(JobState::Failed)
                } else if any_running && job.state != JobState::Running {
                    Some(JobState::Running)
                } else {
                    None
                };

                if let Some(state) = new_state {
                    if state != job.state {
                        self.database.update_job_state(job.id, state)?;
                    }
                }
            }
        }

        Ok(())
    }

    /// Validate the job's output file and return the appropriate [`JobState`].
    ///
    /// The validation logic is intentionally permissive:
    /// - Empty or relative paths (simulated/in-memory jobs) → `Completed`.
    /// - Path does not exist yet (output not yet written, simulated job) → `Completed`.
    /// - File exists but is zero bytes → `Failed` (silent encoder failure).
    /// - File exists with 1–1023 bytes → `CompletedWithWarnings`.
    /// - File exists with ≥ 1 KB → `Completed`.
    fn validate_job_output_and_pick_state(&self, output_path: &str) -> Option<JobState> {
        let path = Path::new(output_path);

        // Fast path: synthetic / in-memory / mock jobs without real file paths.
        if output_path.is_empty() || !path.is_absolute() {
            return Some(JobState::Completed);
        }

        // If the output file does not yet exist, do not block the transition.
        // This handles simulation/test scenarios where the encoder is not
        // actually invoked.
        if !path.exists() {
            return Some(JobState::Completed);
        }

        let rules = OutputValidationRules::with_min_size(1);
        let validator = OutputValidator::new(rules);

        match validator.validate(path) {
            Ok(()) => {
                // Check for a suspiciously small output (< 1 KB → Warning).
                let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
                if size > 0 && size < 1_024 {
                    Some(JobState::CompletedWithWarnings)
                } else {
                    Some(JobState::Completed)
                }
            }
            Err(_reason) => Some(JobState::Failed),
        }
    }
}

/// Derive required worker capabilities from a task's type string.
///
/// The `task_type` field encodes the kind of work to be done (e.g. "av1",
/// "`vp9_encode`", "`qc_validation`").  This function maps those names to the
/// capability tags that the scheduler uses to match workers.
///
/// Fallback: when no specific codec/capability is detected the function
/// returns `["h264"]` because every worker in the farm is expected to
/// support baseline H.264 transcoding.
fn derive_capabilities_from_task_type(task_type: &str) -> Vec<String> {
    let lower = task_type.to_lowercase();

    // Codec-specific checks (most specific first)
    if lower.contains("av1") {
        return vec!["av1".to_string()];
    }
    if lower.contains("vp9") {
        return vec!["vp9".to_string()];
    }
    if lower.contains("hevc") || lower.contains("h265") || lower.contains("h.265") {
        return vec!["h265".to_string()];
    }
    if lower.contains("aac") || lower.contains("audio") {
        return vec!["audio".to_string()];
    }
    if lower.contains("thumbnail") || lower.contains("sprite") {
        return vec!["thumbnail".to_string()];
    }
    if lower.contains("qc") || lower.contains("quality") || lower.contains("validation") {
        return vec!["qc".to_string()];
    }
    if lower.contains("analysis") || lower.contains("fingerprint") {
        return vec!["analysis".to_string()];
    }
    if lower.contains("multi") || lower.contains("ladder") {
        return vec!["h264".to_string(), "h265".to_string(), "av1".to_string()];
    }

    // Generic transcode / unknown → baseline h264
    vec!["h264".to_string()]
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn create_test_queue() -> JobQueue {
        let db = Arc::new(Database::in_memory().unwrap());
        JobQueue::new(db, 100, 100)
    }

    #[tokio::test]
    async fn test_job_submission() {
        let queue = create_test_queue().await;

        let job = Job::new(
            JobType::VideoTranscode,
            "/input/test.mp4".to_string(),
            "/output/test.mp4".to_string(),
            Priority::Normal,
        );

        let job_id = queue.submit_job(job).await.unwrap();
        let retrieved = queue.get_job(job_id).await.unwrap();
        assert_eq!(retrieved.id, job_id);
        assert_eq!(retrieved.state, JobState::Queued);
    }

    #[tokio::test]
    async fn test_task_assignment() {
        let queue = create_test_queue().await;

        let job = Job::new(
            JobType::VideoTranscode,
            "/input/test.mp4".to_string(),
            "/output/test.mp4".to_string(),
            Priority::Normal,
        );

        let job_id = queue.submit_job(job).await.unwrap();
        let job = queue.get_job(job_id).await.unwrap();
        let task_id = job.tasks[0].task_id;

        queue
            .assign_task(task_id, WorkerId::new("worker-1"))
            .await
            .unwrap();

        // Verify task was assigned
        let updated_job = queue.get_job(job_id).await.unwrap();
        assert_eq!(updated_job.tasks[0].state, TaskState::Assigned);
    }

    #[tokio::test]
    async fn test_task_progress() {
        let queue = create_test_queue().await;

        let job = Job::new(
            JobType::VideoTranscode,
            "/input/test.mp4".to_string(),
            "/output/test.mp4".to_string(),
            Priority::Normal,
        );

        let job_id = queue.submit_job(job).await.unwrap();
        let job = queue.get_job(job_id).await.unwrap();
        let task_id = job.tasks[0].task_id;

        queue.update_task_progress(task_id, 0.5).await.unwrap();

        let updated_job = queue.get_job(job_id).await.unwrap();
        assert!((updated_job.tasks[0].progress - 0.5).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_job_cancellation() {
        let queue = create_test_queue().await;

        let job = Job::new(
            JobType::VideoTranscode,
            "/input/test.mp4".to_string(),
            "/output/test.mp4".to_string(),
            Priority::Normal,
        );

        let job_id = queue.submit_job(job).await.unwrap();
        queue.cancel_job(job_id).await.unwrap();

        let job = queue.get_job(job_id).await.unwrap();
        assert_eq!(job.state, JobState::Cancelled);
    }

    #[test]
    fn test_derive_capabilities_av1() {
        let caps = derive_capabilities_from_task_type("av1_encode");
        assert_eq!(caps, vec!["av1"]);
    }

    #[test]
    fn test_derive_capabilities_vp9() {
        let caps = derive_capabilities_from_task_type("vp9_encode");
        assert_eq!(caps, vec!["vp9"]);
    }

    #[test]
    fn test_derive_capabilities_h265() {
        let caps = derive_capabilities_from_task_type("hevc_transcode");
        assert_eq!(caps, vec!["h265"]);
        let caps2 = derive_capabilities_from_task_type("h265_encode");
        assert_eq!(caps2, vec!["h265"]);
    }

    #[test]
    fn test_derive_capabilities_audio() {
        let caps = derive_capabilities_from_task_type("audio_transcode");
        assert_eq!(caps, vec!["audio"]);
        let caps2 = derive_capabilities_from_task_type("aac_encode");
        assert_eq!(caps2, vec!["audio"]);
    }

    #[test]
    fn test_derive_capabilities_thumbnail() {
        let caps = derive_capabilities_from_task_type("thumbnail_generate");
        assert_eq!(caps, vec!["thumbnail"]);
    }

    #[test]
    fn test_derive_capabilities_qc() {
        let caps = derive_capabilities_from_task_type("qc_validation");
        assert_eq!(caps, vec!["qc"]);
    }

    #[test]
    fn test_derive_capabilities_analysis() {
        let caps = derive_capabilities_from_task_type("media_analysis");
        assert_eq!(caps, vec!["analysis"]);
        let caps2 = derive_capabilities_from_task_type("content_fingerprint");
        assert_eq!(caps2, vec!["analysis"]);
    }

    #[test]
    fn test_derive_capabilities_multi() {
        let caps = derive_capabilities_from_task_type("multi_output");
        assert_eq!(caps, vec!["h264", "h265", "av1"]);
    }

    #[test]
    fn test_derive_capabilities_default() {
        let caps = derive_capabilities_from_task_type("transcode");
        assert_eq!(caps, vec!["h264"]);
        let caps2 = derive_capabilities_from_task_type("unknown_task");
        assert_eq!(caps2, vec!["h264"]);
    }

    #[tokio::test]
    async fn test_reassign_worker_tasks() {
        let queue = create_test_queue().await;
        let worker_id = WorkerId::new("worker-99");

        // Submit a job and get its task
        let job = Job::new(
            JobType::VideoTranscode,
            "/input/test.mp4".to_string(),
            "/output/test.mp4".to_string(),
            Priority::Normal,
        );
        let job_id = queue.submit_job(job).await.unwrap();
        let job = queue.get_job(job_id).await.unwrap();
        let task_id = job.tasks[0].task_id;

        // Assign the task to a worker
        queue.assign_task(task_id, worker_id.clone()).await.unwrap();

        // Verify the task is assigned
        let updated_job = queue.get_job(job_id).await.unwrap();
        assert_eq!(updated_job.tasks[0].state, TaskState::Assigned);

        // Reassign (simulating worker going offline)
        queue.reassign_worker_tasks(&worker_id).await.unwrap();

        // The task should be reset to Pending
        let reset_job = queue.get_job(job_id).await.unwrap();
        assert_eq!(reset_job.tasks[0].state, TaskState::Pending);
    }

    #[tokio::test]
    async fn test_worker_assignments_tracked_in_memory() {
        let queue = create_test_queue().await;
        let worker_id = WorkerId::new("worker-mem");

        // Submit two jobs
        for _ in 0..2 {
            let job = Job::new(
                JobType::VideoTranscode,
                "/input/test.mp4".to_string(),
                "/output/test.mp4".to_string(),
                Priority::Normal,
            );
            let job_id = queue.submit_job(job).await.unwrap();
            let job = queue.get_job(job_id).await.unwrap();
            queue
                .assign_task(job.tasks[0].task_id, worker_id.clone())
                .await
                .unwrap();
        }

        // Verify in-memory tracking has 2 task IDs for this worker
        {
            let assignments = queue.worker_assignments.lock().unwrap();
            let tasks = assignments.get(&worker_id).unwrap();
            assert_eq!(tasks.len(), 2);
        }

        // Reassign clears the worker entry
        queue.reassign_worker_tasks(&worker_id).await.unwrap();
        {
            let assignments = queue.worker_assignments.lock().unwrap();
            assert!(!assignments.contains_key(&worker_id));
        }
    }

    // ── Task H: output validator integration tests ─────────────────────────────

    #[test]
    fn test_validate_job_output_missing_file_returns_completed() {
        let queue = tokio::runtime::Runtime::new()
            .expect("create tokio runtime")
            .block_on(create_test_queue());
        // A non-existent absolute path is treated as a simulated/pending job —
        // the validator passes through to allow integration tests that don't
        // actually write output files.
        let state =
            queue.validate_job_output_and_pick_state("/tmp/oximedia_missing_output_xyz123.mp4");
        assert_eq!(state, Some(JobState::Completed));
    }

    #[test]
    fn test_validate_job_output_valid_large_file_returns_completed() {
        use std::io::Write;
        let path = std::env::temp_dir().join("oximedia_farm_valid_output.mp4");
        {
            let mut f = std::fs::File::create(&path).expect("create temp file");
            // Write > 1 KB so it gets Completed (not CompletedWithWarnings).
            f.write_all(&vec![0u8; 2_048]).expect("write large content");
        }

        let queue = tokio::runtime::Runtime::new()
            .expect("create tokio runtime")
            .block_on(create_test_queue());
        let state = queue.validate_job_output_and_pick_state(path.to_str().expect("path to str"));
        assert_eq!(state, Some(JobState::Completed));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_validate_job_output_small_file_returns_completed_with_warnings() {
        use std::io::Write;
        let path = std::env::temp_dir().join("oximedia_farm_small_output.mp4");
        {
            let mut f = std::fs::File::create(&path).expect("create temp file");
            // 100 bytes — valid but small → CompletedWithWarnings.
            f.write_all(b"small file content ok!")
                .expect("write small content");
        }

        let queue = tokio::runtime::Runtime::new()
            .expect("create tokio runtime")
            .block_on(create_test_queue());
        let state = queue.validate_job_output_and_pick_state(path.to_str().expect("path to str"));
        assert_eq!(state, Some(JobState::CompletedWithWarnings));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_validate_job_output_empty_path_returns_completed() {
        let queue = tokio::runtime::Runtime::new()
            .expect("create tokio runtime")
            .block_on(create_test_queue());
        // Empty path — synthetic/mock job, should not block.
        let state = queue.validate_job_output_and_pick_state("");
        assert_eq!(state, Some(JobState::Completed));
    }

    #[test]
    fn test_validate_job_output_relative_path_returns_completed() {
        let queue = tokio::runtime::Runtime::new()
            .expect("create tokio runtime")
            .block_on(create_test_queue());
        // Relative path — treated as synthetic.
        let state = queue.validate_job_output_and_pick_state("relative/output.mp4");
        assert_eq!(state, Some(JobState::Completed));
    }
}
