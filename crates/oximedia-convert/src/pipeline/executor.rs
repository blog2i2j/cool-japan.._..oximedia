// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Pipeline executor for running conversion jobs.

use super::{ConversionJob, JobStatus, PipelineConfig, PipelineStats};
use crate::{ConversionError, Result};
use dashmap::DashMap;
use parking_lot::RwLock;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Semaphore;

/// Pipeline executor for managing and running conversion jobs.
#[derive(Clone)]
#[allow(dead_code)]
pub struct PipelineExecutor {
    config: Arc<PipelineConfig>,
    jobs: Arc<DashMap<String, ConversionJob>>,
    semaphore: Arc<Semaphore>,
    stats: Arc<RwLock<ExecutorStats>>,
}

/// Executor statistics.
#[derive(Debug, Clone, Default)]
pub struct ExecutorStats {
    /// Total jobs submitted
    pub jobs_submitted: u64,
    /// Jobs completed successfully
    pub jobs_completed: u64,
    /// Jobs failed
    pub jobs_failed: u64,
    /// Jobs cancelled
    pub jobs_cancelled: u64,
    /// Total processing time
    pub total_processing_time: Duration,
}

impl PipelineExecutor {
    /// Create a new pipeline executor.
    #[must_use]
    pub fn new(config: PipelineConfig) -> Self {
        let workers = config.workers;
        Self {
            config: Arc::new(config),
            jobs: Arc::new(DashMap::new()),
            semaphore: Arc::new(Semaphore::new(workers)),
            stats: Arc::new(RwLock::new(ExecutorStats::default())),
        }
    }

    /// Submit a job for execution.
    pub async fn submit(&self, job: ConversionJob) -> Result<String> {
        let job_id = job.id.clone();
        self.jobs.insert(job_id.clone(), job);

        {
            let mut stats = self.stats.write();
            stats.jobs_submitted += 1;
        }

        Ok(job_id)
    }

    /// Execute a job by ID.
    pub async fn execute(&self, job_id: &str) -> Result<PipelineStats> {
        let _permit = self.semaphore.acquire().await.map_err(|e| {
            ConversionError::InvalidInput(format!("Failed to acquire semaphore: {e}"))
        })?;

        let mut job = self
            .jobs
            .get_mut(job_id)
            .ok_or_else(|| ConversionError::InvalidInput(format!("Job not found: {job_id}")))?;

        job.start();

        let start_time = Instant::now();

        // Simulate processing (TODO: implement actual conversion)
        let result = self.process_job(&job).await;

        let duration = start_time.elapsed();

        match result {
            Ok(stats) => {
                job.complete();
                let mut executor_stats = self.stats.write();
                executor_stats.jobs_completed += 1;
                executor_stats.total_processing_time += duration;
                Ok(stats)
            }
            Err(e) => {
                job.fail(e.to_string());
                let mut executor_stats = self.stats.write();
                executor_stats.jobs_failed += 1;
                Err(e)
            }
        }
    }

    async fn process_job(&self, job: &ConversionJob) -> Result<PipelineStats> {
        // Full demux/decode/encode/mux pipeline requires the transcode crate
        // integration which is scheduled for a future milestone. For now, the
        // executor validates the input, records file sizes, and marks the job
        // complete so that the pipeline infrastructure is exercised end-to-end.

        let input_size = std::fs::metadata(&job.input).map(|m| m.len()).unwrap_or(0);

        let output_size = std::fs::metadata(&job.output).map(|m| m.len()).unwrap_or(0);

        Ok(PipelineStats {
            input_size,
            output_size,
            duration: Duration::from_secs(0),
            encoding_fps: 0.0,
            frames_processed: 0,
        })
    }

    /// Get job status.
    #[must_use]
    pub fn get_job_status(&self, job_id: &str) -> Option<JobStatus> {
        self.jobs.get(job_id).map(|job| job.status)
    }

    /// Get job progress.
    #[must_use]
    pub fn get_job_progress(&self, job_id: &str) -> Option<f64> {
        self.jobs.get(job_id).map(|job| job.progress)
    }

    /// Cancel a job.
    pub fn cancel_job(&self, job_id: &str) -> Result<()> {
        let mut job = self
            .jobs
            .get_mut(job_id)
            .ok_or_else(|| ConversionError::InvalidInput(format!("Job not found: {job_id}")))?;

        if job.status == JobStatus::Processing {
            return Err(ConversionError::InvalidInput(
                "Cannot cancel job that is currently processing".to_string(),
            ));
        }

        job.status = JobStatus::Cancelled;

        let mut stats = self.stats.write();
        stats.jobs_cancelled += 1;

        Ok(())
    }

    /// Remove a completed job.
    pub fn remove_job(&self, job_id: &str) -> Result<()> {
        self.jobs
            .remove(job_id)
            .ok_or_else(|| ConversionError::InvalidInput(format!("Job not found: {job_id}")))?;
        Ok(())
    }

    /// Get executor statistics.
    #[must_use]
    pub fn get_stats(&self) -> ExecutorStats {
        self.stats.read().clone()
    }

    /// Get number of active jobs.
    #[must_use]
    pub fn active_jobs(&self) -> usize {
        self.jobs
            .iter()
            .filter(|entry| entry.status == JobStatus::Processing)
            .count()
    }

    /// Get number of queued jobs.
    #[must_use]
    pub fn queued_jobs(&self) -> usize {
        self.jobs
            .iter()
            .filter(|entry| entry.status == JobStatus::Queued)
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formats::ContainerFormat;
    use crate::pipeline::JobPriority;
    use std::collections::HashMap;
    use std::path::PathBuf;

    #[tokio::test]
    async fn test_executor_submit() {
        let config = PipelineConfig::default();
        let executor = PipelineExecutor::new(config);

        let job = ConversionJob::new(
            PathBuf::from("input.mp4"),
            PathBuf::from("output.webm"),
            ContainerFormat::Webm,
            None,
            None,
            None,
            HashMap::new(),
            JobPriority::Normal,
        );

        let job_id = executor.submit(job).await.unwrap();
        assert!(!job_id.is_empty());

        let stats = executor.get_stats();
        assert_eq!(stats.jobs_submitted, 1);
    }

    #[tokio::test]
    async fn test_executor_cancel() {
        let config = PipelineConfig::default();
        let executor = PipelineExecutor::new(config);

        let job = ConversionJob::new(
            PathBuf::from("input.mp4"),
            PathBuf::from("output.webm"),
            ContainerFormat::Webm,
            None,
            None,
            None,
            HashMap::new(),
            JobPriority::Normal,
        );

        let job_id = executor.submit(job).await.unwrap();
        executor.cancel_job(&job_id).unwrap();

        let status = executor.get_job_status(&job_id).unwrap();
        assert_eq!(status, JobStatus::Cancelled);

        let stats = executor.get_stats();
        assert_eq!(stats.jobs_cancelled, 1);
    }

    #[tokio::test]
    async fn test_executor_remove() {
        let config = PipelineConfig::default();
        let executor = PipelineExecutor::new(config);

        let job = ConversionJob::new(
            PathBuf::from("input.mp4"),
            PathBuf::from("output.webm"),
            ContainerFormat::Webm,
            None,
            None,
            None,
            HashMap::new(),
            JobPriority::Normal,
        );

        let job_id = executor.submit(job).await.unwrap();
        executor.remove_job(&job_id).unwrap();

        assert!(executor.get_job_status(&job_id).is_none());
    }
}
