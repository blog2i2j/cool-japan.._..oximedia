// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Conversion queue management for batch processing.

use crate::{ConversionError, ConversionOptions, Result};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::path::PathBuf;

/// Queue for managing conversion jobs.
#[derive(Debug, Clone)]
pub struct ConversionQueue {
    jobs: VecDeque<QueuedJob>,
    max_size: Option<usize>,
}

impl ConversionQueue {
    /// Create a new conversion queue.
    #[must_use]
    pub fn new() -> Self {
        Self {
            jobs: VecDeque::new(),
            max_size: None,
        }
    }

    /// Create a queue with a maximum size.
    #[must_use]
    pub fn with_max_size(max_size: usize) -> Self {
        Self {
            jobs: VecDeque::new(),
            max_size: Some(max_size),
        }
    }

    /// Add a job to the queue.
    pub fn enqueue(&mut self, job: QueuedJob) -> Result<()> {
        if let Some(max) = self.max_size {
            if self.jobs.len() >= max {
                return Err(ConversionError::InvalidInput("Queue is full".to_string()));
            }
        }

        self.jobs.push_back(job);
        Ok(())
    }

    /// Add a job with priority (to the front).
    pub fn enqueue_priority(&mut self, job: QueuedJob) -> Result<()> {
        if let Some(max) = self.max_size {
            if self.jobs.len() >= max {
                return Err(ConversionError::InvalidInput("Queue is full".to_string()));
            }
        }

        self.jobs.push_front(job);
        Ok(())
    }

    /// Remove and return the next job.
    pub fn dequeue(&mut self) -> Option<QueuedJob> {
        self.jobs.pop_front()
    }

    /// Peek at the next job without removing it.
    #[must_use]
    pub fn peek(&self) -> Option<&QueuedJob> {
        self.jobs.front()
    }

    /// Get the number of jobs in the queue.
    #[must_use]
    pub fn len(&self) -> usize {
        self.jobs.len()
    }

    /// Check if the queue is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.jobs.is_empty()
    }

    /// Clear all jobs from the queue.
    pub fn clear(&mut self) {
        self.jobs.clear();
    }

    /// Get all jobs in the queue.
    #[must_use]
    pub fn jobs(&self) -> &VecDeque<QueuedJob> {
        &self.jobs
    }

    /// Remove a job by index.
    pub fn remove(&mut self, index: usize) -> Option<QueuedJob> {
        self.jobs.remove(index)
    }

    /// Save the queue to a file for persistence.
    pub fn save_to_file<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        let json = serde_json::to_string_pretty(&self.jobs).map_err(|e| {
            ConversionError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, e))
        })?;

        std::fs::write(path, json).map_err(ConversionError::Io)
    }

    /// Load the queue from a file.
    pub fn load_from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let json = std::fs::read_to_string(path).map_err(ConversionError::Io)?;

        let jobs: VecDeque<QueuedJob> = serde_json::from_str(&json).map_err(|e| {
            ConversionError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, e))
        })?;

        Ok(Self {
            jobs,
            max_size: None,
        })
    }

    /// Get jobs by status.
    #[must_use]
    pub fn jobs_by_status(&self, status: JobStatus) -> Vec<&QueuedJob> {
        self.jobs
            .iter()
            .filter(|job| job.status == status)
            .collect()
    }

    /// Update job status.
    pub fn update_status(&mut self, index: usize, status: JobStatus) -> Result<()> {
        if let Some(job) = self.jobs.get_mut(index) {
            job.status = status;
            Ok(())
        } else {
            Err(ConversionError::InvalidInput(
                "Job index out of bounds".to_string(),
            ))
        }
    }
}

impl Default for ConversionQueue {
    fn default() -> Self {
        Self::new()
    }
}

/// A job in the conversion queue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueuedJob {
    /// Unique job ID
    pub id: String,
    /// Input file path
    pub input: PathBuf,
    /// Output file path
    pub output: PathBuf,
    /// Conversion options (simplified for serialization)
    #[serde(skip)]
    pub options: Option<ConversionOptions>,
    /// Job status
    pub status: JobStatus,
    /// Priority (higher = more important)
    pub priority: i32,
    /// Number of retry attempts
    pub retry_count: u32,
    /// Maximum retries allowed
    pub max_retries: u32,
    /// Error message if failed
    pub error: Option<String>,
}

impl QueuedJob {
    /// Create a new queued job.
    pub fn new<P: Into<PathBuf>, Q: Into<PathBuf>>(
        input: P,
        output: Q,
        options: ConversionOptions,
    ) -> Self {
        Self {
            id: generate_job_id(),
            input: input.into(),
            output: output.into(),
            options: Some(options),
            status: JobStatus::Pending,
            priority: 0,
            retry_count: 0,
            max_retries: 3,
            error: None,
        }
    }

    /// Set the job priority.
    #[must_use]
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Set the maximum number of retries.
    #[must_use]
    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Check if the job can be retried.
    #[must_use]
    pub fn can_retry(&self) -> bool {
        self.retry_count < self.max_retries
    }

    /// Increment retry count.
    pub fn increment_retry(&mut self) {
        self.retry_count += 1;
    }
}

/// Status of a queued job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum JobStatus {
    /// Job is waiting to be processed
    Pending,
    /// Job is currently being processed
    Processing,
    /// Job completed successfully
    Completed,
    /// Job failed
    Failed,
    /// Job was cancelled
    Cancelled,
    /// Job is paused
    Paused,
}

fn generate_job_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros();

    format!("job_{timestamp}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_queue_creation() {
        let queue = ConversionQueue::new();
        assert_eq!(queue.len(), 0);
        assert!(queue.is_empty());
    }

    #[test]
    fn test_enqueue_dequeue() {
        let mut queue = ConversionQueue::new();
        let job = QueuedJob::new(
            PathBuf::from("input.mp4"),
            PathBuf::from("output.mp4"),
            ConversionOptions::default(),
        );

        queue.enqueue(job.clone()).unwrap();
        assert_eq!(queue.len(), 1);

        let dequeued = queue.dequeue().unwrap();
        assert_eq!(dequeued.input, job.input);
        assert!(queue.is_empty());
    }

    #[test]
    fn test_priority_enqueue() {
        let mut queue = ConversionQueue::new();
        let job1 = QueuedJob::new(
            PathBuf::from("input1.mp4"),
            PathBuf::from("output1.mp4"),
            ConversionOptions::default(),
        );
        let job2 = QueuedJob::new(
            PathBuf::from("input2.mp4"),
            PathBuf::from("output2.mp4"),
            ConversionOptions::default(),
        );

        queue.enqueue(job1).unwrap();
        queue.enqueue_priority(job2.clone()).unwrap();

        let dequeued = queue.dequeue().unwrap();
        assert_eq!(dequeued.input, job2.input);
    }

    #[test]
    fn test_max_size() {
        let mut queue = ConversionQueue::with_max_size(2);
        let job = QueuedJob::new(
            PathBuf::from("input.mp4"),
            PathBuf::from("output.mp4"),
            ConversionOptions::default(),
        );

        assert!(queue.enqueue(job.clone()).is_ok());
        assert!(queue.enqueue(job.clone()).is_ok());
        assert!(queue.enqueue(job).is_err());
    }

    #[test]
    fn test_job_status_update() {
        let mut queue = ConversionQueue::new();
        let job = QueuedJob::new(
            PathBuf::from("input.mp4"),
            PathBuf::from("output.mp4"),
            ConversionOptions::default(),
        );

        queue.enqueue(job).unwrap();
        queue.update_status(0, JobStatus::Processing).unwrap();

        assert_eq!(queue.jobs()[0].status, JobStatus::Processing);
    }

    #[test]
    fn test_job_retry() {
        let mut job = QueuedJob::new(
            PathBuf::from("input.mp4"),
            PathBuf::from("output.mp4"),
            ConversionOptions::default(),
        )
        .with_max_retries(2);

        assert!(job.can_retry());
        job.increment_retry();
        assert!(job.can_retry());
        job.increment_retry();
        assert!(!job.can_retry());
    }
}
