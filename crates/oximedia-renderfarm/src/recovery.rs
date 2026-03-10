// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Recovery and checkpointing for long-running renders.

use crate::error::{Error, Result};
use crate::job::JobId;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Checkpoint data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Job ID
    pub job_id: JobId,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Completed frames
    pub completed_frames: Vec<u32>,
    /// In-progress frames
    pub in_progress_frames: Vec<u32>,
    /// Failed frames
    pub failed_frames: Vec<u32>,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

/// Error recovery strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecoveryStrategy {
    /// Retry the failed frame immediately on the same worker.
    RetryImmediate,
    /// Retry on a different worker (reschedule).
    RetryDifferentWorker,
    /// Skip the failed frame and continue.
    SkipFrame,
    /// Abort the entire job on failure.
    AbortJob,
    /// Roll back to the last checkpoint and resume from there.
    RollbackToCheckpoint,
}

/// Outcome of a recovery attempt.
#[derive(Debug, Clone)]
pub struct RecoveryOutcome {
    /// Job that was recovered.
    pub job_id: JobId,
    /// Strategy that was applied.
    pub strategy: RecoveryStrategy,
    /// Frames that should be re-rendered after recovery.
    pub frames_to_rerender: Vec<u32>,
    /// Frames that were skipped.
    pub frames_skipped: Vec<u32>,
    /// Whether the recovery was successful.
    pub success: bool,
    /// Human-readable description of the recovery action.
    pub description: String,
}

/// Per-job retry tracking.
#[derive(Debug, Clone)]
struct RetryState {
    /// Number of retries consumed so far.
    attempts: u32,
    /// Maximum retries allowed.
    max_retries: u32,
    /// Frames that have been retried and how many times.
    frame_retries: HashMap<u32, u32>,
}

impl RetryState {
    fn new(max_retries: u32) -> Self {
        Self {
            attempts: 0,
            max_retries,
            frame_retries: HashMap::new(),
        }
    }

    fn can_retry(&self) -> bool {
        self.attempts < self.max_retries
    }

    fn can_retry_frame(&self, frame: u32, max_per_frame: u32) -> bool {
        self.frame_retries.get(&frame).copied().unwrap_or(0) < max_per_frame
    }

    fn record_retry(&mut self, frame: u32) {
        self.attempts += 1;
        *self.frame_retries.entry(frame).or_insert(0) += 1;
    }
}

/// Recovery manager
pub struct RecoveryManager {
    checkpoints: HashMap<JobId, Vec<Checkpoint>>,
    checkpoint_interval: u64,
    /// Per-job retry state.
    retry_states: HashMap<JobId, RetryState>,
    /// Default maximum retries per job.
    max_retries: u32,
    /// Maximum retries per individual frame.
    max_frame_retries: u32,
    /// Default recovery strategy when a frame fails.
    default_strategy: RecoveryStrategy,
}

impl RecoveryManager {
    /// Create a new recovery manager
    #[must_use]
    pub fn new(checkpoint_interval: u64) -> Self {
        Self {
            checkpoints: HashMap::new(),
            checkpoint_interval,
            retry_states: HashMap::new(),
            max_retries: 5,
            max_frame_retries: 3,
            default_strategy: RecoveryStrategy::RetryDifferentWorker,
        }
    }

    /// Set the default recovery strategy.
    pub fn set_default_strategy(&mut self, strategy: RecoveryStrategy) {
        self.default_strategy = strategy;
    }

    /// Set the maximum number of retries per job.
    pub fn set_max_retries(&mut self, max_retries: u32) {
        self.max_retries = max_retries;
    }

    /// Set the maximum number of retries per individual frame.
    pub fn set_max_frame_retries(&mut self, max_frame_retries: u32) {
        self.max_frame_retries = max_frame_retries;
    }

    /// Get the checkpoint interval in seconds.
    #[must_use]
    pub fn checkpoint_interval(&self) -> u64 {
        self.checkpoint_interval
    }

    /// Create checkpoint
    pub fn create_checkpoint(&mut self, checkpoint: Checkpoint) {
        self.checkpoints
            .entry(checkpoint.job_id)
            .or_default()
            .push(checkpoint);
    }

    /// Get latest checkpoint
    #[must_use]
    pub fn get_latest_checkpoint(&self, job_id: JobId) -> Option<&Checkpoint> {
        self.checkpoints.get(&job_id)?.last()
    }

    /// Recover from checkpoint
    pub fn recover(&self, job_id: JobId) -> Result<Checkpoint> {
        self.get_latest_checkpoint(job_id)
            .cloned()
            .ok_or_else(|| Error::Checkpoint(format!("No checkpoint found for job {job_id}")))
    }

    /// List checkpoints for job
    #[must_use]
    pub fn list_checkpoints(&self, job_id: JobId) -> Vec<&Checkpoint> {
        self.checkpoints
            .get(&job_id)
            .map_or_else(Vec::new, |cps| cps.iter().collect())
    }

    /// Handle a frame failure and determine the recovery action.
    ///
    /// This implements a multi-tier recovery strategy:
    /// 1. If the frame can be retried (under per-frame limit), retry on a different worker.
    /// 2. If per-frame limit is exhausted, try rolling back to the last checkpoint.
    /// 3. If no checkpoint exists and the job retry budget is exhausted, skip or abort.
    pub fn handle_frame_failure(
        &mut self,
        job_id: JobId,
        failed_frame: u32,
    ) -> Result<RecoveryOutcome> {
        // Ensure retry state exists for this job
        let retry_state = self
            .retry_states
            .entry(job_id)
            .or_insert_with(|| RetryState::new(self.max_retries));

        // Tier 1: Can we retry this specific frame?
        if retry_state.can_retry_frame(failed_frame, self.max_frame_retries)
            && retry_state.can_retry()
        {
            retry_state.record_retry(failed_frame);
            let strategy = self.default_strategy;
            return Ok(RecoveryOutcome {
                job_id,
                strategy,
                frames_to_rerender: vec![failed_frame],
                frames_skipped: Vec::new(),
                success: true,
                description: format!(
                    "Retrying frame {failed_frame} (attempt {}/{})",
                    retry_state
                        .frame_retries
                        .get(&failed_frame)
                        .copied()
                        .unwrap_or(1),
                    self.max_frame_retries
                ),
            });
        }

        // Tier 2: Per-frame limit exhausted — try checkpoint rollback
        if let Some(checkpoint) = self.get_latest_checkpoint(job_id) {
            // Compute frames that need re-rendering: in-progress + failed
            let mut frames_to_rerender = checkpoint.in_progress_frames.clone();
            for &f in &checkpoint.failed_frames {
                if !frames_to_rerender.contains(&f) {
                    frames_to_rerender.push(f);
                }
            }
            if !frames_to_rerender.contains(&failed_frame) {
                frames_to_rerender.push(failed_frame);
            }
            frames_to_rerender.sort_unstable();
            frames_to_rerender.dedup();

            return Ok(RecoveryOutcome {
                job_id,
                strategy: RecoveryStrategy::RollbackToCheckpoint,
                frames_to_rerender,
                frames_skipped: Vec::new(),
                success: true,
                description: format!(
                    "Rolling back to checkpoint at {} for frame {failed_frame}",
                    checkpoint.timestamp.format("%H:%M:%S")
                ),
            });
        }

        // Tier 3: No checkpoint available — decide based on default strategy
        match self.default_strategy {
            RecoveryStrategy::AbortJob => Err(Error::Recovery(format!(
                "Frame {failed_frame} failed and retry budget exhausted for job {job_id}; aborting"
            ))),
            RecoveryStrategy::SkipFrame => Ok(RecoveryOutcome {
                job_id,
                strategy: RecoveryStrategy::SkipFrame,
                frames_to_rerender: Vec::new(),
                frames_skipped: vec![failed_frame],
                success: true,
                description: format!("Skipping frame {failed_frame} after exhausting retries"),
            }),
            _ => {
                // Fallback: skip the frame rather than abort
                Ok(RecoveryOutcome {
                    job_id,
                    strategy: RecoveryStrategy::SkipFrame,
                    frames_to_rerender: Vec::new(),
                    frames_skipped: vec![failed_frame],
                    success: true,
                    description: format!(
                        "Skipping frame {failed_frame} (no checkpoint, retries exhausted)"
                    ),
                })
            }
        }
    }

    /// Reset the retry state for a job (e.g., after successful recovery).
    pub fn reset_retry_state(&mut self, job_id: JobId) {
        self.retry_states.remove(&job_id);
    }

    /// Get the number of retries consumed for a job.
    #[must_use]
    pub fn retry_count(&self, job_id: JobId) -> u32 {
        self.retry_states.get(&job_id).map_or(0, |s| s.attempts)
    }

    /// Check whether a job can still be retried.
    #[must_use]
    pub fn can_retry(&self, job_id: JobId) -> bool {
        self.retry_states
            .get(&job_id)
            .map_or(true, |s| s.can_retry())
    }

    /// Prune old checkpoints, keeping only the most recent `keep` per job.
    pub fn prune_checkpoints(&mut self, keep: usize) {
        for cps in self.checkpoints.values_mut() {
            if cps.len() > keep {
                let drain_count = cps.len() - keep;
                cps.drain(..drain_count);
            }
        }
    }
}

impl Default for RecoveryManager {
    fn default() -> Self {
        Self::new(300) // 5 minutes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recovery_manager_creation() {
        let manager = RecoveryManager::new(300);
        assert_eq!(manager.checkpoint_interval(), 300);
    }

    #[test]
    fn test_create_checkpoint() {
        let mut manager = RecoveryManager::new(300);
        let job_id = JobId::new();

        let checkpoint = Checkpoint {
            job_id,
            timestamp: Utc::now(),
            completed_frames: vec![1, 2, 3],
            in_progress_frames: vec![4],
            failed_frames: vec![],
            metadata: HashMap::new(),
        };

        manager.create_checkpoint(checkpoint);
        assert!(manager.get_latest_checkpoint(job_id).is_some());
    }

    #[test]
    fn test_recover_from_checkpoint() -> Result<()> {
        let mut manager = RecoveryManager::new(300);
        let job_id = JobId::new();

        let checkpoint = Checkpoint {
            job_id,
            timestamp: Utc::now(),
            completed_frames: vec![1, 2, 3],
            in_progress_frames: vec![4],
            failed_frames: vec![],
            metadata: HashMap::new(),
        };

        manager.create_checkpoint(checkpoint);

        let recovered = manager.recover(job_id)?;
        assert_eq!(recovered.completed_frames.len(), 3);

        Ok(())
    }

    #[test]
    fn test_handle_frame_failure_retries() -> Result<()> {
        let mut manager = RecoveryManager::new(300);
        manager.set_max_retries(5);
        manager.set_max_frame_retries(3);
        let job_id = JobId::new();

        // First failure should trigger a retry
        let outcome = manager.handle_frame_failure(job_id, 42)?;
        assert!(outcome.success);
        assert_eq!(outcome.strategy, RecoveryStrategy::RetryDifferentWorker);
        assert_eq!(outcome.frames_to_rerender, vec![42]);
        assert!(outcome.frames_skipped.is_empty());

        // Second failure of same frame
        let outcome2 = manager.handle_frame_failure(job_id, 42)?;
        assert!(outcome2.success);
        assert_eq!(outcome2.frames_to_rerender, vec![42]);

        Ok(())
    }

    #[test]
    fn test_handle_frame_failure_exhausted_with_checkpoint() -> Result<()> {
        let mut manager = RecoveryManager::new(300);
        manager.set_max_frame_retries(1);
        let job_id = JobId::new();

        // Create a checkpoint
        manager.create_checkpoint(Checkpoint {
            job_id,
            timestamp: Utc::now(),
            completed_frames: vec![1, 2],
            in_progress_frames: vec![3],
            failed_frames: vec![],
            metadata: HashMap::new(),
        });

        // First failure uses retry
        let _ = manager.handle_frame_failure(job_id, 3)?;

        // Second failure of same frame -> per-frame limit exhausted -> rollback
        let outcome = manager.handle_frame_failure(job_id, 3)?;
        assert_eq!(outcome.strategy, RecoveryStrategy::RollbackToCheckpoint);
        assert!(outcome.frames_to_rerender.contains(&3));

        Ok(())
    }

    #[test]
    fn test_handle_frame_failure_skip_when_no_checkpoint() -> Result<()> {
        let mut manager = RecoveryManager::new(300);
        manager.set_max_retries(1);
        manager.set_max_frame_retries(1);
        manager.set_default_strategy(RecoveryStrategy::SkipFrame);
        let job_id = JobId::new();

        // First failure exhausts retry budget
        let _ = manager.handle_frame_failure(job_id, 10)?;

        // Second failure -> no checkpoint, skip
        let outcome = manager.handle_frame_failure(job_id, 10)?;
        assert_eq!(outcome.strategy, RecoveryStrategy::SkipFrame);
        assert!(outcome.frames_skipped.contains(&10));

        Ok(())
    }

    #[test]
    fn test_retry_count_and_can_retry() {
        let mut manager = RecoveryManager::new(300);
        manager.set_max_retries(2);
        let job_id = JobId::new();

        assert!(manager.can_retry(job_id));
        assert_eq!(manager.retry_count(job_id), 0);

        let _ = manager.handle_frame_failure(job_id, 1);
        assert_eq!(manager.retry_count(job_id), 1);
        assert!(manager.can_retry(job_id));

        let _ = manager.handle_frame_failure(job_id, 2);
        assert_eq!(manager.retry_count(job_id), 2);
        assert!(!manager.can_retry(job_id));
    }

    #[test]
    fn test_reset_retry_state() {
        let mut manager = RecoveryManager::new(300);
        let job_id = JobId::new();

        let _ = manager.handle_frame_failure(job_id, 1);
        assert_eq!(manager.retry_count(job_id), 1);

        manager.reset_retry_state(job_id);
        assert_eq!(manager.retry_count(job_id), 0);
    }

    #[test]
    fn test_prune_checkpoints() {
        let mut manager = RecoveryManager::new(300);
        let job_id = JobId::new();

        for i in 0..5 {
            manager.create_checkpoint(Checkpoint {
                job_id,
                timestamp: Utc::now(),
                completed_frames: vec![i],
                in_progress_frames: vec![],
                failed_frames: vec![],
                metadata: HashMap::new(),
            });
        }

        assert_eq!(manager.list_checkpoints(job_id).len(), 5);
        manager.prune_checkpoints(2);
        assert_eq!(manager.list_checkpoints(job_id).len(), 2);

        // Latest checkpoint should still be available
        let latest = manager.get_latest_checkpoint(job_id);
        assert!(latest.is_some());
        assert_eq!(latest.map(|c| c.completed_frames[0]), Some(4));
    }
}
