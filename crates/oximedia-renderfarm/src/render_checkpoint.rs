#![allow(dead_code)]
//! Checkpoint and resume support for long-running render jobs.
//!
//! Enables saving intermediate render state so that jobs can be resumed after
//! crashes, preemption, or intentional pauses without restarting from scratch.

use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, SystemTime};

/// Unique identifier for a checkpoint.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CheckpointId(String);

impl CheckpointId {
    /// Creates a new checkpoint ID from a string.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Returns the ID as a string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for CheckpointId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// The state of a checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointState {
    /// Checkpoint is being written.
    Writing,
    /// Checkpoint is complete and valid.
    Complete,
    /// Checkpoint is corrupted or incomplete.
    Corrupted,
    /// Checkpoint has been superseded by a newer one.
    Superseded,
}

impl fmt::Display for CheckpointState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Writing => write!(f, "writing"),
            Self::Complete => write!(f, "complete"),
            Self::Corrupted => write!(f, "corrupted"),
            Self::Superseded => write!(f, "superseded"),
        }
    }
}

/// Metadata about a single frame's render progress within a checkpoint.
#[derive(Debug, Clone)]
pub struct FrameProgress {
    /// Frame number.
    pub frame: u64,
    /// Percentage complete (0..100).
    pub percent_complete: f32,
    /// Number of samples completed (for path-tracing renders).
    pub samples_done: u64,
    /// Total samples required.
    pub samples_total: u64,
}

impl FrameProgress {
    /// Creates a new frame progress entry.
    #[allow(clippy::cast_precision_loss)]
    pub fn new(frame: u64, samples_done: u64, samples_total: u64) -> Self {
        let pct = if samples_total > 0 {
            (samples_done as f64 / samples_total as f64 * 100.0) as f32
        } else {
            0.0
        };
        Self {
            frame,
            percent_complete: pct,
            samples_done,
            samples_total,
        }
    }

    /// Returns true if this frame is fully rendered.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.samples_done >= self.samples_total && self.samples_total > 0
    }
}

/// A render checkpoint containing progress and state data.
#[derive(Debug, Clone)]
pub struct RenderCheckpoint {
    /// Unique checkpoint ID.
    pub id: CheckpointId,
    /// Job identifier this checkpoint belongs to.
    pub job_id: String,
    /// When this checkpoint was created.
    pub created_at: SystemTime,
    /// Current state of the checkpoint.
    pub state: CheckpointState,
    /// Per-frame progress information.
    pub frame_progress: Vec<FrameProgress>,
    /// Arbitrary key-value metadata.
    pub metadata: HashMap<String, String>,
    /// Total bytes of checkpoint data on disk.
    pub size_bytes: u64,
}

impl RenderCheckpoint {
    /// Creates a new checkpoint for a given job.
    pub fn new(job_id: impl Into<String>) -> Self {
        let job_id_str = job_id.into();
        let ts = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or(Duration::from_secs(0))
            .as_millis();
        Self {
            id: CheckpointId::new(format!("ckpt_{job_id_str}_{ts}")),
            job_id: job_id_str,
            created_at: SystemTime::now(),
            state: CheckpointState::Writing,
            frame_progress: Vec::new(),
            metadata: HashMap::new(),
            size_bytes: 0,
        }
    }

    /// Marks this checkpoint as complete.
    pub fn mark_complete(&mut self) {
        self.state = CheckpointState::Complete;
    }

    /// Marks this checkpoint as corrupted.
    pub fn mark_corrupted(&mut self) {
        self.state = CheckpointState::Corrupted;
    }

    /// Marks this checkpoint as superseded.
    pub fn mark_superseded(&mut self) {
        self.state = CheckpointState::Superseded;
    }

    /// Adds frame progress to this checkpoint.
    pub fn add_frame_progress(&mut self, progress: FrameProgress) {
        self.frame_progress.push(progress);
    }

    /// Returns the number of fully completed frames.
    #[must_use]
    pub fn completed_frame_count(&self) -> usize {
        self.frame_progress.iter().filter(|f| f.is_complete()).count()
    }

    /// Returns the overall completion percentage.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn overall_percent(&self) -> f32 {
        if self.frame_progress.is_empty() {
            return 0.0;
        }
        let total: f64 = self
            .frame_progress
            .iter()
            .map(|f| f.percent_complete as f64)
            .sum();
        (total / self.frame_progress.len() as f64) as f32
    }

    /// Sets a metadata key-value pair.
    pub fn set_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }

    /// Gets a metadata value by key.
    #[must_use]
    pub fn get_metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(String::as_str)
    }
}

/// Policy controlling when automatic checkpoints are taken.
#[derive(Debug, Clone)]
pub struct CheckpointPolicy {
    /// Interval between automatic checkpoints.
    pub interval: Duration,
    /// Maximum number of checkpoints to retain.
    pub max_retained: usize,
    /// Whether to checkpoint on graceful shutdown.
    pub checkpoint_on_shutdown: bool,
    /// Whether to checkpoint when a frame completes.
    pub checkpoint_on_frame_complete: bool,
}

impl CheckpointPolicy {
    /// Creates a default policy (every 5 minutes, keep 3).
    #[must_use]
    pub fn new() -> Self {
        Self {
            interval: Duration::from_secs(300),
            max_retained: 3,
            checkpoint_on_shutdown: true,
            checkpoint_on_frame_complete: false,
        }
    }

    /// Sets the checkpoint interval.
    #[must_use]
    pub fn with_interval(mut self, interval: Duration) -> Self {
        self.interval = interval;
        self
    }

    /// Sets the maximum number of retained checkpoints.
    #[must_use]
    pub fn with_max_retained(mut self, max: usize) -> Self {
        self.max_retained = max;
        self
    }

    /// Enables/disables checkpointing on frame completion.
    #[must_use]
    pub fn with_frame_checkpoint(mut self, enabled: bool) -> Self {
        self.checkpoint_on_frame_complete = enabled;
        self
    }
}

impl Default for CheckpointPolicy {
    fn default() -> Self {
        Self::new()
    }
}

/// Manages a sequence of checkpoints for a render job.
#[derive(Debug)]
pub struct CheckpointManager {
    /// Job ID this manager tracks.
    job_id: String,
    /// All checkpoints, ordered by creation time.
    checkpoints: Vec<RenderCheckpoint>,
    /// Active policy.
    policy: CheckpointPolicy,
}

impl CheckpointManager {
    /// Creates a new checkpoint manager for a job.
    pub fn new(job_id: impl Into<String>, policy: CheckpointPolicy) -> Self {
        Self {
            job_id: job_id.into(),
            checkpoints: Vec::new(),
            policy,
        }
    }

    /// Creates a new checkpoint and adds it to the list.
    pub fn create_checkpoint(&mut self) -> &RenderCheckpoint {
        // Mark old checkpoints as superseded
        for ckpt in &mut self.checkpoints {
            if ckpt.state == CheckpointState::Complete {
                ckpt.mark_superseded();
            }
        }
        let ckpt = RenderCheckpoint::new(&self.job_id);
        self.checkpoints.push(ckpt);
        // SAFETY: we just pushed an element, so `last()` is guaranteed to be `Some`.
        // Using a match to satisfy no-unwrap policy.
        match self.checkpoints.last() {
            Some(last) => last,
            None => unreachable!(),
        }
    }

    /// Returns the latest complete checkpoint, if any.
    #[must_use]
    pub fn latest_complete(&self) -> Option<&RenderCheckpoint> {
        self.checkpoints
            .iter()
            .rev()
            .find(|c| c.state == CheckpointState::Complete)
    }

    /// Returns the total number of checkpoints.
    #[must_use]
    pub fn count(&self) -> usize {
        self.checkpoints.len()
    }

    /// Prunes old checkpoints exceeding the retention policy.
    pub fn prune(&mut self) {
        let max = self.policy.max_retained;
        let complete_count = self
            .checkpoints
            .iter()
            .filter(|c| c.state == CheckpointState::Complete || c.state == CheckpointState::Superseded)
            .count();
        if complete_count > max {
            let to_remove = complete_count - max;
            let mut removed = 0;
            self.checkpoints.retain(|c| {
                if removed < to_remove
                    && (c.state == CheckpointState::Superseded
                        || c.state == CheckpointState::Corrupted)
                {
                    removed += 1;
                    false
                } else {
                    true
                }
            });
        }
    }

    /// Returns the active checkpoint policy.
    #[must_use]
    pub fn policy(&self) -> &CheckpointPolicy {
        &self.policy
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_id() {
        let id = CheckpointId::new("ckpt_001");
        assert_eq!(id.as_str(), "ckpt_001");
        assert_eq!(format!("{id}"), "ckpt_001");
    }

    #[test]
    fn test_checkpoint_state_display() {
        assert_eq!(format!("{}", CheckpointState::Complete), "complete");
        assert_eq!(format!("{}", CheckpointState::Writing), "writing");
        assert_eq!(format!("{}", CheckpointState::Corrupted), "corrupted");
        assert_eq!(format!("{}", CheckpointState::Superseded), "superseded");
    }

    #[test]
    fn test_frame_progress_new() {
        let fp = FrameProgress::new(1, 50, 100);
        assert_eq!(fp.frame, 1);
        assert!((fp.percent_complete - 50.0).abs() < 0.1);
        assert!(!fp.is_complete());
    }

    #[test]
    fn test_frame_progress_complete() {
        let fp = FrameProgress::new(1, 100, 100);
        assert!(fp.is_complete());
        assert!((fp.percent_complete - 100.0).abs() < 0.1);
    }

    #[test]
    fn test_frame_progress_zero_total() {
        let fp = FrameProgress::new(1, 0, 0);
        assert!(!fp.is_complete());
        assert!((fp.percent_complete).abs() < 0.1);
    }

    #[test]
    fn test_render_checkpoint_new() {
        let ckpt = RenderCheckpoint::new("job_42");
        assert_eq!(ckpt.job_id, "job_42");
        assert_eq!(ckpt.state, CheckpointState::Writing);
        assert!(ckpt.frame_progress.is_empty());
    }

    #[test]
    fn test_render_checkpoint_mark_complete() {
        let mut ckpt = RenderCheckpoint::new("job_1");
        ckpt.mark_complete();
        assert_eq!(ckpt.state, CheckpointState::Complete);
    }

    #[test]
    fn test_render_checkpoint_metadata() {
        let mut ckpt = RenderCheckpoint::new("job_1");
        ckpt.set_metadata("renderer", "cycles");
        assert_eq!(ckpt.get_metadata("renderer"), Some("cycles"));
        assert_eq!(ckpt.get_metadata("missing"), None);
    }

    #[test]
    fn test_completed_frame_count() {
        let mut ckpt = RenderCheckpoint::new("job_1");
        ckpt.add_frame_progress(FrameProgress::new(1, 100, 100));
        ckpt.add_frame_progress(FrameProgress::new(2, 50, 100));
        ckpt.add_frame_progress(FrameProgress::new(3, 100, 100));
        assert_eq!(ckpt.completed_frame_count(), 2);
    }

    #[test]
    fn test_overall_percent() {
        let mut ckpt = RenderCheckpoint::new("job_1");
        ckpt.add_frame_progress(FrameProgress::new(1, 100, 100));
        ckpt.add_frame_progress(FrameProgress::new(2, 0, 100));
        let pct = ckpt.overall_percent();
        assert!((pct - 50.0).abs() < 0.5);
    }

    #[test]
    fn test_overall_percent_empty() {
        let ckpt = RenderCheckpoint::new("job_1");
        assert!((ckpt.overall_percent()).abs() < 0.01);
    }

    #[test]
    fn test_checkpoint_policy_defaults() {
        let policy = CheckpointPolicy::new();
        assert_eq!(policy.interval, Duration::from_secs(300));
        assert_eq!(policy.max_retained, 3);
        assert!(policy.checkpoint_on_shutdown);
        assert!(!policy.checkpoint_on_frame_complete);
    }

    #[test]
    fn test_checkpoint_policy_builder() {
        let policy = CheckpointPolicy::new()
            .with_interval(Duration::from_secs(60))
            .with_max_retained(5)
            .with_frame_checkpoint(true);
        assert_eq!(policy.interval, Duration::from_secs(60));
        assert_eq!(policy.max_retained, 5);
        assert!(policy.checkpoint_on_frame_complete);
    }

    #[test]
    fn test_checkpoint_manager_create() {
        let policy = CheckpointPolicy::new();
        let mut mgr = CheckpointManager::new("job_1", policy);
        let _ = mgr.create_checkpoint();
        assert_eq!(mgr.count(), 1);
    }

    #[test]
    fn test_checkpoint_manager_latest_complete() {
        let policy = CheckpointPolicy::new();
        let mut mgr = CheckpointManager::new("job_1", policy);
        let _ = mgr.create_checkpoint();
        assert!(mgr.latest_complete().is_none());

        // Mark the first complete
        mgr.checkpoints[0].mark_complete();
        assert!(mgr.latest_complete().is_some());
    }
}
