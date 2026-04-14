#![allow(dead_code)]
//! Job replay — re-run failed jobs with modified parameters.
//!
//! This module provides a `JobReplay` facility that snapshots a failed job,
//! allows parameter overrides to be specified, and then produces a new `Job`
//! ready for re-submission with a fresh ID and clean state.

use crate::job::{Job, JobPayload, JobStatus, Priority};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Error type for replay operations.
#[derive(Debug, thiserror::Error)]
pub enum ReplayError {
    /// The original job cannot be replayed (e.g. it has not failed yet).
    #[error("Job is not in a replayable state: {0:?}")]
    NotReplayable(JobStatus),
    /// A parameter override could not be applied to this payload type.
    #[error("Cannot apply parameter override '{key}' to payload type '{payload_type}'")]
    IncompatibleOverride {
        key: String,
        payload_type: String,
    },
    /// Serialisation error.
    #[error("Serialise error: {0}")]
    Serialise(String),
}

/// A parameter override to apply when replaying a job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParamOverride {
    /// Parameter name (payload-type specific).
    pub key: String,
    /// New value as a string.
    pub value: String,
}

impl ParamOverride {
    /// Create a new parameter override.
    #[must_use]
    pub fn new(key: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            value: value.into(),
        }
    }
}

/// A snapshot of a failed job, ready for replay.
#[derive(Debug, Clone)]
pub struct ReplaySpec {
    /// ID of the original (failed) job.
    pub original_id: Uuid,
    /// Name for the replayed job.
    pub replay_name: String,
    /// Priority override for the new job (defaults to original priority).
    pub priority_override: Option<Priority>,
    /// Parameter overrides to apply.
    pub overrides: Vec<ParamOverride>,
    /// Reason for the replay (audit trail).
    pub reason: String,
}

impl ReplaySpec {
    /// Create a replay spec for the given original job ID.
    #[must_use]
    pub fn new(original_id: Uuid, replay_name: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            original_id,
            replay_name: replay_name.into(),
            priority_override: None,
            overrides: Vec::new(),
            reason: reason.into(),
        }
    }

    /// Override the priority for the replayed job.
    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.priority_override = Some(priority);
        self
    }

    /// Add a parameter override.
    pub fn with_override(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.overrides.push(ParamOverride::new(key, value));
        self
    }
}

/// A record of a replay attempt.
#[derive(Debug, Clone)]
pub struct ReplayRecord {
    /// ID of the original job.
    pub original_id: Uuid,
    /// ID of the replayed job.
    pub replayed_id: Uuid,
    /// Overrides that were applied.
    pub overrides_applied: Vec<ParamOverride>,
    /// Reason for the replay.
    pub reason: String,
    /// When the replay was created.
    pub replayed_at: chrono::DateTime<chrono::Utc>,
}

/// Replays failed jobs with optional parameter modifications.
#[derive(Debug, Default)]
pub struct JobReplay {
    /// History of all replay operations.
    history: Vec<ReplayRecord>,
}

impl JobReplay {
    /// Create a new job replay manager.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a new `Job` from `original`, applying the overrides in `spec`.
    ///
    /// The returned job has a fresh UUID and `Pending` status.
    pub fn replay(&mut self, original: &Job, spec: &ReplaySpec) -> Result<Job, ReplayError> {
        // Only replay terminal-failure jobs
        if !matches!(original.status, JobStatus::Failed | JobStatus::Cancelled) {
            return Err(ReplayError::NotReplayable(original.status));
        }
        // Apply payload overrides
        let new_payload = Self::apply_overrides(&original.payload, &spec.overrides)?;
        let priority = spec.priority_override.unwrap_or(original.priority);
        let mut new_job = Job::new(spec.replay_name.clone(), priority, new_payload);
        new_job.tags = original.tags.clone();
        // Record the replay
        self.history.push(ReplayRecord {
            original_id: original.id,
            replayed_id: new_job.id,
            overrides_applied: spec.overrides.clone(),
            reason: spec.reason.clone(),
            replayed_at: chrono::Utc::now(),
        });
        Ok(new_job)
    }

    /// Apply string-valued overrides to a `JobPayload`.
    fn apply_overrides(
        payload: &JobPayload,
        overrides: &[ParamOverride],
    ) -> Result<JobPayload, ReplayError> {
        if overrides.is_empty() {
            return Ok(payload.clone());
        }
        // Build a map for quick lookup
        let map: HashMap<&str, &str> = overrides
            .iter()
            .map(|o| (o.key.as_str(), o.value.as_str()))
            .collect();
        let payload_type = payload.type_name();
        match payload {
            JobPayload::Transcode(p) => {
                let mut p = p.clone();
                for (key, val) in &map {
                    match *key {
                        "input" => p.input = val.to_string(),
                        "output" => p.output = val.to_string(),
                        "video_codec" => p.video_codec = val.to_string(),
                        "audio_codec" => p.audio_codec = val.to_string(),
                        "preset" => p.preset = val.to_string(),
                        "video_bitrate" => {
                            p.video_bitrate = val.parse().unwrap_or(p.video_bitrate);
                        }
                        "audio_bitrate" => {
                            p.audio_bitrate = val.parse().unwrap_or(p.audio_bitrate);
                        }
                        other => {
                            return Err(ReplayError::IncompatibleOverride {
                                key: other.to_string(),
                                payload_type: payload_type.to_string(),
                            })
                        }
                    }
                }
                Ok(JobPayload::Transcode(p))
            }
            JobPayload::Thumbnail(p) => {
                let mut p = p.clone();
                for (key, val) in &map {
                    match *key {
                        "input" => p.input = val.to_string(),
                        "output_dir" => p.output_dir = val.to_string(),
                        "quality" => {
                            p.quality = val.parse().unwrap_or(p.quality);
                        }
                        other => {
                            return Err(ReplayError::IncompatibleOverride {
                                key: other.to_string(),
                                payload_type: payload_type.to_string(),
                            })
                        }
                    }
                }
                Ok(JobPayload::Thumbnail(p))
            }
            JobPayload::Analysis(p) => {
                let mut p = p.clone();
                for (key, val) in &map {
                    match *key {
                        "input" => p.input = val.to_string(),
                        other => {
                            return Err(ReplayError::IncompatibleOverride {
                                key: other.to_string(),
                                payload_type: payload_type.to_string(),
                            })
                        }
                    }
                }
                Ok(JobPayload::Analysis(p))
            }
            other => {
                // For payload types without override support, return as-is when no overrides
                if overrides.is_empty() {
                    Ok(other.clone())
                } else {
                    Err(ReplayError::IncompatibleOverride {
                        key: overrides[0].key.clone(),
                        payload_type: payload_type.to_string(),
                    })
                }
            }
        }
    }

    /// Return all replay history records.
    #[must_use]
    pub fn history(&self) -> &[ReplayRecord] {
        &self.history
    }

    /// Return all replay records for a given original job ID.
    #[must_use]
    pub fn history_for(&self, original_id: Uuid) -> Vec<&ReplayRecord> {
        self.history
            .iter()
            .filter(|r| r.original_id == original_id)
            .collect()
    }
}

/// Extension trait so `JobPayload` can report its type name.
trait PayloadTypeName {
    fn type_name(&self) -> &'static str;
}

impl PayloadTypeName for JobPayload {
    fn type_name(&self) -> &'static str {
        match self {
            JobPayload::Transcode(_) => "Transcode",
            JobPayload::Thumbnail(_) => "Thumbnail",
            JobPayload::SpriteSheet(_) => "SpriteSheet",
            JobPayload::Analysis(_) => "Analysis",
            JobPayload::Batch(_) => "Batch",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::job::{AnalysisParams, AnalysisType, TranscodeParams};

    fn failed_transcode_job() -> Job {
        let params = TranscodeParams {
            input: "input.mp4".to_string(),
            output: "output.mp4".to_string(),
            video_codec: "h264".to_string(),
            audio_codec: "aac".to_string(),
            video_bitrate: 5_000_000,
            audio_bitrate: 192_000,
            resolution: None,
            framerate: None,
            preset: "medium".to_string(),
            hw_accel: None,
        };
        let mut job = Job::new("transcode".to_string(), Priority::Normal, JobPayload::Transcode(params));
        job.status = JobStatus::Failed;
        job.error = Some("codec not found".to_string());
        job
    }

    fn failed_analysis_job() -> Job {
        let params = AnalysisParams {
            input: "video.mp4".to_string(),
            analysis_type: AnalysisType::Scenes,
            output: None,
        };
        let mut job = Job::new("analysis".to_string(), Priority::Normal, JobPayload::Analysis(params));
        job.status = JobStatus::Failed;
        job
    }

    #[test]
    fn test_replay_failed_job() {
        let original = failed_transcode_job();
        let mut replayer = JobReplay::new();
        let spec = ReplaySpec::new(original.id, "transcode-retry", "codec was updated");
        let replayed = replayer.replay(&original, &spec).expect("replay should succeed");
        assert_ne!(replayed.id, original.id);
        assert_eq!(replayed.status, JobStatus::Pending);
        assert_eq!(replayed.name, "transcode-retry");
    }

    #[test]
    fn test_replay_with_input_override() {
        let original = failed_transcode_job();
        let mut replayer = JobReplay::new();
        let spec = ReplaySpec::new(original.id, "retry", "fix input")
            .with_override("input", "fixed_input.mp4");
        let replayed = replayer.replay(&original, &spec).expect("replay should succeed");
        if let JobPayload::Transcode(p) = replayed.payload {
            assert_eq!(p.input, "fixed_input.mp4");
        } else {
            panic!("expected transcode payload");
        }
    }

    #[test]
    fn test_replay_with_priority_override() {
        let original = failed_transcode_job();
        let mut replayer = JobReplay::new();
        let spec = ReplaySpec::new(original.id, "retry", "urgent retry")
            .with_priority(Priority::High);
        let replayed = replayer.replay(&original, &spec).expect("replay should succeed");
        assert_eq!(replayed.priority, Priority::High);
    }

    #[test]
    fn test_replay_non_failed_job_returns_error() {
        let mut job = failed_transcode_job();
        job.status = JobStatus::Running;
        let mut replayer = JobReplay::new();
        let spec = ReplaySpec::new(job.id, "retry", "should fail");
        let result = replayer.replay(&job, &spec);
        assert!(matches!(result, Err(ReplayError::NotReplayable(_))));
    }

    #[test]
    fn test_replay_history_recorded() {
        let original = failed_transcode_job();
        let mut replayer = JobReplay::new();
        let spec = ReplaySpec::new(original.id, "r1", "first retry");
        replayer.replay(&original, &spec).expect("replay");
        let history = replayer.history_for(original.id);
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].original_id, original.id);
    }

    #[test]
    fn test_replay_multiple_times_recorded_separately() {
        let original = failed_transcode_job();
        let mut replayer = JobReplay::new();
        replayer.replay(&original, &ReplaySpec::new(original.id, "r1", "try 1")).expect("1");
        replayer.replay(&original, &ReplaySpec::new(original.id, "r2", "try 2")).expect("2");
        assert_eq!(replayer.history_for(original.id).len(), 2);
        assert_eq!(replayer.history().len(), 2);
    }

    #[test]
    fn test_replay_analysis_job_with_input_override() {
        let original = failed_analysis_job();
        let mut replayer = JobReplay::new();
        let spec = ReplaySpec::new(original.id, "analysis-retry", "new input")
            .with_override("input", "corrected.mp4");
        let replayed = replayer.replay(&original, &spec).expect("replay");
        if let JobPayload::Analysis(p) = replayed.payload {
            assert_eq!(p.input, "corrected.mp4");
        } else {
            panic!("expected analysis payload");
        }
    }

    #[test]
    fn test_replay_incompatible_override_returns_error() {
        let original = failed_analysis_job();
        let mut replayer = JobReplay::new();
        let spec = ReplaySpec::new(original.id, "retry", "bad override")
            .with_override("video_codec", "av1"); // not applicable to Analysis
        let result = replayer.replay(&original, &spec);
        assert!(matches!(result, Err(ReplayError::IncompatibleOverride { .. })));
    }

    #[test]
    fn test_replay_cancelled_job_is_allowed() {
        let mut job = failed_transcode_job();
        job.status = JobStatus::Cancelled;
        let mut replayer = JobReplay::new();
        let spec = ReplaySpec::new(job.id, "retry-cancelled", "user request");
        assert!(replayer.replay(&job, &spec).is_ok());
    }
}
