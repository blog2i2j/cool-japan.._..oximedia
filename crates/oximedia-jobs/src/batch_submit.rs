// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Batch job submission with atomicity, validation, and configurable failure modes.
//!
//! Submits multiple [`Job`]s in a single operation.  Before any work is
//! accepted the module can optionally validate the entire set first
//! (atomicity), stop at the first failure (`fail_fast`), or collect partial
//! results from all submissions.

use crate::job::{Job, JobPayload, Priority};
use std::collections::HashSet;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// JobSpec
// ---------------------------------------------------------------------------

/// A specification for a single job to be submitted in a batch.
///
/// Uses `Job` directly but wrapped here so callers work through the public
/// batch API rather than constructing raw jobs inline.
#[derive(Debug, Clone)]
pub struct JobSpec {
    /// Job name (must be non-empty).
    pub name: String,
    /// Job priority.
    pub priority: Priority,
    /// What the job should do.
    pub payload: JobPayload,
    /// Optional explicit job ID.  When `None`, a random UUID is generated.
    pub id: Option<Uuid>,
    /// Optional tags for this job.
    pub tags: Vec<String>,
}

impl JobSpec {
    /// Create a minimal spec.
    #[must_use]
    pub fn new(name: impl Into<String>, priority: Priority, payload: JobPayload) -> Self {
        Self {
            name: name.into(),
            priority,
            payload,
            id: None,
            tags: Vec::new(),
        }
    }

    /// Override the job ID (useful for deterministic tests).
    #[must_use]
    pub fn with_id(mut self, id: Uuid) -> Self {
        self.id = Some(id);
        self
    }

    /// Add a tag.
    #[must_use]
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }
}

// ---------------------------------------------------------------------------
// BatchRequest
// ---------------------------------------------------------------------------

/// A collection of [`JobSpec`]s to be submitted together.
#[derive(Debug, Clone, Default)]
pub struct BatchRequest {
    /// The jobs to submit.
    pub jobs: Vec<JobSpec>,
}

impl BatchRequest {
    /// Create an empty batch.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a job spec.
    #[must_use]
    pub fn with_job(mut self, spec: JobSpec) -> Self {
        self.jobs.push(spec);
        self
    }

    /// Number of jobs in the request.
    #[must_use]
    pub fn len(&self) -> usize {
        self.jobs.len()
    }

    /// Whether the request is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.jobs.is_empty()
    }
}

// ---------------------------------------------------------------------------
// BatchSubmitConfig
// ---------------------------------------------------------------------------

/// Configuration controlling how a batch is submitted.
#[derive(Debug, Clone)]
pub struct BatchSubmitConfig {
    /// Maximum number of jobs allowed in a single batch.
    pub max_batch_size: usize,
    /// When `true`, validate every job before submitting any of them.
    /// Any validation failure prevents the entire batch from being submitted.
    pub validate_all_before_submit: bool,
    /// When `true`, stop at the first failed submission; when `false`, collect
    /// all successes and failures and return a partial result.
    pub fail_fast: bool,
}

impl BatchSubmitConfig {
    /// Create a config with explicit settings.
    #[must_use]
    pub fn new(max_batch_size: usize, validate_all_before_submit: bool, fail_fast: bool) -> Self {
        Self {
            max_batch_size,
            validate_all_before_submit,
            fail_fast,
        }
    }
}

impl Default for BatchSubmitConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 500,
            validate_all_before_submit: true,
            fail_fast: false,
        }
    }
}

// ---------------------------------------------------------------------------
// BatchResult
// ---------------------------------------------------------------------------

/// The outcome of a batch submission.
#[derive(Debug, Clone, Default)]
pub struct BatchSubmitResult {
    /// IDs of successfully submitted jobs.
    pub submitted: Vec<String>,
    /// Pairs of `(0-based original index, error message)` for failed jobs.
    pub failed: Vec<(usize, String)>,
    /// Total number of successfully submitted jobs.
    pub total_submitted: usize,
}

impl BatchSubmitResult {
    /// Whether every job in the batch was submitted successfully.
    #[must_use]
    pub fn is_fully_successful(&self) -> bool {
        self.failed.is_empty()
    }
}

// ---------------------------------------------------------------------------
// ValidationError
// ---------------------------------------------------------------------------

/// Errors that can occur at the batch level before individual jobs are examined.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationError {
    /// The batch contains no jobs.
    EmptyBatch,
    /// The batch exceeds the configured maximum size.
    BatchTooLarge {
        /// How many jobs are in the batch.
        actual: usize,
        /// The configured limit.
        limit: usize,
    },
    /// Two or more jobs in the batch share the same explicit UUID.
    DuplicateJobId(String),
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyBatch => write!(f, "batch must contain at least one job"),
            Self::BatchTooLarge { actual, limit } => {
                write!(f, "batch size {actual} exceeds maximum {limit}")
            }
            Self::DuplicateJobId(id) => write!(f, "duplicate job id: {id}"),
        }
    }
}

// ---------------------------------------------------------------------------
// SubmitError
// ---------------------------------------------------------------------------

/// Error produced when a single job specification is invalid.
#[derive(Debug, Clone)]
pub struct SubmitError(pub String);

impl std::fmt::Display for SubmitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// BatchSubmitter
// ---------------------------------------------------------------------------

/// Submits batches of jobs with optional whole-batch validation and configurable
/// failure handling.
///
/// This implementation is intentionally synchronous so it can be used without
/// a Tokio runtime.  For async usage wrap calls in `spawn_blocking`.
#[derive(Debug, Default)]
pub struct BatchSubmitter;

impl BatchSubmitter {
    /// Create a new submitter.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Validate and submit a batch according to `config`.
    ///
    /// # Errors
    ///
    /// Returns [`ValidationError`] for batch-level problems (empty, too large,
    /// duplicate IDs).  Individual per-job errors are captured inside
    /// [`BatchSubmitResult::failed`].
    pub fn submit(
        &self,
        batch: BatchRequest,
        config: &BatchSubmitConfig,
    ) -> Result<BatchSubmitResult, ValidationError> {
        // --- Batch-level validation ----------------------------------------
        if batch.is_empty() {
            return Err(ValidationError::EmptyBatch);
        }
        if batch.len() > config.max_batch_size {
            return Err(ValidationError::BatchTooLarge {
                actual: batch.len(),
                limit: config.max_batch_size,
            });
        }
        // Check for duplicate explicit IDs.
        {
            let mut seen: HashSet<Uuid> = HashSet::new();
            for spec in &batch.jobs {
                if let Some(id) = spec.id {
                    if !seen.insert(id) {
                        return Err(ValidationError::DuplicateJobId(id.to_string()));
                    }
                }
            }
        }

        // --- Per-job phase -------------------------------------------------
        if config.validate_all_before_submit {
            // Validate ALL jobs first; only proceed if all are valid.
            let mut errors: Vec<(usize, String)> = Vec::new();
            for (idx, spec) in batch.jobs.iter().enumerate() {
                if let Err(e) = validate_spec(spec) {
                    if config.fail_fast {
                        let mut result = BatchSubmitResult::default();
                        result.failed.push((idx, e.0));
                        return Ok(result);
                    }
                    errors.push((idx, e.0));
                }
            }
            if !errors.is_empty() {
                let mut result = BatchSubmitResult::default();
                result.failed = errors;
                return Ok(result);
            }
        }

        // --- Submit ---------------------------------------------------------
        let mut result = BatchSubmitResult::default();
        for (idx, spec) in batch.jobs.into_iter().enumerate() {
            if !config.validate_all_before_submit {
                // Validate on-the-fly before submitting.
                if let Err(e) = validate_spec(&spec) {
                    result.failed.push((idx, e.0));
                    if config.fail_fast {
                        break;
                    }
                    continue;
                }
            }
            // Convert spec → Job and "submit" (record the ID).
            let job = spec_to_job(spec);
            result.submitted.push(job.id.to_string());
        }

        result.total_submitted = result.submitted.len();
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Lightweight validation of a single [`JobSpec`].
fn validate_spec(spec: &JobSpec) -> Result<(), SubmitError> {
    if spec.name.trim().is_empty() {
        return Err(SubmitError("job name must not be empty".to_string()));
    }
    // Payload-level checks.
    match &spec.payload {
        JobPayload::Transcode(p) => {
            if p.input.trim().is_empty() {
                return Err(SubmitError("transcode: input path must not be empty".to_string()));
            }
            if p.output.trim().is_empty() {
                return Err(SubmitError("transcode: output path must not be empty".to_string()));
            }
        }
        JobPayload::Thumbnail(p) => {
            if p.input.trim().is_empty() {
                return Err(SubmitError("thumbnail: input path must not be empty".to_string()));
            }
            if p.count == 0 {
                return Err(SubmitError("thumbnail: count must be > 0".to_string()));
            }
        }
        JobPayload::SpriteSheet(p) => {
            if p.input.trim().is_empty() {
                return Err(SubmitError(
                    "spritesheet: input path must not be empty".to_string(),
                ));
            }
        }
        JobPayload::Analysis(p) => {
            if p.input.trim().is_empty() {
                return Err(SubmitError("analysis: input path must not be empty".to_string()));
            }
        }
        JobPayload::Batch(p) => {
            if p.inputs.is_empty() {
                return Err(SubmitError("batch: inputs must not be empty".to_string()));
            }
        }
    }
    Ok(())
}

/// Convert a [`JobSpec`] into a full [`Job`].
fn spec_to_job(spec: JobSpec) -> Job {
    let mut job = Job::new(spec.name, spec.priority, spec.payload);
    if let Some(id) = spec.id {
        job.id = id;
    }
    for tag in spec.tags {
        job = job.with_tag(tag);
    }
    job
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::job::{AnalysisParams, AnalysisType, ThumbnailParams, TranscodeParams};

    fn transcode_spec(name: &str) -> JobSpec {
        JobSpec::new(
            name,
            Priority::Normal,
            JobPayload::Transcode(TranscodeParams {
                input: "in.mp4".to_string(),
                output: "out.mp4".to_string(),
                video_codec: "h264".to_string(),
                audio_codec: "aac".to_string(),
                video_bitrate: 5_000_000,
                audio_bitrate: 192_000,
                resolution: None,
                framerate: None,
                preset: "fast".to_string(),
                hw_accel: None,
            }),
        )
    }

    fn thumbnail_spec(name: &str) -> JobSpec {
        JobSpec::new(
            name,
            Priority::Low,
            JobPayload::Thumbnail(ThumbnailParams {
                input: "in.mp4".to_string(),
                output_dir: std::env::temp_dir()
                    .join("oximedia-jobs-batch-thumbs")
                    .to_string_lossy()
                    .into_owned(),
                count: 5,
                width: 320,
                height: 180,
                quality: 80,
            }),
        )
    }

    #[test]
    fn test_empty_batch_returns_error() {
        let submitter = BatchSubmitter::new();
        let batch = BatchRequest::new();
        let result = submitter.submit(batch, &BatchSubmitConfig::default());
        assert!(matches!(result, Err(ValidationError::EmptyBatch)));
    }

    #[test]
    fn test_batch_too_large() {
        let submitter = BatchSubmitter::new();
        let config = BatchSubmitConfig::new(2, false, false);
        let batch = BatchRequest::new()
            .with_job(transcode_spec("j1"))
            .with_job(transcode_spec("j2"))
            .with_job(transcode_spec("j3"));
        let result = submitter.submit(batch, &config);
        assert!(matches!(
            result,
            Err(ValidationError::BatchTooLarge { actual: 3, limit: 2 })
        ));
    }

    #[test]
    fn test_duplicate_job_id_rejected() {
        let submitter = BatchSubmitter::new();
        let shared_id = Uuid::new_v4();
        let batch = BatchRequest::new()
            .with_job(transcode_spec("j1").with_id(shared_id))
            .with_job(transcode_spec("j2").with_id(shared_id));
        let result = submitter.submit(batch, &BatchSubmitConfig::default());
        assert!(matches!(result, Err(ValidationError::DuplicateJobId(_))));
    }

    #[test]
    fn test_valid_batch_all_submitted() {
        let submitter = BatchSubmitter::new();
        let config = BatchSubmitConfig::new(10, false, false);
        let batch = BatchRequest::new()
            .with_job(transcode_spec("j1"))
            .with_job(thumbnail_spec("j2"));
        let result = submitter.submit(batch, &config).expect("ok");
        assert_eq!(result.total_submitted, 2);
        assert!(result.failed.is_empty());
    }

    #[test]
    fn test_fail_fast_stops_at_first_failure() {
        let submitter = BatchSubmitter::new();
        let config = BatchSubmitConfig::new(10, false, true);
        // Empty name is invalid.
        let bad_spec = JobSpec::new(
            "",
            Priority::Normal,
            JobPayload::Transcode(TranscodeParams {
                input: "in.mp4".to_string(),
                output: "out.mp4".to_string(),
                video_codec: "h264".to_string(),
                audio_codec: "aac".to_string(),
                video_bitrate: 0,
                audio_bitrate: 0,
                resolution: None,
                framerate: None,
                preset: "fast".to_string(),
                hw_accel: None,
            }),
        );
        let batch = BatchRequest::new()
            .with_job(bad_spec)
            .with_job(transcode_spec("j2"))
            .with_job(transcode_spec("j3"));
        let result = submitter.submit(batch, &config).expect("batch level ok");
        // Only the first failure is recorded; j2/j3 are never attempted.
        assert_eq!(result.failed.len(), 1);
        assert_eq!(result.total_submitted, 0);
    }

    #[test]
    fn test_partial_results_without_fail_fast() {
        let submitter = BatchSubmitter::new();
        let config = BatchSubmitConfig::new(10, false, false);
        let bad_spec = JobSpec::new(
            "",
            Priority::Normal,
            JobPayload::Analysis(AnalysisParams {
                input: "".to_string(),
                analysis_type: AnalysisType::Quality,
                output: None,
            }),
        );
        let batch = BatchRequest::new()
            .with_job(transcode_spec("j1"))
            .with_job(bad_spec)
            .with_job(transcode_spec("j3"));
        let result = submitter.submit(batch, &config).expect("ok");
        assert_eq!(result.total_submitted, 2);
        assert_eq!(result.failed.len(), 1);
        assert_eq!(result.failed[0].0, 1); // index 1 failed
    }

    #[test]
    fn test_validate_all_before_submit_blocks_on_any_error() {
        let submitter = BatchSubmitter::new();
        let config = BatchSubmitConfig::new(10, true, false);
        let bad_spec = JobSpec::new(
            "",
            Priority::Low,
            JobPayload::Thumbnail(ThumbnailParams {
                input: "".to_string(),
                output_dir: "".to_string(),
                count: 0,
                width: 320,
                height: 180,
                quality: 80,
            }),
        );
        let batch = BatchRequest::new()
            .with_job(transcode_spec("ok-job"))
            .with_job(bad_spec);
        let result = submitter.submit(batch, &config).expect("ok");
        // Even the valid job should NOT be submitted because validate-all blocked.
        assert_eq!(result.total_submitted, 0);
        assert!(!result.failed.is_empty());
    }

    #[test]
    fn test_validate_all_then_submit_all_when_valid() {
        let submitter = BatchSubmitter::new();
        let config = BatchSubmitConfig::new(10, true, false);
        let batch = BatchRequest::new()
            .with_job(transcode_spec("j1"))
            .with_job(thumbnail_spec("j2"))
            .with_job(transcode_spec("j3"));
        let result = submitter.submit(batch, &config).expect("ok");
        assert_eq!(result.total_submitted, 3);
        assert!(result.is_fully_successful());
    }

    #[test]
    fn test_submitted_ids_are_unique() {
        let submitter = BatchSubmitter::new();
        let config = BatchSubmitConfig::new(20, false, false);
        let mut batch = BatchRequest::new();
        for i in 0..10 {
            batch = batch.with_job(transcode_spec(&format!("job-{i}")));
        }
        let result = submitter.submit(batch, &config).expect("ok");
        let unique: HashSet<_> = result.submitted.iter().collect();
        assert_eq!(unique.len(), 10, "all submitted IDs must be unique");
    }

    #[test]
    fn test_explicit_id_preserved() {
        let submitter = BatchSubmitter::new();
        let id = Uuid::new_v4();
        let spec = transcode_spec("explicit").with_id(id);
        let batch = BatchRequest::new().with_job(spec);
        let result = submitter.submit(batch, &BatchSubmitConfig::default()).expect("ok");
        assert_eq!(result.submitted[0], id.to_string());
    }
}
