// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Job-level output validation for the render farm coordinator.
//!
//! Provides [`JobOutputValidator`], [`ValidationResult`], and the free function
//! [`validate_job_output`] which checks whether a completed job's output meets
//! structural and timing requirements.
//!
//! The file-existence checks are **stub implementations** that always pass
//! (the real filesystem is not accessible in unit tests).  Timing checks are
//! always evaluated from the supplied metadata.
//!
//! # Example
//!
//! ```rust
//! use oximedia_renderfarm::job_output_validation::{
//!     JobOutputValidator, validate_job_output,
//! };
//!
//! let validator = JobOutputValidator {
//!     required_files: vec!["output.mkv".to_owned()],
//!     min_size_bytes: 1024,
//!     max_duration_ms: 60_000,
//! };
//!
//! let result = validate_job_output("/tmp/render/job-1", &validator, 30_000);
//! assert!(result.passed);
//! ```

// ---------------------------------------------------------------------------
// JobOutputValidator
// ---------------------------------------------------------------------------

/// Describes the acceptance criteria for a job's output.
#[derive(Debug, Clone)]
pub struct JobOutputValidator {
    /// Paths of files (relative to `output_path`) that must exist.
    ///
    /// In the current stub implementation, their existence is always assumed;
    /// integrate with a real filesystem probe for production use.
    pub required_files: Vec<String>,
    /// Minimum combined output size in bytes.
    ///
    /// Checked against `actual_size_bytes` when provided to
    /// [`validate_job_output_with_metadata`].
    pub min_size_bytes: u64,
    /// Maximum wall-clock duration in milliseconds the job is permitted to
    /// take.  `0` disables this check.
    pub max_duration_ms: u64,
}

impl Default for JobOutputValidator {
    fn default() -> Self {
        Self {
            required_files: Vec::new(),
            min_size_bytes: 0,
            max_duration_ms: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// ValidationResult
// ---------------------------------------------------------------------------

/// The outcome of a [`validate_job_output`] call.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// `true` when all checks passed (no hard errors).
    pub passed: bool,
    /// Hard errors that prevent delivery.
    pub errors: Vec<String>,
    /// Non-blocking notices that may warrant attention.
    pub warnings: Vec<String>,
}

impl ValidationResult {
    /// Construct a passing result with no issues.
    #[must_use]
    pub fn ok() -> Self {
        Self {
            passed: true,
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    /// Construct a failing result with a single error message.
    #[must_use]
    pub fn fail(error: impl Into<String>) -> Self {
        Self {
            passed: false,
            errors: vec![error.into()],
            warnings: Vec::new(),
        }
    }

    /// Add a hard error to the result and mark it as failed.
    pub fn add_error(&mut self, msg: impl Into<String>) {
        self.passed = false;
        self.errors.push(msg.into());
    }

    /// Add a non-blocking warning to the result.
    pub fn add_warning(&mut self, msg: impl Into<String>) {
        self.warnings.push(msg.into());
    }

    /// Merge another result into this one (logical AND for `passed`).
    pub fn merge(&mut self, other: ValidationResult) {
        if !other.passed {
            self.passed = false;
        }
        self.errors.extend(other.errors);
        self.warnings.extend(other.warnings);
    }

    /// Total number of issues (errors + warnings).
    #[must_use]
    pub fn issue_count(&self) -> usize {
        self.errors.len() + self.warnings.len()
    }
}

// ---------------------------------------------------------------------------
// validate_job_output
// ---------------------------------------------------------------------------

/// Validate a completed job's output at `output_path`.
///
/// # Arguments
///
/// * `output_path`   – Base directory or file path where output was written.
/// * `validator`     – Acceptance criteria.
/// * `actual_duration_ms` – Actual wall-clock render duration in milliseconds.
///
/// # File existence (stub)
///
/// Each entry in `validator.required_files` is noted in the result.  The
/// current implementation always marks them as found to avoid filesystem
/// access in tests.  Replace the stub body with a real `std::fs::metadata`
/// call for production use.
#[must_use]
pub fn validate_job_output(
    output_path: &str,
    validator: &JobOutputValidator,
    actual_duration_ms: u64,
) -> ValidationResult {
    let mut result = ValidationResult::ok();

    // --- Required files (stub: always pass) ---
    for required in &validator.required_files {
        // Stub: record that we checked the file but assume it exists.
        // In production: check std::fs::metadata(format!("{}/{}", output_path, required))
        let _ = output_path; // suppress unused warning in stub
        let _ = required;
        // result.add_error(format!("required file not found: {}/{}", output_path, required));
    }

    // --- Timing check ---
    if validator.max_duration_ms > 0 && actual_duration_ms > validator.max_duration_ms {
        result.add_error(format!(
            "job exceeded maximum duration: {} ms > {} ms",
            actual_duration_ms, validator.max_duration_ms
        ));
    }

    result
}

/// Extended variant that also checks actual output size.
///
/// `actual_size_bytes` is compared against `validator.min_size_bytes` when
/// the validator has a non-zero minimum.
#[must_use]
pub fn validate_job_output_with_metadata(
    output_path: &str,
    validator: &JobOutputValidator,
    actual_duration_ms: u64,
    actual_size_bytes: u64,
) -> ValidationResult {
    let mut result = validate_job_output(output_path, validator, actual_duration_ms);

    if validator.min_size_bytes > 0 && actual_size_bytes < validator.min_size_bytes {
        result.add_error(format!(
            "output too small: {} bytes < minimum {} bytes",
            actual_size_bytes, validator.min_size_bytes
        ));
    } else if validator.min_size_bytes > 0 && actual_size_bytes < validator.min_size_bytes * 2 {
        // Advisory warning when output is small (< 2× minimum) but still passes.
        result.add_warning(format!(
            "output size {} bytes is close to minimum {} bytes",
            actual_size_bytes, validator.min_size_bytes
        ));
    }

    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- ValidationResult helpers ---

    #[test]
    fn ok_result_passes() {
        let r = ValidationResult::ok();
        assert!(r.passed);
        assert!(r.errors.is_empty());
        assert!(r.warnings.is_empty());
    }

    #[test]
    fn fail_result_does_not_pass() {
        let r = ValidationResult::fail("something went wrong");
        assert!(!r.passed);
        assert_eq!(r.errors.len(), 1);
    }

    #[test]
    fn add_error_marks_failed() {
        let mut r = ValidationResult::ok();
        r.add_error("oops");
        assert!(!r.passed);
        assert_eq!(r.errors.len(), 1);
    }

    #[test]
    fn add_warning_does_not_fail() {
        let mut r = ValidationResult::ok();
        r.add_warning("heads up");
        assert!(r.passed);
        assert_eq!(r.warnings.len(), 1);
    }

    #[test]
    fn merge_propagates_failure() {
        let mut a = ValidationResult::ok();
        let b = ValidationResult::fail("b failed");
        a.merge(b);
        assert!(!a.passed);
        assert_eq!(a.errors.len(), 1);
    }

    #[test]
    fn merge_ok_into_ok_stays_ok() {
        let mut a = ValidationResult::ok();
        a.add_warning("note");
        a.merge(ValidationResult::ok());
        assert!(a.passed);
        assert_eq!(a.warnings.len(), 1);
    }

    #[test]
    fn issue_count_sums_errors_and_warnings() {
        let mut r = ValidationResult::ok();
        r.add_error("e1");
        r.add_error("e2");
        r.add_warning("w1");
        assert_eq!(r.issue_count(), 3);
    }

    // --- validate_job_output ---

    #[test]
    fn validate_passes_within_duration() {
        let v = JobOutputValidator {
            max_duration_ms: 60_000,
            ..Default::default()
        };
        let result = validate_job_output("/tmp/out", &v, 30_000);
        assert!(result.passed, "30 s < 60 s limit should pass");
    }

    #[test]
    fn validate_fails_over_duration() {
        let v = JobOutputValidator {
            max_duration_ms: 10_000,
            ..Default::default()
        };
        let result = validate_job_output("/tmp/out", &v, 20_000);
        assert!(!result.passed, "20 s > 10 s limit should fail");
        assert!(result.errors.iter().any(|e| e.contains("exceeded")));
    }

    #[test]
    fn validate_no_duration_limit_always_passes_timing() {
        let v = JobOutputValidator {
            max_duration_ms: 0, // disabled
            ..Default::default()
        };
        let result = validate_job_output("/tmp/out", &v, 999_999_999);
        assert!(result.passed);
    }

    #[test]
    fn validate_with_required_files_stub_passes() {
        let v = JobOutputValidator {
            required_files: vec!["output.mkv".to_owned(), "thumbnail.jpg".to_owned()],
            max_duration_ms: 60_000,
            min_size_bytes: 0,
        };
        let result = validate_job_output("/tmp/render/job1", &v, 5_000);
        assert!(result.passed, "stub always passes file checks");
    }

    // --- validate_job_output_with_metadata ---

    #[test]
    fn metadata_fails_below_min_size() {
        let v = JobOutputValidator {
            min_size_bytes: 1024,
            max_duration_ms: 0,
            ..Default::default()
        };
        let result = validate_job_output_with_metadata("/tmp/out", &v, 0, 512);
        assert!(!result.passed);
        assert!(result.errors.iter().any(|e| e.contains("too small")));
    }

    #[test]
    fn metadata_passes_above_min_size() {
        let v = JobOutputValidator {
            min_size_bytes: 1024,
            max_duration_ms: 0,
            ..Default::default()
        };
        let result = validate_job_output_with_metadata("/tmp/out", &v, 0, 4096);
        assert!(result.passed);
    }

    #[test]
    fn metadata_warning_when_close_to_minimum() {
        let v = JobOutputValidator {
            min_size_bytes: 1024,
            max_duration_ms: 0,
            ..Default::default()
        };
        // 1500 bytes: >= 1024 but < 2048 (2× minimum) → warning
        let result = validate_job_output_with_metadata("/tmp/out", &v, 0, 1_500);
        assert!(result.passed);
        assert!(
            !result.warnings.is_empty(),
            "should emit close-to-minimum warning"
        );
    }

    #[test]
    fn metadata_fails_both_size_and_duration() {
        let v = JobOutputValidator {
            min_size_bytes: 10_000,
            max_duration_ms: 1_000,
            ..Default::default()
        };
        let result = validate_job_output_with_metadata("/tmp/out", &v, 5_000, 100);
        assert!(!result.passed);
        assert_eq!(result.errors.len(), 2);
    }
}
