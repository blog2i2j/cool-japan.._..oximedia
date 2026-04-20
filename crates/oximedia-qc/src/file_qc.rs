//! File-level quality control checks.
//!
//! This module validates media files at the file level: size, duration,
//! and other structural properties, returning structured pass/fail results.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// The type of file QC check being performed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FileQcCheck {
    /// File size is within allowed bounds.
    FileSize,
    /// Checksum matches expected value.
    Checksum,
    /// Duration is within allowed bounds.
    Duration,
    /// Resolution matches the expected value.
    Resolution,
    /// Codec is in the list of allowed codecs.
    Codec,
    /// Container format is allowed.
    Container,
    /// Bitrate falls within the specified range.
    BitrateInRange,
}

impl FileQcCheck {
    /// Returns `true` for checks that relate to the file's binary structure
    /// (size, checksum, container).
    #[must_use]
    pub fn is_structural(&self) -> bool {
        matches!(self, Self::FileSize | Self::Checksum | Self::Container)
    }
}

/// The result of a single file QC check.
#[derive(Debug, Clone)]
pub struct FileQcResult {
    /// The check that was performed.
    pub check: FileQcCheck,
    /// Whether the check passed.
    pub passed: bool,
    /// A human-readable description of the result or failure reason.
    pub detail: String,
}

impl FileQcResult {
    /// Creates a new `FileQcResult`.
    #[must_use]
    pub fn new(check: FileQcCheck, passed: bool, detail: impl Into<String>) -> Self {
        Self {
            check,
            passed,
            detail: detail.into(),
        }
    }
}

/// Specification for file-level QC.
#[derive(Debug, Clone)]
pub struct FileSpec {
    /// Minimum acceptable file size in bytes.
    pub min_size_bytes: u64,
    /// Maximum acceptable file size in bytes.
    pub max_size_bytes: u64,
    /// Expected duration in seconds (`None` to skip duration check).
    pub expected_duration_secs: Option<f32>,
    /// Allowable deviation from `expected_duration_secs` in seconds.
    pub duration_tolerance_secs: f32,
}

impl FileSpec {
    /// Creates a new `FileSpec`.
    #[must_use]
    pub fn new(
        min_size_bytes: u64,
        max_size_bytes: u64,
        expected_duration_secs: Option<f32>,
        duration_tolerance_secs: f32,
    ) -> Self {
        Self {
            min_size_bytes,
            max_size_bytes,
            expected_duration_secs,
            duration_tolerance_secs,
        }
    }

    /// Checks whether `actual` bytes falls within `[min_size_bytes, max_size_bytes]`.
    #[must_use]
    pub fn check_size(&self, actual: u64) -> FileQcResult {
        let passed = actual >= self.min_size_bytes && actual <= self.max_size_bytes;
        let detail = if passed {
            format!(
                "File size {} bytes is within [{}, {}]",
                actual, self.min_size_bytes, self.max_size_bytes
            )
        } else {
            format!(
                "File size {} bytes is outside [{}, {}]",
                actual, self.min_size_bytes, self.max_size_bytes
            )
        };
        FileQcResult::new(FileQcCheck::FileSize, passed, detail)
    }

    /// Checks whether `actual_secs` is within the tolerance of the expected duration.
    ///
    /// Returns `None` if no expected duration is set.
    #[must_use]
    pub fn check_duration(&self, actual_secs: f32) -> Option<FileQcResult> {
        let expected = self.expected_duration_secs?;
        let diff = (actual_secs - expected).abs();
        let passed = diff <= self.duration_tolerance_secs;
        let detail = if passed {
            format!(
                "Duration {:.2}s is within {:.2}s of expected {:.2}s",
                actual_secs, self.duration_tolerance_secs, expected
            )
        } else {
            format!(
                "Duration {:.2}s deviates {:.2}s from expected {:.2}s (tolerance {:.2}s)",
                actual_secs, diff, expected, self.duration_tolerance_secs
            )
        };
        Some(FileQcResult::new(FileQcCheck::Duration, passed, detail))
    }
}

/// A collection of `FileQcResult`s for a specific file.
#[derive(Debug, Clone)]
pub struct FileQcReport {
    /// Individual check results.
    pub results: Vec<FileQcResult>,
    /// Path to the file being reported on.
    pub file_path: String,
}

impl FileQcReport {
    /// Creates a new `FileQcReport`.
    #[must_use]
    pub fn new(results: Vec<FileQcResult>, file_path: impl Into<String>) -> Self {
        Self {
            results,
            file_path: file_path.into(),
        }
    }

    /// Returns the fraction of checks that passed.
    ///
    /// Returns `1.0` if there are no checks (vacuously true).
    #[must_use]
    pub fn pass_rate(&self) -> f32 {
        if self.results.is_empty() {
            return 1.0;
        }
        let passed = self.results.iter().filter(|r| r.passed).count();
        passed as f32 / self.results.len() as f32
    }

    /// Returns references to all checks that failed.
    #[must_use]
    pub fn failed_checks(&self) -> Vec<&FileQcResult> {
        self.results.iter().filter(|r| !r.passed).collect()
    }

    /// Returns `true` if every check passed.
    #[must_use]
    pub fn all_passed(&self) -> bool {
        self.results.iter().all(|r| r.passed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_str(name: &str) -> String {
        std::env::temp_dir()
            .join(format!("oximedia-qc-file-{name}"))
            .to_string_lossy()
            .into_owned()
    }

    // --- FileQcCheck tests ---

    #[test]
    fn test_file_size_is_structural() {
        assert!(FileQcCheck::FileSize.is_structural());
    }

    #[test]
    fn test_checksum_is_structural() {
        assert!(FileQcCheck::Checksum.is_structural());
    }

    #[test]
    fn test_container_is_structural() {
        assert!(FileQcCheck::Container.is_structural());
    }

    #[test]
    fn test_duration_not_structural() {
        assert!(!FileQcCheck::Duration.is_structural());
    }

    #[test]
    fn test_codec_not_structural() {
        assert!(!FileQcCheck::Codec.is_structural());
    }

    #[test]
    fn test_bitrate_not_structural() {
        assert!(!FileQcCheck::BitrateInRange.is_structural());
    }

    // --- FileQcResult tests ---

    #[test]
    fn test_result_creation() {
        let r = FileQcResult::new(FileQcCheck::FileSize, true, "OK");
        assert!(r.passed);
        assert_eq!(r.detail, "OK");
    }

    // --- FileSpec tests ---

    #[test]
    fn test_check_size_pass() {
        let spec = FileSpec::new(1000, 5000, None, 0.0);
        let result = spec.check_size(3000);
        assert!(result.passed);
        assert_eq!(result.check, FileQcCheck::FileSize);
    }

    #[test]
    fn test_check_size_fail_too_small() {
        let spec = FileSpec::new(1000, 5000, None, 0.0);
        let result = spec.check_size(500);
        assert!(!result.passed);
    }

    #[test]
    fn test_check_size_fail_too_large() {
        let spec = FileSpec::new(1000, 5000, None, 0.0);
        let result = spec.check_size(6000);
        assert!(!result.passed);
    }

    #[test]
    fn test_check_size_at_boundary_min() {
        let spec = FileSpec::new(1000, 5000, None, 0.0);
        let result = spec.check_size(1000);
        assert!(result.passed);
    }

    #[test]
    fn test_check_size_at_boundary_max() {
        let spec = FileSpec::new(1000, 5000, None, 0.0);
        let result = spec.check_size(5000);
        assert!(result.passed);
    }

    #[test]
    fn test_check_duration_none_when_not_set() {
        let spec = FileSpec::new(0, u64::MAX, None, 1.0);
        assert!(spec.check_duration(30.0).is_none());
    }

    #[test]
    fn test_check_duration_pass() {
        let spec = FileSpec::new(0, u64::MAX, Some(60.0), 1.0);
        let result = spec.check_duration(60.5).expect("should succeed in test");
        assert!(result.passed);
    }

    #[test]
    fn test_check_duration_fail() {
        let spec = FileSpec::new(0, u64::MAX, Some(60.0), 1.0);
        let result = spec.check_duration(62.0).expect("should succeed in test");
        assert!(!result.passed);
    }

    // --- FileQcReport tests ---

    #[test]
    fn test_report_all_passed() {
        let results = vec![
            FileQcResult::new(FileQcCheck::FileSize, true, "ok"),
            FileQcResult::new(FileQcCheck::Duration, true, "ok"),
        ];
        let report = FileQcReport::new(results, tmp_str("test.mp4"));
        assert!(report.all_passed());
        assert!((report.pass_rate() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_report_partial_pass() {
        let results = vec![
            FileQcResult::new(FileQcCheck::FileSize, true, "ok"),
            FileQcResult::new(FileQcCheck::Duration, false, "too short"),
        ];
        let report = FileQcReport::new(results, tmp_str("test.mp4"));
        assert!(!report.all_passed());
        assert!((report.pass_rate() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_report_failed_checks() {
        let results = vec![
            FileQcResult::new(FileQcCheck::FileSize, false, "too small"),
            FileQcResult::new(FileQcCheck::Duration, true, "ok"),
        ];
        let report = FileQcReport::new(results, tmp_str("test.mp4"));
        let failed = report.failed_checks();
        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0].check, FileQcCheck::FileSize);
    }

    #[test]
    fn test_report_empty_is_full_pass() {
        let report = FileQcReport::new(vec![], tmp_str("empty.mp4"));
        assert!((report.pass_rate() - 1.0).abs() < 1e-6);
        assert!(report.all_passed());
    }
}
