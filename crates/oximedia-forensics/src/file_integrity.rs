#![allow(dead_code)]
//! File integrity checking for forensic workflows.
//!
//! Provides structured checks on media files (size, hash consistency, header
//! validity) and aggregates results into a severity-ranked integrity report.

use std::collections::HashMap;

/// Category of integrity issue found.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum IntegrityCheck {
    /// File is intact – no issues.
    Ok,
    /// Minor anomaly (e.g. unusual but valid metadata).
    Info,
    /// Potential issue that warrants review.
    Warning,
    /// Significant problem that likely indicates tampering or corruption.
    Error,
    /// Severe, unrecoverable issue (e.g. truncated stream, invalid header).
    Critical,
}

impl IntegrityCheck {
    /// Numeric severity (higher = worse).
    pub fn severity(&self) -> u8 {
        match self {
            IntegrityCheck::Ok => 0,
            IntegrityCheck::Info => 1,
            IntegrityCheck::Warning => 2,
            IntegrityCheck::Error => 3,
            IntegrityCheck::Critical => 4,
        }
    }

    /// Returns `true` when the check represents an actionable problem.
    pub fn is_problem(&self) -> bool {
        self.severity() >= IntegrityCheck::Warning.severity()
    }

    /// Human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            IntegrityCheck::Ok => "OK",
            IntegrityCheck::Info => "INFO",
            IntegrityCheck::Warning => "WARNING",
            IntegrityCheck::Error => "ERROR",
            IntegrityCheck::Critical => "CRITICAL",
        }
    }
}

/// A single integrity finding for one aspect of a file.
#[derive(Debug, Clone)]
pub struct IntegrityFinding {
    /// The check level.
    pub level: IntegrityCheck,
    /// Short description of what was checked.
    pub check_name: String,
    /// Human-readable description of the finding.
    pub message: String,
}

impl IntegrityFinding {
    /// Create a new finding.
    pub fn new(
        level: IntegrityCheck,
        check_name: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            level,
            check_name: check_name.into(),
            message: message.into(),
        }
    }
}

/// All integrity findings for a single file.
#[derive(Debug, Clone)]
pub struct FileIntegrity {
    /// Path (or identifier) of the file being checked.
    pub path: String,
    /// All findings in evaluation order.
    pub findings: Vec<IntegrityFinding>,
}

impl FileIntegrity {
    /// Create a new, empty integrity record.
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            findings: Vec::new(),
        }
    }

    /// Append a finding.
    pub fn add_finding(&mut self, finding: IntegrityFinding) {
        self.findings.push(finding);
    }

    /// Returns `true` when any finding is at Warning level or above.
    pub fn has_issues(&self) -> bool {
        self.findings.iter().any(|f| f.level.is_problem())
    }

    /// The highest severity level among all findings.
    pub fn max_severity(&self) -> IntegrityCheck {
        self.findings
            .iter()
            .map(|f| f.level)
            .max()
            .unwrap_or(IntegrityCheck::Ok)
    }

    /// Count findings at exactly the given level.
    pub fn count_at_level(&self, level: IntegrityCheck) -> usize {
        self.findings.iter().filter(|f| f.level == level).count()
    }
}

/// Aggregated result of checking one or more files.
#[derive(Debug)]
pub struct IntegrityResult {
    /// Per-file results keyed by path.
    pub files: HashMap<String, FileIntegrity>,
}

impl IntegrityResult {
    /// Create an empty result container.
    pub fn new() -> Self {
        Self {
            files: HashMap::new(),
        }
    }

    /// Insert a per-file result.
    pub fn insert(&mut self, integrity: FileIntegrity) {
        self.files.insert(integrity.path.clone(), integrity);
    }

    /// Count total critical-level findings across all files.
    pub fn critical_count(&self) -> usize {
        self.files
            .values()
            .map(|fi| fi.count_at_level(IntegrityCheck::Critical))
            .sum()
    }

    /// Count total error-or-worse findings.
    pub fn error_count(&self) -> usize {
        self.files
            .values()
            .flat_map(|fi| fi.findings.iter())
            .filter(|f| f.level >= IntegrityCheck::Error)
            .count()
    }

    /// Returns `true` when no file has any issues.
    pub fn all_clean(&self) -> bool {
        self.files.values().all(|fi| !fi.has_issues())
    }

    /// Number of files checked.
    pub fn file_count(&self) -> usize {
        self.files.len()
    }
}

impl Default for IntegrityResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Runs integrity checks on media file byte content.
pub struct FileIntegrityChecker {
    /// Minimum severity level to include in reports.
    pub min_report_level: IntegrityCheck,
}

impl FileIntegrityChecker {
    /// Create a checker that reports everything including info-level findings.
    pub fn new() -> Self {
        Self {
            min_report_level: IntegrityCheck::Info,
        }
    }

    /// Create a checker that only reports warnings and above.
    pub fn warnings_only() -> Self {
        Self {
            min_report_level: IntegrityCheck::Warning,
        }
    }

    /// Check `data` as a media file identified by `path`.
    ///
    /// Performs:
    /// - Empty-file check (Critical)
    /// - Minimum size check (Warning if < 16 bytes)
    /// - Magic-byte presence check (Error if no recognised header)
    /// - Truncation heuristic (Warning if last 4 bytes are all zero)
    pub fn check(&self, path: impl Into<String>, data: &[u8]) -> FileIntegrity {
        let path = path.into();
        let mut fi = FileIntegrity::new(&path);

        if data.is_empty() {
            fi.add_finding(IntegrityFinding::new(
                IntegrityCheck::Critical,
                "empty_file",
                "File is empty",
            ));
            return fi;
        }

        if data.len() < 16 {
            fi.add_finding(IntegrityFinding::new(
                IntegrityCheck::Warning,
                "min_size",
                format!("File is very small ({} bytes)", data.len()),
            ));
        }

        // Check for recognisable media magic bytes.
        let has_magic = self.detect_magic(data);
        if !has_magic {
            fi.add_finding(IntegrityFinding::new(
                IntegrityCheck::Error,
                "magic_bytes",
                "No recognised media container header found",
            ));
        } else {
            fi.add_finding(IntegrityFinding::new(
                IntegrityCheck::Ok,
                "magic_bytes",
                "Valid media header detected",
            ));
        }

        // Truncation heuristic: last 4 bytes all zero is suspicious.
        if data.len() >= 4 && data[data.len() - 4..].iter().all(|&b| b == 0) {
            fi.add_finding(IntegrityFinding::new(
                IntegrityCheck::Warning,
                "truncation",
                "File ends with null bytes – possible truncation",
            ));
        }

        fi
    }

    /// Detect common media magic bytes.
    fn detect_magic(&self, data: &[u8]) -> bool {
        if data.len() < 4 {
            return false;
        }
        let magic = &data[..4];
        // MP4 / MOV: `ftyp` at offset 4 (skip size check for simplicity).
        // JPEG: FF D8 FF
        // PNG: 89 50 4E 47
        // RIFF (WAV/AVI): 52 49 46 46
        // MKV/WebM: 1A 45 DF A3
        matches!(
            magic,
            [0xFF, 0xD8, 0xFF, _]
                | [0x89, 0x50, 0x4E, 0x47]
                | [0x52, 0x49, 0x46, 0x46]
                | [0x1A, 0x45, 0xDF, 0xA3]
                | [0x00, 0x00, 0x00, _] // MP4 size prefix
        )
    }

    /// Run checks on multiple files and aggregate into an `IntegrityResult`.
    pub fn report<'a>(
        &self,
        files: impl IntoIterator<Item = (&'a str, &'a [u8])>,
    ) -> IntegrityResult {
        let mut result = IntegrityResult::new();
        for (path, data) in files {
            let fi = self.check(path, data);
            // Filter by min report level.
            let mut filtered = FileIntegrity::new(fi.path.clone());
            for finding in fi.findings {
                if finding.level >= self.min_report_level {
                    filtered.add_finding(finding);
                }
            }
            result.insert(filtered);
        }
        result
    }
}

impl Default for FileIntegrityChecker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_str(name: &str) -> String {
        std::env::temp_dir()
            .join(format!("oximedia-forensics-{name}"))
            .to_string_lossy()
            .into_owned()
    }

    #[test]
    fn test_integrity_check_severity_ordering() {
        assert!(IntegrityCheck::Ok.severity() < IntegrityCheck::Info.severity());
        assert!(IntegrityCheck::Info.severity() < IntegrityCheck::Warning.severity());
        assert!(IntegrityCheck::Warning.severity() < IntegrityCheck::Error.severity());
        assert!(IntegrityCheck::Error.severity() < IntegrityCheck::Critical.severity());
    }

    #[test]
    fn test_integrity_check_is_problem() {
        assert!(!IntegrityCheck::Ok.is_problem());
        assert!(!IntegrityCheck::Info.is_problem());
        assert!(IntegrityCheck::Warning.is_problem());
        assert!(IntegrityCheck::Error.is_problem());
        assert!(IntegrityCheck::Critical.is_problem());
    }

    #[test]
    fn test_integrity_check_labels() {
        assert_eq!(IntegrityCheck::Ok.label(), "OK");
        assert_eq!(IntegrityCheck::Critical.label(), "CRITICAL");
    }

    #[test]
    fn test_file_integrity_has_no_issues_when_empty() {
        let fi = FileIntegrity::new(tmp_str("clean.mp4"));
        assert!(!fi.has_issues());
    }

    #[test]
    fn test_file_integrity_has_issues_with_warning() {
        let mut fi = FileIntegrity::new(tmp_str("warn.mp4"));
        fi.add_finding(IntegrityFinding::new(
            IntegrityCheck::Warning,
            "test",
            "test warning",
        ));
        assert!(fi.has_issues());
    }

    #[test]
    fn test_file_integrity_max_severity() {
        let mut fi = FileIntegrity::new(tmp_str("test.mp4"));
        fi.add_finding(IntegrityFinding::new(IntegrityCheck::Info, "a", "info"));
        fi.add_finding(IntegrityFinding::new(IntegrityCheck::Error, "b", "error"));
        assert_eq!(fi.max_severity(), IntegrityCheck::Error);
    }

    #[test]
    fn test_file_integrity_count_at_level() {
        let mut fi = FileIntegrity::new(tmp_str("test.mp4"));
        fi.add_finding(IntegrityFinding::new(IntegrityCheck::Warning, "a", "w1"));
        fi.add_finding(IntegrityFinding::new(IntegrityCheck::Warning, "b", "w2"));
        fi.add_finding(IntegrityFinding::new(IntegrityCheck::Error, "c", "e1"));
        assert_eq!(fi.count_at_level(IntegrityCheck::Warning), 2);
        assert_eq!(fi.count_at_level(IntegrityCheck::Error), 1);
    }

    #[test]
    fn test_checker_empty_file_critical() {
        let checker = FileIntegrityChecker::new();
        let fi = checker.check(tmp_str("empty.mp4"), &[]);
        assert_eq!(fi.max_severity(), IntegrityCheck::Critical);
    }

    #[test]
    fn test_checker_jpeg_magic_ok() {
        let checker = FileIntegrityChecker::new();
        // JPEG magic: FF D8 FF E0 + padding
        let data = [
            0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01, 0x01, 0x00,
            0x00, 0x01, 0x01,
        ];
        let fi = checker.check(tmp_str("test.jpg"), &data);
        let ok_count = fi.count_at_level(IntegrityCheck::Ok);
        assert!(
            ok_count > 0,
            "expected at least one OK finding for magic bytes"
        );
    }

    #[test]
    fn test_checker_no_magic_error() {
        let checker = FileIntegrityChecker::new();
        let data = [
            0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88,
            0x99, 0x00,
        ];
        let fi = checker.check(tmp_str("unknown.bin"), &data);
        assert!(fi.count_at_level(IntegrityCheck::Error) > 0);
    }

    #[test]
    fn test_checker_truncation_warning() {
        let checker = FileIntegrityChecker::new();
        // PNG magic followed by trailing zeros
        let mut data = vec![
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48,
            0x44, 0x52,
        ];
        data.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // trailing zeros
        let fi = checker.check(tmp_str("trunc.png"), &data);
        assert!(fi.count_at_level(IntegrityCheck::Warning) > 0);
    }

    #[test]
    fn test_integrity_result_critical_count() {
        let checker = FileIntegrityChecker::new();
        let a = tmp_str("a.mp4");
        let b = tmp_str("b.mp4");
        let result = checker.report([(a.as_str(), [].as_slice()), (b.as_str(), [].as_slice())]);
        assert_eq!(result.critical_count(), 2);
    }

    #[test]
    fn test_integrity_result_all_clean_false_when_issues() {
        let checker = FileIntegrityChecker::new();
        let bad = tmp_str("bad.mp4");
        let result = checker.report([(bad.as_str(), [].as_slice())]);
        assert!(!result.all_clean());
    }

    #[test]
    fn test_integrity_result_file_count() {
        let checker = FileIntegrityChecker::new();
        let jpeg = [
            0xFF, 0xD8, 0xFF, 0xE0u8, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01, 0x01, 0x00,
            0x00, 0x01, 0x01,
        ];
        let a = tmp_str("a.jpg");
        let b = tmp_str("b.jpg");
        let result = checker.report([(a.as_str(), jpeg.as_slice()), (b.as_str(), jpeg.as_slice())]);
        assert_eq!(result.file_count(), 2);
    }
}
