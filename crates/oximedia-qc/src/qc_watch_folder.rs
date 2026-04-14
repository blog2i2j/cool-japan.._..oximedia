//! QC watch-folder scanning — automatically validates files that arrive in a monitored directory.
//!
//! `WatchFolderScanner::scan_once` performs a single-pass scan: it finds files not yet
//! validated (by consulting an in-memory seen-set), runs QC on each, and returns
//! `QcJobResult` values. Callers can embed this in a polling loop or trigger it
//! from a filesystem notification system.

#![allow(dead_code)]

use crate::qc_report::{QcCheckResult, QcFinding, QcReport};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Media file extensions recognised by the watch-folder scanner.
const MEDIA_EXTENSIONS: &[&str] = &[
    "mp4", "mkv", "mov", "mxf", "webm", "avi", "ts", "m2ts", "mts", "flv", "wmv", "ogv", "wav",
    "flac", "opus", "ogg", "aac", "mp3", "m4a",
];

/// Configuration for a QC watch-folder.
#[derive(Debug, Clone)]
pub struct WatchFolderConfig {
    /// Directory to monitor.
    pub path: PathBuf,
    /// Name of the QC template to apply to each file.
    pub template: String,
    /// Directory where QC results are written.
    pub output_dir: PathBuf,
    /// Whether to recurse into sub-directories.
    pub recursive: bool,
    /// Maximum number of files to process per scan pass (0 = unlimited).
    pub max_files_per_scan: usize,
}

impl WatchFolderConfig {
    /// Creates a new watch-folder configuration.
    pub fn new(
        path: impl Into<PathBuf>,
        template: impl Into<String>,
        output_dir: impl Into<PathBuf>,
        recursive: bool,
    ) -> Self {
        Self {
            path: path.into(),
            template: template.into(),
            output_dir: output_dir.into(),
            recursive,
            max_files_per_scan: 0,
        }
    }

    /// Sets the maximum files processed per scan.
    pub fn with_max_files(mut self, max: usize) -> Self {
        self.max_files_per_scan = max;
        self
    }
}

// ---------------------------------------------------------------------------
// Job result
// ---------------------------------------------------------------------------

/// The outcome of running QC on a single file.
#[derive(Debug, Clone)]
pub struct QcJobResult {
    /// Path to the validated file.
    pub file_path: PathBuf,
    /// The QC report for this file.
    pub report: QcReport,
    /// Wall-clock time when validation completed.
    pub validated_at: SystemTime,
}

impl QcJobResult {
    /// Creates a new job result.
    pub fn new(file_path: PathBuf, report: QcReport, validated_at: SystemTime) -> Self {
        Self {
            file_path,
            report,
            validated_at,
        }
    }

    /// Returns `true` if the QC passed overall.
    pub fn passed(&self) -> bool {
        self.report.overall_pass()
    }
}

// ---------------------------------------------------------------------------
// Scanner
// ---------------------------------------------------------------------------

/// Scans a watch folder for new media files and runs QC on them.
///
/// The scanner maintains a set of already-seen file paths so that repeated
/// calls to [`scan_once`](WatchFolderScanner::scan_once) only process new arrivals.
#[derive(Debug, Default)]
pub struct WatchFolderScanner {
    /// Paths that have already been validated in a previous scan pass.
    seen: HashSet<PathBuf>,
}

impl WatchFolderScanner {
    /// Creates a new scanner with an empty seen-set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Resets the seen-set so all files will be re-validated on the next scan.
    pub fn reset(&mut self) {
        self.seen.clear();
    }

    /// Returns the number of files processed so far.
    pub fn seen_count(&self) -> usize {
        self.seen.len()
    }

    /// Performs a single-pass scan of the watch folder.
    ///
    /// Discovers media files not yet seen, runs a lightweight QC analysis on
    /// each (structural checks only — no I/O decode), records them in the
    /// seen-set and returns a `QcJobResult` per processed file.
    ///
    /// # Errors
    ///
    /// Directory I/O errors are silently skipped; the method returns results for
    /// files that could be processed.
    pub fn scan_once(&mut self, config: &WatchFolderConfig) -> Vec<QcJobResult> {
        let mut results = Vec::new();

        let files = self.discover_files(&config.path, config.recursive);

        for file in files {
            if self.seen.contains(&file) {
                continue;
            }

            let limit_reached =
                config.max_files_per_scan > 0 && results.len() >= config.max_files_per_scan;
            if limit_reached {
                break;
            }

            let report = self.run_qc(&file, &config.template);
            let job = QcJobResult::new(file.clone(), report, SystemTime::now());
            results.push(job);
            self.seen.insert(file);
        }

        results
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Discovers media files in `dir`, optionally recursing into sub-directories.
    fn discover_files(&self, dir: &Path, recursive: bool) -> Vec<PathBuf> {
        let mut found = Vec::new();
        self.collect_files(dir, recursive, &mut found);
        found.sort();
        found
    }

    fn collect_files(&self, dir: &Path, recursive: bool, out: &mut Vec<PathBuf>) {
        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => return,
        };

        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if recursive {
                    self.collect_files(&path, recursive, out);
                }
            } else if self.is_media_file(&path) {
                out.push(path);
            }
        }
    }

    /// Returns `true` if the path has a recognised media file extension.
    fn is_media_file(&self, path: &Path) -> bool {
        path.extension()
            .and_then(|e| e.to_str())
            .map(|ext| MEDIA_EXTENSIONS.contains(&ext.to_lowercase().as_str()))
            .unwrap_or(false)
    }

    /// Runs a lightweight structural QC on the given file.
    ///
    /// In a full implementation this would invoke the [`crate::QualityControl`]
    /// pipeline. Here we perform existence / readability checks and return
    /// a structured report.
    fn run_qc(&self, file: &Path, template: &str) -> QcReport {
        let mut report = QcReport::with_label(
            file.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown"),
        );

        // Check file exists and is readable
        match std::fs::metadata(file) {
            Err(e) => {
                let finding = QcFinding::new(
                    "file_access",
                    crate::qc_report::FindingSeverity::Fatal,
                    format!("Cannot access file: {e}"),
                );
                let mut check = QcCheckResult::pass("file_access");
                check.add_finding(finding);
                report.add_result(check);
                return report;
            }
            Ok(meta) => {
                // Zero-byte file check
                if meta.len() == 0 {
                    let finding = QcFinding::new(
                        "file_size",
                        crate::qc_report::FindingSeverity::Error,
                        "File is empty (0 bytes)".to_string(),
                    );
                    let check = QcCheckResult::fail("file_size", vec![finding]);
                    report.add_result(check);
                } else {
                    report.add_result(QcCheckResult::pass("file_size"));
                }
            }
        }

        // Extension / format structural check
        if self.is_media_file(file) {
            report.add_result(QcCheckResult::pass("file_extension"));
        } else {
            let finding = QcFinding::new(
                "file_extension",
                crate::qc_report::FindingSeverity::Warning,
                format!(
                    "File '{}' does not have a recognised media extension",
                    file.display()
                ),
            );
            report.add_result(QcCheckResult::fail("file_extension", vec![finding]));
        }

        // Template name recorded as informational finding
        {
            let mut info = QcCheckResult::pass("template_applied");
            info.add_finding(QcFinding::new(
                "template_applied",
                crate::qc_report::FindingSeverity::Info,
                format!("Applied QC template: {template}"),
            ));
            report.add_result(info);
        }

        report
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn make_temp_dir(suffix: &str) -> PathBuf {
        let mut dir = std::env::temp_dir();
        dir.push(format!(
            "oximedia_qc_watchfolder_test_{suffix}_{}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    fn cleanup(dir: &Path) {
        let _ = fs::remove_dir_all(dir);
    }

    fn touch(dir: &Path, name: &str) -> PathBuf {
        let p = dir.join(name);
        fs::write(&p, b"fake media data for testing purposes").expect("write test file");
        p
    }

    #[test]
    fn test_scan_empty_dir_returns_no_results() {
        let dir = make_temp_dir("empty");
        let out = make_temp_dir("out_empty");
        let config = WatchFolderConfig::new(&dir, "broadcast", &out, false);
        let mut scanner = WatchFolderScanner::new();
        let results = scanner.scan_once(&config);
        assert!(results.is_empty());
        cleanup(&dir);
        cleanup(&out);
    }

    #[test]
    fn test_scan_finds_media_file() {
        let dir = make_temp_dir("find_media");
        let out = make_temp_dir("out_find_media");
        touch(&dir, "video.mp4");
        let config = WatchFolderConfig::new(&dir, "broadcast", &out, false);
        let mut scanner = WatchFolderScanner::new();
        let results = scanner.scan_once(&config);
        assert_eq!(results.len(), 1);
        assert!(results[0].file_path.ends_with("video.mp4"));
        cleanup(&dir);
        cleanup(&out);
    }

    #[test]
    fn test_scan_ignores_non_media_files() {
        let dir = make_temp_dir("non_media");
        let out = make_temp_dir("out_non_media");
        touch(&dir, "readme.txt");
        touch(&dir, "manifest.xml");
        let config = WatchFolderConfig::new(&dir, "broadcast", &out, false);
        let mut scanner = WatchFolderScanner::new();
        let results = scanner.scan_once(&config);
        assert!(results.is_empty());
        cleanup(&dir);
        cleanup(&out);
    }

    #[test]
    fn test_scan_does_not_reprocess_seen_files() {
        let dir = make_temp_dir("no_reprocess");
        let out = make_temp_dir("out_no_reprocess");
        touch(&dir, "video.mkv");
        let config = WatchFolderConfig::new(&dir, "broadcast", &out, false);
        let mut scanner = WatchFolderScanner::new();
        let first = scanner.scan_once(&config);
        assert_eq!(first.len(), 1);
        let second = scanner.scan_once(&config);
        assert!(
            second.is_empty(),
            "Already-seen files should not be re-processed"
        );
        cleanup(&dir);
        cleanup(&out);
    }

    #[test]
    fn test_scan_recursive_finds_nested_file() {
        let dir = make_temp_dir("recursive");
        let out = make_temp_dir("out_recursive");
        let sub = dir.join("sub");
        fs::create_dir_all(&sub).expect("create sub dir");
        touch(&sub, "audio.flac");
        let config = WatchFolderConfig::new(&dir, "broadcast", &out, true);
        let mut scanner = WatchFolderScanner::new();
        let results = scanner.scan_once(&config);
        assert_eq!(results.len(), 1);
        cleanup(&dir);
        cleanup(&out);
    }

    #[test]
    fn test_scan_non_recursive_ignores_nested_file() {
        let dir = make_temp_dir("non_recursive");
        let out = make_temp_dir("out_non_recursive");
        let sub = dir.join("sub");
        fs::create_dir_all(&sub).expect("create sub dir");
        touch(&sub, "audio.wav");
        let config = WatchFolderConfig::new(&dir, "broadcast", &out, false);
        let mut scanner = WatchFolderScanner::new();
        let results = scanner.scan_once(&config);
        assert!(results.is_empty(), "Non-recursive scan should not descend");
        cleanup(&dir);
        cleanup(&out);
    }

    #[test]
    fn test_scan_max_files_limit() {
        let dir = make_temp_dir("max_files");
        let out = make_temp_dir("out_max_files");
        touch(&dir, "a.mp4");
        touch(&dir, "b.mp4");
        touch(&dir, "c.mp4");
        let config = WatchFolderConfig::new(&dir, "broadcast", &out, false).with_max_files(2);
        let mut scanner = WatchFolderScanner::new();
        let results = scanner.scan_once(&config);
        assert_eq!(results.len(), 2, "Should respect max_files_per_scan");
        cleanup(&dir);
        cleanup(&out);
    }

    #[test]
    fn test_scan_reset_reprocesses_files() {
        let dir = make_temp_dir("reset");
        let out = make_temp_dir("out_reset");
        touch(&dir, "video.mp4");
        let config = WatchFolderConfig::new(&dir, "broadcast", &out, false);
        let mut scanner = WatchFolderScanner::new();
        scanner.scan_once(&config);
        scanner.reset();
        let results = scanner.scan_once(&config);
        assert_eq!(results.len(), 1, "After reset, file should be re-processed");
        cleanup(&dir);
        cleanup(&out);
    }

    #[test]
    fn test_job_result_has_validated_at() {
        let dir = make_temp_dir("validated_at");
        let out = make_temp_dir("out_validated_at");
        touch(&dir, "video.mkv");
        let config = WatchFolderConfig::new(&dir, "broadcast", &out, false);
        let mut scanner = WatchFolderScanner::new();
        let results = scanner.scan_once(&config);
        assert_eq!(results.len(), 1);
        // validated_at should be close to now
        let elapsed = results[0].validated_at.elapsed().unwrap_or_default();
        assert!(elapsed.as_secs() < 5, "validated_at should be recent");
        cleanup(&dir);
        cleanup(&out);
    }
}
