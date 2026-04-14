//! Deduplication reporting for media archives.
//!
//! Provides analysis of an archive's deduplication state: space-savings
//! summaries, per-group duplicate listings, reduction ratios, and formatted
//! text/CSV/JSON report output.
//!
//! This module is intentionally dependency-light and operates on in-memory
//! data structures populated by the caller (e.g., from `dedup_archive`).

#![allow(dead_code)]

use crate::{ArchiveError, ArchiveResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Write as FmtWrite;
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Core data types
// ---------------------------------------------------------------------------

/// A group of files that share the same content fingerprint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DuplicateGroup {
    /// Content fingerprint / digest that all files share.
    pub fingerprint: String,
    /// Size of a single copy in bytes.
    pub file_size: u64,
    /// Paths of all copies (at least 2 for a duplicate group).
    pub paths: Vec<PathBuf>,
}

impl DuplicateGroup {
    /// Create a new duplicate group.
    pub fn new(fingerprint: impl Into<String>, file_size: u64, paths: Vec<PathBuf>) -> Self {
        Self {
            fingerprint: fingerprint.into(),
            file_size,
            paths,
        }
    }

    /// Number of redundant copies (total copies − 1).
    #[must_use]
    pub fn redundant_copies(&self) -> usize {
        self.paths.len().saturating_sub(1)
    }

    /// Bytes that could be freed by keeping only one copy.
    #[must_use]
    pub fn reclaimable_bytes(&self) -> u64 {
        self.file_size
            .saturating_mul(self.redundant_copies() as u64)
    }
}

/// Overall deduplication statistics for an archive.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DedupStats {
    /// Total number of files scanned.
    pub total_files: u64,
    /// Number of unique content fingerprints found.
    pub unique_files: u64,
    /// Number of files that are duplicates (total − unique).
    pub duplicate_files: u64,
    /// Total logical size of all files in bytes (including duplicates).
    pub total_bytes: u64,
    /// Size of the unique set in bytes (sum of one copy per fingerprint).
    pub unique_bytes: u64,
    /// Bytes that could be freed by deduplication.
    pub reclaimable_bytes: u64,
    /// Number of duplicate groups (fingerprints with ≥ 2 copies).
    pub duplicate_groups: u64,
}

impl DedupStats {
    /// Deduplication ratio: logical bytes / unique bytes.
    ///
    /// Returns `1.0` when there is nothing to deduplicate.
    #[must_use]
    pub fn dedup_ratio(&self) -> f64 {
        if self.unique_bytes == 0 {
            return 1.0;
        }
        self.total_bytes as f64 / self.unique_bytes as f64
    }

    /// Space savings as a percentage: `(1 - 1/ratio) * 100`.
    #[must_use]
    pub fn savings_pct(&self) -> f64 {
        let ratio = self.dedup_ratio();
        if ratio <= 1.0 {
            return 0.0;
        }
        (1.0 - 1.0 / ratio) * 100.0
    }
}

// ---------------------------------------------------------------------------
// DedupAnalyzer
// ---------------------------------------------------------------------------

/// Analyses a set of (path, size, fingerprint) records and produces
/// a `DedupReport`.
#[derive(Debug, Default)]
pub struct DedupAnalyzer {
    /// All records ingested: (fingerprint, size, path).
    records: Vec<(String, u64, PathBuf)>,
}

impl DedupAnalyzer {
    /// Create a new empty analyzer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a file record.
    pub fn add_file(
        &mut self,
        fingerprint: impl Into<String>,
        size_bytes: u64,
        path: impl Into<PathBuf>,
    ) {
        self.records
            .push((fingerprint.into(), size_bytes, path.into()));
    }

    /// Build a `DedupReport` from all added records.
    pub fn build_report(&self) -> DedupReport {
        // Group paths by fingerprint, tracking size per fingerprint.
        let mut groups: HashMap<String, (u64, Vec<PathBuf>)> = HashMap::new();
        for (fp, size, path) in &self.records {
            let entry = groups
                .entry(fp.clone())
                .or_insert_with(|| (*size, Vec::new()));
            entry.1.push(path.clone());
        }

        let total_files = self.records.len() as u64;
        let mut duplicate_groups_vec: Vec<DuplicateGroup> = Vec::new();
        let mut unique_bytes: u64 = 0;
        let mut total_bytes: u64 = 0;
        let mut duplicate_files: u64 = 0;
        let mut duplicate_group_count: u64 = 0;

        for (fp, (size, paths)) in &groups {
            unique_bytes = unique_bytes.saturating_add(*size);
            total_bytes = total_bytes.saturating_add(size.saturating_mul(paths.len() as u64));

            if paths.len() > 1 {
                duplicate_group_count += 1;
                duplicate_files += (paths.len() - 1) as u64;
                duplicate_groups_vec.push(DuplicateGroup::new(fp.clone(), *size, paths.clone()));
            }
        }

        // Sort duplicate groups by reclaimable bytes descending for readability.
        duplicate_groups_vec.sort_by(|a, b| b.reclaimable_bytes().cmp(&a.reclaimable_bytes()));

        let reclaimable_bytes = total_bytes.saturating_sub(unique_bytes);

        let stats = DedupStats {
            total_files,
            unique_files: groups.len() as u64,
            duplicate_files,
            total_bytes,
            unique_bytes,
            reclaimable_bytes,
            duplicate_groups: duplicate_group_count,
        };

        DedupReport {
            stats,
            duplicate_groups: duplicate_groups_vec,
        }
    }
}

// ---------------------------------------------------------------------------
// DedupReport
// ---------------------------------------------------------------------------

/// Complete deduplication report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DedupReport {
    /// Aggregate statistics.
    pub stats: DedupStats,
    /// Per-group duplicate listings.
    pub duplicate_groups: Vec<DuplicateGroup>,
}

/// Output format for a rendered report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReportFormat {
    /// Human-readable plain text.
    Text,
    /// JSON (pretty-printed).
    Json,
    /// CSV with one row per duplicate pair.
    Csv,
}

impl DedupReport {
    /// Render this report as a string in the requested format.
    pub fn render(&self, format: ReportFormat) -> ArchiveResult<String> {
        match format {
            ReportFormat::Text => self.render_text(),
            ReportFormat::Json => self.render_json(),
            ReportFormat::Csv => self.render_csv(),
        }
    }

    /// Write the report to a file.
    pub fn write_to_file(&self, path: &std::path::Path, format: ReportFormat) -> ArchiveResult<()> {
        let content = self.render(format)?;
        std::fs::write(path, content.as_bytes()).map_err(ArchiveError::Io)
    }

    fn render_text(&self) -> ArchiveResult<String> {
        let s = &self.stats;
        let mut out = String::new();
        writeln!(out, "=== Archive Deduplication Report ===").ok();
        writeln!(out).ok();
        writeln!(out, "Total files scanned : {}", s.total_files).ok();
        writeln!(out, "Unique fingerprints  : {}", s.unique_files).ok();
        writeln!(out, "Duplicate files      : {}", s.duplicate_files).ok();
        writeln!(out, "Duplicate groups     : {}", s.duplicate_groups).ok();
        writeln!(out).ok();
        writeln!(out, "Total logical size   : {} bytes", s.total_bytes).ok();
        writeln!(out, "Unique set size      : {} bytes", s.unique_bytes).ok();
        writeln!(out, "Reclaimable space    : {} bytes", s.reclaimable_bytes).ok();
        writeln!(out, "Deduplication ratio  : {:.3}x", s.dedup_ratio()).ok();
        writeln!(out, "Space savings        : {:.1}%", s.savings_pct()).ok();

        if !self.duplicate_groups.is_empty() {
            writeln!(out).ok();
            writeln!(out, "--- Duplicate Groups (by reclaimable bytes desc) ---").ok();
            for (i, group) in self.duplicate_groups.iter().enumerate() {
                writeln!(
                    out,
                    "\n[Group {}] fingerprint={} size={} copies={} reclaimable={}",
                    i + 1,
                    group.fingerprint,
                    group.file_size,
                    group.paths.len(),
                    group.reclaimable_bytes()
                )
                .ok();
                for (j, path) in group.paths.iter().enumerate() {
                    let marker = if j == 0 { "keep" } else { "dup " };
                    writeln!(out, "  [{marker}] {}", path.display()).ok();
                }
            }
        }

        Ok(out)
    }

    fn render_json(&self) -> ArchiveResult<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| ArchiveError::Validation(format!("JSON error: {e}")))
    }

    fn render_csv(&self) -> ArchiveResult<String> {
        let mut out = String::new();
        writeln!(
            out,
            "fingerprint,file_size_bytes,canonical_path,duplicate_path,reclaimable_bytes"
        )
        .ok();
        for group in &self.duplicate_groups {
            let canonical = group
                .paths
                .first()
                .map(|p| p.display().to_string())
                .unwrap_or_default();
            for dup in group.paths.iter().skip(1) {
                writeln!(
                    out,
                    "{},{},{},{},{}",
                    group.fingerprint,
                    group.file_size,
                    csv_escape(&canonical),
                    csv_escape(&dup.display().to_string()),
                    group.file_size,
                )
                .ok();
            }
        }
        Ok(out)
    }
}

/// Escape a string for CSV output (wrap in quotes if it contains commas/quotes).
fn csv_escape(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

// ---------------------------------------------------------------------------
// Top-level convenience function
// ---------------------------------------------------------------------------

/// Build a dedup report from a slice of `(fingerprint, size, path)` tuples.
pub fn build_report_from_records(records: &[(String, u64, PathBuf)]) -> DedupReport {
    let mut analyzer = DedupAnalyzer::new();
    for (fp, size, path) in records {
        analyzer.add_file(fp.clone(), *size, path.clone());
    }
    analyzer.build_report()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn p(s: &str) -> PathBuf {
        PathBuf::from(s)
    }

    // ── DuplicateGroup ─────────────────────────────────────────────────────────

    #[test]
    fn test_duplicate_group_reclaimable_bytes() {
        let group = DuplicateGroup::new(
            "abc123",
            1_000_000,
            vec![p("/a/file.mkv"), p("/b/file.mkv"), p("/c/file.mkv")],
        );
        assert_eq!(group.redundant_copies(), 2);
        assert_eq!(group.reclaimable_bytes(), 2_000_000);
    }

    #[test]
    fn test_duplicate_group_single_copy_no_redundancy() {
        let group = DuplicateGroup::new("abc", 500, vec![p("/only.mp4")]);
        assert_eq!(group.redundant_copies(), 0);
        assert_eq!(group.reclaimable_bytes(), 0);
    }

    // ── DedupStats ─────────────────────────────────────────────────────────────

    #[test]
    fn test_dedup_stats_ratio_and_savings() {
        let stats = DedupStats {
            total_files: 4,
            unique_files: 2,
            duplicate_files: 2,
            total_bytes: 4_000,
            unique_bytes: 2_000,
            reclaimable_bytes: 2_000,
            duplicate_groups: 2,
        };
        assert!((stats.dedup_ratio() - 2.0).abs() < 1e-9);
        assert!((stats.savings_pct() - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_dedup_stats_no_duplicates() {
        let stats = DedupStats {
            total_files: 3,
            unique_files: 3,
            duplicate_files: 0,
            total_bytes: 3_000,
            unique_bytes: 3_000,
            reclaimable_bytes: 0,
            duplicate_groups: 0,
        };
        assert!((stats.dedup_ratio() - 1.0).abs() < 1e-9);
        assert!(stats.savings_pct() < 1e-9);
    }

    // ── DedupAnalyzer ─────────────────────────────────────────────────────────

    #[test]
    fn test_analyzer_no_duplicates() {
        let mut analyzer = DedupAnalyzer::new();
        analyzer.add_file("fp1", 100, p("/a.mkv"));
        analyzer.add_file("fp2", 200, p("/b.mkv"));
        let report = analyzer.build_report();
        assert_eq!(report.stats.total_files, 2);
        assert_eq!(report.stats.duplicate_files, 0);
        assert!(report.duplicate_groups.is_empty());
    }

    #[test]
    fn test_analyzer_one_duplicate_group() {
        let mut analyzer = DedupAnalyzer::new();
        analyzer.add_file("same_fp", 1_000, p("/copy1.mkv"));
        analyzer.add_file("same_fp", 1_000, p("/copy2.mkv"));
        analyzer.add_file("unique", 500, p("/unique.mkv"));
        let report = analyzer.build_report();
        assert_eq!(report.stats.total_files, 3);
        assert_eq!(report.stats.unique_files, 2);
        assert_eq!(report.stats.duplicate_files, 1);
        assert_eq!(report.stats.duplicate_groups, 1);
        assert_eq!(report.stats.reclaimable_bytes, 1_000);
    }

    #[test]
    fn test_analyzer_multiple_duplicate_groups_sorted() {
        let mut analyzer = DedupAnalyzer::new();
        // group A: small file, 3 copies → 2 reclaimable × 100 = 200
        for i in 0..3 {
            analyzer.add_file("group_a", 100, p(&format!("/a_{i}.mkv")));
        }
        // group B: large file, 2 copies → 1 reclaimable × 10_000 = 10_000
        for i in 0..2 {
            analyzer.add_file("group_b", 10_000, p(&format!("/b_{i}.mkv")));
        }
        let report = analyzer.build_report();
        // Largest reclaimable group should be first
        assert_eq!(report.duplicate_groups[0].fingerprint, "group_b");
        assert_eq!(report.stats.reclaimable_bytes, 10_200);
    }

    // ── DedupReport::render ────────────────────────────────────────────────────

    #[test]
    fn test_render_text_contains_key_fields() {
        let mut analyzer = DedupAnalyzer::new();
        analyzer.add_file("fp", 500, p("/x.mkv"));
        analyzer.add_file("fp", 500, p("/y.mkv"));
        let report = analyzer.build_report();
        let text = report.render(ReportFormat::Text).expect("render");
        assert!(text.contains("Deduplication ratio"));
        assert!(text.contains("Space savings"));
        assert!(text.contains("Duplicate Groups"));
    }

    #[test]
    fn test_render_json_valid() {
        let mut analyzer = DedupAnalyzer::new();
        analyzer.add_file("fp", 100, p("/f.mkv"));
        let report = analyzer.build_report();
        let json = report.render(ReportFormat::Json).expect("render");
        let val: serde_json::Value = serde_json::from_str(&json).expect("valid json");
        assert!(val.get("stats").is_some());
    }

    #[test]
    fn test_render_csv_header_and_rows() {
        let mut analyzer = DedupAnalyzer::new();
        analyzer.add_file("fp1", 1_024, p("/a.mkv"));
        analyzer.add_file("fp1", 1_024, p("/b.mkv"));
        let report = analyzer.build_report();
        let csv = report.render(ReportFormat::Csv).expect("render");
        let lines: Vec<&str> = csv.lines().collect();
        assert!(lines[0].starts_with("fingerprint"));
        assert!(lines.len() >= 2); // header + at least one data row
        assert!(lines[1].contains("fp1"));
    }

    // ── write_to_file ─────────────────────────────────────────────────────────

    #[test]
    fn test_write_report_to_file() {
        let dir = std::env::temp_dir();
        let path = dir.join("oximedia_dedup_report_test.json");
        let _ = std::fs::remove_file(&path);

        let mut analyzer = DedupAnalyzer::new();
        analyzer.add_file("x", 42, p("/file.mkv"));
        let report = analyzer.build_report();
        report
            .write_to_file(&path, ReportFormat::Json)
            .expect("write");

        let content = std::fs::read_to_string(&path).expect("read");
        assert!(!content.is_empty());
        let _ = std::fs::remove_file(&path);
    }

    // ── build_report_from_records ─────────────────────────────────────────────

    #[test]
    fn test_build_report_from_records_convenience() {
        let records = vec![
            ("digest_a".to_string(), 256_u64, p("/a1.wav")),
            ("digest_a".to_string(), 256_u64, p("/a2.wav")),
            ("digest_b".to_string(), 512_u64, p("/b.wav")),
        ];
        let report = build_report_from_records(&records);
        assert_eq!(report.stats.total_files, 3);
        assert_eq!(report.stats.duplicate_groups, 1);
        assert_eq!(report.stats.reclaimable_bytes, 256);
    }

    // ── csv_escape ────────────────────────────────────────────────────────────

    #[test]
    fn test_csv_escape_plain_string() {
        assert_eq!(csv_escape("simple"), "simple");
    }

    #[test]
    fn test_csv_escape_comma() {
        let escaped = csv_escape("a,b");
        assert!(escaped.starts_with('"'));
        assert!(escaped.ends_with('"'));
    }
}
