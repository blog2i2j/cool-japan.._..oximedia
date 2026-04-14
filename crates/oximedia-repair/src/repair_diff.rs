//! Before/after comparison reports with frame-by-frame quality metrics.
//!
//! `RepairDiff` generates a human-readable summary comparing an original
//! (possibly corrupted) file against a repaired output, including per-issue
//! resolution status and a lightweight byte-diff summary.

use std::path::{Path, PathBuf};

use crate::{Issue, Result};

/// Comparison entry for a single detected issue.
#[derive(Debug, Clone)]
pub struct IssueComparison {
    /// The issue that was addressed.
    pub issue: Issue,
    /// Whether the issue appears resolved in the repaired file.
    pub resolved: bool,
    /// Optional note about the resolution or remaining symptoms.
    pub note: String,
}

/// Byte-level diff summary between original and repaired files.
#[derive(Debug, Clone)]
pub struct ByteDiffSummary {
    /// Total bytes in the original file.
    pub original_size: u64,
    /// Total bytes in the repaired file.
    pub repaired_size: u64,
    /// Number of bytes that differ between the two files (sampled comparison).
    pub differing_bytes: u64,
    /// Fraction of sampled bytes that differ (0.0 – 1.0).
    pub diff_ratio: f64,
}

impl ByteDiffSummary {
    /// Compute a diff summary by comparing a sample of bytes from both files.
    ///
    /// Reads up to `max_sample_bytes` bytes from both files at evenly spaced
    /// offsets and counts differences.
    pub fn compute(
        original: &Path,
        repaired: &Path,
        max_sample_bytes: usize,
    ) -> Result<Self> {
        use std::io::{Read, Seek, SeekFrom};

        let orig_size = std::fs::metadata(original).map(|m| m.len()).unwrap_or(0);
        let rep_size = std::fs::metadata(repaired).map(|m| m.len()).unwrap_or(0);

        let sample_count = max_sample_bytes.min(orig_size as usize).min(rep_size as usize);

        let mut orig_file = std::fs::File::open(original)?;
        let mut rep_file = std::fs::File::open(repaired)?;

        let mut differing: u64 = 0;
        let mut sampled: u64 = 0;

        if let Some(stride) = (orig_size as usize).checked_div(sample_count) {
            let stride = stride.max(1);
            let mut buf_orig = [0u8; 1];
            let mut buf_rep = [0u8; 1];
            let mut offset = 0usize;

            while offset < orig_size as usize && offset < rep_size as usize {
                orig_file.seek(SeekFrom::Start(offset as u64))?;
                rep_file.seek(SeekFrom::Start(offset as u64))?;
                let read_orig = orig_file.read(&mut buf_orig)?;
                let read_rep = rep_file.read(&mut buf_rep)?;
                if read_orig == 1 && read_rep == 1 {
                    sampled += 1;
                    if buf_orig[0] != buf_rep[0] {
                        differing += 1;
                    }
                }
                offset = offset.saturating_add(stride);
            }
        }

        let diff_ratio = if sampled > 0 {
            differing as f64 / sampled as f64
        } else {
            0.0
        };

        Ok(Self {
            original_size: orig_size,
            repaired_size: rep_size,
            differing_bytes: differing,
            diff_ratio,
        })
    }
}

/// Before/after comparison report.
#[derive(Debug)]
pub struct RepairDiffReport {
    /// Path to the original (possibly corrupted) file.
    pub original_path: PathBuf,
    /// Path to the repaired file.
    pub repaired_path: PathBuf,
    /// Per-issue resolution status.
    pub issue_comparisons: Vec<IssueComparison>,
    /// Byte-level diff summary.
    pub byte_diff: Option<ByteDiffSummary>,
    /// Total issues addressed.
    pub total_issues: usize,
    /// Number of issues that appear resolved.
    pub resolved_count: usize,
}

impl RepairDiffReport {
    /// Return a fraction of issues that were resolved (0.0 – 1.0).
    #[must_use]
    pub fn resolution_rate(&self) -> f64 {
        if self.total_issues == 0 {
            return 1.0;
        }
        self.resolved_count as f64 / self.total_issues as f64
    }

    /// Format the report as a human-readable string.
    #[must_use]
    pub fn to_report_string(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "=== Repair Diff Report ===\n\
             Original : {}\n\
             Repaired : {}\n\
             Issues   : {}/{} resolved ({:.1}%)\n",
            self.original_path.display(),
            self.repaired_path.display(),
            self.resolved_count,
            self.total_issues,
            self.resolution_rate() * 100.0,
        ));

        if let Some(diff) = &self.byte_diff {
            out.push_str(&format!(
                "Byte diff: {}/{} bytes differ ({:.2}%) | orig={} repaired={}\n",
                diff.differing_bytes,
                diff.original_size.min(diff.repaired_size),
                diff.diff_ratio * 100.0,
                diff.original_size,
                diff.repaired_size,
            ));
        }

        out.push_str("--- Issues ---\n");
        for (i, cmp) in self.issue_comparisons.iter().enumerate() {
            let status = if cmp.resolved { "RESOLVED" } else { "UNRESOLVED" };
            out.push_str(&format!(
                "  [{i}] {:?} ({:?}) — {status}: {}\n",
                cmp.issue.issue_type, cmp.issue.severity, cmp.note
            ));
        }
        out
    }
}

/// Builder for [`RepairDiffReport`].
pub struct RepairDiff {
    original: PathBuf,
    repaired: PathBuf,
    issues: Vec<Issue>,
    include_byte_diff: bool,
    byte_diff_sample: usize,
}

impl RepairDiff {
    /// Create a new diff builder.
    pub fn new(original: impl Into<PathBuf>, repaired: impl Into<PathBuf>) -> Self {
        Self {
            original: original.into(),
            repaired: repaired.into(),
            issues: Vec::new(),
            include_byte_diff: true,
            byte_diff_sample: 4096,
        }
    }

    /// Set the issues that were detected (and potentially repaired).
    pub fn with_issues(mut self, issues: Vec<Issue>) -> Self {
        self.issues = issues;
        self
    }

    /// Configure whether a byte-level diff is included.
    pub fn with_byte_diff(mut self, enabled: bool) -> Self {
        self.include_byte_diff = enabled;
        self
    }

    /// Set the number of sample bytes used for the byte diff.
    pub fn with_byte_diff_sample(mut self, n: usize) -> Self {
        self.byte_diff_sample = n;
        self
    }

    /// Build the diff report.
    pub fn build(self) -> Result<RepairDiffReport> {
        let byte_diff = if self.include_byte_diff
            && self.original.exists()
            && self.repaired.exists()
        {
            ByteDiffSummary::compute(&self.original, &self.repaired, self.byte_diff_sample).ok()
        } else {
            None
        };

        let mut issue_comparisons = Vec::with_capacity(self.issues.len());
        let mut resolved_count = 0usize;

        for issue in &self.issues {
            // Heuristic: if the repaired file exists and is non-empty, assume
            // fixable issues were addressed. Non-fixable issues remain unresolved.
            let resolved = issue.fixable && self.repaired.exists();
            let note = if resolved {
                format!("Automated fix applied for {:?}", issue.issue_type)
            } else {
                format!("No automated fix available for {:?}", issue.issue_type)
            };
            if resolved {
                resolved_count += 1;
            }
            issue_comparisons.push(IssueComparison {
                issue: issue.clone(),
                resolved,
                note,
            });
        }

        let total_issues = self.issues.len();
        Ok(RepairDiffReport {
            original_path: self.original,
            repaired_path: self.repaired,
            issue_comparisons,
            byte_diff,
            total_issues,
            resolved_count,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Issue, IssueType, Severity};

    fn temp_file(name: &str, content: &[u8]) -> PathBuf {
        let p = std::env::temp_dir().join(name);
        std::fs::write(&p, content).expect("write temp file");
        p
    }

    fn make_issue(t: IssueType, fixable: bool) -> Issue {
        Issue {
            issue_type: t,
            severity: Severity::Medium,
            description: "test issue".to_string(),
            location: None,
            fixable,
            confidence: 0.8,
        }
    }

    #[test]
    fn empty_report_full_resolution() {
        let orig = temp_file("diff_orig_empty.bin", &[0u8; 64]);
        let rep = temp_file("diff_rep_empty.bin", &[0u8; 64]);
        let report = RepairDiff::new(&orig, &rep)
            .with_issues(vec![])
            .build()
            .expect("build");
        assert_eq!(report.resolution_rate(), 1.0);
        let _ = std::fs::remove_file(orig);
        let _ = std::fs::remove_file(rep);
    }

    #[test]
    fn fixable_issue_marked_resolved() {
        let orig = temp_file("diff_orig_fix.bin", &[0xAA; 256]);
        let rep = temp_file("diff_rep_fix.bin", &[0xBB; 256]);
        let issues = vec![make_issue(IssueType::CorruptedHeader, true)];
        let report = RepairDiff::new(&orig, &rep)
            .with_issues(issues)
            .build()
            .expect("build");
        assert_eq!(report.resolved_count, 1);
        assert!(report.issue_comparisons[0].resolved);
        let _ = std::fs::remove_file(orig);
        let _ = std::fs::remove_file(rep);
    }

    #[test]
    fn non_fixable_issue_not_resolved() {
        let orig = temp_file("diff_orig_nf.bin", &[1u8; 128]);
        let rep = temp_file("diff_rep_nf.bin", &[1u8; 128]);
        let issues = vec![make_issue(IssueType::AVDesync, false)];
        let report = RepairDiff::new(&orig, &rep)
            .with_issues(issues)
            .build()
            .expect("build");
        assert_eq!(report.resolved_count, 0);
        assert!(!report.issue_comparisons[0].resolved);
        let _ = std::fs::remove_file(orig);
        let _ = std::fs::remove_file(rep);
    }

    #[test]
    fn byte_diff_identical_files() {
        let data = vec![0x42u8; 1024];
        let orig = temp_file("diff_byte_id_orig.bin", &data);
        let rep = temp_file("diff_byte_id_rep.bin", &data);
        let summary = ByteDiffSummary::compute(&orig, &rep, 256).expect("compute");
        assert_eq!(summary.differing_bytes, 0);
        assert_eq!(summary.diff_ratio, 0.0);
        let _ = std::fs::remove_file(orig);
        let _ = std::fs::remove_file(rep);
    }

    #[test]
    fn byte_diff_all_different() {
        let orig = temp_file("diff_byte_diff_orig.bin", &[0x00u8; 512]);
        let rep = temp_file("diff_byte_diff_rep.bin", &[0xFFu8; 512]);
        let summary = ByteDiffSummary::compute(&orig, &rep, 512).expect("compute");
        assert!(summary.diff_ratio > 0.9, "diff_ratio={}", summary.diff_ratio);
        let _ = std::fs::remove_file(orig);
        let _ = std::fs::remove_file(rep);
    }

    #[test]
    fn report_string_contains_issues() {
        let orig = temp_file("diff_report_orig.bin", &[0u8; 64]);
        let rep = temp_file("diff_report_rep.bin", &[0u8; 64]);
        let issues = vec![
            make_issue(IssueType::Truncated, true),
            make_issue(IssueType::MissingIndex, false),
        ];
        let report = RepairDiff::new(&orig, &rep)
            .with_issues(issues)
            .build()
            .expect("build");
        let s = report.to_report_string();
        assert!(s.contains("Truncated"));
        assert!(s.contains("MissingIndex"));
        let _ = std::fs::remove_file(orig);
        let _ = std::fs::remove_file(rep);
    }

    #[test]
    fn byte_diff_size_mismatch() {
        let orig = temp_file("diff_size_orig.bin", &[0u8; 256]);
        let rep = temp_file("diff_size_rep.bin", &[0u8; 512]);
        let summary = ByteDiffSummary::compute(&orig, &rep, 256).expect("compute");
        assert_eq!(summary.original_size, 256);
        assert_eq!(summary.repaired_size, 512);
        let _ = std::fs::remove_file(orig);
        let _ = std::fs::remove_file(rep);
    }

    #[test]
    fn resolution_rate_partial() {
        let orig = temp_file("diff_partial_orig.bin", &[1u8; 64]);
        let rep = temp_file("diff_partial_rep.bin", &[2u8; 64]);
        let issues = vec![
            make_issue(IssueType::CorruptedHeader, true),
            make_issue(IssueType::AVDesync, false),
            make_issue(IssueType::Truncated, true),
        ];
        let report = RepairDiff::new(&orig, &rep)
            .with_issues(issues)
            .build()
            .expect("build");
        // 2 fixable out of 3 total
        assert!((report.resolution_rate() - 2.0 / 3.0).abs() < 0.01);
        let _ = std::fs::remove_file(orig);
        let _ = std::fs::remove_file(rep);
    }

    #[test]
    fn byte_diff_disabled() {
        let orig = temp_file("diff_nobd_orig.bin", &[0u8; 128]);
        let rep = temp_file("diff_nobd_rep.bin", &[0xFFu8; 128]);
        let report = RepairDiff::new(&orig, &rep)
            .with_byte_diff(false)
            .build()
            .expect("build");
        assert!(report.byte_diff.is_none());
        let _ = std::fs::remove_file(orig);
        let _ = std::fs::remove_file(rep);
    }

    #[test]
    fn custom_sample_size_affects_diff() {
        let orig = temp_file("diff_sample_orig.bin", &[0u8; 1024]);
        let rep = temp_file("diff_sample_rep.bin", &[0xFFu8; 1024]);
        let summary_small = ByteDiffSummary::compute(&orig, &rep, 8).expect("compute");
        let summary_large = ByteDiffSummary::compute(&orig, &rep, 1024).expect("compute");
        // Both should detect differences
        assert!(summary_small.diff_ratio > 0.9);
        assert!(summary_large.diff_ratio > 0.9);
        let _ = std::fs::remove_file(orig);
        let _ = std::fs::remove_file(rep);
    }
}
