//! Structured metadata report generator for Dolby Vision RPU QC workflows.
//!
//! [`MetadataReporter`] analyses a collection of [`DolbyVisionRpu`] frames and
//! produces a [`MetadataReport`] that summarises:
//! - Which metadata levels are present and their coverage fraction.
//! - Luminance (PQ) statistics across Level 1 frames.
//! - Trim target inventory from Level 2 metadata.
//! - Per-profile compliance issues.
//!
//! # Example
//!
//! ```rust
//! use oximedia_dolbyvision::{DolbyVisionRpu, Profile, Level1Metadata};
//! use oximedia_dolbyvision::metadata_report::MetadataReporter;
//!
//! let mut rpus = Vec::new();
//! let mut rpu = DolbyVisionRpu::new(Profile::Profile8);
//! rpu.level1 = Some(Level1Metadata { min_pq: 0, max_pq: 3500, avg_pq: 1200 });
//! rpus.push(rpu);
//!
//! let report = MetadataReporter::generate(&rpus);
//! assert_eq!(report.total_frames, 1);
//! assert!(report.level1_coverage > 0.99);
//! ```

use crate::{DolbyVisionRpu, Profile};
use std::collections::HashMap;

// ── LevelCoverage ─────────────────────────────────────────────────────────────

/// Coverage information for a single metadata level.
#[derive(Debug, Clone)]
pub struct LevelCoverage {
    /// Metadata level number (1, 2, 5, 6, 8, 9, 11, …).
    pub level: u8,
    /// Number of frames that contain this level.
    pub frame_count: u64,
    /// Fraction of total frames that contain this level (`[0.0, 1.0]`).
    pub coverage_fraction: f64,
}

// ── PqSummary ─────────────────────────────────────────────────────────────────

/// Summary of PQ values gathered from Level 1 metadata.
#[derive(Debug, Clone, Default)]
pub struct PqSummary {
    /// Global minimum PQ across all Level-1 frames.
    pub global_min_pq: u16,
    /// Global maximum PQ across all Level-1 frames.
    pub global_max_pq: u16,
    /// Arithmetic mean of the per-frame average PQ.
    pub mean_avg_pq: f64,
    /// Arithmetic mean of the per-frame maximum PQ.
    pub mean_max_pq: f64,
    /// Number of frames whose Level 1 max-PQ exceeds 2000 (≈ 203 nits on the
    /// PQ curve — a proxy for "HDR-heavy" content).
    pub hdr_heavy_frames: u64,
}

impl PqSummary {
    /// Approximate peak luminance in nits corresponding to `global_max_pq`.
    ///
    /// Uses the simplified ST.2084 inverse: `(pq / 4095)^(1/0.1593) × 10 000`.
    #[must_use]
    pub fn estimated_peak_nits(&self) -> f32 {
        let pq_norm = f64::from(self.global_max_pq) / 4095.0;
        (pq_norm.powf(1.0 / 0.1593) * 10_000.0) as f32
    }
}

// ── ComplianceIssue ───────────────────────────────────────────────────────────

/// A single compliance issue detected during report generation.
#[derive(Debug, Clone)]
pub struct ComplianceIssue {
    /// Short machine-readable code, e.g. `"MISSING_LEVEL1"`.
    pub code: String,
    /// Human-readable description.
    pub description: String,
    /// Severity level.
    pub severity: IssueSeverity,
}

/// Severity of a compliance issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum IssueSeverity {
    /// Informational — does not affect playback.
    Info,
    /// Warning — may cause sub-optimal display mapping.
    Warning,
    /// Error — likely to cause playback failures or incorrect display.
    Error,
}

impl std::fmt::Display for IssueSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Info => write!(f, "INFO"),
            Self::Warning => write!(f, "WARNING"),
            Self::Error => write!(f, "ERROR"),
        }
    }
}

// ── TrimInventory ─────────────────────────────────────────────────────────────

/// Inventory of Level 2 trim-pass target display indices found in the stream.
#[derive(Debug, Clone, Default)]
pub struct TrimInventory {
    /// Map from `target_display_index` to the count of frames carrying that
    /// trim target.
    pub targets: HashMap<u8, u64>,
}

impl TrimInventory {
    /// Number of distinct trim targets in the stream.
    #[must_use]
    pub fn target_count(&self) -> usize {
        self.targets.len()
    }

    /// Return the target display index that appears most often.
    ///
    /// Returns `None` when no trim data is present.
    #[must_use]
    pub fn most_common_target(&self) -> Option<u8> {
        self.targets
            .iter()
            .max_by_key(|(_, &count)| count)
            .map(|(&idx, _)| idx)
    }
}

// ── MetadataReport ────────────────────────────────────────────────────────────

/// Comprehensive quality-control report for a sequence of Dolby Vision RPU
/// frames.
#[derive(Debug, Clone)]
pub struct MetadataReport {
    /// Total number of RPU frames analysed.
    pub total_frames: u64,
    /// Fraction of frames containing Level 1 metadata (`[0.0, 1.0]`).
    pub level1_coverage: f64,
    /// Fraction of frames containing Level 2 metadata (`[0.0, 1.0]`).
    pub level2_coverage: f64,
    /// Fraction of frames containing Level 5 metadata (`[0.0, 1.0]`).
    pub level5_coverage: f64,
    /// Fraction of frames containing Level 6 metadata (`[0.0, 1.0]`).
    pub level6_coverage: f64,
    /// Per-level coverage details.
    pub level_coverage: Vec<LevelCoverage>,
    /// PQ statistics from Level 1 metadata (empty when no Level 1 present).
    pub pq_summary: Option<PqSummary>,
    /// Level 2 trim target inventory.
    pub trim_inventory: TrimInventory,
    /// Dolby Vision profile observed in the first frame.
    pub observed_profile: Option<Profile>,
    /// Compliance issues detected during analysis.
    pub issues: Vec<ComplianceIssue>,
}

impl MetadataReport {
    /// Returns `true` when no `Error`-severity issues are present.
    #[must_use]
    pub fn is_compliant(&self) -> bool {
        !self
            .issues
            .iter()
            .any(|i| i.severity == IssueSeverity::Error)
    }

    /// Number of issues at or above `severity`.
    #[must_use]
    pub fn issue_count_at_or_above(&self, severity: IssueSeverity) -> usize {
        self.issues
            .iter()
            .filter(|i| i.severity >= severity)
            .count()
    }

    /// Returns `true` when every frame carries Level 1 metadata.
    #[must_use]
    pub fn has_complete_level1(&self) -> bool {
        self.total_frames > 0 && (self.level1_coverage - 1.0_f64).abs() < f64::EPSILON
    }

    /// Format a concise one-line summary.
    #[must_use]
    pub fn summary_line(&self) -> String {
        format!(
            "frames={} L1={:.1}% L2={:.1}% issues={} compliant={}",
            self.total_frames,
            self.level1_coverage * 100.0,
            self.level2_coverage * 100.0,
            self.issues.len(),
            self.is_compliant(),
        )
    }
}

// ── MetadataReporter ──────────────────────────────────────────────────────────

/// Analyses a slice of [`DolbyVisionRpu`] frames and produces a
/// [`MetadataReport`].
pub struct MetadataReporter;

impl MetadataReporter {
    /// Generate a report for the given sequence of RPU frames.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn generate(rpus: &[DolbyVisionRpu]) -> MetadataReport {
        let total = rpus.len() as u64;

        // Per-level counters
        let mut l1_count: u64 = 0;
        let mut l2_count: u64 = 0;
        let mut l5_count: u64 = 0;
        let mut l6_count: u64 = 0;
        let mut l8_count: u64 = 0;
        let mut l9_count: u64 = 0;
        let mut l11_count: u64 = 0;

        // PQ accumulators
        let mut pq_min = u16::MAX;
        let mut pq_max: u16 = 0;
        let mut sum_avg_pq: f64 = 0.0;
        let mut sum_max_pq: f64 = 0.0;
        let mut hdr_heavy: u64 = 0;

        // Trim inventory
        let mut trim_targets: HashMap<u8, u64> = HashMap::new();

        // Profile observation
        let observed_profile = rpus.first().map(|r| r.profile);

        for rpu in rpus {
            if let Some(ref l1) = rpu.level1 {
                l1_count += 1;
                if l1.min_pq < pq_min {
                    pq_min = l1.min_pq;
                }
                if l1.max_pq > pq_max {
                    pq_max = l1.max_pq;
                }
                sum_avg_pq += f64::from(l1.avg_pq);
                sum_max_pq += f64::from(l1.max_pq);
                if l1.max_pq > 2000 {
                    hdr_heavy += 1;
                }
            }
            if let Some(ref l2) = rpu.level2 {
                l2_count += 1;
                *trim_targets
                    .entry(l2.target_display_index)
                    .or_insert(0) += 1;
            }
            if rpu.level5.is_some() {
                l5_count += 1;
            }
            if rpu.level6.is_some() {
                l6_count += 1;
            }
            if rpu.level8.is_some() {
                l8_count += 1;
            }
            if rpu.level9.is_some() {
                l9_count += 1;
            }
            if rpu.level11.is_some() {
                l11_count += 1;
            }
        }

        // PQ summary
        let pq_summary = if l1_count > 0 {
            Some(PqSummary {
                global_min_pq: pq_min,
                global_max_pq: pq_max,
                mean_avg_pq: sum_avg_pq / l1_count as f64,
                mean_max_pq: sum_max_pq / l1_count as f64,
                hdr_heavy_frames: hdr_heavy,
            })
        } else {
            None
        };

        // Coverage fractions
        let frac = |n: u64| -> f64 {
            if total == 0 {
                0.0
            } else {
                n as f64 / total as f64
            }
        };

        let level_coverage = vec![
            LevelCoverage {
                level: 1,
                frame_count: l1_count,
                coverage_fraction: frac(l1_count),
            },
            LevelCoverage {
                level: 2,
                frame_count: l2_count,
                coverage_fraction: frac(l2_count),
            },
            LevelCoverage {
                level: 5,
                frame_count: l5_count,
                coverage_fraction: frac(l5_count),
            },
            LevelCoverage {
                level: 6,
                frame_count: l6_count,
                coverage_fraction: frac(l6_count),
            },
            LevelCoverage {
                level: 8,
                frame_count: l8_count,
                coverage_fraction: frac(l8_count),
            },
            LevelCoverage {
                level: 9,
                frame_count: l9_count,
                coverage_fraction: frac(l9_count),
            },
            LevelCoverage {
                level: 11,
                frame_count: l11_count,
                coverage_fraction: frac(l11_count),
            },
        ];

        // Compliance checks
        let mut issues = Vec::new();
        Self::check_compliance(
            total,
            l1_count,
            l6_count,
            &observed_profile,
            &mut issues,
        );

        MetadataReport {
            total_frames: total,
            level1_coverage: frac(l1_count),
            level2_coverage: frac(l2_count),
            level5_coverage: frac(l5_count),
            level6_coverage: frac(l6_count),
            level_coverage,
            pq_summary,
            trim_inventory: TrimInventory {
                targets: trim_targets,
            },
            observed_profile,
            issues,
        }
    }

    fn check_compliance(
        total: u64,
        l1_count: u64,
        l6_count: u64,
        profile: &Option<Profile>,
        issues: &mut Vec<ComplianceIssue>,
    ) {
        if total == 0 {
            issues.push(ComplianceIssue {
                code: "EMPTY_STREAM".to_string(),
                description: "No RPU frames provided for analysis.".to_string(),
                severity: IssueSeverity::Warning,
            });
            return;
        }

        // Level 1 is required for all profiles
        if l1_count == 0 {
            issues.push(ComplianceIssue {
                code: "MISSING_LEVEL1".to_string(),
                description: "No frames contain Level 1 (min/max/avg PQ) metadata.".to_string(),
                severity: IssueSeverity::Error,
            });
        } else if l1_count < total {
            let missing = total - l1_count;
            issues.push(ComplianceIssue {
                code: "INCOMPLETE_LEVEL1".to_string(),
                description: format!(
                    "Level 1 metadata missing from {missing} of {total} frames."
                ),
                severity: IssueSeverity::Warning,
            });
        }

        // Level 6 (MaxCLL/MaxFALL) strongly recommended for HDR profiles
        if l6_count == 0 {
            if let Some(profile) = profile {
                let needs_l6 = matches!(
                    profile,
                    Profile::Profile5 | Profile::Profile7 | Profile::Profile8 | Profile::Profile8_1
                );
                if needs_l6 {
                    issues.push(ComplianceIssue {
                        code: "MISSING_LEVEL6".to_string(),
                        description: "Level 6 (MaxCLL/MaxFALL) metadata is absent. This is \
                                      required for HDR10 backward compatibility."
                            .to_string(),
                        severity: IssueSeverity::Warning,
                    });
                }
            }
        }
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Level1Metadata, Level2Metadata};

    fn make_rpu(profile: Profile) -> DolbyVisionRpu {
        DolbyVisionRpu::new(profile)
    }

    fn with_level1(mut rpu: DolbyVisionRpu, min: u16, max: u16, avg: u16) -> DolbyVisionRpu {
        rpu.level1 = Some(Level1Metadata {
            min_pq: min,
            max_pq: max,
            avg_pq: avg,
        });
        rpu
    }

    fn with_level2(mut rpu: DolbyVisionRpu, target_idx: u8) -> DolbyVisionRpu {
        rpu.level2 = Some(Level2Metadata {
            target_display_index: target_idx,
            ..Default::default()
        });
        rpu
    }

    // ── Basic report generation ───────────────────────────────────────────────

    #[test]
    fn test_empty_report() {
        let report = MetadataReporter::generate(&[]);
        assert_eq!(report.total_frames, 0);
        assert!((report.level1_coverage - 0.0).abs() < f64::EPSILON);
        assert!(report.pq_summary.is_none());
    }

    #[test]
    fn test_single_frame_no_level1() {
        let rpus = vec![make_rpu(Profile::Profile8)];
        let report = MetadataReporter::generate(&rpus);
        assert_eq!(report.total_frames, 1);
        assert!((report.level1_coverage - 0.0).abs() < f64::EPSILON);
        assert!(report.pq_summary.is_none());
    }

    #[test]
    fn test_single_frame_with_level1() {
        let rpu = with_level1(make_rpu(Profile::Profile8), 10, 3500, 1200);
        let report = MetadataReporter::generate(&[rpu]);
        assert_eq!(report.total_frames, 1);
        assert!((report.level1_coverage - 1.0).abs() < f64::EPSILON);
        let pq = report.pq_summary.as_ref().unwrap();
        assert_eq!(pq.global_min_pq, 10);
        assert_eq!(pq.global_max_pq, 3500);
    }

    #[test]
    fn test_complete_level1_detection() {
        let rpus = vec![
            with_level1(make_rpu(Profile::Profile8), 0, 2000, 800),
            with_level1(make_rpu(Profile::Profile8), 0, 3000, 1200),
        ];
        let report = MetadataReporter::generate(&rpus);
        assert!(report.has_complete_level1());
    }

    #[test]
    fn test_partial_level1_coverage() {
        let rpus = vec![
            with_level1(make_rpu(Profile::Profile8), 0, 2000, 800),
            make_rpu(Profile::Profile8), // no level1
        ];
        let report = MetadataReporter::generate(&rpus);
        assert!((report.level1_coverage - 0.5).abs() < f64::EPSILON);
        assert!(!report.has_complete_level1());
    }

    // ── PQ summary ────────────────────────────────────────────────────────────

    #[test]
    fn test_pq_summary_global_max_min() {
        let rpus = vec![
            with_level1(make_rpu(Profile::Profile8), 50, 1000, 500),
            with_level1(make_rpu(Profile::Profile8), 10, 3500, 1500),
        ];
        let report = MetadataReporter::generate(&rpus);
        let pq = report.pq_summary.unwrap();
        assert_eq!(pq.global_min_pq, 10);
        assert_eq!(pq.global_max_pq, 3500);
    }

    #[test]
    fn test_pq_summary_mean_avg() {
        let rpus = vec![
            with_level1(make_rpu(Profile::Profile8), 0, 2000, 1000),
            with_level1(make_rpu(Profile::Profile8), 0, 2000, 2000),
        ];
        let report = MetadataReporter::generate(&rpus);
        let pq = report.pq_summary.unwrap();
        assert!((pq.mean_avg_pq - 1500.0).abs() < 1.0);
    }

    #[test]
    fn test_pq_summary_hdr_heavy_count() {
        let rpus = vec![
            with_level1(make_rpu(Profile::Profile8), 0, 2500, 1200), // > 2000 → HDR heavy
            with_level1(make_rpu(Profile::Profile8), 0, 1800, 900),  // ≤ 2000 → not
        ];
        let report = MetadataReporter::generate(&rpus);
        let pq = report.pq_summary.unwrap();
        assert_eq!(pq.hdr_heavy_frames, 1);
    }

    #[test]
    fn test_estimated_peak_nits_reasonable() {
        let pq = PqSummary {
            global_max_pq: 3079, // ≈ 1000 nits on ST.2084
            global_min_pq: 0,
            mean_avg_pq: 1000.0,
            mean_max_pq: 3000.0,
            hdr_heavy_frames: 0,
        };
        let nits = pq.estimated_peak_nits();
        // Should be in ballpark of 1000 nits (rough conversion)
        assert!(nits > 100.0 && nits < 10_000.0, "nits={nits}");
    }

    // ── Trim inventory ────────────────────────────────────────────────────────

    #[test]
    fn test_trim_inventory_target_count() {
        let rpus = vec![
            with_level2(make_rpu(Profile::Profile8), 1),
            with_level2(make_rpu(Profile::Profile8), 2),
            with_level2(make_rpu(Profile::Profile8), 1),
        ];
        let report = MetadataReporter::generate(&rpus);
        assert_eq!(report.trim_inventory.target_count(), 2);
    }

    #[test]
    fn test_trim_inventory_most_common() {
        let rpus = vec![
            with_level2(make_rpu(Profile::Profile8), 5),
            with_level2(make_rpu(Profile::Profile8), 5),
            with_level2(make_rpu(Profile::Profile8), 3),
        ];
        let report = MetadataReporter::generate(&rpus);
        assert_eq!(report.trim_inventory.most_common_target(), Some(5));
    }

    #[test]
    fn test_trim_inventory_empty_returns_none() {
        let inv = TrimInventory::default();
        assert!(inv.most_common_target().is_none());
    }

    // ── Compliance issues ─────────────────────────────────────────────────────

    #[test]
    fn test_compliance_error_missing_level1() {
        let rpus = vec![make_rpu(Profile::Profile8)];
        let report = MetadataReporter::generate(&rpus);
        let has_missing_l1 = report
            .issues
            .iter()
            .any(|i| i.code == "MISSING_LEVEL1" && i.severity == IssueSeverity::Error);
        assert!(has_missing_l1, "expected MISSING_LEVEL1 error");
        assert!(!report.is_compliant());
    }

    #[test]
    fn test_compliance_warning_missing_level6() {
        // Profile8 with Level1 but no Level6
        let rpu = with_level1(make_rpu(Profile::Profile8), 0, 2000, 1000);
        let report = MetadataReporter::generate(&[rpu]);
        let has_l6_warn = report
            .issues
            .iter()
            .any(|i| i.code == "MISSING_LEVEL6" && i.severity == IssueSeverity::Warning);
        assert!(has_l6_warn, "expected MISSING_LEVEL6 warning");
    }

    #[test]
    fn test_compliance_empty_stream_warning() {
        let report = MetadataReporter::generate(&[]);
        let has_empty = report
            .issues
            .iter()
            .any(|i| i.code == "EMPTY_STREAM");
        assert!(has_empty);
    }

    #[test]
    fn test_issue_count_at_or_above() {
        let rpus = vec![make_rpu(Profile::Profile8)];
        let report = MetadataReporter::generate(&rpus);
        let errors = report.issue_count_at_or_above(IssueSeverity::Error);
        assert!(errors > 0);
    }

    #[test]
    fn test_summary_line_format() {
        let rpu = with_level1(make_rpu(Profile::Profile8), 0, 2000, 1000);
        let report = MetadataReporter::generate(&[rpu]);
        let line = report.summary_line();
        assert!(line.contains("frames=1"), "line={line}");
        assert!(line.contains("compliant="), "line={line}");
    }

    #[test]
    fn test_issue_severity_ordering() {
        assert!(IssueSeverity::Info < IssueSeverity::Warning);
        assert!(IssueSeverity::Warning < IssueSeverity::Error);
    }
}
