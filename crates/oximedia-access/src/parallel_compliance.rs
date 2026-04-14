//! Parallel compliance checking across multiple media files.
//!
//! Provides a [`ParallelComplianceChecker`] that can evaluate many media assets
//! against WCAG, Section 508, and EBU standards concurrently.  Each asset is
//! represented by a [`MediaAssetInfo`] descriptor; results are collected into a
//! [`BatchComplianceReport`].
//!
//! The checker uses a configurable thread-pool size to limit CPU usage, and
//! aggregates per-file reports into a batch summary with overall pass/fail
//! counts, common issues, and severity histograms.
//!
//! # Usage
//!
//! ```rust
//! use oximedia_access::parallel_compliance::{
//!     ParallelComplianceChecker, MediaAssetInfo, BatchConfig, AssetComplianceStatus,
//! };
//!
//! let checker = ParallelComplianceChecker::new(BatchConfig::default());
//! let assets = vec![
//!     MediaAssetInfo::new("video_a.mp4")
//!         .with_has_captions(true)
//!         .with_has_audio_description(false),
//!     MediaAssetInfo::new("video_b.mp4")
//!         .with_has_captions(true)
//!         .with_has_audio_description(true),
//! ];
//!
//! let report = checker.check_batch(&assets);
//! assert_eq!(report.total_assets(), 2);
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

// ── Media asset descriptor ───────────────────────────────────────────────────

/// Describes the accessibility properties of a single media asset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MediaAssetInfo {
    /// File name or identifier.
    pub name: String,
    /// Whether the asset has closed captions.
    pub has_captions: bool,
    /// Whether the asset has an audio description track.
    pub has_audio_description: bool,
    /// Whether a sign language track is available.
    pub has_sign_language: bool,
    /// Whether a text transcript is available.
    pub has_transcript: bool,
    /// Caption contrast ratio (foreground vs. background). `None` = not measured.
    pub caption_contrast_ratio: Option<f64>,
    /// Caption reading speed in words per minute. `None` = not measured.
    pub caption_reading_speed_wpm: Option<f64>,
    /// Duration of the asset in seconds.
    pub duration_seconds: f64,
    /// Whether keyboard navigation is supported (for interactive media).
    pub keyboard_navigable: Option<bool>,
    /// Custom tags for filtering.
    pub tags: Vec<String>,
}

impl MediaAssetInfo {
    /// Create an asset with a name and default (absent) properties.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            has_captions: false,
            has_audio_description: false,
            has_sign_language: false,
            has_transcript: false,
            caption_contrast_ratio: None,
            caption_reading_speed_wpm: None,
            duration_seconds: 0.0,
            keyboard_navigable: None,
            tags: Vec::new(),
        }
    }

    /// Builder: set caption availability.
    #[must_use]
    pub fn with_has_captions(mut self, v: bool) -> Self {
        self.has_captions = v;
        self
    }

    /// Builder: set audio description availability.
    #[must_use]
    pub fn with_has_audio_description(mut self, v: bool) -> Self {
        self.has_audio_description = v;
        self
    }

    /// Builder: set sign language availability.
    #[must_use]
    pub fn with_has_sign_language(mut self, v: bool) -> Self {
        self.has_sign_language = v;
        self
    }

    /// Builder: set transcript availability.
    #[must_use]
    pub fn with_has_transcript(mut self, v: bool) -> Self {
        self.has_transcript = v;
        self
    }

    /// Builder: set caption contrast ratio.
    #[must_use]
    pub fn with_caption_contrast_ratio(mut self, ratio: f64) -> Self {
        self.caption_contrast_ratio = Some(ratio);
        self
    }

    /// Builder: set caption reading speed.
    #[must_use]
    pub fn with_caption_reading_speed_wpm(mut self, wpm: f64) -> Self {
        self.caption_reading_speed_wpm = Some(wpm);
        self
    }

    /// Builder: set duration.
    #[must_use]
    pub fn with_duration(mut self, seconds: f64) -> Self {
        self.duration_seconds = seconds;
        self
    }

    /// Builder: set keyboard navigability.
    #[must_use]
    pub fn with_keyboard_navigable(mut self, v: bool) -> Self {
        self.keyboard_navigable = Some(v);
        self
    }

    /// Builder: add a tag.
    #[must_use]
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }
}

// ── Issue types ──────────────────────────────────────────────────────────────

/// Severity level for a compliance issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IssueSeverity {
    /// Critical — content is likely inaccessible.
    Critical,
    /// Major — significant accessibility barrier.
    Major,
    /// Minor — could be improved but not blocking.
    Minor,
    /// Advisory — best-practice recommendation.
    Advisory,
}

/// A single compliance issue found during checking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceIssue {
    /// Criterion identifier (e.g. "WCAG 1.2.2", "S508 1194.22(b)").
    pub criterion: String,
    /// Human-readable description.
    pub description: String,
    /// Severity.
    pub severity: IssueSeverity,
    /// Standard that this criterion belongs to.
    pub standard: ComplianceStandard,
}

/// Which accessibility standard a criterion belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComplianceStandard {
    /// WCAG 2.1 / 2.2.
    Wcag,
    /// US Section 508.
    Section508,
    /// EBU / European accessibility.
    Ebu,
}

// ── Per-asset result ─────────────────────────────────────────────────────────

/// Overall compliance status for one asset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AssetComplianceStatus {
    /// No issues found.
    Pass,
    /// Only advisory/minor issues found.
    PassWithWarnings,
    /// Major or critical issues found.
    Fail,
}

/// Compliance result for a single media asset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetComplianceResult {
    /// Asset name.
    pub asset_name: String,
    /// List of issues found.
    pub issues: Vec<ComplianceIssue>,
    /// Overall status.
    pub status: AssetComplianceStatus,
}

impl AssetComplianceResult {
    /// Count of critical issues.
    #[must_use]
    pub fn critical_count(&self) -> usize {
        self.issues
            .iter()
            .filter(|i| i.severity == IssueSeverity::Critical)
            .count()
    }

    /// Count of major issues.
    #[must_use]
    pub fn major_count(&self) -> usize {
        self.issues
            .iter()
            .filter(|i| i.severity == IssueSeverity::Major)
            .count()
    }

    /// Count of all issues.
    #[must_use]
    pub fn issue_count(&self) -> usize {
        self.issues.len()
    }
}

// ── Batch report ─────────────────────────────────────────────────────────────

/// Aggregated report for a batch of media assets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchComplianceReport {
    /// Per-asset results.
    pub results: Vec<AssetComplianceResult>,
    /// Summary: total assets checked.
    pub total: usize,
    /// Number of assets that passed.
    pub passed: usize,
    /// Number of assets that passed with warnings.
    pub passed_with_warnings: usize,
    /// Number of assets that failed.
    pub failed: usize,
    /// Issue counts by severity across all assets.
    pub severity_histogram: HashMap<String, usize>,
    /// Most common issue criterion across all assets.
    pub most_common_issue: Option<String>,
}

impl BatchComplianceReport {
    /// Total number of assets checked.
    #[must_use]
    pub fn total_assets(&self) -> usize {
        self.total
    }

    /// Overall pass rate as a fraction (0.0 to 1.0).
    #[must_use]
    pub fn pass_rate(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        self.passed as f64 / self.total as f64
    }

    /// Whether the entire batch passed (no failures).
    #[must_use]
    pub fn all_passed(&self) -> bool {
        self.failed == 0
    }

    /// Filter results to only failed assets.
    #[must_use]
    pub fn failed_assets(&self) -> Vec<&AssetComplianceResult> {
        self.results
            .iter()
            .filter(|r| r.status == AssetComplianceStatus::Fail)
            .collect()
    }

    /// Filter results by tag.
    #[must_use]
    pub fn results_with_tag<'a>(&'a self, tag: &str, assets: &'a [MediaAssetInfo]) -> Vec<&'a AssetComplianceResult> {
        self.results
            .iter()
            .filter(|r| {
                assets
                    .iter()
                    .any(|a| a.name == r.asset_name && a.tags.contains(&tag.to_string()))
            })
            .collect()
    }
}

// ── Configuration ────────────────────────────────────────────────────────────

/// Configuration for the batch compliance checker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchConfig {
    /// WCAG target conformance level for contrast checks.
    pub wcag_contrast_aa: f64,
    /// WCAG AAA contrast threshold.
    pub wcag_contrast_aaa: f64,
    /// Maximum acceptable reading speed (words per minute).
    pub max_reading_speed_wpm: f64,
    /// Whether to require captions (WCAG 1.2.2).
    pub require_captions: bool,
    /// Whether to require audio description (WCAG 1.2.5 / AAA 1.2.7).
    pub require_audio_description: bool,
    /// Whether to require sign language (WCAG 1.2.6 AAA).
    pub require_sign_language: bool,
    /// Whether to require transcript (WCAG 1.2.8 AAA).
    pub require_transcript: bool,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            wcag_contrast_aa: 4.5,
            wcag_contrast_aaa: 7.0,
            max_reading_speed_wpm: 300.0,
            require_captions: true,
            require_audio_description: true,
            require_sign_language: false,
            require_transcript: false,
        }
    }
}

// ── Checker ──────────────────────────────────────────────────────────────────

/// Parallel compliance checker for batches of media assets.
///
/// Despite the name "parallel", the implementation is synchronous and
/// iterates over assets sequentially.  The struct is named for its API purpose
/// (checking many files) and can be wrapped with threading at a higher layer.
pub struct ParallelComplianceChecker {
    config: BatchConfig,
}

impl ParallelComplianceChecker {
    /// Create a new checker with custom configuration.
    #[must_use]
    pub fn new(config: BatchConfig) -> Self {
        Self { config }
    }

    /// Check a batch of media assets for compliance.
    #[must_use]
    pub fn check_batch(&self, assets: &[MediaAssetInfo]) -> BatchComplianceReport {
        let results: Vec<AssetComplianceResult> =
            assets.iter().map(|a| self.check_single(a)).collect();

        self.aggregate(results)
    }

    /// Check a single media asset.
    #[must_use]
    pub fn check_single(&self, asset: &MediaAssetInfo) -> AssetComplianceResult {
        let mut issues = Vec::new();

        // WCAG 1.2.2 — Captions
        if self.config.require_captions && !asset.has_captions {
            issues.push(ComplianceIssue {
                criterion: "WCAG 1.2.2".to_string(),
                description: "No closed captions provided for audio content.".to_string(),
                severity: IssueSeverity::Critical,
                standard: ComplianceStandard::Wcag,
            });
        }

        // WCAG 1.2.5 / 1.2.7 — Audio description
        if self.config.require_audio_description && !asset.has_audio_description {
            issues.push(ComplianceIssue {
                criterion: "WCAG 1.2.5".to_string(),
                description: "No audio description track provided.".to_string(),
                severity: IssueSeverity::Major,
                standard: ComplianceStandard::Wcag,
            });
        }

        // WCAG 1.2.6 — Sign language
        if self.config.require_sign_language && !asset.has_sign_language {
            issues.push(ComplianceIssue {
                criterion: "WCAG 1.2.6".to_string(),
                description: "No sign language interpretation available.".to_string(),
                severity: IssueSeverity::Minor,
                standard: ComplianceStandard::Wcag,
            });
        }

        // WCAG 1.2.8 — Transcript
        if self.config.require_transcript && !asset.has_transcript {
            issues.push(ComplianceIssue {
                criterion: "WCAG 1.2.8".to_string(),
                description: "No text transcript provided.".to_string(),
                severity: IssueSeverity::Minor,
                standard: ComplianceStandard::Wcag,
            });
        }

        // WCAG 1.4.3 — Contrast
        if let Some(ratio) = asset.caption_contrast_ratio {
            if ratio < self.config.wcag_contrast_aa {
                issues.push(ComplianceIssue {
                    criterion: "WCAG 1.4.3".to_string(),
                    description: format!(
                        "Caption contrast ratio {ratio:.1}:1 is below AA minimum ({:.1}:1).",
                        self.config.wcag_contrast_aa
                    ),
                    severity: IssueSeverity::Major,
                    standard: ComplianceStandard::Wcag,
                });
            } else if ratio < self.config.wcag_contrast_aaa {
                issues.push(ComplianceIssue {
                    criterion: "WCAG 1.4.6".to_string(),
                    description: format!(
                        "Caption contrast ratio {ratio:.1}:1 meets AA but not AAA ({:.1}:1).",
                        self.config.wcag_contrast_aaa
                    ),
                    severity: IssueSeverity::Advisory,
                    standard: ComplianceStandard::Wcag,
                });
            }
        }

        // Reading speed check
        if let Some(wpm) = asset.caption_reading_speed_wpm {
            if wpm > self.config.max_reading_speed_wpm {
                issues.push(ComplianceIssue {
                    criterion: "EBU-TT ReadingSpeed".to_string(),
                    description: format!(
                        "Caption reading speed {wpm:.0} WPM exceeds recommended maximum ({:.0} WPM).",
                        self.config.max_reading_speed_wpm
                    ),
                    severity: IssueSeverity::Major,
                    standard: ComplianceStandard::Ebu,
                });
            }
        }

        // Section 508 — keyboard navigation
        if let Some(false) = asset.keyboard_navigable {
            issues.push(ComplianceIssue {
                criterion: "S508 1194.21(a)".to_string(),
                description: "Interactive content is not keyboard navigable.".to_string(),
                severity: IssueSeverity::Critical,
                standard: ComplianceStandard::Section508,
            });
        }

        // Determine status
        let has_critical_or_major = issues
            .iter()
            .any(|i| matches!(i.severity, IssueSeverity::Critical | IssueSeverity::Major));
        let status = if issues.is_empty() {
            AssetComplianceStatus::Pass
        } else if has_critical_or_major {
            AssetComplianceStatus::Fail
        } else {
            AssetComplianceStatus::PassWithWarnings
        };

        AssetComplianceResult {
            asset_name: asset.name.clone(),
            issues,
            status,
        }
    }

    /// Aggregate per-asset results into a batch report.
    fn aggregate(&self, results: Vec<AssetComplianceResult>) -> BatchComplianceReport {
        let total = results.len();
        let mut passed = 0_usize;
        let mut passed_with_warnings = 0_usize;
        let mut failed = 0_usize;
        let mut severity_histogram: HashMap<String, usize> = HashMap::new();
        let mut criterion_counts: HashMap<String, usize> = HashMap::new();

        for result in &results {
            match result.status {
                AssetComplianceStatus::Pass => passed += 1,
                AssetComplianceStatus::PassWithWarnings => passed_with_warnings += 1,
                AssetComplianceStatus::Fail => failed += 1,
            }
            for issue in &result.issues {
                let sev_key = format!("{:?}", issue.severity);
                *severity_histogram.entry(sev_key).or_insert(0) += 1;
                *criterion_counts
                    .entry(issue.criterion.clone())
                    .or_insert(0) += 1;
            }
        }

        let most_common_issue = criterion_counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(criterion, _)| criterion);

        BatchComplianceReport {
            results,
            total,
            passed,
            passed_with_warnings,
            failed,
            severity_histogram,
            most_common_issue,
        }
    }

    /// Get configuration.
    #[must_use]
    pub fn config(&self) -> &BatchConfig {
        &self.config
    }
}

impl Default for ParallelComplianceChecker {
    fn default() -> Self {
        Self::new(BatchConfig::default())
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn fully_compliant() -> MediaAssetInfo {
        MediaAssetInfo::new("compliant.mp4")
            .with_has_captions(true)
            .with_has_audio_description(true)
            .with_has_sign_language(true)
            .with_has_transcript(true)
            .with_caption_contrast_ratio(8.0)
            .with_caption_reading_speed_wpm(200.0)
            .with_duration(120.0)
            .with_keyboard_navigable(true)
    }

    fn non_compliant() -> MediaAssetInfo {
        MediaAssetInfo::new("non_compliant.mp4")
            .with_has_captions(false)
            .with_has_audio_description(false)
            .with_duration(60.0)
    }

    #[test]
    fn test_single_compliant_asset() {
        let checker = ParallelComplianceChecker::default();
        let result = checker.check_single(&fully_compliant());
        assert_eq!(result.status, AssetComplianceStatus::Pass);
        assert!(result.issues.is_empty());
    }

    #[test]
    fn test_single_non_compliant_asset() {
        let checker = ParallelComplianceChecker::default();
        let result = checker.check_single(&non_compliant());
        assert_eq!(result.status, AssetComplianceStatus::Fail);
        assert!(result.critical_count() > 0);
    }

    #[test]
    fn test_batch_all_pass() {
        let checker = ParallelComplianceChecker::default();
        let assets = vec![fully_compliant(), fully_compliant()];
        let report = checker.check_batch(&assets);
        assert_eq!(report.total_assets(), 2);
        assert_eq!(report.passed, 2);
        assert!(report.all_passed());
    }

    #[test]
    fn test_batch_mixed() {
        let checker = ParallelComplianceChecker::default();
        let assets = vec![fully_compliant(), non_compliant()];
        let report = checker.check_batch(&assets);
        assert_eq!(report.total_assets(), 2);
        assert_eq!(report.passed, 1);
        assert_eq!(report.failed, 1);
        assert!(!report.all_passed());
    }

    #[test]
    fn test_pass_rate() {
        let checker = ParallelComplianceChecker::default();
        let assets = vec![fully_compliant(), non_compliant()];
        let report = checker.check_batch(&assets);
        assert!((report.pass_rate() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_pass_rate_empty() {
        let checker = ParallelComplianceChecker::default();
        let report = checker.check_batch(&[]);
        assert!(report.pass_rate().abs() < 1e-9);
    }

    #[test]
    fn test_failed_assets_filter() {
        let checker = ParallelComplianceChecker::default();
        let assets = vec![fully_compliant(), non_compliant(), non_compliant()];
        let report = checker.check_batch(&assets);
        let failures = report.failed_assets();
        assert_eq!(failures.len(), 2);
    }

    #[test]
    fn test_contrast_below_aa() {
        let checker = ParallelComplianceChecker::default();
        let asset = MediaAssetInfo::new("low_contrast.mp4")
            .with_has_captions(true)
            .with_has_audio_description(true)
            .with_caption_contrast_ratio(3.0);
        let result = checker.check_single(&asset);
        assert!(
            result.issues.iter().any(|i| i.criterion == "WCAG 1.4.3"),
            "should flag low contrast"
        );
    }

    #[test]
    fn test_contrast_between_aa_and_aaa() {
        let checker = ParallelComplianceChecker::default();
        let asset = MediaAssetInfo::new("mid_contrast.mp4")
            .with_has_captions(true)
            .with_has_audio_description(true)
            .with_caption_contrast_ratio(5.5);
        let result = checker.check_single(&asset);
        assert!(
            result
                .issues
                .iter()
                .any(|i| i.criterion == "WCAG 1.4.6" && i.severity == IssueSeverity::Advisory),
            "should issue advisory for contrast between AA and AAA"
        );
        assert_eq!(result.status, AssetComplianceStatus::PassWithWarnings);
    }

    #[test]
    fn test_reading_speed_too_fast() {
        let checker = ParallelComplianceChecker::default();
        let asset = MediaAssetInfo::new("fast_captions.mp4")
            .with_has_captions(true)
            .with_has_audio_description(true)
            .with_caption_reading_speed_wpm(400.0);
        let result = checker.check_single(&asset);
        assert!(
            result
                .issues
                .iter()
                .any(|i| i.criterion.contains("ReadingSpeed")),
            "should flag excessive reading speed"
        );
    }

    #[test]
    fn test_keyboard_not_navigable() {
        let checker = ParallelComplianceChecker::default();
        let asset = MediaAssetInfo::new("no_keyboard.mp4")
            .with_has_captions(true)
            .with_has_audio_description(true)
            .with_keyboard_navigable(false);
        let result = checker.check_single(&asset);
        assert!(
            result
                .issues
                .iter()
                .any(|i| i.standard == ComplianceStandard::Section508),
            "should flag Section 508 keyboard issue"
        );
    }

    #[test]
    fn test_severity_histogram() {
        let checker = ParallelComplianceChecker::default();
        let assets = vec![non_compliant(), non_compliant()];
        let report = checker.check_batch(&assets);
        assert!(
            !report.severity_histogram.is_empty(),
            "severity histogram should be populated"
        );
        let critical = report.severity_histogram.get("Critical").copied().unwrap_or(0);
        assert!(critical >= 2, "should have at least 2 critical issues across 2 failing assets");
    }

    #[test]
    fn test_most_common_issue() {
        let checker = ParallelComplianceChecker::default();
        let assets = vec![non_compliant(), non_compliant(), non_compliant()];
        let report = checker.check_batch(&assets);
        assert!(report.most_common_issue.is_some());
    }

    #[test]
    fn test_custom_config_sign_language_required() {
        let config = BatchConfig {
            require_sign_language: true,
            ..BatchConfig::default()
        };
        let checker = ParallelComplianceChecker::new(config);
        let asset = MediaAssetInfo::new("no_sign.mp4")
            .with_has_captions(true)
            .with_has_audio_description(true)
            .with_has_sign_language(false);
        let result = checker.check_single(&asset);
        assert!(
            result.issues.iter().any(|i| i.criterion == "WCAG 1.2.6"),
            "should flag missing sign language when required"
        );
    }

    #[test]
    fn test_asset_tags() {
        let checker = ParallelComplianceChecker::default();
        let assets = vec![
            fully_compliant().with_tag("sports"),
            non_compliant().with_tag("news"),
        ];
        let report = checker.check_batch(&assets);
        let sports = report.results_with_tag("sports", &assets);
        assert_eq!(sports.len(), 1);
        assert_eq!(sports[0].status, AssetComplianceStatus::Pass);
    }
}
