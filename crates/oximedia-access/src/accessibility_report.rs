//! Accessibility compliance report — presence/absence checklist, gap analysis, and score calculation.
//!
//! This module provides a structured, content-level accessibility report for a
//! single media asset.  It aggregates evidence from the various accessibility
//! layers (captions, audio description, sign language, chapters, etc.) and
//! derives:
//!
//! - A per-criterion checklist (present / absent / partial / not-applicable)
//! - A weighted compliance score in `[0.0, 100.0]`
//! - A conformance level (None / A / AA / AAA) modelled on WCAG 2.x
//! - Human-readable recommendations for each failing criterion

use std::collections::BTreeMap;
use std::fmt;

use crate::{AccessError, AccessResult};

// ─── Criterion ────────────────────────────────────────────────────────────────

/// Accessibility criterion identifier.
///
/// Values map loosely to WCAG 2.1 success criteria and broadcast-accessibility
/// standards (EBU R 115, FCC 47 CFR Part 79, Ofcom guidance).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Criterion {
    // ── Captions / subtitles ────────────────────────────────────────────────
    /// Closed captions or subtitles present (WCAG 1.2.2 / 1.2.4).
    CaptionsPresent,
    /// Captions cover ≥ 95 % of dialogue duration.
    CaptionCoverage,
    /// Caption timing accuracy (word-level synchronisation).
    CaptionTimingAccuracy,
    /// At least one non-English caption track present.
    CaptionMultiLanguage,

    // ── Audio description ───────────────────────────────────────────────────
    /// Audio description track present (WCAG 1.2.3 / 1.2.5).
    AudioDescriptionPresent,
    /// AD coverage ≥ 80 % of significant visual content.
    AudioDescriptionCoverage,
    /// AD delivered at a comfortable speaking rate (< 200 wpm average).
    AudioDescriptionSpeakingRate,

    // ── Sign language ────────────────────────────────────────────────────────
    /// Sign language interpretation track present (WCAG 1.2.6).
    SignLanguagePresent,
    /// Sign language coverage ≥ 90 % of programme duration.
    SignLanguageCoverage,
    /// Sign language interpretation by a qualified interpreter.
    SignLanguageQuality,

    // ── Navigation ──────────────────────────────────────────────────────────
    /// Chapter/cue-point markers present for navigation.
    ChaptersPresent,
    /// Descriptive title and metadata present.
    DescriptiveMetadata,

    // ── Transcript ──────────────────────────────────────────────────────────
    /// Full text transcript available (WCAG 1.2.8).
    TranscriptPresent,

    // ── Audio quality ───────────────────────────────────────────────────────
    /// Audio meets EBU R 128 loudness normalisation target.
    LoudnessNormalised,
    /// Speech intelligibility score ≥ threshold.
    SpeechIntelligibility,
}

impl Criterion {
    /// Short human-readable label.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::CaptionsPresent => "Captions Present",
            Self::CaptionCoverage => "Caption Coverage ≥ 95%",
            Self::CaptionTimingAccuracy => "Caption Timing Accuracy",
            Self::CaptionMultiLanguage => "Multi-language Captions",
            Self::AudioDescriptionPresent => "Audio Description Present",
            Self::AudioDescriptionCoverage => "Audio Description Coverage ≥ 80%",
            Self::AudioDescriptionSpeakingRate => "AD Speaking Rate < 200 wpm",
            Self::SignLanguagePresent => "Sign Language Track Present",
            Self::SignLanguageCoverage => "Sign Language Coverage ≥ 90%",
            Self::SignLanguageQuality => "Qualified Sign Language Interpreter",
            Self::ChaptersPresent => "Chapter Markers Present",
            Self::DescriptiveMetadata => "Descriptive Title & Metadata",
            Self::TranscriptPresent => "Full Text Transcript Present",
            Self::LoudnessNormalised => "EBU R 128 Loudness Normalised",
            Self::SpeechIntelligibility => "Speech Intelligibility ≥ Threshold",
        }
    }

    /// The WCAG 2.1 success criterion reference, if any.
    #[must_use]
    pub fn wcag_reference(&self) -> Option<&'static str> {
        match self {
            Self::CaptionsPresent => Some("1.2.2 / 1.2.4"),
            Self::AudioDescriptionPresent => Some("1.2.3 / 1.2.5"),
            Self::SignLanguagePresent => Some("1.2.6"),
            Self::TranscriptPresent => Some("1.2.8"),
            _ => None,
        }
    }

    /// The minimum WCAG conformance level at which this criterion is required.
    #[must_use]
    pub fn minimum_level(&self) -> WcagLevel {
        match self {
            Self::CaptionsPresent
            | Self::AudioDescriptionPresent
            | Self::DescriptiveMetadata
            | Self::LoudnessNormalised => WcagLevel::A,

            Self::CaptionCoverage
            | Self::CaptionTimingAccuracy
            | Self::AudioDescriptionCoverage
            | Self::ChaptersPresent
            | Self::TranscriptPresent
            | Self::SpeechIntelligibility => WcagLevel::Aa,

            Self::AudioDescriptionSpeakingRate
            | Self::CaptionMultiLanguage
            | Self::SignLanguagePresent
            | Self::SignLanguageCoverage
            | Self::SignLanguageQuality => WcagLevel::Aaa,
        }
    }

    /// Relative weight used in score calculation (higher = more impactful).
    #[must_use]
    pub fn weight(&self) -> f64 {
        match self {
            Self::CaptionsPresent => 10.0,
            Self::CaptionCoverage => 8.0,
            Self::CaptionTimingAccuracy => 5.0,
            Self::CaptionMultiLanguage => 3.0,
            Self::AudioDescriptionPresent => 10.0,
            Self::AudioDescriptionCoverage => 7.0,
            Self::AudioDescriptionSpeakingRate => 4.0,
            Self::SignLanguagePresent => 8.0,
            Self::SignLanguageCoverage => 6.0,
            Self::SignLanguageQuality => 4.0,
            Self::ChaptersPresent => 5.0,
            Self::DescriptiveMetadata => 4.0,
            Self::TranscriptPresent => 8.0,
            Self::LoudnessNormalised => 5.0,
            Self::SpeechIntelligibility => 6.0,
        }
    }
}

// ─── Status ───────────────────────────────────────────────────────────────────

/// Status of a single accessibility criterion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CriterionStatus {
    /// Criterion is fully satisfied.
    Pass,
    /// Criterion is partially satisfied (e.g. coverage below threshold).
    Partial,
    /// Criterion is not satisfied.
    Fail,
    /// Criterion is not applicable for this media type.
    NotApplicable,
}

impl CriterionStatus {
    /// Whether this status counts as a full pass for scoring.
    #[must_use]
    pub fn is_pass(&self) -> bool {
        matches!(self, Self::Pass)
    }

    /// Score contribution factor `[0.0, 1.0]`.
    #[must_use]
    pub fn score_factor(&self) -> f64 {
        match self {
            Self::Pass => 1.0,
            Self::Partial => 0.5,
            Self::Fail => 0.0,
            Self::NotApplicable => 1.0, // N/A does not penalise the score
        }
    }

    /// Whether this status causes a mandatory failure.
    #[must_use]
    pub fn is_failure(&self) -> bool {
        matches!(self, Self::Fail)
    }
}

impl fmt::Display for CriterionStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Pass => "PASS",
            Self::Partial => "PARTIAL",
            Self::Fail => "FAIL",
            Self::NotApplicable => "N/A",
        };
        f.write_str(s)
    }
}

// ─── WCAG Level ───────────────────────────────────────────────────────────────

/// WCAG 2.x conformance level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum WcagLevel {
    /// No conformance.
    None = 0,
    /// Level A — minimum.
    A = 1,
    /// Level AA — recommended for broadcast.
    Aa = 2,
    /// Level AAA — enhanced.
    Aaa = 3,
}

impl fmt::Display for WcagLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::None => "None",
            Self::A => "A",
            Self::Aa => "AA",
            Self::Aaa => "AAA",
        };
        f.write_str(s)
    }
}

// ─── Finding ──────────────────────────────────────────────────────────────────

/// A single finding within the report.
#[derive(Debug, Clone)]
pub struct Finding {
    /// The criterion assessed.
    pub criterion: Criterion,
    /// Assessment outcome.
    pub status: CriterionStatus,
    /// Optional detail or measurement (e.g. `"coverage = 87.4%"`).
    pub detail: Option<String>,
    /// Actionable recommendation when status is `Fail` or `Partial`.
    pub recommendation: Option<String>,
}

impl Finding {
    /// Create a passing finding.
    #[must_use]
    pub fn pass(criterion: Criterion) -> Self {
        Self {
            criterion,
            status: CriterionStatus::Pass,
            detail: None,
            recommendation: None,
        }
    }

    /// Create a failing finding with a recommendation.
    #[must_use]
    pub fn fail(criterion: Criterion, recommendation: impl Into<String>) -> Self {
        Self {
            criterion,
            status: CriterionStatus::Fail,
            detail: None,
            recommendation: Some(recommendation.into()),
        }
    }

    /// Create a partial finding.
    #[must_use]
    pub fn partial(
        criterion: Criterion,
        detail: impl Into<String>,
        recommendation: impl Into<String>,
    ) -> Self {
        Self {
            criterion,
            status: CriterionStatus::Partial,
            detail: Some(detail.into()),
            recommendation: Some(recommendation.into()),
        }
    }

    /// Create a not-applicable finding.
    #[must_use]
    pub fn not_applicable(criterion: Criterion) -> Self {
        Self {
            criterion,
            status: CriterionStatus::NotApplicable,
            detail: None,
            recommendation: None,
        }
    }

    /// Attach extra detail.
    #[must_use]
    pub fn with_detail(mut self, detail: impl Into<String>) -> Self {
        self.detail = Some(detail.into());
        self
    }

    /// Weighted score contribution of this finding.
    #[must_use]
    pub fn score_contribution(&self) -> f64 {
        self.criterion.weight() * self.status.score_factor()
    }
}

// ─── Report Builder ───────────────────────────────────────────────────────────

/// Builder for constructing an [`AccessibilityReport`].
#[derive(Debug, Default)]
pub struct ReportBuilder {
    asset_id: String,
    asset_title: Option<String>,
    findings: BTreeMap<Criterion, Finding>,
}

impl ReportBuilder {
    /// Create a builder for the given asset.
    #[must_use]
    pub fn new(asset_id: impl Into<String>) -> Self {
        Self {
            asset_id: asset_id.into(),
            asset_title: None,
            findings: BTreeMap::new(),
        }
    }

    /// Set a human-readable asset title.
    #[must_use]
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.asset_title = Some(title.into());
        self
    }

    /// Record a finding, replacing any existing finding for the same criterion.
    #[must_use]
    pub fn record(mut self, finding: Finding) -> Self {
        self.findings.insert(finding.criterion, finding);
        self
    }

    /// Build the final report.
    ///
    /// # Errors
    /// Returns [`AccessError::Other`] when no findings have been recorded.
    pub fn build(self) -> AccessResult<AccessibilityReport> {
        if self.findings.is_empty() {
            return Err(AccessError::Other(
                "Cannot build an accessibility report with no findings".to_string(),
            ));
        }
        Ok(AccessibilityReport {
            asset_id: self.asset_id,
            asset_title: self.asset_title,
            findings: self.findings,
        })
    }
}

// ─── Report ───────────────────────────────────────────────────────────────────

/// Accessibility compliance report for a media asset.
#[derive(Debug, Clone)]
pub struct AccessibilityReport {
    /// Asset identifier.
    pub asset_id: String,
    /// Human-readable asset title.
    pub asset_title: Option<String>,
    /// Findings, keyed by criterion and ordered by criterion ordinal.
    findings: BTreeMap<Criterion, Finding>,
}

impl AccessibilityReport {
    /// Return all findings.
    #[must_use]
    pub fn findings(&self) -> impl Iterator<Item = &Finding> {
        self.findings.values()
    }

    /// Return the finding for a specific criterion, if recorded.
    #[must_use]
    pub fn finding_for(&self, criterion: Criterion) -> Option<&Finding> {
        self.findings.get(&criterion)
    }

    /// Number of findings recorded.
    #[must_use]
    pub fn finding_count(&self) -> usize {
        self.findings.len()
    }

    /// Count findings with a given status.
    #[must_use]
    pub fn count_with_status(&self, status: CriterionStatus) -> usize {
        self.findings.values().filter(|f| f.status == status).count()
    }

    /// Compute the weighted accessibility score in `[0.0, 100.0]`.
    ///
    /// Score = sum(weight_i × factor_i) / sum(weight_i) × 100
    ///
    /// N/A criteria contribute their full weight to the denominator only when
    /// there are also non-N/A criteria, so a report of only N/A findings
    /// returns `None`.
    #[must_use]
    pub fn score(&self) -> Option<f64> {
        let applicable: Vec<&Finding> = self
            .findings
            .values()
            .filter(|f| f.status != CriterionStatus::NotApplicable)
            .collect();

        if applicable.is_empty() {
            return None;
        }

        let total_weight: f64 = applicable.iter().map(|f| f.criterion.weight()).sum();
        if total_weight == 0.0 {
            return None;
        }

        let earned: f64 = applicable.iter().map(|f| f.score_contribution()).sum();
        Some((earned / total_weight * 100.0).clamp(0.0, 100.0))
    }

    /// Derive the highest WCAG conformance level achieved.
    ///
    /// A level is achieved when:
    /// 1. All criteria at *that exact level* are recorded and pass (or are N/A).
    /// 2. All criteria at levels *below* that level also pass or are N/A.
    /// 3. At least one criterion at *that exact level* is recorded (non-N/A).
    ///
    /// The check is performed from AAA down to A; the first matching level is returned.
    #[must_use]
    pub fn conformance_level(&self) -> WcagLevel {
        for &level in &[WcagLevel::Aaa, WcagLevel::Aa, WcagLevel::A] {
            // Require at least one applicable criterion recorded at exactly this level.
            let has_this_level = self
                .findings
                .values()
                .any(|f| f.criterion.minimum_level() == level && f.status != CriterionStatus::NotApplicable);

            if !has_this_level {
                continue;
            }

            // All applicable criteria at this level or below must pass.
            let all_pass = self
                .findings
                .values()
                .filter(|f| f.criterion.minimum_level() <= level)
                .filter(|f| f.status != CriterionStatus::NotApplicable)
                .all(|f| f.status == CriterionStatus::Pass);

            if all_pass {
                return level;
            }
        }
        WcagLevel::None
    }

    /// Whether every recorded criterion passes (excluding N/A).
    #[must_use]
    pub fn is_fully_compliant(&self) -> bool {
        self.findings
            .values()
            .filter(|f| f.status != CriterionStatus::NotApplicable)
            .all(|f| f.status == CriterionStatus::Pass)
    }

    /// Return all actionable recommendations (from Fail / Partial findings).
    #[must_use]
    pub fn recommendations(&self) -> Vec<(Criterion, &str)> {
        self.findings
            .values()
            .filter_map(|f| {
                f.recommendation
                    .as_deref()
                    .filter(|_| f.status.is_failure() || f.status == CriterionStatus::Partial)
                    .map(|r| (f.criterion, r))
            })
            .collect()
    }

    /// Generate a brief summary line.
    #[must_use]
    pub fn summary(&self) -> String {
        let score = self.score().map_or_else(|| "N/A".to_string(), |s| format!("{s:.1}"));
        let level = self.conformance_level();
        let passes = self.count_with_status(CriterionStatus::Pass);
        let fails = self.count_with_status(CriterionStatus::Fail);
        let partials = self.count_with_status(CriterionStatus::Partial);
        format!(
            "Asset '{}' — Score: {score}/100 | Level: {level} | Pass: {passes}, Partial: {partials}, Fail: {fails}",
            self.asset_id
        )
    }
}

impl fmt::Display for AccessibilityReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Accessibility Report ===")?;
        if let Some(title) = &self.asset_title {
            writeln!(f, "Title  : {title}")?;
        }
        writeln!(f, "Asset  : {}", self.asset_id)?;
        writeln!(f, "Level  : {}", self.conformance_level())?;
        if let Some(score) = self.score() {
            writeln!(f, "Score  : {score:.1}/100")?;
        }
        writeln!(f, "---")?;
        for finding in self.findings() {
            let wcag = finding
                .criterion
                .wcag_reference()
                .map_or_else(String::new, |r| format!(" [WCAG {r}]"));
            writeln!(
                f,
                "  [{status}] {label}{wcag}",
                status = finding.status,
                label = finding.criterion.label(),
            )?;
            if let Some(detail) = &finding.detail {
                writeln!(f, "    detail: {detail}")?;
            }
            if let Some(rec) = &finding.recommendation {
                if finding.status.is_failure() || finding.status == CriterionStatus::Partial {
                    writeln!(f, "    → {rec}")?;
                }
            }
        }
        Ok(())
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn fully_passing_report(asset_id: &str) -> AccessibilityReport {
        ReportBuilder::new(asset_id)
            .record(Finding::pass(Criterion::CaptionsPresent))
            .record(Finding::pass(Criterion::CaptionCoverage))
            .record(Finding::pass(Criterion::AudioDescriptionPresent))
            .record(Finding::pass(Criterion::AudioDescriptionCoverage))
            .record(Finding::pass(Criterion::TranscriptPresent))
            .record(Finding::pass(Criterion::LoudnessNormalised))
            .record(Finding::pass(Criterion::ChaptersPresent))
            .record(Finding::pass(Criterion::DescriptiveMetadata))
            .build()
            .expect("valid report")
    }

    #[test]
    fn test_builder_requires_findings() {
        let result = ReportBuilder::new("asset-1").build();
        assert!(result.is_err());
    }

    #[test]
    fn test_score_all_pass() {
        let report = fully_passing_report("a1");
        let score = report.score().expect("score available");
        assert!((score - 100.0).abs() < 1e-6, "score={score}");
    }

    #[test]
    fn test_score_all_fail() {
        let report = ReportBuilder::new("a1")
            .record(Finding::fail(Criterion::CaptionsPresent, "Add captions"))
            .record(Finding::fail(Criterion::AudioDescriptionPresent, "Add AD"))
            .build()
            .unwrap();
        let score = report.score().expect("score");
        assert!((score - 0.0).abs() < 1e-6, "score={score}");
    }

    #[test]
    fn test_score_partial_is_half() {
        // Single criterion at partial — score should be 50.
        let report = ReportBuilder::new("a1")
            .record(Finding::partial(
                Criterion::CaptionsPresent,
                "coverage 80%",
                "Increase caption coverage",
            ))
            .build()
            .unwrap();
        let score = report.score().expect("score");
        assert!((score - 50.0).abs() < 1e-6, "score={score}");
    }

    #[test]
    fn test_conformance_level_none_when_fail() {
        let report = ReportBuilder::new("a1")
            .record(Finding::fail(Criterion::CaptionsPresent, "Missing captions"))
            .record(Finding::pass(Criterion::AudioDescriptionPresent))
            .build()
            .unwrap();
        assert_eq!(report.conformance_level(), WcagLevel::None);
    }

    #[test]
    fn test_conformance_level_a_when_level_a_pass() {
        let report = ReportBuilder::new("a1")
            .record(Finding::pass(Criterion::CaptionsPresent))
            .record(Finding::pass(Criterion::AudioDescriptionPresent))
            .record(Finding::pass(Criterion::DescriptiveMetadata))
            .record(Finding::pass(Criterion::LoudnessNormalised))
            .build()
            .unwrap();
        // Only level-A criteria recorded, all pass → level A.
        assert_eq!(report.conformance_level(), WcagLevel::A);
    }

    #[test]
    fn test_fully_compliant_flag() {
        let report = fully_passing_report("a1");
        assert!(report.is_fully_compliant());
    }

    #[test]
    fn test_not_applicable_excluded_from_score() {
        let report = ReportBuilder::new("a1")
            .record(Finding::pass(Criterion::CaptionsPresent))
            .record(Finding::not_applicable(Criterion::SignLanguagePresent))
            .build()
            .unwrap();
        let score = report.score().expect("score");
        // Only CaptionsPresent is applicable → 100.
        assert!((score - 100.0).abs() < 1e-6, "score={score}");
    }

    #[test]
    fn test_recommendations_only_for_fail_and_partial() {
        let report = ReportBuilder::new("a1")
            .record(Finding::pass(Criterion::CaptionsPresent))
            .record(Finding::fail(Criterion::AudioDescriptionPresent, "Add AD track"))
            .record(Finding::partial(
                Criterion::CaptionCoverage,
                "88%",
                "Increase to 95%",
            ))
            .build()
            .unwrap();
        let recs = report.recommendations();
        assert_eq!(recs.len(), 2);
        let criteria: Vec<Criterion> = recs.iter().map(|(c, _)| *c).collect();
        assert!(criteria.contains(&Criterion::AudioDescriptionPresent));
        assert!(criteria.contains(&Criterion::CaptionCoverage));
    }

    #[test]
    fn test_count_with_status() {
        let report = ReportBuilder::new("a1")
            .record(Finding::pass(Criterion::CaptionsPresent))
            .record(Finding::fail(Criterion::AudioDescriptionPresent, "Add AD"))
            .record(Finding::not_applicable(Criterion::SignLanguagePresent))
            .build()
            .unwrap();
        assert_eq!(report.count_with_status(CriterionStatus::Pass), 1);
        assert_eq!(report.count_with_status(CriterionStatus::Fail), 1);
        assert_eq!(report.count_with_status(CriterionStatus::NotApplicable), 1);
    }

    #[test]
    fn test_summary_contains_asset_id() {
        let report = fully_passing_report("asset-xyz");
        let summary = report.summary();
        assert!(summary.contains("asset-xyz"), "summary={summary}");
        assert!(summary.contains("100.0"), "summary={summary}");
    }

    #[test]
    fn test_finding_for_lookup() {
        let report = ReportBuilder::new("a1")
            .record(Finding::pass(Criterion::CaptionsPresent))
            .build()
            .unwrap();
        assert!(report.finding_for(Criterion::CaptionsPresent).is_some());
        assert!(report.finding_for(Criterion::TranscriptPresent).is_none());
    }

    #[test]
    fn test_display_format_contains_level() {
        let report = fully_passing_report("test-asset");
        let display = format!("{report}");
        assert!(display.contains("Level"), "display={display}");
        assert!(display.contains("test-asset"), "display={display}");
    }

    #[test]
    fn test_criterion_weight_positive() {
        for c in [
            Criterion::CaptionsPresent,
            Criterion::AudioDescriptionPresent,
            Criterion::SignLanguagePresent,
            Criterion::TranscriptPresent,
        ] {
            assert!(c.weight() > 0.0, "{c:?} weight must be positive");
        }
    }
}
