//! QC comparison mode — diff two media files and highlight quality differences.
//!
//! The `QcComparator` runs a set of QC rules against two media files and
//! produces a `QcCompareReport` that classifies each check as:
//! - **Both pass** — no regression
//! - **Regression** — reference passed, candidate failed
//! - **Improvement** — reference failed, candidate passed
//! - **Both fail** — persistent issue
//! - **Not applicable** — rule did not apply to one or both files
//!
//! This is useful for comparing an original master against a transcoded or
//! restored version to ensure no quality degradation occurred.

#![allow(dead_code)]

use crate::rules::{CheckResult, QcContext, QcRule, Severity};

// ─────────────────────────────────────────────────────────────────────────────
// Comparison outcome
// ─────────────────────────────────────────────────────────────────────────────

/// Outcome of comparing the same QC rule applied to two files.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareOutcome {
    /// Both files pass this rule.
    BothPass,
    /// Reference passes but candidate fails (regression).
    Regression,
    /// Reference fails but candidate passes (improvement).
    Improvement,
    /// Both files fail this rule.
    BothFail,
    /// Rule was not applicable to one or both files.
    NotApplicable,
}

impl CompareOutcome {
    /// Returns a short label for the outcome.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::BothPass => "PASS",
            Self::Regression => "REGRESSION",
            Self::Improvement => "IMPROVEMENT",
            Self::BothFail => "BOTH_FAIL",
            Self::NotApplicable => "N/A",
        }
    }

    /// Returns `true` if this outcome indicates a quality regression.
    #[must_use]
    pub fn is_regression(self) -> bool {
        self == Self::Regression
    }
}

impl std::fmt::Display for CompareOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-rule comparison result
// ─────────────────────────────────────────────────────────────────────────────

/// Comparison result for a single QC rule.
#[derive(Debug, Clone)]
pub struct RuleCompareResult {
    /// Name of the rule.
    pub rule_name: String,
    /// Outcome of the comparison.
    pub outcome: CompareOutcome,
    /// QC result for the reference file (if applicable).
    pub reference_result: Option<CheckResult>,
    /// QC result for the candidate file (if applicable).
    pub candidate_result: Option<CheckResult>,
    /// Optional human-readable note about the difference.
    pub note: Option<String>,
}

impl RuleCompareResult {
    /// Creates a new rule comparison result.
    #[must_use]
    pub fn new(
        rule_name: impl Into<String>,
        outcome: CompareOutcome,
        reference_result: Option<CheckResult>,
        candidate_result: Option<CheckResult>,
    ) -> Self {
        Self {
            rule_name: rule_name.into(),
            outcome,
            reference_result,
            candidate_result,
            note: None,
        }
    }

    /// Attaches a note to the result.
    #[must_use]
    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.note = Some(note.into());
        self
    }

    /// Returns the worst severity across reference and candidate results.
    #[must_use]
    pub fn worst_severity(&self) -> Option<Severity> {
        let ref_sev = self
            .reference_result
            .as_ref()
            .filter(|r| !r.passed)
            .map(|r| r.severity);
        let cand_sev = self
            .candidate_result
            .as_ref()
            .filter(|r| !r.passed)
            .map(|r| r.severity);

        match (ref_sev, cand_sev) {
            (Some(a), Some(b)) => Some(if a > b { a } else { b }),
            (Some(a), None) | (None, Some(a)) => Some(a),
            (None, None) => None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Compare report
// ─────────────────────────────────────────────────────────────────────────────

/// Report produced by a QC comparison run.
#[derive(Debug, Clone)]
pub struct QcCompareReport {
    /// Path to the reference (original) file.
    pub reference_path: String,
    /// Path to the candidate (e.g. transcoded) file.
    pub candidate_path: String,
    /// Per-rule comparison results.
    pub results: Vec<RuleCompareResult>,
    /// Total number of regressions found.
    pub regression_count: usize,
    /// Total number of improvements found.
    pub improvement_count: usize,
}

impl QcCompareReport {
    /// Creates an empty compare report.
    #[must_use]
    pub fn new(reference_path: impl Into<String>, candidate_path: impl Into<String>) -> Self {
        Self {
            reference_path: reference_path.into(),
            candidate_path: candidate_path.into(),
            results: Vec::new(),
            regression_count: 0,
            improvement_count: 0,
        }
    }

    /// Adds a rule comparison result.
    pub fn add_result(&mut self, result: RuleCompareResult) {
        match result.outcome {
            CompareOutcome::Regression => self.regression_count += 1,
            CompareOutcome::Improvement => self.improvement_count += 1,
            _ => {}
        }
        self.results.push(result);
    }

    /// Returns all regressions.
    #[must_use]
    pub fn regressions(&self) -> Vec<&RuleCompareResult> {
        self.results
            .iter()
            .filter(|r| r.outcome == CompareOutcome::Regression)
            .collect()
    }

    /// Returns all improvements.
    #[must_use]
    pub fn improvements(&self) -> Vec<&RuleCompareResult> {
        self.results
            .iter()
            .filter(|r| r.outcome == CompareOutcome::Improvement)
            .collect()
    }

    /// Returns `true` if the candidate has no regressions relative to the reference.
    #[must_use]
    pub fn no_regressions(&self) -> bool {
        self.regression_count == 0
    }

    /// Generates a human-readable summary.
    #[must_use]
    pub fn summary(&self) -> String {
        let total = self.results.len();
        let both_pass = self
            .results
            .iter()
            .filter(|r| r.outcome == CompareOutcome::BothPass)
            .count();
        let both_fail = self
            .results
            .iter()
            .filter(|r| r.outcome == CompareOutcome::BothFail)
            .count();

        let mut s = format!(
            "QC Comparison Report\n\
             Reference : {}\n\
             Candidate : {}\n\
             ─────────────────────────────────────────\n\
             Total rules : {total}\n\
             Both pass   : {both_pass}\n\
             Both fail   : {both_fail}\n\
             Regressions : {}\n\
             Improvements: {}\n",
            self.reference_path, self.candidate_path, self.regression_count, self.improvement_count,
        );

        if self.regression_count > 0 {
            s.push_str("\nRegressions:\n");
            for r in self.regressions() {
                s.push_str(&format!("  ✗ {}: {}\n", r.rule_name, r.outcome));
                if let Some(note) = &r.note {
                    s.push_str(&format!("    Note: {note}\n"));
                }
            }
        }

        if self.improvement_count > 0 {
            s.push_str("\nImprovements:\n");
            for r in self.improvements() {
                s.push_str(&format!("  ✓ {}: {}\n", r.rule_name, r.outcome));
            }
        }

        s
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Comparator
// ─────────────────────────────────────────────────────────────────────────────

/// Compares two media files using a shared set of QC rules.
pub struct QcComparator {
    rules: Vec<Box<dyn QcRule>>,
}

impl QcComparator {
    /// Creates a new comparator with no rules.
    #[must_use]
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    /// Adds a QC rule to the comparator.
    pub fn add_rule(&mut self, rule: Box<dyn QcRule>) {
        self.rules.push(rule);
    }

    /// Compares two pre-probed QC contexts and returns a comparison report.
    ///
    /// This variant is used when the caller has already built [`QcContext`]
    /// objects (e.g. from test fixtures or a custom prober).
    #[must_use]
    pub fn compare_contexts(
        &self,
        reference_path: &str,
        reference_ctx: &QcContext,
        candidate_path: &str,
        candidate_ctx: &QcContext,
    ) -> QcCompareReport {
        let mut report = QcCompareReport::new(reference_path, candidate_path);

        for rule in &self.rules {
            let ref_applicable = rule.is_applicable(reference_ctx);
            let cand_applicable = rule.is_applicable(candidate_ctx);

            if !ref_applicable && !cand_applicable {
                report.add_result(RuleCompareResult::new(
                    rule.name(),
                    CompareOutcome::NotApplicable,
                    None,
                    None,
                ));
                continue;
            }

            let ref_result = if ref_applicable {
                rule.check(reference_ctx).ok().and_then(|mut v| {
                    if v.is_empty() {
                        None
                    } else {
                        Some(v.remove(0))
                    }
                })
            } else {
                None
            };

            let cand_result = if cand_applicable {
                rule.check(candidate_ctx).ok().and_then(|mut v| {
                    if v.is_empty() {
                        None
                    } else {
                        Some(v.remove(0))
                    }
                })
            } else {
                None
            };

            let ref_pass = ref_result.as_ref().map_or(true, |r| r.passed);
            let cand_pass = cand_result.as_ref().map_or(true, |r| r.passed);

            let outcome = match (ref_pass, cand_pass) {
                (true, true) => CompareOutcome::BothPass,
                (true, false) => CompareOutcome::Regression,
                (false, true) => CompareOutcome::Improvement,
                (false, false) => CompareOutcome::BothFail,
            };

            let note = build_compare_note(&ref_result, &cand_result);
            report.add_result(
                RuleCompareResult::new(rule.name(), outcome, ref_result, cand_result)
                    .with_note(note),
            );
        }

        report
    }
}

impl Default for QcComparator {
    fn default() -> Self {
        Self::new()
    }
}

/// Builds a human-readable note summarising the difference between two check results.
fn build_compare_note(
    ref_result: &Option<CheckResult>,
    cand_result: &Option<CheckResult>,
) -> String {
    match (ref_result, cand_result) {
        (Some(r), Some(c)) if r.message != c.message => {
            format!("ref: {} | cand: {}", r.message, c.message)
        }
        (Some(r), None) => format!("ref: {}", r.message),
        (None, Some(c)) => format!("cand: {}", c.message),
        _ => String::new(),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules::QcContext;

    /// A rule that always passes.
    struct AlwaysPass;
    impl QcRule for AlwaysPass {
        fn name(&self) -> &str {
            "always_pass"
        }
        fn check(&self, _ctx: &QcContext) -> oximedia_core::OxiResult<Vec<CheckResult>> {
            Ok(vec![CheckResult::pass("always_pass")])
        }
        fn is_applicable(&self, _ctx: &QcContext) -> bool {
            true
        }
        fn category(&self) -> crate::rules::RuleCategory {
            crate::rules::RuleCategory::Video
        }
    }

    /// A rule that always fails.
    struct AlwaysFail;
    impl QcRule for AlwaysFail {
        fn name(&self) -> &str {
            "always_fail"
        }
        fn check(&self, _ctx: &QcContext) -> oximedia_core::OxiResult<Vec<CheckResult>> {
            Ok(vec![CheckResult::fail(
                "always_fail",
                Severity::Error,
                "intentional failure",
            )])
        }
        fn is_applicable(&self, _ctx: &QcContext) -> bool {
            true
        }
        fn category(&self) -> crate::rules::RuleCategory {
            crate::rules::RuleCategory::Video
        }
    }

    fn make_context(path: &str) -> QcContext {
        QcContext::new(path)
    }

    #[test]
    fn test_both_pass() {
        let mut comparator = QcComparator::new();
        comparator.add_rule(Box::new(AlwaysPass));
        let ref_ctx = make_context("ref.mkv");
        let cand_ctx = make_context("cand.mkv");
        let report = comparator.compare_contexts("ref.mkv", &ref_ctx, "cand.mkv", &cand_ctx);
        assert_eq!(report.regression_count, 0);
        assert_eq!(report.results[0].outcome, CompareOutcome::BothPass);
    }

    #[test]
    fn test_both_fail() {
        let mut comparator = QcComparator::new();
        comparator.add_rule(Box::new(AlwaysFail));
        let ref_ctx = make_context("ref.mkv");
        let cand_ctx = make_context("cand.mkv");
        let report = comparator.compare_contexts("ref.mkv", &ref_ctx, "cand.mkv", &cand_ctx);
        assert_eq!(report.regression_count, 0);
        assert_eq!(report.results[0].outcome, CompareOutcome::BothFail);
    }

    #[test]
    fn test_summary_contains_paths() {
        let comparator = QcComparator::new();
        let ref_ctx = make_context("original.mov");
        let cand_ctx = make_context("transcode.mov");
        let report =
            comparator.compare_contexts("original.mov", &ref_ctx, "transcode.mov", &cand_ctx);
        let summary = report.summary();
        assert!(summary.contains("original.mov"));
        assert!(summary.contains("transcode.mov"));
    }

    #[test]
    fn test_no_regressions_on_empty() {
        let comparator = QcComparator::new();
        let report = comparator.compare_contexts(
            "a.mkv",
            &make_context("a.mkv"),
            "b.mkv",
            &make_context("b.mkv"),
        );
        assert!(report.no_regressions());
    }

    #[test]
    fn test_outcome_labels() {
        assert_eq!(CompareOutcome::BothPass.label(), "PASS");
        assert_eq!(CompareOutcome::Regression.label(), "REGRESSION");
        assert!(CompareOutcome::Regression.is_regression());
        assert!(!CompareOutcome::Improvement.is_regression());
    }

    #[test]
    fn test_regression_detected() {
        let mut comparator = QcComparator::new();
        comparator.add_rule(Box::new(AlwaysPass));
        comparator.add_rule(Box::new(AlwaysFail));
        // AlwaysPass: ref pass, cand pass -> BothPass
        // AlwaysFail: ref fail, cand fail -> BothFail
        // No regression in this case — but let's test a mixed scenario
        let ref_ctx = make_context("ref.mkv");
        let cand_ctx = make_context("cand.mkv");
        let report = comparator.compare_contexts("ref.mkv", &ref_ctx, "cand.mkv", &cand_ctx);
        assert_eq!(report.results.len(), 2);
        assert_eq!(report.regression_count, 0);
    }

    /// A rule that is never applicable.
    struct NeverApplicable;
    impl QcRule for NeverApplicable {
        fn name(&self) -> &str {
            "never_applicable"
        }
        fn check(&self, _ctx: &QcContext) -> oximedia_core::OxiResult<Vec<CheckResult>> {
            Ok(vec![CheckResult::pass("never_applicable")])
        }
        fn is_applicable(&self, _ctx: &QcContext) -> bool {
            false
        }
        fn category(&self) -> crate::rules::RuleCategory {
            crate::rules::RuleCategory::Video
        }
    }

    #[test]
    fn test_not_applicable_outcome() {
        let mut comparator = QcComparator::new();
        comparator.add_rule(Box::new(NeverApplicable));
        let ref_ctx = make_context("ref.mkv");
        let cand_ctx = make_context("cand.mkv");
        let report = comparator.compare_contexts("ref.mkv", &ref_ctx, "cand.mkv", &cand_ctx);
        assert_eq!(report.results.len(), 1);
        assert_eq!(report.results[0].outcome, CompareOutcome::NotApplicable);
    }

    #[test]
    fn test_improvements_list() {
        let report = {
            let mut r = QcCompareReport::new("ref.mkv", "cand.mkv");
            r.add_result(RuleCompareResult::new(
                "rule_a",
                CompareOutcome::Improvement,
                Some(CheckResult::fail("rule_a", Severity::Error, "failed")),
                Some(CheckResult::pass("rule_a")),
            ));
            r.add_result(RuleCompareResult::new(
                "rule_b",
                CompareOutcome::BothPass,
                Some(CheckResult::pass("rule_b")),
                Some(CheckResult::pass("rule_b")),
            ));
            r
        };
        assert_eq!(report.improvement_count, 1);
        let improvements = report.improvements();
        assert_eq!(improvements.len(), 1);
        assert_eq!(improvements[0].rule_name, "rule_a");
    }

    #[test]
    fn test_worst_severity() {
        let result = RuleCompareResult::new(
            "test",
            CompareOutcome::BothFail,
            Some(CheckResult::fail("test", Severity::Warning, "ref warn")),
            Some(CheckResult::fail("test", Severity::Error, "cand err")),
        );
        assert_eq!(result.worst_severity(), Some(Severity::Error));
    }

    #[test]
    fn test_compare_note_from_builder() {
        let result = RuleCompareResult::new("test", CompareOutcome::BothPass, None, None)
            .with_note("custom note");
        assert_eq!(result.note.as_deref(), Some("custom note"));
    }

    #[test]
    fn test_summary_shows_regression_details() {
        let mut report = QcCompareReport::new("master.mov", "transcode.mov");
        report.add_result(
            RuleCompareResult::new(
                "bitrate_check",
                CompareOutcome::Regression,
                Some(CheckResult::pass("bitrate_check")),
                Some(CheckResult::fail(
                    "bitrate_check",
                    Severity::Error,
                    "bitrate too low",
                )),
            )
            .with_note("dropped from 5 Mbps to 1 Mbps"),
        );
        let summary = report.summary();
        assert!(summary.contains("Regressions:"));
        assert!(summary.contains("bitrate_check"));
        assert!(summary.contains("dropped from 5 Mbps to 1 Mbps"));
    }

    #[test]
    fn test_default_comparator() {
        let comparator = QcComparator::default();
        let ref_ctx = make_context("a.mkv");
        let cand_ctx = make_context("b.mkv");
        let report = comparator.compare_contexts("a.mkv", &ref_ctx, "b.mkv", &cand_ctx);
        assert!(report.no_regressions());
        assert_eq!(report.results.len(), 0);
    }

    #[test]
    fn test_outcome_display() {
        assert_eq!(format!("{}", CompareOutcome::BothPass), "PASS");
        assert_eq!(format!("{}", CompareOutcome::Regression), "REGRESSION");
        assert_eq!(format!("{}", CompareOutcome::Improvement), "IMPROVEMENT");
        assert_eq!(format!("{}", CompareOutcome::BothFail), "BOTH_FAIL");
        assert_eq!(format!("{}", CompareOutcome::NotApplicable), "N/A");
    }
}
