#![allow(dead_code)]
//! Calibration report generation and analysis.
//!
//! This module generates detailed reports from calibration sessions, including
//! accuracy metrics, color deviation analysis, and pass/fail verdicts. Reports
//! can be exported in structured formats for archival and compliance.

use std::collections::HashMap;

/// Overall calibration verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalibrationVerdict {
    /// All metrics within tolerance.
    Pass,
    /// Some metrics outside tolerance but within warning range.
    Warning,
    /// Critical metrics outside tolerance.
    Fail,
}

/// Severity of a single calibration finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum FindingSeverity {
    /// Informational finding.
    Info,
    /// Minor deviation, within acceptable range.
    Minor,
    /// Moderate deviation, attention recommended.
    Moderate,
    /// Major deviation, correction needed.
    Major,
    /// Critical deviation, calibration failed.
    Critical,
}

/// A single finding or observation from calibration.
#[derive(Debug, Clone)]
pub struct CalibrationFinding {
    /// Descriptive label for this finding.
    pub label: String,
    /// Severity of the finding.
    pub severity: FindingSeverity,
    /// Measured value.
    pub measured: f64,
    /// Expected/reference value.
    pub expected: f64,
    /// Deviation (measured - expected).
    pub deviation: f64,
    /// Tolerance threshold.
    pub tolerance: f64,
}

impl CalibrationFinding {
    /// Create a new calibration finding.
    #[must_use]
    pub fn new(label: &str, measured: f64, expected: f64, tolerance: f64) -> Self {
        let deviation = measured - expected;
        let severity = if deviation.abs() <= tolerance * 0.5 {
            FindingSeverity::Info
        } else if deviation.abs() <= tolerance {
            FindingSeverity::Minor
        } else if deviation.abs() <= tolerance * 1.5 {
            FindingSeverity::Moderate
        } else if deviation.abs() <= tolerance * 2.0 {
            FindingSeverity::Major
        } else {
            FindingSeverity::Critical
        };
        Self {
            label: label.to_string(),
            severity,
            measured,
            expected,
            deviation,
            tolerance,
        }
    }

    /// Check if this finding passes its tolerance.
    #[must_use]
    pub fn passes(&self) -> bool {
        self.deviation.abs() <= self.tolerance
    }
}

/// Color accuracy metrics for a calibration session.
#[derive(Debug, Clone)]
pub struct ColorAccuracyMetrics {
    /// Average Delta E (CIE2000) across all patches.
    pub avg_delta_e: f64,
    /// Maximum Delta E across all patches.
    pub max_delta_e: f64,
    /// Median Delta E.
    pub median_delta_e: f64,
    /// Standard deviation of Delta E.
    pub std_delta_e: f64,
    /// Number of patches measured.
    pub patch_count: usize,
    /// Number of patches within tolerance.
    pub patches_in_tolerance: usize,
    /// Delta E values per patch.
    pub per_patch: Vec<f64>,
}

impl ColorAccuracyMetrics {
    /// Compute color accuracy metrics from per-patch Delta E values.
    #[must_use]
    pub fn from_delta_e_values(values: &[f64], tolerance: f64) -> Self {
        if values.is_empty() {
            return Self {
                avg_delta_e: 0.0,
                max_delta_e: 0.0,
                median_delta_e: 0.0,
                std_delta_e: 0.0,
                patch_count: 0,
                patches_in_tolerance: 0,
                per_patch: Vec::new(),
            };
        }

        let n = values.len() as f64;
        let avg = values.iter().sum::<f64>() / n;
        let max = values.iter().copied().fold(0.0_f64, f64::max);
        let variance = values.iter().map(|v| (v - avg).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();
        let in_tol = values.iter().filter(|&&v| v <= tolerance).count();

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = if sorted.len() % 2 == 0 {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) * 0.5
        } else {
            sorted[sorted.len() / 2]
        };

        Self {
            avg_delta_e: avg,
            max_delta_e: max,
            median_delta_e: median,
            std_delta_e: std_dev,
            patch_count: values.len(),
            patches_in_tolerance: in_tol,
            per_patch: values.to_vec(),
        }
    }

    /// Compute percentage of patches within tolerance.
    #[must_use]
    pub fn pass_rate(&self) -> f64 {
        if self.patch_count == 0 {
            return 0.0;
        }
        self.patches_in_tolerance as f64 / self.patch_count as f64 * 100.0
    }
}

/// Gray balance metrics.
#[derive(Debug, Clone)]
pub struct GrayBalanceMetrics {
    /// Average a* deviation across neutral patches.
    pub avg_a_deviation: f64,
    /// Average b* deviation across neutral patches.
    pub avg_b_deviation: f64,
    /// Maximum chromaticity error in neutral patches.
    pub max_chroma_error: f64,
    /// Number of neutral patches measured.
    pub neutral_patch_count: usize,
}

/// Calibration report for a complete session.
#[derive(Debug, Clone)]
pub struct CalibrationReport {
    /// Report title.
    pub title: String,
    /// Overall verdict.
    pub verdict: CalibrationVerdict,
    /// Color accuracy metrics.
    pub color_accuracy: ColorAccuracyMetrics,
    /// Gray balance metrics.
    pub gray_balance: Option<GrayBalanceMetrics>,
    /// Individual findings.
    pub findings: Vec<CalibrationFinding>,
    /// Custom metadata key-value pairs.
    pub metadata: HashMap<String, String>,
    /// Timestamp of the calibration (ISO 8601 string).
    pub timestamp: String,
}

/// Builder for constructing calibration reports.
#[derive(Debug)]
pub struct ReportBuilder {
    /// Report title.
    title: String,
    /// Delta E tolerance for pass/fail.
    delta_e_tolerance: f64,
    /// Gray balance tolerance.
    gray_tolerance: f64,
    /// Collected Delta E values.
    delta_e_values: Vec<f64>,
    /// Collected findings.
    findings: Vec<CalibrationFinding>,
    /// Gray balance data (a*, b* deviations per neutral patch).
    gray_data: Vec<(f64, f64)>,
    /// Metadata.
    metadata: HashMap<String, String>,
    /// Timestamp.
    timestamp: String,
}

impl ReportBuilder {
    /// Create a new report builder.
    #[must_use]
    pub fn new(title: &str) -> Self {
        Self {
            title: title.to_string(),
            delta_e_tolerance: 2.0,
            gray_tolerance: 1.0,
            delta_e_values: Vec::new(),
            findings: Vec::new(),
            gray_data: Vec::new(),
            metadata: HashMap::new(),
            timestamp: String::new(),
        }
    }

    /// Set the Delta E tolerance.
    #[must_use]
    pub fn with_delta_e_tolerance(mut self, tolerance: f64) -> Self {
        self.delta_e_tolerance = tolerance.max(0.01);
        self
    }

    /// Set the gray balance tolerance.
    #[must_use]
    pub fn with_gray_tolerance(mut self, tolerance: f64) -> Self {
        self.gray_tolerance = tolerance.max(0.01);
        self
    }

    /// Set the timestamp.
    #[must_use]
    pub fn with_timestamp(mut self, ts: &str) -> Self {
        self.timestamp = ts.to_string();
        self
    }

    /// Add Delta E values for color patches.
    #[must_use]
    pub fn with_delta_e_values(mut self, values: Vec<f64>) -> Self {
        self.delta_e_values = values;
        self
    }

    /// Add a custom finding.
    #[must_use]
    pub fn add_finding(mut self, finding: CalibrationFinding) -> Self {
        self.findings.push(finding);
        self
    }

    /// Add gray balance data (a*, b* deviation pairs).
    #[must_use]
    pub fn with_gray_data(mut self, data: Vec<(f64, f64)>) -> Self {
        self.gray_data = data;
        self
    }

    /// Add metadata key-value pair.
    #[must_use]
    pub fn add_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }

    /// Build the calibration report.
    #[must_use]
    pub fn build(self) -> CalibrationReport {
        let color_accuracy =
            ColorAccuracyMetrics::from_delta_e_values(&self.delta_e_values, self.delta_e_tolerance);

        let gray_balance = if self.gray_data.is_empty() {
            None
        } else {
            let n = self.gray_data.len() as f64;
            let avg_a = self.gray_data.iter().map(|(a, _)| a.abs()).sum::<f64>() / n;
            let avg_b = self.gray_data.iter().map(|(_, b)| b.abs()).sum::<f64>() / n;
            let max_chroma = self
                .gray_data
                .iter()
                .map(|(a, b)| (a * a + b * b).sqrt())
                .fold(0.0_f64, f64::max);
            Some(GrayBalanceMetrics {
                avg_a_deviation: avg_a,
                avg_b_deviation: avg_b,
                max_chroma_error: max_chroma,
                neutral_patch_count: self.gray_data.len(),
            })
        };

        // Determine verdict
        let has_critical = self
            .findings
            .iter()
            .any(|f| f.severity >= FindingSeverity::Critical);
        let has_major = self
            .findings
            .iter()
            .any(|f| f.severity >= FindingSeverity::Major);
        let color_pass = color_accuracy.avg_delta_e <= self.delta_e_tolerance;

        let verdict = if has_critical || !color_pass {
            CalibrationVerdict::Fail
        } else if has_major {
            CalibrationVerdict::Warning
        } else {
            CalibrationVerdict::Pass
        };

        CalibrationReport {
            title: self.title,
            verdict,
            color_accuracy,
            gray_balance,
            findings: self.findings,
            metadata: self.metadata,
            timestamp: self.timestamp,
        }
    }
}

/// Format a calibration report as a plain-text summary string.
#[must_use]
pub fn format_report_summary(report: &CalibrationReport) -> String {
    let mut lines = Vec::new();
    lines.push(format!("=== {} ===", report.title));
    lines.push(format!("Verdict: {:?}", report.verdict));
    lines.push(format!(
        "Avg Delta E: {:.2}",
        report.color_accuracy.avg_delta_e
    ));
    lines.push(format!(
        "Max Delta E: {:.2}",
        report.color_accuracy.max_delta_e
    ));
    lines.push(format!(
        "Pass rate: {:.1}%",
        report.color_accuracy.pass_rate()
    ));
    if let Some(ref gb) = report.gray_balance {
        lines.push(format!(
            "Gray balance: avg a*={:.3}, avg b*={:.3}, max chroma={:.3}",
            gb.avg_a_deviation, gb.avg_b_deviation, gb.max_chroma_error,
        ));
    }
    lines.push(format!("Findings: {}", report.findings.len()));
    for f in &report.findings {
        lines.push(format!(
            "  [{:?}] {}: measured={:.3}, expected={:.3}, dev={:.3}",
            f.severity, f.label, f.measured, f.expected, f.deviation,
        ));
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_finding_passes() {
        let f = CalibrationFinding::new("test", 5.0, 5.0, 1.0);
        assert!(f.passes());
    }

    #[test]
    fn test_finding_fails() {
        let f = CalibrationFinding::new("test", 7.0, 5.0, 1.0);
        assert!(!f.passes());
    }

    #[test]
    fn test_finding_severity_auto() {
        let info = CalibrationFinding::new("ok", 5.1, 5.0, 1.0);
        assert_eq!(info.severity, FindingSeverity::Info);
        let critical = CalibrationFinding::new("bad", 10.0, 5.0, 1.0);
        assert_eq!(critical.severity, FindingSeverity::Critical);
    }

    #[test]
    fn test_color_accuracy_empty() {
        let m = ColorAccuracyMetrics::from_delta_e_values(&[], 2.0);
        assert_eq!(m.patch_count, 0);
        assert!((m.avg_delta_e - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_color_accuracy_single() {
        let m = ColorAccuracyMetrics::from_delta_e_values(&[1.5], 2.0);
        assert_eq!(m.patch_count, 1);
        assert!((m.avg_delta_e - 1.5).abs() < 1e-10);
        assert!((m.median_delta_e - 1.5).abs() < 1e-10);
        assert_eq!(m.patches_in_tolerance, 1);
    }

    #[test]
    fn test_color_accuracy_metrics() {
        let values = vec![1.0, 1.5, 2.0, 2.5, 3.0];
        let m = ColorAccuracyMetrics::from_delta_e_values(&values, 2.0);
        assert!((m.avg_delta_e - 2.0).abs() < 1e-10);
        assert!((m.max_delta_e - 3.0).abs() < 1e-10);
        assert!((m.median_delta_e - 2.0).abs() < 1e-10);
        assert_eq!(m.patches_in_tolerance, 3);
    }

    #[test]
    fn test_pass_rate() {
        let m = ColorAccuracyMetrics::from_delta_e_values(&[1.0, 2.0, 3.0, 4.0], 2.5);
        assert!((m.pass_rate() - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_report_builder_pass() {
        let report = ReportBuilder::new("Test Calibration")
            .with_delta_e_tolerance(3.0)
            .with_delta_e_values(vec![1.0, 1.5, 2.0])
            .with_timestamp("2026-03-02T12:00:00Z")
            .build();
        assert_eq!(report.verdict, CalibrationVerdict::Pass);
        assert_eq!(report.color_accuracy.patch_count, 3);
    }

    #[test]
    fn test_report_builder_fail() {
        let report = ReportBuilder::new("Failing Calibration")
            .with_delta_e_tolerance(1.0)
            .with_delta_e_values(vec![3.0, 4.0, 5.0])
            .build();
        assert_eq!(report.verdict, CalibrationVerdict::Fail);
    }

    #[test]
    fn test_report_builder_warning() {
        let report = ReportBuilder::new("Warning Calibration")
            .with_delta_e_tolerance(5.0)
            .with_delta_e_values(vec![1.0, 2.0])
            .add_finding(CalibrationFinding::new("major_issue", 10.0, 5.0, 2.5))
            .build();
        assert_eq!(report.verdict, CalibrationVerdict::Warning);
    }

    #[test]
    fn test_report_gray_balance() {
        let report = ReportBuilder::new("Gray Test")
            .with_delta_e_tolerance(5.0)
            .with_delta_e_values(vec![1.0])
            .with_gray_data(vec![(0.5, -0.3), (0.2, 0.1)])
            .build();
        assert!(report.gray_balance.is_some());
        let gb = report
            .gray_balance
            .expect("expected gray_balance to be Some/Ok");
        assert_eq!(gb.neutral_patch_count, 2);
        assert!(gb.max_chroma_error > 0.0);
    }

    #[test]
    fn test_report_metadata() {
        let report = ReportBuilder::new("Meta Test")
            .with_delta_e_tolerance(5.0)
            .with_delta_e_values(vec![1.0])
            .add_metadata("camera", "Sony A7")
            .add_metadata("illuminant", "D65")
            .build();
        assert_eq!(
            report
                .metadata
                .get("camera")
                .expect("expected key to exist"),
            "Sony A7"
        );
        assert_eq!(
            report
                .metadata
                .get("illuminant")
                .expect("expected key to exist"),
            "D65"
        );
    }

    #[test]
    fn test_format_report_summary() {
        let report = ReportBuilder::new("Summary Test")
            .with_delta_e_tolerance(3.0)
            .with_delta_e_values(vec![1.0, 2.0])
            .add_finding(CalibrationFinding::new("white_point", 6505.0, 6500.0, 50.0))
            .build();
        let summary = format_report_summary(&report);
        assert!(summary.contains("Summary Test"));
        assert!(summary.contains("Pass"));
        assert!(summary.contains("white_point"));
    }

    #[test]
    fn test_finding_severity_ordering() {
        assert!(FindingSeverity::Info < FindingSeverity::Minor);
        assert!(FindingSeverity::Minor < FindingSeverity::Moderate);
        assert!(FindingSeverity::Moderate < FindingSeverity::Major);
        assert!(FindingSeverity::Major < FindingSeverity::Critical);
    }
}
