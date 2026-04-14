//! Automated QC (Quality Control) reporting for IMF packages.
//!
//! [`QcReporter`] performs EBU-style QC checks on an IMF package structure
//! and produces a [`QcReport`] listing every pass, warning, and failure.
//!
//! # Example
//! ```no_run
//! use oximedia_imf::qc_report::QcReporter;
//!
//! let report = QcReporter::check("/path/to/imp");
//! println!("{}", report.summary());
//! ```

#![allow(dead_code, missing_docs)]

use std::path::Path;

/// Severity level of a QC finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    Info,
    Warning,
    Error,
}

impl std::fmt::Display for Severity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Info => write!(f, "INFO"),
            Self::Warning => write!(f, "WARN"),
            Self::Error => write!(f, "ERROR"),
        }
    }
}

/// A single QC finding.
#[derive(Debug, Clone)]
pub struct QcFinding {
    pub severity: Severity,
    pub code: String,
    pub message: String,
    pub context: Option<String>,
}

impl QcFinding {
    #[must_use]
    pub fn new(
        severity: Severity,
        code: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            severity,
            code: code.into(),
            message: message.into(),
            context: None,
        }
    }

    #[must_use]
    pub fn with_context(mut self, ctx: impl Into<String>) -> Self {
        self.context = Some(ctx.into());
        self
    }
}

impl std::fmt::Display for QcFinding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(ref ctx) = self.context {
            write!(f, "[{}] {} — {} ({})", self.severity, self.code, self.message, ctx)
        } else {
            write!(f, "[{}] {} — {}", self.severity, self.code, self.message)
        }
    }
}

/// Full QC report for an IMF package.
#[derive(Debug, Clone, Default)]
pub struct QcReport {
    pub package_path: String,
    pub findings: Vec<QcFinding>,
}

impl QcReport {
    #[must_use]
    pub fn new(package_path: impl Into<String>) -> Self {
        Self { package_path: package_path.into(), findings: Vec::new() }
    }

    pub fn add(&mut self, finding: QcFinding) {
        self.findings.push(finding);
    }

    pub fn error(&mut self, code: &str, msg: impl Into<String>) {
        self.add(QcFinding::new(Severity::Error, code, msg));
    }

    pub fn warn(&mut self, code: &str, msg: impl Into<String>) {
        self.add(QcFinding::new(Severity::Warning, code, msg));
    }

    pub fn info(&mut self, code: &str, msg: impl Into<String>) {
        self.add(QcFinding::new(Severity::Info, code, msg));
    }

    #[must_use]
    pub fn is_pass(&self) -> bool {
        !self.findings.iter().any(|f| f.severity == Severity::Error)
    }

    #[must_use]
    pub fn errors(&self) -> Vec<&QcFinding> {
        self.findings.iter().filter(|f| f.severity == Severity::Error).collect()
    }

    #[must_use]
    pub fn warnings(&self) -> Vec<&QcFinding> {
        self.findings.iter().filter(|f| f.severity == Severity::Warning).collect()
    }

    #[must_use]
    pub fn summary(&self) -> String {
        let errors = self.errors().len();
        let warnings = self.warnings().len();
        let infos = self.findings.iter().filter(|f| f.severity == Severity::Info).count();
        let status = if errors == 0 { "PASS" } else { "FAIL" };
        format!("QC {status}: {errors} error(s), {warnings} warning(s), {infos} info(s) — {}", self.package_path)
    }

    #[must_use]
    pub fn render(&self) -> String {
        let mut out = format!("=== IMF QC Report: {} ===\n", self.package_path);
        for f in &self.findings {
            out.push_str(&format!("  {f}\n"));
        }
        out.push_str(&format!("\n{}\n", self.summary()));
        out
    }
}

/// Performs QC checks on an IMF package directory.
pub struct QcReporter;

impl QcReporter {
    #[must_use]
    pub fn check(pkg_path: impl AsRef<Path>) -> QcReport {
        let pkg_path = pkg_path.as_ref();
        let path_str = pkg_path.to_string_lossy().to_string();
        let mut report = QcReport::new(&path_str);

        if !pkg_path.exists() {
            report.error("PKG-001", format!("Package directory not found: '{path_str}'"));
            return report;
        }
        if !pkg_path.is_dir() {
            report.error("PKG-001", format!("Package path is not a directory: '{path_str}'"));
            return report;
        }
        report.info("PKG-001", "Package directory exists");

        let am_path = pkg_path.join("ASSETMAP.xml");
        if !am_path.exists() {
            report.warn("PKG-002", "ASSETMAP.xml not found");
        } else {
            report.info("PKG-002", "ASSETMAP.xml present");
            Self::check_xml_wellformed(&am_path, "PKG-002", &mut report);
        }

        let pkl_count = count_xml_prefix(pkg_path, "PKL");
        if pkl_count == 0 {
            report.warn("PKG-003", "No PKL*.xml found");
        } else {
            report.info("PKG-003", format!("{pkl_count} PKL file(s) found"));
        }

        let cpl_count = count_xml_prefix(pkg_path, "CPL");
        if cpl_count == 0 {
            report.error("PKG-004", "No CPL*.xml found — composition playlist required");
        } else {
            report.info("PKG-004", format!("{cpl_count} CPL file(s) found"));
        }

        if let Ok(entries) = std::fs::read_dir(pkg_path) {
            for entry in entries.flatten() {
                let fp = entry.path();
                if fp.extension().and_then(|e| e.to_str()) == Some("mxf") {
                    if let Ok(m) = std::fs::metadata(&fp) {
                        if m.len() == 0 {
                            report.error("PKG-005", format!("MXF file is empty: {}", fp.display()));
                        }
                    }
                }
            }
        }

        report
    }

    fn check_xml_wellformed(path: &Path, code: &str, report: &mut QcReport) {
        match std::fs::read(path) {
            Err(e) => report.error(code, format!("Cannot read {}: {e}", path.display())),
            Ok(bytes) => {
                let s = String::from_utf8_lossy(&bytes[..bytes.len().min(20)]);
                if !s.trim_start().starts_with('<') {
                    report.error(code, format!("{} is not well-formed XML", path.display()));
                }
            }
        }
    }
}

fn count_xml_prefix(dir: &Path, prefix: &str) -> usize {
    let Ok(entries) = std::fs::read_dir(dir) else { return 0 };
    entries
        .flatten()
        .filter(|e| {
            let n = e.file_name().to_string_lossy().to_uppercase();
            n.starts_with(&prefix.to_uppercase()) && n.ends_with(".XML")
        })
        .count()
}

// ============================================================================
// IMF QC runner (task API) — structured check-based reporting
// ============================================================================

/// Category of a QC check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QcCategory {
    /// Package structural completeness
    Structure,
    /// CPL / PKL metadata validity
    Metadata,
    /// Essence track constraints
    Essence,
    /// Timeline and duration constraints
    Timing,
    /// Application profile compliance
    Profile,
}

impl std::fmt::Display for QcCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Structure => write!(f, "Structure"),
            Self::Metadata => write!(f, "Metadata"),
            Self::Essence => write!(f, "Essence"),
            Self::Timing => write!(f, "Timing"),
            Self::Profile => write!(f, "Profile"),
        }
    }
}

/// Severity of a QC result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum QcSeverity {
    /// Must be fixed; package is non-conformant
    Critical,
    /// Significant deviation; likely interop issues
    Major,
    /// Minor deviation; informational
    Minor,
    /// Purely informational, no action needed
    Info,
}

impl std::fmt::Display for QcSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Critical => write!(f, "CRITICAL"),
            Self::Major => write!(f, "MAJOR"),
            Self::Minor => write!(f, "MINOR"),
            Self::Info => write!(f, "INFO"),
        }
    }
}

/// A defined QC check.
#[derive(Debug, Clone)]
pub struct QcCheck {
    /// Unique check identifier (e.g. "QC-001")
    pub id: String,
    /// Human-readable description
    pub description: String,
    /// Category this check belongs to
    pub category: QcCategory,
}

impl QcCheck {
    /// Create a new [`QcCheck`].
    pub fn new(id: impl Into<String>, description: impl Into<String>, category: QcCategory) -> Self {
        Self {
            id: id.into(),
            description: description.into(),
            category,
        }
    }
}

/// Result of executing a [`QcCheck`].
#[derive(Debug, Clone)]
pub struct QcResult {
    /// The check that was run
    pub check: QcCheck,
    /// Whether the check passed
    pub passed: bool,
    /// Severity if the check failed (ignored when `passed == true`)
    pub severity: QcSeverity,
    /// Optional details string
    pub details: Option<String>,
}

impl QcResult {
    /// Create a passing result.
    pub fn pass(check: QcCheck) -> Self {
        Self { check, passed: true, severity: QcSeverity::Info, details: None }
    }

    /// Create a failing result with the given severity.
    pub fn fail(check: QcCheck, severity: QcSeverity, details: impl Into<String>) -> Self {
        Self { check, passed: false, severity, details: Some(details.into()) }
    }
}

/// Structured QC report for an IMF package.
#[derive(Debug, Clone)]
pub struct ImfQcReport {
    /// Package identifier (UUID string or path)
    pub package_id: String,
    /// All check results
    pub checks: Vec<QcResult>,
    /// Unix timestamp (seconds) when the report was generated
    pub generated_at_secs: u64,
}

impl ImfQcReport {
    /// Create a new, empty report.
    pub fn new(package_id: impl Into<String>) -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        let generated_at_secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        Self { package_id: package_id.into(), checks: Vec::new(), generated_at_secs }
    }

    /// All critical failures.
    pub fn critical_failures(&self) -> Vec<&QcResult> {
        self.checks
            .iter()
            .filter(|r| !r.passed && r.severity == QcSeverity::Critical)
            .collect()
    }

    /// Fraction of checks that passed (0.0 – 1.0).
    pub fn pass_rate(&self) -> f32 {
        if self.checks.is_empty() {
            return 1.0;
        }
        let passed = self.checks.iter().filter(|r| r.passed).count();
        passed as f32 / self.checks.len() as f32
    }

    /// One-line human-readable summary.
    pub fn summary(&self) -> String {
        let total = self.checks.len();
        let passed = self.checks.iter().filter(|r| r.passed).count();
        let failed = total - passed;
        let criticals = self.critical_failures().len();
        format!(
            "IMF QC [{pkg}] — {passed}/{total} passed, {failed} failed ({criticals} critical)",
            pkg = self.package_id,
        )
    }
}

/// Runs a standard set of IMF QC checks.
pub struct ImfQcRunner;

impl ImfQcRunner {
    /// Run the built-in check suite and return a structured [`ImfQcReport`].
    ///
    /// # Parameters
    /// - `package_id` — identifier for the package (UUID or path string)
    /// - `cpl_present` — whether at least one CPL with a video sequence was found
    /// - `pkl_has_hashes` — whether the PKL contains hash entries for all assets
    /// - `asset_count` — number of assets referenced in the asset map
    /// - `edit_rate` — (numerator, denominator) of the composition edit rate
    /// - `duration_frames` — total duration in frames
    pub fn run_basic_checks(
        package_id: impl Into<String>,
        cpl_present: bool,
        pkl_has_hashes: bool,
        asset_count: usize,
        edit_rate: (u32, u32),
        duration_frames: u64,
    ) -> ImfQcReport {
        let mut report = ImfQcReport::new(package_id);

        // QC-001: CPL has at least one video sequence
        {
            let check = QcCheck::new("QC-001", "CPL must contain at least one video sequence", QcCategory::Structure);
            if cpl_present {
                report.checks.push(QcResult::pass(check));
            } else {
                report.checks.push(QcResult::fail(check, QcSeverity::Critical, "No MainImageSequence found in CPL"));
            }
        }

        // QC-002: PKL hash entries present
        {
            let check = QcCheck::new("QC-002", "PKL must have hash entries for all assets", QcCategory::Metadata);
            if pkl_has_hashes {
                report.checks.push(QcResult::pass(check));
            } else {
                report.checks.push(QcResult::fail(check, QcSeverity::Major, "PKL is missing hash entries"));
            }
        }

        // QC-003: Asset map references at least one asset
        {
            let check = QcCheck::new("QC-003", "Asset map must reference at least one package file", QcCategory::Structure);
            if asset_count > 0 {
                report.checks.push(QcResult::pass(check));
            } else {
                report.checks.push(QcResult::fail(check, QcSeverity::Critical, "Asset map is empty"));
            }
        }

        // QC-004: Edit rate is a recognised broadcast/streaming standard
        {
            const STANDARD_RATES: &[(u32, u32)] = &[
                (24, 1), (25, 1), (30, 1), (48, 1), (50, 1), (60, 1),
                (24000, 1001), (30000, 1001), (60000, 1001),
            ];
            let check = QcCheck::new(
                "QC-004",
                "Edit rate must be a standard broadcast rate",
                QcCategory::Profile,
            );
            if STANDARD_RATES.contains(&edit_rate) {
                report.checks.push(QcResult::pass(check));
            } else {
                report.checks.push(QcResult::fail(
                    check,
                    QcSeverity::Minor,
                    format!("Non-standard edit rate {}/{}", edit_rate.0, edit_rate.1),
                ));
            }
        }

        // QC-005: Duration must be > 0
        {
            let check = QcCheck::new("QC-005", "Composition duration must be greater than zero", QcCategory::Timing);
            if duration_frames > 0 {
                report.checks.push(QcResult::pass(check));
            } else {
                report.checks.push(QcResult::fail(check, QcSeverity::Critical, "Duration is zero"));
            }
        }

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nonexistent_path() {
        let r = QcReporter::check("/nonexistent_xyz_test");
        assert!(!r.is_pass());
        assert!(r.errors().iter().any(|e| e.code == "PKG-001"));
    }

    #[test]
    fn test_not_a_directory() {
        let p = std::env::temp_dir().join("oximedia_qc_notdir_test.txt");
        std::fs::write(&p, b"x").ok();
        let r = QcReporter::check(&p);
        assert!(!r.is_pass());
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn test_empty_dir_fails() {
        let dir = std::env::temp_dir().join("oximedia_qc_emptydir");
        std::fs::create_dir_all(&dir).ok();
        let r = QcReporter::check(&dir);
        assert!(!r.is_pass());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_minimal_package_passes() {
        let dir = std::env::temp_dir().join("oximedia_qc_minpkg");
        std::fs::create_dir_all(&dir).ok();
        std::fs::write(dir.join("ASSETMAP.xml"), b"<AssetMap/>").ok();
        std::fs::write(dir.join("PKL_001.xml"), b"<PackingList/>").ok();
        std::fs::write(dir.join("CPL_001.xml"), b"<CompositionPlaylist/>").ok();
        let r = QcReporter::check(&dir);
        assert!(r.is_pass(), "{}", r.summary());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_finding_display() {
        let f = QcFinding::new(Severity::Error, "E001", "test error");
        assert!(format!("{f}").contains("ERROR"));
        assert!(format!("{f}").contains("E001"));
    }

    #[test]
    fn test_report_summary_pass() {
        let mut r = QcReport::new("/pkg");
        r.info("I001", "all good");
        assert!(r.is_pass());
        assert!(r.summary().contains("PASS"));
    }

    // -----------------------------------------------------------------------
    // ImfQcRunner (task API) tests
    // -----------------------------------------------------------------------

    fn good_report() -> ImfQcReport {
        ImfQcRunner::run_basic_checks(
            "urn:uuid:test-pkg-0000-0000-000000000001",
            true,   // cpl_present
            true,   // pkl_has_hashes
            5,      // asset_count
            (24, 1), // edit_rate
            2400,   // duration_frames
        )
    }

    #[test]
    fn test_imf_qc_all_pass() {
        let report = good_report();
        assert_eq!(report.critical_failures().len(), 0);
        assert!((report.pass_rate() - 1.0).abs() < f32::EPSILON);
        assert!(report.summary().contains("5/5"));
    }

    #[test]
    fn test_imf_qc_critical_failure_no_video() {
        let report = ImfQcRunner::run_basic_checks(
            "pkg-no-video",
            false, // cpl_present — no video sequence
            true,  // pkl_has_hashes
            3,     // asset_count
            (25, 1), 1000,
        );
        let criticals = report.critical_failures();
        assert!(!criticals.is_empty());
        assert!(criticals.iter().any(|r| r.check.id == "QC-001"));
    }

    #[test]
    fn test_imf_qc_minor_non_standard_rate() {
        let report = ImfQcRunner::run_basic_checks(
            "pkg-nonstandard",
            true, true, 3,
            (15, 1), // non-standard
            300,
        );
        let minor_fails: Vec<_> = report.checks.iter()
            .filter(|r| !r.passed && r.severity == QcSeverity::Minor)
            .collect();
        assert!(!minor_fails.is_empty(), "expected minor failure for non-standard rate");
        // No criticals expected
        assert_eq!(report.critical_failures().len(), 0);
    }

    #[test]
    fn test_imf_qc_pass_rate_calculation() {
        let report = ImfQcRunner::run_basic_checks(
            "pkg-partial",
            false, // QC-001 fails (critical)
            true, 2,
            (24, 1), 100,
        );
        let total = report.checks.len();
        let passed = report.checks.iter().filter(|r| r.passed).count();
        let expected_rate = passed as f32 / total as f32;
        assert!((report.pass_rate() - expected_rate).abs() < f32::EPSILON);
    }

    #[test]
    fn test_imf_qc_duration_zero_is_critical() {
        let report = ImfQcRunner::run_basic_checks(
            "pkg-zero-dur",
            true, true, 5,
            (24, 1), 0, // zero duration
        );
        let criticals = report.critical_failures();
        assert!(criticals.iter().any(|r| r.check.id == "QC-005"));
    }

    #[test]
    fn test_imf_qc_empty_asset_map_critical() {
        let report = ImfQcRunner::run_basic_checks(
            "pkg-no-assets",
            true, true,
            0, // asset_count = 0
            (24, 1), 1000,
        );
        let criticals = report.critical_failures();
        assert!(criticals.iter().any(|r| r.check.id == "QC-003"));
    }

    #[test]
    fn test_imf_qc_report_summary_format() {
        let report = good_report();
        let s = report.summary();
        assert!(s.contains("IMF QC"));
        assert!(s.contains("passed"));
    }

    #[test]
    fn test_qc_severity_ordering() {
        assert!(QcSeverity::Critical < QcSeverity::Major);
        assert!(QcSeverity::Major < QcSeverity::Minor);
        assert!(QcSeverity::Minor < QcSeverity::Info);
    }

    #[test]
    fn test_qc_category_display() {
        assert_eq!(QcCategory::Structure.to_string(), "Structure");
        assert_eq!(QcCategory::Timing.to_string(), "Timing");
    }
}
