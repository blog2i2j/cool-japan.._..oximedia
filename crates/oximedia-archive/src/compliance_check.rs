#![allow(dead_code)]
//! Regulatory compliance checking for archived media.
//!
//! Validates that archived media assets meet various regulatory and industry
//! standards including GDPR retention, broadcast compliance, and content
//! classification requirements.

use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;
use std::time::{Duration, SystemTime};

/// Compliance standard that an archive must conform to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ComplianceStandard {
    /// General Data Protection Regulation data retention rules.
    Gdpr,
    /// Health Insurance Portability and Accountability Act.
    Hipaa,
    /// Broadcast content compliance (FCC / Ofcom style).
    BroadcastContent,
    /// Sarbanes-Oxley financial record retention.
    Sox,
    /// Payment Card Industry Data Security Standard.
    PciDss,
    /// Custom organizational policy.
    Custom,
}

impl fmt::Display for ComplianceStandard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gdpr => write!(f, "GDPR"),
            Self::Hipaa => write!(f, "HIPAA"),
            Self::BroadcastContent => write!(f, "Broadcast Content"),
            Self::Sox => write!(f, "SOX"),
            Self::PciDss => write!(f, "PCI-DSS"),
            Self::Custom => write!(f, "Custom"),
        }
    }
}

/// Severity level for compliance violations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    /// Informational note, no action required.
    Info,
    /// Warning that should be reviewed.
    Warning,
    /// Non-critical violation requiring remediation.
    Minor,
    /// Significant violation requiring prompt action.
    Major,
    /// Critical violation requiring immediate action.
    Critical,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Info => write!(f, "INFO"),
            Self::Warning => write!(f, "WARNING"),
            Self::Minor => write!(f, "MINOR"),
            Self::Major => write!(f, "MAJOR"),
            Self::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// A single compliance violation found during checking.
#[derive(Debug, Clone)]
pub struct ComplianceViolation {
    /// The standard that was violated.
    pub standard: ComplianceStandard,
    /// Severity of the violation.
    pub severity: Severity,
    /// Path to the offending asset (if applicable).
    pub asset_path: Option<PathBuf>,
    /// Human-readable description of the violation.
    pub description: String,
    /// Suggested remediation.
    pub remediation: String,
    /// Unique rule identifier.
    pub rule_id: String,
}

impl fmt::Display for ComplianceViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] {} ({}): {}",
            self.severity, self.standard, self.rule_id, self.description
        )
    }
}

/// Retention policy for a class of archived assets.
#[derive(Debug, Clone)]
pub struct RetentionPolicy {
    /// Name of the retention class.
    pub name: String,
    /// Minimum retention duration.
    pub min_retention: Duration,
    /// Maximum retention duration (after which data must be deleted).
    pub max_retention: Option<Duration>,
    /// Which compliance standard this policy supports.
    pub standard: ComplianceStandard,
    /// Whether legal hold can override the max retention.
    pub legal_hold_override: bool,
}

impl RetentionPolicy {
    /// Creates a new retention policy.
    pub fn new(name: &str, standard: ComplianceStandard, min_retention: Duration) -> Self {
        Self {
            name: name.to_string(),
            min_retention,
            max_retention: None,
            standard,
            legal_hold_override: true,
        }
    }

    /// Sets the maximum retention period.
    pub fn with_max_retention(mut self, max: Duration) -> Self {
        self.max_retention = Some(max);
        self
    }

    /// Sets whether legal hold can override max retention.
    pub fn with_legal_hold_override(mut self, allow: bool) -> Self {
        self.legal_hold_override = allow;
        self
    }

    /// Checks if an asset with the given creation time is within retention bounds.
    pub fn check_retention(&self, created_at: SystemTime) -> RetentionStatus {
        let age = created_at.elapsed().unwrap_or(Duration::ZERO);
        if age < self.min_retention {
            RetentionStatus::WithinRetention
        } else if let Some(max) = self.max_retention {
            if age > max {
                RetentionStatus::ExceededMaxRetention
            } else {
                RetentionStatus::WithinRetention
            }
        } else {
            RetentionStatus::WithinRetention
        }
    }
}

/// Status of an asset's retention compliance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetentionStatus {
    /// The asset is within the allowed retention window.
    WithinRetention,
    /// The asset has exceeded the maximum retention period.
    ExceededMaxRetention,
    /// The asset is under legal hold and retention is suspended.
    LegalHold,
}

impl fmt::Display for RetentionStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WithinRetention => write!(f, "Within Retention"),
            Self::ExceededMaxRetention => write!(f, "Exceeded Max Retention"),
            Self::LegalHold => write!(f, "Legal Hold"),
        }
    }
}

/// An asset subject to compliance checking.
#[derive(Debug, Clone)]
pub struct ComplianceAsset {
    /// Path to the asset.
    pub path: PathBuf,
    /// When the asset was archived.
    pub archived_at: SystemTime,
    /// Content classification tags.
    pub tags: Vec<String>,
    /// Whether this asset is under legal hold.
    pub legal_hold: bool,
    /// Size of the asset in bytes.
    pub size_bytes: u64,
    /// Whether the asset has been encrypted.
    pub encrypted: bool,
    /// Whether access logging is enabled.
    pub access_logged: bool,
}

impl ComplianceAsset {
    /// Creates a new compliance asset.
    pub fn new(path: PathBuf, archived_at: SystemTime, size_bytes: u64) -> Self {
        Self {
            path,
            archived_at,
            tags: Vec::new(),
            legal_hold: false,
            size_bytes,
            encrypted: false,
            access_logged: false,
        }
    }

    /// Adds a classification tag.
    pub fn with_tag(mut self, tag: &str) -> Self {
        self.tags.push(tag.to_string());
        self
    }

    /// Sets the legal hold flag.
    pub fn with_legal_hold(mut self, hold: bool) -> Self {
        self.legal_hold = hold;
        self
    }

    /// Sets the encrypted flag.
    pub fn with_encrypted(mut self, encrypted: bool) -> Self {
        self.encrypted = encrypted;
        self
    }

    /// Sets access logging flag.
    pub fn with_access_logged(mut self, logged: bool) -> Self {
        self.access_logged = logged;
        self
    }
}

/// Result of a compliance check run.
#[derive(Debug, Clone)]
pub struct ComplianceReport {
    /// When the check was performed.
    pub checked_at: SystemTime,
    /// Violations found.
    pub violations: Vec<ComplianceViolation>,
    /// Number of assets checked.
    pub assets_checked: u64,
    /// Standards that were evaluated.
    pub standards_evaluated: Vec<ComplianceStandard>,
}

impl ComplianceReport {
    /// Creates an empty report.
    pub fn new() -> Self {
        Self {
            checked_at: SystemTime::now(),
            violations: Vec::new(),
            assets_checked: 0,
            standards_evaluated: Vec::new(),
        }
    }

    /// Returns the total number of violations.
    pub fn violation_count(&self) -> usize {
        self.violations.len()
    }

    /// Returns violations filtered by severity.
    pub fn violations_by_severity(&self, severity: Severity) -> Vec<&ComplianceViolation> {
        self.violations
            .iter()
            .filter(|v| v.severity == severity)
            .collect()
    }

    /// Returns violations filtered by standard.
    pub fn violations_by_standard(
        &self,
        standard: ComplianceStandard,
    ) -> Vec<&ComplianceViolation> {
        self.violations
            .iter()
            .filter(|v| v.standard == standard)
            .collect()
    }

    /// Returns the highest severity found, if any.
    pub fn highest_severity(&self) -> Option<Severity> {
        self.violations.iter().map(|v| v.severity).max()
    }

    /// Returns true if there are no critical or major violations.
    pub fn is_compliant(&self) -> bool {
        !self
            .violations
            .iter()
            .any(|v| v.severity == Severity::Critical || v.severity == Severity::Major)
    }

    /// Returns a summary count of violations by severity.
    pub fn severity_summary(&self) -> HashMap<String, usize> {
        let mut summary = HashMap::new();
        for v in &self.violations {
            *summary.entry(v.severity.to_string()).or_insert(0) += 1;
        }
        summary
    }
}

impl Default for ComplianceReport {
    fn default() -> Self {
        Self::new()
    }
}

/// Compliance checker that evaluates assets against configured standards.
#[derive(Debug)]
pub struct ComplianceChecker {
    /// Retention policies.
    retention_policies: Vec<RetentionPolicy>,
    /// Whether to require encryption for HIPAA assets.
    require_hipaa_encryption: bool,
    /// Whether to require access logging.
    require_access_logging: bool,
}

impl ComplianceChecker {
    /// Creates a new compliance checker.
    pub fn new() -> Self {
        Self {
            retention_policies: Vec::new(),
            require_hipaa_encryption: true,
            require_access_logging: true,
        }
    }

    /// Adds a retention policy.
    pub fn add_retention_policy(&mut self, policy: RetentionPolicy) {
        self.retention_policies.push(policy);
    }

    /// Returns the number of configured retention policies.
    pub fn policy_count(&self) -> usize {
        self.retention_policies.len()
    }

    /// Sets whether HIPAA encryption is required.
    pub fn set_require_hipaa_encryption(&mut self, require: bool) {
        self.require_hipaa_encryption = require;
    }

    /// Checks a single asset against all applicable standards.
    pub fn check_asset(
        &self,
        asset: &ComplianceAsset,
        standards: &[ComplianceStandard],
    ) -> Vec<ComplianceViolation> {
        let mut violations = Vec::new();

        for standard in standards {
            match standard {
                ComplianceStandard::Gdpr => {
                    self.check_gdpr(asset, &mut violations);
                }
                ComplianceStandard::Hipaa => {
                    self.check_hipaa(asset, &mut violations);
                }
                ComplianceStandard::BroadcastContent => {
                    self.check_broadcast(asset, &mut violations);
                }
                ComplianceStandard::Sox => {
                    self.check_sox(asset, &mut violations);
                }
                ComplianceStandard::PciDss => {
                    self.check_pci_dss(asset, &mut violations);
                }
                ComplianceStandard::Custom => {}
            }
        }

        violations
    }

    /// Checks a batch of assets and produces a full report.
    pub fn check_batch(
        &self,
        assets: &[ComplianceAsset],
        standards: &[ComplianceStandard],
    ) -> ComplianceReport {
        let mut report = ComplianceReport::new();
        report.standards_evaluated = standards.to_vec();
        report.assets_checked = assets.len() as u64;

        for asset in assets {
            let violations = self.check_asset(asset, standards);
            report.violations.extend(violations);
        }

        report
    }

    /// GDPR-specific checks.
    fn check_gdpr(&self, asset: &ComplianceAsset, violations: &mut Vec<ComplianceViolation>) {
        // Check retention policies for GDPR
        for policy in &self.retention_policies {
            if policy.standard == ComplianceStandard::Gdpr {
                let status = policy.check_retention(asset.archived_at);
                if status == RetentionStatus::ExceededMaxRetention && !asset.legal_hold {
                    violations.push(ComplianceViolation {
                        standard: ComplianceStandard::Gdpr,
                        severity: Severity::Major,
                        asset_path: Some(asset.path.clone()),
                        description: format!(
                            "Asset exceeds GDPR max retention for policy '{}'",
                            policy.name
                        ),
                        remediation: "Delete or anonymize the asset".to_string(),
                        rule_id: "GDPR-RET-001".to_string(),
                    });
                }
            }
        }

        // GDPR requires access logging for personal data
        if asset.tags.iter().any(|t| t == "personal_data") && !asset.access_logged {
            violations.push(ComplianceViolation {
                standard: ComplianceStandard::Gdpr,
                severity: Severity::Major,
                asset_path: Some(asset.path.clone()),
                description: "Personal data asset lacks access logging".to_string(),
                remediation: "Enable access logging for this asset".to_string(),
                rule_id: "GDPR-LOG-001".to_string(),
            });
        }
    }

    /// HIPAA-specific checks.
    fn check_hipaa(&self, asset: &ComplianceAsset, violations: &mut Vec<ComplianceViolation>) {
        if self.require_hipaa_encryption && !asset.encrypted {
            violations.push(ComplianceViolation {
                standard: ComplianceStandard::Hipaa,
                severity: Severity::Critical,
                asset_path: Some(asset.path.clone()),
                description: "HIPAA-regulated asset is not encrypted at rest".to_string(),
                remediation: "Encrypt the asset using AES-256".to_string(),
                rule_id: "HIPAA-ENC-001".to_string(),
            });
        }

        if !asset.access_logged {
            violations.push(ComplianceViolation {
                standard: ComplianceStandard::Hipaa,
                severity: Severity::Major,
                asset_path: Some(asset.path.clone()),
                description: "HIPAA asset lacks access audit logging".to_string(),
                remediation: "Enable access logging".to_string(),
                rule_id: "HIPAA-AUD-001".to_string(),
            });
        }
    }

    /// Broadcast content checks.
    fn check_broadcast(&self, asset: &ComplianceAsset, violations: &mut Vec<ComplianceViolation>) {
        if asset.tags.iter().any(|t| t == "unrated_content") {
            violations.push(ComplianceViolation {
                standard: ComplianceStandard::BroadcastContent,
                severity: Severity::Warning,
                asset_path: Some(asset.path.clone()),
                description: "Broadcast asset has no content rating".to_string(),
                remediation: "Assign a content rating before distribution".to_string(),
                rule_id: "BCAST-RATE-001".to_string(),
            });
        }
    }

    /// SOX-specific checks.
    fn check_sox(&self, asset: &ComplianceAsset, violations: &mut Vec<ComplianceViolation>) {
        for policy in &self.retention_policies {
            if policy.standard == ComplianceStandard::Sox {
                let status = policy.check_retention(asset.archived_at);
                if status == RetentionStatus::ExceededMaxRetention {
                    violations.push(ComplianceViolation {
                        standard: ComplianceStandard::Sox,
                        severity: Severity::Major,
                        asset_path: Some(asset.path.clone()),
                        description: format!(
                            "Asset exceeds SOX retention for policy '{}'",
                            policy.name
                        ),
                        remediation: "Review and archive or delete".to_string(),
                        rule_id: "SOX-RET-001".to_string(),
                    });
                }
            }
        }
    }

    /// PCI-DSS-specific checks.
    fn check_pci_dss(&self, asset: &ComplianceAsset, violations: &mut Vec<ComplianceViolation>) {
        if asset.tags.iter().any(|t| t == "payment_data") && !asset.encrypted {
            violations.push(ComplianceViolation {
                standard: ComplianceStandard::PciDss,
                severity: Severity::Critical,
                asset_path: Some(asset.path.clone()),
                description: "Payment data asset is not encrypted".to_string(),
                remediation: "Encrypt immediately per PCI-DSS requirement 3".to_string(),
                rule_id: "PCI-ENC-001".to_string(),
            });
        }
    }
}

impl Default for ComplianceChecker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_asset(path: &str) -> ComplianceAsset {
        ComplianceAsset::new(PathBuf::from(path), SystemTime::now(), 1024)
    }

    #[test]
    fn test_compliance_standard_display() {
        assert_eq!(ComplianceStandard::Gdpr.to_string(), "GDPR");
        assert_eq!(ComplianceStandard::Hipaa.to_string(), "HIPAA");
        assert_eq!(
            ComplianceStandard::BroadcastContent.to_string(),
            "Broadcast Content"
        );
        assert_eq!(ComplianceStandard::Sox.to_string(), "SOX");
        assert_eq!(ComplianceStandard::PciDss.to_string(), "PCI-DSS");
    }

    #[test]
    fn test_severity_ordering() {
        assert!(Severity::Info < Severity::Warning);
        assert!(Severity::Warning < Severity::Minor);
        assert!(Severity::Minor < Severity::Major);
        assert!(Severity::Major < Severity::Critical);
    }

    #[test]
    fn test_retention_policy_within_bounds() {
        let policy =
            RetentionPolicy::new("test", ComplianceStandard::Gdpr, Duration::from_hours(24));
        let status = policy.check_retention(SystemTime::now());
        assert_eq!(status, RetentionStatus::WithinRetention);
    }

    #[test]
    fn test_retention_policy_with_max() {
        let policy = RetentionPolicy::new("test", ComplianceStandard::Gdpr, Duration::from_secs(0))
            .with_max_retention(Duration::from_secs(1));
        // Just created, should be within
        let status = policy.check_retention(SystemTime::now());
        assert_eq!(status, RetentionStatus::WithinRetention);
    }

    #[test]
    fn test_retention_status_display() {
        assert_eq!(
            RetentionStatus::WithinRetention.to_string(),
            "Within Retention"
        );
        assert_eq!(
            RetentionStatus::ExceededMaxRetention.to_string(),
            "Exceeded Max Retention"
        );
        assert_eq!(RetentionStatus::LegalHold.to_string(), "Legal Hold");
    }

    #[test]
    fn test_compliance_asset_builder() {
        let asset = ComplianceAsset::new(PathBuf::from("/test"), SystemTime::now(), 500)
            .with_tag("personal_data")
            .with_legal_hold(true)
            .with_encrypted(true)
            .with_access_logged(true);
        assert_eq!(asset.tags.len(), 1);
        assert!(asset.legal_hold);
        assert!(asset.encrypted);
        assert!(asset.access_logged);
    }

    #[test]
    fn test_hipaa_encryption_violation() {
        let checker = ComplianceChecker::new();
        let asset = make_asset("/hipaa/record.dat");
        let violations = checker.check_asset(&asset, &[ComplianceStandard::Hipaa]);
        // Should find encryption and logging violations
        assert!(violations.iter().any(|v| v.rule_id == "HIPAA-ENC-001"));
        assert!(violations.iter().any(|v| v.rule_id == "HIPAA-AUD-001"));
    }

    #[test]
    fn test_hipaa_compliant_asset() {
        let checker = ComplianceChecker::new();
        let asset = make_asset("/hipaa/ok.dat")
            .with_encrypted(true)
            .with_access_logged(true);
        let violations = checker.check_asset(&asset, &[ComplianceStandard::Hipaa]);
        assert!(violations.is_empty());
    }

    #[test]
    fn test_broadcast_unrated_warning() {
        let checker = ComplianceChecker::new();
        let asset = make_asset("/broadcast/clip.mxf").with_tag("unrated_content");
        let violations = checker.check_asset(&asset, &[ComplianceStandard::BroadcastContent]);
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].severity, Severity::Warning);
    }

    #[test]
    fn test_pci_dss_unencrypted_payment() {
        let checker = ComplianceChecker::new();
        let asset = make_asset("/payment/data.bin").with_tag("payment_data");
        let violations = checker.check_asset(&asset, &[ComplianceStandard::PciDss]);
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].severity, Severity::Critical);
    }

    #[test]
    fn test_compliance_report_is_compliant() {
        let checker = ComplianceChecker::new();
        let asset = make_asset("/clean.dat")
            .with_encrypted(true)
            .with_access_logged(true);
        let report = checker.check_batch(
            &[asset],
            &[
                ComplianceStandard::Hipaa,
                ComplianceStandard::BroadcastContent,
            ],
        );
        assert!(report.is_compliant());
        assert_eq!(report.assets_checked, 1);
    }

    #[test]
    fn test_compliance_report_severity_summary() {
        let checker = ComplianceChecker::new();
        let asset = make_asset("/test.dat").with_tag("unrated_content");
        let report = checker.check_batch(
            &[asset],
            &[
                ComplianceStandard::Hipaa,
                ComplianceStandard::BroadcastContent,
            ],
        );
        let summary = report.severity_summary();
        assert!(summary.contains_key("CRITICAL"));
        assert!(summary.contains_key("WARNING"));
    }

    #[test]
    fn test_gdpr_personal_data_no_logging() {
        let checker = ComplianceChecker::new();
        let asset = make_asset("/gdpr/user.dat").with_tag("personal_data");
        let violations = checker.check_asset(&asset, &[ComplianceStandard::Gdpr]);
        assert!(violations.iter().any(|v| v.rule_id == "GDPR-LOG-001"));
    }

    #[test]
    fn test_violation_display() {
        let violation = ComplianceViolation {
            standard: ComplianceStandard::Hipaa,
            severity: Severity::Critical,
            asset_path: Some(PathBuf::from("/test")),
            description: "Not encrypted".to_string(),
            remediation: "Encrypt it".to_string(),
            rule_id: "HIPAA-ENC-001".to_string(),
        };
        let display = format!("{}", violation);
        assert!(display.contains("CRITICAL"));
        assert!(display.contains("HIPAA"));
        assert!(display.contains("HIPAA-ENC-001"));
    }
}
