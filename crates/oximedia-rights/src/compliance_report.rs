//! Compliance report generator for regulatory requirements.
//!
//! This module generates machine-readable and human-readable compliance reports
//! covering data rights under major regulatory frameworks, including:
//!
//! * **GDPR** (EU General Data Protection Regulation)
//! * **CCPA** (California Consumer Privacy Act)
//! * **LGPD** (Lei Geral de Proteção de Dados — Brazil)
//! * **PIPEDA** (Personal Information Protection and Electronic Documents Act — Canada)
//!
//! Reports document how personal data embedded in or associated with media
//! assets is handled, stored, processed, and disclosed, enabling auditors and
//! data-protection officers to verify that a media platform's rights management
//! activities comply with applicable law.
//!
//! # Design
//!
//! * [`DataCategory`] classifies the types of personal data involved.
//! * [`LegalBasis`] documents the GDPR/CCPA legal ground for processing.
//! * [`DataSubjectRight`] enumerates the rights data subjects may exercise.
//! * [`ComplianceItem`] is a single assertion about how a data category is
//!   handled (the "building block" of a report).
//! * [`ComplianceReport`] aggregates items into a structured report with
//!   per-regulation findings and an overall status.
//! * [`ComplianceReportBuilder`] provides a fluent API for constructing reports.

#![allow(missing_docs)]
#![allow(clippy::cast_precision_loss)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::{Result, RightsError};

// ── Regulation ────────────────────────────────────────────────────────────────

/// A data-protection or privacy regulation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Regulation {
    /// EU General Data Protection Regulation (2016/679).
    Gdpr,
    /// California Consumer Privacy Act (and CPRA).
    Ccpa,
    /// Brazil Lei Geral de Proteção de Dados (Law 13,709/2018).
    Lgpd,
    /// Canadian Personal Information Protection and Electronic Documents Act.
    Pipeda,
    /// User-defined / jurisdiction-specific regulation.
    Custom(String),
}

impl std::fmt::Display for Regulation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Gdpr => write!(f, "GDPR"),
            Self::Ccpa => write!(f, "CCPA"),
            Self::Lgpd => write!(f, "LGPD"),
            Self::Pipeda => write!(f, "PIPEDA"),
            Self::Custom(s) => write!(f, "{s}"),
        }
    }
}

// ── DataCategory ──────────────────────────────────────────────────────────────

/// The category of personal data processed.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DataCategory {
    /// Name, e-mail address, postal address, phone number.
    ContactData,
    /// Rights holder / licensee financial information.
    FinancialData,
    /// Biometric watermarks or voice prints embedded in media.
    BiometricData,
    /// IP addresses, device identifiers, viewing history.
    BehavioralData,
    /// Content usage analytics tied to identifiable users.
    UsageAnalytics,
    /// Contractual documents containing personal data.
    ContractualData,
    /// Creative credits (name associated with an artistic work).
    CreativeCredits,
    /// Custom / system-specific data category.
    Custom(String),
}

impl std::fmt::Display for DataCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ContactData => write!(f, "Contact Data"),
            Self::FinancialData => write!(f, "Financial Data"),
            Self::BiometricData => write!(f, "Biometric Data"),
            Self::BehavioralData => write!(f, "Behavioral Data"),
            Self::UsageAnalytics => write!(f, "Usage Analytics"),
            Self::ContractualData => write!(f, "Contractual Data"),
            Self::CreativeCredits => write!(f, "Creative Credits"),
            Self::Custom(s) => write!(f, "{s}"),
        }
    }
}

// ── LegalBasis ────────────────────────────────────────────────────────────────

/// The legal basis under which personal data is processed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LegalBasis {
    /// The data subject has given explicit consent (GDPR Art. 6(1)(a)).
    Consent,
    /// Processing is necessary for contract performance (GDPR Art. 6(1)(b)).
    ContractPerformance,
    /// Processing is required to comply with a legal obligation (Art. 6(1)(c)).
    LegalObligation,
    /// Processing protects the vital interests of the data subject (Art. 6(1)(d)).
    VitalInterests,
    /// Processing is in the public interest (Art. 6(1)(e)).
    PublicTask,
    /// Processing serves a legitimate interest that is not overridden (Art. 6(1)(f)).
    LegitimateInterest,
    /// CCPA-specific: data used for operational purposes with opt-out right.
    CcpaOperational,
    /// Custom legal basis or jurisdictional equivalent.
    Custom(String),
}

impl std::fmt::Display for LegalBasis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Consent => write!(f, "Consent"),
            Self::ContractPerformance => write!(f, "Contract Performance"),
            Self::LegalObligation => write!(f, "Legal Obligation"),
            Self::VitalInterests => write!(f, "Vital Interests"),
            Self::PublicTask => write!(f, "Public Task"),
            Self::LegitimateInterest => write!(f, "Legitimate Interest"),
            Self::CcpaOperational => write!(f, "CCPA Operational"),
            Self::Custom(s) => write!(f, "{s}"),
        }
    }
}

// ── DataSubjectRight ──────────────────────────────────────────────────────────

/// Rights that data subjects can exercise under applicable law.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DataSubjectRight {
    /// Right to obtain a copy of personal data held.
    Access,
    /// Right to correct inaccurate personal data.
    Rectification,
    /// Right to have personal data erased ("right to be forgotten").
    Erasure,
    /// Right to restrict processing.
    RestrictionOfProcessing,
    /// Right to receive data in a portable, machine-readable format.
    DataPortability,
    /// Right to object to processing.
    Objection,
    /// Right to opt-out of the sale of personal information (CCPA).
    OptOutOfSale,
    /// Right not to be subject to automated decision-making.
    NonautomatedDecision,
}

impl std::fmt::Display for DataSubjectRight {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Access => write!(f, "Access"),
            Self::Rectification => write!(f, "Rectification"),
            Self::Erasure => write!(f, "Erasure"),
            Self::RestrictionOfProcessing => write!(f, "Restriction of Processing"),
            Self::DataPortability => write!(f, "Data Portability"),
            Self::Objection => write!(f, "Objection"),
            Self::OptOutOfSale => write!(f, "Opt-Out of Sale"),
            Self::NonautomatedDecision => write!(f, "Non-Automated Decision"),
        }
    }
}

// ── RetentionPolicy ───────────────────────────────────────────────────────────

/// Documents how long a data category is retained and why.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionPolicy {
    /// Data category this policy applies to.
    pub data_category: DataCategory,
    /// Maximum retention period in days (0 = session-only / ephemeral).
    pub retention_days: u32,
    /// Justification for the retention period.
    pub justification: String,
    /// Whether the data is anonymised at the end of the retention period
    /// rather than deleted.
    pub anonymise_on_expiry: bool,
}

impl RetentionPolicy {
    /// Create a new retention policy.
    pub fn new(
        data_category: DataCategory,
        retention_days: u32,
        justification: impl Into<String>,
    ) -> Self {
        Self {
            data_category,
            retention_days,
            justification: justification.into(),
            anonymise_on_expiry: false,
        }
    }

    /// Mark data as anonymised rather than deleted on expiry.
    #[must_use]
    pub fn with_anonymise_on_expiry(mut self) -> Self {
        self.anonymise_on_expiry = true;
        self
    }
}

// ── ComplianceStatus ──────────────────────────────────────────────────────────

/// Compliance status of a single item or report.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComplianceStatus {
    /// All checks pass — no violations found.
    Compliant,
    /// At least one non-critical gap was identified.
    PartiallyCompliant,
    /// One or more critical violations were found.
    NonCompliant,
    /// Insufficient information to make a determination.
    Unknown,
}

impl std::fmt::Display for ComplianceStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Compliant => write!(f, "Compliant"),
            Self::PartiallyCompliant => write!(f, "Partially Compliant"),
            Self::NonCompliant => write!(f, "Non-Compliant"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

// ── ComplianceItem ────────────────────────────────────────────────────────────

/// A single assertion within a compliance report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceItem {
    /// Short identifier for this item (e.g. `"gdpr.contact-data.legal-basis"`).
    pub id: String,
    /// Regulation this item applies to.
    pub regulation: Regulation,
    /// The data category this item is about.
    pub data_category: DataCategory,
    /// The legal basis documented for this processing activity.
    pub legal_basis: LegalBasis,
    /// Retention policy for this data category.
    pub retention_policy: Option<RetentionPolicy>,
    /// Data-subject rights available to individuals for this data.
    pub available_rights: Vec<DataSubjectRight>,
    /// Whether this data is transferred to third-party recipients.
    pub third_party_transfers: bool,
    /// Names of third-party recipients, if any.
    pub recipients: Vec<String>,
    /// Whether cross-border (outside EEA / jurisdiction) transfers occur.
    pub cross_border_transfer: bool,
    /// Transfer mechanism for cross-border transfers (e.g. "SCCs", "BCRs").
    pub transfer_mechanism: Option<String>,
    /// Whether a Data Protection Impact Assessment (DPIA) was conducted.
    pub dpia_conducted: bool,
    /// Status of this compliance item.
    pub status: ComplianceStatus,
    /// Any findings, gaps, or recommendations for this item.
    pub findings: Vec<String>,
}

impl ComplianceItem {
    /// Create a new compliance item with minimal fields.
    pub fn new(
        id: impl Into<String>,
        regulation: Regulation,
        data_category: DataCategory,
        legal_basis: LegalBasis,
    ) -> Self {
        Self {
            id: id.into(),
            regulation,
            data_category,
            legal_basis,
            retention_policy: None,
            available_rights: Vec::new(),
            third_party_transfers: false,
            recipients: Vec::new(),
            cross_border_transfer: false,
            transfer_mechanism: None,
            dpia_conducted: false,
            status: ComplianceStatus::Unknown,
            findings: Vec::new(),
        }
    }

    /// Evaluate the item's compliance status based on its configuration.
    ///
    /// Applies a set of heuristic checks that mirror common GDPR/CCPA
    /// requirements.  The result is stored in `self.status` and any
    /// identified gaps are appended to `self.findings`.
    pub fn evaluate(&mut self) {
        self.findings.clear();

        // Check: cross-border transfer without a transfer mechanism
        if self.cross_border_transfer && self.transfer_mechanism.is_none() {
            self.findings.push(
                "Cross-border transfer is enabled but no transfer mechanism (e.g. SCCs) \
                 is documented."
                    .into(),
            );
        }

        // Check: biometric data without DPIA
        if matches!(self.data_category, DataCategory::BiometricData) && !self.dpia_conducted {
            self.findings.push(
                "Biometric data processing requires a Data Protection Impact Assessment (DPIA)."
                    .into(),
            );
        }

        // Check: GDPR requires right of access to be available for personal data
        if matches!(self.regulation, Regulation::Gdpr)
            && !self.available_rights.contains(&DataSubjectRight::Access)
        {
            self.findings.push(
                "GDPR Art. 15 requires the right of access to be granted for this data category."
                    .into(),
            );
        }

        // Check: CCPA requires opt-out right if data is sold / shared commercially
        if matches!(self.regulation, Regulation::Ccpa)
            && self.third_party_transfers
            && !self.available_rights.contains(&DataSubjectRight::OptOutOfSale)
        {
            self.findings.push(
                "CCPA § 1798.120 requires the right to opt-out of sale when data is \
                 transferred to third parties."
                    .into(),
            );
        }

        // Check: financial data should have explicit retention policy
        if matches!(self.data_category, DataCategory::FinancialData)
            && self.retention_policy.is_none()
        {
            self.findings.push(
                "Financial data must have a documented retention policy.".into(),
            );
        }

        // Derive overall status
        self.status = if self.findings.is_empty() {
            ComplianceStatus::Compliant
        } else if self.findings.len() == 1 {
            ComplianceStatus::PartiallyCompliant
        } else {
            ComplianceStatus::NonCompliant
        };
    }
}

// ── ComplianceReport ──────────────────────────────────────────────────────────

/// A complete compliance report for a media platform or service.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceReport {
    /// Unique report identifier.
    pub id: String,
    /// Name of the system or service being assessed.
    pub system_name: String,
    /// Version of the system being assessed.
    pub system_version: Option<String>,
    /// ISO 8601 date when the report was generated (e.g. `"2024-06-01"`).
    pub report_date: String,
    /// Name of the assessor (person or automated system).
    pub assessor: String,
    /// Individual compliance items.
    pub items: Vec<ComplianceItem>,
    /// Per-regulation aggregated status.
    pub regulation_status: HashMap<String, ComplianceStatus>,
    /// Overall compliance status across all regulations.
    pub overall_status: ComplianceStatus,
    /// Free-form executive summary.
    pub summary: String,
    /// Recommended remediation actions.
    pub recommendations: Vec<String>,
}

impl ComplianceReport {
    /// Create a new, empty report.
    pub fn new(
        id: impl Into<String>,
        system_name: impl Into<String>,
        report_date: impl Into<String>,
        assessor: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            system_name: system_name.into(),
            system_version: None,
            report_date: report_date.into(),
            assessor: assessor.into(),
            items: Vec::new(),
            regulation_status: HashMap::new(),
            overall_status: ComplianceStatus::Unknown,
            summary: String::new(),
            recommendations: Vec::new(),
        }
    }

    /// Set the system version string.
    #[must_use]
    pub fn with_system_version(mut self, version: impl Into<String>) -> Self {
        self.system_version = Some(version.into());
        self
    }

    /// Add a compliance item.
    pub fn add_item(&mut self, item: ComplianceItem) {
        self.items.push(item);
    }

    /// (Re-)evaluate all items and recompute aggregated statuses.
    pub fn evaluate_all(&mut self) {
        for item in &mut self.items {
            item.evaluate();
        }
        self.recompute_status();
    }

    /// Recompute per-regulation and overall status from current item states.
    fn recompute_status(&mut self) {
        self.regulation_status.clear();
        self.recommendations.clear();

        // group items by regulation
        let mut by_reg: HashMap<String, Vec<&ComplianceItem>> = HashMap::new();
        for item in &self.items {
            by_reg
                .entry(item.regulation.to_string())
                .or_default()
                .push(item);
        }

        for (reg_name, items) in &by_reg {
            let status = Self::aggregate_status(items);
            self.regulation_status.insert(reg_name.clone(), status);
        }

        // overall status = worst across all regulations
        let all_items: Vec<&ComplianceItem> = self.items.iter().collect();
        self.overall_status = Self::aggregate_status(&all_items);

        // collect unique recommendations from all findings
        let mut seen = std::collections::HashSet::new();
        for item in &self.items {
            for finding in &item.findings {
                if seen.insert(finding.clone()) {
                    self.recommendations.push(finding.clone());
                }
            }
        }

        // derive a concise summary
        let compliant_count = self
            .items
            .iter()
            .filter(|i| i.status == ComplianceStatus::Compliant)
            .count();
        let total = self.items.len();
        self.summary = format!(
            "{}/{} items compliant. Overall status: {}.",
            compliant_count,
            total,
            self.overall_status
        );
    }

    /// Aggregate a slice of items into a single status (worst-case).
    fn aggregate_status(items: &[&ComplianceItem]) -> ComplianceStatus {
        if items.is_empty() {
            return ComplianceStatus::Unknown;
        }
        if items.iter().any(|i| i.status == ComplianceStatus::NonCompliant) {
            return ComplianceStatus::NonCompliant;
        }
        if items
            .iter()
            .any(|i| i.status == ComplianceStatus::PartiallyCompliant)
        {
            return ComplianceStatus::PartiallyCompliant;
        }
        if items.iter().all(|i| i.status == ComplianceStatus::Compliant) {
            return ComplianceStatus::Compliant;
        }
        ComplianceStatus::Unknown
    }

    /// Return items whose status is not [`ComplianceStatus::Compliant`].
    #[must_use]
    pub fn non_compliant_items(&self) -> Vec<&ComplianceItem> {
        self.items
            .iter()
            .filter(|i| i.status != ComplianceStatus::Compliant)
            .collect()
    }

    /// Render the report as a plain-text human-readable string.
    #[must_use]
    pub fn to_text_report(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("COMPLIANCE REPORT — {}\n", self.system_name));
        if let Some(v) = &self.system_version {
            out.push_str(&format!("Version: {v}\n"));
        }
        out.push_str(&format!("Report Date: {}\n", self.report_date));
        out.push_str(&format!("Assessor: {}\n", self.assessor));
        out.push_str(&format!("Overall Status: {}\n", self.overall_status));
        out.push('\n');

        out.push_str("PER-REGULATION STATUS\n");
        let mut sorted_regs: Vec<(&String, &ComplianceStatus)> =
            self.regulation_status.iter().collect();
        sorted_regs.sort_by_key(|(k, _)| k.as_str());
        for (reg, status) in &sorted_regs {
            out.push_str(&format!("  {reg}: {status}\n"));
        }
        out.push('\n');

        out.push_str("ITEMS\n");
        for item in &self.items {
            out.push_str(&format!(
                "  [{}] {} — {} / {} — {}\n",
                item.status,
                item.id,
                item.regulation,
                item.data_category,
                item.legal_basis,
            ));
            for finding in &item.findings {
                out.push_str(&format!("    FINDING: {finding}\n"));
            }
        }

        if !self.recommendations.is_empty() {
            out.push('\n');
            out.push_str("RECOMMENDATIONS\n");
            for (i, rec) in self.recommendations.iter().enumerate() {
                out.push_str(&format!("  {}. {rec}\n", i + 1));
            }
        }

        out
    }

    /// Render the report as a JSON string.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| RightsError::Serialization(e.to_string()))
    }
}

// ── ComplianceReportBuilder ───────────────────────────────────────────────────

/// Fluent builder for constructing and evaluating a [`ComplianceReport`].
pub struct ComplianceReportBuilder {
    report: ComplianceReport,
}

impl ComplianceReportBuilder {
    /// Start building a new compliance report.
    pub fn new(
        id: impl Into<String>,
        system_name: impl Into<String>,
        report_date: impl Into<String>,
        assessor: impl Into<String>,
    ) -> Self {
        Self {
            report: ComplianceReport::new(id, system_name, report_date, assessor),
        }
    }

    /// Set the system version.
    #[must_use]
    pub fn system_version(mut self, version: impl Into<String>) -> Self {
        self.report.system_version = Some(version.into());
        self
    }

    /// Add a compliance item (unevaluated).
    #[must_use]
    pub fn item(mut self, item: ComplianceItem) -> Self {
        self.report.items.push(item);
        self
    }

    /// Evaluate all items and produce the finished report.
    #[must_use]
    pub fn build(mut self) -> ComplianceReport {
        self.report.evaluate_all();
        self.report
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn gdpr_contact_item() -> ComplianceItem {
        let mut item = ComplianceItem::new(
            "gdpr.contact",
            Regulation::Gdpr,
            DataCategory::ContactData,
            LegalBasis::ContractPerformance,
        );
        item.available_rights.push(DataSubjectRight::Access);
        item.available_rights.push(DataSubjectRight::Erasure);
        item
    }

    #[test]
    fn test_compliant_item() {
        let mut item = gdpr_contact_item();
        item.evaluate();
        assert_eq!(item.status, ComplianceStatus::Compliant);
        assert!(item.findings.is_empty());
    }

    #[test]
    fn test_gdpr_missing_access_right() {
        let mut item = ComplianceItem::new(
            "gdpr.usage",
            Regulation::Gdpr,
            DataCategory::UsageAnalytics,
            LegalBasis::LegitimateInterest,
        );
        // do NOT add DataSubjectRight::Access
        item.evaluate();
        assert_ne!(item.status, ComplianceStatus::Compliant);
        assert!(item.findings.iter().any(|f| f.contains("Art. 15")));
    }

    #[test]
    fn test_ccpa_opt_out_required_on_third_party_transfer() {
        let mut item = ComplianceItem::new(
            "ccpa.behavioral",
            Regulation::Ccpa,
            DataCategory::BehavioralData,
            LegalBasis::CcpaOperational,
        );
        item.third_party_transfers = true;
        // no opt-out right added
        item.evaluate();
        assert_ne!(item.status, ComplianceStatus::Compliant);
        assert!(item.findings.iter().any(|f| f.contains("opt-out")));
    }

    #[test]
    fn test_biometric_dpia_required() {
        let mut item = ComplianceItem::new(
            "gdpr.bio",
            Regulation::Gdpr,
            DataCategory::BiometricData,
            LegalBasis::Consent,
        );
        item.available_rights.push(DataSubjectRight::Access);
        item.dpia_conducted = false;
        item.evaluate();
        assert!(item.findings.iter().any(|f| f.contains("DPIA")));
    }

    #[test]
    fn test_cross_border_transfer_without_mechanism() {
        let mut item = ComplianceItem::new(
            "gdpr.fin",
            Regulation::Gdpr,
            DataCategory::FinancialData,
            LegalBasis::LegalObligation,
        );
        item.available_rights.push(DataSubjectRight::Access);
        item.cross_border_transfer = true;
        item.transfer_mechanism = None;
        item.retention_policy = Some(RetentionPolicy::new(
            DataCategory::FinancialData,
            2555,
            "7-year statutory requirement",
        ));
        item.evaluate();
        assert!(item
            .findings
            .iter()
            .any(|f| f.contains("transfer mechanism")));
    }

    #[test]
    fn test_report_builder_evaluate_all() {
        let item1 = gdpr_contact_item();
        let mut item2 = ComplianceItem::new(
            "gdpr.financial",
            Regulation::Gdpr,
            DataCategory::FinancialData,
            LegalBasis::LegalObligation,
        );
        item2.available_rights.push(DataSubjectRight::Access);
        // missing retention policy → partial compliance

        let report = ComplianceReportBuilder::new("r1", "OxiMedia Rights", "2024-06-01", "DPO")
            .system_version("0.1.3")
            .item(item1)
            .item(item2)
            .build();

        assert_ne!(report.overall_status, ComplianceStatus::Unknown);
        assert!(!report.summary.is_empty());
    }

    #[test]
    fn test_non_compliant_items_filter() {
        let item1 = gdpr_contact_item();
        let mut item2 = ComplianceItem::new(
            "gdpr.bio",
            Regulation::Gdpr,
            DataCategory::BiometricData,
            LegalBasis::Consent,
        );
        // missing DPIA and Access right → multiple findings → NonCompliant
        item2.evaluate();

        let mut report =
            ComplianceReport::new("r2", "Test System", "2024-01-01", "Automated");
        let mut item1_eval = item1;
        item1_eval.evaluate();
        report.add_item(item1_eval);
        report.add_item(item2);
        report.evaluate_all();

        let non_compliant = report.non_compliant_items();
        // item2 should appear (biometric without DPIA + missing Access)
        assert!(non_compliant.iter().any(|i| i.id == "gdpr.bio"));
    }

    #[test]
    fn test_report_to_text_contains_status() {
        let report = ComplianceReportBuilder::new("r3", "Media Platform", "2024-01-01", "DPO")
            .item(gdpr_contact_item())
            .build();
        let text = report.to_text_report();
        assert!(text.contains("COMPLIANCE REPORT"));
        assert!(text.contains("GDPR"));
    }

    #[test]
    fn test_report_to_json_valid() {
        let report = ComplianceReportBuilder::new("r4", "Rights Service", "2024-01-01", "DPO")
            .item(gdpr_contact_item())
            .build();
        let json = report.to_json().unwrap();
        assert!(json.contains("\"id\""));
        assert!(json.contains("r4"));
    }

    #[test]
    fn test_retention_policy_fields() {
        let policy = RetentionPolicy::new(DataCategory::FinancialData, 2555, "Legal requirement")
            .with_anonymise_on_expiry();
        assert_eq!(policy.retention_days, 2555);
        assert!(policy.anonymise_on_expiry);
    }
}
