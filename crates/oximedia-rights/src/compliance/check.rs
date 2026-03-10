//! Compliance checking

use crate::{database::RightsDatabase, rights::RightsGrant, Result};
use chrono::Utc;
use serde::{Deserialize, Serialize};

/// Issue severity level
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum IssueSeverity {
    /// Low severity
    Low,
    /// Medium severity
    Medium,
    /// High severity
    High,
    /// Critical severity
    Critical,
}

/// Compliance issue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceIssue {
    /// Issue type
    pub issue_type: String,
    /// Severity
    pub severity: IssueSeverity,
    /// Description
    pub description: String,
    /// Entity type
    pub entity_type: String,
    /// Entity ID
    pub entity_id: String,
}

impl ComplianceIssue {
    /// Create a new compliance issue
    pub fn new(
        issue_type: impl Into<String>,
        severity: IssueSeverity,
        description: impl Into<String>,
        entity_type: impl Into<String>,
        entity_id: impl Into<String>,
    ) -> Self {
        Self {
            issue_type: issue_type.into(),
            severity,
            description: description.into(),
            entity_type: entity_type.into(),
            entity_id: entity_id.into(),
        }
    }
}

/// Compliance checker
pub struct ComplianceChecker<'a> {
    db: &'a RightsDatabase,
}

impl<'a> ComplianceChecker<'a> {
    /// Create a new compliance checker
    pub fn new(db: &'a RightsDatabase) -> Self {
        Self { db }
    }

    /// Check compliance for an asset
    pub async fn check_asset(&self, asset_id: &str) -> Result<Vec<ComplianceIssue>> {
        let mut issues = Vec::new();

        // Check for active grants
        let grants = RightsGrant::list_for_asset(self.db, asset_id).await?;

        if grants.is_empty() {
            issues.push(ComplianceIssue::new(
                "no_grants",
                IssueSeverity::High,
                "No rights grants found for asset",
                "asset",
                asset_id,
            ));
        }

        // Check for expired grants
        for grant in &grants {
            if grant.is_expired() {
                issues.push(ComplianceIssue::new(
                    "expired_grant",
                    IssueSeverity::Critical,
                    format!("Grant {} has expired", grant.id),
                    "grant",
                    &grant.id,
                ));
            }
        }

        // Check for expiring soon grants (within 30 days)
        let now = Utc::now();
        let threshold = now + chrono::Duration::days(30);

        for grant in &grants {
            if let Some(end_date) = grant.end_date {
                if end_date <= threshold && end_date > now {
                    issues.push(ComplianceIssue::new(
                        "expiring_soon",
                        IssueSeverity::Medium,
                        format!("Grant {} expires soon", grant.id),
                        "grant",
                        &grant.id,
                    ));
                }
            }
        }

        Ok(issues)
    }

    /// Check all compliance issues
    pub async fn check_all(&self) -> Result<Vec<ComplianceIssue>> {
        let mut all_issues = Vec::new();

        // Get all assets
        let assets = crate::rights::Asset::list(self.db).await?;

        for asset in assets {
            let issues = self.check_asset(&asset.id).await?;
            all_issues.extend(issues);
        }

        Ok(all_issues)
    }

    /// Get critical issues only
    pub async fn get_critical_issues(&self) -> Result<Vec<ComplianceIssue>> {
        let all_issues = self.check_all().await?;
        Ok(all_issues
            .into_iter()
            .filter(|issue| matches!(issue.severity, IssueSeverity::Critical))
            .collect())
    }
}

// ── GDPR compliance ──────────────────────────────────────────────────────────

/// Personally identifiable data category tracked for GDPR purposes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PiiCategory {
    /// Full name (first, last, middle).
    FullName,
    /// Email address.
    Email,
    /// Physical / mailing address.
    Address,
    /// Phone number.
    Phone,
    /// National ID / passport number.
    NationalId,
    /// Payment / bank account details.
    PaymentInfo,
    /// IP address.
    IpAddress,
    /// Cookie / device identifier.
    DeviceId,
    /// Biometric data (e.g. face geometry).
    Biometric,
    /// Other free-text category.
    Other(String),
}

/// Consent basis under GDPR Article 6.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LawfulBasis {
    /// Data subject has given explicit consent.
    Consent,
    /// Processing is necessary for a contract with the data subject.
    Contract,
    /// Processing is required by law.
    LegalObligation,
    /// Necessary to protect vital interests.
    VitalInterest,
    /// Necessary for a task in the public interest.
    PublicInterest,
    /// Necessary for legitimate interests of the controller.
    LegitimateInterest,
}

/// Record of personal data held about a single data subject.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GdprDataRecord {
    /// Unique data-subject identifier (internal, NOT the PII itself).
    pub subject_id: String,
    /// Categories of PII stored for this subject.
    pub categories: Vec<PiiCategory>,
    /// Lawful basis under which the data is processed.
    pub lawful_basis: LawfulBasis,
    /// Purpose of processing (free text).
    pub purpose: String,
    /// Whether the subject has given explicit consent.
    pub consent_given: bool,
    /// Unix timestamp (seconds) when the record was created.
    pub created_ts: i64,
    /// Optional retention deadline (Unix seconds).  After this timestamp the
    /// record should be deleted or anonymised.
    pub retention_deadline_ts: Option<i64>,
}

/// GDPR compliance checker.
///
/// Validates a collection of [`GdprDataRecord`]s against core GDPR principles:
///
/// * **Lawfulness** – every record must have a valid lawful basis; consent-based
///   records must actually have consent.
/// * **Purpose limitation** – purpose must not be empty.
/// * **Data minimisation** – flags records with an unusually large number of PII
///   categories (configurable threshold).
/// * **Storage limitation** – flags records past their retention deadline.
pub struct GdprChecker {
    /// Maximum PII categories before a minimisation warning is raised.
    max_categories: usize,
}

impl GdprChecker {
    /// Create a new GDPR checker with a maximum PII-category threshold.
    pub fn new(max_categories: usize) -> Self {
        Self { max_categories }
    }

    /// Check a set of data records and return any compliance issues found.
    pub fn check(&self, records: &[GdprDataRecord], now_ts: i64) -> Vec<ComplianceIssue> {
        let mut issues = Vec::new();

        for record in records {
            // Consent check
            if record.lawful_basis == LawfulBasis::Consent && !record.consent_given {
                issues.push(ComplianceIssue::new(
                    "gdpr_consent_missing",
                    IssueSeverity::Critical,
                    format!(
                        "Subject {} processed under Consent basis but consent not recorded",
                        record.subject_id
                    ),
                    "gdpr_record",
                    &record.subject_id,
                ));
            }

            // Purpose limitation
            if record.purpose.trim().is_empty() {
                issues.push(ComplianceIssue::new(
                    "gdpr_purpose_empty",
                    IssueSeverity::High,
                    format!(
                        "Subject {} has no processing purpose specified",
                        record.subject_id
                    ),
                    "gdpr_record",
                    &record.subject_id,
                ));
            }

            // Data minimisation
            if record.categories.len() > self.max_categories {
                issues.push(ComplianceIssue::new(
                    "gdpr_data_minimisation",
                    IssueSeverity::Medium,
                    format!(
                        "Subject {} stores {} PII categories (threshold {})",
                        record.subject_id,
                        record.categories.len(),
                        self.max_categories,
                    ),
                    "gdpr_record",
                    &record.subject_id,
                ));
            }

            // Storage limitation / retention
            if let Some(deadline) = record.retention_deadline_ts {
                if now_ts > deadline {
                    issues.push(ComplianceIssue::new(
                        "gdpr_retention_expired",
                        IssueSeverity::Critical,
                        format!(
                            "Subject {} data past retention deadline (expired {}s ago)",
                            record.subject_id,
                            now_ts - deadline,
                        ),
                        "gdpr_record",
                        &record.subject_id,
                    ));
                }
            }
        }

        issues
    }
}

// ── DMCA takedown workflow ───────────────────────────────────────────────────

/// Current status of a DMCA takedown request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DmcaStatus {
    /// Initial filing received.
    Filed,
    /// Takedown notice validated and content removed / disabled.
    ContentRemoved,
    /// Counter-notification received from the alleged infringer.
    CounterNotification,
    /// Waiting period after counter-notification (typically 10–14 business days).
    WaitingPeriod,
    /// Content restored after waiting period without court action.
    ContentRestored,
    /// Court action filed; content remains down.
    CourtAction,
    /// Request resolved / closed.
    Resolved,
}

/// A single DMCA takedown request with its lifecycle states.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DmcaTakedown {
    /// Unique request identifier.
    pub id: String,
    /// Asset (content) identifier targeted by the takedown.
    pub asset_id: String,
    /// Name or identifier of the claimant (copyright holder).
    pub claimant: String,
    /// Current status.
    pub status: DmcaStatus,
    /// Unix timestamp when the request was filed.
    pub filed_ts: i64,
    /// Optional Unix timestamp when content was removed.
    pub removed_ts: Option<i64>,
    /// Optional Unix timestamp when a counter-notification was filed.
    pub counter_ts: Option<i64>,
    /// Optional Unix timestamp when content was restored.
    pub restored_ts: Option<i64>,
    /// Free-text description of the alleged infringement.
    pub description: String,
}

impl DmcaTakedown {
    /// Create a new DMCA takedown request in `Filed` status.
    pub fn new(
        id: impl Into<String>,
        asset_id: impl Into<String>,
        claimant: impl Into<String>,
        description: impl Into<String>,
        filed_ts: i64,
    ) -> Self {
        Self {
            id: id.into(),
            asset_id: asset_id.into(),
            claimant: claimant.into(),
            status: DmcaStatus::Filed,
            filed_ts,
            removed_ts: None,
            counter_ts: None,
            restored_ts: None,
            description: description.into(),
        }
    }

    /// Advance to `ContentRemoved` state.
    pub fn remove_content(&mut self, ts: i64) {
        self.status = DmcaStatus::ContentRemoved;
        self.removed_ts = Some(ts);
    }

    /// Record a counter-notification from the alleged infringer.
    pub fn file_counter_notification(&mut self, ts: i64) {
        self.status = DmcaStatus::CounterNotification;
        self.counter_ts = Some(ts);
    }

    /// Enter the 10–14 business day waiting period.
    pub fn begin_waiting_period(&mut self) {
        self.status = DmcaStatus::WaitingPeriod;
    }

    /// Restore content after the waiting period expires without court action.
    pub fn restore_content(&mut self, ts: i64) {
        self.status = DmcaStatus::ContentRestored;
        self.restored_ts = Some(ts);
    }

    /// Record that court action has been filed by the claimant.
    pub fn file_court_action(&mut self) {
        self.status = DmcaStatus::CourtAction;
    }

    /// Mark the takedown as resolved / closed.
    pub fn resolve(&mut self) {
        self.status = DmcaStatus::Resolved;
    }

    /// Whether the content is currently offline as a result of this takedown.
    pub fn is_content_down(&self) -> bool {
        matches!(
            self.status,
            DmcaStatus::ContentRemoved | DmcaStatus::WaitingPeriod | DmcaStatus::CourtAction
        )
    }

    /// Days elapsed since filing (given current Unix timestamp).
    pub fn days_since_filing(&self, now_ts: i64) -> u64 {
        let elapsed = (now_ts - self.filed_ts).max(0) as u64;
        elapsed / 86_400
    }
}

/// Manages a collection of DMCA takedown requests.
#[derive(Debug, Default)]
pub struct DmcaWorkflow {
    takedowns: Vec<DmcaTakedown>,
}

impl DmcaWorkflow {
    /// Create an empty workflow.
    pub fn new() -> Self {
        Self::default()
    }

    /// File a new takedown request.
    pub fn file(&mut self, takedown: DmcaTakedown) {
        self.takedowns.push(takedown);
    }

    /// Find a takedown by ID.
    pub fn find(&self, id: &str) -> Option<&DmcaTakedown> {
        self.takedowns.iter().find(|t| t.id == id)
    }

    /// Find a mutable takedown by ID.
    pub fn find_mut(&mut self, id: &str) -> Option<&mut DmcaTakedown> {
        self.takedowns.iter_mut().find(|t| t.id == id)
    }

    /// All takedowns currently causing content to be offline.
    pub fn active_takedowns(&self) -> Vec<&DmcaTakedown> {
        self.takedowns
            .iter()
            .filter(|t| t.is_content_down())
            .collect()
    }

    /// Check compliance: any takedowns stuck in WaitingPeriod past the
    /// 14-business-day deadline should be flagged.
    pub fn check_compliance(&self, now_ts: i64) -> Vec<ComplianceIssue> {
        let mut issues = Vec::new();
        // 14 business days ≈ 20 calendar days as conservative estimate
        let max_waiting_days: u64 = 20;

        for td in &self.takedowns {
            if td.status == DmcaStatus::WaitingPeriod {
                if let Some(counter_ts) = td.counter_ts {
                    let days = ((now_ts - counter_ts).max(0) as u64) / 86_400;
                    if days > max_waiting_days {
                        issues.push(ComplianceIssue::new(
                            "dmca_waiting_overdue",
                            IssueSeverity::High,
                            format!(
                                "Takedown {} waiting period exceeded ({} days > {} limit)",
                                td.id, days, max_waiting_days,
                            ),
                            "dmca_takedown",
                            &td.id,
                        ));
                    }
                }
            }

            // Filed but content not yet removed within 24 hours
            if td.status == DmcaStatus::Filed {
                let hours_since = ((now_ts - td.filed_ts).max(0) as u64) / 3600;
                if hours_since > 24 {
                    issues.push(ComplianceIssue::new(
                        "dmca_removal_delayed",
                        IssueSeverity::Critical,
                        format!(
                            "Takedown {} filed {} hours ago but content not yet removed",
                            td.id, hours_since,
                        ),
                        "dmca_takedown",
                        &td.id,
                    ));
                }
            }
        }

        issues
    }

    /// Total number of takedown requests.
    pub fn len(&self) -> usize {
        self.takedowns.len()
    }

    /// Whether there are no takedown requests.
    pub fn is_empty(&self) -> bool {
        self.takedowns.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rights::Asset;

    #[tokio::test]
    async fn test_compliance_check() {
        let temp_dir = tempfile::tempdir().expect("rights test operation should succeed");
        let db_path = format!("sqlite://{}/test.db", temp_dir.path().display());
        let db = RightsDatabase::new(&db_path)
            .await
            .expect("rights test operation should succeed");

        let asset = Asset::new("Test Asset", crate::rights::AssetType::Video);
        let asset_id = asset.id.clone();
        asset
            .save(&db)
            .await
            .expect("rights test operation should succeed");

        let checker = ComplianceChecker::new(&db);
        let issues = checker
            .check_asset(&asset_id)
            .await
            .expect("rights test operation should succeed");

        // Should have at least one issue (no grants)
        assert!(!issues.is_empty());
    }

    // ── GDPR tests ───────────────────────────────────────────────────────

    #[test]
    fn test_gdpr_consent_missing() {
        let checker = GdprChecker::new(5);
        let records = vec![GdprDataRecord {
            subject_id: "user-1".into(),
            categories: vec![PiiCategory::Email],
            lawful_basis: LawfulBasis::Consent,
            purpose: "Newsletter".into(),
            consent_given: false,
            created_ts: 1_000_000,
            retention_deadline_ts: None,
        }];
        let issues = checker.check(&records, 2_000_000);
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].issue_type, "gdpr_consent_missing");
        assert_eq!(issues[0].severity, IssueSeverity::Critical);
    }

    #[test]
    fn test_gdpr_consent_present_no_issue() {
        let checker = GdprChecker::new(5);
        let records = vec![GdprDataRecord {
            subject_id: "user-2".into(),
            categories: vec![PiiCategory::Email],
            lawful_basis: LawfulBasis::Consent,
            purpose: "Newsletter".into(),
            consent_given: true,
            created_ts: 1_000_000,
            retention_deadline_ts: None,
        }];
        let issues = checker.check(&records, 2_000_000);
        assert!(issues.is_empty());
    }

    #[test]
    fn test_gdpr_empty_purpose() {
        let checker = GdprChecker::new(5);
        let records = vec![GdprDataRecord {
            subject_id: "user-3".into(),
            categories: vec![PiiCategory::FullName],
            lawful_basis: LawfulBasis::Contract,
            purpose: "  ".into(),
            consent_given: false,
            created_ts: 1_000_000,
            retention_deadline_ts: None,
        }];
        let issues = checker.check(&records, 2_000_000);
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].issue_type, "gdpr_purpose_empty");
    }

    #[test]
    fn test_gdpr_data_minimisation() {
        let checker = GdprChecker::new(2);
        let records = vec![GdprDataRecord {
            subject_id: "user-4".into(),
            categories: vec![
                PiiCategory::FullName,
                PiiCategory::Email,
                PiiCategory::Phone,
            ],
            lawful_basis: LawfulBasis::LegitimateInterest,
            purpose: "Marketing".into(),
            consent_given: false,
            created_ts: 1_000_000,
            retention_deadline_ts: None,
        }];
        let issues = checker.check(&records, 2_000_000);
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].issue_type, "gdpr_data_minimisation");
    }

    #[test]
    fn test_gdpr_retention_expired() {
        let checker = GdprChecker::new(10);
        let records = vec![GdprDataRecord {
            subject_id: "user-5".into(),
            categories: vec![PiiCategory::Email],
            lawful_basis: LawfulBasis::Contract,
            purpose: "Account".into(),
            consent_given: false,
            created_ts: 1_000_000,
            retention_deadline_ts: Some(1_500_000),
        }];
        let issues = checker.check(&records, 2_000_000);
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].issue_type, "gdpr_retention_expired");
        assert_eq!(issues[0].severity, IssueSeverity::Critical);
    }

    #[test]
    fn test_gdpr_retention_not_expired() {
        let checker = GdprChecker::new(10);
        let records = vec![GdprDataRecord {
            subject_id: "user-6".into(),
            categories: vec![PiiCategory::Email],
            lawful_basis: LawfulBasis::Contract,
            purpose: "Account".into(),
            consent_given: false,
            created_ts: 1_000_000,
            retention_deadline_ts: Some(3_000_000),
        }];
        let issues = checker.check(&records, 2_000_000);
        assert!(issues.is_empty());
    }

    // ── DMCA tests ───────────────────────────────────────────────────────

    #[test]
    fn test_dmca_takedown_lifecycle() {
        let mut td = DmcaTakedown::new("td-1", "asset-1", "Claimant X", "Infringement", 1000);
        assert_eq!(td.status, DmcaStatus::Filed);
        assert!(!td.is_content_down());

        td.remove_content(2000);
        assert_eq!(td.status, DmcaStatus::ContentRemoved);
        assert!(td.is_content_down());

        td.file_counter_notification(3000);
        assert_eq!(td.status, DmcaStatus::CounterNotification);

        td.begin_waiting_period();
        assert!(td.is_content_down());

        td.restore_content(5000);
        assert_eq!(td.status, DmcaStatus::ContentRestored);
        assert!(!td.is_content_down());

        td.resolve();
        assert_eq!(td.status, DmcaStatus::Resolved);
    }

    #[test]
    fn test_dmca_days_since_filing() {
        let td = DmcaTakedown::new("td-2", "asset-2", "C", "Desc", 1_000_000);
        // 86400 seconds = 1 day
        assert_eq!(td.days_since_filing(1_000_000 + 86_400), 1);
        assert_eq!(td.days_since_filing(1_000_000 + 86_400 * 10), 10);
    }

    #[test]
    fn test_dmca_workflow_file_and_find() {
        let mut wf = DmcaWorkflow::new();
        wf.file(DmcaTakedown::new("td-3", "a", "C", "D", 1000));
        assert_eq!(wf.len(), 1);
        assert!(wf.find("td-3").is_some());
        assert!(wf.find("nonexistent").is_none());
    }

    #[test]
    fn test_dmca_active_takedowns() {
        let mut wf = DmcaWorkflow::new();
        let mut td1 = DmcaTakedown::new("td-4", "a", "C", "D", 1000);
        td1.remove_content(2000);
        wf.file(td1);

        let td2 = DmcaTakedown::new("td-5", "b", "C", "D", 1000);
        wf.file(td2); // Still Filed, not content-down

        assert_eq!(wf.active_takedowns().len(), 1);
    }

    #[test]
    fn test_dmca_compliance_removal_delayed() {
        let mut wf = DmcaWorkflow::new();
        wf.file(DmcaTakedown::new("td-6", "a", "C", "D", 1000));
        // 48 hours later, still not removed
        let issues = wf.check_compliance(1000 + 48 * 3600);
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].issue_type, "dmca_removal_delayed");
    }

    #[test]
    fn test_dmca_compliance_waiting_overdue() {
        let mut wf = DmcaWorkflow::new();
        let mut td = DmcaTakedown::new("td-7", "a", "C", "D", 1000);
        td.remove_content(2000);
        td.file_counter_notification(3000);
        td.begin_waiting_period();
        wf.file(td);

        // 25 days after counter-notification
        let issues = wf.check_compliance(3000 + 25 * 86_400);
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].issue_type, "dmca_waiting_overdue");
    }
}
