//! Compliance report generation

use crate::compliance::{ComplianceChecker, ComplianceIssue, IssueSeverity};
use crate::{database::RightsDatabase, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Compliance report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceReport {
    /// Report generation timestamp
    pub generated_at: DateTime<Utc>,
    /// Total issues found
    pub total_issues: usize,
    /// Critical issues count
    pub critical_count: usize,
    /// High severity count
    pub high_count: usize,
    /// Medium severity count
    pub medium_count: usize,
    /// Low severity count
    pub low_count: usize,
    /// All issues
    pub issues: Vec<ComplianceIssue>,
}

impl ComplianceReport {
    /// Generate a compliance report
    pub async fn generate(db: &RightsDatabase) -> Result<Self> {
        let checker = ComplianceChecker::new(db);
        let issues = checker.check_all().await?;

        let critical_count = issues
            .iter()
            .filter(|i| matches!(i.severity, IssueSeverity::Critical))
            .count();

        let high_count = issues
            .iter()
            .filter(|i| matches!(i.severity, IssueSeverity::High))
            .count();

        let medium_count = issues
            .iter()
            .filter(|i| matches!(i.severity, IssueSeverity::Medium))
            .count();

        let low_count = issues
            .iter()
            .filter(|i| matches!(i.severity, IssueSeverity::Low))
            .count();

        Ok(Self {
            generated_at: Utc::now(),
            total_issues: issues.len(),
            critical_count,
            high_count,
            medium_count,
            low_count,
            issues,
        })
    }

    /// Export report to JSON
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| crate::RightsError::Serialization(e.to_string()))
    }

    /// Check if there are any critical issues
    pub fn has_critical_issues(&self) -> bool {
        self.critical_count > 0
    }

    /// Get summary string
    pub fn summary(&self) -> String {
        format!(
            "Compliance Report: {} total issues ({} critical, {} high, {} medium, {} low)",
            self.total_issues,
            self.critical_count,
            self.high_count,
            self.medium_count,
            self.low_count
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rights::Asset;

    #[tokio::test]
    async fn test_compliance_report() {
        let temp_dir = tempfile::tempdir().expect("rights test operation should succeed");
        let db_path = format!("sqlite://{}/test.db", temp_dir.path().display());
        let db = RightsDatabase::new(&db_path)
            .await
            .expect("rights test operation should succeed");

        let asset = Asset::new("Test Asset", crate::rights::AssetType::Video);
        asset
            .save(&db)
            .await
            .expect("rights test operation should succeed");

        let report = ComplianceReport::generate(&db)
            .await
            .expect("rights test operation should succeed");
        assert!(report.total_issues > 0);
    }

    #[tokio::test]
    async fn test_report_json_export() {
        let temp_dir = tempfile::tempdir().expect("rights test operation should succeed");
        let db_path = format!("sqlite://{}/test.db", temp_dir.path().display());
        let db = RightsDatabase::new(&db_path)
            .await
            .expect("rights test operation should succeed");

        let report = ComplianceReport::generate(&db)
            .await
            .expect("rights test operation should succeed");
        let json = report
            .to_json()
            .expect("rights test operation should succeed");
        assert!(json.contains("generated_at"));
    }
}
