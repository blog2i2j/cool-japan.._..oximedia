//! License agreement tracking

use crate::{database::RightsDatabase, license::LicenseTerms, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::Row;
use uuid::Uuid;

/// License agreement status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AgreementStatus {
    /// Draft agreement
    Draft,
    /// Pending signature
    Pending,
    /// Active agreement
    Active,
    /// Expired agreement
    Expired,
    /// Terminated agreement
    Terminated,
    /// Cancelled agreement
    Cancelled,
}

impl AgreementStatus {
    /// Convert to string representation
    pub fn as_str(&self) -> &str {
        match self {
            AgreementStatus::Draft => "draft",
            AgreementStatus::Pending => "pending",
            AgreementStatus::Active => "active",
            AgreementStatus::Expired => "expired",
            AgreementStatus::Terminated => "terminated",
            AgreementStatus::Cancelled => "cancelled",
        }
    }

    /// Parse from string
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Self {
        match s {
            "draft" => AgreementStatus::Draft,
            "pending" => AgreementStatus::Pending,
            "active" => AgreementStatus::Active,
            "expired" => AgreementStatus::Expired,
            "terminated" => AgreementStatus::Terminated,
            "cancelled" => AgreementStatus::Cancelled,
            _ => AgreementStatus::Draft,
        }
    }
}

/// License agreement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LicenseAgreement {
    /// Unique identifier
    pub id: String,
    /// Associated rights grant ID
    pub grant_id: String,
    /// Agreement number (human-readable)
    pub agreement_number: String,
    /// License terms
    pub terms: LicenseTerms,
    /// Agreement status
    pub status: AgreementStatus,
    /// Signed date
    pub signed_date: Option<DateTime<Utc>>,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last update timestamp
    pub updated_at: DateTime<Utc>,
}

impl LicenseAgreement {
    /// Create a new license agreement
    pub fn new(grant_id: impl Into<String>, agreement_number: impl Into<String>) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            grant_id: grant_id.into(),
            agreement_number: agreement_number.into(),
            terms: LicenseTerms::default(),
            status: AgreementStatus::Draft,
            signed_date: None,
            created_at: now,
            updated_at: now,
        }
    }

    /// Set license terms
    pub fn with_terms(mut self, terms: LicenseTerms) -> Self {
        self.terms = terms;
        self
    }

    /// Mark as pending signature
    pub fn mark_pending(&mut self) {
        self.status = AgreementStatus::Pending;
        self.updated_at = Utc::now();
    }

    /// Sign the agreement
    pub fn sign(&mut self) {
        self.status = AgreementStatus::Active;
        self.signed_date = Some(Utc::now());
        self.updated_at = Utc::now();
    }

    /// Terminate the agreement
    pub fn terminate(&mut self) {
        self.status = AgreementStatus::Terminated;
        self.updated_at = Utc::now();
    }

    /// Cancel the agreement
    pub fn cancel(&mut self) {
        self.status = AgreementStatus::Cancelled;
        self.updated_at = Utc::now();
    }

    /// Save agreement to database
    pub async fn save(&self, db: &RightsDatabase) -> Result<()> {
        let terms_json = serde_json::to_string(&self.terms)
            .map_err(|e| crate::RightsError::Serialization(e.to_string()))?;

        sqlx::query(
            r"
            INSERT INTO license_agreements
            (id, grant_id, agreement_number, terms_json, status, signed_date, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                grant_id = excluded.grant_id,
                agreement_number = excluded.agreement_number,
                terms_json = excluded.terms_json,
                status = excluded.status,
                signed_date = excluded.signed_date,
                updated_at = excluded.updated_at
            ",
        )
        .bind(&self.id)
        .bind(&self.grant_id)
        .bind(&self.agreement_number)
        .bind(&terms_json)
        .bind(self.status.as_str())
        .bind(self.signed_date.map(|d| d.to_rfc3339()))
        .bind(self.created_at.to_rfc3339())
        .bind(self.updated_at.to_rfc3339())
        .execute(db.pool())
        .await?;

        Ok(())
    }

    /// Load agreement from database by ID
    pub async fn load(db: &RightsDatabase, id: &str) -> Result<Option<Self>> {
        let row = sqlx::query(
            r"
            SELECT id, grant_id, agreement_number, terms_json, status, signed_date, created_at, updated_at
            FROM license_agreements WHERE id = ?
            ",
        )
        .bind(id)
        .fetch_optional(db.pool())
        .await?;

        let agreement = match row {
            None => return Ok(None),
            Some(r) => {
                let terms_json: String = r.get("terms_json");
                let terms = serde_json::from_str(&terms_json).unwrap_or_default();

                let signed_date = r
                    .get::<Option<String>, _>("signed_date")
                    .map(|s| {
                        DateTime::parse_from_rfc3339(&s)
                            .map(|dt| dt.with_timezone(&Utc))
                            .map_err(|e| crate::RightsError::Serialization(e.to_string()))
                    })
                    .transpose()?;

                let created_at = DateTime::parse_from_rfc3339(r.get("created_at"))
                    .map(|dt| dt.with_timezone(&Utc))
                    .map_err(|e| crate::RightsError::Serialization(e.to_string()))?;

                let updated_at = DateTime::parse_from_rfc3339(r.get("updated_at"))
                    .map(|dt| dt.with_timezone(&Utc))
                    .map_err(|e| crate::RightsError::Serialization(e.to_string()))?;

                LicenseAgreement {
                    id: r.get("id"),
                    grant_id: r.get("grant_id"),
                    agreement_number: r.get("agreement_number"),
                    terms,
                    status: AgreementStatus::from_str(r.get("status")),
                    signed_date,
                    created_at,
                    updated_at,
                }
            }
        };

        Ok(Some(agreement))
    }

    /// List agreements for a grant
    pub async fn list_for_grant(db: &RightsDatabase, grant_id: &str) -> Result<Vec<Self>> {
        let rows = sqlx::query(
            r"
            SELECT id, grant_id, agreement_number, terms_json, status, signed_date, created_at, updated_at
            FROM license_agreements WHERE grant_id = ?
            ORDER BY created_at DESC
            ",
        )
        .bind(grant_id)
        .fetch_all(db.pool())
        .await?;

        rows.into_iter()
            .map(|r| {
                let terms_json: String = r.get("terms_json");
                let terms = serde_json::from_str(&terms_json).unwrap_or_default();

                let signed_date = r
                    .get::<Option<String>, _>("signed_date")
                    .map(|s| {
                        DateTime::parse_from_rfc3339(&s)
                            .map(|dt| dt.with_timezone(&Utc))
                            .map_err(|e| crate::RightsError::Serialization(e.to_string()))
                    })
                    .transpose()?;

                let created_at = DateTime::parse_from_rfc3339(r.get("created_at"))
                    .map(|dt| dt.with_timezone(&Utc))
                    .map_err(|e| crate::RightsError::Serialization(e.to_string()))?;

                let updated_at = DateTime::parse_from_rfc3339(r.get("updated_at"))
                    .map(|dt| dt.with_timezone(&Utc))
                    .map_err(|e| crate::RightsError::Serialization(e.to_string()))?;

                Ok(LicenseAgreement {
                    id: r.get("id"),
                    grant_id: r.get("grant_id"),
                    agreement_number: r.get("agreement_number"),
                    terms,
                    status: AgreementStatus::from_str(r.get("status")),
                    signed_date,
                    created_at,
                    updated_at,
                })
            })
            .collect()
    }

    /// Delete agreement from database
    pub async fn delete(db: &RightsDatabase, id: &str) -> Result<()> {
        sqlx::query("DELETE FROM license_agreements WHERE id = ?")
            .bind(id)
            .execute(db.pool())
            .await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agreement_creation() {
        let agreement = LicenseAgreement::new("grant123", "AGR-2024-001");
        assert_eq!(agreement.grant_id, "grant123");
        assert_eq!(agreement.agreement_number, "AGR-2024-001");
        assert_eq!(agreement.status, AgreementStatus::Draft);
    }

    #[test]
    fn test_agreement_workflow() {
        let mut agreement = LicenseAgreement::new("grant123", "AGR-2024-001");

        agreement.mark_pending();
        assert_eq!(agreement.status, AgreementStatus::Pending);

        agreement.sign();
        assert_eq!(agreement.status, AgreementStatus::Active);
        assert!(agreement.signed_date.is_some());

        agreement.terminate();
        assert_eq!(agreement.status, AgreementStatus::Terminated);
    }

    #[tokio::test]
    async fn test_agreement_save_and_load() {
        let temp_dir = tempfile::tempdir().expect("rights test operation should succeed");
        let db_path = format!("sqlite://{}/test.db", temp_dir.path().display());
        let db = RightsDatabase::new(&db_path)
            .await
            .expect("rights test operation should succeed");

        // Create asset and owner first
        let asset = crate::rights::Asset::new("Test Asset", crate::rights::AssetType::Video);
        asset
            .save(&db)
            .await
            .expect("rights test operation should succeed");
        let owner = crate::rights::RightsOwner::new("Test Owner");
        owner
            .save(&db)
            .await
            .expect("rights test operation should succeed");

        // Create grant
        let grant = crate::rights::RightsGrant::new(
            &asset.id,
            &owner.id,
            crate::license::LicenseType::Exclusive,
            chrono::Utc::now(),
            Some(chrono::Utc::now() + chrono::Duration::days(30)),
            true,
        );
        let grant_id = grant.id.clone();
        grant
            .save(&db)
            .await
            .expect("rights test operation should succeed");

        let agreement = LicenseAgreement::new(&grant_id, "AGR-2024-001");
        let agreement_id = agreement.id.clone();

        agreement
            .save(&db)
            .await
            .expect("rights test operation should succeed");

        let loaded = LicenseAgreement::load(&db, &agreement_id)
            .await
            .expect("rights test operation should succeed");
        assert!(loaded.is_some());
        let loaded = loaded.expect("rights test operation should succeed");
        assert_eq!(loaded.agreement_number, "AGR-2024-001");
    }
}
