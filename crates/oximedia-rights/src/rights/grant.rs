//! Rights grants management

use crate::{
    database::RightsDatabase,
    license::LicenseType,
    rights::{UsageRestriction, UsageType},
    territory::TerritoryRestriction,
    Result, RightsError,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::Row;
use uuid::Uuid;

/// A grant of rights from an owner for an asset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RightsGrant {
    /// Unique identifier
    pub id: String,
    /// Asset ID
    pub asset_id: String,
    /// Owner ID
    pub owner_id: String,
    /// License type
    pub license_type: LicenseType,
    /// Start date of the grant
    pub start_date: DateTime<Utc>,
    /// End date of the grant (None = perpetual)
    pub end_date: Option<DateTime<Utc>>,
    /// Whether the grant is exclusive
    pub is_exclusive: bool,
    /// Territory restrictions
    pub territory: Option<TerritoryRestriction>,
    /// Usage restrictions
    pub usage_restrictions: Option<UsageRestriction>,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last update timestamp
    pub updated_at: DateTime<Utc>,
}

impl RightsGrant {
    /// Create a new rights grant
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        asset_id: impl Into<String>,
        owner_id: impl Into<String>,
        license_type: LicenseType,
        start_date: DateTime<Utc>,
        end_date: Option<DateTime<Utc>>,
        is_exclusive: bool,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            asset_id: asset_id.into(),
            owner_id: owner_id.into(),
            license_type,
            start_date,
            end_date,
            is_exclusive,
            territory: None,
            usage_restrictions: None,
            created_at: now,
            updated_at: now,
        }
    }

    /// Set territory restrictions
    pub fn with_territory(mut self, territory: TerritoryRestriction) -> Self {
        self.territory = Some(territory);
        self
    }

    /// Set usage restrictions
    pub fn with_usage_restrictions(mut self, restrictions: UsageRestriction) -> Self {
        self.usage_restrictions = Some(restrictions);
        self
    }

    /// Check if the grant is currently active
    pub fn is_active(&self) -> bool {
        let now = Utc::now();
        if now < self.start_date {
            return false;
        }
        if let Some(end_date) = self.end_date {
            if now > end_date {
                return false;
            }
        }
        true
    }

    /// Check if the grant has expired
    pub fn is_expired(&self) -> bool {
        if let Some(end_date) = self.end_date {
            Utc::now() > end_date
        } else {
            false
        }
    }

    /// Check if a usage is allowed under this grant
    pub fn allows_usage(&self, usage: &UsageType, territory: Option<&str>) -> Result<()> {
        // Check if grant is active
        if !self.is_active() {
            return Err(RightsError::Expired("Rights grant has expired".to_string()));
        }

        // Check territory restrictions
        if let Some(territory_code) = territory {
            if let Some(ref territory_restriction) = self.territory {
                if !territory_restriction.is_allowed(territory_code) {
                    return Err(RightsError::TerritoryViolation(format!(
                        "Territory {territory_code} not allowed"
                    )));
                }
            }
        }

        // Check usage restrictions
        if let Some(ref restrictions) = self.usage_restrictions {
            if !restrictions.is_usage_allowed(usage) {
                return Err(RightsError::UsageViolation(format!(
                    "Usage type {usage:?} not allowed"
                )));
            }
        }

        Ok(())
    }

    /// Save grant to database
    pub async fn save(&self, db: &RightsDatabase) -> Result<()> {
        let territory_json = self
            .territory
            .as_ref()
            .map(|t| serde_json::to_string(t).unwrap_or_default());

        let usage_json = self
            .usage_restrictions
            .as_ref()
            .map(|u| serde_json::to_string(u).unwrap_or_default());

        sqlx::query(
            r"
            INSERT INTO rights_grants
            (id, asset_id, owner_id, license_type, start_date, end_date,
             is_exclusive, territory_json, usage_restrictions_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                asset_id = excluded.asset_id,
                owner_id = excluded.owner_id,
                license_type = excluded.license_type,
                start_date = excluded.start_date,
                end_date = excluded.end_date,
                is_exclusive = excluded.is_exclusive,
                territory_json = excluded.territory_json,
                usage_restrictions_json = excluded.usage_restrictions_json,
                updated_at = excluded.updated_at
            ",
        )
        .bind(&self.id)
        .bind(&self.asset_id)
        .bind(&self.owner_id)
        .bind(self.license_type.as_str())
        .bind(self.start_date.to_rfc3339())
        .bind(self.end_date.map(|d| d.to_rfc3339()))
        .bind(self.is_exclusive as i32)
        .bind(territory_json)
        .bind(usage_json)
        .bind(self.created_at.to_rfc3339())
        .bind(self.updated_at.to_rfc3339())
        .execute(db.pool())
        .await?;

        Ok(())
    }

    /// Load grant from database by ID
    pub async fn load(db: &RightsDatabase, id: &str) -> Result<Option<Self>> {
        let row = sqlx::query(
            r"
            SELECT id, asset_id, owner_id, license_type, start_date, end_date,
                   is_exclusive, territory_json, usage_restrictions_json, created_at, updated_at
            FROM rights_grants WHERE id = ?
            ",
        )
        .bind(id)
        .fetch_optional(db.pool())
        .await?;

        row.map(|r| {
            let territory_json: Option<String> = r.get("territory_json");
            let territory = territory_json.and_then(|json| serde_json::from_str(&json).ok());

            let usage_json: Option<String> = r.get("usage_restrictions_json");
            let usage_restrictions = usage_json.and_then(|json| serde_json::from_str(&json).ok());

            let start_date = DateTime::parse_from_rfc3339(r.get("start_date"))
                .map_err(|e| RightsError::InvalidLicense(format!("Invalid start_date: {e}")))?
                .with_timezone(&Utc);
            let end_date = r
                .get::<Option<String>, _>("end_date")
                .map(|s| {
                    DateTime::parse_from_rfc3339(&s)
                        .map_err(|e| RightsError::InvalidLicense(format!("Invalid end_date: {e}")))
                        .map(|dt| dt.with_timezone(&Utc))
                })
                .transpose()?;
            let created_at = DateTime::parse_from_rfc3339(r.get("created_at"))
                .map_err(|e| RightsError::InvalidLicense(format!("Invalid created_at: {e}")))?
                .with_timezone(&Utc);
            let updated_at = DateTime::parse_from_rfc3339(r.get("updated_at"))
                .map_err(|e| RightsError::InvalidLicense(format!("Invalid updated_at: {e}")))?
                .with_timezone(&Utc);

            Ok(RightsGrant {
                id: r.get("id"),
                asset_id: r.get("asset_id"),
                owner_id: r.get("owner_id"),
                license_type: LicenseType::from_str(r.get("license_type")),
                start_date,
                end_date,
                is_exclusive: r.get::<i32, _>("is_exclusive") != 0,
                territory,
                usage_restrictions,
                created_at,
                updated_at,
            })
        })
        .transpose()
    }

    /// List all grants for an asset
    pub async fn list_for_asset(db: &RightsDatabase, asset_id: &str) -> Result<Vec<Self>> {
        let rows = sqlx::query(
            r"
            SELECT id, asset_id, owner_id, license_type, start_date, end_date,
                   is_exclusive, territory_json, usage_restrictions_json, created_at, updated_at
            FROM rights_grants WHERE asset_id = ?
            ORDER BY created_at DESC
            ",
        )
        .bind(asset_id)
        .fetch_all(db.pool())
        .await?;

        rows.into_iter()
            .map(|r| {
                let territory_json: Option<String> = r.get("territory_json");
                let territory = territory_json.and_then(|json| serde_json::from_str(&json).ok());

                let usage_json: Option<String> = r.get("usage_restrictions_json");
                let usage_restrictions =
                    usage_json.and_then(|json| serde_json::from_str(&json).ok());

                let start_date = DateTime::parse_from_rfc3339(r.get("start_date"))
                    .map_err(|e| RightsError::InvalidLicense(format!("Invalid start_date: {e}")))?
                    .with_timezone(&Utc);
                let end_date = r
                    .get::<Option<String>, _>("end_date")
                    .map(|s| {
                        DateTime::parse_from_rfc3339(&s)
                            .map_err(|e| {
                                RightsError::InvalidLicense(format!("Invalid end_date: {e}"))
                            })
                            .map(|dt| dt.with_timezone(&Utc))
                    })
                    .transpose()?;
                let created_at = DateTime::parse_from_rfc3339(r.get("created_at"))
                    .map_err(|e| RightsError::InvalidLicense(format!("Invalid created_at: {e}")))?
                    .with_timezone(&Utc);
                let updated_at = DateTime::parse_from_rfc3339(r.get("updated_at"))
                    .map_err(|e| RightsError::InvalidLicense(format!("Invalid updated_at: {e}")))?
                    .with_timezone(&Utc);

                Ok(RightsGrant {
                    id: r.get("id"),
                    asset_id: r.get("asset_id"),
                    owner_id: r.get("owner_id"),
                    license_type: LicenseType::from_str(r.get("license_type")),
                    start_date,
                    end_date,
                    is_exclusive: r.get::<i32, _>("is_exclusive") != 0,
                    territory,
                    usage_restrictions,
                    created_at,
                    updated_at,
                })
            })
            .collect()
    }

    /// Delete grant from database
    pub async fn delete(db: &RightsDatabase, id: &str) -> Result<()> {
        sqlx::query("DELETE FROM rights_grants WHERE id = ?")
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
    fn test_grant_active_status() {
        let now = Utc::now();
        let grant = RightsGrant::new(
            "asset1",
            "owner1",
            LicenseType::Exclusive,
            now - chrono::Duration::days(1),
            Some(now + chrono::Duration::days(1)),
            true,
        );

        assert!(grant.is_active());
        assert!(!grant.is_expired());
    }

    #[test]
    fn test_grant_expired() {
        let now = Utc::now();
        let grant = RightsGrant::new(
            "asset1",
            "owner1",
            LicenseType::Exclusive,
            now - chrono::Duration::days(10),
            Some(now - chrono::Duration::days(1)),
            true,
        );

        assert!(!grant.is_active());
        assert!(grant.is_expired());
    }

    #[test]
    fn test_grant_not_started() {
        let now = Utc::now();
        let grant = RightsGrant::new(
            "asset1",
            "owner1",
            LicenseType::Exclusive,
            now + chrono::Duration::days(1),
            Some(now + chrono::Duration::days(10)),
            true,
        );

        assert!(!grant.is_active());
        assert!(!grant.is_expired());
    }

    #[test]
    fn test_perpetual_grant() {
        let now = Utc::now();
        let grant = RightsGrant::new(
            "asset1",
            "owner1",
            LicenseType::RoyaltyFree,
            now - chrono::Duration::days(1),
            None,
            false,
        );

        assert!(grant.is_active());
        assert!(!grant.is_expired());
    }
}
