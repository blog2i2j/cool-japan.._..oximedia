//! Expiration tracking

use crate::{database::RightsDatabase, rights::RightsGrant, Result};
use chrono::{DateTime, Duration, Utc};
use sqlx::Row;

/// Expiration tracker
pub struct ExpirationTracker<'a> {
    db: &'a RightsDatabase,
}

impl<'a> ExpirationTracker<'a> {
    /// Create a new expiration tracker
    pub fn new(db: &'a RightsDatabase) -> Self {
        Self { db }
    }

    /// Get all expiring grants within the specified number of days
    pub async fn get_expiring_grants(&self, days: i64) -> Result<Vec<RightsGrant>> {
        let now = Utc::now();
        let threshold = now + Duration::days(days);

        let rows = sqlx::query(
            r"
            SELECT id, asset_id, owner_id, license_type, start_date, end_date,
                   is_exclusive, territory_json, usage_restrictions_json, created_at, updated_at
            FROM rights_grants
            WHERE end_date IS NOT NULL
              AND end_date <= ?
              AND end_date > ?
            ORDER BY end_date ASC
            ",
        )
        .bind(threshold.to_rfc3339())
        .bind(now.to_rfc3339())
        .fetch_all(self.db.pool())
        .await?;

        rows.into_iter()
            .map(|r| {
                let territory_json: Option<String> = r.get("territory_json");
                let territory = territory_json.and_then(|json| serde_json::from_str(&json).ok());

                let usage_json: Option<String> = r.get("usage_restrictions_json");
                let usage_restrictions =
                    usage_json.and_then(|json| serde_json::from_str(&json).ok());

                let start_date = DateTime::parse_from_rfc3339(r.get("start_date"))
                    .map_err(|e| {
                        crate::RightsError::InvalidLicense(format!("Invalid start_date: {e}"))
                    })?
                    .with_timezone(&Utc);
                let end_date = r
                    .get::<Option<String>, _>("end_date")
                    .map(|s| {
                        DateTime::parse_from_rfc3339(&s)
                            .map_err(|e| {
                                crate::RightsError::InvalidLicense(format!("Invalid end_date: {e}"))
                            })
                            .map(|dt| dt.with_timezone(&Utc))
                    })
                    .transpose()?;
                let created_at = DateTime::parse_from_rfc3339(r.get("created_at"))
                    .map_err(|e| {
                        crate::RightsError::InvalidLicense(format!("Invalid created_at: {e}"))
                    })?
                    .with_timezone(&Utc);
                let updated_at = DateTime::parse_from_rfc3339(r.get("updated_at"))
                    .map_err(|e| {
                        crate::RightsError::InvalidLicense(format!("Invalid updated_at: {e}"))
                    })?
                    .with_timezone(&Utc);

                Ok(RightsGrant {
                    id: r.get("id"),
                    asset_id: r.get("asset_id"),
                    owner_id: r.get("owner_id"),
                    license_type: crate::license::LicenseType::from_str(r.get("license_type")),
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

    /// Get all expired grants
    pub async fn get_expired_grants(&self) -> Result<Vec<RightsGrant>> {
        let now = Utc::now();

        let rows = sqlx::query(
            r"
            SELECT id, asset_id, owner_id, license_type, start_date, end_date,
                   is_exclusive, territory_json, usage_restrictions_json, created_at, updated_at
            FROM rights_grants
            WHERE end_date IS NOT NULL AND end_date <= ?
            ORDER BY end_date DESC
            ",
        )
        .bind(now.to_rfc3339())
        .fetch_all(self.db.pool())
        .await?;

        rows.into_iter()
            .map(|r| {
                let territory_json: Option<String> = r.get("territory_json");
                let territory = territory_json.and_then(|json| serde_json::from_str(&json).ok());

                let usage_json: Option<String> = r.get("usage_restrictions_json");
                let usage_restrictions =
                    usage_json.and_then(|json| serde_json::from_str(&json).ok());

                let start_date = DateTime::parse_from_rfc3339(r.get("start_date"))
                    .map_err(|e| {
                        crate::RightsError::InvalidLicense(format!("Invalid start_date: {e}"))
                    })?
                    .with_timezone(&Utc);
                let end_date = r
                    .get::<Option<String>, _>("end_date")
                    .map(|s| {
                        DateTime::parse_from_rfc3339(&s)
                            .map_err(|e| {
                                crate::RightsError::InvalidLicense(format!("Invalid end_date: {e}"))
                            })
                            .map(|dt| dt.with_timezone(&Utc))
                    })
                    .transpose()?;
                let created_at = DateTime::parse_from_rfc3339(r.get("created_at"))
                    .map_err(|e| {
                        crate::RightsError::InvalidLicense(format!("Invalid created_at: {e}"))
                    })?
                    .with_timezone(&Utc);
                let updated_at = DateTime::parse_from_rfc3339(r.get("updated_at"))
                    .map_err(|e| {
                        crate::RightsError::InvalidLicense(format!("Invalid updated_at: {e}"))
                    })?
                    .with_timezone(&Utc);

                Ok(RightsGrant {
                    id: r.get("id"),
                    asset_id: r.get("asset_id"),
                    owner_id: r.get("owner_id"),
                    license_type: crate::license::LicenseType::from_str(r.get("license_type")),
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

    /// Check if a grant is about to expire (within specified days)
    pub fn is_expiring_soon(grant: &RightsGrant, days: i64) -> bool {
        if let Some(end_date) = grant.end_date {
            let now = Utc::now();
            let threshold = now + Duration::days(days);
            end_date <= threshold && end_date > now
        } else {
            false
        }
    }

    /// Get days until expiration (negative if expired)
    pub fn days_until_expiration(grant: &RightsGrant) -> Option<i64> {
        grant.end_date.map(|end_date| {
            let now = Utc::now();
            (end_date - now).num_days()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::license::LicenseType;

    #[test]
    fn test_is_expiring_soon() {
        let now = Utc::now();

        // Grant expiring in 5 days
        let grant = RightsGrant::new(
            "asset1",
            "owner1",
            LicenseType::Exclusive,
            now - Duration::days(10),
            Some(now + Duration::days(5)),
            true,
        );

        assert!(ExpirationTracker::is_expiring_soon(&grant, 7));
        assert!(!ExpirationTracker::is_expiring_soon(&grant, 3));
    }

    #[test]
    fn test_days_until_expiration() {
        let now = Utc::now();

        let grant = RightsGrant::new(
            "asset1",
            "owner1",
            LicenseType::Exclusive,
            now - Duration::days(10),
            Some(now + Duration::days(5)),
            true,
        );

        let days = ExpirationTracker::days_until_expiration(&grant);
        assert!(days.is_some());
        assert!(
            days.expect("rights test operation should succeed") >= 4
                && days.expect("rights test operation should succeed") <= 5
        );
    }

    #[test]
    fn test_perpetual_grant() {
        let now = Utc::now();

        let grant = RightsGrant::new(
            "asset1",
            "owner1",
            LicenseType::RoyaltyFree,
            now - Duration::days(10),
            None,
            false,
        );

        assert!(!ExpirationTracker::is_expiring_soon(&grant, 30));
        assert!(ExpirationTracker::days_until_expiration(&grant).is_none());
    }
}
