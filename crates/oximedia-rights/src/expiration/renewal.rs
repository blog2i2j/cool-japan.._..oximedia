//! Renewal management

use crate::{database::RightsDatabase, expiration::ExpirationTracker, rights::RightsGrant, Result};
use chrono::{DateTime, Duration, Utc};

/// Renewal manager
pub struct RenewalManager<'a> {
    db: &'a RightsDatabase,
}

impl<'a> RenewalManager<'a> {
    /// Create a new renewal manager
    pub fn new(db: &'a RightsDatabase) -> Self {
        Self { db }
    }

    /// Renew a grant with a new end date
    pub async fn renew_grant(
        &self,
        grant_id: &str,
        new_end_date: DateTime<Utc>,
    ) -> Result<RightsGrant> {
        let mut grant = RightsGrant::load(self.db, grant_id)
            .await?
            .ok_or_else(|| crate::RightsError::NotFound(format!("Grant {grant_id} not found")))?;

        grant.end_date = Some(new_end_date);
        grant.updated_at = Utc::now();
        grant.save(self.db).await?;

        Ok(grant)
    }

    /// Extend a grant by a number of days
    pub async fn extend_grant(&self, grant_id: &str, days: i64) -> Result<RightsGrant> {
        let grant = RightsGrant::load(self.db, grant_id)
            .await?
            .ok_or_else(|| crate::RightsError::NotFound(format!("Grant {grant_id} not found")))?;

        let current_end = grant.end_date.unwrap_or_else(Utc::now);
        let new_end_date = current_end + Duration::days(days);

        self.renew_grant(grant_id, new_end_date).await
    }

    /// Get grants eligible for renewal (expiring soon)
    pub async fn get_renewal_candidates(&self, within_days: i64) -> Result<Vec<RightsGrant>> {
        let tracker = ExpirationTracker::new(self.db);
        tracker.get_expiring_grants(within_days).await
    }

    /// Convert grant to perpetual (remove end date)
    pub async fn make_perpetual(&self, grant_id: &str) -> Result<RightsGrant> {
        let mut grant = RightsGrant::load(self.db, grant_id)
            .await?
            .ok_or_else(|| crate::RightsError::NotFound(format!("Grant {grant_id} not found")))?;

        grant.end_date = None;
        grant.updated_at = Utc::now();
        grant.save(self.db).await?;

        Ok(grant)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::license::LicenseType;

    #[tokio::test]
    async fn test_extend_grant() {
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

        let now = Utc::now();
        let grant = RightsGrant::new(
            &asset.id,
            &owner.id,
            LicenseType::NonExclusive,
            now,
            Some(now + Duration::days(30)),
            false,
        );
        let grant_id = grant.id.clone();
        grant
            .save(&db)
            .await
            .expect("rights test operation should succeed");

        let manager = RenewalManager::new(&db);
        let renewed = manager
            .extend_grant(&grant_id, 30)
            .await
            .expect("rights test operation should succeed");

        assert!(renewed.end_date.is_some());
        let days_diff = (renewed
            .end_date
            .expect("rights test operation should succeed")
            - now)
            .num_days();
        assert!((59..=61).contains(&days_diff)); // Around 60 days
    }
}
