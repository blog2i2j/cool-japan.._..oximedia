//! Usage tracking

use crate::{database::RightsDatabase, rights::UsageType, usage::UsageLog, Result};
use chrono::{DateTime, Utc};
use sqlx::Row;

/// Usage tracker
pub struct UsageTracker<'a> {
    db: &'a RightsDatabase,
}

impl<'a> UsageTracker<'a> {
    /// Create a new usage tracker
    pub fn new(db: &'a RightsDatabase) -> Self {
        Self { db }
    }

    /// Track usage of an asset
    pub async fn track_usage(
        &self,
        asset_id: &str,
        usage_type: UsageType,
        grant_id: Option<&str>,
        territory: Option<&str>,
    ) -> Result<UsageLog> {
        let mut log = UsageLog::new(asset_id, usage_type, Utc::now());

        if let Some(gid) = grant_id {
            log = log.with_grant(gid);
        }

        if let Some(terr) = territory {
            log = log.with_territory(terr);
        }

        log.save(self.db).await?;
        Ok(log)
    }

    /// Get usage count for an asset
    pub async fn get_usage_count(&self, asset_id: &str) -> Result<u32> {
        let row = sqlx::query("SELECT COUNT(*) as count FROM usage_logs WHERE asset_id = ?")
            .bind(asset_id)
            .fetch_one(self.db.pool())
            .await?;

        Ok(row.get::<i64, _>("count") as u32)
    }

    /// Get usage count for an asset within a date range
    pub async fn get_usage_count_in_range(
        &self,
        asset_id: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<u32> {
        let row = sqlx::query(
            "SELECT COUNT(*) as count FROM usage_logs WHERE asset_id = ? AND usage_date >= ? AND usage_date <= ?"
        )
        .bind(asset_id)
        .bind(start.to_rfc3339())
        .bind(end.to_rfc3339())
        .fetch_one(self.db.pool())
        .await?;

        Ok(row.get::<i64, _>("count") as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_track_usage() {
        let temp_dir = tempfile::tempdir().expect("rights test operation should succeed");
        let db_path = format!("sqlite://{}/test.db", temp_dir.path().display());
        let db = RightsDatabase::new(&db_path)
            .await
            .expect("rights test operation should succeed");

        // Create asset first to satisfy foreign key constraint
        let asset = crate::rights::Asset::new("Test Asset", crate::rights::AssetType::Video);
        let asset_id = asset.id.clone();
        asset
            .save(&db)
            .await
            .expect("rights test operation should succeed");

        let tracker = UsageTracker::new(&db);
        let log = tracker
            .track_usage(&asset_id, UsageType::Commercial, None, Some("US"))
            .await
            .expect("rights test operation should succeed");

        assert_eq!(log.asset_id, asset_id);
        assert_eq!(log.territory, Some("US".to_string()));
    }
}
