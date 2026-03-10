//! Usage reporting

use crate::{database::RightsDatabase, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::Row;
use std::collections::HashMap;

/// Usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageStats {
    /// Total usage count
    pub total_uses: u32,
    /// Usage by type
    pub by_type: HashMap<String, u32>,
    /// Usage by territory
    pub by_territory: HashMap<String, u32>,
    /// Usage by platform
    pub by_platform: HashMap<String, u32>,
}

/// Usage report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageReport {
    /// Asset ID
    pub asset_id: String,
    /// Start date
    pub start_date: DateTime<Utc>,
    /// End date
    pub end_date: DateTime<Utc>,
    /// Usage statistics
    pub stats: UsageStats,
    /// Generated at
    pub generated_at: DateTime<Utc>,
}

impl UsageReport {
    /// Generate usage report for an asset
    pub async fn generate(
        db: &RightsDatabase,
        asset_id: &str,
        start_date: DateTime<Utc>,
        end_date: DateTime<Utc>,
    ) -> Result<Self> {
        let rows = sqlx::query(
            r"
            SELECT usage_type, territory, platform
            FROM usage_logs
            WHERE asset_id = ? AND usage_date >= ? AND usage_date <= ?
            ",
        )
        .bind(asset_id)
        .bind(start_date.to_rfc3339())
        .bind(end_date.to_rfc3339())
        .fetch_all(db.pool())
        .await?;

        let mut by_type: HashMap<String, u32> = HashMap::new();
        let mut by_territory: HashMap<String, u32> = HashMap::new();
        let mut by_platform: HashMap<String, u32> = HashMap::new();

        for row in &rows {
            let usage_type: String = row.get("usage_type");
            *by_type.entry(usage_type).or_insert(0) += 1;

            if let Some(territory) = row.get::<Option<String>, _>("territory") {
                *by_territory.entry(territory).or_insert(0) += 1;
            }

            if let Some(platform) = row.get::<Option<String>, _>("platform") {
                *by_platform.entry(platform).or_insert(0) += 1;
            }
        }

        Ok(UsageReport {
            asset_id: asset_id.to_string(),
            start_date,
            end_date,
            stats: UsageStats {
                total_uses: rows.len() as u32,
                by_type,
                by_territory,
                by_platform,
            },
            generated_at: Utc::now(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rights::UsageType;
    use crate::usage::UsageLog;

    #[tokio::test]
    async fn test_usage_report() {
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

        let now = Utc::now();
        let log1 = UsageLog::new(&asset_id, UsageType::Commercial, now).with_territory("US");
        let log2 = UsageLog::new(&asset_id, UsageType::Web, now).with_territory("GB");

        log1.save(&db)
            .await
            .expect("rights test operation should succeed");
        log2.save(&db)
            .await
            .expect("rights test operation should succeed");

        let report = UsageReport::generate(
            &db,
            &asset_id,
            now - chrono::Duration::hours(1),
            now + chrono::Duration::hours(1),
        )
        .await
        .expect("rights test operation should succeed");

        assert_eq!(report.stats.total_uses, 2);
    }
}
