//! Expiration alerts

use crate::{database::RightsDatabase, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::Row;
use uuid::Uuid;

/// Type of expiration alert
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AlertType {
    /// Warning before expiration
    Warning,
    /// Critical - expiring very soon
    Critical,
    /// Expired
    Expired,
}

impl AlertType {
    /// Convert to string representation
    pub fn as_str(&self) -> &str {
        match self {
            AlertType::Warning => "warning",
            AlertType::Critical => "critical",
            AlertType::Expired => "expired",
        }
    }

    /// Parse from string
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Self {
        match s {
            "warning" => AlertType::Warning,
            "critical" => AlertType::Critical,
            "expired" => AlertType::Expired,
            _ => AlertType::Warning,
        }
    }
}

/// Expiration alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpirationAlert {
    /// Unique identifier
    pub id: String,
    /// Associated rights grant ID
    pub grant_id: String,
    /// Alert type
    pub alert_type: AlertType,
    /// Alert date (when alert should be shown/sent)
    pub alert_date: DateTime<Utc>,
    /// Whether notification has been sent
    pub notification_sent: bool,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
}

impl ExpirationAlert {
    /// Create a new expiration alert
    pub fn new(
        grant_id: impl Into<String>,
        alert_type: AlertType,
        alert_date: DateTime<Utc>,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            grant_id: grant_id.into(),
            alert_type,
            alert_date,
            notification_sent: false,
            created_at: Utc::now(),
        }
    }

    /// Mark notification as sent
    pub fn mark_sent(&mut self) {
        self.notification_sent = true;
    }

    /// Save alert to database
    pub async fn save(&self, db: &RightsDatabase) -> Result<()> {
        sqlx::query(
            r"
            INSERT INTO expiration_alerts
            (id, grant_id, alert_type, alert_date, notification_sent, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                grant_id = excluded.grant_id,
                alert_type = excluded.alert_type,
                alert_date = excluded.alert_date,
                notification_sent = excluded.notification_sent
            ",
        )
        .bind(&self.id)
        .bind(&self.grant_id)
        .bind(self.alert_type.as_str())
        .bind(self.alert_date.to_rfc3339())
        .bind(self.notification_sent as i32)
        .bind(self.created_at.to_rfc3339())
        .execute(db.pool())
        .await?;

        Ok(())
    }

    /// Load alert from database by ID
    pub async fn load(db: &RightsDatabase, id: &str) -> Result<Option<Self>> {
        let row = sqlx::query(
            r"
            SELECT id, grant_id, alert_type, alert_date, notification_sent, created_at
            FROM expiration_alerts WHERE id = ?
            ",
        )
        .bind(id)
        .fetch_optional(db.pool())
        .await?;

        let alert = match row {
            None => return Ok(None),
            Some(r) => {
                let alert_date = DateTime::parse_from_rfc3339(r.get("alert_date"))
                    .map(|dt| dt.with_timezone(&Utc))
                    .map_err(|e| crate::RightsError::Serialization(e.to_string()))?;
                let created_at = DateTime::parse_from_rfc3339(r.get("created_at"))
                    .map(|dt| dt.with_timezone(&Utc))
                    .map_err(|e| crate::RightsError::Serialization(e.to_string()))?;
                ExpirationAlert {
                    id: r.get("id"),
                    grant_id: r.get("grant_id"),
                    alert_type: AlertType::from_str(r.get("alert_type")),
                    alert_date,
                    notification_sent: r.get::<i32, _>("notification_sent") != 0,
                    created_at,
                }
            }
        };

        Ok(Some(alert))
    }

    /// Get pending alerts (not yet sent and alert date has passed)
    pub async fn get_pending_alerts(db: &RightsDatabase) -> Result<Vec<Self>> {
        let now = Utc::now();

        let rows = sqlx::query(
            r"
            SELECT id, grant_id, alert_type, alert_date, notification_sent, created_at
            FROM expiration_alerts
            WHERE notification_sent = 0 AND alert_date <= ?
            ORDER BY alert_date ASC
            ",
        )
        .bind(now.to_rfc3339())
        .fetch_all(db.pool())
        .await?;

        rows.into_iter()
            .map(|r| {
                let alert_date = DateTime::parse_from_rfc3339(r.get("alert_date"))
                    .map(|dt| dt.with_timezone(&Utc))
                    .map_err(|e| crate::RightsError::Serialization(e.to_string()))?;
                let created_at = DateTime::parse_from_rfc3339(r.get("created_at"))
                    .map(|dt| dt.with_timezone(&Utc))
                    .map_err(|e| crate::RightsError::Serialization(e.to_string()))?;
                Ok(ExpirationAlert {
                    id: r.get("id"),
                    grant_id: r.get("grant_id"),
                    alert_type: AlertType::from_str(r.get("alert_type")),
                    alert_date,
                    notification_sent: r.get::<i32, _>("notification_sent") != 0,
                    created_at,
                })
            })
            .collect()
    }

    /// Get all alerts for a grant
    pub async fn list_for_grant(db: &RightsDatabase, grant_id: &str) -> Result<Vec<Self>> {
        let rows = sqlx::query(
            r"
            SELECT id, grant_id, alert_type, alert_date, notification_sent, created_at
            FROM expiration_alerts WHERE grant_id = ?
            ORDER BY alert_date DESC
            ",
        )
        .bind(grant_id)
        .fetch_all(db.pool())
        .await?;

        rows.into_iter()
            .map(|r| {
                let alert_date = DateTime::parse_from_rfc3339(r.get("alert_date"))
                    .map(|dt| dt.with_timezone(&Utc))
                    .map_err(|e| crate::RightsError::Serialization(e.to_string()))?;
                let created_at = DateTime::parse_from_rfc3339(r.get("created_at"))
                    .map(|dt| dt.with_timezone(&Utc))
                    .map_err(|e| crate::RightsError::Serialization(e.to_string()))?;
                Ok(ExpirationAlert {
                    id: r.get("id"),
                    grant_id: r.get("grant_id"),
                    alert_type: AlertType::from_str(r.get("alert_type")),
                    alert_date,
                    notification_sent: r.get::<i32, _>("notification_sent") != 0,
                    created_at,
                })
            })
            .collect()
    }

    /// Delete alert from database
    pub async fn delete(db: &RightsDatabase, id: &str) -> Result<()> {
        sqlx::query("DELETE FROM expiration_alerts WHERE id = ?")
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
    fn test_alert_creation() {
        let now = Utc::now();
        let alert = ExpirationAlert::new("grant123", AlertType::Warning, now);

        assert_eq!(alert.grant_id, "grant123");
        assert_eq!(alert.alert_type, AlertType::Warning);
        assert!(!alert.notification_sent);
    }

    #[test]
    fn test_mark_sent() {
        let now = Utc::now();
        let mut alert = ExpirationAlert::new("grant123", AlertType::Warning, now);

        alert.mark_sent();
        assert!(alert.notification_sent);
    }

    #[tokio::test]
    async fn test_alert_save_and_load() {
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
            Utc::now(),
            Some(Utc::now() + chrono::Duration::days(30)),
            true,
        );
        let grant_id = grant.id.clone();
        grant
            .save(&db)
            .await
            .expect("rights test operation should succeed");

        let alert = ExpirationAlert::new(&grant_id, AlertType::Critical, Utc::now());
        let alert_id = alert.id.clone();

        alert
            .save(&db)
            .await
            .expect("rights test operation should succeed");

        let loaded = ExpirationAlert::load(&db, &alert_id)
            .await
            .expect("rights test operation should succeed");
        assert!(loaded.is_some());
        let loaded = loaded.expect("rights test operation should succeed");
        assert_eq!(loaded.alert_type, AlertType::Critical);
    }
}
