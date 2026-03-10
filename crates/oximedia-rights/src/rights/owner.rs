//! Rights owner management

use crate::{database::RightsDatabase, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::Row;
use uuid::Uuid;

/// Rights owner (person or organization)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RightsOwner {
    /// Unique identifier
    pub id: String,
    /// Owner name
    pub name: String,
    /// Contact information
    pub contact_info: Option<String>,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last update timestamp
    pub updated_at: DateTime<Utc>,
}

impl RightsOwner {
    /// Create a new rights owner
    pub fn new(name: impl Into<String>) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            name: name.into(),
            contact_info: None,
            created_at: now,
            updated_at: now,
        }
    }

    /// Set contact information
    pub fn with_contact(mut self, contact: impl Into<String>) -> Self {
        self.contact_info = Some(contact.into());
        self
    }

    /// Save owner to database
    pub async fn save(&self, db: &RightsDatabase) -> Result<()> {
        sqlx::query(
            r"
            INSERT INTO rights_owners (id, name, contact_info, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                contact_info = excluded.contact_info,
                updated_at = excluded.updated_at
            ",
        )
        .bind(&self.id)
        .bind(&self.name)
        .bind(&self.contact_info)
        .bind(self.created_at.to_rfc3339())
        .bind(self.updated_at.to_rfc3339())
        .execute(db.pool())
        .await?;

        Ok(())
    }

    /// Load owner from database by ID
    pub async fn load(db: &RightsDatabase, id: &str) -> Result<Option<Self>> {
        let row = sqlx::query(
            r"
            SELECT id, name, contact_info, created_at, updated_at
            FROM rights_owners WHERE id = ?
            ",
        )
        .bind(id)
        .fetch_optional(db.pool())
        .await?;

        row.map(|r| {
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
            Ok(RightsOwner {
                id: r.get("id"),
                name: r.get("name"),
                contact_info: r.get("contact_info"),
                created_at,
                updated_at,
            })
        })
        .transpose()
    }

    /// List all owners
    pub async fn list(db: &RightsDatabase) -> Result<Vec<Self>> {
        let rows = sqlx::query(
            r"
            SELECT id, name, contact_info, created_at, updated_at
            FROM rights_owners
            ORDER BY name ASC
            ",
        )
        .fetch_all(db.pool())
        .await?;

        rows.into_iter()
            .map(|r| {
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
                Ok(RightsOwner {
                    id: r.get("id"),
                    name: r.get("name"),
                    contact_info: r.get("contact_info"),
                    created_at,
                    updated_at,
                })
            })
            .collect()
    }

    /// Delete owner from database
    pub async fn delete(db: &RightsDatabase, id: &str) -> Result<()> {
        sqlx::query("DELETE FROM rights_owners WHERE id = ?")
            .bind(id)
            .execute(db.pool())
            .await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_owner_creation() {
        let owner = RightsOwner::new("John Doe").with_contact("john@example.com");

        assert_eq!(owner.name, "John Doe");
        assert_eq!(owner.contact_info, Some("john@example.com".to_string()));
    }

    #[tokio::test]
    async fn test_owner_save_and_load() {
        let temp_dir = tempfile::tempdir().expect("rights test operation should succeed");
        let db_path = format!("sqlite://{}/test.db", temp_dir.path().display());
        let db = RightsDatabase::new(&db_path)
            .await
            .expect("rights test operation should succeed");

        let owner = RightsOwner::new("Test Owner");
        let owner_id = owner.id.clone();

        owner
            .save(&db)
            .await
            .expect("rights test operation should succeed");

        let loaded = RightsOwner::load(&db, &owner_id)
            .await
            .expect("rights test operation should succeed");
        assert!(loaded.is_some());
        let loaded = loaded.expect("rights test operation should succeed");
        assert_eq!(loaded.name, "Test Owner");
    }
}
