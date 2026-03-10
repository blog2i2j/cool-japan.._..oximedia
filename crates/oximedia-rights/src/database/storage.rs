//! Rights database storage implementation

use crate::Result;
use sqlx::sqlite::{SqliteConnectOptions, SqlitePool, SqlitePoolOptions};
use std::str::FromStr;

/// Rights database using SQLite
pub struct RightsDatabase {
    pool: SqlitePool,
}

impl RightsDatabase {
    /// Create a new rights database
    pub async fn new(path: &str) -> Result<Self> {
        let options = SqliteConnectOptions::from_str(path)?.create_if_missing(true);

        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect_with(options)
            .await?;

        let db = Self { pool };
        db.initialize_schema().await?;
        Ok(db)
    }

    /// Get a reference to the connection pool
    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }

    /// Initialize the database schema
    async fn initialize_schema(&self) -> Result<()> {
        // Rights owners table
        sqlx::query(
            r"
            CREATE TABLE IF NOT EXISTS rights_owners (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                contact_info TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            ",
        )
        .execute(&self.pool)
        .await?;

        // Assets table
        sqlx::query(
            r"
            CREATE TABLE IF NOT EXISTS assets (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                asset_type TEXT NOT NULL,
                description TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            ",
        )
        .execute(&self.pool)
        .await?;

        // Rights grants table
        sqlx::query(
            r"
            CREATE TABLE IF NOT EXISTS rights_grants (
                id TEXT PRIMARY KEY,
                asset_id TEXT NOT NULL,
                owner_id TEXT NOT NULL,
                license_type TEXT NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT,
                is_exclusive INTEGER NOT NULL DEFAULT 0,
                territory_json TEXT,
                usage_restrictions_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (asset_id) REFERENCES assets(id),
                FOREIGN KEY (owner_id) REFERENCES rights_owners(id)
            )
            ",
        )
        .execute(&self.pool)
        .await?;

        // License agreements table
        sqlx::query(
            r"
            CREATE TABLE IF NOT EXISTS license_agreements (
                id TEXT PRIMARY KEY,
                grant_id TEXT NOT NULL,
                agreement_number TEXT NOT NULL,
                terms_json TEXT NOT NULL,
                status TEXT NOT NULL,
                signed_date TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (grant_id) REFERENCES rights_grants(id)
            )
            ",
        )
        .execute(&self.pool)
        .await?;

        // Usage logs table
        sqlx::query(
            r"
            CREATE TABLE IF NOT EXISTS usage_logs (
                id TEXT PRIMARY KEY,
                asset_id TEXT NOT NULL,
                grant_id TEXT,
                usage_type TEXT NOT NULL,
                usage_date TEXT NOT NULL,
                territory TEXT,
                platform TEXT,
                metadata_json TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (asset_id) REFERENCES assets(id),
                FOREIGN KEY (grant_id) REFERENCES rights_grants(id)
            )
            ",
        )
        .execute(&self.pool)
        .await?;

        // Clearances table
        sqlx::query(
            r"
            CREATE TABLE IF NOT EXISTS clearances (
                id TEXT PRIMARY KEY,
                asset_id TEXT NOT NULL,
                clearance_type TEXT NOT NULL,
                status TEXT NOT NULL,
                requester TEXT,
                approver TEXT,
                requested_date TEXT NOT NULL,
                approved_date TEXT,
                expiry_date TEXT,
                notes TEXT,
                metadata_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (asset_id) REFERENCES assets(id)
            )
            ",
        )
        .execute(&self.pool)
        .await?;

        // Royalty payments table
        sqlx::query(
            r"
            CREATE TABLE IF NOT EXISTS royalty_payments (
                id TEXT PRIMARY KEY,
                grant_id TEXT NOT NULL,
                owner_id TEXT NOT NULL,
                amount REAL NOT NULL,
                currency TEXT NOT NULL,
                payment_period_start TEXT NOT NULL,
                payment_period_end TEXT NOT NULL,
                status TEXT NOT NULL,
                payment_date TEXT,
                calculation_data_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (grant_id) REFERENCES rights_grants(id),
                FOREIGN KEY (owner_id) REFERENCES rights_owners(id)
            )
            ",
        )
        .execute(&self.pool)
        .await?;

        // Audit trail table
        sqlx::query(
            r"
            CREATE TABLE IF NOT EXISTS audit_trail (
                id TEXT PRIMARY KEY,
                entity_type TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                action TEXT NOT NULL,
                user_id TEXT,
                changes_json TEXT,
                timestamp TEXT NOT NULL,
                ip_address TEXT
            )
            ",
        )
        .execute(&self.pool)
        .await?;

        // Expiration alerts table
        sqlx::query(
            r"
            CREATE TABLE IF NOT EXISTS expiration_alerts (
                id TEXT PRIMARY KEY,
                grant_id TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                alert_date TEXT NOT NULL,
                notification_sent INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY (grant_id) REFERENCES rights_grants(id)
            )
            ",
        )
        .execute(&self.pool)
        .await?;

        // Watermark configurations table
        sqlx::query(
            r"
            CREATE TABLE IF NOT EXISTS watermark_configs (
                id TEXT PRIMARY KEY,
                asset_id TEXT NOT NULL,
                watermark_type TEXT NOT NULL,
                config_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (asset_id) REFERENCES assets(id)
            )
            ",
        )
        .execute(&self.pool)
        .await?;

        // DRM metadata table
        sqlx::query(
            r"
            CREATE TABLE IF NOT EXISTS drm_metadata (
                id TEXT PRIMARY KEY,
                asset_id TEXT NOT NULL,
                drm_type TEXT NOT NULL,
                encryption_key_id TEXT,
                content_id TEXT,
                license_url TEXT,
                metadata_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (asset_id) REFERENCES assets(id)
            )
            ",
        )
        .execute(&self.pool)
        .await?;

        // Create indices for better query performance
        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_rights_grants_asset ON rights_grants(asset_id)",
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_rights_grants_owner ON rights_grants(owner_id)",
        )
        .execute(&self.pool)
        .await?;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_usage_logs_asset ON usage_logs(asset_id)")
            .execute(&self.pool)
            .await?;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_clearances_asset ON clearances(asset_id)")
            .execute(&self.pool)
            .await?;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_audit_trail_entity ON audit_trail(entity_type, entity_id)")
            .execute(&self.pool)
            .await?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_database_creation() {
        let temp_dir = tempfile::tempdir().expect("rights test operation should succeed");
        let db_path = format!("sqlite://{}/test.db", temp_dir.path().display());
        let db = RightsDatabase::new(&db_path).await;
        assert!(db.is_ok());
    }

    #[tokio::test]
    async fn test_schema_initialization() {
        let temp_dir = tempfile::tempdir().expect("rights test operation should succeed");
        let db_path = format!("sqlite://{}/test.db", temp_dir.path().display());
        let db = RightsDatabase::new(&db_path)
            .await
            .expect("rights test operation should succeed");

        // Verify tables exist by querying sqlite_master
        let result = sqlx::query(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='rights_owners'",
        )
        .fetch_one(db.pool())
        .await;
        assert!(result.is_ok());
    }
}
