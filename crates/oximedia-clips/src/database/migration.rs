//! Database schema migration utilities.

use crate::error::ClipResult;
use sqlx::{Row, SqlitePool};

/// Migrates the database to the latest schema version.
///
/// # Errors
///
/// Returns an error if the migration fails.
pub async fn migrate_database(pool: &SqlitePool) -> ClipResult<()> {
    // Create version table
    sqlx::query(
        r"
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL
        )
        ",
    )
    .execute(pool)
    .await?;

    // Get current version
    let current_version = get_current_version(pool).await?;

    // Apply migrations
    if current_version < 1 {
        apply_migration_v1(pool).await?;
    }

    if current_version < 2 {
        apply_migration_v2(pool).await?;
    }

    Ok(())
}

async fn get_current_version(pool: &SqlitePool) -> ClipResult<i32> {
    let row = sqlx::query("SELECT MAX(version) as version FROM schema_version")
        .fetch_optional(pool)
        .await?;

    if let Some(row) = row {
        Ok(row.try_get("version").unwrap_or(0))
    } else {
        Ok(0)
    }
}

async fn apply_migration_v1(pool: &SqlitePool) -> ClipResult<()> {
    // Create clips table (already done in storage.rs migrate())

    // Record migration
    sqlx::query("INSERT INTO schema_version (version, applied_at) VALUES (1, datetime('now'))")
        .execute(pool)
        .await?;

    Ok(())
}

async fn apply_migration_v2(pool: &SqlitePool) -> ClipResult<()> {
    // Future migration placeholder

    // Create bins table
    sqlx::query(
        r"
        CREATE TABLE IF NOT EXISTS bins (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            color TEXT,
            created_at TEXT NOT NULL,
            modified_at TEXT NOT NULL
        )
        ",
    )
    .execute(pool)
    .await?;

    // Create clip_bins junction table
    sqlx::query(
        r"
        CREATE TABLE IF NOT EXISTS clip_bins (
            clip_id TEXT NOT NULL,
            bin_id TEXT NOT NULL,
            PRIMARY KEY (clip_id, bin_id),
            FOREIGN KEY (clip_id) REFERENCES clips(id) ON DELETE CASCADE,
            FOREIGN KEY (bin_id) REFERENCES bins(id) ON DELETE CASCADE
        )
        ",
    )
    .execute(pool)
    .await?;

    // Record migration
    sqlx::query("INSERT INTO schema_version (version, applied_at) VALUES (2, datetime('now'))")
        .execute(pool)
        .await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use sqlx::sqlite::SqliteConnectOptions;
    use std::str::FromStr;

    #[tokio::test]
    async fn test_migration() {
        let options = SqliteConnectOptions::from_str(":memory:")
            .expect("operation should succeed")
            .create_if_missing(true);
        let pool = SqlitePool::connect_with(options)
            .await
            .expect("connect_with should succeed");

        migrate_database(&pool)
            .await
            .expect("operation should succeed");

        let version = get_current_version(&pool)
            .await
            .expect("get_current_version should succeed");
        assert!(version >= 1);
    }
}
