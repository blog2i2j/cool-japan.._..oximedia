//! `SQLite` database storage for clips.

use crate::clip::{Clip, ClipId};
use crate::error::{ClipError, ClipResult};
use sqlx::sqlite::{SqliteConnectOptions, SqlitePool};
use sqlx::Row;
use std::str::FromStr;

/// `SQLite` database for clip storage.
#[derive(Debug, Clone)]
pub struct ClipDatabase {
    pool: SqlitePool,
}

impl ClipDatabase {
    /// Creates a new clip database.
    ///
    /// # Errors
    ///
    /// Returns an error if the database cannot be opened or migrated.
    pub async fn new(database_url: impl AsRef<str>) -> ClipResult<Self> {
        let options =
            SqliteConnectOptions::from_str(database_url.as_ref())?.create_if_missing(true);

        let pool = SqlitePool::connect_with(options).await?;

        let db = Self { pool };
        db.migrate().await?;

        Ok(db)
    }

    async fn migrate(&self) -> ClipResult<()> {
        sqlx::query(
            r"
            CREATE TABLE IF NOT EXISTS clips (
                id TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                duration INTEGER,
                frame_rate_num INTEGER,
                frame_rate_den INTEGER,
                in_point INTEGER,
                out_point INTEGER,
                rating INTEGER NOT NULL DEFAULT 0,
                is_favorite INTEGER NOT NULL DEFAULT 0,
                is_rejected INTEGER NOT NULL DEFAULT 0,
                keywords TEXT,
                created_at TEXT NOT NULL,
                modified_at TEXT NOT NULL,
                custom_metadata TEXT
            )
            ",
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Saves a clip to the database.
    ///
    /// # Errors
    ///
    /// Returns an error if the clip cannot be saved.
    #[allow(clippy::too_many_arguments)]
    pub async fn save_clip(&self, clip: &Clip) -> ClipResult<()> {
        let keywords_json = serde_json::to_string(&clip.keywords)
            .map_err(|e| ClipError::Serialization(e.to_string()))?;

        let (frame_rate_num, frame_rate_den) = clip
            .frame_rate
            .map_or((None, None), |fr| (Some(fr.num), Some(fr.den)));

        sqlx::query(
            r"
            INSERT INTO clips (
                id, file_path, name, description, duration,
                frame_rate_num, frame_rate_den, in_point, out_point,
                rating, is_favorite, is_rejected, keywords,
                created_at, modified_at, custom_metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                file_path = excluded.file_path,
                name = excluded.name,
                description = excluded.description,
                duration = excluded.duration,
                frame_rate_num = excluded.frame_rate_num,
                frame_rate_den = excluded.frame_rate_den,
                in_point = excluded.in_point,
                out_point = excluded.out_point,
                rating = excluded.rating,
                is_favorite = excluded.is_favorite,
                is_rejected = excluded.is_rejected,
                keywords = excluded.keywords,
                modified_at = excluded.modified_at,
                custom_metadata = excluded.custom_metadata
            ",
        )
        .bind(clip.id.to_string())
        .bind(clip.file_path.to_string_lossy().to_string())
        .bind(&clip.name)
        .bind(&clip.description)
        .bind(clip.duration)
        .bind(frame_rate_num)
        .bind(frame_rate_den)
        .bind(clip.in_point)
        .bind(clip.out_point)
        .bind(i64::from(clip.rating.to_value()))
        .bind(i64::from(clip.is_favorite))
        .bind(i64::from(clip.is_rejected))
        .bind(keywords_json)
        .bind(clip.created_at.to_rfc3339())
        .bind(clip.modified_at.to_rfc3339())
        .bind(&clip.custom_metadata)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Gets a clip by ID.
    ///
    /// # Errors
    ///
    /// Returns an error if the clip is not found or cannot be loaded.
    pub async fn get_clip(&self, clip_id: &ClipId) -> ClipResult<Clip> {
        let row = sqlx::query("SELECT * FROM clips WHERE id = ?")
            .bind(clip_id.to_string())
            .fetch_optional(&self.pool)
            .await?
            .ok_or_else(|| ClipError::ClipNotFound(clip_id.to_string()))?;

        Self::row_to_clip(&row)
    }

    /// Gets all clips.
    ///
    /// # Errors
    ///
    /// Returns an error if clips cannot be loaded.
    pub async fn get_all_clips(&self) -> ClipResult<Vec<Clip>> {
        let rows = sqlx::query("SELECT * FROM clips ORDER BY created_at DESC")
            .fetch_all(&self.pool)
            .await?;

        rows.iter().map(Self::row_to_clip).collect()
    }

    /// Deletes a clip by ID.
    ///
    /// # Errors
    ///
    /// Returns an error if the clip cannot be deleted.
    pub async fn delete_clip(&self, clip_id: &ClipId) -> ClipResult<()> {
        sqlx::query("DELETE FROM clips WHERE id = ?")
            .bind(clip_id.to_string())
            .execute(&self.pool)
            .await?;

        Ok(())
    }

    /// Searches clips by name or keywords.
    ///
    /// # Errors
    ///
    /// Returns an error if the search fails.
    pub async fn search_clips(&self, query: &str) -> ClipResult<Vec<Clip>> {
        let search_pattern = format!("%{query}%");

        let rows = sqlx::query(
            "SELECT * FROM clips WHERE name LIKE ? OR keywords LIKE ? ORDER BY created_at DESC",
        )
        .bind(&search_pattern)
        .bind(&search_pattern)
        .fetch_all(&self.pool)
        .await?;

        rows.iter().map(Self::row_to_clip).collect()
    }

    #[allow(dead_code)]
    fn row_to_clip(row: &sqlx::sqlite::SqliteRow) -> ClipResult<Clip> {
        use chrono::DateTime;
        use oximedia_core::types::Rational;
        use std::path::PathBuf;

        let id_str: String = row.try_get("id")?;
        let id = id_str
            .parse()
            .map_err(|e: uuid::Error| ClipError::Serialization(e.to_string()))?;

        let keywords_json: String = row.try_get("keywords")?;
        let keywords: Vec<String> = serde_json::from_str(&keywords_json)
            .map_err(|e| ClipError::Serialization(e.to_string()))?;

        let created_at_str: String = row.try_get("created_at")?;
        let created_at = DateTime::parse_from_rfc3339(&created_at_str)
            .map_err(|e| ClipError::Serialization(e.to_string()))?
            .with_timezone(&chrono::Utc);

        let modified_at_str: String = row.try_get("modified_at")?;
        let modified_at = DateTime::parse_from_rfc3339(&modified_at_str)
            .map_err(|e| ClipError::Serialization(e.to_string()))?
            .with_timezone(&chrono::Utc);

        let rating_val: i64 = row.try_get("rating")?;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let rating = crate::logging::Rating::from_value(rating_val as u8)
            .unwrap_or(crate::logging::Rating::Unrated);

        let frame_rate = match (
            row.try_get::<Option<i64>, _>("frame_rate_num")?,
            row.try_get::<Option<i64>, _>("frame_rate_den")?,
        ) {
            (Some(num), Some(den)) => Some(Rational::new(num, den)),
            _ => None,
        };

        Ok(Clip {
            id,
            file_path: PathBuf::from(row.try_get::<String, _>("file_path")?),
            name: row.try_get("name")?,
            description: row.try_get("description")?,
            duration: row.try_get("duration")?,
            frame_rate,
            in_point: row.try_get("in_point")?,
            out_point: row.try_get("out_point")?,
            rating,
            is_favorite: row.try_get::<i64, _>("is_favorite")? != 0,
            is_rejected: row.try_get::<i64, _>("is_rejected")? != 0,
            keywords,
            markers: Vec::new(), // Would need separate table
            created_at,
            modified_at,
            custom_metadata: row.try_get("custom_metadata")?,
        })
    }

    /// Returns the number of clips in the database.
    ///
    /// # Errors
    ///
    /// Returns an error if the count query fails.
    pub async fn count_clips(&self) -> ClipResult<i64> {
        let row = sqlx::query("SELECT COUNT(*) as count FROM clips")
            .fetch_one(&self.pool)
            .await?;

        Ok(row.try_get("count")?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[tokio::test]
    async fn test_database_creation() {
        let db = ClipDatabase::new(":memory:")
            .await
            .expect("new should succeed");
        let count = db.count_clips().await.expect("count_clips should succeed");
        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn test_save_and_get_clip() {
        let db = ClipDatabase::new(":memory:")
            .await
            .expect("new should succeed");

        let mut clip = Clip::new(PathBuf::from("/test.mov"));
        clip.set_name("Test Clip");
        clip.add_keyword("test");

        db.save_clip(&clip).await.expect("operation should succeed");

        let loaded = db
            .get_clip(&clip.id)
            .await
            .expect("get_clip should succeed");
        assert_eq!(loaded.name, "Test Clip");
        assert_eq!(loaded.keywords.len(), 1);
    }

    #[tokio::test]
    async fn test_search_clips() {
        let db = ClipDatabase::new(":memory:")
            .await
            .expect("new should succeed");

        let mut clip1 = Clip::new(PathBuf::from("/test1.mov"));
        clip1.set_name("Interview");

        let mut clip2 = Clip::new(PathBuf::from("/test2.mov"));
        clip2.set_name("Action Scene");

        db.save_clip(&clip1)
            .await
            .expect("operation should succeed");
        db.save_clip(&clip2)
            .await
            .expect("operation should succeed");

        let results = db
            .search_clips("interview")
            .await
            .expect("search_clips should succeed");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "Interview");
    }
}
