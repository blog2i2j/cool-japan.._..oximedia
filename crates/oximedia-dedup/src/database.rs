//! SQLite-based deduplication database and indexing.
//!
//! This module provides:
//! - Fast hash and fingerprint storage
//! - Efficient duplicate lookup
//! - Batch operations for performance
//! - Index optimization and statistics

use crate::DedupResult;
use sqlx::sqlite::{SqliteConnectOptions, SqlitePool, SqlitePoolOptions};
use sqlx::Row;
use std::collections::HashMap;
use std::path::Path;
use std::str::FromStr;

/// SQLite database for deduplication.
pub struct DedupDatabase {
    pool: SqlitePool,
}

impl DedupDatabase {
    /// Open or create a database.
    ///
    /// # Errors
    ///
    /// Returns an error if the database cannot be opened or initialized.
    pub async fn open(path: impl AsRef<Path>) -> DedupResult<Self> {
        let path = path.as_ref();

        // Create parent directory if needed
        if let Some(parent) = path.parent() {
            if !parent.exists() {
                std::fs::create_dir_all(parent)?;
            }
        }

        let options = SqliteConnectOptions::from_str(&format!("sqlite:{}", path.display()))?
            .create_if_missing(true);

        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect_with(options)
            .await?;

        let db = Self { pool };
        db.initialize().await?;

        Ok(db)
    }

    /// Open in-memory database (for testing).
    ///
    /// # Errors
    ///
    /// Returns an error if the database cannot be created.
    pub async fn open_memory() -> DedupResult<Self> {
        let options = SqliteConnectOptions::from_str("sqlite::memory:")?.create_if_missing(true);

        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect_with(options)
            .await?;

        let db = Self { pool };
        db.initialize().await?;

        Ok(db)
    }

    /// Initialize database schema.
    async fn initialize(&self) -> DedupResult<()> {
        // Files table
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL UNIQUE,
                size INTEGER NOT NULL,
                hash TEXT NOT NULL,
                created_at INTEGER DEFAULT (strftime('%s', 'now')),
                updated_at INTEGER DEFAULT (strftime('%s', 'now'))
            )
            "#,
        )
        .execute(&self.pool)
        .await?;

        // Create index on hash for fast duplicate lookup
        sqlx::query(
            r#"
            CREATE INDEX IF NOT EXISTS idx_files_hash ON files(hash)
            "#,
        )
        .execute(&self.pool)
        .await?;

        // Fingerprints table (for perceptual hashes, audio fingerprints, etc.)
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS fingerprints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id INTEGER NOT NULL,
                type TEXT NOT NULL,
                data TEXT NOT NULL,
                created_at INTEGER DEFAULT (strftime('%s', 'now')),
                FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE
            )
            "#,
        )
        .execute(&self.pool)
        .await?;

        // Create index on fingerprint type for fast lookup
        sqlx::query(
            r#"
            CREATE INDEX IF NOT EXISTS idx_fingerprints_type ON fingerprints(type)
            "#,
        )
        .execute(&self.pool)
        .await?;

        // Create index on fingerprint data for fast lookup
        sqlx::query(
            r#"
            CREATE INDEX IF NOT EXISTS idx_fingerprints_data ON fingerprints(data)
            "#,
        )
        .execute(&self.pool)
        .await?;

        // Metadata table
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id INTEGER NOT NULL,
                duration REAL,
                width INTEGER,
                height INTEGER,
                bitrate INTEGER,
                framerate REAL,
                sample_rate INTEGER,
                channels INTEGER,
                video_codec TEXT,
                audio_codec TEXT,
                container TEXT,
                FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE
            )
            "#,
        )
        .execute(&self.pool)
        .await?;

        // Chunks table (for content-based deduplication)
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id INTEGER NOT NULL,
                offset INTEGER NOT NULL,
                size INTEGER NOT NULL,
                hash TEXT NOT NULL,
                FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE
            )
            "#,
        )
        .execute(&self.pool)
        .await?;

        // Create index on chunk hash
        sqlx::query(
            r#"
            CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(hash)
            "#,
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Insert a file into the database.
    ///
    /// # Errors
    ///
    /// Returns an error if the insertion fails.
    pub async fn insert_file(&self, path: impl AsRef<Path>, hash: &str) -> DedupResult<i64> {
        let path = path.as_ref().to_string_lossy().to_string();
        let size = std::fs::metadata(path.as_str())?.len() as i64;

        let result = sqlx::query(
            r#"
            INSERT INTO files (path, size, hash)
            VALUES (?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                size = excluded.size,
                hash = excluded.hash,
                updated_at = strftime('%s', 'now')
            RETURNING id
            "#,
        )
        .bind(&path)
        .bind(size)
        .bind(hash)
        .fetch_one(&self.pool)
        .await?;

        Ok(result.get(0))
    }

    /// Get file ID by path.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub async fn get_file_id(&self, path: impl AsRef<Path>) -> DedupResult<Option<i64>> {
        let path = path.as_ref().to_string_lossy().to_string();

        let result = sqlx::query(
            r#"
            SELECT id FROM files WHERE path = ?
            "#,
        )
        .bind(&path)
        .fetch_optional(&self.pool)
        .await?;

        Ok(result.map(|row| row.get(0)))
    }

    /// Insert a fingerprint.
    ///
    /// # Errors
    ///
    /// Returns an error if the insertion fails.
    pub async fn insert_fingerprint(
        &self,
        file_id: i64,
        fingerprint_type: &str,
        data: &str,
    ) -> DedupResult<i64> {
        let result = sqlx::query(
            r#"
            INSERT INTO fingerprints (file_id, type, data)
            VALUES (?, ?, ?)
            RETURNING id
            "#,
        )
        .bind(file_id)
        .bind(fingerprint_type)
        .bind(data)
        .fetch_one(&self.pool)
        .await?;

        Ok(result.get(0))
    }

    /// Insert metadata.
    ///
    /// # Errors
    ///
    /// Returns an error if the insertion fails.
    #[allow(clippy::too_many_arguments)]
    pub async fn insert_metadata(
        &self,
        file_id: i64,
        duration: Option<f64>,
        width: Option<i32>,
        height: Option<i32>,
        video_codec: Option<&str>,
        audio_codec: Option<&str>,
        container: Option<&str>,
    ) -> DedupResult<i64> {
        let result = sqlx::query(
            r#"
            INSERT INTO metadata (file_id, duration, width, height, video_codec, audio_codec, container)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            RETURNING id
            "#,
        )
        .bind(file_id)
        .bind(duration)
        .bind(width)
        .bind(height)
        .bind(video_codec)
        .bind(audio_codec)
        .bind(container)
        .fetch_one(&self.pool)
        .await?;

        Ok(result.get(0))
    }

    /// Insert a chunk.
    ///
    /// # Errors
    ///
    /// Returns an error if the insertion fails.
    pub async fn insert_chunk(
        &self,
        file_id: i64,
        offset: i64,
        size: i64,
        hash: &str,
    ) -> DedupResult<i64> {
        let result = sqlx::query(
            r#"
            INSERT INTO chunks (file_id, offset, size, hash)
            VALUES (?, ?, ?, ?)
            RETURNING id
            "#,
        )
        .bind(file_id)
        .bind(offset)
        .bind(size)
        .bind(hash)
        .fetch_one(&self.pool)
        .await?;

        Ok(result.get(0))
    }

    /// Find files with duplicate hashes.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub async fn find_duplicate_hashes(&self) -> DedupResult<HashMap<String, Vec<String>>> {
        let rows = sqlx::query(
            r#"
            SELECT hash, path
            FROM files
            WHERE hash IN (
                SELECT hash
                FROM files
                GROUP BY hash
                HAVING COUNT(*) > 1
            )
            ORDER BY hash, path
            "#,
        )
        .fetch_all(&self.pool)
        .await?;

        let mut duplicates: HashMap<String, Vec<String>> = HashMap::new();

        for row in rows {
            let hash: String = row.get(0);
            let path: String = row.get(1);

            duplicates.entry(hash).or_insert_with(Vec::new).push(path);
        }

        Ok(duplicates)
    }

    /// Find files with similar fingerprints.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub async fn find_similar_fingerprints(
        &self,
        fingerprint_type: &str,
    ) -> DedupResult<HashMap<String, Vec<String>>> {
        let rows = sqlx::query(
            r#"
            SELECT f.data, fi.path
            FROM fingerprints f
            JOIN files fi ON f.file_id = fi.id
            WHERE f.type = ?
            ORDER BY f.data
            "#,
        )
        .bind(fingerprint_type)
        .fetch_all(&self.pool)
        .await?;

        let mut groups: HashMap<String, Vec<String>> = HashMap::new();

        for row in rows {
            let data: String = row.get(0);
            let path: String = row.get(1);

            groups.entry(data).or_insert_with(Vec::new).push(path);
        }

        // Filter to only groups with multiple files
        Ok(groups
            .into_iter()
            .filter(|(_, paths)| paths.len() > 1)
            .collect())
    }

    /// Find duplicate chunks.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub async fn find_duplicate_chunks(&self) -> DedupResult<HashMap<String, Vec<String>>> {
        let rows = sqlx::query(
            r#"
            SELECT c.hash, f.path
            FROM chunks c
            JOIN files f ON c.file_id = f.id
            WHERE c.hash IN (
                SELECT hash
                FROM chunks
                GROUP BY hash
                HAVING COUNT(*) > 1
            )
            ORDER BY c.hash
            "#,
        )
        .fetch_all(&self.pool)
        .await?;

        let mut duplicates: HashMap<String, Vec<String>> = HashMap::new();

        for row in rows {
            let hash: String = row.get(0);
            let path: String = row.get(1);

            let paths = duplicates.entry(hash).or_insert_with(Vec::new);
            if !paths.contains(&path) {
                paths.push(path);
            }
        }

        Ok(duplicates)
    }

    /// Get all files.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub async fn get_all_files(&self) -> DedupResult<Vec<(String, String)>> {
        let rows = sqlx::query(
            r#"
            SELECT path, hash FROM files ORDER BY path
            "#,
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(rows
            .iter()
            .map(|row| {
                let path: String = row.get(0);
                let hash: String = row.get(1);
                (path, hash)
            })
            .collect())
    }

    /// Count total files.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub async fn count_files(&self) -> DedupResult<usize> {
        let row = sqlx::query(
            r#"
            SELECT COUNT(*) FROM files
            "#,
        )
        .fetch_one(&self.pool)
        .await?;

        let count: i64 = row.get(0);
        Ok(count as usize)
    }

    /// Count unique hashes.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub async fn count_unique_hashes(&self) -> DedupResult<usize> {
        let row = sqlx::query(
            r#"
            SELECT COUNT(DISTINCT hash) FROM files
            "#,
        )
        .fetch_one(&self.pool)
        .await?;

        let count: i64 = row.get(0);
        Ok(count as usize)
    }

    /// Delete file by path.
    ///
    /// # Errors
    ///
    /// Returns an error if the deletion fails.
    pub async fn delete_file(&self, path: impl AsRef<Path>) -> DedupResult<()> {
        let path = path.as_ref().to_string_lossy().to_string();

        sqlx::query(
            r#"
            DELETE FROM files WHERE path = ?
            "#,
        )
        .bind(&path)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Delete files by hash.
    ///
    /// # Errors
    ///
    /// Returns an error if the deletion fails.
    pub async fn delete_by_hash(&self, hash: &str) -> DedupResult<usize> {
        let result = sqlx::query(
            r#"
            DELETE FROM files WHERE hash = ?
            "#,
        )
        .bind(hash)
        .execute(&self.pool)
        .await?;

        Ok(result.rows_affected() as usize)
    }

    /// Optimize database (vacuum and analyze).
    ///
    /// # Errors
    ///
    /// Returns an error if optimization fails.
    pub async fn optimize(&self) -> DedupResult<()> {
        sqlx::query("VACUUM").execute(&self.pool).await?;
        sqlx::query("ANALYZE").execute(&self.pool).await?;
        Ok(())
    }

    /// Get database statistics.
    ///
    /// # Errors
    ///
    /// Returns an error if queries fail.
    pub async fn get_stats(&self) -> DedupResult<DatabaseStats> {
        let total_files = self.count_files().await?;
        let unique_hashes = self.count_unique_hashes().await?;

        let row = sqlx::query(
            r#"
            SELECT COUNT(*) FROM fingerprints
            "#,
        )
        .fetch_one(&self.pool)
        .await?;
        let total_fingerprints: i64 = row.get(0);

        let row = sqlx::query(
            r#"
            SELECT COUNT(*) FROM chunks
            "#,
        )
        .fetch_one(&self.pool)
        .await?;
        let total_chunks: i64 = row.get(0);

        let row = sqlx::query(
            r#"
            SELECT SUM(size) FROM files
            "#,
        )
        .fetch_one(&self.pool)
        .await?;
        let total_size: Option<i64> = row.get(0);

        Ok(DatabaseStats {
            total_files,
            unique_hashes,
            total_fingerprints: total_fingerprints as usize,
            total_chunks: total_chunks as usize,
            total_size: total_size.unwrap_or(0) as u64,
        })
    }

    /// Begin a transaction.
    ///
    /// # Errors
    ///
    /// Returns an error if transaction cannot be started.
    pub async fn begin_transaction(&self) -> DedupResult<sqlx::Transaction<'_, sqlx::Sqlite>> {
        Ok(self.pool.begin().await?)
    }

    /// Close the database.
    ///
    /// # Errors
    ///
    /// Returns an error if closing fails.
    pub async fn close(self) -> DedupResult<()> {
        self.pool.close().await;
        Ok(())
    }
}

/// Database statistics.
#[derive(Debug, Clone)]
pub struct DatabaseStats {
    /// Total number of indexed files
    pub total_files: usize,

    /// Number of unique hashes
    pub unique_hashes: usize,

    /// Total number of fingerprints
    pub total_fingerprints: usize,

    /// Total number of chunks
    pub total_chunks: usize,

    /// Total size of all files in bytes
    pub total_size: u64,
}

impl DatabaseStats {
    /// Calculate duplicate file count.
    #[must_use]
    pub fn duplicate_files(&self) -> usize {
        self.total_files.saturating_sub(self.unique_hashes)
    }

    /// Calculate deduplication ratio.
    #[must_use]
    pub fn dedup_ratio(&self) -> f64 {
        if self.total_files == 0 {
            return 0.0;
        }
        self.duplicate_files() as f64 / self.total_files as f64
    }

    /// Estimate potential storage savings.
    #[must_use]
    pub fn estimated_savings(&self) -> u64 {
        if self.total_files == 0 {
            return 0;
        }
        let avg_size = self.total_size / self.total_files as u64;
        avg_size * self.duplicate_files() as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_database_creation() {
        let db = DedupDatabase::open_memory()
            .await
            .expect("operation should succeed");
        let stats = db.get_stats().await.expect("operation should succeed");
        assert_eq!(stats.total_files, 0);
    }

    #[tokio::test]
    async fn test_insert_file() {
        let db = DedupDatabase::open_memory()
            .await
            .expect("operation should succeed");

        // Create a temporary file
        let temp_file = std::env::temp_dir().join("test_file.txt");
        std::fs::write(&temp_file, b"test content").expect("operation should succeed");

        let file_id = db
            .insert_file(&temp_file, "abcd1234")
            .await
            .expect("operation should succeed");
        assert!(file_id > 0);

        let count = db.count_files().await.expect("operation should succeed");
        assert_eq!(count, 1);

        // Cleanup
        std::fs::remove_file(&temp_file).ok();
    }

    #[tokio::test]
    async fn test_duplicate_detection() {
        let db = DedupDatabase::open_memory()
            .await
            .expect("operation should succeed");

        let temp_dir = std::env::temp_dir();
        let file1 = temp_dir.join("test1.txt");
        let file2 = temp_dir.join("test2.txt");

        std::fs::write(&file1, b"test").expect("operation should succeed");
        std::fs::write(&file2, b"test").expect("operation should succeed");

        let hash = "same_hash";

        db.insert_file(&file1, hash)
            .await
            .expect("operation should succeed");
        db.insert_file(&file2, hash)
            .await
            .expect("operation should succeed");

        let duplicates = db
            .find_duplicate_hashes()
            .await
            .expect("operation should succeed");
        assert_eq!(duplicates.len(), 1);
        assert_eq!(
            duplicates
                .get(hash)
                .expect("operation should succeed")
                .len(),
            2
        );

        // Cleanup
        std::fs::remove_file(&file1).ok();
        std::fs::remove_file(&file2).ok();
    }

    #[tokio::test]
    async fn test_fingerprints() {
        let db = DedupDatabase::open_memory()
            .await
            .expect("operation should succeed");

        let temp_file = std::env::temp_dir().join("test_fp.txt");
        std::fs::write(&temp_file, b"test").expect("operation should succeed");

        let file_id = db
            .insert_file(&temp_file, "hash123")
            .await
            .expect("operation should succeed");
        let fp_id = db
            .insert_fingerprint(file_id, "phash", "abc123")
            .await
            .expect("operation should succeed");

        assert!(fp_id > 0);

        // Cleanup
        std::fs::remove_file(&temp_file).ok();
    }

    #[tokio::test]
    async fn test_chunks() {
        let db = DedupDatabase::open_memory()
            .await
            .expect("operation should succeed");

        let temp_file = std::env::temp_dir().join("test_chunk.txt");
        std::fs::write(&temp_file, b"test").expect("operation should succeed");

        let file_id = db
            .insert_file(&temp_file, "hash456")
            .await
            .expect("operation should succeed");
        let chunk_id = db
            .insert_chunk(file_id, 0, 100, "chunk_hash")
            .await
            .expect("operation should succeed");

        assert!(chunk_id > 0);

        // Cleanup
        std::fs::remove_file(&temp_file).ok();
    }

    #[tokio::test]
    async fn test_delete_file() {
        let db = DedupDatabase::open_memory()
            .await
            .expect("operation should succeed");

        let temp_file = std::env::temp_dir().join("test_delete.txt");
        std::fs::write(&temp_file, b"test").expect("operation should succeed");

        db.insert_file(&temp_file, "hash_del")
            .await
            .expect("operation should succeed");

        let count_before = db.count_files().await.expect("operation should succeed");
        assert_eq!(count_before, 1);

        db.delete_file(&temp_file)
            .await
            .expect("operation should succeed");

        let count_after = db.count_files().await.expect("operation should succeed");
        assert_eq!(count_after, 0);

        // Cleanup
        std::fs::remove_file(&temp_file).ok();
    }

    #[tokio::test]
    async fn test_stats() {
        let db = DedupDatabase::open_memory()
            .await
            .expect("operation should succeed");
        let stats = db.get_stats().await.expect("operation should succeed");

        assert_eq!(stats.total_files, 0);
        assert_eq!(stats.unique_hashes, 0);
        assert_eq!(stats.duplicate_files(), 0);
        assert_eq!(stats.dedup_ratio(), 0.0);
    }
}
