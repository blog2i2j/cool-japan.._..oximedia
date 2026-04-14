//! Asset management and metadata handling
//!
//! Provides asset model and metadata extraction for:
//! - Technical metadata (codec, resolution, duration, etc.)
//! - Descriptive metadata (title, description, keywords)
//! - Rights metadata (copyright, license)
//! - Custom field support

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use uuid::Uuid;

use crate::database::Database;
use crate::{MamError, Result};

/// Asset manager handles asset operations
pub struct AssetManager {
    db: Arc<Database>,
    storage_path: PathBuf,
}

/// Complete asset record with all metadata
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct Asset {
    pub id: Uuid,
    pub filename: String,
    pub file_path: String,
    pub file_size: Option<i64>,
    pub mime_type: Option<String>,
    pub checksum: String,

    // Technical metadata
    pub duration_ms: Option<i64>,
    pub width: Option<i32>,
    pub height: Option<i32>,
    pub frame_rate: Option<f64>,
    pub video_codec: Option<String>,
    pub audio_codec: Option<String>,
    pub bit_rate: Option<i64>,

    // Descriptive metadata
    pub title: Option<String>,
    pub description: Option<String>,
    pub keywords: Option<Vec<String>>,
    pub categories: Option<Vec<String>>,

    // Rights metadata
    pub copyright: Option<String>,
    pub license: Option<String>,
    pub creator: Option<String>,

    // Custom metadata
    pub custom_metadata: Option<serde_json::Value>,

    // Status
    pub status: String,
    pub created_by: Option<Uuid>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Asset creation request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateAssetRequest {
    pub filename: String,
    pub file_path: String,
    pub file_size: Option<i64>,
    pub mime_type: Option<String>,
    pub checksum: String,
    pub technical: Option<TechnicalMetadata>,
    pub descriptive: Option<DescriptiveMetadata>,
    pub rights: Option<RightsMetadata>,
    pub custom: Option<HashMap<String, serde_json::Value>>,
    pub created_by: Option<Uuid>,
}

/// Asset update request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateAssetRequest {
    pub title: Option<String>,
    pub description: Option<String>,
    pub keywords: Option<Vec<String>>,
    pub categories: Option<Vec<String>>,
    pub copyright: Option<String>,
    pub license: Option<String>,
    pub creator: Option<String>,
    pub custom: Option<HashMap<String, serde_json::Value>>,
    pub status: Option<String>,
}

/// Technical metadata extracted from media file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TechnicalMetadata {
    pub duration_ms: Option<i64>,
    pub width: Option<i32>,
    pub height: Option<i32>,
    pub frame_rate: Option<f64>,
    pub video_codec: Option<String>,
    pub audio_codec: Option<String>,
    pub bit_rate: Option<i64>,
    pub audio_sample_rate: Option<i32>,
    pub audio_channels: Option<i32>,
    pub color_space: Option<String>,
    pub pixel_format: Option<String>,
}

/// Descriptive metadata for asset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DescriptiveMetadata {
    pub title: Option<String>,
    pub description: Option<String>,
    pub keywords: Vec<String>,
    pub categories: Vec<String>,
    pub language: Option<String>,
    pub genre: Option<String>,
    pub subject: Option<String>,
}

/// Rights management metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RightsMetadata {
    pub copyright: Option<String>,
    pub license: Option<String>,
    pub creator: Option<String>,
    pub publisher: Option<String>,
    pub rights_holder: Option<String>,
    pub usage_terms: Option<String>,
}

/// Asset search filters
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AssetFilter {
    pub mime_type: Option<String>,
    pub min_duration: Option<i64>,
    pub max_duration: Option<i64>,
    pub min_width: Option<i32>,
    pub max_width: Option<i32>,
    pub min_height: Option<i32>,
    pub max_height: Option<i32>,
    pub keywords: Option<Vec<String>>,
    pub categories: Option<Vec<String>>,
    pub status: Option<String>,
    pub created_after: Option<DateTime<Utc>>,
    pub created_before: Option<DateTime<Utc>>,
    pub created_by: Option<Uuid>,
}

/// Pagination parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pagination {
    pub limit: i64,
    pub offset: i64,
}

impl Default for Pagination {
    fn default() -> Self {
        Self {
            limit: 50,
            offset: 0,
        }
    }
}

/// Asset list result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetList {
    pub assets: Vec<Asset>,
    pub total: i64,
    pub limit: i64,
    pub offset: i64,
}

// ---------------------------------------------------------------------------
// Batch operation result types
// ---------------------------------------------------------------------------

/// Per-asset failure record for batch metadata updates.
#[derive(Debug, Clone)]
pub struct BatchUpdateFailure {
    /// ID of the asset that could not be updated.
    pub asset_id: Uuid,
    /// Human-readable reason for the failure.
    pub reason: String,
}

/// Aggregate result for [`AssetManager::batch_update_metadata`] and related.
#[derive(Debug, Clone)]
pub struct BatchUpdateResult {
    /// Number of assets successfully updated.
    pub success_count: usize,
    /// Number of assets that could not be updated.
    pub failure_count: usize,
    /// Per-asset failure details.
    pub failures: Vec<BatchUpdateFailure>,
}

impl BatchUpdateResult {
    /// Return `true` if all updates succeeded.
    #[must_use]
    pub fn is_all_success(&self) -> bool {
        self.failure_count == 0
    }

    /// Return the total number of assets processed.
    #[must_use]
    pub fn total(&self) -> usize {
        self.success_count + self.failure_count
    }
}

/// Per-asset failure record for batch deletes.
#[derive(Debug, Clone)]
pub struct BatchDeleteFailure {
    /// ID of the asset that could not be deleted.
    pub asset_id: Uuid,
    /// Human-readable reason for the failure.
    pub reason: String,
}

/// Aggregate result for [`AssetManager::batch_delete`].
#[derive(Debug, Clone)]
pub struct BatchDeleteResult {
    /// Number of assets successfully soft-deleted.
    pub deleted_count: usize,
    /// Number of assets that could not be deleted.
    pub failure_count: usize,
    /// Per-asset failure details.
    pub failures: Vec<BatchDeleteFailure>,
}

impl BatchDeleteResult {
    /// Return `true` if all deletes succeeded.
    #[must_use]
    pub fn is_all_success(&self) -> bool {
        self.failure_count == 0
    }

    /// Total assets processed.
    #[must_use]
    pub fn total(&self) -> usize {
        self.deleted_count + self.failure_count
    }
}

/// Per-asset failure record for batch status updates.
#[derive(Debug, Clone)]
pub struct BatchStatusFailure {
    /// ID of the asset whose status could not be changed.
    pub asset_id: Uuid,
    /// Human-readable reason for the failure.
    pub reason: String,
}

/// Aggregate result for [`AssetManager::batch_set_status`].
#[derive(Debug, Clone)]
pub struct BatchStatusResult {
    /// Number of assets whose status was changed.
    pub updated_count: usize,
    /// Number of assets where the status change failed.
    pub failure_count: usize,
    /// Per-asset failure details.
    pub failures: Vec<BatchStatusFailure>,
}

impl BatchStatusResult {
    /// Return `true` if all status updates succeeded.
    #[must_use]
    pub fn is_all_success(&self) -> bool {
        self.failure_count == 0
    }

    /// Total assets processed.
    #[must_use]
    pub fn total(&self) -> usize {
        self.updated_count + self.failure_count
    }
}

impl AssetManager {
    /// Create a new asset manager
    #[must_use]
    pub fn new(db: Arc<Database>, storage_path: String) -> Self {
        Self {
            db,
            storage_path: PathBuf::from(storage_path),
        }
    }

    /// Create a new asset
    ///
    /// # Errors
    ///
    /// Returns an error if the database insert fails
    pub async fn create_asset(&self, req: CreateAssetRequest) -> Result<Asset> {
        let custom_json = req.custom.and_then(|c| serde_json::to_value(c).ok());

        let asset = sqlx::query_as::<_, Asset>(
            "INSERT INTO assets
             (id, filename, file_path, file_size, mime_type, checksum,
              duration_ms, width, height, frame_rate, video_codec, audio_codec, bit_rate,
              title, description, keywords, categories,
              copyright, license, creator,
              custom_metadata, status, created_by, created_at, updated_at)
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, 'active', $22, NOW(), NOW())
             RETURNING *"
        )
        .bind(Uuid::new_v4())
        .bind(&req.filename)
        .bind(&req.file_path)
        .bind(req.file_size)
        .bind(req.mime_type)
        .bind(&req.checksum)
        .bind(req.technical.as_ref().and_then(|t| t.duration_ms))
        .bind(req.technical.as_ref().and_then(|t| t.width))
        .bind(req.technical.as_ref().and_then(|t| t.height))
        .bind(req.technical.as_ref().and_then(|t| t.frame_rate))
        .bind(req.technical.as_ref().and_then(|t| t.video_codec.clone()))
        .bind(req.technical.as_ref().and_then(|t| t.audio_codec.clone()))
        .bind(req.technical.as_ref().and_then(|t| t.bit_rate))
        .bind(req.descriptive.as_ref().and_then(|d| d.title.clone()))
        .bind(req.descriptive.as_ref().and_then(|d| d.description.clone()))
        .bind(req.descriptive.as_ref().map(|d| d.keywords.clone()))
        .bind(req.descriptive.as_ref().map(|d| d.categories.clone()))
        .bind(req.rights.as_ref().and_then(|r| r.copyright.clone()))
        .bind(req.rights.as_ref().and_then(|r| r.license.clone()))
        .bind(req.rights.as_ref().and_then(|r| r.creator.clone()))
        .bind(custom_json)
        .bind(req.created_by)
        .fetch_one(self.db.pool())
        .await?;

        Ok(asset)
    }

    /// Get asset by ID
    ///
    /// # Errors
    ///
    /// Returns an error if the asset is not found
    pub async fn get_asset(&self, asset_id: Uuid) -> Result<Asset> {
        let asset = sqlx::query_as::<_, Asset>("SELECT * FROM assets WHERE id = $1")
            .bind(asset_id)
            .fetch_one(self.db.pool())
            .await
            .map_err(|_| MamError::AssetNotFound(asset_id))?;

        Ok(asset)
    }

    /// Get asset by checksum
    ///
    /// # Errors
    ///
    /// Returns an error if the asset is not found
    pub async fn get_asset_by_checksum(&self, checksum: &str) -> Result<Option<Asset>> {
        let asset = sqlx::query_as::<_, Asset>("SELECT * FROM assets WHERE checksum = $1")
            .bind(checksum)
            .fetch_optional(self.db.pool())
            .await?;

        Ok(asset)
    }

    /// Update asset
    ///
    /// # Errors
    ///
    /// Returns an error if the update fails
    pub async fn update_asset(&self, asset_id: Uuid, req: UpdateAssetRequest) -> Result<Asset> {
        // Get current asset
        let current = self.get_asset(asset_id).await?;

        // Merge custom metadata
        let custom_json = if let Some(new_custom) = req.custom {
            let mut merged: HashMap<String, serde_json::Value> =
                if let Some(existing) = current.custom_metadata {
                    serde_json::from_value(existing).unwrap_or_default()
                } else {
                    HashMap::new()
                };

            merged.extend(new_custom);
            Some(serde_json::to_value(merged)?)
        } else {
            current.custom_metadata
        };

        let asset = sqlx::query_as::<_, Asset>(
            "UPDATE assets SET
                title = COALESCE($2, title),
                description = COALESCE($3, description),
                keywords = COALESCE($4, keywords),
                categories = COALESCE($5, categories),
                copyright = COALESCE($6, copyright),
                license = COALESCE($7, license),
                creator = COALESCE($8, creator),
                custom_metadata = COALESCE($9, custom_metadata),
                status = COALESCE($10, status),
                updated_at = NOW()
             WHERE id = $1
             RETURNING *",
        )
        .bind(asset_id)
        .bind(req.title)
        .bind(req.description)
        .bind(req.keywords)
        .bind(req.categories)
        .bind(req.copyright)
        .bind(req.license)
        .bind(req.creator)
        .bind(custom_json)
        .bind(req.status)
        .fetch_one(self.db.pool())
        .await?;

        Ok(asset)
    }

    /// Delete asset (soft delete)
    ///
    /// # Errors
    ///
    /// Returns an error if the update fails
    pub async fn delete_asset(&self, asset_id: Uuid) -> Result<()> {
        sqlx::query("UPDATE assets SET status = 'deleted', updated_at = NOW() WHERE id = $1")
            .bind(asset_id)
            .execute(self.db.pool())
            .await?;

        Ok(())
    }

    /// Hard delete asset and file
    ///
    /// # Errors
    ///
    /// Returns an error if the delete or file removal fails
    pub async fn purge_asset(&self, asset_id: Uuid) -> Result<()> {
        // Get asset to find file path
        let asset = self.get_asset(asset_id).await?;

        // Delete from database
        sqlx::query("DELETE FROM assets WHERE id = $1")
            .bind(asset_id)
            .execute(self.db.pool())
            .await?;

        // Delete file
        let file_path = Path::new(&asset.file_path);
        if file_path.exists() {
            tokio::fs::remove_file(file_path).await?;
        }

        Ok(())
    }

    /// List assets with filters
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails
    pub async fn list_assets(
        &self,
        filter: AssetFilter,
        pagination: Pagination,
    ) -> Result<AssetList> {
        let mut query = String::from("SELECT * FROM assets WHERE status != 'deleted'");
        let mut count_query = String::from("SELECT COUNT(*) FROM assets WHERE status != 'deleted'");
        let mut bindings: Vec<Box<dyn sqlx::Encode<'_, sqlx::Postgres> + Send>> = Vec::new();
        let mut param_num = 1;

        // Build WHERE clause
        if let Some(mime) = &filter.mime_type {
            query.push_str(&format!(" AND mime_type = ${param_num}"));
            count_query.push_str(&format!(" AND mime_type = ${param_num}"));
            bindings.push(Box::new(mime.clone()));
            param_num += 1;
        }

        if let Some(min_dur) = filter.min_duration {
            query.push_str(&format!(" AND duration_ms >= ${param_num}"));
            count_query.push_str(&format!(" AND duration_ms >= ${param_num}"));
            bindings.push(Box::new(min_dur));
            param_num += 1;
        }

        if let Some(max_dur) = filter.max_duration {
            query.push_str(&format!(" AND duration_ms <= ${param_num}"));
            count_query.push_str(&format!(" AND duration_ms <= ${param_num}"));
            bindings.push(Box::new(max_dur));
            param_num += 1;
        }

        if let Some(status) = &filter.status {
            query.push_str(&format!(" AND status = ${param_num}"));
            count_query.push_str(&format!(" AND status = ${param_num}"));
            bindings.push(Box::new(status.clone()));
            param_num += 1;
        }

        if let Some(created_by) = filter.created_by {
            query.push_str(&format!(" AND created_by = ${param_num}"));
            count_query.push_str(&format!(" AND created_by = ${param_num}"));
            bindings.push(Box::new(created_by));
            param_num += 1;
        }

        // Add ordering and pagination
        query.push_str(&format!(
            " ORDER BY created_at DESC LIMIT ${param_num} OFFSET ${}",
            param_num + 1
        ));

        // Execute count query
        let total: i64 = sqlx::query_scalar(&count_query)
            .fetch_one(self.db.pool())
            .await?;

        // Execute main query
        let assets = sqlx::query_as::<_, Asset>(&query)
            .bind(pagination.limit)
            .bind(pagination.offset)
            .fetch_all(self.db.pool())
            .await?;

        Ok(AssetList {
            assets,
            total,
            limit: pagination.limit,
            offset: pagination.offset,
        })
    }

    /// Search assets by text
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails
    pub async fn search_assets(&self, query: &str, pagination: Pagination) -> Result<AssetList> {
        let search_pattern = format!("%{query}%");

        let total: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM assets
             WHERE status != 'deleted'
             AND (title ILIKE $1 OR description ILIKE $1 OR filename ILIKE $1)",
        )
        .bind(&search_pattern)
        .fetch_one(self.db.pool())
        .await?;

        let assets = sqlx::query_as::<_, Asset>(
            "SELECT * FROM assets
             WHERE status != 'deleted'
             AND (title ILIKE $1 OR description ILIKE $1 OR filename ILIKE $1)
             ORDER BY created_at DESC
             LIMIT $2 OFFSET $3",
        )
        .bind(&search_pattern)
        .bind(pagination.limit)
        .bind(pagination.offset)
        .fetch_all(self.db.pool())
        .await?;

        Ok(AssetList {
            assets,
            total,
            limit: pagination.limit,
            offset: pagination.offset,
        })
    }

    // -------------------------------------------------------------------------
    // Batch / bulk database operations
    // -------------------------------------------------------------------------

    /// Apply the same [`UpdateAssetRequest`] to multiple assets in a single
    /// logical operation.
    ///
    /// Each asset update is executed individually so that partial failures can
    /// be reported without rolling back successful updates.  A summary is
    /// returned describing the per-asset outcome.
    ///
    /// # Errors
    ///
    /// Never returns a top-level `Err`; per-asset failures are captured in
    /// [`BatchUpdateResult::failures`].
    pub async fn batch_update_metadata(
        &self,
        asset_ids: Vec<Uuid>,
        req: UpdateAssetRequest,
    ) -> BatchUpdateResult {
        let mut result = BatchUpdateResult {
            success_count: 0,
            failure_count: 0,
            failures: Vec::new(),
        };

        for id in asset_ids {
            match self.update_asset(id, req.clone()).await {
                Ok(_) => result.success_count += 1,
                Err(e) => {
                    result.failure_count += 1;
                    result.failures.push(BatchUpdateFailure {
                        asset_id: id,
                        reason: e.to_string(),
                    });
                }
            }
        }

        result
    }

    /// Apply individual [`UpdateAssetRequest`] values to specific assets.
    ///
    /// Useful when each asset needs a different metadata update.  Each entry
    /// in `updates` is `(asset_id, request)`.
    ///
    /// Returns a [`BatchUpdateResult`] summarising outcomes.
    ///
    /// # Errors
    ///
    /// Never returns a top-level `Err`; per-asset failures are in the result.
    pub async fn batch_update_metadata_individual(
        &self,
        updates: Vec<(Uuid, UpdateAssetRequest)>,
    ) -> BatchUpdateResult {
        let mut result = BatchUpdateResult {
            success_count: 0,
            failure_count: 0,
            failures: Vec::new(),
        };

        for (id, req) in updates {
            match self.update_asset(id, req).await {
                Ok(_) => result.success_count += 1,
                Err(e) => {
                    result.failure_count += 1;
                    result.failures.push(BatchUpdateFailure {
                        asset_id: id,
                        reason: e.to_string(),
                    });
                }
            }
        }

        result
    }

    /// Soft-delete multiple assets in a single batch operation.
    ///
    /// Each deletion is applied individually; failures are collected and
    /// returned rather than causing the whole batch to abort.
    ///
    /// # Errors
    ///
    /// Never returns a top-level `Err`.
    pub async fn batch_delete(&self, asset_ids: Vec<Uuid>) -> BatchDeleteResult {
        let mut result = BatchDeleteResult {
            deleted_count: 0,
            failure_count: 0,
            failures: Vec::new(),
        };

        for id in asset_ids {
            match self.delete_asset(id).await {
                Ok(()) => result.deleted_count += 1,
                Err(e) => {
                    result.failure_count += 1;
                    result.failures.push(BatchDeleteFailure {
                        asset_id: id,
                        reason: e.to_string(),
                    });
                }
            }
        }

        result
    }

    /// Fetch multiple assets by their IDs in a single call.
    ///
    /// Returns a tuple `(found, missing)` where `found` is the list of
    /// successfully retrieved assets and `missing` contains IDs that were
    /// not found in the database.
    ///
    /// # Errors
    ///
    /// Returns a database error only if the connection itself fails.
    pub async fn batch_get(&self, asset_ids: Vec<Uuid>) -> Result<(Vec<Asset>, Vec<Uuid>)> {
        let mut found = Vec::new();
        let mut missing = Vec::new();

        for id in asset_ids {
            match sqlx::query_as::<_, Asset>("SELECT * FROM assets WHERE id = $1")
                .bind(id)
                .fetch_optional(self.db.pool())
                .await?
            {
                Some(asset) => found.push(asset),
                None => missing.push(id),
            }
        }

        Ok((found, missing))
    }

    /// Change the status field for multiple assets at once.
    ///
    /// This is a lightweight alternative to a full metadata update when only
    /// the workflow status needs to change (e.g. bulk approval, bulk archive).
    ///
    /// Returns the number of assets whose status was successfully changed.
    ///
    /// # Errors
    ///
    /// Returns a database error only if the connection itself fails.
    pub async fn batch_set_status(
        &self,
        asset_ids: Vec<Uuid>,
        new_status: &str,
    ) -> Result<BatchStatusResult> {
        let mut result = BatchStatusResult {
            updated_count: 0,
            failure_count: 0,
            failures: Vec::new(),
        };

        for id in asset_ids {
            let outcome =
                sqlx::query("UPDATE assets SET status = $1, updated_at = NOW() WHERE id = $2")
                    .bind(new_status)
                    .bind(id)
                    .execute(self.db.pool())
                    .await;

            match outcome {
                Ok(r) if r.rows_affected() > 0 => result.updated_count += 1,
                Ok(_) => {
                    result.failure_count += 1;
                    result.failures.push(BatchStatusFailure {
                        asset_id: id,
                        reason: "asset not found".to_string(),
                    });
                }
                Err(e) => {
                    result.failure_count += 1;
                    result.failures.push(BatchStatusFailure {
                        asset_id: id,
                        reason: e.to_string(),
                    });
                }
            }
        }

        Ok(result)
    }

    /// Upsert custom metadata fields for multiple assets.
    ///
    /// For each `(asset_id, fields)` entry the provided key-value pairs are
    /// merged into the asset's `custom_metadata` JSON column.  Existing keys
    /// not present in `fields` are left untouched.
    ///
    /// # Errors
    ///
    /// Returns a database error only if the connection itself fails.
    pub async fn batch_upsert_custom_fields(
        &self,
        updates: Vec<(Uuid, HashMap<String, serde_json::Value>)>,
    ) -> BatchUpdateResult {
        let mut result = BatchUpdateResult {
            success_count: 0,
            failure_count: 0,
            failures: Vec::new(),
        };

        for (id, fields) in updates {
            let req = UpdateAssetRequest {
                title: None,
                description: None,
                keywords: None,
                categories: None,
                copyright: None,
                license: None,
                creator: None,
                custom: Some(fields),
                status: None,
            };
            match self.update_asset(id, req).await {
                Ok(_) => result.success_count += 1,
                Err(e) => {
                    result.failure_count += 1;
                    result.failures.push(BatchUpdateFailure {
                        asset_id: id,
                        reason: e.to_string(),
                    });
                }
            }
        }

        result
    }

    /// Get asset storage path
    #[must_use]
    pub fn get_storage_path(&self, asset_id: Uuid, filename: &str) -> PathBuf {
        // Organize by UUID prefix for better filesystem distribution
        let id_str = asset_id.to_string();
        let prefix = &id_str[..2];

        self.storage_path.join(prefix).join(id_str).join(filename)
    }

    /// Extract technical metadata from file
    ///
    /// # Errors
    ///
    /// Returns an error if metadata extraction fails
    pub async fn extract_technical_metadata(&self, file_path: &Path) -> Result<TechnicalMetadata> {
        tracing::info!("Extracting technical metadata from: {:?}", file_path);

        // Read file bytes for magic byte detection (up to 64KB)
        let file_bytes = {
            use tokio::io::AsyncReadExt;
            let mut f = tokio::fs::File::open(file_path).await?;
            let mut buf = vec![0u8; 65536];
            let n = f.read(&mut buf).await?;
            buf.truncate(n);
            buf
        };

        // Get file size for bit-rate estimation
        let file_size = tokio::fs::metadata(file_path).await?.len();

        // Detect container/codec from magic bytes and parse metadata
        let metadata = extract_metadata_from_bytes(&file_bytes, file_size);

        Ok(metadata)
    }

    /// Calculate file checksum (SHA-256)
    ///
    /// # Errors
    ///
    /// Returns an error if file reading or hashing fails
    pub async fn calculate_checksum(&self, file_path: &Path) -> Result<String> {
        use sha2::{Digest, Sha256};

        let mut file = tokio::fs::File::open(file_path).await?;
        let mut hasher = Sha256::new();
        let mut buffer = vec![0u8; 8192];

        loop {
            use tokio::io::AsyncReadExt;
            let n = file.read(&mut buffer).await?;
            if n == 0 {
                break;
            }
            hasher.update(&buffer[..n]);
        }

        let result = hasher.finalize();
        Ok(result.iter().map(|b| format!("{b:02x}")).collect())
    }

    /// Get MIME type from file extension
    #[must_use]
    pub fn get_mime_type(file_path: &Path) -> Option<String> {
        let extension = file_path.extension()?.to_str()?;

        let mime = match extension.to_lowercase().as_str() {
            "mp4" => "video/mp4",
            "mov" => "video/quicktime",
            "avi" => "video/x-msvideo",
            "mkv" => "video/x-matroska",
            "webm" => "video/webm",
            "mp3" => "audio/mpeg",
            "wav" => "audio/wav",
            "aac" => "audio/aac",
            "flac" => "audio/flac",
            "jpg" | "jpeg" => "image/jpeg",
            "png" => "image/png",
            "gif" => "image/gif",
            "webp" => "image/webp",
            "pdf" => "application/pdf",
            "xml" => "application/xml",
            "json" => "application/json",
            _ => return None,
        };

        Some(mime.to_string())
    }

    /// Validate asset data
    ///
    /// # Errors
    ///
    /// Returns an error if validation fails
    pub fn validate_asset(&self, req: &CreateAssetRequest) -> Result<()> {
        if req.filename.is_empty() {
            return Err(MamError::InvalidInput(
                "Filename cannot be empty".to_string(),
            ));
        }

        if req.file_path.is_empty() {
            return Err(MamError::InvalidInput(
                "File path cannot be empty".to_string(),
            ));
        }

        if req.checksum.is_empty() {
            return Err(MamError::InvalidInput(
                "Checksum cannot be empty".to_string(),
            ));
        }

        // Validate checksum format (should be 64 hex characters for SHA-256)
        if req.checksum.len() != 64 || !req.checksum.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err(MamError::InvalidInput(
                "Invalid checksum format".to_string(),
            ));
        }

        Ok(())
    }
}

/// Detect media format from magic bytes and return a format identifier.
fn detect_media_format(bytes: &[u8]) -> &'static str {
    if bytes.len() < 8 {
        return "unknown";
    }
    // MP4 / MOV / M4V: check ftyp box at offset 4
    if bytes.len() >= 12 && &bytes[4..8] == b"ftyp" {
        return "mp4";
    }
    // Matroska / WebM: EBML header magic
    if bytes.len() >= 4
        && bytes[0] == 0x1A
        && bytes[1] == 0x45
        && bytes[2] == 0xDF
        && bytes[3] == 0xA3
    {
        if bytes[..bytes.len().min(64)]
            .windows(4)
            .any(|w| w == b"webm")
        {
            return "webm";
        }
        return "mkv";
    }
    // AVI: RIFF....AVI
    if bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"AVI " {
        return "avi";
    }
    // WAV: RIFF....WAVE
    if bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WAVE" {
        return "wav";
    }
    // FLAC magic
    if bytes.len() >= 4 && &bytes[0..4] == b"fLaC" {
        return "flac";
    }
    // OGG
    if bytes.len() >= 4 && &bytes[0..4] == b"OggS" {
        return "ogg";
    }
    // MP3: ID3 tag or sync word
    if bytes.len() >= 3 && &bytes[0..3] == b"ID3" {
        return "mp3";
    }
    if bytes.len() >= 2 && bytes[0] == 0xFF && (bytes[1] & 0xE0) == 0xE0 {
        return "mp3";
    }
    // AAC ADTS
    if bytes.len() >= 2 && bytes[0] == 0xFF && (bytes[1] & 0xF6) == 0xF0 {
        return "aac";
    }
    // JPEG
    if bytes.len() >= 3 && bytes[0] == 0xFF && bytes[1] == 0xD8 && bytes[2] == 0xFF {
        return "jpeg";
    }
    // PNG: 8-byte signature
    if bytes.len() >= 8
        && bytes[0] == 0x89
        && bytes[1] == 0x50
        && bytes[2] == 0x4E
        && bytes[3] == 0x47
        && bytes[4] == 0x0D
        && bytes[5] == 0x0A
        && bytes[6] == 0x1A
        && bytes[7] == 0x0A
    {
        return "png";
    }
    // GIF
    if bytes.len() >= 6 && (&bytes[0..6] == b"GIF87a" || &bytes[0..6] == b"GIF89a") {
        return "gif";
    }
    // MPEG-2 Transport Stream: repeating sync byte 0x47 every 188 bytes
    if bytes.len() >= 188 && bytes[0] == 0x47 && bytes[188] == 0x47 {
        return "mpegts";
    }
    "unknown"
}

/// Read a big-endian u32 from a byte slice at given offset.
#[allow(dead_code)]
fn read_u32_be(data: &[u8], offset: usize) -> Option<u32> {
    if data.len() < offset + 4 {
        return None;
    }
    Some(u32::from_be_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ]))
}

/// Read a little-endian u32 from a byte slice at given offset.
#[allow(dead_code)]
fn read_u32_le(data: &[u8], offset: usize) -> Option<u32> {
    if data.len() < offset + 4 {
        return None;
    }
    Some(u32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ]))
}

/// Read a big-endian u16 from a byte slice at given offset.
#[allow(dead_code)]
fn read_u16_be(data: &[u8], offset: usize) -> Option<u16> {
    if data.len() < offset + 2 {
        return None;
    }
    Some(u16::from_be_bytes([data[offset], data[offset + 1]]))
}

/// Parse technical metadata from raw file bytes using format-specific heuristics.
fn extract_metadata_from_bytes(bytes: &[u8], file_size: u64) -> TechnicalMetadata {
    match detect_media_format(bytes) {
        "mp4" => parse_mp4_metadata(bytes, file_size),
        "mkv" => parse_mkv_metadata(bytes, file_size, "mkv"),
        "webm" => parse_mkv_metadata(bytes, file_size, "webm"),
        "wav" => parse_wav_metadata(bytes, file_size),
        "flac" => parse_flac_metadata(bytes, file_size),
        "mp3" => parse_mp3_metadata(bytes, file_size),
        "avi" => parse_avi_metadata(bytes, file_size),
        "jpeg" => parse_image_metadata(bytes, "jpeg"),
        "png" => parse_image_metadata(bytes, "png"),
        "gif" => parse_image_metadata(bytes, "gif"),
        _ => TechnicalMetadata {
            duration_ms: None,
            width: None,
            height: None,
            frame_rate: None,
            video_codec: None,
            audio_codec: None,
            bit_rate: None,
            audio_sample_rate: None,
            audio_channels: None,
            color_space: None,
            pixel_format: None,
        },
    }
}

/// Parse MP4/MOV metadata by walking the box hierarchy.
fn parse_mp4_metadata(bytes: &[u8], file_size: u64) -> TechnicalMetadata {
    let mut meta = TechnicalMetadata {
        video_codec: None,
        audio_codec: None,
        duration_ms: None,
        width: None,
        height: None,
        frame_rate: None,
        bit_rate: None,
        audio_sample_rate: None,
        audio_channels: None,
        color_space: Some("bt709".to_string()),
        pixel_format: Some("yuv420p".to_string()),
    };
    let mut offset = 0usize;
    while offset + 8 <= bytes.len() {
        let box_size = read_u32_be(bytes, offset).unwrap_or(0) as usize;
        if box_size < 8 || offset + box_size > bytes.len() {
            break;
        }
        if &bytes[offset + 4..offset + 8] == b"moov" {
            parse_mp4_moov(&bytes[offset + 8..offset + box_size], &mut meta);
            break;
        }
        offset += box_size;
    }
    if let Some(dur_ms) = meta.duration_ms {
        if dur_ms > 0 {
            meta.bit_rate = Some((file_size * 8 * 1000) as i64 / dur_ms);
        }
    }
    meta
}

fn parse_mp4_moov(data: &[u8], meta: &mut TechnicalMetadata) {
    let mut offset = 0usize;
    while offset + 8 <= data.len() {
        let box_size = read_u32_be(data, offset).unwrap_or(0) as usize;
        if box_size < 8 || offset + box_size > data.len() {
            break;
        }
        let box_type = &data[offset + 4..offset + 8];
        if box_type == b"mvhd" {
            let inner = &data[offset + 8..];
            let version = inner.first().copied().unwrap_or(0);
            if version == 0 && inner.len() >= 20 {
                let timescale = read_u32_be(inner, 12).unwrap_or(1) as i64;
                let duration = read_u32_be(inner, 16).unwrap_or(0) as i64;
                if timescale > 0 {
                    meta.duration_ms = Some(duration * 1000 / timescale);
                }
            } else if version == 1 && inner.len() >= 32 {
                let timescale = read_u32_be(inner, 20).unwrap_or(1) as i64;
                if inner.len() >= 32 {
                    let dur_hi = read_u32_be(inner, 24).unwrap_or(0) as i64;
                    let dur_lo = read_u32_be(inner, 28).unwrap_or(0) as i64;
                    let duration = (dur_hi << 32) | dur_lo;
                    if timescale > 0 {
                        meta.duration_ms = Some(duration * 1000 / timescale);
                    }
                }
            }
        } else if box_type == b"trak" {
            parse_mp4_trak(&data[offset + 8..offset + box_size], meta);
        }
        offset += box_size;
    }
}

fn parse_mp4_trak(data: &[u8], meta: &mut TechnicalMetadata) {
    let mut offset = 0usize;
    while offset + 8 <= data.len() {
        let box_size = read_u32_be(data, offset).unwrap_or(0) as usize;
        if box_size < 8 || offset + box_size > data.len() {
            break;
        }
        if &data[offset + 4..offset + 8] == b"mdia" {
            parse_mp4_mdia(&data[offset + 8..offset + box_size], meta);
        }
        offset += box_size;
    }
}

fn parse_mp4_mdia(data: &[u8], meta: &mut TechnicalMetadata) {
    let mut handler_type = [0u8; 4];
    // First pass: locate hdlr
    let mut offset = 0usize;
    while offset + 8 <= data.len() {
        let box_size = read_u32_be(data, offset).unwrap_or(0) as usize;
        if box_size < 8 || offset + box_size > data.len() {
            break;
        }
        if &data[offset + 4..offset + 8] == b"hdlr" {
            let inner = &data[offset + 8..];
            if inner.len() >= 12 {
                handler_type.copy_from_slice(&inner[8..12]);
            }
        }
        offset += box_size;
    }
    // Second pass: locate minf
    offset = 0;
    while offset + 8 <= data.len() {
        let box_size = read_u32_be(data, offset).unwrap_or(0) as usize;
        if box_size < 8 || offset + box_size > data.len() {
            break;
        }
        if &data[offset + 4..offset + 8] == b"minf" {
            parse_mp4_minf(&data[offset + 8..offset + box_size], &handler_type, meta);
        }
        offset += box_size;
    }
}

fn parse_mp4_minf(data: &[u8], handler: &[u8; 4], meta: &mut TechnicalMetadata) {
    let mut offset = 0usize;
    while offset + 8 <= data.len() {
        let box_size = read_u32_be(data, offset).unwrap_or(0) as usize;
        if box_size < 8 || offset + box_size > data.len() {
            break;
        }
        if &data[offset + 4..offset + 8] == b"stbl" {
            parse_mp4_stbl(&data[offset + 8..offset + box_size], handler, meta);
        }
        offset += box_size;
    }
}

fn parse_mp4_stbl(data: &[u8], handler: &[u8; 4], meta: &mut TechnicalMetadata) {
    let mut offset = 0usize;
    while offset + 8 <= data.len() {
        let box_size = read_u32_be(data, offset).unwrap_or(0) as usize;
        if box_size < 8 || offset + box_size > data.len() {
            break;
        }
        if &data[offset + 4..offset + 8] == b"stsd" {
            parse_mp4_stsd(&data[offset + 8..offset + box_size], handler, meta);
        }
        offset += box_size;
    }
}

fn parse_mp4_stsd(data: &[u8], handler: &[u8; 4], meta: &mut TechnicalMetadata) {
    // stsd: version(1)+flags(3)+entry_count(4)+entries...
    if data.len() < 8 {
        return;
    }
    let entry_count = read_u32_be(data, 4).unwrap_or(0);
    if entry_count == 0 {
        return;
    }
    let entry_start = 8usize;
    if entry_start + 8 > data.len() {
        return;
    }
    let codec_fourcc = &data[entry_start + 4..entry_start + 8];
    let codec_str = std::str::from_utf8(codec_fourcc)
        .unwrap_or("")
        .trim()
        .to_lowercase();

    if handler == b"vide" {
        let codec_name = match codec_str.as_str() {
            "avc1" | "avc2" | "avc3" | "avc4" => "h264",
            "hev1" | "hvc1" => "hevc",
            "av01" => "av1",
            "vp08" => "vp8",
            "vp09" => "vp9",
            "mp4v" => "mpeg4",
            "mjp2" | "mjpg" => "mjpeg",
            "apch" | "apcn" | "apco" | "ap4h" | "ap4x" | "prores" => "prores",
            "dnxhd" | "avdn" => "dnxhd",
            _ => "unknown",
        };
        if meta.video_codec.is_none() {
            meta.video_codec = Some(codec_name.to_string());
        }
        // VisualSampleEntry: entry header(8) + reserved(6) + data_ref_idx(2) +
        //   pre_defined(2) + reserved(2) + pre_defined(12) + width(2) + height(2)
        let width_offset = entry_start + 8 + 6 + 2 + 2 + 2 + 12;
        if width_offset + 4 <= data.len() {
            let width = read_u16_be(data, width_offset).unwrap_or(0);
            let height = read_u16_be(data, width_offset + 2).unwrap_or(0);
            if width > 0 && height > 0 {
                meta.width = Some(i32::from(width));
                meta.height = Some(i32::from(height));
            }
        }
    } else if handler == b"soun" {
        let codec_name = match codec_str.as_str() {
            "mp4a" => "aac",
            "ac-3" | "ac3" => "ac3",
            "ec-3" | "ec3" => "eac3",
            "alac" => "alac",
            "flac" => "flac",
            "opus" => "opus",
            "mp3" | ".mp3" => "mp3",
            _ => "pcm",
        };
        if meta.audio_codec.is_none() {
            meta.audio_codec = Some(codec_name.to_string());
        }
        // AudioSampleEntry: entry header(8) + reserved(6) + data_ref_idx(2) +
        //   reserved(8) + channelcount(2) + samplesize(2) + pre_defined(2) +
        //   reserved(2) + samplerate(4, fixed-point 16.16)
        let chan_offset = entry_start + 8 + 6 + 2 + 8;
        if chan_offset + 8 <= data.len() {
            let channels = read_u16_be(data, chan_offset).unwrap_or(0);
            if channels > 0 {
                meta.audio_channels = Some(i32::from(channels));
            }
            // samplerate upper 16 bits at chan_offset + 4
            let sr_hi = read_u16_be(data, chan_offset + 4).unwrap_or(0);
            if sr_hi > 0 {
                meta.audio_sample_rate = Some(i32::from(sr_hi));
            }
        }
    }
}

/// Parse Matroska/WebM metadata by scanning for codec string patterns.
fn parse_mkv_metadata(bytes: &[u8], file_size: u64, format: &str) -> TechnicalMetadata {
    let mut meta = TechnicalMetadata {
        video_codec: if format == "webm" {
            Some("vp9".to_string())
        } else {
            None
        },
        audio_codec: if format == "webm" {
            Some("opus".to_string())
        } else {
            None
        },
        duration_ms: None,
        width: None,
        height: None,
        frame_rate: None,
        bit_rate: None,
        audio_sample_rate: None,
        audio_channels: None,
        color_space: None,
        pixel_format: Some("yuv420p".to_string()),
    };
    // Scan for known codec ID strings embedded in the binary
    let codec_patterns: &[(&[u8], &str, bool)] = &[
        (b"V_MPEG4/ISO/AVC", "h264", true),
        (b"V_MPEGH/ISO/HEVC", "hevc", true),
        (b"V_AV1", "av1", true),
        (b"V_VP9", "vp9", true),
        (b"V_VP8", "vp8", true),
        (b"V_MPEG2", "mpeg2video", true),
        (b"V_PRORES", "prores", true),
        (b"V_DNXHD", "dnxhd", true),
        (b"A_AAC", "aac", false),
        (b"A_AC3", "ac3", false),
        (b"A_EAC3", "eac3", false),
        (b"A_OPUS", "opus", false),
        (b"A_VORBIS", "vorbis", false),
        (b"A_FLAC", "flac", false),
        (b"A_DTS", "dts", false),
        (b"A_TRUEHD", "truehd", false),
        (b"A_MPEG/L3", "mp3", false),
        (b"A_PCM/INT/LIT", "pcm_s16le", false),
    ];
    for &(pattern, codec, is_video) in codec_patterns {
        if bytes.windows(pattern.len()).any(|w| w == pattern) {
            if is_video && meta.video_codec.is_none() {
                meta.video_codec = Some(codec.to_string());
            } else if !is_video && meta.audio_codec.is_none() {
                meta.audio_codec = Some(codec.to_string());
            }
        }
    }
    if let Some(dur_ms) = meta.duration_ms {
        if dur_ms > 0 {
            meta.bit_rate = Some((file_size * 8 * 1000) as i64 / dur_ms);
        }
    }
    meta
}

/// Parse WAV metadata from RIFF/WAVE structure.
fn parse_wav_metadata(bytes: &[u8], file_size: u64) -> TechnicalMetadata {
    let mut meta = TechnicalMetadata {
        video_codec: None,
        audio_codec: Some("pcm_s16le".to_string()),
        duration_ms: None,
        width: None,
        height: None,
        frame_rate: None,
        bit_rate: None,
        audio_sample_rate: None,
        audio_channels: None,
        color_space: None,
        pixel_format: None,
    };
    if bytes.len() < 44 {
        return meta;
    }
    let mut offset = 12usize; // skip RIFF+size+WAVE
    while offset + 8 <= bytes.len() {
        let chunk_id = &bytes[offset..offset + 4];
        let chunk_size = read_u32_le(bytes, offset + 4).unwrap_or(0) as usize;
        if chunk_id == b"fmt " && chunk_size >= 16 {
            let fmt = &bytes[offset + 8..];
            if fmt.len() >= 16 {
                let audio_format = u16::from_le_bytes([fmt[0], fmt[1]]);
                let channels = u16::from_le_bytes([fmt[2], fmt[3]]);
                let sample_rate = u32::from_le_bytes([fmt[4], fmt[5], fmt[6], fmt[7]]);
                let byte_rate = u32::from_le_bytes([fmt[8], fmt[9], fmt[10], fmt[11]]);
                let bits_per_sample = u16::from_le_bytes([fmt[14], fmt[15]]);
                meta.audio_channels = Some(i32::from(channels));
                meta.audio_sample_rate = Some(sample_rate as i32);
                meta.bit_rate = Some(i64::from(byte_rate) * 8);
                meta.audio_codec = Some(
                    match (audio_format, bits_per_sample) {
                        (1, 8) => "pcm_u8",
                        (1, 16) => "pcm_s16le",
                        (1, 24) => "pcm_s24le",
                        (1, 32) => "pcm_s32le",
                        (3, _) => "pcm_f32le",
                        (6, _) => "pcm_alaw",
                        (7, _) => "pcm_mulaw",
                        _ => "pcm",
                    }
                    .to_string(),
                );
            }
        } else if chunk_id == b"data" {
            let data_size = chunk_size as u64;
            if let (Some(sr), Some(ch)) = (meta.audio_sample_rate, meta.audio_channels) {
                let bps = 2u64; // assume 16-bit
                let denom = bps * ch as u64;
                if denom > 0 && sr > 0 {
                    let total_samples = data_size / denom;
                    meta.duration_ms = Some((total_samples * 1000) as i64 / i64::from(sr));
                }
            }
            if meta.duration_ms.is_none() {
                if let Some(br) = meta.bit_rate {
                    if br > 0 {
                        meta.duration_ms = Some(file_size as i64 * 8 * 1000 / br);
                    }
                }
            }
        }
        offset += 8 + chunk_size;
        if chunk_size % 2 != 0 {
            offset += 1;
        }
    }
    meta
}

/// Parse FLAC metadata from METADATA_BLOCK_STREAMINFO.
fn parse_flac_metadata(bytes: &[u8], _file_size: u64) -> TechnicalMetadata {
    let mut meta = TechnicalMetadata {
        video_codec: None,
        audio_codec: Some("flac".to_string()),
        duration_ms: None,
        width: None,
        height: None,
        frame_rate: None,
        bit_rate: None,
        audio_sample_rate: None,
        audio_channels: None,
        color_space: None,
        pixel_format: None,
    };
    // "fLaC"(4) + block_header(4) + STREAMINFO data (>=34 bytes)
    if bytes.len() < 4 + 4 + 18 {
        return meta;
    }
    let si = &bytes[8..]; // skip magic + block header
    if si.len() < 18 {
        return meta;
    }
    // sample_rate: top 20 bits at si[10..12]
    let sample_rate = ((si[10] as u32) << 12) | ((si[11] as u32) << 4) | ((si[12] as u32) >> 4);
    let channels = ((si[12] & 0x0E) >> 1) + 1;
    let bits_per_sample = (((si[12] & 0x01) << 4) | ((si[13] & 0xF0) >> 4)) + 1;
    let total_samples = (((si[13] & 0x0F) as u64) << 32)
        | ((si[14] as u64) << 24)
        | ((si[15] as u64) << 16)
        | ((si[16] as u64) << 8)
        | (si[17] as u64);
    if sample_rate > 0 {
        meta.audio_sample_rate = Some(sample_rate as i32);
        if total_samples > 0 {
            meta.duration_ms = Some((total_samples * 1000 / sample_rate as u64) as i64);
        }
    }
    meta.audio_channels = Some(i32::from(channels));
    meta.audio_codec = Some(format!("flac ({}b)", bits_per_sample));
    meta
}

/// Parse MP3 metadata from ID3 header and first MPEG sync frame.
fn parse_mp3_metadata(bytes: &[u8], file_size: u64) -> TechnicalMetadata {
    let mut meta = TechnicalMetadata {
        video_codec: None,
        audio_codec: Some("mp3".to_string()),
        duration_ms: None,
        width: None,
        height: None,
        frame_rate: None,
        bit_rate: None,
        audio_sample_rate: None,
        audio_channels: None,
        color_space: None,
        pixel_format: None,
    };
    // Skip ID3v2 tag
    let mut frame_offset = 0usize;
    if bytes.len() >= 10 && &bytes[0..3] == b"ID3" {
        let id3_size = ((bytes[6] as usize & 0x7F) << 21)
            | ((bytes[7] as usize & 0x7F) << 14)
            | ((bytes[8] as usize & 0x7F) << 7)
            | (bytes[9] as usize & 0x7F);
        frame_offset = 10 + id3_size;
    }
    // Find MPEG sync frame
    let mut scan = frame_offset;
    while scan + 4 <= bytes.len() {
        if bytes[scan] == 0xFF && (bytes[scan + 1] & 0xE0) == 0xE0 {
            let h1 = bytes[scan + 1];
            let h2 = bytes[scan + 2];
            let version_id = (h1 >> 3) & 0x03;
            let bitrate_idx = (h2 >> 4) as usize;
            let sr_idx = ((h2 >> 2) & 0x03) as usize;
            let channel_mode = bytes[scan + 3] >> 6;
            let mpeg1_l3 = [
                0u32, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0,
            ];
            let mpeg2_l3 = [
                0u32, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0,
            ];
            let br_kbps = if version_id == 3 {
                mpeg1_l3.get(bitrate_idx).copied().unwrap_or(0)
            } else {
                mpeg2_l3.get(bitrate_idx).copied().unwrap_or(0)
            };
            let sr_table: &[u32] = match version_id {
                3 => &[44100, 48000, 32000, 0],
                2 => &[22050, 24000, 16000, 0],
                _ => &[11025, 12000, 8000, 0],
            };
            let sample_rate = sr_table.get(sr_idx).copied().unwrap_or(0);
            let channels = if channel_mode == 3 { 1i32 } else { 2 };
            if sample_rate > 0 {
                meta.audio_sample_rate = Some(sample_rate as i32);
            }
            meta.audio_channels = Some(channels);
            if br_kbps > 0 {
                let br = i64::from(br_kbps) * 1000;
                meta.bit_rate = Some(br);
                meta.duration_ms = Some(file_size as i64 * 8 * 1000 / br);
            }
            break;
        }
        scan += 1;
    }
    meta
}

/// Parse AVI metadata from RIFF/AVI structure.
fn parse_avi_metadata(bytes: &[u8], file_size: u64) -> TechnicalMetadata {
    let mut meta = TechnicalMetadata {
        video_codec: None,
        audio_codec: None,
        duration_ms: None,
        width: None,
        height: None,
        frame_rate: None,
        bit_rate: None,
        audio_sample_rate: None,
        audio_channels: None,
        color_space: None,
        pixel_format: Some("yuv420p".to_string()),
    };
    if bytes.len() < 56 {
        return meta;
    }
    let mut offset = 12usize;
    while offset + 8 <= bytes.len() {
        let chunk_id = &bytes[offset..offset + 4];
        let chunk_size = read_u32_le(bytes, offset + 4).unwrap_or(0) as usize;
        if chunk_id == b"avih" && chunk_size >= 56 {
            let ah = &bytes[offset + 8..];
            if ah.len() >= 56 {
                let us_per_frame = read_u32_le(ah, 0).unwrap_or(0) as u64;
                if us_per_frame > 0 {
                    meta.frame_rate = Some(1_000_000.0 / us_per_frame as f64);
                }
                let total_frames = read_u32_le(ah, 16).unwrap_or(0) as u64;
                if us_per_frame > 0 && total_frames > 0 {
                    meta.duration_ms = Some((total_frames * us_per_frame / 1000) as i64);
                }
                let width = read_u32_le(ah, 32).unwrap_or(0);
                let height = read_u32_le(ah, 36).unwrap_or(0);
                if width > 0 {
                    meta.width = Some(width as i32);
                }
                if height > 0 {
                    meta.height = Some(height as i32);
                }
            }
        } else if chunk_id == b"strh" && chunk_size >= 56 {
            let sh = &bytes[offset + 8..];
            if sh.len() >= 8 {
                let fcc_type = &sh[0..4];
                let fcc_handler = &sh[4..8];
                let handler_str = std::str::from_utf8(fcc_handler)
                    .unwrap_or("")
                    .trim()
                    .to_lowercase();
                if fcc_type == b"vids" && meta.video_codec.is_none() {
                    meta.video_codec = Some(
                        match handler_str.as_str() {
                            "xvid" | "divx" | "dx50" | "mp4v" => "mpeg4",
                            "h264" | "avc1" => "h264",
                            "hevc" | "hev1" => "hevc",
                            "mjpg" => "mjpeg",
                            _ => handler_str.as_str(),
                        }
                        .to_string(),
                    );
                } else if fcc_type == b"auds" && meta.audio_codec.is_none() {
                    meta.audio_codec = Some("pcm".to_string());
                }
            }
        }
        offset += 8 + chunk_size;
        if chunk_size % 2 != 0 {
            offset += 1;
        }
        if offset >= bytes.len() {
            break;
        }
    }
    if let Some(dur_ms) = meta.duration_ms {
        if dur_ms > 0 {
            meta.bit_rate = Some(file_size as i64 * 8 * 1000 / dur_ms);
        }
    }
    meta
}

/// Parse image metadata for dimensions (JPEG, PNG, GIF).
fn parse_image_metadata(bytes: &[u8], format: &str) -> TechnicalMetadata {
    let mut meta = TechnicalMetadata {
        video_codec: None,
        audio_codec: None,
        duration_ms: None,
        width: None,
        height: None,
        frame_rate: None,
        bit_rate: None,
        audio_sample_rate: None,
        audio_channels: None,
        color_space: None,
        pixel_format: None,
    };
    match format {
        "jpeg" => {
            let mut i = 2usize;
            while i + 4 <= bytes.len() {
                if bytes[i] != 0xFF {
                    i += 1;
                    continue;
                }
                let marker = bytes[i + 1];
                // SOF markers contain image dimensions
                let is_sof = matches!(
                    marker,
                    0xC0 | 0xC1
                        | 0xC2
                        | 0xC3
                        | 0xC5
                        | 0xC6
                        | 0xC7
                        | 0xC9
                        | 0xCA
                        | 0xCB
                        | 0xCD
                        | 0xCE
                        | 0xCF
                );
                if is_sof {
                    if i + 9 <= bytes.len() {
                        let h = read_u16_be(bytes, i + 5).unwrap_or(0);
                        let w = read_u16_be(bytes, i + 7).unwrap_or(0);
                        if w > 0 && h > 0 {
                            meta.width = Some(i32::from(w));
                            meta.height = Some(i32::from(h));
                            meta.pixel_format = Some("yuv420p".to_string());
                        }
                    }
                    break;
                }
                // Skip this segment
                if i + 4 <= bytes.len() {
                    let seg_len = read_u16_be(bytes, i + 2).unwrap_or(2) as usize;
                    i += 2 + seg_len.max(2);
                } else {
                    break;
                }
            }
        }
        "png" => {
            // PNG: 8-byte signature + IHDR length(4) + "IHDR"(4) + width(4) + height(4)
            if bytes.len() >= 24 {
                let w = read_u32_be(bytes, 16).unwrap_or(0);
                let h = read_u32_be(bytes, 20).unwrap_or(0);
                if w > 0 && h > 0 {
                    meta.width = Some(w as i32);
                    meta.height = Some(h as i32);
                }
                if bytes.len() >= 26 {
                    let bit_depth = bytes[24];
                    let color_type = bytes[25];
                    meta.pixel_format = Some(
                        match (color_type, bit_depth) {
                            (0, 8) => "gray",
                            (2, 8) => "rgb24",
                            (2, 16) => "rgb48be",
                            (3, 8) => "pal8",
                            (4, 8) => "ya8",
                            (6, 8) => "rgba",
                            (6, 16) => "rgba64be",
                            _ => "unknown",
                        }
                        .to_string(),
                    );
                }
            }
        }
        "gif" => {
            // GIF: 6-byte header + width(2,LE) + height(2,LE)
            if bytes.len() >= 10 {
                let w = u16::from_le_bytes([bytes[6], bytes[7]]);
                let h = u16::from_le_bytes([bytes[8], bytes[9]]);
                if w > 0 && h > 0 {
                    meta.width = Some(i32::from(w));
                    meta.height = Some(i32::from(h));
                    meta.pixel_format = Some("pal8".to_string());
                }
            }
        }
        _ => {}
    }
    meta
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_mime_type() {
        assert_eq!(
            AssetManager::get_mime_type(Path::new("test.mp4")),
            Some("video/mp4".to_string())
        );
        assert_eq!(
            AssetManager::get_mime_type(Path::new("test.jpg")),
            Some("image/jpeg".to_string())
        );
        assert_eq!(AssetManager::get_mime_type(Path::new("test.unknown")), None);
    }

    #[test]
    fn test_technical_metadata_serialization() {
        let meta = TechnicalMetadata {
            duration_ms: Some(5000),
            width: Some(1920),
            height: Some(1080),
            frame_rate: Some(29.97),
            video_codec: Some("h264".to_string()),
            audio_codec: Some("aac".to_string()),
            bit_rate: Some(5_000_000),
            audio_sample_rate: Some(48000),
            audio_channels: Some(2),
            color_space: Some("bt709".to_string()),
            pixel_format: Some("yuv420p".to_string()),
        };

        let json = serde_json::to_string(&meta).expect("should succeed in test");
        let deserialized: TechnicalMetadata =
            serde_json::from_str(&json).expect("should succeed in test");

        assert_eq!(deserialized.duration_ms, Some(5000));
        assert_eq!(deserialized.width, Some(1920));
    }

    #[test]
    fn test_pagination_default() {
        let pagination = Pagination::default();
        assert_eq!(pagination.limit, 50);
        assert_eq!(pagination.offset, 0);
    }

    // -------------------------------------------------------------------------
    // Batch operation result type unit tests (no DB required)
    // -------------------------------------------------------------------------

    #[test]
    fn test_batch_update_result_is_all_success_when_no_failures() {
        let r = BatchUpdateResult {
            success_count: 5,
            failure_count: 0,
            failures: vec![],
        };
        assert!(r.is_all_success());
        assert_eq!(r.total(), 5);
    }

    #[test]
    fn test_batch_update_result_not_all_success_with_failure() {
        let r = BatchUpdateResult {
            success_count: 3,
            failure_count: 2,
            failures: vec![
                BatchUpdateFailure {
                    asset_id: Uuid::nil(),
                    reason: "not found".to_string(),
                },
                BatchUpdateFailure {
                    asset_id: Uuid::nil(),
                    reason: "permission denied".to_string(),
                },
            ],
        };
        assert!(!r.is_all_success());
        assert_eq!(r.total(), 5);
        assert_eq!(r.failures.len(), 2);
    }

    #[test]
    fn test_batch_delete_result_is_all_success() {
        let r = BatchDeleteResult {
            deleted_count: 10,
            failure_count: 0,
            failures: vec![],
        };
        assert!(r.is_all_success());
        assert_eq!(r.total(), 10);
    }

    #[test]
    fn test_batch_delete_result_partial_failure() {
        let r = BatchDeleteResult {
            deleted_count: 7,
            failure_count: 3,
            failures: vec![
                BatchDeleteFailure {
                    asset_id: Uuid::nil(),
                    reason: "locked".to_string(),
                };
                3
            ],
        };
        assert!(!r.is_all_success());
        assert_eq!(r.total(), 10);
    }

    #[test]
    fn test_batch_status_result_is_all_success() {
        let r = BatchStatusResult {
            updated_count: 4,
            failure_count: 0,
            failures: vec![],
        };
        assert!(r.is_all_success());
        assert_eq!(r.total(), 4);
    }

    #[test]
    fn test_batch_status_result_partial_failure() {
        let r = BatchStatusResult {
            updated_count: 2,
            failure_count: 1,
            failures: vec![BatchStatusFailure {
                asset_id: Uuid::nil(),
                reason: "asset not found".to_string(),
            }],
        };
        assert!(!r.is_all_success());
        assert_eq!(r.total(), 3);
        assert_eq!(r.failures[0].reason, "asset not found");
    }

    #[test]
    fn test_batch_update_failure_fields() {
        let id = Uuid::new_v4();
        let f = BatchUpdateFailure {
            asset_id: id,
            reason: "disk full".to_string(),
        };
        assert_eq!(f.asset_id, id);
        assert_eq!(f.reason, "disk full");
    }

    #[test]
    fn test_batch_delete_failure_fields() {
        let id = Uuid::new_v4();
        let f = BatchDeleteFailure {
            asset_id: id,
            reason: "already deleted".to_string(),
        };
        assert_eq!(f.asset_id, id);
        assert_eq!(f.reason, "already deleted");
    }

    #[test]
    fn test_batch_status_failure_fields() {
        let id = Uuid::new_v4();
        let f = BatchStatusFailure {
            asset_id: id,
            reason: "timeout".to_string(),
        };
        assert_eq!(f.asset_id, id);
        assert_eq!(f.reason, "timeout");
    }

    #[test]
    fn test_batch_update_result_zero_total() {
        let r = BatchUpdateResult {
            success_count: 0,
            failure_count: 0,
            failures: vec![],
        };
        assert!(r.is_all_success()); // vacuously true
        assert_eq!(r.total(), 0);
    }

    #[test]
    fn test_batch_delete_result_zero_total() {
        let r = BatchDeleteResult {
            deleted_count: 0,
            failure_count: 0,
            failures: vec![],
        };
        assert!(r.is_all_success());
        assert_eq!(r.total(), 0);
    }

    #[test]
    fn test_batch_status_result_zero_total() {
        let r = BatchStatusResult {
            updated_count: 0,
            failure_count: 0,
            failures: vec![],
        };
        assert!(r.is_all_success());
        assert_eq!(r.total(), 0);
    }
}
