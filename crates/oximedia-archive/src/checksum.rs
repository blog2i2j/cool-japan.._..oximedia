//! Checksum computation and management
//!
//! Supports multiple hashing algorithms:
//! - BLAKE3 (recommended for new archives)
//! - SHA-256 (widely supported)
//! - MD5 (legacy compatibility only)
//! - CRC32 (fast integrity checks)

use crate::{ArchiveError, ArchiveResult, ChecksumSet, VerificationConfig};
use blake3::Hasher as Blake3Hasher;
use chrono::{DateTime, Utc};
use crc32fast::Hasher as Crc32Hasher;
use md5::{Digest, Md5};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha1::Sha1;
use sha2::Sha256;
use sqlx::Row;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use tokio::fs;
use tracing::{debug, info, warn};

/// Buffer size for reading files (1 MB)
const BUFFER_SIZE: usize = 1024 * 1024;

/// Compute checksums for a file based on configuration
pub async fn compute_checksums(
    path: &Path,
    config: &VerificationConfig,
) -> ArchiveResult<ChecksumSet> {
    let file = File::open(path)?;
    let mut reader = BufReader::with_capacity(BUFFER_SIZE, file);

    let mut blake3_hasher = if config.enable_blake3 {
        Some(Blake3Hasher::new())
    } else {
        None
    };

    let mut md5_hasher = if config.enable_md5 {
        Some(Md5::new())
    } else {
        None
    };

    let mut sha256_hasher = if config.enable_sha256 {
        Some(Sha256::new())
    } else {
        None
    };

    let mut crc32_hasher = if config.enable_crc32 {
        Some(Crc32Hasher::new())
    } else {
        None
    };

    let mut buffer = vec![0u8; BUFFER_SIZE];
    let mut total_bytes = 0u64;

    loop {
        let bytes_read = reader.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }

        total_bytes += bytes_read as u64;
        let chunk = &buffer[..bytes_read];

        if let Some(ref mut hasher) = blake3_hasher {
            hasher.update(chunk);
        }

        if let Some(ref mut hasher) = md5_hasher {
            hasher.update(chunk);
        }

        if let Some(ref mut hasher) = sha256_hasher {
            hasher.update(chunk);
        }

        if let Some(ref mut hasher) = crc32_hasher {
            hasher.update(chunk);
        }
    }

    debug!(
        "Computed checksums for {} ({} bytes)",
        path.display(),
        total_bytes
    );

    Ok(ChecksumSet {
        blake3: blake3_hasher.map(|h| h.finalize().to_hex().to_string()),
        md5: md5_hasher.map(|h| hex::encode(h.finalize())),
        sha256: sha256_hasher.map(|h| hex::encode(h.finalize())),
        crc32: crc32_hasher.map(|h| format!("{:08x}", h.finalize())),
    })
}

/// Compute BLAKE3 checksum only (fastest)
pub async fn compute_blake3(path: &Path) -> ArchiveResult<String> {
    let file = File::open(path)?;
    let mut reader = BufReader::with_capacity(BUFFER_SIZE, file);
    let mut hasher = Blake3Hasher::new();
    let mut buffer = vec![0u8; BUFFER_SIZE];

    loop {
        let bytes_read = reader.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }

    Ok(hasher.finalize().to_hex().to_string())
}

/// Compute MD5 checksum (legacy)
pub async fn compute_md5(path: &Path) -> ArchiveResult<String> {
    let file = File::open(path)?;
    let mut reader = BufReader::with_capacity(BUFFER_SIZE, file);
    let mut hasher = Md5::new();
    let mut buffer = vec![0u8; BUFFER_SIZE];

    loop {
        let bytes_read = reader.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }

    Ok(hex::encode(hasher.finalize()))
}

/// Compute SHA-256 checksum
pub async fn compute_sha256(path: &Path) -> ArchiveResult<String> {
    let file = File::open(path)?;
    let mut reader = BufReader::with_capacity(BUFFER_SIZE, file);
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; BUFFER_SIZE];

    loop {
        let bytes_read = reader.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }

    Ok(hex::encode(hasher.finalize()))
}

/// Compute CRC32 checksum (fast)
pub async fn compute_crc32(path: &Path) -> ArchiveResult<String> {
    let file = File::open(path)?;
    let mut reader = BufReader::with_capacity(BUFFER_SIZE, file);
    let mut hasher = Crc32Hasher::new();
    let mut buffer = vec![0u8; BUFFER_SIZE];

    loop {
        let bytes_read = reader.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }

    Ok(format!("{:08x}", hasher.finalize()))
}

/// Incremental checksum computation for large files
pub struct IncrementalHasher {
    blake3: Option<Blake3Hasher>,
    md5: Option<Md5>,
    sha256: Option<Sha256>,
    crc32: Option<Crc32Hasher>,
    bytes_processed: u64,
}

impl IncrementalHasher {
    /// Create a new incremental hasher
    pub fn new(config: &VerificationConfig) -> Self {
        Self {
            blake3: if config.enable_blake3 {
                Some(Blake3Hasher::new())
            } else {
                None
            },
            md5: if config.enable_md5 {
                Some(Md5::new())
            } else {
                None
            },
            sha256: if config.enable_sha256 {
                Some(Sha256::new())
            } else {
                None
            },
            crc32: if config.enable_crc32 {
                Some(Crc32Hasher::new())
            } else {
                None
            },
            bytes_processed: 0,
        }
    }

    /// Update with a chunk of data
    pub fn update(&mut self, data: &[u8]) {
        if let Some(ref mut hasher) = self.blake3 {
            hasher.update(data);
        }
        if let Some(ref mut hasher) = self.md5 {
            hasher.update(data);
        }
        if let Some(ref mut hasher) = self.sha256 {
            hasher.update(data);
        }
        if let Some(ref mut hasher) = self.crc32 {
            hasher.update(data);
        }
        self.bytes_processed += data.len() as u64;
    }

    /// Get total bytes processed
    pub fn bytes_processed(&self) -> u64 {
        self.bytes_processed
    }

    /// Finalize and get checksums
    pub fn finalize(self) -> ChecksumSet {
        ChecksumSet {
            blake3: self.blake3.map(|h| h.finalize().to_hex().to_string()),
            md5: self.md5.map(|h| hex::encode(h.finalize())),
            sha256: self.sha256.map(|h| hex::encode(h.finalize())),
            crc32: self.crc32.map(|h| format!("{:08x}", h.finalize())),
        }
    }
}

/// Checksum database record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChecksumRecord {
    pub id: Option<i64>,
    pub file_path: String,
    pub file_size: i64,
    pub blake3: Option<String>,
    pub md5: Option<String>,
    pub sha256: Option<String>,
    pub crc32: Option<String>,
    pub created_at: DateTime<Utc>,
    pub last_verified_at: Option<DateTime<Utc>>,
}

impl ChecksumRecord {
    /// Create a new checksum record
    pub fn new(path: &Path, file_size: i64, checksums: ChecksumSet) -> Self {
        Self {
            id: None,
            file_path: path.to_string_lossy().to_string(),
            file_size,
            blake3: checksums.blake3,
            md5: checksums.md5,
            sha256: checksums.sha256,
            crc32: checksums.crc32,
            created_at: Utc::now(),
            last_verified_at: None,
        }
    }

    /// Save to database
    pub async fn save(&self, pool: &sqlx::SqlitePool) -> ArchiveResult<i64> {
        let created_at_str = self.created_at.to_rfc3339();
        let last_verified_str = self.last_verified_at.map(|dt| dt.to_rfc3339());

        let result = sqlx::query(
            r"
            INSERT INTO checksums (file_path, file_size, blake3, md5, sha256, crc32, created_at, last_verified_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ",
        )
        .bind(&self.file_path)
        .bind(self.file_size)
        .bind(&self.blake3)
        .bind(&self.md5)
        .bind(&self.sha256)
        .bind(&self.crc32)
        .bind(&created_at_str)
        .bind(&last_verified_str)
        .execute(pool)
        .await?;

        Ok(result.last_insert_rowid())
    }

    /// Load from database by file path
    pub async fn load(pool: &sqlx::SqlitePool, file_path: &str) -> ArchiveResult<Option<Self>> {
        let row = sqlx::query(
            r"
            SELECT id, file_path, file_size, blake3, md5, sha256, crc32, created_at, last_verified_at
            FROM checksums
            WHERE file_path = ?
            ORDER BY created_at DESC
            LIMIT 1
            ",
        )
        .bind(file_path)
        .fetch_optional(pool)
        .await?;

        if let Some(row) = row {
            let created_at: String = row.get("created_at");
            let last_verified_at: Option<String> = row.get("last_verified_at");

            Ok(Some(Self {
                id: Some(row.get("id")),
                file_path: row.get("file_path"),
                file_size: row.get("file_size"),
                blake3: row.get("blake3"),
                md5: row.get("md5"),
                sha256: row.get("sha256"),
                crc32: row.get("crc32"),
                created_at: DateTime::parse_from_rfc3339(&created_at)
                    .map_err(|e| ArchiveError::Database(sqlx::Error::Decode(Box::new(e))))?
                    .with_timezone(&Utc),
                last_verified_at: last_verified_at
                    .map(|s| DateTime::parse_from_rfc3339(&s).map(|dt| dt.with_timezone(&Utc)))
                    .transpose()
                    .map_err(|e| ArchiveError::Database(sqlx::Error::Decode(Box::new(e))))?,
            }))
        } else {
            Ok(None)
        }
    }

    /// Update last verified timestamp
    pub async fn update_verified(&mut self, pool: &sqlx::SqlitePool) -> ArchiveResult<()> {
        let now = Utc::now();
        self.last_verified_at = Some(now);
        let last_verified_str = now.to_rfc3339();

        sqlx::query(
            r"
            UPDATE checksums
            SET last_verified_at = ?
            WHERE id = ?
            ",
        )
        .bind(&last_verified_str)
        .bind(self.id)
        .execute(pool)
        .await?;

        Ok(())
    }
}

/// Generate sidecar checksum files
pub async fn generate_sidecar_files(
    file_path: &Path,
    checksums: &ChecksumSet,
) -> ArchiveResult<()> {
    info!("Generating sidecar files for {}", file_path.display());

    // Generate .md5 file
    if let Some(ref md5) = checksums.md5 {
        let md5_path = file_path.with_extension("md5");
        let filename = file_path
            .file_name()
            .ok_or_else(|| ArchiveError::Validation("Invalid filename".to_string()))?
            .to_string_lossy();
        let content = format!("{md5}  {filename}\n");
        fs::write(&md5_path, content).await?;
        debug!("Generated {}", md5_path.display());
    }

    // Generate .sha256 file
    if let Some(ref sha256) = checksums.sha256 {
        let sha256_path = file_path.with_extension("sha256");
        let filename = file_path
            .file_name()
            .ok_or_else(|| ArchiveError::Validation("Invalid filename".to_string()))?
            .to_string_lossy();
        let content = format!("{sha256}  {filename}\n");
        fs::write(&sha256_path, content).await?;
        debug!("Generated {}", sha256_path.display());
    }

    // Generate .blake3 file
    if let Some(ref blake3) = checksums.blake3 {
        let blake3_path = file_path.with_extension("blake3");
        let filename = file_path
            .file_name()
            .ok_or_else(|| ArchiveError::Validation("Invalid filename".to_string()))?
            .to_string_lossy();
        let content = format!("{blake3}  {filename}\n");
        fs::write(&blake3_path, content).await?;
        debug!("Generated {}", blake3_path.display());
    }

    // Generate .crc32 file
    if let Some(ref crc32) = checksums.crc32 {
        let crc32_path = file_path.with_extension("crc32");
        let filename = file_path
            .file_name()
            .ok_or_else(|| ArchiveError::Validation("Invalid filename".to_string()))?
            .to_string_lossy();
        let content = format!("{crc32}  {filename}\n");
        fs::write(&crc32_path, content).await?;
        debug!("Generated {}", crc32_path.display());
    }

    Ok(())
}

/// Verify checksums against sidecar files
pub async fn verify_sidecar_files(
    file_path: &Path,
    computed_checksums: &ChecksumSet,
) -> ArchiveResult<SidecarVerificationResult> {
    let mut result = SidecarVerificationResult {
        file_path: file_path.to_path_buf(),
        md5_verified: None,
        sha256_verified: None,
        blake3_verified: None,
        crc32_verified: None,
    };

    // Verify .md5 file
    if let Some(ref computed_md5) = computed_checksums.md5 {
        let md5_path = file_path.with_extension("md5");
        if md5_path.exists() {
            let content = fs::read_to_string(&md5_path).await?;
            let expected_md5 = parse_checksum_from_sidecar(&content)?;
            result.md5_verified = Some(computed_md5 == &expected_md5);
        }
    }

    // Verify .sha256 file
    if let Some(ref computed_sha256) = computed_checksums.sha256 {
        let sha256_path = file_path.with_extension("sha256");
        if sha256_path.exists() {
            let content = fs::read_to_string(&sha256_path).await?;
            let expected_sha256 = parse_checksum_from_sidecar(&content)?;
            result.sha256_verified = Some(computed_sha256 == &expected_sha256);
        }
    }

    // Verify .blake3 file
    if let Some(ref computed_blake3) = computed_checksums.blake3 {
        let blake3_path = file_path.with_extension("blake3");
        if blake3_path.exists() {
            let content = fs::read_to_string(&blake3_path).await?;
            let expected_blake3 = parse_checksum_from_sidecar(&content)?;
            result.blake3_verified = Some(computed_blake3 == &expected_blake3);
        }
    }

    // Verify .crc32 file
    if let Some(ref computed_crc32) = computed_checksums.crc32 {
        let crc32_path = file_path.with_extension("crc32");
        if crc32_path.exists() {
            let content = fs::read_to_string(&crc32_path).await?;
            let expected_crc32 = parse_checksum_from_sidecar(&content)?;
            result.crc32_verified = Some(computed_crc32 == &expected_crc32);
        }
    }

    Ok(result)
}

/// Parse checksum from sidecar file content
fn parse_checksum_from_sidecar(content: &str) -> ArchiveResult<String> {
    let parts: Vec<&str> = content.split_whitespace().collect();
    if parts.is_empty() {
        return Err(ArchiveError::Validation(
            "Invalid sidecar file format".to_string(),
        ));
    }
    Ok(parts[0].to_string())
}

/// Sidecar verification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SidecarVerificationResult {
    pub file_path: PathBuf,
    pub md5_verified: Option<bool>,
    pub sha256_verified: Option<bool>,
    pub blake3_verified: Option<bool>,
    pub crc32_verified: Option<bool>,
}

impl SidecarVerificationResult {
    /// Check if all available sidecars passed verification
    pub fn all_passed(&self) -> bool {
        let checks = [
            self.md5_verified,
            self.sha256_verified,
            self.blake3_verified,
            self.crc32_verified,
        ];

        // If any check failed, return false
        // If no checks were performed, return true (no sidecars to verify)
        checks.iter().all(|check| check.unwrap_or(true))
    }

    /// Check if any sidecar verification failed
    pub fn any_failed(&self) -> bool {
        !self.all_passed()
    }
}

/// Compute checksums for multiple files in parallel
pub async fn compute_checksums_parallel(
    paths: &[PathBuf],
    config: &VerificationConfig,
) -> ArchiveResult<Vec<(PathBuf, ChecksumSet)>> {
    let config_clone = config.clone();

    // Use rayon for parallel processing
    let results: Vec<_> = paths
        .par_iter()
        .map(|path| {
            let rt = tokio::runtime::Handle::current();
            let config = config_clone.clone();
            let path = path.clone();

            rt.block_on(async move {
                let checksums = compute_checksums(&path, &config).await?;
                Ok::<_, ArchiveError>((path, checksums))
            })
        })
        .collect();

    // Convert to regular Result
    results.into_iter().collect()
}

/// Batch checksum verification
pub struct BatchVerifier {
    config: VerificationConfig,
    pool: sqlx::SqlitePool,
}

impl BatchVerifier {
    /// Create a new batch verifier
    pub fn new(config: VerificationConfig, pool: sqlx::SqlitePool) -> Self {
        Self { config, pool }
    }

    /// Verify a batch of files
    pub async fn verify_batch(&self, paths: &[PathBuf]) -> ArchiveResult<BatchVerificationResult> {
        let mut result = BatchVerificationResult {
            total_files: paths.len(),
            verified_files: 0,
            failed_files: 0,
            errors: Vec::new(),
        };

        for path in paths {
            match self.verify_single_file(path).await {
                Ok(true) => result.verified_files += 1,
                Ok(false) => result.failed_files += 1,
                Err(e) => {
                    result.failed_files += 1;
                    result.errors.push((path.clone(), e.to_string()));
                }
            }
        }

        Ok(result)
    }

    /// Verify a single file
    async fn verify_single_file(&self, path: &Path) -> ArchiveResult<bool> {
        // Compute current checksums
        let checksums = compute_checksums(path, &self.config).await?;

        // Load stored checksums
        let stored = ChecksumRecord::load(&self.pool, &path.to_string_lossy()).await?;

        if let Some(stored) = stored {
            // Compare checksums
            let mut all_match = true;

            if let (Some(ref computed), Some(ref stored_val)) = (&checksums.blake3, &stored.blake3)
            {
                if computed != stored_val {
                    warn!(
                        "BLAKE3 mismatch for {}: expected {}, got {}",
                        path.display(),
                        stored_val,
                        computed
                    );
                    all_match = false;
                }
            }

            if let (Some(ref computed), Some(ref stored_val)) = (&checksums.md5, &stored.md5) {
                if computed != stored_val {
                    warn!(
                        "MD5 mismatch for {}: expected {}, got {}",
                        path.display(),
                        stored_val,
                        computed
                    );
                    all_match = false;
                }
            }

            if let (Some(ref computed), Some(ref stored_val)) = (&checksums.sha256, &stored.sha256)
            {
                if computed != stored_val {
                    warn!(
                        "SHA-256 mismatch for {}: expected {}, got {}",
                        path.display(),
                        stored_val,
                        computed
                    );
                    all_match = false;
                }
            }

            if let (Some(ref computed), Some(ref stored_val)) = (&checksums.crc32, &stored.crc32) {
                if computed != stored_val {
                    warn!(
                        "CRC32 mismatch for {}: expected {}, got {}",
                        path.display(),
                        stored_val,
                        computed
                    );
                    all_match = false;
                }
            }

            Ok(all_match)
        } else {
            // No stored checksums, create new record
            let file_size = std::fs::metadata(path)?.len() as i64;
            let record = ChecksumRecord::new(path, file_size, checksums);
            record.save(&self.pool).await?;
            Ok(true)
        }
    }
}

/// Batch verification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchVerificationResult {
    pub total_files: usize,
    pub verified_files: usize,
    pub failed_files: usize,
    pub errors: Vec<(PathBuf, String)>,
}

impl BatchVerificationResult {
    /// Check if all files verified successfully
    pub fn all_verified(&self) -> bool {
        self.failed_files == 0 && self.errors.is_empty()
    }

    /// Get success rate
    pub fn success_rate(&self) -> f64 {
        if self.total_files == 0 {
            return 0.0;
        }
        (self.verified_files as f64) / (self.total_files as f64)
    }
}

// ── New checksum types ────────────────────────────────────────────────────────

/// Hash algorithm selector.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChecksumAlgo {
    /// MD5 — legacy 128-bit hash.
    Md5,
    /// SHA-1 — 160-bit hash.
    Sha1,
    /// SHA-256 — 256-bit hash.
    Sha256,
    /// xxHash — fast non-cryptographic 64-bit hash.
    Xxhash,
}

impl ChecksumAlgo {
    /// Digest length in bytes.
    #[must_use]
    pub const fn digest_len_bytes(self) -> usize {
        match self {
            Self::Md5 => 16,
            Self::Sha1 => 20,
            Self::Sha256 => 32,
            Self::Xxhash => 8,
        }
    }

    /// Human-readable algorithm name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Md5 => "md5",
            Self::Sha1 => "sha1",
            Self::Sha256 => "sha256",
            Self::Xxhash => "xxhash",
        }
    }
}

/// A single checksum record for one file.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct ChecksumEntry {
    /// Path of the file being checksummed.
    pub path: String,
    /// Algorithm used.
    pub algo: ChecksumAlgo,
    /// Hex-encoded digest.
    pub hex_digest: String,
    /// File size in bytes at the time of hashing.
    pub size_bytes: u64,
}

impl ChecksumEntry {
    /// Returns `true` if `hex_digest` is a valid lowercase hex string of the
    /// expected length for the algorithm.
    #[must_use]
    pub fn is_valid_hex(&self) -> bool {
        let expected_len = self.algo.digest_len_bytes() * 2;
        self.hex_digest.len() == expected_len
            && self.hex_digest.chars().all(|c| c.is_ascii_hexdigit())
    }
}

/// A manifest that collects multiple [`ChecksumEntry`] records.
#[allow(dead_code)]
#[derive(Default, Debug)]
pub struct ChecksumManifest {
    /// All checksum records.
    pub records: Vec<ChecksumEntry>,
    /// Unix epoch seconds when this manifest was created.
    pub created_at_epoch: u64,
}

impl ChecksumManifest {
    /// Create an empty manifest.
    #[must_use]
    pub fn new(created_at_epoch: u64) -> Self {
        Self {
            records: Vec::new(),
            created_at_epoch,
        }
    }

    /// Add a record to the manifest.
    pub fn add(&mut self, record: ChecksumEntry) {
        self.records.push(record);
    }

    /// Find the first record with the given path.
    #[must_use]
    pub fn find(&self, path: &str) -> Option<&ChecksumEntry> {
        self.records.iter().find(|r| r.path == path)
    }

    /// Total number of records.
    #[must_use]
    pub fn entry_count(&self) -> usize {
        self.records.len()
    }
}

/// Verify a checksum record against a byte slice.
///
/// Dispatches on [`ChecksumAlgo`] to compute the real cryptographic digest and
/// compares it (case-insensitively) against `record.hex_digest`.
#[allow(dead_code)]
#[must_use]
pub fn verify_checksum(record: &ChecksumEntry, data: &[u8]) -> bool {
    let computed_hex: String = match record.algo {
        ChecksumAlgo::Md5 => {
            let mut h = Md5::new();
            h.update(data);
            hex::encode(h.finalize())
        }
        ChecksumAlgo::Sha1 => {
            let mut h = Sha1::new();
            h.update(data);
            hex::encode(h.finalize())
        }
        ChecksumAlgo::Sha256 => {
            let mut h = Sha256::new();
            h.update(data);
            hex::encode(h.finalize())
        }
        ChecksumAlgo::Xxhash => {
            let hash_value = xxhash_rust::xxh64::xxh64(data, 0);
            format!("{hash_value:016x}")
        }
    };
    computed_hex.eq_ignore_ascii_case(&record.hex_digest)
}

#[cfg(test)]
mod checksum_algo_tests {
    use super::*;

    fn make_record(path: &str, algo: ChecksumAlgo, hex: &str) -> ChecksumEntry {
        ChecksumEntry {
            path: path.to_string(),
            algo,
            hex_digest: hex.to_string(),
            size_bytes: 0,
        }
    }

    #[test]
    fn test_algo_digest_len_md5() {
        assert_eq!(ChecksumAlgo::Md5.digest_len_bytes(), 16);
    }

    #[test]
    fn test_algo_digest_len_sha1() {
        assert_eq!(ChecksumAlgo::Sha1.digest_len_bytes(), 20);
    }

    #[test]
    fn test_algo_digest_len_sha256() {
        assert_eq!(ChecksumAlgo::Sha256.digest_len_bytes(), 32);
    }

    #[test]
    fn test_algo_digest_len_xxhash() {
        assert_eq!(ChecksumAlgo::Xxhash.digest_len_bytes(), 8);
    }

    #[test]
    fn test_algo_name_md5() {
        assert_eq!(ChecksumAlgo::Md5.name(), "md5");
    }

    #[test]
    fn test_algo_name_sha256() {
        assert_eq!(ChecksumAlgo::Sha256.name(), "sha256");
    }

    #[test]
    fn test_checksum_entry_valid_hex_md5() {
        let r = make_record("a.bin", ChecksumAlgo::Md5, &"ab".repeat(16));
        assert!(r.is_valid_hex());
    }

    #[test]
    fn test_checksum_entry_invalid_hex_wrong_len() {
        let r = make_record("a.bin", ChecksumAlgo::Md5, "abcd");
        assert!(!r.is_valid_hex());
    }

    #[test]
    fn test_checksum_entry_invalid_hex_bad_chars() {
        let r = make_record("a.bin", ChecksumAlgo::Md5, &"zz".repeat(16));
        assert!(!r.is_valid_hex());
    }

    #[test]
    fn test_manifest_add_and_count() {
        let mut m = ChecksumManifest::new(1_000_000);
        m.add(make_record(
            "f1.bin",
            ChecksumAlgo::Sha256,
            &"aa".repeat(32),
        ));
        m.add(make_record("f2.bin", ChecksumAlgo::Md5, &"bb".repeat(16)));
        assert_eq!(m.entry_count(), 2);
    }

    #[test]
    fn test_manifest_find_existing() {
        let mut m = ChecksumManifest::new(0);
        m.add(make_record(
            "target.bin",
            ChecksumAlgo::Sha1,
            &"cc".repeat(20),
        ));
        assert!(m.find("target.bin").is_some());
    }

    #[test]
    fn test_manifest_find_missing() {
        let m = ChecksumManifest::new(0);
        assert!(m.find("nope.bin").is_none());
    }

    #[test]
    fn test_manifest_created_at() {
        let m = ChecksumManifest::new(42);
        assert_eq!(m.created_at_epoch, 42);
    }

    #[test]
    fn test_verify_checksum_roundtrip_md5() {
        use md5::{Digest, Md5};
        let data = b"hello world";
        let mut h = Md5::new();
        h.update(data);
        let hex = hex::encode(h.finalize());
        let r = make_record("test.bin", ChecksumAlgo::Md5, &hex);
        assert!(verify_checksum(&r, data));
    }

    #[test]
    fn test_verify_checksum_roundtrip_sha256() {
        use sha2::{Digest, Sha256};
        let data = b"hello world";
        let mut h = Sha256::new();
        h.update(data);
        let hex = hex::encode(h.finalize());
        let r = make_record("test.bin", ChecksumAlgo::Sha256, &hex);
        assert!(verify_checksum(&r, data));
    }

    #[test]
    fn test_verify_checksum_roundtrip_sha1() {
        use sha1::{Digest, Sha1};
        let data = b"hello world";
        let mut h = Sha1::new();
        h.update(data);
        let hex = hex::encode(h.finalize());
        let r = make_record("test.bin", ChecksumAlgo::Sha1, &hex);
        assert!(verify_checksum(&r, data));
    }

    #[test]
    fn test_verify_checksum_roundtrip_xxhash() {
        let data = b"hello world";
        let hash_value = xxhash_rust::xxh64::xxh64(data, 0);
        let hex = format!("{hash_value:016x}");
        let r = make_record("test.bin", ChecksumAlgo::Xxhash, &hex);
        assert!(verify_checksum(&r, data));
    }

    #[test]
    fn test_verify_checksum_mismatch() {
        // All-zero MD5 digest will not match real MD5 of non-empty data.
        let r = make_record("x.bin", ChecksumAlgo::Md5, &"00".repeat(16));
        assert!(!verify_checksum(&r, b"some data that is not all zeros"));
    }
}
