//! Resumable multipart upload that survives process restarts.
//!
//! Standard multipart upload sessions are ephemeral: if the uploading process
//! crashes, the partial upload is lost and must restart from scratch.  This
//! module implements a crash-safe resumable upload by persisting upload state
//! to a JSON checkpoint file.  On restart, callers can detect in-progress
//! uploads and resume from the last successfully uploaded part.
//!
//! # Architecture
//!
//! ```text
//! ResumableUpload
//!   ├── checkpoint file (JSON, written after each part)
//!   ├── UploadSession    — provider upload ID + target object metadata
//!   └── PartRecord[]     — part number, ETag, byte range per uploaded part
//! ```
//!
//! # Workflow
//!
//! ```rust,no_run
//! use oximedia_storage::multipart_resumable::{ResumableUpload, UploadSession};
//!
//! # fn main() -> std::io::Result<()> {
//! let session = UploadSession::new(
//!     "my-bucket", "videos/long.mp4", "video/mp4",
//!     std::path::Path::new("/tmp/upload_state.json"),
//! );
//! let mut upload = ResumableUpload::new_or_resume(session)?;
//!
//! // Upload parts
//! let data = vec![0u8; 5 * 1024 * 1024]; // 5 MiB part
//! upload.record_part(1, "etag-001".to_string(), 0, data.len() as u64)?;
//!
//! // On crash / restart:
//! // let mut upload = ResumableUpload::new_or_resume(session)?;
//! // upload.last_completed_part() tells you where to resume.
//! # Ok(())
//! # }
//! ```

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::io;
use std::path::{Path, PathBuf};

// ─── PartRecord ───────────────────────────────────────────────────────────────

/// Record of a single successfully uploaded part.
#[derive(Debug, Clone)]
pub struct PartRecord {
    /// 1-indexed part number (as required by most providers).
    pub part_number: u32,
    /// ETag returned by the storage provider after the part was uploaded.
    pub etag: String,
    /// Byte offset of the start of this part within the source object.
    pub offset_bytes: u64,
    /// Size of this part in bytes.
    pub size_bytes: u64,
}

impl PartRecord {
    /// Create a new part record.
    pub fn new(
        part_number: u32,
        etag: impl Into<String>,
        offset_bytes: u64,
        size_bytes: u64,
    ) -> Self {
        Self {
            part_number,
            etag: etag.into(),
            offset_bytes,
            size_bytes,
        }
    }
}

// ─── UploadSession ────────────────────────────────────────────────────────────

/// Describes the target object and checkpoint file location.
#[derive(Debug, Clone)]
pub struct UploadSession {
    /// Destination bucket.
    pub bucket: String,
    /// Destination object key.
    pub key: String,
    /// Content type of the object.
    pub content_type: String,
    /// Path where the checkpoint JSON file is stored.
    pub checkpoint_path: PathBuf,
    /// Provider-assigned multipart upload ID (set after initiation).
    pub upload_id: Option<String>,
}

impl UploadSession {
    /// Create a new upload session descriptor.
    pub fn new(
        bucket: impl Into<String>,
        key: impl Into<String>,
        content_type: impl Into<String>,
        checkpoint_path: &Path,
    ) -> Self {
        Self {
            bucket: bucket.into(),
            key: key.into(),
            content_type: content_type.into(),
            checkpoint_path: checkpoint_path.to_path_buf(),
            upload_id: None,
        }
    }

    /// Attach a provider upload ID to the session.
    pub fn with_upload_id(mut self, upload_id: impl Into<String>) -> Self {
        self.upload_id = Some(upload_id.into());
        self
    }
}

// ─── CheckpointData ───────────────────────────────────────────────────────────

/// Serialisable checkpoint state persisted to disk.
#[derive(Debug, Clone, Default)]
struct CheckpointData {
    bucket: String,
    key: String,
    upload_id: Option<String>,
    content_type: String,
    completed_parts: Vec<(u32, String, u64, u64)>, // (part_number, etag, offset, size)
    total_size_bytes: Option<u64>,
    completed: bool,
}

impl CheckpointData {
    fn to_json(&self) -> String {
        let parts_json: Vec<String> = self
            .completed_parts
            .iter()
            .map(|(n, etag, off, sz)| {
                format!(
                    "{{\"part_number\":{n},\"etag\":\"{}\",\"offset\":{off},\"size\":{sz}}}",
                    etag.replace('"', "\\\"")
                )
            })
            .collect();
        let upload_id_json = match &self.upload_id {
            Some(id) => format!("\"{}\"", id.replace('"', "\\\"")),
            None => "null".to_string(),
        };
        let total_json = match self.total_size_bytes {
            Some(n) => n.to_string(),
            None => "null".to_string(),
        };
        format!(
            "{{\"bucket\":\"{}\",\"key\":\"{}\",\"upload_id\":{upload_id_json},\"content_type\":\"{}\",\"total_size_bytes\":{total_json},\"completed\":{},\"parts\":[{}]}}",
            self.bucket.replace('"', "\\\""),
            self.key.replace('"', "\\\""),
            self.content_type.replace('"', "\\\""),
            self.completed,
            parts_json.join(","),
        )
    }

    /// Minimal JSON parser (no external dep).  Extracts only the fields we need.
    fn from_json(json: &str) -> Option<Self> {
        // Use simple substring extraction — sufficient for our controlled format.
        let bucket = extract_string_field(json, "bucket")?;
        let key = extract_string_field(json, "key")?;
        let content_type = extract_string_field(json, "content_type").unwrap_or_default();
        let upload_id = extract_optional_string_field(json, "upload_id");
        let completed = json.contains("\"completed\":true");
        let total_size_bytes = extract_u64_field(json, "total_size_bytes");

        // Parse parts array.
        let mut completed_parts = Vec::new();
        if let Some(parts_start) = json.find("\"parts\":[") {
            let rest = &json[parts_start + 9..];
            let mut pos = 0;
            while pos < rest.len() {
                let Some(obj_start) = rest[pos..].find('{') else {
                    break;
                };
                let abs_start = pos + obj_start;
                let Some(obj_end) = rest[abs_start..].find('}') else {
                    break;
                };
                let obj = &rest[abs_start..abs_start + obj_end + 1];
                if let (Some(pn), Some(etag)) = (
                    extract_u32_field(obj, "part_number"),
                    extract_string_field(obj, "etag"),
                ) {
                    let offset = extract_u64_field(obj, "offset").unwrap_or(0);
                    let size = extract_u64_field(obj, "size").unwrap_or(0);
                    completed_parts.push((pn, etag, offset, size));
                }
                pos = abs_start + obj_end + 1;
            }
        }

        Some(Self {
            bucket,
            key,
            upload_id,
            content_type,
            completed_parts,
            total_size_bytes,
            completed,
        })
    }
}

// Simple field extraction helpers (no external JSON dep).

fn extract_string_field(json: &str, field: &str) -> Option<String> {
    let needle = format!("\"{}\":\"", field);
    let start = json.find(&needle)? + needle.len();
    let rest = &json[start..];
    // Find closing quote, handling escaped quotes.
    let mut end = 0;
    let bytes = rest.as_bytes();
    while end < bytes.len() {
        if bytes[end] == b'"' && (end == 0 || bytes[end - 1] != b'\\') {
            break;
        }
        end += 1;
    }
    Some(rest[..end].replace("\\\"", "\""))
}

fn extract_optional_string_field(json: &str, field: &str) -> Option<String> {
    let null_needle = format!("\"{}\":null", field);
    if json.contains(&null_needle) {
        return None;
    }
    extract_string_field(json, field)
}

fn extract_u64_field(json: &str, field: &str) -> Option<u64> {
    let needle = format!("\"{}\":", field);
    let start = json.find(&needle)? + needle.len();
    let rest = &json[start..];
    // Skip until digit or 'n' (null)
    let num_start = rest.find(|c: char| c.is_ascii_digit())?;
    let num_str: String = rest[num_start..]
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    num_str.parse().ok()
}

fn extract_u32_field(json: &str, field: &str) -> Option<u32> {
    extract_u64_field(json, field).and_then(|v| u32::try_from(v).ok())
}

// ─── ResumableUpload ─────────────────────────────────────────────────────────

/// Manages a crash-safe resumable multipart upload.
pub struct ResumableUpload {
    session: UploadSession,
    checkpoint: CheckpointData,
}

impl ResumableUpload {
    /// Create a new upload or resume an in-progress one from the checkpoint file.
    ///
    /// If the checkpoint file exists and is parseable, the previous state is
    /// loaded.  Otherwise a fresh upload state is initialised.
    ///
    /// # Errors
    ///
    /// Returns an [`io::Error`] only if reading an existing checkpoint file
    /// fails for a reason other than `NotFound`.
    pub fn new_or_resume(session: UploadSession) -> io::Result<Self> {
        let checkpoint = match std::fs::read_to_string(&session.checkpoint_path) {
            Ok(json) => CheckpointData::from_json(&json).unwrap_or_else(|| CheckpointData {
                bucket: session.bucket.clone(),
                key: session.key.clone(),
                upload_id: session.upload_id.clone(),
                content_type: session.content_type.clone(),
                ..CheckpointData::default()
            }),
            Err(e) if e.kind() == io::ErrorKind::NotFound => CheckpointData {
                bucket: session.bucket.clone(),
                key: session.key.clone(),
                upload_id: session.upload_id.clone(),
                content_type: session.content_type.clone(),
                ..CheckpointData::default()
            },
            Err(e) => return Err(e),
        };
        Ok(Self {
            session,
            checkpoint,
        })
    }

    /// Set the provider-assigned upload ID (obtained after initiating the upload).
    pub fn set_upload_id(&mut self, upload_id: impl Into<String>) {
        self.checkpoint.upload_id = Some(upload_id.into());
        self.session.upload_id = self.checkpoint.upload_id.clone();
    }

    /// Set the total expected size of the source object in bytes.
    pub fn set_total_size(&mut self, size_bytes: u64) {
        self.checkpoint.total_size_bytes = Some(size_bytes);
    }

    /// Record a successfully uploaded part and persist the checkpoint.
    ///
    /// # Errors
    ///
    /// Returns an [`io::Error`] if the checkpoint file cannot be written.
    pub fn record_part(
        &mut self,
        part_number: u32,
        etag: String,
        offset_bytes: u64,
        size_bytes: u64,
    ) -> io::Result<()> {
        // Remove any existing record for this part number (idempotent).
        self.checkpoint
            .completed_parts
            .retain(|(n, _, _, _)| *n != part_number);
        self.checkpoint
            .completed_parts
            .push((part_number, etag, offset_bytes, size_bytes));
        // Sort by part number for deterministic checkpoint output.
        self.checkpoint
            .completed_parts
            .sort_by_key(|(n, _, _, _)| *n);
        self.persist_checkpoint()
    }

    /// Mark the upload as fully completed and clean up the checkpoint file.
    ///
    /// # Errors
    ///
    /// Returns an [`io::Error`] if the checkpoint file cannot be removed.
    pub fn complete(&mut self) -> io::Result<()> {
        self.checkpoint.completed = true;
        // Remove the checkpoint file — no longer needed.
        match std::fs::remove_file(&self.session.checkpoint_path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e),
        }
    }

    /// Returns the last completed part number, or 0 if no parts have been uploaded.
    pub fn last_completed_part(&self) -> u32 {
        self.checkpoint
            .completed_parts
            .last()
            .map(|(n, _, _, _)| *n)
            .unwrap_or(0)
    }

    /// Returns the byte offset where the next part should start.
    pub fn next_offset_bytes(&self) -> u64 {
        self.checkpoint
            .completed_parts
            .last()
            .map(|(_, _, off, sz)| off + sz)
            .unwrap_or(0)
    }

    /// Returns the number of completed parts.
    pub fn completed_part_count(&self) -> usize {
        self.checkpoint.completed_parts.len()
    }

    /// Returns the completed parts as [`PartRecord`]s (sorted by part number).
    pub fn completed_parts(&self) -> Vec<PartRecord> {
        self.checkpoint
            .completed_parts
            .iter()
            .map(|(n, etag, off, sz)| PartRecord {
                part_number: *n,
                etag: etag.clone(),
                offset_bytes: *off,
                size_bytes: *sz,
            })
            .collect()
    }

    /// Provider-assigned upload ID (if known).
    pub fn upload_id(&self) -> Option<&str> {
        self.checkpoint.upload_id.as_deref()
    }

    /// Returns `true` if the upload has been marked as completed.
    pub fn is_completed(&self) -> bool {
        self.checkpoint.completed
    }

    /// Session reference.
    pub fn session(&self) -> &UploadSession {
        &self.session
    }

    // ── Private ───────────────────────────────────────────────────────────────

    fn persist_checkpoint(&self) -> io::Result<()> {
        if let Some(parent) = self.session.checkpoint_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = self.checkpoint.to_json();
        std::fs::write(&self.session.checkpoint_path, json)
    }
}

// ─── MultipartConfig ──────────────────────────────────────────────────────────

/// Configuration for parallel multipart uploads.
///
/// Controls how large each chunk is and how many parts are uploaded
/// concurrently using [`tokio::task::JoinSet`].
#[derive(Debug, Clone)]
pub struct MultipartConfig {
    /// Size of each chunk in bytes (default: 8 MiB).
    pub chunk_size_bytes: usize,
    /// Maximum number of parts uploaded concurrently (default: 4).
    pub max_concurrent_parts: usize,
}

impl Default for MultipartConfig {
    fn default() -> Self {
        Self {
            chunk_size_bytes: 8 * 1024 * 1024, // 8 MiB
            max_concurrent_parts: 4,
        }
    }
}

/// A completed part descriptor returned by a parallel upload task.
#[derive(Debug, Clone)]
pub struct PartResult {
    /// 1-indexed part number.
    pub part_number: u32,
    /// ETag or equivalent identifier returned by the storage backend.
    pub etag: String,
    /// Byte offset of this part within the original payload.
    pub offset_bytes: u64,
    /// Actual number of bytes in this part.
    pub size_bytes: u64,
}

/// Split `data` into chunks according to `config` and "upload" all chunks
/// concurrently, returning a sorted list of [`PartResult`]s.
///
/// In this implementation the "upload" is simulated: each chunk is hashed
/// (SHA-256 prefix) to produce a deterministic ETag.  In production the
/// closure would call the actual storage backend.
///
/// # Concurrency
///
/// The function maintains at most `config.max_concurrent_parts` in-flight
/// tokio tasks at any one time using [`tokio::task::JoinSet`].
pub async fn upload_parallel(data: &[u8], config: &MultipartConfig) -> io::Result<Vec<PartResult>> {
    use std::sync::Arc;
    use tokio::task::JoinSet;

    let chunk_size = config.chunk_size_bytes.max(1);
    let num_parts = data.len().div_ceil(chunk_size);

    let data_arc = Arc::new(data.to_vec());
    let mut join_set: JoinSet<io::Result<PartResult>> = JoinSet::new();
    let mut results: Vec<PartResult> = Vec::with_capacity(num_parts);
    let mut next_part = 0usize;

    loop {
        // Spawn up to `max_concurrent_parts` tasks.
        while join_set.len() < config.max_concurrent_parts && next_part < num_parts {
            let part_num = (next_part + 1) as u32;
            let offset = next_part * chunk_size;
            let end = (offset + chunk_size).min(data.len());
            let chunk_len = end - offset;
            let data_ref = data_arc.clone();

            join_set.spawn(async move {
                // Simulate the upload by computing a deterministic ETag from
                // the chunk content using a simple FNV-1a hash.
                let chunk = &data_ref[offset..end];
                let mut h: u64 = 0xcbf2_9ce4_8422_2325;
                for byte in chunk {
                    h ^= *byte as u64;
                    h = h.wrapping_mul(0x0000_0100_0000_01b3);
                }
                let etag = format!("{h:016x}");
                Ok(PartResult {
                    part_number: part_num,
                    etag,
                    offset_bytes: offset as u64,
                    size_bytes: chunk_len as u64,
                })
            });

            next_part += 1;
        }

        if join_set.is_empty() {
            break;
        }

        // Collect the next finished part.
        match join_set.join_next().await {
            Some(Ok(Ok(part))) => results.push(part),
            Some(Ok(Err(e))) => return Err(e),
            Some(Err(join_err)) => {
                return Err(io::Error::other(join_err.to_string()));
            }
            None => break,
        }
    }

    // Stable sort by part number so callers receive an ordered list.
    results.sort_by_key(|r| r.part_number);
    Ok(results)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_checkpoint(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!("oximedia_resumable_{tag}.json"))
    }

    #[test]
    fn test_new_upload_starts_fresh() {
        let cp = tmp_checkpoint("fresh");
        let _ = std::fs::remove_file(&cp);
        let session = UploadSession::new("bucket", "key", "video/mp4", &cp);
        let upload = ResumableUpload::new_or_resume(session).expect("create");
        assert_eq!(upload.completed_part_count(), 0);
        assert_eq!(upload.last_completed_part(), 0);
        assert_eq!(upload.next_offset_bytes(), 0);
    }

    #[test]
    fn test_record_part_and_resume() {
        let cp = tmp_checkpoint("resume");
        let _ = std::fs::remove_file(&cp);

        let session = UploadSession::new("bucket", "obj", "video/mp4", &cp);
        let mut upload = ResumableUpload::new_or_resume(session).expect("create");
        upload.set_upload_id("upload-xyz");
        upload
            .record_part(1, "etag-1".to_string(), 0, 5_242_880)
            .expect("record");
        upload
            .record_part(2, "etag-2".to_string(), 5_242_880, 5_242_880)
            .expect("record");

        // Simulate restart
        let session2 = UploadSession::new("bucket", "obj", "video/mp4", &cp);
        let resumed = ResumableUpload::new_or_resume(session2).expect("resume");
        assert_eq!(resumed.completed_part_count(), 2);
        assert_eq!(resumed.last_completed_part(), 2);
        assert_eq!(resumed.upload_id(), Some("upload-xyz"));
        assert_eq!(resumed.next_offset_bytes(), 10_485_760);

        let _ = std::fs::remove_file(&cp);
    }

    #[test]
    fn test_record_part_idempotent() {
        let cp = tmp_checkpoint("idempotent");
        let _ = std::fs::remove_file(&cp);
        let session = UploadSession::new("b", "k", "application/octet-stream", &cp);
        let mut upload = ResumableUpload::new_or_resume(session).expect("create");
        upload
            .record_part(1, "etag-1".to_string(), 0, 1024)
            .expect("first");
        upload
            .record_part(1, "etag-1-retry".to_string(), 0, 1024)
            .expect("retry");
        // Only one part should remain
        assert_eq!(upload.completed_part_count(), 1);
        assert_eq!(upload.completed_parts()[0].etag, "etag-1-retry");
        let _ = std::fs::remove_file(&cp);
    }

    #[test]
    fn test_complete_removes_checkpoint() {
        let cp = tmp_checkpoint("complete");
        let _ = std::fs::remove_file(&cp);
        let session = UploadSession::new("b", "k", "video/mp4", &cp);
        let mut upload = ResumableUpload::new_or_resume(session).expect("create");
        upload
            .record_part(1, "e".to_string(), 0, 100)
            .expect("record");
        assert!(cp.exists(), "checkpoint should exist after record_part");
        upload.complete().expect("complete");
        assert!(!cp.exists(), "checkpoint should be removed after complete");
    }

    #[test]
    fn test_completed_parts_sorted() {
        let cp = tmp_checkpoint("sorted");
        let _ = std::fs::remove_file(&cp);
        let session = UploadSession::new("b", "k", "video/mp4", &cp);
        let mut upload = ResumableUpload::new_or_resume(session).expect("create");
        upload
            .record_part(3, "e3".to_string(), 2048, 1024)
            .expect("p3");
        upload
            .record_part(1, "e1".to_string(), 0, 1024)
            .expect("p1");
        upload
            .record_part(2, "e2".to_string(), 1024, 1024)
            .expect("p2");
        let parts = upload.completed_parts();
        assert_eq!(parts[0].part_number, 1);
        assert_eq!(parts[1].part_number, 2);
        assert_eq!(parts[2].part_number, 3);
        let _ = std::fs::remove_file(&cp);
    }

    // ── MultipartConfig & parallel upload tests ────────────────────────────────

    #[tokio::test]
    async fn test_multipart_config_default_values() {
        let cfg = MultipartConfig::default();
        assert_eq!(cfg.chunk_size_bytes, 8 * 1024 * 1024);
        assert_eq!(cfg.max_concurrent_parts, 4);
    }

    #[tokio::test]
    async fn test_upload_parallel_single_chunk() {
        let data = vec![0xABu8; 1024];
        let cfg = MultipartConfig {
            chunk_size_bytes: 4096,
            max_concurrent_parts: 2,
        };
        let results = upload_parallel(&data, &cfg).await.expect("upload");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].part_number, 1);
        assert_eq!(results[0].offset_bytes, 0);
        assert_eq!(results[0].size_bytes, 1024);
    }

    #[tokio::test]
    async fn test_upload_parallel_multiple_chunks_ordered() {
        // 10 KB data, 3 KB chunks → 4 parts.
        let data: Vec<u8> = (0u8..=255).cycle().take(10_240).collect();
        let cfg = MultipartConfig {
            chunk_size_bytes: 3_072,
            max_concurrent_parts: 2,
        };
        let results = upload_parallel(&data, &cfg).await.expect("upload");
        assert_eq!(results.len(), 4, "expected 4 chunks");
        // Results must be sorted by part number.
        for (i, r) in results.iter().enumerate() {
            assert_eq!(r.part_number, (i + 1) as u32);
        }
    }

    #[tokio::test]
    async fn test_upload_parallel_covers_all_bytes() {
        let data: Vec<u8> = (0u8..=255).cycle().take(20_000).collect();
        let cfg = MultipartConfig {
            chunk_size_bytes: 4_000,
            max_concurrent_parts: 4,
        };
        let results = upload_parallel(&data, &cfg).await.expect("upload");

        // Sum of part sizes should equal total data length.
        let total: u64 = results.iter().map(|r| r.size_bytes).sum();
        assert_eq!(total, 20_000);
    }

    #[tokio::test]
    async fn test_upload_parallel_last_chunk_smaller() {
        // 10_001 bytes with 5_000-byte chunks → 2 full + 1 partial.
        let data = vec![1u8; 10_001];
        let cfg = MultipartConfig {
            chunk_size_bytes: 5_000,
            max_concurrent_parts: 4,
        };
        let results = upload_parallel(&data, &cfg).await.expect("upload");
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].size_bytes, 5_000);
        assert_eq!(results[1].size_bytes, 5_000);
        assert_eq!(results[2].size_bytes, 1);
    }

    #[tokio::test]
    async fn test_upload_parallel_etags_are_non_empty() {
        let data: Vec<u8> = vec![42u8; 8_192];
        let cfg = MultipartConfig {
            chunk_size_bytes: 4_096,
            max_concurrent_parts: 2,
        };
        let results = upload_parallel(&data, &cfg).await.expect("upload");
        for r in &results {
            assert!(!r.etag.is_empty(), "ETag must not be empty");
        }
    }

    #[tokio::test]
    async fn test_upload_parallel_etags_differ_when_chunks_differ() {
        // Two chunks with different content should produce different ETags.
        let mut data = vec![0u8; 4_096];
        data.extend(vec![0xFFu8; 4_096]);
        let cfg = MultipartConfig {
            chunk_size_bytes: 4_096,
            max_concurrent_parts: 2,
        };
        let results = upload_parallel(&data, &cfg).await.expect("upload");
        assert_eq!(results.len(), 2);
        assert_ne!(results[0].etag, results[1].etag);
    }

    #[tokio::test]
    async fn test_upload_parallel_empty_data() {
        let data: Vec<u8> = vec![];
        let cfg = MultipartConfig::default();
        let results = upload_parallel(&data, &cfg).await.expect("upload empty");
        // Empty input → 0 parts (ceil(0 / chunk_size) = 0).
        assert!(results.is_empty());
    }
}
