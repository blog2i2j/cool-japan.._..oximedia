//! Multipart writer for large file uploads in parallel segments.
//!
//! [`MultipartWriter`] splits large files into fixed-size parts that can be
//! written (or uploaded) in parallel, similar to S3 multipart upload semantics.
//! The writer maintains an ordered list of parts and finalises them in sequence.

use std::collections::BTreeMap;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

/// Minimum allowed part size (5 MiB — matches S3 minimum).
pub const MIN_PART_SIZE: u64 = 5 * 1024 * 1024;

/// Default part size (8 MiB).
pub const DEFAULT_PART_SIZE: u64 = 8 * 1024 * 1024;

/// Identifier for a single multipart segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PartId(pub u32);

impl std::fmt::Display for PartId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "part-{:04}", self.0)
    }
}

/// Status of an individual part.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PartStatus {
    /// Part has not yet been written.
    Pending,
    /// Part is currently being written.
    InProgress,
    /// Part completed successfully; stores the number of bytes written.
    Complete(u64),
    /// Part failed with the given error message.
    Failed(String),
}

/// Metadata for a single part.
#[derive(Debug, Clone)]
pub struct PartInfo {
    /// Part identifier.
    pub id: PartId,
    /// Byte offset in the final file.
    pub offset: u64,
    /// Requested size in bytes.
    pub size: u64,
    /// Current status.
    pub status: PartStatus,
    /// Optional ETag or checksum from the backend.
    pub etag: Option<String>,
}

/// Configuration for [`MultipartWriter`].
#[derive(Debug, Clone)]
pub struct MultipartConfig {
    /// Size of each part in bytes (must be >= [`MIN_PART_SIZE`]).
    pub part_size: u64,
    /// Maximum number of concurrent in-progress parts.
    pub max_concurrency: usize,
    /// Base directory for temporary part files (uses system temp dir if `None`).
    pub temp_dir: Option<PathBuf>,
    /// Whether to clean up temporary files on drop.
    pub cleanup_on_drop: bool,
}

impl Default for MultipartConfig {
    fn default() -> Self {
        Self {
            part_size: DEFAULT_PART_SIZE,
            max_concurrency: 4,
            temp_dir: None,
            cleanup_on_drop: true,
        }
    }
}

/// Result of a completed multipart write operation.
#[derive(Debug, Clone)]
pub struct MultipartResult {
    /// Total bytes written across all parts.
    pub total_bytes: u64,
    /// Number of parts written.
    pub part_count: usize,
    /// Per-part ETags / checksums (in order).
    pub etags: Vec<Option<String>>,
}

/// Error type for multipart writer operations.
#[derive(Debug)]
pub enum MultipartError {
    /// I/O error.
    Io(io::Error),
    /// Part is out of range.
    PartOutOfRange { id: PartId, max: u32 },
    /// Total size not yet known (set_total_size must be called first).
    SizeUnknown,
    /// Configuration error.
    Config(String),
    /// Assembly error.
    Assembly(String),
}

impl std::fmt::Display for MultipartError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "multipart I/O error: {e}"),
            Self::PartOutOfRange { id, max } => {
                write!(f, "part {id} out of range (max part-{max:04})")
            }
            Self::SizeUnknown => write!(f, "total size unknown; call set_total_size first"),
            Self::Config(msg) => write!(f, "multipart config error: {msg}"),
            Self::Assembly(msg) => write!(f, "multipart assembly error: {msg}"),
        }
    }
}

impl std::error::Error for MultipartError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        if let Self::Io(e) = self {
            Some(e)
        } else {
            None
        }
    }
}

impl From<io::Error> for MultipartError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

pub type MultipartWriterResult<T> = Result<T, MultipartError>;

/// Manages writing a large file in parallel segments.
///
/// # Example
///
/// ```no_run
/// use oximedia_io::multipart_writer::{MultipartWriter, MultipartConfig};
///
/// let config = MultipartConfig::default();
/// let mut writer = MultipartWriter::new("output.bin", config).unwrap();
/// writer.set_total_size(64 * 1024 * 1024).unwrap(); // 64 MiB
/// let parts = writer.plan_parts();
/// for part_id in &parts {
///     let (offset, size) = writer.part_range(*part_id).unwrap();
///     // ... read/generate data[offset..offset+size] and write it
///     let data = vec![0u8; size as usize];
///     writer.write_part(*part_id, &data).unwrap();
/// }
/// let result = writer.finalize("output.bin").unwrap();
/// println!("Wrote {} bytes in {} parts", result.total_bytes, result.part_count);
/// ```
pub struct MultipartWriter {
    config: MultipartConfig,
    destination: PathBuf,
    total_size: Option<u64>,
    parts: BTreeMap<PartId, PartInfo>,
    temp_paths: BTreeMap<PartId, PathBuf>,
}

impl MultipartWriter {
    /// Create a new `MultipartWriter` targeting `destination`.
    ///
    /// # Errors
    ///
    /// Returns [`MultipartError::Config`] if `part_size < MIN_PART_SIZE`.
    pub fn new(
        destination: impl AsRef<Path>,
        config: MultipartConfig,
    ) -> MultipartWriterResult<Self> {
        if config.part_size < MIN_PART_SIZE {
            return Err(MultipartError::Config(format!(
                "part_size {} is below minimum {} bytes",
                config.part_size, MIN_PART_SIZE
            )));
        }
        Ok(Self {
            config,
            destination: destination.as_ref().to_path_buf(),
            total_size: None,
            parts: BTreeMap::new(),
            temp_paths: BTreeMap::new(),
        })
    }

    /// Set the total size of the file being written.
    ///
    /// This must be called before [`plan_parts`][Self::plan_parts].
    ///
    /// # Errors
    ///
    /// Returns an error if `size` is zero.
    pub fn set_total_size(&mut self, size: u64) -> MultipartWriterResult<()> {
        if size == 0 {
            return Err(MultipartError::Config(
                "total size must be greater than zero".to_string(),
            ));
        }
        self.total_size = Some(size);
        Ok(())
    }

    /// Plan the parts needed to cover the entire file.
    ///
    /// Returns the list of part IDs in ascending order.  Populates the
    /// internal parts table; subsequent calls with the same total size are
    /// idempotent.
    ///
    /// # Panics
    ///
    /// Panics if `set_total_size` has not been called.
    #[must_use]
    pub fn plan_parts(&mut self) -> Vec<PartId> {
        let total = self.total_size.unwrap_or(0);
        if total == 0 {
            return Vec::new();
        }

        // Re-plan only if parts table is empty.
        if !self.parts.is_empty() {
            return self.parts.keys().copied().collect();
        }

        let part_size = self.config.part_size;
        let mut offset = 0u64;
        let mut part_num = 1u32;

        while offset < total {
            let size = (total - offset).min(part_size);
            let id = PartId(part_num);
            self.parts.insert(
                id,
                PartInfo {
                    id,
                    offset,
                    size,
                    status: PartStatus::Pending,
                    etag: None,
                },
            );
            offset += size;
            part_num += 1;
        }

        self.parts.keys().copied().collect()
    }

    /// Returns the byte range `(offset, size)` for a given part.
    ///
    /// # Errors
    ///
    /// Returns [`MultipartError::PartOutOfRange`] if the part ID is unknown.
    pub fn part_range(&self, id: PartId) -> MultipartWriterResult<(u64, u64)> {
        self.parts
            .get(&id)
            .map(|p| (p.offset, p.size))
            .ok_or_else(|| {
                let max = self.parts.keys().next_back().map(|p| p.0).unwrap_or(0);
                MultipartError::PartOutOfRange { id, max }
            })
    }

    /// Write data for a single part.
    ///
    /// Data is buffered to a temporary file in the configured temp directory.
    /// The part status is updated to `Complete`.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the temp file cannot be written.
    pub fn write_part(&mut self, id: PartId, data: &[u8]) -> MultipartWriterResult<()> {
        if !self.parts.contains_key(&id) {
            let max = self.parts.keys().next_back().map(|p| p.0).unwrap_or(0);
            return Err(MultipartError::PartOutOfRange { id, max });
        }

        // Write data to temp file.
        let temp_dir = self
            .config
            .temp_dir
            .clone()
            .unwrap_or_else(std::env::temp_dir);
        let temp_path = temp_dir.join(format!(
            "oximedia_mpart_{}.bin",
            id
        ));

        let mut file = std::fs::File::create(&temp_path)?;
        file.write_all(data)?;
        file.flush()?;

        let written = data.len() as u64;
        self.temp_paths.insert(id, temp_path);
        if let Some(info) = self.parts.get_mut(&id) {
            info.status = PartStatus::Complete(written);
        }

        Ok(())
    }

    /// Set the ETag for a completed part.
    pub fn set_part_etag(&mut self, id: PartId, etag: impl Into<String>) {
        if let Some(info) = self.parts.get_mut(&id) {
            info.etag = Some(etag.into());
        }
    }

    /// Returns `true` if all planned parts are complete.
    #[must_use]
    pub fn all_parts_complete(&self) -> bool {
        self.parts
            .values()
            .all(|p| matches!(p.status, PartStatus::Complete(_)))
    }

    /// Returns the number of completed parts.
    #[must_use]
    pub fn completed_count(&self) -> usize {
        self.parts
            .values()
            .filter(|p| matches!(p.status, PartStatus::Complete(_)))
            .count()
    }

    /// Assemble all temporary part files into the final `destination` file.
    ///
    /// Parts are written in ascending order. After assembly, temporary files
    /// are removed (if `cleanup_on_drop` is set).
    ///
    /// # Errors
    ///
    /// Returns an error if any part is not complete, or if assembly fails.
    pub fn finalize(&mut self, destination: impl AsRef<Path>) -> MultipartWriterResult<MultipartResult> {
        if !self.all_parts_complete() {
            return Err(MultipartError::Assembly(
                "not all parts are complete".to_string(),
            ));
        }

        let dest = destination.as_ref();
        let mut out = std::fs::File::create(dest)?;
        let mut total_bytes = 0u64;
        let mut etags = Vec::with_capacity(self.parts.len());

        for (id, info) in &self.parts {
            etags.push(info.etag.clone());
            if let Some(temp_path) = self.temp_paths.get(id) {
                let chunk = std::fs::read(temp_path).map_err(|e| {
                    MultipartError::Assembly(format!("reading part {id}: {e}"))
                })?;
                out.write_all(&chunk)?;
                total_bytes += chunk.len() as u64;
            }
        }

        out.flush()?;

        // Cleanup temp files.
        if self.config.cleanup_on_drop {
            for temp_path in self.temp_paths.values() {
                let _ = std::fs::remove_file(temp_path);
            }
        }

        Ok(MultipartResult {
            total_bytes,
            part_count: self.parts.len(),
            etags,
        })
    }

    /// Returns the destination path.
    #[must_use]
    pub fn destination(&self) -> &Path {
        &self.destination
    }

    /// Returns an iterator over all planned parts.
    pub fn parts(&self) -> impl Iterator<Item = &PartInfo> {
        self.parts.values()
    }
}

impl Drop for MultipartWriter {
    fn drop(&mut self) {
        if self.config.cleanup_on_drop {
            for temp_path in self.temp_paths.values() {
                let _ = std::fs::remove_file(temp_path);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_str(name: &str) -> String {
        std::env::temp_dir()
            .join(format!("oximedia-io-mpart-{name}"))
            .to_string_lossy()
            .into_owned()
    }

    #[test]
    fn test_plan_parts_divides_correctly() {
        let config = MultipartConfig {
            part_size: MIN_PART_SIZE,
            ..Default::default()
        };
        let mut writer = MultipartWriter::new(tmp_str("test.bin"), config).unwrap();
        writer.set_total_size(MIN_PART_SIZE * 3 + 100).unwrap();
        let parts = writer.plan_parts();
        assert_eq!(parts.len(), 4);
    }

    #[test]
    fn test_plan_parts_exact_multiple() {
        let config = MultipartConfig {
            part_size: MIN_PART_SIZE,
            ..Default::default()
        };
        let mut writer = MultipartWriter::new(tmp_str("test.bin"), config).unwrap();
        writer.set_total_size(MIN_PART_SIZE * 2).unwrap();
        let parts = writer.plan_parts();
        assert_eq!(parts.len(), 2);
    }

    #[test]
    fn test_part_range() {
        let config = MultipartConfig {
            part_size: MIN_PART_SIZE,
            ..Default::default()
        };
        let mut writer = MultipartWriter::new(tmp_str("test.bin"), config).unwrap();
        writer.set_total_size(MIN_PART_SIZE + 1000).unwrap();
        writer.plan_parts();
        let (off1, sz1) = writer.part_range(PartId(1)).unwrap();
        assert_eq!(off1, 0);
        assert_eq!(sz1, MIN_PART_SIZE);
        let (off2, sz2) = writer.part_range(PartId(2)).unwrap();
        assert_eq!(off2, MIN_PART_SIZE);
        assert_eq!(sz2, 1000);
    }

    #[test]
    fn test_write_and_finalize() {
        let temp_dir = std::env::temp_dir();
        let dest = temp_dir.join("oximedia_mpart_final_test.bin");
        let config = MultipartConfig {
            part_size: MIN_PART_SIZE,
            temp_dir: Some(temp_dir.clone()),
            ..Default::default()
        };
        let total = MIN_PART_SIZE + 512;
        let mut writer = MultipartWriter::new(&dest, config).unwrap();
        writer.set_total_size(total).unwrap();
        let parts = writer.plan_parts();

        for id in &parts {
            let (_, size) = writer.part_range(*id).unwrap();
            let data = vec![id.0 as u8; size as usize];
            writer.write_part(*id, &data).unwrap();
        }

        assert!(writer.all_parts_complete());
        let result = writer.finalize(&dest).unwrap();
        assert_eq!(result.total_bytes, total);
        assert_eq!(result.part_count, 2);

        let written = std::fs::read(&dest).unwrap();
        assert_eq!(written.len() as u64, total);
        let _ = std::fs::remove_file(&dest);
    }

    #[test]
    fn test_config_too_small_part_size() {
        let config = MultipartConfig {
            part_size: 1024, // below minimum
            ..Default::default()
        };
        assert!(MultipartWriter::new(tmp_str("test.bin"), config).is_err());
    }

    #[test]
    fn test_part_out_of_range() {
        let mut writer =
            MultipartWriter::new(tmp_str("test.bin"), MultipartConfig::default()).unwrap();
        writer.set_total_size(MIN_PART_SIZE).unwrap();
        writer.plan_parts();
        assert!(writer.part_range(PartId(999)).is_err());
    }
}
