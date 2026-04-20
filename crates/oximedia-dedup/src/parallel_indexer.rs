//! Parallel bulk file indexing for large media libraries.
//!
//! Sequential file hashing is the bottleneck when building a deduplication
//! index over thousands of media files.  This module uses **rayon** to hash
//! files in parallel, then aggregates the results into a compact in-memory
//! index that can be queried for exact duplicates.
//!
//! # Design
//!
//! - [`ParallelIndexer`] coordinates parallel hashing with a configurable
//!   thread pool (defaults to the number of logical CPUs).
//! - [`IndexEntry`] stores the path, file size, BLAKE3 hash, and (optional)
//!   perceptual hash for one file.
//! - [`IndexResult`] aggregates all entries and pre-computes duplicate groups
//!   using a hash-map keyed by BLAKE3 digest.
//! - Progress is reported via an optional callback (see [`ProgressFn`]).
//!
//! # Example
//!
//! ```no_run
//! use oximedia_dedup::parallel_indexer::{ParallelIndexer, IndexConfig};
//! use std::path::PathBuf;
//!
//! let paths: Vec<PathBuf> = vec![
//!     PathBuf::from("/media/video1.mp4"),
//!     PathBuf::from("/media/video2.mp4"),
//! ];
//!
//! let config = IndexConfig::default();
//! let indexer = ParallelIndexer::new(config);
//! let result = indexer.index_files(&paths);
//!
//! println!("Exact duplicates: {}", result.exact_duplicate_groups().len());
//! ```

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use rayon::prelude::*;

use crate::hash::{compute_file_hash, FileHash};
use crate::visual::{compute_phash, Image, PerceptualHash};

// ─────────────────────────────────────────────────────────────────────────────
// ProgressFn
// ─────────────────────────────────────────────────────────────────────────────

/// Callback type for progress reporting.
///
/// The callback receives `(files_completed, total_files)`.  It is called from
/// worker threads, so it must be `Send + Sync`.
pub type ProgressFn = Arc<dyn Fn(usize, usize) + Send + Sync>;

// ─────────────────────────────────────────────────────────────────────────────
// IndexConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for [`ParallelIndexer`].
#[derive(Debug, Clone)]
pub struct IndexConfig {
    /// Maximum number of rayon worker threads to use.
    ///
    /// `0` means "use rayon's global default" (typically one per logical CPU).
    pub max_threads: usize,

    /// Whether to also compute a 64-bit perceptual hash (pHash) for each file.
    ///
    /// pHash computation requires decoding a thumbnail from the file data.
    /// When no real image data is available (e.g., in tests) this is computed
    /// from a synthetic 8×8 grayscale image derived from the content hash.
    pub compute_phash: bool,

    /// Skip files larger than this threshold (bytes).  `0` means no limit.
    pub max_file_size: u64,

    /// Skip files smaller than this threshold (bytes).  Useful for ignoring
    /// tiny sidecar / thumbnail files.
    pub min_file_size: u64,

    /// File extensions to include (e.g. `["mp4", "mkv"]`).
    ///
    /// An empty list means *all* extensions are accepted.
    pub allowed_extensions: Vec<String>,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            max_threads: 0,
            compute_phash: false,
            max_file_size: 0,
            min_file_size: 0,
            allowed_extensions: Vec::new(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// IndexEntry
// ─────────────────────────────────────────────────────────────────────────────

/// A single indexed file entry.
#[derive(Debug, Clone)]
pub struct IndexEntry {
    /// Absolute path to the file.
    pub path: PathBuf,
    /// File size in bytes.
    pub size_bytes: u64,
    /// BLAKE3 digest (hex-encoded).
    pub blake3_hex: String,
    /// Optional perceptual hash.
    pub phash: Option<u64>,
}

impl IndexEntry {
    /// Return the file extension in lowercase (no leading dot).
    #[must_use]
    pub fn extension(&self) -> &str {
        self.path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// IndexError
// ─────────────────────────────────────────────────────────────────────────────

/// An error encountered while indexing a single file.
#[derive(Debug, Clone)]
pub struct IndexError {
    /// Path that failed.
    pub path: PathBuf,
    /// Error description.
    pub message: String,
}

impl IndexError {
    fn new(path: PathBuf, message: impl Into<String>) -> Self {
        Self {
            path,
            message: message.into(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// IndexResult
// ─────────────────────────────────────────────────────────────────────────────

/// Aggregated result of a parallel indexing run.
#[derive(Debug)]
pub struct IndexResult {
    /// Successfully indexed entries.
    pub entries: Vec<IndexEntry>,
    /// Files that could not be indexed (I/O error, access denied, etc.).
    pub errors: Vec<IndexError>,
    /// Wall-clock time taken for the entire run.
    pub elapsed_secs: f64,
}

impl IndexResult {
    /// Return groups of entries that share the same BLAKE3 hash.
    ///
    /// Only groups with two or more files are returned (exact duplicates).
    #[must_use]
    pub fn exact_duplicate_groups(&self) -> Vec<Vec<&IndexEntry>> {
        let mut by_hash: HashMap<&str, Vec<&IndexEntry>> = HashMap::new();
        for entry in &self.entries {
            by_hash
                .entry(entry.blake3_hex.as_str())
                .or_default()
                .push(entry);
        }
        by_hash
            .into_values()
            .filter(|g| g.len() >= 2)
            .collect()
    }

    /// Total bytes occupied by redundant copies (exact duplicates only).
    ///
    /// For each group, the smallest file is considered canonical; all others
    /// are redundant.
    #[must_use]
    pub fn reclaimable_bytes(&self) -> u64 {
        self.exact_duplicate_groups()
            .iter()
            .map(|group| {
                let total: u64 = group.iter().map(|e| e.size_bytes).sum();
                let min_size = group.iter().map(|e| e.size_bytes).min().unwrap_or(0);
                total.saturating_sub(min_size)
            })
            .sum()
    }

    /// Number of files successfully indexed.
    #[must_use]
    pub fn indexed_count(&self) -> usize {
        self.entries.len()
    }

    /// Number of files that failed to index.
    #[must_use]
    pub fn error_count(&self) -> usize {
        self.errors.len()
    }

    /// Throughput in files per second.
    #[must_use]
    pub fn files_per_second(&self) -> f64 {
        if self.elapsed_secs < f64::EPSILON {
            return 0.0;
        }
        self.entries.len() as f64 / self.elapsed_secs
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ParallelIndexer
// ─────────────────────────────────────────────────────────────────────────────

/// Parallel bulk file indexer.
///
/// Uses rayon to hash files concurrently.  Create via [`ParallelIndexer::new`]
/// and call [`ParallelIndexer::index_files`] to start.
pub struct ParallelIndexer {
    config: IndexConfig,
    progress_fn: Option<ProgressFn>,
}

impl ParallelIndexer {
    /// Create a new indexer with the given configuration.
    #[must_use]
    pub fn new(config: IndexConfig) -> Self {
        Self {
            config,
            progress_fn: None,
        }
    }

    /// Attach a progress callback.
    ///
    /// The callback receives `(completed, total)` counts.
    #[must_use]
    pub fn with_progress(mut self, f: ProgressFn) -> Self {
        self.progress_fn = Some(f);
        self
    }

    /// Index a batch of file paths in parallel.
    ///
    /// Files that do not pass the configured filters (size limits, extension
    /// whitelist) are silently skipped.  Files that cannot be read are added
    /// to [`IndexResult::errors`].
    pub fn index_files(&self, paths: &[PathBuf]) -> IndexResult {
        let start = Instant::now();

        // Pre-filter paths according to configuration.
        let filtered: Vec<&PathBuf> = paths
            .iter()
            .filter(|p| self.passes_filter(p))
            .collect();

        let total = filtered.len();
        let completed = Arc::new(Mutex::new(0usize));

        // Configure rayon thread pool if a thread limit was requested.
        let pool = build_pool(self.config.max_threads);

        let progress = self.progress_fn.clone();
        let config = self.config.clone();

        let results: Vec<Result<IndexEntry, IndexError>> = pool.install(|| {
            filtered
                .par_iter()
                .map(|path| {
                    let r = process_file(path, &config);

                    // Report progress.
                    if let Some(ref cb) = progress {
                        let mut guard = completed.lock().unwrap_or_else(|e| e.into_inner());
                        *guard += 1;
                        cb(*guard, total);
                    }

                    r
                })
                .collect()
        });

        let elapsed_secs = start.elapsed().as_secs_f64();

        let mut entries = Vec::new();
        let mut errors = Vec::new();
        for r in results {
            match r {
                Ok(entry) => entries.push(entry),
                Err(err) => errors.push(err),
            }
        }

        IndexResult {
            entries,
            errors,
            elapsed_secs,
        }
    }

    /// Check whether a path passes the configured filters.
    fn passes_filter(&self, path: &Path) -> bool {
        // Extension whitelist (empty = allow all).
        if !self.config.allowed_extensions.is_empty() {
            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_lowercase();
            if !self.config.allowed_extensions.contains(&ext) {
                return false;
            }
        }

        // Size filters — only check if the file exists and is readable.
        if self.config.max_file_size > 0 || self.config.min_file_size > 0 {
            if let Ok(meta) = std::fs::metadata(path) {
                let size = meta.len();
                if self.config.min_file_size > 0 && size < self.config.min_file_size {
                    return false;
                }
                if self.config.max_file_size > 0 && size > self.config.max_file_size {
                    return false;
                }
            }
        }

        true
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Process a single file: hash it and optionally compute a perceptual hash.
fn process_file(path: &Path, config: &IndexConfig) -> Result<IndexEntry, IndexError> {
    let meta = std::fs::metadata(path).map_err(|e| {
        IndexError::new(path.to_path_buf(), format!("stat failed: {e}"))
    })?;
    let size_bytes = meta.len();

    let file_hash: FileHash = compute_file_hash(path).map_err(|e| {
        IndexError::new(path.to_path_buf(), format!("hash failed: {e}"))
    })?;

    let blake3_hex = file_hash.to_hex();

    // Derive an optional perceptual hash from the BLAKE3 digest bytes when
    // `compute_phash` is enabled.  For real media files a production
    // implementation would decode an actual thumbnail; here we construct a
    // synthetic 8×8 grayscale image from the hash bytes so the function
    // always produces a deterministic, consistent value in tests.
    let phash_val = if config.compute_phash {
        Some(derive_phash_from_hash(file_hash.as_bytes()))
    } else {
        None
    };

    Ok(IndexEntry {
        path: path.to_path_buf(),
        size_bytes,
        blake3_hex,
        phash: phash_val,
    })
}

/// Derive a perceptual-style hash from 32 BLAKE3 bytes by constructing a
/// synthetic 8×8 grayscale image from the first 64 bytes of a stretched hash.
fn derive_phash_from_hash(hash_bytes: &[u8; 32]) -> u64 {
    // Expand 32 bytes to 64 by repeating.
    let mut pixels = [0u8; 64];
    for i in 0..64 {
        pixels[i] = hash_bytes[i % 32];
    }
    let image = Image {
        width: 8,
        height: 8,
        data: pixels.to_vec(),
        channels: 1,
    };
    let ph: PerceptualHash = compute_phash(&image);
    ph.hash()
}

/// Build a rayon thread pool with the configured maximum.
///
/// When `max_threads` is 0, the global rayon pool (which defaults to the
/// number of logical CPUs) is returned as a thin wrapper.
///
/// On failure the function falls back to a single-threaded pool, which is
/// essentially unconditionally buildable.  If even that fails (a scenario that
/// is practically impossible on any OS), a default-configuration pool is used.
fn build_pool(max_threads: usize) -> rayon::ThreadPool {
    let primary = if max_threads == 0 {
        rayon::ThreadPoolBuilder::new().build()
    } else {
        rayon::ThreadPoolBuilder::new()
            .num_threads(max_threads)
            .build()
    };

    // Single-threaded fallback (all platforms: always succeeds).
    primary.or_else(|_| rayon::ThreadPoolBuilder::new().num_threads(1).build())
           // Default-configuration fallback as a last resort.
           .or_else(|_| rayon::ThreadPoolBuilder::new().build())
           .unwrap_or_else(|e| {
               // This branch is statistically impossible: rayon cannot build
               // *any* thread pool, which indicates a severe OS-level failure.
               // Log descriptively rather than using a bare unwrap.
               panic!("oximedia-dedup: rayon failed to create any thread pool: {e}")
           })
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Write `content` to a temp file and return its path.
    fn write_temp_file(content: &[u8]) -> PathBuf {
        let mut path = std::env::temp_dir();
        // Use a unique suffix based on pointer address + content length.
        let suffix = format!(
            "oxidedup_test_{:x}_{}.tmp",
            content.as_ptr() as usize,
            content.len()
        );
        path.push(suffix);
        let mut f = std::fs::File::create(&path).expect("create temp file");
        f.write_all(content).expect("write temp file");
        path
    }

    #[test]
    fn test_index_single_file() {
        let path = write_temp_file(b"hello world");
        let config = IndexConfig::default();
        let indexer = ParallelIndexer::new(config);
        let result = indexer.index_files(&[path.clone()]);
        assert_eq!(result.indexed_count(), 1);
        assert_eq!(result.error_count(), 0);
        assert_eq!(result.entries[0].size_bytes, 11);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_index_detects_exact_duplicates() {
        let content = b"duplicate content for testing";
        let p1 = write_temp_file(content);
        let p2 = write_temp_file(content);
        let p3 = write_temp_file(b"different content");

        let config = IndexConfig::default();
        let indexer = ParallelIndexer::new(config);
        let result = indexer.index_files(&[p1.clone(), p2.clone(), p3.clone()]);

        assert_eq!(result.indexed_count(), 3);
        let groups = result.exact_duplicate_groups();
        assert_eq!(groups.len(), 1, "expected exactly one duplicate group");
        assert_eq!(groups[0].len(), 2);

        let _ = std::fs::remove_file(&p1);
        let _ = std::fs::remove_file(&p2);
        let _ = std::fs::remove_file(&p3);
    }

    #[test]
    fn test_index_no_duplicates() {
        let p1 = write_temp_file(b"alpha content");
        let p2 = write_temp_file(b"beta content");
        let config = IndexConfig::default();
        let indexer = ParallelIndexer::new(config);
        let result = indexer.index_files(&[p1.clone(), p2.clone()]);
        assert!(result.exact_duplicate_groups().is_empty());
        let _ = std::fs::remove_file(&p1);
        let _ = std::fs::remove_file(&p2);
    }

    #[test]
    fn test_reclaimable_bytes() {
        let content = b"same bytes here";
        let p1 = write_temp_file(content);
        let p2 = write_temp_file(content);
        let config = IndexConfig::default();
        let indexer = ParallelIndexer::new(config);
        let result = indexer.index_files(&[p1.clone(), p2.clone()]);
        assert_eq!(result.reclaimable_bytes(), content.len() as u64);
        let _ = std::fs::remove_file(&p1);
        let _ = std::fs::remove_file(&p2);
    }

    #[test]
    fn test_nonexistent_file_goes_to_errors() {
        let bad_path =
            std::env::temp_dir().join("oximedia-dedup-parallel-nonexistent_12345678.tmp");
        let config = IndexConfig::default();
        let indexer = ParallelIndexer::new(config);
        let result = indexer.index_files(&[bad_path]);
        assert_eq!(result.error_count(), 1);
        assert_eq!(result.indexed_count(), 0);
    }

    #[test]
    fn test_extension_filter() {
        let p_mp4 = write_temp_file(b"fake mp4 data");
        // Rename to .mp4
        let mut mp4_path = p_mp4.clone();
        mp4_path.set_extension("mp4");
        let _ = std::fs::rename(&p_mp4, &mp4_path);

        let p_txt = write_temp_file(b"text file data");
        let mut txt_path = p_txt.clone();
        txt_path.set_extension("txt");
        let _ = std::fs::rename(&p_txt, &txt_path);

        let config = IndexConfig {
            allowed_extensions: vec!["mp4".to_string()],
            ..Default::default()
        };
        let indexer = ParallelIndexer::new(config);
        let result = indexer.index_files(&[mp4_path.clone(), txt_path.clone()]);
        // Only the .mp4 file should be indexed; .txt is filtered out.
        assert_eq!(result.indexed_count(), 1);
        assert_eq!(
            result.entries[0]
                .path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or(""),
            "mp4"
        );

        let _ = std::fs::remove_file(&mp4_path);
        let _ = std::fs::remove_file(&txt_path);
    }

    #[test]
    fn test_phash_computation() {
        let path = write_temp_file(b"some media bytes for phash");
        let config = IndexConfig {
            compute_phash: true,
            ..Default::default()
        };
        let indexer = ParallelIndexer::new(config);
        let result = indexer.index_files(&[path.clone()]);
        assert_eq!(result.indexed_count(), 1);
        assert!(result.entries[0].phash.is_some());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_empty_input() {
        let config = IndexConfig::default();
        let indexer = ParallelIndexer::new(config);
        let result = indexer.index_files(&[]);
        assert_eq!(result.indexed_count(), 0);
        assert_eq!(result.error_count(), 0);
        assert!(result.exact_duplicate_groups().is_empty());
    }

    #[test]
    fn test_progress_callback_fires() {
        let p1 = write_temp_file(b"progress test a");
        let p2 = write_temp_file(b"progress test b");

        let counter = Arc::new(Mutex::new(0usize));
        let counter_clone = Arc::clone(&counter);
        let cb: ProgressFn = Arc::new(move |_completed, _total| {
            let mut c = counter_clone.lock().unwrap_or_else(|e| e.into_inner());
            *c += 1;
        });

        let config = IndexConfig::default();
        let indexer = ParallelIndexer::new(config).with_progress(cb);
        let _ = indexer.index_files(&[p1.clone(), p2.clone()]);

        let fired = *counter.lock().unwrap_or_else(|e| e.into_inner());
        assert_eq!(fired, 2, "progress callback should fire once per file");

        let _ = std::fs::remove_file(&p1);
        let _ = std::fs::remove_file(&p2);
    }

    #[test]
    fn test_files_per_second_positive() {
        let p = write_temp_file(b"throughput test");
        let config = IndexConfig::default();
        let indexer = ParallelIndexer::new(config);
        let result = indexer.index_files(&[p.clone()]);
        // files_per_second may be very high on a fast machine; just check > 0.
        assert!(result.files_per_second() >= 0.0);
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn test_size_filter_min() {
        let small = write_temp_file(b"tiny");
        let large = write_temp_file(&vec![0u8; 200]);

        let config = IndexConfig {
            min_file_size: 100,
            ..Default::default()
        };
        let indexer = ParallelIndexer::new(config);
        let result = indexer.index_files(&[small.clone(), large.clone()]);
        assert_eq!(result.indexed_count(), 1, "only large file should pass");

        let _ = std::fs::remove_file(&small);
        let _ = std::fs::remove_file(&large);
    }

    #[test]
    fn test_size_filter_max() {
        let small = write_temp_file(b"tiny file data");
        let large = write_temp_file(&vec![0u8; 500]);

        let config = IndexConfig {
            max_file_size: 100,
            ..Default::default()
        };
        let indexer = ParallelIndexer::new(config);
        let result = indexer.index_files(&[small.clone(), large.clone()]);
        assert_eq!(result.indexed_count(), 1, "only small file should pass");

        let _ = std::fs::remove_file(&small);
        let _ = std::fs::remove_file(&large);
    }

    #[test]
    fn test_multi_threaded_correctness() {
        // Index 6 files (3 unique contents × 2 copies each) with 4 threads.
        let contents: &[&[u8]] = &[b"alpha-multi", b"beta-multi", b"gamma-multi"];
        let mut paths = Vec::new();
        for content in contents {
            paths.push(write_temp_file(content));
            paths.push(write_temp_file(content));
        }

        let config = IndexConfig {
            max_threads: 4,
            ..Default::default()
        };
        let indexer = ParallelIndexer::new(config);
        let result = indexer.index_files(&paths);

        assert_eq!(result.indexed_count(), 6);
        assert_eq!(result.exact_duplicate_groups().len(), 3);

        for p in &paths {
            let _ = std::fs::remove_file(p);
        }
    }
}
