//! Media file scanner for discovering and cataloging source files.

use crate::error::ConformResult;
use crate::media::fingerprint::{extract_media_info, generate_fingerprint};
use crate::types::MediaFile;
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tracing::{debug, info, warn};
use walkdir::WalkDir;

/// Supported media file extensions.
const MEDIA_EXTENSIONS: &[&str] = &[
    "mov", "mp4", "m4v", "avi", "mkv", "mxf", "dpx", "tiff", "tif", "png", "jpg", "jpeg", "exr",
    "r3d", "arri", "braw", "wav", "aif", "aiff", "mp3", "aac", "flac",
];

/// Progress callback for scanning.
#[derive(Clone)]
pub struct ScanProgress {
    /// Total files found.
    pub total_files: Arc<AtomicUsize>,
    /// Files processed.
    pub processed_files: Arc<AtomicUsize>,
    /// Errors encountered.
    pub errors: Arc<AtomicUsize>,
}

impl ScanProgress {
    /// Create a new progress tracker.
    #[must_use]
    pub fn new() -> Self {
        Self {
            total_files: Arc::new(AtomicUsize::new(0)),
            processed_files: Arc::new(AtomicUsize::new(0)),
            errors: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Get the current progress as a percentage.
    #[must_use]
    pub fn percentage(&self) -> f64 {
        let total = self.total_files.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        let processed = self.processed_files.load(Ordering::Relaxed);
        (processed as f64 / total as f64) * 100.0
    }

    /// Check if scanning is complete.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        let total = self.total_files.load(Ordering::Relaxed);
        let processed = self.processed_files.load(Ordering::Relaxed);
        total > 0 && processed >= total
    }
}

impl Default for ScanProgress {
    fn default() -> Self {
        Self::new()
    }
}

/// Media file scanner.
pub struct MediaScanner {
    /// Whether to follow symbolic links.
    follow_links: bool,
    /// Maximum recursion depth (0 = unlimited).
    max_depth: usize,
    /// Generate checksums during scanning.
    generate_checksums: bool,
    /// Extract metadata during scanning.
    extract_metadata: bool,
    /// Custom file extensions to scan.
    custom_extensions: Vec<String>,
}

impl MediaScanner {
    /// Create a new media scanner with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            follow_links: false,
            max_depth: 0,
            generate_checksums: false,
            extract_metadata: true,
            custom_extensions: Vec::new(),
        }
    }

    /// Set whether to follow symbolic links.
    pub fn follow_links(&mut self, follow: bool) -> &mut Self {
        self.follow_links = follow;
        self
    }

    /// Set maximum recursion depth.
    pub fn max_depth(&mut self, depth: usize) -> &mut Self {
        self.max_depth = depth;
        self
    }

    /// Enable checksum generation.
    pub fn generate_checksums(&mut self, generate: bool) -> &mut Self {
        self.generate_checksums = generate;
        self
    }

    /// Enable metadata extraction.
    pub fn extract_metadata(&mut self, extract: bool) -> &mut Self {
        self.extract_metadata = extract;
        self
    }

    /// Add custom file extensions to scan.
    pub fn add_extension(&mut self, ext: String) -> &mut Self {
        self.custom_extensions.push(ext);
        self
    }

    /// Scan a directory for media files.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory cannot be read.
    pub fn scan<P: AsRef<Path>>(
        &self,
        path: P,
        progress: Option<Arc<ScanProgress>>,
    ) -> ConformResult<Vec<MediaFile>> {
        let path = path.as_ref();
        info!("Scanning directory: {}", path.display());

        // Find all media files
        let file_paths = self.find_media_files(path)?;

        if let Some(ref prog) = progress {
            prog.total_files.store(file_paths.len(), Ordering::Relaxed);
        }

        info!("Found {} media files", file_paths.len());

        // Process files in parallel
        let media_files: Vec<MediaFile> = file_paths
            .par_iter()
            .filter_map(|file_path| match self.process_file(file_path) {
                Ok(media) => {
                    if let Some(ref prog) = progress {
                        prog.processed_files.fetch_add(1, Ordering::Relaxed);
                    }
                    Some(media)
                }
                Err(e) => {
                    warn!("Failed to process {}: {}", file_path.display(), e);
                    if let Some(ref prog) = progress {
                        prog.errors.fetch_add(1, Ordering::Relaxed);
                        prog.processed_files.fetch_add(1, Ordering::Relaxed);
                    }
                    None
                }
            })
            .collect();

        info!("Successfully processed {} files", media_files.len());
        Ok(media_files)
    }

    /// Find all media files in a directory.
    fn find_media_files<P: AsRef<Path>>(&self, path: P) -> ConformResult<Vec<PathBuf>> {
        let mut walker = WalkDir::new(path).follow_links(self.follow_links);

        if self.max_depth > 0 {
            walker = walker.max_depth(self.max_depth);
        }

        let files: Vec<PathBuf> = walker
            .into_iter()
            .filter_map(std::result::Result::ok)
            .filter(|entry| entry.file_type().is_file())
            .filter(|entry| self.is_media_file(entry.path()))
            .map(|entry| entry.path().to_path_buf())
            .collect();

        Ok(files)
    }

    /// Check if a file is a media file based on extension.
    fn is_media_file<P: AsRef<Path>>(&self, path: P) -> bool {
        if let Some(ext) = path.as_ref().extension() {
            let ext_str = ext.to_string_lossy().to_lowercase();
            MEDIA_EXTENSIONS.contains(&ext_str.as_str())
                || self.custom_extensions.contains(&ext_str)
        } else {
            false
        }
    }

    /// Process a single file and extract metadata.
    fn process_file(&self, path: &Path) -> ConformResult<MediaFile> {
        debug!("Processing file: {}", path.display());

        let mut media = MediaFile::new(path.to_path_buf());

        // Get file size
        if let Ok(metadata) = std::fs::metadata(path) {
            media.size = Some(metadata.len());
        }

        // Generate fingerprints if enabled
        if self.generate_checksums {
            if let Ok(fingerprint) = generate_fingerprint(path) {
                media.md5 = Some(fingerprint.md5);
                media.xxhash = Some(fingerprint.xxhash);
            }
        }

        // Extract media metadata if enabled
        if self.extract_metadata {
            if let Ok(info) = extract_media_info(path) {
                media.duration = info.duration;
                media.width = info.width;
                media.height = info.height;
                media.fps = info.fps;
                media.timecode_start = info.timecode_start;
            }
        }

        Ok(media)
    }
}

impl Default for MediaScanner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_scanner_creation() {
        let scanner = MediaScanner::new();
        assert!(!scanner.follow_links);
        assert_eq!(scanner.max_depth, 0);
    }

    #[test]
    fn test_scanner_configuration() {
        let mut scanner = MediaScanner::new();
        scanner.follow_links(true).max_depth(5);
        assert!(scanner.follow_links);
        assert_eq!(scanner.max_depth, 5);
    }

    #[test]
    fn test_is_media_file() {
        let scanner = MediaScanner::new();
        assert!(scanner.is_media_file(Path::new("test.mov")));
        assert!(scanner.is_media_file(Path::new("test.mp4")));
        assert!(scanner.is_media_file(Path::new("test.MOV")));
        assert!(!scanner.is_media_file(Path::new("test.txt")));
    }

    #[test]
    fn test_scan_empty_directory() {
        let temp_dir = TempDir::new().expect("temp_dir should be valid");
        let scanner = MediaScanner::new();
        let result = scanner
            .scan(temp_dir.path(), None)
            .expect("result should be valid");
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_scan_with_progress() {
        let temp_dir = TempDir::new().expect("temp_dir should be valid");

        // Create some test files
        fs::write(temp_dir.path().join("test1.mov"), b"test").expect("test expectation failed");
        fs::write(temp_dir.path().join("test2.mp4"), b"test").expect("test expectation failed");
        fs::write(temp_dir.path().join("test.txt"), b"test").expect("test expectation failed");

        let scanner = MediaScanner::new();
        let progress = Arc::new(ScanProgress::new());
        let result = scanner
            .scan(temp_dir.path(), Some(progress.clone()))
            .expect("test expectation failed");

        assert_eq!(result.len(), 2);
        assert!(progress.is_complete());
    }

    #[test]
    fn test_custom_extensions() {
        let mut scanner = MediaScanner::new();
        scanner.add_extension("custom".to_string());
        assert!(scanner.is_media_file(Path::new("test.custom")));
    }
}
