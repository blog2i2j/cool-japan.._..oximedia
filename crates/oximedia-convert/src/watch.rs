// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Watch folder mode for automatic file conversion.
//!
//! Monitors a directory for new media files and automatically converts them
//! using configurable presets and output settings. Implements polling-based
//! file watching for maximum portability (pure Rust, no OS-specific deps).

use crate::formats::ContainerFormat;
use crate::{ConversionError, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

/// Configuration for the watch folder.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatchConfig {
    /// Directory to watch for new files.
    pub watch_dir: PathBuf,
    /// Directory for converted output files.
    pub output_dir: PathBuf,
    /// Target container format for conversion.
    pub target_format: ContainerFormat,
    /// File extensions to watch for (lowercase, without dot).
    pub watch_extensions: Vec<String>,
    /// Polling interval for directory scanning.
    pub poll_interval: Duration,
    /// Whether to delete source files after successful conversion.
    pub delete_after_convert: bool,
    /// Whether to recurse into subdirectories.
    pub recursive: bool,
    /// Minimum file age before processing (to avoid partial writes).
    pub min_file_age: Duration,
    /// Maximum number of concurrent conversions.
    pub max_concurrent: usize,
    /// Preset name to use for conversion (if any).
    pub preset_name: Option<String>,
    /// Whether to preserve the source directory structure in output.
    pub preserve_structure: bool,
}

impl Default for WatchConfig {
    fn default() -> Self {
        Self {
            watch_dir: PathBuf::from("."),
            output_dir: PathBuf::from("./converted"),
            target_format: ContainerFormat::Webm,
            watch_extensions: vec![
                "mp4".to_string(),
                "mov".to_string(),
                "avi".to_string(),
                "mkv".to_string(),
                "flac".to_string(),
                "wav".to_string(),
                "ogg".to_string(),
                "webm".to_string(),
            ],
            poll_interval: Duration::from_secs(5),
            delete_after_convert: false,
            recursive: false,
            min_file_age: Duration::from_secs(2),
            max_concurrent: 2,
            preset_name: None,
            preserve_structure: false,
        }
    }
}

impl WatchConfig {
    /// Create a new watch config for a directory.
    #[must_use]
    pub fn new(watch_dir: impl Into<PathBuf>, output_dir: impl Into<PathBuf>) -> Self {
        Self {
            watch_dir: watch_dir.into(),
            output_dir: output_dir.into(),
            ..Self::default()
        }
    }

    /// Set the target format.
    #[must_use]
    pub fn with_format(mut self, format: ContainerFormat) -> Self {
        self.target_format = format;
        self
    }

    /// Set the watch extensions.
    #[must_use]
    pub fn with_extensions(mut self, extensions: Vec<String>) -> Self {
        self.watch_extensions = extensions;
        self
    }

    /// Set the poll interval.
    #[must_use]
    pub fn with_poll_interval(mut self, interval: Duration) -> Self {
        self.poll_interval = interval;
        self
    }

    /// Enable deletion after conversion.
    #[must_use]
    pub fn with_delete_after_convert(mut self, delete: bool) -> Self {
        self.delete_after_convert = delete;
        self
    }

    /// Enable recursive watching.
    #[must_use]
    pub fn with_recursive(mut self, recursive: bool) -> Self {
        self.recursive = recursive;
        self
    }

    /// Set the minimum file age.
    #[must_use]
    pub fn with_min_file_age(mut self, age: Duration) -> Self {
        self.min_file_age = age;
        self
    }

    /// Set a preset name to use.
    #[must_use]
    pub fn with_preset(mut self, preset: &str) -> Self {
        self.preset_name = Some(preset.to_string());
        self
    }

    /// Enable structure preservation.
    #[must_use]
    pub fn with_preserve_structure(mut self, preserve: bool) -> Self {
        self.preserve_structure = preserve;
        self
    }

    /// Set max concurrent conversions.
    #[must_use]
    pub fn with_max_concurrent(mut self, max: usize) -> Self {
        self.max_concurrent = max.max(1);
        self
    }
}

/// Status of a watched file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WatchFileStatus {
    /// File detected, waiting for stability check.
    Detected,
    /// File is stable (not being written to) and queued for conversion.
    Queued,
    /// File is currently being converted.
    Converting,
    /// File has been converted successfully.
    Completed,
    /// Conversion failed.
    Failed,
    /// File was skipped (unsupported format, already processed, etc.).
    Skipped,
}

/// Entry tracking a watched file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatchEntry {
    /// Source file path.
    pub source_path: PathBuf,
    /// Planned or actual output path.
    pub output_path: PathBuf,
    /// Current status.
    pub status: WatchFileStatus,
    /// File size at last check (used for stability detection).
    pub last_size: u64,
    /// Time when the file was first detected.
    pub detected_at: SystemTime,
    /// Time when conversion started (if any).
    pub started_at: Option<SystemTime>,
    /// Time when conversion completed (if any).
    pub completed_at: Option<SystemTime>,
    /// Error message (if failed).
    pub error: Option<String>,
}

/// Statistics for the watch folder session.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WatchStats {
    /// Total files detected.
    pub files_detected: u64,
    /// Files queued for conversion.
    pub files_queued: u64,
    /// Files currently converting.
    pub files_converting: u64,
    /// Files successfully converted.
    pub files_completed: u64,
    /// Files that failed conversion.
    pub files_failed: u64,
    /// Files skipped.
    pub files_skipped: u64,
    /// Number of scan cycles completed.
    pub scan_cycles: u64,
}

/// Watch folder manager that monitors a directory and tracks files.
#[derive(Debug)]
pub struct WatchFolder {
    config: WatchConfig,
    /// Known files and their processing status.
    entries: HashMap<PathBuf, WatchEntry>,
    /// Set of already-processed file paths (for deduplication).
    processed: HashSet<PathBuf>,
    /// Cumulative statistics.
    stats: WatchStats,
}

impl WatchFolder {
    /// Create a new watch folder manager.
    #[must_use]
    pub fn new(config: WatchConfig) -> Self {
        Self {
            config,
            entries: HashMap::new(),
            processed: HashSet::new(),
            stats: WatchStats::default(),
        }
    }

    /// Get a reference to the configuration.
    #[must_use]
    pub fn config(&self) -> &WatchConfig {
        &self.config
    }

    /// Get current statistics.
    #[must_use]
    pub fn stats(&self) -> &WatchStats {
        &self.stats
    }

    /// Get all current entries.
    #[must_use]
    pub fn entries(&self) -> &HashMap<PathBuf, WatchEntry> {
        &self.entries
    }

    /// Get entries with a specific status.
    #[must_use]
    pub fn entries_with_status(&self, status: WatchFileStatus) -> Vec<&WatchEntry> {
        self.entries
            .values()
            .filter(|e| e.status == status)
            .collect()
    }

    /// Validate the watch folder configuration.
    pub fn validate(&self) -> Result<()> {
        if !self.config.watch_dir.exists() {
            return Err(ConversionError::InvalidInput(format!(
                "Watch directory does not exist: {}",
                self.config.watch_dir.display()
            )));
        }

        if !self.config.watch_dir.is_dir() {
            return Err(ConversionError::InvalidInput(format!(
                "Watch path is not a directory: {}",
                self.config.watch_dir.display()
            )));
        }

        if self.config.watch_extensions.is_empty() {
            return Err(ConversionError::InvalidInput(
                "No file extensions configured to watch".to_string(),
            ));
        }

        if self.config.max_concurrent == 0 {
            return Err(ConversionError::InvalidInput(
                "max_concurrent must be at least 1".to_string(),
            ));
        }

        Ok(())
    }

    /// Ensure the output directory exists.
    pub fn ensure_output_dir(&self) -> Result<()> {
        if !self.config.output_dir.exists() {
            std::fs::create_dir_all(&self.config.output_dir).map_err(|e| {
                ConversionError::InvalidOutput(format!(
                    "Cannot create output directory '{}': {e}",
                    self.config.output_dir.display()
                ))
            })?;
        }
        Ok(())
    }

    /// Scan the watch directory for new files.
    ///
    /// Returns the number of new files detected in this scan cycle.
    pub fn scan(&mut self) -> Result<usize> {
        let now = SystemTime::now();
        let mut new_files = 0;

        let files = self.collect_files(&self.config.watch_dir.clone())?;

        for file_path in files {
            // Skip already tracked or processed files
            if self.entries.contains_key(&file_path) || self.processed.contains(&file_path) {
                // Update size for stability detection on existing entries
                if let Some(entry) = self.entries.get_mut(&file_path) {
                    if entry.status == WatchFileStatus::Detected {
                        let current_size =
                            std::fs::metadata(&file_path).map(|m| m.len()).unwrap_or(0);
                        if current_size == entry.last_size && current_size > 0 {
                            // File size is stable; check age
                            if let Ok(age) = now.duration_since(entry.detected_at) {
                                if age >= self.config.min_file_age {
                                    entry.status = WatchFileStatus::Queued;
                                    self.stats.files_queued += 1;
                                }
                            }
                        } else {
                            entry.last_size = current_size;
                        }
                    }
                }
                continue;
            }

            // Check extension
            let ext = file_path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_lowercase();

            if !self.config.watch_extensions.contains(&ext) {
                continue;
            }

            let file_size = std::fs::metadata(&file_path).map(|m| m.len()).unwrap_or(0);

            if file_size == 0 {
                continue; // Skip empty files
            }

            let output_path = self.compute_output_path(&file_path);

            let entry = WatchEntry {
                source_path: file_path.clone(),
                output_path,
                status: WatchFileStatus::Detected,
                last_size: file_size,
                detected_at: now,
                started_at: None,
                completed_at: None,
                error: None,
            };

            self.entries.insert(file_path, entry);
            self.stats.files_detected += 1;
            new_files += 1;
        }

        self.stats.scan_cycles += 1;
        Ok(new_files)
    }

    /// Get the next batch of files ready for conversion.
    ///
    /// Returns up to `max_concurrent` entries that are in `Queued` status.
    #[must_use]
    pub fn next_batch(&self) -> Vec<PathBuf> {
        let currently_converting = self
            .entries
            .values()
            .filter(|e| e.status == WatchFileStatus::Converting)
            .count();

        let available_slots = self
            .config
            .max_concurrent
            .saturating_sub(currently_converting);

        self.entries
            .values()
            .filter(|e| e.status == WatchFileStatus::Queued)
            .take(available_slots)
            .map(|e| e.source_path.clone())
            .collect()
    }

    /// Mark a file as currently converting.
    pub fn mark_converting(&mut self, path: &Path) -> Result<()> {
        let entry = self.entries.get_mut(path).ok_or_else(|| {
            ConversionError::InvalidInput(format!("Unknown file: {}", path.display()))
        })?;
        entry.status = WatchFileStatus::Converting;
        entry.started_at = Some(SystemTime::now());
        self.stats.files_converting += 1;
        Ok(())
    }

    /// Mark a file as successfully converted.
    pub fn mark_completed(&mut self, path: &Path) -> Result<()> {
        let entry = self.entries.get_mut(path).ok_or_else(|| {
            ConversionError::InvalidInput(format!("Unknown file: {}", path.display()))
        })?;
        entry.status = WatchFileStatus::Completed;
        entry.completed_at = Some(SystemTime::now());
        self.stats.files_completed += 1;
        if self.stats.files_converting > 0 {
            self.stats.files_converting -= 1;
        }
        self.processed.insert(path.to_path_buf());
        Ok(())
    }

    /// Mark a file as failed.
    pub fn mark_failed(&mut self, path: &Path, error: &str) -> Result<()> {
        let entry = self.entries.get_mut(path).ok_or_else(|| {
            ConversionError::InvalidInput(format!("Unknown file: {}", path.display()))
        })?;
        entry.status = WatchFileStatus::Failed;
        entry.completed_at = Some(SystemTime::now());
        entry.error = Some(error.to_string());
        self.stats.files_failed += 1;
        if self.stats.files_converting > 0 {
            self.stats.files_converting -= 1;
        }
        self.processed.insert(path.to_path_buf());
        Ok(())
    }

    /// Skip a file (already converted, unsupported, etc.).
    pub fn mark_skipped(&mut self, path: &Path) -> Result<()> {
        let entry = self.entries.get_mut(path).ok_or_else(|| {
            ConversionError::InvalidInput(format!("Unknown file: {}", path.display()))
        })?;
        entry.status = WatchFileStatus::Skipped;
        self.stats.files_skipped += 1;
        self.processed.insert(path.to_path_buf());
        Ok(())
    }

    /// Clean up completed and failed entries older than the given duration.
    pub fn cleanup_old_entries(&mut self, max_age: Duration) {
        let now = SystemTime::now();
        let to_remove: Vec<PathBuf> = self
            .entries
            .iter()
            .filter(|(_, entry)| {
                matches!(
                    entry.status,
                    WatchFileStatus::Completed | WatchFileStatus::Failed | WatchFileStatus::Skipped
                )
            })
            .filter(|(_, entry)| {
                entry
                    .completed_at
                    .or(Some(entry.detected_at))
                    .and_then(|t| now.duration_since(t).ok())
                    .map_or(false, |age| age > max_age)
            })
            .map(|(path, _)| path.clone())
            .collect();

        for path in to_remove {
            self.entries.remove(&path);
        }
    }

    /// Reset the watch folder state (clear all entries and stats).
    pub fn reset(&mut self) {
        self.entries.clear();
        self.processed.clear();
        self.stats = WatchStats::default();
    }

    /// Compute the output file path for a given input file.
    fn compute_output_path(&self, input: &Path) -> PathBuf {
        let stem = input
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("output");

        let extension = self.config.target_format.extension();

        if self.config.preserve_structure {
            // Preserve the relative directory structure
            if let Ok(relative) = input.strip_prefix(&self.config.watch_dir) {
                if let Some(parent) = relative.parent() {
                    return self
                        .config
                        .output_dir
                        .join(parent)
                        .join(format!("{stem}.{extension}"));
                }
            }
        }

        self.config.output_dir.join(format!("{stem}.{extension}"))
    }

    /// Collect files from a directory (optionally recursive).
    fn collect_files(&self, dir: &Path) -> Result<Vec<PathBuf>> {
        let mut files = Vec::new();

        let entries = std::fs::read_dir(dir).map_err(|e| {
            ConversionError::InvalidInput(format!("Cannot read directory '{}': {e}", dir.display()))
        })?;

        for entry in entries {
            let entry = match entry {
                Ok(e) => e,
                Err(_) => continue,
            };

            let path = entry.path();

            if path.is_dir() && self.config.recursive {
                if let Ok(sub_files) = self.collect_files(&path) {
                    files.extend(sub_files);
                }
            } else if path.is_file() {
                files.push(path);
            }
        }

        Ok(files)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(watch_dir: &Path, output_dir: &Path) -> WatchConfig {
        WatchConfig::new(watch_dir, output_dir)
            .with_min_file_age(Duration::from_millis(0))
            .with_poll_interval(Duration::from_millis(100))
    }

    #[test]
    fn test_watch_config_default() {
        let config = WatchConfig::default();
        assert!(!config.delete_after_convert);
        assert!(!config.recursive);
        assert_eq!(config.max_concurrent, 2);
        assert_eq!(config.target_format, ContainerFormat::Webm);
        assert!(!config.watch_extensions.is_empty());
    }

    #[test]
    fn test_watch_config_builder() {
        let tmp = std::env::temp_dir();
        let config = WatchConfig::new(
            tmp.join("oximedia-convert-watch"),
            tmp.join("oximedia-convert-out"),
        )
        .with_format(ContainerFormat::Mp4)
        .with_recursive(true)
        .with_delete_after_convert(true)
        .with_max_concurrent(4)
        .with_preset("youtube-1080p")
        .with_preserve_structure(true)
        .with_min_file_age(Duration::from_secs(5))
        .with_poll_interval(Duration::from_secs(10))
        .with_extensions(vec!["mp4".to_string(), "mov".to_string()]);

        assert_eq!(config.target_format, ContainerFormat::Mp4);
        assert!(config.recursive);
        assert!(config.delete_after_convert);
        assert_eq!(config.max_concurrent, 4);
        assert_eq!(config.preset_name, Some("youtube-1080p".to_string()));
        assert!(config.preserve_structure);
        assert_eq!(config.min_file_age, Duration::from_secs(5));
        assert_eq!(config.poll_interval, Duration::from_secs(10));
        assert_eq!(config.watch_extensions.len(), 2);
    }

    #[test]
    fn test_watch_config_max_concurrent_minimum() {
        let config = WatchConfig::default().with_max_concurrent(0);
        assert_eq!(config.max_concurrent, 1);
    }

    #[test]
    fn test_watch_folder_creation() {
        let config = WatchConfig::default();
        let wf = WatchFolder::new(config);
        assert!(wf.entries().is_empty());
        assert_eq!(wf.stats().files_detected, 0);
        assert_eq!(wf.stats().scan_cycles, 0);
    }

    #[test]
    fn test_validate_nonexistent_dir() {
        let config = WatchConfig::new(
            PathBuf::from("/nonexistent/watch/dir"),
            std::env::temp_dir().join("oximedia-convert-out"),
        );
        let wf = WatchFolder::new(config);
        assert!(wf.validate().is_err());
    }

    #[test]
    fn test_validate_empty_extensions() {
        let tmp = std::env::temp_dir();
        let config =
            WatchConfig::new(tmp.clone(), tmp.join("oximedia-convert-out")).with_extensions(vec![]);
        let wf = WatchFolder::new(config);
        assert!(wf.validate().is_err());
    }

    #[test]
    fn test_validate_valid_config() {
        let watch_dir = std::env::temp_dir().join("oximedia_watch_test_valid");
        let _ = std::fs::create_dir_all(&watch_dir);
        let config = make_config(&watch_dir, &std::env::temp_dir().join("oximedia_watch_out"));
        let wf = WatchFolder::new(config);
        assert!(wf.validate().is_ok());
        let _ = std::fs::remove_dir_all(&watch_dir);
    }

    #[test]
    fn test_ensure_output_dir_creates() {
        let output_dir = std::env::temp_dir().join("oximedia_watch_ensure_out");
        let _ = std::fs::remove_dir_all(&output_dir);

        let config = make_config(&std::env::temp_dir(), &output_dir);
        let wf = WatchFolder::new(config);
        assert!(wf.ensure_output_dir().is_ok());
        assert!(output_dir.exists());

        let _ = std::fs::remove_dir_all(&output_dir);
    }

    #[test]
    fn test_scan_detects_files() {
        let watch_dir = std::env::temp_dir().join("oximedia_watch_scan");
        let output_dir = std::env::temp_dir().join("oximedia_watch_scan_out");
        let _ = std::fs::create_dir_all(&watch_dir);
        let _ = std::fs::create_dir_all(&output_dir);

        // Create some test files
        std::fs::write(watch_dir.join("test1.mp4"), &[0xAA; 1024]).expect("write test file");
        std::fs::write(watch_dir.join("test2.webm"), &[0xBB; 512]).expect("write test file");
        std::fs::write(watch_dir.join("readme.txt"), b"not a media file").expect("write test file");

        let config = make_config(&watch_dir, &output_dir);
        let mut wf = WatchFolder::new(config);
        let new_count = wf.scan().expect("scan should succeed");

        assert_eq!(new_count, 2); // mp4 and webm, not txt
        assert_eq!(wf.stats().files_detected, 2);
        assert_eq!(wf.stats().scan_cycles, 1);

        let _ = std::fs::remove_dir_all(&watch_dir);
        let _ = std::fs::remove_dir_all(&output_dir);
    }

    #[test]
    fn test_scan_skips_empty_files() {
        let watch_dir = std::env::temp_dir().join("oximedia_watch_empty");
        let output_dir = std::env::temp_dir().join("oximedia_watch_empty_out");
        let _ = std::fs::create_dir_all(&watch_dir);

        std::fs::write(watch_dir.join("empty.mp4"), &[]).expect("write");

        let config = make_config(&watch_dir, &output_dir);
        let mut wf = WatchFolder::new(config);
        let new_count = wf.scan().expect("scan should succeed");
        assert_eq!(new_count, 0);

        let _ = std::fs::remove_dir_all(&watch_dir);
    }

    #[test]
    fn test_scan_recursive() {
        let watch_dir = std::env::temp_dir().join("oximedia_watch_recursive");
        let sub_dir = watch_dir.join("subdir");
        let output_dir = std::env::temp_dir().join("oximedia_watch_recursive_out");
        let _ = std::fs::create_dir_all(&sub_dir);

        std::fs::write(watch_dir.join("root.mp4"), &[0xAA; 256]).expect("write");
        std::fs::write(sub_dir.join("nested.mkv"), &[0xBB; 256]).expect("write");

        let config = make_config(&watch_dir, &output_dir).with_recursive(true);
        let mut wf = WatchFolder::new(config);
        let new_count = wf.scan().expect("scan should succeed");
        assert_eq!(new_count, 2);

        let _ = std::fs::remove_dir_all(&watch_dir);
    }

    #[test]
    fn test_scan_non_recursive_ignores_subdirs() {
        let watch_dir = std::env::temp_dir().join("oximedia_watch_nonrec");
        let sub_dir = watch_dir.join("subdir");
        let output_dir = std::env::temp_dir().join("oximedia_watch_nonrec_out");
        let _ = std::fs::create_dir_all(&sub_dir);

        std::fs::write(watch_dir.join("root.mp4"), &[0xAA; 256]).expect("write");
        std::fs::write(sub_dir.join("nested.mkv"), &[0xBB; 256]).expect("write");

        let config = make_config(&watch_dir, &output_dir).with_recursive(false);
        let mut wf = WatchFolder::new(config);
        let new_count = wf.scan().expect("scan should succeed");
        assert_eq!(new_count, 1); // Only root.mp4

        let _ = std::fs::remove_dir_all(&watch_dir);
    }

    #[test]
    fn test_file_lifecycle() {
        let watch_dir = std::env::temp_dir().join("oximedia_watch_lifecycle");
        let output_dir = std::env::temp_dir().join("oximedia_watch_lifecycle_out");
        let _ = std::fs::create_dir_all(&watch_dir);
        let _ = std::fs::create_dir_all(&output_dir);

        let file_path = watch_dir.join("lifecycle.mp4");
        std::fs::write(&file_path, &[0xCC; 1024]).expect("write");

        let config = make_config(&watch_dir, &output_dir);
        let mut wf = WatchFolder::new(config);

        // Scan: file detected
        wf.scan().expect("scan");
        assert_eq!(wf.entries_with_status(WatchFileStatus::Detected).len(), 1);

        // Second scan: file becomes queued (stable)
        wf.scan().expect("scan");
        assert_eq!(wf.entries_with_status(WatchFileStatus::Queued).len(), 1);

        // Mark converting
        wf.mark_converting(&file_path).expect("mark converting");
        assert_eq!(wf.stats().files_converting, 1);
        assert_eq!(wf.entries_with_status(WatchFileStatus::Converting).len(), 1);

        // Mark completed
        wf.mark_completed(&file_path).expect("mark completed");
        assert_eq!(wf.stats().files_completed, 1);
        assert_eq!(wf.stats().files_converting, 0);

        let _ = std::fs::remove_dir_all(&watch_dir);
        let _ = std::fs::remove_dir_all(&output_dir);
    }

    #[test]
    fn test_mark_failed() {
        let watch_dir = std::env::temp_dir().join("oximedia_watch_failed");
        let output_dir = std::env::temp_dir().join("oximedia_watch_failed_out");
        let _ = std::fs::create_dir_all(&watch_dir);

        let file_path = watch_dir.join("fail.mp4");
        std::fs::write(&file_path, &[0xDD; 512]).expect("write");

        let config = make_config(&watch_dir, &output_dir);
        let mut wf = WatchFolder::new(config);

        wf.scan().expect("scan");
        wf.scan().expect("scan"); // become queued
        wf.mark_converting(&file_path).expect("mark converting");
        wf.mark_failed(&file_path, "test error")
            .expect("mark failed");

        assert_eq!(wf.stats().files_failed, 1);
        let entry = &wf.entries()[&file_path];
        assert_eq!(entry.status, WatchFileStatus::Failed);
        assert_eq!(entry.error, Some("test error".to_string()));

        let _ = std::fs::remove_dir_all(&watch_dir);
    }

    #[test]
    fn test_mark_skipped() {
        let watch_dir = std::env::temp_dir().join("oximedia_watch_skip");
        let output_dir = std::env::temp_dir().join("oximedia_watch_skip_out");
        let _ = std::fs::create_dir_all(&watch_dir);

        let file_path = watch_dir.join("skip.mp4");
        std::fs::write(&file_path, &[0xEE; 256]).expect("write");

        let config = make_config(&watch_dir, &output_dir);
        let mut wf = WatchFolder::new(config);

        wf.scan().expect("scan");
        wf.mark_skipped(&file_path).expect("mark skipped");
        assert_eq!(wf.stats().files_skipped, 1);

        let _ = std::fs::remove_dir_all(&watch_dir);
    }

    #[test]
    fn test_next_batch_respects_concurrency() {
        let watch_dir = std::env::temp_dir().join("oximedia_watch_batch");
        let output_dir = std::env::temp_dir().join("oximedia_watch_batch_out");
        let _ = std::fs::create_dir_all(&watch_dir);

        for i in 0..5 {
            std::fs::write(watch_dir.join(format!("batch{i}.mp4")), &[0xAA; 256]).expect("write");
        }

        let config = make_config(&watch_dir, &output_dir).with_max_concurrent(2);
        let mut wf = WatchFolder::new(config);

        wf.scan().expect("scan");
        wf.scan().expect("scan"); // all become queued

        let batch = wf.next_batch();
        assert!(batch.len() <= 2);

        // Mark one as converting
        if let Some(first) = batch.first() {
            wf.mark_converting(first).expect("mark");
        }

        let batch2 = wf.next_batch();
        assert!(batch2.len() <= 1); // One slot taken

        let _ = std::fs::remove_dir_all(&watch_dir);
    }

    #[test]
    fn test_cleanup_old_entries() {
        let watch_dir = std::env::temp_dir().join("oximedia_watch_cleanup");
        let output_dir = std::env::temp_dir().join("oximedia_watch_cleanup_out");
        let _ = std::fs::create_dir_all(&watch_dir);

        let file_path = watch_dir.join("old.mp4");
        std::fs::write(&file_path, &[0xFF; 256]).expect("write");

        let config = make_config(&watch_dir, &output_dir);
        let mut wf = WatchFolder::new(config);

        wf.scan().expect("scan");
        wf.scan().expect("scan");
        wf.mark_converting(&file_path).expect("mark");
        wf.mark_completed(&file_path).expect("mark");

        assert_eq!(wf.entries().len(), 1);

        // Cleanup with zero duration should remove it
        wf.cleanup_old_entries(Duration::from_secs(0));
        assert!(wf.entries().is_empty());

        let _ = std::fs::remove_dir_all(&watch_dir);
    }

    #[test]
    fn test_reset() {
        let config = WatchConfig::default();
        let mut wf = WatchFolder::new(config);
        wf.stats.files_detected = 10;
        wf.processed.insert(PathBuf::from("test.mp4"));

        wf.reset();

        assert!(wf.entries().is_empty());
        assert_eq!(wf.stats().files_detected, 0);
    }

    #[test]
    fn test_compute_output_path_basic() {
        let tmp = std::env::temp_dir();
        let watch_dir = tmp.join("oximedia-convert-watch-basic");
        let output_dir = tmp.join("oximedia-convert-out-basic");
        let config = WatchConfig::new(watch_dir.clone(), output_dir.clone())
            .with_format(ContainerFormat::Webm);
        let wf = WatchFolder::new(config);

        let input = watch_dir.join("video.mp4");
        let output = wf.compute_output_path(&input);
        assert_eq!(output, output_dir.join("video.webm"));
    }

    #[test]
    fn test_compute_output_path_preserve_structure() {
        let tmp = std::env::temp_dir();
        let watch_dir = tmp.join("oximedia-convert-watch-preserve");
        let output_dir = tmp.join("oximedia-convert-out-preserve");
        let config = WatchConfig::new(watch_dir.clone(), output_dir.clone())
            .with_format(ContainerFormat::Mp4)
            .with_preserve_structure(true);
        let wf = WatchFolder::new(config);

        let input = watch_dir.join("subdir/video.mkv");
        let output = wf.compute_output_path(&input);
        assert_eq!(output, output_dir.join("subdir/video.mp4"));
    }

    #[test]
    fn test_rescan_does_not_duplicate() {
        let watch_dir = std::env::temp_dir().join("oximedia_watch_rescan");
        let output_dir = std::env::temp_dir().join("oximedia_watch_rescan_out");
        let _ = std::fs::create_dir_all(&watch_dir);

        std::fs::write(watch_dir.join("stable.mp4"), &[0xAA; 512]).expect("write");

        let config = make_config(&watch_dir, &output_dir);
        let mut wf = WatchFolder::new(config);

        let first = wf.scan().expect("scan");
        assert_eq!(first, 1);

        let second = wf.scan().expect("scan");
        assert_eq!(second, 0); // Already tracked

        assert_eq!(wf.stats().files_detected, 1); // Only counted once

        let _ = std::fs::remove_dir_all(&watch_dir);
    }

    #[test]
    fn test_mark_unknown_file_fails() {
        let config = WatchConfig::default();
        let mut wf = WatchFolder::new(config);

        assert!(wf.mark_converting(Path::new("/nonexistent.mp4")).is_err());
        assert!(wf.mark_completed(Path::new("/nonexistent.mp4")).is_err());
        assert!(wf
            .mark_failed(Path::new("/nonexistent.mp4"), "err")
            .is_err());
        assert!(wf.mark_skipped(Path::new("/nonexistent.mp4")).is_err());
    }

    #[test]
    fn test_watch_file_status_values() {
        // Ensure all enum variants exist and are distinct
        let statuses = [
            WatchFileStatus::Detected,
            WatchFileStatus::Queued,
            WatchFileStatus::Converting,
            WatchFileStatus::Completed,
            WatchFileStatus::Failed,
            WatchFileStatus::Skipped,
        ];
        for (i, a) in statuses.iter().enumerate() {
            for (j, b) in statuses.iter().enumerate() {
                if i == j {
                    assert_eq!(a, b);
                } else {
                    assert_ne!(a, b);
                }
            }
        }
    }
}
