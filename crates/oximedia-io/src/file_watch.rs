#![allow(dead_code)]
//! File system event watching utilities.
//!
//! Provides polling-based file and directory watching with:
//! - Per-path change detection (create/modify/delete)
//! - Recursive directory scanning
//! - Debounce to suppress rapid repeated events for the same path

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

/// Events that can be reported for a watched file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FileEvent {
    /// The file was created.
    Created(PathBuf),
    /// The file was modified (content or metadata changed).
    Modified(PathBuf),
    /// The file was deleted.
    Deleted(PathBuf),
    /// The file was renamed to the contained path.
    Renamed(PathBuf, PathBuf),
}

impl FileEvent {
    /// Return `true` if this event represents a content or metadata modification.
    #[must_use]
    pub fn is_modification(&self) -> bool {
        matches!(self, FileEvent::Modified(_))
    }

    /// Return the primary path associated with this event.
    #[must_use]
    pub fn path(&self) -> &Path {
        match self {
            FileEvent::Created(p)
            | FileEvent::Modified(p)
            | FileEvent::Deleted(p)
            | FileEvent::Renamed(p, _) => p.as_path(),
        }
    }
}

/// Configuration controlling which paths and event types the watcher monitors.
#[derive(Debug, Clone)]
pub struct WatchConfig {
    /// File extensions to watch (e.g. `["rs", "toml"]`). Empty means watch all.
    pub extensions: Vec<String>,
    /// Whether to emit events for file creation.
    pub watch_create: bool,
    /// Whether to emit events for file modification.
    pub watch_modify: bool,
    /// Whether to emit events for file deletion.
    pub watch_delete: bool,
    /// Minimum interval between repeated events for the same path.
    pub debounce: Duration,
    /// Whether to recursively scan subdirectories when watching a directory.
    pub recursive: bool,
}

impl Default for WatchConfig {
    fn default() -> Self {
        Self {
            extensions: Vec::new(),
            watch_create: true,
            watch_modify: true,
            watch_delete: true,
            debounce: Duration::from_millis(50),
            recursive: false,
        }
    }
}

impl WatchConfig {
    /// Return `true` if the given path should be watched according to this config.
    #[must_use]
    pub fn should_watch(&self, path: &Path) -> bool {
        if self.extensions.is_empty() {
            return true;
        }
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            self.extensions.iter().any(|e| e == ext)
        } else {
            false
        }
    }
}

/// Entry stored per-path to detect changes between poll ticks.
#[derive(Debug, Clone)]
struct WatchEntry {
    last_modified: Option<SystemTime>,
    last_len: u64,
    /// Wall-clock time at which the last event for this path was emitted.
    last_event_at: Option<Instant>,
}

/// A polling-based file watcher that detects changes by comparing metadata.
///
/// Supports recursive directory watching and debouncing: rapid successive
/// changes to a path are collapsed into a single event until `debounce`
/// milliseconds have elapsed since the last emitted event.
#[derive(Debug)]
pub struct FileWatcher {
    config: WatchConfig,
    watched: HashMap<PathBuf, WatchEntry>,
    /// Directories that should be recursively scanned on each poll tick.
    watched_dirs: Vec<PathBuf>,
    pending_events: Vec<FileEvent>,
}

impl FileWatcher {
    /// Create a new `FileWatcher` with the given configuration.
    #[must_use]
    pub fn new(config: WatchConfig) -> Self {
        Self {
            config,
            watched: HashMap::new(),
            watched_dirs: Vec::new(),
            pending_events: Vec::new(),
        }
    }

    /// Create a watcher with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(WatchConfig::default())
    }

    /// Add a path to the watch list.
    ///
    /// If the path exists on disk its current metadata is recorded as a baseline.
    /// Paths that do not satisfy [`WatchConfig::should_watch`] are silently ignored.
    ///
    /// If `path` is a directory and `config.recursive` is `true`, all files
    /// inside it (recursively) are also added, and the directory itself is
    /// registered for new-file detection on subsequent [`check_events`](Self::check_events) calls.
    pub fn add_path(&mut self, path: impl Into<PathBuf>) {
        let path = path.into();

        if path.is_dir() {
            if self.config.recursive {
                // Register the directory for new-file scanning
                if !self.watched_dirs.contains(&path) {
                    self.watched_dirs.push(path.clone());
                }
                // Add all current files in the directory tree
                self.scan_directory_into_watched(&path);
            }
            return;
        }

        if !self.config.should_watch(&path) {
            return;
        }
        let entry = build_watch_entry(&path);
        self.watched.insert(path, entry);
    }

    /// Recursively scan `dir` and add all files matching the filter.
    fn scan_directory_into_watched(&mut self, dir: &Path) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let child = entry.path();
            if child.is_dir() {
                self.scan_directory_into_watched(&child);
            } else if self.config.should_watch(&child) && !self.watched.contains_key(&child) {
                let watch_entry = build_watch_entry(&child);
                self.watched.insert(child, watch_entry);
            }
        }
    }

    /// Check for newly-created files in watched directories.
    fn check_new_files_in_dirs(&mut self) {
        let dirs: Vec<PathBuf> = self.watched_dirs.clone();
        for dir in &dirs {
            self.scan_new_files_in_dir(dir);
        }
    }

    /// Scan one directory (recursively) and emit Created events for new files.
    fn scan_new_files_in_dir(&mut self, dir: &Path) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let child = entry.path();
            if child.is_dir() {
                // Add to watched_dirs if not already there, then recurse
                if !self.watched_dirs.contains(&child) {
                    self.watched_dirs.push(child.clone());
                }
                self.scan_new_files_in_dir(&child);
            } else if self.config.should_watch(&child) && !self.watched.contains_key(&child) {
                // Brand-new file found
                if self.config.watch_create && self.debounce_ok(&child) {
                    self.pending_events.push(FileEvent::Created(child.clone()));
                }
                let watch_entry = build_watch_entry(&child);
                self.watched.insert(child, watch_entry);
            }
        }
    }

    /// Return `true` if enough time has passed since the last event for `path`.
    fn debounce_ok(&self, path: &Path) -> bool {
        if let Some(entry) = self.watched.get(path) {
            if let Some(last) = entry.last_event_at {
                return last.elapsed() >= self.config.debounce;
            }
        }
        true // no prior event — allow
    }

    /// Poll all watched paths for changes and update the internal event queue.
    ///
    /// Call [`event_count`](Self::event_count) or consume events afterwards.
    ///
    /// If recursive watching is enabled, newly-created files inside watched
    /// directories are also detected and emitted as `Created` events.
    ///
    /// # Panics
    ///
    /// Panics if internal state is inconsistent (path registered but not in watched map).
    pub fn check_events(&mut self) {
        // First scan for new files in watched directories (recursive mode)
        if self.config.recursive {
            self.check_new_files_in_dirs();
        }

        let now = Instant::now();
        let paths: Vec<PathBuf> = self.watched.keys().cloned().collect();
        for path in paths {
            if let Ok(meta) = std::fs::metadata(&path) {
                let new_modified = meta.modified().ok();
                let new_len = meta.len();
                let entry = self
                    .watched
                    .get_mut(&path)
                    .expect("invariant: path came from watched map");
                let changed = entry.last_modified != new_modified || entry.last_len != new_len;
                if changed && self.config.watch_modify {
                    // Apply debounce: only emit if enough time has passed
                    let should_emit = entry
                        .last_event_at
                        .map_or(true, |t| t.elapsed() >= self.config.debounce);
                    if should_emit {
                        self.pending_events.push(FileEvent::Modified(path.clone()));
                        // Record that an event was emitted now
                        if let Some(e) = self.watched.get_mut(&path) {
                            e.last_event_at = Some(now);
                        }
                    }
                }
                let entry = self
                    .watched
                    .get_mut(&path)
                    .expect("invariant: path came from watched map");
                entry.last_modified = new_modified;
                entry.last_len = new_len;
            } else {
                // File has disappeared since we last saw it.
                let entry = self
                    .watched
                    .get(&path)
                    .expect("invariant: path came from watched map");
                let was_present = entry.last_modified.is_some();
                let should_emit = entry
                    .last_event_at
                    .map_or(true, |t| t.elapsed() >= self.config.debounce);
                if was_present && self.config.watch_delete && should_emit {
                    self.pending_events.push(FileEvent::Deleted(path.clone()));
                    if let Some(e) = self.watched.get_mut(&path) {
                        e.last_event_at = Some(now);
                    }
                }
                let entry = self
                    .watched
                    .get_mut(&path)
                    .expect("invariant: path came from watched map");
                entry.last_modified = None;
                entry.last_len = 0;
            }
        }
    }

    /// Return the number of pending (unconsumed) events.
    #[must_use]
    pub fn event_count(&self) -> usize {
        self.pending_events.len()
    }

    /// Drain and return all pending events.
    pub fn drain_events(&mut self) -> Vec<FileEvent> {
        std::mem::take(&mut self.pending_events)
    }

    /// Return the number of individual file paths currently being watched.
    #[must_use]
    pub fn watched_count(&self) -> usize {
        self.watched.len()
    }

    /// Return the number of directories registered for recursive scanning.
    #[must_use]
    pub fn watched_dir_count(&self) -> usize {
        self.watched_dirs.len()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Private helpers
// ──────────────────────────────────────────────────────────────────────────────

fn build_watch_entry(path: &Path) -> WatchEntry {
    if let Ok(meta) = std::fs::metadata(path) {
        WatchEntry {
            last_modified: meta.modified().ok(),
            last_len: meta.len(),
            last_event_at: None,
        }
    } else {
        WatchEntry {
            last_modified: None,
            last_len: 0,
            last_event_at: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;

    #[test]
    fn test_file_event_is_modification_true() {
        let ev = FileEvent::Modified(PathBuf::from("foo.rs"));
        assert!(ev.is_modification());
    }

    #[test]
    fn test_file_event_is_modification_false_created() {
        let ev = FileEvent::Created(PathBuf::from("foo.rs"));
        assert!(!ev.is_modification());
    }

    #[test]
    fn test_file_event_is_modification_false_deleted() {
        let ev = FileEvent::Deleted(PathBuf::from("foo.rs"));
        assert!(!ev.is_modification());
    }

    #[test]
    fn test_file_event_is_modification_false_renamed() {
        let ev = FileEvent::Renamed(PathBuf::from("a.rs"), PathBuf::from("b.rs"));
        assert!(!ev.is_modification());
    }

    #[test]
    fn test_file_event_path() {
        let p = PathBuf::from("video.mp4");
        let ev = FileEvent::Modified(p.clone());
        assert_eq!(ev.path(), p.as_path());
    }

    #[test]
    fn test_watch_config_should_watch_all_extensions() {
        let cfg = WatchConfig {
            extensions: vec![],
            ..Default::default()
        };
        assert!(cfg.should_watch(Path::new("any_file.xyz")));
    }

    #[test]
    fn test_watch_config_should_watch_matching_ext() {
        let cfg = WatchConfig {
            extensions: vec!["rs".to_string(), "toml".to_string()],
            ..Default::default()
        };
        assert!(cfg.should_watch(Path::new("lib.rs")));
        assert!(cfg.should_watch(Path::new("Cargo.toml")));
    }

    #[test]
    fn test_watch_config_should_not_watch_non_matching_ext() {
        let cfg = WatchConfig {
            extensions: vec!["rs".to_string()],
            ..Default::default()
        };
        assert!(!cfg.should_watch(Path::new("video.mp4")));
    }

    #[test]
    fn test_watch_config_should_not_watch_no_ext() {
        let cfg = WatchConfig {
            extensions: vec!["rs".to_string()],
            ..Default::default()
        };
        assert!(!cfg.should_watch(Path::new("Makefile")));
    }

    #[test]
    fn test_add_path_nonexistent_is_tracked() {
        let mut watcher = FileWatcher::with_defaults();
        watcher.add_path("/nonexistent/path/file.rs");
        assert_eq!(watcher.watched_count(), 1);
    }

    #[test]
    fn test_add_path_filtered_by_extension() {
        let cfg = WatchConfig {
            extensions: vec!["rs".to_string()],
            ..Default::default()
        };
        let mut watcher = FileWatcher::new(cfg);
        watcher.add_path(std::env::temp_dir().join("oximedia-io-fwatch-video.mp4"));
        assert_eq!(watcher.watched_count(), 0);
    }

    #[test]
    fn test_event_count_initially_zero() {
        let watcher = FileWatcher::with_defaults();
        assert_eq!(watcher.event_count(), 0);
    }

    #[test]
    fn test_check_events_no_watched_paths() {
        let mut watcher = FileWatcher::with_defaults();
        watcher.check_events();
        assert_eq!(watcher.event_count(), 0);
    }

    #[test]
    fn test_modification_detected() {
        let dir = tempfile::tempdir().expect("failed to create temp file");
        let path = dir.path().join("test.rs");
        {
            let mut f = std::fs::File::create(&path).expect("failed to create file");
            f.write_all(b"hello").expect("failed to write");
        }

        let mut watcher = FileWatcher::with_defaults();
        watcher.add_path(&path);

        // Modify the file.
        {
            let mut f = std::fs::OpenOptions::new()
                .append(true)
                .open(&path)
                .expect("operation should succeed");
            f.write_all(b" world").expect("failed to write");
        }

        watcher.check_events();
        // At least one modification event should have been recorded.
        assert!(watcher.event_count() >= 1);
        let events = watcher.drain_events();
        assert!(events.iter().any(|e| e.is_modification()));
    }

    #[test]
    fn test_drain_events_clears_queue() {
        let mut watcher = FileWatcher::with_defaults();
        // Manually push a synthetic event.
        watcher
            .pending_events
            .push(FileEvent::Created(PathBuf::from("x.rs")));
        assert_eq!(watcher.event_count(), 1);
        let drained = watcher.drain_events();
        assert_eq!(drained.len(), 1);
        assert_eq!(watcher.event_count(), 0);
    }
}
