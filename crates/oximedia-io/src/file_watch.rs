#![allow(dead_code)]
//! File system event watching utilities.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

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
}

impl Default for WatchConfig {
    fn default() -> Self {
        Self {
            extensions: Vec::new(),
            watch_create: true,
            watch_modify: true,
            watch_delete: true,
            debounce: Duration::from_millis(50),
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
}

/// A polling-based file watcher that detects changes by comparing metadata.
#[derive(Debug)]
pub struct FileWatcher {
    config: WatchConfig,
    watched: HashMap<PathBuf, WatchEntry>,
    pending_events: Vec<FileEvent>,
}

impl FileWatcher {
    /// Create a new `FileWatcher` with the given configuration.
    #[must_use]
    pub fn new(config: WatchConfig) -> Self {
        Self {
            config,
            watched: HashMap::new(),
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
    pub fn add_path(&mut self, path: impl Into<PathBuf>) {
        let path = path.into();
        if !self.config.should_watch(&path) {
            return;
        }
        let entry = if let Ok(meta) = std::fs::metadata(&path) {
            WatchEntry {
                last_modified: meta.modified().ok(),
                last_len: meta.len(),
            }
        } else {
            WatchEntry {
                last_modified: None,
                last_len: 0,
            }
        };
        self.watched.insert(path, entry);
    }

    /// Poll all watched paths for changes and update the internal event queue.
    ///
    /// Call [`event_count`](Self::event_count) or consume events afterwards.
    ///
    /// # Panics
    ///
    /// Panics if internal state is inconsistent (path registered but not in watched map).
    pub fn check_events(&mut self) {
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
                    self.pending_events.push(FileEvent::Modified(path.clone()));
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
                if entry.last_modified.is_some() && self.config.watch_delete {
                    self.pending_events.push(FileEvent::Deleted(path.clone()));
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

    /// Return the number of paths currently being watched.
    #[must_use]
    pub fn watched_count(&self) -> usize {
        self.watched.len()
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
        watcher.add_path("/tmp/video.mp4");
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
