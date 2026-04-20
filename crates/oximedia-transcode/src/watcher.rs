//! Watch folder automation for automatic transcoding of new files.
//!
//! This module provides [`TranscodeWatcher`] which monitors a directory for
//! newly-created media files and exposes them for downstream transcoding.
//! Stable-file detection prevents processing files that are still being written.

use std::collections::HashSet;

/// Configuration for a watch folder.
#[derive(Debug, Clone)]
pub struct WatchConfig {
    /// Directory to monitor for new files.
    pub watch_dir: String,
    /// Directory where transcoded output files should be written.
    pub output_dir: String,
    /// Name of the transcode preset to apply to discovered files.
    pub preset_name: String,
    /// How often to poll the watch directory, in milliseconds.
    pub poll_interval_ms: u64,
}

impl WatchConfig {
    /// Creates a new [`WatchConfig`] with the given paths and preset.
    #[must_use]
    pub fn new(
        watch_dir: impl Into<String>,
        output_dir: impl Into<String>,
        preset_name: impl Into<String>,
        poll_interval_ms: u64,
    ) -> Self {
        Self {
            watch_dir: watch_dir.into(),
            output_dir: output_dir.into(),
            preset_name: preset_name.into(),
            poll_interval_ms,
        }
    }
}

/// A file discovered by the watcher.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WatchedFile {
    /// Absolute path of the discovered file.
    pub path: String,
    /// File size in bytes at the time of discovery.
    pub size_bytes: u64,
    /// Monotonic-style timestamp (milliseconds) when the file was first seen.
    pub discovered_at_ms: u64,
}

impl WatchedFile {
    /// Creates a new [`WatchedFile`] record.
    #[must_use]
    pub fn new(path: impl Into<String>, size_bytes: u64, discovered_at_ms: u64) -> Self {
        Self {
            path: path.into(),
            size_bytes,
            discovered_at_ms,
        }
    }

    /// Returns `true` when the file has been stable long enough to be
    /// processed safely.
    ///
    /// A file is considered stable when `elapsed_since_discovery_ms >=
    /// min_stable_ms`.  Callers should re-check the file size between polls
    /// and reset the discovery timestamp if it changes.
    #[must_use]
    pub fn is_stable(&self, elapsed_since_discovery_ms: u64, min_stable_ms: u64) -> bool {
        elapsed_since_discovery_ms >= min_stable_ms
    }
}

/// Watch-folder automation engine.
///
/// [`TranscodeWatcher`] tracks which files it has already seen so that each
/// new file is emitted exactly once.  The actual filesystem walk is performed
/// in [`scan_for_new_files`](TranscodeWatcher::scan_for_new_files); on a real
/// system this calls `std::fs::read_dir` — the simulation path returns an
/// empty vec so the module compiles and tests without a live filesystem.
#[derive(Debug)]
pub struct TranscodeWatcher {
    /// Watch folder configuration.
    pub config: WatchConfig,
    /// Set of file paths that have already been discovered (and optionally processed).
    pub seen_files: HashSet<String>,
}

impl TranscodeWatcher {
    /// Creates a new [`TranscodeWatcher`] with an empty seen-files set.
    #[must_use]
    pub fn new(config: WatchConfig) -> Self {
        Self {
            config,
            seen_files: HashSet::new(),
        }
    }

    /// Scans the watch directory for new files and returns those not yet seen.
    ///
    /// `now_ms` is the caller-supplied current time in milliseconds and is
    /// recorded as the discovery timestamp for each new file.
    ///
    /// The default implementation performs an actual `read_dir` scan.  On
    /// platforms where the watch directory does not exist the function returns
    /// an empty vec rather than propagating an error, so the polling loop
    /// continues gracefully.
    #[must_use]
    pub fn scan_for_new_files(&mut self, now_ms: u64) -> Vec<WatchedFile> {
        let mut new_files = Vec::new();

        let read_result = std::fs::read_dir(&self.config.watch_dir);
        let entries = match read_result {
            Ok(rd) => rd,
            Err(_) => return new_files,
        };

        for entry_result in entries {
            let entry = match entry_result {
                Ok(e) => e,
                Err(_) => continue,
            };

            let metadata = match entry.metadata() {
                Ok(m) => m,
                Err(_) => continue,
            };

            // Only consider regular files.
            if !metadata.is_file() {
                continue;
            }

            let path = entry.path();
            let path_str = path.to_string_lossy().into_owned();

            if self.seen_files.contains(&path_str) {
                continue;
            }

            let size_bytes = metadata.len();
            let watched = WatchedFile::new(path_str.clone(), size_bytes, now_ms);
            self.seen_files.insert(path_str);
            new_files.push(watched);
        }

        new_files
    }

    /// Marks a file path as processed so it is not returned again by
    /// [`scan_for_new_files`](TranscodeWatcher::scan_for_new_files).
    pub fn mark_processed(&mut self, path: &str) {
        self.seen_files.insert(path.to_owned());
    }

    /// Returns `true` if the given path has already been seen by this watcher.
    #[must_use]
    pub fn is_known(&self, path: &str) -> bool {
        self.seen_files.contains(path)
    }

    /// Returns the configured poll interval in milliseconds.
    #[must_use]
    pub fn poll_interval_ms(&self) -> u64 {
        self.config.poll_interval_ms
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_str(name: &str) -> String {
        std::env::temp_dir()
            .join(format!("oximedia-transcode-watcher-{name}"))
            .to_string_lossy()
            .into_owned()
    }

    fn make_config() -> WatchConfig {
        WatchConfig::new(
            tmp_str("watch_in"),
            tmp_str("watch_out"),
            "youtube_1080p",
            2000,
        )
    }

    #[test]
    fn test_watcher_starts_empty() {
        let watcher = TranscodeWatcher::new(make_config());
        assert!(watcher.seen_files.is_empty());
    }

    #[test]
    fn test_mark_processed_and_is_known() {
        let mut watcher = TranscodeWatcher::new(make_config());
        let path = tmp_str("watch_in/video.mp4");
        assert!(!watcher.is_known(&path));
        watcher.mark_processed(&path);
        assert!(watcher.is_known(&path));
    }

    #[test]
    fn test_scan_nonexistent_dir_returns_empty() {
        let config = WatchConfig::new(
            "/nonexistent_oximedia_watch_dir_xyz",
            tmp_str("out"),
            "preset",
            1000,
        );
        let mut watcher = TranscodeWatcher::new(config);
        let found = watcher.scan_for_new_files(12345);
        assert!(found.is_empty());
    }

    #[test]
    fn test_scan_real_dir_does_not_duplicate() {
        let tmp = std::env::temp_dir();
        let watch_dir = tmp.join("oximedia_watcher_test");
        let _ = std::fs::create_dir_all(&watch_dir);

        // Create a dummy file.
        let file_path = watch_dir.join("test_video.mp4");
        std::fs::write(&file_path, b"fake mp4 data").ok();

        let config = WatchConfig::new(
            watch_dir.to_string_lossy().as_ref(),
            tmp_str("out"),
            "preset",
            1000,
        );
        let mut watcher = TranscodeWatcher::new(config);

        let first_scan = watcher.scan_for_new_files(1000);
        assert_eq!(first_scan.len(), 1);

        // Second scan must not return the same file again.
        let second_scan = watcher.scan_for_new_files(2000);
        assert_eq!(second_scan.len(), 0);

        // Cleanup.
        let _ = std::fs::remove_file(&file_path);
        let _ = std::fs::remove_dir(&watch_dir);
    }

    #[test]
    fn test_watched_file_is_stable() {
        let f = WatchedFile::new(tmp_str("video.mp4"), 1024, 0);
        assert!(!f.is_stable(4999, 5000));
        assert!(f.is_stable(5000, 5000));
        assert!(f.is_stable(9999, 5000));
    }

    #[test]
    fn test_watched_file_fields() {
        let path = tmp_str("a.mkv");
        let f = WatchedFile::new(&path, 999_000, 42_000);
        assert_eq!(f.path, path);
        assert_eq!(f.size_bytes, 999_000);
        assert_eq!(f.discovered_at_ms, 42_000);
    }

    #[test]
    fn test_config_fields() {
        let c = WatchConfig::new("/in", "/out", "vimeo_4k", 500);
        assert_eq!(c.watch_dir, "/in");
        assert_eq!(c.output_dir, "/out");
        assert_eq!(c.preset_name, "vimeo_4k");
        assert_eq!(c.poll_interval_ms, 500);
    }

    #[test]
    fn test_poll_interval_accessor() {
        let watcher = TranscodeWatcher::new(make_config());
        assert_eq!(watcher.poll_interval_ms(), 2000);
    }
}
