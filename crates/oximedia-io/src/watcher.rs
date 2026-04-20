//! Lightweight file modification watcher using `std::fs::metadata`.
//!
//! [`FileWatcher`] polls the modification time of a file by comparing the
//! Unix timestamp (seconds since epoch) of the last known modification against
//! the current filesystem metadata. It requires no OS-specific APIs and works
//! on all platforms supported by `std::fs`.
//!
//! # Design
//!
//! - **Poll-based**: No background threads or OS file-watch APIs are used.
//!   Callers drive the check frequency by how often they call [`poll`](FileWatcher::poll).
//! - **Modification time resolution**: Uses [`std::time::SystemTime`] with
//!   second-level resolution for maximum portability.
//! - Returns `Some(new_mtime)` if the file has been modified since `last_modified`,
//!   or `None` if unchanged (or if the file cannot be stat'd).
//!
//! # Example
//!
//! ```no_run
//! use oximedia_io::watcher::FileWatcher;
//!
//! let watcher = FileWatcher::new("/tmp/playlist.m3u8");
//! let mut last = 0u64;
//!
//! loop {
//!     if let Some(new_ts) = watcher.poll(last) {
//!         println!("file changed at unix ts {new_ts}");
//!         last = new_ts;
//!     }
//!     std::thread::sleep(std::time::Duration::from_secs(1));
//! }
//! ```

#![allow(dead_code)]

use std::path::PathBuf;
use std::time::UNIX_EPOCH;

/// A poll-based file watcher that detects modifications via `mtime`.
#[derive(Debug, Clone)]
pub struct FileWatcher {
    /// Path to the file being watched.
    path: PathBuf,
}

impl FileWatcher {
    /// Create a new `FileWatcher` for the file at `path`.
    ///
    /// The watcher does not open the file; it only stores the path.
    /// The first poll call will stat the file.
    #[must_use]
    pub fn new(path: &str) -> Self {
        Self {
            path: PathBuf::from(path),
        }
    }

    /// Create a watcher from any `Into<PathBuf>`.
    #[must_use]
    pub fn from_path(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    /// Poll the file for changes.
    ///
    /// Reads the file's modification time using `std::fs::metadata`.
    ///
    /// - If the file cannot be stat'd (missing, permissions, etc.) returns `None`.
    /// - If the current modification time (seconds since UNIX epoch) is **greater than**
    ///   `last_modified`, returns `Some(new_mtime)`.
    /// - Otherwise returns `None` (no change detected).
    ///
    /// `last_modified` should be `0` on the first call to detect any existing file.
    #[must_use]
    pub fn poll(&self, last_modified: u64) -> Option<u64> {
        let mtime = self.current_mtime()?;
        if mtime > last_modified {
            Some(mtime)
        } else {
            None
        }
    }

    /// Return the current modification timestamp of the watched file in seconds
    /// since the UNIX epoch, or `None` if the file cannot be stat'd.
    #[must_use]
    pub fn current_mtime(&self) -> Option<u64> {
        let meta = std::fs::metadata(&self.path).ok()?;
        let modified = meta.modified().ok()?;
        let duration = modified.duration_since(UNIX_EPOCH).ok()?;
        Some(duration.as_secs())
    }

    /// Returns the path this watcher monitors.
    #[must_use]
    pub fn path(&self) -> &PathBuf {
        &self.path
    }

    /// Returns `true` if the watched file currently exists.
    #[must_use]
    pub fn exists(&self) -> bool {
        self.path.exists()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_watcher_nonexistent_file_returns_none() {
        let p = std::env::temp_dir().join("oximedia-io-watcher-nonexistent_xyz.bin");
        let w = FileWatcher::from_path(&p);
        assert!(w.poll(0).is_none());
        assert!(w.current_mtime().is_none());
        assert!(!w.exists());
    }

    #[test]
    fn test_watcher_detects_existing_file() {
        let dir = std::env::temp_dir();
        let path = dir.join("oximedia_watcher_test.bin");
        std::fs::write(&path, b"data").expect("write");

        let w = FileWatcher::from_path(&path);
        assert!(w.exists());

        // Polling with last_modified=0 should detect the file (mtime > 0)
        let result = w.poll(0);
        assert!(result.is_some(), "should detect file with mtime > 0");

        // Polling with the current mtime should return None (no change)
        let mtime = result.expect("mtime available");
        assert!(w.poll(mtime).is_none());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_watcher_path_accessor() {
        let p = std::env::temp_dir().join("oximedia-io-watcher-test.mp4");
        let w = FileWatcher::from_path(&p);
        assert_eq!(w.path(), &p);
    }

    #[test]
    fn test_watcher_future_last_modified_returns_none() {
        let dir = std::env::temp_dir();
        let path = dir.join("oximedia_watcher_future_test.bin");
        std::fs::write(&path, b"future").expect("write");

        let w = FileWatcher::from_path(&path);
        // last_modified set far in the future should return None
        assert!(w.poll(u64::MAX).is_none());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_watcher_current_mtime_returns_reasonable_timestamp() {
        let dir = std::env::temp_dir();
        let path = dir.join("oximedia_watcher_mtime.bin");
        std::fs::write(&path, b"ts").expect("write");

        let w = FileWatcher::from_path(&path);
        let mtime = w.current_mtime().expect("should have mtime");

        // Verify the timestamp is after year 2000 (946684800 = 2000-01-01 UTC)
        assert!(
            mtime > 946_684_800,
            "mtime should be a plausible Unix timestamp"
        );

        let now = std::time::SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        assert!(mtime <= now + 2, "mtime should not be in the future");

        let _ = std::fs::remove_file(&path);
    }
}
