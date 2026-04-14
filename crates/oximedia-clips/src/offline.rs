//! Offline / missing media detection for clips.
//!
//! [`ClipOfflineDetector`] checks whether the media file associated with a
//! clip is accessible on the local filesystem.  A clip is considered
//! **online** when its media path exists as a regular file; it is considered
//! **offline** (missing) otherwise.
//!
//! # Example
//!
//! ```no_run
//! use oximedia_clips::offline::ClipOfflineDetector;
//!
//! // Returns true if the file exists at the given path.
//! let is_online = ClipOfflineDetector::check(42, "/media/footage/shot001.mov");
//! println!("clip 42 online: {is_online}");
//! ```

#![allow(dead_code)]

use std::path::Path;

// ─────────────────────────────────────────────────────────────────────────────
// ClipOfflineDetector
// ─────────────────────────────────────────────────────────────────────────────

/// Stateless helper that checks whether clip media files are accessible.
pub struct ClipOfflineDetector;

impl ClipOfflineDetector {
    /// Check whether the media for `clip_id` is available at `media_path`.
    ///
    /// Returns `true` when `media_path` points to an existing file (the clip
    /// is **online**), `false` when the file is missing or the path is a
    /// directory (the clip is **offline**).
    ///
    /// The `clip_id` parameter is accepted for API consistency and logging but
    /// is not used in the filesystem check itself.
    pub fn check(_clip_id: u64, media_path: &str) -> bool {
        let path = Path::new(media_path);
        path.exists() && path.is_file()
    }

    /// Check a list of `(clip_id, media_path)` pairs and return only the IDs
    /// of clips whose media is offline (missing).
    pub fn offline_clips(pairs: &[(u64, &str)]) -> Vec<u64> {
        pairs
            .iter()
            .filter_map(|(id, path)| {
                if !Self::check(*id, path) {
                    Some(*id)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Return `true` when all supplied clips are online.
    pub fn all_online(pairs: &[(u64, &str)]) -> bool {
        Self::offline_clips(pairs).is_empty()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Create a temporary file in `std::env::temp_dir()` and return its path.
    fn create_temp_file(name: &str) -> String {
        let mut path = std::env::temp_dir();
        path.push(name);
        let mut f = std::fs::File::create(&path).expect("create temp file");
        f.write_all(b"oximedia test").expect("write temp content");
        path.to_str().expect("path str").to_string()
    }

    #[test]
    fn test_check_existing_file_returns_true() {
        let path = create_temp_file("oximedia_offline_test_a.tmp");
        assert!(ClipOfflineDetector::check(1, &path));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_check_missing_file_returns_false() {
        let path = {
            let mut p = std::env::temp_dir();
            p.push("oximedia_does_not_exist_12345_offline.mov");
            p.to_str().expect("path str").to_string()
        };
        // Make sure it doesn't exist.
        let _ = std::fs::remove_file(&path);
        assert!(!ClipOfflineDetector::check(2, &path));
    }

    #[test]
    fn test_check_directory_returns_false() {
        let dir = std::env::temp_dir();
        let dir_str = dir.to_str().expect("path str");
        assert!(!ClipOfflineDetector::check(3, dir_str));
    }

    #[test]
    fn test_offline_clips_empty_when_all_online() {
        let path = create_temp_file("oximedia_offline_test_b.tmp");
        let pairs = vec![(10u64, path.as_str())];
        let offline = ClipOfflineDetector::offline_clips(&pairs);
        assert!(offline.is_empty());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_offline_clips_detects_missing() {
        let missing = {
            let mut p = std::env::temp_dir();
            p.push("oximedia_missing_clip_xyz_offline.mov");
            // Ensure it does not exist.
            let _ = std::fs::remove_file(&p);
            p.to_str().expect("path str").to_string()
        };
        let pairs = vec![(42u64, missing.as_str())];
        let offline = ClipOfflineDetector::offline_clips(&pairs);
        assert_eq!(offline, vec![42]);
    }

    #[test]
    fn test_all_online_true_when_all_files_exist() {
        let path = create_temp_file("oximedia_offline_test_c.tmp");
        let pairs = vec![(1u64, path.as_str())];
        assert!(ClipOfflineDetector::all_online(&pairs));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_all_online_false_when_some_missing() {
        let path = create_temp_file("oximedia_offline_test_d.tmp");
        let missing = {
            let mut p = std::env::temp_dir();
            p.push("oximedia_definitely_missing_abc_offline.mov");
            let _ = std::fs::remove_file(&p);
            p.to_str().expect("path str").to_string()
        };
        let pairs = vec![(1u64, path.as_str()), (2u64, missing.as_str())];
        assert!(!ClipOfflineDetector::all_online(&pairs));
        let _ = std::fs::remove_file(&path);
    }
}
