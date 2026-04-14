//! Archive integrity scanning.
//!
//! `ArchiveIntegrityScanner` verifies each entry in an archive by recomputing
//! its checksum from raw data and comparing it against the stored expected
//! checksum.  It returns a list of `(path, is_ok)` pairs.
//!
//! # Example
//! ```rust
//! use oximedia_archive::integrity::{ArchiveIntegrityScanner, ScanEntry};
//!
//! let entries = vec![
//!     ScanEntry::new("video.mkv", "abc123", b"fake content"),
//! ];
//! let results = ArchiveIntegrityScanner::scan(&entries);
//! // Will show false because "fake content" does not hash to "abc123"
//! assert_eq!(results.len(), 1);
//! ```

#![allow(dead_code)]

/// An entry to be integrity-checked.
///
/// The scanner recomputes a simple checksum over `data` and compares it to
/// `expected_checksum`.  For real archives the `data` field would be populated
/// by reading the file from storage; here it is held in-memory for testing.
#[derive(Debug, Clone)]
pub struct ScanEntry {
    /// Logical path of the entry within the archive.
    pub path: String,
    /// Pre-recorded checksum (hex string, any algorithm).
    pub expected_checksum: String,
    /// Raw content bytes (used to verify the checksum).
    pub data: Vec<u8>,
}

impl ScanEntry {
    /// Construct a new `ScanEntry`.
    #[must_use]
    pub fn new(path: &str, expected_checksum: &str, data: &[u8]) -> Self {
        Self {
            path: path.to_string(),
            expected_checksum: expected_checksum.to_string(),
            data: data.to_vec(),
        }
    }

    /// Build a `ScanEntry` with a correct checksum (useful for test setup).
    #[must_use]
    pub fn with_correct_checksum(path: &str, data: &[u8]) -> Self {
        let checksum = compute_checksum(data);
        Self {
            path: path.to_string(),
            expected_checksum: checksum,
            data: data.to_vec(),
        }
    }
}

/// Integrity scanner for archive entries.
pub struct ArchiveIntegrityScanner;

impl ArchiveIntegrityScanner {
    /// Scan every entry and verify its checksum.
    ///
    /// # Arguments
    /// * `entries` – entries to scan.  Each entry's data is checksummed and
    ///               compared to its `expected_checksum`.
    ///
    /// # Returns
    /// A `Vec<(String, bool)>` where the `String` is the entry path and the
    /// `bool` is `true` when the entry passes (checksum matches) or `false`
    /// when it fails (checksum mismatch or entry data is absent/empty and
    /// the expected checksum is non-empty).
    #[must_use]
    pub fn scan(entries: &[ScanEntry]) -> Vec<(String, bool)> {
        entries
            .iter()
            .map(|entry| {
                let actual = compute_checksum(&entry.data);
                let ok = actual == entry.expected_checksum;
                (entry.path.clone(), ok)
            })
            .collect()
    }

    /// Return only the entries that **failed** the integrity check.
    #[must_use]
    pub fn failed_entries(entries: &[ScanEntry]) -> Vec<String> {
        Self::scan(entries)
            .into_iter()
            .filter_map(|(path, ok)| if ok { None } else { Some(path) })
            .collect()
    }
}

/// Compute a lightweight hex checksum for the given bytes.
///
/// Uses a rolling CRC-32-like polynomial for speed.  For production use,
/// callers should prefer the full BLAKE3 / SHA-256 pipeline in `parallel_checksum`.
#[must_use]
pub fn compute_checksum(data: &[u8]) -> String {
    // Adler-32 variant: fast, deterministic, good enough for in-process integrity checks.
    let mut s1: u32 = 1;
    let mut s2: u32 = 0;
    for &byte in data {
        s1 = s1.wrapping_add(byte as u32) % 65521;
        s2 = s2.wrapping_add(s1) % 65521;
    }
    let checksum = (s2 << 16) | s1;
    format!("{checksum:08x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scan_passing_entry() {
        let data = b"hello archive";
        let cs = compute_checksum(data);
        let entries = vec![ScanEntry::new("file.txt", &cs, data)];
        let results = ArchiveIntegrityScanner::scan(&entries);
        assert_eq!(results.len(), 1);
        assert!(results[0].1, "expected entry to pass integrity check");
    }

    #[test]
    fn test_scan_failing_entry() {
        let entries = vec![ScanEntry::new(
            "corrupt.mkv",
            "deadbeef00000000",
            b"wrong data",
        )];
        let results = ArchiveIntegrityScanner::scan(&entries);
        assert_eq!(results.len(), 1);
        assert!(!results[0].1, "expected entry to fail integrity check");
    }

    #[test]
    fn test_scan_empty() {
        let results = ArchiveIntegrityScanner::scan(&[]);
        assert!(results.is_empty());
    }

    #[test]
    fn test_failed_entries_returns_paths() {
        let data = b"valid";
        let valid = ScanEntry::with_correct_checksum("ok.mp4", data);
        let corrupt = ScanEntry::new("bad.mp4", "00000000", data);
        let entries = vec![valid, corrupt];
        let failed = ArchiveIntegrityScanner::failed_entries(&entries);
        assert_eq!(failed, vec!["bad.mp4".to_string()]);
    }

    #[test]
    fn test_scan_path_preserved() {
        let entry = ScanEntry::with_correct_checksum("a/b/c/file.flac", b"audio");
        let results = ArchiveIntegrityScanner::scan(&[entry]);
        assert_eq!(results[0].0, "a/b/c/file.flac");
    }

    #[test]
    fn test_compute_checksum_deterministic() {
        let cs1 = compute_checksum(b"test");
        let cs2 = compute_checksum(b"test");
        assert_eq!(cs1, cs2);
    }

    #[test]
    fn test_compute_checksum_different_for_different_data() {
        let cs1 = compute_checksum(b"abc");
        let cs2 = compute_checksum(b"xyz");
        assert_ne!(cs1, cs2);
    }

    #[test]
    fn test_compute_checksum_empty() {
        let cs = compute_checksum(b"");
        assert_eq!(cs.len(), 8); // 4-byte hex
    }

    #[test]
    fn test_with_correct_checksum_passes_scan() {
        let entry = ScanEntry::with_correct_checksum("test.wav", b"some audio data");
        let results = ArchiveIntegrityScanner::scan(&[entry]);
        assert!(results[0].1);
    }
}
