//! IMF package diff — compare two IMF packages structurally and by content.
//!
//! [`ImfDiff`] compares two IMF package directories and produces a
//! [`DiffReport`] listing added, removed, and modified files.
//!
//! # Example
//! ```no_run
//! use oximedia_imf::imf_diff::ImfDiff;
//!
//! let report = ImfDiff::compare("/path/to/pkg_v1", "/path/to/pkg_v2")
//!     .expect("diff failed");
//! println!("{}", report.summary());
//! ```

#![allow(dead_code, missing_docs)]

use crate::{ImfError, ImfResult};
use std::collections::HashMap;
use std::path::Path;

/// Change type for a single file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChangeKind {
    Added,
    Removed,
    Modified { left_size: u64, right_size: u64 },
    Unchanged,
}

impl std::fmt::Display for ChangeKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Added => write!(f, "ADDED"),
            Self::Removed => write!(f, "REMOVED"),
            Self::Modified { left_size, right_size } => {
                write!(f, "MODIFIED ({left_size}B → {right_size}B)")
            }
            Self::Unchanged => write!(f, "UNCHANGED"),
        }
    }
}

/// A diff entry for a single file.
#[derive(Debug, Clone)]
pub struct DiffEntry {
    pub filename: String,
    pub kind: ChangeKind,
}

impl std::fmt::Display for DiffEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} — {}", self.kind, self.filename)
    }
}

/// Full diff report between two IMF packages.
#[derive(Debug, Clone, Default)]
pub struct DiffReport {
    pub left_path: String,
    pub right_path: String,
    pub entries: Vec<DiffEntry>,
}

impl DiffReport {
    #[must_use]
    pub fn new(left: impl Into<String>, right: impl Into<String>) -> Self {
        Self { left_path: left.into(), right_path: right.into(), entries: Vec::new() }
    }

    #[must_use]
    pub fn added(&self) -> Vec<&DiffEntry> {
        self.entries.iter().filter(|e| e.kind == ChangeKind::Added).collect()
    }

    #[must_use]
    pub fn removed(&self) -> Vec<&DiffEntry> {
        self.entries.iter().filter(|e| e.kind == ChangeKind::Removed).collect()
    }

    #[must_use]
    pub fn modified(&self) -> Vec<&DiffEntry> {
        self.entries.iter().filter(|e| matches!(e.kind, ChangeKind::Modified { .. })).collect()
    }

    #[must_use]
    pub fn is_identical(&self) -> bool {
        self.added().is_empty() && self.removed().is_empty() && self.modified().is_empty()
    }

    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "Diff: {} added, {} removed, {} modified, {} unchanged",
            self.added().len(),
            self.removed().len(),
            self.modified().len(),
            self.entries.iter().filter(|e| e.kind == ChangeKind::Unchanged).count()
        )
    }

    #[must_use]
    pub fn render(&self) -> String {
        let mut out = format!("=== IMF Diff: {} vs {} ===\n", self.left_path, self.right_path);
        for e in &self.entries {
            if e.kind != ChangeKind::Unchanged {
                out.push_str(&format!("  {e}\n"));
            }
        }
        out.push_str(&format!("\n{}\n", self.summary()));
        out
    }
}

/// Compares two IMF package directories.
pub struct ImfDiff;

impl ImfDiff {
    /// Compare two IMF package directories, matching files by name and size.
    ///
    /// # Errors
    /// Returns `ImfError::InvalidPackage` if either path is not a directory.
    pub fn compare(left: impl AsRef<Path>, right: impl AsRef<Path>) -> ImfResult<DiffReport> {
        let left = left.as_ref();
        let right = right.as_ref();
        if !left.is_dir() {
            return Err(ImfError::InvalidPackage(format!("Not a directory: {}", left.display())));
        }
        if !right.is_dir() {
            return Err(ImfError::InvalidPackage(format!("Not a directory: {}", right.display())));
        }

        let left_files = scan_dir(left)?;
        let right_files = scan_dir(right)?;

        let mut report = DiffReport::new(left.to_string_lossy(), right.to_string_lossy());

        for (name, &left_size) in &left_files {
            match right_files.get(name) {
                None => report.entries.push(DiffEntry { filename: name.clone(), kind: ChangeKind::Removed }),
                Some(&right_size) => {
                    let kind = if left_size == right_size {
                        ChangeKind::Unchanged
                    } else {
                        ChangeKind::Modified { left_size, right_size }
                    };
                    report.entries.push(DiffEntry { filename: name.clone(), kind });
                }
            }
        }

        for (name, _) in &right_files {
            if !left_files.contains_key(name) {
                report.entries.push(DiffEntry { filename: name.clone(), kind: ChangeKind::Added });
            }
        }

        report.entries.sort_by(|a, b| a.filename.cmp(&b.filename));
        Ok(report)
    }

    /// Compare two packages using FNV-1a content hashes for modified files.
    ///
    /// Files that appear modified by size but have the same hash are demoted to Unchanged.
    ///
    /// # Errors
    /// Propagates errors from `compare` and file I/O.
    pub fn compare_with_hash(left: impl AsRef<Path>, right: impl AsRef<Path>) -> ImfResult<DiffReport> {
        let left = left.as_ref();
        let right = right.as_ref();
        let mut report = Self::compare(left, right)?;
        for entry in &mut report.entries {
            if matches!(entry.kind, ChangeKind::Modified { .. }) {
                let lp = left.join(&entry.filename);
                let rp = right.join(&entry.filename);
                if let (Ok(lh), Ok(rh)) = (hash_file(&lp), hash_file(&rp)) {
                    if lh == rh {
                        entry.kind = ChangeKind::Unchanged;
                    }
                }
            }
        }
        Ok(report)
    }
}

fn scan_dir(dir: &Path) -> ImfResult<HashMap<String, u64>> {
    let mut map = HashMap::new();
    for entry in std::fs::read_dir(dir).map_err(|e| ImfError::Other(e.to_string()))?.flatten() {
        if entry.path().is_file() {
            let name = entry.file_name().to_string_lossy().to_string();
            let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
            map.insert(name, size);
        }
    }
    Ok(map)
}

fn hash_file(path: &Path) -> ImfResult<u64> {
    let data = std::fs::read(path).map_err(|e| ImfError::Other(e.to_string()))?;
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in data {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    Ok(hash)
}

// ============================================================================
// CPL segment-level diff (task API)
// ============================================================================

/// The type of change detected between two CPL versions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChangeType {
    /// A segment UUID present in `updated` but absent from `base`
    Added,
    /// A segment UUID present in `base` but absent from `updated`
    Removed,
    /// A segment present in both but with a different attribute/duration
    Modified,
    /// A segment present in both but at a different position
    Reordered,
}

impl std::fmt::Display for ChangeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Added => write!(f, "ADDED"),
            Self::Removed => write!(f, "REMOVED"),
            Self::Modified => write!(f, "MODIFIED"),
            Self::Reordered => write!(f, "REORDERED"),
        }
    }
}

/// A single CPL-level change record.
#[derive(Debug, Clone)]
pub struct CplChange {
    /// Type of change
    pub change_type: ChangeType,
    /// Human-readable description
    pub description: String,
    /// Index in the segment list this change relates to (if applicable)
    pub segment_index: Option<usize>,
}

impl CplChange {
    fn new(change_type: ChangeType, description: impl Into<String>, segment_index: Option<usize>) -> Self {
        Self { change_type, description: description.into(), segment_index }
    }
}

/// Aggregated diff between two CPL segment lists.
#[derive(Debug, Clone)]
pub struct CplDiff {
    /// All individual change records
    pub changes: Vec<CplChange>,
    /// Count of added segments
    pub additions: usize,
    /// Count of removed segments
    pub removals: usize,
    /// Count of modified or reordered segments
    pub modifications: usize,
}

impl CplDiff {
    fn empty() -> Self {
        Self { changes: Vec::new(), additions: 0, removals: 0, modifications: 0 }
    }

    /// Returns `true` if there are any detected changes.
    pub fn has_changes(&self) -> bool {
        self.additions > 0 || self.removals > 0 || self.modifications > 0
    }

    /// One-line summary of the diff.
    pub fn summary(&self) -> String {
        format!(
            "CplDiff: {} added, {} removed, {} modified/reordered",
            self.additions, self.removals, self.modifications,
        )
    }

    /// Returns `true` if the update is a pure, backward-compatible extension:
    /// only new segments were appended, nothing was removed or modified.
    pub fn is_compatible_extension(&self) -> bool {
        self.additions > 0 && self.removals == 0 && self.modifications == 0
    }
}

/// Compares two IMF Composition Playlists at the segment level.
pub struct ImfDiffer;

impl ImfDiffer {
    /// Diff two ordered lists of segment IDs (UUID strings).
    ///
    /// The algorithm:
    /// 1. Compute the sets of UUIDs in each list.
    /// 2. Segments in `updated` but not in `base` → **Added**.
    /// 3. Segments in `base` but not in `updated` → **Removed**.
    /// 4. Segments present in both but at a different index → **Reordered**.
    pub fn diff_segments(base: &[String], updated: &[String]) -> CplDiff {
        use std::collections::HashMap;

        // Build position maps for the common set
        let base_pos: HashMap<&str, usize> = base
            .iter()
            .enumerate()
            .map(|(i, s)| (s.as_str(), i))
            .collect();
        let updated_pos: HashMap<&str, usize> = updated
            .iter()
            .enumerate()
            .map(|(i, s)| (s.as_str(), i))
            .collect();

        let mut changes = Vec::new();
        let mut additions = 0usize;
        let mut removals = 0usize;
        let mut modifications = 0usize;

        // Removed: in base but not in updated
        for (i, seg) in base.iter().enumerate() {
            if !updated_pos.contains_key(seg.as_str()) {
                changes.push(CplChange::new(
                    ChangeType::Removed,
                    format!("Segment removed: {seg}"),
                    Some(i),
                ));
                removals += 1;
            }
        }

        // Added: in updated but not in base
        for (i, seg) in updated.iter().enumerate() {
            if !base_pos.contains_key(seg.as_str()) {
                changes.push(CplChange::new(
                    ChangeType::Added,
                    format!("Segment added: {seg}"),
                    Some(i),
                ));
                additions += 1;
            }
        }

        // Reordered: present in both but relative order changed
        // We use a simple approach: find common segments, compare their relative order
        let common_in_base: Vec<&str> = base
            .iter()
            .filter(|s| updated_pos.contains_key(s.as_str()))
            .map(|s| s.as_str())
            .collect();
        let common_in_updated: Vec<&str> = updated
            .iter()
            .filter(|s| base_pos.contains_key(s.as_str()))
            .map(|s| s.as_str())
            .collect();

        if common_in_base != common_in_updated {
            // Find first differing element
            for (bi, ui) in common_in_base.iter().zip(common_in_updated.iter()) {
                if bi != ui {
                    let idx = base_pos.get(*bi).copied();
                    changes.push(CplChange::new(
                        ChangeType::Reordered,
                        format!("Segment order changed: {bi} moved"),
                        idx,
                    ));
                    modifications += 1;
                    break;
                }
            }
            // If lengths differ it may mean reorders of multiple segments
            if common_in_base.len() != common_in_updated.len() && modifications == 0 {
                changes.push(CplChange::new(
                    ChangeType::Reordered,
                    "Segment order changed".to_string(),
                    None,
                ));
                modifications += 1;
            }
        }

        CplDiff { changes, additions, removals, modifications }
    }

    /// Compare the edit rates of two CPL revisions.
    ///
    /// Returns `Some(CplChange)` if the edit rate changed, `None` if identical.
    pub fn diff_edit_rates(base: (u32, u32), updated: (u32, u32)) -> Option<CplChange> {
        if base == updated {
            return None;
        }
        Some(CplChange::new(
            ChangeType::Modified,
            format!(
                "EditRate changed: {}/{} → {}/{}",
                base.0, base.1, updated.0, updated.1,
            ),
            None,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pkg(suffix: &str, files: &[(&str, &[u8])]) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("oximedia_idf_{suffix}"));
        std::fs::create_dir_all(&dir).ok();
        for (name, content) in files {
            std::fs::write(dir.join(name), content).ok();
        }
        dir
    }

    #[test]
    fn test_identical_packages() {
        let content: &[(&str, &[u8])] = &[("CPL.xml", b"<CPL/>"), ("v.mxf", b"mxf")];
        let l = make_pkg("id_l", content);
        let r = make_pkg("id_r", content);
        let report = ImfDiff::compare(&l, &r).expect("ok");
        assert!(report.is_identical());
        std::fs::remove_dir_all(&l).ok();
        std::fs::remove_dir_all(&r).ok();
    }

    #[test]
    fn test_added_file() {
        let l = make_pkg("add_l", &[("CPL.xml", b"<CPL/>")]);
        let r = make_pkg("add_r", &[("CPL.xml", b"<CPL/>"), ("audio.mxf", b"audio")]);
        let report = ImfDiff::compare(&l, &r).expect("ok");
        assert_eq!(report.added().len(), 1);
        std::fs::remove_dir_all(&l).ok();
        std::fs::remove_dir_all(&r).ok();
    }

    #[test]
    fn test_removed_file() {
        let l = make_pkg("rem_l", &[("CPL.xml", b"<CPL/>"), ("old.mxf", b"data")]);
        let r = make_pkg("rem_r", &[("CPL.xml", b"<CPL/>")]);
        let report = ImfDiff::compare(&l, &r).expect("ok");
        assert_eq!(report.removed().len(), 1);
        std::fs::remove_dir_all(&l).ok();
        std::fs::remove_dir_all(&r).ok();
    }

    #[test]
    fn test_modified_file() {
        let l = make_pkg("mod_l", &[("v.mxf", b"version1")]);
        let r = make_pkg("mod_r", &[("v.mxf", b"version2_longer")]);
        let report = ImfDiff::compare(&l, &r).expect("ok");
        assert_eq!(report.modified().len(), 1);
        std::fs::remove_dir_all(&l).ok();
        std::fs::remove_dir_all(&r).ok();
    }

    #[test]
    fn test_invalid_path() {
        assert!(ImfDiff::compare("/nonexistent/l", "/nonexistent/r").is_err());
    }

    #[test]
    fn test_summary() {
        let mut r = DiffReport::new("l", "r");
        r.entries.push(DiffEntry { filename: "new.mxf".into(), kind: ChangeKind::Added });
        assert!(r.summary().contains("1 added"));
    }

    #[test]
    fn test_compare_with_hash_same_content() {
        let content = b"same content for both";
        let l = make_pkg("ch_l", &[("f.mxf", content)]);
        let r = make_pkg("ch_r", &[("f.mxf", content)]);
        let report = ImfDiff::compare_with_hash(&l, &r).expect("ok");
        assert!(report.is_identical());
        std::fs::remove_dir_all(&l).ok();
        std::fs::remove_dir_all(&r).ok();
    }

    // -----------------------------------------------------------------------
    // ImfDiffer (CPL segment-level diff) tests
    // -----------------------------------------------------------------------

    fn segs(ids: &[&str]) -> Vec<String> {
        ids.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn test_diff_identical_no_changes() {
        let base = segs(&["urn:uuid:aaa", "urn:uuid:bbb", "urn:uuid:ccc"]);
        let diff = ImfDiffer::diff_segments(&base, &base);
        assert!(!diff.has_changes(), "identical lists should produce no changes");
        assert_eq!(diff.additions, 0);
        assert_eq!(diff.removals, 0);
        assert_eq!(diff.modifications, 0);
    }

    #[test]
    fn test_diff_added_segment() {
        let base = segs(&["urn:uuid:aaa", "urn:uuid:bbb"]);
        let updated = segs(&["urn:uuid:aaa", "urn:uuid:bbb", "urn:uuid:ccc"]);
        let diff = ImfDiffer::diff_segments(&base, &updated);
        assert_eq!(diff.additions, 1);
        assert_eq!(diff.removals, 0);
        assert!(diff.changes.iter().any(|c| c.change_type == ChangeType::Added));
    }

    #[test]
    fn test_diff_removed_segment() {
        let base = segs(&["urn:uuid:aaa", "urn:uuid:bbb", "urn:uuid:ccc"]);
        let updated = segs(&["urn:uuid:aaa", "urn:uuid:bbb"]);
        let diff = ImfDiffer::diff_segments(&base, &updated);
        assert_eq!(diff.removals, 1);
        assert_eq!(diff.additions, 0);
        assert!(diff.changes.iter().any(|c| c.change_type == ChangeType::Removed));
    }

    #[test]
    fn test_diff_reordered_segments() {
        let base = segs(&["urn:uuid:aaa", "urn:uuid:bbb", "urn:uuid:ccc"]);
        let updated = segs(&["urn:uuid:aaa", "urn:uuid:ccc", "urn:uuid:bbb"]);
        let diff = ImfDiffer::diff_segments(&base, &updated);
        assert!(diff.has_changes());
        assert!(diff.changes.iter().any(|c| c.change_type == ChangeType::Reordered));
    }

    #[test]
    fn test_diff_compatible_extension() {
        let base = segs(&["urn:uuid:seg1", "urn:uuid:seg2"]);
        let updated = segs(&["urn:uuid:seg1", "urn:uuid:seg2", "urn:uuid:seg3"]);
        let diff = ImfDiffer::diff_segments(&base, &updated);
        assert!(diff.is_compatible_extension(), "only additions = compatible extension");
    }

    #[test]
    fn test_diff_not_compatible_if_removal() {
        let base = segs(&["urn:uuid:seg1", "urn:uuid:seg2", "urn:uuid:seg3"]);
        let updated = segs(&["urn:uuid:seg1", "urn:uuid:seg3", "urn:uuid:seg4"]);
        let diff = ImfDiffer::diff_segments(&base, &updated);
        assert!(!diff.is_compatible_extension());
    }

    #[test]
    fn test_diff_edit_rates_identical() {
        assert!(ImfDiffer::diff_edit_rates((24, 1), (24, 1)).is_none());
        assert!(ImfDiffer::diff_edit_rates((30000, 1001), (30000, 1001)).is_none());
    }

    #[test]
    fn test_diff_edit_rates_changed() {
        let change = ImfDiffer::diff_edit_rates((24, 1), (25, 1));
        assert!(change.is_some());
        let c = change.unwrap();
        assert_eq!(c.change_type, ChangeType::Modified);
        assert!(c.description.contains("24/1"));
        assert!(c.description.contains("25/1"));
    }

    #[test]
    fn test_diff_summary_format() {
        let base = segs(&["urn:uuid:a"]);
        let updated = segs(&["urn:uuid:a", "urn:uuid:b"]);
        let diff = ImfDiffer::diff_segments(&base, &updated);
        let s = diff.summary();
        assert!(s.contains("CplDiff"));
        assert!(s.contains("1 added"));
    }
}
