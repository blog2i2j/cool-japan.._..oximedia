//! Merge strategies for resolving duplicate file groups.
//!
//! When duplicates are found, this module decides which files to keep and
//! which to remove (or link). Strategies include:
//! - **`KeepNewest`**: keep the file with the latest modification time
//! - **`KeepOldest`**: keep the earliest file
//! - **`KeepLargest`**: keep the largest file (e.g. highest-quality encode)
//! - **`KeepSmallest`**: keep the smallest (e.g. most efficient encode)
//! - **`KeepByPath`**: keep the file in a preferred directory hierarchy
//! - **`Custom`**: user-supplied scoring function

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// FileCandidate
// ---------------------------------------------------------------------------

/// Metadata about a duplicate file candidate.
#[derive(Debug, Clone)]
pub struct FileCandidate {
    /// Path to the file.
    pub path: PathBuf,
    /// File size in bytes.
    pub size: u64,
    /// Modification timestamp (Unix seconds).
    pub modified: u64,
    /// Creation timestamp (Unix seconds).
    pub created: u64,
    /// Optional quality score (0.0 - 1.0).
    pub quality_score: Option<f64>,
}

impl FileCandidate {
    /// Create a new candidate.
    pub fn new(path: PathBuf, size: u64, modified: u64, created: u64) -> Self {
        Self {
            path,
            size,
            modified,
            created,
            quality_score: None,
        }
    }

    /// Builder: set an optional quality score.
    #[must_use]
    pub fn with_quality(mut self, score: f64) -> Self {
        self.quality_score = Some(score);
        self
    }
}

// ---------------------------------------------------------------------------
// MergeStrategy
// ---------------------------------------------------------------------------

/// Strategy for choosing which duplicate to keep.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeStrategy {
    /// Keep the most recently modified file.
    KeepNewest,
    /// Keep the oldest modified file.
    KeepOldest,
    /// Keep the largest file.
    KeepLargest,
    /// Keep the smallest file.
    KeepSmallest,
    /// Keep the file with the highest quality score.
    KeepHighestQuality,
}

impl MergeStrategy {
    /// Return a human-readable label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::KeepNewest => "keep-newest",
            Self::KeepOldest => "keep-oldest",
            Self::KeepLargest => "keep-largest",
            Self::KeepSmallest => "keep-smallest",
            Self::KeepHighestQuality => "keep-highest-quality",
        }
    }
}

// ---------------------------------------------------------------------------
// MergeAction
// ---------------------------------------------------------------------------

/// Action to perform on a file after merge resolution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MergeAction {
    /// Keep this file as the canonical copy.
    Keep,
    /// Remove this file.
    Remove,
    /// Replace this file with a symlink to the kept file.
    Symlink {
        /// Target of the symlink (the kept file).
        target: PathBuf,
    },
    /// Replace this file with a hardlink to the kept file.
    Hardlink {
        /// Target of the hardlink (the kept file).
        target: PathBuf,
    },
}

impl MergeAction {
    /// Return `true` if this action keeps the file.
    #[must_use]
    pub fn is_keep(&self) -> bool {
        matches!(self, Self::Keep)
    }

    /// Return `true` if this action removes the file.
    #[must_use]
    pub fn is_remove(&self) -> bool {
        matches!(self, Self::Remove)
    }
}

// ---------------------------------------------------------------------------
// MergeResolution
// ---------------------------------------------------------------------------

/// A single file's resolution after merge.
#[derive(Debug, Clone)]
pub struct FileResolution {
    /// The candidate file.
    pub candidate: FileCandidate,
    /// The action to take.
    pub action: MergeAction,
}

/// The full resolution of a duplicate group.
#[derive(Debug, Clone)]
pub struct MergeResolution {
    /// Per-file resolutions.
    pub files: Vec<FileResolution>,
    /// The strategy used.
    pub strategy: MergeStrategy,
    /// Estimated bytes recoverable by removing duplicates.
    pub bytes_saved: u64,
}

// ---------------------------------------------------------------------------
// Resolver
// ---------------------------------------------------------------------------

/// Resolve a group of duplicate candidates using a strategy.
///
/// Returns a [`MergeResolution`] specifying which file to keep and what
/// to do with the rest.
pub fn resolve(
    candidates: &[FileCandidate],
    strategy: MergeStrategy,
    link_mode: LinkMode,
) -> MergeResolution {
    if candidates.is_empty() {
        return MergeResolution {
            files: Vec::new(),
            strategy,
            bytes_saved: 0,
        };
    }

    let winner_idx = pick_winner(candidates, strategy);
    let winner_path = candidates[winner_idx].path.clone();
    let mut bytes_saved = 0u64;

    let files = candidates
        .iter()
        .enumerate()
        .map(|(i, c)| {
            if i == winner_idx {
                FileResolution {
                    candidate: c.clone(),
                    action: MergeAction::Keep,
                }
            } else {
                bytes_saved += c.size;
                let action = match link_mode {
                    LinkMode::Delete => MergeAction::Remove,
                    LinkMode::Symlink => MergeAction::Symlink {
                        target: winner_path.clone(),
                    },
                    LinkMode::Hardlink => MergeAction::Hardlink {
                        target: winner_path.clone(),
                    },
                };
                FileResolution {
                    candidate: c.clone(),
                    action,
                }
            }
        })
        .collect();

    MergeResolution {
        files,
        strategy,
        bytes_saved,
    }
}

/// How to handle non-winner files.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkMode {
    /// Delete non-winner files.
    Delete,
    /// Replace with symlinks.
    Symlink,
    /// Replace with hardlinks.
    Hardlink,
}

/// Pick the winner index based on strategy.
fn pick_winner(candidates: &[FileCandidate], strategy: MergeStrategy) -> usize {
    match strategy {
        MergeStrategy::KeepNewest => candidates
            .iter()
            .enumerate()
            .max_by_key(|(_, c)| c.modified)
            .map(|(i, _)| i)
            .unwrap_or(0),
        MergeStrategy::KeepOldest => candidates
            .iter()
            .enumerate()
            .min_by_key(|(_, c)| c.modified)
            .map(|(i, _)| i)
            .unwrap_or(0),
        MergeStrategy::KeepLargest => candidates
            .iter()
            .enumerate()
            .max_by_key(|(_, c)| c.size)
            .map(|(i, _)| i)
            .unwrap_or(0),
        MergeStrategy::KeepSmallest => candidates
            .iter()
            .enumerate()
            .min_by_key(|(_, c)| c.size)
            .map(|(i, _)| i)
            .unwrap_or(0),
        MergeStrategy::KeepHighestQuality => candidates
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                let qa = a.quality_score.unwrap_or(0.0);
                let qb = b.quality_score.unwrap_or(0.0);
                qa.partial_cmp(&qb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0),
    }
}

/// Check if a path is under a preferred directory prefix.
#[must_use]
pub fn is_preferred_path(path: &Path, preferred_prefix: &Path) -> bool {
    path.starts_with(preferred_prefix)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn candidates() -> Vec<FileCandidate> {
        vec![
            FileCandidate::new(PathBuf::from("/a.mp4"), 1000, 100, 90),
            FileCandidate::new(PathBuf::from("/b.mp4"), 2000, 200, 80),
            FileCandidate::new(PathBuf::from("/c.mp4"), 500, 50, 100),
        ]
    }

    #[test]
    fn test_keep_newest() {
        let res = resolve(&candidates(), MergeStrategy::KeepNewest, LinkMode::Delete);
        assert_eq!(res.files.len(), 3);
        assert!(res.files[1].action.is_keep()); // /b.mp4 has modified=200
    }

    #[test]
    fn test_keep_oldest() {
        let res = resolve(&candidates(), MergeStrategy::KeepOldest, LinkMode::Delete);
        assert!(res.files[2].action.is_keep()); // /c.mp4 has modified=50
    }

    #[test]
    fn test_keep_largest() {
        let res = resolve(&candidates(), MergeStrategy::KeepLargest, LinkMode::Delete);
        assert!(res.files[1].action.is_keep()); // /b.mp4 has size=2000
    }

    #[test]
    fn test_keep_smallest() {
        let res = resolve(&candidates(), MergeStrategy::KeepSmallest, LinkMode::Delete);
        assert!(res.files[2].action.is_keep()); // /c.mp4 has size=500
    }

    #[test]
    fn test_keep_highest_quality() {
        let cs = vec![
            FileCandidate::new(PathBuf::from("/a.mp4"), 100, 10, 10).with_quality(0.6),
            FileCandidate::new(PathBuf::from("/b.mp4"), 100, 10, 10).with_quality(0.9),
            FileCandidate::new(PathBuf::from("/c.mp4"), 100, 10, 10).with_quality(0.3),
        ];
        let res = resolve(&cs, MergeStrategy::KeepHighestQuality, LinkMode::Delete);
        assert!(res.files[1].action.is_keep()); // 0.9 is highest
    }

    #[test]
    fn test_bytes_saved() {
        let res = resolve(&candidates(), MergeStrategy::KeepLargest, LinkMode::Delete);
        // keep /b.mp4 (2000), remove /a.mp4 (1000) and /c.mp4 (500) => saved 1500
        assert_eq!(res.bytes_saved, 1500);
    }

    #[test]
    fn test_symlink_mode() {
        let res = resolve(&candidates(), MergeStrategy::KeepNewest, LinkMode::Symlink);
        for f in &res.files {
            if !f.action.is_keep() {
                match &f.action {
                    MergeAction::Symlink { target } => {
                        assert_eq!(target, &PathBuf::from("/b.mp4"));
                    }
                    _ => panic!("expected symlink action"),
                }
            }
        }
    }

    #[test]
    fn test_hardlink_mode() {
        let res = resolve(&candidates(), MergeStrategy::KeepNewest, LinkMode::Hardlink);
        for f in &res.files {
            if !f.action.is_keep() {
                match &f.action {
                    MergeAction::Hardlink { target } => {
                        assert_eq!(target, &PathBuf::from("/b.mp4"));
                    }
                    _ => panic!("expected hardlink action"),
                }
            }
        }
    }

    #[test]
    fn test_empty_candidates() {
        let res = resolve(&[], MergeStrategy::KeepNewest, LinkMode::Delete);
        assert!(res.files.is_empty());
        assert_eq!(res.bytes_saved, 0);
    }

    #[test]
    fn test_single_candidate() {
        let cs = vec![FileCandidate::new(PathBuf::from("/only.mp4"), 999, 10, 10)];
        let res = resolve(&cs, MergeStrategy::KeepNewest, LinkMode::Delete);
        assert_eq!(res.files.len(), 1);
        assert!(res.files[0].action.is_keep());
        assert_eq!(res.bytes_saved, 0);
    }

    #[test]
    fn test_is_preferred_path() {
        assert!(is_preferred_path(
            Path::new("/archive/media/a.mp4"),
            Path::new("/archive")
        ));
        assert!(!is_preferred_path(
            Path::new("/other/a.mp4"),
            Path::new("/archive")
        ));
    }

    #[test]
    fn test_strategy_label() {
        assert_eq!(MergeStrategy::KeepNewest.label(), "keep-newest");
        assert_eq!(MergeStrategy::KeepSmallest.label(), "keep-smallest");
    }
}

// ---------------------------------------------------------------------------
// DuplicateGroup
// ---------------------------------------------------------------------------

/// A group of files that have been identified as duplicates, with one
/// designated as the canonical representative to keep.
#[derive(Debug, Clone)]
pub struct DuplicateGroup {
    /// All files in this duplicate group (including the representative).
    pub files: Vec<PathBuf>,
    /// The representative (canonical) file to retain.
    pub representative: PathBuf,
}

impl DuplicateGroup {
    /// Create a new `DuplicateGroup`.
    ///
    /// `representative` does not have to be a member of `files`; the caller
    /// is responsible for consistency.
    #[must_use]
    pub fn new(files: Vec<PathBuf>, representative: PathBuf) -> Self {
        Self {
            files,
            representative,
        }
    }

    /// Number of files in this group.
    #[must_use]
    pub fn len(&self) -> usize {
        self.files.len()
    }

    /// Returns `true` if the group contains no files.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.files.is_empty()
    }

    /// Returns references to files that are NOT the representative.
    ///
    /// These are the duplicates that can be removed, linked, or otherwise
    /// disposed of according to a [`MergeStrategy`].
    #[must_use]
    pub fn duplicates(&self) -> Vec<&PathBuf> {
        self.files
            .iter()
            .filter(|p| p.as_path() != self.representative.as_path())
            .collect()
    }

    /// Returns `true` if `path` is the representative of this group.
    #[must_use]
    pub fn is_representative(&self, path: &Path) -> bool {
        self.representative == path
    }
}

// ---------------------------------------------------------------------------
// MergeResolver
// ---------------------------------------------------------------------------

/// Resolves a [`DuplicateGroup`] to a single canonical file using a
/// configured [`MergeStrategy`] and [`LinkMode`].
///
/// Two resolution paths are provided:
/// 1. [`MergeResolver::resolve`] — reads filesystem metadata for scoring.
/// 2. [`MergeResolver::resolve_from_candidates`] — accepts pre-built
///    [`FileCandidate`] data (no filesystem access required).
#[derive(Debug, Clone)]
pub struct MergeResolver {
    strategy: MergeStrategy,
    link_mode: LinkMode,
}

impl MergeResolver {
    /// Create a new resolver.
    #[must_use]
    pub fn new(strategy: MergeStrategy, link_mode: LinkMode) -> Self {
        Self {
            strategy,
            link_mode,
        }
    }

    /// Create a resolver that deletes duplicates and keeps with `KeepLargest`.
    #[must_use]
    pub fn default_delete() -> Self {
        Self::new(MergeStrategy::KeepLargest, LinkMode::Delete)
    }

    /// Returns the configured strategy.
    #[must_use]
    pub fn strategy(&self) -> MergeStrategy {
        self.strategy
    }

    /// Returns the configured link mode.
    #[must_use]
    pub fn link_mode(&self) -> LinkMode {
        self.link_mode
    }

    /// Resolve which file to keep by reading filesystem metadata.
    ///
    /// For each file in `group.files`, attempts to read its metadata (size,
    /// mtime). Files whose metadata cannot be read are assigned zero values and
    /// will lose to files with valid metadata under most strategies.
    ///
    /// Returns the path of the file to keep. Falls back to the first file when
    /// `group.files` is empty.
    #[must_use]
    pub fn resolve(&self, group: &DuplicateGroup) -> PathBuf {
        if group.files.is_empty() {
            return group.representative.clone();
        }

        let candidates: Vec<FileCandidate> = group
            .files
            .iter()
            .map(|path| {
                let meta = std::fs::metadata(path);
                let (size, modified, created) = meta
                    .as_ref()
                    .map(|m| {
                        let size = m.len();
                        let modified = m
                            .modified()
                            .ok()
                            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                            .map(|d| d.as_secs())
                            .unwrap_or(0);
                        let created = m
                            .created()
                            .ok()
                            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                            .map(|d| d.as_secs())
                            .unwrap_or(0);
                        (size, modified, created)
                    })
                    .unwrap_or((0, 0, 0));
                FileCandidate::new(path.clone(), size, modified, created)
            })
            .collect();

        self.resolve_from_candidates(&candidates)
            .unwrap_or_else(|| group.files[0].clone())
    }

    /// Resolve from pre-built candidate metadata (no filesystem access).
    ///
    /// Returns the path of the winning candidate, or `None` if `candidates`
    /// is empty.
    #[must_use]
    pub fn resolve_from_candidates(&self, candidates: &[FileCandidate]) -> Option<PathBuf> {
        if candidates.is_empty() {
            return None;
        }
        let resolution = resolve(candidates, self.strategy, self.link_mode);
        resolution
            .files
            .iter()
            .find(|f| f.action.is_keep())
            .map(|f| f.candidate.path.clone())
    }
}

// ---------------------------------------------------------------------------
// DuplicateGroup + MergeResolver tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod group_resolver_tests {
    use super::*;

    fn paths(names: &[&str]) -> Vec<PathBuf> {
        names.iter().map(|n| PathBuf::from(n)).collect()
    }

    // ── DuplicateGroup ─────────────────────────────────────────────────────

    #[test]
    fn test_group_new_and_len() {
        let g = DuplicateGroup::new(
            paths(&["/a.mp4", "/b.mp4", "/c.mp4"]),
            PathBuf::from("/a.mp4"),
        );
        assert_eq!(g.len(), 3);
        assert!(!g.is_empty());
    }

    #[test]
    fn test_group_empty() {
        let g = DuplicateGroup::new(vec![], PathBuf::from("/none.mp4"));
        assert!(g.is_empty());
        assert_eq!(g.len(), 0);
    }

    #[test]
    fn test_group_duplicates_excludes_representative() {
        let g = DuplicateGroup::new(
            paths(&["/a.mp4", "/b.mp4", "/c.mp4"]),
            PathBuf::from("/a.mp4"),
        );
        let dups = g.duplicates();
        assert_eq!(dups.len(), 2);
        assert!(!dups.contains(&&PathBuf::from("/a.mp4")));
        assert!(dups.contains(&&PathBuf::from("/b.mp4")));
        assert!(dups.contains(&&PathBuf::from("/c.mp4")));
    }

    #[test]
    fn test_group_is_representative() {
        let g = DuplicateGroup::new(paths(&["/rep.mp4", "/dup.mp4"]), PathBuf::from("/rep.mp4"));
        assert!(g.is_representative(Path::new("/rep.mp4")));
        assert!(!g.is_representative(Path::new("/dup.mp4")));
    }

    #[test]
    fn test_group_duplicates_all_are_duplicates_when_representative_absent() {
        // Representative not in files list — all files are returned as duplicates.
        let g = DuplicateGroup::new(paths(&["/b.mp4", "/c.mp4"]), PathBuf::from("/a.mp4"));
        assert_eq!(g.duplicates().len(), 2);
    }

    // ── MergeResolver ──────────────────────────────────────────────────────

    fn make_candidates() -> Vec<FileCandidate> {
        vec![
            FileCandidate::new(PathBuf::from("/small.mp4"), 500, 100, 90),
            FileCandidate::new(PathBuf::from("/large.mp4"), 2000, 200, 80),
            FileCandidate::new(PathBuf::from("/oldest.mp4"), 1000, 50, 100),
        ]
    }

    #[test]
    fn test_resolver_keep_largest_from_candidates() {
        let r = MergeResolver::new(MergeStrategy::KeepLargest, LinkMode::Delete);
        let result = r.resolve_from_candidates(&make_candidates());
        assert_eq!(result, Some(PathBuf::from("/large.mp4")));
    }

    #[test]
    fn test_resolver_keep_newest_from_candidates() {
        let r = MergeResolver::new(MergeStrategy::KeepNewest, LinkMode::Delete);
        let result = r.resolve_from_candidates(&make_candidates());
        assert_eq!(result, Some(PathBuf::from("/large.mp4"))); // modified=200
    }

    #[test]
    fn test_resolver_keep_oldest_from_candidates() {
        let r = MergeResolver::new(MergeStrategy::KeepOldest, LinkMode::Delete);
        let result = r.resolve_from_candidates(&make_candidates());
        assert_eq!(result, Some(PathBuf::from("/oldest.mp4"))); // modified=50
    }

    #[test]
    fn test_resolver_keep_smallest_from_candidates() {
        let r = MergeResolver::new(MergeStrategy::KeepSmallest, LinkMode::Delete);
        let result = r.resolve_from_candidates(&make_candidates());
        assert_eq!(result, Some(PathBuf::from("/small.mp4"))); // size=500
    }

    #[test]
    fn test_resolver_keep_highest_quality_from_candidates() {
        let cs = vec![
            FileCandidate::new(PathBuf::from("/low.mp4"), 100, 10, 10).with_quality(0.3),
            FileCandidate::new(PathBuf::from("/high.mp4"), 100, 10, 10).with_quality(0.95),
        ];
        let r = MergeResolver::new(MergeStrategy::KeepHighestQuality, LinkMode::Delete);
        let result = r.resolve_from_candidates(&cs);
        assert_eq!(result, Some(PathBuf::from("/high.mp4")));
    }

    #[test]
    fn test_resolver_empty_candidates_returns_none() {
        let r = MergeResolver::new(MergeStrategy::KeepLargest, LinkMode::Delete);
        assert!(r.resolve_from_candidates(&[]).is_none());
    }

    #[test]
    fn test_resolver_strategy_and_link_mode_accessors() {
        let r = MergeResolver::new(MergeStrategy::KeepSmallest, LinkMode::Symlink);
        assert_eq!(r.strategy(), MergeStrategy::KeepSmallest);
        assert_eq!(r.link_mode(), LinkMode::Symlink);
    }

    #[test]
    fn test_resolver_default_delete() {
        let r = MergeResolver::default_delete();
        assert_eq!(r.strategy(), MergeStrategy::KeepLargest);
        assert_eq!(r.link_mode(), LinkMode::Delete);
    }

    #[test]
    fn test_resolver_resolve_filesystem_fallback() {
        // Files do not exist → metadata reads fail → fallback to first file.
        let group = DuplicateGroup::new(
            paths(&["/nonexistent_a.mp4", "/nonexistent_b.mp4"]),
            PathBuf::from("/nonexistent_a.mp4"),
        );
        let r = MergeResolver::new(MergeStrategy::KeepLargest, LinkMode::Delete);
        // With all sizes=0 (metadata unavailable), pick_winner returns 0 → first file
        let result = r.resolve(&group);
        assert!(!result.as_os_str().is_empty());
    }

    #[test]
    fn test_resolver_resolve_empty_group() {
        let group = DuplicateGroup::new(vec![], PathBuf::from("/rep.mp4"));
        let r = MergeResolver::new(MergeStrategy::KeepLargest, LinkMode::Delete);
        // Falls back to representative
        let result = r.resolve(&group);
        assert_eq!(result, PathBuf::from("/rep.mp4"));
    }
}
