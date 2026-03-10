#![allow(dead_code)]
//! Media relinking engine for conform workflows.
//!
//! This module provides tools for relinking offline media references to
//! online media files when paths have changed, files have been moved to
//! different storage, or media has been re-encoded with different codecs.

use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};

/// Status of a relink attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RelinkStatus {
    /// Successfully relinked.
    Linked,
    /// Pending user confirmation.
    Pending,
    /// No candidate found.
    Missing,
    /// Multiple candidates found (ambiguous).
    Ambiguous,
    /// Manually overridden by user.
    Manual,
    /// Skipped by user or rule.
    Skipped,
}

impl fmt::Display for RelinkStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Linked => write!(f, "LINKED"),
            Self::Pending => write!(f, "PENDING"),
            Self::Missing => write!(f, "MISSING"),
            Self::Ambiguous => write!(f, "AMBIGUOUS"),
            Self::Manual => write!(f, "MANUAL"),
            Self::Skipped => write!(f, "SKIPPED"),
        }
    }
}

/// Strategy for finding relink candidates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RelinkStrategy {
    /// Match by exact filename.
    ExactFilename,
    /// Match by filename ignoring case.
    CaseInsensitiveFilename,
    /// Match by file stem (ignoring extension).
    StemOnly,
    /// Match by partial path components.
    PartialPath,
    /// Match by file size.
    FileSize,
    /// Match by content hash.
    ContentHash,
}

impl fmt::Display for RelinkStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ExactFilename => write!(f, "exact_filename"),
            Self::CaseInsensitiveFilename => write!(f, "case_insensitive"),
            Self::StemOnly => write!(f, "stem_only"),
            Self::PartialPath => write!(f, "partial_path"),
            Self::FileSize => write!(f, "file_size"),
            Self::ContentHash => write!(f, "content_hash"),
        }
    }
}

/// A reference to an offline (missing) media file.
#[derive(Debug, Clone)]
pub struct OfflineReference {
    /// Unique clip identifier.
    pub clip_id: String,
    /// Original file path.
    pub original_path: PathBuf,
    /// Expected file size in bytes (if known).
    pub expected_size: Option<u64>,
    /// Expected content hash (if known).
    pub expected_hash: Option<String>,
    /// Reel name or tape name.
    pub reel_name: Option<String>,
    /// Timecode in point (as string).
    pub tc_in: Option<String>,
    /// Timecode out point (as string).
    pub tc_out: Option<String>,
}

impl OfflineReference {
    /// Create a new offline reference.
    pub fn new(clip_id: impl Into<String>, original_path: impl Into<PathBuf>) -> Self {
        Self {
            clip_id: clip_id.into(),
            original_path: original_path.into(),
            expected_size: None,
            expected_hash: None,
            reel_name: None,
            tc_in: None,
            tc_out: None,
        }
    }

    /// Get the filename from the original path.
    #[must_use]
    pub fn filename(&self) -> Option<&str> {
        self.original_path.file_name().and_then(|n| n.to_str())
    }

    /// Get the file stem (name without extension).
    #[must_use]
    pub fn file_stem(&self) -> Option<&str> {
        self.original_path.file_stem().and_then(|n| n.to_str())
    }

    /// Get the file extension.
    #[must_use]
    pub fn extension(&self) -> Option<&str> {
        self.original_path.extension().and_then(|e| e.to_str())
    }
}

/// A candidate match for relinking.
#[derive(Debug, Clone)]
pub struct RelinkCandidate {
    /// Path to the candidate file.
    pub path: PathBuf,
    /// Strategy that found this candidate.
    pub strategy: RelinkStrategy,
    /// Confidence score (0.0 - 1.0).
    pub confidence: f64,
    /// File size in bytes (if checked).
    pub file_size: Option<u64>,
    /// Content hash (if computed).
    pub content_hash: Option<String>,
}

impl RelinkCandidate {
    /// Create a new candidate.
    pub fn new(path: impl Into<PathBuf>, strategy: RelinkStrategy, confidence: f64) -> Self {
        Self {
            path: path.into(),
            strategy,
            confidence: confidence.clamp(0.0, 1.0),
            file_size: None,
            content_hash: None,
        }
    }
}

/// A relink result for a single clip.
#[derive(Debug, Clone)]
pub struct RelinkResult {
    /// The clip ID.
    pub clip_id: String,
    /// The original offline reference.
    pub original_path: PathBuf,
    /// Status of the relink.
    pub status: RelinkStatus,
    /// The chosen target path (if linked).
    pub target_path: Option<PathBuf>,
    /// Strategy used for the match.
    pub strategy_used: Option<RelinkStrategy>,
    /// All candidates found.
    pub candidates: Vec<RelinkCandidate>,
    /// Confidence of the chosen match.
    pub confidence: f64,
}

impl RelinkResult {
    /// Create a missing result (no candidates).
    pub fn missing(clip_id: impl Into<String>, original_path: impl Into<PathBuf>) -> Self {
        Self {
            clip_id: clip_id.into(),
            original_path: original_path.into(),
            status: RelinkStatus::Missing,
            target_path: None,
            strategy_used: None,
            candidates: Vec::new(),
            confidence: 0.0,
        }
    }

    /// Create a linked result.
    pub fn linked(
        clip_id: impl Into<String>,
        original_path: impl Into<PathBuf>,
        target: impl Into<PathBuf>,
        strategy: RelinkStrategy,
        confidence: f64,
    ) -> Self {
        Self {
            clip_id: clip_id.into(),
            original_path: original_path.into(),
            status: RelinkStatus::Linked,
            target_path: Some(target.into()),
            strategy_used: Some(strategy),
            candidates: Vec::new(),
            confidence,
        }
    }

    /// Check if this result is successfully linked.
    #[must_use]
    pub fn is_linked(&self) -> bool {
        self.status == RelinkStatus::Linked && self.target_path.is_some()
    }
}

/// Path mapping rule for bulk relinking.
#[derive(Debug, Clone)]
pub struct PathMapping {
    /// Source path prefix to match.
    pub source_prefix: PathBuf,
    /// Target path prefix to replace with.
    pub target_prefix: PathBuf,
    /// Whether this mapping is case-sensitive.
    pub case_sensitive: bool,
}

impl PathMapping {
    /// Create a new path mapping.
    pub fn new(source: impl Into<PathBuf>, target: impl Into<PathBuf>) -> Self {
        Self {
            source_prefix: source.into(),
            target_prefix: target.into(),
            case_sensitive: true,
        }
    }

    /// Apply this mapping to a path, returning the remapped path if it matches.
    #[must_use]
    pub fn apply(&self, path: &Path) -> Option<PathBuf> {
        let path_str = path.to_string_lossy();
        let source_str = self.source_prefix.to_string_lossy();

        let matches = if self.case_sensitive {
            path_str.starts_with(source_str.as_ref())
        } else {
            path_str
                .to_lowercase()
                .starts_with(&source_str.to_lowercase())
        };

        if matches {
            let remainder = &path_str[source_str.len()..];
            let mut new_path = self.target_prefix.clone();
            new_path.push(remainder.trim_start_matches('/').trim_start_matches('\\'));
            Some(new_path)
        } else {
            None
        }
    }
}

/// Aggregate relink statistics.
#[derive(Debug, Clone, Default)]
pub struct RelinkStats {
    /// Total references processed.
    pub total: usize,
    /// Successfully linked.
    pub linked: usize,
    /// Missing (not found).
    pub missing: usize,
    /// Ambiguous (multiple matches).
    pub ambiguous: usize,
    /// Skipped.
    pub skipped: usize,
    /// Manually resolved.
    pub manual: usize,
    /// Count by strategy.
    pub by_strategy: HashMap<RelinkStrategy, usize>,
}

impl RelinkStats {
    /// Calculate the success rate as a fraction (0.0 - 1.0).
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn success_rate(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        self.linked as f64 / self.total as f64
    }
}

/// The media relink engine.
#[derive(Debug, Clone)]
pub struct MediaRelinker {
    /// Search directories.
    pub search_dirs: Vec<PathBuf>,
    /// Path mappings for bulk relinking.
    pub path_mappings: Vec<PathMapping>,
    /// Strategies to try, in priority order.
    pub strategies: Vec<RelinkStrategy>,
    /// Minimum confidence threshold for auto-linking.
    pub min_confidence: f64,
}

impl Default for MediaRelinker {
    fn default() -> Self {
        Self {
            search_dirs: Vec::new(),
            path_mappings: Vec::new(),
            strategies: vec![
                RelinkStrategy::ExactFilename,
                RelinkStrategy::CaseInsensitiveFilename,
                RelinkStrategy::StemOnly,
            ],
            min_confidence: 0.8,
        }
    }
}

impl MediaRelinker {
    /// Create a new relinker with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a search directory.
    pub fn add_search_dir(&mut self, dir: impl Into<PathBuf>) {
        self.search_dirs.push(dir.into());
    }

    /// Add a path mapping.
    pub fn add_path_mapping(&mut self, mapping: PathMapping) {
        self.path_mappings.push(mapping);
    }

    /// Try to apply path mappings to relink a reference.
    #[must_use]
    pub fn try_path_mapping(&self, reference: &OfflineReference) -> Option<RelinkResult> {
        for mapping in &self.path_mappings {
            if let Some(new_path) = mapping.apply(&reference.original_path) {
                return Some(RelinkResult::linked(
                    &reference.clip_id,
                    &reference.original_path,
                    new_path,
                    RelinkStrategy::PartialPath,
                    0.95,
                ));
            }
        }
        None
    }

    /// Match a filename using exact strategy against a candidate list.
    #[must_use]
    pub fn match_exact_filename(
        reference: &OfflineReference,
        available_files: &[PathBuf],
    ) -> Vec<RelinkCandidate> {
        let target_name = match reference.filename() {
            Some(n) => n,
            None => return Vec::new(),
        };
        available_files
            .iter()
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n == target_name)
            })
            .map(|p| RelinkCandidate::new(p.clone(), RelinkStrategy::ExactFilename, 1.0))
            .collect()
    }

    /// Match a filename using case-insensitive strategy.
    #[must_use]
    pub fn match_case_insensitive(
        reference: &OfflineReference,
        available_files: &[PathBuf],
    ) -> Vec<RelinkCandidate> {
        let target_name = match reference.filename() {
            Some(n) => n.to_lowercase(),
            None => return Vec::new(),
        };
        available_files
            .iter()
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.to_lowercase() == target_name)
            })
            .map(|p| RelinkCandidate::new(p.clone(), RelinkStrategy::CaseInsensitiveFilename, 0.9))
            .collect()
    }

    /// Match by file stem only (ignoring extension).
    #[must_use]
    pub fn match_stem_only(
        reference: &OfflineReference,
        available_files: &[PathBuf],
    ) -> Vec<RelinkCandidate> {
        let target_stem = match reference.file_stem() {
            Some(s) => s.to_lowercase(),
            None => return Vec::new(),
        };
        available_files
            .iter()
            .filter(|p| {
                p.file_stem()
                    .and_then(|s| s.to_str())
                    .is_some_and(|s| s.to_lowercase() == target_stem)
            })
            .map(|p| RelinkCandidate::new(p.clone(), RelinkStrategy::StemOnly, 0.75))
            .collect()
    }

    /// Calculate statistics from a set of relink results.
    #[must_use]
    pub fn compute_stats(results: &[RelinkResult]) -> RelinkStats {
        let mut stats = RelinkStats {
            total: results.len(),
            ..Default::default()
        };
        for r in results {
            match r.status {
                RelinkStatus::Linked => {
                    stats.linked += 1;
                    if let Some(strategy) = r.strategy_used {
                        *stats.by_strategy.entry(strategy).or_insert(0) += 1;
                    }
                }
                RelinkStatus::Missing => stats.missing += 1,
                RelinkStatus::Ambiguous => stats.ambiguous += 1,
                RelinkStatus::Skipped => stats.skipped += 1,
                RelinkStatus::Manual => stats.manual += 1,
                RelinkStatus::Pending => {}
            }
        }
        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_offline_reference_filename() {
        let r = OfflineReference::new("c1", "/media/project/clip_001.mxf");
        assert_eq!(r.filename(), Some("clip_001.mxf"));
    }

    #[test]
    fn test_offline_reference_stem() {
        let r = OfflineReference::new("c1", "/media/project/clip_001.mxf");
        assert_eq!(r.file_stem(), Some("clip_001"));
    }

    #[test]
    fn test_offline_reference_extension() {
        let r = OfflineReference::new("c1", "/media/project/clip_001.mxf");
        assert_eq!(r.extension(), Some("mxf"));
    }

    #[test]
    fn test_relink_result_missing() {
        let r = RelinkResult::missing("c1", "/old/path.mxf");
        assert_eq!(r.status, RelinkStatus::Missing);
        assert!(!r.is_linked());
    }

    #[test]
    fn test_relink_result_linked() {
        let r = RelinkResult::linked(
            "c1",
            "/old/path.mxf",
            "/new/path.mxf",
            RelinkStrategy::ExactFilename,
            1.0,
        );
        assert!(r.is_linked());
        assert_eq!(r.confidence, 1.0);
    }

    #[test]
    fn test_path_mapping_apply() {
        let mapping = PathMapping::new("/old/media", "/new/storage");
        let result = mapping.apply(Path::new("/old/media/clip.mxf"));
        assert!(result.is_some());
        let new_path = result.expect("new_path should be valid");
        assert!(new_path.to_string_lossy().contains("clip.mxf"));
    }

    #[test]
    fn test_path_mapping_no_match() {
        let mapping = PathMapping::new("/old/media", "/new/storage");
        let result = mapping.apply(Path::new("/other/path/clip.mxf"));
        assert!(result.is_none());
    }

    #[test]
    fn test_match_exact_filename() {
        let reference = OfflineReference::new("c1", "/old/clip_001.mxf");
        let available = vec![
            PathBuf::from("/new/clip_001.mxf"),
            PathBuf::from("/new/clip_002.mxf"),
        ];
        let candidates = MediaRelinker::match_exact_filename(&reference, &available);
        assert_eq!(candidates.len(), 1);
        assert!((candidates[0].confidence - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_match_case_insensitive() {
        let reference = OfflineReference::new("c1", "/old/CLIP_001.MXF");
        let available = vec![
            PathBuf::from("/new/clip_001.mxf"),
            PathBuf::from("/new/other.mxf"),
        ];
        let candidates = MediaRelinker::match_case_insensitive(&reference, &available);
        assert_eq!(candidates.len(), 1);
    }

    #[test]
    fn test_match_stem_only() {
        let reference = OfflineReference::new("c1", "/old/clip_001.mov");
        let available = vec![
            PathBuf::from("/new/clip_001.mxf"), // Same stem, different ext
            PathBuf::from("/new/clip_002.mxf"),
        ];
        let candidates = MediaRelinker::match_stem_only(&reference, &available);
        assert_eq!(candidates.len(), 1);
    }

    #[test]
    fn test_compute_stats_all_linked() {
        let results = vec![
            RelinkResult::linked("c1", "/a", "/b", RelinkStrategy::ExactFilename, 1.0),
            RelinkResult::linked("c2", "/c", "/d", RelinkStrategy::StemOnly, 0.8),
        ];
        let stats = MediaRelinker::compute_stats(&results);
        assert_eq!(stats.total, 2);
        assert_eq!(stats.linked, 2);
        assert!((stats.success_rate() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_compute_stats_mixed() {
        let results = vec![
            RelinkResult::linked("c1", "/a", "/b", RelinkStrategy::ExactFilename, 1.0),
            RelinkResult::missing("c2", "/c"),
        ];
        let stats = MediaRelinker::compute_stats(&results);
        assert_eq!(stats.linked, 1);
        assert_eq!(stats.missing, 1);
        assert!((stats.success_rate() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_compute_stats_empty() {
        let stats = MediaRelinker::compute_stats(&[]);
        assert_eq!(stats.total, 0);
        assert!((stats.success_rate() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_relinker_default() {
        let relinker = MediaRelinker::new();
        assert!(relinker.search_dirs.is_empty());
        assert_eq!(relinker.strategies.len(), 3);
    }

    #[test]
    fn test_relinker_add_search_dir() {
        let mut relinker = MediaRelinker::new();
        relinker.add_search_dir("/media/sources");
        assert_eq!(relinker.search_dirs.len(), 1);
    }

    #[test]
    fn test_relink_status_display() {
        assert_eq!(format!("{}", RelinkStatus::Linked), "LINKED");
        assert_eq!(format!("{}", RelinkStatus::Missing), "MISSING");
        assert_eq!(format!("{}", RelinkStatus::Ambiguous), "AMBIGUOUS");
    }

    #[test]
    fn test_relink_strategy_display() {
        assert_eq!(
            format!("{}", RelinkStrategy::ExactFilename),
            "exact_filename"
        );
        assert_eq!(format!("{}", RelinkStrategy::ContentHash), "content_hash");
    }

    #[test]
    fn test_relink_candidate_confidence_clamp() {
        let c = RelinkCandidate::new("/path", RelinkStrategy::ExactFilename, 1.5);
        assert!((c.confidence - 1.0).abs() < f64::EPSILON);
        let c2 = RelinkCandidate::new("/path", RelinkStrategy::ExactFilename, -0.5);
        assert!((c2.confidence - 0.0).abs() < f64::EPSILON);
    }
}
