//! Content indexing for automated media analysis.
//!
//! Builds searchable indices over media assets, supporting segment-level
//! metadata, keyword lookups, and relevance-scored retrieval.

#![allow(dead_code)]

use std::collections::HashMap;

/// A time range within a media asset.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimeRange {
    /// Start time in milliseconds.
    pub start_ms: i64,
    /// End time in milliseconds.
    pub end_ms: i64,
}

impl TimeRange {
    /// Create a new time range.
    pub fn new(start_ms: i64, end_ms: i64) -> Self {
        Self { start_ms, end_ms }
    }

    /// Duration of the range in milliseconds.
    pub fn duration_ms(&self) -> i64 {
        (self.end_ms - self.start_ms).max(0)
    }

    /// Check whether another range overlaps this one.
    pub fn overlaps(&self, other: &TimeRange) -> bool {
        self.start_ms < other.end_ms && other.start_ms < self.end_ms
    }
}

/// An indexed segment of a media asset.
#[derive(Debug, Clone)]
pub struct IndexedSegment {
    /// Unique segment identifier.
    pub id: u64,
    /// Asset identifier.
    pub asset_id: String,
    /// Time range of this segment.
    pub range: TimeRange,
    /// Keywords associated with this segment.
    pub keywords: Vec<String>,
    /// Relevance score for this segment.
    pub score: f32,
    /// Arbitrary metadata.
    pub metadata: HashMap<String, String>,
}

impl IndexedSegment {
    /// Create a new indexed segment.
    pub fn new(
        id: u64,
        asset_id: impl Into<String>,
        range: TimeRange,
        keywords: Vec<String>,
        score: f32,
    ) -> Self {
        Self {
            id,
            asset_id: asset_id.into(),
            range,
            keywords,
            score: score.clamp(0.0, 1.0),
            metadata: HashMap::new(),
        }
    }

    /// Add a metadata entry.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Query parameters for searching the content index.
#[derive(Debug, Clone, Default)]
pub struct IndexQuery {
    /// Keywords to match (OR logic).
    pub keywords: Vec<String>,
    /// Asset ID filter (None = all assets).
    pub asset_id: Option<String>,
    /// Minimum relevance score.
    pub min_score: f32,
    /// Maximum results to return.
    pub max_results: usize,
}

impl IndexQuery {
    /// Create a keyword query.
    pub fn keywords(keywords: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            keywords: keywords.into_iter().map(Into::into).collect(),
            max_results: 100,
            ..Default::default()
        }
    }

    /// Set asset filter.
    pub fn for_asset(mut self, asset_id: impl Into<String>) -> Self {
        self.asset_id = Some(asset_id.into());
        self
    }

    /// Set minimum score.
    pub fn min_score(mut self, score: f32) -> Self {
        self.min_score = score;
        self
    }

    /// Set maximum results.
    pub fn limit(mut self, max: usize) -> Self {
        self.max_results = max;
        self
    }
}

/// Content index holding all indexed segments.
#[derive(Debug, Default)]
pub struct ContentIndex {
    segments: Vec<IndexedSegment>,
    next_id: u64,
    /// Inverted index: keyword -> segment IDs.
    keyword_index: HashMap<String, Vec<u64>>,
}

impl ContentIndex {
    /// Create an empty content index.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a segment to the index.
    pub fn add_segment(
        &mut self,
        asset_id: impl Into<String>,
        range: TimeRange,
        keywords: Vec<String>,
        score: f32,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;

        // Update inverted index
        for kw in &keywords {
            self.keyword_index
                .entry(kw.to_lowercase())
                .or_default()
                .push(id);
        }

        self.segments
            .push(IndexedSegment::new(id, asset_id, range, keywords, score));
        id
    }

    /// Get a segment by ID.
    pub fn get(&self, id: u64) -> Option<&IndexedSegment> {
        self.segments.iter().find(|s| s.id == id)
    }

    /// Remove a segment by ID. Returns true if removed.
    pub fn remove(&mut self, id: u64) -> bool {
        let before = self.segments.len();
        self.segments.retain(|s| s.id != id);
        let removed = self.segments.len() < before;

        if removed {
            for ids in self.keyword_index.values_mut() {
                ids.retain(|&sid| sid != id);
            }
        }

        removed
    }

    /// Search the index using a query.
    pub fn search(&self, query: &IndexQuery) -> Vec<&IndexedSegment> {
        let mut results: Vec<&IndexedSegment> = Vec::new();

        if query.keywords.is_empty() {
            // Return all segments if no keywords specified
            results.extend(self.segments.iter());
        } else {
            // Collect candidate IDs via inverted index
            let mut candidate_ids: std::collections::HashSet<u64> =
                std::collections::HashSet::new();
            for kw in &query.keywords {
                if let Some(ids) = self.keyword_index.get(&kw.to_lowercase()) {
                    candidate_ids.extend(ids);
                }
            }

            for &id in &candidate_ids {
                if let Some(seg) = self.get(id) {
                    results.push(seg);
                }
            }
        }

        // Apply asset filter
        if let Some(ref asset_id) = query.asset_id {
            results.retain(|s| &s.asset_id == asset_id);
        }

        // Apply score filter
        results.retain(|s| s.score >= query.min_score);

        // Sort by score descending
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Limit results
        if query.max_results > 0 {
            results.truncate(query.max_results);
        }

        results
    }

    /// Total number of indexed segments.
    pub fn len(&self) -> usize {
        self.segments.len()
    }

    /// Returns true if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    /// Clear all segments from the index.
    pub fn clear(&mut self) {
        self.segments.clear();
        self.keyword_index.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_index() -> ContentIndex {
        ContentIndex::new()
    }

    #[test]
    fn test_time_range_duration() {
        let r = TimeRange::new(1000, 5000);
        assert_eq!(r.duration_ms(), 4000);
    }

    #[test]
    fn test_time_range_duration_negative_clamps() {
        let r = TimeRange::new(5000, 1000);
        assert_eq!(r.duration_ms(), 0);
    }

    #[test]
    fn test_time_range_overlaps() {
        let a = TimeRange::new(0, 1000);
        let b = TimeRange::new(500, 1500);
        assert!(a.overlaps(&b));
    }

    #[test]
    fn test_time_range_no_overlap() {
        let a = TimeRange::new(0, 1000);
        let b = TimeRange::new(1000, 2000);
        assert!(!a.overlaps(&b));
    }

    #[test]
    fn test_add_and_get_segment() {
        let mut index = make_index();
        let id = index.add_segment(
            "asset1",
            TimeRange::new(0, 5000),
            vec!["action".to_string()],
            0.9,
        );
        let seg = index.get(id);
        assert!(seg.is_some());
        assert_eq!(seg.expect("test expectation failed").asset_id, "asset1");
    }

    #[test]
    fn test_index_len() {
        let mut index = make_index();
        assert_eq!(index.len(), 0);
        index.add_segment("a", TimeRange::new(0, 100), vec![], 0.5);
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn test_remove_segment() {
        let mut index = make_index();
        let id = index.add_segment("a", TimeRange::new(0, 100), vec!["kw".to_string()], 0.5);
        assert!(index.remove(id));
        assert!(index.get(id).is_none());
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn test_remove_nonexistent_returns_false() {
        let mut index = make_index();
        assert!(!index.remove(999));
    }

    #[test]
    fn test_search_by_keyword() {
        let mut index = make_index();
        index.add_segment("a", TimeRange::new(0, 100), vec!["sports".to_string()], 0.8);
        index.add_segment("b", TimeRange::new(0, 100), vec!["music".to_string()], 0.7);

        let q = IndexQuery::keywords(["sports"]);
        let results = index.search(&q);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].asset_id, "a");
    }

    #[test]
    fn test_search_case_insensitive() {
        let mut index = make_index();
        index.add_segment("a", TimeRange::new(0, 100), vec!["Sports".to_string()], 0.8);

        let q = IndexQuery::keywords(["sports"]);
        let results = index.search(&q);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_search_with_asset_filter() {
        let mut index = make_index();
        index.add_segment(
            "asset1",
            TimeRange::new(0, 100),
            vec!["tag".to_string()],
            0.8,
        );
        index.add_segment(
            "asset2",
            TimeRange::new(0, 100),
            vec!["tag".to_string()],
            0.7,
        );

        let q = IndexQuery::keywords(["tag"]).for_asset("asset1");
        let results = index.search(&q);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].asset_id, "asset1");
    }

    #[test]
    fn test_search_min_score_filter() {
        let mut index = make_index();
        index.add_segment("a", TimeRange::new(0, 100), vec!["tag".to_string()], 0.3);
        index.add_segment("b", TimeRange::new(0, 100), vec!["tag".to_string()], 0.9);

        let q = IndexQuery::keywords(["tag"]).min_score(0.5);
        let results = index.search(&q);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].asset_id, "b");
    }

    #[test]
    fn test_search_results_sorted_by_score() {
        let mut index = make_index();
        index.add_segment("a", TimeRange::new(0, 100), vec!["tag".to_string()], 0.5);
        index.add_segment("b", TimeRange::new(0, 100), vec!["tag".to_string()], 0.9);
        index.add_segment("c", TimeRange::new(0, 100), vec!["tag".to_string()], 0.7);

        let q = IndexQuery::keywords(["tag"]);
        let results = index.search(&q);
        for w in results.windows(2) {
            assert!(w[0].score >= w[1].score);
        }
    }

    #[test]
    fn test_search_limit() {
        let mut index = make_index();
        for i in 0..10 {
            index.add_segment(
                format!("a{i}"),
                TimeRange::new(0, 100),
                vec!["tag".to_string()],
                0.5,
            );
        }
        let q = IndexQuery::keywords(["tag"]).limit(3);
        let results = index.search(&q);
        assert!(results.len() <= 3);
    }

    #[test]
    fn test_clear_index() {
        let mut index = make_index();
        index.add_segment("a", TimeRange::new(0, 100), vec!["x".to_string()], 0.5);
        index.clear();
        assert!(index.is_empty());
    }

    #[test]
    fn test_indexed_segment_with_metadata() {
        let seg = IndexedSegment::new(1, "a", TimeRange::new(0, 100), vec![], 0.5)
            .with_metadata("resolution", "1920x1080");
        assert_eq!(
            seg.metadata.get("resolution").map(String::as_str),
            Some("1920x1080")
        );
    }
}
