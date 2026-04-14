//! Search index warming for faster first-query response.
//!
//! Maintains a warm query cache by pre-executing high-frequency queries against
//! the [`search_index::SearchIndex`] during system startup and on a periodic
//! refresh schedule.  Results are stored in an in-memory cache keyed by the
//! normalised query string so that the first real-user request hits the cache
//! rather than the cold index.
//!
//! # Design
//!
//! - [`WarmingConfig`] — tuning parameters (top-K queries, TTL, concurrency).
//! - [`QueryFrequencyTracker`] — records how many times each query has been
//!   executed so that the warmer knows which queries to pre-heat.
//! - [`WarmResultEntry`] — a cached result set with an expiry timestamp.
//! - [`IndexWarmer`] — orchestrates warm-up by replaying popular queries
//!   against a [`SearchIndex`] and storing results.
//! - [`WarmingStats`] — runtime metrics (hits, misses, evictions).

use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::search_index::SearchIndex;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the index warming subsystem.
#[derive(Debug, Clone)]
pub struct WarmingConfig {
    /// Number of most-frequent queries to keep warm at all times.
    pub top_k_queries: usize,
    /// How long a warm cache entry remains valid before re-warming.
    pub entry_ttl: Duration,
    /// Maximum number of result documents stored per cached query.
    pub max_results_per_query: usize,
    /// Minimum number of times a query must have been seen before it qualifies
    /// for warming (avoids warming one-off queries).
    pub min_query_frequency: u64,
}

impl Default for WarmingConfig {
    fn default() -> Self {
        Self {
            top_k_queries: 50,
            entry_ttl: Duration::from_secs(300), // 5 minutes
            max_results_per_query: 100,
            min_query_frequency: 3,
        }
    }
}

// ---------------------------------------------------------------------------
// Query frequency tracker
// ---------------------------------------------------------------------------

/// Tracks how often each query string has been executed.
///
/// Used by [`IndexWarmer`] to determine which queries are worth pre-heating.
#[derive(Debug, Default)]
pub struct QueryFrequencyTracker {
    counts: HashMap<String, u64>,
}

impl QueryFrequencyTracker {
    /// Create a new, empty tracker.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one execution of `query`.
    pub fn record(&mut self, query: &str) {
        let key = Self::normalize(query);
        *self.counts.entry(key).or_insert(0) += 1;
    }

    /// Return the execution count for `query`.
    #[must_use]
    pub fn count(&self, query: &str) -> u64 {
        self.counts.get(&Self::normalize(query)).copied().unwrap_or(0)
    }

    /// Return the top `n` queries by execution count, sorted descending.
    #[must_use]
    pub fn top_queries(&self, n: usize) -> Vec<(String, u64)> {
        let mut pairs: Vec<(String, u64)> = self
            .counts
            .iter()
            .map(|(k, &v)| (k.clone(), v))
            .collect();
        pairs.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        pairs.truncate(n);
        pairs
    }

    /// Return the total number of distinct queries tracked.
    #[must_use]
    pub fn distinct_count(&self) -> usize {
        self.counts.len()
    }

    /// Remove all tracking data.
    pub fn clear(&mut self) {
        self.counts.clear();
    }

    /// Normalise a query string: lowercase and collapse whitespace.
    fn normalize(query: &str) -> String {
        query.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase()
    }
}

// ---------------------------------------------------------------------------
// Cached result entry
// ---------------------------------------------------------------------------

/// A single cached warm result set for one query.
#[derive(Debug, Clone)]
pub struct WarmResultEntry {
    /// The normalised query string.
    pub query: String,
    /// IDs of matching documents (ordered by relevance score descending).
    pub document_ids: Vec<String>,
    /// Highest relevance score among the cached results.
    pub top_score: f32,
    /// When this entry was last refreshed.
    pub warmed_at: Instant,
    /// When this entry expires and must be re-warmed.
    pub expires_at: Instant,
}

impl WarmResultEntry {
    /// Create a new warm result entry.
    #[must_use]
    pub fn new(query: String, document_ids: Vec<String>, top_score: f32, ttl: Duration) -> Self {
        let now = Instant::now();
        Self {
            query,
            document_ids,
            top_score,
            warmed_at: now,
            expires_at: now + ttl,
        }
    }

    /// Returns `true` if this entry is still within its TTL.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        Instant::now() < self.expires_at
    }

    /// Returns how long ago this entry was warmed.
    #[must_use]
    pub fn age(&self) -> Duration {
        self.warmed_at.elapsed()
    }

    /// Number of documents cached for this query.
    #[must_use]
    pub fn result_count(&self) -> usize {
        self.document_ids.len()
    }
}

// ---------------------------------------------------------------------------
// Warming statistics
// ---------------------------------------------------------------------------

/// Runtime statistics for the index warmer.
#[derive(Debug, Default, Clone)]
pub struct WarmingStats {
    /// Number of cache hits (query served from warm cache).
    pub hits: u64,
    /// Number of cache misses (query not in warm cache or expired).
    pub misses: u64,
    /// Number of times expired entries were evicted and re-warmed.
    pub evictions: u64,
    /// Total number of warm-up executions performed.
    pub warm_up_runs: u64,
    /// Total number of queries currently in the warm cache.
    pub cached_queries: usize,
}

impl WarmingStats {
    /// Hit ratio in the range `[0.0, 1.0]`.
    #[must_use]
    pub fn hit_ratio(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

// ---------------------------------------------------------------------------
// Index warmer
// ---------------------------------------------------------------------------

/// Orchestrates index warming: replays popular queries against a
/// [`SearchIndex`] and stores pre-computed results in an in-memory cache.
pub struct IndexWarmer {
    config: WarmingConfig,
    cache: HashMap<String, WarmResultEntry>,
    stats: WarmingStats,
}

impl IndexWarmer {
    /// Create a new warmer with the given configuration.
    #[must_use]
    pub fn new(config: WarmingConfig) -> Self {
        Self {
            config,
            cache: HashMap::new(),
            stats: WarmingStats::default(),
        }
    }

    /// Create a warmer with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(WarmingConfig::default())
    }

    /// Warm the cache for the given set of queries by executing each against
    /// `index`.  Existing valid entries are preserved; expired entries are
    /// replaced.
    ///
    /// Returns the number of queries (re-)warmed.
    pub fn warm(&mut self, queries: &[String], index: &SearchIndex) -> usize {
        let mut warmed = 0usize;
        for raw_query in queries {
            let key = Self::cache_key(raw_query);
            // Skip if still valid
            if self.cache.get(&key).map(|e| e.is_valid()).unwrap_or(false) {
                continue;
            }
            // Execute and cache
            let results = index.search(raw_query);
            let document_ids: Vec<String> = results
                .iter()
                .take(self.config.max_results_per_query)
                .map(|(doc, _)| doc.id.clone())
                .collect();
            let top_score = results.first().map(|(_, s)| *s).unwrap_or(0.0);
            let entry = WarmResultEntry::new(key.clone(), document_ids, top_score, self.config.entry_ttl);
            self.cache.insert(key, entry);
            warmed += 1;
        }
        self.stats.warm_up_runs += 1;
        self.stats.cached_queries = self.cache.len();
        warmed
    }

    /// Warm based on the top-K queries observed by `tracker`, applying the
    /// configured `min_query_frequency` threshold.
    pub fn warm_from_tracker(&mut self, tracker: &QueryFrequencyTracker, index: &SearchIndex) -> usize {
        let candidates: Vec<String> = tracker
            .top_queries(self.config.top_k_queries)
            .into_iter()
            .filter(|(_, count)| *count >= self.config.min_query_frequency)
            .map(|(q, _)| q)
            .collect();
        self.warm(&candidates, index)
    }

    /// Look up a query in the warm cache.
    ///
    /// - Returns `Some(&WarmResultEntry)` on a hit.
    /// - Returns `None` on a miss (entry absent or expired).
    pub fn lookup(&mut self, query: &str) -> Option<&WarmResultEntry> {
        let key = Self::cache_key(query);
        if let Some(entry) = self.cache.get(&key) {
            if entry.is_valid() {
                self.stats.hits += 1;
                return self.cache.get(&key);
            }
            // Expired — evict
            self.stats.evictions += 1;
            self.cache.remove(&key);
        }
        self.stats.misses += 1;
        None
    }

    /// Evict all expired entries from the cache.
    ///
    /// Returns the number of entries evicted.
    pub fn evict_expired(&mut self) -> usize {
        let before = self.cache.len();
        self.cache.retain(|_, entry| entry.is_valid());
        let evicted = before - self.cache.len();
        self.stats.evictions += evicted as u64;
        self.stats.cached_queries = self.cache.len();
        evicted
    }

    /// Invalidate (remove) a specific query from the cache.
    ///
    /// Returns `true` if the entry was present and removed.
    pub fn invalidate(&mut self, query: &str) -> bool {
        let key = Self::cache_key(query);
        let removed = self.cache.remove(&key).is_some();
        self.stats.cached_queries = self.cache.len();
        removed
    }

    /// Clear the entire cache.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
        self.stats.cached_queries = 0;
    }

    /// Return a snapshot of the current warming statistics.
    #[must_use]
    pub fn stats(&self) -> &WarmingStats {
        &self.stats
    }

    /// Return the number of entries currently in the warm cache.
    #[must_use]
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    /// Return the configuration used by this warmer.
    #[must_use]
    pub fn config(&self) -> &WarmingConfig {
        &self.config
    }

    /// Normalise a raw query into a cache key.
    fn cache_key(query: &str) -> String {
        query.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase()
    }
}

// ---------------------------------------------------------------------------
// Scheduled warm-up plan
// ---------------------------------------------------------------------------

/// Describes a set of seed queries to use for warm-up on startup.
#[derive(Debug, Clone, Default)]
pub struct WarmupPlan {
    /// Fixed queries that are always warmed (e.g. common dashboard queries).
    pub seed_queries: Vec<String>,
    /// Whether to also warm from the frequency tracker's top-K list.
    pub include_top_k: bool,
}

impl WarmupPlan {
    /// Create a new plan with the given seed queries.
    #[must_use]
    pub fn new(seed_queries: Vec<String>, include_top_k: bool) -> Self {
        Self { seed_queries, include_top_k }
    }

    /// Execute this plan against `warmer`, `tracker`, and `index`.
    ///
    /// Returns the total number of queries warmed.
    pub fn execute(
        &self,
        warmer: &mut IndexWarmer,
        tracker: &QueryFrequencyTracker,
        index: &SearchIndex,
    ) -> usize {
        let mut total = warmer.warm(&self.seed_queries, index);
        if self.include_top_k {
            total += warmer.warm_from_tracker(tracker, index);
        }
        total
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search_index::{AssetDocument, IndexedField, SearchIndex};

    fn build_index() -> SearchIndex {
        let mut idx = SearchIndex::new();
        let mut doc1 = AssetDocument::new("doc-1");
        doc1.add_field(IndexedField::new("title", "London News Reel", 2.0, true));
        doc1.add_field(IndexedField::new("tags", "news uk london", 1.0, true));
        idx.add(doc1);

        let mut doc2 = AssetDocument::new("doc-2");
        doc2.add_field(IndexedField::new("title", "Sports Highlights 2024", 2.0, true));
        doc2.add_field(IndexedField::new("tags", "sports football", 1.0, true));
        idx.add(doc2);

        let mut doc3 = AssetDocument::new("doc-3");
        doc3.add_field(IndexedField::new("title", "Budget Report", 2.0, true));
        doc3.add_field(IndexedField::new("tags", "finance business news", 1.0, true));
        idx.add(doc3);

        idx
    }

    // --- QueryFrequencyTracker ---

    #[test]
    fn test_tracker_record_and_count() {
        let mut t = QueryFrequencyTracker::new();
        t.record("news");
        t.record("news");
        t.record("sports");
        assert_eq!(t.count("news"), 2);
        assert_eq!(t.count("sports"), 1);
        assert_eq!(t.count("unknown"), 0);
    }

    #[test]
    fn test_tracker_normalizes_case_and_whitespace() {
        let mut t = QueryFrequencyTracker::new();
        t.record("Breaking  NEWS");
        t.record("breaking news");
        assert_eq!(t.count("breaking news"), 2);
    }

    #[test]
    fn test_tracker_top_queries_order() {
        let mut t = QueryFrequencyTracker::new();
        for _ in 0..5 { t.record("sports"); }
        for _ in 0..3 { t.record("news"); }
        t.record("finance");
        let top = t.top_queries(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, "sports");
        assert_eq!(top[1].0, "news");
    }

    #[test]
    fn test_tracker_distinct_count() {
        let mut t = QueryFrequencyTracker::new();
        t.record("a");
        t.record("b");
        t.record("a");
        assert_eq!(t.distinct_count(), 2);
    }

    #[test]
    fn test_tracker_clear() {
        let mut t = QueryFrequencyTracker::new();
        t.record("x");
        t.clear();
        assert_eq!(t.distinct_count(), 0);
    }

    // --- WarmResultEntry ---

    #[test]
    fn test_warm_entry_is_valid_within_ttl() {
        let entry = WarmResultEntry::new(
            "news".into(),
            vec!["doc-1".into()],
            2.0,
            Duration::from_secs(60),
        );
        assert!(entry.is_valid());
        assert_eq!(entry.result_count(), 1);
    }

    #[test]
    fn test_warm_entry_expired_zero_ttl() {
        let entry = WarmResultEntry::new(
            "news".into(),
            vec!["doc-1".into()],
            2.0,
            Duration::from_secs(0),
        );
        // With a zero TTL the entry should expire immediately (allow tiny epsilon)
        std::thread::sleep(Duration::from_millis(1));
        assert!(!entry.is_valid());
    }

    // --- IndexWarmer ---

    #[test]
    fn test_warmer_warm_populates_cache() {
        let index = build_index();
        let mut warmer = IndexWarmer::with_defaults();
        let queries = vec!["news".to_string(), "sports".to_string()];
        let warmed = warmer.warm(&queries, &index);
        assert_eq!(warmed, 2);
        assert_eq!(warmer.cache_size(), 2);
    }

    #[test]
    fn test_warmer_lookup_hit() {
        let index = build_index();
        let mut warmer = IndexWarmer::with_defaults();
        warmer.warm(&["news".to_string()], &index);
        let entry = warmer.lookup("news");
        assert!(entry.is_some());
        assert!(!entry.expect("should be Some").document_ids.is_empty());
    }

    #[test]
    fn test_warmer_lookup_miss_unknown_query() {
        let index = build_index();
        let mut warmer = IndexWarmer::with_defaults();
        warmer.warm(&["sports".to_string()], &index);
        let entry = warmer.lookup("cooking");
        assert!(entry.is_none());
    }

    #[test]
    fn test_warmer_stats_hit_miss_ratio() {
        let index = build_index();
        let mut warmer = IndexWarmer::with_defaults();
        warmer.warm(&["news".to_string()], &index);
        warmer.lookup("news");  // hit
        warmer.lookup("cooking"); // miss
        let stats = warmer.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        let ratio = stats.hit_ratio();
        assert!((ratio - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_warmer_invalidate() {
        let index = build_index();
        let mut warmer = IndexWarmer::with_defaults();
        warmer.warm(&["sports".to_string()], &index);
        assert!(warmer.invalidate("sports"));
        assert_eq!(warmer.cache_size(), 0);
        assert!(!warmer.invalidate("sports"));
    }

    #[test]
    fn test_warmer_clear_cache() {
        let index = build_index();
        let mut warmer = IndexWarmer::with_defaults();
        warmer.warm(&["news".to_string(), "sports".to_string()], &index);
        warmer.clear_cache();
        assert_eq!(warmer.cache_size(), 0);
    }

    #[test]
    fn test_warmer_skips_already_valid_entries() {
        let index = build_index();
        let mut warmer = IndexWarmer::with_defaults();
        warmer.warm(&["news".to_string()], &index);
        // Second warm call should not re-warm valid entries
        let warmed = warmer.warm(&["news".to_string()], &index);
        assert_eq!(warmed, 0);
    }

    #[test]
    fn test_warmer_from_tracker_respects_min_frequency() {
        let index = build_index();
        let mut warmer = IndexWarmer::new(WarmingConfig {
            min_query_frequency: 5,
            top_k_queries: 10,
            ..WarmingConfig::default()
        });
        let mut tracker = QueryFrequencyTracker::new();
        // Record "news" only 3 times — below threshold of 5
        for _ in 0..3 { tracker.record("news"); }
        // Record "sports" 6 times — above threshold
        for _ in 0..6 { tracker.record("sports"); }

        let warmed = warmer.warm_from_tracker(&tracker, &index);
        // Only "sports" qualifies
        assert_eq!(warmed, 1);
        assert!(warmer.lookup("sports").is_some());
        assert!(warmer.lookup("news").is_none());
    }

    #[test]
    fn test_warmup_plan_seed_queries() {
        let index = build_index();
        let mut warmer = IndexWarmer::with_defaults();
        let tracker = QueryFrequencyTracker::new();
        let plan = WarmupPlan::new(vec!["news".to_string()], false);
        let total = plan.execute(&mut warmer, &tracker, &index);
        assert_eq!(total, 1);
    }

    #[test]
    fn test_warming_config_defaults() {
        let cfg = WarmingConfig::default();
        assert_eq!(cfg.top_k_queries, 50);
        assert_eq!(cfg.min_query_frequency, 3);
        assert!(cfg.entry_ttl.as_secs() > 0);
    }
}
