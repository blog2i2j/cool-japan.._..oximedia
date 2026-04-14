//! Batch recommendation generation for offline/pre-computation scenarios.
//!
//! This module provides the ability to generate recommendations for large numbers
//! of users in batch, enabling offline pre-computation, nightly jobs, and
//! scheduled recommendation refreshes.
//!
//! # Architecture
//!
//! ```text
//! BatchRecommendationJob
//!   │
//!   ├── UserBatch[]          → per-shard worker (rayon)
//!   │                              │
//!   │                         ItemScorer (content/cf/hybrid)
//!   │                              │
//!   │                         RecommendationShard[]
//!   │
//!   └── BatchResult          → merge + write to BatchResultStore
//! ```
//!
//! # Pre-computation Flow
//!
//! 1. Create a [`BatchJob`] with user IDs, scoring strategy, and limits.
//! 2. Call [`BatchProcessor::run`] which shards users and scores in parallel.
//! 3. The [`BatchResultStore`] caches results with a TTL (epoch-ms).
//! 4. At serve time, call [`BatchResultStore::get`] for sub-millisecond lookups.

#![allow(dead_code)]

use std::collections::HashMap;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// Strategy used when scoring candidates in batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BatchStrategy {
    /// Score by global item popularity (fast baseline, no user model needed).
    Popularity,
    /// Score using per-user content-based features stored in a feature map.
    ContentBased,
    /// Score using pre-computed collaborative-filtering embeddings.
    Collaborative,
    /// Weighted blend of content-based and collaborative scores.
    Hybrid,
}

impl std::fmt::Display for BatchStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Popularity => write!(f, "popularity"),
            Self::ContentBased => write!(f, "content_based"),
            Self::Collaborative => write!(f, "collaborative"),
            Self::Hybrid => write!(f, "hybrid"),
        }
    }
}

/// A single item available for batch recommendation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchItem {
    /// Item identifier.
    pub item_id: String,
    /// Global popularity score in [0, 1].
    pub popularity: f64,
    /// Category tags.
    pub categories: Vec<String>,
    /// Dense feature vector (content representation).
    pub features: Vec<f64>,
}

impl BatchItem {
    /// Create a batch item with the given ID and popularity.
    #[must_use]
    pub fn new(item_id: impl Into<String>, popularity: f64) -> Self {
        Self {
            item_id: item_id.into(),
            popularity: popularity.clamp(0.0, 1.0),
            categories: Vec::new(),
            features: Vec::new(),
        }
    }

    /// Builder: add categories.
    #[must_use]
    pub fn with_categories(mut self, categories: Vec<String>) -> Self {
        self.categories = categories;
        self
    }

    /// Builder: set feature vector.
    #[must_use]
    pub fn with_features(mut self, features: Vec<f64>) -> Self {
        self.features = features;
        self
    }
}

/// Per-user scoring context provided to the batch processor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserContext {
    /// User identifier.
    pub user_id: String,
    /// User interest feature vector (for content-based scoring).
    pub interest_vector: Vec<f64>,
    /// User collaborative embedding (for CF scoring).
    pub cf_embedding: Vec<f64>,
    /// Items the user has already interacted with (excluded from results).
    pub seen_items: Vec<String>,
}

impl UserContext {
    /// Create a minimal user context with just an ID.
    #[must_use]
    pub fn new(user_id: impl Into<String>) -> Self {
        Self {
            user_id: user_id.into(),
            interest_vector: Vec::new(),
            cf_embedding: Vec::new(),
            seen_items: Vec::new(),
        }
    }

    /// Builder: set interest vector.
    #[must_use]
    pub fn with_interest_vector(mut self, v: Vec<f64>) -> Self {
        self.interest_vector = v;
        self
    }

    /// Builder: set CF embedding.
    #[must_use]
    pub fn with_cf_embedding(mut self, v: Vec<f64>) -> Self {
        self.cf_embedding = v;
        self
    }

    /// Builder: add seen items (will be excluded from results).
    #[must_use]
    pub fn with_seen_items(mut self, items: Vec<String>) -> Self {
        self.seen_items = items;
        self
    }
}

/// A single recommendation produced by the batch processor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchRecommendation {
    /// Item identifier.
    pub item_id: String,
    /// Score assigned.
    pub score: f64,
    /// Rank (1-based).
    pub rank: usize,
    /// Strategy that produced this recommendation.
    pub strategy: BatchStrategy,
}

/// Pre-computed recommendations for one user.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserRecommendations {
    /// User identifier.
    pub user_id: String,
    /// Ordered recommendations (descending score).
    pub recommendations: Vec<BatchRecommendation>,
    /// Unix timestamp (millis) when these were computed.
    pub computed_at_ms: i64,
    /// TTL in milliseconds (0 = never expires).
    pub ttl_ms: i64,
}

impl UserRecommendations {
    /// Returns `true` if this result has expired relative to `now_ms`.
    #[must_use]
    pub fn is_expired(&self, now_ms: i64) -> bool {
        if self.ttl_ms == 0 {
            return false;
        }
        now_ms > self.computed_at_ms + self.ttl_ms
    }
}

/// Configuration for a batch recommendation job.
#[derive(Debug, Clone)]
pub struct BatchJobConfig {
    /// Maximum recommendations per user.
    pub top_k: usize,
    /// Scoring strategy.
    pub strategy: BatchStrategy,
    /// Hybrid blend weight for content-based (0–1).  1 − weight goes to CF.
    pub hybrid_content_weight: f64,
    /// TTL for computed results in milliseconds (0 = never expires).
    pub result_ttl_ms: i64,
    /// Shard size: number of users processed per rayon task.
    pub shard_size: usize,
}

impl Default for BatchJobConfig {
    fn default() -> Self {
        Self {
            top_k: 20,
            strategy: BatchStrategy::Hybrid,
            hybrid_content_weight: 0.5,
            result_ttl_ms: 3_600_000, // 1 hour
            shard_size: 64,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal scoring helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Cosine similarity between two equal-length slices.  Returns 0 if either is zero.
fn cosine_sim(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na < f64::EPSILON || nb < f64::EPSILON {
        return 0.0;
    }
    (dot / (na * nb)).clamp(-1.0, 1.0)
}

/// Score a single (user, item) pair under the given strategy.
fn score_pair(item: &BatchItem, user: &UserContext, config: &BatchJobConfig) -> f64 {
    match config.strategy {
        BatchStrategy::Popularity => item.popularity,
        BatchStrategy::ContentBased => {
            if user.interest_vector.is_empty() || item.features.is_empty() {
                item.popularity * 0.5
            } else {
                cosine_sim(&user.interest_vector, &item.features)
            }
        }
        BatchStrategy::Collaborative => {
            if user.cf_embedding.is_empty() || item.features.is_empty() {
                item.popularity * 0.5
            } else {
                cosine_sim(&user.cf_embedding, &item.features)
            }
        }
        BatchStrategy::Hybrid => {
            let cb = if user.interest_vector.is_empty() || item.features.is_empty() {
                item.popularity * 0.5
            } else {
                cosine_sim(&user.interest_vector, &item.features)
            };
            let cf = if user.cf_embedding.is_empty() || item.features.is_empty() {
                item.popularity * 0.5
            } else {
                cosine_sim(&user.cf_embedding, &item.features)
            };
            let w = config.hybrid_content_weight.clamp(0.0, 1.0);
            w * cb + (1.0 - w) * cf
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BatchProcessor
// ─────────────────────────────────────────────────────────────────────────────

/// Processes batch recommendation jobs.
///
/// Uses rayon to shard the user list and score items in parallel.
#[derive(Debug)]
pub struct BatchProcessor {
    config: BatchJobConfig,
}

impl BatchProcessor {
    /// Create a processor with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: BatchJobConfig::default(),
        }
    }

    /// Create a processor with custom configuration.
    #[must_use]
    pub fn with_config(config: BatchJobConfig) -> Self {
        Self { config }
    }

    /// Run the batch job: score all items for all users in parallel.
    ///
    /// `now_ms` is the current time (epoch milliseconds), used to set the
    /// `computed_at_ms` timestamp on results.
    #[must_use]
    pub fn run(
        &self,
        users: &[UserContext],
        catalog: &[BatchItem],
        now_ms: i64,
    ) -> Vec<UserRecommendations> {
        // Shard users for parallel processing.
        let shards: Vec<&[UserContext]> = users.chunks(self.config.shard_size.max(1)).collect();

        let results: Vec<Vec<UserRecommendations>> = shards
            .par_iter()
            .map(|shard| {
                shard
                    .iter()
                    .map(|user| self.score_user(user, catalog, now_ms))
                    .collect()
            })
            .collect();

        results.into_iter().flatten().collect()
    }

    /// Score one user against all catalog items and return their recommendations.
    fn score_user(
        &self,
        user: &UserContext,
        catalog: &[BatchItem],
        now_ms: i64,
    ) -> UserRecommendations {
        use std::collections::HashSet;
        let seen: HashSet<&str> = user.seen_items.iter().map(String::as_str).collect();

        // Score and filter.
        let mut scored: Vec<(f64, &str)> = catalog
            .iter()
            .filter(|item| !seen.contains(item.item_id.as_str()))
            .map(|item| (score_pair(item, user, &self.config), item.item_id.as_str()))
            .collect();

        // Sort descending.
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(self.config.top_k);

        let recommendations: Vec<BatchRecommendation> = scored
            .into_iter()
            .enumerate()
            .map(|(i, (score, item_id))| BatchRecommendation {
                item_id: item_id.to_string(),
                score,
                rank: i + 1,
                strategy: self.config.strategy,
            })
            .collect();

        UserRecommendations {
            user_id: user.user_id.clone(),
            recommendations,
            computed_at_ms: now_ms,
            ttl_ms: self.config.result_ttl_ms,
        }
    }
}

impl Default for BatchProcessor {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BatchResultStore — in-memory pre-computed result cache
// ─────────────────────────────────────────────────────────────────────────────

/// In-memory store for pre-computed batch recommendations.
///
/// Supports TTL-based invalidation: stale entries are filtered on read.
#[derive(Debug, Default)]
pub struct BatchResultStore {
    results: HashMap<String, UserRecommendations>,
}

impl BatchResultStore {
    /// Create a new empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Store recommendations for a user.
    pub fn put(&mut self, recs: UserRecommendations) {
        self.results.insert(recs.user_id.clone(), recs);
    }

    /// Bulk-store results from a batch run.
    pub fn put_batch(&mut self, batch: Vec<UserRecommendations>) {
        for recs in batch {
            self.put(recs);
        }
    }

    /// Retrieve pre-computed recommendations for a user.
    ///
    /// Returns `None` if the user has no entry or their entry has expired.
    #[must_use]
    pub fn get(&self, user_id: &str, now_ms: i64) -> Option<&UserRecommendations> {
        let recs = self.results.get(user_id)?;
        if recs.is_expired(now_ms) {
            return None;
        }
        Some(recs)
    }

    /// Remove all expired entries relative to `now_ms`.
    pub fn evict_expired(&mut self, now_ms: i64) {
        self.results.retain(|_, v| !v.is_expired(now_ms));
    }

    /// Number of stored users.
    #[must_use]
    pub fn len(&self) -> usize {
        self.results.len()
    }

    /// Returns `true` if no results are stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.results.is_empty()
    }

    /// Clear all stored results.
    pub fn clear(&mut self) {
        self.results.clear();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Batch job descriptor (convenience wrapper)
// ─────────────────────────────────────────────────────────────────────────────

/// Describes a batch recommendation job including users, catalog, and config.
#[derive(Debug)]
pub struct BatchJob {
    /// Users to generate recommendations for.
    pub users: Vec<UserContext>,
    /// Catalog of items to score.
    pub catalog: Vec<BatchItem>,
    /// Job configuration.
    pub config: BatchJobConfig,
}

impl BatchJob {
    /// Create a new batch job.
    #[must_use]
    pub fn new(users: Vec<UserContext>, catalog: Vec<BatchItem>) -> Self {
        Self {
            users,
            catalog,
            config: BatchJobConfig::default(),
        }
    }

    /// Override the configuration.
    #[must_use]
    pub fn with_config(mut self, config: BatchJobConfig) -> Self {
        self.config = config;
        self
    }

    /// Execute the job and return results.
    #[must_use]
    pub fn run(&self, now_ms: i64) -> Vec<UserRecommendations> {
        let processor = BatchProcessor::with_config(self.config.clone());
        processor.run(&self.users, &self.catalog, now_ms)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_catalog() -> Vec<BatchItem> {
        vec![
            BatchItem::new("item_a", 0.9)
                .with_categories(vec!["action".to_string()])
                .with_features(vec![1.0, 0.0, 0.0]),
            BatchItem::new("item_b", 0.7)
                .with_categories(vec!["drama".to_string()])
                .with_features(vec![0.0, 1.0, 0.0]),
            BatchItem::new("item_c", 0.5)
                .with_categories(vec!["comedy".to_string()])
                .with_features(vec![0.0, 0.0, 1.0]),
            BatchItem::new("item_d", 0.6)
                .with_categories(vec!["action".to_string(), "drama".to_string()])
                .with_features(vec![0.7, 0.7, 0.0]),
        ]
    }

    fn make_users() -> Vec<UserContext> {
        vec![
            UserContext::new("alice")
                .with_interest_vector(vec![1.0, 0.0, 0.0])
                .with_cf_embedding(vec![1.0, 0.0, 0.0]),
            UserContext::new("bob")
                .with_interest_vector(vec![0.0, 1.0, 0.0])
                .with_cf_embedding(vec![0.0, 1.0, 0.0]),
        ]
    }

    // ─── BatchItem ──────────────────────────────────────────────────────────

    #[test]
    fn test_batch_item_creation() {
        let item = BatchItem::new("x", 0.75);
        assert_eq!(item.item_id, "x");
        assert!((item.popularity - 0.75).abs() < f64::EPSILON);
        assert!(item.categories.is_empty());
    }

    #[test]
    fn test_batch_item_popularity_clamp() {
        let item = BatchItem::new("x", 1.5);
        assert!((item.popularity - 1.0).abs() < f64::EPSILON);
        let item2 = BatchItem::new("y", -0.5);
        assert!((item2.popularity - 0.0).abs() < f64::EPSILON);
    }

    // ─── UserContext ─────────────────────────────────────────────────────────

    #[test]
    fn test_user_context_seen_items_excluded() {
        let catalog = make_catalog();
        let user = UserContext::new("u1")
            .with_seen_items(vec!["item_a".to_string(), "item_b".to_string()]);
        let config = BatchJobConfig {
            strategy: BatchStrategy::Popularity,
            top_k: 10,
            ..Default::default()
        };
        let processor = BatchProcessor::with_config(config);
        let results = processor.run(&[user], &catalog, 0);
        assert_eq!(results.len(), 1);
        let recs = &results[0].recommendations;
        assert!(recs.iter().all(|r| r.item_id != "item_a"));
        assert!(recs.iter().all(|r| r.item_id != "item_b"));
    }

    // ─── BatchProcessor - Popularity ─────────────────────────────────────────

    #[test]
    fn test_popularity_strategy_sorts_by_popularity() {
        let catalog = make_catalog();
        let users = vec![UserContext::new("u1")];
        let config = BatchJobConfig {
            strategy: BatchStrategy::Popularity,
            top_k: 4,
            ..Default::default()
        };
        let processor = BatchProcessor::with_config(config);
        let results = processor.run(&users, &catalog, 0);
        let recs = &results[0].recommendations;
        assert!(!recs.is_empty());
        // First item should have highest popularity (item_a = 0.9)
        assert_eq!(recs[0].item_id, "item_a");
    }

    // ─── BatchProcessor - ContentBased ───────────────────────────────────────

    #[test]
    fn test_content_based_prefers_similar_items() {
        let catalog = make_catalog();
        // Alice has interest [1,0,0] → item_a ([1,0,0]) should score highest
        let users = vec![UserContext::new("alice").with_interest_vector(vec![1.0, 0.0, 0.0])];
        let config = BatchJobConfig {
            strategy: BatchStrategy::ContentBased,
            top_k: 4,
            ..Default::default()
        };
        let processor = BatchProcessor::with_config(config);
        let results = processor.run(&users, &catalog, 0);
        let recs = &results[0].recommendations;
        assert_eq!(recs[0].item_id, "item_a");
    }

    // ─── BatchProcessor - Hybrid ─────────────────────────────────────────────

    #[test]
    fn test_hybrid_strategy_returns_results() {
        let catalog = make_catalog();
        let users = make_users();
        let config = BatchJobConfig {
            strategy: BatchStrategy::Hybrid,
            top_k: 3,
            hybrid_content_weight: 0.5,
            ..Default::default()
        };
        let processor = BatchProcessor::with_config(config);
        let results = processor.run(&users, &catalog, 1000);
        assert_eq!(results.len(), 2);
        for result in &results {
            assert!(result.recommendations.len() <= 3);
            // Ranks are 1-based and contiguous
            for (i, rec) in result.recommendations.iter().enumerate() {
                assert_eq!(rec.rank, i + 1);
            }
        }
    }

    // ─── BatchProcessor - top_k limit ────────────────────────────────────────

    #[test]
    fn test_top_k_limit_respected() {
        let catalog = make_catalog();
        let users = vec![UserContext::new("u1")];
        let config = BatchJobConfig {
            strategy: BatchStrategy::Popularity,
            top_k: 2,
            ..Default::default()
        };
        let processor = BatchProcessor::with_config(config);
        let results = processor.run(&users, &catalog, 0);
        assert!(results[0].recommendations.len() <= 2);
    }

    // ─── BatchProcessor - parallel shards ────────────────────────────────────

    #[test]
    fn test_parallel_batch_multiple_users() {
        let catalog = make_catalog();
        let users: Vec<UserContext> = (0..20).map(|i| UserContext::new(format!("u{i}"))).collect();
        let config = BatchJobConfig {
            strategy: BatchStrategy::Popularity,
            top_k: 4,
            shard_size: 5,
            ..Default::default()
        };
        let processor = BatchProcessor::with_config(config);
        let results = processor.run(&users, &catalog, 0);
        assert_eq!(results.len(), 20);
    }

    // ─── BatchResultStore ────────────────────────────────────────────────────

    #[test]
    fn test_store_put_and_get() {
        let mut store = BatchResultStore::new();
        let recs = UserRecommendations {
            user_id: "u1".to_string(),
            recommendations: vec![],
            computed_at_ms: 1000,
            ttl_ms: 3600_000,
        };
        store.put(recs);
        assert_eq!(store.len(), 1);
        let retrieved = store.get("u1", 2000);
        assert!(retrieved.is_some());
    }

    #[test]
    fn test_store_ttl_expiry() {
        let mut store = BatchResultStore::new();
        let recs = UserRecommendations {
            user_id: "u1".to_string(),
            recommendations: vec![],
            computed_at_ms: 0,
            ttl_ms: 1000, // expires after 1 second
        };
        store.put(recs);
        // Not expired at t=500
        assert!(store.get("u1", 500).is_some());
        // Expired at t=2000
        assert!(store.get("u1", 2000).is_none());
    }

    #[test]
    fn test_store_evict_expired() {
        let mut store = BatchResultStore::new();
        store.put(UserRecommendations {
            user_id: "u1".to_string(),
            recommendations: vec![],
            computed_at_ms: 0,
            ttl_ms: 1000,
        });
        store.put(UserRecommendations {
            user_id: "u2".to_string(),
            recommendations: vec![],
            computed_at_ms: 0,
            ttl_ms: 0, // never expires
        });
        store.evict_expired(2000);
        assert_eq!(store.len(), 1);
        assert!(store.get("u2", 2000).is_some());
    }

    #[test]
    fn test_store_put_batch() {
        let mut store = BatchResultStore::new();
        let batch: Vec<UserRecommendations> = (0..5)
            .map(|i| UserRecommendations {
                user_id: format!("u{i}"),
                recommendations: vec![],
                computed_at_ms: 0,
                ttl_ms: 0,
            })
            .collect();
        store.put_batch(batch);
        assert_eq!(store.len(), 5);
    }

    // ─── BatchJob ────────────────────────────────────────────────────────────

    #[test]
    fn test_batch_job_run() {
        let catalog = make_catalog();
        let users = make_users();
        let job = BatchJob::new(users, catalog).with_config(BatchJobConfig {
            strategy: BatchStrategy::ContentBased,
            top_k: 3,
            ..Default::default()
        });
        let results = job.run(1_000_000);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].computed_at_ms, 1_000_000);
    }

    // ─── UserRecommendations.is_expired ──────────────────────────────────────

    #[test]
    fn test_user_recommendations_never_expires_when_ttl_zero() {
        let recs = UserRecommendations {
            user_id: "u1".to_string(),
            recommendations: vec![],
            computed_at_ms: 0,
            ttl_ms: 0,
        };
        assert!(!recs.is_expired(i64::MAX));
    }

    // ─── BatchStrategy display ───────────────────────────────────────────────

    #[test]
    fn test_batch_strategy_display() {
        assert_eq!(BatchStrategy::Popularity.to_string(), "popularity");
        assert_eq!(BatchStrategy::Hybrid.to_string(), "hybrid");
        assert_eq!(BatchStrategy::ContentBased.to_string(), "content_based");
        assert_eq!(BatchStrategy::Collaborative.to_string(), "collaborative");
    }

    // ─── Empty catalog / users ───────────────────────────────────────────────

    #[test]
    fn test_empty_catalog_returns_empty_recs() {
        let users = vec![UserContext::new("u1")];
        let processor = BatchProcessor::new();
        let results = processor.run(&users, &[], 0);
        assert_eq!(results.len(), 1);
        assert!(results[0].recommendations.is_empty());
    }

    #[test]
    fn test_empty_users_returns_empty() {
        let catalog = make_catalog();
        let processor = BatchProcessor::new();
        let results = processor.run(&[], &catalog, 0);
        assert!(results.is_empty());
    }
}
