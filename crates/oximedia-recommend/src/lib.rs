//! Content recommendation and discovery engine for `OxiMedia`.
//!
//! `oximedia-recommend` provides comprehensive recommendation capabilities for media platforms,
//! including content-based filtering, collaborative filtering, hybrid approaches, and
//! advanced personalization features.
//!
//! # Features
//!
//! - **Content-based Filtering**: Recommend similar content based on features
//! - **Collaborative Filtering**: User behavior-based recommendations
//! - **Hybrid Approach**: Combine multiple recommendation methods
//! - **Similarity Metrics**: Calculate content similarity using various metrics
//! - **User Profiles**: Build and manage user preference profiles
//! - **View History**: Track and analyze viewing patterns
//! - **Rating System**: Handle explicit and implicit ratings
//! - **Trending Detection**: Identify trending content in real-time
//! - **Personalization**: Context-aware personalized recommendations
//! - **Diversity**: Ensure recommendation diversity and avoid filter bubbles
//! - **Freshness**: Balance popular and new content
//!
//! # Modules
//!
//! - [`content`]: Content-based filtering and similarity
//! - [`collaborative`]: Collaborative filtering algorithms
//! - [`hybrid`]: Hybrid recommendation approaches
//! - [`profile`]: User profile management
//! - [`history`]: View history tracking and analysis
//! - [`rating`]: Rating system (explicit and implicit)
//! - [`trending`]: Trending content detection
//! - [`personalize`]: Personalization engine
//! - [`diversity`]: Diversity enforcement
//! - [`freshness`]: Fresh content promotion
//! - [`rank`]: Ranking and scoring
//! - [`explain`]: Recommendation explanations
//!
//! # Example
//!
//! ```
//! use oximedia_recommend::{RecommendationEngine, RecommendationRequest};
//! use uuid::Uuid;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a recommendation engine
//! let engine = RecommendationEngine::new();
//!
//! // Get recommendations for a user
//! let user_id = Uuid::new_v4();
//! let request = RecommendationRequest {
//!     user_id,
//!     limit: 10,
//!     ..Default::default()
//! };
//!
//! // let recommendations = engine.recommend(&request)?;
//! # Ok(())
//! # }
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::similar_names)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::missing_errors_doc)]
#![allow(dead_code)]

pub mod ab_test;
pub mod als;
pub mod bandits;
pub mod batch_recommend;
pub mod calibration;
pub mod cold_start;
pub mod collab_filter;
pub mod collaborative;
pub mod content;
pub mod content_based;
pub mod content_filter;
pub mod context_signal;
pub mod contextual_bandits;
pub mod cross_domain;
pub mod decay_model;
pub mod dense_linalg;
pub mod diversity;
pub mod diversity_rerank;
pub mod embargo;
pub mod error;
pub mod evaluation;
pub mod explain;
pub mod exploration_policy;
pub mod fairness;
pub mod feature_store;
pub mod federated;
pub mod feedback_signal;
pub mod freshness;
pub mod genre_affinity;
pub mod history;
pub mod hybrid;
pub mod impression_tracker;
pub mod item_similarity;
pub mod knowledge_graph;
pub mod lsh;
pub mod multi_objective;
pub mod novelty;
pub mod personalize;
pub mod playlist_generator;
pub mod popularity_bias;
pub mod profile;
pub mod rank;
pub mod ranking;
pub mod rate_limit;
pub mod rating;
pub mod recommendation_score;
pub mod score_cache;
pub mod sequence_model;
pub mod session;
pub mod session_recommend;
pub mod svd_pp;
pub mod trending;
pub mod trending_detection;
pub mod user_profile;
pub mod user_segment;
pub mod watch_history;

// Re-export commonly used items
pub use error::{RecommendError, RecommendResult};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Main recommendation engine coordinating all recommendation capabilities
pub struct RecommendationEngine {
    /// Content-based recommender
    content_recommender: content::similarity::ContentRecommender,
    /// Collaborative filtering engine
    collaborative_engine: collaborative::matrix::CollaborativeEngine,
    /// Hybrid combiner
    hybrid_combiner: hybrid::combine::HybridCombiner,
    /// User profile manager
    profile_manager: profile::user::UserProfileManager,
    /// View history tracker
    history_tracker: history::track::HistoryTracker,
    /// Rating manager
    rating_manager: rating::explicit::RatingManager,
    /// Trending detector
    trending_detector: trending::detect::TrendingDetector,
    /// Personalization engine
    personalization_engine: personalize::engine::PersonalizationEngine,
    /// Diversity enforcer
    diversity_enforcer: diversity::ensure::DiversityEnforcer,
    /// Freshness balancer
    freshness_balancer: freshness::balance::FreshnessBalancer,
    /// Optional rate limiter (None = no rate limiting)
    rate_limiter: Option<rate_limit::RecommendationRateLimiter>,
}

/// Recommendation request configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecommendationRequest {
    /// User ID to get recommendations for
    pub user_id: Uuid,
    /// Number of recommendations to return
    pub limit: usize,
    /// Content ID to base recommendations on (optional)
    pub content_id: Option<Uuid>,
    /// Recommendation strategy to use
    pub strategy: RecommendationStrategy,
    /// Context information
    pub context: RecommendationContext,
    /// Diversity settings
    pub diversity: DiversitySettings,
    /// Include explanations
    pub include_explanations: bool,
}

/// Recommendation strategy
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum RecommendationStrategy {
    /// Content-based filtering only
    ContentBased,
    /// Collaborative filtering only
    Collaborative,
    /// Hybrid approach (combines multiple methods)
    Hybrid,
    /// Personalized recommendations
    Personalized,
    /// Trending content
    Trending,
}

/// Context information for recommendations
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RecommendationContext {
    /// Current time (unix timestamp)
    pub timestamp: Option<i64>,
    /// Device type
    pub device: Option<String>,
    /// Location
    pub location: Option<String>,
    /// Session ID
    pub session_id: Option<Uuid>,
    /// Time of day
    pub time_of_day: Option<TimeOfDay>,
    /// Day of week
    pub day_of_week: Option<u8>,
}

/// Time of day categories
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TimeOfDay {
    /// Morning (6am-12pm)
    Morning,
    /// Afternoon (12pm-6pm)
    Afternoon,
    /// Evening (6pm-10pm)
    Evening,
    /// Night (10pm-6am)
    Night,
}

/// Diversity settings for recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiversitySettings {
    /// Enable diversity enforcement
    pub enabled: bool,
    /// Minimum category diversity (0.0-1.0)
    pub category_diversity: f32,
    /// Include novel/surprising items
    pub include_serendipity: bool,
    /// Serendipity weight (0.0-1.0)
    pub serendipity_weight: f32,
}

/// Recommendation result item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    /// Content ID
    pub content_id: Uuid,
    /// Recommendation score (0.0-1.0)
    pub score: f32,
    /// Rank in recommendation list
    pub rank: usize,
    /// Reasons for recommendation
    pub reasons: Vec<RecommendationReason>,
    /// Content metadata
    pub metadata: ContentMetadata,
    /// Explanation (if requested)
    pub explanation: Option<String>,
}

/// Reason for recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationReason {
    /// Similar to content the user liked
    SimilarToLiked {
        /// ID of the similar content that the user liked
        content_id: Uuid,
        /// Similarity score (0-1)
        similarity: f32,
    },
    /// Users similar to you also liked this
    CollaborativeFiltering {
        /// Confidence score (0-1)
        confidence: f32,
    },
    /// Trending in your area/globally
    Trending {
        /// Trending score indicating popularity momentum
        trending_score: f32,
    },
    /// Matches your interests
    MatchesProfile {
        /// Categories that matched the user profile
        categories: Vec<String>,
    },
    /// New/fresh content
    FreshContent {
        /// Number of days since publication
        published_days_ago: u32,
    },
    /// Popular content
    Popular {
        /// Total view count
        view_count: u64,
    },
    /// Completes a series/collection
    ContinueWatching {
        /// Watch progress (0-1)
        progress: f32,
    },
}

/// Content metadata for recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentMetadata {
    /// Content title
    pub title: String,
    /// Content description
    pub description: Option<String>,
    /// Categories/genres
    pub categories: Vec<String>,
    /// Duration (milliseconds)
    pub duration_ms: Option<i64>,
    /// Thumbnail URL
    pub thumbnail_url: Option<String>,
    /// Created timestamp
    pub created_at: i64,
    /// Average rating
    pub avg_rating: Option<f32>,
    /// View count
    pub view_count: u64,
}

/// Recommendation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecommendationResults {
    /// User ID
    pub user_id: Uuid,
    /// Recommended items
    pub recommendations: Vec<Recommendation>,
    /// Total candidates evaluated
    pub total_candidates: usize,
    /// Processing time (milliseconds)
    pub processing_time_ms: u64,
    /// Strategy used
    pub strategy: RecommendationStrategy,
}

impl RecommendationEngine {
    /// Create a new recommendation engine with no rate limiting.
    #[must_use]
    pub fn new() -> Self {
        Self {
            content_recommender: content::similarity::ContentRecommender::new(),
            collaborative_engine: collaborative::matrix::CollaborativeEngine::new(),
            hybrid_combiner: hybrid::combine::HybridCombiner::new(),
            profile_manager: profile::user::UserProfileManager::new(),
            history_tracker: history::track::HistoryTracker::new(),
            rating_manager: rating::explicit::RatingManager::new(),
            trending_detector: trending::detect::TrendingDetector::new(),
            personalization_engine: personalize::engine::PersonalizationEngine::new(),
            diversity_enforcer: diversity::ensure::DiversityEnforcer::new(),
            freshness_balancer: freshness::balance::FreshnessBalancer::new(0.3, 30),
            rate_limiter: None,
        }
    }

    /// Create a new recommendation engine with rate limiting enabled.
    ///
    /// `config` controls per-user and global token-bucket parameters.
    /// `now` is the initial Unix timestamp (seconds) used to seed the buckets.
    #[must_use]
    pub fn with_rate_limiter(config: rate_limit::RateLimitConfig, now: i64) -> Self {
        let limiter = rate_limit::RecommendationRateLimiter::new(config, now);
        Self {
            content_recommender: content::similarity::ContentRecommender::new(),
            collaborative_engine: collaborative::matrix::CollaborativeEngine::new(),
            hybrid_combiner: hybrid::combine::HybridCombiner::new(),
            profile_manager: profile::user::UserProfileManager::new(),
            history_tracker: history::track::HistoryTracker::new(),
            rating_manager: rating::explicit::RatingManager::new(),
            trending_detector: trending::detect::TrendingDetector::new(),
            personalization_engine: personalize::engine::PersonalizationEngine::new(),
            diversity_enforcer: diversity::ensure::DiversityEnforcer::new(),
            freshness_balancer: freshness::balance::FreshnessBalancer::new(0.3, 30),
            rate_limiter: Some(limiter),
        }
    }

    /// Enable (or replace) the rate limiter on an existing engine.
    pub fn set_rate_limiter(&mut self, config: rate_limit::RateLimitConfig, now: i64) {
        self.rate_limiter = Some(rate_limit::RecommendationRateLimiter::new(config, now));
    }

    /// Disable rate limiting on this engine.
    pub fn disable_rate_limiter(&mut self) {
        self.rate_limiter = None;
    }

    /// Returns `true` if rate limiting is currently enabled.
    #[must_use]
    pub fn has_rate_limiter(&self) -> bool {
        self.rate_limiter.is_some()
    }

    /// Query how many tokens remain for a user without consuming any.
    ///
    /// Returns `None` if rate limiting is disabled or if the user has no bucket yet.
    #[must_use]
    pub fn user_available_tokens(&self, user_id: &str) -> Option<f64> {
        self.rate_limiter
            .as_ref()
            .and_then(|rl| rl.user_available_tokens(user_id))
    }

    /// Query how many global tokens remain without consuming any.
    ///
    /// Returns `None` if rate limiting is disabled.
    #[must_use]
    pub fn global_available_tokens(&self) -> Option<f64> {
        self.rate_limiter
            .as_ref()
            .map(|rl| rl.global_available_tokens())
    }

    /// Get recommendations for a user
    ///
    /// For the `Hybrid` strategy, all sub-strategies are evaluated in parallel
    /// via rayon.  The resulting candidate lists are merged and deduplicated by
    /// content ID, taking the maximum score for any item that appeared in
    /// multiple strategy outputs.
    ///
    /// If a rate limiter is configured, the request is checked against both the
    /// per-user and global token buckets.  When the limit is exceeded a
    /// [`RecommendError::RateLimited`] error is returned immediately.
    ///
    /// # Errors
    ///
    /// Returns an error if recommendation generation fails or the caller is rate-limited.
    pub fn recommend(
        &mut self,
        request: &RecommendationRequest,
    ) -> RecommendResult<RecommendationResults> {
        use std::collections::HashMap;

        // Check rate limit before doing any work
        if let Some(ref mut rl) = self.rate_limiter {
            let user_key = request.user_id.to_string();
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() as i64;
            let decision = rl.check_and_consume(&user_key, now);
            if !decision.is_allowed() {
                return Err(RecommendError::RateLimited(format!(
                    "User {} exceeded rate limit: {decision:?}",
                    request.user_id
                )));
            }
        }

        let start = std::time::Instant::now();

        // Get candidates based on strategy.
        // For Hybrid, all strategies are evaluated in parallel with rayon.
        let candidates = match request.strategy {
            RecommendationStrategy::ContentBased => {
                self.get_content_based_recommendations(request)?
            }
            RecommendationStrategy::Collaborative => {
                self.get_collaborative_recommendations(request)?
            }
            RecommendationStrategy::Hybrid => self.get_hybrid_parallel(request)?,
            RecommendationStrategy::Personalized => {
                self.get_personalized_recommendations(request)?
            }
            RecommendationStrategy::Trending => self.get_trending_recommendations(request)?,
        };

        // Deduplicate merged candidates: keep highest score per content_id
        let candidates = {
            let mut best: HashMap<uuid::Uuid, Recommendation> = HashMap::new();
            for rec in candidates {
                let entry = best.entry(rec.content_id);
                entry
                    .and_modify(|existing| {
                        if rec.score > existing.score {
                            *existing = rec.clone();
                        }
                    })
                    .or_insert(rec);
            }
            let mut deduped: Vec<Recommendation> = best.into_values().collect();
            deduped.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            deduped
        };

        // Apply diversity if enabled
        let diverse_candidates = if request.diversity.enabled {
            self.diversity_enforcer
                .enforce_diversity(candidates, &request.diversity)?
        } else {
            candidates
        };

        // Apply freshness balancing
        let balanced_candidates = self.freshness_balancer.balance(diverse_candidates)?;

        // Rank and score
        let mut ranked = self.rank_recommendations(balanced_candidates)?;

        // Limit results
        ranked.truncate(request.limit);

        // Add explanations if requested
        if request.include_explanations {
            self.add_explanations(&mut ranked)?;
        }

        let processing_time_ms = start.elapsed().as_millis() as u64;

        let total_candidates = ranked.len();
        Ok(RecommendationResults {
            user_id: request.user_id,
            recommendations: ranked,
            total_candidates,
            processing_time_ms,
            strategy: request.strategy,
        })
    }

    /// Get content-based recommendations
    fn get_content_based_recommendations(
        &self,
        request: &RecommendationRequest,
    ) -> RecommendResult<Vec<Recommendation>> {
        self.content_recommender.recommend(request)
    }

    /// Get collaborative filtering recommendations
    fn get_collaborative_recommendations(
        &self,
        request: &RecommendationRequest,
    ) -> RecommendResult<Vec<Recommendation>> {
        self.collaborative_engine.recommend(request)
    }

    /// Get hybrid recommendations (single-threaded, delegates to HybridCombiner)
    fn get_hybrid_recommendations(
        &self,
        request: &RecommendationRequest,
    ) -> RecommendResult<Vec<Recommendation>> {
        self.hybrid_combiner.recommend(request)
    }

    /// Evaluate all non-Hybrid strategies in parallel via rayon and merge results.
    ///
    /// Each strategy is run concurrently; any strategy that fails is silently
    /// dropped so that a single sub-strategy error never blocks results from
    /// the others.
    fn get_hybrid_parallel(
        &self,
        request: &RecommendationRequest,
    ) -> RecommendResult<Vec<Recommendation>> {
        use rayon::prelude::*;

        // List of strategy labels to evaluate in parallel.
        // We exclude Hybrid itself to avoid recursion.
        let strategies: &[RecommendationStrategy] = &[
            RecommendationStrategy::ContentBased,
            RecommendationStrategy::Collaborative,
            RecommendationStrategy::Personalized,
            RecommendationStrategy::Trending,
        ];

        // Evaluate strategies in parallel; collect successful results.
        let parallel_results: Vec<Vec<Recommendation>> = strategies
            .par_iter()
            .filter_map(|strategy| {
                let result = match strategy {
                    RecommendationStrategy::ContentBased => {
                        self.get_content_based_recommendations(request)
                    }
                    RecommendationStrategy::Collaborative => {
                        self.get_collaborative_recommendations(request)
                    }
                    RecommendationStrategy::Personalized => {
                        self.get_personalized_recommendations(request)
                    }
                    RecommendationStrategy::Trending => self.get_trending_recommendations(request),
                    RecommendationStrategy::Hybrid => return None,
                };
                result.ok()
            })
            .collect();

        // Also get the HybridCombiner result (which has its own weighting logic)
        let combiner_result = self.hybrid_combiner.recommend(request).unwrap_or_default();

        // Merge all candidate lists
        let mut merged: Vec<Recommendation> = parallel_results.into_iter().flatten().collect();
        merged.extend(combiner_result);
        Ok(merged)
    }

    /// Get personalized recommendations
    fn get_personalized_recommendations(
        &self,
        request: &RecommendationRequest,
    ) -> RecommendResult<Vec<Recommendation>> {
        self.personalization_engine.recommend(request)
    }

    /// Get trending recommendations
    fn get_trending_recommendations(
        &self,
        request: &RecommendationRequest,
    ) -> RecommendResult<Vec<Recommendation>> {
        self.trending_detector.get_trending(request.limit)
    }

    /// Rank recommendations
    fn rank_recommendations(
        &self,
        candidates: Vec<Recommendation>,
    ) -> RecommendResult<Vec<Recommendation>> {
        rank::score::rank_recommendations(candidates)
    }

    /// Add explanations to recommendations
    fn add_explanations(&self, recommendations: &mut [Recommendation]) -> RecommendResult<()> {
        for rec in recommendations {
            let explanation = explain::generate::generate_explanation(rec)?;
            rec.explanation = Some(explanation);
        }
        Ok(())
    }

    /// Record a user view event
    ///
    /// # Errors
    ///
    /// Returns an error if recording fails
    pub fn record_view(
        &mut self,
        user_id: Uuid,
        content_id: Uuid,
        watch_time_ms: i64,
        completed: bool,
    ) -> RecommendResult<()> {
        // Record in history
        self.history_tracker
            .record_view(user_id, content_id, watch_time_ms, completed)?;

        // Update user profile
        self.profile_manager
            .update_from_view(user_id, content_id, watch_time_ms, completed)?;

        // Update implicit rating
        self.rating_manager.update_implicit_rating(
            user_id,
            content_id,
            watch_time_ms,
            completed,
        )?;

        Ok(())
    }

    /// Record an explicit rating
    ///
    /// # Errors
    ///
    /// Returns an error if recording fails
    pub fn record_rating(
        &mut self,
        user_id: Uuid,
        content_id: Uuid,
        rating: f32,
    ) -> RecommendResult<()> {
        self.rating_manager
            .record_rating(user_id, content_id, rating)?;
        self.profile_manager
            .update_from_rating(user_id, content_id, rating)?;
        Ok(())
    }

    /// Update trending scores
    ///
    /// # Errors
    ///
    /// Returns an error if update fails
    pub fn update_trending(&mut self) -> RecommendResult<()> {
        self.trending_detector.update_scores()
    }

    /// Get user profile
    ///
    /// # Errors
    ///
    /// Returns an error if retrieval fails
    pub fn get_user_profile(&self, user_id: Uuid) -> RecommendResult<profile::user::UserProfile> {
        self.profile_manager.get_profile(user_id)
    }

    /// Get similar users
    ///
    /// # Errors
    ///
    /// Returns an error if retrieval fails
    pub fn get_similar_users(&self, user_id: Uuid, limit: usize) -> RecommendResult<Vec<Uuid>> {
        self.profile_manager.get_similar_users(user_id, limit)
    }
}

impl Default for RecommendationEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for RecommendationRequest {
    fn default() -> Self {
        Self {
            user_id: Uuid::new_v4(),
            limit: 10,
            content_id: None,
            strategy: RecommendationStrategy::Hybrid,
            context: RecommendationContext::default(),
            diversity: DiversitySettings::default(),
            include_explanations: false,
        }
    }
}

impl Default for DiversitySettings {
    fn default() -> Self {
        Self {
            enabled: true,
            category_diversity: 0.3,
            include_serendipity: true,
            serendipity_weight: 0.1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recommendation_engine_creation() {
        let engine = RecommendationEngine::new();
        assert!(std::mem::size_of_val(&engine) > 0);
    }

    #[test]
    fn test_recommendation_request_default() {
        let request = RecommendationRequest::default();
        assert_eq!(request.limit, 10);
        assert!(matches!(request.strategy, RecommendationStrategy::Hybrid));
    }

    #[test]
    fn test_diversity_settings_default() {
        let settings = DiversitySettings::default();
        assert!(settings.enabled);
        assert!((settings.category_diversity - 0.3).abs() < f32::EPSILON);
    }

    #[test]
    fn test_recommendation_strategy_variants() {
        let strategies = [
            RecommendationStrategy::ContentBased,
            RecommendationStrategy::Collaborative,
            RecommendationStrategy::Hybrid,
            RecommendationStrategy::Personalized,
            RecommendationStrategy::Trending,
        ];
        assert_eq!(strategies.len(), 5);
    }

    #[test]
    fn test_recommend_hybrid_parallel_succeeds() {
        let mut engine = RecommendationEngine::new();
        let request = RecommendationRequest {
            strategy: RecommendationStrategy::Hybrid,
            limit: 10,
            ..Default::default()
        };
        // Hybrid should run in parallel without panicking and return Ok
        let result = engine.recommend(&request);
        assert!(result.is_ok());
        assert!(matches!(
            result.expect("hybrid recommend should succeed").strategy,
            RecommendationStrategy::Hybrid
        ));
    }

    #[test]
    fn test_recommend_all_strategies_run() {
        let mut engine = RecommendationEngine::new();
        for strategy in [
            RecommendationStrategy::ContentBased,
            RecommendationStrategy::Collaborative,
            RecommendationStrategy::Hybrid,
            RecommendationStrategy::Personalized,
            RecommendationStrategy::Trending,
        ] {
            let request = RecommendationRequest {
                strategy,
                limit: 5,
                ..Default::default()
            };
            let result = engine.recommend(&request);
            assert!(result.is_ok(), "strategy {strategy:?} failed");
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Rate limiter integration tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_engine_no_rate_limiter_by_default() {
        let engine = RecommendationEngine::new();
        assert!(!engine.has_rate_limiter());
        assert!(engine.global_available_tokens().is_none());
    }

    #[test]
    fn test_engine_with_rate_limiter_enabled() {
        let config = rate_limit::RateLimitConfig::default();
        let engine = RecommendationEngine::with_rate_limiter(config, 0);
        assert!(engine.has_rate_limiter());
        assert!(engine.global_available_tokens().is_some());
    }

    #[test]
    fn test_engine_rate_limiter_allows_under_limit() {
        let config = rate_limit::RateLimitConfig {
            per_user_capacity: 10.0,
            per_user_refill_rate: 1.0,
            global_capacity: 100.0,
            global_refill_rate: 10.0,
            tokens_per_request: 1.0,
        };
        let mut engine = RecommendationEngine::with_rate_limiter(config, 0);
        let request = RecommendationRequest::default();
        // Should succeed (not rate-limited)
        let result = engine.recommend(&request);
        assert!(result.is_ok(), "should be allowed: {result:?}");
    }

    #[test]
    fn test_engine_rate_limiter_rejects_when_exhausted() {
        let config = rate_limit::RateLimitConfig {
            per_user_capacity: 2.0,
            per_user_refill_rate: 0.0, // no refill
            global_capacity: 1000.0,
            global_refill_rate: 100.0,
            tokens_per_request: 1.0,
        };
        let mut engine = RecommendationEngine::with_rate_limiter(config, 0);
        let user_id = uuid::Uuid::new_v4();
        let request = RecommendationRequest {
            user_id,
            ..Default::default()
        };
        // First two should be allowed
        assert!(engine.recommend(&request).is_ok());
        assert!(engine.recommend(&request).is_ok());
        // Third should be rate-limited
        let result = engine.recommend(&request);
        assert!(result.is_err());
        let err_str = result.unwrap_err().to_string();
        assert!(err_str.contains("Rate limited") || err_str.contains("rate limit"));
    }

    #[test]
    fn test_engine_set_rate_limiter() {
        let mut engine = RecommendationEngine::new();
        assert!(!engine.has_rate_limiter());
        let config = rate_limit::RateLimitConfig::default();
        engine.set_rate_limiter(config, 0);
        assert!(engine.has_rate_limiter());
    }

    #[test]
    fn test_engine_disable_rate_limiter() {
        let config = rate_limit::RateLimitConfig::default();
        let mut engine = RecommendationEngine::with_rate_limiter(config, 0);
        assert!(engine.has_rate_limiter());
        engine.disable_rate_limiter();
        assert!(!engine.has_rate_limiter());
    }

    #[test]
    fn test_engine_user_available_tokens_after_request() {
        let config = rate_limit::RateLimitConfig {
            per_user_capacity: 10.0,
            per_user_refill_rate: 1.0,
            global_capacity: 1000.0,
            global_refill_rate: 100.0,
            tokens_per_request: 1.0,
        };
        let mut engine = RecommendationEngine::with_rate_limiter(config, 0);
        let user_id = uuid::Uuid::new_v4();
        let request = RecommendationRequest {
            user_id,
            ..Default::default()
        };
        engine.recommend(&request).ok();
        let tokens = engine.user_available_tokens(&user_id.to_string());
        assert!(tokens.is_some());
        // Should have consumed 1 token from a 10-token bucket
        let t = tokens.expect("should have bucket");
        assert!((t - 9.0).abs() < f64::EPSILON, "expected 9.0 but got {t}");
    }

    #[test]
    fn test_engine_global_tokens_decrease_per_request() {
        let config = rate_limit::RateLimitConfig {
            per_user_capacity: 100.0,
            per_user_refill_rate: 10.0,
            global_capacity: 50.0,
            global_refill_rate: 0.0,
            tokens_per_request: 1.0,
        };
        let mut engine = RecommendationEngine::with_rate_limiter(config, 0);
        let before = engine
            .global_available_tokens()
            .expect("should have global");
        engine.recommend(&RecommendationRequest::default()).ok();
        let after = engine
            .global_available_tokens()
            .expect("should have global");
        assert!((before - after - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_engine_multiple_users_independent_limits() {
        let config = rate_limit::RateLimitConfig {
            per_user_capacity: 1.0,
            per_user_refill_rate: 0.0,
            global_capacity: 1000.0,
            global_refill_rate: 100.0,
            tokens_per_request: 1.0,
        };
        let mut engine = RecommendationEngine::with_rate_limiter(config, 0);

        let user_a = uuid::Uuid::new_v4();
        let user_b = uuid::Uuid::new_v4();

        let req_a = RecommendationRequest {
            user_id: user_a,
            ..Default::default()
        };
        let req_b = RecommendationRequest {
            user_id: user_b,
            ..Default::default()
        };

        // Both get one request each
        assert!(engine.recommend(&req_a).is_ok(), "user_a first request");
        assert!(engine.recommend(&req_b).is_ok(), "user_b first request");

        // Both are now exhausted
        assert!(
            engine.recommend(&req_a).is_err(),
            "user_a should be rate-limited"
        );
        assert!(
            engine.recommend(&req_b).is_err(),
            "user_b should be rate-limited"
        );
    }
}
