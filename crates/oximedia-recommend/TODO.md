# oximedia-recommend TODO

## Current Status
- 37 modules for content recommendation and discovery engine
- Key types: RecommendationEngine, RecommendationRequest, Recommendation, RecommendationResults
- Strategies: ContentBased, Collaborative, Hybrid, Personalized, Trending
- Core modules: collaborative (matrix factorization), content (similarity), hybrid (combine), profile (user), history (track), rating (explicit/implicit), trending (detect), personalize (engine), diversity (ensure), freshness (balance), rank (score), explain (generate)
- Advanced modules: ab_test, als (Alternating Least Squares), bandits, calibration, cold_start, collab_filter, content_based, context_signal, decay_model, dense_linalg, exploration_policy, feature_store, feedback_signal, impression_tracker, item_similarity, popularity_bias, recommendation_score, score_cache, sequence_model, session, svd_pp (SVD++), user_profile
- Dependencies: oximedia-core, oximedia-search, rayon, chrono, uuid, serde

## Enhancements
- [ ] Add real-time model updating in `collaborative::matrix::CollaborativeEngine` (incremental matrix factorization)
- [ ] Implement user segment-based recommendations in `personalize` (cluster users into segments)
- [ ] Extend `diversity::ensure::DiversityEnforcer` with maximal marginal relevance (MMR) reranking
- [x] Add `impression_tracker` deduplication — never recommend already-seen content
- [ ] Implement `cold_start` with popularity-based fallback and demographic-based initialization
- [ ] Extend `explain::generate` with visual explanation data (feature importance scores)
- [x] Add `ab_test` statistical significance testing (chi-squared, t-test for engagement metrics)
- [x] Implement rate limiting in `RecommendationEngine` to prevent abuse of recommendation API

## New Features
- [ ] Add knowledge graph-based recommendations — leverage media metadata relationships (director, genre, cast)
- [ ] Implement session-based recommendations in `session` module (short-term user intent modeling)
- [ ] Add multi-objective optimization — balance engagement, diversity, and freshness simultaneously
- [ ] Implement federated learning support — train collaborative models without centralizing user data
- [ ] Add content embargo/scheduling — time-gate recommendations for release-date-aware content
- [ ] Implement cross-domain recommendations (recommend audio content to video watchers based on shared interests)
- [ ] Add recommendation fairness metrics — measure and enforce exposure equity across content creators
- [ ] Implement contextual bandits in `bandits` for exploration/exploitation in live recommendations

## Performance
- [ ] Implement approximate nearest neighbor search in `item_similarity` using locality-sensitive hashing
- [ ] Add `score_cache` with LRU eviction and TTL-based invalidation for hot recommendation paths
- [ ] Optimize `dense_linalg` matrix operations using blocked algorithms for cache efficiency
- [ ] Pre-compute user embeddings in `collaborative` and cache for sub-millisecond recommendation serving
- [ ] Parallelize `RecommendationEngine::recommend()` strategy evaluation using rayon
- [ ] Implement batch recommendation generation for offline/pre-computation scenarios

## Testing
- [ ] Add offline evaluation tests using precision@k, recall@k, and NDCG metrics
- [ ] Test `cold_start` behavior for brand-new users with zero interaction history
- [ ] Add diversity measurement tests — verify `DiversityEnforcer` actually increases category spread
- [ ] Test `trending::detect` with synthetic view spikes and verify detection latency
- [ ] Add regression tests for `svd_pp` and `als` with small known-answer datasets

## Documentation
- [ ] Add recommendation algorithm selection guide (when to use content-based vs collaborative vs hybrid)
- [ ] Document A/B testing workflow with metric collection and analysis
- [ ] Add integration guide for connecting RecommendationEngine to a media platform backend
