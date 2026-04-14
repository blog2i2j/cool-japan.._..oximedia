//! Cross-domain recommendation engine.
//!
//! Enables recommending content from one media domain (e.g., audio podcasts) to
//! users who primarily interact with a different domain (e.g., video).  The
//! engine aligns shared interest signals — genre tags, topic embeddings, creator
//! overlap — across domain boundaries and uses a transfer-learning-style scoring
//! to surface relevant cross-domain items.
//!
//! # Overview
//!
//! ```text
//! VideoHistory  ──┐
//!                 ├─► SharedInterestModel ─► CrossDomainScorer ─► ranked list
//! AudioHistory ───┘
//! ```
//!
//! ## Algorithm
//!
//! 1. Each domain item carries a `Vec<String>` of normalised topic tags.
//! 2. A [`SharedInterestModel`] aggregates a user's topic weights from their
//!    **source** domain by counting tag occurrences weighted by implicit rating.
//! 3. The [`CrossDomainScorer`] scores **target** domain candidates by computing
//!    the cosine similarity between the user's topic vector and each candidate's
//!    tag TF-IDF vector, multiplied by a cross-domain alignment weight.
//! 4. An optional creator-overlap boost rewards candidates from creators whose
//!    work the user already appreciates in the source domain.

#![allow(dead_code)]

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Domain type
// ─────────────────────────────────────────────────────────────────────────────

/// Media domain identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Domain {
    /// Long-form or short-form video content.
    Video,
    /// Audio: music, podcasts, audiobooks.
    Audio,
    /// News, articles, blog posts.
    Text,
    /// Games and interactive content.
    Interactive,
    /// Any custom domain label.
    Custom(String),
}

impl std::fmt::Display for Domain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Video => write!(f, "video"),
            Self::Audio => write!(f, "audio"),
            Self::Text => write!(f, "text"),
            Self::Interactive => write!(f, "interactive"),
            Self::Custom(s) => write!(f, "{s}"),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Domain item
// ─────────────────────────────────────────────────────────────────────────────

/// A content item belonging to a specific domain.
#[derive(Debug, Clone)]
pub struct DomainItem {
    /// Unique item identifier (within its domain).
    pub item_id: String,
    /// The domain this item belongs to.
    pub domain: Domain,
    /// Normalised topic/genre tags (lower-case, no spaces).
    pub tags: Vec<String>,
    /// Optional creator/channel identifier.
    pub creator_id: Option<String>,
    /// Popularity score in [0, 1] (used as a tie-breaker).
    pub popularity: f64,
}

impl DomainItem {
    /// Create a new domain item.
    #[must_use]
    pub fn new(item_id: impl Into<String>, domain: Domain, tags: Vec<String>) -> Self {
        Self {
            item_id: item_id.into(),
            domain,
            tags,
            creator_id: None,
            popularity: 0.0,
        }
    }

    /// Set the creator identifier.
    #[must_use]
    pub fn with_creator(mut self, creator_id: impl Into<String>) -> Self {
        self.creator_id = Some(creator_id.into());
        self
    }

    /// Set the popularity score.
    #[must_use]
    pub fn with_popularity(mut self, popularity: f64) -> Self {
        self.popularity = popularity.clamp(0.0, 1.0);
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// User interaction record
// ─────────────────────────────────────────────────────────────────────────────

/// A single interaction event from a user in a specific domain.
#[derive(Debug, Clone)]
pub struct DomainInteraction {
    /// Identifier of the item interacted with.
    pub item_id: String,
    /// Source domain of the interaction.
    pub domain: Domain,
    /// Implicit rating in [0, 1] (e.g., completion ratio × engagement).
    pub rating: f64,
}

impl DomainInteraction {
    /// Create a domain interaction event.
    #[must_use]
    pub fn new(item_id: impl Into<String>, domain: Domain, rating: f64) -> Self {
        Self {
            item_id: item_id.into(),
            domain,
            rating: rating.clamp(0.0, 1.0),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared interest model
// ─────────────────────────────────────────────────────────────────────────────

/// Aggregates topic-tag interest weights from a user's source-domain history.
///
/// The model builds a `tag → weight` map by accumulating `rating * tag_count`
/// for each item the user interacted with.  The weight vector is L1-normalised
/// before querying so that longer histories do not dominate shorter ones.
#[derive(Debug, Default)]
pub struct SharedInterestModel {
    /// Accumulated tag weights from source-domain interactions.
    tag_weights: HashMap<String, f64>,
    /// Total interaction mass (for normalisation tracking).
    total_mass: f64,
    /// Set of creator IDs from source-domain interactions.
    known_creators: std::collections::HashSet<String>,
    /// Number of interactions ingested.
    interaction_count: u64,
}

impl SharedInterestModel {
    /// Create an empty model.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Ingest a source-domain interaction.
    ///
    /// `item` must belong to the source domain; its tags contribute
    /// proportionally to `interaction.rating`.
    pub fn ingest(&mut self, interaction: &DomainInteraction, item: &DomainItem) {
        if interaction.rating <= 0.0 || item.tags.is_empty() {
            return;
        }
        let weight_per_tag = interaction.rating / item.tags.len() as f64;
        for tag in &item.tags {
            *self.tag_weights.entry(tag.clone()).or_insert(0.0) += weight_per_tag;
            self.total_mass += weight_per_tag;
        }
        if let Some(creator) = &item.creator_id {
            self.known_creators.insert(creator.clone());
        }
        self.interaction_count += 1;
    }

    /// Get the normalised weight for a single tag (returns 0.0 if unknown).
    #[must_use]
    pub fn tag_weight(&self, tag: &str) -> f64 {
        if self.total_mass <= 0.0 {
            return 0.0;
        }
        self.tag_weights.get(tag).copied().unwrap_or(0.0) / self.total_mass
    }

    /// Whether the user has interacted with a creator in the source domain.
    #[must_use]
    pub fn knows_creator(&self, creator_id: &str) -> bool {
        self.known_creators.contains(creator_id)
    }

    /// Return the number of distinct tags tracked.
    #[must_use]
    pub fn tag_count(&self) -> usize {
        self.tag_weights.len()
    }

    /// Return the number of interactions ingested.
    #[must_use]
    pub fn interaction_count(&self) -> u64 {
        self.interaction_count
    }

    /// Compute the cosine similarity between this model's tag vector and a
    /// candidate item's tag set.
    ///
    /// Each candidate tag contributes weight 1/|tags|.  The similarity is the
    /// dot product of the normalised model vector and the normalised candidate
    /// vector.
    #[must_use]
    pub fn cosine_similarity_to_item(&self, item: &DomainItem) -> f64 {
        if self.total_mass <= 0.0 || item.tags.is_empty() {
            return 0.0;
        }

        let item_weight = 1.0 / item.tags.len() as f64;
        let mut dot = 0.0;
        let mut model_norm_sq = 0.0;

        for (tag, &w) in &self.tag_weights {
            let normalised_w = w / self.total_mass;
            model_norm_sq += normalised_w * normalised_w;
            if item.tags.contains(tag) {
                dot += normalised_w * item_weight;
            }
        }
        let item_norm_sq = (item_weight * item_weight) * item.tags.len() as f64;

        let denom = (model_norm_sq.sqrt()) * (item_norm_sq.sqrt());
        if denom < 1e-15 {
            return 0.0;
        }
        (dot / denom).clamp(0.0, 1.0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Cross-domain candidate
// ─────────────────────────────────────────────────────────────────────────────

/// A scored candidate from a target domain.
#[derive(Debug, Clone)]
pub struct CrossDomainCandidate {
    /// Item identifier.
    pub item_id: String,
    /// Target domain of the candidate.
    pub domain: Domain,
    /// Combined cross-domain relevance score (0–1).
    pub score: f64,
    /// Tag-based topical similarity component.
    pub topic_score: f64,
    /// Creator-overlap boost component.
    pub creator_boost: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Cross-domain scorer config
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the cross-domain scorer.
#[derive(Debug, Clone)]
pub struct CrossDomainConfig {
    /// Alignment weight for topic cosine similarity (0–1).
    pub topic_weight: f64,
    /// Bonus added when the candidate's creator is known from source domain.
    pub creator_boost: f64,
    /// Popularity blend weight (0 = ignore popularity).
    pub popularity_weight: f64,
    /// Minimum topic similarity threshold to include a candidate.
    pub min_topic_similarity: f64,
}

impl Default for CrossDomainConfig {
    fn default() -> Self {
        Self {
            topic_weight: 0.7,
            creator_boost: 0.2,
            popularity_weight: 0.1,
            min_topic_similarity: 0.05,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Cross-domain scorer
// ─────────────────────────────────────────────────────────────────────────────

/// Scores target-domain items based on a user's source-domain interest model.
#[derive(Debug)]
pub struct CrossDomainScorer {
    /// Scoring configuration.
    config: CrossDomainConfig,
}

impl CrossDomainScorer {
    /// Create a scorer with the given configuration.
    #[must_use]
    pub fn new(config: CrossDomainConfig) -> Self {
        Self { config }
    }

    /// Score a batch of target-domain candidates against a user's shared interest model.
    ///
    /// Returns a sorted (descending) list of [`CrossDomainCandidate`]s that exceed
    /// the minimum topic similarity threshold.
    #[must_use]
    pub fn score(
        &self,
        candidates: &[DomainItem],
        model: &SharedInterestModel,
    ) -> Vec<CrossDomainCandidate> {
        let mut results: Vec<CrossDomainCandidate> = candidates
            .iter()
            .filter_map(|item| {
                let topic_score = model.cosine_similarity_to_item(item);
                if topic_score < self.config.min_topic_similarity {
                    return None;
                }

                let creator_boost = item
                    .creator_id
                    .as_deref()
                    .map(|cid| {
                        if model.knows_creator(cid) {
                            self.config.creator_boost
                        } else {
                            0.0
                        }
                    })
                    .unwrap_or(0.0);

                let score = (self.config.topic_weight * topic_score
                    + creator_boost
                    + self.config.popularity_weight * item.popularity)
                    .clamp(0.0, 1.0);

                Some(CrossDomainCandidate {
                    item_id: item.item_id.clone(),
                    domain: item.domain.clone(),
                    score,
                    topic_score,
                    creator_boost,
                })
            })
            .collect();

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results
    }

    /// Return the configuration.
    #[must_use]
    pub fn config(&self) -> &CrossDomainConfig {
        &self.config
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// High-level façade
// ─────────────────────────────────────────────────────────────────────────────

/// End-to-end cross-domain recommendation pipeline.
///
/// Holds an item catalogue per domain, a per-user shared interest model, and
/// a scorer.  Call [`CrossDomainEngine::ingest_interaction`] as interactions
/// arrive and [`CrossDomainEngine::recommend`] to retrieve ranked cross-domain
/// recommendations.
#[derive(Debug)]
pub struct CrossDomainEngine {
    /// Items indexed by (domain, item_id).
    catalogue: HashMap<(String, String), DomainItem>,
    /// Per-user interest models.
    user_models: HashMap<String, SharedInterestModel>,
    /// Cross-domain scorer.
    scorer: CrossDomainScorer,
}

impl CrossDomainEngine {
    /// Create a new engine with the given scorer configuration.
    #[must_use]
    pub fn new(config: CrossDomainConfig) -> Self {
        Self {
            catalogue: HashMap::new(),
            user_models: HashMap::new(),
            scorer: CrossDomainScorer::new(config),
        }
    }

    /// Register an item in the catalogue.
    pub fn add_item(&mut self, item: DomainItem) {
        self.catalogue
            .insert((item.domain.to_string(), item.item_id.clone()), item);
    }

    /// Ingest a user interaction from the source domain.
    pub fn ingest_interaction(&mut self, user_id: &str, interaction: &DomainInteraction) {
        // Find the item in the catalogue
        let key = (interaction.domain.to_string(), interaction.item_id.clone());
        let Some(item) = self.catalogue.get(&key) else {
            return;
        };
        let item = item.clone(); // avoid borrow conflict
        self.user_models
            .entry(user_id.to_string())
            .or_default()
            .ingest(interaction, &item);
    }

    /// Recommend target-domain items for a user.
    ///
    /// Returns an empty list if the user has no source-domain interactions or
    /// if no candidates meet the minimum topic similarity threshold.
    #[must_use]
    pub fn recommend(
        &self,
        user_id: &str,
        target_domain: &Domain,
        limit: usize,
    ) -> Vec<CrossDomainCandidate> {
        let Some(model) = self.user_models.get(user_id) else {
            return Vec::new();
        };

        let target_domain_str = target_domain.to_string();
        let candidates: Vec<DomainItem> = self
            .catalogue
            .iter()
            .filter(|((dom, _), _)| dom == &target_domain_str)
            .map(|(_, item)| item.clone())
            .collect();

        let mut results = self.scorer.score(&candidates, model);
        results.truncate(limit);
        results
    }

    /// Return the number of items in the catalogue.
    #[must_use]
    pub fn catalogue_size(&self) -> usize {
        self.catalogue.len()
    }

    /// Return the number of users with interest models.
    #[must_use]
    pub fn user_count(&self) -> usize {
        self.user_models.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn video_item(id: &str, tags: &[&str]) -> DomainItem {
        DomainItem::new(
            id,
            Domain::Video,
            tags.iter().map(|s| s.to_string()).collect(),
        )
    }

    fn audio_item(id: &str, tags: &[&str]) -> DomainItem {
        DomainItem::new(
            id,
            Domain::Audio,
            tags.iter().map(|s| s.to_string()).collect(),
        )
    }

    fn video_interaction(id: &str, rating: f64) -> DomainInteraction {
        DomainInteraction::new(id, Domain::Video, rating)
    }

    // ─── Domain ─────────────────────────────────────────────────────────────

    #[test]
    fn test_domain_display() {
        assert_eq!(Domain::Video.to_string(), "video");
        assert_eq!(Domain::Audio.to_string(), "audio");
        assert_eq!(Domain::Custom("live".into()).to_string(), "live");
    }

    // ─── SharedInterestModel ────────────────────────────────────────────────

    #[test]
    fn test_model_ingest_single_item() {
        let mut model = SharedInterestModel::new();
        let item = video_item("v1", &["comedy", "animation"]);
        let ix = video_interaction("v1", 1.0);
        model.ingest(&ix, &item);
        assert_eq!(model.interaction_count(), 1);
        assert_eq!(model.tag_count(), 2);
        // Each tag should have equal weight
        let w_comedy = model.tag_weight("comedy");
        let w_animation = model.tag_weight("animation");
        assert!((w_comedy - w_animation).abs() < 1e-10);
        assert!(w_comedy > 0.0);
    }

    #[test]
    fn test_model_ingest_zero_rating_ignored() {
        let mut model = SharedInterestModel::new();
        let item = video_item("v1", &["drama"]);
        let ix = video_interaction("v1", 0.0);
        model.ingest(&ix, &item);
        assert_eq!(model.interaction_count(), 0);
        assert_eq!(model.tag_count(), 0);
    }

    #[test]
    fn test_model_knows_creator() {
        let mut model = SharedInterestModel::new();
        let item = video_item("v1", &["tech"]).with_creator("creator_abc");
        let ix = video_interaction("v1", 0.8);
        model.ingest(&ix, &item);
        assert!(model.knows_creator("creator_abc"));
        assert!(!model.knows_creator("creator_xyz"));
    }

    #[test]
    fn test_model_cosine_similarity_matching_tags() {
        let mut model = SharedInterestModel::new();
        let v1 = video_item("v1", &["tech", "science"]);
        let v2 = video_item("v2", &["cooking"]);
        model.ingest(&video_interaction("v1", 1.0), &v1);
        model.ingest(&video_interaction("v2", 0.2), &v2);

        // Audio item with matching tags should score higher
        let matching = audio_item("a1", &["tech", "science"]);
        let non_matching = audio_item("a2", &["cooking", "food"]);

        let sim_match = model.cosine_similarity_to_item(&matching);
        let sim_non = model.cosine_similarity_to_item(&non_matching);
        assert!(sim_match >= 0.0 && sim_match <= 1.0, "sim out of range");
        assert!(
            sim_match > sim_non,
            "matching should score higher than non-matching"
        );
    }

    #[test]
    fn test_model_cosine_similarity_empty_model() {
        let model = SharedInterestModel::new();
        let item = audio_item("a1", &["jazz"]);
        assert_eq!(model.cosine_similarity_to_item(&item), 0.0);
    }

    // ─── CrossDomainScorer ──────────────────────────────────────────────────

    #[test]
    fn test_scorer_filters_below_threshold() {
        let mut model = SharedInterestModel::new();
        let v1 = video_item("v1", &["tech"]);
        model.ingest(&video_interaction("v1", 1.0), &v1);

        let config = CrossDomainConfig {
            min_topic_similarity: 0.99, // very high threshold
            ..Default::default()
        };
        let scorer = CrossDomainScorer::new(config);
        // Audio item with no matching tags should score < threshold
        let candidates = vec![audio_item("a1", &["cooking"])];
        let results = scorer.score(&candidates, &model);
        assert!(results.is_empty(), "should be filtered out");
    }

    #[test]
    fn test_scorer_creator_boost_applied() {
        let mut model = SharedInterestModel::new();
        let v1 = video_item("v1", &["tech"]).with_creator("creator1");
        model.ingest(&video_interaction("v1", 1.0), &v1);

        let config = CrossDomainConfig {
            creator_boost: 0.3,
            min_topic_similarity: 0.0,
            ..Default::default()
        };
        let scorer = CrossDomainScorer::new(config);

        let with_creator = audio_item("a1", &["tech"]).with_creator("creator1");
        let without_creator = audio_item("a2", &["tech"]);

        let r_with = scorer.score(&[with_creator], &model);
        let r_without = scorer.score(&[without_creator], &model);

        assert!(!r_with.is_empty());
        assert!(!r_without.is_empty());
        assert!(
            r_with[0].score >= r_without[0].score,
            "creator boost should raise score"
        );
        assert!(r_with[0].creator_boost > 0.0);
    }

    #[test]
    fn test_scorer_sorted_by_score() {
        let mut model = SharedInterestModel::new();
        let v1 = video_item("v1", &["tech", "ai", "robotics"]);
        model.ingest(&video_interaction("v1", 1.0), &v1);

        let scorer = CrossDomainScorer::new(CrossDomainConfig {
            min_topic_similarity: 0.0,
            ..Default::default()
        });
        let candidates = vec![
            audio_item("a_weak", &["cooking"]),
            audio_item("a_strong", &["tech", "ai"]),
        ];
        let results = scorer.score(&candidates, &model);
        assert!(results.len() >= 1);
        // Should be sorted descending
        for window in results.windows(2) {
            assert!(window[0].score >= window[1].score);
        }
    }

    // ─── CrossDomainEngine ──────────────────────────────────────────────────

    #[test]
    fn test_engine_add_item_and_size() {
        let mut engine = CrossDomainEngine::new(CrossDomainConfig::default());
        engine.add_item(video_item("v1", &["sports"]));
        engine.add_item(audio_item("a1", &["sports_talk"]));
        assert_eq!(engine.catalogue_size(), 2);
    }

    #[test]
    fn test_engine_recommend_unknown_user_empty() {
        let engine = CrossDomainEngine::new(CrossDomainConfig::default());
        let results = engine.recommend("nobody", &Domain::Audio, 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_engine_ingest_and_recommend() {
        let mut engine = CrossDomainEngine::new(CrossDomainConfig {
            min_topic_similarity: 0.0,
            ..Default::default()
        });

        engine.add_item(video_item("v1", &["comedy", "animation"]));
        engine.add_item(video_item("v2", &["drama"]));
        engine.add_item(audio_item("a1", &["comedy", "funny"]));
        engine.add_item(audio_item("a2", &["drama", "classic"]));

        engine.ingest_interaction("user1", &video_interaction("v1", 1.0));
        engine.ingest_interaction("user1", &video_interaction("v2", 0.3));

        let recs = engine.recommend("user1", &Domain::Audio, 5);
        // Both audio items should potentially be recommended
        assert!(
            !recs.is_empty(),
            "should return at least one recommendation"
        );
        // comedy audio should rank higher than drama audio (user liked comedy more)
        if recs.len() >= 2 {
            assert!(recs[0].score >= recs[1].score);
        }
    }

    #[test]
    fn test_engine_user_count() {
        let mut engine = CrossDomainEngine::new(CrossDomainConfig::default());
        engine.add_item(video_item("v1", &["tech"]));
        engine.ingest_interaction("u1", &video_interaction("v1", 0.8));
        engine.ingest_interaction("u2", &video_interaction("v1", 0.5));
        assert_eq!(engine.user_count(), 2);
    }

    #[test]
    fn test_engine_recommend_limit_respected() {
        let mut engine = CrossDomainEngine::new(CrossDomainConfig {
            min_topic_similarity: 0.0,
            ..Default::default()
        });
        engine.add_item(video_item("v1", &["tech"]));
        for i in 0..10 {
            engine.add_item(audio_item(&format!("a{i}"), &["tech"]));
        }
        engine.ingest_interaction("user1", &video_interaction("v1", 1.0));
        let recs = engine.recommend("user1", &Domain::Audio, 3);
        assert!(recs.len() <= 3);
    }

    #[test]
    fn test_domain_item_with_popularity() {
        let item = audio_item("a1", &["jazz"]).with_popularity(1.5); // clamped to 1.0
        assert!((item.popularity - 1.0).abs() < f64::EPSILON);
    }
}
