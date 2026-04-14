//! Genre affinity modeling with temporal decay.
//!
//! [`GenreAffinityModel`] maintains a per-user genre preference vector built
//! from accumulated watch events.  Each interaction contributes a weighted
//! signal to one or more genre buckets; older interactions are down-weighted
//! via an exponential half-life so that the model adapts to shifting tastes.
//!
//! # Design
//!
//! - **Interaction record**: content ID + genre tags + watch fraction (0‒1) +
//!   Unix timestamp.
//! - **Affinity score** for a genre `g` at query time `t_now`:
//!   ```text
//!   affinity(g) = Σ  watch_fraction_i · e^(−λ · (t_now − t_i))
//!   ```
//!   where `λ = ln(2) / half_life_secs`.
//! - Scores are normalised to `[0, 1]` across all genres.
//! - [`AffinityScorer`] applies per-user affinity weights to a ranked list of
//!   candidate content items, boosting items whose genres align with the user's
//!   recent preferences.

#![allow(dead_code)]

use std::collections::HashMap;

use crate::error::{RecommendError, RecommendResult};

// ---------------------------------------------------------------------------
// Interaction record
// ---------------------------------------------------------------------------

/// A single watch interaction used to update genre affinity.
#[derive(Debug, Clone)]
pub struct WatchInteraction {
    /// User identifier.
    pub user_id: u64,
    /// Content identifier.
    pub content_id: u64,
    /// Genre tags of the watched item.
    pub genres: Vec<String>,
    /// Fraction of the content watched in `[0.0, 1.0]`.
    pub watch_fraction: f32,
    /// Unix timestamp (seconds) when the interaction started.
    pub timestamp_s: i64,
}

impl WatchInteraction {
    /// Constructs a new interaction record.
    ///
    /// `watch_fraction` is clamped to `[0.0, 1.0]`.
    #[must_use]
    pub fn new(
        user_id: u64,
        content_id: u64,
        genres: Vec<String>,
        watch_fraction: f32,
        timestamp_s: i64,
    ) -> Self {
        Self {
            user_id,
            content_id,
            genres,
            watch_fraction: watch_fraction.clamp(0.0, 1.0),
            timestamp_s,
        }
    }
}

// ---------------------------------------------------------------------------
// Per-genre signal accumulator
// ---------------------------------------------------------------------------

/// Accumulated raw signal for a single genre (prior to normalisation).
#[derive(Debug, Clone, Default)]
struct GenreSignal {
    /// Sum of decay-weighted watch fractions.
    raw_score: f64,
    /// Number of contributing interactions.
    interaction_count: u64,
}

// ---------------------------------------------------------------------------
// GenreAffinityModel
// ---------------------------------------------------------------------------

/// Stores and queries genre affinity for a collection of users.
///
/// # Thread Safety
///
/// This type is not `Send`/`Sync`; wrap in `Arc<Mutex<>>` for shared use.
pub struct GenreAffinityModel {
    /// Half-life in seconds for the temporal decay.
    half_life_secs: f64,
    /// Per-user, per-genre raw accumulated signals.
    /// `signals[user_id][genre] = GenreSignal`
    signals: HashMap<u64, HashMap<String, GenreSignal>>,
    /// Raw interaction log (needed for exact decay re-computation).
    interactions: Vec<WatchInteraction>,
}

impl GenreAffinityModel {
    /// Creates a new model with the specified half-life.
    ///
    /// # Errors
    ///
    /// Returns [`RecommendError::Other`] if `half_life_secs` is non-positive.
    pub fn new(half_life_secs: f64) -> RecommendResult<Self> {
        if half_life_secs <= 0.0 {
            return Err(RecommendError::Other(
                "half_life_secs must be positive".to_string(),
            ));
        }
        Ok(Self {
            half_life_secs,
            signals: HashMap::new(),
            interactions: Vec::new(),
        })
    }

    /// Creates a model with a 30-day (2 592 000 s) half-life.
    #[must_use]
    pub fn with_thirty_day_halflife() -> Self {
        Self {
            half_life_secs: 30.0 * 24.0 * 3600.0,
            signals: HashMap::new(),
            interactions: Vec::new(),
        }
    }

    /// Records a watch interaction and updates the in-memory signal accumulators.
    ///
    /// The decay is computed against `now_s` (current Unix timestamp).
    pub fn record_interaction(&mut self, interaction: WatchInteraction, now_s: i64) {
        let age_s = (now_s - interaction.timestamp_s).max(0) as f64;
        let lambda = std::f64::consts::LN_2 / self.half_life_secs;
        let weight = (-lambda * age_s).exp() * f64::from(interaction.watch_fraction);

        let user_signals = self.signals.entry(interaction.user_id).or_default();
        for genre in &interaction.genres {
            let sig = user_signals.entry(genre.clone()).or_default();
            sig.raw_score += weight;
            sig.interaction_count += 1;
        }
        self.interactions.push(interaction);
    }

    /// Returns the normalised genre affinity vector for `user_id` at time `now_s`.
    ///
    /// The returned map associates genre names with scores in `[0.0, 1.0]`.
    /// An empty map is returned for users with no interaction history.
    ///
    /// The scores are recomputed from the raw interaction log each time this is
    /// called to apply up-to-date decay; for production use a cached version
    /// of this result should be maintained.
    #[must_use]
    pub fn affinity_vector(&self, user_id: u64, now_s: i64) -> HashMap<String, f32> {
        let lambda = std::f64::consts::LN_2 / self.half_life_secs;
        let mut scores: HashMap<String, f64> = HashMap::new();

        for interaction in &self.interactions {
            if interaction.user_id != user_id {
                continue;
            }
            let age_s = (now_s - interaction.timestamp_s).max(0) as f64;
            let weight = (-lambda * age_s).exp() * f64::from(interaction.watch_fraction);
            for genre in &interaction.genres {
                *scores.entry(genre.clone()).or_default() += weight;
            }
        }

        if scores.is_empty() {
            return HashMap::new();
        }

        let max_score = scores.values().cloned().fold(f64::NEG_INFINITY, f64::max);

        if max_score <= 0.0 {
            return HashMap::new();
        }

        scores
            .into_iter()
            .map(|(g, s)| (g, (s / max_score) as f32))
            .collect()
    }

    /// Returns the top-`n` genres for `user_id` at `now_s`, sorted descending.
    #[must_use]
    pub fn top_genres(&self, user_id: u64, now_s: i64, n: usize) -> Vec<(String, f32)> {
        let mut vec: Vec<(String, f32)> =
            self.affinity_vector(user_id, now_s).into_iter().collect();
        vec.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        vec.truncate(n);
        vec
    }

    /// Returns the number of distinct users tracked.
    #[must_use]
    pub fn user_count(&self) -> usize {
        self.signals.len()
    }

    /// Returns the total number of interactions recorded.
    #[must_use]
    pub fn interaction_count(&self) -> usize {
        self.interactions.len()
    }

    /// Evicts interactions older than `max_age_s` seconds relative to `now_s`
    /// and rebuilds the in-memory signal accumulators.
    ///
    /// This is an O(N) operation intended for periodic maintenance.
    pub fn prune_old_interactions(&mut self, now_s: i64, max_age_s: i64) {
        let cutoff = now_s - max_age_s;
        self.interactions.retain(|i| i.timestamp_s >= cutoff);
        // Rebuild signals from scratch
        self.signals.clear();
        let interactions_snapshot = self.interactions.clone();
        for interaction in interactions_snapshot {
            self.record_interaction_no_log(interaction, now_s);
        }
    }

    /// Like [`record_interaction`] but does not push to `self.interactions`
    /// (used during `prune_old_interactions` rebuild).
    fn record_interaction_no_log(&mut self, interaction: WatchInteraction, now_s: i64) {
        let age_s = (now_s - interaction.timestamp_s).max(0) as f64;
        let lambda = std::f64::consts::LN_2 / self.half_life_secs;
        let weight = (-lambda * age_s).exp() * f64::from(interaction.watch_fraction);

        let user_signals = self.signals.entry(interaction.user_id).or_default();
        for genre in &interaction.genres {
            let sig = user_signals.entry(genre.clone()).or_default();
            sig.raw_score += weight;
            sig.interaction_count += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// AffinityScorer
// ---------------------------------------------------------------------------

/// Candidate content item to be scored by affinity.
#[derive(Debug, Clone)]
pub struct AffinityCandidate {
    /// Content identifier.
    pub content_id: u64,
    /// Genres associated with this content.
    pub genres: Vec<String>,
    /// Base recommendation score from an upstream model.
    pub base_score: f32,
}

impl AffinityCandidate {
    /// Creates a new candidate.
    #[must_use]
    pub fn new(content_id: u64, genres: Vec<String>, base_score: f32) -> Self {
        Self {
            content_id,
            genres,
            base_score: base_score.clamp(0.0, 1.0),
        }
    }
}

/// Scored candidate returned by [`AffinityScorer`].
#[derive(Debug, Clone)]
pub struct ScoredCandidate {
    /// Content identifier.
    pub content_id: u64,
    /// Combined score (base + affinity boost).
    pub score: f32,
    /// Affinity component of the score.
    pub affinity_score: f32,
}

/// Boosts candidate scores based on per-user genre affinity.
pub struct AffinityScorer {
    /// Weight applied to the affinity component when combining with base score.
    /// Final score = (1 − affinity_weight) · base + affinity_weight · affinity.
    pub affinity_weight: f32,
}

impl AffinityScorer {
    /// Creates a scorer with the given affinity weight.
    ///
    /// # Errors
    ///
    /// Returns an error if `affinity_weight` is outside `[0, 1]`.
    pub fn new(affinity_weight: f32) -> RecommendResult<Self> {
        if !(0.0..=1.0).contains(&affinity_weight) {
            return Err(RecommendError::Other(
                "affinity_weight must be in [0, 1]".to_string(),
            ));
        }
        Ok(Self { affinity_weight })
    }

    /// Scores `candidates` using the user's affinity vector, returning them
    /// sorted descending by combined score.
    ///
    /// # Arguments
    ///
    /// * `affinity_vec` – normalised affinity vector from
    ///   [`GenreAffinityModel::affinity_vector`].
    /// * `candidates` – items to score.
    #[must_use]
    pub fn score_candidates(
        &self,
        affinity_vec: &HashMap<String, f32>,
        candidates: &[AffinityCandidate],
    ) -> Vec<ScoredCandidate> {
        let mut results: Vec<ScoredCandidate> = candidates
            .iter()
            .map(|c| {
                let affinity_score = self.genre_affinity_score(affinity_vec, &c.genres);
                let score = (1.0 - self.affinity_weight) * c.base_score
                    + self.affinity_weight * affinity_score;
                ScoredCandidate {
                    content_id: c.content_id,
                    score,
                    affinity_score,
                }
            })
            .collect();

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results
    }

    /// Computes a single affinity score for a content item given its genres.
    ///
    /// Score is the max affinity across the item's genres (so an item matching
    /// at least one highly-affinised genre gets a strong boost).
    #[must_use]
    fn genre_affinity_score(&self, affinity_vec: &HashMap<String, f32>, genres: &[String]) -> f32 {
        genres
            .iter()
            .filter_map(|g| affinity_vec.get(g).copied())
            .fold(0.0_f32, f32::max)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const NOW: i64 = 1_700_000_000; // arbitrary fixed timestamp

    fn make_interaction(user: u64, genres: &[&str], frac: f32, age_s: i64) -> WatchInteraction {
        WatchInteraction::new(
            user,
            user * 100 + age_s as u64,
            genres.iter().map(|s| s.to_string()).collect(),
            frac,
            NOW - age_s,
        )
    }

    #[test]
    fn test_new_model_rejects_nonpositive_halflife() {
        assert!(GenreAffinityModel::new(0.0).is_err());
        assert!(GenreAffinityModel::new(-1.0).is_err());
    }

    #[test]
    fn test_empty_user_returns_empty_vector() {
        let model = GenreAffinityModel::with_thirty_day_halflife();
        let vec = model.affinity_vector(999, NOW);
        assert!(vec.is_empty());
    }

    #[test]
    fn test_single_interaction_top_genre() {
        let mut model = GenreAffinityModel::with_thirty_day_halflife();
        model.record_interaction(make_interaction(1, &["rock"], 1.0, 0), NOW);
        let top = model.top_genres(1, NOW, 3);
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].0, "rock");
        assert!((top[0].1 - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_recent_interactions_outweigh_old() {
        let mut model = GenreAffinityModel::new(7.0 * 24.0 * 3600.0).expect("valid halflife");
        // Two recent rock interactions vs one very old jazz interaction
        model.record_interaction(make_interaction(1, &["rock"], 1.0, 3600), NOW);
        model.record_interaction(make_interaction(1, &["rock"], 1.0, 7200), NOW);
        model.record_interaction(make_interaction(1, &["jazz"], 1.0, 90 * 24 * 3600), NOW);

        let vec = model.affinity_vector(1, NOW);
        let rock_score = *vec.get("rock").unwrap_or(&0.0);
        let jazz_score = *vec.get("jazz").unwrap_or(&0.0);
        assert!(
            rock_score > jazz_score,
            "recent rock should outweigh old jazz"
        );
    }

    #[test]
    fn test_affinity_vector_max_is_one() {
        let mut model = GenreAffinityModel::with_thirty_day_halflife();
        model.record_interaction(make_interaction(1, &["rock"], 0.5, 0), NOW);
        model.record_interaction(make_interaction(1, &["pop"], 0.3, 0), NOW);
        let vec = model.affinity_vector(1, NOW);
        let max = vec.values().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            (max - 1.0).abs() < 1e-4,
            "max affinity should be normalised to 1.0"
        );
    }

    #[test]
    fn test_affinity_scorer_boosts_matching_genre() {
        let mut model = GenreAffinityModel::with_thirty_day_halflife();
        model.record_interaction(make_interaction(1, &["rock"], 1.0, 0), NOW);
        let aff_vec = model.affinity_vector(1, NOW);

        let scorer = AffinityScorer::new(0.5).expect("valid weight");
        let candidates = vec![
            AffinityCandidate::new(10, vec!["rock".to_string()], 0.6),
            AffinityCandidate::new(11, vec!["jazz".to_string()], 0.6),
        ];
        let scored = scorer.score_candidates(&aff_vec, &candidates);
        assert_eq!(scored.len(), 2);
        // rock candidate should rank higher
        assert_eq!(scored[0].content_id, 10, "rock candidate should rank first");
    }

    #[test]
    fn test_affinity_scorer_invalid_weight() {
        assert!(AffinityScorer::new(-0.1).is_err());
        assert!(AffinityScorer::new(1.1).is_err());
    }

    #[test]
    fn test_prune_old_interactions() {
        let mut model = GenreAffinityModel::with_thirty_day_halflife();
        model.record_interaction(make_interaction(1, &["rock"], 1.0, 0), NOW);
        model.record_interaction(make_interaction(1, &["jazz"], 1.0, 200 * 24 * 3600), NOW);
        assert_eq!(model.interaction_count(), 2);

        // Prune interactions older than 60 days
        model.prune_old_interactions(NOW, 60 * 24 * 3600);
        assert_eq!(model.interaction_count(), 1);
        let vec = model.affinity_vector(1, NOW);
        assert!(vec.contains_key("rock"));
        assert!(
            !vec.contains_key("jazz"),
            "old jazz interaction should be pruned"
        );
    }

    #[test]
    fn test_top_genres_count_limit() {
        let mut model = GenreAffinityModel::with_thirty_day_halflife();
        for (i, g) in ["rock", "pop", "jazz", "classical", "electronic"]
            .iter()
            .enumerate()
        {
            model.record_interaction(make_interaction(1, &[g], 1.0, i as i64 * 1000), NOW);
        }
        let top = model.top_genres(1, NOW, 3);
        assert_eq!(top.len(), 3);
    }

    #[test]
    fn test_user_count_and_interaction_count() {
        let mut model = GenreAffinityModel::with_thirty_day_halflife();
        model.record_interaction(make_interaction(1, &["rock"], 1.0, 0), NOW);
        model.record_interaction(make_interaction(2, &["pop"], 0.5, 0), NOW);
        model.record_interaction(make_interaction(1, &["jazz"], 0.8, 3600), NOW);
        assert_eq!(model.user_count(), 2);
        assert_eq!(model.interaction_count(), 3);
    }
}
